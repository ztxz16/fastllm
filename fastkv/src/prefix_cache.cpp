#include "internal.h"
#include <algorithm>
#include <cstring>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace fastkv {
namespace {
constexpr size_t PageBytes = 64 << 10;
constexpr uint64_t HashSeed = 1469598103934665603ULL;
uint64_t AppendHash(uint64_t hash, uint32_t value) { return (hash ^ value) * 1099511628211ULL; }
uint64_t NamespaceHash(const std::string &name) {
    uint64_t hash = HashSeed;
    for (unsigned char c : name)
        hash = AppendHash(hash, c);
    return hash;
}
uint64_t PageHash(const Buffer &host) {
    const auto *p = static_cast<const uint8_t *>(host.Data());
    uint64_t hash = HashSeed ^ host.Size();
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= host.Size(); i += sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, p + i, sizeof(word));
        hash = (hash ^ word) * 1099511628211ULL;
    }
    for (; i < host.Size(); ++i)
        hash = AppendHash(hash, p[i]);
    return hash;
}
} // namespace

struct PrefixCache::Impl {
    struct Page {
        std::shared_ptr<void> credit;
        Buffer data;
    };
    struct Payload {
        size_t bytes = 0;
        std::vector<std::shared_ptr<Page>> pages;
    };
    struct State {
        std::shared_ptr<void> credit;
        std::string nameSpace;
        std::vector<int> tokens;
        Bytes metadata;
        std::vector<Payload> payloads;
        // Wire consumers still see the original contiguous tensor boundaries.
        // Only active readers materialize a message; cached versions share pages.
        std::weak_ptr<const Message> materialized;
        size_t logicalBytes = 0;
    };
    struct Entry {
        std::shared_ptr<State> state;
        uint64_t hash = 0, accessed = 0, parent = 0;
        std::set<uint64_t> children;
    };
    mutable std::mutex mutex;
    Budget budget;
    size_t capacity, maxEntries;
    int device;
    uint64_t clock = 0, publications = 0;
    Statistics stats;
    std::map<uint64_t, Entry> entries;
    std::set<uint64_t> roots;
    std::map<size_t, std::unordered_multimap<uint64_t, uint64_t>> lengths;
    std::unordered_map<uint64_t, std::vector<std::weak_ptr<Page>>> pages;

    Impl(size_t bytes, size_t count, int device)
        : budget(bytes), capacity(bytes), maxEntries(count), device(device) {
        Require(device >= -1, "FastKV: invalid prefix storage device");
    }
    uint64_t Match(const std::string &name, const std::vector<int> &tokens, size_t end) const {
        uint64_t hash = NamespaceHash(name);
        std::vector<uint64_t> candidates;
        size_t position = 0;
        for (const auto &length : lengths) {
            if (length.first > end)
                break;
            for (; position < length.first; ++position)
                hash = AppendHash(hash, uint32_t(tokens[position]));
            auto range = length.second.equal_range(hash);
            for (auto it = range.first; it != range.second; ++it)
                candidates.push_back(it->second);
        }
        // Verify longest first, avoiding full-token scans of every ancestor.
        for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
            const auto &state = *entries.at(*it).state;
            if (state.nameSpace == name &&
                std::equal(state.tokens.begin(), state.tokens.end(), tokens.begin()))
                return *it;
        }
        return 0;
    }
    std::set<uint64_t> &Children(uint64_t parent) { return parent ? entries.at(parent).children : roots; }
    void Erase(uint64_t id) {
        auto &entry = entries.at(id);
        auto &siblings = Children(entry.parent);
        for (auto child : entry.children)
            entries.at(child).parent = entry.parent;
        siblings.merge(entry.children);
        siblings.erase(id);
        auto bucket = lengths.find(entry.state->tokens.size());
        auto range = bucket->second.equal_range(entry.hash);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second == id) {
                bucket->second.erase(it);
                break;
            }
        if (bucket->second.empty())
            lengths.erase(bucket);
        stats.logicalBytes -= entry.state->logicalBytes;
        entries.erase(id);
    }
    bool Evict() {
        auto victim = entries.end();
        int best = 3;
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            // An active lookup pins the checkpoint and its page credits.
            if (it->second.state.use_count() > 1)
                continue;
            const auto children = it->second.children.size();
            const int priority = children == 1 ? 0 : children > 1 ? 1 : 2;
            if (victim == entries.end() || priority < best ||
                (priority == best && it->second.accessed < victim->second.accessed)) {
                victim = it;
                best = priority;
            }
        }
        if (victim == entries.end())
            return false;
        Erase(victim->first);
        return true;
    }
    void PrunePages() {
        for (auto it = pages.begin(); it != pages.end();) {
            auto &bucket = it->second;
            bucket.erase(
                std::remove_if(bucket.begin(), bucket.end(), [](const auto &page) { return page.expired(); }),
                bucket.end());
            if (bucket.empty())
                it = pages.erase(it);
            else
                ++it;
        }
    }
    bool SamePage(const Page &page, const Buffer &host) const {
        if (page.data.Size() != host.Size())
            return false;
        if (page.data.Kind() == MemoryKind::Host)
            return !std::memcmp(page.data.Data(), host.Data(), host.Size());
        Bytes staging(host.Size());
        page.data.Read(staging.data(), 0, staging.size());
        return !std::memcmp(staging.data(), host.Data(), host.Size());
    }
    std::shared_ptr<const Message> Restore(const std::shared_ptr<State> &state) {
        if (auto cached = state->materialized.lock())
            return cached;
        Message value;
        value.metadata = state->metadata;
        // Match Snapshot's bounded arenas without an intermediate full copy.
        for (size_t first = 0; first < state->payloads.size();) {
            size_t last = first, bytes = 0;
            while (last < state->payloads.size()) {
                size_t n = state->payloads[last].bytes;
                if (last > first && (bytes >= (64 << 20) || n > (64 << 20) - bytes))
                    break;
                bytes = CheckedAdd(bytes, n);
                ++last;
            }
            auto arena = Buffer::Allocate(bytes, device);
            size_t offset = 0;
            for (size_t i = first; i < last; ++i) {
                const auto &payload = state->payloads[i];
                auto target = arena.Slice(offset, payload.bytes);
                size_t at = 0;
                for (const auto &page : payload.pages) {
                    target.CopyFrom(page->data, 0, at, page->data.Size());
                    at += page->data.Size();
                }
                value.buffers.push_back(target.ReadOnly());
                offset += payload.bytes;
            }
            first = last;
        }
        auto result = std::shared_ptr<const Message>(new Message(std::move(value)),
                                                     [owner = state](const Message *m) mutable {
                                                         delete m;
                                                         owner.reset();
                                                     });
        state->materialized = result;
        return result;
    }
};

PrefixCache::PrefixCache(size_t bytes, size_t limit, int device) : impl(new Impl(bytes, limit, device)) {}
PrefixCache::~PrefixCache() = default;

void PrefixCache::Publish(const std::string &name, const std::vector<int> &tokens, Message value) {
    Require(!name.empty() && !tokens.empty(), "FastKV: empty prefix identity");
    auto &p = *impl;
    std::lock_guard<std::mutex> lock(p.mutex);
    auto existing = p.Match(name, tokens, tokens.size());
    if (existing && p.entries.at(existing).state->tokens.size() == tokens.size()) {
        p.entries.at(existing).accessed = ++p.clock;
        return;
    }
    if (++p.publications % 64 == 0)
        p.PrunePages();
    auto state = std::make_shared<Impl::State>();
    state->nameSpace = name;
    state->tokens = tokens;
    state->metadata = std::move(value.metadata);
    size_t metadataBytes = CheckedAdd(sizeof(Impl::State) + sizeof(Impl::Entry), name.size());
    metadataBytes = CheckedAdd(metadataBytes, CheckedMul(tokens.size(), sizeof(int)));
    metadataBytes = CheckedAdd(metadataBytes, state->metadata.size());
    metadataBytes = CheckedAdd(metadataBytes, CheckedMul(value.buffers.size(), sizeof(Impl::Payload)));
    size_t neededPages = 0;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<Impl::Page>>> newPages;
    for (const auto &source : value.buffers) {
        auto host = source;
        if (source.Kind() != MemoryKind::Host) {
            host = Buffer::Allocate(source.Size());
            host.CopyFrom(source, 0, 0, source.Size());
        }
        Impl::Payload payload;
        payload.bytes = source.Size();
        state->logicalBytes = CheckedAdd(state->logicalBytes, source.Size());
        for (size_t at = 0; at < source.Size();) {
            const size_t n = std::min(PageBytes, source.Size() - at);
            const auto part = host.Slice(at, n);
            const auto hash = PageHash(part);
            std::shared_ptr<Impl::Page> page;
            auto bucket = p.pages.find(hash);
            if (bucket != p.pages.end())
                for (const auto &weak : bucket->second)
                    if (auto candidate = weak.lock())
                        if (p.SamePage(*candidate, part)) {
                            page = std::move(candidate);
                            break;
                        }
            if (!page)
                for (const auto &candidate : newPages[hash])
                    if (p.SamePage(*candidate, part)) {
                        page = candidate;
                        break;
                    }
            if (!page) {
                page = std::make_shared<Impl::Page>();
                page->data = part;
                newPages[hash].push_back(page);
                neededPages = CheckedAdd(neededPages, CheckedAdd(n, sizeof(Impl::Page)));
            }
            payload.pages.push_back(std::move(page));
            metadataBytes = CheckedAdd(metadataBytes, sizeof(std::shared_ptr<Impl::Page>));
            at += n;
        }
        state->payloads.push_back(std::move(payload));
    }
    state->logicalBytes = CheckedAdd(state->logicalBytes, metadataBytes);
    const size_t needed = CheckedAdd(metadataBytes, neededPages);
    size_t minimum = metadataBytes;
    std::unordered_set<const Impl::Page *> requiredPages;
    for (const auto &payload : state->payloads)
        for (const auto &page : payload.pages)
            if (requiredPages.insert(page.get()).second)
                minimum = CheckedAdd(minimum, CheckedAdd(page->data.Size(), sizeof(Impl::Page)));
    Require(minimum <= p.capacity, "FastKV: prefix exceeds cache byte budget");
    while ((p.maxEntries && p.entries.size() >= p.maxEntries) || needed > p.capacity - p.budget.Used())
        Require(p.Evict(), "FastKV: prefix cache is pinned by active readers");
    state->credit = p.budget.Reserve(metadataBytes);
    for (auto &bucket : newPages)
        for (auto &page : bucket.second) {
            auto credit = p.budget.Reserve(CheckedAdd(page->data.Size(), sizeof(Impl::Page)));
            auto data = Buffer::Allocate(page->data.Size(), p.device);
            data.CopyFrom(page->data, 0, 0, page->data.Size());
            page->data = data.ReadOnly();
            page->credit = std::move(credit);
            p.pages[bucket.first].push_back(page);
        }
    // Eviction may have removed the parent found before admission.
    const auto parent = p.Match(name, tokens, tokens.size());
    const auto id = ++p.clock;
    Impl::Entry entry;
    entry.state = std::move(state);
    entry.hash = NamespaceHash(name);
    for (int token : tokens)
        entry.hash = AppendHash(entry.hash, uint32_t(token));
    entry.accessed = id;
    entry.parent = parent;
    auto &siblings = p.Children(parent);
    for (auto childId : siblings) {
        const auto &child = p.entries.at(childId);
        if (child.state->nameSpace == name && child.state->tokens.size() > tokens.size() &&
            std::equal(tokens.begin(), tokens.end(), child.state->tokens.begin()))
            entry.children.insert(childId);
    }
    auto &bucket = p.lengths[tokens.size()];
    auto indexed = bucket.emplace(entry.hash, id);
    try {
        p.entries.emplace(id, std::move(entry));
        siblings.insert(id);
    } catch (...) {
        bucket.erase(indexed);
        if (bucket.empty())
            p.lengths.erase(tokens.size());
        p.entries.erase(id);
        throw;
    }
    // All allocations have succeeded before changing existing parent links.
    auto &added = p.entries.at(id);
    for (auto child : added.children) {
        p.entries.at(child).parent = id;
        siblings.erase(child);
    }
    p.stats.logicalBytes += added.state->logicalBytes;
}

PrefixCache::Hit PrefixCache::Find(const std::string &name, const std::vector<int> &tokens,
                                   size_t maxTokens) {
    auto &p = *impl;
    std::lock_guard<std::mutex> lock(p.mutex);
    ++p.stats.lookups;
    p.stats.inputTokens += tokens.size();
    const auto id = p.Match(name, tokens, std::min(tokens.size(), maxTokens));
    if (!id)
        return {};
    auto &entry = p.entries.at(id);
    auto state = p.Restore(entry.state);
    entry.accessed = ++p.clock;
    ++p.stats.hits;
    p.stats.hitTokens += entry.state->tokens.size();
    return {entry.state->tokens.size(), std::move(state)};
}
size_t PrefixCache::BytesUsed() const { return impl->budget.Used(); }
PrefixCache::Statistics PrefixCache::Stats() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto result = impl->stats;
    result.entries = impl->entries.size();
    result.bytes = impl->budget.Used();
    return result;
}
} // namespace fastkv
