#include "fastkv/host_cache.h"
#include "internal.h"
#include <algorithm>
#include <cstring>
#include <unordered_map>
#ifdef FASTKV_CUDA
#include <cuda_runtime_api.h>
#endif

namespace fastkv {
namespace {
uint64_t Hash(uint64_t h, uint32_t v) { return (h ^ v) * 1099511628211ULL; }
uint64_t Namespace(const std::string &name) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : name) h = Hash(h, c);
    return h;
}
struct Copy { Buffer source, target; };
void AddCopy(std::vector<Copy> &copies, Buffer source, Buffer target) {
    if (source.Size()) copies.push_back({std::move(source), std::move(target)});
}
// Pageable resident storage and a small pinned staging window are separate
// budgets. Each worker owns private streams: no default-stream/device barrier.
// Adjacent regions are staged together, avoiding one DMA submission per page.
class Copier {
#ifdef FASTKV_CUDA
    struct Lane {
        int device;
        cudaStream_t stream = nullptr;
        void *staging = nullptr;
        explicit Lane(int device, size_t bytes) : device(device) {
            Check(cudaSetDevice(device));
            Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            auto result = cudaHostAlloc(&staging, bytes, cudaHostAllocPortable);
            if (result != cudaSuccess) { cudaStreamDestroy(stream); Check(result); }
        }
        ~Lane() {
            cudaSetDevice(device);
            cudaStreamSynchronize(stream);
            cudaFreeHost(staging); cudaStreamDestroy(stream);
        }
    };
    static void Check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }
    std::map<int, std::unique_ptr<Lane>> lanes;
#endif
    size_t stagingBytes;
  public:
    explicit Copier(size_t bytes) : stagingBytes(bytes) {}
    void Run(const std::vector<Copy> &copies, const std::shared_ptr<Cancellation> &cancel = {}) {
        // Group by device to overlap the independent PCIe links. A staging
        // window gathers/scatters CPU regions but transfers contiguous GPU spans.
        std::map<int, std::vector<const Copy *>> groups;
        for (const auto &copy : copies) {
            if (copy.source.Kind() == MemoryKind::Host && copy.target.Kind() == MemoryKind::Host) {
                std::memcpy(copy.target.MutableData(), copy.source.Data(), copy.source.Size());
            } else {
                const int device = copy.source.Kind() == MemoryKind::Cuda ? copy.source.Device() : copy.target.Device();
                groups[device].push_back(&copy);
            }
        }
#ifndef FASTKV_CUDA
        Require(groups.empty(), "FastKV: CUDA host cache requires FASTKV_CUDA");
#else
        struct Piece { const Copy *copy; size_t offset, bytes, staging; };
        struct Cursor { size_t index = 0, offset = 0; std::vector<Piece> pieces; };
        std::map<int, Cursor> cursors;
        for (const auto &group : groups)
            if (!lanes.count(group.first)) lanes[group.first].reset(new Lane(group.first, stagingBytes));
        bool more = true;
        try {
            while (more) {
                if (cancel && cancel->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
                more = false;
                for (const auto &group : groups) {
                    auto &cursor = cursors[group.first]; cursor.pieces.clear();
                    auto &lane = *lanes.at(group.first);
                    Check(cudaSetDevice(lane.device));
                    size_t used = 0;
                    while (cursor.index < group.second.size() && used < stagingBytes) {
                        const Copy *copy = group.second[cursor.index];
                        const size_t bytes = std::min(stagingBytes - used, copy->source.Size() - cursor.offset);
                        cursor.pieces.push_back({copy, cursor.offset, bytes, used});
                        if (copy->source.Kind() == MemoryKind::Host)
                            std::memcpy(static_cast<char *>(lane.staging) + used,
                                        static_cast<const char *>(copy->source.Data()) + cursor.offset, bytes);
                        used += bytes; cursor.offset += bytes;
                        if (cursor.offset == copy->source.Size()) { ++cursor.index; cursor.offset = 0; }
                    }
                    for (size_t i = 0; i < cursor.pieces.size();) {
                        const auto &p = cursor.pieces[i];
                        const bool download = p.copy->source.Kind() == MemoryKind::Cuda;
                        const Buffer &gpu = download ? p.copy->source : p.copy->target;
                        size_t bytes = p.bytes, j = i + 1;
                        for (; j < cursor.pieces.size(); ++j) {
                            const auto &next = cursor.pieces[j];
                            const Buffer &other = download ? next.copy->source : next.copy->target;
                            if ((next.copy->source.Kind() == MemoryKind::Cuda) != download ||
                                other.Id() != gpu.Id() || other.Offset() + next.offset != gpu.Offset() + p.offset + bytes) break;
                            bytes += next.bytes;
                        }
                        void *host = static_cast<char *>(lane.staging) + p.staging;
                        const void *address = static_cast<const char *>(gpu.Data()) + p.offset;
                        Check(cudaMemcpyAsync(download ? host : const_cast<void *>(address),
                                              download ? address : host, bytes,
                                              download ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, lane.stream));
                        i = j;
                    }
                    more |= cursor.index < group.second.size();
                }
                for (const auto &group : groups) {
                    auto &lane = *lanes.at(group.first);
                    Check(cudaSetDevice(lane.device)); Check(cudaStreamSynchronize(lane.stream));
                    for (const auto &p : cursors[group.first].pieces)
                        if (p.copy->source.Kind() == MemoryKind::Cuda)
                            std::memcpy(static_cast<char *>(p.copy->target.MutableData()) + p.offset,
                                        static_cast<char *>(lane.staging) + p.staging, p.bytes);
                }
            }
        } catch (...) {
            // Buffers cannot be released until every previously issued DMA is done.
            for (auto &lane : lanes) {
                cudaSetDevice(lane.first); cudaStreamSynchronize(lane.second->stream);
            }
            throw;
        }
#endif
        if (cancel && cancel->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
    }
};
} // namespace

struct HostBlock { Buffer data; };
struct HostIndex {
    struct Record { std::weak_ptr<HostBlock> block; size_t users = 0; };
    std::mutex mutex;
    std::unordered_map<std::string, Record> blocks;
};
// Each published checkpoint owns its index references. Retiring a version
// removes only its keys, without a whole-cache scan or stale alias accumulation.
struct HostIndexLease {
    std::shared_ptr<HostIndex> index;
    std::vector<std::string> keys;
    ~HostIndexLease() {
        std::lock_guard<std::mutex> lock(index->mutex);
        for (const auto &key : keys) {
            auto it = index->blocks.find(key);
            if (it != index->blocks.end() && --it->second.users == 0) index->blocks.erase(it);
        }
    }
};
struct HostCache::Entry {
    struct Region { size_t offset; std::shared_ptr<HostBlock> block; };
    struct Payload { size_t bytes; bool prefixRange; std::vector<Region> regions; };
    std::shared_ptr<void> credit;
    std::unique_ptr<HostIndexLease> indexLease;
    std::string name;
    std::vector<int> tokens;
    Bytes metadata;
    std::vector<Payload> payloads;
    size_t bytes = 0;
    uint64_t accessed = 0, hash = 0;
};
size_t HostCache::Hit::Tokens() const { return entry ? entry->tokens.size() : 0; }
size_t HostCache::Hit::Bytes() const { return entry ? entry->bytes : 0; }
const fastkv::Bytes &HostCache::Hit::Metadata() const {
    Require(bool(entry), "FastKV: empty host hit"); return entry->metadata;
}
size_t HostCache::Hit::Payloads() const { return entry ? entry->payloads.size() : 0; }

struct HostCache::Impl {
    Options options;
    Budget budget, inflight;
    std::map<int, std::unique_ptr<Budget>> destinationBudgets;
    mutable std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    size_t running = 0;
    uint64_t clock = 0;
    Statistics stats;
    std::vector<std::thread> workers;
    std::deque<std::function<void(Copier &)>> queue, restores;
    std::map<size_t, std::unordered_multimap<uint64_t, std::shared_ptr<Entry>>> entries;
    // The index owns no data and is safe to outlive the cache through a Hit.
    std::shared_ptr<HostIndex> index = std::make_shared<HostIndex>();
    explicit Impl(Options options) : options(options), budget(options.bytes, "host cache"),
                                    inflight(options.inflightBytes, "host cache transfers") {
        Require(options.bytes && options.inflightBytes && options.stagingBytes && options.queueDepth &&
                    options.workers > 0 && options.workers <= 64, "FastKV: invalid host cache options");
        for (const auto &limit : options.restoreDeviceBytes) {
            Require(limit.first >= -1, "FastKV: invalid restore device");
            destinationBudgets[limit.first].reset(new Budget(limit.second, "host cache restore destination"));
        }
        try {
            for (unsigned i = 0; i < options.workers; ++i) workers.emplace_back([this] {
                Copier copier(this->options.stagingBytes);
                while (true) {
                    std::function<void(Copier &)> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock, [&] { return stopping || !queue.empty() || !restores.empty(); });
                        if (queue.empty() && restores.empty()) return;
                        auto &ready = restores.empty() ? queue : restores;
                        job = std::move(ready.front()); ready.pop_front(); ++running;
                    }
                    try { job(copier); }
                    catch (const CapacityBusy &) { std::lock_guard<std::mutex> lock(mutex); ++stats.skipped; }
                    catch (...) { std::lock_guard<std::mutex> lock(mutex); ++stats.failures; }
                    job = {};
                    { std::lock_guard<std::mutex> lock(mutex); --running; }
                    changed.notify_all();
                }
            });
        } catch (...) { Stop(); throw; }
    }
    void Stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_all();
        for (auto &worker : workers) if (worker.joinable()) worker.join();
    }
    ~Impl() { Stop(); }
    std::shared_ptr<Entry> Match(const std::string &name, const std::vector<int> &tokens) const {
        uint64_t hash = Namespace(name);
        size_t position = 0;
        std::shared_ptr<Entry> best;
        for (const auto &length : entries) {
            if (length.first > tokens.size()) break;
            for (; position < length.first; ++position) hash = Hash(hash, uint32_t(tokens[position]));
            const auto range = length.second.equal_range(hash);
            for (auto it = range.first; it != range.second; ++it)
                if (it->second->name == name && std::equal(it->second->tokens.begin(), it->second->tokens.end(), tokens.begin()))
                    best = it->second;
        }
        return best;
    }
    bool Evict() {
        std::shared_ptr<Entry> *victim = nullptr;
        for (auto &length : entries) for (auto &item : length.second)
            if (item.second.use_count() == 1 && (!victim || item.second->accessed < (*victim)->accessed)) victim = &item.second;
        if (!victim) return false;
        const size_t length = (*victim)->tokens.size();
        auto &bucket = entries.at(length);
        for (auto it = bucket.begin(); it != bucket.end(); ++it) if (&it->second == victim) { bucket.erase(it); break; }
        if (bucket.empty()) entries.erase(length);
        ++stats.evictions;
        return true;
    }
    std::shared_ptr<void> Reserve(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        while (bytes > options.bytes - budget.Used())
            if (!Evict()) throw CapacityBusy("FastKV: host cache pinned or full");
        return budget.Reserve(bytes);
    }
    Buffer Allocate(size_t bytes) {
        struct Allocation { std::shared_ptr<void> credit; Buffer data; };
        auto owner = std::make_shared<Allocation>();
        owner->credit = Reserve(bytes); owner->data = Buffer::Allocate(bytes);
        return Buffer::Borrow(owner->data.MutableData(), bytes, MemoryKind::Host, -1, owner);
    }
    bool Enqueue(size_t bytes, bool priority, std::function<void(Copier &, std::shared_ptr<void>)> job) {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping || queue.size() + restores.size() + running >= options.queueDepth) { ++stats.skipped; return false; }
        std::shared_ptr<void> credit;
        try { credit = inflight.Reserve(bytes); }
        catch (const CapacityBusy &) { ++stats.skipped; return false; }
        auto task = [job = std::move(job), credit = std::move(credit)](Copier &c) { job(c, credit); };
        if (priority) restores.emplace_back(std::move(task)); else queue.emplace_back(std::move(task));
        changed.notify_one(); return true;
    }
    void Publish(const std::string &name, const std::vector<int> &tokens, const CacheSnapshot &snapshot,
                 const std::shared_ptr<const Entry> &parent, Copier &copier) {
        auto entry = std::make_shared<Entry>();
        entry->name = name; entry->tokens = tokens; entry->metadata = snapshot.state.metadata;
        entry->hash = Namespace(name);
        for (int token : tokens) entry->hash = Hash(entry->hash, uint32_t(token));
        size_t metadata = sizeof(Entry) + sizeof(HostIndexLease) + name.size() + snapshot.state.metadata.size() + CheckedMul(tokens.size(), sizeof(int));
        for (const auto &payload : snapshot.payloads) {
            metadata = CheckedAdd(metadata, sizeof(Entry::Payload) + CheckedMul(payload.regions.size(), sizeof(Entry::Region) + sizeof(HostBlock) + 128));
            for (const auto &region : payload.regions) metadata = CheckedAdd(metadata, CheckedMul(region.identity.size() + name.size(), 2));
        }
        entry->credit = Reserve(metadata);
        std::vector<Copy> copies;
        std::vector<std::pair<std::string, std::shared_ptr<HostBlock>>> newBlocks;
        uint64_t copied = 0, reused = 0;
        const bool inherit = parent && parent->name == name && parent->tokens.size() <= tokens.size() &&
            std::equal(parent->tokens.begin(), parent->tokens.end(), tokens.begin()) && parent->payloads.size() == snapshot.payloads.size();
        for (size_t i = 0; i < snapshot.payloads.size(); ++i) {
            const auto &description = snapshot.payloads[i]; const auto &source = snapshot.state.buffers[i];
            Entry::Payload payload{source.Size(), description.prefixRange, {}};
            Buffer arena; size_t used = 0;
            for (size_t r = 0; r < description.regions.size(); ++r) {
                const auto &region = description.regions[r];
                std::shared_ptr<HostBlock> block;
                const std::string key = region.identity.empty() ? std::string() : std::to_string(name.size()) + ":" + name + region.identity;
                if (!key.empty()) {
                    std::lock_guard<std::mutex> lock(index->mutex);
                    auto found = index->blocks.find(key); if (found != index->blocks.end()) block = found->second.block.lock();
                    Require(!block || block->data.Size() == region.bytes, "FastKV: immutable region changed size");
                }
                if (!block && inherit && description.prefixRange && parent->payloads[i].prefixRange) {
                    const auto &old = parent->payloads[i].regions;
                    auto it = std::lower_bound(old.begin(), old.end(), region.offset,
                        [](const Entry::Region &v, size_t at) { return v.offset < at; });
                    if (it != old.end() && it->offset == region.offset && it->block->data.Size() == region.bytes)
                        block = it->block;
                }
                if (block) reused += region.bytes;
                else {
                    if (region.bytes > arena.Size() - used) {
                        size_t bytes = region.bytes;
                        for (size_t j = r + 1; j < description.regions.size() && bytes < (2 << 20); ++j) {
                            if (description.regions[j].bytes > (2 << 20) - bytes) break;
                            bytes += description.regions[j].bytes;
                        }
                        arena = Allocate(bytes); used = 0;
                    }
                    auto target = arena.Slice(used, region.bytes); used += region.bytes;
                    AddCopy(copies, source.Slice(region.offset, region.bytes), target);
                    block = std::make_shared<HostBlock>(HostBlock{target.ReadOnly()}); copied += region.bytes;
                }
                payload.regions.push_back({region.offset, block});
                if (!key.empty()) newBlocks.emplace_back(key, block);
            }
            entry->bytes = CheckedAdd(entry->bytes, payload.bytes);
            entry->payloads.push_back(std::move(payload));
        }
        copier.Run(copies);
        std::lock_guard<std::mutex> lock(mutex);
        auto previous = Match(name, tokens);
        if (!previous || previous->tokens.size() != tokens.size()) {
            entry->indexLease.reset(new HostIndexLease{index, {}});
            entry->indexLease->keys.reserve(newBlocks.size());
            {
                std::lock_guard<std::mutex> indexLock(index->mutex);
                for (const auto &block : newBlocks) {
                    entry->indexLease->keys.push_back(block.first);
                    auto &record = index->blocks[block.first];
                    record.block = block.second; ++record.users;
                }
            }
            entry->accessed = ++clock;
            entries[tokens.size()].emplace(entry->hash, entry);
            ++stats.publications;
        }
        stats.copiedToHost += copied; stats.reusedBytes += reused;
    }
};

HostCache::HostCache(Options options) : impl(new Impl(options)) {}
HostCache::~HostCache() = default;
HostCache::Hit HostCache::Find(const std::string &name, const std::vector<int> &tokens) const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto entry = impl->Match(name, tokens);
    if (entry) entry->accessed = ++impl->clock;
    Hit hit; hit.entry = std::move(entry);
    return hit;
}
size_t HostCache::ProbeTokens(const std::string &name, const std::vector<int> &tokens) const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto entry = impl->Match(name, tokens); return entry ? entry->tokens.size() : 0;
}
bool HostCache::PublishAsync(const std::string &name, std::vector<int> tokens, CacheSnapshot snapshot, Hit parent) {
    Require(!name.empty() && !tokens.empty(), "FastKV: empty host cache identity");
    Require(snapshot.payloads.size() == snapshot.state.buffers.size(), "FastKV: incomplete cache descriptor");
    size_t sourceBytes = snapshot.state.metadata.size();
    std::map<int, size_t> bytesByDevice;
    for (size_t i = 0; i < snapshot.payloads.size(); ++i) {
        size_t offset = 0;
        for (const auto &region : snapshot.payloads[i].regions) {
            Require(region.offset == offset && region.bytes, "FastKV: invalid cache region coverage");
            offset = CheckedAdd(offset, region.bytes);
        }
        Require(offset == snapshot.state.buffers[i].Size(), "FastKV: incomplete cache payload");
        sourceBytes = CheckedAdd(sourceBytes, offset);
        const int device = snapshot.state.buffers[i].Device();
        bytesByDevice[device] = CheckedAdd(bytesByDevice[device], offset);
    }
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        for (const auto &item : bytesByDevice) {
            auto limit = impl->options.restoreDeviceBytes.find(item.first);
            if (limit != impl->options.restoreDeviceBytes.end() && item.second > limit->second) {
                ++impl->stats.skipped; return false;
            }
        }
        auto hit = impl->Match(name, tokens);
        if (hit && hit->tokens.size() == tokens.size()) { hit->accessed = ++impl->clock; return true; }
    }
    return impl->Enqueue(sourceBytes, false,
        [p = impl.get(), name, tokens = std::move(tokens), snapshot = std::move(snapshot), parent = std::move(parent)](Copier &copier, std::shared_ptr<void>) {
            p->Publish(name, tokens, snapshot, parent.entry, copier);
        });
}
bool HostCache::RestoreAsync(Hit hit, std::vector<int> devices, std::future<Message> &result,
                             std::shared_ptr<Cancellation> cancellation) {
    Require(bool(hit) && devices.size() == hit.entry->payloads.size(), "FastKV: invalid restore placement");
    Require(!result.valid(), "FastKV: restore future is already in use");
    auto destinationCredit = std::make_shared<std::vector<std::shared_ptr<void>>>();
    std::map<int, size_t> bytesByDevice;
    for (size_t i = 0; i < devices.size(); ++i)
        bytesByDevice[devices[i]] = CheckedAdd(bytesByDevice[devices[i]], hit.entry->payloads[i].bytes);
    try {
        for (const auto &item : bytesByDevice) {
            auto limit = impl->destinationBudgets.find(item.first);
            if (limit != impl->destinationBudgets.end()) {
                Require(item.second <= impl->options.restoreDeviceBytes.at(item.first), "FastKV: checkpoint exceeds restore device budget");
                destinationCredit->push_back(limit->second->Reserve(item.second));
            }
        }
    } catch (const CapacityBusy &) { return false; }
    auto promise = std::make_shared<std::promise<Message>>();
    const size_t bytes = hit.Bytes();
    const bool submitted = impl->Enqueue(bytes, true,
        [p = impl.get(), hit = std::move(hit), devices = std::move(devices), promise, cancellation, destinationCredit](Copier &copier, std::shared_ptr<void> credit) {
            try {
                if (cancellation && cancellation->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
                Message message{hit.entry->metadata, {}};
                std::vector<Copy> copies;
                auto lease = std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<void>>>(credit, destinationCredit);
                // Bound allocations; small tensor payloads share an arena.
                Buffer arena; size_t used = 0;
                for (size_t i = 0; i < devices.size(); ++i) {
                    const auto &payload = hit.entry->payloads[i];
                    if (arena.Device() != devices[i] || payload.bytes > arena.Size() - used) {
                        size_t capacity = payload.bytes;
                        for (size_t j = i + 1; j < devices.size() && devices[j] == devices[i] && capacity < (64 << 20); ++j) {
                            if (hit.entry->payloads[j].bytes > (64 << 20) - capacity) break;
                            capacity += hit.entry->payloads[j].bytes;
                        }
                        struct Allocation { Buffer buffer; std::shared_ptr<void> credit; };
                        auto owner = std::make_shared<Allocation>();
                        owner->credit = lease; owner->buffer = Buffer::Allocate(capacity, devices[i]);
                        arena = Buffer::Borrow(owner->buffer.MutableData(), capacity, owner->buffer.Kind(), devices[i], owner);
                        used = 0;
                    }
                    auto target = arena.Slice(used, payload.bytes); used += payload.bytes;
                    for (const auto &region : payload.regions)
                        AddCopy(copies, region.block->data, target.Slice(region.offset, region.block->data.Size()));
                    message.buffers.push_back(target.ReadOnly());
                }
                const auto begin = std::chrono::steady_clock::now();
                copier.Run(copies, cancellation);
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
                {
                    std::lock_guard<std::mutex> lock(p->mutex);
                    ++p->stats.restores; p->stats.restoredBytes += hit.Bytes();
                    p->stats.restoreMicroseconds += std::max(int64_t(1), int64_t(us));
                }
                promise->set_value(std::move(message));
            } catch (...) {
                { std::lock_guard<std::mutex> lock(p->mutex); ++p->stats.failures; }
                promise->set_exception(std::current_exception());
            }
        });
    if (submitted) result = promise->get_future();
    return submitted;
}
HostCache::Statistics HostCache::Stats() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto stats = impl->stats; stats.bytes = impl->budget.Used(); stats.inflightBytes = impl->inflight.Used();
    stats.queued = impl->queue.size() + impl->restores.size() + impl->running;
    for (const auto &bucket : impl->entries) stats.entries += bucket.second.size();
    return stats;
}
void HostCache::Drain() {
    std::unique_lock<std::mutex> lock(impl->mutex);
    impl->changed.wait(lock, [&] { return impl->queue.empty() && impl->restores.empty() && !impl->running; });
}
} // namespace fastkv
