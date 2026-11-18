#include "fastkv/host_cache.h"
#include "internal.h"
#include "host_pool.h"
#include "numa.h"
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
        Buffer storage;
        bool registered = false;
        cudaEvent_t ready[2]{};
        const DeviceNuma &numa;
        std::unique_ptr<Pool> worker;
        explicit Lane(int device, size_t bytes, int slots) : device(device), numa(NumaForDevice(device)) {
            NumaScope affinity(numa);
            Check(cudaSetDevice(device));
            bytes = CheckedMul(bytes, size_t(slots));
            storage = AllocateNumaHost(bytes, numa);
            staging = storage.MutableData();
            std::memset(staging, 0, bytes);
            Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            auto result = cudaHostRegister(staging, bytes, cudaHostRegisterPortable);
            registered = result == cudaSuccess;
            if (!registered) {
                cudaGetLastError();
                storage = {};
                result = cudaHostAlloc(&staging, bytes, cudaHostAllocPortable);
            }
            if (result != cudaSuccess) { cudaStreamDestroy(stream); Check(result); }
            try {
                for (auto &event : ready) Check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            } catch (...) {
                for (auto event : ready) if (event) cudaEventDestroy(event);
                if (registered) cudaHostUnregister(staging); else cudaFreeHost(staging);
                cudaStreamDestroy(stream); throw;
            }
        }
        ~Lane() {
            worker.reset();
            cudaSetDevice(device);
            cudaStreamSynchronize(stream);
            for (auto event : ready) cudaEventDestroy(event);
            if (registered) cudaHostUnregister(staging); else cudaFreeHost(staging);
            cudaStreamDestroy(stream);
        }
    };
    static void Check(cudaError_t result) {
        if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
    }
    std::map<int, std::unique_ptr<Lane>> lanes;
#endif
    size_t stagingBytes;
    int stages;
  public:
    explicit Copier(size_t bytes) : stagingBytes(std::max(size_t(1), bytes / 2)), stages(bytes > 1 ? 2 : 1) {}
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
        struct Cursor {
            size_t index = 0, offset = 0;
            std::vector<Piece> pieces[2];
            bool pending[2]{};
        };
        for (const auto &group : groups)
            if (!lanes.count(group.first)) lanes[group.first].reset(new Lane(group.first, stagingBytes, stages));
        auto stage = [](const std::vector<Piece> &pieces, char *staging, bool download) {
            for (size_t i = 0; i < pieces.size();) {
                const auto &p = pieces[i];
                if ((p.copy->source.Kind() == MemoryKind::Cuda) != download) { ++i; continue; }
                const Buffer &host = download ? p.copy->target : p.copy->source;
                size_t bytes = p.bytes, j = i + 1;
                for (; j < pieces.size(); ++j) {
                    const auto &next = pieces[j];
                    const Buffer &other = download ? next.copy->target : next.copy->source;
                    if ((next.copy->source.Kind() == MemoryKind::Cuda) != download ||
                        other.Id() != host.Id() || other.Offset() + next.offset != host.Offset() + p.offset + bytes ||
                        next.staging != p.staging + bytes) break;
                    bytes += next.bytes;
                }
                // The identical allocation ID and contiguous offsets prove the
                // combined span stays inside the live allocation, even across
                // logical KV pages. Do not issue one small memcpy per page.
                const char *address = static_cast<const char *>(download ? host.MutableData() : host.Data()) + p.offset;
                std::memcpy(download ? const_cast<char *>(address) : staging + p.staging,
                            download ? staging + p.staging : address, bytes);
                i = j;
            }
        };
        auto finish = [&](Lane &lane, Cursor &cursor, int slot) {
            if (!cursor.pending[slot]) return;
            Check(cudaEventSynchronize(lane.ready[slot]));
            stage(cursor.pieces[slot], static_cast<char *>(lane.staging) + slot * stagingBytes, true);
            cursor.pending[slot] = false;
            cursor.pieces[slot].clear();
        };
        auto run = [&](int device, const std::vector<const Copy *> &group) {
            auto &lane = *lanes.at(device);
            NumaScope affinity(lane.numa);
            Check(cudaSetDevice(lane.device));
            Cursor cursor;
            int slot = 0;
            try {
                while (cursor.index < group.size()) {
                    if (cancel && cancel->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
                    // Reuse only a completed slot. CPU staging of the next
                    // window overlaps DMA of the other, within the same budget.
                    finish(lane, cursor, slot);
                    auto &pieces = cursor.pieces[slot];
                    char *staging = static_cast<char *>(lane.staging) + slot * stagingBytes;
                    size_t used = 0;
                    while (cursor.index < group.size() && used < stagingBytes) {
                        const Copy *copy = group[cursor.index];
                        const size_t bytes = std::min(stagingBytes - used, copy->source.Size() - cursor.offset);
                        pieces.push_back({copy, cursor.offset, bytes, used});
                        used += bytes; cursor.offset += bytes;
                        if (cursor.offset == copy->source.Size()) { ++cursor.index; cursor.offset = 0; }
                    }
                    stage(pieces, staging, false);
                    for (size_t i = 0; i < pieces.size();) {
                        const auto &p = pieces[i];
                        const bool download = p.copy->source.Kind() == MemoryKind::Cuda;
                        const Buffer &gpu = download ? p.copy->source : p.copy->target;
                        size_t bytes = p.bytes, j = i + 1;
                        for (; j < pieces.size(); ++j) {
                            const auto &next = pieces[j];
                            const Buffer &other = download ? next.copy->source : next.copy->target;
                            if ((next.copy->source.Kind() == MemoryKind::Cuda) != download ||
                                other.Id() != gpu.Id() || other.Offset() + next.offset != gpu.Offset() + p.offset + bytes) break;
                            bytes += next.bytes;
                        }
                        void *host = staging + p.staging;
                        const void *address = static_cast<const char *>(gpu.Data()) + p.offset;
                        Check(cudaMemcpyAsync(download ? host : const_cast<void *>(address),
                                              download ? address : host, bytes,
                                              download ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, lane.stream));
                        i = j;
                    }
                    Check(cudaEventRecord(lane.ready[slot], lane.stream));
                    cursor.pending[slot] = true;
                    slot = (slot + 1) % stages;
                }
                for (int s = 0; s < stages; ++s) finish(lane, cursor, s);
            } catch (...) {
                cudaStreamSynchronize(lane.stream);
                throw;
            }
        };
        if (groups.size() == 1) run(groups.begin()->first, groups.begin()->second);
        else {
            // Persistent bounded workers keep CPU staging for different GPUs
            // independent as well as their DMA streams. Reuse FastKV's executor.
            std::vector<std::future<void>> completed;
            completed.reserve(groups.size());
            std::exception_ptr error;
            try {
                for (const auto &group : groups) {
                    auto &lane = *lanes.at(group.first);
                    if (!lane.worker) lane.worker.reset(new Pool(1, 1));
                    auto task = std::make_shared<std::packaged_task<void()>>(
                        [&, device = group.first] { run(device, groups.at(device)); });
                    auto future = task->get_future();
                    Require(lane.worker->TryEnqueue([task] { (*task)(); }), "FastKV: copy lane busy");
                    completed.push_back(std::move(future));
                }
            } catch (...) { error = std::current_exception(); }
            // A failure/cancellation on one GPU cannot free buffers still in
            // use by another GPU, including failure partway through submission.
            for (auto &future : completed) {
                try { future.get(); } catch (...) { if (!error) error = std::current_exception(); }
            }
            if (error) std::rethrow_exception(error);
        }
#endif
        if (cancel && cancel->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
    }
};
} // namespace

void HostCache::Options::SetBudgets(size_t total, int64_t kv, int64_t swa) {
    Require(kv >= -1 && swa >= -1, "FastKV: negative L2 budget");
    if (kv < 0 && swa < 0) {
        bytes = total;
        swaBytes = -1;
        return;
    }
    const size_t specified = CheckedAdd(size_t(std::max<int64_t>(0, kv)), size_t(std::max<int64_t>(0, swa)));
    Require(!total || specified <= total, "FastKV: split L2 budgets exceed the total");
    const size_t effective = (kv >= 0 && swa >= 0) || !total ? specified : total;
    const size_t window = swa >= 0 ? size_t(swa) : effective - size_t(kv);
    Require(window <= INT64_MAX, "FastKV: window budget out of range");
    bytes = effective;
    swaBytes = int64_t(window);
}

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
    struct Payload { size_t bytes; bool prefixRange; CacheStorage storage; std::vector<Region> regions; };
    std::shared_ptr<void> credit;
    std::unique_ptr<HostIndexLease> indexLease;
    std::string name;
    std::vector<int> tokens;
    Bytes metadata;
    std::vector<Payload> payloads;
    size_t bytes = 0, windowBytes = 0;
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
    struct ResidentPool {
        size_t limit;
        Budget budget;
        HostPool slabs;
        explicit ResidentPool(size_t limit) : limit(limit), budget(limit, "host cache") {}
    };
    ResidentPool kv, window;
    Budget inflight;
    std::mutex allocationMutex;
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
    explicit Impl(Options options) : options(options),
                                    kv(options.bytes - size_t(std::max<int64_t>(0, options.swaBytes))),
                                    window(size_t(std::max<int64_t>(0, options.swaBytes))),
                                    inflight(options.inflightBytes, "host cache transfers") {
        Require(options.swaBytes >= -1 && window.limit <= options.bytes && options.bytes && options.inflightBytes && options.stagingBytes && options.queueDepth &&
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
    bool Evict(bool windowOnly = false) {
        std::shared_ptr<Entry> *victim = nullptr;
        for (auto &length : entries) for (auto &item : length.second)
            if ((!windowOnly || item.second->windowBytes) && item.second.use_count() == 1 && (!victim || item.second->accessed < (*victim)->accessed)) victim = &item.second;
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
        if (bytes > kv.limit) throw CapacityBusy("FastKV: metadata exceeds KV budget");
        while (bytes > kv.limit - kv.budget.Used())
            if (!Evict()) throw CapacityBusy("FastKV: host cache pinned or full");
        return kv.budget.Reserve(bytes);
    }
    Buffer Allocate(size_t bytes, int device, CacheStorage storage) {
        const bool windowOnly = options.swaBytes >= 0 && storage == CacheStorage::Window;
        auto &pool = windowOnly ? window : kv;
        if (bytes > pool.limit) throw CapacityBusy("FastKV: payload exceeds its cache budget");
        // Serialize growth, not copying or lease release. Eviction may free
        // holes in an existing slab, so retry reuse after each evicted entry.
        std::lock_guard<std::mutex> allocationLock(allocationMutex);
        const auto &numa = NumaForDevice(device);
        const size_t preferred = std::max(bytes, std::min(size_t(64) << 20, pool.limit / 16));
        for (;;) {
            auto reused = pool.slabs.TryAllocate(bytes, numa.node);
            if (reused.Size()) return reused;
            std::shared_ptr<void> credit;
            {
                std::lock_guard<std::mutex> lock(mutex);
                const size_t available = pool.limit - pool.budget.Used();
                // Do not shrink new slabs to the last few free bytes: that
                // would recreate unbounded small mappings near a full cache.
                if (available >= preferred) {
                    credit = pool.budget.Reserve(preferred);
                } else if (!Evict(windowOnly)) {
                    throw CapacityBusy("FastKV: host cache pinned or full");
                }
            }
            if (credit) return pool.slabs.Add(AllocateNumaHost(preferred, numa), std::move(credit), numa.node, bytes);
        }
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
        for (size_t i = 0; i < snapshot.payloads.size(); ++i) {
            const auto &payload = snapshot.payloads[i];
            const size_t bytes = snapshot.state.buffers[i].Size();
            entry->bytes = CheckedAdd(entry->bytes, bytes);
            if (payload.storage == CacheStorage::Window)
                entry->windowBytes = CheckedAdd(entry->windowBytes, bytes);
            metadata = CheckedAdd(metadata, sizeof(Entry::Payload) + CheckedMul(payload.regions.size(), sizeof(Entry::Region) + sizeof(HostBlock) + 128));
            for (const auto &region : payload.regions) metadata = CheckedAdd(metadata, CheckedMul(region.identity.size() + name.size(), 2));
        }
        // An impossible checkpoint must not evict an otherwise reusable one.
        if (options.swaBytes >= 0) {
            const size_t kvBytes = CheckedAdd(metadata, entry->bytes - entry->windowBytes);
            if (kvBytes > kv.limit || entry->windowBytes > window.limit)
                throw CapacityBusy("FastKV: checkpoint exceeds its KV or SWA budget");
        }
        entry->credit = Reserve(metadata);
        std::vector<Copy> copies;
        std::vector<std::pair<std::string, std::shared_ptr<HostBlock>>> newBlocks;
        uint64_t copied = 0, reused = 0;
        const bool inherit = parent && parent->name == name && parent->tokens.size() <= tokens.size() &&
            std::equal(parent->tokens.begin(), parent->tokens.end(), tokens.begin()) && parent->payloads.size() == snapshot.payloads.size();
        for (size_t i = 0; i < snapshot.payloads.size(); ++i) {
            const auto &description = snapshot.payloads[i]; const auto &source = snapshot.state.buffers[i];
            Entry::Payload payload{source.Size(), description.prefixRange, description.storage, {}};
            Buffer arena; size_t used = 0;
            for (size_t r = 0; r < description.regions.size(); ++r) {
                const auto &region = description.regions[r];
                std::shared_ptr<HostBlock> block;
                const std::string key = region.identity.empty() ? std::string() : std::to_string(name.size()) + ":" + name + (description.storage == CacheStorage::Window ? "W" : "K") + region.identity;
                if (!key.empty()) {
                    std::lock_guard<std::mutex> lock(index->mutex);
                    auto found = index->blocks.find(key); if (found != index->blocks.end()) block = found->second.block.lock();
                    Require(!block || block->data.Size() == region.bytes, "FastKV: immutable region changed size");
                }
                if (!block && inherit && description.prefixRange && parent->payloads[i].prefixRange &&
                    parent->payloads[i].storage == description.storage) {
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
                        arena = Allocate(bytes, source.Device(), description.storage); used = 0;
                    }
                    auto target = arena.Slice(used, region.bytes); used += region.bytes;
                    AddCopy(copies, source.Slice(region.offset, region.bytes), target);
                    block = std::make_shared<HostBlock>(HostBlock{target.ReadOnly()}); copied += region.bytes;
                }
                payload.regions.push_back({region.offset, block});
                if (!key.empty()) newBlocks.emplace_back(key, block);
            }
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
    return Restore(std::move(hit), std::move(devices), {}, result, std::move(cancellation));
}
bool HostCache::RestoreIntoAsync(Hit hit, std::vector<Buffer> targets, std::future<Message> &result) {
    Require(bool(hit) && targets.size() == hit.entry->payloads.size(), "FastKV: invalid target count");
    std::vector<int> devices;
    devices.reserve(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        const size_t bytes = hit.entry->payloads[i].bytes;
        Require(targets[i].Size() == bytes && (!bytes || targets[i].MutableData()),
                "FastKV: incomplete restore target");
        devices.push_back(targets[i].Device());
    }
    return Restore(std::move(hit), std::move(devices), std::move(targets), result, {});
}
bool HostCache::Restore(Hit hit, std::vector<int> devices, std::vector<Buffer> targets,
                        std::future<Message> &result, std::shared_ptr<Cancellation> cancellation) {
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
        [p = impl.get(), hit = std::move(hit), devices = std::move(devices), targets = std::move(targets), promise, cancellation, destinationCredit](Copier &copier, std::shared_ptr<void> credit) {
            try {
                if (cancellation && cancellation->Cancelled()) throw std::runtime_error("FastKV: restore cancelled");
                Message message{hit.entry->metadata, {}};
                std::vector<Copy> copies;
                auto lease = std::make_shared<std::pair<std::shared_ptr<void>, std::shared_ptr<void>>>(credit, destinationCredit);
                // Bound allocations; small tensor payloads share an arena.
                Buffer arena; size_t used = 0;
                for (size_t i = 0; i < devices.size(); ++i) {
                    const auto &payload = hit.entry->payloads[i];
                    if (targets.empty() && (arena.Device() != devices[i] || payload.bytes > arena.Size() - used)) {
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
                    auto target = targets.empty() ? arena.Slice(used, payload.bytes) : targets[i].WithLease(lease);
                    if (targets.empty()) used += payload.bytes;
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
    auto stats = impl->stats;
    const size_t kv = impl->kv.budget.Used(), window = impl->window.budget.Used();
    stats.bytes = kv + window;
    if (impl->options.swaBytes >= 0) { stats.kvBytes = kv; stats.swaBytes = window; }
    stats.inflightBytes = impl->inflight.Used();
    stats.queued = impl->queue.size() + impl->restores.size() + impl->running;
    for (const auto &bucket : impl->entries) stats.entries += bucket.second.size();
    return stats;
}
void HostCache::Drain() {
    std::unique_lock<std::mutex> lock(impl->mutex);
    impl->changed.wait(lock, [&] { return impl->queue.empty() && impl->restores.empty() && !impl->running; });
}
} // namespace fastkv
