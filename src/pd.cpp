#include "pd.h"
#include "fastkv/host_cache.h"
#include "json11.hpp"
#include "models/basellm.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <thread>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
namespace {
void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(std::string("FastKV PD: ") + message);
}
uint64_t TensorBytes(const Data &data) {
    uint64_t elements = data.dims.empty() ? 0 : 1;
    Check(data.dims.size() <= 8, "too many tensor dimensions");
    for (int n : data.dims) {
        Check(n >= 0 && (!n || elements <= (UINT64_MAX / 16) / n), "invalid tensor shape");
        elements *= n;
    }
    if (data.dataType == FP4_E2M1) {
        Check(data.dims.empty() || data.dims.back() % 16 == 0, "FP4 row width must be block16 aligned");
        return elements / 16 * 9;
    }
    Check(data.unitSizeDiv == 1, "packed element type requires a model wire codec");
    return elements * data.unitSize;
}
constexpr uint32_t PDWireMagic = 0x32564450;
constexpr size_t PDTransferArenaBytes = size_t(64) << 20;
bool HasPDEnvelope(const fastkv::Bytes &bytes) {
    if (bytes.size() < 4) return false;
    fastkv::Reader reader(bytes);
    return reader.U32() == PDWireMagic;
}
} // namespace
fastkv::Message PDStatePayload(const fastkv::Message &value) {
    if (!HasPDEnvelope(value.metadata)) return value;
    fastkv::Reader reader(value.metadata);
    reader.U32();
    fastkv::Message result;
    result.metadata = reader.Blob();
    const auto placement = reader.Ints(value.buffers.size());
    Check(placement.size() == value.buffers.size(), "placement count mismatch");
    reader.End();
    result.buffers = value.buffers;
    return result;
}
std::vector<int> PDReceivePlacement(const fastkv::Bytes &metadata,
                                   size_t buffers, const std::vector<int> &devices) {
    if (!HasPDEnvelope(metadata)) return {};
    fastkv::Reader reader(metadata);
    reader.U32(); reader.Blob();
    auto result = reader.Ints(buffers);
    Check(result.size() == buffers, "placement count mismatch");
    reader.End();
    for (int &rank : result) {
        Check(rank >= -1 && (rank == -1 || size_t(rank) < devices.size()), "unknown destination rank");
        if (rank >= 0) rank = devices[rank];
    }
    return result;
}
struct PDStateWriter::Allocations {
    std::mutex mutex;
    std::vector<std::pair<int, size_t>> plan;
    std::vector<fastkv::Buffer> reserved;
    bool prepared = false;
};
PDStateWriter::PDStateWriter(fastkv::BufferPool *pool)
    : pool(pool), allocations(std::make_shared<Allocations>()) {}
PDStateWriter PDStateWriter::Fork() const {
    PDStateWriter child(pool);
    child.allocations = allocations;
    return child;
}
void PDStateWriter::PlanTensor(const Data &data) {
    Check(!allocations->prepared && data.dataDevice == CUDA && data.dataDeviceIds.size() == 1,
          "invalid export allocation plan");
    const size_t bytes = TensorBytes(data);
    if (!bytes) return;
    const int device = data.dataDeviceIds.front();
    const size_t aligned = (plannedOffset + 255) / 256 * 256;
    if (plannedDevice != device || aligned > plannedCapacity || bytes > plannedCapacity - aligned) {
        plannedCapacity = std::max(bytes, PDTransferArenaBytes);
        plannedDevice = device; plannedOffset = 0;
        plannedArenas.push_back(allocations->plan.size());
        allocations->plan.emplace_back(device, plannedCapacity);
    } else plannedOffset = aligned;
    plannedOffset += bytes;
}
void PDStateWriter::Reserve() {
    Check(!allocations->prepared, "export plan already reserved");
    if (pool) allocations->reserved = pool->AcquireBatch(allocations->plan);
    else for (const auto &item : allocations->plan)
        allocations->reserved.push_back(fastkv::Buffer::Allocate(item.second, item.first));
    allocations->prepared = true;
}
fastkv::Buffer PDStateWriter::Allocate(size_t bytes, int device) {
    if (!pool && !allocations->prepared) return fastkv::Buffer::Allocate(bytes, device);
    const size_t aligned = (arenaOffset + 255) / 256 * 256;
    if (!arena.Size() || arena.Device() != device || aligned > arena.Size() || bytes > arena.Size() - aligned) {
        const size_t capacity = std::max(bytes, PDTransferArenaBytes);
        if (allocations->prepared) {
            Check(nextArena < plannedArenas.size(), "unplanned export allocation");
            auto &reserved = allocations->reserved.at(plannedArenas[nextArena++]);
            Check(reserved.Device() == device && reserved.Size() == capacity, "export plan changed");
            arena = std::move(reserved);
        } else {
            {
                std::lock_guard<std::mutex> lock(allocations->mutex);
                allocations->plan.emplace_back(device, capacity);
                try { pool->CheckCapacity(allocations->plan); }
                catch (...) { allocations->plan.pop_back(); throw; }
            }
            for (;;) {
                try { arena = pool->Acquire(capacity, device); break; }
                catch (const fastkv::CapacityBusy &) { pool->WaitForRelease(); }
            }
        }
        arenaOffset = 0;
    } else arenaOffset = aligned;
    auto buffer = arena.Slice(arenaOffset, bytes);
    arenaOffset += bytes;
    return buffer;
}
void PDStateWriter::AllocateTensor(Data &data) {
    Check(data.dataDevice == CUDA && !data.cudaData && !data.cpuData && data.dims.size() <= 8,
          "direct payload requires an empty compact GPU tensor");
    const uint64_t bytes = TensorBytes(data);
    metadata.U32(uint32_t(data.dataType)); metadata.Ints(data.dims);
    metadata.U32(data.kvCacheFp8Indexer); metadata.U64(bytes);
    if (!bytes) return;
    Check(data.dataDeviceIds.size() == 1, "direct payload requires one owner");
    auto buffer = Allocate(bytes, data.dataDeviceIds.front());
    data.cudaData = buffer.MutableData(); data.cudaDataBorrowed = true;
    data.expansionDims = data.dims; data.expansionBytes = bytes;
    buffers.push_back(std::move(buffer));
}

void PDStateWriter::Tensor(const Data &root) {
    const Data *source = &root;
    if (root.multiDeviceData) {
        Check(root.IsTensorParallelReplicated(), "sharded tensors need an explicit model adapter");
        source = nullptr;
        for (const auto &entry : root.multiDeviceDatas)
            if (entry.second && entry.second->cudaData) {
                source = entry.second;
                break;
            }
        Check(source, "replicated tensor has no owner");
    }
    const Data &data = *source;
    Check(!data.isPagedKVCache && !data.kvMemCache, "paged tensors must be materialized by their owner");
    metadata.U32(uint32_t(data.dataType));
    metadata.Ints(data.dims);
    metadata.U32(data.kvCacheFp8Indexer);
    const uint64_t bytes = TensorBytes(data);
    metadata.U64(bytes);
    if (!bytes)
        return;
    Check(data.dataDevice == CPU || data.dataDevice == CUDA, "unsupported tensor memory");
    int device = -1;
#ifdef USE_CUDA
    if (data.dataDevice == CUDA)
        device = data.dataDeviceIds.empty() ? FastllmCudaGetDevice() : data.dataDeviceIds.front();
#endif
    auto output = Allocate(bytes, device);
    auto sourceLease = std::shared_ptr<void>(const_cast<Data *>(&data), [](void *) {});
    auto input = fastkv::Buffer::Borrow(
        data.dataDevice == CUDA ? data.cudaData : data.cpuData,
        std::max<uint64_t>(bytes, data.expansionBytes),
        data.dataDevice == CUDA ? fastkv::MemoryKind::Cuda : fastkv::MemoryKind::Host, device, sourceLease);
    const size_t rowBytes = data.dataType == FP4_E2M1 ? GetDataBytes(data.dataType, 1, data.dims.back())
                                                   : size_t(data.dims.back()) * data.unitSize;
    Check(rowBytes > 0 && data.strides.size() == data.dims.size(), "invalid tensor strides");
    const uint64_t rows = bytes / rowBytes;
    uint64_t stride = 1;
    bool contiguous = true;
    for (int axis = int(data.dims.size()) - 1; axis >= 0; --axis) {
        contiguous &= data.dims[axis] == 1 || data.strides[axis] == stride;
        stride *= data.dims[axis];
    }
    Check(data.dataType != FP4_E2M1 || contiguous,
          "FP4 tensors must be compacted by their model adapter");
    if (contiguous) {
        output.CopyFrom(input, 0, 0, bytes);
        buffers.push_back(std::move(output));
        return;
    }
    for (uint64_t row = 0; row < rows; ++row) {
        uint64_t index = row, offset = 0;
        for (int axis = int(data.dims.size()) - 2; axis >= 0; --axis) {
            offset += (index % data.dims[axis]) * data.strides[axis];
            index /= data.dims[axis];
        }
        output.CopyFrom(input, offset * data.unitSize, row * rowBytes, rowBytes);
    }
    buffers.push_back(std::move(output));
}
fastkv::Message PDStateWriter::Finish(const std::vector<int> &localDevices) {
    fastkv::Message result{std::move(metadata.bytes), std::move(buffers)};
    if (!pool && result.buffers.size() > 1) result = fastkv::Snapshot(result);
    std::vector<int> devices = localDevices, placement;
    if (devices.empty()) {
        for (const auto &buffer : result.buffers)
            if (buffer.Kind() == fastkv::MemoryKind::Cuda &&
                std::find(devices.begin(), devices.end(), buffer.Device()) == devices.end())
                devices.push_back(buffer.Device());
        std::sort(devices.begin(), devices.end());
    }
    for (const auto &buffer : result.buffers) {
        int rank = -1;
        if (buffer.Kind() == fastkv::MemoryKind::Cuda) {
            auto it = std::find(devices.begin(), devices.end(), buffer.Device());
            Check(it != devices.end(), "payload owner absent from device map");
            rank = it - devices.begin();
        }
        placement.push_back(rank);
    }
    fastkv::Writer envelope;
    envelope.U32(PDWireMagic); envelope.Blob(result.metadata.data(), result.metadata.size());
    envelope.Ints(placement); result.metadata = std::move(envelope.bytes);
    return result;
}
PDStateReader::PDStateReader(const fastkv::Message &value)
    : message(PDStatePayload(value)), metadata(message.metadata) {}
void PDStateReader::Tensor(Data &tensor, bool borrowGpu) {
    const auto type = DataType(metadata.U32());
    Check(type == FLOAT32 || type == FLOAT16 || type == BFLOAT16 || type == INT8 || type == FP8_E4M3 ||
              type == INT32PARAM || type == FP4_E2M1, "unsupported tensor wire type");
    const auto dims = metadata.Ints(8);
    const bool fp8Index = metadata.U32() != 0;
    const uint64_t bytes = metadata.U64();
    Data shape(type); shape.Resize(dims);
    Check(TensorBytes(shape) == bytes, "tensor size mismatch");
    Check(!tensor.cudaData && !tensor.cpuData, "PD destination must be empty");
    tensor.dataType = type; tensor.UpdateUnitSize(); tensor.Resize(dims);
    tensor.kvCacheFp8Indexer = fp8Index;
    if (bytes) {
        Check(next < message.buffers.size() && message.buffers[next].Size() == bytes,
              "tensor payload mismatch");
        const auto &buffer = message.buffers[next++];
        if (buffer.Kind() == fastkv::MemoryKind::Cuda) {
            tensor.dataDevice = CUDA; tensor.dataDeviceIds = {buffer.Device()};
            if (borrowGpu) {
                tensor.cudaData = const_cast<void *>(buffer.Data()); tensor.cudaDataBorrowed = true;
                tensor.expansionDims = dims; tensor.expansionBytes = bytes;
            } else {
#ifdef USE_CUDA
                struct DeviceScope {
                    int previous = FastllmCudaGetDevice();
                    explicit DeviceScope(int device) { FastllmCudaSetDevice(device); }
                    ~DeviceScope() { FastllmCudaSetDevice(previous); }
                } scope(buffer.Device());
#endif
                tensor.Allocate();
                auto owner = std::shared_ptr<void>(&tensor, [](void *) {});
                auto dst = fastkv::Buffer::Borrow(tensor.cudaData, bytes, fastkv::MemoryKind::Cuda,
                                                  buffer.Device(), owner);
                dst.CopyFrom(buffer, 0, 0, bytes);
            }
        } else {
            tensor.dataDevice = CPU; tensor.Allocate();
            buffer.Read(tensor.cpuData, 0, bytes);
        }
    }
    tensor.SetKVCache();
}
void PDStateReader::End() {
    metadata.End();
    Check(next == message.buffers.size(), "unclaimed payload");
}

struct PDPending {
    struct Admission { std::shared_ptr<void> slot, tokens; };
    std::future<fastkv::Message> result;
    fastkv::Message request;
    std::shared_ptr<PDReceiveState> receiver;
    std::string peer, sessionKey;
    std::shared_ptr<Admission> admission;
    std::shared_ptr<void> route;
    std::vector<std::future<fastkv::Message>> probes;
    std::vector<std::shared_ptr<fastkv::Cancellation>> probeCancellations;
    std::vector<fastkv::PrefillRouter::Estimate> estimates;
    std::vector<bool> probed;
    fastkv::Message probeRequest;
    std::vector<uint32_t> probeKinds;
    std::vector<bool> sessionCapable, placementCapable;
    fastkv::SessionDirectory::Record placement;
    std::shared_ptr<fastkv::Cancellation> cancellation = std::make_shared<fastkv::Cancellation>();
    int length = 0;
    size_t reservation = 0;
    bool committed = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PDPending() {
        cancellation->Cancel();
        for (const auto &probe : probeCancellations) if (probe) probe->Cancel();
    }
};
struct PDRuntime {
    struct Workload {
        std::atomic<uint64_t> tokens{0}, restoreBytes{0};
        std::atomic<double> tokensPerSecond{10000};
    };
    struct PrefillRequest {
        std::vector<int> input;
        RequestMetadata metadata;
        std::promise<fastkv::Message> result;
        std::shared_ptr<fastkv::Cancellation> cancellation;
        std::shared_ptr<void> groupAdmission;
        std::unique_ptr<ResponseContext> context;
        fastkv::HostCache::Hit hostHit;
        std::future<fastkv::Message> hostRestore;
        std::chrono::steady_clock::time_point restoreBegin;
        bool restoring = false;
        bool prefixRecorded = false;
        std::chrono::steady_clock::time_point exportQueued;
        size_t reservation = 0;
        std::shared_ptr<Workload> work;
        size_t remaining = 0, restoreBytes = 0;
        void SetRestoreBytes(size_t bytes) {
            if (bytes > restoreBytes) work->restoreBytes.fetch_add(bytes - restoreBytes);
            else work->restoreBytes.fetch_sub(restoreBytes - bytes);
            restoreBytes = bytes;
        }
        void SetRemaining(size_t value) {
            if (value > remaining) work->tokens.fetch_add(value - remaining);
            else work->tokens.fetch_sub(remaining - value);
            remaining = value;
        }
        ~PrefillRequest() { if (work) { work->tokens.fetch_sub(remaining); work->restoreBytes.fetch_sub(restoreBytes); } }
        std::chrono::steady_clock::time_point queued = std::chrono::steady_clock::now();
    };
    std::unique_ptr<fastkv::Service> service;
    std::unique_ptr<fastkv::Service> probes;
    std::unique_ptr<fastkv::PrefillRouter> router;
    std::unique_ptr<fastkv::SessionDirectory> sessions;
    struct Announcement {
        fastkv::SessionDirectory::Record record;
        std::vector<bool> complete;
        std::vector<unsigned> attempts;
        std::vector<std::future<fastkv::Message>> replies;
    };
    std::mutex announceMutex;
    std::map<std::string, Announcement> announcements;
    size_t sessionCapacity = 0;
    std::shared_ptr<Workload> workload = std::make_shared<Workload>();
    std::string role, identity;
    std::vector<std::string> peers;
    std::atomic<bool> stopping{false};
    int chunk = 2048;
    size_t tokenCapacity = 0, totalTokenCapacity = 0;
    std::unique_ptr<fastkv::Budget> slots;
    std::vector<std::unique_ptr<fastkv::Budget>> groupTokens;
    std::shared_ptr<void> ReserveGroup(basellm &model, ResponseContext &context, size_t required) {
        std::vector<size_t> available;
        for (const auto &budget : groupTokens) available.push_back(tokenCapacity - budget->Used());
        if (!model.SelectPDRequestGroup(context, available, required))
            throw fastkv::CapacityBusy("PD request group has no capacity");
        const int group = model.PDRequestGroup(context);
        Check(group >= 0 && size_t(group) < groupTokens.size(), "invalid request group");
        return groupTokens[group]->Reserve(required);
    }
    std::unique_ptr<fastkv::PrefixCache> cache;
    std::unique_ptr<fastkv::HostCache> hostCache;
    std::unique_ptr<fastkv::BufferPool> exportPool;
    bool residentPrefix = false;
    std::mutex prefillMutex, prefillStopMutex;
    std::condition_variable prefillCV;
    std::deque<std::shared_ptr<PrefillRequest>> waiting;
    std::thread prefillThread;
    int batchRequests = 1, batchTokens = 1, targetRequests = 1, waitMs = 0;

    ~PDRuntime() {
        StopPrefill();
        if (probes) probes->Stop();
        if (service) service->Stop(); // callbacks capture this runtime
    }
    void StopPrefill() {
        if (exportPool) exportPool->Clear();
        std::lock_guard<std::mutex> stoppingLock(prefillStopMutex);
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            stopping = true;
        }
        prefillCV.notify_all();
        if (prefillThread.joinable()) prefillThread.join();
    }
    size_t CachedTokens(basellm &model, const std::vector<int> &input) {
        return residentPrefix ? model.PreviewPDResidentPrefix(input) :
               cache ? cache->ProbeTokens(identity, input) : 0;
    }
    double HostRestoreBytesPerSecond() const {
        const auto stats = hostCache ? hostCache->Stats() : fastkv::HostCache::Statistics{};
        return stats.restoreMicroseconds ?
            std::max(1.0, double(stats.restoredBytes) * 1e6 / stats.restoreMicroseconds) : 1e10;
    }
    std::future<fastkv::Message> Prefill(std::vector<int> input,
                                       std::shared_ptr<fastkv::Cancellation> cancellation, RequestMetadata metadata = {}) {
        auto request = std::make_shared<PrefillRequest>();
        request->input = std::move(input);
        request->metadata = std::move(metadata);
        request->cancellation = std::move(cancellation);
        request->work = workload;
        // Pin an L2 hit at admission: its saved work is real, not a stale hint.
        if (hostCache) request->hostHit = hostCache->Find(identity, request->input);
        request->SetRemaining(request->input.size() - request->hostHit.Tokens());
        request->SetRestoreBytes(request->hostHit.Bytes());
        auto result = request->result.get_future();
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            Check(!stopping, "prefill is stopping");
            waiting.push_back(request);
        }
        prefillCV.notify_one();
        return result;
    }
    void RunPrefill(basellm &model);
    void FlushAnnouncements() {
        if (!probes) return;
        std::lock_guard<std::mutex> lock(announceMutex);
        unsigned submitted = 0;
        for (auto it = announcements.begin(); it != announcements.end();) {
            auto &a = it->second;
            for (size_t i = 0; i < peers.size(); ++i) {
                auto &reply = a.replies[i];
                if (reply.valid()) {
                    if (reply.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
                    try {
                        auto message = reply.get(); fastkv::Reader reader(message.metadata);
                        Check(message.buffers.empty() && reader.U32() == 7, "invalid placement acknowledgment");
                        reader.End(); a.complete[i] = true;
                    } catch (const std::exception &) { if (a.attempts[i] >= 3) a.complete[i] = true; }
                }
                if (a.complete[i] || submitted >= 16) continue;
                fastkv::Writer writer; writer.U32(7); writer.String(identity); writer.String(it->first);
                writer.U64(a.record.version); writer.String(a.record.owner);
                writer.U32(peers[i] == a.record.owner);
                fastkv::Message message{std::move(writer.bytes), {}};
                if (probes->TrySubmit(peers[i], message, reply)) { ++a.attempts[i]; ++submitted; }
            }
            if (std::find(a.complete.begin(), a.complete.end(), false) == a.complete.end())
                it = announcements.erase(it);
            else ++it;
        }
    }
    void CommitPlacement(const PDPending &pending) {
        if (pending.sessionKey.empty() || !pending.placement.version) return;
        if (!sessions->Update(pending.sessionKey, pending.placement)) return;
        std::lock_guard<std::mutex> lock(announceMutex);
        if (announcements.size() >= sessionCapacity && !announcements.count(pending.sessionKey)) return;
        auto &a = announcements[pending.sessionKey]; a = {};
        a.record = pending.placement;
        a.complete.resize(peers.size()); a.attempts.resize(peers.size()); a.replies.resize(peers.size());
        for (size_t i = 0; i < peers.size(); ++i) a.complete[i] = !pending.placementCapable[i];
    }
    bool SelectPeer(PDPending &pending, bool verbose) {
        if (!pending.peer.empty()) return true;
        if (!pending.probes.empty()) {
            bool submissionBusy = false;
            for (size_t i = 0; i < peers.size(); ++i) {
                auto &future = pending.probes[i];
                if (future.valid()) {
                    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
                    try {
                        const auto reply = future.get();
                        Check(reply.buffers.empty(), "routing reply contains payload");
                        fastkv::Reader reader(reply.metadata);
                        const auto kind = pending.probeKinds[i];
                        Check(reader.U32() == kind, "invalid routing reply");
                        auto &e = pending.estimates[i];
                        e.cachedTokens = reader.U64(); e.pendingTokens = reader.U64();
                        e.tokensPerSecond = reader.U64();
                        if (kind >= 6) {
                            fastkv::SessionDirectory::Record record;
                            record.version = reader.U64(); record.owner = reader.String(512);
                            if (record.version && std::find(peers.begin(), peers.end(), record.owner) != peers.end())
                                sessions->Update(pending.sessionKey, record);
                        }
                        if (kind == 8) {
                            e.hostCachedTokens = reader.U64();
                            const auto bytes = reader.U64(), queuedBytes = reader.U64(), rate = reader.U64();
                            Check(rate > 0 && e.hostCachedTokens <= size_t(pending.length), "invalid host cache estimate");
                            e.restoreSeconds = double(bytes) / rate; e.pendingRestoreSeconds = double(queuedBytes) / rate;
                        }
                        reader.End();
                        Check(e.cachedTokens <= size_t(pending.length) && e.tokensPerSecond > 0,
                              "invalid routing estimate");
                        e.available = true; pending.probed[i] = true;
                        pending.sessionCapable[i] = kind >= 4;
                        pending.placementCapable[i] = kind >= 6;
                    } catch (const std::exception &) {
                        if (pending.probeKinds[i] > 2)
                            pending.probeKinds[i] -= 2; // Retry metadata only, never model work.
                        else { pending.probed[i] = true; pending.estimates[i].available = false; }
                    }
                }
                if (pending.probed[i] || submissionBusy) continue;
                // Retain an unsubmitted message across admission retries;
                // never repeatedly copy a long prompt while control is full.
                if (pending.probeRequest.metadata.empty() || pending.probeRequest.metadata[0] != pending.probeKinds[i]) {
                    pending.probeRequest.metadata = pending.request.metadata;
                    pending.probeRequest.metadata[0] = pending.probeKinds[i];
                    if (pending.probeKinds[i] >= 6) {
                        fastkv::Writer suffix; suffix.String(pending.sessionKey);
                        pending.probeRequest.metadata.insert(pending.probeRequest.metadata.end(), suffix.bytes.begin(), suffix.bytes.end());
                    }
                }
                // A failed capability query cancels its transport operation.
                // Keep it isolated from model work and other peers; cancelling
                // the parent still cancels every outstanding query explicitly.
                auto cancellation = std::make_shared<fastkv::Cancellation>();
                if (probes->TrySubmit(peers[i], pending.probeRequest, future, cancellation, pending.admission))
                    pending.probeCancellations[i] = std::move(cancellation);
                else submissionBusy = true;
            }
            if (std::find(pending.probed.begin(), pending.probed.end(), false) != pending.probed.end()) return false;
        }
        bool available = false;
        for (const auto &e : pending.estimates) available |= e.available;
        // Older producers have no metadata query. Keep a bounded least-work
        // fallback; a routing hint never changes the model/state protocol.
        if (!available) for (auto &e : pending.estimates) e = {};
        // All D ranks discover committed placement from producer probes. Only
        // new sessions or expired metadata fall back to rendezvous hashing.
        size_t preferred = SIZE_MAX;
        uint64_t highest = 0;
        if (!pending.sessionKey.empty()) for (size_t i = 0; i < peers.size(); ++i) {
            uint64_t hash = 14695981039346656037ULL;
            for (unsigned char c : pending.sessionKey + std::string(1, '\0') + peers[i])
                hash = (hash ^ c) * 1099511628211ULL;
            if (preferred == SIZE_MAX || hash > highest) { preferred = i; highest = hash; }
        }
        auto placement = sessions->Get(pending.sessionKey);
        auto owner = std::find(peers.begin(), peers.end(), placement.owner);
        const bool owned = placement.version && owner != peers.end();
        if (owned) preferred = owner - peers.begin();
        auto chosen = router->Reserve(pending.length, pending.estimates, preferred, owned);
        if (!pending.sessionKey.empty() && pending.placementCapable[chosen.peer]) {
            Check(placement.version < UINT64_MAX, "session placement version exhausted");
            pending.placement = placement.owner == peers[chosen.peer] ? placement :
                fastkv::SessionDirectory::Record{placement.version + 1, peers[chosen.peer]};
        }
        if (!pending.sessionKey.empty() && pending.sessionCapable[chosen.peer]) {
            pending.request.metadata[0] = 3;
            fastkv::Writer suffix; suffix.String(pending.sessionKey);
            pending.request.metadata.insert(pending.request.metadata.end(), suffix.bytes.begin(), suffix.bytes.end());
        }
        pending.peer = peers[chosen.peer]; pending.route = std::move(chosen.lease);
        if (verbose && peers.size() > 1) {
            const auto &e = pending.estimates[chosen.peer];
            std::cout << "[FastKV PD route] peer=" << pending.peer << " cached_tokens=" << e.cachedTokens
                      << " pending_tokens=" << e.pendingTokens << " estimated_new_tokens=" << chosen.uncachedTokens
                      << " probe_fallback=" << !available << std::endl;
        }
        pending.probes.clear(); pending.probeCancellations.clear(); pending.probed.clear();
        pending.estimates.clear(); pending.probeRequest = {};
        return true;
    }
};

void PDRuntime::RunPrefill(basellm &model) {
    using Request = std::shared_ptr<PrefillRequest>;
    // Raw RPCs own no cache. Prepared contexts own the same immutable page
    // references and private companion state as active contexts, but have not
    // entered forward yet. Both use one token budget and one retirement path.
    std::deque<Request> active, prepared, exporting;
    size_t reserved = 0, preparedTokens = 0;
    const size_t prepareBudget = totalTokenCapacity / 2;
    const size_t page = residentPrefix ? std::max(1, GetPageLen()) : 1;
    const size_t contextLimit = std::max(batchRequests, model.maxBatch > 0 ? model.maxBatch : 1);
    auto startHostRestore = [&](const Request &request) {
        auto placement = PDReceivePlacement(request->hostHit.Metadata(), request->hostHit.Payloads(), model.PDRequestDevices(*request->context));
        if (placement.empty()) placement.assign(request->hostHit.Payloads(), -1);
        return hostCache->RestoreAsync(request->hostHit, std::move(placement), request->hostRestore,
                                       request->cancellation);
    };
    auto retire = [&](const Request &request) {
        request->SetRemaining(0); request->SetRestoreBytes(0);
        reserved -= request->reservation;
        request->reservation = 0;
        auto context = std::move(request->context);
        if (context) {
            context->isAbort = true;
            model.OnResponseContextRemoved(context.get());
        }
        context.reset();
        request->groupAdmission.reset();
    };
    const auto cancelled = std::make_exception_ptr(std::runtime_error("FastKV PD: request cancelled"));
    auto complete = [&](const Request &request, fastkv::Message *restored) {
        const auto exportBegin = std::chrono::steady_clock::now();
        if (residentPrefix && !request->prefixRecorded) {
            request->context->TryRecordPagedCache(&model);
            request->prefixRecorded = true;
        }
        const auto recorded = std::chrono::steady_clock::now();
        auto reply = restored ? std::move(*restored) : model.ExportPDState(*request->context);
        const auto exported = std::chrono::steady_clock::now();
        if (cache) {
            try { cache->Publish(identity, request->input, reply); }
            catch (const std::exception &error) {
                if (model.verbose) std::cerr << "[FastKV PD] prefix not retained: " << error.what() << std::endl;
            }
        }
        if (hostCache && !restored) {
            try {
                hostCache->PublishAsync(identity, request->input,
                    model.DescribePDCache(*request->context, reply), request->hostHit);
            } catch (const std::exception &error) {
                std::cerr << "[FastKV L2] backup skipped: " << error.what() << std::endl;
            }
        }
        const auto published = std::chrono::steady_clock::now();
        if (hostCache && model.verbose) {
            const auto stats = hostCache->Stats();
            std::cout << "[FastKV L2 cache] entries=" << stats.entries << " bytes=" << stats.bytes
                      << " queued=" << stats.queued << " skipped=" << stats.skipped << " failures=" << stats.failures
                      << " d2h_bytes=" << stats.copiedToHost << " h2d_bytes=" << stats.restoredBytes
                      << " reused_bytes=" << stats.reusedBytes << std::endl;
        }
        retire(request);
        if (model.verbose) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::cout << "[FastKV PD timing] record_ms=" << ms(exportBegin, recorded)
                      << " export_ms=" << ms(recorded, exported)
                      << " export_wait_ms=" << (request->exportQueued.time_since_epoch().count() ?
                          ms(request->exportQueued, exportBegin) : 0.0)
                      << " publish_ms=" << ms(exported, published)
                      << " retire_ms=" << ms(published, std::chrono::steady_clock::now())
                      << " kv_d2h_bytes=" << fastkv::Buffer::Traffic().deviceToHost
                      << " kv_h2d_bytes=" << fastkv::Buffer::Traffic().hostToDevice << std::endl;
        }
        request->result.set_value(std::move(reply));
    };
    try {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(prefillMutex);
                if (active.empty() && prepared.empty() && exporting.empty()) {
                    prefillCV.wait(lock, [&] { return stopping || !waiting.empty(); });
                    if (!stopping && waitMs > 0) {
                        const auto deadline = waiting.front()->queued + std::chrono::milliseconds(waitMs);
                        prefillCV.wait_until(lock, deadline, [&] {
                            if (stopping || int(waiting.size()) >= targetRequests) return true;
                            size_t tokens = 0;
                            for (const auto &r : waiting) tokens += std::min(size_t(chunk), r->input.size());
                            return tokens >= size_t(batchTokens);
                        });
                    }
                }
                if (stopping) break;
                for (auto it = waiting.begin(); it != waiting.end();) {
                    if ((*it)->cancellation->Cancelled()) {
                        (*it)->result.set_exception(cancelled);
                        it = waiting.erase(it);
                    } else ++it;
                }
            }
            // Only the model execution lane touches contexts and GPU state.
            // RPC handlers return futures immediately; network workers never
            // wait for the model or hold forwardLocker.
            std::unique_lock<std::mutex> lock(model.forwardLocker);
            // Completed contexts retain their pages and admission while waiting
            // for a whole-message export reservation. Other requests can still
            // forward; no owner worker waits on a transport buffer lease.
            for (auto it = exporting.begin(); it != exporting.end();) {
                const auto request = *it;
                try {
                    if (request->cancellation->Cancelled()) std::rethrow_exception(cancelled);
                    complete(request, nullptr);
                } catch (const fastkv::CapacityBusy &) { ++it; continue; }
                catch (...) {
                    auto error = std::current_exception();
                    retire(request); request->result.set_exception(error);
                }
                it = exporting.erase(it);
            }
            // Cancellation touches model state only on its execution lane,
            // between completed forward calls. The service drains the future.
            for (auto it = active.begin(); it != active.end();) {
                if ((*it)->cancellation->Cancelled()) {
                    retire(*it);
                    (*it)->result.set_exception(cancelled);
                    it = active.erase(it);
                } else ++it;
            }
            for (auto it = prepared.begin(); it != prepared.end();) {
                if ((*it)->cancellation->Cancelled()) {
                    preparedTokens -= (*it)->reservation;
                    retire(*it);
                    (*it)->result.set_exception(cancelled);
                    it = prepared.erase(it);
                } else ++it;
            }
            for (auto it = prepared.begin(); it != prepared.end();) {
                const auto request = *it;
                try {
                    if (request->restoring) {
                        if (!request->hostRestore.valid()) {
                            if (!startHostRestore(request)) { ++it; continue; }
                        }
                        if (request->hostRestore.wait_for(std::chrono::seconds(0)) != std::future_status::ready) { ++it; continue; }
                        auto state = request->hostRestore.get();
                        auto &context = *request->context;
                        context.allTokens.resize(request->hostHit.Tokens()); context.allTokens.push_back(0);
                        model.ImportPDState(context, state);
                        context.preTokens = request->hostHit.Tokens(); context.allTokens = request->input;
                        request->restoring = false; request->SetRestoreBytes(0);
                        request->SetRemaining(request->input.size() - context.preTokens);
                        if (model.verbose) {
                            const auto stats = hostCache->Stats();
                            std::cout << "[FastKV PD] prefix_hit=" << context.preTokens << " prompt=" << request->input.size()
                                      << " cache_tier=l2 restore_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - request->restoreBegin).count()
                                      << " bytes=" << request->hostHit.Bytes() << " cache_bytes=" << stats.bytes
                                      << " entries=" << stats.entries << " d2h_bytes=" << stats.copiedToHost
                                      << " reused_bytes=" << stats.reusedBytes << std::endl;
                        }
                        if (size_t(context.preTokens) == request->input.size()) {
                            const auto reservation = request->reservation;
                            complete(request, &state);
                            preparedTokens -= reservation;
                            it = prepared.erase(it);
                            continue;
                        }
                    }
                    if (int(active.size()) < batchRequests) {
                        preparedTokens -= request->reservation;
                        active.push_back(request); it = prepared.erase(it);
                    } else ++it;
                } catch (...) {
                    const auto error = std::current_exception();
                    // Never import partially restored state or silently continue
                    // from a damaged checkpoint. Normal request cleanup owns it.
                    preparedTokens -= request->reservation;
                    retire(request); request->result.set_exception(error); it = prepared.erase(it);
                }
            }
            size_t scanned = 0;
            { std::lock_guard<std::mutex> queueLock(prefillMutex); scanned = waiting.size(); }
            for (; scanned > 0 && active.size() + prepared.size() + exporting.size() < contextLimit; --scanned) {
                const bool lookahead = int(active.size()) >= batchRequests;
                if (lookahead && !residentPrefix) break;
                Request request;
                {
                    std::lock_guard<std::mutex> lock(prefillMutex);
                    if (stopping || waiting.empty()) break;
                    const size_t count = std::min(tokenCapacity,
                        ((waiting.front()->input.size() + page - 1) / page) * page);
                    // Reserve the FULL final prompt, not merely its cached
                    // portion. This leaves room to finish every admitted
                    // request and bounds queued retention independently.
                    if (count > totalTokenCapacity - reserved ||
                        (lookahead && (preparedTokens >= prepareBudget || count > prepareBudget - preparedTokens))) break;
                    request = std::move(waiting.front()); waiting.pop_front();
                    reserved += count;
                    request->reservation = count;
                }
                try {
                    if (request->cancellation->Cancelled()) std::rethrow_exception(cancelled);
                    const auto restoreBegin = std::chrono::steady_clock::now();
                    fastkv::PrefixCache::Hit hit;
                    if (cache) hit = cache->Find(identity, request->input, request->input.size());
                    if (hit.state && hit.tokens == request->input.size()) {
                        retire(request);
                        request->result.set_value(*hit.state);
                        continue;
                    }
                    request->context.reset(new ResponseContext);
                    auto &context = *request->context;
                    context.Init(model.block_cnt, model.dataType, model.kvCacheDataType);
                    context.requestMetadata = request->metadata;
                    context.currentTokens = context.allTokens = request->input;
                    context.inputTokens = request->input.size();
                    context.generationConfig.top_k = 1;
                    context.generationConfig.output_token_limit = 1;
                    context.generationConfig.input_token_length = request->input.size() + 1;
                    model.OnResponseContextCreated(&context);
                    try { request->groupAdmission = ReserveGroup(model, context, request->reservation); }
                    catch (const fastkv::CapacityBusy &) {
                        retire(request);
                        request->SetRemaining(request->input.size());
                        std::lock_guard<std::mutex> queueLock(prefillMutex);
                        waiting.push_back(request);
                        continue;
                    }
                    if (hostCache) {
                        auto latest = hostCache->Find(identity, request->input);
                        if (latest.Tokens() > request->hostHit.Tokens()) request->hostHit = std::move(latest);
                    }
                    const size_t residentTokens = request->hostHit && residentPrefix ? size_t(model.PreviewPDResidentPrefix(request->input)) : hit.tokens;
                    // A few extra cached tail tokens do not justify restoring
                    // an entire host checkpoint when L1 already covers the prefix.
                    const bool useHost = request->hostHit.Tokens() > residentTokens &&
                        double(request->hostHit.Tokens() - residentTokens) / workload->tokensPerSecond.load() >
                        double(request->hostHit.Bytes()) / HostRestoreBytesPerSecond();
                    if (useHost) {
                        request->restoring = true; request->restoreBegin = restoreBegin;
                        request->SetRestoreBytes(request->hostHit.Bytes());
                        request->SetRemaining(request->input.size() - request->hostHit.Tokens());
                        // Start the private copy before the next forward. A busy
                        // cache retries from prepared; import remains on this lane.
                        startHostRestore(request);
                        prepared.push_back(request); preparedTokens += request->reservation;
                        continue;
                    }
                    request->SetRestoreBytes(0);
                    request->hostHit = {}; // only an actually restored parent may supply inherited bytes
                    if (residentPrefix) {
                        context.preTokens = model.RestorePDResidentPrefix(&context);
                        hit.tokens = context.preTokens;
                    }
                    if (hit.state) {
                        context.allTokens.resize(hit.tokens);
                        context.allTokens.push_back(0); // adapter validates the N-1 boundary
                        model.ImportPDState(context, *hit.state);
                        context.preTokens = hit.tokens;
                        context.allTokens = request->input;
                    }
                    request->SetRemaining(request->input.size() - context.preTokens);
                    if (model.verbose) {
                        const auto stats = cache ? cache->Stats() : fastkv::PrefixCache::Statistics{};
                        std::cout << "[FastKV PD] prefix_hit=" << hit.tokens << " prompt=" << request->input.size()
                                  << " cache_entries=" << stats.entries << " cache_bytes=" << stats.bytes
                                  << " cache_logical_bytes=" << stats.logicalBytes
                                  << " cache_evictions=" << stats.evictions
                                  << " resident_gpu=" << residentPrefix << std::endl;
                        std::cout << "[FastKV PD timing] restore_ms="
                                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - restoreBegin).count()
                                  << std::endl;
                    }
                    if (lookahead) {
                        prepared.push_back(request);
                        preparedTokens += request->reservation;
                    } else active.push_back(request);
                } catch (...) {
                    auto error = std::current_exception();
                    retire(request);
                    request->result.set_exception(error);
                }
            }
            if (active.empty()) {
                lock.unlock();
                std::unique_lock<std::mutex> queueLock(prefillMutex);
                prefillCV.wait_for(queueLock, std::chrono::milliseconds(1));
                continue;
            }

            std::vector<Request> selected;
            std::vector<int> lengths;
            // Placement is fixed by the model's cache/session affinity. Plan
            // across those owners, just like the ordinary prefill scheduler;
            // FIFO RPC arrival order need not be balanced across DP groups.
            const bool balanceDP = model.prefillScheduling.balanceDP;
            const int groupBudget = std::max(1, batchTokens / std::max(1, model.GetPrefillBatchGroups()));
            PrefillBatchQueue queue;
            std::map<int, int> groupTokens;
            for (int i = 0; i < (int)active.size(); ++i)
                queue.Push(i, balanceDP ? model.GetPrefillBatchGroup(active[i]->context.get()) : 0);
            int budget = batchTokens;
            for (int index; budget && (index = queue.Pop()) >= 0;) {
                const auto &request = active[index];
                const int group = balanceDP ? model.GetPrefillBatchGroup(request->context.get()) : 0;
                const int available = balanceDP ? groupBudget - groupTokens[group] : budget;
                const int count = std::min({chunk, budget, available,
                    int(request->input.size()) - request->context->preTokens});
                if (count <= 0) continue;
                selected.push_back(request); lengths.push_back(count); budget -= count;
                groupTokens[group] += count;
                queue.AddTokens(count);
            }
            const int batch = selected.size();
            std::vector<Data> ids(batch), masks(batch), positions(batch);
            std::vector<Data *> maskRefs, positionRefs;
            std::vector<std::pair<Data *, Data *>> past;
            std::vector<GenerationConfig> configs;
            std::vector<float> packed;
            if (batch > 1)
                packed.reserve(batchTokens - budget);
            for (int i = 0; i < batch; ++i) {
                auto &context = *selected[i]->context;
                const int start = context.preTokens, end = start + lengths[i];
                std::vector<std::vector<float>> input(1);
                input[0].assign(selected[i]->input.begin() + start, selected[i]->input.begin() + end);
                model.FillLLMInputs(input, {{"promptLen", end}, {"index", 0}, {"add_special_tokens", 0}},
                                    ids[i], masks[i], positions[i]);
                ToDataType(masks[i], model.dataType);
                // Preserve the model's input builder rather than assuming its
                // token IDs or positions equal the raw transport payload.
                ids[i].ToDevice(CPU);
                Check(ids[i].dataType == FLOAT32 && ids[i].Count(0) == size_t(lengths[i]),
                      "prefill input builder changed the token count or type");
                if (batch > 1) {
                    auto *data = reinterpret_cast<const float *>(ids[i].cpuData);
                    packed.insert(packed.end(), data, data + lengths[i]);
                }
                maskRefs.push_back(&masks[i]); positionRefs.push_back(&positions[i]);
                configs.push_back(context.generationConfig);
                for (auto &kv : context.pastKeyValues) past.emplace_back(&kv.first, &kv.second);
            }
            struct Intermediate {
                basellm &model;
                bool previous;
                explicit Intermediate(basellm &m) : model(m), previous(m.isIntermediateChunkedPrefill) {
                    model.isIntermediateChunkedPrefill = true;
                }
                ~Intermediate() { model.isIntermediateChunkedPrefill = previous; }
            } intermediate(model);
            const auto begin = std::chrono::steady_clock::now();
            LastTokensManager last(batch, 0);
            if (batch == 1) {
                model.Forward(ids[0], masks[0], positions[0], selected[0]->context->pastKeyValues,
                              configs[0], last, nullptr);
            } else {
                Data inputIds(FLOAT32, {1, int(packed.size())}, packed);
                if (model.use_new_engine) {
                    if (model.CanUseGPUForward())
                        model.ForwardGPU(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
                    else model.ForwardV2(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
                } else model.ForwardBatch(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (seconds > 0) workload->tokensPerSecond.store(
                0.9 * workload->tokensPerSecond.load() + 0.1 * (batchTokens - budget) / seconds);
            if (model.verbose) {
                size_t queued;
                { std::lock_guard<std::mutex> lock(prefillMutex); queued = waiting.size(); }
                std::cout << "[FastKV PD] prefill_batch=" << batch << " tokens=" << batchTokens - budget
                          << " forward_ms=" << seconds * 1000
                          << " active=" << active.size() << " prepared=" << prepared.size()
                          << " exporting=" << exporting.size()
                          << " queued=" << queued << " reserved_tokens=" << reserved
                          << " prepared_tokens=" << preparedTokens << std::endl;
            }
            for (int i = 0; i < batch; ++i) {
                auto request = selected[i];
                active.erase(std::find(active.begin(), active.end(), request));
                request->context->preTokens += lengths[i];
                request->SetRemaining(request->input.size() - request->context->preTokens);
                if (request->cancellation->Cancelled()) {
                    retire(request);
                    request->result.set_exception(cancelled);
                    continue;
                }
                if (size_t(request->context->preTokens) != request->input.size()) {
                    active.push_back(request); // round-robin chunks, no long-request monopolization
                    continue;
                }
                try {
                    complete(request, nullptr);
                } catch (const fastkv::CapacityBusy &) {
                    request->exportQueued = std::chrono::steady_clock::now();
                    exporting.push_back(request);
                } catch (...) {
                    auto error = std::current_exception();
                    retire(request);
                    request->result.set_exception(error);
                }
            }
        }
        throw std::runtime_error("FastKV PD: prefill cancelled by shutdown");
    } catch (...) {
        const auto error = std::current_exception();
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            stopping = true;
            for (const auto &request : waiting) request->result.set_exception(error);
            waiting.clear();
        }
        std::lock_guard<std::mutex> lock(model.forwardLocker);
        for (const auto &request : active) {
            retire(request);
            request->result.set_exception(error);
        }
        for (const auto &request : prepared) {
            retire(request);
            request->result.set_exception(error);
        }
        for (const auto &request : exporting) {
            retire(request);
            request->result.set_exception(error);
        }
    }
}

std::string basellm::PDLayout() const {
    throw std::invalid_argument("FastKV PD: this model has no state adapter");
}
fastkv::Message basellm::ExportPDState(ResponseContext &) { throw std::logic_error("missing PD adapter"); }
fastkv::CacheSnapshot basellm::DescribePDCache(ResponseContext &, const fastkv::Message &state) {
    fastkv::CacheSnapshot result; result.state = state;
    for (const auto &buffer : state.buffers) {
        fastkv::CachePayload payload;
        if (buffer.Size()) payload.regions.push_back({0, buffer.Size(), {}});
        result.payloads.push_back(std::move(payload));
    }
    return result;
}
void basellm::ImportPDState(ResponseContext &, const fastkv::Message &) {
    throw std::logic_error("missing PD adapter");
}

std::vector<int> basellm::PDDevices() const {
#ifdef USE_CUDA
    return {FastllmCudaGetDevice()};
#else
    return {};
#endif
}
fastkv::BufferPool *basellm::PDExportPool() const {
    return pdRuntime ? pdRuntime->exportPool.get() : nullptr;
}

bool basellm::IsPDDecode() const { return pdRuntime && pdRuntime->role == "decode"; }
void basellm::StopPD() {
    if (pdRuntime) {
        pdRuntime->StopPrefill();
        if (pdRuntime->probes) pdRuntime->probes->Stop();
        pdRuntime->service->Stop();
    }
}

void basellm::ConfigurePD(const json11::Json &config) {
    Check(!mainLoop && !pdRuntime, "configure PD once, after warmup and before serving");
    auto runtime = std::make_shared<PDRuntime>();
    runtime->role = config["role"].string_value();
    Check(runtime->role == "prefill" || runtime->role == "decode", "role must be prefill or decode");
    Check(!config["model_id"].string_value().empty(),
          "model_id must identify the same checkpoint on P and D");
    runtime->identity = config["model_id"].string_value() + ":gpu-state-v2:" + PDLayout();
    Check(!UseModelSpecificScheduler(), "PD requires a supported common scheduler");
    Check(!kvMemConfig.enabled || (runtime->role == "decode" && SupportsPDKvMemDecode()),
          "KVMem PD requires a supported decode adapter; prefill must use GPU KV");
    const int concurrent = maxBatch > 0 ? maxBatch : 1;
    runtime->sessionCapacity = size_t(concurrent) * 4;
    runtime->sessions.reset(new fastkv::SessionDirectory(runtime->sessionCapacity));
    runtime->tokenCapacity =
        GetMaxTokens() > 0 ? size_t(GetMaxTokens()) : size_t(max_positions) * concurrent;
    if (tokensLimit > 0)
        runtime->tokenCapacity = std::min(runtime->tokenCapacity, size_t(tokensLimit));
    const int groups = PDGroupCount();
    Check(groups > 0 && runtime->tokenCapacity > 0 &&
          size_t(groups) <= SIZE_MAX / runtime->tokenCapacity, "invalid PD group capacity");
    runtime->totalTokenCapacity = runtime->tokenCapacity * groups;
    for (int group = 0; group < groups; ++group)
        runtime->groupTokens.emplace_back(new fastkv::Budget(runtime->tokenCapacity, "PD group token budget"));
    fastkv::Options options;
    options.workers = std::max(4, std::min(concurrent, 16));
    if (!config["transfer_workers"].is_null()) {
        const double workers = config["transfer_workers"].number_value();
        Check(config["transfer_workers"].is_number() && workers >= 1 && workers <= 64 &&
                  workers == int(workers), "transfer_workers must be an integer in [1, 64]");
        options.workers = int(workers);
    }
    options.maxActiveRequests = concurrent;
    // Every admitted D request can wait on one transfer. P queue capacity is
    // supplied by the router from the D groups that can submit work to it.
    options.queueDepth = std::max(options.queueDepth, concurrent);
    if (config["queue_depth"].is_number()) {
        const double depth = config["queue_depth"].number_value();
        Check(depth >= 1 && depth <= 1048576 && depth == int(depth), "invalid queue depth");
        options.queueDepth = std::max(options.queueDepth, int(depth));
    }
    const std::string transport = config["transport"].string_value();
    Check(transport == "tcp" || transport == "rdma" || transport == "auto", "unknown transport");
    options.transport = transport == "rdma"   ? fastkv::Transport::Rdma
                        : transport == "auto" ? fastkv::Transport::Auto
                                              : fastkv::Transport::Tcp;
    for (const auto &device : config["rdma_devices"].array_items())
        options.rdmaDevices.push_back(device.string_value());
    auto devices = PDDevices();
#ifdef USE_CUDA
    if (devices.empty()) devices.push_back(FastllmCudaGetDevice());
#endif
    options.receivePlacement = [devices](const fastkv::Bytes &metadata, const std::vector<size_t> &sizes) {
        return PDReceivePlacement(metadata, sizes.size(), devices);
    };
    if (config["max_inflight_bytes"].is_number()) {
        double limit = config["max_inflight_bytes"].number_value();
        Check(limit >= (1 << 20) && limit <= double(size_t(1) << 40), "invalid inflight byte limit");
        options.maxInflightBytes = size_t(limit);
        options.maxPayloadBytes = size_t(limit);
    }
#ifdef USE_CUDA
    size_t stagingCapacity = 0;
    for (int device : devices) {
        uint64_t freeBytes = 0, totalBytes = 0;
        if (FastllmCudaGetMemoryInfo(device, freeBytes, totalBytes)) {
            // Leave half the unallocated memory for request/model scratch.
            const size_t limit = freeBytes / 2;
            options.deviceBufferBytes[device] = limit;
            stagingCapacity += limit;
        }
    }
    // Direct KVMem receive uses its reserved host pages; only SWA staging is GPU-backed.
    if (stagingCapacity && !kvMemConfig.enabled)
        options.maxInflightBytes = std::min(options.maxInflightBytes, stagingCapacity);
#endif
    runtime->service.reset(new fastkv::Service(options));
    runtime->exportPool.reset(new fastkv::BufferPool(options.maxInflightBytes, options.deviceBufferBytes));
    runtime->residentPrefix = HasPDResidentPrefixCache();
    for (const auto &peer : config["prefill"].array_items()) {
        Check(peer.is_string() && !peer.string_value().empty(), "invalid prefill endpoint");
        runtime->peers.push_back(peer.string_value());
    }
    if (runtime->role == "decode") {
        Check(!runtime->peers.empty(), "decode needs a prefill execution group");
        runtime->router.reset(new fastkv::PrefillRouter(runtime->peers.size()));
        {
            auto control = options;
            control.workers = 2; control.queueDepth = 16; control.timeoutMs = 2000;
            runtime->probes.reset(new fastkv::Service(control));
        }
        runtime->slots.reset(new fastkv::Budget(concurrent, "decode request slots"));
    } else {
        if (config["chunk_tokens"].is_number())
            runtime->chunk = config["chunk_tokens"].int_value();
        Check(runtime->chunk > 0 && runtime->chunk <= max_positions, "invalid prefill chunk size");
        const double cacheBytes = config["cache_bytes"].number_value();
        Check(cacheBytes >= 0 && cacheBytes <= double(size_t(1) << 40), "invalid prefix cache byte limit");
        if (cacheBytes > 0) {
            fastkv::HostCache::Options host;
            host.bytes = size_t(cacheBytes);
            host.inflightBytes = options.maxInflightBytes / 2;
            for (const auto &limit : options.deviceBufferBytes) host.restoreDeviceBytes[limit.first] = limit.second / 2;
            if (!config["cache_workers"].is_null()) {
                const double workers = config["cache_workers"].number_value();
                Check(workers >= 1 && workers <= 64 && workers == int(workers), "invalid cache_workers");
                host.workers = unsigned(workers);
            }
            runtime->hostCache.reset(new fastkv::HostCache(host));
        }
        if (!runtime->residentPrefix) {
            const double deviceBytes = config["device_cache_bytes"].is_number() ?
                config["device_cache_bytes"].number_value() : double(size_t(1) << 30);
            Check(deviceBytes >= 0 && deviceBytes <= double(size_t(1) << 40), "invalid device cache byte limit");
            if (deviceBytes > 0) runtime->cache.reset(new fastkv::PrefixCache(size_t(deviceBytes), 0, -2));
        }
    }
    pdRuntime = runtime;
    try {
        if (runtime->role == "prefill") {
            Check(!config["listen"].string_value().empty(), "prefill listen endpoint is required");
            const bool batching = SupportsPrefillBatching() && canDoBatchForward;
            runtime->batchRequests = batching ? concurrent : 1;
            if (prefillBatchPolicy.maxRequests > 0)
                runtime->batchRequests = std::min(runtime->batchRequests, prefillBatchPolicy.maxRequests);
            runtime->batchTokens = batching ? std::max(runtime->chunk, GetBatchedPrefillTokenLimit()) : runtime->chunk;
            runtime->targetRequests = prefillBatchPolicy.targetRequests > 0 ?
                std::min(runtime->batchRequests, prefillBatchPolicy.targetRequests) : runtime->batchRequests;
            runtime->waitMs = batching ? prefillBatchPolicy.maxWaitMs : 0;
            runtime->prefillThread = std::thread([this, raw = runtime.get()] { raw->RunPrefill(*this); });
            runtime->service->StartAsync(config["listen"].string_value(),
                [this, raw = runtime.get()](fastkv::Message request,
                                           std::shared_ptr<fastkv::Cancellation> cancellation) {
                fastkv::Reader reader(request.metadata);
                const auto kind = reader.U32();
                Check((kind == 1 || kind == 3) && reader.String() == raw->identity,
                      "incompatible checkpoint or state layout");
                auto tokens = reader.Ints(size_t(max_positions));
                RequestMetadata metadata;
                if (kind == 3) metadata.sessionKey = reader.String(512);
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                return raw->Prefill(std::move(tokens), std::move(cancellation), std::move(metadata));
            }, [this, raw = runtime.get()](const fastkv::Message &request, fastkv::Message &reply) {
                if (request.metadata.size() < 4) return false;
                fastkv::Reader reader(request.metadata);
                const auto kind = reader.U32();
                if (kind != 2 && kind != 4 && kind != 6 && kind != 7 && kind != 8) return false;
                Check(reader.String() == raw->identity, "incompatible routing identity");
                if (kind == 7) {
                    const auto session = reader.String(512);
                    fastkv::SessionDirectory::Record record;
                    record.version = reader.U64(); record.owner = reader.String(512);
                    const auto owned = reader.U32(); reader.End();
                    Check(owned <= 1 && request.buffers.empty(), "invalid placement announcement");
                    // Keep the wire field for older peers. Placement is a
                    // routing hint, never a model-cache eviction priority.
                    raw->sessions->Update(session, record);
                    fastkv::Writer writer; writer.U32(7); reply.metadata = std::move(writer.bytes);
                    return true;
                }
                const auto tokens = reader.Ints(size_t(max_positions));
                const auto session = kind >= 6 ? reader.String(512) : std::string(); reader.End();
                Check(!tokens.empty() && tokens.size() < size_t(max_positions), "invalid routing prompt");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds routing peer capacity");
                fastkv::Writer writer;
                writer.U32(kind); writer.U64(raw->CachedTokens(*this, tokens));
                writer.U64(raw->workload->tokens.load());
                writer.U64(uint64_t(std::max(1.0, raw->workload->tokensPerSecond.load())));
                if (kind >= 6) {
                    const auto record = raw->sessions->Get(session);
                    writer.U64(record.version); writer.String(record.owner);
                }
                if (kind == 8) {
                    const auto hit = raw->hostCache ? raw->hostCache->Find(raw->identity, tokens) : fastkv::HostCache::Hit{};
                    writer.U64(hit.Tokens()); writer.U64(hit.Bytes());
                    writer.U64(raw->workload->restoreBytes.load());
                    writer.U64(uint64_t(raw->HostRestoreBytesPerSecond()));
                }
                reply.metadata = std::move(writer.bytes);
                return true;
            });
        }
    } catch (...) {
        pdRuntime.reset();
        throw;
    }
    std::cout << "[FastKV PD] " << runtime->role << " layout=" << PDLayout() << std::endl;
}

void basellm::BeginPD(ResponseContext *context) {
    if (!IsPDDecode() || context->allTokens.size() < 2)
        return;
    Check(context->multimodalInput.empty(), "multimodal PD state transfer is not implemented");
    Check(context->allTokens.size() < size_t(max_positions), "prompt exceeds context capacity");
    std::vector<int> prefix(context->allTokens.begin(), context->allTokens.end() - 1);
    fastkv::Writer writer;
    writer.U32(1);
    writer.String(pdRuntime->identity);
    writer.Ints(prefix);
    auto pending = std::make_shared<PDPending>();
    pending->length = prefix.size();
    pending->sessionKey = context->requestMetadata.sessionKey;
    pending->sessionCapable.resize(pdRuntime->peers.size(), false);
    pending->placementCapable.resize(pdRuntime->peers.size(), false);
    const int output = context->generationConfig.output_token_limit;
    pending->reservation = output > 0
                               ? std::min(size_t(max_positions), context->allTokens.size() + size_t(output))
                               : size_t(max_positions);
    Check(pending->reservation <= pdRuntime->tokenCapacity,
          "request reservation exceeds the decode token pool");
    // Admission must cover the final partially used KV page as well. If the
    // configured pool is smaller than one page, reserve its entire budget.
    const size_t page = std::max(1, GetPageLen());
    pending->reservation =
        std::min(pdRuntime->tokenCapacity, ((pending->reservation + page - 1) / page) * page);
    pending->estimates.resize(pdRuntime->peers.size());
    pending->probes.resize(pdRuntime->peers.size());
    pending->probeCancellations.resize(pdRuntime->peers.size());
    pending->probed.resize(pdRuntime->peers.size(), false);
    pending->probeKinds.resize(pdRuntime->peers.size(), 8);
    for (auto &e : pending->estimates) e.available = false;
    pending->request.metadata = std::move(writer.bytes);
    context->pdPending = std::move(pending);
}

bool basellm::PollPD(ResponseContext *context) {
    if (pdRuntime) pdRuntime->FlushAnnouncements();
    if (!context->pdPending)
        return true;
    auto pending = context->pdPending;
    if (pending->committed)
        return true;
    if (!pending->result.valid()) {
        // Admission precedes P work and receive allocation. Waiting requests
        // do not consume model pages or block the scheduler on a network call.
        try {
            if (!pending->admission) {
                auto admission = std::make_shared<PDPending::Admission>();
                admission->slot = pdRuntime->slots->Reserve(1);
                {
                    std::lock_guard<std::mutex> lock(forwardLocker);
                    admission->tokens = pdRuntime->ReserveGroup(*this, *context, pending->reservation);
                }
                context->pdAdmission = admission;
                pending->admission = std::move(admission);
            }
        } catch (const fastkv::CapacityBusy &) {
            return false;
        }
        try {
            if (!pdRuntime->SelectPeer(*pending, verbose)) return false;
            if (!pending->receiver) {
                std::lock_guard<std::mutex> lock(forwardLocker);
                const auto begin = std::chrono::steady_clock::now();
                pending->receiver = PreparePDReceive(*context);
                if (pending->receiver && verbose) std::cout << "[FastKV PD timing] receive_prepare_ms="
                    << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()
                    << std::endl;
            }
            fastkv::ReceiveTarget target;
            if (pending->receiver) target = [receiver = pending->receiver](const fastkv::Bytes &metadata,
                                                                          const std::vector<size_t> &sizes) {
                return receiver->Targets(metadata, sizes);
            };
            // Transient transport pressure must preserve the prompt for retry.
            // The Exchange also holds admission until cancellation/I/O drains.
            auto lifetime = std::make_shared<std::pair<std::shared_ptr<PDPending::Admission>,
                                                       std::shared_ptr<void>>>(pending->admission, pending->route);
            const auto devices = PDRequestDevices(*context);
            fastkv::ReceivePlacement placement = [devices](const fastkv::Bytes &metadata,
                                                          const std::vector<size_t> &sizes) {
                return PDReceivePlacement(metadata, sizes.size(), devices);
            };
            pdRuntime->service->TrySubmit(pending->peer, pending->request, pending->result,
                pending->cancellation, std::move(lifetime), std::move(target), std::move(placement));
        } catch (const fastkv::CapacityBusy &) {
            return false;
        } catch (const std::exception &error) {
            std::cerr << "[FastKV PD] admission failed: " << error.what() << std::endl;
            context->error = ResponseContextErrorRuntime;
            context->isEnding = true;
            context->pdPending.reset();
        }
        return false;
    }
    if (pending->result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return false;
    try {
        auto state = pending->result.get();
        pending->route.reset();
        std::lock_guard<std::mutex> lock(forwardLocker);
        const auto importBegin = std::chrono::steady_clock::now();
        if (pending->receiver) pending->receiver->Commit(*context);
        else ImportPDState(*context, state);
        pending->receiver.reset();
        if (verbose) std::cout << "[FastKV PD timing] import_ms="
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - importBegin).count()
            << " kv_d2h_bytes=" << fastkv::Buffer::Traffic().deviceToHost
            << " kv_h2d_bytes=" << fastkv::Buffer::Traffic().hostToDevice << std::endl;
        context->preTokens = pending->length;
        context->currentTokens = {context->allTokens.back()};
        context->intParams = {
            {"promptLen", context->inputTokens}, {"index", -1}, {"add_special_tokens", 0}};
        if (verbose) {
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pending->start)
                    .count();
            size_t bytes = 0;
            for (const auto &buffer : state.buffers)
                bytes += buffer.Size();
            std::cout << "[FastKV PD] committed tokens=" << pending->length << " bytes=" << bytes
                      << " handoff_ms=" << ms << std::endl;
        }
        pending->committed = true;
        pdRuntime->CommitPlacement(*pending);
        pdRuntime->FlushAnnouncements();
    } catch (const std::exception &error) {
        std::cerr << "[FastKV PD] request failed: " << error.what() << std::endl;
        context->error = ResponseContextErrorRuntime;
        context->isEnding = true;
    }
    if (context->isEnding)
        context->pdPending.reset();
    NotifyResponseReady();
    return !context->isEnding;
}
} // namespace fastllm
