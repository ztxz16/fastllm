#include "pd.h"
#include "fastkv/host_cache.h"
#include "json11.hpp"
#include "models/basellm.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <sstream>
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
// This probe negotiates a per-request cache-hit count after the reply's model metadata.
constexpr uint32_t PDUsageProbe = 10, PDPrefillWithUsage = 9, PDShardPrefill = 11;
template<class Receive>
Receive WithoutPDUsageTrailer(Receive receive) {
    return [receive = std::move(receive)](const fastkv::Bytes &metadata, const std::vector<size_t> &sizes) {
        Check(metadata.size() >= sizeof(uint32_t), "missing PD cache usage");
        return receive(fastkv::Bytes(metadata.begin(), metadata.end() - sizeof(uint32_t)), sizes);
    };
}
constexpr size_t PDTransferArenaBytes = size_t(64) << 20;
fastkv::Bytes PDEnvelope(const fastkv::Bytes &metadata, const std::vector<int> &placement) {
    fastkv::Writer envelope;
    envelope.U32(PDWireMagic);
    envelope.Blob(metadata.data(), metadata.size());
    envelope.Ints(placement);
    return std::move(envelope.bytes);
}
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
fastkv::Message PDRowMessage::Encode() const {
    fastkv::Writer header;
    header.String(layout); header.Blob(metadata.data(), metadata.size()); header.U32(shards.size());
    fastkv::Message wire;
    for (const auto &shard : shards) {
        header.String(shard.component); header.U64(shard.offset); header.U64(shard.width);
        wire.buffers.push_back(shard.buffer);
    }
    wire.metadata = PDEnvelope(header.bytes, std::vector<int>(shards.size(), 0));
    return wire;
}
PDRowMessage::PDRowMessage(const fastkv::Message &wire) {
    const auto state = PDStatePayload(wire);
    fastkv::Reader header(state.metadata);
    layout = header.String(); metadata = header.Blob();
    Check(header.U32() == state.buffers.size(), "invalid shard count");
    for (const auto &buffer : state.buffers)
        shards.push_back({header.String(128), header.U64(), header.U64(), buffer});
    header.End();
}
void PDRowMessage::Scatter(const std::map<std::string, PDRowDestination> &targets, int device) const {
#ifdef USE_CUDA
    std::map<std::string, std::vector<fastkv::Shard>> sources;
    for (const auto &s : shards) {
        auto it = targets.find(s.component);
        Check(it != targets.end(), "unknown PD component");
        const auto &t = it->second;
        Check(t.rows && s.width && s.width <= SIZE_MAX / t.rows && s.buffer.Size() == s.width * t.rows &&
              s.buffer.Kind() == fastkv::MemoryKind::Cuda && s.buffer.Device() == device, "invalid PD row shard");
        sources[s.component].push_back({s.component, layout, {}, s.buffer, {t.rowBytes}, {s.offset}, {s.width}, 1, 0});
    }
    std::vector<void *> dst; std::vector<const void *> src;
    std::vector<size_t> dstPitch, srcPitch, widths, heights;
    struct PeerCopy { void *dst; int device; const void *src; size_t bytes; };
    std::vector<PeerCopy> peers;
    for (const auto &item : targets) {
        const auto &t = item.second;
        Check(t.rows && t.rowBytes && !t.spans.empty(), "missing PD row destination");
        const auto &inputs = sources.at(item.first);
        fastkv::Shard row{item.first, layout, {}, t.spans.front(), {t.rowBytes}, {0}, {t.rowBytes}, 1, 0};
        // Validate one row's coverage once and repeat strided copies per page
        // run. A 200k-token prompt never expands into 200k transport spans.
        const auto plan = fastkv::PlanCopies(inputs, {row});
        size_t first = 0;
        for (const auto &span : t.spans) {
            const size_t count = span.Size() / t.rowBytes;
            Check(span.Kind() == fastkv::MemoryKind::Cuda && count && span.Size() % t.rowBytes == 0 &&
                  count <= t.rows - first, "invalid PD page span");
            for (const auto &copy : plan) {
                const auto &in = inputs[copy.source];
                const size_t pitch = in.shape.front();
                auto *to = static_cast<char *>(span.MutableData()) + copy.targetOffset;
                const auto *from = static_cast<const char *>(in.buffer.Data()) + first * pitch + copy.sourceOffset;
                if (span.Device() == device) {
                    dst.push_back(to); src.push_back(from); dstPitch.push_back(t.rowBytes);
                    srcPitch.push_back(pitch); widths.push_back(copy.bytes); heights.push_back(count);
                } else {
                    Check(pitch == copy.bytes && t.rowBytes == pitch, "cross-device row slice needs contiguous storage");
                    peers.push_back({to, span.Device(), from, count * pitch});
                }
            }
            first += count;
        }
        Check(first == t.rows, "incomplete PD destination rows");
    }
    const int previous = FastllmCudaGetDevice(); FastllmCudaSetDevice(device);
    bool ok = FastllmCudaBatchCopy2DAsyncCurrentThread(dst.data(), src.data(), dstPitch.data(),
        srcPitch.data(), widths.data(), heights.data(), widths.size());
    for (const auto &peer : peers)
        if (ok) ok = FastllmCudaMemcpyPeerAsyncCurrentThread(peer.device, peer.dst, device, peer.src, peer.bytes);
    FastllmCudaSyncCurrentThreadStream();
    ok = ok && !FastllmCudaGetThreadError();
    FastllmCudaSetDevice(previous);
    Check(ok, "GPU row shard transfer failed");
#else
    throw std::runtime_error("PD row scatter requires CUDA");
#endif
}

struct PDStateWriter::Allocations {
    std::mutex mutex;
    std::vector<std::pair<int, size_t>> plan;
    std::unique_ptr<fastkv::BufferPool::PackedLayout> layout;
    std::vector<fastkv::Buffer> reserved;
    bool prepared = false;
    int producer = -1;
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
    plannedTensors.push_back(allocations->plan.size());
    allocations->plan.emplace_back(data.dataDeviceIds.front(), bytes);
}
void PDStateWriter::Reserve(bool allowPeerWrites) {
    Check(!allocations->prepared, "export plan already reserved");
    if (pool) {
        if (!allocations->layout)
            allocations->layout = std::make_unique<fastkv::BufferPool::PackedLayout>(allocations->plan);
        if (allowPeerWrites && !allocations->plan.empty()) {
            const int device = allocations->plan.front().first;
            Check(std::all_of(allocations->plan.begin(), allocations->plan.end(),
                [device](const auto &part) { return part.first == device; }), "peer export needs a single producer");
            allocations->producer = device;
            allocations->reserved = pool->AcquirePeerPackedLayout(*allocations->layout);
        } else allocations->reserved = pool->AcquirePackedLayout(*allocations->layout);
    } else for (const auto &item : allocations->plan)
        allocations->reserved.push_back(fastkv::Buffer::Allocate(item.second, item.first));
    allocations->prepared = true;
}
fastkv::Buffer PDStateWriter::Allocate(size_t bytes, int device) {
    if (allocations->prepared) {
        Check(nextTensor < plannedTensors.size(), "unplanned export allocation");
        auto &reserved = allocations->reserved.at(plannedTensors[nextTensor++]);
        Check((reserved.Device() == device || allocations->producer == device) &&
              reserved.Size() == bytes, "export plan changed");
        return std::move(reserved);
    }
    if (!pool) return fastkv::Buffer::Allocate(bytes, device);
    const size_t aligned = (arenaOffset + 255) / 256 * 256;
    if (!arena.Size() || arena.Device() != device || aligned > arena.Size() || bytes > arena.Size() - aligned) {
        const size_t capacity = std::max(bytes, PDTransferArenaBytes);
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
    if (devices.empty() && allocations->producer >= 0) devices.push_back(allocations->producer);
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
            // Wire ranks name the model shard. A peer-backed export may be
            // transmitted by a different physical GPU and its local NIC.
            int owner = allocations->producer >= 0 ? allocations->producer : buffer.Device();
            auto it = std::find(devices.begin(), devices.end(), owner);
            Check(it != devices.end(), "payload owner absent from device map");
            rank = it - devices.begin();
        }
        placement.push_back(rank);
    }
    result.metadata = PDEnvelope(result.metadata, placement);
    return result;
}
fastkv::Buffer PDBindTensor(fastkv::Reader &metadata, Data &storage,
                            const std::shared_ptr<void> &lease, int boundedAxis) {
    const auto type = metadata.U32();
    const auto dims = metadata.Ints(8);
    const auto fp8 = metadata.U32();
    const auto bytes = metadata.U64();
    Check(type == uint32_t(storage.dataType) && fp8 == uint32_t(storage.kvCacheFp8Indexer) &&
          dims.size() == storage.dims.size() && storage.dataDevice == CUDA &&
          storage.dataDeviceIds.size() == 1 && storage.cudaData, "invalid direct tensor descriptor");
    for (size_t i = 0; i < dims.size(); ++i)
        Check(int(i) == boundedAxis ? dims[i] > 0 && dims[i] <= storage.dims[i] :
              dims[i] == storage.dims[i], "invalid direct tensor shape");
    storage.Resize(dims);
    Check(bytes && TensorBytes(storage) == bytes, "invalid direct tensor size");
    return fastkv::Buffer::Borrow(storage.cudaData, bytes, fastkv::MemoryKind::Cuda,
                                  storage.dataDeviceIds.front(), lease);
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
    std::vector<bool> probed, shardedPeers;
    fastkv::Message probeRequest;
    std::vector<uint32_t> probeKinds;
    fastkv::SessionDirectory::Record placement;
    std::shared_ptr<fastkv::Cancellation> cancellation = std::make_shared<fastkv::Cancellation>();
    int length = 0;
    size_t reservation = 0;
    bool committed = false, reportUsage = false, sharded = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PDPending() {
        cancellation->Cancel();
        for (const auto &probe : probeCancellations) if (probe) probe->Cancel();
    }
};

bool ResponseContext::HasPendingPD() const {
    return pdPending && !pdPending->committed;
}

bool ResponseContext::NeedsPDSubmission() const {
    return HasPendingPD() && !pdPending->result.valid();
}

bool ResponseContext::IsPDTransferPending() const {
    return HasPendingPD() && pdPending->result.valid() &&
        pdPending->result.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
}

struct PDRuntime {
    struct Workload {
        std::atomic<uint64_t> tokens{0}, restoreBytes{0};
        std::atomic<double> tokensPerSecond{10000};
    };
    struct PrefillRequest {
        std::vector<int> input;
        RequestMetadata metadata;
        std::promise<fastkv::Message> result;
        bool reportUsage = false;
        const basellm *shardAdapter = nullptr;
        void Reply(fastkv::Message state, uint32_t cachedTokens) {
            if (shardAdapter) state = shardAdapter->PDAlternateState(state);
            if (reportUsage) {
                fastkv::Writer trailer;
                trailer.bytes = std::move(state.metadata);
                trailer.U32(cachedTokens);
                state.metadata = std::move(trailer.bytes);
            }
            result.set_value(std::move(state));
        }
        std::shared_ptr<fastkv::Cancellation> cancellation;
        std::shared_ptr<void> groupAdmission;
        std::unique_ptr<ResponseContext> context;
        fastkv::HostCache::Hit hostHit;
        std::future<fastkv::Message> hostRestore;
        std::chrono::steady_clock::time_point restoreBegin;
        bool restoring = false;
        double prefixRecordMs = 0;
        std::function<fastkv::Message()> pack;
        int exportGroup = 0;
        std::atomic<bool> exportFinished{false};
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
    std::string role, identity, alternateIdentity;
    bool Accepts(const std::string &name) const {
        return name == identity || (!alternateIdentity.empty() && name == alternateIdentity);
    }
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
        std::lock_guard<std::mutex> stoppingLock(prefillStopMutex);
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            stopping = true;
        }
        if (exportPool) exportPool->Clear();
        prefillCV.notify_all();
        if (prefillThread.joinable()) prefillThread.join();
    }
    size_t CachedTokens(basellm &model, const std::vector<int> &input, const std::string &session) {
        if (residentPrefix) {
            const int group = sessions->LocalGroup(session, model.PDGroupCount(), false);
            return model.PreviewPDResidentPrefix(input, group);
        }
        return cache ? cache->ProbeTokens(identity, input) : 0;
    }
    double HostRestoreBytesPerSecond() const {
        const auto stats = hostCache ? hostCache->Stats() : fastkv::HostCache::Statistics{};
        return stats.restoreMicroseconds ?
            std::max(1.0, double(stats.restoredBytes) * 1e6 / stats.restoreMicroseconds) : 1e10;
    }
    std::future<fastkv::Message> Prefill(std::vector<int> input,
                                       std::shared_ptr<fastkv::Cancellation> cancellation, RequestMetadata metadata,
                                       bool reportUsage, const basellm *shardAdapter) {
        auto request = std::make_shared<PrefillRequest>();
        request->input = std::move(input);
        request->metadata = std::move(metadata);
        request->reportUsage = reportUsage;
        request->shardAdapter = shardAdapter;
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
        for (size_t i = 0; i < peers.size(); ++i) a.complete[i] = pending.probeKinds[i] < 6;
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
                        if (kind >= 8) {
                            e.hostCachedTokens = reader.U64();
                            const auto bytes = reader.U64(), queuedBytes = reader.U64(), rate = reader.U64();
                            Check(rate > 0 && e.hostCachedTokens <= size_t(pending.length), "invalid host cache estimate");
                            e.restoreSeconds = double(bytes) / rate; e.pendingRestoreSeconds = double(queuedBytes) / rate;
                        }
                        const bool sharded = kind == PDUsageProbe && reader.Remaining();
                        if (sharded) {
                            Check(reader.U32() == PDShardPrefill, "unknown PD transfer mode");
                        }
                        reader.End();
                        Check(e.cachedTokens <= size_t(pending.length) && e.tokensPerSecond > 0,
                              "invalid routing estimate");
                        if (sharded) {
                            if (pending.shardedPeers.empty()) pending.shardedPeers.resize(peers.size());
                            pending.shardedPeers[i] = true;
                        }
                        e.available = true; pending.probed[i] = true;
                    } catch (const std::exception &) {
                        if (pending.probeKinds[i] > 2)
                            pending.probeKinds[i] -= 2; // Retry metadata only, never model work.
                        else { pending.probeKinds[i] = 0; pending.probed[i] = true; }
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
        auto placement = sessions->Get(pending.sessionKey);
        auto owner = std::find(peers.begin(), peers.end(), placement.owner);
        const size_t preferred = placement.version && owner != peers.end() ? size_t(owner - peers.begin()) :
            fastkv::PrefillRouter::SessionPeer(pending.sessionKey, peers);
        auto chosen = router->Reserve(pending.length, pending.estimates, preferred, preferred != SIZE_MAX);
        const auto probeKind = pending.probeKinds[chosen.peer];
        if (!pending.sessionKey.empty() && probeKind >= 6) {
            Check(placement.version < UINT64_MAX, "session placement version exhausted");
            pending.placement = placement.owner == peers[chosen.peer] ? placement :
                fastkv::SessionDirectory::Record{placement.version + 1, peers[chosen.peer]};
        }
        pending.reportUsage = probeKind == PDUsageProbe;
        pending.sharded = !pending.shardedPeers.empty() && pending.shardedPeers[chosen.peer];
        if (pending.reportUsage || (!pending.sessionKey.empty() && probeKind >= 4)) {
            pending.request.metadata[0] = pending.sharded ? PDShardPrefill : pending.reportUsage ? PDPrefillWithUsage : 3;
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
    // Only the model lane retires contexts. The export lane owns no model
    // scheduler state and only sees completed, pinned requests.
    std::mutex exportMutex;
    std::condition_variable exportCV;
    std::deque<Request> readyToPack;
    std::thread exportThread;
    std::atomic<bool> closeExports{false};
    const auto cancelled = std::make_exception_ptr(std::runtime_error("FastKV PD: request cancelled"));
    auto recordPrefix = [&](const Request &request) {
        if (!residentPrefix) return;
        const auto begin = std::chrono::steady_clock::now();
        request->context->TryRecordPagedCache(&model);
        request->prefixRecordMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    };
    auto complete = [&](const Request &request, fastkv::Message *restored) {
        const auto exportBegin = std::chrono::steady_clock::now();
        auto reply = restored ? std::move(*restored) :
            request->pack ? request->pack() : model.ExportPDState(*request->context);
        if (request->pack && (stopping || request->cancellation->Cancelled()))
            std::rethrow_exception(cancelled);
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
        const int cachedTokens = request->context->pdCachedInputTokens;
        if (hostCache && model.verbose) {
            const auto stats = hostCache->Stats();
            std::cout << "[FastKV L2 cache] entries=" << stats.entries << " bytes=" << stats.bytes
                      << " queued=" << stats.queued << " skipped=" << stats.skipped << " failures=" << stats.failures
                      << " d2h_bytes=" << stats.copiedToHost << " h2d_bytes=" << stats.restoredBytes
                      << " reused_bytes=" << stats.reusedBytes
                      << " pinned_bytes=" << stats.pinnedResidentBytes
                      << " direct_d2h_bytes=" << stats.directToHost << " direct_h2d_bytes=" << stats.directToDevice
                      << " d2h_copy_us=" << stats.publishCopyMicroseconds << " h2d_copy_us=" << stats.restoreMicroseconds << std::endl;
        }
        if (request->pack) model.ReleasePDExportSources(*request->context);
        else retire(request);
        if (model.verbose) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            std::ostringstream timing;
            timing << "[FastKV PD timing] record_ms=" << request->prefixRecordMs
                      << " export_ms=" << ms(exportBegin, exported)
                      << " export_wait_ms=" << (request->exportQueued.time_since_epoch().count() ?
                          ms(request->exportQueued, exportBegin) : 0.0)
                      << " publish_ms=" << ms(exported, published);
            if (request->pack) timing << " source_release_ms=" << ms(published, std::chrono::steady_clock::now())
                                      << " retire_deferred=1";
            else timing << " retire_ms=" << ms(published, std::chrono::steady_clock::now());
            timing << " kv_d2h_bytes=" << fastkv::Buffer::Traffic().deviceToHost
                      << " kv_h2d_bytes=" << fastkv::Buffer::Traffic().hostToDevice << '\n';
            std::cout << timing.str() << std::flush;
        }
        request->Reply(std::move(reply), cachedTokens);
    };
    auto runExports = [&] {
        std::deque<Request> pending;
        while (!closeExports && !stopping) {
            {
                std::unique_lock<std::mutex> lock(exportMutex);
                if (pending.empty()) exportCV.wait(lock, [&] { return closeExports || !readyToPack.empty(); });
                pending.insert(pending.end(), readyToPack.begin(), readyToPack.end());
                readyToPack.clear();
            }
            bool progressed = false;
            std::set<int> blockedGroups;
            for (auto it = pending.begin(); it != pending.end() && !closeExports && !stopping;) {
                const auto &request = *it;
                try {
                    if (request->cancellation->Cancelled()) std::rethrow_exception(cancelled);
                    if (blockedGroups.count(request->exportGroup)) {
                        ++it;
                        continue;
                    }
                    complete(request, nullptr);
                } catch (const fastkv::CapacityBusy &) {
                    blockedGroups.insert(request->exportGroup);
                    ++it;
                    continue;
                } catch (...) {
                    request->result.set_exception(std::current_exception());
                }
                request->exportFinished.store(true, std::memory_order_release);
                it = pending.erase(it);
                progressed = true;
            }
            if (progressed) prefillCV.notify_one();
            else if (!pending.empty() && !stopping && !closeExports) {
                try { exportPool->WaitForRelease(); }
                catch (const std::runtime_error &) { return; } // StopPrefill closed the pool.
            }
        }
    };
    auto startExport = [&](const Request &request) {
        recordPrefix(request);
        request->pack = model.PreparePDExport(*request->context);
        if (!request->pack) {
            try {
                complete(request, nullptr);
            } catch (const fastkv::CapacityBusy &) {
                request->exportQueued = std::chrono::steady_clock::now();
                exporting.push_back(request);
            }
            return;
        }
        request->exportQueued = std::chrono::steady_clock::now();
        request->exportGroup = model.PDRequestGroup(*request->context);
        if (!exportThread.joinable()) exportThread = std::thread(runExports);
        exporting.push_back(request);
        try {
            std::lock_guard<std::mutex> lock(exportMutex);
            readyToPack.push_back(request);
        } catch (...) {
            exporting.pop_back();
            throw;
        }
        exportCV.notify_one();
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
            // The model lane owns context mutation and retirement.
            // RPC handlers return futures immediately; network workers never
            // wait for the model or hold forwardLocker.
            std::unique_lock<std::mutex> lock(model.forwardLocker);
            // Completed contexts retain their pages and admission while waiting
            // for a whole-message export reservation. Other requests can still
            // forward; no owner worker waits on a transport buffer lease.
            for (auto it = exporting.begin(); it != exporting.end();) {
                const auto request = *it;
                if (request->pack) {
                    if (request->exportFinished.load(std::memory_order_acquire)) {
                        retire(request);
                        it = exporting.erase(it);
                    } else ++it;
                    continue;
                }
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
                        context.pdCachedInputTokens = context.preTokens;
                        request->restoring = false; request->SetRestoreBytes(0);
                        request->SetRemaining(request->input.size() - context.preTokens);
                        if (model.verbose) {
                            const auto stats = hostCache->Stats();
                            std::cout << "[FastKV PD] prefix_hit=" << context.preTokens << " prompt=" << request->input.size()
                                      << " cache_tier=l2 restore_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - request->restoreBegin).count()
                                      << " bytes=" << request->hostHit.Bytes() << " cache_bytes=" << stats.bytes
                                      << " entries=" << stats.entries << " d2h_bytes=" << stats.copiedToHost
                                      << " reused_bytes=" << stats.reusedBytes << " group=" << model.PDRequestGroup(context) << std::endl;
                        }
                        if (size_t(context.preTokens) == request->input.size()) {
                            const auto reservation = request->reservation;
                            recordPrefix(request);
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
                        request->Reply(*hit.state, hit.tokens);
                        continue;
                    }
                    request->context.reset(new ResponseContext);
                    auto &context = *request->context;
                    context.Init(model.block_cnt, model.dataType, model.kvCacheDataType);
                    context.requestMetadata = request->metadata;
                    context.pdPrefillGroup = sessions->LocalGroup(request->metadata.sessionKey, model.PDGroupCount(), true);
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
                    const size_t residentTokens = request->hostHit && residentPrefix
                        ? model.PreviewPDResidentPrefix(request->input, model.PDRequestGroup(context)) : hit.tokens;
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
                    context.pdCachedInputTokens = context.preTokens;
                    request->SetRemaining(request->input.size() - context.preTokens);
                    if (model.verbose) {
                        const auto stats = cache ? cache->Stats() : fastkv::PrefixCache::Statistics{};
                        std::cout << "[FastKV PD] prefix_hit=" << hit.tokens << " prompt=" << request->input.size()
                                  << " cache_entries=" << stats.entries << " cache_bytes=" << stats.bytes
                                  << " cache_logical_bytes=" << stats.logicalBytes
                                  << " cache_evictions=" << stats.evictions
                                  << " resident_gpu=" << residentPrefix << " group=" << model.PDRequestGroup(context) << std::endl;
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
                    startExport(request);
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
            std::lock_guard<std::mutex> lock(exportMutex);
            closeExports = true;
        }
        exportCV.notify_one();
        if (exportThread.joinable()) exportThread.join();
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
            if (!request->exportFinished) request->result.set_exception(error);
        }
    }
}

std::string basellm::PDLayout() const {
    throw std::invalid_argument("FastKV PD: this model has no state adapter");
}
void PDReceiveState::CommitShards(ResponseContext &, const fastkv::Message &) {
    throw std::logic_error("PD receiver has no shard adapter");
}
fastkv::Message basellm::PDAlternateState(const fastkv::Message &) const {
    throw std::logic_error("PD producer has no shard adapter");
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
    Check(!pdPrefillOnly || runtime->role == "prefill", "prefill-only warmup requires a prefill PD role");
    Check(!config["model_id"].string_value().empty(),
          "model_id must identify the same checkpoint on P and D");
    const auto identityPrefix = config["model_id"].string_value() + ":gpu-state-v2:";
    runtime->identity = identityPrefix + PDLayout();
    if (runtime->role == "prefill") {
        const auto alternate = PDAlternateLayout();
        if (!alternate.empty()) runtime->alternateIdentity = identityPrefix + alternate;
    }
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
    options.allocateBuffer = [](size_t bytes, int device) {
        if (device < 0) return fastkv::Buffer::Allocate(bytes, device);
        struct RestoreDevice {
            int previous = FastllmCudaGetDevice();
            ~RestoreDevice() { FastllmCudaSetDevice(previous); }
        } restore;
        FastllmCudaSetDevice(device);
        void *data = nullptr;
        // Reuse the engine's pressure recovery. It protects live/graph-owned
        // allocations and reclaims only idle eager scratch before retrying.
        const auto result = FastllmCudaTryDirectMalloc(&data, std::max(size_t(1), bytes));
        if (result == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE)
            throw fastkv::CapacityBusy("FastKV: engine GPU allocation pressure");
        Check(result == FASTLLM_CUDA_TRY_MALLOC_SUCCESS && data, "GPU buffer allocation failed");
        auto owner = std::shared_ptr<void>(data, [device](void *pointer) {
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            FastllmCudaDirectFree(pointer);
            FastllmCudaSetDevice(previous);
        });
        return fastkv::Buffer::Borrow(data, bytes, fastkv::MemoryKind::Cuda, device, std::move(owner));
    };
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
    runtime->exportPool.reset(new fastkv::BufferPool(options.maxInflightBytes,
        options.deviceBufferBytes, options.allocateBuffer));
#ifdef USE_CUDA
    if (runtime->role == "prefill" && groups > 1 && groups == int(devices.size()) &&
        options.transport == fastkv::Transport::Rdma)
        runtime->exportPool->Preallocate(devices, fastkv::EnableCudaPeerAccess(devices));
#endif
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
        auto cacheLimit = [&](const char *key, int64_t fallback) -> int64_t {
            const auto &value = config[key];
            if (value.is_null()) return fallback;
            const double n = value.number_value();
            Check(value.is_number() && n >= 0 && n <= double(1ULL << 40) && n == int64_t(n),
                  "prefix cache budgets must be integer byte counts in [0, 1 TiB]");
            return int64_t(n);
        };
        fastkv::HostCache::Options host;
        host.SetBudgets(cacheLimit("cache_bytes", 0), cacheLimit("cache_kv_bytes", -1),
                        cacheLimit("cache_swa_bytes", -1));
        host.pinnedBytes = cacheLimit("cache_pinned_bytes", std::min(host.pinnedBytes, host.bytes));
        Check(host.pinnedBytes <= host.bytes, "pinned cache budget exceeds total L2 capacity");
        Check(host.bytes <= (1ULL << 40), "total prefix cache budget exceeds 1 TiB");
        if (host.bytes) {
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
                const auto requested = reader.String();
                Check((kind == 1 || kind == 3 || kind == PDPrefillWithUsage || kind == PDShardPrefill) &&
                      raw->Accepts(requested) && ((kind == PDShardPrefill) == (requested != raw->identity)),
                      "incompatible checkpoint or state layout");
                auto tokens = reader.Ints(size_t(max_positions));
                RequestMetadata metadata;
                if (kind != 1) metadata.sessionKey = reader.String(512);
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                return raw->Prefill(std::move(tokens), std::move(cancellation), std::move(metadata),
                                    kind == PDPrefillWithUsage || kind == PDShardPrefill,
                                    kind == PDShardPrefill ? this : nullptr);
            }, [this, raw = runtime.get()](const fastkv::Message &request, fastkv::Message &reply) {
                if (request.metadata.size() < 4) return false;
                fastkv::Reader reader(request.metadata);
                const auto kind = reader.U32();
                if (kind != 2 && kind != 4 && kind != 6 && kind != 7 && kind != 8 && kind != PDUsageProbe) return false;
                const auto requested = reader.String();
                // Only the usage probe can negotiate a sharded response.
                Check(requested == raw->identity || ((kind == PDUsageProbe || kind == 7) && raw->Accepts(requested)),
                      "incompatible routing identity");
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
                writer.U32(kind); writer.U64(raw->CachedTokens(*this, tokens, session));
                writer.U64(raw->workload->tokens.load());
                writer.U64(uint64_t(std::max(1.0, raw->workload->tokensPerSecond.load())));
                if (kind >= 6) {
                    const auto record = raw->sessions->Get(session);
                    writer.U64(record.version); writer.String(record.owner);
                }
                if (kind >= 8) {
                    const auto hit = raw->hostCache ? raw->hostCache->Find(raw->identity, tokens) : fastkv::HostCache::Hit{};
                    writer.U64(hit.Tokens()); writer.U64(hit.Bytes());
                    writer.U64(raw->workload->restoreBytes.load());
                    writer.U64(uint64_t(raw->HostRestoreBytesPerSecond()));
                }
                if (kind == PDUsageProbe && requested != raw->identity) writer.U32(PDShardPrefill);
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
    if (context->allTokens.size() >= size_t(max_positions)) {
        context->error = ResponseContextErrorPromptTooLong;
        context->isEnding = true;
        return;
    }
    std::vector<int> prefix(context->allTokens.begin(), context->allTokens.end() - 1);
    fastkv::Writer writer;
    writer.U32(1);
    writer.String(pdRuntime->identity);
    writer.Ints(prefix);
    auto pending = std::make_shared<PDPending>();
    pending->length = prefix.size();
    pending->sessionKey = context->requestMetadata.sessionKey;
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
    pending->probeKinds.resize(pdRuntime->peers.size(), PDUsageProbe);
    for (auto &e : pending->estimates) e.available = false;
    pending->request.metadata = std::move(writer.bytes);
    context->pdPending = std::move(pending);
}

void basellm::PreparePendingPD(ResponseContext *context) {
    auto pending = context->pdPending;
    // Admission precedes P work and receive allocation. Waiting requests
    // do not consume model pages or block the scheduler on a network call.
    try {
        if (!pending->admission) {
            auto admission = std::make_shared<PDPending::Admission>();
            admission->slot = pdRuntime->slots->Reserve(1);
            admission->tokens = pdRuntime->ReserveGroup(*this, *context, pending->reservation);
            context->pdAdmission = admission;
            pending->admission = std::move(admission);
        }
        if (!pdRuntime->SelectPeer(*pending, verbose)) return;
        if (!pending->receiver) {
            const auto begin = std::chrono::steady_clock::now();
            pending->receiver = PreparePDReceive(*context);
            if (pending->receiver && verbose) std::cout << "[FastKV PD timing] receive_prepare_ms="
                << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()
                << std::endl;
        }
        fastkv::ReceiveTarget target;
        fastkv::ReceivePlacement placement;
        if (pending->receiver && !pending->sharded) {
            target = [receiver = pending->receiver](const fastkv::Bytes &metadata, const std::vector<size_t> &sizes) {
                return receiver->Targets(metadata, sizes);
            };
            if (pending->reportUsage) target = WithoutPDUsageTrailer(std::move(target));
        } else {
            placement = [devices = PDRequestDevices(*context)](const fastkv::Bytes &metadata,
                                                                const std::vector<size_t> &sizes) {
                return PDReceivePlacement(metadata, sizes.size(), devices);
            };
            if (pending->reportUsage) placement = WithoutPDUsageTrailer(std::move(placement));
        }
        // Transient transport pressure must preserve the prompt for retry.
        // The Exchange also holds admission until cancellation/I/O drains.
        auto lifetime = std::make_shared<std::pair<std::shared_ptr<PDPending::Admission>,
                                                   std::shared_ptr<void>>>(pending->admission, pending->route);
        pdRuntime->service->TrySubmit(pending->peer, pending->request, pending->result,
            pending->cancellation, std::move(lifetime), std::move(target), std::move(placement));
    } catch (const fastkv::CapacityBusy &) {
        return;
    } catch (const std::exception &error) {
        std::cerr << "[FastKV PD] admission failed: " << error.what() << std::endl;
        context->error = ResponseContextErrorRuntime;
        context->isEnding = true;
        context->pdPending.reset();
        NotifyResponseReady();
    }
}

bool basellm::PollPD(ResponseContext *context) {
    if (pdRuntime) pdRuntime->FlushAnnouncements();
    if (!context->pdPending)
        return true;
    auto pending = context->pdPending;
    if (pending->committed)
        return true;
    if (!pending->result.valid()) {
        std::lock_guard<std::mutex> lock(forwardLocker);
        PreparePendingPD(context);
        return false;
    }
    if (pending->result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return false;
    try {
        auto state = pending->result.get();
        int cachedTokens = -1;
        if (pending->reportUsage) {
            Check(state.metadata.size() >= sizeof(uint32_t), "missing PD cache usage");
            const size_t size = state.metadata.size() - sizeof(uint32_t);
            const auto count = fastkv::Reader(state.metadata.data() + size, sizeof(uint32_t)).U32();
            Check(count <= uint32_t(pending->length), "invalid PD cached token count");
            cachedTokens = int(count);
            state.metadata.resize(size);
        }
        pending->route.reset();
        std::lock_guard<std::mutex> lock(forwardLocker);
        const auto importBegin = std::chrono::steady_clock::now();
        if (pending->sharded) {
            Check(bool(pending->receiver), "missing destination for PD shards");
            pending->receiver->CommitShards(*context, state);
        } else if (pending->receiver) pending->receiver->Commit(*context);
        else ImportPDState(*context, state);
        pending->receiver.reset();
        context->pdCachedInputTokens = cachedTokens;
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
