#include "pd.h"
#include "devices/cuda/cudaworkspace.h"
#include "fastkv/host_cache.h"
#include "json11.hpp"
#include "models/basellm.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <optional>
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
constexpr uint32_t PDFirstTokenProbe = 12, PDFirstTokenPrefill = 13;

// Only the effective first-step distribution crosses the wire. The grammar
// state machine and its output history remain owned by the decode request.
void WriteFirstTokenSampling(fastkv::Writer &writer, const GenerationConfig &cfg) {
    writer.U32(cfg.do_sample);
    writer.Ints({cfg.top_k});
    for (float x : {cfg.temperature, cfg.top_p, cfg.repeat_penalty}) {
        uint32_t bits;
        std::memcpy(&bits, &x, sizeof(bits));
        writer.U32(bits);
    }
    writer.Ints(cfg.tool_call_allowed_token_ids);
    const auto &mask = cfg.tool_call_token_mask;
    writer.U32(mask ? mask->size() : 0);
    if (mask) for (uint32_t word : *mask) writer.U32(word);
}

GenerationConfig ReadFirstTokenSampling(fastkv::Reader &reader) {
    GenerationConfig cfg;
    const auto sample = reader.U32();
    const auto top = reader.Ints(1);
    Check(sample <= 1 && top.size() == 1, "invalid first-token sampler");
    cfg.do_sample = sample; cfg.top_k = top.front();
    for (float *out : {&cfg.temperature, &cfg.top_p, &cfg.repeat_penalty}) {
        const uint32_t bits = reader.U32();
        std::memcpy(out, &bits, sizeof(bits));
        Check(std::isfinite(*out), "invalid first-token sampling value");
    }
    cfg.tool_call_allowed_token_ids = reader.Ints();
    const uint32_t words = reader.U32();
    Check(words <= (1u << 17) && words <= reader.Remaining() / 4, "invalid first-token mask");
    if (words) {
        auto mask = std::make_shared<std::vector<uint32_t>>(words);
        for (auto &word : *mask) word = reader.U32();
        cfg.tool_call_token_mask = std::move(mask);
    }
    return cfg;
}

template<class Receive>
Receive WithoutPDUsageTrailer(Receive receive, size_t bytes) {
    return [receive = std::move(receive), bytes](const fastkv::Bytes &metadata, const std::vector<size_t> &sizes) {
        Check(metadata.size() >= bytes, "missing PD reply trailer");
        return receive(fastkv::Bytes(metadata.begin(), metadata.end() - bytes), sizes);
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
    bool committed = false, reportUsage = false, sharded = false, firstToken = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PDPending() {
        cancellation->Cancel();
        for (const auto &probe : probeCancellations) if (probe) probe->Cancel();
    }
};

int ResponseContext::PDTransferLength() const {
    return pdPending ? pdPending->length : int(allTokens.size()) - 1;
}

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
    std::vector<std::shared_ptr<CudaMemoryReservation>> prefillReservations;
    struct Workload {
        std::atomic<uint64_t> tokens{0}, restoreBytes{0};
        std::atomic<double> tokensPerSecond{10000};
    };
    struct PrefillRequest {
        std::vector<int> input;
        RequestMetadata metadata;
        std::promise<fastkv::Message> result;
        bool reportUsage = false;
        std::optional<GenerationConfig> sampling;
        int firstToken = -1;
        const basellm *shardAdapter = nullptr;
        void Reply(fastkv::Message state, uint32_t cachedTokens) {
            if (shardAdapter) state = shardAdapter->PDAlternateState(state);
            if (reportUsage) {
                fastkv::Writer trailer;
                trailer.bytes = std::move(state.metadata);
                if (sampling) {
                    Check(firstToken >= 0, "missing sampled first token");
                    trailer.U32(firstToken);
                }
                trailer.U32(cachedTokens);
                state.metadata = std::move(trailer.bytes);
            }
            result.set_value(std::move(state));
        }
        std::shared_ptr<fastkv::Cancellation> cancellation;
        std::shared_ptr<void> groupAdmission;
        std::unique_ptr<ResponseContext> context;
        std::unique_ptr<ResponseContext> preparedInput;
        std::shared_ptr<void> preparationSlot;
        bool ahead = false;
        PDPrefillPagePlan pages;

        std::unique_ptr<ResponseContext> MakeInputContext(const basellm &model) const {
            auto ctx = std::make_unique<ResponseContext>();
            ctx->Init(model.block_cnt, model.dataType, model.kvCacheDataType);
            ctx->currentTokens = ctx->allTokens = input;
            ctx->inputTokens = input.size();
            ctx->requestMetadata = metadata;
            ctx->generationConfig.top_k = 1;
            ctx->generationConfig.output_token_limit = 1;
            if (sampling) ctx->generationConfig = *sampling;
            ctx->generationConfig.input_token_length = input.size() + (sampling ? 0 : 1);
            return ctx;
        }
        fastkv::HostCache::Hit hostHit;
        std::future<fastkv::Message> hostRestore;
        std::chrono::steady_clock::time_point restoreBegin;
        bool restoring = false;
        double prefixRecordMs = 0;
        std::function<fastkv::Message()> pack;
        int exportGroup = 0;
        size_t publicationOrder = 0;
        std::atomic<bool> exportFinished{false};
        std::chrono::steady_clock::time_point exportQueued;
        int affinity = -1;
        uint64_t readyOrder = 0;
        int branchEnd = 0;
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
    // Keep the last owner on the model lane. A cancelled receive may outlive
    // its request while FastKV finishes DMA or destroys the exchange.
    std::vector<std::shared_ptr<PDReceiveState>> receivers;
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
    int chunk = 0;
    size_t tokenCapacity = 0;
    std::unique_ptr<fastkv::Budget> slots;
    std::vector<std::unique_ptr<fastkv::Budget>> groupTokens;
    std::shared_ptr<void> ReserveGroup(basellm &model, ResponseContext &context, size_t required,
                                       int preferred = -1) {
        std::vector<size_t> available;
        for (const auto &budget : groupTokens) available.push_back(tokenCapacity - budget->Used());
        if (preferred >= 0 && preferred != model.PDRequestGroup(context)) {
            std::vector<size_t> candidate(available.size(), 0);
            candidate.at(preferred) = available.at(preferred);
            if (model.SelectPDRequestGroup(context, candidate, required))
                return groupTokens.at(model.PDRequestGroup(context))->Reserve(required);
        }
        if (!model.SelectPDRequestGroup(context, available, required))
            throw fastkv::CapacityBusy("PD request group has no capacity");
        const int group = model.PDRequestGroup(context);
        Check(group >= 0 && size_t(group) < groupTokens.size(), "invalid request group");
        return groupTokens[group]->Reserve(required);
    }
    std::unique_ptr<fastkv::PrefixCache> cache;
    std::unique_ptr<fastkv::HostCache> hostCache;
    std::unique_ptr<fastkv::BufferPool> exportPool;
    std::unique_ptr<fastkv::Budget> preparationSlots;
    bool residentPrefix = false;
    // Only read during a synchronous model forward, on the prefill execution lane.
    std::unordered_map<const Data *, int> branchCheckpoints;
    std::mutex prefillMutex, prefillStopMutex;
    std::condition_variable prefillCV;
    std::deque<std::shared_ptr<PrefillRequest>> waiting;
    std::thread prefillThread;
    int batchRequests = 1, batchTokens = 1, targetRequests = 1, waitMs = 0;
    int rankRequests = 1, rankTokens = 1;

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
        const double measured = hostCache ? hostCache->RestoreBytesPerSecond() : 0;
        return measured > 0 ? std::max(1.0, measured) : 1e10;
    }
    size_t ResidentPageTokens() const { return residentPrefix ? std::max(1, GetPageLen()) : 1; }
    bool PreferHostPrefix(size_t resident, const fastkv::HostCache::Hit &hit,
                          double forwardRate, double restoreRate) const {
        return fastkv::PrefillRouter::PreferHostPrefix(resident, hit.Tokens(), ResidentPageTokens(),
                                                       forwardRate, double(hit.Bytes()) / restoreRate);
    }
    std::future<fastkv::Message> Prefill(basellm &model, std::vector<int> input,
                                        std::shared_ptr<fastkv::Cancellation> cancellation, RequestMetadata metadata,
                                        uint32_t kind, std::optional<GenerationConfig> sampling) {
        auto request = std::make_shared<PrefillRequest>();
        request->input = std::move(input);
        request->metadata = std::move(metadata);
        request->sampling = std::move(sampling);
        request->reportUsage = request->sampling.has_value() || kind == PDPrefillWithUsage || kind == PDShardPrefill;
        request->shardAdapter = kind == PDShardPrefill ? &model : nullptr;
        request->cancellation = std::move(cancellation);
        request->work = workload;
        // Keep the lease for the final DP placement. Charge the waiting queue
        // for the tier we expect to use, including a recomputed L1 tail.
        if (hostCache) request->hostHit = hostCache->Find(identity, request->input);
        size_t cached = 0;
        if (request->hostHit) {
            cached = CachedTokens(model, request->input, request->metadata.sessionKey);
            if (PreferHostPrefix(cached, request->hostHit, workload->tokensPerSecond.load(), HostRestoreBytesPerSecond())) {
                cached = request->hostHit.Tokens();
                request->SetRestoreBytes(request->hostHit.Bytes());
            }
        }
        // RPC workers can prepare immutable input while the model lane runs the
        // preceding batch. No cache lookup, page reservation or model hook is
        // performed here; admission observes the latest published state.
        if (preparationSlots && !request->cancellation->Cancelled()) {
            try { request->preparationSlot = preparationSlots->Reserve(1); }
            catch (const fastkv::CapacityBusy &) { }
            if (request->preparationSlot) {
                auto ctx = request->MakeInputContext(model);
                ctx->prefillPreview = std::make_unique<PrefillPrefixPreview>();
                auto &preview = *ctx->prefillPreview;
                preview.Reset(request->input);
                preview.Hashes(ResidentPageTokens());
                preview.TokenHash(request->input.size());
                request->preparedInput = std::move(ctx);
            }
        }
        request->SetRemaining(request->input.size() - cached);
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
    std::function<void(const PDPrefillPageBuilder &, const std::atomic<bool> &, bool)> prepareNext;
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
    bool SelectPeer(PDPending &pending, basellm &model, ResponseContext &context, bool verbose) {
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
        pending.firstToken = probeKind == PDFirstTokenProbe;
        pending.reportUsage = probeKind >= PDUsageProbe;
        pending.sharded = !pending.shardedPeers.empty() && pending.shardedPeers[chosen.peer];
        if (pending.reportUsage || (!pending.sessionKey.empty() && probeKind >= 4)) {
            pending.request.metadata[0] = pending.sharded ? PDShardPrefill : pending.reportUsage ? PDPrefillWithUsage : 3;
            fastkv::Writer suffix; suffix.String(pending.sessionKey);
            if (pending.firstToken) {
                pending.request.metadata[0] = PDFirstTokenPrefill;
                suffix.U32(context.allTokens.back());
                auto sampling = context.generationConfig;
                model.PrepareToolCallConstraint(&context, sampling);
                WriteFirstTokenSampling(suffix, sampling);
                ++pending.length;
            }
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

namespace {
// Completed FULL tables publish in FIFO order within each owner. The scheduler
// checks the producer's ticket before admitting a row that can reuse its prefix.
class PrefixPublicationLane {
    using Request = std::shared_ptr<PDRuntime::PrefillRequest>;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Request> queue;
    size_t submitted = 0, completed = 0;
    bool closing = false;
    std::thread worker;

public:
    template <class Publish>
    size_t Submit(const Request &request, const Publish &publish) {
        if (!worker.joinable()) {
            try {
                worker = std::thread([this, publish] {
                    std::unique_lock<std::mutex> lock(mutex);
                    for (;;) {
                        cv.wait(lock, [&] { return closing || !queue.empty(); });
                        if (queue.empty()) return;
                        Request next = std::move(queue.front());
                        queue.pop_front();
                        lock.unlock();
                        publish(next);
                        lock.lock();
                        ++completed;
                        cv.notify_all();
                    }
                });
            } catch (const std::system_error &) {
                publish(request);
                return 0;
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(request);
        cv.notify_all();
        return ++submitted;
    }

    bool Published(size_t ticket) {
        std::lock_guard<std::mutex> lock(mutex);
        return completed >= ticket;
    }

    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return completed == submitted; });
    }

    ~PrefixPublicationLane() {
        if (!worker.joinable()) return;
        { std::lock_guard<std::mutex> lock(mutex); closing = true; }
        cv.notify_all();
        worker.join();
    }
};
} // namespace

int basellm::PDBranchCheckpointEnd(const Data &key) const {
    if (!pdRuntime) return 0;
    const auto it = pdRuntime->branchCheckpoints.find(&key);
    return it == pdRuntime->branchCheckpoints.end() ? 0 : it->second;
}

bool PDPrefillPagePlan::Valid(int past) const {
    for (const auto &table : tables) {
        const auto &local = *table.cache;
        if (!local.pagedKVCacheData->MatchesPinnedPrefix(local.pageIndex, past / local.pageLen,
                                                         table.prefixGeneration)) return false;
    }
    return true;
}

void PDPrefillPagePlan::Trim(int end) {
    for (const auto &table : tables) {
        auto &local = *table.cache;
        const size_t keep = (size_t(end) + local.pageLen - 1) / local.pageLen;
        while (local.pageIndex.size() > keep) {
            local.pagedKVCacheData->ReleasePageIndex(local.pageIndex.back());
            local.pageIndex.pop_back();
        }
    }
    *this = {};
}

void basellm::RunWithPDPrefillLookahead(const std::function<void()> &forward, const PDPrefillPageBuilder &build,
                                       bool independentFullPublication) {
    if (!pdRuntime || !pdRuntime->prepareNext) { forward(); return; }
    std::atomic<bool> done{false};
    auto &runtime = *pdRuntime;
    auto prepare = std::move(runtime.prepareNext);
    std::future<void> planner;
    try { planner = std::async(std::launch::async, [&] { prepare(build, done, independentFullPublication); }); }
    catch (const std::system_error &) {
        forward();
        return;
    }
    auto finish = [&] {
        { std::lock_guard<std::mutex> lock(runtime.prefillMutex); done = true; }
        runtime.prefillCV.notify_all();
    };
    try { forward(); }
    catch (...) { finish(); planner.wait(); throw; }
    finish();
    planner.get();
}

void PDRuntime::RunPrefill(basellm &model) {
    using Request = std::shared_ptr<PrefillRequest>;
    struct RankQueue {
        std::deque<Request> active, prepared;
        size_t preparedTokens = 0;
        std::weak_ptr<PrefillRequest> branchProducer;
    };
    const int groups = model.PDGroupCount();
    std::vector<RankQueue> ranks(groups);
    // The extra queue holds anonymous RPCs until the model resolves their prefix owner.
    std::vector<std::deque<Request>> pending(groups + 1);
    std::deque<Request> exporting;
    size_t liveContexts = 0;
    uint64_t readyOrder = 0;
    size_t nextPending = 0;
    int firstRank = 0;
    const size_t prepareBudget = tokenCapacity / 2;
    const size_t page = ResidentPageTokens();
    const size_t contextLimit = std::max(batchRequests, model.maxBatch > 0 ? model.maxBatch : 1);
    const size_t rankContextLimit = std::max(size_t(rankRequests), (contextLimit + groups - 1) / groups);
    auto enqueueActive = [&](RankQueue &rank, const Request &request) {
        rank.active.push_back(request);
        request->readyOrder = ++readyOrder;
    };
    struct AdmissionBudget {
        std::vector<int> rows, tokens;
        int totalRows, totalTokens;
        bool Fits(int group) const { return totalRows > 0 && totalTokens > 0 && rows[group] > 0 && tokens[group] > 0; }
        int Charge(int group, int desired) {
            const int count = std::min({desired, tokens[group], totalTokens});
            --rows[group]; --totalRows; tokens[group] -= count; totalTokens -= count;
            return count;
        }
    };
    std::vector<Request> published;
    auto startHostRestore = [&](const Request &request) {
        auto placement = PDReceivePlacement(request->hostHit.Metadata(), request->hostHit.Payloads(), model.PDRequestDevices(*request->context));
        if (placement.empty()) placement.assign(request->hostHit.Payloads(), -1);
        return hostCache->RestoreAsync(request->hostHit, std::move(placement), request->hostRestore,
                                       request->cancellation);
    };
    auto releaseReservation = [&](const Request &request) {
        if (request->reservation) --liveContexts;
        request->reservation = 0;
        request->pages = {};
        request->ahead = false;
        auto context = std::move(request->context);
        if (context) {
            context->isAbort = true;
            model.OnResponseContextRemoved(context.get());
        }
        context.reset();
        request->groupAdmission.reset();
    };
    auto retire = [&](const Request &request) {
        request->SetRemaining(0); request->SetRestoreBytes(0);
        releaseReservation(request);
    };
    auto deferAdmission = [&](const Request &request, size_t lane) {
        releaseReservation(request);
        pending[lane].push_back(request);
    };
    // Only the model lane retires contexts. The export lane owns no model
    // scheduler state and only sees completed, pinned requests.
    std::mutex exportMutex;
    std::condition_variable exportCV;
    std::deque<Request> readyToPack;
    std::thread exportThread;
    std::atomic<bool> closeExports{false};
    uint64_t exportGeneration = 0; // protected by prefillMutex
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
            if (progressed) {
                { std::lock_guard<std::mutex> lock(prefillMutex); ++exportGeneration; }
                prefillCV.notify_one();
            }
            else if (!pending.empty() && !stopping && !closeExports) {
                try { exportPool->WaitForRelease(); }
                catch (const std::runtime_error &) { return; } // StopPrefill closed the pool.
            }
        }
    };
    auto queuePack = [&](const Request &request) {
        { std::lock_guard<std::mutex> lock(exportMutex); readyToPack.push_back(request); }
        exportCV.notify_one();
    };
    auto publish = [&](const Request &request) {
        try {
            const auto begin = std::chrono::steady_clock::now();
            request->context->PublishPagedCacheTables(&model);
            request->prefixRecordMs += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            queuePack(request);
        } catch (...) {
            request->result.set_exception(std::current_exception());
            request->exportFinished.store(true, std::memory_order_release);
        }
    };
    std::vector<PrefixPublicationLane> lanes(std::min(groups, std::max(1, GetThreads())));
    auto laneFor = [&](const Request &request) -> PrefixPublicationLane & {
        return lanes[request->exportGroup % lanes.size()];
    };
    bool hasPublications = false;
    auto finishPublication = [&] {
        if (!hasPublications) return;
        for (auto &lane : lanes) lane.Wait();
        hasPublications = false;
    };
    auto startExport = [&](const Request &request, bool staged) {
        if (staged) {
            const auto begin = std::chrono::steady_clock::now();
            if (model.DeferHistoryCacheRestore()) model.TryRecordResponseContext(request->context.get());
            request->prefixRecordMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        } else recordPrefix(request);
        request->pack = model.PreparePDExport(*request->context);
        if (!request->pack) {
            if (staged) request->context->PublishPagedCacheTables(&model);
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
            if (staged) {
                request->publicationOrder = laneFor(request).Submit(request, publish);
                if (request->publicationOrder) hasPublications = true;
            } else queuePack(request);
        } catch (...) {
            exporting.pop_back();
            throw;
        }
    };
    // The scheduling lane, including its joined lookahead, owns retirement.
    // A background pass handles fully packed sources only. A legacy export
    // that still needs transfer credit stays on the ordinary completion path.
    auto reapExports = [&](bool retryPending) {
        for (auto it = exporting.begin(); it != exporting.end();) {
            const auto request = *it;
            if (request->pack) {
                if (request->exportFinished.load(std::memory_order_acquire)) {
                    retire(request);
                    it = exporting.erase(it);
                } else ++it;
                continue;
            }
            if (!retryPending) { ++it; continue; }
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
    };
    auto drainInput = [&] {
        std::lock_guard<std::mutex> queueLock(prefillMutex);
        while (!waiting.empty()) {
            const auto &request = waiting.front();
            if (request->cancellation->Cancelled()) request->result.set_exception(cancelled);
            else {
                request->affinity = sessions->LocalGroup(request->metadata.sessionKey, groups, true);
                const int lane = request->affinity < 0 ? groups : request->affinity;
                pending[lane].push_back(request);
            }
            waiting.pop_front();
        }
    };
    auto admit = [&](AdmissionBudget *budget) {
        auto canPrepare = [&](size_t group, size_t count) {
            const auto &rank = ranks[group];
            const size_t candidates = rank.active.size() + rank.prepared.size();
            if (budget && !budget->Fits(group)) return false;
            return candidates < size_t(rankRequests) || (residentPrefix && candidates < rankContextLimit &&
                rank.preparedTokens <= prepareBudget && count <= prepareBudget - rank.preparedTokens);
        };
        auto canAdmitFrom = [&](size_t lane, int affinity, size_t count) {
            auto fits = [&](size_t group) {
                return canPrepare(group, count) && count <= tokenCapacity - groupTokens[group]->Used();
            };
            if (affinity >= 0) return fits(affinity);
            // A deferred anonymous request can still move if its owner's KV budget fills.
            if (lane < ranks.size() && !canPrepare(lane, count)) return false;
            for (size_t group = 0; group < ranks.size(); ++group)
                if (fits(group)) return true;
            return false;
        };
        auto fail = [&](const Request &request, std::exception_ptr error) {
            if (budget && !request->cancellation->Cancelled()) {
                try { std::rethrow_exception(error); }
                catch (const CudaWorkspaceExhausted &) {
                    const size_t lane = request->affinity < 0 ? groups : request->affinity;
                    request->branchEnd = 0;
                    for (auto &rank : ranks)
                        if (rank.branchProducer.lock() == request) rank.branchProducer.reset();
                    deferAdmission(request, lane);
                    return;
                } catch (...) {}
            }
            retire(request);
            request->result.set_exception(error);
        };
        std::vector<size_t> toScan;
        size_t outstanding = 0;
        for (auto &queue : pending) {
            for (auto it = queue.begin(); it != queue.end();) {
                if ((*it)->cancellation->Cancelled()) {
                    (*it)->result.set_exception(cancelled);
                    it = queue.erase(it);
                } else ++it;
            }
            toScan.push_back(queue.size());
            outstanding += queue.size();
        }
        // Snapshot unfinished compute once per admission round. New entries
        // join the estimate after cache restoration determines their actual work.
        std::vector<size_t> queuedTokens;
        if (outstanding && groups > 1 && model.prefillScheduling.balanceDP) {
            queuedTokens.resize(groups);
            for (int g = 0; g < groups; ++g)
                for (const auto *queue : {&ranks[g].active, &ranks[g].prepared})
                    for (const auto &item : *queue) queuedTokens[g] += item->remaining;
        }
        std::vector<PDResidentPrefixHints> prefixHints(residentPrefix && outstanding ? groups : 0);
        struct RestoreJob {
            Request request;
            PDResidentRestore restore;
            std::exception_ptr error;
        };
        std::vector<std::vector<RestoreJob>> restores(groups);
        std::function<void(const std::function<void(int)> &)> runRestores;
        // Visit known session owners in turn before allocating a context.
        // A full rank or an oversized head cannot block another rank.
        while (outstanding && liveContexts < contextLimit && !stopping) {
            const size_t lane = nextPending;
            nextPending = (nextPending + 1) % pending.size();
            if (!toScan[lane]) continue;
            --toScan[lane]; --outstanding;
            auto &queue = pending[lane];
            Request request = std::move(queue.front());
            queue.pop_front();
            const size_t count = std::min(tokenCapacity, ((request->input.size() + page - 1) / page) * page);
            if (!canAdmitFrom(lane, request->affinity, count)) {
                queue.push_back(request);
                continue;
            }
            ++liveContexts;
            request->reservation = count;
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
                request->context = request->preparedInput ? std::move(request->preparedInput) :
                    request->MakeInputContext(model);
                request->preparationSlot.reset();
                auto &context = *request->context;
                context.pdPrefillGroup = request->affinity;
                model.OnResponseContextCreated(&context);
                int preferred = -1;
                if (request->affinity < 0 && !queuedTokens.empty()) {
                    std::vector<size_t> available(groups);
                    for (int g = 0; g < groups; ++g)
                        if (canPrepare(g, count)) available[g] = tokenCapacity - groupTokens[g]->Used();
                    const auto cached = model.QueryPagedPrefixCacheExtra(&context, int(request->input.size()));
                    preferred = ChoosePDPrefillGroup(model.PDRequestGroup(context), available,
                        count, queuedTokens, request->input.size(), std::max(0, cached));
                }
                try { request->groupAdmission = ReserveGroup(model, context, count, preferred); }
                catch (const fastkv::CapacityBusy &) {
                    deferAdmission(request, lane);
                    continue;
                }
                const int owner = model.PDRequestGroup(context);
                auto &rank = ranks.at(owner);
                if (!canPrepare(owner, count)) {
                    deferAdmission(request, owner);
                    continue;
                }
                if (hostCache) {
                    auto latest = hostCache->Find(identity, request->input);
                    if (latest.Tokens() > request->hostHit.Tokens()) request->hostHit = std::move(latest);
                }
                const size_t residentTokens = request->hostHit && residentPrefix
                    ? model.PreviewPDResidentPrefix(request->input, model.PDRequestGroup(context)) : hit.tokens;
                if (request->hostHit && PreferHostPrefix(residentTokens, request->hostHit,
                        workload->tokensPerSecond.load(), HostRestoreBytesPerSecond())) {
                    request->restoring = true; request->restoreBegin = restoreBegin;
                    request->SetRestoreBytes(request->hostHit.Bytes());
                    request->SetRemaining(request->input.size() - request->hostHit.Tokens());
                    // Start the private copy before the next forward. A busy
                    // cache retries from prepared; import remains on this lane.
                    startHostRestore(request);
                    rank.prepared.push_back(request); rank.preparedTokens += request->reservation;
                    if (!queuedTokens.empty()) queuedTokens[owner] += request->remaining;
                    continue;
                }
                request->SetRestoreBytes(0);
                request->hostHit = {}; // only an actually restored parent may supply inherited bytes
                if (residentPrefix) {
                    int fullMatch;
                    PDResidentRestore restore;
                    if (!hit.state && !hostCache && groups > 1) {
                        if (!context.prefillPreview) context.prefillPreview = std::make_unique<PrefillPrefixPreview>();
                        context.prefillPreview->restore = &restore;
                    }
                    // The preview may be replaced if a host restore recycles pages.
                    struct ClearRestore {
                        ResponseContext &context;
                        ~ClearRestore() { if (context.prefillPreview) context.prefillPreview->restore = nullptr; }
                    } clear{context};
                    context.preTokens = model.RestorePDResidentPrefix(&context, &fullMatch, &prefixHints[owner]);
                    if (context.preTokens > 0 && restore.work) {
                        if (!runRestores) runRestores = std::move(restore.runRanks);
                        restores[owner].push_back({request, std::move(restore), {}});
                    }
                    // The ordinary end checkpoint already covers the last partial page.
                    // Repair on demand, with at most one producer per attention group.
                    if (context.preTokens == 0 && fullMatch > 0 && request->input.size() - fullMatch > page &&
                        rank.branchProducer.expired() && !context.intParams.count("paged_prefix_restore_disabled")) {
                        request->branchEnd = fullMatch;
                        rank.branchProducer = request;
                    }
                    hit.tokens = context.preTokens;
                }
                if (hit.state) {
                    context.allTokens.resize(hit.tokens);
                    context.allTokens.push_back(0); // adapter validates the N-1 boundary
                    model.ImportPDState(context, *hit.state);
                    context.preTokens = hit.tokens;
                    context.allTokens = request->input;
                }
                request->ahead = budget != nullptr;
                if (budget) budget->Charge(owner, std::min<size_t>(chunk, request->input.size() - context.preTokens));
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
                if (rank.active.size() < size_t(rankRequests)) enqueueActive(rank, request);
                else {
                    rank.prepared.push_back(request);
                    rank.preparedTokens += request->reservation;
                }
                if (!queuedTokens.empty()) queuedTokens[owner] += request->remaining;
            } catch (...) {
                auto error = std::current_exception();
                fail(request, error);
            }
        }
        if (runRestores) {
            // Each request and its source checkpoint stay alive through the
            // owner-stream fence, including requests cancelled during restore.
            try {
                runRestores([&](int owner) {
                    for (auto &job : restores.at(owner)) if (job.request->context) {
                        try { job.restore.work(); }
                        catch (...) { job.error = std::current_exception(); }
                    }
                });
            } catch (...) {
                const auto error = std::current_exception();
                for (auto &jobs : restores) for (auto &job : jobs) job.error = error;
            }
            for (int g = 0; g < groups; ++g) for (auto &job : restores[g]) {
                if (!job.error || !job.request->context) continue;
                auto &rank = ranks[g];
                for (auto *queue : {&rank.active, &rank.prepared}) {
                    auto it = std::find(queue->begin(), queue->end(), job.request);
                    if (it == queue->end()) continue;
                    if (queue == &rank.prepared) rank.preparedTokens -= job.request->reservation;
                    queue->erase(it);
                    break;
                }
                fail(job.request, job.error);
            }
            restores.clear();
        }
    };
    try {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(prefillMutex);
                if (!liveContexts && std::all_of(pending.begin(), pending.end(),
                        [](const auto &queue) { return queue.empty(); })) {
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
            }
            // The model lane owns context mutation and retirement.
            // RPC handlers return futures immediately; network workers never
            // wait for the model or hold forwardLocker.
            std::unique_lock<std::mutex> lock(model.forwardLocker);
            drainInput();
            reapExports(true);
            bool deferredPublication = false;
            for (auto &rank : ranks) for (auto *queue : {&rank.active, &rank.prepared}) {
                for (auto it = queue->begin(); it != queue->end();) {
                    const auto request = *it;
                    if (request->cancellation->Cancelled() || (!request->ahead &&
                        (!hasPublications || request->context->preTokens != request->context->pdCachedInputTokens))) {
                        ++it; continue;
                    }
                    auto &ctx = *request->context;
                    bool stale = !request->pages.Valid(ctx.preTokens);
                    const size_t next = size_t(ctx.preTokens) + page;
                    bool matchingSource = false;
                    if (!stale && next < request->input.size()) for (const auto &previous : published) {
                        if (request->affinity >= 0 && previous->publicationOrder &&
                            previous->exportGroup != request->affinity) continue;
                        if (previous->input.size() < next ||
                            !std::equal(request->input.begin() + ctx.preTokens, request->input.begin() + next,
                                        previous->input.begin() + ctx.preTokens) ||
                            !std::equal(request->input.begin(), request->input.begin() + ctx.preTokens,
                                        previous->input.begin())) continue;
                        if (previous->publicationOrder && !laneFor(previous).Published(previous->publicationOrder)) {
                            // Re-admit this row after its producer publishes. Other
                            // ready rows can enter target while publication finishes.
                            deferredPublication = stale = true;
                            break;
                        }
                        matchingSource = true;
                    }
                    if (!stale && matchingSource)
                        stale = model.PreviewPDResidentPrefix(request->input, request->affinity) > ctx.preTokens;
                    if (!stale) { ++it; continue; }
                    if (queue == &rank.prepared) rank.preparedTokens -= request->reservation;
                    if (rank.branchProducer.lock() == request) rank.branchProducer.reset();
                    request->branchEnd = 0;
                    releaseReservation(request);
                    pending[request->affinity < 0 ? groups : request->affinity].push_front(request);
                    it = queue->erase(it);
                }
            }
            published.clear();
            // Cancellation touches model state only on its execution lane,
            // between completed forward calls. The service drains the future.
            for (auto &rank : ranks) {
                auto &active = rank.active, &prepared = rank.prepared;
                auto &preparedTokens = rank.preparedTokens;
                for (auto it = active.begin(); it != active.end();) {
                    if ((*it)->cancellation->Cancelled()) {
                        retire(*it);
                        (*it)->result.set_exception(cancelled);
                        it = active.erase(it);
                    } else ++it;
                }
                for (auto it = prepared.begin(); it != prepared.end();) {
                    const auto request = *it;
                    try {
                        if (request->cancellation->Cancelled()) std::rethrow_exception(cancelled);
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
                        if (int(active.size()) < rankRequests) {
                            preparedTokens -= request->reservation;
                            enqueueActive(rank, request); it = prepared.erase(it);
                        } else ++it;
                    } catch (...) {
                        const auto error = std::current_exception();
                        preparedTokens -= request->reservation;
                        retire(request); request->result.set_exception(error); it = prepared.erase(it);
                    }
                }
            }
            // Fresh admission must see every publication. If a dependent row
            // was deferred, let any remaining ready row enter the target first.
            const bool dispatchPrepared = hasPublications && (deferredPublication
                ? std::any_of(ranks.begin(), ranks.end(), [](const RankQueue &rank) {
                    return !rank.active.empty();
                })
                : std::all_of(ranks.begin(), ranks.end(), [&](const RankQueue &rank) {
                    return rank.active.size() >= size_t(rankRequests);
                }));
            if (!dispatchPrepared) {
                finishPublication();
                admit(nullptr);
            }
            // Cold requests can reach this queue before any peer publishes FULL KV.
            // Only split a real, page-aligned fork visible in the next forward.
            if (residentPrefix) for (auto &rank : ranks) {
                if (rank.active.size() < 2 || !rank.branchProducer.expired()) continue;
                const auto &producer = rank.active[0], &peer = rank.active[1];
                if (producer->context->preTokens || peer->context->preTokens ||
                    producer->context->intParams.count("paged_prefix_restore_disabled") ||
                    peer->context->intParams.count("paged_prefix_restore_disabled")) continue;
                const auto &a = producer->input, &b = peer->input;
                const size_t limit = std::min({a.size(), b.size(), size_t(chunk), size_t(rankTokens)});
                const size_t common = std::mismatch(a.begin(), a.begin() + limit, b.begin()).first - a.begin();
                const size_t end = common / page * page;
                if (!end || common == limit || std::min(a.size(), b.size()) - end <= page) continue;
                producer->branchEnd = int(end);
                rank.branchProducer = producer;
            }
            std::vector<Request> selected;
            std::vector<int> lengths, sequences(groups), tokens(groups);
            int budget = batchTokens;
            while (budget > 0 && selected.size() < size_t(batchRequests)) {
                int owner = -1;
                for (int offset = 0; offset < groups; ++offset) {
                    const int g = (firstRank + offset) % groups;
                    if (size_t(sequences[g]) >= ranks[g].active.size() || tokens[g] >= rankTokens) continue;
                    if (owner < 0) owner = g;
                    else if (model.prefillScheduling.balanceDP) {
                        if (tokens[g] < tokens[owner]) owner = g;
                    } else if (ranks[g].active[sequences[g]]->readyOrder <
                               ranks[owner].active[sequences[owner]]->readyOrder) owner = g;
                }
                if (owner < 0) break;
                firstRank = (owner + 1) % groups;
                const auto &request = ranks[owner].active[sequences[owner]++];
                const int end = request->branchEnd ? request->branchEnd : int(request->input.size());
                const int count = std::min({chunk, budget, rankTokens - tokens[owner],
                    end - request->context->preTokens});
                // Draft state can span multiple chunks leading up to this checkpoint.
                if (request->branchEnd)
                    branchCheckpoints.emplace(&request->context->pastKeyValues.front().first, end);
                Check(count > 0, "empty prefill candidate");
                selected.push_back(request);
                lengths.push_back(count);
                tokens[owner] += count;
                budget -= count;
            }
            const int batch = selected.size();
            if (!batch) {
                lock.unlock();
                std::unique_lock<std::mutex> queueLock(prefillMutex);
                prefillCV.wait_for(queueLock, std::chrono::milliseconds(1));
                continue;
            }
            for (int i = 0; i < batch; ++i) {
                auto &request = *selected[i];
                const int end = request.context->preTokens + lengths[i];
                request.pages.Trim(end);
                request.ahead = false;
            }
            for (int g = 0; g < groups; ++g) {
                auto discard = [](const Request &r) {
                    if (r->pages.context) r->pages.Trim(r->context->preTokens);
                };
                for (size_t i = sequences[g]; i < ranks[g].active.size(); ++i) discard(ranks[g].active[i]);
                for (const auto &r : ranks[g].prepared) discard(r);
            }
            std::vector<Data> ids(batch), masks(batch), positions(batch);
            std::vector<Data *> maskRefs, positionRefs;
            std::vector<std::pair<Data *, Data *>> past;
            std::vector<GenerationConfig> configs;
            std::vector<float> packed;
            if (batch > 1)
                packed.reserve(batchTokens - budget);
            bool sampleFinalRows = false;
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
                const bool final = selected[i]->sampling && size_t(end) == selected[i]->input.size();
                configs.back().prefill_chunk_remaining = final ? 0 : std::max(1, int(selected[i]->input.size()) - end);
                sampleFinalRows |= final;
                for (auto &kv : context.pastKeyValues) past.emplace_back(&kv.first, &kv.second);
            }
            struct Intermediate {
                basellm &model;
                bool previous;
                Intermediate(basellm &m, bool intermediate) : model(m), previous(m.isIntermediateChunkedPrefill) {
                    model.isIntermediateChunkedPrefill = intermediate;
                }
                ~Intermediate() { model.isIntermediateChunkedPrefill = previous; }
            } intermediate(model, !sampleFinalRows);
            struct ResetLookahead {
                PDRuntime &runtime;
                ~ResetLookahead() { runtime.prepareNext = {}; }
            } reset{*this};
            // Preserve the admission and cache-warming order while a burst
            // ramps up. Use asynchronous retirement only on saturated forwards
            // with existing resident prefixes and no new branch checkpoint.
            bool publishInBackground = false;
            const bool warmBatchFull = (batch == batchRequests || budget == 0) && branchCheckpoints.empty() &&
                std::all_of(selected.begin(), selected.end(), [](const Request &row) {
                    return row->context->pdCachedInputTokens > 0;
                });
            if (batch > 1 && residentPrefix && groups > 1 && !hostCache && !cache) {
                prepareNext = [&](const PDPrefillPageBuilder &build, const std::atomic<bool> &done, bool independentFullPublication) {
                    publishInBackground = independentFullPublication && warmBatchFull;
                    // The target is free to compute; admission for its successor
                    // sees all prefix publications from the previous batch.
                    finishPublication();
                    // The target owns only selected contexts until this callback joins.
                    // Its pages and checkpoint destinations have already been reserved.
                    AdmissionBudget next{std::vector<int>(groups, rankRequests),
                        std::vector<int>(groups, rankTokens), batchRequests, batchTokens};
                    for (int i = 0; i < batch; ++i) {
                        const auto &req = selected[i];
                        const int left = int(req->input.size()) - req->context->preTokens - lengths[i];
                        const int g = model.PDRequestGroup(*req->context);
                        if (left > 0 && next.Fits(g)) next.Charge(g, std::min(left, chunk));
                    }
                    std::vector<size_t> seen(groups);
                    auto planRows = [&] {
                        std::vector<std::vector<PDPrefillPagePlan *>> rows(groups);
                        bool any = false;
                        for (int g = 0; g < groups; ++g) {
                            const auto &rank = ranks[g];
                            const size_t active = rank.active.size() - sequences[g];
                            while (seen[g] < active + rank.prepared.size() && next.Fits(g)) {
                                const size_t i = seen[g]++;
                                const auto &req = i < active ? rank.active[sequences[g] + i] : rank.prepared[i - active];
                                if (req->cancellation->Cancelled() || req->restoring || !req->context) continue;
                                auto &ctx = *req->context;
                                const int count = next.Charge(g, std::min<int>(chunk, req->input.size() - ctx.preTokens));
                                if (!ctx.preTokens || req->branchEnd || !count) continue;
                                req->ahead = true;
                                req->pages.context = &ctx;
                                req->pages.end = ctx.preTokens + count;
                                rows[g].push_back(&req->pages);
                                any = true;
                            }
                        }
                        if (any) build(rows, (rankTokens + int(page) - 1) / int(page) + rankRequests);
                    };
                    planRows(); // Existing candidates precede fresh RPCs, including continuing chunks.
                    std::unique_lock<std::mutex> queueLock(prefillMutex);
                    auto observedExports = exportGeneration;
                    while (!done && !stopping) {
                        queueLock.unlock();
                        bool room = false;
                        for (int g = 0; g < groups; ++g) room |= next.Fits(g);
                        const bool admitInput = room && liveContexts < contextLimit;
                        if (admitInput) {
                            drainInput();
                            auto available = next;
                            admit(&available);
                            planRows();
                        } else {
                            // Reclaim only after the next batch has filled its
                            // budget. Returning grants must not expand this
                            // lookahead's admission window or alter cold routing.
                            if (room || !publishInBackground || exporting.empty()) break;
                            reapExports(false);
                            if (exporting.empty()) break;
                        }
                        queueLock.lock();
                        prefillCV.wait(queueLock, [&] {
                            return done || stopping || (admitInput && !waiting.empty()) ||
                                (!admitInput && exportGeneration != observedExports);
                        });
                        observedExports = exportGeneration;
                    }
                };
            }
            const auto begin = std::chrono::steady_clock::now();
            LastTokensManager last(batch, 0);
            std::vector<int> sampled;
            if (batch == 1) {
                sampled = {model.Forward(ids[0], masks[0], positions[0], selected[0]->context->pastKeyValues,
                              configs[0], last, nullptr)};
            } else {
                Data inputIds(FLOAT32, {1, int(packed.size())}, packed);
                if (model.use_new_engine) {
                    if (model.CanUseGPUForward())
                        sampled = model.ForwardGPU(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
                    else sampled = model.ForwardV2(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
                } else sampled = model.ForwardBatch(batch, inputIds, maskRefs, positionRefs, lengths, past, configs, last);
            }
            if (sampleFinalRows) {
                Check(sampled.size() == selected.size(), "invalid PD first-token batch");
                for (size_t i = 0; i < selected.size(); ++i)
                    if (configs[i].prefill_chunk_remaining == 0) selected[i]->firstToken = sampled[i];
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            if (seconds > 0) workload->tokensPerSecond.store(
                0.9 * workload->tokensPerSecond.load() + 0.1 * (batchTokens - budget) / seconds);
            if (model.verbose) {
                size_t queued;
                { std::lock_guard<std::mutex> lock(prefillMutex); queued = waiting.size(); }
                size_t activeCount = 0, preparedCount = 0, preparedTokens = 0, reserved = 0;
                for (const auto &credits : groupTokens) reserved += credits->Used();
                for (const auto &rank : ranks) {
                    activeCount += rank.active.size(); preparedCount += rank.prepared.size();
                    preparedTokens += rank.preparedTokens;
                }
                for (const auto &queue : pending) queued += queue.size();
                std::cout << "[FastKV PD] prefill_batch=" << batch << " tokens=" << batchTokens - budget
                          << " forward_ms=" << seconds * 1000
                          << " active=" << activeCount << " prepared=" << preparedCount
                          << " exporting=" << exporting.size()
                          << " queued=" << queued << " reserved_tokens=" << reserved
                          << " prepared_tokens=" << preparedTokens << std::endl;
            }
            finishPublication(); // also covers a forward without the lookahead callback
            const bool stageFull = publishInBackground &&
                std::any_of(ranks.begin(), ranks.end(), [](const RankQueue &rank) { return !rank.prepared.empty(); });
            std::vector<Request> repairedBranches;
            for (int i = 0; i < batch; ++i) {
                auto request = selected[i];
                auto &rank = ranks[model.PDRequestGroup(*request->context)];
                rank.active.pop_front();
                request->context->preTokens += lengths[i];
                request->SetRemaining(request->input.size() - request->context->preTokens);
                if (request->cancellation->Cancelled()) {
                    retire(request);
                    request->result.set_exception(cancelled);
                    continue;
                }
                if (request->branchEnd && request->context->preTokens == request->branchEnd) {
                    request->branchEnd = 0;
                    rank.branchProducer.reset();
                    // Peers publish FULL pages through their ordinary completion
                    // path. Adding a window must not rebind their page chains.
                    model.TryRecordResponseContext(request->context.get());
                    published.push_back(request);
                    repairedBranches.push_back(request);
                }
                if (size_t(request->context->preTokens) != request->input.size()) {
                    enqueueActive(rank, request); // round-robin chunks, no long-request monopolization
                    continue;
                }
                try {
                    startExport(request, stageFull);
                    if (residentPrefix && groups > 1) published.push_back(request);
                } catch (...) {
                    auto error = std::current_exception();
                    retire(request);
                    request->result.set_exception(error);
                }
            }
            branchCheckpoints.clear();
            // A published window changes the budget only for matching requests that
            // have not forwarded yet. Readmit to release any shorter restored state.
            for (const auto &source : repairedBranches) {
                const int owner = model.PDRequestGroup(*source->context);
                auto &rank = ranks[owner];
                const int length = source->context->preTokens;
                if (model.PreviewPDResidentPrefix(source->input, owner) < length) continue;
                std::deque<Request> retry;
                for (auto *queue : {&rank.active, &rank.prepared}) {
                    for (auto it = queue->begin(); it != queue->end();) {
                        const auto &r = *it;
                        const auto &ctx = *r->context;
                        if (r->restoring || r->hostHit || ctx.preTokens != ctx.pdCachedInputTokens ||
                            ctx.preTokens >= length || r->input.size() <= size_t(length) ||
                            !std::equal(source->input.begin(), source->input.begin() + length, r->input.begin())) {
                            ++it;
                            continue;
                        }
                        if (queue == &rank.prepared) rank.preparedTokens -= r->reservation;
                        releaseReservation(r);
                        r->affinity = owner;
                        retry.push_back(r);
                        it = queue->erase(it);
                    }
                }
                pending[owner].insert(pending[owner].begin(), retry.begin(), retry.end());
            }
        }
        throw std::runtime_error("FastKV PD: prefill cancelled by shutdown");
    } catch (...) {
        const auto error = std::current_exception();
        // Drain publications before destroying the export lane or retiring sources.
        lanes.clear();
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
        for (const auto &queue : pending)
            for (const auto &request : queue) request->result.set_exception(error);
        for (const auto &rank : ranks)
            for (const auto *queue : {&rank.active, &rank.prepared})
                for (const auto &request : *queue) {
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
        std::lock_guard<std::mutex> lock(forwardLocker);
        pdRuntime->receivers.clear();
    }
}

PDPrefillLimits basellm::GetPDPrefillLimits(const json11::Json &config) {
    PDPrefillLimits limits{};
    limits.chunk = config["chunk_tokens"].is_number() ? config["chunk_tokens"].int_value() : 2048;
    Check(limits.chunk > 0 && limits.chunk <= max_positions, "invalid prefill chunk size");
    const int concurrent = maxBatch > 0 ? maxBatch : 1;
    const bool batching = SupportsPrefillBatching() && canDoBatchForward;
    limits.batchRequests = batching ? concurrent : 1;
    if (prefillBatchPolicy.maxRequests > 0)
        limits.batchRequests = std::min(limits.batchRequests, prefillBatchPolicy.maxRequests);
    limits.batchTokens = batching ? std::max(limits.chunk, GetBatchedPrefillTokenLimit()) : limits.chunk;
    if (batching && prefillBatchPolicy.targetTokens > 0)
        limits.batchTokens = std::min(limits.batchTokens, prefillBatchPolicy.targetTokens);
    auto rankLimit = [&](const char *key, int fallback) {
        if (!config.object_items().count(key)) return fallback;
        const auto &value = config[key];
        const double n = value.number_value();
        if (!value.is_number() || n < 1 || n > INT_MAX || n != int(n))
            throw std::invalid_argument(std::string(key) + " must be a positive 32-bit integer");
        return int(n);
    };
    limits.workspaceLimitMiB = rankLimit("max_workspace_mib_per_rank", 0);
    limits.rankRequests = rankLimit("max_requests_per_rank", limits.batchRequests);
    limits.rankTokens = rankLimit("max_new_tokens_per_rank", prefillScheduling.balanceDP ?
        std::max(1, limits.batchTokens / PDGroupCount()) : limits.batchTokens);
    return limits;
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
    Check(runtime->role == "prefill" || !config.object_items().count("max_workspace_mib_per_rank"),
          "max_workspace_mib_per_rank is only valid for prefill");
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
            const auto limits = GetPDPrefillLimits(config);
            runtime->chunk = limits.chunk;
            runtime->batchRequests = limits.batchRequests;
            runtime->batchTokens = limits.batchTokens;
            runtime->rankRequests = limits.rankRequests;
            runtime->rankTokens = limits.rankTokens;
            if (runtime->residentPrefix && limits.batchRequests > 1)
                runtime->preparationSlots = std::make_unique<fastkv::Budget>(limits.batchRequests, "prefill input preparation");
            runtime->prefillReservations = ActivatePDPrefillMemory(limits);
            Check(!limits.workspaceLimitMiB || !runtime->prefillReservations.empty(),
                "max_workspace_mib_per_rank requires a warmed GPU Naive attention-DP prefill worker without draft");
            const bool batching = SupportsPrefillBatching() && canDoBatchForward;
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
                Check((kind == 1 || kind == 3 || kind == PDPrefillWithUsage || kind == PDShardPrefill || kind == PDFirstTokenPrefill) &&
                      raw->Accepts(requested) && ((kind == PDShardPrefill) == (requested != raw->identity)),
                      "incompatible checkpoint or state layout");
                auto tokens = reader.Ints(size_t(max_positions));
                RequestMetadata metadata;
                if (kind != 1) metadata.sessionKey = reader.String(512);
                std::optional<GenerationConfig> sampling;
                if (kind == PDFirstTokenPrefill) {
                    Check(SupportsPDFirstToken() && raw->residentPrefix && !raw->hostCache && !raw->cache,
                          "sampled PD handoff unsupported");
                    const auto last = reader.U32();
                    Check(last <= INT_MAX, "invalid final prompt token");
                    tokens.push_back(last);
                    sampling = ReadFirstTokenSampling(reader);
                }
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                return raw->Prefill(*this, std::move(tokens), std::move(cancellation), std::move(metadata), kind, std::move(sampling));
            }, [this, raw = runtime.get()](const fastkv::Message &request, fastkv::Message &reply) {
                if (request.metadata.size() < 4) return false;
                fastkv::Reader reader(request.metadata);
                const auto kind = reader.U32();
                if (kind != 2 && kind != 4 && kind != 6 && kind != 7 && kind != 8 && kind != PDUsageProbe && kind != PDFirstTokenProbe) return false;
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
                if (kind == PDFirstTokenProbe)
                    Check(SupportsPDFirstToken() && raw->residentPrefix && !raw->hostCache && !raw->cache &&
                          tokens.size() < raw->tokenCapacity, "sampled PD handoff unsupported");
                fastkv::Writer writer;
                const size_t resident = raw->CachedTokens(*this, tokens, session);
                const auto forwardRate = uint64_t(std::max(1.0, raw->workload->tokensPerSecond.load()));
                writer.U32(kind); writer.U64(resident);
                writer.U64(raw->workload->tokens.load());
                writer.U64(forwardRate);
                if (kind >= 6) {
                    const auto record = raw->sessions->Get(session);
                    writer.U64(record.version); writer.String(record.owner);
                }
                if (kind >= 8) {
                    const auto hit = raw->hostCache ? raw->hostCache->Find(raw->identity, tokens) : fastkv::HostCache::Hit{};
                    const auto restoreRate = uint64_t(raw->HostRestoreBytesPerSecond());
                    // The producer knows its page size. Filtering the quote here
                    // works with older decoders and different P/D page layouts.
                    const bool useHost = raw->PreferHostPrefix(resident, hit, forwardRate, restoreRate);
                    writer.U64(useHost ? hit.Tokens() : 0); writer.U64(useHost ? hit.Bytes() : 0);
                    writer.U64(raw->workload->restoreBytes.load());
                    writer.U64(restoreRate);
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
    const auto &config = context->generationConfig;
    const bool canSample = SupportsPDFirstToken() && !config.output_logits && config.output_token_least <= 0;
    pending->probeKinds.resize(pdRuntime->peers.size(), canSample ? PDFirstTokenProbe : PDUsageProbe);
    for (auto &e : pending->estimates) e.available = false;
    pending->request.metadata = std::move(writer.bytes);
    context->pdPending = std::move(pending);
}

void basellm::PreparePendingPD(ResponseContext *context) {
    auto pending = context->pdPending;
    if (pdRuntime->stopping) return;
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
        if (!pdRuntime->SelectPeer(*pending, *this, *context, verbose)) return;
        if (!pending->receiver) {
            const auto begin = std::chrono::steady_clock::now();
            pending->receiver = PreparePDReceive(*context);
            if (pending->receiver) pdRuntime->receivers.push_back(pending->receiver);
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
            if (pending->reportUsage) target = WithoutPDUsageTrailer(std::move(target), pending->firstToken ? 8 : 4);
        } else {
            placement = [devices = PDRequestDevices(*context)](const fastkv::Bytes &metadata,
                                                                const std::vector<size_t> &sizes) {
                return PDReceivePlacement(metadata, sizes.size(), devices);
            };
            if (pending->reportUsage) placement = WithoutPDUsageTrailer(std::move(placement), pending->firstToken ? 8 : 4);
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

void basellm::FlushPDAnnouncements() {
    if (pdRuntime) pdRuntime->FlushAnnouncements();
}

bool basellm::ReapPDReceivers() {
    if (!IsPDDecode()) return false;
    std::lock_guard<std::mutex> lock(forwardLocker);
    auto &receivers = pdRuntime->receivers;
    receivers.erase(std::remove_if(receivers.begin(), receivers.end(), [](const auto &receiver) {
        if (receiver.use_count() != 1) return false;
        receiver->Recycle();
        return true;
    }), receivers.end());
    return !receivers.empty();
}

bool basellm::PollPD(ResponseContext *context) {
    ReapPDReceivers();
    FlushPDAnnouncements();
    return PollPendingPD(context, 0);
}

bool basellm::PollPendingPD(ResponseContext *context, int maxTotalLens) {
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
        int cachedTokens = -1, firstToken = -1;
        if (pending->reportUsage) {
            Check(state.metadata.size() >= sizeof(uint32_t), "missing PD cache usage");
            const size_t size = state.metadata.size() - sizeof(uint32_t);
            const auto count = fastkv::Reader(state.metadata.data() + size, sizeof(uint32_t)).U32();
            Check(count <= uint32_t(pending->length), "invalid PD cached token count");
            cachedTokens = int(count);
            state.metadata.resize(size);
            if (pending->firstToken) {
                Check(size >= 4, "missing sampled PD first token");
                const auto token = fastkv::Reader(state.metadata.data() + size - 4, 4).U32();
                Check(token <= INT_MAX, "invalid sampled PD token");
                firstToken = int(token);
                state.metadata.resize(size - 4);
            }
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
        if (!pending->firstToken) context->currentTokens = {context->allTokens.back()};
        context->intParams = {
            {"promptLen", context->inputTokens}, {"index", pending->firstToken ? 0 : -1}, {"add_special_tokens", 0}};
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
        if (pending->firstToken) CommitGeneratedToken(*context, firstToken, maxTotalLens);
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
