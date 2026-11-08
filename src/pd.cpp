#include "pd.h"
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
};
PDStateWriter::PDStateWriter(fastkv::BufferPool *pool)
    : pool(pool), allocations(std::make_shared<Allocations>()) {}
PDStateWriter PDStateWriter::Fork() const {
    PDStateWriter child(pool);
    child.allocations = allocations;
    return child;
}
fastkv::Buffer PDStateWriter::Allocate(size_t bytes, int device) {
    if (!pool) return fastkv::Buffer::Allocate(bytes, device);
    const size_t aligned = (arenaOffset + 255) / 256 * 256;
    if (!arena.Size() || arena.Device() != device || aligned > arena.Size() || bytes > arena.Size() - aligned) {
        const size_t capacity = std::max(bytes, size_t(64) << 20);
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
    std::string peer;
    std::shared_ptr<Admission> admission;
    std::shared_ptr<fastkv::Cancellation> cancellation = std::make_shared<fastkv::Cancellation>();
    int length = 0;
    size_t reservation = 0;
    bool committed = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~PDPending() { cancellation->Cancel(); }
};
struct PDRuntime {
    struct PrefillRequest {
        std::vector<int> input;
        std::promise<fastkv::Message> result;
        std::shared_ptr<fastkv::Cancellation> cancellation;
        std::unique_ptr<ResponseContext> context;
        bool reserved = false;
        std::chrono::steady_clock::time_point queued = std::chrono::steady_clock::now();
    };
    std::unique_ptr<fastkv::Service> service;
    std::string role, identity;
    std::vector<std::string> peers;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> stopping{false};
    int chunk = 2048;
    size_t tokenCapacity = 0;
    std::unique_ptr<fastkv::Budget> slots, tokens;
    std::unique_ptr<fastkv::PrefixCache> cache;
    std::unique_ptr<fastkv::BufferPool> exportPool;
    bool residentPrefix = false;
    // Routing stores only token keys and an endpoint, never KV payloads.
    std::unique_ptr<fastkv::PrefixCache> routes;
    std::mutex prefillMutex, prefillStopMutex;
    std::condition_variable prefillCV;
    std::deque<std::shared_ptr<PrefillRequest>> waiting;
    std::thread prefillThread;
    int batchRequests = 1, batchTokens = 1, targetRequests = 1, waitMs = 0;

    ~PDRuntime() {
        StopPrefill();
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
    std::future<fastkv::Message> Prefill(std::vector<int> input,
                                       std::shared_ptr<fastkv::Cancellation> cancellation) {
        auto request = std::make_shared<PrefillRequest>();
        request->input = std::move(input);
        request->cancellation = std::move(cancellation);
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
};

void PDRuntime::RunPrefill(basellm &model) {
    using Request = std::shared_ptr<PrefillRequest>;
    std::deque<Request> active;
    size_t reserved = 0;
    auto retire = [&](const Request &request) {
        if (request->reserved) {
            reserved -= request->input.size();
            request->reserved = false;
        }
        auto context = std::move(request->context);
        if (context) {
            context->isAbort = true;
            model.OnResponseContextRemoved(context.get());
        }
    };
    const auto cancelled = std::make_exception_ptr(std::runtime_error("FastKV PD: request cancelled"));
    try {
        while (true) {
            std::vector<Request> admitted;
            {
                std::unique_lock<std::mutex> lock(prefillMutex);
                if (active.empty()) {
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
                while (!waiting.empty() && int(active.size() + admitted.size()) < batchRequests &&
                       waiting.front()->input.size() <= tokenCapacity - reserved) {
                    auto request = waiting.front(); waiting.pop_front();
                    reserved += request->input.size();
                    request->reserved = true;
                    admitted.push_back(std::move(request));
                }
            }
            // Only the model execution lane touches contexts and GPU state.
            // RPC handlers return futures immediately; network workers never
            // wait for the model or hold forwardLocker.
            std::lock_guard<std::mutex> lock(model.forwardLocker);
            // Cancellation touches model state only on its execution lane,
            // between completed forward calls. The service drains the future.
            for (auto it = active.begin(); it != active.end();) {
                if ((*it)->cancellation->Cancelled()) {
                    retire(*it);
                    (*it)->result.set_exception(cancelled);
                    it = active.erase(it);
                } else ++it;
            }
            for (const auto &request : admitted) {
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
                    context.currentTokens = context.allTokens = request->input;
                    context.inputTokens = request->input.size();
                    context.generationConfig.top_k = 1;
                    context.generationConfig.output_token_limit = 1;
                    context.generationConfig.input_token_length = request->input.size() + 1;
                    model.OnResponseContextCreated(&context);
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
                    if (model.verbose) {
                        const auto stats = cache ? cache->Stats() : fastkv::PrefixCache::Statistics{};
                        std::cout << "[FastKV PD] prefix_hit=" << hit.tokens << " prompt=" << request->input.size()
                                  << " cache_entries=" << stats.entries << " cache_bytes=" << stats.bytes
                                  << " cache_logical_bytes=" << stats.logicalBytes
                                  << " resident_gpu=" << residentPrefix << std::endl;
                        std::cout << "[FastKV PD timing] restore_ms="
                                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - restoreBegin).count()
                                  << std::endl;
                    }
                    active.push_back(request);
                } catch (...) {
                    auto error = std::current_exception();
                    retire(request);
                    request->result.set_exception(error);
                }
            }
            if (active.empty()) continue;

            std::vector<Request> selected;
            std::vector<int> lengths;
            int budget = batchTokens;
            for (const auto &request : active) {
                if (!budget) break;
                const int count = std::min({chunk, budget,
                    int(request->input.size()) - request->context->preTokens});
                selected.push_back(request); lengths.push_back(count); budget -= count;
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
            if (model.verbose) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                std::cout << "[FastKV PD] prefill_batch=" << batch << " tokens=" << batchTokens - budget
                          << " forward_ms=" << seconds * 1000 << std::endl;
            }
            for (int i = 0; i < batch; ++i) {
                auto request = active.front(); active.pop_front();
                request->context->preTokens += lengths[i];
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
                    const auto exportBegin = std::chrono::steady_clock::now();
                    if (residentPrefix) request->context->TryRecordPagedCache(&model);
                    const auto recorded = std::chrono::steady_clock::now();
                    auto reply = model.ExportPDState(*request->context);
                    const auto exported = std::chrono::steady_clock::now();
                    if (cache) {
                        try { cache->Publish(identity, request->input, reply); }
                        catch (const std::exception &error) {
                            if (model.verbose) std::cerr << "[FastKV PD] prefix not retained: " << error.what() << std::endl;
                        }
                    }
                    const auto published = std::chrono::steady_clock::now();
                    retire(request);
                    if (model.verbose) {
                        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                        std::cout << "[FastKV PD timing] record_ms=" << ms(exportBegin, recorded)
                                  << " export_ms=" << ms(recorded, exported)
                                  << " publish_ms=" << ms(exported, published)
                                  << " retire_ms=" << ms(published, std::chrono::steady_clock::now())
                                  << " kv_d2h_bytes=" << fastkv::Buffer::Traffic().deviceToHost
                                  << " kv_h2d_bytes=" << fastkv::Buffer::Traffic().hostToDevice << std::endl;
                    }
                    request->result.set_value(std::move(reply));
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
    }
}

std::string basellm::PDLayout() const {
    throw std::invalid_argument("FastKV PD: this model has no state adapter");
}
fastkv::Message basellm::ExportPDState(ResponseContext &) { throw std::logic_error("missing PD adapter"); }
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
    Check(!kvMemConfig.enabled, "the sparse/offloaded KV adapter is not implemented");
    const int concurrent = maxBatch > 0 ? maxBatch : 1;
    runtime->tokenCapacity =
        GetMaxTokens() > 0 ? size_t(GetMaxTokens()) : size_t(max_positions) * concurrent;
    if (tokensLimit > 0)
        runtime->tokenCapacity = std::min(runtime->tokenCapacity, size_t(tokensLimit));
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
    if (stagingCapacity) options.maxInflightBytes = std::min(options.maxInflightBytes, stagingCapacity);
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
        runtime->routes.reset(new fastkv::PrefixCache(64 << 20, std::max(64, concurrent * 4)));
        runtime->slots.reset(new fastkv::Budget(concurrent, "decode request slots"));
        runtime->tokens.reset(new fastkv::Budget(runtime->tokenCapacity, "decode token budget"));
    } else {
        if (config["chunk_tokens"].is_number())
            runtime->chunk = config["chunk_tokens"].int_value();
        Check(runtime->chunk > 0 && runtime->chunk <= max_positions, "invalid prefill chunk size");
        const double cacheBytes = config["cache_bytes"].number_value();
        Check(cacheBytes >= 0 && cacheBytes <= double(size_t(1) << 40), "invalid prefix cache byte limit");
        if (!runtime->residentPrefix) {
            const double deviceBytes = config["device_cache_bytes"].is_number() ?
                config["device_cache_bytes"].number_value() : double(size_t(1) << 30);
            Check(deviceBytes >= 0 && deviceBytes <= double(size_t(1) << 40), "invalid device cache byte limit");
            if (cacheBytes > 0) runtime->cache.reset(new fastkv::PrefixCache(size_t(cacheBytes)));
            else if (deviceBytes > 0) runtime->cache.reset(new fastkv::PrefixCache(size_t(deviceBytes), 0, -2));
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
                Check(reader.U32() == 1 && reader.String() == raw->identity,
                      "incompatible checkpoint or state layout");
                auto tokens = reader.Ints(size_t(max_positions));
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                return raw->Prefill(std::move(tokens), std::move(cancellation));
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
    auto route = pdRuntime->routes->Find(pdRuntime->identity, prefix, prefix.size());
    if (route.state) {
        fastkv::Reader chosen(route.state->metadata);
        pending->peer = chosen.String(); chosen.End();
    } else pending->peer = pdRuntime->peers[pdRuntime->next.fetch_add(1) % pdRuntime->peers.size()];
    fastkv::Writer chosen; chosen.String(pending->peer);
    try { pdRuntime->routes->Publish(pdRuntime->identity, prefix, {std::move(chosen.bytes), {}}); }
    catch (const std::runtime_error &) { /* Routing history is an optional bounded hint. */ }
    pending->request.metadata = std::move(writer.bytes);
    context->pdPending = std::move(pending);
}

bool basellm::PollPD(ResponseContext *context) {
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
                admission->tokens = pdRuntime->tokens->Reserve(pending->reservation);
                pending->admission = std::move(admission);
            }
        } catch (const fastkv::CapacityBusy &) {
            return false;
        }
        try {
            // Transient transport pressure must preserve the prompt for retry.
            // The Exchange also holds admission until cancellation/I/O drains.
            pdRuntime->service->TrySubmit(pending->peer, pending->request, pending->result,
                                          pending->cancellation, pending->admission);
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
        std::lock_guard<std::mutex> lock(forwardLocker);
        const auto importBegin = std::chrono::steady_clock::now();
        ImportPDState(*context, state);
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
    } catch (const std::exception &error) {
        std::cerr << "[FastKV PD] request failed: " << error.what() << std::endl;
        context->error = ResponseContextErrorRuntime;
        context->isEnding = true;
    }
    if (context->isEnding)
        context->pdPending.reset();
    dictCV.notify_all();
    return !context->isEnding;
}
} // namespace fastllm
