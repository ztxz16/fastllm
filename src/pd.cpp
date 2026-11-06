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
    Check(data.unitSizeDiv == 1, "packed element type requires a model wire codec");
    return elements * data.unitSize;
}
} // namespace

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
    auto output = fastkv::Buffer::Allocate(bytes, device);
    auto sourceLease = std::shared_ptr<void>(const_cast<Data *>(&data), [](void *) {});
    auto input = fastkv::Buffer::Borrow(
        data.dataDevice == CUDA ? data.cudaData : data.cpuData,
        std::max<uint64_t>(bytes, data.expansionBytes),
        data.dataDevice == CUDA ? fastkv::MemoryKind::Cuda : fastkv::MemoryKind::Host, device, sourceLease);
    const size_t rowBytes = size_t(data.dims.back()) * data.unitSize;
    Check(rowBytes > 0 && data.strides.size() == data.dims.size(), "invalid tensor strides");
    const uint64_t rows = bytes / rowBytes;
    uint64_t stride = 1;
    bool contiguous = true;
    for (int axis = int(data.dims.size()) - 1; axis >= 0; --axis) {
        contiguous &= data.dims[axis] == 1 || data.strides[axis] == stride;
        stride *= data.dims[axis];
    }
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
fastkv::Message PDStateWriter::Finish() {
    fastkv::Message result{std::move(metadata.bytes), std::move(buffers)};
    return result.buffers.size() > 1 ? fastkv::Snapshot(result) : std::move(result);
}
PDStateReader::PDStateReader(const fastkv::Message &value) : metadata(value.metadata), message(value) {}
void PDStateReader::Tensor(Data &tensor) {
    const auto type = DataType(metadata.U32());
    Check(type == FLOAT32 || type == FLOAT16 || type == BFLOAT16 || type == INT8 || type == FP8_E4M3 ||
              type == INT32PARAM,
          "unsupported tensor wire type");
    auto dims = metadata.Ints(8);
    const bool fp8Index = metadata.U32() != 0;
    const uint64_t bytes = metadata.U64();
    Data value(type);
    value.Resize(dims);
    Check(TensorBytes(value) == bytes, "tensor size mismatch");
    if (bytes) {
        Check(next < message.buffers.size() && message.buffers[next].Size() == bytes,
              "tensor payload mismatch");
        value.Allocate();
        message.buffers[next++].Read(value.cpuData, 0, bytes);
    }
    value.kvCacheFp8Indexer = fp8Index;
    tensor.CopyFrom(value);
    tensor.SetKVCache();
}
void PDStateReader::End() {
    metadata.End();
    Check(next == message.buffers.size(), "unclaimed payload");
}

struct PDPending {
    std::future<fastkv::Message> result;
    fastkv::Message request;
    std::string peer;
    std::shared_ptr<void> slot, tokens;
    int length = 0;
    size_t reservation = 0;
    bool committed = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
struct PDRuntime {
    struct PrefillRequest {
        std::vector<int> input;
        std::promise<fastkv::Message> result;
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
    std::mutex prefillMutex, prefillStopMutex;
    std::condition_variable prefillCV;
    std::deque<std::shared_ptr<PrefillRequest>> waiting;
    std::thread prefillThread;
    int batchRequests = 1, batchTokens = 1, targetRequests = 1, waitMs = 0;

    ~PDRuntime() { StopPrefill(); }
    void StopPrefill() {
        std::lock_guard<std::mutex> stoppingLock(prefillStopMutex);
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            stopping = true;
        }
        prefillCV.notify_all();
        if (prefillThread.joinable()) prefillThread.join();
    }
    fastkv::Message Prefill(std::vector<int> input) {
        auto request = std::make_shared<PrefillRequest>();
        request->input = std::move(input);
        auto result = request->result.get_future();
        {
            std::lock_guard<std::mutex> lock(prefillMutex);
            Check(!stopping, "prefill is stopping");
            waiting.push_back(request);
        }
        prefillCV.notify_one();
        return result.get();
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
                while (!waiting.empty() && int(active.size() + admitted.size()) < batchRequests &&
                       waiting.front()->input.size() <= tokenCapacity - reserved) {
                    auto request = waiting.front(); waiting.pop_front();
                    reserved += request->input.size();
                    request->reserved = true;
                    admitted.push_back(std::move(request));
                }
            }
            // Only the model execution lane touches contexts and GPU state.
            // RPC handlers wait on their independent result; they never hold
            // forwardLocker while waiting for another request to arrive.
            std::lock_guard<std::mutex> lock(model.forwardLocker);
            for (const auto &request : admitted) {
                try {
                    fastkv::PrefixCache::Hit hit;
                    if (cache) hit = cache->Find(identity, request->input, request->input.size());
                    if (model.verbose) {
                        const auto stats = cache ? cache->Stats() : fastkv::PrefixCache::Statistics{};
                        std::cout << "[FastKV PD] prefix_hit=" << hit.tokens << " prompt=" << request->input.size()
                                  << " cache_entries=" << stats.entries << " cache_bytes=" << stats.bytes
                                  << " cache_logical_bytes=" << stats.logicalBytes << std::endl;
                    }
                    if (hit.state && hit.tokens == request->input.size()) {
                        retire(request);
                        request->result.set_value(*hit.state);
                        continue;
                    }
                    request->context.reset(new ResponseContext);
                    auto &context = *request->context;
                    context.Init(model.block_cnt, model.dataType, model.kvCacheDataType);
                    context.allTokens = request->input;
                    context.inputTokens = request->input.size();
                    context.generationConfig.top_k = 1;
                    context.generationConfig.output_token_limit = 1;
                    context.generationConfig.input_token_length = request->input.size() + 1;
                    model.OnResponseContextCreated(&context);
                    if (hit.state) {
                        context.allTokens.resize(hit.tokens);
                        context.allTokens.push_back(0); // adapter validates the N-1 boundary
                        model.ImportPDState(context, *hit.state);
                        context.preTokens = hit.tokens;
                        context.allTokens = request->input;
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
            int remaining = int(std::min(size_t(batchTokens), active.size()));
            for (const auto &request : active) {
                if (!remaining) break;
                const int count = std::min({chunk, budget / remaining,
                    int(request->input.size()) - request->context->preTokens});
                selected.push_back(request); lengths.push_back(count); budget -= count;
                --remaining;
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
                if (size_t(request->context->preTokens) != request->input.size()) {
                    active.push_back(request); // round-robin chunks, no long-request monopolization
                    continue;
                }
                try {
                    auto reply = model.ExportPDState(*request->context);
                    if (cache) {
                        try { cache->Publish(identity, request->input, reply); }
                        catch (const std::exception &error) {
                            if (model.verbose) std::cerr << "[FastKV PD] prefix not retained: " << error.what() << std::endl;
                        }
                    }
                    retire(request);
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
    runtime->identity = config["model_id"].string_value() + ":" + PDLayout();
    Check(!UseModelSpecificScheduler(), "PD requires a supported common scheduler");
    Check(!kvMemConfig.enabled, "the sparse/offloaded KV adapter is not implemented");
    const int concurrent = maxBatch > 0 ? maxBatch : 1;
    runtime->tokenCapacity =
        GetMaxTokens() > 0 ? size_t(GetMaxTokens()) : size_t(max_positions) * concurrent;
    fastkv::Options options;
    options.workers = std::max(4, std::min(concurrent, 16));
    const std::string transport = config["transport"].string_value();
    Check(transport == "tcp" || transport == "rdma" || transport == "auto", "unknown transport");
    options.transport = transport == "rdma"   ? fastkv::Transport::Rdma
                        : transport == "auto" ? fastkv::Transport::Auto
                                              : fastkv::Transport::Tcp;
    for (const auto &device : config["rdma_devices"].array_items())
        options.rdmaDevices.push_back(device.string_value());
    // Model adapters currently import through host staging. GPU transport
    // remains available in the independent component, not as a model knob.
    if (config["max_inflight_bytes"].is_number()) {
        double limit = config["max_inflight_bytes"].number_value();
        Check(limit >= (1 << 20) && limit <= double(size_t(1) << 40), "invalid inflight byte limit");
        options.maxInflightBytes = size_t(limit);
        options.maxPayloadBytes = size_t(limit);
    }
    runtime->service.reset(new fastkv::Service(options));
    for (const auto &peer : config["prefill"].array_items()) {
        Check(peer.is_string() && !peer.string_value().empty(), "invalid prefill endpoint");
        runtime->peers.push_back(peer.string_value());
    }
    if (runtime->role == "decode") {
        Check(!runtime->peers.empty(), "decode needs a prefill execution group");
        runtime->slots.reset(new fastkv::Budget(concurrent));
        runtime->tokens.reset(new fastkv::Budget(runtime->tokenCapacity));
    } else {
        if (config["chunk_tokens"].is_number())
            runtime->chunk = config["chunk_tokens"].int_value();
        Check(runtime->chunk > 0 && runtime->chunk <= max_positions, "invalid prefill chunk size");
        const double cacheBytes = config["cache_bytes"].number_value();
        Check(cacheBytes >= 0 && cacheBytes <= double(size_t(1) << 40), "invalid prefix cache byte limit");
        if (cacheBytes > 0)
            runtime->cache.reset(new fastkv::PrefixCache(size_t(cacheBytes)));
    }
    pdRuntime = runtime;
    try {
        if (runtime->role == "prefill") {
            Check(!config["listen"].string_value().empty(), "prefill listen endpoint is required");
            const bool batching = SupportsPrefillBatching() && canDoBatchForward;
            runtime->batchRequests = batching ? std::min(concurrent, options.workers) : 1;
            if (prefillBatchPolicy.maxRequests > 0)
                runtime->batchRequests = std::min(runtime->batchRequests, prefillBatchPolicy.maxRequests);
            runtime->batchTokens = batching ? std::max(runtime->chunk, GetBatchedPrefillTokenLimit()) : runtime->chunk;
            runtime->targetRequests = prefillBatchPolicy.targetRequests > 0 ?
                std::min(runtime->batchRequests, prefillBatchPolicy.targetRequests) : runtime->batchRequests;
            runtime->waitMs = batching ? prefillBatchPolicy.maxWaitMs : 0;
            runtime->prefillThread = std::thread([this, raw = runtime.get()] { raw->RunPrefill(*this); });
            runtime->service->Start(config["listen"].string_value(), [this, raw = runtime.get()](
                                                                         const fastkv::Message &request) {
                fastkv::Reader reader(request.metadata);
                Check(reader.U32() == 1 && reader.String() == raw->identity,
                      "incompatible checkpoint or state layout");
                auto tokens = reader.Ints(size_t(max_positions));
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                return raw->Prefill(std::move(tokens));
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
    pending->peer = pdRuntime->peers[pdRuntime->next.fetch_add(1) % pdRuntime->peers.size()];
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
            auto slot = pdRuntime->slots->Reserve(1);
            auto tokens = pdRuntime->tokens->Reserve(pending->reservation);
            pending->slot = std::move(slot);
            pending->tokens = std::move(tokens);
        } catch (const std::runtime_error &) {
            return false;
        }
        try {
            pending->result = pdRuntime->service->Submit(pending->peer, std::move(pending->request));
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
        ImportPDState(*context, state);
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
