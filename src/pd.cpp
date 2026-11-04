#include "pd.h"
#include "json11.hpp"
#include "models/basellm.h"
#include <algorithm>
#include <chrono>
#include <iostream>
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
    std::unique_ptr<fastkv::Service> service;
    std::string role, identity;
    std::vector<std::string> peers;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> stopping{false};
    int chunk = 2048;
    size_t tokenCapacity = 0;
    std::unique_ptr<fastkv::Budget> slots, tokens;
    std::unique_ptr<fastkv::PrefixCache> cache;
};

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
        pdRuntime->stopping = true;
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
            runtime->service->Start(config["listen"].string_value(), [this, raw = runtime.get()](
                                                                         const fastkv::Message &request) {
                fastkv::Reader reader(request.metadata);
                Check(reader.U32() == 1 && reader.String() == raw->identity,
                      "incompatible checkpoint or state layout");
                const auto tokens = reader.Ints(size_t(max_positions));
                reader.End();
                Check(request.buffers.empty() && !tokens.empty() && tokens.size() < size_t(max_positions),
                      "invalid prefill request");
                Check(tokens.size() <= raw->tokenCapacity, "prompt exceeds the prefill token pool");
                std::lock_guard<std::mutex> lock(forwardLocker);
                Check(!raw->stopping, "model is shutting down");
                fastkv::PrefixCache::Hit hit;
                if (raw->cache)
                    hit = raw->cache->Find(raw->identity, tokens, tokens.size());
                if (hit.state) {
                    if (verbose)
                        std::cout << "[FastKV PD] prefix_hit=" << hit.tokens << " prompt=" << tokens.size()
                                  << std::endl;
                    if (hit.tokens == tokens.size())
                        return *hit.state;
                }
                ResponseContext context;
                context.Init(block_cnt, dataType, kvCacheDataType);
                context.allTokens = tokens;
                context.inputTokens = tokens.size();
                context.generationConfig.top_k = 1;
                context.generationConfig.output_token_limit = 1;
                context.generationConfig.input_token_length = tokens.size() + 1;
                OnResponseContextCreated(&context);
                struct Cleanup {
                    basellm *model;
                    ResponseContext *context;
                    bool intermediate;
                    ~Cleanup() {
                        model->isIntermediateChunkedPrefill = intermediate;
                        context->isAbort = true;
                        model->OnResponseContextRemoved(context);
                    }
                } cleanup{this, &context, isIntermediateChunkedPrefill};
                isIntermediateChunkedPrefill = true;
                if (hit.state) {
                    // Import validates the boundary against N-1. The cached
                    // prefix has no sampler state; the extra token is a marker
                    // used only for validation, never fed to the model.
                    context.allTokens.resize(hit.tokens);
                    context.allTokens.push_back(0);
                    ImportPDState(context, *hit.state);
                    context.preTokens = hit.tokens;
                    context.allTokens = tokens;
                }
                for (size_t start = hit.tokens; start < tokens.size();) {
                    Check(!raw->stopping, "prefill cancelled by shutdown");
                    const size_t end = std::min(tokens.size(), start + raw->chunk);
                    std::vector<std::vector<float>> input(1);
                    for (size_t i = start; i < end; ++i)
                        input[0].push_back(tokens[i]);
                    Data ids, mask, positions;
                    FillLLMInputs(input, {{"promptLen", int(end)}, {"index", 0}, {"add_special_tokens", 0}},
                                  ids, mask, positions);
                    ToDataType(mask, dataType);
                    LastTokensManager last(1, 0);
                    Forward(ids, mask, positions, context.pastKeyValues, context.generationConfig, last,
                            nullptr);
                    context.preTokens = end;
                    start = end;
                }
                auto reply = ExportPDState(context);
                if (raw->cache) {
                    try {
                        raw->cache->Publish(raw->identity, tokens, reply);
                    } catch (const std::exception &error) {
                        // A cache is optional. Pinned entries or an oversized
                        // object must not fail an otherwise complete handoff.
                        if (verbose)
                            std::cerr << "[FastKV PD] prefix not retained: " << error.what() << std::endl;
                    }
                }
                return reply;
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
