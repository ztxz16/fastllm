#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudaworkspace.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "models/naive_n05_flash.h"
#include "../../src/models/naive_n05_tp.h"
#include "models/speculative_sampling.h"
#include "utils/utils.h"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
#include <atomic>
#include <dlfcn.h>
static std::atomic<int> verifyGraphLaunches{0}, verifyGraphCaptures{0};
static std::atomic<int> directAllocations{0};
static std::mutex draftPrefixAllocationMutex;
static std::set<void *> draftPrefixAllocations;
static std::atomic<bool> inspectDraftPrefixAllocations{false};
extern "C" void FastllmCudaFree(void *pointer) {
    if (inspectDraftPrefixAllocations) {
        std::lock_guard<std::mutex> guard(draftPrefixAllocationMutex);
        draftPrefixAllocations.erase(pointer);
    }
    static auto fn = (void (*)(void *))dlsym(RTLD_NEXT, "FastllmCudaFree");
    fn(pointer);
}
extern "C" void *FastllmCudaDirectMalloc(size_t bytes) {
    ++directAllocations;
    static auto fn = (void *(*)(size_t))dlsym(RTLD_NEXT, "FastllmCudaDirectMalloc");
    return fn(bytes);
}
static bool inspectGpuSampling = false;
static std::vector<float> sampledGpuLogits;
static int gpuSamplingCalls = 0;
extern "C" bool FastllmCudaTopKTopPSampling(float *logits, float *temperatures,
        int *topKs, float *topPs, int *output, int batch, int vocab) {
    ++gpuSamplingCalls;
    if (inspectGpuSampling) {
        sampledGpuLogits.resize((size_t)batch * vocab);
        auto error = cudaMemcpy(sampledGpuLogits.data(), logits,
            sampledGpuLogits.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
    static auto fn = (bool (*)(float *, float *, int *, float *, int *, int, int))
        dlsym(RTLD_NEXT, "FastllmCudaTopKTopPSampling");
    return fn(logits, temperatures, topKs, topPs, output, batch, vocab);
}
static bool failVerifyBegin = false, failVerifyInstantiate = false, expectVerifyGraph = true;
static int failGraphDevice = 3;
extern "C" bool FastllmCudaGraphLaunch(void *exec) {
    ++verifyGraphLaunches;
    static auto fn = (bool (*)(void *))dlsym(RTLD_NEXT, "FastllmCudaGraphLaunch");
    return fn(exec);
}
extern "C" bool FastllmCudaGraphBeginCapture() {
    ++verifyGraphCaptures;
    int device = 0;
    cudaGetDevice(&device);
    if (failVerifyBegin && device == failGraphDevice)
        return false;
    static auto fn = (bool (*)())dlsym(RTLD_NEXT, "FastllmCudaGraphBeginCapture");
    return fn();
}
extern "C" bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    int device = 0;
    cudaGetDevice(&device);
    if (failVerifyInstantiate && device == failGraphDevice) {
        *exec = nullptr;
        return false;
    }
    static auto fn = (bool (*)(void *, void **))dlsym(RTLD_NEXT, "FastllmCudaGraphInstantiate");
    return fn(graph, exec);
}
#endif
using namespace fastllm;
namespace fastllm {
struct NaiveN05GraphTestAccess {
    static auto &DraftPrefixPool(NaiveN05FlashModel &model) { return model.prefixPool; }
    static void CheckHeadReturn(const NaiveN05FlashModel &model, const Data &output, bool deviceInput) {
        const bool borrowed = deviceInput && !model.tpDevices.empty() && FastllmCudaPeerAccessInit(model.tpDevices);
        if (output.isFake != borrowed || (borrowed && output.cudaData != model.tpDraftHeadOutput.cudaData))
            throw std::runtime_error("head return lost its borrowed/owned storage contract");
    }
    static void CheckDraftWarmup(NaiveN05FlashModel &model) {
        const char *warmup = std::getenv("FASTLLM_CUDA_SERVING_WARMUP");
        if (!warmup || std::string(warmup) != "1" || !model.draftTP) return;
        const int vocab = model.weight["lm_head.weight"].dims.front();
        const size_t bytes = size_t(model.maxBatch) * (model.draftTokens + 1) * vocab * sizeof(float);
        if (model.tpSamplingLogits.GetBytes() != bytes)
            throw std::runtime_error("draft warmup sized verification logits for the training block");
        if (model.tpBatchVerifyStorage) for (const auto &rank : model.tpBatchVerifyStorage->ranks) {
            const auto &scratch = rank->storage->buffers.decode;
            const size_t rows = size_t(model.maxBatch) * (model.draftTokens + 1);
            if (scratch.indexQuery.Count(0) < rows * 16 * 128 ||
                scratch.indexScale.Count(0) < rows * 16 || scratch.topk.dims.empty())
                throw std::runtime_error("draft warmup omitted verification Indexer scratch");
        }
    }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    static void WatchDraftPrefix(NaiveN05FlashModel &model, const std::vector<std::pair<Data, Data>> &kv) {
        if (!inspectDraftPrefixAllocations) return;
        std::lock_guard<std::mutex> historyGuard(model.historyMutex);
        auto it = model.draftContexts.find(&kv);
        if (it != model.draftContexts.end() && it->second && it->second->prefixHidden.cudaData && !it->second->prefixHidden.isFake) {
            std::lock_guard<std::mutex> guard(draftPrefixAllocationMutex);
            draftPrefixAllocations.insert(it->second->prefixHidden.cudaData);
        }
    }
#endif
    static void CheckRejectedVerificationReuse(NaiveN05FlashModel &model,
            const Data &ids, const NaiveN05FlashModel::TargetBatch &batch) {
        auto saved = model.tpDecodeState;
        model.tpDecodeState = std::make_shared<NaiveN05FlashModel::TPDecodeState>();
        auto &embedding = *model.weight["model.embed_tokens.weight"].multiDeviceDatas.at(model.tpDevices.front());
        const auto type = embedding.dataType;
        struct Restore {
            NaiveN05FlashModel &model; std::shared_ptr<NaiveN05FlashModel::TPDecodeState> saved;
            Data &embedding; DataType type;
            ~Restore() { embedding.dataType = type; model.tpDecodeState = std::move(saved); }
        } restore{model, saved, embedding, type};
        std::vector<std::pair<Data, Data>> unused;
        std::vector<int *> lengths;
        NaiveN05FlashModel::LogitsSelection selection;
        selection.cudaLogits = true;
        // Reject after cache metadata is populated, before any rank is made.
        embedding.dataType = FLOAT32;
        if (model.PrepareTensorParallelDecode(ids, unused, true, selection, lengths, &batch, false) ||
            !model.tpDecodeState->dynamicCache || !model.tpDecodeState->disabled ||
            !model.tpDecodeState->ranks.empty())
            throw std::runtime_error("verification rejection fixture did not reach partial state");
        embedding.dataType = type;
        for (int repeat = 0; repeat < 3; ++repeat)
            if (model.PrepareTensorParallelDecode(ids, unused, true, selection, lengths, &batch, false) ||
                !lengths.empty())
                throw std::runtime_error("rejected verification graph did not retain eager fallback");
    }
    static void CheckSharedVerificationWorkspace(const NaiveN05FlashModel &model, int ranks) {
        // Acceptance tails and request churn must not consume one Lt lease
        // per graph: the bounded pool is shared with concurrent owners.
        for (int rank = 0; rank < ranks; ++rank) {
            void *lease = nullptr;
            for (const auto &state : model.tpBatchVerifyGraphs) {
                if (state->disabled) {
                    if (!state->reservedPointers.empty())
                        throw std::runtime_error("failed verification capture retained pool pins");
                    for (const auto &part : state->ranks)
                        if (part->graph || part->exec)
                            throw std::runtime_error("failed verification capture retained graph resources");
                }
                if (rank >= (int)state->ranks.size()) continue;
                if (state->dynamicCache && model.tpBatchVerifyStorage &&
                    state->ranks[rank]->storage != model.tpBatchVerifyStorage->ranks[rank]->storage)
                    throw std::runtime_error("verification shapes duplicated their activation workspace");
                void *current = state->ranks[rank]->storage->linearGraphWorkspace.get();
                if (current && lease && current != lease)
                    throw std::runtime_error("verification shapes duplicated their Lt lease");
                if (current) lease = current;
            }
        }
    }
};
}
// Cross-algorithm BF16 logits can differ because GEMM and single-row GEMV
// use different FP32 reduction orders. Keep the existing model-smoke relative
// error/cosine limits, and bound isolated outliers too. Metadata and same-path
// rollback/replay comparisons below remain exact.
struct LogitError {
    double relative = 0, cosine = 1, maximum = 0, scale = 0;
    bool acceptable = true;
};
static LogitError CompareLogits(const float *actual, const float *expected, size_t count) {
    LogitError result;
    double error = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < count; ++i) {
        double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.acceptable = false;
            return result;
        }
        error += (a - b) * (a - b);
        aa += a * a; bb += b * b; ab += a * b;
        result.maximum = std::max(result.maximum, std::abs(a - b));
        result.scale = std::max(result.scale, std::abs(b));
    }
    result.relative = std::sqrt(error / std::max(bb, 1e-30));
    result.cosine = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    result.acceptable = count > 0 && result.relative <= .02 && result.cosine >= .999 &&
                        result.maximum <= 1e-6 + .02 * result.scale;
    return result;
}
static double largestLogitRelative = 0, largestLogitAbsolute = 0;
static void RequireCloseLogits(const float *actual, const std::vector<float> &expected,
                               const std::string &context) {
    auto error = CompareLogits(actual, expected.data(), expected.size());
    largestLogitRelative = std::max(largestLogitRelative, error.relative);
    largestLogitAbsolute = std::max(largestLogitAbsolute, error.maximum);
    if (!error.acceptable) {
        std::cerr << context << " relative_rmse=" << error.relative
                  << " cosine=" << error.cosine << " max_abs=" << error.maximum
                  << " reference_max=" << error.scale << std::endl;
        throw std::runtime_error(context);
    }
}
static void CheckLogitComparison() {
    const float reference[] = {1, -2, 3, -4};
    float actual[] = {1.001f, -2.002f, 3.003f, -4.004f};
    if (!CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("small BF16 error rejected");
    actual[0] = 2;
    if (CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("large logit error accepted");
    actual[0] = std::numeric_limits<float>::quiet_NaN();
    if (CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("non-finite logit accepted");
}
// Shared deterministic draft weights; existing target embeddings stay untouched.
static void AddDraftTestWeights(WeightMap &weight, int layers, int kvHeads, int vocab, bool merged,
                                int hidden = 256, int head = 32, int intermediate = 512) {
    unsigned seed = 17;
    auto add = [&](const std::string &name, std::vector<int> dims, bool norm = false) {
        weight.AddEmptyWeight(name, dims, norm ? FLOAT32 : BFLOAT16);
        Data &w = weight[name];
        w.Allocate();
        for (int i = 0; i < w.Count(0); ++i) {
            seed = seed * 1664525u + 1013904223u;
            float value = norm ? 1.f : .04f * ((int)(seed >> 16) - 32768) / 32768.f;
            if (norm)
                ((float *)w.cpuData)[i] = value;
            else
                ((uint16_t *)w.cpuData)[i] = Float32ToBFloat16RNEBits(value);
        }
    };
    if (!weight.weight.count("model.embed_tokens.weight"))
        add("model.embed_tokens.weight", {256, hidden});
    add("dspark.mask_embedding", {hidden});
    add("dspark.norm.weight", {hidden}, true);
    add("dspark.markov_head.markov_w1.weight", {vocab, hidden});
    add("dspark.markov_head.markov_w2.weight", {vocab, hidden});
    add("dspark.fc.weight", {hidden, 2 * hidden});
    add("dspark.hidden_norm.weight", {hidden}, true);
    for (int i = 0; i < layers; ++i) {
        std::string p = "dspark.layers." + std::to_string(i) + ".";
        add(p + "input_layernorm.weight", {hidden}, true);
        add(p + "post_attention_layernorm.weight", {hidden}, true);
        for (const char *n : {"q", "k", "v"})
            add(p + "self_attn." + n + "_proj.weight", {n[0] == 'q' ? hidden : kvHeads * head, hidden});
        add(p + "self_attn.o_proj.weight", {hidden, hidden});
        add(p + "self_attn.q_norm.weight", {head}, true);
        add(p + "self_attn.k_norm.weight", {head}, true);
        add(p + "mlp.gate_proj.weight", {intermediate, hidden});
        add(p + "mlp.up_proj.weight", {intermediate, hidden});
        add(p + "mlp.down_proj.weight", {hidden, intermediate});
    }
    if (merged) for (int i = 0; i < layers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        auto merge = [&](std::vector<std::string> names, std::string out) {
            int rows = 0;
            for (auto &name : names) rows += weight[p + name].dims[0];
            Data &w = weight[p + out];
            w.dataType = BFLOAT16; w.isModelWeight = true; w.Resize({rows, hidden}); w.Allocate();
            size_t offset = 0;
            for (auto &name : names) {
                Data &part = weight[p + name];
                memcpy(w.cpuData + offset, part.cpuData, part.GetBytes());
                offset += part.GetBytes();
                weight.weight.erase(p + name);
            }
        };
        merge({"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"}, "self_attn.mergeqkv.weight");
        merge({"mlp.gate_proj.weight", "mlp.up_proj.weight"}, "mlp.gateup_proj.weight");
    }
}
class Fixture : public NaiveN05FlashModel {
  public:
    int ranks;
    bool useRealScheduler = false;
    bool failDecodeAllocation = false;
    bool inspectDraftPrefix = false;
    int batchedPrefills = 0;
    std::atomic<int> largestDecodeBatch{0};
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &masks,
            const std::vector<Data *> &positions, const std::vector<int> &lengths,
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &configs,
            const LastTokensManager &tokens = LastTokensManager(), std::vector<std::vector<float> *> *logits = nullptr) override {
        batchedPrefills += batch > 1 && std::any_of(lengths.begin(), lengths.end(), [](int n) { return n > 1; });
        if (std::all_of(lengths.begin(), lengths.end(), [](int n) { return n == 1; }))
            largestDecodeBatch = std::max(largestDecodeBatch.load(), batch);
        return NaiveN05FlashModel::ForwardBatch(batch, ids, masks, positions, lengths, kv, configs, tokens, logits);
    }
    std::vector<float> observedDraft;
    int Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                std::vector<std::pair<Data, Data>> &kv,
                const GenerationConfig &config = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector<float> *logits = nullptr) override {
        if (failDecodeAllocation && inputIds.dims == std::vector<int>({1, 1})) {
            failDecodeAllocation = false;
            throw CudaWorkspaceExhausted("injected decode allocation failure");
        }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        NaiveN05GraphTestAccess::WatchDraftPrefix(*this, kv);
#endif
        int token = NaiveN05FlashModel::Forward(inputIds, attentionMask, positionIds, kv, config, lastTokens, logits);
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        NaiveN05GraphTestAccess::WatchDraftPrefix(*this, kv);
#endif
        if (inspectDraftPrefix && !isIntermediateChunkedPrefill && config.output_logits) {
            auto draft = draftContexts.at(&kv);
            if (draft->committed == config.input_token_length) {
                Data h = RunDraft(17, *draft), head = RunDraftHead(h);
                // The head may borrow persistent logits; convert an owned oracle.
                Data scores(head);
                ToDataType(scores, FLOAT32); scores.ToDevice(DataDevice::CPU);
                observedDraft.assign((float *)scores.cpuData, (float *)scores.cpuData + scores.Count(0));
            }
        }
        return token;
    }
    explicit Fixture(int ranks, bool packed = false, bool graph = false, bool swaRing = false,
                     bool serving = false, bool secondDsa = false) : ranks(ranks) {
        setenv("FASTLLM_TP", ranks > 1 ? std::to_string(ranks).c_str() : "false", 1);
        deviceMap = {{"cuda:0", 1}};
        moeDeviceMap = deviceMap;
        weight.dicts = {{"num_hidden_layers", "2"},
                        {"hidden_size", "256"},
                        {"num_attention_heads", "16"},
                        {"num_key_value_heads", "4"},
                        {"head_dim", "32"},
                        {"v_head_dim", "16"},
                        {"swa_num_attention_heads", "16"},
                        {"swa_num_key_value_heads", "8"},
                        {"swa_head_dim", "32"},
                        {"swa_v_head_dim", "16"},
                        {"hybrid_layer_pattern", "[0,1]"},
                        {"moe_layer_freq", "[0,1]"},
                        {"scoring_func", "sigmoid"},
                        {"sliding_window", "8"},
                        {"index_top_k", "16"},
                        {"indexer_activation_dtype", "bf16"},
                        {"n_routed_experts", "8"},
                        {"num_experts_per_tok", "2"},
                        {"max_position_embeddings", "1024"}};
        if (graph) {
            weight.dicts["index_top_k"] = "2048";
            weight.dicts["max_position_embeddings"] = "8192";
        }
        if (swaRing) {
            weight.dicts["swa_num_attention_heads"] = "64";
            weight.dicts["swa_head_dim"] = "192";
            weight.dicts["swa_v_head_dim"] = "128";
            weight.dicts["sliding_window"] = "128";
        }
        if (serving) {
            weight.dicts["num_attention_heads"] = "64";
            weight.dicts["head_dim"] = "192";
            weight.dicts["v_head_dim"] = "128";
            weight.dicts["moe_layer_freq"] = "[0,0]";
        }
        if (secondDsa) {
            weight.dicts["hybrid_layer_pattern"] = "[0,0]";
            weight.dicts["indexer_activation_dtype"] = "fp8";
        }
        InitParams();
        SetSaveHistoryChat(false);
        unsigned seed = 7;
        auto add = [&](std::string name, std::vector<int> dims, DataType type, float scale,
                       bool norm = false) {
            if (packed && name.find(".mlp.experts.") != std::string::npos) {
                weight.AddEmptyWeight(name, dims, NVFP4_BLOCK_16_E4M3_PACKED);
                auto &w = weight[name];
                w.directMemory = true;
                w.blockK = 1;
                w.blockM = 16;
                w.Allocate(false);
                size_t stride = w.GetBytes() / dims[0];
                for (int row = 0; row < dims[0]; ++row) {
                    uint8_t *dst = w.cpuData + row * stride;
                    float global =
                        name.find("gateup_proj") != std::string::npos && row >= dims[0] / 2 ? .006f
                                                                                            : .004f;
                    std::memcpy(dst, &global, sizeof(float));
                    for (int group = 0; group < dims[1] / 16; ++group) {
                        for (int j = 0; j < 8; ++j) {
                            seed = seed * 1664525 + 1013904223;
                            dst[4 + group * 9 + j] = seed >> 16;
                        }
                        dst[12 + group * 9] = 56; // E4M3 1.0
                    }
                }
                return;
            }
            weight.AddEmptyWeight(name, dims, type);
            auto &w = weight[name];
            w.Allocate();
            for (int i = 0; i < w.Count(0); ++i) {
                seed = seed * 1664525 + 1013904223;
                float v = norm ? 1.f : scale * (int(seed >> 16) - 32768) / 32768.f;
                if (type == FLOAT32)
                    ((float *)w.cpuData)[i] = v;
                else {
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    ((uint16_t *)w.cpuData)[i] = bits >> 16;
                }
            }
        };
        add("model.embed_tokens.weight", {256, 256}, BFLOAT16, .2f);
        add("model.norm.weight", {256}, FLOAT32, 0, true);
        add("lm_head.weight", {256, 256}, BFLOAT16, .03f);
        for (int i = 0; i < 2; ++i) {
            auto p = "model.layers." + std::to_string(i);
            auto ap = p + ".self_attn.";
            const int heads = (i && swaRing) || serving ? 64 : 16, kv = i && !secondDsa ? 8 : 4;
            const int dim = (i && swaRing) || serving ? 192 : 32, valueDim = (i && swaRing) || serving ? 128 : 16;
            add(p + ".input_layernorm.weight", {256}, FLOAT32, 0, true);
            add(p + ".post_attention_layernorm.weight", {256}, FLOAT32, 0, true);
            add(ap + "q_proj.weight", {heads * dim, 256}, BFLOAT16, .03f);
            add(ap + "k_proj.weight", {kv * dim, 256}, BFLOAT16, .03f);
            add(ap + "v_proj.weight", {kv * valueDim, 256}, BFLOAT16, .03f);
            add(ap + "o_proj.weight", {256, heads * valueDim}, BFLOAT16, .03f);
            add(ap + "attention_sink_bias", {heads}, FLOAT32, .1f);
            if (i == 0 || serving) {
                add(ap + "indexer.wq.weight", {2048, 256}, BFLOAT16, .03f);
                add(ap + "indexer.wk.weight", {128, 256}, BFLOAT16, .03f);
                add(ap + "indexer.weights_proj.weight", {16, 256}, BFLOAT16, .03f);
                add(ap + "indexer.k_norm.weight", {128}, FLOAT32, 0, true);
                add(ap + "indexer.k_norm.bias", {128}, FLOAT32, .01f);
                add(p + ".mlp.gate_proj.weight", {1024, 256}, BFLOAT16, .02f);
                add(p + ".mlp.up_proj.weight", {1024, 256}, BFLOAT16, .02f);
                add(p + ".mlp.down_proj.weight", {256, 1024}, BFLOAT16, .02f);
            } else {
                add(p + ".mlp.gate.weight", {8, 256}, FLOAT32, .2f);
                add(p + ".mlp.gate.e_score_correction_bias", {8}, FLOAT32, .01f);
                for (int e = 0; e < 8; ++e) {
                    auto ep = p + ".mlp.experts." + std::to_string(e) + ".";
                    add(ep + "gateup_proj.weight", {2048, 256}, BFLOAT16, .02f);
                    add(ep + "down_proj.weight", {256, 1024}, BFLOAT16, .02f);
                }
            }
        }
    }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    std::vector<float> SparseDraftChecks(bool enabled, int block, int resident, bool graphSupported = true, int chunk = 1024) {
        draftEnabled = canDoConcurrentForward = true;
        draftLayers = 2; draftBlock = draftTokens = block;
        draftHeads = 8; draftKvHeads = 2; draftHeadDim = 32; draftWindow = 1024;
        draftConfidenceThreshold = 0; draftTargetLayers = {0, 1};
        AddDraftTestWeights(weight, draftLayers, draftKvHeads, 256, true);
        maxBatch = 4; max_positions = 16384; SetMaxTokens(65536); SetChunkedPrefillSize(chunk);
        if (enabled) {
            KvMemConfig cfg; cfg.enabled = true; cfg.maxTokens = max_positions;
            cfg.residentTokens = resident; cfg.prefillTokens = chunk; cfg.hostBytes = 1ULL << 30;
            ConfigureKvMem(cfg);
        }
        WarmUp();
        std::vector<float> result;
        auto record = [&](const Data &x) {
            Data host(x); host.ToDevice(DataDevice::CPU);
            for (size_t i = 0; i < host.Count(0); ++i)
                result.push_back(host.dataType == FLOAT32 ? ((float *)host.cpuData)[i] :
                    BFloat16BitsToFloat32(((uint16_t *)host.cpuData)[i]));
        };
        using Cache = std::vector<std::pair<Data, Data>>;
        GenerationConfig config; config.top_k = 0; config.top_p = .95f; config.temperature = 1.f;
        config.input_token_length = 5000; config.output_token_limit = 128;
        LogitsSelection selection; selection.cudaLogits = true;
        for (int wave = 0; wave < 2; ++wave) {
            std::vector<Cache> kv(4);
            std::vector<std::shared_ptr<DraftContext>> draft;
            std::vector<int> past{126, 2047, 2051, 4100};
            for (int owner = 0; owner < 4; ++owner) {
                draft.push_back(CreateDraftContext());
                for (int layer = 0; layer < block_cnt; ++layer) kv[owner].emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
                for (int start = 0; start < past[owner]; start += 1024) {
                    const int count = std::min(1024, past[owner] - start);
                    std::vector<float> ids(count), positions(count);
                    for (int row = 0; row < count; ++row) { ids[row] = (start + row + owner * 7 + wave) % 251; positions[row] = start + row; }
                    TargetCapture capture; capture.selection = &selection;
                    Data logits = RunDraftTarget(Data(FLOAT32, {1, count}, ids), Data(FLOAT32, {1, count}, positions), kv[owner], config, capture);
                    record(logits); CommitDraftContext(capture, count, *draft[owner], kv[owner]);
                }
            }
            for (int turn = 0; turn < 8; ++turn) {
                const int count = turn % 4 == 0 ? 1 : turn % 4 == 1 ? 4 : 2;
                std::vector<int> owners, lengths, before, kept, anchors;
                std::vector<float> ids, positions;
                std::vector<std::pair<Data *, Data *>> pointers;
                std::vector<Cache *> caches;
                std::vector<DraftContext *> contexts;
                for (int i = 0; i < count; ++i) {
                    int owner = (i + turn) % 4, rows = turn % 3 == 0 ? block + 1 : 1 + (i + turn) % (block + 1);
                    owners.push_back(owner); lengths.push_back(rows); before.push_back(past[owner]);
                    kept.push_back(turn % 4 == 0 ? 0 : turn % 4 == 1 ? rows : std::max(1, rows / 2));
                    caches.push_back(&kv[owner]); contexts.push_back(draft[owner].get()); anchors.push_back((turn * 11 + owner) % 251);
                    for (auto &pair : kv[owner]) pointers.emplace_back(&pair.first, &pair.second);
                    for (int row = 0; row < rows; ++row) {
                        ids.push_back((past[owner] + row + turn * 23 + wave) % 251);
                        positions.push_back(past[owner] + row);
                    }
                }
                std::vector<GenerationConfig> configs(count, config);
                TargetBatch batch{pointers, configs, lengths, block_cnt};
                TargetCapture capture; capture.verifying = true; capture.borrowOutputs = true; capture.selection = &selection;
                Cache unused;
                const bool single = count == 1 && turn % 8 == 4;
                Cache &target = single ? kv[owners.front()] : unused;
                const TargetBatch *targetBatch = single ? nullptr : &batch;
                SetCudaGraph(turn % 4 != 2);
                Data logits = RunDraftTarget(Data(FLOAT32, {1, (int)ids.size()}, ids),
                    Data(FLOAT32, {1, (int)positions.size()}, positions), target, config, capture, targetBatch);
                record(logits);
                const int launches = verifyGraphLaunches;
                for (int replay = 0; replay < 3 && GetFastllmEnv().cudaGraph; ++replay) {
                    CommitTargetCaches(caches, before, std::vector<int>(count, 0), capture.ringVerify);
                    TargetCapture repeated; repeated.verifying = true; repeated.borrowOutputs = true; repeated.selection = &selection;
                    Data again = RunDraftTarget(Data(FLOAT32, {1, (int)ids.size()}, ids),
                        Data(FLOAT32, {1, (int)positions.size()}, positions), target, config, repeated, targetBatch);
                    record(again); capture = std::move(repeated);
                }
                if (GetFastllmEnv().cudaGraph && graphSupported && verifyGraphLaunches <= launches)
                    throw std::runtime_error("sparse/native verification did not replay its graph");
                for (const auto &feature : capture.hidden) record(feature.second);
                CommitTargetCaches(caches, before, kept, capture.ringVerify);
                if (std::find(kept.begin(), kept.end(), 0) == kept.end()) {
                    if (!CommitDraftContexts(capture, lengths, kept, contexts))
                        throw std::runtime_error("sparse draft context batch fell back");
                } else {
                    // Zero acceptance here is a storage rollback oracle; real
                    // speculative rounds always retain the anchor input.
                    if (count != 1) throw std::runtime_error("rollback fixture shape");
                }
                for (int i = 0; i < count; ++i) past[owners[i]] += kept[i];
                Data hidden;
                if (!RunDraftBatch(anchors, contexts, hidden)) throw std::runtime_error("sparse draft TP batch fell back");
                record(hidden); Data head = RunDraftHead(hidden); record(head);
                // A different token overwrites the first rejected candidate.
                // Exercise ordinary target decoding after each partial rollback.
                for (int owner : owners) {
                    TargetCapture next; next.selection = &selection;
                    Data token(FLOAT32, {1, 1}, {(float)((past[owner] + turn * 31 + 137) % 251)});
                    Data pos(FLOAT32, {1, 1}, {(float)past[owner]});
                    Data nextLogits = RunDraftTarget(token, pos, kv[owner], config, next);
                    record(nextLogits); CommitDraftContext(next, 1, *draft[owner], kv[owner]); ++past[owner];
                }
            }
            SetCudaGraph(true);
            // A single-request graph must observe accepted length growth on
            // replay, not a history staging length frozen during capture.
            const int launches = verifyGraphLaunches;
            for (int step = 0; step < 6; ++step) {
                const int owner = 3, rows = block + 1, kept = 1 + step % 3;
                std::vector<float> ids(rows), positions(rows);
                for (int row = 0; row < rows; ++row) {
                    ids[row] = (past[owner] + row + step * 19) % 251;
                    positions[row] = past[owner] + row;
                }
                TargetCapture capture; capture.verifying = true; capture.selection = &selection;
                Data logits = RunDraftTarget(Data(FLOAT32, {1, rows}, ids),
                    Data(FLOAT32, {1, rows}, positions), kv[owner], config, capture);
                record(logits);
                for (const auto &feature : capture.hidden) record(feature.second);
                CommitTargetCaches({&kv[owner]}, {past[owner]}, {kept}, capture.ringVerify);
                past[owner] += kept;
            }
            if (graphSupported && verifyGraphLaunches <= launches)
                throw std::runtime_error("growing single-request verification did not replay its graph");
        }
        return result;
    }
    std::vector<float> SparseKvRun(bool enabled) {
        maxBatch = 4; SetMaxTokens(16384); SetChunkedPrefillSize(1024);
        if (enabled) {
            SetMaxTokens(13312); // Less than 4 * 4096, enough for the actual mixed histories.
            KvMemConfig config;
            config.enabled = true; config.maxTokens = 4096; config.residentTokens = 2048;
            config.prefillTokens = 1024; config.hostBytes = 1ULL << 30;
            ConfigureKvMem(config);
        } else max_positions = 4096;
        WarmUp();
        if (enabled && (tokensLimit != 13312 ||
            GetPagedKVCacheManager(0, true)->maxPages * GetPageLen() != 13312))
            throw std::runtime_error("KVMem ignored the independent physical pool budget");
        const int captures = verifyGraphCaptures;
        if (!captures) throw std::runtime_error("KVMem test requires startup CUDA Graph");
        std::vector<float> result;
        using Cache = std::vector<std::pair<Data, Data>>;
        GenerationConfig config;
        config.top_k = 1; config.repeat_penalty = 1; config.output_logits = true;
        config.input_token_length = 3072; config.output_token_limit = 64;
        // Recreate requests twice: every slot must be reset before reuse.
        // Both short implicit selection and >resident exact TopK are covered.
        for (int round = 0; round < 2; ++round) {
            std::vector<Cache> requests(4);
            std::vector<int> past(4, round ? 3072 : 12);
            for (int row = 0; row < 4; ++row) {
                for (int layer = 0; layer < block_cnt; ++layer)
                    requests[row].emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
                for (int start = 0; start < past[row]; start += 1024) {
                    int length = std::min(1024, past[row] - start);
                    std::vector<float> ids(length), positions(length), logits;
                    for (int i = 0; i < length; ++i) { ids[i] = ((start + i) * 3 + row + round) % 251; positions[i] = start + i; }
                    Forward(Data(FLOAT32, {1, length}, ids), Data(), Data(FLOAT32, {1, length}, positions),
                        requests[row], config, LastTokensManager(), &logits);
                    result.insert(result.end(), logits.begin(), logits.end());
                }
            }
            for (int step = 0; step < 12; ++step) {
                const int count = (step % 4) + 1;
                std::vector<float> ids(count);
                std::vector<Data> positions(count);
                std::vector<Data *> pos;
                std::vector<std::pair<Data *, Data *>> caches;
                std::vector<std::vector<float>> logits(count);
                std::vector<std::vector<float> *> outputs;
                for (int row = 0; row < count; ++row) {
                    const int owner = step % 2 ? 3 - row : row;
                    ids[row] = (past[owner] * 3 + owner + round) % 251;
                    positions[row].CopyFrom(Data(FLOAT32, {1, 1}, {(float)past[owner]}));
                    pos.push_back(&positions[row]); outputs.push_back(&logits[row]);
                    for (auto &layer : requests[owner]) caches.emplace_back(&layer.first, &layer.second);
                    ++past[owner];
                }
                const bool graph = step % 3 != 0;
                SetCudaGraph(graph);
                const int launches = verifyGraphLaunches;
                ForwardBatch(count, Data(FLOAT32, {1, count}, ids), {}, pos, std::vector<int>(count, 1), caches,
                    std::vector<GenerationConfig>(count, config), LastTokensManager(), &outputs);
                if (graph && verifyGraphLaunches - launches != ranks)
                    throw std::runtime_error("KVMem test failed to replay all ranks");
                if (verifyGraphCaptures != captures) throw std::runtime_error("KVMem serving recaptured graphs");
                for (auto &row : logits) result.insert(result.end(), row.begin(), row.end());
            }
            SetCudaGraph(true);
        }
        // Mixed-length batched prefill shares staging, but never cache owners.
        std::vector<Cache> merged(4);
        std::vector<int> lengths{23, 31, 47, 61};
        std::vector<float> ids;
        std::vector<Data> positions(4);
        std::vector<Data *> pos;
        std::vector<std::pair<Data *, Data *>> caches;
        std::vector<std::vector<float>> logits(4);
        std::vector<std::vector<float> *> outputs;
        for (int row = 0; row < 4; ++row) {
            for (int layer = 0; layer < block_cnt; ++layer)
                merged[row].emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
            for (auto &layer : merged[row]) caches.emplace_back(&layer.first, &layer.second);
            std::vector<float> coordinates(lengths[row]);
            for (int i = 0; i < lengths[row]; ++i) { ids.push_back((i * 7 + row) % 251); coordinates[i] = i; }
            positions[row].CopyFrom(Data(FLOAT32, {1, lengths[row]}, coordinates));
            pos.push_back(&positions[row]); outputs.push_back(&logits[row]);
        }
        ForwardBatch(4, Data(FLOAT32, {1, (int)ids.size()}, ids), {}, pos, lengths, caches,
            std::vector<GenerationConfig>(4, config), LastTokensManager(), &outputs);
        for (auto &row : logits) result.insert(result.end(), row.begin(), row.end());
        if (verifyGraphCaptures != captures) throw std::runtime_error("KVMem prefill recaptured graphs");
        merged.clear();
        if (enabled) {
            eos_token_id = -1; eos_token_ids.clear(); useRealScheduler = true;
            SetPrefillBatching(16, 20);
            config.output_logits = false; config.output_token_limit = 16;
            const int launches = verifyGraphLaunches;
            std::vector<int> fullBatch;
            for (int i = 0; i < maxBatch; ++i)
                fullBatch.push_back(LaunchResponseTokens({1, 2, 3, 4}, config));
            for (int handle : fullBatch) {
                for (int i = 0; i < config.output_token_limit; ++i)
                    if (FetchResponseTokens(handle) < 0)
                        throw std::runtime_error("KVMem full batch ended early");
                if (FetchResponseTokens(handle) != -1)
                    throw std::runtime_error("KVMem full batch output count");
            }
            if (verifyGraphLaunches - launches != ranks * (config.output_token_limit - 1))
                throw std::runtime_error("KVMem full slots serialized decode");
            config.output_logits = false; config.output_token_limit = 1;
            std::vector<int> handles;
            for (int i = 0; i < 12; ++i) handles.push_back(LaunchResponseTokens({1, 2, 3, 4}, config));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!std::all_of(handles.begin(), handles.end(), [&](int h) { return CanFetchResponse(h); })) {
                if (std::chrono::steady_clock::now() > deadline)
                    throw std::runtime_error("KVMem slots held by completed output queues");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (int handle : handles) {
                if (FetchResponseTokens(handle) < 0 || FetchResponseTokens(handle) != -1)
                    throw std::runtime_error("KVMem completed request lost queued output");
            }
            ShutdownRuntime();
        }
        return result;
    }
#endif
    mutable int pagedRestores = 0, lastPrefixLength = 0;
    mutable bool failPrefixRestore = false;
    int hostRestores = 0;
    bool RestorePagedPrefixCacheExtra(ResponseContext *ctx, int length) const override {
        bool restored = NaiveN05FlashModel::RestorePagedPrefixCacheExtra(ctx, length);
        if (restored && failPrefixRestore) {
            failPrefixRestore = false;
            throw CudaWorkspaceExhausted("injected prefix allocation failure");
        }
        pagedRestores += restored;
        if (restored) lastPrefixLength = length;
        return restored;
    }
    bool TryRestoreHistoryCache(std::vector<int> &tokens, int &length) override {
        bool restored = NaiveN05FlashModel::TryRestoreHistoryCache(tokens, length);
        hostRestores += restored;
        return restored;
    }
    bool UseModelSpecificScheduler() const override { return !useRealScheduler; }
    void RunModelSpecificScheduler() override {}
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    void SharedPoolServingChecks(bool kvmem = true, bool batchPrefill = true) {
        maxBatch = 8; SetMaxTokens(7937); SetChunkedPrefillSize(128);
        KvMemConfig cfg; cfg.enabled = true; cfg.maxTokens = 7937;
        cfg.residentTokens = 2048; cfg.prefillTokens = 128; cfg.hostBytes = 1ULL << 30;
        if (kvmem) {
            bool rejected = false;
            try { ConfigureKvMem(cfg); } catch (const std::invalid_argument &) { rejected = true; }
            if (!rejected) throw std::runtime_error("pool smaller than one rounded request was accepted");
        }
        SetMaxTokens(8193); // The physical pool rounds down to complete pages.
        if (kvmem) ConfigureKvMem(cfg);
        else max_positions = cfg.maxTokens;
        WarmUp();
        if (kvmem && (tokensLimit != 8192 || GetPagedKVCacheManager(0, true)->maxPages != 64))
            throw std::runtime_error("independent pool allocation mismatch");
        if (!SupportsSharedPagedKVPool()) throw std::runtime_error("physical pool budget disabled");
        useRealScheduler = true; eos_token_id = -1; eos_token_ids.clear();
        if (batchPrefill) SetPrefillBatching(8192, 20);
        const int captures = verifyGraphCaptures;
        auto run = [&](const std::vector<std::vector<int>> &prompts, int count,
                       std::vector<std::vector<int>> *sampled = nullptr) {
            std::vector<int> handles;
            std::vector<std::vector<std::vector<float>>> output(prompts.size());
            if (sampled) sampled->resize(prompts.size());
            for (const auto &tokens : prompts) {
                GenerationConfig gen; gen.top_k = 1; gen.output_logits = true;
                gen.input_token_length = tokens.size(); gen.output_token_limit = count;
                handles.push_back(LaunchResponseTokens(tokens, gen));
            }
            for (size_t row = 0; row < handles.size(); ++row) {
                for (int step = 0; step < count; ++step) {
                    output[row].emplace_back();
                    const int token = FetchResponseLogits(handles[row], output[row].back());
                    if (token < 0)
                        throw std::runtime_error("shared pool truncated a response");
                    if (sampled) (*sampled)[row].push_back(token);
                }
                std::vector<float> unused;
                if (FetchResponseLogits(handles[row], unused) != -1)
                    throw std::runtime_error("shared pool completion failed");
            }
            return output;
        };
        std::vector<int> prefix(6144);
        for (int i = 0; i < (int)prefix.size(); ++i) prefix[i] = i * 3 % 251;
        SetSaveHistoryChat(true);
        run({prefix}, 1);
        std::vector<std::vector<int>> forks(8, prefix);
        for (int i = 0; i < 8; ++i) forks[i].push_back(240 + i);
        const int hits = pagedRestores;
        largestDecodeBatch = 0;
        std::vector<std::vector<int>> sharedTokens;
        const auto shared = run(forks, 132, &sharedTokens); // Nearly fills the physical pool.
        if (pagedRestores - hits != 8)
            throw std::runtime_error("physical pool double-counted shared prefixes");
        if (largestDecodeBatch.load() != 8)
            throw std::runtime_error("shared prefixes did not decode as a full batch");
        // Larger suffixes require queuing even though all eight prefixes hit.
        // Rejected candidates must release references and restore them later.
        std::vector<std::vector<int>> queuedTokens;
        const auto queued = run(forks, 384, &queuedTokens);
        SetSaveHistoryChat(false);
        for (int row = 0; row < 8; ++row) {
            // Compare identical histories. Independent greedy runs can choose
            // different near-tied tokens and diverge after many decode steps.
            for (int step : {0, 1, 126, 127, 131}) {
                auto input = forks[row];
                input.insert(input.end(), sharedTokens[row].begin(), sharedTokens[row].begin() + step);
                auto cold = run({input}, 1);
                RequireCloseLogits(shared[row][step].data(), cold[0][0], "shared pool logits");
            }
            auto input = forks[row];
            input.insert(input.end(), queuedTokens[row].begin(), queuedTokens[row].end() - 1);
            auto cold = run({input}, 1);
            RequireCloseLogits(queued[row].back().data(), cold[0][0], "queued prefix logits");
        }
        std::vector<std::vector<int>> independent(8, std::vector<int>(2048));
        for (int row = 0; row < 8; ++row)
            for (int i = 0; i < 2048; ++i) independent[row][i] = (i * 3 + row) % 251;
        std::vector<std::vector<int>> pressureTokens;
        const auto pressured = run(independent, 256, &pressureTokens);
        for (int row = 0; row < 8; ++row) {
            for (int step : {0, 127, 128, 255}) {
                auto input = independent[row];
                input.insert(input.end(), pressureTokens[row].begin(), pressureTokens[row].begin() + step);
                auto cold = run({input}, 1);
                RequireCloseLogits(pressured[row][step].data(), cold[0][0], "pool pressure logits");
            }
        }
        run({std::vector<int>(7936, 7)}, 1); // Exactly reaches input + output limit.
        GenerationConfig overlong; overlong.output_token_limit = 1;
        if (FetchResponseTokens(LaunchResponseTokens(std::vector<int>(7937, 7), overlong)) != -2)
            throw std::runtime_error("shared pool exceeded the per-request context limit");
        if (verifyGraphCaptures != captures) throw std::runtime_error("shared pool recaptured graphs");
        ShutdownRuntime();
        std::cout << (kvmem ? "KVMEM POOL PASS max_relative=" : "GPU POOL PASS max_relative=") << largestLogitRelative
                  << " max_abs=" << largestLogitAbsolute << std::endl;
    }
#endif
    void DraftPrefillOracle() {
        SetSaveHistoryChat(false);
        auto scores = [&](DraftContext &context) {
            Data hidden = RunDraft(17, context), head = RunDraftHead(hidden), owned(head);
            ToDataType(owned, FLOAT32); owned.ToDevice(DataDevice::CPU);
            return std::vector<float>((float *)owned.cpuData, (float *)owned.cpuData + owned.Count(0));
        };
        int checks = 0;
        for (int count : {2, 4, 8, 16, 24}) {
            std::vector<std::unique_ptr<ResponseContext>> reference(count), actual(count);
            for (int i = 0; i < count; ++i) for (auto *group : {&reference, &actual}) {
                auto &ctx = (*group)[i]; ctx.reset(new ResponseContext());
                ctx->Init(block_cnt, dataType, kvCacheDataType);
                OnResponseContextCreated(ctx.get());
            }
            std::vector<int> past(count, 0);
            for (int round = 0; round < 3; ++round) {
                std::vector<float> packed;
                std::vector<int> lengths(count);
                std::vector<Data> positions(count);
                std::vector<Data *> pos;
                std::vector<GenerationConfig> configs(count);
                std::vector<std::pair<Data *, Data *>> caches;
                std::vector<std::vector<float>> expected(count), draftExpected(count), observed(count);
                std::vector<std::vector<float> *> outputs;
                for (int i = 0; i < count; ++i) {
                    // One-token cold prompts, unequal boundaries, and long
                    // append after a short prefill all share the same batch.
                    const int choices[] = {1, 31, 257, 1023};
                    lengths[i] = round == 0 ? choices[i % 4] : round == 1 ? 1 + i % 5 : 129;
                    if (count == 2 && round == 0) lengths[i] = 1;
                    auto &config = configs[i];
                    config.input_token_length = past[i] + lengths[i];
                    config.output_token_limit = 32; config.output_logits = true;
                    config.top_k = 0; config.top_p = .95f; config.temperature = 0.7f + .1f * (i % 4);
                    std::vector<float> ids(lengths[i]), coords(lengths[i]);
                    for (int j = 0; j < lengths[i]; ++j) {
                        ids[j] = ((past[i] + j) * 7 + i * 13) % 251; coords[j] = past[i] + j;
                    }
                    positions[i].CopyFrom(Data(FLOAT32, {1, lengths[i]}, coords));
                    Forward(Data(FLOAT32, {1, lengths[i]}, ids), Data(), positions[i], reference[i]->pastKeyValues,
                        config, LastTokensManager(), &expected[i]);
                    draftExpected[i] = scores(*draftContexts.at(&reference[i]->pastKeyValues));
                    packed.insert(packed.end(), ids.begin(), ids.end());
                    pos.push_back(&positions[i]); outputs.push_back(&observed[i]);
                    for (auto &layer : actual[i]->pastKeyValues) caches.emplace_back(&layer.first, &layer.second);
                }
                ForwardBatch(count, Data(FLOAT32, {1, (int)packed.size()}, packed), {}, pos, lengths,
                    caches, configs, LastTokensManager(), &outputs);
                for (int i = 0; i < count; ++i) {
                    past[i] += lengths[i];
                    if (observed[i].size() != expected[i].size()) throw std::runtime_error("prefill logits count");
                    RequireCloseLogits(observed[i].data(), expected[i], "batched prefill target logits");
                    auto draft = draftContexts.at(&actual[i]->pastKeyValues);
                    if (!draft || draft->committed != past[i] || !draft->pending.empty())
                        throw std::runtime_error("batched prefill draft position or pending queue");
                    auto actualDraft = scores(*draft);
                    RequireCloseLogits(actualDraft.data(), draftExpected[i], "batched prefill draft logits");
                    ++checks;
                }
            }
            for (auto *group : {&reference, &actual}) for (auto &ctx : *group) OnResponseContextRemoved(ctx.get());
        }
        // Cancellation before the first forward leaves only an ownership entry.
        ResponseContext cancelled; cancelled.Init(block_cnt, dataType, kvCacheDataType);
        OnResponseContextCreated(&cancelled); OnResponseContextRemoved(&cancelled);
        if (draftContexts.count(&cancelled.pastKeyValues)) throw std::runtime_error("cancelled prefill retained draft owner");
        std::cout << "DSPARK PREFILL ORACLE PASS cases=" << checks
                  << " max_relative=" << largestLogitRelative << " max_abs=" << largestLogitAbsolute << std::endl;
    }

    void DraftPrefixChecks(bool sparse = false, bool batching = false) {
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        inspectDraftPrefixAllocations = true;
#endif
        draftEnabled = canDoConcurrentForward = true;
        draftLayers = 2; draftBlock = 7; draftTokens = 3;
        draftHeads = 8; draftKvHeads = 2; draftHeadDim = 32; draftWindow = 1024;
        draftConfidenceThreshold = 0; draftTargetLayers = {0, 1};
        AddDraftTestWeights(weight, draftLayers, draftKvHeads, 256, true);
        maxBatch = batching ? 48 : 16; SetMaxTokens(131072); SetChunkedPrefillSize(batching ? 8192 : 1024);
        if (batching && !SetPrefillBatching(4096, 20, 4, 24, 1))
            throw std::runtime_error("DSpark rejected existing prefill options");
        if (sparse) {
            KvMemConfig cfg; cfg.enabled = true; cfg.maxTokens = 8192; cfg.residentTokens = 8192;
            cfg.prefillTokens = 1024; cfg.hostBytes = 1ULL << 30; ConfigureKvMem(cfg);
        }
        WarmUp();
        if (batching) DraftPrefillOracle();
        useRealScheduler = true; eos_token_id = -1; eos_token_ids.clear();
        NaiveN05GraphTestAccess::CheckDraftWarmup(*this);
        auto &prefixPool = NaiveN05GraphTestAccess::DraftPrefixPool(*this);
        if (!prefixPool || prefixPool->draftStorage.size() != size_t(ranks))
            throw std::runtime_error("draft checkpoints are not distributed across TP ranks");
        int prefixSlots = 0;
        for (const auto &shard : prefixPool->draftStorage)
            if (!shard.dims.empty()) prefixSlots += shard.dims.front();
        size_t draftBytes = 0;
        for (int rank = 0; rank < ranks; ++rank) {
            auto &shard = prefixPool->draftStorage[rank];
            const size_t expected = size_t(PrefixPool::DraftShardSlots(prefixSlots, rank, ranks)) *
                prefixPool->draftSlotBytes;
            if (shard.GetBytes() != expected || (expected && shard.dataDeviceIds != std::vector<int>{rank}))
                throw std::runtime_error("draft checkpoint shard budget or device mismatch");
            draftBytes += shard.GetBytes();
        }
        if (draftBytes != size_t(prefixSlots) * prefixPool->draftSlotBytes)
            throw std::runtime_error("draft checkpoint sharding lost or duplicated slots");
        for (int length : {127, 1023, 4097}) {
            PrefixSnapshots snapshots;
            for (int rank = 0; rank < ranks; ++rank) {
                auto snapshot = std::make_shared<PrefixSnapshot>();
                snapshot->pool = prefixPool; snapshot->slot = prefixPool->freeSlots.back();
                prefixPool->freeSlots.pop_back(); snapshots.push_back(snapshot);
                const int rows = std::min(length, draftWindow - 1);
                std::vector<uint16_t> expected(size_t(rows) * embed_dim), actual(expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                    expected[i] = Float32ToBFloat16RNEBits(std::sin((i + rank * 7 + length) * .019f));
                auto &shard = prefixPool->DraftShard(snapshot->slot);
                FastllmCudaSetDevice(shard.dataDeviceIds.front());
                FastllmCudaCopyFromHostToDevice((uint8_t *)shard.cudaData + prefixPool->DraftOffset(snapshot->slot),
                                               expected.data(), expected.size() * sizeof(uint16_t));
                auto restored = CreateDraftContext();
                RestoreDraftPrefix(*snapshot, length, *restored);
                FastllmCudaCopyFromDeviceToHost(actual.data(), restored->prefixHidden.cudaData,
                                               actual.size() * sizeof(uint16_t));
                if (actual != expected || restored->committed != length)
                    throw std::runtime_error("distributed draft checkpoint changed normalized inputs or position");
            }
        }
        auto prompt = [](int length, int salt = 0) {
            std::vector<int> result(length);
            for (int i = 0; i < length; ++i) result[i] = (i * 3 + salt) % 251;
            return result;
        };
        auto infer = [&](const std::vector<int> &tokens, std::vector<float> &logits) {
            GenerationConfig cfg; cfg.top_k = 1; cfg.output_logits = true;
            cfg.output_token_limit = 1; cfg.input_token_length = tokens.size();
            int handle = LaunchResponseTokens(tokens, cfg);
            if (FetchResponseLogits(handle, logits) < 0) throw std::runtime_error("draft prefix request failed");
            std::vector<float> unused;
            if (FetchResponseLogits(handle, unused) >= 0) throw std::runtime_error("draft prefix did not finish");
        };
        inspectDraftPrefix = true;
        for (int length : {127, 128, 1023, 1024, 1151, 1152, 2048, 2113, 4097}) {
            SetSaveHistoryChat(false); SetSaveHistoryChat(true);
            const auto tokens = prompt(length);
            std::vector<float> cold, hot;
            infer(tokens, cold); const auto draft = observedDraft;
            if (draft.empty() || std::any_of(draft.begin(), draft.end(), [](float x) { return !std::isfinite(x); }))
                throw std::runtime_error("cold draft logits are non-finite at prefix " + std::to_string(length));
            const int before = pagedRestores;
            infer(tokens, hot);
            if (length > 128 && (pagedRestores != before + 1 ||
                lastPrefixLength != (length - 1) / GetPageLen() * GetPageLen()))
                throw std::runtime_error("draft prefix did not restore a complete checkpoint");
            RequireCloseLogits(hot.data(), cold, "draft prefix target logits");
            if (observedDraft.empty() || observedDraft.size() != draft.size())
                throw std::runtime_error("draft prefix probabilities missing");
            RequireCloseLogits(observedDraft.data(), draft, "restored draft logits");
            auto fork = tokens; fork.back() = (fork.back() + 13) % 251; fork.push_back(252);
            infer(fork, hot); const auto forkDraft = observedDraft;
            SetSaveHistoryChat(false); infer(fork, cold);
            RequireCloseLogits(hot.data(), cold, "forked speculative target logits");
            if (forkDraft.empty() || observedDraft.empty())
                throw std::runtime_error("fork draft logits missing at length " + std::to_string(length) +
                    " hot=" + std::to_string(forkDraft.size()) + " cold=" + std::to_string(observedDraft.size()));
            RequireCloseLogits(forkDraft.data(), observedDraft, "forked draft logits");
        }
        inspectDraftPrefix = false;
        // Exercise actual speculative verification across a page boundary,
        // with independent request lengths, shared prefixes and request churn.
        auto batch = [&](const std::vector<std::vector<int>> &prompts, bool sampled = false) {
            std::vector<int> handles;
            for (const auto &tokens : prompts) {
                GenerationConfig cfg; cfg.top_k = sampled ? 0 : 1;
                // Different completion lengths repeatedly shrink the TP draft
                // graph's request tuple while other sampled requests continue.
                cfg.output_token_limit = sampled ? 48 + (int)(handles.size() % 4) * 17 : 20;
                cfg.temperature = 1.f; cfg.top_p = .95f;
                cfg.input_token_length = tokens.size();
                handles.push_back(LaunchResponseTokens(tokens, cfg));
            }
            std::vector<std::vector<int>> result(handles.size());
            for (size_t i = 0; i < handles.size(); ++i) {
                int token;
                while ((token = FetchResponseTokens(handles[i])) >= 0) result[i].push_back(token);
                if (result[i].size() != (sampled ? 48 + i % 4 * 17 : 20))
                    throw std::runtime_error("speculative prefix batch ended early");
            }
            return result;
        };
        for (int proposed : {1, 3, 5, 7}) for (int count : {1, 2, 8, 16}) {
            draftTokens = proposed;
            std::vector<std::vector<int>> prompts;
            for (int i = 0; i < count; ++i) prompts.push_back(prompt(2040 + i % 4, i % 3));
            SetSaveHistoryChat(false); SetSaveHistoryChat(true);
            batch(prompts); const int before = pagedRestores;
            auto hot = batch(prompts);
            if (pagedRestores - before != count) throw std::runtime_error("concurrent draft prefix miss");
            SetSaveHistoryChat(false); auto cold = batch(prompts);
            for (int row = 0; row < count; ++row) if (hot[row] != cold[row]) {
                // Different prefill GEMM shapes can round near-tied BF16 logits
                // differently (also with FP8 KV). Compare the same teacher-forced
                // history at the first divergence, never unrelated continuations.
                size_t first = 0;
                while (hot[row][first] == cold[row][first]) ++first;
                auto probe = prompts[row];
                probe.insert(probe.end(), hot[row].begin(), hot[row].begin() + first);
                SetSaveHistoryChat(true); batch({prompts[row]});
                std::vector<float> h, c; infer(probe, h);
                SetSaveHistoryChat(false); infer(probe, c);
                RequireCloseLogits(h.data(), c, "speculative prefix first divergence");
                for (const auto *values : {&h, &c}) {
                    const float best = *std::max_element(values->begin(), values->end());
                    const float ulp = std::ldexp(1.f, std::ilogb(std::fabs(best)) - 7);
                    for (int token : {hot[row][first], cold[row][first]})
                        if (best - values->at(token) > 2 * ulp + 1e-7f)
                            throw std::runtime_error("speculative cached batch changed a non-tied greedy token");
                }
                std::cout << "DSPARK PREFIX near_tie batch=" << count << " draft=" << proposed
                          << " row=" << row << " token=" << first << std::endl;
            }
            std::cout << "DSPARK PREFIX batch=" << count << " draft=" << proposed << " pass\n";
        }
        // Grow and reuse proposal/acceptance arenas after a block7/prefix3
        // warmup. Stochastic acceptance naturally produces ragged verify groups.
        for (int proposed : {3, 5, 7, 3}) for (int count : {2, 8, 16}) {
            draftTokens = proposed;
            std::vector<std::vector<int>> prompts;
            for (int i = 0; i < count; ++i) prompts.push_back(prompt(2040 + i % 4, i % 3));
            SetSaveHistoryChat(false); SetSaveHistoryChat(true);
            batch(prompts, true); const int before = pagedRestores;
            batch(prompts, true);
            if (pagedRestores - before != count) throw std::runtime_error("sampled draft prefix miss");
        }
        ShutdownRuntime();
        if (batching && !batchedPrefills) throw std::runtime_error("prefill batching was never exercised");
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        idleDraftContext.reset();
        inspectDraftPrefixAllocations = false;
        std::lock_guard<std::mutex> guard(draftPrefixAllocationMutex);
        if (!draftPrefixAllocations.empty())
            throw std::runtime_error("DSpark leaked " + std::to_string(draftPrefixAllocations.size()) +
                                     " prefill hidden allocations after request teardown");
#endif
        std::cout << "DSPARK PREFIX PASS target/draft logits, forks, page rollback, batches and draft lengths"
                  << " max_relative=" << largestLogitRelative << " max_abs=" << largestLogitAbsolute << std::endl;
    }

    void HistoryServingChecks(bool graph = true, bool sparse = false) {
        maxBatch = 32; SetMaxTokens(262144); SetChunkedPrefillSize(128);
        if (sparse) {
            KvMemConfig cfg; cfg.enabled = true; cfg.maxTokens = 7937;
            cfg.residentTokens = 2048; cfg.prefillTokens = 128; cfg.hostBytes = 1ULL << 30;
            ConfigureKvMem(cfg);
        }
        WarmUp();
        useRealScheduler = true;
        eos_token_id = -1; eos_token_ids.clear();
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        const int captures = verifyGraphCaptures;
#endif
        auto infer = [&](const std::vector<int> &tokens, int count,
                         std::vector<std::vector<float>> &logits) {
            GenerationConfig cfg;
            cfg.top_k = 1; cfg.output_logits = true;
            cfg.output_token_limit = count;
            cfg.input_token_length = tokens.size();
            int handle = LaunchResponseTokens(tokens, cfg);
            std::vector<int> sampled;
            for (int i = 0; i < count; ++i) {
                logits.emplace_back();
                int token = FetchResponseLogits(handle, logits.back());
                if (token < 0) throw std::runtime_error("prefix response ended early");
                sampled.push_back(token);
            }
            std::vector<float> unused;
            if (FetchResponseLogits(handle, unused) >= 0)
                throw std::runtime_error("prefix response did not end");
            return sampled;
        };
        for (int prefix : {128, 256, 2048, 4096}) {
            SetSaveHistoryChat(false); SetSaveHistoryChat(true);
            std::vector<int> tokens(prefix);
            for (int i = 0; i < prefix; ++i) tokens[i] = i * 3 % 251;
            std::vector<std::vector<float>> uninterrupted, restored;
            auto sampled = infer(tokens, 2, uninterrupted);
            tokens.push_back(sampled[0]);
            const int gpuBefore = pagedRestores;
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            const int before = verifyGraphLaunches;
#endif
            infer(tokens, 1, restored);
            if (pagedRestores != gpuBefore + 1 || hostRestores != 0)
                throw std::runtime_error("resident prefix did not use GPU state");
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            if (graph && (verifyGraphLaunches - before != ranks || verifyGraphCaptures != captures))
                throw std::runtime_error("restored prefix did not replay its prepared graph");
#endif
            if (restored[0] != uninterrupted[1])
                throw std::runtime_error("restored serving graph changed logits");
            // A partial TP page hit must recompute, never restore a host copy.
            auto *pool = GetPagedKVCacheManager(0, true);
            std::vector<int> pressure;
            for (int i = 0; i < pool->maxPages; ++i) pressure.push_back(pool->GetUnusedPageIndex(true));
            pool->ReleasePageIndices(pressure);
            std::vector<std::vector<float>> fallback;
            infer(tokens, 1, fallback);
            if (hostRestores != 0 || pagedRestores != gpuBefore + 1)
                throw std::runtime_error("partial TP prefix was incorrectly reused");
            RequireCloseLogits(fallback[0].data(), uninterrupted[1], "evicted prefix logits");
        }
        // More than five independent conversations must retain usable GPU
        // prefixes. Check exact repeats and forks within a page against cold
        // prefill, exercising the saved SWA boundary and unaligned suffixes.
        SetSaveHistoryChat(false); SetSaveHistoryChat(true);
        std::vector<std::vector<int>> prompts(50, std::vector<int>(1024));
        for (int i = 0; i < 50; ++i) {
            for (int j = 0; j < 1024; ++j) prompts[i][j] = (i * 7 + j * 3) % 251;
            std::vector<std::vector<float>> unused;
            infer(prompts[i], 1, unused);
        }
        const int hits = pagedRestores;
        for (auto tokens : prompts) {
            std::vector<std::vector<float>> hot;
            infer(tokens, 1, hot);
        }
        if (pagedRestores - hits != 50) throw std::runtime_error("GPU prefix lost one of 50 conversations");
        // Fill the bounded window pool beyond capacity while full DSA pages
        // remain resident. An evicted window must force safe recomputation.
        for (int i = 0; i < 130; ++i) {
            auto tokens = prompts.front(); tokens[0] = 252; tokens[1] = i;
            std::vector<std::vector<float>> unused;
            infer(tokens, 1, unused);
        }
        std::vector<int> residentPages;
        GetPagedKVCacheManager(0, true)->Query(prompts.front(), residentPages);
        if ((int)residentPages.size() * GetPageLen() < (int)prompts.front().size())
            throw std::runtime_error("window pressure test also evicted full DSA pages");
        const int beforePressure = pagedRestores;
        std::vector<std::vector<float>> evicted, coldPressure;
        infer(prompts.front(), 1, evicted);
        if (pagedRestores != beforePressure)
            throw std::runtime_error("evicted SWA window was reused");
        SetSaveHistoryChat(false);
        infer(prompts.front(), 1, coldPressure);
        RequireCloseLogits(evicted[0].data(), coldPressure[0], "window pool eviction logits");
        SetSaveHistoryChat(true);
        for (int boundary : {127, 128, 767, 768, 800, 895, 896, 1023, 1024}) {
            auto tokens = prompts.back();
            tokens.resize(boundary);
            tokens.push_back(252);
            std::vector<std::vector<float>> hot, cold;
            infer(tokens, 1, hot);
            SetSaveHistoryChat(false);
            infer(tokens, 1, cold);
            RequireCloseLogits(hot[0].data(), cold[0], "forked prefix logits");
            SetSaveHistoryChat(true);
            std::vector<std::vector<float>> unused;
            infer(prompts.back(), 1, unused);
        }
        // More pending forks than hot slots: the shared immutable prefix must
        // not consume a private slot until admission. Each fork then appends
        // different tokens without changing its siblings' pages or Indexer.
        SetSaveHistoryChat(false); SetSaveHistoryChat(true);
        std::vector<int> shared(3072);
        for (int i = 0; i < (int)shared.size(); ++i) shared[i] = i * 3 % 251;
        std::vector<std::vector<float>> seed;
        infer(shared, 1, seed);
        const int forkCount = sparse ? maxBatch + 8 : 16, forkSteps = 4, beforeForks = pagedRestores;
        std::vector<int> forks;
        std::vector<std::vector<int>> forkPrompts(forkCount);
        std::vector<std::vector<std::vector<float>>> forkLogits(forkCount);
        for (int row = 0; row < forkCount; ++row) {
            auto &prompt = forkPrompts[row]; prompt = shared;
            prompt.push_back(252); prompt.push_back(row);
            GenerationConfig cfg; cfg.top_k = 1; cfg.output_logits = true;
            cfg.output_token_limit = forkSteps; cfg.input_token_length = prompt.size();
            forks.push_back(LaunchResponseTokens(prompt, cfg));
        }
        for (int row = 0; row < forkCount; ++row) {
            for (int step = 0; step < forkSteps; ++step) {
                forkLogits[row].emplace_back();
                if (FetchResponseLogits(forks[row], forkLogits[row].back()) < 0)
                    throw std::runtime_error("shared prefix fork ended early");
            }
            std::vector<float> unused;
            if (FetchResponseLogits(forks[row], unused) >= 0)
                throw std::runtime_error("shared prefix fork did not end");
        }
        if (pagedRestores - beforeForks != forkCount)
            throw std::runtime_error("shared prefix fork missed resident pages");
        SetSaveHistoryChat(false);
        for (int row = 0; row < forkCount; ++row) {
            std::vector<std::vector<float>> cold;
            infer(forkPrompts[row], forkSteps, cold);
            for (int step = 0; step < forkSteps; ++step)
                RequireCloseLogits(forkLogits[row][step].data(), cold[step], "shared prefix concurrent logits");
        }
        // Capture a boundary during decode, then resume its output prefix.
        SetSaveHistoryChat(false); SetSaveHistoryChat(true);
        auto tokens = prompts.front(); tokens.resize(240);
        std::vector<std::vector<float>> decoded, resumed;
        auto sampled = infer(tokens, 18, decoded);
        tokens.insert(tokens.end(), sampled.begin(), sampled.begin() + 17);
        const int beforeResume = pagedRestores;
        infer(tokens, 1, resumed);
        if (pagedRestores != beforeResume + 1 || lastPrefixLength != 256 || hostRestores != 0)
            throw std::runtime_error("decode checkpoint was not reusable");
        RequireCloseLogits(resumed[0].data(), decoded[17], "decode checkpoint logits");
        // A failure after restoring SWA must affect only this request, and its
        // partial allocation must be released before a later cache hit.
        failPrefixRestore = true;
        GenerationConfig failedConfig;
        failedConfig.top_k = 1; failedConfig.output_token_limit = 1;
        int failedHandle = LaunchResponseTokens(tokens, failedConfig);
        if (FetchResponseTokens(failedHandle) != -2 || failPrefixRestore)
            throw std::runtime_error("prefix allocation error was not isolated");
        std::vector<std::vector<float>> recovered;
        infer(tokens, 1, recovered);
        if (recovered != resumed)
            throw std::runtime_error("prefix allocation failure changed the next request");
        // Ragged requests cross the same page on different decode steps, with
        // several checkpoints submitted together on each rank. Every saved
        // window must restore its own request after the batch has finished.
        SetSaveHistoryChat(false); SetSaveHistoryChat(true);
        const int batchSize = 16, outputs = 40;
        std::vector<std::vector<int>> batchPrompts(batchSize), batchTokens(batchSize);
        std::vector<std::vector<float>> batchLastLogits(batchSize);
        std::vector<int> handles;
        for (int row = 0; row < batchSize; ++row) {
            auto &prompt = batchPrompts[row];
            prompt.resize(232 + (row % 4) * 4);
            for (int j = 0; j < (int)prompt.size(); ++j) prompt[j] = (row * 7 + j * 3) % 251;
            GenerationConfig cfg;
            cfg.top_k = 1; cfg.output_logits = true;
            cfg.output_token_limit = outputs; cfg.input_token_length = prompt.size();
            handles.push_back(LaunchResponseTokens(prompt, cfg));
        }
        for (int step = 0; step < outputs; ++step) for (int row = 0; row < batchSize; ++row) {
            auto &logits = batchLastLogits[row];
            logits.clear();
            int token = FetchResponseLogits(handles[row], logits);
            if (token < 0) throw std::runtime_error("batched prefix response ended early");
            batchTokens[row].push_back(token);
        }
        for (int handle : handles) {
            std::vector<float> unused;
            if (FetchResponseLogits(handle, unused) >= 0)
                throw std::runtime_error("batched prefix response did not end");
        }
        for (int row = 0; row < batchSize; ++row) {
            auto prompt = batchPrompts[row];
            prompt.insert(prompt.end(), batchTokens[row].begin(), batchTokens[row].end() - 1);
            const int beforeHit = pagedRestores;
            std::vector<std::vector<float>> hot;
            infer(prompt, 1, hot);
            if (pagedRestores != beforeHit + 1 || lastPrefixLength != 256 || hostRestores != 0)
                throw std::runtime_error("batched decode checkpoint was not reusable");
            RequireCloseLogits(hot[0].data(), batchLastLogits[row], "batched decode checkpoint logits");
        }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        if (verifyGraphCaptures != captures) throw std::runtime_error("prefix serving recaptured a graph");
#endif
        std::cout << "TP GPU PREFIX PASS ranks=" << ranks << " graph=" << graph
                  << " max_relative=" << largestLogitRelative << " max_abs=" << largestLogitAbsolute << std::endl;
    }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    void CudaSamplingChecks() {
        // Compare empirical GPU probabilities with the existing CPU sampling
        // oracle, with a different temperature/top-p for each group of rows.
        const int vocab = 256, groups = 6, repeats = 4096;
        std::vector<float> values(groups * repeats * vocab);
        std::vector<GenerationConfig> configs(groups * repeats);
        std::vector<std::vector<float>> expected(groups);
        for (int group = 0; group < groups; ++group) {
            auto &cfg = configs[group * repeats];
            cfg.top_k = vocab; cfg.temperature = .6f + .3f * (group % 3);
            cfg.top_p = group < 3 ? .73f : 1.0f;
            std::vector<float> base(vocab);
            for (int token = 0; token < vocab; ++token)
                base[token] = -.25f * ((token * 73) % vocab);
            expected[group] = SpeculativeDistribution(base.data(), vocab, cfg, LastTokensUnit());
            for (int i = 0; i < repeats; ++i) {
                int row = group * repeats + i;
                configs[row] = cfg;
                std::copy(base.begin(), base.end(), values.begin() + row * vocab);
            }
        }
        Data probabilities(FLOAT32, {groups * repeats, vocab}, values);
        probabilities.ToDevice(DataDevice::CUDA, {0}, true);
        inspectGpuSampling = true;
        auto sampled = SampleCudaLogits(probabilities, configs);
        if (sampledGpuLogits != values) throw std::runtime_error("input logits changed");
        inspectGpuSampling = false;
        for (int group = 0; group < groups; ++group) {
            std::vector<int> counts(vocab);
            for (int row = group * repeats; row < (group + 1) * repeats; ++row) {
                int token = sampled[row];
                if (token < 0 || token >= vocab || expected[group][token] <= 0)
                    throw std::runtime_error("GPU draw outside CPU nucleus support");
                ++counts[token];
            }
            for (int token = 0; token < vocab; ++token)
                if (std::abs((double)counts[token] / repeats - expected[group][token]) > .035)
                    throw std::runtime_error("GPU sampling frequency differs from CPU distribution");
        }
        GenerationConfig cfg; cfg.top_k = vocab; cfg.top_p = .95f;
        if (!SelectLogits(cfg).cudaSampling || SelectLogits(cfg, true).cudaSampling)
            throw std::runtime_error("incorrect ordinary/speculative sampling selection");
        for (int k : {1, 4, 50}) {
            cfg.top_k = k;
            if (SelectLogits(cfg).cudaSampling) throw std::runtime_error("finite top-k distribution changed");
        }
        cfg.top_k = vocab; cfg.output_token_limit = 32; cfg.input_token_length = 3;
        maxBatch = 128; SetMaxTokens(131072);
        WarmUp();
        int captures = verifyGraphCaptures, checks = 0;
        using Cache = std::vector<std::pair<Data, Data>>;
        std::vector<Cache> actual(48), reference(48);
        std::vector<int> past(48, 3);
        inspectGpuSampling = true;
        for (int row = 0; row < 48; ++row) {
            for (int layer = 0; layer < block_cnt; ++layer) {
                actual[row].emplace_back(Data(BFLOAT16), Data(BFLOAT16));
                reference[row].emplace_back(Data(BFLOAT16), Data(BFLOAT16));
            }
            Data ids(FLOAT32, {1, 3}, {float(row + 1), 2, 3});
            Data positions(FLOAT32, {1, 3}, {0, 1, 2});
            auto ref = cfg; ref.output_logits = true;
            std::vector<float> logits;
            Forward(ids, Data(), positions, reference[row], ref, LastTokensManager(), &logits);
            int calls = gpuSamplingCalls;
            Forward(ids, Data(), positions, actual[row], cfg);
            if (gpuSamplingCalls != calls + 1 || sampledGpuLogits != logits)
                throw std::runtime_error("prefill GPU sampling logits differ from full CPU output");
            ++checks;
        }
        for (int rows : {1, 4, 5, 7, 9, 16, 31, 39, 41, 44, 48, 40, 1}) {
            if (ranks == 1 && rows != 1) continue;
            std::vector<Data> positions(rows);
            std::vector<Data *> pos;
            std::vector<float> ids;
            std::vector<std::pair<Data *, Data *>> a, r;
            std::vector<GenerationConfig> cs(rows, cfg), refs(rows, cfg);
            std::vector<std::vector<float>> expectedRows(rows);
            std::vector<std::vector<float> *> outputs;
            for (int row = 0; row < rows; ++row) {
                ids.push_back(float((past[row] * 3 + row) % 251));
                positions[row].CopyFrom(Data(FLOAT32, {1, 1}, {float(past[row])}));
                pos.push_back(&positions[row]); outputs.push_back(&expectedRows[row]);
                cs[row].temperature = .5f + .3f * (row % 4); cs[row].top_p = .7f + .07f * (row % 4);
                refs[row] = cs[row]; refs[row].output_logits = true;
                for (int layer = 0; layer < block_cnt; ++layer) {
                    a.emplace_back(&actual[row][layer].first, &actual[row][layer].second);
                    r.emplace_back(&reference[row][layer].first, &reference[row][layer].second);
                }
            }
            Data input(FLOAT32, {1, rows}, ids);
            int calls = gpuSamplingCalls, launches = verifyGraphLaunches;
            if (rows == 1) {
                Forward(input, Data(), positions[0], reference[0], refs[0], LastTokensManager(), &expectedRows[0]);
                Forward(input, Data(), positions[0], actual[0], cs[0]);
            } else {
                ForwardBatch(rows, input, {}, pos, std::vector<int>(rows, 1), r, refs, LastTokensManager(), &outputs);
                ForwardBatch(rows, input, {}, pos, std::vector<int>(rows, 1), a, cs);
            }
            std::vector<float> expectedFlat;
            for (auto &row : expectedRows) expectedFlat.insert(expectedFlat.end(), row.begin(), row.end());
            if (gpuSamplingCalls != calls + 1 || sampledGpuLogits != expectedFlat)
                throw std::runtime_error("decode GPU sampling logits differ from full CPU output");
            if (ranks > 1 && verifyGraphLaunches - launches != ranks * 2) {
                std::cerr << "rows=" << rows << " launches=" << verifyGraphLaunches - launches
                          << " expected=" << ranks * 2 << std::endl;
                throw std::runtime_error("CUDA sampling lost serving graph replay");
            }
            for (int row = 0; row < rows; ++row) ++past[row];
            ++checks;
        }
        inspectGpuSampling = false;
        if (verifyGraphCaptures != captures) throw std::runtime_error("sampling parameter change recaptured graph");
        std::cout << "CUDA SAMPLING PASS ranks=" << ranks << " draws=" << sampled.size()
                  << " model_checks=" << checks << " no_recapture=1" << std::endl;
    }
    void ServingGraphInvalidationChecks() {
        maxBatch = 4; SetMaxTokens(16384); WarmUp();
        const int captures = verifyGraphCaptures;
        std::vector<std::pair<Data, Data>> cache;
        for (int layer = 0; layer < block_cnt; ++layer) cache.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
        GenerationConfig config;
        config.top_k = 1; config.repeat_penalty = 1;
        config.input_token_length = 3; config.output_token_limit = 32;
        int past = 0;
        auto run = [&](int rows) {
            std::vector<float> ids(rows), positions(rows);
            for (int i = 0; i < rows; ++i) { ids[i] = (past + i) * 3 % 251; positions[i] = past + i; }
            Forward(Data(FLOAT32, {1, rows}, ids), Data(), Data(FLOAT32, {1, rows}, positions), cache, config);
            past += rows;
        };
        run(3);
        const int before = verifyGraphLaunches;
        run(1);
        if (verifyGraphLaunches - before != ranks) throw std::runtime_error("initial serving graph did not replay");
        const int launches = verifyGraphLaunches;
        // An incompatible setting can first be observed on a prefill call.
        // Restoring it must not revive a graph whose resources may have moved.
        SetMoeAtype(FLOAT16); run(2);
        SetMoeAtype(BFLOAT16); run(1);
        if (verifyGraphCaptures != captures || verifyGraphLaunches != launches)
            throw std::runtime_error("incompatible serving graph was revived");
        std::cout << "SERVING INVALIDATION PASS" << std::endl;
    }
    void ServingGraphChecks(bool padding = false, bool ragged = false, bool large = false) {
        // An unaligned admission limit must still prepare the padded endpoint.
        // Run graph/reference sequentially so both obey that request limit.
        maxBatch = large ? 401 : ragged ? 32 : padding ? 47 : 16;
        SetChunkedPrefillSize(4096);
        SetMaxTokens(large ? 4194304 : ragged ? 65536 : padding ? 1048576 : 262144);
        WarmUp();
        if (!verifyGraphCaptures) throw std::runtime_error("serving warmup captured no graphs");
        const int captures = verifyGraphCaptures;
        int checks = 0, replayed = 0, eager = 0;
        for (int prefix : (ragged ? std::vector<int>{3070} : std::vector<int>{3, 254, 2046, 4094})) {
            using Cache = std::vector<std::pair<Data, Data>>;
            const int requests = large ? maxBatch : ragged ? 9 : padding ? 47 : 8;
            std::vector<std::vector<std::vector<float>>> capturedLogits;
            std::vector<std::vector<std::vector<int>>> capturedMetadata;
            for (bool graph : {true, false}) {
                std::vector<Cache> actual(requests);
                std::vector<int> past(requests, prefix);
                // Nine requests pad to 16. The longest exceeds pool / 16 = 4096,
                // while the total live KV fits the pool.
                if (ragged) past[0] += 3072;
                GenerationConfig config;
                config.input_token_length = prefix;
                config.output_token_limit = 64;
                config.top_k = 1; config.repeat_penalty = 1; config.output_logits = true;
                auto prefill = [&](Cache &cache, int row) {
                    for (int i = 0; i < block_cnt; ++i) cache.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
                    const int length = past[row];
                    std::vector<float> ids(length), positions(length);
                    for (int i = 0; i < length; ++i) { ids[i] = (i * 3 + row) % 251; positions[i] = i; }
                    Forward(Data(FLOAT32, {1, length}, ids), Data(), Data(FLOAT32, {1, length}, positions), cache, config);
                };
                SetCudaGraph(graph);
                for (int row = 0; row < requests; ++row) prefill(actual[row], row);
                int step = 0;
                const std::vector<int> counts = large ? std::vector<int>{257, 263, 399, 400, 401, 256, 8} : ragged ? std::vector<int>{9, 8, 1, 9, 7, 9, 3, 9} : padding
                    ? std::vector<int>{5, 7, 9, 15, 17, 31, 33, 39, 41, 44, 47, 40, 46, 1, 43}
                    : std::vector<int>{8, 4, 3, 2, 1, 4, 8, 7, 8};
                for (int count : counts) {
                    std::vector<int> order(count);
                    for (int row = 0; row < count; ++row) order[row] = step % 2 ? requests - 1 - row : row;
                    std::vector<float> ids;
                    std::vector<Data> positions(count);
                    std::vector<Data *> pos;
                    std::vector<std::pair<Data *, Data *>> caches;
                    std::vector<std::vector<float>> logits(count);
                    std::vector<std::vector<float> *> outputs;
                    for (int row = 0; row < count; ++row) {
                        int sequence = order[row];
                        ids.push_back((past[sequence] * 3 + sequence) % 251);
                        positions[row].CopyFrom(Data(FLOAT32, {1, 1}, {(float)past[sequence]}));
                        pos.push_back(&positions[row]); outputs.push_back(&logits[row]);
                        for (auto &layer : actual[sequence]) caches.emplace_back(&layer.first, &layer.second);
                    }
                    const int before = verifyGraphLaunches;
                    ForwardBatch(count, Data(FLOAT32, {1, count}, ids), {}, pos, std::vector<int>(count, 1),
                                 caches, std::vector<GenerationConfig>(count, config), LastTokensManager(), &outputs);
                    int launched = verifyGraphLaunches - before;
                    if (launched != 0 && launched != ranks) throw std::runtime_error("partial-rank serving replay");
                    if (graph && (padding || large) && !failVerifyInstantiate && (prefix == 3 || prefix == 4094) && launched != ranks)
                        throw std::runtime_error("padded serving batch did not replay");
                    if (graph && ragged && !failVerifyInstantiate && launched != ranks)
                        throw std::runtime_error("ragged serving batch did not replay");
                    if (graph) { if (launched) ++replayed; else ++eager; }
                    std::vector<std::vector<int>> metadata;
                    for (int row = 0; row < count; ++row) {
                        if (!graph) RequireCloseLogits(capturedLogits[step][row].data(), logits[row],
                                                       "serving graph changed logits");
                        int sequence = order[row]; ++past[sequence];
                        for (int layer = 0; layer < block_cnt; ++layer)
                        for (int part = 0; part < 2; ++part) {
                            auto &a = part ? actual[sequence][layer].second : actual[sequence][layer].first;
                            metadata.push_back(a.dims);
                            for (int rank = 0; rank < ranks; ++rank)
                                metadata.push_back(a.multiDeviceDatas.at(rank)->dims);
                        }
                        if (!graph) ++checks;
                    }
                    if (graph) {
                        capturedLogits.push_back(std::move(logits));
                        capturedMetadata.push_back(std::move(metadata));
                    }
                    else if (capturedMetadata[step] != metadata)
                        throw std::runtime_error("serving KV metadata mismatch");
                    if (verifyGraphCaptures != captures) throw std::runtime_error("serving recaptured a graph");
                    ++step;
                }
            }
        }
        SetCudaGraph(true);
        // Exercise the owned scheduler's cached validation plan at a real
        // allocation boundary: unlimited generation reserves only the prompt.
        GenerationConfig growing;
        growing.top_k = 1; growing.repeat_penalty = 1;
        growing.input_token_length = 126; growing.output_token_limit = -1;
        std::vector<int> prompt(126), expectedTokens;
        std::vector<float> ids(126), positions(126);
        for (int i = 0; i < 126; ++i) { prompt[i] = (i * 3) % 251; ids[i] = prompt[i]; positions[i] = i; }
        std::vector<std::pair<Data, Data>> reference;
        for (int layer = 0; layer < block_cnt; ++layer) reference.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
        SetCudaGraph(false);
        int token = Forward(Data(FLOAT32, {1, 126}, ids), Data(), Data(FLOAT32, {1, 126}, positions), reference, growing);
        expectedTokens.push_back(token);
        for (int i = 0; i < 7; ++i) {
            token = Forward(Data(FLOAT32, {1, 1}, {(float)token}), Data(),
                            Data(FLOAT32, {1, 1}, {(float)(126 + i)}), reference, growing);
            expectedTokens.push_back(token);
        }
        SetCudaGraph(true);
        eos_token_id = -1; eos_token_ids.clear();
        useRealScheduler = true;
        const int handle = LaunchResponseTokens(prompt, growing);
        for (int expected : expectedTokens)
            if (FetchResponseTokens(handle) != expected) throw std::runtime_error("scheduler KV growth changed tokens");
        AbortResponse(handle);
        ShutdownRuntime();
        if (verifyGraphCaptures != captures) throw std::runtime_error("scheduler KV growth recaptured a graph");
        if ((!failVerifyInstantiate && replayed < (large ? 14 : ragged ? 8 : 20)) ||
            (failVerifyInstantiate && replayed != 0) || (!ragged && eager == 0))
            throw std::runtime_error("missing serving replay/fallback coverage");
        std::cout << "SERVING GRAPH PASS logits=" << checks << " replayed=" << replayed
                  << " eager=" << eager << " serving_captures=" << verifyGraphCaptures - captures
                  << " relative_max=" << largestLogitRelative << std::endl;
    }
    void ServingPageGrowthChecks() {
        maxBatch = 8; SetMaxTokens(131072); SetChunkedPrefillSize(4096); WarmUp();
        const int captures = verifyGraphCaptures;
        std::vector<float> reference;
        std::exception_ptr error;
        std::atomic<bool> ready{false};
        int growthSteps = 0;
        // Exercise the owned-thread fast path, with identical teacher-forced
        // tokens and graph shapes. Invalidating every reference step retains
        // the full validation/allocation path as an exact numerical oracle.
        mainLoop = new std::thread([&]() {
            while (!ready.load()) std::this_thread::yield();
            try {
                for (bool forceValidation : {true, false}) {
                    std::vector<std::vector<std::pair<Data, Data>>> cache(8);
                    ResponseContext invalidation;
                    invalidation.Init(block_cnt, dataType, kvCacheDataType);
                    GenerationConfig config;
                    config.top_k = 1; config.repeat_penalty = 1; config.output_logits = true;
                    config.output_token_limit = 256;
                    std::vector<int> past(8);
                    for (int row = 0; row < 8; ++row) {
                        past[row] = 3072 + row * 17;
                        std::vector<float> ids(past[row]), positions(past[row]);
                        for (int i = 0; i < past[row]; ++i) { ids[i] = (i * 3 + row) % 251; positions[i] = i; }
                        for (int i = 0; i < block_cnt; ++i)
                            cache[row].emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
                        Forward(Data(FLOAT32, {1, past[row]}, ids), Data(),
                            Data(FLOAT32, {1, past[row]}, positions), cache[row], config);
                    }
                    size_t offset = 0;
                    for (int step = 0; step < 256; ++step) {
                        std::vector<float> ids(8);
                        std::vector<Data> positions(8);
                        std::vector<Data *> pos;
                        std::vector<std::pair<Data *, Data *>> kv;
                        std::vector<std::vector<float>> logits(8);
                        std::vector<std::vector<float> *> output;
                        std::vector<GenerationConfig> configs(8, config);
                        std::vector<int> lengths(8, 1);
                        for (int row = 0; row < 8; ++row) {
                            ids[row] = (past[row] * 3 + row) % 251;
                            positions[row].CopyFrom(Data(FLOAT32, {1, 1}, {(float)past[row]}));
                            pos.push_back(&positions[row]); output.push_back(&logits[row]);
                            for (auto &layer : cache[row]) kv.emplace_back(&layer.first, &layer.second);
                        }
                        if (forceValidation) OnResponseContextCreated(&invalidation);
                        else {
                            bool growing = false;
                            for (int row = 0; row < 8; ++row) {
                                const auto &local = *cache[row][0].first.multiDeviceDatas.begin()->second;
                                growing |= past[row] >= (int)local.pageIndex.size() * local.pageLen;
                            }
                            growthSteps += growing;
                        }
                        ForwardBatch(8, Data(FLOAT32, {1, 8}, ids), {}, pos, lengths, kv,
                                     configs, LastTokensManager(), &output);
                        for (int row = 0; row < 8; ++row) {
                            if (forceValidation) reference.insert(reference.end(), logits[row].begin(), logits[row].end());
                            else for (float value : logits[row])
                                if (value != reference.at(offset++))
                                    throw std::runtime_error("paged growth changed logits");
                            ++past[row];
                            for (int layer = 0; layer < block_cnt; ++layer)
                            for (Data *root : {&cache[row][layer].first, &cache[row][layer].second})
                            for (auto &entry : root->multiDeviceDatas) {
                                const auto &local = *entry.second;
                                if (local.isPagedKVCache &&
                                    (local.dims[1] != past[row] || local.lastPageLen != (past[row] - 1) % local.pageLen + 1))
                                    throw std::runtime_error("paged growth metadata mismatch");
                            }
                        }
                    }
                }
            } catch (...) { error = std::current_exception(); }
        });
        ready.store(true);
        mainLoop->join(); delete mainLoop; mainLoop = nullptr;
        if (error) std::rethrow_exception(error);
        if (growthSteps < 8 || verifyGraphCaptures != captures)
            throw std::runtime_error("paged growth coverage or graph capture changed");
        std::cout << "SERVING PAGE GROWTH PASS exact_logits=" << reference.size()
                  << " growth_steps=" << growthSteps << " serving_captures=0" << std::endl;
    }
    std::vector<float> IndexerStorageRun(bool sharded, bool graph) {
        SetIndexerTPStorage(sharded);
        maxBatch = 4; SetMaxTokens(65536); SetChunkedPrefillSize(4096); WarmUp();
        struct AsyncCollectives {
            bool previous = FastllmCudaGetNcclForceSync();
            AsyncCollectives() { FastllmCudaSetNcclForceSync(false); }
            ~AsyncCollectives() { FastllmCudaSetNcclForceSync(previous); }
        } asyncCollectives;
        const int captures = verifyGraphCaptures, launches = verifyGraphLaunches;
        std::vector<float> result;
        GenerationConfig config;
        config.top_k = 1; config.repeat_penalty = 1; config.output_logits = true;
        config.output_token_limit = 160;
        std::vector<std::vector<std::pair<Data, Data>>> cache(4);
        std::vector<int> past(4);
        for (int row = 0; row < 4; ++row) {
            for (int layer = 0; layer < block_cnt; ++layer)
                cache[row].emplace_back(Data(kvCacheDataType), Data(kvCacheDataType));
            // Chunk boundaries, the 2048 Indexer transition, and nonaligned
            // page ends all appear before the mixed-length decode batch.
            for (int length : {127, 129, 1791 + row * 19}) {
                std::vector<float> ids(length), positions(length), logits;
                for (int i = 0; i < length; ++i) {
                    positions[i] = past[row] + i;
                    ids[i] = ((past[row] + i) * 3 + row) % 251;
                }
                Forward(Data(FLOAT32, {1, length}, ids), Data(), Data(FLOAT32, {1, length}, positions),
                        cache[row], config, LastTokensManager(), &logits);
                result.insert(result.end(), logits.begin(), logits.end());
                past[row] += length;
            }
        }
        for (int step = 0; step < 136; ++step) {
            const int count = step % 4 + 1;
            std::vector<float> ids(count);
            std::vector<Data> positions(count);
            std::vector<Data *> pos;
            std::vector<std::pair<Data *, Data *>> kv;
            std::vector<std::vector<float>> logits(count);
            std::vector<std::vector<float> *> output;
            std::vector<GenerationConfig> configs(count, config);
            std::vector<int> lengths(count, 1), requests;
            for (int row = 0; row < count; ++row) {
                const int request = (step + row) % 4;
                requests.push_back(request);
                ids[row] = (past[request] * 3 + request) % 251;
                positions[row].CopyFrom(Data(FLOAT32, {1, 1}, {(float)past[request]}));
                pos.push_back(&positions[row]); output.push_back(&logits[row]);
                for (auto &layer : cache[request]) kv.emplace_back(&layer.first, &layer.second);
            }
            ForwardBatch(count, Data(FLOAT32, {1, count}, ids), {}, pos, lengths, kv,
                         configs, LastTokensManager(), &output);
            for (int row = 0; row < count; ++row) {
                result.insert(result.end(), logits[row].begin(), logits[row].end());
                ++past[requests[row]];
            }
        }
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto &key = cache[0][layer].first;
            const int owner = layer % ranks;
            const int ownerWidth = key.multiDeviceDatas.at(owner)->dims[2];
            const int tail = kvCacheDataType == FP8_E4M3 ? 144 : 72;
            const int indexWidth = key.multiDeviceDatas.at(owner)->kvCacheFp8Indexer ? tail : 128;
            for (int rank = 0; rank < ranks; ++rank) {
                const auto &local = *key.multiDeviceDatas.at(rank);
                const int expected = ownerWidth - (sharded && rank != owner ? indexWidth : 0);
                if (local.dims[2] != expected || !local.isPagedKVCache ||
                    local.pagedKVCacheData->dims.back() != expected)
                    throw std::runtime_error("Indexer shard cache width is incorrect");
            }
            const auto &weights = weight["model.layers." + std::to_string(layer) +
                ".self_attn.indexer.wq.weight"].multiDeviceDatas;
            if (weights.size() != (sharded ? 1 : ranks) || !weights.count(owner))
                throw std::runtime_error("Indexer weight copies were not reduced");
        }
        if (verifyGraphCaptures != captures || (graph && verifyGraphLaunches == launches))
            throw std::runtime_error("Indexer storage graph replay/capture mismatch");
        std::cout << "INDEXER STORAGE RUN shard=" << sharded << " graph=" << graph
                  << " logits=" << result.size() << " launches=" << verifyGraphLaunches - launches << std::endl;
        return result;
    }
    void RingGraphToggleChecks(bool initialGraph) {
        int checks = 0, graphLaunches = 0;
        const std::vector<bool> modes = {true, true, false, false, true, true, false, true};
        GenerationConfig config;
        config.top_k = 1; config.output_logits = true; config.output_token_limit = 64;
        for (int prefix : {126, 255, 707}) {
            std::vector<std::pair<Data, Data>> reference(2), candidate(2);
            config.input_token_length = prefix;
            auto forward = [&](std::vector<std::pair<Data, Data>> &kv, int past, int rows) {
                std::vector<float> ids(rows), positions(rows), logits;
                for (int i = 0; i < rows; ++i) { ids[i] = (past + i) * 3 % 251; positions[i] = past + i; }
                Data input(FLOAT32, {1, rows}, ids), pos(FLOAT32, {1, rows}, positions);
                Forward(input, Data(), pos, kv, config, LastTokensManager(), &logits);
                return logits;
            };
            // The first prefill fixes the layout. Subsequent changes to graph
            // execution must preserve it across wraps and eager fallback.
            SetCudaGraph(initialGraph);
            forward(reference, 0, prefix); forward(candidate, 0, prefix);
            std::vector<std::vector<float>> expected;
            for (int step = 0; step < (int)modes.size(); ++step)
                expected.push_back(forward(reference, prefix + step, 1));
            int past = prefix;
            for (int step = 0; step < (int)modes.size(); ++step) {
                const bool graph = modes[step];
                SetCudaGraph(graph);
                const int launches = verifyGraphLaunches;
                const auto actual = forward(candidate, past, 1);
                if (actual.size() != expected[step].size()) throw std::runtime_error("ring toggle logits missing");
                RequireCloseLogits(actual.data(), expected[step], "ring graph toggle logits");
                if (!graph && verifyGraphLaunches != launches)
                    throw std::runtime_error("ring toggle launched a disabled graph");
                graphLaunches += verifyGraphLaunches - launches;
                ++past; ++checks;
            }
            for (int layer = 0; layer < 2; ++layer) for (bool value : {false, true}) {
                const auto &a = value ? candidate[layer].second : candidate[layer].first;
                const auto &e = value ? reference[layer].second : reference[layer].first;
                for (int device = 0; device < ranks; ++device) {
                    FastllmCudaSetDevice(device);
                    const Data &actual = *a.multiDeviceDatas.at(device), &expected = *e.multiDeviceDatas.at(device);
                    if (actual.dims != expected.dims) throw std::runtime_error("ring toggle KV shape mismatch");
                    const int rows = actual.dims[1], columns = actual.dims[2];
                    const bool ring = initialGraph && layer == 1;
                    const int storage = ring ? 128 : rows;
                    std::vector<uint16_t> av(storage * columns), ev(storage * columns);
                    FastllmCudaCopyFromDeviceToHost(av.data(), actual.cudaData, av.size() * sizeof(uint16_t));
                    FastllmCudaCopyFromDeviceToHost(ev.data(), expected.cudaData, ev.size() * sizeof(uint16_t));
                    std::vector<float> af(rows * columns), ef(rows * columns);
                    for (int row = 0; row < rows; ++row) for (int col = 0; col < columns; ++col) {
                        int offset = (ring ? ((past - rows + row) & 127) : row) * columns + col;
                        uint32_t x = uint32_t(av[offset]) << 16, y = uint32_t(ev[offset]) << 16;
                        std::memcpy(&af[row * columns + col], &x, 4);
                        std::memcpy(&ef[row * columns + col], &y, 4);
                    }
                    RequireCloseLogits(af.data(), ef, "ring graph toggle KV contents");
                    ++checks;
                }
            }
        }
        if (!graphLaunches) throw std::runtime_error("ring toggle did not replay a graph");
        std::cout << "SWA RING TOGGLE PASS initial_graph=" << initialGraph << " checks=" << checks
                  << " relative=" << largestLogitRelative << std::endl;
    }
#endif
    void FeatureGraphChecks() {
        draftTargetLayers = {0, 1};
        std::vector<std::vector<float>> expectedLogits;
        std::vector<std::map<int, std::vector<char>>> expectedHidden;
        for (bool graph : {false, true}) {
            SetCudaGraph(graph);
            std::vector<std::pair<Data, Data>> kv(2);
            GenerationConfig cfg;
            cfg.input_token_length = 12;
            cfg.output_token_limit = 64;
            int past = 0;
            for (int step = 0; step < 7; ++step) {
                int rows = step ? 1 : 12;
                std::vector<float> ids(rows), positions(rows);
                for (int i = 0; i < rows; ++i) {
                    ids[i] = (past + i) * 3 % 251;
                    positions[i] = past + i;
                }
                Data input(FLOAT32, {1, rows}, ids), pos(FLOAT32, {1, rows}, positions);
                TargetCapture capture;
    #ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                int before = verifyGraphLaunches;
    #endif
                Data logits = RunDraftTarget(input, pos, kv, cfg, capture);
                const float *values = (const float *)logits.cpuData;
                std::map<int, std::vector<char>> hidden;
                for (auto &item : capture.hidden) {
                    item.second.ToDevice(DataDevice::CPU);
                    auto *data = (char *)item.second.cpuData;
                    hidden[item.first] = std::vector<char>(data, data + item.second.GetBytes());
                }
                if (capture.hidden.size() != 2) throw std::runtime_error("single-token draft features missing");
                if (!graph) {
                    expectedLogits.emplace_back(values, values + logits.Count(0));
                    expectedHidden.push_back(std::move(hidden));
                } else {
                    RequireCloseLogits(values, expectedLogits[step], "single-token feature graph logits");
                    if (hidden != expectedHidden[step])
                        throw std::runtime_error("single-token feature graph changed hidden states");
    #ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                    if (step >= 3 && verifyGraphLaunches - before != ranks)
                        throw std::runtime_error("single-token feature graph did not replay");
    #endif
                }
                past += rows;
            }
        }
        std::cout << "FEATURE GRAPH PASS ranks=" << ranks << std::endl;
    }
    void BatchChecks(bool graph, bool merged) {
        if (!canDoBatchForward) throw std::runtime_error("ordinary TP did not advertise batching");
        if (merged) for (int layer = 0; layer < 2; ++layer) {
            std::string base = "model.layers." + std::to_string(layer) + ".self_attn.";
            int rows = 0;
            for (auto name : {"q_proj.weight", "k_proj.weight", "v_proj.weight"})
                rows += weight[base + name].dims[0];
            weight.AddEmptyWeight(base + "mergeqkv.weight", {rows, 256}, BFLOAT16);
            auto &dst = weight[base + "mergeqkv.weight"];
            dst.Allocate(false);
            size_t offset = 0;
            for (auto name : {"q_proj.weight", "k_proj.weight", "v_proj.weight"}) {
                auto &src = weight[base + name];
                std::memcpy(dst.cpuData + offset, src.cpuData, src.GetBytes());
                offset += src.GetBytes();
                weight.weight.erase(base + name);
            }
        }
        int checks = 0;
        // Default direct-call descriptors use FLOAT32; serving creates BF16
        // roots. Cover both the general metadata path and the replay fast path.
        for (DataType rootType : {FLOAT32, BFLOAT16})
        for (auto prompts : {std::pair<int,int>{3, 11}, {126, 250}, {254, 510}, {2045, 2049}, {4090, 13}}) {
            std::vector<std::vector<int>> schedule(12, {0, 1});
            schedule[4] = {1, 0}; schedule[5] = {1, 0};
            schedule[8] = {0}; // One request leaves the active decode batch.
            int counts[2] = {0, 0};
            for (const auto &order : schedule) for (int sequence : order) ++counts[sequence];
            std::vector<std::pair<Data,Data>> reference[2], candidate[2];
            std::vector<std::vector<float>> expected[2];
            std::vector<int> expectedTokens[2];
            int lastToken = 0;
            const int prompt[2] = {prompts.first, prompts.second};
            GenerationConfig cfg[2];
            auto forward = [&](std::vector<std::pair<Data,Data>> &kv, int sequence, int start, int rows) {
                std::vector<float> ids(rows), positions(rows);
                for (int j=0;j<rows;++j) { ids[j]=(start+j)*3%251+sequence; positions[j]=start+j; }
                Data input(FLOAT32,{1,rows},ids), pos(FLOAT32,{1,rows},positions);
                std::vector<float> logits;
                lastToken = Forward(input,Data(),pos,kv,cfg[sequence],LastTokensManager(),&logits);
                return logits;
            };
            for (int b=0;b<2;++b) {
                cfg[b].input_token_length=prompt[b]; cfg[b].output_token_limit=64;
                cfg[b].output_logits=true; cfg[b].top_k=1;
                reference[b].resize(2);
                for (int layer = 0; layer < 2; ++layer)
                    candidate[b].emplace_back(Data(rootType), Data(rootType));
                SetCudaGraph(false);
                forward(reference[b],b,0,prompt[b]);
                forward(candidate[b],b,0,prompt[b]);
                for (int step=0;step<counts[b];++step) {
                    expected[b].push_back(forward(reference[b],b,prompt[b]+step,1));
                    expectedTokens[b].push_back(lastToken);
                }
            }
            int steps[2]={0,0};
            for (int round=0;round<(int)schedule.size();++round) {
                const auto &order=schedule[round];
                SetCudaGraph(graph);
                if (order.size()==1) {
                    int b=order[0];auto actual=forward(candidate[b],b,prompt[b]+steps[b],1);
                    RequireCloseLogits(actual.data(),expected[b][steps[b]++],"batch survivor logits");
                    ++checks;continue;
                }
                std::vector<std::pair<Data*,Data*>> caches;
                std::vector<Data> positions(order.size());
                std::vector<Data*> pos;
                std::vector<float> ids;
                std::vector<GenerationConfig> configs;
                std::vector<std::vector<float>> actual(order.size());
                std::vector<std::vector<float>*> out;
                for (int i=0;i<(int)order.size();++i) {
                    int b=order[i],past=prompt[b]+steps[b];
                    ids.push_back(past*3%251+b);
                    positions[i].dataType=FLOAT32;positions[i].Resize({1,1});positions[i].Allocate();
                    ((float*)positions[i].cpuData)[0]=past;pos.push_back(&positions[i]);
                    for (auto &layer:candidate[b]) caches.emplace_back(&layer.first,&layer.second);
                    configs.push_back(cfg[b]);out.push_back(&actual[i]);
                    // Exercise compact GPU selection as well as full-logit sampling.
                    configs.back().output_logits = round != 3 && round != 6 && round != 10;
                }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                int launches=verifyGraphLaunches, captures=verifyGraphCaptures;
#endif
                Data input(FLOAT32,{1,(int)order.size()},ids);
                auto tokens = ForwardBatch(order.size(),input,{},pos,std::vector<int>(order.size(),1),caches,configs,LastTokensManager(),&out);
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                if (graph && !failVerifyBegin && !failVerifyInstantiate && prompts.first==3 && round==2 &&
                    (verifyGraphLaunches-launches!=ranks || verifyGraphCaptures!=captures))
                    throw std::runtime_error("batch did not replay exactly one graph per rank");
#endif
                for (int i=0;i<(int)order.size();++i) {
                    int b=order[i];
                    const auto &logits = expected[b][steps[b]++];
                    if (configs[i].output_logits) {
                        if (actual[i].size() != logits.size()) throw std::runtime_error("batch logits missing");
                        RequireCloseLogits(actual[i].data(), logits, "packed batch logits");
                    } else if (tokens[i] != expectedTokens[b][steps[b] - 1]) {
                        throw std::runtime_error("compact batch greedy token mismatch");
                    }
                    if (candidate[b][0].first.dims[1]!=prompt[b]+steps[b] ||
                        candidate[b][1].first.dims[1]!=std::min(prompt[b]+steps[b],7))
                        throw std::runtime_error("batch KV lengths crossed requests");
                    ++checks;
                }
            }
            for (int b=0;b<2;++b) for (int layer=0;layer<2;++layer) for (bool value:{false,true}) {
                auto &a=value?candidate[b][layer].second:candidate[b][layer].first;
                auto &e=value?reference[b][layer].second:reference[b][layer].first;
                for (const auto &entry:a.multiDeviceDatas) {
                    FastllmCudaSetDevice(entry.first);
                    const Data &actual=*entry.second,&expected=*e.multiDeviceDatas.at(entry.first);
                    if (actual.dims!=expected.dims) throw std::runtime_error("batch KV shape mismatch");
                    std::vector<uint16_t> av(actual.Count(0)),ev(expected.Count(0));
                    FastllmCudaCopyFromDeviceToHost(av.data(),actual.cudaData,actual.GetBytes());
                    FastllmCudaCopyFromDeviceToHost(ev.data(),expected.cudaData,expected.GetBytes());
                    std::vector<float> af(av.size()),ef(ev.size());
                    for(size_t j=0;j<av.size();++j){uint32_t x=uint32_t(av[j])<<16,y=uint32_t(ev[j])<<16;std::memcpy(&af[j],&x,4);std::memcpy(&ef[j],&y,4);}
                    RequireCloseLogits(af.data(),ef,"independent batch KV contents");++checks;
                }
            }
        }
        std::cout<<"BATCH PASS ranks="<<ranks<<" graph="<<graph<<" merged="<<merged
                 <<" checks="<<checks<<" relative="<<largestLogitRelative<<std::endl;
    }

    void PackedPrefillChecks(bool longPrefill = false) {
        GenerationConfig config;
        config.output_logits = true;
        config.output_token_limit = 16;
        config.top_k = 1;
        std::vector<int> lengths = longPrefill ? std::vector<int>{4096, 3983, 4211} : std::vector<int>{3, 7, 1};
        std::vector<std::vector<std::pair<Data, Data>>> reference(3), candidate(3);
        std::vector<Data> positions(3);
        std::vector<Data *> positionPtrs;
        std::vector<std::pair<Data *, Data *>> kv;
        std::vector<float> ids;
        std::vector<std::vector<float>> expected(3), actual(3);
        std::vector<std::vector<float> *> outputs;
        SetCudaGraph(false);
        for (int b = 0; b < 3; ++b) {
            reference[b].resize(2); candidate[b].resize(2);
            std::vector<float> tokens(lengths[b]), pos(lengths[b]);
            for (int j = 0; j < lengths[b]; ++j) { tokens[j] = (5 * j + b) % weight["model.embed_tokens.weight"].dims[0]; pos[j] = j; }
            Data input(FLOAT32, {1, lengths[b]}, tokens), position(FLOAT32, {1, lengths[b]}, pos);
            Forward(input, Data(), position, reference[b], config, LastTokensManager(), &expected[b]);
            positions[b].CopyFrom(position); positionPtrs.push_back(&positions[b]);
            ids.insert(ids.end(), tokens.begin(), tokens.end());
            for (auto &layer : candidate[b]) kv.emplace_back(&layer.first, &layer.second);
            outputs.push_back(&actual[b]);
        }
        Data input(FLOAT32, {1, (int)ids.size()}, ids);
        std::vector<GenerationConfig> configs(3, config);
        ForwardBatch(3, input, {}, positionPtrs, lengths, kv, configs, LastTokensManager(), &outputs);
        for (int b = 0; b < 3; ++b) {
            if (actual[b].size() != expected[b].size()) throw std::runtime_error("prefill logits missing");
            RequireCloseLogits(actual[b].data(), expected[b], "packed prefill logits");
        }
        // Mixed lengths become a three-request decode batch with independent KV.
        for (int b = 0; b < 3; ++b) {
            Data token(FLOAT32, {1, 1}, {float(50 + b)}), pos(FLOAT32, {1, 1}, {float(lengths[b])});
            Forward(token, Data(), pos, reference[b], config, LastTokensManager(), &expected[b]);
            positions[b].CopyFrom(pos);
        }
        Data decode(FLOAT32, {1, 3}, {50, 51, 52});
        for (bool graph : {false, true}) {
            SetCudaGraph(graph);
            ForwardBatch(3, decode, {}, positionPtrs, {1, 1, 1}, kv, configs, LastTokensManager(), &outputs);
            for (int b = 0; b < 3; ++b) {
                RequireCloseLogits(actual[b].data(), expected[b], "prefill to batch decode logits");
                if (graph) continue;
                Data token(FLOAT32, {1, 1}, {float(50 + b)}), pos(FLOAT32, {1, 1}, {float(lengths[b] + 1)});
                Forward(token, Data(), pos, reference[b], config, LastTokensManager(), &expected[b]);
                positions[b].CopyFrom(pos);
            }
        }
        std::cout << "PACKED PREFILL PASS ranks=" << ranks << " relative=" << largestLogitRelative << std::endl;
    }

#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    void CachePoolChurnChecks() {
        maxBatch = 160; max_positions = 200000;
        SetMaxTokens(524288); SetChunkedPrefillSize(128); WarmUp();
        FastllmCudaSetDevice(0);
        const int before = directAllocations;
        const int freePages = GetPagedKVCacheManager(0, true)->FreePageCount();
        std::vector<std::vector<Data>> requests(maxBatch);
        std::vector<int> order(maxBatch), tags(maxBatch);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(20261017);
        auto allocate = [&](int slot, bool hit, int pass) {
            if (requests[slot].empty()) requests[slot].resize(4);
            const bool swa = hit ? pass == 0 : pass == 1;
            for (int part = 0; part < 2; ++part) {
                Data &cache = requests[slot][(swa ? 2 : 0) + part];
                cache.dataType = kvCacheDataType; cache.UpdateUnitSize();
                cache.dataDevice = DataDevice::CUDA; cache.dataDeviceIds = {0};
                if (swa) {
                    ReserveCache(cache, 128, std::max(1, 8 / ranks) * (part ? 128 : 192));
                    uint64_t value = tags[slot];
                    FastllmCudaCopyFromHostToDevice(cache.cudaData, &value, sizeof(value));
                } else {
                    PreparePagedCache(cache, *GetPagedKVCacheManager(0, !part), 1);
                }
            }
        };
        for (int slot : order) for (int pass = 0; pass < 2; ++pass) allocate(slot, false, pass);
        for (int turn = 0; turn < 2000; ++turn) {
            std::shuffle(order.begin(), order.end(), random);
            int count = 1 + random() % 16;
            bool hit = random() % 5 != 0;
            for (int i = 0; i < count; ++i) {
                requests[order[i]].clear(); tags[order[i]] = turn + 1;
            }
            // Batch restores reserve every SWA window before page tables;
            // cold prefills reserve page tables first. Mix both orders.
            for (int pass = 0; pass < 2; ++pass)
                for (int i = 0; i < count; ++i) allocate(order[i], hit, pass);
            if (turn % 100 == 0) for (int slot = 0; slot < maxBatch; ++slot) {
                for (int part = 2; part < 4; ++part) {
                    uint64_t value;
                    FastllmCudaCopyFromDeviceToHost(&value, requests[slot][part].cudaData, sizeof(value));
                    if (value != (uint64_t)tags[slot]) throw std::runtime_error("request KV slots overlap");
                }
            }
        }
        requests.clear();
        if (directAllocations != before || GetPagedKVCacheManager(0, true)->FreePageCount() != freePages)
            throw std::runtime_error("request churn allocated device memory or leaked KV pages");
        std::cout << "KV CHURN PASS ranks=" << ranks << " batch=160 turns=2000" << std::endl;
    }

    void SwaPrefillWorkspaceChecks() {
        // Compare against the original contiguous-cache path before WarmUp
        // installs the paged pool. Exercise short windows, wraps, multi-chunk
        // continuations and direct calls larger than the configured chunk.
        const std::vector<int> lengths = {3, 125, 1, 127, 4096, 17, 1};
        std::vector<std::vector<float>> expected;
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) {
                SetCudaGraph(true);
                maxBatch = 4; SetMaxTokens(32768); SetChunkedPrefillSize(1024);
                WarmUp();
            }
            std::vector<std::pair<Data, Data>> kv(2);
            std::vector<void *> pointers;
            int past = 0;
            for (int step = 0; step < (int)lengths.size(); ++step) {
                int length = lengths[step];
                SetCudaGraph(step != 2); // Eager decode followed by ring prefill.
                std::vector<float> ids(length), positions(length), logits;
                for (int i = 0; i < length; ++i) {
                    ids[i] = (past + i) * 3 % 251; positions[i] = past + i;
                }
                Data input(FLOAT32, {1, length}, ids), pos(FLOAT32, {1, length}, positions);
                GenerationConfig config;
                config.top_k = 1; config.output_logits = true;
                config.input_token_length = 6000; config.output_token_limit = 16;
                Forward(input, Data(), pos, kv, config, LastTokensManager(), &logits);
                if (!pass) expected.push_back(logits);
                else {
                    RequireCloseLogits(logits.data(), expected[step], "shared SWA prefill logits");
                    std::vector<void *> current;
                    for (Data *root : {&kv[1].first, &kv[1].second}) {
                        if (root->expansionDims[1] != 128)
                            throw std::runtime_error("SWA reservation depends on prefill length");
                        for (int device = 0; device < ranks; ++device) {
                            const Data &cache = *root->multiDeviceDatas.at(device);
                            if (cache.expansionDims[1] != 128 || !IsCudaWorkspacePointer(cache.cudaData))
                                throw std::runtime_error("SWA request cache escaped fixed pool window");
                            current.push_back(cache.cudaData);
                        }
                    }
                    if (step && current != pointers)
                        throw std::runtime_error("SWA prefill changed decode cache addresses");
                    pointers = std::move(current);
                }
                past += length;
            }
        }
        std::cout << "SWA WORKSPACE PASS ranks=" << ranks
                  << " relative=" << largestLogitRelative << std::endl;
    }

    void CachePoolChecks() {
        maxBatch = 4;
        SetMaxTokens(8192);
        WarmUp();
        FastllmCudaSetDevice(0);
        int before = directAllocations;
        {
            auto *pool = GetPagedKVCacheManager(0, true);
            Data owner(BFLOAT16), fork(BFLOAT16);
            PreparePagedCache(owner, *pool, 127);
            const int width = pool->dims[3];
            owner.Resize({1, 127, width}); owner.lastPageLen = 127;
            std::vector<uint16_t> pattern(127 * width, 0x3f80), copied(pattern.size());
            auto address = [&](const Data &cache) {
                return (uint8_t *)pool->cudaData + (size_t)cache.pageIndex[0] * pool->pageLen * width * 2;
            };
            FastllmCudaCopyFromHostToDevice(address(owner), pattern.data(), pattern.size() * 2);
            fork.dataDevice = DataDevice::CUDA; fork.dataDeviceIds = {0};
            fork.Resize(owner.dims); fork.pageLen = owner.pageLen;
            fork.isPagedKVCache = true; fork.pagedKVCacheData = pool;
            fork.pageIndex = owner.pageIndex; fork.lastPageLen = 127;
            pool->Pick(fork.pageIndex);
            PreparePagedCache(fork, *pool, 128);
            if (fork.pageIndex[0] == owner.pageIndex[0]) throw std::runtime_error("partial shared page was overwritten");
            FastllmCudaCopyFromDeviceToHost(copied.data(), address(fork), copied.size() * 2);
            if (copied != pattern) throw std::runtime_error("partial page copy lost prior KV");
            FastllmCudaCopyFromDeviceToHost(copied.data(), address(owner), copied.size() * 2);
            if (copied != pattern) throw std::runtime_error("fork mutated its shared ancestor");
            void *table = owner.cudaData;
            owner.dims[1] = 128; owner.lastPageLen = 128;
            PreparePagedCache(owner, *pool, 129);
            if (owner.cudaData != table || owner.pageIndex.size() != 2 || directAllocations != before)
                throw std::runtime_error("paged growth reallocated device storage");
        }
        Data cache(BFLOAT16);
        cache.dataDevice = DataDevice::CUDA;
        cache.dataDeviceIds = {0};
        cache.isKVCache = true;
        ReserveCache(cache, 128, 16);
        if (!IsCudaWorkspacePointer(cache.cudaData)) throw std::runtime_error("KV escaped startup pool");
        cache.Resize({1, 127, 16});
        std::vector<uint16_t> input(127 * 16), output(input.size());
        for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i);
        FastllmCudaCopyFromHostToDevice(cache.cudaData, input.data(), input.size() * 2);
        ReserveCache(cache, 512, 16);
        FastllmCudaCopyFromDeviceToHost(output.data(), cache.cudaData, output.size() * 2);
        if (input != output || cache.dims != std::vector<int>({1, 127, 16}))
            throw std::runtime_error("pool growth corrupted KV or logical length");
        void *pointer = cache.cudaData;
        bool exhausted = false;
        try { ReserveCache(cache, 1 << 24, 16); }
        catch (const CudaWorkspaceExhausted &) { exhausted = true; }
        if (!exhausted || cache.cudaData != pointer)
            throw std::runtime_error("pool exhaustion did not preserve existing KV");
        cache.FreeSpace();
        if (IsCudaWorkspacePointer(pointer)) throw std::runtime_error("KV lease was not released");
        cache.dims.clear();
        cache.expansionDims.clear();
        ReserveCache(cache, 640, 16);
        if (!IsCudaWorkspacePointer(cache.cudaData) || directAllocations != before)
            throw std::runtime_error("KV growth/reuse allocated CUDA memory after startup");
        // Exhaust only rank 0. Other ranks must not enter collectives while
        // this rank rejects the allocation, and the next request must recover.
        std::vector<std::unique_ptr<Data>> pressure;
        try {
            for (;;) {
                auto block = std::make_unique<Data>(BFLOAT16);
                block->dataDevice = DataDevice::CUDA;
                block->dataDeviceIds = {0};
                ReserveCache(*block, 128, std::max(1, 8 / ranks) * 32);
                pressure.push_back(std::move(block));
            }
        } catch (const CudaWorkspaceExhausted &) {}
        eos_token_id = -1;
        useRealScheduler = true;
        GenerationConfig config;
        config.top_k = 1;
        config.output_token_limit = 4;
        int failed = LaunchResponseTokens({1, 2, 3}, config);
        if (FetchResponseTokens(failed) != -2)
            throw std::runtime_error("scheduler lost KV capacity error");
        pressure.clear();
        int handle = LaunchResponseTokens({1, 2, 3}, config), count = 0, token;
        while ((token = FetchResponseTokens(handle)) >= 0) ++count;
        if (token != -1 || count != 4)
            throw std::runtime_error("scheduler failed after KV exhaustion");
        failDecodeAllocation = true;
        failed = LaunchResponseTokens({1, 2, 3}, config);
        if (FetchResponseTokens(failed) < 0 || FetchResponseTokens(failed) != -2 || failDecodeAllocation)
            throw std::runtime_error("decode allocation error was not isolated");
        handle = LaunchResponseTokens({1, 2, 3}, config); count = 0;
        while ((token = FetchResponseTokens(handle)) >= 0) ++count;
        if (token != -1 || count != 4)
            throw std::runtime_error("decode error freed reusable position metadata");
        std::cout << "KV POOL PASS ranks=" << ranks << std::endl;
    }
#endif

    std::vector<std::vector<float>> Run() {
        if (RetainCudaWorkspace() != (ranks > 1))
            throw std::runtime_error("unexpected serial/TP workspace policy");
        WarmUp();
        if (elementsInKVCachePerToken != std::max(4, ranks) * 48 + ranks * 128)
            throw std::runtime_error("physical KV cache accounting mismatch");
        std::vector<std::pair<Data, Data>> kv;
        for (int i = 0; i < 2; ++i)
            kv.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
        std::vector<std::vector<float>> out;
        int past = 0;
        for (int n : {3, 17, 1, 1}) {
            std::vector<float> ids(n), positions(n);
            for (int i = 0; i < n; ++i) {
                ids[i] = (past + i) * 3 % 256;
                positions[i] = past + i;
            }
            Data input(FLOAT32, {1, n}, ids), pos(FLOAT32, {1, n}, positions);
            GenerationConfig cfg;
            cfg.output_logits = true;
            std::vector<float> logits;
            Forward(input, Data(), pos, kv, cfg, LastTokensManager(), &logits);
            out.push_back(logits);
            past += n;
            if (kv[0].first.dims[1] != past || kv[1].first.dims[1] != std::min(past, 7))
                throw std::runtime_error("cache metadata mismatch");
        }
        return out;
    }

    void VerifyBatches() {
        draftTargetLayers = {0, 1};
        GenerationConfig cfg;
        for (std::vector<int> past : {std::vector<int>{257, 2307, 43, 511, 2043, 4089, 31, 3007},
                                     std::vector<int>{2045, 4093, 127, 2047, 1023, 63, 41, 3017},
                                     std::vector<int>{2307, 2312, 2317, 2322, 2327, 2332, 2337, 2342}}) {
        using Cache = std::vector<std::pair<Data, Data>>;
        std::vector<Cache> reference(past.size()), actual(past.size());
        auto input = [](int sequence, int start, int rows, bool positions) {
            std::vector<float> values(rows);
            for (int i = 0; i < rows; ++i)
                values[i] = positions ? start + i : ((start + i) * 13 + sequence * 71) % 256;
            return Data(FLOAT32, {1, rows}, values);
        };
        for (int b = 0; b < (int)past.size(); ++b) for (Cache *cache : {&reference[b], &actual[b]}) {
            cache->resize(block_cnt);
            TargetCapture capture;
            RunDraftTarget(input(b, 0, past[b], false), input(b, 0, past[b], true), *cache, cfg, capture);
        }
        const bool rebind = past.front() == 2307;
        const bool boundary = past.front() == 2045;
        int checks = 0;
        for (auto lengths : {std::vector<int>{6, 6}, {6, 6}, {6, 6}, {6, 3, 1}, {6, 3, 1}, {2, 6, 4}, {2, 6, 4}, {6, 6, 6, 6, 6, 6}, {6, 6, 6, 6, 6, 6}, {6, 6, 6, 6, 6, 6, 6, 6}, {6, 6, 6, 6, 6, 6, 6, 6}, {8, 7, 6, 5, 4, 3, 2, 1}, {8, 7, 6, 5, 4, 3, 2, 1}, {6, 6}}) {
            if (rebind && checks) {
                std::rotate(actual.begin(), actual.begin() + 1, actual.end());
                std::rotate(reference.begin(), reference.begin() + 1, reference.end());
                std::rotate(past.begin(), past.begin() + 1, past.end());
            }
            const int batch = lengths.size();
            std::vector<float> ids, positions;
            std::vector<std::vector<float>> expected(batch);
            std::vector<bool> referenceRing(batch);
            std::vector<std::pair<Data *, Data *>> caches;
            std::vector<GenerationConfig> configs(batch, cfg);
            for (int b = 0; b < batch; ++b) {
                Data id = input(b, past[b], lengths[b], false), pos = input(b, past[b], lengths[b], true);
                ids.insert(ids.end(), (float *)id.cpuData, (float *)id.cpuData + lengths[b]);
                positions.insert(positions.end(), (float *)pos.cpuData, (float *)pos.cpuData + lengths[b]);
                TargetCapture capture; capture.verifying = true;
                SetCudaGraph(false);
                Data logits = RunDraftTarget(id, pos, reference[b], cfg, capture);
                SetCudaGraph(true);
                referenceRing[b] = capture.ringVerify;
                logits.ToDevice(DataDevice::CPU);
                expected[b].assign((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
                for (auto &layer : actual[b]) caches.emplace_back(&layer.first, &layer.second);
            }
            TargetBatch requests{caches, configs, lengths, block_cnt};
            TargetCapture capture; capture.verifying = true;
            capture.borrowOutputs = (checks % 3) != 0;
            LogitsSelection selection; selection.cudaLogits = true; capture.selection = &selection;
            Cache unused;
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            const int launches = verifyGraphLaunches, captures = verifyGraphCaptures;
#endif
            Data logits = RunDraftTarget(Data(FLOAT32, {1, (int)ids.size()}, ids),
                Data(FLOAT32, {1, (int)positions.size()}, positions), unused, cfg, capture, &requests);
            #ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            if (checks == 4 && !failVerifyBegin && !failVerifyInstantiate &&
                verifyGraphLaunches - launches != ranks)
                throw std::runtime_error("multi-request verification did not replay one graph per rank");
            if (rebind && checks == 4 && !failVerifyBegin && !failVerifyInstantiate &&
                verifyGraphCaptures != captures)
                throw std::runtime_error("request replacement recaptured its verification graph");
#endif
            if (!boundary && checks == 0)
                NaiveN05GraphTestAccess::CheckRejectedVerificationReuse(*this,
                    Data(FLOAT32, {1, (int)ids.size()}, ids), requests);
            NaiveN05GraphTestAccess::CheckSharedVerificationWorkspace(*this, ranks);
            if (logits.dataDevice != DataDevice::CUDA || logits.Count(0) != ids.size() * 256)
                throw std::runtime_error("batch verification did not return all device rows");
            if (logits.isFake != capture.borrowOutputs)
                throw std::runtime_error("batch verification output ownership mismatch");
            // Borrowed CUDA views cannot migrate their backing allocation.
            // Materialize an owned host copy for the numerical oracle.
            Data hostLogits(logits);
            hostLogits.ToDevice(DataDevice::CPU);
            std::vector<Cache *> commitCaches;
            std::vector<int> commitPast, commitCounts;
            for (int b = 0; b < batch; ++b) {
                commitCaches.push_back(&actual[b]); commitPast.push_back(past[b]);
                commitCounts.push_back(std::min(lengths[b], 1 + (b + checks + b) % 5));
            }
            CommitTargetCaches(commitCaches, commitPast, commitCounts, capture.ringVerify);
            int offset = 0;
            for (int b = 0; b < batch; ++b) {
                RequireCloseLogits((float *)hostLogits.cpuData + offset * 256, expected[b], "batch verify logits b=" + std::to_string(b) + " past=" + std::to_string(past[b]) + " check=" + std::to_string(checks));
                offset += lengths[b];
                // Different rollback lengths, followed by another verify round,
                // exercise sequence isolation and the second request's SWA ring.
                const int keep = std::min(lengths[b], 1 + (b + checks) % 5);
                CommitTargetCache(reference[b], past[b], keep, referenceRing[b]);

                past[b] += keep;
                for (int layer = 0; layer < block_cnt; ++layer) {
                    if (actual[b][layer].first.dims != reference[b][layer].first.dims)
                        throw std::runtime_error("batch verify rollback metadata mismatch");
                    for (const auto &rank : actual[b][layer].first.multiDeviceDatas)
                        if (rank.second->dims != reference[b][layer].first.multiDeviceDatas.at(rank.first)->dims)
                            throw std::runtime_error("batch verify rank rollback metadata mismatch");
                }
                ++checks;
            }
        }
        std::cout << "VERIFY BATCH PASS checks=" << checks << " relative=" << largestLogitRelative
                  << " maximum=" << largestLogitAbsolute << std::endl;
        }
    }

    void VerifySelections(int ranks) {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int topk : {1, 4, 50, 256}) {
            GenerationConfig cfg; cfg.top_k = topk; cfg.temperature = .7f; cfg.top_p = .8f;
            std::vector<std::vector<float>> expectedLogits;
            std::vector<std::map<int, std::vector<char>>> expectedHidden;
            for (bool reference : {true, false}) {
                std::vector<std::pair<Data, Data>> kv(2);
                int past = 0, step = 0;
                for (int rows : {3, 8, 8, 8, 1, 3, 8, 8}) {
                    std::vector<float> ids(rows), pos(rows);
                    for (int row = 0; row < rows; ++row) { ids[row] = (past + row) * 3 % 256; pos[row] = past + row; }
                    Data input(FLOAT32, {1, rows}, ids), positions(FLOAT32, {1, rows}, pos);
                    TargetCapture capture; capture.verifying = past > 0;
                    auto selection = SelectLogits(cfg, true);
                    if (topk == 256) selection.cudaLogits = true;
                    if (!reference) capture.selection = &selection;
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                    int launches = verifyGraphLaunches;
#endif
                    Data logits = RunDraftTarget(input, positions, kv, cfg, capture);
                    if (reference) {
                        logits.ToDevice(DataDevice::CPU);
                        expectedLogits.emplace_back((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
                        expectedHidden.emplace_back();
                    } else {
                        if (topk == 256) {
                            if (logits.dataDevice != DataDevice::CUDA ||
                                logits.dataDeviceIds != std::vector<int>{0} ||
                                !selection.candidates.dims.empty() || !selection.sampledTokens.empty())
                                throw std::runtime_error("verify device logits missing or sampled early");
                            logits.ToDevice(DataDevice::CPU);
                            if (logits.Count(0) != expectedLogits[step].size() ||
                                std::memcmp(logits.cpuData, expectedLogits[step].data(), logits.GetBytes()))
                                throw std::runtime_error("verify device logits differ from CPU shards");
                        } else if (selection.candidates.dims.empty() || !logits.dims.empty())
                            throw std::runtime_error("verify compact output missing");
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                        if (step == 3 && verifyGraphLaunches - launches != ranks)
                            throw std::runtime_error("compact verify did not replay its graph");
#endif
                        int outputRows = past ? rows : 1;
                        if (topk == 1) {
                            Data full(FLOAT32, {1, outputRows, 256}, expectedLogits[step]);
                            ApplyDraftDevice(); Data top; TopK(full, top, 1); top.ToDevice(DataDevice::CPU);
                            for (int row = 0; row < outputRows; ++row)
                                if (((float *)top.cpuData)[row * 2] != ((float *)selection.candidates.cpuData)[row * 2])
                                    throw std::runtime_error("compact greedy verify token mismatch");
                        } else if (topk < 256) {
                            for (int row = 0; row < outputRows; ++row) {
                                auto p = SpeculativeDistribution(expectedLogits[step].data() + row * 256, 256, cfg, LastTokensUnit());
                                auto q = SpeculativeTopKDistribution((float *)selection.candidates.cpuData + row * topk * 2, topk, 256, cfg);
                                if (p != q) throw std::runtime_error("compact verify probabilities mismatch");
                            }
                        }
                        if (capture.hidden.size() != expectedHidden[step].size())
                            throw std::runtime_error("compact verify features missing");
                    }
                    for (auto &feature : capture.hidden) {
                        feature.second.ToDevice(DataDevice::CPU);
                        auto *data = (char *)feature.second.cpuData;
                        std::vector<char> bytes(data, data + feature.second.GetBytes());
                        if (reference) expectedHidden.back()[feature.first] = bytes;
                        else if (expectedHidden[step].at(feature.first) != bytes)
                            throw std::runtime_error("compact verify hidden state mismatch");
                    }
                    int keep = past && rows > 1 ? rows - 1 : rows;
                    CommitTargetCache(kv, past, keep, capture.ringVerify); past += keep; ++step;
                    if (!reference) ++checks;
                }
            }
        }
        std::cout << "VERIFY SELECTION PASS checks=" << checks << std::endl;
    }

    void VerifyHead() {
        Data referenceHead(weight["lm_head.weight"]);
        std::vector<std::pair<Data, Data>> kv(2);
        Data ids(FLOAT32, {1, 3}, {1, 2, 3}), pos(FLOAT32, {1, 3}, {0, 1, 2});
        GenerationConfig cfg;
        TargetCapture capture;
        RunDraftTarget(ids, pos, kv, cfg, capture);
        for (int rows : {1, 2, 7, 3, 8, 7, 5, 10, 15, 5, 380, 343, 350, 360}) for (bool deviceInput : {true, false}) {
            Data hidden(BFLOAT16, {1, rows, 256});
            hidden.Allocate();
            for (int i = 0; i < hidden.Count(0); ++i) {
                float x = std::sin((i + rows) * .1f);
                uint32_t bits;
                std::memcpy(&bits, &x, 4);
                ((uint16_t *)hidden.cpuData)[i] = bits >> 16;
            }
            if (deviceInput) hidden.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            Data result = RunDraftHead(hidden), expected;
            NaiveN05GraphTestAccess::CheckHeadReturn(*this, result, deviceInput);
            Data actual(result); // Materialize only the test oracle before moving it to CPU.
            hidden.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            ApplyDraftDevice();
            if (rows > 1)
                MatMulTransB(hidden, referenceHead, expected);
            else
                Linear(hidden, referenceHead, Data(), expected);
            actual.ToDevice(DataDevice::CPU);
            expected.ToDevice(DataDevice::CPU);
            if (actual.dataType != BFLOAT16 || expected.dataType != BFLOAT16 ||
                actual.dims != expected.dims)
                throw std::runtime_error("head output shape or dtype differs");
            int differing = 0;
            for (int i = 0; i < actual.Count(0); ++i)
                differing += ((uint16_t *)actual.cpuData)[i] != ((uint16_t *)expected.cpuData)[i];
            std::cout << "HEAD difference=" << differing << " / " << actual.Count(0) << std::endl;
            bool nonzero = false;
            for (int i = 0; i < expected.Count(0); ++i)
                nonzero |= (((uint16_t *)expected.cpuData)[i] & 0x7fff) != 0;
            if (!nonzero)
                throw std::runtime_error("head regression requires nonzero logits");
            if (differing)
                throw std::runtime_error("head differs");
        }
        std::cout << "HEAD PASS checks=28" << std::endl;
    }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    void VerifyGraphBlocks() {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int prompt : {3, 120, 254, 510, 2045, 2049, 4090})
        for (int block : {1, 2, 3, 4, 5, 6, 7, 8}) {
            const int previousLaunches = verifyGraphLaunches;
            std::vector<std::pair<Data, Data>> reference(2), candidate(2);
            GenerationConfig cfg;
            cfg.input_token_length = prompt;
            cfg.output_token_limit = 96;
            cfg.output_logits = true;
            bool candidateRing = false;
            auto forward = [&](auto &kv, int start, int count, bool verifying) {
                std::vector<float> ids(count), pos(count);
                for (int row = 0; row < count; ++row) {
                    ids[row] = (start + row) * 3 % 256;
                    pos[row] = start + row;
                }
                Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                TargetCapture capture;
                capture.verifying = verifying;
                Data out = RunDraftTarget(input, positions, kv, cfg, capture);
                if (&kv == &candidate) candidateRing = capture.ringVerify;
                out.ToDevice(DataDevice::CPU);
                if (out.dims != std::vector<int>({1, verifying ? count : 1, 256}) ||
                    capture.hidden.size() != 2)
                    throw std::runtime_error("missing graph logits/features");
                for (int layer = 0; layer < 2; ++layer)
                    if (capture.hidden.at(layer).Count(0) != count * 256)
                        throw std::runtime_error("graph feature shape mismatch");
                return std::vector<float>((float *)out.cpuData,
                                          (float *)out.cpuData + out.Count(0));
            };
            forward(reference, 0, prompt, false);
            forward(candidate, 0, prompt, false);
            int past = prompt;
            for (int requested : {1, 4, 8, 3, 7, 8}) {
                const int keep = std::min(requested, block);
                auto logits = forward(candidate, past, block, true);
                for (int row = 0; row < keep; ++row) {
                    auto one = forward(reference, past + row, 1, false);
                    RequireCloseLogits(logits.data() + row * 256, one,
                                       "graph verification differs from decode at " +
                                       std::to_string(past) + " row " + std::to_string(row));
                    ++checks;
                }
                CommitTargetCache(candidate, past, keep, candidateRing);
                for (int layer = 0; layer < 2; ++layer)
                    if (candidate[layer].first.dims != reference[layer].first.dims)
                        throw std::runtime_error("graph rollback metadata mismatch");
                past += keep;
            }
            if (prompt == 2049 && expectVerifyGraph && !failVerifyBegin &&
                !failVerifyInstantiate && verifyGraphLaunches == previousLaunches)
                throw std::runtime_error("verification block shape never replayed: " + std::to_string(block));
        }
        if (expectVerifyGraph &&
            (!verifyGraphCaptures ||
             ((!failVerifyBegin && !failVerifyInstantiate) != (verifyGraphLaunches > 0))))
            throw std::runtime_error("verification graph execution/fallback was not exercised");
        std::cout << "LOGIT ERROR relative_max=" << largestLogitRelative
                  << " absolute_max=" << largestLogitAbsolute << std::endl;
        std::cout << "VERIFY GRAPH PASS checks=" << checks << " captures=" << verifyGraphCaptures
                  << " launches=" << verifyGraphLaunches << std::endl;
    }

#endif
    void VerifyRollback() {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int past : {3, 7, 15}) for (int rows : {1, 7, 8})
            for (int keep : {0, 1, rows / 2, rows}) {
                std::vector<std::pair<Data, Data>> clean(2), poisoned(2);
                GenerationConfig cfg;
                cfg.input_token_length = past;
                cfg.output_token_limit = 32;
                cfg.output_logits = true;
                auto forward = [&](auto &kv, int start, int count, bool verifying, bool poison) {
                    std::vector<float> ids(count), pos(count);
                    for (int row = 0; row < count; ++row) {
                        ids[row] = ((start + row) * 3 + (poison && row >= keep ? 71 : 0)) % 256;
                        pos[row] = start + row;
                    }
                    Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                    TargetCapture capture;
                    capture.verifying = verifying;
                    Data out = RunDraftTarget(input, positions, kv, cfg, capture);
                    out.ToDevice(DataDevice::CPU);
                    if (out.dims != std::vector<int>({1, verifying ? count : 1, 256}))
                        throw std::runtime_error("rollback logit shape mismatch");
                    return std::vector<float>((float *)out.cpuData,
                                              (float *)out.cpuData + out.Count(0));
                };
                forward(clean, 0, past, false, false);
                forward(poisoned, 0, past, false, false);
                auto expected = forward(clean, past, rows, true, false);
                auto actual = forward(poisoned, past, rows, true, true);
                if (!std::equal(expected.begin(), expected.begin() + keep * 256, actual.begin()))
                    throw std::runtime_error("rejected token changed earlier verify rows");
                // Repeating the same width/input after rejecting every row must be exact.
                CommitTargetCache(clean, past, 0);
                if (forward(clean, past, rows, true, false) != expected)
                    throw std::runtime_error("same-path rollback/replay differs");
                CommitTargetCache(clean, past, keep);
                CommitTargetCache(poisoned, past, keep);
                for (int layer = 0; layer < 2; ++layer)
                    if (clean[layer].first.dims != poisoned[layer].first.dims ||
                        clean[layer].second.dims != poisoned[layer].second.dims)
                        throw std::runtime_error("same-path rollback metadata differs");
                if (forward(clean, past + keep, 1, false, false) !=
                    forward(poisoned, past + keep, 1, false, false))
                    throw std::runtime_error("rejected suffix changed same-path subsequent decode");
                ++checks;
            }
        std::cout << "VERIFY ROLLBACK PASS checks=" << checks << std::endl;
    }
    void VerifyBlocks() {
        // Compare against the same rank count: TP has its own reduction tree.
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int past : {3, 7, 15, 17})
            for (int rows : {2, 3, 4, 5, 6, 7, 8}) {
                for (int keep : {1, rows / 2, rows}) {
                    std::vector<std::pair<Data, Data>> reference(2), candidate(2);
                    GenerationConfig cfg;
                    cfg.input_token_length = past;
                    cfg.output_token_limit = 32;
                    cfg.output_logits = true;
                    TargetCapture referenceCapture;
                    auto forward = [&](auto &kv, int start, int count) {
                        std::vector<float> ids(count), pos(count), result;
                        for (int i = 0; i < count; ++i) {
                            ids[i] = (start + i) * 3 % 256;
                            pos[i] = start + i;
                        }
                        Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                        Data logits = RunDraftTarget(input, positions, kv, cfg, referenceCapture);
                        logits.ToDevice(DataDevice::CPU);
                        result.assign((float *)logits.cpuData,
                                      (float *)logits.cpuData + logits.Count(0));
                        return result;
                    };
                    forward(reference, 0, past);
                    forward(candidate, 0, past);
                    std::vector<float> ids(rows), pos(rows);
                    for (int i = 0; i < rows; ++i) {
                        ids[i] = (past + i) * 3 % 256;
                        pos[i] = past + i;
                    }
                    Data input(FLOAT32, {1, rows}, ids), positions(FLOAT32, {1, rows}, pos);
                    TargetCapture capture;
                    capture.verifying = true;
                    Data block = RunDraftTarget(input, positions, candidate, cfg, capture);
                    block.ToDevice(DataDevice::CPU);
                    if (block.dims != std::vector<int>({1, rows, 256}) ||
                        capture.hidden.size() != 2)
                        throw std::runtime_error("missing verification logits/features");
                    for (int row = 0; row < keep; ++row) {
                        auto one = forward(reference, past + row, 1);
                        const float *actual = (const float *)block.cpuData + row * 256;
                        RequireCloseLogits(actual, one, "block verification differs from decode");
                        ++checks;
                    }
                    CommitTargetCache(candidate, past, keep);
                    for (int layer = 0; layer < 2; ++layer) {
                        if (candidate[layer].first.dims != reference[layer].first.dims)
                            throw std::runtime_error("rollback root length mismatch");
                        if (ranks > 1)
                            for (int device = 0; device < ranks; ++device)
                                if (candidate[layer].first.multiDeviceDatas.at(device)->dims !=
                                    reference[layer].first.multiDeviceDatas.at(device)->dims)
                                    throw std::runtime_error("rollback rank length mismatch");
                    }
                    auto expectedNext = forward(reference, past + keep, 1);
                    auto actualNext = forward(candidate, past + keep, 1);
                    if (actualNext.size() != expectedNext.size())
                        throw std::runtime_error("post-commit logit shape mismatch");
                    RequireCloseLogits(actualNext.data(), expectedNext,
                                       "rejected suffix changed subsequent decode");
                    ++checks;
                }
            }
        std::cout << "LOGIT ERROR relative_max=" << largestLogitRelative
                  << " absolute_max=" << largestLogitAbsolute << std::endl;
        std::cout << "VERIFY BLOCK PASS ranks=" << ranks << " checks=" << checks << std::endl;
    }
};
// A 4096-wide BF16 verification block crosses the TP8 48-KiB policy
// boundary at six rows. Compare full-block reduction against ordinary decode.
static void VerifyDraftReduction(int ranks) {
    std::vector<int> devices(ranks);
    for(int i=0;i<ranks;++i) devices[i]=i;
    void *group=FastllmCudaNaiveDraftTPCreate(devices);
    if(!group) throw std::runtime_error("draft reduction group");
    int checks=0;
    {
        PersistentWorkerGroup workers;
        for(int rows : {5,40,640}) {
            constexpr int width=4096;
            const size_t count=(size_t)rows*width;
            std::vector<double> expected(count), absolute(count);
            std::vector<Data> inputs(ranks);
            for(int rank=0;rank<ranks;++rank) {
                auto &x=inputs[rank];x.dataType=FLOAT32;x.UpdateUnitSize();x.Resize({rows,width});x.Allocate();
                unsigned seed=43+rank;
                for(size_t i=0;i<count;++i) {
                    seed=seed*1664525u+1013904223u;
                    float value=std::ldexp((int(seed>>16)-32768)/32768.f,int((i+rank*7)%25)-12);
                    value=BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(value));
                    ((float*)x.cpuData)[i]=value;expected[i]+=value;absolute[i]+=std::abs(double(value));
                }
                FastllmCudaSetDevice(rank);x.ToDevice(DataDevice::CUDA,std::vector<int>{rank});
            }
            std::vector<std::exception_ptr> errors(ranks);
            workers.Run(devices,[&](int rank) {
                FastllmCudaSetDevice(rank);
                FastllmCudaNaiveDraftTPReduce(group,rank,inputs[rank]);ForceDeviceSync();
            },errors);
            for(auto error:errors) if(error) std::rethrow_exception(error);
            for(int rank=0;rank<ranks;++rank) {
                FastllmCudaSetDevice(rank);inputs[rank].ToDevice(DataDevice::CPU);
                for(size_t i=0;i<count;++i) {
                    double actual=((float*)inputs[rank].cpuData)[i];
                    // FP32 summation forward-error bound, independent of the
                    // NCCL tree/channel order; the CPU sum is exact for these BF16 inputs.
                    double bound=ranks*std::numeric_limits<float>::epsilon()*absolute[i];
                    if(!std::isfinite(actual) || std::abs(actual-expected[i])>bound)
                        throw std::runtime_error("draft FP32 reduction violates CPU error bound");
                }
                ++checks;
            }
        }
    }
    FastllmCudaNaiveDraftTPDestroy(group);FastllmCudaSetDevice(0);
    std::cout<<"DRAFT REDUCTION CPU PASS checks="<<checks<<std::endl;
}

static void VerifyChunkedRowAllReduce(int ranks) {
    SetCudaGraph(true);
    std::vector<int> devices(ranks);
    std::iota(devices.begin(), devices.end(), 0);
    if (!FastllmInitNccl(devices)) throw std::runtime_error("collective init");
    PersistentWorkerGroup workers;
    std::vector<std::exception_ptr> errors(ranks);
    auto run = [&](const std::function<void(int)> &fn) {
        std::fill(errors.begin(), errors.end(), nullptr);
        workers.Run(devices, [&](int rank) { FastllmCudaSetDevice(rank); fn(rank); }, errors);
        for (auto e : errors) if (e) std::rethrow_exception(e);
    };
    auto cuda = [](cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    };
    int checks = 0;
    for (DataType type : {FLOAT16, BFLOAT16, FLOAT32}) {
        const int unit = type == FLOAT32 ? 4 : 2;
        auto store = [type](void *p, int i, float value) -> float {
            if (type == FLOAT32) { ((float *)p)[i] = value; return value; }
            if (type == BFLOAT16) {
                const uint16_t bits = Float32ToBFloat16RNEBits(value);
                ((uint16_t *)p)[i] = bits; return BFloat16BitsToFloat32(bits);
            }
            const __half half = __float2half_rn(value);
            std::memcpy((uint16_t *)p + i, &half, sizeof(half));
            return __half2float(half);
        };
        for (auto shape : std::vector<std::pair<int, int>>{
                {4096, 8 * 1024 * 1024 / (4096 * unit) - 1},
                {4096, 8 * 1024 * 1024 / (4096 * unit)},
                {4096, 8 * 1024 * 1024 / (4096 * unit) + 1},
                {4096, 1520}, {4096, 2 * 8 * 1024 * 1024 / (4096 * unit) + 3},
                {3072, 1801}}) {
            const int width = shape.first, rows = shape.second, count = rows * width;
            const size_t bytes = (size_t)count * unit;
            std::vector<std::vector<uint8_t>> host(ranks, std::vector<uint8_t>(bytes));
            std::vector<float> sum(count, 0.0f);
            std::vector<uint8_t> expected(bytes);
            std::vector<void *> input(ranks), output(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                unsigned seed = 71 + rank;
                for (int i = 0; i < count; ++i) {
                    seed = seed * 1664525u + 1013904223u;
                    const float value = std::ldexp((int(seed >> 16) - 32768) / 32768.f,
                                                   (i + rank * 3) % 17 - 8);
                    sum[i] += store(host[rank].data(), i, value);
                }
            }
            for (int i = 0; i < count; ++i) store(expected.data(), i, sum[i]);
            run([&](int rank) {
                cuda(cudaMalloc(&input[rank], bytes)); cuda(cudaMalloc(&output[rank], bytes));
            });
            auto reset = [&]() {
                run([&](int rank) {
                    cuda(cudaMemcpy(input[rank], host[rank].data(), bytes, cudaMemcpyHostToDevice));
                    cuda(cudaMemset(output[rank], 0xa5, bytes));
                    cuda(cudaStreamSynchronize(cudaStreamPerThread));
                });
            };
            auto check = [&](bool inplace) {
                run([&](int rank) {
                    std::vector<uint8_t> actual(bytes);
                    cuda(cudaMemcpy(actual.data(), inplace ? input[rank] : output[rank],
                                    bytes, cudaMemcpyDeviceToHost));
                    if (actual != expected) throw std::runtime_error("chunked reduction differs from CPU rank sum");
                    if (!inplace) {
                        cuda(cudaMemcpy(actual.data(), input[rank], bytes, cudaMemcpyDeviceToHost));
                        if (actual != host[rank]) throw std::runtime_error("out-of-place input changed");
                    }
                });
                checks += ranks;
            };
            reset();
            // Even when the first chunk is registered, a cold later chunk must
            // reject the entire capture before enqueueing a reduction.
            if (bytes > 8 * 1024 * 1024) {
                const int first = (8 * 1024 * 1024 / (width * unit)) * width;
                // Reset registrations so allocator address reuse cannot make
                // this deliberately cold second chunk look warm.
                FastllmCudaCustomAllReduceReset();
                if (!FastllmCudaCustomAllReduceInit(devices)) throw std::runtime_error("collective reinit");
                run([&](int rank) {
                    if (!FastllmCudaCustomAllReduceRows(input[rank], output[rank], first, width, type, rank))
                        throw std::runtime_error("first chunk registration");
                    ForceDeviceSync();
                });
                reset();
                run([&](int rank) {
                    cudaGraph_t graph = nullptr;
                    cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
                    bool accepted = FastllmCudaCustomAllReduceRows(input[rank], input[rank], count, width, type, rank);
                    cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    size_t nodes = 0; cuda(cudaGraphGetNodes(graph, nullptr, &nodes));
                    cuda(cudaGraphDestroy(graph));
                    if (accepted || nodes) throw std::runtime_error("cold chunk partially captured");
                    std::vector<uint8_t> actual(bytes);
                    cuda(cudaMemcpy(actual.data(), input[rank], bytes, cudaMemcpyDeviceToHost));
                    if (actual != host[rank]) throw std::runtime_error("cold capture modified input");
                });
                checks += ranks;
                run([&](int rank) {
                    void *bad = rank ? input[rank] : (uint8_t *)input[rank] + unit;
                    if (FastllmCudaCustomAllReduceRows(bad, input[rank], count, width, type, rank))
                        throw std::runtime_error("misaligned rank accepted");
                    std::vector<uint8_t> actual(bytes);
                    cuda(cudaMemcpy(actual.data(), input[rank], bytes, cudaMemcpyDeviceToHost));
                    if (actual != host[rank]) throw std::runtime_error("rejected reduction modified input");
                });
                checks += ranks;
            }
            for (bool inplace : {false, true}) {
                reset();
                run([&](int rank) {
                    if (!FastllmCudaCustomAllReduceRows(input[rank], inplace ? input[rank] : output[rank],
                                                       count, width, type, rank))
                        throw std::runtime_error("chunked row reduction fell back");
                    ForceDeviceSync();
                });
                check(inplace);
                std::vector<cudaGraphExec_t> execs(ranks);
                run([&](int rank) {
                    cudaGraph_t graph = nullptr;
                    cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
                    bool accepted = FastllmCudaCustomAllReduceRows(input[rank], inplace ? input[rank] : output[rank],
                                                                  count, width, type, rank);
                    cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    if (!accepted) throw std::runtime_error("warm chunk graph fell back");
                    size_t nodes = 0; cuda(cudaGraphGetNodes(graph, nullptr, &nodes));
                    if (!nodes || nodes > 16) throw std::runtime_error("unexpected chunk graph size");
                    cuda(cudaGraphInstantiateWithFlags(&execs[rank], graph, 0));
                    cuda(cudaGraphDestroy(graph));
                });
                for (int replay = 0; replay < 3; ++replay) {
                    reset();
                    run([&](int rank) { cuda(cudaGraphLaunch(execs[rank], cudaStreamPerThread)); ForceDeviceSync(); });
                    check(inplace);
                }
                run([&](int rank) { cuda(cudaGraphExecDestroy(execs[rank])); });
            }
            run([&](int rank) { cuda(cudaFree(input[rank])); cuda(cudaFree(output[rank])); });
        }
    }
    FastllmCudaCustomAllReduceReset();
    std::cout << "CHUNKED ROW ALLREDUCE PASS checks=" << checks << std::endl;
}

static void VerifyRowAllReduce(int ranks) {
    SetCudaGraph(true);
    std::vector<int> devices(ranks);
    for (int i = 0; i < ranks; ++i)
        devices[i] = i;
    if (!FastllmInitNccl(devices))
        throw std::runtime_error("collective init");
    PersistentWorkerGroup workers;
    std::vector<std::exception_ptr> errors(ranks);
    auto run = [&](const std::function<void(int)> &fn) {
        std::fill(errors.begin(), errors.end(), nullptr);
        workers.Run(
            devices,
            [&](int rank) {
                FastllmCudaSetDevice(rank);
                fn(rank);
            },
            errors);
        for (auto e : errors)
            if (e)
                std::rethrow_exception(e);
    };
    int checks = 0;
    for (DataType type : {BFLOAT16, FLOAT32})
        for (int rows : {2, 5, 6, 7, 8}) {
            constexpr int width = 4096;
            std::vector<Data> reference(ranks), block(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                Data input(type, {rows, width});
                input.Allocate();
                unsigned seed = 37 + rank;
                for (int i = 0; i < rows * width; ++i) {
                    seed = seed * 1664525 + 1013904223;
                    float value = (int(seed >> 16) - 32768) / 2048.f;
                    if (type == FLOAT32)
                        ((float *)input.cpuData)[i] = value;
                    else {
                        uint32_t bits;
                        std::memcpy(&bits, &value, 4);
                        ((uint16_t *)input.cpuData)[i] = bits >> 16;
                    }
                }
                reference[rank].CopyFrom(input);
                block[rank].CopyFrom(input);
                reference[rank].ToDevice(DataDevice::CUDA, std::vector<int>{rank});
                block[rank].ToDevice(DataDevice::CUDA, std::vector<int>{rank});
            }
            run([&](int rank) {
                auto &d = reference[rank];
                for (int row = 0; row < rows; ++row) {
                    void *ptr = (uint8_t *)d.cudaData + (size_t)row * width * d.unitSize;
                    FastllmNcclAllReduce(ptr, ptr, width, type, rank);
                }
                ForceDeviceSync();
            });
            run([&](int rank) {
                auto &d = block[rank];
                if (!FastllmCudaCustomAllReduceRows(d.cudaData, d.cudaData, d.Count(0), width, type,
                                                    rank)) {
                    for (int row = 0; row < rows; ++row) {
                        void *ptr = (uint8_t *)d.cudaData + (size_t)row * width * d.unitSize;
                        FastllmNcclAllReduce(ptr, ptr, width, type, rank);
                    }
                }
                ForceDeviceSync();
            });
            for (int rank = 0; rank < ranks; ++rank) {
                reference[rank].ToDevice(DataDevice::CPU);
                block[rank].ToDevice(DataDevice::CPU);
                if (std::memcmp(reference[rank].cpuData, block[rank].cpuData,
                                block[rank].GetBytes()))
                    throw std::runtime_error("block collective differs from single-row reduction");
                ++checks;
            }
        }
    std::cout << "ROW ALLREDUCE PASS checks=" << checks << std::endl;
}
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
class DraftFixture : public NaiveN05FlashModel {
  public:
    DraftFixture(bool merged = false, int kvHeads = 2, int owner = 0, int block = 7, bool large = false, int layers = 2) {
        embed_dim = large ? 4096 : 256;
        block_cnt = 2;
        draftLayers = layers;
        draftBlock = block;
        draftHeads = large ? 32 : 8;
        draftKvHeads = kvHeads;
        draftHeadDim = large ? 128 : 32;
        draftWindow = 1024;
        deviceMap = {{"cuda:" + std::to_string(owner), 1}};
        draftTargetLayers = {0, 1};
        AddDraftTestWeights(weight, draftLayers, draftKvHeads, 513, merged,
                            embed_dim, draftHeadDim, large ? 4096 : 512);
        // Match the model loader's GPU embedding placement in both paths.
        weight["model.embed_tokens.weight"].ToDevice(DataDevice::CUDA, std::vector<int>{owner});
        weight["dspark.mask_embedding"].ToDevice(DataDevice::CUDA, std::vector<int>{owner});
    }
    void PrefillCommitChecks() {
        setenv("FASTLLM_DSPARK_TP", "1", 1);
        auto actual = CreateDraftContext(), reference = CreateDraftContext();
        auto bits = [](const Data &x) {
            Data host(x); host.ToDevice(DataDevice::CPU);
            return std::vector<uint16_t>((uint16_t *)host.cpuData, (uint16_t *)host.cpuData + host.Count(0));
        };
        int total = 0;
        for (int rows : {8192, 3, 512, 7, 4096, 1}) {
            TargetCapture capture;
            for (int layer : draftTargetLayers) {
                Data &feature = capture.hidden[layer];
                feature.dataType = BFLOAT16; feature.Resize({1, rows, embed_dim}); feature.Allocate();
                for (size_t i = 0; i < feature.Count(0); ++i)
                    ((uint16_t *)feature.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + total + layer * 19) * .031f));
                feature.ToDevice(DataDevice::CUDA, {0});
            }
            std::vector<std::pair<Data, Data>> unused;
            CommitDraftContext(capture, rows, *actual, unused);
            Data combined, projected, hidden;
            Cat(capture.hidden.at(0), capture.hidden.at(1), -1, combined);
            if (rows > 1) MatMulTransB(combined, weight["dspark.fc.weight"], projected);
            else Linear(combined, weight["dspark.fc.weight"], Data(), projected);
            KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, hidden);
            AppendDraftContext(hidden, reference->committed, *reference);
            total += rows;
            if (actual->committed != total || reference->committed != total)
                throw std::runtime_error("prefill tail changed logical context length");
            for (int layer = 0; layer < draftLayers; ++layer)
                if (bits(actual->kv[layer].first) != bits(reference->kv[layer].first) ||
                    bits(actual->kv[layer].second) != bits(reference->kv[layer].second))
                    throw std::runtime_error("prefill tail projection changed visible draft KV");
        }
        std::cout << "DRAFT PREFILL TAIL PASS" << std::endl;
    }
    static void RunTPChecks(int ranks, bool mlpOnly, bool singleKV = false, int owner = 0) {
        const std::string tp = (mlpOnly ? "mlp:" : "") + std::to_string(ranks);
        DraftFixture reference(true, singleKV ? 1 : ranks, owner);
        DraftFixture candidate(true, singleKV ? 1 : ranks, owner);
        if (!FastllmInitNccl(std::vector<int>{0, 1})) throw std::runtime_error("target group init");
        const auto generation = FastllmGetNcclGeneration();
        auto bits = [](const Data &x) {
            size_t count = 1; for (int d : x.dims) count *= d;
            std::vector<uint16_t> out(count);
            if (cudaMemcpy(out.data(), x.cudaData, count * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("TP fixture read");
            return out;
        };
        auto floats = [&](const Data &x) {
            auto raw = bits(x); std::vector<float> out;
            for (auto value : raw) out.push_back(BFloat16BitsToFloat32(value));
            return out;
        };
        auto hidden = [&](int rows, int seed) {
            Data x(BFLOAT16, {1, rows, 256}); x.Allocate();
            for (int i = 0; i < x.Count(0); ++i)
                ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
            x.ToDevice(DataDevice::CUDA, std::vector<int>{owner}); return x;
        };
        int checks = 0;
        for (int prefix : {3, 80, 249, 250, 1023, 1024, 32768, 3}) {
            auto a = reference.CreateDraftContext(), b = candidate.CreateDraftContext();
            int count = std::min(prefix, candidate.draftWindow - 1);
            Data h = hidden(count, prefix);
            setenv("FASTLLM_DSPARK_TP", "1", 1); reference.AppendDraftContext(h, prefix - count, *a);
            setenv("FASTLLM_DSPARK_TP", tp.c_str(), 1); candidate.AppendDraftContext(h, prefix - count, *b);
            if (FastllmGetNcclGeneration() != generation) throw std::runtime_error("draft replaced target group");
            for (int layer = 0; layer < candidate.draftLayers; ++layer) {
                const std::string p = "dspark.layers." + std::to_string(layer) + ".";
                for (const char *name : {"self_attn.mergeqkv.weight", "self_attn.o_proj.weight",
                                         "mlp.gateup_proj.weight", "mlp.down_proj.weight"}) {
                    Data &w = candidate.weight[p + name];
                    if (w.cpuData || w.cudaData || w.multiDeviceDatas.size() != (size_t)ranks)
                        throw std::runtime_error("Draft TP retained source or missing rank");
                    const bool replicated = mlpOnly && std::string(name).find("self_attn.") == 0;
                    size_t total = 0; std::vector<uint16_t> first;
                    for (auto &entry : w.multiDeviceDatas) {
                        Data &local = *entry.second; total += local.GetBytes();
                        if (replicated) {
                            if (local.dims != w.dims || !w.IsTensorParallelReplicated())
                                throw std::runtime_error("attention replica shape");
                            FastllmCudaSetDevice(entry.first); auto current = bits(local);
                            if (first.empty()) first = current;
                            else if (first != current) throw std::runtime_error("attention replicas differ");
                        }
                    }
                    if (total != w.GetBytes() * (replicated ? ranks : 1))
                        throw std::runtime_error("Draft TP physical weight bytes");
                }
            }
            FastllmCudaSetDevice(owner);
            for (int round = 0; round < 4; ++round) {
                SetCudaGraph(false); Data expected = reference.RunDraft(11 + round, *a);
                auto ef = floats(expected);
                SetCudaGraph(true); Data actual = candidate.RunDraft(11 + round, *b);
                auto af = floats(actual);
                RequireCloseLogits(af.data(), ef, "TP versus single draft");
                SetCudaGraph(false); Data eager = candidate.RunDraft(11 + round, *b);
                if (bits(actual) != bits(eager)) throw std::runtime_error("TP graph/eager bits differ");
                if (a->committed != b->committed) throw std::runtime_error("TP advanced committed prefix");
                for (int layer = 0; layer < candidate.draftLayers; ++layer) {
                    for (int part = 0; part < 2; ++part) {
                        const Data &ka = part ? a->kv[layer].second : a->kv[layer].first;
                        const Data &kb = part ? b->kv[layer].second : b->kv[layer].first;
                        auto av = floats(ka), bv = floats(kb); std::vector<float> wanted;
                        if (ka.dims[1] != kb.dims[1] || ka.dims[2] != (mlpOnly ? 1 : ranks) * kb.dims[2]) throw std::runtime_error("TP cache layout");
                        for (int row = 0; row < ka.dims[1]; ++row)
                            wanted.insert(wanted.end(), av.begin() + row * ka.dims[2], av.begin() + row * ka.dims[2] + kb.dims[2]);
                        RequireCloseLogits(bv.data(), wanted, "TP cache heads");
                    }
                }
                const int accepted = round == 3 ? 8 : round + 1;
                Data next = hidden(accepted, prefix + round + 91);
                setenv("FASTLLM_DSPARK_TP", "1", 1); reference.AppendDraftContext(next, a->committed, *a);
                setenv("FASTLLM_DSPARK_TP", tp.c_str(), 1); candidate.AppendDraftContext(next, b->committed, *b);
                if (cudaMemsetAsync(next.cudaData, 0, next.GetBytes(), cudaStreamPerThread) != cudaSuccess)
                    throw std::runtime_error("TP context input reuse");
                ++checks;
            }
            reference.idleDraftContext = std::move(a); candidate.idleDraftContext = std::move(b);
        }
        if (!failVerifyBegin && !failVerifyInstantiate && verifyGraphLaunches == 0) throw std::runtime_error("no TP graph replay");
        if ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches != 0) throw std::runtime_error("TP graph failure not rolled back");
        std::cout << "DRAFT " << (mlpOnly ? "MLP " : "") << "TP" << ranks << " PASS checks=" << checks << " relative=" << largestLogitRelative
                  << " maximum=" << largestLogitAbsolute << " captures=" << verifyGraphCaptures << " launches=" << verifyGraphLaunches << std::endl;
        unsetenv("FASTLLM_DSPARK_TP");
    }
    void RunWideBatchChecks() {
        constexpr int count = 128;
        auto read = [](const Data &x) {
            std::vector<uint16_t> bits(x.Count(0));
            if (cudaMemcpy(bits.data(), x.cudaData, bits.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("wide draft read");
            return bits;
        };
        auto equal = [&](const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, const char *what) {
            if (a==b) return;
            std::vector<float> af, bf; size_t changed=0;
            for(size_t i=0;i<a.size();++i) {af.push_back(BFloat16BitsToFloat32(a[i]));bf.push_back(BFloat16BitsToFloat32(b[i]));changed+=a[i]!=b[i];}
            auto e=CompareLogits(af.data(),bf.data(),af.size());
            std::cerr<<what<<" changed="<<changed<<" relative="<<e.relative<<" max_abs="<<e.maximum<<" reference_max="<<e.scale<<std::endl;
            // FP32 NCCL can reorder additions (notably NVLS); small BF16
            // boundary changes may propagate through the five draft layers.
            // Keep a much tighter bound than the cross-GEMM model oracle.
            if (!e.acceptable || e.relative > 1.0/1024 || e.maximum > .01*e.scale)
                throw std::runtime_error(what);
        };
        std::vector<std::shared_ptr<DraftContext>> owned;
        std::vector<DraftContext *> contexts;
        std::vector<int> anchors;
        Data hidden(BFLOAT16, {1, draftWindow - 1, embed_dim}); hidden.Allocate();
        for (int i=0;i<hidden.Count(0);++i)
            ((uint16_t*)hidden.cpuData)[i]=Float32ToBFloat16RNEBits(std::sin(i*.137f));
        hidden.ToDevice(DataDevice::CUDA,std::vector<int>{0});
        for (int i=0;i<count;++i) {
            owned.push_back(CreateDraftContext()); contexts.push_back(owned.back().get());
            AppendDraftContext(hidden, i * 1024, *contexts.back());
            anchors.push_back((i*37+11)%256);
        }
        int checks=0;
        for (int turn=0;turn<3;++turn) {
            Data output;
            if (!RunDraftBatch(anchors,contexts,output)) throw std::runtime_error("wide draft fallback");
            auto expected=read(output);
            for (int trial=0;trial<10;++trial) {
                Data replay;
                if (!RunDraftBatch(anchors,contexts,replay)) throw std::runtime_error("wide replay fallback");
                equal(read(replay),expected,"wide draft replay exceeds FP32 reduction tolerance");
                ++checks;
            }
            SetCudaGraph(false); Data eager;
            if (!RunDraftBatch(anchors,contexts,eager)) throw std::runtime_error("wide eager fallback");
            equal(read(eager),expected,"wide draft graph/eager exceeds FP32 reduction tolerance");
            SetCudaGraph(true);
            std::vector<int> reordered;
            std::vector<DraftContext*> selected;
            for (int i=0;i<count;++i) {int at=(i*37+11)%count;reordered.push_back(anchors[at]);selected.push_back(contexts[at]);}
            Data permuted;
            if (!RunDraftBatch(reordered,selected,permuted)) throw std::runtime_error("wide reordered draft fallback");
            auto values=read(permuted);
            const size_t width=draftBlock*embed_dim;
            for(int i=0;i<count;++i) {
                int at=(i*37+11)%count;
                equal(std::vector<uint16_t>(values.begin()+i*width,values.begin()+(i+1)*width),
                    std::vector<uint16_t>(expected.begin()+at*width,expected.begin()+(at+1)*width),
                    "wide draft request permutation differs");
                ++checks;
            }
            for (auto &anchor : anchors) anchor=(anchor+17)%256;
        }
        std::cout << "DRAFT WIDE PASS checks=" << checks << std::endl;
    }
    void RunBatchChecks() {
        auto hidden = [&](int rows, int seed) {
            Data h(BFLOAT16, {1, rows, embed_dim}); h.Allocate();
            for (int i = 0; i < h.Count(0); ++i)
                ((uint16_t *)h.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
            h.ToDevice(DataDevice::CUDA, std::vector<int>{0}); return h;
        };
        auto read = [](const Data &x) {
            size_t count = 1;
            for (int dim : x.dims) count *= dim;
            std::vector<uint16_t> bits(count);
            if (x.dataType != BFLOAT16 || x.dataDevice != DataDevice::CUDA ||
                cudaMemcpy(bits.data(), x.cudaData, count * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("draft batch read failed");
            std::vector<float> values;
            values.reserve(count);
            for (uint16_t bit : bits) values.push_back(BFloat16BitsToFloat32(bit));
            return values;
        };
        std::vector<std::shared_ptr<DraftContext>> reference, actual;
        std::vector<DraftContext *> contexts;
        std::vector<int> anchors{11, 57, 19, 101, 7, 83, 29, 217}, past{80, 1023, 32768, 249, 250, 3, 1024, 511};
        for (int p : past) {
            reference.push_back(CreateDraftContext()); actual.push_back(CreateDraftContext());
            Data h = hidden(std::min(p, draftWindow - 1), p);
            AppendDraftContext(h, p - h.dims[1], *reference.back());
            AppendDraftContext(h, p - h.dims[1], *actual.back());
            contexts.push_back(actual.back().get());
        }
        int checks = 0, batchReplays = 0;
        for (int round = 0; round < 5; ++round) {
            std::vector<std::vector<float>> expected;
            for (int b = 0; b < (int)anchors.size(); ++b) {
                Data h = RunDraft(anchors[b], *reference[b]); expected.push_back(read(h));
            }
            Data result;
            int launches = verifyGraphLaunches;
            if (!RunDraftBatch(anchors, contexts, result)) throw std::runtime_error("draft batch fell back");
            batchReplays += verifyGraphLaunches - launches;
            auto values = read(result);
            SetCudaGraph(false);
            Data eager;
            if (!RunDraftBatch(anchors, contexts, eager) || values != read(eager))
                throw std::runtime_error("batch draft graph/eager differs bitwise");
            SetCudaGraph(true);
            if (values.size() != anchors.size() * draftBlock * embed_dim) throw std::runtime_error("draft batch rows missing");
            for (int b = 0; b < (int)anchors.size(); ++b) {
                RequireCloseLogits(values.data() + b * draftBlock * embed_dim, expected[b], "draft batch hidden");
                Data solo = RunDraft(anchors[b], *actual[b]);
                auto soloValues = read(solo);
                RequireCloseLogits(soloValues.data(), expected[b], "draft batch to single transition");
                if (actual[b]->committed != past[b]) throw std::runtime_error("draft batch committed proposals");
                for (int layer = 0; layer < draftLayers; ++layer) {
                    for (int part = 0; part < 2; ++part) {
                        Data &a = part ? actual[b]->kv[layer].second : actual[b]->kv[layer].first;
                        Data &r = part ? reference[b]->kv[layer].second : reference[b]->kv[layer].first;
                        if (a.dims != r.dims || read(a) != read(r))
                            throw std::runtime_error("draft batch overwrote committed KV");
                    }
                }
                const int keep = 1 + (round + b) % 6;
                Data h = hidden(keep, past[b] + b * 91);
                AppendDraftContext(h, past[b], *reference[b]); AppendDraftContext(h, past[b], *actual[b]);
                past[b] += keep; anchors[b] = (anchors[b] + 29) % 256; ++checks;
            }
        }
        // Compare mixed accepted lengths against independent commits, then
        // consume the committed caches in the next draft block.
        for (int turn = 0; turn < 5; ++turn) {
            std::vector<int> lengths(anchors.size(), 8), counts;
            TargetCapture capture;
            for (int layer : draftTargetLayers) capture.hidden[layer].CopyFrom(hidden(8 * anchors.size(), 1009 + layer + turn));
            for (int b = 0; b < (int)anchors.size(); ++b) {
                counts.push_back(1 + (b + turn) % 8);
                TargetCapture single;
                for (int layer : draftTargetLayers) Split(capture.hidden.at(layer), 1, b * 8, (b + 1) * 8, single.hidden[layer]);
                std::vector<std::pair<Data, Data>> unused;
                CommitDraftContext(single, counts.back(), *reference[b], unused);
            }
            if (!CommitDraftContexts(capture, lengths, counts, contexts)) throw std::runtime_error("draft commit batch fell back");
            for (int b = 0; b < (int)anchors.size(); ++b) {
                if (actual[b]->committed != reference[b]->committed) throw std::runtime_error("batch commit length differs");
                for (int layer = 0; layer < draftLayers; ++layer) for (int part = 0; part < 2; ++part) {
                    auto &a = part ? actual[b]->kv[layer].second : actual[b]->kv[layer].first;
                    auto &r = part ? reference[b]->kv[layer].second : reference[b]->kv[layer].first;
                    if (a.dims != r.dims) throw std::runtime_error("batch commit KV shape differs");
                    auto av = read(a), rv = read(r); RequireCloseLogits(av.data(), rv, "batch committed KV");
                }
                Data a = RunDraft(anchors[b], *actual[b]), r = RunDraft(anchors[b], *reference[b]);
                auto av = read(a), rv = read(r); RequireCloseLogits(av.data(), rv, "draft after batch commit");
                ++checks;
            }
        }
        // Reorder and resize the active set, exceeding the bounded graph cache.
        // Each captured KV binding must remain isolated from the next request set.
        for (int rotation = 0; rotation < 3; ++rotation) for (int count : {2, 3, 4, 8}) {
            std::vector<int> ids;
            std::vector<DraftContext *> selected;
            for (int i = 0; i < count; ++i) {
                const int at = (i + rotation) % contexts.size();
                ids.push_back(anchors[at]); selected.push_back(contexts[at]);
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                Data actual, eager;
                SetCudaGraph(true);
                int launches = verifyGraphLaunches;
                if (!RunDraftBatch(ids, selected, actual)) throw std::runtime_error("resized draft batch fell back");
                batchReplays += verifyGraphLaunches - launches;
                auto values = read(actual);
                SetCudaGraph(false);
                if (!RunDraftBatch(ids, selected, eager) || values != read(eager))
                    throw std::runtime_error("resized draft graph/eager differs bitwise");
                SetCudaGraph(true); ++checks;
            }
        }
        if (!failVerifyBegin && !failVerifyInstantiate && batchReplays == 0)
            throw std::runtime_error("batch draft never replayed a graph");
        if ((failVerifyBegin || failVerifyInstantiate) && batchReplays != 0)
            throw std::runtime_error("batch draft failed capture did not fall back");
        std::cout << "DRAFT BATCH PASS checks=" << checks << " graph_replays=" << batchReplays << " relative=" << largestLogitRelative
                  << " maximum=" << largestLogitAbsolute << std::endl;
    }

    void Run() {
        int checks = 0;
        auto read = [](const Data &value) {
            size_t count = 1;
            for (int dim : value.dims)
                count *= dim;
            std::vector<uint16_t> bits(count);
            if (cudaMemcpy(bits.data(), value.cudaData, count * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("draft read failed");
            return bits;
        };
        auto hidden = [&](int count, int seed) {
            Data x(BFLOAT16, {1, count, embed_dim});
            x.Allocate();
            for (int i = 0; i < x.Count(0); ++i)
                ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
            x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            return x;
        };
        for (int prefix : {3, 80, 249, 250, 1023, 1024, 32768, 3}) {
            auto candidate = CreateDraftContext();
            auto reference = std::make_shared<DraftContext>();
            int count = std::min(prefix, draftWindow - 1);
            Data h = hidden(count, prefix);
            AppendDraftContext(h, prefix - count, *candidate);
            AppendDraftContext(h, prefix - count, *reference);
            for (int round = 0; round < 4; ++round) {
                SetCudaGraph(false);
                Data expected = RunDraft(11 + round, *reference);
                auto bits = read(expected);
                SetCudaGraph(true);
                Data actual = RunDraft(11 + round, *candidate);
                if (bits != read(actual))
                    throw std::runtime_error("draft backbone differs from eager");
                if (candidate->committed != reference->committed)
                    throw std::runtime_error("draft proposal changed committed length");
                Data base(BFLOAT16, {1, draftBlock, 513});
                base.Allocate();
                for (int i = 0; i < base.Count(0); ++i)
                    ((uint16_t *)base.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + round + prefix) * .019f));
                base.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                std::vector<int> expectedIds, actualIds;
                int previous = 11 + round;
                for (int step = 0; step < draftBlock; ++step) {
                    Data id(FLOAT32, {1, 1}, {(float)previous}), latent, bias, logits, top;
                    Embedding(id, weight["dspark.markov_head.markov_w1.weight"], latent);
                    ToDataType(latent, BFLOAT16);
                    Linear(latent, weight["dspark.markov_head.markov_w2.weight"], Data(), bias);
                    Split(base, 1, step, step + 1, logits);
                    AddTo(logits, bias); ToDataType(logits, FLOAT32); TopK(logits, top, 1);
                    top.ToDevice(DataDevice::CPU);
                    previous = (int)((float *)top.cpuData)[0]; expectedIds.push_back(previous);
                }
                bool proposalGraph = RunDraftProposalGraph(11 + round, base, *candidate, actualIds);
                if (!failVerifyBegin && !failVerifyInstantiate) {
                    if (!proposalGraph || actualIds != expectedIds)
                        throw std::runtime_error("GPU proposal chain differs from eager");
                } else if (proposalGraph) throw std::runtime_error("failed proposal graph did not fall back");
                Data tail;
                Split(base, 1, 0, draftBlock - 1, tail);
                if (RunDraftProposalGraph(11, tail, *candidate, actualIds))
                    throw std::runtime_error("incomplete proposal block did not fall back");
                for (int layer = 0; layer < draftLayers; ++layer) {
                    if (read(candidate->kv[layer].first) != read(reference->kv[layer].first) ||
                        read(candidate->kv[layer].second) != read(reference->kv[layer].second))
                        throw std::runtime_error("draft proposal changed visible KV");
                }
                int accepted = round == 3 ? 8 : round + 1;
                Data next = hidden(accepted, round + prefix);
                TargetCapture capture;
                capture.hidden.emplace(0, next);
                capture.hidden.emplace(1, hidden(accepted, round + prefix + 17));
                Data joined, projected, normed;
                Cat(capture.hidden.at(0), capture.hidden.at(1), -1, joined);
                if (accepted > 1) MatMulTransB(joined, weight["dspark.fc.weight"], projected);
                else Linear(joined, weight["dspark.fc.weight"], Data(), projected);
                KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, normed);
                std::vector<std::pair<Data, Data>> unused;
                CommitDraftContext(capture, accepted, *candidate, unused);
                AppendDraftContext(normed, reference->committed, *reference);
                for (int layer = 0; layer < draftLayers; ++layer) {
                    if (read(candidate->kv[layer].first) != read(reference->kv[layer].first) ||
                        read(candidate->kv[layer].second) != read(reference->kv[layer].second))
                        throw std::runtime_error("draft commit workspace changed KV");
                }
                ++checks;
            }
            // Exercise the same bounded storage transfer as request removal.
            idleDraftContext = std::move(candidate);
        }
        if (!failVerifyBegin && !failVerifyInstantiate && verifyGraphLaunches == 0)
            throw std::runtime_error("draft did not replay a graph");
        if ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches != 0)
            throw std::runtime_error("failed draft graph was launched");
        std::cout << "DRAFT GRAPH PASS checks=" << checks << " captures=" << verifyGraphCaptures
                  << " launches=" << verifyGraphLaunches << std::endl;
    }
};
#endif

int main(int argc, char **argv) {
    try {
        CheckLogitComparison();
        int ranks = argc > 1 ? std::stoi(argv[1]) : 8;
        if (ranks != 1 && ranks != 2 && ranks != 4 && ranks != 8)
            return 2;
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < ranks)
            return 77;
        if (argc > 2 && std::string(argv[2]) == "reduce_rows") {
            VerifyRowAllReduce(ranks);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "reduce_chunks") {
            VerifyChunkedRowAllReduce(ranks);
            return 0;
        }
        SetThreads(4);
        SetDeviceMap({{"cuda:0", 1}});
        FastllmCudaSetDevice(0);
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        if (argc > 2 && std::string(argv[2]) == "draft_prefill_tail") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            DraftFixture fixture(true, 2, 0, 5); fixture.PrefillCommitChecks(); return 0;
        }
#endif
        if (argc > 2 && std::string(argv[2]) == "draft_prefix") {
            SetCudaEmbedding(true); SetCudaGraph(!(argc > 4 && std::string(argv[4]) == "eager"));
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.DraftPrefixChecks(); return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "verify_batch") {
            SetCudaEmbedding(true); SetCudaGraph(true);
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            failVerifyBegin = argc > 4 && std::string(argv[4]) == "failbegin";
            failVerifyInstantiate = argc > 4 && std::string(argv[4]) == "failinstantiate";
#endif
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.VerifyBatches();
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
            if (!verifyGraphCaptures || ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches))
                throw std::runtime_error("batch graph capture/fallback was not exercised");
            std::cout << "BATCH GRAPH captures=" << verifyGraphCaptures << " launches=" << verifyGraphLaunches << std::endl;
#endif
            return 0;
        }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        if (argc > 2 && std::string(argv[2]) == "draft_tp_wide") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            VerifyDraftReduction(ranks);
            DraftFixture fixture(true, 4, 0, 5, true, 5);
            fixture.RunWideBatchChecks();
            unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "draft_tp_large") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            DraftFixture fixture(true, 4, 0, 5, true);
            fixture.RunBatchChecks();
            unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "draft_batch") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            failGraphDevice = 0;
            if (argc > 5) {
                failVerifyBegin = std::string(argv[5]) == "failbegin";
                failVerifyInstantiate = std::string(argv[5]) == "failinstantiate";
            }
            if (ranks > 1) setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            DraftFixture fixture(argc > 3 && std::string(argv[3]) == "merged", ranks > 1 ? std::min(4, ranks) : 2,
                                 0, argc > 4 ? std::stoi(argv[4]) : 7);
            fixture.RunBatchChecks();
            unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "kvmem_draft" || std::string(argv[2]) == "gpu_context_draft")) {
            const bool gpuContext = std::string(argv[2]) == "gpu_context_draft";
            SetCudaEmbedding(true); SetCudaGraph(true);
            setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            failGraphDevice = 0;
            failVerifyBegin = argc > 7 && std::string(argv[7]) == "failbegin";
            failVerifyInstantiate = argc > 7 && std::string(argv[7]) == "failinstantiate";
            const int block = argc > 4 ? std::stoi(argv[4]) : 5;
            const int resident = argc > 5 ? std::stoi(argv[5]) : 16384;
            std::vector<float> reference;
            for (bool enabled : {false, true}) {
                Fixture fixture(ranks, false, true, true, true);
                const std::string storage = argc > 6 ? argv[6] : "layer_shard";
                fixture.SetIndexerTPStorage(storage != "replicated" && !(enabled && storage == "context_shard"),
                                            enabled && storage == "context_shard");
                if (storage == "scaled_fp8" || storage == "context_shard") fixture.SetIndexerDataType(FP8_E4M3);
                if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
                auto result = fixture.SparseDraftChecks(enabled && !gpuContext, block, resident, !failVerifyBegin && !failVerifyInstantiate,
                    argc > 7 && std::string(argv[7]) == "warmup" ? 4096 : 1024);
                if (!enabled) reference = std::move(result);
                else {
                    if (result.size() != reference.size()) throw std::runtime_error("sparse draft trace shape");
                    RequireCloseLogits(result.data(), reference, "sparse draft target/hidden/backbone/head differs");
                    size_t exact = 0; for (size_t i = 0; i < result.size(); ++i) exact += result[i] == reference[i];
                    std::cout << (gpuContext ? "GPU CONTEXT DSPARK PASS exact=" : "KVMEM DSPARK PASS exact=") << exact << "/" << result.size()
                              << " relative=" << largestLogitRelative << " max_abs=" << largestLogitAbsolute << std::endl;
                }
            }
            if ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches)
                throw std::runtime_error("failed verification capture was replayed");
            std::cout << "KVMEM VERIFY GRAPH captures=" << verifyGraphCaptures
                      << " launches=" << verifyGraphLaunches << std::endl;
            unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "draft_prefill_batch") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            Fixture fixture(ranks, false, true, true, true);
            const bool context = argc > 4 && std::string(argv[4]) == "gpu_context";
            const bool sparse = !context && (argc <= 4 || std::string(argv[4]) != "ordinary");
            fixture.SetIndexerTPStorage(!sparse && !context, sparse || context);
            fixture.SetIndexerDataType(FP8_E4M3);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.DraftPrefixChecks(sparse, true); unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "kvmem_draft_prefix") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            setenv("FASTLLM_DSPARK_TP", std::to_string(ranks).c_str(), 1);
            Fixture fixture(ranks, false, true, true, true);
            fixture.SetIndexerTPStorage(true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.DraftPrefixChecks(true); unsetenv("FASTLLM_DSPARK_TP"); return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "kvmem_pool" || std::string(argv[2]) == "gpu_pool")) {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            const bool context = argc > 4 && std::string(argv[4]) == "context_shard";
            fixture.SetIndexerTPStorage(!context, context);
            if (context) fixture.SetIndexerDataType(FP8_E4M3);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.SharedPoolServingChecks(std::string(argv[2]) == "kvmem_pool",
                argc <= 5 || std::string(argv[5]) != "single_prefill");
            return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "kvmem" || std::string(argv[2]) == "gpu_context")) {
            const bool gpuContext = std::string(argv[2]) == "gpu_context";
            SetCudaEmbedding(true); SetCudaGraph(true);
            std::vector<float> reference;
            for (bool enabled : {false, true}) {
                Fixture fixture(ranks, false, true, true, true);
                fixture.SetIndexerTPStorage(!(gpuContext && enabled), gpuContext && enabled);
                if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
                if (argc > 4 && std::string(argv[4]) == "fp8") fixture.SetIndexerDataType(FP8_E4M3);
                auto result = fixture.SparseKvRun(enabled && !gpuContext);
                if (!enabled) reference = std::move(result);
                else {
                    if (reference.size() != result.size()) throw std::runtime_error("KVMem logits missing");
                    RequireCloseLogits(result.data(), reference, "KVMem changed model logits");
                    size_t exact = 0;
                    for (size_t i = 0; i < result.size(); ++i) exact += result[i] == reference[i];
                    std::cout << (gpuContext ? "GPU CONTEXT MODEL PASS ranks=" : "KVMEM MODEL PASS ranks=") << ranks << " exact=" << exact << "/" << result.size()
                        << " relative=" << largestLogitRelative << " max_abs=" << largestLogitAbsolute << std::endl;
                }
            }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "indexer_storage") {
            SetCudaEmbedding(true);
            const bool graph = argc <= 5 || std::string(argv[5]) != "eager";
            SetCudaGraph(graph);
            std::vector<float> reference;
            for (bool shard : {false, true}) {
                Fixture fixture(ranks, false, true, true, true, true);
                if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
                if (argc > 4 && std::string(argv[4]) == "fp8") fixture.SetIndexerDataType(FP8_E4M3);
                auto result = fixture.IndexerStorageRun(shard, graph);
                if (!shard) reference = std::move(result);
                else {
                    if (reference.size() != result.size()) throw std::runtime_error("Indexer storage logits missing");
                    RequireCloseLogits(result.data(), reference, "Indexer layer storage changed logits");
                    size_t exact = 0;
                    for (size_t i = 0; i < result.size(); ++i) exact += result[i] == reference[i];
                    std::cout << "INDEXER STORAGE PASS ranks=" << ranks << " exact=" << exact
                              << "/" << result.size() << " relative=" << largestLogitRelative << std::endl;
                }
            }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "serving_page_growth") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            if (argc > 4 && std::string(argv[4]) == "layer_shard") fixture.SetIndexerTPStorage(true);
            fixture.ServingPageGrowthChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "cuda_sampling") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            fixture.CudaSamplingChecks();
            return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "history_serving" || std::string(argv[2]) == "kvmem_prefix")) {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            if (argc > 4 && std::string(argv[4]) == "fp8") fixture.SetIndexerDataType(FP8_E4M3);
            if (argc > 5) fixture.SetIndexerTPStorage(std::string(argv[5]) == "layer_shard",
                std::string(argv[5]) == "context_shard");
            fixture.HistoryServingChecks(true, std::string(argv[2]) == "kvmem_prefix");
            return 0;
        }
#endif
        if (argc > 2 && std::string(argv[2]) == "feature_graph") {
            SetCudaEmbedding(true);
            Fixture fixture(ranks, true, true);
            fixture.FeatureGraphChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "history") {
            bool graph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaEmbedding(true);
            SetCudaGraph(graph);
            Fixture fixture(ranks, false, true, true, true);
            fixture.HistoryServingChecks(graph);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "selection") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, true, true); fixture.VerifySelections(ranks);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "head") {
            Fixture fixture(ranks);
            fixture.VerifyHead();
            return 0;
        }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        if (argc > 2 && std::string(argv[2]) == "serving_invalidation") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            fixture.ServingGraphInvalidationChecks();
            return 0;
        }
        if (argc > 2 && (std::string(argv[2]) == "serving_graph" || std::string(argv[2]) == "serving_padding" ||
                        std::string(argv[2]) == "serving_ragged" || std::string(argv[2]) == "serving_large")) {
            SetCudaEmbedding(true); SetCudaGraph(true);
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            failGraphDevice = ranks - 1;
            Fixture fixture(ranks, false, true, true, true);
            fixture.ServingGraphChecks(std::string(argv[2]) == "serving_padding",
                                       std::string(argv[2]) == "serving_ragged", std::string(argv[2]) == "serving_large");
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "swa_ring_toggle") {
            const bool graph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaEmbedding(true); SetCudaGraph(graph);
            Fixture fixture(ranks, true, true, true);
            fixture.RingGraphToggleChecks(graph);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "kv_pool") {
            Fixture fixture(ranks);
            fixture.CachePoolChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "kv_pool_churn") {
            SetCudaGraph(false);
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.CachePoolChurnChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "swa_workspace") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, false, true, true, true);
            if (argc > 3 && std::string(argv[3]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.SwaPrefillWorkspaceChecks();
            return 0;
        }
        const std::string draftMode = argc > 2 ? argv[2] : "";
        if (draftMode == "draft_tp" || draftMode == "draft_mlp_tp" ||
            draftMode == "draft_mlp_tp_mqa" || draftMode == "draft_mlp_tp_offset") {
            const bool offset = draftMode == "draft_mlp_tp_offset";
            if (offset && devices < 3) return 77;
            SetCudaEmbedding(true); SetCudaGraph(true);
            failGraphDevice = offset ? 0 : ranks - 1;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            DraftFixture::RunTPChecks(ranks, draftMode != "draft_tp",
                                     draftMode == "draft_mlp_tp_mqa", offset ? 2 : 0);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "draft_graph") {
            SetCudaEmbedding(true);
            failGraphDevice = 0;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            { DraftFixture fixture; fixture.Run(); }
            { DraftFixture fixture(true); fixture.Run(); }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "batch_prefill") {
            SetCudaEmbedding(true);
            Fixture fixture(ranks, true, true);
            fixture.PackedPrefillChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "batch_prefill_long") {
            SetCudaEmbedding(true);
            Fixture fixture(ranks, false, true, true, true);
            fixture.PackedPrefillChecks(true);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "batch") {
            bool graph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaEmbedding(true); SetCudaGraph(graph);
            failGraphDevice = ranks - 1;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            Fixture fixture(ranks, true, true);
            fixture.BatchChecks(graph, argc <= 4 || std::string(argv[4]) != "separate");
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "verify_graph") {
            expectVerifyGraph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaGraph(expectVerifyGraph);
            SetCudaEmbedding(true);
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            Fixture fixture(ranks, true, true, argc > 5 && std::string(argv[5]) == "ring");
            if (argc > 4 && std::string(argv[4]) == "fp8") fixture.SetKVCacheDataType(FP8_E4M3);
            fixture.VerifyGraphBlocks();
            return 0;
        }
#endif
        if (argc > 2 && std::string(argv[2]) == "rollback") {
            SetCudaGraph(ranks > 1);
            Fixture fixture(ranks, true, ranks > 1);
            fixture.VerifyRollback();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "verify") {
            Fixture fixture(ranks, argc > 3 && std::string(argv[3]) == "packed");
            fixture.VerifyBlocks();
            return 0;
        }
        std::vector<std::vector<float>> reference;
        {
            Fixture serial(1);
            reference = serial.Run();
        }
        Fixture parallel(ranks);
        auto actual = parallel.Run();
        for (int t = 0; t < 4; ++t) {
            double err = 0, bb = 0, aa = 0, ab = 0, mx = 0;
            for (size_t i = 0; i < actual[t].size(); ++i) {
                double a = actual[t][i], b = reference[t][i];
                err += (a - b) * (a - b);
                mx = std::max(mx, std::abs(a - b));
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            double relative = std::sqrt(err / bb), cos = ab / std::sqrt(aa * bb);
            std::cout << "step=" << t << " relative_rmse=" << relative << " cosine=" << cos
                      << " max_abs=" << mx << std::endl;
            if (!std::isfinite(relative) || relative > .02 || cos < .999)
                throw std::runtime_error("TP differs from serial logits");
        }
        std::cout << "MODEL SMOKE PASS" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
