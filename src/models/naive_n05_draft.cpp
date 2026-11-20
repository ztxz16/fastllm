#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "models/mimo_tool_grammar.h"
#include "pd.h"
#include "json11.hpp"
#include "executor.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <unordered_map>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-token-constraint.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
#ifdef USE_CUDA
    int CheckedDraftToken(float token, int vocab) {
        AssertInFastLLM(token >= 0 && token < vocab && token == int(token), "Invalid graph draft token.");
        return int(token);
    }
#endif

    bool DraftIsGreedy(const GenerationConfig &config) {
        if (!config.tool_call_token_mask) return config.IsSimpleGreedy();
        auto unmasked = config;
        unmasked.tool_call_token_mask.reset();
        return unmasked.IsSimpleGreedy();
    }

    int DraftGraphBucket(int count, int limit) {
        int bucket = 1;
        while (bucket < count && bucket <= INT_MAX / 2) bucket *= 2;
        return std::max(count, std::min(bucket, limit));
    }

    void CheckDraftPD(bool ok, const char *message) {
        if (!ok) throw std::runtime_error(std::string("FastKV Naive draft: ") + message);
    }

    void ClearDraftPrefixHidden(Data &hidden) {
        // Data's default assignment would discard the owning GPU pointer.
        hidden.FreeSpace();
        hidden.dims.clear();
        hidden.strides.clear();
        hidden.expansionDims.clear();
    }

    void DraftLinear(Data &input, Data &weight, Data &output) {
        // The block backbone benefits from tensor-core GEMM even at 2..7 rows.
        // MatMulTransB uses the same BF16 weights and FP32 accumulation, and
        // avoids the generic Linear small-batch GEMV dispatch.
        // Non-owning weight views must not enter Linear's persistent bias cache.
        if (weight.isFake || input.Count(0) / input.dims.back() > 1) MatMulTransB(input, weight, output);
        else Linear(input, weight, Data(), output);
    }
#ifdef USE_CUDA
    // Both receive paths keep these addresses stable for subsequent verify graphs.
    void ReserveDraftCache(Data &cache, int device, const std::vector<int> &shape, int capacity) {
        cache.dataType = BFLOAT16;
        cache.UpdateUnitSize();
        cache.ToDevice(CUDA, std::vector<int>{device}, false);
        if (cache.expansionDims.empty() || cache.expansionDims[1] < capacity)
            cache.Expansion({1, capacity, shape.back()});
        cache.Resize(shape);
        cache.SetKVCache();
    }

    void GatherDraftHeadInput(const Data &hidden, const std::vector<int> &limits, int block, Data &input) {
        const int width = hidden.dims.back(), rows = std::accumulate(limits.begin(), limits.end(), 0);
        input.dataType = hidden.dataType; input.UpdateUnitSize(); input.Resize({1, rows, width});
        input.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false); input.Allocate(false);
        std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
        size_t offset = 0;
        for (size_t row = 0; row < limits.size(); ++row) {
            dst.push_back((uint8_t *)input.cudaData + offset);
            src.push_back((const uint8_t *)hidden.cudaData + row * block * width * hidden.unitSize);
            const size_t size = (size_t)limits[row] * width * hidden.unitSize;
            bytes.push_back(size); offset += size;
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
            "Cannot gather batch draft head input.");
    }
#endif
}

void NaiveN05FlashModel::ConfigureDraftParallel(int ranks, bool replicas) {
    if (ranks < 1 || tpPrepared || !draftContexts.empty())
        throw std::invalid_argument("configure a positive draft group size before warmup");
    draftParallelLayout = (replicas ? "dp:" : "") + std::to_string(ranks);
}

#ifdef USE_CUDA
struct NaiveN05FlashModel::DraftWorkspace {
    struct Layer { Data normed, q, k, v, qkv, query, packed, attention, output, reduced, gate, up, gateUp, scores; };
    struct Graph {
        // The CUDA graph allocation pool has one capture session per process.
        // DP owners can replay independently after their captures finish.
        inline static std::mutex captureMutex;
        void *graph = nullptr, *exec = nullptr;
        std::vector<void *> reserved;
        void Clear() {
            FastllmCudaGraphExecDestroy(exec);
            FastllmCudaGraphDestroy(graph);
            exec = graph = nullptr;
            FastllmCudaGraphMemoryPoolRelease(reserved);
            reserved.clear();
        }
        // The caller owns warmup, replay and any state (including RNG) updates.
        // Capture failures disable this entry and leave eager fallback to it.
        template <class Body>
        bool Capture(Body &&body, bool &disabled) {
            std::lock_guard<std::mutex> lock(captureMutex);
            bool pool = FastllmCudaGraphPrepareCaptureDevice() && FastllmCudaGraphMemoryPoolBegin();
            bool ok = false;
            if (pool) {
                FastllmCudaClearThreadError();
                if (FastllmCudaGraphBeginCapture()) {
                    try { body(); }
                    catch (...) {
                        FastllmCudaGraphEndCapture(&graph);
                        FastllmCudaGraphMemoryPoolEnd(reserved);
                        Clear();
                        disabled = true;
                        throw;
                    }
                    const bool clean = !FastllmCudaGetThreadError();
                    ok = FastllmCudaGraphEndCapture(&graph) && clean;
                }
                ok = FastllmCudaGraphMemoryPoolEnd(reserved) && ok;
                if (ok) ok = FastllmCudaGraphInstantiate(graph, &exec);
            }
            if (!ok) {
                Clear();
                disabled = true;
                FastllmCudaClearLastError();
                FastllmCudaClearThreadError();
            }
            return ok;
        }
    } graphs[2];
    struct Proposal {
        Graph graph;
        bool disabled = false;
        std::vector<void *> weights;
        Data logits, ids, latent, bias, partial;
    } proposal;
    struct SamplingProposal {
        std::vector<Graph> graphs;
        std::array<void *, 3> inputs{};
        Data logits, ids, latent, bias, row, mask, plainMask;
        Data base, probabilities, stepProbabilities, target, verifyProposal;
        int rows = 1, steps = 0;
        std::shared_ptr<void> constraints;
        bool constrained = false;
        void *sampler = nullptr;
        bool ownsSampler = true;
        float temperature = -1, topP = -1;
        int topK = 0;
        bool disabled = false;
        float *hostIds = nullptr;
        void *idsReady = nullptr;
        void Clear() {
            if (idsReady) {
                FastllmCudaEventSynchronize(idsReady);
                FastllmCudaEventDestroy(idsReady);
                idsReady = nullptr;
            }
            if (hostIds) FastllmCudaHostFree(hostIds);
            hostIds = nullptr;
            for (auto &graph : graphs) graph.Clear();
            graphs.clear();
            if (ownsSampler) FastllmCudaMtpDestroyDraftSampler(sampler);
            sampler = nullptr;
            constraints.reset(); constrained = false;
        }
    } sampling;
    struct ContextLayer { Data raw, key, value; };
    int device = -1;
    bool disabled = false, warmed = false;
    std::vector<void *> inputs;
    std::vector<bool> shortRows;
    std::vector<int> rankRows;
    Data id, live, positions, hidden, normalized, batchKV;
    Data combined, projected, committedHidden, prefixStorage, contextSelected, contextRaw, contextPointers;
    std::vector<Layer> layers;
    // Captured QKV descriptors belong to their graph variant; layer scratch is shared.
    std::vector<Data> batchDescriptors;
    std::vector<ContextLayer> contextLayers;
    std::vector<std::shared_ptr<DraftWorkspace>> tpRanks;
    std::vector<std::shared_ptr<DraftWorkspace>> proposals, bucketProposals;
    std::map<int, std::shared_ptr<DraftWorkspace>> batchGraphs;
    std::shared_ptr<void> samplingLinear;
    Data samplingW1, samplingW2, samplingInput, samplingTarget;
    std::vector<Data> samplingTargetViews; // Host descriptors borrowing this sampling owner's target arena.
    std::vector<std::pair<Data, Data>> tpKV;
    void *done = nullptr;
    std::vector<void *> Pointers() const {
        std::vector<void *> out;
        for (const Data *p : {&id, &live, &positions, &hidden, &normalized, &combined, &batchKV}) out.push_back(p->isFake ? nullptr : p->cudaData);
        for (const auto &l : layers)
            for (const Data *p : {&l.normed, &l.q, &l.k, &l.v, &l.qkv, &l.query, &l.packed,
                    &l.attention, &l.output, &l.reduced, &l.gate, &l.up, &l.gateUp, &l.scores}) out.push_back(p->isFake ? nullptr : p->cudaData);
        return out;
    }
    ~DraftWorkspace() {
        int previous = FastllmCudaGetDevice();
        if (device >= 0) FastllmCudaSetDevice(device);
        for (auto &graph : graphs) graph.Clear();
        proposal.graph.Clear();
        proposals.clear(); bucketProposals.clear(); batchGraphs.clear();
        sampling.Clear();
        if (done) FastllmCudaEventDestroy(done);
        // Data frees use their recorded device; restore the caller's selection.
        FastllmCudaSetDevice(previous);
    }
};
// Draft subgroup state is independent of the target's TP communicator.
struct NaiveN05FlashModel::DraftTPState {
    std::vector<int> devices;
    int attentionDivisor = 1;
    bool dataParallel = false;
    size_t nextContextRank = 0;
    void *group = nullptr;
    PersistentWorkerGroup workers;
    std::vector<std::unique_ptr<Executor>> executors;
    ~DraftTPState() { workers.Stop(); FastllmCudaNaiveDraftTPDestroy(group); }
    void Run(const std::function<void(int)> &body) {
        std::vector<std::exception_ptr> errors(devices.size());
        workers.RunWithCaller(devices, [&](int rank) {
            FastllmCudaSetDevice(devices[rank]);
            // Each rank retains its dispatcher along with its worker. Building
            // every backend's operator registry on each launch hides TP savings.
            auto &executor = executors[rank];
            if (!executor) {
                executor = std::make_unique<Executor>();
                executor->SetFirstDevice("cuda:" + std::to_string(devices[rank]));
            }
            struct Restore { void *old; ~Restore() { SetCurrentThreadExecutor(old); } } restore{GetExecutor()};
            SetCurrentThreadExecutor(executor.get());
            body(rank);
        }, errors);
        FastllmCudaSetDevice(devices.front());
        for (auto error : errors) if (error) std::rethrow_exception(error);
    }
};

bool NaiveN05FlashModel::PrepareDraftTP() {
    if (draftTP) return true;
    const char *option = draftParallelLayout.empty() ? std::getenv("FASTLLM_DSPARK_TP") : draftParallelLayout.c_str();
    const bool automatic = !option || !*option;
    const std::string setting = automatic ? std::to_string(std::max<size_t>(1, tpDevices.size())) : option;
    auto singleDevice = [&]() {
        if (!draftSingleDeviceLogged) {
            printf("[Naive draft] backbone_tp=1 attention_tp=1 mode=%s\n",
                   automatic ? "auto-fallback" : "explicit");
            draftSingleDeviceLogged = true;
        }
        return false;
    };
    if (setting == "1" || setting == "dp:1") return singleDevice();
    const bool mlpOnly = setting.compare(0, 4, "mlp:") == 0;
    const bool dataParallel = setting.compare(0, 3, "dp:") == 0;
    const std::string count = dataParallel ? setting.substr(3) : mlpOnly ? setting.substr(4) : setting;
    const int available = FastllmCudaGetDeviceCount();
    AssertInFastLLM(!count.empty() && count.size() <= 9 &&
        std::all_of(count.begin(), count.end(), [](char c) { return c >= '0' && c <= '9'; }),
        "FASTLLM_DSPARK_TP must be a rank count, mlp:<rank count> or dp:<rank count>.");
    const int ranks = std::stoi(count);
    AssertInFastLLM(ranks >= 2 && ranks <= available, "Draft TP needs at least two available CUDA devices.");
    const bool supported = GetCudaEmbedding() && !GetLowMemMode() && draftHeads > 0 &&
        draftKvHeads > 0 && draftHeads % draftKvHeads == 0 && draftHeadDim > 0 &&
        draftHeadDim <= 256 && draftHeadDim % 4 == 0 && draftBlock > 1 && draftBlock < 32 &&
        draftLayers > 0 && embed_dim > 0 && draftWindow > 1 &&
        (int64_t)draftWindow + draftBlock <= INT_MAX &&
        (draftHeads + 2LL * draftKvHeads) * draftHeadDim <= INT_MAX &&
        (dataParallel || mlpOnly || (draftHeads % ranks == 0 && (draftKvHeads % ranks == 0 || ranks % draftKvHeads == 0)));
    if (automatic && !supported) return singleDevice();
    AssertInFastLLM(supported,
        "Draft TP requires dense BF16 weights, supported Q/KV heads and CUDA embedding.");
    auto state = std::make_shared<DraftTPState>();
    state->dataParallel = dataParallel;
    state->attentionDivisor = dataParallel || mlpOnly ? 1 : ranks;
    AssertInFastLLM(tpDevices.empty() || tpDevices.size() >= (size_t)ranks,
                    "Draft TP needs enough target devices.");
    if (!tpDevices.empty()) {
        state->devices.assign(tpDevices.begin(), tpDevices.begin() + ranks);
    } else {
        ApplyDraftDevice();
        state->devices.push_back(FastllmCudaGetDevice());
        for (int device = 0; device < available && (int)state->devices.size() < ranks; ++device)
            if (device != state->devices.front()) state->devices.push_back(device);
    }
    state->executors.resize(ranks);
    // Reject unsupported projections before transferring ownership of any weight.
    auto projection = [&](const std::string &name, const std::vector<int> &dims) {
        auto it = weight.weight.find(name);
        const bool valid = it != weight.weight.end() && it->second.dataType == BFLOAT16 &&
            !it->second.multiDeviceData && it->second.dims == dims;
        if (automatic) return valid;
        AssertInFastLLM(valid,
            "Draft TP needs a dense BF16 projection with matching shape: " + name);
        return true;
    };
    const int kvWidth = draftKvHeads * draftHeadDim;
    const std::vector<int> widths = draftKVOnly ? std::vector<int>{kvWidth, kvWidth} :
        std::vector<int>{draftHeads * draftHeadDim, kvWidth, kvWidth};
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        if (!projection(p + "self_attn." + DraftKVProjectionName(),
                        {(DraftContextQueryHeads() + 2 * draftKvHeads) * draftHeadDim, embed_dim})) return singleDevice();
        if (draftKVOnly) continue;
        auto down = weight.weight.find(p + "mlp.down_proj.weight");
        const bool valid = down != weight.weight.end() && down->second.dims.size() == 2 &&
            down->second.dims[1] > 0 && down->second.dims[1] <= INT_MAX / 2 &&
            (dataParallel || down->second.dims[1] % ranks == 0);
        if (automatic && !valid) return singleDevice();
        AssertInFastLLM(valid, "Draft MLP width must be divisible by its TP size.");
        const int mid = down->second.dims[1];
        if (!projection(p + "self_attn.o_proj.weight", {embed_dim, draftHeads * draftHeadDim}) ||
            !projection(p + "mlp.gateup_proj.weight", {2 * mid, embed_dim}) ||
            !projection(p + "mlp.down_proj.weight", {embed_dim, mid})) return singleDevice();
    }
    const bool peerAccess = FastllmCudaPeerAccessInit(state->devices);
    if (automatic && !peerAccess) return singleDevice();
    AssertInFastLLM(peerAccess, "Draft TP requires peer access.");
    auto split = [&](const std::string &name, int axis, const std::vector<int> &widths, bool replicated = false, bool qkv = false) {
        Data &w = weight[name], bias;
        AssertInFastLLM(w.dataType == BFLOAT16 && w.dims.size() == 2 && !w.multiDeviceData,
                        "Draft TP needs unsharded BF16 projection " + name);
        replicated = replicated || dataParallel;
        DivisionScheme scheme;
        int offset = 0;
        for (size_t part = 0; part < widths.size(); ++part) {
            const int width = widths[part];
            // When TP exceeds the KV-head count, neighboring query shards
            // share one whole KV head. Never split a head's RoPE channels.
            const int copies = qkv && part + 2 >= widths.size() && !replicated ? std::max(1, ranks / draftKvHeads) : 1;
            const int divisions = ranks / copies;
            AssertInFastLLM(width > 0 && (replicated || width % divisions == 0), "Draft TP projection cannot be split.");
            for (int rank = 0; rank < ranks; ++rank) {
                const int shard = rank / copies;
                scheme[state->devices[rank]].push_back(replicated ? std::make_pair(offset, offset + width) :
                    std::make_pair(offset + shard * width / divisions, offset + (shard + 1) * width / divisions));
            }
            offset += width;
        }
        AssertInFastLLM(offset == w.dims[axis], "Draft TP projection shape mismatch.");
        w.tpLinearType = replicated ? TP_LINEAR_NONE : axis == 0 ? TP_LINEAR_ROW : TP_LINEAR_COLUMN;
        AssertInFastLLM(SplitMultiCudaWeight(w, bias, state->devices, scheme, axis, true), "Draft TP split failed.");
        AssertInFastLLM(!w.cudaData && !w.cpuData, "Draft TP retained an unsplit weight copy.");
        if (replicated) {
            w.tpLayout = TP_LAYOUT_REPLICATED;
            w.tpAxis = -1;
            w.tpRanges.clear();
        }
    };
    auto replicate = [&](const std::string &name) {
        Data &w = weight[name];
        if (w.multiDeviceData) {
            AssertInFastLLM(w.IsTensorParallelReplicated(), "Draft input must be replicated: " + name);
            for (int device : state->devices) AssertInFastLLM(w.multiDeviceDatas.count(device), "Missing replicated draft input.");
        } else PrepareMultiCudaReplicatedData(w, state->devices, true);
    };
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        // MLP-only TP replicates attention using the same ownership transfer
        // as shards; neither strategy retains a second source weight payload.
        split(p + "self_attn." + DraftKVProjectionName(), 0, widths, mlpOnly, true);
        if (draftKVOnly) {
            replicate(p + "self_attn.k_norm.weight");
            continue;
        }
        split(p + "self_attn.o_proj.weight", mlpOnly ? 0 : 1,
              {mlpOnly ? embed_dim : draftHeads * draftHeadDim}, mlpOnly);
        const int mid = weight[p + "mlp.down_proj.weight"].dims.at(1);
        split(p + "mlp.gateup_proj.weight", 0, {mid, mid});
        split(p + "mlp.down_proj.weight", 1, {mid});
        for (const char *name : {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"}) replicate(p + name);
    }
    if (!draftKVOnly) {
        replicate("dspark.norm.weight");
        replicate("dspark.mask_embedding");
        replicate("model.embed_tokens.weight");
    }
    // Context rows are independent. Replicate this dense projection
    // so high-concurrency commits can divide their rows across draft workers.
    auto fc = weight.weight.find("dspark.fc.weight");
    auto contextNorm = weight.weight.find("dspark.hidden_norm.weight");
    if (fc != weight.weight.end() && fc->second.dataType == BFLOAT16 &&
        fc->second.dims == std::vector<int>{embed_dim, (int)draftTargetLayers.size() * embed_dim} &&
        contextNorm != weight.weight.end() && contextNorm->second.dims == std::vector<int>{embed_dim}) {
        if (draftKVOnly) split("dspark.fc.weight", 0, {embed_dim}, true);
        else replicate("dspark.fc.weight");
        replicate("dspark.hidden_norm.weight");
    }
    state->group = FastllmCudaNaiveDraftTPCreate(state->devices);
    AssertInFastLLM(state->group, "Draft TP communicator creation failed.");
    draftTP = std::move(state);
    printf("[Naive draft] backbone_tp=%d backbone_dp=%d attention_tp=%d mode=%s\n",
           dataParallel ? 1 : ranks, dataParallel ? ranks : 1,
           draftTP->attentionDivisor, automatic ? "auto" : "explicit");
    ApplyDraftDevice();
    return true;
}

void NaiveN05FlashModel::PrepareDraftContextTP(DraftContext &context) {
    auto &tp = *draftTP;
    if (tp.dataParallel && context.deviceRank < 0)
        context.deviceRank = tp.nextContextRank++ % tp.devices.size();
    if (!context.workspace) {
        context.workspace = std::make_shared<DraftWorkspace>();
        context.workspace->device = tp.devices.front();
        for (int device : tp.devices) {
            auto rank = std::make_shared<DraftWorkspace>();
            rank->device = device;
            if (!tp.dataParallel && device != tp.devices.front()) rank->tpKV.resize(draftLayers);
            context.workspace->tpRanks.push_back(std::move(rank));
        }
    }
    context.kv.resize(draftLayers);
}

bool NaiveN05FlashModel::AppendDraftContextTP(Data &hidden, int start, DraftContext &context) {
    if (!PrepareDraftTP()) return false;
    auto &tp = *draftTP;
    ApplyDraftDevice();
    PrepareDraftContextTP(context);
    if (!draftContextStorage) {
        draftContextStorage = std::make_shared<DraftWorkspace>();
        draftContextStorage->device = tp.devices.front();
        for (int device : tp.devices) {
            auto rank = std::make_shared<DraftWorkspace>(); rank->device = device;
            rank->contextLayers.resize(draftLayers);
            FastllmCudaSetDevice(device); rank->done = FastllmCudaEventCreate();
            draftContextStorage->tpRanks.push_back(std::move(rank));
        }
        FastllmCudaSetDevice(tp.devices.front());
    }
    auto &ws = *draftContextStorage;
    const int begin = std::max(0, hidden.dims[1] - draftWindow + 1);
    // A peer must finish consuming the root input before its storage can be reused.
    Data *selected = &hidden;
    if (begin || hidden.dataDevice != DataDevice::CUDA ||
        hidden.dataDeviceIds != std::vector<int>{tp.devices.front()} || hidden.multiDeviceData ||
        hidden.strides != std::vector<uint64_t>{(uint64_t)hidden.dims[1] * hidden.dims[2], (uint64_t)hidden.dims[2], 1}) {
        Split(hidden, 1, begin, hidden.dims[1], ws.contextSelected);
        ws.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{tp.devices.front()});
        selected = &ws.contextSelected;
    }
    const int heads = DraftContextQueryHeads() / tp.attentionDivisor, kvHeads = std::max(1, draftKvHeads / tp.attentionDivisor);
    const int rows = selected->dims[1], width = kvHeads * draftHeadDim;
    // Publish the root PTDS writes before another device reads the selected rows.
    if (!tp.dataParallel || context.deviceRank != 0) FastllmCudaSyncCurrentThreadStream();
    tp.Run([&](int rank) {
        if (tp.dataParallel && rank != context.deviceRank) return;
        auto &r = *ws.tpRanks[rank];
        Data *input = selected;
        if (rank) {
            r.contextSelected.dataType = BFLOAT16;
            r.contextSelected.UpdateUnitSize();
            r.contextSelected.Resize(selected->dims);
            r.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{r.device}, false);
            r.contextSelected.Allocate(false);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, r.contextSelected.cudaData,
                tp.devices.front(), selected->cudaData, selected->GetBytes()), "Draft TP context transfer failed.");
            input = &r.contextSelected;
        }
        std::vector<Data> views(draftLayers), parts(draftLayers);
        std::vector<const Data *> raw, norms, weights;
        for (int i = 0; i < draftLayers; ++i) {
            const std::string p = "dspark.layers." + std::to_string(i) + ".self_attn.";
            Data &qkv = *weight[p + DraftKVProjectionName()].multiDeviceDatas.at(r.device);
            views[i].FakeFrom(qkv, (size_t)heads * draftHeadDim * embed_dim * sizeof(uint16_t));
            views[i].Resize({2 * width, embed_dim});
            weights.push_back(&views[i]);
            norms.push_back(weight[p + "k_norm.weight"].multiDeviceDatas.at(r.device));
        }
        // Batch the K/V row views without packing or copying the weights.
        bool batched = FastllmCudaNaiveDraftKVProject(*input, weights, r.contextRaw, r.contextPointers);
        for (int i = 0; i < draftLayers; ++i) {
            if (batched) {
                parts[i].FakeFrom(r.contextRaw, (size_t)i * rows * 2 * width * sizeof(uint16_t));
                parts[i].Resize({1, rows, 2 * width});
                raw.push_back(&parts[i]);
            } else {
                DraftLinear(*input, views[i], r.contextLayers[i].raw);
                raw.push_back(&r.contextLayers[i].raw);
            }
        }
        auto &kv = rank && !tp.dataParallel ? context.workspace->tpRanks[rank]->tpKV : context.kv;
        AssertInFastLLM(FastllmCudaNaiveDraftKV(raw, norms, start + begin, kv, kvHeads,
            draftHeadDim, draftWindow, draftWindow + draftBlock, draftEps, draftTheta), "Draft TP KV append failed.");
        if (!tp.dataParallel) FastllmCudaEventRecordCurrentThread(r.done);
        else if (rank) FastllmCudaSyncCurrentThreadStream();
    });
    if (!tp.dataParallel)
        for (size_t rank = 1; rank < tp.devices.size(); ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    context.committed = start + hidden.dims[1];
    return true;
}

#endif

std::string NaiveN05FlashModel::DraftPDLayout(bool localReplica) const {
    if (!draftEnabled) return {};
    int ranks = 1, divisor = 1;
#ifdef USE_CUDA
    if (draftTP) {
        if (attentionDP > 1) {
            CheckDraftPD(AttentionTPSize() == 1 && draftTP->dataParallel && draftTP->devices == tpDevices,
                "Group PD requires a full draft replica on each attention owner.");
        } else {
            CheckDraftPD(!draftTP->dataParallel, "Draft DP requires group-local PD placement.");
            ranks = draftTP->devices.size(); divisor = draftTP->attentionDivisor;
        }
        // The generic PD placement map is the target's device group. Draft
        // subgroups use its first N ranks, including nonzero physical devices.
        CheckDraftPD(tpDevices.size() >= size_t(ranks) &&
            std::equal(draftTP->devices.begin(), draftTP->devices.end(), tpDevices.begin()),
            "Naive PD draft TP requires a subgroup of target TP.");
    }
#endif
    std::string layout = ":draft-v2";
    if (localReplica) {
        ranks = divisor = 1;
        CheckDraftPD(attentionDP == 1 || bool(draftTP), "Group PD requires initialized draft replicas.");
        auto checkpoint = weight.dicts.find("dspark.model_path");
        layout += ":local:" + (checkpoint == weight.dicts.end() ? std::string() : checkpoint->second);
    }
    for (int value : {draftLayers, draftBlock, draftHeads, draftKvHeads, draftHeadDim,
                      draftWindow, ranks, divisor}) layout += ":" + std::to_string(value);
    return layout;
}

void NaiveN05FlashModel::ExportDraftPDState(PDStateWriter &writer, const DraftContext &context, int length, bool planning) {
#ifdef USE_CUDA
    CheckDraftPD(context.pending.empty() && context.committed == length,
        "Naive PD requires a committed draft prefix.");
    const int ranks = draftTP && !draftTP->dataParallel ? draftTP->devices.size() : 1;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int width = std::max(1, draftKvHeads / divisor) * draftHeadDim;
    const std::vector<int> shape{1, std::min(length, draftWindow - 1), width};
    CheckDraftPD(!draftTP || (context.workspace && context.workspace->tpRanks.size() == draftTP->devices.size()),
        "Naive PD draft rank state is missing.");
    // Finish all owner streams before FastKV snapshots their committed KV.
    if (!planning) {
        if (draftTP) draftTP->Run([](int) { FastllmCudaSyncCurrentThreadStream(); });
        else FastllmCudaSyncCurrentThreadStream();
        writer.metadata.U32(context.committed);
        writer.metadata.U32(draftLayers); writer.metadata.U32(ranks);
    }
    auto tensor = [&](const Data &value) {
        if (planning) writer.PlanTensor(value);
        else writer.Tensor(value);
    };
    for (int rank = 0; rank < ranks; ++rank) {
        const auto &kv = rank ? context.workspace->tpRanks[rank]->tpKV : context.kv;
        CheckDraftPD(kv.size() == size_t(draftLayers), "Naive PD draft layer state is missing.");
        for (const auto &pair : kv) for (const Data *cache : {&pair.first, &pair.second}) {
            CheckDraftPD(cache->dataType == BFLOAT16 && cache->dims == shape,
                "Naive PD draft KV shape mismatch.");
            tensor(*cache);
        }
    }
    // Normalized prefill tails are needed by aligned prefix checkpoints on D.
    // Proposal distributions, RNG and graph workspaces are local to decode.
    if (!planning) writer.metadata.U32(!context.prefixHidden.dims.empty());
    if (!context.prefixHidden.dims.empty()) tensor(context.prefixHidden);
#endif
}

std::shared_ptr<NaiveN05FlashModel::DraftContext> NaiveN05FlashModel::ImportDraftPDState(
        PDStateReader &reader, int length, int device) {
    std::shared_ptr<DraftContext> context;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        context = CreateDraftContext({device});
    }
#ifdef USE_CUDA
    ApplyDraftDevice();
    if (draftTP) PrepareDraftContextTP(*context);
    const auto devices = draftTP && draftTP->dataParallel ? std::vector<int>{draftTP->devices.at(context->deviceRank)} :
        draftTP ? draftTP->devices : std::vector<int>{FastllmCudaGetDevice()};
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int width = std::max(1, draftKvHeads / divisor) * draftHeadDim;
    const std::vector<int> shape{1, std::min(length, draftWindow - 1), width};
    const int committed = reader.metadata.U32();
    const auto layers = reader.metadata.U32(), ranks = reader.metadata.U32();
    CheckDraftPD(committed == length && layers == size_t(draftLayers) && ranks == devices.size(),
        "Naive PD draft prefix or TP size mismatch.");
    if (!draftTP) context->kv.resize(draftLayers);
    for (size_t rank = 0; rank < devices.size(); ++rank) {
        auto &kv = rank ? context->workspace->tpRanks[rank]->tpKV : context->kv;
        FastllmCudaSetDevice(devices[rank]);
        for (auto &pair : kv) for (Data *cache : {&pair.first, &pair.second}) {
            Data value;
            reader.Tensor(value, true);
            CheckDraftPD(value.dataType == BFLOAT16 && value.dims == shape && !value.kvCacheFp8Indexer,
                "Naive PD draft KV shape mismatch.");
            value.ToDevice(CUDA, std::vector<int>{devices[rank]});
            ReserveDraftCache(*cache, devices[rank], shape, draftWindow + draftBlock);
            // Retain warmup/recycled addresses and copy directly from the
            // borrowed transport tensor into its final draft allocation.
            const bool copied = FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                cache->cudaData, value.cudaData, value.GetBytes());
            FastllmCudaSyncCurrentThreadStream();
            CheckDraftPD(copied, "KV restore copy failed.");
        }
    }
    FastllmCudaSetDevice(devices.front());
    if (reader.metadata.U32()) reader.Tensor(context->prefixHidden);
    const auto &tail = context->prefixHidden;
    CheckDraftPD(tail.dims.empty() || (tail.dataType == BFLOAT16 && tail.dims.size() == 3 &&
        tail.dims[0] == 1 && tail.dims[1] > 0 && tail.dims[1] <= std::min(length, draftWindow + GetPageLen() - 1) &&
        tail.dims[2] == embed_dim), "Naive PD draft prefix tail shape mismatch.");
    if (!tail.dims.empty()) context->prefixHidden.ToDevice(CUDA, std::vector<int>{draftTP ? draftTP->devices.front() : devices.front()});
    reader.End();
    context->committed = length;
#endif
    return context;
}

#ifdef USE_CUDA
std::shared_ptr<NaiveN05FlashModel::DraftContext> NaiveN05FlashModel::ReservePDDraft(int length, int device) {
    std::shared_ptr<DraftContext> draft;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        draft = CreateDraftContext({device});
    }
    PrepareDraftContextTP(*draft);
    const std::vector<int> shape{1, std::min(length, draftWindow - 1), draftKvHeads * draftHeadDim};
    for (auto &pair : draft->kv) {
        ReserveDraftCache(pair.first, device, shape, draftWindow + draftBlock);
        ReserveDraftCache(pair.second, device, shape, draftWindow + draftBlock);
    }
    // The normalized prefill features are consumed on the projection device.
    // Keep the optional wire tail within the existing per-request workspace.
    Data &tail = draft->workspace->prefixStorage;
    tail.dataType = BFLOAT16; tail.UpdateUnitSize();
    tail.Resize({1,std::min(length, draftWindow + GetPageLen() - 1),embed_dim});
    tail.ToDevice(CUDA, std::vector<int>{draftTP->devices.front()}, false); tail.Allocate(false);
    draft->prefixHidden.FakeFrom(tail, 0); draft->prefixHidden.Resize(tail.dims);
    draft->committed = length;
    return draft;
}
#endif

std::shared_ptr<NaiveN05FlashModel::DraftContext> NaiveN05FlashModel::CreateDraftContext(const std::vector<int> &devices) {
    // The request lifecycle and caller hold historyMutex. Reuse storage only;
    // reset every logical prefix, proposal and sampling state for a new request.
    auto context = std::make_shared<DraftContext>();
    int owner = -1;
#ifdef USE_CUDA
    if (devices.size() == 1 && draftTP && draftTP->dataParallel) {
        auto it = std::find(draftTP->devices.begin(), draftTP->devices.end(), devices.front());
        if (it != draftTP->devices.end()) owner = int(it - draftTP->devices.begin());
    }
#endif
    context->deviceRank = owner;
    if (idleDraftContexts.empty()) return context;
    auto slot = std::prev(idleDraftContexts.end());
    if (owner >= 0 && (*slot)->deviceRank != owner) {
        auto match = std::find_if(idleDraftContexts.begin(), slot,
            [owner](const auto &idle) { return idle->deviceRank == owner; });
        if (match == slot) return context;
        std::iter_swap(match, slot);
    }
    auto idle = std::move(*slot);
    idleDraftContexts.pop_back();
    context->deviceRank = idle->deviceRank;
    auto &storage = idle->kvBacking.empty() ? idle->kv : idle->kvBacking;
    context->kv.swap(storage);
    idle->kv.clear(); // Drop retired views before reusing their backing storage.
    context->workspace.swap(idle->workspace);
    auto reset = [](auto &kv) {
        for (auto &pair : kv) for (Data *cache : {&pair.first, &pair.second})
            if (cache->dims.size() == 3) cache->Resize({1, 0, cache->dims[2]});
    };
    reset(context->kv);
#ifdef USE_CUDA
    if (context->workspace)
        for (auto &rank : context->workspace->tpRanks) reset(rank->tpKV);
#endif
    return context;
}

bool NaiveN05FlashModel::RunDraftGraph(int anchor, DraftContext &context, Data &output) {
#ifdef USE_CUDA
    if (!GetFastllmEnv().cudaGraph || !GetCudaEmbedding() || GetLowMemMode() ||
        draftBlock < 2 || draftBlock >= 32 || draftHeadDim % 4 || draftHeadDim > 256 ||
        context.kv.size() != (size_t)draftLayers) return false;
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    const int device = FastllmCudaGetDevice();
    Data &embedding = DraftWeight("model.embed_tokens.weight");
    Data &mask = weight["dspark.mask_embedding"];
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 ||
        mask.Count(0) != embed_dim) return false;
    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
    for (auto &pair : context.kv) for (Data *cache : {&pair.first, &pair.second}) {
        if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
            cache->dataDeviceIds != std::vector<int>{device} || cache->expansionDims.empty() ||
            cache->expansionDims[1] < draftWindow + draftBlock) return false;
        inputs.push_back(cache->cudaData);
    }
    if (context.workspace && (context.workspace->device != device || context.workspace->inputs != inputs))
        context.workspace.reset();
    if (!context.workspace) {
        context.workspace = std::make_shared<DraftWorkspace>();
        context.workspace->device = device;
        context.workspace->inputs = std::move(inputs);
        context.workspace->layers.resize(draftLayers);
        context.workspace->contextLayers.resize(draftLayers);
    }
    auto &state = *context.workspace;
    if (state.disabled) return false;
    auto upload = [&](Data &target, DataType type, void *value) {
        if (target.dims.empty()) {
            target.dataType = type;
            target.UpdateUnitSize();
            target.dataDevice = DataDevice::CUDA;
            target.dataDeviceIds = {device};
            target.Resize({1});
            target.Allocate();
        }
        FastllmCudaCopyFromHostToDevice(target.cudaData, value, sizeof(int));
    };
    float token = anchor;
    int live = context.committed + 1;
    upload(state.id, FLOAT32, &token);
    upload(state.live, INT32, &live);
    const bool shortAttention = std::min(context.committed, draftWindow - 1) + draftBlock <= 256;
    auto &graph = state.graphs[shortAttention ? 0 : 1];
    auto body = [&]() {
        FastllmCudaNaiveDraftInput(state.id, embedding, mask, state.live, draftBlock,
                                  state.hidden, state.positions);
        for (int i = 0; i < draftLayers; ++i) {
            const std::string layer = "dspark.layers." + std::to_string(i) + ".";
            auto &b = state.layers[i];
            KimiK3RMSNorm(state.hidden, weight[layer + "input_layernorm.weight"], draftEps, b.normed);
            auto &cache = context.kv[i];
            const bool mergedQKV = weight.weight.count(layer + "self_attn.mergeqkv.weight");
            if (mergedQKV) DraftLinear(b.normed, weight[layer + "self_attn.mergeqkv.weight"], b.qkv);
            if (!mergedQKV || !FastllmCudaNaiveDraftQKV(b.qkv,
                    weight[layer + "self_attn.q_norm.weight"], weight[layer + "self_attn.k_norm.weight"],
                    state.positions, state.live, cache.first, cache.second, b.q,
                    draftHeads, draftKvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
                if (mergedQKV) {
                    const int qw = draftHeads * draftHeadDim, kw = draftKvHeads * draftHeadDim;
                    Split(b.qkv, -1, 0, qw, b.q);
                    Split(b.qkv, -1, qw, qw + kw, b.k);
                    Split(b.qkv, -1, qw + kw, qw + 2 * kw, b.v);
                } else {
                    DraftLinear(b.normed, weight[layer + "self_attn.q_proj.weight"], b.q);
                    DraftLinear(b.normed, weight[layer + "self_attn.k_proj.weight"], b.k);
                    DraftLinear(b.normed, weight[layer + "self_attn.v_proj.weight"], b.v);
                }
                b.q.Reshape({1, draftBlock * draftHeads, draftHeadDim});
                b.k.Reshape({1, draftBlock * draftKvHeads, draftHeadDim});
                KimiK3RMSNorm(b.q, weight[layer + "self_attn.q_norm.weight"], draftEps, b.q);
                KimiK3RMSNorm(b.k, weight[layer + "self_attn.k_norm.weight"], draftEps, b.k);
                b.q.Reshape({1, draftBlock, draftHeads * draftHeadDim});
                b.k.Reshape({1, draftBlock, draftKvHeads * draftHeadDim});
                FastllmCudaNaiveRope(b.q, state.positions, draftHeads, draftHeadDim, draftHeadDim, draftTheta);
                FastllmCudaNaiveRope(b.k, state.positions, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
                FastllmCudaNaiveAppendVerifyCache(cache.first, cache.second, b.k, b.v, state.live, draftWindow);
            }
            FastllmCudaNaiveDraftAttention(b.q, cache.first, cache.second, state.live,
                draftHeads, draftKvHeads, draftHeadDim, draftWindow, shortAttention, b.scores, b.attention);
            DraftLinear(b.attention, weight[layer + "self_attn.o_proj.weight"], b.output);
            AddTo(state.hidden, b.output);
            KimiK3RMSNorm(state.hidden, weight[layer + "post_attention_layernorm.weight"], draftEps, b.normed);
            if (weight.weight.count(layer + "mlp.gateup_proj.weight")) {
                DraftLinear(b.normed, weight[layer + "mlp.gateup_proj.weight"], b.gateUp);
                FastllmCudaNaiveDraftSwiGLU(b.gateUp, b.gate);
            } else {
                DraftLinear(b.normed, weight[layer + "mlp.gate_proj.weight"], b.gate);
                DraftLinear(b.normed, weight[layer + "mlp.up_proj.weight"], b.up);
                Silu(b.gate, b.gate);
                MulTo(b.gate, b.up);
            }
            DraftLinear(b.gate, weight[layer + "mlp.down_proj.weight"], b.output);
            AddTo(state.hidden, b.output);
        }
        KimiK3RMSNorm(state.hidden, weight["dspark.norm.weight"], draftEps, state.normalized);
    };
    if (graph.exec) {
        AssertInFastLLM(FastllmCudaGraphLaunch(graph.exec), "Naive draft graph launch failed.");
    } else {
        body();
        // Warmup resolves weights, cuBLAS and all persistent workspaces. Capture
        // does not commit any KV metadata or advance the proposal RNG.
        if (!graph.Capture(body, state.disabled)) {
            fprintf(stderr, "[Fastllm] Naive draft graph capture failed; using eager.\n");
            return false;
        }
    }
    Copy(state.normalized, output);
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RunDraftProposalGraph(int anchor, const Data &baseLogits,
        DraftContext &context, std::vector<int> &proposed) {
#ifdef USE_CUDA
    if (!GetFastllmEnv().cudaGraph || !context.workspace ||
        baseLogits.dataType != BFLOAT16 || baseLogits.dims.size() != 3 ||
        baseLogits.dims[0] != 1 || baseLogits.dims[1] != draftBlock ||
        draftBlock < 2 || draftBlock >= 32 || baseLogits.dims[2] >= (1 << 24)) return false;
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    auto &state = *context.workspace;
    const int device = FastllmCudaGetDevice(), vocab = baseLogits.dims.back();
    if (state.device != device || baseLogits.dataDevice != DataDevice::CUDA ||
        baseLogits.dataDeviceIds != std::vector<int>{device}) return false;
    Data &w1 = weight["dspark.markov_head.markov_w1.weight"];
    Data &w2 = weight["dspark.markov_head.markov_w2.weight"];
    if (w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.dims.size() != 2 ||
        w1.dims[0] != vocab || w2.dims != w1.dims || anchor < 0 || anchor >= vocab) return false;
    w1.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    w2.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    auto &p = state.proposal;
    std::vector<void *> weights{w1.cudaData, w2.cudaData};
    if (p.weights != weights || p.logits.dims != baseLogits.dims) {
        p.graph.Clear();
        p.weights = std::move(weights);
        p.disabled = false;
    }
    if (p.disabled) return false;
    // The graph owns stable buffers, not RunDraftHead's temporary return value.
    Copy(baseLogits, p.logits);
    p.ids.dataType = FLOAT32;
    p.ids.Resize({draftBlock + 1});
    p.ids.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    p.ids.Allocate(false);
    float token = anchor;
    FastllmCudaCopyFromHostToDevice(p.ids.cudaData, &token, sizeof(token));
    auto body = [&]() {
        for (int step = 0; step < draftBlock; ++step) {
            FastllmCudaNaiveDraftEmbedding(p.ids, step, w1, p.latent);
            DraftLinear(p.latent, w2, p.bias);
            FastllmCudaNaiveDraftArgmax(p.logits, p.bias, step, p.partial, p.ids);
        }
    };
    if (p.graph.exec) {
        AssertInFastLLM(FastllmCudaGraphLaunch(p.graph.exec), "Naive draft proposal graph launch failed.");
    } else {
        body();
        if (!p.graph.Capture(body, p.disabled)) return false;
    }
    std::vector<float> ids(draftBlock + 1);
    FastllmCudaCopyFromDeviceToHost(ids.data(), p.ids.cudaData, ids.size() * sizeof(float));
    proposed.assign(ids.begin() + 1, ids.end());
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RunDraftSamplingStep(int previous, int step, const Data &baseLogits,
        DraftContext &context, const GenerationConfig &config, int &token) {
#ifdef USE_CUDA
    if (!GetFastllmEnv().cudaGraph || !context.workspace || baseLogits.dataType != BFLOAT16 ||
        baseLogits.dims.size() != 3 || baseLogits.dims[0] != 1 ||
        baseLogits.dims[1] <= 0 || baseLogits.dims[1] > draftBlock ||
        step < 0 || step >= baseLogits.dims[1] || baseLogits.dims[2] >= (1 << 24))
        return false;
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    const int device = FastllmCudaGetDevice(), vocab = baseLogits.dims.back();
    const int rows = baseLogits.dims[1], words = (vocab + 31) / 32;
    const auto &devices = baseLogits.dataDeviceIds;
    if (context.workspace->device != device || baseLogits.dataDevice != DataDevice::CUDA ||
        (devices.size() != 1 || devices.front() != device) || previous < 0 || previous >= vocab ||
        context.proposalProbs.dims != std::vector<int>({rows, vocab}) ||
        context.proposalProbs.dataDevice != DataDevice::CUDA ||
        context.proposalProbs.dataDeviceIds != devices)
        return false;
    Data &w1 = weight["dspark.markov_head.markov_w1.weight"];
    Data &w2 = weight["dspark.markov_head.markov_w2.weight"];
    if (w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.dims.size() != 2 ||
        w1.dims[0] != vocab || w2.dims != w1.dims) return false;
    w1.ToDevice(DataDevice::CUDA, devices);
    w2.ToDevice(DataDevice::CUDA, devices);
    auto &p = context.workspace->sampling;
    const std::array<void *, 3> inputs{w1.cudaData, w2.cudaData, context.proposalProbs.cudaData};
    if (p.inputs != inputs || p.logits.dims != baseLogits.dims ||
        p.temperature != config.temperature || p.topK != config.top_k || p.topP != config.top_p) {
        p.Clear(); p.disabled = false;
        p.inputs = inputs;
        p.temperature = config.temperature; p.topK = config.top_k; p.topP = config.top_p;
    }
    if (p.disabled || (step > 0 && !p.sampler)) return false;
    if (step == 0) {
        Copy(baseLogits, p.logits);
        for (auto item : {std::make_pair(&p.ids, rows + 1), std::make_pair(&p.mask, words)}) {
            item.first->dataType = item.first == &p.ids ? FLOAT32 : INT32;
            item.first->Resize({item.second});
            item.first->ToDevice(DataDevice::CUDA, devices, false);
            item.first->Allocate(false);
        }
        p.graphs.resize(rows);
        if (!p.sampler)
            p.sampler = FastllmCudaMtpCreateDraftSampler(vocab, config.temperature, config.top_k, config.top_p);
        if (!p.sampler) { p.disabled = true; return false; }
    }
    if (config.tool_call_token_mask) {
        AssertInFastLLM(config.tool_call_token_mask->size() >= (size_t)words, "Invalid draft sampling mask.");
        FastllmCudaCopyFromHostToDevice(p.mask.cudaData, const_cast<uint32_t *>(config.tool_call_token_mask->data()), words * sizeof(uint32_t));
    } else if (step == 0) {
        std::vector<uint32_t> mask(words, ~uint32_t(0));
        FastllmCudaCopyFromHostToDevice(p.mask.cudaData, mask.data(), words * sizeof(uint32_t));
    }
    float id = previous;
    FastllmCudaCopyFromHostToDevice((float *)p.ids.cudaData + step, &id, sizeof(id));
    AssertInFastLLM(FastllmCudaMtpDraftSamplerSetSeed(p.sampler), "Cannot update draft sampling seed.");
    auto body = [&] {
        FastllmCudaNaiveDraftEmbedding(p.ids, step, w1, p.latent);
        DraftLinear(p.latent, w2, p.bias);
        FastllmCudaNaiveDraftSamplingLogits(p.logits, p.bias, p.mask, step, p.row);
        AssertInFastLLM(FastllmCudaMtpDraftSamplerRun(p.sampler, (float *)p.row.cudaData,
            (float *)context.proposalProbs.cudaData + (size_t)step * vocab,
            (float *)p.ids.cudaData + step + 1), "Draft graph sampling failed.");
    };
    auto &graph = p.graphs[step];
    if (graph.exec) {
        AssertInFastLLM(FastllmCudaGraphLaunch(graph.exec), "Draft sampling graph replay failed.");
    } else {
        body();
        if (!graph.Capture(body, p.disabled)) return false;
    }
    FastllmCudaCopyFromDeviceToHost(&id, (float *)p.ids.cudaData + step + 1, sizeof(id));
    token = (int)id;
    AssertInFastLLM(token >= 0 && token < vocab, "Draft sampling graph produced an invalid token.");
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::ApplyDraftDevice() {
    if (tpDevices.empty()) ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    else {
        ApplyDeviceMap({{"cuda:" + std::to_string(tpDevices.front()), 1}}, 1, 1);
#ifdef USE_CUDA
        // The dispatcher does not select the CUDA context. Events and raw
        // stream operations must use the draft owner's device as well.
        FastllmCudaSetDevice(tpDevices.front());
#endif
    }
}

Data &NaiveN05FlashModel::DraftWeight(const std::string &name) {
    Data &data = weight[name];
    return tpDevices.empty() || !data.multiDeviceData ? data :
        *data.multiDeviceDatas.at(tpDevices.front());
}

Data NaiveN05FlashModel::RunDraftTarget(const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
        TargetCapture &capture, const TargetBatch *batch) {
    return tpDevices.empty() ? RunTarget(inputIds, positions, kv, config, &capture, -1, nullptr, nullptr, batch) :
        ForwardTensorParallel(inputIds, positions, kv, config, &capture, capture.selection, batch);
}

void NaiveN05FlashModel::CommitTargetCache(std::vector<std::pair<Data, Data>> &kv,
        int past, int count, bool ringVerify) {
#ifdef USE_CUDA
    if (ringVerify && attentionDP > 1 && CommitBoundVerifyCaches({&kv}, {past}, {count})) return;
    auto commit = [&](int device) {
        if (device >= 0 && !kv.front().first.multiDeviceDatas.count(device)) return;
        for (int layer = 0; layer < block_cnt; ++layer) {
            const int length = (slidingLayers[layer] ? std::min(past, window - 1) : past) + count;
            auto &pair = kv[layer];
            Data &key = device < 0 ? pair.first : *pair.first.multiDeviceDatas.at(device);
            Data &value = device < 0 ? pair.second : *pair.second.multiDeviceDatas.at(device);
            if (slidingLayers[layer] && ringVerify) {
                FastllmCudaSetDevice(key.dataDeviceIds.at(0));
                FastllmCudaNaiveCommitSwaVerify(key, value, past, count);
                continue;
            }
            key.Resize({1, length, key.dims[2]});
            value.Resize({1, length, value.dims[2]});
            for (Data *cache : {&key, &value})
                if (cache->isPagedKVCache) cache->lastPageLen = length ? (length - 1) % cache->pageLen + 1 : 0;
            if (slidingLayers[layer]) {
                FastllmCudaSetDevice(key.dataDeviceIds.at(0));
                FastllmCudaNaiveTrimCache(key, value, window - 1);
                if (device >= 0 && tpSwaRingCache)
                    FastllmCudaNaiveSwaCacheLayout(key, value, past + count, true);
            }
        }
    };
    if (tpDevices.empty()) commit(-1);
    else {
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            const int device = tpDevices[rank];
            FastllmCudaSetDevice(device);
            commit(device);
            ForceDeviceSync();
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        for (int layer = 0; layer < block_cnt; ++layer) {
            auto &pair = kv[layer];
            const int length = (slidingLayers[layer] ? std::min(past, window - 1) : past) + count;
            int kept = slidingLayers[layer] ? std::min(length, window - 1) : length;
            pair.first.Resize({1, kept, pair.first.dims[2]});
            pair.second.Resize({1, kept, pair.second.dims[2]});
        }
    }
#endif
}

void NaiveN05FlashModel::CommitTargetCaches(const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
        const std::vector<int> &past, const std::vector<int> &counts, bool ringVerify, const std::vector<char> *written) {
    AssertInFastLLM(caches.size() == past.size() && caches.size() == counts.size(), "Invalid target commit batch.");
#ifdef USE_CUDA
    if (!ringVerify) {
        for (size_t i = 0; i < caches.size(); ++i) CommitTargetCache(*caches[i], past[i], counts[i], false);
        return;
    }
    if (CommitBoundVerifyCaches(caches, past, counts, written)) return;
    // DP owns complete requests. Partition once, before entering the workers,
    // so an EP8 worker visits only its local KV. TP peers share the root
    // descriptors; the first owner publishes those once.
    std::map<int, std::vector<size_t>> deviceRows;
    for (size_t i = 0; i < caches.size(); ++i) {
        if (tpDevices.empty()) deviceRows[-1].push_back(i);
        else for (int device : caches[i]->front().first.dataDeviceIds) deviceRows[device].push_back(i);
    }
    auto commit = [&](int device) {
        std::map<int, std::vector<FastllmNaiveCacheCommit>> updates;
        auto group = deviceRows.find(device);
        if (group == deviceRows.end()) return;
        for (size_t i : group->second) for (int layer = 0; layer < block_cnt; ++layer) {
            auto &pair = caches[i]->at(layer);
            Data &key = device < 0 ? pair.first : *pair.first.multiDeviceDatas.at(device);
            Data &value = device < 0 ? pair.second : *pair.second.multiDeviceDatas.at(device);
            if (slidingLayers[layer]) updates[key.dataDeviceIds.at(0)].push_back({&key, &value, past[i], counts[i]});
            else for (Data *cache : {&key, &value}) {
                const int length = past[i] + counts[i];
                FastllmNaiveResizeKVLength(*cache, length, cache->dims[2]);
                if (cache->isPagedKVCache) cache->lastPageLen = length ? (length - 1) % cache->pageLen + 1 : 0;
            }
        }
        for (auto &entry : updates) { FastllmCudaSetDevice(entry.first); FastllmCudaNaiveBatchCommitSwaVerify(entry.second); }
    };
    if (tpDevices.empty()) { ApplyDraftDevice(); commit(-1); }
    else {
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]);
            commit(tpDevices[rank]);
            FastllmCudaSyncCurrentThreadStream();
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        // Local CUDA work is complete. Partition the independent root metadata
        // across the same workers, retaining Resize for nonstandard layouts.
        auto commitRoots = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) for (int layer = 0; layer < block_cnt; ++layer) {
                int kept = slidingLayers[layer] ? std::min(past[i] + counts[i], window - 1) : past[i] + counts[i];
                auto &pair = caches[i]->at(layer);
                FastllmNaiveResizeKVLength(pair.first, kept, pair.first.dims[2]);
                FastllmNaiveResizeKVLength(pair.second, kept, pair.second.dims[2]);
            }
        };
        if (caches.size() < 32) commitRoots(0, caches.size());
        else {
            tpWorkers.Run(tpDevices, [&](int rank) {
                commitRoots(caches.size() * rank / tpDevices.size(),
                    caches.size() * (rank + 1) / tpDevices.size());
            }, errors);
            for (auto error : errors) if (error) std::rethrow_exception(error);
        }
    }
    RememberVerifyCommit(caches, past, counts);
#endif
}

std::set<std::string> NaiveN05FlashModel::PDDraftContextWeightNames() const {
    if (!draftKVOnly) return {};
    std::set<std::string> names{"fc.weight", "hidden_norm.weight"};
    for (int layer = 0; layer < draftLayers; ++layer) {
        const auto prefix = "layers." + std::to_string(layer) + ".self_attn.";
        for (const char *part : {"k_proj.weight", "v_proj.weight", "k_norm.weight"})
            names.insert(prefix + part);
    }
    return names;
}

void NaiveN05FlashModel::InitDraft() {
    draftEnabled = weight.dicts.count("dspark.model_path") != 0;
    draftKVOnly = draftEnabled && pdPrefillOnly;
    if (!draftEnabled) return;
    auto number = [&](const std::string &name) {
        return std::stof(weight.dicts.at("dspark." + name));
    };
    draftLayers = number("num_hidden_layers");
    draftBlock = number("block_size");
    draftHeads = number("num_attention_heads");
    draftKvHeads = number("num_key_value_heads");
    draftHeadDim = number("head_dim");
    draftWindow = number("sliding_window");
    draftEps = number("rms_norm_eps");
    draftTheta = number("rope_parameters.rope_theta");
    std::string error;
    auto layers = json11::Json::parse(weight.dicts.at("dspark.dflash_config.target_layer_ids"), error);
    AssertInFastLLM(error.empty(), "Invalid DSpark target_layer_ids.");
    draftTargetLayers.clear();
    for (auto &layer : layers.array_items()) draftTargetLayers.push_back(layer.int_value());
    AssertInFastLLM(draftLayers > 0 && draftBlock > 1 && draftWindow > 1 && draftKvHeads > 0 &&
                    draftHeads > 0 && draftHeadDim > 0 && draftHeadDim <= 256 &&
                    draftHeads % draftKvHeads == 0 && number("hidden_size") == embed_dim &&
                    number("num_target_layers") == block_cnt &&
                    number("vocab_size") == std::stof(weight.dicts.at("vocab_size")) &&
                    !draftTargetLayers.empty() &&
                    weight.dicts.at("dspark.markov_head_type") == "vanilla" &&
                    weight.dicts.at("dspark.dflash_config.use_mask_embedding") == "true",
                    "Unsupported Naive DSpark configuration.");
    auto layerTypes = json11::Json::parse(weight.dicts.at("dspark.layer_types"), error);
    AssertInFastLLM(error.empty() && (int)layerTypes.array_items().size() == draftLayers &&
                    weight.dicts["dspark.attention_bias"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv_inject"] != "true" &&
                    weight.dicts["dspark.dflash_config.use_target_kv_fuse"] != "true",
                    "Unsupported Naive DSpark attention/context mode.");
    for (const auto &type : layerTypes.array_items())
        AssertInFastLLM(type.string_value() == "sliding_attention", "Naive DSpark expects sliding draft layers.");
    for (int layer : draftTargetLayers)
        AssertInFastLLM(layer >= 0 && layer < block_cnt, "DSpark target layer out of range.");
    draftTokens = draftBlock;
    if (const char *value = std::getenv("FASTLLM_DSPARK_TOKENS")) {
        int requested = std::stoi(value);
        AssertInFastLLM(requested > 0 && requested <= draftBlock,
                        "Naive DSpark draft_tokens must be in [1, block_size].");
        draftTokens = requested;
    }
    if (const char *value = std::getenv("FASTLLM_DSPARK_CONFIDENCE_THRESHOLD"))
        draftConfidenceThreshold = std::stof(value);
    AssertInFastLLM(draftConfidenceThreshold >= 0 && draftConfidenceThreshold <= 1,
                    "Invalid DSpark confidence threshold.");
    weight.embeddingNames.insert("dspark.markov_head.markov_w1.weight");
    for (auto name : {"dspark.fc.weight", "dspark.markov_head.markov_w2.weight",
                      "dspark.confidence_head.proj.weight", "dspark.layers.*.self_attn.*_proj.weight",
                      "dspark.layers.*.mlp.*_proj.weight"}) weight.linearNames.insert(name);
    // Use the loader's normal merge lifecycle: each merged weight owns its
    // storage, and the original entries are erased before inference begins.
    for (int i = 0; i < draftLayers; ++i) {
        const std::string layer = "dspark.layers." + std::to_string(i) + ".";
        const std::string attn = layer + "self_attn.", mlp = layer + "mlp.";
        if (draftKVOnly) {
            weightMergeRules.push_back(WeightMergeRule({WeightMergeRuleSingle(
                {attn + "k_proj.weight", attn + "v_proj.weight"}, attn + DraftKVProjectionName(), "linear")}));
            continue;
        }
        weightMergeRules.push_back(WeightMergeRule({WeightMergeRuleSingle(
            {attn + "q_proj.weight", attn + "k_proj.weight", attn + "v_proj.weight"},
            attn + "mergeqkv.weight", "linear")}));
        weightMergeRules.push_back(WeightMergeRule({WeightMergeRuleSingle(
            {mlp + "gate_proj.weight", mlp + "up_proj.weight"},
            mlp + "gateup_proj.weight", "linear")}));
    }
}

bool NaiveN05FlashModel::AppendDraftContextFused(Data &hidden, int start, DraftContext &context) {
#ifdef USE_CUDA
    const std::vector<int> devices{FastllmCudaGetDevice()};
    // Avoid a prefill-sized intermediate and preserve the original low-memory path.
    if (GetLowMemMode() || draftLayers <= 0 || draftKvHeads <= 0 ||
        draftHeadDim <= 0 || draftHeadDim > 256 || draftHeadDim % 2 ||
        hidden.dims.size() != 3 || hidden.dims[0] != 1 || hidden.dims[1] <= 0 ||
        (int64_t)hidden.dims[1] > (int64_t)draftBlock + 1 || hidden.dims[1] >= draftWindow ||
        hidden.dataType != BFLOAT16 || hidden.dataDevice != DataDevice::CUDA ||
        !hidden.cudaData || hidden.multiDeviceData ||
        hidden.dataDeviceIds != devices ||
        hidden.strides.size() != 3 || hidden.strides[2] != 1 ||
        hidden.strides[1] != hidden.dims[2] ||
        hidden.strides[0] != (uint64_t)hidden.dims[1] * hidden.dims[2]) return false;
    const int64_t width64 = (int64_t)draftKvHeads * draftHeadDim;
    if (width64 > INT_MAX / 2 / draftLayers || hidden.dims[2] <= 0 ||
        (int64_t)draftWindow + draftBlock > INT_MAX) return false;
    const int width = (int)width64;
    std::vector<Data> local(context.workspace ? 0 : draftLayers);
    std::vector<const Data *> raw, norms;
    // K/V are row views into the loader-owned QKV matrix. Never pack a second
    // copy of the weights, even when all layers share the context input.
    for (int i = 0; i < draftLayers; ++i) {
        const auto name = "dspark.layers." + std::to_string(i) + ".self_attn.";
        auto merged = weight.weight.find(name + DraftKVProjectionName());
        if (merged == weight.weight.end()) return false;
        const Data &qkv = merged->second;
        if (qkv.dataType != BFLOAT16 || qkv.multiDeviceData ||
            qkv.dims != std::vector<int>{(DraftContextQueryHeads() + 2 * draftKvHeads) * draftHeadDim, hidden.dims[2]} ||
            qkv.strides != std::vector<uint64_t>{(uint64_t)hidden.dims[2], 1} ||
            qkv.dataDevice != DataDevice::CUDA || qkv.dataDeviceIds != devices || !qkv.cudaData)
            return false;
        Data &norm = weight[name + "k_norm.weight"];
        if (norm.dataType != FLOAT32 || norm.multiDeviceData ||
            norm.dims != std::vector<int>{draftHeadDim} || norm.strides != std::vector<uint64_t>{1} ||
            norm.dataDevice != DataDevice::CUDA || norm.dataDeviceIds != devices || !norm.cudaData)
            return false;
        norms.push_back(&norm);
    }
    for (int i = 0; i < draftLayers; ++i) {
        Data &qkv = weight["dspark.layers." + std::to_string(i) + ".self_attn." + DraftKVProjectionName()];
        Data view;
        view.FakeFrom(qkv, (size_t)DraftContextQueryHeads() * draftHeadDim * hidden.dims[2] * sizeof(uint16_t));
        view.Resize({2 * width, hidden.dims[2]});
        Data &output = context.workspace ? context.workspace->contextLayers[i].raw : local[i];
        DraftLinear(hidden, view, output);
        raw.push_back(&output);
    }
    if (!FastllmCudaNaiveDraftKV(raw, norms, start, context.kv,
            draftKvHeads, draftHeadDim, draftWindow,
            draftWindow + draftBlock, draftEps, draftTheta)) return false;
    context.committed = start + hidden.dims[1];
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::AppendDraftContext(Data &hidden, int start, DraftContext &context) {
#ifdef USE_CUDA
    ApplyDraftDevice();
    if (AppendDraftContextTP(hidden, start, context)) return;
    if (AppendDraftContextFused(hidden, start, context)) return;
    int length = hidden.dims[1];
    // Only the last window - 1 context positions can be visible to the next block.
    int begin = std::max(0, length - draftWindow + 1);
    // Retain only small decode/verification workspaces, never a prefill-sized
    // hidden feature matrix. The request already owns their graph lifetime.
    auto *buffers = context.workspace && length <= draftBlock + 1
        ? context.workspace.get() : nullptr;
    Data localSelected;
    Data &selected = buffers ? buffers->contextSelected : localSelected;
    Split(hidden, 1, begin, length, selected);
    length -= begin;
    std::vector<float> positions(length);
    for (int i = 0; i < length; ++i) positions[i] = start + begin + i;
    Data pos(FLOAT32, {1, length}, positions);
    context.kv.resize(draftLayers);
    for (int i = 0; i < draftLayers; ++i) {
        std::string layer = "dspark.layers." + std::to_string(i) + ".self_attn.";
        Data localKey, localValue;
        Data &key = buffers ? buffers->contextLayers[i].key : localKey;
        Data &value = buffers ? buffers->contextLayers[i].value : localValue;
        auto merged = weight.weight.find(layer + DraftKVProjectionName());
        if (merged != weight.weight.end()) {
            Data &qkvWeight = merged->second;
            qkvWeight.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()});
            Data keyWeight, valueWeight;
            const int qw = DraftContextQueryHeads() * draftHeadDim, kw = draftKvHeads * draftHeadDim;
            keyWeight.FakeFrom(qkvWeight, (size_t)qw * embed_dim * sizeof(uint16_t));
            valueWeight.FakeFrom(qkvWeight, (size_t)(qw + kw) * embed_dim * sizeof(uint16_t));
            keyWeight.Resize({kw, embed_dim});
            valueWeight.Resize({kw, embed_dim});
            DraftLinear(selected, keyWeight, key);
            DraftLinear(selected, valueWeight, value);
        } else {
            DraftLinear(selected, weight[layer + "k_proj.weight"], key);
            DraftLinear(selected, weight[layer + "v_proj.weight"], value);
        }
        key.Reshape({1, length * draftKvHeads, draftHeadDim});
        KimiK3RMSNorm(key, weight[layer + "k_norm.weight"], draftEps, key);
        key.Reshape({1, length, draftKvHeads * draftHeadDim});
        pos.ToDevice(key.dataDevice, key.dataDeviceIds);
        FastllmCudaNaiveRope(key, pos, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
        // Draft KV is bounded per request and retains its existing allocation
        // path; the target pool reserves only target pages and SWA slots.
        AppendCache(context.kv[i].first, key, BFLOAT16, draftWindow + draftBlock, false);
        AppendCache(context.kv[i].second, value, BFLOAT16, draftWindow + draftBlock, false);
        // Retain bounded draft KV storage between proposal/commit rounds.
        // Copying the suffix discards capacity and reallocates ten buffers
        // on every round once the draft window is full.
        FastllmCudaNaiveTrimCache(context.kv[i].first, context.kv[i].second, draftWindow - 1);
    }
    context.committed = start + begin + length;
#endif
}

void NaiveN05FlashModel::RestoreDraftPrefix(const PrefixSnapshot &snapshot, int length, DraftContext &context) {
#ifdef USE_CUDA
    ApplyDraftDevice();
    const int rows = std::min(length, draftWindow - 1);
    Data &shard = snapshot.pool->DraftShard(snapshot.slot);
    Data local, hidden;
    Data &storage = context.workspace ? context.workspace->prefixStorage : local;
    storage.dataType = BFLOAT16; storage.UpdateUnitSize();
    storage.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
    storage.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false); storage.Allocate(false);
    void *src = (uint8_t *)shard.cudaData + snapshot.pool->DraftOffset(snapshot.slot);
    const size_t bytes = (size_t)rows * embed_dim * sizeof(uint16_t);
    if (shard.dataDeviceIds.front() == storage.dataDeviceIds.front()) {
        AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(storage.cudaData, src, bytes),
                       "Draft prefix restore failed.");
    } else {
        FastllmCudaMemcpyBetweenDevices(storage.dataDeviceIds.front(), storage.cudaData,
            shard.dataDeviceIds.front(), src, bytes);
    }
    hidden.FakeFrom(storage, 0); hidden.Resize({1, rows, embed_dim});
    AppendDraftContext(hidden, length - rows, context);
    if (context.workspace) {
        context.prefixHidden.FreeSpace();
        context.prefixHidden.FakeFrom(storage, 0); context.prefixHidden.Resize(hidden.dims);
    } else Copy(hidden, context.prefixHidden);
    FastllmCudaSyncCurrentThreadStream();
#endif
}

void NaiveN05FlashModel::ProjectDraftContext(Data &combined, Data &projected, Data &hidden) {
    Data &fc = weight["dspark.fc.weight"], &norm = weight["dspark.hidden_norm.weight"];
#ifdef USE_CUDA
    const int rows = combined.dims[1], width = combined.dims[2];
    const int ranks = draftTP ? draftTP->devices.size() : 1;
    if (draftTP && draftContextStorage && fc.IsTensorParallelReplicated() &&
        rows >= ranks * 16 && combined.dataType == BFLOAT16 &&
        combined.dataDevice == DataDevice::CUDA && !combined.multiDeviceData) {
        auto &ws = *draftContextStorage;
        const int root = draftTP->devices.front();
        hidden.dataType = BFLOAT16; hidden.UpdateUnitSize(); hidden.Resize({1, rows, embed_dim});
        hidden.ToDevice(DataDevice::CUDA, {root}, false); hidden.Allocate(false);
        // Publish the large context input once before peer streams read their rows.
        // This path can move from the warmup thread to the serving thread.
        FastllmCudaSyncCurrentThreadStream();
        draftTP->Run([&](int rank) {
            auto &r = *ws.tpRanks[rank];
            const int first = int(int64_t(rows) * rank / ranks);
            const int count = int(int64_t(rows) * (rank + 1) / ranks) - first;
            Data view; view.FakeFrom(combined, size_t(first) * width * sizeof(uint16_t));
            view.Resize({1, count, width});
            Data *input = &view;
            if (rank) {
                auto &local = r.combined;
                local.dataType = BFLOAT16; local.UpdateUnitSize(); local.Resize(view.dims);
                local.ToDevice(DataDevice::CUDA, {r.device}, false); local.Allocate(false);
                AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, local.cudaData,
                    root, view.cudaData, view.GetBytes()), "Draft context row transfer failed.");
                input = &local;
            }
            DraftLinear(*input, *fc.multiDeviceDatas.at(r.device), r.projected);
            KimiK3RMSNorm(r.projected, *norm.multiDeviceDatas.at(r.device), draftEps, r.committedHidden);
            void *destination = (uint16_t *)hidden.cudaData + size_t(first) * embed_dim;
            bool copied = rank ? FastllmCudaMemcpyPeerAsyncCurrentThread(root, destination,
                    r.device, r.committedHidden.cudaData, r.committedHidden.GetBytes()) :
                FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(destination,
                    r.committedHidden.cudaData, r.committedHidden.GetBytes());
            AssertInFastLLM(copied, "Draft context result transfer failed.");
            FastllmCudaEventRecordCurrentThread(r.done);
        });
        for (int rank = 1; rank < ranks; ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
        return;
    }
    const int root = draftTP ? draftTP->devices.front() : FastllmCudaGetDevice();
    Data &localFC = fc.IsTensorParallelReplicated() ? *fc.multiDeviceDatas.at(root) : fc;
    Data &localNorm = norm.IsTensorParallelReplicated() ? *norm.multiDeviceDatas.at(root) : norm;
    DraftLinear(combined, localFC, projected);
    KimiK3RMSNorm(projected, localNorm, draftEps, hidden);
#else
    DraftLinear(combined, fc, projected);
    KimiK3RMSNorm(projected, norm, draftEps, hidden);
#endif
}

void NaiveN05FlashModel::CommitDraftContext(TargetCapture &capture, int tokens,
        DraftContext &context, std::vector<std::pair<Data, Data>> &kv) {
    ApplyDraftDevice();
    Data localCombined, localProjected, localHidden;
#ifdef USE_CUDA
    auto *buffers = draftContextStorage ? draftContextStorage.get() :
        context.workspace && tokens <= draftBlock + 1 ? context.workspace.get() : nullptr;
    Data &combined = buffers ? buffers->combined : localCombined;
    Data &projected = buffers ? buffers->projected : localProjected;
    Data &hidden = buffers ? buffers->committedHidden : localHidden;
    // Only the sliding draft window and the preceding aligned checkpoint can
    // survive a prefill. Discard earlier rows before concatenation/projection.
    const int firstRow = capture.verifying ? 0 : std::max(0, tokens - draftWindow - GetPageLen() + 1);
    const int rowsToCommit = tokens - firstRow, oldCommitted = context.committed;
    std::vector<Data> views(draftTargetLayers.size());
    std::vector<const Data *> inputs;
    for (size_t i = 0; i < draftTargetLayers.size(); ++i) {
        const Data &source = capture.hidden.at(draftTargetLayers[i]);
        if (firstRow) {
            views[i].FakeFrom(source, (size_t)firstRow * embed_dim * source.unitSize);
            views[i].Resize({1, rowsToCommit, embed_dim}); inputs.push_back(&views[i]);
        } else inputs.push_back(&source);
    }
    bool joinedOnDevice = FastllmCudaNaiveDraftConcat(inputs, rowsToCommit, combined);
#else
    Data &combined = localCombined, &projected = localProjected, &hidden = localHidden;
    const int firstRow = 0, oldCommitted = context.committed;
    bool joinedOnDevice = false;
#endif
    if (!joinedOnDevice) {
        bool first = true;
        for (int layer : draftTargetLayers) {
            Data selected, joined;
            Split(capture.hidden.at(layer), 1, firstRow, tokens, selected);
            if (first) { Copy(selected, combined); first = false; }
            else { Cat(combined, selected, -1, joined); Copy(joined, combined); }
        }
    }
#ifdef USE_CUDA
    // Place the retained projections before their first use on the root device.
    if (draftKVOnly && !draftTP) PrepareDraftTP();
#endif
    ProjectDraftContext(combined, projected, hidden);
    AppendDraftContext(hidden, oldCommitted + firstRow, context);
#ifdef USE_CUDA
    if (!capture.verifying && tokens > 1 && saveHistoryChat && prefixPool && prefixPool->draftSlotBytes) {
        // Keep enough normalized prefill input for an aligned checkpoint's
        // complete draft window, even when the final chunk is very short.
        const int keep = draftWindow + GetPageLen() - 1;
        Data joined;
        if (hidden.dims[1] < keep && !context.prefixHidden.dims.empty())
            Cat(context.prefixHidden, hidden, 1, joined);
        else { joined.Resize(hidden.dims); joined.FakeFrom(hidden, 0); }
        const int rows = std::min(keep, joined.dims[1]);
        if (context.workspace) {
            auto &storage = context.workspace->prefixStorage;
            storage.dataType = BFLOAT16; storage.UpdateUnitSize(); storage.Resize({1, keep, embed_dim});
            storage.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false); storage.Allocate(false);
            context.prefixHidden.FreeSpace();
            context.prefixHidden.FakeFrom(storage, 0); context.prefixHidden.Resize({1, rows, embed_dim});
            FastllmCudaCopyFromDeviceToDevice(context.prefixHidden.cudaData,
                (uint16_t *)joined.cudaData + (size_t)(joined.dims[1] - rows) * embed_dim,
                (size_t)rows * embed_dim * sizeof(uint16_t));
        } else {
            Data tail;
            Split(joined, 1, joined.dims[1] - rows, joined.dims[1], tail);
            context.prefixHidden.CopyFrom(tail);
        }
        const int start = context.committed - rows;
        if (!capture.prefixes.empty()) for (const auto &snapshot : capture.prefixes.front()) {
            const int count = std::min(snapshot->length, draftWindow - 1);
            const int begin = snapshot->length - count - start;
            if (begin < 0 || begin + count > rows) continue;
            Data &storage = prefixPool->DraftShard(snapshot->slot);
            void *dst = (uint8_t *)storage.cudaData + prefixPool->DraftOffset(snapshot->slot);
            void *src = (uint16_t *)context.prefixHidden.cudaData + (size_t)begin * embed_dim;
            const size_t bytes = (size_t)count * embed_dim * sizeof(uint16_t);
            if (storage.dataDeviceIds.front() == hidden.dataDeviceIds.front()) {
                AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(dst, src, bytes),
                               "Draft prefix copy failed.");
            } else {
                // Complete the source writes before handing the slot to another
                // device; the common copy retains its bounded host-staged fallback.
                FastllmCudaSyncCurrentThreadStream();
                FastllmCudaMemcpyBetweenDevices(storage.dataDeviceIds.front(), dst,
                    hidden.dataDeviceIds.front(), src, bytes);
            }
            snapshot->draftReady = true;
        }
    } else ClearDraftPrefixHidden(context.prefixHidden);
    // TP workers use other per-thread streams but share the allocation pool.
    // Finish draft KV writes before retiring captured features/temporaries:
    // the next prefill chunk may immediately reuse their storage on a worker.
    if (tpDevices.size() > 1) FastllmCudaSyncCurrentThreadStream();
#endif
}

#ifdef USE_CUDA
// The dynamic draft graph binds request KV addresses at replay. Slide the
// chronological view into the existing rounded allocation; when its spare
// rows run out, let the ordinary batched trim compact the suffix to row zero.
int NaiveN05FlashModel::DraftContext::SlidingCapacity(int reserve, int width) const {
    if (kv.empty() || kv.front().first.dims.size() != 3) return 0;
    const int old = kv.front().first.dims[1];
    auto &storage = kvBacking.empty() ? kv : kvBacking;
    int capacity = INT_MAX;
    for (size_t layer = 0; layer < kv.size(); ++layer) {
        const Data *live[] = {&kv[layer].first, &kv[layer].second};
        const Data *base[] = {&storage[layer].first, &storage[layer].second};
        for (int i = 0; i < 2; ++i) {
            if (live[i]->dims.size() != 3 || live[i]->dims[1] != old || !base[i]->cudaData ||
                base[i]->dataType != BFLOAT16 || base[i]->dataDevice != CUDA || base[i]->isFake || base[i]->cudaDataBorrowed ||
                !base[i]->isKVCache || !FastllmNaiveCanResizeKVLength(*base[i], reserve, width) ||
                live[i]->cudaData != (uint16_t *)base[i]->cudaData + size_t(kvOffset) * width) return 0;
            capacity = std::min(capacity, base[i]->expansionDims[1]);
        }
    }
    return capacity;
}

void NaiveN05FlashModel::DraftContext::BindKVView(int next, int length, int width) {
    if (kvBacking.empty()) {
        std::vector<std::pair<Data, Data>> views(kv.size());
        kvBacking.swap(kv);
        kv.swap(views);
        for (size_t layer = 0; layer < kv.size(); ++layer) {
            auto init = [](Data &view, const Data &owner) {
                view.dataType = owner.dataType; view.UpdateUnitSize();
                view.dataDevice = owner.dataDevice; view.dataDeviceIds = owner.dataDeviceIds;
                view.dims = owner.dims; view.strides = owner.strides;
                view.expansionDims = owner.expansionDims;
                view.isKVCache = view.cudaDataBorrowed = true;
            };
            init(kv[layer].first, kvBacking[layer].first);
            init(kv[layer].second, kvBacking[layer].second);
        }
    }
    for (size_t layer = 0; layer < kv.size(); ++layer) {
        auto bind = [&](Data &view, const Data &owner) {
            view.cudaData = (uint16_t *)owner.cudaData + size_t(next) * width;
            view.dims[1] = length;
            view.expansionDims[1] = owner.expansionDims[1] - next;
            view.expansionSize = view.strides[0] = size_t(view.expansionDims[1]) * width;
            view.expansionBytes = view.expansionSize * sizeof(uint16_t);
        };
        bind(kv[layer].first, kvBacking[layer].first);
        bind(kv[layer].second, kvBacking[layer].second);
    }
    kvOffset = next;
}

void NaiveN05FlashModel::DraftContext::SlideKV(int count, int window, int reserve, int width) {
    const int old = kv.front().first.dims[1];
    const int drop = std::max(0, old + count - (window - 1));
    if (!drop) return;
    const int capacity = SlidingCapacity(reserve, width);
    if (capacity <= reserve) return;
    const int next = drop <= capacity - reserve - kvOffset ? kvOffset + drop : 0;
    BindKVView(next, old + kvOffset - next, width);
}

bool NaiveN05FlashModel::DraftContext::PrepareKVAppend(int maxCount, int window, int reserve, int width) {
    const int capacity = SlidingCapacity(reserve, width);
    if (capacity <= reserve) return false;
    const int old = kv.front().first.dims[1];
    if (old < 0 || old > window - 1) return false;
    const int maxDrop = std::max(0, old + maxCount - (window - 1));
    if (maxDrop > capacity - reserve) return false;
    if (maxDrop > capacity - reserve - kvOffset) {
        // Compact only the live prefix, before sampling. Every legal accepted
        // length can now append in place and advance the view without copying.
        BindKVView(0, old + kvOffset, width);
        for (auto &pair : kv) FastllmCudaNaiveTrimCache(pair.first, pair.second, old);
    }
    return true;
}
#endif

#ifdef USE_CUDA
// Shifted cache views require a dynamic pointer table on every draft rank.
bool NaiveN05FlashModel::CanSlideDraftKV() const {
    return draftTP && draftTP->dataParallel && (!GetFastllmEnv().cudaGraph ||
        (draftBatchStorage && draftBatchStorage->tpRanks.size() == draftTP->devices.size() &&
         std::all_of(draftBatchStorage->tpRanks.begin(), draftBatchStorage->tpRanks.end(),
                     [](const auto &rank) { return !rank->batchGraphs.empty(); })));
}

bool NaiveN05FlashModel::ProjectDraftKVWeights(Data &input, int device, int heads, int width,
                                                Data &output, Data &pointers) {
    std::vector<Data> weightViews(draftLayers);
    std::vector<const Data *> weights;
    for (int layer = 0; layer < draftLayers; ++layer) {
        const auto name = "dspark.layers." + std::to_string(layer) + ".self_attn." + DraftKVProjectionName();
        const auto merged = weight.weight.find(name);
        if (merged == weight.weight.end()) return false;
        Data &w = draftTP ? *merged->second.multiDeviceDatas.at(device) : merged->second;
        if (w.dataType != BFLOAT16 || w.dims != std::vector<int>{heads * draftHeadDim + 2 * width, embed_dim})
            return false;
        w.ToDevice(CUDA, std::vector<int>{device});
        auto &view = weightViews[layer];
        view.Resize({2 * width, embed_dim});
        view.FakeFrom(w, (size_t)heads * draftHeadDim * embed_dim * sizeof(uint16_t));
        weights.push_back(&view);
    }
    return FastllmCudaNaiveDraftKVProject(input, weights, output, pointers);
}
#endif

void NaiveN05FlashModel::UnpackDraftFeatures(TargetCapture &capture, const std::vector<int> &lengths) {
#ifdef USE_CUDA
    if (capture.packedRows.empty()) return;
    ApplyDraftDevice();
    const int rows = std::accumulate(lengths.begin(), lengths.end(), 0);
    const size_t bytes = (size_t)embed_dim * sizeof(uint16_t);
    std::vector<void *> dst;
    std::vector<const void *> src;
    std::vector<size_t> sourcePitch, destPitch, widths, heights;
    for (size_t l = 0; l < draftTargetLayers.size(); ++l) {
        auto &out = capture.hidden[draftTargetLayers[l]];
        out.dataType = BFLOAT16; out.UpdateUnitSize(); out.Resize({1, rows, embed_dim});
        out.ToDevice(CUDA, {FastllmCudaGetDevice()}, false); out.Allocate(false);
        size_t offset = 0;
        for (size_t i = 0; i < lengths.size(); ++i) {
            dst.push_back((uint16_t *)out.cudaData + offset * embed_dim);
            src.push_back(capture.packedRows.at(i).data + l * embed_dim);
            sourcePitch.push_back(bytes * draftTargetLayers.size()); destPitch.push_back(bytes);
            widths.push_back(bytes); heights.push_back(lengths[i]); offset += lengths[i];
        }
    }
    AssertInFastLLM(FastllmCudaBatchCopy2DAsyncCurrentThread(dst.data(), src.data(), destPitch.data(),
        sourcePitch.data(), widths.data(), heights.data(), dst.size()), "Cannot restore draft feature layout");
    FastllmCudaSyncCurrentThreadStream();
    capture.packedRows.clear();
#endif
}

bool NaiveN05FlashModel::CommitDraftContexts(TargetCapture &capture, const std::vector<int> &lengths,
        const std::vector<int> &counts, const std::vector<DraftContext *> &contexts) {
#ifdef USE_CUDA
    if (GetLowMemMode() || (draftTP && !draftContextStorage) ||
        contexts.size() < 2 || contexts.size() != counts.size() ||
        lengths.size() != counts.size() || draftHeadDim <= 0 || draftHeadDim > 256 || draftHeadDim % 2) return false;
    ApplyDraftDevice();
    const std::vector<int> devices{FastllmCudaGetDevice()};
    const int ranks = draftTP ? draftTP->devices.size() : 1;
    const bool dataParallel = draftTP && draftTP->dataParallel;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int heads = DraftContextQueryHeads() / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
    const int width = kvHeads * draftHeadDim;
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(draftTP->devices[rank]) : weight[name];
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank && !dataParallel ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    int total = 0, captured = 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        auto *context = contexts[i];
        if (!context || counts[i] < 1 || counts[i] > lengths[i] || counts[i] >= draftWindow ||
            (dataParallel && (context->deviceRank < 0 || context->deviceRank >= ranks)) ||
            context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != (size_t)ranks))) return false;
        total += counts[i]; captured += lengths[i];
        for (int rank = 0; rank < ranks; ++rank) if (!dataParallel || rank == context->deviceRank)
        for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds.size() != 1 ||
                cache->dataDeviceIds.front() != (draftTP ? draftTP->devices[rank] : devices.front()) ||
                cache->dims.size() != 3 || cache->dims[0] != 1 || cache->dims[2] != width || !cache->cudaData) return false;
    }
    for (int rank = 0; rank < ranks; ++rank) for (int i = 0; i < draftLayers; ++i) {
        const auto name = "dspark.layers." + std::to_string(i) + ".self_attn.";
        auto merged = weight.weight.find(name + DraftKVProjectionName());
        if (merged != weight.weight.end()) {
            const Data &w = weightAt(name + DraftKVProjectionName(), rank);
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims !=
                std::vector<int>{(heads + 2 * kvHeads) * draftHeadDim, embed_dim}) return false;
        } else for (auto part : {"k_proj.weight", "v_proj.weight"}) {
            const Data &w = weightAt(name + part, rank);
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims != std::vector<int>{width, embed_dim}) return false;
        }
        const Data &norm = weightAt(name + "k_norm.weight", rank);
        if (norm.dataType != FLOAT32 || norm.multiDeviceData || norm.dims != std::vector<int>{draftHeadDim}) return false;
    }
    Data localCombined, localProjected, localHidden;
    auto *scratch = draftContextStorage.get();
    Data &combined = scratch ? scratch->combined : localCombined;
    Data &projected = scratch ? scratch->projected : localProjected;
    Data &hidden = scratch ? scratch->committedHidden : localHidden;
    Data &contextFC = weight["dspark.fc.weight"], &contextNorm = weight["dspark.hidden_norm.weight"];
    const bool localProjection = dataParallel && contextFC.IsTensorParallelReplicated() &&
        contextNorm.IsTensorParallelReplicated() && contextFC.dataType == BFLOAT16;
    if (!localProjection) UnpackDraftFeatures(capture, lengths);
    if (capture.packedRows.empty()) {
        std::vector<const Data *> inputs;
        for (int layer : draftTargetLayers) inputs.push_back(&capture.hidden.at(layer));
        std::vector<int> selected;
        if (total != captured) {
            selected.reserve(total);
            int sourceRow = 0;
            for (size_t i = 0; i < contexts.size(); ++i) {
                for (int j = 0; j < counts[i]; ++j) selected.push_back(sourceRow + j);
                sourceRow += lengths[i];
            }
        }
        if (!FastllmCudaNaiveDraftConcatRows(inputs, captured, selected, combined)) return false;
    }
    if (!localProjection) ProjectDraftContext(combined, projected, hidden);
    std::vector<std::vector<int>> rankRequests(ranks);
    std::vector<int> rowOffsets(contexts.size());
    int offset = 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        rowOffsets[i] = offset; offset += counts[i];
        if (dataParallel) rankRequests[contexts[i]->deviceRank].push_back(i);
        else for (auto &requests : rankRequests) requests.push_back(i);
    }
    const bool slideViews = CanSlideDraftKV();
    auto project = [&](int rank, Data &input) {
        const auto &requests = rankRequests[rank];
        const int total = input.dims[1];
        if (slideViews)
            for (int i : requests) contexts[i]->SlideKV(counts[i], draftWindow, draftWindow + draftBlock, width);
        const std::vector<int> localDevices{draftTP ? draftTP->devices[rank] : devices.front()};
        std::vector<Data> raw(draftLayers);
        std::vector<FastllmNaiveDraftKVUpdate> updates;
        updates.reserve(requests.size() * draftLayers);
        // Every layer projects the same committed hidden rows. Reuse the existing
        // batched GEMM and keep the per-layer path for split or unsupported weights.
        Data localKV, localPointers;
        auto *rankScratch = draftTP ? draftContextStorage->tpRanks[rank].get() : nullptr;
        Data &projectedKV = rankScratch ? rankScratch->contextRaw : localKV;
        Data &pointers = rankScratch ? rankScratch->contextPointers : localPointers;
        const bool batched = ProjectDraftKVWeights(input, localDevices.front(), heads, width, projectedKV, pointers);
        for (int layer = 0; layer < draftLayers; ++layer) {
            const auto name = "dspark.layers." + std::to_string(layer) + ".self_attn.";
            auto merged = weight.weight.find(name + DraftKVProjectionName());
            if (batched) {
                raw[layer].Resize({1, total, 2 * width});
                raw[layer].FakeFrom(projectedKV, (size_t)layer * total * 2 * width * sizeof(uint16_t));
            } else if (merged != weight.weight.end()) {
                Data &w = weightAt(name + DraftKVProjectionName(), rank);
                w.ToDevice(DataDevice::CUDA, localDevices);
                Data view; view.Resize({2 * width, embed_dim});
                view.FakeFrom(w, (size_t)heads * draftHeadDim * embed_dim * sizeof(uint16_t));
                DraftLinear(input, view, raw[layer]);
            } else {
                Data key, value;
                DraftLinear(input, weightAt(name + "k_proj.weight", rank), key);
                DraftLinear(input, weightAt(name + "v_proj.weight", rank), value);
                Cat(key, value, -1, raw[layer]);
            }
            Data &norm = weightAt(name + "k_norm.weight", rank); norm.ToDevice(DataDevice::CUDA, localDevices);
            int offset = 0;
            for (int i : requests) {
                auto &pair = cacheAt(*contexts[i], rank, layer);
                updates.push_back({&raw[layer], &norm, &pair.first, &pair.second,
                    contexts[i]->committed, offset, counts[i]});
                offset += counts[i];
            }
        }
        AssertInFastLLM(FastllmCudaNaiveBatchDraftKV(updates, kvHeads, draftHeadDim, draftWindow,
            draftWindow + draftBlock, draftEps, draftTheta), "Invalid draft context batch.");
    };
    if (draftTP) {
        auto &ws = *draftContextStorage;
        const bool waitForRoot = capture.packedRows.empty() && std::any_of(rankRequests.begin() + 1,
            rankRequests.end(), [](const auto &requests) { return !requests.empty(); });
        if (waitForRoot) {
            if (!ws.done) ws.done = FastllmCudaEventCreate();
            FastllmCudaEventRecordCurrentThread(ws.done);
        }
        draftTP->Run([&](int rank) {
            auto &r = *ws.tpRanks[rank];
            const auto &requests = rankRequests[rank];
            if (requests.empty()) { FastllmCudaEventRecordCurrentThread(r.done); return; }
            if (rank && waitForRoot) FastllmCudaCurrentThreadStreamWaitEvent(ws.done);
            Data &source = localProjection ? combined : hidden;
            const int width = capture.packedRows.empty() ? source.dims.back() : embed_dim * draftTargetLayers.size();
            Data *input = &source;
            if (dataParallel || rank) {
                auto &local = localProjection ? r.combined : r.contextSelected;
                int rows = 0; for (int i : requests) rows += counts[i];
                local.dataType = BFLOAT16; local.UpdateUnitSize(); local.Resize({1, rows, width});
                local.ToDevice(DataDevice::CUDA, {r.device}, false); local.Allocate(false);
                if (dataParallel) {
                    size_t offset = 0;
                    std::vector<void *> destinations;
                    std::vector<const void *> sources;
                    std::vector<size_t> sizes;
                    for (int i : requests) {
                        destinations.push_back((uint16_t *)local.cudaData + offset * width);
                        sources.push_back(capture.packedRows.empty()
                            ? (const uint16_t *)source.cudaData + size_t(rowOffsets[i]) * width
                            : capture.packedRows.at(i).data);
                        sizes.push_back(size_t(counts[i]) * width * sizeof(uint16_t));
                        offset += counts[i];
                    }
                    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
                        sources.data(), sizes.data(), sizes.size()), "Draft DP context transfer failed.");
                } else AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, local.cudaData,
                    devices.front(), source.cudaData, source.GetBytes()), "Draft TP batch context transfer failed.");
                input = &local;
            }
            if (localProjection) {
                DraftLinear(*input, *contextFC.multiDeviceDatas.at(r.device), r.projected);
                KimiK3RMSNorm(r.projected, *contextNorm.multiDeviceDatas.at(r.device), draftEps, r.committedHidden);
                input = &r.committedHidden;
            }
            project(rank, *input);
            FastllmCudaEventRecordCurrentThread(r.done);
        });
        for (int rank = 1; rank < ranks; ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    } else project(0, hidden);
    for (size_t i = 0; i < contexts.size(); ++i) {
        contexts[i]->committed += counts[i];
        ClearDraftPrefixHidden(contexts[i]->prefixHidden);
    }
    if (draftTP || tpDevices.size() > 1) FastllmCudaSyncCurrentThreadStream();
    return true;
#else
    return false;
#endif
}

Data NaiveN05FlashModel::RunDraft(int anchor, DraftContext &context) {
    Data normalized;
#ifdef USE_CUDA
    if (draftTP && RunDraftBatch({anchor}, {&context}, normalized)) return normalized;
#endif
    if (RunDraftGraph(anchor, context, normalized)) return normalized;
#ifdef USE_CUDA
    ApplyDraftDevice();
    Data id(FLOAT32, {1, 1}, {(float)anchor}), hidden;
    Embedding(id, DraftWeight("model.embed_tokens.weight"), hidden);
    ToDataType(hidden, BFLOAT16);
    Data &mask = weight["dspark.mask_embedding"];
    AssertInFastLLM(mask.Count(0) == embed_dim, "Invalid DSpark mask embedding.");
    mask.Reshape({1, 1, embed_dim});
    for (int i = 1; i < draftBlock; ++i) {
        Data joined;
        Cat(hidden, mask, 1, joined);
        Copy(joined, hidden);
    }
    std::vector<float> positions(draftBlock);
    for (int i = 0; i < draftBlock; ++i) positions[i] = context.committed + i;
    Data pos(FLOAT32, {1, draftBlock}, positions);
    for (int i = 0; i < draftLayers; ++i) {
        const std::string layer = "dspark.layers." + std::to_string(i) + ".";
        Data normed, q, k, v, qkv, attention, output, gate, up, gateUp;
        KimiK3RMSNorm(hidden, weight[layer + "input_layernorm.weight"], draftEps, normed);
        auto &cache = context.kv[i];
        const int past = cache.first.dims[1];
        const bool mergedQKV = weight.weight.count(layer + "self_attn.mergeqkv.weight");
        if (mergedQKV) DraftLinear(normed, weight[layer + "self_attn.mergeqkv.weight"], qkv);
        pos.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()});
        if (mergedQKV && FastllmCudaNaiveDraftQKV(qkv,
                weight[layer + "self_attn.q_norm.weight"], weight[layer + "self_attn.k_norm.weight"],
                pos, Data(), cache.first, cache.second, q,
                draftHeads, draftKvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
            cache.first.Resize({1, past + draftBlock, draftKvHeads * draftHeadDim});
            cache.second.Resize({1, past + draftBlock, draftKvHeads * draftHeadDim});
        } else {
            if (mergedQKV) {
                const int qw = draftHeads * draftHeadDim, kw = draftKvHeads * draftHeadDim;
                Split(qkv, -1, 0, qw, q);
                Split(qkv, -1, qw, qw + kw, k);
                Split(qkv, -1, qw + kw, qw + 2 * kw, v);
            } else {
                DraftLinear(normed, weight[layer + "self_attn.q_proj.weight"], q);
                DraftLinear(normed, weight[layer + "self_attn.k_proj.weight"], k);
                DraftLinear(normed, weight[layer + "self_attn.v_proj.weight"], v);
            }
            q.Reshape({1, draftBlock * draftHeads, draftHeadDim});
            k.Reshape({1, draftBlock * draftKvHeads, draftHeadDim});
            KimiK3RMSNorm(q, weight[layer + "self_attn.q_norm.weight"], draftEps, q);
            KimiK3RMSNorm(k, weight[layer + "self_attn.k_norm.weight"], draftEps, k);
            q.Reshape({1, draftBlock, draftHeads * draftHeadDim});
            k.Reshape({1, draftBlock, draftKvHeads * draftHeadDim});
            pos.ToDevice(q.dataDevice, q.dataDeviceIds);
            FastllmCudaNaiveRope(q, pos, draftHeads, draftHeadDim, draftHeadDim, draftTheta);
            FastllmCudaNaiveRope(k, pos, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
            AppendCache(cache.first, k, BFLOAT16);
            AppendCache(cache.second, v, BFLOAT16);
        }
        FastllmCudaNaiveAttention(q, cache.first, cache.second, Data(), Data(),
            draftHeads, draftKvHeads, draftHeadDim, draftHeadDim, past, draftWindow, attention, false);
        cache.first.Resize({1, past, draftKvHeads * draftHeadDim});
        cache.second.Resize({1, past, draftKvHeads * draftHeadDim});
        DraftLinear(attention, weight[layer + "self_attn.o_proj.weight"], output);
        AddTo(hidden, output);
        KimiK3RMSNorm(hidden, weight[layer + "post_attention_layernorm.weight"], draftEps, normed);
        if (weight.weight.count(layer + "mlp.gateup_proj.weight")) {
            DraftLinear(normed, weight[layer + "mlp.gateup_proj.weight"], gateUp);
            FastllmCudaNaiveDraftSwiGLU(gateUp, gate);
        } else {
            DraftLinear(normed, weight[layer + "mlp.gate_proj.weight"], gate);
            DraftLinear(normed, weight[layer + "mlp.up_proj.weight"], up);
            Silu(gate, gate);
            MulTo(gate, up);
        }
        DraftLinear(gate, weight[layer + "mlp.down_proj.weight"], output);
        AddTo(hidden, output);
    }
    KimiK3RMSNorm(hidden, weight["dspark.norm.weight"], draftEps, normalized);
#endif
    return normalized;
}


bool NaiveN05FlashModel::CanShardDraftHead() const {
#ifdef USE_CUDA
    return draftTP && draftTP->dataParallel && draftTP->devices == tpDevices;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::RunDraftBatch(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &normalized, int reserveRows,
        DraftHeadShards *headShards) {
#ifdef USE_CUDA
    if (contexts.size() < (draftTP ? 1 : 2) || anchors.size() != contexts.size() ||
        !GetCudaEmbedding() || GetLowMemMode() || draftBlock < 2 || draftBlock >= 32 ||
        draftHeadDim % 4 || draftHeadDim > 256) return false;
    ApplyDraftDevice();
    const std::vector<int> devices = draftTP ? draftTP->devices : std::vector<int>{FastllmCudaGetDevice()};
    const int ranks = devices.size(), device = devices.front();
    const bool dataParallel = draftTP && draftTP->dataParallel;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int batch = contexts.size();
    // Keep each request's KV on a stable DP owner, including cancellation,
    // replacement and reordered batches. Only feature rows cross devices.
    std::vector<std::vector<int>> rankRequests(ranks);
    for (size_t i = 0; i < contexts.size(); ++i) {
        if (!contexts[i]) return false;
        if (dataParallel) {
            const int rank = contexts[i]->deviceRank;
            if (rank < 0 || rank >= ranks) return false;
            rankRequests[rank].push_back(i);
        } else for (auto &requests : rankRequests) requests.push_back(i);
    }
    if (headShards && (!CanShardDraftHead() || headShards->limits.size() != contexts.size() ||
        std::any_of(headShards->limits.begin(), headShards->limits.end(), [&](int n) { return n < 1 || n > draftBlock; })))
        return false;
    const int heads = draftHeads / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(devices[rank]) : DraftWeight(name);
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank && !dataParallel ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    Data &embedding = weightAt("model.embed_tokens.weight", 0), &mask = weightAt("dspark.mask_embedding", 0);
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 || mask.Count(0) != embed_dim) return false;
    for (auto *context : contexts) {
        if (context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != devices.size()))) return false;
        for (int rank = 0; rank < ranks; ++rank) if (!dataParallel || rank == context->deviceRank)
        for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds.size() != 1 || cache->dataDeviceIds.front() != devices[rank] ||
                cache->dims.size() != 3 ||
                cache->dims[2] != kvHeads * draftHeadDim || cache->expansionDims.size() != 3 ||
                cache->expansionDims[1] < draftWindow + draftBlock) return false;
    }
    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    for (auto *context : contexts) if (!context->workspace) {
        context->workspace = std::make_shared<DraftWorkspace>();
        auto &state = *context->workspace;
        state.device = device; state.inputs = {embedding.cudaData, mask.cudaData};
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second}) state.inputs.push_back(cache->cudaData);
        state.layers.resize(draftLayers); state.contextLayers.resize(draftLayers);
    }
    // A stable device table decouples graph lifetime from request KV lifetime.
    // The fused attention path reads both lengths and KV pointers on device;
    // other layouts keep their existing address/short-attention keyed fallback.
    bool dynamicKV = true;
    std::vector<int> rankRows;
    for (int rank = 0; rank < ranks; ++rank) {
        rankRows.push_back(rankRequests[rank].size());
        if (!rankRows.back()) continue;
        FastllmCudaSetDevice(devices[rank]);
        dynamicKV = (dataParallel || rankRows.back() >= 8) &&
            FastllmCudaNaiveDraftCacheBindingsSupported(rankRows.back(), draftBlock,
            heads, kvHeads, draftHeadDim, draftWindow) && dynamicKV;
    }
    FastllmCudaSetDevice(device);
    if (!draftBatchStorage || draftBatchStorage->device != device) {
        draftBatchStorage = std::make_shared<DraftWorkspace>();
        draftBatchStorage->device = device;
        for (int gpu : devices) {
            auto scratch = std::make_shared<DraftWorkspace>(); scratch->device = gpu;
            draftBatchStorage->tpRanks.push_back(std::move(scratch));
        }
    }
    auto scratch = [&](int rank) -> DraftWorkspace & { return *draftBatchStorage->tpRanks.at(rank); };
    const bool bucketedDP = dataParallel && dynamicKV && GetFastllmEnv().cudaGraph;
    std::shared_ptr<DraftWorkspace> state;
    if (bucketedDP) {
        const int maxLocal = WindowRequestsPerGroup(maxBatch > 0 ? maxBatch : 512, ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            if (!rankRows[rank]) continue;
            rankRows[rank] = DraftGraphBucket(std::max(rankRows[rank], reserveRows), maxLocal);
            auto &entry = scratch(rank).batchGraphs[rankRows[rank]];
            if (entry) continue;
            FastllmCudaSetDevice(devices[rank]);
            entry = std::make_shared<DraftWorkspace>();
            entry->device = devices[rank];
            entry->done = FastllmCudaEventCreate();
            entry->batchDescriptors.resize(draftLayers);
        }
    } else {
        std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
        std::vector<bool> shortRows;
        if (!dynamicKV) for (auto *context : contexts) {
            shortRows.push_back(std::min(context->committed, draftWindow - 1) + draftBlock <= 256);
            for (int rank = 0; rank < ranks; ++rank) if (!dataParallel || rank == context->deviceRank)
            for (int layer = 0; layer < draftLayers; ++layer)
            for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
                inputs.push_back(cache->cudaData);
        }
        if (GetFastllmEnv().cudaGraph) {
            auto found = std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(),
                [&](const std::shared_ptr<DraftWorkspace> &entry) {
                    return entry->device == device && entry->inputs == inputs && entry->shortRows == shortRows &&
                        entry->rankRows == rankRows;
                });
            if (found != draftBatchWorkspaces.end()) {
                state = *found;
                draftBatchWorkspaces.erase(found);
            }
        }
        if (!state) {
            state = std::make_shared<DraftWorkspace>();
            state->device = device;
            state->inputs = std::move(inputs);
            state->shortRows = std::move(shortRows);
            state->rankRows = rankRows;
            if (draftTP) for (int gpu : devices) {
                FastllmCudaSetDevice(gpu);
                auto local = std::make_shared<DraftWorkspace>();
                local->device = gpu;
                local->done = FastllmCudaEventCreate();
                state->tpRanks.push_back(std::move(local));
            }
        }
        if (GetFastllmEnv().cudaGraph) {
            draftBatchWorkspaces.push_back(state);
            if (draftBatchWorkspaces.size() > 8) draftBatchWorkspaces.erase(draftBatchWorkspaces.begin());
        }
    }
    FastllmCudaSetDevice(device);
    auto localState = [&](int rank) -> DraftWorkspace & {
        if (bucketedDP) return *scratch(rank).batchGraphs.at(rankRows[rank]);
        return draftTP ? *state->tpRanks[rank] : *state;
    };
    // Install map entries on the caller before rank workers read them.
    for (int gpu : devices) draftLinearGraphWorkspaces[gpu];
    auto run = [&](const std::function<void(int)> &fn) {
        if (draftTP) draftTP->Run(fn); else fn(0);
    };
    auto upload = [&](int rank) {
        auto &r = scratch(rank);
        const auto &requests = rankRequests[rank];
        if (requests.empty()) return;
        const int count = rankRows[rank];
        auto copy = [&](Data &target, DataType type, const void *host, const std::vector<int> &dims) {
            target.dataType = type; target.UpdateUnitSize(); target.Resize(dims);
            target.ToDevice(DataDevice::CUDA, {devices[rank]}, false); target.Allocate(false);
            FastllmCudaCopyFromHostToDevice(target.cudaData, const_cast<void *>(host), target.GetBytes());
        };
        std::vector<float> localTokens(count);
        std::vector<int> localLengths(count);
        for (size_t row = 0; row < requests.size(); ++row) {
            const int seq = requests[row];
            localTokens[row] = anchors[seq];
            localLengths[row] = contexts[seq]->committed + 1;
        }
        copy(r.id, FLOAT32, localTokens.data(), {1, count});
        copy(r.live, INT32, localLengths.data(), {count});
        if (dynamicKV) {
            std::vector<uint64_t> bindings(size_t(draftLayers) * count * 2);
            for (int layer = 0; layer < draftLayers; ++layer) for (size_t row = 0; row < requests.size(); ++row) {
                auto &cache = cacheAt(*contexts[requests[row]], rank, layer);
                const size_t slot = (size_t(layer) * count + row) * 2;
                bindings[slot] = (uint64_t)cache.first.cudaData;
                bindings[slot + 1] = (uint64_t)cache.second.cudaData;
            }
            r.batchKV.dataType = INT32; r.batchKV.UpdateUnitSize();
            r.batchKV.Resize({draftLayers, std::max(maxBatch, count), 4});
            r.batchKV.ToDevice(DataDevice::CUDA, {devices[rank]}, false); r.batchKV.Allocate(false);
            FastllmCudaCopyFromHostToDevice(r.batchKV.cudaData, bindings.data(), bindings.size() * sizeof(uint64_t));
        }
    };
    auto body = [&](int rank, bool reduce = true) {
        const auto &selected = rankRequests[rank];
        if (selected.empty()) return;
        std::vector<DraftContext *> localContexts;
        for (int i : selected) localContexts.push_back(contexts[i]);
        // Only geometry is captured. Padded slots bind null in the replay table.
        if (bucketedDP) localContexts.resize(rankRows[rank], localContexts.front());
#ifndef USE_ROCM
        FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces.at(devices[rank]));
#endif
        auto w = [&](const std::string &name) -> Data & { return weightAt(name, rank); };
        auto linear = [&](Data &input, Data &weight, Data &output) {
            // TP uses the existing cached Lt/GEMV dispatch. The strided-GEMM
            // API can lazily change CUDA kernel attributes between collectives.
            if (draftTP) Linear(input, weight, Data(), output);
            else DraftLinear(input, weight, output);
        };
        auto allReduce = [&](Data &value, Data &reduced) {
            if (!draftTP || dataParallel) return;
            // Keep both typed allocations stable across graph variants.
            // In-place conversion replaces the shared BF16 pointer during
            // capture/replay transitions and defeats workspace reuse.
            ToDataType(value, reduced, FLOAT32);
            if (reduce) FastllmCudaNaiveDraftTPReduce(draftTP->group, rank, reduced);
            ToDataType(reduced, value, BFLOAT16);
        };
        auto &r = scratch(rank);
        if (r.layers.empty()) r.layers.resize(draftLayers);
        Data &ids = r.id, &live = r.live, &result = r.normalized;
        auto addNorm = [&](Data &hidden, Data &branch, Data &gamma, Data &output) {
            gamma.ToDevice(DataDevice::CUDA, std::vector<int>{devices[rank]});
            if (hidden.dataType == BFLOAT16 && branch.dataType == BFLOAT16 &&
                    gamma.dataType == FLOAT32 && gamma.Count(0) == hidden.dims.back()) {
                FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, gamma, draftEps, output);
            } else {
                AddTo(hidden, branch);
                KimiK3RMSNorm(hidden, gamma, draftEps, output);
            }
        };
        Data &embedding = w("model.embed_tokens.weight"), &mask = w("dspark.mask_embedding");
        struct Request { Data positions, live; };
        std::vector<Request> requests(localContexts.size());
        auto append = [](Data &batch, Data &part, bool first) {
            if (first) Copy(part, batch);
            else { Data joined; Cat(batch, part, 1, joined); Copy(joined, batch); }
        };
        Data &hidden = r.hidden, &positions = r.positions, &normed = r.combined;
        FastllmCudaNaiveDraftInput(ids, embedding, mask, live, draftBlock, hidden, positions);
        for (size_t row = 0; row < localContexts.size(); ++row) {
            requests[row].positions.Resize({1, draftBlock});
            requests[row].positions.FakeFrom(positions, row * draftBlock * sizeof(float));
            requests[row].live.Resize({1}); requests[row].live.FakeFrom(live, row * sizeof(int));
        }
        KimiK3RMSNorm(hidden, w("dspark.layers.0.input_layernorm.weight"), draftEps, normed);
        for (int layer = 0; layer < draftLayers; ++layer) {
            const std::string p = "dspark.layers." + std::to_string(layer) + ".";
            auto &l = r.layers[layer];
            Data &qkv = l.qkv, &qs = l.q, &ks = l.k, &vs = l.v, &attention = l.attention;
            Data &output = l.output, &gate = l.gate, &up = l.up, &gateUp = l.gateUp;
            const bool merged = weight.weight.count(p + "self_attn.mergeqkv.weight");
            if (merged) linear(normed, w(p + "self_attn.mergeqkv.weight"), qkv);
            else {
                linear(normed, w(p + "self_attn.q_proj.weight"), qs);
                linear(normed, w(p + "self_attn.k_proj.weight"), ks);
                linear(normed, w(p + "self_attn.v_proj.weight"), vs);
            }
            std::vector<FastllmNaiveDraftRow> descriptors;
            for (size_t row = 0; row < localContexts.size(); ++row) {
                auto *context = localContexts[row];
                auto &cache = cacheAt(*context, rank, layer);
                descriptors.push_back({&cache.first, &cache.second, &requests[row].positions,
                    &requests[row].live, !dynamicKV && std::min(context->committed, draftWindow - 1) + draftBlock <= 256});
            }
            Data &packed = l.packed, &query = l.query, &scores = l.scores;
            Data bindings(INT32, {(int)localContexts.size(), 4});
            if (dynamicKV) bindings.FakeFrom(r.batchKV, (size_t)layer * localContexts.size() * 2 * sizeof(uint64_t));
            if (!merged) {
                Data qk;
                Cat(qs, ks, -1, qk); Cat(qk, vs, -1, packed);
            }
            const bool batchedAttention = FastllmCudaNaiveBatchDraftQKV(merged ? qkv : packed,
                    w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                    descriptors, query, heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta,
                    &localState(rank).batchDescriptors[layer], dynamicKV ? &bindings : nullptr) &&
                FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kvHeads,
                    draftHeadDim, draftWindow, scores, attention, dynamicKV ? &bindings : nullptr);
            if (!batchedAttention) {
                AssertInFastLLM(!dynamicKV, "Naive draft dynamic KV layout changed after validation.");
                for (size_t row = 0; row < localContexts.size(); ++row) {
                    auto &r = requests[row]; auto &cache = cacheAt(*localContexts[row], rank, layer);
                    Data raw, q, k, v, scores, part;
                    const int begin = row * draftBlock, end = begin + draftBlock;
                    if (merged) Split(qkv, 1, begin, end, raw);
                    if (!merged || !FastllmCudaNaiveDraftQKV(raw,
                            w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                            r.positions, r.live, cache.first, cache.second, q,
                            heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
                        if (merged) {
                            const int qw = heads * draftHeadDim, kw = kvHeads * draftHeadDim;
                            Split(raw, -1, 0, qw, q); Split(raw, -1, qw, qw + kw, k);
                            Split(raw, -1, qw + kw, qw + 2 * kw, v);
                        } else {
                            Split(qs, 1, begin, end, q); Split(ks, 1, begin, end, k); Split(vs, 1, begin, end, v);
                        }
                        q.Reshape({1, draftBlock * heads, draftHeadDim});
                        k.Reshape({1, draftBlock * kvHeads, draftHeadDim});
                        KimiK3RMSNorm(q, w(p + "self_attn.q_norm.weight"), draftEps, q);
                        KimiK3RMSNorm(k, w(p + "self_attn.k_norm.weight"), draftEps, k);
                        q.Reshape({1, draftBlock, heads * draftHeadDim});
                        k.Reshape({1, draftBlock, kvHeads * draftHeadDim});
                        FastllmCudaNaiveRope(q, r.positions, heads, draftHeadDim, draftHeadDim, draftTheta);
                        FastllmCudaNaiveRope(k, r.positions, kvHeads, draftHeadDim, draftHeadDim, draftTheta);
                        FastllmCudaNaiveAppendVerifyCache(cache.first, cache.second, k, v, r.live, draftWindow);
                    }
                    const bool shortAttention = std::min(localContexts[row]->committed, draftWindow - 1) + draftBlock <= 256;
                    FastllmCudaNaiveDraftAttention(q, cache.first, cache.second, r.live,
                        heads, kvHeads, draftHeadDim, draftWindow, shortAttention, scores, part);
                    append(attention, part, row == 0);
                }
            }
            linear(attention, w(p + "self_attn.o_proj.weight"), output);
            if (draftTP && draftTP->attentionDivisor > 1) allReduce(output, l.reduced);
            addNorm(hidden, output, w(p + "post_attention_layernorm.weight"), normed);
            if (weight.weight.count(p + "mlp.gateup_proj.weight")) {
                linear(normed, w(p + "mlp.gateup_proj.weight"), gateUp);
                FastllmCudaNaiveDraftSwiGLU(gateUp, gate);
            } else {
                linear(normed, w(p + "mlp.gate_proj.weight"), gate);
                linear(normed, w(p + "mlp.up_proj.weight"), up); Silu(gate, gate); MulTo(gate, up);
            }
            linear(gate, w(p + "mlp.down_proj.weight"), output);
            allReduce(output, l.reduced);
            if (layer + 1 == draftLayers)
                addNorm(hidden, output, w("dspark.norm.weight"), result);
            else
                addNorm(hidden, output, w("dspark.layers." + std::to_string(layer + 1) +
                    ".input_layernorm.weight"), normed);
        }
    };
    if (bucketedDP) {
        run([&](int rank) {
            if (rankRequests[rank].empty()) return;
            auto &entry = localState(rank);
            auto &graph = entry.graphs[0];
            const bool warm = !graph.exec;
            const auto before = warm ? scratch(rank).Pointers() : std::vector<void *>{};
            upload(rank);
            if (warm) {
                body(rank);
                FastllmCudaSyncCurrentThreadStream();
                if (before != scratch(rank).Pointers())
                    for (auto &other : scratch(rank).batchGraphs)
                        if (other.second.get() != &entry) other.second->graphs[0].Clear();
                if (!entry.disabled) graph.Capture([&] { body(rank); }, entry.disabled);
            }
            if (graph.exec) AssertInFastLLM(FastllmCudaGraphLaunch(graph.exec), "Cannot replay draft DP graph");
            FastllmCudaEventRecordCurrentThread(entry.done);
        });
        if (!headShards) for (int rank = 1; rank < ranks; ++rank) if (!rankRequests[rank].empty())
            FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
    } else {
        for (int rank = 0; rank < ranks; ++rank) localState(rank).batchDescriptors.resize(draftLayers);
        std::vector<std::vector<void *>> before;
        for (int rank = 0; rank < ranks; ++rank) before.push_back(scratch(rank).Pointers());
        auto invalidateMoved = [&]() {
            bool changed = false;
            for (int rank = 0; rank < ranks; ++rank) changed = changed || before[rank] != scratch(rank).Pointers();
            if (!changed) return;
            for (auto &other : draftBatchWorkspaces) if (other != state) {
                other->graphs[0].Clear();
                for (auto &rank : other->tpRanks) rank->graphs[0].Clear();
            }
        };
        run(upload);
        if (draftTP && !state->warmed) {
            // Initialize every GEMM plan/kernel before any rank enters NCCL.
            // Only speculative scratch is written; committed KV is unchanged.
            // A CUDA lazy-load/attribute update while another rank is already
            // waiting in a collective can otherwise deadlock the driver.
            run([&](int rank) { body(rank, false); ForceDeviceSync(); });
            state->warmed = true;
        }
        bool useGraph = GetFastllmEnv().cudaGraph && !state->disabled;
        if (useGraph && !localState(0).graphs[0].exec) {
            run([&](int rank) { body(rank); ForceDeviceSync(); });
            invalidateMoved();
            if (!draftTP) useGraph = state->graphs[0].Capture([&] { body(0); }, state->disabled);
            else {
                std::vector<int> ready(ranks);
                auto allReady = [&] { return std::all_of(ready.begin(), ready.end(), [](int n) { return n != 0; }); };
                run([&](int rank) { ready[rank] = FastllmCudaGraphPrepareCaptureDevice(); });
                bool pool = allReady() && FastllmCudaGraphMemoryPoolBegin(), ok = pool;
                if (pool) {
                    run([&](int rank) { FastllmCudaClearThreadError(); ready[rank] = FastllmCudaGraphBeginCapture(); });
                    ok = allReady();
                    run([&](int rank) {
                        if (!ready[rank]) return;
                        if (ok) body(rank);
                        ready[rank] = FastllmCudaGraphEndCapture(&localState(rank).graphs[0].graph) && !FastllmCudaGetThreadError();
                    });
                    ok = FastllmCudaGraphMemoryPoolEnd(state->graphs[0].reserved) && ok && allReady();
                    if (ok) {
                        run([&](int rank) { auto &g = localState(rank).graphs[0]; ready[rank] = FastllmCudaGraphInstantiate(g.graph, &g.exec); });
                        ok = allReady();
                    }
                }
                if (!ok) {
                    run([&](int rank) { localState(rank).graphs[0].Clear(); FastllmCudaClearLastError(); FastllmCudaClearThreadError(); });
                    state->graphs[0].Clear(); state->disabled = true; useGraph = false;
                }
            }
        }
        // Capture can also materialize operator-owned buffers after warmup.
        // Retire other variants before replay if any shared address moved.
        invalidateMoved();
        run([&](int rank) {
            auto &r = localState(rank);
            if (useGraph) AssertInFastLLM(FastllmCudaGraphLaunch(r.graphs[0].exec), "Naive batch draft graph launch failed.");
            else body(rank);
            if (draftTP) FastllmCudaEventRecordCurrentThread(r.done);
        });
        if (draftTP && !headShards) for (int rank = 1; rank < ranks; ++rank)
            FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
        if (!useGraph) invalidateMoved();
    }
    if (headShards) {
        // Each vocabulary shard reads the live prefix directly from each DP owner.
        // Source rows are owner-local and can be reordered or padded by the backbone graph.
        std::vector<int> offsets(batch + 1);
        std::partial_sum(headShards->limits.begin(), headShards->limits.end(), offsets.begin() + 1);
        std::vector<const void *> sources(batch);
        std::vector<size_t> bytes(batch);
        for (int rank = 0; rank < ranks; ++rank) {
            const auto &requests = rankRequests[rank];
            const auto *source = static_cast<const uint16_t *>(scratch(rank).normalized.cudaData);
            for (size_t row = 0; row < requests.size(); ++row) {
                const int i = requests[row];
                sources[i] = source + row * draftBlock * embed_dim;
                bytes[i] = size_t(headShards->limits[i]) * embed_dim * sizeof(uint16_t);
            }
        }
        headShards->inputs.resize(ranks);
        headShards->logits.resize(ranks);
        run([&](int rank) {
            auto &input = headShards->inputs[rank];
            input.dataType = BFLOAT16; input.UpdateUnitSize();
            input.Resize({1, offsets.back(), embed_dim});
            input.ToDevice(CUDA, {devices[rank]}, false); input.Allocate(false);
            std::vector<void *> destinations(batch);
            for (int i = 0; i < batch; ++i)
                destinations[i] = static_cast<uint16_t *>(input.cudaData) + size_t(offsets[i]) * embed_dim;
            for (int peer = 0; peer < ranks; ++peer) if (peer != rank && !rankRequests[peer].empty())
                FastllmCudaCurrentThreadStreamWaitEvent(localState(peer).done);
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
                sources.data(), bytes.data(), batch), "Cannot gather DP rows on the draft vocabulary shard.");
            auto &head = *weight["lm_head.weight"].multiDeviceDatas.at(devices[rank]);
            if (offsets.back() > 1) MatMulTransB(input, head, headShards->logits[rank]);
            else Linear(input, head, Data(), headShards->logits[rank]);
            // Separate from the backbone event: a slower peer can still be enqueueing its input reads.
            auto &ready = scratch(rank).done;
            if (!ready) ready = FastllmCudaEventCreate();
            FastllmCudaEventRecordCurrentThread(ready);
        });
        return true;
    }
    if (dataParallel) {
        normalized.dataType = BFLOAT16; normalized.UpdateUnitSize();
        normalized.Resize({1, batch * draftBlock, embed_dim});
        normalized.ToDevice(DataDevice::CUDA, {device}, false); normalized.Allocate(false);
        std::vector<void *> destinations;
        std::vector<const void *> sources;
        std::vector<size_t> sizes;
        const size_t width = size_t(draftBlock) * embed_dim;
        for (int rank = 0; rank < ranks; ++rank) {
            const auto &requests = rankRequests[rank];
            for (size_t row = 0; row < requests.size(); ++row) {
                destinations.push_back((uint16_t *)normalized.cudaData + size_t(requests[row]) * width);
                sources.push_back((const uint16_t *)scratch(rank).normalized.cudaData + row * width);
                sizes.push_back(width * sizeof(uint16_t));
            }
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
            sources.data(), sizes.data(), sizes.size()), "Draft DP output gathering failed.");
        // Workers may reuse their scratch on the next round. Complete the
        // root-stream peer reads before returning ownership to the scheduler.
        FastllmCudaSyncCurrentThreadStream();
    } else {
        scratch(0).normalized.Resize({1, batch * draftBlock, embed_dim});
        Copy(scratch(0).normalized, normalized);
    }
    return true;
#else
    return false;
#endif
}

struct NaiveN05FlashModel::DraftRound {
    NaiveN05FlashModel &model;
    DraftContext &context;
    const GenerationConfig &config;
    std::optional<LastTokensUnit> lastTokens;
    LastTokensUnit CopyHistory() const {
        return lastTokens ? *lastTokens : LastTokensUnit(config.last_n);
    }
    int anchor, limit, oldLength, vocab = 0;
    bool greedy = false;
    int sampledAccepted = -1, sampledNext = -1;
    bool targetCommitted = false, draftCommitted = false;
    TargetCapture::FeatureRows feature;
    bool targetPayloadWritten = false;
    bool standalone = false; // Only this round may borrow the shared sampling arena.
    LogitsSelection selection;
    std::vector<int> proposed;
    std::vector<std::vector<float>> q;
    size_t proposalOffset = 0; // FP32 elements from this batch's probability storage
    int samplingRank = -1;
    Data *targetLogits = nullptr;
    std::shared_ptr<DraftWorkspace> proposalOwner;
    bool deferIds = false;
    int proposalRow = -1;
    const Data &ProposalStorage() const {
#ifdef USE_CUDA
        if (proposalOwner) return proposalOwner->sampling.probabilities;
#endif
        return context.proposalProbs;
    }
    float *ProposalData() const {
        return static_cast<float *>(ProposalStorage().cudaData) + proposalOffset;
    }

    DraftRound(NaiveN05FlashModel &model, DraftContext &context, const GenerationConfig &config,
               int anchor, int limit)
        : model(model), context(context), config(config),
          anchor(anchor), limit(limit), oldLength(context.committed) {}

    int Publish(int accepted, int next) {
        ++context.rounds;
        context.proposed += proposed.size();
        context.accepted += accepted;
        proposed.resize(accepted);
        proposed.push_back(next);
        for (int i = 1; i < (int)proposed.size(); ++i)
            context.pending.emplace_back(proposed[i - 1], proposed[i]);
        return proposed[0];
    }

#ifdef USE_CUDA
    // Only target argmax decides acceptance. Mask the bonus row as well if
    // it falls before min_tokens. Proposal logits can remain unmasked.
    void MaskGreedyStops(Data &logits, int offset) const {
        if (!greedy || config.output_token_least <= 0) return;
        const int remaining = config.output_token_least - (oldLength + 1 - config.input_token_length);
        const int rows = std::min<int>(proposed.size() + 1, remaining);
        if (rows <= 0) return;
        std::vector<int> stop;
        auto add = [&](int token) {
            if (token >= 0 && token < vocab && std::find(stop.begin(), stop.end(), token) == stop.end())
                stop.push_back(token);
        };
        add(model.eos_token_id);
        for (int token : model.eos_token_ids) add(token);
        for (int token : config.stop_token_ids) add(token);
        Data view;
        view.FakeFrom(logits, size_t(offset) * vocab * sizeof(float));
        view.Resize({rows, vocab});
        if (logits.dataDevice == CUDA) FastllmResetLogitsOfEOSAll(rows, &view, stop);
        else for (int row = 0; row < rows; ++row)
            for (int id : stop) reinterpret_cast<float *>(view.cpuData)[size_t(row) * vocab + id] = -1e30f;
    }
#endif

    std::vector<float> Distribution(Data &logits, int row, int position,
            const LastTokensUnit &history, const GenerationConfig &stepConfig) {
        logits.ToDevice(DataDevice::CPU);
        float *values = (float *)logits.cpuData + (size_t)row * vocab;
        if (config.output_token_least > position - config.input_token_length) {
            if (model.eos_token_id >= 0 && model.eos_token_id < vocab) values[model.eos_token_id] = -1e30f;
            for (int id : model.eos_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
            for (int id : config.stop_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
        }
        return SpeculativeDistribution(values, vocab, stepConfig, history);
    }
};

void NaiveN05FlashModel::PrepareDraftRound(DraftRound &round, const Data *preparedLogits) {
    auto &context = round.context;
    const auto &config = round.config;
    const int anchor = round.anchor, limit = round.limit, oldLength = round.oldLength;
    Data draftHidden = preparedLogits ? Data() : RunDraft(anchor, context);
    Data baseLogits = [&]() {
        if (preparedLogits) {
            Data view(preparedLogits->dataType, preparedLogits->dims);
            view.FakeFrom(*preparedLogits, 0);
            return view;
        }
        Data selected;
        // DSpark predicts after the anchor at slot 0, unlike DFlash's masked-slot-only head.
        Split(draftHidden, 1, 0, limit, selected);
        return RunDraftHead(selected);
    }();
    const int vocab = round.vocab = baseLogits.dims.back();
#ifdef USE_CUDA
    const bool localSampling = draftTP && draftTP->dataParallel && !draftSamplingRanks.empty() &&
        draftSamplingRanks.front() && draftSamplingRanks.front()->samplingW1.cudaData;
    if (round.standalone && (config.tool_call_grammar || localSampling) && draftConfidenceThreshold == 0 && baseLogits.dataDevice == DataDevice::CUDA &&
            baseLogits.dataType == BFLOAT16 && baseLogits.dims.size() == 3) {
        std::vector<std::unique_ptr<DraftRound>> batch;
        batch.emplace_back(new DraftRound(*this, context, config, anchor, limit));
        batch.front()->lastTokens = round.lastTokens;
        if (PrepareDraftBatchGraph({batch.front().get()}, baseLogits, localSampling ? 0 : -1)) {
            auto &r = *batch.front();
            round.greedy = r.greedy; round.selection = r.selection; round.proposed = std::move(r.proposed);
            round.proposalOwner = r.proposalOwner;
            round.proposalOffset = r.proposalOffset;
            return;
        }
    }
#endif
    bool canMaskStops = config.output_token_least <= 0;
#ifdef USE_CUDA
    canMaskStops |= baseLogits.dataDevice == CUDA;
#endif
    const bool greedy = round.greedy = DraftIsGreedy(config) && canMaskStops;
    LastTokensUnit samplingTokens = round.CopyHistory();
    auto &selection = round.selection;
    selection = SelectLogits(config, true);
#ifdef USE_CUDA
    // Grammar masks are applied per verification row. Greedy verification also
    // keeps full logits on the device, then transfers only the masked argmax.
    if (baseLogits.dataDevice == DataDevice::CUDA &&
        ((greedy && (config.tool_call_grammar || config.output_token_least > 0)) ||
         (!greedy && SupportsCudaRejectionSampling(config)))) {
        if (!greedy) {
            context.proposalProbs.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds, false);
            context.proposalProbs.Resize({limit, vocab});
            context.proposalProbs.Allocate(false);
        }
        selection.count = 0; // Even a vocabulary fitting compact top-k needs full device logits.
        selection.cudaLogits = true;
    }
#endif
    Data proposalPartial, proposalTop;
    std::vector<float> proposalCandidates(selection.count * 2);
    auto &proposed = round.proposed;
    auto &q = round.q;
    int previous = anchor;
    GenerationConfig proposalConfig = config;
    // Rejection sampling only requires q to match the proposal's actual law.
    // Keep temperature and grammar masks, but omit q's expensive nucleus sort;
    // target p still applies the user's complete sampling configuration.
    if (!greedy && selection.cudaLogits) {
        proposalConfig.top_k = 0;
        proposalConfig.top_p = 1.f;
    }
    bool gpuProposal = !config.tool_call_grammar && greedy && draftConfidenceThreshold == 0 &&
        RunDraftProposalGraph(anchor, baseLogits, context, proposed);
    for (int step = 0; !gpuProposal && step < limit; ++step) {
        if (config.tool_call_grammar) {
            PrepareToolCallConstraint(proposalConfig);
        }
        bool sampled = false;
#ifdef USE_CUDA
        if (!greedy && selection.cudaLogits && draftConfidenceThreshold == 0)
            sampled = RunDraftSamplingStep(previous, step, baseLogits, context, proposalConfig, previous);
#endif
        if (!sampled) {
            Data previousId(FLOAT32, {1, 1}, {(float)previous}), latent, bias, logits;
            Embedding(previousId, weight["dspark.markov_head.markov_w1.weight"], latent);
            ToDataType(latent, BFLOAT16);
            if (step > 0 && draftConfidenceThreshold > 0) {
                Data h, features, confidence;
                Split(draftHidden, 1, step, step + 1, h);
                Cat(h, latent, -1, features);
                ToDataType(features, FLOAT32);
                Linear(features, weight["dspark.confidence_head.proj.weight"],
                       weight["dspark.confidence_head.proj.bias"], confidence);
                confidence.ToDevice(DataDevice::CPU);
                float probability = 1.0f / (1.0f + std::exp(-((float *)confidence.cpuData)[0]));
                if (probability < draftConfidenceThreshold) break;
            }
            DraftLinear(latent, weight["dspark.markov_head.markov_w2.weight"], bias);
            Split(baseLogits, 1, step, step + 1, logits);
            AddTo(logits, bias);
            ToDataType(logits, FLOAT32);
            if (config.tool_call_grammar) {
                ApplyToolTokenMask(logits, 0, proposalConfig);
            }
            if (greedy) {
                Data top;
                TopK(logits, top, 1);
                top.ToDevice(DataDevice::CPU);
                previous = (int)((float *)top.cpuData)[0];
            }
#ifdef USE_CUDA
            else if (selection.cudaLogits) {
                FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
                AssertInFastLLM(FastllmCudaMtpSampleDraft((float *)logits.cudaData,
                    (float *)context.proposalProbs.cudaData + (size_t)step * vocab,
                    &proposalConfig.temperature, &proposalConfig.top_k, &proposalConfig.top_p, &previous, 1, vocab),
                    "Naive CUDA proposal sampling failed.");
            }
#endif
            else {
#ifdef USE_CUDA
                if (logits.dataDevice == DataDevice::CUDA &&
                    FastllmNaiveCanSelectLogits(vocab, 0, selection.count, false)) {
                    int count = std::min(selection.count, vocab);
                    FastllmCudaNaiveLogitsSelect(logits, 0, count, false, 1.0f, proposalPartial, proposalTop);
                    FastllmCudaCopyFromDeviceToHost(proposalCandidates.data(), proposalTop.cudaData, proposalTop.GetBytes());
                    q.push_back(SpeculativeTopKDistribution(proposalCandidates.data(), count, vocab, config));
                } else
#endif
                    q.push_back(round.Distribution(logits, 0, oldLength + step + 1, samplingTokens, proposalConfig));
                previous = SampleSpeculativeDistribution(q.back(), context.Uniform());
            }
        }
        proposed.push_back(previous);
        samplingTokens.Push(previous);
        if (config.tool_call_grammar) AdvanceToolCallGrammar(proposalConfig, previous);
    }
}

void NaiveN05FlashModel::PrepareDraftSamplingMemory(int count, int steps, int vocab, int latentWidth, int rank) {
#ifdef USE_CUDA
    if (rank < 0) ApplyDraftDevice();
    else FastllmCudaSetDevice(draftTP->devices.at(rank));
    auto &storage = rank < 0 ? draftProposalStorage : draftSamplingRanks.at(rank);
    const int device = FastllmCudaGetDevice(), words = (vocab + 31) / 32;
    if (!storage) {
        storage = std::make_shared<DraftWorkspace>();
        storage->device = device;
    }
    auto &shared = storage->sampling;
    const int rows = std::max(count, maxBatch > 0 ? maxBatch : 512);
    steps = std::max({1, steps, draftTokens});
    if (shared.rows >= rows && shared.steps >= steps && !shared.probabilities.dims.empty()) return;
    // Graph and eager/masked proposals execute serially and share this bounded
    // arena. Ragged verification groups have their own packing buffers.
    for (auto *variants : {&storage->proposals, &storage->bucketProposals}) {
        for (auto &entry : *variants) entry->sampling.Clear();
        variants->clear();
    }
    shared.Clear();
    shared.rows = rows; shared.steps = steps;
    auto allocate = [&](Data &data, DataType type, const std::vector<int> &dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, {device}, false); data.Allocate(false);
    };
    std::vector<float> temperatures(rows, 1.f), topPs(rows, 1.f);
    std::vector<int> topKs(rows, 0);
    shared.sampler = FastllmCudaMtpCreateDraftSamplerBatch(rows, vocab,
        temperatures.data(), topKs.data(), topPs.data());
    allocate(shared.ids, FLOAT32, {steps + 1, rows});
    allocate(shared.probabilities, FLOAT32, {rows, steps, vocab});
    allocate(shared.stepProbabilities, FLOAT32, {rows, vocab});
    allocate(shared.logits, FLOAT32, {rows, vocab});
    allocate(shared.base, BFLOAT16, {1, rows, vocab});
    allocate(shared.bias, BFLOAT16, {1, rows, vocab});
    allocate(shared.latent, BFLOAT16, {1, rows, latentWidth});
    allocate(shared.mask, INT32, {rows, words});
    allocate(shared.plainMask, INT32, {rows, words});
    shared.constraints = FastllmCudaCreateTokenConstraintBatch(rows, steps);
    allocate(shared.target, FLOAT32, {rows * (steps + 1), vocab});
    allocate(shared.verifyProposal, FLOAT32, {rows * steps, vocab});
    if (rank >= 0) {
        allocate(storage->samplingInput, BFLOAT16, {1, rows * steps, vocab});
        allocate(storage->samplingTarget, FLOAT32, {rows * (steps + 1), vocab});
    }
    std::vector<uint32_t> mask((size_t)rows * words, ~uint32_t(0));
    FastllmCudaCopyFromHostToDevice(shared.plainMask.cudaData, mask.data(), mask.size() * sizeof(uint32_t));
#endif
}

bool NaiveN05FlashModel::PrepareDraftBatchGraph(
        const std::vector<DraftRound *> &rounds, const Data &baseLogits, int rank, int reserveRows) {
#ifdef USE_CUDA
    const int live = rounds.size(), vocab = baseLogits.dims.back();
    const bool bucketed = draftTP && draftTP->dataParallel && !draftConfidenceThreshold &&
        std::none_of(rounds.begin(), rounds.end(), [](const DraftRound *r) { return bool(r->config.tool_call_grammar); });
    const int count = bucketed ? DraftGraphBucket(std::max(reserveRows, live),
        WindowRequestsPerGroup(maxBatch > 0 ? maxBatch : 512, draftTP->devices.size())) : live;
    const int steps = bucketed ? std::min(draftTokens, draftBlock) : rounds.front()->limit;
    int baseRows = 0;
    for (const auto *r : rounds) baseRows += r->limit;
    if (!GetFastllmEnv().cudaGraph || steps < 1 || vocab <= 0 || steps > INT_MAX / vocab ||
        baseLogits.dims[1] != baseRows || vocab >= (1 << 24)) return false;
    std::vector<float> temperatures;
    bool constrained = false;
    for (auto &owner : rounds) {
        const auto &r = *owner;
        if ((bucketed ? (r.limit <= 0 || r.limit > steps) : r.limit != steps) || DraftIsGreedy(r.config) || r.config.output_token_least > 0 ||
            (r.config.tool_call_token_mask && !r.config.tool_call_grammar) ||
            !SupportsCudaRejectionSampling(r.config)) return false;
        temperatures.push_back(r.config.temperature);
        constrained |= bool(r.config.tool_call_grammar);
    }
    PrepareDraftSamplingMemory(count, steps, vocab, weight["dspark.markov_head.markov_w1.weight"].dims.back(), rank);
    auto &storage = rank < 0 ? draftProposalStorage : draftSamplingRanks.at(rank);
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(storage->samplingLinear);
#endif
    const int device = FastllmCudaGetDevice(), words = (vocab + 31) / 32;
    Data &w1 = rank < 0 ? weight["dspark.markov_head.markov_w1.weight"] : storage->samplingW1;
    Data &w2 = rank < 0 ? weight["dspark.markov_head.markov_w2.weight"] : storage->samplingW2;
    w1.ToDevice(DataDevice::CUDA, std::vector<int>{device}); w2.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    auto &shared = storage->sampling;
    // Acceptance reuses this FP32 packing buffer after the proposal has
    // finished. A BF16 view gives the captured head a stable replay address.
    Data staged;
    const Data *base = &baseLogits;
    if (bucketed) {
        staged.FakeFrom(shared.verifyProposal, 0); staged.dataType = BFLOAT16; staged.UpdateUnitSize();
        staged.Resize({1, count * steps, vocab});
        const size_t bytes = size_t(vocab) * sizeof(uint16_t);
        if (baseRows == live * steps) {
            if (baseLogits.cudaData != staged.cudaData)
                FastllmCudaCopyFromDeviceToDevice(staged.cudaData, baseLogits.cudaData, size_t(baseRows) * bytes);
        } else {
            AssertInFastLLM(baseLogits.cudaData != staged.cudaData, "Ragged proposal input aliases its packed output");
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> sizes;
            int source = 0;
            for (int row = 0; row < live; ++row) {
                dst.push_back((char *)staged.cudaData + size_t(row) * steps * bytes);
                src.push_back((const char *)baseLogits.cudaData + size_t(source) * bytes);
                sizes.push_back(size_t(rounds[row]->limit) * bytes); source += rounds[row]->limit;
                if (rounds[row]->limit < steps)
                    FastllmCudaMemset0((char *)dst.back() + sizes.back(), (steps - rounds[row]->limit) * bytes);
            }
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                dst.data(), src.data(), sizes.data(), dst.size()), "Cannot pack ragged proposal logits");
        }
        if (live < count) FastllmCudaMemset0((char *)staged.cudaData + size_t(live) * steps * bytes,
            size_t(count - live) * steps * bytes);
        base = &staged;
    }
    std::vector<void *> inputs{w1.cudaData, w2.cudaData, base->cudaData};
    std::shared_ptr<DraftWorkspace> state;
    auto &variants = bucketed ? storage->bucketProposals : storage->proposals;
    auto found = std::find_if(variants.begin(), variants.end(),
        [&](const std::shared_ptr<DraftWorkspace> &s) {
            const auto &p = s->sampling;
            return s->device == device && s->inputs == inputs && p.rows == count && p.steps == steps && p.constrained == constrained &&
                p.base.dims == base->dims;
        });
    if (found != variants.end()) {
        state = *found; variants.erase(found);
    } else {
        state = std::make_shared<DraftWorkspace>(); state->device = device; state->inputs = inputs;
        auto &p = state->sampling;
        p.rows = count; p.steps = steps;
        p.constrained = constrained; p.constraints = shared.constraints;
        auto view = [](Data &dst, Data &src, const std::vector<int> &dims) {
            dst.FakeFrom(src, 0); dst.Resize(dims);
        };
        view(p.ids, shared.ids, {steps + 1, count});
        view(p.probabilities, shared.probabilities, {count, steps, vocab});
        view(p.logits, shared.logits, {count, vocab});
        view(p.bias, shared.bias, {1, count, vocab});
        view(p.latent, shared.latent, {1, count, w1.dims.back()});
        view(p.mask, constrained ? shared.mask : shared.plainMask, {count, words});
        // Proposals retain the full temperature-softmax distribution for rejection sampling.
        p.sampler = shared.sampler;
        p.ownsSampler = false;
        p.disabled = !p.sampler;
        p.graphs.resize(1);
    }
    variants.push_back(state);
    if (!bucketed && variants.size() > 4) variants.erase(variants.begin());
    auto &p = state->sampling;
    if (p.disabled) return false;
    if (constrained) {
        std::vector<std::shared_ptr<const TokenConstraintProgram>> programs;
        std::vector<int> states;
        for (auto &r : rounds) {
            if (r->config.tool_call_grammar) {
                auto config = r->config;
                PrepareToolCallConstraint(config);
                auto compiled = config.tool_call_grammar->Compile(config.tool_call_grammar_state);
                programs.push_back(compiled.first); states.push_back(compiled.second);
            } else { programs.push_back(nullptr); states.push_back(0); }
        }
        FastllmCudaPrepareTokenConstraintBatch(p.constraints.get(), programs, states, vocab);
    }
    temperatures.resize(count, 1.0f);
    p.base.FakeFrom(*base, 0); p.base.Resize(base->dims);
    std::vector<float> anchors;
    for (auto &r : rounds) anchors.push_back(r->anchor);
    anchors.resize(count, 0);
    auto prepareReplay = [&] {
        return FastllmCudaMtpPrepareDraftReplay(p.sampler, static_cast<float *>(p.ids.cudaData),
            count, anchors.data(), temperatures.data());
    };
    auto body = [&]() {
        for (int step = 0; step < steps; ++step) {
            Data ids(FLOAT32, {1, count}); ids.FakeFrom(p.ids, (size_t)step * count * sizeof(float));
            EmbeddingDirect(ids, w1, p.latent); DraftLinear(p.latent, w2, p.bias);
            if (p.constrained) FastllmCudaTokenConstraintMask(p.constraints.get(), step, (uint32_t *)p.mask.cudaData);
            float *probabilities = (float *)p.probabilities.cudaData + (size_t)step * vocab;
            float *idsOut = (float *)p.ids.cudaData + (size_t)(step + 1) * count;
            if (!FastllmCudaMtpDraftSamplerRunBF16BiasBatch(p.sampler, p.base.cudaData, p.bias.cudaData,
                    (const uint32_t *)p.mask.cudaData, step, steps, probabilities, idsOut,
                    (uint64_t)step * 4, steps * vocab)) {
                FastllmCudaNaiveBatchDraftSamplingLogits(p.base, p.bias, p.mask, p.logits, step, steps);
                AssertInFastLLM(FastllmCudaMtpDraftSamplerRunBatchStrided(p.sampler, (float *)p.logits.cudaData,
                    probabilities, idsOut, (uint64_t)step * 4, steps * vocab), "Graph batch draft sampling failed.");
            }
            if (p.constrained) FastllmCudaTokenConstraintAdvance(p.constraints.get(), step,
                idsOut, step == steps - 1);
        }
    };
    auto &graph = p.graphs.front();
    if (!graph.exec) {
        if (!prepareReplay()) return false;
        body();
        if (!graph.Capture(body, p.disabled)) return false;
    }
    if (!prepareReplay()) return false;
    AssertInFastLLM(FastllmCudaGraphLaunch(graph.exec), "Cannot replay batch draft sampling graph.");
    for (int row = 0; row < live; ++row) {
        auto &r = *rounds[row]; r.vocab = vocab; r.greedy = false;
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        r.proposalOwner = state;
        r.proposalOffset = size_t(row) * steps * vocab;
    }
    // Proposal width is known before sampling finishes. Keep a host mirror
    // for publication and eager fallback without blocking target preparation.
    if (rank >= 0 && rounds.front()->deferIds && FastllmCudaMtpReadDraftIdsAsync(
            (float *)p.ids.cudaData + count, size_t(steps) * count, p.hostIds, p.idsReady)) {
        for (int row = 0; row < live; ++row) {
            rounds[row]->proposalRow = row;
            rounds[row]->proposed.resize(rounds[row]->limit);
        }
        return true;
    }
    std::vector<float> tokens((size_t)steps * count);
    FastllmCudaCopyFromDeviceToHost(tokens.data(), (float *)p.ids.cudaData + count, tokens.size() * sizeof(float));
    for (int step = 0; step < steps; ++step) for (int row = 0; row < live; ++row) {
        if (step >= rounds[row]->limit) continue;
        const float token = tokens[(size_t)step * count + row];
        rounds[row]->proposed.push_back(CheckedDraftToken(token, vocab));
    }
    return true;
#else
    return false;
#endif
}


bool NaiveN05FlashModel::PrepareDraftBatch(std::vector<std::unique_ptr<DraftRound>> &rounds,
        Data &baseLogits, int reserveRows, bool *backboneReady) {
#ifdef USE_CUDA
    std::vector<int> anchors, limits;
    std::vector<DraftContext *> contexts;
    bool distributed = draftTP && draftTP->dataParallel && rounds.size() > 1 &&
        draftConfidenceThreshold == 0 && GetCudaEmbedding();
    int total = 0, steps = 0;
    for (const auto &round : rounds) {
        anchors.push_back(round->anchor); limits.push_back(round->limit); contexts.push_back(&round->context);
        distributed = distributed && round->context.deviceRank >= 0 &&
            round->context.deviceRank < (int)draftTP->devices.size() && round->limit > 0 &&
            (DraftIsGreedy(round->config) || SupportsCudaRejectionSampling(round->config));
        total += round->limit; steps = std::max(steps, round->limit);
    }
    if (distributed) {
        const auto &w1 = weight["dspark.markov_head.markov_w1.weight"];
        const auto &w2 = weight["dspark.markov_head.markov_w2.weight"];
        distributed = w1.dataType == BFLOAT16 && w2.dataType == BFLOAT16 && w1.dims.size() == 2 &&
            w1.dims[0] == weight["lm_head.weight"].dims[0] && w2.dims == w1.dims;
    }
    const bool sharded = distributed && CanShardDraftHead();
    Data hidden;
    DraftHeadShards headShards{limits, tpDraftHeadInputs, tpDraftHeadLogits};
    const bool ready = RunDraftBatch(anchors, contexts, hidden, reserveRows, sharded ? &headShards : nullptr);
    if (backboneReady) *backboneReady = ready;
    if (!ready) return false;
    if (!sharded) {
        Data headInput;
        GatherDraftHeadInput(hidden, limits, draftBlock, headInput);
        RunDraftHead(headInput, baseLogits);
        if (baseLogits.dims.empty()) return false;
        if (!distributed || baseLogits.dataType != BFLOAT16 || baseLogits.dataDevice != CUDA ||
            baseLogits.dims != std::vector<int>{1, total, weight["lm_head.weight"].dims[0]}) {
            std::vector<DraftRound *> local;
            local.reserve(rounds.size());
            for (auto &round : rounds) local.push_back(round.get());
            return PrepareDraftBatchLocal(local, baseLogits, -1);
        }
    }
    auto &w1 = weight["dspark.markov_head.markov_w1.weight"];
    auto &w2 = weight["dspark.markov_head.markov_w2.weight"];
    const int vocab = weight["lm_head.weight"].dims[0], ranks = draftTP->devices.size();
    ApplyDraftDevice();
    const std::vector<int> rootDevice{draftTP->devices.front()};
    w1.ToDevice(CUDA, rootDevice); w2.ToDevice(CUDA, rootDevice);
    // Joined logits and the first replication of Markov weights originate on the root stream.
    if (!sharded || draftSamplingRanks.empty()) FastllmCudaSyncCurrentThreadStream();
    draftSamplingRanks.resize(ranks);
    std::vector<std::vector<DraftRound *>> groups(ranks);
    std::vector<std::vector<int>> offsets(ranks);
    int offset = 0;
    for (auto &round : rounds) {
        const int rank = round->context.deviceRank;
        groups[rank].push_back(round.get()); offsets[rank].push_back(offset);
        offset += round->limit;
    }
    std::vector<const Data *> shards;
    if (sharded) for (auto &logits : tpDraftHeadLogits) shards.push_back(&logits);
    draftTP->Run([&](int rank) {
        // Reserve the configured worst-case batch on each owner: cancellation
        // or replacement may leave all live requests on a single device.
        PrepareDraftSamplingMemory(groups[rank].size(), steps, vocab, w1.dims.back(), rank);
        auto &storage = *draftSamplingRanks[rank];
        auto replicate = [&](Data &out, const Data &source) {
            if (out.cudaData) return;
            out.dataType = source.dataType; out.Resize(source.dims);
            out.ToDevice(CUDA, {draftTP->devices[rank]}, false); out.Allocate(false);
            const size_t bytes = source.GetBytes();
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                out.cudaData, bytes, source.cudaData, bytes, bytes, 1), "Cannot copy draft sampling weights.");
        };
        replicate(storage.samplingW1, w1); replicate(storage.samplingW2, w2);
        if (sharded) for (int peer = 0; peer < ranks; ++peer) if (peer != rank)
            FastllmCudaCurrentThreadStreamWaitEvent(draftBatchStorage->tpRanks[peer]->done);
        if (groups[rank].empty()) { FastllmCudaSyncCurrentThreadStream(); return; }
        std::vector<int> sourceRows;
        if (sharded) sourceRows.reserve(groups[rank].size() * size_t(steps));
        int rows = 0;
        for (size_t i = 0; i < groups[rank].size(); ++i) {
            const int n = groups[rank][i]->limit;
            if (sharded) for (int j = 0; j < n; ++j) sourceRows.push_back(offsets[rank][i] + j);
            rows += n;
        }
        const bool fixed = GetFastllmEnv().cudaGraph && std::all_of(groups[rank].begin(), groups[rank].end(),
            [&](const DraftRound *r) { return r->limit == std::min(draftTokens, draftBlock) &&
                !r->config.tool_call_grammar && !r->config.IsSimpleGreedy(); });
        Data &input = fixed ? storage.sampling.verifyProposal : storage.samplingInput;
        Data local;
        local.FakeFrom(input, 0); local.dataType = BFLOAT16; local.UpdateUnitSize(); local.Resize({1, rows, vocab});
        if (sharded && !FastllmCudaNaiveGatherDraftLogits(shards, tpDevices, total, sourceRows, local)) {
            // Unequal vocabulary shards and wider topologies keep the pitched copies.
            std::vector<void *> dst;
            std::vector<const void *> src;
            std::vector<size_t> dstPitch, srcPitch, heights;
            for (int shard = 0; shard < (int)tpDevices.size(); ++shard) {
                const auto range = tpVocabRanges[shard];
                const size_t width = range.second - range.first;
                int destRow = 0;
                for (size_t i = 0; i < groups[rank].size(); ++i) {
                    int n = groups[rank][i]->limit;
                    src.push_back((const uint16_t *)tpDraftHeadLogits[shard].cudaData +
                        size_t(offsets[rank][i]) * width);
                    dst.push_back((uint16_t *)input.cudaData + size_t(destRow) * vocab + range.first);
                    srcPitch.push_back(width * sizeof(uint16_t));
                    dstPitch.push_back(size_t(vocab) * sizeof(uint16_t));
                    heights.push_back(n);
                    destRow += n;
                }
            }
            AssertInFastLLM(FastllmCudaBatchCopy2DAsyncCurrentThread(dst.data(), src.data(),
                dstPitch.data(), srcPitch.data(), srcPitch.data(), heights.data(), dst.size()),
                "Cannot assemble draft logits on their sampling owner.");
        } else if (!sharded) {
            int destRow = 0;
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
            for (size_t i = 0; i < groups[rank].size(); ++i) {
                auto &r = *groups[rank][i];
                dst.push_back((uint16_t *)input.cudaData + (size_t)destRow * vocab);
                src.push_back((const uint16_t *)baseLogits.cudaData + (size_t)offsets[rank][i] * vocab);
                bytes.push_back((size_t)r.limit * vocab * sizeof(uint16_t)); destRow += r.limit;
            }
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                dst.data(), src.data(), bytes.data(), dst.size()), "Cannot distribute draft proposal logits.");
        }
        AssertInFastLLM(PrepareDraftBatchLocal(groups[rank], local, rank), "Cannot sample local draft batch.");
        // All consumers finish before this owner prepares another batch.
        // Allocate a fresh descriptor array on growth to avoid copying CUDA views.
        auto &views = storage.samplingTargetViews;
        if (views.size() < groups[rank].size()) views = std::vector<Data>(groups[rank].size());
        int targetRow = 0;
        for (size_t i = 0; i < groups[rank].size(); ++i) {
            auto *round = groups[rank][i];
            round->targetLogits = &views[i];
            round->samplingRank = rank;
            round->targetLogits->FakeFrom(storage.samplingTarget, (size_t)targetRow * vocab * sizeof(float));
            round->targetLogits->Resize({(int)round->proposed.size() + 1, vocab});
            targetRow += round->proposed.size() + 1;
        }
        if (groups[rank].front()->proposalRow < 0) FastllmCudaSyncCurrentThreadStream();
    });
    return true;
#else
    return false;
#endif
}

bool NaiveN05FlashModel::PrepareDraftBatchLocal(const std::vector<DraftRound *> &rounds,
        const Data &baseLogits, int rank) {
#ifdef USE_CUDA
    if (rounds.empty() || (rank < 0 && rounds.size() < 2) || draftConfidenceThreshold != 0 || baseLogits.dataType != BFLOAT16 ||
        baseLogits.dataDevice != DataDevice::CUDA || baseLogits.dims.size() != 3 || baseLogits.dims[0] != 1)
        return false;
    Data &w1 = rank < 0 ? weight["dspark.markov_head.markov_w1.weight"] : draftSamplingRanks.at(rank)->samplingW1;
    Data &w2 = rank < 0 ? weight["dspark.markov_head.markov_w2.weight"] : draftSamplingRanks.at(rank)->samplingW2;
    const int vocab = baseLogits.dims.back(), words = (vocab + 31) / 32;
    if (w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.dims.size() != 2 ||
        w1.dims[0] != vocab || w2.dims != w1.dims || !GetCudaEmbedding()) return false;
    if (PrepareDraftBatchGraph(rounds, baseLogits, rank)) return true;
    int total = 0, steps = 0;
    std::vector<int> offsets;
    std::vector<GenerationConfig> configs;
    for (auto &owner : rounds) {
        auto &r = *owner;
        if (r.limit < 1 ||
            (!DraftIsGreedy(r.config) && !SupportsCudaRejectionSampling(r.config))) return false;
        offsets.push_back(total); total += r.limit; steps = std::max(steps, r.limit);
        configs.push_back(r.config);
    }
    if (total != baseLogits.dims[1]) return false;
    PrepareDraftSamplingMemory(rounds.size(), steps, vocab, w1.dims.back(), rank);
    auto &storage = rank < 0 ? draftProposalStorage : draftSamplingRanks.at(rank);
    w1.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds); w2.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds);
    auto allocate = [&](Data &data, DataType type, std::vector<int> dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds, false); data.Allocate(false);
    };
    auto &shared = storage->sampling;
    for (size_t row = 0; row < rounds.size(); ++row) {
        auto &r = *rounds[row];
        r.vocab = vocab; r.greedy = DraftIsGreedy(r.config);
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        if (!r.greedy) {
            r.proposalOwner = storage;
            r.proposalOffset = size_t(row) * shared.steps * vocab;
        }
    }
    Data &base = shared.base, &latent = shared.latent, &bias = shared.bias;
    Data &logits = shared.logits, &probs = shared.stepProbabilities;
    for (int step = 0; step < steps; ++step) {
        std::vector<int> active;
        std::vector<float> ids, temperatures, topPs;
        std::vector<int> topKs;
        bool sampling = false, greedy = false;
        for (int i = 0; i < (int)rounds.size(); ++i) if (step < rounds[i]->limit) {
            auto &r = *rounds[i]; active.push_back(i);
            ids.push_back(step ? r.proposed.back() : r.anchor);
            temperatures.push_back(r.greedy ? 1.f : r.config.temperature);
            topPs.push_back(1.f); topKs.push_back(r.greedy ? 1 : 0);
            sampling |= !r.greedy; greedy |= r.greedy;
        }
        const int count = active.size();
        Data input(FLOAT32, {1, count}, ids), mask(INT32, {count, words});
        input.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds);
        // Keep the reused Markov buffer in BF16. Embedding's default FP32
        // output would widen it on the next step without growing its capacity.
        EmbeddingDirect(input, w1, latent);
        DraftLinear(latent, w2, bias);
        allocate(base, BFLOAT16, {1, count, vocab});
        std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
        mask.Allocate(); std::fill_n((uint32_t *)mask.cpuData, (size_t)count * words, ~uint32_t(0));
        for (int row = 0; row < count; ++row) {
            const int i = active[row]; auto &config = configs[i];
            if (config.tool_call_grammar) PrepareToolCallConstraint(config);
            if (config.tool_call_token_mask) {
                AssertInFastLLM(config.tool_call_token_mask->size() >= (size_t)words, "Invalid batch draft mask.");
                std::copy_n(config.tool_call_token_mask->data(), words, (uint32_t *)mask.cpuData + (size_t)row * words);
            }
            dst.push_back((uint16_t *)base.cudaData + (size_t)row * vocab);
            src.push_back((const uint16_t *)baseLogits.cudaData + (size_t)(offsets[i] + step) * vocab);
            bytes.push_back((size_t)vocab * sizeof(uint16_t));
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), count),
            "Cannot gather batch draft logits.");
        mask.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds);
        FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, logits);
        std::vector<int> tokens(count);
        // Compute greedy IDs before sampling: normalization may reuse logits.
        std::vector<int> greedyIds;
        if (greedy) {
            Data top; TopK(logits, top, 1); top.ToDevice(DataDevice::CPU);
            for (int row = 0; row < count; ++row) greedyIds.push_back((int)((float *)top.cpuData)[row * 2]);
        }
        if (sampling) {
            allocate(probs, FLOAT32, {count, vocab});
            AssertInFastLLM(FastllmCudaMtpSampleDraft((float *)logits.cudaData, (float *)probs.cudaData,
                temperatures.data(), topKs.data(), topPs.data(), tokens.data(), count, vocab), "Batch draft sampling failed.");
        }
        dst.clear(); src.clear(); bytes.clear();
        for (int row = 0; row < count; ++row) {
            const int i = active[row]; auto &r = *rounds[i];
            if (r.greedy) tokens[row] = greedyIds[row];
            else {
                dst.push_back(r.ProposalData() + (size_t)step * vocab);
                src.push_back((const float *)probs.cudaData + (size_t)row * vocab);
                bytes.push_back((size_t)vocab * sizeof(float));
            }
            r.proposed.push_back(tokens[row]);
            if (configs[i].tool_call_grammar) AdvanceToolCallGrammar(configs[i], tokens[row]);
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
            "Cannot retain batch proposal probabilities.");
    }
    return true;
#else
    return false;
#endif
}

#ifdef USE_CUDA
bool NaiveN05FlashModel::ProjectVerifyDraftRows(const std::vector<DraftRound *> &rounds,
        const std::vector<int> &group, int rank, int rows) {
    if (rank < 0 || !draftTP || !draftTP->dataParallel || !draftContextStorage ||
        group.empty() || rows < 2 || rows >= draftWindow || draftHeadDim % 2 || draftHeadDim > 256)
        return false;
    const int device = draftTP->devices[rank], width = embed_dim * draftTargetLayers.size();
    Data &fc = weight["dspark.fc.weight"], &norm = weight["dspark.hidden_norm.weight"];
    if (!fc.IsTensorParallelReplicated() || !norm.IsTensorParallelReplicated() || fc.dataType != BFLOAT16) return false;
    const auto &first = rounds[group[0]]->feature;
    bool contiguous = first.device == device;
    const size_t block = (size_t)rows * width;
    for (size_t i = 0; i < group.size(); ++i) {
        const auto &r = *rounds[group[i]];
        if (!r.feature.data || r.context.deviceRank != rank || r.context.kv.size() != (size_t)draftLayers)
            return false;
        contiguous &= r.feature.device == device && (uintptr_t)r.feature.data == (uintptr_t)first.data + i * block * sizeof(uint16_t);
    }
    auto &ws = *draftContextStorage->tpRanks.at(rank);
    const std::vector<int> shape{1, int(group.size()) * rows, width};
    Data view(BFLOAT16, shape);
    Data &input = contiguous ? view : ws.combined;
    if (contiguous) {
        view.cudaDataBorrowed = true;
        view.cudaData = const_cast<uint16_t *>(first.data);
        view.dataDevice = CUDA; view.dataDeviceIds = {device};
    } else {
        input.dataType = BFLOAT16; input.UpdateUnitSize(); input.Resize(shape);
        input.ToDevice(CUDA, std::vector<int>{device}, false); input.Allocate(false);
        std::vector<void *> dst;
        std::vector<const void *> src;
        std::vector<size_t> sizes(group.size(), block * sizeof(uint16_t));
        for (size_t i = 0; i < group.size(); ++i) {
            dst.push_back((uint16_t *)input.cudaData + i * block);
            src.push_back(rounds[group[i]]->feature.data);
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(),
            sizes.data(), group.size()), "Cannot gather local verification features");
    }
    DraftLinear(input, *fc.multiDeviceDatas.at(device), ws.projected);
    KimiK3RMSNorm(ws.projected, *norm.multiDeviceDatas.at(device), draftEps, ws.committedHidden);
    return ProjectDraftKVWeights(ws.committedHidden, device, draftHeads, draftKvHeads * draftHeadDim,
                                 ws.contextRaw, ws.contextPointers);
}
#endif

void NaiveN05FlashModel::PrepareDraftAcceptance(std::vector<std::unique_ptr<DraftRound>> &rounds, Data &logits) {
#ifdef USE_CUDA
    if (draftTP && draftTP->dataParallel && !rounds.empty() &&
        std::all_of(rounds.begin(), rounds.end(), [](const auto &r) { return r->samplingRank >= 0; })) {
        const int ranks = draftTP->devices.size(), vocab = rounds.front()->vocab;
        std::vector<std::vector<DraftRound *>> groups(ranks);
        std::vector<std::vector<int>> offsets(ranks);
        int offset = 0;
        for (auto &round : rounds) {
            groups[round->samplingRank].push_back(round.get());
            offsets[round->samplingRank].push_back(offset); offset += round->proposed.size() + 1;
        }
        // Attention DP writes directly to these destinations. Other target
        // layouts retain their normal gather and redistribute only at this boundary.
        if (!logits.dims.empty()) { ApplyDraftDevice(); FastllmCudaSyncCurrentThreadStream(); }
        draftTP->Run([&](int rank) {
            if (groups[rank].empty()) return;
            int rows = 0;
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
            for (size_t i = 0; i < groups[rank].size(); ++i) {
                auto &r = *groups[rank][i]; rows += r.proposed.size() + 1;
                if (!logits.dims.empty()) {
                    dst.push_back(r.targetLogits->cudaData);
                    src.push_back((const float *)logits.cudaData + (size_t)offsets[rank][i] * vocab);
                    bytes.push_back(r.targetLogits->GetBytes());
                }
            }
            AssertInFastLLM(dst.empty() || FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                dst.data(), src.data(), bytes.data(), dst.size()), "Cannot distribute target verification logits.");
            Data local;
            local.FakeFrom(draftSamplingRanks[rank]->samplingTarget, 0); local.Resize({rows, vocab});
            PrepareDraftAcceptanceLocal(groups[rank], local, rank);
            for (auto *r : groups[rank]) AssertInFastLLM(r->sampledAccepted >= 0, "Local rejection sampling incomplete.");
            FastllmCudaSyncCurrentThreadStream();
        });
        return;
    }
#endif
    std::vector<DraftRound *> all;
    for (auto &round : rounds) all.push_back(round.get());
    PrepareDraftAcceptanceLocal(all, logits, -1);
}

void NaiveN05FlashModel::PrepareDraftAcceptanceLocal(const std::vector<DraftRound *> &rounds, Data &logits, int rank) {
#ifdef USE_CUDA
    if (logits.dataDevice != DataDevice::CUDA || logits.dataType != FLOAT32 || rounds.empty()) return;
    const int vocab = logits.dims.back();
    int steps = 0;
    for (const auto &round : rounds) steps = std::max(steps, (int)round->proposed.size());
    PrepareDraftSamplingMemory(rounds.size(), steps, vocab,
        weight["dspark.markov_head.markov_w1.weight"].dims.back(), rank);
    auto &storage = rank < 0 ? draftProposalStorage : draftSamplingRanks.at(rank);
    auto &packing = storage->sampling;
    std::vector<int> offsets;
    std::map<std::pair<bool, int>, std::vector<int>> groups;
    auto owner = rounds.front()->proposalOwner;
    const bool deviceMasks = owner && owner->sampling.constrained && owner->sampling.rows == (int)rounds.size() &&
        owner->sampling.steps == steps && std::all_of(rounds.begin(), rounds.end(), [&](const auto &r) {
            return r->proposalOwner == owner && (int)r->proposed.size() == steps;
        });
    if (deviceMasks) FastllmCudaTokenConstraintApply(owner->sampling.constraints.get(), (float *)logits.cudaData, steps);
    int offset = 0;
    for (int i = 0; i < (int)rounds.size(); ++i) {
        auto &r = *rounds[i]; offsets.push_back(offset); offset += r.proposed.size() + 1;
        if (r.vocab != vocab || (!r.greedy && !r.selection.cudaLogits)) continue;
        r.MaskGreedyStops(logits, offsets.back());
        if (r.config.tool_call_grammar && !deviceMasks) {
            auto config = r.config;
            for (int row = 0; row <= (int)r.proposed.size(); ++row) {
                PrepareToolCallConstraint(config); ApplyToolTokenMask(logits, offsets[i] + row, config);
                if (row < (int)r.proposed.size()) AdvanceToolCallGrammar(config, r.proposed[row]);
            }
        }
        groups[{r.greedy, (int)r.proposed.size()}].push_back(i);
    }
    for (const auto &entry : groups) {
        const bool greedy = entry.first.first;
        const int drafts = entry.first.second, rows = drafts + 1, count = entry.second.size();
        // The sampler's small-batch algorithm has its own exact fallback.
        bool masked = !greedy && rank >= 0 && rounds[entry.second.front()]->feature.data &&
            !FastllmMtpUsesSmallRejection(count, drafts, vocab) && draftContextStorage && CanSlideDraftKV();
        // Launch the two shape-only projections while the host packs sampling
        // parameters and checks where each accepted prefix may be appended.
        if (masked) masked = ProjectVerifyDraftRows(rounds, entry.second, rank, rows);
        Data target(FLOAT32, {count * rows, vocab}), proposal(FLOAT32, {count * drafts, vocab});
        const int first = entry.second.front();
        bool targetView = true, proposalView = !greedy && rounds[first]->proposalOwner;
        for (int b = 0; b < count; ++b) {
            const int i = entry.second[b];
            targetView = targetView && offsets[i] == offsets[first] + b * rows;
            proposalView = proposalView && rounds[i]->proposalOwner == rounds[first]->proposalOwner &&
                rounds[i]->ProposalData() ==
                    rounds[first]->ProposalData() + (size_t)b * drafts * vocab;
        }
        auto borrow = [](Data &data, Data &backing) {
            auto shape = data.dims; data.FakeFrom(backing, 0); data.Resize(shape);
        };
        if (targetView) {
            target.FakeFrom(logits, (size_t)offsets[first] * vocab * sizeof(float));
            target.Resize({count * rows, vocab});
        } else borrow(target, packing.target);
        if (proposalView) {
            proposal.FakeFrom(rounds[first]->ProposalStorage(), rounds[first]->proposalOffset * sizeof(float));
            proposal.Resize({count * drafts, vocab});
        } else if (!greedy) borrow(proposal, packing.verifyProposal);
        std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
        std::vector<int> ids, ks, output(count * rows), accepted(count);
        std::vector<float> temperatures, ps;
        for (int b = 0; b < count; ++b) {
            const int i = entry.second[b]; auto &r = *rounds[i];
            if (!targetView) {
                dst.push_back((float *)target.cudaData + (size_t)b * rows * vocab);
                src.push_back((const float *)logits.cudaData + (size_t)offsets[i] * vocab);
                bytes.push_back((size_t)rows * vocab * sizeof(float));
            }
            if (!greedy) {
                if (!proposalView) {
                    dst.push_back((float *)proposal.cudaData + (size_t)b * drafts * vocab);
                    src.push_back(r.ProposalData()); bytes.push_back((size_t)drafts * vocab * sizeof(float));
                }
                ids.insert(ids.end(), r.proposed.begin(), r.proposed.end());
                temperatures.insert(temperatures.end(), rows, r.config.temperature);
                ps.insert(ps.end(), rows, r.config.top_p); ks.insert(ks.end(), rows, r.config.top_k);
            }
        }
        AssertInFastLLM(dst.empty() || FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
            "Cannot pack verification sampling batch.");
        const int kvWidth = draftKvHeads * draftHeadDim;
        if (masked) for (int i : entry.second) {
            auto &r = *rounds[i];
            if (!r.context.PrepareKVAppend(rows, draftWindow, draftWindow + draftBlock, kvWidth)) {
                masked = false;
                break;
            }
        }
        struct Write {
            Data *raw = nullptr;
            void *slots = nullptr;
            void *targetGraph = nullptr;
            int count, rows, layers, heads, dim;
            float eps, theta;
        } write{nullptr, nullptr, nullptr, count, rows, draftLayers, draftKvHeads, draftHeadDim, draftEps, draftTheta};
        std::vector<FastllmNaiveDraftKVSlot> slots;
        if (masked) {
            auto &ws = *draftContextStorage->tpRanks[rank];
            const auto &first = rounds[entry.second.front()]->feature;
            if (first.targetCommit && first.targetRows == count &&
                std::all_of(entry.second.begin(), entry.second.end(), [&](int i) {
                    const auto &r = *rounds[i];
                    return r.feature.device == draftTP->devices[rank] && r.feature.targetCommit == first.targetCommit &&
                           r.oldLength == r.context.committed;
                }))
                write.targetGraph = first.targetCommit;
            slots.reserve((size_t)draftLayers * count);
            for (int l = 0; l < draftLayers; ++l) {
                auto name = "dspark.layers." + std::to_string(l) + ".self_attn.k_norm.weight";
                const auto &norm = *weight[name].multiDeviceDatas.at(draftTP->devices[rank]);
                if (norm.dataType != FLOAT32 || norm.dims != std::vector<int>{draftHeadDim} || !norm.cudaData) {
                    masked = false; break;
                }
                for (int i : entry.second) {
                    auto &c = rounds[i]->context;
                    const auto &pair = c.kv[l];
                    const size_t offset = (size_t)pair.first.dims[1] * kvWidth;
                    slots.push_back({(uint16_t *)pair.first.cudaData + offset, (uint16_t *)pair.second.cudaData + offset,
                                     (const float *)norm.cudaData, c.committed, write.targetGraph ? rounds[i]->feature.targetPair : nullptr});
                }
            }
            // The FC output is dead after hidden norm, and its stream orders
            // that read before this descriptor upload. No extra GPU workspace.
            masked = masked && FastllmCudaNaivePrepareDraftKVSlots(slots, ws.projected);
            write.raw = &ws.contextRaw; write.slots = ws.projected.cudaData;
        }
        if (greedy) {
            Data top; TopK(target, top, 1); top.ToDevice(DataDevice::CPU);
            for (int b = 0; b < count; ++b) {
                auto &r = *rounds[entry.second[b]];
                int &keep = accepted[b];
                while (keep < drafts && r.proposed[keep] == (int)((float *)top.cpuData)[(b * rows + keep) * 2]) ++keep;
                output[b * rows + keep] = (int)((float *)top.cpuData)[(b * rows + keep) * 2];
            }
        } else if (masked) {
            auto append = [](void *opaque, const int *emitted) {
                auto &w = *static_cast<Write *>(opaque);
                FastllmCudaNaiveWriteAcceptedDraftKV(*w.raw, w.slots, emitted, w.count, w.rows, w.layers,
                                                     w.heads, w.dim, w.eps, w.theta);
                if (w.targetGraph)
                    AssertInFastLLM(FastllmCudaGraphLaunch(w.targetGraph), "Cannot replay local accepted SWA commit");
            };
            AssertInFastLLM(FastllmCudaMtpRejectionWithAccepted((float *)target.cudaData, (float *)proposal.cudaData,
                temperatures.data(), ks.data(), ps.data(), ids.data(), output.data(), accepted.data(),
                count, drafts, vocab, append, &write), "Device draft KV acceptance failed");
        } else AssertInFastLLM(FastllmCudaMtpRejectionSamplingTopKFirst((float *)target.cudaData, (float *)proposal.cudaData,
            temperatures.data(), ks.data(), ps.data(), ids.data(), output.data(), accepted.data(), count, drafts, vocab),
            "Batch rejection sampling failed.");
        for (int b = 0; b < count; ++b) {
            auto &r = *rounds[entry.second[b]];
            r.sampledAccepted = accepted[b]; r.sampledNext = output[b * rows + accepted[b]];
        }
        if (masked) for (int b = 0; b < count; ++b) {
            auto &r = *rounds[entry.second[b]];
            const int n = accepted[b] + 1;
            r.context.SlideKV(n, draftWindow, draftWindow + draftBlock, kvWidth);
            for (auto &pair : r.context.kv) {
                pair.first.dims[1] += n; pair.second.dims[1] += n;
            }
            r.context.committed += n;
            r.draftCommitted = true;
            r.targetPayloadWritten = write.targetGraph != nullptr;
            ClearDraftPrefixHidden(r.context.prefixHidden);
        }
    }
#endif
}

int NaiveN05FlashModel::FinishDraftRound(DraftRound &round, Data &logits,
        TargetCapture &capture, std::vector<std::pair<Data, Data>> &kv) {
    auto &context = round.context;
    const auto &config = round.config;
    auto &proposed = round.proposed;
    auto &q = round.q;
    auto &selection = round.selection;
    const int oldLength = round.oldLength, vocab = round.vocab;
    const bool greedy = round.greedy;
    // Row i predicts after the emitted anchor and proposals [0, i). Never
    // publish a speculative state: the scheduler advances only emitted tokens.
    std::vector<GenerationConfig> verifyConfigs;
    bool deviceMasks = false;
#ifdef USE_CUDA
    if (round.sampledAccepted < 0) round.MaskGreedyStops(logits, 0);
    if (round.sampledAccepted < 0 && round.proposalOwner && round.proposalOwner->sampling.constrained &&
            round.proposalOwner->sampling.rows == 1 && logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32) {
        ApplyDraftDevice();
        FastllmCudaTokenConstraintApply(round.proposalOwner->sampling.constraints.get(),
            (float *)logits.cudaData, proposed.size());
        deviceMasks = true;
    }
#endif
    if (round.sampledAccepted < 0 && config.tool_call_grammar && !deviceMasks) {
        verifyConfigs.resize(proposed.size() + 1);
        GenerationConfig stepConfig = config;
        for (int row = 0; row <= (int)proposed.size(); ++row) {
            PrepareToolCallConstraint(stepConfig);
            verifyConfigs[row] = stepConfig;
            ApplyToolTokenMask(logits, row, stepConfig);
            if (row < (int)proposed.size()) AdvanceToolCallGrammar(stepConfig, proposed[row]);
        }
    }
    ApplyDraftDevice();
    int accepted = round.sampledAccepted, next = round.sampledNext;
    if (accepted < 0) {
        accepted = 0;
#ifdef USE_CUDA
        if (!greedy && selection.cudaLogits) {
            AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
                logits.dataDeviceIds == round.ProposalStorage().dataDeviceIds,
                "Naive CUDA verification logits are on the wrong device.");
            FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
            const int rows = proposed.size() + 1;
            std::vector<float> temperatures(rows, config.temperature), topPs(rows, config.top_p);
            std::vector<int> topKs(rows, config.top_k), sampled(rows);
            AssertInFastLLM(FastllmCudaMtpRejectionSamplingTopKFirst((float *)logits.cudaData,
                round.ProposalData(), temperatures.data(), topKs.data(), topPs.data(),
                proposed.data(), sampled.data(), &accepted, 1, proposed.size(), vocab),
                "Naive CUDA rejection sampling failed.");
            next = sampled[accepted];
        } else
#endif
        if (greedy) {
            // For point-mass p and q, standard rejection sampling reduces exactly
            // to matching argmax tokens and emitting the target argmax on rejection.
            Data top;
            const float *values;
            if (!selection.candidates.dims.empty()) values = (float *)selection.candidates.cpuData;
            else {
                TopK(logits, top, 1);
                top.ToDevice(DataDevice::CPU);
                values = (float *)top.cpuData;
            }
            while (accepted < (int)proposed.size() && proposed[accepted] == (int)values[accepted * 2])
                ++accepted;
            next = (int)values[accepted * 2];
        } else {
            LastTokensUnit samplingTokens = round.CopyHistory();
            auto targetDistribution = [&](int row) {
                if (!selection.candidates.dims.empty())
                    return SpeculativeTopKDistribution((float *)selection.candidates.cpuData +
                        (size_t)row * selection.count * 2, selection.count, vocab, config);
                return round.Distribution(logits, row, oldLength + row + 1, samplingTokens, verifyConfigs.empty() ? config : verifyConfigs[row]);
            };
            while (accepted < (int)proposed.size()) {
                auto p = targetDistribution(accepted);
                if (!AcceptSpeculativeToken(proposed[accepted], p, q[accepted], context.Uniform())) {
                    next = SampleSpeculativeResidual(p, q[accepted], context.Uniform());
                    break;
                }
                samplingTokens.Push(proposed[accepted++]);
            }
            if (next < 0) {
                auto p = targetDistribution(accepted);
                next = SampleSpeculativeDistribution(p, context.Uniform());
            }
        }
    }
    int committed = accepted + 1;
    if (!round.targetCommitted) CommitTargetCache(kv, oldLength, committed, capture.ringVerify);
    if (!round.draftCommitted) CommitDraftContext(capture, committed, context, kv);
    return round.Publish(accepted, next);
}


std::vector<int> NaiveN05FlashModel::ForwardDraftBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
        std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits) {
    AssertInFastLLM(batch > 0 && (int)seqLens.size() == batch &&
        (int)configs.size() == batch && (int)positionIds.size() == batch &&
        (int)pastKeyValues.size() == batch * block_cnt &&
        std::all_of(seqLens.begin(), seqLens.end(), [](int n) { return n > 0; }),
        "DSpark batch expects complete nonempty request descriptors.");
    std::vector<std::vector<std::pair<Data, Data>> *> caches(batch, nullptr);
    std::vector<std::shared_ptr<DraftContext>> owners(batch);
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        std::unordered_map<const Data *, decltype(draftContexts)::iterator> byFirstKey;
        byFirstKey.reserve(draftContexts.size());
        for (auto it = draftContexts.begin(); it != draftContexts.end(); ++it)
            if (!it->first->empty()) byFirstKey.emplace(&it->first->front().first, it);
        for (int i = 0; i < batch; ++i) {
            auto found = byFirstKey.find(pastKeyValues[i * block_cnt].first);
            if (found == byFirstKey.end()) continue;
            auto &entry = *found->second;
            caches[i] = const_cast<std::vector<std::pair<Data, Data>> *>(entry.first);
            if (!entry.second) entry.second = CreateDraftContext(entry.first->front().first.dataDeviceIds);
            owners[i] = entry.second;
        }
    }
    // A warm one-token PD/chunked-prefill tail is still prefill. Its target
    // intentionally omits logits, so it must never enter speculative decode.
    bool prefill = draftKVOnly || isIntermediateChunkedPrefill;
    for (int i = 0; i < batch; ++i) {
        AssertInFastLLM(caches[i] && owners[i] && positionIds[i],
                       "DSpark request has no registered draft context.");
        const auto &cache = *caches[i];
        const auto first = pastKeyValues.begin() + size_t(i) * block_cnt;
        AssertInFastLLM(cache.size() == size_t(block_cnt) &&
            std::equal(cache.begin(), cache.end(), first, [](const auto &layer, const auto &binding) {
                return binding.first == &layer.first && binding.second == &layer.second;
            }), "DSpark batch mixed request KV descriptors.");
        prefill |= configs[i].prefill_chunk_remaining >= 0 || seqLens[i] != 1 || owners[i]->committed == 0 || owners[i]->kv.empty();
    }
    if (prefill) {
        // Preserve the scheduler's coalescing target while reusing KVMem's
        // existing bounded prefill workspace. All projections/MoE still batch
        // within each group; oversized single requests use the normal path.
        if (kvMemConfig.enabled && batch > 1 && inputIds.Count(0) > (size_t)GetChunkedPrefillSize()) {
            std::vector<int> tokens;
            int offset = 0;
            for (int first = 0; first < batch;) {
                int end = first + 1, count = seqLens[first];
                while (end < batch && count + seqLens[end] <= GetChunkedPrefillSize())
                    count += seqLens[end++];
                Data input;
                Split(inputIds, 1, offset, offset + count, input);
                std::vector<Data *> positions(positionIds.begin() + first, positionIds.begin() + end);
                std::vector<int> lengths(seqLens.begin() + first, seqLens.begin() + end);
                std::vector<GenerationConfig> groupConfigs(configs.begin() + first, configs.begin() + end);
                std::vector<std::pair<Data *, Data *>> groupKV(pastKeyValues.begin() + first * block_cnt,
                    pastKeyValues.begin() + end * block_cnt);
                LastTokensManager history;
                std::vector<std::vector<float> *> outputs;
                for (int i = first; i < end; ++i) {
                    history.units.push_back(i < (int)lastTokens.units.size()
                        ? lastTokens.units[i] : LastTokensUnit(configs[i].last_n));
                    outputs.push_back(retLogits && i < (int)retLogits->size() ? (*retLogits)[i] : nullptr);
                }
                auto result = ForwardDraftBatch(end - first, input, positions, lengths, groupKV,
                    groupConfigs, history, retLogits ? &outputs : nullptr);
                tokens.insert(tokens.end(), result.begin(), result.end());
                offset += count; first = end;
            }
            return tokens;
        }

        if (batch == 1) {
            LastTokensManager history;
            history.units.push_back(lastTokens.units.empty()
                ? LastTokensUnit(configs[0].last_n) : lastTokens.units[0]);
            return {ForwardDraft(inputIds, *positionIds[0], *caches[0], configs[0],
                history, retLogits && !retLogits->empty() ? (*retLogits)[0] : nullptr)};
        }
        // A one-token cold prompt is also prefill. The scheduler separates
        // phases; direct callers with queued speculative output retain the
        // ordinary per-request path instead of overwriting pending state.
        if (tpDevices.size() <= 1 || std::any_of(owners.begin(), owners.end(),
                [](const auto &owner) { return !owner->pending.empty(); })) {
            std::vector<int> tokens(batch);
            int offset = 0;
            for (int i = 0; i < batch; offset += seqLens[i++]) {
                Data input;
                Split(inputIds, 1, offset, offset + seqLens[i], input);
                LastTokensManager history;
                history.units.push_back(i < (int)lastTokens.units.size()
                    ? lastTokens.units[i] : LastTokensUnit(configs[i].last_n));
                tokens[i] = ForwardDraft(input, *positionIds[i], *caches[i], configs[i], history,
                    retLogits && i < (int)retLogits->size() ? (*retLogits)[i] : nullptr);
            }
            return tokens;
        }
        TargetCapture capture;
        capture.borrowOutputs = true;
        capture.collectHidden = false;
        std::vector<bool> collect(batch);
        for (int i = 0; i < batch; ++i) {
            const auto &key = caches[i]->front().first;
            const int past = key.dims.empty() ? 0 : key.dims[1];
            AssertInFastLLM(owners[i]->committed == past, "Naive target/draft cache length mismatch.");
            const int start = std::max(0, configs[i].input_token_length - draftWindow - GetPageLen() + 1);
            collect[i] = past + seqLens[i] > start;
            capture.collectHidden |= collect[i];
        }
        auto tokens = ForwardTargetBatch(batch, inputIds, {}, positionIds, seqLens,
            pastKeyValues, configs, lastTokens, retLogits, &capture);
        int offset = 0;
        for (int i = 0; i < batch; offset += seqLens[i++]) {
            if (!collect[i]) {
                owners[i]->committed += seqLens[i];
                ClearDraftPrefixHidden(owners[i]->prefixHidden);
                continue;
            }
            // Reuse the bounded single-request commit, including normalized
            // prefix checkpoints and the generic draft projection fallback.
            // Views borrow this prefill's features until all commits finish.
            TargetCapture features;
            for (int layer : draftTargetLayers) {
                const Data &source = capture.hidden.at(layer);
                auto &view = features.hidden[layer];
                view.FakeFrom(source, (size_t)offset * embed_dim * source.unitSize);
                view.Resize({1, seqLens[i], embed_dim});
            }
            if (i < (int)capture.prefixes.size()) features.prefixes = {capture.prefixes[i]};
            CommitDraftContext(features, seqLens[i], *owners[i], *caches[i]);
        }
        return tokens;
    }
    Data ids(inputIds);
    ids.ToDevice(DataDevice::CPU);
    AssertInFastLLM(ids.dataType == FLOAT32 && ids.Count(0) == (size_t)batch,
                   "DSpark batch token count mismatch.");
    std::vector<int> anchors(batch), result(batch);
    for (int i = 0; i < batch; ++i) {
        anchors[i] = (int)((float *)ids.cpuData)[i];
        if (!owners[i]->pending.empty())
            AssertInFastLLM(anchors[i] == owners[i]->pending.front().first,
                           "DSpark batch output queue is out of sync.");
    }
    auto history = [&](int i) {
        return i < (int)lastTokens.units.size() ? lastTokens.units[i] : LastTokensUnit(configs[i].last_n);
    };
    auto single = [&](int i) {
        Data input(FLOAT32, {1, 1}, {(float)anchors[i]});
        LastTokensManager tokens;
        tokens.units.push_back(history(i));
        return ForwardDraft(input, *positionIds[i], *caches[i], configs[i], tokens,
            retLogits && i < (int)retLogits->size() ? (*retLogits)[i] : nullptr);
    };
    const bool needRound = std::any_of(owners.begin(), owners.end(),
        [](const auto &owner) { return owner->pending.empty(); });
    std::vector<int> selected, limits;
    // Requests drain one token per scheduler turn but accept different amounts
    // per round. Allow one batch worth of bounded lookahead to absorb that
    // skew, while retaining the existing two-round limit for small batches.
    const size_t pendingLimit = (size_t)draftTokens * std::max(2, batch);
    if (needRound && tpDevices.size() > 1 && draftConfidenceThreshold == 0) {
        for (int i = 0; i < batch; ++i) {
            const auto &config = configs[i];
            const auto &context = *owners[i];
            bool terminal = std::any_of(context.pending.begin(), context.pending.end(), [&](const auto &item) {
                return item.second == eos_token_id || eos_token_ids.count(item.second) ||
                    config.stop_token_ids.count(item.second);
            });
            // Bound lookahead before queued tokens are emitted. A grammar owns
            // immutable transitions, so only its request-local cursor advances.
            if (!context.committed || context.kv.empty() || terminal ||
                context.pending.size() > pendingLimit ||
                config.output_logits ||
                (!config.tool_call_grammar && (config.tool_call_token_mask ||
                    config.tool_call_name_constraint_enabled ||
                    config.tool_call_parameter_name_constraint_enabled ||
                    config.tool_call_content_sampling_enabled)) ||
                (!DraftIsGreedy(config) && !SupportsCudaRejectionSampling(config))) continue;
            int limit = std::min(draftTokens, max_positions - context.committed - 1);
            if (config.output_token_limit > 0)
                limit = std::min(limit, config.output_token_limit -
                    (context.committed - config.input_token_length) - 1);
            if (limit > 0) { selected.push_back(i); limits.push_back(limit); }
        }
    }
    if (selected.size() < 2) {
        for (int i = 0; i < batch; ++i) result[i] = single(i);
        return result;
    }
    std::vector<std::unique_ptr<DraftRound>> rounds;
    rounds.reserve(selected.size());
#ifdef USE_CUDA
    const bool deferIds = attentionDP > 1 && tpServingState && !GetLowMemMode() &&
        draftTP && draftTP->dataParallel && draftTP->devices == tpDevices;
#endif
    // Only grammar cursors advance privately. Other rounds borrow the caller's config.
    std::deque<GenerationConfig> grammarConfigs;
    const bool fullBatch = selected.size() == size_t(batch);
    std::vector<int> pending, lengths;
    std::vector<float> verifyIds, verifyPositions;
    std::vector<GenerationConfig> verifyConfigs;
    std::vector<std::pair<Data *, Data *>> verifyCaches;
    if (!fullBatch) {
        verifyConfigs.reserve(selected.size());
        verifyCaches.reserve(selected.size() * block_cnt);
    }
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row];
        auto &context = *owners[i];
        const GenerationConfig *config = &configs[i];
        if (config->tool_call_grammar) {
            auto &cursor = grammarConfigs.emplace_back(*config);
            for (const auto &item : context.pending) AdvanceToolCallGrammar(cursor, item.second);
            PrepareToolCallConstraint(cursor);
            config = &cursor;
        }
        int anchor = context.pending.empty() ? anchors[i] : context.pending.back().second;
        rounds.emplace_back(new DraftRound(*this, context, *config, anchor, limits[row]));
#ifdef USE_CUDA
        rounds.back()->deferIds = deferIds;
#endif
        if (config->NeedsTokenHistory()) {
            auto &tokens = rounds.back()->lastTokens.emplace(history(i));
            for (const auto &item : context.pending) tokens.Push(item.second);
        }
    }
    Data headLogits;
    const bool preparedBatch = PrepareDraftBatch(rounds, headLogits);
    size_t headOffset = 0;
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row];
        auto &context = *owners[i];
        auto &round = *rounds[row];
        if (!preparedBatch) {
            Data head;
            if (!headLogits.dims.empty()) {
                head.dataType = headLogits.dataType;
                head.Resize({1, limits[row], headLogits.dims.back()});
                head.FakeFrom(headLogits, headOffset);
                headOffset += head.GetBytes();
            }
            PrepareDraftRound(round, head.dims.empty() ? nullptr : &head);
        }
        // Keep full per-row logits for each request's own rejection sampler.
        round.selection.count = 0;
        round.selection.cudaSampling = false;
        round.selection.cudaLogits = true;
        pending.push_back(context.pending.size());
        lengths.push_back(round.proposed.size() + 1);
        verifyIds.push_back((float)round.anchor);
        for (int token : round.proposed) verifyIds.push_back((float)token);
        for (int j = 0; j < lengths.back(); ++j)
            verifyPositions.push_back((float)(context.committed + j));
        if (!fullBatch) {
            verifyConfigs.push_back(configs[i]);
            auto first = pastKeyValues.begin() + size_t(i) * block_cnt;
            verifyCaches.insert(verifyCaches.end(), first, first + block_cnt);
        }
    }
    Data input(FLOAT32, {1, (int)verifyIds.size()}, verifyIds);
    Data positions(FLOAT32, {1, (int)verifyPositions.size()}, verifyPositions);
    // Selection preserves scheduler order. Full batches can borrow the validated
    // descriptors for this forward, including after requests have changed slots.
    TargetBatch requests{fullBatch ? pastKeyValues : verifyCaches,
                         fullBatch ? configs : verifyConfigs, lengths, block_cnt};
#ifdef USE_CUDA
    DraftVerifyInput proposals;
    bool materialized = false;
    for (const auto &round : rounds) {
        if (round->proposalRow < 0) continue;
        const auto &p = round->proposalOwner->sampling;
        // Each sampling owner records one event for its whole group.
        if (round->proposalRow == 0) proposals.ready.push_back(p.idsReady);
        proposals.rows.push_back((const float *)p.ids.cudaData + round->proposalRow);
        proposals.strides.push_back(p.rows);
    }
    if (!proposals.ready.empty()) {
        proposals.materialize = [&] {
            if (materialized) return;
            for (auto event : proposals.ready) FastllmCudaEventSynchronize(event);
            size_t offset = 0;
            auto *ids = (float *)input.cpuData;
            for (auto &r : rounds) {
                if (r->proposalRow >= 0) {
                    const auto &p = r->proposalOwner->sampling;
                    for (size_t step = 0; step < r->proposed.size(); ++step) {
                        const float id = p.hostIds[step * p.rows + r->proposalRow];
                        r->proposed[step] = CheckedDraftToken(id, r->vocab);
                        ids[offset + 1 + step] = id;
                    }
                }
                offset += r->proposed.size() + 1;
            }
            materialized = true;
        };
        // Draft DP established peer access for this device domain. Target DP
        // consumes global scheduler order even when owners are interleaved.
        if (proposals.rows.size() != rounds.size()) proposals.rows.clear();
        requests.proposals = &proposals;
    }
#endif
    TargetCapture capture;
    // All acceptance and KV commits below complete before another target
    // forward can overwrite the stable graph/logits workspace.
    capture.borrowOutputs = true;
    capture.consumePacked = true;
    LogitsSelection selection;
    selection.cudaLogits = true;
    capture.verifying = true;
    capture.selection = &selection;
    if (attentionDP > 1 && std::all_of(rounds.begin(), rounds.end(),
            [](const auto &r) { return r->samplingRank >= 0; }))
        for (auto &round : rounds) capture.logitsDestinations.push_back(round->targetLogits);
    std::vector<std::pair<Data, Data>> unused;
    Data logits = RunDraftTarget(input, positions, unused, requests.configs.front(), capture, &requests);
    if (requests.proposals) requests.proposals->materialize();
    const int vocab = rounds.front()->vocab;
    int offset = 0;
#ifdef USE_CUDA
    // Commit on the sampler stream only for a complete, homogeneous set of
    // owner-local rows. A shrinking batch, mixed proposal widths or a changed
    // binding keeps its single coordinated cache commit on the normal path.
    if (fullBatch && capture.packedRows.size() == rounds.size()) {
        bool local = true;
        for (size_t i = 0; local && i < rounds.size(); ++i) {
            const auto &feature = capture.packedRows[i];
            const auto &r = *rounds[i];
            local = !r.greedy && r.samplingRank >= 0 && r.samplingRank == r.context.deviceRank &&
                feature.targetCommit && feature.targetRows > kFastllmMtpSmallRejectionMaxBatch &&
                r.oldLength == r.context.committed &&
                feature.device == tpDevices.at(r.samplingRank) && lengths[i] == lengths.front();
        }
        if (local)
            for (size_t i = 0; i < rounds.size(); ++i) rounds[i]->feature = capture.packedRows[i];
    }
#endif
    PrepareDraftAcceptance(rounds, logits);
    if (std::all_of(rounds.begin(), rounds.end(), [](const auto &r) { return r->sampledAccepted >= 0; })) {
        std::vector<int> past, counts;
        std::vector<char> written;
        std::vector<std::vector<std::pair<Data, Data>> *> commitCaches;
        for (size_t row = 0; row < selected.size(); ++row) {
            past.push_back(rounds[row]->oldLength); counts.push_back(rounds[row]->sampledAccepted + 1);
            commitCaches.push_back(caches[selected[row]]);
            written.push_back(rounds[row]->targetPayloadWritten);
        }
        CommitTargetCaches(commitCaches, past, counts, capture.ringVerify, &written);
        std::vector<DraftContext *> contexts;
        for (auto &round : rounds) {
            round->targetCommitted = true;
            if (!round->draftCommitted) contexts.push_back(&round->context);
        }
        if (contexts.size() == rounds.size() && CommitDraftContexts(capture, lengths, counts, contexts))
            for (auto &round : rounds) round->draftCommitted = true;
    }
    if (std::any_of(rounds.begin(), rounds.end(), [](const auto &r) { return !r->draftCommitted; }))
        UnpackDraftFeatures(capture, lengths);
    std::vector<bool> handled(batch, false);
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row], count = lengths[row];
        auto &context = *owners[i];
        auto &round = *rounds[row];
        int token;
        if (round.sampledAccepted >= 0 && round.targetCommitted && round.draftCommitted) {
            token = round.Publish(round.sampledAccepted, round.sampledNext);
        } else {
            Data rows(FLOAT32, {count, vocab});
            if (rounds[row]->samplingRank >= 0) rows.FakeFrom(*rounds[row]->targetLogits, 0);
            else rows.FakeFrom(logits, (size_t)offset * vocab * sizeof(float));
            TargetCapture features;
            features.verifying = true;
            features.ringVerify = capture.ringVerify;
            if (!rounds[row]->draftCommitted) for (int layer : draftTargetLayers)
                Split(capture.hidden.at(layer), 1, offset, offset + count, features.hidden[layer]);
            token = FinishDraftRound(*rounds[row], rows, features, *caches[i]);
        }
        if (pending[row]) {
            // Bridge the old queue's final token to the freshly prepared block.
            context.pending.insert(context.pending.begin() + pending[row],
                                   {rounds[row]->anchor, token});
            token = context.pending.front().second;
            context.pending.pop_front();
        }
        result[i] = token;
        handled[i] = true;
        offset += count;
    }
    for (int i = 0; i < batch; ++i) if (!handled[i]) result[i] = single(i);
    return result;
}

bool NaiveN05FlashModel::PopPendingResponseToken(ResponseContext *response,
        int previousToken, int &token) {
    if (!draftEnabled || !response || response->generationConfig.output_logits) return false;
    std::lock_guard<std::mutex> guard(historyMutex);
    auto it = draftContexts.find(&response->pastKeyValues);
    if (it == draftContexts.end() || !it->second || it->second->pending.empty()) return false;
    auto &pending = it->second->pending;
    AssertInFastLLM(pending.front().first == previousToken,
                   "DSpark published token chain is out of sync.");
    token = pending.front().second;
    pending.pop_front();
    return true;
}

int NaiveN05FlashModel::ForwardDraft(
        const Data &inputIds, const Data &positions, std::vector<std::pair<Data, Data>> &kv,
        const GenerationConfig &config, const LastTokensManager &lastTokens,
        std::vector<float> *retLogits) {
    std::shared_ptr<DraftContext> owner;
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        auto &entry = draftContexts[&kv];
        if (!entry) entry = kv.empty() ? CreateDraftContext() : CreateDraftContext(kv.front().first.dataDeviceIds);
        owner = entry;
    }
    auto &context = *owner;
    Data ids;
    ids.CopyFrom(inputIds);
    ids.ToDevice(DataDevice::CPU);
    const int anchor = (int)((float *)ids.cpuData)[0];
    if (isIntermediateChunkedPrefill && !context.pending.empty())
        throw std::logic_error("Naive prefill cannot consume pending speculative output.");
    if (!context.pending.empty()) {
        AssertInFastLLM(inputIds.dims[1] == 1 && anchor == context.pending.front().first,
                        "Naive speculative output queue is out of sync.");
        int token = context.pending.front().second;
        context.pending.pop_front();
        return token;
    }
    int oldLength = kv[0].first.dims.empty() ? 0 : kv[0].first.dims[1];
    AssertInFastLLM(context.committed == oldLength, "Naive target/draft cache length mismatch.");
    // Legacy name/content constraints retain their single-token fallback.
    // MiMo grammar below advances a private state for every proposal/verify row.
    bool constrained = config.tool_call_name_constraint_enabled ||
                       config.tool_call_parameter_name_constraint_enabled ||
                       config.tool_call_content_sampling_enabled;
    int limit = std::min(draftTokens, max_positions - oldLength - 1);
    if (config.output_token_limit > 0)
        limit = std::min(limit, config.output_token_limit - (oldLength - config.input_token_length) - 1);
    if (draftKVOnly || isIntermediateChunkedPrefill || config.prefill_chunk_remaining >= 0 || context.committed == 0 || context.kv.empty() ||
        inputIds.dims[1] != 1 || config.output_logits || constrained || limit <= 0) {
        TargetCapture capture;
        // Only the final draft window is used; there is no CPU prefix archive.
        // Preserve absolute position accounting while skipping invisible
        // prompt chunks' feature copies, projection and draft KV construction.
        const int draftStart = std::max(0, config.input_token_length - draftWindow - GetPageLen() + 1);
        capture.collectHidden = oldLength + inputIds.dims[1] > draftStart;
        capture.borrowOutputs = true;
        auto selection = SelectLogits(config);
        capture.selection = &selection;
        Data logits = RunDraftTarget(inputIds, positions, kv, config, capture);
        if (capture.collectHidden) CommitDraftContext(capture, inputIds.dims[1], context, kv);
        else {
            context.committed += inputIds.dims[1];
            ClearDraftPrefixHidden(context.prefixHidden);
        }
        if (isIntermediateChunkedPrefill || config.prefill_chunk_remaining > 0) return 0;
        return SampleTarget(logits, kv, config, lastTokens, retLogits, &selection);
    }
    DraftRound round(*this, context, config, anchor, limit);
    if (config.NeedsTokenHistory() && !lastTokens.units.empty())
        round.lastTokens.emplace(lastTokens.units[0]);
    round.standalone = true;
    PrepareDraftRound(round);
    auto &proposed = round.proposed;
    auto &selection = round.selection;
    std::vector<float> verifyIds{(float)anchor}, verifyPositions;
    for (int token : proposed) verifyIds.push_back((float)token);
    // Confidence may shorten proposals while an eight-row graph is ready.
    // Causal padding preserves every useful logit. Acceptance/commit below
    // remain bounded by proposed.size(), and limit already checks both the
    // remaining context and output budget. Unsupported backends stay eager.
    if (limit == 7 && HasVerificationGraph()) verifyIds.resize(8, (float)anchor);
    for (int i = 0; i < (int)verifyIds.size(); ++i) verifyPositions.push_back(oldLength + i);
    Data verifyInput(FLOAT32, {1, (int)verifyIds.size()}, verifyIds);
    Data verifyPos(FLOAT32, {1, (int)verifyIds.size()}, verifyPositions);
    TargetCapture capture;
    capture.verifying = true;
    capture.selection = &selection;
    Data logits = RunDraftTarget(verifyInput, verifyPos, kv, config, capture);
    return FinishDraftRound(round, logits, capture, kv);
}

void NaiveN05FlashModel::PrepareDraftServingMemory(bool history) {
#ifdef USE_CUDA
    const int batch = maxBatch > 0 ? maxBatch : 512;
    if (!draftEnabled || !PrepareDraftTP()) return;
    const int steps = std::max(1, std::min(draftTokens, draftBlock)), verifyRows = steps + 1;
    ApplyDraftDevice();
    Data hidden(BFLOAT16, {1, draftWindow - 1, embed_dim});
    hidden.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); hidden.Allocate(true);
    std::vector<std::shared_ptr<DraftContext>> owners;
    std::vector<DraftContext *> contexts;
    for (int i = 0; i < batch; ++i) {
        auto context = CreateDraftContext();
        AppendDraftContext(hidden, 0, *context);
        if (history) {
            auto &tail = context->workspace->prefixStorage;
            tail.dataType = BFLOAT16; tail.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            tail.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); tail.Allocate(false);
        }
        contexts.push_back(context.get()); owners.push_back(std::move(context));
    }
    auto warmPrefillCommit = [&]() {
        TargetCapture capture;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        std::vector<std::pair<Data, Data>> unused;
        CommitDraftContext(capture, draftWindow + GetPageLen() - 1, *contexts.front(), unused);
    };
    if (pdPrefillOnly) {
        warmPrefillCommit();
        idleDraftContexts = std::move(owners);
        return;
    }
    const int ranks = draftTP->devices.size();
    const int localLimit = draftTP->dataParallel ? WindowRequestsPerGroup(batch, ranks) : 0;
    const auto buckets = localLimit && GetFastllmEnv().cudaGraph ? SpeculativeGraphBuckets(localLimit) : std::vector<int>{};
    bool backboneReady = false;
    const int vocab = weight["lm_head.weight"].dims[0];
    Data head;
    std::vector<GenerationConfig> configs(batch);
    std::vector<std::unique_ptr<DraftRound>> rounds;
    for (int i = 0; i < batch; ++i) {
        configs[i].temperature = 1.0f; configs[i].top_p = 0.95f;
        configs[i].top_k = vocab;
        rounds.emplace_back(new DraftRound(*this, *contexts[i], configs[i], 1, steps));
    }
    if (PrepareDraftBatch(rounds, head, localLimit, &backboneReady)) {
        tpSamplingLogits.Resize({batch * verifyRows, vocab});
        tpSamplingLogits.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false);
        tpSamplingLogits.Allocate(false);
        FastllmCudaMemset0(tpSamplingLogits.cudaData, tpSamplingLogits.GetBytes());
        PrepareDraftAcceptance(rounds, tpSamplingLogits);
    }
    if (backboneReady) {
        if (!buckets.empty() && !draftConfidenceThreshold && !draftSamplingRanks.empty()) {
            // Materialize each owner's stable proposal graphs while the weights
            // and pools are warm. Subsequent admission only uploads live inputs.
            std::vector<std::vector<DraftRound *>> groups(ranks);
            for (auto &round : rounds) groups.at(round->context.deviceRank).push_back(round.get());
            draftTP->Run([&](int rank) {
                auto &rows = groups[rank];
                if (rows.empty()) return;
                auto &storage = *draftSamplingRanks[rank];
                for (int bucket : buckets) {
                    const int n = std::min<int>(rows.size(), bucket);
                    std::vector<DraftRound *> selected(rows.begin(), rows.begin() + n);
                    Data staged;
                    staged.FakeFrom(storage.samplingInput, 0); staged.Resize({1, n * steps, vocab});
                    FastllmCudaMemset0(staged.cudaData, staged.GetBytes());
                    PrepareDraftBatchGraph(selected, staged, rank, bucket);
                }
            });
            ApplyDraftDevice();
        }
        TargetCapture capture;
        capture.verifying = true;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, batch * verifyRows, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        CommitDraftContexts(capture, std::vector<int>(batch, verifyRows), std::vector<int>(batch, verifyRows), contexts);
        capture.hidden.clear();
        warmPrefillCommit();
    }
    if (!buckets.empty()) {
        std::vector<std::vector<DraftContext *>> groups(ranks);
        for (auto *context : contexts) groups.at(context->deviceRank).push_back(context);
        for (int bound : buckets) {
            if (backboneReady && bound == localLimit) break; // Already warmed by the full backbone run.
            std::vector<DraftContext *> sample;
            for (const auto &group : groups)
                sample.insert(sample.end(), group.begin(), group.begin() + std::min<size_t>(bound, group.size()));
            if (!sample.empty()) {
                Data result;
                RunDraftBatch(std::vector<int>(sample.size(), 1), sample, result);
            }
        }
    }
    idleDraftContexts = std::move(owners);
#endif
}

}
