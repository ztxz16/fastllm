#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "json11.hpp"
#include "executor.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <numeric>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
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
}

#ifdef USE_CUDA
struct NaiveN05FlashModel::DraftWorkspace {
    struct Layer { Data normed, q, k, v, qkv, query, packed, attention, output, reduced, gate, up, gateUp, scores, descriptors; };
    struct Graph {
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
        Data logits, ids, latent, bias, row, mask;
        Data base, probabilities, stepProbabilities, target, verifyProposal;
        int rows = 1, steps = 0;
        void *sampler = nullptr;
        bool ownsSampler = true;
        float temperature = -1, topP = -1;
        int topK = 0;
        bool disabled = false;
        void Clear() {
            for (auto &graph : graphs) graph.Clear();
            graphs.clear();
            if (ownsSampler) FastllmCudaMtpDestroyDraftSampler(sampler);
            sampler = nullptr;
        }
    } sampling;
    struct ContextLayer { Data raw, key, value; };
    int device = -1;
    bool disabled = false, warmed = false;
    std::vector<void *> inputs;
    std::vector<bool> shortRows;
    Data id, live, positions, hidden, normalized;
    Data combined, projected, committedHidden, contextAll, prefixStorage, contextSelected, contextRaw, contextPointers;
    std::vector<Layer> layers;
    std::vector<ContextLayer> contextLayers;
    std::vector<std::shared_ptr<DraftWorkspace>> tpRanks;
    std::vector<std::pair<Data, Data>> tpKV;
    void *ready = nullptr, *done = nullptr;
    std::vector<void *> Pointers() const {
        std::vector<void *> out;
        for (const Data *p : {&id, &live, &positions, &hidden, &normalized, &combined}) out.push_back(p->isFake ? nullptr : p->cudaData);
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
        sampling.Clear();
        if (ready) FastllmCudaEventDestroy(ready);
        if (done) FastllmCudaEventDestroy(done);
        // Data frees use their recorded device; restore the caller's selection.
        FastllmCudaSetDevice(previous);
    }
};
// Draft subgroup state is independent of the target's TP communicator.
struct NaiveN05FlashModel::DraftTPState {
    std::vector<int> devices;
    int attentionDivisor = 1;
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
    const char *option = std::getenv("FASTLLM_DSPARK_TP");
    if (!option || !*option || std::string(option) == "1") return false;
    const std::string setting(option);
    const bool mlpOnly = setting.compare(0, 4, "mlp:") == 0;
    const std::string count = mlpOnly ? setting.substr(4) : setting;
    const int available = FastllmCudaGetDeviceCount();
    AssertInFastLLM(!count.empty() && count.size() <= 9 &&
        std::all_of(count.begin(), count.end(), [](char c) { return c >= '0' && c <= '9'; }),
        "FASTLLM_DSPARK_TP must be a rank count or mlp:<rank count>.");
    const int ranks = std::stoi(count);
    AssertInFastLLM(ranks >= 2 && ranks <= available, "Draft TP needs at least two available CUDA devices.");
    AssertInFastLLM(GetCudaEmbedding() && !GetLowMemMode() && draftHeads > 0 &&
        draftKvHeads > 0 && draftHeads % draftKvHeads == 0 && draftHeadDim > 0 &&
        draftHeadDim <= 256 && draftHeadDim % 4 == 0 && draftBlock > 1 && draftBlock < 32 &&
        draftLayers > 0 && embed_dim > 0 && draftWindow > 1 &&
        (int64_t)draftWindow + draftBlock <= INT_MAX &&
        (draftHeads + 2LL * draftKvHeads) * draftHeadDim <= INT_MAX &&
        (mlpOnly || (draftHeads % ranks == 0 && (draftKvHeads % ranks == 0 || ranks % draftKvHeads == 0))),
        "Draft TP requires dense BF16 weights, supported Q/KV heads and CUDA embedding.");
    auto state = std::make_shared<DraftTPState>();
    state->attentionDivisor = mlpOnly ? 1 : ranks;
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
        AssertInFastLLM(it != weight.weight.end() && it->second.dataType == BFLOAT16 &&
            !it->second.multiDeviceData && it->second.dims == dims,
            "Draft TP needs a dense BF16 projection with matching shape: " + name);
    };
    for (int i = 0; i < draftLayers; ++i) {
        const std::string p = "dspark.layers." + std::to_string(i) + ".";
        auto down = weight.weight.find(p + "mlp.down_proj.weight");
        AssertInFastLLM(down != weight.weight.end() && down->second.dims.size() == 2 &&
            down->second.dims[1] > 0 && down->second.dims[1] <= INT_MAX / 2 &&
            down->second.dims[1] % ranks == 0, "Draft MLP width must be divisible by its TP size.");
        const int mid = down->second.dims[1];
        projection(p + "self_attn.mergeqkv.weight",
                   {(draftHeads + 2 * draftKvHeads) * draftHeadDim, embed_dim});
        projection(p + "self_attn.o_proj.weight", {embed_dim, draftHeads * draftHeadDim});
        projection(p + "mlp.gateup_proj.weight", {2 * mid, embed_dim});
        projection(p + "mlp.down_proj.weight", {embed_dim, mid});
    }
    AssertInFastLLM(FastllmCudaPeerAccessInit(state->devices), "Draft TP requires peer access.");
    auto split = [&](const std::string &name, int axis, const std::vector<int> &widths, bool replicated = false, bool qkv = false) {
        Data &w = weight[name], bias;
        AssertInFastLLM(w.dataType == BFLOAT16 && w.dims.size() == 2 && !w.multiDeviceData,
                        "Draft TP needs unsharded BF16 projection " + name);
        DivisionScheme scheme;
        int offset = 0;
        for (size_t part = 0; part < widths.size(); ++part) {
            const int width = widths[part];
            // When TP exceeds the KV-head count, neighboring query shards
            // share one whole KV head. Never split a head's RoPE channels.
            const int copies = qkv && part > 0 && !replicated ? std::max(1, ranks / draftKvHeads) : 1;
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
        split(p + "self_attn.mergeqkv.weight", 0,
              {draftHeads * draftHeadDim, draftKvHeads * draftHeadDim, draftKvHeads * draftHeadDim}, mlpOnly, true);
        split(p + "self_attn.o_proj.weight", mlpOnly ? 0 : 1,
              {mlpOnly ? embed_dim : draftHeads * draftHeadDim}, mlpOnly);
        const int mid = weight[p + "mlp.down_proj.weight"].dims.at(1);
        split(p + "mlp.gateup_proj.weight", 0, {mid, mid});
        split(p + "mlp.down_proj.weight", 1, {mid});
        for (const char *name : {"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"}) replicate(p + name);
    }
    replicate("dspark.norm.weight");
    replicate("dspark.mask_embedding");
    replicate("model.embed_tokens.weight");
    state->group = FastllmCudaNaiveDraftTPCreate(state->devices);
    AssertInFastLLM(state->group, "Draft TP communicator creation failed.");
    draftTP = std::move(state);
    ApplyDraftDevice();
    return true;
}

bool NaiveN05FlashModel::AppendDraftContextTP(Data &hidden, int start, DraftContext &context) {
    if (!PrepareDraftTP()) return false;
    auto &tp = *draftTP;
    ApplyDraftDevice();
    if (!context.workspace) {
        context.workspace = std::make_shared<DraftWorkspace>();
        context.workspace->device = tp.devices.front();
        for (int device : tp.devices) {
            auto rank = std::make_shared<DraftWorkspace>();
            rank->device = device;
            if (device != tp.devices.front()) rank->tpKV.resize(draftLayers);
            context.workspace->tpRanks.push_back(std::move(rank));
        }
    }
    if (!draftContextStorage) {
        draftContextStorage = std::make_shared<DraftWorkspace>();
        draftContextStorage->device = tp.devices.front();
        draftContextStorage->ready = FastllmCudaEventCreate();
        for (int device : tp.devices) {
            auto rank = std::make_shared<DraftWorkspace>(); rank->device = device;
            rank->contextLayers.resize(draftLayers);
            FastllmCudaSetDevice(device); rank->done = FastllmCudaEventCreate();
            draftContextStorage->tpRanks.push_back(std::move(rank));
        }
        FastllmCudaSetDevice(tp.devices.front());
    }
    auto &ws = *draftContextStorage;
    context.kv.resize(draftLayers);
    const int begin = std::max(0, hidden.dims[1] - draftWindow + 1);
    // The usual commit input is already dense on rank 0. Its stream waits
    // for all other ranks below before the caller can overwrite/reuse this storage.
    Data *selected = &hidden;
    if (begin || hidden.dataDevice != DataDevice::CUDA ||
        hidden.dataDeviceIds != std::vector<int>{tp.devices.front()} || hidden.multiDeviceData ||
        hidden.strides != std::vector<uint64_t>{(uint64_t)hidden.dims[1] * hidden.dims[2], (uint64_t)hidden.dims[2], 1}) {
        Split(hidden, 1, begin, hidden.dims[1], ws.contextSelected);
        ws.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{tp.devices.front()});
        selected = &ws.contextSelected;
    }
    const int heads = draftHeads / tp.attentionDivisor, kvHeads = std::max(1, draftKvHeads / tp.attentionDivisor);
    const int rows = selected->dims[1], width = kvHeads * draftHeadDim;
    FastllmCudaEventRecordCurrentThread(ws.ready);
    tp.Run([&](int rank) {
        auto &r = *ws.tpRanks[rank];
        Data *input = selected;
        if (rank) {
            r.contextSelected.dataType = BFLOAT16;
            r.contextSelected.UpdateUnitSize();
            r.contextSelected.Resize(selected->dims);
            r.contextSelected.ToDevice(DataDevice::CUDA, std::vector<int>{r.device}, false);
            r.contextSelected.Allocate(false);
            FastllmCudaCurrentThreadStreamWaitEvent(ws.ready);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, r.contextSelected.cudaData,
                tp.devices.front(), selected->cudaData, selected->GetBytes()), "Draft TP context transfer failed.");
            input = &r.contextSelected;
        }
        std::vector<Data> views(draftLayers), parts(draftLayers);
        std::vector<const Data *> raw, norms, weights;
        for (int i = 0; i < draftLayers; ++i) {
            const std::string p = "dspark.layers." + std::to_string(i) + ".self_attn.";
            Data &qkv = *weight[p + "mergeqkv.weight"].multiDeviceDatas.at(r.device);
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
        auto &kv = rank ? context.workspace->tpRanks[rank]->tpKV : context.kv;
        AssertInFastLLM(FastllmCudaNaiveDraftKV(raw, norms, start + begin, kv, kvHeads,
            draftHeadDim, draftWindow, draftWindow + draftBlock, draftEps, draftTheta), "Draft TP KV append failed.");
        FastllmCudaEventRecordCurrentThread(r.done);
    });
    for (size_t rank = 1; rank < tp.devices.size(); ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    context.committed = start + hidden.dims[1];
    return true;
}

#endif

std::shared_ptr<NaiveN05FlashModel::DraftContext> NaiveN05FlashModel::CreateDraftContext() {
    // The request lifecycle and caller hold historyMutex. Reuse storage only;
    // reset every logical prefix, proposal and sampling state for a new request.
    auto context = std::make_shared<DraftContext>();
    if (!idleDraftContext && !idleDraftContexts.empty()) {
        idleDraftContext = std::move(idleDraftContexts.back()); idleDraftContexts.pop_back();
    }
    if (idleDraftContext) {
        context->kv.swap(idleDraftContext->kv);
        context->workspace.swap(idleDraftContext->workspace);
#ifdef USE_CUDA
        if (context->workspace) for (auto &rank : context->workspace->tpRanks)
            for (auto &pair : rank->tpKV) for (Data *cache : {&pair.first, &pair.second})
                if (cache->dims.size() == 3) cache->Resize({1, 0, cache->dims[2]});
#endif
        idleDraftContext.reset();
        for (auto &pair : context->kv)
            for (Data *cache : {&pair.first, &pair.second})
                if (cache->dims.size() == 3) cache->Resize({1, 0, cache->dims[2]});
    }
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
    auto commit = [&](int device) {
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
        const std::vector<int> &past, const std::vector<int> &counts, bool ringVerify) {
    AssertInFastLLM(caches.size() == past.size() && caches.size() == counts.size(), "Invalid target commit batch.");
#ifdef USE_CUDA
    if (!ringVerify) {
        for (size_t i = 0; i < caches.size(); ++i) CommitTargetCache(*caches[i], past[i], counts[i], false);
        return;
    }
    auto commit = [&](int device) {
        std::map<int, std::vector<FastllmNaiveCacheCommit>> updates;
        for (size_t i = 0; i < caches.size(); ++i) for (int layer = 0; layer < block_cnt; ++layer) {
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
#endif
}

void NaiveN05FlashModel::InitDraft() {
    draftEnabled = weight.dicts.count("dspark.model_path") != 0;
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
        auto merged = weight.weight.find(name + "mergeqkv.weight");
        if (merged == weight.weight.end()) return false;
        const Data &qkv = merged->second;
        if (qkv.dataType != BFLOAT16 || qkv.multiDeviceData ||
            qkv.dims != std::vector<int>{(draftHeads + 2 * draftKvHeads) * draftHeadDim, hidden.dims[2]} ||
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
        Data &qkv = weight["dspark.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight"];
        Data view;
        view.FakeFrom(qkv, (size_t)draftHeads * draftHeadDim * hidden.dims[2] * sizeof(uint16_t));
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
        auto merged = weight.weight.find(layer + "mergeqkv.weight");
        if (merged != weight.weight.end()) {
            Data &qkvWeight = merged->second;
            qkvWeight.ToDevice(DataDevice::CUDA, std::vector<int>{FastllmCudaGetDevice()});
            Data keyWeight, valueWeight;
            const int qw = draftHeads * draftHeadDim, kw = draftKvHeads * draftHeadDim;
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
    DraftLinear(combined, weight["dspark.fc.weight"], projected);
    KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, hidden);
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

bool NaiveN05FlashModel::CommitDraftContexts(TargetCapture &capture, const std::vector<int> &lengths,
        const std::vector<int> &counts, const std::vector<DraftContext *> &contexts) {
#ifdef USE_CUDA
    if (GetLowMemMode() || (draftTP && !draftContextStorage) ||
        contexts.size() < 2 || contexts.size() != counts.size() ||
        lengths.size() != counts.size() || draftHeadDim <= 0 || draftHeadDim > 256 || draftHeadDim % 2) return false;
    ApplyDraftDevice();
    const std::vector<int> devices{FastllmCudaGetDevice()};
    const int ranks = draftTP ? draftTP->devices.size() : 1;
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int heads = draftHeads / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
    const int width = kvHeads * draftHeadDim;
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(draftTP->devices[rank]) : weight[name];
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    int total = 0, captured = 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        auto *context = contexts[i];
        if (!context || counts[i] < 1 || counts[i] > lengths[i] || counts[i] >= draftWindow ||
            context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != (size_t)ranks))) return false;
        total += counts[i]; captured += lengths[i];
        for (int rank = 0; rank < ranks; ++rank) for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != std::vector<int>{draftTP ? draftTP->devices[rank] : devices.front()} ||
                cache->dims.size() != 3 || cache->dims[0] != 1 || cache->dims[2] != width || !cache->cudaData) return false;
    }
    for (int rank = 0; rank < ranks; ++rank) for (int i = 0; i < draftLayers; ++i) {
        const auto name = "dspark.layers." + std::to_string(i) + ".self_attn.";
        auto merged = weight.weight.find(name + "mergeqkv.weight");
        if (merged != weight.weight.end()) {
            const Data &w = weightAt(name + "mergeqkv.weight", rank);
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims !=
                std::vector<int>{(heads + 2 * kvHeads) * draftHeadDim, embed_dim}) return false;
        } else for (auto part : {"k_proj.weight", "v_proj.weight"}) {
            const Data &w = weightAt(name + part, rank);
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims != std::vector<int>{width, embed_dim}) return false;
        }
        const Data &norm = weightAt(name + "k_norm.weight", rank);
        if (norm.dataType != FLOAT32 || norm.multiDeviceData || norm.dims != std::vector<int>{draftHeadDim}) return false;
    }
    std::vector<const Data *> inputs;
    for (int layer : draftTargetLayers) inputs.push_back(&capture.hidden.at(layer));
    Data localAll, localCombined, localProjected, localHidden;
    auto *scratch = draftContextStorage.get();
    Data &all = scratch ? scratch->contextAll : localAll;
    Data &combined = scratch ? scratch->combined : localCombined;
    Data &projected = scratch ? scratch->projected : localProjected;
    Data &hidden = scratch ? scratch->committedHidden : localHidden;
    if (!FastllmCudaNaiveDraftConcat(inputs, captured, all)) return false;
    combined.dataType = BFLOAT16; combined.UpdateUnitSize(); combined.Resize({1, total, all.dims[2]});
    combined.ToDevice(DataDevice::CUDA, devices, false); combined.Allocate(false);
    std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
    int inputOffset = 0, outputOffset = 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        dst.push_back((uint16_t *)combined.cudaData + (size_t)outputOffset * all.dims[2]);
        src.push_back((const uint16_t *)all.cudaData + (size_t)inputOffset * all.dims[2]);
        bytes.push_back((size_t)counts[i] * all.dims[2] * sizeof(uint16_t));
        inputOffset += lengths[i]; outputOffset += counts[i];
    }
    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
        "Cannot gather committed draft features.");
    DraftLinear(combined, weight["dspark.fc.weight"], projected);
    KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, hidden);
    auto project = [&](int rank, Data &input) {
        const std::vector<int> localDevices{draftTP ? draftTP->devices[rank] : devices.front()};
        std::vector<Data> raw(draftLayers), views(contexts.size() * draftLayers);
        std::vector<FastllmNaiveDraftKVUpdate> updates;
        // Every layer projects the same committed hidden rows. Reuse the existing
        // batched GEMM and keep the per-layer path for split or unsupported weights.
        std::vector<Data> weightViews(draftLayers);
        std::vector<const Data *> weights;
        Data localKV, localPointers;
        auto *rankScratch = draftTP ? draftContextStorage->tpRanks[rank].get() : nullptr;
        Data &projectedKV = rankScratch ? rankScratch->contextRaw : localKV;
        Data &pointers = rankScratch ? rankScratch->contextPointers : localPointers;
        for (int layer = 0; layer < draftLayers; ++layer) {
            const auto name = "dspark.layers." + std::to_string(layer) + ".self_attn.mergeqkv.weight";
            if (!weight.weight.count(name)) break;
            Data &w = weightAt(name, rank);
            w.ToDevice(DataDevice::CUDA, localDevices);
            auto &view = weightViews[layer];
            view.Resize({2 * width, embed_dim});
            view.FakeFrom(w, (size_t)heads * draftHeadDim * embed_dim * sizeof(uint16_t));
            weights.push_back(&view);
        }
        const bool batched = weights.size() == (size_t)draftLayers &&
            FastllmCudaNaiveDraftKVProject(input, weights, projectedKV, pointers);
        for (int layer = 0; layer < draftLayers; ++layer) {
            const auto name = "dspark.layers." + std::to_string(layer) + ".self_attn.";
            auto merged = weight.weight.find(name + "mergeqkv.weight");
            if (batched) {
                raw[layer].Resize({1, total, 2 * width});
                raw[layer].FakeFrom(projectedKV, (size_t)layer * total * 2 * width * sizeof(uint16_t));
            } else if (merged != weight.weight.end()) {
                Data &w = weightAt(name + "mergeqkv.weight", rank);
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
            for (size_t i = 0; i < contexts.size(); ++i) {
                Data &view = views[layer * contexts.size() + i];
                view.Resize({1, counts[i], 2 * width}); view.FakeFrom(raw[layer], (size_t)offset * 2 * width * sizeof(uint16_t));
                auto &pair = cacheAt(*contexts[i], rank, layer);
                updates.push_back({&view, &norm, &pair.first, &pair.second, contexts[i]->committed}); offset += counts[i];
            }
        }
        AssertInFastLLM(FastllmCudaNaiveBatchDraftKV(updates, kvHeads, draftHeadDim, draftWindow,
            draftWindow + draftBlock, draftEps, draftTheta), "Invalid draft context batch.");
    };
    if (draftTP) {
        auto &ws = *draftContextStorage;
        FastllmCudaEventRecordCurrentThread(ws.ready);
        draftTP->Run([&](int rank) {
            auto &r = *ws.tpRanks[rank];
            Data *input = &hidden;
            if (rank) {
                auto &local = r.contextSelected;
                local.dataType = BFLOAT16; local.UpdateUnitSize(); local.Resize(hidden.dims);
                local.ToDevice(DataDevice::CUDA, {r.device}, false); local.Allocate(false);
                FastllmCudaCurrentThreadStreamWaitEvent(ws.ready);
                AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(r.device, local.cudaData,
                    devices.front(), hidden.cudaData, hidden.GetBytes()), "Draft TP batch context transfer failed.");
                input = &local;
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


bool NaiveN05FlashModel::RunDraftBatch(const std::vector<int> &anchors,
        const std::vector<DraftContext *> &contexts, Data &normalized) {
#ifdef USE_CUDA
    if (contexts.size() < (draftTP ? 1 : 2) || anchors.size() != contexts.size() ||
        !GetCudaEmbedding() || GetLowMemMode() || draftBlock < 2 || draftBlock >= 32 ||
        draftHeadDim % 4 || draftHeadDim > 256) return false;
    ApplyDraftDevice();
    const std::vector<int> devices = draftTP ? draftTP->devices : std::vector<int>{FastllmCudaGetDevice()};
    const int ranks = devices.size(), device = devices.front();
    const int divisor = draftTP ? draftTP->attentionDivisor : 1;
    const int heads = draftHeads / divisor, kvHeads = std::max(1, draftKvHeads / divisor);
    auto weightAt = [&](const std::string &name, int rank) -> Data & {
        return draftTP ? *weight[name].multiDeviceDatas.at(devices[rank]) : DraftWeight(name);
    };
    auto cacheAt = [&](DraftContext &context, int rank, int layer) -> std::pair<Data, Data> & {
        return rank ? context.workspace->tpRanks[rank]->tpKV[layer] : context.kv[layer];
    };
    Data &embedding = weightAt("model.embed_tokens.weight", 0), &mask = weightAt("dspark.mask_embedding", 0);
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 || mask.Count(0) != embed_dim) return false;
    for (auto *context : contexts) {
        if (!context || context->kv.size() != (size_t)draftLayers ||
            (draftTP && (!context->workspace || context->workspace->tpRanks.size() != devices.size()))) return false;
        for (int rank = 0; rank < ranks; ++rank) for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != std::vector<int>{devices[rank]} || cache->dims.size() != 3 ||
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
    std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
    std::vector<bool> shortRows;
    for (auto *context : contexts) {
        shortRows.push_back(std::min(context->committed, draftWindow - 1) + draftBlock <= 256);
        for (int rank = 0; rank < ranks; ++rank) for (int layer = 0; layer < draftLayers; ++layer)
        for (Data *cache : {&cacheAt(*context, rank, layer).first, &cacheAt(*context, rank, layer).second})
            inputs.push_back(cache->cudaData);
    }
    std::shared_ptr<DraftWorkspace> state;
    if (GetFastllmEnv().cudaGraph) {
        auto found = std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(),
            [&](const std::shared_ptr<DraftWorkspace> &entry) {
                return entry->device == device && entry->inputs == inputs && entry->shortRows == shortRows;
            });
        if (found != draftBatchWorkspaces.end()) {
            state = *found; draftBatchWorkspaces.erase(found);
        }
    }
    if (!state) {
        state = std::make_shared<DraftWorkspace>();
        state->device = device; state->inputs = std::move(inputs); state->shortRows = std::move(shortRows);
        if (draftTP) for (int gpu : devices) {
            auto local = std::make_shared<DraftWorkspace>(); local->device = gpu;
            FastllmCudaSetDevice(gpu); local->done = FastllmCudaEventCreate();
            state->tpRanks.push_back(std::move(local));
        }
        FastllmCudaSetDevice(device);
    }
    if (GetFastllmEnv().cudaGraph) {
        draftBatchWorkspaces.push_back(state);
        if (draftBatchWorkspaces.size() > 8) draftBatchWorkspaces.erase(draftBatchWorkspaces.begin());
    }
    auto localState = [&](int rank) -> DraftWorkspace & { return draftTP ? *state->tpRanks[rank] : *state; };
    if (!draftBatchStorage || draftBatchStorage->device != device) {
        draftBatchStorage = std::make_shared<DraftWorkspace>();
        draftBatchStorage->device = device;
        for (int gpu : devices) {
            auto scratch = std::make_shared<DraftWorkspace>(); scratch->device = gpu;
            draftBatchStorage->tpRanks.push_back(std::move(scratch));
        }
    }
    auto scratch = [&](int rank) -> DraftWorkspace & { return *draftBatchStorage->tpRanks.at(rank); };
    // Install map entries on the caller before rank workers read them.
    for (int gpu : devices) draftLinearGraphWorkspaces[gpu];
    auto run = [&](const std::function<void(int)> &fn) {
        if (draftTP) draftTP->Run(fn); else fn(0);
    };
    std::vector<float> tokenIds(anchors.begin(), anchors.end());
    std::vector<int> lengths(contexts.size());
    for (size_t row = 0; row < contexts.size(); ++row) lengths[row] = contexts[row]->committed + 1;
    auto upload = [&](int rank) {
        auto &r = scratch(rank);
        auto copy = [&](Data &target, DataType type, const void *host, const std::vector<int> &dims) {
            target.dataType = type; target.UpdateUnitSize(); target.Resize(dims);
            target.ToDevice(DataDevice::CUDA, {devices[rank]}, false); target.Allocate(false);
            FastllmCudaCopyFromHostToDevice(target.cudaData, const_cast<void *>(host), target.GetBytes());
        };
        copy(r.id, FLOAT32, tokenIds.data(), {1, (int)anchors.size()});
        copy(r.live, INT32, lengths.data(), {(int)anchors.size()});
    };
    auto body = [&](int rank, bool reduce = true) {
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
            if (!draftTP) return;
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
        std::vector<Request> requests(contexts.size());
        auto append = [](Data &batch, Data &part, bool first) {
            if (first) Copy(part, batch);
            else { Data joined; Cat(batch, part, 1, joined); Copy(joined, batch); }
        };
        Data &hidden = r.hidden, &positions = r.positions, &normed = r.combined;
        FastllmCudaNaiveDraftInput(ids, embedding, mask, live, draftBlock, hidden, positions);
        for (size_t row = 0; row < contexts.size(); ++row) {
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
            for (size_t row = 0; row < contexts.size(); ++row) {
                auto &cache = cacheAt(*contexts[row], rank, layer);
                descriptors.push_back({&cache.first, &cache.second, &requests[row].positions,
                    &requests[row].live, std::min(contexts[row]->committed, draftWindow - 1) + draftBlock <= 256});
            }
            Data &packed = l.packed, &query = l.query, &scores = l.scores;
            if (!merged) {
                Data qk;
                Cat(qs, ks, -1, qk); Cat(qk, vs, -1, packed);
            }
            const bool batchedAttention = FastllmCudaNaiveBatchDraftQKV(merged ? qkv : packed,
                    w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                    descriptors, query, heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta,
                    &localState(rank).layers[layer].descriptors) &&
                FastllmCudaNaiveBatchDraftAttention(query, descriptors, heads, kvHeads,
                    draftHeadDim, draftWindow, scores, attention);
            if (!batchedAttention) {
                for (size_t row = 0; row < contexts.size(); ++row) {
                    auto &r = requests[row]; auto &cache = cacheAt(*contexts[row], rank, layer);
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
                    const bool shortAttention = std::min(contexts[row]->committed, draftWindow - 1) + draftBlock <= 256;
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
    for (int rank = 0; rank < ranks; ++rank) localState(rank).layers.resize(draftLayers);
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
    if (draftTP) for (int rank = 1; rank < ranks; ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(localState(rank).done);
    if (!useGraph) invalidateMoved();
    scratch(0).normalized.Resize({1, (int)contexts.size() * draftBlock, embed_dim});
    Copy(scratch(0).normalized, normalized);
    return true;
#else
    return false;
#endif
}

struct NaiveN05FlashModel::DraftRound {
    NaiveN05FlashModel &model;
    DraftContext &context;
    const GenerationConfig &config;
    LastTokensUnit lastTokens;
    int anchor, limit, oldLength, vocab = 0;
    bool greedy = false;
    int sampledAccepted = -1, sampledNext = -1;
    bool targetCommitted = false, draftCommitted = false;
    LogitsSelection selection;
    std::vector<int> proposed;
    std::vector<std::vector<float>> q;
    Data batchProbabilities;
    std::shared_ptr<DraftWorkspace> proposalOwner;
    Data &ProposalProbabilities() {
        return proposalOwner ? batchProbabilities : context.proposalProbs;
    }

    DraftRound(NaiveN05FlashModel &model, DraftContext &context, const GenerationConfig &config,
               const LastTokensUnit &lastTokens, int anchor, int limit)
        : model(model), context(context), config(config), lastTokens(lastTokens),
          anchor(anchor), limit(limit), oldLength(context.committed) {}

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
    GenerationConfig samplingConfig = config;
    samplingConfig.tool_call_token_mask.reset();
    const bool greedy = round.greedy = samplingConfig.IsSimpleGreedy() && config.output_token_least <= 0;
    LastTokensUnit samplingTokens = round.lastTokens;
    auto &selection = round.selection;
    selection = SelectLogits(config, true);
    // Reuse Qwen3.5's GPU proposal/rejection sampler for unrestricted top-p.
    // This shares ordinary CUDA sampling's top-p boundary/tie handling.
    // Its finite top-k semantics differ from the legacy CPU sampler; penalties,
    // legacy name masks and minimum-output constraints retain that generic fallback.
#ifdef USE_CUDA
    // Grammar masks are applied per verification row. Greedy verification also
    // keeps full logits on the device, then transfers only the masked argmax.
    if (baseLogits.dataDevice == DataDevice::CUDA &&
        ((greedy && config.tool_call_grammar) || (!greedy && SelectLogits(config).cudaSampling))) {
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

void NaiveN05FlashModel::PrepareDraftSamplingMemory(int count, int steps, int vocab, int latentWidth) {
#ifdef USE_CUDA
    ApplyDraftDevice();
    const int device = FastllmCudaGetDevice(), words = (vocab + 31) / 32;
    if (!draftProposalStorage) {
        draftProposalStorage = std::make_shared<DraftWorkspace>();
        draftProposalStorage->device = device;
    }
    auto &shared = draftProposalStorage->sampling;
    const int rows = std::max(count, maxBatch > 0 ? maxBatch : 512);
    steps = std::max(steps, draftBlock);
    if (shared.rows >= rows && shared.steps >= steps && !shared.probabilities.dims.empty()) return;
    // Graph and eager/masked proposals execute serially and share this bounded
    // arena. Ragged verification groups have their own packing buffers.
    for (auto &entry : draftProposalWorkspaces) entry->sampling.Clear();
    draftProposalWorkspaces.clear(); shared.Clear();
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
    allocate(shared.target, FLOAT32, {rows * (steps + 1), vocab});
    allocate(shared.verifyProposal, FLOAT32, {rows * steps, vocab});
    std::vector<uint32_t> mask((size_t)rows * words, ~uint32_t(0));
    FastllmCudaCopyFromHostToDevice(shared.mask.cudaData, mask.data(), mask.size() * sizeof(uint32_t));
#endif
}

bool NaiveN05FlashModel::PrepareDraftBatchGraph(
        std::vector<std::unique_ptr<DraftRound>> &rounds, const Data &baseLogits) {
#ifdef USE_CUDA
    const int count = rounds.size(), steps = rounds.front()->limit, vocab = baseLogits.dims.back();
    if (!GetFastllmEnv().cudaGraph || steps < 1 || baseLogits.dims[1] != count * steps || vocab >= (1 << 24)) return false;
    std::vector<float> temperatures;
    for (auto &owner : rounds) {
        const auto &r = *owner;
        if (r.limit != steps || r.config.IsSimpleGreedy() || r.config.output_token_least > 0 ||
            r.config.tool_call_grammar || r.config.tool_call_token_mask ||
            !SelectLogits(r.config).cudaSampling) return false;
        temperatures.push_back(r.config.temperature);
    }
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    const int device = FastllmCudaGetDevice(), words = (vocab + 31) / 32;
    Data &w1 = weight["dspark.markov_head.markov_w1.weight"], &w2 = weight["dspark.markov_head.markov_w2.weight"];
    w1.ToDevice(DataDevice::CUDA, std::vector<int>{device}); w2.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    std::vector<void *> inputs{w1.cudaData, w2.cudaData, baseLogits.cudaData};
    PrepareDraftSamplingMemory(count, steps, vocab, w1.dims.back());
    auto &shared = draftProposalStorage->sampling;
    std::shared_ptr<DraftWorkspace> state;
    auto found = std::find_if(draftProposalWorkspaces.begin(), draftProposalWorkspaces.end(),
        [&](const std::shared_ptr<DraftWorkspace> &s) {
            const auto &p = s->sampling;
            return s->device == device && s->inputs == inputs && p.rows == count && p.steps == steps &&
                p.base.dims == baseLogits.dims;
        });
    if (found != draftProposalWorkspaces.end()) {
        state = *found; draftProposalWorkspaces.erase(found);
    } else {
        state = std::make_shared<DraftWorkspace>(); state->device = device; state->inputs = inputs;
        auto &p = state->sampling;
        p.rows = count; p.steps = steps;
        auto view = [](Data &dst, Data &src, const std::vector<int> &dims) {
            dst.FakeFrom(src, 0); dst.Resize(dims);
        };
        view(p.ids, shared.ids, {steps + 1, count});
        view(p.probabilities, shared.probabilities, {count, steps, vocab});
        view(p.stepProbabilities, shared.stepProbabilities, {count, vocab});
        view(p.logits, shared.logits, {count, vocab});
        view(p.bias, shared.bias, {1, count, vocab});
        view(p.latent, shared.latent, {1, count, w1.dims.back()});
        view(p.mask, shared.mask, {count, words});
        // Proposals retain the full temperature-softmax distribution for rejection sampling.
        p.sampler = shared.sampler;
        p.ownsSampler = false;
        p.disabled = !p.sampler;
        p.graphs.resize(1);
    }
    draftProposalWorkspaces.push_back(state);
    if (draftProposalWorkspaces.size() > 4) draftProposalWorkspaces.erase(draftProposalWorkspaces.begin());
    auto &p = state->sampling;
    if (p.disabled) return false;
    std::vector<int> topKs(count, 0);
    std::vector<float> topPs(count, 1.f);
    // Temperatures are replay inputs; only the shape selects a graph variant.
    if (!FastllmCudaMtpDraftSamplerSetParameters(p.sampler, count, temperatures.data(), topKs.data(), topPs.data()))
        return false;
    // The caller retains the head output until this round has been sampled.
    // Its address is part of the graph key; no full-vocabulary staging copy.
    p.base.FakeFrom(baseLogits, 0); p.base.Resize(baseLogits.dims);
    std::vector<float> anchors;
    for (auto &r : rounds) anchors.push_back(r->anchor);
    FastllmCudaCopyFromHostToDevice(p.ids.cudaData, anchors.data(), count * sizeof(float));
    auto body = [&]() {
        for (int step = 0; step < steps; ++step) {
            Data ids(FLOAT32, {1, count}); ids.FakeFrom(p.ids, (size_t)step * count * sizeof(float));
            EmbeddingDirect(ids, w1, p.latent); DraftLinear(p.latent, w2, p.bias);
            FastllmCudaNaiveBatchDraftSamplingLogits(p.base, p.bias, p.mask, p.logits, step, steps);
            AssertInFastLLM(FastllmCudaMtpDraftSamplerRunBatch(p.sampler, (float *)p.logits.cudaData,
                (float *)p.stepProbabilities.cudaData, (float *)p.ids.cudaData + (size_t)(step + 1) * count,
                (uint64_t)step * 4), "Graph batch draft sampling failed.");
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
            for (int row = 0; row < count; ++row) {
                dst.push_back((float *)p.probabilities.cudaData + (size_t)(row * steps + step) * vocab);
                src.push_back((float *)p.stepProbabilities.cudaData + (size_t)row * vocab);
                bytes.push_back((size_t)vocab * sizeof(float));
            }
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), count),
                "Cannot retain graph proposal probabilities.");
        }
    };
    auto &graph = p.graphs.front();
    if (!graph.exec) {
        AssertInFastLLM(FastllmCudaMtpDraftSamplerSetSeed(p.sampler), "Cannot seed batch draft warmup.");
        body();
        if (!graph.Capture(body, p.disabled)) return false;
        // Warmup wrote later ID rows; only the first row is an external input.
        FastllmCudaCopyFromHostToDevice(p.ids.cudaData, anchors.data(), count * sizeof(float));
    }
    AssertInFastLLM(FastllmCudaMtpDraftSamplerSetSeed(p.sampler) && FastllmCudaGraphLaunch(graph.exec),
        "Cannot replay batch draft sampling graph.");
    for (int row = 0; row < count; ++row) {
        auto &r = *rounds[row]; r.vocab = vocab; r.greedy = false;
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        r.proposalOwner = state;
        r.batchProbabilities.FakeFrom(p.probabilities, (size_t)row * steps * vocab * sizeof(float));
        r.batchProbabilities.Resize({steps, vocab});
    }
    std::vector<float> tokens((size_t)steps * count);
    FastllmCudaCopyFromDeviceToHost(tokens.data(), (float *)p.ids.cudaData + count, tokens.size() * sizeof(float));
    for (int step = 0; step < steps; ++step) for (int row = 0; row < count; ++row) {
        const float token = tokens[(size_t)step * count + row];
        AssertInFastLLM(std::isfinite(token) && token >= 0 && token < vocab && token == (int)token,
            "Invalid graph draft token.");
        rounds[row]->proposed.push_back((int)token);
    }
    return true;
#else
    return false;
#endif
}


bool NaiveN05FlashModel::PrepareDraftBatch(std::vector<std::unique_ptr<DraftRound>> &rounds,
        const Data &baseLogits) {
#ifdef USE_CUDA
    if (rounds.size() < 2 || draftConfidenceThreshold != 0 || baseLogits.dataType != BFLOAT16 ||
        baseLogits.dataDevice != DataDevice::CUDA || baseLogits.dims.size() != 3 || baseLogits.dims[0] != 1)
        return false;
    Data &w1 = weight["dspark.markov_head.markov_w1.weight"], &w2 = weight["dspark.markov_head.markov_w2.weight"];
    const int vocab = baseLogits.dims.back(), words = (vocab + 31) / 32;
    if (w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.dims.size() != 2 ||
        w1.dims[0] != vocab || w2.dims != w1.dims || !GetCudaEmbedding()) return false;
    if (PrepareDraftBatchGraph(rounds, baseLogits)) return true;
    int total = 0, steps = 0;
    std::vector<int> offsets;
    std::vector<GenerationConfig> configs;
    for (auto &owner : rounds) {
        auto &r = *owner; auto config = r.config; config.tool_call_token_mask.reset();
        if (r.limit < 1 || r.config.output_token_least > 0 ||
            (!config.IsSimpleGreedy() && !SelectLogits(r.config).cudaSampling)) return false;
        offsets.push_back(total); total += r.limit; steps = std::max(steps, r.limit);
        configs.push_back(r.config);
    }
    if (total != baseLogits.dims[1]) return false;
    ApplyDraftDevice();
    w1.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds); w2.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds);
    auto allocate = [&](Data &data, DataType type, std::vector<int> dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, baseLogits.dataDeviceIds, false); data.Allocate(false);
    };
    PrepareDraftSamplingMemory(rounds.size(), steps, vocab, w1.dims.back());
    auto &shared = draftProposalStorage->sampling;
    for (size_t row = 0; row < rounds.size(); ++row) {
        auto &r = *rounds[row]; auto config = r.config; config.tool_call_token_mask.reset();
        r.vocab = vocab; r.greedy = config.IsSimpleGreedy();
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        if (!r.greedy) {
            r.proposalOwner = draftProposalStorage;
            r.batchProbabilities.FakeFrom(shared.probabilities, (size_t)row * shared.steps * vocab * sizeof(float));
            r.batchProbabilities.Resize({r.limit, vocab});
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
                dst.push_back((float *)r.ProposalProbabilities().cudaData + (size_t)step * vocab);
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

void NaiveN05FlashModel::PrepareDraftAcceptance(std::vector<std::unique_ptr<DraftRound>> &rounds, Data &logits) {
#ifdef USE_CUDA
    if (logits.dataDevice != DataDevice::CUDA || logits.dataType != FLOAT32 || rounds.empty()) return;
    ApplyDraftDevice();
    const int vocab = logits.dims.back();
    PrepareDraftSamplingMemory(rounds.size(), draftBlock, vocab,
        weight["dspark.markov_head.markov_w1.weight"].dims.back());
    auto &packing = draftProposalStorage->sampling;
    std::vector<int> offsets;
    std::map<std::pair<bool, int>, std::vector<int>> groups;
    int offset = 0;
    for (int i = 0; i < (int)rounds.size(); ++i) {
        auto &r = *rounds[i]; offsets.push_back(offset); offset += r.proposed.size() + 1;
        if (r.vocab != vocab || (!r.greedy && !r.selection.cudaLogits)) continue;
        if (r.config.tool_call_grammar) {
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
        Data target(FLOAT32, {count * rows, vocab}), proposal(FLOAT32, {count * drafts, vocab});
        const int first = entry.second.front();
        bool targetView = true, proposalView = !greedy && rounds[first]->proposalOwner;
        for (int b = 0; b < count; ++b) {
            const int i = entry.second[b];
            targetView = targetView && offsets[i] == offsets[first] + b * rows;
            proposalView = proposalView && rounds[i]->proposalOwner == rounds[first]->proposalOwner &&
                rounds[i]->ProposalProbabilities().cudaData ==
                    (float *)rounds[first]->ProposalProbabilities().cudaData + (size_t)b * drafts * vocab;
        }
        auto borrow = [](Data &data, Data &backing) {
            auto shape = data.dims; data.FakeFrom(backing, 0); data.Resize(shape);
        };
        if (targetView) {
            target.FakeFrom(logits, (size_t)offsets[first] * vocab * sizeof(float));
            target.Resize({count * rows, vocab});
        } else borrow(target, packing.target);
        if (proposalView) {
            proposal.FakeFrom(rounds[first]->ProposalProbabilities(), 0);
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
                    src.push_back(r.ProposalProbabilities().cudaData); bytes.push_back((size_t)drafts * vocab * sizeof(float));
                }
                ids.insert(ids.end(), r.proposed.begin(), r.proposed.end());
                temperatures.insert(temperatures.end(), rows, r.config.temperature);
                ps.insert(ps.end(), rows, r.config.top_p); ks.insert(ks.end(), rows, r.config.top_k);
            }
        }
        AssertInFastLLM(dst.empty() || FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
            "Cannot pack verification sampling batch.");
        if (greedy) {
            Data top; TopK(target, top, 1); top.ToDevice(DataDevice::CPU);
            for (int b = 0; b < count; ++b) {
                auto &r = *rounds[entry.second[b]];
                int &keep = accepted[b];
                while (keep < drafts && r.proposed[keep] == (int)((float *)top.cpuData)[(b * rows + keep) * 2]) ++keep;
                output[b * rows + keep] = (int)((float *)top.cpuData)[(b * rows + keep) * 2];
            }
        } else AssertInFastLLM(FastllmCudaMtpRejectionSampling((float *)target.cudaData, (float *)proposal.cudaData,
            temperatures.data(), ks.data(), ps.data(), ids.data(), output.data(), accepted.data(), count, drafts, vocab),
            "Batch rejection sampling failed.");
        for (int b = 0; b < count; ++b) {
            auto &r = *rounds[entry.second[b]];
            r.sampledAccepted = accepted[b]; r.sampledNext = output[b * rows + accepted[b]];
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
    if (round.sampledAccepted < 0 && config.tool_call_grammar) {
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
    LastTokensUnit samplingTokens = round.lastTokens;
    int accepted = round.sampledAccepted, next = round.sampledNext;
    if (accepted < 0) {
        accepted = 0;
#ifdef USE_CUDA
        if (!greedy && selection.cudaLogits) {
            Data &proposal = round.ProposalProbabilities();
            AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
                logits.dataDeviceIds == proposal.dataDeviceIds,
                "Naive CUDA verification logits are on the wrong device.");
            FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
            const int rows = proposed.size() + 1;
            std::vector<float> temperatures(rows, config.temperature), topPs(rows, config.top_p);
            std::vector<int> topKs(rows, config.top_k), sampled(rows);
            AssertInFastLLM(FastllmCudaMtpRejectionSampling((float *)logits.cudaData,
                (float *)proposal.cudaData, temperatures.data(), topKs.data(), topPs.data(),
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
    ++context.rounds;
    context.proposed += proposed.size();
    context.accepted += accepted;
    proposed.resize(accepted);
    proposed.push_back(next);
    for (int i = 1; i < (int)proposed.size(); ++i)
        context.pending.emplace_back(proposed[i - 1], proposed[i]);
    return proposed[0];
}


std::vector<int> NaiveN05FlashModel::ForwardDraftBatch(int batch, const Data &inputIds,
        const std::vector<Data *> &positionIds, const std::vector<int> &seqLens,
        std::vector<std::pair<Data *, Data *>> &pastKeyValues,
        const std::vector<GenerationConfig> &configs, const LastTokensManager &lastTokens,
        std::vector<std::vector<float> *> *retLogits) {
    AssertInFastLLM(batch > 0 && (int)seqLens.size() == batch &&
        (int)configs.size() == batch && (int)positionIds.size() == batch &&
        (int)pastKeyValues.size() == batch * block_cnt &&
        std::all_of(seqLens.begin(), seqLens.end(), [](int n) { return n == 1; }),
        "DSpark batch expects initialized single-token request descriptors.");
    std::vector<std::vector<std::pair<Data, Data>> *> caches(batch, nullptr);
    std::vector<std::shared_ptr<DraftContext>> owners(batch);
    {
        std::lock_guard<std::mutex> guard(historyMutex);
        for (const auto &entry : draftContexts) {
            if (entry.first->empty()) continue;
            for (int i = 0; i < batch; ++i)
                if (&entry.first->front().first == pastKeyValues[i * block_cnt].first) {
                    caches[i] = const_cast<std::vector<std::pair<Data, Data>> *>(entry.first);
                    owners[i] = entry.second;
                }
        }
    }
    Data ids(inputIds);
    ids.ToDevice(DataDevice::CPU);
    AssertInFastLLM(ids.dataType == FLOAT32 && ids.Count(0) == (size_t)batch,
                   "DSpark batch token count mismatch.");
    std::vector<int> anchors(batch), result(batch);
    for (int i = 0; i < batch; ++i) {
        AssertInFastLLM(caches[i] && owners[i] && positionIds[i],
                       "DSpark request has no initialized draft context.");
        for (int layer = 0; layer < block_cnt; ++layer)
            AssertInFastLLM(pastKeyValues[i * block_cnt + layer].first == &(*caches[i])[layer].first &&
                pastKeyValues[i * block_cnt + layer].second == &(*caches[i])[layer].second,
                "DSpark batch mixed request KV descriptors.");
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
            GenerationConfig sampling = config;
            sampling.tool_call_token_mask.reset();
            if (!context.committed || context.kv.empty() || terminal ||
                context.pending.size() > pendingLimit ||
                config.output_logits || config.output_token_least > 0 ||
                (!config.tool_call_grammar && (config.tool_call_token_mask ||
                    config.tool_call_name_constraint_enabled ||
                    config.tool_call_parameter_name_constraint_enabled ||
                    config.tool_call_content_sampling_enabled)) ||
                (!sampling.IsSimpleGreedy() && !SelectLogits(config).cudaSampling)) continue;
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
    std::vector<GenerationConfig> roundConfigs;
    roundConfigs.reserve(selected.size()); // DraftRound borrows these configs.
    std::vector<int> pending, lengths;
    std::vector<float> verifyIds, verifyPositions;
    std::vector<GenerationConfig> verifyConfigs;
    std::vector<std::pair<Data *, Data *>> verifyCaches;
    std::vector<int> draftAnchors;
    std::vector<DraftContext *> draftContexts;
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row];
        auto &context = *owners[i];
        auto tokens = history(i);
        roundConfigs.push_back(configs[i]);
        auto &config = roundConfigs.back();
        for (const auto &item : context.pending) {
            tokens.Push(item.second);
            if (config.tool_call_grammar) AdvanceToolCallGrammar(config, item.second);
        }
        if (config.tool_call_grammar) PrepareToolCallConstraint(config);
        int anchor = context.pending.empty() ? anchors[i] : context.pending.back().second;
        rounds.emplace_back(new DraftRound(*this, context, config, tokens, anchor, limits[row]));
        draftAnchors.push_back(anchor);
        draftContexts.push_back(&context);
    }
    Data hidden, headInput;
    if (RunDraftBatch(draftAnchors, draftContexts, hidden)) {
#ifdef USE_CUDA
        const int rows = std::accumulate(limits.begin(), limits.end(), 0);
        headInput.dataType = hidden.dataType; headInput.UpdateUnitSize(); headInput.Resize({1, rows, embed_dim});
        headInput.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false); headInput.Allocate(false);
        std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
        size_t offset = 0;
        for (size_t row = 0; row < selected.size(); ++row) {
            dst.push_back((uint8_t *)headInput.cudaData + offset);
            src.push_back((const uint8_t *)hidden.cudaData + row * draftBlock * embed_dim * hidden.unitSize);
            const size_t size = (size_t)limits[row] * embed_dim * hidden.unitSize;
            bytes.push_back(size); offset += size;
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
            "Cannot gather batch draft head input.");
#endif
    }
    Data headLogits = headInput.dims.empty() ? Data() : RunDraftHead(headInput);
    const bool preparedBatch = !headLogits.dims.empty() && PrepareDraftBatch(rounds, headLogits);
    size_t headOffset = 0;
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row];
        auto &context = *owners[i];
        auto &round = *rounds[row];
        Data head;
        if (!headLogits.dims.empty()) {
            head.dataType = headLogits.dataType;
            head.Resize({1, limits[row], headLogits.dims.back()});
            head.FakeFrom(headLogits, headOffset);
            headOffset += head.GetBytes();
        }
        if (!preparedBatch) PrepareDraftRound(round, head.dims.empty() ? nullptr : &head);
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
        verifyConfigs.push_back(configs[i]);
        for (auto &layer : *caches[i]) verifyCaches.emplace_back(&layer.first, &layer.second);
    }
    Data input(FLOAT32, {1, (int)verifyIds.size()}, verifyIds);
    Data positions(FLOAT32, {1, (int)verifyPositions.size()}, verifyPositions);
    TargetBatch requests{verifyCaches, verifyConfigs, lengths, block_cnt};
    TargetCapture capture;
    // All acceptance and KV commits below complete before another target
    // forward can overwrite the stable graph/logits workspace.
    capture.borrowOutputs = true;
    LogitsSelection selection;
    selection.cudaLogits = true;
    capture.verifying = true;
    capture.selection = &selection;
    std::vector<std::pair<Data, Data>> unused;
    Data logits = RunDraftTarget(input, positions, unused, verifyConfigs[0], capture, &requests);
    const int vocab = logits.dims.back();
    int offset = 0;
    PrepareDraftAcceptance(rounds, logits);
    if (std::all_of(rounds.begin(), rounds.end(), [](const auto &r) { return r->sampledAccepted >= 0; })) {
        std::vector<int> past, counts;
        std::vector<std::vector<std::pair<Data, Data>> *> commitCaches;
        for (size_t row = 0; row < selected.size(); ++row) {
            past.push_back(rounds[row]->oldLength); counts.push_back(rounds[row]->sampledAccepted + 1);
            commitCaches.push_back(caches[selected[row]]);
        }
        CommitTargetCaches(commitCaches, past, counts, capture.ringVerify);
        const bool draftCommitted = CommitDraftContexts(capture, lengths, counts, draftContexts);
        for (auto &round : rounds) { round->targetCommitted = true; round->draftCommitted = draftCommitted; }
    }
    std::vector<bool> handled(batch, false);
    for (size_t row = 0; row < selected.size(); ++row) {
        const int i = selected[row], count = lengths[row];
        auto &context = *owners[i];
        Data rows(FLOAT32, {count, vocab});
        rows.FakeFrom(logits, (size_t)offset * vocab * sizeof(float));
        TargetCapture features;
        features.verifying = true;
        features.ringVerify = capture.ringVerify;
        if (!rounds[row]->draftCommitted) for (int layer : draftTargetLayers)
            Split(capture.hidden.at(layer), 1, offset, offset + count, features.hidden[layer]);
        int token = FinishDraftRound(*rounds[row], rows, features, *caches[i]);
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
        if (!entry) entry = CreateDraftContext();
        owner = entry;
    }
    auto &context = *owner;
    Data ids;
    ids.CopyFrom(inputIds);
    ids.ToDevice(DataDevice::CPU);
    const int anchor = (int)((float *)ids.cpuData)[0];
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
    if (context.committed == 0 || context.kv.empty() || inputIds.dims[1] != 1 || config.output_logits || constrained || limit <= 0) {
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
        if (isIntermediateChunkedPrefill) return 0;
        return SampleTarget(logits, kv, config, lastTokens, retLogits, &selection);
    }
    LastTokensUnit history = lastTokens.units.empty() ? LastTokensUnit(config.last_n) : lastTokens.units[0];
    DraftRound round(*this, context, config, history, anchor, limit);
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
    ApplyDraftDevice();
    Data hidden(BFLOAT16, {1, draftWindow - 1, embed_dim});
    hidden.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); hidden.Allocate(true);
    std::vector<std::shared_ptr<DraftContext>> owners;
    std::vector<DraftContext *> contexts;
    std::vector<int> anchors(batch, 1);
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
    Data normalized;
    if (RunDraftBatch(anchors, contexts, normalized)) {
        Data head = RunDraftHead(normalized);
        std::vector<GenerationConfig> configs(batch);
        std::vector<std::unique_ptr<DraftRound>> rounds;
        LastTokensUnit history;
        for (int i = 0; i < batch; ++i) {
            configs[i].temperature = 1.0f; configs[i].top_p = 0.95f;
            configs[i].top_k = head.dims.back();
            rounds.emplace_back(new DraftRound(*this, *contexts[i], configs[i], history, 1, draftBlock));
        }
        if (PrepareDraftBatch(rounds, head)) {
            tpSamplingLogits.Resize({batch * (draftBlock + 1), head.dims.back()});
            tpSamplingLogits.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false);
            tpSamplingLogits.Allocate(false);
            FastllmCudaMemset0(tpSamplingLogits.cudaData, tpSamplingLogits.GetBytes());
            PrepareDraftAcceptance(rounds, tpSamplingLogits);
        }
        TargetCapture capture;
        capture.verifying = true;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, batch * draftBlock, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        CommitDraftContexts(capture, std::vector<int>(batch, draftBlock), std::vector<int>(batch, draftBlock), contexts);
        capture.hidden.clear(); capture.verifying = false;
        for (int layer : draftTargetLayers) {
            Data &feature = capture.hidden[layer];
            feature.dataType = BFLOAT16; feature.Resize({1, draftWindow + GetPageLen() - 1, embed_dim});
            feature.ToDevice(DataDevice::CUDA, {draftTP->devices.front()}, false); feature.Allocate(true);
        }
        std::vector<std::pair<Data, Data>> unused;
        CommitDraftContext(capture, draftWindow + GetPageLen() - 1, *contexts.front(), unused);
    }
    idleDraftContexts = std::move(owners);
#endif
}

}
