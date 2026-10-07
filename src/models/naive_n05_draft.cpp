#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "json11.hpp"
#include "executor.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
namespace {
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
    struct Layer { Data normed, q, k, v, qkv, attention, output, gate, up, gateUp, scores; };
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
    } graphs[2];
    struct Proposal {
        Graph graph;
        bool disabled = false;
        std::vector<void *> weights;
        Data logits, ids, latent, bias, partial;
    } proposal;
    struct ContextLayer { Data raw, key, value; };
    int device = -1;
    bool disabled = false;
    std::vector<void *> inputs;
    Data id, live, positions, hidden, normalized;
    Data combined, projected, committedHidden, contextSelected, contextRaw, contextPointers;
    std::vector<Layer> layers;
    std::vector<ContextLayer> contextLayers;
    std::vector<std::shared_ptr<DraftWorkspace>> tpRanks;
    std::vector<std::pair<Data, Data>> tpKV;
    void *ready = nullptr, *done = nullptr;
    ~DraftWorkspace() {
        int previous = FastllmCudaGetDevice();
        if (device >= 0) FastllmCudaSetDevice(device);
        for (auto &graph : graphs) graph.Clear();
        proposal.graph.Clear();
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
        (mlpOnly || (draftHeads % ranks == 0 && draftKvHeads % ranks == 0)),
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
    auto split = [&](const std::string &name, int axis, const std::vector<int> &widths, bool replicated = false) {
        Data &w = weight[name], bias;
        AssertInFastLLM(w.dataType == BFLOAT16 && w.dims.size() == 2 && !w.multiDeviceData,
                        "Draft TP needs unsharded BF16 projection " + name);
        DivisionScheme scheme;
        int offset = 0;
        for (int width : widths) {
            AssertInFastLLM(width > 0 && (replicated || width % ranks == 0), "Draft TP projection cannot be split.");
            for (int rank = 0; rank < ranks; ++rank)
                scheme[state->devices[rank]].push_back(replicated ? std::make_pair(offset, offset + width) :
                    std::make_pair(offset + rank * width / ranks, offset + (rank + 1) * width / ranks));
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
              {draftHeads * draftHeadDim, draftKvHeads * draftHeadDim, draftKvHeads * draftHeadDim}, mlpOnly);
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
            rank->layers.resize(draftLayers);
            rank->contextLayers.resize(draftLayers);
            if (device != tp.devices.front()) rank->tpKV.resize(draftLayers);
            FastllmCudaSetDevice(device);
            rank->done = FastllmCudaEventCreate();
            context.workspace->tpRanks.push_back(std::move(rank));
        }
        ApplyDraftDevice();
        FastllmCudaSetDevice(tp.devices.front());
        context.workspace->ready = FastllmCudaEventCreate();
    }
    auto &ws = *context.workspace;
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
    const int heads = draftHeads / tp.attentionDivisor, kvHeads = draftKvHeads / tp.attentionDivisor;
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
        auto &kv = rank ? r.tpKV : context.kv;
        AssertInFastLLM(FastllmCudaNaiveDraftKV(raw, norms, start + begin, kv, kvHeads,
            draftHeadDim, draftWindow, draftWindow + draftBlock, draftEps, draftTheta), "Draft TP KV append failed.");
        FastllmCudaEventRecordCurrentThread(r.done);
    });
    for (size_t rank = 1; rank < tp.devices.size(); ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    context.committed = start + hidden.dims[1];
    return true;
}

bool NaiveN05FlashModel::RunDraftTP(int anchor, DraftContext &context, Data &output) {
    if (!draftTP) return false;
    auto &tp = *draftTP;
    auto &ws = *context.workspace;
    const int ranks = tp.devices.size();
    const int heads = draftHeads / tp.attentionDivisor, kvHeads = draftKvHeads / tp.attentionDivisor;
    const bool shortAttention = std::min(context.committed, draftWindow - 1) + draftBlock <= 256;
    const int slot = shortAttention ? 0 : 1;
    auto local = [&](const std::string &name, int device) -> Data & { return *weight[name].multiDeviceDatas.at(device); };
    auto uploadInputs = [&](int rank) {
        auto &r = *ws.tpRanks[rank];
        auto upload = [&](Data &x, DataType type, void *value) {
            if (x.dims.empty()) {
                x.dataType = type; x.UpdateUnitSize(); x.Resize({1});
                x.ToDevice(DataDevice::CUDA, std::vector<int>{r.device}, false); x.Allocate(false);
            }
            FastllmCudaCopyFromHostToDevice(x.cudaData, value, sizeof(int));
        };
        float token = anchor; int live = context.committed + 1;
        upload(r.id, FLOAT32, &token); upload(r.live, INT32, &live);
    };
    auto body = [&](int rank) {
        auto &r = *ws.tpRanks[rank];
        auto &kv = rank ? r.tpKV : context.kv;
        auto w = [&](const std::string &name) -> Data & { return local(name, r.device); };
        FastllmCudaNaiveDraftInput(r.id, w("model.embed_tokens.weight"), w("dspark.mask_embedding"), r.live,
                                  draftBlock, r.hidden, r.positions);
        for (int i = 0; i < draftLayers; ++i) {
            const std::string p = "dspark.layers." + std::to_string(i) + ".";
            auto &b = r.layers[i];
            KimiK3RMSNorm(r.hidden, w(p + "input_layernorm.weight"), draftEps, b.normed);
            DraftLinear(b.normed, w(p + "self_attn.mergeqkv.weight"), b.qkv);
            AssertInFastLLM(FastllmCudaNaiveDraftQKV(b.qkv, w(p + "self_attn.q_norm.weight"), w(p + "self_attn.k_norm.weight"),
                r.positions, r.live, kv[i].first, kv[i].second, b.q, heads, kvHeads, draftHeadDim, draftWindow, draftEps, draftTheta), "Draft TP QKV failed.");
            FastllmCudaNaiveDraftAttention(b.q, kv[i].first, kv[i].second, r.live, heads, kvHeads, draftHeadDim,
                draftWindow, shortAttention, b.scores, b.attention);
            DraftLinear(b.attention, w(p + "self_attn.o_proj.weight"), b.output);
            if (tp.attentionDivisor > 1) FastllmCudaNaiveDraftTPReduce(tp.group, rank, b.output);
            AddTo(r.hidden, b.output);
            KimiK3RMSNorm(r.hidden, w(p + "post_attention_layernorm.weight"), draftEps, b.normed);
            DraftLinear(b.normed, w(p + "mlp.gateup_proj.weight"), b.gateUp);
            FastllmCudaNaiveDraftSwiGLU(b.gateUp, b.gate);
            DraftLinear(b.gate, w(p + "mlp.down_proj.weight"), b.output);
            FastllmCudaNaiveDraftTPReduce(tp.group, rank, b.output);
            AddTo(r.hidden, b.output);
        }
        KimiK3RMSNorm(r.hidden, w("dspark.norm.weight"), draftEps, r.normalized);
    };
    bool useGraph = GetFastllmEnv().cudaGraph && !ws.disabled;
    if (useGraph && !ws.tpRanks[0]->graphs[slot].exec) {
        tp.Run(uploadInputs);
        tp.Run([&](int rank) { body(rank); ForceDeviceSync(); });
        std::vector<int> ready(ranks);
        auto allReady = [&]() { return std::all_of(ready.begin(), ready.end(), [](int value) { return value != 0; }); };
        tp.Run([&](int rank) { ready[rank] = FastllmCudaGraphPrepareCaptureDevice(); });
        bool pool = allReady() && FastllmCudaGraphMemoryPoolBegin();
        bool ok = pool;
        if (pool) {
            tp.Run([&](int rank) { FastllmCudaClearThreadError(); ready[rank] = FastllmCudaGraphBeginCapture(); });
            ok = allReady();
            tp.Run([&](int rank) {
                if (!ready[rank]) return;
                if (ok) body(rank);
                ready[rank] = FastllmCudaGraphEndCapture(&ws.tpRanks[rank]->graphs[slot].graph) && !FastllmCudaGetThreadError();
            });
            ok = FastllmCudaGraphMemoryPoolEnd(ws.graphs[slot].reserved) && ok && allReady();
            if (ok) {
                tp.Run([&](int rank) { auto &g = ws.tpRanks[rank]->graphs[slot]; ready[rank] = FastllmCudaGraphInstantiate(g.graph, &g.exec); });
                ok = allReady();
            }
        }
        if (!ok) {
            tp.Run([&](int rank) { ws.tpRanks[rank]->graphs[slot].Clear(); FastllmCudaClearLastError(); FastllmCudaClearThreadError(); });
            ws.graphs[slot].Clear(); ws.disabled = true; useGraph = false;
        }
    }
    tp.Run([&](int rank) {
        uploadInputs(rank);
        auto &r = *ws.tpRanks[rank];
        if (useGraph) AssertInFastLLM(FastllmCudaGraphLaunch(r.graphs[slot].exec), "Draft TP graph launch failed.");
        else body(rank);
        FastllmCudaEventRecordCurrentThread(r.done);
    });
    for (size_t rank = 1; rank < tp.devices.size(); ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(ws.tpRanks[rank]->done);
    Copy(ws.tpRanks[0]->normalized, output);
    return true;
}

#endif

std::shared_ptr<NaiveN05FlashModel::DraftContext> NaiveN05FlashModel::CreateDraftContext() {
    // The request lifecycle and caller hold historyMutex. Reuse storage only;
    // reset every logical prefix, proposal and sampling state for a new request.
    auto context = std::make_shared<DraftContext>();
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
        bool pool = FastllmCudaGraphPrepareCaptureDevice() && FastllmCudaGraphMemoryPoolBegin();
        bool ok = false;
        if (pool) {
            FastllmCudaClearThreadError();
            if (FastllmCudaGraphBeginCapture()) {
                try { body(); }
                catch (...) {
                    FastllmCudaGraphEndCapture(&graph.graph);
                    FastllmCudaGraphMemoryPoolEnd(graph.reserved);
                    graph.Clear();
                    state.disabled = true;
                    throw;
                }
                bool clean = !FastllmCudaGetThreadError();
                ok = FastllmCudaGraphEndCapture(&graph.graph) && clean;
            }
            ok = FastllmCudaGraphMemoryPoolEnd(graph.reserved) && ok;
            if (ok) ok = FastllmCudaGraphInstantiate(graph.graph, &graph.exec);
        }
        if (!ok) {
            graph.Clear();
            state.disabled = true;
            FastllmCudaClearLastError();
            FastllmCudaClearThreadError();
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
        bool pool = FastllmCudaGraphPrepareCaptureDevice() && FastllmCudaGraphMemoryPoolBegin();
        bool ok = false;
        if (pool) {
            FastllmCudaClearThreadError();
            if (FastllmCudaGraphBeginCapture()) {
                try { body(); }
                catch (...) {
                    FastllmCudaGraphEndCapture(&p.graph.graph);
                    FastllmCudaGraphMemoryPoolEnd(p.graph.reserved);
                    p.graph.Clear(); p.disabled = true;
                    throw;
                }
                bool clean = !FastllmCudaGetThreadError();
                ok = FastllmCudaGraphEndCapture(&p.graph.graph) && clean;
            }
            ok = FastllmCudaGraphMemoryPoolEnd(p.graph.reserved) && ok;
            if (ok) ok = FastllmCudaGraphInstantiate(p.graph.graph, &p.graph.exec);
        }
        if (!ok) {
            p.graph.Clear(); p.disabled = true;
            FastllmCudaClearLastError(); FastllmCudaClearThreadError();
            return false;
        }
    }
    std::vector<float> ids(draftBlock + 1);
    FastllmCudaCopyFromDeviceToHost(ids.data(), p.ids.cudaData, ids.size() * sizeof(float));
    proposed.assign(ids.begin() + 1, ids.end());
    return true;
#else
    return false;
#endif
}

void NaiveN05FlashModel::ApplyDraftDevice() {
    if (tpDevices.empty()) ApplyDeviceMap(deviceMap, block_cnt, block_cnt);
    else ApplyDeviceMap({{"cuda:" + std::to_string(tpDevices.front()), 1}}, 1, 1);
}

Data &NaiveN05FlashModel::DraftWeight(const std::string &name) {
    Data &data = weight[name];
    return tpDevices.empty() || !data.multiDeviceData ? data :
        *data.multiDeviceDatas.at(tpDevices.front());
}

Data NaiveN05FlashModel::RunDraftTarget(const Data &inputIds, const Data &positions,
        std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &config,
        TargetCapture &capture) {
    return tpDevices.empty() ? RunTarget(inputIds, positions, kv, config, &capture) :
        ForwardTensorParallel(inputIds, positions, kv, config, &capture, capture.selection);
}

void NaiveN05FlashModel::CommitTargetCache(std::vector<std::pair<Data, Data>> &kv,
        int past, int count) {
#ifdef USE_CUDA
    auto commit = [&](int device) {
        for (int layer = 0; layer < block_cnt; ++layer) {
            const int length = (slidingLayers[layer] ? std::min(past, window - 1) : past) + count;
            auto &pair = kv[layer];
            Data &key = device < 0 ? pair.first : *pair.first.multiDeviceDatas.at(device);
            Data &value = device < 0 ? pair.second : *pair.second.multiDeviceDatas.at(device);
            key.Resize({1, length, key.dims[2]});
            value.Resize({1, length, value.dims[2]});
            if (slidingLayers[layer]) {
                FastllmCudaSetDevice(key.dataDeviceIds.at(0));
                FastllmCudaNaiveTrimCache(key, value, window - 1);
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
        AppendCache(context.kv[i].first, key, draftWindow + draftBlock);
        AppendCache(context.kv[i].second, value, draftWindow + draftBlock);
        // Retain bounded draft KV storage between proposal/commit rounds.
        // Copying the suffix discards capacity and reallocates ten buffers
        // on every round once the draft window is full.
        FastllmCudaNaiveTrimCache(context.kv[i].first, context.kv[i].second, draftWindow - 1);
    }
    context.committed = start + begin + length;
#endif
}

void NaiveN05FlashModel::CommitDraftContext(TargetCapture &capture, int tokens,
        DraftContext &context, std::vector<std::pair<Data, Data>> &kv) {
    ApplyDraftDevice();
    Data localCombined, localProjected, localHidden;
#ifdef USE_CUDA
    auto *buffers = context.workspace && tokens <= draftBlock + 1
        ? context.workspace.get() : nullptr;
    Data &combined = buffers ? buffers->combined : localCombined;
    Data &projected = buffers ? buffers->projected : localProjected;
    Data &hidden = buffers ? buffers->committedHidden : localHidden;
    std::vector<const Data *> inputs;
    for (int layer : draftTargetLayers) inputs.push_back(&capture.hidden.at(layer));
    bool joinedOnDevice = FastllmCudaNaiveDraftConcat(inputs, tokens, combined);
#else
    Data &combined = localCombined, &projected = localProjected, &hidden = localHidden;
    bool joinedOnDevice = false;
#endif
    if (!joinedOnDevice) {
        bool first = true;
        for (int layer : draftTargetLayers) {
            Data selected, joined;
            Split(capture.hidden.at(layer), 1, 0, tokens, selected);
            if (first) { Copy(selected, combined); first = false; }
            else { Cat(combined, selected, -1, joined); Copy(joined, combined); }
        }
    }
    DraftLinear(combined, weight["dspark.fc.weight"], projected);
    KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, hidden);
    if (capture.history) {
        auto &chunk = *capture.history;
        chunk.length = std::min(chunk.length, tokens);
        chunk.bytes = chunk.length * historyBytesPerToken;
        for (auto &pair : chunk.layers) {
            for (Data *tensor : {&pair.first, &pair.second}) {
                if (tensor->dims[1] != chunk.length) {
                    Data compact;
                    CopyHistoryTensor(*tensor, compact, chunk.length);
                    tensor->CopyFrom(compact);
                    tensor->lockInCPU = true;
                }
            }
        }
        CopyHistoryTensor(hidden, chunk.draftHidden, chunk.length);
        FinishHistoryChunk(kv, capture.history);
    }
    AppendDraftContext(hidden, context.committed, context);
}

Data NaiveN05FlashModel::RunDraft(int anchor, DraftContext &context) {
    Data normalized;
#ifdef USE_CUDA
    if (RunDraftTP(anchor, context, normalized)) return normalized;
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
            AppendCache(cache.first, k);
            AppendCache(cache.second, v);
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
    if (!context.restoredHidden.dims.empty()) {
        int length = context.restoredHidden.dims[1];
        AppendDraftContext(context.restoredHidden, context.committed - length, context);
        context.restoredHidden = Data();
    }
    int oldLength = kv[0].first.dims.empty() ? 0 : kv[0].first.dims[1];
    AssertInFastLLM(context.committed == oldLength, "Naive target/draft cache length mismatch.");
    // Tool constraints and content sampling change after each emitted prefix.
    // Keep these requests single-token until branch-local masks are available.
    bool constrained = config.tool_call_name_constraint_enabled ||
                       config.tool_call_parameter_name_constraint_enabled ||
                       config.tool_call_content_sampling_enabled;
    int limit = std::min(draftTokens, max_positions - oldLength - 1);
    if (config.output_token_limit > 0)
        limit = std::min(limit, config.output_token_limit - (oldLength - config.input_token_length) - 1);
    if (context.committed == 0 || context.kv.empty() || inputIds.dims[1] != 1 || config.output_logits || constrained || limit <= 0) {
        TargetCapture capture;
        // With history disabled, only the final draft window is ever used.
        // Preserve absolute position accounting while skipping invisible
        // prompt chunks' feature copies, projection and draft KV construction.
        const int draftStart = std::max(0, config.input_token_length - draftWindow + 1);
        capture.collectHidden = saveHistoryChat || oldLength + inputIds.dims[1] > draftStart;
        auto selection = SelectLogits(config);
        capture.selection = &selection;
        Data logits = RunDraftTarget(inputIds, positions, kv, config, capture);
        if (capture.collectHidden) CommitDraftContext(capture, inputIds.dims[1], context, kv);
        else context.committed += inputIds.dims[1];
        if (isIntermediateChunkedPrefill) return 0;
        return SampleTarget(logits, kv, config, lastTokens, retLogits, &selection);
    }
    Data draftHidden = RunDraft(anchor, context), selected, baseLogits;
    // DSpark predicts after the anchor at slot 0, unlike DFlash's masked-slot-only head.
    Split(draftHidden, 1, 0, limit, selected);
    baseLogits = RunDraftHead(selected);
    const int vocab = baseLogits.dims.back();
    const bool greedy = config.IsSimpleGreedy() && config.output_token_least <= 0;
    LastTokensUnit samplingTokens = lastTokens.units.empty() ? LastTokensUnit(config.last_n) : lastTokens.units[0];
    auto distribution = [&](Data &logits, int row, int position, const LastTokensUnit &history) {
        logits.ToDevice(DataDevice::CPU);
        float *values = (float *)logits.cpuData + (size_t)row * vocab;
        if (config.output_token_least > position - config.input_token_length) {
            if (eos_token_id >= 0 && eos_token_id < vocab) values[eos_token_id] = -1e30f;
            for (int id : eos_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
            for (int id : config.stop_token_ids) if (id >= 0 && id < vocab) values[id] = -1e30f;
        }
        return SpeculativeDistribution(values, vocab, config, history);
    };
    auto selection = SelectLogits(config, true);
    Data proposalPartial, proposalTop;
    std::vector<float> proposalCandidates(selection.count * 2);
    std::vector<int> proposed;
    std::vector<std::vector<float>> q;
    int previous = anchor;
    bool gpuProposal = greedy && draftConfidenceThreshold == 0 &&
        RunDraftProposalGraph(anchor, baseLogits, context, proposed);
    for (int step = 0; !gpuProposal && step < limit; ++step) {
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
        if (greedy) {
            Data top;
            TopK(logits, top, 1);
            top.ToDevice(DataDevice::CPU);
            previous = (int)((float *)top.cpuData)[0];
        } else {
#ifdef USE_CUDA
            if (logits.dataDevice == DataDevice::CUDA &&
                FastllmNaiveCanSelectLogits(vocab, 0, selection.count, false)) {
                int count = std::min(selection.count, vocab);
                FastllmCudaNaiveLogitsSelect(logits, 0, count, false, 1.0f, proposalPartial, proposalTop);
                FastllmCudaCopyFromDeviceToHost(proposalCandidates.data(), proposalTop.cudaData, proposalTop.GetBytes());
                q.push_back(SpeculativeTopKDistribution(proposalCandidates.data(), count, vocab, config));
            } else
#endif
                q.push_back(distribution(logits, 0, oldLength + step + 1, samplingTokens));
            previous = SampleSpeculativeDistribution(q.back(), context.Uniform());
        }
        proposed.push_back(previous);
        samplingTokens.Push(previous);
    }
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
    ApplyDraftDevice();
    samplingTokens = lastTokens.units.empty() ? LastTokensUnit(config.last_n) : lastTokens.units[0];
    int accepted = 0, next = -1;
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
            return distribution(logits, row, oldLength + row + 1, samplingTokens);
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
    int committed = accepted + 1;
    CommitTargetCache(kv, oldLength, committed);
    CommitDraftContext(capture, committed, context, kv);
    ++context.rounds;
    context.proposed += proposed.size();
    context.accepted += accepted;
    proposed.resize(accepted);
    proposed.push_back(next);
    for (int i = 1; i < (int)proposed.size(); ++i)
        context.pending.emplace_back(proposed[i - 1], proposed[i]);
    return proposed[0];
}

}
