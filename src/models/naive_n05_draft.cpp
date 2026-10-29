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
        Data base, probabilities, stepProbabilities;
        std::vector<float> parameters;
        std::vector<int> topKs;
        int rows = 1, steps = 0;
        void *sampler = nullptr;
        float temperature = -1, topP = -1;
        int topK = 0;
        bool disabled = false;
        void Clear() {
            for (auto &graph : graphs) graph.Clear();
            graphs.clear();
            FastllmCudaMtpDestroyDraftSampler(sampler);
            sampler = nullptr;
        }
    } sampling;
    struct ContextLayer { Data raw, key, value; };
    int device = -1;
    bool disabled = false;
    std::vector<void *> inputs;
    std::vector<bool> shortRows;
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
    else ApplyDeviceMap({{"cuda:" + std::to_string(tpDevices.front()), 1}}, 1, 1);
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
            else { key.Resize({1, past[i] + counts[i], key.dims[2]}); value.Resize({1, past[i] + counts[i], value.dims[2]}); }
        }
        for (auto &entry : updates) { FastllmCudaSetDevice(entry.first); FastllmCudaNaiveBatchCommitSwaVerify(entry.second); }
    };
    if (tpDevices.empty()) { ApplyDraftDevice(); commit(-1); }
    else {
        std::vector<std::exception_ptr> errors(tpDevices.size());
        tpWorkers.Run(tpDevices, [&](int rank) {
            FastllmCudaSetDevice(tpDevices[rank]); commit(tpDevices[rank]); ForceDeviceSync();
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
        for (size_t i = 0; i < caches.size(); ++i) for (int layer = 0; layer < block_cnt; ++layer) {
            int kept = slidingLayers[layer] ? std::min(past[i] + counts[i], window - 1) : past[i] + counts[i];
            auto &pair = caches[i]->at(layer);
            pair.first.Resize({1, kept, pair.first.dims[2]}); pair.second.Resize({1, kept, pair.second.dims[2]});
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
        AppendCache(context.kv[i].first, key, BFLOAT16, draftWindow + draftBlock);
        AppendCache(context.kv[i].second, value, BFLOAT16, draftWindow + draftBlock);
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
    AppendDraftContext(hidden, context.committed, context);
#ifdef USE_CUDA
    // TP workers use other per-thread streams but share the allocation pool.
    // Finish draft KV writes before retiring captured features/temporaries:
    // the next prefill chunk may immediately reuse their storage on a worker.
    if (tpDevices.size() > 1) FastllmCudaSyncCurrentThreadStream();
#endif
}

bool NaiveN05FlashModel::CommitDraftContexts(TargetCapture &capture, const std::vector<int> &lengths,
        const std::vector<int> &counts, const std::vector<DraftContext *> &contexts) {
#ifdef USE_CUDA
    if (draftTP || GetLowMemMode() || contexts.size() < 2 || contexts.size() != counts.size() ||
        lengths.size() != counts.size() || draftHeadDim <= 0 || draftHeadDim > 256 || draftHeadDim % 2) return false;
    ApplyDraftDevice();
    const std::vector<int> devices{FastllmCudaGetDevice()};
    const int width = draftKvHeads * draftHeadDim;
    int total = 0, captured = 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        auto *context = contexts[i];
        if (!context || counts[i] < 1 || counts[i] > lengths[i] || counts[i] >= draftWindow ||
            context->kv.size() != (size_t)draftLayers) return false;
        total += counts[i]; captured += lengths[i];
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA || cache->dataDeviceIds != devices ||
                cache->dims.size() != 3 || cache->dims[0] != 1 || cache->dims[2] != width || !cache->cudaData) return false;
    }
    for (int i = 0; i < draftLayers; ++i) {
        const auto name = "dspark.layers." + std::to_string(i) + ".self_attn.";
        auto merged = weight.weight.find(name + "mergeqkv.weight");
        if (merged != weight.weight.end()) {
            const Data &w = merged->second;
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims !=
                std::vector<int>{(draftHeads + 2 * draftKvHeads) * draftHeadDim, embed_dim}) return false;
        } else for (auto part : {"k_proj.weight", "v_proj.weight"}) {
            const Data &w = weight[name + part];
            if (w.dataType != BFLOAT16 || w.multiDeviceData || w.dims != std::vector<int>{width, embed_dim}) return false;
        }
        const Data &norm = weight[name + "k_norm.weight"];
        if (norm.dataType != FLOAT32 || norm.multiDeviceData || norm.dims != std::vector<int>{draftHeadDim}) return false;
    }
    std::vector<const Data *> inputs;
    for (int layer : draftTargetLayers) inputs.push_back(&capture.hidden.at(layer));
    Data all, combined, projected, hidden;
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
    std::vector<Data> raw(draftLayers), views(contexts.size() * draftLayers);
    std::vector<FastllmNaiveDraftKVUpdate> updates;
    for (int layer = 0; layer < draftLayers; ++layer) {
        const auto name = "dspark.layers." + std::to_string(layer) + ".self_attn.";
        auto merged = weight.weight.find(name + "mergeqkv.weight");
        if (merged != weight.weight.end()) {
            Data &w = merged->second; w.ToDevice(DataDevice::CUDA, devices);
            Data view; view.Resize({2 * width, embed_dim});
            view.FakeFrom(w, (size_t)draftHeads * draftHeadDim * embed_dim * sizeof(uint16_t));
            DraftLinear(hidden, view, raw[layer]);
        } else {
            Data key, value;
            DraftLinear(hidden, weight[name + "k_proj.weight"], key);
            DraftLinear(hidden, weight[name + "v_proj.weight"], value);
            Cat(key, value, -1, raw[layer]);
        }
        Data &norm = weight[name + "k_norm.weight"]; norm.ToDevice(DataDevice::CUDA, devices);
        int offset = 0;
        for (size_t i = 0; i < contexts.size(); ++i) {
            Data &view = views[layer * contexts.size() + i];
            view.Resize({1, counts[i], 2 * width}); view.FakeFrom(raw[layer], (size_t)offset * 2 * width * sizeof(uint16_t));
            auto &pair = contexts[i]->kv[layer];
            updates.push_back({&view, &norm, &pair.first, &pair.second, contexts[i]->committed}); offset += counts[i];
        }
    }
    AssertInFastLLM(FastllmCudaNaiveBatchDraftKV(updates, draftKvHeads, draftHeadDim, draftWindow,
        draftWindow + draftBlock, draftEps, draftTheta), "Invalid draft context batch.");
    for (size_t i = 0; i < contexts.size(); ++i) contexts[i]->committed += counts[i];
    if (tpDevices.size() > 1) FastllmCudaSyncCurrentThreadStream();
    return true;
#else
    return false;
#endif
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
    // Share projections and attention kernels across independent request caches.
    // Unsupported layouts retain the existing single-request fallback.
    if (draftTP || contexts.size() < 2 || anchors.size() != contexts.size() ||
        !GetCudaEmbedding() || GetLowMemMode() || draftBlock < 2 || draftBlock >= 32 ||
        draftHeadDim % 4 || draftHeadDim > 256) return false;
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    const int device = FastllmCudaGetDevice();
    Data &embedding = DraftWeight("model.embed_tokens.weight"), &mask = weight["dspark.mask_embedding"];
    if (embedding.dataType != BFLOAT16 || mask.dataType != BFLOAT16 || mask.Count(0) != embed_dim)
        return false;
    for (auto *context : contexts) {
        if (!context || context->kv.size() != (size_t)draftLayers) return false;
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second})
            if (cache->dataType != BFLOAT16 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds != std::vector<int>{device} || cache->dims.size() != 3 ||
                cache->expansionDims.size() != 3 || cache->expansionDims[1] < draftWindow + draftBlock)
                return false;
    }
    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    // A newly admitted request can enter the batched backbone before ever
    // running RunDraftGraph. Its proposal sampler still needs a private workspace.
    for (auto *context : contexts) if (!context->workspace) {
        context->workspace = std::make_shared<DraftWorkspace>();
        auto &state = *context->workspace;
        state.device = device;
        state.inputs = {embedding.cudaData, mask.cudaData};
        for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second})
            state.inputs.push_back(cache->cudaData);
        state.layers.resize(draftLayers);
        state.contextLayers.resize(draftLayers);
    }
    std::shared_ptr<DraftWorkspace> graphState;
    if (GetFastllmEnv().cudaGraph) {
        std::vector<void *> inputs{embedding.cudaData, mask.cudaData};
        std::vector<bool> shortRows;
        for (auto *context : contexts) {
            shortRows.push_back(std::min(context->committed, draftWindow - 1) + draftBlock <= 256);
            for (auto &pair : context->kv) for (Data *cache : {&pair.first, &pair.second})
                inputs.push_back(cache->cudaData);
        }
        auto found = std::find_if(draftBatchWorkspaces.begin(), draftBatchWorkspaces.end(),
            [&](const std::shared_ptr<DraftWorkspace> &state) {
                return state->device == device && state->inputs == inputs && state->shortRows == shortRows;
            });
        if (found != draftBatchWorkspaces.end()) {
            graphState = *found; draftBatchWorkspaces.erase(found);
        } else {
            graphState = std::make_shared<DraftWorkspace>();
            graphState->device = device; graphState->inputs = std::move(inputs);
            graphState->shortRows = std::move(shortRows);
        }
        draftBatchWorkspaces.push_back(graphState);
        // Retain only a few active cache bindings; rolling admission must not
        // keep graph workspaces alive for every request seen by the service.
        if (draftBatchWorkspaces.size() > 8) draftBatchWorkspaces.erase(draftBatchWorkspaces.begin());
    }
    Data eagerIds, eagerLive;
    Data &ids = graphState ? graphState->id : eagerIds;
    Data &live = graphState ? graphState->live : eagerLive;
    Data &result = graphState ? graphState->normalized : normalized;
    auto upload = [&](Data &target, DataType type, const void *host, const std::vector<int> &dims) {
        target.dataType = type; target.UpdateUnitSize(); target.Resize(dims);
        target.ToDevice(DataDevice::CUDA, {device}, false); target.Allocate(false);
        FastllmCudaCopyFromHostToDevice(target.cudaData, (void *)host, target.GetBytes());
    };
    std::vector<float> tokenIds(anchors.begin(), anchors.end());
    std::vector<int> lengths(contexts.size());
    for (size_t row = 0; row < contexts.size(); ++row) lengths[row] = contexts[row]->committed + 1;
    upload(ids, FLOAT32, tokenIds.data(), {1, (int)anchors.size()});
    upload(live, INT32, lengths.data(), {(int)anchors.size()});
    if (graphState && graphState->graphs[0].exec) {
        AssertInFastLLM(FastllmCudaGraphLaunch(graphState->graphs[0].exec), "Naive batch draft graph launch failed.");
        Copy(result, normalized); return true;
    }
    auto body = [&]() {
    struct Request { Data positions, live; };
    std::vector<Request> requests(contexts.size());
    auto append = [](Data &batch, Data &part, bool first) {
        if (first) Copy(part, batch);
        else { Data joined; Cat(batch, part, 1, joined); Copy(joined, batch); }
    };
    Data hidden, positions;
    FastllmCudaNaiveDraftInput(ids, embedding, mask, live, draftBlock, hidden, positions);
    for (size_t row = 0; row < contexts.size(); ++row) {
        requests[row].positions.Resize({1, draftBlock});
        requests[row].positions.FakeFrom(positions, row * draftBlock * sizeof(float));
        requests[row].live.Resize({1}); requests[row].live.FakeFrom(live, row * sizeof(int));
    }
    for (int layer = 0; layer < draftLayers; ++layer) {
        const std::string p = "dspark.layers." + std::to_string(layer) + ".";
        Data normed, qkv, qs, ks, vs, attention, output, gate, up, gateUp;
        KimiK3RMSNorm(hidden, weight[p + "input_layernorm.weight"], draftEps, normed);
        const bool merged = weight.weight.count(p + "self_attn.mergeqkv.weight");
        if (merged) DraftLinear(normed, weight[p + "self_attn.mergeqkv.weight"], qkv);
        else {
            DraftLinear(normed, weight[p + "self_attn.q_proj.weight"], qs);
            DraftLinear(normed, weight[p + "self_attn.k_proj.weight"], ks);
            DraftLinear(normed, weight[p + "self_attn.v_proj.weight"], vs);
        }
        std::vector<FastllmNaiveDraftRow> descriptors;
        for (size_t row = 0; row < contexts.size(); ++row) {
            auto &cache = contexts[row]->kv[layer];
            descriptors.push_back({&cache.first, &cache.second, &requests[row].positions,
                &requests[row].live, std::min(contexts[row]->committed, draftWindow - 1) + draftBlock <= 256});
        }
        Data packed, query, scores;
        if (!merged) {
            Data qk;
            Cat(qs, ks, -1, qk); Cat(qk, vs, -1, packed);
        }
        const bool batchedAttention = FastllmCudaNaiveBatchDraftQKV(merged ? qkv : packed,
                weight[p + "self_attn.q_norm.weight"], weight[p + "self_attn.k_norm.weight"],
                descriptors, query, draftHeads, draftKvHeads, draftHeadDim, draftWindow, draftEps, draftTheta) &&
            FastllmCudaNaiveBatchDraftAttention(query, descriptors, draftHeads, draftKvHeads,
                draftHeadDim, draftWindow, scores, attention);
        if (!batchedAttention) {
            for (size_t row = 0; row < contexts.size(); ++row) {
                auto &r = requests[row]; auto &cache = contexts[row]->kv[layer];
                Data raw, q, k, v, scores, part;
                const int begin = row * draftBlock, end = begin + draftBlock;
                if (merged) Split(qkv, 1, begin, end, raw);
                if (!merged || !FastllmCudaNaiveDraftQKV(raw,
                        weight[p + "self_attn.q_norm.weight"], weight[p + "self_attn.k_norm.weight"],
                        r.positions, r.live, cache.first, cache.second, q,
                        draftHeads, draftKvHeads, draftHeadDim, draftWindow, draftEps, draftTheta)) {
                    if (merged) {
                        const int qw = draftHeads * draftHeadDim, kw = draftKvHeads * draftHeadDim;
                        Split(raw, -1, 0, qw, q); Split(raw, -1, qw, qw + kw, k);
                        Split(raw, -1, qw + kw, qw + 2 * kw, v);
                    } else {
                        Split(qs, 1, begin, end, q); Split(ks, 1, begin, end, k); Split(vs, 1, begin, end, v);
                    }
                    q.Reshape({1, draftBlock * draftHeads, draftHeadDim});
                    k.Reshape({1, draftBlock * draftKvHeads, draftHeadDim});
                    KimiK3RMSNorm(q, weight[p + "self_attn.q_norm.weight"], draftEps, q);
                    KimiK3RMSNorm(k, weight[p + "self_attn.k_norm.weight"], draftEps, k);
                    q.Reshape({1, draftBlock, draftHeads * draftHeadDim});
                    k.Reshape({1, draftBlock, draftKvHeads * draftHeadDim});
                    FastllmCudaNaiveRope(q, r.positions, draftHeads, draftHeadDim, draftHeadDim, draftTheta);
                    FastllmCudaNaiveRope(k, r.positions, draftKvHeads, draftHeadDim, draftHeadDim, draftTheta);
                    FastllmCudaNaiveAppendVerifyCache(cache.first, cache.second, k, v, r.live, draftWindow);
                }
                const bool shortAttention = std::min(contexts[row]->committed, draftWindow - 1) + draftBlock <= 256;
                FastllmCudaNaiveDraftAttention(q, cache.first, cache.second, r.live,
                    draftHeads, draftKvHeads, draftHeadDim, draftWindow, shortAttention, scores, part);
                append(attention, part, row == 0);
            }
        }
        DraftLinear(attention, weight[p + "self_attn.o_proj.weight"], output); AddTo(hidden, output);
        KimiK3RMSNorm(hidden, weight[p + "post_attention_layernorm.weight"], draftEps, normed);
        if (weight.weight.count(p + "mlp.gateup_proj.weight")) {
            DraftLinear(normed, weight[p + "mlp.gateup_proj.weight"], gateUp);
            FastllmCudaNaiveDraftSwiGLU(gateUp, gate);
        } else {
            DraftLinear(normed, weight[p + "mlp.gate_proj.weight"], gate);
            DraftLinear(normed, weight[p + "mlp.up_proj.weight"], up); Silu(gate, gate); MulTo(gate, up);
        }
        DraftLinear(gate, weight[p + "mlp.down_proj.weight"], output); AddTo(hidden, output);
    }
    KimiK3RMSNorm(hidden, weight["dspark.norm.weight"], draftEps, result);
    };
    body();
    if (graphState && !graphState->disabled) {
        auto &graph = graphState->graphs[0];
        if (!graph.Capture(body, graphState->disabled)) {
            // Capture may have changed output storage. Recompute with the same
            // batched implementation; no request KV metadata has been committed.
            body();
        }
    }
    if (graphState) Copy(result, normalized);
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
                    &config.temperature, &config.top_k, &config.top_p, &previous, 1, vocab),
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

bool NaiveN05FlashModel::PrepareDraftBatchGraph(
        std::vector<std::unique_ptr<DraftRound>> &rounds, const Data &baseLogits) {
#ifdef USE_CUDA
    const int count = rounds.size(), steps = rounds.front()->limit, vocab = baseLogits.dims.back();
    if (!GetFastllmEnv().cudaGraph || steps < 1 || baseLogits.dims[1] != count * steps || vocab >= (1 << 24)) return false;
    std::vector<float> temperatures, topPs, parameters;
    std::vector<int> topKs;
    for (auto &owner : rounds) {
        const auto &r = *owner;
        if (r.limit != steps || r.config.IsSimpleGreedy() || r.config.output_token_least > 0 ||
            r.config.tool_call_grammar || r.config.tool_call_token_mask ||
            !SelectLogits(r.config).cudaSampling) return false;
        temperatures.push_back(r.config.temperature); topPs.push_back(r.config.top_p); topKs.push_back(r.config.top_k);
    }
    parameters = temperatures; parameters.insert(parameters.end(), topPs.begin(), topPs.end());
    ApplyDraftDevice();
#ifndef USE_ROCM
    FastllmCudaLinearGraphScope linearScope(draftLinearGraphWorkspaces[FastllmCudaGetDevice()]);
#endif
    const int device = FastllmCudaGetDevice(), words = (vocab + 31) / 32;
    Data &w1 = weight["dspark.markov_head.markov_w1.weight"], &w2 = weight["dspark.markov_head.markov_w2.weight"];
    w1.ToDevice(DataDevice::CUDA, std::vector<int>{device}); w2.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    std::vector<void *> inputs{w1.cudaData, w2.cudaData};
    std::shared_ptr<DraftWorkspace> state;
    auto found = std::find_if(draftProposalWorkspaces.begin(), draftProposalWorkspaces.end(),
        [&](const std::shared_ptr<DraftWorkspace> &s) {
            const auto &p = s->sampling;
            return s->device == device && s->inputs == inputs && p.rows == count && p.steps == steps &&
                p.base.dims == baseLogits.dims && p.parameters == parameters && p.topKs == topKs;
        });
    auto allocate = [&](Data &data, DataType type, const std::vector<int> &dims) {
        data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
        data.ToDevice(DataDevice::CUDA, {device}, false); data.Allocate(false);
    };
    if (found != draftProposalWorkspaces.end()) {
        state = *found; draftProposalWorkspaces.erase(found);
    } else {
        state = std::make_shared<DraftWorkspace>(); state->device = device; state->inputs = inputs;
        auto &p = state->sampling;
        p.rows = count; p.steps = steps; p.parameters = parameters; p.topKs = topKs;
        allocate(p.base, BFLOAT16, baseLogits.dims);
        allocate(p.ids, FLOAT32, {steps + 1, count});
        allocate(p.row, BFLOAT16, {1, count, vocab});
        allocate(p.probabilities, FLOAT32, {count, steps, vocab});
        allocate(p.stepProbabilities, FLOAT32, {count, vocab});
        p.mask.dataType = INT32; p.mask.Resize({count, words}); p.mask.Allocate();
        std::fill_n((uint32_t *)p.mask.cpuData, (size_t)count * words, ~uint32_t(0));
        p.mask.ToDevice(DataDevice::CUDA, std::vector<int>{device});
        p.sampler = FastllmCudaMtpCreateDraftSamplerBatch(count, vocab, temperatures.data(), topKs.data(), topPs.data());
        p.disabled = !p.sampler;
        p.graphs.resize(1);
    }
    draftProposalWorkspaces.push_back(state);
    if (draftProposalWorkspaces.size() > 4) draftProposalWorkspaces.erase(draftProposalWorkspaces.begin());
    auto &p = state->sampling;
    if (p.disabled) return false;
    std::vector<float> anchors;
    for (auto &r : rounds) anchors.push_back(r->anchor);
    FastllmCudaCopyFromHostToDevice(p.ids.cudaData, anchors.data(), count * sizeof(float));
    FastllmCudaCopyFromDeviceToDevice(p.base.cudaData, baseLogits.cudaData, baseLogits.GetBytes());
    auto body = [&]() {
        for (int step = 0; step < steps; ++step) {
            Data ids(FLOAT32, {1, count}); ids.FakeFrom(p.ids, (size_t)step * count * sizeof(float));
            EmbeddingDirect(ids, w1, p.latent); DraftLinear(p.latent, w2, p.bias);
            std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
            for (int row = 0; row < count; ++row) {
                dst.push_back((uint16_t *)p.row.cudaData + (size_t)row * vocab);
                src.push_back((uint16_t *)p.base.cudaData + (size_t)(row * steps + step) * vocab);
                bytes.push_back((size_t)vocab * sizeof(uint16_t));
            }
            AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), count),
                "Cannot gather graph draft logits.");
            FastllmCudaNaiveBatchDraftSamplingLogits(p.row, p.bias, p.mask, p.logits);
            AssertInFastLLM(FastllmCudaMtpDraftSamplerRunBatch(p.sampler, (float *)p.logits.cudaData,
                (float *)p.stepProbabilities.cudaData, (float *)p.ids.cudaData + (size_t)(step + 1) * count,
                (uint64_t)step * 4), "Graph batch draft sampling failed.");
            for (int row = 0; row < count; ++row) {
                dst[row] = (float *)p.probabilities.cudaData + (size_t)(row * steps + step) * vocab;
                src[row] = (float *)p.stepProbabilities.cudaData + (size_t)row * vocab;
                bytes[row] = (size_t)vocab * sizeof(float);
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
    std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
    for (int row = 0; row < count; ++row) {
        auto &r = *rounds[row]; r.vocab = vocab; r.greedy = false;
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        allocate(r.context.proposalProbs, FLOAT32, {steps, vocab});
        dst.push_back(r.context.proposalProbs.cudaData);
        src.push_back((float *)p.probabilities.cudaData + (size_t)row * steps * vocab);
        bytes.push_back((size_t)steps * vocab * sizeof(float));
    }
    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), count),
        "Cannot commit graph proposal probabilities.");
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
    for (auto &owner : rounds) {
        auto &r = *owner; auto config = r.config; config.tool_call_token_mask.reset();
        r.vocab = vocab; r.greedy = config.IsSimpleGreedy();
        r.selection = SelectLogits(r.config, true); r.selection.count = 0; r.selection.cudaLogits = true;
        if (!r.greedy) allocate(r.context.proposalProbs, FLOAT32, {r.limit, vocab});
    }
    Data base, latent, bias, logits, probs;
    for (int step = 0; step < steps; ++step) {
        std::vector<int> active;
        std::vector<float> ids, temperatures, topPs;
        std::vector<int> topKs;
        bool sampling = false, greedy = false;
        for (int i = 0; i < (int)rounds.size(); ++i) if (step < rounds[i]->limit) {
            auto &r = *rounds[i]; active.push_back(i);
            ids.push_back(step ? r.proposed.back() : r.anchor);
            temperatures.push_back(r.greedy ? 1.f : r.config.temperature);
            topPs.push_back(r.greedy ? 1.f : r.config.top_p); topKs.push_back(r.greedy ? 1 : r.config.top_k);
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
                dst.push_back((float *)r.context.proposalProbs.cudaData + (size_t)step * vocab);
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
        for (Data *data : {&target, &proposal}) {
            data->ToDevice(DataDevice::CUDA, logits.dataDeviceIds, false); data->Allocate(false);
        }
        std::vector<void *> dst; std::vector<const void *> src; std::vector<size_t> bytes;
        std::vector<int> ids, ks, output(count * rows), accepted(count);
        std::vector<float> temperatures, ps;
        for (int b = 0; b < count; ++b) {
            const int i = entry.second[b]; auto &r = *rounds[i];
            dst.push_back((float *)target.cudaData + (size_t)b * rows * vocab);
            src.push_back((const float *)logits.cudaData + (size_t)offsets[i] * vocab);
            bytes.push_back((size_t)rows * vocab * sizeof(float));
            if (!greedy) {
                dst.push_back((float *)proposal.cudaData + (size_t)b * drafts * vocab);
                src.push_back(r.context.proposalProbs.cudaData); bytes.push_back((size_t)drafts * vocab * sizeof(float));
                ids.insert(ids.end(), r.proposed.begin(), r.proposed.end());
                temperatures.insert(temperatures.end(), rows, r.config.temperature);
                ps.insert(ps.end(), rows, r.config.top_p); ks.insert(ks.end(), rows, r.config.top_k);
            }
        }
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(dst.data(), src.data(), bytes.data(), dst.size()),
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
            AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.dataType == FLOAT32 &&
                logits.dataDeviceIds == context.proposalProbs.dataDeviceIds,
                "Naive CUDA verification logits are on the wrong device.");
            FastllmCudaSetDevice(logits.dataDeviceIds.at(0));
            const int rows = proposed.size() + 1;
            std::vector<float> temperatures(rows, config.temperature), topPs(rows, config.top_p);
            std::vector<int> topKs(rows, config.top_k), sampled(rows);
            AssertInFastLLM(FastllmCudaMtpRejectionSampling((float *)logits.cudaData,
                (float *)context.proposalProbs.cudaData, temperatures.data(), topKs.data(), topPs.data(),
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
        const int draftStart = std::max(0, config.input_token_length - draftWindow + 1);
        capture.collectHidden = oldLength + inputIds.dims[1] > draftStart;
        auto selection = SelectLogits(config);
        capture.selection = &selection;
        Data logits = RunDraftTarget(inputIds, positions, kv, config, capture);
        if (capture.collectHidden) CommitDraftContext(capture, inputIds.dims[1], context, kv);
        else context.committed += inputIds.dims[1];
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

}
