#pragma once
#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "models/mimo_tool_grammar.h"
#include "pd.h"
#include "json11.hpp"
#include "executor.h"
#include "utils/decode_graph.h"
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_map>
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/naive-n05-draft-backend.h"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mxfp4-ep-generic.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-event.h"
#include "devices/cuda/fastllm-cuda-token-constraint.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/multicuda/fastllm-draft-verify.h"
#endif

namespace fastllm {
namespace naive_draft_detail {
inline int DraftGraphBucket(int count, int limit) {
    int bucket = 1;
    while (bucket < count && bucket <= INT_MAX / 2) bucket *= 2;
    return std::max(count, std::min(bucket, limit));
}

inline void ClearDraftPrefixHidden(Data &hidden) {
    // Data's default assignment would discard the owning GPU pointer.
    hidden.FreeSpace();
    hidden.dims.clear();
    hidden.strides.clear();
    hidden.expansionDims.clear();
}

inline void DraftLinear(Data &input, Data &weight, Data &output) {
    // The block backbone benefits from tensor-core GEMM even at 2..7 rows.
    // MatMulTransB uses the same BF16 weights and FP32 accumulation, and
    // avoids the generic Linear small-batch GEMV dispatch.
    // Non-owning weight views must not enter Linear's persistent bias cache.
    if (weight.isFake || input.Count(0) / input.dims.back() > 1) MatMulTransB(input, weight, output);
    else Linear(input, weight, Data(), output);
}

#ifdef USE_CUDA
inline void GatherDraftHeadInput(const Data &hidden, const std::vector<int> &limits, int block, Data &input) {
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
using naive_draft_detail::DraftLinear;
#ifdef USE_CUDA
struct NaiveN05FlashModel::DraftMoeScratch {
    Data fp32, logits, index, score, routed, gateUp, gate;
    Data w1, w2, w3, currentInput, currentOutput;
    std::map<int, std::vector<Data *>> experts;
    std::vector<Data *> biases;
};

struct NaiveN05FlashModel::DraftWorkspace {
    struct Layer {
        Data normed, q, k, v, qkv, query, packed, attention, output, reduced, gate, up, gateUp, scores;
        FastllmDraftLayerProjection attentionProjection, mlpProjection;
    };
    FastllmDraftKernelPlan proposalHead, contextKernel;
    FastllmDraftReplay replay;
    std::unique_ptr<DraftMoeScratch> moe;
    struct Graph {
        // The CUDA graph allocation pool has one capture session per process.
        // DP owners can replay independently after their captures finish.
        inline static std::mutex captureMutex;
        FastllmNaiveKVBodyGraph kvBody;
        void *graph = nullptr, *exec = nullptr, *acceptedExec = nullptr;
        std::vector<void *> reserved;
        bool canContinue = false;
        void Clear() {
            FastllmCudaGraphExecDestroy(kvBody.exec);
            FastllmCudaGraphDestroy(kvBody.graph);
            if (kvBody.kvReady) FastllmCudaEventDestroy(kvBody.kvReady);
            kvBody = {};
            canContinue = false;
            FastllmCudaGraphExecDestroy(acceptedExec);
            acceptedExec = nullptr;
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
                    const bool clean = !FastllmCudaGetThreadError() && !disabled;
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
        Data base, probabilities, stepProbabilities, target, verifyProposal, hidden, confidence;
        int rows = 1, steps = 0;
        std::shared_ptr<void> constraints;
        bool constrained = false;
        void *sampler = nullptr;
        bool ownsSampler = true, confidenceEnabled = false;
        float temperature = -1, topP = -1;
        int topK = 0;
        bool disabled = false;
        float *hostIds = nullptr;
        void *idsReady = nullptr;
        void PrepareConstraintRows(const std::vector<DraftRound *> &rounds, int vocab);

        std::shared_ptr<void> markovCache;
#ifndef USE_ROCM
        void BindMarkovCache(SamplingProposal &owner, const Data &w1, const Data &w2) {
            // These graph variants already serialize their shared IDs, bias and
            // sampler. They can also share one bounded cache of immutable rows.
            if (!owner.markovCache) owner.markovCache = FastllmCudaCreateDraftMarkovCache(w1, w2);
            markovCache = owner.markovCache;
        }
#endif
        void ProjectMarkovStep(int step, Data &embedding, Data &projection, bool needLatent) {
            Data tokens(FLOAT32, {1, rows});
            tokens.FakeFrom(ids, size_t(step) * rows * sizeof(float));
#ifndef USE_ROCM
            if (!needLatent && FastllmCudaDraftIndexedBf16Gemv(tokens, embedding, projection, bias, markovCache.get())) return;
#endif
            EmbeddingDirect(tokens, embedding, latent);
            DraftLinear(latent, projection, bias);
        }

        bool TryProjectedHead(Data &embedding, Data &weight) {
            return !constrained && TryCudaDraftHead(DraftProjectedHead{base, embedding, weight,
                ids, probabilities, sampler}, steps);
        }
        void SampleMarkovStep(int step) {
            const int vocab = base.dims.back();
            if (constrained) FastllmCudaTokenConstraintMask(constraints.get(), step, (uint32_t *)mask.cudaData);
            float *stepProbs = (float *)probabilities.cudaData + (size_t)step * vocab;
            float *idsOut = (float *)ids.cudaData + (size_t)(step + 1) * rows;
            if (!FastllmCudaMtpDraftSamplerRunBF16BiasBatch(sampler, base.cudaData, bias.cudaData,
                    (const uint32_t *)mask.cudaData, step, steps, stepProbs, idsOut,
                    (uint64_t)step * 4, steps * vocab)) {
                FastllmCudaNaiveBatchDraftSamplingLogits(base, bias, mask, logits, step, steps);
                AssertInFastLLM(FastllmCudaMtpDraftSamplerRunBatchStrided(sampler, (float *)logits.cudaData,
                    stepProbs, idsOut, (uint64_t)step * 4, steps * vocab), "Graph batch draft sampling failed.");
            }
            if (constrained) FastllmCudaTokenConstraintAdvance(constraints.get(), step,
                idsOut, step == steps - 1);
        }

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
    void *fused = nullptr;
    bool fusedUnavailable = false, fusedContextReady = false;
    bool fusedHeadReady = false, fusedHeadUnavailable = false;
    bool targetCommitReady = false;
    TargetCapture *pendingContext = nullptr;
    std::vector<DraftContext *> pendingContextOwners;
    void *pinnedHeadOutput = nullptr;
    int pinnedHeadCapacity = 0;
    std::vector<const void *> fusedContexts;
    // A speculative backbone may be reused only by these exact request
    // owners at these committed lengths. Retain owners until all readers
    // finish; weak identity additionally rejects canceled/recycled requests.
    std::shared_ptr<DraftWorkspace> nextGraph;
    std::vector<std::weak_ptr<DraftContext>> nextOwners;
    std::vector<int> nextAnchors, nextLengths;
    bool nextReady = false, nextInFlight = false, nextConsumed = false;
    std::vector<std::shared_ptr<DraftContext>> nextRetained;
    bool nextProposed = false;
    uint64_t predictedSeed = 0;
    float lastHeadTemperature = 0, predictedTemperature = 0;
    void *backboneReady = nullptr; // Borrowed from the current producer.
    void *prefetchStream = nullptr, *prefetchReady = nullptr;
    bool prefetchSubmitted = false;
    bool pendingBackbone = false;
    bool CanPrefetch(int count) const {
        if (!nextGraph || nextGraph->disabled || nextGraph->rankRows.empty() ||
            nextGraph->rankRows.front() != count || tpRanks.size() != 8 ||
            nextGraph->tpRanks.size() != tpRanks.size()) return false;
        for (size_t rank = 0; rank < tpRanks.size(); ++rank) {
            const auto &local = *tpRanks[rank];
            if (!nextGraph->tpRanks[rank]->graphs[0].exec || local.inputCapacity < count ||
                !local.prefetchStream || !local.prefetchReady) return false;
        }
        return true;
    }
    void DrainPendingBackbone() {
        if (!pendingBackbone) return;
        const int previous = FastllmCudaGetDevice();
        for (auto &local : tpRanks) {
            FastllmCudaSetDevice(local->device);
            FastllmCudaEventSynchronize(local->prefetchReady);
        }
        FastllmCudaSetDevice(previous);
        pendingBackbone = false;
    }
    bool ConsumeLookahead(const std::vector<int> &anchors, const std::vector<DraftContext *> &contexts) {
        if (!nextReady || nextAnchors != anchors || nextOwners.size() != contexts.size() ||
            nextLengths.size() != contexts.size() || fusedContexts.size() != contexts.size()) return false;
        for (size_t row = 0; row < contexts.size(); ++row)
            if (!contexts[row] || nextOwners[row].lock().get() != contexts[row] ||
                fusedContexts[row] != contexts[row] || contexts[row]->committed + 1 != nextLengths[row]) return false;
        nextReady = false;
        nextConsumed = true;
        return true;
    }
    void JoinLookahead() {
        if (!nextInFlight) return;
        const int previous = FastllmCudaGetDevice();
        for (auto &rank : tpRanks) if (rank->prefetchSubmitted) {
            FastllmCudaSetDevice(rank->device);
            FastllmCudaEventSynchronize(rank->prefetchReady);
            rank->prefetchSubmitted = false;
        }
        FastllmCudaSetDevice(previous);
        nextReady = nextInFlight = nextConsumed = false; nextRetained.clear();
    }
    std::vector<void *> inputs;
    // The graph reads request KV and publishes into shared owner scratch. A different
    // request or geometry must drain it before either allocation can be reused.
    struct OwnerReplay {
        std::mutex mutex;
        std::shared_ptr<DraftContext> request;
        std::shared_ptr<DraftWorkspace> bucket;
        int anchor = -1, length = 0;
        void Join() {
            if (bucket) FastllmCudaEventSynchronize(bucket->done);
            bucket.reset(); request.reset(); anchor = -1;
        }
    } ownerReplay;
    void *ownerPacket = nullptr;
    bool UploadOwner(const DraftContext &context, int anchor, int length, Data &table) {
        const size_t slots = 2 * context.kv.size(), bytes = (slots + 1) * sizeof(uint64_t);
        if (!ownerPacket || bytes > table.GetBytes()) return false;
        auto *bindings = static_cast<uint64_t *>(ownerPacket);
        for (size_t i = 0; i < context.kv.size(); ++i) {
            bindings[2 * i] = reinterpret_cast<uintptr_t>(context.kv[i].first.cudaData);
            bindings[2 * i + 1] = reinterpret_cast<uintptr_t>(context.kv[i].second.cudaData);
        }
        const float token = anchor;
        auto *tail = reinterpret_cast<unsigned char *>(bindings + slots);
        std::memcpy(tail, &token, sizeof(token));
        std::memcpy(tail + sizeof(token), &length, sizeof(length));
        return FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(table.cudaData, ownerPacket, bytes);
    }
    bool lowLatencyBody = false;
    std::vector<int> rankRows;
    Data id, live, positions, hidden, normalized, cacheTable, batchKV;
    Data inputPacket;
    void *pinnedInput = nullptr, *inputUploaded = nullptr;
    int inputCapacity = 0;
    bool inputInFlight = false;
    std::vector<const void *> uploadedCacheBindings;
    void *uploadedCacheTable = nullptr;
    void UploadInputs(const std::vector<int> &anchors, const std::vector<int> &lengths, int capacity) {
        // Each rank owns its pinned packet until this upload finishes. The
        // graph and every smaller batch retain the same token/length offsets.
        if (inputInFlight) FastllmCudaEventSynchronize(inputUploaded);
        const int count = anchors.size();
        if (inputCapacity < count) {
            if (pinnedInput) FastllmCudaHostFree(pinnedInput);
            inputCapacity = std::max(count, capacity);
            pinnedInput = FastllmCudaHostMalloc(size_t(inputCapacity) * 2 * sizeof(int));
            AssertInFastLLM(pinnedInput, "Cannot allocate draft input packet.");
            std::memset(pinnedInput, 0, size_t(inputCapacity) * 2 * sizeof(int));
            if (!inputUploaded) inputUploaded = FastllmCudaEventCreate();
            inputPacket.dataType = INT32; inputPacket.UpdateUnitSize();
            inputPacket.Resize({2, inputCapacity});
            inputPacket.ToDevice(DataDevice::CUDA, {device}, false); inputPacket.Allocate(false);
            id.FreeSpace(); live.FreeSpace();
            id.FakeFrom(inputPacket, 0); id.dataType = FLOAT32; id.UpdateUnitSize();
            live.FakeFrom(inputPacket, size_t(inputCapacity) * sizeof(int));
        }
        id.Resize({1,count}); live.Resize({count});
        for (int row = 0; row < count; ++row) ((float *)pinnedInput)[row] = anchors[row];
        std::memcpy((int *)pinnedInput + inputCapacity, lengths.data(), size_t(count) * sizeof(int));
        AssertInFastLLM(FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
            inputPacket.cudaData, pinnedInput, size_t(inputCapacity + count) * sizeof(int)),
            "Draft input upload failed.");
        FastllmCudaEventRecordCurrentThread(inputUploaded); inputInFlight = true;
    }
    Data combined, projected, committedHidden, contextAll, prefixStorage, contextSelected, contextRaw, contextPointers;
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
    void *ready = nullptr, *done = nullptr, *verifyLogitsReady = nullptr;
    void *readySource = nullptr, *readyStream = nullptr;
    // A different device may wait after the producer changes CPU threads.
    // Publish from a stable explicit stream on the owning device.
    void RecordPeerEvent(void *&event) {
        if (!event) event = FastllmCudaEventCreate();
        if (!readySource) readySource = FastllmCudaEventCreate();
        if (!readyStream) readyStream = FastllmCudaStreamCreate();
        FastllmCudaEventRecordForPeer(event, readySource, readyStream);
    }
    std::vector<void *> Pointers() const {
        std::vector<void *> out;
        for (const Data *p : {&id, &live, &positions, &hidden, &normalized, &combined, &cacheTable, &batchKV})
            out.push_back((p == &id || p == &live || !p->isFake) ? p->cudaData : nullptr);
        for (const auto &l : layers) {
            for (const Data *p : {&l.normed, &l.q, &l.k, &l.v, &l.qkv, &l.query, &l.packed,
                    &l.attention, &l.output, &l.reduced, &l.gate, &l.up, &l.gateUp, &l.scores})
                out.push_back(p->isFake ? nullptr : p->cudaData);
            l.attentionProjection.AppendGraphPointers(out);
            l.mlpProjection.AppendGraphPointers(out);
        }
        if (moe) {
            for (const Data *p : {&moe->fp32, &moe->logits, &moe->index, &moe->score,
                                  &moe->routed, &moe->gateUp, &moe->gate}) out.push_back(p->cudaData);
        }
        return out;
    }
    ~DraftWorkspace();
};
// Draft subgroup state is independent of the target's TP communicator.
struct NaiveN05FlashModel::DraftTPState {
    std::vector<int> devices;
    int attentionDivisor = 1;
    bool dataParallel = false;
    std::shared_ptr<FastllmCudaMimoEp8Mega> moe;
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
#endif
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
    std::vector<std::pair<Data, Data>> *ownerCommitKV = nullptr;
    bool targetCommitted = false, draftCommitted = false, singleOwnerBody = false;
    bool collectiveDynamicKV = false;
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
    TargetCapture *earlyCapture = nullptr;
    std::function<void(const std::vector<TargetCapture::DraftInput> &,const std::function<void()> &)> enqueueVerification;
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

    int NextBackboneAnchor(int accepted, int next) const {
        // This lookahead advances neither sampling state nor committed KV.
        // A full next-round budget and no terminal token avoid wasted work at
        // request tails; unsupported proposals use the ordinary next call.
        if (!proposalOwner || limit != model.draftTokens || accepted < 0 || accepted > (int)proposed.size() || next < 0)
            return -1;
        const int committed = oldLength + accepted + 1;
        int remaining = std::min(model.draftTokens,model.max_positions-committed-1);
        if (config.output_token_limit > 0)
            remaining = std::min(remaining,config.output_token_limit-(committed-config.input_token_length)-1);
        if (remaining < model.draftTokens) return -1;
        auto terminal = [&](int token) {
            return token == model.eos_token_id || model.eos_token_ids.count(token) || config.stop_token_ids.count(token);
        };
        if (terminal(next) || std::any_of(proposed.begin(),proposed.begin()+accepted,terminal) ||
            std::any_of(context.pending.begin(),context.pending.end(),[&](const auto &item) { return terminal(item.second); }))
            return -1;
        return next;
    }
    int Publish(int accepted, int next) {
        ++context.rounds;
        context.proposed += proposed.size();
        context.accepted += accepted;
        for (int i = 1; i < accepted; ++i)
            context.pending.emplace_back(proposed[i - 1], proposed[i]);
        if (!accepted) return next;
        context.pending.emplace_back(proposed[accepted - 1], next);
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
}
