#pragma once
#include "fastllm.h"
#include <cstring>
#include <functional>

struct FastllmDraftLayerSupport {
    bool attention, mlp, reduceNorm;
};
FastllmDraftLayerSupport FastllmCudaDraftLayerSupport(int ranks, int requests, int steps, int heads,
                                                     int kvHeads, int headDim, int window);

struct FastllmDraftLayerProjection {
    fastllm::Data packed, scratch;
    const void *sources[2]{};

    void AppendGraphPointers(std::vector<void *> &pointers) const {
        for (const auto *data : {&packed, &scratch})
            pointers.push_back(data->isFake ? nullptr : data->cudaData);
    }
};

bool FastllmCudaDraftAttention7(fastllm::Data &activation, fastllm::Data &hidden, const fastllm::Data &qkv,
                                const fastllm::Data &out, const fastllm::Data &qNorm,
                                const fastllm::Data &kNorm, const fastllm::Data &gamma,
                                const fastllm::Data &live, const fastllm::Data &bindings, int layer,
                                float eps, float theta, int gpu, FastllmDraftLayerProjection &storage);

bool FastllmCudaDraftMlp7(const fastllm::Data &input, fastllm::Data &hidden, fastllm::Data &output,
                          const fastllm::Data &gateUp, const fastllm::Data &down, const fastllm::Data &gamma,
                          float eps, int gpu, FastllmDraftLayerProjection &storage);

// Uploaded alongside the root's KV bindings for each replay.
struct FastllmDraftSamplingRequest {
    uint64_t seed;
    float *probabilities;
    float temperature;
    int anchor;

    void AppendTo(std::vector<uint64_t> &bindings) const {
        static_assert(sizeof(*this) % sizeof(uint64_t) == 0);
        const size_t offset = bindings.size();
        bindings.resize(offset + sizeof(*this) / sizeof(uint64_t));
        std::memcpy(bindings.data() + offset, this, sizeof(*this));
    }
};
struct FastllmDraftHeadCall {
    void *state;
    FastllmDraftSamplingRequest request;
};
struct FastllmDraftKernelPlan {
    std::shared_ptr<void> handle;
    bool unavailable = false;
};
struct FastllmDraftGraphRank {
    void *exec, *bindings;
};

// A ticket belongs to one request. The weak owner protects the captured graph;
// rank executables are resolved again after invalidation or recapture.
struct FastllmDraftReplay {
    void Begin(bool withHead) {
        ++generation;
        if (!withHead) owner.reset();
    }
    void Bind(const std::shared_ptr<void> &graph, void *head,
              std::function<FastllmDraftGraphRank(int)> lookup);
    bool Consume(uint64_t ticket, int length, const FastllmDraftHeadCall &call);
    uint64_t Issue(const FastllmDraftSamplingRequest &used);
    std::shared_ptr<void> ForCommit(uint64_t ticket, int anchor, void *head, const void *probs) const;
    uint64_t Complete(int length);
    FastllmDraftSamplingRequest request{};
    std::function<FastllmDraftGraphRank(int)> rank;
private:
    std::weak_ptr<void> owner;
    void *graphHead = nullptr;
    uint64_t generation = 1, issued = 0;
    int committed = -1;
};

void FastllmCudaLaunchDraftProposal(void *state, int rank, const fastllm::Data &normalized,
                                   const fastllm::Data &bindings, int layers);
void FastllmCudaReadDraftProposal(void *state, int *tokens);

bool FastllmCudaPrepareDraftEmbedding(const fastllm::Data &id, const fastllm::Data &live,
                                  const fastllm::Data &embedding, const fastllm::Data &mask,
                                  const fastllm::Data &gamma, fastllm::Data &hidden, fastllm::Data &positions,
                                  fastllm::Data &normed, float eps, int gpu);

bool FastllmCudaCanContinueDraftGraph(void *graph);

using FastllmDraftRankRunner = std::function<void(const std::function<void(int)> &)>;
struct FastllmDraftContextRank {
    const fastllm::Data *feature;
    std::vector<std::pair<fastllm::Data, fastllm::Data>> *kv;
    void *done;
};
struct FastllmDraftContextShape {
    int layers, features, hidden, headDim, steps, window, attentionRanks;
};
struct FastllmDraftContinuation {
    FastllmDraftReplay *replay;
    uint64_t issued;
    uint64_t &nextTicket;
    void *head;
    const void *probabilities;
};

bool FastllmCudaCommitDraftFeatures(const std::vector<int> &devices,
        const std::vector<FastllmDraftContextRank> &ranks, fastllm::WeightMap &weights,
        FastllmDraftKernelPlan &plan, FastllmDraftContextShape shape, bool preparing, int count, int position,
        float eps, float theta, int anchor, FastllmDraftContinuation continuation,
        const FastllmDraftRankRunner &run, const std::function<void *()> &publish);

struct FastllmDraftProjectionRank { fastllm::Data *input, *output; void *done; };
bool FastllmCudaProjectDraftFeatures(const std::vector<int> &devices,
        const std::vector<FastllmDraftProjectionRank> &ranks, fastllm::Data &combined, fastllm::Data &fc,
        fastllm::Data &norm, fastllm::Data &projected, fastllm::Data &hidden, float eps,
        const FastllmDraftRankRunner &run, const std::function<void *()> &publish);
bool FastllmCudaProposeDraft(const std::vector<int> &devices, const std::vector<std::pair<int,int>> &ranges,
        fastllm::WeightMap &weights, FastllmDraftKernelPlan &plan, int hidden, int steps, int limit, int anchor,
        float temperature, bool preparing, uint64_t &ticket, int length, int &vocab, std::vector<int> &tokens,
        const std::function<float *(int, int, int)> &samplingMemory,
        const std::function<FastllmDraftReplay *()> &replay,
        const std::function<bool(const FastllmDraftHeadCall &)> &run);
#ifndef USE_ROCM
bool FastllmCudaTryDraftReduceAddNorm(int device, float eps, fastllm::Data &hidden, fastllm::Data &branch,
                                      fastllm::Data &gamma, fastllm::Data &output);
#endif
