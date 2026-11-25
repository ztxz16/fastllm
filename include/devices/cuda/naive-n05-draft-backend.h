#pragma once
#include <vector>
#include <utility>
#include <cstdint>
#include <cstddef>

struct FastllmNaiveDraftRow;
struct FastllmDraftLayerProjection;
namespace fastllm {
class Data;

// Logical tensor partition and the device executing this rank.
struct DraftLayerParallel {
    const std::vector<int> &devices;
    int rank, tpSize;
};

struct DraftCollectiveLayer {
    bool enabled = false;
    FastllmDraftLayerProjection *storage = nullptr;
    const Data *live = nullptr, *bindings = nullptr;
    int layer = 0;
};

struct DraftMlpWeights {
    Data *gateUp;
    Data &down;
};

// Gate/up -> SwiGLU -> down -> optional TP sum -> residual add -> weighted RMSNorm.
// normalized and output may alias. A decline leaves caller tensors untouched;
// an error after submission propagates. Collective implementations must agree
// on admission across the group before any rank submits work.
bool TryCudaDraftMlp(const DraftLayerParallel &parallel, int requests, const Data &normalized,
    const DraftMlpWeights &weights, Data &residual, const Data &gamma, float eps,
    Data &middle, Data &scratch, Data &output, const DraftCollectiveLayer *collective = nullptr);

// Draft heads consume either normalized hidden rows with prepared vocabulary shards,
// or owner-local projected BF16 logits. Markov sampling retains the full q for verification.
struct DraftVocabShard {
    int device;
    std::pair<int,int> columns;
    Data &weight, &staging, &logits;
    Data *owner; // Optional pitched destination shared by all vocabulary shards.
};
void ProjectCudaDraftHeadShard(const DraftVocabShard &shard, const Data &local, int hidden,
    const std::vector<const void *> &sources, const std::vector<int> &rowOffsets,
    const std::vector<std::size_t> &bytes);

struct DraftProjectedHead {
    const Data &base, &embedding, &projection;
    Data &ids, &probabilities;
    void *sampler;
};
struct DraftShardedHead {
    void *state;
    int rank, row, anchor;
    const void *normalized;
    uint64_t seed;
    float temperature;
};
bool TryCudaDraftHead(const DraftProjectedHead &head, int steps);
bool TryCudaDraftHead(const DraftShardedHead &head, int steps);

struct DraftAttentionGeometry {
    int heads, kvHeads, headDim, window;
    float eps, theta;
};

struct DraftAttentionWeights {
    Data *qkv;
    Data &o, &qNorm, &kNorm, &postNorm;
};

// Reads the current KV binding and length on each replay. Writes the speculative KV
// suffix, residual and post-attention normalized input. Scratch tensors are disjoint,
// and splitScratch can be shared by layers serialized on the owner stream. A decline submits no work.
bool TryCudaDraftAttention(const DraftLayerParallel &parallel, const DraftAttentionGeometry &geometry,
    const std::vector<FastllmNaiveDraftRow> &requests, const Data *bindings, Data &normalized,
    const DraftAttentionWeights &weights,
    Data &residual, Data &qkv, Data &attention, Data &branch, Data &splitScratch,
    const DraftCollectiveLayer *collective = nullptr);

bool PreferCudaDraftReplicaMatMul(const Data &input, const Data &weight);

#ifdef FASTLLM_ENABLE_DRAFT_FUSED
bool CudaDraftHeadSupportsShard(const Data &normalized, int device, int ranks, int requests,
    int steps, int hidden, int vocab);
bool PrepareCudaDraftShardedHead(void *state, const std::vector<int> &devices,
    const std::vector<std::pair<int,int>> &ranges, const Data &head, Data &w1, Data &w2,
    Data &confidence, Data &bias, double budget);
bool ExportCudaDraftShardedHead(void *state, const Data &hidden, int steps, int vocab,
    const std::vector<float> &temperatures, const std::vector<int> &limits, Data &q,
    int64_t *ids, float *confidence);
// TP can fuse attention and MLP together. The model supplies logical geometry
// and borrowed tensors in rank-major, then layer-major order.
struct DraftBodyShape {
    int attentionTP, headsPerRank, kvHeadsPerRank, headDim, rows, hidden, layers, window;
};
struct DraftBodyWeights {
    const Data *qkv, *o, *inputNorm, *qNorm, *kNorm, *gateUp, *down, *postNorm;
};
bool CudaDraftBodySupportsShape(int ranks, const DraftBodyShape &shape, int requests);
void *CreateCudaDraftBody(const std::vector<int> &devices, const DraftBodyShape &shape,
    const std::vector<DraftBodyWeights> &weights, float eps, float theta);
void RunPreparedCudaDraftBody(void *state, int rank, int device, int requests, int rows, const Data &ids,
    const Data &embedding, const Data &mask, const Data &live, const Data &cacheTable, Data &gamma,
    float eps, Data &hidden, Data &positions, Data &output);
bool CudaDraftUseBody(int requests);
int CudaDraftBodyCapacity(int requests, int maxBatch);
bool CudaDraftCanShardContext(const std::vector<int> &devices, int targetLayers, int layers,
    int hidden, int headDim, int window, const Data &fc, const Data &gamma);
bool PrepareCudaDraftContext(void *state, int device, const Data &fc, const Data &gamma);
#endif
} // namespace fastllm
