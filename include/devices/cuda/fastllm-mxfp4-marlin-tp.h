#pragma once

#include "fastllm.h"
#include "devices/cuda/fastllm-mxfp8-projection.h"
#include <memory>

namespace fastllm {
struct Mxfp4MarlinRank;
struct Mxfp4MarlinNorm {
    Data &hidden;
    const Data &gamma;
    Data &output;
    float eps;
    FastllmMxfp8Activation next;
    void *feature = nullptr;
    int featureStride = 0;
    bool ready = false;
    void BindOutputs(const Data &input, Data *featureTensor, int featureIndex, int featureCount);
};

// Check all resident shards and reserve rank-local workspace before changing any weights.
std::shared_ptr<Mxfp4MarlinRank> PrepareMxfp4MarlinRank(int device,
        const std::vector<std::vector<Data *>> &layers, int hidden, int experts, int topk);
void PackMxfp4MarlinRank(Mxfp4MarlinRank &rank, const std::vector<std::vector<Data *>> &layers);
struct Mxfp4MarlinRouter {
    Mxfp4MarlinRank *rank;
    Data &ids, &scores;
    const Data &bias;
    int topk;
    bool normalize;
    float scale;

    bool Project(const Data &input, const Data &weight, const Data *hidden,
                 const Data *gamma, float eps) const;
    bool Select(const Data &probabilities) const;
};
bool RunMxfp4MarlinTP(Mxfp4MarlinRank &rank, int layer, const Data &input,
        const Data &ids, const Data &scores, Data &activation, Data &output, bool *tpReduced = nullptr,
        Mxfp4MarlinNorm *norm = nullptr, bool routesReady = false);
}
