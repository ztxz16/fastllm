#pragma once

#include "fastllm.h"
#include <memory>

namespace fastllm {
struct Mxfp4MarlinRank;

// Check all resident shards and reserve rank-local workspace before changing any weights.
std::shared_ptr<Mxfp4MarlinRank> PrepareMxfp4MarlinRank(int device,
        const std::vector<std::vector<Data *>> &layers, int hidden, int experts, int topk);
void PackMxfp4MarlinRank(Mxfp4MarlinRank &rank, const std::vector<std::vector<Data *>> &layers);
bool RunMxfp4MarlinTP(Mxfp4MarlinRank &rank, int layer, const Data &input,
        const Data &ids, const Data &scores, Data &activation, Data &output);
}
