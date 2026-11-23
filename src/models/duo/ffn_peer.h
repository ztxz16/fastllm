#pragma once

#include "models/duo/model.h"
#include "models/minimax_h3_image/fp8.h"

namespace fastllm::duo {
// Split independent FFN input rows. Both devices use full FP8 weights and
// per-row activation scales, so no partial sums or cross-rank scale reductions are needed.
class FfnPeer {
  public:
    static std::unique_ptr<FfnPeer> create(int primary, int peer, const ModelSource &, const Config &,
                                           const std::map<std::string, std::string> &shards);
    ~FfnPeer();
    Tensor run(Runtime &primary, const Tensor &x, const std::string &up, const std::string &down,
               minimax_h3_image::Fp8Weights &upWeights, minimax_h3_image::Fp8Weights &downWeights);

  private:
    struct Impl;
    explicit FfnPeer(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace fastllm::duo
