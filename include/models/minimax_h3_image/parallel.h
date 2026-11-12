#pragma once
#include "models/minimax_h3_image/model.h"

namespace fastllm::minimax_h3_image {
// Text TP and DiT TP/SP. Reference encoding and sampling stay on rank 0;
// independent resident VAE decoder tiles are distributed across ranks.
class ParallelModel {
  public:
    ParallelModel(const std::string &, const std::vector<int> &, const MemoryPolicy &, double gpuLimit,
                  nomas::RowReduction, bool sequenceParallel = false, const std::string &quantizedWeights = "");
    ~ParallelModel();
    Tensor encode(const std::vector<int> &, const std::vector<minimax_h3::VisionInput> &, Trace);
    void prepare(Progress);
    void prepareText(const Tensor &);
    Tensor forward(const Tensor &, const Tensor &, int height, int width, int step, Trace,
                   const Conditioning *);
    const std::vector<float> &sigmas() const;
    void releaseStreamedWeights();
    Tensor decodeTiles(const Tensor &, int height, int width, int batch);

  private:
    struct Rank;
    nomas::ParallelGroup group_;
    std::vector<std::unique_ptr<Rank>> ranks_;
};
} // namespace fastllm::minimax_h3_image
