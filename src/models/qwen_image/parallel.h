#pragma once
#include "models/qwen_image/runtime.h"

namespace fastllm::qwen_image {
// DiT-only TP. Text/vision, reference VAE and sampling remain on rank 0.
// Each rank owns its runtime, compressed weight shard and attention cache.
class ParallelDiT {
  public:
    ParallelDiT(const std::string &, const std::vector<int> &, bool fp32);
    ~ParallelDiT();
    bool canKeep(size_t weights, size_t workspace);
    size_t rootLoadedBytes() const;
    void prepare(const Condition &, bool fullCache);
    Tensor forward(const Tensor &, float timestep, int height, int width);
    // Always discard request state; optionally release resident weight shards.
    void release(bool unload);

  private:
    struct Rank;
    nomas::ParallelGroup group_;
    std::vector<std::unique_ptr<Rank>> ranks_;
};
} // namespace fastllm::qwen_image
