#pragma once
#include "models/hy_nomas/parallel.h"

namespace fastllm { namespace nomas {
// Each persistent rank thread owns its decoder and CUDA runtime. The DiT,
// sampler and world history stay on rank 0, independent of decoding precision.
class ParallelPixelDecoder {
public:
    ParallelPixelDecoder(const std::string &, const ParallelOptions &, DType);
    ~ParallelPixelDecoder();
    Tensor decode(const Tensor &, const Tensor &, int, int, int);
    void unload();
private:
    struct Rank;
    ParallelGroup group_;
    std::vector<std::unique_ptr<Rank>> ranks_;
};
}} // namespace fastllm::nomas
