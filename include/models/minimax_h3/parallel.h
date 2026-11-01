#pragma once
#include "models/minimax_h3/runtime.h"

namespace fastllm { namespace minimax_h3 {
// Coordinates model forwards; all partition/communication semantics live in
// nomas::ParallelContext. CPU conditioning, sampler and media remain on rank 0.
class ParallelModel {
public:
    ParallelModel(const std::string &, const std::vector<int> &, bool fp32, DType videoPrecision, VideoVAE &,
                  nomas::RowReduction);
    ~ParallelModel();
    Tensor encode(const std::vector<int> &, const std::vector<VisionInput> &, Trace);
    void prepare(const std::vector<std::vector<float>> &, bool reference,
                 std::function<void(int,int)>);
    std::pair<Tensor,Tensor> forward(const Tensor &,const Tensor &,const Tensor &,
                                   const PackedLayout &,Trace);
    void unload();
    void unloadVideo();
    std::vector<Tensor> videoTiles(size_t, const std::function<Tensor(VideoVAE &,size_t)> &);
private:
    struct Rank;
    nomas::ParallelGroup group_;
    std::vector<std::unique_ptr<Rank>> ranks_;
};
}} // namespace fastllm::minimax_h3
