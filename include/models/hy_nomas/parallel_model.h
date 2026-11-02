#pragma once
#include "models/hy_nomas/parallel.h"

namespace fastllm { namespace nomas {
// One runtime, DiT and decoder per persistent rank thread. Sampling and text
// encoding remain on rank 0; snapshots own a separate KV cache for every rank.
class ParallelModel {
public:
    ParallelModel(const std::string &, const ParallelOptions &, DType);
    ~ParallelModel();
    void prepareCamera(const Tensor &,int spatial);
    Tensor forward(const Tensor &input,const Tensor &features,float timestep,int t,int h,int w,int offset,
                   std::vector<DiTCache> *cache,bool commit,bool camera,
                   std::vector<DiTCache> *workspace = nullptr);
    Tensor decode(const Tensor &, const Tensor &, int, int, int);
    void unload();
private:
    struct Rank;
    ParallelGroup group_;
    std::vector<std::unique_ptr<Rank>> ranks_;
};
}} // namespace fastllm::nomas
