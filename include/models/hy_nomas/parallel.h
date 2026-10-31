#pragma once
#include "models/hy_nomas/runtime.h"

namespace fastllm { namespace nomas {

// One independent CUDA runtime/weight cache per rank. Rank 0 runs on the
// submitting thread; other ranks reuse the LLM backend's persistent workers.
// Callers serialize run(), and submit every collective in the same rank order.
class ParallelGroup {
public:
    explicit ParallelGroup(const std::vector<int> &devices);
    ~ParallelGroup();
    ParallelGroup(const ParallelGroup &) = delete;
    ParallelGroup &operator=(const ParallelGroup &) = delete;
    int size() const;
    void run(const std::function<void(int)> &);
    Tensor broadcast(const Tensor &root, int rank);
    Tensor gatherColumns(const Tensor &, int rank, int groups = 1);
    void sum(Tensor &, int rank);
    // Sum FP32 partials before rounding. Reduce-scatter in FP32, then gather
    // rounded shards, avoiding a full FP32 all-reduce for 16-bit outputs.
    Tensor sumCast(Runtime &, Tensor partial, const Tensor *bias, DType output, int rank);
    // Queue a row chunk on an independent communication stream. Inputs are
    // retained until wait() inserts the dependency back into the forward stream.
    void sumCastAsync(const Tensor &partial, const Tensor *bias, Tensor &output, int rank);
    void wait(int rank);
    // Independent jobs (e.g. VAE tiles), cyclically assigned to the same workers.
    // Returns ordered tensors on rank 0; peer copies finish before sources retire.
    std::vector<Tensor> mapToRoot(size_t count, const std::function<Tensor(int,size_t)> &);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Column parallel: replicate input, shard output (optionally gather).
// Row parallel: consume sharded input, sum partial outputs, add bias once.
// Norms/residuals stay replicated. This contract is independent of the model.
struct ParallelContext {
    ParallelGroup *group = nullptr;
    int rank = 0;
    int size() const { return group ? group->size() : 1; }
    void columnWeights(Weights &, const std::string &prefix, int groups = 1) const;
    void rowWeights(Weights &, const std::string &prefix) const;
    Tensor column(Weights &, const Tensor &, const std::string &prefix, bool gather = false) const;
    Tensor row(Runtime &, Weights &, const Tensor &, const std::string &prefix) const;
    // Chunked GEMMs may write FP32 partials into row slices with linearInto,
    // then reduce the entire output, rounding only after summing rank partials.
    Tensor finishRow(Runtime &, Weights &, Tensor partial, const std::string &prefix, DType output) const;
    void finishRowAsync(Runtime &, Weights &, const Tensor &partial, const std::string &prefix,
                        Tensor &output) const;
};

// CLI device indices refer to CUDA-visible devices, not physical host indices.
std::vector<int> parseParallelDevices(const std::string &);

}} // namespace fastllm::nomas
