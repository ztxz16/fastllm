#pragma once
#include "execution.h"
#include <cstddef>
#include <cstdint>

namespace fastllm::cuda::low_latency {

// All strides count elements. Inputs/weights are row-major BF16, gamma is FP32,
// output is FP32 containing BF16-rounded projections. No weight repacking.
// z = bf16(input + optional branch)
// h = bf16(bf16(z * rsqrt(mean(z*z) + eps)) * gamma)
// output = float(bf16(h @ weight.T)), with FP32 accumulation.
// This is deliberately distinct from RMSNorm implementations with one rounding.
struct NormLinearShape {
    int rows = 0, inner = 0, channels = 0;
    int64_t inputStride = 0, branchStride = 0, weightStride = 0, outputStride = 0;
};

struct NormLinearBindings {
    ConstBuffer input, gamma, weight, branch;
    Buffer output, mirror;
    // Optional FP32 mirror of each output row, e.g. a caller-owned vocabulary
    // slice. Its pointer already includes the slice offset. Caller owns peer
    // access/lifetime and waits for every producing stream before consuming it.
    int64_t mirrorStride = 0;
    float epsilon = 1e-5f;
};

class NormLinearPlan {
public:
    bool Ready() const { return kernel_ != nullptr; }
    const NormLinearShape &Shape() const { return shape_; }
    size_t WorkspaceBytes() const { return 0; }
    int RowTile() const { return tile_; }
private:
    NormLinearShape shape_{};
    const void *kernel_ = nullptr;
    int device_ = -1, blocks_ = 0, tile_ = 0;
    size_t shared_ = 0;
    bool mirror_ = false;
    friend cudaError_t PrepareNormLinear(const NormLinearShape &, bool, NormLinearPlan &, cudaStream_t, PrepareOptions);
    friend bool CanRunNormLinear(const NormLinearPlan &, const NormLinearBindings &);
    friend cudaError_t RunNormLinear(const NormLinearPlan &, const NormLinearBindings &, cudaStream_t);
};

// Pure shape query: no CUDA calls, allocation, compilation or work submission.
// A physical tile contains <=8 rows; grid.y covers the entire logical batch.
bool SupportsNormLinearShape(const NormLinearShape &);
// Call before capture on the intended device. Checks the loaded kernel image
// and resources. On failure the destination plan is unchanged. No GPU memory,
// tensor writes, autotuning, graph capture or hidden workspace is involved.
cudaError_t PrepareNormLinear(const NormLinearShape &, bool mirror, NormLinearPlan &,
                             cudaStream_t stream, PrepareOptions options = {});
// Pure binding/capacity/overlap check. A rejected call cannot write any tensor.
bool CanRunNormLinear(const NormLinearPlan &, const NormLinearBindings &);
// Reusable concurrently with independent bindings/streams; never prepares on
// demand. Invalid bindings return cudaErrorInvalidValue before any launch.
// Pointers must reside on the prepared device (mirror may be peer accessible).
cudaError_t RunNormLinear(const NormLinearPlan &, const NormLinearBindings &, cudaStream_t);
} // namespace fastllm::cuda::low_latency
