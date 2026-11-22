#pragma once
#include "execution.h"

namespace fastllm::cuda::low_latency {
// Contiguous BF16 row-major tensors, no repacking. The caller supplies already
// normalized activations. All projections accumulate in FP32; gate/up, SiLU,
// product and final projection retain their individual BF16 rounding points.
// output is a local partial: TP reduction and residual addition belong to the
// caller and must occur in that order. No collective or model state is owned.
struct DenseShape { int rows = 0, hidden = 0, intermediate = 0; };
struct DenseBindings {
    ConstBuffer input, gate, up, down;
    Buffer intermediate, output;
};
class DensePlan {
public:
    bool Ready() const { return gate_ && down_; }
    const DenseShape &Shape() const { return shape_; }
    int RowTile() const { return tile_; }
    // The caller owns intermediate [rows, intermediate] and output tensors.
    // These are the same buffers needed by the unfused path, not hidden scratch.
    size_t WorkspaceBytes() const { return 0; }
private:
    DenseShape shape_{};
    const void *gate_ = nullptr, *down_ = nullptr;
    int device_ = -1, tile_ = 0;
    size_t gateShared_ = 0, downShared_ = 0;
    friend cudaError_t PrepareDense(const DenseShape &, DensePlan &, cudaStream_t, PrepareOptions);
    friend bool CanRunDense(const DensePlan &, const DenseBindings &);
    friend cudaError_t RunDense(const DensePlan &, const DenseBindings &, cudaStream_t);
};
// Instantiated width pairs are explicit. Unsupported widths use the caller's
// original operators; logical batch is tiled rather than padded/repacked.
bool SupportsDenseShape(const DenseShape &);
// Warmup only. Failure leaves the destination plan and every tensor unchanged.
// BF16 MMA requires SM80+; image and shared-memory checks also apply.
cudaError_t PrepareDense(const DenseShape &, DensePlan &, cudaStream_t, PrepareOptions = {});
bool CanRunDense(const DensePlan &, const DenseBindings &);
// No allocation/preparation/capture. Concurrent calls use independent bindings.
// Failure after a launch is fatal to this execution; it must never fall back.
cudaError_t RunDense(const DensePlan &, const DenseBindings &, cudaStream_t);
} // namespace fastllm::cuda::low_latency
