#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include <cstdint>
#include <stdexcept>

namespace {
using Bf16 = cutlass::bfloat16_t;
using Row = cutlass::layout::RowMajor;
using Column = cutlass::layout::ColumnMajor;
using Bf16Output = cutlass::epilogue::thread::LinearCombination<Bf16, 8, float, float>;

struct SoftcapParams { float limit; };

template <typename Fragment> struct Bf16Softcap {
    using Arguments = SoftcapParams;

    CUTLASS_HOST_DEVICE Fragment operator()(const Fragment &acc, const Arguments &params) const {
        Fragment result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Fragment::kElements; ++i) {
            // The checkpoint rounds the projection to BF16 before softcapping.
            result[i] = params.limit * tanhf(float(Bf16(acc[i])) / params.limit);
        }
        return result;
    }
};
using CappedOutput = cutlass::epilogue::thread::LinearCombinationGeneric<
    Bf16Softcap, float, 2, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;

template <int Channels, class OutputOp>
bool project(const void *input, const void *weight, typename OutputOp::ElementOutput *output,
             int rows, int width, int columns, const typename OutputOp::Params &epilogue) {
    using Gemm = cutlass::gemm::device::Gemm<Bf16, Row, Bf16, Column, typename OutputOp::ElementOutput, Column,
        float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<Channels, 64, 64>, cutlass::gemm::GemmShape<Channels / 2, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>, OutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    typename Gemm::Arguments args({columns, rows, width}, {static_cast<const Bf16 *>(weight), width},
        {static_cast<const Bf16 *>(input), width}, {output, columns}, {output, columns}, epilogue);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    auto status = gemm(args, nullptr, cudaStreamPerThread);
    if (status != cutlass::Status::kSuccess) throw std::runtime_error(cutlassGetStatusString(status));
    return true;
}

template <class OutputOp>
bool projectVocab(const void *input, const void *weight, typename OutputOp::ElementOutput *output,
                  int rows, int width, int vocab, const typename OutputOp::Params &epilogue) {
    constexpr int elements = OutputOp::kCount;
    constexpr size_t alignment = elements * sizeof(typename OutputOp::ElementOutput);
    if (rows <= 0 || width <= 0 || width % 8 || vocab <= 0 || vocab % elements ||
        ((uintptr_t(input) | uintptr_t(weight)) & 15) || uintptr_t(output) % alignment)
        return false;
    if (rows > 32)
        return project<64, OutputOp>(input, weight, output, rows, width, vocab, epilogue);
    return project<32, OutputOp>(input, weight, output, rows, width, vocab, epilogue);
}

bool attentionProjection(const void *input, const void *weight, void *output,
                         int rows, int width, int columns) {
    if (rows <= 0 || width <= 0 || columns <= 0 || width % 8 || columns % 8 ||
        ((uintptr_t(input) | uintptr_t(weight) | uintptr_t(output)) & 15)) return false;
    // QKV expansions have enough channels to amortize the larger tile once
    // its row dimension is at least half occupied. Both tiles use the same K order.
    if (rows > 32 && columns / 2 >= width)
        return project<64, Bf16Output>(input, weight, static_cast<Bf16 *>(output), rows, width, columns, {1.f, 0.f});
    return project<32, Bf16Output>(input, weight, static_cast<Bf16 *>(output), rows, width, columns, {1.f, 0.f});
}

bool cappedLogits(const void *input, const void *weight, float *logits,
                  int rows, int width, int vocab, float limit) {
    if (!(limit > 0)) return false;
    CappedOutput::Params epilogue;
    epilogue.limit = limit;
    return projectVocab<CappedOutput>(input, weight, logits, rows, width, vocab, epilogue);
}

bool draftLogits(const void *input, const void *weight, void *output,
                 int rows, int width, int vocab) {
    using DraftOutput = cutlass::epilogue::thread::LinearCombination<Bf16, 2, float, float>;
    return projectVocab<DraftOutput>(input, weight, static_cast<Bf16 *>(output), rows, width, vocab, {1.f, 0.f});
}
} // namespace

const FastllmBf16MmaProjections *FastllmCudaGetBf16MmaProjections() {
    static constexpr FastllmBf16MmaProjections ops{attentionProjection, draftLogits, cappedLogits};
    return FastllmCudaRuntimeArch() == 120 ? &ops : nullptr;
}
