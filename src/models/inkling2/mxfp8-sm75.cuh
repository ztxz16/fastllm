#pragma once

#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>

namespace fastllm::inkling2::sm75 {

constexpr int Experts = 128;
using Half = cutlass::half_t;
template<int TileM>
using DenseKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Half, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Half, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<TileM, 128, 32>, cutlass::gemm::GemmShape<(TileM == 16 ? 16 : TileM / 2), 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>, cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2>::GemmKernel;

struct ExpertWeight {
    const uint8_t *codes;
    const float *scales;
};

// CUTLASS owns the grouped scheduler, shared tiles and MMA loop. Replace only
// its global B load. Eight adjacent E4M3 codes share one exact power-of-two scale.
template<class Iterator> struct WeightIterator {
    using Element = Half;
    using Layout = typename Iterator::Layout;
    using AccessType = typename Iterator::AccessType;
    using Fragment = typename Iterator::Fragment;
    using Map = typename Iterator::ThreadMap;
    static_assert(Map::kElementsPerAccess == 8);

    const uint8_t *codes;
    const float *scales;
    int stride, column, columns, posK;
    bool enabled = true;

    CUTLASS_DEVICE WeightIterator(Layout layout, Half *ptr, cutlass::MatrixCoord extent,
                                  int thread, cutlass::MatrixCoord origin) {
        const auto w = *reinterpret_cast<const ExpertWeight *>(ptr);
        auto offset = Map::initial_offset(thread);
        codes = w.codes;
        scales = w.scales;
        stride = layout.stride(0);
        column = origin.column() + offset.strided();
        columns = extent.column();
        posK = offset.contiguous(); // the dispatch requires K divisible by the 32-wide tile
    }
    CUTLASS_DEVICE void clear_mask(bool clear) { if (clear) enabled = false; }
    CUTLASS_DEVICE WeightIterator &operator++() { posK += 32; return *this; }
    CUTLASS_DEVICE void load(Fragment &frag) {
        if (!enabled) return; // CUTLASS clears the fragment before loading
        #pragma unroll
        for (int s = 0; s < Map::Iterations::kStrided; ++s) {
            int n = column + s * Map::Delta::kStrided;
            if (n >= columns) continue;
            #pragma unroll
            for (int c = 0; c < Map::Iterations::kContiguous; ++c) {
                size_t idx = size_t(n) * stride + posK + c * Map::Delta::kContiguous;
                uint2 bytes = *reinterpret_cast<const uint2 *>(codes + idx);
                half2 scale = __float2half2_rn(scales[idx / 32] * 256.f);
                int slot = (s * Map::Iterations::kContiguous + c) * 8;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    unsigned word = j < 2 ? bytes.x : bytes.y;
                    // Place two sign/magnitude bytes in the high bytes of adjacent half lanes.
                    unsigned bits = __byte_perm(word, 0, (j % 2) ? 0x3424 : 0x1404);
                    bits = (bits & 0x80008000u) | ((bits & 0x7f007f00u) >> 1);
                    half2 pair = __hmul2(__halves2half2(__ushort_as_half(bits), __ushort_as_half(bits >> 16)), scale);
                    memcpy(reinterpret_cast<char *>(&frag) + (slot + 2 * j) * sizeof(Half), &pair, sizeof(pair));
                }
            }
        }
    }
};

// Add each short tensor-core dot to its running sum with FP32 round-to-nearest.
template<class Warp> struct RoundedMma : Warp {
    using Acc = typename Warp::FragmentC;
    CUTLASS_DEVICE void operator()(Acc &out, const typename Warp::TransformedFragmentA &a,
                                   const typename Warp::TransformedFragmentB &b, const Acc &acc) const {
        Acc zero, partial;
        zero.clear();
        Warp::operator()(partial, a, b, zero);
        #pragma unroll
        for (int i = 0; i < Acc::kElements; ++i) out[i] = acc[i] + partial[i];
    }
};

template<int TileM> struct Grouped {
    using Dense = DenseKernel<TileM>;
    using Base = typename Dense::Mma;
    using Policy = typename Base::Policy;
    using Rounded = cutlass::gemm::threadblock::MmaPolicy<RoundedMma<typename Policy::Operator>,
        typename Policy::SmemPaddingA, typename Policy::SmemPaddingB, Policy::kPartitionsK>;
    using Mma = cutlass::gemm::threadblock::MmaPipelined<
        typename Base::Shape, typename Base::IteratorA, typename Base::SmemIteratorA,
        WeightIterator<typename Base::IteratorB>, typename Base::SmemIteratorB,
        float, typename Base::LayoutC, Rounded>;
    using Kernel = cutlass::gemm::kernel::GemmGrouped<Mma, typename Dense::Epilogue,
        typename Dense::ThreadblockSwizzle, Dense::kGroupScheduleMode>;
    using Op = cutlass::gemm::device::GemmGrouped<Kernel>;
};

struct Plan {
    int counts[Experts], offsets[Experts];
    ExpertWeight weights[Experts];
    cutlass::gemm::GemmCoord shapes[Experts];
    Half *a[Experts], *b[Experts];
    float *d[Experts];
    int64_t ldK[Experts], ldN[Experts];
};

// Each expert owns a contiguous interval in both activation matrices.
static __global__ void prefixRoutes(Plan *plan) {
    int offset = 0;
    for (int e = 0; e < Experts; ++e) {
        plan->offsets[e] = offset;
        offset += plan->counts[e];
    }
}

static __global__ void prepareProjection(Plan *plan, half *a, float *d, uint8_t **codes, float **scales, int n, int k) {
    const int e = threadIdx.x;
    plan->shapes[e] = {plan->counts[e], n, k};
    plan->a[e] = reinterpret_cast<Half *>(a) + size_t(plan->offsets[e]) * k;
    plan->weights[e] = {codes[e], scales[e]};
    plan->b[e] = reinterpret_cast<Half *>(plan->weights + e);
    plan->d[e] = d + size_t(plan->offsets[e]) * n;
    plan->ldK[e] = k;
    plan->ldN[e] = n;
}

template<int TileM> void project(Plan *plan, half *a, float *d, uint8_t **codes, float **scales,
                                int n, int k, int blocks, cudaStream_t stream) {
    prepareProjection<<<1, Experts, 0, stream>>>(plan, a, d, codes, scales, n, k);
    using Gemm = typename Grouped<TileM>::Op;
    Gemm gemm;
    typename Gemm::Arguments args(plan->shapes, Experts, blocks, {1.f, 0.f}, plan->a, plan->b,
                                  plan->d, plan->d, plan->ldK, plan->ldK, plan->ldN, plan->ldN);
    auto status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string("Inkling SM75 grouped GEMM: ") + cutlassGetStatusString(status));
}
} // namespace fastllm::inkling2::sm75
