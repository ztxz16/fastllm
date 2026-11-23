#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace fastllm::cuda::low_latency::detail {
using BF16 = __nv_bfloat16;
__device__ __forceinline__ float RoundBF16(float x) { return float(__float2bfloat16_rn(x)); }
// A coalesced SIMT specialization for very small row counts. Each warp streams
// two independent output rows; all CTAs normalize once and reuse that vector.
template<int Rows, bool Mirror = false, bool Dense = false>
__global__ __launch_bounds__(256) void NormLinearSimt(const BF16 *input, const BF16 *branch,
        const float *gamma, const BF16 *weight, float *output, int inner, int channels, float eps,
        float *mirror, int64_t mirrorStride, int rows, int64_t inputStride,
        int64_t branchStride, int64_t weightStride, int64_t outputStride) {
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int rowBase = blockIdx.y * Rows;
    extern __shared__ __align__(16) unsigned char dynamic[];
    BF16 *normalized = reinterpret_cast<BF16 *>(dynamic);
    __shared__ float partial[Rows][8], inverse[Rows];
    for (int row = 0; row < Rows; ++row) {
        float sum = 0;
        for (int k = t; k < inner; k += 256) {
            const size_t i = size_t(row) * inner + k;
            float value = 0;
            if constexpr (Dense) {
                value = branch ? RoundBF16(float(input[i]) + float(branch[i])) : float(input[i]);
            } else if (rowBase + row < rows) {
                value = float(input[(Dense ? size_t(row) * inner : int64_t(rowBase + row) * inputStride) + k]);
                if (branch) value = RoundBF16(value + float(branch[(Dense ? size_t(row) * inner : int64_t(rowBase + row) * branchStride) + k]));
            }
            normalized[i] = __float2bfloat16_rn(value);
            sum += value * value;
        }
        for (int shift = 16; shift; shift /= 2) sum += __shfl_down_sync(0xffffffff, sum, shift);
        if (lane == 0) partial[row][warp] = sum;
    }
    __syncthreads();
    if (t < Rows) {
        float sum = 0;
        for (int w = 0; w < 8; ++w) sum += partial[t][w];
        inverse[t] = rsqrtf(sum / inner + eps);
    }
    __syncthreads();
    for (int i = t; i < Rows * inner; i += 256)
        normalized[i] = __float2bfloat16_rn(RoundBF16(float(normalized[i]) * inverse[i / inner]) * gamma[i % inner]);
    __syncthreads();
    using Index = std::conditional_t<Dense, int, int64_t>;
    for (Index first = Index(blockIdx.x) * 16; first < channels; first += gridDim.x * 16) {
        float a[Rows][4] = {}, b[Rows][4] = {};
        const Index row0 = first + warp, row1 = row0 + 8;
        #pragma unroll 4
        for (int k = lane * 8; k < inner; k += 256) {
            uint4 wa = {}, wb = {};
            if (row0 < channels) wa = *reinterpret_cast<const uint4 *>(weight + (Dense ? size_t(row0) * inner : int64_t(row0) * weightStride) + k);
            if (row1 < channels) wb = *reinterpret_cast<const uint4 *>(weight + (Dense ? size_t(row1) * inner : int64_t(row1) * weightStride) + k);
            const BF16 *av = reinterpret_cast<const BF16 *>(&wa), *bv = reinterpret_cast<const BF16 *>(&wb);
            #pragma unroll
            for (int row = 0; row < Rows; ++row) {
                uint4 xx = *reinterpret_cast<const uint4 *>(normalized + size_t(row) * inner + k);
                const BF16 *v = reinterpret_cast<const BF16 *>(&xx);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    a[row][j % 4] = fmaf(float(v[j]), float(av[j]), a[row][j % 4]);
                    b[row][j % 4] = fmaf(float(v[j]), float(bv[j]), b[row][j % 4]);
                }
            }
        }
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float x = (a[row][0] + a[row][1]) + (a[row][2] + a[row][3]);
            float y = (b[row][0] + b[row][1]) + (b[row][2] + b[row][3]);
            for (int shift = 16; shift; shift /= 2) {
                x += __shfl_down_sync(0xffffffff, x, shift);
                y += __shfl_down_sync(0xffffffff, y, shift);
            }
            if (lane == 0 && (Dense || rowBase + row < rows)) {
                if (row0 < channels) {
                    output[(Dense ? size_t(row) * channels : int64_t(rowBase + row) * outputStride) + row0] = RoundBF16(x);
                    if constexpr (Mirror) mirror[int64_t(Dense ? row : rowBase + row) * mirrorStride + row0] = RoundBF16(x);
                }
                if (row1 < channels) {
                    output[(Dense ? size_t(row) * channels : int64_t(rowBase + row) * outputStride) + row1] = RoundBF16(y);
                    if constexpr (Mirror) mirror[int64_t(Dense ? row : rowBase + row) * mirrorStride + row1] = RoundBF16(y);
                }
            }
        }
    }
    // Every writer publishes peer stores before this stream records completion.
    // The consumer still waits for all rank events before sampling.
    if constexpr (Mirror) if (lane == 0) __threadfence_system();
}

} // namespace fastllm::cuda::low_latency::detail
