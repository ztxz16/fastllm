#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if !defined(USE_ROCM) && !defined(CUDA_NO_TENSOR_CORE)
#include <mma.h>
#endif

namespace fastllm {
namespace cuda {
namespace v41_compact {
// One warp per output row, four consecutive FP8 weights per lane.
// Conversion is exact before applying the checkpoint scale and BF16 rounding.
__global__ void Decode(const __nv_bfloat16 *input, const uint8_t *weight, const float *scales,
                       __nv_bfloat16 *output, int groups, int rank, int hidden) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32, group = blockIdx.y, token = blockIdx.z;
    if (row >= rank)
        return;
    const size_t weightBase = (size_t(group) * rank + row) * hidden;
    const size_t inputBase = (size_t(token) * groups + group) * hidden;
    float sum = 0.f;
    for (int k = lane * 4; k < hidden; k += 128) {
        __nv_fp8x4_e4m3 q;
        q.__x = *reinterpret_cast<const uint32_t *>(weight + weightBase + k);
        const ::float4 v = static_cast<::float4>(q);
        const float scale = scales[((group * rank + row) / 32) * (hidden / 32) + k / 32];
        const auto x = *reinterpret_cast<const uint2 *>(input + inputBase + k);
        const auto *a = reinterpret_cast<const __nv_bfloat16 *>(&x);
        sum = fmaf(float(a[0]), float(__float2bfloat16_rn(v.x * scale)), sum);
        sum = fmaf(float(a[1]), float(__float2bfloat16_rn(v.y * scale)), sum);
        sum = fmaf(float(a[2]), float(__float2bfloat16_rn(v.z * scale)), sum);
        sum = fmaf(float(a[3]), float(__float2bfloat16_rn(v.w * scale)), sum);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[(size_t(token) * groups + group) * rank + row] = __float2bfloat16_rn(sum);
}

// Verification rows share a BF16 Tensor Core tile. Four warps split K,
// then reduce their FP32 partial sums before the final BF16 output rounding.
__global__ void MultiRows(const __nv_bfloat16 *input, const uint8_t *weight, const float *scales,
                          __nv_bfloat16 *output, int tokens, int groups, int rank, int hidden) {
#if __CUDA_ARCH__ >= 800 && !defined(USE_ROCM) && !defined(CUDA_NO_TENSOR_CORE)
    using namespace nvcuda;
    constexpr int Warps = 4, Reduction = 64, Pitch = Reduction + 8;
    __shared__ __align__(32) __nv_bfloat16 a[Warps][16 * Pitch], b[Warps][16 * Pitch];
    __shared__ __align__(32) float sums[Warps][256];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, group = blockIdx.y, first = blockIdx.x * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int base = warp * Reduction; base < hidden; base += Warps * Reduction) {
        for (int j = lane; j < 16 * Reduction / 4; j += 32) {
            int i = j * 4, r = i / Reduction, k = base + i % Reduction;
            uint2 av = {};
            if (r < tokens && k < hidden)
                av = *reinterpret_cast<const uint2 *>(input + (size_t(r) * groups + group) * hidden + k);
            *reinterpret_cast<uint2 *>(a[warp] + r * Pitch + i % Reduction) = av;
            ::float4 v = {};
            if (first + r < rank && k < hidden) {
                __nv_fp8x4_e4m3 q;
                q.__x = *reinterpret_cast<const uint32_t *>(weight +
                                                            (size_t(group) * rank + first + r) * hidden + k);
                v = static_cast<::float4>(q);
                float scale = scales[((group * rank + first + r) / 32) * (hidden / 32) + k / 32];
                v.x *= scale;
                v.y *= scale;
                v.z *= scale;
                v.w *= scale;
            }
            alignas(8) __nv_bfloat16 bv[4] = {__float2bfloat16_rn(v.x), __float2bfloat16_rn(v.y),
                                   __float2bfloat16_rn(v.z), __float2bfloat16_rn(v.w)};
            *reinterpret_cast<uint2 *>(b[warp] + r * Pitch + i % Reduction) = *reinterpret_cast<uint2 *>(bv);
        }
        __syncwarp();
#pragma unroll
        for (int k = 0; k < Reduction; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> x;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> w;
            wmma::load_matrix_sync(x, a[warp] + k, Pitch);
            wmma::load_matrix_sync(w, b[warp] + k, Pitch);
            wmma::mma_sync(acc, x, w, acc);
        }
        __syncwarp();
    }
    wmma::store_matrix_sync(sums[warp], acc, 16, wmma::mem_row_major);
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += Warps * 32) {
        int t = i / 16, r = first + i % 16;
        if (t < tokens && r < rank) {
            float v = 0;
#pragma unroll
            for (int j = 0; j < Warps; j++)
                v += sums[j][i];
            output[(size_t(t) * groups + group) * rank + r] = __float2bfloat16_rn(v);
        }
    }
#endif
}

__global__ void Restore(const uint8_t *weight, const float *scales, __nv_bfloat16 *output, int rows,
                        int columns) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * columns)
        return;
    __nv_fp8_e4m3 code;
    code.__x = weight[i];
    output[i] =
        __float2bfloat16_rn(float(code) * scales[(i / columns / 32) * (columns / 32) + i % columns / 32]);
}
} // namespace v41_compact
} // namespace cuda
} // namespace fastllm
