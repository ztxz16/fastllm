#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace fastllm {
namespace cuda {
namespace v41_compact {
#ifdef USE_ROCM
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

#endif

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
