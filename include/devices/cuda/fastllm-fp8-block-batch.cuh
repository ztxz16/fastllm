#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fastllm { namespace fp8block {
// Each warp owns a disjoint K slice of sixteen output channels. A decoded
// weight fragment serves both eight-token MMA fragments before it is retired.
// Scale groups must align to the sixteen-element MMA reduction step.
template<int Tokens, int Warps>
__global__ void BlockFp8MmaBatch(const __nv_bfloat16 *__restrict__ input,
        const uint8_t *__restrict__ weights, const float *__restrict__ scales,
        const __nv_bfloat16 *__restrict__ bias, __nv_bfloat16 *__restrict__ output,
        int rows, int inner, int columns, int blockM, int blockK) {
#if __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    static_assert(Tokens == 8 || Tokens == 16);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int group = lane / 4, part = lane % 4;
    const int channel = blockIdx.x * 16 + group, first = blockIdx.y * Tokens;
    const int scaleColumns = (inner + blockM - 1) / blockM;
    float accum[Tokens / 8][4] = {};
    for (int base = warp * 64; base < inner; base += Warps * 64) {
        uint4 low{}, high{};
        if (channel < columns)
            low = *reinterpret_cast<const uint4 *>(weights + size_t(channel) * inner + base + part * 16);
        if (channel + 8 < columns)
            high = *reinterpret_cast<const uint4 *>(weights + size_t(channel + 8) * inner + base + part * 16);
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int source = group * 4 + step;
            const uint32_t lo0 = __shfl_sync(0xffffffff, low.x, source);
            const uint32_t lo1 = __shfl_sync(0xffffffff, low.y, source);
            const uint32_t lo2 = __shfl_sync(0xffffffff, low.z, source);
            const uint32_t lo3 = __shfl_sync(0xffffffff, low.w, source);
            const uint32_t hi0 = __shfl_sync(0xffffffff, high.x, source);
            const uint32_t hi1 = __shfl_sync(0xffffffff, high.y, source);
            const uint32_t hi2 = __shfl_sync(0xffffffff, high.z, source);
            const uint32_t hi3 = __shfl_sync(0xffffffff, high.w, source);
            const uint32_t packed[4] = {part < 2 ? lo0 : lo1, part < 2 ? hi0 : hi1,
                                        part < 2 ? lo2 : lo3, part < 2 ? hi2 : hi3};
            const int k = base + step * 16;
            const float s0 = channel < columns ? scales[size_t(channel / blockK) * scaleColumns + k / blockM] : 0.f;
            const float s1 = channel + 8 < columns ? scales[size_t((channel + 8) / blockK) * scaleColumns + k / blockM] : 0.f;
            uint32_t a[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_fp8x2_e4m3 code;
                code.__x = uint16_t(packed[j] >> ((part % 2) * 16));
                const float2 f = __half22float2(static_cast<__half2>(code));
                const float scale = j % 2 ? s1 : s0;
                const __nv_bfloat162 pair = __floats2bfloat162_rn(f.x * scale, f.y * scale);
                a[j] = *reinterpret_cast<const uint32_t *>(&pair);
            }
#pragma unroll
            for (int tile = 0; tile < Tokens / 8; ++tile) {
                uint32_t b0 = 0, b1 = 0;
                const int token = first + tile * 8 + group;
                if (token < rows) {
                    b0 = *reinterpret_cast<const uint32_t *>(input + size_t(token) * inner + k + part * 2);
                    b1 = *reinterpret_cast<const uint32_t *>(input + size_t(token) * inner + k + part * 2 + 8);
                }
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(accum[tile][0]), "+f"(accum[tile][1]), "+f"(accum[tile][2]), "+f"(accum[tile][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
            }
        }
    }
    __shared__ float partial[Warps][Tokens][16];
#pragma unroll
    for (int tile = 0; tile < Tokens / 8; ++tile) {
        partial[warp][tile * 8 + part * 2][group] = accum[tile][0];
        partial[warp][tile * 8 + part * 2 + 1][group] = accum[tile][1];
        partial[warp][tile * 8 + part * 2][group + 8] = accum[tile][2];
        partial[warp][tile * 8 + part * 2 + 1][group + 8] = accum[tile][3];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < Tokens * 16; i += Warps * 32) {
        const int token = i / 16, channel = blockIdx.x * 16 + i % 16;
        if (first + token >= rows || channel >= columns) continue;
        float sum = 0;
#pragma unroll
        for (int w = 0; w < Warps; ++w) sum += partial[w][token][i % 16];
        if (bias) sum += __bfloat162float(bias[channel]);
        output[size_t(first + token) * columns + channel] = __float2bfloat16_rn(sum);
    }
#endif
}
} }
