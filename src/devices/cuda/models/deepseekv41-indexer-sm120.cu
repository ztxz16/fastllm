// Native FP4 indexer prefill. The caller supplies FP4-round-tripped BF16 Q
// and the existing [64 payload bytes, 4 UE8M0 scales] key rows.
// This unit is isolated from generic CUDA targets: no external kernel runtime.
#include "deepseekv41-indexer.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

namespace fastllm_v41_indexer {
namespace {

// Re-encode already quantized Q. No further quantization is allowed at the
// call site: scaling an E2M1 value by a power of two preserves its value.
__global__ void PackQuery(const __nv_bfloat16 *q, uint8_t *packed, int groups) {
    const int lane = threadIdx.x & 31;
    const int group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (group >= groups) return;
    const float value = __bfloat162float(q[size_t(group) * 32 + lane]);
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    const float lower = fmaxf(maximum, ldexpf(6.f, -126)) / 6.f;
    const unsigned bits = __float_as_uint(lower);
    const unsigned scaleBits = (bits & 0x7f800000u) + ((bits & 0x7fffffu) ? 0x800000u : 0u);
    const float scaled = fabsf(value / __uint_as_float(scaleBits));
    unsigned code = scaled >= .5f;
    code += scaled >= 1.f; code += scaled >= 1.5f; code += scaled >= 2.f;
    code += scaled >= 3.f; code += scaled >= 4.f; code += scaled >= 6.f;
    code |= signbit(value) ? 8 : 0;
    const unsigned next = __shfl_down_sync(0xffffffffu, code, 1);
    uint8_t *row = packed + size_t(group / 4) * 68;
    if (!(lane & 1)) row[(group % 4) * 16 + lane / 2] = code | (next << 4);
    if (!lane) row[64 + group % 4] = scaleBits >> 23;
}

__device__ __forceinline__ void Matrix4(unsigned *r, const void *ptr) {
    const unsigned address = __cvta_generic_to_shared(ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
}
__device__ __forceinline__ void Matrix2(unsigned *r, const void *ptr) {
    const unsigned address = __cvta_generic_to_shared(ptr);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(address));
}
__device__ __forceinline__ void Dot64(float *d, const unsigned *a, const unsigned *b,
                                      unsigned sa, unsigned sb) {
    const unsigned short zero = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "h"(zero), "h"(zero), "r"(sb), "h"(zero), "h"(zero));
}

constexpr int kQueriesPerBlock = 4;
__global__ __launch_bounds__(256) void ScoreFp4(const uint8_t *q, const float *weight,
        const uint8_t *keys, float *output, int tokens, int count, int ratio, int position) {
    // Padding gives each 16-byte matrix load aligned, independent shared rows.
    __shared__ __align__(16) uint8_t keyTile[128][80];
    __shared__ __align__(16) uint8_t queryTile[kQueriesPerBlock * 32][80];
    __shared__ float headWeight[kQueriesPerBlock * 32];
    const int firstKey = blockIdx.x * 128, firstQuery = blockIdx.y * kQueriesPerBlock;
    const int batch = blockIdx.z, tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int lastQuery = min(tokens, firstQuery + kQueriesPerBlock) - 1;
    const int maxVisible = min(count, (position + lastQuery + 1) / ratio);
    if (firstKey >= maxVisible) {
        for (int i = tid; i < kQueriesPerBlock * 128; i += 256) {
            const int qr = firstQuery + i / 128, kr = firstKey + i % 128;
            if (qr < tokens && kr < count) output[(size_t(batch) * tokens + qr) * count + kr] = -INFINITY;
        }
        return;
    }
    for (int i = tid; i < 128 * 17; i += 256) {
        const int row = i / 17, word = i % 17;
        reinterpret_cast<unsigned *>(keyTile[row])[word] = firstKey + row < count
            ? reinterpret_cast<const unsigned *>(keys + (size_t(batch) * count + firstKey + row) * 68)[word] : 0;
    }
    for (int i = tid; i < kQueriesPerBlock * 32 * 17; i += 256) {
        const int row = i / 17, word = i % 17;
        reinterpret_cast<unsigned *>(queryTile[row])[word] = firstQuery + row / 32 < tokens
            ? reinterpret_cast<const unsigned *>(q + ((size_t(batch) * tokens + firstQuery) * 32 + row) * 68)[word] : 0;
    }
    for (int i = tid; i < kQueriesPerBlock * 32; i += 256)
        headWeight[i] = firstQuery + i / 32 < tokens ? weight[(size_t(batch) * tokens + firstQuery) * 32 + i] : 0;
    __syncthreads();
    const int keyRow = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    unsigned a[2][4];
#pragma unroll
    for (int k = 0; k < 2; ++k) Matrix4(a[k], keyTile[keyRow] + k * 32 + (lane >> 4) * 16);
    const unsigned scalesA = *reinterpret_cast<unsigned *>(keyTile[warp * 16 + lane / 4 + (lane & 1) * 8] + 64);
#pragma unroll
    for (int qi = 0; qi < kQueriesPerBlock; ++qi) {
        float sum0 = 0, sum1 = 0;
#pragma unroll
        for (int heads = 0; heads < 4; ++heads) {
            float dot[4] = {};
            const int row = qi * 32 + heads * 8;
            const unsigned scalesB = *reinterpret_cast<unsigned *>(queryTile[row + lane / 4] + 64);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                unsigned b[2];
                Matrix2(b, queryTile[row + (lane & 7)] + k * 32 + ((lane >> 3) & 1) * 16);
                Dot64(dot, a[k], b, (scalesA >> (16 * k)) & 65535, (scalesB >> (16 * k)) & 65535);
            }
            const float w0 = headWeight[row + (lane & 3) * 2];
            const float w1 = headWeight[row + (lane & 3) * 2 + 1];
            sum0 += fmaxf(dot[0], 0.f) * w0 + fmaxf(dot[1], 0.f) * w1;
            sum1 += fmaxf(dot[2], 0.f) * w0 + fmaxf(dot[3], 0.f) * w1;
        }
        for (int offset = 1; offset <= 2; offset *= 2) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
        }
        if (!(lane & 3) && firstQuery + qi < tokens) {
            const int kr = firstKey + warp * 16 + lane / 4;
            const int visible = min(count, (position + firstQuery + qi + 1) / ratio);
            float *dst = output + (size_t(batch) * tokens + firstQuery + qi) * count;
            if (kr < count) dst[kr] = kr < visible ? sum0 : -INFINITY;
            if (kr + 8 < count) dst[kr + 8] = kr + 8 < visible ? sum1 : -INFINITY;
        }
    }
}
} // namespace

// scratch is caller-owned and remains live until its CUDA stream completes.
bool Fp4Prefill(const void *q, const float *weights, const uint8_t *keys,
        float *output, void *scratch, int batch, int tokens, int count, int ratio, int position) {
    const int groups = batch * tokens * 32 * 4;
    PackQuery<<<(groups + 7) / 8, 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(q), static_cast<uint8_t *>(scratch), groups);
    ScoreFp4<<<dim3((count - 1) / 128 + 1, (tokens - 1) / kQueriesPerBlock + 1, batch), 256, 0, cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(scratch), weights, keys, output, tokens, count, ratio, position);
    return cudaGetLastError() == cudaSuccess;
}
} // namespace fastllm_v41_indexer
