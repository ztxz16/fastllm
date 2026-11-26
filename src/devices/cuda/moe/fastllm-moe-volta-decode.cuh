// Small routed NVFP4 projections on Volta. The route plan is rebuilt on the
// device at each replay; all intermediate storage belongs to the caller.
#pragma once

#include <mma.h>

namespace fastllm::volta_decode {
namespace {
constexpr int tileM = 16;
constexpr int tileN = 64;
constexpr int tileK = 64;

__device__ __forceinline__ void unpack4(unsigned q, float scale, half *dst) {
    const unsigned v = legacy_experts::fp4HighBytes(q);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const half h = __ushort_as_half(((v >> (8 * j)) & 255) << 8);
        dst[j] = __float2half_rn(__half2float(h) * scale);
    }
}

__global__ void projectGrouped(const float *x, const int *plan, int experts, int maxTiles, uint8_t *const *weights,
                               int top, int K, int N, float *y) {
#if __CUDA_ARCH__ >= 700
    namespace wm = nvcuda::wmma;
    const int tile = blockIdx.y;
    if (tile >= plan[experts + 1]) return;
    const int *tiles = plan + experts + 2, *order = tiles + maxTiles * 2;
    const int expert = tiles[2 * tile], begin = tiles[2 * tile + 1];
    const int rows = min(tileM, plan[expert + 1] - begin);
    constexpr int stride = tileK + 8;
    __shared__ __align__(32) half a[tileM][stride];
    __shared__ __align__(32) half b[tileN][stride];
    __shared__ __align__(32) float c[tileM][tileN];
    const int warp = threadIdx.x / 32;
    const int n0 = blockIdx.x * tileN;
    const auto *blocks = reinterpret_cast<const uint3 *>(weights[expert]);
    wm::fragment<wm::accumulator, 16, 16, 16, float> accum;
    wm::fill_fragment(accum, 0.f);
    for (int k = 0; k < K; k += tileK) {
        for (int idx = threadIdx.x; idx < tileM * tileK / 4; idx += blockDim.x) {
            const int r = idx / (tileK / 4), col = idx % (tileK / 4) * 4;
            float4 value{};
            if (r < rows && k + col < K)
                value = *reinterpret_cast<const float4 *>(x + size_t(order[begin + r] / top) * K + k + col);
            auto *dst = reinterpret_cast<half2 *>(&a[r][col]);
            dst[0] = __floats2half2_rn(value.x, value.y);
            dst[1] = __floats2half2_rn(value.z, value.w);
        }
        for (int idx = threadIdx.x; idx < tileN * tileK / 16; idx += blockDim.x) {
            const int n = idx / (tileK / 16), col = idx % (tileK / 16) * 16;
            uint3 block{};
            if (n0 + n < N && k + col < K) block = blocks[size_t(n0 + n) * (K / 16) + (k + col) / 16];
            const float scale = __uint_as_float(block.z);
            unpack4(block.x, scale, &b[n][col]);
            unpack4(block.x >> 16, scale, &b[n][col + 4]);
            unpack4(block.y, scale, &b[n][col + 8]);
            unpack4(block.y >> 16, scale, &b[n][col + 12]);
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < tileK; j += 16) {
            wm::fragment<wm::matrix_a, 16, 16, 16, half, wm::row_major> lhs;
            wm::fragment<wm::matrix_b, 16, 16, 16, half, wm::col_major> rhs;
            wm::load_matrix_sync(lhs, &a[0][j], stride);
            wm::load_matrix_sync(rhs, &b[warp * 16][j], stride);
            wm::mma_sync(accum, lhs, rhs, accum);
        }
        __syncthreads();
    }
    wm::store_matrix_sync(&c[0][warp * 16], accum, tileN, wm::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < rows * tileN; idx += blockDim.x) {
        const int r = idx / tileN, n = idx % tileN;
        if (n0 + n < N) y[size_t(order[begin + r]) * N + n0 + n] = c[r][n];
    }
#endif
}

// Two signed digits per activation keep the block quantization error below
// max(abs(block))/(2*127*128). NVFP4 values times two are exact signed bytes.
__global__ void packDigits(const float *src, float *scratch, int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned mask = __ballot_sync(0xffffffff, k < K);
    if (k >= K) return;
    const int lane = k % 16;
    const size_t pos = size_t(blockIdx.y) * K + k;
    const float value = src[pos];
    float largest = fabsf(value);
#pragma unroll
    for (int d = 8; d; d /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(mask, largest, d, 16));
    const float normalized = largest != 0.f ? (value / largest) * 127.f : 0.f;
    const int hi = __float2int_rn(normalized);
    const int lo = __float2int_rn((normalized - float(hi)) * 128.f);
    const unsigned digits = (hi & 255) | ((lo & 255) << 8);
    const unsigned next = __shfl_down_sync(mask, digits, 1, 16);
    const unsigned pair = digits | (next << 16);
    const unsigned other = __shfl_down_sync(mask, pair, 2, 16);
    // Down packs in place; finish reading the source block before aliasing stores.
    __syncwarp(mask);
    auto *buf = reinterpret_cast<unsigned *>(scratch + pos - lane);
    if (lane % 4 == 0) {
        buf[lane / 4] = __byte_perm(pair, other, 0x6420);
        buf[4 + lane / 4] = __byte_perm(pair, other, 0x7531);
        if (lane == 0) buf[8] = __float_as_uint(largest * (0.5f / 127.f));
    }
}

__device__ __forceinline__ int signedFp4Bytes(unsigned q) {
    const unsigned mag = __byte_perm(0x03020100, 0x0c080604, q & 0x7777);
    const unsigned neg = __byte_perm(0x0000ff00, 0, (q >> 3) & 0x1111);
    return __vsub4(mag ^ neg, neg);
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int dotBlock(int4 w, int4 x) {
    int sum = __dp4a(w.x, x.x, 0);
    sum = __dp4a(w.y, x.y, sum);
    sum = __dp4a(w.z, x.z, sum);
    return __dp4a(w.w, x.w, sum);
}
#endif

__global__ void projectSparse(const float *x, const int *ids, uint8_t *const *weights, int top, int K, int N,
                              float *y) {
#if __CUDA_ARCH__ >= 610
    const int r = blockIdx.y, n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (n >= N) return;
    const auto *w = reinterpret_cast<const uint3 *>(weights[ids[r]]) + size_t(n) * (K / 16);
    const float *a = x + size_t(r / top) * K;
    float sum = 0.f;
    for (int j = threadIdx.x % 32; j < K / 16; j += 32) {
        const uint3 blk = w[j];
        const int4 hi = *reinterpret_cast<const int4 *>(a + j * 16);
        const int4 lo = *reinterpret_cast<const int4 *>(a + j * 16 + 4);
        const int4 values = make_int4(signedFp4Bytes(blk.x), signedFp4Bytes(blk.x >> 16), signedFp4Bytes(blk.y),
                                      signedFp4Bytes(blk.y >> 16));
        const float dot = float(dotBlock(values, hi)) + float(dotBlock(values, lo)) * (1.f / 128.f);
        sum += dot * a[j * 16 + 8] * __uint_as_float(blk.z);
    }
    for (int width = 16; width; width /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, width);
    if ((threadIdx.x % 32) == 0) y[size_t(r) * N + n] = sum;
#endif
}

bool run(const Data &input, Data &output, const Data &ids, const Data &scores, Data &pair, Data &middle,
         int experts, int inter, const CudaIndexedMoeSlots &slots, const CudaMoeActivation &activation) {
    if (!legacy_experts::hasDecodeKernel<projectGrouped, 70>() ||
        !legacy_experts::validDecodeLayout(input, output, ids, scores, pair, middle, experts, inter, slots))
        return false;
    const int rows = input.dims[0], hidden = input.dims[1], top = ids.dims[1], routes = rows * top;
    const int tiles = std::min(routes, experts + (routes - 1) / tileM);
    // Around two routes per expert, a shared 16-row MMA tile starts to
    // amortize its unpack over enough inputs to beat independent sparse dots.
    const bool grouped = routes >= 2 * experts;
    if ((reinterpret_cast<uintptr_t>(middle.cudaData) % alignof(int4)) ||
        (grouped && reinterpret_cast<uintptr_t>(input.cudaData) % alignof(float4)) ||
        (grouped && output.Count(0) < size_t(experts + 2 + 2 * tiles + routes)) ||
        (!grouped && middle.Count(0) < size_t(rows) * hidden))
        return false;
    auto stream = cudaStreamPerThread;
    auto *plan = static_cast<int *>(output.cudaData);
    auto *gate = static_cast<float *>(pair.cudaData);
    const auto *indices = static_cast<const int *>(ids.cudaData);
    if (grouped)
        legacy_experts::makeDecodePlan<tileM>
            <<<1, 256, experts * sizeof(int), stream>>>(indices, routes, experts, tiles, plan);
    auto launch = [&](const float *src, uint8_t *const *weights, int divisor, int K, int N) {
        if (grouped) {
            projectGrouped<<<dim3((N + tileN - 1) / tileN, tiles), 128, 0, stream>>>(src, plan, experts, tiles,
                                                                                     weights, divisor, K, N, gate);
        } else {
            // Use the FP32 middle buffer before the activation and repack it
            // in place afterwards. Each 16-lane subgroup reads its whole
            // 64-byte block before any lane overwrites that same block.
            auto *digits = static_cast<float *>(middle.cudaData);
            packDigits<<<dim3((K + 255) / 256, routes / divisor), 256, 0, stream>>>(src, digits, K);
            constexpr int warps = 8;
            projectSparse<<<dim3((N + warps - 1) / warps, routes), 32 * warps, 0, stream>>>(
                digits, indices, weights, divisor, K, N, gate);
        }
    };
    launch(static_cast<const float *>(input.cudaData), slots.gate, top, hidden, 2 * inter);
    activation.Run(pair, scores, middle, routes, inter);
    launch(static_cast<const float *>(middle.cudaData), slots.down, 1, inter, hidden);
    legacy_experts::sumRoutes<<<rows, 256, 0, stream>>>(gate, static_cast<float *>(output.cudaData), top, hidden,
                                                        nullptr);
    legacy_experts::check(cudaGetLastError());
    return true;
}
} // namespace
} // namespace fastllm::volta_decode
