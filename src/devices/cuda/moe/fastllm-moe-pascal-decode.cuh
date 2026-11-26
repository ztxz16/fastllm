// NVFP4 expert projection for GP100. Keep the weights in their canonical block-16
// layout and reduce short half2 dot products in FP32.
#pragma once

namespace fastllm::pascal_decode {
namespace {
constexpr int projectionThreads = 128;

__device__ __forceinline__ half2 fp4Pair(unsigned high, bool upper) {
    const unsigned bits = __byte_perm(0, high, upper ? 0x7060 : 0x5040);
    return __halves2half2(__ushort_as_half(bits), __ushort_as_half(bits >> 16));
}

// Pack one FP32 activation block into the same 64 bytes. Normalization
// bounds every FP16 intermediate even for large activations.
__global__ void packHalf(const float *x, float *out, int K) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned mask = __ballot_sync(0xffffffff, k < K);
    if (k >= K) return;
    const int lane = k % 16;
    const size_t pos = size_t(blockIdx.y) * K + k;
    const float v = x[pos];
    float scale = fabsf(v);
    for (int d = 8; d; d /= 2)
        scale = fmaxf(scale, __shfl_xor_sync(mask, scale, d, 16));
    const unsigned h = __half_as_ushort(__float2half_rn(scale != 0.f ? v / scale : 0.f));
    const unsigned next = __shfl_down_sync(mask, h, 1, 16);
    // The down projection packs in place. Complete all source reads before
    // writing packed pairs to other lanes' original FP32 slots.
    __syncwarp(mask);
    float *block = out + pos - lane;
    if (!(lane & 1)) reinterpret_cast<unsigned *>(block)[lane / 2] = h | (next << 16);
    if (lane == 0) block[8] = scale;
}

template <int groupRows, int lanesK>
__global__ void projectTiled(const float *x, const int *ids, const int *plan, int experts, int tiles,
                             uint8_t *const *weights, int top, int K, int N, float *y) {
#if __CUDA_ARCH__ >= 530
    constexpr int channels = projectionThreads / lanesK, step = lanesK * 16;
    // Neighboring lanes split K for one output channel. Each block loads its
    // input panel once and reuses each unpacked weight across the expert tile.
    const int split = threadIdx.x % lanesK, n = blockIdx.x * channels + threadIdx.x / lanesK;
    int expert, count = 1, route[groupRows];
    if constexpr (groupRows == 1) {
        route[0] = blockIdx.y;
        expert = ids[blockIdx.y];
    } else {
        if (blockIdx.y >= plan[experts + 1]) return;
        const int *desc = plan + experts + 2, *order = desc + 2 * tiles;
        expert = desc[2 * blockIdx.y];
        const int first = desc[2 * blockIdx.y + 1];
        count = min(groupRows, plan[expert + 1] - first);
#pragma unroll
        for (int m = 0; m < groupRows; m++)
            route[m] = m < count ? order[first + m] : 0;
    }
    const auto *w = reinterpret_cast<const uint3 *>(weights[expert]);
    __shared__ half2 a[groupRows][step / 2];
    __shared__ float scales[groupRows][lanesK];
    float sums[groupRows] = {};
    for (int k = 0; k < K; k += step) {
        for (int v = threadIdx.x; v < groupRows * step / 2; v += projectionThreads) {
            const int m = v / (step / 2), j = v % (step / 2);
            if (m < count && k + 2 * j < K) {
                const float *s = x + size_t(route[m] / top) * K + k + j / 8 * 16;
                a[m][j] = reinterpret_cast<const half2 *>(s)[j % 8];
                if (j % 8 == 0) scales[m][j / 8] = s[8];
            }
        }
        const uint3 rec = n < N && k + split * 16 < K ? w[size_t(n) * (K / 16) + k / 16 + split] : uint3{};
        half2 hw[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned v = legacy_experts::fp4HighBytes((i < 2 ? rec.x : rec.y) >> ((i % 2) * 16));
            hw[2 * i] = fp4Pair(v, false);
            hw[2 * i + 1] = fp4Pair(v, true);
        }
        __syncthreads();
        if (k + split * 16 < K) {
#pragma unroll
            for (int m = 0; m < groupRows; m++) {
                if (m >= count) continue;
                const half2 *src = &a[m][split * 8];
                half2 d0 = __hmul2(src[0], hw[0]), d1 = __hmul2(src[1], hw[1]);
#pragma unroll
                for (int i = 2; i < 8; i += 2) {
                    d0 = __hfma2(src[i], hw[i], d0);
                    d1 = __hfma2(src[i + 1], hw[i + 1], d1);
                }
                const float2 f0 = __half22float2(d0), f1 = __half22float2(d1);
                sums[m] += ((f0.x + f0.y + f1.x + f1.y) * scales[m][split]) * __uint_as_float(rec.z);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < groupRows; m++) {
        if (m >= count) continue;
        for (int d = lanesK / 2; d; d /= 2)
            sums[m] += __shfl_down_sync(0xffffffff, sums[m], d, lanesK);
        if (split == 0 && n < N) y[size_t(route[m]) * N + n] = sums[m];
    }
#endif
}

template <int groupRows, int lanesK>
bool apply(const Data &input, Data &output, const Data &ids, const Data &scores, Data &pair, Data &middle,
           int experts, int inter, const CudaIndexedMoeSlots &slots, const CudaMoeActivation &activation) {
    const int rows = input.dims[0], h = input.dims[1], top = ids.dims[1], routes = rows * top;
    const int tiles = std::min(routes, experts + (routes - 1) / groupRows);
    if constexpr (groupRows > 1) {
        if (output.Count(0) < size_t(experts + 2 + 2 * tiles + routes)) return false;
    }
    auto *plan = static_cast<int *>(output.cudaData);
    auto *idx = static_cast<const int *>(ids.cudaData);
    auto *gate = static_cast<float *>(pair.cudaData), *scratch = static_cast<float *>(middle.cudaData);
    auto stream = cudaStreamPerThread;
    if constexpr (groupRows > 1) {
        legacy_experts::makeDecodePlan<groupRows><<<1, 256, experts * sizeof(int), stream>>>(idx, routes, experts, tiles, plan);
    }
    auto launch = [&](const float *src, uint8_t *const *weights, int div, int K, int N) {
        packHalf<<<dim3((K + 255) / 256, routes / div), 256, 0, stream>>>(src, scratch, K);
        constexpr int channels = projectionThreads / lanesK;
        const dim3 grid((N + channels - 1) / channels, groupRows > 1 ? tiles : routes);
        projectTiled<groupRows, lanesK><<<grid, projectionThreads, 0, stream>>>(
            scratch, idx, plan, experts, tiles, weights, div, K, N, gate);
    };
    launch(static_cast<const float *>(input.cudaData), slots.gate, top, h, 2 * inter);
    activation.Run(pair, scores, middle, routes, inter);
    launch(scratch, slots.down, 1, inter, h);
    legacy_experts::sumRoutes<<<rows, 256, 0, stream>>>(gate, static_cast<float *>(output.cudaData), top, h,
                                                        nullptr);
    legacy_experts::check(cudaGetLastError());
    return true;
}

bool run(const Data &input, Data &output, const Data &ids, const Data &scores, Data &pair, Data &middle,
         int experts, int inter, const CudaIndexedMoeSlots &slots, const CudaMoeActivation &activation) {
    if (!legacy_experts::hasDecodeKernel<projectTiled<1, 8>, 60>() ||
        !legacy_experts::validDecodeLayout(input, output, ids, scores, pair, middle, experts, inter, slots) ||
        middle.Count(0) < size_t(input.dims[0]) * input.dims[1])
        return false;
    const int routes = ids.dims[0] * ids.dims[1];
    // Use larger expert tiles when multiple routes can reuse the same weight.
    if (routes >= 8 * experts)
        return apply<8, 4>(input, output, ids, scores, pair, middle, experts, inter, slots, activation);
    if (routes >= experts)
        return apply<4, 8>(input, output, ids, scores, pair, middle, experts, inter, slots, activation);
    return apply<1, 8>(input, output, ids, scores, pair, middle, experts, inter, slots, activation);
}
} // namespace
} // namespace fastllm::pascal_decode
