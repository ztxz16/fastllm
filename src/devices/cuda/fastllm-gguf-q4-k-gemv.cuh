#pragma once
#include <map>
// Include after fastllm-gguf-mmvq-common.cuh.
namespace fastllm_gguf_q4_k {
// Decode the packed six-bit scales with word loads and bit extraction.
// FP32 products and accumulation order remain identical to legacy Q4_K.
static __device__ __forceinline__ float Dot(const void *weight, const block_q8_1 *input, int block, int iqs) {
    const auto &w = static_cast<const block_q4_K *>(weight)[block];
    const int group = iqs / 8, offset = 2 * group, q = iqs / 2 % 4;
    const auto *scales = reinterpret_cast<const uint32_t *>(w.scales);
    const unsigned low = scales[0] >> (16 * (group & 1));
    const unsigned mins = scales[1] >> (16 * (group & 1));
    const unsigned high = scales[2] >> (16 * (group & 1));
    const unsigned sc = group < 2 ? low & 0x3f3f : (high & 0x0f0f) | ((low >> 2) & 0x3030);
    const unsigned mn = group < 2 ? mins & 0x3f3f : ((high >> 4) & 0x0f0f) | ((mins >> 2) & 0x3030);
    const auto *quants = reinterpret_cast<const int *>(w.qs + 16 * offset) + q;
    const int v0 = quants[0], v1 = quants[4];
    float sd = 0.f, sm = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const auto *values = reinterpret_cast<const int *>(input[offset + i].qs) + q;
        const int u0 = values[0], u1 = values[4];
        const int d =
            ggml_cuda_dp4a((v1 >> (4 * i)) & 0x0f0f0f0f, u1, ggml_cuda_dp4a((v0 >> (4 * i)) & 0x0f0f0f0f, u0, 0));
        const int m = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, 0));
        const float scale = __low2float(input[offset + i].ds);
        sd += scale * (d * int((sc >> (8 * i)) & 255));
        sm += scale * (m * int((mn >> (8 * i)) & 255));
    }
    const float2 dm = __half22float2(w.dm);
    return dm.x * sd - dm.y * sm;
}
template <typename Output, int RowsPerBlock, int WarpsPerRow>
static __global__ void PackedGemv(const void *__restrict__ weight, const block_q8_1 *__restrict__ input,
                                  Output *__restrict__ output, int columns, int rows) {
    const int localRow = threadIdx.x / (32 * WarpsPerRow);
    const int row = blockIdx.x * RowsPerBlock + localRow;
    const int warp = threadIdx.x / 32 % WarpsPerRow, lane = threadIdx.x % 32;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = get_vdr_mmvq(GGML_TYPE_Q4_K);
    constexpr int step = vdr * 128 / qi;
    const int blocks = columns / 256;
    __shared__ float partial[RowsPerBlock][4][32];
#pragma unroll
    for (int group = 0; group < 4 / WarpsPerRow; ++group) {
        const int oldWarp = warp + group * WarpsPerRow;
        const int tid = lane + 32 * oldWarp;
        float value = 0;
        if (row < rows)
            for (int b = tid / (qi / vdr); b < blocks; b += step)
                value += Dot(weight, input + 8 * b, row * blocks + b, vdr * (tid % (qi / vdr)));
        partial[localRow][oldWarp][lane] = value;
    }
    __syncthreads();
    if (warp != 0 || row >= rows)
        return;
    float value = partial[localRow][0][lane];
#pragma unroll
    for (int group = 1; group < 4; ++group)
        value += partial[localRow][group][lane];
    value = warp_reduce_sum(value);
    if (lane == 0)
        FastllmGgufStore<0>(output + row, value);
}
// A warp owns one output row, executing the four legacy warp fragments in
// their original order. Packing rows reduces CTA overhead without changing
// dot products, FP32 summation or the final output conversion.
template <typename Output, int RowsPerBlock = 4>
static __global__ void Gemv(const void *__restrict__ weight, const block_q8_1 *__restrict__ input,
                            Output *__restrict__ output, int columns, int rows) {
    const int row = blockIdx.x * RowsPerBlock + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows)
        return;
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr int vdr = get_vdr_mmvq(GGML_TYPE_Q4_K);
    constexpr int step = vdr * 128 / qi;
    const int blocks = columns / 256;
    float partial[4] = {};
#pragma unroll
    for (int warp = 0; warp < 4; ++warp) {
        const int tid = lane + 32 * warp;
        for (int b = tid / (qi / vdr); b < blocks; b += step)
            partial[warp] += Dot(weight, input + 8 * b, row * blocks + b, vdr * (tid % (qi / vdr)));
    }
    const float value = warp_reduce_sum(((partial[0] + partial[1]) + partial[2]) + partial[3]);
    if (lane == 0)
        FastllmGgufStore<0>(output + row, value);
}

static bool Supports(const void *weights, const void *input, int columns, int rows) {
    // Enable only the broad SM90 projections measured against the legacy
    // kernel. Narrow projections and unaligned views retain the old path.
    const bool shape =
        (columns == 1280 && rows >= 8192) || (columns == 2304 && rows >= 4096) || (columns == 5120 && rows >= 32768);
    if (!shape || !weights || !input || (reinterpret_cast<uintptr_t>(weights) & 3) ||
        (reinterpret_cast<uintptr_t>(input) & 3))
        return false;
    static thread_local std::map<int, bool> supported;
    const int device = FastllmCudaGetDevice();
    auto found = supported.find(device);
    if (found == supported.end()) {
        int major = 0, minor = 0;
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess;
        if (!ok)
            cudaGetLastError();
        found = supported.emplace(device, ok && major == 9 && minor == 0).first;
    }
    return found->second;
}

template <typename Output>
static void Launch(const void *weights, const block_q8_1 *input, Output *output, int columns, int rows,
                   cudaStream_t stream) {
    if (columns == 2304) {
        PackedGemv<Output, 2, 2><<<(rows + 1) / 2, 128, 0, stream>>>(weights, input, output, columns, rows);
    } else {
        Gemv<Output, 4><<<(rows + 3) / 4, 128, 0, stream>>>(weights, input, output, columns, rows);
    }
}
} // namespace fastllm_gguf_q4_k
