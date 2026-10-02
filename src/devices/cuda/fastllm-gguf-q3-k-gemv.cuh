#pragma once

// Single-token Q3_K MMVQ. Included after the GGUF vector-dot helpers.
// Large projections pack eight rows per block; smaller projections retain
// four warps per row. Both preserve the weight layout and summation order.
namespace fastllm_gguf_q3_k {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerRow = 4;
constexpr int kRowsPerBlock = 8;
constexpr int kInputBlocksPerWeightBlock = QK_K/QK8_1;
constexpr int kBlocksPerIteration = kWarpsPerRow*kWarpSize/QI3_K;

static __device__ __forceinline__ float Dot(
    const void *weights, const block_q8_1 *input, int block, int iqs) {
    const auto &q = static_cast<const block_q3_K *>(weights)[block];
    const int offset = iqs/8*4;
    const int scaleOffset = iqs/8*8 + (iqs%8)/4;
    // Four six-bit sub-scales use alternating bytes of scales[0..7]
    // and paired bit fields of scales[8..11]. Load each field once.
    const uint32_t low0 = get_int_b2(q.scales, 0);
    const uint32_t low1 = get_int_b2(q.scales, 1);
    const uint32_t high = get_int_b2(q.scales, 2);
    const uint32_t lo = __byte_perm(low0, low1, (scaleOffset&1) ? 0x7531 : 0x6420);
    const uint32_t hi = __byte_perm(high, high, (scaleOffset&1) ? 0x3131 : 0x2020);
    const uint32_t factors = ((lo >> (scaleOffset/8*4)) & 0x0f0f0f0f) |
        ((((hi >> (scaleOffset/2)) & 0x00000303) |
          ((hi >> (scaleOffset/2+2)) & 0x03030000)) << 4);
    const uint32_t vl = get_int_b2(q.qs, iqs);
    const uint32_t vh = uint32_t(~get_int_b2(q.hmask, iqs%8)) >> offset;
    float sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int scale = int((factors >> (8*i)) & 255) - 32;
        // Values lie in [-4,3]. Sign extension is an OR with 0xfc in
        // negative bytes, avoiding unnecessary saturating byte subtraction.
        const int values = ((vl >> (2*i)) & 0x03030303) |
                           (((vh >> i) & 0x01010101)*0xfc);
        sum += __low2float(input[offset+i].ds) *
            (ggml_cuda_dp4a(values, get_int_b4(input[offset+i].qs, iqs%8), 0)*scale);
    }
    return __half2float(q.d)*sum;
}

// One legacy warp lane's K slice, shared by both launch geometries.
static __device__ __forceinline__ float PartialDot(
    const block_q3_K *weight, const block_q8_1 *input, int columns, int tid) {
    float sum = 0;
    for (int block = tid/QI3_K; block < columns/QK_K; block += kBlocksPerIteration)
        sum += Dot(weight, input + block*kInputBlocksPerWeightBlock, block, tid%QI3_K);
    return sum;
}

template<typename Output>
__global__ void Gemv(const block_q3_K *weight, const block_q8_1 *input,
                     Output *output, int columns, int rows) {
    extern __shared__ uint32_t activation[];
    const int words = columns/QK8_1*int(sizeof(block_q8_1))/4;
    for (int i = threadIdx.x; i < words; i += blockDim.x)
        activation[i] = reinterpret_cast<const uint32_t *>(input)[i];
    __syncthreads();
    const int lane = threadIdx.x%kWarpSize;
    const int row = blockIdx.x*kRowsPerBlock+threadIdx.x/kWarpSize;
    if (row >= rows) return;
    const auto *x = reinterpret_cast<const block_q8_1 *>(activation);
    const auto *w = weight+size_t(row)*(columns/QK_K);
    float partial[kWarpsPerRow];
#pragma unroll
    for (int warp = 0; warp < kWarpsPerRow; ++warp)
        partial[warp] = PartialDot(w, x, columns, lane+kWarpSize*warp);
    // Match the original four-warp shared-memory reduction before reducing
    // lanes. Combining the K loop into one accumulator would change logits.
    float sum = partial[0];
#pragma unroll
    for (int warp = 1; warp < kWarpsPerRow; ++warp) sum += partial[warp];
    sum = warp_reduce_sum(sum);
    if (lane == 0) output[row] = static_cast<Output>(sum);
}

// Retain four physical warps per row when there are too few rows for
// eight-row blocks, or when a long K makes serial virtual warps expensive.
template<typename Output>
__global__ void GemvFourWarps(const block_q3_K *weight, const block_q8_1 *input,
                              Output *output, int columns) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const auto *row = weight + size_t(blockIdx.x)*(columns/QK_K);
    float sum = PartialDot(row, input, columns, lane+kWarpSize*warp);
    __shared__ float partial[kWarpsPerRow-1][kWarpSize];
    if (warp) partial[warp-1][lane] = sum;
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int i = 0; i < kWarpsPerRow-1; ++i) sum += partial[i][lane];
    sum = warp_reduce_sum(sum);
    if (lane == 0) output[blockIdx.x] = static_cast<Output>(sum);
}

static bool Supports(const void *input, int columns, int rows) {
    // Keep the established path for tiny projections or unaligned views.
    return input != nullptr && rows >= 128 && columns >= 256 && columns <= 18432 &&
           columns%QK_K == 0 && (reinterpret_cast<uintptr_t>(input)&15) == 0;
}

template<typename Output>
static void Launch(const void *weight, const block_q8_1 *input, Output *output,
                    int columns, int rows, cudaStream_t stream) {
    if (rows >= 1024 && columns <= 8192) {
        Gemv<<<(rows+kRowsPerBlock-1)/kRowsPerBlock, kRowsPerBlock*kWarpSize,
               size_t(columns/QK8_1)*sizeof(block_q8_1), stream>>>(
            static_cast<const block_q3_K *>(weight), input, output, columns, rows);
    } else {
        GemvFourWarps<<<rows, dim3(kWarpSize,kWarpsPerRow), 0, stream>>>(
            static_cast<const block_q3_K *>(weight), input, output, columns);
    }
}
} // namespace fastllm_gguf_q3_k
