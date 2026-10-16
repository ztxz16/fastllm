#pragma once
// GLM hybrid Q2_K prefill retains the original Q8_K quantizer and weight
// values. Fold the integer subblock scale into each 2-bit code (0..45),
// accumulate exactly in INT32, then apply the two FP32 block scales.
// FP32 accumulation order differs from the generic per-subblock path.
// The three redundant Q8 scale slots hold eight signed 12-bit sums; all
// are exact because 16 signed Q8 values sum within [-2032, 2032].
namespace hybrid_q2_mmq {
constexpr int kOutputTile = 128, kWarps = 16;
static_assert(sizeof(block_q8_1_mmq) == MMQ_TILE_Y_K * sizeof(int), "Q8 MMQ row layout");
__device__ void Load(const char *x, int *tile, const int &block, const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q2_K;
    for (int row0 = 0; row0 < kOutputTile; row0 += kWarps) {
        int row = row0 + threadIdx.y, src = min(row, imax);
        const auto &w = reinterpret_cast<const block_q2_K *>(x + src * stride)[block];
        for (int word = threadIdx.x; word < 64; word += 32) {
            unsigned q = (unsigned(get_int_b2(w.qs, (word / 32) * 8 + word % 8)) >> (2 * ((word % 32) / 8))) &
                         0x03030303;
            tile[row * Pitch + word] = q * (w.scales[word / 4] & 15);
        }
        if (threadIdx.x < 16)
            tile[row * Pitch + 64 + threadIdx.x] = (w.scales[threadIdx.x] >> 4) * 0x01010101;
        if (threadIdx.x == 0)
            *reinterpret_cast<float2 *>(tile + row * Pitch + 80) = __half22float2(w.dm);
    }
#endif
}
__global__ void PackSums(block_q8_1_mmq *input, int rows, int capacity) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row >= rows)
        return;
    auto &q = input[size_t(blockIdx.y) * capacity + row];
    int v = __dp4a(reinterpret_cast<const int *>(q.qs)[lane], 0x01010101, 0);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    unsigned packed = 0;
#pragma unroll
    for (int g = 0; g < 8; g++) {
        unsigned z = unsigned(__shfl_sync(0xffffffff, v, g * 4)) & 4095;
        int shift = g * 12 - lane * 32;
        if (shift >= 0 && shift < 32)
            packed |= z << shift;
        else if (shift < 0 && shift > -12)
            packed |= z >> (-shift);
    }
    if (lane < 3)
        reinterpret_cast<unsigned *>(q.d4)[lane + 1] = packed;
}
__device__ __forceinline__ int GetSum(const int *y, int group) {
    int bit = group * 12, word = 1 + bit / 32;
    unsigned a = y[word], b = word < 3 ? unsigned(y[word + 1]) : 0;
    return int(__funnelshift_r(a, b, bit % 32) << 20) >> 20;
}
template <int Tile> __device__ __forceinline__ void Dot(const int *x, const int *y, float *sum, int k00) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q2_K;
    int row = threadIdx.y * 8 + threadIdx.x / 4;
    float2 dm = *reinterpret_cast<const float2 *>(x + row * Pitch + 80);
#pragma unroll
    for (int j0 = 0; j0 < Tile; j0 += 8) {
        int cq0 = 0, cq1 = 0, cm0 = 0, cm1 = 0;
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
            int a = x[row * Pitch + k00 + k + threadIdx.x % 4], m = x[row * Pitch + 64 + (k00 + k) / 4] & 255;
            int b = y[(j0 + threadIdx.x / 4) * MMQ_TILE_Y_K + 4 + k + threadIdx.x % 4];
            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};"
                : "+r"(cq0), "+r"(cq1)
                : "r"(a), "r"(b));
            cm0 += m * GetSum(y + (j0 + 2 * (threadIdx.x % 4)) * MMQ_TILE_Y_K, k / 4);
            cm1 += m * GetSum(y + (j0 + 2 * (threadIdx.x % 4) + 1) * MMQ_TILE_Y_K, k / 4);
        }
        float db0 = reinterpret_cast<const float *>(y)[(j0 + 2 * (threadIdx.x % 4)) * MMQ_TILE_Y_K];
        float db1 = reinterpret_cast<const float *>(y)[(j0 + 2 * (threadIdx.x % 4) + 1) * MMQ_TILE_Y_K];
        sum[j0 / 4] += (cq0 * dm.x - cm0 * dm.y) * db0;
        sum[j0 / 4 + 1] += (cq1 * dm.x - cm1 * dm.y) * db1;
    }
#endif
}
template <int Tile>
__device__ void Compute(const uint8_t *weight, const block_q8_1_mmq *input, float *output, int columns,
                        int width, int stride, int count, int capacity, int localTile) {
#ifdef INT8_MMA_AVAILABLE
    extern __shared__ int shared[];
    int *y = shared, *x = y + Tile * MMQ_TILE_Y_K;
    float sum[Tile / 4] = {};
    int rowBase = blockIdx.x * kOutputTile, queryBase = localTile * Tile;
    for (int kb = 0; kb < columns / 256; ++kb) {
        Load(reinterpret_cast<const char *>(weight) + size_t(rowBase) * stride, x, kb, width - rowBase - 1,
             stride);
        for (int half = 0; half < 2; ++half) {
            auto *src = reinterpret_cast<const int *>(input + size_t(kb * 2 + half) * capacity + queryBase);
            for (int l = threadIdx.y * 32 + threadIdx.x; l < Tile * MMQ_TILE_Y_K; l += kWarps * WARP_SIZE)
                y[l] = src[min(l / MMQ_TILE_Y_K, count - queryBase - 1) * MMQ_TILE_Y_K + l % MMQ_TILE_Y_K];
            __syncthreads();
            Dot<Tile>(x, y, sum, half * 32);
            __syncthreads();
        }
    }
    int row = rowBase + threadIdx.y * 8 + threadIdx.x / 4;
#pragma unroll
    for (int j0 = 0; j0 < Tile; j0 += 8) {
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            int q = queryBase + j0 + 2 * (threadIdx.x % 4) + l;
            if (row < width && q < count)
                output[size_t(q) * width + row] = sum[j0 / 4 + l];
        }
    }
#endif
}
} // namespace hybrid_q2_mmq
