#pragma once
// Small-row BF16 SWA. Each warp projects one scalar or one rotary pair,
// preserving the existing compensated GEMV and every BF16 rounding boundary.
// The following attention kernel consumes query and the new ring entries.
__device__ __forceinline__ void SwaQkvRopeRow(
        const BF16 *input, const BF16 *weight, BF16 *q,
        BF16 *key, BF16 *value, const void *const *cache,
        const float *position, const int *live,
        float theta, float valueScale, int verifyOffset) {
    if (cache) { key=(BF16*)cache[0]; value=(BF16*)cache[1]; }
    const int lane = threadIdx.x, job = blockIdx.x;
    const int length = *live, row = verifyOffset >= 0 ? 128 + verifyOffset : (length - 1) & 127;
    if (job >= 1568) return;
    if(length<=0) { if(lane==0 && job<1536)q[job]=__float2bfloat16_rn(0.f); return; }
    if(job<288){
        int head=job/32,d=job%32;
        BF16 a=fastllm::cuda::Bf16OrderedWarpDot<4096>(input,weight+(size_t(head)*192+d)*4096);
        BF16 b=fastllm::cuda::Bf16OrderedWarpDot<4096>(input,weight+(size_t(head)*192+d+32)*4096);
        if(lane==0){
            const float angle=*position*powf(theta,-2.f*d/64);
            const float c=RoundBF16(cosf(angle)),s=RoundBF16(sinf(angle));
            BF16 *dst=head<8?q+head*192:key+row*192;
            dst[d]=__float2bfloat16_rn(RoundBF16(float(a)*c)-RoundBF16(float(b)*s));
            dst[d+32]=__float2bfloat16_rn(RoundBF16(float(b)*c)+RoundBF16(float(a)*s));
        }
    }else if(job<1440){
        int head=(job-288)/128,d=(job-288)%128+64;
        BF16 a=fastllm::cuda::Bf16OrderedWarpDot<4096>(input,weight+(size_t(head)*192+d)*4096);
        if(lane==0)(head<8?q+head*192:key+row*192)[d]=a;
    }else{
        int d=job-1440;
        BF16 a=fastllm::cuda::Bf16OrderedWarpDot<4096>(input,weight+size_t(1728+d)*4096);
        if(lane==0)value[row*128+d]=__float2bfloat16_rn(float(a)*RoundBF16(valueScale));
    }
}

struct SwaQkvBinding {
    BF16 *key, *value;
    const void *const *cache;
    int verifyOffset;
};
struct SwaQkvBatch { SwaQkvBinding rows[16]; };
__global__ __launch_bounds__(32) void NaiveSwaQkvRope(
        const BF16 *input, const BF16 *weight, BF16 *q,
        BF16 *key, BF16 *value, const void *const *cache,
        const float *position, const int *live, float theta, float valueScale) {
    SwaQkvRopeRow(input, weight, q, key, value, cache, position, live, theta, valueScale, -1);
}
__global__ __launch_bounds__(32) void NaiveSwaQkvVerify(
        const BF16 *input, const BF16 *weight, BF16 *q, SwaQkvBatch batch,
        const float *position, const int *live, float theta, float valueScale) {
    const int row = blockIdx.y;
    const auto &r = batch.rows[row];
    SwaQkvRopeRow(input + row * 4096, weight, q + row * 1536, r.key, r.value, r.cache,
        position + row, live + row, theta, valueScale, r.verifyOffset);
}

namespace swa_qkv_rows {
constexpr int Threads = 512, InputBytes = 8 * 4096 * sizeof(BF16);
constexpr int StagingBytes = InputBytes + 16 * 4096 * sizeof(BF16);

// Pair the two rotary columns in one CTA. The remaining Q/K and V columns
// stay contiguous in the original weight allocation; no packed copy is needed.
using fastllm_swa_mma::QkvColumn;
using fastllm_swa_mma::InputOffset;
using fastllm_swa_mma::Step;

// Reuse each weight fragment across up to eight independent query rows.
// Projection accumulates in FP32 with an MMA reduction order, then retains
// the original BF16 projection, RoPE products, and value-scale boundaries.
template<bool StageWeights, int WorkThreads>
__device__ __forceinline__ void ProjectRope(
        const BF16 *input, const BF16 *weight, BF16 *q, const SwaQkvBatch &batch,
        const float *position, const int *live, int rows, float theta, float valueScale) {
    constexpr int Threads = WorkThreads;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int g = lane / 4, quad = lane % 4, tile = blockIdx.x, first = blockIdx.y * 8;
    extern __shared__ __align__(128) unsigned char inputTile[];
    unsigned char *weightTile = inputTile + InputBytes;
    __shared__ float partial[Threads / 32][16][8];
    __shared__ BF16 projected[16][8];
    for (int i = t; i < 8 * 512; i += Threads) {
        const int row = i / 512, col = i % 512 * 8;
        void *dst = inputTile + InputOffset(row, col);
        if (first + row < rows) {
            const BF16 *src = input + size_t(first + row) * 4096 + col;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            unsigned address = unsigned(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;"
                :: "r"(address), "l"(src) : "memory");
#else
            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
#endif
        } else {
            *reinterpret_cast<uint4 *>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
    // A full eight-row tile benefits from coalesced asynchronous weight loads
    // when the model's layer weights exceed L2. Multiple row tiles retain
    // direct loads so the extra shared memory does not reduce their throughput.
    if constexpr (StageWeights) for (int i = t; i < 16 * 512; i += Threads) {
        const int row = i / 512, col = i % 512 * 8;
        const int offset = col / 64 * 2048 + row * 128 + ((col % 64 / 8) ^ (row & 7)) * 16;
        void *dst = weightTile + offset;
        const BF16 *src = weight + size_t(QkvColumn(tile, row)) * 4096 + col;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        unsigned address = unsigned(__cvta_generic_to_shared(dst));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;"
            :: "r"(address), "l"(src) : "memory");
#else
        *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
    float c[4] = {};
    #pragma unroll 4
    for (int col = warp * 32; col < 4096; col += Threads) {
        const int k = col + quad * 8;
        uint4 a, b;
        if constexpr (StageWeights) {
            const int offset = k / 64 * 2048 + g * 128 + ((k % 64 / 8) ^ g) * 16;
            a = *reinterpret_cast<const uint4 *>(weightTile + offset);
            b = *reinterpret_cast<const uint4 *>(weightTile + offset + 8 * 128);
        } else {
            a = *reinterpret_cast<const uint4 *>(weight + size_t(QkvColumn(tile, g)) * 4096 + k);
            b = *reinterpret_cast<const uint4 *>(weight + size_t(QkvColumn(tile, g + 8)) * 4096 + k);
        }
        uint4 x = *reinterpret_cast<const uint4 *>(inputTile + InputOffset(g, k));
        Step(c, a, b, x);
    }
    partial[warp][g][quad * 2] = c[0];
    partial[warp][g][quad * 2 + 1] = c[1];
    partial[warp][g + 8][quad * 2] = c[2];
    partial[warp][g + 8][quad * 2 + 1] = c[3];
    __syncthreads();
    if (t < 128) {
        float sum = 0;
        #pragma unroll
        for (int w = 0; w < Threads / 32; ++w) sum += partial[w][t / 8][t % 8];
        projected[t / 8][t % 8] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    if (t >= 128) return;
    const int out = t / 8, r = t % 8, row = first + r;
    if (row >= rows) return;
    const int col = QkvColumn(tile, out);
    if (live[row] <= 0) {
        if (col < 1536) q[size_t(row) * 1536 + col] = __float2bfloat16_rn(0.f);
        return;
    }
    const auto &binding = batch.rows[row];
    BF16 *key = binding.cache ? (BF16 *)binding.cache[0] : binding.key;
    BF16 *value = binding.cache ? (BF16 *)binding.cache[1] : binding.value;
    const int target = 128 + binding.verifyOffset;
    if (tile < 36) {
        if (out >= 8) return;
        const int d = tile % 4 * 8 + out;
        const float angle = position[row] * powf(theta, -2.f * d / 64.f);
        const float cosine = RoundBF16(cosf(angle)), sine = RoundBF16(sinf(angle));
        const float a = float(projected[out][r]), b = float(projected[out + 8][r]);
        BF16 *dst = col < 1536 ? q + size_t(row) * 1536 + col : key + target * 192 + col - 1536;
        dst[0] = __float2bfloat16_rn(RoundBF16(a * cosine) - RoundBF16(b * sine));
        dst[32] = __float2bfloat16_rn(RoundBF16(b * cosine) + RoundBF16(a * sine));
    } else if (col < 1536) {
        q[size_t(row) * 1536 + col] = projected[out][r];
    } else if (col < 1728) {
        key[target * 192 + col - 1536] = projected[out][r];
    } else {
        value[target * 128 + col - 1728] = __float2bfloat16_rn(float(projected[out][r]) * RoundBF16(valueScale));
    }
}
template<bool StageWeights>
__global__ __launch_bounds__(Threads) void QkvRope(
        const BF16 *input, const BF16 *weight, BF16 *q,
        #if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
        const __grid_constant__
#endif
        SwaQkvBatch batch,
        const float *position, const int *live, int rows, float theta, float valueScale) {
    ProjectRope<StageWeights, Threads>(input, weight, q, batch, position, live, rows, theta, valueScale);
}
} // namespace swa_qkv_rows
