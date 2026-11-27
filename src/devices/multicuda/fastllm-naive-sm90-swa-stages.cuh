#pragma once

// Four-stage verification for up to eight BF16 ring rows, included inside hopper_verify.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define FASTLLM_SWA_GRID_ARG const __grid_constant__ Args
#else
#define FASTLLM_SWA_GRID_ARG const Args
#endif

// The preceding FFN already materialized the residual and input norm on SM90.
__global__ __launch_bounds__(Threads) void SwaPrepare(FASTLLM_SWA_GRID_ARG a, BF *feature) {
    const int i = blockIdx.x * Threads + threadIdx.x;
    const float frequency = i < a.rows * 32 ? powf(a.theta, -2.f * (i % 32) / 64.f) : 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // QKV can stage immutable weights while TP finishes. Scratch, positions
    // and the shared tagged-transport epochs all remain behind the wait.
    cudaTriggerProgrammaticLaunchCompletion();
    asm volatile("" :: "f"(frequency));
    cudaGridDependencySynchronize();
#endif
    if (feature) {
        for (int i = blockIdx.x * Threads + threadIdx.x; i < a.rows * 512; i += gridDim.x * Threads)
            reinterpret_cast<uint4 *>(feature)[i] = reinterpret_cast<const uint4 *>(a.hidden)[i];
    }
    BeginRingReduction(a);
    if (i < a.rows * 32) {
        const float angle = a.pos[i / 32] * frequency;
        reinterpret_cast<float2 *>(a.scores + 2)[i] =
            make_float2(CustomArNormRound(cosf(angle)), CustomArNormRound(sinf(angle)));
    }
    Pack<true>(a.input, 4096, a);
}

// Read immutable QKV weights while the preceding graph node packs activations.
__global__ __launch_bounds__(Threads) void SwaQkv(FASTLLM_SWA_GRID_ARG a) {
    extern __shared__ __align__(16) uint8_t stage[];
    __shared__ float part[16][128];
    StageQkv(a, stage);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
    Project<true, true, true, 1, true>(a, part, stage);
}

__device__ void RotateAppendSwa(const Args &a) {
    for (int i = blockIdx.x * Threads + threadIdx.x; i < a.rows * 1856; i += gridDim.x * Threads) {
        const int row = i / 1856, col = i % 1856;
        const auto desc = FastllmNaiveResolveDSARow(a.Row(row));
        const int live = *desc.liveKeys, dim = col % 192;
        float value = float(a.projection[i]);
        if (col < 1728 && dim < 64) {
            const float other = float(a.projection[row * 1856 + col / 192 * 192 + (dim ^ 32)]);
            const float2 rot = reinterpret_cast<const float2 *>(a.scores + 2)[row * 32 + (dim & 31)];
            value = CustomArNormRound(value * rot.x) + (dim < 32 ? -1.f : 1.f) * CustomArNormRound(other * rot.y);
        }
        if (col < 1536) {
            a.query[row * 1536 + col] = __float2bfloat16_rn(live > 0 ? value : 0.f);
        } else if (live > 0 && live <= desc.capacity) {
            const int slot = 128 + desc.verifyOffset;
            if (col < 1728)
                static_cast<BF *>(const_cast<void *>(desc.key))[slot * 192 + col - 1536] = __float2bfloat16_rn(value);
            else
                static_cast<BF *>(const_cast<void *>(desc.value))[slot * 128 + col - 1728] = __float2bfloat16_rn(value * a.valueScale);
        }
    }
}

__global__ __launch_bounds__(Threads, 2) void SwaPackedOutput(FASTLLM_SWA_GRID_ARG a) {
    __shared__ __align__(16) BF values[128 * RingValueTile];
    __shared__ float prob[Heads * 128], part[16][128];
    PrefetchRingOutput(a);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    RotateAppendSwa(a);
    cooperative_groups::this_grid().sync();
    RingAttention<false>(a, values, prob);
    cooperative_groups::this_grid().sync();
    ProjectAndReduceOutput<true, true>(a, part);
}

// Each CTA owns four experts and four rows. Keeping rounded norm values in
// shared memory preserves the router's reduction tree without a grid barrier.
__global__ __launch_bounds__(Threads) void SwaNormRouter(FASTLLM_SWA_GRID_ARG a) {
    constexpr int Rows = 4;
    const int t = threadIdx.x, first = blockIdx.x / 64 * Rows;
    __shared__ float squares[Rows][16], scratch[4 * Rows * 8];
    __shared__ __align__(16) BF tile[Rows][4096];
    const auto gamma = reinterpret_cast<const float4 *>(a.gamma);
    const float4 lo = gamma[t * 2], hi = gamma[t * 2 + 1];
    const float w[] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const float *expert = a.routerWeight + (blockIdx.x % 64) * 4 * 4096;
    for (int i = t; i < 4 * 4096 / 16; i += Threads)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(expert + i * 16));
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    CustomArNormPacketType v[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        v[r] = first + r < a.rows ? reinterpret_cast<const CustomArNormPacketType *>(a.hidden)[(first + r) * Threads + t]
                                : CustomArNormPacketType{};
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float x = float(v[r].data[i]);
            sum += x * x;
        }
        sum = CustomArNormWarpSum(sum);
        if (!(t % 32)) squares[r][t / 32] = sum;
    }
    __syncthreads();
    if (t < Rows * 32) {
        const int r = t / 32, lane = t % 32;
        const float total = CustomArNormWarpSum(lane < 16 ? squares[r][lane] : 0.f);
        if (lane == 0) squares[r][0] = rsqrtf(total / 4096 + a.eps);
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const float inv = squares[r][0];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            v[r].data[i] = __float2bfloat16_rn(CustomArNormRound(float(v[r].data[i]) * inv) * w[i]);
        reinterpret_cast<CustomArNormPacketType *>(tile[r])[t] = v[r];
        if (first + r < a.rows && blockIdx.x % 64 == 0)
            reinterpret_cast<CustomArNormPacketType *>(a.normed)[(first + r) * Threads + t] = v[r];
    }
    __syncthreads();
    RouteTiles<Rows, true>(a, scratch, tile[0]);
}
#undef FASTLLM_SWA_GRID_ARG
