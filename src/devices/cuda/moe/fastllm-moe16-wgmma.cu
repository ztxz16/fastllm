// Native SM90a grouped BF16 expert GEMMs. This translation unit is built
// separately so other architectures retain their existing CUDA kernels.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {
constexpr int Moe16WgmmaRows = 64, Moe16WgmmaColumns = 64, Moe16WgmmaStages = 3;
template<bool Gate, int K = 64> constexpr int Moe16WgmmaSharedBytes() {
    return Moe16WgmmaStages * (Moe16WgmmaRows + Moe16WgmmaColumns * (Gate ? 2 : 1)) *
        K * sizeof(__nv_bfloat16);
}
struct Moe16ExpertMaps { CUtensorMap input, weight; };
__device__ __forceinline__ unsigned Moe16SharedAddress(const void *p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void Moe16TmaLoad(void *dst, const CUtensorMap *map,
        int k, int row, unsigned long long *barrier) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::
        "r"(Moe16SharedAddress(dst)), "l"(map), "r"(k), "r"(row),
        "r"(Moe16SharedAddress(barrier)) : "memory");
#endif
}
__device__ __forceinline__ void Moe16TmaWait(unsigned long long *barrier, int phase) {
#if __CUDA_ARCH__ >= 900
    asm volatile("{ .reg .pred p; wait_loop: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra wait_loop; }" ::
        "r"(Moe16SharedAddress(barrier)), "r"(phase) : "memory");
#endif
}
// Instruction and shared-descriptor encoding follow NVIDIA PTX ISA:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions
// CUDA/PTX implementation; no CUTLASS or SGLang runtime dependency.
template<int K>
__device__ __forceinline__ unsigned long long Moe16WgmmaDescriptor(const void *p) {
    // K-major BF16, with a 16-byte leading unit and eight rows per stride.
    // The descriptor swizzle must match the TMA map's 64/128-byte layout.
    return ((K == 32 ? 2ull : 1ull) << 62) | ((unsigned long long)K << 32) | (1ull << 16) |
        ((Moe16SharedAddress(p) >> 4) & 0x3fff);
}
__device__ __forceinline__ void Moe16Wgmma64(float *c, unsigned long long a, unsigned long long b) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;" :
        "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]),
        "+f"(c[8]), "+f"(c[9]), "+f"(c[10]), "+f"(c[11]), "+f"(c[12]), "+f"(c[13]), "+f"(c[14]), "+f"(c[15]),
        "+f"(c[16]), "+f"(c[17]), "+f"(c[18]), "+f"(c[19]), "+f"(c[20]), "+f"(c[21]), "+f"(c[22]), "+f"(c[23]),
        "+f"(c[24]), "+f"(c[25]), "+f"(c[26]), "+f"(c[27]), "+f"(c[28]), "+f"(c[29]), "+f"(c[30]), "+f"(c[31]) : "l"(a), "l"(b));
#endif
}

__device__ __forceinline__ void Moe16StoreExpertPair(__nv_bfloat16 *p, float x, float y) {
    // Preserve both BF16 rounding boundaries; tensor views may be unaligned.
    auto v = __floats2bfloat162_rn(x, y);
    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0)
        *(__nv_bfloat162 *)p = v;
    else {
        p[0] = __low2bfloat16(v);
        p[1] = __high2bfloat16(v);
    }
}

template<bool Gate, int K = 64>
__global__ void Moe16WgmmaExpertGemm(const __grid_constant__ Moe16ExpertMaps maps,
        const float *scores, const int *workspace, __nv_bfloat16 *output,
        __nv_bfloat16 *packedMiddle, int rows, int hidden, int intermediate, int experts) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    constexpr int M = Moe16WgmmaRows, N = Moe16WgmmaColumns, Stages = Moe16WgmmaStages;
    constexpr int StageElements = (M + N * (Gate ? 2 : 1)) * K;
    const int *tiles = workspace + (size_t)experts * (rows + 1);
    int tile = blockIdx.y, column = blockIdx.x * N;
    if constexpr (!Gate && K == 32) {
        // Interleave nearby M tiles before advancing N to reuse expert weights
        // in L2. Use the actual tile count for the final partial group.
        int count = -tiles[0], linear = blockIdx.y * gridDim.x + blockIdx.x;
        if (linear >= count * int(gridDim.x)) return;
        int group = linear / (8 * int(gridDim.x));
        int groupRows = min(8, count - group * 8), within = linear % (8 * int(gridDim.x));
        tile = group * 8 + within % groupRows;
        column = within / groupRows * N;
    } else if (tile >= -tiles[0]) return;
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    int index = tiles[tile + 1], expert = index / (rows + 1), start = index % (rows + 1);
    int inputWidth = Gate ? hidden : intermediate, outputWidth = Gate ? intermediate : hidden;
    extern __shared__ __align__(1024) __nv_bfloat16 storage[];
    __shared__ unsigned long long barriers[Stages];
    __shared__ int tasks[M];
    if (tid < M) tasks[tid] = start + tid < workspace[expert * (rows + 1) + rows] ? workspace[index + tid] : -1;
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < Stages; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(Moe16SharedAddress(barriers + i)) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto load = [&](int step) {
        if (tid == 0 && step * K < inputWidth) {
            int stage = step % Stages;
            auto *ptr = storage + stage * StageElements;
            int weightRow = expert * outputWidth * (Gate ? 2 : 1) + column;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                "r"(Moe16SharedAddress(barriers + stage)), "r"(StageElements * 2) : "memory");
            Moe16TmaLoad(ptr, &maps.input, step * K, tile * M, barriers + stage);
            Moe16TmaLoad(ptr + M * K, &maps.weight, step * K, weightRow, barriers + stage);
            if (Gate) Moe16TmaLoad(ptr + (M + N) * K, &maps.weight, step * K, weightRow + outputWidth, barriers + stage);
        }
    };
    load(0); load(1);
    float acc[32] = {}, up[Gate ? 32 : 1] = {};
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    for (int step = 0; step * K < inputWidth; ++step) {
        int stage = step % Stages;
        Moe16TmaWait(barriers + stage, (step / Stages) & 1);
        auto *a = storage + stage * StageElements, *b = a + M * K;
#pragma unroll
        for (int j = 0; j < K; j += 16) {
            auto ad = Moe16WgmmaDescriptor<K>(a + j), bd = Moe16WgmmaDescriptor<K>(b + j);
            Moe16Wgmma64(acc, ad, bd);
            if (Gate) Moe16Wgmma64(up, ad, Moe16WgmmaDescriptor<K>(b + N * K + j));
        }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // Keep the newest group in flight. At step zero, stage two is unused;
        // later, wait_group 1 retires the oldest consumer before recycling its
        // stage. The final wait below protects the accumulator reads.
        if(step>0)asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        load(step + Stages - 1);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if constexpr (!Gate && K == 32) {
        // The MMA fragments scatter short column pairs. Reuse the retired
        // staging buffer to coalesce the down output into 16-byte stores.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
        int row = warp * 16 + lane / 4 + ((j % 4) / 2) * 8;
        int col = column + (j / 4) * 8 + (lane % 4) * 2, task = tasks[row];
        float values[2] = {acc[j], acc[j + 1]};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (Gate) {
                // Round the gate/up projections before the SiLU multiplication.
                float g = __bfloat162float(__float2bfloat16(values[h]));
                float u = __bfloat162float(__float2bfloat16(up[j + h]));
                values[h] = g / (1.f + expf(-g)) * u;
            } else if (task >= 0)
                values[h] *= scores[task];
        }
        if constexpr (Gate || K == 64) {
            if (task >= 0)
                Moe16StoreExpertPair(output + (size_t)task * outputWidth + col, values[0], values[1]);
        }
        if constexpr (Gate) {
            Moe16StoreExpertPair(packedMiddle + ((size_t)tile * M + row) * intermediate + col,
                                  task >= 0 ? values[0] : 0.f, task >= 0 ? values[1] : 0.f);
        } else if constexpr (K == 32) {
            int offset = row * N + col - column;
            Moe16StoreExpertPair(storage + (offset ^ ((row & 7) << 3)), values[0], values[1]);
        }
    }
    if constexpr (!Gate && K == 32) {
        __syncthreads();
        for (int offset = tid * 8; offset < M * N; offset += blockDim.x * 8) {
            int row = offset / N, task = tasks[row];
            if (task < 0) continue;
            auto *src = storage + (offset ^ ((row & 7) << 3));
            auto *dst = output + (size_t)task * outputWidth + column + offset % N;
            if ((reinterpret_cast<uintptr_t>(dst) & 15) == 0)
                *(uint4 *)dst = *(const uint4 *)src;
            else {
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = src[i];
            }
        }
    }
#endif
}
}

bool FastllmCudaMoe16WgmmaAvailable() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local int previous = -1;
    static thread_local bool available = false;
    if (device != previous) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) return false;
        available = false;
        // SM90a code is architecture-specific. Do not query its kernel on a
        // different architecture: that can leave a no-kernel-image error.
        if (major == 9 && minor == 0) {
            cudaFuncAttributes attributes;
            if (cudaFuncGetAttributes(&attributes, Moe16WgmmaExpertGemm<true>) != cudaSuccess ||
                cudaFuncSetAttribute(Moe16WgmmaExpertGemm<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, Moe16WgmmaSharedBytes<true>()) != cudaSuccess ||
                cudaFuncSetAttribute(Moe16WgmmaExpertGemm<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, Moe16WgmmaSharedBytes<false>()) != cudaSuccess) return false;
            available = attributes.binaryVersion == 90 && attributes.ptxVersion >= 90;
        }
        previous = device;
    }
    return available;
}

void FastllmCudaMoe16WgmmaExperts(const CUtensorMap *inputMap, const CUtensorMap *gateMap,
        const CUtensorMap *middleMap, const CUtensorMap *downMap, const float *scores,
        const int *workspace, void *middle, void *parts, void *packedMiddle,
        int rows, int hidden, int intermediate, int experts, int tiles, cudaStream_t stream) {
    Moe16ExpertMaps gate{*inputMap, *gateMap}, down{*middleMap, *downMap};
    Moe16WgmmaExpertGemm<true><<<dim3(intermediate / 64, tiles), 128, Moe16WgmmaSharedBytes<true>(), stream>>>(gate,
        scores, workspace, (__nv_bfloat16 *)middle, (__nv_bfloat16 *)packedMiddle,
        rows, hidden, intermediate, experts);
    // Keep the established K64 path for longer reductions. The tensor-map
    // selection in fastllm-moe16-sm90.cuh uses the same short-reduction bound.
    if (intermediate <= 256)
        Moe16WgmmaExpertGemm<false, 32><<<dim3(hidden / 64, tiles), 128, Moe16WgmmaSharedBytes<false, 32>(), stream>>>(down,
            scores, workspace, (__nv_bfloat16 *)parts, nullptr, rows, hidden, intermediate, experts);
    else
        Moe16WgmmaExpertGemm<false><<<dim3(hidden / 64, tiles), 128, Moe16WgmmaSharedBytes<false>(), stream>>>(down,
            scores, workspace, (__nv_bfloat16 *)parts, nullptr, rows, hidden, intermediate, experts);
}
