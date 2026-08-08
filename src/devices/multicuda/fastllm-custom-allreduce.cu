// Graph-safe small-tensor all-reduce for single-process MultiCUDA.
//
// The kernel/barrier design is adapted from vLLM's Apache-2.0 licensed
// csrc/custom_all_reduce.cuh. Decode-sized messages use its one-stage direct
// peer-read shape; larger TP>=4 messages use reduce-scatter/all-gather. Fixed
// graph workspaces let us register every pointer during warmup.

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kCustomArMaxRanks = 8;
constexpr int kCustomArMaxBlocks = 36;
constexpr int kCustomArThreads = 512;
constexpr int kCustomArPushMaxBlocks = 256;
constexpr size_t kCustomArPushMaxBytes = 1ULL * 1024ULL * 1024ULL;
constexpr size_t kCustomArAutoSmallBytes = 16ULL * 1024ULL;
constexpr size_t kCustomArAutoLargeBytes = 1ULL * 1024ULL * 1024ULL;
constexpr size_t kCustomArAutoTp2LargeBoundary = 256ULL * 1024ULL;
constexpr float kCustomArAutoRequiredRatio = 0.97f;
constexpr uint32_t kCustomArFp16SmallPath = 1U << 0;
constexpr uint32_t kCustomArFp16LargePath = 1U << 1;
constexpr uint32_t kCustomArBf16SmallPath = 1U << 2;
constexpr uint32_t kCustomArBf16LargePath = 1U << 3;
constexpr uint32_t kCustomArFloatSmallPath = 1U << 4;
constexpr uint32_t kCustomArFloatLargePath = 1U << 5;
constexpr uint32_t kCustomArAllPaths = kCustomArFp16SmallPath |
    kCustomArFp16LargePath | kCustomArBf16SmallPath |
    kCustomArBf16LargePath | kCustomArFloatSmallPath |
    kCustomArFloatLargePath;
using CustomArFlag = uint32_t;

enum class CustomArEnableMode {
    Disabled,
    Auto,
    Forced,
};

CustomArEnableMode CustomArModeFromEnv() {
    static const CustomArEnableMode mode = []() {
        const char *env = std::getenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE");
        if (env == nullptr || env[0] == '\0') {
            return CustomArEnableMode::Auto;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        if (value == "1" || value == "true" || value == "on" ||
            value == "yes") {
            return CustomArEnableMode::Forced;
        }
        if (value == "auto") {
            return CustomArEnableMode::Auto;
        }
        return CustomArEnableMode::Disabled;
    }();
    return mode;
}

bool CustomArPushEnabledFromEnv() {
    static const bool enabled = []() {
        const char *env = std::getenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return value != "0" && value != "false" && value != "off" &&
               value != "no";
    }();
    return enabled;
}

bool CustomArPushPdlEnabledFromEnv() {
    const char *env = std::getenv(
        "FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH_PDL");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value != "0" && value != "false" && value != "off" &&
           value != "no";
}

int CustomArPushBlocksFromEnv(int availableBlocks) {
    const char *env = std::getenv(
        "FASTLLM_CUDA_CUSTOM_ALLREDUCE_PUSH_BLOCKS");
    if (env == nullptr || env[0] == '\0') {
        return availableBlocks;
    }
    char *end = nullptr;
    long requested = std::strtol(env, &end, 10);
    if (end == env || *end != '\0' || requested <= 0) {
        return availableBlocks;
    }
    return std::min(availableBlocks, (int)std::min(
        (long)kCustomArPushMaxBlocks, requested));
}

bool CustomArFusedCopyBackEnabledByEnv() {
    static const bool enabled = []() {
        const char *env = std::getenv(
            "FASTLLM_CUDA_CUSTOM_ALLREDUCE_FUSED_COPYBACK");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return value != "0" && value != "false" && value != "off" &&
               value != "no";
    }();
    return enabled;
}

void LogCustomArDisabledOnce() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::fprintf(stderr,
                     "[Fastllm] graph-safe custom all-reduce is disabled; "
                     "unset FASTLLM_CUDA_CUSTOM_ALLREDUCE or set it to auto "
                     "to benchmark the current 2/4/6/8-GPU path, or set it "
                     "to 1 to force-enable.\n");
        std::fflush(stderr);
    });
}

struct CustomArSignal {
    alignas(128) CustomArFlag start[kCustomArMaxBlocks][kCustomArMaxRanks];
    alignas(128) CustomArFlag end[kCustomArMaxBlocks][kCustomArMaxRanks];
    alignas(128) CustomArFlag flag[kCustomArMaxBlocks];
    alignas(128) CustomArFlag pushCounter[kCustomArPushMaxBlocks];
};

struct __align__(16) CustomArRankData {
    const void *ptrs[kCustomArMaxRanks];
};

struct __align__(16) CustomArRankSignals {
    CustomArSignal *signals[kCustomArMaxRanks];
};

struct __align__(16) CustomArRankScratch {
    void *scratch[kCustomArMaxRanks];
};

struct __align__(16) CustomArPushWorkspaces {
    void *workspaces[kCustomArMaxRanks];
};

template <typename T, int N>
struct __align__(sizeof(T) * N) CustomArArray {
    T data[N];
    using Type = T;
};

template <typename T>
struct CustomArPacked {
    static constexpr int size = 16 / sizeof(T);
    using P = CustomArArray<T, 16 / sizeof(T)>;
    using A = CustomArArray<float, 16 / sizeof(T)>;
};

__device__ __forceinline__ float CustomArUpcast(float v) { return v; }
__device__ __forceinline__ float CustomArUpcast(half v) { return __half2float(v); }
__device__ __forceinline__ float CustomArUpcast(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T CustomArDowncast(float v);

template <>
__device__ __forceinline__ float CustomArDowncast<float>(float v) { return v; }

template <>
__device__ __forceinline__ half CustomArDowncast<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 CustomArDowncast<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ void CustomArStoreVolatile(CustomArFlag *ptr,
                                                       CustomArFlag value) {
    asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(value), "l"(ptr));
}

__device__ __forceinline__ CustomArFlag CustomArLoadVolatile(CustomArFlag *ptr) {
    CustomArFlag value;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

__device__ __forceinline__ void CustomArStoreRelease(CustomArFlag *ptr,
                                                      CustomArFlag value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(value), "l"(ptr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::
                 "r"(value), "l"(ptr));
#endif
}

__device__ __forceinline__ CustomArFlag CustomArLoadAcquire(CustomArFlag *ptr) {
    CustomArFlag value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" :
                 "=r"(value) : "l"(ptr));
#endif
    return value;
}

template <int Ranks>
__device__ __forceinline__ void CustomArBarrierStart(
        const CustomArRankSignals &allSignals, CustomArSignal *self,
        int rank) {
    CustomArFlag next = self->flag[blockIdx.x] + 1;
    if (threadIdx.x < Ranks) {
        CustomArStoreVolatile(
            &allSignals.signals[threadIdx.x]->start[blockIdx.x][rank], next);
        while (CustomArLoadVolatile(
                   &self->start[blockIdx.x][threadIdx.x]) != next) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = next;
    }
}

// The final barrier only prevents in-place overwrite while peers are still
// reading. Keep this byte-for-byte equivalent to the original fast volatile
// handshake used by decode-sized one-stage reductions.
template <int Ranks>
__device__ __forceinline__ void CustomArBarrierFinal(
        const CustomArRankSignals &allSignals, CustomArSignal *self,
        int rank) {
    __syncthreads();
    CustomArFlag next = self->flag[blockIdx.x] + 1;
    if (threadIdx.x < Ranks) {
        CustomArStoreVolatile(
            &allSignals.signals[threadIdx.x]->end[blockIdx.x][rank], next);
        while (CustomArLoadVolatile(
                   &self->end[blockIdx.x][threadIdx.x]) != next) {
        }
    }
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = next;
    }
}

// A publish barrier orders scratch writes before peer GPUs consume them in the
// all-gather stage, so it requires system-scope release/acquire operations and
// a trailing block barrier.
template <int Ranks>
__device__ __forceinline__ void CustomArBarrierPublish(
        const CustomArRankSignals &allSignals, CustomArSignal *self,
        int rank) {
    __syncthreads();
    CustomArFlag next = self->flag[blockIdx.x] + 1;
    if (threadIdx.x < Ranks) {
        CustomArStoreRelease(
            &allSignals.signals[threadIdx.x]->end[blockIdx.x][rank], next);
        while (CustomArLoadAcquire(
                   &self->end[blockIdx.x][threadIdx.x]) != next) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = next;
    }
}

template <typename T, int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceKernel(CustomArRankData *rankData,
                                  CustomArRankSignals allSignals,
                                  CustomArSignal *selfSignal,
                                  T *__restrict__ output,
                                  int rank, int packedCount,
                                  bool writeAfterBarrier) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData pointers = *rankData;
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    P deferredResult;
    int deferredIndex = -1;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < packedCount; index += gridDim.x * blockDim.x) {
        const P *first = reinterpret_cast<const P *>(pointers.ptrs[0]);
        P firstValue = first[index];
        A sum;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            sum.data[item] = CustomArUpcast(firstValue.data[item]);
        }
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            P value = reinterpret_cast<const P *>(pointers.ptrs[peer])[index];
#pragma unroll
            for (int item = 0; item < CustomArPacked<T>::size; ++item) {
                sum.data[item] += CustomArUpcast(value.data[item]);
            }
        }
        P result;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            result.data[item] = CustomArDowncast<T>(sum.data[item]);
        }
        if (writeAfterBarrier) {
            // The launch side enables this only when each thread owns at most
            // one packed vector, so the exact result can stay in registers.
            deferredResult = result;
            deferredIndex = index;
        } else {
            reinterpret_cast<P *>(output)[index] = result;
        }
    }
    CustomArBarrierFinal<Ranks>(allSignals, selfSignal, rank);
    if (writeAfterBarrier) {
        // All signal lanes must finish the cross-GPU final barrier before any
        // rank overwrites its input.  This replaces the old scratch write plus
        // 8 KiB graph memcpy node with one local store, while preserving the
        // rank-ordered FP32 accumulation byte for byte.
        __syncthreads();
        if (deferredIndex >= 0) {
            reinterpret_cast<P *>(output)[deferredIndex] = deferredResult;
        }
    }
}

// The one-stage kernel makes every rank read the complete tensor from every
// peer. That is ideal for decode-sized messages, but it is not the algorithm
// vLLM uses for larger TP>=4 messages. Reduce-scatter into per-rank scratch,
// publish it with a system-scope barrier, then all-gather into the destination.
// The publish barrier also makes in-place output safe: all peer reads from the
// original inputs have completed before any rank starts overwriting them.
template <typename T, int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceTwoStageKernel(
        CustomArRankData *rankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData pointers = *rankData;
    P *localScratch = reinterpret_cast<P *>(allScratch.scratch[rank]);
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int part = packedCount / Ranks;
    const int localStart = rank * part;
    const int localEnd = rank == Ranks - 1 ? packedCount : localStart + part;
    const int largestPart = part + packedCount % Ranks;

    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    for (int index = localStart + thread; index < localEnd; index += stride) {
        P firstValue = reinterpret_cast<const P *>(pointers.ptrs[0])[index];
        A sum;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            sum.data[item] = CustomArUpcast(firstValue.data[item]);
        }
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            P value = reinterpret_cast<const P *>(pointers.ptrs[peer])[index];
#pragma unroll
            for (int item = 0; item < CustomArPacked<T>::size; ++item) {
                sum.data[item] += CustomArUpcast(value.data[item]);
            }
        }
        P result;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            result.data[item] = CustomArDowncast<T>(sum.data[item]);
        }
        localScratch[index - localStart] = result;
    }

    CustomArBarrierPublish<Ranks>(allSignals, selfSignal, rank);
    P *packedOutput = reinterpret_cast<P *>(output);
    // Keep the same global thread mapping as reduce-scatter: the system-scope
    // publish barrier guarantees peer visibility for matching producer and
    // consumer threads.  The start barrier of the next collective also keeps
    // this shared scratch from being reused before every rank finishes gather.
    for (int offset = thread; offset < largestPart; offset += stride) {
#pragma unroll
        for (int source = 0; source < Ranks; ++source) {
            const int sourceCount = source == Ranks - 1
                ? packedCount - source * part : part;
            if (offset < sourceCount) {
                const P *sourceScratch =
                    reinterpret_cast<const P *>(allScratch.scratch[source]);
                packedOutput[source * part + offset] = sourceScratch[offset];
            }
        }
    }
}

template <typename T>
struct CustomArPairType {
    using Type = T;
};

template <>
struct CustomArPairType<half> {
    using Type = half2;
};

template <>
struct CustomArPairType<__nv_bfloat16> {
    using Type = __nv_bfloat162;
};

template <typename T>
__device__ __forceinline__ T CustomArPairAdd(T a, T b) {
    return a + b;
}

template <>
__device__ __forceinline__ half CustomArPairAdd(half a, half b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16 CustomArPairAdd(
        __nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ half2 CustomArPairAdd(half2 a, half2 b) {
    return __hadd2(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat162 CustomArPairAdd(
        __nv_bfloat162 a, __nv_bfloat162 b) {
    return __hadd2(a, b);
}

template <typename T>
__device__ __forceinline__ void CustomArAddPacket(
        typename CustomArPacked<T>::P &first,
        const typename CustomArPacked<T>::P &second) {
    using Pair = typename CustomArPairType<T>::Type;
    Pair *firstPairs = reinterpret_cast<Pair *>(&first);
    const Pair *secondPairs = reinterpret_cast<const Pair *>(&second);
#pragma unroll
    for (int item = 0; item < 16 / (int)sizeof(Pair); ++item) {
        firstPairs[item] = CustomArPairAdd(firstPairs[item],
                                           secondPairs[item]);
    }
}

// Reduce the routed and shared-expert partials independently, including the
// destination-type rounding boundary of each collective, and only then add
// them.  A rank-local pre-add would instead round before the cross-rank sum and
// changes DeepSeek-V4 target logits.
template <typename T, int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReducePairAddKernel(
        CustomArRankData *firstRankData,
        CustomArRankData *secondRankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData firstPointers = *firstRankData;
    CustomArRankData secondPointers = *secondRankData;
    P *localScratch = reinterpret_cast<P *>(allScratch.scratch[rank]);
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    for (int index = thread; index < packedCount; index += stride) {
        P firstValue =
            reinterpret_cast<const P *>(firstPointers.ptrs[0])[index];
        P secondValue =
            reinterpret_cast<const P *>(secondPointers.ptrs[0])[index];
        A firstSum;
        A secondSum;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            firstSum.data[item] = CustomArUpcast(firstValue.data[item]);
            secondSum.data[item] = CustomArUpcast(secondValue.data[item]);
        }
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            firstValue =
                reinterpret_cast<const P *>(firstPointers.ptrs[peer])[index];
            secondValue =
                reinterpret_cast<const P *>(secondPointers.ptrs[peer])[index];
#pragma unroll
            for (int item = 0; item < CustomArPacked<T>::size; ++item) {
                firstSum.data[item] += CustomArUpcast(firstValue.data[item]);
                secondSum.data[item] += CustomArUpcast(secondValue.data[item]);
            }
        }
        P result;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            const T firstRounded =
                CustomArDowncast<T>(firstSum.data[item]);
            const T secondRounded =
                CustomArDowncast<T>(secondSum.data[item]);
            result.data[item] = CustomArPairAdd(firstRounded, secondRounded);
        }
        localScratch[index] = result;
    }

    // All peers must finish reading both source tensors before an in-place
    // destination may overwrite the first source.
    CustomArBarrierFinal<Ranks>(allSignals, selfSignal, rank);
    __syncthreads();
    P *packedOutput = reinterpret_cast<P *>(output);
    for (int index = thread; index < packedCount; index += stride) {
        packedOutput[index] = localScratch[index];
    }
}

template <typename T, int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReducePairAddTwoStageKernel(
        CustomArRankData *firstRankData,
        CustomArRankData *secondRankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData firstPointers = *firstRankData;
    CustomArRankData secondPointers = *secondRankData;
    P *localScratch = reinterpret_cast<P *>(allScratch.scratch[rank]);
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int part = packedCount / Ranks;
    const int localStart = rank * part;
    const int localEnd = rank == Ranks - 1 ? packedCount : localStart + part;
    const int largestPart = part + packedCount % Ranks;

    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    for (int index = localStart + thread; index < localEnd; index += stride) {
        P firstValue =
            reinterpret_cast<const P *>(firstPointers.ptrs[0])[index];
        P secondValue =
            reinterpret_cast<const P *>(secondPointers.ptrs[0])[index];
        A firstSum;
        A secondSum;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            firstSum.data[item] = CustomArUpcast(firstValue.data[item]);
            secondSum.data[item] = CustomArUpcast(secondValue.data[item]);
        }
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            firstValue =
                reinterpret_cast<const P *>(firstPointers.ptrs[peer])[index];
            secondValue =
                reinterpret_cast<const P *>(secondPointers.ptrs[peer])[index];
#pragma unroll
            for (int item = 0; item < CustomArPacked<T>::size; ++item) {
                firstSum.data[item] += CustomArUpcast(firstValue.data[item]);
                secondSum.data[item] += CustomArUpcast(secondValue.data[item]);
            }
        }
        P result;
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            const T firstRounded =
                CustomArDowncast<T>(firstSum.data[item]);
            const T secondRounded =
                CustomArDowncast<T>(secondSum.data[item]);
            result.data[item] = CustomArPairAdd(firstRounded, secondRounded);
        }
        localScratch[index - localStart] = result;
    }

    CustomArBarrierPublish<Ranks>(allSignals, selfSignal, rank);
    P *packedOutput = reinterpret_cast<P *>(output);
    for (int offset = thread; offset < largestPart; offset += stride) {
#pragma unroll
        for (int source = 0; source < Ranks; ++source) {
            const int sourceCount = source == Ranks - 1
                ? packedCount - source * part : part;
            if (offset < sourceCount) {
                const P *sourceScratch =
                    reinterpret_cast<const P *>(allScratch.scratch[source]);
                packedOutput[source * part + offset] = sourceScratch[offset];
            }
        }
    }
}

// Qwen's tensor-parallel row projections reduce a rank-local partial and then
// add a replicated residual. Folding both operations into the direct-read
// kernel removes the rank-0 add/rank-1 copy, in-place scratch output and
// scratch copy-back that the generic graph path otherwise needs. The two
// half-precision additions intentionally preserve the eager TP=2 order:
// (rank0 + rank1) + residual.
template <typename T>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceAddKernel(CustomArRankData *rankData,
                                     CustomArRankSignals allSignals,
                                     CustomArSignal *selfSignal,
                                     const T *__restrict__ residual,
                                     T *__restrict__ output,
                                     int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    CustomArRankData pointers = *rankData;
    CustomArBarrierStart<2>(allSignals, selfSignal, rank);
    const P *rank0 = reinterpret_cast<const P *>(pointers.ptrs[0]);
    const P *rank1 = reinterpret_cast<const P *>(pointers.ptrs[1]);
    const P *packedResidual = reinterpret_cast<const P *>(residual);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < packedCount; index += gridDim.x * blockDim.x) {
        P first = rank0[index];
        P second = rank1[index];
        P residualValue = packedResidual[index];
        CustomArAddPacket<T>(first, second);
        CustomArAddPacket<T>(first, residualValue);
        reinterpret_cast<P *>(output)[index] = first;
    }
    CustomArBarrierFinal<2>(allSignals, selfSignal, rank);
}

template <typename T>
struct CustomArZeroBits;

template <>
struct CustomArZeroBits<half> {
    using Type = uint16_t;
    static constexpr Type positive = 0x0000U;
    static constexpr Type negative = 0x8000U;
};

template <>
struct CustomArZeroBits<__nv_bfloat16> {
    using Type = uint16_t;
    static constexpr Type positive = 0x0000U;
    static constexpr Type negative = 0x8000U;
};

template <>
struct CustomArZeroBits<float> {
    using Type = uint32_t;
    static constexpr Type positive = 0x00000000U;
    static constexpr Type negative = 0x80000000U;
};

template <typename T>
__device__ __forceinline__ void CustomArClearPositiveZero(T &value) {
    using Bits = typename CustomArZeroBits<T>::Type;
    Bits *bits = reinterpret_cast<Bits *>(&value);
    if (*bits == CustomArZeroBits<T>::positive) {
        *bits = CustomArZeroBits<T>::negative;
    }
}

template <typename T>
__device__ __forceinline__ bool CustomArIsPositiveZero(const T &value) {
    using Bits = typename CustomArZeroBits<T>::Type;
    return *reinterpret_cast<const Bits *>(&value) ==
           CustomArZeroBits<T>::positive;
}

template <typename P>
__device__ __forceinline__ void CustomArLoadGlobal16(
        P &value, const void *base, int index) {
    static_assert(sizeof(P) == 16 && alignof(P) == 16,
                  "custom all-reduce packets must be 16-byte aligned");
    const char *address = reinterpret_cast<const char *>(base) +
                          (size_t)index * sizeof(P);
    uint4 raw;
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                 : "l"(address));
    value = *reinterpret_cast<const P *>(&raw);
}

template <typename P>
__device__ __forceinline__ void CustomArLoadRelaxedSystem16(
        P &value, const void *base, int index) {
    static_assert(sizeof(P) == 16 && alignof(P) == 16,
                  "custom all-reduce packets must be 16-byte aligned");
    const char *address = reinterpret_cast<const char *>(base) +
                          (size_t)index * sizeof(P);
    uint4 raw;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                 : "l"(address));
#else
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                 : "l"(address));
#endif
    value = *reinterpret_cast<const P *>(&raw);
}

template <typename P>
__device__ __forceinline__ void CustomArStoreGlobal16(
        const P &value, void *base, int index) {
    static_assert(sizeof(P) == 16 && alignof(P) == 16,
                  "custom all-reduce packets must be 16-byte aligned");
    const uint4 raw = *reinterpret_cast<const uint4 *>(&value);
    char *address = reinterpret_cast<char *>(base) +
                    (size_t)index * sizeof(P);
    asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};" ::
                 "r"(raw.x), "r"(raw.y), "r"(raw.z), "r"(raw.w),
                 "l"(address));
}

template <typename P>
__device__ __forceinline__ void CustomArStoreRelaxedSystem16(
        const P &value, void *base, int index) {
    static_assert(sizeof(P) == 16 && alignof(P) == 16,
                  "custom all-reduce packets must be 16-byte aligned");
    const uint4 raw = *reinterpret_cast<const uint4 *>(&value);
    char *address = reinterpret_cast<char *>(base) +
                    (size_t)index * sizeof(P);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};" ::
                 "r"(raw.x), "r"(raw.y), "r"(raw.z), "r"(raw.w),
                 "l"(address));
#else
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::
                 "r"(raw.x), "r"(raw.y), "r"(raw.z), "r"(raw.w),
                 "l"(address));
#endif
}

template <bool UsePdl>
__device__ __forceinline__ void CustomArPdlWait() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (UsePdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
}

template <bool UsePdl>
__device__ __forceinline__ void CustomArPdlTrigger() {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (UsePdl) {
        asm volatile("griddepcontrol.launch_dependents;" :::);
    }
#endif
}

// TP=2 Lamport-style push reduction. Each source writes a 16-byte packet only
// to its peer's workspace; each target combines its direct local input with
// the peer packet polled from local memory. Positive zero is the empty-slot
// marker, so transmitted source +0 values are changed to -0.
// Double buffering and a per-block phase counter avoid a whole-grid peer
// barrier while keeping consecutive CUDA Graph replays independent.
template <typename T, bool UsePdl, int Rank>
__global__ __launch_bounds__(1024, 1)
void FastllmCustomAllReducePushAddKernel(
        const T *__restrict__ input,
        void *localWorkspace,
        void *peerWorkspace,
        CustomArSignal *selfSignal,
        const T *__restrict__ residual,
        T *__restrict__ output,
        int packedCount, size_t workspaceStride) {
    using P = typename CustomArPacked<T>::P;
    static_assert(Rank == 0 || Rank == 1, "TP=2 rank must be 0 or 1");
    CustomArPdlWait<UsePdl>();

    const int phase = selfSignal->pushCounter[blockIdx.x] & 1U;
    const size_t phaseOffset =
        (size_t)phase * 2ULL * workspaceStride;
    constexpr int peer = Rank ^ 1;
    void *pushPeer =
        reinterpret_cast<char *>(peerWorkspace) +
        phaseOffset + (size_t)Rank * workspaceStride;
    const int globalThread = blockIdx.x * blockDim.x + threadIdx.x;
    const int globalStride = gridDim.x * blockDim.x;

    for (int index = globalThread; index < packedCount;
         index += globalStride) {
        P value;
        CustomArLoadGlobal16(value, input, index);
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            CustomArClearPositiveZero(value.data[item]);
        }
        CustomArStoreRelaxedSystem16(value, pushPeer, index);
    }

    void *pollPeer =
        reinterpret_cast<char *>(localWorkspace) +
        phaseOffset + (size_t)peer * workspaceStride;
    const P *packedResidual = reinterpret_cast<const P *>(residual);
    P empty{};
    for (int index = globalThread; index < packedCount;
         index += globalStride) {
        P local;
        P remote;
        bool ready;
        CustomArLoadGlobal16(local, input, index);
        do {
            CustomArLoadRelaxedSystem16(remote, pollPeer, index);
            ready = true;
#pragma unroll
            for (int item = 0; item < CustomArPacked<T>::size; ++item) {
                ready = ready &&
                        !CustomArIsPositiveZero(remote.data[item]);
            }
        } while (!ready);

        P residualValue = packedResidual[index];
        if constexpr (Rank == 0) {
            CustomArAddPacket<T>(local, remote);
            CustomArAddPacket<T>(local, residualValue);
            CustomArStoreGlobal16(local, output, index);
        } else {
            CustomArAddPacket<T>(remote, local);
            CustomArAddPacket<T>(remote, residualValue);
            CustomArStoreGlobal16(remote, output, index);
        }
        CustomArStoreGlobal16(empty, pollPeer, index);
    }

    CustomArPdlTrigger<UsePdl>();
    __syncthreads();
    if (threadIdx.x == 0) {
        selfSignal->pushCounter[blockIdx.x]++;
    }
}

struct CustomArState {
    std::mutex mutex;
    std::condition_variable condition;
    std::atomic<bool> runtimeEnabled{false};
    std::atomic<uint32_t> enabledPaths{0};
    bool resetting = false;
    bool initializing = false;
    int activeCalls = 0;
    bool initAttempted = false;
    bool initialized = false;
    bool logged = false;
    uint64_t ncclGeneration = 0;
    std::vector<int> devices;
    std::map<int, int> rankByDevice;
    CustomArRankSignals allSignals{};
    CustomArRankScratch allScratch{};
    CustomArPushWorkspaces allPushWorkspaces{};
    void *inplaceScratch[kCustomArMaxRanks]{};
    int pushBlocks = 0;
    bool pushAvailable = false;
    bool pushUsePdl = false;
    std::map<std::vector<uintptr_t>, std::vector<CustomArRankData *> > registrations;

    uint64_t generation = 0;
    int arrived = 0;
    int pendingCount = 0;
    int pendingType = -1;
    bool pendingMismatch = false;
    bool lastRegistrationOk = false;
    bool lastCaptureRegistrationMiss = false;
    std::vector<void *> pendingInputs;
    std::vector<char> pendingCapturing;
    std::vector<CustomArRankData *> lastRankData;
};

CustomArState &GetCustomArState() {
    static CustomArState *state = new CustomArState();
    return *state;
}

struct CustomArAllocation {
    int device = -1;
    void *pointer = nullptr;
};

std::vector<CustomArAllocation> DetachCustomArAllocationsLocked(
        CustomArState &state) {
    std::vector<CustomArAllocation> allocations;
    auto collect = [&](int rank, void *pointer) {
        if (pointer != nullptr && rank >= 0 &&
            rank < (int)state.devices.size()) {
            allocations.push_back({state.devices[rank], pointer});
        }
    };
    for (int rank = 0; rank < (int)state.devices.size() &&
                         rank < kCustomArMaxRanks; ++rank) {
        collect(rank, state.allSignals.signals[rank]);
        collect(rank, state.inplaceScratch[rank]);
        collect(rank, state.allPushWorkspaces.workspaces[rank]);
    }
    for (const auto &registration : state.registrations) {
        for (int rank = 0; rank < (int)registration.second.size(); ++rank) {
            collect(rank, registration.second[rank]);
        }
    }
    state.allSignals = CustomArRankSignals{};
    state.allScratch = CustomArRankScratch{};
    state.allPushWorkspaces = CustomArPushWorkspaces{};
    for (void *&scratch : state.inplaceScratch) {
        scratch = nullptr;
    }
    state.pushBlocks = 0;
    state.pushAvailable = false;
    state.pushUsePdl = false;
    state.registrations.clear();
    return allocations;
}

void DrainAndReleaseCustomArAllocations(
        const std::vector<int> &devices,
        const std::vector<CustomArAllocation> &allocations) {
    int originalDevice = FastllmCudaGetDevice();
    for (int device : devices) {
        FastllmCudaSetDevice(device);
        FastllmCudaSyncDevice(device);
    }
    for (const CustomArAllocation &allocation : allocations) {
        if (allocation.device < 0 || allocation.pointer == nullptr) {
            continue;
        }
        FastllmCudaSetDevice(allocation.device);
        FastllmCudaFree(allocation.pointer);
    }
    if (originalDevice >= 0) {
        FastllmCudaSetDevice(originalDevice);
    }
}

constexpr size_t CustomArMaxBytes() {
    // Match vLLM's custom-all-reduce registration envelope. TP=2 always uses
    // the one-stage direct-read kernel; larger TP groups switch to two-stage
    // instead of falling back at the old DeepSeek-only limit.
    return 8ULL * 1024ULL * 1024ULL;
}

bool CustomArUseTwoStage(size_t ranks, size_t bytes) {
    const size_t threshold = ranks <= 4 ? 512ULL * 1024ULL :
        (ranks >= 8 ? 48ULL * 1024ULL : 256ULL * 1024ULL);
    return ranks > 2 && bytes >= threshold;
}

bool CustomArUsePushAdd(const CustomArState &state, size_t bytes) {
    return state.devices.size() == 2 && state.pushAvailable &&
           bytes <= kCustomArPushMaxBytes;
}

uint32_t CustomArPathFor(size_t ranks, size_t bytes, int dataType) {
    // TP>=4 changes algorithm at this boundary. TP=2 remains one-stage, but
    // separating its bandwidth-oriented messages prevents a decode win from
    // automatically opting large prefill tensors into the same policy.
    bool large = ranks == 2
        ? bytes >= kCustomArAutoTp2LargeBoundary
        : CustomArUseTwoStage(ranks, bytes);
    uint32_t fp16Path = large
        ? kCustomArFp16LargePath : kCustomArFp16SmallPath;
    if (dataType == fastllm::DataType::BFLOAT16) {
        return fp16Path << 2;
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return fp16Path << 4;
    }
    return fp16Path;
}

bool CustomArPathEnabled(const CustomArState &state, size_t bytes,
                         int dataType) {
    if (!state.runtimeEnabled.load(std::memory_order_acquire)) {
        return false;
    }
    uint32_t enabled = state.enabledPaths.load(std::memory_order_acquire);
    return (enabled & CustomArPathFor(
        state.devices.size(), bytes, dataType)) != 0;
}

class CustomArCallLease {
public:
    CustomArCallLease() = default;
    CustomArCallLease(const CustomArCallLease &) = delete;
    CustomArCallLease &operator=(const CustomArCallLease &) = delete;

    ~CustomArCallLease() {
        Release();
    }

    bool Acquire(CustomArState &target, size_t bytes, int dataType,
                 int deviceId, bool requireEnabledPath, int &rank) {
        std::lock_guard<std::mutex> lock(target.mutex);
        if (target.resetting || !target.initialized ||
            target.ncclGeneration != FastllmGetNcclGeneration() ||
            (requireEnabledPath &&
             !CustomArPathEnabled(target, bytes, dataType))) {
            return false;
        }
        auto rankIt = target.rankByDevice.find(deviceId);
        if (rankIt == target.rankByDevice.end()) {
            return false;
        }
        target.activeCalls++;
        state = &target;
        rank = rankIt->second;
        return true;
    }

private:
    void Release() {
        if (state == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(state->mutex);
        state->activeCalls--;
        state->condition.notify_all();
        state = nullptr;
    }

    CustomArState *state = nullptr;
};

size_t CustomArTypeBytes(int dataType) {
    if (dataType == fastllm::DataType::FLOAT16 ||
        dataType == fastllm::DataType::BFLOAT16) {
        return 2;
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return 4;
    }
    return 0;
}

bool BuildCustomArRegistration(CustomArState &state,
                               const std::vector<void *> &inputs,
                               std::vector<CustomArRankData *> &rankData) {
    CustomArRankData hostData{};
    for (int rank = 0; rank < (int)inputs.size(); ++rank) {
        hostData.ptrs[rank] = inputs[rank];
    }
    int originalDevice = FastllmCudaGetDevice();
    rankData.assign(inputs.size(), nullptr);
    bool ok = true;
    for (int rank = 0; rank < (int)inputs.size(); ++rank) {
        FastllmCudaSetDevice(state.devices[rank]);
        rankData[rank] = reinterpret_cast<CustomArRankData *>(
            FastllmCudaMalloc(sizeof(CustomArRankData)));
        if (rankData[rank] == nullptr) {
            ok = false;
            break;
        }
        cudaError_t copyState = cudaMemcpy(rankData[rank], &hostData,
                                           sizeof(hostData),
                                           cudaMemcpyHostToDevice);
        if (copyState != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
    }
    FastllmCudaSetDevice(originalDevice);
    if (!ok) {
        std::vector<CustomArAllocation> allocations;
        for (int rank = 0; rank < (int)rankData.size(); ++rank) {
            if (rankData[rank] != nullptr) {
                allocations.push_back({state.devices[rank], rankData[rank]});
            }
        }
        DrainAndReleaseCustomArAllocations(state.devices, allocations);
        rankData.assign(inputs.size(), nullptr);
    }
    return ok;
}

bool FindOrRegisterCustomArPointers(CustomArState &state, int rank,
                                    void *input, int count, int dataType,
                                    CustomArRankData *&rankData) {
    // Registration is itself collective.  A local-pointer-only lookup is
    // unsafe because allocators can reuse an address on some ranks but not on
    // others; hit ranks would launch while miss ranks wait for registration.
    // Rendezvous on the full pointer tuple every time.  CUDA Graph replay does
    // not execute this host path, so steady-state decode pays no host barrier.
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t captureState = cudaStreamIsCapturing(cudaStreamPerThread,
                                                     &captureStatus);
    bool captureQueryOk = captureState == cudaSuccess;
    bool capturing = captureQueryOk &&
                     captureStatus != cudaStreamCaptureStatusNone;
    if (!captureQueryOk) {
        cudaGetLastError();
    }

    std::unique_lock<std::mutex> lock(state.mutex);
    uint64_t waitGeneration = state.generation;
    if (state.arrived == 0) {
        state.pendingCount = count;
        state.pendingType = dataType;
        state.pendingMismatch = false;
        state.pendingInputs.assign(state.devices.size(), nullptr);
        state.pendingCapturing.assign(state.devices.size(), 0);
    } else if (state.pendingCount != count || state.pendingType != dataType) {
        state.pendingMismatch = true;
    }
    if (rank < 0 || rank >= (int)state.pendingInputs.size() ||
        state.pendingInputs[rank] != nullptr) {
        state.pendingMismatch = true;
    } else {
        state.pendingInputs[rank] = input;
        state.pendingCapturing[rank] = capturing ? 1 : 0;
    }
    state.pendingMismatch = state.pendingMismatch || !captureQueryOk;
    state.arrived++;

    if (state.arrived == (int)state.devices.size()) {
        std::vector<void *> inputs = state.pendingInputs;
        bool inputsValid = !state.pendingMismatch;
        for (void *ptr : inputs) {
            inputsValid = inputsValid && ptr != nullptr;
        }
        std::vector<uintptr_t> key;
        key.reserve(inputs.size());
        for (void *ptr : inputs) {
            key.push_back(reinterpret_cast<uintptr_t>(ptr));
        }
        bool anyCapturing = std::any_of(
            state.pendingCapturing.begin(), state.pendingCapturing.end(),
            [](char value) { return value != 0; });
        std::vector<CustomArRankData *> rankDatas;
        auto existing = state.registrations.find(key);
        bool alreadyRegistered = existing != state.registrations.end();
        bool ok = inputsValid && alreadyRegistered;
        if (ok) {
            rankDatas = existing->second;
        } else if (inputsValid && !anyCapturing) {
            lock.unlock();
            ok = BuildCustomArRegistration(state, inputs, rankDatas);
            lock.lock();
            if (ok) {
                state.registrations[key] = rankDatas;
            }
        }
        state.lastCaptureRegistrationMiss =
            inputsValid && anyCapturing && !alreadyRegistered;
        state.lastRegistrationOk = ok;
        state.lastRankData = ok ? rankDatas :
            std::vector<CustomArRankData *>();
        state.arrived = 0;
        state.generation++;
        state.condition.notify_all();
    } else {
        state.condition.wait(lock, [&state, waitGeneration]() {
            return state.generation != waitGeneration;
        });
    }

    if (!state.lastRegistrationOk) {
        if (state.lastCaptureRegistrationMiss) {
            FastllmCudaSetThreadError();
            if (rank == 0) {
                std::fprintf(stderr,
                             "[Fastllm] custom all-reduce rejected an "
                             "unregistered pointer tuple during capture.\n");
                std::fflush(stderr);
            }
        }
        return false;
    }
    if (rank < 0 || rank >= (int)state.lastRankData.size() ||
        state.lastRankData[rank] == nullptr) {
        return false;
    }
    rankData = state.lastRankData[rank];
    return true;
}

template <typename T>
bool LaunchCustomAr(CustomArState &state, CustomArRankData *rankData,
                    T *output, int rank, int count, bool writeAfterBarrier) {
    constexpr int packedWidth = CustomArPacked<T>::size;
    int packedCount = count / packedWidth;
    const size_t bytes = (size_t)count * sizeof(T);
    const bool useTwoStage = CustomArUseTwoStage(state.devices.size(), bytes);
    // In the two-stage algorithm every rank reduces only its 1/RANKS shard
    // and then gathers equally sized peer shards.  Size the grid for that
    // largest shard; using the full tensor count creates RANKS-1 idle blocks
    // for DSpark's 7/8-row hidden states and multiplies barrier traffic.
    const int ranks = (int)state.devices.size();
    int workPackedCount = useTwoStage
        ? packedCount / ranks + packedCount % ranks
        : packedCount;
    int blocks = std::max(1, std::min(kCustomArMaxBlocks,
                          (workPackedCount + kCustomArThreads - 1) /
                              kCustomArThreads));
#define CUSTOM_AR_RANK_CASE(RANKS)                                           \
    case RANKS:                                                              \
        if (useTwoStage) {                                                   \
            FastllmCustomAllReduceTwoStageKernel<T, RANKS>                   \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(       \
                    rankData, state.allSignals,                              \
                    state.allSignals.signals[rank], state.allScratch,        \
                    output, rank, packedCount);                              \
        } else {                                                             \
            FastllmCustomAllReduceKernel<T, RANKS>                           \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(       \
                    rankData, state.allSignals,                              \
                    state.allSignals.signals[rank], output, rank,            \
                    packedCount, writeAfterBarrier);                         \
        }                                                                    \
        break
    switch (state.devices.size()) {
        CUSTOM_AR_RANK_CASE(2);
        CUSTOM_AR_RANK_CASE(4);
        CUSTOM_AR_RANK_CASE(6);
        CUSTOM_AR_RANK_CASE(8);
        default:
            return false;
    }
#undef CUSTOM_AR_RANK_CASE
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce launch failed on GPU %d: %s.\n",
                     state.devices[rank], cudaGetErrorString(launchState));
        std::fflush(stderr);
        return false;
    }
    return true;
}

template <typename T>
bool LaunchCustomArAdd(CustomArState &state, CustomArRankData *rankData,
                       const T *input, const T *residual, T *output,
                       int rank, int count) {
    if (state.devices.size() != 2) {
        return false;
    }
    constexpr int packedWidth = CustomArPacked<T>::size;
    int packedCount = count / packedWidth;
    const size_t bytes = (size_t)count * sizeof(T);
    if (CustomArUsePushAdd(state, bytes)) {
        // Match the compact pull kernel's occupancy for decode-sized
        // messages. Launching one CTA per SM left most CTAs empty at B8 and
        // displaced adjacent CUDA Graph work without improving PCIe traffic.
        int threads = kCustomArThreads;
        while (threads < 1024 &&
               (packedCount + threads - 1) / threads > state.pushBlocks) {
            threads *= 2;
        }
        const int blocks = std::max(
            1, std::min(state.pushBlocks,
                        (packedCount + threads - 1) / threads));
        cudaError_t launchState = cudaSuccess;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
        if (state.pushUsePdl) {
            cudaLaunchConfig_t config{};
            config.gridDim = dim3(blocks);
            config.blockDim = dim3(threads);
            config.dynamicSmemBytes = 0;
            config.stream = cudaStreamPerThread;
            cudaLaunchAttribute attribute{};
            attribute.id =
                cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute.val.programmaticStreamSerializationAllowed = 1;
            config.attrs = &attribute;
            config.numAttrs = 1;
            if (rank == 0) {
                launchState = cudaLaunchKernelEx(
                    &config,
                    FastllmCustomAllReducePushAddKernel<T, true, 0>,
                    input, state.allPushWorkspaces.workspaces[0],
                    state.allPushWorkspaces.workspaces[1],
                    state.allSignals.signals[0], residual, output,
                    packedCount, kCustomArPushMaxBytes);
            } else {
                launchState = cudaLaunchKernelEx(
                    &config,
                    FastllmCustomAllReducePushAddKernel<T, true, 1>,
                    input, state.allPushWorkspaces.workspaces[1],
                    state.allPushWorkspaces.workspaces[0],
                    state.allSignals.signals[1], residual, output,
                    packedCount, kCustomArPushMaxBytes);
            }
        } else
#endif
        {
            if (rank == 0) {
                FastllmCustomAllReducePushAddKernel<T, false, 0>
                    <<<blocks, threads, 0, cudaStreamPerThread>>>(
                        input, state.allPushWorkspaces.workspaces[0],
                        state.allPushWorkspaces.workspaces[1],
                        state.allSignals.signals[0], residual, output,
                        packedCount, kCustomArPushMaxBytes);
            } else {
                FastllmCustomAllReducePushAddKernel<T, false, 1>
                    <<<blocks, threads, 0, cudaStreamPerThread>>>(
                        input, state.allPushWorkspaces.workspaces[1],
                        state.allPushWorkspaces.workspaces[0],
                        state.allSignals.signals[1], residual, output,
                        packedCount, kCustomArPushMaxBytes);
            }
            launchState = cudaGetLastError();
        }
        if (launchState != cudaSuccess) {
            FastllmCudaSetThreadError();
            std::fprintf(stderr,
                         "[Fastllm] fused push all-reduce-add launch failed "
                         "on GPU %d: %s.\n",
                         state.devices[rank],
                         cudaGetErrorString(launchState));
            std::fflush(stderr);
            return false;
        }
        return true;
    }

    int blocks = std::max(1, std::min(kCustomArMaxBlocks,
                          (packedCount + kCustomArThreads - 1) /
                              kCustomArThreads));
    FastllmCustomAllReduceAddKernel<T>
        <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(
            rankData, state.allSignals, state.allSignals.signals[rank],
            residual, output, rank, packedCount);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr,
                     "[Fastllm] fused custom all-reduce-add launch failed "
                     "on GPU %d: %s.\n",
                     state.devices[rank], cudaGetErrorString(launchState));
        std::fflush(stderr);
        return false;
    }
    return true;
}

template <typename T>
bool LaunchCustomArPairAdd(CustomArState &state,
                           CustomArRankData *firstRankData,
                           CustomArRankData *secondRankData,
                           T *output, int rank, int count) {
    constexpr int packedWidth = CustomArPacked<T>::size;
    const int packedCount = count / packedWidth;
    const size_t bytes = (size_t)count * sizeof(T);
    const bool useTwoStage = CustomArUseTwoStage(state.devices.size(), bytes);
    const int ranks = (int)state.devices.size();
    const int workPackedCount = useTwoStage
        ? packedCount / ranks + packedCount % ranks
        : packedCount;
    const int blocks = std::max(
        1, std::min(kCustomArMaxBlocks,
                    (workPackedCount + kCustomArThreads - 1) /
                        kCustomArThreads));
#define CUSTOM_AR_PAIR_RANK_CASE(RANKS)                              \
    case RANKS:                                                      \
        if (useTwoStage) {                                           \
            FastllmCustomAllReducePairAddTwoStageKernel<T, RANKS>    \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(\
                    firstRankData, secondRankData, state.allSignals, \
                    state.allSignals.signals[rank], state.allScratch,\
                    output, rank, packedCount);                      \
        } else {                                                     \
            FastllmCustomAllReducePairAddKernel<T, RANKS>            \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(\
                    firstRankData, secondRankData, state.allSignals, \
                    state.allSignals.signals[rank], state.allScratch,\
                    output, rank, packedCount);                      \
        }                                                            \
        break
    switch (state.devices.size()) {
        CUSTOM_AR_PAIR_RANK_CASE(2);
        CUSTOM_AR_PAIR_RANK_CASE(4);
        CUSTOM_AR_PAIR_RANK_CASE(6);
        CUSTOM_AR_PAIR_RANK_CASE(8);
        default:
            return false;
    }
#undef CUSTOM_AR_PAIR_RANK_CASE
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr,
                     "[Fastllm] paired custom all-reduce-add launch failed "
                     "on GPU %d: %s.\n",
                     state.devices[rank], cudaGetErrorString(launchState));
        std::fflush(stderr);
        return false;
    }
    return true;
}

bool RunCustomArCandidate(void *data, void *dest, int count,
                          int dataType, int deviceId,
                          bool requireEnabledPath = false) {
    if (data == nullptr || dest == nullptr || count <= 0) {
        return false;
    }
    size_t typeBytes = CustomArTypeBytes(dataType);
    size_t bytes = (size_t)count * typeBytes;
    if (typeBytes == 0 || bytes == 0 || bytes > CustomArMaxBytes() ||
        bytes % 16 != 0) {
        return false;
    }

    CustomArState &state = GetCustomArState();
    int rank = -1;
    CustomArCallLease lease;
    if (!lease.Acquire(state, bytes, dataType, deviceId,
                       requireEnabledPath, rank)) {
        return false;
    }

    CustomArRankData *rankData = nullptr;
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, rankData)) {
        return false;
    }
    const bool useTwoStage = CustomArUseTwoStage(state.devices.size(), bytes);
    const bool fusedCopyBack =
        data == dest && !useTwoStage &&
        bytes <= (size_t)kCustomArMaxBlocks * kCustomArThreads * 16 &&
        CustomArFusedCopyBackEnabledByEnv();
    // Two-stage reduction is already safe in-place.  Decode-sized one-stage
    // reductions keep their result in registers until every peer has consumed
    // the inputs; larger messages retain the established scratch copy-back.
    void *kernelDest = data == dest && !useTwoStage && !fusedCopyBack
        ? state.inplaceScratch[rank] : dest;
    bool launched = false;
    if (dataType == fastllm::DataType::FLOAT16) {
        launched = LaunchCustomAr(state, rankData,
                                  reinterpret_cast<half *>(kernelDest),
                                  rank, count, fusedCopyBack);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        launched = LaunchCustomAr(
            state, rankData, reinterpret_cast<__nv_bfloat16 *>(kernelDest),
            rank, count, fusedCopyBack);
    } else if (dataType == fastllm::DataType::FLOAT32) {
        launched = LaunchCustomAr(state, rankData,
                                  reinterpret_cast<float *>(kernelDest),
                                  rank, count, fusedCopyBack);
    }
    if (!launched) {
        return false;
    }
    if (kernelDest != dest) {
        cudaError_t copyState = cudaMemcpyAsync(
            dest, kernelDest, bytes, cudaMemcpyDeviceToDevice,
            cudaStreamPerThread);
        if (copyState != cudaSuccess) {
            FastllmCudaSetThreadError();
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce copy-back failed on "
                         "GPU %d: %s.\n",
                         deviceId, cudaGetErrorString(copyState));
            std::fflush(stderr);
            return false;
        }
    }
    return true;
}

bool RunCustomArPairAddCandidate(void *first, void *second, void *dest,
                                 int count, int dataType, int deviceId,
                                 bool requireEnabledPath = false) {
    if (first == nullptr || second == nullptr || dest == nullptr ||
        count <= 0) {
        return false;
    }
    const size_t typeBytes = CustomArTypeBytes(dataType);
    const size_t bytes = (size_t)count * typeBytes;
    if (typeBytes == 0 || bytes == 0 || bytes > CustomArMaxBytes() ||
        bytes % 16 != 0) {
        return false;
    }

    CustomArState &state = GetCustomArState();
    int rank = -1;
    CustomArCallLease lease;
    if (!lease.Acquire(state, bytes, dataType, deviceId,
                       requireEnabledPath, rank)) {
        return false;
    }

    CustomArRankData *firstRankData = nullptr;
    CustomArRankData *secondRankData = nullptr;
    if (!FindOrRegisterCustomArPointers(
            state, rank, first, count, dataType, firstRankData) ||
        !FindOrRegisterCustomArPointers(
            state, rank, second, count, dataType, secondRankData)) {
        return false;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        return LaunchCustomArPairAdd(
            state, firstRankData, secondRankData,
            reinterpret_cast<half *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::BFLOAT16) {
        return LaunchCustomArPairAdd(
            state, firstRankData, secondRankData,
            reinterpret_cast<__nv_bfloat16 *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return LaunchCustomArPairAdd(
            state, firstRankData, secondRankData,
            reinterpret_cast<float *>(dest), rank, count);
    }
    return false;
}

bool RunCustomArAddCandidate(void *data, void *dest, int count,
                             int dataType, int deviceId,
                             bool requireEnabledPath = false) {
    if (data == nullptr || dest == nullptr || count <= 0) {
        return false;
    }
    size_t typeBytes = CustomArTypeBytes(dataType);
    size_t bytes = (size_t)count * typeBytes;
    if (typeBytes == 0 || bytes == 0 || bytes > CustomArMaxBytes() ||
        bytes % 16 != 0) {
        return false;
    }

    CustomArState &state = GetCustomArState();
    int rank = -1;
    CustomArCallLease lease;
    if (!lease.Acquire(state, bytes, dataType, deviceId,
                       requireEnabledPath, rank) ||
        state.devices.size() != 2) {
        return false;
    }

    CustomArRankData *rankData = nullptr;
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, rankData)) {
        return false;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const half *>(data),
            reinterpret_cast<const half *>(dest),
            reinterpret_cast<half *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::BFLOAT16) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const __nv_bfloat16 *>(data),
            reinterpret_cast<const __nv_bfloat16 *>(dest),
            reinterpret_cast<__nv_bfloat16 *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const float *>(data),
            reinterpret_cast<const float *>(dest),
            reinterpret_cast<float *>(dest), rank, count);
    }
    return false;
}

class CustomArThreadBarrier {
public:
    explicit CustomArThreadBarrier(int participants)
        : participants(participants) {
    }

    bool Wait(bool localOk = true) {
        std::unique_lock<std::mutex> lock(mutex);
        int waitGeneration = generation;
        roundOk = roundOk && localOk;
        arrived++;
        if (arrived == participants) {
            lastResult = roundOk;
            roundOk = true;
            arrived = 0;
            generation++;
            condition.notify_all();
            return lastResult;
        }
        condition.wait(lock, [&]() { return generation != waitGeneration; });
        return lastResult;
    }

private:
    int participants;
    int arrived = 0;
    int generation = 0;
    bool roundOk = true;
    bool lastResult = true;
    std::mutex mutex;
    std::condition_variable condition;
};

using CustomArRankOperation = std::function<bool(int)>;

bool RunCustomArRankOperation(const std::vector<int> &devices,
                              int iterations,
                              const CustomArRankOperation &operation,
                              float &averageUs) {
    if (devices.empty() || iterations <= 0) {
        return false;
    }
    CustomArThreadBarrier barrier((int)devices.size());
    std::vector<char> rankOk(devices.size(), 0);
    std::vector<float> rankMilliseconds(devices.size(), 0.0f);
    std::vector<std::thread> workers;
    workers.reserve(devices.size());
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        workers.emplace_back([&, rank]() {
            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            bool setupOk = cudaSetDevice(devices[rank]) == cudaSuccess;
            setupOk = setupOk &&
                cudaEventCreateWithFlags(&begin, cudaEventDefault) == cudaSuccess;
            setupOk = setupOk &&
                cudaEventCreateWithFlags(&end, cudaEventDefault) == cudaSuccess;
            bool run = barrier.Wait(setupOk);
            bool localOk = setupOk && run;
            if (run) {
                FastllmCudaClearThreadError();
                if (cudaEventRecord(begin, cudaStreamPerThread) != cudaSuccess) {
                    localOk = false;
                }
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    bool iterationOk = operation(rank);
                    localOk = localOk && iterationOk;
                }
                if (cudaEventRecord(end, cudaStreamPerThread) != cudaSuccess ||
                    cudaEventSynchronize(end) != cudaSuccess) {
                    localOk = false;
                }
                if (localOk &&
                    cudaEventElapsedTime(&rankMilliseconds[rank], begin, end) !=
                        cudaSuccess) {
                    localOk = false;
                }
                localOk = localOk && !FastllmCudaGetThreadError();
            }
            if (begin != nullptr) {
                cudaEventDestroy(begin);
            }
            if (end != nullptr) {
                cudaEventDestroy(end);
            }
            if (!localOk) {
                cudaGetLastError();
            }
            rankOk[rank] = localOk ? 1 : 0;
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    bool ok = std::all_of(rankOk.begin(), rankOk.end(),
                          [](char value) { return value != 0; });
    if (!ok) {
        return false;
    }
    float slowestMilliseconds = *std::max_element(
        rankMilliseconds.begin(), rankMilliseconds.end());
    averageUs = slowestMilliseconds * 1000.0f / (float)iterations;
    return std::isfinite(averageUs) && averageUs > 0.0f;
}

bool CaptureCustomArRankGraphs(const std::vector<int> &devices,
                               const CustomArRankOperation &operation,
                               std::vector<cudaGraphExec_t> &execs) {
    execs.assign(devices.size(), nullptr);
    CustomArThreadBarrier barrier((int)devices.size());
    std::vector<char> rankOk(devices.size(), 0);
    std::vector<std::thread> workers;
    workers.reserve(devices.size());
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        workers.emplace_back([&, rank]() {
            bool setupOk = cudaSetDevice(devices[rank]) == cudaSuccess &&
                cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
            if (!barrier.Wait(setupOk)) {
                rankOk[rank] = 0;
                return;
            }

            FastllmCudaClearThreadError();
            cudaGraph_t graph = nullptr;
            bool began = cudaStreamBeginCapture(
                cudaStreamPerThread, cudaStreamCaptureModeRelaxed) == cudaSuccess;
            bool allBegan = barrier.Wait(began);
            if (!allBegan) {
                if (began) {
                    cudaStreamEndCapture(cudaStreamPerThread, &graph);
                    if (graph != nullptr) {
                        cudaGraphDestroy(graph);
                    }
                }
                cudaGetLastError();
                rankOk[rank] = 0;
                return;
            }

            bool operationOk = operation(rank) &&
                !FastllmCudaGetThreadError();
            bool allOperationsOk = barrier.Wait(operationOk);
            cudaError_t endState = cudaStreamEndCapture(
                cudaStreamPerThread, &graph);
            bool localOk = allOperationsOk && endState == cudaSuccess &&
                graph != nullptr;
            if (localOk) {
                localOk = cudaGraphInstantiate(
                    &execs[rank], graph, nullptr, nullptr, 0) == cudaSuccess;
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            if (!localOk) {
                cudaGetLastError();
            }
            rankOk[rank] = localOk ? 1 : 0;
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    return std::all_of(rankOk.begin(), rankOk.end(),
                       [](char value) { return value != 0; });
}

void DestroyCustomArRankGraphs(const std::vector<int> &devices,
                               std::vector<cudaGraphExec_t> &execs) {
    for (int rank = 0; rank < (int)execs.size(); ++rank) {
        if (execs[rank] == nullptr) {
            continue;
        }
        cudaSetDevice(devices[rank]);
        cudaGraphExecDestroy(execs[rank]);
        execs[rank] = nullptr;
    }
}

struct CustomArBenchBuffers {
    std::vector<void *> inputs;
    std::vector<void *> outputs;
};

void FreeCustomArBenchBuffers(const std::vector<int> &devices,
                              CustomArBenchBuffers &buffers) {
    int originalDevice = FastllmCudaGetDevice();
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        if (rank < (int)buffers.inputs.size() &&
            buffers.inputs[rank] != nullptr) {
            cudaFree(buffers.inputs[rank]);
        }
        if (rank < (int)buffers.outputs.size() &&
            buffers.outputs[rank] != nullptr) {
            cudaFree(buffers.outputs[rank]);
        }
    }
    FastllmCudaSetDevice(originalDevice);
    buffers.inputs.clear();
    buffers.outputs.clear();
}

bool AllocateCustomArBenchBuffers(const std::vector<int> &devices,
                                  size_t bytes,
                                  CustomArBenchBuffers &buffers) {
    buffers.inputs.assign(devices.size(), nullptr);
    buffers.outputs.assign(devices.size(), nullptr);
    int originalDevice = FastllmCudaGetDevice();
    bool ok = true;
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        if (cudaMalloc(&buffers.inputs[rank], bytes) != cudaSuccess ||
            cudaMalloc(&buffers.outputs[rank], bytes) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
    }
    FastllmCudaSetDevice(originalDevice);
    if (!ok) {
        FreeCustomArBenchBuffers(devices, buffers);
    }
    return ok;
}

template <typename T>
T CustomArHostFromFloat(float value);

template <>
half CustomArHostFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__nv_bfloat16 CustomArHostFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
float CustomArHostFromFloat<float>(float value) {
    return value;
}

template <typename T>
float CustomArHostToFloat(T value);

template <>
float CustomArHostToFloat<half>(half value) {
    return __half2float(value);
}

template <>
float CustomArHostToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
float CustomArHostToFloat<float>(float value) {
    return value;
}

template <typename T>
bool CheckCustomArCorrectnessTyped(const CustomArState &state,
                                   CustomArBenchBuffers &buffers,
                                   size_t bytes, int dataType) {
    const int count = (int)(bytes / sizeof(T));
    std::vector<float> expected(count, 0.0f);
    std::vector<T> host(count);
    int originalDevice = FastllmCudaGetDevice();
    bool ok = true;
    for (int rank = 0; rank < (int)state.devices.size() && ok; ++rank) {
        for (int index = 0; index < count; ++index) {
            float value = (float)(rank + 1) +
                (float)(index % 17) * 0.03125f;
            host[index] = CustomArHostFromFloat<T>(value);
            expected[index] += CustomArHostToFloat<T>(host[index]);
        }
        FastllmCudaSetDevice(state.devices[rank]);
        ok = cudaMemcpy(buffers.inputs[rank], host.data(), bytes,
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemset(buffers.outputs[rank], 0, bytes) == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
        }
    }
    FastllmCudaSetDevice(originalDevice);
    if (!ok) {
        return false;
    }

    float ignoredUs = 0.0f;
    ok = RunCustomArRankOperation(
        state.devices, 1,
        [&](int rank) {
            return RunCustomArCandidate(
                buffers.inputs[rank], buffers.outputs[rank], count,
                dataType, state.devices[rank]);
        },
        ignoredUs);
    if (!ok) {
        return false;
    }

    for (int rank = 0; rank < (int)state.devices.size() && ok; ++rank) {
        FastllmCudaSetDevice(state.devices[rank]);
        if (cudaMemcpy(host.data(), buffers.outputs[rank], bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        for (int index = 0; index < count; ++index) {
            float expectedRounded = CustomArHostToFloat<T>(
                CustomArHostFromFloat<T>(expected[index]));
            float actual = CustomArHostToFloat<T>(host[index]);
            if (std::fabs(actual - expectedRounded) > 0.001f) {
                std::fprintf(stderr,
                             "[Fastllm] custom all-reduce auto-test "
                             "mismatch on GPU %d at element %d: expected "
                             "%.6f, got %.6f.\n",
                             state.devices[rank], index,
                             expectedRounded, actual);
                ok = false;
                break;
            }
        }
    }
    FastllmCudaSetDevice(originalDevice);
    return ok;
}

bool CheckCustomArCorrectness(const CustomArState &state,
                              CustomArBenchBuffers &buffers,
                              size_t bytes, int dataType) {
    if (dataType == fastllm::DataType::FLOAT16) {
        return CheckCustomArCorrectnessTyped<half>(
            state, buffers, bytes, dataType);
    }
    if (dataType == fastllm::DataType::BFLOAT16) {
        return CheckCustomArCorrectnessTyped<__nv_bfloat16>(
            state, buffers, bytes, dataType);
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return CheckCustomArCorrectnessTyped<float>(
            state, buffers, bytes, dataType);
    }
    return false;
}

bool ZeroCustomArBenchInputs(const std::vector<int> &devices,
                             CustomArBenchBuffers &buffers,
                             size_t bytes) {
    int originalDevice = FastllmCudaGetDevice();
    bool ok = true;
    for (int rank = 0; rank < (int)devices.size() && ok; ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        ok = cudaMemset(buffers.inputs[rank], 0, bytes) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
        if (!ok) {
            cudaGetLastError();
        }
    }
    FastllmCudaSetDevice(originalDevice);
    return ok;
}

float CustomArMedian(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool BenchmarkCustomArPath(const CustomArState &state,
                           CustomArBenchBuffers &buffers,
                           size_t bytes, int dataType, bool useCudaGraph,
                           float &customUs, float &ncclUs) {
    size_t typeBytes = CustomArTypeBytes(dataType);
    if (typeBytes == 0 || bytes % typeBytes != 0) {
        return false;
    }
    const int count = (int)(bytes / typeBytes);
    CustomArRankOperation eagerCustom = [&](int rank) {
        return RunCustomArCandidate(
            buffers.inputs[rank], buffers.inputs[rank], count,
            dataType, state.devices[rank]);
    };
    CustomArRankOperation eagerNccl = [&](int rank) {
        FastllmNcclAllReduceNoCustom(
            buffers.inputs[rank], buffers.inputs[rank], count,
            dataType, state.devices[rank]);
        return !FastllmCudaGetThreadError();
    };

    std::vector<cudaGraphExec_t> customGraphs;
    std::vector<cudaGraphExec_t> ncclGraphs;
    CustomArRankOperation measuredCustom = eagerCustom;
    CustomArRankOperation measuredNccl = eagerNccl;
    float ignoredUs = 0.0f;

    // Register the benchmark pointer tuple outside capture. A registration
    // miss during capture is intentionally rejected by the production path.
    if (!ZeroCustomArBenchInputs(state.devices, buffers, bytes) ||
        !RunCustomArRankOperation(state.devices, 1, eagerCustom, ignoredUs)) {
        return false;
    }

    if (useCudaGraph) {
        if (!CaptureCustomArRankGraphs(
                state.devices, eagerCustom, customGraphs) ||
            !CaptureCustomArRankGraphs(
                state.devices, eagerNccl, ncclGraphs)) {
            DestroyCustomArRankGraphs(state.devices, customGraphs);
            DestroyCustomArRankGraphs(state.devices, ncclGraphs);
            return false;
        }
        measuredCustom = [&](int rank) {
            return cudaGraphLaunch(customGraphs[rank], cudaStreamPerThread) ==
                cudaSuccess;
        };
        measuredNccl = [&](int rank) {
            return cudaGraphLaunch(ncclGraphs[rank], cudaStreamPerThread) ==
                cudaSuccess;
        };
    }

    auto cleanupGraphs = [&]() {
        DestroyCustomArRankGraphs(state.devices, customGraphs);
        DestroyCustomArRankGraphs(state.devices, ncclGraphs);
    };
    constexpr int warmupIterations = 8;
    int iterations = bytes <= kCustomArAutoSmallBytes ? 100 : 20;
    if (!ZeroCustomArBenchInputs(state.devices, buffers, bytes) ||
        !RunCustomArRankOperation(
            state.devices, warmupIterations, measuredCustom, ignoredUs) ||
        !ZeroCustomArBenchInputs(state.devices, buffers, bytes) ||
        !RunCustomArRankOperation(
            state.devices, warmupIterations, measuredNccl, ignoredUs)) {
        cleanupGraphs();
        return false;
    }

    std::vector<float> customSamples;
    std::vector<float> ncclSamples;
    // Alternate order and take five samples so clock ramp and a single busy
    // sample cannot flip the startup policy near the selection margin.
    for (int round = 0; round < 5; ++round) {
        float firstUs = 0.0f;
        float secondUs = 0.0f;
        CustomArRankOperation &first =
            (round & 1) == 0 ? measuredCustom : measuredNccl;
        CustomArRankOperation &second =
            (round & 1) == 0 ? measuredNccl : measuredCustom;
        if (!ZeroCustomArBenchInputs(state.devices, buffers, bytes) ||
            !RunCustomArRankOperation(
                state.devices, iterations, first, firstUs) ||
            !ZeroCustomArBenchInputs(state.devices, buffers, bytes) ||
            !RunCustomArRankOperation(
                state.devices, iterations, second, secondUs)) {
            cleanupGraphs();
            return false;
        }
        if ((round & 1) == 0) {
            customSamples.push_back(firstUs);
            ncclSamples.push_back(secondUs);
        } else {
            ncclSamples.push_back(firstUs);
            customSamples.push_back(secondUs);
        }
    }
    cleanupGraphs();
    customUs = CustomArMedian(customSamples);
    ncclUs = CustomArMedian(ncclSamples);
    return std::isfinite(customUs) && std::isfinite(ncclUs) &&
        customUs > 0.0f && ncclUs > 0.0f;
}

uint32_t AutoTuneCustomAr(CustomArState &state) {
    struct ScopedNcclForceSync {
        bool previous = FastllmCudaGetNcclForceSync();
        ScopedNcclForceSync() {
            FastllmCudaSetNcclForceSync(false);
        }
        ~ScopedNcclForceSync() {
            FastllmCudaSetNcclForceSync(previous);
        }
    } scopedNcclForceSync;

    const bool useCudaGraph = fastllm::GetFastllmEnv().cudaGraph;
    struct PathCase {
        uint32_t bit;
        size_t bytes;
        int dataType;
        const char *typeName;
        const char *sizeName;
    };
    const PathCase paths[] = {
        {kCustomArFp16SmallPath, kCustomArAutoSmallBytes,
         (int)fastllm::DataType::FLOAT16, "FP16", "small"},
        {kCustomArFp16LargePath, kCustomArAutoLargeBytes,
         (int)fastllm::DataType::FLOAT16, "FP16", "large"},
        {kCustomArBf16SmallPath, kCustomArAutoSmallBytes,
         (int)fastllm::DataType::BFLOAT16, "BF16", "small"},
        {kCustomArBf16LargePath, kCustomArAutoLargeBytes,
         (int)fastllm::DataType::BFLOAT16, "BF16", "large"},
        {kCustomArFloatSmallPath, kCustomArAutoSmallBytes,
         (int)fastllm::DataType::FLOAT32, "FP32", "small"},
        {kCustomArFloatLargePath, kCustomArAutoLargeBytes,
         (int)fastllm::DataType::FLOAT32, "FP32", "large"},
    };

    uint32_t enabledPaths = 0;
    for (const PathCase &path : paths) {
        CustomArBenchBuffers buffers;
        if (!AllocateCustomArBenchBuffers(state.devices, path.bytes, buffers)) {
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce auto-test could not "
                         "allocate the %s %s-message buffers; keeping NCCL "
                         "for that path.\n",
                         path.typeName, path.sizeName);
            continue;
        }
        bool correctnessOk = CheckCustomArCorrectness(
            state, buffers, path.bytes, path.dataType);
        float customUs = 0.0f;
        float ncclUs = 0.0f;
        bool benchmarkOk = correctnessOk && BenchmarkCustomArPath(
            state, buffers, path.bytes, path.dataType,
            useCudaGraph, customUs, ncclUs);
        FreeCustomArBenchBuffers(state.devices, buffers);

        if (!correctnessOk) {
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce auto-test failed "
                         "correctness for the %s %s-message path; using "
                         "NCCL.\n",
                         path.typeName, path.sizeName);
            continue;
        }
        if (!benchmarkOk) {
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce auto-test could not "
                         "benchmark the %s %s-message %s path; using NCCL.\n",
                         path.typeName, path.sizeName,
                         useCudaGraph ? "CUDA Graph" : "eager");
            continue;
        }

        bool faster = customUs <= ncclUs * kCustomArAutoRequiredRatio;
        if (faster) {
            enabledPaths |= path.bit;
        }
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce auto-test: %zu GPUs, %s, "
                     "%s %s %zu KiB: custom %.3f us, NCCL %.3f us -> %s "
                     "(requires >= %.0f%% speedup).\n",
                     state.devices.size(),
                     useCudaGraph ? "CUDA Graph replay" : "eager",
                     path.typeName, path.sizeName,
                     (size_t)(path.bytes / 1024ULL),
                     customUs, ncclUs,
                     faster ? "custom enabled" : "NCCL retained",
                     (1.0f - kCustomArAutoRequiredRatio) * 100.0f);
    }
    std::fflush(stderr);
    return enabledPaths;
}

}  // namespace

bool FastllmCudaCustomAllReduceEnabled() {
    CustomArState &state = GetCustomArState();
    if (!state.runtimeEnabled.load(std::memory_order_acquire)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(state.mutex);
    return !state.resetting && state.initialized &&
        state.ncclGeneration == FastllmGetNcclGeneration();
}

bool FastllmCudaCustomAllReduceCanRun(int count, int dataType,
                                      int deviceId) {
    if (count <= 0) {
        return false;
    }
    size_t typeBytes = CustomArTypeBytes(dataType);
    size_t bytes = (size_t)count * typeBytes;
    if (typeBytes == 0 || bytes == 0 || bytes > CustomArMaxBytes() ||
        bytes % 16 != 0) {
        return false;
    }
    CustomArState &state = GetCustomArState();
    std::lock_guard<std::mutex> lock(state.mutex);
    return !state.resetting && state.initialized &&
        state.ncclGeneration == FastllmGetNcclGeneration() &&
        CustomArPathEnabled(state, bytes, dataType) &&
        state.rankByDevice.find(deviceId) != state.rankByDevice.end();
}

void FastllmCudaCustomAllReduceReset() {
    CustomArState &state = GetCustomArState();
    std::vector<int> oldDevices;
    std::vector<CustomArAllocation> allocations;
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        // Stop publishing the old policy before waiting. Calls which already
        // acquired a lease are allowed to enqueue their final kernels; the
        // device drain below keeps their GPU-side state alive until completion.
        state.enabledPaths.store(0, std::memory_order_release);
        state.runtimeEnabled.store(false, std::memory_order_release);
        state.resetting = true;
        state.condition.wait(lock, [&state]() {
            return !state.initializing && state.activeCalls == 0;
        });

        oldDevices = state.devices;
        allocations = DetachCustomArAllocationsLocked(state);
        state.rankByDevice.clear();
        state.initialized = false;
        state.initAttempted = false;
        state.logged = false;
        state.ncclGeneration = 0;
        state.devices.clear();
        state.arrived = 0;
        state.pendingCount = 0;
        state.pendingType = -1;
        state.pendingMismatch = false;
        state.lastRegistrationOk = false;
        state.lastCaptureRegistrationMiss = false;
        state.pendingInputs.clear();
        state.pendingCapturing.clear();
        state.lastRankData.clear();
        state.generation++;
    }

    DrainAndReleaseCustomArAllocations(oldDevices, allocations);

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.resetting = false;
        state.condition.notify_all();
    }
}

bool FastllmCudaCustomAllReduceInit(const std::vector<int> &devices) {
    CustomArEnableMode mode = CustomArModeFromEnv();
    if (mode == CustomArEnableMode::Disabled) {
        LogCustomArDisabledOnce();
        return false;
    }
    const bool autoMode = mode == CustomArEnableMode::Auto;
    if (devices.size() != 2 && devices.size() != 4 &&
        devices.size() != 6 && devices.size() != 8) {
        return false;
    }
    const uint64_t ncclGeneration = FastllmGetNcclGeneration();
    if (ncclGeneration == 0) {
        return false;
    }

    CustomArState &state = GetCustomArState();
    std::unique_lock<std::mutex> lock(state.mutex);
    state.condition.wait(lock, [&state]() {
        return !state.resetting && !state.initializing;
    });
    if (state.initialized) {
        return state.runtimeEnabled.load(std::memory_order_relaxed) &&
               state.devices == devices &&
               state.ncclGeneration == ncclGeneration;
    }
    if (state.initAttempted) {
        return false;
    }
    state.initAttempted = true;
    state.initializing = true;
    state.devices = devices;
    state.ncclGeneration = ncclGeneration;
    state.rankByDevice.clear();
    state.pushBlocks = 0;
    state.pushAvailable = false;
    state.pushUsePdl = false;

    int originalDevice = FastllmCudaGetDevice();
    bool ok = true;
    for (int rank = 0; rank < (int)devices.size() && ok; ++rank) {
        int device = devices[rank];
        state.rankByDevice[device] = rank;
        FastllmCudaSetDevice(device);
        for (int peer : devices) {
            if (peer == device) {
                continue;
            }
            int canAccess = 0;
            cudaError_t peerState = cudaDeviceCanAccessPeer(&canAccess,
                                                             device, peer);
            if (peerState != cudaSuccess || !canAccess) {
                if (peerState != cudaSuccess) {
                    cudaGetLastError();
                }
                ok = false;
                break;
            }
            peerState = cudaDeviceEnablePeerAccess(peer, 0);
            if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();
            } else if (peerState != cudaSuccess) {
                cudaGetLastError();
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }
        CustomArSignal *signal = reinterpret_cast<CustomArSignal *>(
            FastllmCudaMalloc(sizeof(CustomArSignal)));
        if (signal == nullptr) {
            ok = false;
            break;
        }
        state.allSignals.signals[rank] = signal;
        if (cudaMemset(signal, 0, sizeof(CustomArSignal)) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        // The one-stage direct-read algorithm cannot overwrite an input before
        // every peer has consumed it. FastLLM's TP reductions are normally
        // in-place, so reduce into a fixed per-rank scratch buffer and enqueue
        // the copy-back after the kernel's final cross-GPU barrier.
        state.inplaceScratch[rank] = FastllmCudaMalloc(CustomArMaxBytes());
        if (state.inplaceScratch[rank] == nullptr) {
            ok = false;
            break;
        }
        state.allScratch.scratch[rank] = state.inplaceScratch[rank];
    }
    const bool wantPush = ok && devices.size() == 2 &&
                          CustomArPushEnabledFromEnv();
    if (wantPush) {
        bool pushOk = true;
        int minSmCount = kCustomArPushMaxBlocks;
        int minComputeMajor = 100;
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            int smCount = 0;
            int computeMajor = 0;
            if (cudaDeviceGetAttribute(&smCount,
                                       cudaDevAttrMultiProcessorCount,
                                       device) != cudaSuccess ||
                cudaDeviceGetAttribute(&computeMajor,
                                       cudaDevAttrComputeCapabilityMajor,
                                       device) != cudaSuccess ||
                smCount <= 0) {
                cudaGetLastError();
                pushOk = false;
                break;
            }
            minSmCount = std::min(minSmCount, smCount);
            minComputeMajor = std::min(minComputeMajor, computeMajor);
        }
        const size_t workspaceBytes =
            2ULL * devices.size() * kCustomArPushMaxBytes;
        for (int rank = 0; rank < (int)devices.size() && pushOk; ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            void *workspace = FastllmCudaMalloc(workspaceBytes);
            state.allPushWorkspaces.workspaces[rank] = workspace;
            if (workspace == nullptr ||
                cudaMemset(workspace, 0, workspaceBytes) != cudaSuccess) {
                cudaGetLastError();
                pushOk = false;
                break;
            }
        }
        if (pushOk) {
            state.pushBlocks = CustomArPushBlocksFromEnv(minSmCount);
            state.pushAvailable = state.pushBlocks > 0;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
            state.pushUsePdl = minComputeMajor >= 9 &&
                               CustomArPushPdlEnabledFromEnv();
#endif
        } else {
            for (int rank = 0; rank < (int)devices.size(); ++rank) {
                void *workspace = state.allPushWorkspaces.workspaces[rank];
                if (workspace == nullptr) {
                    continue;
                }
                FastllmCudaSetDevice(devices[rank]);
                FastllmCudaFree(workspace);
                state.allPushWorkspaces.workspaces[rank] = nullptr;
            }
        }
    }
    if (ok) {
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                cudaGetLastError();
                ok = false;
                break;
            }
        }
    }
    FastllmCudaSetDevice(originalDevice);
    if (!ok) {
        state.enabledPaths.store(0, std::memory_order_release);
        state.runtimeEnabled.store(false, std::memory_order_release);
        std::vector<int> failedDevices = state.devices;
        std::vector<CustomArAllocation> failedAllocations =
            DetachCustomArAllocationsLocked(state);
        state.rankByDevice.clear();
        state.devices.clear();
        state.ncclGeneration = 0;
        lock.unlock();
        DrainAndReleaseCustomArAllocations(failedDevices,
                                           failedAllocations);
        lock.lock();
        state.initializing = false;
        state.condition.notify_all();
        lock.unlock();
        std::fprintf(stderr,
                     "[Fastllm] graph-safe custom all-reduce is unavailable; "
                     "falling back to NCCL.\n");
        std::fflush(stderr);
        return false;
    }
    state.initialized = true;
    state.enabledPaths.store(kCustomArAllPaths, std::memory_order_release);
    // Keep the public path disabled while auto mode exercises the private
    // candidate. Concurrent callers safely retain NCCL until selection ends.
    state.runtimeEnabled.store(false, std::memory_order_release);
    const bool pushAvailable = state.pushAvailable;
    lock.unlock();

    bool policyPublished = false;
    auto publishPolicy = [&](uint32_t enabledPaths) {
        std::lock_guard<std::mutex> publishLock(state.mutex);
        bool valid = !state.resetting && state.initialized &&
            state.devices == devices &&
            state.ncclGeneration == ncclGeneration &&
            ncclGeneration == FastllmGetNcclGeneration();
        if (valid) {
            state.enabledPaths.store(enabledPaths, std::memory_order_release);
            state.runtimeEnabled.store(enabledPaths != 0,
                                       std::memory_order_release);
        } else {
            state.enabledPaths.store(0, std::memory_order_release);
            state.runtimeEnabled.store(false, std::memory_order_release);
        }
        policyPublished = valid;
        state.initializing = false;
        state.condition.notify_all();
        return valid && enabledPaths != 0;
    };

    if (!autoMode) {
        bool enabled = publishPolicy(kCustomArAllPaths);
        if (!enabled) {
            return false;
        }
        std::fprintf(stderr,
                     "[Fastllm] graph-safe custom all-reduce force-enabled "
                     "for %zu GPUs%s.\n",
                     devices.size(),
                     pushAvailable
                         ? "; TP=2 fused push path enabled" : "");
        std::fflush(stderr);
        return true;
    }

    uint32_t enabledPaths = AutoTuneCustomAr(state);
    bool enabled = publishPolicy(enabledPaths);
    if (!policyPublished) {
        return false;
    }
    if (enabledPaths == 0) {
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce auto mode retained NCCL "
                     "for every tested path.\n");
    } else {
        int selectedPathCount = 0;
        for (uint32_t bit = 1; bit <= kCustomArFloatLargePath; bit <<= 1) {
            selectedPathCount += (enabledPaths & bit) != 0 ? 1 : 0;
        }
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce auto-enabled for %zu GPUs "
                     "(%d of 6 dtype/message paths).\n",
                     devices.size(), selectedPathCount);
    }
    std::fflush(stderr);
    return enabled;
}

bool FastllmCudaCustomAllReduce(void *data, void *dest, int count,
                                int dataType, int deviceId) {
    return RunCustomArCandidate(data, dest, count, dataType, deviceId, true);
}

bool FastllmCudaCustomAllReducePairAdd(void *first, void *second, void *dest,
                                      int count, int dataType, int deviceId) {
    return RunCustomArPairAddCandidate(first, second, dest, count, dataType,
                                       deviceId, true);
}

bool FastllmCudaCustomAllReduceAdd(void *data, void *dest, int count,
                                   int dataType, int deviceId) {
    return RunCustomArAddCandidate(data, dest, count, dataType, deviceId,
                                   true);
}
