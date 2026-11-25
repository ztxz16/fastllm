// Graph-safe small-tensor all-reduce for single-process MultiCUDA.
//
// The kernel/barrier design is adapted from vLLM's Apache-2.0 licensed
// csrc/custom_all_reduce.cuh. Decode-sized messages use its one-stage direct
// peer-read shape; larger TP>=4 messages use reduce-scatter/all-gather. Fixed
// graph workspaces let us register every pointer during warmup.

#include "devices/cuda/fastllm-cuda-pdl.cuh"
#include "devices/cuda/fastllm-cuda-norm.cuh"
#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
#include "devices/cuda/fastllm-swa-prepare.cuh"
#endif
#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"
#include "fastllm-draft-sm120.h"
#include "naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-naive-verify.h"
#include "devices/cuda/naive-n05-swa-verify.cuh"
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "fastllm-custom-allreduce-tagged.cuh"
#include "fastllm-swa-output-state.cuh"

#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
#include "../cuda/models/naive-n05-swa-mma.cuh"
#include "devices/cuda/naive-n05-kv.cuh"
#include "../cuda/models/naive-n05-verify-softmax.cuh"
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <curand_kernel.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#endif

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
    Preferred,
    Auto,
    Forced,
};

CustomArEnableMode CustomArModeFromEnv() {
    static const CustomArEnableMode mode = []() {
        const char *env = std::getenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE");
        if (env == nullptr || env[0] == '\0') {
            return CustomArEnableMode::Preferred;
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
        if (value == "prefer") {
            return CustomArEnableMode::Preferred;
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
                     "unset FASTLLM_CUDA_CUSTOM_ALLREDUCE or set it to prefer "
                     "to use validated custom paths, set it to auto to "
                     "benchmark against NCCL, or to 1 to force-enable.\n");
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

// A captured fixed workspace keeps its registered device table. Transient
// operands travel with the kernel launch, including its graph node if captured.
struct CustomArPointers {
    const CustomArRankData *registered;
    CustomArRankData inputs;
    __device__ __forceinline__ CustomArRankData Load() const {
        return registered ? *registered : inputs;
    }
};

thread_local bool customArTransientInputs = false;

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
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.sys;" :
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
        // Publishing the start flag also publishes every rank-local input
        // write queued before this collective.  Plain volatile accesses only
        // constrain the issuing GPU and are not a cross-device memory-order
        // primitive, which allowed a skewed CUDA Graph replay to observe the
        // flag before the peer input was globally visible.
        CustomArStoreRelease(
            &allSignals.signals[threadIdx.x]->start[blockIdx.x][rank], next);
        while (CustomArLoadAcquire(
                   &self->start[blockIdx.x][threadIdx.x]) != next) {
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        self->flag[blockIdx.x] = next;
    }
}

// The final barrier prevents in-place overwrite or allocator reuse while a
// peer is still reading.  It therefore needs the same system-scope ordering as
// the publish barrier: a peer's completion flag must become visible only after
// all of its reads have completed, and observing that flag must precede the
// local overwrite.
template <int Ranks>
__device__ __forceinline__ void CustomArBarrierFinal(
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

// Owned slices are never overwritten, so peers can copy them directly without
// a staging buffer or arithmetic. Matching start/final barriers publish prior
// writes and prevent the next operator from reusing a source before all reads.
template <int Ranks, bool Flatten, bool Push = false>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllGatherInplaceKernel(CustomArPointers rankData,
        CustomArRankSignals allSignals, CustomArSignal *selfSignal,
        uint4 *output, int rank, int part) {
    CustomArRankData pointers = rankData.Load();
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    const int stride = gridDim.x * blockDim.x;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (Push) {
        // The owned slice is ready on this stream. Send it to each peer's
        // disjoint receive slot, rotating destinations to spread PCIe writes.
        for (int offset = thread; offset < part; offset += stride) {
            const int index = rank * part + offset;
            const uint4 value = output[index];
            #pragma unroll
            for (int i = 1; i < Ranks; ++i) {
                const int dest = (rank + i) % Ranks;
                static_cast<uint4 *>(const_cast<void *>(pointers.ptrs[dest]))[index] = value;
            }
        }
    } else if constexpr (Flatten) {
        // Small messages have fewer than one CTA of payload per peer.
        // Flattening (source, offset) pairs keeps every participating thread
        // on one remote copy instead of serializing all peers through the few
        // payload threads.
        const int copies = part * (Ranks - 1);
        for (int index = thread; index < copies; index += stride) {
            int source = index / part;
            const int offset = index - source * part;
            if (source >= rank) ++source;
            output[source * part + offset] =
                reinterpret_cast<const uint4 *>(pointers.ptrs[source])[source * part + offset];
        }
    } else {
        for (int offset = thread; offset < part; offset += stride) {
#pragma unroll
            for (int source = 0; source < Ranks; ++source)
                if (source != rank)
                    output[source * part + offset] =
                        reinterpret_cast<const uint4 *>(pointers.ptrs[source])[source * part + offset];
        }
    }
    CustomArBarrierFinal<Ranks>(allSignals, selfSignal, rank);
}

// A joint small-message collective: retain rank-ordered FP32 arithmetic and
// gather each owner's already-rounded BF16 rows without doing arithmetic.
// A packet belongs to the same CTA on every rank. Both results stay in
// registers until all peers have finished reading that packet.
template <int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceAndGatherKernel(CustomArPointers routedData,
        CustomArPointers sharedData, CustomArRankSignals allSignals,
        CustomArSignal *selfSignal, float4 *routedOutput, uint2 *sharedOutput,
        int rank, int packedCount) {
    CustomArRankData routed = routedData.Load(), shared = sharedData.Load();
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    float4 sum{};
    uint2 value{};
    if (index < packedCount) {
        sum = reinterpret_cast<const float4 *>(routed.ptrs[0])[index];
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            float4 x = reinterpret_cast<const float4 *>(routed.ptrs[peer])[index];
            sum.x += x.x; sum.y += x.y; sum.z += x.z; sum.w += x.w;
        }
        const int owner = index / (packedCount / Ranks);
        value = reinterpret_cast<const uint2 *>(shared.ptrs[owner])[index];
    }
    CustomArBarrierFinal<Ranks>(allSignals, selfSignal, rank);
    __syncthreads();
    if (index < packedCount) {
        routedOutput[index] = sum;
        sharedOutput[index] = value;
    }
}

// Scalar uint64 maximum: exact integer ordering, including encoded argmax keys.
template <int Ranks, bool Markov>
__global__ void FastllmCustomScalarMaxUint64Kernel(CustomArPointers data,
        CustomArRankSignals signals, CustomArSignal *self, unsigned long long *output, int rank, int *tokens, const __nv_bfloat16 *embed, float *context, int steps) {
    const CustomArRankData pointers = data.Load();
    CustomArBarrierStart<Ranks>(signals, self, rank);
    // One warp owns one request. All warps share one inter-rank rendezvous.
    const int lane = threadIdx.x % 32, row = threadIdx.x / 32;
    unsigned long long value = lane < Ranks ?
        static_cast<const unsigned long long *>(pointers.ptrs[lane])[row] : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const unsigned long long other = __shfl_down_sync(0xffffffffu, value, offset);
        value = value > other ? value : other;
    }
    CustomArBarrierFinal<Ranks>(signals, self, rank);
    __syncthreads();
    if constexpr (Markov) {
        tokens += row * (steps + 1);
        // The caller offsets tokens and context to this speculative step.
        if (context) {
            const auto previous = static_cast<int64_t>(tokens[0]);
#pragma unroll
            for (int i = 0; i < 256; i += 32)
                context[row * steps * 256 + i + lane] = __bfloat162float(embed[previous * 256 + i + lane]);
        }
        if (lane == 0) tokens[1] = int(0xffffffffu - unsigned(value));
    }
    if (lane == 0) output[row] = value;
}

template <typename T, int Ranks, bool OwnerOnly = false>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceKernel(CustomArPointers rankData,
                                  CustomArRankSignals allSignals,
                                  CustomArSignal *selfSignal,
                                  T *__restrict__ output,
                                  int rank, int packedCount,
                                  bool writeAfterBarrier) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData pointers = rankData.Load();
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    P deferredResult;
    int deferredIndex = -1;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         (!OwnerOnly || rank == 0) && index < packedCount; index += gridDim.x * blockDim.x) {
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
        CustomArPointers rankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData pointers = rankData.Load();
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
        P values[Ranks];
#pragma unroll
        for (int source = 0; source < Ranks; ++source) {
            const int sourceCount = source == Ranks - 1
                ? packedCount - source * part : part;
            if (offset < sourceCount) {
                const P *sourceScratch =
                    reinterpret_cast<const P *>(allScratch.scratch[source]);
                values[source] = sourceScratch[offset];
            }
        }
        // Issue peer reads before their dependent output stores, allowing
        // remote memory latency to overlap across ranks. This compiler fence
        // does not replace the system-scope publication barrier above.
        asm volatile("" ::: "memory");
#pragma unroll
        for (int source = 0; source < Ranks; ++source) {
            const int sourceCount = source == Ranks - 1
                ? packedCount - source * part : part;
            if (offset < sourceCount) {
                packedOutput[source * part + offset] = values[source];
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
        CustomArPointers firstRankData,
        CustomArPointers secondRankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData firstPointers = firstRankData.Load();
    CustomArRankData secondPointers = secondRankData.Load();
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
        CustomArPointers firstRankData,
        CustomArPointers secondRankData,
        CustomArRankSignals allSignals,
        CustomArSignal *selfSignal,
        CustomArRankScratch allScratch,
        T *__restrict__ output,
        int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData firstPointers = firstRankData.Load();
    CustomArRankData secondPointers = secondRankData.Load();
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
void FastllmCustomAllReduceAddKernel(CustomArPointers rankData,
                                     CustomArRankSignals allSignals,
                                     CustomArSignal *selfSignal,
                                     const T *__restrict__ residual,
                                     T *__restrict__ output,
                                     int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    CustomArRankData pointers = rankData.Load();
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

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
// BF16 collective/residual rounding stays unchanged. Packed small-row RMSNorm
// uses a 512-thread FP32 square-sum tree; the larger path keeps the established
// 256-thread tree and overlaps affine loads with communication through PDL.
using CustomArNormBF16 = __nv_bfloat16;
__device__ __forceinline__ float CustomArNormRound(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float CustomArNormWarpSum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}
using CustomArNormPacketType = CustomArPacked<CustomArNormBF16>::P;
__device__ __forceinline__ void CustomArNormPacket(CustomArNormPacketType reduced, CustomArNormBF16 *hidden,
                                                 const float *weight, CustomArNormBF16 *output, float eps,
                                                 float *sums, int t) {
    CustomArNormPacketType residual = ((CustomArNormPacketType *)hidden)[t];
    float4 wa = ((const float4 *)weight)[t * 2], wb = ((const float4 *)weight)[t * 2 + 1];
    float cached[8], affine[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w}, sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        cached[i] = CustomArNormRound((float)residual.data[i] + (float)reduced.data[i]);
        sum += cached[i] * cached[i];
    }
    sum = CustomArNormWarpSum(sum);
    if (t % 32 == 0)
        sums[t / 32] = sum;
    __syncthreads();
    if (t < 32) {
        float total = CustomArNormWarpSum(t < 16 ? sums[t] : 0);
        if (t == 0)
            sums[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(sums[0] / 4096 + eps);
    CustomArNormPacketType norm;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        residual.data[i] = __float2bfloat16_rn(cached[i]);
        norm.data[i] = __float2bfloat16_rn(CustomArNormRound(cached[i] * scale) * affine[i]);
    }
    ((CustomArNormPacketType *)hidden)[t] = residual;
    ((CustomArNormPacketType *)output)[t] = norm;
}
template <int Ranks>
__device__ __forceinline__ CustomArNormPacketType CustomArReducePacket(CustomArRankData p, int offset) {
    CustomArNormPacketType first = ((const CustomArNormPacketType *)p.ptrs[0])[offset];
    float sum[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        sum[i] = (float)first.data[i];
#pragma unroll
    for (int peer = 1; peer < Ranks; ++peer) {
        CustomArNormPacketType v = ((const CustomArNormPacketType *)p.ptrs[peer])[offset];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            sum[i] += (float)v.data[i];
    }
    CustomArNormPacketType result;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        result.data[i] = __float2bfloat16_rn(sum[i]);
    return result;
}

// Slot zero is no longer needed once this rank has summed the expert rows.
// Each CTA publishes and consumes exactly its own packet range on every rank.
struct ExpertNormPack {
    FastllmMxfp8Activation target;
    const CustomArNormBF16 *residual;
    CustomArNormBF16 *feature;
    int stride;
    __device__ void operator()(int row, int col, CustomArNormBF16 x) const {
        if (feature) feature[size_t(row) * stride + col] = residual[row * 4096 + col];
        if (target.values) {
            const CustomArNormBF16 one[] = {x};
            fastllm_mxfp8::StoreGroup(target, row, col, one);
        }
    }
};
// Each row is split into eight contiguous 1 KiB slices. Write each slice to
// its reduction owner, so all eight operands of the TP sum are local reads.
template <bool Early>
__global__ void CustomArExpertRowsPeerPush(CustomArRankScratch boxes, CustomArRankSignals signals,
        CustomArNormPacketType *partials, const int32_t *ids, CustomArNormBF16 *out,
        int rank, CustomArNormBF16 *residual, const float *weight, CustomArNormBF16 *normed,
        float eps, FastllmMxfp8Activation next, CustomArNormBF16 *feature, int featureStride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    const int row = blockIdx.x, t = threadIdx.x;
    if (!residual || next.values) cudaTriggerProgrammaticLaunchCompletion();
    unsigned mask = 0;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        int id;
        asm volatile("ld.global.s32 %0, [%1];" : "=r"(id) : "l"(ids + row * 8 + slot) : "memory");
        mask |= unsigned(id >= 0) << slot;
    }
    if constexpr (Early) {
        if (residual) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"(weight + t * 8) : "memory");
            asm volatile("prefetch.global.L2 [%0];" :: "l"(residual + row * 4096 + t * 8) : "memory");
        }
        fastllm::cuda_pdl::Wait();
    }
    auto box = [&](int peer) { return static_cast<CustomArNormPacketType *>(boxes.scratch[peer]); };
    {
        const int col = t;
        auto reduce = [&](auto all) {
            float2 sums[4]{};
#pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                if (!all && !(mask & (1u << slot))) continue;
                const auto value = partials[(row * 8 + slot) * 512 + col];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const auto f = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(&value)[i]);
                    sums[i].x += f.x;
                    sums[i].y += f.y;
                }
            }
            CustomArNormPacketType result;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                reinterpret_cast<__nv_bfloat162 *>(&result)[i] = __floats2bfloat162_rn(sums[i].x, sums[i].y);
            return result;
        };
        const auto result = mask == 255 ? reduce(std::true_type{}) : reduce(std::false_type{});
        CustomArStoreGlobal16(result, box(col / 64), (row * 8 + rank) * 64 + col % 64);
    }
    // The second rendezvous of the previous invocation protects the incoming
    // slices. This first rendezvous also retires reads of its returned row.
    __syncthreads();
    CustomArBarrierStart<8>(signals, signals.signals[rank], rank);
    __shared__ CustomArNormPacketType slice[64];
    if (t < 64) {
        float sum[8]{};
#pragma unroll
        for (int peer = 0; peer < 8; ++peer) {
            const auto x = box(rank)[(row * 8 + peer) * 64 + t];
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[i] += float(x.data[i]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) slice[t].data[i] = __float2bfloat16_rn(sum[i]);
    }
    __syncthreads();
    const auto reduced = slice[t % 64];
    CustomArStoreGlobal16(reduced, box(t / 64), 4096 + row * 512 + rank * 64 + t % 64);
    CustomArBarrierPublish<8>(signals, signals.signals[rank], rank);
    const int offset = row * 512 + t;
    const auto result = box(rank)[4096 + offset];
    reinterpret_cast<CustomArNormPacketType *>(out)[offset] = result;
    if (!residual) return;
    auto h = reinterpret_cast<const CustomArNormPacketType *>(residual)[offset];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        reinterpret_cast<__nv_bfloat162 *>(&h)[i] = __hadd2(
            reinterpret_cast<const __nv_bfloat162 *>(&h)[i],
            reinterpret_cast<const __nv_bfloat162 *>(&result)[i]);
    reinterpret_cast<CustomArNormPacketType *>(residual)[offset] = h;
    if (feature) {
        auto *dst = feature + size_t(row) * featureStride + t * 8;
        if (!(reinterpret_cast<uintptr_t>(dst) & 15))
            *reinterpret_cast<CustomArNormPacketType *>(dst) = h;
        else
            for (int i = 0; i < 8; ++i) dst[i] = h.data[i];
    }
    __shared__ CustomArNormPacketType rowHidden[512];
    __shared__ float partial[8];
    rowHidden[t] = h;
    __syncthreads();
    // Preserve the 256-lane reference sum tree before packing eight BF16s per lane.
    if (t < 256) {
        const auto *values = reinterpret_cast<const CustomArNormBF16 *>(rowHidden);
        float sum = 0.f;
#pragma unroll
        for (int i = t; i < 4096; i += 256) {
            const float x = float(values[i]);
            sum += x * x;
        }
        sum = fastllm::cuda::WarpSum(sum);
        if (t % 32 == 0) partial[t / 32] = sum;
    }
    __syncthreads();
    const int lane = t % 32;
    const float inv = rsqrtf(fastllm::cuda::WarpSum(lane < 8 ? partial[lane] : 0.f) / 4096.f + eps);
    CustomArNormPacketType normalized;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        normalized.data[i] = __float2bfloat16_rn(float(__float2bfloat16_rn(float(h.data[i]) * inv)) * weight[t * 8 + i]);
    reinterpret_cast<CustomArNormPacketType *>(normed)[offset] = normalized;
    if (next.values) fastllm_mxfp8::Store8(next, row, t * 8, *reinterpret_cast<const uint4 *>(&normalized));
#endif
}

template <int Ranks, bool Norm = false>
__global__ void CustomArExpertRowsKernel(CustomArPointers registered, CustomArRankSignals signals,
        CustomArSignal *self, CustomArNormPacketType *local, const int32_t *ids, CustomArNormBF16 *out,
        int rank, int packets, int rowPackets, int topk, CustomArNormBF16 *residual = nullptr,
        const float *weight = nullptr, CustomArNormBF16 *normed = nullptr, float eps = 0, FastllmMxfp8Activation next = {},
        CustomArNormBF16 *feature = nullptr, int featureStride = 0) {
    const CustomArRankData peers = registered.Load();
    const int base = blockIdx.x * (Norm ? 512 : blockDim.x) + threadIdx.x;
    constexpr int parts = Norm ? 2 : 1;
    int offsets[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int packet = base + part * blockDim.x;
        if (packet >= packets) continue;
        const int row = packet / rowPackets;
        const int first = row * topk * rowPackets + packet % rowPackets;
        offsets[part] = first;
        float sum[8] = {};
        for (int slot = 0; slot < (Norm ? 8 : topk); ++slot) {
            if (ids[row * topk + slot] < 0) continue;
            const auto value = local[first + slot * rowPackets];
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[i] += float(value.data[i]);
        }
        CustomArNormPacketType result;
#pragma unroll
        for (int i = 0; i < 8; ++i) result.data[i] = __float2bfloat16_rn(sum[i]);
        local[first] = result;
    }
    __syncthreads();
    CustomArBarrierStart<Ranks>(signals, self, rank);
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int packet = base + part * blockDim.x;
        if (packet < packets)
            reinterpret_cast<CustomArNormPacketType *>(out)[packet] = CustomArReducePacket<Ranks>(peers, offsets[part]);
    }
    CustomArBarrierFinal<Ranks>(signals, self, rank);
    if constexpr (Norm) {
        __syncthreads();
        fastllm::cuda_norm::AddRMSNormBody<CustomArNormBF16, 4096, true, false>(
            residual, out, weight, normed, 4096, eps, false, ExpertNormPack{next, residual, feature, featureStride});
    }
}

// Publish each rank's slice in row-major scratch. Both consumers retain their
// own RMS reduction tree and BF16 rounding after gathering the same row.
template <int Ranks>
__device__ __forceinline__ void CustomArPublishRowSlice(CustomArRankData peers, CustomArRankScratch scratch,
        CustomArRankSignals signals, CustomArSignal *self, int rank) {
    constexpr int width = 512 / Ranks, pitch = width + 512 % Ranks;
    if (threadIdx.x < (rank == Ranks - 1 ? pitch : width)) {
        auto *local = static_cast<CustomArNormPacketType *>(scratch.scratch[rank]);
        local[blockIdx.x * pitch + threadIdx.x] =
            CustomArReducePacket<Ranks>(peers, blockIdx.x * 512 + rank * width + threadIdx.x);
    }
    CustomArBarrierPublish<Ranks>(signals, self, rank);
}

template <int Ranks>
__device__ __forceinline__ CustomArNormPacketType CustomArGatherRowPacket(CustomArRankScratch scratch, int col) {
    constexpr int width = 512 / Ranks, pitch = width + 512 % Ranks;
    const int owner = min(col / width, Ranks - 1);
    const auto *source = static_cast<const CustomArNormPacketType *>(scratch.scratch[owner]);
    return source[blockIdx.x * pitch + col - owner * width];
}

// Match the draft's standalone 256-lane RMS reduction and BF16 affine boundary.
__global__ void DraftReduceAddNorm7(CustomArPointers registered, CustomArRankSignals signals,
        CustomArSignal *self, CustomArRankScratch scratch, CustomArNormBF16 *residual, const float *weight,
        CustomArNormBF16 *output, float eps, int rank) {
    CustomArBarrierStart<8>(signals, self, rank);
    CustomArPublishRowSlice<8>(registered.Load(), scratch, signals, self, rank);
    for (int i = threadIdx.x; i < 512; i += 256)
        reinterpret_cast<CustomArNormPacketType *>(output)[blockIdx.x * 512 + i] =
            CustomArGatherRowPacket<8>(scratch, i);
    __syncthreads();
    fastllm::cuda_norm::AddRMSNormBody<CustomArNormBF16, 4096, true, false>(residual, output,
        weight, output, 4096, eps, false, fastllm::cuda_norm::NoNormSideOutput{});
}

template <int Ranks, bool Direct>
__global__ __launch_bounds__(512, 1) void FastllmCustomAllReduceRMSNormKernel(
    CustomArPointers data, CustomArRankSignals signals, CustomArSignal *self, CustomArRankScratch scratch,
    CustomArNormBF16 *hidden, const float *weight, CustomArNormBF16 *output, float eps, int rank) {
    __shared__ float sums[16];
    // One CTA owns one complete row; the launcher caps the grid at the
    // number of available collective signal slots.
    int t = threadIdx.x, row = blockIdx.x;
    CustomArRankData p = data.Load();
    CustomArBarrierStart<Ranks>(signals, self, rank);
    if constexpr (!Direct) CustomArPublishRowSlice<Ranks>(p, scratch, signals, self, rank);
    // Publish completes all reads of the original input. Two-stage peers now
    // read only scratch; the next collective's start barrier cannot pass until
    // every peer has finished this kernel, so it also protects scratch reuse.
    CustomArNormPacketType v;
    if constexpr (Direct)
        v = CustomArReducePacket<Ranks>(p, row * 512 + t);
    else
        v = CustomArGatherRowPacket<Ranks>(scratch, t);
    CustomArNormPacket(v, hidden + row * 4096, weight, output + row * 4096, eps, sums, t);
    if constexpr (Direct)
        CustomArBarrierFinal<Ranks>(signals, self, rank);
}
template <int Ranks, bool TwoStage>
__global__ void FastllmCustomAllReduceResidualKernel(CustomArPointers data, CustomArRankSignals signals,
                                                   CustomArSignal *self, CustomArRankScratch scratch,
                                                   CustomArNormBF16 *hidden, int rank, int packedCount) {
    CustomArRankData p = data.Load();
    int t = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    CustomArBarrierStart<Ranks>(signals, self, rank);
    if constexpr (TwoStage) {
        int part = packedCount / Ranks, first = rank * part,
            last = rank == Ranks - 1 ? packedCount : first + part;
        for (int i = first + t; i < last; i += stride)
            ((CustomArNormPacketType *)scratch.scratch[rank])[i - first] = CustomArReducePacket<Ranks>(p, i);
        CustomArBarrierPublish<Ranks>(signals, self, rank);
        for (int offset = t; offset < part + packedCount % Ranks; offset += stride) {
            CustomArNormPacketType values[Ranks], residuals[Ranks];
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                int n = peer == Ranks - 1 ? packedCount - peer * part : part;
                if (offset < n) {
                    values[peer] = ((const CustomArNormPacketType *)scratch.scratch[peer])[offset];
                    residuals[peer] = ((CustomArNormPacketType *)hidden)[peer * part + offset];
                }
            }
            asm volatile("" ::: "memory");
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                int n = peer == Ranks - 1 ? packedCount - peer * part : part;
                if (offset < n) {
                    CustomArAddPacket<CustomArNormBF16>(values[peer], residuals[peer]);
                    ((CustomArNormPacketType *)hidden)[peer * part + offset] = values[peer];
                }
            }
        }
    } else {
        for (int i = t; i < packedCount; i += stride) {
            CustomArNormPacketType value = CustomArReducePacket<Ranks>(p, i),
                                   residual = ((CustomArNormPacketType *)hidden)[i];
            CustomArAddPacket<CustomArNormBF16>(value, residual);
            ((CustomArNormPacketType *)hidden)[i] = value;
        }
        CustomArBarrierFinal<Ranks>(signals, self, rank);
    }
}
__global__ void FastllmCustomRMSNormPdlKernel(const CustomArNormBF16 *input, const float *weight,
                                           CustomArNormBF16 *output, float eps) {
    __shared__ float sums[8];
    float value[16], affine[16], partial = 0;
    int row = blockIdx.x;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        affine[i] = weight[threadIdx.x + i * 256];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#pragma unroll
    for (int i = 0; i < 16; ++i)
        value[i] = (float)input[row * 4096 + threadIdx.x + i * 256];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        partial += value[i] * value[i];
    partial = CustomArNormWarpSum(partial);
    if (threadIdx.x % 32 == 0)
        sums[threadIdx.x / 32] = partial;
    __syncthreads();
    if (threadIdx.x < 32) {
        float total = CustomArNormWarpSum(threadIdx.x < 8 ? sums[threadIdx.x] : 0);
        if (threadIdx.x == 0)
            sums[0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(sums[0] / 4096 + eps);
#pragma unroll
    for (int i = 0; i < 16; ++i)
        output[row * 4096 + threadIdx.x + i * 256] =
            __float2bfloat16_rn(CustomArNormRound(value[i] * scale) * affine[i]);
}

template <int Ranks, int RowsPerBlock>
__global__ __launch_bounds__(256 * RowsPerBlock, 1)
void FastllmCustomAllGatherRMSNormKernel(CustomArPointers data,
        CustomArRankSignals signals, CustomArSignal *self, CustomArNormBF16 *hidden,
        const float *weight, CustomArNormBF16 *output, float eps, int rank, int rows) {
    __shared__ float sums[RowsPerBlock][8];
    CustomArRankData pointers = data.Load();
    CustomArBarrierStart<Ranks>(signals, self, rank);
    // The launcher fits all row groups into the existing collective signal
    // slots. Each group keeps the same 256-lane RMSNorm reduction tree.
    const int t = threadIdx.x % 256, slot = threadIdx.x / 256;
    const int row = blockIdx.x * RowsPerBlock + slot;
    // The owner ranges are [floor(N*r/R), floor(N*(r+1)/R)). This also
    // handles N < R without reading an empty owner's allocation.
    const int owner = ((int64_t)(row + 1) * Ranks - 1) / rows;
    const auto *branch = row < rows
        ? (const CustomArNormBF16 *)pointers.ptrs[owner] + row * 4096 : nullptr;
    float value[16], affine[16], partial = 0;
    if (row < rows) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            int c = t + i * 256;
            value[i] = CustomArNormRound((float)hidden[row * 4096 + c] + (float)branch[c]);
            affine[i] = weight[c];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            hidden[row * 4096 + t + i * 256] = __float2bfloat16_rn(value[i]);
            partial += value[i] * value[i];
        }
    }
    partial = CustomArNormWarpSum(partial);
    if (t % 32 == 0) sums[slot][t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        float total = CustomArNormWarpSum(t < 8 ? sums[slot][t] : 0);
        if (t == 0) sums[slot][0] = total;
    }
    __syncthreads();
    float scale = rsqrtf(sums[slot][0] / 4096 + eps);
    if (row < rows) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
            output[row * 4096 + t + i * 256] =
                __float2bfloat16_rn(CustomArNormRound(value[i] * scale) * affine[i]);
    }
    // Protect the owner slices until every peer has finished reading, even
    // when the next graph node immediately reuses the MoE output allocation.
    CustomArBarrierFinal<Ranks>(signals, self, rank);
}

template <int Ranks>
__global__ void FastllmCustomAllGatherResidualKernel(CustomArPointers data,
        CustomArRankSignals signals, CustomArSignal *self,
        CustomArNormBF16 *hidden, int rank, int rows) {
    CustomArRankData pointers = data.Load();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
    CustomArBarrierStart<Ranks>(signals, self, rank);
    const int maxPart = ((rows + Ranks - 1) / Ranks) * (4096 / 8);
    for (int offset = blockIdx.x * blockDim.x + threadIdx.x;
         offset < maxPart; offset += gridDim.x * blockDim.x) {
        CustomArNormPacketType values[Ranks], residuals[Ranks];
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            const int first = (int64_t)rows * peer / Ranks * (4096 / 8);
            const int end = (int64_t)rows * (peer + 1) / Ranks * (4096 / 8);
            if (first + offset < end) {
                values[peer] = ((const CustomArNormPacketType *)pointers.ptrs[peer])[first + offset];
                residuals[peer] = ((const CustomArNormPacketType *)hidden)[first + offset];
            }
        }
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            const int first = (int64_t)rows * peer / Ranks * (4096 / 8);
            const int end = (int64_t)rows * (peer + 1) / Ranks * (4096 / 8);
            if (first + offset < end) {
                CustomArAddPacket<CustomArNormBF16>(values[peer], residuals[peer]);
                ((CustomArNormPacketType *)hidden)[first + offset] = values[peer];
            }
        }
    }
    CustomArBarrierFinal<Ranks>(signals, self, rank);
}
#endif

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
    CustomArRankScratch expertBoxes{};
    CustomArPushWorkspaces allPushWorkspaces{};
    std::unique_ptr<CustomArTaggedPeer> taggedPeer;
    std::unique_ptr<SwaOutputState> swaOutput;
    void *inplaceScratch[kCustomArMaxRanks]{};
    int pushBlocks = 0;
    bool sm120EightRanks = false, expertRowsPush = false;
    bool fastSwaVerify = false, fastDsaVerify = false;
    int splitDsaGrid = 0;
    bool pushAvailable = false;
    bool pushUsePdl = false;
    bool captureFallbackLogged = false;
    std::map<std::vector<uintptr_t>, std::vector<CustomArRankData *> > registrations;

    std::atomic<uint64_t> generation{0};
    int arrived = 0;
    int pendingCount = 0;
    int pendingType = -1;
    bool pendingMismatch = false;
    bool pendingTransientInputs = false;
    // Per-rank call ordinals identify *which* tensor a peer is on.  The
    // pointer tuple alone pairs ranks by arrival order, so any asymmetry in
    // the ranks' collective call streams shifts every later round (silent
    // wrong results, or an out-of-range read of a freed peer buffer).
    uint64_t rankCallSequence[kCustomArMaxRanks]{};
    uint64_t pendingSequences[kCustomArMaxRanks]{};
    bool sequenceDesyncLogged = false;
    bool lastRegistrationOk = false;
    bool lastRegistrationMissDuringCapture = false;
    std::vector<void *> pendingInputs;
    std::vector<char> pendingCapturing;
    // Pointer-local guard results (non-null and 16-byte aligned).  Unlike the
    // byte size and dtype, a pointer's alignment can differ from rank to rank,
    // so this must not be decided locally: every rank folds its own flag in
    // here and the rendezvous makes the launch-or-fall-back decision
    // collective.
    std::vector<char> pendingAligned;
    // Every rank consumes the published round before the next round can finish.
    // A successful round uses either a stable map entry or the inline peer tuple.
    const std::vector<CustomArRankData *> *lastRegistration = nullptr;
    CustomArRankData lastInputs{};
};

CustomArState &GetCustomArState() {
    static CustomArState *state = new CustomArState();
    return *state;
}

struct CustomArAllocation {
    int device = -1;
    void *pointer = nullptr;
    std::unique_ptr<CustomArTaggedPeer> taggedPeer;
    std::unique_ptr<SwaOutputState> swaOutput;
};

std::vector<CustomArAllocation> DetachCustomArAllocationsLocked(
        CustomArState &state) {
    std::vector<CustomArAllocation> allocations;
    if (state.taggedPeer) {
        allocations.push_back({-1, nullptr, std::move(state.taggedPeer)});
    }
    if (state.swaOutput) {
        CustomArAllocation allocation;
        allocation.swaOutput = std::move(state.swaOutput);
        allocations.push_back(std::move(allocation));
    }
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
    state.expertBoxes = CustomArRankScratch{};
    state.allPushWorkspaces = CustomArPushWorkspaces{};
    for (void *&scratch : state.inplaceScratch) {
        scratch = nullptr;
    }
    state.pushBlocks = 0;
    state.pushAvailable = false;
    state.pushUsePdl = false;
    state.lastRegistration = nullptr;
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
        if (allocation.swaOutput) {
            allocation.swaOutput->ReleaseAfterDrain();
            continue;
        }
        if (allocation.taggedPeer) {
            allocation.taggedPeer->ReleaseAfterDrain();
            continue;
        }
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
        FastllmCudaGraphRecordCollective();
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

bool CustomArBuffersDisjoint(const void *a, size_t bytesA, const void *b, size_t bytesB) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x >= bytesA : x - y >= bytesB;
}

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
        // Registration runs on the last arriving worker, including for its
        // peers' devices. Their pools may contain temporaries returned while
        // writes are still queued on another per-thread stream. Drain those
        // writers before reusing storage for this persistent pointer table.
        // This is only the cold registration path; cached tuples and graph
        // replay do not synchronize here.
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        rankData[rank] = reinterpret_cast<CustomArRankData *>(
            FastllmCudaMalloc(sizeof(CustomArRankData)));
        if (rankData[rank] == nullptr) {
            ok = false;
            break;
        }
        cudaError_t copyState = cudaMemcpy(rankData[rank], &hostData,
                                           sizeof(hostData),
                                           cudaMemcpyHostToDevice);
        // The last arriving worker publishes metadata for every GPU on its
        // own per-thread streams. A pageable H2D cudaMemcpy may return before
        // the device copy completes; the other workers consume these tables
        // on different streams as soon as the host rendezvous is released.
        // Finish each cold registration before publishing the pointer tuple.
        if (copyState == cudaSuccess) {
            copyState = cudaStreamSynchronize(cudaStreamPerThread);
        }
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
                                    bool localEligible,
                                    CustomArPointers &rankData,
                                    int hostSpinMicroseconds = 0) {
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
        state.pendingTransientInputs = false;
        state.pendingCount = count;
        state.pendingType = dataType;
        state.pendingMismatch = false;
        state.pendingInputs.assign(state.devices.size(), nullptr);
        state.pendingCapturing.assign(state.devices.size(), 0);
        state.pendingAligned.assign(state.devices.size(), 0);
    } else if (state.pendingCount != count || state.pendingType != dataType) {
        state.pendingMismatch = true;
    }
    if (rank < 0 || rank >= (int)state.pendingInputs.size() ||
        state.pendingInputs[rank] != nullptr) {
        state.pendingMismatch = true;
    } else {
        state.pendingInputs[rank] = input;
        state.pendingCapturing[rank] = capturing ? 1 : 0;
        state.pendingAligned[rank] = localEligible ? 1 : 0;
        state.pendingSequences[rank] = state.rankCallSequence[rank]++;
    }
    state.pendingMismatch = state.pendingMismatch || !captureQueryOk;
    state.pendingTransientInputs |= customArTransientInputs;
    state.arrived++;

    if (state.arrived == (int)state.devices.size()) {
        const auto &inputs = state.pendingInputs;
        bool inputsValid = !state.pendingMismatch;
        for (void *ptr : inputs) {
            inputsValid = inputsValid && ptr != nullptr;
        }
        // Fold in every rank's local pointer-eligibility (non-null and 16-byte
        // aligned).  A rank with an unusable local view must not have bailed
        // out before arriving here: had it done so its peers would spin in this
        // rendezvous, or in the device barrier, forever.  One bad rank now
        // drops all ranks to NCCL together.
        for (char eligible : state.pendingAligned) {
            inputsValid = inputsValid && eligible != 0;
        }
        // A desynchronised call stream shifts the arrival-order pairing by
        // one; refuse the round so it falls back to NCCL instead of
        // reducing the wrong peer tensor (or reading a freed buffer).
        for (int other = 1; other < (int)state.devices.size() && inputsValid;
             ++other) {
            if (state.pendingSequences[other] != state.pendingSequences[0]) {
                inputsValid = false;
                state.pendingMismatch = true;
                if (rank == 0 && !state.sequenceDesyncLogged) {
                    state.sequenceDesyncLogged = true;
                    std::fprintf(stderr,
                                 "[Fastllm] custom all-reduce call "
                                 "sequences desynchronized across ranks; "
                                 "falling back to NCCL.\n");
                    std::fflush(stderr);
                }
            }
        }
        state.lastRegistration = nullptr;
        state.lastRegistrationMissDuringCapture = false;
        state.lastRegistrationOk = inputsValid;
        if (state.pendingTransientInputs) {
            std::copy(inputs.begin(), inputs.end(), state.lastInputs.ptrs);
        } else if (inputsValid) {
            std::vector<uintptr_t> key;
            key.reserve(inputs.size());
            for (void *ptr : inputs) {
                key.push_back(reinterpret_cast<uintptr_t>(ptr));
            }
            auto existing = state.registrations.find(key);
            if (existing != state.registrations.end()) {
                state.lastRegistration = &existing->second;
            } else if (std::any_of(state.pendingCapturing.begin(), state.pendingCapturing.end(),
                                   [](char value) { return value != 0; })) {
                state.lastRegistrationMissDuringCapture = true;
            } else {
                const auto registrationInputs = inputs;
                std::vector<CustomArRankData *> tables;
                lock.unlock();
                const bool ok = BuildCustomArRegistration(state, registrationInputs, tables);
                lock.lock();
                if (ok) {
                    auto entry = state.registrations.emplace(std::move(key), std::move(tables));
                    state.lastRegistration = &entry.first->second;
                }
            }
            state.lastRegistrationOk = state.lastRegistration != nullptr;
        }
        state.arrived = 0;
        state.generation++;
        state.condition.notify_all();
    } else {
        // Dedicated decode submitters can meet without a scheduler wake-up.
        // Keep the full tuple/eligibility rendezvous and its sleeping fallback;
        // only the generation is read without holding the registration mutex.
        if (hostSpinMicroseconds > 0 && !capturing) {
            const auto deadline = std::chrono::steady_clock::now() +
                std::chrono::microseconds(hostSpinMicroseconds);
            lock.unlock();
            unsigned spins = 0;
            while (state.generation.load(std::memory_order_acquire) == waitGeneration) {
                if ((++spins & 63U) == 0) {
                    if (std::chrono::steady_clock::now() >= deadline) break;
                    std::this_thread::yield();
                }
            }
            lock.lock();
        }
        state.condition.wait(lock, [&state, waitGeneration]() {
            return state.generation != waitGeneration;
        });
    }

    if (!state.lastRegistrationOk) {
        if (state.lastRegistrationMissDuringCapture) {
            // Registration allocates device metadata and copies the peer
            // pointer tuple, so it cannot be built while a stream is being
            // captured.  No custom kernel has launched yet, and the
            // rendezvous above makes this miss collective across all ranks.
            // Let the caller capture its NCCL/non-fused fallback instead of
            // poisoning an otherwise valid graph.
            if (rank == 0 && !state.captureFallbackLogged) {
                state.captureFallbackLogged = true;
                std::fprintf(stderr,
                             "[Fastllm] custom all-reduce is falling back "
                             "inside CUDA Graph for pointer tuples first "
                             "seen during capture.\n");
                std::fflush(stderr);
            }
        }
        return false;
    }
    if (!state.lastRegistration) {
        rankData.registered = nullptr;
        rankData.inputs = state.lastInputs;
        return true;
    }
    const auto &tables = *state.lastRegistration;
    if (rank < 0 || rank >= (int)tables.size() || tables[rank] == nullptr) {
        return false;
    }
    rankData.registered = tables[rank];
    return true;
}

template <typename T>
bool LaunchCustomAr(CustomArState &state, CustomArPointers rankData,
                    T *output, int rank, int count, bool writeAfterBarrier, int threads) {
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
                          (workPackedCount + threads - 1) /
                              threads));
#define CUSTOM_AR_RANK_CASE(RANKS)                                           \
    case RANKS:                                                              \
        if (useTwoStage) {                                                   \
            FastllmCustomAllReduceTwoStageKernel<T, RANKS>                   \
                <<<blocks, threads, 0, cudaStreamPerThread>>>(               \
                    rankData, state.allSignals,                              \
                    state.allSignals.signals[rank], state.allScratch,        \
                    output, rank, packedCount);                              \
        } else {                                                             \
            FastllmCustomAllReduceKernel<T, RANKS>                           \
                <<<blocks, threads, 0, cudaStreamPerThread>>>(               \
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
bool LaunchCustomArAdd(CustomArState &state, CustomArPointers rankData,
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
                           CustomArPointers firstRankData,
                           CustomArPointers secondRankData,
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

// Registration/eligibility is complete before this function enqueues work.
// Keep one-row and batched reductions on the same rank-ordered arithmetic.
bool LaunchRegisteredCustomAr(CustomArState &state, CustomArPointers rankData,
                              void *data, void *dest, int count, int dataType,
                              int deviceId, int rank, int rowElements) {
    const size_t bytes = (size_t)count * CustomArTypeBytes(dataType);
    const size_t policyBytes = rowElements
        ? (size_t)rowElements * CustomArTypeBytes(dataType) : bytes;
    if ((dataType == fastllm::DataType::BFLOAT16 || dataType == fastllm::DataType::FLOAT32) &&
        state.taggedPeer && state.taggedPeer->CanUse(count, dataType == fastllm::DataType::FLOAT32)) {
        const bool launched = state.taggedPeer->Launch(
            data, dest, count, rank, dataType == fastllm::DataType::FLOAT32);
        if (cudaGetLastError() != cudaSuccess) {
            FastllmCudaSetThreadError();
            return false;
        }
        return launched;
    }
    const bool useTwoStage = CustomArUseTwoStage(state.devices.size(), bytes);
    // A small block of decode rows gives each rank only a few KiB of the
    // reduce-scatter shard. More, smaller CTAs overlap peer-read latency while
    // retaining every start/publish barrier and the rank-ordered FP32 sum.
    // Row-wise collectives keep their existing compact geometry.
    const bool compactRows = rowElements > 0 && count > rowElements &&
        count / rowElements <= 8 && policyBytes <= kCustomArAutoSmallBytes;
    // A two-stage shard of at most 32 KiB otherwise occupies only one or
    // two 512-thread CTAs. Smaller CTAs overlap peer traffic on more SMs.
    const bool smallShard = state.devices.size() == 8 && bytes <= 256 * 1024;
    const int threads = !useTwoStage ? kCustomArThreads : compactRows ? 64 :
        smallShard ? 128 : kCustomArThreads;
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
                                  rank, count, fusedCopyBack, threads);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        launched = LaunchCustomAr(
            state, rankData, reinterpret_cast<__nv_bfloat16 *>(kernelDest),
            rank, count, fusedCopyBack, threads);
    } else if (dataType == fastllm::DataType::FLOAT32) {
        launched = LaunchCustomAr(state, rankData,
                                  reinterpret_cast<float *>(kernelDest),
                                  rank, count, fusedCopyBack, threads);
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

bool RunCustomArCandidate(void *data, void *dest, int count,
                          int dataType, int deviceId,
                          bool requireEnabledPath = false, int rowElements = 0,
                          int hostSpinMicroseconds = 0) {
    if (count <= 0 || rowElements < 0 || (rowElements && count % rowElements)) {
        return false;
    }
    // The kernels access data/dest as 16-byte aligned packets; a misaligned or
    // null view would fault on the device (cudaErrorMisalignedAddress, Xid
    // 13).  Whether *this* rank's pointers are usable is local information and
    // must not be decided here: a rank that returned early would leave its
    // peers waiting in the rendezvous (or in the device barrier) forever.
    // Record the flag and let the collective rendezvous below drop every rank
    // to NCCL together.
    const bool localEligible = data != nullptr && dest != nullptr &&
        (((uintptr_t)data | (uintptr_t)dest) & 15u) == 0;
    size_t typeBytes = CustomArTypeBytes(dataType);
    size_t bytes = (size_t)count * typeBytes;
    if (typeBytes == 0 || bytes == 0 || bytes > CustomArMaxBytes() ||
        bytes % 16 != 0) {
        return false;
    }

    CustomArState &state = GetCustomArState();
    int rank = -1;
    CustomArCallLease lease;
    // Both custom kernels sum ranks 0..N-1 in FP32 and round once.
    // Keep that arithmetic when a verification block crosses the automatic
    // message-size threshold; otherwise switching to NCCL changes logits.
    const size_t policyBytes = rowElements ? (size_t)rowElements * typeBytes : bytes;
    if (policyBytes % 16 != 0) return false;
    if (!lease.Acquire(state, policyBytes, dataType, deviceId,
                       requireEnabledPath, rank)) {
        return false;
    }

    CustomArPointers rankData{};
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, localEligible, rankData, hostSpinMicroseconds)) {
        return false;
    }
    return LaunchRegisteredCustomAr(state, rankData, data, dest, count,
                                    dataType, deviceId, rank, rowElements);
}

bool RunCustomArPairAddCandidate(void *first, void *second, void *dest,
                                 int count, int dataType, int deviceId,
                                 bool requireEnabledPath = false) {
    if (count <= 0) {
        return false;
    }
    // The kernel reads first/second and writes dest as 16-byte aligned
    // packets; a misaligned or null view would fault on the device
    // (cudaErrorMisalignedAddress, Xid 13).  This is local information, so it
    // goes into the collective rendezvous rather than an early return (see
    // RunCustomArCandidate).
    const bool localEligible = first != nullptr && second != nullptr &&
        dest != nullptr &&
        (((uintptr_t)first | (uintptr_t)second | (uintptr_t)dest) & 15u) == 0;
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

    CustomArPointers firstRankData{};
    CustomArPointers secondRankData{};
    if (!FindOrRegisterCustomArPointers(
            state, rank, first, count, dataType, localEligible,
            firstRankData) ||
        !FindOrRegisterCustomArPointers(
            state, rank, second, count, dataType, localEligible,
            secondRankData)) {
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
    if (count <= 0) {
        return false;
    }
    // The kernel reads data/dest as 16-byte aligned packets; a misaligned or
    // null view would fault on the device (cudaErrorMisalignedAddress, Xid
    // 13).  This is local information, so it goes into the collective
    // rendezvous rather than an early return (see RunCustomArCandidate).
    const bool localEligible = data != nullptr && dest != nullptr &&
        (((uintptr_t)data | (uintptr_t)dest) & 15u) == 0;
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

    CustomArPointers rankData{};
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, localEligible, rankData)) {
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
        // The rank workers consume these buffers on other per-thread streams.
        // Pageable H2D copies and memset may still be running when they return.
        ok = cudaMemcpy(buffers.inputs[rank], host.data(), bytes,
                        cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemset(buffers.outputs[rank], 0, bytes) == cudaSuccess &&
             cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
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
            if (!std::isfinite(actual) || std::fabs(actual - expectedRounded) > 0.001f) {
                std::fprintf(stderr,
                             "[Fastllm] custom all-reduce self-test "
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

    // Register the benchmark pointer tuple outside capture so this benchmark
    // measures the custom path instead of the graph-safe NCCL fallback.
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

uint32_t SelectCustomArPaths(CustomArState &state, bool benchmarkPerformance) {
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
                         "[Fastllm] custom all-reduce self-test could not "
                         "allocate the %s %s-message buffers; keeping NCCL "
                         "for that path.\n",
                         path.typeName, path.sizeName);
            continue;
        }
        bool correctnessOk = CheckCustomArCorrectness(
            state, buffers, path.bytes, path.dataType);
        float customUs = 0.0f;
        float ncclUs = 0.0f;
        bool benchmarkOk = correctnessOk && (!benchmarkPerformance ||
            BenchmarkCustomArPath(state, buffers, path.bytes, path.dataType,
                                  useCudaGraph, customUs, ncclUs));
        FreeCustomArBenchBuffers(state.devices, buffers);

        if (!correctnessOk) {
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce self-test failed "
                         "correctness for the %s %s-message path; using "
                         "NCCL.\n",
                         path.typeName, path.sizeName);
            continue;
        }
        if (!benchmarkPerformance) {
            enabledPaths |= path.bit;
            std::fprintf(stderr,
                         "[Fastllm] custom all-reduce self-test: %zu GPUs, "
                         "%s %s %zu KiB: correctness passed -> custom preferred.\n",
                         state.devices.size(), path.typeName, path.sizeName,
                         (size_t)(path.bytes / 1024ULL));
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

#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
#include "fastllm-naive-sm120-verify.cuh"
#include "fastllm-draft-attention-sm120.cuh"
#include "fastllm-draft-mlp-sm120.cuh"
#include "fastllm-draft-head-sm120.cuh"
#include "fastllm-draft-context-sm120.cuh"
#endif
}  // namespace

fastllm::TransientAllReduceInputs::TransientAllReduceInputs(bool enabled)
    : previous(customArTransientInputs) {
    customArTransientInputs |= enabled;
}

fastllm::TransientAllReduceInputs::~TransientAllReduceInputs() {
    customArTransientInputs = previous;
}

#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120

const uint8_t *FastllmCudaSm120GemvScales(const fastllm::Data &, int, int);

struct FastllmNaiveVerifyIndexerSm120 {
    const fastllm::Data *wq, *wk, *weights, *norm, *bias;
    fastllm::Data *indices;
    int rank;
    bool scaled;
};

static bool RunPackedVerifyAttention(const fastllm::Data &input, const fastllm::Data &qkvWeight,
                                       const fastllm::Data &bias, const fastllm::Data &outWeight,
                                       const fastllm::Data &norm, const fastllm::Data &sink,
                                       const fastllm::Data &positions, fastllm::Data &q, fastllm::Data &attn,
                                       fastllm::Data &projected, fastllm::Data &hidden, fastllm::Data &normed,
                                       fastllm::Data &scratch, const std::vector<FastllmNaiveDSARow> &rows,
                                       int stride, bool ring, float theta, float valueScale, float eps, int gpu,
                                       const FastllmNaiveVerifyIndexerSm120 *indexer, FastllmMxfp8Activation prepared,
                                       const fastllm::Data *branch, const fastllm::Data *inputNorm,
                                       fastllm::Data *feature, bool deferPostNorm) {
    using namespace fastllm;
    if (FastllmCudaRuntimeArch() != 120 || (ring == (indexer != nullptr)))
        return false;
    CustomArCallLease lease;
    sm120_verify::Args a{};
    auto &state = GetCustomArState();
    if (!lease.Acquire(state, 8192, BFLOAT16, gpu, true, a.rank) || state.devices.size() != 8)
        return false;
    const int count = rows.size();
    auto local = [&](const Data &x, int type) {
        if (x.dims.empty() || x.strides.size() != x.dims.size())
            return false;
        size_t stride = 1;
        for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
            if (x.dims[i] <= 0 || x.strides[i] != stride)
                return false;
            stride *= x.dims[i];
        }
        return x.dataDevice == CUDA && x.dataDeviceIds == std::vector<int>{gpu} && x.dataType == type &&
               x.cudaData && !(reinterpret_cast<uintptr_t>(x.cudaData) & 15);
    };
    auto matrix = [&](const Data &w, int n, int k) {
        return local(w, FP8_E4M3) && w.dims == std::vector<int>({n, k}) && !w.IsRepacked && w.blockM == 32 &&
               w.blockK == 1 && !w.extraCudaData.empty() && w.extraCudaData[0];
    };
    bool eligible = std::isfinite(eps) && eps > 0 && std::isfinite(theta) && theta > 0 &&
                    std::isfinite(valueScale) && count == 8 &&
                    (ring ? stride == 128 : stride > 0 && stride <= 2048) && local(input, BFLOAT16) &&
                    input.Count(0) == size_t(count) * 4096 && local(hidden, BFLOAT16) &&
                    hidden.Count(0) == input.Count(0) && local(norm, FLOAT32) && norm.Count(0) == 4096 &&
                    (sink.dims.empty() || (local(sink, FLOAT32) && sink.Count(0) == 8)) &&
                    local(positions, FLOAT32) && positions.Count(0) >= count && bias.dims.empty() &&
                    matrix(qkvWeight, 1856, 4096) && matrix(outWeight, 4096, 1024);
    if (!(ring ? state.fastSwaVerify : state.fastDsaVerify)) return false;
    for (const auto &r : rows)
        eligible = eligible && r.fp8 && r.cache && r.liveKeys && r.capacity > 0 && r.keyStride == 192 && !r.fp4 &&
                   r.key && r.value &&
                   (ring ? !r.pageLen && r.verifyOffset >= 0 && r.verifyOffset < 8 : r.verifyOffset < 0);
    int shardWidth = 1024;
    if (indexer) {
        eligible = eligible && indexer->wq && indexer->wk && indexer->weights && indexer->norm &&
                   indexer->bias && indexer->indices;
        auto indexWeight = [&](const Data &w, int n) {
            return local(w, BFLOAT16) && w.dims == std::vector<int>({n, 4096});
        };
        eligible = eligible && !ring && indexer->rank == a.rank && stride == 2048 &&
                   indexWeight(*indexer->wq, 2048) && indexWeight(*indexer->wk, 128) &&
                   indexWeight(*indexer->weights, 16) && local(*indexer->norm, FLOAT32) &&
                   indexer->norm->Count(0) == 128 && local(*indexer->bias, FLOAT32) &&
                   indexer->bias->Count(0) == 128;
        for (const auto &r : rows) {
            if (!r.fp8 || r.capacity <= 2048 || r.pageLen < 8 || r.pageLen % 8) {
                eligible = false;
                break;
            }
            const int64_t pages = (int64_t(r.capacity) + r.pageLen - 1) / r.pageLen;
            const int64_t localKeys = pages * (r.pageLen / 8);
            if (localKeys > 1024)
                shardWidth = std::max(shardWidth, int((localKeys + 2047) / 2048 * 2048));
        }
    }
    // A cooperative grid must fit together, including the CTAs waiting for TP peers.
    auto kernel = ring ? sm120_verify::SwaPackedLayer : sm120_verify::DsaLayer<>;
    int sm = 0, coop = 0, active = 0;
    if (cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, gpu) != cudaSuccess ||
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, gpu) != cudaSuccess || !coop ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, sm120_verify::Threads, 0) !=
            cudaSuccess ||
        active * sm < (ring ? sm120_verify::SwaGrid : sm120_verify::IndexProjectionTiles))
        eligible = false;
    if (branch)
        eligible = eligible && ring && inputNorm && local(*branch, BFLOAT16) && branch->Count(0) == input.Count(0) &&
                   local(*inputNorm, FLOAT32) && inputNorm->Count(0) == 4096 &&
                   (!feature || (local(*feature, BFLOAT16) && feature->Count(0) == input.Count(0)));
    auto output = [&](Data &x, DataType type, std::vector<int> dims) {
        x.dataType = type;
        x.UpdateUnitSize();
        x.Resize(dims);
        x.ToDevice(CUDA, {gpu}, false);
        x.Allocate(false);
    };
    if (eligible) {
        a.swaScales = FastllmCudaSm120GemvScales(qkvWeight, 1856, 4096);
        eligible = a.swaScales && FastllmCudaSm120GemvScales(outWeight, 4096, 1024);
    }
    if (eligible)
        output(projected, BFLOAT16, {1, count, 4096});
    CustomArPointers registered{};
    if (!FindOrRegisterCustomArPointers(state, a.rank, projected.cudaData, count * 4096, BFLOAT16, eligible,
                                        registered))
        return false;
    output(q, BFLOAT16, {1, count, 1536});
    output(attn, BFLOAT16, {1, count, 1024});
    output(normed, BFLOAT16, {1, count, 4096});
    // The FP8 ring needs just the rotary pairs; DSA also materializes attention scores.
    const int scoreFloats = ring ? count * 32 : count * 8 * stride, partialFloats = ring ? 0 : count * ((stride + 63) / 64) * 1024;
    const int packedFloats = count * 4096 / 4, scaleFloats = 4096 / 128 * 512 / 4;
    // Per row, Indexer projection, FP8 queries, scores and merge lists expire
    // before the value partials are produced. Reuse the larger of those arenas.
    const int64_t listWords = std::max<int64_t>(8192, int64_t(shardWidth) * (shardWidth > 2048 ? 2 : 1));
    const int64_t indexFloats = indexer ? 2192 / 2 + 2048 / 4 + 16 + shardWidth + 2 * listWords : 0;
    const int64_t scratchPerRow = (packedFloats + scaleFloats + scoreFloats) / count +
                                  std::max<int64_t>(partialFloats / count, indexFloats);
    output(scratch, FLOAT32, {count, static_cast<int>(scratchPerRow)});
    auto *base = static_cast<uint8_t *>(scratch.cudaData);
    a.quant = {base, base + packedFloats * 4, count, 4096, count, 0};
    a.prepared = prepared.Matches(count,4096) && prepared.swizzledRowOffset >= 0 ? prepared : FastllmMxfp8Activation{};
    a.scores = reinterpret_cast<float *>(base) + packedFloats + scaleFloats;
    a.partials = a.scores + scoreFloats;
    a.input = static_cast<const __nv_bfloat16 *>(input.cudaData);
    a.qkvWeight = qkvWeight.cudaData;
    a.outWeight = outWeight.cudaData;
    a.qkvScales = static_cast<const float *>(qkvWeight.extraCudaData[0]);
    a.outScales = static_cast<const float *>(outWeight.extraCudaData[0]);
    a.query = static_cast<__nv_bfloat16 *>(q.cudaData);
    a.attn = static_cast<__nv_bfloat16 *>(attn.cudaData);
    a.projection = static_cast<__nv_bfloat16 *>(projected.cudaData);
    a.hidden = static_cast<__nv_bfloat16 *>(hidden.cudaData);
    a.normed = static_cast<__nv_bfloat16 *>(normed.cudaData);
    a.gamma = deferPostNorm ? nullptr : static_cast<const float *>(norm.cudaData);
    a.pos = static_cast<const float *>(positions.cudaData);
    a.sink = sink.dims.empty() ? nullptr : static_cast<const float *>(sink.cudaData);
    a.rows = count;
    a.stride = stride;
    a.theta = theta;
    a.valueScale = float(__float2bfloat16_rn(valueScale));
    a.eps = eps;
    a.signals = state.allSignals;
    for (int rank = 0; rank < 8; ++rank)
        a.exchange.scratch[rank] = static_cast<char *>(state.allScratch.scratch[rank]) + CustomArMaxBytes();
    std::copy(rows.begin(), rows.end(), a.batch.rows);
    if (indexer) {
        a.scaledIndex = indexer->scaled;
        a.iqWeight = static_cast<const __nv_bfloat16 *>(indexer->wq->cudaData);
        a.ikWeight = static_cast<const __nv_bfloat16 *>(indexer->wk->cudaData);
        a.iwWeight = static_cast<const __nv_bfloat16 *>(indexer->weights->cudaData);
        a.indexNorm = static_cast<const float *>(indexer->norm->cudaData);
        a.indexBias = static_cast<const float *>(indexer->bias->cudaData);
        // Index scratch expires before sparse attention starts using the value partials.
        a.shardWidth = shardWidth;
        a.indexRaw = reinterpret_cast<__nv_bfloat16 *>(a.partials);
        a.iqBytes = reinterpret_cast<uint8_t *>(a.indexRaw + count * 2192);
        a.iqScale = reinterpret_cast<float *>(a.iqBytes + count * 2048);
        a.indexScores = reinterpret_cast<unsigned *>(a.iqScale + count * 16);
        a.lists = reinterpret_cast<unsigned long long *>(a.indexScores + size_t(count) * shardWidth);
        output(*indexer->indices, INT32, {count, stride});
        a.selected = static_cast<int *>(indexer->indices->cudaData);
        for (int i = 0; i < count; ++i) {
            a.batch.rows[i].indices = a.selected + i * stride;
            a.batch.rows[i].selectedCount = stride;
        }
    }
    int grid = ring ? sm120_verify::SwaGrid : std::min(256, active * sm);
    void *args[] = {&a};
    if (ring) {
        a.prepared = {};
        auto *rotary = reinterpret_cast<__nv_bfloat162 *>(a.scores);
        a.rotary = rotary;
        const bool afterExpert = cuda_pdl::Prepare(cuda_pdl::Producer::VerifyExpertReduce);
        auto prepare = afterExpert ? fastllm_swa_qkv::PrepareSwaQkv<true, true> :
                                     fastllm_swa_qkv::PrepareSwaQkv<false, true>;
        prepare<<<count + 1, 512, 0, cudaStreamPerThread>>>(
            a.hidden, branch ? static_cast<const __nv_bfloat16 *>(branch->cudaData) : nullptr,
            inputNorm ? static_cast<const float *>(inputNorm->cudaData) : nullptr, eps,
            nullptr, nullptr, a.quant.values, a.quant.scales,
            count, a.quant.pitch, a.pos, theta, rotary, branch ? nullptr : a.input,
            feature ? static_cast<__nv_bfloat16 *>(feature->cudaData) : nullptr);
        cuda_pdl::Mark(cuda_pdl::Producer::SwaPrepare);
        cuda_pdl::Prepare(cuda_pdl::Producer::SwaPrepare);
        sm120_verify::SwaQkvGemm<<<116, sm120_verify::Threads, 0, cudaStreamPerThread>>>(a);
        if (cudaPeekAtLastError() != cudaSuccess)
            throw std::runtime_error("SM120 SWA QKV launch failed");
        cuda_pdl::Mark(cuda_pdl::Producer::SwaQkv);
        cuda_pdl::Prepare(cuda_pdl::Producer::SwaQkv);
    }
    auto launch = [&](auto stage) {
        auto status = cudaLaunchCooperativeKernel((const void *)stage, grid, sm120_verify::Threads, args, 0,
                                                   cudaStreamPerThread);
        if (status != cudaSuccess) {
            FastllmCudaSetThreadError();
            throw std::runtime_error(std::string("SM120 verify launch failed: ") + cudaGetErrorString(status));
        }
    };
    if (!ring && state.splitDsaGrid) {
        using namespace sm120_verify;
        grid = state.splitDsaGrid;
        if (a.prepared.values) fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::VerifyExpertReduce);
        launch(DsaLayer<DsaStage::IndexScores>);
        launch(DsaLayer<DsaStage::Select>);
        kernel = DsaLayer<DsaStage::AttentionOutput>;
    }
    launch(kernel);
    if (ring) cuda_pdl::Mark(cuda_pdl::Producer::SwaOutput);
    return true;
}

#endif

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
        for (uint64_t &sequence : state.rankCallSequence) {
            sequence = 0;
        }
        for (uint64_t &sequence : state.pendingSequences) {
            sequence = 0;
        }
        state.sequenceDesyncLogged = false;
        state.lastRegistrationOk = false;
        state.lastRegistrationMissDuringCapture = false;
        state.captureFallbackLogged = false;
        state.pendingInputs.clear();
        state.pendingCapturing.clear();
        state.pendingAligned.clear();
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
    state.sm120EightRanks = devices.size() == 8;
    state.expertRowsPush = state.sm120EightRanks;
    state.fastSwaVerify = state.fastDsaVerify = state.sm120EightRanks && SwaOutputState::Enabled();
    state.splitDsaGrid = 256;
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
        state.sm120EightRanks = state.sm120EightRanks && FastllmCudaRuntimeArch() == 120;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
        if (state.expertRowsPush) {
            cudaFuncAttributes image{};
            state.expertRowsPush = state.sm120EightRanks &&
                cudaFuncGetAttributes(&image, CustomArExpertRowsPeerPush<true>) == cudaSuccess &&
                image.ptxVersion >= 120 && image.maxThreadsPerBlock >= 512;
            if (!state.expertRowsPush) cudaGetLastError();
        }
#else
        state.expertRowsPush = false;
#endif
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
        // Choose one transport and grid size for the entire peer group, including heterogeneous groups.
        if (state.fastSwaVerify || state.fastDsaVerify) {
            int sms = 0, coop = 0;
            const bool cooperative = state.sm120EightRanks &&
                cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
                cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, device) == cudaSuccess && coop;
            auto fits = [&](auto kernel, int minimumBlocks) {
                int blocks = 0;
                if (cooperative && cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &blocks, kernel, sm120_verify::Threads, 0) == cudaSuccess)
                    return blocks * sms >= minimumBlocks;
                cudaGetLastError();
                return false;
            };
            state.fastSwaVerify = state.fastSwaVerify && fits(sm120_verify::SwaPackedLayer, sm120_verify::SwaGrid);
            state.fastDsaVerify = state.fastDsaVerify && fits(sm120_verify::DsaLayer<>, sm120_verify::IndexProjectionTiles);
            if (state.fastDsaVerify && state.splitDsaGrid) {
                using namespace sm120_verify;
                for (auto stage : {DsaLayer<DsaStage::IndexScores>, DsaLayer<DsaStage::Select>,
                                   DsaLayer<DsaStage::AttentionOutput>}) {
                    int blocks = 0;
                    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, stage, Threads, 0) != cudaSuccess) {
                        cudaGetLastError();
                        blocks = 0;
                    }
                    state.splitDsaGrid = std::min(state.splitDsaGrid, sms * blocks);
                }
                if (state.splitDsaGrid < IndexProjectionTiles) state.splitDsaGrid = 0;
            }
        }
#endif
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
        size_t scratchBytes = CustomArMaxBytes();
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
        // Projection and Indexer peer writes must not alias live buffers from adjacent collectives.
        if (state.fastSwaVerify || state.fastDsaVerify) scratchBytes += sm120_verify::VerifyExchangeBytes;
        if (state.fastDsaVerify) scratchBytes += sm120_verify::DsaListBytes;
        if (state.fastDsaVerify && state.splitDsaGrid) scratchBytes += sm120_verify::DsaSelectionBytes;
#endif
        const size_t expertOffset = scratchBytes;
        if (state.expertRowsPush) scratchBytes += 2 * 8 * 4096 * sizeof(uint16_t);
        state.inplaceScratch[rank] = FastllmCudaMalloc(scratchBytes);
        if (state.inplaceScratch[rank] == nullptr) {
            ok = false;
            break;
        }
        state.allScratch.scratch[rank] = state.inplaceScratch[rank];
        if (state.expertRowsPush)
            state.expertBoxes.scratch[rank] = static_cast<char *>(state.inplaceScratch[rank]) + expertOffset;
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
    if (ok) {
        state.taggedPeer = CustomArTaggedPeer::Create(devices);
        state.swaOutput = SwaOutputState::Create(devices);
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
    // Keep the public path disabled while selection exercises the private
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

    if (mode == CustomArEnableMode::Forced) {
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

    uint32_t enabledPaths = SelectCustomArPaths(state, autoMode);
    bool enabled = publishPolicy(enabledPaths);
    if (!policyPublished) {
        return false;
    }
    if (enabledPaths == 0) {
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce %s mode retained NCCL "
                     "for every tested path.\n", autoMode ? "auto" : "prefer");
    } else {
        int selectedPathCount = 0;
        for (uint32_t bit = 1; bit <= kCustomArFloatLargePath; bit <<= 1) {
            selectedPathCount += (enabledPaths & bit) != 0 ? 1 : 0;
        }
        std::fprintf(stderr,
                     "[Fastllm] custom all-reduce %s for %zu GPUs "
                     "(%d of 6 dtype/message paths).\n",
                     autoMode ? "auto-selected" : "preferred",
                     devices.size(), selectedPathCount);
    }
    std::fflush(stderr);
    return enabled;
}

bool FastllmCudaCustomAllReduce(void *data, void *dest, int count,
                                int dataType, int deviceId) {
    return RunCustomArCandidate(data, dest, count, dataType, deviceId, true);
}

// The consumer lives on the first device. Peers publish their BF16 inputs
// and keep them alive through the final barrier without reading other shards.
bool FastllmCudaCustomReduceOwnerBF16(void *data, int count, int owner, int device) {
    if (count <= 0 || count % 8 || size_t(count) * 2 > CustomArMaxBytes()) return false;
    auto &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    const int type = fastllm::DataType::BFLOAT16;
    if (!lease.Acquire(state, size_t(count) * 2, type, device, true, rank) ||
        state.devices.size() != 8 || state.devices.front() != owner) return false;
    CustomArPointers pointers{};
    const bool eligible = data && (reinterpret_cast<uintptr_t>(data) % 16 == 0);
    if (!FindOrRegisterCustomArPointers(state, rank, data, count, type, eligible, pointers)) return false;
    const int packed = count / 8;
    const int threads = 128;
    const int blocks = std::min(kCustomArMaxBlocks, (packed + threads - 1) / threads);
    FastllmCustomAllReduceKernel<__nv_bfloat16, 8, true><<<blocks, threads, 0, cudaStreamPerThread>>>(
        pointers, state.allSignals, state.allSignals.signals[rank], static_cast<__nv_bfloat16 *>(data),
        rank, packed, false);
    if (cudaGetLastError() != cudaSuccess) {
        FastllmCudaSetThreadError();
        return false;
    }
    return true;
}

bool FastllmCudaCustomAllReduceWithSpin(void *data, void *dest, int count,
                                      int dataType, int deviceId, int hostSpinMicroseconds) {
    return RunCustomArCandidate(data, dest, count, dataType, deviceId, true, 0,
                                hostSpinMicroseconds);
}

bool FastllmCudaCustomAllReduceRows(void *data, void *dest, int count,
                                    int rowElements, int dataType, int deviceId) {
    if (count <= 0 || rowElements <= 0 || count % rowElements) return false;
    const size_t typeBytes = CustomArTypeBytes(dataType);
    const size_t bytes = (size_t)count * typeBytes;
    if (bytes <= CustomArMaxBytes())
        return RunCustomArCandidate(data, dest, count, dataType, deviceId, true, rowElements);
    const size_t rowBytes = (size_t)rowElements * typeBytes;
    if (!typeBytes || rowBytes > CustomArMaxBytes() || rowBytes % 16) return false;
    const int chunkElements = (CustomArMaxBytes() / rowBytes) * rowElements;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, rowBytes, dataType, deviceId, true, rank)) return false;
    const bool eligible = data && dest &&
        (((uintptr_t)data | (uintptr_t)dest) & 15u) == 0;
    auto at = [typeBytes](void *pointer, int offset) -> void * {
        return pointer ? (uint8_t *)pointer + (size_t)offset * typeBytes : nullptr;
    };
    // Register every chunk before launching any of them. A cold Graph tuple or
    // one misaligned rank must reject the whole operation without modifying an
    // in-place input that the caller may subsequently reduce through fallback.
    std::vector<CustomArPointers> pointers;
    for (int offset = 0; offset < count;) {
        const int size = std::min(chunkElements, count - offset);
        CustomArPointers registered{};
        if (!FindOrRegisterCustomArPointers(state, rank, at(data, offset), size,
                                            dataType, eligible, registered)) return false;
        pointers.push_back(registered);
        offset += size;
    }
    size_t chunk = 0;
    for (int offset = 0; offset < count;) {
        const int size = std::min(chunkElements, count - offset);
        if (!LaunchRegisteredCustomAr(state, pointers[chunk++], at(data, offset),
                at(dest, offset), size, dataType, deviceId, rank, rowElements)) {
            // A CUDA launch error is not a recoverable eligibility miss: earlier
            // chunks may already have written their reductions back in place.
            throw std::runtime_error("chunked custom all-reduce launch failed");
        }
        offset += size;
    }
    return true;
}


// Preserve the original API and its accumulation policy for existing callers.
bool FastllmCudaSwaOutputAddRMSNorm(const fastllm::Data &input,
        fastllm::Data &weight, fastllm::Data &projected,
        fastllm::Data &hidden, const fastllm::Data &normWeight,
        fastllm::Data &normed, float eps, int deviceId,
        const fastllm::Data *routerWeight, fastllm::Data *router, bool orderedProjection) {
    return FastllmCudaSwaOutputAddRMSNormWithProjection(input, weight, projected, hidden,
        normWeight, normed, eps, deviceId, routerWeight, router, orderedProjection, false);
}

bool FastllmCudaSwaOutputAddRMSNormWithProjection(const fastllm::Data &input,
        fastllm::Data &weight, fastllm::Data &projected,
        fastllm::Data &hidden, const fastllm::Data &normWeight,
        fastllm::Data &normed, float eps, int deviceId,
        const fastllm::Data *routerWeight, fastllm::Data *router, bool orderedProjection, bool allowMmaProjection) {
    return FastllmCudaSwaOutputAddRMSNormPrepared(input, weight, projected, hidden,
        normWeight, normed, eps, deviceId, routerWeight, router, orderedProjection, allowMmaProjection, nullptr);
}

// Row-major E4M3 values and the resident FP32 K32 scales owned by this weight.
static const float *SingleRowMxScales(const fastllm::Data &w, int n, int k) {
    if (w.dataType != fastllm::FP8_E4M3 || w.IsRepacked || w.isFake || w.blockK != 1 ||
        w.blockM != 32 || w.scales.size() != size_t(n) * k / 32 || w.extraCudaData.empty())
        return nullptr;
    const auto *scales = static_cast<const float *>(w.extraCudaData[0]);
    return (uintptr_t(scales) & 15) ? nullptr : scales;
}

bool FastllmCudaSwaOutputAddRMSNormPrepared(const fastllm::Data &input,
        fastllm::Data &weight, fastllm::Data &projected,
        fastllm::Data &hidden, const fastllm::Data &normWeight,
        fastllm::Data &normed, float eps, int deviceId,
        const fastllm::Data *routerWeight, fastllm::Data *router, bool orderedProjection, bool allowMmaProjection,
        const FastllmMoePreparedInput *prepared) {
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, 8192, fastllm::BFLOAT16, deviceId, true, rank)) return false;
    if (!state.swaOutput) return false; // A group-wide, immutable initialization result.
    auto tensor = [&](const fastllm::Data &x, int type, size_t count) {
        if (x.dataType != type || x.dataDevice != fastllm::DataDevice::CUDA ||
            x.dataDeviceIds != std::vector<int>{deviceId} || !x.cudaData ||
            (reinterpret_cast<uintptr_t>(x.cudaData) & 15) || x.dims.empty() ||
            x.strides.size() != x.dims.size()) return false;
        size_t stride = 1;
        for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
            if (x.dims[i] <= 0 || x.strides[i] != stride ||
                stride > count / size_t(x.dims[i])) return false;
            stride *= x.dims[i];
        }
        return stride == count;
    };
    auto prepare = [&](fastllm::Data &x) {
        x.dataType = fastllm::BFLOAT16; x.UpdateUnitSize();
        x.Resize(hidden.dims); x.ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false);
        x.Allocate(false);
    };
    auto disjoint = [](const void *a, size_t na, const void *b, size_t nb) {
        uintptr_t x = uintptr_t(a), y = uintptr_t(b);
        return x <= y ? y - x >= na : x - y >= nb;
    };
    int rows = input.dims.empty() || input.dims.back() != 1024 ? 0 : 1;
    for (size_t i = 0; rows && i + 1 < input.dims.size(); ++i) {
        if (input.dims[i] <= 0 || rows > fastllm_swa_output::MaxRows / input.dims[i]) rows = 0;
        else rows *= input.dims[i];
    }
    const size_t elements = size_t(rows) * 4096, bytes = elements * 2;
    const bool rowKernel = rows > 1 || orderedProjection;
    const bool mma = allowMmaProjection && !orderedProjection && rows > 1 &&
        state.swaOutput->RowsMmaSupported();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool captureQueryOk = !rowKernel ||
        cudaStreamIsCapturing(cudaStreamPerThread, &capture) == cudaSuccess;
    if (!captureQueryOk) cudaGetLastError();
    const bool mx = weight.dataType == fastllm::FP8_E4M3;
    const float *scales = mx && rows == 1 && !rowKernel && !prepared
        ? SingleRowMxScales(weight, 4096, 1024) : nullptr;
    const bool withRouter = routerWeight && router;
    bool eligible = captureQueryOk && rows > 0 && (!mx || (scales && state.swaOutput->MxOutputSupported())) &&
        (!prepared || (withRouter && state.swaOutput->PreparedRouterSupported(prepared->routed, rows) && prepared->Valid(rows))) && (!rowKernel || state.swaOutput->RowsSupported()) &&
        !hidden.dims.empty() && hidden.dims.back() == 4096 &&
        bool(routerWeight) == bool(router) &&
        (!withRouter || rows == 1 || rows < fastllm::FastllmCudaGetLinearExactBatchThreshold()) &&
        (!withRouter || rowKernel || state.swaOutput->OutputRouterSupported()) &&
        tensor(input, fastllm::BFLOAT16, size_t(rows) * 1024) &&
        tensor(weight, mx ? fastllm::FP8_E4M3 : fastllm::BFLOAT16, 4096 * 1024) &&
        weight.dims == std::vector<int>({4096, 1024}) &&
        // Bias-free BF16 Linear needs no bias cache on an owned weight.
        // Keep the prepared-view requirement for borrowed eight-row GEMM weights.
        (mma || rows != 8 || (!weight.isFake && !weight.cudaDataBorrowed) || weight.extraCudaData.size() >= 2) &&
        tensor(hidden, fastllm::BFLOAT16, elements) &&
        tensor(normWeight, fastllm::FLOAT32, 4096) && std::isfinite(eps) && eps > 0;
    if (withRouter) {
        eligible = eligible && &projected != routerWeight && &normed != routerWeight &&
            tensor(*routerWeight, fastllm::FLOAT32, 256 * 4096) &&
            routerWeight->dims == std::vector<int>({256, 4096});
        for (const fastllm::Data *x : std::initializer_list<const fastllm::Data *>{
                &input, &weight, &hidden, &normWeight, &projected, &normed, routerWeight})
            eligible = eligible && router != x;
        if (router->isFake || router->cudaDataBorrowed)
            eligible = eligible && tensor(*router, fastllm::FLOAT32, size_t(rows) * 256);
    }
    // Validate sources before preparing destinations. In particular, an
    // unsupported input or aliased Data object must not be resized/retagged.
    eligible = eligible && &projected != &normed;
    for (const fastllm::Data *source : std::initializer_list<const fastllm::Data *>{
            &input, &weight, &hidden, &normWeight})
        eligible = eligible && &projected != source && &normed != source;
    for (const fastllm::Data *dest : {&projected, &normed})
        if (dest->isFake || dest->cudaDataBorrowed)
            eligible = eligible && tensor(*dest, fastllm::BFLOAT16, elements);
    if (capture != cudaStreamCaptureStatusNone) {
        // Capture may reuse a registered output with new weights or buffers.
        // Reject collectively before any lazy allocation or GPU write. The
        // ordinary eager path prepares the GEMM on this rank.
        eligible = eligible && tensor(projected, fastllm::BFLOAT16, elements) &&
            tensor(normed, fastllm::BFLOAT16, elements) &&
            (!withRouter || tensor(*router, fastllm::FLOAT32, size_t(rows) * 256)) &&
            (mma || rows != 8 || state.swaOutput->RowsGemmPrepared(rank));
    }
    if (eligible) {
        prepare(projected); prepare(normed);
        if (withRouter) {
            auto dims = hidden.dims; dims.back() = 256;
            router->dataType = fastllm::FLOAT32; router->UpdateUnitSize(); router->Resize(dims);
            router->ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false); router->Allocate(false);
        }
    }
    eligible = eligible && tensor(projected, fastllm::BFLOAT16, elements) &&
        tensor(normed, fastllm::BFLOAT16, elements);
    const void *outputs[] = {projected.cudaData, hidden.cudaData, normed.cudaData};
    for (int i = 0; i < 3; ++i) {
        eligible = eligible && disjoint(outputs[i], bytes, input.cudaData, size_t(rows) * 2048) &&
            disjoint(outputs[i], bytes, weight.cudaData, weight.GetBytes()) &&
            (!scales || disjoint(outputs[i], bytes, scales, 4096 * 1024 / 32 * sizeof(float))) &&
            disjoint(outputs[i], bytes, normWeight.cudaData, 4096 * 4);
        for (int j = 0; j < i; ++j)
            eligible = eligible && disjoint(outputs[i], bytes, outputs[j], bytes);
    }
    if (withRouter) {
        eligible = eligible && tensor(*router, fastllm::FLOAT32, size_t(rows) * 256);
        if (scales) eligible = eligible && disjoint(router->cudaData, size_t(rows)*256*4, scales, 4096*1024/32*sizeof(float));
        for (const void *out : outputs)
            eligible = eligible && disjoint(out, bytes, routerWeight->cudaData, 256 * 4096 * 4) &&
                disjoint(out, bytes, router->cudaData, size_t(rows) * 256 * 4);
        for (const fastllm::Data *source : std::initializer_list<const fastllm::Data *>{
                &input, &weight, &normWeight, routerWeight})
            eligible = eligible && disjoint(router->cudaData, size_t(rows) * 256 * 4, source->cudaData, source->GetBytes());
    }
    if (prepared) {
        for (const fastllm::Data *x : std::initializer_list<const fastllm::Data *>{
                &input, &weight, &hidden, &normWeight, &projected, &normed, routerWeight, router})
            if (x) eligible = eligible && prepared->Disjoint(x->cudaData, x->GetBytes());
    }
    // Reuse the ordinary collective rendezvous and Graph pointer registration.
    // A local layout/shape miss rejects the operation on all ranks before writes.
    CustomArPointers registered{};
    if (!FindOrRegisterCustomArPointers(state, rank, projected.cudaData, int(elements),
            fastllm::BFLOAT16, eligible, registered)) return false;
    // Preserve the established eight-row GEMM algorithm and rounding. It is
    // prepared on the same borrowed weights during ordinary startup warmup.
    if (rows == 8 && !mma) {
        fastllm::Data bias;
        if (!FastllmCudaBFloat16MatMulBFloat16(input, weight, bias, projected, rows, 1024, 4096)) {
            FastllmCudaSetThreadError();
            throw std::runtime_error("attention output GEMM failed after collective admission");
        }
        if (capture == cudaStreamCaptureStatusNone) state.swaOutput->MarkRowsGemmPrepared(rank);
    }
    const cudaError_t status = mx
        ? state.swaOutput->LaunchMxOutput(input.cudaData, weight.cudaData, scales, projected.cudaData,
            hidden.cudaData, (const float *)normWeight.cudaData, normed.cudaData, eps,
            withRouter ? (const float *)routerWeight->cudaData : nullptr, withRouter ? (float *)router->cudaData : nullptr, rank)
        : rowKernel
        ? state.swaOutput->LaunchRows(input.cudaData, weight.cudaData,
            projected.cudaData, hidden.cudaData, (const float *)normWeight.cudaData,
            normed.cudaData, eps, withRouter ? (const float *)routerWeight->cudaData : nullptr,
            withRouter ? (float *)router->cudaData : nullptr, rank, rows, mma, prepared)
        : withRouter
        ? state.swaOutput->LaunchOutputRouter(input.cudaData, weight.cudaData,
            projected.cudaData, hidden.cudaData, (const float *)normWeight.cudaData,
            normed.cudaData, eps, (const float *)routerWeight->cudaData, (float *)router->cudaData, rank)
        : state.swaOutput->Launch(input.cudaData, weight.cudaData,
            projected.cudaData, hidden.cudaData, (const float *)normWeight.cudaData,
            normed.cudaData, eps, rank);
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        throw std::runtime_error(std::string("SWA fused output launch failed: ") + cudaGetErrorString(status));
    }
    return true;
}

bool FastllmCudaSwaLayerSupported(int deviceId) {
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease; int rank = -1;
    return lease.Acquire(state, 8192, fastllm::BFLOAT16, deviceId, true, rank) &&
        state.swaOutput && (state.swaOutput->LayerSupported() || state.swaOutput->LayerSupported(true));
}

bool FastllmCudaSwaLayer(const fastllm::Data &branch, fastllm::Data &hidden,
        const fastllm::Data &inputNorm, const fastllm::Data &qkvWeight,
        const fastllm::Data &qkvBias, const fastllm::Data &oWeight,
        const fastllm::Data &postNorm, const fastllm::Data &routerWeight,
        const fastllm::Data &sink, const fastllm::Data &position,
        const fastllm::Data &live, const fastllm::Data &key, const fastllm::Data &value,
        const void *const *cache, fastllm::Data &q, fastllm::Data &attn,
        fastllm::Data &projected, fastllm::Data &normed, fastllm::Data &router,
        FastllmSwaLayerConfig c, int deviceId) {
    using namespace fastllm;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    const bool mx = qkvWeight.dataType == FP8_E4M3;
    if (!lease.Acquire(state, 8192, BFLOAT16, deviceId, true, rank) ||
        !state.swaOutput) return false;
    auto located = [&](const Data &x, int type, size_t alignment = 16) {
        return x.dataType == type && x.dataDevice == DataDevice::CUDA &&
            x.dataDeviceIds.size() == 1 && x.dataDeviceIds.front() == deviceId && x.cudaData &&
            !(uintptr_t(x.cudaData) & (alignment - 1)) && !x.dims.empty() &&
            x.strides.size() == x.dims.size();
    };
    auto tensor = [&](const Data &x, int type, size_t count, size_t alignment = 16) {
        if (!located(x, type, alignment)) return false;
        size_t stride = 1;
        for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
            if (x.dims[i] <= 0 || x.strides[i] != stride || stride > count / x.dims[i]) return false;
            stride *= x.dims[i];
        }
        return stride == count;
    };
    auto ring = [&](const Data &x, int width) {
        return located(x, BFLOAT16) && !x.isPagedKVCache &&
            x.dims.size() == 3 && x.dims[0] == 1 && x.dims[2] == width &&
            x.strides[2] == 1 && x.strides[1] == width &&
            x.expansionDims.size() == 3 && x.expansionDims[1] >= 128;
    };
    const auto *qkvScales = mx ? SingleRowMxScales(qkvWeight,1856,4096) : nullptr;
    const auto *oScales = mx ? SingleRowMxScales(oWeight,4096,1024) : nullptr;
    const auto weightType = mx ? FP8_E4M3 : BFLOAT16;
    bool eligible = state.swaOutput->LayerSupported(mx) && (!mx || (qkvScales && oScales)) && c.heads == 8 && c.kvHeads == 1 && c.headDim == 192 &&
        c.valueDim == 128 && c.rotaryDim == 64 && c.window == 128 &&
        std::isfinite(c.eps) && c.eps > 0 && std::isfinite(c.theta) && c.theta > 0 &&
        std::isfinite(c.valueScale) && tensor(hidden, BFLOAT16, 4096) && hidden.dims.back() == 4096 &&
        tensor(branch, BFLOAT16, 4096) && branch.dims.back() == 4096 && tensor(inputNorm, FLOAT32, 4096) &&
        tensor(qkvWeight, weightType, 1856 * 4096) && qkvBias.dims.empty() &&
        qkvWeight.dims == std::vector<int>({1856,4096}) &&
        tensor(oWeight, weightType, 4096 * 1024) && oWeight.dims == std::vector<int>({4096,1024}) &&
        tensor(postNorm, FLOAT32, 4096) && tensor(routerWeight, FLOAT32, 256 * 4096) &&
        routerWeight.dims == std::vector<int>({256,4096}) &&
        (sink.dims.empty() || tensor(sink, FLOAT32, 8)) &&
        tensor(position, FLOAT32, 1, 4) && tensor(live, INT32, 1, 4) &&
        ring(key, 192) && ring(value, 128) && (!cache || !(uintptr_t(cache) & 7));
    Data *outputs[] = {&q, &attn, &projected, &normed, &router};
    const size_t counts[] = {1536,1024,4096,4096,256};
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < i; ++j) eligible = eligible && outputs[i] != outputs[j];
        for (const Data *source : {&branch, (const Data *)&hidden, &inputNorm, &qkvWeight, &qkvBias,
                &oWeight, &postNorm, &routerWeight, &sink, &position, &live, &key, &value})
            eligible = eligible && outputs[i] != source;
        if (outputs[i]->isFake || outputs[i]->cudaDataBorrowed)
            eligible = eligible && tensor(*outputs[i], i == 4 ? FLOAT32 : BFLOAT16, counts[i]);
    }
    if (eligible) for (int i = 0; i < 5; ++i) {
        auto &x = *outputs[i]; x.dataType = i == 4 ? FLOAT32 : BFLOAT16; x.UpdateUnitSize();
        x.Resize({1,1,int(counts[i])}); x.ToDevice(DataDevice::CUDA, {deviceId}, false); x.Allocate(false);
    }
    auto disjoint = [](const void *a, size_t na, const void *b, size_t nb) {
        uintptr_t x = uintptr_t(a), y = uintptr_t(b);
        return x <= y ? y - x >= na : x - y >= nb;
    };
    std::pair<const void *, size_t> writes[8] = {{hidden.cudaData,8192},
        {key.cudaData,128*192*2},{value.cudaData,128*128*2}};
    for (int i = 0; i < 5; ++i) {
        eligible = eligible && tensor(*outputs[i], i == 4 ? FLOAT32 : BFLOAT16, counts[i]);
        writes[i + 3] = {outputs[i]->cudaData, counts[i] * (i == 4 ? 4 : 2)};
    }
    const std::pair<const void *, size_t> reads[] = {
        {branch.cudaData,8192},{inputNorm.cudaData,16384},
        {qkvWeight.cudaData,qkvWeight.GetBytes()},{oWeight.cudaData,oWeight.GetBytes()},
        {qkvScales,qkvScales ? 1856u*4096/32*4 : 0u},{oScales,oScales ? 4096u*1024/32*4 : 0u},
        {postNorm.cudaData,16384},{routerWeight.cudaData,256*4096*4},
        {sink.cudaData,sink.dims.empty()?0u:32u},{position.cudaData,4},{live.cudaData,4},{cache,cache?32u:0u}};
    for (size_t i = 0; i < 8; ++i) {
        for (const auto &r : reads) if (r.second)
            eligible = eligible && disjoint(writes[i].first,writes[i].second,r.first,r.second);
        for (size_t j = 0; j < i; ++j)
            eligible = eligible && disjoint(writes[i].first,writes[i].second,writes[j].first,writes[j].second);
    }
    CustomArPointers registered{};
    if (!FindOrRegisterCustomArPointers(state,rank,projected.cudaData,4096,BFLOAT16,eligible,registered)) return false;
    fastllm_swa_layer::Args args{};
    args.qkvScales = qkvScales; args.oScales = oScales;
    args.branch = (const __nv_bfloat16 *)branch.cudaData;
    args.qkvWeight = (const __nv_bfloat16 *)qkvWeight.cudaData;
    args.oWeight = (const __nv_bfloat16 *)oWeight.cudaData;
    args.hidden = (__nv_bfloat16 *)hidden.cudaData; args.normed = (__nv_bfloat16 *)normed.cudaData;
    args.q = (__nv_bfloat16 *)q.cudaData; args.k = (__nv_bfloat16 *)key.cudaData;
    args.v = (__nv_bfloat16 *)value.cudaData; args.attn = (__nv_bfloat16 *)attn.cudaData;
    args.projected = (__nv_bfloat16 *)projected.cudaData;
    args.inputNorm = (const float *)inputNorm.cudaData; args.postNorm = (const float *)postNorm.cudaData;
    args.routerWeight = (const float *)routerWeight.cudaData; args.routerOut = (float *)router.cudaData;
    args.sink = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    args.position = (const float *)position.cudaData; args.live = (const int *)live.cudaData;
    args.cache = cache; args.eps = c.eps; args.theta = c.theta; args.valueScale = c.valueScale;
    const cudaError_t status = state.swaOutput->LaunchLayer(args,rank);
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        throw std::runtime_error(std::string("SWA layer launch failed: ") + cudaGetErrorString(status));
    }
    return true;
}

bool FastllmCudaSwaPreparedRouterSupported(int deviceId, bool routed, int rows) {
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease; int rank = -1;
    return lease.Acquire(state, 8192, fastllm::BFLOAT16, deviceId, true, rank) &&
        state.swaOutput && state.swaOutput->PreparedRouterSupported(routed, rows);
}

bool FastllmCudaSwaVerifyLayerSupported(int deviceId, int rows) {
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease; int rank = -1;
    return lease.Acquire(state, 8192, fastllm::BFLOAT16, deviceId, true, rank) &&
        state.swaOutput && state.swaOutput->VerifySupported(rows);
}

static bool RunSwaVerifyLayer(const fastllm::Data &branch, fastllm::Data &hidden,
        const fastllm::Data &inputNorm, const fastllm::Data &qkvWeight,
        const fastllm::Data &qkvBias, const fastllm::Data &oWeight,
        const fastllm::Data &postNorm, const fastllm::Data &sink,
        const fastllm::Data &position, const fastllm::Data &live,
        const std::vector<FastllmNaiveRopeCacheRow> &caches,
        fastllm::Data &q, fastllm::Data &attn, fastllm::Data &projected,
        fastllm::Data &normed, fastllm::Data *feature,
        const fastllm::Data *routerWeight, fastllm::Data *router,
        FastllmSwaLayerConfig c, int deviceId,
        const FastllmMoePreparedInput *prepared, bool afterExpertReduce) {
    using namespace fastllm;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease; int rank = -1;
    // Only group-wide state may reject before the collective rendezvous.
    if (!lease.Acquire(state, 8192, BFLOAT16, deviceId, true, rank) || !state.swaOutput)
        return false;
    const int rows = int(caches.size());
    const size_t elements = rows > 0 && rows <= fastllm_swa_verify::RowLimit ? size_t(rows) * 4096 : 0;
    auto located = [&](const Data &x, int type, size_t alignment = 16) {
        return x.dataType == type && x.dataDevice == DataDevice::CUDA &&
            x.dataDeviceIds == std::vector<int>{deviceId} && x.cudaData &&
            !(uintptr_t(x.cudaData) & (alignment - 1)) && !x.dims.empty() &&
            x.strides.size() == x.dims.size();
    };
    auto tensor = [&](const Data &x, int type, size_t count, size_t alignment = 16) {
        if (!located(x, type, alignment) || !count) return false;
        size_t stride = 1;
        for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
            if (x.dims[i] <= 0 || x.strides[i] != stride || stride > count / size_t(x.dims[i]))
                return false;
            stride *= x.dims[i];
        }
        return stride == count;
    };
    auto ring = [&](const Data &x, int width) {
        const int capacity = x.expansionDims.size() == 3 ? x.expansionDims[1]
            : x.dims.size() == 3 ? x.dims[1] : 0;
        return located(x, BFLOAT16) && !x.isPagedKVCache && x.dims.size() == 3 &&
            x.dims[0] == 1 && x.dims[2] == width && capacity >= 136 &&
            x.strides[2] == 1 && x.strides[1] == size_t(width);
    };
    struct Range { const void *data; size_t bytes; };
    auto disjoint = [](Range a, Range b) {
        if (!a.bytes || !b.bytes) return true;
        const uintptr_t x = uintptr_t(a.data), y = uintptr_t(b.data);
        return x <= y ? y - x >= a.bytes : x - y >= b.bytes;
    };
    const bool withRouter = routerWeight && router;
    const bool mx = qkvWeight.dataType == FP8_E4M3;
    const auto *qkvScales = mx ? FastllmCudaVerifyMxfp8Scales(qkvWeight,1856,4096,deviceId) : nullptr;
    const auto *outputScales = mx ? FastllmCudaVerifyMxfp8Scales(oWeight,4096,1024,deviceId) : nullptr;
    const auto weightType = mx ? FP8_E4M3 : BFLOAT16;
    bool eligible = state.swaOutput->VerifySupported(rows, mx) &&
        (!mx || (qkvScales && outputScales)) &&
        (!prepared || (withRouter && state.swaOutput->PreparedRouterSupported(prepared->routed, rows) && prepared->Valid(rows))) &&
        c.heads == 8 && c.kvHeads == 1 && c.headDim == 192 && c.valueDim == 128 &&
        c.rotaryDim == 64 && c.window == 128 && std::isfinite(c.eps) && c.eps > 0 &&
        std::isfinite(c.theta) && c.theta > 0 && std::isfinite(c.valueScale) &&
        tensor(hidden, BFLOAT16, elements) && hidden.dims.back() == 4096 &&
        tensor(branch, BFLOAT16, elements) && branch.dims == hidden.dims &&
        tensor(inputNorm, FLOAT32, 4096) && tensor(postNorm, FLOAT32, 4096) &&
        tensor(qkvWeight, weightType, 1856 * 4096) && qkvBias.dims.empty() &&
        qkvWeight.dims == std::vector<int>({1856, 4096}) &&
        tensor(oWeight, weightType, 4096 * 1024) && oWeight.dims == std::vector<int>({4096, 1024}) &&
        (sink.dims.empty() || tensor(sink, FLOAT32, 8, 4)) &&
        tensor(position, FLOAT32, rows, 4) && tensor(live, INT32, rows, 4) &&
        bool(routerWeight) == bool(router) &&
        (!feature || (tensor(*feature, BFLOAT16, elements) && feature->dims == hidden.dims)) &&
        (!withRouter || ((rows == 1 || rows < FastllmCudaGetLinearExactBatchThreshold()) &&
            tensor(*routerWeight, FLOAT32, 256 * 4096) &&
            routerWeight->dims == std::vector<int>({256, 4096})));
    std::vector<const Data *> sources{&branch, &hidden, &inputNorm, &qkvWeight, &qkvBias,
        &oWeight, &postNorm, &sink, &position, &live};
    if (routerWeight) sources.push_back(routerWeight);
    if (feature) sources.push_back(feature);
    std::vector<Data *> outputs{&q, &attn, &projected, &normed};
    std::vector<int> widths{1536, 1024, 4096, 4096};
    if (withRouter) { outputs.push_back(router); widths.push_back(256); }
    fastllm_swa_verify::Args args{};
    std::vector<Range> keys, values;
    std::vector<Range> reads{
        {branch.cudaData, elements * 2}, {inputNorm.cudaData, 4096 * 4},
        {qkvWeight.cudaData, qkvWeight.GetBytes()}, {oWeight.cudaData, oWeight.GetBytes()},
        {postNorm.cudaData, 4096 * 4}, {sink.cudaData, sink.dims.empty() ? 0u : 32u},
        {position.cudaData, size_t(std::max(0, rows)) * 4},
        {live.cudaData, size_t(std::max(0, rows)) * 4}};
    if (mx) {
        reads.push_back({qkvScales,1856 * 4096 / 32 * sizeof(float)});
        reads.push_back({outputScales,4096 * 1024 / 32 * sizeof(float)});
    }
    if (withRouter) reads.push_back({routerWeight->cudaData, 256 * 4096 * 4});
    if (eligible) for (int i = 0; i < rows; ++i) {
        const auto &r = caches[i];
        if (!r.key || !r.value || !ring(*r.key, 192) || !ring(*r.value, 128) ||
            (r.indexKey && !r.indexKey->dims.empty()) || r.indexCache || r.indexerContextRank ||
            r.cacheOffset >= 0 || r.verifyOffset < 0 || r.verifyOffset >= 8 ||
            (r.cache && uintptr_t(r.cache) % alignof(void *))) { eligible = false; break; }
        sources.push_back(r.key); sources.push_back(r.value);
        keys.push_back({r.key->cudaData, 136 * 192 * 2});
        values.push_back({r.value->cudaData, 136 * 128 * 2});
        if (r.cache) reads.push_back({r.cache, 2 * sizeof(void *)});
        args.bindings.rows[i] = {(fastllm_swa_output::BF16 *)r.key->cudaData,
            (fastllm_swa_output::BF16 *)r.value->cudaData, r.cache, r.verifyOffset};
        // Request-owned tables may rebind at replay. Shared host views must
        // have distinct candidate slots; partial or inconsistent aliases reject.
        for (int j = 0; j < i; ++j) {
            const bool sameKey = keys[i].data == keys[j].data;
            const bool sameValue = values[i].data == values[j].data;
            if ((!disjoint(keys[i], keys[j]) && !sameKey) ||
                (!disjoint(values[i], values[j]) && !sameValue) ||
                sameKey != sameValue ||
                ((sameKey || (r.cache && r.cache == caches[j].cache)) &&
                 r.verifyOffset == caches[j].verifyOffset)) eligible = false;
        }
    }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    const bool captureOk = cudaStreamIsCapturing(cudaStreamPerThread, &capturing) == cudaSuccess;
    if (!captureOk) cudaGetLastError();
    eligible = eligible && captureOk;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const int type = i == 4 ? FLOAT32 : BFLOAT16;
        for (const Data *source : sources) eligible = eligible && outputs[i] != source;
        for (size_t j = 0; j < i; ++j) eligible = eligible && outputs[i] != outputs[j];
        if (outputs[i]->isFake || outputs[i]->cudaDataBorrowed || capturing != cudaStreamCaptureStatusNone)
            eligible = eligible && tensor(*outputs[i], type, size_t(rows) * widths[i]);
    }
    if (eligible) for (size_t i = 0; i < outputs.size(); ++i) {
        auto &out = *outputs[i]; out.dataType = i == 4 ? FLOAT32 : BFLOAT16; out.UpdateUnitSize();
        auto dims = hidden.dims; dims.back() = widths[i]; out.Resize(dims);
        out.ToDevice(DataDevice::CUDA, {deviceId}, false); out.Allocate(false);
    }
    std::vector<Range> writes{{hidden.cudaData, elements * 2}};
    for (size_t i = 0; i < outputs.size(); ++i) {
        eligible = eligible && tensor(*outputs[i], i == 4 ? FLOAT32 : BFLOAT16, size_t(rows) * widths[i]);
        writes.push_back({outputs[i]->cudaData, size_t(std::max(0, rows)) * widths[i] * (i == 4 ? 4u : 2u)});
    }
    if (feature) writes.push_back({feature->cudaData, elements * 2});
    for (size_t i = 0; i < writes.size(); ++i) {
        for (size_t j = 0; j < i; ++j) eligible = eligible && disjoint(writes[i], writes[j]);
        for (auto r : reads) eligible = eligible && disjoint(writes[i], r);
        for (auto k : keys) eligible = eligible && disjoint(writes[i], k);
        for (auto v : values) eligible = eligible && disjoint(writes[i], v);
    }
    for (auto k : keys) {
        for (auto r : reads) eligible = eligible && disjoint(k, r);
        for (auto v : values) eligible = eligible && disjoint(k, v);
    }
    for (auto v : values) for (auto r : reads) eligible = eligible && disjoint(v, r);
    if (prepared) {
        for (const auto *ranges : {&reads, &writes, &keys, &values})
            for (auto range : *ranges) eligible = eligible && prepared->Disjoint(range.data, range.bytes);
    }
    CustomArPointers registered{};
    if (!FindOrRegisterCustomArPointers(state, rank, projected.cudaData, int(elements),
            BFLOAT16, eligible, registered)) return false;
    using BF16Value = fastllm_swa_output::BF16;
    args.outputScales = outputScales;
    args.branch = (const BF16Value *)branch.cudaData;
    args.weight = (const BF16Value *)qkvWeight.cudaData;
    args.oWeight = (const BF16Value *)oWeight.cudaData;
    args.hidden = (BF16Value *)hidden.cudaData; args.q = (BF16Value *)q.cudaData;
    args.attn = (BF16Value *)attn.cudaData; args.projected = (BF16Value *)projected.cudaData;
    args.normed = (BF16Value *)normed.cudaData;
    args.feature = feature ? (BF16Value *)feature->cudaData : nullptr;
    args.gamma = (const float *)inputNorm.cudaData;
    args.positions = (const float *)position.cudaData;
    args.bindings.length = (const int *)live.cudaData;
    args.sink = sink.dims.empty() ? nullptr : (const float *)sink.cudaData;
    args.rows = rows; args.eps = c.eps; args.theta = c.theta; args.scale = c.valueScale;
    if (mx) {
        // Preserve native QKV rounding before attention and O re-quantize
        // its result. Borrow the larger projected buffer until the fused tail.
        if (!FastllmCudaMxfp8SwaQkv(hidden.cudaData, branch.cudaData, args.gamma, c.eps,
                                    qkvWeight, qkvScales, projected.cudaData, rows, args.positions, c.theta, normed.cudaData,
                                    afterExpertReduce)) {
            Data packed(BFLOAT16);
            auto dims = hidden.dims;
            dims.back() = 1856;
            packed.Resize(dims);
            packed.FakeFrom(projected, 0);
            FastllmCudaNaiveAddDecodeRMSNorm(hidden, branch, inputNorm, c.eps, normed);
            if (!FastllmCudaBFloat16MatMulFP8E4M3(normed, const_cast<Data &>(qkvWeight), qkvBias,
                                                   packed, rows, 4096, 1856))
                throw std::runtime_error("SWA MXFP8 QKV projection failed after collective admission");
        } else {
            args.rotary = static_cast<const __nv_bfloat162 *>(normed.cudaData);
        }
        args.branch = nullptr;
    }
    const cudaError_t status = state.swaOutput->LaunchVerify(args, rank, (const float *)postNorm.cudaData,
        withRouter ? (const float *)routerWeight->cudaData : nullptr,
        withRouter ? (float *)router->cudaData : nullptr, prepared);
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        throw std::runtime_error(std::string("SWA verification layer launch failed: ") + cudaGetErrorString(status));
    }
    return true;
}

static bool LocalDsaVerifyAvailable() {
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    return FastllmCudaRuntimeArch() == 103;
#else
    return false;
#endif
}

bool FastllmCudaLocalDsaLayerSupported(const std::vector<int> &requests, int rows, int capacity,
                                      int heads, int dim, int topK) {
    return LocalDsaVerifyAvailable() && rows <= 16 && capacity > 8192 && capacity <= 65536 &&
        heads == 16 && dim == 128 && topK == 2048 &&
        std::all_of(requests.begin(), requests.end(), [](int n) { return n >= 4 && n <= 8; });
}

bool FastllmCudaLocalDsaIndexerSupported(const std::vector<FastllmNaiveVerifyIndexerItem> &items,
                                        int heads, int kvHeads, int dim, int valueDim) {
    return LocalDsaVerifyAvailable() && !items.empty() && heads == 8 && kvHeads == 1 && dim == 192 &&
        valueDim == 128 && std::all_of(items.begin(), items.end(), [](const auto &item) {
            return item.rows >= 4 && item.rows <= 8 && (item.capacity ? item.capacity : item.past + item.rows) > 8192;
        });
}

bool FastllmCudaVerifyLocalDsaIndexer(const fastllm::Data &query, const fastllm::Data &indexQuery,
                                    const fastllm::Data &weights, const fastllm::Data &sink,
                                    const std::vector<FastllmNaiveVerifyIndexerItem> &items,
                                    const std::vector<FastllmNaiveDSARow> &rows, bool fp8Query, int heads,
                                    int kvHeads, int dim, int valueDim, FastllmNaiveDecodeScratch &scratch,
                                    fastllm::Data &output) {
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    return FastllmCudaNaiveFusedDSAVerify(query, indexQuery, weights, sink, items, rows, fp8Query,
                                          heads, kvHeads, dim, valueDim, scratch, output);
#else
    return false;
#endif
}

bool FastllmCudaPackedVerifySupported(int rows, int ranks) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    return rows == 8 && ranks == 8 && FastllmCudaRuntimeArch() == 120;
#else
    return false;
#endif
}

// A replicated attention rank owns the entire output projection. The caller still
// performs its ordinary residual and norm after any required DP row exchange.
static bool RunReplicaVerifyAttention(const FastllmNaiveVerifyAttentionInput &x) {
    const auto &c = x.config;
    if (x.branch || !x.projectionScratch || !x.qkv || x.requests.size() != 1 ||
        x.requests.front() < 2 || x.requests.front() > 8 || x.caches.size() != x.requests.front() ||
        c.heads != 64 || c.headDim != 192 || c.valueDim != 128 || c.rotaryDim != 64) return false;
    const auto &cache = x.caches.front();
    if (!cache.key || !cache.value || !cache.cache) return false;
    if (x.sliding) {
        if (c.kvHeads != 8 || c.window != 128) return false;
    } else {
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
        if (c.kvHeads != 4 || x.indexHeads != 16 || x.indexDim != 128 || x.topK != 2048 ||
            !x.fp8Query || x.fp8IndexerCache || x.capacity <= 8192 || x.capacity > 65536) return false;
#else
        return false;
#endif
    }
    const auto *projection = FastllmCudaGetMxfp8ProjectionBackend();
    if (!projection) return false;
    const int rows = x.requests.front();
    auto packed = projection->Borrow(*x.outputWeight, rows, *x.projectionScratch);
    if (!packed.Matches(rows, c.heads * c.valueDim)) return false;
    if (!projection->TryProject(x.qkvActivation, *x.normed, *x.qkvWeight, *x.qkvBias, *x.qkv, true))
        fastllm::Linear(*x.normed, *x.qkvWeight, *x.qkvBias, *x.qkv);
    bool ready = false;
    if (x.sliding)
        ready = FastllmCudaTryPackedSwaVerify(*x.qkv, *x.positions, *x.live, cache, *x.sink,
            c.heads, c.kvHeads, x.theta, x.valueScale, packed, *x.attention, projection->projectPdl != nullptr);
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    else
        ready = FastllmCudaTryDpDSAVerify(*x.normed, *x.qkv, *x.indexKeyWeight, *x.indexQueryWeight,
            *x.indexWeight, *x.indexNorm, *x.indexBias, *x.sink, *x.positions, *x.live,
            *cache.key, *cache.value, cache.cache, x.capacity, x.theta, x.valueScale,
            packed, *x.scratch, *x.attention, true);
#endif
    if (!ready) return false;
    if (!projection->TryProject(packed, *x.attention, *x.outputWeight, fastllm::Data(), *x.projected, true))
        fastllm::Linear(*x.attention, *x.outputWeight, fastllm::Data(), *x.projected);
    return true;
}

FastllmVerifyOutput FastllmCudaVerifyAttention(const FastllmNaiveVerifyAttentionInput &x, int device,
                                               const FastllmMoePreparedInput *prepared) {
    using Result = FastllmVerifyOutput;
    const auto &cfg = x.config;
    if (x.replica) return RunReplicaVerifyAttention(x) ? Result::Projected : Result::Unsupported;
    const bool verifyShape = cfg.heads == 8 && cfg.kvHeads == 1 && cfg.headDim == 192 &&
                             cfg.valueDim == 128 && cfg.rotaryDim == 64;
    if (!x.sliding && !verifyShape) return Result::Unsupported;
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    if (FastllmCudaRuntimeArch() == 120) {
        if (!x.packedTp || !verifyShape || x.requests.size() != 1 || x.requests.front() != 8 || x.caches.size() != 8 ||
            x.hidden->dims.empty() || x.hidden->dims.back() != 4096 ||
            (!x.sliding && (!x.indices || x.branch || x.indexHeads != 16 || x.indexDim != 128 ||
                            x.topK != 2048 || x.capacity <= x.topK)) || (x.sliding && cfg.window != 128))
            return Result::Unsupported;
        std::vector<FastllmNaiveDSARow> bindings(8);
        for (int i = 0; i < 8; ++i) {
            const auto &cache = x.caches[i];
            const auto &k = *cache.key, &v = *cache.value;
            if (k.isPagedKVCache != v.isPagedKVCache || k.dataType != fastllm::FP8_E4M3 ||
                v.dataType != k.dataType || k.dims.size() != 3 || v.dims.size() != 3 || v.dims.back() != 128 ||
                (k.isPagedKVCache && (k.pageLen != v.pageLen || !k.pagedKVCacheData || !v.pagedKVCacheData)) ||
                (x.sliding && (k.isPagedKVCache || k.expansionDims.size() != 3 || v.expansionDims.size() != 3 ||
                               std::min(k.expansionDims[1], v.expansionDims[1]) < 136)))
                continue; // All ranks still enter collective admission with their local eligibility.
            auto &row = bindings[i];
            row.key = k.isPagedKVCache ? k.pagedKVCacheData->cudaData : k.cudaData;
            row.value = v.isPagedKVCache ? v.pagedKVCacheData->cudaData : v.cudaData;
            row.pageLen = k.isPagedKVCache ? k.pageLen : 0;
            row.keyStride = k.dims.back();
            row.fp8 = true;
            row.capacity = x.capacity;
            row.cache = cache.cache;
            row.verifyOffset = x.sliding ? i : -1;
            row.liveKeys = static_cast<const int *>(x.live->cudaData) + i;
        }
        FastllmNaiveVerifyIndexerSm120 indexer{x.indexQueryWeight, x.indexKeyWeight, x.indexWeight,
            x.indexNorm, x.indexBias, x.indices, x.attentionRank, x.fp8IndexerCache};
        const bool deferNorm = x.sliding && x.deferPostNorm;
        if (!RunPackedVerifyAttention(*x.normed, *x.qkvWeight, *x.qkvBias, *x.outputWeight,
                *x.postNorm, *x.sink, *x.positions, *x.query, *x.attention, *x.projected, *x.hidden,
                *x.normed, x.scratch->attentionPartial, bindings, x.sliding ? 128 : std::min(x.topK, x.capacity),
                x.sliding, x.theta, x.valueScale, x.eps, device, x.sliding ? nullptr : &indexer,
                x.qkvActivation, x.branch, x.branch ? x.inputNorm : nullptr, x.feature, deferNorm))
            return Result::Unsupported;
        return deferNorm ? Result::DeferredNorm : Result::Normalized;
    }
#endif
    bool launched = false;
    if (x.sliding && x.branch) {
        launched = RunSwaVerifyLayer(*x.branch, *x.hidden, *x.inputNorm, *x.qkvWeight, *x.qkvBias,
                *x.outputWeight, *x.postNorm, *x.sink, *x.positions, *x.live, x.caches, *x.query,
                *x.attention, *x.projected, *x.normed, x.feature, x.routerWeight, x.router,
                x.config, device, prepared, x.afterExpertReduce);
    }
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    else if (!x.sliding)
        launched = FastllmCudaNaiveDSALayer(x, device, prepared, x.deferPostNorm);
#endif
    if (!launched) return Result::Unsupported;
    return x.router ? Result::Router : Result::Normalized;
}

bool FastllmCudaSwaVerifyLayer(const fastllm::Data &branch, fastllm::Data &hidden,
        const fastllm::Data &inputNorm, const fastllm::Data &qkvWeight,
        const fastllm::Data &qkvBias, const fastllm::Data &oWeight,
        const fastllm::Data &postNorm, const fastllm::Data &sink,
        const fastllm::Data &position, const fastllm::Data &live,
        const std::vector<FastllmNaiveRopeCacheRow> &caches,
        fastllm::Data &q, fastllm::Data &attn, fastllm::Data &projected,
        fastllm::Data &normed, fastllm::Data *feature,
        const fastllm::Data *routerWeight, fastllm::Data *router,
        FastllmSwaLayerConfig c, int deviceId) {
    return RunSwaVerifyLayer(branch, hidden, inputNorm, qkvWeight, qkvBias,
        oWeight, postNorm, sink, position, live, caches, q, attn, projected, normed,
        feature, routerWeight, router, c, deviceId, nullptr, false);
}

bool FastllmCudaSwaVerifyLayerPrepared(const fastllm::Data &branch, fastllm::Data &hidden,
        const fastllm::Data &inputNorm, const fastllm::Data &qkvWeight,
        const fastllm::Data &qkvBias, const fastllm::Data &oWeight,
        const fastllm::Data &postNorm, const fastllm::Data &sink,
        const fastllm::Data &position, const fastllm::Data &live,
        const std::vector<FastllmNaiveRopeCacheRow> &caches,
        fastllm::Data &q, fastllm::Data &attn, fastllm::Data &projected,
        fastllm::Data &normed, fastllm::Data *feature,
        const fastllm::Data *routerWeight, fastllm::Data *router,
        FastllmSwaLayerConfig c, int deviceId,
        const FastllmMoePreparedInput *prepared) {
    return RunSwaVerifyLayer(branch, hidden, inputNorm, qkvWeight, qkvBias, oWeight,
        postNorm, sink, position, live, caches, q, attn, projected, normed,
        feature, routerWeight, router, c, deviceId, prepared, false);
}

bool FastllmCudaSwaVerifyLayerAfterExpertReduce(const fastllm::Data &branch, fastllm::Data &hidden,
        const fastllm::Data &inputNorm, const fastllm::Data &qkvWeight,
        const fastllm::Data &qkvBias, const fastllm::Data &oWeight,
        const fastllm::Data &postNorm, const fastllm::Data &sink,
        const fastllm::Data &position, const fastllm::Data &live,
        const std::vector<FastllmNaiveRopeCacheRow> &caches,
        fastllm::Data &q, fastllm::Data &attn, fastllm::Data &projected,
        fastllm::Data &normed, fastllm::Data *feature,
        const fastllm::Data *routerWeight, fastllm::Data *router,
        FastllmSwaLayerConfig c, int deviceId,
        const FastllmMoePreparedInput *prepared) {
    return RunSwaVerifyLayer(branch, hidden, inputNorm, qkvWeight, qkvBias, oWeight,
        postNorm, sink, position, live, caches, q, attn, projected, normed,
        feature, routerWeight, router, c, deviceId, prepared, true);
}

#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input, int deviceId,
                              const FastllmMoePreparedInput *prepared) {
    return FastllmCudaNaiveDSALayer(input,deviceId,prepared,false);
}
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input,int deviceId,
                              const FastllmMoePreparedInput *prepared,bool deferPostNorm) {
    using namespace fastllm;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, 8192, BFLOAT16, deviceId, true, rank) || !state.swaOutput)
        return false;
    const int rows = input.caches.size();
    const bool withRouter = input.routerWeight && input.router;
    FastllmNaiveDSALayerArgs args{};
    const Data *const routerSources[] = {
        input.branch,
        input.inputNorm,
        input.postNorm,
        input.qkvWeight,
        input.indexKeyWeight,
        input.indexQueryWeight,
        input.indexWeight,
        input.indexNorm,
        input.indexBias,
        input.outputWeight,
        input.sink,
        input.positions,
        input.live,
        input.routerWeight,
        input.hidden,
        input.feature,
        input.query,
        input.attention,
        input.projected,
        input.normed,
        input.scratch ? &input.scratch->dsaLayer : nullptr,
        input.scratch ? &input.scratch->dsaFused : nullptr,
    };
    bool routerDisjoint = true;
    if (input.router) {
        for (const Data *d : routerSources)
            routerDisjoint = routerDisjoint && input.router != d;
        for (const auto &cache : input.caches)
            routerDisjoint = routerDisjoint && input.router != cache.key && input.router != cache.value;
    }
    bool eligible = routerDisjoint && FastllmCudaNaiveDSALayerPrepare(input, deviceId, args) &&
                    bool(input.routerWeight) == bool(input.router) &&
                    (!prepared || (withRouter && state.swaOutput->PreparedRouterSupported(prepared->routed, rows) &&
                                   prepared->Valid(rows)));
    if (withRouter) {
        const auto &w = *input.routerWeight;
        eligible = eligible && (rows == 1 || rows < FastllmCudaGetLinearExactBatchThreshold()) &&
                   w.dataType == FLOAT32 && w.dataDevice == CUDA && w.cudaData &&
                   w.dataDeviceIds == std::vector<int>{deviceId} && w.dims == std::vector<int>({256, 4096}) &&
                   uintptr_t(w.cudaData) % 16 == 0;
        if (eligible) {
            auto &out = *input.router;
            if ((FastllmCudaGraphIsCapturingFast() || out.isFake || out.cudaDataBorrowed) &&
                (!out.cudaData || out.dataType != FLOAT32 || out.Count(0) != size_t(rows) * 256))
                eligible = false;
            if (eligible) {
                out.dataType = FLOAT32;
                out.UpdateUnitSize();
                auto dims = input.hidden->dims;
                dims.back() = 256;
                out.Resize(dims);
                out.ToDevice(CUDA, {deviceId}, false);
                out.Allocate(false);
            }
        }
    }
    if (eligible && withRouter) {
        const uintptr_t start = uintptr_t(input.router->cudaData);
        const size_t bytes = size_t(rows) * 256 * sizeof(float);
        auto disjoint = [&](const Data *d) {
            if (!d || !d->cudaData || d->dims.empty())
                return true;
            const uintptr_t other = uintptr_t(d->cudaData);
            // Paged cache cudaData is an int page table, not the logical KV
            // tensor described by dims/strides. The physical pools are checked
            // separately below. Keep real table aliases on the fallback path.
            if (d->isPagedKVCache && d->pageLen <= 0) return false;
            const size_t count = d->isPagedKVCache
                ? (size_t(input.capacity) + d->pageLen - 1) / d->pageLen * sizeof(int)
                : d->Count(0) * d->unitSize;
            return start <= other ? other - start >= bytes : start - other >= count;
        };
        eligible = !(start % 16);
        for (const Data *d : routerSources)
            eligible = eligible && disjoint(d);
        for (const auto &c : input.caches) {
            eligible = eligible && disjoint(c.key) && disjoint(c.value);
            if (c.key->isPagedKVCache)
                eligible =
                    eligible && disjoint(c.key->pagedKVCacheData) && disjoint(c.value->pagedKVCacheData);
        }
    }
    if (eligible && prepared) {
        for (Data *d : {input.hidden, input.feature, input.query, input.attention, input.projected,
                        input.normed, input.router, &input.scratch->dsaLayer, &input.scratch->dsaFused})
            if (d)
                eligible = eligible && prepared->Disjoint(d->cudaData, d->Count(0) * d->unitSize);
    }
    // Every rank participates even when a local weight/cache/layout rejects.
    // The all-rank decision is complete before the fused writer is launched.
    CustomArPointers registered{};
    if (!FindOrRegisterCustomArPointers(state, rank, input.projected ? input.projected->cudaData : nullptr,
                                        rows >= 1 && rows <= 16 ? rows * 4096 : 0, BFLOAT16, eligible,
                                        registered))
        return false;
    auto status =
        state.swaOutput->LaunchDsa(args, rank, (const float *)input.postNorm->cudaData,
                                   withRouter ? (const float *)input.routerWeight->cudaData : nullptr,
                                   withRouter ? (float *)input.router->cudaData : nullptr, prepared, deferPostNorm && !withRouter);
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        throw std::runtime_error(std::string("DSA layer launch failed: ") + cudaGetErrorString(status));
    }
    return true;
}
#endif

static bool ReduceExpertRows(void *partials, const int32_t *ids, void *output,
        int rows, int hidden, int topk, int deviceId, bool norm, void *residual, const float *weight, void *normed, float eps,
        const FastllmMxfp8Activation &next, void *feature, int featureStride) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    constexpr int threads = 256;
    if (rows <= 0 || hidden <= 0 || hidden % 8 || topk <= 0 || topk > 16 ||
        (int64_t)rows * hidden > 8 * threads * kCustomArMaxBlocks) return false;
    const int packets = rows * hidden / 8, blocks = norm ? rows : (packets + threads - 1) / threads;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, size_t(hidden) * 2, fastllm::BFLOAT16, deviceId, true, rank)) return false;
    CustomArPointers peers{};
    const size_t outputBytes = size_t(packets) * sizeof(CustomArNormPacketType);
    const size_t partialBytes = outputBytes * topk, idBytes = size_t(rows) * topk * sizeof(*ids);
    const bool normEligible = !norm || (hidden == 4096 && rows == 8 && topk == 8 && state.devices.size() == 8 &&
        residual && weight && normed && std::isfinite(eps) && eps > 0 &&
        ((uintptr_t(residual) | uintptr_t(normed)) & 15) == 0 &&
        CustomArBuffersDisjoint(output, outputBytes, residual, outputBytes) &&
        CustomArBuffersDisjoint(output, outputBytes, normed, outputBytes) &&
        CustomArBuffersDisjoint(residual, outputBytes, normed, outputBytes) &&
        CustomArBuffersDisjoint(partials, partialBytes, residual, outputBytes) &&
        CustomArBuffersDisjoint(partials, partialBytes, normed, outputBytes));
    const bool eligible = normEligible && partials && ids && output &&
        ((uintptr_t(partials) | uintptr_t(output)) & 15) == 0 && uintptr_t(ids) % alignof(int32_t) == 0 &&
        CustomArBuffersDisjoint(partials, partialBytes, output, outputBytes) &&
        CustomArBuffersDisjoint(partials, partialBytes, ids, idBytes) &&
        CustomArBuffersDisjoint(output, outputBytes, ids, idBytes);
    if (!FindOrRegisterCustomArPointers(state, rank, partials, rows * hidden, fastllm::BFLOAT16,
                                        eligible, peers)) return false;
    auto launch = [&](auto ranks) {
        CustomArExpertRowsKernel<decltype(ranks)::value><<<blocks, threads, 0, cudaStreamPerThread>>>(
            peers, state.allSignals, state.allSignals.signals[rank],
            static_cast<CustomArNormPacketType *>(partials), ids,
            static_cast<CustomArNormBF16 *>(output), rank, packets, hidden / 8, topk);
    };
    if (state.expertRowsPush && hidden == 4096 && rows == 8 && topk == 8) {
        const bool early = fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::VerifyExperts);
        auto kernel = early ? CustomArExpertRowsPeerPush<true> : CustomArExpertRowsPeerPush<false>;
        kernel<<<rows, 512, 0, cudaStreamPerThread>>>(state.expertBoxes, state.allSignals,
            static_cast<CustomArNormPacketType *>(partials), ids, static_cast<CustomArNormBF16 *>(output),
            rank, static_cast<CustomArNormBF16 *>(residual), weight, static_cast<CustomArNormBF16 *>(normed),
            eps, next, static_cast<CustomArNormBF16 *>(feature), featureStride);
        if (!norm || next.values) fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::VerifyExpertReduce);
    } else if (norm) {
        CustomArExpertRowsKernel<8, true><<<blocks, threads, 0, cudaStreamPerThread>>>(
            peers, state.allSignals, state.allSignals.signals[rank], static_cast<CustomArNormPacketType *>(partials),
            ids, static_cast<CustomArNormBF16 *>(output), rank, packets, hidden / 8, topk,
            static_cast<CustomArNormBF16 *>(residual), weight, static_cast<CustomArNormBF16 *>(normed), eps, next,
            static_cast<CustomArNormBF16 *>(feature), featureStride);
    } else switch (state.devices.size()) {
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    // The caller checks the launch error before consuming output. A single rank cannot retry a collective.
    if (cudaPeekAtLastError() != cudaSuccess) FastllmCudaSetThreadError();
    return true;
#else
    return false;
#endif
}

bool FastllmCudaCustomAllReduceExpertRows(void *partials, const int32_t *ids, void *output,
        int rows, int hidden, int topk, int deviceId) {
    return ReduceExpertRows(partials, ids, output, rows, hidden, topk, deviceId, false, nullptr, nullptr, nullptr, 0, {}, nullptr, 0);
}
bool FastllmCudaCustomAllReduceExpertRowsNorm(void *partials, const int32_t *ids, void *output,
        int rows, int hidden, int topk, int deviceId, void *residual, const float *weight, void *normed, float eps,
        const FastllmMxfp8Activation &next, void *feature, int featureStride) {
    return ReduceExpertRows(partials, ids, output, rows, hidden, topk, deviceId, true, residual, weight, normed, eps, next, feature, featureStride);
}

static bool CustomReduceAddNorm(void *data, void *residual,
        const float *weight, void *output, int count, int channels,
        float eps, int dataType, int deviceId, bool draft) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    if (dataType != fastllm::BFLOAT16 || channels != 4096 || count <= 0 ||
        count % channels || (size_t)count * 2 > CustomArMaxBytes() ||
        !std::isfinite(eps) || eps <= 0) return false;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, channels * 2, dataType, deviceId, true, rank)) return false;
    // Fold pointer/architecture eligibility into the collective rendezvous:
    // a single unsupported rank must send every peer through the fallback.
    const bool aligned = data && residual && weight && output &&
        (((uintptr_t)data | (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)output) & 15u) == 0;
    auto disjoint = [count](const void *a, const void *b) {
        uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
        return (x > y ? x - y : y - x) >= (size_t)count * 2;
    };
    const int arch = FastllmCudaRuntimeArch();
    // The direct one-CTA-per-row kernel uses only portable packet loads and warp reductions.
    const int rows = count / channels;
    const bool eligible = aligned && ((rows <= 8 && arch >= 70) || arch == 90 || arch == 103) &&
        disjoint(data, residual) && disjoint(data, output) && disjoint(residual, output);
    CustomArPointers pointers{};
    if (!FindOrRegisterCustomArPointers(state, rank, data, count, dataType,
                                      eligible, pointers)) return false;
    auto *hidden = (CustomArNormBF16 *)residual;
    auto *normed = (CustomArNormBF16 *)output;
    if (draft) {
        if (!state.sm120EightRanks || rows != 7) return false;
        DraftReduceAddNorm7<<<rows, 256, 0, cudaStreamPerThread>>>(pointers, state.allSignals,
            state.allSignals.signals[rank], state.allScratch, hidden, weight, normed, eps, rank);
        if (cudaGetLastError() != cudaSuccess) return false;
        fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::DraftNorm);
        return true;
    }
    const bool twoStage = CustomArUseTwoStage(state.devices.size(), (size_t)count * 2);
    const int threads = twoStage && rows <= 128 ? 256 : 512;
    const int packed = count / 8, ranks = state.devices.size();
    const int work = twoStage ? packed / ranks + packed % ranks : packed;
    const int blocks = std::min(kCustomArMaxBlocks, (work + threads - 1) / threads);
    const bool directRows = rows <= 8 && !(state.sm120EightRanks && rows == 8);
#define CUSTOM_AR_NORM_RANK_CASE(RANKS)                                      \
    case RANKS:                                                             \
        if (directRows || (RANKS == 2 && rows <= kCustomArMaxBlocks))            \
            FastllmCustomAllReduceRMSNormKernel<RANKS, true>                 \
                <<<rows, 512, 0, cudaStreamPerThread>>>(                    \
                    pointers, state.allSignals, state.allSignals.signals[rank], \
                    state.allScratch, hidden, weight, normed, eps, rank);    \
        else if (rows <= kCustomArMaxBlocks)                                 \
            FastllmCustomAllReduceRMSNormKernel<RANKS, false>                \
                <<<rows, 512, 0, cudaStreamPerThread>>>(                    \
                    pointers, state.allSignals, state.allSignals.signals[rank], \
                    state.allScratch, hidden, weight, normed, eps, rank);    \
        else if (twoStage)                                                  \
            FastllmCustomAllReduceResidualKernel<RANKS, true>                \
                <<<blocks, threads, 0, cudaStreamPerThread>>>(               \
                    pointers, state.allSignals, state.allSignals.signals[rank], \
                    state.allScratch, hidden, rank, packed);                \
        else                                                                \
            FastllmCustomAllReduceResidualKernel<RANKS, false>               \
                <<<blocks, threads, 0, cudaStreamPerThread>>>(               \
                    pointers, state.allSignals, state.allSignals.signals[rank], \
                    state.allScratch, hidden, rank, packed);                \
        break
    switch (ranks) {
        CUSTOM_AR_NORM_RANK_CASE(2);
        CUSTOM_AR_NORM_RANK_CASE(4);
        CUSTOM_AR_NORM_RANK_CASE(6);
        CUSTOM_AR_NORM_RANK_CASE(8);
        default: return false;
    }
#undef CUSTOM_AR_NORM_RANK_CASE
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && rows > kCustomArMaxBlocks) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(rows);
        config.blockDim = dim3(256);
        config.stream = cudaStreamPerThread;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        status = cudaLaunchKernelEx(&config, FastllmCustomRMSNormPdlKernel,
            (const CustomArNormBF16 *)hidden, weight, normed, eps);
    }
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr, "[Fastllm] fused all-reduce RMSNorm failed on GPU %d: %s.\n",
                     deviceId, cudaGetErrorString(status));
        return false;
    }
    return true;
#else
    return false;
#endif
}
bool FastllmCudaCustomAllReduceAddRMSNorm(void *data, void *residual, const float *weight,
        void *output, int count, int channels, float eps, int dataType, int deviceId) {
    return CustomReduceAddNorm(data, residual, weight, output, count, channels, eps, dataType, deviceId, false);
}
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
static bool DraftTensor(const fastllm::Data &x, fastllm::DataType type, size_t count, int gpu) {
    if (x.dims.empty() || x.dataType != type || x.Count(0) != count || !x.cudaData ||
        x.dataDevice != fastllm::CUDA || x.dataDeviceIds != std::vector<int>{gpu} ||
        x.strides.size() != x.dims.size() || (reinterpret_cast<uintptr_t>(x.cudaData) & 3))
        return false;
    size_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.strides[i] != stride)
            return false;
        stride *= x.dims[i];
    }
    return true;
}
static void DraftBuffer(fastllm::Data &x, fastllm::DataType type, const std::vector<int> &dims, int gpu) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize(dims);
    x.ToDevice(fastllm::CUDA, {gpu}, false);
    x.Allocate(false);
}
template <class Kernel> static bool DraftGridFits(Kernel kernel, int gpu, int grid = sm120_draft::Grid) {
    int sm = 0, active = 0, coop = 0;
    return cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, gpu) == cudaSuccess &&
           cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, gpu) == cudaSuccess && coop &&
           cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, sm120_draft::Threads, 0) ==
               cudaSuccess &&
           active * sm >= grid;
}
// All ranks must agree on their operand shapes before packing or entering a peer collective.
static void PackDraftProjections(const fastllm::Data &first, const fastllm::Data &second, bool pairedGateUp,
                                 fastllm::Data &packed, const void *(&sources)[2], int gpu) {
    using namespace sm120_draft;
    const int firstSize = first.Count(0), secondSize = second.Count(0);
    DraftBuffer(packed, fastllm::BFLOAT16, {firstSize + secondSize}, gpu);
    auto *dst = static_cast<uint4 *>(packed.cudaData);
    if (pairedGateUp) {
        PackMlpGateUp<<<firstSize / (4 * 256), 256, 0, cudaStreamPerThread>>>(
            static_cast<const B *>(first.cudaData), reinterpret_cast<uint2 *>(dst));
    } else {
        PackProjection<<<firstSize / (8 * 256), 256, 0, cudaStreamPerThread>>>(
            static_cast<const B *>(first.cudaData), dst, first.dims[1]);
    }
    PackProjection<<<secondSize / (8 * 256), 256, 0, cudaStreamPerThread>>>(
        static_cast<const B *>(second.cudaData), dst + firstSize / 8, second.dims[1]);
    if (cudaPeekAtLastError() != cudaSuccess)
        throw std::runtime_error("SM120 draft weight packing failed");
    sources[0] = first.cudaData;
    sources[1] = second.cudaData;
}
#endif
bool FastllmCudaDraftAttention7(fastllm::Data &activation, fastllm::Data &hidden, const fastllm::Data &qkv,
                                const fastllm::Data &out, const fastllm::Data &qNorm,
                                const fastllm::Data &kNorm, const fastllm::Data &gamma,
                                const fastllm::Data &live, const fastllm::Data &bindings, int layer,
                                float eps, float theta, int gpu, FastllmDraftLayerProjection &storage) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    auto &[packed, workspace, sources] = storage;
    using namespace fastllm;
    using namespace sm120_draft;
    auto &state = GetCustomArState();
    CustomArCallLease lease;
    Args a{};
    if (!lease.Acquire(state, 8192, BFLOAT16, gpu, true, a.rank) || !state.sm120EightRanks)
        return false;
    bool ok =
        std::isfinite(eps) && eps > 0 && std::isfinite(theta) && theta > 0 &&
        DraftTensor(activation, BFLOAT16, Rows * 4096, gpu) &&
        DraftTensor(hidden, BFLOAT16, Rows * 4096, gpu) && DraftTensor(qkv, BFLOAT16, 768 * 4096, gpu) &&
        qkv.dims == std::vector<int>{768, 4096} && DraftTensor(out, BFLOAT16, 4096 * 512, gpu) &&
        out.dims == std::vector<int>{4096, 512} && !qkv.IsRepacked && !out.IsRepacked &&
        DraftTensor(qNorm, FLOAT32, Dim, gpu) && DraftTensor(kNorm, FLOAT32, Dim, gpu) &&
        DraftTensor(gamma, FLOAT32, 4096, gpu) && DraftTensor(live, INT32, 1, gpu) && layer >= 0 &&
        bindings.cudaData && bindings.dataType == INT32 && bindings.dataDevice == CUDA &&
        bindings.dataDeviceIds == std::vector<int>{gpu} &&
        bindings.GetBytes() >= (layer + 1) * 2 * sizeof(void *) && activation.cudaData != hidden.cudaData;
    ok = ok && DraftGridFits(FusedDraftAttentionSm120, gpu);
    const bool pack = sources[0] != qkv.cudaData || sources[1] != out.cudaData || !packed.cudaData;
    ok = ok && !(pack && FastllmCudaGraphIsCapturingFast());
    if (ok)
        DraftBuffer(workspace, INT32, {int((sizeof(Workspace) + 3) / 4)}, gpu);
    if (!FindOrRegisterCustomArPointers(state, a.rank, workspace.cudaData, Rows * 4096, BFLOAT16, ok,
                                        a.peers))
        return false;
    if (pack)
        PackDraftProjections(qkv, out, false, packed, sources, gpu);
    a.qkv = static_cast<uint4 *>(packed.cudaData);
    a.out = a.qkv + 768 * 4096 / 8;
    a.input = static_cast<const B *>(activation.cudaData);
    a.normed = static_cast<B *>(activation.cudaData);
    a.hidden = static_cast<B *>(hidden.cudaData);
    a.qNorm = static_cast<const float *>(qNorm.cudaData);
    a.kNorm = static_cast<const float *>(kNorm.cudaData);
    a.gamma = static_cast<const float *>(gamma.cudaData);
    a.live = static_cast<const int *>(live.cudaData);
    a.cache = static_cast<B *const *>(bindings.cudaData) + layer * 2;
    a.w = static_cast<Workspace *>(workspace.cudaData);
    a.eps = eps;
    a.theta = theta;
    a.signals = state.allSignals;
    void *args[] = {&a};
    fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::DraftNorm);
    auto err = cudaLaunchCooperativeKernel((const void *)FusedDraftAttentionSm120, Grid, Threads, args, 0,
                                           cudaStreamPerThread);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("SM120 draft attention launch failed: ") +
                                 cudaGetErrorString(err));
    fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::DraftNorm);
    return true;
#else
    return false;
#endif
}
bool FastllmCudaDraftMlp7(const fastllm::Data &input, fastllm::Data &hidden, fastllm::Data &output,
                          const fastllm::Data &gateUp, const fastllm::Data &down, const fastllm::Data &gamma,
                          float eps, int gpu, FastllmDraftLayerProjection &storage) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    auto &[packed, workspace, sources] = storage;
    using namespace fastllm;
    using namespace sm120_draft;
    auto &state = GetCustomArState();
    CustomArCallLease lease;
    MlpArgs a{};
    if (!lease.Acquire(state, 8192, BFLOAT16, gpu, true, a.rank) || !state.sm120EightRanks)
        return false;
    auto kernel = FusedDraftMlpSm120;
    bool ok = std::isfinite(eps) && eps > 0 && DraftTensor(input, BFLOAT16, Rows * 4096, gpu) &&
              input.dims.back() == 4096 && input.dims == hidden.dims &&
              DraftTensor(hidden, BFLOAT16, Rows * 4096, gpu) && hidden.cudaData != input.cudaData &&
              DraftTensor(gamma, FLOAT32, 4096, gpu) && gateUp.dims == std::vector<int>{1024, 4096} &&
              DraftTensor(gateUp, BFLOAT16, 1024 * 4096, gpu) && down.dims == std::vector<int>{4096, 512} &&
              DraftTensor(down, BFLOAT16, 4096 * 512, gpu) && !gateUp.IsRepacked && !down.IsRepacked &&
              DraftGridFits(kernel, gpu, MlpGrid);
    const bool pack = !packed.cudaData || sources[0] != gateUp.cudaData || sources[1] != down.cudaData;
    ok = ok && !(pack && FastllmCudaGraphIsCapturingFast());
    if (ok)
        DraftBuffer(workspace, INT32, {int((sizeof(MlpWorkspace) + 3) / 4)}, gpu);
    if (!FindOrRegisterCustomArPointers(state, a.rank, workspace.cudaData, Rows * 4096, BFLOAT16, ok,
                                        a.peers))
        return false;
    if (pack)
        PackDraftProjections(gateUp, down, true, packed, sources, gpu);
    if (&output != &input)
        DraftBuffer(output, BFLOAT16, input.dims, gpu);
    a.input = static_cast<const B *>(input.cudaData);
    a.gateUp = static_cast<const uint2 *>(packed.cudaData);
    a.down = static_cast<const uint4 *>(packed.cudaData) + 1024 * 4096 / 8;
    a.hidden = static_cast<B *>(hidden.cudaData);
    a.normed = static_cast<B *>(output.cudaData);
    a.gamma = static_cast<const float *>(gamma.cudaData);
    a.w = static_cast<MlpWorkspace *>(workspace.cudaData);
    a.signals = state.allSignals;
    a.eps = eps;
    void *args[] = {&a};
    fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::DraftNorm);
    auto err =
        cudaLaunchCooperativeKernel((const void *)kernel, MlpGrid, Threads, args, 0, cudaStreamPerThread);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("SM120 draft MLP launch failed: ") + cudaGetErrorString(err));
    fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::DraftNorm);
    return true;
#else
    return false;
#endif
}
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
namespace {
struct DraftHead120State {
    std::array<int, 8> devices{};
    std::array<sm120_draft::HeadArgs120, 8> ranks{};
    ~DraftHead120State() {
        int old = 0;
        cudaGetDevice(&old);
        for (int rank = 0; rank < 8; ++rank) {
            auto &a = ranks[rank];
            cudaSetDevice(devices[rank]);
            if (a.weight)
                cudaFree(const_cast<uint4 *>(a.weight));
            if (a.epoch)
                cudaFree(a.epoch);
            if (!rank && a.w2)
                cudaFree(const_cast<uint4 *>(a.w2));
            if (!rank && a.scratch)
                cudaFree(a.scratch);
        }
        cudaSetDevice(old);
    }
};
} // namespace
#endif
std::shared_ptr<void> FastllmCudaCreateDraftHead120(const std::vector<int> &devices,
                                                    const std::vector<const fastllm::Data *> &weights,
                                                    const fastllm::Data &w1, const fastllm::Data &w2) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    using namespace sm120_draft;
    if (devices.size() != 8 || weights.size() != 8 || FastllmCudaGraphIsCapturingFast() ||
        !DraftTensor(w1, fastllm::BFLOAT16, HeadVocab * 256, devices.front()) ||
        !DraftTensor(w2, fastllm::BFLOAT16, HeadVocab * 256, devices.front()) || w1.IsRepacked ||
        w2.IsRepacked)
        return {};
    for (int rank = 0; rank < 8; ++rank) {
        const auto *w = weights[rank];
        int major = 0, minor = 0;
        if (!w || w->dims != std::vector<int>{HeadShard, 4096} || w->IsRepacked ||
            !DraftTensor(*w, fastllm::BFLOAT16, HeadShard * 4096, devices[rank]) ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devices[rank]) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devices[rank]) != cudaSuccess ||
            major != 12 || minor != 0)
            return {};
    }
    if (!FastllmCudaPeerAccessInit(devices))
        return {};
    auto state = std::make_shared<DraftHead120State>();
    std::copy(devices.begin(), devices.end(), state->devices.begin());
    bool ok = true;
    for (int rank = 0; rank < 8 && ok; ++rank) {
        cudaSetDevice(devices[rank]);
        auto kernel = rank ? FusedDraftHeadProposalSm120<false> : FusedDraftHeadProposalSm120<true>;
        auto &a = state->ranks[rank];
        uint4 *packed = nullptr;
        ok = DraftGridFits(kernel, devices[rank], HeadGrid) &&
             cudaMalloc(&packed, size_t(HeadShard) * 4096 * sizeof(B)) == cudaSuccess;
        a.weight = packed;
        if (!ok)
            break;
        if (!rank) {
            ok = cudaMalloc(&a.scratch, sizeof(ProposalScratch)) == cudaSuccess &&
                 cudaMemsetAsync(a.scratch->ready, 0, sizeof(a.scratch->ready), cudaStreamPerThread) ==
                     cudaSuccess &&
                 cudaMemsetAsync(a.scratch->selected, 0, sizeof(a.scratch->selected), cudaStreamPerThread) ==
                     cudaSuccess &&
                 cudaMemsetAsync(a.scratch->tags, 0, sizeof(a.scratch->tags), cudaStreamPerThread) ==
                     cudaSuccess;
        } else
            a.scratch = state->ranks[0].scratch;
        a.w1 = static_cast<const B *>(w1.cudaData);
        if (!rank && ok) {
            uint4 *biasWeights = nullptr;
            ok = cudaMalloc(&biasWeights, size_t(HeadVocab) * 256 * sizeof(B)) == cudaSuccess;
            a.w2 = biasWeights;
            if (ok) {
                PackProjection<<<HeadVocab / 8, 256, 0, cudaStreamPerThread>>>(
                    static_cast<const B *>(w2.cudaData), biasWeights, 256);
                ok = cudaGetLastError() == cudaSuccess;
            }
        }
        a.rank = rank;
        ok = ok && cudaMalloc(&a.epoch, sizeof(unsigned)) == cudaSuccess &&
             cudaMemsetAsync(a.epoch, 0, sizeof(unsigned), cudaStreamPerThread) == cudaSuccess;
        if (!ok)
            break;
        PackProjection<<<HeadShard * 4096 / (8 * 256), 256, 0, cudaStreamPerThread>>>(
            static_cast<const B *>(weights[rank]->cudaData), packed, 4096);
        ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }
    cudaSetDevice(devices.front());
    if (ok)
        return state;
    cudaGetLastError();
#endif
    return {};
}
void FastllmCudaLaunchDraftProposal(void *opaque, int rank, const fastllm::Data &normalized,
                                   const fastllm::Data &bindings, int layers) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    using namespace sm120_draft;
    auto a = static_cast<DraftHead120State *>(opaque)->ranks[rank];
    // Single-request binding layout: a K/V pair per layer, one token/length word, then the request.
    a.request = rank ? nullptr : reinterpret_cast<const FastllmDraftSamplingRequest *>(
        static_cast<const uint64_t *>(bindings.cudaData) + 2 * layers + 1);
    a.input = static_cast<const B *>(normalized.cudaData);
    void *args[] = {&a};
    auto kernel = rank ? FusedDraftHeadProposalSm120<false> : FusedDraftHeadProposalSm120<true>;
    fastllm::cuda_pdl::Prepare(fastllm::cuda_pdl::Producer::DraftNorm);
    auto error =
        cudaLaunchCooperativeKernel((const void *)kernel, HeadGrid, Threads, args, 0, cudaStreamPerThread);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("SM120 draft head launch: ") + cudaGetErrorString(error));
#endif
}
void FastllmCudaReadDraftProposal(void *opaque, int *tokens) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    auto &a = static_cast<DraftHead120State *>(opaque)->ranks[0];
    if (cudaMemcpyAsync(tokens, a.scratch->ids, sizeof(a.scratch->ids), cudaMemcpyDeviceToHost,
                        cudaStreamPerThread) != cudaSuccess ||
        cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
        throw std::runtime_error("SM120 draft token read failed");
#endif
}

#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
namespace {
struct DraftContext120State {
    std::array<int, 8> devices{};
    std::array<sm120_draft::ContextArgs, 8> args{};
    ~DraftContext120State() {
        int previous;
        cudaGetDevice(&previous);
        for (int r = 0; r < 8; ++r) {
            cudaSetDevice(devices[r]);
            const auto &a = args[r];
            if (a.fc)
                cudaFree(const_cast<uint4 *>(a.fc));
            if (a.peers[r])
                cudaFree(a.peers[r]);
        }
        cudaSetDevice(previous);
    }
};
} // namespace
#endif

std::shared_ptr<void> FastllmCudaCreateDraftContext120(const std::vector<int> &devices,
                                                       const FastllmDraftContextWeights120 (&weights)[8]) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    using namespace sm120_draft;
    using fastllm::BFLOAT16;
    using fastllm::FLOAT32;
    if (devices.size() != 8 || FastllmCudaGraphIsCapturingFast())
        return {};
    for (int r = 0; r < 8; ++r) {
        int gpu = devices[r], major = 0, minor = 0;
        const auto &w = weights[r];
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, gpu) != cudaSuccess ||
            major != 12 || minor != 0 || !DraftGridFits(FusedDraftContextSm120, gpu, 256) || !w.fc ||
            !w.gamma || w.fc->IsRepacked || w.fc->dims != std::vector<int>{4096, 32768} ||
            !DraftTensor(*w.fc, BFLOAT16, 4096 * 32768, gpu) || !DraftTensor(*w.gamma, FLOAT32, 4096, gpu))
            return {};
        for (int l = 0; l < 5; ++l)
            if (!w.qkv[l] || !w.kNorm[l] || w.qkv[l]->IsRepacked ||
                w.qkv[l]->dims != std::vector<int>{768, 4096} ||
                !DraftTensor(*w.qkv[l], BFLOAT16, 768 * 4096, gpu) ||
                !DraftTensor(*w.kNorm[l], FLOAT32, 128, gpu))
                return {};
    }
    if (!FastllmCudaPeerAccessInit(devices))
        return {};
    auto s = std::make_shared<DraftContext120State>();
    std::copy(devices.begin(), devices.end(), s->devices.begin());
    bool ok = true;
    for (int r = 0; r < 8 && ok; ++r) {
        cudaSetDevice(devices[r]);
        const auto &w = weights[r];
        auto &a = s->args[r];
        uint4 *packed = nullptr;
        ok = cudaMalloc(&packed, size_t(4096 + 1280) * 4096 * sizeof(B)) == cudaSuccess;
        a.fc = packed;
        if (!ok)
            break;
        a.kv = packed + 4096 * 4096 / 8;
        ok = cudaMalloc(&a.peers[r], sizeof(ContextWork)) == cudaSuccess &&
             cudaMemsetAsync(a.peers[r]->ready, 0, sizeof(a.peers[r]->ready), cudaStreamPerThread) ==
                 cudaSuccess;
        if (!ok)
            break;
        a.gamma = static_cast<const float *>(w.gamma->cudaData);
        a.rank = r;
        PackContextFc<<<4096 * 4096 / (8 * 256), 256, 0, cudaStreamPerThread>>>(
            static_cast<const B *>(w.fc->cudaData), packed, r);
        for (int l = 0; l < 5; ++l) {
            a.kNorm[l] = static_cast<const float *>(w.kNorm[l]->cudaData);
            PackProjection<<<256 * 4096 / (8 * 256), 256, 0, cudaStreamPerThread>>>(
                static_cast<const B *>(w.qkv[l]->cudaData) + 512 * 4096,
                packed + size_t(4096 + l * 256) * 4096 / 8, 4096);
        }
        ok = cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }
    cudaSetDevice(devices.front());
    if (ok) {
        for (auto &a : s->args)
            for (int r = 0; r < 8; ++r)
                a.peers[r] = s->args[r].peers[r];
        return s;
    }
    cudaGetLastError();
#endif
    return {};
}

void FastllmCudaCommitDraftContext120(void *state, int rank, const FastllmDraftContextInput120 &input,
                                      int count, int old, int position, float eps, float theta) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    using namespace sm120_draft;
    auto &a = static_cast<DraftContext120State *>(state)->args[rank];
    a.feature = static_cast<const B *>(input.feature);
    a.nextBindings = static_cast<uint64_t *>(input.nextBindings);
    a.anchor = input.anchor;
    a.request = input.request;
    for (int i = 0; i < 10; ++i)
        a.cache[i] = static_cast<B *>(input.cache[i]);
    a.count = count;
    a.old = old;
    a.position = position;
    a.eps = eps;
    a.theta = theta;
    ++a.epoch;
    void *args[] = {&a};
    auto err = cudaLaunchCooperativeKernel((const void *)FusedDraftContextSm120, 256, Threads, args, 0,
                                           cudaStreamPerThread);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("SM120 draft context: ") + cudaGetErrorString(err));
#endif
}

// A completed producer stream has no outstanding feature writes for peers to wait on.
bool FastllmCudaDraftFeaturesReady120() {
    auto status = cudaStreamQuery(cudaStreamPerThread);
    if (status == cudaSuccess)
        return true;
    if (status != cudaErrorNotReady)
        throw std::runtime_error(std::string("SM120 draft feature readiness: ") + cudaGetErrorString(status));
    return false;
}

// Only a captured embedding with packed bindings can consume device-published inputs.
bool FastllmCudaCanContinueDraftGraph(void *opaque) {
#if defined(FASTLLM_ENABLE_NAIVE_VERIFY_SM120) && CUDART_VERSION >= 12030
    using namespace sm120_draft;
    auto graph = static_cast<cudaGraph_t>(opaque);
    size_t roots = 0, children = 0;
    cudaGraphNode_t embedding = nullptr, attention = nullptr;
#if CUDART_VERSION >= 13000
    auto dependentNodes = cudaGraphNodeGetDependentNodes;
#else
    auto dependentNodes = cudaGraphNodeGetDependentNodes_v2;
#endif
    cudaGraphEdgeData edge{};
    cudaKernelNodeParams kernel{};
    if (cudaGraphGetRootNodes(graph, nullptr, &roots) != cudaSuccess || roots != 1 ||
        cudaGraphGetRootNodes(graph, &embedding, &roots) != cudaSuccess ||
        cudaGraphKernelNodeGetParams(embedding, &kernel) != cudaSuccess ||
        kernel.func != (void *)PrepareDraftEmbeddingSm120 ||
        dependentNodes(embedding, nullptr, nullptr, &children) != cudaSuccess || children != 1 ||
        dependentNodes(embedding, &attention, &edge, &children) != cudaSuccess ||
        cudaGraphKernelNodeGetParams(attention, &kernel) != cudaSuccess ||
        kernel.func != (void *)FusedDraftAttentionSm120) {
        cudaGetLastError();
        return false;
    }
    return true;
#else
    return false;
#endif
}

bool FastllmCudaPrepareDraftEmbedding(const fastllm::Data &id, const fastllm::Data &live,
                                  const fastllm::Data &embedding, const fastllm::Data &mask,
                                  const fastllm::Data &gamma, fastllm::Data &hidden, fastllm::Data &positions,
                                  fastllm::Data &normed, float eps, int gpu) {
#ifdef FASTLLM_ENABLE_NAIVE_VERIFY_SM120
    using namespace sm120_draft;
    using namespace fastllm;
    if (!DraftTensor(id, FLOAT32, 1, gpu) || !DraftTensor(live, INT32, 1, gpu) ||
        embedding.dims.size() != 2 || embedding.dims[1] != 4096 ||
        !DraftTensor(embedding, BFLOAT16, embedding.Count(0), gpu) ||
        !DraftTensor(mask, BFLOAT16, 4096, gpu) || !DraftTensor(gamma, FLOAT32, 4096, gpu))
        return false;
    DraftBuffer(hidden, BFLOAT16, {1, Rows, 4096}, gpu);
    DraftBuffer(normed, BFLOAT16, {1, Rows, 4096}, gpu);
    DraftBuffer(positions, FLOAT32, {1, Rows}, gpu);
    PrepareDraftEmbeddingSm120<<<Rows, Threads, 0, cudaStreamPerThread>>>(
        static_cast<const float *>(id.cudaData), static_cast<const int *>(live.cudaData),
        static_cast<const B *>(embedding.cudaData), static_cast<const B *>(mask.cudaData),
        static_cast<const float *>(gamma.cudaData), static_cast<B *>(hidden.cudaData),
        static_cast<B *>(normed.cudaData), static_cast<float *>(positions.cudaData), eps);
    if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("SM120 draft embedding launch failed");
    fastllm::cuda_pdl::Mark(fastllm::cuda_pdl::Producer::DraftNorm);
    return true;
#else
    return false;
#endif
}

bool FastllmCudaDraftReduceAddNorm7(void *data, void *residual, const float *weight,
        void *output, float eps, int deviceId) {
    return CustomReduceAddNorm(data, residual, weight, output, 7 * 4096, 4096, eps, fastllm::BFLOAT16, deviceId, true);
}

bool FastllmCudaCustomAllReduceAndGather(void *routed, void *dest,
        void *shared, int count, int deviceId) {
    if (count <= 0 || count % 8 ||
        count / 4 > kCustomArThreads * kCustomArMaxBlocks ||
        (size_t)count * sizeof(float) > CustomArMaxBytes()) return false;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, (size_t)count * sizeof(float), fastllm::FLOAT32,
                       deviceId, true, rank)) return false;
    const int ranks = (int)state.devices.size();
    if (count % (ranks * 8)) return false;
    const size_t routedBytes = (size_t)count * 4, sharedBytes = (size_t)count * 2;
    // Rank-local eligibility must participate in the pointer rendezvous: no
    // rank can return early while other ranks wait for its pointer or barrier.
    const bool eligible = routed && dest && shared &&
        (((uintptr_t)routed | (uintptr_t)dest | (uintptr_t)shared) & 15u) == 0 &&
        (routed == dest || CustomArBuffersDisjoint(routed, routedBytes, dest, routedBytes)) &&
        CustomArBuffersDisjoint(routed, routedBytes, shared, sharedBytes) &&
        CustomArBuffersDisjoint(dest, routedBytes, shared, sharedBytes);
    CustomArPointers routedData{}, sharedData{};
    if (!FindOrRegisterCustomArPointers(state, rank, routed, count,
            fastllm::FLOAT32, eligible, routedData) ||
        !FindOrRegisterCustomArPointers(state, rank, shared, count,
            fastllm::BFLOAT16, eligible, sharedData)) return false;
    if (state.taggedPeer && state.taggedPeer->CanUseJoint(count)) {
        const bool launched = state.taggedPeer->LaunchJoint(routed, dest, shared, count, rank);
        if (cudaGetLastError() != cudaSuccess) {
            FastllmCudaSetThreadError();
            return false;
        }
        return launched;
    }
    // Spread small messages across more CTAs without exceeding signal capacity.
    // Larger messages retain the original 512-thread blocks.
    const int threads =
        count / 4 <= 128 * kCustomArMaxBlocks ? 128 : kCustomArThreads;
    const int blocks = (count / 4 + threads - 1) / threads;
#define CUSTOM_AR_GATHER_CASE(RANKS)                                        \
    case RANKS:                                                            \
        FastllmCustomAllReduceAndGatherKernel<RANKS>                        \
            <<<blocks, threads, 0, cudaStreamPerThread>>>(                   \
                routedData, sharedData, state.allSignals,                  \
                state.allSignals.signals[rank], (float4 *)dest,            \
                (uint2 *)shared, rank, count / 4);                         \
        break
    switch (ranks) {
        CUSTOM_AR_GATHER_CASE(2);
        CUSTOM_AR_GATHER_CASE(4);
        CUSTOM_AR_GATHER_CASE(6);
        CUSTOM_AR_GATHER_CASE(8);
        default: return false;
    }
#undef CUSTOM_AR_GATHER_CASE
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr, "[Fastllm] joint reduce/gather failed on GPU %d: %s.\n",
                     deviceId, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaCustomAllGatherRows(void *data, int count, int rowElements,
                                   int dataType, int deviceId) {
    if (count <= 0 || rowElements <= 0 || count % rowElements) return false;
    const size_t typeBytes = CustomArTypeBytes(dataType);
    const size_t bytes = (size_t)count * typeBytes;
    const size_t policyBytes = (size_t)rowElements * typeBytes;
    if (!typeBytes || bytes > CustomArMaxBytes() || policyBytes % 16) return false;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, policyBytes, dataType, deviceId, true, rank)) return false;
    const int ranks = (int)state.devices.size();
    if (bytes % (ranks * 16)) return false;
    CustomArPointers rankData{};
    if (!FindOrRegisterCustomArPointers(state, rank, data, count, dataType,
            data && ((uintptr_t)data & 15u) == 0, rankData)) return false;
    const int part = bytes / (ranks * 16);
    const int blocks = std::max(1, std::min(kCustomArMaxBlocks,
        (part + kCustomArThreads - 1) / kCustomArThreads));
    if (state.sm120EightRanks && count / rowElements == 64 && dataType == fastllm::FLOAT32 &&
        part >= kCustomArThreads) {
        FastllmCustomAllGatherInplaceKernel<8, false, true><<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(
            rankData, state.allSignals, state.allSignals.signals[rank], static_cast<uint4 *>(data), rank, part);
        return cudaGetLastError() == cudaSuccess;
    }
#define CUSTOM_AG_RANK_CASE(RANKS)                                           \
    case RANKS:                                                             \
        if (RANKS > 2 && part <= kCustomArThreads / 2)                       \
            FastllmCustomAllGatherInplaceKernel<RANKS, true>                  \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(       \
                    rankData, state.allSignals, state.allSignals.signals[rank], \
                    reinterpret_cast<uint4 *>(data), rank, part);            \
        else                                                                 \
            FastllmCustomAllGatherInplaceKernel<RANKS, false>                 \
                <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(       \
                    rankData, state.allSignals, state.allSignals.signals[rank], \
                    reinterpret_cast<uint4 *>(data), rank, part);            \
        break
    switch (ranks) {
        CUSTOM_AG_RANK_CASE(2);
        CUSTOM_AG_RANK_CASE(4);
        CUSTOM_AG_RANK_CASE(6);
        CUSTOM_AG_RANK_CASE(8);
        default: return false;
    }
#undef CUSTOM_AG_RANK_CASE
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        std::fprintf(stderr, "[Fastllm] custom all-gather launch failed on GPU %d: %s.\n",
                     deviceId, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaCustomAllGatherAddRMSNorm(void *data, void *residual,
        const float *weight, void *output, int count, int channels,
        float eps, int dataType, int deviceId) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    // This path reads the owner shards directly and writes the residual;
    // unlike AllReduce it does not use the bounded in-place scratch buffer.
    // Its collective grid is capped below and loops over larger row batches.
    if (dataType != fastllm::BFLOAT16 || channels != 4096 || count <= 0 ||
        count % channels ||
        !std::isfinite(eps) || eps <= 0) return false;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, channels * 2, dataType, deviceId, true, rank)) return false;
    const int rows = count / channels, ranks = state.devices.size();
    const bool aligned = data && residual && weight && output &&
        (((uintptr_t)data | (uintptr_t)residual | (uintptr_t)weight | (uintptr_t)output) & 15u) == 0;
    auto disjoint = [count](const void *a, const void *b) {
        uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
        return (x > y ? x - y : y - x) >= (size_t)count * 2;
    };
    const bool eligible = aligned && FastllmCudaRuntimeArch() == 90 &&
        disjoint(data, residual) && disjoint(data, output) && disjoint(residual, output);
    CustomArPointers pointers{};
    if (!FindOrRegisterCustomArPointers(state, rank, data, count, dataType,
                                      eligible, pointers)) return false;
    auto *hidden = (CustomArNormBF16 *)residual;
    auto *normed = (CustomArNormBF16 *)output;
    const int rowsPerBlock = rows <= 32 ? 1 : rows <= 64 ? 2 : 4;
    // Once the bounded collective grid would serialize row groups, coalesce
    // the peer reads and residual stores, then let PDL launch one Norm CTA per
    // row. Both paths use the same 256-thread reduction and existing buffers.
    const bool splitNorm = rows > 4 * kCustomArMaxBlocks;
    const int part = ((rows + ranks - 1) / ranks) * (channels / 8);
    const int blocks = std::min(splitNorm ? (part + 511) / 512 :
                               (rows + rowsPerBlock - 1) / rowsPerBlock, kCustomArMaxBlocks);
#define CUSTOM_AG_NORM_RANK_CASE(RANKS)                                      \
    case RANKS:                                                             \
        if (splitNorm)                                                      \
            FastllmCustomAllGatherResidualKernel<RANKS>                      \
            <<<blocks, 512, 0, cudaStreamPerThread>>>(                       \
                pointers, state.allSignals, state.allSignals.signals[rank], \
                hidden, rank, rows);                                       \
        else if (rowsPerBlock == 1)                                         \
            FastllmCustomAllGatherRMSNormKernel<RANKS, 1>                    \
            <<<blocks, 256, 0, cudaStreamPerThread>>>(                       \
                pointers, state.allSignals, state.allSignals.signals[rank], \
                hidden, weight, normed, eps, rank, rows);                   \
        else if (rowsPerBlock == 2)                                         \
            FastllmCustomAllGatherRMSNormKernel<RANKS, 2>                    \
            <<<blocks, 512, 0, cudaStreamPerThread>>>(                       \
                pointers, state.allSignals, state.allSignals.signals[rank], \
                hidden, weight, normed, eps, rank, rows);                   \
        else                                                               \
            FastllmCustomAllGatherRMSNormKernel<RANKS, 4>                    \
            <<<blocks, 1024, 0, cudaStreamPerThread>>>(                      \
                pointers, state.allSignals, state.allSignals.signals[rank], \
                hidden, weight, normed, eps, rank, rows);                   \
        break
    switch (ranks) {
        CUSTOM_AG_NORM_RANK_CASE(2);
        CUSTOM_AG_NORM_RANK_CASE(4);
        CUSTOM_AG_NORM_RANK_CASE(6);
        CUSTOM_AG_NORM_RANK_CASE(8);
        default: return false;
    }
#undef CUSTOM_AG_NORM_RANK_CASE
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess && splitNorm) {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(rows);
        config.blockDim = dim3(256);
        config.stream = cudaStreamPerThread;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        status = cudaLaunchKernelEx(&config, FastllmCustomRMSNormPdlKernel,
            (const CustomArNormBF16 *)hidden, weight, normed, eps);
    }
    if (status != cudaSuccess) {
        FastllmCudaSetThreadError();
        // In the split path the residual kernel may already be queued. A
        // fallback would add it twice; only pre-launch rejection returns false.
        throw std::runtime_error(std::string("Fused all-gather RMSNorm launch failed: ") +
                                 cudaGetErrorString(status));
    }
    return true;
#else
    return false;
#endif
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

template <bool Markov>
static bool CustomScalarMax(void *value, int deviceId, int *tokens, const void *embed, float *context, int batch = 1, int steps = 0) {
    if (batch < 1 || batch > 8) return false;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    // Reuse the small-message P2P eligibility; the integer kernel uses exactly
    // the same system-scope barriers as the calibrated float path.
    if (!lease.Acquire(state, batch * sizeof(unsigned long long), fastllm::FLOAT32,
                       deviceId, true, rank)) return false;
    CustomArPointers pointers{};
    // A distinct rendezvous tag prevents pairing a mismatched SUM/MAX round.
    constexpr int scalarMaxRegistration = Markov ? -7 : -2;
    const bool eligible = (!Markov || (tokens && (!context || embed))) && value && (((uintptr_t)value & 7u) == 0);
    if (!FindOrRegisterCustomArPointers(state, rank, value, batch,
            scalarMaxRegistration, eligible, pointers)) return false;
#define CUSTOM_MAX_CASE(N) case N: FastllmCustomScalarMaxUint64Kernel<N, Markov> \
        <<<1, batch * 32, 0, cudaStreamPerThread>>>(pointers, state.allSignals, \
            state.allSignals.signals[rank], (unsigned long long*)value, rank, tokens, \
            static_cast<const __nv_bfloat16 *>(embed), context, steps); break
    switch (state.devices.size()) {
        CUSTOM_MAX_CASE(2);
        CUSTOM_MAX_CASE(4);
        CUSTOM_MAX_CASE(6);
        CUSTOM_MAX_CASE(8);
        default: return false;
    }
#undef CUSTOM_MAX_CASE
    if (cudaGetLastError() != cudaSuccess) { FastllmCudaSetThreadError(); return false; }
    return true;
}

bool FastllmCudaCustomScalarMaxUint64(void *value, int deviceId) {
    return CustomScalarMax<false>(value, deviceId, nullptr, nullptr, nullptr);
}

bool FastllmCudaCustomMarkovWinner(void *value, int device, int *tokens, const void *embed, float *context) {
    return CustomScalarMax<true>(value, device, tokens, embed, context);
}

bool FastllmCudaCustomMarkovWinners(void *values, int device, int *tokens, const void *embed,
                                     float *context, int batch, int steps) {
    return CustomScalarMax<true>(values, device, tokens, embed, context, batch, steps);
}
