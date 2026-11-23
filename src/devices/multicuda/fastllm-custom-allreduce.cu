// Graph-safe small-tensor all-reduce for single-process MultiCUDA.
//
// The kernel/barrier design is adapted from vLLM's Apache-2.0 licensed
// csrc/custom_all_reduce.cuh. Decode-sized messages use its one-stage direct
// peer-read shape; larger TP>=4 messages use reduce-scatter/all-gather. Fixed
// graph workspaces let us register every pointer during warmup.

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"
#include "naive-n05-cuda.cuh"

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
template <int Ranks, bool Flatten>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllGatherInplaceKernel(CustomArPointers rankData,
        CustomArRankSignals allSignals, CustomArSignal *selfSignal,
        uint4 *output, int rank, int part) {
    CustomArRankData pointers = rankData.Load();
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
    const int stride = gridDim.x * blockDim.x;
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (Flatten) {
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
template <int Ranks>
__global__ void CustomArExpertRowsKernel(CustomArPointers registered, CustomArRankSignals signals,
        CustomArSignal *self, CustomArNormPacketType *local, const int32_t *ids, CustomArNormBF16 *out,
        int rank, int packets, int rowPackets, int topk) {
    const CustomArRankData peers = registered.Load();
    const int packet = blockIdx.x * blockDim.x + threadIdx.x;
    int first = 0;
    if (packet < packets) {
        const int row = packet / rowPackets;
        first = row * topk * rowPackets + packet % rowPackets;
        float sum[8] = {};
        for (int slot = 0; slot < topk; ++slot) {
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
    if (packet < packets)
        reinterpret_cast<CustomArNormPacketType *>(out)[packet] = CustomArReducePacket<Ranks>(peers, first);
    CustomArBarrierFinal<Ranks>(signals, self, rank);
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
    constexpr int part = 512 / Ranks, pitch = part + 512 % Ranks;
    CustomArBarrierStart<Ranks>(signals, self, rank);
    if constexpr (!Direct) {
        if (t < (rank == Ranks - 1 ? pitch : part))
            ((CustomArNormPacketType *)scratch.scratch[rank])[row * pitch + t] =
                CustomArReducePacket<Ranks>(p, row * 512 + rank * part + t);
        CustomArBarrierPublish<Ranks>(signals, self, rank);
    }
    // Publish completes all reads of the original input. Two-stage peers now
    // read only scratch; the next collective's start barrier cannot pass until
    // every peer has finished this kernel, so it also protects scratch reuse.
    CustomArNormPacketType v;
    if constexpr (Direct)
        v = CustomArReducePacket<Ranks>(p, row * 512 + t);
    else {
        int peer = min(t / part, Ranks - 1);
        v = ((const CustomArNormPacketType *)scratch.scratch[peer])[row * pitch + t - peer * part];
    }
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
    CustomArPushWorkspaces allPushWorkspaces{};
    std::unique_ptr<CustomArTaggedPeer> taggedPeer;
    std::unique_ptr<SwaOutputState> swaOutput;
    void *inplaceScratch[kCustomArMaxRanks]{};
    int pushBlocks = 0;
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

}  // namespace

fastllm::TransientAllReduceInputs::TransientAllReduceInputs(bool enabled)
    : previous(customArTransientInputs) {
    customArTransientInputs |= enabled;
}

fastllm::TransientAllReduceInputs::~TransientAllReduceInputs() {
    customArTransientInputs = previous;
}

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
    return FastllmCudaSwaVerifyLayerPrepared(branch, hidden, inputNorm, qkvWeight, qkvBias,
        oWeight, postNorm, sink, position, live, caches, q, attn, projected, normed,
        feature, routerWeight, router, c, deviceId, nullptr);
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
                                    qkvWeight, qkvScales, projected.cudaData, rows, args.positions, c.theta, normed.cudaData)) {
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

bool FastllmCudaCustomAllReduceExpertRows(void *partials, const int32_t *ids, void *output,
        int rows, int hidden, int topk, int deviceId) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    constexpr int threads = 256;
    if (rows <= 0 || hidden <= 0 || hidden % 8 || topk <= 0 || topk > 16 ||
        (int64_t)rows * hidden > 8 * threads * kCustomArMaxBlocks) return false;
    const int packets = rows * hidden / 8, blocks = (packets + threads - 1) / threads;
    CustomArState &state = GetCustomArState();
    CustomArCallLease lease;
    int rank = -1;
    if (!lease.Acquire(state, size_t(hidden) * 2, fastllm::BFLOAT16, deviceId, true, rank)) return false;
    CustomArPointers peers{};
    const size_t outputBytes = size_t(packets) * sizeof(CustomArNormPacketType);
    const size_t partialBytes = outputBytes * topk, idBytes = size_t(rows) * topk * sizeof(*ids);
    const bool eligible = partials && ids && output &&
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
    switch (state.devices.size()) {
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

bool FastllmCudaCustomAllReduceAddRMSNorm(void *data, void *residual,
        const float *weight, void *output, int count, int channels,
        float eps, int dataType, int deviceId) {
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
    const bool twoStage = CustomArUseTwoStage(state.devices.size(), (size_t)count * 2);
    const int threads = twoStage && rows <= 128 ? 256 : 512;
    const int packed = count / 8, ranks = state.devices.size();
    const int work = twoStage ? packed / ranks + packed % ranks : packed;
    const int blocks = std::min(kCustomArMaxBlocks, (work + threads - 1) / threads);
#define CUSTOM_AR_NORM_RANK_CASE(RANKS)                                      \
    case RANKS:                                                             \
        if (rows <= 8 || (RANKS == 2 && rows <= kCustomArMaxBlocks))          \
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
