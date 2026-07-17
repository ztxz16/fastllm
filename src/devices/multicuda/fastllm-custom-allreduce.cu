// Graph-safe small-tensor all-reduce for single-process MultiCUDA.
//
// The kernel/barrier design is adapted from vLLM's Apache-2.0 licensed
// csrc/custom_all_reduce.cuh.  FastLLM only needs the one-stage path here:
// DeepSeek-V4 decode reduces 4096 BF16/FP16 values, all GPUs are peer-accessible,
// and fixed graph workspaces let us register every pointer during warmup.

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kCustomArMaxRanks = 8;
constexpr int kCustomArMaxBlocks = 36;
constexpr int kCustomArThreads = 512;
using CustomArFlag = uint32_t;

struct CustomArSignal {
    alignas(128) CustomArFlag start[kCustomArMaxBlocks][kCustomArMaxRanks];
    alignas(128) CustomArFlag end[kCustomArMaxBlocks][kCustomArMaxRanks];
    alignas(128) CustomArFlag flag[kCustomArMaxBlocks];
};

struct __align__(16) CustomArRankData {
    const void *ptrs[kCustomArMaxRanks];
};

struct __align__(16) CustomArRankSignals {
    CustomArSignal *signals[kCustomArMaxRanks];
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

template <int Ranks>
__device__ __forceinline__ void CustomArBarrierEnd(
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

template <typename T, int Ranks>
__global__ __launch_bounds__(kCustomArThreads, 1)
void FastllmCustomAllReduceKernel(CustomArRankData *rankData,
                                  CustomArRankSignals allSignals,
                                  CustomArSignal *selfSignal,
                                  T *__restrict__ output,
                                  int rank, int packedCount) {
    using P = typename CustomArPacked<T>::P;
    using A = typename CustomArPacked<T>::A;
    CustomArRankData pointers = *rankData;
    CustomArBarrierStart<Ranks>(allSignals, selfSignal, rank);
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
        reinterpret_cast<P *>(output)[index] = result;
    }
    CustomArBarrierEnd<Ranks>(allSignals, selfSignal, rank);
}

template <typename T>
__device__ __forceinline__ T CustomArAdd(T a, T b);

template <>
__device__ __forceinline__ half CustomArAdd(half a, half b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16 CustomArAdd(
        __nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ float CustomArAdd(float a, float b) {
    return a + b;
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
#pragma unroll
        for (int item = 0; item < CustomArPacked<T>::size; ++item) {
            first.data[item] = CustomArAdd(first.data[item], second.data[item]);
            first.data[item] = CustomArAdd(first.data[item],
                                           residualValue.data[item]);
        }
        reinterpret_cast<P *>(output)[index] = first;
    }
    CustomArBarrierEnd<2>(allSignals, selfSignal, rank);
}

struct CustomArState {
    std::mutex mutex;
    std::condition_variable condition;
    bool initAttempted = false;
    bool initialized = false;
    bool logged = false;
    std::vector<int> devices;
    std::map<int, int> rankByDevice;
    CustomArRankSignals allSignals{};
    void *inplaceScratch[kCustomArMaxRanks]{};
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

constexpr size_t CustomArMaxBytes() {
    // Match vLLM's custom-all-reduce registration envelope. TP=2 always uses
    // the one-stage direct-read kernel; batch 32 x hidden 5120 is already
    // 320 KiB and must not fall back to NCCL at the old DeepSeek-only limit.
    return 8ULL * 1024ULL * 1024ULL;
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
                    T *output, int rank, int count) {
    constexpr int packedWidth = CustomArPacked<T>::size;
    int packedCount = count / packedWidth;
    int blocks = std::max(1, std::min(kCustomArMaxBlocks,
                          (packedCount + kCustomArThreads - 1) /
                              kCustomArThreads));
#define CUSTOM_AR_RANK_CASE(RANKS)                                              \
    case RANKS:                                                                 \
        FastllmCustomAllReduceKernel<T, RANKS>                                  \
            <<<blocks, kCustomArThreads, 0, cudaStreamPerThread>>>(              \
                rankData, state.allSignals, state.allSignals.signals[rank],      \
                output, rank, packedCount);                                      \
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
                       const T *residual, T *output,
                       int rank, int count) {
    if (state.devices.size() != 2) {
        return false;
    }
    constexpr int packedWidth = CustomArPacked<T>::size;
    int packedCount = count / packedWidth;
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

}  // namespace

bool FastllmCudaCustomAllReduceInit(const std::vector<int> &devices) {
    CustomArState &state = GetCustomArState();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.initialized) {
        return state.devices == devices;
    }
    if (state.initAttempted) {
        return false;
    }
    state.initAttempted = true;
    if (devices.size() != 2 && devices.size() != 4 &&
        devices.size() != 6 && devices.size() != 8) {
        return false;
    }

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
        if (signal == nullptr ||
            cudaMemset(signal, 0, sizeof(CustomArSignal)) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        state.allSignals.signals[rank] = signal;
        // The one-stage direct-read algorithm cannot overwrite an input before
        // every peer has consumed it.  FastLLM's TP reductions are normally
        // in-place, so reduce into a fixed per-rank scratch buffer and enqueue
        // the copy-back after the kernel's final cross-GPU barrier.
        state.inplaceScratch[rank] = FastllmCudaMalloc(CustomArMaxBytes());
        if (state.inplaceScratch[rank] == nullptr) {
            ok = false;
            break;
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
        std::fprintf(stderr,
                     "[Fastllm] graph-safe custom all-reduce is unavailable; "
                     "falling back to NCCL.\n");
        std::fflush(stderr);
        return false;
    }
    state.devices = devices;
    state.initialized = true;
    std::fprintf(stderr,
                 "[Fastllm] graph-safe custom all-reduce initialized for %zu GPUs.\n",
                 devices.size());
    std::fflush(stderr);
    return true;
}

bool FastllmCudaCustomAllReduce(void *data, void *dest, int count,
                                int dataType, int deviceId) {
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
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized) {
            return false;
        }
        auto rankIt = state.rankByDevice.find(deviceId);
        if (rankIt == state.rankByDevice.end()) {
            return false;
        }
        rank = rankIt->second;
    }

    CustomArRankData *rankData = nullptr;
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, rankData)) {
        return false;
    }
    void *kernelDest = data == dest ? state.inplaceScratch[rank] : dest;
    bool launched = false;
    if (dataType == fastllm::DataType::FLOAT16) {
        launched = LaunchCustomAr(state, rankData,
                                  reinterpret_cast<half *>(kernelDest),
                                  rank, count);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        launched = LaunchCustomAr(
            state, rankData, reinterpret_cast<__nv_bfloat16 *>(kernelDest),
            rank, count);
    } else if (dataType == fastllm::DataType::FLOAT32) {
        launched = LaunchCustomAr(state, rankData,
                                  reinterpret_cast<float *>(kernelDest),
                                  rank, count);
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

bool FastllmCudaCustomAllReduceAdd(void *data, void *dest, int count,
                                   int dataType, int deviceId) {
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
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.initialized || state.devices.size() != 2) {
            return false;
        }
        auto rankIt = state.rankByDevice.find(deviceId);
        if (rankIt == state.rankByDevice.end()) {
            return false;
        }
        rank = rankIt->second;
    }

    CustomArRankData *rankData = nullptr;
    if (!FindOrRegisterCustomArPointers(state, rank, data, count,
                                        dataType, rankData)) {
        return false;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const half *>(dest),
            reinterpret_cast<half *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::BFLOAT16) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const __nv_bfloat16 *>(dest),
            reinterpret_cast<__nv_bfloat16 *>(dest), rank, count);
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        return LaunchCustomArAdd(
            state, rankData, reinterpret_cast<const float *>(dest),
            reinterpret_cast<float *>(dest), rank, count);
    }
    return false;
}
