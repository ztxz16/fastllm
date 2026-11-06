/* #include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/functional.h> */

#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"

#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <cuda_fp8.h>
#include "sampling.cuh"

extern "C" bool FastllmNcclGraphPeerCopy(int dstDevice, void *dst,
                                          int srcDevice, const void *src,
                                          size_t bytes);

static bool FastllmCudaDataHasDenseStrides(const fastllm::Data &data);
static bool FastllmCudaResolveDataDeviceId(const fastllm::Data &data,
                                           int &device);
static bool FastllmCudaDataCanShareDevice(const fastllm::Data &reference,
                                          const fastllm::Data &other);

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#endif

// 线程级 CUDA 错误标志：showError 报错时置位。CUDA graph 捕获路径在捕获体前清零、
// 捕获体后检查，任何算子报错都能被感知并中止捕获，避免带着坏状态继续运行。
static thread_local bool fastllmCudaThreadErrorFlag = false;
// Whole-step capture spans the scheduler thread plus one persistent worker per
// GPU.  A thread-local flag alone loses failures raised by worker-side cuBLAS,
// NCCL, or custom kernels, so retain a capture-epoch aggregate as well.
static std::atomic<bool> fastllmCudaGraphErrorFlag(false);

namespace {
    // A pool miss must not unwind one tensor-parallel rank while its peers are
    // still recording NCCL collectives. CUDA capture records kernel arguments
    // without executing them, so a small, process-lifetime address can carry the
    // failed rank to the common post-body abort barrier. Managed whole-step
    // callers check the graph error flag before instantiation, guaranteeing
    // that a graph containing this address is never launched.
    static std::mutex fastllmCudaGraphAllocationFailurePlaceholderMutex;
    static std::map<int, void*>
        fastllmCudaGraphAllocationFailurePlaceholders;

    static cudaError_t FastllmCudaGraphPrepareAllocationFailurePlaceholder() {
        int device = -1;
        cudaError_t state = cudaGetDevice(&device);
        if (state != cudaSuccess) {
            return state;
        }
        std::lock_guard<std::mutex> guard(
            fastllmCudaGraphAllocationFailurePlaceholderMutex);
        if (fastllmCudaGraphAllocationFailurePlaceholders.find(device) !=
                fastllmCudaGraphAllocationFailurePlaceholders.end()) {
            return cudaSuccess;
        }
        // Keep one tiny process-lifetime allocation per device. It is resolved
        // before stream capture and intentionally stays outside FastLLM's
        // reusable pool, so a failed Data owner can mark it borrowed without
        // affecting allocator ownership. This also works on the ROCm build,
        // where runtime symbol-address lookup is unavailable.
        void *ptr = nullptr;
        state = FastllmCudaCheckedMalloc(&ptr, 256, __FILE__, __LINE__);
        if (state == cudaSuccess && ptr != nullptr) {
            fastllmCudaGraphAllocationFailurePlaceholders[device] = ptr;
        }
        return state;
    }
}

void FastllmCudaClearThreadError() {
    fastllmCudaThreadErrorFlag = false;
}

void FastllmCudaSetThreadError() {
    fastllmCudaThreadErrorFlag = true;
    fastllmCudaGraphErrorFlag.store(true, std::memory_order_release);
}

bool FastllmCudaGetThreadError() {
    return fastllmCudaThreadErrorFlag;
}

void FastllmCudaClearGraphError() {
    fastllmCudaGraphErrorFlag.store(false, std::memory_order_release);
}

bool FastllmCudaGetGraphError() {
    return fastllmCudaGraphErrorFlag.load(std::memory_order_acquire);
}

void showError(cudaError_t result, char const* const message, const char* const file,
           int const line) {
    if (cudaSuccess != result) {
        fastllmCudaThreadErrorFlag = true;
        fastllmCudaGraphErrorFlag.store(true, std::memory_order_release);
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n",
            message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
        fflush(stdout);
    }  
}

static std::atomic<bool> fastllmCudaMallocDisabled(false);
static std::atomic<int> fastllmCudaMallocRejectLogCount(0);
static std::mutex fastllmCudaMallocCheckMutex;

// A graph keeps raw kernel arguments after host-side Data temporaries have been
// destroyed. Track pool pointers returned by streams participating in the
// whole-step capture. External allocator threads must not reuse those pointers
// while capture finalization is deciding which idle blocks the graph owns.
enum FastllmCudaGraphPoolPhase {
    FASTLLM_CUDA_GRAPH_POOL_IDLE = 0,
    FASTLLM_CUDA_GRAPH_POOL_CAPTURING = 1,
    FASTLLM_CUDA_GRAPH_POOL_FINALIZING = 2,
};

struct FastllmCudaGraphCaptureIdentity {
    int device = -1;
    unsigned long long id = 0;
    bool valid = false;
};

static std::mutex fastllmCudaGraphPoolMutex;
static std::atomic<int> fastllmCudaGraphPoolPhase(
    FASTLLM_CUDA_GRAPH_POOL_IDLE);
static std::set<void*> fastllmCudaGraphPoolTouchedDuringCapture;
static std::set<std::pair<int, unsigned long long>>
    fastllmCudaGraphPoolCaptureIds;

static FastllmCudaGraphCaptureIdentity
FastllmCudaGraphCurrentCaptureIdentity() {
    FastllmCudaGraphCaptureIdentity identity;
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) !=
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
        return identity;
    }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long captureId = 0;
    cudaError_t state = cudaStreamGetCaptureInfo(
        cudaStreamPerThread, &status, &captureId);
    if (state != cudaSuccess) {
        cudaGetLastError();
        FastllmCudaSetThreadError();
        return identity;
    }
    if (status == cudaStreamCaptureStatusNone ||
        status == cudaStreamCaptureStatusInvalidated ||
        cudaGetDevice(&identity.device) != cudaSuccess) {
        return identity;
    }
    identity.id = captureId;
    identity.valid = true;
    return identity;
}

static void FastllmCudaGraphRegisterCurrentCapture() {
    FastllmCudaGraphCaptureIdentity identity =
        FastllmCudaGraphCurrentCaptureIdentity();
    if (!identity.valid) {
        FastllmCudaSetThreadError();
        return;
    }
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) ==
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
        fastllmCudaGraphPoolCaptureIds.insert(
            std::make_pair(identity.device, identity.id));
    }
}

static bool FastllmCudaGraphCaptureIdentityRegisteredLocked(
        const FastllmCudaGraphCaptureIdentity &identity) {
    return identity.valid &&
        fastllmCudaGraphPoolCaptureIds.find(
            std::make_pair(identity.device, identity.id)) !=
        fastllmCudaGraphPoolCaptureIds.end();
}

// Called while the owning device-pool lock is held. All pool transitions use
// the same pool -> graph lock order, so finalization cannot race an allocation
// or free of the pointer being classified.
static bool FastllmCudaGraphPoolPointerReusableLocked(
        void *ptr, const FastllmCudaGraphCaptureIdentity &identity) {
    int phase = fastllmCudaGraphPoolPhase.load(std::memory_order_acquire);
    if (phase == FASTLLM_CUDA_GRAPH_POOL_IDLE) {
        return true;
    }
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    if (fastllmCudaGraphPoolTouchedDuringCapture.find(ptr) ==
            fastllmCudaGraphPoolTouchedDuringCapture.end()) {
        return true;
    }
    return fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) ==
               FASTLLM_CUDA_GRAPH_POOL_CAPTURING &&
           FastllmCudaGraphCaptureIdentityRegisteredLocked(identity);
}

static bool FastllmCudaGraphPoolPointerProtectedLocked(void *ptr) {
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    return fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) !=
               FASTLLM_CUDA_GRAPH_POOL_IDLE &&
           fastllmCudaGraphPoolTouchedDuringCapture.find(ptr) !=
               fastllmCudaGraphPoolTouchedDuringCapture.end();
}

static bool FastllmCudaGraphPoolBeforeFreeLocked(
        void *ptr, const FastllmCudaGraphCaptureIdentity &identity) {
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) ==
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING &&
        FastllmCudaGraphCaptureIdentityRegisteredLocked(identity)) {
        fastllmCudaGraphPoolTouchedDuringCapture.insert(ptr);
    }
    return true;
}

static void FastllmCudaGraphPoolAfterAllocLocked(
        void *ptr, const FastllmCudaGraphCaptureIdentity &identity) {
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) !=
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
        return;
    }
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) ==
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING &&
        FastllmCudaGraphCaptureIdentityRegisteredLocked(identity)) {
        fastllmCudaGraphPoolTouchedDuringCapture.insert(ptr);
    }
}

// 张量并行下 NCCL 集合通信异步发射在 cudaStreamPerThread 上且不立即同步。此时执行真实 cudaMalloc
// 会持有 CUDA 驱动分配锁并隐式同步设备，与 NCCL 主机 proxy 线程争用驱动锁，形成跨 rank 死锁。
// 该标志由 multicuda 在 NCCL 初始化成功后置位；置位后真实 cudaMalloc 前会先排空在途集合通信，
// 使分配发生在任何在途 NCCL 之外。仅在冷池真实分配时触发，热池命中无影响。
std::atomic<bool> fastllmCudaNcclActive(false);
void FastllmCudaSetNcclActive(bool value) {
    fastllmCudaNcclActive.store(value, std::memory_order_relaxed);
}

// 是否要求 NCCL 集合通信「发射后立即同步」。默认 true（安全）：权重加载与 warmup 阶段几乎每次都会
// 发生真实 cudaMalloc，与在途集合通信争用 CUDA 驱动锁会导致跨 rank 死锁，故全程同步发射。
// warmup 成功结束后由 basellm 置为 false：此时内存池已热、稳态前向基本不再有真实 cudaMalloc，
// 异步发射安全且能恢复通信/计算重叠带来的吞吐。malloc 护栏继续作为稳态兜底。
std::atomic<bool> fastllmCudaNcclForceSync(true);
void FastllmCudaSetNcclForceSync(bool value) {
    fastllmCudaNcclForceSync.store(value, std::memory_order_relaxed);
}
bool FastllmCudaGetNcclForceSync() {
    return fastllmCudaNcclForceSync.load(std::memory_order_relaxed);
}

static void FastllmCudaPrintMallocStack(size_t size, const char *file, int line, bool rejected) {
    std::lock_guard<std::mutex> lock(fastllmCudaMallocCheckMutex);
    fprintf(stderr,
            "[FASTLLM_CUDA_MEM_CHECK] cudaMalloc %s size=%zu bytes (%.2f MB) at %s:%d\n",
            rejected ? "rejected" : "called",
            size,
            (double)size / (1024.0 * 1024.0),
            file == nullptr ? "<unknown>" : file,
            line);
#if defined(__linux__) || defined(__APPLE__)
    const int maxFrames = 64;
    void *frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char **symbols = backtrace_symbols(frames, numFrames);
    if (symbols != nullptr) {
        int skip = 2;
        int end = std::min(numFrames, skip + 32);
        for (int i = skip; i < end; i++) {
            fprintf(stderr, "  #%d %s\n", i - skip, symbols[i]);
        }
        free(symbols);
    }
#else
    fprintf(stderr, "  call stack is not available on this platform.\n");
#endif
    fflush(stderr);
}

cudaError_t FastllmCudaCheckedMalloc(void **ret, size_t size, const char *file, int line) {
    bool rejected = fastllmCudaMallocDisabled.load(std::memory_order_relaxed);
    int rejectLogIndex = rejected ?
        fastllmCudaMallocRejectLogCount.fetch_add(1, std::memory_order_relaxed) : -1;
    // A rejected post-startup allocation is always actionable.  Print the first
    // few call stacks even when the verbose startup allocation audit is off, so
    // a frozen server can identify the missing warmup path without flooding the
    // log after the first failure cascades through concurrent requests.
    if (fastllm::GetFastllmEnv().cudaMemCheck ||
        (rejected && rejectLogIndex < 4)) {
        FastllmCudaPrintMallocStack(size, file, line, rejected);
    }
    // The serving frontend arms this freeze only when
    // FASTLLM_CUDA_MEM_CHECK is enabled.
    if (rejected) {
        if (ret != nullptr) {
            *ret = nullptr;
        }
        return cudaErrorMemoryAllocation;
    }
    if (fastllmCudaNcclActive.load(std::memory_order_relaxed)) {
        // 真实 cudaMalloc 前排空在途 NCCL 集合通信，避免与 cudaMalloc 争用 CUDA 驱动锁导致跨 rank 死锁。
        cudaDeviceSynchronize();
    }
    return cudaMalloc(ret, size);
}

void DisableCudaMalloc() {
    if (!fastllm::GetFastllmEnv().cudaMemCheck) {
        return;
    }
    bool wasDisabled = fastllmCudaMallocDisabled.exchange(
        true, std::memory_order_acq_rel);
    if (!wasDisabled) {
        fprintf(stderr,
                "[Fastllm] CUDA allocation frozen; future pool misses will be rejected.\n");
        fflush(stderr);
    }
}

/*
size_t totalMalloc = 0;
std::map <void*, size_t> mallocMap;
std::map <size_t, int> mallocCnt;

template <typename T>
cudaError_t CCMalloc(T **ret, size_t size) {
printf("malloc %f m\n", (double)size / 1e6);
totalMalloc += size;
printf("total malloc %f m\n", (double)totalMalloc / 1e6);
    cudaError_t sta = cudaMalloc(ret, size);
mallocMap[ret[0]] = size;
mallocCnt[size]++;
    return sta;
}

template <typename T>
cudaError_t CCFree(T *ret) {
printf("free %f m\n", (double)mallocMap[ret] / 1e6);
totalMalloc -= mallocMap[ret];
mallocCnt[mallocMap[ret]]--;
printf("total malloc %f m\n", (double)totalMalloc / 1e6);
for (auto &it : mallocCnt) {
    if (it.second > 0) printf("(%f: %d) ", (double)it.first / 1e6, it.second);
}
printf("\n");
    cudaError_t sta = cudaFree(ret);
    return sta;
}
*/

static std::map<int, cublasHandle_t> s_fastllmCublasHandleMap;
static std::mutex s_fastllmCublasHandleMapMutex;
cublasHandle_t getFastllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    // 线程级张量并行时，多个 worker 线程会并发访问该全局 map，
    // 必须加锁，否则并发读写 std::map 会破坏其内部结构导致死循环/崩溃。
    std::lock_guard<std::mutex> guard(s_fastllmCublasHandleMapMutex);
    auto it = s_fastllmCublasHandleMap.find(id);
    if (it != s_fastllmCublasHandleMap.end()) {
        cublasSetStream(it->second, cudaStreamPerThread);
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Error: CUBLAS initialization failed. state %d.\n", stat);
        exit(0);
    } else {
        cublasSetStream(handler, cudaStreamPerThread);
        s_fastllmCublasHandleMap[id] = handler;
    }

    return handler;
}

std::vector <long long> FastllmCudaGetFreeSizes() {
    int deviceCount;
    auto error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return {};
    }
    std::vector <long long> ret;
    
    // 遍历所有设备  
    int id = -1;
    cudaGetDevice(&id);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess) {
            // printf("Device %d: \"%s\"\n", i, prop.name);
            // printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            // printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
            
            // 获取当前设备的显存使用情况  
            cudaSetDevice(i);
            size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            ret.push_back(free);
            // printf("  Free memory: %zu bytes\n", free);
            // printf("  Remaining memory: %zu bytes\n", total - free);
        } else {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        }
    }
    cudaSetDevice(id);
    return ret;
}

long long FastllmCudaGetFreeSize() {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    const cudaError_t error = cudaMemGetInfo(&freeBytes, &totalBytes);
    return error == cudaSuccess ? (long long)freeBytes : 0;
}

std::vector <long long> FastllmCudaGetTotalSizes() {
    int deviceCount;
    auto error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return {};
    }
    std::vector <long long> ret;

    int id = -1;
    cudaGetDevice(&id);

    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        ret.push_back(total);
    }
    cudaSetDevice(id);
    return ret;
}

__global__ void GetCudaInfoKernel(int *infos) {
#if defined(__CUDA_ARCH__)
    infos[0] = __CUDA_ARCH__;
#else
    infos[0] = 0; // cuda arch
#endif
}

CudaInfos::CudaInfos() {
    int infoLen = 10;
    int *infos;
    cudaError_t state = FastllmCudaCheckedMalloc((void **)&infos, infoLen * sizeof(int), __FILE__, __LINE__);
    if (cudaSuccess != state) {
        cudaArch = 0;
        hasTensorCore = false;
        checkCudaErrors("Error: CUDA error when allocating cuda info buffer!", state);
        return;
    }
    GetCudaInfoKernel <<<1, 1>>> (infos);
    int *infosInCpu = new int[infoLen];
    cudaMemcpy(infosInCpu, infos, infoLen * sizeof(int), cudaMemcpyDeviceToHost);

    cudaArch = infosInCpu[0];
    hasTensorCore = cudaArch >= 700;

    cudaFree(infos);
    delete[] infosInCpu;

    printf("CUDA_ARCH: %d\n", cudaArch);
    printf("USE_TENSOR_CORE: %d\n", hasTensorCore);
}

CudaInfos *cudaInfos = nullptr;

CudaInfos *getCudaInfos() {
    if (cudaInfos == nullptr) {
        cudaInfos = new CudaInfos();
    }
    return cudaInfos;
}

static bool FastllmCudaEnvFlagEnabled(const char *name) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    return std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0 &&
           std::strcmp(v, "off") != 0 && std::strcmp(v, "OFF") != 0 &&
           std::strcmp(v, "disable") != 0 && std::strcmp(v, "DISABLE") != 0;
}

bool FastllmCudaPagedDebugEnabled() {
    static const bool enabled =
        FastllmCudaEnvFlagEnabled("FASTLLM_CUDA_PAGED_DEBUG");
    return enabled;
}

/** 输出分页诊断中的单个张量描述，不读取张量内容。 */
static void FastllmCudaPrintPagedDebugTensor(
        const char *name, const fastllm::Data *data) {
    if (data == nullptr) {
        fprintf(stderr, "[fastllm][paged-debug] %s=null\n", name);
        return;
    }

    std::string shape = "[";
    for (size_t i = 0; i < data->dims.size(); i++) {
        if (i > 0) {
            shape += ",";
        }
        shape += std::to_string(data->dims[i]);
    }
    shape += "]";
    int pointerDevice = data->cudaData == nullptr
        ? -1 : GetPointerDeviceId(data->cudaData);
    fprintf(stderr,
            "[fastllm][paged-debug] %s dtype=%s shape=%s cuda=%p "
            "pointer_device=%d data_device=%d unit=%d\n",
            name, fastllm::GetDataTypeName(data->dataType).c_str(),
            shape.c_str(), data->cudaData, pointerDevice,
            (int)data->dataDevice, data->unitSize);
}

bool FastllmCudaPagedDebugSynchronize(
        const char *stage, int layer, const fastllm::Data *first,
        const fastllm::Data *second) {
    if (!FastllmCudaPagedDebugEnabled()) {
        return true;
    }

    int device = -1;
    cudaGetDevice(&device);
    cudaError_t pending = cudaPeekAtLastError();
    cudaError_t synchronized = cudaDeviceSynchronize();
    cudaError_t state = pending != cudaSuccess ? pending : synchronized;
    if (state == cudaSuccess) {
        return true;
    }

    FastllmCudaSetThreadError();
    fprintf(stderr,
            "[fastllm][paged-debug] FAILED stage=%s layer=%d device=%d "
            "pending=%d(%s) synchronize=%d(%s)\n",
            stage == nullptr ? "unknown" : stage, layer, device,
            (int)pending, cudaGetErrorName(pending),
            (int)synchronized, cudaGetErrorName(synchronized));
    FastllmCudaPrintPagedDebugTensor("first", first);
    FastllmCudaPrintPagedDebugTensor("second", second);
    fflush(stderr);
    return false;
}

bool FastllmCudaFlashInferSupported() {
    static thread_local std::map<int, bool> supportedByDevice;
    static thread_local std::map<int, bool> loggedByDevice;
    static thread_local std::map<int, bool> forceNativeLoggedByDevice;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return false;
    }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        supportedByDevice[dev] = false;
        return false;
    }
    if (FastllmCudaEnvFlagEnabled("FASTLLM_FORCE_NATIVE_ATTN")) {
        if (forceNativeLoggedByDevice.find(dev) == forceNativeLoggedByDevice.end()) {
            printf("[Fastllm] FlashInfer attention disabled by FASTLLM_FORCE_NATIVE_ATTN on GPU %d (CC %d.%d), using native attention.\n",
                   dev, major, minor);
            forceNativeLoggedByDevice[dev] = true;
        }
        return false;
    }
    auto it = supportedByDevice.find(dev);
    if (it != supportedByDevice.end()) {
        return it->second;
    }
    bool supported = major * 10 + minor >= 75;
    supportedByDevice[dev] = supported;
    if (!supported && loggedByDevice.find(dev) == loggedByDevice.end()) {
        printf("[Fastllm] FlashInfer attention disabled on GPU %d (CC %d.%d), using native attention.\n",
               dev, major, minor);
        loggedByDevice[dev] = true;
    }
    return supported;
}

bool FastllmCudaFlashInferDataTypeSupported(fastllm::DataType dataType) {
    if (!FastllmCudaFlashInferSupported()) {
        return false;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        return true;
    }
    if (dataType != fastllm::DataType::BFLOAT16) {
        return false;
    }

    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return false;
    }
    return major * 10 + minor >= 80;
}

void DeviceSync() {
    if (fastllm::GetFastllmEnv().cudaSync) {
        cudaDeviceSynchronize();
    }
}

void ForceDeviceSync() {
    cudaError_t state = cudaDeviceSynchronize();
    checkCudaErrors("Error: CUDA error when synchronizing device!", state);
}

void FastllmCudaSyncCurrentThreadStream() {
    cudaError_t state = cudaStreamSynchronize(cudaStreamPerThread);
    checkCudaErrors("Error: CUDA error when synchronizing the per-thread stream!", state);
}

void *FastllmCudaStreamCreate(bool nonBlocking) {
    cudaStream_t stream;
    unsigned int flags = nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
    cudaError_t state = cudaStreamCreateWithFlags(&stream, flags);
    checkCudaErrors("Error: CUDA error when creating stream!", state);
    return (void*)stream;
}

void FastllmCudaStreamDestroy(void *stream) {
    cudaError_t state = cudaStreamDestroy((cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when destroying stream!", state);
}

void FastllmCudaStreamSynchronize(void *stream) {
    cudaError_t state = cudaStreamSynchronize((cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when synchronizing stream!", state);
}

void *FastllmCudaEventCreate() {
    cudaEvent_t event;
    cudaError_t state = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    checkCudaErrors("Error: CUDA error when creating event!", state);
    return (void*)event;
}

void *FastllmCudaEventCreateTiming() {
    cudaEvent_t event;
    cudaError_t state = cudaEventCreate(&event);
    checkCudaErrors("Error: CUDA error when creating timing event!", state);
    return (void*)event;
}

void FastllmCudaEventDestroy(void *event) {
    cudaError_t state = cudaEventDestroy((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when destroying event!", state);
}

void FastllmCudaEventRecord(void *event, void *stream) {
    cudaError_t state = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when recording event!", state);
}

void FastllmCudaEventRecordCurrentThread(void *event) {
    cudaError_t state = cudaEventRecord((cudaEvent_t)event, cudaStreamPerThread);
    checkCudaErrors("Error: CUDA error when recording event on the per-thread stream!", state);
}

void FastllmCudaEventSynchronize(void *event) {
    cudaError_t state = cudaEventSynchronize((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when synchronizing event!", state);
}

float FastllmCudaEventElapsedTime(void *start, void *end) {
    float elapsedMs = 0.0f;
    cudaError_t state = cudaEventElapsedTime(
        &elapsedMs, (cudaEvent_t)start, (cudaEvent_t)end);
    checkCudaErrors("Error: CUDA error when measuring event time!", state);
    return elapsedMs;
}

void FastllmCudaStreamWaitEvent(void *stream, void *event) {
    cudaError_t state = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
    checkCudaErrors("Error: CUDA error when stream waiting event!", state);
}

void FastllmCudaCurrentThreadStreamWaitEvent(void *event) {
    cudaError_t state = cudaStreamWaitEvent(cudaStreamPerThread, (cudaEvent_t)event, 0);
    checkCudaErrors("Error: CUDA error when the per-thread stream is waiting for event!", state);
}

static thread_local std::string fastllmCudaGraphLastError;

static bool FastllmCudaGraphSetError(const char *stage, cudaError_t err) {
    if (err == cudaSuccess) {
        fastllmCudaGraphLastError.clear();
        return true;
    }
    fastllmCudaGraphLastError = std::string(stage) + ": " + cudaGetErrorString(err);
    return false;
}

bool FastllmCudaGraphBeginCapture() {
    cudaError_t state = cudaSuccess;
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) ==
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
        // Allocate the process-lifetime failure address before capture; even a
        // tiny cudaMalloc is forbidden once stream capture has begun.
        state = FastllmCudaGraphPrepareAllocationFailurePlaceholder();
        if (state != cudaSuccess) {
            return FastllmCudaGraphSetError(
                "prepare allocation-failure placeholder", state);
        }
    }
    state = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    bool ok = FastllmCudaGraphSetError("cudaStreamBeginCapture", state);
    if (ok && fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) ==
                  FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
        FastllmCudaGraphRegisterCurrentCapture();
    }
    return ok;
}

bool FastllmCudaGraphPrepareCaptureDevice() {
    return FastllmCudaGraphSetError(
        "prepare allocation-failure placeholder",
        FastllmCudaGraphPrepareAllocationFailurePlaceholder());
}

// 检查当前线程的捕获是否已被 invalidate（捕获体内任何算子报错都会导致失效）。
// 不通过 showError 上报的错误路径（如 FlashInfer、独立 kernel 封装内的 printf）
// 也会 invalidate 捕获，因此该检查是错误标志之外的兜底。
bool FastllmCudaGraphCaptureInvalidated() {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return true;
    }
    return captureStatus == cudaStreamCaptureStatusInvalidated;
}

bool FastllmCudaGraphIsCapturing() {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return captureStatus != cudaStreamCaptureStatusNone;
}

bool FastllmCudaGraphGetAllocationFailurePlaceholder(void **ptr) {
    if (ptr == nullptr) {
        return false;
    }
    *ptr = nullptr;
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) !=
            FASTLLM_CUDA_GRAPH_POOL_CAPTURING ||
        !FastllmCudaGetThreadError() || !FastllmCudaGraphIsCapturing()) {
        return false;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    std::lock_guard<std::mutex> guard(
        fastllmCudaGraphAllocationFailurePlaceholderMutex);
    auto it = fastllmCudaGraphAllocationFailurePlaceholders.find(device);
    if (it == fastllmCudaGraphAllocationFailurePlaceholders.end()) {
        return false;
    }
    *ptr = it->second;
    return *ptr != nullptr;
}

namespace {
    static thread_local bool fastllmCudaGraphQwen35MoeMarkersEnabled = true;

    __global__ void FastllmCudaQwen35MoeForkMarkerKernel(int layer) {
        (void)layer;
    }

    __global__ void FastllmCudaQwen35MoeSharedDoneMarkerKernel(int layer) {
        (void)layer;
    }

    __global__ void FastllmCudaQwen35MoeRoutedBeginMarkerKernel(int layer) {
        (void)layer;
    }

    __global__ void FastllmCudaQwen35MoeJoinMarkerKernel(int layer) {
        (void)layer;
    }

    static bool FastllmCudaGraphIsCapturingCurrentThread() {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        return cudaStreamIsCapturing(cudaStreamPerThread, &status) == cudaSuccess &&
               status == cudaStreamCaptureStatusActive;
    }

    static bool FastllmCudaGraphGetAdjacentNodes(cudaGraphNode_t node, bool dependents,
                                                 std::vector<cudaGraphNode_t> &result) {
        size_t count = 0;
        cudaError_t state;
        // CUDA 12.3-12.x exposes edge-data queries as *_v2; CUDA 13 promotes
        // those signatures to the unversioned runtime APIs.
#if CUDART_VERSION >= 13000
        state = dependents
            ? cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &count)
            : cudaGraphNodeGetDependencies(node, nullptr, nullptr, &count);
#elif CUDART_VERSION >= 12030
        state = dependents
            ? cudaGraphNodeGetDependentNodes_v2(node, nullptr, nullptr, &count)
            : cudaGraphNodeGetDependencies_v2(node, nullptr, nullptr, &count);
#else
        state = dependents
            ? cudaGraphNodeGetDependentNodes(node, nullptr, &count)
            : cudaGraphNodeGetDependencies(node, nullptr, &count);
#endif
        if (state != cudaSuccess) {
            FastllmCudaGraphSetError(dependents ? "cudaGraphNodeGetDependentNodes(count)"
                                                : "cudaGraphNodeGetDependencies(count)", state);
            return false;
        }
        result.resize(count);
        if (count == 0) {
            return true;
        }
#if CUDART_VERSION >= 12030
        std::vector<cudaGraphEdgeData> edgeData(count);
#if CUDART_VERSION >= 13000
        state = dependents
            ? cudaGraphNodeGetDependentNodes(node, result.data(), edgeData.data(), &count)
            : cudaGraphNodeGetDependencies(node, result.data(), edgeData.data(), &count);
#else
        state = dependents
            ? cudaGraphNodeGetDependentNodes_v2(node, result.data(), edgeData.data(), &count)
            : cudaGraphNodeGetDependencies_v2(node, result.data(), edgeData.data(), &count);
#endif
#else
        state = dependents
            ? cudaGraphNodeGetDependentNodes(node, result.data(), &count)
            : cudaGraphNodeGetDependencies(node, result.data(), &count);
#endif
        if (state != cudaSuccess) {
            FastllmCudaGraphSetError(dependents ? "cudaGraphNodeGetDependentNodes"
                                                : "cudaGraphNodeGetDependencies", state);
            return false;
        }
        result.resize(count);
        return true;
    }

    static cudaError_t FastllmCudaGraphAddDefaultDependencies(
            cudaGraph_t graph, const std::vector<cudaGraphNode_t> &from,
            const std::vector<cudaGraphNode_t> &to) {
        if (from.empty()) {
            return cudaSuccess;
        }
#if CUDART_VERSION >= 13000
        return cudaGraphAddDependencies(graph, from.data(), to.data(), nullptr, from.size());
#elif CUDART_VERSION >= 12030
        return cudaGraphAddDependencies_v2(graph, from.data(), to.data(), nullptr, from.size());
#else
        return cudaGraphAddDependencies(graph, from.data(), to.data(), from.size());
#endif
    }

    struct FastllmCudaQwen35MoeMarkerNodes {
        cudaGraphNode_t fork = nullptr;
        cudaGraphNode_t sharedDone = nullptr;
        cudaGraphNode_t routedBegin = nullptr;
        cudaGraphNode_t join = nullptr;
    };
}

bool FastllmCudaGraphSetQwen35MoeMarkersEnabled(bool enabled) {
    bool previous = fastllmCudaGraphQwen35MoeMarkersEnabled;
    fastllmCudaGraphQwen35MoeMarkersEnabled = enabled;
    return previous;
}

void FastllmCudaGraphMarkQwen35MoeFork(int layer) {
    if (fastllmCudaGraphQwen35MoeMarkersEnabled &&
        FastllmCudaGraphIsCapturingCurrentThread()) {
        FastllmCudaQwen35MoeForkMarkerKernel<<<1, 1, 0, cudaStreamPerThread>>>(layer);
    }
}

void FastllmCudaGraphMarkQwen35MoeSharedDone(int layer) {
    if (fastllmCudaGraphQwen35MoeMarkersEnabled &&
        FastllmCudaGraphIsCapturingCurrentThread()) {
        FastllmCudaQwen35MoeSharedDoneMarkerKernel<<<1, 1, 0, cudaStreamPerThread>>>(layer);
    }
}

void FastllmCudaGraphMarkQwen35MoeRoutedBegin(int layer) {
    if (fastllmCudaGraphQwen35MoeMarkersEnabled &&
        FastllmCudaGraphIsCapturingCurrentThread()) {
        FastllmCudaQwen35MoeRoutedBeginMarkerKernel<<<1, 1, 0, cudaStreamPerThread>>>(layer);
    }
}

void FastllmCudaGraphMarkQwen35MoeJoin(int layer) {
    if (fastllmCudaGraphQwen35MoeMarkersEnabled &&
        FastllmCudaGraphIsCapturingCurrentThread()) {
        FastllmCudaQwen35MoeJoinMarkerKernel<<<1, 1, 0, cudaStreamPerThread>>>(layer);
    }
}

int FastllmCudaGraphOptimizeQwen35Moe(void *graph) {
    if (graph == nullptr) {
        fastllmCudaGraphLastError = "Qwen3.5 MoE graph optimizer received a null graph";
        return -1;
    }
    cudaGraph_t cudaGraph = (cudaGraph_t)graph;
    size_t nodeCount = 0;
    cudaError_t state = cudaGraphGetNodes(cudaGraph, nullptr, &nodeCount);
    if (!FastllmCudaGraphSetError("cudaGraphGetNodes(count)", state)) {
        return -1;
    }
    std::vector<cudaGraphNode_t> nodes(nodeCount);
    state = cudaGraphGetNodes(cudaGraph, nodes.data(), &nodeCount);
    if (!FastllmCudaGraphSetError("cudaGraphGetNodes", state)) {
        return -1;
    }
    nodes.resize(nodeCount);

    // CUDA forbids destroying nodes in a graph containing allocation/free
    // nodes. Detect that structural restriction before adding dependencies so
    // the caller can safely discard this untouched graph and recapture without
    // markers.
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        state = cudaGraphNodeGetType(node, &type);
        if (!FastllmCudaGraphSetError("cudaGraphNodeGetType(memory preflight)", state)) {
            return -1;
        }
#if CUDART_VERSION >= 11040
        if (type == cudaGraphNodeTypeMemAlloc ||
            type == cudaGraphNodeTypeMemFree) {
            fastllmCudaGraphLastError.clear();
            return FASTLLM_CUDA_GRAPH_MOE_RECAPTURE_WITHOUT_MARKERS;
        }
#endif
    }

    std::map<int, FastllmCudaQwen35MoeMarkerNodes> markers;
    size_t markerCount = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type;
        state = cudaGraphNodeGetType(node, &type);
        if (!FastllmCudaGraphSetError("cudaGraphNodeGetType", state)) {
            return -1;
        }
        if (type != cudaGraphNodeTypeKernel) {
            continue;
        }
        cudaKernelNodeParams params = {};
        state = cudaGraphKernelNodeGetParams(node, &params);
        // NCCL and runtime-loaded module kernels can be captured into the same
        // graph, but CUDA 13 does not always expose their function handles to
        // cudaGraphKernelNodeGetParams.  They cannot be one of our statically
        // linked marker kernels, so skip just those opaque nodes and clear the
        // runtime's sticky error before inspecting the remaining graph.
        if (state == cudaErrorInvalidDeviceFunction) {
            cudaGetLastError();
            continue;
        }
        if (!FastllmCudaGraphSetError("cudaGraphKernelNodeGetParams", state)) {
            return -1;
        }
        cudaGraphNode_t FastllmCudaQwen35MoeMarkerNodes::*slot = nullptr;
        if (params.func == (void*)FastllmCudaQwen35MoeForkMarkerKernel) {
            slot = &FastllmCudaQwen35MoeMarkerNodes::fork;
        } else if (params.func == (void*)FastllmCudaQwen35MoeSharedDoneMarkerKernel) {
            slot = &FastllmCudaQwen35MoeMarkerNodes::sharedDone;
        } else if (params.func == (void*)FastllmCudaQwen35MoeRoutedBeginMarkerKernel) {
            slot = &FastllmCudaQwen35MoeMarkerNodes::routedBegin;
        } else if (params.func == (void*)FastllmCudaQwen35MoeJoinMarkerKernel) {
            slot = &FastllmCudaQwen35MoeMarkerNodes::join;
        }
        if (slot == nullptr) {
            continue;
        }
        if (params.kernelParams == nullptr || params.kernelParams[0] == nullptr) {
            fastllmCudaGraphLastError = "Qwen3.5 MoE graph marker has no layer argument";
            return -1;
        }
        int layer = *(int*)params.kernelParams[0];
        auto &layerMarkers = markers[layer];
        if (layerMarkers.*slot != nullptr) {
            fastllmCudaGraphLastError = "Qwen3.5 MoE graph contains duplicate markers for layer " +
                                        std::to_string(layer);
            return -1;
        }
        layerMarkers.*slot = node;
        markerCount++;
    }

    if (markerCount == 0) {
        fastllmCudaGraphLastError.clear();
        return 0;
    }
    if (markerCount != markers.size() * 4) {
        fastllmCudaGraphLastError = "Qwen3.5 MoE graph has an incomplete marker set";
        return -1;
    }

    std::set<std::pair<cudaGraphNode_t, cudaGraphNode_t> > desiredEdges;
    std::vector<cudaGraphNode_t> markerNodes;
    markerNodes.reserve(markerCount);
    for (auto &it : markers) {
        auto &m = it.second;
        if (m.fork == nullptr || m.sharedDone == nullptr ||
            m.routedBegin == nullptr || m.join == nullptr) {
            fastllmCudaGraphLastError = "Qwen3.5 MoE graph is missing a marker for layer " +
                                        std::to_string(it.first);
            return -1;
        }
        std::vector<cudaGraphNode_t> forkDependencies, sharedRoots;
        std::vector<cudaGraphNode_t> sharedTails, routedRoots;
        std::vector<cudaGraphNode_t> routedTails, joinRoots;
        if (!FastllmCudaGraphGetAdjacentNodes(m.fork, false, forkDependencies) ||
            !FastllmCudaGraphGetAdjacentNodes(m.fork, true, sharedRoots) ||
            !FastllmCudaGraphGetAdjacentNodes(m.sharedDone, false, sharedTails) ||
            !FastllmCudaGraphGetAdjacentNodes(m.routedBegin, true, routedRoots) ||
            !FastllmCudaGraphGetAdjacentNodes(m.join, false, routedTails) ||
            !FastllmCudaGraphGetAdjacentNodes(m.join, true, joinRoots)) {
            return -1;
        }
        if (forkDependencies.empty() || sharedRoots.empty() || sharedTails.empty() ||
            routedRoots.empty() || routedTails.empty() || joinRoots.empty()) {
            fastllmCudaGraphLastError = "Qwen3.5 MoE graph marker adjacency is empty for layer " +
                                        std::to_string(it.first);
            return -1;
        }
        for (cudaGraphNode_t dependency : forkDependencies) {
            for (cudaGraphNode_t root : sharedRoots) {
                desiredEdges.insert({dependency, root});
            }
            for (cudaGraphNode_t root : routedRoots) {
                desiredEdges.insert({dependency, root});
            }
        }
        for (cudaGraphNode_t tail : sharedTails) {
            for (cudaGraphNode_t root : joinRoots) {
                desiredEdges.insert({tail, root});
            }
        }
        for (cudaGraphNode_t tail : routedTails) {
            for (cudaGraphNode_t root : joinRoots) {
                desiredEdges.insert({tail, root});
            }
        }
        markerNodes.push_back(m.fork);
        markerNodes.push_back(m.sharedDone);
        markerNodes.push_back(m.routedBegin);
        markerNodes.push_back(m.join);
    }

    std::vector<cudaGraphNode_t> from;
    std::vector<cudaGraphNode_t> to;
    from.reserve(desiredEdges.size());
    to.reserve(desiredEdges.size());
    for (auto &edge : desiredEdges) {
        from.push_back(edge.first);
        to.push_back(edge.second);
    }
    state = FastllmCudaGraphAddDefaultDependencies(cudaGraph, from, to);
    if (!FastllmCudaGraphSetError("cudaGraphAddDependencies(Qwen3.5 MoE)", state)) {
        return -1;
    }
    for (cudaGraphNode_t marker : markerNodes) {
        state = cudaGraphDestroyNode(marker);
        if (!FastllmCudaGraphSetError("cudaGraphDestroyNode(Qwen3.5 MoE marker)", state)) {
            return -1;
        }
    }
    fastllmCudaGraphLastError.clear();
    return (int)markers.size();
}

namespace {
    __global__ void FastllmCudaQwen35MoeSelfTestBranchKernel(int *output, int value) {
        if (threadIdx.x == 0) {
            *output = value;
        }
    }

    __global__ void FastllmCudaQwen35MoeSelfTestJoinKernel(
            const int *shared, const int *routed, int *output) {
        if (threadIdx.x == 0) {
            *output = *shared + *routed;
        }
    }
}

static bool FastllmCudaGraphQwen35MoeParallelSelfTest() {
    int *shared = nullptr;
    int *routed = nullptr;
    int *output = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    auto cleanup = [&]() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (shared != nullptr) cudaFree(shared);
        if (routed != nullptr) cudaFree(routed);
        if (output != nullptr) cudaFree(output);
    };
    if (cudaMalloc(&shared, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&routed, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&output, sizeof(int)) != cudaSuccess) {
        cleanup();
        return false;
    }
    if (!FastllmCudaGraphBeginCapture()) {
        cleanup();
        return false;
    }
    cudaMemsetAsync(shared, 0, sizeof(int), cudaStreamPerThread);
    cudaMemsetAsync(routed, 0, sizeof(int), cudaStreamPerThread);
    cudaMemsetAsync(output, 0, sizeof(int), cudaStreamPerThread);
    FastllmCudaGraphMarkQwen35MoeFork(0);
    FastllmCudaQwen35MoeSelfTestBranchKernel<<<1, 1, 0, cudaStreamPerThread>>>(shared, 11);
    FastllmCudaGraphMarkQwen35MoeSharedDone(0);
    FastllmCudaGraphMarkQwen35MoeRoutedBegin(0);
    FastllmCudaQwen35MoeSelfTestBranchKernel<<<1, 1, 0, cudaStreamPerThread>>>(routed, 31);
    FastllmCudaGraphMarkQwen35MoeJoin(0);
    FastllmCudaQwen35MoeSelfTestJoinKernel<<<1, 1, 0, cudaStreamPerThread>>>(
        shared, routed, output);
    void *capturedGraph = nullptr;
    if (!FastllmCudaGraphEndCapture(&capturedGraph) || capturedGraph == nullptr) {
        cleanup();
        return false;
    }
    graph = (cudaGraph_t)capturedGraph;
    if (FastllmCudaGraphOptimizeQwen35Moe(graph) != 1 ||
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(exec, cudaStreamPerThread) != cudaSuccess ||
        cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        cleanup();
        return false;
    }
    int result = 0;
    bool ok = cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess &&
              result == 42;
    cleanup();
    return ok;
}

static bool FastllmCudaGraphQwen35MoeAllocationFallbackSelfTest() {
#if CUDART_VERSION < 11040
    return true;
#else
    int device = 0;
    int driverVersion = 0;
    int memoryPoolsSupported = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDriverGetVersion(&driverVersion) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (driverVersion < 11040) {
        return true;
    }
    if (cudaDeviceGetAttribute(&memoryPoolsSupported,
                               cudaDevAttrMemoryPoolsSupported,
                               device) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (!memoryPoolsSupported) {
        return true;
    }

    int *shared = nullptr;
    int *routed = nullptr;
    int *output = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    auto cleanup = [&]() {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (shared != nullptr) cudaFree(shared);
        if (routed != nullptr) cudaFree(routed);
        if (output != nullptr) cudaFree(output);
    };
    if (cudaMalloc(&shared, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&routed, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&output, sizeof(int)) != cudaSuccess) {
        cleanup();
        return false;
    }

    auto capture = [&](bool enableMarkers, cudaGraph_t &captured) {
        bool previousMarkers =
            FastllmCudaGraphSetQwen35MoeMarkersEnabled(enableMarkers);
        void *temporary = nullptr;
        void *capturedGraph = nullptr;
        bool began = FastllmCudaGraphBeginCapture();
        bool ok = began;
        if (ok) {
            ok = cudaMallocAsync(&temporary, 4096, cudaStreamPerThread) ==
                     cudaSuccess;
        }
        if (ok) {
            cudaMemsetAsync(shared, 0, sizeof(int), cudaStreamPerThread);
            cudaMemsetAsync(routed, 0, sizeof(int), cudaStreamPerThread);
            cudaMemsetAsync(output, 0, sizeof(int), cudaStreamPerThread);
            cudaMemsetAsync(temporary, 0, 4096, cudaStreamPerThread);
            FastllmCudaGraphMarkQwen35MoeFork(0);
            FastllmCudaQwen35MoeSelfTestBranchKernel<<<1, 1, 0, cudaStreamPerThread>>>(shared, 11);
            FastllmCudaGraphMarkQwen35MoeSharedDone(0);
            FastllmCudaGraphMarkQwen35MoeRoutedBegin(0);
            FastllmCudaQwen35MoeSelfTestBranchKernel<<<1, 1, 0, cudaStreamPerThread>>>(routed, 31);
            FastllmCudaGraphMarkQwen35MoeJoin(0);
            FastllmCudaQwen35MoeSelfTestJoinKernel<<<1, 1, 0, cudaStreamPerThread>>>(
                shared, routed, output);
            ok = cudaFreeAsync(temporary, cudaStreamPerThread) == cudaSuccess;
        }
        bool ended = began && FastllmCudaGraphEndCapture(&capturedGraph);
        FastllmCudaGraphSetQwen35MoeMarkersEnabled(previousMarkers);
        if (!ok || !ended || capturedGraph == nullptr) {
            if (capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
            cudaGetLastError();
            return false;
        }
        captured = (cudaGraph_t)capturedGraph;
        return true;
    };

    // The first graph contains both markers and graph allocation/free nodes.
    // The optimizer must reject it before mutating any dependencies.
    if (!capture(true, graph) ||
        FastllmCudaGraphOptimizeQwen35Moe(graph) !=
            FASTLLM_CUDA_GRAPH_MOE_RECAPTURE_WITHOUT_MARKERS) {
        cleanup();
        return false;
    }
    cudaGraphDestroy(graph);
    graph = nullptr;

    // A markerless recapture preserves the original sequential MoE topology
    // and remains a valid executable graph even with allocation/free nodes.
    if (!capture(false, graph) ||
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(exec, cudaStreamPerThread) != cudaSuccess ||
        cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        cleanup();
        return false;
    }
    int result = 0;
    bool ok = cudaMemcpy(&result, output, sizeof(int), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              result == 42;
    cleanup();
    return ok;
#endif
}

bool FastllmCudaGraphQwen35MoeSelfTest() {
    return FastllmCudaGraphQwen35MoeParallelSelfTest() &&
           FastllmCudaGraphQwen35MoeAllocationFallbackSelfTest();
}

bool FastllmCudaGraphEndCapture(void **graph) {
    cudaGraph_t cudaGraph = nullptr;
    cudaError_t state = cudaStreamEndCapture(cudaStreamPerThread, &cudaGraph);
    if (graph != nullptr) {
        *graph = (void*)cudaGraph;
    }
    return FastllmCudaGraphSetError("cudaStreamEndCapture", state);
}

bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    cudaGraphExec_t cudaExec = nullptr;
    cudaError_t state = cudaGraphInstantiate(&cudaExec, (cudaGraph_t)graph, nullptr, nullptr, 0);
    if (exec != nullptr) {
        *exec = (void*)cudaExec;
    }
    return FastllmCudaGraphSetError("cudaGraphInstantiate", state);
}

bool FastllmCudaTensorParallelGreedyGatherGraphCreate(
        int rootDevice, int ranks,
        const void *const *ids, const void *const *scores,
        void *cudaGather, void *hostGather,
        size_t rankBytes, size_t scoreBase, size_t totalBytes,
        void **exec) {
    if (exec != nullptr) {
        *exec = nullptr;
    }
    if (rootDevice < 0 || ranks <= 0 || ids == nullptr || scores == nullptr ||
        cudaGather == nullptr || hostGather == nullptr || rankBytes == 0 ||
        totalBytes == 0 || exec == nullptr) {
        return FastllmCudaGraphSetError(
            "greedy gather graph arguments", cudaErrorInvalidValue);
    }

    cudaError_t state = cudaSetDevice(rootDevice);
    if (state != cudaSuccess) {
        return FastllmCudaGraphSetError(
            "greedy gather graph cudaSetDevice", state);
    }

    cudaGraph_t graph = nullptr;
    state = cudaGraphCreate(&graph, 0);
    if (state != cudaSuccess) {
        return FastllmCudaGraphSetError(
            "greedy gather graph cudaGraphCreate", state);
    }

    std::vector<cudaGraphNode_t> copyNodes;
    copyNodes.reserve((size_t)2 * ranks);
    for (int rank = 0; rank < ranks && state == cudaSuccess; ++rank) {
        if (ids[rank] == nullptr || scores[rank] == nullptr) {
            state = cudaErrorInvalidValue;
            break;
        }
        cudaGraphNode_t idNode = nullptr;
        state = cudaGraphAddMemcpyNode1D(
            &idNode, graph, nullptr, 0,
            (uint8_t*)cudaGather + (size_t)rank * rankBytes,
            ids[rank], rankBytes, cudaMemcpyDefault);
        if (state == cudaSuccess) {
            copyNodes.push_back(idNode);
        }
        cudaGraphNode_t scoreNode = nullptr;
        if (state == cudaSuccess) {
            state = cudaGraphAddMemcpyNode1D(
                &scoreNode, graph, nullptr, 0,
                (uint8_t*)cudaGather + scoreBase +
                    (size_t)rank * rankBytes,
                scores[rank], rankBytes, cudaMemcpyDefault);
        }
        if (state == cudaSuccess) {
            copyNodes.push_back(scoreNode);
        }
    }
    if (state != cudaSuccess) {
        cudaGraphDestroy(graph);
        return FastllmCudaGraphSetError(
            "greedy gather graph cudaGraphAddMemcpyNode1D(peer)", state);
    }

    cudaGraphNode_t hostNode = nullptr;
    state = cudaGraphAddMemcpyNode1D(
        &hostNode, graph, copyNodes.data(), copyNodes.size(),
        hostGather, cudaGather, totalBytes, cudaMemcpyDefault);
    if (state != cudaSuccess) {
        cudaGraphDestroy(graph);
        return FastllmCudaGraphSetError(
            "greedy gather graph cudaGraphAddMemcpyNode1D(host)", state);
    }

    cudaGraphExec_t cudaExec = nullptr;
    state = cudaGraphInstantiate(&cudaExec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (state != cudaSuccess) {
        return FastllmCudaGraphSetError(
            "greedy gather graph cudaGraphInstantiate", state);
    }
    *exec = (void*)cudaExec;
    return FastllmCudaGraphSetError(
        "greedy gather graph create", cudaSuccess);
}

bool FastllmCudaGraphLaunch(void *exec) {
    cudaError_t state = cudaGraphLaunch((cudaGraphExec_t)exec, cudaStreamPerThread);
    return FastllmCudaGraphSetError("cudaGraphLaunch", state);
}

void FastllmCudaGraphDestroy(void *graph) {
    if (graph != nullptr) {
        cudaGraphDestroy((cudaGraph_t)graph);
    }
}

void FastllmCudaGraphExecDestroy(void *exec) {
    if (exec != nullptr) {
        cudaGraphExecDestroy((cudaGraphExec_t)exec);
    }
}

const char *FastllmCudaGraphLastError() {
    return fastllmCudaGraphLastError.c_str();
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
};

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmCudaFloatEmbeddingKernel(float *input, T *weight, T *output, int embSize) {
    input += blockIdx.x;
    output += (int64_t)blockIdx.x * embSize;
    int token = (int)(input[0] + 1e-5);
    weight += (int64_t)token * embSize;
    for (int i = threadIdx.x; i < embSize; i+= THREAD_PER_BLOCK) {
        output[i] = weight[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmCudaFloatEmbeddingVectorKernel(
        const float *input, const uint4 *weight, uint4 *output,
        int vectorsPerRow) {
    int row = blockIdx.x;
    int token = (int)(input[row] + 1e-5f);
    const uint4 *src = weight + (int64_t)token * vectorsPerRow;
    uint4 *dst = output + (int64_t)row * vectorsPerRow;
    int vector = blockIdx.y * THREAD_PER_BLOCK + threadIdx.x;
    int stride = gridDim.y * THREAD_PER_BLOCK;
    for (; vector < vectorsPerRow; vector += stride) {
        dst[vector] = src[vector];
    }
}

template <typename T>
static bool FastllmLaunchFloatEmbeddingVector(
        const float *input, const T *weight, T *output,
        uint64_t inputLen, int embSize) {
    size_t rowBytes = (size_t)embSize * sizeof(T);
    if (inputLen == 0 || rowBytes == 0 ||
        rowBytes % sizeof(uint4) != 0 ||
        ((uintptr_t)weight % alignof(uint4)) != 0 ||
        ((uintptr_t)output % alignof(uint4)) != 0) {
        return false;
    }
    int vectorsPerRow = (int)(rowBytes / sizeof(uint4));
    int blocksPerRow = std::min(
        8, (vectorsPerRow + 128 - 1) / 128);
    dim3 grid((unsigned int)inputLen, (unsigned int)blocksPerRow);
    FastllmCudaFloatEmbeddingVectorKernel<128><<<grid, 128>>>(
        input, (const uint4*)weight, (uint4*)output, vectorsPerRow);
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmCudaBFloat16EmbeddingToFloatKernel(
        const float *input, const __nv_bfloat16 *weight, float *output,
        int embSize) {
    int token = (int)(input[blockIdx.x] + 1e-5f);
    weight += (int64_t)token * embSize;
    output += (int64_t)blockIdx.x * embSize;
    for (int i = threadIdx.x; i < embSize; i += THREAD_PER_BLOCK) {
        output[i] = __bfloat162float(weight[i]);
    }
}

__global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half_rz(a[idx]);
    }
}

__global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void FastllmCudaBF162FloatKernel(uint16_t* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        ((uint32_t*)b)[idx] = a[idx] << 16;
    }
}

__global__ void FastllmCudaFloat2Bf16Kernel(float* a, __nv_bfloat16* b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2bfloat16_rn(a[idx]);
    }
}

__global__ void FastllmCudaBF162HalfKernel(uint16_t* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        uint32_t val = (uint32_t)a[idx] << 16;
        float f = __uint_as_float(val);
        b[idx] = __float2half_rz(f);
    }
}

__global__ void FastllmCudaHalf2BF16Kernel(half* a, __nv_bfloat16 *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2bfloat16_rn(__half2float(a[idx]));
    }
}

__global__ void FastllmCudaBiasKernel(__nv_bfloat16* a, __nv_bfloat16* bias, int k) {
    __nv_bfloat16* now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] = __float2bfloat16_rn(__bfloat162float(now[i]) + __bfloat162float(bias[i]));
    }
}

__global__ void FastllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

__global__ void FastllmCudaBiasKernel(half *a, half *bias, int k) {
    half *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
#ifdef CUDA_NO_TENSOR_CORE
        now[i] = __float2half(__half2float(now[i]) + __half2float(bias[i]));
#else
        now[i] = __hadd(now[i], bias[i]);
#endif
    }
}

__global__ void FastllmReluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x > 0 ? x : 0;
    }
}

__global__ void FastllmExpKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = exp((double)x);
    }
}

__global__ void FastllmExpKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __half2float(a[idx]);
        b[idx] = __float2half(exp((double)x));
    }
}

__global__ void FastllmGeluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x * 0.5f * (1.0f + erff(x / 1.41421));
    }
}

__global__ void FastllmGeluKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __half2float(a[idx]);
        b[idx] = __float2half(x * 0.5f * (1.0f + erff(x / 1.41421)));
    }
}

__global__ void FastllmGeluNewKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void FastllmSiluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

__global__ void FastllmSiluKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[idx]);
        b[idx] = __float2half((x / (1.0 + expf(-x))));
#else
        half x = a[idx];
        b[idx] = __hdiv(x, __hadd(__float2half(1.0), hexp(-x)));
#endif
    }
}

__global__ void FastllmSigmoidKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 1.0 / (1.0 + expf(-x));
    }
}

__global__ void FastllmSigmoidKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[idx]);
        b[idx] = __float2half(1.0 / (1.0 + expf(-x)));
#else
        half x = a[idx];
        b[idx] = __hdiv(1.0, __hadd(__float2half(1.0), hexp(-x)));
#endif
    }
}

__global__ void FastllmSigmoidKernel(__nv_bfloat16* a,
                                     __nv_bfloat16 *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __bfloat162float(a[idx]);
        b[idx] = __float2bfloat16_rn(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void FastllmSigmoidMulToKernel(float *input, const float *gate,
                                          int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        input[idx] *= 1.0f / (1.0f + expf(-gate[idx]));
    }
}

__global__ void FastllmSigmoidMulToKernel(half *input, const half *gate,
                                          int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __half2float(gate[idx]);
        float value = __half2float(input[idx]);
        input[idx] = __float2half_rn(value / (1.0f + expf(-x)));
    }
}

__global__ void FastllmSigmoidMulToKernel(__nv_bfloat16 *input,
                                          const __nv_bfloat16 *gate,
                                          int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __bfloat162float(gate[idx]);
        float value = __bfloat162float(input[idx]);
        input[idx] = __float2bfloat16_rn(value / (1.0f + expf(-x)));
    }
}

__device__ float softplus(float x) {
    return  x > 20.0f ? x : log1p(expf(x));
}

__device__ __forceinline__ float softplus_fast(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return __expf(x);
    }
    return __logf(1.0f + __expf(x));
}

__global__ void FastllmMambaSoftplusKernel(float* inputData, float *outputData, float *aLog, float *dtBias, int channels, float outputScale) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = outputScale * -expf((double)aLog[i]) * softplus(inputData[o * channels + i] + dtBias[i]);
    }
}

__global__ void FastllmMambaSoftplusKernel(half* inputData, half *outputData, float *aLog, float *dtBias, int channels, float outputScale) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = __float2half(outputScale * -exp((double)aLog[i]) * softplus(__half2float(inputData[o * channels + i]) + dtBias[i]));
    }
}

__global__ void FastllmSigmoidMambaSoftplusKernel(float *sigmoidData, const float *softplusInputData, float *softplusOutputData,
                                                  const float *aLog, const float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int idx = o * channels + i;
        float x = sigmoidData[idx];
        sigmoidData[idx] = 1.0f / (1.0f + expf(-x));
        softplusOutputData[idx] = -expf((double)aLog[i]) * softplus(softplusInputData[idx] + dtBias[i]);
    }
}

__global__ void FastllmSigmoidMambaSoftplusKernel(half *sigmoidData, const half *softplusInputData, half *softplusOutputData,
                                                  const float *aLog, const float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int idx = o * channels + i;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(sigmoidData[idx]);
        sigmoidData[idx] = __float2half(1.0f / (1.0f + expf(-x)));
#else
        half x = sigmoidData[idx];
        sigmoidData[idx] = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(-x)));
#endif
        softplusOutputData[idx] = __float2half(-exp((double)aLog[i]) * softplus(__half2float(softplusInputData[idx]) + dtBias[i]));
    }
}

__global__ void FastllmSigmoidMambaSoftplusCombinedHalfKernel(
        const half *inputData, half *sigmoidOutputData,
        half *softplusOutputData, const float *aLog,
        const float *dtBias, int inputChannels,
        int baOffset, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int inputIdx = o * inputChannels + baOffset + i;
        int outputIdx = o * channels + i;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(inputData[inputIdx]);
        sigmoidOutputData[outputIdx] =
            __float2half(1.0f / (1.0f + expf(-x)));
#else
        half x = inputData[inputIdx];
        sigmoidOutputData[outputIdx] =
            __hdiv(__float2half(1.0f),
                   __hadd(__float2half(1.0f), hexp(-x)));
#endif
        softplusOutputData[outputIdx] = __float2half(
            -exp((double)aLog[i]) *
            softplus(__half2float(inputData[inputIdx + channels]) +
                     dtBias[i]));
    }
}

__global__ void FastllmSwigluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float x = a[id], y = a[id + mid];
        b[idx] = (x / (1.0f + expf(-x))) * y;
    }
}

__global__ void FastllmSwigluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[id]), y = __half2float(a[id + mid]);
        b[idx] = __float2half((x / (1.0 + expf(-x))) * y);
#else
        half x = a[id], y = a[id + mid];
        b[idx] = __hmul(__hdiv(x, __hadd(__float2half(1.0), hexp(-x))), y);
#endif
    }
}

__global__ void FastllmSwigluKernel(__nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float x = __bfloat162float(a[id]), y = __bfloat162float(a[id + mid]);
        b[idx] = __float2bfloat16((x / (1.0f + expf(-x))) * y);
    }
}

__global__ void FastllmGegluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = a[id], up = a[id + mid];
        b[idx] = gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up;
    }
}

__global__ void FastllmGegluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = __half2float(a[id]), up = __half2float(a[id + mid]);
        b[idx] = __float2half(gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

__global__ void FastllmGegluKernel(__nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = __bfloat162float(a[id]), up = __bfloat162float(a[id + mid]);
        b[idx] = __float2bfloat16(gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

// CrossSwiglu: 交替存储格式, y[i] = x[i*2+1] * silu(x[i*2])
__global__ void FastllmCrossSwigluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int outer = idx / mid;
        int inner = idx % mid;
        int id = outer * spatial + inner * 2;
        float x = a[id], y = a[id + 1];
        b[idx] = (x / (1.0f + expf(-x))) * y;
    }
}

__global__ void FastllmCrossSwigluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int outer = idx / mid;
        int inner = idx % mid;
        int id = outer * spatial + inner * 2;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[id]), y = __half2float(a[id + 1]);
        b[idx] = __float2half((x / (1.0 + expf(-x))) * y);
#else
        half x = a[id], y = a[id + 1];
        b[idx] = __hmul(__hdiv(x, __hadd(__float2half(1.0), hexp(-x))), y);
#endif
    }
}

__global__ void FastllmCrossSwigluKernel(__nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int outer = idx / mid;
        int inner = idx % mid;
        int id = outer * spatial + inner * 2;
        float x = __bfloat162float(a[id]), y = __bfloat162float(a[id + 1]);
        b[idx] = __float2bfloat16((x / (1.0f + expf(-x))) * y);
    }
}

__global__ void FastllmAddKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] + v;
    }
}

__global__ void FastllmAddKernel(half* a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) + __half2float(v));
#else
        b[idx] = __hadd(a[idx], v);
#endif
    }
}

__global__ void FastllmCopyKernel(uint8_t* a, uint8_t *b, uint64_t len) {
    uint64_t idx = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx];
    }
}

__global__ void FastllmMulKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] * v;
    }
}

__global__ void FastllmMulKernel(half* a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) * __half2float(v));
#else
        b[idx] = __hmul(a[idx], v);
#endif
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMulBatchKernel(float** pointer, int batch, float v) {
    float *input = pointer[blockIdx.x];
    float *output = pointer[blockIdx.x + batch];
    int len = (int)((unsigned long long)pointer[blockIdx.x + batch * 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        output[i] = input[i] * v;
    }
}


__global__ void FastllmReduceKernel(float *output, float* input, int len, int threadNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        output[idx] = 0;
        for (int i = 0; i < threadNum; i++) {
            output[idx] += input[idx + i * len];
        }
    }
}

__global__ void FastllmReduceKernel(half *output, half* input, int len, int threadNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        output[idx] = (half)0;
        for (int i = 0; i < threadNum; i++) {
            output[idx] = __hadd(output[idx], input[idx + i * len]);
        }
    }
}

__global__ void FastllmReduceKernel(__nv_bfloat16 *output, __nv_bfloat16 *input, int len, int threadNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float value = 0.0f;
        for (int i = 0; i < threadNum; i++) {
            value += __bfloat162float(input[idx + i * len]);
        }
        output[idx] = __float2bfloat16_rn(value);
    }
}

__global__ void FastllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

__global__ void FastllmAddToKernel(half* a, half *b, half alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]) * __half2float(alpha));
#else
        a[idx] = __hadd(a[idx], __hmul(b[idx], alpha));
#endif
    }
}

__global__ void FastllmAddToKernel(__nv_bfloat16* a, __nv_bfloat16 *b, __nv_bfloat16 alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] = __float2bfloat16_rn(__bfloat162float(a[idx]) + __bfloat162float(b[idx]) * __bfloat162float(alpha));
    }
}

__global__ void FastllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

__global__ void FastllmMulToKernel(half* a, half *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[idx]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[idx] * alpha);
#endif
    }
}

__global__ void FastllmMulToKernel(__nv_bfloat16* a,
                                   __nv_bfloat16 *b,
                                   float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] = __float2bfloat16_rn(
            __bfloat162float(a[idx]) * __bfloat162float(b[idx]) * alpha);
    }
}

__global__ void FastllmMulSingleToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[0] * alpha;
    }
}

__global__ void FastllmMulSingleToKernel(half* a, half *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[0]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[0] * alpha);
#endif
    }
}

__global__ void FastllmMulSingleToKernel(__nv_bfloat16* a,
                                         __nv_bfloat16 *b,
                                         float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] = __float2bfloat16_rn(
            __bfloat162float(a[idx]) * __bfloat162float(b[0]) * alpha);
    }
}

template <bool HAS_GATE, bool ADD_RESIDUAL>
__global__ void FastllmCudaQwen35FusedMoeJoinHalfKernel(
        half *destination,
        const half *routedOutput,
        const half *sharedOutput,
        const half *sharedGate,
        int rows, int hidden, bool broadcastGate,
        bool sharedGateAlreadySigmoid) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ half computedGateValue;
    half gateValue;
    if (HAS_GATE) {
        if (sharedGateAlreadySigmoid) {
            // The fused router/shared-gate GEMV has already rounded the
            // linear result to FP16 and applied the exact legacy sigmoid.
            // A uniform cached load is cheaper here than a block barrier.
            gateValue = sharedGate[broadcastGate ? 0 : row];
        } else {
            if (threadIdx.x == 0) {
                const half inputGate =
                    sharedGate[broadcastGate ? 0 : row];
#ifdef CUDA_NO_TENSOR_CORE
                computedGateValue = __float2half(
                    1.0f / (1.0f + expf(-__half2float(inputGate))));
#else
                // Match FastllmSigmoidKernel(half) exactly. In particular,
                // keep the sigmoid result in FP16 before multiplying
                // sharedOutput.
                computedGateValue = __hdiv(
                    1.0, __hadd(__float2half(1.0), hexp(-inputGate)));
#endif
            }
            __syncthreads();
            gateValue = computedGateValue;
        }
    }

    const int base = row * hidden;
    const half one = __float2half_rn(1.0f);
#ifndef CUDA_NO_TENSOR_CORE
    const uintptr_t pointerBits =
        (uintptr_t)destination | (uintptr_t)routedOutput |
        (uintptr_t)sharedOutput;
    if ((hidden & 1) == 0 &&
        (pointerBits & (alignof(half2) - 1)) == 0) {
        const int packedHidden = hidden / 2;
        const int packedBase = row * packedHidden;
        half2 *destination2 = reinterpret_cast<half2*>(destination);
        const half2 *routed2 =
            reinterpret_cast<const half2*>(routedOutput);
        const half2 *shared2 =
            reinterpret_cast<const half2*>(sharedOutput);
        const half2 one2 = __half2half2(one);
        half2 gate2;
        if (HAS_GATE) {
            gate2 = __half2half2(
                (half)((float)gateValue * 1.0f));
        }
        for (int column = threadIdx.x;
             column < packedHidden; column += blockDim.x) {
            const int index = packedBase + column;
            half2 shared = shared2[index];
            if (HAS_GATE) {
                shared = __hmul2_rn(shared, gate2);
            }
            half2 merged = __hadd2_rn(
                routed2[index], __hmul2_rn(shared, one2));
            if (ADD_RESIDUAL) {
                merged = __hadd2_rn(
                    destination2[index], __hmul2_rn(merged, one2));
            }
            destination2[index] = merged;
        }
        return;
    }
#endif
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        const int index = base + column;
        half shared = sharedOutput[index];
        if (HAS_GATE) {
#ifdef CUDA_NO_TENSOR_CORE
            shared = __float2half(
                __half2float(gateValue) * __half2float(shared));
#else
            // Match FastllmMulSingleToKernel(half), including its FP16
            // conversion of the scalar multiplier. The _rn intrinsic is
            // important here: without it nvcc may contract this multiply with
            // the following add, removing the legacy kernel boundary's FP16
            // rounding point.
            shared = __hmul_rn(
                shared, (half)((float)gateValue * 1.0f));
#endif
        }
#ifdef CUDA_NO_TENSOR_CORE
        half merged = __float2half(
            __half2float(routedOutput[index]) + __half2float(shared));
        if (ADD_RESIDUAL) {
            merged = __float2half(
                __half2float(destination[index]) + __half2float(merged));
        }
#else
        // Preserve both AddTo rounding points even though the operations now
        // execute in one kernel.
        half merged = __hadd_rn(
            routedOutput[index], __hmul_rn(shared, one));
        if (ADD_RESIDUAL) {
            merged = __hadd_rn(
                destination[index], __hmul_rn(merged, one));
        }
#endif
        destination[index] = merged;
    }
}

__global__ void FastllmChannelMulToKernel(float* a, float *b, float alpha, int len, int channelLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx / channelLen] * alpha;
    }
}

__global__ void FastllmChannelMulToKernel(half* a, half *b, float alpha, int len, int channelLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[idx / channelLen]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[idx / channelLen] * alpha);
#endif
    }
}

__global__ void FastllmChannelMulToKernel(__nv_bfloat16* a,
                                          __nv_bfloat16 *b,
                                          float alpha, int len,
                                          int channelLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] = __float2bfloat16_rn(
            __bfloat162float(a[idx]) *
            __bfloat162float(b[idx / channelLen]) * alpha);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAlibiMaskKernel(float* a, float *b, float maskValue, int n, int m, int spn, int spm, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    float now = b[om];
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        int idi = i / spm, idj = i % spm;
        if (idj <= spm - spn + idi) {
            a[o * spatial + i] += now * idj;
        } else {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmApplyLognAttnKernel(float* input, float *logn, float *pos, int b, int s, int spatial) {
    int ob = blockIdx.x / s;
    int os = blockIdx.x % s;
    int o = ob * s + os;
    int idx = threadIdx.x;
    int curPos = (int)(pos[0]);

    float v = logn[os + curPos];
    float *curInput = input + o * spatial;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        curInput[i] = curInput[i] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRepeatPenaltyKernel(float* input, float *penalty, float *penaltyScaleData, int tokens, int vocabs) {
    unsigned int bid = blockIdx.x;
    input += bid * vocabs;
    penalty += bid * tokens;
    float scale = penaltyScaleData[bid];
    for (int i = threadIdx.x; i < tokens; i += THREAD_PER_BLOCK) {
        int token = (int)(penalty[i] + 1e-6);
        if (token >= 0) {
            input[token] = input[token] < 0 ? input[token] * scale : input[token] / scale;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m, col = blockIdx.x % m;
    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmTransposeLastTwoByBatchKernel(
        uint8_t *dst, const uint8_t *ori, int batch, int n, int m, int k) {
    int batchIndex = blockIdx.x / (n * m);
    int inner = blockIdx.x % (n * m);
    int row = inner / m;
    int col = inner % m;
    const uint8_t *curInput = ori +
        ((batchIndex * n + row) * m + col) * k;
    uint8_t *curOutput = dst +
        ((batchIndex * m + col) * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

// The Qwen3.5 MTP verify path transposes [1, 4, channels] to
// [1, channels, 4], then performs the inverse transpose after conv1d.  Moving
// four 16-bit values per thread keeps both the strided side and the packed side
// coalesced without launching one CUDA block per scalar element.
__global__ void FastllmTransposeN4HalfKernel(
        uint16_t *dst, const uint16_t *ori, int batch, int m) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * m;
    if (index >= total) {
        return;
    }
    int batchIndex = index / m;
    int col = index - batchIndex * m;
    const uint16_t *input = ori + batchIndex * 4 * m + col;
    uint2 value;
    value.x = (uint32_t)input[0] | ((uint32_t)input[m] << 16);
    value.y = (uint32_t)input[2 * m] | ((uint32_t)input[3 * m] << 16);
    ((uint2*)dst)[index] = value;
}

__global__ void FastllmTransposeM4HalfKernel(
        uint16_t *dst, const uint16_t *ori, int batch, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n;
    if (index >= total) {
        return;
    }
    int batchIndex = index / n;
    int row = index - batchIndex * n;
    uint2 value = ((const uint2*)ori)[index];
    uint16_t value0 = (uint16_t)value.x;
    uint16_t value1 = (uint16_t)(value.x >> 16);
    uint16_t value2 = (uint16_t)value.y;
    uint16_t value3 = (uint16_t)(value.y >> 16);
    uint16_t *output = dst + batchIndex * 4 * n + row;
    output[0] = value0;
    output[n] = value1;
    output[2 * n] = value2;
    output[3 * n] = value3;
}

template <typename T>
__global__ void FastllmTransposeLastTwoElementKernel(
        T *dst, const T *ori, int n, int m, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    int batchIndex = index / (n * m);
    int inner = index - batchIndex * n * m;
    int row = inner / m;
    int col = inner - row * m;
    dst[(batchIndex * m + col) * n + row] = ori[index];
}

static bool FastllmLaunchTransposeLastTwo(
        void *dst, const void *ori, int batch, int n, int m,
        int unitSize, cudaStream_t stream) {
    int len = batch * n * m;
    int blocks = (len + 255) / 256;
    if (unitSize == 2 && n == 4) {
        int packedBlocks = (batch * m + 255) / 256;
        FastllmTransposeN4HalfKernel<<<packedBlocks, 256, 0, stream>>>(
            (uint16_t*)dst, (const uint16_t*)ori, batch, m);
    } else if (unitSize == 2 && m == 4) {
        int packedBlocks = (batch * n + 255) / 256;
        FastllmTransposeM4HalfKernel<<<packedBlocks, 256, 0, stream>>>(
            (uint16_t*)dst, (const uint16_t*)ori, batch, n);
    } else if (unitSize == 4) {
        FastllmTransposeLastTwoElementKernel<uint32_t>
            <<<blocks, 256, 0, stream>>>(
                (uint32_t*)dst, (const uint32_t*)ori, n, m, len);
    } else if (unitSize == 2) {
        FastllmTransposeLastTwoElementKernel<uint16_t>
            <<<blocks, 256, 0, stream>>>(
                (uint16_t*)dst, (const uint16_t*)ori, n, m, len);
    } else if (unitSize == 1) {
        FastllmTransposeLastTwoElementKernel<uint8_t>
            <<<blocks, 256, 0, stream>>>(
                (uint8_t*)dst, (const uint8_t*)ori, n, m, len);
    } else {
        return false;
    }
    return true;
}

template <typename T>
__global__ void FastllmPermuteKernel(T *dst, T *ori, int *temp, int axisLen, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        int old = 0;
        int idx = i;
        for (int j = 0; j < axisLen; ++j) {
            int order = temp[j];
            old += (idx / temp[j + 2 * axisLen]) * temp[order + 1 * axisLen];
            idx %= temp[j + 2 * axisLen];
        }
        dst[i] = ori[old];
    }
}

__global__ void FastllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}

__global__ void FastllmLlamaRotatePosition2DKernel(half *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + m / 2]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + m / 2] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmLlamaRotatePosition2DKernel(__nv_bfloat16 *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    __nv_bfloat16 *d = (__nv_bfloat16 *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + m / 2]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + m / 2] = __float2bfloat16(va * curSin + vb * curCos);
}

__global__ void FastllmRopeEncodingKernel(float *data, float *positionIds,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                   float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int half = rotateDim / 2;
    int index = (int) (positionIds[b * partStride + l]);
    float position = (float)index / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + half];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + half] = va * curSin + vb * curCos;
}

__global__ void FastllmRopeEncodingKernel(half *data, float *positionIds,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                   float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int index = (int) (positionIds[b * partStride + l]);
    float position = (float)index / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + half_dim]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmRopeEncodingKernel(__nv_bfloat16 *data, float *positionIds,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                   float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int index = (int) (positionIds[b * partStride + l]);
    float position = (float)index / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    __nv_bfloat16 *d = (__nv_bfloat16 *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + half_dim]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2bfloat16(va * curSin + vb * curCos);
}

__device__ __forceinline__ float FastllmYarnInvFreq(int dim, int rotateDim,
                                                    float ropeTheta, float factor,
                                                    float correctionLow, float correctionHigh) {
    // CUDA powf may differ from the host result by one ULP.  Multiplying that
    // error by a 256K position noticeably shifts the phase, so compute the
    // power accurately and round to float at the same point as the old cache.
    float posFreq = (float)pow((double)ropeTheta,
                               (double)(2 * dim) / rotateDim);
    float extrapolation = 1.0f / posFreq;
    float interpolation = 1.0f / (factor * posFreq);
    float ramp = fmaxf(0.0f, fminf(1.0f,
        (dim - correctionLow) / (correctionHigh - correctionLow)));
    float extrapolationFactor = 1.0f - ramp;
    return interpolation * (1.0f - extrapolationFactor) +
           extrapolation * extrapolationFactor;
}

__global__ void FastllmYarnRopeEncodingKernel(float *data, float *positionIds,
                                               int len, int spatial, int n, int m,
                                               int positionStride, int rotateDim,
                                               float ropeTheta, float factor, float attentionFactor,
                                               float correctionLow, float correctionHigh) {
    int token = blockIdx.x;
    int batch = token / len;
    int localToken = token % len;
    int dim = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = (float)(int)positionIds[batch * positionStride + localToken];
    float angle = position * FastllmYarnInvFreq(
        dim, rotateDim, ropeTheta, factor, correctionLow, correctionHigh);
    float curSin, curCos;
    sincosf(angle, &curSin, &curCos);
    curSin *= attentionFactor;
    curCos *= attentionFactor;
    float *d = data + token * spatial + dim;
    for (int head = 0; head < n; head++) {
        int offset = head * m;
        float a = d[offset], b = d[offset + halfDim];
        d[offset] = a * curCos - b * curSin;
        d[offset + halfDim] = a * curSin + b * curCos;
    }
}

__global__ void FastllmYarnRopeEncodingKernel(half *data, float *positionIds,
                                               int len, int spatial, int n, int m,
                                               int positionStride, int rotateDim,
                                               float ropeTheta, float factor, float attentionFactor,
                                               float correctionLow, float correctionHigh) {
    int token = blockIdx.x;
    int batch = token / len;
    int localToken = token % len;
    int dim = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = (float)(int)positionIds[batch * positionStride + localToken];
    float angle = position * FastllmYarnInvFreq(
        dim, rotateDim, ropeTheta, factor, correctionLow, correctionHigh);
    float curSin, curCos;
    sincosf(angle, &curSin, &curCos);
    curSin *= attentionFactor;
    curCos *= attentionFactor;
    half *d = data + token * spatial + dim;
    for (int head = 0; head < n; head++) {
        int offset = head * m;
        float a = __half2float(d[offset]);
        float b = __half2float(d[offset + halfDim]);
        d[offset] = __float2half(a * curCos - b * curSin);
        d[offset + halfDim] = __float2half(a * curSin + b * curCos);
    }
}

__global__ void FastllmYarnRopeEncodingKernel(__nv_bfloat16 *data, float *positionIds,
                                               int len, int spatial, int n, int m,
                                               int positionStride, int rotateDim,
                                               float ropeTheta, float factor, float attentionFactor,
                                               float correctionLow, float correctionHigh) {
    int token = blockIdx.x;
    int batch = token / len;
    int localToken = token % len;
    int dim = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = (float)(int)positionIds[batch * positionStride + localToken];
    float angle = position * FastllmYarnInvFreq(
        dim, rotateDim, ropeTheta, factor, correctionLow, correctionHigh);
    float curSin, curCos;
    sincosf(angle, &curSin, &curCos);
    curSin *= attentionFactor;
    curCos *= attentionFactor;
    __nv_bfloat16 *d = data + token * spatial + dim;
    for (int head = 0; head < n; head++) {
        int offset = head * m;
        float a = __bfloat162float(d[offset]);
        float b = __bfloat162float(d[offset + halfDim]);
        d[offset] = __float2bfloat16(a * curCos - b * curSin);
        d[offset + halfDim] = __float2bfloat16(a * curSin + b * curCos);
    }
}

__device__ __forceinline__ float FastllmLlama3InvFreq(float invFreq, float factor, float originalMaxPosition,
                                                      float lowFreqFactor, float highFreqFactor) {
    float wavelen = 2.0f * (float)M_PI / invFreq;
    float lowWavelen = originalMaxPosition / lowFreqFactor;
    float highWavelen = originalMaxPosition / highFreqFactor;
    float invLlama = wavelen > lowWavelen ? invFreq / factor : invFreq;
    if (!(wavelen < highWavelen) && !(wavelen > lowWavelen)) {
        float smooth = (originalMaxPosition / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
        invLlama = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return invLlama;
}

__global__ void FastllmLlama3RopeEncodingKernel(float *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int half = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    float *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + half];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + half] = va * curSin + vb * curCos;
}

__global__ void FastllmLlama3RopeEncodingKernel(half *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    half *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + halfDim]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + halfDim] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmLlama3RopeEncodingKernel(__nv_bfloat16 *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    __nv_bfloat16 *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + halfDim]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + halfDim] = __float2bfloat16(va * curSin + vb * curCos);
}

__global__ void FastllmQwen35InterleavedRopeKernel(float *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + half];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + half] = va * curSin + vb * curCos;
}

__global__ void FastllmQwen35InterleavedRopeKernel(half *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + half_dim]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmQwen35InterleavedRopeKernel(__nv_bfloat16 *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    __nv_bfloat16 *d = (__nv_bfloat16 *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + half_dim]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2bfloat16(va * curSin + vb * curCos);
}

__global__ void FastllmLlamaRotatePosition2DPartKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int part) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + part / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + part / 2] = va * curSin + vb * curCos;
}

__global__ void FastllmLlamaRotatePosition2DPartKernel(half *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int part) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + part / 2]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + part / 2] = __float2half(va * curSin + vb * curCos);
}


__global__ void FastllmNearlyRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + 1];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + 1] = va * curSin + vb * curCos;
}

__global__ void FastllmNearlyRotatePosition2DKernel(half *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + 1]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + 1] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                              int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n) / 2;
    int l = o / bs;
    int b = o % bs;
    int part = (blockIdx.x / n) % 2;
    int j = threadIdx.x;
    int index = (int) (positionIds[(b * 2 + part) * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + part * m / 2 + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 4];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 4] = va * curSin + vb * curCos;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS > 0 ? NUM_WARPS : 1];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 1. 向量化加载 (float4)，每个线程累加平方和
    int f4_channels = channels / 4;
    const float4 *input_f4 = reinterpret_cast<const float4 *>(input);
    float sum2 = 0.0f;
    for (int i = tid; i < f4_channels; i += THREAD_PER_BLOCK) {
        float4 v = input_f4[i];
        sum2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    // 处理尾部元素
    int tail_start = f4_channels * 4;
    for (int i = tail_start + tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum2 += x * x;
    }

    // 2. Warp shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (THREAD_PER_BLOCK > WARP_SIZE) {
        if (lane_id == 0) {
            warp_sums[warp_id] = sum2;
        }
        __syncthreads();

        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                scale = rsqrtf(val / channels + eps);
            }
        }
        __syncthreads();
    } else {
        // 只有一个 warp 的情况
        if (tid == 0) {
            scale = rsqrtf(sum2 / channels + eps);
        }
        __syncthreads();
    }

    // 3. 向量化写出
    float s = scale;
    float4 *output_f4 = reinterpret_cast<float4 *>(output);
    const float4 *weight_f4 = reinterpret_cast<const float4 *>(weight);
    for (int i = tid; i < f4_channels; i += THREAD_PER_BLOCK) {
        float4 v = input_f4[i];
        float4 w = weight_f4[i];
        float4 out_v;
        out_v.x = v.x * s * w.x;
        out_v.y = v.y * s * w.y;
        out_v.z = v.z * s * w.z;
        out_v.w = v.w * s * w.w;
        output_f4[i] = out_v;
    }
    for (int i = tail_start + tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = input[i] * s * __ldg(&weight[i]);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(half *input, float *weight, half *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    // 使用 warp shuffle reduction，仅需少量 shared memory 给跨 warp 汇总
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 1. 向量化加载 (half2)，每个线程累加平方和
    int half2_channels = channels / 2;
    const half2 *input_h2 = reinterpret_cast<const half2 *>(input);
    float sum2 = 0.0f;
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);
        sum2 += fv.x * fv.x + fv.y * fv.y;
    }
    // 处理 channels 为奇数的尾部元素
    if (channels & 1) {
        int last = channels - 1;
        if (tid == 0) {
            float x = __half2float(input[last]);
            sum2 += x * x;
        }
    }

    // 2. Warp shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();

    // 跨 warp 汇总（由第一个 warp 完成）
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / channels + eps);
        }
    }
    __syncthreads();

    // 3. 向量化写出
    float s = scale;
    half2 *output_h2 = reinterpret_cast<half2 *>(output);
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);
        float w0 = __ldg(&weight[i * 2]);
        float w1 = __ldg(&weight[i * 2 + 1]);
        float2 out_f;
        out_f.x = fv.x * s * w0;
        out_f.y = fv.y * s * w1;
        output_h2[i] = __float22half2_rn(out_f);
    }
    // 处理 channels 为奇数的尾部元素
    if ((channels & 1) && tid == 0) {
        int last = channels - 1;
        output[last] = __float2half(__half2float(input[last]) * s * __ldg(&weight[last]));
    }
}

// Preserve the reduction order of the 1024-thread RMSNorm kernel while using
// 512 physical threads. Each thread evaluates the work of virtual threads tid
// and tid + 512 independently; their 32 warp sums are then reduced in the same
// order as the original launch. This permits more resident blocks without
// changing the floating-point grouping.
__global__ __launch_bounds__(512)
void FastllmRMSNormHalfVirtual1024Kernel(
        half *input, float *weight, half *output,
        int channels, float eps) {
    constexpr int PHYSICAL_THREADS = 512;
    constexpr int VIRTUAL_THREADS = 1024;
    constexpr int PHYSICAL_WARPS = PHYSICAL_THREADS / 32;
    __shared__ float warpSums[VIRTUAL_THREADS / 32];
    __shared__ float scale;

    int row = blockIdx.x;
    input += (size_t)row * channels;
    output += (size_t)row * channels;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int half2Channels = channels / 2;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);

    float sum0 = 0.0f;
    for (int i = tid; i < half2Channels; i += VIRTUAL_THREADS) {
        float2 value = __half22float2(input2[i]);
        sum0 += value.x * value.x + value.y * value.y;
    }
    float sum1 = 0.0f;
    for (int i = tid + PHYSICAL_THREADS;
         i < half2Channels; i += VIRTUAL_THREADS) {
        float2 value = __half22float2(input2[i]);
        sum1 += value.x * value.x + value.y * value.y;
    }
    if ((channels & 1) && tid == 0) {
        float value = __half2float(input[channels - 1]);
        sum0 += value * value;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    if (lane == 0) {
        warpSums[warp] = sum0;
        warpSums[warp + PHYSICAL_WARPS] = sum1;
    }
    __syncthreads();

    if (warp == 0) {
        float value = warpSums[lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            scale = rsqrtf(value / channels + eps);
        }
    }
    __syncthreads();

    half2 *output2 = reinterpret_cast<half2 *>(output);
    float rowScale = scale;
    for (int i = tid; i < half2Channels; i += VIRTUAL_THREADS) {
        float2 value = __half22float2(input2[i]);
        float2 normalized;
        normalized.x =
            value.x * rowScale * __ldg(weight + i * 2);
        normalized.y =
            value.y * rowScale * __ldg(weight + i * 2 + 1);
        output2[i] = __float22half2_rn(normalized);
    }
    for (int i = tid + PHYSICAL_THREADS;
         i < half2Channels; i += VIRTUAL_THREADS) {
        float2 value = __half22float2(input2[i]);
        float2 normalized;
        normalized.x =
            value.x * rowScale * __ldg(weight + i * 2);
        normalized.y =
            value.y * rowScale * __ldg(weight + i * 2 + 1);
        output2[i] = __float22half2_rn(normalized);
    }
    if ((channels & 1) && tid == 0) {
        int last = channels - 1;
        output[last] = __float2half(
            __half2float(input[last]) * rowScale * __ldg(weight + last));
    }
}

union FastllmRMSNormHalf8 {
    uint4 packed;
    half2 values[4];
};

__global__ __launch_bounds__(512)
void FastllmRMSNormHalf4096Kernel(
        const half *__restrict__ input, const float *__restrict__ weight,
        half *__restrict__ output, float eps) {
    constexpr int CHANNELS = 4096;
    constexpr int VALUES_PER_THREAD = 8;
    constexpr int WARP_COUNT = 16;
    int tid = threadIdx.x;
    int row = blockIdx.x;
    int offset = tid * VALUES_PER_THREAD;
    input += (int64_t)row * CHANNELS;
    output += (int64_t)row * CHANNELS;

    FastllmRMSNormHalf8 inputPack;
    inputPack.packed = *(const uint4*)(input + offset);
    float2 inputValues[4];
    float sum2 = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        inputValues[i] = __half22float2(inputPack.values[i]);
        sum2 += inputValues[i].x * inputValues[i].x +
                inputValues[i].y * inputValues[i].y;
    }

#pragma unroll
    for (int shift = 16; shift > 0; shift >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, shift);
    }
    __shared__ float warpSums[WARP_COUNT];
    __shared__ float scale;
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        warpSums[warp] = sum2;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARP_COUNT ? warpSums[lane] : 0.0f;
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, shift);
        }
        if (lane == 0) {
            scale = rsqrtf(value / (float)CHANNELS + eps);
        }
    }
    __syncthreads();

    const float4 *weight4 = (const float4*)(weight + offset);
    float4 weightLo = weight4[0];
    float4 weightHi = weight4[1];
    float weights[8] = {
        weightLo.x, weightLo.y, weightLo.z, weightLo.w,
        weightHi.x, weightHi.y, weightHi.z, weightHi.w
    };
    FastllmRMSNormHalf8 outputPack;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        outputPack.values[i] = __floats2half2_rn(
            inputValues[i].x * scale * weights[i * 2],
            inputValues[i].y * scale * weights[i * 2 + 1]);
    }
    *(uint4*)(output + offset) = outputPack.packed;
}

// The legacy hidden=128 launch uses two warps.  One physical warp can evaluate
// both reductions independently and combine the two old warp sums in the exact
// same final addition, avoiding block-wide shared-memory barriers.
__global__ __launch_bounds__(32) void FastllmRMSNormHalf128ExactKernel(
        const half *input, const float *weight, half *output, float eps) {
    constexpr int CHANNELS = 128;
    int row = blockIdx.x;
    int lane = threadIdx.x;
    input += (size_t)row * CHANNELS;
    output += (size_t)row * CHANNELS;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);

    float2 value0 = __half22float2(input2[lane]);
    float2 value1 = __half22float2(input2[lane + 32]);
    float sum0 = value0.x * value0.x + value0.y * value0.y;
    float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    float scale = 0.0f;
    if (lane == 0) {
        scale = rsqrtf((sum0 + sum1) / CHANNELS + eps);
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);

    half2 *output2 = reinterpret_cast<half2 *>(output);
    float2 normalized0, normalized1;
    normalized0.x = value0.x * scale * __ldg(weight + lane * 2);
    normalized0.y = value0.y * scale * __ldg(weight + lane * 2 + 1);
    normalized1.x = value1.x * scale * __ldg(weight + (lane + 32) * 2);
    normalized1.y = value1.y * scale * __ldg(weight + (lane + 32) * 2 + 1);
    output2[lane] = __float22half2_rn(normalized0);
    output2[lane + 32] = __float22half2_rn(normalized1);
}

// Normalize Q and K directly from the token-major combined convolution
// output. Each Q/K row uses the same lane mapping, reduction order, rsqrt,
// weight application, and fp16 rounding as FastllmRMSNormHalf128ExactKernel.
__global__ __launch_bounds__(32) void FastllmRMSNormCombinedQKHalf128ExactKernel(
        const half *qkvInput, const float *weight,
        half *qOutput, half *kOutput,
        int keyHeads, int valueHeads, float eps) {
    constexpr int CHANNELS = 128;
    int rowHead = blockIdx.x;
    int row = rowHead / keyHeads;
    int head = rowHead - row * keyHeads;
    int lane = threadIdx.x;
    size_t inputStride =
        (size_t)(keyHeads * 2 + valueHeads) * CHANNELS;
    const half *qInput =
        qkvInput + (size_t)row * inputStride + head * CHANNELS;
    const half *kInput =
        qkvInput + (size_t)row * inputStride +
        (keyHeads + head) * CHANNELS;
    qOutput += (size_t)rowHead * CHANNELS;
    kOutput += (size_t)rowHead * CHANNELS;

    const half2 *qInput2 = reinterpret_cast<const half2 *>(qInput);
    const half2 *kInput2 = reinterpret_cast<const half2 *>(kInput);
    float2 qValue0 = __half22float2(qInput2[lane]);
    float2 qValue1 = __half22float2(qInput2[lane + 32]);
    float2 kValue0 = __half22float2(kInput2[lane]);
    float2 kValue1 = __half22float2(kInput2[lane + 32]);
    float qSum0 = qValue0.x * qValue0.x + qValue0.y * qValue0.y;
    float qSum1 = qValue1.x * qValue1.x + qValue1.y * qValue1.y;
    float kSum0 = kValue0.x * kValue0.x + kValue0.y * kValue0.y;
    float kSum1 = kValue1.x * kValue1.x + kValue1.y * kValue1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qSum0 += __shfl_down_sync(0xffffffffu, qSum0, offset);
        qSum1 += __shfl_down_sync(0xffffffffu, qSum1, offset);
        kSum0 += __shfl_down_sync(0xffffffffu, kSum0, offset);
        kSum1 += __shfl_down_sync(0xffffffffu, kSum1, offset);
    }
    float qScale = 0.0f;
    float kScale = 0.0f;
    if (lane == 0) {
        qScale = rsqrtf((qSum0 + qSum1) / CHANNELS + eps);
        kScale = rsqrtf((kSum0 + kSum1) / CHANNELS + eps);
    }
    qScale = __shfl_sync(0xffffffffu, qScale, 0);
    kScale = __shfl_sync(0xffffffffu, kScale, 0);

    float weights[4] = {
        __ldg(weight + lane * 2),
        __ldg(weight + lane * 2 + 1),
        __ldg(weight + (lane + 32) * 2),
        __ldg(weight + (lane + 32) * 2 + 1)
    };
    float2 qNormalized0, qNormalized1;
    float2 kNormalized0, kNormalized1;
    qNormalized0.x = qValue0.x * qScale * weights[0];
    qNormalized0.y = qValue0.y * qScale * weights[1];
    qNormalized1.x = qValue1.x * qScale * weights[2];
    qNormalized1.y = qValue1.y * qScale * weights[3];
    kNormalized0.x = kValue0.x * kScale * weights[0];
    kNormalized0.y = kValue0.y * kScale * weights[1];
    kNormalized1.x = kValue1.x * kScale * weights[2];
    kNormalized1.y = kValue1.y * kScale * weights[3];

    half2 *qOutput2 = reinterpret_cast<half2 *>(qOutput);
    half2 *kOutput2 = reinterpret_cast<half2 *>(kOutput);
    qOutput2[lane] = __float22half2_rn(qNormalized0);
    qOutput2[lane + 32] = __float22half2_rn(qNormalized1);
    kOutput2[lane] = __float22half2_rn(kNormalized0);
    kOutput2[lane + 32] = __float22half2_rn(kNormalized1);
}

template <bool WRITE_DEAD_OUTPUTS>
__global__ __launch_bounds__(32)
void FastllmQwen35GdnPostConvExactHalf128Kernel(
        const half *qkvInput, const float *weight,
        const half *gInput, const half *betaInput,
        half *qOutput, half *kOutput, half *vOutput,
        half *gOutput, half *betaOutput,
        half *kBetaOutput, half *vBetaOutput,
        int seqLen, int paddedSeqLen,
        int keyHeads, int valueHeads, float eps, half qScale) {
    constexpr int CHANNELS = 128;
    int rowHead = blockIdx.x;
    int paddedRow = rowHead / keyHeads;
    int head = rowHead - paddedRow * keyHeads;
    int batchIndex = paddedRow / paddedSeqLen;
    int token = paddedRow - batchIndex * paddedSeqLen;
    int lane = threadIdx.x;
    int headGroup = valueHeads / keyHeads;
    if (token >= seqLen) {
        if (lane == 0) {
            for (int group = 0; group < headGroup; group++) {
                int valueHead = head * headGroup + group;
                size_t outputRow =
                    ((size_t)batchIndex * valueHeads + valueHead) *
                        paddedSeqLen + token;
                gOutput[outputRow] = __float2half(0.0f);
                if constexpr (WRITE_DEAD_OUTPUTS) {
                    betaOutput[outputRow] = __float2half(0.0f);
                }
            }
        }
        return;
    }
    int row = batchIndex * seqLen + token;
    size_t inputStride =
        (size_t)(keyHeads * 2 + valueHeads) * CHANNELS;
    const half *qInput =
        qkvInput + (size_t)row * inputStride + head * CHANNELS;
    const half *kInput =
        qkvInput + (size_t)row * inputStride +
        (keyHeads + head) * CHANNELS;

    const half2 *qInput2 = reinterpret_cast<const half2 *>(qInput);
    const half2 *kInput2 = reinterpret_cast<const half2 *>(kInput);
    float2 qValue0 = __half22float2(qInput2[lane]);
    float2 qValue1 = __half22float2(qInput2[lane + 32]);
    float2 kValue0 = __half22float2(kInput2[lane]);
    float2 kValue1 = __half22float2(kInput2[lane + 32]);
    float qSum0 = qValue0.x * qValue0.x + qValue0.y * qValue0.y;
    float qSum1 = qValue1.x * qValue1.x + qValue1.y * qValue1.y;
    float kSum0 = kValue0.x * kValue0.x + kValue0.y * kValue0.y;
    float kSum1 = kValue1.x * kValue1.x + kValue1.y * kValue1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qSum0 += __shfl_down_sync(0xffffffffu, qSum0, offset);
        qSum1 += __shfl_down_sync(0xffffffffu, qSum1, offset);
        kSum0 += __shfl_down_sync(0xffffffffu, kSum0, offset);
        kSum1 += __shfl_down_sync(0xffffffffu, kSum1, offset);
    }
    float qNormScale = 0.0f;
    float kNormScale = 0.0f;
    if (lane == 0) {
        qNormScale = rsqrtf((qSum0 + qSum1) / CHANNELS + eps);
        kNormScale = rsqrtf((kSum0 + kSum1) / CHANNELS + eps);
    }
    qNormScale = __shfl_sync(0xffffffffu, qNormScale, 0);
    kNormScale = __shfl_sync(0xffffffffu, kNormScale, 0);

    float weights[4] = {
        __ldg(weight + lane * 2),
        __ldg(weight + lane * 2 + 1),
        __ldg(weight + (lane + 32) * 2),
        __ldg(weight + (lane + 32) * 2 + 1)
    };
    float2 qNormalized0, qNormalized1;
    float2 kNormalized0, kNormalized1;
    qNormalized0.x = qValue0.x * qNormScale * weights[0];
    qNormalized0.y = qValue0.y * qNormScale * weights[1];
    qNormalized1.x = qValue1.x * qNormScale * weights[2];
    qNormalized1.y = qValue1.y * qNormScale * weights[3];
    kNormalized0.x = kValue0.x * kNormScale * weights[0];
    kNormalized0.y = kValue0.y * kNormScale * weights[1];
    kNormalized1.x = kValue1.x * kNormScale * weights[2];
    kNormalized1.y = kValue1.y * kNormScale * weights[3];
    half2 qNormalizedHalf0 = __float22half2_rn(qNormalized0);
    half2 qNormalizedHalf1 = __float22half2_rn(qNormalized1);
    half2 kNormalizedHalf0 = __float22half2_rn(kNormalized0);
    half2 kNormalizedHalf1 = __float22half2_rn(kNormalized1);
    half2 qScale2 = __halves2half2(qScale, qScale);

    for (int group = 0; group < headGroup; group++) {
        int valueHead = head * headGroup + group;
        size_t scalarInputIndex =
            (size_t)row * valueHeads + valueHead;
        half betaValue = betaInput[scalarInputIndex];
        half2 beta2 = __halves2half2(betaValue, betaValue);
        size_t outputRow =
            ((size_t)batchIndex * valueHeads + valueHead) *
                paddedSeqLen + token;
        size_t outputBase = outputRow * CHANNELS;
        half2 *qOutput2 =
            reinterpret_cast<half2 *>(qOutput + outputBase);
        half2 *kOutput2 =
            reinterpret_cast<half2 *>(kOutput + outputBase);
        half2 *vOutput2 =
            reinterpret_cast<half2 *>(vOutput + outputBase);
        half2 *kBetaOutput2 =
            reinterpret_cast<half2 *>(kBetaOutput + outputBase);
        half2 *vBetaOutput2 =
            reinterpret_cast<half2 *>(vBetaOutput + outputBase);
        const half *vInput =
            qkvInput + (size_t)row * inputStride +
            (keyHeads * 2 + valueHead) * CHANNELS;
        const half2 *vInput2 = reinterpret_cast<const half2 *>(vInput);
        half2 vValue0 = vInput2[lane];
        half2 vValue1 = vInput2[lane + 32];

        qOutput2[lane] = __hmul2(qNormalizedHalf0, qScale2);
        qOutput2[lane + 32] = __hmul2(qNormalizedHalf1, qScale2);
        kOutput2[lane] = kNormalizedHalf0;
        kOutput2[lane + 32] = kNormalizedHalf1;
        if constexpr (WRITE_DEAD_OUTPUTS) {
            vOutput2[lane] = vValue0;
            vOutput2[lane + 32] = vValue1;
        }
        kBetaOutput2[lane] = __hmul2(kNormalizedHalf0, beta2);
        kBetaOutput2[lane + 32] = __hmul2(kNormalizedHalf1, beta2);
        vBetaOutput2[lane] = __hmul2(vValue0, beta2);
        vBetaOutput2[lane + 32] = __hmul2(vValue1, beta2);
        if (lane == 0) {
            gOutput[outputRow] = gInput[scalarInputIndex];
            if constexpr (WRITE_DEAD_OUTPUTS) {
                betaOutput[outputRow] = betaValue;
            }
        }
    }
}

// Exact ragged counterpart of the uniform post-conv kernel.  Q/K remain in
// their native key-head cardinality; value-head expansion is performed only
// for beta-dependent K and V.  This preserves the legacy half rounding while
// avoiding split, repeat, RMSNorm, and five independent pack launches.
__global__ __launch_bounds__(32)
void FastllmQwen35GdnPostConvRaggedExactHalf128Kernel(
        const half *qkvInput, const float *weight,
        const half *combinedBaInput, const float *aLog,
        const float *dtBias, const int *chunkTokenBases,
        const int *chunkValidTokens,
        half *qOutput, half *kOutput, half *gOutput,
        half *kBetaOutput, half *vBetaOutput,
        int totalChunks, int baChannels, int baOffset,
        int keyHeads, int valueHeads, float eps, float qScale) {
    constexpr int CHANNELS = 128;
    constexpr int CHUNK_SIZE = 64;
    int rowHead = blockIdx.x;
    int packedRow = rowHead / keyHeads;
    int keyHead = rowHead - packedRow * keyHeads;
    int chunk = packedRow / CHUNK_SIZE;
    int tokenInChunk = packedRow - chunk * CHUNK_SIZE;
    int lane = threadIdx.x;
    int headGroup = valueHeads / keyHeads;
    int validTokens = chunkValidTokens[chunk];
    size_t keyOutputBase =
        ((size_t)keyHead * totalChunks * CHUNK_SIZE + packedRow) * CHANNELS;
    half2 *qOutput2 = reinterpret_cast<half2 *>(qOutput + keyOutputBase);
    half2 *kOutput2 = reinterpret_cast<half2 *>(kOutput + keyOutputBase);

    if (tokenInChunk >= validTokens) {
        half2 zero2 = __float2half2_rn(0.0f);
        qOutput2[lane] = zero2;
        qOutput2[lane + 32] = zero2;
        kOutput2[lane] = zero2;
        kOutput2[lane + 32] = zero2;
        for (int group = 0; group < headGroup; group++) {
            int valueHead = keyHead * headGroup + group;
            size_t valueRow =
                (size_t)valueHead * totalChunks * CHUNK_SIZE + packedRow;
            half2 *kBetaOutput2 = reinterpret_cast<half2 *>(
                kBetaOutput + valueRow * CHANNELS);
            half2 *vBetaOutput2 = reinterpret_cast<half2 *>(
                vBetaOutput + valueRow * CHANNELS);
            kBetaOutput2[lane] = zero2;
            kBetaOutput2[lane + 32] = zero2;
            vBetaOutput2[lane] = zero2;
            vBetaOutput2[lane + 32] = zero2;
            if (lane == 0) {
                gOutput[valueRow] = __float2half(0.0f);
            }
        }
        return;
    }

    int row = chunkTokenBases[chunk] + tokenInChunk;
    size_t qkvStride =
        (size_t)(keyHeads * 2 + valueHeads) * CHANNELS;
    const half *qInput =
        qkvInput + (size_t)row * qkvStride + keyHead * CHANNELS;
    const half *kInput =
        qkvInput + (size_t)row * qkvStride +
        (keyHeads + keyHead) * CHANNELS;
    const half2 *qInput2 = reinterpret_cast<const half2 *>(qInput);
    const half2 *kInput2 = reinterpret_cast<const half2 *>(kInput);
    float2 qValue0 = __half22float2(qInput2[lane]);
    float2 qValue1 = __half22float2(qInput2[lane + 32]);
    float2 kValue0 = __half22float2(kInput2[lane]);
    float2 kValue1 = __half22float2(kInput2[lane + 32]);
    float qSum0 = qValue0.x * qValue0.x + qValue0.y * qValue0.y;
    float qSum1 = qValue1.x * qValue1.x + qValue1.y * qValue1.y;
    float kSum0 = kValue0.x * kValue0.x + kValue0.y * kValue0.y;
    float kSum1 = kValue1.x * kValue1.x + kValue1.y * kValue1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qSum0 += __shfl_down_sync(0xffffffffu, qSum0, offset);
        qSum1 += __shfl_down_sync(0xffffffffu, qSum1, offset);
        kSum0 += __shfl_down_sync(0xffffffffu, kSum0, offset);
        kSum1 += __shfl_down_sync(0xffffffffu, kSum1, offset);
    }
    float qNormScale = 0.0f;
    float kNormScale = 0.0f;
    if (lane == 0) {
        qNormScale = rsqrtf((qSum0 + qSum1) / CHANNELS + eps);
        kNormScale = rsqrtf((kSum0 + kSum1) / CHANNELS + eps);
    }
    qNormScale = __shfl_sync(0xffffffffu, qNormScale, 0);
    kNormScale = __shfl_sync(0xffffffffu, kNormScale, 0);

    float weights[4] = {
        __ldg(weight + lane * 2),
        __ldg(weight + lane * 2 + 1),
        __ldg(weight + (lane + 32) * 2),
        __ldg(weight + (lane + 32) * 2 + 1)
    };
    half2 qNormalized0 = __floats2half2_rn(
        qValue0.x * qNormScale * weights[0],
        qValue0.y * qNormScale * weights[1]);
    half2 qNormalized1 = __floats2half2_rn(
        qValue1.x * qNormScale * weights[2],
        qValue1.y * qNormScale * weights[3]);
    half2 kNormalized0 = __floats2half2_rn(
        kValue0.x * kNormScale * weights[0],
        kValue0.y * kNormScale * weights[1]);
    half2 kNormalized1 = __floats2half2_rn(
        kValue1.x * kNormScale * weights[2],
        kValue1.y * kNormScale * weights[3]);
    float2 qNormalizedFloat0 = __half22float2(qNormalized0);
    float2 qNormalizedFloat1 = __half22float2(qNormalized1);
    qOutput2[lane] = __floats2half2_rn(
        qNormalizedFloat0.x * qScale,
        qNormalizedFloat0.y * qScale);
    qOutput2[lane + 32] = __floats2half2_rn(
        qNormalizedFloat1.x * qScale,
        qNormalizedFloat1.y * qScale);
    kOutput2[lane] = kNormalized0;
    kOutput2[lane + 32] = kNormalized1;

    const half *baRow = combinedBaInput + (size_t)row * baChannels;
    for (int group = 0; group < headGroup; group++) {
        int valueHead = keyHead * headGroup + group;
        half betaRaw = baRow[baOffset + valueHead];
#ifdef CUDA_NO_TENSOR_CORE
        half betaValue = __float2half(
            1.0f / (1.0f + expf(-__half2float(betaRaw))));
#else
        half betaValue = __hdiv(
            __float2half(1.0f),
            __hadd(__float2half(1.0f), hexp(-betaRaw)));
#endif
        half gateValue = __float2half(
            -exp((double)aLog[valueHead]) *
            softplus(__half2float(
                baRow[baOffset + valueHeads + valueHead]) +
                dtBias[valueHead]));
        half2 beta2 = __halves2half2(betaValue, betaValue);
        size_t valueRow =
            (size_t)valueHead * totalChunks * CHUNK_SIZE + packedRow;
        half2 *kBetaOutput2 = reinterpret_cast<half2 *>(
            kBetaOutput + valueRow * CHANNELS);
        half2 *vBetaOutput2 = reinterpret_cast<half2 *>(
            vBetaOutput + valueRow * CHANNELS);
        const half *vInput =
            qkvInput + (size_t)row * qkvStride +
            (keyHeads * 2 + valueHead) * CHANNELS;
        const half2 *vInput2 = reinterpret_cast<const half2 *>(vInput);
        kBetaOutput2[lane] = __hmul2(kNormalized0, beta2);
        kBetaOutput2[lane + 32] = __hmul2(kNormalized1, beta2);
        vBetaOutput2[lane] = __hmul2(vInput2[lane], beta2);
        vBetaOutput2[lane + 32] =
            __hmul2(vInput2[lane + 32], beta2);
        if (lane == 0) {
            gOutput[valueRow] = gateValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(__nv_bfloat16 *input, float *weight, __nv_bfloat16 *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 1. 向量化加载 (nv_bfloat162)，每个线程累加平方和
    int bf2_channels = channels / 2;
    const __nv_bfloat162 *input_bf2 = reinterpret_cast<const __nv_bfloat162 *>(input);
    float sum2 = 0.0f;
    for (int i = tid; i < bf2_channels; i += THREAD_PER_BLOCK) {
        __nv_bfloat162 v = input_bf2[i];
        float lo = __bfloat162float(v.x);
        float hi = __bfloat162float(v.y);
        sum2 += lo * lo + hi * hi;
    }
    if (channels & 1) {
        int last = channels - 1;
        if (tid == 0) {
            float x = __bfloat162float(input[last]);
            sum2 += x * x;
        }
    }

    // 2. Warp shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / channels + eps);
        }
    }
    __syncthreads();

    // 3. 向量化写出
    float s = scale;
    __nv_bfloat162 *output_bf2 = reinterpret_cast<__nv_bfloat162 *>(output);
    for (int i = tid; i < bf2_channels; i += THREAD_PER_BLOCK) {
        __nv_bfloat162 v = input_bf2[i];
        float lo = __bfloat162float(v.x);
        float hi = __bfloat162float(v.y);
        float w0 = __ldg(&weight[i * 2]);
        float w1 = __ldg(&weight[i * 2 + 1]);
        __nv_bfloat162 out_val;
        out_val.x = __float2bfloat16_rn(lo * s * w0);
        out_val.y = __float2bfloat16_rn(hi * s * w1);
        output_bf2[i] = out_val;
    }
    if ((channels & 1) && tid == 0) {
        int last = channels - 1;
        output[last] = __float2bfloat16_rn(__bfloat162float(input[last]) * s * __ldg(&weight[last]));
    }
}

#ifndef USE_ROCM
// hidden=3072 uses three bfloat162 values per thread in the legacy 512-thread
// kernel.  A physical thread evaluates two independent legacy thread lanes and
// keeps all six packed values in registers until the scale is available.  The
// two warp reductions are stored in the same 16 slots and combined in the same
// order as the legacy kernel, so this removes the second input read without
// changing the reduction tree.
__global__ __launch_bounds__(256) void FastllmRMSNormBFloat16Hidden3072ExactKernel(
        const __nv_bfloat16 *input, const float *weight,
        __nv_bfloat16 *output, float eps) {
    constexpr int CHANNELS = 3072;
    constexpr int BF16_PAIRS = CHANNELS / 2;
    constexpr int LEGACY_THREADS = 512;
    constexpr int PHYSICAL_THREADS = 256;
    constexpr int PHYSICAL_WARPS = PHYSICAL_THREADS / 32;
    constexpr int LEGACY_WARPS = LEGACY_THREADS / 32;
    constexpr int PAIRS_PER_LEGACY_THREAD = BF16_PAIRS / LEGACY_THREADS;
    static_assert(LEGACY_THREADS == PHYSICAL_THREADS * 2,
                  "each physical thread must emulate two legacy threads");
    static_assert(BF16_PAIRS % LEGACY_THREADS == 0,
                  "BF16 pairs must divide evenly across legacy threads");

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    input += (size_t)row * CHANNELS;
    output += (size_t)row * CHANNELS;

    const __nv_bfloat162 *input2 =
        reinterpret_cast<const __nv_bfloat162 *>(input);
    __nv_bfloat162 values0[PAIRS_PER_LEGACY_THREAD];
    __nv_bfloat162 values1[PAIRS_PER_LEGACY_THREAD];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#pragma unroll
    for (int i = 0; i < PAIRS_PER_LEGACY_THREAD; i++) {
        int pair0 = tid + i * LEGACY_THREADS;
        int pair1 = pair0 + PHYSICAL_THREADS;
        values0[i] = input2[pair0];
        values1[i] = input2[pair1];
        float lo0 = __bfloat162float(values0[i].x);
        float hi0 = __bfloat162float(values0[i].y);
        float lo1 = __bfloat162float(values1[i].x);
        float hi1 = __bfloat162float(values1[i].y);
        sum0 += lo0 * lo0 + hi0 * hi0;
        sum1 += lo1 * lo1 + hi1 * hi1;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }

    __shared__ float warpSums[LEGACY_WARPS];
    __shared__ float scale;
    if (lane == 0) {
        warpSums[warp] = sum0;
        warpSums[warp + PHYSICAL_WARPS] = sum1;
    }
    __syncthreads();

    if (warp == 0) {
        float total = lane < LEGACY_WARPS ? warpSums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) {
            scale = rsqrtf(total / CHANNELS + eps);
        }
    }
    __syncthreads();

    __nv_bfloat162 *output2 = reinterpret_cast<__nv_bfloat162 *>(output);
    float rowScale = scale;
#pragma unroll
    for (int i = 0; i < PAIRS_PER_LEGACY_THREAD; i++) {
        int pair0 = tid + i * LEGACY_THREADS;
        int pair1 = pair0 + PHYSICAL_THREADS;
        float lo0 = __bfloat162float(values0[i].x);
        float hi0 = __bfloat162float(values0[i].y);
        float lo1 = __bfloat162float(values1[i].x);
        float hi1 = __bfloat162float(values1[i].y);
        __nv_bfloat162 normalized0;
        __nv_bfloat162 normalized1;
        normalized0.x = __float2bfloat16_rn(
            lo0 * rowScale * __ldg(weight + pair0 * 2));
        normalized0.y = __float2bfloat16_rn(
            hi0 * rowScale * __ldg(weight + pair0 * 2 + 1));
        normalized1.x = __float2bfloat16_rn(
            lo1 * rowScale * __ldg(weight + pair1 * 2));
        normalized1.y = __float2bfloat16_rn(
            hi1 * rowScale * __ldg(weight + pair1 * 2 + 1));
        output2[pair0] = normalized0;
        output2[pair1] = normalized1;
    }
}
#endif

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(float *input, float *gamma, float *beta, float *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] - mean) / var * gamma[i] + beta[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(half *input, float *gamma, float *beta, half *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = __half2float(input[i]);
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = __float2half((__half2float(input[i]) - mean) / var * gamma[i] + beta[i]);
    }
}

__device__ __forceinline__ bool FastllmGreedyIsBetter(
        float value, int id, float bestValue, int bestId) {
    return value > bestValue ||
           (value == bestValue && id < bestId);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    idData[tid] = tid;
    maxData[tid] = -1e100;
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        if (inputData[j] > maxData[tid]) {
            maxData[tid] = inputData[j];
            idData[tid] = j;
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && FastllmGreedyIsBetter(
                maxData[tid + s], (int)idData[tid + s],
                maxData[tid], (int)idData[tid])) {
            maxData[tid] = maxData[tid + s];
            idData[tid] = idData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputData[0] = idData[0];
        outputData[1] = maxData[0];
    }
}

template <int THREAD_PER_BLOCK, int MAXK>
__global__ void FastllmLayerNormKernelTopK(float *input, float *output, int K, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2 * K;
    int tid = threadIdx.x;
    idData[tid][0] = tid;
    for (int i = 0; i < K; i++) {
        maxData[tid][i] = -1e100;
    }
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        float cur = inputData[j];
        for (int l = 0; l < K; l++) {
            if (cur > maxData[tid][l]) {
                for (int x = K - 1; x > l; x--) {
                    maxData[tid][x] = maxData[tid][x - 1];
                    idData[tid][x] = idData[tid][x - 1];
                }
                maxData[tid][l] = cur;
                idData[tid][l] = j;
                break;
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < K) {
                if (maxData[tid][pos0] > maxData[tid + s][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            int pos = K - 1;
            while (pos >= 0) {
                if (pos1 < 0 || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < K; i++) {
            outputData[i * 2] = idData[0][i];
            outputData[i * 2 + 1] = maxData[0][i];
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSplitBatchKernel(uint8_t *input, uint8_t **outputs, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = input + oid * channels * inner + bid * inner;
    uint8_t *curOutput = outputs[bid] + oid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmCatBatchKernel(uint8_t **inputs, uint8_t *output, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = inputs[bid] + oid * inner;
    uint8_t *curOutput = output + oid * channels * inner + bid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

void *FastllmCudaPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = FastllmCudaMalloc(input.expansionBytes);
        if (ret == nullptr) {
            return nullptr;
        }
        auto state = cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
            FastllmCudaFree(ret);
            return nullptr;
        }
    }
    return ret;
}

void FastllmCudaFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        FastllmCudaFree(data);
    }
}

void *FastllmCudaPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (float*)output.cudaData;
    } else {
        ret = (float*)FastllmCudaMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmCudaFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::CUDA) {
        auto state = cudaMemcpy(output.cpuData, data, output.expansionBytes, cudaMemcpyDeviceToHost);
        checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
        FastllmCudaFree(data);
    }

    DeviceSync();
}

/** 保存Data分配时的紧凑快照；只用于OOM报告，不持有或解引用owner。 */
struct CudaAllocationMetadata {
    const void *owner = nullptr;
    int dataType = -1;
    int dimCount = 0;
    std::array<int, 4> dims = {0, 0, 0, 0};
    bool isFake = false;
    bool isKVCache = false;
};

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;
    int graphPins;
    cudaEvent_t reuseReadyEvent;
    bool reusePending;
    const char *sourceFile;
    int sourceLine;
    size_t requestedSize;
    CudaAllocationMetadata metadata;

    CudaMemoryBuffer () : data(nullptr), size(0), busy(false), graphPins(0),
            reuseReadyEvent(nullptr), reusePending(false),
            sourceFile("<unknown>"), sourceLine(0), requestedSize(0) {}

    CudaMemoryBuffer (void *data, size_t size, bool busy,
                      const char *sourceFile = nullptr, int sourceLine = 0,
                      const CudaAllocationMetadata *metadata = nullptr) :
            data(data), size(size), busy(busy), graphPins(0),
            reuseReadyEvent(nullptr), reusePending(false),
            sourceFile(sourceFile == nullptr ? "<unknown>" : sourceFile),
            sourceLine(sourceLine), requestedSize(size),
            metadata(metadata == nullptr ? CudaAllocationMetadata() : *metadata) {}
};

/** 更新池中Buffer当前使用者的源码位置。 */
static void FastllmCudaSetBufferSource(CudaMemoryBuffer &buffer,
                                       const char *file, int line,
                                       size_t requestedSize,
                                       const CudaAllocationMetadata *metadata) {
    buffer.sourceFile = file == nullptr ? "<unknown>" : file;
    buffer.sourceLine = line;
    buffer.requestedSize = requestedSize;
    buffer.metadata = metadata == nullptr
        ? CudaAllocationMetadata() : *metadata;
}

static bool FastllmCudaBufferReadyForReuseLocked(CudaMemoryBuffer &buffer) {
    if (!buffer.reusePending) {
        return true;
    }
    cudaError_t state = cudaEventQuery(buffer.reuseReadyEvent);
    if (state == cudaSuccess) {
        buffer.reusePending = false;
        return true;
    }
    if (state == cudaErrorNotReady) {
        return false;
    }
    checkCudaErrors("Error: CUDA error when checking deferred pool reuse!", state);
    return false;
}

static void FastllmCudaDestroyReuseEventLocked(CudaMemoryBuffer &buffer) {
    if (buffer.reuseReadyEvent == nullptr) {
        return;
    }
    cudaError_t state = cudaEventDestroy(buffer.reuseReadyEvent);
    buffer.reuseReadyEvent = nullptr;
    buffer.reusePending = false;
    checkCudaErrors("Error: CUDA error when destroying deferred pool event!", state);
}
std::map<int, std::vector <CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, int> cudaBuffersMinId; // 最小的空闲id
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

// fastllmCudaMemPoolMetaMutex 仅保护上面这些 map 结构本身（查找/插入新设备条目），
// 持锁期间绝不调用任何 CUDA API，避免跨设备线程互相阻塞。
static std::mutex fastllmCudaMemPoolMetaMutex;
// 每个设备一把锁：内存池的 cudaMalloc/cudaFree（会隐式同步本设备）只在对应设备锁内进行，
// 不会阻塞其它设备线程。这是张量并行下避免死锁的关键：
// 否则一个 rank 持有跨设备全局锁时执行 cudaMalloc/cudaFree（隐式同步本设备），
// 而本设备 stream 上挂着需要其它 rank 共同完成的 NCCL 集合通信，
// 其它 rank 又卡在等待这把全局锁，从而形成跨 rank 死锁
// （典型现象：单卡 100%、其余 0%，且卡住的卡不固定）。
static std::map<int, std::unique_ptr<std::mutex>> fastllmCudaMemPoolDeviceLocks;

// 某个设备的内存池视图：均为指向全局 map 内元素的稳定指针。
// std::map 保证元素引用/指针在其它 key 插入或删除时仍然有效，
// 因此在元锁下取得这些指针后，可在设备锁（而非元锁）下安全地修改其内容。
struct FastllmCudaMemPoolView {
    std::mutex *lock = nullptr;
    std::vector<CudaMemoryBuffer> *bigBuffers = nullptr;
    std::vector<CudaMemoryBuffer> *smallBuffers = nullptr;
    int *minId = nullptr;
    size_t *noBusy = nullptr;
    int device = -1;
};

// 在元锁下创建/获取指定设备的池视图与设备锁。
static FastllmCudaMemPoolView FastllmGetCudaMemPoolView(int id) {
    std::lock_guard<std::mutex> meta(fastllmCudaMemPoolMetaMutex);
    FastllmCudaMemPoolView view;
    view.device = id;
    view.bigBuffers = &bigBuffersMap[id];
    view.smallBuffers = &cudaBuffersMap[id];
    view.minId = &cudaBuffersMinId[id];
    view.noBusy = &noBusyCnt[id];
    auto &lk = fastllmCudaMemPoolDeviceLocks[id];
    if (lk == nullptr) {
        lk = std::unique_ptr<std::mutex>(new std::mutex());
    }
    view.lock = lk.get();
    return view;
}

// 在元锁下对当前所有设备做一次快照（仅复制稳定指针），
// 用于需要跨设备查找/清理的场景（FastllmCudaFree / FastllmCudaClearBigBuffer）。
// 拿到快照后即释放元锁，再逐设备各自加锁处理，绝不同时持有两把设备锁去调用 CUDA。
static std::vector<FastllmCudaMemPoolView> FastllmSnapshotCudaMemPoolViews() {
    std::lock_guard<std::mutex> meta(fastllmCudaMemPoolMetaMutex);
    std::vector<FastllmCudaMemPoolView> views;
    for (auto &it : bigBuffersMap) {
        int id = it.first;
        FastllmCudaMemPoolView view;
        view.device = id;
        view.bigBuffers = &it.second;
        view.smallBuffers = &cudaBuffersMap[id];
        view.minId = &cudaBuffersMinId[id];
        view.noBusy = &noBusyCnt[id];
        auto &lk = fastllmCudaMemPoolDeviceLocks[id];
        if (lk == nullptr) {
            lk = std::unique_ptr<std::mutex>(new std::mutex());
        }
        view.lock = lk.get();
        views.push_back(view);
    }
    for (auto &it : cudaBuffersMap) {
        int id = it.first;
        if (bigBuffersMap.find(id) != bigBuffersMap.end()) {
            continue;
        }
        FastllmCudaMemPoolView view;
        view.device = id;
        view.bigBuffers = &bigBuffersMap[id];
        view.smallBuffers = &it.second;
        view.minId = &cudaBuffersMinId[id];
        view.noBusy = &noBusyCnt[id];
        auto &lk = fastllmCudaMemPoolDeviceLocks[id];
        if (lk == nullptr) {
            lk = std::unique_ptr<std::mutex>(new std::mutex());
        }
        view.lock = lk.get();
        views.push_back(view);
    }
    return views;
}

static size_t FastllmCudaReleaseIdleBigBuffersLocked(int id, std::vector<CudaMemoryBuffer> &bigBuffers) {
    size_t released = 0;
    std::vector<CudaMemoryBuffer> keep;
    keep.reserve(bigBuffers.size());
    cudaError_t state = cudaSetDevice(id);
    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when set device before release idle memory!", state);
        return 0;
    }
    if (fastllmCudaNcclActive.load(std::memory_order_relaxed)) {
        cudaDeviceSynchronize();
    }
    for (auto &buffer : bigBuffers) {
        if (buffer.busy || !FastllmCudaBufferReadyForReuseLocked(buffer)) {
            keep.push_back(buffer);
            continue;
        }
        FastllmCudaDestroyReuseEventLocked(buffer);
        state = cudaFree(buffer.data);
        if (cudaSuccess == state) {
            released += buffer.size;
        } else {
            printf("Error: CUDA error when release idle pooled memory on device %d!", id);
            checkCudaErrors("", state);
            keep.push_back(buffer);
        }
    }
    bigBuffers.swap(keep);
    return released;
}

static cudaError_t FastllmCudaCheckedMallocWithIdlePoolRetry(
        void **ret, size_t size, int id, FastllmCudaMemPoolView &view, const char *file, int line) {
    if (ret != nullptr) {
        *ret = nullptr;
    }
    cudaError_t state = FastllmCudaCheckedMalloc(ret, size, file, line);
    if (cudaSuccess == state || fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
        return state;
    }
    cudaGetLastError();
    size_t released = FastllmCudaReleaseIdleBigBuffersLocked(id, *view.bigBuffers);
    if (released == 0) {
        return state;
    }
    return FastllmCudaCheckedMalloc(ret, size, file, line);
}

static size_t fastllmCudaMemPoolAllocated = 0;
static size_t fastllmCudaMemPoolPeak = 0;

struct FastllmCudaWeightSlab {
    void *base = nullptr;
    size_t size = 0;
    size_t used = 0;
    int activeBlocks = 0;
    std::string group;
};

struct FastllmCudaWeightSlabPtr {
    int device = -1;
    void *base = nullptr;
};

static std::mutex fastllmCudaWeightSlabMutex;
static std::atomic<size_t> fastllmCudaWeightSlabBytes(0);
static std::atomic<size_t> fastllmCudaWeightSlabPtrCount(0);
static std::map<int, std::vector<FastllmCudaWeightSlab> > fastllmCudaWeightSlabs;
static std::map<void*, FastllmCudaWeightSlabPtr> fastllmCudaWeightSlabPtrs;

void FastllmCudaSetWeightSlabBytes(size_t bytes) {
    fastllmCudaWeightSlabBytes.store(bytes, std::memory_order_relaxed);
}

size_t FastllmCudaGetWeightSlabBytes() {
    return fastllmCudaWeightSlabBytes.load(std::memory_order_relaxed);
}

static size_t FastllmCudaAlignBytes(size_t size, size_t align) {
    return ((size + align - 1) / align) * align;
}

static std::string FastllmCudaWeightSlabGroup(const std::string &name) {
    // Expert-parallel source tensors are consolidated and released one layer
    // at a time during the first ForwardGPU call.  Do not mix different layers
    // in the same slab, otherwise one live tensor from a later layer pins all
    // already-consumed blocks and makes the repack peak grow every layer.
    const char *markers[] = {".moe.experts.", ".ffn.experts."};
    for (const char *marker : markers) {
        size_t pos = name.find(marker);
        if (pos != std::string::npos) {
            return name.substr(0, pos + std::strlen(marker));
        }
    }
    return "";
}

void *FastllmCudaMallocModelWeight(size_t size, const std::string &name) {
    size_t slabBytes = FastllmCudaGetWeightSlabBytes();
    if (slabBytes == 0 || size == 0 || size > slabBytes / 2) {
        return FastllmCudaMalloc(size);
    }

    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);

    const size_t align = 256;
    size_t aligned = FastllmCudaAlignBytes(size, align);
    std::string group = FastllmCudaWeightSlabGroup(name);
    std::lock_guard<std::mutex> lock(fastllmCudaWeightSlabMutex);

    auto &slabs = fastllmCudaWeightSlabs[id];
    int slabId = -1;
    for (int i = 0; i < slabs.size(); i++) {
        if (slabs[i].group == group && slabs[i].size >= slabs[i].used &&
            slabs[i].size - slabs[i].used >= aligned) {
            slabId = i;
            break;
        }
    }

    if (slabId < 0) {
        void *base = nullptr;
        // Weight preparation can run after temporary CUDA buffers have already
        // warmed the regular allocator pool.  Near the capacity limit (for
        // example, Laguna FP8 on four 32 GB GPUs), those idle buffers can make
        // a new weight slab fail even though they are immediately reclaimable.
        // Match the retry policy used by ordinary and direct CUDA allocations
        // before reporting a real model-weight OOM.
        FastllmCudaMemPoolView poolView = FastllmGetCudaMemPoolView(id);
        {
            std::lock_guard<std::mutex> poolLock(*poolView.lock);
            state = FastllmCudaCheckedMallocWithIdlePoolRetry(
                &base, slabBytes, id, poolView, __FILE__, __LINE__);
        }
        if (cudaSuccess != state) {
            printf("Error: CUDA error when allocating model weight slab %lu MB memory! maybe there's no enough memory left on device.",
                   slabBytes >> 20);
            checkCudaErrors("", state);
            return nullptr;
        }
        FastllmCudaWeightSlab slab;
        slab.base = base;
        slab.size = slabBytes;
        slab.used = 0;
        slab.activeBlocks = 0;
        slab.group = group;
        slabs.push_back(slab);
        slabId = (int)slabs.size() - 1;
    }

    FastllmCudaWeightSlab &slab = slabs[slabId];
    void *ret = (void*)((uint8_t*)slab.base + slab.used);
    slab.used += aligned;
    slab.activeBlocks++;
    fastllmCudaWeightSlabPtrs[ret] = {id, slab.base};
    fastllmCudaWeightSlabPtrCount.fetch_add(1, std::memory_order_relaxed);
    return ret;
}

static bool FastllmCudaTryFreeWeightSlabPtr(void *ret) {
    if (ret == nullptr) {
        return false;
    }
    if (fastllmCudaWeightSlabPtrCount.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(fastllmCudaWeightSlabMutex);
    auto it = fastllmCudaWeightSlabPtrs.find(ret);
    if (it == fastllmCudaWeightSlabPtrs.end()) {
        return false;
    }

    int id = it->second.device;
    void *base = it->second.base;
    fastllmCudaWeightSlabPtrs.erase(it);
    fastllmCudaWeightSlabPtrCount.fetch_sub(1, std::memory_order_relaxed);

    auto slabsIt = fastllmCudaWeightSlabs.find(id);
    if (slabsIt != fastllmCudaWeightSlabs.end()) {
        auto &slabs = slabsIt->second;
        for (int i = 0; i < slabs.size(); i++) {
            if (slabs[i].base == base) {
                slabs[i].activeBlocks--;
                if (slabs[i].activeBlocks <= 0) {
                    int oriId = -1;
                    cudaGetDevice(&oriId);
                    cudaSetDevice(id);
                    cudaError_t state = cudaFree(slabs[i].base);
                    if (oriId >= 0 && oriId != id) {
                        cudaSetDevice(oriId);
                    }
                    slabs.erase(slabs.begin() + i);
                    checkCudaErrors("CUDA error when release model weight slab!", state);
                }
                return true;
            }
        }
    }
    return true;
}

#ifdef CUDA_MEM_DEBUG
#include <execinfo.h>
#include <cxxabi.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <algorithm>

struct CudaMemDebugInfo {
    size_t size;
    std::string callstack;
};

static std::mutex cudaMemDebugMutex;
static std::map<void*, CudaMemDebugInfo> cudaMemDebugMap;
static bool cudaMemDebugThreadStarted = false;
static size_t cudaMemDebugPeakUsed = 0;

static std::string CudaMemDebugGetCallStack() {
    const int maxFrames = 128;
    void *frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char **symbols = backtrace_symbols(frames, numFrames);
    std::string result;
    if (symbols) {
        int skip = 0;
        int end = std::min(numFrames, skip + 16);
        for (int i = skip; i < end; i++) {
            result += "  #" + std::to_string(i - skip) + " " + symbols[i] + "\n";
        }
        free(symbols);
    }
    return result;
}

// caller must hold cudaMemDebugMutex; suffix is appended to filename (e.g. "" or "_peak_12345MB")
static void CudaMemDebugWriteReport(const std::string &suffix) {
    mkdir("Debug", 0755);

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::tm tm_buf;
    localtime_r(&t, &tm_buf);

    std::ostringstream fnss;
    fnss << "Debug/"
         << std::put_time(&tm_buf, "%Y%m%d_%H%M%S")
         << "_" << std::setfill('0') << std::setw(3) << ms.count()
         << suffix << ".txt";
    std::string filename = fnss.str();

    size_t totalSize = 0;
    size_t totalCount = cudaMemDebugMap.size();
    std::map<size_t, size_t> sizeDistribution;
    for (auto &it : cudaMemDebugMap) {
        totalSize += it.second.size;
        sizeDistribution[it.second.size]++;
    }

    size_t bigPoolTotal = 0, bigPoolBusy = 0, bigPoolFreeCount = 0, bigPoolBusyCount = 0;
    size_t smallPoolTotal = 0, smallPoolBusy = 0, smallPoolFreeCount = 0, smallPoolBusyCount = 0;
    for (auto &dev : bigBuffersMap) {
        for (auto &b : dev.second) {
            bigPoolTotal += b.size;
            if (b.busy) { bigPoolBusy += b.size; bigPoolBusyCount++; }
            else { bigPoolFreeCount++; }
        }
    }
    for (auto &dev : cudaBuffersMap) {
        for (auto &b : dev.second) {
            smallPoolTotal += b.size;
            if (b.busy) { smallPoolBusy += b.size; smallPoolBusyCount++; }
            else { smallPoolFreeCount++; }
        }
    }

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;

    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;

    ofs << "========== CUDA Memory Debug Report ==========\n";
    ofs << "Time: " << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "\n";
    if (!suffix.empty()) ofs << "Trigger: PEAK memory\n";
    ofs << "\n";

    ofs << "--- Summary ---\n";
    ofs << "GPU Memory: used " << (usedMem >> 20) << " MB, free " << (freeMem >> 20) << " MB / total " << (totalMem >> 20) << " MB\n";
    ofs << "Tracked allocations: " << totalCount << " pointers, total " << std::fixed << std::setprecision(2) << (double)totalSize / (1024.0 * 1024.0) << " MB\n\n";

    ofs << "Big buffer pool:   total " << (bigPoolTotal >> 20) << " MB, busy " << (bigPoolBusy >> 20) << " MB"
        << " (busy " << bigPoolBusyCount << ", free " << bigPoolFreeCount << ")\n";
    ofs << "Small buffer pool: total " << (smallPoolTotal >> 20) << " MB, busy " << (smallPoolBusy >> 20) << " MB"
        << " (busy " << smallPoolBusyCount << ", free " << smallPoolFreeCount << ")\n";
    ofs << "Pool allocated total: " << (fastllmCudaMemPoolAllocated >> 20) << " MB, peak: " << (fastllmCudaMemPoolPeak >> 20) << " MB\n\n";

    ofs << "--- Size Distribution (tracked) ---\n";
    std::vector<std::pair<size_t, size_t>> sortedDist(sizeDistribution.begin(), sizeDistribution.end());
    std::sort(sortedDist.begin(), sortedDist.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });
    for (auto &p : sortedDist) {
        double sizeMB = (double)p.first / (1024.0 * 1024.0);
        if (sizeMB >= 1.0)
            ofs << "  " << std::fixed << std::setprecision(2) << sizeMB << " MB : " << p.second << " blocks\n";
        else
            ofs << "  " << (p.first / 1024) << " KB : " << p.second << " blocks\n";
    }

    ofs << "\n--- Free Buffers in Pool ---\n";
    for (auto &dev : bigBuffersMap) {
        size_t devFreeSize = 0, devFreeCount = 0;
        for (auto &b : dev.second) {
            if (!b.busy) { devFreeSize += b.size; devFreeCount++; }
        }
        if (devFreeCount == 0) continue;
        ofs << "  [Big Pool] Device " << dev.first << ": " << devFreeCount << " free blocks, "
            << std::fixed << std::setprecision(2) << (double)devFreeSize / (1024.0 * 1024.0) << " MB\n";
        for (auto &b : dev.second) {
            if (!b.busy) {
                ofs << "    ptr=" << b.data << ", size=" << std::fixed << std::setprecision(2)
                    << (double)b.size / (1024.0 * 1024.0) << " MB (" << b.size << " bytes)\n";
            }
        }
    }
    for (auto &dev : cudaBuffersMap) {
        size_t devFreeSize = 0, devFreeCount = 0;
        for (auto &b : dev.second) {
            if (!b.busy) { devFreeSize += b.size; devFreeCount++; }
        }
        if (devFreeCount == 0) continue;
        ofs << "  [Small Pool] Device " << dev.first << ": " << devFreeCount << " free blocks, "
            << std::fixed << std::setprecision(2) << (double)devFreeSize / (1024.0 * 1024.0) << " MB\n";
        for (auto &b : dev.second) {
            if (!b.busy) {
                ofs << "    ptr=" << b.data << ", size=" << std::fixed << std::setprecision(2)
                    << (double)b.size / (1024.0 * 1024.0) << " MB (" << b.size << " bytes)\n";
            }
        }
    }

    ofs << "\n--- Unreleased Blocks Detail (" << totalCount << " blocks) ---\n";
    for (auto &it : cudaMemDebugMap) {
        double sizeMB = (double)it.second.size / (1024.0 * 1024.0);
        ofs << "ptr=" << it.first << ", size=" << std::fixed << std::setprecision(2) << sizeMB << " MB (" << it.second.size << " bytes)\n";
        ofs << "  callstack:\n" << it.second.callstack << "\n";
    }

    ofs << "========== End of Report ==========\n";
    ofs.close();

    printf("[CUDA_MEM_DEBUG] Report saved to %s (%zu pointers, %.2f MB tracked, GPU used %zu MB)\n",
           filename.c_str(), totalCount, (double)totalSize / (1024.0 * 1024.0), usedMem >> 20);
    fflush(stdout);
}

static void CudaMemDebugReportThread() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
        CudaMemDebugWriteReport("");
    }
}

static void CudaMemDebugEnsureThread() {
    if (!cudaMemDebugThreadStarted) {
        cudaMemDebugThreadStarted = true;
        std::thread(CudaMemDebugReportThread).detach();
    }
}

static void CudaMemDebugRecord(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
    CudaMemDebugEnsureThread();
    cudaMemDebugMap[ptr] = {size, CudaMemDebugGetCallStack()};

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;
    if (usedMem > cudaMemDebugPeakUsed) {
        cudaMemDebugPeakUsed = usedMem;
        std::string suffix = "_peak_" + std::to_string(usedMem >> 20) + "MB";
        CudaMemDebugWriteReport(suffix);
    }
}

static void CudaMemDebugRemove(void *ptr) {
    std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
    cudaMemDebugMap.erase(ptr);
}
#endif // CUDA_MEM_DEBUG

void * FastllmCudaDirectMalloc(size_t size) {
    if (FastllmCudaGraphIsCapturing()) {
        FastllmCudaSetThreadError();
        return nullptr;
    }
    void * ret = nullptr;
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    state = FastllmCudaCheckedMallocWithIdlePoolRetry(&ret, size, id, view, __FILE__, __LINE__);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRecord(ret, size);
#endif
    return ret;
}

void FastllmCudaDirectFree(void *ret) {
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    cudaError_t state = cudaFree(ret);
    //checkCudaErrors("Error: CUDA error when release memory!", state);
}

void FastllmCudaMemset0(void *ret, size_t size) {
    cudaMemset(ret, 0, size);
}

void FastllmCudaMemPoolStats() {
    int id = -1;
    cudaGetDevice(&id);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    size_t bigTotal = 0, bigBusy = 0;
    size_t smallTotal = 0, smallBusy = 0;
    auto &bigBuffers = *view.bigBuffers;
    for (auto &b : bigBuffers) {
        bigTotal += b.size;
        if (b.busy) bigBusy += b.size;
    }
    auto &cudaBuffers = *view.smallBuffers;
    for (auto &b : cudaBuffers) {
        smallTotal += b.size;
        if (b.busy) smallBusy += b.size;
    }
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("[CUDA_MEM_POOL] dev=%d bigPool: %zu/%zu MB (%zu bufs), smallPool: %zu/%zu MB (%zu bufs), "
           "poolTotal: %zu MB, peak: %zu MB, gpuFree: %zu MB / %zu MB\n",
           id,
           bigBusy >> 20, bigTotal >> 20, bigBuffers.size(),
           smallBusy >> 20, smallTotal >> 20, cudaBuffers.size(),
           fastllmCudaMemPoolAllocated >> 20, fastllmCudaMemPoolPeak >> 20,
           freeMem >> 20, totalMem >> 20);
}

static bool FastllmCudaCanReusePooledBigBuffer(size_t bufferSize, size_t requestSize) {
    if (bufferSize < requestSize) {
        return false;
    }
    size_t maxWaste = 16 * 1024 * 1024;
    if (requestSize >= 64 * 1024 * 1024) {
        return bufferSize <= requestSize * 3;
    }
    if (requestSize >= 32 * 1024 * 1024) {
        return bufferSize <= requestSize * 4;
    }
    if (requestSize >= 8 * 1024 * 1024) {
        return bufferSize <= requestSize * 5;
    }
    return bufferSize <= requestSize * 2 || bufferSize - requestSize < maxWaste;
}

static void FastllmCudaPrintPoolRejectStateLocked(int id, size_t requestSize,
                                                  std::vector<CudaMemoryBuffer> *bigBuffersPtr,
                                                  std::vector<CudaMemoryBuffer> *smallBuffersPtr) {
    if (!fastllm::GetFastllmEnv().cudaMemCheck &&
        (!fastllmCudaMallocDisabled.load(std::memory_order_relaxed) ||
         fastllmCudaMallocRejectLogCount.load(std::memory_order_relaxed) >= 4)) {
        return;
    }
    fprintf(stderr, "[FASTLLM_CUDA_MEM_CHECK] pooled buffers on device %d before rejecting %.2f MB:\n",
            id, requestSize / 1048576.0);
    if (bigBuffersPtr == nullptr || bigBuffersPtr->empty()) {
        fprintf(stderr, "  bigPool: empty\n");
    } else {
        int printed = 0;
        for (int i = 0; i < (int)bigBuffersPtr->size(); i++) {
            auto &b = (*bigBuffersPtr)[i];
            if (b.size < 1024 * 1024 && printed >= 16) {
                continue;
            }
            fprintf(stderr, "  big[%d]: %.2f MB %s %s\n", i, b.size / 1048576.0,
                    b.busy ? "busy" : "free",
                    FastllmCudaCanReusePooledBigBuffer(b.size, requestSize) ? "fits" : "skip");
            printed++;
        }
    }
    if (smallBuffersPtr == nullptr || smallBuffersPtr->empty()) {
        fprintf(stderr, "  smallPool: empty\n");
    } else {
        size_t smallFree = 0, smallBusy = 0;
        int freeCount = 0, busyCount = 0;
        for (auto &b : *smallBuffersPtr) {
            if (b.busy) {
                smallBusy += b.size;
                busyCount++;
            } else {
                smallFree += b.size;
                freeCount++;
            }
        }
        fprintf(stderr, "  smallPool: free %.2f MB (%d), busy %.2f MB (%d)\n",
                smallFree / 1048576.0, freeCount, smallBusy / 1048576.0, busyCount);
    }
    fflush(stderr);
}

/**
 * 在真实OOM后输出设备内存池与请求生命周期摘要。
 *
 * 调用方必须已经持有目标设备的内存池锁。本函数只读取池元数据，不释放
 * Buffer、不执行同步，也不改变正常分配和回退行为。
 *
 * @param id               CUDA设备编号。
 * @param requestSize      本次失败的申请字节数。
 * @param bigBuffersPtr    大块内存池。
 * @param smallBuffersPtr  小块内存池。
 */
static void FastllmCudaPrintOomPoolStateLocked(
        int id, size_t requestSize,
        const std::vector<CudaMemoryBuffer> *bigBuffersPtr,
        const std::vector<CudaMemoryBuffer> *smallBuffersPtr) {
    struct PoolSummary {
        size_t busyBytes = 0;
        size_t freeBytes = 0;
        size_t pinnedBytes = 0;
        size_t pendingBytes = 0;
        int busyCount = 0;
        int freeCount = 0;
        int pinnedCount = 0;
        int pendingCount = 0;
    };
    auto summarize = [](const std::vector<CudaMemoryBuffer> *buffers) {
        PoolSummary summary;
        if (buffers == nullptr) {
            return summary;
        }
        for (const auto &buffer : *buffers) {
            if (buffer.busy) {
                summary.busyBytes += buffer.size;
                summary.busyCount++;
            } else {
                summary.freeBytes += buffer.size;
                summary.freeCount++;
            }
            if (buffer.graphPins > 0) {
                summary.pinnedBytes += buffer.size;
                summary.pinnedCount++;
            }
            if (buffer.reusePending) {
                summary.pendingBytes += buffer.size;
                summary.pendingCount++;
            }
        }
        return summary;
    };

    const PoolSummary big = summarize(bigBuffersPtr);
    const PoolSummary small = summarize(smallBuffersPtr);
    fprintf(stderr,
            "[FASTLLM_CUDA_OOM] device=%d request_mb=%.2f active_response_contexts=%d\n",
            id, requestSize / 1048576.0,
            fastllm::GetActiveResponseContextCount());
    fprintf(stderr,
            "[FASTLLM_CUDA_OOM] big_pool busy_mb=%.2f busy_count=%d "
            "free_mb=%.2f free_count=%d pinned_mb=%.2f pinned_count=%d "
            "pending_mb=%.2f pending_count=%d\n",
            big.busyBytes / 1048576.0, big.busyCount,
            big.freeBytes / 1048576.0, big.freeCount,
            big.pinnedBytes / 1048576.0, big.pinnedCount,
            big.pendingBytes / 1048576.0, big.pendingCount);
    fprintf(stderr,
            "[FASTLLM_CUDA_OOM] small_pool busy_mb=%.2f busy_count=%d "
            "free_mb=%.2f free_count=%d pinned_mb=%.2f pinned_count=%d "
            "pending_mb=%.2f pending_count=%d\n",
            small.busyBytes / 1048576.0, small.busyCount,
            small.freeBytes / 1048576.0, small.freeCount,
            small.pinnedBytes / 1048576.0, small.pinnedCount,
            small.pendingBytes / 1048576.0, small.pendingCount);

    struct OwnerSummary {
        size_t bytes = 0;
        int count = 0;
        const void *exampleOwner = nullptr;
    };
    auto describeMetadata = [](const CudaAllocationMetadata &metadata) {
        if (metadata.owner == nullptr) {
            return std::string("data=none");
        }
        std::string description = "dtype=" + fastllm::GetDataTypeName(
            (fastllm::DataType)metadata.dataType) + " shape=[";
        for (int i = 0; i < metadata.dimCount; i++) {
            if (i > 0) {
                description += ",";
            }
            description += std::to_string(metadata.dims[i]);
        }
        description += "] fake=" + std::to_string((int)metadata.isFake) +
            " kv=" + std::to_string((int)metadata.isKVCache);
        return description;
    };
    std::map<std::tuple<std::string, int, size_t, size_t, std::string>,
             OwnerSummary> owners;
    auto collectOwners = [&](const std::vector<CudaMemoryBuffer> *buffers) {
        if (buffers == nullptr) {
            return;
        }
        for (const auto &buffer : *buffers) {
            if (!buffer.busy) {
                continue;
            }
            auto &owner = owners[{std::string(buffer.sourceFile),
                                  buffer.sourceLine, buffer.size,
                                  buffer.requestedSize,
                                  describeMetadata(buffer.metadata)}];
            owner.bytes += buffer.size;
            owner.count++;
            if (owner.exampleOwner == nullptr) {
                owner.exampleOwner = buffer.metadata.owner;
            }
        }
    };
    collectOwners(bigBuffersPtr);
    collectOwners(smallBuffersPtr);
    std::vector<std::pair<
        std::tuple<std::string, int, size_t, size_t, std::string>,
        OwnerSummary>> rankedOwners(
        owners.begin(), owners.end());
    std::sort(rankedOwners.begin(), rankedOwners.end(),
              [](const auto &left, const auto &right) {
                  return left.second.bytes > right.second.bytes;
              });
    const int ownerLimit = std::min(20, (int)rankedOwners.size());
    for (int i = 0; i < ownerLimit; i++) {
        const auto &key = rankedOwners[i].first;
        const auto &value = rankedOwners[i].second;
        fprintf(stderr,
                "[FASTLLM_CUDA_OOM] busy_owner rank=%d source=%s:%d "
                "buffer_mb=%.2f requested_mb=%.2f total_mb=%.2f count=%d "
                "owner=%p %s\n",
                i + 1, std::get<0>(key).c_str(), std::get<1>(key),
                std::get<2>(key) / 1048576.0,
                std::get<3>(key) / 1048576.0,
                value.bytes / 1048576.0, value.count,
                const_cast<void *>(value.exampleOwner),
                std::get<4>(key).c_str());
    }
    fflush(stderr);
}

static void FastllmCudaReleaseIdleCachedBuffersForDevice(int id) {
    cudaError_t state = cudaSetDevice(id);
    checkCudaErrors("Error: CUDA error when switching device to release cached memory!", state);

    auto bigIt = bigBuffersMap.find(id);
    if (bigIt != bigBuffersMap.end()) {
        auto &bigBuffers = bigIt->second;
        std::vector<CudaMemoryBuffer> busyBuffers;
        busyBuffers.reserve(bigBuffers.size());
        for (auto &buffer : bigBuffers) {
            if (buffer.busy || buffer.graphPins > 0 ||
                !FastllmCudaBufferReadyForReuseLocked(buffer) ||
                FastllmCudaGraphPoolPointerProtectedLocked(buffer.data)) {
                busyBuffers.push_back(buffer);
            } else {
                FastllmCudaDestroyReuseEventLocked(buffer);
                state = cudaFree(buffer.data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when releasing idle big buffer on device %d!", id);
                    checkCudaErrors("", state);
                }
            }
        }
        bigBuffers.swap(busyBuffers);
    }

    auto smallIt = cudaBuffersMap.find(id);
    if (smallIt != cudaBuffersMap.end()) {
        auto &cudaBuffers = smallIt->second;
        std::vector<CudaMemoryBuffer> busyBuffers;
        busyBuffers.reserve(cudaBuffers.size());
        for (auto &buffer : cudaBuffers) {
            if (buffer.busy || buffer.graphPins > 0 ||
                !FastllmCudaBufferReadyForReuseLocked(buffer) ||
                FastllmCudaGraphPoolPointerProtectedLocked(buffer.data)) {
                busyBuffers.push_back(buffer);
            } else {
                FastllmCudaDestroyReuseEventLocked(buffer);
                state = cudaFree(buffer.data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when releasing idle buffer on device %d!", id);
                    checkCudaErrors("", state);
                }
            }
        }
        cudaBuffers.swap(busyBuffers);
        noBusyCnt[id] = 0;
        cudaBuffersMinId[id] = (int)cudaBuffers.size();
    }
}

static bool FastllmCudaRetryMallocAfterReleasingIdle(size_t size, void **ret, int id, const char *file, int line) {
    // cudaFree is forbidden while a stream is being captured. A capture-time
    // pool miss must be reported to the graph path instead of trying the
    // ordinary OOM recovery, which would invalidate every participating rank.
    if (FastllmCudaGraphIsCapturing()) {
        FastllmCudaSetThreadError();
        return false;
    }
    // Once serving has frozen allocations, idle blocks are the reserve that
    // future requests must reuse.  Releasing them cannot make a forbidden
    // allocation succeed and would only destroy the warmed pool.
    if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
        return false;
    }
    cudaGetLastError();
    FastllmCudaReleaseIdleCachedBuffersForDevice(id);
    cudaError_t state = FastllmCudaCheckedMalloc(ret, size, file, line);
    return state == cudaSuccess;
}

bool FastllmCudaGraphMemoryPoolBegin() {
    FastllmCudaClearGraphError();
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) !=
            FASTLLM_CUDA_GRAPH_POOL_IDLE) {
        return false;
    }
    fastllmCudaGraphPoolTouchedDuringCapture.clear();
    fastllmCudaGraphPoolCaptureIds.clear();
    fastllmCudaGraphPoolPhase.store(
        FASTLLM_CUDA_GRAPH_POOL_CAPTURING, std::memory_order_release);
    return true;
}

bool FastllmCudaGraphMemoryPoolEnd(std::vector<void*> &reservedPointers) {
    reservedPointers.clear();
    std::set<void*> remaining;
    {
        std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
        if (fastllmCudaGraphPoolPhase.load(std::memory_order_relaxed) !=
                FASTLLM_CUDA_GRAPH_POOL_CAPTURING) {
            return false;
        }
        fastllmCudaGraphPoolPhase.store(
            FASTLLM_CUDA_GRAPH_POOL_FINALIZING, std::memory_order_release);
        remaining = fastllmCudaGraphPoolTouchedDuringCapture;
    }

    // Pin every captured pool address while holding the same device-pool lock
    // used by malloc/free. A block may simultaneously be busy on behalf of a
    // persistent Data/workspace; graphPins is an independent ownership count,
    // so either owner may release first without exposing the address early.
    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    for (auto &view : views) {
        std::lock_guard<std::mutex> lock(*view.lock);
        for (auto &buffer : *view.smallBuffers) {
            auto it = remaining.find(buffer.data);
            if (it == remaining.end()) {
                continue;
            }
            remaining.erase(it);
            buffer.graphPins++;
            reservedPointers.push_back(buffer.data);
        }
        for (auto &buffer : *view.bigBuffers) {
            auto it = remaining.find(buffer.data);
            if (it == remaining.end()) {
                continue;
            }
            remaining.erase(it);
            buffer.graphPins++;
            reservedPointers.push_back(buffer.data);
        }
        *view.noBusy = 0;
        *view.minId = (int)view.smallBuffers->size();
        for (int i = 0; i < (int)view.smallBuffers->size(); ++i) {
            const CudaMemoryBuffer &buffer = (*view.smallBuffers)[i];
            if (!buffer.busy && buffer.graphPins == 0) {
                *view.noBusy += buffer.size;
                *view.minId = std::min(*view.minId, i);
            }
        }
    }

    bool ok = remaining.empty();
    {
        std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
        fastllmCudaGraphPoolTouchedDuringCapture.clear();
        fastllmCudaGraphPoolCaptureIds.clear();
        fastllmCudaGraphPoolPhase.store(
            FASTLLM_CUDA_GRAPH_POOL_IDLE, std::memory_order_release);
    }
    if (!ok) {
        std::vector<void*> rollback = reservedPointers;
        reservedPointers.clear();
        FastllmCudaGraphMemoryPoolRelease(rollback);
    }
    return ok;
}

void FastllmCudaGraphMemoryPoolAbort() {
    std::lock_guard<std::mutex> guard(fastllmCudaGraphPoolMutex);
    fastllmCudaGraphPoolTouchedDuringCapture.clear();
    fastllmCudaGraphPoolCaptureIds.clear();
    fastllmCudaGraphPoolPhase.store(
        FASTLLM_CUDA_GRAPH_POOL_IDLE, std::memory_order_release);
}

void FastllmCudaGraphMemoryPoolRelease(const std::vector<void*> &reservedPointers) {
    std::set<void*> remaining(reservedPointers.begin(), reservedPointers.end());
    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    for (auto &view : views) {
        std::lock_guard<std::mutex> lock(*view.lock);
        for (int i = 0; i < (int)view.smallBuffers->size(); ++i) {
            CudaMemoryBuffer &buffer = (*view.smallBuffers)[i];
            auto it = remaining.find(buffer.data);
            if (it == remaining.end()) {
                continue;
            }
            if (buffer.graphPins > 0) {
                buffer.graphPins--;
                if (buffer.graphPins == 0 && !buffer.busy) {
                    *view.minId = std::min(*view.minId, i);
                }
            } else {
                FastllmCudaSetThreadError();
            }
            remaining.erase(it);
        }
        for (auto &buffer : *view.bigBuffers) {
            auto it = remaining.find(buffer.data);
            if (it == remaining.end()) {
                continue;
            }
            if (buffer.graphPins > 0) {
                buffer.graphPins--;
            } else {
                FastllmCudaSetThreadError();
            }
            remaining.erase(it);
        }
        *view.noBusy = 0;
        *view.minId = (int)view.smallBuffers->size();
        for (int i = 0; i < (int)view.smallBuffers->size(); ++i) {
            const CudaMemoryBuffer &buffer = (*view.smallBuffers)[i];
            if (!buffer.busy && buffer.graphPins == 0) {
                *view.noBusy += buffer.size;
                *view.minId = std::min(*view.minId, i);
            }
        }
    }
    if (!remaining.empty()) {
        FastllmCudaSetThreadError();
    }
}

/**
 * 从当前设备的CUDA内存池分配Buffer，并记录当前调用位置。
 *
 * @param size        请求字节数。
 * @param sourceFile  发起分配的源码文件；未知时为nullptr。
 * @param sourceLine  发起分配的源码行号；未知时为0。
 * @param metadata    可选Data元数据快照；函数返回前完成复制。
 * @return 成功时返回设备地址，失败时返回nullptr并设置CUDA错误状态。
 */
static void *FastllmCudaMallocImpl(size_t size, const char *sourceFile,
                                   int sourceLine,
                                   const CudaAllocationMetadata *metadata) {
    sourceFile = sourceFile == nullptr ? "<unknown>" : sourceFile;

    int id = -1;
    cudaError_t state = cudaSuccess;
    state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    FastllmCudaGraphCaptureIdentity captureIdentity =
        FastllmCudaGraphCurrentCaptureIdentity();
    const bool capturePoolOnly = captureIdentity.valid ||
        FastllmCudaGraphIsCapturing();
    const bool useAnyFittingPooledBuffer = capturePoolOnly ||
        fastllmCudaMallocDisabled.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(*view.lock);
    if (size > 1024 * 1024) {
        auto &bigBuffers = *view.bigBuffers;
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && bigBuffers[i].graphPins == 0 &&
                FastllmCudaBufferReadyForReuseLocked(bigBuffers[i]) &&
                FastllmCudaGraphPoolPointerReusableLocked(
                    bigBuffers[i].data, captureIdentity) &&
                FastllmCudaCanReusePooledBigBuffer(bigBuffers[i].size, size)) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            FastllmCudaSetBufferSource(
                bigBuffers[selId], sourceFile, sourceLine, size, metadata);
            FastllmCudaGraphPoolAfterAllocLocked(
                bigBuffers[selId].data, captureIdentity);
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
            return bigBuffers[selId].data;
        }
        if (useAnyFittingPooledBuffer) {
            for (int i = 0; i < bigBuffers.size(); i++) {
                if (!bigBuffers[i].busy && bigBuffers[i].graphPins == 0 &&
                    FastllmCudaBufferReadyForReuseLocked(bigBuffers[i]) &&
                    bigBuffers[i].size >= size &&
                    FastllmCudaGraphPoolPointerReusableLocked(
                        bigBuffers[i].data, captureIdentity)) {
                    if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                        selId = i;
                    }
                }
            }
            if (selId != -1) {
                bigBuffers[selId].busy = true;
                FastllmCudaSetBufferSource(
                    bigBuffers[selId], sourceFile, sourceLine, size, metadata);
                FastllmCudaGraphPoolAfterAllocLocked(
                    bigBuffers[selId].data, captureIdentity);
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
                return bigBuffers[selId].data;
            }
        }

        void * ret = nullptr;
        if (useAnyFittingPooledBuffer) {
            FastllmCudaPrintPoolRejectStateLocked(id, size, view.bigBuffers, view.smallBuffers);
        }
        if (capturePoolOnly) {
            FastllmCudaSetThreadError();
            return nullptr;
        }
        state = FastllmCudaCheckedMalloc(
            &ret, size, sourceFile, sourceLine);
        if (cudaSuccess != state && FastllmCudaRetryMallocAfterReleasingIdle(
                size, &ret, id, sourceFile, sourceLine)) {
            state = cudaSuccess;
        }
        if (cudaSuccess != state) {
            FastllmCudaPrintOomPoolStateLocked(
                id, size, view.bigBuffers, view.smallBuffers);
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            printf("Error: CUDA error when allocating %lu MB memory on device %d! "
                   "gpuFree: %lu MB / %lu MB.\n",
                   size >> 20, id, freeMem >> 20, totalMem >> 20);
            fflush(stdout);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(
            ret, size, true, sourceFile, sourceLine, metadata));
        FastllmCudaGraphPoolAfterAllocLocked(ret, captureIdentity);
#ifdef CUDA_MEM_DEBUG
        CudaMemDebugRecord(ret, size);
#endif
        return ret;
    }
    auto &cudaBuffers = *view.smallBuffers;
    for (int i = *view.minId; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy &&
            cudaBuffers[i].graphPins == 0 &&
            FastllmCudaBufferReadyForReuseLocked(cudaBuffers[i]) &&
            FastllmCudaGraphPoolPointerReusableLocked(
                cudaBuffers[i].data, captureIdentity)) {
            cudaBuffers[i].busy = true;
            FastllmCudaSetBufferSource(
                cudaBuffers[i], sourceFile, sourceLine, size, metadata);
            FastllmCudaGraphPoolAfterAllocLocked(
                cudaBuffers[i].data, captureIdentity);
            *view.noBusy -= cudaBuffers[i].size;
            while (*view.minId < cudaBuffers.size() &&
                   (cudaBuffers[*view.minId].busy ||
                    cudaBuffers[*view.minId].graphPins > 0)) {
                (*view.minId)++;
            }
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(cudaBuffers[i].data, size);
#endif
            return cudaBuffers[i].data;
        }
    }
    if (useAnyFittingPooledBuffer) {
        auto &bigBuffers = *view.bigBuffers;
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && bigBuffers[i].graphPins == 0 &&
                FastllmCudaBufferReadyForReuseLocked(bigBuffers[i]) &&
                bigBuffers[i].size >= size &&
                FastllmCudaGraphPoolPointerReusableLocked(
                    bigBuffers[i].data, captureIdentity)) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            FastllmCudaSetBufferSource(
                bigBuffers[selId], sourceFile, sourceLine, size, metadata);
            FastllmCudaGraphPoolAfterAllocLocked(
                bigBuffers[selId].data, captureIdentity);
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
            return bigBuffers[selId].data;
        }
    }
    void * ret = nullptr;
    if (useAnyFittingPooledBuffer) {
        FastllmCudaPrintPoolRejectStateLocked(id, size, view.bigBuffers, view.smallBuffers);
    }
    if (capturePoolOnly) {
        FastllmCudaSetThreadError();
        return nullptr;
    }
    state = FastllmCudaCheckedMalloc(&ret, size, sourceFile, sourceLine);
    if (cudaSuccess != state && FastllmCudaRetryMallocAfterReleasingIdle(
            size, &ret, id, sourceFile, sourceLine)) {
        state = cudaSuccess;
    }
    if (cudaSuccess != state) {
        FastllmCudaPrintOomPoolStateLocked(
            id, size, view.bigBuffers, view.smallBuffers);
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        printf("Error: CUDA error when allocating %lu KB memory on device %d! "
               "gpuFree: %lu MB / %lu MB.\n",
               size >> 10, id, freeMem >> 20, totalMem >> 20);
        fflush(stdout);
        checkCudaErrors("", state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(
        ret, size, true, sourceFile, sourceLine, metadata));
    FastllmCudaGraphPoolAfterAllocLocked(ret, captureIdentity);
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRecord(ret, size);
#endif
    return ret;
}

/** 使用未知调用位置执行兼容的CUDA池分配。 */
void *FastllmCudaMalloc(size_t size) {
    return FastllmCudaMallocImpl(size, nullptr, 0, nullptr);
}

/** 使用源码位置标记执行CUDA池分配，供OOM诊断归因。 */
void *FastllmCudaMallocWithSource(size_t size, const char *file, int line) {
    return FastllmCudaMallocImpl(size, file, line, nullptr);
}

/**
 * 使用Data的紧凑元数据标记CUDA池分配。
 *
 * 仅复制owner地址、dtype、前四维形状和所有权标记；OOM报告不会解引用
 * owner，因此Data析构后地址失效也不会造成访问越界。正常分配路径不构造
 * 字符串，避免给模型热路径引入明显开销。
 *
 * @param size      请求字节数。
 * @param owner     发起分配的Data地址，仅作为诊断标识。
 * @param dataType  DataType枚举整数值。
 * @param dims      逻辑形状数组，可为nullptr。
 * @param dimCount  dims维数；报告最多保存前四维。
 * @param isFake    Data是否为非拥有型视图。
 * @param isKVCache Data是否为KV Cache。
 * @param file      发起分配的源码文件。
 * @param line      发起分配的源码行号。
 * @return 成功时返回设备地址，失败时返回nullptr。
 */
void *FastllmCudaMallocWithDataTag(size_t size, const void *owner,
                                   int dataType, const int *dims, int dimCount,
                                   bool isFake, bool isKVCache,
                                   const char *file, int line) {
    CudaAllocationMetadata metadata;
    metadata.owner = owner;
    metadata.dataType = dataType;
    metadata.dimCount = dims == nullptr
        ? 0 : std::max(0, std::min(dimCount, 4));
    for (int i = 0; i < metadata.dimCount; i++) {
        metadata.dims[i] = dims[i];
    }
    metadata.isFake = isFake;
    metadata.isKVCache = isKVCache;
    return FastllmCudaMallocImpl(size, file, line, &metadata);
}

void FastllmCudaForceFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (FastllmCudaTryFreeWeightSlabPtr(ret)) {
        return;
    }
    int oriId = FastllmCudaGetDevice();
    cudaError_t state = cudaSuccess;

    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    for (auto &view : views) {
        std::lock_guard<std::mutex> lock(*view.lock);
        auto &cudaBuffers = *view.smallBuffers;
        for (int i = 0; i < (int)cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                if (cudaBuffers[i].graphPins > 0 ||
                    FastllmCudaGraphPoolPointerProtectedLocked(ret)) {
                    if (cudaBuffers[i].busy) {
                        cudaBuffers[i].busy = false;
                        if (cudaBuffers[i].graphPins == 0) {
                            *view.noBusy += cudaBuffers[i].size;
                            *view.minId = std::min(*view.minId, i);
                        }
                    }
#ifdef CUDA_MEM_DEBUG
                    CudaMemDebugRemove(ret);
#endif
                    return;
                }
                state = cudaSetDevice(view.device);
                FastllmCudaDestroyReuseEventLocked(cudaBuffers[i]);
                state = cudaFree(cudaBuffers[i].data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when force releasing memory on device %d!", view.device);
                }
                cudaBuffers.erase(cudaBuffers.begin() + i);
                *view.noBusy = 0;
                *view.minId = (int)cudaBuffers.size();
                for (int j = 0; j < (int)cudaBuffers.size(); j++) {
                    if (!cudaBuffers[j].busy && cudaBuffers[j].graphPins == 0) {
                        *view.noBusy += cudaBuffers[j].size;
                        *view.minId = std::min(*view.minId, j);
                    }
                }
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                FastllmCudaSetDevice(oriId);
                checkCudaErrors("CUDA error when force releasing memory!", state);
                return;
            }
        }
        auto &bigBuffers = *view.bigBuffers;
        for (int i = 0; i < (int)bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                if (bigBuffers[i].graphPins > 0 ||
                    FastllmCudaGraphPoolPointerProtectedLocked(ret)) {
                    bigBuffers[i].busy = false;
#ifdef CUDA_MEM_DEBUG
                    CudaMemDebugRemove(ret);
#endif
                    return;
                }
                state = cudaSetDevice(view.device);
                FastllmCudaDestroyReuseEventLocked(bigBuffers[i]);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when force releasing big memory on device %d!", view.device);
                }
                bigBuffers.erase(bigBuffers.begin() + i);
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                FastllmCudaSetDevice(oriId);
                checkCudaErrors("CUDA error when force releasing big memory!", state);
                return;
            }
        }
    }
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    state = cudaFree(ret);
    FastllmCudaSetDevice(oriId);
    checkCudaErrors("CUDA error when force releasing uncached memory!", state);
}

bool FastllmCudaFreeAfterCurrentThreadStream(void *ret) {
    if (ret == nullptr) {
        return true;
    }
    FastllmCudaGraphCaptureIdentity captureIdentity =
        FastllmCudaGraphCurrentCaptureIdentity();
    if (captureIdentity.valid || FastllmCudaGraphIsCapturing()) {
        return false;
    }

    int device = FastllmCudaGetDevice();
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(device);
    std::lock_guard<std::mutex> lock(*view.lock);
    auto deferBuffer = [&](CudaMemoryBuffer &buffer, int smallIndex) {
        if (!buffer.busy || buffer.reusePending ||
            !FastllmCudaGraphPoolBeforeFreeLocked(ret, captureIdentity)) {
            return false;
        }
        if (buffer.reuseReadyEvent == nullptr) {
            cudaError_t createState = cudaEventCreateWithFlags(
                &buffer.reuseReadyEvent, cudaEventDisableTiming);
            if (createState != cudaSuccess) {
                checkCudaErrors(
                    "Error: CUDA error when creating deferred pool event!",
                    createState);
                return false;
            }
        }
        cudaError_t recordState = cudaEventRecord(
            buffer.reuseReadyEvent, cudaStreamPerThread);
        if (recordState != cudaSuccess) {
            checkCudaErrors(
                "Error: CUDA error when recording deferred pool event!",
                recordState);
            return false;
        }
        buffer.reusePending = true;
        buffer.busy = false;
        if (smallIndex >= 0 && buffer.graphPins == 0) {
            *view.noBusy += buffer.size;
            *view.minId = std::min(*view.minId, smallIndex);
        }
#ifdef CUDA_MEM_DEBUG
        CudaMemDebugRemove(ret);
#endif
        return true;
    };

    for (int i = 0; i < (int)view.smallBuffers->size(); ++i) {
        CudaMemoryBuffer &buffer = (*view.smallBuffers)[i];
        if (buffer.data == ret) {
            return deferBuffer(buffer, i);
        }
    }
    for (CudaMemoryBuffer &buffer : *view.bigBuffers) {
        if (buffer.data == ret) {
            return deferBuffer(buffer, -1);
        }
    }
    return false;
}

void FastllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (FastllmCudaTryFreeWeightSlabPtr(ret)) {
        return;
    }
    int oriId = FastllmCudaGetDevice();
    FastllmCudaGraphCaptureIdentity captureIdentity =
        FastllmCudaGraphCurrentCaptureIdentity();

    // 优先只查当前 rank 的本地池。归还池内指针只修改 busy 标记，
    // 不调用 CUDA API。若本地未命中，再在不持锁时查询指针真实设备，
    // 仅检查 owner 的池；避免其它 rank 为释放本地临时张量而先抢 device 0 的锁。
    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    auto releaseFromPool = [&](FastllmCudaMemPoolView &view) {
        std::lock_guard<std::mutex> lock(*view.lock);
        auto &cudaBuffers = *view.smallBuffers;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                if (!FastllmCudaGraphPoolBeforeFreeLocked(
                        ret, captureIdentity)) {
                    return true;
                }
                if (cudaBuffers[i].busy) {
                    cudaBuffers[i].busy = false;
                    if (cudaBuffers[i].graphPins == 0) {
                        *view.noBusy += cudaBuffers[i].size;
                        *view.minId = std::min(*view.minId, i);
                    }
                }
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                return true;
            }
        }
        auto &bigBuffers = *view.bigBuffers;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                if (!FastllmCudaGraphPoolBeforeFreeLocked(
                        ret, captureIdentity)) {
                    return true;
                }
                bigBuffers[i].busy = false;
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                return true;
            }
        }
        return false;
    };

    auto currentView = std::find_if(views.begin(), views.end(),
                                    [&](const FastllmCudaMemPoolView &view) {
                                        return view.device == oriId;
                                    });
    if (currentView != views.end() && releaseFromPool(*currentView)) {
        return;
    }

    int pointerDevice = -1;
    cudaPointerAttributes attributes;
    cudaError_t attributeState = cudaPointerGetAttributes(&attributes, ret);
    if (attributeState == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
        if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
        if (attributes.type == cudaMemoryTypeDevice ||
            attributes.type == cudaMemoryTypeManaged) {
#endif
            pointerDevice = attributes.device;
        }
    } else {
        cudaGetLastError();
    }

    if (pointerDevice >= 0) {
        if (pointerDevice != oriId) {
            auto ownerView = std::find_if(views.begin(), views.end(),
                                          [&](const FastllmCudaMemPoolView &view) {
                                              return view.device == pointerDevice;
                                          });
            if (ownerView != views.end() && releaseFromPool(*ownerView)) {
                return;
            }
        }
    } else {
        // Compatibility fallback for pointers whose attributes cannot be
        // queried. Never hold more than one device lock at a time.
        for (auto &view : views) {
            if (currentView != views.end() && view.lock == currentView->lock) {
                continue;
            }
            if (releaseFromPool(view)) {
                return;
            }
        }
    }

    // 未在任何池中找到：直接释放。此处不持有任何池锁，避免跨设备阻塞。
    if (FastllmCudaGraphPoolPointerProtectedLocked(ret)) {
        return;
    }
    if (fastllmCudaGraphPoolPhase.load(std::memory_order_acquire) !=
            FASTLLM_CUDA_GRAPH_POOL_IDLE) {
        // Whole-step capture only supports allocations owned by FastLLM's
        // tracked pool. Freeing an untracked pointer would invalidate a graph
        // that may retain it, so reject the capture instead.
        FastllmCudaSetThreadError();
        return;
    }
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    if (pointerDevice >= 0 && pointerDevice != oriId) {
        cudaError_t switchState = cudaSetDevice(pointerDevice);
        if (switchState != cudaSuccess) {
            checkCudaErrors("CUDA error when switching to the pointer owner before release!", switchState);
            cudaError_t restoreState = cudaSetDevice(oriId);
            checkCudaErrors("CUDA error when restoring device after a failed release switch!", restoreState);
            return;
        }
    }
    if (fastllmCudaNcclActive.load(std::memory_order_relaxed)) {
        // 同 cudaMalloc：真实 cudaFree 前排空在途 NCCL，避免争用 CUDA 驱动锁导致跨 rank 死锁。
        cudaError_t syncState = cudaDeviceSynchronize();
        checkCudaErrors("CUDA error when synchronizing before release!", syncState);
    }
    cudaError_t state = cudaFree(ret);
    cudaError_t restoreState = cudaSetDevice(oriId);
    checkCudaErrors("CUDA error when release memory!", state);
    checkCudaErrors("CUDA error when restoring device after release!", restoreState);
}

int FastllmCudaTryMallocBigBuffers(size_t size, int count) {
    if (size == 0 || count <= 0) {
        return 0;
    }
    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    if (state != cudaSuccess || id < 0) {
        checkCudaErrors("Error: CUDA error when finding device for reserve allocation!", state);
        return 0;
    }

    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    auto &bigBuffers = *view.bigBuffers;
    int allocated = 0;
    for (; allocated < count; allocated++) {
        void *ret = nullptr;
        // A reserve is itself reusable idle-pool capacity.  Do not use the
        // ordinary OOM retry here: that retry releases existing idle blocks and
        // could turn a partially successful reserve into one final block while
        // still reporting success to the caller.
        state = FastllmCudaCheckedMalloc(&ret, size, __FILE__, __LINE__);
        if (state == cudaSuccess && ret == nullptr) {
            state = cudaErrorMemoryAllocation;
        }
        if (state != cudaSuccess || ret == nullptr) {
            if (ret != nullptr) {
                cudaFree(ret);
                ret = nullptr;
            }
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            fprintf(stderr,
                    "[Fastllm] CUDA reserve allocation stopped on device %d: "
                    "%d/%d blocks of %.0f MB allocated, gpuFree %.0f/%.0f MB, "
                    "error=%s. Existing pool blocks were preserved.\n",
                    id, allocated, count, size / 1048576.0,
                    freeMem / 1048576.0, totalMem / 1048576.0,
                    cudaGetErrorString(state));
            fflush(stderr);
            // Capacity failures are handled by the return value.  Clear the
            // runtime's last-error slot without setting FastLLM's capture-wide
            // error flags; callers that require the full reserve will abort.
            cudaGetLastError();
            if (state != cudaErrorMemoryAllocation) {
                checkCudaErrors(
                    "Error: unexpected CUDA failure during reserve allocation!",
                    state);
            }
            break;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, false));
    }
    return allocated;
}

void FastllmCudaMallocBigBuffer(size_t size) {
    void * ret = nullptr;
    int id = -1;
    cudaGetDevice(&id);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    auto &bigBuffers = *view.bigBuffers;
    auto state = FastllmCudaCheckedMallocWithIdlePoolRetry(&ret, size, id, view, __FILE__, __LINE__);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
        checkCudaErrors("", state);
        return;
    }
    bigBuffers.push_back(CudaMemoryBuffer(ret, size, false));
}

void FastllmCudaClearBigBuffer() {
    if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
        return;
    }
    int id = -1;
    cudaGetDevice(&id);
    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    if (views.empty())
        return;
    cudaError_t state = cudaSuccess;
    // 逐设备各自加锁清理：cudaSetDevice/cudaFree 只在对应设备锁内进行，
    // 不会同时持有多把设备锁，避免跨设备阻塞。
    for (auto &view : views) {
        std::lock_guard<std::mutex> lock(*view.lock);
        state = cudaSetDevice(view.device);
        checkCudaErrors(
            "Error: CUDA error when switching device to clear big buffers!",
            state);
        auto &bigBuffers = *view.bigBuffers;
        std::vector <CudaMemoryBuffer> temp;
        long long littleMemSum = 0;        
        long long littleMemSumLimit = 300 * 1024 * 1024; // 留一小部分复用  
        std::vector <std::pair <std::size_t, int > > v;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && bigBuffers[i].graphPins == 0 &&
                FastllmCudaBufferReadyForReuseLocked(bigBuffers[i])) {
                v.push_back(std::make_pair(bigBuffers[i].size, i));
            }
        }
        std::sort(v.begin(), v.end());
        std::set <int> littleMemIds;
        for (int i = 0; i < v.size(); i++) {
            littleMemSum += v[i].first;
            if (littleMemSum > littleMemSumLimit) {
                break;
            }
            littleMemIds.insert(v[i].second);
        }
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && bigBuffers[i].graphPins == 0 &&
                FastllmCudaBufferReadyForReuseLocked(bigBuffers[i]) &&
                littleMemIds.find(i) == littleMemIds.end() &&
                !FastllmCudaGraphPoolPointerProtectedLocked(
                    bigBuffers[i].data)) {
                state = cudaSetDevice(view.device);
                FastllmCudaDestroyReuseEventLocked(bigBuffers[i]);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state)
                    printf("Error: CUDA error when release memory on device %d!", view.device);
                checkCudaErrors("", state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    cudaSetDevice(id);
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaCopyFromPinnedHostToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
    checkCudaErrors("Error: CUDA error when async copy from pinned memory to GPU!", state);
}

void FastllmCudaCopyFromHostToDeviceAsync(void *dst, void *src, size_t size, void *stream) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when async copy from memory to GPU!", state);
}

void FastllmCudaCopyFromPinnedHostToDeviceAsync(void *dst, void *src, size_t size, void *stream) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when async copy from pinned memory to GPU!", state);
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
    //cudaDeviceSynchronize();
}

bool FastllmCudaCopyFromDeviceToPinnedHostAsync(
        void *dst, const void *src, size_t size, void *stream) {
    if (size == 0) {
        return true;
    }
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                           (cudaStream_t)stream) == cudaSuccess;
}

bool FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
        void *dst, const void *src, size_t size) {
    if (size == 0) {
        return true;
    }
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                           cudaStreamPerThread) == cudaSuccess;
}

bool FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
        void *dst, const void *src, size_t size) {
    if (size == 0) {
        return true;
    }
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                           cudaStreamPerThread) == cudaSuccess;
}

bool FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
        void *dst, const void *src, size_t size) {
    if (size == 0 || dst == src) {
        return true;
    }
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                           cudaStreamPerThread) == cudaSuccess;
}

namespace {
    constexpr int FASTLLM_CUDA_BATCH_COPY_MAX_SEGMENTS = 128;
    constexpr int FASTLLM_CUDA_BATCH_COPY_BLOCKS_PER_SEGMENT = 4;

    struct FastllmCudaBatchCopyParams {
        const uint8_t *srcs[FASTLLM_CUDA_BATCH_COPY_MAX_SEGMENTS];
        uint8_t *dsts[FASTLLM_CUDA_BATCH_COPY_MAX_SEGMENTS];
        size_t sizes[FASTLLM_CUDA_BATCH_COPY_MAX_SEGMENTS];
        int count;
    };

    __global__ void FastllmCudaBatchCopyKernel(FastllmCudaBatchCopyParams params) {
        int segment = blockIdx.x / FASTLLM_CUDA_BATCH_COPY_BLOCKS_PER_SEGMENT;
        int segmentBlock = blockIdx.x % FASTLLM_CUDA_BATCH_COPY_BLOCKS_PER_SEGMENT;
        if (segment >= params.count) {
            return;
        }

        const uint8_t *src = params.srcs[segment];
        uint8_t *dst = params.dsts[segment];
        size_t bytes = params.sizes[segment];
        size_t first = (size_t)segmentBlock * blockDim.x + threadIdx.x;
        size_t stride = (size_t)FASTLLM_CUDA_BATCH_COPY_BLOCKS_PER_SEGMENT * blockDim.x;
        bool vectorAligned = (((uintptr_t)src | (uintptr_t)dst | bytes) & 15) == 0;
        if (vectorAligned) {
            const uint4 *vectorSrc = reinterpret_cast<const uint4*>(src);
            uint4 *vectorDst = reinterpret_cast<uint4*>(dst);
            size_t vectors = bytes / sizeof(uint4);
            for (size_t i = first; i < vectors; i += stride) {
                vectorDst[i] = vectorSrc[i];
            }
        } else {
            for (size_t i = first; i < bytes; i += stride) {
                dst[i] = src[i];
            }
        }
    }
}

bool FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
        void *const *dsts, const void *const *srcs, const size_t *sizes, int count) {
    if (count < 0 || count > FASTLLM_CUDA_BATCH_COPY_MAX_SEGMENTS ||
        (count > 0 && (dsts == nullptr || srcs == nullptr || sizes == nullptr))) {
        return false;
    }

    FastllmCudaBatchCopyParams params;
    params.count = 0;
    for (int i = 0; i < count; i++) {
        if (sizes[i] == 0 || dsts[i] == srcs[i]) {
            continue;
        }
        if (dsts[i] == nullptr || srcs[i] == nullptr) {
            return false;
        }
        int index = params.count++;
        params.srcs[index] = reinterpret_cast<const uint8_t*>(srcs[i]);
        params.dsts[index] = reinterpret_cast<uint8_t*>(dsts[i]);
        params.sizes[index] = sizes[i];
    }
    if (params.count == 0) {
        return true;
    }

    int blocks = params.count * FASTLLM_CUDA_BATCH_COPY_BLOCKS_PER_SEGMENT;
    FastllmCudaBatchCopyKernel<<<blocks, 256, 0, cudaStreamPerThread>>>(params);
    cudaError_t state = cudaGetLastError();
    checkCudaErrors("Error: CUDA error when launching batched device copy!", state);
    return state == cudaSuccess;
}

void *FastllmCudaHostMalloc(size_t size) {
    void *ptr = nullptr;
    cudaError_t state = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    checkCudaErrors("Error: CUDA error when allocating pinned memory!", state);
    return ptr;
}

void FastllmCudaHostFree(void *ptr) {
    if (ptr != nullptr) {
        cudaError_t state = cudaFreeHost(ptr);
        checkCudaErrors("Error: CUDA error when freeing pinned memory!", state);
    }
}

bool FastllmCudaHostRegister(void *ptr, size_t size) {
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Warning: cudaHostRegister failed (%s), falling back to unpinned memory\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

void FastllmCudaHostUnregister(void *ptr) {
    if (ptr != nullptr) {
        cudaHostUnregister(ptr);
    }
}

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    if (size == 0 || dst == src) {
        return;
    }

    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
    checkCudaErrors("Error: CUDA error when checking CUDA graph capture status!", state);
    if (captureStatus != cudaStreamCaptureStatusNone) {
        state = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        checkCudaErrors("Error: CUDA error when async copy on GPU!", state);
        return;
    }

    state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (state != cudaSuccess && FastllmCudaPagedDebugEnabled()) {
        int device = -1;
        cudaGetDevice(&device);
        fprintf(stderr,
                "[fastllm][paged-debug] FAILED stage=device_to_device_copy "
                "device=%d dst=%p src=%p bytes=%zu error=%d(%s)\n",
                device, dst, src, size, (int)state, cudaGetErrorName(state));
        fflush(stderr);
    }
    checkCudaErrors("Error: CUDA error when copy on GPU!", state);
    //cudaDeviceSynchronize();
}

bool FastllmCudaValidatePointerRange(const void *ptr, size_t bytes,
                                     int expectedDevice) {
    if (ptr == nullptr) {
        return bytes == 0;
    }

    int originalDevice = FastllmCudaGetDevice();
    cudaError_t state = cudaSetDevice(expectedDevice);
    CUdeviceptr base = 0;
    size_t allocation = 0;
    CUresult rangeState = CUDA_ERROR_INVALID_VALUE;
    if (state == cudaSuccess) {
        rangeState = cuMemGetAddressRange(&base, &allocation,
                                          (CUdeviceptr)ptr);
    }
    cudaSetDevice(originalDevice);
    if (state != cudaSuccess || rangeState != CUDA_SUCCESS || base == 0) {
        cudaGetLastError();
        return false;
    }
    size_t offset = (size_t)((CUdeviceptr)ptr - base);
    return offset <= allocation && bytes <= allocation - offset;
}

void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    if (size == 0 || dst == src) {
        return;
    }
    if (dst == nullptr || src == nullptr) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer),
                 "Error: CUDA copy Between GPUs got null pointer. dstId = %d, srcId = %d, dst = %p, src = %p, size = %lu.\n",
                 dstId, srcId, dst, src, size);
        fastllm::ErrorInFastLLM(std::string(buffer));
        return;
    }
    int oriId = FastllmCudaGetDevice();
    int canPeerAccess = 0;
    cudaError_t state = cudaDeviceCanAccessPeer(&canPeerAccess, dstId, srcId);
    if (state != cudaSuccess) {
        cudaGetLastError();
        canPeerAccess = 0;
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t switchState = cudaSetDevice(dstId);
    if (switchState == cudaSuccess) {
        cudaError_t captureState = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
        if (captureState != cudaSuccess) {
            cudaGetLastError();
            captureStatus = cudaStreamCaptureStatusNone;
        }
    }
    cudaSetDevice(oriId);

    if (captureStatus != cudaStreamCaptureStatusNone) {
        // Record a send node on the source graph and a matching receive node on
        // the destination graph.  Unlike a destination-side peer-read kernel,
        // this dependency is generation-safe across repeated graph replays.
        if (!FastllmNcclGraphPeerCopy(dstId, dst, srcId, src, size)) {
            FastllmCudaSetThreadError();
        }
        cudaSetDevice(oriId);
        return;
    }
    const char *failedStage = "cudaMemcpyPeer";
    if (canPeerAccess) {
        state = cudaMemcpyPeer(dst, dstId, src, srcId, size);
        if (state == cudaSuccess) {
            FastllmCudaSetDevice(dstId);
            DeviceSync();
            FastllmCudaSetDevice(oriId);
            return;
        }
        cudaGetLastError();
    }

    uint8_t *cpuData = new uint8_t[size];
    state = cudaSetDevice(srcId);
    failedStage = "cudaSetDevice(src)";
    if (state == cudaSuccess) {
        state = cudaMemcpy(cpuData, src, size, cudaMemcpyDeviceToHost);
        failedStage = "cudaMemcpyDeviceToHost";
    }
    if (state == cudaSuccess) {
        state = cudaSetDevice(dstId);
        failedStage = "cudaSetDevice(dst)";
    }
    if (state == cudaSuccess) {
        state = cudaMemcpy(dst, cpuData, size, cudaMemcpyHostToDevice);
        failedStage = "cudaMemcpyHostToDevice";
    }
    delete[] cpuData;
    if (state != cudaSuccess) {
        printf("Error: CUDA copy Between GPUs failed in %s. dstId = %d, srcId = %d, "
               "dst = %p, src = %p, size = %lu, canPeerAccess = %d.\n",
               failedStage, dstId, srcId, dst, src, size, canPeerAccess);
        fflush(stdout);
    }
    checkCudaErrors("Error: CUDA error when copy Between GPUs!", state);
    if (state == cudaSuccess) {
        FastllmCudaSetDevice(dstId);
        DeviceSync();
    }
    FastllmCudaSetDevice(oriId);
}

bool FastllmCudaMemcpyPeerAsyncCurrentThread(
        int dstId, void *dst, int srcId, const void *src, size_t size) {
    if (size == 0 || (dstId == srcId && dst == src)) {
        return true;
    }
    if (dstId == srcId) {
        return FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
            dst, src, size);
    }
    cudaError_t state = cudaMemcpyPeerAsync(dst, dstId, src, srcId, size,
                                            cudaStreamPerThread);
    if (state != cudaSuccess) {
        // Callers may deliberately fall back to a staged synchronous copy.
        // Consume the sticky runtime error before that fallback issues more
        // CUDA work.
        cudaGetLastError();
        return false;
    }
    return true;
}

void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    if (width == 0 || height == 0 || dst == src) {
        return;
    }

    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
    checkCudaErrors("Error: CUDA error when checking CUDA graph capture status!", state);
    if (captureStatus != cudaStreamCaptureStatusNone) {
        state = cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        checkCudaErrors("Error: CUDA error when async 2D copy on GPU!", state);
        return;
    }

    state = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    checkCudaErrors("Error: CUDA error when 2D copy on GPU!", state);
    DeviceSync();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmShiftAppendWindowKernel(uint8_t *cache, const uint8_t *newToken,
                                               int channels, int window, int unitSize) {
    int channel = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;
    if (channel >= channels) {
        return;
    }

    uint8_t *cacheRow = cache + (size_t) channel * window * unitSize;
    const uint8_t *newTokenRow = newToken + (size_t) channel * unitSize;
    int shiftBytes = (window - 1) * unitSize;
    for (int i = 0; i < shiftBytes; i++) {
        cacheRow[i] = cacheRow[i + unitSize];
    }
    for (int i = 0; i < unitSize; i++) {
        cacheRow[shiftBytes + i] = newTokenRow[i];
    }
}

void FastllmCudaShiftAppendWindow(uint8_t *cache, const uint8_t *newToken, int channels, int window, int unitSize) {
    if (channels <= 0 || window <= 0 || unitSize <= 0) {
        return;
    }
    const int kThreads = 256;
    FastllmShiftAppendWindowKernel<kThreads>
        <<< (channels + kThreads - 1) / kThreads, kThreads >>>(cache, newToken, channels, window, unitSize);
    DeviceSync();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpy2DKernel (uint8_t * 	dst, size_t 	dpitch, uint8_t * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    int id = blockIdx.x;
    dst += id * dpitch;
    src += id * spitch;
    for (int i = threadIdx.x; i < width; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpyBatchKernel (uint8_t** pointer) {
    int id = blockIdx.x;
    uint8_t *dst = pointer[id * 3];
    uint8_t *src = pointer[id * 3 + 1];
    size_t len = (size_t)(pointer[id * 3 + 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRepeatKernel (void *inputOri, void *outputOri, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    int id = blockIdx.x;
    int i = id / repeatTimes, j = id % repeatTimes;
    uint8_t *output = (uint8_t*)outputOri + i * outputStride0 + j * outputStride1;
    uint8_t *input = (uint8_t*)inputOri + i * inputStride;
    for (int x = threadIdx.x; x < copyLen; x += THREAD_PER_BLOCK) {
        output[x] = input[x];
    }
}

void FastllmCudaRepeat(void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    FastllmRepeatKernel <256> <<< outer * repeatTimes, 256 >>> (input, output, outer, repeatTimes, inputStride, outputStride0, outputStride1, copyLen);
    DeviceSync();
}

void FastllmCudaMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                            size_t *	spitchs, size_t *widths, size_t *	heights,
                                            int batch) {
    int total = 0;
    for (int i = 0; i < batch; i++) {
        total += heights[i];
    }
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;
    for (int i = 0; i < batch; i++) {
        for (int h = 0; h < heights[i]; h++) {
            cpuPointers[cur * 3 + 0] = (uint8_t*)dsts[i] + h * dpitchs[i];
            cpuPointers[cur * 3 + 1] = (uint8_t*)srcs[i] + h * spitchs[i];
            cpuPointers[cur * 3 + 2] = (uint8_t*)(widths[i]);

            cur++;
        }
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
    FastllmMemcpyBatchKernel <256> <<<total, 256>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
}

bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmExpKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmExpKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    } else {
        printf("Exp datatype error.\n");
        exit(0);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmReluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else {
        printf("Relu datatype error.\n");
        exit(0);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    FastllmGeluNewKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGeglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int threadPerBlock = std::min(1024, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmGegluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmGegluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmGegluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaInput, (__nv_bfloat16*)cudaOutput, len, spatial, mid);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSigmoidKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSigmoidKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmSigmoidKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaInput, (__nv_bfloat16*)cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSigmoidMulTo(fastllm::Data &input,
                             const fastllm::Data &gate) {
    if (input.dataType != gate.dataType || input.dims != gate.dims ||
        input.Count(0) <= 0) {
        return false;
    }
    int len = input.Count(0);
    void *inputData = FastllmCudaPrepareInput(input);
    void *gateData = FastllmCudaPrepareInput(gate);
    int threads = std::min(256, len);
    int blocks = (len + threads - 1) / threads;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSigmoidMulToKernel<<<blocks, threads>>>(
            (float*)inputData, (const float*)gateData, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSigmoidMulToKernel<<<blocks, threads>>>(
            (half*)inputData, (const half*)gateData, len);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmSigmoidMulToKernel<<<blocks, threads>>>(
            (__nv_bfloat16*)inputData,
            (const __nv_bfloat16*)gateData, len);
    } else {
        FastllmCudaFinishInput(gate, gateData);
        FastllmCudaFinishOutput(input, inputData);
        return false;
    }
    FastllmCudaFinishInput(gate, gateData);
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData, float outputScale) {
    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float *aLog = (float*) FastllmCudaPrepareInput(aLogData);
    float *dtBias = (float*) FastllmCudaPrepareInput(dtBiasData);

    int threadPerBlock = std::min(64, channels);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> (cudaInput, cudaOutput, aLog, dtBias, channels, outputScale);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> ((half*)cudaInput, (half*)cudaOutput, aLog, dtBias, channels, outputScale);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishInput(aLogData, aLog);
    FastllmCudaFinishInput(dtBiasData, dtBias);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSigmoidMambaSoftplus(fastllm::Data &sigmoidInputOutput, const fastllm::Data &softplusInput,
                                     fastllm::Data &softplusOutput, const fastllm::Data &aLogData, const fastllm::Data &dtBiasData) {
    if (sigmoidInputOutput.dataDevice != fastllm::DataDevice::CUDA ||
        softplusInput.dataDevice != fastllm::DataDevice::CUDA ||
        softplusOutput.dataDevice != fastllm::DataDevice::CUDA ||
        aLogData.dataDevice != fastllm::DataDevice::CUDA ||
        dtBiasData.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    int dimsLen = softplusInput.dims.size();
    int outer = softplusInput.Count(0) / softplusInput.Count(dimsLen - 1);
    int channels = softplusInput.dims[dimsLen - 1];
    int threadPerBlock = std::min(64, channels);
    if (sigmoidInputOutput.dataType == fastllm::DataType::FLOAT32) {
        FastllmSigmoidMambaSoftplusKernel<<<outer, threadPerBlock>>>(
            (float *) sigmoidInputOutput.cudaData, (const float *) softplusInput.cudaData,
            (float *) softplusOutput.cudaData, (const float *) aLogData.cudaData, (const float *) dtBiasData.cudaData, channels);
    } else if (sigmoidInputOutput.dataType == fastllm::DataType::FLOAT16) {
        FastllmSigmoidMambaSoftplusKernel<<<outer, threadPerBlock>>>(
            (half *) sigmoidInputOutput.cudaData, (const half *) softplusInput.cudaData,
            (half *) softplusOutput.cudaData, (const float *) aLogData.cudaData, (const float *) dtBiasData.cudaData, channels);
    } else {
        return false;
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaSigmoidMambaSoftplus.", cudaGetLastError());
    return true;
}

bool FastllmCudaSigmoidMambaSoftplusCombinedFloat16(
        const fastllm::Data &input,
        const fastllm::Data &aLogData,
        const fastllm::Data &dtBiasData,
        int batch, int seqLen, int inputChannels,
        int baOffset, int channels,
        fastllm::Data &sigmoidOutput,
        fastllm::Data &softplusOutput) {
    auto isDense = [](const fastllm::Data &data) {
        if (data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (batch <= 0 || seqLen <= 0 || inputChannels <= 0 ||
        baOffset < 0 || channels <= 0 ||
        baOffset + channels * 2 > inputChannels ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        input.cudaData == nullptr || !isDense(input) ||
        input.dims.back() != inputChannels ||
        input.Count(0) !=
            (uint64_t)batch * seqLen * inputChannels ||
        aLogData.dataDevice != fastllm::DataDevice::CUDA ||
        aLogData.dataType != fastllm::DataType::FLOAT32 ||
        aLogData.cudaData == nullptr || !isDense(aLogData) ||
        aLogData.Count(0) != (uint64_t)channels ||
        dtBiasData.dataDevice != fastllm::DataDevice::CUDA ||
        dtBiasData.dataType != fastllm::DataType::FLOAT32 ||
        dtBiasData.cudaData == nullptr || !isDense(dtBiasData) ||
        dtBiasData.Count(0) != (uint64_t)channels) {
        return false;
    }

    auto prepareOutput = [&](fastllm::Data &output) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = input.dataDevice;
        output.dataDeviceIds = input.dataDeviceIds;
        output.Resize({batch, seqLen, channels});
        output.Allocate(false);
        return output.cudaData != nullptr;
    };
    if (!prepareOutput(sigmoidOutput) ||
        !prepareOutput(softplusOutput)) {
        return false;
    }

    int outer = batch * seqLen;
    int threadPerBlock = std::min(64, channels);
    FastllmSigmoidMambaSoftplusCombinedHalfKernel
        <<<outer, threadPerBlock>>>(
            (const half *)input.cudaData,
            (half *)sigmoidOutput.cudaData,
            (half *)softplusOutput.cudaData,
            (const float *)aLogData.cudaData,
            (const float *)dtBiasData.cudaData,
            inputChannels, baOffset, channels);
    checkCudaErrors(
        "Error: CUDA error in "
        "FastllmCudaSigmoidMambaSoftplusCombinedFloat16.",
        cudaGetLastError());
    return true;
}

bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;

    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaInput, (__nv_bfloat16*)cudaOutput, len, spatial, mid);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;

    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmCrossSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmCrossSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmCrossSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaInput, (__nv_bfloat16*)cudaOutput, len, spatial, mid);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAdd(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmAddKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    } else {
        FastllmAddKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, __float2half_rn(v), len);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaCopy(const fastllm::Data &input, fastllm::Data &output) {
    uint64_t len = input.GetBytes();
    if (len == 0) {
        return true;
    }
    uint8_t *cudaInput = (uint8_t *) FastllmCudaPrepareInput(input);
    uint8_t *cudaOutput = (uint8_t *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = 256;

    FastllmCopyKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    } else {
        FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, __float2half_rn(v), len);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(1024, len);
    if (input0.dataType == fastllm::DataType::FLOAT32) {
        FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    } else if (input0.dataType == fastllm::DataType::FLOAT16) {
        FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, __float2half_rn(alpha), len);
    } else if (input0.dataType == fastllm::DataType::BFLOAT16) {
        FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaData, (__nv_bfloat16*)input1Data, __float2bfloat16_rn(alpha), len);
    }

    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

void FastllmCudaAddHostToDevice(void *dst, void *hostSrc, int len, fastllm::DataType dataType) {
    size_t bytes;
    if (dataType == fastllm::DataType::FLOAT32) {
        bytes = (size_t)len * sizeof(float);
    } else if (dataType == fastllm::DataType::FLOAT16 || dataType == fastllm::DataType::BFLOAT16) {
        bytes = (size_t)len * sizeof(uint16_t);
    } else {
        printf("FastllmCudaAddHostToDevice: unsupported dataType.\n");
        return;
    }

    void *tmpGpu = FastllmCudaMalloc(bytes);
    FastllmCudaCopyFromHostToDevice(tmpGpu, hostSrc, bytes);

    int threadPerBlock = std::min(1024, len);
    int blocks = (len - 1) / threadPerBlock + 1;
    if (dataType == fastllm::DataType::FLOAT32) {
        FastllmAddToKernel<<<blocks, threadPerBlock>>>((float*)dst, (float*)tmpGpu, 1.0f, len);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        FastllmAddToKernel<<<blocks, threadPerBlock>>>((half*)dst, (half*)tmpGpu, __float2half_rn(1.0f), len);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FastllmAddToKernel<<<blocks, threadPerBlock>>>((__nv_bfloat16*)dst, (__nv_bfloat16*)tmpGpu, __float2bfloat16_rn(1.0f), len);
    }

    FastllmCudaFree(tmpGpu);
    DeviceSync();
}

bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    if (input1.Count(0) == 1) {
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
        } else if (input0.dataType == fastllm::DataType::FLOAT16) {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        } else if (input0.dataType == fastllm::DataType::BFLOAT16) {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaData, (__nv_bfloat16*)input1Data, alpha, len);
        }
    } else if (input0.dims == input1.dims) {
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
        } else if (input0.dataType == fastllm::DataType::FLOAT16) {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        } else if (input0.dataType == fastllm::DataType::BFLOAT16) {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaData, (__nv_bfloat16*)input1Data, alpha, len);
        }
    } else {
        int channelLen = input0.Count(0) / input1.Count(0);
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len, channelLen);
        } else if (input0.dataType == fastllm::DataType::FLOAT16) {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len, channelLen);
        } else if (input0.dataType == fastllm::DataType::BFLOAT16) {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((__nv_bfloat16*)cudaData, (__nv_bfloat16*)input1Data, alpha, len, channelLen);
        }
    }
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

__device__ inline float FastllmMulToCausalMaskValue(
        float a, float b, float alpha) {
    return a * (b * alpha);
}

__device__ inline half FastllmMulToCausalMaskValue(
        half a, half b, float alpha) {
#ifdef CUDA_NO_TENSOR_CORE
    return __float2half(__half2float(b) * alpha * __half2float(a));
#else
    return __hmul(a, (half)((float)b * alpha));
#endif
}

__device__ inline __nv_bfloat16 FastllmMulToCausalMaskValue(
        __nv_bfloat16 a, __nv_bfloat16 b, float alpha) {
    return __float2bfloat16_rn(
        __bfloat162float(a) * __bfloat162float(b) * alpha);
}

template <typename T>
__global__ void FastllmMulToCausalMaskKernel(
        T *data, const T *scale, float alpha, int len,
        int n, int m, int base, T maskValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    int remainder = idx % (n * m);
    int i = remainder / m;
    int j = remainder % m;
    if (j >= i + base) {
        data[idx] = maskValue;
    } else {
        data[idx] =
            FastllmMulToCausalMaskValue(data[idx], scale[idx], alpha);
    }
}

template <typename T>
__global__ void FastllmMulCausalMaskKernel(
        const T *input, const T *scale, T *output,
        float alpha, int len, int n, int m, int base, T maskValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    int remainder = idx % (n * m);
    int i = remainder / m;
    int j = remainder % m;
    if (j >= i + base) {
        output[idx] = maskValue;
    } else {
        output[idx] =
            FastllmMulToCausalMaskValue(input[idx], scale[idx], alpha);
    }
}

bool FastllmCudaMulToCausalMask(
        fastllm::Data &input0,
        const fastllm::Data &input1,
        float alpha,
        int base,
        float maskValue) {
    if (input0.dims.size() < 2 || input0.dims != input1.dims ||
        input0.dataType != input1.dataType) {
        return false;
    }
    if (input0.dataType != fastllm::DataType::FLOAT32 &&
        input0.dataType != fastllm::DataType::FLOAT16 &&
        input0.dataType != fastllm::DataType::BFLOAT16) {
        return false;
    }

    int len = input0.Count(0);
    if (len <= 0) {
        return true;
    }
    void *input0Data = FastllmCudaPrepareInput(input0);
    void *input1Data = FastllmCudaPrepareInput(input1);
    int dimsLen = input0.dims.size();
    int n = input0.dims[dimsLen - 2];
    int m = input0.dims[dimsLen - 1];
    int blockSize = std::min(256, len);
    int gridSize = (len + blockSize - 1) / blockSize;

    if (input0.dataType == fastllm::DataType::FLOAT32) {
        FastllmMulToCausalMaskKernel<float><<<gridSize, blockSize>>>(
            (float*)input0Data, (const float*)input1Data, alpha,
            len, n, m, base, maskValue);
    } else if (input0.dataType == fastllm::DataType::FLOAT16) {
        FastllmMulToCausalMaskKernel<half><<<gridSize, blockSize>>>(
            (half*)input0Data, (const half*)input1Data, alpha,
            len, n, m, base, __float2half(maskValue));
    } else {
        FastllmMulToCausalMaskKernel<__nv_bfloat16>
            <<<gridSize, blockSize>>>(
                (__nv_bfloat16*)input0Data,
                (const __nv_bfloat16*)input1Data, alpha,
                len, n, m, base, __float2bfloat16_rn(maskValue));
    }

    DeviceSync();
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, input0Data);
    return true;
}

bool FastllmCudaMulCausalMask(
        const fastllm::Data &input0,
        const fastllm::Data &input1,
        fastllm::Data &output,
        float alpha,
        int base,
        float maskValue) {
    if (input0.dims.size() < 2 || input0.dims != input1.dims ||
        output.dims != input0.dims ||
        input0.dataType != input1.dataType ||
        output.dataType != input0.dataType ||
        input0.dataDevice != fastllm::DataDevice::CUDA ||
        input1.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input0.cudaData == nullptr || input1.cudaData == nullptr ||
        output.cudaData == nullptr) {
        return false;
    }
    if (input0.dataType != fastllm::DataType::FLOAT32 &&
        input0.dataType != fastllm::DataType::FLOAT16 &&
        input0.dataType != fastllm::DataType::BFLOAT16) {
        return false;
    }

    int len = input0.Count(0);
    if (len <= 0) {
        return true;
    }
    void *input0Data = FastllmCudaPrepareInput(input0);
    void *input1Data = FastllmCudaPrepareInput(input1);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (input0Data == nullptr || input1Data == nullptr ||
        outputData == nullptr) {
        FastllmCudaFinishInput(input0, input0Data);
        FastllmCudaFinishInput(input1, input1Data);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }
    int dimsLen = input0.dims.size();
    int n = input0.dims[dimsLen - 2];
    int m = input0.dims[dimsLen - 1];
    int blockSize = std::min(256, len);
    int gridSize = (len + blockSize - 1) / blockSize;

    if (input0.dataType == fastllm::DataType::FLOAT32) {
        FastllmMulCausalMaskKernel<float><<<gridSize, blockSize>>>(
            (const float*)input0Data, (const float*)input1Data,
            (float*)outputData, alpha, len, n, m, base, maskValue);
    } else if (input0.dataType == fastllm::DataType::FLOAT16) {
        FastllmMulCausalMaskKernel<half><<<gridSize, blockSize>>>(
            (const half*)input0Data, (const half*)input1Data,
            (half*)outputData, alpha, len, n, m, base,
            __float2half(maskValue));
    } else {
        FastllmMulCausalMaskKernel<__nv_bfloat16>
            <<<gridSize, blockSize>>>(
                (const __nv_bfloat16*)input0Data,
                (const __nv_bfloat16*)input1Data,
                (__nv_bfloat16*)outputData, alpha, len, n, m, base,
                __float2bfloat16_rn(maskValue));
    }

    DeviceSync();
    FastllmCudaFinishInput(input0, input0Data);
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(output, outputData);
    return true;
}

bool FastllmCudaQwen35FusedMoeJoin(
        fastllm::Data &destination,
        const fastllm::Data &routedOutput,
        const fastllm::Data &sharedOutput,
        const fastllm::Data *sharedGate,
        bool addResidual,
        bool sharedGateAlreadySigmoid) {
    const int len = destination.Count(0);
    if (len <= 0 ||
        destination.dataDevice != fastllm::DataDevice::CUDA ||
        routedOutput.dataDevice != fastllm::DataDevice::CUDA ||
        sharedOutput.dataDevice != fastllm::DataDevice::CUDA ||
        destination.cudaData == nullptr ||
        routedOutput.cudaData == nullptr ||
        sharedOutput.cudaData == nullptr ||
        destination.dataType != fastllm::DataType::FLOAT16 ||
        routedOutput.dataType != destination.dataType ||
        sharedOutput.dataType != destination.dataType ||
        routedOutput.Count(0) != len ||
        sharedOutput.Count(0) != len ||
        sharedOutput.dims.empty()) {
        return false;
    }

    const int hidden = sharedOutput.dims.back();
    if (hidden <= 0 || len % hidden != 0) {
        return false;
    }
    const int rows = len / hidden;
    bool broadcastGate = false;
    const half *gateData = nullptr;
    if (sharedGate != nullptr) {
        if (sharedGate->dataDevice != fastllm::DataDevice::CUDA ||
            sharedGate->cudaData == nullptr ||
            sharedGate->dataType != destination.dataType ||
            (sharedGate->Count(0) != rows && sharedGate->Count(0) != 1)) {
            return false;
        }
        broadcastGate = sharedGate->Count(0) == 1;
        gateData = (const half*)sharedGate->cudaData;
    }

    const int threads = std::min(1024, hidden);
    half *destinationData = (half*)destination.cudaData;
    const half *routedData = (const half*)routedOutput.cudaData;
    const half *sharedData = (const half*)sharedOutput.cudaData;
    if (gateData != nullptr) {
        if (addResidual) {
            FastllmCudaQwen35FusedMoeJoinHalfKernel<true, true>
                <<<rows, threads>>>(destinationData, routedData, sharedData,
                                    gateData, rows, hidden, broadcastGate,
                                    sharedGateAlreadySigmoid);
        } else {
            FastllmCudaQwen35FusedMoeJoinHalfKernel<true, false>
                <<<rows, threads>>>(destinationData, routedData, sharedData,
                                    gateData, rows, hidden, broadcastGate,
                                    sharedGateAlreadySigmoid);
        }
    } else if (addResidual) {
        FastllmCudaQwen35FusedMoeJoinHalfKernel<false, true>
            <<<rows, threads>>>(destinationData, routedData, sharedData,
                                nullptr, rows, hidden, false, false);
    } else {
        FastllmCudaQwen35FusedMoeJoinHalfKernel<false, false>
            <<<rows, threads>>>(destinationData, routedData, sharedData,
                                nullptr, rows, hidden, false, false);
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaQwen35FusedMoeJoin.",
                    cudaGetLastError());
    return true;
}

bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int n = input.dims[0], m = input.dims[1];
    int spn = input.dims[2], spm = input.dims[3];
    int spatial = input.Count(2);
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    FastllmAlibiMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                   n, m, spn, spm, spatial);
    FastllmCudaFinishInput(mask, maskData);
    FastllmCudaFinishOutput(input, cudaData);
    return true;
}

__device__ __forceinline__ float FastllmCudaValueToFloat(float value) {
    return value;
}

__device__ __forceinline__ float FastllmCudaValueToFloat(half value) {
    return __half2float(value);
}

__device__ __forceinline__ float FastllmCudaValueToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<typename T>
__device__ __forceinline__ T FastllmCudaFloatToValue(float value);

template<>
__device__ __forceinline__ float FastllmCudaFloatToValue<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ half FastllmCudaFloatToValue<half>(float value) {
    return __float2half(value);
}

template<>
__device__ __forceinline__ __nv_bfloat16 FastllmCudaFloatToValue<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 FastllmCudaFloatToValue<__nv_fp8_e4m3>(float value) {
    return __nv_fp8_e4m3(value);
}

template <typename T>
__global__ void FastllmClampKernel(T *data, int len, bool hasMin, float minValue, bool hasMax, float maxValue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) {
        return;
    }
    float x = FastllmCudaValueToFloat(data[idx]);
    if (hasMin && x < minValue) {
        x = minValue;
    }
    if (hasMax && x > maxValue) {
        x = maxValue;
    }
    data[idx] = FastllmCudaFloatToValue<T>(x);
}

bool FastllmCudaClamp(fastllm::Data &input, bool hasMin, float minValue, bool hasMax, float maxValue) {
    if (!hasMin && !hasMax) {
        return true;
    }
    if (input.dataDevice != fastllm::DataDevice::CUDA || input.cudaData == nullptr) {
        return false;
    }
    int len = input.Count(0);
    if (len <= 0) {
        return true;
    }
    int threadPerBlock = std::min(1024, len);
    int blocks = (len - 1) / threadPerBlock + 1;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmClampKernel <<< blocks, threadPerBlock >>> (
            (float*)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmClampKernel <<< blocks, threadPerBlock >>> (
            (half*)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmClampKernel <<< blocks, threadPerBlock >>> (
            (__nv_bfloat16*)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

template<typename T>
__global__ void TransferAttnKernelFused(T *data, int n, int m, int outer) {
    extern __shared__ float shared[];

    int o = blockIdx.x;
    if (o >= outer) return;

    T *batchData = data + (size_t)o * n * m;
    float *matrix = shared;
    float *row = shared + n * m;

    for (int idx = threadIdx.x; idx < n * m; idx += blockDim.x) {
        matrix[idx] = FastllmCudaValueToFloat(batchData[idx]);
    }
    __syncthreads();

    for (int i = 1; i < n; i++) {
        for (int j = threadIdx.x; j < i; j += blockDim.x) {
            row[j] = matrix[i * m + j];
        }
        __syncthreads();

        for (int j = threadIdx.x; j < i; j += blockDim.x) {
            float sum = row[j];
            for (int k = 0; k < i; k++) {
                sum += row[k] * matrix[k * m + j];
            }
            matrix[i * m + j] = sum;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < n * m; idx += blockDim.x) {
        int i = idx / m;
        int j = idx % m;
        float value = matrix[idx] + (i == j ? 1.0f : 0.0f);
        batchData[idx] = FastllmCudaFloatToValue<T>(value);
    }
}

template<typename T>
__global__ void TransferAttnKernelRow(T *data, int n, int m, int outer, int row_idx) {
    extern __shared__ float shared[];

    int o = blockIdx.z;
    if (o >= outer) return;

    int tid = threadIdx.x;
    int j = tid + blockIdx.x * blockDim.x;

    T *batchData = data + (size_t)o * n * m;
    float *row_i = shared;

    for (int idx = tid; idx < row_idx; idx += blockDim.x) {
        row_i[idx] = FastllmCudaValueToFloat(batchData[row_idx * m + idx]);
    }
    __syncthreads();

    if (j < row_idx) {
        float sum = row_i[j];
        for (int k = 0; k < row_idx; k++) {
            sum += row_i[k] * FastllmCudaValueToFloat(batchData[k * m + j]);
        }
        batchData[row_idx * m + j] = FastllmCudaFloatToValue<T>(sum);
    }
}

// CUDA kernel for adding identity matrix
template<typename T>
__global__ void AddIdentityKernel(T *data, int n, int m, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * n;
    
    if (idx < total) {
        int o = idx / n;
        int i = idx % n;
        int offset = o * n * m + i * m + i;
        float cur = FastllmCudaValueToFloat(data[offset]);
        data[offset] = FastllmCudaFloatToValue<T>(cur + 1.0f);
    }
}

bool FastllmCudaTransferAttn(fastllm::Data &input) {
    void *inputData = FastllmCudaPrepareInput(input);

    int dimsLen = input.dims.size();
    int n = input.dims[dimsLen - 2];
    int m = input.dims[dimsLen - 1]; 
    int outer = input.Count(0) / input.Count(dimsLen - 2);

    bool useFusedTransferAttn = (n == m && n <= 64 && m <= 64) &&
                                fastllm::GetFastllmEnv().useFusedTransferAttn;

    if (useFusedTransferAttn) {
        int threadsPerBlock = 64;
        int sharedMemSize = (n * m + n) * sizeof(float);
        if (input.dataType == fastllm::DataType::FLOAT32) {
            TransferAttnKernelFused<<<outer, threadsPerBlock, sharedMemSize>>>(
                (float*)inputData, n, m, outer);
        } else if (input.dataType == fastllm::DataType::FLOAT16) {
            TransferAttnKernelFused<<<outer, threadsPerBlock, sharedMemSize>>>(
                (half*)inputData, n, m, outer);
        } else if (input.dataType == fastllm::DataType::BFLOAT16) {
            TransferAttnKernelFused<<<outer, threadsPerBlock, sharedMemSize>>>(
                (__nv_bfloat16*)inputData, n, m, outer);
        }
    } else {
        for (int i = 1; i < n; i++) {
            int elementsToProcess = i;
            int threadsPerBlock = min(256, elementsToProcess);
            int blocksPerGrid = (elementsToProcess + threadsPerBlock - 1) / threadsPerBlock;

            dim3 blocks(blocksPerGrid, 1, outer);
            dim3 threads(threadsPerBlock, 1, 1);
            int sharedMemSize = elementsToProcess * sizeof(float);

            if (input.dataType == fastllm::DataType::FLOAT32) {
                TransferAttnKernelRow<<<blocks, threads, sharedMemSize>>>(
                    (float*)inputData, n, m, outer, i);
            } else if (input.dataType == fastllm::DataType::FLOAT16) {
                TransferAttnKernelRow<<<blocks, threads, sharedMemSize>>>(
                    (half*)inputData, n, m, outer, i);
            } else if (input.dataType == fastllm::DataType::BFLOAT16) {
                TransferAttnKernelRow<<<blocks, threads, sharedMemSize>>>(
                    (__nv_bfloat16*)inputData, n, m, outer, i);
            }
            cudaDeviceSynchronize();
        }

        int totalDiag = outer * n;
        int threadsPerBlock = 256;
        int blocksPerGrid = (totalDiag + threadsPerBlock - 1) / threadsPerBlock;
        if (input.dataType == fastllm::DataType::FLOAT32) {
            AddIdentityKernel<<<blocksPerGrid, threadsPerBlock>>>((float*)inputData, n, m, outer);
        } else if (input.dataType == fastllm::DataType::FLOAT16) {
            AddIdentityKernel<<<blocksPerGrid, threadsPerBlock>>>((half*)inputData, n, m, outer);
        } else if (input.dataType == fastllm::DataType::BFLOAT16) {
            AddIdentityKernel<<<blocksPerGrid, threadsPerBlock>>>((__nv_bfloat16*)inputData, n, m, outer);
        }
    }

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

// CUDA核函数模板，支持float / half / bfloat16
template<typename T>
__global__ void CumSumLastDimKernel(T* data, int dim, int outer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < outer) {
        T* row = data + tid * dim;
        
        // 对每一行进行累积和
        for (int j = 1; j < dim; j++) {
            float sum = FastllmCudaValueToFloat(row[j]) + FastllmCudaValueToFloat(row[j - 1]);
            row[j] = FastllmCudaFloatToValue<T>(sum);
        }
    }
}

bool FastllmCudaCumSumLastDim(fastllm::Data &input) {
    void *inputData = FastllmCudaPrepareInput(input);
    
    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    
    // 配置CUDA执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (outer + threadsPerBlock - 1) / threadsPerBlock;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        CumSumLastDimKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (float*)inputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        CumSumLastDimKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (half*)inputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        CumSumLastDimKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (__nv_bfloat16*)inputData, dim, outer);
    }

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    
    return true; // 添加返回值
}

template<typename T>
__global__ void ApplyChunkDecayByLastLogGKernel(T *input, const T *g, int dim, int channels, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * dim * channels;

    if (idx < total) {
        int tokenIdx = (idx / channels) % dim;
        int outerIdx = idx / (dim * channels);
        float last = FastllmCudaValueToFloat(g[outerIdx * dim + dim - 1]);
        float cur = FastllmCudaValueToFloat(g[outerIdx * dim + tokenIdx]);
        float scale = expf(last - cur);
        float value = FastllmCudaValueToFloat(input[idx]) * scale;
        input[idx] = FastllmCudaFloatToValue<T>(value);
    }
}

bool FastllmCudaApplyChunkDecayByLastLogG(fastllm::Data &input, const fastllm::Data &g) {
    void *inputData = FastllmCudaPrepareInput(input);
    void *gData = FastllmCudaPrepareInput(g);

    int dim = input.dims[input.dims.size() - 2];
    int channels = input.dims.back();
    int outer = g.Count(0) / dim;
    int total = outer * dim * channels;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    if (input.dataType == fastllm::DataType::FLOAT32) {
        ApplyChunkDecayByLastLogGKernel<float><<<gridSize, blockSize>>>(
            (float*)inputData, (float*)gData, dim, channels, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        ApplyChunkDecayByLastLogGKernel<half><<<gridSize, blockSize>>>(
            (half*)inputData, (half*)gData, dim, channels, outer);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        ApplyChunkDecayByLastLogGKernel<__nv_bfloat16><<<gridSize, blockSize>>>(
            (__nv_bfloat16*)inputData, (__nv_bfloat16*)gData, dim, channels, outer);
    }

    DeviceSync();
    FastllmCudaFinishInput(g, gData);
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

// CUDA核函数模板，支持float和half
template<typename T>
__global__ void CausalMaskKernel(T *data, int n, int m, int outer, int base, T maskValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * n * m;
    
    if (idx < total) {
        int o = idx / (n * m);
        int remainder = idx % (n * m);
        int i = remainder / m;
        int j = remainder % m;
        
        if (j >= i + base) {
            data[idx] = maskValue;
        }
    }
}

bool FastllmCudaCausalMask(fastllm::Data &input, int base, float maskValue) {
    void *inputData = FastllmCudaPrepareInput(input);
    int dimsLen = input.dims.size();
    int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);
    
    int total = outer * n * m;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        float *floatData = (float *)inputData;
        CausalMaskKernel<float><<<gridSize, blockSize>>>(floatData, n, m, outer, base, maskValue);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        __half *halfData = (__half *)inputData;
        __half halfMaskValue = __float2half(maskValue);
        CausalMaskKernel<__half><<<gridSize, blockSize>>>(halfData, n, m, outer, base, halfMaskValue);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        __nv_bfloat16 *bf16Data = (__nv_bfloat16 *)inputData;
        __nv_bfloat16 bf16MaskValue = __float2bfloat16_rn(maskValue);
        CausalMaskKernel<__nv_bfloat16><<<gridSize, blockSize>>>(bf16Data, n, m, outer, base, bf16MaskValue);
    }
    
    // 等待核函数执行完成
    DeviceSync();
    
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

// CUDA核函数定义
template<typename T>
__global__ void MakeDecayMaskKernel(const T* input, T* output, int dim, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer * dim * dim;
    
    if (idx < total_elements) {
        int o = idx / (dim * dim);
        int remainder = idx % (dim * dim);
        int i = remainder / dim;
        int j = remainder % dim;
        
        if (j <= i) {
            float val_i = FastllmCudaValueToFloat(input[o * dim + i]);
            float val_j = FastllmCudaValueToFloat(input[o * dim + j]);
            output[idx] = FastllmCudaFloatToValue<T>(expf(val_i - val_j));
        } else {
            output[idx] = FastllmCudaFloatToValue<T>(0.0f);
        }
    }
}

bool FastllmCudaMakeDecayMask(fastllm::Data &input, fastllm::Data &output) {
    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareInput(output);

    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    int total_elements = outer * dim * dim;
    
    // 配置CUDA执行参数
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        MakeDecayMaskKernel<float><<<gridSize, blockSize>>>(
            (float*)inputData, (float*)outputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        MakeDecayMaskKernel<half><<<gridSize, blockSize>>>(
            (half*)inputData, (half*)outputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        MakeDecayMaskKernel<__nv_bfloat16><<<gridSize, blockSize>>>(
            (__nv_bfloat16*)inputData, (__nv_bfloat16*)outputData, dim, outer);
    }

    // 等待核函数执行完成
    DeviceSync();
    
    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    
    return true;
}

template<typename T>
__global__ void CumSumLastDimMakeDecayMaskKernel(
        T *input, T *output, int dim, int outer) {
    int o = blockIdx.x;
    if (o >= outer) {
        return;
    }
    T *row = input + (size_t)o * dim;
    if (threadIdx.x == 0) {
        for (int j = 1; j < dim; j++) {
            float sum = FastllmCudaValueToFloat(row[j]) +
                        FastllmCudaValueToFloat(row[j - 1]);
            row[j] = FastllmCudaFloatToValue<T>(sum);
        }
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < dim * dim; idx += blockDim.x) {
        int i = idx / dim;
        int j = idx % dim;
        T *out = output + (size_t)o * dim * dim;
        if (j <= i) {
            float valI = FastllmCudaValueToFloat(row[i]);
            float valJ = FastllmCudaValueToFloat(row[j]);
            out[idx] = FastllmCudaFloatToValue<T>(expf(valI - valJ));
        } else {
            out[idx] = FastllmCudaFloatToValue<T>(0.0f);
        }
    }
}

template<typename T>
__global__ void CumSumDecayMaskNegMulCausalKernel(
        T *input, const T *matrix, T *decayMask, T *output,
        int dim, int outer) {
    int o = blockIdx.x;
    if (o >= outer) {
        return;
    }
    T *row = input + (size_t)o * dim;
    if (threadIdx.x == 0) {
        for (int j = 1; j < dim; j++) {
            float sum = FastllmCudaValueToFloat(row[j]) +
                        FastllmCudaValueToFloat(row[j - 1]);
            row[j] = FastllmCudaFloatToValue<T>(sum);
        }
    }
    __syncthreads();

    const T *matrixRow = matrix + (size_t)o * dim * dim;
    T *decayRow = decayMask + (size_t)o * dim * dim;
    T *outputRow = output + (size_t)o * dim * dim;
    for (int idx = threadIdx.x; idx < dim * dim; idx += blockDim.x) {
        int i = idx / dim;
        int j = idx % dim;
        if (j <= i) {
            float valI = FastllmCudaValueToFloat(row[i]);
            float valJ = FastllmCudaValueToFloat(row[j]);
            T decay = FastllmCudaFloatToValue<T>(expf(valI - valJ));
            decayRow[idx] = decay;
            if (j < i) {
                outputRow[idx] = FastllmMulToCausalMaskValue(
                    matrixRow[idx], decay, -1.0f);
            } else {
                outputRow[idx] = FastllmCudaFloatToValue<T>(0.0f);
            }
        } else {
            decayRow[idx] = FastllmCudaFloatToValue<T>(0.0f);
            outputRow[idx] = FastllmCudaFloatToValue<T>(0.0f);
        }
    }
}

bool FastllmCudaCumSumLastDimMakeDecayMask(
        fastllm::Data &input, fastllm::Data &output) {
    if (input.dataType != output.dataType ||
        (input.dataType != fastllm::DataType::FLOAT32 &&
         input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        input.dims.empty() || input.dims.back() <= 0 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }
    std::vector<int> expectedOutputDims = input.dims;
    expectedOutputDims.push_back(input.dims.back());
    if (output.dims != expectedOutputDims) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareInput(output);
    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    if (outer <= 0) {
        FastllmCudaFinishOutput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return true;
    }

    int blockSize = 256;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        CumSumLastDimMakeDecayMaskKernel<float>
            <<<outer, blockSize>>>(
                (float*)inputData, (float*)outputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        CumSumLastDimMakeDecayMaskKernel<half>
            <<<outer, blockSize>>>(
                (half*)inputData, (half*)outputData, dim, outer);
    } else {
        CumSumLastDimMakeDecayMaskKernel<__nv_bfloat16>
            <<<outer, blockSize>>>(
                (__nv_bfloat16*)inputData,
                (__nv_bfloat16*)outputData, dim, outer);
    }

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return true;
}

bool FastllmCudaCumSumDecayMaskNegMulCausal(
        fastllm::Data &input, const fastllm::Data &matrix,
        fastllm::Data &decayMask, fastllm::Data &output) {
    if (input.dataType != matrix.dataType ||
        input.dataType != decayMask.dataType ||
        input.dataType != output.dataType ||
        (input.dataType != fastllm::DataType::FLOAT32 &&
         input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        input.dims.empty() || input.dims.back() <= 0 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        matrix.dataDevice != fastllm::DataDevice::CUDA ||
        decayMask.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || matrix.cudaData == nullptr ||
        decayMask.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }
    std::vector<int> expectedMatrixDims = input.dims;
    expectedMatrixDims.push_back(input.dims.back());
    if (matrix.dims != expectedMatrixDims ||
        decayMask.dims != expectedMatrixDims ||
        output.dims != expectedMatrixDims) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *matrixData = FastllmCudaPrepareInput(matrix);
    void *decayData = FastllmCudaPrepareOutput(decayMask);
    void *outputData = FastllmCudaPrepareOutput(output);
    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    if (outer <= 0) {
        FastllmCudaFinishOutput(input, inputData);
        FastllmCudaFinishInput(matrix, matrixData);
        FastllmCudaFinishOutput(decayMask, decayData);
        FastllmCudaFinishOutput(output, outputData);
        return true;
    }

    int blockSize = 256;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        CumSumDecayMaskNegMulCausalKernel<float>
            <<<outer, blockSize>>>(
                (float*)inputData, (const float*)matrixData,
                (float*)decayData, (float*)outputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        CumSumDecayMaskNegMulCausalKernel<half>
            <<<outer, blockSize>>>(
                (half*)inputData, (const half*)matrixData,
                (half*)decayData, (half*)outputData, dim, outer);
    } else {
        CumSumDecayMaskNegMulCausalKernel<__nv_bfloat16>
            <<<outer, blockSize>>>(
                (__nv_bfloat16*)inputData,
                (const __nv_bfloat16*)matrixData,
                (__nv_bfloat16*)decayData,
                (__nv_bfloat16*)outputData, dim, outer);
    }

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    FastllmCudaFinishInput(matrix, matrixData);
    FastllmCudaFinishOutput(decayMask, decayData);
    FastllmCudaFinishOutput(output, outputData);
    return true;
}

static bool LaunchFastllmRMSNormFloat16(
        const half *input, const float *weight, half *output,
        int outer, int channels, float eps, int threadCount) {
    if (channels == 128 && threadCount == 32) {
        FastllmRMSNormHalf128ExactKernel<<<outer, 32>>>(
            input, weight, output, eps);
        return true;
    }
    if (threadCount != 0) {
        return false;
    }
    if (channels < 512) {
        FastllmRMSNormKernelInner1<64><<<outer, 64>>>(
            (half*)input, (float*)weight, output, outer, channels, eps);
        return true;
    } else if (channels < 4096) {
        FastllmRMSNormKernelInner1<512><<<outer, 512>>>(
            (half*)input, (float*)weight, output, outer, channels, eps);
        return true;
    }

    // A 4096-wide decode row maps exactly one aligned 16-byte vector to each
    // of 512 threads. Other shapes retain the generic launch.
    if (channels == 4096 &&
        ((uintptr_t)input % alignof(uint4)) == 0 &&
        ((uintptr_t)output % alignof(uint4)) == 0 &&
        ((uintptr_t)weight % alignof(float4)) == 0) {
        FastllmRMSNormHalf4096Kernel<<<outer, 512>>>(
            input, weight, output, eps);
        return true;
    }
    const char *virtualThreadsEnv = std::getenv(
        "FASTLLM_CUDA_RMSNORM_VIRTUAL_1024");
    bool useVirtualThreads =
        virtualThreadsEnv == nullptr || virtualThreadsEnv[0] == '\0' ||
        FastllmCudaEnvFlagEnabled(
            "FASTLLM_CUDA_RMSNORM_VIRTUAL_1024");
    if (useVirtualThreads && channels == 5120 && outer >= 64) {
        FastllmRMSNormHalfVirtual1024Kernel<<<outer, 512>>>(
            (half*)input, (float*)weight, output, channels, eps);
        return true;
    }
    FastllmRMSNormKernelInner1<1024><<<outer, 1024>>>(
        (half*)input, (float*)weight, output, outer, channels, eps);
    return true;
}

static bool LaunchFastllmRMSNormBFloat16(
        __nv_bfloat16 *input, float *weight, __nv_bfloat16 *output,
        int outer, int channels, float eps, int threadCount) {
#ifndef USE_ROCM
    if (channels == 3072 && threadCount == 256) {
        FastllmRMSNormBFloat16Hidden3072ExactKernel<<<outer, 256>>>(
            input, weight, output, eps);
        return true;
    }
#endif
    if (threadCount != 0) {
        return false;
    }
    if (channels < 512) {
        FastllmRMSNormKernelInner1<64><<<outer, 64>>>(
            input, weight, output, outer, channels, eps);
    } else if (channels < 4096) {
        FastllmRMSNormKernelInner1<512><<<outer, 512>>>(
            input, weight, output, outer, channels, eps);
    } else {
        FastllmRMSNormKernelInner1<1024><<<outer, 1024>>>(
            input, weight, output, outer, channels, eps);
    }
    return true;
}

bool FastllmCudaRMSNormFloat16WithThreadCount(
        const fastllm::Data &input, fastllm::Data &weight,
        fastllm::Data &output, float eps, int threadCount) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        input.dims.empty() || input.dims != output.dims ||
        input.strides.empty() || output.strides.empty() ||
        input.strides.back() != 1 || output.strides.back() != 1 ||
        weight.dims.size() != 1 || weight.dims[0] != input.dims.back() ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        weight.cudaData == nullptr) {
        return false;
    }
    int axis = (int)input.dims.size() - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    return LaunchFastllmRMSNormFloat16(
        (const half*)input.cudaData, (const float*)weight.cudaData,
        (half*)output.cudaData, outer, channels, eps, threadCount);
}

bool FastllmCudaRMSNormBFloat16WithThreadCount(
        const fastllm::Data &input, fastllm::Data &weight,
        fastllm::Data &output, float eps, int threadCount) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        input.dims.empty() || input.dims != output.dims ||
        input.strides.empty() || output.strides.empty() ||
        input.strides.back() != 1 || output.strides.back() != 1 ||
        weight.dims.size() != 1 || weight.dims[0] != input.dims.back() ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        weight.cudaData == nullptr) {
        return false;
    }
    int axis = (int)input.dims.size() - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    return LaunchFastllmRMSNormBFloat16(
        (__nv_bfloat16*)input.cudaData, (float*)weight.cudaData,
        (__nv_bfloat16*)output.cudaData, outer, channels, eps, threadCount);
}

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (input.dataType == fastllm::DataType::FLOAT32) {
        if (channels < 64) {
            FastllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                           channels, eps);
        } else if (channels < 512) {
            FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                             channels, eps);
        } else if (channels < 4096) {
            FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                               channels, eps);
        } else {
            FastllmRMSNormKernelInner1<1024> <<< outer, 1024 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                                 channels, eps);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        LaunchFastllmRMSNormFloat16(
            (half*)cudaInput, (float*)weight.cudaData, (half*)cudaOutput,
            outer, channels, eps, channels == 128 ? 32 : 0);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        int threadCount = 0;
#ifndef USE_ROCM
        threadCount = channels == 3072 ? 256 : 0;
#endif
        LaunchFastllmRMSNormBFloat16(
            (__nv_bfloat16*)cudaInput, (float*)weight.cudaData,
            (__nv_bfloat16*)cudaOutput, outer, channels, eps, threadCount);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRMSNormCombinedQKFloat16(
        const fastllm::Data &qkvInput, const fastllm::Data &weight,
        int batch, int seqLen, int keyHeads, int valueHeads,
        int kDim, int vDim, float eps,
        fastllm::Data &q, fastllm::Data &k) {
    auto isDense = [](const fastllm::Data &data) {
        if (data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (batch <= 0 || seqLen <= 0 ||
        keyHeads <= 0 || valueHeads < keyHeads ||
        kDim != 128 || vDim != 128 ||
        !std::isfinite(eps) || eps < 0.0f ||
        qkvInput.dataDevice != fastllm::DataDevice::CUDA ||
        qkvInput.dataType != fastllm::DataType::FLOAT16 ||
        qkvInput.cudaData == nullptr || !isDense(qkvInput) ||
        qkvInput.Count(0) !=
            (uint64_t)batch * seqLen *
            (keyHeads * kDim * 2 + valueHeads * vDim) ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        weight.cudaData == nullptr || !isDense(weight) ||
        weight.Count(0) != (uint64_t)kDim) {
        return false;
    }

    auto prepareOutput = [&](fastllm::Data &output) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = qkvInput.dataDevice;
        output.dataDeviceIds = qkvInput.dataDeviceIds;
        output.Resize({batch, seqLen, keyHeads, kDim});
        output.Allocate(false);
        return output.cudaData != nullptr;
    };
    if (!prepareOutput(q) || !prepareOutput(k)) {
        return false;
    }

    int rows = batch * seqLen;
    FastllmRMSNormCombinedQKHalf128ExactKernel<<<rows * keyHeads, 32>>>(
        (const half *)qkvInput.cudaData,
        (const float *)weight.cudaData,
        (half *)q.cudaData, (half *)k.cudaData,
        keyHeads, valueHeads, eps);
    checkCudaErrors(
        "Error: CUDA error in FastllmCudaRMSNormCombinedQKFloat16.",
        cudaGetLastError());
    return true;
}

bool FastllmCudaQwen35GdnPostConvExactFloat16(
        const fastllm::Data &qkvInput,
        const fastllm::Data &normWeight,
        const fastllm::Data &gInput,
        const fastllm::Data &betaInput,
        int batch, int seqLen, int keyHeads, int valueHeads,
        int kDim, int vDim, float normEps, float qScale,
        fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
        fastllm::Data &g, fastllm::Data &beta,
        fastllm::Data &kBeta, fastllm::Data &vBeta) {
    auto isDenseCuda = [](const fastllm::Data &data) {
        if (data.dataDevice != fastllm::DataDevice::CUDA ||
            data.cudaData == nullptr || data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (batch <= 0 || seqLen <= 0 || seqLen % 64 != 0 ||
        keyHeads <= 0 || valueHeads < keyHeads ||
        valueHeads % keyHeads != 0 ||
        kDim != 128 || vDim != 128 ||
        !std::isfinite(normEps) || normEps < 0.0f ||
        !std::isfinite(qScale) ||
        qkvInput.dataType != fastllm::DataType::FLOAT16 ||
        gInput.dataType != fastllm::DataType::FLOAT16 ||
        betaInput.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        !isDenseCuda(qkvInput) || !isDenseCuda(gInput) ||
        !isDenseCuda(betaInput) || !isDenseCuda(normWeight) ||
        qkvInput.Count(0) !=
            (uint64_t)batch * seqLen *
            (keyHeads * kDim * 2 + valueHeads * vDim) ||
        gInput.Count(0) !=
            (uint64_t)batch * seqLen * valueHeads ||
        betaInput.Count(0) !=
            (uint64_t)batch * seqLen * valueHeads ||
        normWeight.Count(0) != (uint64_t)kDim) {
        return false;
    }

    int paddedSeqLen = ((seqLen + 63) / 64) * 64;
    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = qkvInput.dataDevice;
        output.dataDeviceIds = qkvInput.dataDeviceIds;
        output.Resize(dims);
        output.Allocate(false);
        return output.cudaData != nullptr;
    };
    if (!prepareOutput(q, {batch, valueHeads, paddedSeqLen, kDim}) ||
        !prepareOutput(k, {batch, valueHeads, paddedSeqLen, kDim}) ||
        !prepareOutput(v, {batch, valueHeads, paddedSeqLen, vDim}) ||
        !prepareOutput(g, {batch, valueHeads, paddedSeqLen}) ||
        !prepareOutput(beta, {batch, valueHeads, paddedSeqLen}) ||
        !prepareOutput(kBeta, {batch, valueHeads, paddedSeqLen, kDim}) ||
        !prepareOutput(vBeta, {batch, valueHeads, paddedSeqLen, vDim})) {
        return false;
    }

    int rows = batch * paddedSeqLen;
    const char *skipDeadOutputsEnv = std::getenv(
        "FASTLLM_CUDA_QWEN35_GDN_SKIP_DEAD_POSTCONV_OUTPUTS");
    bool skipDeadOutputs =
        skipDeadOutputsEnv == nullptr || skipDeadOutputsEnv[0] == '\0' ||
        FastllmCudaEnvFlagEnabled(
            "FASTLLM_CUDA_QWEN35_GDN_SKIP_DEAD_POSTCONV_OUTPUTS");
    if (skipDeadOutputs) {
        FastllmQwen35GdnPostConvExactHalf128Kernel<false>
            <<<rows * keyHeads, 32>>>(
            (const half *)qkvInput.cudaData,
            (const float *)normWeight.cudaData,
            (const half *)gInput.cudaData,
            (const half *)betaInput.cudaData,
            (half *)q.cudaData, (half *)k.cudaData,
            (half *)v.cudaData, (half *)g.cudaData,
            (half *)beta.cudaData, (half *)kBeta.cudaData,
            (half *)vBeta.cudaData,
            seqLen, paddedSeqLen, keyHeads, valueHeads,
            normEps, __float2half(qScale));
    } else {
        FastllmQwen35GdnPostConvExactHalf128Kernel<true>
            <<<rows * keyHeads, 32>>>(
            (const half *)qkvInput.cudaData,
            (const float *)normWeight.cudaData,
            (const half *)gInput.cudaData,
            (const half *)betaInput.cudaData,
            (half *)q.cudaData, (half *)k.cudaData,
            (half *)v.cudaData, (half *)g.cudaData,
            (half *)beta.cudaData, (half *)kBeta.cudaData,
            (half *)vBeta.cudaData,
            seqLen, paddedSeqLen, keyHeads, valueHeads,
            normEps, __float2half(qScale));
    }
    checkCudaErrors(
        "Error: CUDA error in "
        "FastllmCudaQwen35GdnPostConvExactFloat16.",
        cudaGetLastError());
    return true;
}

bool FastllmCudaQwen35GdnPostConvRaggedExactFloat16(
        const fastllm::Data &qkvInput,
        const fastllm::Data &normWeight,
        const fastllm::Data &combinedBaInput,
        const fastllm::Data &aLog,
        const fastllm::Data &dtBias,
        int baOffset, const std::vector<int> &seqLens,
        int chunkSize, int keyHeads, int valueHeads,
        int kDim, int vDim, float normEps, float qScale,
        fastllm::Data &q, fastllm::Data &k,
        fastllm::Data &g, fastllm::Data &kBeta,
        fastllm::Data &vBeta) {
    auto isDenseCuda = [](const fastllm::Data &data) {
        if (data.dataDevice != fastllm::DataDevice::CUDA ||
            data.cudaData == nullptr || data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (seqLens.size() <= 1 || chunkSize != 64 ||
        keyHeads <= 0 || valueHeads <= keyHeads ||
        valueHeads % keyHeads != 0 || kDim != 128 || vDim != 128 ||
        baOffset < 0 || !std::isfinite(normEps) || normEps < 0.0f ||
        !std::isfinite(qScale) ||
        qkvInput.dataType != fastllm::DataType::FLOAT16 ||
        combinedBaInput.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        !isDenseCuda(qkvInput) || !isDenseCuda(combinedBaInput) ||
        !isDenseCuda(normWeight) || !isDenseCuda(aLog) ||
        !isDenseCuda(dtBias) ||
        normWeight.Count(0) != (uint64_t)kDim ||
        aLog.Count(0) != (uint64_t)valueHeads ||
        dtBias.Count(0) != (uint64_t)valueHeads) {
        return false;
    }
    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(
            seqLens, chunkSize, metadata)) {
        return false;
    }
    const int totalTokens = metadata.totalTokens;
    const int totalChunks = metadata.totalChunks;
    const int qkvChannels =
        keyHeads * kDim * 2 + valueHeads * vDim;
    const int baChannels = combinedBaInput.dims.back();
    if (totalTokens <= 0 || totalChunks <= 0 ||
        baOffset + valueHeads * 2 > baChannels ||
        qkvInput.dims.back() != qkvChannels ||
        qkvInput.Count(0) !=
            (uint64_t)totalTokens * qkvChannels ||
        combinedBaInput.Count(0) !=
            (uint64_t)totalTokens * baChannels) {
        return false;
    }
    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(qkvInput, device) ||
        FastllmCudaGetDevice() != device ||
        !FastllmCudaDataCanShareDevice(qkvInput, combinedBaInput) ||
        !FastllmCudaDataCanShareDevice(qkvInput, normWeight) ||
        !FastllmCudaDataCanShareDevice(qkvInput, aLog) ||
        !FastllmCudaDataCanShareDevice(qkvInput, dtBias)) {
        return false;
    }
    std::set<fastllm::Data*> outputs = {&q, &k, &g, &kBeta, &vBeta};
    if (outputs.size() != 5) {
        return false;
    }
    const int packedTokens = totalChunks * chunkSize;
    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.Resize(dims);
        output.ToDevice(
            fastllm::DataDevice::CUDA, std::vector<int>{device});
        output.Allocate(false);
        output.isKVCache = false;
        output.isLinearAttention = false;
        output.isLinearAttentionTransposed = false;
        return output.cudaData != nullptr &&
               FastllmCudaDataHasDenseStrides(output) &&
               FastllmCudaDataCanShareDevice(qkvInput, output);
    };
    if (!prepareOutput(q, {1, keyHeads, packedTokens, kDim}) ||
        !prepareOutput(k, {1, keyHeads, packedTokens, kDim}) ||
        !prepareOutput(g, {1, valueHeads, packedTokens}) ||
        !prepareOutput(kBeta, {1, valueHeads, packedTokens, kDim}) ||
        !prepareOutput(vBeta, {1, valueHeads, packedTokens, vDim})) {
        return false;
    }

    int rows = packedTokens * keyHeads;
    FastllmQwen35GdnPostConvRaggedExactHalf128Kernel<<<rows, 32>>>(
        (const half *)qkvInput.cudaData,
        (const float *)normWeight.cudaData,
        (const half *)combinedBaInput.cudaData,
        (const float *)aLog.cudaData,
        (const float *)dtBias.cudaData,
        metadata.chunkTokenBases, metadata.chunkValidTokens,
        (half *)q.cudaData, (half *)k.cudaData,
        (half *)g.cudaData, (half *)kBeta.cudaData,
        (half *)vBeta.cudaData, totalChunks, baChannels, baOffset,
        keyHeads, valueHeads, normEps, qScale);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in ragged exact GDN post-conv.",
            launchState);
        return false;
    }
    return true;
}

namespace {
    constexpr int KIMI_K3_CUDA_THREADS = 256;

    template <int THREADS>
    __device__ float KimiK3BlockReduceSum(float value, float *warpSums) {
        constexpr int warpSize = 32;
        constexpr int warps = (THREADS + warpSize - 1) / warpSize;
        int lane = threadIdx.x & (warpSize - 1);
        int warp = threadIdx.x / warpSize;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            warpSums[warp] = value;
        }
        __syncthreads();
        if (warp == 0) {
            float total = lane < warps ? warpSums[lane] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                total += __shfl_down_sync(0xffffffff, total, offset);
            }
            if (lane == 0) {
                warpSums[0] = total;
            }
        }
        __syncthreads();
        return warpSums[0];
    }

    __device__ __forceinline__ float KimiK3CudaSigmoid(float value) {
        return 1.0f / (1.0f + expf(-value));
    }

    __global__ void KimiK3RMSNormKernel(
            const __nv_bfloat16 *input, const float *weight,
            __nv_bfloat16 *output, int rows, int channels, float eps) {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        __shared__ float warpSums[KIMI_K3_CUDA_THREADS / 32];
        const __nv_bfloat16 *source = input + (size_t)row * channels;
        __nv_bfloat16 *destination = output + (size_t)row * channels;
        float partial = 0.0f;
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            float value = __bfloat162float(source[channel]);
            partial += value * value;
        }
        float squareSum =
            KimiK3BlockReduceSum<KIMI_K3_CUDA_THREADS>(partial, warpSums);
        float scale = rsqrtf(squareSum / channels + eps);
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            float value = __bfloat162float(source[channel]);
            float normalized = __bfloat162float(
                __float2bfloat16_rn(value * scale));
            destination[channel] =
                __float2bfloat16_rn(normalized * weight[channel]);
        }
    }

    __global__ void KimiK3L2NormKernel(
            const __nv_bfloat16 *input, __nv_bfloat16 *output,
            int rows, int channels, float eps) {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        __shared__ float warpSums[KIMI_K3_CUDA_THREADS / 32];
        const __nv_bfloat16 *source = input + (size_t)row * channels;
        __nv_bfloat16 *destination = output + (size_t)row * channels;
        float partial = 0.0f;
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            float value = __bfloat162float(source[channel]);
            partial += value * value;
        }
        float squareSum =
            KimiK3BlockReduceSum<KIMI_K3_CUDA_THREADS>(partial, warpSums);
        float scale = rsqrtf(squareSum + eps);
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            destination[channel] = __float2bfloat16_rn(
                __bfloat162float(source[channel]) * scale);
        }
    }

    __global__ void KimiK3RMSNormSigmoidGateKernel(
            const __nv_bfloat16 *input, const __nv_bfloat16 *gate,
            const float *weight, __nv_bfloat16 *output,
            int rows, int channels, float eps) {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        __shared__ float warpSums[KIMI_K3_CUDA_THREADS / 32];
        size_t base = (size_t)row * channels;
        float partial = 0.0f;
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            float value = __bfloat162float(input[base + channel]);
            partial += value * value;
        }
        float squareSum =
            KimiK3BlockReduceSum<KIMI_K3_CUDA_THREADS>(partial, warpSums);
        float scale = rsqrtf(squareSum / channels + eps);
        for (int channel = threadIdx.x; channel < channels;
             channel += blockDim.x) {
            float value = __bfloat162float(input[base + channel]);
            float gateValue = __bfloat162float(gate[base + channel]);
            output[base + channel] = __float2bfloat16_rn(
                value * scale * weight[channel] *
                KimiK3CudaSigmoid(gateValue));
        }
    }

    __global__ void KimiK3SiTUAndMulKernel(
            const __nv_bfloat16 *gate, const __nv_bfloat16 *up,
            __nv_bfloat16 *output, size_t count,
            float beta, float linearBeta) {
        size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) {
            return;
        }
        float gateValue = __bfloat162float(gate[index]);
        float upValue = __bfloat162float(up[index]);
        float situ = beta * tanhf(gateValue / beta) *
                     KimiK3CudaSigmoid(gateValue);
        float boundedUp = linearBeta > 0.0f ?
            linearBeta * tanhf(upValue / linearBeta) : upValue;
        output[index] = __float2bfloat16_rn(situ * boundedUp);
    }

    __global__ void KimiK3CausalConv1DKernel(
            const __nv_bfloat16 *input, const float *weight,
            const __nv_bfloat16 *cache, __nv_bfloat16 *output,
            int batch, int sequence, int channels, int kernelSize) {
        int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= batch * channels) {
            return;
        }
        int batchIndex = item / channels;
        int channel = item % channels;
        int history = kernelSize - 1;
        for (int token = 0; token < sequence; token++) {
            float value = 0.0f;
            for (int kernel = 0; kernel < kernelSize; kernel++) {
                int sourceToken = token - history + kernel;
                float sourceValue = 0.0f;
                if (sourceToken >= 0) {
                    size_t sourceIndex =
                        ((size_t)batchIndex * sequence + sourceToken) *
                        channels + channel;
                    sourceValue = __bfloat162float(input[sourceIndex]);
                } else if (cache != nullptr) {
                    int cacheToken = history + sourceToken;
                    size_t cacheIndex =
                        ((size_t)batchIndex * history + cacheToken) *
                        channels + channel;
                    sourceValue = __bfloat162float(cache[cacheIndex]);
                }
                value += sourceValue *
                    weight[(size_t)channel * kernelSize + kernel];
            }
            size_t outputIndex =
                ((size_t)batchIndex * sequence + token) * channels + channel;
            output[outputIndex] = __float2bfloat16_rn(
                value * KimiK3CudaSigmoid(value));
        }
    }

    __global__ void KimiK3CausalConv1DUpdateCacheKernel(
            const __nv_bfloat16 *input, __nv_bfloat16 *cache,
            int batch, int sequence, int channels, int history) {
        int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= batch * channels || history <= 0) {
            return;
        }
        int batchIndex = item / channels;
        int channel = item % channels;
        for (int token = 0; token < history; token++) {
            int combined = sequence + token;
            __nv_bfloat16 value;
            if (combined < history) {
                size_t sourceIndex =
                    ((size_t)batchIndex * history + combined) * channels +
                    channel;
                value = cache[sourceIndex];
            } else {
                int inputToken = combined - history;
                size_t sourceIndex =
                    ((size_t)batchIndex * sequence + inputToken) * channels +
                    channel;
                value = input[sourceIndex];
            }
            size_t destinationIndex =
                ((size_t)batchIndex * history + token) * channels + channel;
            cache[destinationIndex] = value;
        }
    }

    __global__ void KimiK3UpdatePackedConvCacheKernel(
            const __nv_bfloat16 *q, const __nv_bfloat16 *k,
            const __nv_bfloat16 *v, __nv_bfloat16 *cache,
            int batch, int sequence, int channels, int history,
            int tokens) {
        int item = blockIdx.x * blockDim.x + threadIdx.x;
        if (item >= 3 * batch * channels) {
            return;
        }
        int stream = item / (batch * channels);
        int withinStream = item % (batch * channels);
        int batchIndex = withinStream / channels;
        int channel = withinStream % channels;
        const __nv_bfloat16 *source = stream == 0 ? q :
            (stream == 1 ? k : v);
        size_t cacheBase =
            ((size_t)stream * batch + batchIndex) * history * channels;
        // `tokens` is positive, so any retained cache slot is read from a
        // strictly higher index.  Ascending writes are therefore safe in
        // place and need no temporary history buffer.
        for (int slot = 0; slot < history; slot++) {
            int combined = tokens + slot;
            __nv_bfloat16 value;
            if (combined < history) {
                value = cache[
                    cacheBase + (size_t)combined * channels + channel];
            } else {
                int inputToken = combined - history;
                value = source[
                    ((size_t)batchIndex * sequence + inputToken) * channels +
                    channel];
            }
            cache[cacheBase + (size_t)slot * channels + channel] = value;
        }
    }

    __global__ void KimiK3RecurrentKDAKernel(
            const __nv_bfloat16 *q, const __nv_bfloat16 *k,
            const __nv_bfloat16 *v, const __nv_bfloat16 *rawGate,
            const float *rawBeta, const float *aLog, const float *dtBias,
            float *state, __nv_bfloat16 *output, float *decay,
            float *activatedBeta, int batch, int inputSequence, int tokens,
            int heads, int dimension, int aLogCount, float lowerBound) {
        int item = blockIdx.x;
        int batchIndex = item / heads;
        int head = item % heads;
        int channel = threadIdx.x;
        if (batchIndex >= batch) {
            return;
        }
        extern __shared__ float shared[];
        float *key = shared;
        float *query = key + dimension;
        float *delta = query + dimension;
        float *retention = delta + dimension;
        float *sharedBeta = retention + dimension;
        float *headState = state +
            ((size_t)batchIndex * heads + head) * dimension * dimension;
        float outputScale = rsqrtf((float)dimension);

        for (int token = 0; token < tokens; token++) {
            size_t vectorBase =
                (((size_t)batchIndex * inputSequence + token) * heads + head) *
                dimension;
            size_t betaIndex =
                ((size_t)batchIndex * inputSequence + token) * heads + head;
            if (channel == 0) {
                sharedBeta[0] = KimiK3CudaSigmoid(rawBeta[betaIndex]);
                if (activatedBeta != nullptr) {
                    activatedBeta[betaIndex] = sharedBeta[0];
                }
            }
            if (channel < dimension) {
                key[channel] = __bfloat162float(k[vectorBase + channel]);
                if (output != nullptr) {
                    query[channel] =
                        __bfloat162float(q[vectorBase + channel]);
                }
                float raw = __bfloat162float(rawGate[vectorBase + channel]);
                float a = aLogCount == heads ? aLog[head] : aLog[channel];
                float gate = lowerBound * KimiK3CudaSigmoid(
                    expf(a) *
                    (raw + dtBias[(size_t)head * dimension + channel]));
                if (decay != nullptr) {
                    decay[vectorBase + channel] = gate;
                }
                retention[channel] = expf(gate);
            }
            __syncthreads();

            // Traverse the row-major state as a flat array.  Adjacent CUDA
            // threads now touch adjacent values, instead of every thread
            // walking a different row with a dimension-sized stride.
            int matrixItems = dimension * dimension;
            for (int index = channel; index < matrixItems;
                index += blockDim.x) {
                int row = index / dimension;
                headState[index] *= retention[row];
            }
            __syncthreads();

            if (channel < dimension) {
                float prediction = 0.0f;
                for (int sourceChannel = 0; sourceChannel < dimension;
                     sourceChannel++) {
                    prediction += key[sourceChannel] *
                        headState[(size_t)sourceChannel * dimension + channel];
                }
                delta[channel] =
                    (__bfloat162float(v[vectorBase + channel]) - prediction) *
                    sharedBeta[0];
            }
            __syncthreads();

            for (int index = channel; index < matrixItems;
                 index += blockDim.x) {
                int row = index / dimension;
                int value = index - row * dimension;
                headState[index] += key[row] * delta[value];
            }
            __syncthreads();

            if (output != nullptr && channel < dimension) {
                float result = 0.0f;
                for (int sourceChannel = 0; sourceChannel < dimension;
                     sourceChannel++) {
                    result += query[sourceChannel] *
                        headState[(size_t)sourceChannel * dimension + channel];
                }
                output[vectorBase + channel] =
                    __float2bfloat16_rn(result * outputScale);
            }
            __syncthreads();
        }
    }

    __global__ void KimiK3AttnResScoreKernel(
            const __nv_bfloat16 *prefixSum,
            const __nv_bfloat16 *blockResidual,
            const float *projection, const float *norm, float *scores,
            int rows, int blocks, int dimension, float eps) {
        int vector = blockIdx.x;
        int row = vector / (blocks + 1);
        int block = vector % (blocks + 1);
        if (row >= rows) {
            return;
        }
        const __nv_bfloat16 *source = block == blocks ?
            prefixSum + (size_t)row * dimension :
            blockResidual + ((size_t)row * blocks + block) * dimension;
        __shared__ float warpSums[KIMI_K3_CUDA_THREADS / 32];
        float partialSquares = 0.0f;
        for (int channel = threadIdx.x; channel < dimension;
             channel += blockDim.x) {
            float value = __bfloat162float(source[channel]);
            partialSquares += value * value;
        }
        float squareSum = KimiK3BlockReduceSum<KIMI_K3_CUDA_THREADS>(
            partialSquares, warpSums);
        float scale = rsqrtf(squareSum / dimension + eps);
        float partialScore = 0.0f;
        for (int channel = threadIdx.x; channel < dimension;
             channel += blockDim.x) {
            float value = __bfloat162float(source[channel]);
            partialScore += value * scale * norm[channel] * projection[channel];
        }
        float score = KimiK3BlockReduceSum<KIMI_K3_CUDA_THREADS>(
            partialScore, warpSums);
        if (threadIdx.x == 0) {
            scores[(size_t)row * (blocks + 1) + block] = score;
        }
    }

    __global__ void KimiK3AttnResMixKernel(
            const __nv_bfloat16 *prefixSum,
            const __nv_bfloat16 *blockResidual, float *scores,
            __nv_bfloat16 *output, int rows, int blocks, int dimension) {
        int row = blockIdx.x;
        if (row >= rows) {
            return;
        }
        float *probabilities = scores + (size_t)row * (blocks + 1);
        if (threadIdx.x == 0) {
            float maximum = probabilities[blocks];
            for (int block = 0; block < blocks; block++) {
                maximum = fmaxf(maximum, probabilities[block]);
            }
            if (blocks == 1) {
                float prefixExp = expf(probabilities[1] - maximum);
                float residualExp = expf(probabilities[0] - maximum);
                probabilities[0] = residualExp / (prefixExp + residualExp);
                probabilities[1] = 1.0f - probabilities[0];
            } else {
                float denominator = 0.0f;
                for (int item = 0; item <= blocks; item++) {
                    probabilities[item] = expf(probabilities[item] - maximum);
                    denominator += probabilities[item];
                }
                for (int item = 0; item <= blocks; item++) {
                    probabilities[item] /= denominator;
                }
            }
        }
        __syncthreads();
        size_t prefixBase = (size_t)row * dimension;
        for (int channel = threadIdx.x; channel < dimension;
             channel += blockDim.x) {
            float value = probabilities[blocks] *
                __bfloat162float(prefixSum[prefixBase + channel]);
            for (int block = 0; block < blocks; block++) {
                size_t residualIndex =
                    ((size_t)row * blocks + block) * dimension + channel;
                value += probabilities[block] *
                    __bfloat162float(blockResidual[residualIndex]);
            }
            output[prefixBase + channel] = __float2bfloat16_rn(value);
        }
    }

    bool KimiK3CudaDataReady(const fastllm::Data &data,
                             fastllm::DataType type) {
        return data.dataDevice == fastllm::DataDevice::CUDA &&
               data.dataType == type && data.cudaData != nullptr;
    }

    bool KimiK3CudaLastError(const char *message) {
        cudaError_t status = cudaGetLastError();
        showError(status, message, __FILE__, __LINE__);
        return status == cudaSuccess;
    }
}

bool FastllmCudaKimiK3RMSNorm(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data &output, float eps) {
    if (!KimiK3CudaDataReady(input, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(weight, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        input.dims.empty() || input.dims != output.dims ||
        weight.Count(0) != (uint64_t)input.dims.back()) {
        return false;
    }
    int channels = input.dims.back();
    int rows = (int)(input.Count(0) / channels);
    KimiK3RMSNormKernel<<<rows, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)input.cudaData,
        (const float*)weight.cudaData,
        (__nv_bfloat16*)output.cudaData, rows, channels, eps);
    return KimiK3CudaLastError("KimiK3RMSNorm CUDA kernel failed.");
}

bool FastllmCudaKimiK3L2Norm(
        const fastllm::Data &input, fastllm::Data &output, float eps) {
    if (!KimiK3CudaDataReady(input, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        input.dims.empty() || input.dims != output.dims) {
        return false;
    }
    int channels = input.dims.back();
    int rows = (int)(input.Count(0) / channels);
    KimiK3L2NormKernel<<<rows, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)input.cudaData,
        (__nv_bfloat16*)output.cudaData, rows, channels, eps);
    return KimiK3CudaLastError("KimiK3L2Norm CUDA kernel failed.");
}

bool FastllmCudaKimiK3RMSNormSigmoidGate(
        const fastllm::Data &input, const fastllm::Data &gate,
        const fastllm::Data &weight, fastllm::Data &output, float eps) {
    if (!KimiK3CudaDataReady(input, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(gate, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(weight, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        input.dims.empty() || input.dims != gate.dims ||
        input.dims != output.dims ||
        weight.Count(0) != (uint64_t)input.dims.back()) {
        return false;
    }
    int channels = input.dims.back();
    int rows = (int)(input.Count(0) / channels);
    KimiK3RMSNormSigmoidGateKernel<<<rows, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)input.cudaData,
        (const __nv_bfloat16*)gate.cudaData,
        (const float*)weight.cudaData,
        (__nv_bfloat16*)output.cudaData, rows, channels, eps);
    return KimiK3CudaLastError(
        "KimiK3RMSNormSigmoidGate CUDA kernel failed.");
}

bool FastllmCudaKimiK3SiTUAndMul(
        const fastllm::Data &gate, const fastllm::Data &up,
        fastllm::Data &output, float beta, float linearBeta) {
    if (!KimiK3CudaDataReady(gate, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(up, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        gate.dims != up.dims || gate.dims != output.dims || beta <= 0.0f) {
        return false;
    }
    size_t count = gate.Count(0);
    int blocks = (int)((count + KIMI_K3_CUDA_THREADS - 1) /
                       KIMI_K3_CUDA_THREADS);
    KimiK3SiTUAndMulKernel<<<blocks, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)gate.cudaData,
        (const __nv_bfloat16*)up.cudaData,
        (__nv_bfloat16*)output.cudaData, count, beta, linearBeta);
    return KimiK3CudaLastError("KimiK3SiTUAndMul CUDA kernel failed.");
}

bool FastllmCudaKimiK3CausalConv1D(
        const fastllm::Data &input, const fastllm::Data &weight,
        fastllm::Data *cache, fastllm::Data &output, int kernelSize,
        bool initializeCache) {
    if (!KimiK3CudaDataReady(input, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(weight, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        input.dims.size() != 3 || weight.dims.size() != 3 ||
        output.dims != input.dims || weight.dims[1] != 1 ||
        weight.dims[2] != kernelSize || weight.dims[0] != input.dims[2] ||
        kernelSize <= 0 ||
        (cache != nullptr &&
         (!KimiK3CudaDataReady(*cache, fastllm::DataType::BFLOAT16) ||
          cache->dims != std::vector<int>({input.dims[0], kernelSize - 1,
                                           input.dims[2]})))) {
        return false;
    }
    if (initializeCache && cache != nullptr) {
        FastllmCudaMemset0(cache->cudaData, cache->GetBytes());
    }
    int batch = input.dims[0];
    int sequence = input.dims[1];
    int channels = input.dims[2];
    int items = batch * channels;
    int blocks = (items + KIMI_K3_CUDA_THREADS - 1) /
                 KIMI_K3_CUDA_THREADS;
    KimiK3CausalConv1DKernel<<<blocks, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)input.cudaData,
        (const float*)weight.cudaData,
        cache == nullptr ? nullptr :
            (const __nv_bfloat16*)cache->cudaData,
        (__nv_bfloat16*)output.cudaData,
        batch, sequence, channels, kernelSize);
    if (cache != nullptr && kernelSize > 1) {
        KimiK3CausalConv1DUpdateCacheKernel
            <<<blocks, KIMI_K3_CUDA_THREADS>>>(
                (const __nv_bfloat16*)input.cudaData,
                (__nv_bfloat16*)cache->cudaData,
                batch, sequence, channels, kernelSize - 1);
    }
    return KimiK3CudaLastError("KimiK3CausalConv1D CUDA kernel failed.");
}

bool FastllmCudaKimiK3UpdatePackedConvCache(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, fastllm::Data &cache,
        int history, int tokens) {
    if (!KimiK3CudaDataReady(q, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(k, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(v, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(cache, fastllm::DataType::BFLOAT16) ||
        q.dims.size() != 3 || q.dims != k.dims || q.dims != v.dims ||
        history <= 0 || tokens <= 0 || tokens > q.dims[1]) {
        return false;
    }
    int batch = q.dims[0];
    int sequence = q.dims[1];
    int channels = q.dims[2];
    bool legacyLayout = batch == 1 &&
        cache.dims == std::vector<int>({3, history, channels});
    bool batchedLayout = cache.dims ==
        std::vector<int>({3, batch, history, channels});
    if (!legacyLayout && !batchedLayout) {
        return false;
    }
    int items = 3 * batch * channels;
    int blocks = (items + KIMI_K3_CUDA_THREADS - 1) /
                 KIMI_K3_CUDA_THREADS;
    KimiK3UpdatePackedConvCacheKernel
        <<<blocks, KIMI_K3_CUDA_THREADS>>>(
            (const __nv_bfloat16*)q.cudaData,
            (const __nv_bfloat16*)k.cudaData,
            (const __nv_bfloat16*)v.cudaData,
            (__nv_bfloat16*)cache.cudaData,
            batch, sequence, channels, history, tokens);
    return KimiK3CudaLastError(
        "KimiK3UpdatePackedConvCache CUDA kernel failed.");
}

bool FastllmCudaKimiK3RecurrentKDA(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &rawGate,
        const fastllm::Data &rawBeta, const fastllm::Data &aLog,
        const fastllm::Data &dtBias, fastllm::Data &state,
        fastllm::Data &output, fastllm::Data &decay,
        fastllm::Data &beta, float lowerBound, bool initializeState,
        int tokenLimit, bool stateOnly, bool outputAux) {
    if (!KimiK3CudaDataReady(q, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(k, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(v, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(rawGate, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(rawBeta, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(aLog, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(dtBias, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(state, fastllm::DataType::FLOAT32) ||
        q.dims.size() != 4 || q.dims != k.dims || q.dims != v.dims ||
        q.dims != rawGate.dims ||
        q.dims[3] <= 0 || q.dims[3] > 256) {
        return false;
    }
    int batch = q.dims[0];
    int sequence = q.dims[1];
    int heads = q.dims[2];
    int dimension = q.dims[3];
    int tokens = tokenLimit < 0 ? sequence : tokenLimit;
    if (rawBeta.dims != std::vector<int>({batch, sequence, heads}) ||
        state.dims != std::vector<int>({batch, heads, dimension, dimension}) ||
        tokens <= 0 || tokens > sequence ||
        (!stateOnly && tokens != sequence) ||
        (aLog.Count(0) != (uint64_t)heads &&
         aLog.Count(0) != (uint64_t)dimension) ||
        dtBias.Count(0) != (uint64_t)heads * dimension) {
        return false;
    }
    if (!stateOnly) {
        if (!KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
            output.dims != q.dims) {
            return false;
        }
        if (outputAux &&
            (!KimiK3CudaDataReady(decay, fastllm::DataType::FLOAT32) ||
             !KimiK3CudaDataReady(beta, fastllm::DataType::FLOAT32) ||
             decay.dims != q.dims ||
             beta.dims != std::vector<int>({batch, sequence, heads}))) {
            return false;
        }
    }
    if (initializeState) {
        FastllmCudaMemset0(state.cudaData, state.GetBytes());
    }
    // Vector reductions retain one owner per state column, while all threads
    // cooperate on the row-major decay and rank-one update passes.
    int threads = KIMI_K3_CUDA_THREADS;
    size_t sharedBytes = ((size_t)dimension * 4 + 1) * sizeof(float);
    KimiK3RecurrentKDAKernel<<<batch * heads, threads, sharedBytes>>>(
        (const __nv_bfloat16*)q.cudaData,
        (const __nv_bfloat16*)k.cudaData,
        (const __nv_bfloat16*)v.cudaData,
        (const __nv_bfloat16*)rawGate.cudaData,
        (const float*)rawBeta.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (float*)state.cudaData,
        stateOnly ? nullptr : (__nv_bfloat16*)output.cudaData,
        stateOnly || !outputAux ? nullptr : (float*)decay.cudaData,
        stateOnly || !outputAux ? nullptr : (float*)beta.cudaData,
        batch, sequence, tokens, heads, dimension,
        (int)aLog.Count(0), lowerBound);
    return KimiK3CudaLastError("KimiK3RecurrentKDA CUDA kernel failed.");
}

bool FastllmCudaKimiK3AttnRes(
        const fastllm::Data &prefixSum,
        const fastllm::Data &blockResidual,
        const fastllm::Data &projection, const fastllm::Data &norm,
        fastllm::Data &output, float eps) {
    if (!KimiK3CudaDataReady(prefixSum, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(blockResidual, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(projection, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(norm, fastllm::DataType::FLOAT32) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        prefixSum.dims.empty() || prefixSum.dims != output.dims ||
        blockResidual.dims.size() != 3) {
        return false;
    }
    int dimension = prefixSum.dims.back();
    int rows = (int)(prefixSum.Count(0) / dimension);
    int blocks = blockResidual.dims[1];
    if (blockResidual.dims[0] != rows ||
        blockResidual.dims[2] != dimension || blocks <= 0 ||
        projection.Count(0) != (uint64_t)dimension ||
        norm.Count(0) != (uint64_t)dimension) {
        return false;
    }
    size_t scratchBytes = (size_t)rows * (blocks + 1) * sizeof(float);
    size_t borrowedBytes = 0;
    bool own = false;
    void *scratch = FastllmBorrowCudaTempBuffer(
        scratchBytes, &borrowedBytes, &own);
    if (scratch == nullptr || borrowedBytes < scratchBytes) {
        FastllmReleaseCudaTempBuffer(scratch, own);
        return false;
    }
    KimiK3AttnResScoreKernel
        <<<rows * (blocks + 1), KIMI_K3_CUDA_THREADS>>>(
            (const __nv_bfloat16*)prefixSum.cudaData,
            (const __nv_bfloat16*)blockResidual.cudaData,
            (const float*)projection.cudaData,
            (const float*)norm.cudaData,
            (float*)scratch, rows, blocks, dimension, eps);
    KimiK3AttnResMixKernel<<<rows, KIMI_K3_CUDA_THREADS>>>(
        (const __nv_bfloat16*)prefixSum.cudaData,
        (const __nv_bfloat16*)blockResidual.cudaData,
        (float*)scratch, (__nv_bfloat16*)output.cudaData,
        rows, blocks, dimension);
    FastllmReleaseCudaTempBuffer(scratch, own);
    return KimiK3CudaLastError("KimiK3AttnRes CUDA kernel failed.");
}

namespace {
    __global__ void KimiK3CausalAttentionKernel(
            const __nv_bfloat16 *q, const __nv_bfloat16 *k,
            const __nv_bfloat16 *v, float *scores,
            __nv_bfloat16 *output,
            int batch, int heads, int queryLength, int keyLength,
            int queryDimension, int valueDimension, int standardCache,
            uint64_t qStride0, uint64_t qStride1,
            uint64_t qStride2, uint64_t qStride3,
            uint64_t kStride0, uint64_t kStride1,
            uint64_t kStride2, uint64_t kStride3,
            uint64_t vStride0, uint64_t vStride1,
            uint64_t vStride2, uint64_t vStride3,
            uint64_t oStride0, uint64_t oStride1,
            uint64_t oStride2, uint64_t oStride3,
            float scale) {
        int item = blockIdx.x;
        int queryIndex = item % queryLength;
        int headItem = item / queryLength;
        int head = headItem % heads;
        int batchIndex = headItem / heads;
        if (batchIndex >= batch) {
            return;
        }
        int lastKey = keyLength - queryLength + queryIndex;
        size_t scoreBase = (size_t)item * keyLength;
        uint64_t qBase = (uint64_t)batchIndex * qStride0 +
                         (uint64_t)head * qStride1 +
                         (uint64_t)queryIndex * qStride2;
        for (int keyIndex = threadIdx.x; keyIndex <= lastKey;
             keyIndex += blockDim.x) {
            uint64_t kBase = standardCache ?
                (uint64_t)head * kStride0 +
                    (uint64_t)keyIndex * kStride1 :
                (uint64_t)batchIndex * kStride0 +
                    (uint64_t)head * kStride1 +
                    (uint64_t)keyIndex * kStride2;
            float dot = 0.0f;
            for (int channel = 0; channel < queryDimension; channel++) {
                dot += __bfloat162float(q[qBase +
                    (uint64_t)channel * qStride3]) *
                       __bfloat162float(k[kBase +
                    (uint64_t)channel *
                        (standardCache ? kStride2 : kStride3)]);
            }
            // Match eager_attention_forward: einsum output and its scaled
            // score are BF16 before the FP32 softmax.
            float roundedDot = __bfloat162float(__float2bfloat16_rn(dot));
            scores[scoreBase + keyIndex] = __bfloat162float(
                __float2bfloat16_rn(roundedDot * scale));
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float maximum = -3.402823466e+38F;
            for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                maximum = fmaxf(maximum, scores[scoreBase + keyIndex]);
            }
            float denominator = 0.0f;
            for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                float probability =
                    expf(scores[scoreBase + keyIndex] - maximum);
                scores[scoreBase + keyIndex] = probability;
                denominator += probability;
            }
            for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                // Store the BF16 probability as a float so the following
                // value reduction observes the same rounded operand as CPU.
                scores[scoreBase + keyIndex] = __bfloat162float(
                    __float2bfloat16_rn(
                        scores[scoreBase + keyIndex] / denominator));
            }
        }
        __syncthreads();

        uint64_t outputBase = (uint64_t)batchIndex * oStride0 +
                              (uint64_t)head * oStride1 +
                              (uint64_t)queryIndex * oStride2;
        for (int channel = threadIdx.x; channel < valueDimension;
             channel += blockDim.x) {
            float value = 0.0f;
            for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                uint64_t vIndex = standardCache ?
                    (uint64_t)head * vStride0 +
                        (uint64_t)keyIndex * vStride1 +
                        (uint64_t)channel * vStride2 :
                    (uint64_t)batchIndex * vStride0 +
                        (uint64_t)head * vStride1 +
                        (uint64_t)keyIndex * vStride2 +
                        (uint64_t)channel * vStride3;
                value += scores[scoreBase + keyIndex] *
                         __bfloat162float(v[vIndex]);
            }
            output[outputBase + (uint64_t)channel * oStride3] =
                __float2bfloat16_rn(value);
        }
    }
}

bool FastllmCudaKimiK3CausalAttention(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, fastllm::Data &output, float scale) {
    if (!KimiK3CudaDataReady(q, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(k, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(v, fastllm::DataType::BFLOAT16) ||
        !KimiK3CudaDataReady(output, fastllm::DataType::BFLOAT16) ||
        q.dims.size() != 4 || output.dims.size() != 4) {
        return false;
    }
    bool standardCache = k.dims.size() == 3 && v.dims.size() == 3;
    bool currentTensors = k.dims.size() == 4 && v.dims.size() == 4;
    if (!standardCache && !currentTensors) {
        return false;
    }
    int batch = q.dims[0];
    int heads = q.dims[1];
    int queryLength = q.dims[2];
    int queryDimension = q.dims[3];
    int keyLength = standardCache ? k.dims[1] : k.dims[2];
    int valueDimension = v.dims.back();
    if (batch <= 0 || heads <= 0 || queryLength <= 0 ||
        keyLength < queryLength || queryDimension <= 0 ||
        valueDimension <= 0 ||
        output.dims != std::vector<int>(
            {batch, heads, queryLength, valueDimension})) {
        return false;
    }
    if (standardCache) {
        if (batch != 1 || k.dims[0] != heads || v.dims[0] != heads ||
            k.dims[1] != v.dims[1] || k.dims[2] != queryDimension) {
            return false;
        }
    } else if (q.dims[0] != k.dims[0] || q.dims[0] != v.dims[0] ||
               q.dims[1] != k.dims[1] || q.dims[1] != v.dims[1] ||
               k.dims[2] != v.dims[2] ||
               k.dims[3] != queryDimension) {
        return false;
    }

    size_t items = (size_t)batch * heads * queryLength;
    if (items > SIZE_MAX / (size_t)keyLength / sizeof(float)) {
        return false;
    }
    size_t scratchBytes = items * keyLength * sizeof(float);
    size_t borrowedBytes = 0;
    bool own = false;
    void *scratch = FastllmBorrowCudaTempBuffer(
        scratchBytes, &borrowedBytes, &own);
    if (scratch == nullptr || borrowedBytes < scratchBytes) {
        FastllmReleaseCudaTempBuffer(scratch, own);
        return false;
    }
    uint64_t kStride3 = standardCache ? 0 : k.strides[3];
    uint64_t vStride3 = standardCache ? 0 : v.strides[3];
    KimiK3CausalAttentionKernel<<<(int)items, 128>>>(
        (const __nv_bfloat16*)q.cudaData,
        (const __nv_bfloat16*)k.cudaData,
        (const __nv_bfloat16*)v.cudaData,
        (float*)scratch, (__nv_bfloat16*)output.cudaData,
        batch, heads, queryLength, keyLength,
        queryDimension, valueDimension, standardCache ? 1 : 0,
        q.strides[0], q.strides[1], q.strides[2], q.strides[3],
        k.strides[0], k.strides[1], k.strides[2], kStride3,
        v.strides[0], v.strides[1], v.strides[2], vStride3,
        output.strides[0], output.strides[1],
        output.strides[2], output.strides[3], scale);
    FastllmReleaseCudaTempBuffer(scratch, own);
    return KimiK3CudaLastError(
        "KimiK3CausalAttention CUDA kernel failed.");
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormSiluMulHalfKernel(const half *input, const float *weight,
                                                const half *gateInput, half *output,
                                                int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    gateInput = gateInput + o * channels;
    output = output + o * channels;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int half2_channels = channels / 2;
    const half2 *input_h2 = reinterpret_cast<const half2 *>(input);
    float sum2 = 0.0f;
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);
        sum2 += fv.x * fv.x + fv.y * fv.y;
    }
    if ((channels & 1) && tid == 0) {
        float x = __half2float(input[channels - 1]);
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / channels + eps);
        }
    }
    __syncthreads();

    float s = scale;
    half2 *output_h2 = reinterpret_cast<half2 *>(output);
    const half2 *gate_h2 = reinterpret_cast<const half2 *>(gateInput);
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);

        half2 gateVec = gate_h2[i];
        half gate0In = __low2half(gateVec);
        half gate1In = __high2half(gateVec);

#ifdef CUDA_NO_TENSOR_CORE
        float gate0Float = __half2float(gate0In);
        float gate1Float = __half2float(gate1In);
        half gate0 = __float2half(gate0Float / (1.0f + expf(-gate0Float)));
        half gate1 = __float2half(gate1Float / (1.0f + expf(-gate1Float)));
#else
        half gate0 = __hdiv(gate0In, __hadd(__float2half(1.0f), hexp(-gate0In)));
        half gate1 = __hdiv(gate1In, __hadd(__float2half(1.0f), hexp(-gate1In)));
#endif

        half rms0 = __float2half_rn(fv.x * s * __ldg(&weight[i * 2]));
        half rms1 = __float2half_rn(fv.y * s * __ldg(&weight[i * 2 + 1]));

#ifdef CUDA_NO_TENSOR_CORE
        half out0 = __float2half(__half2float(rms0) * __half2float(gate0));
        half out1 = __float2half(__half2float(rms1) * __half2float(gate1));
#else
        half out0 = __hmul(rms0, gate0);
        half out1 = __hmul(rms1, gate1);
#endif
        output_h2[i] = __halves2half2(out0, out1);
    }

    if ((channels & 1) && tid == 0) {
        int last = channels - 1;
#ifdef CUDA_NO_TENSOR_CORE
        float gateFloat = __half2float(gateInput[last]);
        half gate = __float2half(gateFloat / (1.0f + expf(-gateFloat)));
        half rms = __float2half(__half2float(input[last]) * s * __ldg(&weight[last]));
        output[last] = __float2half(__half2float(rms) * __half2float(gate));
#else
        half gate = __hdiv(gateInput[last], __hadd(__float2half(1.0f), hexp(-gateInput[last])));
        half rms = __float2half_rn(__half2float(input[last]) * s * __ldg(&weight[last]));
        output[last] = __hmul(rms, gate);
#endif
    }
}

// Apply the same exact two-reduction mapping to the fused recurrent gate path,
// while keeping both input and gate values in registers for in-place safety.
__global__ __launch_bounds__(32) void FastllmRMSNormSiluMulHalf128ExactKernel(
        const half *input, const float *weight, const half *gateInput,
        half *output, float eps) {
    constexpr int CHANNELS = 128;
    int row = blockIdx.x;
    int lane = threadIdx.x;
    input += (size_t)row * CHANNELS;
    gateInput += (size_t)row * CHANNELS;
    output += (size_t)row * CHANNELS;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    const half2 *gate2 = reinterpret_cast<const half2 *>(gateInput);

    float2 value0 = __half22float2(input2[lane]);
    float2 value1 = __half22float2(input2[lane + 32]);
    float sum0 = value0.x * value0.x + value0.y * value0.y;
    float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    float scale = 0.0f;
    if (lane == 0) {
        scale = rsqrtf((sum0 + sum1) / CHANNELS + eps);
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);

    float2 inputValues[2] = {value0, value1};
    half2 gateValues[2] = {gate2[lane], gate2[lane + 32]};
    half2 *output2 = reinterpret_cast<half2 *>(output);
#pragma unroll
    for (int part = 0; part < 2; part++) {
        int index = lane + part * 32;
        float2 value = inputValues[part];
        half gate0In = __low2half(gateValues[part]);
        half gate1In = __high2half(gateValues[part]);
#ifdef CUDA_NO_TENSOR_CORE
        float gate0Float = __half2float(gate0In);
        float gate1Float = __half2float(gate1In);
        half gate0 = __float2half(gate0Float / (1.0f + expf(-gate0Float)));
        half gate1 = __float2half(gate1Float / (1.0f + expf(-gate1Float)));
#else
        half gate0 = __hdiv(gate0In, __hadd(__float2half(1.0f), hexp(-gate0In)));
        half gate1 = __hdiv(gate1In, __hadd(__float2half(1.0f), hexp(-gate1In)));
#endif
        half rms0 = __float2half_rn(value.x * scale * __ldg(weight + index * 2));
        half rms1 = __float2half_rn(value.y * scale * __ldg(weight + index * 2 + 1));
#ifdef CUDA_NO_TENSOR_CORE
        half out0 = __float2half(__half2float(rms0) * __half2float(gate0));
        half out1 = __float2half(__half2float(rms1) * __half2float(gate1));
#else
        half out0 = __hmul(rms0, gate0);
        half out1 = __hmul(rms1, gate1);
#endif
        output2[index] = __halves2half2(out0, out1);
    }
}

// Keep the exact 128-channel RMSNorm and SiLU arithmetic above, but read the
// gate from one slice of a wider token-major projection.  Each logical input
// row is one value head, while combinedGateInput advances once per token.
__global__ __launch_bounds__(32) void FastllmRMSNormSiluMulHalf128CombinedGateExactKernel(
        const half *input, const float *weight,
        const half *combinedGateInput, half *output,
        int gateStride, int gateOffset, int gateHeads, float eps) {
    constexpr int CHANNELS = 128;
    int row = blockIdx.x;
    int lane = threadIdx.x;
    int token = row / gateHeads;
    int head = row - token * gateHeads;
    input += (size_t)row * CHANNELS;
    combinedGateInput +=
        (size_t)token * gateStride + gateOffset + head * CHANNELS;
    output += (size_t)row * CHANNELS;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    const half2 *gate2 =
        reinterpret_cast<const half2 *>(combinedGateInput);

    float2 value0 = __half22float2(input2[lane]);
    float2 value1 = __half22float2(input2[lane + 32]);
    float sum0 = value0.x * value0.x + value0.y * value0.y;
    float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    float scale = 0.0f;
    if (lane == 0) {
        scale = rsqrtf((sum0 + sum1) / CHANNELS + eps);
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);

    float2 inputValues[2] = {value0, value1};
    half2 gateValues[2] = {gate2[lane], gate2[lane + 32]};
    half2 *output2 = reinterpret_cast<half2 *>(output);
#pragma unroll
    for (int part = 0; part < 2; part++) {
        int index = lane + part * 32;
        float2 value = inputValues[part];
        half gate0In = __low2half(gateValues[part]);
        half gate1In = __high2half(gateValues[part]);
#ifdef CUDA_NO_TENSOR_CORE
        float gate0Float = __half2float(gate0In);
        float gate1Float = __half2float(gate1In);
        half gate0 =
            __float2half(gate0Float / (1.0f + expf(-gate0Float)));
        half gate1 =
            __float2half(gate1Float / (1.0f + expf(-gate1Float)));
#else
        half gate0 =
            __hdiv(gate0In, __hadd(__float2half(1.0f), hexp(-gate0In)));
        half gate1 =
            __hdiv(gate1In, __hadd(__float2half(1.0f), hexp(-gate1In)));
#endif
        half rms0 = __float2half_rn(
            value.x * scale * __ldg(weight + index * 2));
        half rms1 = __float2half_rn(
            value.y * scale * __ldg(weight + index * 2 + 1));
#ifdef CUDA_NO_TENSOR_CORE
        half out0 =
            __float2half(__half2float(rms0) * __half2float(gate0));
        half out1 =
            __float2half(__half2float(rms1) * __half2float(gate1));
#else
        half out0 = __hmul(rms0, gate0);
        half out1 = __hmul(rms1, gate1);
#endif
        output2[index] = __halves2half2(out0, out1);
    }
}

// Fuse [batch, heads, paddedSeq, 128] -> [batch, seq, heads, 128]
// with the exact output RMSNorm and z gate.  This avoids the temporary copy
// required by an in-place head/sequence transpose.
__global__ __launch_bounds__(32) void FastllmRMSNormSiluMulHalf128HeadMajorCombinedGateExactKernel(
        const half *headMajorInput, const float *weight,
        const half *combinedGateInput, half *output,
        int seqLen, int paddedSeqLen,
        int gateStride, int gateOffset, int gateHeads, float eps) {
    constexpr int CHANNELS = 128;
    int row = blockIdx.x;
    int lane = threadIdx.x;
    int token = row / gateHeads;
    int head = row - token * gateHeads;
    int batch = token / seqLen;
    int tokenInBatch = token - batch * seqLen;
    headMajorInput +=
        (((size_t)batch * gateHeads + head) * paddedSeqLen +
         tokenInBatch) * CHANNELS;
    combinedGateInput +=
        (size_t)token * gateStride + gateOffset + head * CHANNELS;
    output += (size_t)row * CHANNELS;
    const half2 *input2 =
        reinterpret_cast<const half2 *>(headMajorInput);
    const half2 *gate2 =
        reinterpret_cast<const half2 *>(combinedGateInput);

    float2 value0 = __half22float2(input2[lane]);
    float2 value1 = __half22float2(input2[lane + 32]);
    float sum0 = value0.x * value0.x + value0.y * value0.y;
    float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    float scale = 0.0f;
    if (lane == 0) {
        scale = rsqrtf((sum0 + sum1) / CHANNELS + eps);
    }
    scale = __shfl_sync(0xffffffffu, scale, 0);

    float2 inputValues[2] = {value0, value1};
    half2 gateValues[2] = {gate2[lane], gate2[lane + 32]};
    half2 *output2 = reinterpret_cast<half2 *>(output);
#pragma unroll
    for (int part = 0; part < 2; part++) {
        int index = lane + part * 32;
        float2 value = inputValues[part];
        half gate0In = __low2half(gateValues[part]);
        half gate1In = __high2half(gateValues[part]);
#ifdef CUDA_NO_TENSOR_CORE
        float gate0Float = __half2float(gate0In);
        float gate1Float = __half2float(gate1In);
        half gate0 =
            __float2half(gate0Float / (1.0f + expf(-gate0Float)));
        half gate1 =
            __float2half(gate1Float / (1.0f + expf(-gate1Float)));
#else
        half gate0 =
            __hdiv(gate0In, __hadd(__float2half(1.0f), hexp(-gate0In)));
        half gate1 =
            __hdiv(gate1In, __hadd(__float2half(1.0f), hexp(-gate1In)));
#endif
        half rms0 = __float2half_rn(
            value.x * scale * __ldg(weight + index * 2));
        half rms1 = __float2half_rn(
            value.y * scale * __ldg(weight + index * 2 + 1));
#ifdef CUDA_NO_TENSOR_CORE
        half out0 =
            __float2half(__half2float(rms0) * __half2float(gate0));
        half out1 =
            __float2half(__half2float(rms1) * __half2float(gate1));
#else
        half out0 = __hmul(rms0, gate0);
        half out1 = __hmul(rms1, gate1);
#endif
        output2[index] = __halves2half2(out0, out1);
    }
}

static bool LaunchFastllmRMSNormSiluMulFloat16(
        const half *input, const float *weight, const half *gateInput,
        half *output, int outer, int channels, float eps, int threadCount) {
    if (channels == 128 && threadCount == 32) {
        FastllmRMSNormSiluMulHalf128ExactKernel<<<outer, 32>>>(
            input, weight, gateInput, output, eps);
        return true;
    }
    if (threadCount != 0) {
        return false;
    }
    if (channels < 512) {
        FastllmRMSNormSiluMulHalfKernel<64><<<outer, 64>>>(
            input, weight, gateInput, output, channels, eps);
    } else if (channels < 4096) {
        FastllmRMSNormSiluMulHalfKernel<512><<<outer, 512>>>(
            input, weight, gateInput, output, channels, eps);
    } else {
        FastllmRMSNormSiluMulHalfKernel<1024><<<outer, 1024>>>(
            input, weight, gateInput, output, channels, eps);
    }
    return true;
}

bool FastllmCudaRMSNormSiluMulFloat16WithThreadCount(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &gateInput, fastllm::Data &output,
        float eps, int threadCount) {
    if (input.dataDevice != fastllm::DataDevice::CUDA || gateInput.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (input.dataType != fastllm::DataType::FLOAT16 || gateInput.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 || weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (input.dims.size() == 0 || input.dims != gateInput.dims || output.dims != input.dims ||
        input.strides.empty() || gateInput.strides.empty() || output.strides.empty() ||
        input.strides.back() != 1 || gateInput.strides.back() != 1 || output.strides.back() != 1 ||
        weight.dims.size() != 1 || weight.dims[0] != input.dims.back()) {
        return false;
    }

    int channels = input.dims.back();
    int outer = input.Count(0) / channels;
    const half *cudaInput = (const half *) input.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const half *cudaGateInput = (const half *) gateInput.cudaData;
    half *cudaOutput = (half *) output.cudaData;

    if (!LaunchFastllmRMSNormSiluMulFloat16(
            cudaInput, cudaWeight, cudaGateInput, cudaOutput,
            outer, channels, eps, threadCount)) {
        return false;
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaRMSNormSiluMulFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaRMSNormSiluMulFloat16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &gateInput, fastllm::Data &output, float eps) {
    return FastllmCudaRMSNormSiluMulFloat16WithThreadCount(
        input, weight, gateInput, output, eps,
        !input.dims.empty() && input.dims.back() == 128 ? 32 : 0);
}

bool FastllmCudaRMSNormSiluMulFloat16CombinedGate(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &combinedGateInput,
        int gateOffset, int gateHeads,
        fastllm::Data &output, float eps) {
    auto isDense = [](const fastllm::Data &data) {
        if (data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        combinedGateInput.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        combinedGateInput.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        input.cudaData == nullptr || combinedGateInput.cudaData == nullptr ||
        output.cudaData == nullptr || weight.cudaData == nullptr ||
        !isDense(input) || !isDense(combinedGateInput) ||
        !isDense(output) || input.dims != output.dims ||
        input.dims.back() != 128 || weight.dims.size() != 1 ||
        weight.dims[0] != 128 || gateHeads <= 0 || gateOffset < 0 ||
        !std::isfinite(eps) || eps < 0.0f) {
        return false;
    }

    int gateStride = combinedGateInput.dims.back();
    if (gateOffset > gateStride ||
        (int64_t)gateHeads * 128 > gateStride - gateOffset) {
        return false;
    }
    int64_t inputCount = input.Count(0);
    int64_t combinedCount = combinedGateInput.Count(0);
    if (inputCount <= 0 || inputCount % 128 != 0 ||
        combinedCount <= 0 || combinedCount % gateStride != 0) {
        return false;
    }
    int64_t outer64 = inputCount / 128;
    int64_t tokens64 = combinedCount / gateStride;
    if (outer64 != tokens64 * gateHeads ||
        outer64 > std::numeric_limits<int>::max()) {
        return false;
    }

    FastllmRMSNormSiluMulHalf128CombinedGateExactKernel<<<
        (int)outer64, 32>>>(
        (const half *)input.cudaData,
        (const float *)weight.cudaData,
        (const half *)combinedGateInput.cudaData,
        (half *)output.cudaData,
        gateStride, gateOffset, gateHeads, eps);
    checkCudaErrors(
        "Error: CUDA error in "
        "FastllmCudaRMSNormSiluMulFloat16CombinedGate.",
        cudaGetLastError());
    return true;
}

bool FastllmCudaRMSNormSiluMulFloat16HeadMajorCombinedGate(
        const fastllm::Data &headMajorInput, fastllm::Data &weight,
        const fastllm::Data &combinedGateInput,
        int batch, int seqLen,
        int gateOffset, int gateHeads,
        fastllm::Data &output, float eps) {
    auto isDense = [](const fastllm::Data &data) {
        if (data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };
    if (batch <= 0 || seqLen <= 0 || gateHeads <= 0 ||
        gateOffset < 0 || !std::isfinite(eps) || eps < 0.0f ||
        headMajorInput.dataDevice != fastllm::DataDevice::CUDA ||
        combinedGateInput.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        headMajorInput.dataType != fastllm::DataType::FLOAT16 ||
        combinedGateInput.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        headMajorInput.cudaData == nullptr ||
        combinedGateInput.cudaData == nullptr ||
        weight.cudaData == nullptr ||
        !isDense(headMajorInput) || !isDense(combinedGateInput) ||
        headMajorInput.dims.size() != 4 ||
        headMajorInput.dims[0] != batch ||
        headMajorInput.dims[1] != gateHeads ||
        headMajorInput.dims[2] < seqLen ||
        headMajorInput.dims[3] != 128 ||
        weight.dims.size() != 1 || weight.dims[0] != 128) {
        return false;
    }

    int paddedSeqLen = headMajorInput.dims[2];
    int gateStride = combinedGateInput.dims.back();
    if (gateOffset > gateStride ||
        (int64_t)gateHeads * 128 > gateStride - gateOffset ||
        combinedGateInput.Count(0) !=
            (uint64_t)batch * seqLen * gateStride) {
        return false;
    }
    int64_t outer64 = (int64_t)batch * seqLen * gateHeads;
    if (outer64 <= 0 || outer64 > std::numeric_limits<int>::max()) {
        return false;
    }

    output.dataType = fastllm::DataType::FLOAT16;
    output.dataDevice = headMajorInput.dataDevice;
    output.dataDeviceIds = headMajorInput.dataDeviceIds;
    output.Resize({batch, seqLen, gateHeads, 128});
    output.Allocate(false);
    if (output.cudaData == nullptr || !isDense(output)) {
        return false;
    }

    FastllmRMSNormSiluMulHalf128HeadMajorCombinedGateExactKernel<<<(
        int)outer64, 32>>>(
        (const half *)headMajorInput.cudaData,
        (const float *)weight.cudaData,
        (const half *)combinedGateInput.cudaData,
        (half *)output.cudaData,
        seqLen, paddedSeqLen,
        gateStride, gateOffset, gateHeads, eps);
    checkCudaErrors(
        "Error: CUDA error in "
        "FastllmCudaRMSNormSiluMulFloat16HeadMajorCombinedGate.",
        cudaGetLastError());
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormPartKernel(float *input, float *weight, float *output,
                                          int outer, int channels, int start, int end, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;
    int partChannels = end - start;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS > 0 ? NUM_WARPS : 1];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Copy [0, start) and [end, channels)
    for (int i = tid; i < start; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }
    for (int i = end + tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }

    // Compute sum of squares over [start, end)
    float sum2 = 0.0f;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        float x = input[start + i];
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (THREAD_PER_BLOCK > WARP_SIZE) {
        if (lane_id == 0) {
            warp_sums[warp_id] = sum2;
        }
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                scale = rsqrtf(val / partChannels + eps);
            }
        }
        __syncthreads();
    } else {
        if (tid == 0) {
            scale = rsqrtf(sum2 / partChannels + eps);
        }
        __syncthreads();
    }

    float s = scale;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        output[start + i] = input[start + i] * s * __ldg(&weight[i]);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormPartKernel(half *input, float *weight, half *output,
                                          int outer, int channels, int start, int end, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;
    int partChannels = end - start;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    for (int i = tid; i < start; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }
    for (int i = end + tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }

    float sum2 = 0.0f;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        float x = __half2float(input[start + i]);
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / partChannels + eps);
        }
    }
    __syncthreads();

    float s = scale;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        output[start + i] = __float2half(__half2float(input[start + i]) * s * __ldg(&weight[i]));
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormPartKernel(__nv_bfloat16 *input, float *weight, __nv_bfloat16 *output,
                                          int outer, int channels, int start, int end, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;
    int partChannels = end - start;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    for (int i = tid; i < start; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }
    for (int i = end + tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = input[i];
    }

    float sum2 = 0.0f;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        float x = __bfloat162float(input[start + i]);
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / partChannels + eps);
        }
    }
    __syncthreads();

    float s = scale;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        output[start + i] = __float2bfloat16_rn(__bfloat162float(input[start + i]) * s * __ldg(&weight[i]));
    }
}

bool FastllmCudaRMSNormPart(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps, int start, int end) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int partChannels = end - start;

    if (input.dataType == fastllm::DataType::FLOAT32) {
        if (partChannels < 64) {
            FastllmRMSNormPartKernel<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, start, end, eps);
        } else if (partChannels < 512) {
            FastllmRMSNormPartKernel<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, start, end, eps);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartKernel<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, start, end, eps);
        } else {
            FastllmRMSNormPartKernel<1024> <<< outer, 1024 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, start, end, eps);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        if (partChannels < 512) {
            FastllmRMSNormPartKernel<64> <<< outer, 64 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer, channels, start, end, eps);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartKernel<512> <<< outer, 512 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer, channels, start, end, eps);
        } else {
            FastllmRMSNormPartKernel<1024> <<< outer, 1024 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer, channels, start, end, eps);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        if (partChannels < 512) {
            FastllmRMSNormPartKernel<64> <<< outer, 64 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer, channels, start, end, eps);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartKernel<512> <<< outer, 512 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer, channels, start, end, eps);
        } else {
            FastllmRMSNormPartKernel<1024> <<< outer, 1024 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer, channels, start, end, eps);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmRMSNormPartSum2Kernel(const T *input, float *sumOut,
                                             int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    input = input + o * channels;
    int partChannels = end - start;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS > 0 ? NUM_WARPS : 1];

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float sum2 = 0.0f;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        float x = (float)input[start + i];
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (THREAD_PER_BLOCK > WARP_SIZE) {
        if (lane_id == 0) {
            warp_sums[warp_id] = sum2;
        }
        __syncthreads();
        if (warp_id == 0) {
            float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane_id == 0) {
                sumOut[o] = val;
            }
        }
    } else {
        if (tid == 0) {
            sumOut[o] = sum2;
        }
    }
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, half>(const half *input, float *sumOut,
                                                       int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const half *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = __half2float(base[i]);
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, float>(const float *input, float *sumOut,
                                                        int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const float *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = base[i];
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, __nv_bfloat16>(const __nv_bfloat16 *input, float *sumOut,
                                                                int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const __nv_bfloat16 *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = __bfloat162float(base[i]);
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

bool FastllmCudaRMSNormPartSum2(const fastllm::Data &input, float *sumOut, int start, int end) {
    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int partChannels = end - start;
    if (outer <= 0 || partChannels <= 0) {
        return true;
    }

    void *cudaInput = (void*) FastllmCudaPrepareInput(input);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        const float *p = (const float*) cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, float> <<< outer, 1 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, float> <<< outer, 64 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, float> <<< outer, 512 >>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, float> <<< outer, 1024 >>>(p, sumOut, outer, channels, start, end);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        const half *p = (const half*) cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, half> <<< outer, 1 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, half> <<< outer, 64 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, half> <<< outer, 512 >>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, half> <<< outer, 1024 >>>(p, sumOut, outer, channels, start, end);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        const __nv_bfloat16 *p = (const __nv_bfloat16*) cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, __nv_bfloat16> <<< outer, 1 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, __nv_bfloat16> <<< outer, 64 >>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, __nv_bfloat16> <<< outer, 512 >>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, __nv_bfloat16> <<< outer, 1024 >>>(p, sumOut, outer, channels, start, end);
        }
    } else {
        printf("Error: FastllmCudaRMSNormPartSum2 unsupported dtype %d.\n", (int)input.dataType);
        return false;
    }

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmRMSNormPartApplyKernel(const T *input, const float *weight, T *output,
                                              const float *sumIn,
                                              int outer, int channels, int start, int end,
                                              int partChannelsGlobal, float eps) {
    int o = blockIdx.x;
    const T *inRow = input + o * channels;
    T *outRow = output + o * channels;
    int partChannels = end - start;

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(sumIn[o] / partChannelsGlobal + eps);
    }
    __syncthreads();

    if (input != output) {
        for (int i = threadIdx.x; i < start; i += THREAD_PER_BLOCK) {
            outRow[i] = inRow[i];
        }
        for (int i = end + threadIdx.x; i < channels; i += THREAD_PER_BLOCK) {
            outRow[i] = inRow[i];
        }
    }

    float s = scale;
    for (int i = threadIdx.x; i < partChannels; i += THREAD_PER_BLOCK) {
        outRow[start + i] = (T)((float)inRow[start + i] * s * __ldg(&weight[i]));
    }
}

bool FastllmCudaRMSNormPartApply(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output,
                                 const float *sumIn, float eps, int start, int end, int partChannelsGlobal) {
    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int partChannels = end - start;
    if (outer <= 0) {
        return true;
    }

    void *cudaInput = (void*) FastllmCudaPrepareInput(input);
    void *cudaOutput = (void*) FastllmCudaPrepareInput(output);

    auto pickThreads = [](int n) -> int {
        if (n < 64) return 64;
        if (n < 512) return 64;
        if (n < 4096) return 512;
        return 1024;
    };
    int threads = pickThreads(partChannels);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        const float *p = (const float*) cudaInput;
        float *o = (float*) cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, float> <<< outer, 64 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, float> <<< outer, 512 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, float> <<< outer, 1024 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        const half *p = (const half*) cudaInput;
        half *o = (half*) cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, half> <<< outer, 64 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, half> <<< outer, 512 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, half> <<< outer, 1024 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        const __nv_bfloat16 *p = (const __nv_bfloat16*) cudaInput;
        __nv_bfloat16 *o = (__nv_bfloat16*) cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, __nv_bfloat16> <<< outer, 64 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, __nv_bfloat16> <<< outer, 512 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, __nv_bfloat16> <<< outer, 1024 >>>(p, (const float*) weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else {
        printf("Error: FastllmCudaRMSNormPartApply unsupported dtype %d.\n", (int)input.dataType);
        return false;
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];

    if (inner == 1) {
        if (gamma.dataType != fastllm::DataType::FLOAT32 || beta.dataType != fastllm::DataType::FLOAT32) {
            printf("layernorm datatype error.\n");
            exit(0);    
        } else if (input.dataType == fastllm::DataType::FLOAT32) {
            if (channels < 64) {
                FastllmLayerNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, cudaOutput,
                                                                outer, channels);
            } else if (channels < 512) {
                FastllmLayerNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, cudaOutput,
                                                                outer, channels);
            } else {
                FastllmLayerNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) gamma.cudaData,
                                                                    (float *) beta.cudaData, cudaOutput,
                                                                    outer, channels);
            }
        } else if (input.dataType == fastllm::DataType::FLOAT16) {
            if (channels < 64) {
                FastllmLayerNormKernelInner1<1> <<< outer, 1 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, (half*)cudaOutput,
                                                                outer, channels);
            } else if (channels < 512) {
                FastllmLayerNormKernelInner1<64> <<< outer, 64 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, (half*)cudaOutput,
                                                                outer, channels);
            } else {
                FastllmLayerNormKernelInner1<512> <<< outer, 512 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                    (float *) beta.cudaData, (half*)cudaOutput,
                                                                    outer, channels);
            }
        } else {
            printf("layernorm datatype error.\n");
            exit(0);    
        }
    } else {
        printf("layernorm error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

/*#ifndef USE_ROCM
// 自定义函子，用于处理每一行的 TopK 操作  
struct TopKFunctor {
    float* cudaInput;        // 指向原始输入数据的设备指针  
    float* cudaOutput;       // 指向输出数据的设备指针  
    int channels;
    int topk;

    // 构造函数  
    TopKFunctor(float* cudaInput, float* cudaOutput, int channels, int topk)
        : cudaInput(cudaInput), cudaOutput(cudaOutput), channels(channels), topk(topk) {}

    // thrust::for_each 会为每个行索引 i 调用这个操作符  
    // __host__ __device__ 使得函子可以在主机和设备上被调用（Thrust 要求）  
    __host__ __device__
    void operator()(int i) const {
        thrust::device_ptr<float> d_input(cudaInput);
        thrust::device_ptr<float> d_output(cudaOutput);

        // 当前行的起始位置  
        thrust::device_ptr<float> row_start = d_input + i * channels;
        thrust::device_ptr<float> row_end = row_start + channels;
        
        // 创建索引序列 [0, 1, 2, ..., channels-1]
        thrust::device_vector<int> indices(channels);
        thrust::sequence(indices.begin(), indices.end());
        
        // 使用zip迭代器将值和索引组合在一起  
        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(row_start, indices.begin()));
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(row_end, indices.end()));
        
        // 按值降序排序  
        thrust::sort(begin, end, 
            thrust::greater<thrust::tuple<float, int>>());
        
        // 复制前topk个结果到输出  
        for (int k = 0; k < topk; ++k) {
            d_output[i * topk * 2 + k * 2] = indices[k];     // 索引  
            d_output[i * topk * 2 + k * 2 + 1] = row_start[k]; // 值  
        }
    }
};

// 主函数/调用部分  
void topk_parallel_thrust(float* d_input, float* d_output, int outer, int channels, int topk) {
    // 创建函子实例  
    TopKFunctor functor(d_input, d_output, channels, topk);

    // 使用 thrust::for_each 和 counting_iterator 来并行处理每一行  
    thrust::for_each(
        thrust::counting_iterator<int>(0),      // 起始迭代器 (0)
        thrust::counting_iterator<int>(outer),  // 结束迭代器 (outer)
        functor                                 // 应用于每个元素的函子  
    );
}
#endif */

bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk) {
    if (topk > 50) {
        printf("topk: unsupport topk > 50.");
        exit(0);
    }

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

// #ifdef USE_ROCM
    if (topk == 1) {
        FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
    } else {
        FastllmLayerNormKernelTopK <64, 50> <<< outer, 64 >>> (cudaInput, cudaOutput, topk, channels);
    }
/*
#else
    if (outer > 4 || topk == 1) {
        if (topk == 1) {
            FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
        } else {
            FastllmLayerNormKernelTopK <64, 50> <<< outer, 64 >>> (cudaInput, cudaOutput, topk, channels);
        }    
    } else {
        TopKFunctor functor(cudaInput, cudaOutput, channels, topk);
        for (int i = 0; i < outer; ++i) {
            functor(i);
        }
    }
#endif */
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__device__ __forceinline__ float FastllmRouterLogitToFloat(float value) {
    return value;
}

__device__ __forceinline__ float FastllmRouterLogitToFloat(half value) {
    return __half2float(value);
}

__device__ __forceinline__ float FastllmRouterLogitToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

#ifndef USE_ROCM
template <typename T>
__device__ __forceinline__ void FastllmRouterLoad8(
        const T *source, float (&values)[8]) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        values[i] = FastllmRouterLogitToFloat(source[i]);
    }
}

template <>
__device__ __forceinline__ void FastllmRouterLoad8<float>(
        const float *source, float (&values)[8]) {
    float4 lo = reinterpret_cast<const float4*>(source)[0];
    float4 hi = reinterpret_cast<const float4*>(source)[1];
    values[0] = lo.x;
    values[1] = lo.y;
    values[2] = lo.z;
    values[3] = lo.w;
    values[4] = hi.x;
    values[5] = hi.y;
    values[6] = hi.z;
    values[7] = hi.w;
}

template <>
__device__ __forceinline__ void FastllmRouterLoad8<half>(
        const half *source, float (&values)[8]) {
    uint4 packed = *reinterpret_cast<const uint4*>(source);
    half2 pair0 = *reinterpret_cast<half2*>(&packed.x);
    half2 pair1 = *reinterpret_cast<half2*>(&packed.y);
    half2 pair2 = *reinterpret_cast<half2*>(&packed.z);
    half2 pair3 = *reinterpret_cast<half2*>(&packed.w);
    float2 value0 = __half22float2(pair0);
    float2 value1 = __half22float2(pair1);
    float2 value2 = __half22float2(pair2);
    float2 value3 = __half22float2(pair3);
    values[0] = value0.x;
    values[1] = value0.y;
    values[2] = value1.x;
    values[3] = value1.y;
    values[4] = value2.x;
    values[5] = value2.y;
    values[6] = value3.x;
    values[7] = value3.y;
}

template <>
__device__ __forceinline__ void FastllmRouterLoad8<__nv_bfloat16>(
        const __nv_bfloat16 *source, float (&values)[8]) {
    uint4 packed = *reinterpret_cast<const uint4*>(source);
    __nv_bfloat162 pair0 = *reinterpret_cast<__nv_bfloat162*>(&packed.x);
    __nv_bfloat162 pair1 = *reinterpret_cast<__nv_bfloat162*>(&packed.y);
    __nv_bfloat162 pair2 = *reinterpret_cast<__nv_bfloat162*>(&packed.z);
    __nv_bfloat162 pair3 = *reinterpret_cast<__nv_bfloat162*>(&packed.w);
    float2 value0 = __bfloat1622float2(pair0);
    float2 value1 = __bfloat1622float2(pair1);
    float2 value2 = __bfloat1622float2(pair2);
    float2 value3 = __bfloat1622float2(pair3);
    values[0] = value0.x;
    values[1] = value0.y;
    values[2] = value1.x;
    values[3] = value1.y;
    values[4] = value2.x;
    values[5] = value2.y;
    values[6] = value3.x;
    values[7] = value3.y;
}
#endif

__device__ __forceinline__ float FastllmRouterBiasToFloat(
        const void *bias, int biasType, int expert) {
    if (biasType == 1) {
        return __half2float(((const half*)bias)[expert]);
    }
    if (biasType == 2) {
        return __bfloat162float(((const __nv_bfloat16*)bias)[expert]);
    }
    return ((const float*)bias)[expert];
}

struct FastllmRouterCandidate {
    float key;
    int id;
};

__device__ __forceinline__ bool FastllmRouterCandidateBetter(
        const FastllmRouterCandidate &a, const FastllmRouterCandidate &b) {
    if (a.id < 0) {
        return false;
    }
    if (b.id < 0) {
        return true;
    }
    return a.key != b.key ? a.key > b.key : a.id < b.id;
}

__device__ __forceinline__ void FastllmRouterCandidateCompareSwap(
        FastllmRouterCandidate &a, FastllmRouterCandidate &b) {
    if (FastllmRouterCandidateBetter(b, a)) {
        FastllmRouterCandidate temp = a;
        a = b;
        b = temp;
    }
}

#ifndef USE_ROCM
// Laguna's sigmoid router has a fixed 256-expert, top-10 shape. Keep the
// transformed row and the TopK scan in one warp so the normal decode path does
// not need shared-memory staging or block-wide synchronization. Exact ties are
// uncommon for model logits; when they occur, reproduce the legacy 64-thread
// merge order below so expert ordering remains compatible.
template <typename T>
__global__ void FastllmFusedSigmoidSelectExpert256Top10Kernel(
        const T *logits, const void *bias, int32_t *index, float *score,
        int tokens, int hasBias, int biasType, int needNorm, float routeScale) {
    constexpr int EXPERTS = 256;
    constexpr int TOPK = 10;
    constexpr int LEGACY_THREADS = 64;
    constexpr float NEG_INF = -1.0e30f;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;

    __shared__ float selectedKeys[TOPK];
    __shared__ int selectedIds[TOPK];
    __shared__ float selectedProb[TOPK];
    __shared__ float legacyKeys[LEGACY_THREADS][TOPK];
    __shared__ float legacyIds[LEGACY_THREADS][TOPK];

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int lane = threadIdx.x;
    int firstExpert = lane * 8;
    const T *tokenLogits = logits + (size_t)token * EXPERTS;

    float rawLogits[8];
    float probabilities[8];
    float choiceKeys[8];
    float correctionBias[8];
    FastllmRouterLoad8(tokenLogits + firstExpert, rawLogits);
    if (hasBias) {
        if (biasType == 1) {
            FastllmRouterLoad8((const half*)bias + firstExpert, correctionBias);
        } else if (biasType == 2) {
            FastllmRouterLoad8((const __nv_bfloat16*)bias + firstExpert, correctionBias);
        } else {
            FastllmRouterLoad8((const float*)bias + firstExpert, correctionBias);
        }
    } else {
#pragma unroll
        for (int part = 0; part < 8; part++) {
            correctionBias[part] = 0.0f;
        }
    }
#pragma unroll
    for (int part = 0; part < 8; part++) {
        float probability = 1.0f / (1.0f + __expf(-rawLogits[part]));
        probabilities[part] = probability;
        choiceKeys[part] = probability + correctionBias[part];
    }

#pragma unroll
    for (int rank = 0; rank < TOPK; rank++) {
        FastllmRouterCandidate best = {choiceKeys[0], firstExpert};
#pragma unroll
        for (int part = 1; part < 8; part++) {
            FastllmRouterCandidate candidate = {choiceKeys[part], firstExpert + part};
            if (FastllmRouterCandidateBetter(candidate, best)) {
                best = candidate;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            FastllmRouterCandidate other;
            other.key = __shfl_xor_sync(FULL_WARP_MASK, best.key, mask);
            other.id = __shfl_xor_sync(FULL_WARP_MASK, best.id, mask);
            if (FastllmRouterCandidateBetter(other, best)) {
                best = other;
            }
        }

        int owner = best.id >> 3;
        int ownerPart = best.id & 7;
        float probability = probabilities[ownerPart];
        probability = __shfl_sync(FULL_WARP_MASK, probability, owner);
        if (lane == 0) {
            selectedKeys[rank] = best.key;
            selectedIds[rank] = best.id;
            selectedProb[rank] = probability;
        }

        if (lane == owner) {
#pragma unroll
            for (int part = 0; part < 8; part++) {
                if (part == ownerPart) {
                    choiceKeys[part] = NEG_INF;
                }
            }
        }
    }

    __syncwarp(FULL_WARP_MASK);
    float cutoff = lane == 0 ? selectedKeys[TOPK - 1] : 0.0f;
    cutoff = __shfl_sync(FULL_WARP_MASK, cutoff, 0);
    int useLegacyMerge = 0;
#pragma unroll
    for (int part = 0; part < 8; part++) {
        useLegacyMerge |= choiceKeys[part] == cutoff;
    }
    useLegacyMerge = __any_sync(FULL_WARP_MASK, useLegacyMerge);
    if (lane == 0) {
#pragma unroll
        for (int i = 1; i < TOPK; i++) {
            useLegacyMerge |= selectedKeys[i] == selectedKeys[i - 1];
        }
    }
    useLegacyMerge = __shfl_sync(FULL_WARP_MASK, useLegacyMerge, 0);

    if (!useLegacyMerge) {
        if (lane == 0) {
            float selectedSum = 1.0f;
            if (needNorm) {
                selectedSum = 0.0f;
#pragma unroll
                for (int rank = 0; rank < TOPK; rank++) {
                    selectedSum += selectedProb[rank];
                }
            }
            int32_t *tokenIndex = index + (size_t)token * TOPK;
            float *tokenScore = score + (size_t)token * TOPK;
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                tokenIndex[rank] = selectedIds[rank];
                tokenScore[rank] = selectedProb[rank] / selectedSum * routeScale;
            }
        }
        return;
    }

    // The fast scan marks winners as consumed in choiceKeys. Restore those
    // entries before constructing the exact legacy lists.
#pragma unroll
    for (int rank = 0; rank < TOPK; rank++) {
        int expert = lane == 0 ? selectedIds[rank] : 0;
        float key = lane == 0 ? selectedKeys[rank] : 0.0f;
        expert = __shfl_sync(FULL_WARP_MASK, expert, 0);
        key = __shfl_sync(FULL_WARP_MASK, key, 0);
        int owner = expert >> 3;
        int ownerPart = expert & 7;
        if (lane == owner) {
#pragma unroll
            for (int part = 0; part < 8; part++) {
                if (part == ownerPart) {
                    choiceKeys[part] = key;
                }
            }
        }
    }

    // Populate the same 64 virtual lists as FastllmSelectExpertKernel. One
    // physical warp handles two virtual threads per lane; this path is only
    // taken for exact ranking-key ties.
#pragma unroll
    for (int half = 0; half < 2; half++) {
        int virtualTid = lane + half * 32;
#pragma unroll
        for (int rank = 0; rank < TOPK; rank++) {
            legacyKeys[virtualTid][rank] = -1.0e100;
            legacyIds[virtualTid][rank] = -1.0f;
        }
#pragma unroll
        for (int part = 0; part < 4; part++) {
            int expert = virtualTid + part * LEGACY_THREADS;
            int owner = expert >> 3;
            int ownerPart = expert & 7;
            float key = NEG_INF;
#pragma unroll
            for (int sourcePart = 0; sourcePart < 8; sourcePart++) {
                float candidate = __shfl_sync(
                    FULL_WARP_MASK, choiceKeys[sourcePart], owner);
                if (sourcePart == ownerPart) {
                    key = candidate;
                }
            }
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                if (key > legacyKeys[virtualTid][rank]) {
#pragma unroll
                    for (int shift = TOPK - 1; shift > rank; shift--) {
                        legacyKeys[virtualTid][shift] = legacyKeys[virtualTid][shift - 1];
                        legacyIds[virtualTid][shift] = legacyIds[virtualTid][shift - 1];
                    }
                    legacyKeys[virtualTid][rank] = key;
                    legacyIds[virtualTid][rank] = (float)expert;
                    break;
                }
            }
        }
    }
    __syncwarp(FULL_WARP_MASK);

    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            int pos0 = 0;
            int pos1 = 0;
            while (pos0 + pos1 < TOPK) {
                if (legacyKeys[lane][pos0] > legacyKeys[lane + stride][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            for (int pos = TOPK - 1; pos >= 0; pos--) {
                if (pos1 < 0 ||
                    (pos0 >= 0 && legacyKeys[lane][pos0] < legacyKeys[lane + stride][pos1])) {
                    legacyKeys[lane][pos] = legacyKeys[lane][pos0];
                    legacyIds[lane][pos] = legacyIds[lane][pos0];
                    pos0--;
                } else {
                    legacyKeys[lane][pos] = legacyKeys[lane + stride][pos1];
                    legacyIds[lane][pos] = legacyIds[lane + stride][pos1];
                    pos1--;
                }
            }
        }
        __syncwarp(FULL_WARP_MASK);
    }

#pragma unroll
    for (int rank = 0; rank < TOPK; rank++) {
        int expert = lane == 0 ? (int)legacyIds[0][rank] : 0;
        expert = __shfl_sync(FULL_WARP_MASK, expert, 0);
        int owner = expert >> 3;
        int ownerPart = expert & 7;
        float probability = probabilities[ownerPart];
        probability = __shfl_sync(FULL_WARP_MASK, probability, owner);
        if (lane == 0) {
            selectedIds[rank] = expert;
            selectedProb[rank] = probability;
        }
    }

    if (lane == 0) {
        float selectedSum = 1.0f;
        if (needNorm) {
            selectedSum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                selectedSum += selectedProb[rank];
            }
        }
        int32_t *tokenIndex = index + (size_t)token * TOPK;
        float *tokenScore = score + (size_t)token * TOPK;
#pragma unroll
        for (int rank = 0; rank < TOPK; rank++) {
            tokenIndex[rank] = selectedIds[rank];
            tokenScore[rank] = selectedProb[rank] / selectedSum * routeScale;
        }
    }
}
#endif

__device__ __forceinline__ void FastllmWriteNormalizedLogitsTop8(
        float key, int expert, int lane,
        int32_t *tokenIndex, float *tokenScore, float routeScale) {
    constexpr int TOPK = 8;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
    float maxKey = __shfl_sync(FULL_WARP_MASK, key, 0);
    float selected = lane < TOPK ? expf(key - maxKey) : 0.0f;
    float selectedSum = selected;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        selectedSum +=
            __shfl_down_sync(FULL_WARP_MASK, selectedSum, offset);
    }
    selectedSum =
        __shfl_sync(FULL_WARP_MASK, selectedSum, 0);
    if (lane < TOPK) {
        tokenIndex[lane] = expert;
        tokenScore[lane] = selected / selectedSum * routeScale;
    }
}

// Reproduce the original 64-thread SelectExpert merge tree with one physical
// warp.  This is only needed when equal routing keys make a fixed total-order
// comparator insufficient: the legacy tie order depends on how candidates are
// distributed across the merge tree.
template <typename T>
__device__ __forceinline__ void FastllmWriteLegacyNormalizedLogitsTop8(
        const T *tokenLogits, int lane,
        float *legacyKeys, int *legacyIds,
        int32_t *tokenIndex, float *tokenScore, float routeScale) {
    constexpr int THREADS = 64;
    constexpr int TOPK = 8;
    constexpr float NEG_INF = -1.0e30f;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;

    // Each physical lane initializes the two logical selector threads that
    // occupied the same lane in the original two-warp implementation.
#pragma unroll
    for (int logicalThread = lane; logicalThread < THREADS;
         logicalThread += 32) {
#pragma unroll
        for (int rank = 0; rank < TOPK; rank++) {
            legacyKeys[logicalThread * TOPK + rank] = -1.0e100;
            legacyIds[logicalThread * TOPK + rank] = -1;
        }
#pragma unroll
        for (int part = 0; part < 4; part++) {
            int expert = logicalThread + part * THREADS;
            float key = FastllmRouterLogitToFloat(tokenLogits[expert]);
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                int offset = logicalThread * TOPK + rank;
                if (key > legacyKeys[offset]) {
#pragma unroll
                    for (int shift = TOPK - 1; shift > rank; shift--) {
                        int dst = logicalThread * TOPK + shift;
                        legacyKeys[dst] = legacyKeys[dst - 1];
                        legacyIds[dst] = legacyIds[dst - 1];
                    }
                    legacyKeys[offset] = key;
                    legacyIds[offset] = expert;
                    break;
                }
            }
        }
    }
    __syncwarp(FULL_WARP_MASK);

    int row0 = lane * TOPK;
    int row1 = (lane + 32) * TOPK;
    int pos0 = 0;
    int pos1 = 0;
    while (pos0 + pos1 < TOPK) {
        if (legacyKeys[row0 + pos0] > legacyKeys[row1 + pos1]) {
            pos0++;
        } else {
            pos1++;
        }
    }
    pos0--;
    pos1--;
    for (int pos = TOPK - 1; pos >= 0; pos--) {
        if (pos1 < 0 ||
            (pos0 >= 0 && legacyKeys[row0 + pos0] < legacyKeys[row1 + pos1])) {
            legacyKeys[row0 + pos] = legacyKeys[row0 + pos0];
            legacyIds[row0 + pos] = legacyIds[row0 + pos0];
            pos0--;
        } else {
            legacyKeys[row0 + pos] = legacyKeys[row1 + pos1];
            legacyIds[row0 + pos] = legacyIds[row1 + pos1];
            pos1--;
        }
    }
    __syncwarp(FULL_WARP_MASK);

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            row0 = lane * TOPK;
            row1 = (lane + stride) * TOPK;
            pos0 = 0;
            pos1 = 0;
            while (pos0 + pos1 < TOPK) {
                if (legacyKeys[row0 + pos0] > legacyKeys[row1 + pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            for (int pos = TOPK - 1; pos >= 0; pos--) {
                if (pos1 < 0 ||
                    (pos0 >= 0 &&
                     legacyKeys[row0 + pos0] < legacyKeys[row1 + pos1])) {
                    legacyKeys[row0 + pos] = legacyKeys[row0 + pos0];
                    legacyIds[row0 + pos] = legacyIds[row0 + pos0];
                    pos0--;
                } else {
                    legacyKeys[row0 + pos] = legacyKeys[row1 + pos1];
                    legacyIds[row0 + pos] = legacyIds[row1 + pos1];
                    pos1--;
                }
            }
        }
        __syncwarp(FULL_WARP_MASK);
    }

    int expert = lane < TOPK ? legacyIds[lane] : -1;
    float key = lane < TOPK ?
        FastllmRouterLogitToFloat(tokenLogits[expert]) : NEG_INF;
    FastllmWriteNormalizedLogitsTop8(
        key, expert, lane, tokenIndex, tokenScore, routeScale);
}

// Pack a finite FP16 logit and a deterministic equal-key priority into one
// sortable integer. Ambiguous top-k ties are detected and sent through the
// exact legacy merge path below.
__device__ __forceinline__ uint32_t FastllmPackFp16RouterCandidate(
        half key, int expert) {
    uint32_t bits = __half_as_ushort(key);
    // The float comparator treats +0 and -0 as equal.
    if ((bits & 0x7fffu) == 0) {
        bits = 0;
    }
    uint32_t ordered =
        (bits & 0x8000u) ? (~bits & 0xffffu) : (bits ^ 0x8000u);
    int selectorThread = expert & 63;
    int reversedThread =
        __brev((unsigned int)selectorThread) >> 26;
    int selectorPart = expert >> 6;
    uint32_t tiePriority =
        (uint32_t)((reversedThread << 2) | (3 - selectorPart));
    return (ordered << 16) | (tiePriority << 8) |
           (uint32_t)expert;
}

__device__ __forceinline__ void FastllmPackedRouterCompareSwap(
        uint32_t &a, uint32_t &b) {
    if (b > a) {
        uint32_t temp = a;
        a = b;
        b = temp;
    }
}

// FP16 no-bias Qwen3.5 router specialization.  One warp owns one token and
// ranks all 256 logits in registers.  Distinct top-k values avoid the generic
// two-warp merge; ambiguous ties use the exact legacy merge in the same block.
// The kernel has no cross-CTA state, so it remains safe under CUDA Graph replay
// and concurrent tokens.
__global__ __launch_bounds__(32)
void FastllmFusedSoftmaxSelectExpertFp16Packed256Top8Kernel(
        const half *logits, int32_t *index, float *score,
        int tokens, float routeScale) {
    constexpr int EXPERTS = 256;
    constexpr int TOPK = 8;
    constexpr float NEG_INF = -1.0e30f;
    constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;

    __shared__ float legacyKeys[64][TOPK];
    __shared__ int legacyIds[64][TOPK];

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int lane = threadIdx.x;
    const half *tokenLogits = logits + (size_t)token * EXPERTS;

    uint32_t local[8];
#pragma unroll
    for (int part = 0; part < 8; part++) {
        int expert = lane + part * 32;
        local[part] =
            FastllmPackFp16RouterCandidate(tokenLogits[expert], expert);
    }

    // Sort each lane's eight candidates in descending order.
    FastllmPackedRouterCompareSwap(local[0], local[1]);
    FastllmPackedRouterCompareSwap(local[2], local[3]);
    FastllmPackedRouterCompareSwap(local[4], local[5]);
    FastllmPackedRouterCompareSwap(local[6], local[7]);
    FastllmPackedRouterCompareSwap(local[0], local[2]);
    FastllmPackedRouterCompareSwap(local[1], local[3]);
    FastllmPackedRouterCompareSwap(local[4], local[6]);
    FastllmPackedRouterCompareSwap(local[5], local[7]);
    FastllmPackedRouterCompareSwap(local[1], local[2]);
    FastllmPackedRouterCompareSwap(local[5], local[6]);
    FastllmPackedRouterCompareSwap(local[0], local[4]);
    FastllmPackedRouterCompareSwap(local[3], local[7]);
    FastllmPackedRouterCompareSwap(local[1], local[5]);
    FastllmPackedRouterCompareSwap(local[2], local[6]);
    FastllmPackedRouterCompareSwap(local[1], local[4]);
    FastllmPackedRouterCompareSwap(local[3], local[6]);
    FastllmPackedRouterCompareSwap(local[2], local[4]);
    FastllmPackedRouterCompareSwap(local[3], local[5]);
    FastllmPackedRouterCompareSwap(local[3], local[4]);

    int selectedExpert = -1;
    float selectedKey = NEG_INF;
#pragma unroll
    for (int rank = 0; rank < TOPK + 1; rank++) {
        uint32_t best = local[0];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            uint32_t other =
                __shfl_down_sync(FULL_WARP_MASK, best, offset);
            if (lane + offset < 32 && other > best) {
                best = other;
            }
        }
        uint32_t winner =
            __shfl_sync(FULL_WARP_MASK, best, 0);
        int winnerExpert = (int)(winner & 0xffu);
        if (lane == rank) {
            selectedExpert = winnerExpert;
            selectedKey =
                FastllmRouterLogitToFloat(tokenLogits[winnerExpert]);
        }
        if (local[0] == winner) {
#pragma unroll
            for (int part = 0; part < 7; part++) {
                local[part] = local[part + 1];
            }
            local[7] = 0;
        }
    }

    bool ambiguous = false;
#pragma unroll
    for (int i = 1; i < TOPK; i++) {
        float keyI = __shfl_sync(FULL_WARP_MASK, selectedKey, i);
#pragma unroll
        for (int j = 0; j < i; j++) {
            float keyJ = __shfl_sync(FULL_WARP_MASK, selectedKey, j);
            ambiguous |= keyI == keyJ;
        }
    }
    ambiguous |=
        __shfl_sync(FULL_WARP_MASK, selectedKey, TOPK) ==
        __shfl_sync(FULL_WARP_MASK, selectedKey, TOPK - 1);

    int32_t *tokenIndex = index + (size_t)token * TOPK;
    float *tokenScore = score + (size_t)token * TOPK;
    if (ambiguous) {
        FastllmWriteLegacyNormalizedLogitsTop8(
            tokenLogits, lane, &legacyKeys[0][0], &legacyIds[0][0],
            tokenIndex, tokenScore, routeScale);
    } else {
        FastllmWriteNormalizedLogitsTop8(
            selectedKey, selectedExpert, lane,
            tokenIndex, tokenScore, routeScale);
    }
}

// Qwen3.5 batch-1 decode specialization for 256 experts and top-8.  The
// correction-bias path retains the full shared-memory softmax.  Without a bias,
// ranking can use logits directly; when selected scores are normalized, the
// global softmax denominator cancels, so only the winning eight need expf.
// Equal-key ordering follows the original 64-thread SelectExpert merge tree.
template <typename T, int BLOCK_THREADS, bool NORMALIZE_SELECTED_LOGITS>
__global__ __launch_bounds__(BLOCK_THREADS)
void FastllmFusedSoftmaxSelectExpert256Top8Kernel(
        const T *logits, const float *bias, int32_t *index, float *score,
        int tokens, int hasBias, int needNorm, float routeScale) {
    constexpr int THREADS = 64;
    constexpr int EXPERTS = 256;
    constexpr int TOPK = 8;
    constexpr int CANDIDATES = TOPK + 1;
    constexpr float NEG_INF = -1.0e30f;
    static_assert(BLOCK_THREADS >= THREADS &&
                      BLOCK_THREADS <= EXPERTS &&
                      EXPERTS % BLOCK_THREADS == 0,
                  "router block width must divide 256 and cover 64 selectors");

    __shared__ float reduceData[THREADS];
    __shared__ float probabilities[EXPERTS];
    __shared__ float maxValue;
    __shared__ float sumValue;
    __shared__ float warpTopKeys[2][CANDIDATES];
    __shared__ int warpTopIds[2][CANDIDATES];
    __shared__ float standardTopKeys[CANDIDATES];
    __shared__ int standardTopIds[CANDIDATES];
    __shared__ float legacyKeys[THREADS][TOPK];
    __shared__ int legacyIds[THREADS][TOPK];
    __shared__ int useLegacyMerge;

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const T *tokenLogits = logits + (size_t)token * EXPERTS;

    // Use all 256 threads for the expensive exponential while preserving the
    // old 64-thread softmax reduction tree exactly.  Each of the first 64
    // threads still combines experts tid + {0, 64, 128, 192} in the original
    // order, so scores and close correction-bias boundaries remain unchanged.
#pragma unroll
    for (int expert = tid; expert < EXPERTS; expert += BLOCK_THREADS) {
        probabilities[expert] =
            FastllmRouterLogitToFloat(tokenLogits[expert]);
    }
    __syncthreads();

    float localMax = NEG_INF;
    if constexpr (!NORMALIZE_SELECTED_LOGITS) {
        if (tid < THREADS) {
#pragma unroll
            for (int part = 0; part < 4; part++) {
                int expert = tid + part * THREADS;
                localMax = fmaxf(localMax, probabilities[expert]);
            }
            reduceData[tid] = localMax;
        }
        __syncthreads();
        if (warp == 0) {
            float reducedMax = fmaxf(localMax, reduceData[lane + 32]);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other =
                    __shfl_down_sync(0xffffffffu, reducedMax, offset);
                if (lane < offset) {
                    reducedMax = fmaxf(reducedMax, other);
                }
            }
            if (lane == 0) {
                maxValue = reducedMax;
            }
        }
        __syncthreads();

#pragma unroll
        for (int expert = tid; expert < EXPERTS;
             expert += BLOCK_THREADS) {
            probabilities[expert] =
                expf(probabilities[expert] - maxValue);
        }
        __syncthreads();

        float localSum = 0.0f;
        if (tid < THREADS) {
#pragma unroll
            for (int part = 0; part < 4; part++) {
                localSum += probabilities[tid + part * THREADS];
            }
            reduceData[tid] = localSum;
        }
        __syncthreads();
        if (warp == 0) {
            float reducedSum = localSum + reduceData[lane + 32];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other =
                    __shfl_down_sync(0xffffffffu, reducedSum, offset);
                if (lane < offset) {
                    reducedSum += other;
                }
            }
            if (lane == 0) {
                sumValue =
                    fabsf(reducedSum) < 1.0e-6f ? 1.0e-4f : reducedSum;
            }
        }
        __syncthreads();

#pragma unroll
        for (int expert = tid; expert < EXPERTS;
             expert += BLOCK_THREADS) {
            probabilities[expert] /= sumValue;
        }
        __syncthreads();
    }

    FastllmRouterCandidate local[4];
    if (tid < THREADS) {
#pragma unroll
        for (int part = 0; part < 4; part++) {
            int expert = tid + part * THREADS;
            if constexpr (NORMALIZE_SELECTED_LOGITS) {
                local[part].key = probabilities[expert];
            } else {
                local[part].key =
                    probabilities[expert] +
                    (hasBias ? bias[expert] : 0.0f);
            }
            local[part].id = expert;
        }

        FastllmRouterCandidateCompareSwap(local[0], local[1]);
        FastllmRouterCandidateCompareSwap(local[2], local[3]);
        FastllmRouterCandidateCompareSwap(local[0], local[2]);
        FastllmRouterCandidateCompareSwap(local[1], local[3]);
        FastllmRouterCandidateCompareSwap(local[1], local[2]);

        unsigned int warpMask = 0xffffffffu;
        int cursor = 0;
#pragma unroll
        for (int rank = 0; rank < CANDIDATES; rank++) {
            FastllmRouterCandidate best = cursor < 4 ?
                local[cursor] :
                FastllmRouterCandidate{NEG_INF, -1};
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                FastllmRouterCandidate other;
                other.key =
                    __shfl_down_sync(warpMask, best.key, offset);
                other.id =
                    __shfl_down_sync(warpMask, best.id, offset);
                if (lane + offset < 32 &&
                    FastllmRouterCandidateBetter(other, best)) {
                    best = other;
                }
            }
            int winner = __shfl_sync(warpMask, best.id, 0);
            if (lane == 0) {
                warpTopKeys[warp][rank] = best.key;
                warpTopIds[warp][rank] = best.id;
            }
            if (cursor < 4 && local[cursor].id == winner) {
                cursor++;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        int positions[2] = {0, 0};
#pragma unroll
        for (int rank = 0; rank < CANDIDATES; rank++) {
            FastllmRouterCandidate first = positions[0] < CANDIDATES ?
                FastllmRouterCandidate{warpTopKeys[0][positions[0]], warpTopIds[0][positions[0]]} :
                FastllmRouterCandidate{NEG_INF, -1};
            FastllmRouterCandidate second = positions[1] < CANDIDATES ?
                FastllmRouterCandidate{warpTopKeys[1][positions[1]], warpTopIds[1][positions[1]]} :
                FastllmRouterCandidate{NEG_INF, -1};
            bool takeFirst = FastllmRouterCandidateBetter(first, second);
            FastllmRouterCandidate chosen = takeFirst ? first : second;
            positions[takeFirst ? 0 : 1]++;
            standardTopKeys[rank] = chosen.key;
            standardTopIds[rank] = chosen.id;
        }
        if constexpr (NORMALIZE_SELECTED_LOGITS) {
            bool ambiguous = false;
#pragma unroll
            for (int i = 1; i < TOPK; i++) {
#pragma unroll
                for (int j = 0; j < i; j++) {
                    ambiguous |= standardTopKeys[i] == standardTopKeys[j];
                }
            }
            useLegacyMerge = ambiguous ||
                standardTopKeys[TOPK] == standardTopKeys[TOPK - 1];
        } else {
            useLegacyMerge = 0;
        }
    }
    __syncthreads();

    if constexpr (NORMALIZE_SELECTED_LOGITS) {
        if (useLegacyMerge) {
            if (warp == 0) {
                FastllmWriteLegacyNormalizedLogitsTop8(
                    tokenLogits, lane,
                    &legacyKeys[0][0], &legacyIds[0][0],
                    index + (size_t)token * TOPK,
                    score + (size_t)token * TOPK,
                    routeScale);
            }
        } else if (warp == 0) {
            float key =
                lane < TOPK ? standardTopKeys[lane] : NEG_INF;
            int expert =
                lane < TOPK ? standardTopIds[lane] : -1;
            FastllmWriteNormalizedLogitsTop8(
                key, expert, lane,
                index + (size_t)token * TOPK,
                score + (size_t)token * TOPK,
                routeScale);
        }
    } else if (tid == 0) {
        float selectedSum = 1.0f;
        if (needNorm) {
            selectedSum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                selectedSum += probabilities[standardTopIds[rank]];
            }
        }
        int32_t *tokenIndex = index + (size_t)token * TOPK;
        float *tokenScore = score + (size_t)token * TOPK;
#pragma unroll
        for (int rank = 0; rank < TOPK; rank++) {
            int expert = standardTopIds[rank];
            tokenIndex[rank] = expert;
            tokenScore[rank] = probabilities[expert] / selectedSum * routeScale;
        }
    }
}

// General 256-expert router used by both the softmax/top-8 and
// sigmoid/top-10 entry points.  Keep this path feature-complete for Laguna and
// for bias types that the Qwen3.5 no-bias specializations do not handle.
template <typename T, int ROUTER_TOPK, bool ROUTER_SIGMOID>
__global__ void FastllmFusedSelectExpert256Kernel(
        const T *logits, const void *bias, int32_t *index, float *score,
        int tokens, int hasBias, int biasType, int needNorm, float routeScale) {
    constexpr int THREADS = 64;
    constexpr int EXPERTS = 256;
    constexpr int TOPK = ROUTER_TOPK;
    constexpr int CANDIDATES = TOPK + 1;
    constexpr float NEG_INF = -1.0e30f;

    __shared__ float reduceData[THREADS];
    __shared__ float probabilities[EXPERTS];
    __shared__ float maxValue;
    __shared__ float sumValue;
    __shared__ float selectKeys[THREADS][TOPK];
    __shared__ float selectIds[THREADS][TOPK];
    __shared__ float warpTopKeys[2][CANDIDATES];
    __shared__ int warpTopIds[2][CANDIDATES];
    __shared__ float standardTopKeys[CANDIDATES];
    __shared__ int standardTopIds[CANDIDATES];
    __shared__ int useLegacyMerge;

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const T *tokenLogits = logits + (size_t)token * EXPERTS;

    // Use all 256 threads for the score transform. The softmax specialization
    // preserves the old 64-thread reduction tree exactly.
    probabilities[tid] = FastllmRouterLogitToFloat(tokenLogits[tid]);
    __syncthreads();

    if constexpr (ROUTER_SIGMOID) {
        probabilities[tid] = 1.0f / (1.0f + expf(-probabilities[tid]));
    } else {
        float localMax = NEG_INF;
        if (tid < THREADS) {
#pragma unroll
            for (int part = 0; part < 4; part++) {
                int expert = tid + part * THREADS;
                localMax = fmaxf(localMax, probabilities[expert]);
            }
            reduceData[tid] = localMax;
        }
        __syncthreads();
        if (warp == 0) {
            float reducedMax = fmaxf(localMax, reduceData[lane + 32]);
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffffu, reducedMax, offset);
                if (lane < offset) {
                    reducedMax = fmaxf(reducedMax, other);
                }
            }
            if (lane == 0) {
                maxValue = reducedMax;
            }
        }
        __syncthreads();

        probabilities[tid] = expf(probabilities[tid] - maxValue);
        __syncthreads();

        float localSum = 0.0f;
        if (tid < THREADS) {
#pragma unroll
            for (int part = 0; part < 4; part++) {
                localSum += probabilities[tid + part * THREADS];
            }
            reduceData[tid] = localSum;
        }
        __syncthreads();
        if (warp == 0) {
            float reducedSum = localSum + reduceData[lane + 32];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xffffffffu, reducedSum, offset);
                if (lane < offset) {
                    reducedSum += other;
                }
            }
            if (lane == 0) {
                sumValue = fabsf(reducedSum) < 1.0e-6f ? 1.0e-4f : reducedSum;
            }
        }
        __syncthreads();

        probabilities[tid] /= sumValue;
    }
    __syncthreads();

    FastllmRouterCandidate local[4];
    if (tid < THREADS) {
#pragma unroll
        for (int part = 0; part < 4; part++) {
            int expert = tid + part * THREADS;
            local[part].key = probabilities[expert] +
                (hasBias ? FastllmRouterBiasToFloat(bias, biasType, expert) : 0.0f);
            local[part].id = expert;
        }

        FastllmRouterCandidateCompareSwap(local[0], local[1]);
        FastllmRouterCandidateCompareSwap(local[2], local[3]);
        FastllmRouterCandidateCompareSwap(local[0], local[2]);
        FastllmRouterCandidateCompareSwap(local[1], local[3]);
        FastllmRouterCandidateCompareSwap(local[1], local[2]);

        unsigned int warpMask = 0xffffffffu;
        int cursor = 0;
#pragma unroll
        for (int rank = 0; rank < CANDIDATES; rank++) {
            FastllmRouterCandidate best = cursor < 4 ?
                local[cursor] : FastllmRouterCandidate{NEG_INF, -1};
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                FastllmRouterCandidate other;
                other.key = __shfl_down_sync(warpMask, best.key, offset);
                other.id = __shfl_down_sync(warpMask, best.id, offset);
                if (lane + offset < 32 && FastllmRouterCandidateBetter(other, best)) {
                    best = other;
                }
            }
            int winner = __shfl_sync(warpMask, best.id, 0);
            if (lane == 0) {
                warpTopKeys[warp][rank] = best.key;
                warpTopIds[warp][rank] = best.id;
            }
            if (cursor < 4 && local[cursor].id == winner) {
                cursor++;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        int positions[2] = {0, 0};
#pragma unroll
        for (int rank = 0; rank < CANDIDATES; rank++) {
            FastllmRouterCandidate first = positions[0] < CANDIDATES ?
                FastllmRouterCandidate{warpTopKeys[0][positions[0]], warpTopIds[0][positions[0]]} :
                FastllmRouterCandidate{NEG_INF, -1};
            FastllmRouterCandidate second = positions[1] < CANDIDATES ?
                FastllmRouterCandidate{warpTopKeys[1][positions[1]], warpTopIds[1][positions[1]]} :
                FastllmRouterCandidate{NEG_INF, -1};
            bool takeFirst = FastllmRouterCandidateBetter(first, second);
            FastllmRouterCandidate chosen = takeFirst ? first : second;
            positions[takeFirst ? 0 : 1]++;
            standardTopKeys[rank] = chosen.key;
            standardTopIds[rank] = chosen.id;
        }

        bool ambiguous = false;
#pragma unroll
        for (int i = 1; i < TOPK; i++) {
#pragma unroll
            for (int j = 0; j < i; j++) {
                ambiguous |= standardTopKeys[i] == standardTopKeys[j];
            }
        }
        useLegacyMerge = ambiguous || standardTopKeys[TOPK] == standardTopKeys[TOPK - 1];
    }
    __syncthreads();

    if (useLegacyMerge) {
        if (tid < THREADS) {
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                selectKeys[tid][rank] = -1.0e100;
                selectIds[tid][rank] = -1.0f;
            }
#pragma unroll
            for (int part = 0; part < 4; part++) {
                int expert = tid + part * THREADS;
                float key = probabilities[expert] +
                    (hasBias ? FastllmRouterBiasToFloat(bias, biasType, expert) : 0.0f);
#pragma unroll
                for (int rank = 0; rank < TOPK; rank++) {
                    if (key > selectKeys[tid][rank]) {
#pragma unroll
                        for (int shift = TOPK - 1; shift > rank; shift--) {
                            selectKeys[tid][shift] = selectKeys[tid][shift - 1];
                            selectIds[tid][shift] = selectIds[tid][shift - 1];
                        }
                        selectKeys[tid][rank] = key;
                        selectIds[tid][rank] = (float)expert;
                        break;
                    }
                }
            }
        }
        __syncthreads();

        if (tid < 32) {
            int pos0 = 0;
            int pos1 = 0;
            while (pos0 + pos1 < TOPK) {
                if (selectKeys[tid][pos0] > selectKeys[tid + 32][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            for (int pos = TOPK - 1; pos >= 0; pos--) {
                if (pos1 < 0 ||
                    (pos0 >= 0 && selectKeys[tid][pos0] < selectKeys[tid + 32][pos1])) {
                    selectKeys[tid][pos] = selectKeys[tid][pos0];
                    selectIds[tid][pos] = selectIds[tid][pos0];
                    pos0--;
                } else {
                    selectKeys[tid][pos] = selectKeys[tid + 32][pos1];
                    selectIds[tid][pos] = selectIds[tid + 32][pos1];
                    pos1--;
                }
            }
        }
        __syncthreads();

        if (warp == 0) {
            constexpr unsigned int FULL_WARP_MASK = 0xffffffffu;
            for (int stride = 16; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    int pos0 = 0;
                    int pos1 = 0;
                    while (pos0 + pos1 < TOPK) {
                        if (selectKeys[tid][pos0] > selectKeys[tid + stride][pos1]) {
                            pos0++;
                        } else {
                            pos1++;
                        }
                    }
                    pos0--;
                    pos1--;
                    for (int pos = TOPK - 1; pos >= 0; pos--) {
                        if (pos1 < 0 ||
                            (pos0 >= 0 && selectKeys[tid][pos0] < selectKeys[tid + stride][pos1])) {
                            selectKeys[tid][pos] = selectKeys[tid][pos0];
                            selectIds[tid][pos] = selectIds[tid][pos0];
                            pos0--;
                        } else {
                            selectKeys[tid][pos] = selectKeys[tid + stride][pos1];
                            selectIds[tid][pos] = selectIds[tid + stride][pos1];
                            pos1--;
                        }
                    }
                }
                __syncwarp(FULL_WARP_MASK);
            }

            if (lane == 0) {
                float selectedSum = 1.0f;
                if (needNorm) {
                    selectedSum = 0.0f;
#pragma unroll
                    for (int rank = 0; rank < TOPK; rank++) {
                        selectedSum += probabilities[(int)selectIds[0][rank]];
                    }
                }
                int32_t *tokenIndex = index + (size_t)token * TOPK;
                float *tokenScore = score + (size_t)token * TOPK;
#pragma unroll
                for (int rank = 0; rank < TOPK; rank++) {
                    int expert = (int)selectIds[0][rank];
                    tokenIndex[rank] = expert;
                    tokenScore[rank] = probabilities[expert] / selectedSum * routeScale;
                }
            }
        }
    } else if (tid == 0) {
        float selectedSum = 1.0f;
        if (needNorm) {
            selectedSum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < TOPK; rank++) {
                selectedSum += probabilities[standardTopIds[rank]];
            }
        }
        int32_t *tokenIndex = index + (size_t)token * TOPK;
        float *tokenScore = score + (size_t)token * TOPK;
#pragma unroll
        for (int rank = 0; rank < TOPK; rank++) {
            int expert = standardTopIds[rank];
            tokenIndex[rank] = expert;
            tokenScore[rank] = probabilities[expert] / selectedSum * routeScale;
        }
    }
}

template <int ROUTER_TOPK, bool ROUTER_SIGMOID>
static bool FastllmCudaFusedSelectExpert256(
        const fastllm::Data &logits, const fastllm::Data *gateBias,
        fastllm::Data &index, fastllm::Data &score,
        bool needNorm, float routeScale) {
    bool hasBias = gateBias != nullptr && !gateBias->dims.empty();

    const void *cudaLogits = FastllmCudaPrepareInput(logits);
    const void *cudaBias = hasBias ? FastllmCudaPrepareInput(*gateBias) : nullptr;
    int32_t *cudaIndex = (int32_t*)FastllmCudaPrepareOutput(index);
    float *cudaScore = (float*)FastllmCudaPrepareOutput(score);
    if (cudaLogits == nullptr || cudaIndex == nullptr || cudaScore == nullptr || (hasBias && cudaBias == nullptr)) {
        FastllmCudaFinishInput(logits, (void*)cudaLogits);
        if (hasBias) {
            FastllmCudaFinishInput(*gateBias, (void*)cudaBias);
        }
        return false;
    }

    int tokens = logits.Count(0) / 256;
    int biasType = !hasBias || gateBias->dataType == fastllm::DataType::FLOAT32 ? 0 :
                   (gateBias->dataType == fastllm::DataType::FLOAT16 ? 1 : 2);
#ifndef USE_ROCM
    if constexpr (ROUTER_SIGMOID) {
        if (logits.dataType == fastllm::DataType::FLOAT16) {
            FastllmFusedSigmoidSelectExpert256Top10Kernel<half><<<tokens, 32>>>(
                (const half*)cudaLogits, cudaBias, cudaIndex, cudaScore,
                tokens, hasBias ? 1 : 0, biasType,
                needNorm ? 1 : 0, routeScale);
        } else if (logits.dataType == fastllm::DataType::BFLOAT16) {
            FastllmFusedSigmoidSelectExpert256Top10Kernel<__nv_bfloat16>
                <<<tokens, 32>>>(
                    (const __nv_bfloat16*)cudaLogits, cudaBias,
                    cudaIndex, cudaScore, tokens, hasBias ? 1 : 0,
                    biasType, needNorm ? 1 : 0, routeScale);
        } else {
            FastllmFusedSigmoidSelectExpert256Top10Kernel<float><<<tokens, 32>>>(
                (const float*)cudaLogits, cudaBias, cudaIndex, cudaScore,
                tokens, hasBias ? 1 : 0, biasType,
                needNorm ? 1 : 0, routeScale);
        }
    } else
#endif
    {
    bool useSelectedLogitFastPath =
        !ROUTER_SIGMOID && ROUTER_TOPK == 8 && !hasBias && needNorm;
    if (logits.dataType == fastllm::DataType::FLOAT16) {
        if (useSelectedLogitFastPath) {
            FastllmFusedSoftmaxSelectExpertFp16Packed256Top8Kernel
                <<<tokens, 32>>>(
                    (const half*)cudaLogits, cudaIndex, cudaScore,
                    tokens, routeScale);
        } else {
            FastllmFusedSelectExpert256Kernel<
                half, ROUTER_TOPK, ROUTER_SIGMOID><<<tokens, 256>>>(
                    (const half*)cudaLogits, cudaBias, cudaIndex, cudaScore,
                    tokens, hasBias ? 1 : 0, biasType,
                    needNorm ? 1 : 0, routeScale);
        }
    } else if (logits.dataType == fastllm::DataType::BFLOAT16) {
        if (useSelectedLogitFastPath) {
            FastllmFusedSoftmaxSelectExpert256Top8Kernel<
                __nv_bfloat16, 64, true><<<tokens, 64>>>(
                    (const __nv_bfloat16*)cudaLogits, nullptr,
                    cudaIndex, cudaScore, tokens, 0, 1, routeScale);
        } else {
            FastllmFusedSelectExpert256Kernel<
                __nv_bfloat16, ROUTER_TOPK, ROUTER_SIGMOID><<<tokens, 256>>>(
                    (const __nv_bfloat16*)cudaLogits, cudaBias,
                    cudaIndex, cudaScore, tokens, hasBias ? 1 : 0, biasType,
                    needNorm ? 1 : 0, routeScale);
        }
    } else {
        if (useSelectedLogitFastPath) {
            FastllmFusedSoftmaxSelectExpert256Top8Kernel<float, 64, true>
                <<<tokens, 64>>>(
                    (const float*)cudaLogits, nullptr,
                    cudaIndex, cudaScore, tokens, 0, 1, routeScale);
        } else {
            FastllmFusedSelectExpert256Kernel<
                float, ROUTER_TOPK, ROUTER_SIGMOID><<<tokens, 256>>>(
                    (const float*)cudaLogits, cudaBias, cudaIndex, cudaScore,
                    tokens, hasBias ? 1 : 0, biasType,
                    needNorm ? 1 : 0, routeScale);
        }
    }
    }

    cudaError_t state = cudaGetLastError();
    bool success = state == cudaSuccess;
    if (!success) {
        checkCudaErrors("Error: fused SelectExpert launch failed!", state);
    }
    FastllmCudaFinishInput(logits, (void*)cudaLogits);
    if (hasBias) {
        FastllmCudaFinishInput(*gateBias, (void*)cudaBias);
    }
    FastllmCudaFinishOutput(index, cudaIndex);
    FastllmCudaFinishOutput(score, cudaScore);
    return success;
}

bool FastllmCudaFusedSoftmaxSelectExpert(
        const fastllm::Data &logits, const fastllm::Data *gateBias,
        fastllm::Data &index, fastllm::Data &score,
        int topk, bool needNorm, float routeScale) {
    if (topk != 8 || logits.dims.empty() || logits.dims.back() != 256 || logits.Count(0) == 0 ||
        (logits.dataType != fastllm::DataType::FLOAT16 &&
         logits.dataType != fastllm::DataType::BFLOAT16 &&
         logits.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    bool hasBias = gateBias != nullptr && !gateBias->dims.empty();
    if (hasBias && (gateBias->Count(0) != 256 ||
                    (gateBias->dataType != fastllm::DataType::FLOAT32 &&
                     gateBias->dataType != fastllm::DataType::FLOAT16 &&
                     gateBias->dataType != fastllm::DataType::BFLOAT16))) {
        return false;
    }
    return FastllmCudaFusedSelectExpert256<8, false>(
        logits, gateBias, index, score, needNorm, routeScale);
}

bool FastllmCudaFusedSigmoidSelectExpert(
        const fastllm::Data &logits, const fastllm::Data *gateBias,
        fastllm::Data &index, fastllm::Data &score,
        int topk, bool needNorm, float routeScale) {
    if (topk != 10 || logits.dims.empty() || logits.dims.back() != 256 || logits.Count(0) == 0 ||
        (logits.dataType != fastllm::DataType::FLOAT16 &&
         logits.dataType != fastllm::DataType::BFLOAT16 &&
         logits.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    bool hasBias = gateBias != nullptr && !gateBias->dims.empty();
    if (hasBias && (gateBias->Count(0) != 256 ||
                    (gateBias->dataType != fastllm::DataType::FLOAT32 &&
                     gateBias->dataType != fastllm::DataType::FLOAT16 &&
                     gateBias->dataType != fastllm::DataType::BFLOAT16))) {
        return false;
    }
    return FastllmCudaFusedSelectExpert256<10, true>(
        logits, gateBias, index, score, needNorm, routeScale);
}

// CUDA kernel for SelectExpert
template <int THREAD_PER_BLOCK, int MAXK>
__global__ void FastllmSelectExpertKernel(float *logits, float *bias, int32_t *index, float *score, 
    int n, int numExperts, int topk, int hasBias, bool needNorm, float routeScale) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];
    
    int tokenIdx = blockIdx.x;
    float *inputData = logits + tokenIdx * numExperts;
    int32_t *outputIndex = index + tokenIdx * topk;
    float *outputScore = score + tokenIdx * topk;
    
    int tid = threadIdx.x;
    
    // Initialize
    for (int i = 0; i < topk; i++) {
        maxData[tid][i] = -1e100;
        idData[tid][i] = -1;
    }
    
    // Find topk experts
    for (int j = tid; j < numExperts; j += THREAD_PER_BLOCK) {
        float cur = inputData[j];
        if (hasBias) {
            cur += bias[j];
        }
        // A non-finite router value used to leave idData at -1.  The writeback
        // below would then read inputData[-1], turning a numerical issue into an
        // illegal CUDA access and poisoning the entire device context.  Rank a
        // non-finite value last while retaining a valid expert id instead.
        if (!isfinite(cur)) {
            cur = -FLT_MAX;
        }
        
        for (int l = 0; l < topk; l++) {
            if (cur > maxData[tid][l]) {
                for (int x = topk - 1; x > l; x--) {
                    maxData[tid][x] = maxData[tid][x - 1];
                    idData[tid][x] = idData[tid][x - 1];
                }
                maxData[tid][l] = cur;
                idData[tid][l] = j;
                break;
            }
        }
    }
    __syncthreads();
    
    // Merge results from all threads
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < topk) {
                if (maxData[tid][pos0] > maxData[tid + s][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            int pos = topk - 1;
            while (pos >= 0) {
                if (pos1 < 0 || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (tid == 0) {
        // Calculate sum for normalization
        float sum = 1.0f;
        if (needNorm) {
            sum = 0.0f;
            for (int i = 0; i < topk; i++) {
                int expertIdx = idData[0][i];
                if (expertIdx >= 0 && expertIdx < numExperts &&
                    isfinite(inputData[expertIdx])) {
                    sum += inputData[expertIdx];
                }
            }
            if (!isfinite(sum) || fabsf(sum) < 1.0e-20f) {
                sum = 1.0f;
            }
        }
        
        // Write index and score
        for (int i = 0; i < topk; i++) {
            int expertIdx = idData[0][i];
            if (expertIdx < 0 || expertIdx >= numExperts) {
                expertIdx = 0;
            }
            float expertScore = inputData[expertIdx];
            if (!isfinite(expertScore)) {
                expertScore = 0.0f;
            }
            outputIndex[i] = expertIdx;
            outputScore[i] = expertScore / sum * routeScale;
        }
    }
}

bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale) {
    if (logits.dims.empty() || logits.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int dimsLen = logits.dims.size();
    int numExperts = logits.dims[dimsLen - 1]; // number of experts
    int n = numExperts > 0 ? logits.Count(0) / numExperts : 0; // number of tokens
    if (topk <= 0 || topk > 50 || topk > numExperts || n <= 0) {
        printf("SelectExpert: unsupported shape or topk (tokens=%d, experts=%d, topk=%d), "
               "falling back to CPU implementation.\n", n, numExperts, topk);
        return false; // 返回 false 表示不支持，应该回退到 CPU
    }
    
    float *cudaLogits = (float *) FastllmCudaPrepareInput(logits);
    float *cudaBias = nullptr;
    int hasBias = 0;
    if (gateBias != nullptr && gateBias->dims.size() > 0) {
        cudaBias = (float *) FastllmCudaPrepareInput(*gateBias);
        hasBias = 1;
    }
    int32_t *cudaIndex = (int32_t *) FastllmCudaPrepareOutput(index);
    float *cudaScore = (float *) FastllmCudaPrepareOutput(score);
    if (cudaLogits == nullptr || cudaIndex == nullptr || cudaScore == nullptr ||
        (hasBias && cudaBias == nullptr)) {
        FastllmCudaFinishInput(logits, cudaLogits);
        if (hasBias) {
            FastllmCudaFinishInput(*gateBias, cudaBias);
        }
        return false;
    }

    // Use 64 threads to stay within shared memory limit (64*50*4*2 = 25KB < 48KB)
#ifdef USE_ROCM
    FastllmSelectExpertKernel<64, 50> <<< n, 64 >>> 
        (cudaLogits, cudaBias, cudaIndex, cudaScore, n, numExperts, topk, hasBias, needNorm, routeScale);
#else
    FastllmSelectExpertKernel<64, 50> <<< n, 64 >>> 
        (cudaLogits, cudaBias, cudaIndex, cudaScore, n, numExperts, topk, hasBias, needNorm, routeScale);
#endif
    
    FastllmCudaFinishInput(logits, cudaLogits);
    if (gateBias != nullptr && gateBias->dims.size() > 0) {
        FastllmCudaFinishInput(*gateBias, cudaBias);
    }
    FastllmCudaFinishOutput(index, cudaIndex);
    FastllmCudaFinishOutput(score, cudaScore);
    return true;
}

__global__ void FastllmCudaMaskAndRemapExpertsKernel(int32_t *index, float *score,
                                                     int total, int expertStart, int expertEnd) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }
    int expert = index[tid];
    if (expert >= expertStart && expert < expertEnd) {
        index[tid] = expert - expertStart;
    } else {
        // All fused local-expert backends treat a negative id as an inactive
        // route.  Keeping expert 0 here would still run its full gate/up and
        // down projections before the zero score discarded the result.
        index[tid] = -1;
        score[tid] = 0.0f;
    }
}

bool FastllmCudaMaskAndRemapExpertsForLocalRange(fastllm::Data &index, fastllm::Data &score,
                                                 int expertStart, int expertEnd) {
    if (expertStart < 0 || expertStart >= expertEnd ||
        index.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        index.dataType != fastllm::DataType::INT32 ||
        score.dataType != fastllm::DataType::FLOAT32 ||
        index.cudaData == nullptr || score.cudaData == nullptr ||
        index.Count(0) != score.Count(0)) {
        return false;
    }
    int total = index.Count(0);
    if (total <= 0) {
        return true;
    }
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmCudaMaskAndRemapExpertsKernel <<< blocks, threads, 0, cudaStreamPerThread >>> (
        (int32_t*)index.cudaData, (float*)score.cudaData, total, expertStart, expertEnd);
    DeviceSync();
    return true;
}

bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }
    int len = input.Count(0);
    size_t tempBytes = (size_t)len * input.unitSize;
    bool permuteFastPath =
        axis == std::vector <int> {1, 0, 2} ||
        axis == std::vector <int> {0, 2, 1} ||
        axis == std::vector <int> {2, 0, 1, 3} ||
        axis == std::vector <int> {1, 2, 0, 3} ||
        axis == std::vector <int> {0, 2, 1, 3};
    size_t alignedTempBytes = FastllmCudaAlignBytes(tempBytes, 256);
    size_t axisTempBytes = permuteFastPath ? 0 : FastllmCudaAlignBytes(axis.size() * 3 * sizeof(int), 256);
    size_t tempBufferBytes = 0;
    bool tempOwn = false;
    uint8_t *tempData = (uint8_t *)FastllmBorrowCudaTempBuffer(
        alignedTempBytes + axisTempBytes, &tempBufferBytes, &tempOwn);
    if (tempData == nullptr || tempBufferBytes < alignedTempBytes + axisTempBytes) {
        printf("FastllmCudaPermute: failed to borrow CUDA temp buffer.\n");
        fflush(stdout);
        return false;
    }
    cudaMemcpy(tempData, input.cudaData, len * input.unitSize, cudaMemcpyDeviceToDevice);

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }
    if (axis == std::vector <int> {1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {1, 2, 0, 3}) {
        int n = input.dims[0];
        int m = input.dims[1] * input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {0, 2, 1}) {
        int batch = input.dims[0];
        int n = input.dims[1];
        int m = input.dims[2];
        if (!FastllmLaunchTransposeLastTwo(
                input.cudaData, tempData, batch, n, m,
                input.unitSize, 0)) {
            FastllmTransposeLastTwoByBatchKernel<256><<<batch * n * m, 256>>>(
                (uint8_t*)input.cudaData, (uint8_t*)tempData,
                batch, n, m, input.unitSize);
        }
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {0, 2, 1, 3}) {
        int batch = input.dims[0];
        int n = input.dims[1];
        int m = input.dims[2];
        int rowBytes = input.dims[3] * input.unitSize;
        FastllmTransposeLastTwoByBatchKernel<256>
            <<<batch * n * m, 256, 0, cudaStreamPerThread>>>(
                (uint8_t*)input.cudaData, (const uint8_t*)tempData,
                batch, n, m, rowBytes);
        input.Resize(new_dims);
    } else {
        std::vector<int> temp;
        int len = input.Count(0);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(axis[i]);
        }
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }
        input.Resize(new_dims);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }

        int *cudaTemp = (int *)(tempData + alignedTempBytes);
        cudaMemcpy(cudaTemp, temp.data(), temp.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threadPerBlock = std::min(256, len);
        if (input.unitSize == 4) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (float *) input.cudaData,(float *)tempData, cudaTemp,(int) axis.size(), len);
        } else if (input.unitSize == 2) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (uint16_t *) input.cudaData,(uint16_t *)tempData, cudaTemp,(int) axis.size(), len);
        } else if (input.unitSize == 1) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (uint8_t *) input.cudaData,(uint8_t *)tempData, cudaTemp,(int) axis.size(), len);
        }

    }

    DeviceSync();
    FastllmReleaseCudaTempBuffer(tempData, tempOwn);
    return true;
}

bool FastllmCudaPermuteTo(const fastllm::Data &input, fastllm::Data &output,
                          const std::vector<int> &axis) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        axis.size() != input.dims.size() ||
        input.dataType != output.dataType ||
        input.Count(0) != output.Count(0)) {
        return false;
    }

    if (axis == std::vector<int>{0, 2, 1} && input.dims.size() == 3) {
        int batch = input.dims[0];
        int n = input.dims[1];
        int m = input.dims[2];
        if (!FastllmLaunchTransposeLastTwo(
                output.cudaData, input.cudaData, batch, n, m,
                input.unitSize, cudaStreamPerThread)) {
            return false;
        }
    } else if (axis == std::vector<int>{0, 2, 1, 3} &&
               input.dims.size() == 4) {
        int batch = input.dims[0];
        int n = input.dims[1];
        int m = input.dims[2];
        int rowBytes = input.dims[3] * input.unitSize;
        FastllmTransposeLastTwoByBatchKernel<256><<<batch * n * m, 256,
                                                   0, cudaStreamPerThread>>>(
            (uint8_t*)output.cudaData, (const uint8_t*)input.cudaData,
            batch, n, m, rowBytes);
    } else {
        return false;
    }
    DeviceSync();
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmFloatToHalf(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((float*)a, (half*)b, len);
    DeviceSync();
    return true;
}

bool FastllmHalfToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)a, (float*)b, len);
    DeviceSync();
    return true;
}

bool FastllmBF16ToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaBF162FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint16_t*)a, (float*)b, len);
    DeviceSync();
    return true;
}

bool FastllmFloatToBF16(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaFloat2Bf16Kernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((float*)a, (__nv_bfloat16*)b, len);
    DeviceSync();
    return true;
}

bool FastllmBF16ToHalf(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaBF162HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint16_t*)a, (half*)b, len);
    DeviceSync();
    return true;
}

bool FastllmHalfToBF16(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaHalf2BF16Kernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)a, (__nv_bfloat16*)b, len);
    DeviceSync();
    return true;
}

bool FastllmCudaEmbedding(const fastllm::Data &input, const fastllm::Data &weight,fastllm::Data &output) {
    int vocabSize = weight.dims[0], embSize = weight.dims[1];
    uint64_t inputLen = input.Count(0);

    float *inputData = (float*)input.cudaData;
    float *dstOutputData = (float*)output.cudaData;

    if (weight.dataType == fastllm::DataType::FLOAT32) {
        float *outputData = (float *) dstOutputData;
        float *weightData = (float *) weight.cudaData;
        if (!FastllmLaunchFloatEmbeddingVector(
                inputData, weightData, outputData, inputLen, embSize)) {
            FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(
                inputData, weightData, outputData, embSize);
        }
    } else if (weight.dataType == fastllm::DataType::FLOAT16) {
        half *outputData = (half *) dstOutputData;
        half *weightData = (half *) weight.cudaData;
        if (!FastllmLaunchFloatEmbeddingVector(
                inputData, weightData, outputData, inputLen, embSize)) {
            FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(
                inputData, weightData, outputData, embSize);
        }
    } else if (weight.dataType == fastllm::DataType::BFLOAT16) {
        float *outputData = (float *) dstOutputData;
        __nv_bfloat16 *weightData = (__nv_bfloat16 *) weight.cudaData;
        FastllmCudaBFloat16EmbeddingToFloatKernel<128><<<inputLen, 128>>>(
            inputData, weightData, outputData, embSize);
    } else {
        
    }

    DeviceSync();
    return true;
}

bool FastllmCudaEmbeddingDirect(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output) {
    int vocabSize = weight.dims[0], embSize = weight.dims[1];
    uint64_t inputLen = input.Count(0);

    float *inputData = (float*)input.cudaData;

    if (weight.dataType == fastllm::DataType::FLOAT32) {
        float *outputData = (float *) output.cudaData;
        float *weightData = (float *) weight.cudaData;
        if (!FastllmLaunchFloatEmbeddingVector(
                inputData, weightData, outputData, inputLen, embSize)) {
            FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(
                inputData, weightData, outputData, embSize);
        }
    } else if (weight.dataType == fastllm::DataType::FLOAT16) {
        half *outputData = (half *) output.cudaData;
        half *weightData = (half *) weight.cudaData;
        if (!FastllmLaunchFloatEmbeddingVector(
                inputData, weightData, outputData, inputLen, embSize)) {
            FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(
                inputData, weightData, outputData, embSize);
        }
    } else if (weight.dataType == fastllm::DataType::BFLOAT16) {
        __nv_bfloat16 *outputData = (__nv_bfloat16 *) output.cudaData;
        __nv_bfloat16 *weightData = (__nv_bfloat16 *) weight.cudaData;
        if (!FastllmLaunchFloatEmbeddingVector(
                inputData, weightData, outputData, inputLen, embSize)) {
            FastllmCudaFloatEmbeddingKernel<128><<<inputLen, 128>>>(
                inputData, weightData, outputData, embSize);
        }
    }

    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

    if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        k, n, m, &alpha,
                                        cudaInput1, input1Stride, input1Spatial,
                                        cudaInput0, input0Stride, input0Spatial,
                                        &beta,
                                        cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT16 && input1.dataType == fastllm::DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m, &h_alpha,
                (half*)cudaInput1, input1Stride, input1Spatial,
                (half*)cudaInput0, input0Stride, input0Spatial,
                &h_beta,
                (half*)cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::BFLOAT16 &&
               input1.dataType == fastllm::DataType::BFLOAT16) {
        status = cublasGemmStridedBatchedEx(
                fastllmCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m, &alpha,
                cudaInput1, CUDA_R_16BF, input1Stride, input1Spatial,
                cudaInput0, CUDA_R_16BF, input0Stride, input0Spatial,
                &beta, cudaOutput, CUDA_R_16BF, k, k * n, batch,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        auto runHalfGemm = [&]() {
            size_t tempInput0Bytes = input0.Count(0) * sizeof(half);
            size_t tempOutputBytes = output.Count(0) * sizeof(half);
            size_t tempInput0AlignedBytes = FastllmCudaAlignBytes(tempInput0Bytes, 256);
            size_t needBytes = tempInput0AlignedBytes + tempOutputBytes;
            size_t workspaceBytes = 0;
            bool workspaceOwn = false;
            uint8_t *workspace = (uint8_t*)FastllmBorrowCudaTempBuffer(needBytes, &workspaceBytes, &workspaceOwn);
            bool useWorkspace = workspace != nullptr && workspaceBytes >= needBytes;
            half *tempInput0 = nullptr;
            half *tempOutput = nullptr;
            if (useWorkspace) {
                tempInput0 = (half*)workspace;
                tempOutput = (half*)(workspace + tempInput0AlignedBytes);
            } else {
                FastllmReleaseCudaTempBuffer(workspace, workspaceOwn);
                tempInput0 = (half*)FastllmCudaMalloc(tempInput0Bytes);
                tempOutput = (half*)FastllmCudaMalloc(tempOutputBytes);
                if (tempInput0 == nullptr || tempOutput == nullptr) {
                    FastllmCudaFree(tempInput0);
                    FastllmCudaFree(tempOutput);
                    return CUBLAS_STATUS_ALLOC_FAILED;
                }
            }
            FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

            half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
            cublasStatus_t halfStatus = cublasHgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    k, n, m, &h_alpha,
                    (half*)cudaInput1, input1Stride, input1Spatial,
                    (half*)tempInput0, input0Stride, input0Spatial,
                    &h_beta,
                    (half*)tempOutput, k, k * n, batch);
            if (halfStatus == CUBLAS_STATUS_SUCCESS) {
                FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
            }
            if (useWorkspace) {
                FastllmReleaseCudaTempBuffer(workspace, workspaceOwn);
            } else {
                FastllmCudaFree(tempInput0);
                FastllmCudaFree(tempOutput);
            }
            return halfStatus;
        };

        size_t tempInput1Bytes = input1.Count(0) * sizeof(float);
        size_t workspaceBytes = 0;
        bool workspaceOwn = false;
        float *tempInput1 = (float*)FastllmBorrowCudaTempBuffer(tempInput1Bytes, &workspaceBytes, &workspaceOwn);
        if (tempInput1 != nullptr && workspaceBytes >= tempInput1Bytes) {
            FastllmHalfToFloat(cudaInput1, tempInput1, input1.Count(0));
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    k, n, m, &alpha,
                    tempInput1, input1Stride, input1Spatial,
                    cudaInput0, input0Stride, input0Spatial,
                    &beta,
                    cudaOutput, k, k * n, batch);
            FastllmReleaseCudaTempBuffer(tempInput1, workspaceOwn);
        } else {
            FastllmReleaseCudaTempBuffer(tempInput1, workspaceOwn);
            status = runHalfGemm();
        }
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMul.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void FastllmMappedGdnKktPointerKernel(
        const half *input0, const half *input1, half *output,
        const half **input0Pointers, const half **input1Pointers,
        half **outputPointers, int totalChunks, int keyHeads,
        int valueHeads, int matrixElements, int outputElements) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = valueHeads * totalChunks;
    if (index >= count) {
        return;
    }
    int valueHead = index / totalChunks;
    int chunk = index - valueHead * totalChunks;
    int keyHead = valueHead / (valueHeads / keyHeads);
    input0Pointers[index] =
        input0 + ((size_t)valueHead * totalChunks + chunk) * matrixElements;
    input1Pointers[index] =
        input1 + ((size_t)keyHead * totalChunks + chunk) * matrixElements;
    outputPointers[index] =
        output + ((size_t)valueHead * totalChunks + chunk) * outputElements;
}

static cublasStatus_t FastllmRunRepeatedHeadGdnKkt(
        cublasHandle_t handle,
        const half *input0, const half *input1, half *output,
        int totalChunks, int keyHeads, int valueHeads,
        int rows, int columns, int inner, float alpha) {
    if (handle == nullptr || input0 == nullptr || input1 == nullptr ||
        output == nullptr || totalChunks <= 0 || keyHeads <= 0 ||
        valueHeads <= keyHeads || valueHeads % keyHeads != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const int headGroup = valueHeads / keyHeads;
    const long long input0MatrixElements = (long long)rows * inner;
    const long long input1MatrixElements = (long long)columns * inner;
    const long long outputMatrixElements = (long long)rows * columns;
    half hAlpha = __float2half(alpha);
    half hBeta = __float2half(0.0f);
    for (int valueHead = 0; valueHead < valueHeads; valueHead++) {
        int keyHead = valueHead / headGroup;
        cublasStatus_t status = cublasHgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            columns, rows, inner, &hAlpha,
            input1 + (long long)keyHead * totalChunks * input1MatrixElements,
            inner, input1MatrixElements,
            input0 + (long long)valueHead * totalChunks * input0MatrixElements,
            inner, input0MatrixElements, &hBeta,
            output + (long long)valueHead * totalChunks * outputMatrixElements,
            columns, outputMatrixElements, totalChunks);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

bool FastllmCudaBatchMatMulTransBHeadMapped(
        const fastllm::Data &input0, const fastllm::Data &input1,
        fastllm::Data &output, int headGroup, float alpha) {
    if (headGroup <= 1 || !std::isfinite(alpha) ||
        input0.dataDevice != fastllm::DataDevice::CUDA ||
        input1.dataDevice != fastllm::DataDevice::CUDA ||
        input0.dataType != fastllm::DataType::FLOAT16 ||
        input1.dataType != fastllm::DataType::FLOAT16 ||
        input0.cudaData == nullptr || input1.cudaData == nullptr ||
        input0.dims.size() != 5 || input1.dims.size() != 5 ||
        input0.dims[0] != 1 || input1.dims[0] != 1 ||
        input0.dims[1] != input1.dims[1] * headGroup ||
        input0.dims[2] != input1.dims[2] ||
        input0.dims[3] != input1.dims[3] ||
        input0.dims[4] != input1.dims[4] ||
        input0.dims[3] != 64 || input0.dims[4] != 128 ||
        !FastllmCudaDataHasDenseStrides(input0) ||
        !FastllmCudaDataHasDenseStrides(input1) ||
        !FastllmCudaDataCanShareDevice(input0, input1)) {
        return false;
    }
    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(input0, device) ||
        FastllmCudaGetDevice() != device) {
        return false;
    }
    const int valueHeads = input0.dims[1];
    const int keyHeads = input1.dims[1];
    const int totalChunks = input0.dims[2];
    const int rows = input0.dims[3];
    const int inner = input0.dims[4];
    const int columns = input1.dims[3];
    const int batch = valueHeads * totalChunks;
    if (batch <= 0) {
        return false;
    }
    output.dataType = fastllm::DataType::FLOAT16;
    output.Resize({1, valueHeads, totalChunks, rows, columns});
    output.ToDevice(
        fastllm::DataDevice::CUDA, std::vector<int>{device});
    output.Allocate(false);
    if (output.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(output) ||
        !FastllmCudaDataCanShareDevice(input0, output)) {
        return false;
    }

    auto runRepeatedHeadFallback = [&](const char *reason,
                                       int mappedStatus) -> bool {
        static bool warned = false;
        if (reason != nullptr && !warned) {
            printf("FastLLM CUDA: mapped ragged GDN KKT %s (%d); "
                   "falling back to strided repeated-head GEMMs.\n",
                   reason, mappedStatus);
            warned = true;
        }
        cublasStatus_t fallbackStatus = FastllmRunRepeatedHeadGdnKkt(
            getFastllmCublasHandle(),
            (const half *)input0.cudaData,
            (const half *)input1.cudaData,
            (half *)output.cudaData,
            totalChunks, keyHeads, valueHeads,
            rows, columns, inner, alpha);
        if (fallbackStatus != CUBLAS_STATUS_SUCCESS) {
            printf("Error: repeated-head GDN KKT fallback failed (%d).\n",
                   (int)fallbackStatus);
            return false;
        }
        return true;
    };

    const char *pointerMode =
        std::getenv("FASTLLM_CUDA_GDN_MAPPED_KKT_BATCHED_POINTERS");
    bool useMappedPointers = pointerMode == nullptr || pointerMode[0] == '\0' ||
        FastllmCudaEnvFlagEnabled(
            "FASTLLM_CUDA_GDN_MAPPED_KKT_BATCHED_POINTERS");
    if (!useMappedPointers) {
        return runRepeatedHeadFallback(nullptr, 0);
    }

    struct PointerScratch {
        void *data = nullptr;
        size_t capacity = 0;
        int device = -1;
        ~PointerScratch() {
            if (data == nullptr || device < 0) {
                return;
            }
            int oldDevice = 0;
            cudaGetDevice(&oldDevice);
            cudaSetDevice(device);
            FastllmCudaFree(data);
            cudaSetDevice(oldDevice);
        }
    };
    static thread_local PointerScratch scratch;
    size_t pointerBytes = (size_t)batch * sizeof(half*);
    size_t needBytes = pointerBytes * 3;
    if (scratch.device != device || scratch.capacity < needBytes) {
        if (scratch.data != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(scratch.data);
        }
        scratch.data = FastllmCudaMalloc(needBytes);
        if (scratch.data == nullptr) {
            scratch.capacity = 0;
            scratch.device = -1;
            return runRepeatedHeadFallback("pointer scratch allocation failed", 0);
        }
        scratch.capacity = needBytes;
        scratch.device = device;
    }
    const half **input0Pointers = (const half **)scratch.data;
    const half **input1Pointers = (const half **)(
        (uint8_t*)scratch.data + pointerBytes);
    half **outputPointers = (half **)(
        (uint8_t*)scratch.data + pointerBytes * 2);
    int threads = 256;
    FastllmMappedGdnKktPointerKernel<<<
        (batch + threads - 1) / threads, threads>>>(
        (const half *)input0.cudaData,
        (const half *)input1.cudaData,
        (half *)output.cudaData,
        input0Pointers, input1Pointers, outputPointers,
        totalChunks, keyHeads, valueHeads, rows * inner,
        rows * columns);
    cudaError_t pointerState = cudaGetLastError();
    if (pointerState != cudaSuccess) {
        return runRepeatedHeadFallback(
            "pointer setup failed", (int)pointerState);
    }

    half hAlpha = __float2half(alpha);
    half hBeta = __float2half(0.0f);
    cublasStatus_t status = cublasHgemmBatched(
        getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
        columns, rows, inner, &hAlpha,
        input1Pointers, inner,
        input0Pointers, inner, &hBeta,
        outputPointers, columns, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return runRepeatedHeadFallback("batched GEMM failed", (int)status);
    }
    return true;
}

bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status = CUBLAS_STATUS_NOT_SUPPORTED;

    if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT16 && input1.dataType == fastllm::DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        k, n, m, &h_alpha,
                                        (half*)cudaInput1, input1Stride, input1Spatial,
                                        (half*)cudaInput0, input0Stride, input0Spatial,
                                        &h_beta,
                                        (half*)cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::BFLOAT16 &&
               input1.dataType == fastllm::DataType::BFLOAT16) {
        status = cublasGemmStridedBatchedEx(
                fastllmCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                k, n, m, &alpha,
                cudaInput1, CUDA_R_16BF, input1Stride, input1Spatial,
                cudaInput0, CUDA_R_16BF, input0Stride, input0Spatial,
                &beta, cudaOutput, CUDA_R_16BF, k, k * n, batch,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        auto runHalfGemm = [&]() {
            size_t tempInput0Bytes = input0.Count(0) * sizeof(half);
            size_t tempOutputBytes = output.Count(0) * sizeof(half);
            size_t tempInput0AlignedBytes = FastllmCudaAlignBytes(tempInput0Bytes, 256);
            size_t needBytes = tempInput0AlignedBytes + tempOutputBytes;
            size_t workspaceBytes = 0;
            bool workspaceOwn = false;
            uint8_t *workspace = (uint8_t*)FastllmBorrowCudaTempBuffer(needBytes, &workspaceBytes, &workspaceOwn);
            bool useWorkspace = workspace != nullptr && workspaceBytes >= needBytes;
            half *tempInput0 = nullptr;
            half *tempOutput = nullptr;
            if (useWorkspace) {
                tempInput0 = (half*)workspace;
                tempOutput = (half*)(workspace + tempInput0AlignedBytes);
            } else {
                FastllmReleaseCudaTempBuffer(workspace, workspaceOwn);
                tempInput0 = (half*)FastllmCudaMalloc(tempInput0Bytes);
                tempOutput = (half*)FastllmCudaMalloc(tempOutputBytes);
                if (tempInput0 == nullptr || tempOutput == nullptr) {
                    FastllmCudaFree(tempInput0);
                    FastllmCudaFree(tempOutput);
                    return CUBLAS_STATUS_ALLOC_FAILED;
                }
            }
            FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

            half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
            cublasStatus_t halfStatus = cublasHgemmStridedBatched(fastllmCublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            k, n, m, &h_alpha,
                                            (half*)cudaInput1, input1Stride, input1Spatial,
                                            (half*)tempInput0, input0Stride, input0Spatial,
                                            &h_beta,
                                            (half*)tempOutput, k, k * n, batch);
            if (halfStatus == CUBLAS_STATUS_SUCCESS) {
                FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
            }
            if (useWorkspace) {
                FastllmReleaseCudaTempBuffer(workspace, workspaceOwn);
            } else {
                FastllmCudaFree(tempInput0);
                FastllmCudaFree(tempOutput);
            }
            return halfStatus;
        };

        size_t tempInput1Bytes = input1.Count(0) * sizeof(float);
        size_t workspaceBytes = 0;
        bool workspaceOwn = false;
        float *tempInput1 = (float*)FastllmBorrowCudaTempBuffer(tempInput1Bytes, &workspaceBytes, &workspaceOwn);
        if (tempInput1 != nullptr && workspaceBytes >= tempInput1Bytes) {
            FastllmHalfToFloat(cudaInput1, tempInput1, input1.Count(0));
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            k, n, m, &alpha,
                                            tempInput1, input1Stride, input1Spatial,
                                            cudaInput0, input0Stride, input0Spatial,
                                            &beta,
                                            cudaOutput, k, k * n, batch);
            FastllmReleaseCudaTempBuffer(tempInput1, workspaceOwn);
        } else {
            FastllmReleaseCudaTempBuffer(tempInput1, workspaceOwn);
            status = runHalfGemm();
        }
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMulTransB.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmRotatePosition2DKernel <<< outer * 2 * n, std::min(rotaryDim, m / 4) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                len, bs, spatial, n, m,
                                                                                (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);

    return true;
}

bool FastllmCudaNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                       const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int positionStride) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    positionStride = (int)positionIds.dims.back() * positionStride;

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmNearlyRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                    len, bs, spatial, n, m,
                                                                                    positionStride, (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmNearlyRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                    len, bs, spatial, n, m,
                                                                                    positionStride, (int)sinData.dims[1], rotaryDim);
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> ((__nv_bfloat16*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlamaRotatePosition2DPart(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int part) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlamaRotatePosition2DPartKernel <<< outer * n, std::min(rotaryDim, part / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], part);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlamaRotatePosition2DPartKernel <<< outer * n, std::min(rotaryDim, part / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], part);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

// ============================================================
// Fused QKV RMSNorm + RoPE Kernel
// 在一个 kernel 里对 qkv 拼接张量完成:
//   - 对 q 部分: RMSNorm + RoPE
//   - 对 k 部分: RMSNorm + RoPE
//   - v 部分: 不做处理
//
// qkv 布局: [bs * seqlen, total_dim]
// total_dim = q_heads * head_dim + k_heads * head_dim + v_heads * head_dim
//
// 每个 block 处理一个 (token, head) 对应的 head_dim 维向量
// grid: (bs * seqlen * (q_heads + k_heads))
// ============================================================
template <int THREAD_PER_BLOCK>
__global__ void FastllmQKVRMSNormRopeKernel(
    float *qkvData,          // [outer, total_dim]
    float *qNormWeight,      // [head_dim]
    float *kNormWeight,      // [head_dim]
    float *positionIds,      // [bs, seqlen] or [bs, partStride]
    int outer,               // bs * seqlen
    int total_dim,           // q_heads * head_dim + k_heads * head_dim + v_heads * head_dim
    int q_heads,
    int k_heads,
    int head_dim,
    int bs,
    int seqlen,
    int partStride,          // positionIds.dims.back()
    int rotateDim,
    float eps,
    float ropeTheta,
    float ropeScale
) {
    int block_id = blockIdx.x;               // block id in [0, outer * (q_heads + k_heads))
    int token_id = block_id / (q_heads + k_heads);  // which token [0, outer)
    int head_id = block_id % (q_heads + k_heads);   // which head in q+k space
    
    int b = token_id / seqlen;   // batch index
    int l = token_id % seqlen;   // position in sequence

    // 确定当前 head 在 qkv 中的偏移
    // q 部分: offset = head_id * head_dim
    // k 部分: offset = q_heads * head_dim + (head_id - q_heads) * head_dim
    bool is_q = (head_id < q_heads);
    int offset_in_total;
    float *normWeight;
    if (is_q) {
        offset_in_total = head_id * head_dim;
        normWeight = qNormWeight;
    } else {
        offset_in_total = q_heads * head_dim + (head_id - q_heads) * head_dim;
        normWeight = kNormWeight;
    }

    float *base = qkvData + token_id * total_dim + offset_in_total;
    unsigned int tid = threadIdx.x;

    // ======== Step 1: RMSNorm ========
    // 1.1 计算平方和
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float scale;

    float local_sum2 = 0.0f;
    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        float x = base[i];
        local_sum2 += x * x;
    }
    sdata[tid] = local_sum2;
    __syncthreads();

    // 1.2 reduce 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 1.3 计算 scale
    if (tid == 0) {
        scale = 1.0f / sqrtf(sdata[0] / head_dim + eps);
    }
    __syncthreads();

    // 1.4 应用 RMSNorm: output[i] = input[i] * scale * weight[i]
    // 同时用 shared memory 暂存归一化后的值以便 RoPE 使用
    // 注意: head_dim 通常是 128，THREAD_PER_BLOCK=128 时每个线程正好处理 1 个元素
    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        base[i] = base[i] * scale * normWeight[i];
    }
    __syncthreads();

    // ======== Step 2: RoPE Encoding ========
    // RoPE 只处理前 rotateDim 个维度 (每次处理一对)
    int half_rotate = rotateDim / 2;
    if ((int)tid < half_rotate) {
        int j = tid;
        int index = (int)(positionIds[b * partStride + l]);
        float position = (float)index / ropeScale;
        float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
        float curSin = sinf(freq);
        float curCos = cosf(freq);

        float va = base[j];
        float vb = base[j + half_rotate];
        base[j]               = va * curCos - vb * curSin;
        base[j + half_rotate] = va * curSin + vb * curCos;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmQKVRMSNormRopeKernel(
    half *qkvData,
    float *qNormWeight,
    float *kNormWeight,
    float *positionIds,
    int outer,
    int total_dim,
    int q_heads,
    int k_heads,
    int head_dim,
    int bs,
    int seqlen,
    int partStride,
    int rotateDim,
    float eps,
    float ropeTheta,
    float ropeScale
) {
    int block_id = blockIdx.x;
    int token_id = block_id / (q_heads + k_heads);
    int head_id = block_id % (q_heads + k_heads);
    
    int b = token_id / seqlen;
    int l = token_id % seqlen;

    bool is_q = (head_id < q_heads);
    int offset_in_total;
    float *normWeight;
    if (is_q) {
        offset_in_total = head_id * head_dim;
        normWeight = qNormWeight;
    } else {
        offset_in_total = q_heads * head_dim + (head_id - q_heads) * head_dim;
        normWeight = kNormWeight;
    }

    half *base = qkvData + token_id * total_dim + offset_in_total;
    unsigned int tid = threadIdx.x;

    // ======== Step 1: RMSNorm ========
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float scale;

    float local_sum2 = 0.0f;
    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        float x = __half2float(base[i]);
        local_sum2 += x * x;
    }
    sdata[tid] = local_sum2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *now = sdata;
        now[tid] += now[tid + 32];
        now[tid] += now[tid + 16];
        now[tid] += now[tid + 8];
        now[tid] += now[tid + 4];
        now[tid] += now[tid + 2];
        now[tid] += now[tid + 1];
    }
    __syncthreads();

    if (tid == 0) {
        scale = 1.0f / sqrtf(sdata[0] / head_dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        base[i] = __float2half(__half2float(base[i]) * scale * normWeight[i]);
    }
    __syncthreads();

    // ======== Step 2: RoPE Encoding ========
    int half_rotate = rotateDim / 2;
    if ((int)tid < half_rotate) {
        int j = tid;
        int index = (int)(positionIds[b * partStride + l]);
        float position = (float)index / ropeScale;
        float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
        float curSin = sinf(freq);
        float curCos = cosf(freq);

        float va = __half2float(base[j]);
        float vb = __half2float(base[j + half_rotate]);
        base[j]               = __float2half(va * curCos - vb * curSin);
        base[j + half_rotate] = __float2half(va * curSin + vb * curCos);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmQKVRMSNormRopeKernel(
    __nv_bfloat16 *qkvData,
    float *qNormWeight,
    float *kNormWeight,
    float *positionIds,
    int outer,
    int total_dim,
    int q_heads,
    int k_heads,
    int head_dim,
    int bs,
    int seqlen,
    int partStride,
    int rotateDim,
    float eps,
    float ropeTheta,
    float ropeScale
) {
    int block_id = blockIdx.x;
    int token_id = block_id / (q_heads + k_heads);
    int head_id = block_id % (q_heads + k_heads);
    
    int b = token_id / seqlen;
    int l = token_id % seqlen;

    bool is_q = (head_id < q_heads);
    int offset_in_total;
    float *normWeight;
    if (is_q) {
        offset_in_total = head_id * head_dim;
        normWeight = qNormWeight;
    } else {
        offset_in_total = q_heads * head_dim + (head_id - q_heads) * head_dim;
        normWeight = kNormWeight;
    }

    __nv_bfloat16 *base = qkvData + token_id * total_dim + offset_in_total;
    unsigned int tid = threadIdx.x;

    // ======== Step 1: RMSNorm ========
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float scale;

    float local_sum2 = 0.0f;
    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        float x = __bfloat162float(base[i]);
        local_sum2 += x * x;
    }
    sdata[tid] = local_sum2;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *now = sdata;
        now[tid] += now[tid + 32];
        now[tid] += now[tid + 16];
        now[tid] += now[tid + 8];
        now[tid] += now[tid + 4];
        now[tid] += now[tid + 2];
        now[tid] += now[tid + 1];
    }
    __syncthreads();

    if (tid == 0) {
        scale = 1.0f / sqrtf(sdata[0] / head_dim + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
        base[i] = __float2bfloat16(__bfloat162float(base[i]) * scale * normWeight[i]);
    }
    __syncthreads();

    // ======== Step 2: RoPE Encoding ========
    int half_rotate = rotateDim / 2;
    if ((int)tid < half_rotate) {
        int j = tid;
        int index = (int)(positionIds[b * partStride + l]);
        float position = (float)index / ropeScale;
        float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
        float curSin = sinf(freq);
        float curCos = cosf(freq);

        float va = __bfloat162float(base[j]);
        float vb = __bfloat162float(base[j + half_rotate]);
        base[j]               = __float2bfloat16(va * curCos - vb * curSin);
        base[j + half_rotate] = __float2bfloat16(va * curSin + vb * curCos);
    }
}

bool FastllmCudaQKVRMSNormRope(
    fastllm::Data &qkv,
    fastllm::Data &qNormWeight,
    fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale
) {
    float *cudaQKV = (float *) FastllmCudaPrepareInput(qkv);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int bs = qkv.dims[0];
    int seqlen = qkv.dims[1];
    int total_dim = qkv.dims[2];
    int outer = bs * seqlen;
    int total_heads = q_heads + k_heads;
    int grid_size = outer * total_heads;
    int partStride = (int)positionIds.dims.back();

    // 选择 block 大小: head_dim 通常是 128
    if (qkv.dataType == fastllm::DataType::FLOAT32) {
        if (head_dim <= 64) {
            FastllmQKVRMSNormRopeKernel<64> <<< grid_size, 64 >>>(
                cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else if (head_dim <= 128) {
            FastllmQKVRMSNormRopeKernel<128> <<< grid_size, 128 >>>(
                cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else {
            FastllmQKVRMSNormRopeKernel<512> <<< grid_size, 512 >>>(
                cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        }
    } else if (qkv.dataType == fastllm::DataType::FLOAT16) {
        if (head_dim <= 64) {
            FastllmQKVRMSNormRopeKernel<64> <<< grid_size, 64 >>>(
                (half*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else if (head_dim <= 128) {
            FastllmQKVRMSNormRopeKernel<128> <<< grid_size, 128 >>>(
                (half*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else {
            FastllmQKVRMSNormRopeKernel<512> <<< grid_size, 512 >>>(
                (half*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        }
    } else if (qkv.dataType == fastllm::DataType::BFLOAT16) {
        if (head_dim <= 64) {
            FastllmQKVRMSNormRopeKernel<64> <<< grid_size, 64 >>>(
                (__nv_bfloat16*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else if (head_dim <= 128) {
            FastllmQKVRMSNormRopeKernel<128> <<< grid_size, 128 >>>(
                (__nv_bfloat16*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        } else {
            FastllmQKVRMSNormRopeKernel<512> <<< grid_size, 512 >>>(
                (__nv_bfloat16*)cudaQKV, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, outer, total_dim, q_heads, k_heads, head_dim,
                bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale);
        }
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(qkv, cudaQKV);
    return true;
}

// ============================================================
// 融合 QKVRMSNormRope + Split + AppendPagedCacheBatch
// 每个 block 处理一个 (token, head) 对应的 head_dim 维向量
// grid: (bs * seqlen * (q_heads + k_heads + v_heads))
//   - head_id < q_heads: Q head -> RMSNorm + RoPE -> 写入 qOutput (permuted)
//   - q_heads <= head_id < q_heads + k_heads: K head -> RMSNorm + RoPE -> 写入 paged K cache
//   - head_id >= q_heads + k_heads: V head -> 直接拷贝到 paged V cache
// ============================================================
template <int THREAD_PER_BLOCK, typename T, typename TKV>
__global__ void FastllmQKVRMSNormRopeSplitAppendPagedCacheKernel(
    T *qkvData,              // [bs, seqlen, total_dim], 物理布局; 逻辑含义为 batch 个 token
    float *qNormWeight,      // [head_dim]
    float *kNormWeight,      // [head_dim]
    float *positionIds,      // [bs, partStride]
    T *qOutputData,          // [bsz * q_heads, seqlen, head_dim] (permuted output)
    TKV *pagedKData,         // paged K cache raw data
    TKV *pagedVData,         // paged V cache raw data
    int32_t *insertIndexs,   // [batch] page index for each batch (逻辑 batch)
    int32_t *insertPositions,// [batch] page offset for each batch (逻辑 batch)
    int32_t *lastPageLens,   // optional [batch], filled with page offset after append
    int outer,               // bs * seqlen = 总 token 数
    int total_dim,           // (q_heads + k_heads + v_heads) * head_dim
    int q_heads,
    int k_heads,
    int v_heads,
    int head_dim,
    int bs,                  // qkv.dims[0], 物理 batch 维
    int seqlen,              // qkv.dims[1], 物理 seqlen 维
    int partStride,          // positionIds.dims.back()
    int rotateDim,
    float eps,
    float ropeTheta,
    float ropeScale,
    int pageLen,             // page length for paged cache
    int maxPages,            // max pages in paged cache
    int batch,               // 逻辑 batch 数（= insertIndexs 长度）
    int doQKNorm,            // 是否做 QK RMSNorm（0 = 跳过）
    int useLlama3,
    float llama3Factor,
    float llama3OriginalMaxPosition,
    float llama3LowFreqFactor,
    float llama3HighFreqFactor,
    int useYarn,
    float yarnFactor,
    float yarnAttentionFactor,
    float yarnCorrectionLow,
    float yarnCorrectionHigh
) {
    int total_heads = q_heads + k_heads + v_heads;
    int block_id = blockIdx.x;
    int token_id = block_id / total_heads;  // [0, outer), 即第几个 token
    int head_id = block_id % total_heads;

    // 物理维度索引（用于定位 qkv 和 positionIds）
    int phys_b = token_id / seqlen;   // qkv 的物理 batch 索引
    int phys_l = token_id % seqlen;   // qkv 的物理 seq 索引

    // 逻辑 batch 索引（用于 insertIndexs / insertPositions）
    // 在 decode 路径: bs=1, seqlen=batch, 逻辑 batch_idx = token_id
    // 在单 batch 路径: bs=1, seqlen=1, batch=1, 逻辑 batch_idx = 0
    int batch_idx = token_id;  // 每个 token 对应一个逻辑 batch（decode 模式下 seqlen_per_batch=1）

    unsigned int tid = threadIdx.x;

    if (batch_idx >= batch) {
        return;
    }

    int insertPageIdx = insertIndexs[batch_idx];
    int insertPageOffset = insertPositions[batch_idx];
    bool validInsert = insertPageIdx >= 0 && insertPageIdx < maxPages &&
                       insertPageOffset >= 0 && insertPageOffset < pageLen;

    if (lastPageLens != nullptr && head_id == 0 && tid == 0) {
        lastPageLens[batch_idx] = validInsert ? insertPageOffset + 1 : 0;
    }

    // 确定当前 head 在 qkv 中的偏移
    int offset_in_total;
    if (head_id < q_heads) {
        offset_in_total = head_id * head_dim;
    } else if (head_id < q_heads + k_heads) {
        offset_in_total = q_heads * head_dim + (head_id - q_heads) * head_dim;
    } else {
        offset_in_total = (q_heads + k_heads) * head_dim + (head_id - q_heads - k_heads) * head_dim;
    }

    T *base = qkvData + token_id * total_dim + offset_in_total;

    if (head_id < q_heads + k_heads) {
        // ======== Q or K head: (optional) RMSNorm + RoPE ========
        if (doQKNorm) {
            float *normWeight = (head_id < q_heads) ? qNormWeight : kNormWeight;

            // Step 1: RMSNorm
            __shared__ float sdata[THREAD_PER_BLOCK];
            __shared__ float scale;

            float local_sum2 = 0.0f;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                float x = (float)base[i];
                local_sum2 += x * x;
            }
            sdata[tid] = local_sum2;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            if (tid < 32) {
                volatile float *now = sdata;
                now[tid] += now[tid + 32];
                now[tid] += now[tid + 16];
                now[tid] += now[tid + 8];
                now[tid] += now[tid + 4];
                now[tid] += now[tid + 2];
                now[tid] += now[tid + 1];
            }
            __syncthreads();

            if (tid == 0) {
                scale = 1.0f / sqrtf(sdata[0] / head_dim + eps);
            }
            __syncthreads();

            // Apply RMSNorm in-place
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                base[i] = (T)((float)base[i] * scale * normWeight[i]);
            }
            __syncthreads();
        }

        // Step 2: RoPE Encoding
        // Single-token batch decode may arrive as either [1, batch] or [batch, 1].
        // The logical token order is still token_id, matching insertIndexs/insertPositions.
        int half_rotate = rotateDim / 2;
        if ((int)tid < half_rotate) {
            int j = tid;
            int positionOffset = phys_b * partStride + phys_l;
            if (outer == batch && batch > 1) {
                positionOffset = (partStride == 1) ? phys_b : batch_idx;
            }
            float rawPosition = positionIds[positionOffset];
            float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
            float freq;
            if (useYarn) {
                invFreq = FastllmYarnInvFreq(j, rotateDim, ropeTheta,
                                              yarnFactor,
                                              yarnCorrectionLow,
                                              yarnCorrectionHigh);
                freq = (float)((int)rawPosition) * invFreq;
            } else if (useLlama3) {
                invFreq = FastllmLlama3InvFreq(invFreq, llama3Factor,
                                                llama3OriginalMaxPosition,
                                                llama3LowFreqFactor,
                                                llama3HighFreqFactor);
                freq = rawPosition * invFreq;
            } else {
                float position = (float)((int)rawPosition) / ropeScale;
                freq = position * invFreq;
            }
            float curSin = sinf(freq);
            float curCos = cosf(freq);
            if (useYarn) {
                curSin *= yarnAttentionFactor;
                curCos *= yarnAttentionFactor;
            }

            float va = (float)base[j];
            float vb = (float)base[j + half_rotate];
            base[j]               = (T)(va * curCos - vb * curSin);
            base[j + half_rotate] = (T)(va * curSin + vb * curCos);
        }
        __syncthreads();

        // Step 3: Write output
        if (head_id < q_heads) {
            // Q head: 写入 qOutput，布局 [bsz * q_heads, seqlen, head_dim]
            // Permute: [bs, seqlen, q_heads, head_dim] -> [bs, q_heads, seqlen, head_dim] -> [bs * q_heads, seqlen, head_dim]
            // 即 (phys_b, phys_l, head_id) -> (phys_b * q_heads + head_id, phys_l, :)
            T *dst = qOutputData + ((phys_b * q_heads + head_id) * seqlen + phys_l) * head_dim;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                dst[i] = base[i];
            }
        } else {
            // K head: 直接写入 paged K cache
            // pagedData layout: [maxPages, pageLen, numHeads, headDim]
            // 用逻辑 batch_idx 索引 insertIndexs / insertPositions
            if (!validInsert) {
                return;
            }
            int kh = head_id - q_heads;
            int pageStride = pageLen * k_heads * head_dim;
            int tokenStride = k_heads * head_dim;
            TKV *dst = pagedKData + (size_t)insertPageIdx * pageStride + insertPageOffset * tokenStride + kh * head_dim;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                dst[i] = FastllmCudaFloatToValue<TKV>(FastllmCudaValueToFloat(base[i]));
            }
        }
    } else {
        // ======== V head: 直接拷贝到 paged V cache（无需 RMSNorm/RoPE）========
        // 用逻辑 batch_idx 索引 insertIndexs / insertPositions
        if (!validInsert) {
            return;
        }
        int vh = head_id - q_heads - k_heads;
        int pageStride = pageLen * v_heads * head_dim;
        int tokenStride = v_heads * head_dim;
        TKV *dst = pagedVData + (size_t)insertPageIdx * pageStride + insertPageOffset * tokenStride + vh * head_dim;
        for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
            dst[i] = FastllmCudaFloatToValue<TKV>(FastllmCudaValueToFloat(base[i]));
        }
    }
}

bool FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qkv,
    fastllm::Data &qNormWeight,
    fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    uint8_t *pagedKData,
    uint8_t *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale,
    int pageLen, int maxPages, fastllm::DataType pagedDataType, int batch,
    int doQKNorm,
    int useLlama3, float llama3Factor,
    float llama3OriginalMaxPosition,
    float llama3LowFreqFactor,
    float llama3HighFreqFactor,
    int useYarn, float yarnFactor,
    float yarnAttentionFactor,
    float yarnCorrectionLow,
    float yarnCorrectionHigh
) {
    float *cudaQKV = (float *) FastllmCudaPrepareInput(qkv);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int bs = qkv.dims[0];
    int seqlen = qkv.dims[1];
    int total_dim = qkv.dims[2];
    int v_heads = k_heads; // v_heads == k_heads
    int outer = bs * seqlen;
    int total_heads = q_heads + k_heads + v_heads;
    int grid_size = outer * total_heads;
    int partStride = (int)positionIds.dims.back();

    // 确保 qOutput 已分配
    float *cudaQOutput = (float*)qOutput.cudaData;

    auto launch = [&](auto TPB, auto *qkvPtr, auto *qOutputPtr, auto *pagedTag) {
        using QT = std::remove_pointer_t<decltype(qkvPtr)>;
        using KVT = std::remove_pointer_t<decltype(pagedTag)>;
        FastllmQKVRMSNormRopeSplitAppendPagedCacheKernel<decltype(TPB)::value, QT, KVT><<<grid_size, decltype(TPB)::value>>>(
            qkvPtr, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
            cudaPositionIds, qOutputPtr,
            (KVT*)pagedKData, (KVT*)pagedVData, insertIndexs, insertPositions, lastPageLens,
            outer, total_dim, q_heads, k_heads, v_heads, head_dim,
            bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale, pageLen, maxPages, batch, doQKNorm,
            useLlama3, llama3Factor, llama3OriginalMaxPosition,
            llama3LowFreqFactor, llama3HighFreqFactor,
            useYarn, yarnFactor, yarnAttentionFactor,
            yarnCorrectionLow, yarnCorrectionHigh);
    };

    auto launchByPagedType = [&](auto TPB, auto *qkvPtr, auto *qOutputPtr) {
        if (pagedDataType == fastllm::DataType::FLOAT32) {
            launch(TPB, qkvPtr, qOutputPtr, (float*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FLOAT16) {
            launch(TPB, qkvPtr, qOutputPtr, (half*)nullptr);
        } else if (pagedDataType == fastllm::DataType::BFLOAT16) {
            launch(TPB, qkvPtr, qOutputPtr, (__nv_bfloat16*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FP8_E4M3) {
            launch(TPB, qkvPtr, qOutputPtr, (__nv_fp8_e4m3*)nullptr);
        } else {
            fastllm::ErrorInFastLLM("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: unsupported pagedDataType.\n");
        }
    };

    if (qkv.dataType == fastllm::DataType::FLOAT32) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (float*)cudaQKV, (float*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (float*)cudaQKV, (float*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (float*)cudaQKV, (float*)cudaQOutput);
    } else if (qkv.dataType == fastllm::DataType::FLOAT16) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (half*)cudaQKV, (half*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (half*)cudaQKV, (half*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (half*)cudaQKV, (half*)cudaQOutput);
    } else if (qkv.dataType == fastllm::DataType::BFLOAT16) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
    } else {
        fastllm::ErrorInFastLLM("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: unsupported qkv dataType.\n");
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: kernel launch failed: %s\n",
               cudaGetErrorString(launchError));
        return false;
    }
    // 注意: 不需要 FinishOutput qkv，因为 qkv 内容已经不再需要
    return true;
}

template <int HEAD_DIM>
__global__ void FastllmQwen35QGateKVPrefillHalfKernel(
    const half *qgatekvData,
    const float *qNormWeight,
    const float *kNormWeight,
    const float *positionIds,
    half *qOutputData,
    half *gateOutputData,
    half *kOutputData,
    half *vOutputData,
    int totalDim,
    int qHeads,
    int kHeads,
    int seqlen,
    int positionStride,
    int rotaryDim,
    int sectionH,
    int sectionW,
    int useInterleavedRope,
    float eps,
    float ropeTheta,
    float ropeScale) {
    constexpr int WARP_SIZE = 32;
    constexpr int RMS_THREADS = 64;
    constexpr int NUM_WARPS = RMS_THREADS / WARP_SIZE;
    __shared__ float warpSums[NUM_WARPS];
    __shared__ float normScale;

    int totalHeads = qHeads + kHeads + kHeads;
    int tokenId = blockIdx.x / totalHeads;
    int headId = blockIdx.x - tokenId * totalHeads;
    int tid = threadIdx.x;
    int qGateDim = qHeads * HEAD_DIM * 2;
    const half *tokenBase = qgatekvData + (size_t)tokenId * totalDim;

    if (headId >= qHeads + kHeads) {
        int vHead = headId - qHeads - kHeads;
        const half2 *src = reinterpret_cast<const half2 *>(
            tokenBase + qGateDim + kHeads * HEAD_DIM +
            vHead * HEAD_DIM);
        half2 *dst = reinterpret_cast<half2 *>(
            vOutputData +
            ((size_t)vHead * seqlen + tokenId) * HEAD_DIM);
        for (int i = tid; i < HEAD_DIM / 2; i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    bool isQ = headId < qHeads;
    int localHead = isQ ? headId : headId - qHeads;
    const half *srcBase = isQ
        ? tokenBase + localHead * HEAD_DIM * 2
        : tokenBase + qGateDim + localHead * HEAD_DIM;
    const float *normWeight = isQ ? qNormWeight : kNormWeight;
    const half2 *src2 = reinterpret_cast<const half2 *>(srcBase);
    float sum2 = 0.0f;
    if (tid < RMS_THREADS) {
        for (int i = tid; i < HEAD_DIM / 2;
             i += RMS_THREADS) {
            float2 value = __half22float2(src2[i]);
            sum2 +=
                value.x * value.x + value.y * value.y;
        }
    }

    int laneId = tid & (WARP_SIZE - 1);
    int warpId = tid / WARP_SIZE;
    if (tid < RMS_THREADS) {
        for (int offset = WARP_SIZE / 2;
             offset > 0; offset >>= 1) {
            sum2 += __shfl_down_sync(
                0xffffffff, sum2, offset);
        }
        if (laneId == 0) {
            warpSums[warpId] = sum2;
        }
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        float val = laneId < NUM_WARPS ? warpSums[laneId] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (laneId == 0) {
            normScale = rsqrtf(val / HEAD_DIM + eps);
        }
    }
    __syncthreads();

    half *outputBase = isQ
        ? qOutputData +
            ((size_t)localHead * seqlen + tokenId) * HEAD_DIM
        : kOutputData +
            ((size_t)localHead * seqlen + tokenId) * HEAD_DIM;
    half2 *output2 = reinterpret_cast<half2 *>(outputBase);
    if (tid < RMS_THREADS) {
        float scale = normScale;
        for (int i = tid; i < HEAD_DIM / 2;
             i += RMS_THREADS) {
            float2 value = __half22float2(src2[i]);
            float2 normalized;
            normalized.x =
                value.x * scale *
                __ldg(normWeight + i * 2);
            normalized.y =
                value.y * scale *
                __ldg(normWeight + i * 2 + 1);
            output2[i] = __float22half2_rn(normalized);
        }
    }

    if (isQ) {
        const half2 *gateSrc = reinterpret_cast<const half2 *>(
            srcBase + HEAD_DIM);
        half2 *gateDst = reinterpret_cast<half2 *>(
            gateOutputData +
            ((size_t)tokenId * qHeads + localHead) * HEAD_DIM);
        for (int i = tid; i < HEAD_DIM / 2; i += blockDim.x) {
            gateDst[i] = gateSrc[i];
        }
    }
    __syncthreads();

    int halfRotary = rotaryDim / 2;
    if (tid < halfRotary) {
        int row = 0;
        float position;
        if (useInterleavedRope) {
            if (tid % 3 == 1 && tid < sectionH * 3) {
                row = 1;
            } else if (tid % 3 == 2 && tid < sectionW * 3) {
                row = 2;
            }
            position =
                positionIds[row * positionStride + tokenId] /
                ropeScale;
        } else {
            int index = (int)positionIds[tokenId];
            position = (float)index / ropeScale;
        }
        float freq =
            position /
            powf(ropeTheta, (float)(2 * tid) / rotaryDim);
        float curSin = sinf(freq);
        float curCos = cosf(freq);
        float va = __half2float(outputBase[tid]);
        float vb =
            __half2float(outputBase[tid + halfRotary]);
        outputBase[tid] =
            __float2half(va * curCos - vb * curSin);
        outputBase[tid + halfRotary] =
            __float2half(va * curSin + vb * curCos);
    }
}

bool FastllmCudaQwen35QGateKVPrefill(
    const fastllm::Data &qgatekv,
    const fastllm::Data &qNormWeight,
    const fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    fastllm::Data &gateOutput,
    fastllm::Data &kOutput,
    fastllm::Data &vOutput,
    int qHeads, int kHeads, int headDim,
    int rotaryDim, int sectionT, int sectionH, int sectionW,
    float eps, float ropeTheta, float ropeScale) {
    if (qgatekv.dataType != fastllm::DataType::FLOAT16 ||
        qNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        kNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        positionIds.dataType != fastllm::DataType::FLOAT32 ||
        qgatekv.dims.size() != 3 || qgatekv.dims[0] != 1 ||
        (headDim != 128 && headDim != 256) ||
        rotaryDim <= 0 || rotaryDim > headDim ||
        (rotaryDim % 2) != 0 ||
        qHeads <= 0 || kHeads <= 0 ||
        qgatekv.dims[2] !=
            qHeads * headDim * 2 + kHeads * headDim * 2 ||
        qNormWeight.Count(0) != headDim ||
        kNormWeight.Count(0) != headDim ||
        qOutput.cudaData == nullptr ||
        gateOutput.cudaData == nullptr ||
        kOutput.cudaData == nullptr ||
        vOutput.cudaData == nullptr) {
        return false;
    }
    int seqlen = qgatekv.dims[1];
    bool useInterleavedRope =
        positionIds.dims.size() == 2 && positionIds.dims[0] == 3;
    if (useInterleavedRope &&
        sectionT + sectionH + sectionW != rotaryDim / 2) {
        return false;
    }
    if ((!useInterleavedRope &&
         (positionIds.dims.empty() ||
          positionIds.dims.back() < seqlen)) ||
        (useInterleavedRope &&
         positionIds.dims.back() < seqlen)) {
        return false;
    }

    void *qgatekvData = FastllmCudaPrepareInput(qgatekv);
    void *positionData = FastllmCudaPrepareInput(positionIds);
    if (qgatekvData == nullptr || positionData == nullptr) {
        FastllmCudaFinishInput(qgatekv, qgatekvData);
        FastllmCudaFinishInput(positionIds, positionData);
        return false;
    }
    int totalHeads = qHeads + kHeads + kHeads;
    auto launch = [&](auto HeadDim) {
        FastllmQwen35QGateKVPrefillHalfKernel<
            decltype(HeadDim)::value>
            <<<(size_t)seqlen * totalHeads, 128, 0,
               cudaStreamPerThread>>>(
            (const half*)qgatekvData,
            (const float*)qNormWeight.cudaData,
            (const float*)kNormWeight.cudaData,
            (const float*)positionData,
            (half*)qOutput.cudaData,
            (half*)gateOutput.cudaData,
            (half*)kOutput.cudaData,
            (half*)vOutput.cudaData,
            qgatekv.dims[2], qHeads, kHeads, seqlen,
            (int)positionIds.dims.back(), rotaryDim,
            sectionH, sectionW,
            useInterleavedRope ? 1 : 0,
            eps, ropeTheta, ropeScale);
    };
    if (headDim == 128) {
        launch(std::integral_constant<int, 128>{});
    } else {
        launch(std::integral_constant<int, 256>{});
    }
    FastllmCudaFinishInput(qgatekv, qgatekvData);
    FastllmCudaFinishInput(positionIds, positionData);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        return false;
    }
    return true;
}

// ============================================================
// Qwen3.5 gated attention decode fusion:
//   input layout: [Q/Gate interleaved per Q head, K, V]
//   Q -> RMSNorm + RoPE -> qOutput
//   Gate -> gateOutput
//   K -> RMSNorm + RoPE -> paged K cache
//   V -> paged V cache
// ============================================================
template <int THREAD_PER_BLOCK, typename T, typename TKV>
__global__ void FastllmQwen35QGateKVRMSNormRopeSplitAppendPagedCacheKernel(
    T *qgatekvData,
    float *qNormWeight,
    float *kNormWeight,
    float *positionIds,
    T *qOutputData,
    T *gateOutputData,
    TKV *pagedKData,
    TKV *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int outer,
    int totalDim,
    int qHeads,
    int kHeads,
    int headDim,
    int bs,
    int seqlen,
    int positionStride,
    int rotaryDim,
    int sectionH,
    int sectionW,
    int useInterleavedRope,
    float eps,
    float ropeTheta,
    float ropeScale,
    int pageLen,
    int batch,
    int doQKNorm) {
    int totalHeads = qHeads + kHeads + kHeads;
    int blockId = blockIdx.x;
    int tokenId = blockId / totalHeads;
    int headId = blockId % totalHeads;
    int physB = tokenId / seqlen;
    int physL = tokenId % seqlen;
    int batchIdx = tokenId;
    unsigned int tid = threadIdx.x;

    if (lastPageLens != nullptr && headId == 0 && tid == 0 && batchIdx < batch) {
        lastPageLens[batchIdx] = insertPositions[batchIdx] + 1;
    }

    int qGateDim = qHeads * headDim * 2;
    T *base = nullptr;
    float *normWeight = nullptr;
    bool isQ = headId < qHeads;
    bool isK = headId >= qHeads && headId < qHeads + kHeads;
    if (isQ) {
        base = qgatekvData + (size_t)tokenId * totalDim + headId * headDim * 2;
        normWeight = qNormWeight;
    } else if (isK) {
        int kh = headId - qHeads;
        base = qgatekvData + (size_t)tokenId * totalDim + qGateDim + kh * headDim;
        normWeight = kNormWeight;
    }

    if (isQ || isK) {
        if (doQKNorm) {
            __shared__ float sdata[THREAD_PER_BLOCK];
            __shared__ float scale;

            float localSum2 = 0.0f;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                float x = FastllmCudaValueToFloat(base[i]);
                localSum2 += x * x;
            }
            sdata[tid] = localSum2;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            if (tid < 32) {
                volatile float *now = sdata;
                now[tid] += now[tid + 32];
                now[tid] += now[tid + 16];
                now[tid] += now[tid + 8];
                now[tid] += now[tid + 4];
                now[tid] += now[tid + 2];
                now[tid] += now[tid + 1];
            }
            __syncthreads();

            if (tid == 0) {
                scale = 1.0f / sqrtf(sdata[0] / headDim + eps);
            }
            __syncthreads();

            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                base[i] = FastllmCudaFloatToValue<T>(
                    FastllmCudaValueToFloat(base[i]) * scale * normWeight[i]);
            }
            __syncthreads();
        }

        int halfRotate = rotaryDim / 2;
        if ((int)tid < halfRotate) {
            int j = tid;
            float rawPosition;
            if (useInterleavedRope) {
                int row = 0;
                if (j % 3 == 1 && j < sectionH * 3) {
                    row = 1;
                } else if (j % 3 == 2 && j < sectionW * 3) {
                    row = 2;
                }
                int logicalL = (outer == batch && batch > 1) ? batchIdx : physL;
                rawPosition = positionIds[row * positionStride + logicalL];
            } else {
                int positionOffset = physB * positionStride + physL;
                if (outer == batch && batch > 1) {
                    positionOffset = (positionStride == 1) ? physB : batchIdx;
                }
                rawPosition = positionIds[positionOffset];
            }
            float position = rawPosition / ropeScale;
            float freq = position / powf(ropeTheta, (float)(2 * j) / rotaryDim);
            float curSin = sinf(freq);
            float curCos = cosf(freq);
            float va = FastllmCudaValueToFloat(base[j]);
            float vb = FastllmCudaValueToFloat(base[j + halfRotate]);
            base[j] = FastllmCudaFloatToValue<T>(va * curCos - vb * curSin);
            base[j + halfRotate] = FastllmCudaFloatToValue<T>(va * curSin + vb * curCos);
        }
        __syncthreads();

        if (isQ) {
            T *qDst = qOutputData + ((physB * qHeads + headId) * seqlen + physL) * headDim;
            T *gateBase = qgatekvData + (size_t)tokenId * totalDim + headId * headDim * 2 + headDim;
            T *gateDst = gateOutputData + (size_t)tokenId * qHeads * headDim + headId * headDim;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                qDst[i] = base[i];
                gateDst[i] = gateBase[i];
            }
        } else {
            int kh = headId - qHeads;
            int pageIdx = insertIndexs[batchIdx];
            int pageOffset = insertPositions[batchIdx];
            int pageStride = pageLen * kHeads * headDim;
            int tokenStride = kHeads * headDim;
            TKV *kDst = pagedKData + (size_t)pageIdx * pageStride +
                pageOffset * tokenStride + kh * headDim;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                kDst[i] = FastllmCudaFloatToValue<TKV>(FastllmCudaValueToFloat(base[i]));
            }
        }
    } else {
        int vh = headId - qHeads - kHeads;
        T *vBase = qgatekvData + (size_t)tokenId * totalDim + qGateDim + kHeads * headDim + vh * headDim;
        int pageIdx = insertIndexs[batchIdx];
        int pageOffset = insertPositions[batchIdx];
        int pageStride = pageLen * kHeads * headDim;
        int tokenStride = kHeads * headDim;
        TKV *vDst = pagedVData + (size_t)pageIdx * pageStride +
            pageOffset * tokenStride + vh * headDim;
        for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
            vDst[i] = FastllmCudaFloatToValue<TKV>(FastllmCudaValueToFloat(vBase[i]));
        }
    }
}

bool FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qgatekv,
    fastllm::Data &qNormWeight,
    fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    fastllm::Data &gateOutput,
    uint8_t *pagedKData,
    uint8_t *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int qHeads, int kHeads, int headDim,
    int rotaryDim, int sectionT, int sectionH, int sectionW,
    float eps, float ropeTheta, float ropeScale,
    int pageLen, fastllm::DataType pagedDataType, int batch,
    int doQKNorm) {
    fastllm::AssertInFastLLM(qgatekv.dims.size() == 3,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache expects [bs, seq, dim].\n");
    int bs = qgatekv.dims[0];
    int seqlen = qgatekv.dims[1];
    int totalDim = qgatekv.dims[2];
    int outer = bs * seqlen;
    int expectedDim = qHeads * headDim * 2 + kHeads * headDim * 2;
    fastllm::AssertInFastLLM(totalDim == expectedDim,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache got invalid qgatekv dim.\n");
    fastllm::AssertInFastLLM(outer == batch,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache is decode-only.\n");
    int useInterleavedRope = positionIds.dims.size() == 2 && positionIds.dims[0] == 3;
    if (useInterleavedRope) {
        fastllm::AssertInFastLLM(sectionT + sectionH + sectionW == rotaryDim / 2,
                                 "Qwen3.5 fused decode RoPE section sizes must sum to rotary_dim / 2.\n");
    }

    float *cudaQGateKV = (float*)FastllmCudaPrepareInput(qgatekv);
    float *cudaPositionIds = (float*)FastllmCudaPrepareInput(positionIds);
    float *cudaQOutput = (float*)qOutput.cudaData;
    float *cudaGateOutput = (float*)gateOutput.cudaData;

    int totalHeads = qHeads + kHeads + kHeads;
    int gridSize = outer * totalHeads;
    int positionStride = (int)positionIds.dims.back();

    auto launch = [&](auto TPB, auto *qgatekvPtr, auto *qOutputPtr, auto *gateOutputPtr, auto *pagedTag) {
        using QT = std::remove_pointer_t<decltype(qgatekvPtr)>;
        using KVT = std::remove_pointer_t<decltype(pagedTag)>;
        FastllmQwen35QGateKVRMSNormRopeSplitAppendPagedCacheKernel<decltype(TPB)::value, QT, KVT>
            <<<gridSize, decltype(TPB)::value>>>(
                qgatekvPtr, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, qOutputPtr, gateOutputPtr,
                (KVT*)pagedKData, (KVT*)pagedVData,
                insertIndexs, insertPositions, lastPageLens,
                outer, totalDim, qHeads, kHeads, headDim,
                bs, seqlen, positionStride, rotaryDim,
                sectionH, sectionW, useInterleavedRope,
                eps, ropeTheta, ropeScale, pageLen, batch, doQKNorm);
    };

    auto launchByPagedType = [&](auto TPB, auto *qgatekvPtr, auto *qOutputPtr, auto *gateOutputPtr) {
        if (pagedDataType == fastllm::DataType::FLOAT32) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (float*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FLOAT16) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (half*)nullptr);
        } else if (pagedDataType == fastllm::DataType::BFLOAT16) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (__nv_bfloat16*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FP8_E4M3) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (__nv_fp8_e4m3*)nullptr);
        } else {
            fastllm::ErrorInFastLLM("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: unsupported pagedDataType.\n");
        }
    };

    if (qgatekv.dataType == fastllm::DataType::FLOAT32) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        }
    } else if (qgatekv.dataType == fastllm::DataType::FLOAT16) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        }
    } else if (qgatekv.dataType == fastllm::DataType::BFLOAT16) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        }
    } else {
        fastllm::ErrorInFastLLM("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: unsupported qgatekv dataType.\n");
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: kernel launch failed: %s\n",
               cudaGetErrorString(launchError));
        return false;
    }
    return true;
}

__global__ void FastllmAdvanceDecodeMetaKernel(
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    int32_t oldLen = lastPageLens[b];
    insertPositions[b] = oldLen;
    lastPageLens[b] = oldLen + 1;
}

bool FastllmCudaAdvanceDecodeMeta(
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int batch) {
    if (batch <= 0) {
        return true;
    }
    if (insertPositions == nullptr || lastPageLens == nullptr) {
        fastllm::ErrorInFastLLM("FastllmCudaAdvanceDecodeMeta: null metadata pointer.\n");
        return false;
    }
    const int threads = 128;
    int blocks = (batch + threads - 1) / threads;
    FastllmAdvanceDecodeMetaKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
        insertPositions, lastPageLens, batch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaAdvanceDecodeMeta: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaRopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim, float ropeTheta, float ropeScale) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    int halfDim = rotaryDim / 2;
    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmRopeEncodingKernel <<< outer * n, halfDim >>> (cudaData, cudaPositionIds,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), rotaryDim, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmRopeEncodingKernel <<< outer * n, halfDim >>> ((half*)cudaData, cudaPositionIds,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), rotaryDim, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmRopeEncodingKernel <<< outer * n, halfDim >>> ((__nv_bfloat16*)cudaData, cudaPositionIds,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), rotaryDim, ropeTheta, ropeScale);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaYarnRopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                 float ropeTheta, float factor, float attentionFactor,
                                 float correctionLow, float correctionHigh) {
    fastllm::AssertInFastLLM(data.dims.size() == 4,
                             "YaRN RoPE expects [batch, seq, heads, dim] input.");
    fastllm::AssertInFastLLM(positionIds.dataType == fastllm::DataType::FLOAT32,
                             "YaRN RoPE expects FLOAT32 position ids.");
    fastllm::AssertInFastLLM(rotaryDim > 0 && rotaryDim % 2 == 0 &&
                             rotaryDim <= data.dims[3] && rotaryDim / 2 <= 1024,
                             "Invalid YaRN rotary_dim for CUDA input.");
    fastllm::AssertInFastLLM(data.dataType == fastllm::DataType::FLOAT32 ||
                             data.dataType == fastllm::DataType::FLOAT16 ||
                             data.dataType == fastllm::DataType::BFLOAT16,
                             "CUDA YaRN RoPE supports FLOAT32, FLOAT16 and BFLOAT16 input.");

    float *cudaData = (float *)FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *)FastllmCudaPrepareInput(positionIds);
    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[1], n = data.dims[2], m = data.dims[3];
    int halfDim = rotaryDim / 2;
    int positionStride = (int)positionIds.dims.back();

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmYarnRopeEncodingKernel <<< outer, halfDim >>> (
            cudaData, cudaPositionIds, len, spatial, n, m, positionStride, rotaryDim,
            ropeTheta, factor, attentionFactor, correctionLow, correctionHigh);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmYarnRopeEncodingKernel <<< outer, halfDim >>> (
            (half*)cudaData, cudaPositionIds, len, spatial, n, m, positionStride, rotaryDim,
            ropeTheta, factor, attentionFactor, correctionLow, correctionHigh);
    } else {
        FastllmYarnRopeEncodingKernel <<< outer, halfDim >>> (
            (__nv_bfloat16*)cudaData, cudaPositionIds, len, spatial, n, m, positionStride, rotaryDim,
            ropeTheta, factor, attentionFactor, correctionLow, correctionHigh);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlama3RopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                   float ropeTheta, float factor, float originalMaxPosition,
                                   float lowFreqFactor, float highFreqFactor) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    int halfDim = rotaryDim / 2;

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            (half*)cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            (__nv_bfloat16*)cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaQwen35InterleavedRope(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                      int sectionT, int sectionH, int sectionW,
                                      float ropeTheta, float ropeScale) {
    fastllm::AssertInFastLLM(data.dims.size() == 4, "Qwen3.5 interleaved RoPE expects [batch, seq, heads, dim] input.");
    fastllm::AssertInFastLLM(data.dims[0] == 1, "Qwen3.5 interleaved RoPE currently supports batch size 1 only.");
    fastllm::AssertInFastLLM(positionIds.dims.size() == 2 && positionIds.dims[0] == 3,
                             "Qwen3.5 interleaved RoPE expects position ids with shape [3, seq].");
    fastllm::AssertInFastLLM(sectionT + sectionH + sectionW == rotaryDim / 2,
                             "Qwen3.5 interleaved RoPE section sizes must sum to rotary_dim / 2.");

    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    int halfDim = rotaryDim / 2;
    int positionStride = (int) positionIds.dims.back();

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            (half*) cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            (__nv_bfloat16*) cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds) {
    float *inputData = (float *) input.cudaData;
    float *lognData = (float *) lognAttn.cudaData;
    float *posData = (float *) positionIds.cudaData;
    int batch = input.dims[0];
    int seqLen = input.dims[1];
    int spatial = input.Count(2);

    FastllmApplyLognAttnKernel <256> <<<batch * seqLen, 256>>> (inputData, lognData, posData, batch, seqLen, spatial);
    return true;
}

bool FastllmCudaRepeatPenalty (fastllm::Data &input, fastllm::Data &penalty, fastllm::Data &penaltyScale) {
    float *inputData = (float*)input.cudaData;
    float *penaltyData = (float*)penalty.cudaData;
    float *penaltyScaleData = (float*)penaltyScale.cudaData;
    int batch = penalty.dims[0], tokens = penalty.dims[1];
    int vocabs = input.dims.back();

    FastllmRepeatPenaltyKernel <64> <<<batch, 64>>> (inputData, penaltyData, penaltyScaleData, tokens, vocabs);
    return true;
}

template <int BLOCK_THREADS>
__global__ void FastllmTemperatureSoftmaxKernel(float *logits, float *probs, float *temperatures, int vocabSize) {
    int bid = blockIdx.x;
    float invTemp = 1.0f / temperatures[bid];
    float *input = logits + (long long)bid * vocabSize;
    float *output = probs + (long long)bid * vocabSize;

    __shared__ float sMaxVal;
    __shared__ float sSumExp;

    float localMax = -1e30f;
    for (int i = threadIdx.x; i < vocabSize; i += BLOCK_THREADS) {
        localMax = fmaxf(localMax, input[i] * invTemp);
    }
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    float blockMax = BlockReduce(tempStorage).Reduce(localMax, MaxReduceOp{});
    if (threadIdx.x == 0) sMaxVal = blockMax;
    __syncthreads();
    float maxVal = sMaxVal;

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < vocabSize; i += BLOCK_THREADS) {
        localSum += expf(input[i] * invTemp - maxVal);
    }
    __syncthreads();
    float blockSum = BlockReduce(tempStorage).Sum(localSum);
    if (threadIdx.x == 0) sSumExp = blockSum;
    __syncthreads();
    float sumExp = sSumExp;

    float invSum = 1.0f / sumExp;
    for (int i = threadIdx.x; i < vocabSize; i += BLOCK_THREADS) {
        output[i] = expf(input[i] * invTemp - maxVal) * invSum;
    }
}

template <int BLOCK_THREADS>
__global__ void FastllmTypicalAcceptanceKernel(
        const float *probs, const int *candidateIds,
        const int *candidateRows,
        const int *topKArr, const float *topPArr,
        unsigned char *accepted, int *recoveredIds,
        int vocabSize, float posteriorThreshold, float posteriorAlpha) {
    int candidateIndex = blockIdx.x;
    int rowIndex = candidateRows[candidateIndex];
    int tid = threadIdx.x;
    const float *row = probs + (long long)rowIndex * vocabSize;
    int candidateId = candidateIds[candidateIndex];
    float candidateProb = candidateId >= 0 && candidateId < vocabSize ?
                          row[candidateId] : 0.0f;

    float localEntropy = 0.0f;
    float localGreaterMass = 0.0f;
    float localGreaterCount = 0.0f;
    float localMax = -1.0f;
    int localMaxId = 0;
    for (int i = tid; i < vocabSize; i += BLOCK_THREADS) {
        float p = row[i];
        localEntropy -= p * logf(p + 1.0e-5f);
        if (p > candidateProb) {
            localGreaterMass += p;
            localGreaterCount += 1.0f;
        }
        if (p > localMax) {
            localMax = p;
            localMaxId = i;
        }
    }

    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    __shared__ float entropy;
    __shared__ float greaterMass;
    __shared__ float greaterCount;
    float blockEntropy = BlockReduce(reduceStorage).Sum(localEntropy);
    if (tid == 0) {
        entropy = blockEntropy;
    }
    __syncthreads();
    float blockGreaterMass = BlockReduce(reduceStorage).Sum(localGreaterMass);
    if (tid == 0) {
        greaterMass = blockGreaterMass;
    }
    __syncthreads();
    float blockGreaterCount = BlockReduce(reduceStorage).Sum(localGreaterCount);
    if (tid == 0) {
        greaterCount = blockGreaterCount;
    }
    __syncthreads();

    __shared__ float maxData[BLOCK_THREADS];
    __shared__ int maxIdData[BLOCK_THREADS];
    maxData[tid] = localMax;
    maxIdData[tid] = localMaxId;
    __syncthreads();
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride && maxData[tid] < maxData[tid + stride]) {
            maxData[tid] = maxData[tid + stride];
            maxIdData[tid] = maxIdData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // Medusa/vLLM typical acceptance: p(candidate) > min(epsilon, alpha * exp(-H)).
        float dynamicThreshold = fminf(
            posteriorThreshold, expf(-entropy) * posteriorAlpha);
        bool allowedByTopK = topKArr[rowIndex] <= 0 ||
                             greaterCount < topKArr[rowIndex];
        bool allowedByTopP = topPArr[rowIndex] >= 1.0f ||
                             greaterMass < topPArr[rowIndex];
        accepted[candidateIndex] = allowedByTopK && allowedByTopP &&
                                   candidateProb > dynamicThreshold ? 1 : 0;
        recoveredIds[candidateIndex] = maxIdData[0];
    }
}

bool FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
                                  float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize,
                                  const int *typicalCandidateIds,
                                  const int *typicalCandidateRows,
                                  unsigned char *typicalAccepted,
                                  int *typicalRecoveredIds,
                                  int typicalCount,
                                  float typicalPosteriorThreshold,
                                  float typicalPosteriorAlpha) {
    size_t probsBytes = (size_t)batch * vocabSize * sizeof(float);
    int actualTypicalCount = typicalCandidateIds != nullptr &&
                             typicalAccepted != nullptr &&
                             typicalRecoveredIds != nullptr ?
                             std::max(0, std::min(typicalCount, batch)) : 0;
    if (typicalCandidateRows != nullptr) {
        for (int i = 0; i < actualTypicalCount; i++) {
            if (typicalCandidateRows[i] < 0 ||
                typicalCandidateRows[i] >= batch) {
                return false;
            }
        }
    }

    // temperatures | top-k | top-p | sampled ids | candidates | candidate rows |
    // recovered ids | accepted flags | FlashInfer sampling-valid flags
    size_t paramBytes = batch * (sizeof(float) + sizeof(int) + sizeof(float) + sizeof(int)) +
                        actualTypicalCount * (sizeof(int) + sizeof(int) + sizeof(int) +
                                              sizeof(unsigned char)) +
                        batch * sizeof(bool);
    size_t alignedProbsBytes = FastllmCudaAlignBytes(probsBytes, 256);
    size_t alignedParamBytes = FastllmCudaAlignBytes(paramBytes, 256);
    size_t scratchBytes = 0;
    bool scratchOwn = false;
    uint8_t *scratch = (uint8_t *)FastllmBorrowDequantScratch(
        alignedProbsBytes + alignedParamBytes, &scratchBytes, &scratchOwn);
    if (scratch == nullptr || scratchBytes < alignedProbsBytes + alignedParamBytes) {
        FastllmReleaseDequantScratch(scratch, scratchOwn);
        printf("FastllmCudaTopKTopPSampling: failed to borrow CUDA temp buffer.\n");
        fflush(stdout);
        return false;
    }
    float *cudaProbs = (float *)scratch;
    uint8_t *cudaParamBuf = scratch + alignedProbsBytes;
    float *cudaTemperatures = (float *)(cudaParamBuf);
    int   *cudaTopKArr      = (int   *)(cudaParamBuf + batch * sizeof(float));
    float *cudaTopPArr      = (float *)(cudaParamBuf + batch * (sizeof(float) + sizeof(int)));
    int   *cudaOutput       = (int   *)(cudaParamBuf + batch * (sizeof(float) + sizeof(int) + sizeof(float)));
    int   *cudaTypicalCandidates = cudaOutput + batch;
    int   *cudaTypicalRows = cudaTypicalCandidates + actualTypicalCount;
    int   *cudaTypicalRecovered = cudaTypicalRows + actualTypicalCount;
    unsigned char *cudaTypicalAccepted =
        (unsigned char *)(cudaTypicalRecovered + actualTypicalCount);
    bool *cudaSamplingValid =
        (bool *)(cudaTypicalAccepted + actualTypicalCount);

    static thread_local std::vector<uint8_t> hostParamBuf;
    hostParamBuf.resize(batch * (sizeof(float) + sizeof(int) + sizeof(float)));
    memcpy(hostParamBuf.data(), temperatures, batch * sizeof(float));
    memcpy(hostParamBuf.data() + batch * sizeof(float), topKArr, batch * sizeof(int));
    memcpy(hostParamBuf.data() + batch * (sizeof(float) + sizeof(int)), topPArr, batch * sizeof(float));
    FastllmCudaCopyFromHostToDevice(cudaParamBuf, hostParamBuf.data(), hostParamBuf.size());
    if (actualTypicalCount > 0) {
        FastllmCudaCopyFromHostToDevice(cudaTypicalCandidates,
                                        (void*)typicalCandidateIds,
                                        actualTypicalCount * sizeof(int));
        if (typicalCandidateRows != nullptr) {
            FastllmCudaCopyFromHostToDevice(cudaTypicalRows,
                                            (void*)typicalCandidateRows,
                                            actualTypicalCount * sizeof(int));
        } else {
            static thread_local std::vector<int> identityTypicalRows;
            identityTypicalRows.resize(actualTypicalCount);
            for (int i = 0; i < actualTypicalCount; i++) {
                identityTypicalRows[i] = i;
            }
            FastllmCudaCopyFromHostToDevice(cudaTypicalRows,
                                            identityTypicalRows.data(),
                                            actualTypicalCount * sizeof(int));
        }
    }

    FastllmTemperatureSoftmaxKernel<1024><<<batch, 1024>>>(logits, cudaProbs, cudaTemperatures, vocabSize);
    if (actualTypicalCount > 0) {
        FastllmTypicalAcceptanceKernel<1024><<<actualTypicalCount, 1024>>>(
            cudaProbs, cudaTypicalCandidates, cudaTypicalRows,
            cudaTopKArr, cudaTopPArr,
            cudaTypicalAccepted, cudaTypicalRecovered, vocabSize,
            typicalPosteriorThreshold, typicalPosteriorAlpha);
    }

    static std::mt19937 rng(std::random_device{}());
    uint64_t seed = rng();

    cudaError_t samplingState =
        flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        cudaProbs, cudaTopKArr, cudaTopPArr, cudaOutput,
        cudaSamplingValid,
        (int *)nullptr,
        (uint32_t)batch, (int)0, 0.0f,
        (uint32_t)vocabSize, false,
        (uint64_t *)nullptr, seed,
        (uint64_t *)nullptr, 0, 0);
    if (samplingState != cudaSuccess) {
        FastllmReleaseDequantScratch(scratch, scratchOwn);
        printf("FastllmCudaTopKTopPSampling: launch failed: %s\n",
               cudaGetErrorString(samplingState));
        fflush(stdout);
        return false;
    }

    FastllmCudaCopyFromDeviceToHost(output, cudaOutput, batch * sizeof(int));
    if (actualTypicalCount > 0) {
        FastllmCudaCopyFromDeviceToHost(typicalAccepted, cudaTypicalAccepted,
                                        actualTypicalCount * sizeof(unsigned char));
        FastllmCudaCopyFromDeviceToHost(typicalRecoveredIds,
                                        cudaTypicalRecovered,
                                        actualTypicalCount * sizeof(int));
    }
    DeviceSync();

    FastllmReleaseDequantScratch(scratch, scratchOwn);
    return true;
}

template <int BLOCK_THREADS>
__global__ void FastllmRepeatPenaltyFactorsKernel(
        float *logits, const int *penaltyIds,
        const float *penaltyFactors, int penaltyTokens,
        int vocabSize) {
    int row = blockIdx.x;
    float *rowLogits = logits + (long long)row * vocabSize;
    const int *rowIds = penaltyIds + (long long)row * penaltyTokens;
    const float *rowFactors =
        penaltyFactors + (long long)row * penaltyTokens;
    for (int i = threadIdx.x; i < penaltyTokens; i += BLOCK_THREADS) {
        int token = rowIds[i];
        if (token >= 0 && token < vocabSize) {
            float factor = rowFactors[i];
            float value = rowLogits[token];
            rowLogits[token] = value < 0.0f ? value * factor : value / factor;
        }
    }
}

__global__ void FastllmSamplingIdsToFloatKernel(
        const int *input, float *output, int batch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch) {
        output[index] = (float)input[index];
    }
}

bool FastllmCudaTopKTopPSamplingToDevice(
                                  float *logits, float *probs,
                                  float *temperatures, int *topKArr,
                                  float *topPArr,
                                  int *penaltyIds, float *penaltyFactors,
                                  int penaltyTokens,
                                  int *output, float *floatOutput,
                                  int batch, int vocabSize) {
    if (logits == nullptr || probs == nullptr || temperatures == nullptr ||
        topKArr == nullptr || topPArr == nullptr || output == nullptr ||
        floatOutput == nullptr || batch <= 0 || vocabSize <= 0 ||
        penaltyTokens < 0 ||
        (penaltyTokens > 0 &&
         (penaltyIds == nullptr || penaltyFactors == nullptr))) {
        return false;
    }

    if (penaltyTokens > 0) {
        FastllmRepeatPenaltyFactorsKernel<64><<<batch, 64>>>(
            logits, penaltyIds, penaltyFactors,
            penaltyTokens, vocabSize);
    }
    FastllmTemperatureSoftmaxKernel<1024><<<batch, 1024>>>(
        logits, probs, temperatures, vocabSize);

    static thread_local std::mt19937 rng(std::random_device{}());
    uint64_t seed = rng();
    cudaError_t samplingState =
        flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        probs, topKArr, topPArr, output,
        (bool *)floatOutput,
        (int *)nullptr,
        (uint32_t)batch, (int)0, 0.0f,
        (uint32_t)vocabSize, false,
        (uint64_t *)nullptr, seed,
        (uint64_t *)nullptr, 0, 0);
    if (samplingState != cudaSuccess) {
        printf("FastllmCudaTopKTopPSamplingToDevice: launch failed: %s\n",
               cudaGetErrorString(samplingState));
        fflush(stdout);
        return false;
    }

    int threads = 256;
    FastllmSamplingIdsToFloatKernel<<<(batch + threads - 1) / threads,
                                      threads>>>(
        output, floatOutput, batch);
    cudaError_t state = cudaGetLastError();
    if (state != cudaSuccess) {
        printf("FastllmCudaTopKTopPSamplingToDevice: launch failed: %s\n",
               cudaGetErrorString(state));
        fflush(stdout);
        return false;
    }
    return true;
}

bool FastllmCudaTopKTopPSampling(float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize) {
    return FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
        logits, temperatures, topKArr, topPArr, output, batch, vocabSize,
        nullptr, nullptr, nullptr, nullptr, 0, 0.09f, 0.3f);
}

struct FastllmGreedyPartial {
    float value;
    int id;
};

template <int THREAD_PER_BLOCK>
__global__ void FastllmGreedySamplingKernel(float *logits, int *output,
                                            float *floatOutput, int vocabSize) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -INFINITY;
    int localId = 0;
    for (int i = tid; i < vocabSize; i += THREAD_PER_BLOCK) {
        float v = row[i];
        // Token IDs increase monotonically in this loop, so strict greater
        // already preserves the smallest ID for equal values in one lane.
        if (v > localMax) {
            localMax = v;
            localId = i;
        }
    }
    maxData[tid] = localMax;
    idData[tid] = localId;
    __syncthreads();

    for (int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && FastllmGreedyIsBetter(
                maxData[tid + s], idData[tid + s],
                maxData[tid], idData[tid])) {
            maxData[tid] = maxData[tid + s];
            idData[tid] = idData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = idData[0];
        if (floatOutput != nullptr) {
            floatOutput[b] = (float)idData[0];
        }
    }
}

__device__ __forceinline__ void FastllmGreedyWarpReduce(
        float &value, int &id) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float otherValue = __shfl_down_sync(0xffffffff, value, offset);
        int otherId = __shfl_down_sync(0xffffffff, id, offset);
        if (FastllmGreedyIsBetter(otherValue, otherId, value, id)) {
            value = otherValue;
            id = otherId;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGreedySamplingPartialKernel(
        const float *logits, FastllmGreedyPartial *partials,
        int vocabSize, int partCount) {
    int part = blockIdx.x;
    int batch = blockIdx.y;
    int chunk = (vocabSize + partCount - 1) / partCount;
    int start = part * chunk;
    int end = min(vocabSize, start + chunk);
    const float *row = logits + (int64_t)batch * vocabSize;

    float localMax = -INFINITY;
    int localId = start;
    for (int i = start + threadIdx.x; i < end;
         i += THREAD_PER_BLOCK) {
        float value = row[i];
        if (value > localMax) {
            localMax = value;
            localId = i;
        }
    }
    FastllmGreedyWarpReduce(localMax, localId);

    constexpr int WARP_COUNT = THREAD_PER_BLOCK / 32;
    __shared__ float warpMax[WARP_COUNT];
    __shared__ int warpId[WARP_COUNT];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warpMax[warp] = localMax;
        warpId[warp] = localId;
    }
    __syncthreads();
    if (warp == 0) {
        localMax = lane < WARP_COUNT ? warpMax[lane] : -INFINITY;
        localId = lane < WARP_COUNT ? warpId[lane] : vocabSize;
        FastllmGreedyWarpReduce(localMax, localId);
        if (lane == 0) {
            partials[(int64_t)batch * partCount + part] =
                {localMax, localId};
        }
    }
}

__global__ void FastllmGreedySamplingFinalizeKernel(
        const FastllmGreedyPartial *partials, int *output,
        float *floatOutput, int partCount) {
    int batch = blockIdx.x;
    int lane = threadIdx.x;
    float localMax = -INFINITY;
    int localId = 0;
    if (lane < partCount) {
        FastllmGreedyPartial partial =
            partials[(int64_t)batch * partCount + lane];
        localMax = partial.value;
        localId = partial.id;
    }
    FastllmGreedyWarpReduce(localMax, localId);
    if (lane == 0) {
        output[batch] = localId;
        if (floatOutput != nullptr) {
            floatOutput[batch] = (float)localId;
        }
    }
}

static bool FastllmLaunchGreedySampling(
        float *logits, int *output, float *floatOutput,
        int batch, int vocabSize) {
    if (vocabSize >= 32768 && batch <= 8) {
        int partCount = std::max(4, 32 / batch);
        size_t scratchBytes = (size_t)batch * partCount *
                              sizeof(FastllmGreedyPartial);
        size_t scratchCapacity = 0;
        bool scratchOwn = false;
        void *scratch = FastllmBorrowCudaTempBuffer(
            scratchBytes, &scratchCapacity, &scratchOwn);
        if (scratch != nullptr && scratchCapacity >= scratchBytes) {
            dim3 partialGrid((unsigned int)partCount,
                             (unsigned int)batch);
            FastllmGreedySamplingPartialKernel<256>
                <<<partialGrid, 256>>>(
                    logits, (FastllmGreedyPartial*)scratch,
                    vocabSize, partCount);
            FastllmGreedySamplingFinalizeKernel<<<batch, 32>>>(
                (const FastllmGreedyPartial*)scratch, output,
                floatOutput, partCount);
            FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
            return true;
        }
        FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
    }
    FastllmGreedySamplingKernel<256><<<batch, 256>>>(
        logits, output, floatOutput, vocabSize);
    return true;
}

bool FastllmCudaGreedySampling(float *logits, int *output,
                               int batch, int vocabSize) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || output == nullptr || vocabSize <= 0) {
        fastllm::ErrorInFastLLM("FastllmCudaGreedySampling: invalid input.\n");
        return false;
    }
    // Keep the public host-output path independent from the shared temporary
    // workspace.  The multi-CTA reduction below is reserved for the serialized
    // GPU token-handoff path, where the next decode step consumes floatOutput.
    FastllmGreedySamplingKernel<256><<<batch, 256>>>(
        logits, output, nullptr, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySampling: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaGreedySamplingWithFloatOutput(float *logits, int *output,
                                              float *floatOutput,
                                              int batch, int vocabSize) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || output == nullptr || floatOutput == nullptr ||
        vocabSize <= 0) {
        fastllm::ErrorInFastLLM(
            "FastllmCudaGreedySamplingWithFloatOutput: invalid input.\n");
        return false;
    }
    FastllmLaunchGreedySampling(logits, output, floatOutput,
                                batch, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySamplingWithFloatOutput: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

struct FastllmGreedyCandidate {
    int id;
    float score;
};

static_assert(sizeof(FastllmGreedyCandidate) == 2 * sizeof(int),
              "greedy candidate must stay compact for peer copies");

template <int THREAD_PER_BLOCK>
__global__ void FastllmGreedySamplingWithScoresKernel(float *logits, int *output,
                                                      float *scores,
                                                      FastllmGreedyCandidate *packed,
                                                      int vocabSize, int idOffset) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -INFINITY;
    int localId = 0;
    for (int i = tid; i < vocabSize; i += THREAD_PER_BLOCK) {
        float v = row[i];
        if (v > localMax) {
            localMax = v;
            localId = i;
        }
    }
    maxData[tid] = localMax;
    idData[tid] = localId;
    __syncthreads();

    for (int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = maxData[tid + s];
            int otherId = idData[tid + s];
            if (FastllmGreedyIsBetter(other, otherId,
                                      maxData[tid], idData[tid])) {
                maxData[tid] = other;
                idData[tid] = otherId;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int id = idData[0] + idOffset;
        if (packed != nullptr) {
            packed[b] = {id, maxData[0]};
        } else {
            output[b] = id;
            scores[b] = maxData[0];
        }
    }
}

bool FastllmCudaGreedySamplingWithScores(float *logits, int *output,
                                         float *scores, int batch,
                                         int vocabSize) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || output == nullptr || scores == nullptr ||
        vocabSize <= 0) {
        fastllm::ErrorInFastLLM("FastllmCudaGreedySamplingWithScores: invalid input.\n");
        return false;
    }
    FastllmGreedySamplingWithScoresKernel<256><<<batch, 256>>>(
        logits, output, scores, nullptr, vocabSize, 0);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySamplingWithScores: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaGreedySamplingPackedCandidateWithIdOffset(
        float *logits, void *packedCandidates, int batch,
        int vocabSize, int idOffset) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || packedCandidates == nullptr ||
        vocabSize <= 0 || idOffset < 0) {
        fastllm::ErrorInFastLLM(
            "FastllmCudaGreedySamplingPackedCandidateWithIdOffset: invalid input.\n");
        return false;
    }
    FastllmGreedySamplingWithScoresKernel<256><<<batch, 256>>>(
        logits, nullptr, nullptr,
        (FastllmGreedyCandidate*)packedCandidates,
        vocabSize, idOffset);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySamplingPackedCandidateWithIdOffset: "
               "kernel launch failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void FastllmMergeShardedGreedyCandidatesKernel(
        const FastllmGreedyCandidate *candidates,
        int *output, float *floatOutput, int ranks, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    int bestId = 0;
    float bestScore = -INFINITY;
    for (int r = 0; r < ranks; r++) {
        int index = r * batch + b;
        int id = candidates[index].id;
        float score = candidates[index].score;
        if (r == 0 || score > bestScore ||
            (score == bestScore && id < bestId)) {
            bestId = id;
            bestScore = score;
        }
    }
    output[b] = bestId;
    floatOutput[b] = (float)bestId;
}

bool FastllmCudaMergeShardedGreedyCandidates(
        const void *packedCandidates,
        int *output, float *floatOutput, int ranks, int batch) {
    if (batch <= 0) {
        return true;
    }
    if (packedCandidates == nullptr ||
        output == nullptr || floatOutput == nullptr || ranks <= 0) {
        fastllm::ErrorInFastLLM(
            "FastllmCudaMergeShardedGreedyCandidates: invalid input.\n");
        return false;
    }
    int threads = 128;
    int blocks = (batch + threads - 1) / threads;
    FastllmMergeShardedGreedyCandidatesKernel<<<blocks, threads>>>(
        (const FastllmGreedyCandidate*)packedCandidates,
        output, floatOutput, ranks, batch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaMergeShardedGreedyCandidates: "
               "kernel launch failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void FastllmSampleTopKKernel(float *topk, float *temperatures,
                                        int *topKArr, float *topPArr,
                                        float *randoms, int *output,
                                        int maxTopK) {
    int b = blockIdx.x;
    float *base = topk + (long long)b * maxTopK * 2;
    int curTopK = topKArr[b];
    if (curTopK <= 1 || maxTopK <= 1 || temperatures[b] <= 1.0e-6f) {
        output[b] = (int)(base[0] + 1.0e-3f);
        return;
    }
    curTopK = min(curTopK, maxTopK);

    float topP = topPArr[b];
    float invTemp = 1.0f / temperatures[b];
    float maxValue = base[1] * invTemp;
    for (int i = 1; i < curTopK; i++) {
        maxValue = max(maxValue, base[i * 2 + 1] * invTemp);
    }

    float sum = 0.0f;
    for (int i = 0; i < curTopK; i++) {
        sum += expf(base[i * 2 + 1] * invTemp - maxValue);
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        output[b] = (int)(base[0] + 1.0e-3f);
        return;
    }

    float cutoffSum = 0.0f;
    int cutoff = curTopK;
    for (int i = 0; i < curTopK; i++) {
        cutoffSum += expf(base[i * 2 + 1] * invTemp - maxValue) / sum;
        if (cutoffSum > topP) {
            cutoff = i + 1;
            break;
        }
    }
    cutoffSum = max(cutoffSum, 1.0e-20f);

    float rnd = randoms[b];
    rnd = min(max(rnd, 0.0f), 0.99999994f) * cutoffSum;
    float curSum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        curSum += expf(base[i * 2 + 1] * invTemp - maxValue) / sum;
        if (curSum > rnd || i == cutoff - 1) {
            output[b] = (int)(base[i * 2] + 1.0e-3f);
            return;
        }
    }
    output[b] = (int)(base[0] + 1.0e-3f);
}

bool FastllmCudaSampleTopK(float *topk, float *temperatures,
                           int *topKArr, float *topPArr, float *randoms,
                           int *output,
                           int batch, int maxTopK) {
    FastllmSampleTopKKernel<<<batch, 1>>>(topk, temperatures, topKArr, topPArr,
                                          randoms, output, maxTopK);
    return true;
}

bool FastllmCudaSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis) {
    int part = input.dims[axis];
    int outer = input.Count(0) / input.Count(axis);
    int inputStride = input.Count(axis);
    int outputStride = outputs[0]->Count(axis);
    int inner = input.strides[axis];
    int unitSize = input.unitSize;

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)outputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, cudaMemcpyHostToDevice);
    FastllmSplitBatchKernel <256> <<< part * outer, 256 >>> ((uint8_t*)input.cudaData, pointers, outer, part, inner * unitSize);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis) {
    int part = output.dims[axis];
    int outer = output.Count(0) / output.Count(axis);
    int inputStride = inputs[0]->Count(axis);
    int outputStride = output.Count(axis);
    int inner = output.strides[axis];
    int unitSize = output.unitSize;

    // CatBatch is commonly placed between operators running on
    // cudaStreamPerThread (FlashInfer, cuBLAS and NCCL).  Launching this copy
    // on the legacy default stream can race both its producers and consumers.
    // Keep a small per-thread/per-device pointer table alive so it also cannot
    // be returned to FastLLM's caching allocator while the kernel is in flight.
    thread_local static std::map<int, std::pair<uint8_t**, int> > pointerBuffers;
    int device = FastllmCudaGetDevice();
    auto &pointerBuffer = pointerBuffers[device];
    if (pointerBuffer.second < part) {
        if (pointerBuffer.first != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(pointerBuffer.first);
        }
        pointerBuffer.first = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * part);
        pointerBuffer.second = part;
    }
    uint8_t **pointers = pointerBuffer.first;
    uint8_t **cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)inputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, cudaMemcpyHostToDevice);
    FastllmCatBatchKernel <256> <<< part * outer, 256, 0, cudaStreamPerThread >>> (
        pointers, (uint8_t*)output.cudaData, outer, part, inner * unitSize);

    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs) {
    float ** pointers = (float**)FastllmCudaMalloc(sizeof(float*) * batch * 3);
    float ** cpuPointers = new float*[batch * 3];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = (float*)inputs[i]->cudaData;
        cpuPointers[i + batch] = (float*)outputs[i]->cudaData;
        cpuPointers[i + batch * 2] = (float*)(inputs[i]->Count(0));
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(float*) * batch * 3, cudaMemcpyHostToDevice);
    FastllmMulBatchKernel <256> <<< batch, 256 >>> (pointers, batch, v);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

__global__ void FastllmCudaNaiveConv2DKernel(float *input, float *weight, float *bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inputHeight, int inputWidth, int outputHeight, int outputWidth, float *output) {
    int oc = blockIdx.x;
    output += oc * outputHeight * outputWidth;
    {
        float *startWeight = weight + oc * (inputChannels * kernelH * kernelW);
        for (int t = threadIdx.x; t < outputHeight * outputWidth; t += blockDim.x) {
            int oh = t / outputWidth;
            int ow = t % outputWidth;

            int ih = oh * strideH - padH;
            int iw = ow * strideW - padW;
            float value = bias[oc];
            float *curWeight = startWeight;
            for (int c = 0; c < inputChannels; c++) {
                float *curInput = (float*)input + c * inputHeight * inputWidth;
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        float inputValue = 0;
                        if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                            inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                        }
                        value += inputValue * (*(curWeight++));
                    }
                }
            }
            output[oh * outputWidth + ow] = value;
        }
    }
}

__global__ void FastllmCudaNaiveConv2DHalfKernel(float *input, half *weight, float *bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inputHeight, int inputWidth, int outputHeight, int outputWidth, float *output) {
    int oc = blockIdx.x;
    output += oc * outputHeight * outputWidth;
    {
        half *startWeight = weight + oc * (inputChannels * kernelH * kernelW);
        for (int t = threadIdx.x; t < outputHeight * outputWidth; t += blockDim.x) {
            int oh = t / outputWidth;
            int ow = t % outputWidth;

            int ih = oh * strideH - padH;
            int iw = ow * strideW - padW;
            float value = bias[oc];
            half *curWeight = startWeight;
            for (int c = 0; c < inputChannels; c++) {
                float *curInput = (float*)input + c * inputHeight * inputWidth;
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        float inputValue = 0;
                        if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                            inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                        }
                        value += inputValue * __half2float(*(curWeight++));
                    }
                }
            }
            output[oh * outputWidth + ow] = value;
        }
    }
}

template <typename T>
// CUDA kernel for per-channel 1D convolution
__global__ void Conv1DPerChannelKernel(
    T* input,
    const float* weight,
    const float* bias,
    T* output,
    int batchSize,
    int inputChannels,
    int outputChannels,
    int inputLength,
    int outputLength,
    int kernelSize,
    int stride,
    int padding,
    int groups) {
    
    // 计算当前线程处理的位置
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * outputChannels * outputLength;
    
    if (tid >= totalElements) return;
    
    // 解析输出位置 (batch, channel, position)
    int ol = tid % outputLength;
    int oc = (tid / outputLength) % outputChannels;
    int b = tid / (outputChannels * outputLength);
    
    // 对于逐通道卷积，每个输出通道对应一个输入通道
    int g = oc;  // group index (因为 groups = inputChannels)
    int ic = g;  // 对应的输入通道
    
    // 计算输入起始位置
    int il_start = ol * stride - padding;
    
    // 初始化输出值（加上bias）
    float value = bias ? bias[oc] : 0.0f;
    
    // 获取权重和输入的指针
    const float* curWeight = weight + oc * kernelSize;
    const T* curInput = input + b * inputChannels * inputLength + ic * inputLength;
    
    // 执行卷积
    #pragma unroll
    for (int k = 0; k < kernelSize; k++) {
        int inputPos = il_start + k;
        
        // 边界检查
        if (inputPos >= 0 && inputPos < inputLength) {
            value += (float)curInput[inputPos] * curWeight[k];
        }
    }
    
    // 写入输出
    output[tid] = (T)value;
}

// 主函数
bool FastllmCudaConv1DPerChannelFloat32(
    const fastllm::Data &input, 
    fastllm::Data &weight, 
    fastllm::Data &bias, 
    int inputChannels, 
    int outputChannels, 
    int kernelSize, 
    int stride, 
    int padding, 
    fastllm::Data &output) {
    
    int groups = inputChannels;
    std::vector<int> dims = input.dims;
    int batchSize = dims[0];
    int inputLength = dims[2];
    int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;
    
    // 准备输出维度
    output.Resize({batchSize, outputChannels, outputLength});
    
    // 获取设备指针
    float *d_input = (float*)input.cudaData;
    float *d_weight = (float*)weight.cudaData;
    float *d_bias = bias.dims.size() > 0 ? (float*)bias.cudaData : nullptr;
    float *d_output = (float*)output.cudaData;
    
    // 配置kernel参数
    int totalElements = batchSize * outputChannels * outputLength;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    if (input.dataType == fastllm::DataType::FLOAT32) {
        Conv1DPerChannelKernel<float> <<<blocksPerGrid, threadsPerBlock>>>(
                d_input, d_weight, d_bias, d_output,
                batchSize, inputChannels, outputChannels,
                inputLength, outputLength,
                kernelSize, stride, padding, groups
        );
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Conv1DPerChannelKernel<half> <<<blocksPerGrid, threadsPerBlock>>>(
                (half*)d_input, d_weight, d_bias, (half*)d_output,
                batchSize, inputChannels, outputChannels,
                inputLength, outputLength,
                kernelSize, stride, padding, groups
        );
    }

    DeviceSync();
    return true;
}

__global__ void FastllmConv1DPerChannelSiluSingleTokenHalfKernel(const half *input, const float *weight,
                                                                 const float *bias, half *output, int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    const half *curInput = input + c * 4;
    const float *curWeight = weight + c * 4;
    float value = bias ? bias[c] : 0.0f;
    value += __half2float(curInput[0]) * curWeight[0];
    value += __half2float(curInput[1]) * curWeight[1];
    value += __half2float(curInput[2]) * curWeight[2];
    value += __half2float(curInput[3]) * curWeight[3];

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[c] = __float2half(x / (1.0f + expf(-x)));
#else
    output[c] = __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfKernel(
    half *cache, const half *newToken, const float *weight, const float *bias,
    half *output, int batch, int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int c = row % channels;
    half *cacheRow = cache + row * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[row] = __float2half(x / (1.0f + expf(-x)));
#else
    output[row] = __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluTwoTokenHalfKernel(
    half *cache, const half *newTokens, const float *weight, const float *bias,
    half *output, half *firstTokenCache, int batch, int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int c = row % channels;
    half *cacheRow = cache + row * 4;
    const half *tokenRow = newTokens + row * 2;
    half t0 = tokenRow[0];
    half t1 = tokenRow[1];
    half x1 = cacheRow[1];
    half x2 = cacheRow[2];
    half x3 = cacheRow[3];

    if (firstTokenCache) {
        half *firstRow = firstTokenCache + row * 4;
        firstRow[0] = x1;
        firstRow[1] = x2;
        firstRow[2] = x3;
        firstRow[3] = t0;
    }

    cacheRow[0] = x2;
    cacheRow[1] = x3;
    cacheRow[2] = t0;
    cacheRow[3] = t1;

    const float *curWeight = weight + c * 4;
    float value0 = bias ? bias[c] : 0.0f;
    value0 += __half2float(x1) * curWeight[0];
    value0 += __half2float(x2) * curWeight[1];
    value0 += __half2float(x3) * curWeight[2];
    value0 += __half2float(t0) * curWeight[3];

    float value1 = bias ? bias[c] : 0.0f;
    value1 += __half2float(x2) * curWeight[0];
    value1 += __half2float(x3) * curWeight[1];
    value1 += __half2float(t0) * curWeight[2];
    value1 += __half2float(t1) * curWeight[3];

    half conv0 = __float2half_rn(value0);
    half conv1 = __float2half_rn(value1);
    half *outputRow = output + row * 2;
#ifdef CUDA_NO_TENSOR_CORE
    float y0 = __half2float(conv0);
    float y1 = __half2float(conv1);
    outputRow[0] = __float2half(y0 / (1.0f + expf(-y0)));
    outputRow[1] = __float2half(y1 / (1.0f + expf(-y1)));
#else
    outputRow[0] = __hdiv(conv0, __hadd(__float2half(1.0f), hexp(-conv0)));
    outputRow[1] = __hdiv(conv1, __hadd(__float2half(1.0f), hexp(-conv1)));
#endif
}

static constexpr int FASTLLM_CUDA_MTP_FAST_SEQ_MAX = 6;
static constexpr int FASTLLM_CUDA_MTP_PREFIX_SNAPSHOT_MAX =
    FASTLLM_CUDA_MTP_FAST_SEQ_MAX - 1;
// Ordinary batched prefill does not materialize per-token MTP snapshots.  Its
// conv cache update can therefore reuse the pointer kernel for a much longer
// sequence without growing the fixed snapshot-pointer array.
static constexpr int FASTLLM_CUDA_BATCH_PREFILL_SEQ_MAX = 4096;

static bool FastllmCudaDataHasDenseStrides(const fastllm::Data &data) {
    if (data.dims.empty() || data.strides.size() != data.dims.size()) {
        return false;
    }
    uint64_t expected = 1;
    for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
        if (data.dims[i] < 0 || data.strides[i] != expected) {
            return false;
        }
        if (data.dims[i] > 0 && expected > ~(uint64_t)0 / (uint64_t)data.dims[i]) {
            return false;
        }
        expected *= (uint64_t)data.dims[i];
    }
    return true;
}

static bool FastllmCudaResolveDataDeviceId(const fastllm::Data &data,
                                            int &deviceId) {
    if (data.dataDevice != fastllm::DataDevice::CUDA ||
        data.dataDeviceIds.size() > 1) {
        return false;
    }
    if (data.cudaData == nullptr) {
        return false;
    }
    // Fake/view tensors may omit dataDeviceIds, and a reused view may retain a
    // stale id from its previous owner.  Resolve the allocation itself and,
    // when metadata is present, require both sources to agree.
    int pointerDevice = GetPointerDeviceId(data.cudaData);
    if (pointerDevice < 0 ||
        (!data.dataDeviceIds.empty() && data.dataDeviceIds[0] != pointerDevice)) {
        return false;
    }
    deviceId = pointerDevice;
    return true;
}

static bool FastllmCudaDataCanShareDevice(const fastllm::Data &reference,
                                          const fastllm::Data &other) {
    if (reference.multiDeviceData || other.multiDeviceData ||
        reference.dataDevice != fastllm::DataDevice::CUDA ||
        other.dataDevice != fastllm::DataDevice::CUDA ||
        reference.dataDeviceIds.size() > 1 || other.dataDeviceIds.size() > 1) {
        return false;
    }
    int referenceDevice = -1;
    int otherDevice = -1;
    return FastllmCudaResolveDataDeviceId(reference, referenceDevice) &&
           FastllmCudaResolveDataDeviceId(other, otherDevice) &&
           referenceDevice == otherDevice;
}

// Batched linear-attention kernels consume small arrays of request-local
// cache pointers. Keep one device staging buffer per TP worker thread so the
// hot decode path only enqueues a tiny H2D copy instead of allocating, copying
// synchronously (which waits for all earlier work), and freeing on every
// layer. The std::vector storage is pageable, so the CUDA runtime stages the
// source before this function returns; the device copy and its consumer stay
// ordered on cudaStreamPerThread.
static void **FastllmCudaStagePointers(const std::vector<void*> &pointers) {
    struct PointerScratch {
        void *data = nullptr;
        size_t capacity = 0;
        int device = -1;

        ~PointerScratch() {
            if (data == nullptr || device < 0) {
                return;
            }
            int oldDevice = 0;
            cudaGetDevice(&oldDevice);
            cudaSetDevice(device);
            FastllmCudaFree(data);
            cudaSetDevice(oldDevice);
        }
    };
    static thread_local PointerScratch scratch;
    if (pointers.empty()) {
        return nullptr;
    }
    int device = FastllmCudaGetDevice();
    size_t bytes = pointers.size() * sizeof(void*);
    if (scratch.device != device || scratch.capacity < bytes) {
        if (scratch.data != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(scratch.data);
        }
        scratch.data = FastllmCudaMalloc(bytes);
        scratch.capacity = bytes;
        scratch.device = device;
    }
    checkCudaErrors(
        "Error: CUDA error when staging batched cache pointers.",
        cudaMemcpyAsync(scratch.data, pointers.data(), bytes,
                        cudaMemcpyHostToDevice, cudaStreamPerThread));
    return (void**)scratch.data;
}

bool FastllmCudaGetRaggedGdnMetadata(
        const std::vector<int> &seqLens, int chunkSize,
        FastllmCudaRaggedGdnMetadataView &view) {
    struct RaggedMetadataCache {
        int *data = nullptr;
        size_t capacity = 0;
        int device = -1;
        int chunkSize = 0;
        std::vector<int> seqLens;
        FastllmCudaRaggedGdnMetadataView view;

        ~RaggedMetadataCache() {
            if (data == nullptr || device < 0) {
                return;
            }
            int oldDevice = 0;
            cudaGetDevice(&oldDevice);
            cudaSetDevice(device);
            FastllmCudaFree(data);
            cudaSetDevice(oldDevice);
        }
    };
    static thread_local RaggedMetadataCache cache;
    view = FastllmCudaRaggedGdnMetadataView();
    if (seqLens.empty() || chunkSize <= 0) {
        return false;
    }
    int device = FastllmCudaGetDevice();
    if (cache.data != nullptr && cache.device == device &&
        cache.chunkSize == chunkSize && cache.seqLens == seqLens) {
        view = cache.view;
        return true;
    }

    const int batch = (int)seqLens.size();
    std::vector<int> tokenOffsets(batch + 1, 0);
    std::vector<int> chunkOffsets(batch + 1, 0);
    int maxChunks = 0;
    int maxPaddedTokens = 0;
    for (int request = 0; request < batch; request++) {
        const int len = seqLens[request];
        if (len <= 0 || tokenOffsets[request] >
                std::numeric_limits<int>::max() - len) {
            return false;
        }
        const int chunks = (len + chunkSize - 1) / chunkSize;
        if (chunkOffsets[request] >
                std::numeric_limits<int>::max() - chunks) {
            return false;
        }
        tokenOffsets[request + 1] = tokenOffsets[request] + len;
        chunkOffsets[request + 1] = chunkOffsets[request] + chunks;
        maxChunks = std::max(maxChunks, chunks);
        maxPaddedTokens = std::max(maxPaddedTokens, chunks * chunkSize);
    }
    const int totalChunks = chunkOffsets.back();
    std::vector<int> metadata;
    metadata.reserve((batch + 1) * 2 + totalChunks * 2);
    metadata.insert(metadata.end(), tokenOffsets.begin(), tokenOffsets.end());
    metadata.insert(metadata.end(), chunkOffsets.begin(), chunkOffsets.end());
    for (int request = 0; request < batch; request++) {
        int token = tokenOffsets[request];
        for (int chunk = chunkOffsets[request];
             chunk < chunkOffsets[request + 1]; chunk++) {
            metadata.push_back(token);
            token += chunkSize;
        }
    }
    for (int request = 0; request < batch; request++) {
        int remaining = seqLens[request];
        for (int chunk = chunkOffsets[request];
             chunk < chunkOffsets[request + 1]; chunk++) {
            metadata.push_back(std::min(chunkSize, remaining));
            remaining -= chunkSize;
        }
    }

    const size_t bytes = metadata.size() * sizeof(int);
    if (cache.device != device || cache.capacity < bytes) {
        if (cache.data != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(cache.data);
        }
        cache.data = (int*)FastllmCudaMalloc(bytes);
        if (cache.data == nullptr) {
            cache = RaggedMetadataCache();
            return false;
        }
        cache.capacity = bytes;
        cache.device = device;
    }
    cudaError_t copyState = cudaMemcpyAsync(
        cache.data, metadata.data(), bytes, cudaMemcpyHostToDevice,
        cudaStreamPerThread);
    if (copyState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error caching ragged GDN metadata.", copyState);
        return false;
    }
    cache.chunkSize = chunkSize;
    cache.seqLens = seqLens;
    cache.view.tokenOffsets = cache.data;
    cache.view.chunkOffsets = cache.data + batch + 1;
    cache.view.chunkTokenBases = cache.data + (batch + 1) * 2;
    cache.view.chunkValidTokens =
        cache.view.chunkTokenBases + totalChunks;
    cache.view.batch = batch;
    cache.view.totalTokens = tokenOffsets.back();
    cache.view.totalChunks = totalChunks;
    cache.view.maxChunks = maxChunks;
    cache.view.maxPaddedTokens = maxPaddedTokens;
    view = cache.view;
    return true;
}

// 处理最多6个新token的conv1d(kernel=4)滑窗更新, 并在处理完第t个token后
// 把当时的滑窗状态写入对应快照 (用于MTP验证的逐token状态回滚)。
__global__ void FastllmShiftAppendConv1DPerChannelSiluMultiTokenHalfKernel(
    half *cache, const half *newTokens, const float *weight, const float *bias,
    half *output,
    half *snap0, half *snap1, half *snap2, half *snap3, half *snap4, half *snap5,
    int numSnaps,
    int batch, int channels, int numTokens) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int c = row % channels;
    half *cacheRow = cache + (size_t)row * 4;
    const half *tokenRow = newTokens + (size_t)row * numTokens;
    half x0 = cacheRow[0];
    half x1 = cacheRow[1];
    half x2 = cacheRow[2];
    half x3 = cacheRow[3];

    const float *curWeight = weight + (size_t)c * 4;
    float biasVal = bias ? bias[c] : 0.0f;
    half *outputRow = output + (size_t)row * numTokens;
    for (int t = 0; t < numTokens; t++) {
        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = tokenRow[t];
        float value = biasVal;
        value += __half2float(x0) * curWeight[0];
        value += __half2float(x1) * curWeight[1];
        value += __half2float(x2) * curWeight[2];
        value += __half2float(x3) * curWeight[3];
        half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
        float y = __half2float(conv);
        outputRow[t] = __float2half(y / (1.0f + expf(-y)));
#else
        outputRow[t] = __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
        half *snapBase = nullptr;
        if (t < numSnaps) {
            switch (t) {
                case 0: snapBase = snap0; break;
                case 1: snapBase = snap1; break;
                case 2: snapBase = snap2; break;
                case 3: snapBase = snap3; break;
                case 4: snapBase = snap4; break;
                case 5: snapBase = snap5; break;
                default: break;
            }
        }
        if (snapBase != nullptr) {
            half *snapRow = snapBase + (size_t)row * 4;
            snapRow[0] = x0;
            snapRow[1] = x1;
            snapRow[2] = x2;
            snapRow[3] = x3;
        }
    }

    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluMultiTokenHalfPointerKernel(
    half **pointers, const half *newTokens, const float *weight,
    const float *bias, half *output, int batch, int channels,
    int numTokens, int numSnaps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }
    int batchIndex = row / channels;
    int channel = row - batchIndex * channels;
    half *cacheRow = pointers[batchIndex] + (size_t)channel * 4;
    const half *tokenRow = newTokens + (size_t)row * numTokens;
    half x0 = cacheRow[0];
    half x1 = cacheRow[1];
    half x2 = cacheRow[2];
    half x3 = cacheRow[3];
    const float *curWeight = weight + (size_t)channel * 4;
    float biasValue = bias == nullptr ? 0.0f : bias[channel];
    half *outputRow = output + (size_t)row * numTokens;
    for (int token = 0; token < numTokens; token++) {
        x0 = x1;
        x1 = x2;
        x2 = x3;
        x3 = tokenRow[token];
        float value = biasValue;
        value += __half2float(x0) * curWeight[0];
        value += __half2float(x1) * curWeight[1];
        value += __half2float(x2) * curWeight[2];
        value += __half2float(x3) * curWeight[3];
        half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
        float y = __half2float(conv);
        outputRow[token] = __float2half(y / (1.0f + expf(-y)));
#else
        outputRow[token] =
            __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
        if (token < numSnaps) {
            half *snapshot = pointers[batch + batchIndex * numSnaps + token];
            if (snapshot != nullptr) {
                half *snapshotRow = snapshot + (size_t)channel * 4;
                snapshotRow[0] = x0;
                snapshotRow[1] = x1;
                snapshotRow[2] = x2;
                snapshotRow[3] = x3;
            }
        }
    }
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluPrefillHalfPointerKernel(
    half **pointers, const half *newTokens, const float *weight,
    const float *bias, half *output, int channels, int numTokens,
    size_t totalElements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalElements) {
        return;
    }

    int token = index % numTokens;
    size_t row = index / numTokens;
    int batchIndex = row / channels;
    int channel = row - (size_t)batchIndex * channels;
    const half *cacheRow =
        pointers[batchIndex] + (size_t)channel * 4;
    const half *tokenRow = newTokens + row * numTokens;
    const float *curWeight = weight + (size_t)channel * 4;

    float value = bias == nullptr ? 0.0f : bias[channel];
#pragma unroll
    for (int kernelIndex = 0; kernelIndex < 4; kernelIndex++) {
        int sourceToken = token + kernelIndex - 3;
        half source = sourceToken < 0
            ? cacheRow[sourceToken + 4]
            : tokenRow[sourceToken];
        value += __half2float(source) * curWeight[kernelIndex];
    }

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[index] = __float2half(x / (1.0f + expf(-x)));
#else
    output[index] =
        __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

__global__ void FastllmShiftAppendConv1DPerChannelPrefillCacheHalfPointerKernel(
    half **pointers, const half *newTokens, int channels, int numTokens,
    size_t totalRows) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= totalRows) {
        return;
    }

    int batchIndex = row / channels;
    int channel = row - (size_t)batchIndex * channels;
    half *cacheRow = pointers[batchIndex] + (size_t)channel * 4;
    const half *tokenRow = newTokens + row * numTokens;
    half oldCache[4] = {
        cacheRow[0], cacheRow[1], cacheRow[2], cacheRow[3]
    };
#pragma unroll
    for (int cacheIndex = 0; cacheIndex < 4; cacheIndex++) {
        int sourceToken = numTokens - 4 + cacheIndex;
        cacheRow[cacheIndex] = sourceToken < 0
            ? oldCache[sourceToken + 4]
            : tokenRow[sourceToken];
    }
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerKernel(
    half **pointers, const half *newTokens, const float *weight,
    const float *bias, half *output, int channels, int numTokens,
    int inputStride, int inputOffset, size_t totalElements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalElements) {
        return;
    }

    int channel = index % channels;
    size_t tokenRow = index / channels;
    int token = tokenRow % numTokens;
    int batchIndex = tokenRow / numTokens;
    const half *cacheRow =
        pointers[batchIndex] + (size_t)channel * 4;
    const float *curWeight = weight + (size_t)channel * 4;

    float value = bias == nullptr ? 0.0f : bias[channel];
#pragma unroll
    for (int kernelIndex = 0; kernelIndex < 4; kernelIndex++) {
        int sourceToken = token + kernelIndex - 3;
        half source = sourceToken < 0
            ? cacheRow[sourceToken + 4]
            : newTokens[
                ((size_t)batchIndex * numTokens + sourceToken) *
                    inputStride + inputOffset + channel];
        value += __half2float(source) * curWeight[kernelIndex];
    }

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[index] = __float2half(x / (1.0f + expf(-x)));
#else
    output[index] =
        __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

template <int TOKENS_PER_GROUP>
__global__ void FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerGroupedKernel(
    half **pointers, const half *__restrict__ newTokens,
    const float *__restrict__ weight, const float *__restrict__ bias,
    half *__restrict__ output, int channels, int numTokens,
    int inputStride, int inputOffset, size_t totalGroups) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalGroups) {
        return;
    }

    int channel = index % channels;
    size_t groupRow = index / channels;
    int tokenGroup = groupRow % (numTokens / TOKENS_PER_GROUP);
    int batchIndex = groupRow / (numTokens / TOKENS_PER_GROUP);
    int tokenBase = tokenGroup * TOKENS_PER_GROUP;
    const half *cacheRow =
        pointers[batchIndex] + (size_t)channel * 4;
    const float *curWeight = weight + (size_t)channel * 4;

    half source[TOKENS_PER_GROUP + 3];
#pragma unroll
    for (int sourceIndex = 0;
         sourceIndex < TOKENS_PER_GROUP + 3; sourceIndex++) {
        int sourceToken = tokenBase + sourceIndex - 3;
        source[sourceIndex] = sourceToken < 0
            ? cacheRow[sourceToken + 4]
            : newTokens[
                ((size_t)batchIndex * numTokens + sourceToken) *
                    inputStride + inputOffset + channel];
    }

    float biasValue = bias == nullptr ? 0.0f : bias[channel];
#pragma unroll
    for (int tokenOffset = 0;
         tokenOffset < TOKENS_PER_GROUP; tokenOffset++) {
        float value = biasValue;
#pragma unroll
        for (int kernelIndex = 0; kernelIndex < 4; kernelIndex++) {
            value += __half2float(source[tokenOffset + kernelIndex]) *
                     curWeight[kernelIndex];
        }
        half conv = __float2half_rn(value);
        size_t outputIndex =
            ((size_t)batchIndex * numTokens + tokenBase + tokenOffset) *
                channels + channel;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(conv);
        output[outputIndex] = __float2half(x / (1.0f + expf(-x)));
#else
        output[outputIndex] =
            __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
    }
}

__global__ void FastllmShiftAppendConv1DPerChannelPrefillCacheTokenMajorHalfPointerKernel(
    half **pointers, const half *newTokens, int channels,
    int numTokens, int inputStride, int inputOffset,
    size_t totalRows) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= totalRows) {
        return;
    }

    int batchIndex = row / channels;
    int channel = row - (size_t)batchIndex * channels;
    half *cacheRow = pointers[batchIndex] + (size_t)channel * 4;
    half oldCache[4] = {
        cacheRow[0], cacheRow[1], cacheRow[2], cacheRow[3]
    };
#pragma unroll
    for (int cacheIndex = 0; cacheIndex < 4; cacheIndex++) {
        int sourceToken = numTokens - 4 + cacheIndex;
        cacheRow[cacheIndex] = sourceToken < 0
            ? oldCache[sourceToken + 4]
            : newTokens[
                ((size_t)batchIndex * numTokens + sourceToken) *
                    inputStride + inputOffset + channel];
    }
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluRaggedPrefillHalfPointerKernel(
    half **pointers, const int *tokenOffsets, const half *newTokens,
    const float *weight, const float *bias, half *output,
    int channels) {
    int batchIndex = blockIdx.y;
    int begin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - begin;
    size_t localIndex = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t localElements = (size_t)numTokens * channels;
    if (localIndex >= localElements) {
        return;
    }

    int channel = localIndex % channels;
    int token = localIndex / channels;
    const half *cacheRow = pointers[batchIndex] + (size_t)channel * 4;
    const float *curWeight = weight + (size_t)channel * 4;

    float value = bias == nullptr ? 0.0f : bias[channel];
#pragma unroll
    for (int kernelIndex = 0; kernelIndex < 4; kernelIndex++) {
        int sourceToken = token + kernelIndex - 3;
        half source = sourceToken < 0
            ? cacheRow[sourceToken + 4]
            : newTokens[((size_t)begin + sourceToken) * channels + channel];
        value += __half2float(source) * curWeight[kernelIndex];
    }

    half conv = __float2half_rn(value);
    size_t outputIndex = ((size_t)begin + token) * channels + channel;
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[outputIndex] = __float2half(x / (1.0f + expf(-x)));
#else
    output[outputIndex] =
        __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

__global__ void FastllmShiftAppendConv1DPerChannelRaggedPrefillCacheHalfPointerKernel(
    half **pointers, const int *tokenOffsets, const half *newTokens,
    int channels) {
    int batchIndex = blockIdx.y;
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }

    int begin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - begin;
    half *cacheRow = pointers[batchIndex] + (size_t)channel * 4;
    half oldCache[4] = {
        cacheRow[0], cacheRow[1], cacheRow[2], cacheRow[3]
    };
#pragma unroll
    for (int cacheIndex = 0; cacheIndex < 4; cacheIndex++) {
        int sourceToken = numTokens - 4 + cacheIndex;
        cacheRow[cacheIndex] = sourceToken < 0
            ? oldCache[sourceToken + 4]
            : newTokens[((size_t)begin + sourceToken) * channels + channel];
    }
}

__global__ void FastllmPackRaggedGdnHalfKernel(
    const half *input, const int *tokenOffsets, half *output,
    int heads, int dim, int paddedSeqLen, float scale) {
    int batchIndex = blockIdx.y;
    int begin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - begin;
    size_t localIndex = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t paddedElements = (size_t)heads * paddedSeqLen * dim;
    if (localIndex >= paddedElements) {
        return;
    }

    int d = localIndex % dim;
    size_t row = localIndex / dim;
    int token = row % paddedSeqLen;
    int head = row / paddedSeqLen;
    half value = __float2half(0.0f);
    if (token < numTokens) {
        size_t inputIndex =
            (((size_t)begin + token) * heads + head) * dim + d;
        value = input[inputIndex];
        if (scale != 1.0f) {
            value = __float2half_rn(__half2float(value) * scale);
        }
    }
    output[((size_t)batchIndex * paddedElements) + localIndex] = value;
}

__global__ void FastllmUnpackRaggedGdnHalfKernel(
    const half *input, const int *tokenOffsets, half *output,
    int heads, int dim, int paddedSeqLen) {
    int batchIndex = blockIdx.y;
    int begin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - begin;
    size_t localIndex = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t requestElements = (size_t)numTokens * heads * dim;
    if (localIndex >= requestElements) {
        return;
    }

    int d = localIndex % dim;
    size_t row = localIndex / dim;
    int head = row % heads;
    int token = row / heads;
    size_t inputIndex =
        (((size_t)batchIndex * heads + head) * paddedSeqLen + token) * dim + d;
    size_t outputIndex =
        (((size_t)begin + token) * heads + head) * dim + d;
    output[outputIndex] = input[inputIndex];
}

__global__ void FastllmPackRaggedGdnChunksHalfKernel(
    const half *input, const int *tokenOffsets, const int *chunkOffsets,
    half *output, int totalChunks, int heads, int dim, int chunkSize,
    float scale) {
    int batchIndex = blockIdx.y;
    int tokenBegin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - tokenBegin;
    int chunkBegin = chunkOffsets[batchIndex];
    int paddedTokens =
        (chunkOffsets[batchIndex + 1] - chunkBegin) * chunkSize;
    size_t localIndex = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t requestElements = (size_t)heads * paddedTokens * dim;
    if (localIndex >= requestElements) {
        return;
    }

    int d = localIndex % dim;
    size_t row = localIndex / dim;
    int token = row % paddedTokens;
    int head = row / paddedTokens;
    half value = __float2half(0.0f);
    if (token < numTokens) {
        size_t inputIndex =
            (((size_t)tokenBegin + token) * heads + head) * dim + d;
        value = input[inputIndex];
        if (scale != 1.0f) {
            value = __float2half_rn(__half2float(value) * scale);
        }
    }
    size_t packedToken = (size_t)chunkBegin * chunkSize + token;
    size_t outputIndex =
        ((size_t)head * totalChunks * chunkSize + packedToken) * dim + d;
    output[outputIndex] = value;
}

__global__ void FastllmUnpackRaggedGdnChunksHalfKernel(
    const half *input, const int *tokenOffsets, const int *chunkOffsets,
    half *output, int totalChunks, int heads, int dim, int chunkSize) {
    int batchIndex = blockIdx.y;
    int tokenBegin = tokenOffsets[batchIndex];
    int numTokens = tokenOffsets[batchIndex + 1] - tokenBegin;
    int chunkBegin = chunkOffsets[batchIndex];
    size_t localIndex = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t requestElements = (size_t)numTokens * heads * dim;
    if (localIndex >= requestElements) {
        return;
    }

    int d = localIndex % dim;
    size_t row = localIndex / dim;
    int head = row % heads;
    int token = row / heads;
    size_t packedToken = (size_t)chunkBegin * chunkSize + token;
    size_t inputIndex =
        ((size_t)head * totalChunks * chunkSize + packedToken) * dim + d;
    size_t outputIndex =
        (((size_t)tokenBegin + token) * heads + head) * dim + d;
    output[outputIndex] = input[inputIndex];
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfPointerKernel(
    half **caches, const half *newToken, const float *weight, const float *bias,
    half *output, int batch, int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int b = row / channels;
    int c = row - b * channels;
    half *cacheRow = caches[b] + c * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[row] = __float2half(x / (1.0f + expf(-x)));
#else
    output[row] = __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

__global__ void FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfSlotKernel(
    half *cachePool, const int *slotIds, const half *newToken, const float *weight, const float *bias,
    half *output, int batch, int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int b = row / channels;
    int c = row - b * channels;
    int slot = slotIds[b];
    half *cacheRow = cachePool + ((size_t)slot * channels + c) * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];

    half conv = __float2half_rn(value);
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(conv);
    output[row] = __float2half(x / (1.0f + expf(-x)));
#else
    output[row] = __hdiv(conv, __hadd(__float2half(1.0f), hexp(-conv)));
#endif
}

bool FastllmCudaConv1DPerChannelSiluSingleTokenFloat16(const fastllm::Data &input, fastllm::Data &weight,
                                                       fastllm::Data &bias, fastllm::Data &output) {
    if (input.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (input.dataType != fastllm::DataType::FLOAT16 || weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == input.dims[1] && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == input.dims[1] && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (input.dims.size() != 3 || input.dims[0] != 1 || input.dims[2] != 4 ||
        input.strides.empty() || input.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != input.dims[1]))) {
        return false;
    }

    output.dataType = input.dataType;
    output.Resize({1, input.dims[1], 1});
    output.ToDevice(input.dataDevice, input.dataDeviceIds);
    output.Allocate();

    int channels = input.dims[1];
    int threadsPerBlock = 256;
    int blocksPerGrid = (channels + threadsPerBlock - 1) / threadsPerBlock;
    const half *cudaInput = (const half *) input.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const float *cudaBias = bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr;
    half *cudaOutput = (half *) output.cudaData;
    FastllmConv1DPerChannelSiluSingleTokenHalfKernel<<<blocksPerGrid, threadsPerBlock>>>(
        cudaInput, cudaWeight, cudaBias, cudaOutput, channels
    );
    checkCudaErrors("Error: CUDA error in FastllmCudaConv1DPerChannelSiluSingleTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(fastllm::Data &cache, const fastllm::Data &newToken,
                                                                  fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (cache.dataDevice != fastllm::DataDevice::CUDA || newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (cache.dataType != fastllm::DataType::FLOAT16 || newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (cache.dims.size() != 3 || cache.dims[0] <= 0 || cache.dims[2] != 4 ||
        newToken.dims.size() != 3 || newToken.dims[0] != cache.dims[0] || newToken.dims[1] != cache.dims[1] || newToken.dims[2] != 1 ||
        cache.strides.empty() || newToken.strides.empty() ||
        cache.strides.back() != 1 || newToken.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != cache.dims[1]))) {
        return false;
    }

    output.dataType = cache.dataType;
    output.Resize({cache.dims[0], cache.dims[1], 1});
    output.ToDevice(cache.dataDevice, cache.dataDeviceIds);
    output.Allocate();

    int batch = cache.dims[0];
    int channels = cache.dims[1];
    int total = batch * channels;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    half *cudaCache = (half *) cache.cudaData;
    const half *cudaNewToken = (const half *) newToken.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const float *cudaBias = bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr;
    half *cudaOutput = (half *) output.cudaData;
    FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfKernel<<<blocksPerGrid, threadsPerBlock>>>(
        cudaCache, cudaNewToken, cudaWeight, cudaBias, cudaOutput, batch, channels
    );
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluTwoTokenFloat16(fastllm::Data &cache, const fastllm::Data &newTokens,
                                                               fastllm::Data &weight, fastllm::Data &bias,
                                                               fastllm::Data &output, fastllm::Data *firstTokenCache) {
    if (cache.dataDevice != fastllm::DataDevice::CUDA || newTokens.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (cache.dataType != fastllm::DataType::FLOAT16 || newTokens.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (cache.dims.size() != 3 || cache.dims[0] <= 0 || cache.dims[2] != 4 ||
        newTokens.dims.size() != 3 || newTokens.dims[0] != cache.dims[0] ||
        newTokens.dims[1] != cache.dims[1] || newTokens.dims[2] != 2 ||
        cache.strides.empty() || newTokens.strides.empty() ||
        cache.strides.back() != 1 || newTokens.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != cache.dims[1]))) {
        return false;
    }

    output.dataType = cache.dataType;
    output.Resize({cache.dims[0], cache.dims[1], 2});
    output.ToDevice(cache.dataDevice, cache.dataDeviceIds);
    output.Allocate();

    if (firstTokenCache) {
        firstTokenCache->dataType = cache.dataType;
        firstTokenCache->Resize(cache.dims);
        firstTokenCache->ToDevice(cache.dataDevice, cache.dataDeviceIds);
        firstTokenCache->Allocate();
    }

    int batch = cache.dims[0];
    int channels = cache.dims[1];
    int total = batch * channels;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    half *cudaCache = (half *) cache.cudaData;
    const half *cudaNewTokens = (const half *) newTokens.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const float *cudaBias = bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr;
    half *cudaOutput = (half *) output.cudaData;
    half *cudaFirstTokenCache = firstTokenCache ? (half *) firstTokenCache->cudaData : nullptr;
    FastllmShiftAppendConv1DPerChannelSiluTwoTokenHalfKernel<<<blocksPerGrid, threadsPerBlock>>>(
        cudaCache, cudaNewTokens, cudaWeight, cudaBias, cudaOutput, cudaFirstTokenCache, batch, channels
    );
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluTwoTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
    fastllm::Data &cache, const fastllm::Data &newTokens,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output,
    fastllm::Data **tokenCaches, int numTokenCaches) {
    if (!FastllmCudaDataCanShareDevice(cache, newTokens) ||
        !FastllmCudaDataCanShareDevice(cache, weight) ||
        (bias.dims.size() > 0 && !FastllmCudaDataCanShareDevice(cache, bias)) ||
        output.isFake || output.cudaDataBorrowed || output.isPagedKVCache || output.multiDeviceData ||
        &output == &cache || &output == &weight || &output == &bias ||
        static_cast<const fastllm::Data*>(&output) == &newTokens) {
        return false;
    }
    int cacheDevice = -1;
    if (!FastllmCudaResolveDataDeviceId(cache, cacheDevice) ||
        FastllmCudaGetDevice() != cacheDevice) {
        return false;
    }
    if (cache.dataType != fastllm::DataType::FLOAT16 || newTokens.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    if (cache.dims.size() != 3 || cache.dims[0] <= 0 || cache.dims[1] <= 0 || cache.dims[2] != 4 ||
        newTokens.dims.size() != 3 || newTokens.dims[0] != cache.dims[0] ||
        newTokens.dims[1] != cache.dims[1] ||
        newTokens.dims[2] < 1 ||
        newTokens.dims[2] > FASTLLM_CUDA_MTP_FAST_SEQ_MAX ||
        !FastllmCudaDataHasDenseStrides(cache) ||
        !FastllmCudaDataHasDenseStrides(newTokens) ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != cache.dims[1]))) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == cache.dims[1] && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (!validWeightShape || !FastllmCudaDataHasDenseStrides(weight) ||
        (bias.dims.size() > 0 && !FastllmCudaDataHasDenseStrides(bias)) ||
        cache.cudaData == nullptr || newTokens.cudaData == nullptr || weight.cudaData == nullptr ||
        (bias.dims.size() > 0 && bias.cudaData == nullptr) ||
        numTokenCaches < 0 ||
        numTokenCaches > FASTLLM_CUDA_MTP_FAST_SEQ_MAX ||
        numTokenCaches > newTokens.dims[2] ||
        (numTokenCaches > 0 && tokenCaches == nullptr)) {
        return false;
    }

    for (int t = 0; t < numTokenCaches; t++) {
        fastllm::Data *snap = tokenCaches[t];
        if (snap == nullptr) {
            continue;
        }
        if (snap->isFake || snap->cudaDataBorrowed || snap->isPagedKVCache ||
            snap->multiDeviceData || snap == &cache || snap == &weight ||
            snap == &bias || snap == &output ||
            static_cast<const fastllm::Data*>(snap) == &newTokens) {
            return false;
        }
        for (int other = 0; other < t; other++) {
            if (tokenCaches[other] == snap) {
                return false;
            }
        }
    }

    int numTokens = newTokens.dims[2];
    uint64_t totalRows = (uint64_t)cache.dims[0] * cache.dims[1];
    if (totalRows > 0x7fffffffULL) {
        return false;
    }
    output.dataType = cache.dataType;
    output.Resize({cache.dims[0], cache.dims[1], numTokens});
    output.ToDevice(cache.dataDevice, std::vector<int>{cacheDevice});
    output.Allocate();
    output.isKVCache = false;
    output.isLinearAttention = false;
    output.isLinearAttentionTransposed = false;
    if (output.cudaData == nullptr || !FastllmCudaDataHasDenseStrides(output) ||
        !FastllmCudaDataCanShareDevice(cache, output)) {
        return false;
    }

    half *snaps[FASTLLM_CUDA_MTP_FAST_SEQ_MAX] = {};
    for (int t = 0; t < numTokenCaches; t++) {
        fastllm::Data *snap = tokenCaches[t];
        if (snap == nullptr) {
            continue;
        }
        snap->dataType = cache.dataType;
        snap->Resize(cache.dims);
        snap->ToDevice(cache.dataDevice, std::vector<int>{cacheDevice});
        snap->Allocate();
        snap->isLinearAttentionTransposed = false;
        if (snap->cudaData == nullptr || !FastllmCudaDataHasDenseStrides(*snap) ||
            !FastllmCudaDataCanShareDevice(cache, *snap)) {
            return false;
        }
        snaps[t] = (half *) snap->cudaData;
    }

    int batch = cache.dims[0];
    int channels = cache.dims[1];
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((totalRows + threadsPerBlock - 1) / threadsPerBlock);
    cudaError_t pendingState = cudaGetLastError();
    if (pendingState != cudaSuccess) {
        checkCudaErrors("Error: stale CUDA error before FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16.", pendingState);
        return false;
    }
    FastllmShiftAppendConv1DPerChannelSiluMultiTokenHalfKernel<<<blocksPerGrid, threadsPerBlock>>>(
        (half *) cache.cudaData, (const half *) newTokens.cudaData,
        (const float *) weight.cudaData,
        bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr,
        (half *) output.cudaData,
        snaps[0], snaps[1], snaps[2], snaps[3], snaps[4], snaps[5],
        numTokenCaches, batch, channels, numTokens
    );
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16.", launchState);
        return false;
    }
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16BatchPointers(
    const std::vector<fastllm::Data*> &caches,
    const fastllm::Data &newTokens,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output,
    const std::vector<fastllm::Data*> &tokenCaches, int numTokenCaches,
    int tokenMajorInputOffset) {
    if (caches.empty() || caches[0] == nullptr ||
        tokenMajorInputOffset < 0 || numTokenCaches < 0 ||
        numTokenCaches > FASTLLM_CUDA_MTP_PREFIX_SNAPSHOT_MAX ||
        tokenCaches.size() != caches.size() * (size_t)numTokenCaches ||
        output.isFake || output.cudaDataBorrowed || output.isPagedKVCache ||
        output.multiDeviceData) {
        return false;
    }
    std::set<const fastllm::Data*> tensors = {
        &newTokens, &weight, &bias, &output
    };
    if (tensors.size() != 4) {
        return false;
    }
    fastllm::Data &first = *caches[0];
    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(first, device) ||
        FastllmCudaGetDevice() != device ||
        first.dataType != fastllm::DataType::FLOAT16 ||
        first.dims.size() != 3 || first.dims[0] != 1 ||
        first.dims[1] <= 0 || first.dims[2] != 4 ||
        !FastllmCudaDataHasDenseStrides(first) ||
        newTokens.dataType != fastllm::DataType::FLOAT16 ||
        newTokens.dims.size() != 3 ||
        newTokens.dims[0] != (int)caches.size() ||
        !FastllmCudaDataHasDenseStrides(newTokens) ||
        !FastllmCudaDataCanShareDevice(first, newTokens) ||
        !FastllmCudaDataCanShareDevice(first, weight) ||
        (bias.dims.size() > 0 &&
         !FastllmCudaDataCanShareDevice(first, bias)) ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    int batch = (int)caches.size();
    int channels = first.dims[1];
    bool tokenMajorPrefill =
        numTokenCaches == 0 &&
        tokenMajorInputOffset + channels <= newTokens.dims[2] &&
        newTokens.dims[1] > 1 &&
        newTokens.dims[1] <= FASTLLM_CUDA_BATCH_PREFILL_SEQ_MAX;
    bool channelMajor =
        tokenMajorInputOffset == 0 &&
        newTokens.dims[1] == channels &&
        newTokens.dims[2] > 1 &&
        newTokens.dims[2] <= (numTokenCaches == 0 ?
            FASTLLM_CUDA_BATCH_PREFILL_SEQ_MAX :
            FASTLLM_CUDA_MTP_FAST_SEQ_MAX);
    if (!tokenMajorPrefill && !channelMajor) {
        return false;
    }
    int numTokens =
        tokenMajorPrefill ? newTokens.dims[1] : newTokens.dims[2];
    if (numTokenCaches > numTokens) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == channels &&
         weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == channels &&
         weight.dims[1] == 1 && weight.dims[2] == 4);
    if (!validWeightShape || !FastllmCudaDataHasDenseStrides(weight) ||
        (bias.dims.size() > 0 &&
         (bias.dims.size() != 1 || bias.dims[0] != channels ||
          !FastllmCudaDataHasDenseStrides(bias)))) {
        return false;
    }

    std::vector<void*> pointers;
    pointers.reserve(batch * (1 + numTokenCaches));
    for (fastllm::Data *cache : caches) {
        if (cache == nullptr || cache->dataType != first.dataType ||
            cache->dims != first.dims ||
            !FastllmCudaDataHasDenseStrides(*cache) ||
            !FastllmCudaDataCanShareDevice(first, *cache) ||
            !tensors.insert(cache).second) {
            return false;
        }
        pointers.push_back(cache->cudaData);
    }
    for (int b = 0; b < batch; b++) {
        for (int token = 0; token < numTokenCaches; token++) {
            fastllm::Data *snapshot =
                tokenCaches[b * numTokenCaches + token];
            if (snapshot == nullptr || snapshot->isFake ||
                snapshot->cudaDataBorrowed || snapshot->isPagedKVCache ||
                snapshot->multiDeviceData ||
                !tensors.insert(snapshot).second) {
                return false;
            }
            snapshot->dataType = first.dataType;
            snapshot->Resize(first.dims);
            snapshot->ToDevice(fastllm::DataDevice::CUDA,
                               std::vector<int>{device});
            snapshot->Allocate();
            snapshot->isLinearAttentionTransposed = false;
            if (snapshot->cudaData == nullptr ||
                !FastllmCudaDataHasDenseStrides(*snapshot) ||
                !FastllmCudaDataCanShareDevice(first, *snapshot)) {
                return false;
            }
            pointers.push_back(snapshot->cudaData);
        }
    }

    output.dataType = first.dataType;
    output.Resize(tokenMajorPrefill ?
        std::vector<int>{batch, numTokens, channels} :
        std::vector<int>{batch, channels, numTokens});
    output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
    output.Allocate();
    output.isKVCache = false;
    output.isLinearAttention = false;
    output.isLinearAttentionTransposed = false;
    if (output.cudaData == nullptr || !FastllmCudaDataHasDenseStrides(output) ||
        !FastllmCudaDataCanShareDevice(first, output)) {
        return false;
    }
    void **devicePointers = FastllmCudaStagePointers(pointers);
    int total = batch * channels;
    int threads = 256;
    if (numTokenCaches == 0) {
        size_t totalElements = (size_t)total * numTokens;
        int elementBlocks =
            (int)((totalElements + threads - 1) / threads);
        int cacheBlocks = (total + threads - 1) / threads;
        if (tokenMajorPrefill) {
            const char *token16Env = std::getenv(
                "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN16");
            bool useToken16 =
                (token16Env == nullptr || token16Env[0] == '\0' ||
                 FastllmCudaEnvFlagEnabled(
                     "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN16")) &&
                numTokens >= 16 && numTokens % 16 == 0;
            const char *token8Env = std::getenv(
                "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN8");
            bool useToken8 =
                (token8Env == nullptr || token8Env[0] == '\0' ||
                 FastllmCudaEnvFlagEnabled(
                     "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN8")) &&
                numTokens >= 8 && numTokens % 8 == 0;
            const char *token4Env = std::getenv(
                "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN4");
            bool useToken4 =
                (token4Env == nullptr || token4Env[0] == '\0' ||
                 FastllmCudaEnvFlagEnabled(
                     "FASTLLM_CUDA_QWEN35_GDN_CONV_TOKEN4")) &&
                numTokens >= 4 && numTokens % 4 == 0;
            if (useToken16) {
                size_t totalGroups = totalElements / 16;
                int groupBlocks =
                    (int)((totalGroups + threads - 1) / threads);
                FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerGroupedKernel<16>
                    <<<groupBlocks, threads>>>(
                        (half**)devicePointers,
                        (const half*)newTokens.cudaData,
                        (const float*)weight.cudaData,
                        bias.dims.empty() ?
                            nullptr : (const float*)bias.cudaData,
                        (half*)output.cudaData, channels, numTokens,
                        newTokens.dims[2], tokenMajorInputOffset,
                        totalGroups);
            } else if (useToken8) {
                size_t totalGroups = totalElements / 8;
                int groupBlocks =
                    (int)((totalGroups + threads - 1) / threads);
                FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerGroupedKernel<8>
                    <<<groupBlocks, threads>>>(
                        (half**)devicePointers,
                        (const half*)newTokens.cudaData,
                        (const float*)weight.cudaData,
                        bias.dims.empty() ?
                            nullptr : (const float*)bias.cudaData,
                        (half*)output.cudaData, channels, numTokens,
                        newTokens.dims[2], tokenMajorInputOffset,
                        totalGroups);
            } else if (useToken4) {
                size_t totalGroups = totalElements / 4;
                int groupBlocks =
                    (int)((totalGroups + threads - 1) / threads);
                FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerGroupedKernel<4>
                    <<<groupBlocks, threads>>>(
                        (half**)devicePointers,
                        (const half*)newTokens.cudaData,
                        (const float*)weight.cudaData,
                        bias.dims.empty() ?
                            nullptr : (const float*)bias.cudaData,
                        (half*)output.cudaData, channels, numTokens,
                        newTokens.dims[2], tokenMajorInputOffset,
                        totalGroups);
            } else {
                FastllmShiftAppendConv1DPerChannelSiluPrefillTokenMajorHalfPointerKernel
                    <<<elementBlocks, threads>>>(
                        (half**)devicePointers,
                        (const half*)newTokens.cudaData,
                        (const float*)weight.cudaData,
                        bias.dims.empty() ?
                            nullptr : (const float*)bias.cudaData,
                        (half*)output.cudaData, channels, numTokens,
                        newTokens.dims[2], tokenMajorInputOffset,
                        totalElements);
            }
            FastllmShiftAppendConv1DPerChannelPrefillCacheTokenMajorHalfPointerKernel
                <<<cacheBlocks, threads>>>(
                    (half**)devicePointers, (const half*)newTokens.cudaData,
                    channels, numTokens, newTokens.dims[2],
                    tokenMajorInputOffset, (size_t)total);
        } else {
            FastllmShiftAppendConv1DPerChannelSiluPrefillHalfPointerKernel
                <<<elementBlocks, threads>>>(
                    (half**)devicePointers, (const half*)newTokens.cudaData,
                    (const float*)weight.cudaData,
                    bias.dims.empty() ? nullptr : (const float*)bias.cudaData,
                    (half*)output.cudaData, channels, numTokens,
                    totalElements);
            FastllmShiftAppendConv1DPerChannelPrefillCacheHalfPointerKernel
                <<<cacheBlocks, threads>>>(
                    (half**)devicePointers, (const half*)newTokens.cudaData,
                    channels, numTokens, (size_t)total);
        }
    } else {
        int blocks = (total + threads - 1) / threads;
        FastllmShiftAppendConv1DPerChannelSiluMultiTokenHalfPointerKernel
            <<<blocks, threads>>>(
                (half**)devicePointers, (const half*)newTokens.cudaData,
                (const float*)weight.cudaData,
                bias.dims.empty() ? nullptr : (const float*)bias.cudaData,
                (half*)output.cudaData, batch, channels, numTokens,
                numTokenCaches);
    }
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in batched multi-token MTP conv.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluRaggedPrefillFloat16BatchPointers(
    const std::vector<fastllm::Data*> &caches,
    const fastllm::Data &newTokens, const std::vector<int> &seqLens,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (caches.empty() || caches.size() != seqLens.size() ||
        caches[0] == nullptr || output.isFake || output.cudaDataBorrowed ||
        output.isPagedKVCache || output.multiDeviceData) {
        return false;
    }
    std::set<const fastllm::Data*> tensors = {
        &newTokens, &weight, &bias, &output
    };
    if (tensors.size() != 4) {
        return false;
    }

    fastllm::Data &first = *caches[0];
    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(first, device) ||
        FastllmCudaGetDevice() != device ||
        first.dataType != fastllm::DataType::FLOAT16 ||
        first.dims.size() != 3 || first.dims[0] != 1 ||
        first.dims[1] <= 0 || first.dims[2] != 4 ||
        !FastllmCudaDataHasDenseStrides(first) ||
        newTokens.dataType != fastllm::DataType::FLOAT16 ||
        newTokens.dims.size() != 3 || newTokens.dims[0] != 1 ||
        !FastllmCudaDataHasDenseStrides(newTokens) ||
        !FastllmCudaDataCanShareDevice(first, newTokens) ||
        !FastllmCudaDataCanShareDevice(first, weight) ||
        (bias.dims.size() > 0 &&
         !FastllmCudaDataCanShareDevice(first, bias)) ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 &&
         bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }

    int batch = (int)caches.size();
    int channels = first.dims[1];
    if (newTokens.dims[2] != channels) {
        return false;
    }
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == channels &&
         weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == channels &&
         weight.dims[1] == 1 && weight.dims[2] == 4);
    if (!validWeightShape || !FastllmCudaDataHasDenseStrides(weight) ||
        (bias.dims.size() > 0 &&
         (bias.dims.size() != 1 || bias.dims[0] != channels ||
          !FastllmCudaDataHasDenseStrides(bias)))) {
        return false;
    }

    std::vector<int> tokenOffsets(batch + 1, 0);
    int maxSeqLen = 0;
    for (int b = 0; b < batch; b++) {
        if (seqLens[b] <= 0 ||
            seqLens[b] > FASTLLM_CUDA_BATCH_PREFILL_SEQ_MAX) {
            return false;
        }
        tokenOffsets[b + 1] = tokenOffsets[b] + seqLens[b];
        maxSeqLen = std::max(maxSeqLen, seqLens[b]);
    }
    if (tokenOffsets.back() != newTokens.dims[1]) {
        return false;
    }

    std::vector<void*> pointers;
    pointers.reserve(batch);
    for (fastllm::Data *cache : caches) {
        if (cache == nullptr || cache->dataType != first.dataType ||
            cache->dims != first.dims ||
            !FastllmCudaDataHasDenseStrides(*cache) ||
            !FastllmCudaDataCanShareDevice(first, *cache) ||
            !tensors.insert(cache).second) {
            return false;
        }
        pointers.push_back(cache->cudaData);
    }

    output.dataType = first.dataType;
    output.Resize(newTokens.dims);
    output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
    output.Allocate();
    output.isKVCache = false;
    output.isLinearAttention = false;
    output.isLinearAttentionTransposed = false;
    if (output.cudaData == nullptr || !FastllmCudaDataHasDenseStrides(output) ||
        !FastllmCudaDataCanShareDevice(first, output)) {
        return false;
    }

    void **devicePointers = FastllmCudaStagePointers(pointers);
    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(seqLens, 64, metadata) ||
        metadata.totalTokens != tokenOffsets.back()) {
        return false;
    }
    const int *deviceOffsets = metadata.tokenOffsets;
    int threads = 256;
    int elementBlocks = (int)(((size_t)maxSeqLen * channels + threads - 1) /
                              threads);
    int cacheBlocks = (channels + threads - 1) / threads;
    FastllmShiftAppendConv1DPerChannelSiluRaggedPrefillHalfPointerKernel
        <<<dim3(elementBlocks, batch), threads>>>(
            (half**)devicePointers, deviceOffsets,
            (const half*)newTokens.cudaData, (const float*)weight.cudaData,
            bias.dims.empty() ? nullptr : (const float*)bias.cudaData,
            (half*)output.cudaData, channels);
    FastllmShiftAppendConv1DPerChannelRaggedPrefillCacheHalfPointerKernel
        <<<dim3(cacheBlocks, batch), threads>>>(
            (half**)devicePointers, deviceOffsets,
            (const half*)newTokens.cudaData, channels);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in ragged batched prefill conv.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaPackRaggedGdnPrefillFloat16(
    const fastllm::Data &q, const fastllm::Data &k,
    const fastllm::Data &v, const fastllm::Data &b,
    const fastllm::Data &g, const std::vector<int> &seqLens,
    int paddedSeqLen, float qScale,
    fastllm::Data &qPadded, fastllm::Data &kPadded,
    fastllm::Data &vPadded, fastllm::Data &bPadded,
    fastllm::Data &gPadded) {
    std::set<const fastllm::Data*> inputs = {&q, &k, &v, &b, &g};
    if (seqLens.empty() || paddedSeqLen <= 0 ||
        inputs.size() != 5 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        q.dims.size() != 4 || q.dims[0] != 1 ||
        k.dims != q.dims || v.dims.size() != 4 || v.dims[0] != 1 ||
        b.dims.size() != 3 || b.dims[0] != 1 || g.dims != b.dims ||
        v.dims[1] != q.dims[1] || b.dims[1] != q.dims[1] ||
        v.dims[2] != q.dims[2] || b.dims[2] != q.dims[2] ||
        q.cudaData == nullptr || k.cudaData == nullptr ||
        v.cudaData == nullptr || b.cudaData == nullptr || g.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(q) ||
        !FastllmCudaDataHasDenseStrides(k) ||
        !FastllmCudaDataHasDenseStrides(v) ||
        !FastllmCudaDataHasDenseStrides(b) ||
        !FastllmCudaDataHasDenseStrides(g) ||
        !FastllmCudaDataCanShareDevice(q, k) ||
        !FastllmCudaDataCanShareDevice(q, v) ||
        !FastllmCudaDataCanShareDevice(q, b) ||
        !FastllmCudaDataCanShareDevice(q, g)) {
        return false;
    }
    std::set<fastllm::Data*> outputs = {
        &qPadded, &kPadded, &vPadded, &bPadded, &gPadded
    };
    if (outputs.size() != 5) {
        return false;
    }
    for (fastllm::Data *output : outputs) {
        if (inputs.count(output) != 0 || output->isFake ||
            output->cudaDataBorrowed ||
            output->isPagedKVCache || output->multiDeviceData) {
            return false;
        }
    }

    int batch = (int)seqLens.size();
    int totalTokens = 0;
    std::vector<int> tokenOffsets(batch + 1, 0);
    for (int request = 0; request < batch; request++) {
        if (seqLens[request] <= 0 || seqLens[request] > paddedSeqLen) {
            return false;
        }
        tokenOffsets[request + 1] = tokenOffsets[request] + seqLens[request];
    }
    totalTokens = tokenOffsets.back();
    if (totalTokens != q.dims[1]) {
        return false;
    }

    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(q, device) ||
        FastllmCudaGetDevice() != device) {
        return false;
    }
    int heads = q.dims[2];
    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) -> bool {
        output.dataType = fastllm::DataType::FLOAT16;
        output.Resize(dims);
        output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
        output.Allocate(false);
        output.isKVCache = false;
        output.isLinearAttention = false;
        output.isLinearAttentionTransposed = false;
        return output.cudaData != nullptr &&
               FastllmCudaDataHasDenseStrides(output) &&
               FastllmCudaDataCanShareDevice(q, output);
    };
    if (!prepareOutput(qPadded, {batch, heads, paddedSeqLen, q.dims[3]}) ||
        !prepareOutput(kPadded, {batch, heads, paddedSeqLen, k.dims[3]}) ||
        !prepareOutput(vPadded, {batch, heads, paddedSeqLen, v.dims[3]}) ||
        !prepareOutput(bPadded, {batch, heads, paddedSeqLen}) ||
        !prepareOutput(gPadded, {batch, heads, paddedSeqLen})) {
        return false;
    }

    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(seqLens, 64, metadata) ||
        metadata.totalTokens != tokenOffsets.back()) {
        return false;
    }
    const int *deviceOffsets = metadata.tokenOffsets;
    int threads = 256;
    auto launchPack = [&](const fastllm::Data &input,
                          fastllm::Data &output, int dim, float scale) {
        size_t requestElements = (size_t)heads * paddedSeqLen * dim;
        int blocks = (int)((requestElements + threads - 1) / threads);
        FastllmPackRaggedGdnHalfKernel<<<dim3(blocks, batch), threads>>>(
            (const half*)input.cudaData, deviceOffsets,
            (half*)output.cudaData, heads, dim, paddedSeqLen, scale);
    };
    launchPack(q, qPadded, q.dims[3], qScale);
    launchPack(k, kPadded, k.dims[3], 1.0f);
    launchPack(v, vPadded, v.dims[3], 1.0f);
    launchPack(b, bPadded, 1, 1.0f);
    launchPack(g, gPadded, 1, 1.0f);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error when packing ragged GDN prefill.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaUnpackRaggedGdnPrefillFloat16(
    const fastllm::Data &padded, const std::vector<int> &seqLens,
    fastllm::Data &ragged) {
    if (seqLens.empty() || padded.dataDevice != fastllm::DataDevice::CUDA ||
        padded.dataType != fastllm::DataType::FLOAT16 ||
        padded.dims.size() != 4 || padded.dims[0] != (int)seqLens.size() ||
        padded.cudaData == nullptr || !FastllmCudaDataHasDenseStrides(padded) ||
        ragged.isFake || ragged.cudaDataBorrowed || ragged.isPagedKVCache ||
        ragged.multiDeviceData || &ragged == &padded) {
        return false;
    }
    int batch = (int)seqLens.size();
    int heads = padded.dims[1];
    int paddedSeqLen = padded.dims[2];
    int dim = padded.dims[3];
    int maxSeqLen = 0;
    std::vector<int> tokenOffsets(batch + 1, 0);
    for (int request = 0; request < batch; request++) {
        if (seqLens[request] <= 0 || seqLens[request] > paddedSeqLen) {
            return false;
        }
        tokenOffsets[request + 1] = tokenOffsets[request] + seqLens[request];
        maxSeqLen = std::max(maxSeqLen, seqLens[request]);
    }

    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(padded, device) ||
        FastllmCudaGetDevice() != device) {
        return false;
    }
    ragged.dataType = fastllm::DataType::FLOAT16;
    ragged.Resize({1, tokenOffsets.back(), heads, dim});
    ragged.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
    ragged.Allocate(false);
    ragged.isKVCache = false;
    ragged.isLinearAttention = false;
    ragged.isLinearAttentionTransposed = false;
    if (ragged.cudaData == nullptr || !FastllmCudaDataHasDenseStrides(ragged) ||
        !FastllmCudaDataCanShareDevice(padded, ragged)) {
        return false;
    }

    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(seqLens, 64, metadata) ||
        metadata.totalTokens != tokenOffsets.back()) {
        return false;
    }
    const int *deviceOffsets = metadata.tokenOffsets;
    int threads = 256;
    int blocks = (int)(((size_t)maxSeqLen * heads * dim + threads - 1) /
                       threads);
    FastllmUnpackRaggedGdnHalfKernel<<<dim3(blocks, batch), threads>>>(
        (const half*)padded.cudaData, deviceOffsets, (half*)ragged.cudaData,
        heads, dim, paddedSeqLen);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error when unpacking ragged GDN prefill.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaPackRaggedGdnPrefillChunksFloat16(
    const fastllm::Data &q, const fastllm::Data &k,
    const fastllm::Data &v, const fastllm::Data &b,
    const fastllm::Data &g, const std::vector<int> &seqLens,
    int chunkSize, float qScale,
    fastllm::Data &qPacked, fastllm::Data &kPacked,
    fastllm::Data &vPacked, fastllm::Data &bPacked,
    fastllm::Data &gPacked) {
    std::set<const fastllm::Data*> inputs = {&q, &k, &v, &b, &g};
    if (seqLens.empty() || chunkSize <= 0 || inputs.size() != 5 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        q.dims.size() != 4 || q.dims[0] != 1 || k.dims != q.dims ||
        v.dims.size() != 4 || v.dims[0] != 1 ||
        b.dims.size() != 3 || b.dims[0] != 1 || g.dims != b.dims ||
        v.dims[1] != q.dims[1] || b.dims[1] != q.dims[1] ||
        v.dims[2] != q.dims[2] || b.dims[2] != q.dims[2] ||
        q.cudaData == nullptr || k.cudaData == nullptr ||
        v.cudaData == nullptr || b.cudaData == nullptr || g.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(q) ||
        !FastllmCudaDataHasDenseStrides(k) ||
        !FastllmCudaDataHasDenseStrides(v) ||
        !FastllmCudaDataHasDenseStrides(b) ||
        !FastllmCudaDataHasDenseStrides(g) ||
        !FastllmCudaDataCanShareDevice(q, k) ||
        !FastllmCudaDataCanShareDevice(q, v) ||
        !FastllmCudaDataCanShareDevice(q, b) ||
        !FastllmCudaDataCanShareDevice(q, g)) {
        return false;
    }
    std::set<fastllm::Data*> outputs = {
        &qPacked, &kPacked, &vPacked, &bPacked, &gPacked
    };
    if (outputs.size() != 5) {
        return false;
    }
    for (fastllm::Data *output : outputs) {
        if (inputs.count(output) != 0 || output->isFake ||
            output->cudaDataBorrowed || output->isPagedKVCache ||
            output->multiDeviceData) {
            return false;
        }
    }

    int batch = (int)seqLens.size();
    int maxPaddedTokens = 0;
    std::vector<int> tokenOffsets(batch + 1, 0);
    std::vector<int> chunkOffsets(batch + 1, 0);
    for (int request = 0; request < batch; request++) {
        int len = seqLens[request];
        if (len <= 0) {
            return false;
        }
        int chunks = (len + chunkSize - 1) / chunkSize;
        tokenOffsets[request + 1] = tokenOffsets[request] + len;
        chunkOffsets[request + 1] = chunkOffsets[request] + chunks;
        maxPaddedTokens = std::max(maxPaddedTokens, chunks * chunkSize);
    }
    int totalTokens = tokenOffsets.back();
    int totalChunks = chunkOffsets.back();
    if (totalTokens != q.dims[1] || totalChunks <= 0) {
        return false;
    }

    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(q, device) ||
        FastllmCudaGetDevice() != device) {
        return false;
    }
    int heads = q.dims[2];
    int packedTokens = totalChunks * chunkSize;
    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) -> bool {
        output.dataType = fastllm::DataType::FLOAT16;
        output.Resize(dims);
        output.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
        output.Allocate(false);
        output.isKVCache = false;
        output.isLinearAttention = false;
        output.isLinearAttentionTransposed = false;
        return output.cudaData != nullptr &&
               FastllmCudaDataHasDenseStrides(output) &&
               FastllmCudaDataCanShareDevice(q, output);
    };
    if (!prepareOutput(qPacked, {1, heads, packedTokens, q.dims[3]}) ||
        !prepareOutput(kPacked, {1, heads, packedTokens, k.dims[3]}) ||
        !prepareOutput(vPacked, {1, heads, packedTokens, v.dims[3]}) ||
        !prepareOutput(bPacked, {1, heads, packedTokens}) ||
        !prepareOutput(gPacked, {1, heads, packedTokens})) {
        return false;
    }

    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(
            seqLens, chunkSize, metadata) ||
        metadata.totalTokens != totalTokens ||
        metadata.totalChunks != totalChunks) {
        return false;
    }
    const int *deviceTokenOffsets = metadata.tokenOffsets;
    const int *deviceChunkOffsets = metadata.chunkOffsets;
    int threads = 256;
    auto launchPack = [&](const fastllm::Data &input,
                          fastllm::Data &output, int dim, float scale) {
        size_t requestElements = (size_t)heads * maxPaddedTokens * dim;
        int blocks = (int)((requestElements + threads - 1) / threads);
        FastllmPackRaggedGdnChunksHalfKernel
            <<<dim3(blocks, batch), threads>>>(
                (const half*)input.cudaData, deviceTokenOffsets,
                deviceChunkOffsets, (half*)output.cudaData,
                totalChunks, heads, dim, chunkSize, scale);
    };
    launchPack(q, qPacked, q.dims[3], qScale);
    launchPack(k, kPacked, k.dims[3], 1.0f);
    launchPack(v, vPacked, v.dims[3], 1.0f);
    launchPack(b, bPacked, 1, 1.0f);
    launchPack(g, gPacked, 1, 1.0f);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error when packing chunk-major ragged GDN prefill.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaUnpackRaggedGdnPrefillChunksFloat16(
    const fastllm::Data &packed, const std::vector<int> &seqLens,
    int chunkSize, fastllm::Data &ragged) {
    if (seqLens.empty() || chunkSize <= 0 ||
        packed.dataDevice != fastllm::DataDevice::CUDA ||
        packed.dataType != fastllm::DataType::FLOAT16 ||
        packed.dims.size() != 4 || packed.dims[0] != 1 ||
        packed.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(packed) ||
        ragged.isFake || ragged.cudaDataBorrowed ||
        ragged.isPagedKVCache || ragged.multiDeviceData ||
        &ragged == &packed) {
        return false;
    }
    int batch = (int)seqLens.size();
    int maxSeqLen = 0;
    std::vector<int> tokenOffsets(batch + 1, 0);
    std::vector<int> chunkOffsets(batch + 1, 0);
    for (int request = 0; request < batch; request++) {
        int len = seqLens[request];
        if (len <= 0) {
            return false;
        }
        tokenOffsets[request + 1] = tokenOffsets[request] + len;
        chunkOffsets[request + 1] =
            chunkOffsets[request] + (len + chunkSize - 1) / chunkSize;
        maxSeqLen = std::max(maxSeqLen, len);
    }
    int totalChunks = chunkOffsets.back();
    if (packed.dims[2] != totalChunks * chunkSize) {
        return false;
    }
    int heads = packed.dims[1];
    int dim = packed.dims[3];
    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(packed, device) ||
        FastllmCudaGetDevice() != device) {
        return false;
    }

    ragged.dataType = fastllm::DataType::FLOAT16;
    ragged.Resize({1, tokenOffsets.back(), heads, dim});
    ragged.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{device});
    ragged.Allocate(false);
    ragged.isKVCache = false;
    ragged.isLinearAttention = false;
    ragged.isLinearAttentionTransposed = false;
    if (ragged.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(ragged) ||
        !FastllmCudaDataCanShareDevice(packed, ragged)) {
        return false;
    }

    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(
            seqLens, chunkSize, metadata) ||
        metadata.totalTokens != tokenOffsets.back() ||
        metadata.totalChunks != totalChunks) {
        return false;
    }
    const int *deviceTokenOffsets = metadata.tokenOffsets;
    const int *deviceChunkOffsets = metadata.chunkOffsets;
    int threads = 256;
    int blocks = (int)(((size_t)maxSeqLen * heads * dim + threads - 1) /
                       threads);
    FastllmUnpackRaggedGdnChunksHalfKernel
        <<<dim3(blocks, batch), threads>>>(
            (const half*)packed.cudaData, deviceTokenOffsets,
            deviceChunkOffsets, (half*)ragged.cudaData,
            totalChunks, heads, dim, chunkSize);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error when unpacking chunk-major ragged GDN prefill.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(
    const std::vector<fastllm::Data*> &caches, const fastllm::Data &newToken,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (caches.empty() || caches[0] == nullptr) {
        return false;
    }
    fastllm::Data *first = caches[0];
    if (first->dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (first->dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    if (first->dims.size() != 3 || first->dims[0] != 1 || first->dims[2] != 4 ||
        first->strides.empty() || first->strides.back() != 1) {
        return false;
    }

    int batch = (int)caches.size();
    int channels = first->dims[1];
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == channels && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == channels && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch || newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != channels))) {
        return false;
    }
    for (int i = 0; i < batch; i++) {
        fastllm::Data *cur = caches[i];
        if (cur == nullptr ||
            cur->dataDevice != first->dataDevice ||
            cur->dataType != first->dataType ||
            cur->dims != first->dims ||
            cur->strides.empty() ||
            cur->strides.back() != 1 ||
            cur->cudaData == nullptr) {
            return false;
        }
    }

    std::vector<void*> pointers(batch);
    for (int i = 0; i < batch; i++) {
        pointers[i] = caches[i]->cudaData;
    }
    void **cudaPointers = FastllmCudaStagePointers(pointers);

    return FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(
        cudaPointers, batch, *first, newToken, weight, bias, output);
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(
    void *cudaCachePointers, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (cudaCachePointers == nullptr || batch <= 0) {
        return false;
    }
    if (firstCache.dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (firstCache.dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    if (firstCache.dims.size() != 3 || firstCache.dims[0] != 1 || firstCache.dims[2] != 4 ||
        firstCache.strides.empty() || firstCache.strides.back() != 1 ||
        firstCache.cudaData == nullptr) {
        return false;
    }

    int channels = firstCache.dims[1];
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == channels && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == channels && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch || newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != channels))) {
        return false;
    }

    output.dataType = firstCache.dataType;
    output.Resize({batch, channels, 1});
    output.ToDevice(firstCache.dataDevice, firstCache.dataDeviceIds);
    output.Allocate();

    int total = batch * channels;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    const half *cudaNewToken = (const half *) newToken.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const float *cudaBias = bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr;
    half *cudaOutput = (half *) output.cudaData;
    FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfPointerKernel<<<blocksPerGrid, threadsPerBlock>>>(
        (half**)cudaCachePointers, cudaNewToken, cudaWeight, cudaBias, cudaOutput, batch, channels
    );
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots(
    void *cudaCachePool, void *cudaSlotIds, int batch, const fastllm::Data &firstCache,
    const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias,
    fastllm::Data &output) {
    if (cudaCachePool == nullptr || cudaSlotIds == nullptr || batch <= 0) {
        return false;
    }
    if (firstCache.dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (firstCache.dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32 ||
        (bias.dims.size() > 0 && (bias.dataDevice != fastllm::DataDevice::CUDA || bias.dataType != fastllm::DataType::FLOAT32))) {
        return false;
    }
    if (firstCache.dims.size() != 3 || firstCache.dims[0] != 1 || firstCache.dims[2] != 4 ||
        firstCache.strides.empty() || firstCache.strides.back() != 1 ||
        firstCache.cudaData == nullptr) {
        return false;
    }

    int channels = firstCache.dims[1];
    bool validWeightShape =
        (weight.dims.size() == 2 && weight.dims[0] == channels && weight.dims[1] == 4) ||
        (weight.dims.size() == 3 && weight.dims[0] == channels && weight.dims[1] == 1 && weight.dims[2] == 4);
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch || newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1 ||
        !validWeightShape ||
        (bias.dims.size() > 0 && (bias.dims.size() != 1 || bias.dims[0] != channels))) {
        return false;
    }

    output.dataType = firstCache.dataType;
    output.Resize({batch, channels, 1});
    output.ToDevice(firstCache.dataDevice, firstCache.dataDeviceIds);
    output.Allocate();

    int total = batch * channels;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    const half *cudaNewToken = (const half *) newToken.cudaData;
    const float *cudaWeight = (const float *) weight.cudaData;
    const float *cudaBias = bias.dims.size() > 0 ? (const float *) bias.cudaData : nullptr;
    half *cudaOutput = (half *) output.cudaData;
    FastllmShiftAppendConv1DPerChannelSiluSingleTokenHalfSlotKernel<<<blocksPerGrid, threadsPerBlock>>>(
        (half*)cudaCachePool, (const int*)cudaSlotIds, cudaNewToken, cudaWeight, cudaBias,
        cudaOutput, batch, channels
    );
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots.", cudaGetLastError());
    return true;
}

bool FastllmCudaConv2DFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, fastllm::Data &output) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData = (float *)FastllmCudaMalloc(outputChannels * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, outputChannels * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, outputChannels * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    std::vector <int> dims = input.dims;
    int inputHeight = dims[2], inputWidth = dims[3];
    int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
    int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

    if (weight.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaNaiveConv2DHalfKernel <<< outputChannels, 256 >>> (
            cudaInput, (half*)weight.cudaData, cudaBiasData, 
            inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, 
            inputHeight, inputWidth, outputHeight, outputWidth, 
            cudaOutput
        );
    } else {
        FastllmCudaNaiveConv2DKernel <<< outputChannels, 256 >>> (
            cudaInput, (float*)weight.cudaData, cudaBiasData, 
            inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, 
            inputHeight, inputWidth, outputHeight, outputWidth, 
            cudaOutput
        );
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void FastllmReduce(uint8_t *output, uint8_t* partOutput, int len, int threadNum, fastllm::DataType dataType) {
    int threadPerBlock = std::min(256, len);
    if (dataType == fastllm::DataType::FLOAT32) {
        FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>> ((float*)output, (float*)partOutput, len, threadNum);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>> ((half*)output, (half*)partOutput, len, threadNum);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>> ((__nv_bfloat16*)output, (__nv_bfloat16*)partOutput, len, threadNum);
    }
}

void FastllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}

int FastllmCudaGetDevice() {
    int id = -1;
    cudaGetDevice(&id);
    return id;
}

int GetPointerDeviceId(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
        if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
        if (attributes.type == cudaMemoryTypeDevice) {
#endif
            int device = attributes.device;
            // printf("Pointer belongs to device %d\n", device);
            return device;
        } else {
            printf("Pointer is not device memory\n");
            return -1;
        }
    } else {
        printf("Error: %s\n", cudaGetErrorString(err));
        cudaGetLastError();
        return -1;
    }
}

int FastllmCudaGetDeviceCount() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

__global__ void FastllmCudaResetLogitsOfEOS(int batch, int stride, float *logits, int *res_lens, int *eos_nums, int *eos_ids) {
    int base = 0;
    for (int b = 0; b < batch; b++) {
        if (res_lens[b] > 0) {
            for (int i = 0; i < eos_nums[b]; i++) {
                logits[stride * b + eos_ids[base + i]] = -1.0e30f;
            }
        }
        base += eos_nums[b];
    }
    return;
}
void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lens, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids) {
    int originalDevice = FastllmCudaGetDevice();
    // Sampling logits may be a view whose dataDeviceIds is empty or stale.  The
    // allocation itself is the authoritative source for the launch device.
    int logitsDevice = GetPointerDeviceId(logits->cudaData);
    if (logitsDevice < 0) {
        logitsDevice = logits->dataDeviceIds.empty() ? originalDevice : logits->dataDeviceIds[0];
    }
    FastllmCudaSetDevice(logitsDevice);
    cudaError_t state = cudaSuccess;
    int *cuda_res_lens = (int*)FastllmCudaMalloc(sizeof(int) * res_lens.size());
    state = cudaMemcpyAsync(cuda_res_lens, res_lens.data(), sizeof(int) * res_lens.size(), cudaMemcpyHostToDevice, 0);
    int *cuda_eos_nums = (int*)FastllmCudaMalloc(sizeof(int) * eos_nums.size());
    state = cudaMemcpyAsync(cuda_eos_nums, eos_nums.data(), sizeof(int) * eos_nums.size(), cudaMemcpyHostToDevice, 0);
    int *cuda_eos_ids = (int*)FastllmCudaMalloc(sizeof(int) * eos_ids.size());    
    state = cudaMemcpyAsync(cuda_eos_ids, eos_ids.data(), sizeof(int) * eos_ids.size(), cudaMemcpyHostToDevice, 0);
    FastllmCudaResetLogitsOfEOS <<<1,1>>> (batch, logits->Count(0) / batch, (float*)logits->cudaData, cuda_res_lens, cuda_eos_nums, cuda_eos_ids);
    checkCudaErrors("Error: CUDA error when reset logtis of EOS!", state);
    FastllmCudaFree(cuda_res_lens);
    FastllmCudaFree(cuda_eos_nums);
    FastllmCudaFree(cuda_eos_ids);
    FastllmCudaSetDevice(originalDevice);
    return;
}

__global__ void FastllmCudaResetLogitsOfEOSAll(int total, int eos_num, int stride, float *logits, int *eos_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int b = idx / eos_num;
    int e = idx - b * eos_num;
    logits[stride * b + eos_ids[e]] = -1.0e30f;
}

void FastllmResetLogitsOfEOSAll(int batch, fastllm::Data *logits, const std::vector<int> &eos_ids) {
    if (batch <= 0 || eos_ids.empty()) {
        return;
    }
    int originalDevice = FastllmCudaGetDevice();
    // Sampling logits may be a view whose dataDeviceIds is empty or stale.  The
    // allocation itself is the authoritative source for the launch device.
    int logitsDevice = GetPointerDeviceId(logits->cudaData);
    if (logitsDevice < 0) {
        logitsDevice = logits->dataDeviceIds.empty() ? originalDevice : logits->dataDeviceIds[0];
    }
    cudaError_t switchState = cudaSetDevice(logitsDevice);
    checkCudaErrors("Error: CUDA error when switching to logits device!", switchState);
    if (switchState != cudaSuccess) {
        return;
    }
    struct ResetEosAllCache {
        int *cuda_eos_ids = nullptr;
        int capacity = 0;
        std::vector<int> last_eos_ids;
    };
    static thread_local std::map<int, ResetEosAllCache> caches;

    int device = FastllmCudaGetDevice();
    ResetEosAllCache &cache = caches[device];
    if (cache.cuda_eos_ids == nullptr || cache.capacity < (int)eos_ids.size()) {
        if (cache.cuda_eos_ids != nullptr) {
            FastllmCudaFree(cache.cuda_eos_ids);
        }
        cache.cuda_eos_ids = (int*)FastllmCudaMalloc(sizeof(int) * eos_ids.size());
        cache.capacity = (int)eos_ids.size();
        cache.last_eos_ids.clear();
    }
    if (cache.last_eos_ids != eos_ids) {
        FastllmCudaCopyFromHostToDevice(cache.cuda_eos_ids, (void*)eos_ids.data(), sizeof(int) * eos_ids.size());
        cache.last_eos_ids = eos_ids;
    }
    int total = batch * (int)eos_ids.size();
    FastllmCudaResetLogitsOfEOSAll <<< (total + 255) / 256, 256 >>> (
        total, (int)eos_ids.size(), logits->Count(0) / batch, (float*)logits->cudaData, cache.cuda_eos_ids);
    cudaError_t launchState = cudaGetLastError();
    checkCudaErrors("Error: CUDA error when reset logits of EOS all!", launchState);
    cudaError_t restoreState = cudaSetDevice(originalDevice);
    checkCudaErrors("Error: CUDA error when restoring device after reset logits of EOS all!", restoreState);
}

template <typename T>
__global__ void FastllmRecurrentGatedDeltaRuleKernel(
    T* last_recurrent_state,  // [n0, n1, n2, n3]
    const T* g_t,              // [n0, n1]
    const T* k_t,              // [n0, n1, n2]
    const T* v_t,              // [n0, n1, n3]
    const T* b_t,              // [n0, n1]
    const T* q_t,              // [n0, n1, n2]
    T* core_attn_out,          // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float qScale)
{
    // Each block handles one (n0, n1) position
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= n0 || head_idx >= n1) return;
    
    int base_idx = batch_idx * n1 + head_idx;
    int tid = threadIdx.x;
    
    // Shared memory for temporary storage
    extern __shared__ float shared_mem[];
    float* kv_mem = shared_mem;  // size: n3
    float* delta = &shared_mem[n3];  // size: n3
    
    // Step 1: Scale last_recurrent_state by g_t
    float g_val = expf((float)g_t[base_idx]);
    
    // Each thread handles multiple elements if n2*n3 > blockDim.x
    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int state_idx = base_idx * n2 * n3 + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] * g_val);
    }
    __syncthreads();
    
    // Step 2: Compute kv_mem = sum(last_recurrent_state * k_t.unsqueeze(-1), dim=-2)
    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = (float)k_t[base_idx / group * n2 + j];
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * k_val;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();
    
    // Step 3: Compute delta = (v_t - kv_mem) * b_t
    float b_val = (float)b_t[base_idx];
    if (tid < n3) {
        float v_val = (float)v_t[base_idx * n3 + tid];
        delta[tid] = (v_val - kv_mem[tid]) * b_val;
    }
    __syncthreads();
    
    // Step 4: Update last_recurrent_state += k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int j = idx / n3;
        int k = idx % n3;
        float k_val = (float)k_t[base_idx / group * n2 + j];
        int state_idx = base_idx * n2 * n3 + idx;

        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] + k_val * delta[k]);
    }
    __syncthreads();
    
    // Step 5: Compute core_attn_out = sum(last_recurrent_state * q_t.unsqueeze(-1), dim=-2)
    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float q_val = FastllmCudaValueToFloat(q_t[base_idx / group * n2 + j]);
            if (qScale != 1.0f) {
                if constexpr (std::is_same_v<T, float>) {
                    q_val *= qScale;
                } else if constexpr (std::is_same_v<T, half>) {
                    half qScaleHalf = __float2half_rn(qScale);
#ifdef CUDA_NO_TENSOR_CORE
                    q_val = __half2float(__float2half(__half2float(q_t[base_idx / group * n2 + j]) * __half2float(qScaleHalf)));
#else
                    q_val = __half2float(__hmul(q_t[base_idx / group * n2 + j], qScaleHalf));
#endif
                } else {
                    q_val = FastllmCudaValueToFloat(FastllmCudaFloatToValue<T>(q_val * qScale));
                }
            }
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * q_val;
        }
        core_attn_out[base_idx * n3 + tid] = (T)sum;
    }
}

template <typename T>
__global__ void FastllmRecurrentGatedDeltaRuleBatchPointerKernel(
    T** last_recurrent_states, // batch pointers, each [1, n1, n2, n3]
    const T* g_t,              // [batch, n1]
    const T* k_t,              // [batch, n1 / group, n2]
    const T* v_t,              // [batch, n1, n3]
    const T* b_t,              // [batch, n1]
    const T* q_t,              // [batch, n1 / group, n2]
    T* core_attn_out,          // [batch, n1, n3]
    int batch, int n1, int n2, int n3, int group, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (batch_idx >= batch || head_idx >= n1) return;

    T* last_recurrent_state = last_recurrent_states[batch_idx];
    int base_idx = batch_idx * n1 + head_idx;
    int state_head_base = head_idx * n2 * n3;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* kv_mem = shared_mem;
    float* delta = &shared_mem[n3];

    float g_val = expf((float)g_t[base_idx]);

    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int state_idx = state_head_base + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] * g_val);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = (float)k_t[base_idx / group * n2 + j];
            int state_idx = state_head_base + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * k_val;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();

    float b_val = (float)b_t[base_idx];
    if (tid < n3) {
        float v_val = (float)v_t[base_idx * n3 + tid];
        delta[tid] = (v_val - kv_mem[tid]) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int j = idx / n3;
        int k = idx % n3;
        float k_val = (float)k_t[base_idx / group * n2 + j];
        int state_idx = state_head_base + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] + k_val * delta[k]);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float q_val = FastllmCudaValueToFloat(q_t[base_idx / group * n2 + j]);
            if (qScale != 1.0f) {
                if constexpr (std::is_same_v<T, float>) {
                    q_val *= qScale;
                } else if constexpr (std::is_same_v<T, half>) {
                    half qScaleHalf = __float2half_rn(qScale);
#ifdef CUDA_NO_TENSOR_CORE
                    q_val = __half2float(__float2half(__half2float(q_t[base_idx / group * n2 + j]) * __half2float(qScaleHalf)));
#else
                    q_val = __half2float(__hmul(q_t[base_idx / group * n2 + j], qScaleHalf));
#endif
                } else {
                    q_val = FastllmCudaValueToFloat(FastllmCudaFloatToValue<T>(q_val * qScale));
                }
            }
            int state_idx = state_head_base + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * q_val;
        }
        core_attn_out[base_idx * n3 + tid] = (T)sum;
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleHalfTileKernel(
    half* last_recurrent_state,  // [n0, n1, n2, n3]
    const half* g_t,             // [n0, n1]
    const half* k_t,             // [n0, n1 / group, n2]
    const half* v_t,             // [n0, n1, n3]
    const half* b_t,             // [n0, n1]
    const half* q_t,             // [n0, n1 / group, n2]
    half* core_attn_out,         // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) return;

    int tile_v = n3 - v_base;
    if (tile_v > TILE_V) {
        tile_v = TILE_V;
    }

    int tid = threadIdx.x;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3 + v_base;

    extern __shared__ float shared_mem[];
    float *state_tile = shared_mem;                 // [n2, tile_v]
    float *delta = state_tile + n2 * tile_v;        // [tile_v]

    float g_val = expf(__half2float(g_t[base_idx]));

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        int state_idx = state_base + j * n3 + v_base + tv;
        float scaled = __half2float(__float2half_rn(__half2float(last_recurrent_state[state_idx]) * g_val));
        state_tile[idx] = scaled;
    }
    __syncthreads();

    float b_val = __half2float(b_t[base_idx]);
    if (tid < tile_v) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = __half2float(k_t[qk_base + j]);
            sum += state_tile[j * tile_v + tid] * k_val;
        }
        float v_val = __half2float(v_t[out_base + tid]);
        delta[tid] = (v_val - sum) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        float k_val = __half2float(k_t[qk_base + j]);
        float updated = __half2float(__float2half_rn(state_tile[idx] + k_val * delta[tv]));
        state_tile[idx] = updated;
        int state_idx = state_base + j * n3 + v_base + tv;
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < tile_v) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < n2; j++) {
            float q_val;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                q_val = __half2float(__float2half(__half2float(q_t[qk_base + j]) * __half2float(qScaleHalf)));
#else
                q_val = __half2float(__hmul(q_t[qk_base + j], qScaleHalf));
#endif
            } else {
                q_val = __half2float(q_t[qk_base + j]);
            }
            sum += state_tile[j * tile_v + tid] * q_val;
        }
        core_attn_out[out_base + tid] = __float2half_rn(sum);
    }
}

__global__ void FastllmLinearAttentionStateTransposeHalfKernel(
    const half *input, half *output, size_t totalHeads, int kdim, int vdim, bool kvToVk) {
    size_t stride = (size_t)kdim * vdim;
    size_t total = totalHeads * stride;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t head = idx / stride;
        size_t rem = idx - head * stride;
        if (kvToVk) {
            int k = rem / vdim;
            int v = rem - (size_t)k * vdim;
            output[head * stride + (size_t)v * kdim + k] = input[idx];
        } else {
            int v = rem / kdim;
            int k = rem - (size_t)v * kdim;
            output[head * stride + (size_t)k * vdim + v] = input[idx];
        }
    }
}

__global__ void FastllmLinearAttentionStateTransposeTiledHalfKernel(
    const half *input, half *output, int rows, int cols, size_t stride) {
    size_t head = blockIdx.z;
    int inputRow = blockIdx.y * 32 + threadIdx.y;
    int inputCol = blockIdx.x * 32 + threadIdx.x;
    __shared__ half tile[32][33];

#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int row = inputRow + offset;
        if (row < rows && inputCol < cols) {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[head * stride + (size_t)row * cols + inputCol];
        }
    }
    __syncthreads();

    int outputRow = blockIdx.x * 32 + threadIdx.y;
    int outputCol = blockIdx.y * 32 + threadIdx.x;
#pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        int row = outputRow + offset;
        if (row < cols && outputCol < rows) {
            output[head * stride + (size_t)row * rows + outputCol] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleNormBaTransposedHalfWarpKernel(
    half* last_recurrent_state,  // physical [n0, n1, n3, n2], logical [n0, n1, n2, n3]
    const half* a_t,             // [n0, n1]
    const half* b_t,             // [n0, n1]
    const half* k_t,             // [n0, n1 / group, n2], unnormalized
    const half* v_t,             // [n0, n1, n3]
    const half* q_t,             // [n0, n1 / group, n2], unnormalized
    const float* norm_weight,    // [n2]
    const float* a_log,          // [n1]
    const float* dt_bias,        // [n1]
    half* core_attn_out,         // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float eps, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + n2;
    float *warp_q = k_norm + n2;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    if (tid < 64) {
        int norm_warp = tid >> 5;
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float q_sum2 = qf.x * qf.x + qf.y * qf.y;
        float k_sum2 = kf.x * kf.x + kf.y * kf.y;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
            k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
        }
        if (lane_id == 0) {
            warp_q[norm_warp] = q_sum2;
            warp_k[norm_warp] = k_sum2;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float q_val = tid < 2 ? warp_q[tid] : 0.0f;
        float k_val = tid < 2 ? warp_k[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_val += __shfl_down_sync(0xffffffff, q_val, offset);
            k_val += __shfl_down_sync(0xffffffff, k_val, offset);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(q_val / n2 + eps);
            scales[1] = rsqrtf(k_val / n2 + eps);
        }
    }
    __syncthreads();

    if (tid < 64) {
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float w0 = __ldg(&norm_weight[tid * 2]);
        float w1 = __ldg(&norm_weight[tid * 2 + 1]);
        q_norm[tid * 2] = qf.x * scales[0] * w0;
        q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
        k_norm[tid * 2] = kf.x * scales[1] * w0;
        k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
    }

    if (tid == 0) {
        float b_raw = __half2float(b_t[base_idx]);
        float g_raw = -__expf(a_log[head_idx]) *
                      softplus_fast(__half2float(a_t[base_idx]) + dt_bias[head_idx]);
        ba_values[0] = 1.0f / (1.0f + __expf(-b_raw));
        ba_values[1] = __expf(g_raw);
    }
    __syncthreads();

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= n3) {
        return;
    }

    half *state_row = last_recurrent_state + state_base + (size_t)v_col * n2;
    float g_val = ba_values[1];
    float sum_k = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        sum_k += (__half2float(state_row[j]) * g_val) * k_norm[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_k += __shfl_down_sync(0xffffffff, sum_k, offset);
    }
    float delta = (FastllmCudaValueToFloat(v_t[out_base + v_col]) - __shfl_sync(0xffffffff, sum_k, 0)) * ba_values[0];

    float sum_q = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        float updated = __half2float(state_row[j]) * g_val + k_norm[j] * delta;
        state_row[j] = __float2half_rn(updated);
        sum_q += updated * (q_norm[j] * qScale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_q += __shfl_down_sync(0xffffffff, sum_q, offset);
    }
    if (lane_id == 0) {
        core_attn_out[out_base + v_col] = __float2half_rn(sum_q);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleNormTransposedHalfWarpKernel(
    half* last_recurrent_state,  // physical [n0, n1, n3, n2], logical [n0, n1, n2, n3]
    const half* g_t,             // [n0, n1]
    const half* k_t,             // [n0, n1 / group, n2], unnormalized
    const half* v_t,             // [n0, n1, n3]
    const half* b_t,             // [n0, n1]
    const half* q_t,             // [n0, n1 / group, n2], unnormalized
    const float* norm_weight,    // [n2]
    half* core_attn_out,         // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float eps, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + n2;
    float *warp_q = k_norm + n2;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;

    if (tid < 64) {
        int norm_warp = tid >> 5;
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float q_sum2 = qf.x * qf.x + qf.y * qf.y;
        float k_sum2 = kf.x * kf.x + kf.y * kf.y;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
            k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
        }
        if (lane_id == 0) {
            warp_q[norm_warp] = q_sum2;
            warp_k[norm_warp] = k_sum2;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float q_val = tid < 2 ? warp_q[tid] : 0.0f;
        float k_val = tid < 2 ? warp_k[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_val += __shfl_down_sync(0xffffffff, q_val, offset);
            k_val += __shfl_down_sync(0xffffffff, k_val, offset);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(q_val / n2 + eps);
            scales[1] = rsqrtf(k_val / n2 + eps);
        }
    }
    __syncthreads();

    if (tid < 64) {
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float w0 = __ldg(&norm_weight[tid * 2]);
        float w1 = __ldg(&norm_weight[tid * 2 + 1]);
        q_norm[tid * 2] = qf.x * scales[0] * w0;
        q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
        k_norm[tid * 2] = kf.x * scales[1] * w0;
        k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
    }
    __syncthreads();

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= n3) {
        return;
    }

    half *state_row = last_recurrent_state + state_base + (size_t)v_col * n2;
    float g_val = expf(__half2float(g_t[base_idx]));
    float sum_k = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        sum_k += (__half2float(state_row[j]) * g_val) * k_norm[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_k += __shfl_down_sync(0xffffffff, sum_k, offset);
    }
    float delta = (FastllmCudaValueToFloat(v_t[out_base + v_col]) - __shfl_sync(0xffffffff, sum_k, 0)) *
                  __half2float(b_t[base_idx]);

    float sum_q = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        float updated = __half2float(state_row[j]) * g_val + k_norm[j] * delta;
        state_row[j] = __float2half_rn(updated);
        sum_q += updated * (q_norm[j] * qScale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_q += __shfl_down_sync(0xffffffff, sum_q, offset);
    }
    if (lane_id == 0) {
        core_attn_out[out_base + v_col] = __float2half_rn(sum_q);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleBatchPointerHalfTileKernel(
    half** last_recurrent_states, // batch pointers, each [1, n1, n2, n3]
    const half* g_t,              // [batch, n1]
    const half* k_t,              // [batch, n1 / group, n2]
    const half* v_t,              // [batch, n1, n3]
    const half* b_t,              // [batch, n1]
    const half* q_t,              // [batch, n1 / group, n2]
    half* core_attn_out,          // [batch, n1, n3]
    int batch, int n1, int n2, int n3, int group, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= batch || head_idx >= n1 || v_base >= n3) return;

    int tile_v = n3 - v_base;
    if (tile_v > TILE_V) {
        tile_v = TILE_V;
    }

    int tid = threadIdx.x;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_head_base = head_idx * n2 * n3;
    int out_base = base_idx * n3 + v_base;
    half *last_recurrent_state = last_recurrent_states[batch_idx];

    extern __shared__ float shared_mem[];
    float *state_tile = shared_mem;
    float *delta = state_tile + n2 * tile_v;

    float g_val = expf(__half2float(g_t[base_idx]));

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        int state_idx = state_head_base + j * n3 + v_base + tv;
        float scaled = __half2float(__float2half_rn(__half2float(last_recurrent_state[state_idx]) * g_val));
        state_tile[idx] = scaled;
    }
    __syncthreads();

    float b_val = __half2float(b_t[base_idx]);
    if (tid < tile_v) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = __half2float(k_t[qk_base + j]);
            sum += state_tile[j * tile_v + tid] * k_val;
        }
        float v_val = __half2float(v_t[out_base + tid]);
        delta[tid] = (v_val - sum) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        float k_val = __half2float(k_t[qk_base + j]);
        float updated = __half2float(__float2half_rn(state_tile[idx] + k_val * delta[tv]));
        state_tile[idx] = updated;
        int state_idx = state_head_base + j * n3 + v_base + tv;
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < tile_v) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < n2; j++) {
            float q_val;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                q_val = __half2float(__float2half(__half2float(q_t[qk_base + j]) * __half2float(qScaleHalf)));
#else
                q_val = __half2float(__hmul(q_t[qk_base + j], qScaleHalf));
#endif
            } else {
                q_val = __half2float(q_t[qk_base + j]);
            }
            sum += state_tile[j * tile_v + tid] * q_val;
        }
        core_attn_out[out_base + tid] = __float2half_rn(sum);
    }
}

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float qScale) {
    // Get dimensions
    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    
    // Move data to GPU if not already there
    float *d_last_state = (float*)last_recurrent_state.cudaData;
    float *d_g = (float*)g.cudaData;
    float *d_k = (float*)k.cudaData;
    float *d_v = (float*)v.cudaData;
    float *d_b = (float*)b.cudaData;
    float *d_q = (float*)q.cudaData;
    float *d_out = (float*)core_attn_out.cudaData;

    int group = v.dims[1] / q.dims[1];
    
    // Configure kernel launch parameters
    dim3 gridDim(n0, n1);  // One block per (batch, head) pair
    int threadsPerBlock = min(256, CUDA_MAX(n2 * n3, n3));
    
    // Calculate shared memory size
    size_t sharedMemSize = 2 * n3 * sizeof(float);  // for kv_mem and delta
    
    // Launch kernel
    if (q.dataType == fastllm::DataType::FLOAT32) {
        FastllmRecurrentGatedDeltaRuleKernel <float> <<<gridDim, threadsPerBlock, sharedMemSize>>>(
            d_last_state, d_g, d_k, d_v, d_b, d_q, d_out,
            n0, n1, n2, n3, group, qScale
        ); 
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        constexpr int tileV = 8;
        size_t tileSharedMemSize = ((size_t)n2 * tileV + tileV) * sizeof(float);
        if (n2 > 0 && n3 > 0 && tileSharedMemSize <= 48 * 1024) {
            dim3 tileGrid(n0, n1, (n3 + tileV - 1) / tileV);
            int tileThreads = 256;
            FastllmRecurrentGatedDeltaRuleHalfTileKernel<tileV><<<tileGrid, tileThreads, tileSharedMemSize>>>(
                (half*)d_last_state, (half*)d_g, (half*)d_k, (half*)d_v, (half*)d_b, (half*)d_q, (half*)d_out,
                n0, n1, n2, n3, group, qScale
            );
        } else {
            FastllmRecurrentGatedDeltaRuleKernel <half> <<<gridDim, threadsPerBlock, sharedMemSize>>>(
                (half*)d_last_state, (half*)d_g, (half*)d_k, (half*)d_v, (half*)d_b, (half*)d_q, (half*)d_out,
                n0, n1, n2, n3, group, qScale
            );
        }
    }
    
    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRule.", cudaGetLastError());
}

static bool FastllmLinearAttentionStateTransposeFloat16(fastllm::Data &last_recurrent_state, bool kvToVk) {
    if (last_recurrent_state.isFake ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        last_recurrent_state.cudaData == nullptr ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] <= 0 ||
        last_recurrent_state.dims[1] <= 0 ||
        last_recurrent_state.dims[2] <= 0 ||
        last_recurrent_state.dims[3] <= 0) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    size_t totalHeads = (size_t)n0 * n1;
    size_t total = totalHeads * n2 * n3;
    size_t bytes = last_recurrent_state.GetBytes();
    void *oldData = last_recurrent_state.cudaData;
    bool oldBorrowed = last_recurrent_state.cudaDataBorrowed;
    void *newData = last_recurrent_state.directMemory ? FastllmCudaDirectMalloc(bytes) : FastllmCudaMalloc(bytes);
    if (newData == nullptr) {
        return false;
    }

    if (totalHeads <= 65535) {
        int rows = kvToVk ? n2 : n3;
        int cols = kvToVk ? n3 : n2;
        dim3 blocks((cols + 31) / 32, (rows + 31) / 32,
                    (unsigned int)totalHeads);
        dim3 threads(32, 8);
        FastllmLinearAttentionStateTransposeTiledHalfKernel
            <<<blocks, threads>>>(
                (const half*)oldData, (half*)newData, rows, cols,
                (size_t)n2 * n3);
    } else {
        int threads = 256;
        int blocks =
            (int)std::min<size_t>((total + threads - 1) / threads, 65535);
        FastllmLinearAttentionStateTransposeHalfKernel<<<blocks, threads>>>(
            (const half*)oldData, (half*)newData, totalHeads, n2, n3,
            kvToVk);
    }
    checkCudaErrors("Error: CUDA error in FastllmLinearAttentionStateTransposeFloat16.", cudaGetLastError());

    if (!oldBorrowed) {
        if (last_recurrent_state.directMemory) {
            FastllmCudaDirectFree(oldData);
        } else {
            FastllmCudaFree(oldData);
        }
    } else if (last_recurrent_state.isPagedKVCache &&
               last_recurrent_state.pagedKVCacheData != nullptr &&
               !last_recurrent_state.pageIndex.empty()) {
        last_recurrent_state.pagedKVCacheData->ReleasePageIndices(last_recurrent_state.pageIndex);
        last_recurrent_state.pageIndex.clear();
        last_recurrent_state.pagedKVCacheData = nullptr;
        last_recurrent_state.isPagedKVCache = false;
    }
    last_recurrent_state.cudaData = newData;
    last_recurrent_state.cudaDataBorrowed = false;
    return true;
}

bool FastllmLinearAttentionStateTransposeKVToVKFloat16(fastllm::Data &last_recurrent_state) {
    return FastllmLinearAttentionStateTransposeFloat16(last_recurrent_state, true);
}

bool FastllmLinearAttentionStateTransposeVKToKVFloat16(fastllm::Data &last_recurrent_state) {
    return FastllmLinearAttentionStateTransposeFloat16(last_recurrent_state, false);
}

bool FastllmRecurrentGatedDeltaRuleNormTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        g.dataDevice != fastllm::DataDevice::CUDA ||
        b.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (last_recurrent_state.dims.size() != 4 ||
        q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        g.dims.size() != 3 || b.dims.size() != 3 ||
        normWeight.dims.size() != 1) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    if (n0 <= 0 || n1 <= 0 || n2 != 128 || n3 <= 0 ||
        normWeight.dims[0] != n2 ||
        q.dims[0] != n0 || k.dims[0] != n0 || v.dims[0] != n0 ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        v.dims[1] != n1 ||
        g.dims[0] != n0 || b.dims[0] != n0 ||
        g.dims[1] != n1 || b.dims[1] != n1 ||
        g.dims[2] != 1 || b.dims[2] != 1 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] ||
        n1 % q.dims[1] != 0) {
        return false;
    }

    int group = n1 / q.dims[1];
    constexpr int tileV = 8;
    size_t sharedMemSize = (2 * (size_t)n2 + 8) * sizeof(float);
    if (sharedMemSize > 48 * 1024) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({n0, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 grid(n0, n1, (n3 + tileV - 1) / tileV);
    FastllmRecurrentGatedDeltaRuleNormTransposedHalfWarpKernel<tileV><<<grid, tileV * 32, sharedMemSize>>>(
        (half*)last_recurrent_state.cudaData,
        (half*)g.cudaData,
        (half*)k.cudaData,
        (half*)v.cudaData,
        (half*)b.cudaData,
        (half*)q.cudaData,
        (float*)normWeight.cudaData,
        (half*)core_attn_out.cudaData,
        n0, n1, n2, n3, group, eps, qScale
    );
    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleNormTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &a, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &aLog, fastllm::Data &dtBias, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        a.dataDevice != fastllm::DataDevice::CUDA ||
        b.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        a.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (last_recurrent_state.dims.size() != 4 ||
        q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        a.dims.size() != 3 || b.dims.size() != 3 ||
        normWeight.dims.size() != 1 ||
        aLog.dims.size() != 1 ||
        dtBias.dims.size() != 1) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    if (n0 <= 0 || n1 <= 0 || n2 != 128 || n3 <= 0 ||
        normWeight.dims[0] != n2 ||
        aLog.dims[0] != n1 ||
        dtBias.dims[0] != n1 ||
        q.dims[0] != n0 || k.dims[0] != n0 || v.dims[0] != n0 ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        v.dims[1] != n1 ||
        a.dims[0] != n0 || b.dims[0] != n0 ||
        a.dims[1] != n1 || b.dims[1] != n1 ||
        a.dims[2] != 1 || b.dims[2] != 1 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] ||
        n1 % q.dims[1] != 0) {
        return false;
    }

    int group = n1 / q.dims[1];
    constexpr int tileV = 8;
    size_t sharedMemSize = (2 * (size_t)n2 + 10) * sizeof(float);
    if (sharedMemSize > 48 * 1024) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({n0, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 grid(n0, n1, (n3 + tileV - 1) / tileV);
    FastllmRecurrentGatedDeltaRuleNormBaTransposedHalfWarpKernel<tileV><<<grid, tileV * 32, sharedMemSize>>>(
        (half*)last_recurrent_state.cudaData,
        (half*)a.cudaData,
        (half*)b.cudaData,
        (half*)k.cudaData,
        (half*)v.cudaData,
        (half*)q.cudaData,
        (float*)normWeight.cudaData,
        (float*)aLog.cudaData,
        (float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        n0, n1, n2, n3, group, eps, qScale
    );
    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int n1 = first_recurrent_state.dims[1];
    int n2 = first_recurrent_state.dims[2];
    int n3 = first_recurrent_state.dims[3];
    if (q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        g.dims.size() != 3 || b.dims.size() != 3 ||
        q.dims[0] != batch || k.dims[0] != batch || v.dims[0] != batch ||
        g.dims[0] != batch || b.dims[0] != batch ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        g.dims[2] != 1 || b.dims[2] != 1 ||
        v.dims[1] != n1 || g.dims[1] != n1 || b.dims[1] != n1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] || n1 % q.dims[1] != 0) {
        return false;
    }
    int group = v.dims[1] / q.dims[1];

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 gridDim(batch, n1);
    int threadsPerBlock = min(256, CUDA_MAX(n2 * n3, n3));
    size_t sharedMemSize = 2 * n3 * sizeof(float);

    if (q.dataType == fastllm::DataType::FLOAT32) {
        FastllmRecurrentGatedDeltaRuleBatchPointerKernel<float><<<gridDim, threadsPerBlock, sharedMemSize>>>(
            (float**)cudaStatePointers, (float*)g.cudaData, (float*)k.cudaData, (float*)v.cudaData,
            (float*)b.cudaData, (float*)q.cudaData, (float*)core_attn_out.cudaData,
            batch, n1, n2, n3, group, qScale
        );
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        constexpr int tileV = 8;
        size_t tileSharedMemSize = ((size_t)n2 * tileV + tileV) * sizeof(float);
        if (n2 > 0 && n3 > 0 && tileSharedMemSize <= 48 * 1024) {
            dim3 tileGrid(batch, n1, (n3 + tileV - 1) / tileV);
            int tileThreads = 256;
            FastllmRecurrentGatedDeltaRuleBatchPointerHalfTileKernel<tileV><<<tileGrid, tileThreads, tileSharedMemSize>>>(
                (half**)cudaStatePointers, (half*)g.cudaData, (half*)k.cudaData, (half*)v.cudaData,
                (half*)b.cudaData, (half*)q.cudaData, (half*)core_attn_out.cudaData,
                batch, n1, n2, n3, group, qScale
            );
        } else {
            FastllmRecurrentGatedDeltaRuleBatchPointerKernel<half><<<gridDim, threadsPerBlock, sharedMemSize>>>(
                (half**)cudaStatePointers, (half*)g.cudaData, (half*)k.cudaData, (half*)v.cudaData,
                (half*)b.cudaData, (half*)q.cudaData, (half*)core_attn_out.cudaData,
                batch, n1, n2, n3, group, qScale
            );
        }
    } else {
        return false;
    }

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchDevicePointers.", cudaGetLastError());
    return true;
}

void FastllmRecurrentGatedDeltaRuleBatch(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out, float qScale) {
    int batch = (int)last_recurrent_states.size();
    std::vector<void*> pointers(batch);
    for (int i = 0; i < batch; i++) {
        pointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = FastllmCudaStagePointers(pointers);

    FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
        q, k, v, g, b, *last_recurrent_states[0], cudaPointers, batch, core_attn_out, qScale);
}

__global__ void FastllmRecurrentGatedDeltaRuleBatchFromConvBaHalfKernel(
    half **last_recurrent_states,
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *core_attn_out,
    int batch, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *statePool, const int *slotIds, int stateStride) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (batch_idx >= batch || head_idx >= numVHeads) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int numWarps = (blockDim.x + 31) / 32;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int convBase = batch_idx * qkvDim;
    int qOffset = convBase + qHead * headKDim;
    int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
    int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;

    extern __shared__ float shared_mem[];
    float *kv_mem = shared_mem;
    float *delta = kv_mem + headVDim;
    float *warpQ = delta + headVDim;
    float *warpK = warpQ + numWarps;
    float *scales = warpK + numWarps;

    float qSum2 = 0.0f;
    float kSum2 = 0.0f;
    for (int j = tid; j < headKDim; j += blockDim.x) {
        float qx = __half2float(convOutput[qOffset + j]);
        float kx = __half2float(convOutput[kOffset + j]);
        qSum2 += qx * qx;
        kSum2 += kx * kx;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        qSum2 += __shfl_down_sync(0xffffffff, qSum2, offset);
        kSum2 += __shfl_down_sync(0xffffffff, kSum2, offset);
    }
    if (lane_id == 0) {
        warpQ[warp_id] = qSum2;
        warpK[warp_id] = kSum2;
    }
    __syncthreads();
    if (warp_id == 0) {
        float qVal = lane_id < numWarps ? warpQ[lane_id] : 0.0f;
        float kVal = lane_id < numWarps ? warpK[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            qVal += __shfl_down_sync(0xffffffff, qVal, offset);
            kVal += __shfl_down_sync(0xffffffff, kVal, offset);
        }
        if (lane_id == 0) {
            scales[0] = rsqrtf(qVal / headKDim + eps);
            scales[1] = rsqrtf(kVal / headKDim + eps);
        }
    }
    __syncthreads();
    float qNormScale = scales[0];
    float kNormScale = scales[1];

    const half *baRow = ba + batch_idx * 2 * numVHeads;
    float bRaw = __half2float(baRow[head_idx]);
#ifdef CUDA_NO_TENSOR_CORE
    float bVal = 1.0f / (1.0f + expf(-bRaw));
#else
    half bHalfRaw = baRow[head_idx];
    float bVal = __half2float(__hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(-bHalfRaw))));
#endif
    float aRaw = __half2float(baRow[numVHeads + head_idx]);
    float gStored = __half2float(__float2half_rn(-exp((double)aLog[head_idx]) * softplus(aRaw + dtBias[head_idx])));
    float gVal = expf(gStored);

    half *last_recurrent_state = statePool != nullptr ?
        (slotIds == nullptr ? statePool : statePool + (size_t)slotIds[batch_idx] * stateStride) :
        last_recurrent_states[batch_idx];
    int stateHeadBase = head_idx * headKDim * headVDim;

    for (int idx = tid; idx < headKDim * headVDim; idx += blockDim.x) {
        int state_idx = stateHeadBase + idx;
        last_recurrent_state[state_idx] = __float2half_rn(__half2float(last_recurrent_state[state_idx]) * gVal);
    }
    __syncthreads();

    if (tid < headVDim) {
        float sum = 0.0f;
        for (int j = 0; j < headKDim; j++) {
            float kRaw = __half2float(convOutput[kOffset + j]);
            float kNorm = __half2float(__float2half_rn(kRaw * kNormScale * normWeight[j]));
            int state_idx = stateHeadBase + j * headVDim + tid;
            sum += __half2float(last_recurrent_state[state_idx]) * kNorm;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();

    if (tid < headVDim) {
        float vVal = __half2float(convOutput[vOffset + tid]);
        delta[tid] = (vVal - kv_mem[tid]) * bVal;
    }
    __syncthreads();

    for (int idx = tid; idx < headKDim * headVDim; idx += blockDim.x) {
        int j = idx / headVDim;
        int k = idx % headVDim;
        float kRaw = __half2float(convOutput[kOffset + j]);
        float kNorm = __half2float(__float2half_rn(kRaw * kNormScale * normWeight[j]));
        int state_idx = stateHeadBase + idx;
        float updated = __half2float(last_recurrent_state[state_idx]) + kNorm * delta[k];
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < headVDim) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < headKDim; j++) {
            float qRaw = __half2float(convOutput[qOffset + j]);
            half qNormHalf = __float2half_rn(qRaw * qNormScale * normWeight[j]);
            float qVal;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                qVal = __half2float(__float2half(__half2float(qNormHalf) * __half2float(qScaleHalf)));
#else
                qVal = __half2float(__hmul(qNormHalf, qScaleHalf));
#endif
            } else {
                qVal = __half2float(qNormHalf);
            }
            int state_idx = stateHeadBase + j * headVDim + tid;
            sum += __half2float(last_recurrent_state[state_idx]) * qVal;
        }
        core_attn_out[(batch_idx * numVHeads + head_idx) * headVDim + tid] = __float2half_rn(sum);
    }
}

template <int TILE_V, bool EXACT_NORM_128>
__global__ void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel(
    half **last_recurrent_states,
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *core_attn_out,
    int batch, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *statePool, const int *slotIds, int stateStride) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= batch || head_idx >= numVHeads || v_base >= headVDim) {
        return;
    }

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int convBase = batch_idx * qkvDim;
    int qOffset = convBase + qHead * headKDim;
    int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
    int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;
    int outBase = (batch_idx * numVHeads + head_idx) * headVDim;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + headKDim;
    float *warp_q = k_norm + headKDim;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    if constexpr (EXACT_NORM_128) {
        // The legacy path reduces half2[0..31] and half2[32..63] in two
        // independent warps, then adds the two warp sums.  One warp can keep
        // those two reduction trees separate, preserve the exact final add,
        // reuse the q/k loads, and eliminate two block-wide barriers.
        if (warp_id == 0) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            float2 q0 = __half22float2(q_h2[lane_id]);
            float2 q1 = __half22float2(q_h2[lane_id + 32]);
            float2 k0 = __half22float2(k_h2[lane_id]);
            float2 k1 = __half22float2(k_h2[lane_id + 32]);
            float qSum0 = q0.x * q0.x + q0.y * q0.y;
            float qSum1 = q1.x * q1.x + q1.y * q1.y;
            float kSum0 = k0.x * k0.x + k0.y * k0.y;
            float kSum1 = k1.x * k1.x + k1.y * k1.y;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                qSum0 += __shfl_down_sync(0xffffffffu, qSum0, offset);
                qSum1 += __shfl_down_sync(0xffffffffu, qSum1, offset);
                kSum0 += __shfl_down_sync(0xffffffffu, kSum0, offset);
                kSum1 += __shfl_down_sync(0xffffffffu, kSum1, offset);
            }
            float qNormScale = 0.0f;
            float kNormScale = 0.0f;
            if (lane_id == 0) {
                qNormScale = rsqrtf((qSum0 + qSum1) / 128.0f + eps);
                kNormScale = rsqrtf((kSum0 + kSum1) / 128.0f + eps);
            }
            qNormScale = __shfl_sync(0xffffffffu, qNormScale, 0);
            kNormScale = __shfl_sync(0xffffffffu, kNormScale, 0);

            int index0 = lane_id * 2;
            int index1 = (lane_id + 32) * 2;
            float w00 = __ldg(normWeight + index0);
            float w01 = __ldg(normWeight + index0 + 1);
            float w10 = __ldg(normWeight + index1);
            float w11 = __ldg(normWeight + index1 + 1);
            q_norm[index0] = q0.x * qNormScale * w00;
            q_norm[index0 + 1] = q0.y * qNormScale * w01;
            q_norm[index1] = q1.x * qNormScale * w10;
            q_norm[index1 + 1] = q1.y * qNormScale * w11;
            k_norm[index0] = k0.x * kNormScale * w00;
            k_norm[index0 + 1] = k0.y * kNormScale * w01;
            k_norm[index1] = k1.x * kNormScale * w10;
            k_norm[index1 + 1] = k1.y * kNormScale * w11;
        }
        if (tid == 32) {
            const half *baRow = ba + (size_t)batch_idx * (numVHeads * 2);
            float bRaw = __half2float(baRow[head_idx]);
            float aRaw = __half2float(baRow[numVHeads + head_idx]);
            float gRaw = -__expf(aLog[head_idx]) * softplus_fast(aRaw + dtBias[head_idx]);
            ba_values[0] = 1.0f / (1.0f + __expf(-bRaw));
            ba_values[1] = __expf(gRaw);
        }
        __syncthreads();
    } else {
        if (tid < 64) {
            int norm_warp = tid >> 5;
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float q_sum2 = qf.x * qf.x + qf.y * qf.y;
            float k_sum2 = kf.x * kf.x + kf.y * kf.y;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
                k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
            }
            if (lane_id == 0) {
                warp_q[norm_warp] = q_sum2;
                warp_k[norm_warp] = k_sum2;
            }
        }
        __syncthreads();

        if (tid < 32) {
            float q_val = tid < 2 ? warp_q[tid] : 0.0f;
            float k_val = tid < 2 ? warp_k[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_val += __shfl_down_sync(0xffffffff, q_val, offset);
                k_val += __shfl_down_sync(0xffffffff, k_val, offset);
            }
            if (tid == 0) {
                scales[0] = rsqrtf(q_val / headKDim + eps);
                scales[1] = rsqrtf(k_val / headKDim + eps);
            }
        }
        __syncthreads();

        if (tid < 64) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float w0 = __ldg(&normWeight[tid * 2]);
            float w1 = __ldg(&normWeight[tid * 2 + 1]);
            q_norm[tid * 2] = qf.x * scales[0] * w0;
            q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
            k_norm[tid * 2] = kf.x * scales[1] * w0;
            k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
        }

        if (tid == 0) {
            const half *baRow = ba + (size_t)batch_idx * (numVHeads * 2);
            float bRaw = __half2float(baRow[head_idx]);
            float aRaw = __half2float(baRow[numVHeads + head_idx]);
            float gRaw = -__expf(aLog[head_idx]) * softplus_fast(aRaw + dtBias[head_idx]);
            ba_values[0] = 1.0f / (1.0f + __expf(-bRaw));
            ba_values[1] = __expf(gRaw);
        }
        __syncthreads();
    }

    constexpr int warpsPerBlock = 8;
    static_assert(TILE_V % warpsPerBlock == 0,
                  "recurrent V tile must divide evenly across eight warps");
    constexpr int rowsPerWarp = TILE_V / warpsPerBlock;
    int first_v_col = v_base + warp_id * rowsPerWarp;
    if (warp_id >= warpsPerBlock || first_v_col >= headVDim) {
        return;
    }

    half *last_recurrent_state = statePool != nullptr ?
        (slotIds == nullptr ? statePool : statePool + (size_t)slotIds[batch_idx] * stateStride) :
        last_recurrent_states[batch_idx];
    int stateHeadBase = head_idx * headKDim * headVDim;
    float gVal = ba_values[1];

    // headKDim is fixed to 128 for this specialization. For large batches a
    // warp processes four adjacent V rows. This matches vLLM's 32-row packed
    // decode tile and amortizes Q/K normalization and block scheduling four
    // ways, while completing one row at a time keeps register pressure close
    // to the original 8-row kernel.
#pragma unroll
    for (int row = 0; row < rowsPerWarp; row++) {
        int v_col = first_v_col + row;
        if (v_col >= headVDim) {
            continue;
        }
        half *state_row = last_recurrent_state + stateHeadBase +
                          (size_t)v_col * headKDim;
        float decayedState[4];
        float sumK = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            int j = lane_id + item * 32;
            decayedState[item] = __half2float(state_row[j]) * gVal;
            sumK += decayedState[item] * k_norm[j];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sumK += __shfl_down_sync(0xffffffff, sumK, offset);
        }
        float delta = (__half2float(convOutput[vOffset + v_col]) -
                       __shfl_sync(0xffffffff, sumK, 0)) * ba_values[0];

        float sumQ = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            int j = lane_id + item * 32;
            float updated = decayedState[item] + k_norm[j] * delta;
            state_row[j] = __float2half_rn(updated);
            sumQ += updated * (q_norm[j] * qScale);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sumQ += __shfl_down_sync(0xffffffff, sumQ, offset);
        }
        if (lane_id == 0) {
            core_attn_out[outBase + v_col] = __float2half_rn(sumQ);
        }
    }
}

static bool LaunchFastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarp(
    half **lastRecurrentStates, const half *convOutput, const half *ba,
    const float *normWeight, const float *aLog, const float *dtBias,
    half *coreAttnOut,
    int batch, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *statePool, const int *slotIds, int stateStride,
    int tileV, bool exactNorm128) {
    if (headKDim != 128 || tileV != 8) {
        return false;
    }
    constexpr int threads = 8 * 32;
    bool useTile32 = batch >= 8 &&
        !FastllmCudaEnvFlagEnabled(
            "FASTLLM_QWEN35_DISABLE_RECURRENT_TILE32");
    size_t sharedBytes = (2 * (size_t)headKDim + 8) * sizeof(float);

#define FASTLLM_LAUNCH_RECURRENT_FROM_CONV_BA(TILE, EXACT) \
    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<TILE, EXACT> \
        <<<grid, threads, sharedBytes>>>( \
            lastRecurrentStates, convOutput, ba, normWeight, aLog, dtBias, \
            coreAttnOut, batch, numKHeads, numVHeads, headKDim, headVDim, \
            eps, qScale, statePool, slotIds, stateStride)

    if (useTile32) {
        dim3 grid(batch, numVHeads, (headVDim + 31) / 32);
        FASTLLM_LAUNCH_RECURRENT_FROM_CONV_BA(32, false);
    } else if (exactNorm128) {
        dim3 grid(batch, numVHeads, (headVDim + 7) / 8);
        FASTLLM_LAUNCH_RECURRENT_FROM_CONV_BA(8, true);
    } else {
        dim3 grid(batch, numVHeads, (headVDim + 7) / 8);
        FASTLLM_LAUNCH_RECURRENT_FROM_CONV_BA(8, false);
    }
#undef FASTLLM_LAUNCH_RECURRENT_FROM_CONV_BA
    return true;
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel(
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *last_recurrent_state, half **statePointers,
    half *core_attn_out,
    int seqLen, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *snap0, half *snap1, half *snap2, half *snap3, half *snap4,
    half **snapshotPointers, int numSnaps) {
    int head_idx = blockIdx.x;
    int v_base = blockIdx.y * TILE_V;
    if (head_idx >= numVHeads || v_base >= headVDim) {
        return;
    }

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int v_col = v_base + warp_id;
    bool activeV = warp_id < TILE_V && v_col < headVDim;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + headKDim;
    float *warp_q = k_norm + headKDim;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    int batchIndex = blockIdx.z;
    if (statePointers != nullptr) {
        last_recurrent_state = statePointers[batchIndex];
    }
    int stateHeadBase = head_idx * headKDim * headVDim;
    half *state_row = activeV ?
        last_recurrent_state + stateHeadBase + (size_t)v_col * headKDim :
        last_recurrent_state;

    for (int token = 0; token < seqLen; token++) {
        int convBase = (batchIndex * seqLen + token) * qkvDim;
        int qOffset = convBase + qHead * headKDim;
        int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
        int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;
        int outBase = ((batchIndex * seqLen + token) * numVHeads + head_idx) *
                      headVDim;

        if (tid < 64) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float q_sum2 = qf.x * qf.x + qf.y * qf.y;
            float k_sum2 = kf.x * kf.x + kf.y * kf.y;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
                k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
            }
            if (lane_id == 0) {
                int norm_warp = tid >> 5;
                warp_q[norm_warp] = q_sum2;
                warp_k[norm_warp] = k_sum2;
            }
        }
        __syncthreads();

        if (tid < 32) {
            float q_val = tid < 2 ? warp_q[tid] : 0.0f;
            float k_val = tid < 2 ? warp_k[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_val += __shfl_down_sync(0xffffffff, q_val, offset);
                k_val += __shfl_down_sync(0xffffffff, k_val, offset);
            }
            if (tid == 0) {
                scales[0] = rsqrtf(q_val / headKDim + eps);
                scales[1] = rsqrtf(k_val / headKDim + eps);
            }
        }
        __syncthreads();

        if (tid < 64) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float w0 = __ldg(&normWeight[tid * 2]);
            float w1 = __ldg(&normWeight[tid * 2 + 1]);
            q_norm[tid * 2] = qf.x * scales[0] * w0;
            q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
            k_norm[tid * 2] = kf.x * scales[1] * w0;
            k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
        }

        if (tid == 0) {
            const half *baRow = ba +
                (size_t)(batchIndex * seqLen + token) * (numVHeads * 2);
            float bRaw = __half2float(baRow[head_idx]);
            float aRaw = __half2float(baRow[numVHeads + head_idx]);
            float gRaw = -__expf(aLog[head_idx]) * softplus_fast(aRaw + dtBias[head_idx]);
            ba_values[0] = 1.0f / (1.0f + __expf(-bRaw));
            ba_values[1] = __expf(gRaw);
        }
        __syncthreads();

        if (activeV) {
            float gVal = ba_values[1];
            float sumK = 0.0f;
            for (int j = lane_id; j < headKDim; j += 32) {
                sumK += (__half2float(state_row[j]) * gVal) * k_norm[j];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sumK += __shfl_down_sync(0xffffffff, sumK, offset);
            }
            float delta = (__half2float(convOutput[vOffset + v_col]) -
                           __shfl_sync(0xffffffff, sumK, 0)) * ba_values[0];

            float sumQ = 0.0f;
            half *snapBase = nullptr;
            if (token < numSnaps && snapshotPointers != nullptr) {
                snapBase = snapshotPointers[batchIndex * numSnaps + token];
            } else if (token < numSnaps) {
                switch (token) {
                    case 0: snapBase = snap0; break;
                    case 1: snapBase = snap1; break;
                    case 2: snapBase = snap2; break;
                    case 3: snapBase = snap3; break;
                    case 4: snapBase = snap4; break;
                    default: break;
                }
            }
            half *snap_row = snapBase != nullptr ?
                snapBase + stateHeadBase + (size_t)v_col * headKDim : nullptr;
            for (int j = lane_id; j < headKDim; j += 32) {
                float updated = __half2float(state_row[j]) * gVal + k_norm[j] * delta;
                half updatedHalf = __float2half_rn(updated);
                state_row[j] = updatedHalf;
                if (snap_row != nullptr) {
                    snap_row[j] = updatedHalf;
                }
                sumQ += updated * (q_norm[j] * qScale);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sumQ += __shfl_down_sync(0xffffffff, sumQ, offset);
            }
            if (lane_id == 0) {
                core_attn_out[outBase + v_col] = __float2half_rn(sumQ);
            }
        }
        __syncthreads();
    }
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        first_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim <= 0 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dims[0] != 1 ||
        first_recurrent_state.dims[1] != numVHeads ||
        first_recurrent_state.dims[2] != headKDim ||
        first_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    int threadsPerBlock = 256;
    int numWarps = (threadsPerBlock + 31) / 32;
    size_t sharedMemSize = (2 * headVDim + 2 * numWarps + 2) * sizeof(float);
    dim3 gridDim(batch, numVHeads);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaHalfKernel<<<gridDim, threadsPerBlock, sharedMemSize>>>(
        (half**)cudaStatePointers,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        nullptr, nullptr, 0
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        first_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !first_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dims[0] != 1 ||
        first_recurrent_state.dims[1] != numVHeads ||
        first_recurrent_state.dims[2] != headKDim ||
        first_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    LaunchFastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarp(
        (half**)cudaStatePointers,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        nullptr, nullptr, 0, 8,
        batch == 1 && numKHeads == 8 && numVHeads == 16 &&
            headKDim == 128 && headVDim == 128
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers.", cudaGetLastError());
    return true;
}

static bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16Impl(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale, int tileV, bool exactNorm128) {
    if (convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !last_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        last_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] != 1 ||
        last_recurrent_state.dims[1] != numVHeads ||
        last_recurrent_state.dims[2] != headKDim ||
        last_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({1, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    if (!LaunchFastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarp(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        1, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)last_recurrent_state.cudaData, nullptr, 0, tileV, exactNorm128)) {
        return false;
    }

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16WithConfig(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale, int tileV, bool exactNorm128) {
    return FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16Impl(
        convOutput, ba, normWeight, aLog, dtBias,
        last_recurrent_state, core_attn_out,
        numKHeads, numVHeads, headKDim, headVDim,
        eps, qScale, tileV, exactNorm128);
}

bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    return FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16Impl(
        convOutput, ba, normWeight, aLog, dtBias,
        last_recurrent_state, core_attn_out,
        numKHeads, numVHeads, headKDim, headVDim,
        eps, qScale, 8,
        numKHeads == 8 && numVHeads == 16 &&
            headKDim == 128 && headVDim == 128);
}

bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    return FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
        convOutput, ba, normWeight, aLog, dtBias,
        last_recurrent_state, core_attn_out,
        nullptr, 0,
        numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
}

bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    fastllm::Data **tokenStates, int numTokenStates,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (!FastllmCudaDataCanShareDevice(last_recurrent_state, convOutput) ||
        !FastllmCudaDataCanShareDevice(last_recurrent_state, ba) ||
        !FastllmCudaDataCanShareDevice(last_recurrent_state, normWeight) ||
        !FastllmCudaDataCanShareDevice(last_recurrent_state, aLog) ||
        !FastllmCudaDataCanShareDevice(last_recurrent_state, dtBias) ||
        core_attn_out.isFake || core_attn_out.cudaDataBorrowed ||
        core_attn_out.isPagedKVCache || core_attn_out.multiDeviceData ||
        &core_attn_out == &convOutput || &core_attn_out == &ba ||
        &core_attn_out == &normWeight || &core_attn_out == &aLog ||
        &core_attn_out == &dtBias || &core_attn_out == &last_recurrent_state ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !last_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        normWeight.cudaData == nullptr ||
        aLog.cudaData == nullptr ||
        dtBias.cudaData == nullptr ||
        last_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int stateDevice = -1;
    if (!FastllmCudaResolveDataDeviceId(last_recurrent_state, stateDevice) ||
        FastllmCudaGetDevice() != stateDevice) {
        return false;
    }
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        headVDim > 8 * 65535 || numVHeads % numKHeads != 0 ||
        !std::isfinite(eps) || eps < 0.0f || !std::isfinite(qScale)) {
        return false;
    }
    if (convOutput.dims.size() != 3 || ba.dims.size() != 3) {
        return false;
    }
    int seqLen = convOutput.dims[1];
    if (seqLen <= 1 || seqLen > FASTLLM_CUDA_MTP_FAST_SEQ_MAX) {
        return false;
    }
    long long qkvDim64 = 2LL * numKHeads * headKDim + 1LL * numVHeads * headVDim;
    long long baDim64 = 2LL * numVHeads;
    long long stateElements = 1LL * numVHeads * headKDim * headVDim;
    long long outputElements = 1LL * seqLen * numVHeads * headVDim;
    if (qkvDim64 <= 0 || qkvDim64 > 0x7fffffffLL || (qkvDim64 & 1) != 0 ||
        baDim64 <= 0 || baDim64 > 0x7fffffffLL ||
        stateElements <= 0 || stateElements > 0x7fffffffLL ||
        1LL * seqLen * qkvDim64 > 0x7fffffffLL ||
        1LL * seqLen * baDim64 > 0x7fffffffLL ||
        outputElements > 0x7fffffffLL ||
        ((uintptr_t)convOutput.cudaData & (alignof(half2) - 1)) != 0) {
        return false;
    }
    int qkvDim = (int)qkvDim64;
    if (convOutput.dims[0] != 1 ||
        !FastllmCudaDataHasDenseStrides(convOutput) ||
        convOutput.dims.back() != qkvDim ||
        ba.dims[0] != 1 ||
        ba.dims.back() != (int)baDim64 || !FastllmCudaDataHasDenseStrides(ba) ||
        convOutput.dims[1] != ba.dims[1] ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        !FastllmCudaDataHasDenseStrides(normWeight) ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        !FastllmCudaDataHasDenseStrides(aLog) ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        !FastllmCudaDataHasDenseStrides(dtBias) ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] != 1 ||
        last_recurrent_state.dims[1] != numVHeads ||
        last_recurrent_state.dims[2] != headKDim ||
        last_recurrent_state.dims[3] != headVDim ||
        !FastllmCudaDataHasDenseStrides(last_recurrent_state)) {
        return false;
    }

    if (numTokenStates < 0 ||
        numTokenStates > FASTLLM_CUDA_MTP_PREFIX_SNAPSHOT_MAX ||
        numTokenStates > seqLen ||
        (numTokenStates > 0 && tokenStates == nullptr)) {
        return false;
    }
    for (int t = 0; t < numTokenStates; t++) {
        fastllm::Data *snap = tokenStates[t];
        if (snap == nullptr) {
            continue;
        }
        if (snap->isFake || snap->cudaDataBorrowed || snap->isPagedKVCache ||
            snap->multiDeviceData || snap == &convOutput || snap == &ba ||
            snap == &normWeight || snap == &aLog || snap == &dtBias ||
            snap == &last_recurrent_state || snap == &core_attn_out) {
            return false;
        }
        for (int other = 0; other < t; other++) {
            if (tokenStates[other] == snap) {
                return false;
            }
        }
    }
    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.Resize({1, seqLen, numVHeads, headVDim});
    core_attn_out.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{stateDevice});
    core_attn_out.Allocate(false);
    core_attn_out.isKVCache = false;
    core_attn_out.isLinearAttention = false;
    core_attn_out.isLinearAttentionTransposed = false;
    if (core_attn_out.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(core_attn_out) ||
        !FastllmCudaDataCanShareDevice(last_recurrent_state, core_attn_out)) {
        return false;
    }

    half *snaps[FASTLLM_CUDA_MTP_PREFIX_SNAPSHOT_MAX] = {};
    for (int t = 0; t < numTokenStates; t++) {
        fastllm::Data *snap = tokenStates[t];
        if (snap == nullptr) {
            continue;
        }
        snap->dataType = last_recurrent_state.dataType;
        snap->Resize(last_recurrent_state.dims);
        snap->ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{stateDevice});
        snap->Allocate();
        snap->isLinearAttentionTransposed = true;
        if (snap->cudaData == nullptr || !FastllmCudaDataHasDenseStrides(*snap) ||
            !FastllmCudaDataCanShareDevice(last_recurrent_state, *snap)) {
            return false;
        }
        snaps[t] = (half*)snap->cudaData;
    }

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(numVHeads, (headVDim + tileV - 1) / tileV);

    cudaError_t pendingState = cudaGetLastError();
    if (pendingState != cudaSuccess) {
        checkCudaErrors("Error: stale CUDA error before FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16.", pendingState);
        return false;
    }
    FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)last_recurrent_state.cudaData, nullptr,
        (half*)core_attn_out.cudaData,
        seqLen, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        snaps[0], snaps[1], snaps[2], snaps[3], snaps[4],
        nullptr, numTokenStates
    );

    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16.", launchState);
        return false;
    }
    return true;
}

bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16BatchSnapshots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    const std::vector<fastllm::Data*> &lastRecurrentStates,
    fastllm::Data &coreAttnOut,
    const std::vector<fastllm::Data*> &tokenStates, int numTokenStates,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (lastRecurrentStates.empty() || lastRecurrentStates[0] == nullptr ||
        numTokenStates < 0 ||
        numTokenStates > FASTLLM_CUDA_MTP_PREFIX_SNAPSHOT_MAX ||
        tokenStates.size() !=
            lastRecurrentStates.size() * (size_t)numTokenStates ||
        coreAttnOut.isFake || coreAttnOut.cudaDataBorrowed ||
        coreAttnOut.isPagedKVCache || coreAttnOut.multiDeviceData) {
        return false;
    }
    std::set<const fastllm::Data*> tensors = {
        &convOutput, &ba, &normWeight, &aLog, &dtBias, &coreAttnOut
    };
    if (tensors.size() != 6) {
        return false;
    }
    fastllm::Data &first = *lastRecurrentStates[0];
    int device = -1;
    int batch = (int)lastRecurrentStates.size();
    if (!FastllmCudaResolveDataDeviceId(first, device) ||
        FastllmCudaGetDevice() != device ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        first.dataType != fastllm::DataType::FLOAT16 ||
        !first.isLinearAttentionTransposed ||
        convOutput.dims.size() != 3 || convOutput.dims[0] != batch ||
        ba.dims.size() != 3 || ba.dims[0] != batch ||
        convOutput.dims[1] <= 1 ||
        convOutput.dims[1] > FASTLLM_CUDA_MTP_FAST_SEQ_MAX ||
        ba.dims[1] != convOutput.dims[1] ||
        numTokenStates > convOutput.dims[1] ||
        !FastllmCudaDataHasDenseStrides(convOutput) ||
        !FastllmCudaDataHasDenseStrides(ba) ||
        !FastllmCudaDataCanShareDevice(first, convOutput) ||
        !FastllmCudaDataCanShareDevice(first, ba) ||
        !FastllmCudaDataCanShareDevice(first, normWeight) ||
        !FastllmCudaDataCanShareDevice(first, aLog) ||
        !FastllmCudaDataCanShareDevice(first, dtBias) ||
        numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 ||
        headVDim <= 0 || numVHeads % numKHeads != 0 ||
        !std::isfinite(eps) || eps < 0.0f || !std::isfinite(qScale)) {
        return false;
    }
    int seqLen = convOutput.dims[1];
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (convOutput.dims[2] != qkvDim || ba.dims[2] != numVHeads * 2 ||
        normWeight.dims != std::vector<int>({headKDim}) ||
        aLog.dims != std::vector<int>({numVHeads}) ||
        dtBias.dims != std::vector<int>({numVHeads}) ||
        first.dims != std::vector<int>({1, numVHeads, headKDim, headVDim})) {
        return false;
    }

    std::vector<void*> pointers;
    pointers.reserve(batch * (1 + numTokenStates));
    for (fastllm::Data *state : lastRecurrentStates) {
        if (state == nullptr || state->dims != first.dims ||
            state->dataType != first.dataType ||
            !state->isLinearAttentionTransposed ||
            !FastllmCudaDataHasDenseStrides(*state) ||
            !FastllmCudaDataCanShareDevice(first, *state) ||
            !tensors.insert(state).second) {
            return false;
        }
        pointers.push_back(state->cudaData);
    }
    for (int b = 0; b < batch; b++) {
        for (int token = 0; token < numTokenStates; token++) {
            fastllm::Data *snapshot =
                tokenStates[b * numTokenStates + token];
            if (snapshot == nullptr || snapshot->isFake ||
                snapshot->cudaDataBorrowed || snapshot->isPagedKVCache ||
                snapshot->multiDeviceData ||
                !tensors.insert(snapshot).second) {
                return false;
            }
            snapshot->dataType = first.dataType;
            snapshot->Resize(first.dims);
            snapshot->ToDevice(fastllm::DataDevice::CUDA,
                               std::vector<int>{device});
            snapshot->Allocate();
            snapshot->isLinearAttentionTransposed = true;
            if (snapshot->cudaData == nullptr ||
                !FastllmCudaDataHasDenseStrides(*snapshot) ||
                !FastllmCudaDataCanShareDevice(first, *snapshot)) {
                return false;
            }
            pointers.push_back(snapshot->cudaData);
        }
    }
    void **devicePointers = FastllmCudaStagePointers(pointers);

    coreAttnOut.dataType = first.dataType;
    coreAttnOut.Resize({batch, seqLen, numVHeads, headVDim});
    coreAttnOut.ToDevice(fastllm::DataDevice::CUDA,
                         std::vector<int>{device});
    coreAttnOut.Allocate(false);
    coreAttnOut.isKVCache = false;
    coreAttnOut.isLinearAttention = false;
    coreAttnOut.isLinearAttentionTransposed = false;
    if (coreAttnOut.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(coreAttnOut) ||
        !FastllmCudaDataCanShareDevice(first, coreAttnOut)) {
        return false;
    }
    constexpr int tileV = 8;
    int threads = tileV * 32;
    size_t sharedBytes = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 grid(numVHeads, (headVDim + tileV - 1) / tileV, batch);
    FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel<tileV>
        <<<grid, threads, sharedBytes>>>(
            (const half*)convOutput.cudaData,
            (const half*)ba.cudaData,
            (const float*)normWeight.cudaData,
            (const float*)aLog.cudaData,
            (const float*)dtBias.cudaData,
            nullptr, (half**)devicePointers,
            (half*)coreAttnOut.cudaData,
            seqLen, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            (half**)(devicePointers + batch), numTokenStates);
    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in batched sequence MTP recurrent kernel.",
            launchState);
        return false;
    }
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    void *cudaStatePool, void *cudaSlotIds, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePool == nullptr || cudaSlotIds == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads) {
        return false;
    }

    core_attn_out.dataType = fastllm::DataType::FLOAT16;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = convOutput.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    int stateStride = numVHeads * headVDim * headKDim;

    LaunchFastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarp(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)cudaStatePool, (const int*)cudaSlotIds, stateStride, 8,
        batch == 1 && numKHeads == 8 && numVHeads == 16 &&
            headKDim == 128 && headVDim == 128
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots.", cudaGetLastError());
    return true;
}

void FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int batch = (int)last_recurrent_states.size();
    std::vector<void*> pointers(batch);
    for (int i = 0; i < batch; i++) {
        pointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = FastllmCudaStagePointers(pointers);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
}

void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposed(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int batch = (int)last_recurrent_states.size();
    std::vector<void*> pointers(batch);
    for (int i = 0; i < batch; i++) {
        pointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = FastllmCudaStagePointers(pointers);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
}

template <typename T>
__global__ void FastllmChunkGatedDeltaRuleBuildQScaledChunkKernel(
    const T *q, const T *g, T *qScaled,
    int chunks, int ci, int chunk_size, int kdim, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        size_t inner = (size_t)chunk_size * kdim;
        size_t bh = idx / inner;
        size_t rem = idx % inner;
        size_t t = rem / kdim;
        size_t kd = rem % kdim;
        size_t gIdx = ((bh * chunks + (size_t)ci) * chunk_size + t);
        size_t qIdx = gIdx * kdim + kd;
        qScaled[idx] = FastllmCudaFloatToValue<T>(
            FastllmCudaValueToFloat(q[qIdx]) * expf(FastllmCudaValueToFloat(g[gIdx])));
    }
}

template <typename T>
__global__ void FastllmChunkGatedDeltaRuleBuildChunkScaleKernel(
    const T *g, float *gScale, float *gLastExp,
    int bhCount, int chunks, int chunk_size, int ci) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bhCount * chunk_size;
    if (idx < total) {
        int bh = idx / chunk_size;
        int t = idx % chunk_size;
        int base = (bh * chunks + ci) * chunk_size;
        float last = FastllmCudaValueToFloat(g[base + chunk_size - 1]);
        gScale[idx] = expf(last - FastllmCudaValueToFloat(g[base + t]));
        if (t == 0) {
            gLastExp[bh] = expf(last);
        }
    }
}

template <typename T>
__global__ void FastllmChunkGatedDeltaRuleScaleStateKernel(
    T *state, const float *gLastExp,
    int stateSize, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int bh = idx / stateSize;
        state[idx] = FastllmCudaFloatToValue<T>(
            FastllmCudaValueToFloat(state[idx]) * gLastExp[bh]);
    }
}

template <typename T>
__global__ void FastllmChunkGatedDeltaRuleBuildKScaledTransKernel(
    const T *k, const float *gScale, T *kScaledTrans,
    int chunks, int ci, int chunk_size, int kdim, size_t total) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        size_t inner = (size_t)kdim * chunk_size;
        int bh = idx / inner;
        int rem = idx % inner;
        int kd = rem / chunk_size;
        int t = rem % chunk_size;
        size_t src = ((((size_t)bh * chunks + ci) * chunk_size + t) * kdim + kd);
        kScaledTrans[idx] = FastllmCudaFloatToValue<T>(
            FastllmCudaValueToFloat(k[src]) * gScale[(size_t)bh * chunk_size + t]);
    }
}

static void FastllmChunkGatedDeltaRuleBatchedMatMul(
    const void *input0, const void *input1, void *output,
    fastllm::DataType dataType,
    int batch, int n, int m, int k,
    long long stride0, long long stride1, long long strideOut,
    float alpha, float beta) {
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    if (dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           k, n, m, &alpha,
                                           (const float*)input1, k, stride1,
                                           (const float*)input0, m, stride0,
                                           &beta,
                                           (float*)output, k, strideOut, batch);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        half hAlpha = __float2half(alpha);
        half hBeta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           k, n, m, &hAlpha,
                                           (const half*)input1, k, stride1,
                                           (const half*)input0, m, stride0,
                                           &hBeta,
                                           (half*)output, k, strideOut, batch);
    } else {
        printf("Error: unsupported data type in FastllmChunkGatedDeltaRuleBatchedMatMul.\n");
        throw("unsupported data type");
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("batch = %d, n = %d, m = %d, k = %d\n", batch, n, m, k);
        printf("Error: cublas error in ChunkGatedDeltaRule batched matmul.\n");
        throw("cublas error");
    }
}

template <typename T, int BV, int BK>
__global__ void FastllmChunkGatedDeltaRulePrefillHKernel(
    const T *k, const T *v, const T *g, const T *k_cumdecay,
    T *h_states, T *v_new_store, T *last_recurrent_state,
    int batch, int heads, int chunks, int chunk_size, int kdim, int vdim) {
    int vTile = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / heads, h = bh % heads;
    int vStart = vTile * BV;
    if (b >= batch || h >= heads || vStart >= vdim) {
        return;
    }

    int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float *stateTile = shared_mem;                         // [kdim, BV]
    float *vNewTile = stateTile + kdim * BV;              // [chunk_size, BV]
    float *tempTile = vNewTile + chunk_size * BV;         // [chunk_size, BK]
    float *gScale = tempTile + chunk_size * BK;           // [chunk_size]

    const size_t chunkStrideK = (size_t)chunk_size * kdim;
    const size_t chunkStrideV = (size_t)chunk_size * vdim;
    const size_t chunkStrideG = (size_t)chunk_size;
    const size_t headBaseK = ((size_t)b * heads + h) * chunks * chunkStrideK;
    const size_t headBaseV = ((size_t)b * heads + h) * chunks * chunkStrideV;
    const size_t headBaseG = ((size_t)b * heads + h) * chunks * chunkStrideG;
    const size_t stateBase = ((size_t)b * heads + h) * kdim * vdim;

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        stateTile[idx] = (vcol < vdim) ? FastllmCudaValueToFloat(last_recurrent_state[stateBase + (size_t)kd * vdim + vcol]) : 0.0f;
    }
    __syncthreads();

    for (int ci = 0; ci < chunks; ci++) {
        const T *kChunk = k + (size_t)ci * chunkStrideK + headBaseK;
        const T *vChunk = v + (size_t)ci * chunkStrideV + headBaseV;
        const T *gChunk = g + (size_t)ci * chunkStrideG + headBaseG;
        const T *kCumChunk = k_cumdecay + (size_t)ci * chunkStrideK + headBaseK;
        T *hChunk = h_states + ((((size_t)b * heads + h) * chunks + ci) * kdim * vdim + vStart);
        T *vNewChunk = v_new_store + ((((size_t)b * heads + h) * chunks + ci) * chunkStrideV + vStart);

        for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
            int kd = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            if (vcol < vdim) {
                hChunk[(size_t)kd * vdim + vo] = FastllmCudaFloatToValue<T>(stateTile[idx]);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
            int t = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            vNewTile[idx] = (vcol < vdim) ? FastllmCudaValueToFloat(vChunk[t * vdim + vcol]) : 0.0f;
        }
        __syncthreads();

        for (int ks = 0; ks < kdim; ks += BK) {
            int curBK = min(BK, kdim - ks);
            for (int idx = tid; idx < chunk_size * curBK; idx += blockDim.x) {
                int t = idx / curBK;
                int kk = idx % curBK;
                tempTile[t * BK + kk] = FastllmCudaValueToFloat(kCumChunk[t * kdim + ks + kk]);
            }
            __syncthreads();

            for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
                int t = idx / BV;
                int vo = idx % BV;
                float sum = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < BK; kk++) {
                    if (kk < curBK) {
                        sum += tempTile[t * BK + kk] * stateTile[(ks + kk) * BV + vo];
                    }
                }
                vNewTile[idx] -= sum;
            }
            __syncthreads();
        }

        float gLast = FastllmCudaValueToFloat(gChunk[chunk_size - 1]);
        float gLastExp = expf(gLast);
        for (int idx = tid; idx < chunk_size; idx += blockDim.x) {
            gScale[idx] = expf(gLast - FastllmCudaValueToFloat(gChunk[idx]));
        }
        __syncthreads();

        for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
            int t = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            if (vcol < vdim) {
                vNewChunk[t * vdim + vo] = FastllmCudaFloatToValue<T>(vNewTile[idx]);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
            stateTile[idx] *= gLastExp;
        }
        __syncthreads();

        for (int ks = 0; ks < kdim; ks += BK) {
            int curBK = min(BK, kdim - ks);
            for (int idx = tid; idx < chunk_size * curBK; idx += blockDim.x) {
                int t = idx / curBK;
                int kk = idx % curBK;
                tempTile[t * BK + kk] = FastllmCudaValueToFloat(kChunk[t * kdim + ks + kk]) * gScale[t];
            }
            __syncthreads();

            for (int idx = tid; idx < curBK * BV; idx += blockDim.x) {
                int kk = idx / BV;
                int vo = idx % BV;
                float update = 0.0f;
                #pragma unroll
                for (int t = 0; t < chunk_size; t++) {
                    update += tempTile[t * BK + kk] * vNewTile[t * BV + vo];
                }
                stateTile[(ks + kk) * BV + vo] += update;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        if (vcol < vdim) {
            last_recurrent_state[stateBase + (size_t)kd * vdim + vcol] = FastllmCudaFloatToValue<T>(stateTile[idx]);
        }
    }
}

template <typename T, int BV>
__global__ void FastllmChunkGatedDeltaRulePrefillOKernel(
    const T *q, const T *g, const T *attn,
    const T *h_states, const T *v_new_store, T *core_attn_out,
    int batch, int heads, int chunks, int chunk_size, int kdim, int vdim) {
    int vTile = blockIdx.x;
    int ci = blockIdx.y;
    int bh = blockIdx.z;
    int b = bh / heads, h = bh % heads;
    int vStart = vTile * BV;
    if (b >= batch || h >= heads || ci >= chunks || vStart >= vdim) {
        return;
    }

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warpCount = (blockDim.x + 31) >> 5;
    const unsigned int warpMask = 0xffffffffu;
    extern __shared__ float shared_mem[];
    float *hTile = shared_mem;                            // [kdim, BV]
    float *vNewTile = hTile + kdim * BV;                 // [chunk_size, BV]

    const size_t chunkStrideQ = (size_t)chunk_size * kdim;
    const size_t chunkStrideV = (size_t)chunk_size * vdim;
    const size_t chunkStrideG = (size_t)chunk_size;
    const size_t chunkStrideAttn = (size_t)chunk_size * chunk_size;
    const size_t headBaseQ = ((size_t)b * heads + h) * chunks * chunkStrideQ;
    const size_t headBaseG = ((size_t)b * heads + h) * chunks * chunkStrideG;
    const size_t headBaseAttn = ((size_t)b * heads + h) * chunks * chunkStrideAttn;
    const T *qChunk = q + (size_t)ci * chunkStrideQ + headBaseQ;
    const T *gChunk = g + (size_t)ci * chunkStrideG + headBaseG;
    const T *attnChunk = attn + (size_t)ci * chunkStrideAttn + headBaseAttn;
    const T *hChunk = h_states + ((((size_t)b * heads + h) * chunks + ci) * kdim * vdim + vStart);
    const T *vNewChunk = v_new_store + ((((size_t)b * heads + h) * chunks + ci) * chunkStrideV + vStart);
    T *outChunk = core_attn_out + ((((size_t)b * heads + h) * chunks + ci) * chunk_size * vdim + vStart);

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        hTile[idx] = (vcol < vdim) ? FastllmCudaValueToFloat(hChunk[(size_t)kd * vdim + vo]) : 0.0f;
    }

    for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
        int t = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        vNewTile[idx] = (vcol < vdim) ? FastllmCudaValueToFloat(vNewChunk[t * vdim + vo]) : 0.0f;
    }
    __syncthreads();

    for (int t = warp; t < chunk_size; t += warpCount) {
        int vcol = vStart + lane;
        if (vcol < vdim) {
            float gExp = 0.0f;
            if (lane == 0) {
                gExp = expf(FastllmCudaValueToFloat(gChunk[t]));
            }
            gExp = __shfl_sync(warpMask, gExp, 0);

            float sum = 0.0f;
            for (int kd = 0; kd < kdim; kd++) {
                float qValue = 0.0f;
                if (lane == 0) {
                    qValue = FastllmCudaValueToFloat(qChunk[t * kdim + kd]);
                }
                qValue = __shfl_sync(warpMask, qValue, 0);
                sum += qValue * hTile[(size_t)kd * BV + lane];
            }
            sum *= gExp;

            const T *attnRow = attnChunk + (size_t)t * chunk_size;
            for (int j = 0; j < chunk_size; j++) {
                float attnValue = 0.0f;
                if (lane == 0) {
                    attnValue = FastllmCudaValueToFloat(attnRow[j]);
                }
                attnValue = __shfl_sync(warpMask, attnValue, 0);
                sum += attnValue * vNewTile[j * BV + lane];
            }
            outChunk[t * vdim + lane] = FastllmCudaFloatToValue<T>(sum);
        }
    }
}

template <typename T, int BV, int BK>
__global__ void FastllmChunkGatedDeltaRuleVarlenPrefillHKernel(
    const T *k, const T *v, const T *g, const T *k_cumdecay,
    const T *last_recurrent_state, T *next_recurrent_state,
    const int *chunk_offsets, T *h_states, T *v_new_store,
    int key_heads, int heads, int total_chunks,
    int chunk_size, int kdim, int vdim) {
    int vTile = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / heads, h = bh % heads;
    int keyHead = h * key_heads / heads;
    int vStart = vTile * BV;
    if (h >= heads || vStart >= vdim) {
        return;
    }

    int tid = threadIdx.x;
    extern __shared__ float shared_mem[];
    float *stateTile = shared_mem;                         // [kdim, BV]
    float *vNewTile = stateTile + kdim * BV;              // [chunk_size, BV]
    float *tempTile = vNewTile + chunk_size * BV;         // [chunk_size, BK]
    float *gScale = tempTile + chunk_size * BK;           // [chunk_size]

    const size_t chunkStrideK = (size_t)chunk_size * kdim;
    const size_t chunkStrideV = (size_t)chunk_size * vdim;
    const size_t chunkStrideG = (size_t)chunk_size;
    const size_t headBaseK =
        (size_t)keyHead * total_chunks * chunkStrideK;
    const size_t headBaseKCum =
        (size_t)h * total_chunks * chunkStrideK;
    const size_t headBaseV = (size_t)h * total_chunks * chunkStrideV;
    const size_t headBaseG = (size_t)h * total_chunks * chunkStrideG;
    const size_t stateBase = (size_t)bh * kdim * vdim;

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        stateTile[idx] = vcol < vdim
            ? FastllmCudaValueToFloat(
                  last_recurrent_state[stateBase + (size_t)kd * vdim + vcol])
            : 0.0f;
    }
    __syncthreads();

    int chunkBegin = chunk_offsets[b];
    int chunkEnd = chunk_offsets[b + 1];
    for (int ci = chunkBegin; ci < chunkEnd; ci++) {
        const T *kChunk = k + (size_t)ci * chunkStrideK + headBaseK;
        const T *vChunk = v + (size_t)ci * chunkStrideV + headBaseV;
        const T *gChunk = g + (size_t)ci * chunkStrideG + headBaseG;
        const T *kCumChunk =
            k_cumdecay + (size_t)ci * chunkStrideK + headBaseKCum;
        T *hChunk = h_states +
            ((size_t)h * total_chunks + ci) * kdim * vdim + vStart;
        T *vNewChunk = v_new_store +
            ((size_t)h * total_chunks + ci) * chunkStrideV + vStart;

        for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
            int kd = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            if (vcol < vdim) {
                hChunk[(size_t)kd * vdim + vo] =
                    FastllmCudaFloatToValue<T>(stateTile[idx]);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
            int t = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            vNewTile[idx] = vcol < vdim
                ? FastllmCudaValueToFloat(vChunk[t * vdim + vcol])
                : 0.0f;
        }
        __syncthreads();

        for (int ks = 0; ks < kdim; ks += BK) {
            int curBK = min(BK, kdim - ks);
            for (int idx = tid; idx < chunk_size * curBK;
                 idx += blockDim.x) {
                int t = idx / curBK;
                int kk = idx % curBK;
                tempTile[t * BK + kk] = FastllmCudaValueToFloat(
                    kCumChunk[t * kdim + ks + kk]);
            }
            __syncthreads();
            for (int idx = tid; idx < chunk_size * BV;
                 idx += blockDim.x) {
                int t = idx / BV;
                int vo = idx % BV;
                float sum = 0.0f;
#pragma unroll
                for (int kk = 0; kk < BK; kk++) {
                    if (kk < curBK) {
                        sum += tempTile[t * BK + kk] *
                               stateTile[(ks + kk) * BV + vo];
                    }
                }
                vNewTile[idx] -= sum;
            }
            __syncthreads();
        }

        float gLast =
            FastllmCudaValueToFloat(gChunk[chunk_size - 1]);
        float gLastExp = expf(gLast);
        for (int idx = tid; idx < chunk_size; idx += blockDim.x) {
            gScale[idx] =
                expf(gLast - FastllmCudaValueToFloat(gChunk[idx]));
        }
        __syncthreads();

        for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
            int t = idx / BV;
            int vo = idx % BV;
            int vcol = vStart + vo;
            if (vcol < vdim) {
                vNewChunk[t * vdim + vo] =
                    FastllmCudaFloatToValue<T>(vNewTile[idx]);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
            stateTile[idx] *= gLastExp;
        }
        __syncthreads();

        for (int ks = 0; ks < kdim; ks += BK) {
            int curBK = min(BK, kdim - ks);
            for (int idx = tid; idx < chunk_size * curBK;
                 idx += blockDim.x) {
                int t = idx / curBK;
                int kk = idx % curBK;
                tempTile[t * BK + kk] =
                    FastllmCudaValueToFloat(
                        kChunk[t * kdim + ks + kk]) * gScale[t];
            }
            __syncthreads();
            for (int idx = tid; idx < curBK * BV; idx += blockDim.x) {
                int kk = idx / BV;
                int vo = idx % BV;
                float update = 0.0f;
#pragma unroll
                for (int t = 0; t < chunk_size; t++) {
                    update += tempTile[t * BK + kk] *
                              vNewTile[t * BV + vo];
                }
                stateTile[(ks + kk) * BV + vo] += update;
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        if (vcol < vdim) {
            next_recurrent_state[stateBase + (size_t)kd * vdim + vcol] =
                FastllmCudaFloatToValue<T>(stateTile[idx]);
        }
    }
}

template <typename T, int BV>
__global__ void FastllmChunkGatedDeltaRuleVarlenPrefillOKernel(
    const T *q, const T *g, const T *attn, const T *decay_mask,
    const T *h_states, const T *v_new_store, T *core_attn_out,
    const int *chunk_token_bases, const int *chunk_valid_tokens,
    int total_tokens, int key_heads, int heads, int total_chunks,
    int chunk_size, int kdim, int vdim, bool apply_decay_mask) {
    int vTile = blockIdx.x;
    int ci = blockIdx.y;
    int h = blockIdx.z;
    int keyHead = h * key_heads / heads;
    int vStart = vTile * BV;
    if (h >= heads || ci >= total_chunks || vStart >= vdim) {
        return;
    }

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warpCount = (blockDim.x + 31) >> 5;
    const unsigned int warpMask = 0xffffffffu;
    extern __shared__ float shared_mem[];
    float *hTile = shared_mem;                            // [kdim, BV]
    float *vNewTile = hTile + kdim * BV;                 // [chunk_size, BV]

    const size_t chunkStrideQ = (size_t)chunk_size * kdim;
    const size_t chunkStrideV = (size_t)chunk_size * vdim;
    const size_t chunkStrideG = (size_t)chunk_size;
    const size_t chunkStrideAttn = (size_t)chunk_size * chunk_size;
    const size_t headBaseQ =
        (size_t)keyHead * total_chunks * chunkStrideQ;
    const size_t headBaseG = (size_t)h * total_chunks * chunkStrideG;
    const size_t headBaseAttn =
        (size_t)keyHead * total_chunks * chunkStrideAttn;
    const size_t headBaseDecay =
        (size_t)h * total_chunks * chunkStrideAttn;
    const T *qChunk = q + (size_t)ci * chunkStrideQ + headBaseQ;
    const T *gChunk = g + (size_t)ci * chunkStrideG + headBaseG;
    const T *attnChunk =
        attn + (size_t)ci * chunkStrideAttn + headBaseAttn;
    const T *decayChunk = apply_decay_mask
        ? decay_mask + (size_t)ci * chunkStrideAttn + headBaseDecay
        : nullptr;
    const T *hChunk = h_states +
        ((size_t)h * total_chunks + ci) * kdim * vdim + vStart;
    const T *vNewChunk = v_new_store +
        ((size_t)h * total_chunks + ci) * chunkStrideV + vStart;
    int tokenBase = chunk_token_bases[ci];
    int validTokens = chunk_valid_tokens[ci];

    for (int idx = tid; idx < kdim * BV; idx += blockDim.x) {
        int kd = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        hTile[idx] = vcol < vdim
            ? FastllmCudaValueToFloat(hChunk[(size_t)kd * vdim + vo])
            : 0.0f;
    }
    for (int idx = tid; idx < chunk_size * BV; idx += blockDim.x) {
        int t = idx / BV;
        int vo = idx % BV;
        int vcol = vStart + vo;
        vNewTile[idx] = vcol < vdim
            ? FastllmCudaValueToFloat(vNewChunk[t * vdim + vo])
            : 0.0f;
    }
    __syncthreads();

    for (int t = warp; t < validTokens; t += warpCount) {
        int vcol = vStart + lane;
        if (vcol < vdim) {
            float gExp = 0.0f;
            if (lane == 0) {
                gExp = expf(FastllmCudaValueToFloat(gChunk[t]));
            }
            gExp = __shfl_sync(warpMask, gExp, 0);
            float sum = 0.0f;
            for (int kd = 0; kd < kdim; kd++) {
                float qValue = 0.0f;
                if (lane == 0) {
                    qValue =
                        FastllmCudaValueToFloat(qChunk[t * kdim + kd]);
                }
                qValue = __shfl_sync(warpMask, qValue, 0);
                sum += qValue * hTile[(size_t)kd * BV + lane];
            }
            sum *= gExp;
            const T *attnRow = attnChunk + (size_t)t * chunk_size;
            for (int j = 0; j < chunk_size; j++) {
                float attnValue = 0.0f;
                if (lane == 0) {
                    attnValue = FastllmCudaValueToFloat(attnRow[j]);
                    if (apply_decay_mask) {
                        attnValue = j <= t
                            ? FastllmCudaValueToFloat(
                                  FastllmCudaFloatToValue<T>(
                                      attnValue * FastllmCudaValueToFloat(
                                          decayChunk[(size_t)t *
                                              chunk_size + j])))
                            : 0.0f;
                    }
                }
                attnValue = __shfl_sync(warpMask, attnValue, 0);
                sum += attnValue * vNewTile[j * BV + lane];
            }
            int outputToken = tokenBase + t;
            if (outputToken < total_tokens) {
                core_attn_out[
                    ((size_t)outputToken * heads + h) * vdim +
                    vStart + lane] = FastllmCudaFloatToValue<T>(sum);
            }
        }
    }
}

namespace {
struct FastllmChunkGdnVarlenScratch {
    void *h = nullptr;
    void *vNew = nullptr;
    void *nextState = nullptr;
    size_t hBytes = 0;
    size_t vNewBytes = 0;
    size_t stateBytes = 0;
};

static bool FastllmEnsureChunkGdnVarlenScratch(
    size_t hBytes, size_t vNewBytes, size_t stateBytes,
    FastllmChunkGdnVarlenScratch *&scratch) {
    thread_local static std::map<int, FastllmChunkGdnVarlenScratch> scratches;
    int device = FastllmCudaGetDevice();
    FastllmChunkGdnVarlenScratch &cached = scratches[device];
    if (cached.hBytes < hBytes || cached.vNewBytes < vNewBytes ||
        cached.stateBytes < stateBytes) {
        if (cached.h != nullptr || cached.vNew != nullptr ||
            cached.nextState != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
        }
        if (cached.h != nullptr) FastllmCudaFree(cached.h);
        if (cached.vNew != nullptr) FastllmCudaFree(cached.vNew);
        if (cached.nextState != nullptr) FastllmCudaFree(cached.nextState);
        cached = FastllmChunkGdnVarlenScratch();
        cached.h = FastllmCudaMalloc(hBytes);
        cached.vNew = FastllmCudaMalloc(vNewBytes);
        cached.nextState = FastllmCudaMalloc(stateBytes);
        if (cached.h == nullptr || cached.vNew == nullptr ||
            cached.nextState == nullptr) {
            if (cached.h != nullptr) FastllmCudaFree(cached.h);
            if (cached.vNew != nullptr) FastllmCudaFree(cached.vNew);
            if (cached.nextState != nullptr) FastllmCudaFree(cached.nextState);
            cached = FastllmChunkGdnVarlenScratch();
            return false;
        }
        cached.hBytes = hBytes;
        cached.vNewBytes = vNewBytes;
        cached.stateBytes = stateBytes;
    }
    scratch = &cached;
    return true;
}
}

bool FastllmChunkGatedDeltaRuleVarlenPrefillNative(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &k_cumdecay, fastllm::Data &last_recurrent_state,
    const std::vector<int> &seqLens, fastllm::Data &core_attn_out,
    const fastllm::Data *decay_mask, bool apply_decay_mask) {
    if (seqLens.empty() || q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        attn.dataType != fastllm::DataType::FLOAT16 ||
        k_cumdecay.dataType != fastllm::DataType::FLOAT16 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        (apply_decay_mask &&
         (decay_mask == nullptr ||
          decay_mask->dataType != fastllm::DataType::FLOAT16 ||
          decay_mask->cudaData == nullptr)) ||
        q.cudaData == nullptr || k.cudaData == nullptr ||
        v.cudaData == nullptr || g.cudaData == nullptr ||
        attn.cudaData == nullptr || k_cumdecay.cudaData == nullptr ||
        last_recurrent_state.cudaData == nullptr ||
        q.dims.size() != 5 || q.dims[0] != 1 || k.dims != q.dims) {
        return false;
    }
    int batch = (int)seqLens.size();
    int keyHeads = q.dims[1];
    int heads = v.dims.size() == 5 ? v.dims[1] : 0;
    int totalChunks = q.dims[2];
    int chunkSize = q.dims[3];
    int kdim = q.dims[4];
    int vdim = v.dims.size() == 5 ? v.dims[4] : 0;
    if (keyHeads <= 0 || heads < keyHeads || heads % keyHeads != 0 ||
        totalChunks <= 0 || chunkSize != 64 ||
        kdim != 128 || vdim != 128 ||
        v.dims != std::vector<int>({1, heads, totalChunks, chunkSize, vdim}) ||
        g.dims != std::vector<int>({1, heads, totalChunks, chunkSize}) ||
        attn.dims !=
            std::vector<int>({1, keyHeads, totalChunks,
                              chunkSize, chunkSize}) ||
        (apply_decay_mask && decay_mask->dims !=
            std::vector<int>({1, heads, totalChunks,
                              chunkSize, chunkSize})) ||
        k_cumdecay.dims !=
            std::vector<int>({1, heads, totalChunks, chunkSize, kdim}) ||
        last_recurrent_state.dims !=
            std::vector<int>({batch, heads, kdim, vdim})) {
        return false;
    }
    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(
            seqLens, chunkSize, metadata) ||
        metadata.totalChunks != totalChunks) {
        return false;
    }

    int device = -1;
    if (!FastllmCudaResolveDataDeviceId(q, device) ||
        FastllmCudaGetDevice() != device ||
        !FastllmCudaDataCanShareDevice(q, k) ||
        !FastllmCudaDataCanShareDevice(q, v) ||
        !FastllmCudaDataCanShareDevice(q, g) ||
        !FastllmCudaDataCanShareDevice(q, attn) ||
        !FastllmCudaDataCanShareDevice(q, k_cumdecay) ||
        !FastllmCudaDataCanShareDevice(q, last_recurrent_state) ||
        (apply_decay_mask &&
         !FastllmCudaDataCanShareDevice(q, *decay_mask))) {
        return false;
    }

    core_attn_out.dataType = fastllm::DataType::FLOAT16;
    core_attn_out.Resize({1, metadata.totalTokens, heads, vdim});
    core_attn_out.ToDevice(
        fastllm::DataDevice::CUDA, std::vector<int>{device});
    core_attn_out.Allocate(false);
    if (core_attn_out.cudaData == nullptr ||
        !FastllmCudaDataHasDenseStrides(core_attn_out)) {
        return false;
    }

    size_t hBytes =
        (size_t)heads * totalChunks * kdim * vdim * sizeof(half);
    size_t vNewBytes =
        (size_t)heads * totalChunks * chunkSize * vdim * sizeof(half);
    size_t stateBytes =
        (size_t)batch * heads * kdim * vdim * sizeof(half);
    FastllmChunkGdnVarlenScratch *scratch = nullptr;
    if (!FastllmEnsureChunkGdnVarlenScratch(
            hBytes, vNewBytes, stateBytes, scratch)) {
        return false;
    }
    constexpr int BVH = 32;
    constexpr int BKH = 64;
    constexpr int BVO = 32;
    int threadsH = 256;
    int threadsO = 256;
    size_t sharedH =
        (size_t)(kdim * BVH + chunkSize * BVH +
                 chunkSize * BKH + chunkSize) * sizeof(float);
    size_t sharedO =
        (size_t)(kdim * BVO + chunkSize * BVO) * sizeof(float);
    dim3 gridH((vdim + BVH - 1) / BVH, batch * heads);
    dim3 gridO((vdim + BVO - 1) / BVO, totalChunks, heads);
    FastllmChunkGatedDeltaRuleVarlenPrefillHKernel<half, BVH, BKH>
        <<<gridH, threadsH, sharedH, cudaStreamPerThread>>>(
            (const half*)k.cudaData, (const half*)v.cudaData,
            (const half*)g.cudaData, (const half*)k_cumdecay.cudaData,
            (const half*)last_recurrent_state.cudaData,
            (half*)scratch->nextState, metadata.chunkOffsets,
            (half*)scratch->h, (half*)scratch->vNew,
            keyHeads, heads, totalChunks, chunkSize, kdim, vdim);
    cudaError_t hState = cudaGetLastError();
    if (hState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in packed varlen GDN H fallback.", hState);
        return false;
    }
    FastllmChunkGatedDeltaRuleVarlenPrefillOKernel<half, BVO>
        <<<gridO, threadsO, sharedO, cudaStreamPerThread>>>(
            (const half*)q.cudaData, (const half*)g.cudaData,
            (const half*)attn.cudaData,
            apply_decay_mask ? (const half*)decay_mask->cudaData : nullptr,
            (const half*)scratch->h,
            (const half*)scratch->vNew, (half*)core_attn_out.cudaData,
            metadata.chunkTokenBases, metadata.chunkValidTokens,
            metadata.totalTokens, keyHeads, heads, totalChunks,
            chunkSize, kdim, vdim, apply_decay_mask);
    cudaError_t oState = cudaGetLastError();
    if (oState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error in packed varlen GDN O fallback.", oState);
        return false;
    }
    cudaError_t commitState = cudaMemcpyAsync(
        last_recurrent_state.cudaData, scratch->nextState, stateBytes,
        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    if (commitState != cudaSuccess) {
        checkCudaErrors(
            "Error: CUDA error committing packed varlen GDN state.",
            commitState);
        return false;
    }
    return true;
}

void FastllmChunkGatedDeltaRulePrefill(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn, fastllm::Data &k_cumdecay,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out) {
    int batch = q.dims[0];
    int heads = q.dims[1];
    int chunks = q.dims[2];
    int chunk_size = q.dims[3];
    int kdim = q.dims[4];
    int vdim = v.dims[4];

    core_attn_out.dataType = v.dataType;
    core_attn_out.dataDevice = v.dataDevice;
    core_attn_out.dataDeviceIds = v.dataDeviceIds;
    core_attn_out.Resize({batch, heads, chunks, chunk_size, vdim});
    core_attn_out.Allocate();

    size_t unitBytes = ((size_t)v.unitSize + v.unitSizeDiv - 1) / v.unitSizeDiv;
    long long bhCount = (long long)batch * heads;
    long long stateStride = (long long)kdim * vdim;
    long long qChunkStride = (long long)chunk_size * kdim;
    long long vChunkStride = (long long)chunk_size * vdim;
    long long attnChunkStride = (long long)chunk_size * chunk_size;
    bool useBatchedGemm = q.dataType == fastllm::DataType::FLOAT32 ||
                          q.dataType == fastllm::DataType::FLOAT16;

    size_t hElems = (size_t)batch * heads * chunks * kdim * vdim;
    size_t vNewElems = useBatchedGemm ? (size_t)bhCount * vChunkStride
                                      : (size_t)batch * heads * chunks * chunk_size * vdim;
    void *hData = useBatchedGemm ? nullptr : FastllmCudaMalloc(hElems * unitBytes);
    void *vNewData = FastllmCudaMalloc(vNewElems * unitBytes);

    void *qData = FastllmCudaPrepareInput(q);
    void *kData = FastllmCudaPrepareInput(k);
    void *vData = FastllmCudaPrepareInput(v);
    void *gData = FastllmCudaPrepareInput(g);
    void *attnData = FastllmCudaPrepareInput(attn);
    void *kCumData = FastllmCudaPrepareInput(k_cumdecay);
    void *stateData = FastllmCudaPrepareInput(last_recurrent_state);
    void *outData = FastllmCudaPrepareOutput(core_attn_out);
    void *qScaledData = nullptr;
    void *kScaledTransData = nullptr;
    float *gScaleData = nullptr;
    float *gLastExpData = nullptr;

    if (useBatchedGemm) {
        size_t qScaledElems = (size_t)bhCount * qChunkStride;
        size_t stateElems = (size_t)bhCount * stateStride;
        size_t kScaledTransElems = (size_t)bhCount * kdim * chunk_size;
        qScaledData = FastllmCudaMalloc(qScaledElems * unitBytes);
        kScaledTransData = FastllmCudaMalloc(kScaledTransElems * unitBytes);
        gScaleData = (float*)FastllmCudaMalloc((size_t)bhCount * chunk_size * sizeof(float));
        gLastExpData = (float*)FastllmCudaMalloc((size_t)bhCount * sizeof(float));

        const int threads = 256;
        int qScaleBlocks = (int)((qScaledElems + threads - 1) / threads);
        int chunkScaleBlocks = (int)(((size_t)bhCount * chunk_size + threads - 1) / threads);
        int stateScaleBlocks = (int)((stateElems + threads - 1) / threads);
        int kScaleBlocks = (int)((kScaledTransElems + threads - 1) / threads);
        long long outStride = (long long)chunks * vChunkStride;
        for (int ci = 0; ci < chunks; ci++) {
            uint8_t *kCumSlice = (uint8_t*)kCumData + (size_t)ci * qChunkStride * unitBytes;
            uint8_t *vSlice = (uint8_t*)vData + (size_t)ci * vChunkStride * unitBytes;
            uint8_t *attnSlice = (uint8_t*)attnData + (size_t)ci * attnChunkStride * unitBytes;
            uint8_t *outSlice = (uint8_t*)outData + (size_t)ci * vChunkStride * unitBytes;

            if (q.dataType == fastllm::DataType::FLOAT32) {
                FastllmChunkGatedDeltaRuleBuildQScaledChunkKernel<float><<<qScaleBlocks, threads>>>(
                    (float*)qData, (float*)gData, (float*)qScaledData,
                    chunks, ci, chunk_size, kdim, qScaledElems);
            } else {
                FastllmChunkGatedDeltaRuleBuildQScaledChunkKernel<half><<<qScaleBlocks, threads>>>(
                    (half*)qData, (half*)gData, (half*)qScaledData,
                    chunks, ci, chunk_size, kdim, qScaledElems);
            }

            cudaError_t cudaState = cudaMemcpy2DAsync(vNewData, (size_t)vChunkStride * unitBytes,
                                                      vSlice, (size_t)chunks * vChunkStride * unitBytes,
                                                      (size_t)vChunkStride * unitBytes, bhCount,
                                                      cudaMemcpyDeviceToDevice, 0);
            checkCudaErrors("Error: CUDA error when gathering chunk v data!", cudaState);

            FastllmChunkGatedDeltaRuleBatchedMatMul(
                kCumSlice, stateData, vNewData, q.dataType,
                (int)bhCount, chunk_size, kdim, vdim,
                (long long)chunks * qChunkStride, stateStride, vChunkStride,
                -1.0f, 1.0f);

            FastllmChunkGatedDeltaRuleBatchedMatMul(
                qScaledData, stateData, outSlice, q.dataType,
                (int)bhCount, chunk_size, kdim, vdim,
                qChunkStride, stateStride, outStride,
                1.0f, 0.0f);

            FastllmChunkGatedDeltaRuleBatchedMatMul(
                attnSlice, vNewData, outSlice, q.dataType,
                (int)bhCount, chunk_size, chunk_size, vdim,
                (long long)chunks * attnChunkStride, vChunkStride, outStride,
                1.0f, 1.0f);

            if (q.dataType == fastllm::DataType::FLOAT32) {
                FastllmChunkGatedDeltaRuleBuildChunkScaleKernel<float><<<chunkScaleBlocks, threads>>>(
                    (float*)gData, gScaleData, gLastExpData, (int)bhCount, chunks, chunk_size, ci);
                FastllmChunkGatedDeltaRuleScaleStateKernel<float><<<stateScaleBlocks, threads>>>(
                    (float*)stateData, gLastExpData, (int)stateStride, stateElems);
                FastllmChunkGatedDeltaRuleBuildKScaledTransKernel<float><<<kScaleBlocks, threads>>>(
                    (float*)kData, gScaleData, (float*)kScaledTransData,
                    chunks, ci, chunk_size, kdim, kScaledTransElems);
            } else {
                FastllmChunkGatedDeltaRuleBuildChunkScaleKernel<half><<<chunkScaleBlocks, threads>>>(
                    (half*)gData, gScaleData, gLastExpData, (int)bhCount, chunks, chunk_size, ci);
                FastllmChunkGatedDeltaRuleScaleStateKernel<half><<<stateScaleBlocks, threads>>>(
                    (half*)stateData, gLastExpData, (int)stateStride, stateElems);
                FastllmChunkGatedDeltaRuleBuildKScaledTransKernel<half><<<kScaleBlocks, threads>>>(
                    (half*)kData, gScaleData, (half*)kScaledTransData,
                    chunks, ci, chunk_size, kdim, kScaledTransElems);
            }

            FastllmChunkGatedDeltaRuleBatchedMatMul(
                kScaledTransData, vNewData, stateData, q.dataType,
                (int)bhCount, kdim, chunk_size, vdim,
                (long long)kdim * chunk_size, vChunkStride, stateStride,
                1.0f, 1.0f);
        }
    } else {
        const int BVHSmall = 16;
        const int BVHLarge = 32;
        const int BVO = 32;
        const int BKHSmall = 32;
        const int BKHLarge = 64;
        const size_t maxDynamicSharedMem = 48 * 1024;
        bool useSmallBVH = vdim <= 64;
        int bvh = useSmallBVH ? BVHSmall : BVHLarge;
        size_t sharedMemSizeHSmallBK = (size_t)(kdim * bvh + chunk_size * bvh + chunk_size * BKHSmall + chunk_size) * sizeof(float);
        size_t sharedMemSizeHLargeBK = (size_t)(kdim * bvh + chunk_size * bvh + chunk_size * BKHLarge + chunk_size) * sizeof(float);
        bool useLargeBKH = chunk_size == 64 && sharedMemSizeHLargeBK <= maxDynamicSharedMem;
        dim3 gridH(useSmallBVH ? (vdim + BVHSmall - 1) / BVHSmall : (vdim + BVHLarge - 1) / BVHLarge, batch * heads);
        dim3 gridO((vdim + BVO - 1) / BVO, chunks, batch * heads);
        int threadsPerBlockH = 256;
        int threadsPerBlockO = chunk_size >= 64 ? 256 : 128;
        size_t sharedMemSizeH = useLargeBKH ? sharedMemSizeHLargeBK : sharedMemSizeHSmallBK;
        size_t sharedMemSizeO = (size_t)(kdim * BVO + chunk_size * BVO) * sizeof(float);

        if (useSmallBVH) {
            if (useLargeBKH) {
                FastllmChunkGatedDeltaRulePrefillHKernel<__nv_bfloat16, BVHSmall, BKHLarge><<<gridH, threadsPerBlockH, sharedMemSizeH>>>(
                    (__nv_bfloat16*)kData, (__nv_bfloat16*)vData, (__nv_bfloat16*)gData, (__nv_bfloat16*)kCumData,
                    (__nv_bfloat16*)hData, (__nv_bfloat16*)vNewData, (__nv_bfloat16*)stateData,
                    batch, heads, chunks, chunk_size, kdim, vdim);
            } else {
                FastllmChunkGatedDeltaRulePrefillHKernel<__nv_bfloat16, BVHSmall, BKHSmall><<<gridH, threadsPerBlockH, sharedMemSizeH>>>(
                    (__nv_bfloat16*)kData, (__nv_bfloat16*)vData, (__nv_bfloat16*)gData, (__nv_bfloat16*)kCumData,
                    (__nv_bfloat16*)hData, (__nv_bfloat16*)vNewData, (__nv_bfloat16*)stateData,
                    batch, heads, chunks, chunk_size, kdim, vdim);
            }
        } else {
            if (useLargeBKH) {
                FastllmChunkGatedDeltaRulePrefillHKernel<__nv_bfloat16, BVHLarge, BKHLarge><<<gridH, threadsPerBlockH, sharedMemSizeH>>>(
                    (__nv_bfloat16*)kData, (__nv_bfloat16*)vData, (__nv_bfloat16*)gData, (__nv_bfloat16*)kCumData,
                    (__nv_bfloat16*)hData, (__nv_bfloat16*)vNewData, (__nv_bfloat16*)stateData,
                    batch, heads, chunks, chunk_size, kdim, vdim);
            } else {
                FastllmChunkGatedDeltaRulePrefillHKernel<__nv_bfloat16, BVHLarge, BKHSmall><<<gridH, threadsPerBlockH, sharedMemSizeH>>>(
                    (__nv_bfloat16*)kData, (__nv_bfloat16*)vData, (__nv_bfloat16*)gData, (__nv_bfloat16*)kCumData,
                    (__nv_bfloat16*)hData, (__nv_bfloat16*)vNewData, (__nv_bfloat16*)stateData,
                    batch, heads, chunks, chunk_size, kdim, vdim);
            }
        }
        FastllmChunkGatedDeltaRulePrefillOKernel<__nv_bfloat16, BVO><<<gridO, threadsPerBlockO, sharedMemSizeO>>>(
            (__nv_bfloat16*)qData, (__nv_bfloat16*)gData, (__nv_bfloat16*)attnData,
            (__nv_bfloat16*)hData, (__nv_bfloat16*)vNewData, (__nv_bfloat16*)outData,
            batch, heads, chunks, chunk_size, kdim, vdim);
    }

    FastllmCudaFinishInput(q, qData);
    FastllmCudaFinishInput(k, kData);
    FastllmCudaFinishInput(v, vData);
    FastllmCudaFinishInput(g, gData);
    FastllmCudaFinishInput(attn, attnData);
    FastllmCudaFinishInput(k_cumdecay, kCumData);
    FastllmCudaFinishInput(last_recurrent_state, stateData);
    FastllmCudaFinishOutput(core_attn_out, outData);
    if (qScaledData != nullptr) FastllmCudaFree(qScaledData);
    if (kScaledTransData != nullptr) FastllmCudaFree(kScaledTransData);
    if (gScaleData != nullptr) FastllmCudaFree(gScaleData);
    if (gLastExpData != nullptr) FastllmCudaFree(gLastExpData);
    if (hData != nullptr) FastllmCudaFree(hData);
    if (vNewData != nullptr) FastllmCudaFree(vNewData);
}

void FastllmPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *cudaIndex) {
    for (int i = 0; i < rows; i++) {
        int index = cudaIndex[i];
        for (int j = 0; j < cols; j++) {
            partInput[i * cols + j] = input[index * cols + j];
        }
    }
}

// CUDA Kernel 函数
// 每个线程负责搬运一个 uint8_t 元素
__global__ void FastllmPickInputKernel(uint8_t *input, uint8_t *partInput, int rows, int cols, int *index) {
    // blockIdx.y 对应行索引 i
    int row = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查：防止越界访问
    if (row < rows && col < cols) {
        // 读取该行在源数据中对应的真实行号
        int srcRow = index[row];
        
        // 计算扁平化的内存偏移量
        // 使用 long long 防止在大模型显存较大时 int32 溢出
        long long dstOffset = (long long)row * cols + col;
        long long srcOffset = (long long)srcRow * cols + col;
        // 执行拷贝
        partInput[dstOffset] = input[srcOffset];
    }
}

__global__ void FastllmPickInputKernelVec16(const uint4 *__restrict__ input,
                                            uint4 *__restrict__ partInput,
                                            int rows, int cols16, int *index) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols16) {
        int srcRow = index[row];
        partInput[(size_t)row * cols16 + col] = input[(size_t)srcRow * cols16 + col];
    }
}

// Host 调用函数
void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *index) {
    // 设定 Block 大小：256 是通过是一个比较通用的高性能值
    dim3 block(256);
    if ((cols & 15) == 0 &&
        (((uintptr_t)input | (uintptr_t)partInput) & 15) == 0) {
        int cols16 = cols / 16;
        dim3 grid((cols16 + 255) / 256, rows);
        FastllmPickInputKernelVec16<<<grid, block>>>(
            (const uint4*)input, (uint4*)partInput, rows, cols16, index);
        return;
    }
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整除以 256
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    FastllmPickInputKernel<<<grid, block>>>(input, partInput, rows, cols, index);
}

// CUDA Kernel 函数
// 每个线程负责一个 float 元素的计算和累加
__global__ void FastllmPickOutputKernel(float *partOutput, float *output, int rows, int cols, int *index, float *scales) {
    // blockIdx.y 对应行索引 i (partOutput 中的行)
    int i = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查
    if (i < rows && j < cols) {
        // 获取目标行号 idx 和 缩放因子 sca
        int idx = index[i];
        float sca = scales[i];
        // 计算扁平化的内存偏移量
        // 使用 long long 防止大模型显存地址溢出
        long long srcOffset = (long long)i * cols + j;
        long long dstOffset = (long long)idx * cols + j;
        // 执行 CPU 逻辑: output[idx * cols + j] += sca * partOutput[i * cols + j];
        // 注意：这里假设 index 映射的目标行通常是唯一的（在 LLM Batch 推理中通常如此）。
        // 如果多个 i 映射到同一个 idx，这里存在竞争冒险，但在 FastLLM 上下文中通常是 Scatter 操作。
        output[dstOffset] += sca * partOutput[srcOffset];
    }
}
// Host 调用函数
void FastllmCudaPickOutput(float *partOutput, float *output, int rows, int cols, int *index, float *scales) {
    // 设定 Block 大小：使用 256 作为通用高性能值
    dim3 block(256);
    
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    FastllmPickOutputKernel<<<grid, block>>>(partOutput, output, rows, cols, index, scales);
}

// CUDA Kernel 函数
// 每个线程负责一个 float 元素的计算和累加
__global__ void FastllmPickOutputKernel(half *partOutput, half *output, int rows, int cols, int *index, float *scales) {
    // blockIdx.y 对应行索引 i (partOutput 中的行)
    int i = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查
    if (i < rows && j < cols) {
        // 获取目标行号 idx 和 缩放因子 sca
        int idx = index[i];
        float sca = scales[i];
        // 计算扁平化的内存偏移量
        // 使用 long long 防止大模型显存地址溢出
        long long srcOffset = (long long)i * cols + j;
        long long dstOffset = (long long)idx * cols + j;
        // 执行 CPU 逻辑: output[idx * cols + j] += sca * partOutput[i * cols + j];
        // 注意：这里假设 index 映射的目标行通常是唯一的（在 LLM Batch 推理中通常如此）。
        // 如果多个 i 映射到同一个 idx，这里存在竞争冒险，但在 FastLLM 上下文中通常是 Scatter 操作。
        output[dstOffset] = (half)((float)output[dstOffset] + sca * (float)partOutput[srcOffset]);
    }
}

__global__ void FastllmPickOutputHalf2Kernel(const half2 *__restrict__ partOutput,
                                             half2 *__restrict__ output,
                                             int rows, int cols2, int *index, float *scales) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols2) {
        int idx = index[i];
        float scale = scales[i];
        half2 src = partOutput[(size_t)i * cols2 + j];
        half2 dst = output[(size_t)idx * cols2 + j];
        float2 src2 = __half22float2(src);
        float2 dst2 = __half22float2(dst);
        output[(size_t)idx * cols2 + j] = __floats2half2_rn(dst2.x + scale * src2.x,
                                                            dst2.y + scale * src2.y);
    }
}

// CUDA Kernel 函数
// 每个线程负责一个 bfloat16 元素的计算和累加
__global__ void FastllmPickOutputKernel(__nv_bfloat16 *partOutput, __nv_bfloat16 *output, int rows, int cols, int *index, float *scales) {
    // blockIdx.y 对应行索引 i (partOutput 中的行)
    int i = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查
    if (i < rows && j < cols) {
        // 获取目标行号 idx 和 缩放因子 sca
        int idx = index[i];
        float sca = scales[i];
        // 计算扁平化的内存偏移量
        // 使用 long long 防止大模型显存地址溢出
        long long srcOffset = (long long)i * cols + j;
        long long dstOffset = (long long)idx * cols + j;
        // 执行逻辑: output[idx * cols + j] += sca * partOutput[i * cols + j];
        // 注意：这里假设 index 映射的目标行通常是唯一的（在 LLM Batch 推理中通常如此）。
        // 如果多个 i 映射到同一个 idx，这里存在竞争冒险，但在 FastLLM 上下文中通常是 Scatter 操作。
        output[dstOffset] = (__nv_bfloat16)((float)output[dstOffset] + sca * (float)partOutput[srcOffset]);
    }
}

// Host 调用函数
void FastllmCudaPickOutput(uint8_t *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, fastllm::DataType dataType) {
    // 设定 Block 大小：使用 256 作为通用高性能值
    dim3 block(256);
    
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    if (dataType == fastllm::DataType::FLOAT32) {
        FastllmPickOutputKernel<<<grid, block>>>((float*)partOutput, (float*)output, rows, cols, index, scales);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        if ((cols & 1) == 0 &&
            (((uintptr_t)partOutput | (uintptr_t)output) & 3) == 0) {
            int cols2 = cols / 2;
            dim3 grid2((cols2 + 255) / 256, rows);
            FastllmPickOutputHalf2Kernel<<<grid2, block>>>(
                (const half2*)partOutput, (half2*)output, rows, cols2, index, scales);
            return;
        }
        FastllmPickOutputKernel<<<grid, block>>>((half*)partOutput, (half*)output, rows, cols, index, scales);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FastllmPickOutputKernel<<<grid, block>>>((__nv_bfloat16*)partOutput, (__nv_bfloat16*)output, rows, cols, index, scales);
    } else {
        printf("FastllmCudaPickOutput Error: datatype error.\n");
        exit(0);
    }
}
