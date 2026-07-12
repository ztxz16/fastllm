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
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <map>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include <cuda_fp8.h>
#include "sampling.cuh"

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#endif

// 线程级 CUDA 错误标志：showError 报错时置位。CUDA graph 捕获路径在捕获体前清零、
// 捕获体后检查，任何算子报错都能被感知并中止捕获，避免带着坏状态继续运行。
static thread_local bool fastllmCudaThreadErrorFlag = false;

void FastllmCudaClearThreadError() {
    fastllmCudaThreadErrorFlag = false;
}

void FastllmCudaSetThreadError() {
    fastllmCudaThreadErrorFlag = true;
}

bool FastllmCudaGetThreadError() {
    return fastllmCudaThreadErrorFlag;
}

void showError(cudaError_t result, char const* const message, const char* const file,
           int const line) {
    if (cudaSuccess != result) {
        fastllmCudaThreadErrorFlag = true;
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n",
            message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
        fflush(stdout);
    }  
}

static std::atomic<bool> fastllmCudaMallocDisabled(false);
static std::mutex fastllmCudaMallocCheckMutex;

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
    if (fastllm::GetFastllmEnv().cudaMemCheck) {
        bool rejected = fastllmCudaMallocDisabled.load(std::memory_order_relaxed);
        FastllmCudaPrintMallocStack(size, file, line, rejected);
        if (rejected) {
            if (ret != nullptr) {
                *ret = nullptr;
            }
            return cudaErrorMemoryAllocation;
        }
    }
    if (fastllmCudaNcclActive.load(std::memory_order_relaxed)) {
        // 真实 cudaMalloc 前排空在途 NCCL 集合通信，避免与 cudaMalloc 争用 CUDA 驱动锁导致跨 rank 死锁。
        cudaDeviceSynchronize();
    }
    return cudaMalloc(ret, size);
}

void DisableCudaMalloc() {
    fastllmCudaMallocDisabled.store(true, std::memory_order_relaxed);
    if (fastllm::GetFastllmEnv().cudaMemCheck) {
        fprintf(stderr, "[FASTLLM_CUDA_MEM_CHECK] cudaMalloc disabled.\n");
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

void DeviceSync() {
    if (fastllm::GetFastllmEnv().cudaSync) {
        cudaDeviceSynchronize();
    }
}

void ForceDeviceSync() {
    cudaError_t state = cudaDeviceSynchronize();
    checkCudaErrors("Error: CUDA error when synchronizing device!", state);
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

void FastllmCudaEventDestroy(void *event) {
    cudaError_t state = cudaEventDestroy((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when destroying event!", state);
}

void FastllmCudaEventRecord(void *event, void *stream) {
    cudaError_t state = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when recording event!", state);
}

void FastllmCudaEventSynchronize(void *event) {
    cudaError_t state = cudaEventSynchronize((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when synchronizing event!", state);
}

void FastllmCudaStreamWaitEvent(void *stream, void *event) {
    cudaError_t state = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
    checkCudaErrors("Error: CUDA error when stream waiting event!", state);
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
    cudaError_t state = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    return FastllmCudaGraphSetError("cudaStreamBeginCapture", state);
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

__global__ void FastllmMambaSoftplusKernel(float* inputData, float *outputData, float *aLog, float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = -expf((double)aLog[i]) * softplus(inputData[o * channels + i] + dtBias[i]);
    }
}

__global__ void FastllmMambaSoftplusKernel(half* inputData, half *outputData, float *aLog, float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = __float2half(-exp((double)aLog[i]) * softplus(__half2float(inputData[o * channels + i]) + dtBias[i]));
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
        if (tid < s) {
            if (maxData[tid] < maxData[tid + s]) {
                maxData[tid] = maxData[tid + s];
                idData[tid] = idData[tid + s];
            }
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

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer () {}

    CudaMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};
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

static size_t fastllmCudaMemPoolAllocated = 0;
static size_t fastllmCudaMemPoolPeak = 0;

struct FastllmCudaWeightSlab {
    void *base = nullptr;
    size_t size = 0;
    size_t used = 0;
    int activeBlocks = 0;
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

void *FastllmCudaMallocModelWeight(size_t size) {
    size_t slabBytes = FastllmCudaGetWeightSlabBytes();
    if (slabBytes == 0 || size == 0 || size > slabBytes / 2) {
        return FastllmCudaMalloc(size);
    }

    int id = -1;
    cudaError_t state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);

    const size_t align = 256;
    size_t aligned = FastllmCudaAlignBytes(size, align);
    std::lock_guard<std::mutex> lock(fastllmCudaWeightSlabMutex);

    auto &slabs = fastllmCudaWeightSlabs[id];
    int slabId = -1;
    for (int i = 0; i < slabs.size(); i++) {
        if (slabs[i].size >= slabs[i].used && slabs[i].size - slabs[i].used >= aligned) {
            slabId = i;
            break;
        }
    }

    if (slabId < 0) {
        void *base = nullptr;
        state = FastllmCudaCheckedMalloc(&base, slabBytes, __FILE__, __LINE__);
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
    void * ret;
    cudaError_t state = FastllmCudaCheckedMalloc(&ret, size, __FILE__, __LINE__);
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
    if (!fastllm::GetFastllmEnv().cudaMemCheck) {
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

static void FastllmCudaReleaseIdleCachedBuffersForDevice(int id) {
    cudaError_t state = cudaSetDevice(id);
    checkCudaErrors("Error: CUDA error when switching device to release cached memory!", state);

    auto bigIt = bigBuffersMap.find(id);
    if (bigIt != bigBuffersMap.end()) {
        auto &bigBuffers = bigIt->second;
        std::vector<CudaMemoryBuffer> busyBuffers;
        busyBuffers.reserve(bigBuffers.size());
        for (auto &buffer : bigBuffers) {
            if (buffer.busy) {
                busyBuffers.push_back(buffer);
            } else {
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
            if (buffer.busy) {
                busyBuffers.push_back(buffer);
            } else {
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
    cudaGetLastError();
    FastllmCudaReleaseIdleCachedBuffersForDevice(id);
    cudaError_t state = FastllmCudaCheckedMalloc(ret, size, file, line);
    return state == cudaSuccess;
}

void * FastllmCudaMalloc(size_t size) {
    int id = -1;
    cudaError_t state = cudaSuccess;
    state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    if (size > 1024 * 1024) {
        auto &bigBuffers = *view.bigBuffers;
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && FastllmCudaCanReusePooledBigBuffer(bigBuffers[i].size, size)) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
            return bigBuffers[selId].data;
        }
        if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
            for (int i = 0; i < bigBuffers.size(); i++) {
                if (!bigBuffers[i].busy && bigBuffers[i].size >= size) {
                    if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                        selId = i;
                    }
                }
            }
            if (selId != -1) {
                bigBuffers[selId].busy = true;
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
                return bigBuffers[selId].data;
            }
        }

        void * ret;
        if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
            FastllmCudaPrintPoolRejectStateLocked(id, size, view.bigBuffers, view.smallBuffers);
        }
        state = FastllmCudaCheckedMalloc(&ret, size, __FILE__, __LINE__);
        if (cudaSuccess != state && FastllmCudaRetryMallocAfterReleasingIdle(size, &ret, id, __FILE__, __LINE__)) {
            state = cudaSuccess;
        }
        if (cudaSuccess != state) {
            size_t freeMem = 0, totalMem = 0;
            cudaMemGetInfo(&freeMem, &totalMem);
            printf("Error: CUDA error when allocating %lu MB memory on device %d! "
                   "gpuFree: %lu MB / %lu MB.\n",
                   size >> 20, id, freeMem >> 20, totalMem >> 20);
            fflush(stdout);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
#ifdef CUDA_MEM_DEBUG
        CudaMemDebugRecord(ret, size);
#endif
        return ret;
    }
    auto &cudaBuffers = *view.smallBuffers;
    for (int i = *view.minId; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            *view.noBusy -= cudaBuffers[i].size;
            while (*view.minId < cudaBuffers.size() && cudaBuffers[*view.minId].busy) {
                (*view.minId)++;
            }
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(cudaBuffers[i].data, size);
#endif
            return cudaBuffers[i].data;
        }
    }
    if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
        auto &bigBuffers = *view.bigBuffers;
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && bigBuffers[i].size >= size) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
            return bigBuffers[selId].data;
        }
    }
    void * ret;
    if (fastllmCudaMallocDisabled.load(std::memory_order_relaxed)) {
        FastllmCudaPrintPoolRejectStateLocked(id, size, view.bigBuffers, view.smallBuffers);
    }
    state = FastllmCudaCheckedMalloc(&ret, size, __FILE__, __LINE__);
    if (cudaSuccess != state && FastllmCudaRetryMallocAfterReleasingIdle(size, &ret, id, __FILE__, __LINE__)) {
        state = cudaSuccess;
    }
    if (cudaSuccess != state) {
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        printf("Error: CUDA error when allocating %lu KB memory on device %d! "
               "gpuFree: %lu MB / %lu MB.\n",
               size >> 10, id, freeMem >> 20, totalMem >> 20);
        fflush(stdout);
        checkCudaErrors("", state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRecord(ret, size);
#endif
    return ret;
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
                state = cudaSetDevice(view.device);
                state = cudaFree(cudaBuffers[i].data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when force releasing memory on device %d!", view.device);
                }
                cudaBuffers.erase(cudaBuffers.begin() + i);
                *view.noBusy = 0;
                *view.minId = (int)cudaBuffers.size();
                for (int j = 0; j < (int)cudaBuffers.size(); j++) {
                    if (!cudaBuffers[j].busy) {
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
                state = cudaSetDevice(view.device);
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

void FastllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (FastllmCudaTryFreeWeightSlabPtr(ret)) {
        return;
    }
    int oriId = FastllmCudaGetDevice();

    // 优先只查当前 rank 的本地池。归还池内指针只修改 busy 标记，
    // 不调用 CUDA API。若本地未命中，再在不持锁时查询指针真实设备，
    // 仅检查 owner 的池；避免其它 rank 为释放本地临时张量而先抢 device 0 的锁。
    std::vector<FastllmCudaMemPoolView> views = FastllmSnapshotCudaMemPoolViews();
    auto releaseFromPool = [&](FastllmCudaMemPoolView &view) {
        std::lock_guard<std::mutex> lock(*view.lock);
        auto &cudaBuffers = *view.smallBuffers;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                *view.noBusy += cudaBuffers[i].size;
                cudaBuffers[i].busy = false;
                *view.minId = std::min(*view.minId, i);
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                return true;
            }
        }
        auto &bigBuffers = *view.bigBuffers;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
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

void FastllmCudaMallocBigBuffer(size_t size) {
    void * ret;
    int id = -1;
    cudaGetDevice(&id);
    FastllmCudaMemPoolView view = FastllmGetCudaMemPoolView(id);
    std::lock_guard<std::mutex> lock(*view.lock);
    auto &bigBuffers = *view.bigBuffers;
    auto state = FastllmCudaCheckedMalloc(&ret, size, __FILE__, __LINE__);
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
        auto &bigBuffers = *view.bigBuffers;
        std::vector <CudaMemoryBuffer> temp;
        long long littleMemSum = 0;        
        long long littleMemSumLimit = 300 * 1024 * 1024; // 留一小部分复用  
        std::vector <std::pair <std::size_t, int > > v;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
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
            if (!bigBuffers[i].busy && littleMemIds.find(i) == littleMemIds.end()) {
                state = cudaSetDevice(view.device);
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
    checkCudaErrors("Error: CUDA error when copy on GPU!", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    if (size == 0 || dst == src) {
        return;
    }
    int oriId = FastllmCudaGetDevice();
    int canPeerAccess = 0;
    cudaError_t state = cudaDeviceCanAccessPeer(&canPeerAccess, dstId, srcId);
    if (state != cudaSuccess) {
        cudaGetLastError();
        canPeerAccess = 0;
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
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData) {
    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float *aLog = (float*) FastllmCudaPrepareInput(aLogData);
    float *dtBias = (float*) FastllmCudaPrepareInput(dtBiasData);

    int threadPerBlock = std::min(64, channels);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> (cudaInput, cudaOutput, aLog, dtBias, channels);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> ((half*)cudaInput, (half*)cudaOutput, aLog, dtBias, channels);
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
        } else {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        }
    } else if (input0.dims == input1.dims) {
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
        } else {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        }
    } else {
        int channelLen = input0.Count(0) / input1.Count(0);
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len, channelLen);
        } else {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len, channelLen);
        }
    }
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
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
        if (channels < 512) {
            FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer,
                                                             channels, eps);
        } else if (channels < 4096) {
            FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer,
                                                               channels, eps);
        } else {
            FastllmRMSNormKernelInner1<1024> <<< outer, 1024 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer,
                                                                 channels, eps);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        if (channels < 512) {
            FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer,
                                                              channels, eps);
        } else if (channels < 4096) {
            FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer,
                                                                channels, eps);
        } else {
            FastllmRMSNormKernelInner1<1024> <<< outer, 1024 >>>((__nv_bfloat16*)cudaInput, (float*) weight.cudaData, (__nv_bfloat16*)cudaOutput, outer,
                                                                  channels, eps);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
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

bool FastllmCudaRMSNormSiluMulFloat16(const fastllm::Data &input, fastllm::Data &weight,
                                      const fastllm::Data &gateInput, fastllm::Data &output, float eps) {
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

    if (channels < 512) {
        FastllmRMSNormSiluMulHalfKernel<64><<<outer, 64>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    } else if (channels < 4096) {
        FastllmRMSNormSiluMulHalfKernel<512><<<outer, 512>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    } else {
        FastllmRMSNormSiluMulHalfKernel<1024><<<outer, 1024>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaRMSNormSiluMulFloat16.", cudaGetLastError());
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
                sum += inputData[expertIdx];
            }
        }
        
        // Write index and score
        for (int i = 0; i < topk; i++) {
            int expertIdx = idData[0][i];
            outputIndex[i] = expertIdx;
            outputScore[i] = inputData[expertIdx] / sum * routeScale;
        }
    }
}

bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale) {
    if (topk > 50) {
        printf("SelectExpert: unsupport topk > 50, falling back to CPU implementation.\n");
        return false; // 返回 false 表示不支持，应该回退到 CPU
    }
    
    float *cudaLogits = (float *) FastllmCudaPrepareInput(logits);
    float *cudaBias = nullptr;
    int hasBias = 0;
    if (gateBias != nullptr && gateBias->dims.size() > 0) {
        cudaBias = (float *) FastllmCudaPrepareInput(*gateBias);
        hasBias = 1;
    }
    int32_t *cudaIndex = (int32_t *) FastllmCudaPrepareInput(index);
    float *cudaScore = (float *) FastllmCudaPrepareInput(score);
    
    int dimsLen = logits.dims.size();
    int n = logits.Count(0) / logits.dims[dimsLen - 1]; // number of tokens
    int numExperts = logits.dims[dimsLen - 1]; // number of experts
    
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
        index[tid] = 0;
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
        axis == std::vector <int> {2, 0, 1, 3} ||
        axis == std::vector <int> {1, 2, 0, 3} ||
        (axis == std::vector <int> {0, 2, 1, 3} && input.dims[0] == 1);
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
    } else if (axis == std::vector <int> {0, 2, 1, 3} && input.dims[0] == 1) {
        int n = input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
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
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::FLOAT16) {
        half *outputData = (half *) dstOutputData;
        half *weightData = (half *) weight.cudaData;
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::BFLOAT16) {
        std::vector <float> cpuInputData = std::vector <float> (inputLen, 0.0f);
        FastllmCudaCopyFromDeviceToHost(cpuInputData.data(), inputData, cpuInputData.size() * sizeof(float));
        float *outputData = (float *) dstOutputData;
        uint16_t *weightData = (uint16_t *) weight.cudaData;
        for (int i = 0; i < inputLen; i++) {
            int token = (int) (cpuInputData[i] + 1e-9);
            for (int j = 0; j < embSize; j++) {
                FastllmBF16ToFloat(outputData + i * embSize, weightData + token * embSize, embSize);
            }
        }
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
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::FLOAT16) {
        half *outputData = (half *) output.cudaData;
        half *weightData = (half *) weight.cudaData;
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::BFLOAT16) {
        __nv_bfloat16 *outputData = (__nv_bfloat16 *) output.cudaData;
        __nv_bfloat16 *weightData = (__nv_bfloat16 *) weight.cudaData;
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
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
    cublasStatus_t status;

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
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        half *tempInput0 = (half*)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half*)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m, &h_alpha,
                (half*)cudaInput1, input1Stride, input1Spatial,
                (half*)tempInput0, input0Stride, input0Spatial,
                &h_beta,
                (half*)tempOutput, k, k * n, batch);
        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
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

bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

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
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        half *tempInput0 = (half*)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half*)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        k, n, m, &h_alpha,
                                        (half*)cudaInput1, input1Stride, input1Spatial,
                                        (half*)tempInput0, input0Stride, input0Spatial,
                                        &h_beta,
                                        (half*)tempOutput, k, k * n, batch);
        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
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
    float llama3HighFreqFactor
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
            if (useLlama3) {
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
    float llama3HighFreqFactor
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
            llama3LowFreqFactor, llama3HighFreqFactor);
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
        const int *topKArr, const float *topPArr,
        unsigned char *accepted, int *recoveredIds,
        int vocabSize, float posteriorThreshold, float posteriorAlpha) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const float *row = probs + (long long)bid * vocabSize;
    int candidateId = candidateIds[bid];
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
        bool allowedByTopK = topKArr[bid] <= 0 || greaterCount < topKArr[bid];
        bool allowedByTopP = topPArr[bid] >= 1.0f || greaterMass < topPArr[bid];
        accepted[bid] = allowedByTopK && allowedByTopP &&
                        candidateProb > dynamicThreshold ? 1 : 0;
        recoveredIds[bid] = maxIdData[0];
    }
}

bool FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
                                  float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize,
                                  const int *typicalCandidateIds,
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

    // temperatures | top-k | top-p | sampled ids | candidates | recovered ids | accepted flags
    size_t paramBytes = batch * (sizeof(float) + sizeof(int) + sizeof(float) + sizeof(int)) +
                        actualTypicalCount * (sizeof(int) + sizeof(int) + sizeof(unsigned char));
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
    int   *cudaTypicalRecovered = cudaTypicalCandidates + actualTypicalCount;
    unsigned char *cudaTypicalAccepted =
        (unsigned char *)(cudaTypicalRecovered + actualTypicalCount);

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
    }

    FastllmTemperatureSoftmaxKernel<1024><<<batch, 1024>>>(logits, cudaProbs, cudaTemperatures, vocabSize);
    if (actualTypicalCount > 0) {
        FastllmTypicalAcceptanceKernel<1024><<<actualTypicalCount, 1024>>>(
            cudaProbs, cudaTypicalCandidates, cudaTopKArr, cudaTopPArr,
            cudaTypicalAccepted, cudaTypicalRecovered, vocabSize,
            typicalPosteriorThreshold, typicalPosteriorAlpha);
    }

    static std::mt19937 rng(std::random_device{}());
    uint64_t seed = rng();

    flashinfer::sampling::TopKTopPSamplingFromProb<float, int>(
        cudaProbs, cudaTopKArr, cudaTopPArr, cudaOutput,
        (int *)nullptr,
        (uint32_t)batch, (int)0, 0.0f,
        (uint32_t)vocabSize, false, seed, 0, 0);

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

bool FastllmCudaTopKTopPSampling(float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize) {
    return FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
        logits, temperatures, topKArr, topPArr, output, batch, vocabSize,
        nullptr, nullptr, nullptr, 0, 0.09f, 0.3f);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGreedySamplingKernel(float *logits, int *output, int vocabSize) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -1.0e30f;
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
        if (tid < s && maxData[tid] < maxData[tid + s]) {
            maxData[tid] = maxData[tid + s];
            idData[tid] = idData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = idData[0];
    }
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
    FastllmGreedySamplingKernel<256><<<batch, 256>>>(logits, output, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySampling: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGreedySamplingWithScoresKernel(float *logits, int *output,
                                                      float *scores, int vocabSize) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -1.0e30f;
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
            if (maxData[tid] < other ||
                (maxData[tid] == other && idData[tid] > otherId)) {
                maxData[tid] = other;
                idData[tid] = otherId;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = idData[0];
        scores[b] = maxData[0];
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
        logits, output, scores, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySamplingWithScores: kernel launch failed: %s\n",
               cudaGetErrorString(status));
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

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)inputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, cudaMemcpyHostToDevice);
    FastllmCatBatchKernel <256> <<< part * outer, 256 >>> (pointers, (uint8_t*)output.cudaData, outer, part, inner * unitSize);

    FastllmCudaFree(pointers);
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

    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = caches[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy conv cache pointers to GPU!", state);

    bool ret = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(
        cudaPointers, batch, *first, newToken, weight, bias, output);
    FastllmCudaFree(cudaPointers);
    return ret;
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
                logits[stride * b + eos_ids[base + i]] = 0;
            }
        }
        base += eos_nums[b];
    }
    return;
}
void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lens, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids) {
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
    return;
}

__global__ void FastllmCudaResetLogitsOfEOSAll(int total, int eos_num, int stride, float *logits, int *eos_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int b = idx / eos_num;
    int e = idx - b * eos_num;
    logits[stride * b + eos_ids[e]] = 0;
}

void FastllmResetLogitsOfEOSAll(int batch, fastllm::Data *logits, const std::vector<int> &eos_ids) {
    if (batch <= 0 || eos_ids.empty()) {
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
    checkCudaErrors("Error: CUDA error when reset logits of EOS all!", cudaGetLastError());
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

    int threads = 256;
    int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535);
    FastllmLinearAttentionStateTransposeHalfKernel<<<blocks, threads>>>(
        (const half*)oldData, (half*)newData, totalHeads, n2, n3, kvToVk
    );
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
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
        q, k, v, g, b, *last_recurrent_states[0], cudaPointers, batch, core_attn_out, qScale);
    FastllmCudaFree(cudaPointers);
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

template <int TILE_V>
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

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= headVDim) {
        return;
    }

    half *last_recurrent_state = statePool != nullptr ?
        (slotIds == nullptr ? statePool : statePool + (size_t)slotIds[batch_idx] * stateStride) :
        last_recurrent_states[batch_idx];
    int stateHeadBase = head_idx * headKDim * headVDim;
    half *state_row = last_recurrent_state + stateHeadBase + (size_t)v_col * headKDim;
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
    for (int j = lane_id; j < headKDim; j += 32) {
        float updated = __half2float(state_row[j]) * gVal + k_norm[j] * delta;
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

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel(
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *last_recurrent_state,
    half *core_attn_out,
    int seqLen, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *snap0, half *snap1, half *snap2, half *snap3, half *snap4,
    int numSnaps) {
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

    int stateHeadBase = head_idx * headKDim * headVDim;
    half *state_row = activeV ?
        last_recurrent_state + stateHeadBase + (size_t)v_col * headKDim :
        last_recurrent_state;

    for (int token = 0; token < seqLen; token++) {
        int convBase = token * qkvDim;
        int qOffset = convBase + qHead * headKDim;
        int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
        int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;
        int outBase = (token * numVHeads + head_idx) * headVDim;

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
            const half *baRow = ba + (size_t)token * (numVHeads * 2);
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
            if (token < numSnaps) {
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

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(batch, numVHeads, (headVDim + tileV - 1) / tileV);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
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

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
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

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(1, numVHeads, (headVDim + tileV - 1) / tileV);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        1, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)last_recurrent_state.cudaData, nullptr, 0
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16.", cudaGetLastError());
    return true;
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
        (half*)last_recurrent_state.cudaData,
        (half*)core_attn_out.cudaData,
        seqLen, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        snaps[0], snaps[1], snaps[2], snaps[3], snaps[4], numTokenStates
    );

    cudaError_t launchState = cudaGetLastError();
    if (launchState != cudaSuccess) {
        checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16.", launchState);
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

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(batch, numVHeads, (headVDim + tileV - 1) / tileV);
    int stateStride = numVHeads * headVDim * headKDim;

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)cudaStatePool, (const int*)cudaSlotIds, stateStride
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
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
    FastllmCudaFree(cudaPointers);
}

void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposed(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int batch = (int)last_recurrent_states.size();
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy transposed recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
    FastllmCudaFree(cudaPointers);
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
