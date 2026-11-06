#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {
struct LoadedTritonKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
};

static std::mutex g_tritonKernelMutex;
static std::map<std::string, LoadedTritonKernel> g_tritonKernels;
static std::mutex g_driverInitMutex;
static bool g_driverInitialized = false;

static bool CheckCu(CUresult result, const char *message) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    const char *name = nullptr;
    const char *text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    printf("Fastllm Triton CUDA error: %s (%s: %s)\n",
           message, name == nullptr ? "unknown" : name, text == nullptr ? "unknown" : text);
    return false;
}

static bool ReadBinaryFile(const char *path, std::vector<char> &bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        printf("Fastllm Triton CUDA error: failed to open cubin %s\n", path);
        return false;
    }
    file.seekg(0, std::ios::end);
    std::streamoff size = file.tellg();
    if (size <= 0) {
        printf("Fastllm Triton CUDA error: empty cubin %s\n", path);
        return false;
    }
    file.seekg(0, std::ios::beg);
    bytes.resize((size_t)size);
    file.read(bytes.data(), size);
    return file.good();
}

static bool EnsureCudaDriverInitialized() {
    std::lock_guard<std::mutex> guard(g_driverInitMutex);
    if (g_driverInitialized) {
        return true;
    }
    if (!CheckCu(cuInit(0), "cuInit")) {
        return false;
    }
    g_driverInitialized = true;
    return true;
}

static bool EnsureCudaDriverContext() {
    CUcontext context = nullptr;
    if (!CheckCu(cuCtxGetCurrent(&context), "cuCtxGetCurrent")) {
        return false;
    }
    if (context != nullptr) {
        return true;
    }

    int runtimeDevice = 0;
    cudaGetDevice(&runtimeDevice);
    CUdevice device = 0;
    if (!CheckCu(cuDeviceGet(&device, runtimeDevice), "cuDeviceGet")) {
        return false;
    }
    if (!CheckCu(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain")) {
        return false;
    }
    return CheckCu(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

static LoadedTritonKernel *LoadTritonKernel(const char *cubinPath, const char *kernelName, int shared) {
    int device = -1;
    cudaGetDevice(&device);

    if (!EnsureCudaDriverInitialized()) {
        return nullptr;
    }
    if (!EnsureCudaDriverContext()) {
        return nullptr;
    }
    CUcontext context = nullptr;
    if (!CheckCu(cuCtxGetCurrent(&context), "cuCtxGetCurrent")) {
        return nullptr;
    }

    std::string key = std::to_string(device) + ":" +
                      std::to_string((uintptr_t)context) + ":" +
                      cubinPath + ":" + kernelName;

    std::lock_guard<std::mutex> guard(g_tritonKernelMutex);
    auto it = g_tritonKernels.find(key);
    if (it != g_tritonKernels.end()) {
        return &it->second;
    }

    std::vector<char> cubin;
    if (!ReadBinaryFile(cubinPath, cubin)) {
        return nullptr;
    }

    LoadedTritonKernel loaded;
    loaded.context = context;
    if (!CheckCu(cuModuleLoadData(&loaded.module, cubin.data()), "cuModuleLoadData")) {
        return nullptr;
    }
    if (!CheckCu(cuModuleGetFunction(&loaded.function, loaded.module, kernelName), "cuModuleGetFunction")) {
        cuModuleUnload(loaded.module);
        return nullptr;
    }

    if (shared > 49152) {
        int optin = 0;
        CUdevice cuDevice;
        if (CheckCu(cuCtxGetDevice(&cuDevice), "cuCtxGetDevice") &&
            CheckCu(cuDeviceGetAttribute(&optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice),
                    "cuDeviceGetAttribute")) {
            if (optin > 49152) {
                CheckCu(cuFuncSetCacheConfig(loaded.function, CU_FUNC_CACHE_PREFER_SHARED),
                        "cuFuncSetCacheConfig");
                CheckCu(cuFuncSetAttribute(loaded.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin),
                        "cuFuncSetAttribute");
            }
        }
    }

    auto inserted = g_tritonKernels.emplace(key, loaded);
    return &inserted.first->second;
}

static CUresult LaunchTritonKernel(
    LoadedTritonKernel *kernel, unsigned int gridX, unsigned int gridY, unsigned int gridZ,
    unsigned int blockX, unsigned int shared, void **args, CUstream stream = nullptr) {
    if (gridX == 0 || gridY == 0 || gridZ == 0) {
        return CUDA_SUCCESS;
    }
    if (kernel == nullptr || kernel->function == nullptr || kernel->context == nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (!CheckCu(cuCtxSetCurrent(kernel->context), "cuCtxSetCurrent")) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    cudaGetLastError();
    CUlaunchConfig config = {};
    config.gridDimX = gridX;
    config.gridDimY = gridY;
    config.gridDimZ = gridZ;
    config.blockDimX = blockX;
    config.blockDimY = 1;
    config.blockDimZ = 1;
    config.sharedMemBytes = shared;
    config.hStream = stream;
    config.attrs = nullptr;
    config.numAttrs = 0;
    return cuLaunchKernelEx(&config, kernel->function, args, nullptr);
}

static bool TritonEnvFlagEnabled(const char *name) {
    const char *v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

static bool TritonEnvFlagDefaultEnabled(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    return TritonEnvFlagEnabled(name);
}

template <typename T>
__global__ void FastllmLinearFp8GroupQuant128Kernel(
    const T *__restrict__ input, uint8_t *__restrict__ output, float *__restrict__ scales,
    int totalGroups, int groupsPerRow) {
    constexpr int groupSize = 128;
    constexpr int threadsPerGroup = 16;
    constexpr int valuesPerThread = groupSize / threadsPerGroup;

    int localGroup = threadIdx.x / threadsPerGroup;
    int lane = threadIdx.x % threadsPerGroup;
    int groupsPerBlock = blockDim.x / threadsPerGroup;
    int globalGroup = blockIdx.x * groupsPerBlock + localGroup;
    if (globalGroup >= totalGroups) {
        return;
    }

    int row = globalGroup / groupsPerRow;
    int group = globalGroup - row * groupsPerRow;
    int base = (row * groupsPerRow + group) * groupSize;

    float values[valuesPerThread];
    float localAbsMax = 1.0e-10f;
#pragma unroll
    for (int i = 0; i < valuesPerThread; i++) {
        int offset = lane + i * threadsPerGroup;
        float value = static_cast<float>(input[base + offset]);
        values[i] = value;
        localAbsMax = fmaxf(localAbsMax, fabsf(value));
    }

    unsigned int mask = (threadIdx.x % 32 >= 16) ? 0xffff0000u : 0x0000ffffu;
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 8));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 4));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 2));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 1));

    float scale = localAbsMax * (1.0f / 448.0f);
    if (lane == 0) {
        scales[globalGroup] = scale;
    }
    float invScale = 1.0f / scale;
#pragma unroll
    for (int i = 0; i < valuesPerThread; i++) {
        int offset = lane + i * threadsPerGroup;
        float q = fminf(fmaxf(values[i] * invScale, -448.0f), 448.0f);
        output[base + offset] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
}

static bool LaunchFastllmLinearFp8NativeQuant128(
    const void *input, fastllm::DataType inputType, uint8_t *output, float *scales,
    int rows, int cols) {
    if (input == nullptr || output == nullptr || scales == nullptr || rows <= 0 ||
        cols <= 0 || (cols % 128) != 0) {
        return false;
    }
    int groupsPerRow = cols / 128;
    int totalGroups = rows * groupsPerRow;
    int groupsPerBlock = 8;
    dim3 block(groupsPerBlock * 16);
    dim3 grid((totalGroups + groupsPerBlock - 1) / groupsPerBlock);
    cudaStream_t stream = cudaStreamPerThread;
    cudaError_t state = cudaGetLastError();
    (void)state;
    if (inputType == fastllm::DataType::FLOAT16) {
        FastllmLinearFp8GroupQuant128Kernel<half><<<grid, block, 0, stream>>>(
            (const half*)input, output, scales, totalGroups, groupsPerRow);
    } else if (inputType == fastllm::DataType::BFLOAT16) {
        FastllmLinearFp8GroupQuant128Kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            (const __nv_bfloat16*)input, output, scales, totalGroups, groupsPerRow);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
__global__ void FastllmDeepSeekV4WoAQuant128Kernel(
    const T *__restrict__ input, uint8_t *__restrict__ output, float *__restrict__ scales,
    int totalGroups) {
    constexpr int groupSize = 128;
    constexpr int threadsPerGroup = 16;
    constexpr int valuesPerThread = groupSize / threadsPerGroup;

    int localGroup = threadIdx.x / threadsPerGroup;
    int lane = threadIdx.x % threadsPerGroup;
    int groupsPerBlock = blockDim.x / threadsPerGroup;
    int globalGroup = blockIdx.x * groupsPerBlock + localGroup;
    if (globalGroup >= totalGroups) {
        return;
    }

    int base = globalGroup * groupSize;
    float values[valuesPerThread];
    float localAbsMax = 1.0e-10f;
#pragma unroll
    for (int i = 0; i < valuesPerThread; i++) {
        int offset = lane + i * threadsPerGroup;
        float value = static_cast<float>(input[base + offset]);
        values[i] = value;
        localAbsMax = fmaxf(localAbsMax, fabsf(value));
    }

    unsigned int mask = (threadIdx.x % 32 >= 16) ? 0xffff0000u : 0x0000ffffu;
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 8));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 4));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 2));
    localAbsMax = fmaxf(localAbsMax, __shfl_xor_sync(mask, localAbsMax, 1));

    // DeepSeek-V4 stores activation scales with UE8M0 semantics.  vLLM keeps
    // the SM12x buffer as FP32, but still rounds every scale up to a power of
    // two before quantizing.
    float scaleRaw = localAbsMax * (1.0f / 448.0f);
    float scale = exp2f(ceilf(log2f(scaleRaw)));
    if (lane == 0) {
        scales[globalGroup] = scale;
    }
    float invScale = 1.0f / scale;
#pragma unroll
    for (int i = 0; i < valuesPerThread; i++) {
        int offset = lane + i * threadsPerGroup;
        float q = fminf(fmaxf(values[i] * invScale, -448.0f), 448.0f);
        output[base + offset] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
}

static bool LaunchFastllmDeepSeekV4WoAQuant128(
    const void *input, fastllm::DataType inputType,
    uint8_t *output, float *scales, int elements) {
    if (input == nullptr || output == nullptr || scales == nullptr ||
        elements <= 0 || (elements % 128) != 0) {
        return false;
    }
    int totalGroups = elements / 128;
    constexpr int groupsPerBlock = 8;
    dim3 block(groupsPerBlock * 16);
    dim3 grid((totalGroups + groupsPerBlock - 1) / groupsPerBlock);
    cudaGetLastError();
    if (inputType == fastllm::DataType::BFLOAT16) {
        FastllmDeepSeekV4WoAQuant128Kernel<__nv_bfloat16><<<grid, block>>>(
            (const __nv_bfloat16*)input, output, scales, totalGroups);
    } else if (inputType == fastllm::DataType::FLOAT16) {
        FastllmDeepSeekV4WoAQuant128Kernel<half><<<grid, block>>>(
            (const half*)input, output, scales, totalGroups);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

struct TritonMoeFp8ExpertTable {
    bool inited = false;
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gateBlockM = 0;
    int gateBlockK = 0;
    int downBlockM = 0;
    int downBlockK = 0;
    uint8_t **gateWeights = nullptr;
    float **gateScales = nullptr;
    uint8_t **downWeights = nullptr;
    float **downScales = nullptr;
    bool fusedSeparateGateUp = false;
    uint8_t *fusedGateWeights = nullptr;
    uint8_t *fusedUpWeights = nullptr;
    float *fusedGateScales = nullptr;
    float *fusedUpScales = nullptr;
    bool packedInited = false;
    bool packedBlock128 = false;
    uint8_t *packedGateWeights = nullptr;
    float *packedGateScales = nullptr;
    uint8_t *packedDownWeights = nullptr;
    float *packedDownScales = nullptr;
    bool sourceWeightsReleased = false;
};

static std::mutex g_tritonMoeFp8TableMutex;
static std::map<std::pair<int, const void*>, TritonMoeFp8ExpertTable> g_tritonMoeFp8ExpertTables;
static std::mutex g_tritonMoeFp8FusedTableMutex;
static std::map<std::tuple<int, const void*, const void*, const void*>, TritonMoeFp8ExpertTable> g_tritonMoeFp8FusedTables;

enum TritonMergeMoeFp8KernelId {
    kTritonMoeInitCount = 0,
    kTritonMoeZeroRoute = 1,
    kTritonMoeCount = 2,
    kTritonMoePrefix = 3,
    kTritonMoeFillSorted = 4,
    kTritonMoeScatterBlocks = 5,
    kTritonMoeQuantInput = 6,
    kTritonMoeGateUp = 7,
    kTritonMoeGateUpFused = 8,
    kTritonMoeSwigluQuant = 9,
    kTritonMoeDown = 10,
    kTritonMoeSumOutput = 11,
    kTritonMoeKernelCount = 12,
};

struct TritonMoeKernelSet {
    LoadedTritonKernel *kernels[kTritonMoeKernelCount] = {};
};

static std::mutex g_tritonMoeKernelSetMutex;
static std::map<std::string, TritonMoeKernelSet> g_tritonMoeKernelSets;

static bool LoadTritonMoeKernelSet(
    const char *const *cubinPaths, const char *const *kernelNames, const int *shared,
    LoadedTritonKernel **kernels) {
    if (cubinPaths == nullptr || kernelNames == nullptr || shared == nullptr || kernels == nullptr) {
        return false;
    }
    for (int i = 0; i < kTritonMoeKernelCount; i++) {
        if (cubinPaths[i] == nullptr || kernelNames[i] == nullptr ||
            cubinPaths[i][0] == '\0' || kernelNames[i][0] == '\0') {
            return false;
        }
    }

    int device = -1;
    cudaGetDevice(&device);
    if (!EnsureCudaDriverInitialized() || !EnsureCudaDriverContext()) {
        return false;
    }
    CUcontext context = nullptr;
    if (!CheckCu(cuCtxGetCurrent(&context), "cuCtxGetCurrent")) {
        return false;
    }

    std::string key = std::to_string(device) + ":" + std::to_string((uintptr_t)context);
    for (int i = 0; i < kTritonMoeKernelCount; i++) {
        key += ":";
        key += cubinPaths[i];
        key += ":";
        key += kernelNames[i];
        key += ":";
        key += std::to_string(shared[i]);
    }
    {
        std::lock_guard<std::mutex> guard(g_tritonMoeKernelSetMutex);
        auto it = g_tritonMoeKernelSets.find(key);
        if (it != g_tritonMoeKernelSets.end()) {
            for (int i = 0; i < kTritonMoeKernelCount; i++) {
                kernels[i] = it->second.kernels[i];
            }
            return true;
        }
    }

    TritonMoeKernelSet loaded;
    for (int i = 0; i < kTritonMoeKernelCount; i++) {
        loaded.kernels[i] = LoadTritonKernel(cubinPaths[i], kernelNames[i], shared[i]);
        if (loaded.kernels[i] == nullptr) {
            return false;
        }
    }

    std::lock_guard<std::mutex> guard(g_tritonMoeKernelSetMutex);
    auto it = g_tritonMoeKernelSets.find(key);
    if (it == g_tritonMoeKernelSets.end()) {
        it = g_tritonMoeKernelSets.emplace(key, loaded).first;
    }
    for (int i = 0; i < kTritonMoeKernelCount; i++) {
        kernels[i] = it->second.kernels[i];
    }
    return true;
}

struct TritonMoeFp8Scratch {
    int expertCapacity = 0;
    int taskCapacity = 0;
    int downOutputCapacity = 0;
    int inputQuantCapacity = 0;
    int inputScaleCapacity = 0;
    int gateUpCapacity = 0;
    int activationQuantCapacity = 0;
    int activationScaleCapacity = 0;
    int32_t *expertCounts = nullptr;
    int32_t *expertOffsets = nullptr;
    int32_t *expertCursors = nullptr;
    int32_t *expertBlockOffsets = nullptr;
    int32_t *sortedTasks = nullptr;
    int32_t *blockExperts = nullptr;
    int32_t *blockStarts = nullptr;
    int32_t *totalBlocks = nullptr;
    uint8_t *downOutput = nullptr;
    uint8_t *inputQuant = nullptr;
    float *inputScale = nullptr;
    uint8_t *gateUp = nullptr;
    uint8_t *activationQuant = nullptr;
    float *activationScale = nullptr;
    std::vector<void*> retiredPointers;
};

static std::mutex g_tritonMoeFp8ScratchMutex;
static std::map<int, TritonMoeFp8Scratch> g_tritonMoeFp8Scratch;

template <typename T>
static void FreeTritonScratchPtr(T *&ptr) {
    if (ptr != nullptr) {
        FastllmCudaFree(ptr);
        ptr = nullptr;
    }
}

template <typename T>
static void ReleaseOrRetainTritonScratchPtr(
    T *&ptr, std::vector<void*> &retiredPointers) {
    if (ptr == nullptr) {
        return;
    }
    // Captured graphs keep the scratch addresses embedded in their kernel
    // nodes. Preserve old generations in graph mode; geometric growth keeps
    // their total size below the current generation's capacity.
    if (fastllm::GetFastllmEnv().cudaGraph) {
        retiredPointers.push_back(ptr);
        ptr = nullptr;
    } else {
        FreeTritonScratchPtr(ptr);
    }
}

static bool PreserveTritonScratchAddresses() {
    return fastllm::GetFastllmEnv().cudaGraph;
}

static int NextTritonScratchCapacity(int current, int required) {
    if (current <= 0) {
        return required;
    }
    int64_t doubled = (int64_t)current * 2;
    return doubled > required && doubled <= INT32_MAX ? (int)doubled : required;
}

template <typename T>
static bool GrowTritonScratchPtr(
    T *&ptr, int &capacity, int required, std::vector<void*> &retiredPointers) {
    if (capacity >= required && ptr != nullptr) {
        return true;
    }
    int newCapacity = NextTritonScratchCapacity(
        ptr == nullptr ? 0 : capacity, required);
    if (!PreserveTritonScratchAddresses()) {
        FreeTritonScratchPtr(ptr);
    }
    T *replacement = (T*)FastllmCudaMalloc((size_t)newCapacity * sizeof(T));
    if (replacement == nullptr) {
        if (ptr == nullptr) {
            capacity = 0;
        }
        return false;
    }
    ReleaseOrRetainTritonScratchPtr(ptr, retiredPointers);
    ptr = replacement;
    capacity = newCapacity;
    return true;
}

struct TritonLinearFp8Scratch {
    int inputQuantCapacity = 0;
    int inputScaleCapacity = 0;
    uint8_t *inputQuant = nullptr;
    float *inputScale = nullptr;
    // CUDA graphs retain the scratch addresses observed during capture.
    // Keep superseded allocations alive when pre-capture grows the batch.
    std::vector<uint8_t*> retiredInputQuant;
    std::vector<float*> retiredInputScale;
};

static std::mutex g_tritonLinearFp8ScratchMutex;
static std::map<int, TritonLinearFp8Scratch> g_tritonLinearFp8Scratch;

static bool EnsureTritonLinearFp8Scratch(
    int inputQuantElements, int inputScaleElements, TritonLinearFp8Scratch *&scratch) {
    if (inputQuantElements <= 0 || inputScaleElements <= 0) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_tritonLinearFp8ScratchMutex);
    TritonLinearFp8Scratch &cached = g_tritonLinearFp8Scratch[deviceId];
    if (cached.inputQuantCapacity < inputQuantElements) {
        uint8_t *newInputQuant =
            (uint8_t*)FastllmCudaMalloc((size_t)inputQuantElements);
        if (newInputQuant == nullptr) {
            return false;
        }
        if (cached.inputQuant != nullptr) {
            cached.retiredInputQuant.push_back(cached.inputQuant);
        }
        cached.inputQuant = newInputQuant;
        cached.inputQuantCapacity = inputQuantElements;
    }
    if (cached.inputScaleCapacity < inputScaleElements) {
        float *newInputScale =
            (float*)FastllmCudaMalloc((size_t)inputScaleElements * sizeof(float));
        if (newInputScale == nullptr) {
            return false;
        }
        if (cached.inputScale != nullptr) {
            cached.retiredInputScale.push_back(cached.inputScale);
        }
        cached.inputScale = newInputScale;
        cached.inputScaleCapacity = inputScaleElements;
    }
    if (cached.inputQuant == nullptr || cached.inputScale == nullptr) {
        return false;
    }
    scratch = &cached;
    return true;
}

struct TritonDeepSeekV4WoAScratch {
    int inputQuantCapacity = 0;
    int inputScaleCapacity = 0;
    uint8_t *inputQuant = nullptr;
    float *inputScale = nullptr;
};

static std::mutex g_tritonDeepSeekV4WoAScratchMutex;
static std::map<int, TritonDeepSeekV4WoAScratch> g_tritonDeepSeekV4WoAScratch;

static bool EnsureTritonDeepSeekV4WoAScratch(
    int inputQuantElements, int inputScaleElements,
    TritonDeepSeekV4WoAScratch *&scratch) {
    if (inputQuantElements <= 0 || inputScaleElements <= 0) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_tritonDeepSeekV4WoAScratchMutex);
    TritonDeepSeekV4WoAScratch &cached = g_tritonDeepSeekV4WoAScratch[deviceId];
    if (cached.inputQuantCapacity < inputQuantElements) {
        FreeTritonScratchPtr(cached.inputQuant);
        cached.inputQuantCapacity = inputQuantElements;
        cached.inputQuant = (uint8_t*)FastllmCudaMalloc((size_t)inputQuantElements);
    }
    if (cached.inputScaleCapacity < inputScaleElements) {
        FreeTritonScratchPtr(cached.inputScale);
        cached.inputScaleCapacity = inputScaleElements;
        cached.inputScale = (float*)FastllmCudaMalloc(
            (size_t)inputScaleElements * sizeof(float));
    }
    if (cached.inputQuant == nullptr || cached.inputScale == nullptr) {
        return false;
    }
    scratch = &cached;
    return true;
}

struct TritonDeepSeekV4SparseDecodeScratch {
    int partialOutputCapacity = 0;
    int partialMaxCapacity = 0;
    int partialDenomCapacity = 0;
    float *partialOutput = nullptr;
    float *partialMax = nullptr;
    float *partialDenom = nullptr;
    std::vector<void*> retiredPointers;
};

static std::mutex g_tritonDeepSeekV4SparseDecodeScratchMutex;
static std::map<int, TritonDeepSeekV4SparseDecodeScratch>
    g_tritonDeepSeekV4SparseDecodeScratch;

static bool EnsureTritonDeepSeekV4SparseDecodeScratch(
    int partialOutputElements, int partialStatsElements,
    TritonDeepSeekV4SparseDecodeScratch *&scratch) {
    if (partialOutputElements <= 0 || partialStatsElements <= 0) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(
        g_tritonDeepSeekV4SparseDecodeScratchMutex);
    TritonDeepSeekV4SparseDecodeScratch &cached =
        g_tritonDeepSeekV4SparseDecodeScratch[deviceId];
    if (!GrowTritonScratchPtr(
            cached.partialOutput, cached.partialOutputCapacity,
            partialOutputElements, cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.partialMax, cached.partialMaxCapacity,
            partialStatsElements, cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.partialDenom, cached.partialDenomCapacity,
            partialStatsElements, cached.retiredPointers)) {
        return false;
    }
    if (cached.partialOutput == nullptr || cached.partialMax == nullptr ||
        cached.partialDenom == nullptr) {
        return false;
    }
    scratch = &cached;
    return true;
}

static void ReleaseTritonMoeFp8CudaPtr(void *ptr, std::vector<void*> &released) {
    if (ptr == nullptr) {
        return;
    }
    for (void *old : released) {
        if (old == ptr) {
            return;
        }
    }
    released.push_back(ptr);
    FastllmCudaForceFree(ptr);
}

static void ReleaseTritonMoeFp8SourceWeight(fastllm::Data *weight) {
    if (weight == nullptr || weight->isFake) {
        return;
    }
    std::vector<void*> released;
    for (void *ptr : weight->extraCudaData) {
        ReleaseTritonMoeFp8CudaPtr(ptr, released);
    }
    for (void *ptr : weight->extraCudaHalfData) {
        ReleaseTritonMoeFp8CudaPtr(ptr, released);
    }
    weight->extraCudaData.clear();
    weight->extraCudaHalfData.clear();

    if (weight->cudaData != nullptr) {
        ReleaseTritonMoeFp8CudaPtr(weight->cudaData, released);
        weight->cudaData = nullptr;
    }
    if (weight->cpuData != nullptr) {
#ifdef USE_MMAP
        if (weight->name.empty()) {
            delete[] weight->cpuData;
        }
#else
        delete[] weight->cpuData;
#endif
        weight->cpuData = nullptr;
    }
    weight->expansionSize = 0;
    weight->expansionBytes = 0;
}

static void ReleaseTritonMoeFp8SourceWeights(
    fastllm::Data **weights, int experts,
    TritonMoeFp8ExpertTable &cached) {
    if (cached.sourceWeightsReleased) {
        return;
    }
    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        ReleaseTritonMoeFp8SourceWeight(weights[idx]);
        ReleaseTritonMoeFp8SourceWeight(weights[idx + 1]);
    }
    cached.sourceWeightsReleased = true;
}

static bool EnsureTritonMoeFp8Scratch(
    int experts, int totalTasks, int downOutputBytes, int inputQuantElements,
    int inputScaleElements, int gateUpBytes, int activationQuantElements, int activationScaleElements,
    TritonMoeFp8Scratch *&scratch) {
    if (experts <= 0 || totalTasks <= 0 || downOutputBytes <= 0) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8ScratchMutex);
    TritonMoeFp8Scratch &cached = g_tritonMoeFp8Scratch[deviceId];

    if (cached.expertCapacity < experts || cached.expertCounts == nullptr ||
        cached.expertOffsets == nullptr || cached.expertCursors == nullptr ||
        cached.expertBlockOffsets == nullptr) {
        int currentCapacity = cached.expertCounts == nullptr ||
                              cached.expertOffsets == nullptr ||
                              cached.expertCursors == nullptr ||
                              cached.expertBlockOffsets == nullptr ?
            0 : cached.expertCapacity;
        int newCapacity = NextTritonScratchCapacity(currentCapacity, experts);
        if (!PreserveTritonScratchAddresses()) {
            FreeTritonScratchPtr(cached.expertCounts);
            FreeTritonScratchPtr(cached.expertOffsets);
            FreeTritonScratchPtr(cached.expertCursors);
            FreeTritonScratchPtr(cached.expertBlockOffsets);
        }
        int32_t *expertCounts =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        int32_t *expertOffsets =
            (int32_t*)FastllmCudaMalloc((size_t)(newCapacity + 1) * sizeof(int32_t));
        int32_t *expertCursors =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        int32_t *expertBlockOffsets =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        if (expertCounts == nullptr || expertOffsets == nullptr ||
            expertCursors == nullptr || expertBlockOffsets == nullptr) {
            FreeTritonScratchPtr(expertCounts);
            FreeTritonScratchPtr(expertOffsets);
            FreeTritonScratchPtr(expertCursors);
            FreeTritonScratchPtr(expertBlockOffsets);
            if (cached.expertCounts == nullptr || cached.expertOffsets == nullptr ||
                cached.expertCursors == nullptr ||
                cached.expertBlockOffsets == nullptr) {
                cached.expertCapacity = 0;
            }
            return false;
        }
        ReleaseOrRetainTritonScratchPtr(
            cached.expertCounts, cached.retiredPointers);
        ReleaseOrRetainTritonScratchPtr(
            cached.expertOffsets, cached.retiredPointers);
        ReleaseOrRetainTritonScratchPtr(
            cached.expertCursors, cached.retiredPointers);
        ReleaseOrRetainTritonScratchPtr(
            cached.expertBlockOffsets, cached.retiredPointers);
        cached.expertCapacity = newCapacity;
        cached.expertCounts = expertCounts;
        cached.expertOffsets = expertOffsets;
        cached.expertCursors = expertCursors;
        cached.expertBlockOffsets = expertBlockOffsets;
    }
    if (cached.taskCapacity < totalTasks || cached.sortedTasks == nullptr ||
        cached.blockExperts == nullptr || cached.blockStarts == nullptr) {
        int currentCapacity = cached.sortedTasks == nullptr ||
                              cached.blockExperts == nullptr ||
                              cached.blockStarts == nullptr ?
            0 : cached.taskCapacity;
        int newCapacity = NextTritonScratchCapacity(currentCapacity, totalTasks);
        if (!PreserveTritonScratchAddresses()) {
            FreeTritonScratchPtr(cached.sortedTasks);
            FreeTritonScratchPtr(cached.blockExperts);
            FreeTritonScratchPtr(cached.blockStarts);
        }
        int32_t *sortedTasks =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        int32_t *blockExperts =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        int32_t *blockStarts =
            (int32_t*)FastllmCudaMalloc((size_t)newCapacity * sizeof(int32_t));
        if (sortedTasks == nullptr || blockExperts == nullptr || blockStarts == nullptr) {
            FreeTritonScratchPtr(sortedTasks);
            FreeTritonScratchPtr(blockExperts);
            FreeTritonScratchPtr(blockStarts);
            if (cached.sortedTasks == nullptr || cached.blockExperts == nullptr ||
                cached.blockStarts == nullptr) {
                cached.taskCapacity = 0;
            }
            return false;
        }
        ReleaseOrRetainTritonScratchPtr(
            cached.sortedTasks, cached.retiredPointers);
        ReleaseOrRetainTritonScratchPtr(
            cached.blockExperts, cached.retiredPointers);
        ReleaseOrRetainTritonScratchPtr(
            cached.blockStarts, cached.retiredPointers);
        cached.taskCapacity = newCapacity;
        cached.sortedTasks = sortedTasks;
        cached.blockExperts = blockExperts;
        cached.blockStarts = blockStarts;
    }
    if (cached.totalBlocks == nullptr) {
        cached.totalBlocks = (int32_t*)FastllmCudaMalloc(sizeof(int32_t));
    }
    if (cached.totalBlocks == nullptr ||
        !GrowTritonScratchPtr(
            cached.downOutput, cached.downOutputCapacity, downOutputBytes,
            cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.inputQuant, cached.inputQuantCapacity, inputQuantElements,
            cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.inputScale, cached.inputScaleCapacity, inputScaleElements,
            cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.gateUp, cached.gateUpCapacity, gateUpBytes,
            cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.activationQuant, cached.activationQuantCapacity,
            activationQuantElements, cached.retiredPointers) ||
        !GrowTritonScratchPtr(
            cached.activationScale, cached.activationScaleCapacity,
            activationScaleElements, cached.retiredPointers) ||
        cached.expertCounts == nullptr || cached.expertOffsets == nullptr ||
        cached.expertCursors == nullptr || cached.expertBlockOffsets == nullptr ||
        cached.sortedTasks == nullptr || cached.blockExperts == nullptr ||
        cached.blockStarts == nullptr || cached.totalBlocks == nullptr ||
        cached.downOutput == nullptr || cached.inputQuant == nullptr ||
        cached.inputScale == nullptr || cached.gateUp == nullptr || cached.activationQuant == nullptr ||
        cached.activationScale == nullptr) {
        return false;
    }
    scratch = &cached;
    return true;
}

static bool PackTritonMoeFp8ExpertTable(
    fastllm::Data **weights, int experts, int hidden, int inter,
    TritonMoeFp8ExpertTable &cached) {
    if (cached.packedInited) {
        return true;
    }
    int gateScaleRows = (inter * 2 + cached.gateBlockK - 1) / cached.gateBlockK;
    int gateScaleCols = (hidden + cached.gateBlockM - 1) / cached.gateBlockM;
    int downScaleRows = (hidden + cached.downBlockK - 1) / cached.downBlockK;
    int downScaleCols = (inter + cached.downBlockM - 1) / cached.downBlockM;
    size_t gateWeightBytes = (size_t)inter * 2 * hidden;
    size_t downWeightBytes = (size_t)hidden * inter;
    size_t gateScaleBytes = (size_t)gateScaleRows * gateScaleCols * sizeof(float);
    size_t downScaleBytes = (size_t)downScaleRows * downScaleCols * sizeof(float);

    cached.packedGateWeights = (uint8_t*)FastllmCudaMalloc((size_t)experts * gateWeightBytes);
    cached.packedDownWeights = (uint8_t*)FastllmCudaMalloc((size_t)experts * downWeightBytes);
    cached.packedGateScales = (float*)FastllmCudaMalloc((size_t)experts * gateScaleBytes);
    cached.packedDownScales = (float*)FastllmCudaMalloc((size_t)experts * downScaleBytes);
    if (cached.packedGateWeights == nullptr || cached.packedDownWeights == nullptr ||
        cached.packedGateScales == nullptr || cached.packedDownScales == nullptr) {
        return false;
    }

    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        fastllm::Data *gateup = weights[idx];
        fastllm::Data *down = weights[idx + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->cudaData == nullptr || down->cudaData == nullptr ||
            gateup->extraCudaData.empty() || down->extraCudaData.empty()) {
            return false;
        }
        cudaError_t state = cudaMemcpyAsync(
            cached.packedGateWeights + (size_t)e * gateWeightBytes,
            gateup->cudaData, gateWeightBytes, cudaMemcpyDeviceToDevice);
        if (state != cudaSuccess) {
            checkCudaErrors("Error: CUDA error when packing Triton MoE gate weights!", state);
            return false;
        }
        state = cudaMemcpyAsync(
            cached.packedDownWeights + (size_t)e * downWeightBytes,
            down->cudaData, downWeightBytes, cudaMemcpyDeviceToDevice);
        if (state != cudaSuccess) {
            checkCudaErrors("Error: CUDA error when packing Triton MoE down weights!", state);
            return false;
        }
        state = cudaMemcpyAsync(
            cached.packedGateScales + (size_t)e * gateScaleRows * gateScaleCols,
            gateup->extraCudaData[0], gateScaleBytes, cudaMemcpyDeviceToDevice);
        if (state != cudaSuccess) {
            checkCudaErrors("Error: CUDA error when packing Triton MoE gate scales!", state);
            return false;
        }
        state = cudaMemcpyAsync(
            cached.packedDownScales + (size_t)e * downScaleRows * downScaleCols,
            down->extraCudaData[0], downScaleBytes, cudaMemcpyDeviceToDevice);
        if (state != cudaSuccess) {
            checkCudaErrors("Error: CUDA error when packing Triton MoE down scales!", state);
            return false;
        }
    }
    cudaError_t syncState = cudaDeviceSynchronize();
    if (syncState != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when synchronizing Triton MoE packed weights!", syncState);
        return false;
    }
    if (!FastllmCudaRegisterMoeFp8ExpertTableFromPacked(
            weights, (experts + 1) * 2, hidden, inter,
            cached.packedGateWeights, cached.packedGateScales,
            cached.packedDownWeights, cached.packedDownScales,
            cached.gateBlockM, cached.gateBlockK, cached.downBlockM, cached.downBlockK)) {
        return false;
    }
    cached.packedInited = true;
    ReleaseTritonMoeFp8SourceWeights(weights, experts, cached);
    return true;
}

static bool GetTritonMoeFp8ExpertTable(
    fastllm::Data **weights, int weightsBatch, int hidden, int inter, bool packWeights,
    TritonMoeFp8ExpertTable *&table) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1)) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8TableMutex);
    TritonMoeFp8ExpertTable &cached = g_tritonMoeFp8ExpertTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter) {
            return false;
        }
        if (!packWeights && cached.sourceWeightsReleased) {
            return false;
        }
        if (packWeights && !PackTritonMoeFp8ExpertTable(weights, experts, hidden, inter, cached)) {
            return false;
        }
        table = &cached;
        return true;
    }

    fastllm::Data emptyBias;
    std::vector<uint8_t*> hGateWeights(experts), hDownWeights(experts);
    std::vector<float*> hGateScales(experts), hDownScales(experts);
    int gateBlockM = -1, gateBlockK = -1, downBlockM = -1, downBlockK = -1;

    for (int e = 0; e < experts; e++) {
        int idx = (e + 1) * 2;
        fastllm::Data *gateup = weights[idx];
        fastllm::Data *down = weights[idx + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != fastllm::DataType::FP8_E4M3 ||
            down->dataType != fastllm::DataType::FP8_E4M3 ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[1] != hidden || gateup->dims[0] != inter * 2 ||
            down->dims[1] != inter || down->dims[0] != hidden ||
            gateup->blockM <= 0 || gateup->blockK <= 0 ||
            down->blockM <= 0 || down->blockK <= 0 ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
        if (gateBlockM < 0) {
            gateBlockM = gateup->blockM;
            gateBlockK = gateup->blockK;
            downBlockM = down->blockM;
            downBlockK = down->blockK;
        } else if (gateBlockM != gateup->blockM || gateBlockK != gateup->blockK ||
                   downBlockM != down->blockM || downBlockK != down->blockK) {
            return false;
        }

        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*gateup, emptyBias, inter);
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(*down, emptyBias, hidden);
        if (gateup->extraCudaData.empty() || down->extraCudaData.empty()) {
            return false;
        }
        hGateWeights[e] = (uint8_t*)gateup->cudaData;
        hDownWeights[e] = (uint8_t*)down->cudaData;
        hGateScales[e] = (float*)gateup->extraCudaData[0];
        hDownScales[e] = (float*)down->extraCudaData[0];
    }

    size_t ptrBytes = (size_t)experts * sizeof(void*);
    cached.gateWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.gateScales = (float**)FastllmCudaMalloc(ptrBytes);
    cached.downWeights = (uint8_t**)FastllmCudaMalloc(ptrBytes);
    cached.downScales = (float**)FastllmCudaMalloc(ptrBytes);
    if (cached.gateWeights == nullptr || cached.gateScales == nullptr ||
        cached.downWeights == nullptr || cached.downScales == nullptr) {
        return false;
    }

    cudaError_t state = cudaMemcpyAsync(cached.gateWeights, hGateWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when caching Triton MoE gate pointer table!", state);
        return false;
    }
    state = cudaMemcpyAsync(cached.gateScales, hGateScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when caching Triton MoE gate scale table!", state);
        return false;
    }
    state = cudaMemcpyAsync(cached.downWeights, hDownWeights.data(), ptrBytes, cudaMemcpyHostToDevice);
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when caching Triton MoE down pointer table!", state);
        return false;
    }
    state = cudaMemcpyAsync(cached.downScales, hDownScales.data(), ptrBytes, cudaMemcpyHostToDevice);
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when caching Triton MoE down scale table!", state);
        return false;
    }

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = gateBlockM;
    cached.gateBlockK = gateBlockK;
    cached.downBlockM = downBlockM;
    cached.downBlockK = downBlockK;
    if (packWeights && !PackTritonMoeFp8ExpertTable(weights, experts, hidden, inter, cached)) {
        return false;
    }
    table = &cached;
    return true;
}

/**
 * 将FastLLM交错Block128专家权重整理为Triton grouped MoE连续权重缓存。
 *
 * 每个权重逻辑行为FP8值，每128个值后紧跟一个FP32 scale。这里只按expert
 * 顺序拼接原布局，不拆分scale，也不释放源权重；后续kernel直接按交错布局
 * 读取。任一专家类型、形状或设备指针不满足约束时返回false，由正式主流程
 * 回退已有CUDA实现。
 *
 * @param weights       FastLLM专家权重数组，第0对为可选shared expert。
 * @param weightsBatch  数组元素数量，必须为2 * (expert数量 + 1)。
 * @param hidden        输入及输出隐藏维度，必须按128对齐。
 * @param inter         专家中间维度，必须按128对齐。
 * @param table         返回可供Triton grouped MoE消费的连续缓存。
 * @return true表示缓存已存在或成功建立；false表示布局不受支持或复制失败。
 */
static bool GetTritonMoeFp8Block128ExpertTable(
    fastllm::Data **weights, int weightsBatch, int hidden, int inter,
    TritonMoeFp8ExpertTable *&table) {
    table = nullptr;
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        hidden <= 0 || inter <= 0 || (hidden % 128) != 0 ||
        (inter % 128) != 0) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0 || weights[2] == nullptr) {
        return false;
    }

    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8TableMutex);
    TritonMoeFp8ExpertTable &cached = g_tritonMoeFp8ExpertTables[key];
    if (cached.inited) {
        if (!cached.packedBlock128 || !cached.packedInited ||
            cached.experts != experts || cached.hidden != hidden ||
            cached.inter != inter) {
            return false;
        }
        table = &cached;
        return true;
    }

    const size_t gatePerRow = (size_t)hidden + (size_t)(hidden / 128) * sizeof(float);
    const size_t downPerRow = (size_t)inter + (size_t)(inter / 128) * sizeof(float);
    const size_t gateBytes = (size_t)inter * 2 * gatePerRow;
    const size_t downBytes = (size_t)hidden * downPerRow;
    for (int expert = 0; expert < experts; ++expert) {
        int index = (expert + 1) * 2;
        fastllm::Data *gateup = weights[index];
        fastllm::Data *down = weights[index + 1];
        if (gateup == nullptr || down == nullptr ||
            gateup->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
            down->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 ||
            gateup->dims.size() != 2 || down->dims.size() != 2 ||
            gateup->dims[0] != inter * 2 || gateup->dims[1] != hidden ||
            down->dims[0] != hidden || down->dims[1] != inter ||
            gateup->cudaData == nullptr || down->cudaData == nullptr) {
            return false;
        }
    }

    cached.packedGateWeights =
        (uint8_t*)FastllmCudaMalloc((size_t)experts * gateBytes);
    cached.packedDownWeights =
        (uint8_t*)FastllmCudaMalloc((size_t)experts * downBytes);
    if (cached.packedGateWeights == nullptr ||
        cached.packedDownWeights == nullptr) {
        FreeTritonScratchPtr(cached.packedGateWeights);
        FreeTritonScratchPtr(cached.packedDownWeights);
        return false;
    }

    for (int expert = 0; expert < experts; ++expert) {
        int index = (expert + 1) * 2;
        fastllm::Data *gateup = weights[index];
        fastllm::Data *down = weights[index + 1];

        cudaError_t state = cudaMemcpyAsync(
            cached.packedGateWeights + (size_t)expert * gateBytes,
            gateup->cudaData, gateBytes, cudaMemcpyDeviceToDevice,
            cudaStreamPerThread);
        if (state != cudaSuccess) {
            checkCudaErrors(
                "Error: CUDA error when packing Block128 MoE gate weights!",
                state);
            FreeTritonScratchPtr(cached.packedGateWeights);
            FreeTritonScratchPtr(cached.packedDownWeights);
            return false;
        }
        state = cudaMemcpyAsync(
            cached.packedDownWeights + (size_t)expert * downBytes,
            down->cudaData, downBytes, cudaMemcpyDeviceToDevice,
            cudaStreamPerThread);
        if (state != cudaSuccess) {
            checkCudaErrors(
                "Error: CUDA error when packing Block128 MoE down weights!",
                state);
            FreeTritonScratchPtr(cached.packedGateWeights);
            FreeTritonScratchPtr(cached.packedDownWeights);
            return false;
        }
    }

    cached.inited = true;
    cached.packedInited = true;
    cached.packedBlock128 = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = 128;
    cached.gateBlockK = 128;
    cached.downBlockM = 128;
    cached.downBlockK = 128;
    table = &cached;
    return true;
}

/**
 * 将3D FusedMOE权重绑定为Triton grouped MoE专家表。
 *
 * 普通FP8权重继续使用独立scale；Block128权重直接保留每行交错存储的
 * FP8值与FP32 scale，并标记为packed布局。函数不取得源权重所有权，目标
 * kernel不可用时仍可由原CUDA FusedMOE读取同一权重完成fallback。
 *
 * @param gate       gate权重，逻辑形状为[experts, inter, hidden]。
 * @param up         up权重，逻辑形状为[experts, inter, hidden]。
 * @param down       down权重，逻辑形状为[experts, hidden, inter]。
 * @param experts    专家数量。
 * @param hidden     隐藏维度；Block128布局要求按128对齐。
 * @param inter      专家中间维度；Block128布局要求按128对齐。
 * @param table      返回与源权重生命周期绑定的专家表。
 * @return true表示布局及形状有效；false表示应回退原CUDA实现。
 */
static bool GetTritonMoeFp8FusedExpertTable(
    fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
    int experts, int hidden, int inter, TritonMoeFp8ExpertTable *&table) {
    table = nullptr;
    bool packedBlock128 =
        gate.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128 &&
        up.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128 &&
        down.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128;
    bool separateScales =
        gate.dataType == fastllm::DataType::FP8_E4M3 &&
        up.dataType == fastllm::DataType::FP8_E4M3 &&
        down.dataType == fastllm::DataType::FP8_E4M3;
    if (experts <= 0 || hidden <= 0 || inter <= 0 ||
        gate.dataDevice != fastllm::DataDevice::CUDA ||
        up.dataDevice != fastllm::DataDevice::CUDA ||
        down.dataDevice != fastllm::DataDevice::CUDA ||
        (!packedBlock128 && !separateScales) ||
        (packedBlock128 && ((hidden % 128) != 0 || (inter % 128) != 0)) ||
        gate.cudaData == nullptr || up.cudaData == nullptr || down.cudaData == nullptr ||
        (separateScales &&
         (gate.extraCudaData.empty() || up.extraCudaData.empty() || down.extraCudaData.empty())) ||
        gate.dims.size() != 3 || up.dims.size() != 3 || down.dims.size() != 3 ||
        gate.dims[0] != experts || gate.dims[1] != inter || gate.dims[2] != hidden ||
        up.dims[0] != experts || up.dims[1] != inter || up.dims[2] != hidden ||
        down.dims[0] != experts || down.dims[1] != hidden || down.dims[2] != inter ||
        (separateScales &&
         (gate.blockM <= 0 || gate.blockK <= 0 || down.blockM <= 0 || down.blockK <= 0 ||
          up.blockM != gate.blockM || up.blockK != gate.blockK))) {
        return false;
    }
    int gateBlockM = packedBlock128 ? 128 : gate.blockM;
    int gateBlockK = packedBlock128 ? 128 : gate.blockK;
    int downBlockM = packedBlock128 ? 128 : down.blockM;
    int downBlockK = packedBlock128 ? 128 : down.blockK;
    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_tuple(deviceId, (const void*)gate.cudaData, (const void*)up.cudaData, (const void*)down.cudaData);
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8FusedTableMutex);
    TritonMoeFp8ExpertTable &cached = g_tritonMoeFp8FusedTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter ||
            cached.packedBlock128 != packedBlock128 ||
            cached.gateBlockM != gateBlockM || cached.gateBlockK != gateBlockK ||
            cached.downBlockM != downBlockM || cached.downBlockK != downBlockK) {
            return false;
        }
        table = &cached;
        return true;
    }

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = gateBlockM;
    cached.gateBlockK = gateBlockK;
    cached.downBlockM = downBlockM;
    cached.downBlockK = downBlockK;
    cached.fusedSeparateGateUp = true;
    cached.fusedGateWeights = (uint8_t*)gate.cudaData;
    cached.fusedUpWeights = (uint8_t*)up.cudaData;
    cached.fusedGateScales = separateScales ? (float*)gate.extraCudaData[0] : nullptr;
    cached.fusedUpScales = separateScales ? (float*)up.extraCudaData[0] : nullptr;
    cached.packedDownWeights = (uint8_t*)down.cudaData;
    cached.packedDownScales = separateScales ? (float*)down.extraCudaData[0] : nullptr;
    cached.packedInited = packedBlock128;
    cached.packedBlock128 = packedBlock128;
    table = &cached;
    return true;
}
}

extern "C" bool FastllmCudaTritonMergeMOEFP8E4M3IndexedIsPacked(
    fastllm::Data **weights, int weightsBatch, int hidden, int inter) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        weights[2] == nullptr || hidden <= 0 || inter <= 0) {
        return false;
    }
    int experts = weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_pair(deviceId, (const void*)weights[2]);
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8TableMutex);
    auto it = g_tritonMoeFp8ExpertTables.find(key);
    if (it == g_tritonMoeFp8ExpertTables.end()) {
        return false;
    }
    const TritonMoeFp8ExpertTable &cached = it->second;
    return cached.inited && cached.packedInited &&
           cached.experts == experts && cached.hidden == hidden && cached.inter == inter;
}

extern "C" int FastllmCudaRuntimeArch() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 0;
    }
    // 计算能力是稳定设备属性；同一CPU线程固定使用当前GPU时直接复用。
    static thread_local int cachedDevice = -1;
    static thread_local int cachedArch = 0;
    if (cachedDevice == device) {
        return cachedArch;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }
    cachedDevice = device;
    cachedArch = prop.major * 10 + prop.minor;
    return cachedArch;
}

extern "C" bool FastllmCudaTritonLinearFP8E4M3Block128(
    const char *quantCubitPath, const char *quantKernelName, int quantNumWarps, int quantShared,
    const char *matmulCubitPath, const char *matmulKernelName, int matmulNumWarps, int matmulShared,
    int blockM, int blockN, int blockK, int groupSizeM, bool packedWeight, bool stridedMatmul,
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output,
    int n, int m, int k) {
    if (quantCubitPath == nullptr || quantKernelName == nullptr ||
        matmulCubitPath == nullptr || matmulKernelName == nullptr ||
        quantNumWarps <= 0 || matmulNumWarps <= 0 ||
        blockM <= 0 || blockN <= 0 || blockK != 128 || groupSizeM <= 0 ||
        n <= 0 || m <= 0 || k <= 0 ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        (packedWeight ? weight.dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 :
                        weight.dataType != fastllm::DataType::FP8_E4M3) ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType) {
        return false;
    }
    if (!packedWeight && (weight.blockK != 128 || weight.blockM != 128 || weight.scales.empty())) {
        return false;
    }
    bool hasBias = bias.dims.size() > 0;
    if (hasBias && (bias.cudaData == nullptr || bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    if (stridedMatmul && (packedWeight || hasBias)) {
        return false;
    }

    if (!packedWeight && weight.extraCudaData.empty()) {
        float *cudaScales = nullptr;
        cudaError_t state = cudaMalloc(&cudaScales, weight.scales.size() * sizeof(float));
        if (state != cudaSuccess || cudaScales == nullptr) {
            return false;
        }
        state = cudaMemcpy(cudaScales, weight.scales.data(),
                           weight.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (state != cudaSuccess) {
            cudaFree(cudaScales);
            return false;
        }
        weight.extraCudaData.push_back((void*)cudaScales);
    }

    bool useNativeQuant = TritonEnvFlagDefaultEnabled("FASTLLM_CUDA_TRITON_LINEAR_FP8_NATIVE_QUANT", true);
    LoadedTritonKernel *quantKernel = useNativeQuant ? nullptr :
        LoadTritonKernel(quantCubitPath, quantKernelName, quantShared);
    LoadedTritonKernel *matmulKernel = LoadTritonKernel(matmulCubitPath, matmulKernelName, matmulShared);
    if ((!useNativeQuant && quantKernel == nullptr) || matmulKernel == nullptr) {
        return false;
    }

    int inputScaleCols = (m + blockK - 1) / blockK;
    TritonLinearFp8Scratch *scratch = nullptr;
    if (!EnsureTritonLinearFp8Scratch(n * m, n * inputScaleCols, scratch)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    int perRow = m + ((m - 1) / 128 + 1) * (int)sizeof(float);
    int scaleCols = (m + blockK - 1) / blockK;
    int32_t mArg = n;
    int32_t nArg = k;
    int32_t kArg = m;
    int32_t perRowArg = perRow;
    int32_t scaleColsArg = scaleCols;
    CUdeviceptr inputPtr = (CUdeviceptr)inputData;
    CUdeviceptr inputQuantPtr = (CUdeviceptr)scratch->inputQuant;
    CUdeviceptr inputScalePtr = (CUdeviceptr)scratch->inputScale;
    CUdeviceptr weightPtr = (CUdeviceptr)weight.cudaData;
    CUdeviceptr weightScalePtr = packedWeight ? (CUdeviceptr)0 : (CUdeviceptr)weight.extraCudaData[0];
    CUdeviceptr biasPtr = hasBias ? (CUdeviceptr)bias.cudaData : (CUdeviceptr)0;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);

    CUresult result = CUDA_SUCCESS;
    if (useNativeQuant) {
        if (!LaunchFastllmLinearFp8NativeQuant128(
                inputData, input.dataType, scratch->inputQuant, scratch->inputScale, n, m)) {
            FastllmCudaFinishInput(input, inputData);
            FastllmCudaFinishOutput(output, outputData);
            return false;
        }
    } else {
        void *quantArgs[] = {
            &inputPtr,
            &inputQuantPtr,
            &inputScalePtr,
            &mArg,
            &kArg,
            &globalScratch,
            &profileScratch,
        };
        result = LaunchTritonKernel(
            quantKernel,
            (unsigned int)n, (unsigned int)inputScaleCols, 1,
            (unsigned int)(quantNumWarps * 32), (unsigned int)quantShared,
            quantArgs, stream);
        if (!CheckCu(result, "cuLaunchKernel linear_fp8_block128_quant_input")) {
            FastllmCudaFinishInput(input, inputData);
            FastllmCudaFinishOutput(output, outputData);
            return false;
        }
    }

    int gridM = (n + blockM - 1) / blockM;
    int gridN = (k + blockN - 1) / blockN;
    int32_t groupNArg = 128;
    int32_t groupKArg = 128;
    int32_t strideAmArg = m;
    int32_t strideBnArg = m;
    int32_t strideCmArg = k;
    int32_t strideAsMArg = scaleCols;
    int32_t strideBsNArg = scaleCols;
    void *matmulArgsFastllm[] = {
        &inputQuantPtr,
        &inputScalePtr,
        &weightPtr,
        &weightScalePtr,
        &biasPtr,
        &outputPtr,
        &mArg,
        &nArg,
        &kArg,
        &perRowArg,
        &scaleColsArg,
        &globalScratch,
        &profileScratch,
    };
    void *matmulArgsStrided[] = {
        &inputQuantPtr,
        &weightPtr,
        &outputPtr,
        &inputScalePtr,
        &weightScalePtr,
        &mArg,
        &nArg,
        &kArg,
        &groupNArg,
        &groupKArg,
        &strideAmArg,
        &strideBnArg,
        &strideCmArg,
        &strideAsMArg,
        &strideBsNArg,
        &globalScratch,
        &profileScratch,
    };
    result = LaunchTritonKernel(
        matmulKernel,
        (unsigned int)(gridM * gridN), 1, 1,
        (unsigned int)(matmulNumWarps * 32), (unsigned int)matmulShared,
        stridedMatmul ? matmulArgsStrided : matmulArgsFastllm, stream);

    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return CheckCu(result, "cuLaunchKernel linear_fp8_block128_matmul");
}

extern "C" bool FastllmCudaTritonDeepSeekV4WoA(
    const char *cubinPath, const char *kernelName, int numWarps, int shared,
    int blockTokens, int blockOut, int blockHidden,
    const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output,
    int numTokens, int groups, int outRank, int hiddenSize) {
    if (cubinPath == nullptr || kernelName == nullptr || numWarps <= 0 ||
        blockTokens <= 0 || blockOut != 128 || blockHidden != 128 ||
        numTokens <= 0 || groups <= 0 || outRank <= 0 || hiddenSize <= 0 ||
        numTokens > blockTokens || (outRank % blockOut) != 0 ||
        (hiddenSize % blockHidden) != 0 ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        output.cudaData == nullptr || weight.blockK != 128 || weight.blockM != 128 ||
        weight.scales.size() !=
            (size_t)groups * (outRank / blockOut) * (hiddenSize / blockHidden)) {
        return false;
    }

    fastllm::Data emptyBias;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, emptyBias, groups * outRank);
    if (weight.extraCudaData.empty() || weight.extraCudaData[0] == nullptr) {
        return false;
    }

    LoadedTritonKernel *kernel = LoadTritonKernel(cubinPath, kernelName, shared);
    if (kernel == nullptr) {
        return false;
    }

    int inputElements = numTokens * groups * hiddenSize;
    int inputScaleElements = inputElements / blockHidden;
    TritonDeepSeekV4WoAScratch *scratch = nullptr;
    if (!EnsureTritonDeepSeekV4WoAScratch(
            inputElements, inputScaleElements, scratch)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }
    if (!LaunchFastllmDeepSeekV4WoAQuant128(
            inputData, input.dataType, scratch->inputQuant,
            scratch->inputScale, inputElements)) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    CUdeviceptr inputQuantPtr = (CUdeviceptr)scratch->inputQuant;
    CUdeviceptr inputScalePtr = (CUdeviceptr)scratch->inputScale;
    CUdeviceptr weightPtr = (CUdeviceptr)weight.cudaData;
    CUdeviceptr weightScalePtr = (CUdeviceptr)weight.extraCudaData[0];
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *args[] = {
        &inputQuantPtr,
        &inputScalePtr,
        &weightPtr,
        &weightScalePtr,
        &outputPtr,
        &globalScratch,
        &profileScratch,
    };
    CUresult result = LaunchTritonKernel(
        kernel,
        (unsigned int)((numTokens + blockTokens - 1) / blockTokens),
        (unsigned int)(outRank / blockOut),
        (unsigned int)groups,
        (unsigned int)(numWarps * 32),
        (unsigned int)shared,
        args);

    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return CheckCu(result, "cuLaunchKernel deepseek_v4_fp8_woa");
}

extern "C" bool FastllmCudaTritonDeepSeekV4SparseAttentionDecodeGraph(
    const char *splitCubinPath, const char *splitKernelName,
    int splitNumWarps, int splitShared,
    const char *mergeCubinPath, const char *mergeKernelName,
    int mergeNumWarps, int mergeShared,
    int compressedCapacity, int numSplits, int splitSize,
    int splitHeadBlock, int blockD, int mergeBlockD, const fastllm::Data &q,
    const fastllm::Data &windowKV, const fastllm::Data &compressedKV,
    const fastllm::Data &attnSink, int windowSize, int compressRatio,
    const int32_t *decodeMeta, float softmaxScale, float *output) {
    if (splitCubinPath == nullptr || splitKernelName == nullptr ||
        mergeCubinPath == nullptr || mergeKernelName == nullptr ||
        splitNumWarps <= 0 || mergeNumWarps <= 0 ||
        compressedCapacity <= 0 || numSplits <= 0 || numSplits > 256 ||
        (splitHeadBlock != 1 && splitHeadBlock != 16) ||
        (splitSize != 8 && splitSize != 16 &&
         splitSize != 32 && splitSize != 64) ||
        blockD <= 0 || blockD > 1024 ||
        (mergeBlockD != 16 && mergeBlockD != 32 &&
         mergeBlockD != 64 && mergeBlockD != 128) ||
        windowSize <= 0 || compressRatio < 0 ||
        decodeMeta == nullptr || output == nullptr ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 || q.cudaData == nullptr ||
        q.dims.size() != 4 || q.dims[0] != 1 || q.dims[1] != 1 ||
        q.dims[2] <= 0 || q.dims[3] <= 0 || blockD < q.dims[3] ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.cudaData == nullptr ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
        compressedKV.cudaData == nullptr || compressedKV.dims.size() != 3 ||
        compressedKV.dims[0] != 1 || compressedKV.dims[2] != q.dims[3] ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.cudaData == nullptr) {
        return false;
    }
    int runtimeCompressedCapacity = compressedKV.dims[1];
    if (compressedKV.expansionDims.size() >= 3) {
        runtimeCompressedCapacity = std::max(
            runtimeCompressedCapacity, compressedKV.expansionDims[1]);
    }
    int expectedSplits =
        (windowSize + compressedCapacity + splitSize - 1) / splitSize;
    if (runtimeCompressedCapacity != compressedCapacity ||
        expectedSplits != numSplits) {
        return false;
    }
    LoadedTritonKernel *splitKernel = LoadTritonKernel(
        splitCubinPath, splitKernelName, splitShared);
    LoadedTritonKernel *mergeKernel = LoadTritonKernel(
        mergeCubinPath, mergeKernelName, mergeShared);
    if (splitKernel == nullptr || mergeKernel == nullptr) {
        return false;
    }
    int numHeads = q.dims[2];
    int headDim = q.dims[3];
    int64_t partialStatsElements64 = (int64_t)numHeads * numSplits;
    int64_t partialOutputElements64 = partialStatsElements64 * headDim;
    if (partialStatsElements64 <= 0 ||
        partialOutputElements64 > INT32_MAX) {
        return false;
    }
    TritonDeepSeekV4SparseDecodeScratch *scratch = nullptr;
    if (!EnsureTritonDeepSeekV4SparseDecodeScratch(
            (int)partialOutputElements64, (int)partialStatsElements64,
            scratch)) {
        return false;
    }

    CUdeviceptr qPtr = (CUdeviceptr)q.cudaData;
    CUdeviceptr windowPtr = (CUdeviceptr)windowKV.cudaData;
    CUdeviceptr compressedPtr = (CUdeviceptr)compressedKV.cudaData;
    CUdeviceptr decodeMetaPtr = (CUdeviceptr)decodeMeta;
    CUdeviceptr partialOutputPtr = (CUdeviceptr)scratch->partialOutput;
    CUdeviceptr partialMaxPtr = (CUdeviceptr)scratch->partialMax;
    CUdeviceptr partialDenomPtr = (CUdeviceptr)scratch->partialDenom;
    float scaleArg = softmaxScale;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *splitArgs[] = {
        &qPtr,
        &windowPtr,
        &compressedPtr,
        &decodeMetaPtr,
        &partialOutputPtr,
        &partialMaxPtr,
        &partialDenomPtr,
        &scaleArg,
        &globalScratch,
        &profileScratch,
    };
    CUresult splitResult = LaunchTritonKernel(
        splitKernel, 1,
        (unsigned int)((numHeads + splitHeadBlock - 1) / splitHeadBlock),
        (unsigned int)numSplits,
        (unsigned int)(splitNumWarps * 32), (unsigned int)splitShared,
        splitArgs);
    if (!CheckCu(splitResult, "cuLaunchKernel deepseek_v4_sparse_decode_split")) {
        return false;
    }

    CUdeviceptr sinkPtr = (CUdeviceptr)attnSink.cudaData;
    CUdeviceptr outputPtr = (CUdeviceptr)output;
    void *mergeArgs[] = {
        &partialOutputPtr,
        &partialMaxPtr,
        &partialDenomPtr,
        &sinkPtr,
        &decodeMetaPtr,
        &outputPtr,
        &globalScratch,
        &profileScratch,
    };
    CUresult mergeResult = LaunchTritonKernel(
        mergeKernel, 1, (unsigned int)numHeads,
        (unsigned int)((headDim + mergeBlockD - 1) / mergeBlockD),
        (unsigned int)(mergeNumWarps * 32), (unsigned int)mergeShared,
        mergeArgs);
    return CheckCu(
        mergeResult, "cuLaunchKernel deepseek_v4_sparse_decode_merge");
}

extern "C" bool FastllmCudaTritonDeepSeekV4SqrtSoftplusRouter(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared, int numExperts, int topk, int blockN,
    const fastllm::Data &logits, const fastllm::Data &gateBias,
    float routeScale, fastllm::Data &expertIndex,
    fastllm::Data &expertScore) {
    if (cubinPath == nullptr || kernelName == nullptr ||
        numWarps != 1 || numExperts != 256 || topk != 6 || blockN != 256 ||
        logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.cudaData == nullptr || logits.dims.empty() ||
        logits.dims.back() != numExperts || logits.Count(0) == 0 ||
        logits.Count(0) % numExperts != 0 ||
        gateBias.dataDevice != fastllm::DataDevice::CUDA ||
        gateBias.dataType != fastllm::DataType::FLOAT32 ||
        gateBias.cudaData == nullptr || gateBias.Count(0) != numExperts ||
        expertIndex.dataDevice != fastllm::DataDevice::CUDA ||
        expertIndex.dataType != fastllm::DataType::INT32 ||
        expertIndex.cudaData == nullptr ||
        expertScore.dataDevice != fastllm::DataDevice::CUDA ||
        expertScore.dataType != fastllm::DataType::FLOAT32 ||
        expertScore.cudaData == nullptr || !isfinite(routeScale)) {
        return false;
    }
    int tokens = (int)(logits.Count(0) / numExperts);
    if (expertIndex.Count(0) != (uint64_t)tokens * topk ||
        expertScore.Count(0) != (uint64_t)tokens * topk) {
        return false;
    }
    LoadedTritonKernel *kernel =
        LoadTritonKernel(cubinPath, kernelName, shared);
    if (kernel == nullptr) {
        return false;
    }

    void *logitsData = FastllmCudaPrepareInput(logits);
    void *biasData = FastllmCudaPrepareInput(gateBias);
    void *indexData = FastllmCudaPrepareOutput(expertIndex);
    void *scoreData = FastllmCudaPrepareOutput(expertScore);
    if (logitsData == nullptr || biasData == nullptr ||
        indexData == nullptr || scoreData == nullptr) {
        FastllmCudaFinishInput(logits, logitsData);
        FastllmCudaFinishInput(gateBias, biasData);
        FastllmCudaFinishOutput(expertIndex, indexData);
        FastllmCudaFinishOutput(expertScore, scoreData);
        return false;
    }

    CUdeviceptr logitsPtr = (CUdeviceptr)logitsData;
    CUdeviceptr biasPtr = (CUdeviceptr)biasData;
    CUdeviceptr indexPtr = (CUdeviceptr)indexData;
    CUdeviceptr scorePtr = (CUdeviceptr)scoreData;
    float routeScaleArg = routeScale;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *args[] = {
        &logitsPtr,
        &biasPtr,
        &indexPtr,
        &scorePtr,
        &routeScaleArg,
        &globalScratch,
        &profileScratch,
    };
    CUresult result = LaunchTritonKernel(
        kernel, (unsigned int)tokens, 1, 1,
        (unsigned int)(numWarps * 32), (unsigned int)shared, args);

    FastllmCudaFinishInput(logits, logitsData);
    FastllmCudaFinishInput(gateBias, biasData);
    FastllmCudaFinishOutput(expertIndex, indexData);
    FastllmCudaFinishOutput(expertScore, scoreData);
    return CheckCu(
        result, "cuLaunchKernel deepseek_v4_sqrtsoftplus_router_sm120");
}

static bool LaunchTritonMergeMOEFP8E4M3Table(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN, int groupBlockK, int groupSizeM,
    const fastllm::Data &input, fastllm::Data &output,
    const int32_t *indices, const float *scores,
    int batch, int topk, int hidden, int inter,
    const std::vector<int> &outputDims, TritonMoeFp8ExpertTable *table) {
    if (batch <= 0 || topk <= 0 || hidden <= 0 || inter <= 0 ||
        indices == nullptr || scores == nullptr ||
        cubinPaths == nullptr || kernelNames == nullptr || numWarps == nullptr || shared == nullptr ||
        routeBlockT <= 0 || maxExperts <= 0 || groupBlockM <= 0 || groupBlockN <= 0 || groupBlockK <= 0 ||
        groupSizeM < 0 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        (input.dataType != fastllm::DataType::FLOAT16 && input.dataType != fastllm::DataType::BFLOAT16) ||
        outputDims.empty() || table == nullptr) {
        return false;
    }
    if (table->experts > maxExperts) {
        return false;
    }
    if (table->gateBlockM != groupBlockK || table->downBlockM != groupBlockK ||
        table->gateBlockK != groupBlockN || table->downBlockK != groupBlockN) {
        return false;
    }
    if (table->packedBlock128) {
        if (!table->packedInited ||
            (table->fusedSeparateGateUp &&
             (table->fusedGateWeights == nullptr || table->fusedUpWeights == nullptr)) ||
            (!table->fusedSeparateGateUp && table->packedGateWeights == nullptr) ||
            table->packedDownWeights == nullptr) {
            return false;
        }
    } else if (table->fusedSeparateGateUp) {
        if (table->fusedGateWeights == nullptr || table->fusedUpWeights == nullptr ||
            table->fusedGateScales == nullptr || table->fusedUpScales == nullptr ||
            table->packedDownWeights == nullptr || table->packedDownScales == nullptr) {
            return false;
        }
    } else if (table->packedGateWeights == nullptr || table->packedGateScales == nullptr ||
               table->packedDownWeights == nullptr || table->packedDownScales == nullptr) {
        return false;
    }

    LoadedTritonKernel *kernels[kTritonMoeKernelCount];
    for (int i = 0; i < kTritonMoeKernelCount; i++) {
        if (numWarps[i] <= 0) {
            return false;
        }
    }
    if (!LoadTritonMoeKernelSet(cubinPaths, kernelNames, shared, kernels)) {
        return false;
    }

    int totalTasks = batch * topk;
    int outputElements = batch * hidden;
    if (totalTasks <= 0 || outputElements <= 0) {
        return false;
    }
    if (groupBlockN != groupBlockK || groupBlockN <= 0 || groupBlockK <= 0) {
        return false;
    }
    int32_t maxLaunchBlocksArg =
        (totalTasks + table->experts * (groupBlockM - 1) + groupBlockM - 1) / groupBlockM;
    int sortedTaskCapacity = maxLaunchBlocksArg * groupBlockM;
    int inputScaleCols = (hidden + groupBlockK - 1) / groupBlockK;
    int activationScaleCols = (inter + groupBlockN - 1) / groupBlockN;
    int inputQuantElements = batch * hidden;
    int inputScaleElements = batch * inputScaleCols;
    int gateUpElements = totalTasks * inter * 2;
    int gateUpBytes = gateUpElements * (int)sizeof(uint16_t);
    int downOutputBytes = totalTasks * hidden * (int)sizeof(uint16_t);
    int activationQuantElements = totalTasks * inter;
    int activationScaleElements = totalTasks * activationScaleCols;
    TritonMoeFp8Scratch *scratch = nullptr;
    if (!EnsureTritonMoeFp8Scratch(
            table->experts, sortedTaskCapacity, downOutputBytes,
            inputQuantElements, inputScaleElements,
            gateUpBytes,
            activationQuantElements, activationScaleElements, scratch)) {
        return false;
    }

    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = input.dataType;
    output.Resize(outputDims);
    output.Allocate(false);

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    auto finishPrepared = [&]() {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
    };
    if (inputData == nullptr || outputData == nullptr) {
        finishPrepared();
        return false;
    }

    CUdeviceptr inputPtr = (CUdeviceptr)inputData;
    CUdeviceptr indexPtr = (CUdeviceptr)indices;
    CUdeviceptr scorePtr = (CUdeviceptr)scores;
    CUdeviceptr expertCountsPtr = (CUdeviceptr)scratch->expertCounts;
    CUdeviceptr expertOffsetsPtr = (CUdeviceptr)scratch->expertOffsets;
    CUdeviceptr expertCursorsPtr = (CUdeviceptr)scratch->expertCursors;
    CUdeviceptr expertBlockOffsetsPtr = (CUdeviceptr)scratch->expertBlockOffsets;
    CUdeviceptr sortedTasksPtr = (CUdeviceptr)scratch->sortedTasks;
    CUdeviceptr blockExpertsPtr = (CUdeviceptr)scratch->blockExperts;
    CUdeviceptr blockStartsPtr = (CUdeviceptr)scratch->blockStarts;
    CUdeviceptr totalBlocksPtr = (CUdeviceptr)scratch->totalBlocks;
    CUdeviceptr downOutputPtr = (CUdeviceptr)scratch->downOutput;
    CUdeviceptr inputQuantPtr = (CUdeviceptr)scratch->inputQuant;
    CUdeviceptr inputScalePtr = (CUdeviceptr)scratch->inputScale;
    CUdeviceptr gateUpPtr = (CUdeviceptr)scratch->gateUp;
    CUdeviceptr activationQuantPtr = (CUdeviceptr)scratch->activationQuant;
    CUdeviceptr activationScalePtr = (CUdeviceptr)scratch->activationScale;
    CUdeviceptr gateWeightPtrs = (CUdeviceptr)(table->fusedSeparateGateUp ?
        table->fusedGateWeights : table->packedGateWeights);
    CUdeviceptr upWeightPtrs = (CUdeviceptr)table->fusedUpWeights;
    CUdeviceptr gateScalePtrs = table->packedBlock128 ? 0 :
        (CUdeviceptr)(table->fusedSeparateGateUp ?
            table->fusedGateScales : table->packedGateScales);
    CUdeviceptr upScalePtrs = (CUdeviceptr)table->fusedUpScales;
    CUdeviceptr downWeightPtrs = (CUdeviceptr)table->packedDownWeights;
    CUdeviceptr downScalePtrs = table->packedBlock128 ? 0 :
        (CUdeviceptr)table->packedDownScales;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    int32_t batchArg = batch;
    int32_t topkArg = topk;
    int32_t totalTasksArg = totalTasks;
    int32_t expertsArg = table->experts;
    int32_t hiddenArg = hidden;
    int32_t interArg = inter;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    auto launchTriton = [&](int kernelId, unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                            void **args) -> CUresult {
        return LaunchTritonKernel(
            kernels[kernelId],
            gridX, gridY, gridZ,
            (unsigned int)(numWarps[kernelId] * 32),
            (unsigned int)shared[kernelId],
            args, stream);
    };

    CUresult result = CUDA_SUCCESS;
    if (totalTasks <= routeBlockT) {
        void *initCountArgs[] = {
            &indexPtr,
            &expertCountsPtr,
            &expertOffsetsPtr,
            &expertCursorsPtr,
            &expertBlockOffsetsPtr,
            &totalBlocksPtr,
            &totalTasksArg,
            &expertsArg,
            &globalScratch,
            &profileScratch,
        };
        result = launchTriton(
            kTritonMoeInitCount,
            1, 1, 1,
            initCountArgs);
        if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_init_count")) {
            finishPrepared();
            return false;
        }
    } else {
        void *zeroArgs[] = {
            &expertCountsPtr,
            &expertOffsetsPtr,
            &expertCursorsPtr,
            &expertBlockOffsetsPtr,
            &totalBlocksPtr,
            &expertsArg,
            &globalScratch,
            &profileScratch,
        };
        result = launchTriton(
            kTritonMoeZeroRoute,
            1, 1, 1,
            zeroArgs);
        if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_zero_route")) {
            finishPrepared();
            return false;
        }

        void *countArgs[] = {
            &indexPtr,
            &expertCountsPtr,
            &totalTasksArg,
            &expertsArg,
            &globalScratch,
            &profileScratch,
        };
        result = launchTriton(
            kTritonMoeCount,
            (unsigned int)((totalTasks + routeBlockT - 1) / routeBlockT), 1, 1,
            countArgs);
        if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_count")) {
            finishPrepared();
            return false;
        }
    }

    void *prefixArgs[] = {
        &expertCountsPtr,
        &expertOffsetsPtr,
        &expertCursorsPtr,
        &expertBlockOffsetsPtr,
        &totalBlocksPtr,
        &expertsArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoePrefix,
        1, 1, 1,
        prefixArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_prefix")) {
        finishPrepared();
        return false;
    }

    void *fillSortedArgs[] = {
        &sortedTasksPtr,
        &expertOffsetsPtr,
        &totalTasksArg,
        &expertsArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeFillSorted,
        (unsigned int)((sortedTaskCapacity + routeBlockT - 1) / routeBlockT), 1, 1,
        fillSortedArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_fill_sorted")) {
        finishPrepared();
        return false;
    }

    int32_t launchBlocksArg = maxLaunchBlocksArg;
    const char *hostBlocksEnv = std::getenv("FASTLLM_CUDA_TRITON_MERGE_MOE_HOST_BLOCKS");
    bool useHostBlocks = hostBlocksEnv == nullptr || hostBlocksEnv[0] == '\0' ||
                         TritonEnvFlagEnabled("FASTLLM_CUDA_TRITON_MERGE_MOE_HOST_BLOCKS");
    // A synchronous device-to-host read invalidates CUDA stream capture.
    // During capture, launch the safe upper-bound grid and let device-side
    // routing metadata mask unused blocks.
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t captureState = cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus);
    bool isCapturing = captureState == cudaSuccess &&
                       captureStatus != cudaStreamCaptureStatusNone;
    if (captureState != cudaSuccess) {
        cudaGetLastError();
    }
    useHostBlocks = useHostBlocks && !isCapturing;
    if (useHostBlocks) {
        int32_t hostBlocks = 0;
        cudaError_t copyResult = cudaMemcpy(&hostBlocks, scratch->totalBlocks, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (copyResult == cudaSuccess && hostBlocks > 0 && hostBlocks <= maxLaunchBlocksArg) {
            launchBlocksArg = hostBlocks;
        } else {
            cudaGetLastError();
        }
    }

    unsigned int gateUpGridX = (unsigned int)launchBlocksArg;
    unsigned int gateUpGridY = (unsigned int)((inter * 2 + groupBlockN - 1) / groupBlockN);
    unsigned int downGridX = (unsigned int)launchBlocksArg;
    unsigned int downGridY = (unsigned int)((hidden + groupBlockN - 1) / groupBlockN);
    if (groupSizeM > 0) {
        gateUpGridX *= gateUpGridY;
        gateUpGridY = 1;
        downGridX *= downGridY;
        downGridY = 1;
    }

    CUdeviceptr nullPtr = 0;
    CUdeviceptr numTokensPostPaddedPtr =
        expertOffsetsPtr + (CUdeviceptr)((size_t)table->experts * sizeof(int32_t));
    int32_t emArg = launchBlocksArg * groupBlockM;

    void *scatterArgs[] = {
        &indexPtr,
        &expertOffsetsPtr,
        &expertCursorsPtr,
        &expertBlockOffsetsPtr,
        &sortedTasksPtr,
        &blockExpertsPtr,
        &blockStartsPtr,
        &totalTasksArg,
        &expertsArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeScatterBlocks,
        (unsigned int)((totalTasks + routeBlockT - 1) / routeBlockT), 1, 1,
        scatterArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_scatter_blocks")) {
        finishPrepared();
        return false;
    }

    void *quantInputArgs[] = {
        &inputPtr,
        &inputQuantPtr,
        &inputScalePtr,
        &batchArg,
        &hiddenArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeQuantInput,
        (unsigned int)batch, (unsigned int)inputScaleCols, 1,
        quantInputArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_quant_input")) {
        finishPrepared();
        return false;
    }

    if (table->fusedSeparateGateUp) {
        void *gateUpArgs[] = {
            &inputQuantPtr,
            &gateWeightPtrs,
            &upWeightPtrs,
            &gateUpPtr,
            &inputScalePtr,
            &gateScalePtrs,
            &upScalePtrs,
            &sortedTasksPtr,
            &blockExpertsPtr,
            &numTokensPostPaddedPtr,
            &emArg,
            &totalTasksArg,
            &globalScratch,
            &profileScratch,
        };
        result = launchTriton(
            kTritonMoeGateUpFused,
            gateUpGridX, gateUpGridY, 1,
            gateUpArgs);
    } else {
        void *gateUpArgs[] = {
            &inputQuantPtr,
            &gateWeightPtrs,
            &gateUpPtr,
            &nullPtr,
            &inputScalePtr,
            &gateScalePtrs,
            &scorePtr,
            &sortedTasksPtr,
            &blockExpertsPtr,
            &numTokensPostPaddedPtr,
            &emArg,
            &totalTasksArg,
            &globalScratch,
            &profileScratch,
        };
        result = launchTriton(
            kTritonMoeGateUp,
            gateUpGridX, gateUpGridY, 1,
            gateUpArgs);
    }
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_gateup")) {
        finishPrepared();
        return false;
    }

    void *swigluQuantArgs[] = {
        &gateUpPtr,
        &activationQuantPtr,
        &activationScalePtr,
        &totalTasksArg,
        &interArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeSwigluQuant,
        (unsigned int)totalTasks, (unsigned int)activationScaleCols, 1,
        swigluQuantArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_swiglu_quant")) {
        finishPrepared();
        return false;
    }

    void *downArgs[] = {
        &activationQuantPtr,
        &downWeightPtrs,
        &downOutputPtr,
        &nullPtr,
        &activationScalePtr,
        &downScalePtrs,
        &scorePtr,
        &sortedTasksPtr,
        &blockExpertsPtr,
        &numTokensPostPaddedPtr,
        &emArg,
        &totalTasksArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeDown,
        downGridX, downGridY, 1,
        downArgs);
    if (!CheckCu(result, "cuLaunchKernel merge_moe_fp8_down")) {
        finishPrepared();
        return false;
    }

    void *sumOutputArgs[] = {
        &downOutputPtr,
        &outputPtr,
        &batchArg,
        &topkArg,
        &hiddenArg,
        &globalScratch,
        &profileScratch,
    };
    result = launchTriton(
        kTritonMoeSumOutput,
        (unsigned int)((outputElements + routeBlockT - 1) / routeBlockT), 1, 1,
        sumOutputArgs);

    finishPrepared();
    return CheckCu(result, "cuLaunchKernel merge_moe_fp8_sum_output");
}

extern "C" bool FastllmCudaTritonChunkGdnPostConv(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared, int blockT,
    const fastllm::Data &qInput, const fastllm::Data &kInput,
    const fastllm::Data &qkvInput, const fastllm::Data &gInput,
    const fastllm::Data &betaInput,
    int batch, int seqLen, int keyHeads, int valueHeads,
    int kDim, int vDim, float qScale,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &beta,
    fastllm::Data &kBeta, fastllm::Data &vBeta) {
    int chunks = (seqLen + 63) / 64;
    if (cubinPath == nullptr || kernelName == nullptr ||
        numWarps <= 0 || shared < 0 || blockT != 16 ||
        batch <= 0 || seqLen <= 0 || chunks <= 0 ||
        keyHeads <= 0 || valueHeads < keyHeads ||
        valueHeads % keyHeads != 0 ||
        kDim != 128 || vDim != 128 ||
        qInput.dataType != fastllm::DataType::FLOAT16 ||
        kInput.dataType != fastllm::DataType::FLOAT16 ||
        qkvInput.dataType != fastllm::DataType::FLOAT16 ||
        gInput.dataType != fastllm::DataType::FLOAT16 ||
        betaInput.dataType != fastllm::DataType::FLOAT16 ||
        qInput.cudaData == nullptr || kInput.cudaData == nullptr ||
        qkvInput.cudaData == nullptr || gInput.cudaData == nullptr ||
        betaInput.cudaData == nullptr ||
        qInput.Count(0) !=
            (uint64_t)batch * seqLen * keyHeads * kDim ||
        kInput.Count(0) !=
            (uint64_t)batch * seqLen * keyHeads * kDim ||
        qkvInput.Count(0) !=
            (uint64_t)batch * seqLen *
            (keyHeads * kDim * 2 + valueHeads * vDim) ||
        gInput.Count(0) !=
            (uint64_t)batch * seqLen * valueHeads ||
        betaInput.Count(0) !=
            (uint64_t)batch * seqLen * valueHeads) {
        return false;
    }

    LoadedTritonKernel *kernel =
        LoadTritonKernel(cubinPath, kernelName, shared);
    if (kernel == nullptr) {
        return false;
    }

    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = qkvInput.dataDevice;
        output.dataDeviceIds = qkvInput.dataDeviceIds;
        output.Resize(dims);
        output.Allocate(false);
        return output.cudaData != nullptr;
    };
    if (!prepareOutput(q, {batch, valueHeads, chunks * 64, kDim}) ||
        !prepareOutput(k, {batch, valueHeads, chunks * 64, kDim}) ||
        !prepareOutput(v, {batch, valueHeads, chunks * 64, vDim}) ||
        !prepareOutput(g, {batch, valueHeads, chunks * 64}) ||
        !prepareOutput(beta, {batch, valueHeads, chunks * 64}) ||
        !prepareOutput(kBeta, {batch, valueHeads, chunks * 64, kDim}) ||
        !prepareOutput(vBeta, {batch, valueHeads, chunks * 64, vDim})) {
        return false;
    }

    void *qInputData = FastllmCudaPrepareInput(qInput);
    void *kInputData = FastllmCudaPrepareInput(kInput);
    void *qkvInputData = FastllmCudaPrepareInput(qkvInput);
    void *gInputData = FastllmCudaPrepareInput(gInput);
    void *betaInputData = FastllmCudaPrepareInput(betaInput);
    void *qData = FastllmCudaPrepareOutput(q);
    void *kData = FastllmCudaPrepareOutput(k);
    void *vData = FastllmCudaPrepareOutput(v);
    void *gData = FastllmCudaPrepareOutput(g);
    void *betaData = FastllmCudaPrepareOutput(beta);
    void *kBetaData = FastllmCudaPrepareOutput(kBeta);
    void *vBetaData = FastllmCudaPrepareOutput(vBeta);
    auto finishPrepared = [&]() {
        FastllmCudaFinishInput(qInput, qInputData);
        FastllmCudaFinishInput(kInput, kInputData);
        FastllmCudaFinishInput(qkvInput, qkvInputData);
        FastllmCudaFinishInput(gInput, gInputData);
        FastllmCudaFinishInput(betaInput, betaInputData);
        FastllmCudaFinishOutput(q, qData);
        FastllmCudaFinishOutput(k, kData);
        FastllmCudaFinishOutput(v, vData);
        FastllmCudaFinishOutput(g, gData);
        FastllmCudaFinishOutput(beta, betaData);
        FastllmCudaFinishOutput(kBeta, kBetaData);
        FastllmCudaFinishOutput(vBeta, vBetaData);
    };
    if (qInputData == nullptr || kInputData == nullptr ||
        qkvInputData == nullptr || gInputData == nullptr ||
        betaInputData == nullptr ||
        qData == nullptr || kData == nullptr ||
        vData == nullptr || gData == nullptr || betaData == nullptr ||
        kBetaData == nullptr || vBetaData == nullptr) {
        finishPrepared();
        return false;
    }

    CUdeviceptr qInputPtr = (CUdeviceptr)qInputData;
    CUdeviceptr kInputPtr = (CUdeviceptr)kInputData;
    CUdeviceptr qkvInputPtr = (CUdeviceptr)qkvInputData;
    CUdeviceptr gInputPtr = (CUdeviceptr)gInputData;
    CUdeviceptr betaInputPtr = (CUdeviceptr)betaInputData;
    CUdeviceptr qPtr = (CUdeviceptr)qData;
    CUdeviceptr kPtr = (CUdeviceptr)kData;
    CUdeviceptr vPtr = (CUdeviceptr)vData;
    CUdeviceptr gPtr = (CUdeviceptr)gData;
    CUdeviceptr betaPtr = (CUdeviceptr)betaData;
    CUdeviceptr kBetaPtr = (CUdeviceptr)kBetaData;
    CUdeviceptr vBetaPtr = (CUdeviceptr)vBetaData;
    int32_t seqLenArg = seqLen;
    int32_t chunksArg = chunks;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *args[] = {
        &qInputPtr, &kInputPtr, &qkvInputPtr, &gInputPtr, &betaInputPtr,
        &qPtr, &kPtr, &vPtr, &gPtr, &betaPtr, &kBetaPtr, &vBetaPtr,
        &seqLenArg, &chunksArg, &qScale,
        &globalScratch, &profileScratch,
    };
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    CUresult result = LaunchTritonKernel(
        kernel, (unsigned int)(chunks * 64 / blockT),
        (unsigned int)batch, (unsigned int)(keyHeads + valueHeads),
        (unsigned int)(numWarps * 32), (unsigned int)shared, args, stream);
    finishPrepared();
    return CheckCu(result, "cuLaunchKernel chunk_gdn_postconv");
}

namespace {
struct TritonChunkGdnScaleScratch {
    void *rowScale = nullptr;
    void *stateScale = nullptr;
    size_t rowScaleBytes = 0;
    size_t stateScaleBytes = 0;
    int batchHeads = 0;
    int chunks = 0;
    int chunkSize = 0;
    bool valid = false;
};

static std::map<int, TritonChunkGdnScaleScratch> &
TritonChunkGdnScaleScratches() {
    thread_local static std::map<int, TritonChunkGdnScaleScratch> scratches;
    return scratches;
}

static TritonChunkGdnScaleScratch *FindTritonChunkGdnScaleScratch() {
    auto &scratches = TritonChunkGdnScaleScratches();
    auto it = scratches.find(FastllmCudaGetDevice());
    return it == scratches.end() ? nullptr : &it->second;
}

static bool EnsureTritonChunkGdnScaleScratch(
    size_t rowScaleBytes, size_t stateScaleBytes,
    int batchHeads, int chunks, int chunkSize,
    TritonChunkGdnScaleScratch *&scratch) {
    auto &cached =
        TritonChunkGdnScaleScratches()[FastllmCudaGetDevice()];
    cached.valid = false;
    if (cached.rowScaleBytes < rowScaleBytes ||
        cached.stateScaleBytes < stateScaleBytes) {
        if (cached.rowScale != nullptr || cached.stateScale != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
        }
        if (cached.rowScale != nullptr) {
            FastllmCudaFree(cached.rowScale);
        }
        if (cached.stateScale != nullptr) {
            FastllmCudaFree(cached.stateScale);
        }
        cached.rowScale = FastllmCudaMalloc(rowScaleBytes);
        cached.stateScale = FastllmCudaMalloc(stateScaleBytes);
        if (cached.rowScale == nullptr || cached.stateScale == nullptr) {
            if (cached.rowScale != nullptr) {
                FastllmCudaFree(cached.rowScale);
            }
            if (cached.stateScale != nullptr) {
                FastllmCudaFree(cached.stateScale);
            }
            cached = TritonChunkGdnScaleScratch();
            return false;
        }
        cached.rowScaleBytes = rowScaleBytes;
        cached.stateScaleBytes = stateScaleBytes;
    }
    cached.batchHeads = batchHeads;
    cached.chunks = chunks;
    cached.chunkSize = chunkSize;
    scratch = &cached;
    return true;
}
}

extern "C" bool FastllmCudaTritonChunkGdnRecompute(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared,
    bool precomputeScale, bool internalExp, int blockD,
    const fastllm::Data &attn, const fastllm::Data &vBeta,
    const fastllm::Data &kBeta, const fastllm::Data &gExp,
    const fastllm::Data &g,
    fastllm::Data &vOutput, fastllm::Data &kOutput) {
    if (cubinPath == nullptr || kernelName == nullptr ||
        numWarps <= 0 || shared < 0 || blockD != 64 ||
        attn.dataType != fastllm::DataType::FLOAT16 ||
        vBeta.dataType != fastllm::DataType::FLOAT16 ||
        kBeta.dataType != fastllm::DataType::FLOAT16 ||
        (!internalExp &&
         gExp.dataType != fastllm::DataType::FLOAT16) ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        attn.cudaData == nullptr || vBeta.cudaData == nullptr ||
        kBeta.cudaData == nullptr ||
        (!internalExp && gExp.cudaData == nullptr) ||
        g.cudaData == nullptr ||
        attn.dims.size() != 5 ||
        vBeta.dims.size() != 5 ||
        kBeta.dims.size() != 5 ||
        (!internalExp && gExp.dims.size() != 4) ||
        g.dims.size() != 4) {
        return false;
    }
    int batch = attn.dims[0];
    int heads = attn.dims[1];
    int chunks = attn.dims[2];
    int chunkSize = attn.dims[3];
    constexpr int kDim = 128;
    constexpr int vDim = 128;
    if (batch <= 0 || heads <= 0 || chunks <= 0 ||
        chunkSize != 64 || attn.dims[4] != chunkSize ||
        vBeta.dims != std::vector<int>(
            {batch, heads, chunks, chunkSize, vDim}) ||
        kBeta.dims != std::vector<int>(
            {batch, heads, chunks, chunkSize, kDim}) ||
        (!internalExp &&
         gExp.dims != std::vector<int>(
             {batch, heads, chunks, chunkSize})) ||
        g.dims != std::vector<int>(
            {batch, heads, chunks, chunkSize})) {
        return false;
    }

    LoadedTritonKernel *kernel =
        LoadTritonKernel(cubinPath, kernelName, shared);
    if (kernel == nullptr) {
        return false;
    }

    auto prepareOutput = [&](fastllm::Data &output,
                             const std::vector<int> &dims) {
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = attn.dataDevice;
        output.dataDeviceIds = attn.dataDeviceIds;
        output.Resize(dims);
        output.Allocate(false);
        return output.cudaData != nullptr;
    };
    if (!prepareOutput(vOutput, vBeta.dims) ||
        !prepareOutput(kOutput, kBeta.dims)) {
        return false;
    }

    int batchHeads = batch * heads;
    TritonChunkGdnScaleScratch *scaleScratch =
        FindTritonChunkGdnScaleScratch();
    if (precomputeScale) {
        size_t rowScaleBytes =
            (size_t)batchHeads * chunks * chunkSize * sizeof(float);
        size_t stateScaleBytes =
            (size_t)batchHeads * chunks * sizeof(float);
        if (!EnsureTritonChunkGdnScaleScratch(
                rowScaleBytes, stateScaleBytes,
                batchHeads, chunks, chunkSize, scaleScratch)) {
            return false;
        }
    } else if (scaleScratch != nullptr) {
        scaleScratch->valid = false;
    }

    void *attnData = FastllmCudaPrepareInput(attn);
    void *vBetaData = FastllmCudaPrepareInput(vBeta);
    void *kBetaData = FastllmCudaPrepareInput(kBeta);
    void *gExpData =
        internalExp ? nullptr : FastllmCudaPrepareInput(gExp);
    void *gData = FastllmCudaPrepareInput(g);
    void *vOutputData = FastllmCudaPrepareOutput(vOutput);
    void *kOutputData = FastllmCudaPrepareOutput(kOutput);
    auto finishPrepared = [&]() {
        FastllmCudaFinishInput(attn, attnData);
        FastllmCudaFinishInput(vBeta, vBetaData);
        FastllmCudaFinishInput(kBeta, kBetaData);
        if (!internalExp) {
            FastllmCudaFinishInput(gExp, gExpData);
        }
        FastllmCudaFinishInput(g, gData);
        FastllmCudaFinishOutput(vOutput, vOutputData);
        FastllmCudaFinishOutput(kOutput, kOutputData);
    };
    if (attnData == nullptr || vBetaData == nullptr ||
        kBetaData == nullptr ||
        (!internalExp && gExpData == nullptr) ||
        gData == nullptr ||
        vOutputData == nullptr || kOutputData == nullptr) {
        finishPrepared();
        return false;
    }

    CUdeviceptr attnPtr = (CUdeviceptr)attnData;
    CUdeviceptr vBetaPtr = (CUdeviceptr)vBetaData;
    CUdeviceptr kBetaPtr = (CUdeviceptr)kBetaData;
    CUdeviceptr gExpPtr = (CUdeviceptr)gExpData;
    CUdeviceptr gPtr = (CUdeviceptr)gData;
    CUdeviceptr vOutputPtr = (CUdeviceptr)vOutputData;
    CUdeviceptr kOutputPtr = (CUdeviceptr)kOutputData;
    CUdeviceptr rowScalePtr = precomputeScale
        ? (CUdeviceptr)scaleScratch->rowScale : 0;
    CUdeviceptr stateScalePtr = precomputeScale
        ? (CUdeviceptr)scaleScratch->stateScale : 0;
    int32_t chunksArg = chunks;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *args[] = {
        &attnPtr, &vBetaPtr, &kBetaPtr, &gExpPtr, &gPtr,
        &vOutputPtr, &kOutputPtr, &rowScalePtr, &stateScalePtr,
        &chunksArg,
        &globalScratch, &profileScratch,
    };
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    CUresult result = LaunchTritonKernel(
        kernel, (unsigned int)chunks,
        (unsigned int)(batch * heads), 1,
        (unsigned int)(numWarps * 32),
        (unsigned int)shared, args, stream);
    if (scaleScratch != nullptr) {
        scaleScratch->valid =
            precomputeScale && result == CUDA_SUCCESS;
    }
    finishPrepared();
    return CheckCu(result, "cuLaunchKernel chunk_gdn_recompute");
}

namespace {
struct TritonChunkGdnPrefillScratch {
    void *h = nullptr;
    void *vNew = nullptr;
    // H computes the next recurrent state here.  The caller commits it only
    // after O has also been launched, so a failed O launch can safely fall
    // back without advancing the input state twice.
    void *nextState = nullptr;
    size_t hBytes = 0;
    size_t vNewBytes = 0;
    size_t stateBytes = 0;
};

static bool EnsureTritonChunkGdnPrefillScratch(
    size_t hBytes, size_t vNewBytes, size_t stateBytes,
    TritonChunkGdnPrefillScratch *&scratch) {
    thread_local static std::map<int, TritonChunkGdnPrefillScratch> scratches;
    int device = FastllmCudaGetDevice();
    TritonChunkGdnPrefillScratch &cached = scratches[device];
    if (cached.hBytes < hBytes || cached.vNewBytes < vNewBytes ||
        cached.stateBytes < stateBytes) {
        if (cached.h != nullptr || cached.vNew != nullptr ||
            cached.nextState != nullptr) {
            FastllmCudaSyncCurrentThreadStream();
        }
        if (cached.h != nullptr) {
            FastllmCudaFree(cached.h);
            cached.h = nullptr;
            cached.hBytes = 0;
        }
        if (cached.vNew != nullptr) {
            FastllmCudaFree(cached.vNew);
            cached.vNew = nullptr;
            cached.vNewBytes = 0;
        }
        if (cached.nextState != nullptr) {
            FastllmCudaFree(cached.nextState);
            cached.nextState = nullptr;
            cached.stateBytes = 0;
        }
        cached.h = FastllmCudaMalloc(hBytes);
        cached.vNew = FastllmCudaMalloc(vNewBytes);
        cached.nextState = FastllmCudaMalloc(stateBytes);
        if (cached.h == nullptr || cached.vNew == nullptr ||
            cached.nextState == nullptr) {
            if (cached.h != nullptr) {
                FastllmCudaFree(cached.h);
            }
            if (cached.vNew != nullptr) {
                FastllmCudaFree(cached.vNew);
            }
            if (cached.nextState != nullptr) {
                FastllmCudaFree(cached.nextState);
            }
            cached = TritonChunkGdnPrefillScratch();
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

extern "C" bool FastllmCudaTritonChunkGatedDeltaRulePrefill(
    const char *hCubinPath, const char *hKernelName, int hNumWarps, int hShared,
    const char *oCubinPath, const char *oKernelName, int oNumWarps, int oShared,
    const char *oFusedDecayCubinPath,
    const char *oFusedDecayKernelName,
    int oFusedDecayNumWarps, int oFusedDecayShared,
    const char *hPrecomputedScaleCubinPath,
    const char *hPrecomputedScaleKernelName,
    int hPrecomputedScaleNumWarps, int hPrecomputedScaleShared,
    bool precomputeScale, bool fuseDecayMask,
    int chunks, int chunkSize, int kDim, int vDim,
    int hBlockV, int oBlockV,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &decayMask, fastllm::Data &kCumdecay,
    fastllm::Data &lastRecurrentState, fastllm::Data &coreAttnOut) {
    if (hCubinPath == nullptr || hKernelName == nullptr ||
        oCubinPath == nullptr || oKernelName == nullptr ||
        oFusedDecayCubinPath == nullptr ||
        oFusedDecayKernelName == nullptr ||
        hPrecomputedScaleCubinPath == nullptr ||
        hPrecomputedScaleKernelName == nullptr ||
        hNumWarps <= 0 || oNumWarps <= 0 ||
        oFusedDecayNumWarps <= 0 ||
        hPrecomputedScaleNumWarps <= 0 ||
        chunks <= 0 || chunkSize != 64 || kDim != 128 || vDim != 128 ||
        (hBlockV != 32 && hBlockV != 64) ||
        (oBlockV != 32 && oBlockV != 64) ||
        q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        attn.dataType != fastllm::DataType::FLOAT16 ||
        decayMask.dataType != fastllm::DataType::FLOAT16 ||
        kCumdecay.dataType != fastllm::DataType::FLOAT16 ||
        lastRecurrentState.dataType != fastllm::DataType::FLOAT16 ||
        q.cudaData == nullptr || k.cudaData == nullptr || v.cudaData == nullptr ||
        g.cudaData == nullptr || attn.cudaData == nullptr ||
        decayMask.cudaData == nullptr ||
        kCumdecay.cudaData == nullptr || lastRecurrentState.cudaData == nullptr ||
        q.dims.size() != 5 || k.dims != q.dims ||
        q.dims[2] != chunks || q.dims[3] != chunkSize ||
        q.dims[4] != kDim || v.dims.size() != 5 ||
        v.dims[0] != q.dims[0] || v.dims[1] != q.dims[1] ||
        v.dims[2] != chunks || v.dims[3] != chunkSize ||
        v.dims[4] != vDim ||
        g.dims != std::vector<int>({q.dims[0], q.dims[1], chunks, chunkSize}) ||
        attn.dims != std::vector<int>({q.dims[0], q.dims[1], chunks,
                                       chunkSize, chunkSize}) ||
        decayMask.dims != attn.dims ||
        kCumdecay.dims != q.dims ||
        lastRecurrentState.dims !=
            std::vector<int>({q.dims[0], q.dims[1], kDim, vDim})) {
        return false;
    }

    LoadedTritonKernel *hKernel =
        LoadTritonKernel(hCubinPath, hKernelName, hShared);
    LoadedTritonKernel *oKernel =
        LoadTritonKernel(oCubinPath, oKernelName, oShared);
    LoadedTritonKernel *oFusedDecayKernel =
        LoadTritonKernel(
            oFusedDecayCubinPath, oFusedDecayKernelName,
            oFusedDecayShared);
    LoadedTritonKernel *hPrecomputedScaleKernel =
        LoadTritonKernel(
            hPrecomputedScaleCubinPath, hPrecomputedScaleKernelName,
            hPrecomputedScaleShared);
    if (hKernel == nullptr || oKernel == nullptr ||
        oFusedDecayKernel == nullptr ||
        hPrecomputedScaleKernel == nullptr) {
        return false;
    }

    int batchHeads = q.dims[0] * q.dims[1];
    TritonChunkGdnScaleScratch *scaleScratch =
        FindTritonChunkGdnScaleScratch();
    if (precomputeScale &&
        (scaleScratch == nullptr || !scaleScratch->valid ||
         scaleScratch->batchHeads != batchHeads ||
         scaleScratch->chunks != chunks ||
         scaleScratch->chunkSize != chunkSize)) {
        return false;
    }
    if (precomputeScale) {
        // Consume the handoff before any later allocation or launch can fail,
        // so a fallback cannot leave stale scales valid for the next layer.
        scaleScratch->valid = false;
    }
    size_t hBytes =
        (size_t)batchHeads * chunks * kDim * vDim * sizeof(half);
    size_t vNewBytes =
        (size_t)batchHeads * chunks * chunkSize * vDim * sizeof(half);
    size_t stateBytes =
        (size_t)batchHeads * kDim * vDim * sizeof(half);
    TritonChunkGdnPrefillScratch *scratch = nullptr;
    if (!EnsureTritonChunkGdnPrefillScratch(
            hBytes, vNewBytes, stateBytes, scratch)) {
        return false;
    }

    coreAttnOut.dataType = fastllm::DataType::FLOAT16;
    coreAttnOut.dataDevice = v.dataDevice;
    coreAttnOut.dataDeviceIds = v.dataDeviceIds;
    coreAttnOut.Resize({q.dims[0], q.dims[1], chunks, chunkSize, vDim});
    coreAttnOut.Allocate(false);
    if (coreAttnOut.cudaData == nullptr) {
        return false;
    }

    void *qData = FastllmCudaPrepareInput(q);
    void *kData = FastllmCudaPrepareInput(k);
    void *vData = FastllmCudaPrepareInput(v);
    void *gData = FastllmCudaPrepareInput(g);
    void *attnData = FastllmCudaPrepareInput(attn);
    void *decayMaskData = FastllmCudaPrepareInput(decayMask);
    void *kCumdecayData = FastllmCudaPrepareInput(kCumdecay);
    void *stateData = FastllmCudaPrepareInput(lastRecurrentState);
    void *outputData = FastllmCudaPrepareOutput(coreAttnOut);
    if (qData == nullptr || kData == nullptr || vData == nullptr ||
        gData == nullptr || attnData == nullptr ||
        decayMaskData == nullptr || kCumdecayData == nullptr ||
        stateData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(q, qData);
        FastllmCudaFinishInput(k, kData);
        FastllmCudaFinishInput(v, vData);
        FastllmCudaFinishInput(g, gData);
        FastllmCudaFinishInput(attn, attnData);
        FastllmCudaFinishInput(decayMask, decayMaskData);
        FastllmCudaFinishInput(kCumdecay, kCumdecayData);
        FastllmCudaFinishInput(lastRecurrentState, stateData);
        FastllmCudaFinishOutput(coreAttnOut, outputData);
        return false;
    }

    CUdeviceptr qPtr = (CUdeviceptr)qData;
    CUdeviceptr kPtr = (CUdeviceptr)kData;
    CUdeviceptr vPtr = (CUdeviceptr)vData;
    CUdeviceptr gPtr = (CUdeviceptr)gData;
    CUdeviceptr attnPtr = (CUdeviceptr)attnData;
    CUdeviceptr decayMaskPtr = (CUdeviceptr)decayMaskData;
    CUdeviceptr kCumdecayPtr = (CUdeviceptr)kCumdecayData;
    CUdeviceptr statePtr = (CUdeviceptr)stateData;
    CUdeviceptr nextStatePtr = (CUdeviceptr)scratch->nextState;
    CUdeviceptr hPtr = (CUdeviceptr)scratch->h;
    CUdeviceptr vNewPtr = (CUdeviceptr)scratch->vNew;
    CUdeviceptr rowScalePtr = precomputeScale
        ? (CUdeviceptr)scaleScratch->rowScale : 0;
    CUdeviceptr stateScalePtr = precomputeScale
        ? (CUdeviceptr)scaleScratch->stateScale : 0;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *hArgs[] = {
        &kPtr, &vPtr, &gPtr, &kCumdecayPtr, &statePtr, &nextStatePtr,
        &hPtr, &vNewPtr, &rowScalePtr, &stateScalePtr,
        &globalScratch, &profileScratch,
    };
    void *oArgs[] = {
        &qPtr, &gPtr, &attnPtr, &decayMaskPtr,
        &hPtr, &vNewPtr, &outputPtr,
        &globalScratch, &profileScratch,
    };
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    unsigned int hVBlocks =
        (unsigned int)((vDim + hBlockV - 1) / hBlockV);
    unsigned int oVBlocks =
        (unsigned int)((vDim + oBlockV - 1) / oBlockV);
    LoadedTritonKernel *selectedHKernel =
        precomputeScale ? hPrecomputedScaleKernel : hKernel;
    int selectedHNumWarps =
        precomputeScale ? hPrecomputedScaleNumWarps : hNumWarps;
    int selectedHShared =
        precomputeScale ? hPrecomputedScaleShared : hShared;
    LoadedTritonKernel *selectedOKernel =
        fuseDecayMask ? oFusedDecayKernel : oKernel;
    int selectedONumWarps =
        fuseDecayMask ? oFusedDecayNumWarps : oNumWarps;
    int selectedOShared =
        fuseDecayMask ? oFusedDecayShared : oShared;
    CUresult hResult = LaunchTritonKernel(
        selectedHKernel, hVBlocks, (unsigned int)batchHeads, 1,
        (unsigned int)(selectedHNumWarps * 32),
        (unsigned int)selectedHShared, hArgs, stream);
    CUresult oResult = hResult == CUDA_SUCCESS
        ? LaunchTritonKernel(
              selectedOKernel, oVBlocks, (unsigned int)chunks,
              (unsigned int)batchHeads,
              (unsigned int)(selectedONumWarps * 32),
              (unsigned int)selectedOShared, oArgs, stream)
        : hResult;
    CUresult stateCommitResult =
        hResult == CUDA_SUCCESS && oResult == CUDA_SUCCESS
            ? cuMemcpyDtoDAsync(statePtr, nextStatePtr, stateBytes, stream)
            : oResult;

    FastllmCudaFinishInput(q, qData);
    FastllmCudaFinishInput(k, kData);
    FastllmCudaFinishInput(v, vData);
    FastllmCudaFinishInput(g, gData);
    FastllmCudaFinishInput(attn, attnData);
    FastllmCudaFinishInput(decayMask, decayMaskData);
    FastllmCudaFinishInput(kCumdecay, kCumdecayData);
    FastllmCudaFinishInput(lastRecurrentState, stateData);
    FastllmCudaFinishOutput(coreAttnOut, outputData);
    return CheckCu(hResult, "cuLaunchKernel chunk_gdn_prefill_h") &&
           CheckCu(oResult, "cuLaunchKernel chunk_gdn_prefill_o") &&
           CheckCu(stateCommitResult,
                   "cuMemcpyDtoDAsync chunk_gdn_prefill_state_commit");
}

extern "C" bool FastllmCudaTritonChunkGatedDeltaRuleVarlenPrefill(
    const char *hCubinPath, const char *hKernelName,
    int hNumWarps, int hShared,
    const char *oCubinPath, const char *oKernelName,
    int oNumWarps, int oShared,
    const char *oFusedDecayCubinPath,
    const char *oFusedDecayKernelName,
    int oFusedDecayNumWarps, int oFusedDecayShared,
    const char *hPrecomputedScaleCubinPath,
    const char *hPrecomputedScaleKernelName,
    int hPrecomputedScaleNumWarps, int hPrecomputedScaleShared,
    const char *oDirectQkCubinPath,
    const char *oDirectQkKernelName,
    int oDirectQkNumWarps, int oDirectQkShared,
    bool precomputeScale, bool fuseDecayMask, bool directOutputQk,
    int maxChunks, int chunkSize, int kDim, int vDim,
    int hBlockV, int oBlockV, const std::vector<int> &seqLens,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &decayMask, fastllm::Data &kCumdecay,
    fastllm::Data &lastRecurrentState,
    fastllm::Data &coreAttnOut) {
    if (hCubinPath == nullptr || hKernelName == nullptr ||
        oCubinPath == nullptr || oKernelName == nullptr ||
        oFusedDecayCubinPath == nullptr ||
        oFusedDecayKernelName == nullptr ||
        hPrecomputedScaleCubinPath == nullptr ||
        hPrecomputedScaleKernelName == nullptr ||
        oDirectQkCubinPath == nullptr ||
        oDirectQkKernelName == nullptr ||
        hNumWarps <= 0 || oNumWarps <= 0 ||
        oFusedDecayNumWarps <= 0 || hPrecomputedScaleNumWarps <= 0 ||
        oDirectQkNumWarps <= 0 ||
        maxChunks <= 0 ||
        chunkSize != 64 || kDim != 128 || vDim != 128 ||
        (hBlockV != 32 && hBlockV != 64) ||
        (oBlockV != 32 && oBlockV != 64) || seqLens.empty() ||
        q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        (!directOutputQk &&
         attn.dataType != fastllm::DataType::FLOAT16) ||
        ((fuseDecayMask || directOutputQk) &&
         decayMask.dataType != fastllm::DataType::FLOAT16) ||
        kCumdecay.dataType != fastllm::DataType::FLOAT16 ||
        lastRecurrentState.dataType != fastllm::DataType::FLOAT16 ||
        q.cudaData == nullptr || k.cudaData == nullptr ||
        v.cudaData == nullptr || g.cudaData == nullptr ||
        (!directOutputQk && attn.cudaData == nullptr) ||
        ((fuseDecayMask || directOutputQk) &&
         decayMask.cudaData == nullptr) ||
        kCumdecay.cudaData == nullptr ||
        lastRecurrentState.cudaData == nullptr ||
        q.dims.size() != 5 || q.dims[0] != 1 || k.dims != q.dims) {
        return false;
    }

    int batch = (int)seqLens.size();
    int keyHeads = q.dims[1];
    int heads = v.dims.size() == 5 ? v.dims[1] : 0;
    int totalChunks = q.dims[2];
    if (keyHeads <= 0 || heads < keyHeads || heads % keyHeads != 0 ||
        totalChunks <= 0 || q.dims[3] != chunkSize ||
        q.dims[4] != kDim ||
        v.dims != std::vector<int>({1, heads, totalChunks,
                                     chunkSize, vDim}) ||
        g.dims != std::vector<int>({1, heads, totalChunks, chunkSize}) ||
        (!directOutputQk && attn.dims !=
            std::vector<int>({1, keyHeads, totalChunks,
                              chunkSize, chunkSize})) ||
        ((fuseDecayMask || directOutputQk) && decayMask.dims !=
            std::vector<int>({1, heads, totalChunks,
                              chunkSize, chunkSize})) ||
        kCumdecay.dims != std::vector<int>({1, heads, totalChunks,
                                            chunkSize, kDim}) ||
        lastRecurrentState.dims !=
            std::vector<int>({batch, heads, kDim, vDim})) {
        return false;
    }
    FastllmCudaRaggedGdnMetadataView metadata;
    if (!FastllmCudaGetRaggedGdnMetadata(
            seqLens, chunkSize, metadata) ||
        metadata.totalChunks != totalChunks ||
        metadata.maxChunks > maxChunks) {
        return false;
    }

    LoadedTritonKernel *hKernel =
        LoadTritonKernel(hCubinPath, hKernelName, hShared);
    LoadedTritonKernel *oKernel =
        LoadTritonKernel(oCubinPath, oKernelName, oShared);
    LoadedTritonKernel *oFusedDecayKernel = LoadTritonKernel(
        oFusedDecayCubinPath, oFusedDecayKernelName,
        oFusedDecayShared);
    LoadedTritonKernel *hPrecomputedScaleKernel = LoadTritonKernel(
        hPrecomputedScaleCubinPath, hPrecomputedScaleKernelName,
        hPrecomputedScaleShared);
    LoadedTritonKernel *oDirectQkKernel = directOutputQk
        ? LoadTritonKernel(
              oDirectQkCubinPath, oDirectQkKernelName,
              oDirectQkShared)
        : nullptr;
    if (hKernel == nullptr || oKernel == nullptr ||
        oFusedDecayKernel == nullptr || hPrecomputedScaleKernel == nullptr ||
        (directOutputQk && oDirectQkKernel == nullptr)) {
        return false;
    }
    TritonChunkGdnScaleScratch *scaleScratch =
        FindTritonChunkGdnScaleScratch();
    bool usePrecomputedScale =
        precomputeScale && scaleScratch != nullptr && scaleScratch->valid &&
        scaleScratch->batchHeads == heads &&
        scaleScratch->chunks == totalChunks &&
        scaleScratch->chunkSize == chunkSize;
    if (usePrecomputedScale) {
        // The recompute kernel produced scales in the same packed
        // [head, global_chunk] order. Consume the one-layer handoff here.
        scaleScratch->valid = false;
    }

    size_t hBytes =
        (size_t)heads * totalChunks * kDim * vDim * sizeof(half);
    size_t vNewBytes =
        (size_t)heads * totalChunks * chunkSize * vDim * sizeof(half);
    size_t stateBytes =
        (size_t)batch * heads * kDim * vDim * sizeof(half);
    TritonChunkGdnPrefillScratch *scratch = nullptr;
    if (!EnsureTritonChunkGdnPrefillScratch(
            hBytes, vNewBytes, stateBytes, scratch)) {
        return false;
    }

    coreAttnOut.dataType = fastllm::DataType::FLOAT16;
    coreAttnOut.dataDevice = v.dataDevice;
    coreAttnOut.dataDeviceIds = v.dataDeviceIds;
    coreAttnOut.Resize({1, metadata.totalTokens, heads, vDim});
    coreAttnOut.Allocate(false);
    if (coreAttnOut.cudaData == nullptr) {
        return false;
    }

    void *qData = FastllmCudaPrepareInput(q);
    void *kData = FastllmCudaPrepareInput(k);
    void *vData = FastllmCudaPrepareInput(v);
    void *gData = FastllmCudaPrepareInput(g);
    void *attnData = directOutputQk
        ? qData : FastllmCudaPrepareInput(attn);
    void *decayMaskData = (fuseDecayMask || directOutputQk)
        ? FastllmCudaPrepareInput(decayMask) : qData;
    void *kCumdecayData = FastllmCudaPrepareInput(kCumdecay);
    void *stateData = FastllmCudaPrepareInput(lastRecurrentState);
    void *outputData = FastllmCudaPrepareOutput(coreAttnOut);
    auto finishPrepared = [&]() {
        FastllmCudaFinishInput(q, qData);
        FastllmCudaFinishInput(k, kData);
        FastllmCudaFinishInput(v, vData);
        FastllmCudaFinishInput(g, gData);
        if (!directOutputQk) {
            FastllmCudaFinishInput(attn, attnData);
        }
        if (fuseDecayMask || directOutputQk) {
            FastllmCudaFinishInput(decayMask, decayMaskData);
        }
        FastllmCudaFinishInput(kCumdecay, kCumdecayData);
        FastllmCudaFinishInput(lastRecurrentState, stateData);
        FastllmCudaFinishOutput(coreAttnOut, outputData);
    };
    if (qData == nullptr || kData == nullptr || vData == nullptr ||
        gData == nullptr || attnData == nullptr ||
        decayMaskData == nullptr ||
        kCumdecayData == nullptr || stateData == nullptr ||
        outputData == nullptr) {
        finishPrepared();
        return false;
    }

    CUdeviceptr qPtr = (CUdeviceptr)qData;
    CUdeviceptr kPtr = (CUdeviceptr)kData;
    CUdeviceptr vPtr = (CUdeviceptr)vData;
    CUdeviceptr gPtr = (CUdeviceptr)gData;
    CUdeviceptr attnPtr = (CUdeviceptr)attnData;
    CUdeviceptr decayMaskPtr = (CUdeviceptr)decayMaskData;
    CUdeviceptr kCumdecayPtr = (CUdeviceptr)kCumdecayData;
    CUdeviceptr statePtr = (CUdeviceptr)stateData;
    CUdeviceptr nextStatePtr = (CUdeviceptr)scratch->nextState;
    CUdeviceptr chunkOffsetsPtr =
        (CUdeviceptr)metadata.chunkOffsets;
    CUdeviceptr chunkTokenBasesPtr =
        (CUdeviceptr)metadata.chunkTokenBases;
    CUdeviceptr chunkValidTokensPtr =
        (CUdeviceptr)metadata.chunkValidTokens;
    CUdeviceptr hPtr = (CUdeviceptr)scratch->h;
    CUdeviceptr vNewPtr = (CUdeviceptr)scratch->vNew;
    CUdeviceptr rowScalePtr = usePrecomputedScale
        ? (CUdeviceptr)scaleScratch->rowScale : 0;
    CUdeviceptr stateScalePtr = usePrecomputedScale
        ? (CUdeviceptr)scaleScratch->stateScale : 0;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    int32_t totalChunksArg = totalChunks;
    int32_t totalTokensArg = metadata.totalTokens;
    int32_t keyHeadsArg = keyHeads;
    int32_t headsArg = heads;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *hArgs[] = {
        &kPtr, &vPtr, &gPtr, &kCumdecayPtr,
        &statePtr, &nextStatePtr, &chunkOffsetsPtr,
        &hPtr, &vNewPtr, &rowScalePtr, &stateScalePtr,
        &totalChunksArg, &keyHeadsArg, &headsArg,
        &globalScratch, &profileScratch,
    };
    void *oArgs[] = {
        &qPtr, &kPtr, &gPtr, &attnPtr, &decayMaskPtr,
        &hPtr, &vNewPtr, &chunkTokenBasesPtr,
        &chunkValidTokensPtr, &outputPtr,
        &totalChunksArg, &totalTokensArg,
        &keyHeadsArg, &headsArg,
        &globalScratch, &profileScratch,
    };
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    unsigned int hVBlocks =
        (unsigned int)((vDim + hBlockV - 1) / hBlockV);
    unsigned int oVBlocks =
        (unsigned int)((vDim + oBlockV - 1) / oBlockV);
    LoadedTritonKernel *selectedHKernel =
        usePrecomputedScale ? hPrecomputedScaleKernel : hKernel;
    int selectedHNumWarps =
        usePrecomputedScale ? hPrecomputedScaleNumWarps : hNumWarps;
    int selectedHShared =
        usePrecomputedScale ? hPrecomputedScaleShared : hShared;
    LoadedTritonKernel *selectedOKernel = directOutputQk
        ? oDirectQkKernel
        : (fuseDecayMask ? oFusedDecayKernel : oKernel);
    int selectedONumWarps = directOutputQk
        ? oDirectQkNumWarps
        : (fuseDecayMask ? oFusedDecayNumWarps : oNumWarps);
    int selectedOShared = directOutputQk
        ? oDirectQkShared
        : (fuseDecayMask ? oFusedDecayShared : oShared);
    CUresult hResult = LaunchTritonKernel(
        selectedHKernel, hVBlocks, (unsigned int)(batch * heads), 1,
        (unsigned int)(selectedHNumWarps * 32),
        (unsigned int)selectedHShared,
        hArgs, stream);
    CUresult oResult = hResult == CUDA_SUCCESS
        ? LaunchTritonKernel(
              selectedOKernel, oVBlocks, (unsigned int)totalChunks,
              (unsigned int)heads,
              (unsigned int)(selectedONumWarps * 32),
              (unsigned int)selectedOShared, oArgs, stream)
        : hResult;
    CUresult stateCommitResult =
        hResult == CUDA_SUCCESS && oResult == CUDA_SUCCESS
            ? cuMemcpyDtoDAsync(statePtr, nextStatePtr, stateBytes, stream)
            : oResult;
    finishPrepared();
    return CheckCu(hResult,
                   "cuLaunchKernel chunk_gdn_varlen_prefill_h") &&
           CheckCu(oResult,
                   "cuLaunchKernel chunk_gdn_varlen_prefill_o") &&
           CheckCu(stateCommitResult,
                   "cuMemcpyDtoDAsync chunk_gdn_varlen_state_commit");
}

extern "C" bool FastllmCudaTritonChunkGdnKkt(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared,
    const fastllm::Data &kBeta, const fastllm::Data &k,
    int headGroup, fastllm::Data &output) {
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
    if (cubinPath == nullptr || kernelName == nullptr ||
        numWarps <= 0 || headGroup <= 1 ||
        kBeta.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        kBeta.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        kBeta.cudaData == nullptr || k.cudaData == nullptr ||
        !isDense(kBeta) || !isDense(k) ||
        kBeta.dims.size() != 5 || k.dims.size() != 5 ||
        kBeta.dims[0] != 1 || k.dims[0] != 1) {
        return false;
    }
    int valueHeads = kBeta.dims[1];
    int keyHeads = k.dims[1];
    int totalChunks = kBeta.dims[2];
    constexpr int chunkSize = 64;
    constexpr int kDim = 128;
    if (valueHeads <= 0 || keyHeads <= 0 || totalChunks <= 0 ||
        valueHeads != keyHeads * headGroup ||
        kBeta.dims != std::vector<int>(
            {1, valueHeads, totalChunks, chunkSize, kDim}) ||
        k.dims != std::vector<int>(
            {1, keyHeads, totalChunks, chunkSize, kDim})) {
        return false;
    }

    LoadedTritonKernel *kernel =
        LoadTritonKernel(cubinPath, kernelName, shared);
    if (kernel == nullptr) {
        return false;
    }
    output.dataType = fastllm::DataType::FLOAT16;
    output.dataDevice = kBeta.dataDevice;
    output.dataDeviceIds = kBeta.dataDeviceIds;
    output.Resize({1, valueHeads, totalChunks, chunkSize, chunkSize});
    output.Allocate(false);
    if (output.cudaData == nullptr) {
        return false;
    }

    void *kBetaData = FastllmCudaPrepareInput(kBeta);
    void *kData = FastllmCudaPrepareInput(k);
    void *outputData = FastllmCudaPrepareOutput(output);
    auto finishPrepared = [&]() {
        FastllmCudaFinishInput(kBeta, kBetaData);
        FastllmCudaFinishInput(k, kData);
        FastllmCudaFinishOutput(output, outputData);
    };
    if (kBetaData == nullptr || kData == nullptr || outputData == nullptr) {
        finishPrepared();
        return false;
    }
    CUdeviceptr kBetaPtr = (CUdeviceptr)kBetaData;
    CUdeviceptr kPtr = (CUdeviceptr)kData;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    int32_t totalChunksArg = totalChunks;
    int32_t keyHeadsArg = keyHeads;
    int32_t valueHeadsArg = valueHeads;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    void *args[] = {
        &kBetaPtr, &kPtr, &outputPtr,
        &totalChunksArg, &keyHeadsArg, &valueHeadsArg,
        &globalScratch, &profileScratch,
    };
    CUstream stream = reinterpret_cast<CUstream>(cudaStreamPerThread);
    CUresult result = LaunchTritonKernel(
        kernel, (unsigned int)totalChunks,
        (unsigned int)valueHeads, 1,
        (unsigned int)(numWarps * 32),
        (unsigned int)shared, args, stream);
    finishPrepared();
    return CheckCu(result, "cuLaunchKernel chunk_gdn_kkt");
}

extern "C" bool FastllmCudaTritonMergeMOEFP8E4M3Indexed(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN, int groupBlockK, int groupSizeM,
    const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
    fastllm::Data **weights, int weightsBatch, const int32_t *indices, const float *scores,
    int batch, int topk, int hidden, int inter) {
    (void)w1;
    TritonMoeFp8ExpertTable *table = nullptr;
    if (!GetTritonMoeFp8ExpertTable(weights, weightsBatch, hidden, inter, true, table)) {
        return false;
    }
    return LaunchTritonMergeMOEFP8E4M3Table(
        cubinPaths, kernelNames, numWarps, shared,
        routeBlockT, maxExperts, groupBlockM, groupBlockN, groupBlockK, groupSizeM,
        input, output, indices, scores, batch, topk, hidden, inter,
        {batch, hidden}, table);
}

/**
 * 使用交错Block128专家权重执行Triton MergeMOE。
 *
 * @param weights FastLLM独立专家权重数组，第0对保留给shared expert。
 * @param indices CUDA端路由expert索引，形状为[batch, topk]。
 * @param scores CUDA端路由权重，形状为[batch, topk]。
 * @return true表示专家表准备及全部kernel提交成功；false表示调用方应回退。
 */
extern "C" bool FastllmCudaTritonMergeMOEFP8E4M3Block128Indexed(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN,
    int groupBlockK, int groupSizeM, const fastllm::Data &input,
    fastllm::Data &w1, fastllm::Data &output, fastllm::Data **weights,
    int weightsBatch, const int32_t *indices, const float *scores,
    int batch, int topk, int hidden, int inter) {
    (void)w1;
    TritonMoeFp8ExpertTable *table = nullptr;
    if (!GetTritonMoeFp8Block128ExpertTable(
            weights, weightsBatch, hidden, inter, table)) {
        return false;
    }
    return LaunchTritonMergeMOEFP8E4M3Table(
        cubinPaths, kernelNames, numWarps, shared,
        routeBlockT, maxExperts, groupBlockM, groupBlockN, groupBlockK,
        groupSizeM, input, output, indices, scores, batch, topk, hidden, inter,
        {batch, hidden}, table);
}

extern "C" bool FastllmCudaTritonFusedMOEFP8E4M3(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN, int groupBlockK, int groupSizeM,
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
    const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &w1, fastllm::Data &output,
    int batch, int topk, int hidden, int inter, int experts) {
    (void)w1;
    if (batch <= 0 || topk <= 0 || hidden <= 0 || inter <= 0 || experts <= 0 ||
        index.dataDevice != fastllm::DataDevice::CUDA || index.dataType != fastllm::DataType::INT32 ||
        score.dataDevice != fastllm::DataDevice::CUDA || score.dataType != fastllm::DataType::FLOAT32 ||
        index.cudaData == nullptr || score.cudaData == nullptr ||
        index.Count(0) != (uint64_t)batch * topk ||
        score.Count(0) != (uint64_t)batch * topk) {
        return false;
    }
    TritonMoeFp8ExpertTable *table = nullptr;
    if (!GetTritonMoeFp8FusedExpertTable(gate, up, down, experts, hidden, inter, table)) {
        return false;
    }
    return LaunchTritonMergeMOEFP8E4M3Table(
        cubinPaths, kernelNames, numWarps, shared,
        routeBlockT, maxExperts, groupBlockM, groupBlockN, groupBlockK, groupSizeM,
        input, output, (const int32_t*)index.cudaData, (const float*)score.cudaData,
        batch, topk, hidden, inter, input.dims, table);
}

/**
 * 使用3D交错Block128 gate/up/down权重执行Triton FusedMOE。
 *
 * @param gate gate权重，逻辑形状为[experts, inter, hidden]。
 * @param up up权重，逻辑形状为[experts, inter, hidden]。
 * @param down down权重，逻辑形状为[experts, hidden, inter]。
 * @return true表示全部kernel提交成功；false表示调用方应回退原CUDA实现。
 */
extern "C" bool FastllmCudaTritonFusedMOEFP8E4M3Block128(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN,
    int groupBlockK, int groupSizeM, const fastllm::Data &input,
    fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
    const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &w1, fastllm::Data &output,
    int batch, int topk, int hidden, int inter, int experts) {
    return FastllmCudaTritonFusedMOEFP8E4M3(
        cubinPaths, kernelNames, numWarps, shared,
        routeBlockT, maxExperts, groupBlockM, groupBlockN, groupBlockK,
        groupSizeM, input, gate, up, down, index, score, w1, output,
        batch, topk, hidden, inter, experts);
}
