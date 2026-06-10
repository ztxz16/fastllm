#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda.h>
#include <cuda_runtime.h>

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
    unsigned int blockX, unsigned int shared, void **args) {
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
    config.hStream = nullptr;
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

struct TritonLinearFp8Scratch {
    int inputQuantCapacity = 0;
    int inputScaleCapacity = 0;
    uint8_t *inputQuant = nullptr;
    float *inputScale = nullptr;
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
        FreeTritonScratchPtr(cached.inputQuant);
        cached.inputQuantCapacity = inputQuantElements;
        cached.inputQuant = (uint8_t*)FastllmCudaMalloc((size_t)inputQuantElements);
    }
    if (cached.inputScaleCapacity < inputScaleElements) {
        FreeTritonScratchPtr(cached.inputScale);
        cached.inputScaleCapacity = inputScaleElements;
        cached.inputScale = (float*)FastllmCudaMalloc((size_t)inputScaleElements * sizeof(float));
    }
    if (cached.inputQuant == nullptr || cached.inputScale == nullptr) {
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

    if (cached.expertCapacity < experts) {
        FreeTritonScratchPtr(cached.expertCounts);
        FreeTritonScratchPtr(cached.expertOffsets);
        FreeTritonScratchPtr(cached.expertCursors);
        FreeTritonScratchPtr(cached.expertBlockOffsets);
        cached.expertCapacity = experts;
        cached.expertCounts = (int32_t*)FastllmCudaMalloc((size_t)experts * sizeof(int32_t));
        cached.expertOffsets = (int32_t*)FastllmCudaMalloc((size_t)(experts + 1) * sizeof(int32_t));
        cached.expertCursors = (int32_t*)FastllmCudaMalloc((size_t)experts * sizeof(int32_t));
        cached.expertBlockOffsets = (int32_t*)FastllmCudaMalloc((size_t)experts * sizeof(int32_t));
    }
    if (cached.taskCapacity < totalTasks) {
        FreeTritonScratchPtr(cached.sortedTasks);
        FreeTritonScratchPtr(cached.blockExperts);
        FreeTritonScratchPtr(cached.blockStarts);
        cached.taskCapacity = totalTasks;
        cached.sortedTasks = (int32_t*)FastllmCudaMalloc((size_t)totalTasks * sizeof(int32_t));
        cached.blockExperts = (int32_t*)FastllmCudaMalloc((size_t)totalTasks * sizeof(int32_t));
        cached.blockStarts = (int32_t*)FastllmCudaMalloc((size_t)totalTasks * sizeof(int32_t));
    }
    if (cached.totalBlocks == nullptr) {
        cached.totalBlocks = (int32_t*)FastllmCudaMalloc(sizeof(int32_t));
    }
    if (cached.downOutputCapacity < downOutputBytes) {
        FreeTritonScratchPtr(cached.downOutput);
        cached.downOutputCapacity = downOutputBytes;
        cached.downOutput = (uint8_t*)FastllmCudaMalloc((size_t)downOutputBytes);
    }
    if (cached.inputQuantCapacity < inputQuantElements) {
        FreeTritonScratchPtr(cached.inputQuant);
        cached.inputQuantCapacity = inputQuantElements;
        cached.inputQuant = (uint8_t*)FastllmCudaMalloc((size_t)inputQuantElements);
    }
    if (cached.inputScaleCapacity < inputScaleElements) {
        FreeTritonScratchPtr(cached.inputScale);
        cached.inputScaleCapacity = inputScaleElements;
        cached.inputScale = (float*)FastllmCudaMalloc((size_t)inputScaleElements * sizeof(float));
    }
    if (cached.gateUpCapacity < gateUpBytes) {
        FreeTritonScratchPtr(cached.gateUp);
        cached.gateUpCapacity = gateUpBytes;
        cached.gateUp = (uint8_t*)FastllmCudaMalloc((size_t)gateUpBytes);
    }
    if (cached.activationQuantCapacity < activationQuantElements) {
        FreeTritonScratchPtr(cached.activationQuant);
        cached.activationQuantCapacity = activationQuantElements;
        cached.activationQuant = (uint8_t*)FastllmCudaMalloc((size_t)activationQuantElements);
    }
    if (cached.activationScaleCapacity < activationScaleElements) {
        FreeTritonScratchPtr(cached.activationScale);
        cached.activationScaleCapacity = activationScaleElements;
        cached.activationScale = (float*)FastllmCudaMalloc((size_t)activationScaleElements * sizeof(float));
    }
    if (cached.expertCounts == nullptr || cached.expertOffsets == nullptr ||
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

static bool GetTritonMoeFp8FusedExpertTable(
    fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
    int experts, int hidden, int inter, TritonMoeFp8ExpertTable *&table) {
    table = nullptr;
    if (experts <= 0 || hidden <= 0 || inter <= 0 ||
        gate.dataDevice != fastllm::DataDevice::CUDA ||
        up.dataDevice != fastllm::DataDevice::CUDA ||
        down.dataDevice != fastllm::DataDevice::CUDA ||
        gate.dataType != fastllm::DataType::FP8_E4M3 ||
        up.dataType != fastllm::DataType::FP8_E4M3 ||
        down.dataType != fastllm::DataType::FP8_E4M3 ||
        gate.cudaData == nullptr || up.cudaData == nullptr || down.cudaData == nullptr ||
        gate.extraCudaData.empty() || up.extraCudaData.empty() || down.extraCudaData.empty() ||
        gate.dims.size() != 3 || up.dims.size() != 3 || down.dims.size() != 3 ||
        gate.dims[0] != experts || gate.dims[1] != inter || gate.dims[2] != hidden ||
        up.dims[0] != experts || up.dims[1] != inter || up.dims[2] != hidden ||
        down.dims[0] != experts || down.dims[1] != hidden || down.dims[2] != inter ||
        gate.blockM <= 0 || gate.blockK <= 0 || down.blockM <= 0 || down.blockK <= 0 ||
        up.blockM != gate.blockM || up.blockK != gate.blockK) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    auto key = std::make_tuple(deviceId, (const void*)gate.cudaData, (const void*)up.cudaData, (const void*)down.cudaData);
    std::lock_guard<std::mutex> guard(g_tritonMoeFp8FusedTableMutex);
    TritonMoeFp8ExpertTable &cached = g_tritonMoeFp8FusedTables[key];
    if (cached.inited) {
        if (cached.experts != experts || cached.hidden != hidden || cached.inter != inter ||
            cached.gateBlockM != gate.blockM || cached.gateBlockK != gate.blockK ||
            cached.downBlockM != down.blockM || cached.downBlockK != down.blockK) {
            return false;
        }
        table = &cached;
        return true;
    }

    cached.inited = true;
    cached.experts = experts;
    cached.hidden = hidden;
    cached.inter = inter;
    cached.gateBlockM = gate.blockM;
    cached.gateBlockK = gate.blockK;
    cached.downBlockM = down.blockM;
    cached.downBlockK = down.blockK;
    cached.fusedSeparateGateUp = true;
    cached.fusedGateWeights = (uint8_t*)gate.cudaData;
    cached.fusedUpWeights = (uint8_t*)up.cudaData;
    cached.fusedGateScales = (float*)gate.extraCudaData[0];
    cached.fusedUpScales = (float*)up.extraCudaData[0];
    cached.packedDownWeights = (uint8_t*)down.cudaData;
    cached.packedDownScales = (float*)down.extraCudaData[0];
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
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 0;
    }
    return prop.major * 10 + prop.minor;
}

extern "C" bool FastllmCudaTritonLinearFP8E4M3Block128(
    const char *quantCubitPath, const char *quantKernelName, int quantNumWarps, int quantShared,
    const char *matmulCubitPath, const char *matmulKernelName, int matmulNumWarps, int matmulShared,
    int blockM, int blockN, int blockK, int groupSizeM, bool packedWeight,
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

    LoadedTritonKernel *quantKernel = LoadTritonKernel(quantCubitPath, quantKernelName, quantShared);
    LoadedTritonKernel *matmulKernel = LoadTritonKernel(matmulCubitPath, matmulKernelName, matmulShared);
    if (quantKernel == nullptr || matmulKernel == nullptr) {
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

    void *quantArgs[] = {
        &inputPtr,
        &inputQuantPtr,
        &inputScalePtr,
        &mArg,
        &kArg,
        &globalScratch,
        &profileScratch,
    };
    CUresult result = LaunchTritonKernel(
        quantKernel,
        (unsigned int)n, (unsigned int)inputScaleCols, 1,
        (unsigned int)(quantNumWarps * 32), (unsigned int)quantShared,
        quantArgs);
    if (!CheckCu(result, "cuLaunchKernel linear_fp8_block128_quant_input")) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    int gridM = (n + blockM - 1) / blockM;
    int gridN = (k + blockN - 1) / blockN;
    void *matmulArgs[] = {
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
    result = LaunchTritonKernel(
        matmulKernel,
        (unsigned int)(gridM * gridN), 1, 1,
        (unsigned int)(matmulNumWarps * 32), (unsigned int)matmulShared,
        matmulArgs);

    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return CheckCu(result, "cuLaunchKernel linear_fp8_block128_matmul");
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
    if (table->fusedSeparateGateUp) {
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
    CUdeviceptr gateScalePtrs = (CUdeviceptr)(table->fusedSeparateGateUp ?
        table->fusedGateScales : table->packedGateScales);
    CUdeviceptr upScalePtrs = (CUdeviceptr)table->fusedUpScales;
    CUdeviceptr downWeightPtrs = (CUdeviceptr)table->packedDownWeights;
    CUdeviceptr downScalePtrs = (CUdeviceptr)table->packedDownScales;
    CUdeviceptr outputPtr = (CUdeviceptr)outputData;
    int32_t batchArg = batch;
    int32_t topkArg = topk;
    int32_t totalTasksArg = totalTasks;
    int32_t expertsArg = table->experts;
    int32_t hiddenArg = hidden;
    int32_t interArg = inter;
    CUdeviceptr globalScratch = 0;
    CUdeviceptr profileScratch = 0;
    auto launchTriton = [&](int kernelId, unsigned int gridX, unsigned int gridY, unsigned int gridZ,
                            void **args) -> CUresult {
        return LaunchTritonKernel(
            kernels[kernelId],
            gridX, gridY, gridZ,
            (unsigned int)(numWarps[kernelId] * 32),
            (unsigned int)shared[kernelId],
            args);
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
