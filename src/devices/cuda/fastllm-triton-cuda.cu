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
