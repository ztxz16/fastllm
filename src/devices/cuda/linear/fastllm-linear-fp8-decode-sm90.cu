/*
 * Hopper small-batch (1 <= rows < 32) block-scaled W8A8 projection.
 * The swapped-operand WGMMA/TMA mainloop is from TensorRT-LLM v1.3.0rc28.
 * See third_party/trtllm_deep_gemm/UPSTREAM.txt for provenance and licenses.
 * Swapping operands lets decode use a 16/24/32-column MMA instead of padding
 * activation rows to the hardware's 64-row WGMMA minimum.
 */
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>
#include <cuda.h>
#include <algorithm>
#include <map>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include "trtllm_deep_gemm/fp8_gemm_impl.cuh"

namespace {
using namespace fastllm_trtllm_deep_gemm;

struct Scratch {
    void *data = nullptr;
    size_t bytes = 0;
    // CUDA graphs retain earlier pointers; growing the scratch buffer must not
    // invalidate a previously captured graph. Grow geometrically to bound
    // retained allocations to less than twice the largest buffer per worker.
    std::vector<void *> retired;
};

static thread_local std::map<int, Scratch> scratchByDevice;

template <typename T>
__global__ void Quantize(const T *input, __nv_fp8_e4m3 *output,
                         float *scales, int rows, int cols, int scaleRows) {
    const int task = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int groups = cols / 128;
    const int row = task / groups;
    const int group = task % groups;
    if (row >= scaleRows) return;
    float values[4];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        values[i] = row < rows ? float(input[(size_t)row * cols + group * 128 + lane * 4 + i]) : 0.0f;
        amax = fmaxf(amax, fabsf(values[i]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    const float scale = fmaxf(amax * (1.0f / 448.0f), 1.0e-10f);
    if (lane == 0) scales[(size_t)group * scaleRows + row] = scale;
    if (row < rows) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            output[(size_t)row * cols + group * 128 + lane * 4 + i] = __nv_fp8_e4m3(values[i] / scale);
    }
}

template <typename T>
__global__ void AddBias(T *out, const float *bias, size_t count, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = T(float(out[i]) + bias[i % cols]);
}

static bool MakeTma(CUtensorMap &desc, CUtensorMapDataType dtype,
                    void *ptr, uint64_t inner, uint64_t outer,
                    uint64_t stride, uint32_t boxInner, uint32_t boxOuter,
                    CUtensorMapSwizzle swizzle) {
    const cuuint64_t dims[] = {inner, outer};
    const cuuint64_t strides[] = {stride};
    const cuuint32_t box[] = {boxInner, boxOuter};
    const cuuint32_t elementStrides[] = {1, 1};
    return cuTensorMapEncodeTiled(&desc,
        dtype, 2, ptr, dims, strides, box, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <typename T, int N, int K, int BN, int stages>
bool LaunchSwap(void *input, void *weight, float *inputScales,
                float *weightScales, void *output, int rows, int scaleRows,
                int workers, cudaStream_t stream) {
    constexpr int BM = 128;
    CUtensorMap a, b, d, sfa;
    if (!MakeTma(a,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, weight, K, N, K,
            128, BM, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !MakeTma(b,
            CU_TENSOR_MAP_DATA_TYPE_UINT8, input, K, rows, K,
            128, BN, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !MakeTma(d,
            std::is_same_v<T, half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            output, N, rows, N * sizeof(T),
            BM, std::min(rows, BN), CU_TENSOR_MAP_SWIZZLE_NONE) ||
        !MakeTma(sfa,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32, inputScales, scaleRows, K / 128,
            scaleRows * 4, BN, 1, CU_TENSOR_MAP_SWIZZLE_NONE)) return false;
    using Scheduler = NormalSchedulerSwapAB<N, BM, BN, 1, 1>;
    auto kernel = fp8_gemm_kernel_swapAB<N, K, BM, BN, 128, 1, stages,
        128, 128, 1, Scheduler, NormalSchedulerInputSwapAB, T>;
    constexpr int smem = BM * BN * 2 + stages * (BM * 128 + BN * 128 + 128)
        + ((K / 128 * 4 + 7) / 8) * 8 + stages * 16;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem) != cudaSuccess) return false;
    NormalSchedulerInputSwapAB params{static_cast<uint32_t>(rows), nullptr};
    kernel<<<workers, 384, smem, stream>>>(static_cast<T *>(output),
        weightScales, params, a, b, sfa, d);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T, int N, int K>
bool LaunchForRows(void *input, void *weight, float *inputScales,
                   float *weightScales, void *output, int rows, int scaleRows,
                   int workers, cudaStream_t stream) {
    // Match TRT-LLM v1.3.0rc28 get_best_gemm_config for the validated shapes
    // and rows < 32: BM=128, minimize waves, then maximize final-wave use,
    // then prefer the smaller BN. BN > 32 cannot improve these small batches.
    // Use the actual SM count: H100/H200 can prefer a different tile than H20.
    int bestN = 16, bestWaves = 0, bestUtil = 0;
    for (int bn : {16, 24, 32}) {
        const int blocks = ((N + 127) / 128) * ((rows + bn - 1) / bn);
        const int waves = (blocks + workers - 1) / workers;
        const int util = blocks % workers ? blocks % workers : workers;
        if (bestWaves == 0 || waves < bestWaves ||
            (waves == bestWaves && util > bestUtil)) {
            bestN = bn;
            bestWaves = waves;
            bestUtil = util;
        }
    }
    if (bestN == 16) return LaunchSwap<T, N, K, 16, 8>(input, weight, inputScales,
        weightScales, output, rows, scaleRows, workers, stream);
    if (bestN == 24) return LaunchSwap<T, N, K, 24, 6>(input, weight, inputScales,
        weightScales, output, rows, scaleRows, workers, stream);
    return LaunchSwap<T, N, K, 32, 8>(input, weight, inputScales,
        weightScales, output, rows, scaleRows, workers, stream);
}

template <typename T>
static bool Run(
        const void *input, void *weight, float *weightScales, const float *bias,
        void *output, int rows, int cols, int outCols, Scratch &scratch,
        cudaStreamCaptureStatus capture, int workers) {
    const cudaStream_t stream = cudaStreamPerThread;
    const int scaleRows = (rows + 31) / 32 * 32;
    const size_t inputBytes = ((size_t)rows * cols + 255) / 256 * 256;
    const size_t bytes = inputBytes + (size_t)scaleRows * (cols / 128) * sizeof(float);
    if (scratch.bytes < bytes) {
        if (capture != cudaStreamCaptureStatusNone) return false;
        const size_t capacity = std::max(bytes, scratch.bytes * 2);
        void *next = FastllmCudaMalloc(capacity);
        if (!next) return false;
        if (scratch.data) scratch.retired.push_back(scratch.data);
        scratch.data = next;
        scratch.bytes = capacity;
    }
    auto *quant = static_cast<__nv_fp8_e4m3 *>(scratch.data);
    auto *scales = reinterpret_cast<float *>(static_cast<char *>(scratch.data) + inputBytes);
    const size_t tasks = (size_t)scaleRows * (cols / 128);
    Quantize<<<(tasks + 7) / 8, 256, 0, stream>>>(static_cast<const T *>(input), quant, scales, rows, cols, scaleRows);
    if (cudaGetLastError() != cudaSuccess) return false;
    bool ok = false;
    if (outCols == 34816 && cols == 5120) {
        ok = LaunchForRows<T, 34816, 5120>(quant, weight, scales, weightScales,
            output, rows, scaleRows, workers, stream);
    } else if (outCols == 5120 && cols == 17408) {
        ok = LaunchForRows<T, 5120, 17408>(quant, weight, scales, weightScales,
            output, rows, scaleRows, workers, stream);
    } else if (outCols == 5120 && cols == 6144) {
        ok = LaunchForRows<T, 5120, 6144>(quant, weight, scales, weightScales,
            output, rows, scaleRows, workers, stream);
    } else if (outCols == 16384 && cols == 5120) {
        ok = LaunchForRows<T, 16384, 5120>(quant, weight, scales, weightScales,
            output, rows, scaleRows, workers, stream);
    } else if (outCols == 14336 && cols == 5120) {
        ok = LaunchForRows<T, 14336, 5120>(quant, weight, scales, weightScales,
            output, rows, scaleRows, workers, stream);
    }
    if (ok && bias) {
        const size_t count = (size_t)rows * outCols;
        AddBias<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<T *>(output), bias, count, outCols);
        ok = cudaGetLastError() == cudaSuccess;
    }
    return ok;
}
}  // namespace

bool FastllmCudaDeepGemmDecodeFp8Sm90(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    // Match TRT-LLM's swapAB boundary (rows < 32) for validated shapes.
    // Exact verification, other layouts/dtypes and non-SM90 devices use fallback.
    if (n < 1 || n >= 32 || fastllm::FastllmCudaGetLinearExactBatchThreshold() != 0 ||
        !((m == 5120 && (k == 34816 || k == 16384 || k == 14336)) ||
          (k == 5120 && (m == 17408 || m == 6144))) ||
        !input.cudaData || !weight.cudaData || !output.cudaData ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 ||
        weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        output.dataType != input.dataType ||
        weight.scales.size() != (size_t)(m / 128) * (k / 128) ||
        FastllmCudaHasFp8MarlinLayout(weight) ||
        (!bias.dims.empty() && (bias.dataType != fastllm::DataType::FLOAT32 ||
            bias.dataDevice != fastllm::DataDevice::CUDA || bias.Count(0) != k || !bias.cudaData))) return false;
    int device = 0, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        major != 9 || minor != 0) return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        (capture != cudaStreamCaptureStatusNone && weight.extraCudaData.empty())) return false;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, k);
    if (weight.extraCudaData.empty() || !weight.extraCudaData[0]) return false;
    int workers = 0;
    if (cudaDeviceGetAttribute(&workers, cudaDevAttrMultiProcessorCount,
                              device) != cudaSuccess || workers <= 0) return false;
    auto run = input.dataType == fastllm::DataType::BFLOAT16
        ? Run<__nv_bfloat16> : Run<half>;
    return run(input.cudaData, weight.cudaData,
        static_cast<float *>(weight.extraCudaData[0]),
        bias.dims.empty() ? nullptr : static_cast<const float *>(bias.cudaData),
        output.cudaData, n, m, k, scratchByDevice[device], capture, workers);
}
