/*
 * Hopper block-scaled W8A8 linear. The WGMMA/TMA mainloop is vendored from
 * the DeepGEMM implementation used by vLLM 0.28.0 (MIT license).
 */
#include "fastllm-cuda.cuh"
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <algorithm>
#include <map>
#include <vector>

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

namespace {
using namespace deep_gemm;

// A persistent grid may have fewer or more CTAs than physical SMs. Its size
// must agree with the scheduler's template argument, not the device SM count.
constexpr int kWorkers = 114;
constexpr int kBlockN = 128;

struct Scratch {
    void *data = nullptr;
    size_t bytes = 0;
    // CUDA graphs retain earlier pointers; growing a prefill buffer must not
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

static bool MakeTma(cute::TmaDescriptor &desc, CUtensorMapDataType dtype,
                    void *ptr, uint64_t inner, uint64_t outer,
                    uint64_t stride, uint32_t boxInner, uint32_t boxOuter,
                    CUtensorMapSwizzle swizzle) {
    const cuuint64_t dims[] = {inner, outer};
    const cuuint64_t strides[] = {stride};
    const cuuint32_t box[] = {boxInner, boxOuter};
    const cuuint32_t elementStrides[] = {1, 1};
    return cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&desc),
        dtype, 2, ptr, dims, strides, box, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <typename Out, int BlockM, int BlockN = kBlockN,
          int Stages = (BlockM == 256 ? 3 : 4), int Workers = kWorkers>
static auto Kernel() {
    return &sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K, 0, 0, 0, 1,
        BlockM, BlockN, 128, 128, 128, (BlockN % 64 == 0 ? 128 : 0), Stages,
        128, (BlockM == 64 ? 128 : 256), (BlockM == 256 ? 2 : 1), true, Workers, GemmType::Normal, Out,
        epilogue::transform::EpilogueIdentity>;
}

template <typename Out, int BlockM, int BlockN = kBlockN,
          int Stages = (BlockM == 256 ? 3 : 4), int Workers = kWorkers>
static bool Launch(void *input, void *weight, float *sa, float *sb,
                   void *output, int rows, int cols, int outCols,
                   int scaleRows, cudaStream_t stream) {
    cute::TmaDescriptor a, b, d, sfa;
    if (!MakeTma(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, input, cols, rows,
                 cols, 128, BlockM, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !MakeTma(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, weight, cols, outCols,
                 cols, 128, BlockN, CU_TENSOR_MAP_SWIZZLE_128B) ||
        !MakeTma(d, cute::is_same_v<Out, cutlass::half_t>
                     ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                 output, outCols, rows,
                 outCols * 2, (BlockN % 64 == 0 ? 64 : BlockN), BlockM,
                 (BlockN % 64 == 0 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE)) ||
        !MakeTma(sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sa, scaleRows, cols / 128,
                 scaleRows * 4, BlockM, 1, CU_TENSOR_MAP_SWIZZLE_NONE)) return false;
    // A non-divisor N tile can cross a 128-column weight-scale boundary.
    // Reserve two scale rows, and use unswizzled output TMA for narrow tiles.
    const int sharedBytes = BlockM * BlockN * 2 +
        Stages * (BlockM * 128 + BlockN * 128 + BlockM * 4) +
        ((cols / 128 * 4 * (128 % BlockN == 0 ? 1 : 2) + 7) / 8) * 8 + Stages * 16;
    auto kernel = Kernel<Out, BlockM, BlockN, Stages, Workers>();
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedBytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if constexpr (BlockM == 256) {
        cudaLaunchAttribute attr{};
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = 2;
        attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(Workers);
        config.blockDim = dim3(384);
        config.dynamicSmemBytes = sharedBytes;
        config.stream = stream;
        config.attrs = &attr;
        config.numAttrs = 1;
        const cudaError_t status = cudaLaunchKernelEx(&config, kernel, sb, static_cast<int *>(nullptr),
            uint32_t(rows), uint32_t(outCols), uint32_t(cols), a, b, d, sfa);
        const cudaError_t lastError = cudaGetLastError();
        return status == cudaSuccess && lastError == cudaSuccess;
    } else {
        kernel<<<Workers, (BlockM == 64 ? 256 : 384), sharedBytes, stream>>>(sb, nullptr, rows, outCols, cols, a, b, d, sfa);
    }
    return cudaGetLastError() == cudaSuccess;
}

template <typename In, typename Out>
static bool Run(
        const void *input, void *weight, float *weightScales, const float *bias,
        void *output, int rows, int cols, int outCols, Scratch &scratch,
        cudaStreamCaptureStatus capture, bool tunedDecode) {
    const cudaStream_t stream = cudaStreamPerThread;
    const int scaleRows = (rows + 3) / 4 * 4;
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
    Quantize<<<(tasks + 7) / 8, 256, 0, stream>>>(static_cast<const In *>(input), quant, scales, rows, cols, scaleRows);
    if (cudaGetLastError() != cudaSuccess) return false;
    auto launch = rows >= 1024 ? Launch<Out, 256>
        : rows <= 128 ? Launch<Out, 64> : Launch<Out, 128>;
    // H20 (78 SMs), FP16 decode only. Other shapes keep the original dispatch.
    // Narrow N tiles match TRT-LLM for one M tile; two M tiles favor BN128
    // for gate/up and full-attention QKV in the measured sweep.
    if constexpr (cute::is_same_v<Out, cutlass::half_t>) {
        if (tunedDecode) {
            if (outCols == 5120 && (cols == 17408 || cols == 6144)) {
                launch = Launch<Out, 64, 72, 6, 78>;
            } else if (cols == 5120 && (outCols == 34816 || outCols == 16384)) {
                launch = rows > 64 && outCols == 34816
                    ? Launch<Out, 64, 128, 6, 78> : Launch<Out, 64, 112, 6, 78>;
            } else if (cols == 5120 && outCols == 14336) {
                launch = rows > 64
                    ? Launch<Out, 64, 128, 6, 78> : Launch<Out, 64, 96, 6, 78>;
            }
        }
    }
    bool ok = launch(quant, weight, scales, weightScales,
                     output, rows, cols, outCols, scaleRows, stream);
    if (ok && bias) {
        const size_t count = (size_t)rows * outCols;
        AddBias<<<(count + 255) / 256, 256, 0, stream>>>(static_cast<In *>(output), bias, count, outCols);
        ok = cudaGetLastError() == cudaSuccess;
    }
    return ok;
}
}  // namespace

bool FastllmCudaDeepGemmDecodeFp8Sm90(
        const fastllm::Data &, fastllm::Data &, const fastllm::Data &, fastllm::Data &, int, int, int);

bool FastllmCudaDeepGemmLinearFp8Sm90(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (n > 0 && n < 32) return FastllmCudaDeepGemmDecodeFp8Sm90(input, weight, bias, output, n, m, k);
    if (n < std::max(32, fastllm::FastllmCudaGetLinearExactBatchThreshold()) ||
        m <= 0 || k <= 0 || m % 128 || k % 128 ||
        !input.cudaData || !weight.cudaData || !output.cudaData ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 ||
        weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m ||
        (input.dataType != fastllm::DataType::FLOAT16 && input.dataType != fastllm::DataType::BFLOAT16) ||
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
    auto run = input.dataType == fastllm::DataType::BFLOAT16
        ? Run<__nv_bfloat16, cutlass::bfloat16_t> : Run<half, cutlass::half_t>;
    int smCount = 0;
    const bool tunedDecode = n >= 32 && n <= 128 &&
        input.dataType == fastllm::DataType::FLOAT16 &&
        fastllm::FastllmCudaGetLinearExactBatchThreshold() == 0 &&
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
        smCount == 78;
    return run(input.cudaData, weight.cudaData,
        static_cast<float *>(weight.extraCudaData[0]),
        bias.dims.empty() ? nullptr : static_cast<const float *>(bias.cudaData),
        output.cudaData, n, m, k, scratchByDevice[device], capture, tunedDecode);
}
