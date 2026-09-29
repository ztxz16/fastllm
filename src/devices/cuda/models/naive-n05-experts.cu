#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <cstring>

namespace {
using BF16 = __nv_bfloat16;
__device__ float RoundBF(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__global__ void DecodeWeight(const uint8_t *packed, BF16 *values, float *scales,
                             int columns) {
    int row = blockIdx.y, block = blockIdx.x, d = threadIdx.x;
    int blocks = columns / 128;
    const uint8_t *p = packed + ((size_t)row * blocks + block) * 132;
    __nv_fp8_e4m3 x; x.__x = p[d];
    values[(size_t)row * columns + block * 128 + d] = __float2bfloat16_rn(float(x));
    if (d == 0) scales[row * blocks + block] = *(const float *)(p + 128);
}

// Quantize the gathered BF16 input or the interleaved gate/up result. Retain
// the reference's FP32 scale, FP16 RTZ boundary and each BF16 rounding step.
template<bool Swiglu>
__global__ void QuantizeRows(const BF16 *input, const float *gateUp,
        const int *routes, const float *silu, BF16 *values, float *scales,
        int topk, int columns) {
    __shared__ float maxima[128];
    int row = blockIdx.y, block = blockIdx.x, d = threadIdx.x;
    int col = block * 128 + d;
    float x;
    if (Swiglu) {
        const float *p = gateUp + (size_t)row * columns * 2 + col * 2;
        BF16 gate = __float2bfloat16_rn(p[0]);
        x = RoundBF(__fmul_rn(RoundBF(silu[__bfloat16_as_ushort(gate)]), RoundBF(p[1])));
    } else {
        x = float(input[(size_t)(routes[row] / topk) * columns + col]);
    }
    maxima[d] = fabsf(x);
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (d < s) maxima[d] = fmaxf(maxima[d], maxima[d + s]);
        __syncthreads();
    }
    float scale = __fmul_rn(maxima[0], 1.0f / 448.0f);
    float normalized = __fdiv_rn(x, fmaxf(scale, 1e-12f));
    normalized = __uint_as_float(__float_as_uint(normalized) & 0xffffe000u);
    __nv_fp8_e4m3 fp8(normalized);
    values[(size_t)row * columns + col] = __float2bfloat16_rn(float(fp8));
    if (d == 0) scales[row * (columns / 128) + block] = scale;
}

template<bool Routed>
__global__ void SumBlocks(const float *parts, const float *as, const float *bs,
        float *out, const float *scores, const int *routes,
        int rows, int columns, int blocks) {
    int col = blockIdx.x * 256 + threadIdx.x, row = blockIdx.y;
    if (col >= columns) return;
    float value = 0;
    for (int b = 0; b < blocks; ++b) {
        float dot = parts[((size_t)b * rows + row) * columns + col];
        value = __fadd_rn(value, __fmul_rn(__fmul_rn(dot, as[row * blocks + b]), bs[col * blocks + b]));
    }
    if (Routed) value = RoundBF(__fmul_rn(RoundBF(value), RoundBF(scores[routes[row]])));
    out[(size_t)row * columns + col] = value;
}

struct Buffer {
    void *ptr = nullptr;
    size_t capacity = 0;
    bool pinned = false;
    bool Reserve(size_t bytes) {
        if (bytes <= capacity) return true;
        size_t reserved = std::max<size_t>(4096, capacity);
        while (reserved < bytes) reserved *= 2;
        if (ptr) { if (pinned) cudaFreeHost(ptr); else cudaFree(ptr); }
        ptr = nullptr; capacity = 0;
        cudaError_t result = pinned ? cudaMallocHost(&ptr, reserved) : cudaMalloc(&ptr, reserved);
        if (result != cudaSuccess) return false;
        capacity = reserved; return true;
    }
    ~Buffer() { if (ptr) { if (pinned) cudaFreeHost(ptr); else cudaFree(ptr); } }
    template<class T> T *As() { return (T *)ptr; }
};
struct Workspace {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    Buffer input, scores, routes, lookup, packed, weight, weightScales;
    Buffer activation, activationScales, parts, gateUp, output, hostOutput;
    int device;
    explicit Workspace(int d) : device(d) { hostOutput.pinned = true; }
    bool Init() {
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (!handle && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
        return cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS;
    }
    ~Workspace() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (handle) cublasDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }
};
std::mutex &Mutex() { static std::mutex mutex; return mutex; }
std::map<int, std::unique_ptr<Workspace>> &Workspaces() {
    // Explicitly released while CUDA is alive, like the NUMA runtime cache.
    static auto *workspaces = new std::map<int, std::unique_ptr<Workspace>>;
    return *workspaces;
}

bool Gemm(Workspace &w, const uint8_t *hostWeight, int rows, int input, int output) {
    int blocks = input / 128;
    size_t bytes = (size_t)output * blocks * 132;
    if (cudaMemcpyAsync(w.packed.ptr, hostWeight, bytes, cudaMemcpyHostToDevice, w.stream) != cudaSuccess) return false;
    DecodeWeight<<<dim3(blocks, output), 128, 0, w.stream>>>(
        w.packed.As<uint8_t>(), w.weight.As<BF16>(), w.weightScales.As<float>(), input);
    const float one = 1, zero = 0;
    // Every batch is one independent 128-element scale block. Decode only
    // unscaled E4M3 values to BF16; apply the two FP32 scales afterwards.
    return cublasGemmStridedBatchedEx(w.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        output, rows, 128, &one, w.weight.ptr, CUDA_R_16BF, input, 128,
        w.activation.ptr, CUDA_R_16BF, input, 128, &zero,
        w.parts.ptr, CUDA_R_32F, output, (long long)rows * output, blocks,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
}
}

bool FastllmCudaNaiveExpertPrefill(int device, const uint16_t *input,
        const float *scores, int tokens, int topk, int hidden, int intermediate,
        const std::vector<FastllmNaiveFP8ExpertTask> &tasks,
        const float *siluLookup, float *perRouteOutput) {
    if (tasks.empty() || hidden % 128 || intermediate % 128) return false;
    std::lock_guard<std::mutex> guard(Mutex());
    if (cudaSetDevice(device) != cudaSuccess) return false;
    auto &entry = Workspaces()[device];
    if (!entry) entry.reset(new Workspace(device));
    auto &w = *entry;
    if (!w.Init()) return false;
    // On every failure, finish already submitted copies before the caller
    // retries on CPU or releases its input and routing buffers.
    struct Drain { cudaStream_t stream; ~Drain() { cudaStreamSynchronize(stream); } } drain{w.stream};
    std::vector<int> routes;
    int maxRows = 0;
    for (const auto &task : tasks) {
        maxRows = std::max(maxRows, (int)task.routes.size());
        routes.insert(routes.end(), task.routes.begin(), task.routes.end());
    }
    int maxInput = std::max(hidden, intermediate);
    size_t weightElements = (size_t)hidden * intermediate * 2;
    size_t outputBytes = routes.size() * hidden * sizeof(float);
    bool ready = w.input.Reserve((size_t)tokens * hidden * 2) &&
        w.scores.Reserve((size_t)tokens * topk * 4) && w.routes.Reserve(routes.size() * 4) &&
        w.lookup.Reserve(65536 * sizeof(float)) && w.packed.Reserve(weightElements / 128 * 132) &&
        w.weight.Reserve(weightElements * 2) && w.weightScales.Reserve(weightElements / 128 * 4) &&
        w.activation.Reserve((size_t)maxRows * maxInput * 2) &&
        w.activationScales.Reserve((size_t)maxRows * (maxInput / 128) * 4) &&
        w.parts.Reserve((size_t)maxRows * weightElements / 128 * 4) &&
        w.gateUp.Reserve((size_t)maxRows * intermediate * 2 * 4) &&
        w.output.Reserve(outputBytes) && w.hostOutput.Reserve(outputBytes);
    if (!ready) return false;
    auto copy = [&](void *dst, const void *src, size_t bytes) {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, w.stream) == cudaSuccess;
    };
    if (!copy(w.input.ptr, input, (size_t)tokens * hidden * 2) ||
        !copy(w.scores.ptr, scores, (size_t)tokens * topk * 4) ||
        !copy(w.routes.ptr, routes.data(), routes.size() * 4) ||
        !copy(w.lookup.ptr, siluLookup, 65536 * sizeof(float))) return false;
    int offset = 0;
    for (const auto &task : tasks) {
        int n = task.routes.size();
        if (!n) continue;
        const int *ids = w.routes.As<int>() + offset;
        QuantizeRows<false><<<dim3(hidden / 128, n), 128, 0, w.stream>>>(
            w.input.As<BF16>(), nullptr, ids, nullptr,
            w.activation.As<BF16>(), w.activationScales.As<float>(), topk, hidden);
        if (!Gemm(w, task.gateWeight, n, hidden, intermediate * 2)) return false;
        SumBlocks<false><<<dim3((intermediate * 2 + 255) / 256, n), 256, 0, w.stream>>>(
            w.parts.As<float>(), w.activationScales.As<float>(), w.weightScales.As<float>(),
            w.gateUp.As<float>(), nullptr, nullptr, n, intermediate * 2, hidden / 128);
        QuantizeRows<true><<<dim3(intermediate / 128, n), 128, 0, w.stream>>>(
            nullptr, w.gateUp.As<float>(), ids, w.lookup.As<float>(),
            w.activation.As<BF16>(), w.activationScales.As<float>(), topk, intermediate);
        if (!Gemm(w, task.downWeight, n, intermediate, hidden)) return false;
        SumBlocks<true><<<dim3((hidden + 255) / 256, n), 256, 0, w.stream>>>(
            w.parts.As<float>(), w.activationScales.As<float>(), w.weightScales.As<float>(),
            w.output.As<float>() + (size_t)offset * hidden, w.scores.As<float>(), ids,
            n, hidden, intermediate / 128);
        offset += n;
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(w.hostOutput.ptr, w.output.ptr, outputBytes, cudaMemcpyDeviceToHost, w.stream) != cudaSuccess ||
        cudaStreamSynchronize(w.stream) != cudaSuccess) return false;
    const float *result = w.hostOutput.As<float>();
    for (size_t row = 0; row < routes.size(); ++row)
        std::memcpy(perRouteOutput + (size_t)routes[row] * hidden, result + row * hidden, hidden * sizeof(float));
    return true;
}

void FastllmCudaNaiveClearExpertPrefill() {
    std::lock_guard<std::mutex> guard(Mutex());
    if (Workspaces().empty()) return;
    int previous = 0; cudaGetDevice(&previous);
    Workspaces().clear();
    cudaSetDevice(previous);
}
