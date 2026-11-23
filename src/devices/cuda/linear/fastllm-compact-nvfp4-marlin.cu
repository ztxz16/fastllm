// Decode view of compact NVFP4 weights. Keep the original allocation for
// prefill and preserve each merged projection's FP32 global scale.
#if !defined(USE_ROCM) && !defined(CUDA_NO_TENSOR_CORE)
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-nvfp4-prefill.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>

#define MARLIN_NAMESPACE_NAME fastllm_compact_marlin
#include "marlin_dense_fp8/kernel.h"
#include "marlin_dense_fp8/marlin_template.h"

namespace {
constexpr int tileN = 128;
using Kernel = void (*)(MARLIN_KERNEL_PARAMS);
Kernel DecodeKernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return nullptr;
#else
    return fastllm_compact_marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
            vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, 4, 1, false,
            true, false, false, false, false, false, false, false, false, true>;
#endif
}

void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

using Buffer = std::shared_ptr<uint8_t>;
Buffer Allocate(size_t bytes, int device) {
    uint8_t *ptr = nullptr;
    cudaError_t rc = cudaMalloc(&ptr, bytes);
    if (rc == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        return {};
    }
    Check(rc);
    return Buffer(ptr, [device](uint8_t *p) {
        int previous = device;
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
        cudaFree(p);
        if (previous != device) cudaSetDevice(previous);
    });
}

// Bit-exact conversion of positive normal E4M3 (or zero) to Marlin's
// S0E5M3(scale * 128). Reject scales Marlin would otherwise flush or clip.
__global__ void PrepareCompact(const uint8_t *src, uint32_t *q, uint8_t *s,
        int *unsupported, int n, int k) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pitch = 4 + size_t(k / 16) * 9;
    if (i < size_t(n) * (k / 8)) {
        int col = i % n, pack = i / n;
        const uint8_t *p = src + col * pitch + 4 + (pack / 2) * 9 + (pack % 2) * 4;
        q[i] = unsigned(p[0]) | (unsigned(p[1]) << 8) | (unsigned(p[2]) << 16) | (unsigned(p[3]) << 24);
    }
    if (i < size_t(n) * (k / 16)) {
        int pos = i % 64;
        int perm = (pos & ~3) + ((pos & 1) << 1) + ((pos & 2) >> 1);
        size_t idx = (i & ~size_t(63)) + (perm >> 3) + 8 * (perm & 7);
        uint8_t raw = src[(idx % n) * pitch + 4 + (idx / n) * 9 + 8];
        if (raw && (raw < 8 || raw >= 127)) atomicExch(unsupported, 1);
        s[i] = raw ? raw + 120 : 0;
    }
}

struct DeviceState {
    int sms = 0, shared = 0;
    std::map<const fastllm::Data*, Buffer> weights;
    std::map<uint64_t, Buffer> workspaces;
};

Buffer Repack(const fastllm::Data &weight, const FastllmNvfp4ScaleRuns &runs,
        int device, int n, int k) {
    std::vector<float> globals;
    for (auto &run : runs) {
        float scaled = std::ldexp(run.scale, 119);  // BF16 Marlin exponent compensation
        if (run.endRow % tileN || !std::isfinite(scaled) || scaled <= 0) return {};
        globals.resize(run.endRow / tileN, scaled);
    }
    if (globals.size() != n / tileN) return {};
    const size_t qBytes = size_t(n) * k / 2, scaleBytes = qBytes / 8;
    auto packed = Allocate(qBytes + scaleBytes + globals.size() * sizeof(float), device);
    auto tmp = Allocate(qBytes + sizeof(int), device);
    if (!packed || !tmp) return {};
    int *invalid = reinterpret_cast<int *>(tmp.get() + qBytes);
    Check(cudaMemsetAsync(invalid, 0, sizeof(int), cudaStreamPerThread));
    size_t count = qBytes / sizeof(uint32_t);
    PrepareCompact<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), reinterpret_cast<uint32_t *>(tmp.get()),
        packed.get() + qBytes, invalid, n, k);
    Check(cudaGetLastError());
    int rejected = 0;
    Check(cudaMemcpyAsync(&rejected, invalid, sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    if (rejected) return {};
    if (!FastllmCudaGptqMarlinRepackBitsStream(reinterpret_cast<uint32_t *>(tmp.get()),
            reinterpret_cast<uint32_t *>(packed.get()), k, n, 4, cudaStreamPerThread)) return {};
    Check(cudaMemcpyAsync(packed.get() + qBytes + scaleBytes, globals.data(), globals.size() * sizeof(float),
                          cudaMemcpyHostToDevice, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    return packed;
}
}  // namespace

class FastllmCompactMarlinCache {
public:
    std::mutex mutex;
    std::map<int, DeviceState> devices;
};

std::shared_ptr<FastllmCompactMarlinCache> FastllmCreateCompactMarlinCache() {
    return std::make_shared<FastllmCompactMarlinCache>();
}

bool FastllmCudaCompactMarlinLinear(FastllmCompactMarlinCache &cache, const FastllmNvfp4PrefillPlan &plan,
        const fastllm::Data &weight, const void *input, void *output, int rows, int n, int k) {
    if (rows < 1 || n < 4096 || n % tileN || (k != 4096 && k != 12288)) return false;
    int device;
    Check(cudaGetDevice(&device));
    std::lock_guard<std::mutex> guard(cache.mutex);
    auto &state = cache.devices[device];
    if (!state.shared) {
        cudaDeviceProp props;
        Check(cudaGetDeviceProperties(&props, device));
        if (props.major != 12 || props.minor != 0) { state.shared = -1; return false; }
        Check(cudaFuncSetAttribute(DecodeKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   props.sharedMemPerBlockOptin));
        state.sms = props.multiProcessorCount;
        state.shared = props.sharedMemPerBlockOptin;
    }
    if (state.shared < 0) return false;
    auto found = state.weights.find(&weight);
    if (found == state.weights.end()) {
        auto runs = plan.find(&weight);
        if (runs == plan.end()) return false;
        cudaStreamCaptureStatus capture;
        Check(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
        if (capture != cudaStreamCaptureStatusNone) return false;
        found = state.weights.emplace(&weight, Repack(weight, runs->second, device, n, k)).first;
    }
    if (!found->second) return false;
    // Each worker launches on its per-thread stream; immutable weights may be
    // shared, but simultaneous split-K reductions need independent scratch.
    static std::atomic<uint64_t> nextWorker{0};
    static thread_local const uint64_t worker = nextWorker++;
    auto &ws = state.workspaces[worker];
    size_t tmpBytes = size_t(state.sms) * 8 * 256 * sizeof(float);
    if (!ws) {
        cudaStreamCaptureStatus capture;
        Check(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
        if (capture != cudaStreamCaptureStatusNone) return false;
        ws = Allocate(tmpBytes + state.sms * 4 * sizeof(int), device);
        if (!ws) return false;
        Check(cudaMemsetAsync(ws.get() + tmpBytes, 0, state.sms * 4 * sizeof(int), cudaStreamPerThread));
    }
    const uint8_t *packed = found->second.get();
    size_t qBytes = size_t(n) * k / 2;
    for (int row = 0; row < rows; row += 8) {
        auto *x = static_cast<const uint16_t *>(input) + size_t(row) * k;
        auto *y = static_cast<uint16_t *>(output) + size_t(row) * n;
        DecodeKernel()<<<state.sms, 256, state.shared, cudaStreamPerThread>>>(
            reinterpret_cast<const int4 *>(x), reinterpret_cast<const int4 *>(packed), reinterpret_cast<int4 *>(y),
            reinterpret_cast<int4 *>(ws.get()), nullptr, nullptr, reinterpret_cast<const int4 *>(packed + qBytes),
            reinterpret_cast<const float *>(packed + qBytes + qBytes / 8), nullptr, nullptr, k / 16,
            std::min(8, rows - row), n, k, k, reinterpret_cast<int *>(ws.get() + tmpBytes), false, false, true, state.shared);
        Check(cudaPeekAtLastError());
    }
    return true;
}
#endif
