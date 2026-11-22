#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "devices/cuda/fastllm-linear-scratch.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Require(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}
static void Check(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
static float BFloat16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}
static float E4M3(uint8_t bits) {
    int exp = (bits & 127) >> 3;
    float x = exp ? std::ldexp(1.f + (bits & 7) / 8.f, exp - 7) : std::ldexp(float(bits & 7), -9);
    return bits & 128 ? -x : x;
}
// Independent RN-even FP8 reference. Only tested input blocks are materialized.
static float RoundE4M3(float x) {
    static const auto codes = [] {
        std::vector<float> v(127);
        for (int i = 0; i < 127; i++)
            v[i] = E4M3(i);
        return v;
    }();
    float a = std::abs(x);
    auto it = std::lower_bound(codes.begin(), codes.end(), a);
    int i = std::min(int(it - codes.begin()), 126);
    if (i && (a - codes[i - 1] < codes[i] - a || (a - codes[i - 1] == codes[i] - a && (i & 1)))) --i;
    return std::copysign(codes[i], x);
}

static void Run(int m, int n, int k, bool invalid = false, bool wideGemv = false, bool borrowScratch = false) {
    std::mt19937 rng(24816);
    std::vector<float> values(size_t(m) * k);
    // All inputs are exactly representable in BF16; ranges differ by both row and K32 block.
    for (size_t i = 0; i < values.size(); i++)
        values[i] = std::ldexp(float(int(rng() % 193) - 96), int((i / 32 + i / k) % 9) - 8);
    Data x(FLOAT32, {m, k}, values), y(BFLOAT16, {m, n}), bias;
    ToDataType(x, BFLOAT16);
    x.ToDevice(CUDA, {0}, true);
    y.ToDevice(CUDA, {0}, false);
    y.Allocate(false);
    Data w(FP8_E4M3, {n, k});
    w.blockM = 32;
    w.blockK = 1;
    w.Allocate();
    std::vector<uint8_t> bytes(size_t(n) * k);
    for (auto &b : bytes)
        b = uint8_t((rng() % 120) | ((rng() % 2) << 7));
    std::memcpy(w.cpuData, bytes.data(), bytes.size());
    w.scales.resize(bytes.size() / 32);
    for (size_t i = 0; i < w.scales.size(); ++i)
        w.scales[i] = std::ldexp(1.f, int(i % 11) - 10);
    if (invalid) w.scales.back() = 1.25f;
    w.ToDevice(CUDA, {0}, true);
    auto call = [&] {
        Require(FastllmCudaBFloat16MatMulFP8E4M3(x, w, bias, y, m, k, n), "public entry failed");
    };
    Data scratch;
    if (borrowScratch) {
        scratch.Resize({3 << 20});
        scratch.ToDevice(CUDA, {0}, false);
        scratch.Allocate(false);
    }
    CudaLinearScratchScope loan(borrowScratch ? &scratch : nullptr);
    call(); // upload legacy scales and validate metadata before capture
    bool supported = FastllmCudaMxfp8LinearSM103(x.cudaData, w, y.cudaData, m, k, n);
    const bool quantized = !invalid && (m > 1 || wideGemv);
    Require(supported == quantized, "backend eligibility mismatch");
    Check(cudaDeviceSynchronize());
    if (borrowScratch) {
        // A serving worker may own prefill scratch without a thread-local cuBLASLt lease.
        std::async(std::launch::async, [&] {
            Check(cudaSetDevice(0));
            CudaLinearScratchScope workerLoan(&scratch);
            Check(cudaMemsetAsync(y.cudaData, 0, y.GetBytes(), cudaStreamPerThread));
            Require(FastllmCudaMxfp8LinearSM103(x.cudaData, w, y.cudaData, m, k, n),
                    "prefill worker rejected its own scratch");
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }).get();
    }
    std::vector<uint16_t> eager;
    if (m >= 256 && n % 256 == 0) {
        eager.resize(size_t(m) * n);
        Check(cudaMemcpy(eager.data(), y.cudaData, eager.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    call();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiateWithFlags(&exec, graph, 0));
    for (int iteration = 0; iteration < 2; ++iteration) {
        if (iteration) {
            for (float &v : values)
                v = -v;
            Data replacement(FLOAT32, {m, k}, values);
            ToDataType(replacement, BFLOAT16);
            replacement.ToDevice(CUDA, {0}, true);
            Check(cudaMemcpy(x.cudaData, replacement.cudaData, values.size() * 2, cudaMemcpyDeviceToDevice));
        }
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<uint16_t> output(size_t(m) * n);
        Check(cudaMemcpy(output.data(), y.cudaData, output.size() * 2, cudaMemcpyDeviceToHost));
        if (!eager.empty() && !iteration) {
            double diff = 0, reference = 0;
            for (size_t j = 0; j < eager.size(); ++j) {
                const double expected = BFloat16(output[j]);
                const double delta = BFloat16(eager[j]) - expected;
                diff += delta * delta;
                reference += expected * expected;
            }
            const double relative = std::sqrt(diff / std::max(reference, 1e-30));
            std::printf("eager versus graph, full output rel_l2=%.8f\n", relative);
            Require(relative < 0.004, "eager output disagrees with captured projection");
        }
        double error = 0, norm = 0, bf16Error = 0, bf16Norm = 0;
        for (int row : {0, m / 2, m - 1}) {
            std::vector<float> a(values.begin() + size_t(row) * k, values.begin() + size_t(row + 1) * k);
            if (quantized)
                for (int block = 0; block < k; block += 32) {
                    float maximum = 0;
                    for (int i = 0; i < 32; i++)
                        maximum = std::max(maximum, std::abs(a[block + i]));
                    float scale = maximum == 0 ? 1.f : std::exp2(std::ceil(std::log2(maximum / 448.f)));
                    for (int i = 0; i < 32; i++)
                        a[block + i] = RoundE4M3(a[block + i] / scale) * scale;
                }
            for (int col = 0; col < n; col += std::max(1, n / 64)) {
                double expected = 0, bf16Expected = 0;
                for (int i = 0; i < k; i++) {
                    size_t index = size_t(col) * k + i;
                    const double weight = double(E4M3(bytes[index])) * w.scales[index / 32];
                    expected += a[i] * weight;
                    bf16Expected += values[size_t(row) * k + i] * weight;
                }
                float actual = BFloat16(output[size_t(row) * n + col]);
                Require(std::isfinite(actual), "nonfinite output");
                error += (actual - expected) * (actual - expected);
                norm += expected * expected;
                bf16Error += (actual - bf16Expected) * (actual - bf16Expected);
                bf16Norm += bf16Expected * bf16Expected;
            }
        }
        double l2 = std::sqrt(error / std::max(1e-30, norm));
        std::printf("M=%d N=%d K=%d invalid=%d replay=%d rel_l2=%.7f bf16_l2=%.7f\n", m, n, k, invalid,
                    iteration, l2, std::sqrt(bf16Error / std::max(bf16Norm, 1e-30)));
        Require(l2 < 0.005, "projection differs from independent reference");
    }
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
    if (m == 3) {
        // Changing placement invalidates the old device scales. Re-upload a
        // non-UE8M0 scale and verify the backend rejects the new allocation.
        w.ToDevice(CPU);
        Require(w.extraCudaData.empty() || !w.extraCudaData[0], "device move retained stale MXFP8 scales");
        w.scales.back() = 1.25f;
        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        const int target = devices > 1 ? 1 : 0;
        FastllmCudaSetDevice(target);
        std::shared_ptr<void> targetLease;
        FastllmCudaLinearGraphScope targetScope(targetLease);
        w.ToDevice(CUDA, {target}, true);
        x.ToDevice(CUDA, {target}, true);
        y.ToDevice(CUDA, {target}, false);
        y.Allocate(false);
        call();
        cudaPointerAttributes attributes{};
        Check(cudaPointerGetAttributes(&attributes, w.extraCudaData[0]));
        Require(attributes.device == target, "MXFP8 scales are on the wrong device");
        Require(!FastllmCudaMxfp8LinearSM103(x.cudaData, w, y.cudaData, m, k, n),
                "old scale validation survived device move");
        uint16_t output;
        Check(cudaMemcpy(&output, static_cast<uint16_t *>(y.cudaData) + n - 1, sizeof(output), cudaMemcpyDeviceToHost));
        double ref = 0;
        for (int i = 0; i < k; ++i) {
            const size_t index = size_t(n - 1) * k + i;
            ref += double(values[i]) * E4M3(bytes[index]) * w.scales[index / 32];
        }
        Require(std::abs(BFloat16(output) - ref) <= 0.005 * std::abs(ref) + 1.e-4, "fallback used stale scale values");
        FastllmCudaSetDevice(0);
    }
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    cudaDeviceProp prop;
    Check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || prop.minor != 3 || prop.multiProcessorCount != 148) return 77;
    try {
        FastllmCudaSetDevice(0);
        std::shared_ptr<void> lease;
        FastllmCudaLinearGraphScope scope(lease);
        Run(1, 256, 512);
        Run(1, 6784, 4096, false, true); // wide GEMV selects native MXFP8
        Run(3, 192, 256);
        Run(8, 1856, 4096);
        Run(32, 4096, 1024);
        Run(33, 320, 512);
        Run(129, 4160, 512); // partial M pair, 64-column tiles and an odd number of N tiles
        Run(256, 4096, 8192); // verification attention output projection
        Run(256, 8192, 512); // 128-column tiles fill a single wave
        Run(257, 8192, 512); // unpaired M tail at the same width
        Run(1933, 512, 512); // partially filled second M tile
        Run(2061, 4096, 4096); // bounded lease needs multiple GEMMs
        Run(2061, 4096, 4096, false, false, true); // bulk clusters plus a short tail
        Run(2053, 64, 4096); // crosses the workspace chunk boundary
        Run(2, 256, 256, true);
        std::puts("PASS Blackwell MXFP8 Linear and graph replay");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
