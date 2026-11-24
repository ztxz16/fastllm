// Check both grouped projections against a CPU dot product using the production Marlin packer.
#include "../../src/models/inkling2/runtime.cuh"
#include "../../src/devices/cuda/moe/fastllm-grouped-sm120.cuh"
#include "devices/cuda/fastllm-cuda-moe-activation.h"
#include <cuda_fp8.h>
#include <random>

using namespace fastllm;
using inkling2::Buffer;
using inkling2::check;
namespace mx = cuda::routed_mxfp8;
namespace grouped = cuda::grouped_sm120;

namespace {
constexpr int experts = 6, hidden = 256, intermediate = 128;
constexpr int offsets[] = {0, 0, 1, 128, 256, 385, 682}; // Empty, partial and multiple scale tiles.

template<class T> Buffer<T> upload(const std::vector<T> &host) {
    Buffer<T> buffer(host.size());
    check(cudaMemcpy(buffer.p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return buffer;
}

struct Projection {
    int n, k;
    std::vector<float> weights;
    Buffer<uint8_t> device;
    std::vector<std::unique_ptr<Data>> expertsData;

    Projection(int n, int k, std::mt19937 &random) : n(n), k(k), weights(size_t(experts) * n * k) {
        std::vector<uint8_t> raw(weights.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            weights[i] = int(random() % 7) - 3;
            raw[i] = __nv_fp8_e4m3(weights[i]).__x;
        }
        device = upload(raw);
        for (int e = 0; e < experts; ++e) {
            const size_t base = size_t(e) * n * k;
            auto w = std::make_unique<Data>(FP8_E4M3, std::vector<int>{n, k}, CUDA, device.p + base);
            w->blockM = 32;
            w->blockK = 1;
            w->scales.resize(n * k / 32);
            for (size_t group = 0; group < w->scales.size(); ++group) {
                float scale = std::ldexp(1.f, int(random() % 4) - 1);
                w->scales[group] = scale;
                for (int j = 0; j < 32; ++j) weights[base + group * 32 + j] *= scale;
            }
            expertsData.push_back(std::move(w));
        }
    }
};

void verify(const Projection &w, CudaMarlinMxfp8Weights packed, int smCount, std::mt19937 &random) {
    constexpr int rows = offsets[experts];
    const int groups = w.k / 32;
    std::vector<uint8_t> raw(rows * w.k), scales(grouped::scaleRows(rows, experts) * groups);
    std::vector<float> input(raw.size());
    for (int e = 0; e < experts; ++e) {
        for (int r = offsets[e]; r < offsets[e + 1]; ++r) {
            const size_t sfRow = grouped::scaleRow(offsets[e], e, r - offsets[e]);
            for (int g = 0; g < groups; ++g) {
                const int exponent = int(random() % 4) - 1;
                scales[grouped::scaleOffset(sfRow, g, groups)] = exponent + 127;
                for (int j = 0; j < 32; ++j) {
                    float v = int(random() % 7) - 3;
                    const int i = r * w.k + g * 32 + j;
                    raw[i] = __nv_fp8_e4m3(v).__x;
                    input[i] = std::ldexp(v, exponent);
                }
            }
        }
    }
    auto a = upload(raw), sf = upload(scales);
    auto starts = upload(std::vector<int>(offsets, offsets + experts + 1));
    Buffer<uint8_t> unpacked(w.weights.size() + w.weights.size() / 32);
    Buffer<__nv_bfloat16> output(rows * w.n);
    const auto scratch = mx::scratchSize(experts, smCount);
    Buffer<uint8_t> metadata(scratch.metadata), workspace(scratch.gemm);
    check(mx::project(packed, {a.p, sf.p, unpacked.p, output.p, metadata.p, workspace.p},
                      starts.p, w.n, w.k, experts, smCount, cudaStreamPerThread));
    std::vector<__nv_bfloat16> actual(rows * w.n);
    check(cudaMemcpy(actual.data(), output.p, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
    for (int e = 0; e < experts; ++e) {
        for (int r = offsets[e]; r < offsets[e + 1]; ++r) {
            for (int col = 0; col < w.n; ++col) {
                float sum = 0;
                for (int k = 0; k < w.k; ++k)
                    sum += input[r * w.k + k] * w.weights[(e * w.n + col) * w.k + k];
                if (float(actual[r * w.n + col]) != float(__float2bfloat16(sum)))
                    throw std::runtime_error("MXFP8 projection differs from CPU at row " +
                        std::to_string(r) + ", column " + std::to_string(col));
            }
        }
    }
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    cudaDeviceProp device;
    check(cudaGetDeviceProperties(&device, 0));
    if (device.major != 12 || device.minor != 0) return 77;
    std::mt19937 random(707);
    Projection gate(2 * intermediate, hidden, random), down(hidden, intermediate, random);
    std::vector<Data *> weights(2 + 2 * experts);
    for (int e = 0; e < experts; ++e) {
        weights[2 + 2 * e] = gate.expertsData[e].get();
        weights[3 + 2 * e] = down.expertsData[e].get();
    }
    auto packed = PrepareCudaGroupedMxfp8Moe(weights.data(), weights.size());
    if (!packed) throw std::runtime_error("MXFP8 Marlin preparation failed");
    verify(gate, packed->MarlinWeights(false), device.multiProcessorCount, random);
    verify(down, packed->MarlinWeights(true), device.multiProcessorCount, random);
}
