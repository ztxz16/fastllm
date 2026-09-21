#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime.h>
#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;
namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
Data Tensor(DataType type, const std::vector<int> &dims, float amplitude, int seed) {
    Data data(type, dims); data.Allocate(false);
    uint32_t state = seed;
    for (size_t i = 0; i < data.Count(0); ++i) {
        state = state * 1664525u + 1013904223u;
        float value = (float(int(state >> 8) - 8388608) / 8388608.f) * amplitude;
        if (type == FLOAT16) ((uint16_t *)data.cpuData)[i] = float_to_half(value);
        else ((float *)data.cpuData)[i] = value;
    }
    data.ToDevice(CUDA, {0}, true); return data;
}
std::vector<float> Read(const Data &data) {
    const size_t count = data.Count(0);
    std::vector<float> result(count);
    if (data.dataType == FLOAT32) {
        Require(cudaMemcpy(result.data(), data.cudaData, count * 4, cudaMemcpyDeviceToHost) == cudaSuccess, "read float");
    } else {
        std::vector<uint16_t> values(count);
        Require(cudaMemcpy(values.data(), data.cudaData, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "read half");
        for (size_t i = 0; i < count; ++i) result[i] = half_to_float(values[i]);
    }
    return result;
}
void Near(const std::vector<float> &a, const std::vector<float> &b, float atol, float rtol) {
    Require(a.size() == b.size(), "size mismatch");
    float maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        maximum = std::max(maximum, std::abs(a[i] - b[i]));
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) {
            std::fprintf(stderr, "mismatch at %zu: %.9g vs %.9g\n", i, a[i], b[i]);
            throw std::runtime_error("numeric mismatch");
        }
    }
    std::printf(" max_error=%g", maximum);
}
void Gdn(int chunks, int batch, DataType stateType) {
    constexpr int heads = 2, tile = 64, dim = 128;
    auto q = Tensor(FLOAT16, {batch, heads, chunks, tile, dim}, .02f, 1);
    auto k = Tensor(FLOAT16, q.dims, .02f, 2);
    auto v = Tensor(FLOAT16, q.dims, .03f, 3);
    auto kc = Tensor(FLOAT16, q.dims, .005f, 4);
    auto a = Tensor(FLOAT16, {batch, heads, chunks, tile, tile}, .003f, 5);
    auto g = Tensor(FLOAT16, {batch, heads, chunks, tile}, 0, 6);
    std::vector<uint16_t> decay(g.Count(0));
    for (size_t i = 0; i < decay.size(); ++i) decay[i] = float_to_half(-.001f * (i % tile + 1));
    Require(cudaMemcpy(g.cudaData, decay.data(), decay.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess, "decay init");
    auto referenceState = Tensor(stateType, {batch, heads, dim, dim}, .02f, 7);
    auto actualState = Tensor(stateType, referenceState.dims, .02f, 7);
    Require(FastllmCudaCanUseTritonChunkGdnPrefill(batch, chunks, tile, dim, dim, stateType == FLOAT32), "long GDN fast path rejected");
    Data reference, actual;
    FastllmChunkGatedDeltaRulePrefill(q, k, v, g, a, kc, referenceState, reference);
    static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:0");
    ChunkGatedDeltaRulePrefill(q, k, v, g, a, kc, actualState, actual);
    std::printf("GDN chunks=%d batch=%d state=%d output", chunks, batch, int(stateType));
    Near(Read(actual), Read(reference), 1e-4f, .01f);
    std::printf(" state"); Near(Read(actualState), Read(referenceState), 1e-4f, .01f);
    // Continue a second prefill from the computed state. This observes state
    // commits and persistent scratch reuse, not just one output tensor.
    FastllmChunkGatedDeltaRulePrefill(q, k, v, g, a, kc, referenceState, reference);
    ChunkGatedDeltaRulePrefill(q, k, v, g, a, kc, actualState, actual);
    std::printf(" continued_output"); Near(Read(actual), Read(reference), 1e-4f, .01f);
    std::printf(" continued_state"); Near(Read(actualState), Read(referenceState), 1e-4f, .01f);
    std::puts(" PASS");
}
int GdnLargeOffsets(DataType stateType) {
    // H's saved states cross 2^31 elements at head 32. Share the read-only
    // zero inputs to exercise the real kernels with less than 14 GiB of VRAM.
    constexpr int heads = 33, chunks = 4096, tile = 64, dim = 128;
    size_t freeBytes = 0, totalBytes = 0;
    Require(cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess, "query free VRAM");
    if (freeBytes < (16ULL << 30)) {
        std::puts("SKIP large GDN offsets: requires 16 GiB of free VRAM");
        return 77;
    }
    Require(FastllmCudaCanUseTritonChunkGdnPrefill(1, chunks, tile, dim, dim,
                                                 stateType == FLOAT32),
            "large GDN fast path rejected");
    auto gpuHalf = [](const std::vector<int> &dims, float value) {
        Data data(FLOAT16, dims);
        data.dataDevice = CUDA;
        data.dataDeviceIds = {0};
        data.Allocate(false);
        Require(cuMemsetD16((CUdeviceptr)data.cudaData, float_to_half(value),
                           data.Count(0)) == CUDA_SUCCESS, "initialize GPU tensor");
        return data;
    };
    auto q = gpuHalf({1, heads, chunks, tile, dim}, 1.f);
    auto zero = gpuHalf(q.dims, 0.f);
    auto attn = gpuHalf({1, heads, chunks, tile, tile}, 0.f);
    auto g = gpuHalf({1, heads, chunks, tile}, 0.f);
    Data state(stateType, {1, heads, dim, dim});
    state.Allocate(false);
    std::vector<float> expectedState(state.Count(0));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < dim * dim; ++i) {
            const size_t offset = size_t(h) * dim * dim + i;
            const float value = float(h + 1) / dim;
            expectedState[offset] = value;
            if (stateType == FLOAT32) ((float *)state.cpuData)[offset] = value;
            else ((uint16_t *)state.cpuData)[offset] = float_to_half(value);
        }
    }
    state.ToDevice(CUDA, {0}, true);
    Data output;
    static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:0");
    // Zero keys/values and zero log-decay preserve the state. With Q=1,
    // every output channel is the sum of one state column, exactly h+1.
    ChunkGatedDeltaRulePrefill(q, zero, zero, g, attn, zero, state, output);
    Require(cudaDeviceSynchronize() == cudaSuccess, "large GDN execution");
    for (int h = 0; h < heads; ++h) {
        for (int chunk : {0, 1, chunks - 1}) {
            for (int token : {0, tile - 1}) {
                const size_t offset = ((size_t(h) * chunks + chunk) * tile + token) * dim;
                uint16_t actual[dim];
                Require(cudaMemcpy(actual, (uint16_t *)output.cudaData + offset,
                                   sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess,
                        "read large GDN output");
                for (int d = 0; d < dim; ++d) {
                    if (half_to_float(actual[d]) != float(h + 1)) {
                        std::fprintf(stderr,
                            "large GDN head=%d chunk=%d token=%d channel=%d: %g vs %d\n",
                            h, chunk, token, d, half_to_float(actual[d]), h + 1);
                        throw std::runtime_error("large GDN output mismatch");
                    }
                }
            }
        }
    }
    Near(Read(state), expectedState, 0.f, 0.f);
    std::printf(" PASS GDN >2^31 state elements, state type=%d\n", int(stateType));
    return 0;
}
void Sparse(int group, int dim, int sequence, int keyLength, int width, bool padded) {
    constexpr int kvHeads = 2;
    const int heads = kvHeads * group;
    auto query = Tensor(FLOAT16, {heads, sequence + (padded ? 5 : 0), dim + (padded ? 3 : 0)}, .5f, 11);
    auto key = Tensor(FLOAT16, {kvHeads, keyLength + (padded ? 71 : 0), dim + (padded ? 5 : 0)}, .5f, 12);
    auto value = Tensor(FLOAT16, {kvHeads, keyLength + (padded ? 103 : 0), dim + (padded ? 7 : 0)}, .3f, 13);
    auto output = Tensor(FLOAT16, {heads, sequence + (padded ? 9 : 0), dim + (padded ? 9 : 0)}, 0, 14);
    for (Data *d : {&query, &key, &value, &output}) d->expansionDims = d->dims;
    query.Resize({heads, sequence, dim}); key.Resize({kvHeads, keyLength, dim});
    value.Resize(key.dims); output.Resize(query.dims);
    const int indexStride = width + (padded ? 7 : 0);
    Data indices(INT32, {sequence, indexStride}); indices.Allocate(false);
    std::vector<int> ids(sequence * indexStride, -1);
    for (int row = 0; row < sequence; ++row) {
        for (int j = 0; j < width; ++j) {
            int index = (row * 29 + j * 17) % keyLength;
            if (j % 11 == 0) index = -2;
            if (j % 13 == 0) index = keyLength + 3;
            if (row == 0 || (row == 1 && j > 0)) index = -1;
            if (row == 1 && j == 0) index = 0;
            ids[row * indexStride + j] = index;
        }
    }
    std::memcpy(indices.cpuData, ids.data(), ids.size() * 4);
    indices.ToDevice(CUDA, {0}, true); indices.expansionDims = indices.dims;
    indices.Resize({sequence, width});
    const auto q = Read(query), k = Read(key), v = Read(value);
    const float scale = 1.f / std::sqrt(float(dim));
    Require(FastllmCudaTryTritonQwen4SparseAttention(query, key, value, indices, group, scale, output), "sparse fast path rejected");
    const auto result = Read(output);
    std::vector<float> actual, expected;
    std::vector<double> scores(width), sums(dim);
    for (int h = 0; h < heads; ++h) for (int row = 0; row < sequence; ++row) {
        double maxScore = -1e30, denominator = 0;
        for (int j = 0; j < width; ++j) {
            const int index = ids[row * indexStride + j]; scores[j] = -1e30;
            if (index < 0 || index >= keyLength) continue;
            double dot = 0;
            for (int d = 0; d < dim; ++d) dot += double(q[h * query.strides[0] + row * query.strides[1] + d]) * k[(h / group) * key.strides[0] + index * key.strides[1] + d];
            scores[j] = dot * scale; maxScore = std::max(maxScore, scores[j]);
        }
        std::fill(sums.begin(), sums.end(), 0.0);
        for (int j = 0; j < width; ++j) {
            const int index = ids[row * indexStride + j];
            if (index < 0 || index >= keyLength) continue;
            const double probability = std::exp(scores[j] - maxScore); denominator += probability;
            for (int d = 0; d < dim; ++d) sums[d] += probability * v[(h / group) * value.strides[0] + index * value.strides[1] + d];
        }
        for (int d = 0; d < dim; ++d) {
            actual.push_back(result[h * output.strides[0] + row * output.strides[1] + d]);
            expected.push_back(denominator > 0 ? sums[d] / denominator : 0);
        }
    }
    std::printf("Sparse group=%d dim=%d sequence=%d keys=%d width=%d padded=%d", group, dim, sequence, keyLength, width, padded);
    Near(actual, expected, 4e-4f, .002f); std::puts(" PASS");
}
}
int main(int argc, char **argv) {
    // Triton is optional; opt into these compile/launch regressions explicitly.
    const char *enabled = std::getenv("FASTLLM_CUDA_TRITON");
    if (enabled == nullptr || std::strcmp(enabled, "1") != 0) return 77;
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    try {
        FastllmCudaSetDevice(0); SetThreads(2);
        if (argc >= 2 && std::strcmp(argv[1], "gdn_large_fp32") == 0)
            return GdnLargeOffsets(FLOAT32);
        if (argc >= 2 && std::strcmp(argv[1], "gdn_large_fp16") == 0)
            return GdnLargeOffsets(FLOAT16);
        if (argc < 2 || std::strcmp(argv[1], "gdn") == 0) {
            Gdn(64, 1, FLOAT32); Gdn(65, 1, FLOAT32); Gdn(66, 2, FLOAT32);
            Gdn(128, 1, FLOAT32); Gdn(257, 1, FLOAT32); Gdn(66, 1, FLOAT16);
        }
        if (argc < 2 || std::strcmp(argv[1], "sparse") == 0) {
            for (bool padded : {false, true}) {
                Sparse(1, 16, 3, 19, 7, padded);
                Sparse(3, 64, 17, 137, 67, padded);
                Sparse(4, 128, 33, 4196, 2048, padded);
                Sparse(16, 256, 5, 8211, 257, padded);
            }
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "prefill regression: %s\n", e.what()); return 1;
    }
    return 0;
}
