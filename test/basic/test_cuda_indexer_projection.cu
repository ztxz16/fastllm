#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;
using BF = __nv_bfloat16;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Check(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
static uint16_t Round(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}
static float Float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static void Allocate(Data &data, std::vector<int> dims) {
    data.Resize(dims);
    data.ToDevice(DataDevice::CUDA, {0}, false);
    data.Allocate();
}
__global__ void Fill(BF *values, int count, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned x = (i + seed) * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
    x = (x >> 22) ^ x;
    values[i] = __float2bfloat16_rn(((int)(x >> 16) - 32768) / 32768.f);
}
__global__ void ToDouble(const BF *input, double *output, int count, bool absolute) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        double value = __bfloat162float(input[i]);
        output[i] = absolute ? fabs(value) : value;
    }
}

int main() {
    try {
        FastllmCudaSetDevice(0);
        if (FastllmCudaRuntimeArch() != 90) {
            std::cout << "SKIP: Hopper Indexer projection" << std::endl;
            return 0;
        }
        Data wq(BFLOAT16), wk(BFLOAT16), ws(BFLOAT16);
        std::vector<uint16_t> original;
        unsigned seed = 17;
        for (auto item : {std::make_pair(&wq, 2048), std::make_pair(&wk, 128), std::make_pair(&ws, 16)}) {
            Data &w = *item.first;
            Allocate(w, {item.second, 4096});
            Fill<<<(w.Count(0) + 255) / 256, 256>>>((BF *)w.cudaData, w.Count(0), seed++);
            std::vector<uint16_t> values(w.Count(0));
            Check(cudaMemcpy(values.data(), w.cudaData, w.GetBytes(), cudaMemcpyDeviceToHost));
            original.insert(original.end(), values.begin(), values.end());
        }
        Data input(BFLOAT16), query(BFLOAT16), key(BFLOAT16), scores(BFLOAT16), scratch(BFLOAT16);
        // One extra row exercises the large-batch fallback with real storage.
        constexpr int capacity = 257;
        Allocate(input, {1, capacity, 4096});
        Fill<<<capacity * 16, 256>>>((BF *)input.cudaData, capacity * 4096, 713);
        Allocate(query, {1, capacity, 2048}); Allocate(key, {1, capacity, 128});
        Allocate(scores, {1, capacity, 16});
        // The model lends its already allocated O-projection output as scratch.
        Allocate(scratch, {1, capacity, 4096});
        input.Resize({1, 256, 4096});
        auto project = [&]() { return FastllmCudaIndexerProjection(input, wq, wk, ws, query, key, scores, scratch); };
        Require(!project(), "unpacked weights must fall back");
        ws.dataType = FLOAT16;
        Require(!FastllmCudaPackIndexerWeights(wq, wk, ws), "mixed storage must fall back");
        ws.dataType = BFLOAT16;
        Require(FastllmCudaPackIndexerWeights(wq, wk, ws), "pack failed");
        void *owner = wq.cudaData;
        Require(FastllmCudaPackIndexerWeights(wq, wk, ws) && owner == wq.cudaData, "pack must be idempotent");
        Require(!wq.cudaDataBorrowed && wk.cudaDataBorrowed && ws.cudaDataBorrowed &&
            wq.expansionBytes == original.size() * 2, "packed ownership/size");
        std::vector<uint16_t> packed(original.size());
        Check(cudaMemcpy(packed.data(), owner, packed.size() * 2, cudaMemcpyDeviceToHost));
        Require(original == packed, "packing changed weights");
        const void *pointers[] = {query.cudaData, key.cudaData, scores.cudaData, scratch.cudaData};
        input.Resize({1, 257, 4096});
        Require(!project() && query.dims[1] == capacity, "large batch fallback changed output");
        input.Resize({1, 256, 4096});
        input.dataType = FLOAT16;
        Require(!project(), "FP16 input must fall back");
        input.dataType = BFLOAT16;
        ++input.strides[1];
        Require(!project(), "strided input must fall back");
        --input.strides[1];
        void *aligned = input.cudaData;
        input.cudaData = (uint8_t *)aligned + 2;
        Require(!project(), "unaligned input must fall back");
        input.cudaData = aligned;

        double *wd, *xd, *reference, *absolute;
        Check(cudaMalloc(&wd, original.size() * sizeof(double)));
        Check(cudaMalloc(&xd, capacity * 4096 * sizeof(double)));
        Check(cudaMalloc(&reference, capacity * 2192 * sizeof(double)));
        Check(cudaMalloc(&absolute, capacity * 2192 * sizeof(double)));
        auto handle = getFastllmCublasHandle();
        std::shared_ptr<void> lease;
        FastllmCudaLinearGraphScope scope(lease);
        std::vector<int> batches{256, 128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1};
        std::vector<cudaGraphExec_t> graphs;
        for (int rows : batches) {
            input.Resize({1, rows, 4096});
            Require(project(), "warmup declined supported batch");
            Check(cudaDeviceSynchronize());
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(project(), "capture declined supported batch");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            Check(cudaGraphDestroy(graph));
            graphs.push_back(exec);
            Require(pointers[0] == query.cudaData && pointers[1] == key.cudaData &&
                pointers[2] == scores.cudaData && pointers[3] == scratch.cudaData, "batch switch reallocated buffers");
        }
        double worst = 0;
        int checks = 0;
        // Replay changed inputs, then verify the original Linear + Mul path
        // against the same FP64 reference using the packed weight slices.
        for (int trial = 0; trial < 3; ++trial)
        for (int g = (int)graphs.size() - 1; g >= (trial == 2 ? -1 : 0); --g) {
            const int rows = g < 0 ? capacity : batches[g];
            Fill<<<(rows * 4096 + 255) / 256, 256>>>((BF *)input.cudaData, rows * 4096, 713 + trial);
            if (trial < 2) Check(cudaGraphLaunch(graphs[g], cudaStreamPerThread));
            else {
                input.Resize({1, rows, 4096});
                query.Resize({1, rows, 2048}); key.Resize({1, rows, 128}); scores.Resize({1, rows, 16});
                Data bias;
                Require(FastllmCudaBFloat16MatMulBFloat16(input, wq, bias, query, rows, 4096, 2048) &&
                    FastllmCudaBFloat16MatMulBFloat16(input, wk, bias, key, rows, 4096, 128) &&
                    FastllmCudaBFloat16MatMulBFloat16(input, ws, bias, scores, rows, 4096, 16) &&
                    FastllmCudaMul(scores, .25f, scores), "separate Linear fallback failed");
            }
            const double alpha = 1, beta = 0;
            for (bool abs : {false, true}) {
                ToDouble<<<(original.size() + 255) / 256, 256>>>((BF *)owner, wd, original.size(), abs);
                ToDouble<<<(rows * 4096 + 255) / 256, 256>>>((BF *)input.cudaData, xd, rows * 4096, abs);
                Require(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2192, rows, 4096,
                    &alpha, wd, 4096, xd, 4096, &beta, abs ? absolute : reference, 2192) == CUBLAS_STATUS_SUCCESS,
                    "FP64 reference failed");
            }
            std::vector<double> ref(rows * 2192), sumAbs(ref.size());
            Check(cudaMemcpy(ref.data(), reference, ref.size() * sizeof(double), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(sumAbs.data(), absolute, ref.size() * sizeof(double), cudaMemcpyDeviceToHost));
            int start = 0;
            for (auto item : {std::make_pair(&query, 2048), std::make_pair(&key, 128), std::make_pair(&scores, 16)}) {
                std::vector<uint16_t> actual(rows * item.second);
                Check(cudaMemcpy(actual.data(), item.first->cudaData, actual.size() * 2, cudaMemcpyDeviceToHost));
                double error = 0, norm = 0;
                for (int row = 0; row < rows; ++row) for (int col = 0; col < item.second; ++col) {
                    const int i = row * 2192 + start + col;
                    uint16_t expected = Round((float)ref[i]);
                    const float scale = start == 2176 ? .25f : 1.f;
                    expected = Round(Float(expected) * scale);
                    const auto bits = actual[row * item.second + col];
                    const double delta = Float(bits) - Float(expected);
                    // Different FP32 reduction trees can straddle a BF16 rounding
                    // boundary. Near cancellation, bound absolute roundoff by
                    // eight FP32 epsilons times the sum of absolute products.
                    Require(std::isfinite(Float(bits)) &&
                        (std::abs((int)bits - (int)expected) <= 1 ||
                         std::abs(delta) <= 8 * 0x1p-23 * sumAbs[i] * scale), "element precision");
                    error += delta * delta; norm += (double)Float(expected) * Float(expected);
                }
                const double relative = std::sqrt(error / std::max(norm, 1e-30));
                Require(relative < .001, "projection relative precision");
                worst = std::max(worst, relative);
                ++checks; start += item.second;
            }
        }
        for (auto graph : graphs) Check(cudaGraphExecDestroy(graph));
        Check(cudaFree(wd)); Check(cudaFree(xd)); Check(cudaFree(reference)); Check(cudaFree(absolute));
        std::cout << "INDEXER PROJECTION PASS checks=" << checks << " graphs=" << graphs.size()
                  << " max_nrmse=" << worst << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
