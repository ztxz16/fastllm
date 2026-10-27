#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace fastllm;
namespace {
using BF = __nv_bfloat16;
void Check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::abort(); }
}
void Require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
void Init(Data &data, const std::vector<int> &dims, int rank) {
    data.Resize(dims); data.ToDevice(DataDevice::CUDA, {rank}, false); data.Allocate(false);
}
double Round(double value) { return __bfloat162float(__float2bfloat16_rn((float)value)); }

__global__ void CheckOutput(const BF *input, const BF *source, const BF *hidden,
        const BF *expected, const BF *output, const BF *lower, const BF *upper,
        const BF *referenceNorm, bool exactNorm, int count, int *errors) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        float value = __bfloat162float(output[i]);
        if (((const unsigned short *)input)[i] != ((const unsigned short *)source)[i] ||
            ((const unsigned short *)hidden)[i] != ((const unsigned short *)expected)[i] ||
            (exactNorm && ((const unsigned short *)output)[i] != ((const unsigned short *)referenceNorm)[i]) ||
            !(value >= (float)lower[i] && value <= (float)upper[i])) atomicAdd(errors, 1);
    }
}

struct Case {
    int count, rank;
    bool gather;
    Data input{BFLOAT16}, hidden{BFLOAT16}, output{BFLOAT16}, weight{FLOAT32};
    Data source{BFLOAT16}, residual{BFLOAT16}, expected{BFLOAT16}, lower{BFLOAT16}, upper{BFLOAT16};
    Data referenceNorm{BFLOAT16};
    cudaGraphExec_t graph = nullptr;
    Case(int rows, int rank, bool gather) : count(rows * 4096), rank(rank), gather(gather) {
        for (Data *d : {&input, &hidden, &output, &source, &residual, &expected, &lower, &upper})
            Init(*d, {1, rows, 4096}, rank);
        if (gather) Init(referenceNorm, {1, rows, 4096}, rank);
        Init(weight, {4096}, rank);
        unsigned seed = 31 + rank + rows * 19;
        auto random = [&]() { seed = seed * 1664525u + 1013904223u; return (int(seed >> 16) - 32768) / 30000.f; };
        std::vector<BF> x(count), h(count), lo(count), hi(count);
        std::vector<float> w(4096);
        float scale = rows % 3 == 0 ? 1e-3f : rows % 3 == 1 ? 1e3f : 1.f;
        for (auto &v : x) v = __float2bfloat16_rn(random() * scale);
        for (auto &v : h) v = __float2bfloat16_rn(random() * scale);
        for (auto &v : w) v = 1.f + random();
        if (rows > 1) for (int i = 0; i < 4096; ++i) x[i] = h[i] = __float2bfloat16_rn(0);
        Check(cudaMemcpy(source.cudaData, x.data(), count * 2, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(residual.cudaData, h.data(), count * 2, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(weight.cudaData, w.data(), 4096 * 4, cudaMemcpyHostToDevice));
        Reset();
        Require(gather ? FastllmCudaCustomAllGatherRows(input.cudaData, count, 4096, BFLOAT16, rank)
            : FastllmCudaCustomAllReduceRows(input.cudaData, input.cudaData, count, 4096, BFLOAT16, rank),
            "reference collective rejected");
        FastllmCudaNaiveAddDecodeRMSNorm(hidden, input, weight, 1e-5f, output);
        if (gather)
            Check(cudaMemcpyAsync(referenceNorm.cudaData, output.cudaData, count * 2,
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        Check(cudaMemcpy(h.data(), hidden.cudaData, count * 2, cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(expected.cudaData, h.data(), count * 2, cudaMemcpyHostToDevice));
        // FP64 reference, with a conservative bound covering the FP32 sum,
        // division and rsqrt. Apply both BF16 rounding points to the interval;
        // this checks every output, including values close to a rounding tie.
        for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int c = 0; c < 4096; ++c) { double v = (float)h[row * 4096 + c]; sum += v * v; }
            double inverse = 1. / std::sqrt(sum / 4096. + (double)1e-5f);
            constexpr double bound = 32 * 1.1920928955078125e-7;
            for (int c = 0; c < 4096; ++c) {
                int i = row * 4096 + c;
                double v = (float)h[i];
                double a = Round(Round(v * inverse * (1 - bound)) * w[c]);
                double b = Round(Round(v * inverse * (1 + bound)) * w[c]);
                lo[i] = __float2bfloat16_rn(std::min(a, b));
                hi[i] = __float2bfloat16_rn(std::max(a, b));
            }
        }
        Check(cudaMemcpy(lower.cudaData, lo.data(), count * 2, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(upper.cudaData, hi.data(), count * 2, cudaMemcpyHostToDevice));
    }
    ~Case() { if (graph) cudaGraphExecDestroy(graph); }
    void Reset() {
        Check(cudaMemcpyAsync(input.cudaData, source.cudaData, count * 2, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        Check(cudaMemcpyAsync(hidden.cudaData, residual.cudaData, count * 2, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    }
    bool Fused(void *dest) {
        if (gather)
            return FastllmCudaCustomAllGatherAddRMSNorm(input.cudaData, hidden.cudaData,
                (const float *)weight.cudaData, dest, count, 4096, 1e-5f, BFLOAT16, rank);
        return FastllmCudaCustomAllReduceAddRMSNorm(input.cudaData, hidden.cudaData,
            (const float *)weight.cudaData, dest, count, 4096, 1e-5f, BFLOAT16, rank);
    }
    void Validate(int *errors) {
        CheckOutput<<<32, 256, 0, cudaStreamPerThread>>>((const BF *)input.cudaData,
            (const BF *)source.cudaData, (const BF *)hidden.cudaData, (const BF *)expected.cudaData,
            (const BF *)output.cudaData, (const BF *)lower.cudaData, (const BF *)upper.cudaData,
            (const BF *)referenceNorm.cudaData, gather, count, errors);
        Check(cudaGetLastError());
    }
    void Run(int *errors) {
        Reset(); Require(Fused(output.cudaData), "fusion rejected"); Validate(errors);
        // Immediately reuse the source and the shared collective scratch. The
        // following ordinary AR uses a different geometry from fused RMSNorm.
        Check(cudaMemsetAsync(input.cudaData, 0, count * 2, cudaStreamPerThread));
        Require(FastllmCudaCustomAllReduceRows(input.cudaData, input.cudaData, count, 4096, BFLOAT16, rank), "following AR rejected");
    }
};
}

int main(int argc, char **argv) {
    int ranks = argc > 1 ? std::atoi(argv[1]) : 8, available = 0;
    const bool gather = argc > 2 && std::string(argv[2]) == "gather";
    if (ranks != 2 && ranks != 4 && ranks != 6 && ranks != 8) return 2;
    if (cudaGetDeviceCount(&available) != cudaSuccess || available < ranks) return 77;
    std::vector<int> devices;
    for (int rank = 0; rank < ranks; ++rank) {
        Check(cudaSetDevice(rank));
        if (FastllmCudaRuntimeArch() != 90) return 77;
        for (int peer = 0; peer < ranks; ++peer) if (peer != rank) {
            int enabled = 0; Check(cudaDeviceCanAccessPeer(&enabled, rank, peer)); if (!enabled) return 77;
        }
        devices.push_back(rank);
    }
    setenv("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "1", 1);
    Require(FastllmInitNccl(devices) && FastllmCudaCustomAllReduceInit(devices), "initialization failed");
    std::vector<int> shapes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 36, 37, 64, 65, 127, 128, 129, 255, 256, 512, 1024};
    if (gather) {
        shapes.clear();
        for (int n = ranks; n <= 256; n += ranks) shapes.push_back(n);
        shapes.push_back(512 / ranks * ranks);
        shapes.push_back(1024 / ranks * ranks);
    }
    std::vector<std::thread> workers;
    std::vector<int> errors(ranks);
    for (int rank = 0; rank < ranks; ++rank) workers.emplace_back([&, rank]() {
        FastllmCudaSetDevice(rank);
        int *deviceErrors; Check(cudaMalloc(&deviceErrors, sizeof(int))); Check(cudaMemset(deviceErrors, 0, sizeof(int)));
        std::vector<std::unique_ptr<Case>> cases;
        for (int rows : shapes) {
            cases.emplace_back(new Case(rows, rank, gather)); auto &c = *cases.back();
            c.Reset(); Require(c.Fused(c.output.cudaData), "warmup rejected"); c.Validate(deviceErrors);
            Require(!c.Fused(rank == 0 ? (char *)c.output.cudaData + 2 : c.output.cudaData), "asymmetric alignment accepted");
            c.Validate(deviceErrors);
            Require(!c.Fused(rank == 0 ? c.hidden.cudaData : c.output.cudaData), "asymmetric alias accepted");
            c.Validate(deviceErrors);
            if (gather) {
                Require(!FastllmCudaCustomAllGatherAddRMSNorm(c.input.cudaData, c.hidden.cudaData,
                    (const float *)c.weight.cudaData, c.output.cudaData, c.count - 4096, 4096,
                    1e-5f, BFLOAT16, rank), "unequal row shards accepted");
                c.Validate(deviceErrors);
            }
            c.Run(deviceErrors); Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            c.Run(deviceErrors);
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&c.graph, graph, 0)); Check(cudaGraphDestroy(graph));
        }
        for (int replay = 0; replay < 32; ++replay) for (int i = 0; i < (int)cases.size(); ++i) {
            int index = replay % 2 ? (int)cases.size() - 1 - i : i;
            if ((replay + i + rank) % 11 == 0) std::this_thread::sleep_for(std::chrono::microseconds(23 * (rank + 1)));
            Check(cudaGraphLaunch(cases[index]->graph, cudaStreamPerThread));
        }
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaMemcpy(&errors[rank], deviceErrors, sizeof(int), cudaMemcpyDeviceToHost));
        Check(cudaFree(deviceErrors));
    });
    for (auto &worker : workers) worker.join();
    FastllmCudaCustomAllReduceReset();
    for (int n : errors) if (n) { std::fprintf(stderr, "FAIL: %d incorrect elements\n", n); return 1; }
    std::printf("PASS: TP%d, %s, %zu shapes, %zu skewed graph replays, exact residual, %s, collective fallback\n",
        ranks, gather ? "AllGather" : "AllReduce", shapes.size(), shapes.size() * 32,
        gather ? "bitwise norm and FP64 bounds" : "FP64 norm bounds");
    return 0;
}
