#include "fastllm.h"
#include "blocks/baseblock.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;

constexpr int kGraphRepeats = 256;
constexpr int kTimingSamples = 11;

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

struct Shape { int tokens, heads, kvHeads, dim, valueDim, rotaryDim; };

static void Init(Data &data, const std::vector<int> &dims) {
    data.Resize(dims);
    data.Allocate();
    data.ToDevice(DataDevice::CUDA, {0}, false);
}

static void Fill(Data &data, unsigned seed) {
    std::vector<uint16_t> values(data.Count(0));
    for (auto &value : values) {
        seed = 1664525u * seed + 1013904223u;
        // Cover signs, mantissas, normal exponents and subnormals; exclude
        // Inf/NaN inputs so payload propagation is not the equality criterion.
        value = seed >> 16;
        if ((value & 0x7f80) == 0x7f80) value &= 0x807f;
    }
    Check(cudaMemcpy(data.cudaData, values.data(), values.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

static void CopyDeviceData(const Data &source, Data &target) {
    Check(cudaMemcpy(target.cudaData, source.cudaData, source.Count(0) * sizeof(uint16_t), cudaMemcpyDeviceToDevice));
}

static void Equal(const Data &a, const Data &b) {
    std::vector<uint16_t> x(a.Count(0)), y(b.Count(0));
    Check(cudaMemcpy(x.data(), a.cudaData, x.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(y.data(), b.cudaData, y.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    Require(x == y, "Fused RoPE differs from separate RoPE/Mul (including untouched tail)");
}

static void Position(Data &p, int start, int replay) {
    std::vector<float> positions(p.Count(0));
    for (size_t i = 0; i < positions.size(); ++i)
        positions[i] = start + ((i * 17 + replay) % 513);
    Check(cudaMemcpy(p.cudaData, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice));
}

static void Run(const Shape &s, float theta, float scale, int start, int &checks) {
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), qr(BFLOAT16), kr(BFLOAT16), vr(BFLOAT16), p(FLOAT32);
    Init(q, {1, s.tokens, s.heads * s.dim}); Init(qr, q.dims);
    Init(k, {1, s.tokens, s.kvHeads * s.dim}); Init(kr, k.dims);
    Init(v, {1, s.tokens, s.kvHeads * s.valueDim}); Init(vr, v.dims);
    Init(p, {1, s.tokens});
    auto fused = [&] {
        FastllmCudaNaiveRopeQKScaleV(q, k, v, p, s.heads, s.kvHeads, s.dim,
            s.valueDim, s.rotaryDim, theta, scale);
    };
    auto reference = [&] {
        FastllmCudaNaiveRope(qr, p, s.heads, s.dim, s.rotaryDim, theta);
        FastllmCudaNaiveRope(kr, p, s.kvHeads, s.dim, s.rotaryDim, theta);
        Mul(vr, scale, vr);
    };
    Fill(q, 11); Fill(k, 13); Fill(v, 17); Position(p, start, 0);
    fused();
    Check(cudaDeviceSynchronize());
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    fused();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, 0));
    for (int replay = 0; replay < 4; ++replay) {
        Fill(q, 31 + replay); Fill(k, 37 + replay); Fill(v, 41 + replay);
        Position(p, start, replay);
        CopyDeviceData(q, qr); CopyDeviceData(k, kr); CopyDeviceData(v, vr);
        reference();
        if (replay == 0) fused();
        else Check(cudaGraphLaunch(executable, cudaStreamPerThread));
        Equal(q, qr); Equal(k, kr); Equal(v, vr);
        ++checks;
    }
    Check(cudaGraphExecDestroy(executable));
    Check(cudaGraphDestroy(graph));
}

static float TimeGraph(cudaGraphExec_t graph) {
    for (int i = 0; i < 3; ++i) Check(cudaGraphLaunch(graph, cudaStreamPerThread));
    cudaEvent_t begin, end;
    Check(cudaEventCreate(&begin));
    Check(cudaEventCreate(&end));
    Check(cudaEventRecord(begin, cudaStreamPerThread));
    Check(cudaGraphLaunch(graph, cudaStreamPerThread));
    Check(cudaEventRecord(end, cudaStreamPerThread));
    Check(cudaEventSynchronize(end));
    float ms;
    Check(cudaEventElapsedTime(&ms, begin, end));
    Check(cudaEventDestroy(begin));
    Check(cudaEventDestroy(end));
    return ms * 1000 / kGraphRepeats;
}

static void Bench(int tokens, int kvHeads, float theta, int position) {
    Data q(BFLOAT16), k(BFLOAT16), v(BFLOAT16), p(FLOAT32);
    Init(q, {1, tokens, 64 * 192});
    Init(k, {1, tokens, kvHeads * 192});
    Init(v, {1, tokens, kvHeads * 128});
    Init(p, {1, tokens});
    Fill(q, 1); Fill(k, 2); Fill(v, 3); Position(p, position, 0);
    cudaGraph_t graph[2];
    cudaGraphExec_t executable[2];
    for (int version = 0; version < 2; ++version) {
        auto run = [&] {
            if (version) {
                FastllmCudaNaiveRopeQKScaleV(q, k, v, p, 64, kvHeads, 192, 128, 64, theta, 1.f);
            } else {
                FastllmCudaNaiveRope(q, p, 64, 192, 64, theta);
                FastllmCudaNaiveRope(k, p, kvHeads, 192, 64, theta);
                Mul(v, 1.f, v);
            }
        };
        run();
        Check(cudaDeviceSynchronize());
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < kGraphRepeats; ++i) run();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph[version]));
        Check(cudaGraphInstantiate(&executable[version], graph[version], 0));
    }
    std::vector<float> times[2];
    for (int sample = 0; sample < kTimingSamples; ++sample) for (int j = 0; j < 2; ++j) {
        int version = (sample + j) % 2;
        times[version].push_back(TimeGraph(executable[version]));
    }
    for (auto &samples : times) std::sort(samples.begin(), samples.end());
    std::printf("{\"tokens\":%d,\"kv_heads\":%d,\"theta\":%.0f,\"position\":%d,\"baseline_us\":%.6f,\"fused_us\":%.6f}\n",
        tokens, kvHeads, theta, position, times[0][kTimingSamples / 2], times[1][kTimingSamples / 2]);
    for (int version = 0; version < 2; ++version) {
        Check(cudaGraphExecDestroy(executable[version]));
        Check(cudaGraphDestroy(graph[version]));
    }
}

int main(int argc, char **argv) {
    try {
        Check(cudaSetDevice(0));
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            for (int tokens : {1, 512}) for (int position : {128, 32824, 105615, 131128}) {
                Bench(tokens, 4, 1e7f, position);
                Bench(tokens, 8, 1e4f, position);
            }
            return 0;
        }
        bool quick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
        int checks = 0;
        for (const auto &shape : {Shape{1, 64, 4, 192, 128, 64}, Shape{1, 64, 8, 192, 128, 64},
                Shape{3, 3, 1, 96, 73, 96}, Shape{17, 16, 4, 128, 129, 128}, Shape{512, 64, 8, 192, 128, 64}})
            for (float theta : {1e4f, 1e7f}) for (float scale : {1.f, 1.003f, -0.9371f})
                for (int position : {0, 32824, 105614, 105615, 105616, 131128, 1048000}) {
                    if (quick && (shape.tokens > 17 || theta != 1e4f || scale != 1.003f || position != 105615)) continue;
                    Run(shape, theta, scale, position, checks);
                }
        std::printf("Naive fused Q/K RoPE + V scale regression passed: %d cases\n", checks);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
