#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<typename T> static T Convert(float x);
template<> float Convert<float>(float x) { return x; }
template<> half Convert<half>(float x) { return __float2half(x); }
template<> __nv_bfloat16 Convert<__nv_bfloat16>(float x) { return __float2bfloat16(x); }
template<typename T> static float Value(T x) { return float(x); }
template<typename T> static void Upload(Data &data, const std::vector<T> &values) {
    data.ToDevice(DataDevice::CUDA, {0}, false);
    data.Allocate(false);
    Require(cudaMemcpy(data.cudaData, values.data(), values.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess, "upload failed");
}
template<typename T> static std::vector<T> Read(const Data &data) {
    std::vector<T> values(data.Count(0));
    Require(cudaMemcpy(values.data(), data.cudaData, values.size() * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "read failed");
    return values;
}
template<typename T> static void Check(DataType type, int vocab, int rank, int block, bool tied = false) {
    std::vector<T> embed(size_t(vocab) * rank), head(embed.size());
    std::vector<float> logits(size_t(block) * vocab);
    uint32_t rng = 271828u;
    auto random = [&]() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return float(int(rng % 20001u) - 10000) / 10000.f;
    };
    for (size_t i = 0; i < embed.size(); ++i) {
        embed[i] = Convert<T>(random() * .5f);
        head[i] = Convert<T>(tied ? 0.f : random() * .5f);
    }
    for (float &v : logits) v = tied ? 0.f : random() * 2.f;
    Data e(type, {vocab, rank}), h(type, {vocab, rank});
    Data l(FLOAT32, {1, block, vocab}), tokens(INT32, {block + 1}), values, indices, embeds;
    Upload(e, embed); Upload(h, head); Upload(l, logits);
    std::vector<int> initial(block + 1, 0);
    Upload(tokens, initial);
    auto run = [&]() {
        Require(FastllmCudaDeepSeekV41MarkovChainDevice(l, e, h, block, tokens, values, indices, embeds),
                "chain rejected valid inputs");
    };
    run();
    Require(cudaDeviceSynchronize() == cudaSuccess, "warmup failed");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
            "capture begin failed");
    run();
    Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "capture end failed");
    Require(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess, "instantiate failed");
    double maxScaled = 0;
    for (int anchor : {vocab - 1, 0, vocab / 2}) {
        std::vector<int> expected(block + 1, anchor);
        const int chunks = values.Count(0), perChunk = (vocab + chunks - 1) / chunks;
        std::vector<float> partial(chunks, -FLT_MAX), expectedEmbeds(size_t(block) * rank);
        for (int step = 0; step < block; ++step) {
            const T *row = embed.data() + size_t(expected[step]) * rank;
            for (int d = 0; d < rank; ++d) expectedEmbeds[size_t(step) * rank + d] = Value(row[d]);
            float best = -FLT_MAX;
            int bestIndex = vocab;
            for (int v = 0; v < vocab; ++v) {
                float dot = 0;
                for (int d = 0; d < rank; ++d)
                    dot = std::fma(Value(head[size_t(v) * rank + d]), Value(row[d]), dot);
                float score = logits[size_t(step) * vocab + v] + dot;
                if (score > best) { best = score; bestIndex = v; }
                if (step == block - 1) partial[v / perChunk] = std::max(partial[v / perChunk], score);
            }
            expected[step + 1] = bestIndex;
        }
        for (bool replay : {false, true}) {
            initial.assign(block + 1, -1); initial[0] = anchor;
            Upload(tokens, initial);
            Require(cudaMemsetAsync(embeds.cudaData, 0xff, embeds.GetBytes(), cudaStreamPerThread) == cudaSuccess,
                    "poison failed");
            if (replay) Require(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "replay failed");
            else run();
            Require(Read<int>(tokens) == expected, "Markov tokens differ from CPU reference");
            Require(Read<float>(embeds) == expectedEmbeds, "Markov embeddings differ from CPU reference");
            auto scores = Read<float>(values);
            for (int i = 0; i < chunks; ++i) {
                Require(std::isfinite(scores[i]), "nonfinite score");
                double error = std::abs(double(scores[i]) - partial[i]) / std::max(1., std::abs(double(partial[i])));
                maxScaled = std::max(maxScaled, error);
                Require(error < 5e-5, "Markov score precision regression");
            }
        }
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    std::printf("type=%d vocab=%d rank=%d block=%d tied=%d max_scaled=%.9g eager+graph PASS\n",
                int(type), vocab, rank, block, tied, maxScaled);
}
int main() {
    try {
        FastllmCudaSetDevice(0);
        Check<__nv_bfloat16>(BFLOAT16, 129280, 256, 5);
        Check<__nv_bfloat16>(BFLOAT16, 32769, 256, 8);
        Check<__nv_bfloat16>(BFLOAT16, 33001, 256, 1, true);
        Check<__nv_bfloat16>(BFLOAT16, 128, 256, 5);
        Check<__nv_bfloat16>(BFLOAT16, 129, 31, 5);
        Check<__nv_bfloat16>(BFLOAT16, 1023, 512, 5);
        Check<half>(FLOAT16, 33001, 256, 5);
        Check<float>(FLOAT32, 1023, 64, 5);
        std::puts("DSpark Markov regression PASS");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
