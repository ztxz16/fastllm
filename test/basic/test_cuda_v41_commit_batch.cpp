#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-v41-full-decode.h"
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "fastllm.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T> struct Buffer {
    T *p = nullptr;
    size_t n;

    explicit Buffer(const std::vector<T> &v) : n(v.size()) {
        Check(cudaMalloc(&p, n * sizeof(T)));
        Set(v);
    }
    ~Buffer() { cudaFree(p); }
    void Set(const std::vector<T> &v) {
        Check(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> Read() {
        std::vector<T> v(n);
        Check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

void Case(int batch, int rows, int window, int bytes) {
    const int layers = 5, total = batch * rows;
    std::vector<uint8_t> initial(size_t(layers) * batch * window * bytes, 0x93);
    std::vector<uint8_t> source(size_t(layers) * total * bytes);
    std::vector<uint32_t> raw(size_t(layers) * 2 * total * 512), score(raw.size());
    std::vector<uint32_t> tail(size_t(layers) * batch * 512, 0x78901234);
    for (size_t i = 0; i < source.size(); ++i) source[i] = (i * 37 + i / 17) % 251;
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = i * 13;
        score[i] = i * 17;
    }
    Buffer<uint8_t> ring(initial), saved(source);
    Buffer<uint32_t> kv(raw), scores(score), tails(tail), tailScores(tail);
    std::vector<FastllmV41FullDecodeRequest> request(layers * total);
    std::vector<int> spans(batch * 2);
    for (int b = 0; b < batch; ++b) {
        spans[2 * b] = 1 + (b * 7) % rows;
        spans[2 * b + 1] = b * rows;
    }
    for (int l = 0; l < layers; ++l) for (int b = 0; b < batch; ++b) {
        auto &q = request[l * total + b * rows];
        q.position = window - 1 + b;
        q.chunkBegin = b * rows;
        q.chunkLength = rows;
        q.windowKV = ring.p + size_t(l * batch + b) * window * bytes;
        q.rawTailKV = tails.p + size_t(l * batch + b) * 512;
        q.rawTailScore = tailScores.p + size_t(l * batch + b) * 512;
    }
    Buffer<FastllmV41FullDecodeRequest> requests(request);
    Buffer<int> deviceSpans(spans);
    std::vector<FastllmV41VerifyCommitLayer> descriptors(layers);
    for (int l = 0; l < layers; ++l) {
        descriptors[l] = {requests.p + l * total, saved.p + size_t(l) * total * bytes,
            kv.p + size_t(l) * 2 * total * 512, scores.p + size_t(l) * 2 * total * 512,
            window, l % 3, bytes};
    }
    Buffer<FastllmV41VerifyCommitLayer> d(descriptors);
    auto expected = initial;
    auto expectedTail = tail, expectedScore = tail;
    for (int l = 0; l < layers; ++l) for (int b = 0; b < batch; ++b) {
        const int accepted = spans[2 * b], pos = request[l * total + b * rows].position;
        for (int row = std::max(0, accepted - window); row < accepted; ++row) {
            std::copy_n(source.data() + size_t(l * total + b * rows + row) * bytes, bytes,
                expected.data() + (size_t(l * batch + b) * window + (pos + row) % window) * bytes);
        }
        if (l % 3 == 2 && (pos + accepted) % 2) {
            const size_t off = (size_t(l) * 2 * total + 2 * b * rows + pos % 2 + accepted - 1) * 512;
            std::copy_n(raw.data() + off, 512, expectedTail.data() + size_t(l * batch + b) * 512);
            std::copy_n(score.data() + off, 512, expectedScore.data() + size_t(l * batch + b) * 512);
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        ring.Set(initial);
        tails.Set(tail);
        tailScores.Set(tail);
        if (mode == 0) {
            for (int b = 0; b < batch; ++b) {
                Require(FastllmCudaV41CommitVerifiedPrefix(d.p, layers, spans[2 * b], rows, b * rows, total), "prefix");
            }
        } else if (mode == 1) {
            Require(FastllmCudaV41CommitVerifiedBatch(d.p, layers, deviceSpans.p, batch, total), "batch");
        } else {
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Require(FastllmCudaV41CommitVerifiedBatch(d.p, layers, deviceSpans.p, batch, total), "capture");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            for (int i = 0; i < 2; ++i) Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaGraphExecDestroy(exec));
            Check(cudaGraphDestroy(graph));
        }
        Require(ring.Read() == expected && tails.Read() == expectedTail && tailScores.Read() == expectedScore,
            "commit differs from CPU prefix/ring/tail oracle");
    }
}
void ContextStore(int batch, int rows) {
    using namespace fastllm;
    const int window = 13, width = 32;
    Data input(BFLOAT16), ring(BFLOAT16);
    input.Resize({batch, rows, width});
    ring.Resize({batch, window, width});
    input.ToDevice(CUDA, {0}, false);
    input.Allocate(false);
    ring.ToDevice(CUDA, {0}, false);
    ring.Allocate(false);
    std::vector<uint16_t> source(input.Count(0)), initial(ring.Count(0), 0x3f90);
    for (size_t i = 0; i < source.size(); ++i) source[i] = i % 16381;
    Check(cudaMemcpy(input.cudaData, source.data(), source.size() * 2, cudaMemcpyHostToDevice));
    std::vector<int> metadata(2 * batch);
    Buffer<int> device(metadata);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(FastllmCudaDeepSeekV41WindowStoreBatchPrefix(input, ring, window, device.p, device.p + batch),
        "batch window capture");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int phase = 0; phase <= rows; ++phase) {
        auto expected = initial;
        std::vector<uint16_t> actual(initial.size());
        for (int b = 0; b < batch; ++b) {
            const int start = 127 + b + phase, accepted = (phase + b) % (rows + 1);
            metadata[b] = start;
            metadata[batch + b] = accepted;
            for (int j = std::max(0, accepted - window); j < accepted; ++j) {
                std::copy_n(source.data() + size_t(b * rows + j) * width, width,
                    expected.data() + size_t(b * window + (start + j) % window) * width);
            }
        }
        device.Set(metadata);
        Check(cudaMemcpy(ring.cudaData, initial.data(), initial.size() * 2, cudaMemcpyHostToDevice));
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaMemcpy(actual.data(), ring.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost));
        Require(actual == expected, "dynamic window prefix/replay differs from CPU oracle");
    }
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
}

} // namespace
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        for (int batch : {1, 2, 3, 4, 9}) for (int rows : {2, 6, 17, 129}) for (int bytes : {544, 1024})
            Case(batch, rows, 13, bytes);
        for (int batch : {1, 3, 9}) for (int rows : {1, 6, 17, 129}) ContextStore(batch, rows);
        Require(!FastllmCudaV41CommitVerifiedBatch(nullptr, 1, nullptr, 1, 1), "invalid arguments");
        puts("PASS batched V41 commit: CPU oracle, original entry, graph replay, irregular requests/prefixes, ring wrap, packed/BF16 rows");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
