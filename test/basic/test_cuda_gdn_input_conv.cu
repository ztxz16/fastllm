#include "fastllm.h"
#include "devices/cuda/fastllm-cuda-gdn.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Require(bool b, const char *s) {
    if (!b)
        throw std::runtime_error(s);
}
static void Allocate(Data &d) {
    d.dataDevice = DataDevice::CUDA;
    d.dataDeviceIds = {0};
    d.Allocate();
}
template <class T> static void Upload(Data &d, const std::vector<T> &v) {
    Check(cudaMemcpy(d.cudaData, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <class T> static std::vector<T> Download(Data &d) {
    std::vector<T> v(d.Count(0));
    Check(cudaMemcpy(v.data(), d.cudaData, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}

template <class T> void Run(DataType type, int K, int N, int C) {
    const int Z = N - C, S = 16;
    printf("SHAPE type=%d K=%d N=%d C=%d\n", int(type), K, N, C);
    Data w(DataType::FP8_E4M3, {N, K}), cw(DataType::FLOAT32, {C, 4}), bias(DataType::FLOAT32, {N}),
        cb(DataType::FLOAT32, {C});
    w.blockK = 1;
    w.blockM = K;
    w.scales.resize(N);
    std::vector<uint8_t> codes(size_t(N) * K);
    std::vector<float> table(256), conv(C * 4), pb(N), bc(C);
    for (int i = 0; i < 256; ++i) {
        __nv_fp8_e4m3 word;
        word.__x = i;
        table[i] = float(word);
    }
    for (int r = 0; r < N; ++r) {
        w.scales[r] = float(__nv_bfloat16(.002f * (1 + r % 9)));
        pb[r] = .01f * std::sin(r * .1f);
        for (int k = 0; k < K; ++k)
            codes[size_t(r) * K + k] =
                uint8_t(0x28 + ((r * 3 + k * 13 + k / 17) % 24)) | (((k / 11 + r) % 2) ? 128 : 0);
    }
    for (int i = 0; i < C * 4; ++i)
        conv[i] = .1f * std::cos(i * .02f);
    for (int i = 0; i < C; ++i)
        bc[i] = .01f * std::sin(i * .04f);
    Allocate(w);
    Allocate(cw);
    Allocate(bias);
    Allocate(cb);
    Upload(w, codes);
    Upload(cw, conv);
    Upload(bias, pb);
    Upload(cb, bc);
    std::vector<int> rows = {0, 1, 31, 127, N / 2, C - 1, C, N - 2, N - 1};
    for (int testBatch : {1, 2, 3, 4, 5, 6, 7, 8, 9, -1}) {
        int batch = std::abs(testBatch);
        bool forcedFallback = testBatch < 0;
        setenv("FASTLLM_CUDA_GDN_INPUT_CONV", forcedFallback ? "0" : "1", 1);
        Data input(type, {1, batch, K}), cache(type, {S, 1, C, 4}), ids(DataType::INT32, {batch}), out(type),
            z(type), scratch(type);
        Allocate(input);
        Allocate(cache);
        Allocate(ids);
        std::vector<T> x(batch * K), history(S * C * 4);
        std::vector<int> slots(batch);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = T(.3f * std::sin(i * .013f));
        for (size_t i = 0; i < history.size(); ++i)
            history[i] = T(.1f * std::cos(i * .017f));
        for (int b = 0; b < batch; ++b)
            slots[b] = S - 1 - b;
        Upload(input, x);
        Upload(cache, history);
        Upload(ids, slots);
        bool admitted = FastllmCudaGdnInputConvCanRun(input, w, bias, cw, cb, cache, &ids, batch);
        Require(admitted == (batch <= 8 && !forcedFallback), "CanRun batch admission wrong");
        if (batch == 1) {
            w.blockM = 128;
            Require(!FastllmCudaGdnInputConvCanRun(input, w, bias, cw, cb, cache, &ids, batch),
                    "block scales admitted");
            w.blockM = K;
            w.IsRepacked = true;
            Require(!FastllmCudaGdnInputConvCanRun(input, w, bias, cw, cb, cache, &ids, batch),
                    "repacked weight admitted");
            w.IsRepacked = false;
            input.Reshape({1, 2, K / 2});
            Require(!FastllmCudaGdnInputConvCanRun(input, w, bias, cw, cb, cache, &ids, batch),
                    "bad shape admitted");
            input.Reshape({1, 1, K});
            auto valid = [&]() {
                return FastllmCudaGdnInputConvValidInputs(input, w, bias, cw, cb, cache, &ids, batch);
            };
            Require(valid(), "valid fallback contract rejected");
            void *inputPointer = input.cudaData;
            input.cudaData = static_cast<char *>(inputPointer) + 2;
            Require(!valid(), "misaligned Linear input admitted to fallback");
            input.cudaData = inputPointer;
            void *weightPointer = w.cudaData;
            w.cudaData = static_cast<char *>(weightPointer) + 1;
            Require(!valid(), "misaligned Linear weight admitted to fallback");
            w.cudaData = weightPointer;
            cache.strides.back() = 2;
            Require(!valid(), "non-contiguous cache admitted to fallback");
            cache.strides.back() = 1;
            auto slotType = ids.dataType;
            ids.dataType = DataType::FLOAT32;
            Require(!valid(), "wrong slot dtype admitted to fallback");
            ids.dataType = slotType;
            auto convType = cw.dataType;
            cw.dataType = DataType::FLOAT16;
            Require(!valid(), "wrong convolution dtype admitted to fallback");
            cw.dataType = convType;
            cw.Reshape({C, 2, 2});
            Require(!valid(), "wrong convolution layout admitted to fallback");
            cw.Reshape({C, 4});
        }
        auto launch = [&]() {
            return CudaGdnInputConvBlock(input, w, bias, cw, cb, cache, &ids, out, z, scratch, batch);
        };
        Require(launch() == admitted, "Block path mismatch");
        Check(cudaDeviceSynchronize());
        // Warm allocation above; restore cache before capture, which must not execute.
        Upload(cache, history);
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        launch();
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int step = 0; step < 3; ++step) {
            // Change both request-to-slot mapping and activations between graph replays.
            std::rotate(slots.begin(), slots.begin() + 1, slots.end());
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = T(.3f * std::sin(i * .013f + step * .37f));
            Upload(input, x);
            Upload(ids, slots);
            Upload(cache, history);
            Check(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            auto actual = Download<T>(out), zz = Download<T>(z), after = Download<T>(cache);
            for (T v : actual)
                Require(std::isfinite(float(v)), "non-finite output");
            for (T v : zz)
                Require(std::isfinite(float(v)), "non-finite z output");
            float worst = 0;
            for (int b = 0; b < batch; ++b)
                for (int r : rows) {
                    double dot = 0;
                    for (int k = 0; k < K; ++k)
                        dot += double(float(x[b * K + k])) * table[codes[size_t(r) * K + k]];
                    T projected = T(float(dot * w.scales[r] + pb[r]));
                    float ref = float(projected), got;
                    if (r < C) {
                        size_t off = (size_t(slots[b]) * C + r) * 4;
                        double cv = bc[r];
                        for (int t = 0; t < 3; ++t)
                            cv += double(float(history[off + t + 1])) * conv[r * 4 + t];
                        cv += double(float(projected)) * conv[r * 4 + 3];
                        float rounded = float(T(float(cv)));
                        ref = float(T(rounded / (1 + std::exp(-rounded))));
                        got = float(actual[b * C + r]);
                        float cacheError = std::abs(float(after[off + 3]) - float(projected));
                        Require(cacheError < (type == DataType::FLOAT16 ? .004f : .025f),
                                "projected cache differs from FP64 oracle");
                    } else
                        got = float(zz[b * Z + r - C]);
                    float error = std::abs(got - ref);
                    worst = std::max(worst, error);
                    Require(error < (type == DataType::FLOAT16 ? .004f : .025f),
                            "output differs from FP64 oracle");
                }
            for (int slot = 0; slot < S; ++slot) {
                bool active = std::find(slots.begin(), slots.end(), slot) != slots.end();
                for (int c = 0; c < C; ++c)
                    for (int t = 0; t < (active ? 3 : 4); ++t) {
                        size_t i = (size_t(slot) * C + c) * 4 + t;
                        Require(float(after[i]) == float(history[i + (active ? 1 : 0)]),
                                "cache slot/shift changed incorrectly");
                    }
            }
            history = std::move(after);
            std::printf("PASS dtype=%d batch=%d graph_step=%d fused=%d max_abs=%.7f\n", int(type), batch,
                        step, admitted, worst);
        }
        Check(cudaGraphExecDestroy(exec));
        Check(cudaGraphDestroy(graph));
    }
}
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            return 77;
        Check(cudaSetDevice(0));
        cudaDeviceProp properties{};
        Check(cudaGetDeviceProperties(&properties, 0));
        if (properties.major * 10 + properties.minor < 75)
            return 77;
        setenv("FASTLLM_CUDA_GDN_INPUT_CONV", "1", 1);
        for (auto shape : {std::vector<int>{5120,16384,10240}, {5120,8192,5120},
                           {5120,6144,3840}, {5120,5120,3200}, {5120,4096,2560},
                           {3072,4111,2567}}) {
            Run<half>(DataType::FLOAT16, shape[0], shape[1], shape[2]);
            Run<__nv_bfloat16>(DataType::BFLOAT16, shape[0], shape[1], shape[2]);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
