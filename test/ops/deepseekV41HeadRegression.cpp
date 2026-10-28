#include "models/deepseekv41.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace fastllm;
struct HeadProbe : DeepSeekV41Model {
    HeadProbe() {
        executionMode = ExecutionMode::Gpu;
    }
    using DeepSeekV41Model::TryBFloat16HeadLinear;
    using DeepSeekV41Model::v41TargetHeadWeights;
};
static void Require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static float Bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    std::memcpy(&x, &u, 4);
    return x;
}
static std::vector<float> Read(const Data &d) {
    std::vector<float> out(d.Count(0));
    Require(d.dataType == FLOAT32 &&
                cudaMemcpy(out.data(), d.cudaData, out.size() * 4, cudaMemcpyDeviceToHost) == cudaSuccess,
            "read logits failed");
    return out;
}
int main() {
    try {
        setenv("FASTLLM_TP", "8", 1);
        int ranks = 0;
        Require(cudaGetDeviceCount(&ranks) == cudaSuccess && ranks >= 8, "requires eight GPUs");
        ranks = 8;
        std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        SetThreads(16);
        ApplyDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}}, 0, 1);
        FastllmMultiCudaSetDevice(devices);
        FastllmCudaSetNcclForceSync(false);
        auto sync = [&] {
            for (int d : devices)
                FastllmCudaSyncDevice(d);
            FastllmCudaSetDevice(0);
        };
        int cases = 0;
        for (auto shape : {std::make_pair(1024, 512), std::make_pair(130048, 5120)}) {
            const int n = shape.first, k = shape.second;
            HeadProbe model;
            std::vector<float> weights(size_t(n) * k);
            for (size_t i = 0; i < weights.size(); ++i)
                weights[i] =
                    Bf16(float(int((i * 1664525u + (i / k) * 1013904223u) % 65521) - 32760) * 0.00001f);
            Data &w = model.weight.weight["head.weight"];
            w.CopyFrom(Data(FLOAT32, {n, k}, weights));
            w.tpLinearType = TP_LINEAR_ROW;
            for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 16, 64}) {
                std::vector<float> values(size_t(rows) * k);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = Bf16(std::sin(float((i * 37) % 1031)) * .3f);
                Data input(FLOAT32, {1, rows, k}, values), fp32Input, reference, actual;
                ToDataType(input, BFLOAT16);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                sync();
                PrepareMultiCudaReplicatedData(input, devices, true);
                ToDataType(input, fp32Input, FLOAT32);
                Linear(fp32Input, w, *GetEmptyData(), reference);
                sync();
                if (!model.TryBFloat16HeadLinear(input, actual)) {
                    std::fprintf(stderr,
                                 "head rejected: mode=%d input_type=%d replicated=%d weight_type=%d "
                                 "sharded=%d axis=%d locals=%zu\n",
                                 int(model.GetExecutionMode()), int(input.dataType),
                                 int(input.IsTensorParallelReplicated()), int(w.dataType),
                                 int(w.IsTensorParallelSharded()), w.tpAxis, w.multiDeviceDatas.size());
                    throw std::runtime_error("BF16 head rejected exact weights");
                }
                sync();
                Require(actual.IsTensorParallelSharded() && actual.tpRanges == reference.tpRanges &&
                            actual.dims == reference.dims,
                        "head vocabulary layout changed");
                double norm = 0, error = 0, maxScaled = 0;
                std::vector<std::vector<float>> expected(ranks);
                for (int d : devices) {
                    FastllmCudaSetDevice(d);
                    auto ref = Read(*reference.multiDeviceDatas.at(d));
                    auto got = Read(*actual.multiDeviceDatas.at(d));
                    expected[d] = got;
                    for (size_t i = 0; i < got.size(); ++i) {
                        Require(std::isfinite(got[i]), "nonfinite head");
                        double delta = got[i] - ref[i];
                        error += delta * delta;
                        norm += double(ref[i]) * ref[i];
                        maxScaled =
                            std::max(maxScaled, std::abs(delta) / std::max(1., std::abs(double(ref[i]))));
                    }
                    int width = actual.multiDeviceDatas.at(d)->dims.back();
                    int start = actual.tpRanges.at(d).front().first;
                    for (int row : {0, rows / 2, rows - 1})
                        for (int col : {0, width / 2, width - 1}) {
                            double exact = 0;
                            for (int j = 0; j < k; ++j)
                                exact += double(values[size_t(row) * k + j]) *
                                         weights[size_t(start + col) * k + j];
                            Require(std::abs(double(got[size_t(row) * width + col]) - exact) <
                                        5e-5 * std::max(1., std::abs(exact)),
                                    "FP64 head oracle mismatch");
                        }
                }
                Require(error <= 1e-9 * std::max(norm, 1.) && maxScaled < 5e-5,
                        "FP32 head precision regression");
                bool oldAsync = MultiCudaSetPersistentAsyncDispatch(true),
                     oldCapture = MultiCudaSetWorkerOrderedCapture(true);
                std::vector<void *> graphs(ranks), execs(ranks);
                MultiCudaRunDeviceCallbacks(
                    devices, [&](int, int) { Require(FastllmCudaGraphPrepareCaptureDevice(), "prepare"); });
                MultiCudaRunDeviceCallbacks(
                    devices, [&](int, int) { Require(FastllmCudaGraphBeginCapture(), "begin"); });
                Require(model.TryBFloat16HeadLinear(input, actual), "captured head rejected");
                MultiCudaRunDeviceCallbacks(devices, [&](int rank, int) {
                    Require(FastllmCudaGraphEndCapture(&graphs[rank]), "end");
                    Require(FastllmCudaGraphInstantiate(graphs[rank], &execs[rank]), "instantiate");
                });
                for (int replay = 0; replay < 4; ++replay) {
                    MultiCudaRunDeviceCallbacks(devices, [&](int rank, int d) {
                        Require(cudaMemsetAsync(actual.multiDeviceDatas.at(d)->cudaData, 0xFF,
                                                actual.multiDeviceDatas.at(d)->GetBytes(),
                                                cudaStreamPerThread) == cudaSuccess,
                                "poison");
                        Require(FastllmCudaGraphLaunch(execs[rank]), "launch");
                    });
                    sync();
                    for (int d : devices) {
                        FastllmCudaSetDevice(d);
                        Require(Read(*actual.multiDeviceDatas.at(d)) == expected[d],
                                "head graph changed bits");
                    }
                }
                MultiCudaRunDeviceCallbacks(devices, [&](int rank, int) {
                    FastllmCudaGraphExecDestroy(execs[rank]);
                    FastllmCudaGraphDestroy(graphs[rank]);
                });
                MultiCudaSetWorkerOrderedCapture(oldCapture);
                MultiCudaSetPersistentAsyncDispatch(oldAsync);
                std::printf("head N=%d K=%d M=%d relative_rms=%.9g max_scaled=%.9g PASS\n", n, k, rows,
                            std::sqrt(error / std::max(norm, 1.)), maxScaled);
                ++cases;
                if (rows == 64) {
                    model.v41TargetHeadWeights.reset();
                    FastllmCudaSetDevice(0);
                    float inexact = 1.000001f;
                    Require(cudaMemcpy(w.multiDeviceDatas.at(0)->cudaData, &inexact, 4,
                                       cudaMemcpyHostToDevice) == cudaSuccess,
                            "modify probe");
                    Require(!model.TryBFloat16HeadLinear(input, actual),
                            "inexact FP32 weight must fall back");
                    Require(Read(*actual.multiDeviceDatas.at(0)) == expected[0], "fallback changed output");
                }
            }
        }
        std::printf("PASS head: %d cases, exact weight packing, FP64 oracle, TP8 layout, repeated graphs, "
                    "inexact-weight fallback\n",
                    cases);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
