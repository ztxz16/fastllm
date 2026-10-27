// Packed grouped projections must retain their layout across TP and batching.
#include "fastllm.h"
#include "gguf.h"
#include "executor.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static std::vector<uint8_t> Read(Data &data) {
    Data cpu; cpu.CopyFrom(data); cpu.ToDevice(CPU);
    return {cpu.cpuData, cpu.cpuData + cpu.GetBytes()};
}
static void Case(DataType type, ggml_type quant, int groups, int rows, int columns) {
    Data weight(DATA_GGUF_FORMAT, quant, {groups, rows, columns});
    weight.Allocate(false);
    uint32_t seed = 721;
    for (size_t i = 0; i < weight.Count(0); ++i) {
        seed = seed * 1664525U + 1013904223U; weight.cpuData[i] = seed >> 24;
    }
    const uint16_t scale = 0x0c00; // FP16 2^-12
    for (size_t i = 0; i < weight.Count(0); i += ggml_type_size(quant)) {
        memcpy(weight.cpuData + i, &scale, 2);
        if (quant == GGML_TYPE_Q5_K) memcpy(weight.cpuData + i + 2, &scale, 2);
    }
    Data referenceWeight; referenceWeight.CopyFrom(weight);
    referenceWeight.ToDevice(CUDA, {0}, true);
    const size_t rowBytes = ggml_row_size(quant, columns);
    Data output;
    for (int tokens : {1, 2, 5, 8, 33, 1}) {
        std::vector<float> values(size_t(tokens) * groups * columns);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i * 17 + tokens * 13)) * .25f;
        Data input(type, {1, tokens, groups, columns}, values), reference(type, {1, tokens, groups * rows});
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        input.ToDevice(CUDA, {0}, true); reference.ToDevice(CUDA, {0}, false); reference.Allocate(false);
        // Independent single-row projections are the numerical oracle.
        for (int t = 0; t < tokens; ++t) for (int g = 0; g < groups; ++g) {
            Data x, w, y;
            x.FakeFrom(input, size_t(t * groups + g) * columns * input.unitSize); x.Resize({1, columns});
            w.FakeFrom(referenceWeight, size_t(g) * rows * rowBytes); w.ggmlType = quant; w.dims = {rows, columns};
            y.FakeFrom(reference, size_t(t * groups + g) * rows * input.unitSize); y.Resize({1, rows});
            Linear(x, w, Data(), y);
        }
        auto expected = Read(reference);
        Data single;
        ((Executor *)GetExecutor())->Run("DeepSeekV4WoA", {{"input", &input}, {"weight", &referenceWeight},
            {"output", &single}}, {}, {{"groups", groups}, {"oRank", rows}});
        if (tokens == 1 || (quant == GGML_TYPE_Q5_K && tokens <= 8)) Check(Read(single) == expected, "batched WoA differs from independent GEMV");
        else expected = Read(single); // Unchanged fallback algorithms are compared across PP/TP.
        Data sharded;
        const int halfGroups = groups / 2;
        DivisionScheme scheme{{0, {{halfGroups, groups}}}, {1, {{0, halfGroups}}}};
        PrepareMultiCudaShardedData(sharded, {0, 1}, input.dims, 2, scheme);
        for (int d : {0, 1}) {
            const int begin = scheme[d][0].first;
            std::vector<float> localValues(size_t(tokens) * halfGroups * columns);
            for (int t = 0; t < tokens; ++t)
                memcpy(localValues.data() + size_t(t) * halfGroups * columns,
                    values.data() + size_t(t * groups + begin) * columns, size_t(halfGroups) * columns * sizeof(float));
            Data local(type, {1, tokens, halfGroups, columns}, localValues);
            local.ToDevice(CUDA, {d}, true); sharded.multiDeviceDatas.at(d)->CopyFrom(local);
        }
        ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
        FastllmMultiCudaSetDevice({0, 1});
        ((Executor *)GetExecutor())->Run("DeepSeekV4WoA", {{"input", &sharded}, {"weight", &weight},
            {"output", &output}}, {}, {{"groups", groups}, {"oRank", rows}});
        for (int d : {0, 1}) {
            Check(weight.multiDeviceDatas.at(d)->dims == std::vector<int>({halfGroups * rows, columns}),
                "GGUF TP shard has wrong matrix dimensions");
            auto actual = Read(*output.multiDeviceDatas.at(d));
            const size_t localBytes = size_t(halfGroups) * rows * input.unitSize;
            Check(actual.size() == tokens * localBytes, "GGUF TP output size mismatch");
            for (int t = 0; t < tokens; ++t)
                Check(!memcmp(actual.data() + t * localBytes,
                    expected.data() + size_t(t * groups + scheme[d][0].first) * rows * input.unitSize, localBytes),
                    "GGUF TP WoA differs from independent GEMV");
        }
        Check(cudaDeviceSynchronize() == cudaSuccess, "GGUF TP CUDA failure");
    }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    try {
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16}) {
            Case(type, GGML_TYPE_Q5_K, 4, 129, 256);
            Case(type, GGML_TYPE_Q5_K, 8, 1024, 4096);
            Case(type, GGML_TYPE_Q8_0, 4, 129, 256);
        }
        std::cout << "PASS: GGUF grouped WoA TP and batch (54 cases, both ranks match PP; decode matches independent GEMV)\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
