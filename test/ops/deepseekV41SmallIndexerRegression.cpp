// Small indexer chunks must match independent M=1 queries, including packed
// KV, multiple requests, causal tails and graph replay after input changes.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
static void Cuda(cudaError_t e) {
    Check(e == cudaSuccess, cudaGetErrorString(e));
}
static void Fill(Data &d, const std::vector<int> &dims, int seed,
                 DataType type = DataType::BFLOAT16) {
    size_t n = 1;
    for (int v : dims)
        n *= v;
    std::vector<float> values(n);
    float pattern[1009];
    for (int i = 0; i < 1009; ++i)
        pattern[i] = std::sin(float(i)) * .3f;
    for (size_t i = 0; i < n; ++i)
        values[i] = pattern[(i * 17 + seed * 13) % 1009];
    Data cpu(DataType::FLOAT32, dims, values);
    ToDataType(cpu, type);
    d.CopyFrom(cpu);
    d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void View(Data &v, const Data &d, size_t offset, const std::vector<int> &dims) {
    v.FakeFrom(d, offset);
    v.expansionDims.clear();
    v.Resize(dims);
    v.expansionDims = dims;
    v.expansionSize = v.Count(0);
    v.expansionBytes = v.GetBytes();
}

int main() {
    int devices = 0, major = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1 ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess || major < 8)
        return 77;
    try {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        int checked = 0;
        for (int batch : {1, 2}) for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 9})
        for (int heads : {17, 32}) for (int count : {513, 16385})
        for (int mode : {0, 1, 3}) for (int ratio : {1, 2}) {
            const int start = count * ratio - rows;
            Data q, changed, weights, keys, packed, actual;
            Fill(q, {batch, rows, heads, 128}, 1);
            Fill(changed, q.dims, 13);
            Fill(weights, {batch, rows, heads}, 2, FLOAT32);
            Fill(keys, {batch, count, 128}, 3);
            if (mode)
                Check(FastllmCudaDeepSeekV41QuantizeKV(keys, packed, mode, mode == 3 ? 16 : 32), "pack keys");
            Data &key = mode ? packed : keys;
            auto launch = [&] {
                Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, key, ratio, start, actual), "chunk indexer");
            };
            launch();
            Cuda(cudaDeviceSynchronize());
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&exec, graph, 0));
            for (int replay = 0; replay < 2; ++replay) {
                if (replay)
                    Cuda(cudaMemcpyAsync(q.cudaData, changed.cudaData, q.GetBytes(), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                Cuda(cudaMemsetAsync(actual.cudaData, 0x55, actual.GetBytes(), cudaStreamPerThread));
                Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                std::vector<float> output(actual.Count(0));
                Cuda(cudaMemcpy(output.data(), actual.cudaData, actual.GetBytes(), cudaMemcpyDeviceToHost));
                for (int b = 0; b < batch; ++b) for (int row = 0; row < rows; ++row) {
                    Data query, weight, sharedKeys, reference;
                    View(query, q, size_t(b * rows + row) * heads * 128 * 2, {1, 1, heads, 128});
                    View(weight, weights, size_t(b * rows + row) * heads * 4, {1, 1, heads});
                    View(sharedKeys, key, size_t(b) * key.GetBytes() / batch, {1, count, key.dims.back()});
                    Check(FastllmCudaDeepSeekV41IndexerScore(query, weight, sharedKeys, ratio, start + row, reference), "single-row reference");
                    Cuda(cudaDeviceSynchronize());
                    const int visible = std::min(count, (start + row + 1) / ratio);
                    std::vector<float> expected(visible);
                    Cuda(cudaMemcpy(expected.data(), reference.cudaData, visible * sizeof(float), cudaMemcpyDeviceToHost));
                    Check(!std::memcmp(expected.data(), output.data() + size_t(b * rows + row) * count,
                                       visible * sizeof(float)), "indexer bits changed");
                    ++checked;
                }
            }
            Cuda(cudaGraphExecDestroy(exec));
            Cuda(cudaGraphDestroy(graph));
        }
        // Flattening 8192 requests x 8 rows exceeds CUDA grid.y's 65535 limit.
        // The general prefill kernel must remain available for this shape.
        {
            auto ones = [](Data &target, const std::vector<int> &dims, DataType type) {
                size_t count = 1;
                for (int dim : dims) count *= dim;
                Data cpu(FLOAT32, dims, std::vector<float>(count, 1.f));
                ToDataType(cpu, type);
                target.CopyFrom(cpu);
                target.ToDevice(CUDA, std::vector<int>{0});
            };
            Data q, weights, keys, output;
            ones(q, {8192, 8, 1, 128}, BFLOAT16);
            ones(weights, {8192, 8, 1}, FLOAT32);
            ones(keys, {8192, 16, 128}, BFLOAT16);
            Check(FastllmCudaDeepSeekV41IndexerScore(q, weights, keys, 1, 15, output),
                  "large-batch grid fallback");
            std::vector<float> actual(output.Count(0));
            Cuda(cudaMemcpy(actual.data(), output.cudaData, output.GetBytes(), cudaMemcpyDeviceToHost));
            for (float score : actual) Check(score == 128.f, "large-batch score mismatch");
            checked += 8192 * 8;
        }

        std::cout << "PASS " << checked << " small-chunk indexer rows bitwise identical" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
