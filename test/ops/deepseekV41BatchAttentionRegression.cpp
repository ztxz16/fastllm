// Batched PV must match independent single-request attention, including packed
// KV fallbacks and batches whose first request is BF16 but later ones are not.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void Fill(Data &data, const std::vector<int> &dims, int seed) {
    size_t count = 1;
    for (int dim : dims)
        count *= dim;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = std::sin(float((i * 17 + seed * 13) % 1009)) * 0.3f;
    Data source(DataType::FLOAT32, dims, values);
    ToDataType(source, DataType::BFLOAT16);
    data.CopyFrom(source);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
int main() {
    int devices = 0;
    cudaDeviceProp properties{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 ||
        cudaGetDeviceProperties(&properties, 0) != cudaSuccess || properties.major < 8)
        return 77;
    std::string current;
    try {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        int checked = 0;
        for (int ringMode : {0, 1})
            for (int cmpMode : {0, 1, 3})
                for (int heads : {8, 17, 64})
                    for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
                        current = "ring=" + std::to_string(ringMode) + " cmp=" + std::to_string(cmpMode) +
                                  " heads=" + std::to_string(heads) + " batch=" + std::to_string(batch);
                        std::vector<Data> q(batch), kv(batch), ring(batch), cmp(batch), indices(batch),
                            reference(batch), output(batch), packedRing(batch), packedCmp(batch);
                        Data sink(DataType::FLOAT32, {heads}, std::vector<float>(heads, 0.01f));
                        sink.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                        std::vector<FastllmCudaV41AttentionRequest> requests;
                        for (int i = 0; i < batch; ++i) {
                            const int candidates = i % 3 == 0 ? 0 : i % 3 == 1 ? 64 : 512;
                            Fill(q[i], {1, 1, heads, 512}, i);
                            Fill(kv[i], {1, 1, 512}, i + 1);
                            Fill(ring[i], {1, 128, 512}, i + 2);
                            Fill(cmp[i], {1, 1024, 512}, i + 3);
                            indices[i].dataType = DataType::INT32;
                            indices[i].Resize({1, 1, candidates});
                            indices[i].Allocate();
                            for (int j = 0; j < candidates; ++j)
                                ((int *)indices[i].cpuData)[j] = j % 19 == 0 ? -1 : (j * 13 + i) % 1024;
                            if (candidates)
                                indices[i].ToDevice(DataDevice::CUDA, std::vector<int>{0});
                            // Mixed layouts ensure dispatch checks every request.
                            const bool packRing = i % 2 && ringMode;
                            const bool packCmp = i % 2 && cmpMode;
                            if (packRing)
                                Check(FastllmCudaDeepSeekV41QuantizeKV(ring[i], packedRing[i], ringMode, 32),
                                      "ring packing failed");
                            if (packCmp)
                                Check(FastllmCudaDeepSeekV41QuantizeKV(cmp[i], packedCmp[i], cmpMode,
                                                                       cmpMode == 3 ? 16 : 32),
                                      "KV packing failed");
                            requests.push_back({&q[i], &kv[i], packRing ? &packedRing[i] : &ring[i],
                                                candidates ? (packCmp ? &packedCmp[i] : &cmp[i]) : nullptr,
                                                candidates ? &indices[i] : nullptr, &output[i],
                                                33 + i * 129});
                            auto &r = requests.back();
                            Check(FastllmCudaDeepSeekV41SparseAttention(
                                      *r.q, *r.chunkKV, r.ringKV, r.compressedKV, r.indices, sink, 128,
                                      r.startPos, 0.0441941738f, reference[i]),
                                  "reference failed");
                        }
                        Check(FastllmCudaDeepSeekV41SparseAttentionBatch(requests, sink, 128, 0.0441941738f),
                              "batch rejected valid inputs");
                        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution failed");
                        for (int i = 0; i < batch; ++i) {
                            Data a, b;
                            a.CopyFrom(reference[i]);
                            b.CopyFrom(output[i]);
                            a.ToDevice(DataDevice::CPU);
                            b.ToDevice(DataDevice::CPU);
                            Check(a.GetBytes() == b.GetBytes() &&
                                      !std::memcmp(a.cpuData, b.cpuData, a.GetBytes()),
                                  "batch changed attention bits");
                            ++checked;
                        }
                    }
        std::cout << "PASS " << checked << " mixed KV attention requests bitwise identical\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << current << ": " << error.what() << '\n';
        return 1;
    }
}
