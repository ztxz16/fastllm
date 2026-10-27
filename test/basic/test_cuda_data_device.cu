#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void DelayedFill(float *p, float value) {
    const unsigned long long start = clock64();
    while (clock64() - start < 40000000ULL) {}
    p[0] = value;
}
int main() {
    try {
        int count = 0; Check(cudaGetDeviceCount(&count));
        if (count < 2) { std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: requires two GPUs"); return 0; }
        int failures = 0;
        for (int owner = 0; owner < 2; ++owner) {
            const int caller = 1 - owner;
            Check(cudaSetDevice(owner));
            fastllm::Data source(fastllm::FLOAT32, {1});
            source.dataDevice = fastllm::CUDA; source.dataDeviceIds = {owner};
            source.Allocate(false);
            Check(cudaMemset(source.cudaData, 0, sizeof(float)));
            DelayedFill<<<1, 1, 0, cudaStreamPerThread>>>(static_cast<float *>(source.cudaData), 17.f + owner);
            Check(cudaSetDevice(caller));
            source.ToDevice(fastllm::CPU);
            int current = -1; Check(cudaGetDevice(&current));
            if (reinterpret_cast<float *>(source.cpuData)[0] != 17.f + owner || current != caller) {
                std::printf("FAIL source-device CPU read: owner=%d caller=%d value=%g current=%d\n",
                    owner, caller, reinterpret_cast<float *>(source.cpuData)[0], current); ++failures;
            }
            Check(cudaSetDevice(owner)); Check(cudaDeviceSynchronize());
            reinterpret_cast<float *>(source.cpuData)[0] = 29.f + owner;
            source.ToDevice(fastllm::CUDA, {owner}, true);
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaSetDevice(caller));
            fastllm::Data copy;
            copy.CopyFrom(source);
            Check(cudaGetDevice(&current));
            cudaPointerAttributes attr{}; Check(cudaPointerGetAttributes(&attr, copy.cudaData));
            if (copy.dataDeviceIds != source.dataDeviceIds || attr.device != owner || current != caller) {
                std::printf("FAIL CopyFrom placement: owner=%d allocation=%d current=%d metadata=%d\n",
                    owner, attr.device, current, copy.dataDeviceIds.empty() ? -1 : copy.dataDeviceIds[0]); ++failures;
            }
            copy.ToDevice(fastllm::CPU);
            if (reinterpret_cast<float *>(copy.cpuData)[0] != 29.f + owner) {
                std::puts("FAIL CopyFrom values"); ++failures;
            }
            Check(cudaSetDevice(owner)); Check(cudaDeviceSynchronize());
        }
        fastllm::SetCudaGraph(true);
        fastllm::SetCudaEmbedding(false);
        for (int device = 0; device < 2; ++device) {
            fastllm::SetDeviceMap({{"cuda:" + std::to_string(device), 1}});
            fastllm::Data table(fastllm::FLOAT32, {4, 8});
            table.Allocate(false);
            table.lockInCPU = true;
            float *values = reinterpret_cast<float *>(table.cpuData);
            for (int i = 0; i < 32; ++i) values[i] = i * 0.25f;
            for (bool direct : {false, true}) {
                fastllm::Data ids(fastllm::FLOAT32, {1, 2});
                ids.Allocate(false);
                reinterpret_cast<float *>(ids.cpuData)[0] = 3;
                reinterpret_cast<float *>(ids.cpuData)[1] = 1;
                ids.ToDevice(fastllm::CUDA, std::vector<int>{device});
                fastllm::Data output;
                if (direct) fastllm::EmbeddingDirect(ids, table, output);
                else fastllm::Embedding(ids, table, output);
                output.ToDevice(fastllm::CPU);
                const float *actual = reinterpret_cast<float *>(output.cpuData);
                for (int i = 0; i < 16; ++i)
                    if (actual[i] != values[(i < 8 ? 3 : 1) * 8 + i % 8]) ++failures;
                if (table.dataDevice != fastllm::CPU || table.cudaData != nullptr ||
                    !fastllm::GetCudaEmbedding() || fastllm::GetCudaEmbeddingRequested()) ++failures;
            }
        }
        fastllm::SetCudaGraph(false);
        if (failures) return 1;
        std::puts("PASS: locked CPU embedding with CUDA Graph on both GPUs");
        std::puts("PASS: source-device CPU read and CopyFrom preserve CUDA placement and caller");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
