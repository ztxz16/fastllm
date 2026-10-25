// Finite BF16 values, changing token IDs and graph replay on every CUDA device.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Allocate(Data &data, DataType type, const std::vector<int> &dims, int device) {
    data.dataType = type;
    data.UpdateUnitSize();
    data.dataDevice = DataDevice::CUDA;
    data.dataDeviceIds = {device};
    data.Resize(dims);
    data.Allocate(false);
}

__global__ void RoundTrip(const __nv_bfloat16 *weight, __nv_bfloat16 *output, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(__bfloat162float(weight[i]));
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        constexpr int vocab = 257, dim = 5120, hc = 4, rounds = 20;
        for (int device = 0; device < count; ++device) {
            Cuda(cudaSetDevice(device));
            Data weight, roundtrip;
            Allocate(weight, BFLOAT16, {vocab, dim}, device);
            Allocate(roundtrip, BFLOAT16, {vocab, dim}, device);
            std::vector<uint16_t> host(vocab * dim), rt(host.size());
            for (size_t i = 0; i < host.size(); ++i) {
                host[i] = uint16_t(i * 1327 + device * 17);
                if ((host[i] & 0x7f80) == 0x7f80) host[i] &= 0xff7f;
            }
            Cuda(cudaMemcpy(weight.cudaData, host.data(), host.size() * 2, cudaMemcpyHostToDevice));
            RoundTrip<<<(host.size() + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)weight.cudaData,
                (__nv_bfloat16 *)roundtrip.cudaData, host.size());
            Cuda(cudaMemcpy(rt.data(), roundtrip.cudaData, rt.size() * 2, cudaMemcpyDeviceToHost));
            Check(rt == host, "BF16 float round-trip changed a finite value");
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32}) {
                Data ids, embed, hidden;
                Allocate(ids, FLOAT32, {1, batch}, device);
                Allocate(embed, BFLOAT16, {1, batch, 1, dim}, device);
                Allocate(hidden, BFLOAT16, {1, batch, hc, dim}, device);
                std::vector<float> tokens(batch, 0);
                std::vector<uint16_t> result(batch * hc * dim);
                Cuda(cudaMemcpy(ids.cudaData, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice));
                auto run = [&] {
                    Check(FastllmCudaEmbeddingDirect(ids, weight, embed), "embedding rejected");
                    FastllmCudaRepeat(embed.cudaData, hidden.cudaData, batch, hc,
                                      dim * 2, dim * hc * 2, dim * 2, dim * 2);
                };
                run();
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                cudaGraph_t graph;
                cudaGraphExec_t exec;
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                run();
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                for (int r = 0; r < rounds; ++r) {
                    for (int i = 0; i < batch; ++i)
                        tokens[i] = r == 0 ? 0 : r == 1 ? vocab - 1 : (r * 19 + i * 37) % vocab;
                    Cuda(cudaMemcpy(ids.cudaData, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice));
                    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                    Cuda(cudaMemcpy(result.data(), hidden.cudaData, result.size() * 2, cudaMemcpyDeviceToHost));
                    for (int i = 0; i < batch; ++i)
                        for (int k = 0; k < hc; ++k)
                            for (int d = 0; d < dim; ++d)
                                Check(result[(i * hc + k) * dim + d] == host[(int)tokens[i] * dim + d],
                                      "embedding/repeat graph mismatch");
                }
                Cuda(cudaGraphExecDestroy(exec));
                Cuda(cudaGraphDestroy(graph));
            }
        }
        std::cout << "PASS " << count << " GPUs x 10 shapes x 20 changing-token graph replays; "
                     "finite BF16 round-trip and HC replicas exact" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
