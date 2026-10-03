// Compare the public CUDA entry point against the original serial-token kernel.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using namespace fastllm;
using Bfloat = __nv_bfloat16;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
static unsigned Mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}
static void Fill(Data &d, unsigned seed, float amplitude) {
    d.Allocate(false);
    for (size_t i = 0; i < d.Count(0); ++i) {
        float value = (int(Mix(unsigned(i) + seed) % 20001) - 10000) * (amplitude / 10000);
        if (d.dataType == DataType::BFLOAT16) ((Bfloat*)d.cpuData)[i] = __float2bfloat16_rn(value);
        else ((float*)d.cpuData)[i] = value;
    }
    d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void Allocate(Data &d) {
    d.ToDevice(DataDevice::CUDA, std::vector<int>{0}); d.Allocate(false);
}
static std::vector<unsigned char> Read(const Data &d) {
    std::vector<unsigned char> bytes(d.GetBytes());
    Cuda(cudaMemcpy(bytes.data(), d.cudaData, bytes.size(), cudaMemcpyDeviceToHost)); return bytes;
}

__global__ void SerialConv(
        const __nv_bfloat16 *input, const float *weight,
        const __nv_bfloat16 *cache, __nv_bfloat16 *output,
        int batch, int sequence, int channels, int kernelSize) {
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= batch * channels) {
        return;
    }
    int batchIndex = item / channels;
    int channel = item % channels;
    int history = kernelSize - 1;
    for (int token = 0; token < sequence; token++) {
        float value = 0.0f;
        for (int kernel = 0; kernel < kernelSize; kernel++) {
            int sourceToken = token - history + kernel;
            float sourceValue = 0.0f;
            if (sourceToken >= 0) {
                size_t sourceIndex =
                    ((size_t)batchIndex * sequence + sourceToken) *
                    channels + channel;
                sourceValue = __bfloat162float(input[sourceIndex]);
            } else if (cache != nullptr) {
                int cacheToken = history + sourceToken;
                size_t cacheIndex =
                    ((size_t)batchIndex * history + cacheToken) *
                    channels + channel;
                sourceValue = __bfloat162float(cache[cacheIndex]);
            }
            value += sourceValue *
                weight[(size_t)channel * kernelSize + kernel];
        }
        size_t outputIndex =
            ((size_t)batchIndex * sequence + token) * channels + channel;
        output[outputIndex] = __float2bfloat16_rn(
            value * (1.0f / (1.0f + expf(-value))));
    }
}

struct Fixture {
    int batch, sequence, channels, kernel;
    Data input, weight, cache, originalCache, output, expected;
    Fixture(int b, int t, int c, int k)
        : batch(b), sequence(t), channels(c), kernel(k),
          input(DataType::BFLOAT16,{b,t,c}), weight(DataType::FLOAT32,{c,1,k}),
          cache(DataType::BFLOAT16,{b,k-1,c}), originalCache(cache.dataType,cache.dims),
          output(input.dataType,input.dims), expected(input.dataType,input.dims) {
        Fill(input,17,4); Fill(weight,29,2);
        if (k > 1) { Fill(cache,41,3); Fill(originalCache,41,3); }
        Allocate(output); Allocate(expected);
    }
    void Reference(bool cached, bool initialize) {
        if (cached && initialize) Cuda(cudaMemset(originalCache.cudaData,0,originalCache.GetBytes()));
        SerialConv<<<(batch*channels+255)/256,256>>>(
            (const Bfloat*)input.cudaData,(const float*)weight.cudaData,
            cached ? (const Bfloat*)originalCache.cudaData : nullptr,
            (Bfloat*)expected.cudaData,batch,sequence,channels,kernel);
        Cuda(cudaGetLastError());
    }
    void Run(bool cached, bool initialize) {
        Check(FastllmCudaKimiK3CausalConv1D(input,weight,cached ? &cache : nullptr,output,kernel,initialize),
              "causal convolution rejected input");
    }
    void Compare(bool cached) {
        auto bytes = Read(output);
        Check(bytes == Read(expected), "convolution output differs from original kernel");
        for (size_t i = 0; i < bytes.size(); i += sizeof(Bfloat)) {
            Bfloat v; std::memcpy(&v,bytes.data()+i,sizeof(v));
            Check(std::isfinite(__bfloat162float(v)), "nonfinite convolution output");
        }
        if (!cached) return;
        // Independently construct the final history from the original cache and input.
        auto old = Read(originalCache), x = Read(input), actual = Read(cache);
        size_t rowBytes = channels * sizeof(Bfloat);
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < kernel-1; ++h) {
                int token = sequence + h - (kernel-1);
                const auto &source = token < 0 ? old : x;
                size_t sourceRow = token < 0 ? b*(kernel-1)+sequence+h : b*sequence+token;
                Check(std::memcmp(actual.data()+(b*(kernel-1)+h)*rowBytes,
                                  source.data()+sourceRow*rowBytes,rowBytes) == 0,
                      "convolution history cache differs");
            }
        }
    }
};
static void RunCase(int batch, int sequence, int channels, int kernel, bool cached, bool initialize = false) {
    Fixture f(batch,sequence,channels,kernel);
    auto input = Read(f.input), weight = Read(f.weight);
    if (initialize && cached) Cuda(cudaMemset(f.cache.cudaData,0xff,f.cache.GetBytes()));
    f.Reference(cached,initialize); f.Run(cached,initialize); f.Compare(cached);
    Check(Read(f.input) == input && Read(f.weight) == weight,"convolution mutated input or weight");
    std::printf("B=%d T=%d C=%d K=%d cache=%d init=%d: bitwise equal\n",
                batch,sequence,channels,kernel,cached,initialize);
}
static void CheckContinuation() {
    Fixture full(2,1037,259,4);
    full.Reference(true,false); full.Run(true,false); full.Compare(true);
    std::vector<unsigned char> joined(full.output.GetBytes());
    auto initial = Read(full.originalCache);
    int offset = 0;
    size_t rowBytes = full.channels * sizeof(Bfloat);
    for (int count : {1,2,7,65,962}) {
        Fixture part(2,count,259,4);
        for (int b = 0; b < full.batch; ++b)
            Cuda(cudaMemcpyAsync((char*)part.input.cudaData+b*count*rowBytes,
                (char*)full.input.cudaData+(b*full.sequence+offset)*rowBytes,
                count*rowBytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        Cuda(cudaMemcpy(part.cache.cudaData,initial.data(),initial.size(),cudaMemcpyHostToDevice));
        part.Run(true,false); auto bytes = Read(part.output); initial = Read(part.cache);
        for (int b = 0; b < full.batch; ++b)
            std::memcpy(joined.data()+(b*full.sequence+offset)*rowBytes,
                        bytes.data()+b*count*rowBytes,count*rowBytes);
        offset += count;
    }
    Check(joined == Read(full.expected),"continued convolution output differs");
    Check(initial == Read(full.cache),"continued convolution cache differs");
    std::puts("1+2+7+65+962 continuation: bitwise equal");
}
static void CheckGraph(int sequence) {
    Fixture f(2,sequence,259,4);
    f.Reference(true,true); Cuda(cudaDeviceSynchronize());
    void *graph = nullptr, *exec = nullptr;
    Check(FastllmCudaGraphBeginCapture(), "graph capture failed");
    f.Run(true,true);
    Check(FastllmCudaGraphEndCapture(&graph), "graph capture end failed");
    Check(FastllmCudaGraphInstantiate(graph,&exec), "graph instantiate failed");
    for (int i = 0; i < 3; ++i) {
        Cuda(cudaMemset(f.cache.cudaData,0xff,f.cache.GetBytes()));
        Check(FastllmCudaGraphLaunch(exec), "graph launch failed"); f.Compare(true);
    }
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    std::printf("T=%d CUDA Graph capture and 3 replays: bitwise equal\n",sequence);
}
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        FastllmCudaSetDevice(0); SetThreads(4);
        bool quick = argc == 2 && std::string(argv[1]) == "--sanitizer";
        for (int t : {1,2,3,7,8,9,63,64,65}) {
            RunCase(2,t,259,4,true); RunCase(1,t,33,4,false);
        }
        for (int k : {2,4,8}) { RunCase(2,2,17,k,true,true); RunCase(2,17,255,k,true); }
        RunCase(2,17,257,1,false); RunCase(1,1,8192,4,true);
        CheckGraph(1); CheckGraph(65); CheckContinuation();
        if (!quick) {
            RunCase(1,1024,8192,4,true); RunCase(1,1024,8192,4,true,true);
            RunCase(1,524289,1,4,true); // More token blocks than the grid.y limit.
        }
        std::puts("causal convolution regression passed"); return 0;
    } catch (const std::exception &error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
