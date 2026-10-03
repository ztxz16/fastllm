// Compare the model-facing prefill dispatch with the original recurrence.
// Auxiliary outputs force the reference path without a runtime feature switch.
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
static unsigned Mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16);
}
static void Fill(Data &d, unsigned seed, float amplitude, float offset = 0) {
    d.Allocate(false);
    for (size_t i = 0; i < d.Count(0); ++i) {
        float value = (int(Mix(unsigned(i) + seed) % 20001) - 10000) * (amplitude / 10000) + offset;
        if (d.dataType == DataType::BFLOAT16) ((__nv_bfloat16*)d.cpuData)[i] = __float2bfloat16_rn(value);
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
struct Fixture {
    int sequence;
    Data q, k, v, g, b, a, dt, initial, reference, actual, expected, output, decay, beta;
    Fixture(int batch, int sequence, int heads, int dimension = 128,
            float gate = -2, bool channelwise = false)
        : sequence(sequence),
          q(DataType::BFLOAT16, {batch,sequence,heads,dimension}), k(q.dataType,q.dims),
          v(q.dataType,q.dims), g(q.dataType,q.dims), b(DataType::FLOAT32,{batch,sequence,heads}),
          a(DataType::FLOAT32,{channelwise ? dimension : heads}), dt(DataType::FLOAT32,{heads,dimension}),
          initial(DataType::FLOAT32,{batch,heads,dimension,dimension}), reference(initial.dataType,initial.dims),
          actual(initial.dataType,initial.dims), expected(q.dataType,q.dims), output(q.dataType,q.dims),
          decay(DataType::FLOAT32,q.dims), beta(DataType::FLOAT32,b.dims) {
        Fill(q,17,2); Fill(k,29,2); Fill(v,41,3); Fill(g,53,3,gate);
        Fill(b,67,12); Fill(a,79,1); Fill(dt,89,.2f); Fill(initial,101,.25f);
        for (Data *d : {&reference,&actual,&expected,&output,&decay,&beta}) Allocate(*d);
    }
    void Reset(Data &state) {
        Cuda(cudaMemcpyAsync(state.cudaData,initial.cudaData,initial.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    }
    void Run(bool auxiliary, bool initialize = false, bool normalize = true, bool round = true,
             bool stateOnly = false, int tokenLimit = -1, float lowerBound = -5) {
        Check(FastllmCudaKimiK3RecurrentKDA(q,k,v,g,b,a,dt,
            auxiliary ? reference : actual, auxiliary ? expected : output,decay,beta,
            lowerBound,initialize,tokenLimit,stateOnly,auxiliary,normalize,round), "KDA rejected input");
    }
    void Compare(bool stateOnly = false) {
        Cuda(cudaDeviceSynchronize());
        auto stateBytes = Read(actual);
        Check(Read(reference) == stateBytes, "FP32 state differs from original recurrence");
        for (size_t i = 0; i < stateBytes.size(); i += sizeof(float)) {
            float value; std::memcpy(&value,stateBytes.data()+i,sizeof(value));
            Check(std::isfinite(value), "nonfinite state");
        }
        if (!stateOnly) {
            auto outputBytes = Read(output);
            Check(Read(expected) == outputBytes, "BF16 output differs from original recurrence");
            for (size_t i = 0; i < outputBytes.size(); i += sizeof(__nv_bfloat16)) {
                __nv_bfloat16 value; std::memcpy(&value,outputBytes.data()+i,sizeof(value));
                Check(std::isfinite(__bfloat162float(value)), "nonfinite output");
            }
        }
    }
};
static void RunCase(int batch, int sequence, int heads, int dimension = 128,
                    float gate = -2, bool initialize = false, bool normalize = true,
                    bool round = true, bool channelwise = false, bool stateOnly = false,
                    int tokenLimit = -1, float lowerBound = -5) {
    Fixture f(batch,sequence,heads,dimension,gate,channelwise);
    auto beforeQ = Read(f.q), beforeK = Read(f.k), beforeV = Read(f.v), beforeG = Read(f.g);
    Cuda(cudaMemset(f.decay.cudaData,0xff,f.decay.GetBytes()));
    Cuda(cudaMemset(f.beta.cudaData,0xff,f.beta.GetBytes()));
    if (initialize) {
        Cuda(cudaMemset(f.reference.cudaData,0xff,f.reference.GetBytes()));
        Cuda(cudaMemset(f.actual.cudaData,0xff,f.actual.GetBytes()));
    } else {
        f.Reset(f.reference); f.Reset(f.actual);
    }
    f.Run(true,initialize,normalize,round,stateOnly,tokenLimit,lowerBound);
    if (!stateOnly) {
        // Prove that the oracle actually executed the auxiliary-output path.
        std::vector<float> decay(f.decay.Count(0)), beta(f.beta.Count(0));
        Cuda(cudaMemcpy(decay.data(),f.decay.cudaData,f.decay.GetBytes(),cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(beta.data(),f.beta.cudaData,f.beta.GetBytes(),cudaMemcpyDeviceToHost));
        for (float x : decay) Check(std::isfinite(x), "reference did not write decay");
        for (float x : beta) Check(std::isfinite(x), "reference did not write beta");
    }
    f.Run(false,initialize,normalize,round,stateOnly,tokenLimit,lowerBound); f.Compare(stateOnly);
    Check(beforeQ == Read(f.q) && beforeK == Read(f.k) && beforeV == Read(f.v) && beforeG == Read(f.g), "input mutated");
    std::printf("B=%d T=%d H=%d D=%d gate=%g init=%d norm=%d round=%d channelwise=%d stateOnly=%d: bitwise equal\n",
        batch,sequence,heads,dimension,gate,initialize,normalize,round,channelwise,stateOnly);
}
static void CheckColdGraph() {
    Fixture f(2,65,3);
    f.Reset(f.reference); f.Run(true); Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    f.Run(false);
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int i = 0; i < 3; ++i) {
        f.Reset(f.actual); Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); f.Compare();
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    std::puts("cold CUDA Graph capture and 3 replays: bitwise equal");
}
static void CheckNormalizationEdges() {
    Fixture f(1,65,3);
    std::vector<__nv_bfloat16> values(f.q.Count(0));
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = __float2bfloat16_rn(i % 3 == 0 ? 0.f : (i % 3 == 1 ? 1e-8f : -1e-8f));
    Cuda(cudaMemcpy(f.q.cudaData,values.data(),f.q.GetBytes(),cudaMemcpyHostToDevice));
    Cuda(cudaMemset(f.k.cudaData,0,f.k.GetBytes()));
    f.Reset(f.reference); f.Reset(f.actual); f.Run(true); f.Run(false); f.Compare();
    std::puts("zero/tiny QK normalization: bitwise equal");
}
static void CheckContinuation() {
    Fixture f(1,1024,8,128,-18);
    f.Reset(f.reference); f.Run(true); f.Reset(f.actual);
    std::vector<unsigned char> joined;
    int offset = 0;
    for (int count : {63,1,65,895}) {
        Fixture part(1,count,8,128,-18);
        for (auto pair : {std::make_pair(&part.q,&f.q),std::make_pair(&part.k,&f.k),
                          std::make_pair(&part.v,&f.v),std::make_pair(&part.g,&f.g),std::make_pair(&part.b,&f.b)}) {
            size_t bytesPerToken = pair.second->GetBytes() / f.sequence;
            Cuda(cudaMemcpyAsync(pair.first->cudaData,(char*)pair.second->cudaData + offset * bytesPerToken,
                count * bytesPerToken,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        }
        Cuda(cudaMemcpyAsync(part.actual.cudaData,f.actual.cudaData,f.actual.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        part.Run(false); auto bytes = Read(part.output); joined.insert(joined.end(),bytes.begin(),bytes.end());
        Cuda(cudaMemcpyAsync(f.actual.cudaData,part.actual.cudaData,f.actual.GetBytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread)); offset += count;
    }
    Check(joined == Read(f.expected), "continued prefill output differs");
    Check(Read(f.actual) == Read(f.reference), "continued prefill state differs");
    std::puts("63+1+65+895 state continuation: bitwise equal");
}
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        FastllmCudaSetDevice(0); SetThreads(4);
        CheckColdGraph(); // Must precede allocation of the prefill scratch buffer.
        bool quick = argc == 2 && std::string(argv[1]) == "--sanitizer";
        for (int t : {1,63,64,65}) RunCase(1,t,3);
        RunCase(1,1,64); RunCase(1,1,64,128,-40,true);
        CheckNormalizationEdges();
        RunCase(2,65,3); RunCase(1,65,1,128,-40,true);
        RunCase(1,65,8,64); RunCase(1,65,8,128,-2,false,false);
        RunCase(1,65,8,128,-2,false,true,false);
        RunCase(1,65,8,128,-2,false,true,true,true);
        RunCase(1,65,8,128,-2,false,true,true,false,true,31);
        RunCase(1,65,8,128,-2,false,true,true,false,false,-1,-2.75f);
        if (!quick) {
            RunCase(1,256,8,128,-18); RunCase(1,1024,64);
            RunCase(1,1024,64,128,-40); CheckContinuation();
        }
        std::puts("KDA prefill regression passed"); return 0;
    } catch (const std::exception &error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
}
