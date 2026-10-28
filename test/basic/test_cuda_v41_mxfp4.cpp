#include "fastllm.h"
#include "utils.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static std::vector<float> Read(Data &data) {
    Require(data.dataType == BFLOAT16, "expected BF16 tensor");
    std::vector<uint16_t> bits(data.Count(0));
    if (data.dataDevice == CUDA)
        Require(cudaMemcpy(bits.data(),data.cudaData,bits.size()*2,cudaMemcpyDeviceToHost)==cudaSuccess,"read failed");
    else std::memcpy(bits.data(),data.cpuData,bits.size()*2);
    std::vector<float> result(bits.size());
    for (size_t i=0;i<bits.size();++i) { uint32_t value=uint32_t(bits[i])<<16; std::memcpy(&result[i],&value,4); }
    return result;
}
static float Time(const std::function<void()> &run) {
    cudaEvent_t begin, end;
    cudaEventCreate(&begin); cudaEventCreate(&end);
    run(); cudaDeviceSynchronize();
    cudaEventRecord(begin, cudaStreamPerThread);
    for (int i = 0; i < 10; ++i) run();
    cudaEventRecord(end, cudaStreamPerThread);
    Require(cudaEventSynchronize(end) == cudaSuccess, "GEMM failed asynchronously");
    float ms = 0; cudaEventElapsedTime(&ms, begin, end);
    cudaEventDestroy(begin); cudaEventDestroy(end);
    return ms / 10;
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices ||
            FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() > 121) return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        unsetenv("FASTLLM_CUTLASS_MXFP4");
        Data scratch;
        int checks = 0;
        for (auto shape : std::vector<std::vector<int>>{{128,256,256}, {131,256,256},
                {256,4608,4096}, {768,4608,4096}, {1024,4608,4096},
                {256,4096,2304}, {768,4096,2304}, {1024,4096,2304},
                {256,4608,5120}, {256,5120,2304}, {128,256,256}}) {
            const int m=shape[0], n=shape[1], k=shape[2];
            std::mt19937 rng(917 + m);
            std::vector<float> values(size_t(m)*k);
            for (size_t i=0;i<values.size();++i)
                values[i] = float(int(rng()%20001)-10000)/10000.f * std::ldexp(1.f, int(i/32%9)-4);
            Data input(FLOAT32,{m,k},values), weight(NVFP4_BLOCK_32_E8M0,{n,k}), expected, output;
            ToDataType(input,BFLOAT16); input.ToDevice(CUDA,std::vector<int>{0});
            Require(FastllmCudaDeepSeekV41QuantizeActivation(input,input), "quantization rejected");
            weight.Allocate(false);
            for (int block=0;block<n*(k/32);++block) {
                for (int i=0;i<16;++i) weight.cpuData[size_t(block)*17+i]=rng()>>24;
                weight.cpuData[size_t(block)*17+16] = 119 + rng()%8;
            }
            weight.ToDevice(CUDA,std::vector<int>{0});
            expected.ToDevice(CUDA,std::vector<int>{0});
            auto reference=[&] { DoCudaLinearReshape(input,weight,expected); DoCudaLinear(input,weight,*GetEmptyData(),expected); };
            auto native=[&] { Require(FastllmCudaV41Mxfp4Linear(input,weight,output,scratch),"native GEMM rejected"); };
            reference(); native();
            Require(cudaDeviceSynchronize() == cudaSuccess, "projection execution failed");
            auto a=Read(output), b=Read(expected);
            double error=0,norm=0,maxError=0,peak=0;
            for (size_t i=0;i<a.size();++i) {
                Require(std::isfinite(a[i]),"nonfinite native output");
                const double d=double(a[i])-b[i]; error+=d*d; norm+=double(b[i])*b[i];
                maxError=std::max(maxError,std::abs(d)); peak=std::max(peak,double(std::abs(b[i])));
            }
            const double relative=std::sqrt(error/std::max(norm,1e-30));
            Require(relative<.003 && maxError<=.008*peak+1e-5,"native GEMM exceeds BF16 reference tolerance");
            const float baselineMs=Time(reference), nativeMs=Time(native);
            std::printf("M=%d N=%d K=%d rel_l2=%.8g max_relative=%.8g baseline_ms=%.6f native_ms=%.6f speedup=%.3f\n",
                m,n,k,relative,maxError/std::max(peak,1e-30),baselineMs,nativeMs,baselineMs/nativeMs);
            std::vector<uint8_t> weightAfter(weight.GetBytes());
            Require(cudaMemcpy(weightAfter.data(), weight.cudaData, weightAfter.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "weight read failed");
            Require(std::memcmp(weightAfter.data(), weight.cpuData, weightAfter.size()) == 0,
                    "native GEMM changed compact weights shared with the host/cache");
            setenv("FASTLLM_CUTLASS_MXFP4", "0", 1);
            Require(!FastllmCudaV41Mxfp4Linear(input,weight,output,scratch), "disabled GEMM must fall back");
            unsetenv("FASTLLM_CUTLASS_MXFP4");
            ++checks;
            input.Resize({1,k});
            Require(!FastllmCudaV41Mxfp4Linear(input,weight,output,scratch),"decode must retain its existing path");
        }
        std::printf("PASS V4.1 W4A8: %d projection shapes, compact weights and reused scratch\n",checks);
    } catch(const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
