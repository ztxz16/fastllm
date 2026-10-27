#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "models/deepseekv41.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
struct TestModel : DeepSeekV41Model {
    using DeepSeekV41Model::LinearWithActivationQuant;
    using DeepSeekV41Model::quantizedLinearNames;
};
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static float Fp8(uint8_t code) {
    const int a = code & 127, e = a >> 3, f = a & 7;
    const float value = e ? std::ldexp(1.f + f / 8.f, e - 7) : std::ldexp(float(f), -9);
    return code & 128 ? -value : value;
}
static std::vector<float> Read(Data &x) {
    Data y; y.CopyFrom(x); ToDataType(y, DataType::FLOAT32); y.ToDevice(DataDevice::CPU);
    return {(float*)y.cpuData, (float*)y.cpuData + y.Count(0)};
}
static void Run(int rows, int inner, int columns) {
    std::mt19937 rng(1234);
    std::vector<float> values((size_t)rows * inner), weights((size_t)columns * inner);
    for (size_t i=0; i<values.size(); ++i)
        values[i] = std::ldexp(float(int(rng()%199)-99)/32.f, int((i/32)%7)-4);
    Data input(DataType::FLOAT32, {rows,inner}, values), quant;
    ToDataType(input, DataType::BFLOAT16); input.ToDevice(DataDevice::CUDA);
    quant.dataType=DataType::BFLOAT16; quant.Resize({rows,inner}); quant.ToDevice(DataDevice::CUDA);
    Check(FastllmCudaDeepSeekV41QuantizeActivation(input, quant), "quant reference failed");
    Data weight(DataType::FP8_E4M3, {columns,inner}); weight.blockM=weight.blockK=32;
    weight.scales.resize((columns/32)*(inner/32)); weight.Allocate();
    for (size_t i=0; i<weight.scales.size(); ++i) weight.scales[i]=std::ldexp(1.f,int(i%9)-12);
    for (int r=0; r<columns; ++r) for(int c=0; c<inner; ++c) {
        const size_t i=(size_t)r*inner+c;
        uint8_t code=uint8_t((rng()%119)|((rng()&1)<<7)); weight.cpuData[i]=code;
        weights[i]=Fp8(code)*weight.scales[(r/32)*(inner/32)+c/32];
    }
    Data referenceWeight(DataType::FLOAT32, {columns,inner}, weights);
    ToDataType(referenceWeight,DataType::BFLOAT16); referenceWeight.ToDevice(DataDevice::CUDA);
    weight.ToDevice(DataDevice::CUDA);
    Data output(DataType::BFLOAT16,{rows,columns}), expected(DataType::BFLOAT16,{rows,columns}), bias;
    output.ToDevice(DataDevice::CUDA); output.Allocate();
    expected.ToDevice(DataDevice::CUDA); expected.Allocate();
    Check(FastllmCudaTryMxfp8QuantizedLinear(input,weight,output), "MXFP8 rejected supported shape");
    FastllmCudaBFloat16MatMulBFloat16(quant,referenceWeight,bias,expected,rows,inner,columns);
    Check(cudaDeviceSynchronize()==cudaSuccess,"CUDA execution failure");
    auto actual=Read(output), ref=Read(expected);
    double error=0, norm=0; float maxAbs=0;
    for(size_t i=0;i<ref.size();++i) {
        Check(std::isfinite(actual[i]),"nonfinite MXFP8 output");
        double d=actual[i]-ref[i]; error+=d*d; norm+=double(ref[i])*ref[i];
        maxAbs=std::max(maxAbs,float(std::abs(d)));
    }
    const double rel=std::sqrt(error/std::max(norm,1e-20));
    std::printf("MXFP8 M=%d N=%d K=%d relative_rms=%.8g max_abs=%.8g\n",rows,columns,inner,rel,maxAbs);
    Check(rel<0.004,"MXFP8 differs from quantized BF16 reference");
    Data fused;
    Check(FastllmCudaTryMxfp8QuantizedLinear(quant,weight,fused),"materialized input quantization rejected");
    Check(Read(fused)==actual,"fusing input quantization changed output");
    if (rows == 128) {
        TestModel model;
        auto &projectionWeight = model.weight["projection.weight"];
        projectionWeight.CopyFrom(weight);
        // CopyFrom copies tensor storage; checkpoint metadata is set separately.
        projectionWeight.blockM = weight.blockM; projectionWeight.blockK = weight.blockK;
        projectionWeight.scales = weight.scales;
        model.quantizedLinearNames.insert("projection.weight");
        Data scratch, projection;
        model.LinearWithActivationQuant(input,"projection.weight",projection,false,&scratch);
        Check(Read(projection)==actual && scratch.dims.empty(),"model fused dispatch mismatch");
        model.LinearWithActivationQuant(input,"projection.weight",projection,false,&scratch,false,nullptr,true);
        Check(Read(projection)==actual && Read(scratch)==Read(quant),"shared quantization scratch not preserved");
        model.LinearWithActivationQuant(scratch,"projection.weight",projection,false,nullptr,true);
        Check(Read(projection)==actual,"sibling projection scratch reuse mismatch");
    }
    setenv("FASTLLM_CUTLASS_MXFP8","0",1);
    Check(!FastllmCudaTryMxfp8QuantizedLinear(input,weight,output),"disable switch ignored");
    setenv("FASTLLM_CUTLASS_MXFP8","1",1);
    FastllmCudaReleaseMxfp8CutlassCache(&weight);
    Check(FastllmCudaTryMxfp8QuantizedLinear(input,weight,output),"scale cache recreation failed");
    Check(cudaDeviceSynchronize()==cudaSuccess,"cache recreation execution failed");
    Data replacement; replacement.CopyFrom(weight);
    weight.CopyFrom(replacement); weight.scales[0]=0.3f;
    Check(!FastllmCudaTryMxfp8QuantizedLinear(input,weight,output),"CopyFrom retained stale scales or accepted non-UE8M0 scale");
}
static void WoABf16(int rows, int groups, int rank, int inner, bool benchmark) {
    std::mt19937 rng(9876);
    std::vector<float> values((size_t)rows*groups*inner), weights((size_t)groups*rank*inner);
    for (auto &v: values) v=float(int(rng()%199)-99)/32.f;
    for (size_t i=0; i<weights.size(); ++i)
        weights[i]=Fp8(uint8_t((rng()%119)|((rng()&1)<<7))) * std::ldexp(1.f,int((i/32)%7)-12);
    Data input(DataType::FLOAT32,{1,rows,groups,inner},values), quant;
    ToDataType(input,DataType::BFLOAT16); input.ToDevice(DataDevice::CUDA);
    quant.dataType=DataType::BFLOAT16; quant.Resize(input.dims);
    quant.ToDevice(DataDevice::CUDA); quant.Allocate();
    Check(FastllmCudaDeepSeekV41QuantizeActivation(input,quant),"WoA block32 quantization failed");
    Data weight(DataType::FLOAT32,{groups*rank,inner},weights);
    ToDataType(weight,DataType::BFLOAT16); weight.ToDevice(DataDevice::CUDA);
    Data expected, output;
    Check(FastllmCudaDeepSeekV4WoA(quant,weight,groups,rank,expected,false),"BF16 WoA reference failed");
    Check(FastllmCudaTryMxfp8WoA(quant,weight,groups,rank,output),"lossless BF16 weight rejected");
    auto actual=Read(output),ref=Read(expected); double error=0,norm=0;
    for(size_t i=0;i<ref.size();++i) { double d=actual[i]-ref[i]; error+=d*d; norm+=double(ref[i])*ref[i]; }
    const double rel=std::sqrt(error/std::max(norm,1e-20));
    std::printf("WoA lossless BF16 M=%d G=%d N=%d K=%d relative_rms=%.8g\n",rows,groups,rank,inner,rel);
    Check(std::isfinite(rel)&&rel<0.004,"lossless BF16 WoA mismatch");
    if (benchmark) {
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
        for (int mode=0; mode<2; ++mode) {
            float total=0;
            for (int i=0; i<8; ++i) {
                cudaEventRecord(start,cudaStreamPerThread);
                Check(mode ? FastllmCudaTryMxfp8WoA(quant,weight,groups,rank,output) :
                    FastllmCudaDeepSeekV4WoA(quant,weight,groups,rank,output,false),"WoA benchmark failed");
                cudaEventRecord(stop,cudaStreamPerThread); cudaEventSynchronize(stop);
                float ms=0; cudaEventElapsedTime(&ms,start,stop); if(i>1) total+=ms;
            }
            std::printf("WoA BENCH M=%d G=%d K=%d mode=%s ms=%.5f\n",rows,groups,inner,
                        mode ? "mxfp8" : "bf16",total/6);
        }
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
    FastllmCudaReleaseMxfp8CutlassCache(&weight);
    Check(FastllmCudaTryMxfp8WoA(quant,weight,groups,rank,output),"lossless cache recreation failed");
    std::fill(weights.begin(),weights.end(),1.f); weights[0]=1.0078125f;
    Data inexact(DataType::FLOAT32,{groups*rank,inner},weights);
    ToDataType(inexact,DataType::BFLOAT16); inexact.ToDevice(DataDevice::CUDA);
    Check(!FastllmCudaTryMxfp8WoA(quant,inexact,groups,rank,output),"lossy BF16 reconstruction accepted");
    Check(!FastllmCudaTryMxfp8WoA(quant,inexact,groups,rank,output),"cached lossy reconstruction accepted");
}
static void FrozenAllocatorFallback() {
    std::vector<float> values(128*2*128,1.f), weights(256*128,1.f);
    Data input(DataType::FLOAT32,{1,128,2,128},values), weight(DataType::FLOAT32,{256,128},weights), output;
    ToDataType(input,DataType::BFLOAT16); input.ToDevice(DataDevice::CUDA);
    ToDataType(weight,DataType::BFLOAT16); weight.ToDevice(DataDevice::CUDA);
    Data denseInput(DataType::BFLOAT16,{128,128}), denseOutput(DataType::BFLOAT16,{128,128});
    denseInput.ToDevice(DataDevice::CUDA); denseInput.Allocate();
    denseOutput.ToDevice(DataDevice::CUDA); denseOutput.Allocate();
    Data fp8(DataType::FP8_E4M3,{128,128}); fp8.blockM=fp8.blockK=32;
    fp8.scales.assign(16,1.f); fp8.Allocate(); std::memset(fp8.cpuData,0,fp8.Count(0));
    fp8.ToDevice(DataDevice::CUDA);
    Check(cudaDeviceSynchronize()==cudaSuccess,"pre-freeze execution failed");
    DisableCudaMalloc();
    Check(!FastllmCudaTryMxfp8WoA(input,weight,2,128,output),"WoA bypassed allocation freeze");
    Check(!FastllmCudaTryMxfp8QuantizedLinear(denseInput,fp8,denseOutput),
          "linear scale cache bypassed allocation freeze");
    Check(!FastllmCudaGetThreadError() && cudaGetLastError()==cudaSuccess,"optional cache failure poisoned CUDA state");
    std::puts("PASS: optional caches respect allocation freeze and preserve fallback state");
}
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    if (FastllmCudaRuntimeArch()<120 || FastllmCudaRuntimeArch()>121) return 77;
    try {
        setenv("FASTLLM_CUTLASS_MXFP8","1",1);
        if (argc > 1 && !std::strcmp(argv[1], "--freeze")) {
            Check(GetFastllmEnv().cudaMemCheck,"freeze test requires FASTLLM_CUDA_MEM_CHECK=1");
            FrozenAllocatorFallback(); return 0;
        }
        Run(128,128,128); Run(131,160,96); Run(256,512,384);
        Run(1024,5120,1024); Run(16384,512,1024); Run(32768,128,4096);
        const bool benchmark = argc > 1 && !std::strcmp(argv[1], "--bench");
        WoABf16(131,2,128,256,benchmark); WoABf16(128,8,1024,4096,benchmark);
        WoABf16(16384,8,1024,4096,benchmark);
        std::puts("PASS: SM120 MXFP8 linear");
    } catch(const std::exception &e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
