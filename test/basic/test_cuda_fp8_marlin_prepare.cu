#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Check(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }
float Decode(unsigned char v) {
    int exponent = (v >> 3) & 15, mantissa = v & 7;
    float x = exponent ? std::ldexp(1.0f + mantissa / 8.0f, exponent - 7)
                       : std::ldexp(mantissa / 8.0f, -6);
    return v & 128 ? -x : x;
}
bool Run(int device, int k, int n) {
    using namespace fastllm;
    FastllmCudaSetDevice(device);
    std::mt19937 rng(123 + device);
    Data weight(FP8_E4M3, {n,k}), referenceWeight(FLOAT16,{n,k}), bias;
    weight.blockM = weight.blockK = 128;
    weight.scales.resize((n/128)*(k/128));
    for (float &s : weight.scales) s = float(1 + rng()%9) / 4096.0f;
    weight.Allocate();referenceWeight.Allocate();
    auto *bytes = reinterpret_cast<unsigned char*>(weight.cpuData);
    auto *halfs = reinterpret_cast<half*>(referenceWeight.cpuData);
    for (int row=0;row<n;++row) for (int col=0;col<k;++col) {
        size_t i=(size_t)row*k+col;
        bytes[i]=static_cast<unsigned char>((rng()%119)|((rng()&1)<<7));
        halfs[i]=__float2half(Decode(bytes[i])*weight.scales[(row/128)*(k/128)+col/128]);
    }
    Data lazy; lazy.CopyFrom(weight); lazy.scales=weight.scales; lazy.blockM=lazy.blockK=128;
    lazy.ToDevice(CUDA,std::vector<int>{device});
    weight.ToDevice(CUDA,std::vector<int>{device});referenceWeight.ToDevice(CUDA,std::vector<int>{device});
    if (!FastllmCudaPrepareFp8MarlinLayout(weight)) {
        std::puts("SKIP: explicit FP8 Marlin layout unsupported by this build/device");
        return false;
    }
    Require(FastllmCudaHasFp8MarlinLayout(weight), "prepared FP8 layout missing");
    auto *packed=weight.cudaData;
    Require(FastllmCudaPrepareFp8MarlinLayout(weight) && packed==weight.cudaData,
            "FP8 preparation is not idempotent");
    cublasHandle_t handle;Require(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS,"cublas create");
    Require(cublasSetStream(handle,cudaStreamPerThread)==CUBLAS_STATUS_SUCCESS,"cublas stream");
    for(int m : {1,2,4,8,10,16,24,32,64,128}) {
        Data input(FLOAT16,{m,k}), output(FLOAT16,{m,n}), expected(FLOAT16,{m,n}), legacy(FLOAT16,{m,n});
        input.Allocate();auto *a=reinterpret_cast<half*>(input.cpuData);
        for (int i=0;i<m*k;++i) a[i]=__float2half(float(int(rng()%201)-100)/64);
        input.ToDevice(CUDA,std::vector<int>{device});output.ToDevice(CUDA,{device},false);output.Allocate();
        expected.ToDevice(CUDA,{device},false);expected.Allocate();
        Require(FastllmCudaTryMarlinHalfMatMulFloatFP8E4M3(input,weight,bias,output,m,k,n), "prepared FP8 dispatch failed");
        legacy.ToDevice(CUDA,{device},false);legacy.Allocate();
        Require(FastllmCudaHalfMatMulFloatFP8E4M3(input,lazy,bias,legacy,m,k,n),"legacy FP8 dispatch failed");
        Require(FastllmCudaHasFp8MarlinLayout(lazy),"legacy warmup did not prepare Marlin");
        const float alpha=1,beta=0;
        Require(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,
            referenceWeight.cudaData,CUDA_R_16F,k,input.cudaData,CUDA_R_16F,k,&beta,
            expected.cudaData,CUDA_R_16F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS,"reference gemm failed");
        std::vector<half> actual(m*n),reference(m*n),old(m*n);
        Check(cudaMemcpy(actual.data(),output.cudaData,actual.size()*2,cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(reference.data(),expected.cudaData,reference.size()*2,cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(old.data(),legacy.cudaData,old.size()*2,cudaMemcpyDeviceToHost));
        double error2=0,reference2=0;float maxError=0;
        for (size_t i=0;i<actual.size();++i) {
            float x=__half2float(actual[i]),y=__half2float(reference[i]),error=std::fabs(x-y);
            Require(std::isfinite(x),"prepared FP8 nonfinite output");
            Require(x==__half2float(old[i]),"explicit preparation differs from existing lazy Marlin");
            error2+=double(error)*error;reference2+=double(y)*y;maxError=std::max(maxError,error);
        }
        // Existing Marlin accumulation differs from FP32 cuBLAS; the strict
        // regression oracle is bitwise equality to lazy Marlin above.
        Require(error2<=2.5e-5*reference2,"prepared FP8 relative RMS tolerance failed");
        std::printf("device=%d M=%d N=%d K=%d max_abs=%g relative_rms=%g PASS\n",device,m,n,k,maxError,std::sqrt(error2/reference2));
        std::fflush(stdout);

    }
    cublasDestroy(handle);
    return true;
}
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    try {
        for(int device=0;device<devices;++device) {
            if (!Run(device,512,384) || !Run(device,5120,8704) || !Run(device,8704,5120)) return 77;
        }
        std::puts("FP8 explicit Marlin preparation: PASS");
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
