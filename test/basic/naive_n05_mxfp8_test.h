#pragma once
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <vector>

inline void FillVerifyMxWeight(fastllm::Data &w, unsigned seed) {
    w.blockK = 1;
    w.blockM = 32;
    std::vector<uint8_t> data(w.Count(0));
    w.scales.resize(data.size()/32);
    for (size_t i=0;i<data.size();++i) {
        seed = 1664525u*seed + 1013904223u;
        data[i] = uint8_t(0x20u+(seed>>24)%48u) | uint8_t((seed>>16)&128u);
        if (i%32==0) w.scales[i/32]=std::ldexp(1.f,-9+int((seed>>22)%4));
    }
    if (cudaMemcpy(w.cudaData,data.data(),data.size(),cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("could not initialize MXFP8 projection test weight");
    fastllm::Data bias;
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(w,bias,w.dims[0]);
}

// Catch accidental W8A16 reference dispatch in a cold test thread.
inline bool VerifyMxLinearReference(const fastllm::Data &x, fastllm::Data &w,
        const fastllm::Data &bias, fastllm::Data &out, int rows, int k, int n) {
    if (!FastllmCudaVerifyMxfp8Scales(w,n,k,FastllmCudaGetDevice()))
        throw std::runtime_error("MXFP8 reference workspace or scales not ready");
    return FastllmCudaBFloat16MatMulFP8E4M3(x,w,bias,out,rows,k,n);
}
