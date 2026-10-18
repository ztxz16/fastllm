// FP8 WoA: generic Tensor Core dispatch, independent FP64 oracle, and fallback.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static float Decode(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.f + m / 8.f, e - 7);
    return b & 128 ? -v : v;
}
static float Quantize(float x, float scale) {
    const float v = std::min(448.f, std::abs(x / scale));
    // Independent nearest-even search over the finite positive E4M3 grid.
    int lo=0, hi=126;
    while(lo<hi) {
        int mid=(lo+hi)/2;
        if(Decode(mid)<v) lo=mid+1; else hi=mid;
    }
    int best=lo;
    if(lo>0) {
        const float left=v-Decode(lo-1), right=Decode(lo)-v;
        if(left<right || (left==right && !((lo-1)&1))) best=lo-1;
    }
    return std::copysign(Decode(best) * scale, x);
}
static std::vector<float> Read(const Data &input) {
    Data cpu; cpu.CopyFrom(input); cpu.ToDevice(DataDevice::CPU);
    std::vector<float> result(cpu.Count(0));
    for (size_t i=0; i<result.size(); ++i)
        result[i] = BFloat16BitsToFloat32(((uint16_t*)cpu.cpuData)[i]);
    return result;
}
struct Case { int batch, seq, groups, rank, cols, blockRows, blockCols; };
static void Check(Case c, DataType type, float amplitude, bool arbitraryScales, bool bench) {
    const int tokens=c.batch*c.seq, fullDim=c.groups*c.cols, outDim=c.groups*c.rank;
    std::mt19937 rng(392017 + tokens + c.cols);
    std::normal_distribution<float> normal(0, .25f);
    std::vector<float> xv((size_t)tokens*fullDim);
    for (float &v:xv) {
        v=normal(rng)*amplitude;
        v=type==DataType::BFLOAT16 ? BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(v)) :
          type==DataType::FLOAT16 ? __half2float(__float2half_rn(v)) : v;
    }
    Data x(DataType::FLOAT32,{c.batch,c.seq,c.groups,c.cols},xv);
    ToDataType(x,type); x.ToDevice(DataDevice::CUDA);
    std::vector<float> qx=xv;
    if (FastllmCudaRuntimeArch()>=120) {
        for (int t=0; t<tokens; ++t) for(int begin=0; begin<fullDim; begin+=128) {
            float maximum=1e-10f;
            for (int d=begin; d<std::min(begin+128,fullDim); ++d)
                maximum=std::max(maximum,std::abs(xv[(size_t)t*fullDim+d]));
            float scale=std::exp2(std::ceil(std::log2(maximum/448.f)));
            for(int d=begin; d<std::min(begin+128,fullDim); ++d) {
                const size_t i=(size_t)t*fullDim+d;
                qx[i]=BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(Quantize(xv[i],scale)));
            }
        }
    }
    Data w(DataType::FP8_E4M3,{outDim,c.cols});
    w.blockK=c.blockRows; w.blockM=c.blockCols;
    const int sr=(outDim+c.blockRows-1)/c.blockRows, sc=(c.cols+c.blockCols-1)/c.blockCols;
    w.scales.resize((size_t)sr*sc);
    for(size_t i=0;i<w.scales.size();++i)
        w.scales[i]=std::ldexp(arbitraryScales ? 1.f+float(i%11)/17.f : 1.f, int(i%7)-11);
    w.Allocate();
    std::vector<uint8_t> packed(w.Count(0));
    for(auto &v:packed) { auto bits=rng(); v=(bits%127)|(bits&128); }
    memcpy(w.cpuData,packed.data(),packed.size());
    w.ToDevice(DataDevice::CUDA);
    Data ref,out;
    setenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM","1",1);
    Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA synchronization failed");
    auto a=std::chrono::steady_clock::now();
    Require(FastllmCudaDeepSeekV4WoA(x,w,c.groups,c.rank,ref,false),"SIMT fallback rejected");
    Require(cudaDeviceSynchronize()==cudaSuccess,"SIMT CUDA failure");
    double oldMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
    unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM");
    a=std::chrono::steady_clock::now();
    Require(FastllmCudaDeepSeekV4WoA(x,w,c.groups,c.rank,out,false),"GEMM dispatch rejected");
    Require(cudaDeviceSynchronize()==cudaSuccess,"GEMM CUDA failure");
    double newMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
    Require(out.dims==std::vector<int>({c.batch,c.seq,outDim}),"incorrect output layout");
    auto got=Read(out), baseline=Read(ref);
    size_t identical=0; double error2=0, ref2=0, peak=0;
    for(size_t i=0;i<got.size();++i) {
        Require(std::isfinite(got[i]),"nonfinite result");
        identical+=got[i]==baseline[i];
        const double diff=double(got[i])-baseline[i];
        error2+=diff*diff; ref2+=double(baseline[i])*baseline[i];
        peak=std::max(peak,std::abs(diff));
    }
    const double nrmse=std::sqrt(error2/std::max(1e-90,ref2));
    Require(nrmse<0.002,"Tensor Core/SIMT normalized error exceeds BF16 tolerance");
    if(tokens<16) Require(identical==got.size(),"small batch fallback changed bits");
    // Check all small outputs and a fixed random sample of large matrices using
    // a separately decoded, FP64 dot product. The allowance covers BF16 RNE
    // and FP32 accumulation, including near-zero cancellation.
    const size_t checks=std::min<size_t>(256,got.size());
    for(size_t sample=0;sample<checks;++sample) {
        size_t i=got.size()<=256 ? sample : (uint64_t(rng())*got.size())>>32;
        if(sample==0) i=0;
        if(sample+1==checks) i=got.size()-1;
        const int t=i/outDim, r=i%outDim, g=r/c.rank;
        double exact=0, sumAbs=0;
        for(int d=0;d<c.cols;++d) {
            const float ws=w.scales[(r/c.blockRows)*sc+d/c.blockCols];
            const float wd=__half2float(__float2half_rn(Decode(packed[(size_t)r*c.cols+d])*ws));
            const double product=double(qx[(size_t)t*fullDim+g*c.cols+d])*wd;
            exact+=product; sumAbs+=std::abs(product);
        }
        const double tolerance=.004*std::abs(exact)+2e-6*sumAbs+1e-30;
        if(std::abs(double(got[i])-exact)>tolerance) {
            std::cerr<<"oracle failure index="<<i<<" got="<<got[i]<<" exact="<<exact<<" tolerance="<<tolerance<<"\n";
            throw std::runtime_error("FP64 oracle mismatch");
        }
    }
    if(bench) {
        std::vector<double> oldTimes,newTimes;
        for(int repeat=0;repeat<5;++repeat) for(int mode=0;mode<2;++mode) {
            // Compare the supported SIMT fallback with automatic GEMM dispatch.
            if(mode==0) setenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM","1",1);
            else unsetenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM");
            a=std::chrono::steady_clock::now();
            Require(FastllmCudaDeepSeekV4WoA(x,w,c.groups,c.rank,out,false),"benchmark failure");
            Require(cudaDeviceSynchronize()==cudaSuccess,"benchmark CUDA failure");
            auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
            (mode==0?oldTimes:newTimes).push_back(ms);
        }
        std::sort(oldTimes.begin(),oldTimes.end()); std::sort(newTimes.begin(),newTimes.end());
        oldMs=oldTimes[2]; newMs=newTimes[2];
    }
    std::cout<<"CASE M="<<tokens<<" N="<<c.rank<<" K="<<c.cols<<" G="<<c.groups
      <<" block="<<c.blockRows<<"x"<<c.blockCols<<" dtype="<<int(type)<<" amplitude="<<amplitude
      <<" arbitrary_scales="<<arbitraryScales<<" nrmse="<<nrmse<<" bit_equal="<<double(identical)/got.size()
      <<" max_abs="<<peak<<" simt_ms="<<oldMs<<" gemm_ms="<<newMs<<"\n";
}
int main(int argc,char **argv) {
    try {
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        FastllmCudaSetDevice(0);
        const bool bench=argc>1 && std::string(argv[1])=="--bench";
        const Case cases[]={
            {1,1,3,33,130,32,32},{1,7,2,64,128,128,128},
            {2,4,2,64,128,32,32},{1,15,1,33,127,7,13},
            {1,16,3,33,130,32,32},{3,11,2,65,129,128,128},
            {1,127,4,32,256,32,128},{1,128,1,128,512,128,32},
            {1,129,3,17,65,7,13},{1,1025,2,64,128,32,32}
        };
        for(auto c:cases) for(DataType type:{DataType::BFLOAT16,DataType::FLOAT16,DataType::FLOAT32})
            for(bool arbitrary:{false,true}) Check(c,type,1.f,arbitrary,false);
        for(float amplitude:{0.f,1e-7f,1e6f})
            Check({1,17,2,48,256,32,32},DataType::BFLOAT16,amplitude,true,false);
        for(int m:{16,33,128})
            Check({1,m,8,1024,4096,32,32},DataType::BFLOAT16,1.f,false,bench);
        Check({1,4096,8,1024,4096,32,32},DataType::BFLOAT16,1.f,false,bench);
        Check({1,1031,1,1024,4096,128,128},DataType::BFLOAT16,1.f,true,bench);
        std::cout<<"PASS: 68 cases, FP64 oracle, fallback, tails and real WoA geometry\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
