// Native activation storage must match explicit F32 staging bit for bit.
// Exercise mixed job types, noncompact strides, guard regions, split reductions,
// >8-column batches, typed graph replay, and both modern/legacy decode dispatch.
#include "strata/kernels/exl3.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace mx = strata::kernels::exl3;
static void Check(bool v, const char* s) { if (!v) throw std::runtime_error(s); }
static void Cuda(cudaError_t e) { Check(e==cudaSuccess,cudaGetErrorString(e)); }
static uint16_t Pack(float x,int type) {
    if(type==1)return __half_as_ushort(__float2half_rn(x));
    uint32_t u;std::memcpy(&u,&x,4);
    if((u&0x7fffffff)>0x7f800000)u|=0x00400000;
    else u+=0x7fff+((u>>16)&1);
    return u>>16;
}
static float Unpack(uint16_t u,int type) {
    if(type==1)return __half2float(__ushort_as_half(u));
    uint32_t v=uint32_t(u)<<16;float x;std::memcpy(&x,&v,4);return x;
}
struct Buffer {
    void* ptr;
    explicit Buffer(size_t n) { Cuda(cudaMalloc(&ptr,n)); }
    ~Buffer(){cudaFree(ptr);}
    template<class T>T* as(){return static_cast<T*>(ptr);}
    template<class T>void put(const std::vector<T>& v){Cuda(cudaMemcpy(ptr,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    template<class T>std::vector<T> get(size_t n){std::vector<T> v(n);Cuda(cudaMemcpy(v.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static void Case(int k2,int rows,int k,int itype,int otype) {
    constexpr int n=128,jobs=2,pad=13;
    const size_t xs=k+11,ys=jobs*n+pad,count=rows*ys+pad;
    std::mt19937 rng(153+k2+rows+k+itype*7+otype*19);
    std::vector<uint32_t> tr(mx::trellis_bytes_k2(k,n,k2)/4);
    for(auto& v:tr)v=rng();
    std::vector<uint16_t> scales(k+n);
    for(auto& v:scales)v=Pack((int(rng()%63)-31)/1024.f,1);
    Buffer trellis(tr.size()*4),scale(scales.size()*2);trellis.put(tr);scale.put(scales);
    mx::Mat mat;mat.k=k;mat.n=n;mat.k2=k2;mat.trellis=trellis.as<uint32_t>();mat.suh=scale.as<uint16_t>();mat.svh=mat.suh+k;
    std::vector<float> bias(jobs*n);for(auto& v:bias)v=(int(rng()%511)-255)/32768.f;
    Buffer dbias(bias.size()*4);dbias.put(bias);
    Buffer x0(rows*xs*4),x1(rows*xs*4),x016(rows*xs*2),x116(rows*xs*2);
    Buffer ref(count*4),out(count*4),out16(count*2);
    std::vector<float> sentinel(count,-123.f);std::vector<uint16_t> sentinel16(count,0x5a5a);
    ref.put(sentinel);out.put(sentinel);out16.put(sentinel16);
    mx::MvJob f[jobs],typed[jobs];
    for(int j=0;j<jobs;++j){
        const int it=j==0?itype:(itype+1)%3;
        std::vector<float> xv(rows*xs);std::vector<uint16_t> xv16(rows*xs);
        for(size_t i=0;i<xv.size();++i){
            float value=(int(rng()%8193)-4096)/1024.f;
            xv16[i]=Pack(value,it);xv[i]=it?Unpack(xv16[i],it):value;
        }
        auto& dx=j==0?x0:x1;auto& dx16=j==0?x016:x116;dx.put(xv);dx16.put(xv16);
        auto& a=f[j];a.m=&mat;a.xf=dx.as<float>();a.y=ref.as<float>()+j*n;a.bias=j?dbias.as<float>()+j*n:nullptr;
        a.alpha=j?0.75f:1.f;a.ncol=rows;a.xf_cs=xs;a.y_cs=ys;
        auto& b=typed[j];b=a;
        if(it){b.xf=nullptr;b.x16=dx16.as<uint16_t>();b.x_bf16=it==2;}
        if(otype){b.y=nullptr;b.y16=out16.as<uint16_t>()+j*n;b.y_bf16=otype==2;}
        else b.y=out.as<float>()+j*n;
    }
    Check(mx::mv(f,jobs,cudaStreamPerThread),"reference launch");
    Check(mx::mv(typed,jobs,cudaStreamPerThread),"typed launch");
    if(k2==6 && rows==9 && itype==2 && otype==2){
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));cudaGraph_t g;cudaGraphExec_t e;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        Check(mx::mv(typed,jobs,cudaStreamPerThread),"typed graph capture");
        Cuda(cudaStreamEndCapture(cudaStreamPerThread,&g));Cuda(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
        for(int i=0;i<16;++i)Cuda(cudaGraphLaunch(e,cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));Cuda(cudaGraphExecDestroy(e));Cuda(cudaGraphDestroy(g));
    }
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));auto reference=ref.get<float>(count);
    if(otype){
        auto actual=out16.get<uint16_t>(count);
        for(size_t i=0;i<count;++i){
            bool written=i<rows*ys && i%ys<jobs*n;
            Check(actual[i]==(written?Pack(reference[i],otype):0x5a5a),"typed 16-bit output / guard mismatch");
        }
    }else{
        auto actual=out.get<float>(count);Check(!std::memcmp(actual.data(),reference.data(),count*4),"typed F32 output / guard mismatch");
    }
}
static void MoeCase(int type,bool rounded) {
    constexpr int k=256,mid=128,experts=2,k2=6;
    const size_t trbytes=mx::trellis_bytes_k2(k,mid,k2),part=trbytes+2*(k+mid),stride=3*part;
    std::mt19937 rng(901+type);
    std::vector<uint8_t> weights(experts*stride);
    mx::MoeArgs a;a.k=experts;a.n_embd=k;a.n_ff=mid;a.limit=2.f;a.round_bf16=rounded;
    for(int role=0;role<3;++role){
        a.lay.k2[role]=k2;a.lay.trellis[role]=role*part;
        a.lay.suh[role]=role*part+trbytes;a.lay.svh[role]=a.lay.suh[role]+2*(role==2?mid:k);
        for(int e=0;e<experts;++e){
            auto* base=weights.data()+e*stride+role*part;
            for(size_t i=0;i<trbytes;++i)base[i]=rng();
            for(int i=0;i<k+mid;++i){uint16_t v=Pack((int(rng()%63)-31)/128.f,1);std::memcpy(base+trbytes+2*i,&v,2);}
        }
    }
    Buffer dw(weights.size());dw.put(weights);
    std::vector<unsigned long long> ptrs{reinterpret_cast<unsigned long long>(dw.ptr),reinterpret_cast<unsigned long long>(dw.ptr)+stride};
    std::vector<float> mix{0.375f,0.625f};Buffer dp(16),dm(8);dp.put(ptrs);dm.put(mix);
    a.plan_ptr=dp.as<unsigned long long>();a.plan_w=dm.as<float>();
    std::vector<float> x(k);std::vector<uint16_t> x16(k);
    for(int i=0;i<k;++i){x16[i]=Pack((int(rng()%2049)-1024)/512.f,type);x[i]=Unpack(x16[i],type);}
    Buffer dx(k*4),dx16(k*2),hr(experts*mid*4),ht(experts*mid*4),yr(k*4),yt(k*4);dx.put(x);dx16.put(x16);
    mx::moe_gate_up(a,dx.as<float>(),hr.as<float>(),cudaStreamPerThread);
    mx::moe_down(a,hr.as<float>(),nullptr,yr.as<float>(),cudaStreamPerThread);
    a.x16=dx16.as<uint16_t>();a.x_bf16=type==2;
    mx::moe_gate_up(a,nullptr,ht.as<float>(),cudaStreamPerThread);
    mx::moe_down(a,ht.as<float>(),nullptr,yt.as<float>(),cudaStreamPerThread);
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Check(hr.get<float>(experts*mid)==ht.get<float>(experts*mid),"typed MoE gate/up mismatch");
    Check(yr.get<float>(k)==yt.get<float>(k),"typed MoE down mismatch");
}
int main(){
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    Cuda(cudaSetDevice(0));int cases=0;
    for(int bits:{2,3,4,5,6,7,8,10,12,14,16})for(int rows:{1,9,16})for(int it:{0,1,2})for(int ot:{0,1,2}){
        Case(bits,rows,256,it,ot);++cases;
    }
    for(int bits:{3,6,10})for(int rows:{1,9,16})for(int it:{0,1,2})for(int ot:{0,1,2}){
        Case(bits,rows,5120,it,ot);++cases;
    }
    for(int type:{1,2})for(bool rounded:{false,true})MoeCase(type,rounded);
    Check(mx::launch_errors()==0,"kernel launch errors");
    std::cout<<"EXL3 MAYA IO PASS: "<<cases<<" exact staged-reference cases, padded strides and typed graphs; 4 MoE cases"<<std::endl;
    return 0;
}
