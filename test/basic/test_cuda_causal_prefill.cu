#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-conv.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void C(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
static void GPU(Data &d) { d.dataDevice=CUDA; d.dataDeviceIds={0}; d.Allocate(false); }
template<class T> __host__ __device__ T Cast(float x) { return T(x); }
template<class T> __device__ T OutputCast(float x) { return T(x); }
template<> __device__ half OutputCast<half>(float x) { return __float2half_rz(x); }
template<class T, class O>
__global__ void Reference(const T *input, const float *weight, const float *state, O *output,
        uint64_t total, int sequence, int channels, int kernel, bool silu) {
    for (uint64_t i = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x; i < total; i += uint64_t(blockDim.x)*gridDim.x) {
        const int channel = i % channels, token = i / channels % sequence, batch = i / channels / sequence;
        float value = cuda_conv::At<T,float,float,false>(input + uint64_t(batch)*sequence*channels + channel,
            state + (uint64_t(batch)*channels + channel)*kernel, weight + channel*kernel,
            token, channels, kernel, kernel);
        if (silu) value = value / (1.0 + expf(-value));
        output[i] = OutputCast<O>(value);
    }
}
template<class F> static float Time(F f) {
    for(int i=0;i<3;++i) f();
    C(cudaDeviceSynchronize());
    cudaEvent_t a,b; C(cudaEventCreate(&a)); C(cudaEventCreate(&b));
    C(cudaEventRecord(a,cudaStreamPerThread));
    for(int i=0;i<20;++i) f();
    C(cudaEventRecord(b,cudaStreamPerThread)); C(cudaEventSynchronize(b));
    float ms=0;C(cudaEventElapsedTime(&ms,a,b));C(cudaEventDestroy(a));C(cudaEventDestroy(b));return ms/20;
}
template<class T, class O> static void Run(DataType it, DataType ot, int batch, int sequence,
        int channels, int kernel, bool silu, bool initialize, bool timing=false) {
    Data input(it,{batch,sequence,channels}), weight(FLOAT32,{channels,1,kernel}),
        state(FLOAT32,{batch,channels,kernel}), history(FLOAT32,state.dims),
        expected(ot,input.dims), actual(ot,input.dims);
    for(auto *p:{&input,&weight,&state,&history,&expected,&actual})GPU(*p);
    std::vector<T> x(input.Count(0)); std::vector<float>w(weight.Count(0)),h(state.Count(0));
    for(size_t i=0;i<x.size();++i)x[i]=Cast<T>(std::sin(i*.17f)*3);
    for(size_t i=0;i<w.size();++i)w[i]=std::sin(i*.31f)*.4f;
    for(size_t i=0;i<h.size();++i)h[i]=initialize?0.f:std::cos(i*.27f);
    C(cudaMemcpy(input.cudaData,x.data(),input.GetBytes(),cudaMemcpyHostToDevice));
    C(cudaMemcpy(weight.cudaData,w.data(),weight.GetBytes(),cudaMemcpyHostToDevice));
    C(cudaMemcpy(history.cudaData,h.data(),history.GetBytes(),cudaMemcpyHostToDevice));
    C(cudaMemcpy(state.cudaData,h.data(),state.GetBytes(),cudaMemcpyHostToDevice));
    if(initialize) C(cudaMemset(state.cudaData,0x7f,state.GetBytes()));
    auto old=[&]{Reference<T,O><<<std::min<uint64_t>(65535,(x.size()+255)/256),256,0,cudaStreamPerThread>>>(
        (T*)input.cudaData,(float*)weight.cudaData,(float*)history.cudaData,(O*)expected.cudaData,
        x.size(),sequence,channels,kernel,silu);};
    auto now=[&]{Check(FastllmCudaCausalDepthwiseConv1DPrefill(input,weight,state,actual,kernel,silu,initialize),"conv dispatch");};
    old();now();
    std::vector<O>a(x.size()),b(x.size());std::vector<float>finalState(h.size());
    C(cudaMemcpy(a.data(),expected.cudaData,expected.GetBytes(),cudaMemcpyDeviceToHost));
    C(cudaMemcpy(b.data(),actual.cudaData,actual.GetBytes(),cudaMemcpyDeviceToHost));
    C(cudaMemcpy(finalState.data(),state.cudaData,state.GetBytes(),cudaMemcpyDeviceToHost));
    if (std::memcmp(a.data(),b.data(),actual.GetBytes())) {
        fprintf(stderr,"MISMATCH input=%d output=%d batch=%d sequence=%d channels=%d kernel=%d silu=%d init=%d\n",
            int(it),int(ot),batch,sequence,channels,kernel,int(silu),int(initialize));
        Check(false,"optimized convolution changed output bits");
    }
    for(int n=0;n<batch;++n)for(int c=0;c<channels;++c)for(int k=0;k<kernel;++k){
        const int token=sequence-kernel+k;
        const float v=token<0?h[(n*channels+c)*kernel+kernel+token]:float(x[(size_t(n)*sequence+token)*channels+c]);
        Check(finalState[(n*channels+c)*kernel+k]==v,"convolution changed history state");
    }
    if(timing)printf("CONV input=%d output=%d rows=%d channels=%d old_ms=%.5f optimized_ms=%.5f PASS\n",
        int(it),int(ot),sequence,channels,Time(old),Time(now));
}
template<class T,class O> static void Suite(DataType it,DataType ot){
    for(int sequence:{1,3,63,64,65,129})for(bool initialize:{false,true})for(bool silu:{false,true})
        Run<T,O>(it,ot,2,sequence,257,4,silu,initialize);
    Run<T,O>(it,ot,1,65,31,3,true,false);
    Run<T,O>(it,ot,1,4096,257,4,true,false);
}
__global__ void CheckSilu(unsigned *bad, int mode) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned bits = i * 747796405u + 2891336453u;
    bits = ((bits >> ((bits >> 28) + 4)) ^ bits) * 277803737u;
    bits = (bits >> 22) ^ bits;
    float x = mode == 0 ? __uint_as_float(bits) : (int(bits % 200000001) - 100000000) * 0.000001f;
    const float reference = float(double(x) / (1.0 + double(expf(-x))));
    const float actual = cuda_conv::SiluPromoted(x);
    if (__float_as_uint(reference) != __float_as_uint(actual) && !(isnan(reference) && isnan(actual)))
        atomicAdd(bad, 1u);
}
static void SiluAccuracy() {
    unsigned *bad = nullptr, count = 0;
    C(cudaMalloc(&bad, sizeof(unsigned))); C(cudaMemset(bad, 0, sizeof(unsigned)));
    for(int mode:{0,1}) CheckSilu<<<65536,256,0,cudaStreamPerThread>>>(bad,mode);
    C(cudaMemcpy(&count,bad,sizeof(unsigned),cudaMemcpyDeviceToHost)); C(cudaFree(bad));
    printf("SILU samples=%u mismatches=%u\n",2u*65536*256,count);
    Check(count==0,"promoted SiLU float rounding changed");
}
int main(){
    try {
        if(!FastllmCudaGetDeviceCount()){puts("FASTLLM_TEST_SKIP_NO_DEVICE");return 0;}
        FastllmCudaSetDevice(0);
        SiluAccuracy();
#define SUITE(T,I) Suite<T,float>(I,FLOAT32);Suite<T,half>(I,FLOAT16);Suite<T,__nv_bfloat16>(I,BFLOAT16)
        SUITE(float,FLOAT32);SUITE(half,FLOAT16);SUITE(__nv_bfloat16,BFLOAT16);
#undef SUITE
        Run<half,float>(FLOAT16,FLOAT32,1,8192,8192,4,true,false,true);
        Run<float,float>(FLOAT32,FLOAT32,1,8192,8192,4,true,false,true);
        puts("PASS causal prefill convolution: exact output and state, dtype and sequence boundaries");return 0;
    }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
