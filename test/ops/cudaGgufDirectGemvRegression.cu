// Fixtures contain packed weights and independent llama.cpp CPU-decoded FP32
// weights. Test both the decoded layouts and GEMV accumulation, not a second
// invocation of the GPU implementation under test.
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "fastllm-gguf-gemv.cuh"
#ifndef FASTLLM_GGUF_GEMV_STANDALONE
#include "fastllm-gguf-dequant.cuh"
#else
// A CPU-only dependency avoids the full runtime's CUDA static initialization
// when running Compute Sanitizer on the new kernels in isolation.
extern "C" int64_t reference_block_size(ggml_type) asm("ggml_blck_size");
extern "C" size_t reference_row_size(ggml_type, int64_t) asm("ggml_row_size");
int64_t ggml_blck_size(ggml_type type) { return reference_block_size(type); }
size_t ggml_row_size(ggml_type type, int64_t columns) { return reference_row_size(type, columns); }
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CUDA_OK(expr) do { auto e=(expr); if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)
template<typename T> T cast(float x) { return static_cast<T>(x); }
template<> half cast(float x) { return __float2half_rn(x); }
template<> __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }

template<typename T>
void Check(ggml_type type, int rows, int columns, const std::vector<char> &packed,
           const std::vector<float> &reference, int &checks, double &worst) {
    void *w; T *x,*y;
    CUDA_OK(cudaMalloc(&w,packed.size()));
    CUDA_OK(cudaMalloc(&x,columns*sizeof(T)));
    CUDA_OK(cudaMalloc(&y,(rows+2)*sizeof(T)));
    CUDA_OK(cudaMemcpy(w,packed.data(),packed.size(),cudaMemcpyHostToDevice));
#ifndef FASTLLM_GGUF_GEMV_STANDALONE
    // Refactoring the decoder must preserve standalone dequantization used
    // by prefill. Compare it directly to the independent CPU fixture too.
    T *expanded;
    CUDA_OK(cudaMalloc(&expanded,(size_t)rows*columns*sizeof(T)));
    if constexpr (std::is_same<T,half>::value) {
        ggml_get_to_fp16_cuda(type)(w,expanded,rows,columns,cudaStreamPerThread);
    } else if constexpr (std::is_same<T,float>::value) {
        ggml_get_to_fp32_cuda(type)(w,expanded,rows,columns,cudaStreamPerThread);
    } else {
        ggml_get_to_bf16_cuda(type)(w,expanded,rows,columns,cudaStreamPerThread);
    }
    CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<T> decoded((size_t)rows*columns);
    CUDA_OK(cudaMemcpy(decoded.data(),expanded,decoded.size()*sizeof(T),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<decoded.size();i++) {
        float expected=float(cast<T>(reference[i])),actual=float(decoded[i]);
        float eps=std::is_same<T,half>::value ? .001f : std::is_same<T,__nv_bfloat16>::value ? .008f : .00001f;
        if(!std::isfinite(actual)||std::abs(expected-actual)>eps*std::abs(expected)+1e-7f) {
            std::cerr<<"DEQUANT_FAIL type="<<int(type)<<" columns="<<columns<<" index="<<i<<" expected="<<expected<<" actual="<<actual<<"\n";
            throw std::runtime_error("standalone dequant reference mismatch");
        }
    }
    CUDA_OK(cudaFree(expanded));
#endif
    std::mt19937 rng(1921+columns);std::normal_distribution<float> normal(0,.2f);
    std::vector<T> input(columns),output(rows+2);
    for(int pattern=0;pattern<3;pattern++) {
        for(int j=0;j<columns;j++) input[j]=cast<T>(pattern==0 ? normal(rng) : pattern==1 ? ((j%3)-1)*.125f : 0.0f);
        CUDA_OK(cudaMemcpy(x,input.data(),columns*sizeof(T),cudaMemcpyHostToDevice));
        for(int count : {1, std::min(7,rows), rows}) {
            std::fill(output.begin(),output.end(),cast<T>(-17));
            CUDA_OK(cudaMemcpy(y,output.data(),output.size()*sizeof(T),cudaMemcpyHostToDevice));
            if(!FastllmGgufDirectGemv(x,w,y+1,type,columns,count,cudaStreamPerThread)) throw std::runtime_error("unsupported fixture");
            CUDA_OK(cudaGetLastError());CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
            CUDA_OK(cudaMemcpy(output.data(),y,output.size()*sizeof(T),cudaMemcpyDeviceToHost));
            if(float(output.front())!=-17 || float(output[count+1])!=-17)throw std::runtime_error("output guard overwritten");
            double error2=0,ref2=0,maxabs=0;
            for(int i=0;i<count;i++) {
                double expected=0,absoluteSum=0;
                for(int j=0;j<columns;j++) {
                    double term=double(float(cast<T>(reference[(size_t)i*columns+j])))*float(input[j]);
                    expected+=term;absoluteSum+=std::abs(term);
                }
                const double target=float(cast<T>(expected)),actual=float(output[i+1]);
                const double error=std::abs(target-actual);
                const double rounding=std::is_same<T,half>::value ? .001 : std::is_same<T,__nv_bfloat16>::value ? .008 : .000005;
                if(!std::isfinite(actual)||error>rounding*std::abs(target)+3e-6*absoluteSum+1e-6) {
                    std::cerr<<"FAIL type="<<int(type)<<" cols="<<columns<<" rows="<<count<<" pattern="<<pattern<<" row="<<i<<" expected="<<target<<" actual="<<actual<<" abs_sum="<<absoluteSum<<"\n";
                    throw std::runtime_error("independent reference mismatch");
                }
                error2+=error*error;ref2+=target*target;maxabs=std::max(maxabs,error);
            }
            double relative=std::sqrt(error2/std::max(ref2,1e-30));worst=std::max(worst,relative);
            const double limit=std::is_same<T,half>::value ? .001 : std::is_same<T,__nv_bfloat16>::value ? .008 : .00005;
            if(relative>limit)throw std::runtime_error("relative L2 exceeds threshold");
            // The path must also be capture-safe: no scratch allocation,
            // initialization, or host synchronization inside the dispatcher.
            if(pattern==0 && count==rows) {
                cudaGraph_t graph;cudaGraphExec_t exec;
                CUDA_OK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                FastllmGgufDirectGemv(x,w,y+1,type,columns,count,cudaStreamPerThread);
                CUDA_OK(cudaStreamEndCapture(cudaStreamPerThread,&graph));
                CUDA_OK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                CUDA_OK(cudaGraphLaunch(exec,cudaStreamPerThread));CUDA_OK(cudaStreamSynchronize(cudaStreamPerThread));
                std::vector<T> replay(output.size());CUDA_OK(cudaMemcpy(replay.data(),y,replay.size()*sizeof(T),cudaMemcpyDeviceToHost));
                if(memcmp(replay.data(),output.data(),output.size()*sizeof(T)))throw std::runtime_error("graph replay mismatch");
                CUDA_OK(cudaGraphExecDestroy(exec));CUDA_OK(cudaGraphDestroy(graph));
            }
            checks++;
        }
    }
    CUDA_OK(cudaFree(w));CUDA_OK(cudaFree(x));CUDA_OK(cudaFree(y));
}
int main(int argc,char**argv)try {
    if(argc!=2)throw std::runtime_error("usage: cudaGgufDirectGemvRegression fixtures.bin");
    std::ifstream f(argv[1],std::ios::binary);f.exceptions(std::ios::failbit|std::ios::badbit);
    uint32_t cases;f.read((char*)&cases,4);int checks=0;double halfWorst=0,floatWorst=0,bfWorst=0;
    for(uint32_t i=0;i<cases;i++) {
        uint32_t h[4];f.read((char*)h,sizeof(h));std::vector<char>w(h[3]);std::vector<float>ref((size_t)h[1]*h[2]);
        f.read(w.data(),w.size());f.read((char*)ref.data(),ref.size()*4);
        Check<half>((ggml_type)h[0],h[1],h[2],w,ref,checks,halfWorst);
        Check<float>((ggml_type)h[0],h[1],h[2],w,ref,checks,floatWorst);
        Check<__nv_bfloat16>((ggml_type)h[0],h[1],h[2],w,ref,checks,bfWorst);
        std::cout<<"PASS case="<<i<<" type="<<h[0]<<" shape="<<h[1]<<"x"<<h[2]<<"\n";
    }
    std::cout<<"ALL_PASS cases="<<cases<<" checks="<<checks<<" max_relative_L2_fp16="<<halfWorst<<" fp32="<<floatWorst<<" bf16="<<bfWorst<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
