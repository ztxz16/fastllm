#include "fastllm-gguf-mmvq-kernels.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void Cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> static T *Upload(const std::vector<T> &v) {
    T *p; Cuda(cudaMalloc(&p, v.size()*sizeof(T)));
    Cuda(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template<class Output, int StoreMode> static void Check(int columns, int rows, int seed, bool capture) {
    std::mt19937 rng(seed);
    std::vector<block_q4_K> weights(size_t(rows)*columns/QK_K);
    const int inputStride = columns+256, outputStride = rows+7;
    std::vector<block_q8_1> input(inputStride/QK8_1);
    for (auto &w : weights) {
        w.dm = __floats2half2_rn((int(rng()%7)-3)*.001f, (int(rng()%5)-2)*.001f);
        for (auto &q : w.qs) q = rng();
        for (auto &q : w.scales) q = rng();
    }
    for (auto &x : input) {
        int sum=0;
        for (auto &q : x.qs) { q=int(rng()%255)-127; sum+=q; }
        x.ds=__floats2half2_rn(.01f,sum*.01f);
    }
    auto *w=Upload(weights); auto *x=Upload(input);
    Output *out; const size_t bytes=size_t(outputStride)*sizeof(Output);
    Cuda(cudaMalloc(&out,bytes)); Cuda(cudaMemset(out,0x5a,bytes));
    // Freeze the legacy four-physical-warp result, including output padding.
    mul_mat_vec_q<GGML_TYPE_Q4_K,1,4,Output,StoreMode><<<rows,dim3(32,4),0,cudaStreamPerThread>>>(
        w,x,out,nullptr,columns,rows,inputStride,outputStride,0,0,0,0);
    std::vector<uint8_t> reference(bytes), actual(bytes);
    Cuda(cudaMemcpy(reference.data(),out,bytes,cudaMemcpyDeviceToHost));
    auto run=[&] {
        mul_mat_vec_q_cuda<GGML_TYPE_Q4_K,Output,StoreMode>(w,x,out,nullptr,
            columns,rows,inputStride,1,outputStride,1,0,0,0,0,cudaStreamPerThread);
    };
    run(); Cuda(cudaDeviceSynchronize());
    Cuda(cudaMemset(out,0x5a,bytes));
    if (capture) {
        cudaGraph_t graph; cudaGraphExec_t executable;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        run(); Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        Cuda(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        Cuda(cudaGraphLaunch(executable,cudaStreamPerThread)); Cuda(cudaDeviceSynchronize());
        Cuda(cudaGraphExecDestroy(executable)); Cuda(cudaGraphDestroy(graph));
    } else run();
    Cuda(cudaMemcpy(actual.data(),out,bytes,cudaMemcpyDeviceToHost));
    if (reference != actual) throw std::runtime_error("Q4_K output bits or padding changed");
    Cuda(cudaFree(w)); Cuda(cudaFree(x)); Cuda(cudaFree(out));
}
int main() {
    int devices=0; if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        for (int device=0; device<std::min(2,devices); ++device) {
            FastllmCudaSetDevice(device);
            for (auto shape : std::vector<std::pair<int,int>>{{1280,32768},{2304,5120},
                    {1280,1025},{2304,1027},{1280,1023},{5120,4608},{8192,5120}}) {
                Check<float,0>(shape.first,shape.second,91,false);
                Check<half,0>(shape.first,shape.second,317,false);
                Check<half,1>(shape.first,shape.second,629,true);
                Check<half,2>(shape.first,shape.second,873,true);
            }
        }
        puts("PASS: Q4_K short GEMV legacy bits, output modes, tails and graphs"); return 0;
    } catch (const std::exception &e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
