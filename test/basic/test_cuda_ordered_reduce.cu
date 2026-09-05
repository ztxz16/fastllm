#include "fastllm-cuda-ordered-reduce.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<int Threads,class Op>
__device__ float ReferenceReduce(float value,float *scratch,Op op) {
    int tid=threadIdx.x;
    if(tid<Threads) scratch[tid]=value;
    __syncthreads();
    for(int stride=Threads/2;stride;stride>>=1) {
        if(tid<stride) scratch[tid]=op(scratch[tid],scratch[tid+stride]);
        __syncthreads();
    }
    // Use a distinct result cell for immediate consecutive invocations.
    if(tid==0) scratch[Threads]=scratch[0];
    __syncthreads();
    return scratch[Threads];
}
template<class T,int Threads,bool Optimized>
__global__ void Softmax(const T *input,float *output,int width) {
    __shared__ float scratch[Threads+1];
    int tid=threadIdx.x,row=blockIdx.x;
    float maximum=-INFINITY;
    if(tid<Threads) for(int i=tid;i<width;i+=Threads) maximum=max(maximum,float(input[row*width+i]));
    if constexpr(Optimized) maximum=fastllm::cuda::OrderedBlockReduce<Threads>(maximum,scratch,fastllm::cuda::OrderedMax{});
    else maximum=ReferenceReduce<Threads>(maximum,scratch,fastllm::cuda::OrderedMax{});
    float sum=0;
    if(tid<Threads) for(int i=tid;i<width;i+=Threads) {
        float value=expf(float(input[row*width+i])-maximum);
        output[row*width+i]=value; sum+=value;
    }
    if constexpr(Optimized) sum=fastllm::cuda::OrderedBlockReduce<Threads>(sum,scratch,fastllm::cuda::OrderedSum{});
    else sum=ReferenceReduce<Threads>(sum,scratch,fastllm::cuda::OrderedSum{});
    if(fabsf(sum)<1e-6f) sum=1e-4f;
    if(tid<Threads) for(int i=tid;i<width;i+=Threads) output[row*width+i]/=sum;
}
template<class T,int Threads> void Run(const char *dtype) {
    constexpr int rows=64;
    std::mt19937 rng(345);
    std::normal_distribution<float> normal(0,20);
    for(int width:{7,32,129,256,512,1025,4099}) {
        std::vector<T> input(rows*width);
        for(int r=0;r<rows;++r) for(int i=0;i<width;++i) {
            float value=normal(rng);
            if(r==0) value=0;
            if(r==1) value=-1000;
            if(r==2) value=float(i%3);
            if(r==3) value=i==0 ? INFINITY : -INFINITY;
            if(r==4) value=i==0 ? NAN : float(i%2);
            input[r*width+i]=T(value);
        }
        T *x; float *a,*b; size_t bytes=rows*width*sizeof(float);
        Check(cudaMalloc(&x,input.size()*sizeof(T)));Check(cudaMalloc(&a,bytes));Check(cudaMalloc(&b,bytes));
        Check(cudaMemcpy(x,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
        // Exercise logical reductions smaller than the physical block.
        Softmax<T,Threads,false><<<rows,256>>>(x,a,width);
        Softmax<T,Threads,true><<<rows,256>>>(x,b,width);
        std::vector<float> expected(rows*width),actual(rows*width);
        Check(cudaMemcpy(expected.data(),a,bytes,cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(actual.data(),b,bytes,cudaMemcpyDeviceToHost));
        if(std::memcmp(expected.data(),actual.data(),bytes)!=0) throw std::runtime_error("softmax bitwise mismatch");
        cudaFree(x);cudaFree(a);cudaFree(b);
    }
    std::printf("PASS %s logical_threads=%d widths=7..4099 bitwise\n",dtype,Threads);
}
template<int Items>
__global__ void ArgMax(const float *input, int *output, int size) {
    float keys[Items];
    int first = threadIdx.x * Items;
    #pragma unroll
    for (int i = 0; i < Items; ++i)
        keys[i] = first + i < size ? input[blockIdx.x * size + first + i] : -INFINITY;
    auto result = fastllm::cuda::WarpArgMax(keys, first, size);
    output[blockIdx.x * 32 + threadIdx.x] = result.index;
}
template<int Items> void RunArgMax() {
    constexpr int rows = 64;
    std::mt19937 rng(513);
    for (int size : {1, Items * 32 - 3, Items * 32}) {
        std::vector<float> input(rows * size);
        for (int r = 0; r < rows; ++r) for (int i = 0; i < size; ++i)
            input[r * size + i] = r < 4 ? float(r - 2) : float(int(rng() % 9) - 4);
        float *x; int *y;
        Check(cudaMalloc(&x, input.size() * sizeof(float)));
        Check(cudaMalloc(&y, rows * 32 * sizeof(int)));
        Check(cudaMemcpy(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        ArgMax<Items><<<rows, 32>>>(x, y, size);
        std::vector<int> actual(rows * 32);
        Check(cudaMemcpy(actual.data(), y, actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            int best = 0;
            for (int i = 1; i < size; ++i) if (input[r * size + i] > input[r * size + best]) best = i;
            for (int lane = 0; lane < 32; ++lane)
                if (actual[r * 32 + lane] != best) throw std::runtime_error("argmax tie/index mismatch");
        }
        cudaFree(x); cudaFree(y);
    }
    std::printf("PASS argmax items=%d partial rows/ties/all lanes\n", Items);
}
int main() {
    try {
        Run<float,32>("fp32");Run<float,64>("fp32");Run<float,128>("fp32");Run<float,256>("fp32");
        Run<half,32>("fp16");Run<half,64>("fp16");Run<half,128>("fp16");Run<half,256>("fp16");
        Run<__nv_bfloat16,32>("bf16");Run<__nv_bfloat16,64>("bf16");Run<__nv_bfloat16,128>("bf16");Run<__nv_bfloat16,256>("bf16");
        RunArgMax<1>();RunArgMax<2>();RunArgMax<4>();RunArgMax<8>();RunArgMax<16>();RunArgMax<32>();
        std::puts("ALL_PASS"); return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
