#include "models/lingbot_world/vae.h"
#include "cutlass/gemm/device/gemm.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
using Tensor=nomas::Tensor;
void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
float *ptr(const Tensor &x) {
    if(!x.fp32) throw std::invalid_argument("Wan VAE attention requires FP32");
    return reinterpret_cast<float *>(x.data);
}
// Reuse the repository's CUTLASS primitives. The official FP32 SDPA backend
// uses three TF32 products per FP32 product, including its accumulation order.
template<class LayoutB> void matmul(const Tensor &a,const Tensor &b,Tensor &out,
                                   int m,int n,int k,int ldb) {
    using Gemm=cutlass::gemm::device::Gemm<float,cutlass::layout::RowMajor,
        float,LayoutB,float,cutlass::layout::RowMajor,float,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32,128,32>,cutlass::gemm::GemmShape<32,32,32>,
        cutlass::gemm::GemmShape<16,8,8>,
        cutlass::epilogue::thread::LinearCombination<float,1,float,float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,4,4,false,
        cutlass::arch::OpMultiplyAddFastF32>;
    typename Gemm::Arguments args({m,n,k},{ptr(a),a.cols},{ptr(b),ldb},
                                  {ptr(out),out.cols},{ptr(out),out.cols},{1.f,0.f});
    auto status=Gemm{}(args,nullptr,cudaStreamPerThread);
    if(status!=cutlass::Status::kSuccess)
        throw std::runtime_error("Wan VAE attention GEMM: "+std::to_string(int(status)));
}
__global__ void softmax(float *scores,float *state,int rows,int count,float scale,bool first) {
    int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=rows) return;
    float values[8],maximum=-INFINITY;
    #pragma unroll
    for(int i=0;i<8;++i) {
        int key=(lane/4)*32+(lane%4)*2+(i/2)*8+i%2;
        // The bias-capable reference kernel applies scale before converting
        // to base-2 exponents, with a separate FP32 rounding at each multiply.
        values[i]=lane<16 && key<count?(scores[size_t(row)*128+key]*scale)*1.4426950408889634074f:-INFINITY;
        maximum=fmaxf(maximum,values[i]);
    }
    for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    float old=first?-INFINITY:state[row*3],sum=first?0.f:state[row*3+1];
    maximum=fmaxf(maximum,old);
    float rescale=maximum>old?exp2f(old-maximum):1.f;
    sum*=rescale;
    float partial=0;
    #pragma unroll
    for(int i=0;i<8;++i) {
        int key=(lane/4)*32+(lane%4)*2+(i/2)*8+i%2;
        float value=lane<16 && key<count?exp2f(values[i]-maximum):0.f;
        if(lane<16) scores[size_t(row)*128+key]=value;
        partial+=value;
    }
    for(int d=1;d<4;d*=2) partial+=__shfl_xor_sync(0xffffffff,partial,d);
    for(int group=0;group<4;++group) sum+=__shfl_sync(0xffffffff,partial,group*4);
    if(lane==0) { state[row*3]=maximum; state[row*3+1]=sum; state[row*3+2]=rescale; }
}
__global__ void merge(const float *value,const float *state,float *out,
                     size_t n,int channels,bool first,bool last) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        size_t row=i/channels;
        float alpha=last?1.f/state[row*3+1]:1.f;
        if(first) out[i]=alpha*value[i];
        else if(last) out[i]=fmaf(alpha,value[i],(alpha*state[row*3+2])*out[i]);
        else out[i]=fmaf(state[row*3+2],out[i],value[i]);
    }
}
}
nomas::Tensor VAERuntime::spatialAttention(const Tensor &q,const Tensor &k,const Tensor &v) {
    if(q.rows<=0 || k.rows<=0 || q.cols!=384 || k.cols!=q.cols || v.cols!=q.cols || k.rows!=v.rows)
        throw std::invalid_argument("Wan VAE attention shape");
    if(!q.fp32) {
        if(!low_ || k.fp32 || v.fp32) throw std::invalid_argument("Wan attention precision mismatch");
        return low_->attention(q,k,v,1);
    }
    Tensor out(q.rows,q.cols,true),state(q.rows,3,true);
    float scale=float(1./std::sqrt(double(q.cols)));
    for(int start=0;start<k.rows;start+=128) {
        int count=std::min(128,k.rows-start);
        Tensor scores(q.rows,128,true),values(q.rows,q.cols,true);
        matmul<cutlass::layout::ColumnMajor>(q,k.rowSlice(start,count),scores,q.rows,count,q.cols,q.cols);
        softmax<<<(q.rows+3)/4,128,0,cudaStreamPerThread>>>(ptr(scores),ptr(state),q.rows,count,scale,start==0);
        check(cudaGetLastError());
        Tensor block=v.rowSlice(start,count);
        if(count<128) {
            Tensor padded(128,v.cols,true);
            check(cudaMemsetAsync(padded.data,0,padded.bytes(),cudaStreamPerThread));
            check(cudaMemcpyAsync(padded.data,block.data,block.bytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
            block=padded;
        }
        matmul<cutlass::layout::RowMajor>(scores,block,values,q.rows,q.cols,128,q.cols);
        merge<<<std::min<size_t>((out.size()+255)/256,4096),256,0,cudaStreamPerThread>>>(
            ptr(values),ptr(state),ptr(out),out.size(),q.cols,start==0,start+count==k.rows);
        check(cudaGetLastError());
    }
    return out;
}
}} // namespace fastllm::lingbot
