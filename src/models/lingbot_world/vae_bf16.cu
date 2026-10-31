#include "models/lingbot_world/vae.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "cutlass/conv/kernel/default_conv3d_fprop.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
using Tensor = nomas::Tensor;
using B = __nv_bfloat16;
void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
int grid(size_t n) { return int(std::min<size_t>((n+255)/256,4096)); }
template<class T> T *ptr(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
template<class Out,class In> __global__ void convert(const In *x,Out *out,size_t n) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256)
        out[i]=Out(float(x[i]));
}
template<class T> __global__ void im2col(const T *x,B *out,int frames,int h,int w,int channels,
    int kt,int kernel,int strideT,int padT,int strideS,int padS,int oh,int ow,int start,size_t n) {
    int inner=channels*kt*kernel*kernel;
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int p=start+i/inner,k=i%inner,ch=k%channels,spatial=k/channels;
        int dx=spatial%kernel,dy=spatial/kernel%kernel,dt=spatial/(kernel*kernel)%kt;
        int t=p/(oh*ow)*strideT-padT+dt,y=p/ow%oh*strideS-padS+dy,z=p%ow*strideS-padS+dx;
        out[i]=B(t>=0 && t<frames && y>=0 && y<h && z>=0 && z<w
            ?float(x[((size_t(t)*h+y)*w+z)*channels+ch]):0.f);
    }
}
// Reuse the vendored CUTLASS Tensor Core convolution without materializing
// the large im2col matrix. All accumulation remains FP32; the bias is added
// separately after BF16 rounding, matching CUDA autocast's boundary.
bool implicitConvolution(const Tensor &x,const Tensor &weight,const Tensor &out,
    int frames,int h,int w,int kt,int kernel,int st,int pt,int ss,int ps,int ot,int oh,int ow) {
    using Element=cutlass::bfloat16_t;
    using Layout=cutlass::layout::TensorNDHWC;
    using Kernel=typename cutlass::conv::kernel::DefaultConv3dFprop<
        Element,Layout,Element,Layout,Element,Layout,float,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
        cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<Element,8,float,float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,
        cutlass::arch::OpMultiplyAdd,cutlass::conv::IteratorAlgorithm::kOptimized,
        cutlass::conv::StrideSupport::kStrided>::Kernel;
    using Conv=cutlass::conv::device::ImplicitGemmConvolution<Kernel>;
    cutlass::Tensor5DCoord input(1,frames,h,w,x.cols),filter(weight.rows,kt,kernel,kernel,x.cols),output(1,ot,oh,ow,weight.rows);
    cutlass::conv::Conv3dProblemSize problem(input,filter,cutlass::make_Coord(pt,ps,ps),
        cutlass::make_Coord(st,ss,ss),cutlass::make_Coord(1,1,1),output);
    typename Conv::Arguments args(problem,
        {ptr<Element>(x),Layout::packed(input)},
        {ptr<Element>(weight),Layout::packed(filter)},
        {ptr<Element>(out),Layout::packed(output)},
        {ptr<Element>(out),Layout::packed(output)},{1.f,0.f});
    if(Conv::can_implement(args)!=cutlass::Status::kSuccess) return false;
    Conv conv;
    auto status=conv.initialize(args,nullptr,cudaStreamPerThread);
    if(status==cutlass::Status::kSuccess) status=conv(cudaStreamPerThread);
    if(status!=cutlass::Status::kSuccess) throw std::runtime_error("Wan BF16 implicit convolution failed");
    return true;
}
__global__ void packWeight(const uint16_t *x,uint16_t *out,size_t n,int channels,int spatial) {
    int inner=channels*spatial;
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int k=i%inner;
        out[i]=x[(i/inner)*inner+(k%channels)*spatial+k/channels];
    }
}
__global__ void resize(const uint16_t *x,uint16_t *out,size_t n,int w,int channels) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        size_t p=i/channels;
        out[i]=x[((p/(2*w)/2)*w+(p%(2*w))/2)*channels+i%channels];
    }
}
}
Tensor VAERuntime::cast(const Tensor &x,bool full) {
    if(x.fp32==full) return x;
    Tensor out(x.rows,x.cols,full);
    if(full) convert<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<float>(out),x.size());
    else convert<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<float>(x),ptr<B>(out),x.size());
    check(cudaGetLastError()); return out;
}
Tensor VAERuntime::resize2x(const Tensor &x,int h,int w) {
    if(x.fp32) return qwen_image::Runtime::resize2x(x,h,w);
    if(x.rows!=h*w) throw std::invalid_argument("Wan resize shape");
    Tensor out(x.rows*4,x.cols,false);
    resize<<<grid(out.size()),256,0,cudaStreamPerThread>>>(x.data,out.data,out.size(),w,x.cols);
    check(cudaGetLastError()); return out;
}
Tensor VAERuntime::convolutionBF16(const Tensor &x,const Tensor &weight,const Tensor *bias,
    int frames,int h,int w,int kt,int kernel,int strideT,int padT,int strideS,int padS,int oh,int ow) {
    if(!low_ || weight.fp32 || (bias && bias->fp32))
        throw std::invalid_argument("Wan BF16 convolution precision mismatch");
    auto packed=weight;
    if(kt*kernel*kernel>1) {
        auto &entry=packedWeights_[weight.data];
        if(!entry.first.rows) {
            // Keep the source allocation alive with the packed view: pointer
            // reuse can never retrieve another checkpoint's packed weights.
            entry.first=weight; entry.second=Tensor(weight.rows,weight.cols,false);
            packWeight<<<grid(weight.size()),256,0,cudaStreamPerThread>>>(weight.data,entry.second.data,
                weight.size(),x.cols,kt*kernel*kernel);
            check(cudaGetLastError());
        }
        packed=entry.second;
    }
    int ot=(frames+padT-kt)/strideT+1;
    Tensor out(ot*oh*ow,weight.rows,false);
    if(kt*kernel*kernel>1 && x.cols%8==0 && weight.rows%8==0) {
        auto input=cast(x,false);
        if(implicitConvolution(input,packed,out,frames,h,w,kt,kernel,strideT,padT,strideS,padS,ot,oh,ow))
            return bias?low_->bias(out,*bias):out;
    }
    // Bound scratch memory; reuse the same BF16 Tensor Core GEMM and FP32
    // accumulation as other continuous models. cuDNN autocast rounds the
    // convolution before adding its BF16 bias, so keep those two boundaries.
    for(int start=0;start<out.rows;start+=4096) {
        int count=std::min(4096,out.rows-start);
        Tensor columns;
        if(kt==1 && kernel==1 && strideT==1 && strideS==1 && !padT && !padS)
            columns=cast(x.rowSlice(start,count),false);
        else {
            columns=Tensor(count,weight.cols,false);
            if(x.fp32) im2col<<<grid(columns.size()),256,0,cudaStreamPerThread>>>(ptr<float>(x),ptr<B>(columns),
                frames,h,w,x.cols,kt,kernel,strideT,padT,strideS,padS,oh,ow,start,columns.size());
            else im2col<<<grid(columns.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(columns),
                frames,h,w,x.cols,kt,kernel,strideT,padT,strideS,padS,oh,ow,start,columns.size());
            check(cudaGetLastError());
        }
        auto value=low_->linear(columns,packed);
        if(bias) value=low_->bias(value,*bias);
        check(cudaMemcpyAsync(out.rowSlice(start,count).data,value.data,value.bytes(),
                              cudaMemcpyDeviceToDevice,cudaStreamPerThread));
    }
    return out;
}
}} // namespace fastllm::lingbot
