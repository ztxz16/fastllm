#include "models/lingbot_world/vae.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
using Tensor = nomas::Tensor;
void check(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
float *ptr(const Tensor &x) {
    if(!x.fp32) throw std::invalid_argument("Wan VAE requires FP32 tensors");
    return reinterpret_cast<float *>(x.data);
}
int grid(size_t n) { return int(std::min<size_t>((n+255)/256,4096)); }
// Match the channel-first FP32 vector norm before division. The reference
// maintains four independent accumulation chains and reduces channel groups.
// Storage remains THWC so subsequent native spatial operators can reuse it.
template<class In,class Out> __global__ void channelNormKernel(const In *x,const float *weight,Out *out,
                                  int rows,int channels,int groups,bool silu) {
    int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows) return;
    float accum[4]={0,0,0,0};
    if(lane<groups) {
        for(int base=lane;base<channels;base+=groups*4) {
            #pragma unroll
            for(int i=0;i<4;++i) {
                int c=base+i*groups;
                if(c<channels) {
                    float v=float(x[size_t(row)*channels+c]);
                    accum[i]=fmaf(v,v,accum[i]);
                }
            }
        }
    }
    float sum=((accum[0]+accum[1])+accum[2])+accum[3];
    for(int offset=groups/2;offset;offset/=2)
        sum+=__shfl_down_sync(0xffffffff,sum,offset);
    float denom=fmaxf(sqrtf(__shfl_sync(0xffffffff,sum,0)),1e-12f);
    float scale=sqrtf(float(channels));
    for(int c=lane;c<channels;c+=32) {
        float v=(float(x[size_t(row)*channels+c])/denom)*scale*weight[c];
        if(silu) v=v/(1.f+expf(-v));
        out[size_t(row)*channels+c]=Out(v);
    }
}
int normalizationGroups(int rows,int cols) {
    auto power=[](int value) { int result=1; while(result<=value/2) result*=2; return result; };
    int width=std::min(power(rows/4),32),height=std::min(power(cols),128/width);
    return cols>=std::min(height*16,256)?height:1;
}
__global__ void im2colConvolution(const float *x,float *out,int frames,int h,int w,int c,
                        int kt,int kernel,int strideT,int padT,int strideS,int padS,
                        int oh,int ow,int start,size_t n) {
    int kc=c*kt*kernel*kernel;
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int p=start+i/kc,k=i%kc,ch=k/(kt*kernel*kernel);
        int dx=k%kernel,dy=k/kernel%kernel,dt=k/(kernel*kernel)%kt;
        int it=p/(oh*ow)*strideT-padT+dt,ih=p/ow%oh*strideS-padS+dy,iw=p%ow*strideS-padS+dx;
        out[i]=(it>=0 && it<frames && ih>=0 && ih<h && iw>=0 && iw<w)
            ?x[((size_t(it)*h+ih)*w+iw)*c+ch]:0.f;
    }
}
// FP32 convolution in the reference accumulates each filter in C,T,H,W order.
// General SGEMM may split that reduction, which can change later BF16 rounding.
// Tile rows/output channels while preserving the per-output FMA sequence.
__global__ void convolutionMatmul(const float *x,const float *weight,const float *bias,
                                  float *out,int rows,int columns,int inner) {
    // Pad the leading stride: warp lanes read different output channels
    // at the same reduction index. A stride of 32 aliases those banks.
    __shared__ float a[32][33],b[64][33];
    int tid=threadIdx.x,r=tid/16,c=tid%16;
    float accum[2][4]={};
    for(int base=0;base<inner;base+=32) {
        for(int i=tid;i<32*32;i+=256) {
            int row=blockIdx.y*32+i/32,k=base+i%32;
            a[i/32][i%32]=row<rows && k<inner?x[size_t(row)*inner+k]:0.f;
        }
        for(int i=tid;i<64*32;i+=256) {
            int col=blockIdx.x*64+i/32,k=base+i%32;
            b[i/32][i%32]=col<columns && k<inner?weight[size_t(col)*inner+k]:0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<32;++k) {
            #pragma unroll
            for(int i=0;i<2;++i) {
                float v=a[r+i*16][k];
                #pragma unroll
                for(int j=0;j<4;++j) accum[i][j]=fmaf(v,b[c+j*16][k],accum[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<2;++i) {
        int row=blockIdx.y*32+r+i*16;
        #pragma unroll
        for(int j=0;j<4;++j) {
            int col=blockIdx.x*64+c+j*16;
            if(row<rows && col<columns)
                out[size_t(row)*columns+col]=accum[i][j]+(bias?bias[col]:0.f);
        }
    }
}
Tensor convolution(const Tensor &x,const Tensor &w,const Tensor *bias,int frames,int h,int width,
                   int kt,int kernel,int strideT,int padT,int strideS,int padS,int oh,int ow) {
    int ot=(frames+padT-kt)/strideT+1;
    Tensor out(ot*oh*ow,w.rows,true);
    for(int start=0;start<out.rows;start+=1024) {
        int count=std::min(1024,out.rows-start);
        Tensor col(count,w.cols,true);
        im2colConvolution<<<grid(col.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(col),frames,h,width,x.cols,
            kt,kernel,strideT,padT,strideS,padS,oh,ow,start,col.size());
        check(cudaGetLastError());
        convolutionMatmul<<<dim3((w.rows+63)/64,(count+31)/32),256,0,cudaStreamPerThread>>>(
            ptr(col),ptr(w),bias?ptr(*bias):nullptr,ptr(out.rowSlice(start,count)),count,w.rows,w.cols);
        check(cudaGetLastError());
    }
    return out;
}
template<class T> __global__ void timeShuffle(const T *x,T *out,size_t n,int hw,int c) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        size_t p=i/c,t=p/hw;
        out[i]=x[((t/2)*hw+p%hw)*2*c+(t%2)*c+i%c];
    }
}
__constant__ float latentMean[16]={-.7571f,-.7089f,-.9113f,.1075f,-.1745f,.9653f,-.1517f,1.5508f,
    .4134f,-.0715f,.5517f,-.3632f,-.1922f,-.9497f,.2503f,-.2921f};
__constant__ float latentStd[16]={2.8184f,1.4541f,2.3275f,2.6558f,1.2196f,1.7708f,2.6052f,2.0743f,
    3.2687f,2.1526f,2.8652f,1.5579f,1.6382f,1.1253f,2.8251f,1.9160f};
__global__ void scaleLatent(const float *x,float *out,size_t n,bool decode) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int c=i%16; float inv=1.f/latentStd[c];
        out[i]=decode?x[i]/inv+latentMean[c]:(x[i]-latentMean[c])*inv;
    }
}
__global__ void clamp(const float *x,float *out,size_t n) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256)
        out[i]=isnan(x[i])?x[i]:fminf(1.f,fmaxf(-1.f,x[i]));
}
}
nomas::Tensor VAERuntime::channelNorm(const Tensor &x,const Tensor &weight) {
    if(!x.fp32) return channelNorm(cast(x,true),weight);
    if(x.rows<=0 || x.cols<=0 || weight.size()!=size_t(x.cols))
        throw std::invalid_argument("Wan VAE channel norm shape");
    if(x.rows<16 || x.rows%4) return specialNorm(x,weight,1);
    Tensor out(x.rows,x.cols,true);
    channelNormKernel<<<(x.rows+3)/4,128,0,cudaStreamPerThread>>>(ptr(x),ptr(weight),ptr(out),
        x.rows,x.cols,normalizationGroups(x.rows,x.cols),false);
    check(cudaGetLastError()); return out;
}
nomas::Tensor VAERuntime::channelNormForConv(const Tensor &x,const Tensor &weight,bool silu) {
    if(!low_ || x.rows<16 || x.rows%4) {
        auto out=channelNorm(x,weight);
        if(silu) out=activation(out);
        return cast(out,!low_);
    }
    if(x.cols<=0 || !weight.fp32 || weight.size()!=size_t(x.cols))
        throw std::invalid_argument("Wan VAE channel norm shape");
    // Keep the FP32 reduction and activation arithmetic; round only where
    // the separate convolution input cast previously rounded to BF16.
    Tensor out(x.rows,x.cols,false);
    int groups=normalizationGroups(x.rows,x.cols);
    auto result=reinterpret_cast<__nv_bfloat16 *>(out.data);
    if(x.fp32) channelNormKernel<<<(x.rows+3)/4,128,0,cudaStreamPerThread>>>(
        ptr(x),ptr(weight),result,x.rows,x.cols,groups,silu);
    else channelNormKernel<<<(x.rows+3)/4,128,0,cudaStreamPerThread>>>(
        reinterpret_cast<const __nv_bfloat16 *>(x.data),ptr(weight),result,x.rows,x.cols,groups,silu);
    check(cudaGetLastError()); return out;
}
nomas::Tensor VAERuntime::conv3d(const Tensor &x,const Tensor &w,const Tensor *bias,
    int frames,int h,int width,int kt,int kernel,int strideT,int padT) {
    if(frames<1 || h<1 || width<1 || x.rows!=frames*h*width ||
       (kt!=1 && kt!=3) || (kernel!=1 && kernel!=3) || strideT<1 || strideT>2 ||
       padT<0 || padT>2 || w.cols!=x.cols*kt*kernel*kernel)
        throw std::invalid_argument("Wan VAE convolution shape");
    int ot=(frames+padT-kt)/strideT+1;
    if(frames+padT<kt || ot<1) throw std::invalid_argument("empty Wan VAE convolution");
    if(low_) return convolutionBF16(x,w,bias,frames,h,width,kt,kernel,strideT,padT,1,kernel/2,h,width);
    return convolution(x,w,bias,frames,h,width,kt,kernel,strideT,padT,1,kernel/2,h,width);
}
nomas::Tensor VAERuntime::conv2d(const Tensor &x,const Tensor &w,const Tensor *bias,
                               int h,int width,int kernel,int stride,int pad,int padEnd) {
    if(h<1 || width<1 || x.rows!=h*width || (kernel!=1 && kernel!=3) ||
       stride<1 || stride>2 || pad<0 || padEnd<0 || w.cols!=x.cols*kernel*kernel)
        throw std::invalid_argument("Wan VAE spatial convolution shape");
    int oh=(h+2*pad+padEnd-kernel)/stride+1,ow=(width+2*pad+padEnd-kernel)/stride+1;
    if(h+2*pad+padEnd<kernel || width+2*pad+padEnd<kernel)
        throw std::invalid_argument("empty Wan VAE spatial convolution");
    if(low_) return convolutionBF16(x,w,bias,1,h,width,1,kernel,1,0,stride,pad,oh,ow);
    return convolution(x,w,bias,1,h,width,1,kernel,1,0,stride,pad,oh,ow);
}
nomas::Tensor VAERuntime::temporalShuffle(const Tensor &x,int frames,int h,int w) {
    if(x.rows!=frames*h*w || x.cols%2) throw std::invalid_argument("Wan temporal shuffle shape");
    Tensor out(x.rows*2,x.cols/2,x.fp32);
    if(!x.fp32) {
        timeShuffle<<<grid(out.size()),256,0,cudaStreamPerThread>>>(x.data,out.data,out.size(),h*w,out.cols);
        check(cudaGetLastError()); return out;
    }
    timeShuffle<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(out),out.size(),h*w,out.cols);
    check(cudaGetLastError()); return out;
}
nomas::Tensor VAERuntime::latentScale(const Tensor &x,bool decode) {
    if(!x.fp32) return latentScale(cast(x,true),decode);
    if(x.cols!=16) throw std::invalid_argument("Wan latent channels");
    Tensor out(x.rows,x.cols,true);
    scaleLatent<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(out),out.size(),decode);
    check(cudaGetLastError()); return out;
}
nomas::Tensor VAERuntime::clampPixels(const Tensor &x) {
    if(!x.fp32) return clampPixels(cast(x,true));
    Tensor out(x.rows,x.cols,true);
    clamp<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(out),out.size());
    check(cudaGetLastError()); return out;
}
}} // namespace fastllm::lingbot
