#include "models/lingbot_world/runtime.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include "mma.cuh"
#include "math.cuh"
#include <algorithm>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
using B = __nv_bfloat16;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(cublasStatus_t error) {
    if(error!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("LingBot cuBLAS error: "+std::to_string(int(error)));
}
void require(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); }
int grid(size_t n) { return int(std::min<size_t>((n + 255) / 256, 65535)); }
template<class T> T *ptr(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
template<class T> __device__ float value(T x) { return float(x); }
template<class T> __global__ void joinKernel(const T *x,const T *y,T *out,size_t n,int a,int b) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        size_t row=i/(a+b); int col=i%(a+b);
        out[i]=col<a?x[row*a+col]:y[row*b+col-a];
    }
}
__global__ void affineKernel(const float *x, const float *s, const float *b, float *y,
                             size_t n, int width, int rows, bool roundScale) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        size_t j=rows==1?i%width:i;
        float a=1.f+s[j];
        if (roundScale) a=value(B(a));
        y[i]=x[i]*a+b[j];
    }
}
__global__ void gateKernel(const float *x,const float *y,const float *g,float *out,
                           size_t n,int width,int rows) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        out[i]=x[i]+y[i]*g[rows==1?i%width:i];
}
template<class T> __global__ void patchKernel(const T *x,T *y,size_t n,int height,int width,
                                             int channels,bool unpack) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        if (!unpack) {
            int c=i%(channels*4)/4, dy=i%4/2, dx=i%2;
            size_t pixel=i/(channels*4);
            int px=pixel%(width/2),py=pixel/(width/2)%(height/2);
            size_t t=pixel/(size_t(height/2)*(width/2));
            y[i]=x[((t*height+py*2+dy)*width+px*2+dx)*channels+c];
        } else {
            int c=i%channels,px=i/channels%width,py=i/channels/width%height;
            size_t t=i/(size_t(channels)*height*width);
            size_t token=(t*(height/2)+py/2)*(width/2)+px/2;
            y[i]=x[token*4*channels+(py%2*2+px%2)*channels+c];
        }
    }
}
__global__ void ropeTable(double2 *table,int rows,int gh,int gw,int offset) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<size_t(rows)*64;i+=size_t(gridDim.x)*256) {
        int pair=i%64,axis=pair<22?0:(pair<43?1:2);
        int local=pair-(axis==0?0:(axis==1?22:43)),count=axis==0?22:21;
        size_t token=i/64;
        int pos=axis==0?int(token/(gh*gw))+offset:(axis==1?int(token/gw%gh):int(token%gw));
        double angle=double(pos)*pow(10000.0,-double(local)/count);
        table[i]=make_double2(cos(angle),sin(angle));
    }
}
template<class T> __global__ void ropeKernel(const T *x,B *y,const double2 *table,size_t pairs,int width) {
    for(size_t pair=size_t(blockIdx.x)*256+threadIdx.x;pair<pairs;pair+=size_t(gridDim.x)*256) {
        size_t i=pair*2;
        auto angle=table[(i/width)*64+pair%64];
        double a=double(value(x[i])),b=double(value(x[i+1]));
        y[i]=B(float(a*angle.x-b*angle.y));
        y[i+1]=B(float(b*angle.x+a*angle.y));
    }
}
template<bool Weights> __global__ void convPatchKernel(const B *x,B *out,size_t n,
                                                       int channels,int padded,int height,int width) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int c=i%padded,position=i/padded%4;
        size_t row=i/(padded*4);
        if(c>=channels) { out[i]=B(0.f); continue; }
        if constexpr (Weights) out[i]=x[row*channels*4+c*4+position];
        else {
            int px=row%(width/2),py=row/(width/2)%(height/2);
            size_t frame=row/(size_t(height/2)*(width/2));
            out[i]=x[((frame*height+py*2+position/2)*width+px*2+position%2)*channels+c];
        }
    }
}
__global__ void t5GeluKernel(const B *x,B *y,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        float v=value(x[i]);
        // Eager BF16 pow(x, 3) rounds x*x before multiplying by x again.
        float cube=value(B(value(B(v*v))*v));
        float inner=value(B(v+value(B(.044715f*cube))));
        float t=value(B(tanhf(value(B(.7978845608028654f*inner)))));
        y[i]=B(value(B(.5f*v))*value(B(1.f+t)));
    }
}
template<class W> __global__ void eagerRmsKernel(const B *x,const W *weight,W *out,int width) {
    // Four independent sums per lane preserve the eager FP32 reduction order.
    // BF16 rounding at the RMS/weight boundary matters across 24 T5 blocks.
    int row=blockIdx.x,lane=threadIdx.x;
    float sums[4]={0,0,0,0};
    for(int j=lane*4;j<width;j+=128)
        for(int k=0;k<4;++k) { float v=value(x[size_t(row)*width+j+k]); sums[k]+=v*v; }
    float total=((sums[0]+sums[1])+sums[2])+sums[3];
    for(int d=16;d;d/=2) total+=__shfl_down_sync(0xffffffff,total,d);
    // Eager mean multiplies by a rounded FP32 reciprocal. Division differs
    // for Wan's non-power-of-two width (1536), occasionally changing BF16 RMS.
    float mean=__shfl_sync(0xffffffff,total,0)*(1.f/float(width));
    float inv=rsqrtf(mean+1e-6f);
    for(int j=lane;j<width;j+=32)
        out[size_t(row)*width+j]=W(value(B(value(x[size_t(row)*width+j])*inv))*value(weight[j]));
}
struct Moments { float mean, variance, count; };
__device__ Moments combine(Moments left,Moments right) {
    float count=left.count+right.count;
    if(!count) return {0,0,0};
    float inverse=1.f/count,a=right.count*inverse,b=left.count*inverse;
    float delta=left.mean-right.mean;
    return {__fmaf_rn(a,right.mean,b*left.mean),
        __fmaf_rn(delta*delta*right.count,b,left.variance+right.variance),count};
}
template<class T> __global__ void wanLayerNormKernel(const T *x,const float *weight,const float *bias,T *out,int width) {
    __shared__ Moments warps[4];
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    size_t base=size_t(blockIdx.x)*width;
    Moments state{0,0,0};
    for(int first=threadIdx.x*4;first<width;first+=512) for(int j=0;j<4;++j) {
        float v=value(x[base+first+j]),delta=v-state.mean;
        state.count+=1;
        state.mean=__fmaf_rn(delta,1.f/state.count,state.mean);
        state.variance=__fmaf_rn(delta,v-state.mean,state.variance);
    }
    for(int offset=16;offset;offset/=2) {
        Moments other{__shfl_down_sync(0xffffffff,state.mean,offset),
            __shfl_down_sync(0xffffffff,state.variance,offset),__shfl_down_sync(0xffffffff,state.count,offset)};
        state=combine(state,other);
    }
    if(lane==0) warps[warp]=state;
    __syncthreads();
    if(threadIdx.x==0) warps[0]=combine(combine(warps[0],warps[2]),combine(warps[1],warps[3]));
    __syncthreads();
    float mean=warps[0].mean,inv=rsqrtf(warps[0].variance/width+1e-6f);
    for(int j=threadIdx.x;j<width;j+=128) {
        float v=inv*(value(x[base+j])-mean);
        if(weight && bias) v=__fmaf_rn(weight[j],v,bias[j]);
        else { if(weight) v*=weight[j]; if(bias) v+=bias[j]; }
        out[base+j]=T(v);
    }
}
__global__ void relativeSoftmax(B *scores,const B *bias,int n,int heads,int validKeys) {
    int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=heads*n) return;
    int query=row%n,head=row/n;
    float values[16];
    float maximum=-INFINITY;
    // Match eager FP32 softmax's lane-strided accumulation. A block-wide
    // tree can round BF16 probabilities differently for prompts >32 tokens;
    // those differences accumulate through the text encoder's 24 blocks.
    for(int j=0;j<16;++j) {
        int key=lane+j*32;
        float score=-INFINITY;
        if(key<n) {
            int relative=key-query,distance=abs(relative),bucket=relative>0?16:0;
            bucket+=distance<8?distance:min(15,8+int(logf(float(distance)/8.f)/logf(16.f)*8.f));
            score=key<validKeys?value(B(value(scores[size_t(row)*n+key])+value(bias[bucket*heads+head]))):-3.3895313892515355e38f;
        }
        values[j]=score; maximum=fmaxf(maximum,score);
    }
    for(int d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    float sum=0;
    for(int j=0;j<16;++j) { values[j]=expf(values[j]-maximum); sum+=values[j]; }
    for(int d=16;d;d/=2) sum+=__shfl_xor_sync(0xffffffff,sum,d);
    for(int j=0;j<16;++j) if(lane+j*32<n) scores[size_t(row)*n+lane+j*32]=B(values[j]/sum);
}
__global__ void headsToRows(const B *x,B *y,size_t n,int rows,int heads,int dim) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int c=i%(heads*dim),row=i/(heads*dim);
        y[i]=x[(size_t(c/dim)*rows+row)*dim+c%dim];
    }
}

// A fused version of the Wan FA2-compatible path. Keep reverse 64-key tiles
// and four FP32 denominator sums: changing these changes BF16 PV rounding in
// recurrent rollouts. MMA and exp/reciprocal use the existing FlashInfer code.
__device__ uint32_t packed(B a,B b) {
    return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b))<<16);
}
__global__ void wanAttentionKernel(const B *q,const B *k,const B *v,B *out,int rows,int keys) {
    constexpr int width=1536;
    __shared__ float scores[16][64],stats[16][5],rescale[16];
    __shared__ B prob[16][64];
    int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.y;
    int firstRow=blockIdx.x*16;
    float acc[2][8]={};
    bool first=true;
    for(int base=(keys-1)/64*64;base>=0;base-=64) {
        float dot[8]={};
        #pragma unroll
        for(int inner=0;inner<128;inner+=16) {
            uint32_t a[4],b[4];
            #pragma unroll
            for(int j=0;j<4;++j) {
                int row=firstRow+lane/4+(j%2)*8,dim=inner+(lane%4)*2+(j/2)*8;
                a[j]=row<rows?packed(q[size_t(row)*width+head*128+dim],q[size_t(row)*width+head*128+dim+1]):0;
                int key=base+warp*16+lane/4+(j/2)*8;
                dim=inner+(lane%4)*2+(j%2)*8;
                b[j]=key<keys?packed(k[size_t(key)*width+head*128+dim],k[size_t(key)*width+head*128+dim+1]):0;
            }
            flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B>(dot,a,b);
        }
        #pragma unroll
        for(int j=0;j<8;++j) {
            int row=lane/4+(j%4/2)*8,col=warp*16+(lane%4)*2+(j/4)*8+j%2;
            scores[row][col]=base+col<keys?dot[j]:-INFINITY;
        }
        __syncthreads();
        int r=threadIdx.x/8,l=threadIdx.x%8;
        float maximum=-INFINITY;
        #pragma unroll
        for(int i=0;i<8;++i) maximum=fmaxf(maximum,scores[r][l+i*8]);
        for(int d=4;d;d/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,8));
        float previous=first?-INFINITY:stats[r][0];
        maximum=fmaxf(previous,maximum);
        constexpr float scale=0.08838834764831844f*1.4426950408889634f;
        float factor=first?0.f:flashinfer::math::ptx_exp2((previous-maximum)*scale);
        if(l==0) { stats[r][0]=maximum; rescale[r]=factor; }
        #pragma unroll
        for(int i=0;i<8;++i) {
            int col=l+i*8;
            float p=base+col<keys?flashinfer::math::ptx_exp2(__fmaf_rn(scores[r][col],scale,-maximum*scale)):0.f;
            scores[r][col]=p; prob[r][col]=B(p);
        }
        __syncthreads();
        if(l<4) {
            float sum=first?scores[r][l*2]:__fmaf_rn(stats[r][1+l],factor,scores[r][l*2]);
            #pragma unroll
            for(int i=1;i<16;++i) sum+=scores[r][(i/2)*8+l*2+i%2];
            stats[r][1+l]=sum;
        }
        #pragma unroll
        for(int part=0;part<2;++part) {
            #pragma unroll
            for(int j=0;j<8;++j) acc[part][j]*=rescale[lane/4+(j%4/2)*8];
            #pragma unroll
            for(int inner=0;inner<64;inner+=16) {
                uint32_t a[4],b[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    int row=lane/4+(j%2)*8,key=inner+(lane%4)*2+(j/2)*8;
                    a[j]=packed(prob[row][key],prob[row][key+1]);
                    int col=part*64+warp*16+lane/4+(j/2)*8;
                    key=base+inner+(lane%4)*2+(j%2)*8;
                    b[j]=packed(key<keys?v[size_t(key)*width+head*128+col]:B(0.f),
                        key+1<keys?v[size_t(key+1)*width+head*128+col]:B(0.f));
                }
                flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B>(acc[part],a,b);
            }
        }
        __syncthreads();
        first=false;
    }
    #pragma unroll
    for(int part=0;part<2;++part) {
        #pragma unroll
        for(int j=0;j<8;++j) {
            int r=lane/4+(j%4/2)*8,row=firstRow+r,col=part*64+warp*16+(lane%4)*2+(j/4)*8+j%2;
            float inverse=flashinfer::math::ptx_rcp((stats[r][1]+stats[r][3])+(stats[r][2]+stats[r][4]));
            if(row<rows) out[size_t(row)*width+head*128+col]=B(acc[part][j]*inverse);
        }
    }
}

}
struct Runtime::Rotary {
    // Tensor owns aligned device storage; each row holds 64 FP64 (cos,sin)
    // pairs. Keep the official complex128 arithmetic and cache only one grid.
    Tensor table;
    int frames=0,height=0,width=0,offset=0;
};
struct Runtime::FullLinear {
    struct Plan {
        cublasLtMatmulDesc_t op=nullptr;
        cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
        cublasLtMatmulPreference_t preference=nullptr;
        cublasLtMatmulAlgo_t algorithm{};
        ~Plan() {
            if(op) cublasLtMatmulDescDestroy(op);
            if(a) cublasLtMatrixLayoutDestroy(a);
            if(b) cublasLtMatrixLayoutDestroy(b);
            if(c) cublasLtMatrixLayoutDestroy(c);
            if(preference) cublasLtMatmulPreferenceDestroy(preference);
        }
    };
    cublasLtHandle_t handle=nullptr;
    Tensor workspace{1,262144,true};
    std::unordered_map<std::string,std::unique_ptr<Plan>> plans;
    FullLinear() { check(cublasLtCreate(&handle)); }
    ~FullLinear() { if(handle) cublasLtDestroy(handle); }
};
Runtime::Runtime(int device):nomas::Runtime(device,false),fp32(device,true) {
    cublasHandle_t handle;
    check(cublasCreate(&handle)); blas_=handle;
    check(cublasSetStream(handle,cudaStreamPerThread));
    check(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
}
Runtime::~Runtime() { activate(); fullLinear_.reset(); if(blas_) cublasDestroy(static_cast<cublasHandle_t>(blas_)); }
Tensor Runtime::wanLinearFP32(const Tensor &x,const Tensor &w,const Tensor &bias) {
    require(x.fp32 && w.fp32 && bias.fp32 && x.cols==w.cols && bias.size()==size_t(w.rows),
            "invalid Wan FP32 linear shape/dtype");
    if(!fullLinear_) fullLinear_=std::make_unique<FullLinear>();
    auto &state=*fullLinear_;
    auto key=std::to_string(x.rows)+":"+std::to_string(x.cols)+":"+std::to_string(w.rows);
    auto &plan=state.plans[key];
    if(!plan) {
        auto p=std::make_unique<FullLinear::Plan>();
        check(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
        cublasOperation_t trans=CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
        cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;
        check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
        check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias.data,sizeof(void *)));
        check(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_32F,w.cols,w.rows,w.cols));
        check(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_32F,x.cols,x.rows,x.cols));
        check(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,w.rows,x.rows,w.rows));
        check(cublasLtMatmulPreferenceCreate(&p->preference));
        size_t bytes=state.workspace.bytes();
        check(cublasLtMatmulPreferenceSetAttribute(p->preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
        // Keep the reference's fused FP32 bias epilogue and workspace size.
        // SGEMM followed by a bias kernel chooses a different accumulation path.
        uint32_t alignment=16;
        for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                       CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
            check(cublasLtMatmulPreferenceSetAttribute(p->preference,attr,&alignment,sizeof(alignment)));
        cublasLtMatmulHeuristicResult_t choice{}; int count=0;
        check(cublasLtMatmulAlgoGetHeuristic(state.handle,p->op,p->a,p->b,p->c,p->c,p->preference,1,&choice,&count));
        require(count>0,"no cuBLASLt FP32 linear algorithm"); p->algorithm=choice.algo; plan=std::move(p);
    }
    check(cublasLtMatmulDescSetAttribute(plan->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias.data,sizeof(void *)));
    Tensor out(x.rows,w.rows,true); float one=1,zero=0;
    check(cublasLtMatmul(state.handle,plan->op,&one,w.data,plan->a,x.data,plan->b,
        &zero,out.data,plan->c,out.data,plan->c,&plan->algorithm,state.workspace.data,state.workspace.bytes(),cudaStreamPerThread));
    return out;
}
Tensor Runtime::patchConv2d(const Tensor &x,const Tensor &w,const Tensor &bias,int frames,int height,int width) {
    require(!x.fp32 && !w.fp32 && !bias.fp32 && frames>0 && height>0 && width>0 &&
            height%2==0 && width%2==0 && x.rows==frames*height*width && w.cols==x.cols*4 &&
            bias.size()==size_t(w.rows),"invalid Wan patch convolution");
    // Conv3d's 1x2x2 BF16 path traverses spatial positions with channels
    // padded to complete 16-wide MMA tiles. Flattening C,H,W into one dot
    // changes the accumulation around BF16 rounding boundaries.
    int padded=(x.cols+15)/16*16;
    Tensor input(x.rows/4,padded*4),weight(w.rows,padded*4);
    convPatchKernel<false><<<grid(input.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(input),input.size(),x.cols,padded,height,width);
    convPatchKernel<true><<<grid(weight.size()),256,0,cudaStreamPerThread>>>(ptr<B>(w),ptr<B>(weight),weight.size(),x.cols,padded,height,width);
    check(cudaGetLastError());
    return this->bias(linear(input,weight),bias);
}
Tensor Runtime::join(const Tensor &x,const Tensor &y) {
    require(x.rows==y.rows && x.fp32==y.fp32,"LingBot join shape/dtype mismatch");
    Tensor out(x.rows,x.cols+y.cols,x.fp32);
    if(x.fp32) joinKernel<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr<float>(x),ptr<float>(y),ptr<float>(out),out.size(),x.cols,y.cols);
    else joinKernel<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(y),ptr<B>(out),out.size(),x.cols,y.cols);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::affine(const Tensor &x,const Tensor &scale,const Tensor &shift,bool roundScale) {
    require(scale.cols==x.cols && shift.cols==x.cols && scale.rows==shift.rows &&
            (scale.rows==1 || scale.rows==x.rows),"LingBot affine shape mismatch");
    auto a=cast(x,true),s=cast(scale,true),b=cast(shift,true); Tensor out(x.rows,x.cols,true);
    affineKernel<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<float>(a),ptr<float>(s),ptr<float>(b),
        ptr<float>(out),x.size(),x.cols,s.rows,roundScale);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::gated(const Tensor &x,const Tensor &branch,const Tensor &gate) {
    require(x.rows==branch.rows && x.cols==branch.cols && gate.cols==x.cols &&
            (gate.rows==1 || gate.rows==x.rows),"LingBot gated residual shape mismatch");
    auto a=cast(x,true),b=cast(branch,true),g=cast(gate,true); Tensor out(x.rows,x.cols,true);
    gateKernel<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<float>(a),ptr<float>(b),ptr<float>(g),
        ptr<float>(out),x.size(),x.cols,g.rows);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::patch(const Tensor &x,int frames,int height,int width,bool unpack) {
    require(frames>0 && height>0 && width>0 && height%2==0 && width%2==0 &&
        x.rows==frames*height*width/(unpack?4:1) && (!unpack || x.cols%4==0),"LingBot patch shape mismatch");
    int channels=x.cols/(unpack?4:1);
    Tensor out(frames*height*width/(unpack?1:4),channels*(unpack?1:4),x.fp32);
    if(x.fp32) patchKernel<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<float>(x),ptr<float>(out),x.size(),height,width,channels,unpack);
    else patchKernel<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(out),x.size(),height,width,channels,unpack);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::rope3d(const Tensor &x,int frames,int height,int width,int offset) {
    require(frames>0 && height>0 && width>0 && offset>=0 && x.rows==frames*height*width && x.cols%128==0,"LingBot RoPE shape mismatch");
    if(!rotary_) rotary_=std::make_unique<Rotary>();
    auto &r=*rotary_;
    if(r.frames!=frames || r.height!=height || r.width!=width || r.offset!=offset) {
        r.table=Tensor(x.rows,256,true);
        ropeTable<<<grid(size_t(x.rows)*64),256,0,cudaStreamPerThread>>>(ptr<double2>(r.table),x.rows,height,width,offset);
        check(cudaGetLastError());
        r.frames=frames; r.height=height; r.width=width; r.offset=offset;
    }
    Tensor out(x.rows,x.cols,false);
    if(x.fp32) ropeKernel<<<grid(x.size()/2),256,0,cudaStreamPerThread>>>(ptr<float>(x),ptr<B>(out),ptr<double2>(r.table),x.size()/2,x.cols);
    else ropeKernel<<<grid(x.size()/2),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(out),ptr<double2>(r.table),x.size()/2,x.cols);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::t5Gelu(const Tensor &x) {
    require(!x.fp32,"T5 GELU requires BF16"); Tensor out(x.rows,x.cols,false);
    t5GeluKernel<<<grid(x.size()),256,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(out),x.size());
    check(cudaGetLastError()); return out;
}
Tensor Runtime::t5Norm(const Tensor &x,const Tensor &w) {
    require(!x.fp32 && !w.fp32 && x.cols==4096 && w.size()==4096,"invalid T5 norm shape");
    Tensor out(x.rows,x.cols,false);
    eagerRmsKernel<<<x.rows,32,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<B>(w),ptr<B>(out),x.cols);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::wanRms(const Tensor &x,const Tensor &w) {
    require(!x.fp32 && w.fp32 && x.cols==1536 && w.size()==1536,"invalid Wan RMS norm shape");
    Tensor out(x.rows,x.cols,true);
    eagerRmsKernel<<<x.rows,32,0,cudaStreamPerThread>>>(ptr<B>(x),ptr<float>(w),ptr<float>(out),x.cols);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::wanLayerNorm(const Tensor &x,const Tensor *w,const Tensor *b) {
    require(x.cols==1536 && (!w || (w->fp32 && w->size()==1536)) &&
            (!b || (b->fp32 && b->size()==1536)),"invalid Wan LayerNorm shape");
    Tensor out(x.rows,x.cols,x.fp32);
    if(x.fp32) wanLayerNormKernel<<<x.rows,128,0,cudaStreamPerThread>>>(ptr<float>(x),w?ptr<float>(*w):nullptr,b?ptr<float>(*b):nullptr,ptr<float>(out),x.cols);
    else wanLayerNormKernel<<<x.rows,128,0,cudaStreamPerThread>>>(ptr<B>(x),w?ptr<float>(*w):nullptr,b?ptr<float>(*b):nullptr,ptr<B>(out),x.cols);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::relativeAttention(const Tensor &q,const Tensor &k,const Tensor &v,const Tensor &bias,int validKeys) {
    require(!q.fp32 && !k.fp32 && !v.fp32 && !bias.fp32 && q.cols==4096 && k.cols==q.cols &&
        v.cols==q.cols && q.rows==k.rows && k.rows==v.rows && q.rows<=512 && bias.rows==32 && bias.cols==64 && validKeys>0 && validKeys<=q.rows,
        "invalid UMT5 attention shape");
    constexpr int heads=64,dim=64;
    int n=q.rows; Tensor scores(heads*n,n),values(heads*n,dim),out(n,q.cols);
    auto handle=static_cast<cublasHandle_t>(blas_); float one=1,zero=0;
    check(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,n,dim,&one,
        k.data,CUDA_R_16BF,k.cols,dim,q.data,CUDA_R_16BF,q.cols,dim,&zero,
        scores.data,CUDA_R_16BF,n,int64_t(n)*n,heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    relativeSoftmax<<<(heads*n+3)/4,128,0,cudaStreamPerThread>>>(ptr<B>(scores),ptr<B>(bias),n,heads,validKeys);
    check(cudaGetLastError());
    check(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,dim,n,n,&one,
        v.data,CUDA_R_16BF,v.cols,dim,scores.data,CUDA_R_16BF,n,int64_t(n)*n,&zero,
        values.data,CUDA_R_16BF,dim,int64_t(n)*dim,heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    headsToRows<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr<B>(values),ptr<B>(out),out.size(),n,heads,dim);
    check(cudaGetLastError()); return out;
}
Tensor Runtime::wanAttention(const Tensor &q,const Tensor &k,const Tensor &v) {
    require(!q.fp32 && !k.fp32 && !v.fp32 && q.cols==1536 && k.cols==q.cols && v.cols==q.cols &&
            k.rows==v.rows && q.rows>0 && k.rows>0,"invalid Wan attention shape");
    // The continuous backend already provides BF16 FlashInfer attention with
    // FP32 accumulation. Reuse its fused path instead of launching separate
    // QK / softmax / PV kernels for each 128-query by 64-key tile.
    Tensor out(q.rows,q.cols,false);
    wanAttentionKernel<<<dim3((q.rows+15)/16,12),128,0,cudaStreamPerThread>>>(ptr<B>(q),ptr<B>(k),ptr<B>(v),ptr<B>(out),q.rows,k.rows);
    check(cudaGetLastError()); return out;
}
}} // namespace fastllm::lingbot
