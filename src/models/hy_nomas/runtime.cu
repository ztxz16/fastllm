#include "models/hy_nomas/runtime.h"
#include "json11.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fastllm { namespace nomas {
cudaError_t fusedAttention(const void *,const void *,const void *,void *,int,int,int,int,int,bool,bool,int,int,int,float,cudaStream_t);
namespace {
using B = __nv_bfloat16;
using H = __half;
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("HY-Nomas cuBLAS error " + std::to_string(int(s))); }
void require(bool ok, const char *message) { if (!ok) throw std::invalid_argument(message); }
void copy(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
    check(cudaMemcpyAsync(dst,src,bytes,kind,cudaStreamPerThread));
    // Device copies share the inference stream and need no host barrier.
    // Host buffers may be temporary, so keep their lifetime synchronous.
    if(kind!=cudaMemcpyDeviceToDevice) check(cudaStreamSynchronize(cudaStreamPerThread));
}
// Never change the process-wide default pool: other fastllm models may use it.
// Tensors retain this pool so their storage may safely outlive Runtime.
struct MemoryPool {
    cudaMemPool_t handle = nullptr;
    int device;
    explicit MemoryPool(int device, uint64_t retained=2ULL*1024*1024*1024):device(device) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        check(cudaMemPoolCreate(&handle, &props));
        auto status = cudaMemPoolSetAttribute(handle, cudaMemPoolAttrReleaseThreshold, &retained);
        if (status != cudaSuccess) {
            cudaMemPoolDestroy(handle);
            check(status);
        }
    }
    ~MemoryPool() { cudaMemPoolDestroy(handle); }
};
thread_local std::shared_ptr<MemoryPool> activePool;
std::shared_ptr<MemoryPool> memoryPool(int device) {
    if(activePool && activePool->device==device) return activePool;
    static std::mutex mutex;
    static std::unordered_map<int, std::weak_ptr<MemoryPool>> pools;
    std::lock_guard<std::mutex> lock(mutex);
    auto pool = pools[device].lock();
    if (!pool) {
        pool = std::make_shared<MemoryPool>(device);
        pools[device] = pool;
    }
    return pool;
}
std::shared_ptr<void> allocate(size_t bytes) {
    require(bytes > 0, "empty CUDA allocation");
    int device; check(cudaGetDevice(&device));
    auto pool = memoryPool(device);
    void *p = nullptr;
    check(cudaMallocFromPoolAsync(&p,bytes,pool->handle,cudaStreamPerThread));
    return {p, [device,pool](void *x) {
        int previous = device; cudaGetDevice(&previous); cudaSetDevice(device);
        cudaFreeAsync(x,cudaStreamPerThread);
        cudaSetDevice(previous);
    }};
}
template<class T> T *as(const Tensor &x) { return reinterpret_cast<T *>(x.data); }
B *ptr(const Tensor &x) { return reinterpret_cast<B *>(x.data); }
__device__ float f(float x) { return x; }
__device__ float f(B x) { return __bfloat162float(x); }
__device__ float f(H x) { return __half2float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
template<class T> __device__ T cast(float x) { return T(x); }
template<> __device__ B cast<B>(float x) { return b(x); }
template<class T> __device__ float roundTo(float x) { return f(cast<T>(x)); }
template<class T> __global__ void pixelRGBKernel(const T *x, uint8_t *rgb, int *invalid,
                                                size_t n, int width, int height, int firstFrame) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        size_t pixel=i/3;
        int c=i%3, px=pixel%width, py=pixel/width%height;
        int frame=int(pixel/(size_t(width)*height))+firstFrame;
        size_t token=(size_t(frame/4)*(height/16)+py/16)*(width/16)+px/16;
        size_t channel=((frame%4*16+py%16)*16+px%16)*3+c;
        float value=f(x[token*3072+channel]);
        if(!isfinite(value)) { atomicExch(invalid,1); rgb[i]=0; continue; }
        // Preserve the CPU path's separate FP32 operations and ties-to-even.
        value=__fadd_rn(__fmul_rn(value,.5f),.5f);
        value=__fmul_rn(fminf(fmaxf(value,0.f),1.f),255.f);
        rgb[i]=uint8_t(__float2uint_rn(value));
    }
}
template<class T> __global__ void element(const T *x, const T *y, T *z, size_t n, int op) {
    for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n; i+=size_t(gridDim.x)*blockDim.x) {
        float v=f(x[i]);
        // Preserve the BF16 SiLU result before multiplying by the up branch.
        z[i]=cast<T>(op==4 ? roundTo<T>(v/(1.f+expf(-v)))*f(y[i]) : op==0 ? v+f(y[i]) : op==3 ? v*f(y[i]) : op==1 ? v/(1.f+expf(-v)) : .5f*v*(1.f+tanhf(.7978845608028654f*(v+.044715f*v*v*v))));
    }
}
template<class T> __global__ void addBias(const T *x,const T *bias,T *y,size_t n,int cols) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) y[i]=cast<T>(f(x[i])+f(bias[i%cols]));
}
__device__ float sumBlock(float v, float *scratch) {
    scratch[threadIdx.x]=v; __syncthreads();
    for (int d=blockDim.x/2; d; d/=2) { if(threadIdx.x<d) scratch[threadIdx.x]+=scratch[threadIdx.x+d]; __syncthreads(); }
    return scratch[0];
}
template<class T> __global__ void normKernel(const T *x, T *y, const T *weight, const T *bias, int width, bool rms, float eps) {
    __shared__ float scratch[256]; __shared__ float mean, inv;
    int row=blockIdx.x; float s=0;
    for(int c=threadIdx.x;c<width;c+=blockDim.x) s+=f(x[size_t(row)*width+c]);
    float total=sumBlock(s,scratch);
    if(threadIdx.x==0) mean=rms?0.f:total/width;
    __syncthreads(); s=0;
    for(int c=threadIdx.x;c<width;c+=blockDim.x) { float v=f(x[size_t(row)*width+c])-mean; s+=v*v; }
    total=sumBlock(s,scratch);
    if(threadIdx.x==0) inv=rsqrtf(total/width+eps);
    __syncthreads();
    for(int c=threadIdx.x;c<width;c+=blockDim.x) {
        float v=(f(x[size_t(row)*width+c])-mean)*inv;
        // Official RMSNorm casts before applying weight; LayerNorm has a fused affine.
        if(rms) v=roundTo<T>(v);
        if constexpr (std::is_same_v<T,B>) {
            if(!rms && weight && bias) v=__fmaf_rn(v,f(weight[c]),f(bias[c]));
            else { if(weight) v*=f(weight[c]); if(bias) v+=f(bias[c]); }
        } else { if(weight) v*=f(weight[c]); if(bias) v+=f(bias[c]); }
        y[size_t(row)*width+c]=cast<T>(v);
    }
}
template<class T> __global__ void modulation(const T *x, const T *y, const T *mod, T *z, size_t n, int c, int a, int d, bool residual) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int j=i%c;
        // Match eager BF16 boundaries, including the rounding of (1 + scale).
        float v=residual ? f(x[i])+roundTo<T>(f(y[i])*f(mod[a*c+j]))
                         : roundTo<T>(f(x[i])*roundTo<T>(1.f+f(mod[d*c+j])))+f(mod[a*c+j]);
        z[i]=cast<T>(v);
    }
}
template<class T> __global__ void repeatRowsKernel(const T *x,T *y,size_t n,int cols,int copies) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        y[i]=x[(i/cols/copies)*cols+i%cols];
}
template<class T> __global__ void columnKernel(const T *x, T *y, int rows, int cols, int first, int count) {
    size_t n=size_t(rows)*count;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) y[i]=x[(i/count)*cols+first+i%count];
}
template<class T> __global__ void ropeKernel(const T *x, T *y, const float *cs, const float *sn, size_t n, int width, int dim) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        size_t p=(i/width)*dim+i%dim;
        float r=(i%2 ? f(x[i-1]) : -f(x[i+1]));
        y[i]=cast<T>(f(x[i])*cs[p]+r*sn[p]);
    }
}
__global__ void softmax(float *s, const uint8_t *mask, int n, int queryStart,
                        bool causal, int tile, int height, int width, float *inverseSum, int batchRows=0) {
    __shared__ float scratch[256]; __shared__ float mx, inv;
    int row=blockIdx.x, query= batchRows ? row%batchRows : row; float m=-INFINITY;
    for(int j=threadIdx.x;j<n;j+=blockDim.x) {
        bool visible=!mask || mask[size_t(query+queryStart)*n+j];
        if(causal && j>query+queryStart) visible=false;
        if(tile>0) {
            int qh=((query+queryStart)/width%height)/tile,qw=((query+queryStart)%width)/tile;
            int centerH=max(1,min(qh,max((height+tile-1)/tile-2,1)));
            int centerW=max(1,min(qw,max((width+tile-1)/tile-2,1)));
            if(abs(j/width%height/tile-centerH)>1 || abs(j%width/tile-centerW)>1) visible=false;
        }
        if(!visible) s[size_t(row)*n+j]=-INFINITY;
        m=fmaxf(m,s[size_t(row)*n+j]);
    }
    scratch[threadIdx.x]=m; __syncthreads();
    for(int d=128;d;d/=2) { if(threadIdx.x<d) scratch[threadIdx.x]=fmaxf(scratch[threadIdx.x],scratch[threadIdx.x+d]); __syncthreads(); }
    if(threadIdx.x==0) mx=scratch[0]; __syncthreads();
    float sum=0;
    for(int j=threadIdx.x;j<n;j+=blockDim.x) { float p=isfinite(mx)?expf(s[size_t(row)*n+j]-mx):0.f; s[size_t(row)*n+j]=p; sum+=p; }
    sum=sumBlock(sum,scratch); if(threadIdx.x==0) inv=sum>0?1.f/sum:0.f; __syncthreads();
    if(inverseSum) { if(threadIdx.x==0) inverseSum[row]=inv; }
    else for(int j=threadIdx.x;j<n;j+=blockDim.x) s[size_t(row)*n+j]*=inv;
}
template<class T> __global__ void scatterHead(const float *x, T *y, int rows, int dim, int width, const float *inverseSum) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*dim;i+=gridDim.x*blockDim.x) y[size_t(i/dim)*width+i%dim]=cast<T>(x[i]*(inverseSum?inverseSum[i/dim]:1.f));
}
__global__ void scatterHeads(const float *x,float *y,int rows,int dim,int heads,int groups,int group) {
    size_t n=size_t(rows)*dim*heads;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int head=i/(rows*dim),row=i/dim%rows,c=i%dim;
        y[size_t(row)*heads*groups*dim+(head*groups+group)*dim+c]=x[i];
    }
}
int grid(size_t n) { return int(std::min<size_t>((n+255)/256,4096)); }
#define DISPATCH_TENSOR(TYPE, T, ...) \
    if ((TYPE)==DType::FP32) { using T=float; __VA_ARGS__; } \
    else if ((TYPE)==DType::FP16) { using T=H; __VA_ARGS__; } \
    else { using T=B; __VA_ARGS__; }
template<class T,class U> __global__ void castKernel(const T *x,U *y,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) y[i]=cast<U>(f(x[i]));
}

template<class T> __global__ void resizeHeads(const T *x,T *out,size_t n,int from,int to) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int d=i%to; out[i]=d<from ? x[i/to*from+d] : cast<T>(0.f);
    }
}
template<class T> __global__ void meanKernel(const T *x,T *y,int rows,int cols) {
    for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<cols;c+=gridDim.x*blockDim.x) {
        float sum=0; for(int r=0;r<rows;++r) sum+=f(x[size_t(r)*cols+c]);
        // Official masked sum is returned in BF16 before the BF16 division.
        y[c]=cast<T>(roundTo<T>(sum)/roundTo<T>(float(rows)));
    }
}
template<class T> __global__ void conditionKernel(const T *x,T *y,int n,int height,int width) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(n)*4096;i+=size_t(gridDim.x)*blockDim.x) {
        int token=i/4096,c=i%4096,w=token%(2*width),h=token/(2*width)%(2*height),t=token/(4*width*height);
        y[i]=x[size_t((t*height+h/2)*width+w/2)*16384+(h%2*2+w%2)*4096+c];
    }
}
template<class T> __global__ void pixelQKKernel(const T *x,T *y,const T *wt,const T *wh,int cols,int height,int width) {
    // One block per token/head. Norm and RoPE deliberately stay in FP32 here;
    // this differs from the DiT's BF16 QK-norm boundary.
    __shared__ float scratch[256],inv[2],normalized[128];
    int head=blockIdx.x, lane=threadIdx.x;
    float v=lane<128?f(x[size_t(head)*128+lane]):0.f;
    float sum=sumBlock(lane<64?v*v:0.f,scratch); if(lane==0) inv[0]=rsqrtf(sum/64.f+1e-6f);
    sum=sumBlock(lane>=64 && lane<128?v*v:0.f,scratch); if(lane==0) inv[1]=rsqrtf(sum/64.f+1e-6f);
    __syncthreads();
    if(lane<128) normalized[lane]=v*inv[lane/64]*f(lane<64?wt[lane]:wh[lane-64]);
    __syncthreads();
    if(lane<128) {
        int token=head/(cols/128),dim=lane<64?64:32,base=lane<64?0:lane<96?64:96,local=lane-base;
        int position=lane<64?token/(height*width):lane<96?token/width%height:token%width;
        float theta=lane<64?1000000.f:10000.f;
        float angle=float(position)*powf(theta,-2.f*float(local%(dim/2))/dim);
        float rotated=local<dim/2?-normalized[base+local+dim/2]:normalized[base+local-dim/2];
        y[size_t(head)*128+lane]=cast<T>(normalized[lane]*cosf(angle)+rotated*sinf(angle));
    }
}
template<class T> __global__ void qwenVisionPatches(const uint8_t *first,const uint8_t *second,T *out,int width,int gw) {
    int i=blockIdx.x,py=(i/4/(gw/2))*2+i/2%2,px=(i/4%(gw/2))*2+i%2;
    for(int j=threadIdx.x;j<1536;j+=blockDim.x) {
        int c=j/512,t=j/256%2,y=j/16%16,x=j%16;
        size_t source=(size_t(py*16+y)*width+px*16+x)*4;
        const auto *pixels=t && second?second:first;
        int alpha=pixels[source+3],value=(pixels[source+c]*alpha+255*(255-alpha)+127)/255;
        float v=(value*(1.f/255.f)-.5f)*2.f;
        out[size_t(i)*1536+j]=cast<T>(v);
    }
}
template<class T> __global__ void qwenVisionPositions(const T *learned,T *out,int *positions,int gh,int gw) {
    int i=blockIdx.x,py=(i/4/(gw/2))*2+i/2%2,px=(i/4%(gw/2))*2+i%2;
    if(!threadIdx.x) { positions[i]=py; positions[gh*gw+i]=px; }
    float fy=float(py)*47.f/(gh-1),fx=float(px)*47.f/(gw-1);
    int y0=int(fy),x0=int(fx),y1=min(y0+1,47),x1=min(x0+1,47);
    float wy=fy-y0,wx=fx-x0;
    for(int c=threadIdx.x;c<1152;c+=blockDim.x) {
        float v=f(learned[(y0*48+x0)*1152+c])*(1-wy)*(1-wx)+
                f(learned[(y0*48+x1)*1152+c])*(1-wy)*wx+
                f(learned[(y1*48+x0)*1152+c])*wy*(1-wx)+
                f(learned[(y1*48+x1)*1152+c])*wy*wx;
        out[size_t(i)*1152+c]=cast<T>(v);
    }
}
template<class T, bool Vision>
__global__ void qwenRopeKernel(const T *x,T *y,const int *positions,size_t n,int cols,int dim,
                               float theta,bool multimodal) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int j=i%dim, half=dim/2, freq=j%half;
        int axis=Vision ? freq/(dim/4) : (multimodal && freq<60 ? freq%3 : 0);
        float power=Vision ? -2.f*(freq%(dim/4))/half : -2.f*freq/dim;
        float angle=positions[axis*(n/cols)+i/cols]*powf(theta,power);
        float rotated=j<half ? -f(x[i+half]) : f(x[i-half]);
        if constexpr(Vision)
            y[i]=cast<T>(f(x[i])*cosf(angle)+rotated*sinf(angle));
        else
            y[i]=cast<T>(roundTo<T>(f(x[i])*roundTo<T>(cosf(angle)))+
                          roundTo<T>(rotated*roundTo<T>(sinf(angle))));
    }
}
template<class T> __global__ void geluExactKernel(const T *x,T *y,size_t n) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        float v=f(x[i]); y[i]=cast<T>(.5f*v*(1.f+erff(v*.7071067811865475f)));
    }
}
template<class T> __global__ void textRopeKernel(const T *x,T *y,size_t n,int cols,float theta) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        int d=i%128;
        float angle=float(i/cols)*powf(theta,-2.f*float(d%64)/128.f);
        float rotated=d<64?-f(x[i+64]):f(x[i-64]);
        // Transformers' Qwen RoPE cos/sin are cast to BF16, and its eager
        // multiply and add each return BF16 (unlike the DiT/PixelUMM RoPE).
        y[i]=cast<T>(roundTo<T>(f(x[i])*roundTo<T>(cosf(angle)))+roundTo<T>(rotated*roundTo<T>(sinf(angle))));
    }
}
}

TensorPool::TensorPool(int device):pool_(std::make_shared<MemoryPool>(device,UINT64_MAX)) {}
TensorPool::Scope::Scope(const TensorPool &pool):previous_(activePool) {
    activePool=std::static_pointer_cast<MemoryPool>(pool.pool_);
}
TensorPool::Scope::~Scope() { activePool=std::static_pointer_cast<MemoryPool>(previous_); }
size_t TensorPool::reusableBytes() const {
    auto pool=std::static_pointer_cast<MemoryPool>(pool_);
    uint64_t reserved=0,used=0;
    check(cudaMemPoolGetAttribute(pool->handle,cudaMemPoolAttrReservedMemCurrent,&reserved));
    check(cudaMemPoolGetAttribute(pool->handle,cudaMemPoolAttrUsedMemCurrent,&used));
    return reserved>used?reserved-used:0;
}
void TensorPool::trim() {
    check(cudaMemPoolTrimTo(std::static_pointer_cast<MemoryPool>(pool_)->handle,0));
}

struct Runtime::LtState {
    struct Plan {
        cublasLtMatmulDesc_t op=nullptr;
        cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
        cublasLtMatmulPreference_t preference=nullptr;
        cublasLtMatmulAlgo_t algorithm;
        ~Plan() { if(op) cublasLtMatmulDescDestroy(op); if(a) cublasLtMatrixLayoutDestroy(a); if(b) cublasLtMatrixLayoutDestroy(b); if(c) cublasLtMatrixLayoutDestroy(c); if(preference) cublasLtMatmulPreferenceDestroy(preference); }
    };
    // Fixed workspaces keep the BF16 algorithm choice reproducible. cuBLASLt
    // and cuBLAS have separate defaults in the reference runtime.
    size_t workspaceBytes=1024*1024;
    std::shared_ptr<void> workspace=allocate(workspaceBytes);
    std::shared_ptr<void> blasWorkspace=allocate(32*1024*1024);
    std::unordered_map<std::string,std::unique_ptr<Plan>> plans;
};
Tensor::Tensor(int r,int c,bool full):Tensor(r,c,full ? DType::FP32 : DType::BF16) {}
Tensor::Tensor(int r,int c,DType type):rows(r),cols(c),dtype(type) {
    require(type==DType::FP32 || type==DType::FP16 || type==DType::BF16,"invalid tensor dtype");
    require(r>0 && c>0 && size_t(r)<=SIZE_MAX/size_t(c)/4,"invalid tensor shape");
    storage=allocate(bytes()); data=static_cast<uint16_t *>(storage.get());
}
Tensor Tensor::rowSlice(int first,int count) const {
    require(first>=0 && count>0 && first<=rows-count,"row slice out of range");
    Tensor out=*this; out.data+=size_t(first)*cols*(isFloat32()?2:1); out.rows=count; return out;
}
namespace {
bool runtimePrecision() {
    const char *p = std::getenv("FASTLLM_NOMAS_PRECISION");
    if (p && std::string(p)!="fp32" && std::string(p)!="bf16")
        throw std::invalid_argument("FASTLLM_NOMAS_PRECISION must be fp32 or bf16");
    return p && std::string(p)=="fp32";
}
}
Runtime::Runtime(int device):Runtime(device, runtimePrecision()) {}
Runtime::Runtime(int device, bool fp32):Runtime(device,fp32 ? DType::FP32 : DType::BF16) {}
Runtime::Runtime(int device, DType type):device_(device),dtype_(type) {
    require(type==DType::FP32 || type==DType::FP16 || type==DType::BF16,"invalid runtime dtype");
    check(cudaSetDevice(device)); cublasHandle_t h; check(cublasCreate(&h)); blas_=h;
    check(cublasSetStream(h,cudaStreamPerThread));
    check(cublasSetMathMode(h,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    lt_.reset(new LtState);
    check(cublasSetWorkspace(h,lt_->blasWorkspace.get(),32*1024*1024));
    // HTTP constructs the runtime on one thread and invokes it on its worker.
    check(cudaStreamSynchronize(cudaStreamPerThread));
}
Runtime::~Runtime() { cudaSetDevice(device_); lt_.reset(); if(blas_) cublasDestroy(static_cast<cublasHandle_t>(blas_)); }
void Runtime::activate() { check(cudaSetDevice(device_)); }
Tensor Runtime::cast(const Tensor &x,DType type) {
    if(x.dtype==type) return x;
    Tensor out(x.rows,x.cols,type);
    DISPATCH_TENSOR(type, U,
        if(x.dtype==DType::FP32) { castKernel<float,U><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<float>(x),as<U>(out),x.size()); }
        else if(x.dtype==DType::FP16) { castKernel<H,U><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<H>(x),as<U>(out),x.size()); }
        else { castKernel<B,U><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<B>(x),as<U>(out),x.size()); });
    check(cudaGetLastError()); return out;
}
Tensor Runtime::upload(const float *x,int r,int c) {
    require(x,"null tensor input"); check(cudaSetDevice(device_)); Tensor tmp(r,c,true);
    copy(tmp.data,x,tmp.bytes(),cudaMemcpyHostToDevice);
    return cast(tmp,dtype_);
}
Tensor Runtime::cast(const Tensor &x,bool full) {
    return cast(x,full ? DType::FP32 : DType::BF16);
}
std::vector<float> Runtime::download(const Tensor &x) {
    auto tmp=cast(x,DType::FP32); std::vector<float> out(x.size());
    copy(out.data(),tmp.data,tmp.bytes(),cudaMemcpyDeviceToHost); return out;
}
std::vector<uint8_t> Runtime::downloadPixelRGB(const Tensor &x,int width,int height,int firstFrame,int frames) {
    require(x.data && x.cols==3072 && x.rows>0 && width>0 && height>0 &&
            width%16==0 && height%16==0 && firstFrame>=0 && frames>0,
            "invalid PixelUMM RGB shape");
    const size_t spatial=size_t(width/16)*(height/16);
    require(size_t(x.rows)%spatial==0 &&
            size_t(firstFrame)+size_t(frames)<=size_t(x.rows)/spatial*4,
            "PixelUMM RGB frame range exceeds tensor");
    std::vector<uint8_t> out(size_t(frames)*height*width*3);
    auto packed=allocate(out.size()+sizeof(int));
    auto invalid=static_cast<int *>(packed.get());
    auto rgb=reinterpret_cast<uint8_t *>(invalid+1);
    check(cudaMemsetAsync(invalid,0,sizeof(int),cudaStreamPerThread));
    DISPATCH_TENSOR(x.dtype, T, pixelRGBKernel<<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<T>(x),rgb,invalid,out.size(),width,height,firstFrame));
    check(cudaGetLastError());
    int error=0;
    check(cudaMemcpyAsync(out.data(),rgb,out.size(),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaMemcpyAsync(&error,invalid,sizeof(error),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    if(error) throw std::runtime_error("non-finite PixelUMM output");
    return out;
}
Tensor Runtime::linear(const Tensor &x,const Tensor &w,const Tensor *bias) {
    Tensor out(x.rows,w.rows,x.dtype);
    linearInto(x,w,out,bias);
    return out;
}
void Runtime::linearInto(const Tensor &x,const Tensor &w,Tensor &out,const Tensor *bias) {
    require(x.cols==w.cols,"linear dimension mismatch"); require(!bias || bias->size()==size_t(w.rows),"linear bias mismatch");
    require(x.dtype==w.dtype && (!bias || bias->dtype==x.dtype),"linear dtype mismatch");
    require(out.data && out.rows==x.rows && out.cols==w.rows && out.dtype==x.dtype,"linear output shape/dtype mismatch");
    require(out.storage!=x.storage && out.storage!=w.storage && (!bias || out.storage!=bias->storage),
            "linear output must not share input storage");
    require(x.isFloat32() || !bias || reinterpret_cast<uintptr_t>(out.data)%256==0,
            "16-bit biased linear output must be 256-byte aligned");
    float one=1,zero=0;
    if(x.isFloat32()) {
        check(cublasSgemm(static_cast<cublasHandle_t>(blas_),CUBLAS_OP_T,CUBLAS_OP_N,w.rows,x.rows,x.cols,
            &one,as<float>(w),w.cols,as<float>(x),x.cols,&zero,as<float>(out),out.cols));
        if(bias) addBias<float><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<float>(out),as<float>(*bias),as<float>(out),out.size(),out.cols);
        check(cudaGetLastError()); return;
    }
    auto scalarType=x.dtype==DType::FP16 ? CUDA_R_16F : CUDA_R_16BF;
    if(bias) {
        auto key=std::to_string(int(x.dtype))+":"+std::to_string(x.rows)+":"+std::to_string(x.cols)+":"+std::to_string(w.rows);
        auto &plan=lt_->plans[key];
        if(!plan) {
            auto p=std::make_unique<LtState::Plan>();
            check(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
            cublasOperation_t trans=CUBLAS_OP_T; check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
            cublasLtEpilogue_t epilogue=CUBLASLT_EPILOGUE_BIAS;
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
            check(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias->data,sizeof(void *)));
            check(cublasLtMatrixLayoutCreate(&p->a,scalarType,w.cols,w.rows,w.cols));
            check(cublasLtMatrixLayoutCreate(&p->b,scalarType,x.cols,x.rows,x.cols));
            check(cublasLtMatrixLayoutCreate(&p->c,scalarType,w.rows,x.rows,w.rows));
            check(cublasLtMatmulPreferenceCreate(&p->preference));
            check(cublasLtMatmulPreferenceSetAttribute(p->preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&lt_->workspaceBytes,sizeof(size_t)));
            uint32_t reduction=CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE|CUBLASLT_REDUCTION_SCHEME_NONE,alignment=256;
            check(cublasLtMatmulPreferenceSetAttribute(p->preference,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&reduction,sizeof(reduction)));
            for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
                check(cublasLtMatmulPreferenceSetAttribute(p->preference,attr,&alignment,sizeof(alignment)));
            int count=0; cublasLtMatmulHeuristicResult_t choice{};
            check(cublasLtMatmulAlgoGetHeuristic(reinterpret_cast<cublasLtHandle_t>(blas_),p->op,p->a,p->b,p->c,p->c,p->preference,1,&choice,&count));
            require(count>0,"no cuBLASLt 16-bit linear algorithm"); p->algorithm=choice.algo; plan=std::move(p);
        }
        check(cublasLtMatmulDescSetAttribute(plan->op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias->data,sizeof(void *)));
        check(cublasLtMatmul(reinterpret_cast<cublasLtHandle_t>(blas_),plan->op,&one,w.data,plan->a,x.data,plan->b,&zero,
            out.data,plan->c,out.data,plan->c,&plan->algorithm,lt_->workspace.get(),lt_->workspaceBytes,cudaStreamPerThread));
    } else {
        check(cublasGemmEx(static_cast<cublasHandle_t>(blas_),CUBLAS_OP_T,CUBLAS_OP_N,
            w.rows,x.rows,x.cols,&one,w.data,scalarType,w.cols,x.data,scalarType,x.cols,
            &zero,out.data,scalarType,w.rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
    check(cudaGetLastError());
}
Tensor Runtime::norm(const Tensor &x,bool rms,const Tensor *w,const Tensor *bias,float eps,int width) {
    if(!width) width=x.cols;
    require(width>0 && x.cols%width==0,"norm dimension mismatch");
    require((!w || w->dtype==x.dtype) && (!bias || bias->dtype==x.dtype),"norm dtype mismatch");
    require((!w || w->size()==size_t(width)) && (!bias || bias->size()==size_t(width)),"norm weight mismatch");
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, normKernel<T><<<x.size()/width,256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),w?as<T>(*w):nullptr,bias?as<T>(*bias):nullptr,width,rms,eps)); check(cudaGetLastError()); return out;
}
Tensor Runtime::activation(const Tensor &x,bool gelu) {
    Tensor out(x.rows,x.cols,x.dtype); DISPATCH_TENSOR(x.dtype, T, element<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),nullptr,as<T>(out),x.size(),gelu?2:1)); check(cudaGetLastError()); return out;
}
Tensor Runtime::add(const Tensor &x,const Tensor &y) {
    require(x.rows==y.rows && x.cols==y.cols && x.dtype==y.dtype,"add shape/dtype mismatch"); Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, element<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(y),as<T>(out),x.size(),0)); check(cudaGetLastError()); return out;
}
Tensor Runtime::multiply(const Tensor &x,const Tensor &y) {
    require(x.rows==y.rows && x.cols==y.cols && x.dtype==y.dtype,"multiply shape/dtype mismatch"); Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, element<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(y),as<T>(out),x.size(),3)); check(cudaGetLastError()); return out;
}
Tensor Runtime::siluMultiply(const Tensor &x,const Tensor &y) {
    require(x.rows==y.rows && x.cols==y.cols && x.dtype==y.dtype,"SiLU multiply shape/dtype mismatch");
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, element<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(y),as<T>(out),x.size(),4));
    check(cudaGetLastError()); return out;
}

Tensor Runtime::bias(const Tensor &x,const Tensor &bias) {
    require(bias.size()==size_t(x.cols) && bias.dtype==x.dtype,"bias shape/dtype mismatch"); Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, addBias<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(bias),as<T>(out),x.size(),x.cols)); check(cudaGetLastError()); return out;
}
Tensor Runtime::repeatRows(const Tensor &x,int copies) {
    require(copies>0 && int64_t(x.rows)*copies<=16384,"invalid row replication");
    Tensor out(x.rows*copies,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, repeatRowsKernel<T><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),out.size(),x.cols,copies));
    check(cudaGetLastError()); return out;
}
Tensor Runtime::mean(const Tensor &x) {
    Tensor out(1,x.cols,x.dtype); DISPATCH_TENSOR(x.dtype, T, meanKernel<T><<<grid(x.cols),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),x.rows,x.cols)); check(cudaGetLastError()); return out;
}
Tensor Runtime::pixelCondition(const Tensor &x,int t,int h,int w) {
    require(x.rows==t*h*w && x.cols==16384,"invalid pixel condition shape"); Tensor out(x.rows*4,4096,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, conditionKernel<T><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),out.rows,h,w)); check(cudaGetLastError()); return out;
}
Tensor Runtime::pixelQKRope(const Tensor &x,const Tensor &nt,const Tensor &nh,int t,int h,int w) {
    require(t>0 && h>0 && w>0 && x.rows==t*h*w && x.cols%128==0 && nt.size()==64 && nh.size()==64 && x.dtype==nt.dtype && x.dtype==nh.dtype,"invalid PixelUMM QK shape");
    Tensor out(x.rows,x.cols,x.dtype); DISPATCH_TENSOR(x.dtype, T, pixelQKKernel<T><<<x.size()/128,256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),as<T>(nt),as<T>(nh),x.cols,h,w)); check(cudaGetLastError()); return out;
}
Tensor Runtime::modulate(const Tensor &x,const Tensor &m,int shift,int scale) {
    require(m.dtype==x.dtype && m.rows==1 && shift>=0 && scale>=0 && m.cols>=(std::max(shift,scale)+1)*x.cols,"modulation dimension mismatch");
    Tensor out(x.rows,x.cols,x.dtype); DISPATCH_TENSOR(x.dtype, T, modulation<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),nullptr,as<T>(m),as<T>(out),x.size(),x.cols,shift,scale,false)); check(cudaGetLastError()); return out;
}
Tensor Runtime::residual(const Tensor &x,const Tensor &y,const Tensor &m,int gate) {
    require(x.dtype==y.dtype && x.dtype==m.dtype && x.rows==y.rows && x.cols==y.cols && m.rows==1 && gate>=0 && m.cols>=(gate+1)*x.cols,"residual dimension mismatch");
    Tensor out(x.rows,x.cols,x.dtype); DISPATCH_TENSOR(x.dtype, T, modulation<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(y),as<T>(m),as<T>(out),x.size(),x.cols,gate,0,true)); check(cudaGetLastError()); return out;
}
Tensor Runtime::columns(const Tensor &x,int first,int count) {
    require(first>=0 && count>0 && first<=x.cols-count,"column slice out of range"); Tensor out(x.rows,count,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, columnKernel<T><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),x.rows,x.cols,first,count)); check(cudaGetLastError()); return out;
}
Tensor Runtime::embedding(const Tensor &weight,const std::vector<int> &tokens) {
    require(!tokens.empty() && tokens.size()<=65536,"invalid token count"); Tensor out(tokens.size(),weight.cols,weight.dtype);
    for(size_t i=0;i<tokens.size();++i) {
        require(tokens[i]>=0 && tokens[i]<weight.rows,"token out of vocabulary");
        copy(out.data+i*weight.cols*(weight.isFloat32()?2:1),weight.data+size_t(tokens[i])*weight.cols*(weight.isFloat32()?2:1),weight.cols*(weight.isFloat32()?4:2),cudaMemcpyDeviceToDevice);
    }
    return out;
}
Tensor Runtime::indices(const std::vector<int> &values) {
    require(!values.empty() && values.size()<=INT32_MAX,"invalid index count");
    Tensor out(values.size(),1,true);
    copy(out.data,values.data(),values.size()*sizeof(int),cudaMemcpyHostToDevice);
    return out;
}
Tensor Runtime::textRope(const Tensor &x,const std::vector<int> &pos,float theta,int dim) {
    return textRope(x,indices(pos),theta,dim);
}
Tensor Runtime::textRope(const Tensor &x,const Tensor &positions,float theta,int dim) {
    require(positions.isFloat32() && positions.cols==1 && positions.data &&
            (positions.size()==size_t(x.rows) || positions.size()==size_t(x.rows)*3) &&
            dim>0 && dim%2==0 && x.cols%dim==0,"text RoPE shape");
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype,T,(qwenRopeKernel<T,false><<<grid(x.size()),256,0,cudaStreamPerThread>>>(
        as<T>(x),as<T>(out),reinterpret_cast<int *>(positions.data),x.size(),x.cols,dim,theta,
        positions.size()!=size_t(x.rows))));
    check(cudaGetLastError()); return out;
}
Tensor Runtime::visionRope(const Tensor &x,const std::vector<int> &pos,int dim) {
    return visionRope(x,indices(pos),dim);
}
Tensor Runtime::visionRope(const Tensor &x,const Tensor &positions,int dim) {
    require(positions.isFloat32() && positions.cols==1 && positions.data &&
            positions.size()==size_t(x.rows)*2 && dim>0 && dim%4==0 && x.cols%dim==0,"vision RoPE shape");
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype,T,(qwenRopeKernel<T,true><<<grid(x.size()),256,0,cudaStreamPerThread>>>(
        as<T>(x),as<T>(out),reinterpret_cast<int *>(positions.data),x.size(),x.cols,dim,10000.f,true)));
    check(cudaGetLastError()); return out;
}
std::array<Tensor,3> Runtime::visionInputs(int width,int height,const std::vector<uint8_t> &rgba,
                                          const Tensor &learned,const std::vector<uint8_t> *secondFrame) {
    require(width>0 && height>0 && width%32==0 && height%32==0 &&
            rgba.size()==size_t(width)*height*4 && (!secondFrame || secondFrame->size()==rgba.size()),
            "vision input must be aligned RGBA");
    require(learned.dtype==dtype_ && learned.rows==48*48 && learned.cols==1152,"vision position embedding shape");
    int gh=height/16,gw=width/16;
    require(int64_t(gh)*gw<=INT32_MAX/1536,"vision input too large");
    activate();
    int n=gh*gw;
    Tensor first(width*height,1,true),second;
    copy(first.data,rgba.data(),rgba.size(),cudaMemcpyHostToDevice);
    if(secondFrame) {
        second=Tensor(width*height,1,true);
        copy(second.data,secondFrame->data(),secondFrame->size(),cudaMemcpyHostToDevice);
    }
    std::array<Tensor,3> out{Tensor(n,1536,dtype_),Tensor(n,1152,dtype_),Tensor(n*2,1,true)};
    DISPATCH_TENSOR(dtype_,T,qwenVisionPatches<T><<<n,256,0,cudaStreamPerThread>>>(
        reinterpret_cast<const uint8_t *>(first.data),reinterpret_cast<const uint8_t *>(second.data),
        as<T>(out[0]),width,gw));
    DISPATCH_TENSOR(dtype_,T,qwenVisionPositions<T><<<n,256,0,cudaStreamPerThread>>>(
        as<T>(learned),as<T>(out[1]),reinterpret_cast<int *>(out[2].data),gh,gw));
    check(cudaGetLastError()); return out;
}
Tensor Runtime::geluExact(const Tensor &x) {
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype,T,geluExactKernel<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(
        as<T>(x),as<T>(out),x.size()));
    check(cudaGetLastError()); return out;
}
Tensor Runtime::textRope(const Tensor &x) { return textRope(x,1000000.f); }
Tensor Runtime::textRope(const Tensor &x,float theta) {
    require(x.cols%128==0,"invalid Qwen RoPE shape"); Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, textRopeKernel<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),x.size(),x.cols,theta)); check(cudaGetLastError()); return out;
}
Tensor Runtime::concat(const std::vector<Tensor> &xs) {
    require(!xs.empty(),"empty concat"); int rows=0;
    for(auto &x:xs) { require(x.cols==xs[0].cols && x.dtype==xs[0].dtype && x.rows<=INT32_MAX-rows,"concat shape mismatch"); rows+=x.rows; }
    Tensor out(rows,xs[0].cols,xs[0].dtype); size_t pos=0;
    for(auto &x:xs) { copy(out.data+pos,x.data,x.bytes(),cudaMemcpyDeviceToDevice); pos+=x.size()*(x.isFloat32()?2:1); } return out;
}
Tensor Runtime::rope(const Tensor &x,const float *cos,const float *sin,int dim) {
    return rope(x,prepareRope(cos,sin,x.rows,dim));
}
RopeTable Runtime::prepareRope(const float *cos,const float *sin,int rows,int dim) {
    require(cos && sin && rows>0 && dim>0 && dim%2==0,"invalid RoPE table");
    // Position tables stay FP32 even when the network's activations are BF16.
    RopeTable table{Tensor(rows,dim,true),Tensor(rows,dim,true)};
    copy(table.cos.data,cos,table.cos.bytes(),cudaMemcpyHostToDevice);
    copy(table.sin.data,sin,table.sin.bytes(),cudaMemcpyHostToDevice);
    return table;
}
Tensor Runtime::rope(const Tensor &x,const RopeTable &table) {
    int dim=table.cos.cols;
    require(dim>0 && dim%2==0 && x.cols%dim==0 && table.cos.rows==x.rows && table.sin.rows==x.rows &&
            table.sin.cols==dim && table.cos.fp32 && table.sin.fp32,"invalid RoPE shape");
    Tensor out(x.rows,x.cols,x.dtype);
    DISPATCH_TENSOR(x.dtype, T, ropeKernel<T><<<grid(x.size()),256,0,cudaStreamPerThread>>>(as<T>(x),as<T>(out),as<float>(table.cos),as<float>(table.sin),x.size(),x.cols,dim));
    check(cudaGetLastError()); return out;
}
#undef DISPATCH_TENSOR
Tensor Runtime::attention(const Tensor &q,const Tensor &k,const Tensor &v,int heads,const uint8_t *mask,int chunk,
                          int kvHeads,bool causal,int tile,int height,int width) {
    require(heads>0 && q.cols%heads==0,"invalid attention heads");
    return attentionScaled(q,k,v,heads,1.f/std::sqrt(float(q.cols/heads)),mask,chunk,kvHeads,causal,tile,height,width);
}
Tensor Runtime::attentionScaled(const Tensor &q,const Tensor &k,const Tensor &v,int heads,float scoreScale,
                                const uint8_t *mask,int chunk,int kvHeads,bool causal,int tile,int height,int width) {
    if(!kvHeads) kvHeads=heads;
    require(heads>0 && kvHeads>0 && heads%kvHeads==0 && q.cols%heads==0 && k.cols%kvHeads==0 && q.cols/heads==k.cols/kvHeads && k.rows==v.rows && k.cols==v.cols && chunk>0,"attention shape mismatch");
    require(!causal || q.rows==k.rows,"causal attention requires equal lengths");
    require(tile>=0 && (!tile || (height>0 && width>0 && q.rows==k.rows && q.rows%(height*width)==0)),"invalid local attention grid");
    require(q.dtype==k.dtype && q.dtype==v.dtype,"attention dtype mismatch");
    int dim=q.cols/heads; chunk=std::min(chunk,q.rows); Tensor out(q.rows,q.cols,q.dtype);
    require(std::isfinite(scoreScale) && (q.isFloat32() || scoreScale==1.f/std::sqrt(float(dim))),"custom attention scale requires FP32");
    if(!q.isFloat32() && !mask && (dim==64 || dim==128)) {
        check(fusedAttention(q.data,k.data,v.data,out.data,q.rows,k.rows,heads,kvHeads,dim,q.dtype==DType::FP16,causal,tile,height,width,scoreScale,cudaStreamPerThread));
        return out;
    }
    if(!q.isFloat32() && !mask && dim==72 && !tile) {
        // Qwen3-VL uses 72-wide heads. Pad the dot-product axes, retaining the
        // original softmax scale, to reuse the existing 128-wide fused kernel.
        Tensor qp(q.rows,heads*128,q.dtype),kp(k.rows,kvHeads*128,k.dtype),vp(v.rows,kvHeads*128,v.dtype),
               result(q.rows,heads*128,q.dtype);
        for(auto pair : {std::make_pair(&q,&qp),std::make_pair(&k,&kp),std::make_pair(&v,&vp)}) {
            if(q.dtype==DType::BF16) resizeHeads<B><<<grid(pair.second->size()),256,0,cudaStreamPerThread>>>(
                as<B>(*pair.first),as<B>(*pair.second),pair.second->size(),72,128);
            else resizeHeads<H><<<grid(pair.second->size()),256,0,cudaStreamPerThread>>>(
                as<H>(*pair.first),as<H>(*pair.second),pair.second->size(),72,128);
        }
        check(cudaGetLastError());
        check(fusedAttention(qp.data,kp.data,vp.data,result.data,q.rows,k.rows,heads,kvHeads,128,
            q.dtype==DType::FP16,causal,0,height,width,scoreScale,cudaStreamPerThread));
        if(q.dtype==DType::BF16) resizeHeads<B><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<B>(result),as<B>(out),out.size(),128,72);
        else resizeHeads<H><<<grid(out.size()),256,0,cudaStreamPerThread>>>(as<H>(result),as<H>(out),out.size(),128,72);
        check(cudaGetLastError()); return out;
    }
    require(q.dtype!=DType::FP16,"FP16 attention requires the fused 64/128-dimensional backend");
    if(q.isFloat32()) {
        // Reuse cuBLAS batched GEMM, also used by fastllm's general attention
        // implementation. All heads in a GQA group share one launch. FP32 QK,
        // softmax and PV retain the validated math path without a BF16 cast.
        auto scores=allocate(size_t(kvHeads)*chunk*k.rows*4);
        auto values=allocate(size_t(kvHeads)*chunk*dim*4);
        std::shared_ptr<void> maskDevice;
        if(mask) { maskDevice=allocate(size_t(q.rows)*k.rows); copy(maskDevice.get(),mask,size_t(q.rows)*k.rows,cudaMemcpyHostToDevice); }
        float scale=scoreScale,one=1,zero=0;
        auto h=static_cast<cublasHandle_t>(blas_); int groups=heads/kvHeads;
        for(int start=0;start<q.rows;start+=chunk) for(int group=0;group<groups;++group) {
            int m=std::min(chunk,q.rows-start);
            check(cublasSgemmStridedBatched(h,CUBLAS_OP_T,CUBLAS_OP_N,k.rows,m,dim,&scale,
                as<float>(k),k.cols,dim,as<float>(q)+size_t(start)*q.cols+group*dim,q.cols,groups*dim,
                &zero,static_cast<float *>(scores.get()),k.rows,int64_t(m)*k.rows,kvHeads));
            softmax<<<m*kvHeads,256,0,cudaStreamPerThread>>>(static_cast<float *>(scores.get()),static_cast<uint8_t *>(maskDevice.get()),k.rows,start,causal,tile,height,width,nullptr,m);
            check(cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,dim,m,k.rows,&one,
                as<float>(v),v.cols,dim,static_cast<float *>(scores.get()),k.rows,int64_t(m)*k.rows,
                &zero,static_cast<float *>(values.get()),dim,int64_t(m)*dim,kvHeads));
            scatterHeads<<<grid(size_t(m)*dim*kvHeads),256,0,cudaStreamPerThread>>>(static_cast<float *>(values.get()),as<float>(out)+size_t(start)*q.cols,m,dim,kvHeads,groups,group);
            check(cudaGetLastError());
        }
        return out;
    }
    auto scores=allocate(size_t(chunk)*k.rows*4), vf=allocate(v.size()*4), values=allocate(size_t(chunk)*dim*4);
    auto probabilities=allocate(size_t(chunk)*k.rows*2),inverse=allocate(size_t(chunk)*4);
    std::shared_ptr<void> maskDevice;
    if(mask) { maskDevice=allocate(size_t(q.rows)*k.rows); copy(maskDevice.get(),mask,size_t(q.rows)*k.rows,cudaMemcpyHostToDevice); }
    castKernel<B,float><<<grid(v.size()),256,0,cudaStreamPerThread>>>(ptr(v),static_cast<float *>(vf.get()),v.size()); check(cudaGetLastError());
    float alpha=scoreScale,one=1,zero=0; auto h=static_cast<cublasHandle_t>(blas_);
    for(int head=0;head<heads;++head) for(int start=0;start<q.rows;start+=chunk) {
        int m=std::min(chunk,q.rows-start);
        check(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,k.rows,m,dim,&alpha,
            ptr(k)+(head/(heads/kvHeads))*dim,CUDA_R_16BF,k.cols,ptr(q)+size_t(start)*q.cols+head*dim,CUDA_R_16BF,q.cols,
            &zero,scores.get(),CUDA_R_32F,k.rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        softmax<<<m,256,0,cudaStreamPerThread>>>(static_cast<float *>(scores.get()),static_cast<uint8_t *>(maskDevice.get()),k.rows,start,causal,tile,height,width,
            mask?nullptr:static_cast<float *>(inverse.get())); check(cudaGetLastError());
        if(mask) {
            check(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,dim,m,k.rows,&one,
                static_cast<float *>(vf.get())+(head/(heads/kvHeads))*dim,v.cols,static_cast<float *>(scores.get()),k.rows,
                &zero,static_cast<float *>(values.get()),dim));
        } else {
            // FlashAttention's BF16 path rounds the unnormalized exponentials
            // before PV, then divides the FP32 accumulator by the FP32 row sum.
            castKernel<float,B><<<grid(size_t(m)*k.rows),256,0,cudaStreamPerThread>>>(static_cast<float *>(scores.get()),static_cast<B *>(probabilities.get()),size_t(m)*k.rows);
            check(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,dim,m,k.rows,&one,
                ptr(v)+(head/(heads/kvHeads))*dim,CUDA_R_16BF,v.cols,probabilities.get(),CUDA_R_16BF,k.rows,
                &zero,values.get(),CUDA_R_32F,dim,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        }
        scatterHead<B><<<grid(size_t(m)*dim),256,0,cudaStreamPerThread>>>(static_cast<float *>(values.get()),ptr(out)+size_t(start)*q.cols+head*dim,m,dim,q.cols,
            mask?nullptr:static_cast<float *>(inverse.get())); check(cudaGetLastError());
    }
    return out;
}
void Runtime::synchronize() { check(cudaDeviceSynchronize()); }

struct Weights::Impl {
    struct HostCache {
        struct Buffer { std::shared_ptr<void> data; bool pinned; };
        std::unordered_map<std::string,Buffer> tensors;
        size_t bytes=0,budget=0;
        cudaEvent_t copied=nullptr;
        int device=0;
        std::shared_ptr<void> thread;
        ~HostCache() {
            if(!copied) return;
            int previous=device; cudaGetDevice(&previous); cudaSetDevice(device);
            cudaEventSynchronize(copied);
            cudaEventDestroy(copied);
            tensors.clear();
            cudaSetDevice(previous);
        }
        void upload(void *dst,const void *src,size_t count) {
            // Hold a unique per-thread token: OS thread IDs can be reused even
            // though the replacement thread has a different CUDA stream.
            static thread_local auto current=std::make_shared<int>(0);
            if(!copied) {
                check(cudaGetDevice(&device));
                check(cudaEventCreateWithFlags(&copied,cudaEventDisableTiming));
            } else if(thread!=current) {
                check(cudaEventSynchronize(copied));
            }
            auto status=cudaMemcpyAsync(dst,src,count,cudaMemcpyHostToDevice,cudaStreamPerThread);
            if(status==cudaSuccess) status=cudaEventRecord(copied,cudaStreamPerThread);
            if(status!=cudaSuccess) {
                // An unsuccessful record must not leave the source unprotected.
                cudaStreamSynchronize(cudaStreamPerThread);
                check(status);
            }
            thread=current;
        }
    };
    struct PackedCache {
        std::unordered_map<std::string,Tensor> tensors;
        size_t bytes=0,budget=0;
    };
    Runtime &rt; std::ifstream file; json11::Json header; uint64_t base=0,fileSize=0;
    std::shared_ptr<void> mapping;
    std::unordered_map<std::string,Tensor> cache;
    struct HostWeight { int rows,cols; DType dtype; std::vector<uint8_t> bytes; };
    std::unordered_map<std::string,HostWeight> offloaded;
    std::shared_ptr<PackedCache> packed;
    std::shared_ptr<HostCache> host;
    std::string directory,checkpoint;
    std::unordered_map<std::string,std::unique_ptr<Weights>> shards;
    Impl(Runtime &rt,const std::string &path):rt(rt),checkpoint(path) {
        if(std::filesystem::is_directory(path)) {
            directory=path;
            std::ifstream index(path+"/model.safetensors.index.json");
            if (!index) { index.clear(); index.open(path+"/diffusion_pytorch_model.safetensors.index.json"); }
            require(bool(index),"missing sharded safetensors index");
            std::string s((std::istreambuf_iterator<char>(index)),std::istreambuf_iterator<char>()),err;
            header=json11::Json::parse(s,err)["weight_map"];
            require(err.empty() && header.is_object(),"invalid sharded weight map"); return;
        }
        file.open(path,std::ios::binary);
        require(bool(file),"cannot open HY-Nomas safetensors checkpoint");
        file.seekg(0,std::ios::end); fileSize=file.tellg(); file.seekg(0);
        uint64_t len=0; file.read(reinterpret_cast<char *>(&len),8);
        require(fileSize>=8 && len>0 && len<=64*1024*1024 && len<=fileSize-8,"invalid safetensors header length");
        std::string s(len,'\0'); file.read(&s[0],len); require(bool(file),"truncated safetensors header");
        std::string err; header=json11::Json::parse(s,err); require(err.empty() && header.is_object(),"invalid safetensors JSON"); base=8+len;
#ifndef _WIN32
        // Stream directly from the OS page cache instead of copying each layer through
        // a second large host allocation. Mapping is read-only and optional.
        int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd>=0) {
            void *address=mmap(nullptr,fileSize,PROT_READ,MAP_PRIVATE,fd,0);
            close(fd);
            if(address!=MAP_FAILED) mapping={address,[size=fileSize](void *p) { munmap(p,size); }};
        }
#endif
    }
};
Weights::Weights(Runtime &rt,const std::string &path):impl_(new Impl(rt,path)) {}
Weights::~Weights()=default;
const Tensor *Weights::optional(const std::string &name) {
    if(impl_->header[name].is_null()) return nullptr; return &get(name);
}
const Tensor &Weights::get(const std::string &name) {
    impl_->rt.activate();
    auto &i=*impl_; auto cached=i.cache.find(name); if(cached!=i.cache.end()) return cached->second;
    auto saved=i.offloaded.find(name);
    if(saved!=i.offloaded.end()) {
        const auto &h=saved->second;
        Tensor value(h.rows,h.cols,h.dtype);
        copy(value.data,h.bytes.data(),h.bytes.size(),cudaMemcpyHostToDevice);
        return i.cache.emplace(name,std::move(value)).first->second;
    }
    auto meta=i.header[name]; if(meta.is_null()) throw std::runtime_error("missing HY-Nomas weight: "+name);
    if(!i.directory.empty()) {
        auto file=meta.string_value();
        require(!file.empty() && file.find('/')==std::string::npos && file.find('\\')==std::string::npos && file!="..","invalid shard filename");
        auto &shard=i.shards[file];
        if(!shard) {
            shard.reset(new Weights(i.rt,i.directory+"/"+file));
            shard->impl_->packed=i.packed;
            shard->impl_->host=i.host;
        }
        return shard->get(name);
    }
    auto shape=meta["shape"].array_items(); require(!shape.empty(),"scalar weight unsupported");
    uint64_t n=1;
    for(auto &d:shape) { double v=d.number_value(); require(v>=1 && v<=INT32_MAX && std::floor(v)==v && n<=SIZE_MAX/uint64_t(v)/4,"invalid weight dimensions"); n*=uint64_t(v); }
    int rows=shape.size()==1?1:shape[0].int_value(); require(n/rows<=INT32_MAX,"weight dimension too large"); int cols=int(n/rows);
    auto offsets=meta["data_offsets"].array_items(); require(offsets.size()==2,"invalid safetensors offsets");
    auto dtype=meta["dtype"].string_value(); require(dtype=="F32" || dtype=="BF16","HY-Nomas baseline supports F32/BF16 checkpoints only");
    double a=offsets[0].number_value(),z=offsets[1].number_value();
    require(a>=0 && z>=a && std::floor(a)==a && std::floor(z)==z && z<=double(i.fileSize-i.base),"weight offset outside file");
    require(z-a==double(n*(dtype=="F32"?4:2)),"weight size mismatch");
    Tensor raw;
    if(i.packed && dtype=="BF16" && i.rt.isFloat32()) {
        auto cached=i.packed->tensors.find(name);
        if(cached!=i.packed->tensors.end()) raw=cached->second;
    }
    std::vector<uint8_t> buffer;
    const void *source=nullptr;
    bool pinned=false;
    if(!raw.data && i.host) {
        auto cached=i.host->tensors.find(name);
        if(cached!=i.host->tensors.end()) {
            source=cached->second.data.get(); pinned=cached->second.pinned;
        }
    }
    if(!raw.data && !source && i.mapping) source=static_cast<const uint8_t *>(i.mapping.get())+i.base+uint64_t(a);
    else if(!raw.data && !source) {
        buffer.resize(uint64_t(z-a)); i.file.clear(); i.file.seekg(i.base+uint64_t(a));
        i.file.read(reinterpret_cast<char *>(buffer.data()),buffer.size()); require(bool(i.file),"truncated weight");
        source=buffer.data();
    }
    if(source && i.host && !i.host->tensors.count(name) && uint64_t(z-a)<=i.host->budget-i.host->bytes) {
        void *p=nullptr;
        auto status=cudaHostAlloc(&p,uint64_t(z-a),cudaHostAllocPortable);
        pinned=status==cudaSuccess;
        if(status==cudaErrorInvalidValue) {
            // Large pinned allocations can be rejected despite valid size/flags.
            // Cache ordinary host bytes instead, avoiding expensive retries and
            // file-backed uploads on subsequent requests.
            cudaGetLastError();
            p=std::malloc(uint64_t(z-a));
            if(!p) i.host->budget=i.host->bytes;
        } else if(status==cudaErrorMemoryAllocation) {
            cudaGetLastError();
            p=nullptr;
            i.host->budget=i.host->bytes;
        } else check(status);
        if(p) {
            std::shared_ptr<void> storage(p,[pinned](void *x) {
                if(pinned) cudaFreeHost(x); else std::free(x);
            });
            std::memcpy(p,source,uint64_t(z-a));
            i.host->tensors.emplace(name,Impl::HostCache::Buffer{std::move(storage),pinned});
            i.host->bytes+=uint64_t(z-a);
            source=p;
        }
    }
    if(!raw.data) {
        raw=Tensor(rows,cols,dtype=="F32" ? DType::FP32 : DType::BF16);
        // Immutable cached host buffers outlive the last recorded upload. Keep
        // inference stream ordering without a host barrier for every weight.
        if(pinned) i.host->upload(raw.data,source,raw.bytes());
        else copy(raw.data,source,raw.bytes(),cudaMemcpyHostToDevice);
        if(i.packed && dtype=="BF16" && raw.bytes()<=i.packed->budget-i.packed->bytes) {
            i.packed->tensors.emplace(name,raw);
            i.packed->bytes+=raw.bytes();
        }
    }
    auto value=i.rt.cast(raw,i.rt.dtype());
    return i.cache.emplace(name,std::move(value)).first->second;
}
Tensor Weights::linear(const Tensor &x,const std::string &prefix,bool separateBias) {
    return linear(x,prefix,separateBias,nullptr);
}
Tensor Weights::linear(const Tensor &x,const std::string &prefix,bool separateBias,Tensor *output) {
    require(!output || !separateBias,"linear output cannot be combined with separate bias");
    auto &w=get(prefix+".weight"); auto bias=optional(prefix+".bias");
    Tensor out;
    if(output) {
        impl_->rt.linearInto(x,w,*output,bias);
        out=*output;
    } else {
        out=impl_->rt.linear(x,w,separateBias?nullptr:bias);
    }
    if(separateBias && bias) out=impl_->rt.bias(out,*bias);
    return out;
}
void Weights::clear() { impl_->cache.clear(); impl_->offloaded.clear(); impl_->shards.clear(); impl_->packed.reset(); }
void Weights::offload() {
    impl_->rt.activate();
    for(const auto &entry:impl_->cache) {
        if(impl_->offloaded.count(entry.first)) continue;
        const auto &x=entry.second;
        Impl::HostWeight h{x.rows,x.cols,x.dtype,std::vector<uint8_t>(x.bytes())};
        copy(h.bytes.data(),x.data,x.bytes(),cudaMemcpyDeviceToHost);
        impl_->offloaded.emplace(entry.first,std::move(h));
    }
    impl_->cache.clear();
    for(auto &shard:impl_->shards) shard.second->offload();
}
void Weights::enableHostCache(size_t maxBytes) {
    auto &i=*impl_;
    i.host.reset();
    // Leave most host RAM available to concurrent services and the file cache.
    // If the platform cannot report available RAM, retain the existing upload path.
    std::ifstream memory("/proc/meminfo");
    std::string line;
    while(maxBytes && std::getline(memory,line)) {
        if(line.compare(0,13,"MemAvailable:")!=0) continue;
        size_t available=std::stoull(line.substr(13))*1024;
        i.host=std::make_shared<Impl::HostCache>();
        i.host->budget=std::min(maxBytes,available/4);
        break;
    }
    for(auto &shard:i.shards) shard.second->impl_->host=i.host;
}
void Weights::shareHostCache(const Weights &source) {
    require(&impl_->rt==&source.impl_->rt && impl_->checkpoint==source.impl_->checkpoint,
            "shared host cache requires the same runtime and checkpoint");
    impl_->host=source.impl_->host;
    for(auto &shard:impl_->shards) shard.second->impl_->host=impl_->host;
}
void Weights::enablePackedCache(size_t reserveBytes) {
    auto &i=*impl_;
    if(!i.rt.isFloat32() || i.packed) return;
    i.rt.activate();
    size_t available=0,total=0; check(cudaMemGetInfo(&available,&total));
    if(available<=reserveBytes) return;
    i.packed=std::make_shared<Impl::PackedCache>();
    i.packed->budget=std::min(available/2,available-reserveBytes);
    // A single budget is shared by every shard, including those already opened.
    for(auto &shard:i.shards) shard.second->impl_->packed=i.packed;
}
void Weights::erasePrefix(const std::string &prefix) {
    for(auto it=impl_->cache.begin();it!=impl_->cache.end();) {
        if(it->first.compare(0,prefix.size(),prefix)==0) it=impl_->cache.erase(it);
        else ++it;
    }
    for(auto &shard:impl_->shards) shard.second->erasePrefix(prefix);
}

}} // namespace fastllm::nomas
