#include "models/hy_nomas/runtime.h"
#include "resolution.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace fastllm { namespace nomas {
namespace {
using B = __nv_bfloat16;
B *ptr(const Tensor &x) { return reinterpret_cast<B *>(x.data); }
int grid(size_t n) { return int(std::min<size_t>((n+255)/256,65535)); }
void checkLaunch() {
    auto status=cudaGetLastError();
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Volume {
    Tensor x;
    int time, height, width;
};

__global__ void tileImage(const B *frame,B *out,size_t n,int frameSize) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)
        out[i]=frame[i%frameSize];
}

// cuBLAS consumes columns in (time, y, x, channel) order for coalesced reads.
__global__ void reorderFilter(const B *src,B *dst,size_t n,int channels) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        int channel=i%channels;
        size_t kernel=i/channels;
        dst[i]=src[((kernel/27)*channels+channel)*27+kernel%27];
    }
}

__global__ void gatherPatch(const B *src,B *dst,size_t n,int first,int channels,
                            int h,int w,int chunk) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        int k=(i/channels)%27;
        int row=first+int(i/(27*channels));
        int t=row/(h*w), y=row/w%h+k/3%3-1, x=row%w+k%3-1;
        int dt=k/9-1, sourceTime=t+dt;
        // Each 40-frame chunk sees the preceding chunk's last feature, but
        // does not see the first feature of the next chunk (DCAE-V cache).
        bool valid=sourceTime>=0 && !(dt==1 && (t+1)%chunk==0) &&
                   unsigned(y)<unsigned(h) && unsigned(x)<unsigned(w);
        dst[i]=valid?src[((size_t(sourceTime)*h+y)*w+x)*channels+i%channels]:B(0.f);
    }
}

__global__ void downsample(const B *main,const B *skip,B *out,size_t n,
                           int c,int outC,int h,int w,int sf,int tf) {
    const int factor=tf*sf*sf, group=c*factor/outC;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        int channel=i%outC, row=i/outC, x=(row%(w/sf))*sf, y=(row/(w/sf)%(h/sf))*sf;
        int t=row/((w/sf)*(h/sf))*tf, offset=channel%factor;
        size_t spatial=(size_t(t+offset/(sf*sf))*h+y+offset/sf%sf)*w+x+offset%sf;
        B value=main[spatial*(outC/factor)+channel/factor];
        if(skip) {
            float sum=0;
            for(int j=channel*group;j<(channel+1)*group;++j) {
                int p=j%factor;
                size_t pos=(size_t(t+p/(sf*sf))*h+y+p/sf%sf)*w+x+p%sf;
                sum+=float(skip[pos*c+j/factor]);
            }
            value=B(float(value)+float(B(sum/group)));
        }
        out[i]=value;
    }
}

__global__ void residualNorm(B *values,const B *skip,const B *scale,const B *bias,int rows,int channels) {
    __shared__ float sums[256];
    for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
        B *v=values+size_t(row)*channels;
        float sum=0;
        for(int j=threadIdx.x;j<channels;j+=blockDim.x) sum+=float(B(float(v[j])*float(v[j])));
        sums[threadIdx.x]=sum; __syncthreads();
        for(int k=128;k;k/=2) { if(threadIdx.x<k) sums[threadIdx.x]+=sums[threadIdx.x+k]; __syncthreads(); }
        float denom=float(B(sqrtf(float(B(float(B(sums[0]/channels))+1e-5f)))));
        for(int j=threadIdx.x;j<channels;j+=blockDim.x) {
            B a=B(float(v[j])/denom);
            B b=B(float(a)*float(scale[j]));
            B c=B(float(b)+float(bias[j]));
            v[j]=B(float(c)+float(skip[size_t(row)*channels+j]));
        }
        __syncthreads();
    }
}

Tensor conv(Runtime &rt,Weights &weights,const std::string &prefix,const Volume &v) {
    const auto &filter=weights.get(prefix+".conv.weight");
    if(filter.cols!=27*v.x.cols) throw std::runtime_error("unexpected reference weight shape: "+prefix);
    Tensor reordered(filter.rows,filter.cols);
    reorderFilter<<<grid(filter.size()),256,0,cudaStreamPerThread>>>(ptr(filter),ptr(reordered),filter.size(),v.x.cols);
    checkLaunch();
    auto bias=weights.optional(prefix+".conv.bias");
    Tensor result(v.x.rows,filter.rows);
    const int slab=std::min(v.x.rows,2048);
    Tensor patches(slab,filter.cols);
    for(int first=0;first<v.x.rows;first+=slab) {
        int count=std::min(slab,v.x.rows-first);
        auto p=patches.rowSlice(0,count);
        gatherPatch<<<grid(p.size()),256,0,cudaStreamPerThread>>>(ptr(v.x),ptr(p),p.size(),first,v.x.cols,
                                                                  v.height,v.width,v.time==30?10:40);
        checkLaunch();
        auto target=result.rowSlice(first,count);
        rt.linearInto(p,reordered,target,bias);
    }
    return result;
}

Volume reduce(Runtime &rt,Weights &weights,const Volume &v,const std::string &prefix,
              int spatial,int temporal,bool skip) {
    auto main=conv(rt,weights,prefix,v);
    Volume out{Tensor(v.x.rows/(spatial*spatial*temporal),main.cols*spatial*spatial*temporal),
               v.time/temporal,v.height/spatial,v.width/spatial};
    downsample<<<grid(out.x.size()),256,0,cudaStreamPerThread>>>(ptr(main),skip?ptr(v.x):nullptr,ptr(out.x),
        out.x.size(),v.x.cols,out.x.cols,v.height,v.width,spatial,temporal);
    checkLaunch();
    return out;
}
} // namespace

std::vector<float> encodeReferenceImage(int device,const std::string &path,int width,int height,
                                        const uint8_t *rgb,size_t bytes) {
    if(!validResolution(width,height) || !rgb || bytes!=size_t(width)*height*3)
        throw std::invalid_argument("reference must be a fitted RGB24 frame");
    // The released encoder and its intermediate rounding use BF16 even when
    // the primary DiT is configured for FP32 diagnostics.
    Runtime rt(device,DType::BF16);
    Weights weights(rt,path);
    std::vector<float> frame(bytes);
    for(size_t i=0;i<bytes;++i) frame[i]=float(B(float(B(rgb[i]/127.5f))-1.f));
    auto first=rt.upload(frame.data(),width*height,3);
    Volume v{Tensor(120*width*height,3),120,height,width};
    tileImage<<<grid(v.x.size()),256,0,cudaStreamPerThread>>>(ptr(first),ptr(v.x),v.x.size(),int(bytes));
    checkLaunch();
    v=reduce(rt,weights,v,"encoder.project_in.conv",2,1,false);
    const int depths[]={0,5,10,4,4,4,4};
    for(int stage=1;stage<=6;++stage) {
        auto prefix="encoder.stages."+std::to_string(stage)+".op_list.";
        for(int b=0;b<depths[stage];++b) {
            auto name=prefix+std::to_string(b)+".main.";
            auto branch=conv(rt,weights,name+"conv1",v);
            Volume mid{rt.activation(branch,false),v.time,v.height,v.width};
            auto out=conv(rt,weights,name+"conv2",mid);
            const auto &scale=weights.get(name+"conv2.norm.weight"), &bias=weights.get(name+"conv2.norm.bias");
            residualNorm<<<std::min(out.rows,65535),256,0,cudaStreamPerThread>>>(ptr(out),ptr(v.x),ptr(scale),ptr(bias),out.rows,out.cols);
            checkLaunch();
            v.x=std::move(out);
        }
        if(stage<6)
            v=reduce(rt,weights,v,prefix+std::to_string(depths[stage])+".main.conv",stage==5?1:2,stage==5?4:1,true);
    }
    return rt.download(conv(rt,weights,"encoder.project_out",v));
}
}} // namespace fastllm::nomas
