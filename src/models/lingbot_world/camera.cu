#include "models/lingbot_world/pipeline.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
__global__ void rays(const float *poses,__nv_bfloat16 *out,size_t n,int h,int w,float fx,float fy,float cx,float cy) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256) {
        int channel=i%384/64,dy=i%64/8,dx=i%8;
        size_t pixel=i/384;
        int px=pixel%(w/8)*8+dx,py=pixel/(w/8)%(h/8)*8+dy;
        const float *p=poses+(pixel/((h/8)*(w/8)))*16;
        float v;
        if(channel<3) v=p[channel*4+3];
        else {
            float x=(float(px)+.5f-cx)/fx,y=(float(py)+.5f-cy)/fy;
            // The three-component reference norm reduces lanes 0+2, then 1.
            // Preserve this order before the final BF16 rounding.
            float norm=sqrtf((x*x+1.f)+y*y);
            x/=norm; y/=norm; float z=1.f/norm;
            int row=(channel-3)*4;
            v=p[row]*x+p[row+1]*y+p[row+2]*z;
        }
        // Upstream names this Plucker embedding, but its implementation packs
        // ray origin + direction (not origin cross direction).
        out[i]=__nv_bfloat16(v);
    }
}
}
Tensor cameraEmbedding(Runtime &rt,const std::vector<Pose> &poses,const Intrinsics &k,int h,int w) {
    if(poses.empty() || h<8 || w<8 || h%8 || w%8 || !(k[0]>0) || !(k[1]>0))
        throw std::invalid_argument("invalid LingBot camera shape or intrinsics");
    for(float value:k) if(!std::isfinite(value)) throw std::invalid_argument("nonfinite camera intrinsics");
    for(const auto &p:poses) for(float value:p) if(!std::isfinite(value)) throw std::invalid_argument("nonfinite camera pose");
    auto p=rt.fp32.upload(poses[0].data(),poses.size(),16);
    Tensor out(poses.size()*(h/8)*(w/8),384);
    int blocks=int(std::min<size_t>((out.size()+255)/256,4096));
    rays<<<blocks,256,0,cudaStreamPerThread>>>(reinterpret_cast<const float *>(p.data),
        reinterpret_cast<__nv_bfloat16 *>(out.data),out.size(),h,w,k[0],k[1],k[2],k[3]);
    auto error=cudaGetLastError(); if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return out;
}
}} // namespace fastllm::lingbot
