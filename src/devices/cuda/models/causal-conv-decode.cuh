#pragma once
namespace fastllm_causal_conv {
// Four-tap depthwise decode. One vector load of history and weights per
// channel; history input/output may alias. The last tap reads the new token.
template<typename T,int THREADS>
__global__ __launch_bounds__(THREADS) void Decode4(
        const T *input,const float4 *weight,const float4 *history,
        float4 *next,float *output,size_t total,int channels,bool silu) {
    const size_t item=size_t(blockIdx.x)*THREADS+threadIdx.x;
    if(item>=total)return;
    const float4 old=history[item],w=weight[item%channels];
    const float4 h=make_float4(old.y,old.z,old.w,float(input[item]));
    float value=0.f;
    value=fmaf(h.x,w.x,value);value=fmaf(h.y,w.y,value);
    value=fmaf(h.z,w.z,value);value=fmaf(h.w,w.w,value);
    next[item]=h;
    if(silu)value=value/(1.f+expf(-value));
    output[item]=value;
}
}
