#include "models/lingbot_world/pipeline.h"
#include <cstring>
#include <string>

// Validation-only bridge. This target is not part of the production library.
using namespace fastllm::lingbot;
namespace {
thread_local std::string error;
template<class F> int checked(F function) {
    try { function(); error.clear(); return 0; }
    catch(const std::exception &exception) { error=exception.what(); return -1; }
}
}
extern "C" {
const char *lingbot_test_error() { return error.c_str(); }
int lingbot_test_modulation(const float *input,const float *first,const float *second,float *output,
                            int rows,int cols,int modRows,int fullFlags,int mode) {
    return checked([&] {
        Runtime rt(0);
        auto x=rt.cast(rt.fp32.upload(input,rows,cols),fullFlags&1);
        auto a=rt.cast(rt.fp32.upload(first,mode==2?rows:modRows,cols),fullFlags&2);
        auto b=rt.cast(rt.fp32.upload(second,modRows,cols),fullFlags&4);
        auto out=mode==2?rt.gated(x,a,b):rt.affine(x,a,b,mode==1);
        auto values=rt.download(out);
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_camera(const float *poses,const float *intrinsics,float *output,int frames,int height,int width) {
    return checked([&] {
        Runtime rt(0);
        std::vector<Pose> trajectory(frames);
        std::memcpy(trajectory.data(),poses,size_t(frames)*16*sizeof(float));
        Intrinsics k; std::copy(intrinsics,intrinsics+4,k.begin());
        auto values=rt.download(cameraEmbedding(rt,trajectory,k,height,width));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_poses(const float *absolute,const float *previous,float *relative,int frames) {
    return checked([&] {
        std::vector<Pose> poses(frames);
        std::memcpy(poses.data(),absolute,size_t(frames)*16*sizeof(float));
        Pose old;
        if(previous) std::copy(previous,previous+16,old.begin());
        auto values=relativePoses(poses,previous?&old:nullptr);
        std::memcpy(relative,values.data(),values.size()*16*sizeof(float));
    });
}
int lingbot_test_vae_norm(const float *input,const float *weight,float *output,int rows,int cols) {
    return checked([&] {
        VAERuntime rt(0);
        auto x=rt.upload(input,rows,cols),w=rt.upload(weight,1,cols);
        auto values=rt.download(rt.channelNorm(x,w));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_attention(const float *input,float *output,int rows) {
    return checked([&] {
        VAERuntime rt(0);
        auto x=rt.upload(input,rows,1152);
        auto values=rt.download(rt.spatialAttention(rt.columns(x,0,384),rt.columns(x,384,384),rt.columns(x,768,384)));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_conv(const float *input,const float *weight,const float *bias,float *output,
                          int frames,int height,int width,int in,int out,int kt,int kernel,int stride,int pad) {
    return checked([&] {
        VAERuntime rt(0);
        auto x=rt.upload(input,frames*height*width,in),w=rt.upload(weight,out,in*kt*kernel*kernel);
        Tensor b;
        if(bias) b=rt.upload(bias,1,out);
        auto values=rt.download(rt.conv3d(x,w,bias?&b:nullptr,frames,height,width,kt,kernel,stride,pad));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_conv2d(const float *input,const float *weight,const float *bias,float *output,
                            int height,int width,int in,int out,int kernel,int stride,int pad,int padEnd) {
    return checked([&] {
        VAERuntime rt(0);
        auto x=rt.upload(input,height*width,in),w=rt.upload(weight,out,in*kernel*kernel);
        auto b=rt.upload(bias,1,out);
        auto values=rt.download(rt.conv2d(x,w,&b,height,width,kernel,stride,pad,padEnd));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_bf16_norm(const float *input,const float *weight,float *output,int rows,int cols) {
    return checked([&] {
        VAERuntime rt(0,true);
        auto x=rt.convolutionRuntime().upload(input,rows,cols),w=rt.upload(weight,1,cols);
        auto values=rt.download(rt.channelNorm(x,w));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_bf16_attention(const float *input,float *output,int rows) {
    return checked([&] {
        VAERuntime rt(0,true);
        auto x=rt.convolutionRuntime().upload(input,rows,1152);
        auto values=rt.download(rt.spatialAttention(rt.columns(x,0,384),rt.columns(x,384,384),rt.columns(x,768,384)));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_bf16_conv(const float *input,const float *weight,const float *bias,float *output,
                          int frames,int height,int width,int in,int out,int kt,int kernel,int stride,int pad) {
    return checked([&] {
        VAERuntime rt(0,true);
        auto x=rt.upload(input,frames*height*width,in),w=rt.convolutionRuntime().upload(weight,out,in*kt*kernel*kernel);
        Tensor b;
        if(bias) b=rt.convolutionRuntime().upload(bias,1,out);
        auto values=rt.download(rt.conv3d(x,w,bias?&b:nullptr,frames,height,width,kt,kernel,stride,pad));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_vae_bf16_conv2d(const float *input,const float *weight,const float *bias,float *output,
                            int height,int width,int in,int out,int kernel,int stride,int pad,int padEnd) {
    return checked([&] {
        VAERuntime rt(0,true);
        auto x=rt.upload(input,height*width,in),w=rt.convolutionRuntime().upload(weight,out,in*kernel*kernel);
        auto b=rt.convolutionRuntime().upload(bias,1,out);
        auto values=rt.download(rt.conv2d(x,w,&b,height,width,kernel,stride,pad,padEnd));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_norm(const float *input,const float *weight,const float *bias,
                      float *output,int rows,int full,int rms) {
    return checked([&] {
        Runtime rt(0);
        auto x=full?rt.fp32.upload(input,rows,1536):rt.upload(input,rows,1536);
        Tensor w,b;
        if(weight) w=rt.fp32.upload(weight,1,1536);
        if(bias) b=rt.fp32.upload(bias,1,1536);
        auto out=rms?rt.wanRms(x,w):rt.wanLayerNorm(x,weight?&w:nullptr,bias?&b:nullptr);
        auto values=rt.download(out);
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_activation(const float *input,float *output,int rows,int cols,int full,int gelu) {
    return checked([&] {
        Runtime rt(0);
        auto x=full?rt.fp32.upload(input,rows,cols):rt.upload(input,rows,cols);
        auto values=rt.download(rt.activation(x,gelu));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_rope(const float *input,float *output,int frames,int height,int width,int offset,int full) {
    return checked([&] {
        Runtime rt(0);
        auto x=full?rt.fp32.upload(input,frames*height*width,1536):rt.upload(input,frames*height*width,1536);
        rt.rope3d(x,frames,height,width,offset+1);
        rt.rope3d(x,frames,height,width,offset);
        auto result=rt.download(rt.rope3d(x,frames,height,width,offset));
        std::memcpy(output,result.data(),result.size()*sizeof(float));
    });
}
int lingbot_test_attention(const float *queries,const float *keys,const float *values,
                           float *output,int rows,int count) {
    return checked([&] {
        Runtime rt(0);
        auto q=rt.upload(queries,rows,1536),k=rt.upload(keys,count,1536),v=rt.upload(values,count,1536);
        auto result=rt.download(rt.wanAttention(q,k,v));
        std::memcpy(output,result.data(),result.size()*sizeof(float));
    });
}
int lingbot_test_patch(const float *input,const float *weight,const float *bias,
                       float *output,int frames,int height,int width) {
    return checked([&] {
        Runtime rt(0);
        auto x=rt.upload(input,frames*height*width,36),w=rt.upload(weight,1536,144),b=rt.upload(bias,1,1536);
        auto result=rt.download(rt.patchConv2d(x,w,b,frames,height,width));
        std::memcpy(output,result.data(),result.size()*sizeof(float));
    });
}
int lingbot_test_linear(const float *input,const float *weight,const float *bias,
                        float *output,int rows,int in,int out,int full) {
    return checked([&] {
        Runtime rt(0);
        auto &compute=full?rt.fp32:static_cast<fastllm::nomas::Runtime &>(rt);
        auto x=compute.upload(input,rows,in),w=compute.upload(weight,out,in);
        Tensor b;
        if(bias) b=compute.upload(bias,1,out);
        auto values=compute.download(full && bias?rt.wanLinearFP32(x,w,b):compute.linear(x,w,bias?&b:nullptr));
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
int lingbot_test_image(const unsigned char *rgba,int sourceHeight,int sourceWidth,
                       float *output,int height,int width) {
    return checked([&] {
        fastllm::qwen_image::Image image;
        image.width=sourceWidth; image.height=sourceHeight;
        image.rgba.assign(rgba,rgba+size_t(sourceHeight)*sourceWidth*4);
        auto values=prepareImage(image,height,width);
        std::memcpy(output,values.data(),values.size()*sizeof(float));
    });
}
}
