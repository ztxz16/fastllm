#include "models/lingbot_world/runtime.h"
#include "models/lingbot_world/vae.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using fastllm::nomas::Tensor;
namespace {
uint32_t bits(float value) {
    uint32_t result; std::memcpy(&result,&value,sizeof(result)); return result;
}
float rounded(float value) {
    uint32_t word=bits(value);
    word=(word+0x7fff+((word>>16)&1))&0xffff0000;
    std::memcpy(&value,&word,sizeof(word)); return value;
}
void equal(const std::vector<float> &actual,const std::vector<float> &expected) {
    if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))
        throw std::runtime_error("shared tensor operation differs from reference");
}
void checkCast(fastllm::nomas::Runtime &rt) {
    // Adjacent BF16 ties with even/odd mantissas, signed zero and finite extremes.
    const float edges[]={0.f,-0.f,1.f,1.00390625f,1.01171875f,-1.00390625f,
        std::numeric_limits<float>::min(),std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity()};
    for(int count:{1,7,257,1048833}) {
        std::vector<float> input(count),expected(count);
        for(int i=0;i<count;++i) { input[i]=edges[i%10]; expected[i]=rounded(input[i]); }
        fastllm::nomas::Runtime upload(0,true);
        auto x=upload.upload(input.data(),count,1),alias=rt.cast(x,true);
        if(alias.data!=x.data || alias.storage!=x.storage) throw std::runtime_error("same precision must alias");
        auto low=rt.cast(x,false),same=rt.cast(low,false);
        if(same.data!=low.data || same.storage!=low.storage) throw std::runtime_error("BF16 must alias");
        auto high=rt.cast(low,true);
        equal(rt.download(low),expected); equal(rt.download(high),expected);
        if(count>1) equal(rt.download(rt.cast(low.rowSlice(1,count-1),true)),
                          std::vector<float>(expected.begin()+1,expected.end()));
    }
}
void checkDTypes(fastllm::lingbot::Runtime &rt,fastllm::lingbot::VAERuntime &vae) {
    using fastllm::nomas::DType;
    // Layout-only operations preserve the dtype, including FP16's raw bits.
    const std::vector<float> values{1,2,3,4,5,6,7,8};
    for(auto dtype:{DType::FP32,DType::BF16,DType::FP16}) {
        auto x=rt.cast(rt.upload(values.data(),4,2),dtype);
        auto joined=rt.join(x,x),patched=rt.patch(x,1,2,2),shuffled=vae.temporalShuffle(x,1,2,2);
        // Encoder patches use [channel,dy,dx]; the output head uses [dy,dx,channel].
        auto head=rt.cast(rt.upload(values.data(),1,8),dtype),unpacked=rt.patch(head,1,2,2,true);
        if(joined.dtype!=dtype || patched.dtype!=dtype || unpacked.dtype!=dtype || shuffled.dtype!=dtype)
            throw std::runtime_error("layout operation changed dtype");
        equal(rt.download(joined),{1,2,1,2,3,4,3,4,5,6,5,6,7,8,7,8});
        equal(rt.download(patched),{1,3,5,7,2,4,6,8});
        equal(rt.download(unpacked),values);
        equal(rt.download(shuffled),{1,3,5,7,2,4,6,8});
    }
    auto make=[&](int rows,int cols,DType dtype) {
        std::vector<float> values(size_t(rows)*cols,.5f);
        return rt.cast(rt.upload(values.data(),rows,cols),dtype);
    };
    auto rejects=[](auto fn) {
        try { fn(); } catch(const std::invalid_argument &) { return; }
        throw std::runtime_error("FP16 was interpreted as BF16");
    };
    auto h=make(1,4096,DType::FP16),b=rt.cast(h,false);
    rejects([&]{rt.t5Gelu(h);});
    rejects([&]{rt.t5Norm(h,b);});
    rejects([&]{rt.t5Norm(b,h);});
    rejects([&]{rt.relativeAttention(h,b,b,make(32,64,DType::BF16),1);});
    h=make(1,1536,DType::FP16); b=rt.cast(h,false);
    rejects([&]{rt.wanRms(h,rt.cast(h,true));});
    rejects([&]{rt.wanLayerNorm(h);});
    rejects([&]{rt.wanAttention(h,b,b);});
    rejects([&]{rt.rope3d(h,1,1,1,0);});
    h=make(4,1,DType::FP16); b=rt.cast(h,false);
    rejects([&]{rt.patchConv2d(h,make(1,4,DType::BF16),make(1,1,DType::BF16),1,2,2);});
    rejects([&]{vae.conv2d(h,make(1,9,DType::BF16),nullptr,2,2,3);});
    rejects([&]{vae.conv2d(b,make(1,9,DType::FP16),nullptr,2,2,3);});
    rejects([&]{vae.channelNormForConv(h,make(1,1,DType::FP32),true);});
    h=make(1,384,DType::FP16); b=rt.cast(h,false);
    rejects([&]{vae.spatialAttention(h,b,b);});
    auto low=make(4,2,DType::BF16),half=rt.cast(low,DType::FP16);
    rejects([&]{rt.join(low,half);});
    // FP16 modulation uses the existing FP32 cast path, never the BF16 kernel.
    equal(rt.download(rt.affine(rt.cast(low,true),half,half)),std::vector<float>(8,1.25f));
}
void checkResize(fastllm::qwen_image::Runtime &rt) {
    for(bool full:{false,true}) for(int h:{1,7,16}) for(int w:{1,5,32}) for(int channels:{3,96}) {
        std::vector<float> input(h*w*channels),expected(h*w*channels*4);
        for(size_t i=0;i<input.size();++i) input[i]=(int(i%113)-56)*.03125f;
        auto x=rt.cast(rt.upload(input.data(),h*w,channels),full);
        for(int y=0;y<h;++y) for(int z=0;z<w;++z) for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            for(int c=0;c<channels;++c)
                expected[((y*2+dy)*w*2+z*2+dx)*channels+c]=input[(y*w+z)*channels+c];
        equal(rt.download(rt.resize2x(x,h,w)),expected);
    }
    // Qwen-Image's temporal/spatial shortcut shares the same kernel template.
    for(int temporal:{1,2}) {
        std::vector<float> input(8*2*3);
        for(size_t i=0;i<input.size();++i) input[i]=float(i);
        auto x=rt.upload(input.data(),2*3,8);
        auto full=rt.duplicateUpsample(x,2,3,4,temporal);
        auto low=rt.duplicateUpsample(rt.cast(x,false),2,3,4,temporal);
        equal(rt.download(low),rt.download(full));
    }
}
}
int main() { try {
    fastllm::nomas::Runtime common(0,false);
    fastllm::lingbot::Runtime dit(0);
    fastllm::lingbot::VAERuntime vae(0,true);
    fastllm::qwen_image::Runtime qwen(0);
    checkCast(common); checkCast(dit); checkCast(vae);
    checkResize(vae); checkResize(qwen); checkDTypes(dit,vae);
    std::cout<<"shared cast / nearest upsample / Qwen shortcut / dtype boundaries: bitwise passed\n";
    return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
