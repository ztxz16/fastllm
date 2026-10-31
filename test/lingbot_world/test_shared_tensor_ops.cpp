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
    checkResize(vae); checkResize(qwen);
    std::cout<<"shared cast / nearest upsample / Qwen shortcut: bitwise passed\n";
    return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; } }
