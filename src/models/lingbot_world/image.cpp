#include "models/lingbot_world/pipeline.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
float cubic(float x) {
    constexpr float a=-.75f;
    x=std::abs(x);
    return x<=1.f?((a+2.f)*x-(a+3.f))*x*x+1.f:((a*x-5.f*a)*x+8.f*a)*x-4.f*a;
}
}
std::vector<float> prepareImage(const qwen_image::Image &image,int h,int w) {
    if(image.width<1 || image.height<1 || image.rgba.size()!=size_t(image.width)*image.height*4 || h<1 || w<1 || h>832 || w>832)
        throw std::invalid_argument("invalid LingBot source image");
    std::vector<float> input(size_t(image.width)*image.height*3),out(size_t(h)*w*3);
    for(size_t p=0;p<input.size()/3;++p) for(int c=0;c<3;++c)
        input[p*3+c]=(float(image.rgba[p*4+c])/255.f-.5f)/.5f;
    if(image.width==w && image.height==h) return input;
    // Match align_corners=False, antialias=False bicubic interpolation on
    // normalized FP32 RGB. Do not quantize or clamp bicubic overshoot.
    float sx=float(image.width)/w,sy=float(image.height)/h;
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        float ix=sx*(float(x)+.5f)-.5f,iy=sy*(float(y)+.5f)-.5f;
        int left=int(std::floor(ix)),top=int(std::floor(iy));
        float wx[4],wy[4];
        for(int j=0;j<4;++j) { wx[j]=cubic(ix-left-(j-1)); wy[j]=cubic(iy-top-(j-1)); }
        for(int c=0;c<3;++c) {
            float value=0;
            for(int dy=0;dy<4;++dy) {
                float row=0;
                for(int dx=0;dx<4;++dx) {
                    int xx=std::clamp(left+dx-1,0,image.width-1),yy=std::clamp(top+dy-1,0,image.height-1);
                    row+=input[(size_t(yy)*image.width+xx)*3+c]*wx[dx];
                }
                value+=row*wy[dy];
            }
            out[(size_t(y)*w+x)*3+c]=value;
        }
    }
    return out;
}
std::vector<uint8_t> encodeFramePNG(const Segment &segment,int frame) {
    if(segment.height<=0 || segment.width<=0 || frame<0 || frame>=segment.frames ||
       segment.pixels.size()!=size_t(segment.frames)*segment.height*segment.width*3)
        throw std::invalid_argument("invalid LingBot output frame");
    qwen_image::Image image;
    image.width=segment.width; image.height=segment.height;
    size_t pixels=size_t(image.width)*image.height;
    image.rgba.resize(pixels*4);
    for(size_t i=0;i<pixels;++i) {
        for(int c=0;c<3;++c) image.rgba[i*4+c]=uint8_t(std::nearbyint(
            std::clamp(segment.pixels[(size_t(frame)*pixels+i)*3+c]*.5f+.5f,0.f,1.f)*255.f));
        image.rgba[i*4+3]=255;
    }
    return qwen_image::encodePNG(image);
}
}} // namespace fastllm::lingbot
