// No checkpoints required. Compare the GPU packer with the previous CPU path.
#include "models/hy_nomas/runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fastllm::nomas;

static void require(bool ok,const char *message) {
    if(!ok) throw std::runtime_error(message);
}
static std::vector<uint8_t> reference(const std::vector<float> &pixels,int width,int height,int first,int frames) {
    std::vector<uint8_t> rgb(size_t(frames)*height*width*3);
    for(int f=0;f<frames;++f) for(int y=0;y<height;++y) for(int x=0;x<width;++x) for(int c=0;c<3;++c) {
        int absolute=f+first;
        size_t token=(size_t(absolute/4)*(height/16)+y/16)*(width/16)+x/16;
        size_t channel=((absolute%4*16+y%16)*16+x%16)*3+c;
        float value=pixels[token*3072+channel];
        if(!std::isfinite(value)) throw std::runtime_error("non-finite reference pixel");
        rgb[((size_t(f)*height+y)*width+x)*3+c]=uint8_t(std::nearbyint(std::clamp(value*.5f+.5f,0.f,1.f)*255.f));
    }
    return rgb;
}
template<class F> static void fails(F fn) {
    bool failed=false;
    try { fn(); } catch(const std::exception &) { failed=true; }
    require(failed,"expected rejected input");
}
static std::vector<float> values(size_t size) {
    std::vector<float> patterns;
    // Every finite BF16 bit pattern, including subnormals and signed zero.
    for(uint32_t bits=0;bits<65536;++bits) {
        uint32_t wide=bits<<16; float value;
        std::memcpy(&value,&wide,sizeof(value));
        if(std::isfinite(value)) patterns.push_back(value);
    }
    // FP32 values immediately around every RGB quantization midpoint.
    for(int i=0;i<255;++i) {
        float value=2.f*((float(i)+.5f)/255.f)-1.f;
        patterns.push_back(std::nextafter(value,-INFINITY));
        patterns.push_back(value);
        patterns.push_back(std::nextafter(value,INFINITY));
    }
    std::vector<float> out(size);
    for(size_t i=0;i<size;++i) out[i]=patterns[i%patterns.size()];
    return out;
}
int main() {
    try {
        for(bool fp32:{false,true}) {
            Runtime rt(0,fp32);
            // Single patch, initial segment, sliding history, rectangular and
            // maximum video shapes; also allow a non-aligned selected range.
            const int shapes[][5]={{16,16,4,0,4},{64,96,8,1,6},{384,224,20,0,20},
                                   {384,224,40,20,20},{512,288,40,20,20},{512,512,40,0,40}};
            for(const auto &s:shapes) {
                int width=s[0],height=s[1],frames=s[2],rows=frames/4*(height/16)*(width/16);
                auto host=values(size_t(rows)*3072);
                auto x=rt.upload(host.data(),rows,3072);
                auto expected=reference(rt.download(x),width,height,s[3],s[4]);
                require(rt.downloadPixelRGB(x,width,height,s[3],s[4])==expected,"RGB bytes differ");
            }
            auto host=values(2*3072);
            auto x=rt.upload(host.data(),2,3072);
            fails([&]{rt.downloadPixelRGB(x,15,16,0,4);});
            fails([&]{rt.downloadPixelRGB(x,16,16,-1,4);});
            fails([&]{rt.downloadPixelRGB(x,16,16,0,0);});
            fails([&]{rt.downloadPixelRGB(x,16,16,5,4);});
            fails([&]{rt.downloadPixelRGB(x,48,16,0,4);});
            fails([&]{rt.downloadPixelRGB(Tensor{},16,16,0,4);});
            fails([&]{rt.downloadPixelRGB(rt.upload(host.data(),1,16),16,16,0,4);});
            // Discarded history must not cause errors. Any non-finite value
            // within the selected output must still reject the entire result.
            for(float bad:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
                host[0]=bad; x=rt.upload(host.data(),2,3072);
                require(rt.downloadPixelRGB(x,16,16,4,4)==reference(rt.download(x),16,16,4,4),"discarded history changed output");
                fails([&]{rt.downloadPixelRGB(x,16,16,0,4);});
                host.back()=bad; x=rt.upload(host.data(),2,3072);
                fails([&]{rt.downloadPixelRGB(x,16,16,4,4);});
                host.back()=0;
            }
            std::cout << (fp32?"FP32":"BF16") << ": PASS (byte equality, shapes, rounding, invalid inputs)\n";
        }
    } catch(const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
