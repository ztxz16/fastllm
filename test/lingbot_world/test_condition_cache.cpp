#include "models/lingbot_world/vae.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm::lingbot;
int main(int argc,char **argv) { try {
    if(argc<2) throw std::invalid_argument("usage: test_condition_cache checkpoint [height width]");
    int h=argc>2?std::stoi(argv[2]):64,w=argc>3?std::stoi(argv[3]):96;
    VAERuntime rt(0); VAE vae(rt,argv[1]);
    for(int stream=0;stream<2;++stream) {
        VAEState reference,optimized;
        std::vector<float> pixels(size_t(h)*w*3);
        for(size_t i=0;i<pixels.size();++i) pixels[i]=std::sin(float(i)*.013f+stream);
        auto image=rt.upload(pixels.data(),h*w,3);
        int processed=0;
        for(int step=0;processed<153;++step) {
            int count=stream?1+step%5:3;
            int frames=step?count*4:1+(count-1)*4;
            std::vector<float> input(size_t(frames)*h*w*3,0.f);
            if(!step) std::copy(pixels.begin(),pixels.end(),input.begin());
            auto x=rt.upload(input.data(),frames*h*w,3);
            auto expected=rt.download(vae.encode(x,frames,h,w,reference));
            auto saved=optimized;
            auto actual=rt.download(vae.encodeCondition(step?fastllm::nomas::Tensor{}:image,count,h,w,optimized));
            if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))
                throw std::runtime_error("condition differs at stream="+std::to_string(stream)+" frame="+std::to_string(processed));
            for(float v:actual) if(!std::isfinite(v)) throw std::runtime_error("nonfinite condition");
            if(step==4 || processed>=112) {
                auto replay=rt.download(vae.encodeCondition(fastllm::nomas::Tensor{},count,h,w,saved));
                if(replay.size()!=actual.size() || std::memcmp(replay.data(),actual.data(),actual.size()*sizeof(float)))
                    throw std::runtime_error("condition snapshot replay differs");
            }
            processed+=frames;
        }
        std::cout<<"condition stream "<<stream<<": "<<processed<<" frames, bitwise exact\n";
    }
    std::cout<<"condition cache and snapshot replay: passed\n";
} catch(const std::exception &e) { std::cerr<<e.what()<<std::endl; return 1; } }
