#include "models/lingbot_world/pipeline.h"
#include "json11.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace fastllm::lingbot;
using Json=json11::Json;
namespace {
std::string read(const std::string &path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot read "+path);
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
void write(const std::string &path,const void *data,size_t size) {
    std::ofstream f(path,std::ios::binary);
    f.write(static_cast<const char *>(data),size);
    if(!f) throw std::runtime_error("cannot write "+path);
}
}
int main(int argc,char **argv) {
    try {
        std::string directory,input,inputFloat,prompt,output="lingbot-output",traceDir,actions="stay",intrinsics;
        int width=832,height=480,device=0; uint64_t seed=42; bool bf16Decode=true;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            if(i+1>=argc) throw std::invalid_argument("missing value for "+key);
            std::string value=argv[++i];
            if(key=="--model") directory=value;
            else if(key=="--image") input=value;
            else if(key=="--input-f32") inputFloat=value;
            else if(key=="--prompt") prompt=value;
            else if(key=="--prompt-file") prompt=read(value);
            else if(key=="--output") output=value;
            else if(key=="--trace") traceDir=value;
            else if(key=="--width") width=std::stoi(value);
            else if(key=="--height") height=std::stoi(value);
            else if(key=="--device") device=std::stoi(value);
            else if(key=="--vae-decode-precision") {
                if(value!="bf16" && value!="fp32") throw std::invalid_argument("VAE precision must be bf16 or fp32");
                bf16Decode=value=="bf16";
            }
            else if(key=="--seed") seed=std::stoull(value);
            else if(key=="--actions") actions=value;
            else if(key=="--intrinsics") intrinsics=value;
            else throw std::invalid_argument("unknown argument "+key);
        }
        if(directory.empty() || (input.empty()==inputFloat.empty()))
            throw std::invalid_argument("--model and exactly one of --image PNG / --input-f32 are required");
        if(width<64 || height<64 || width>832 || height>832 || width%16 || height%16)
            throw std::invalid_argument("width/height must be 64..832, multiples of 16");
        std::vector<float> pixels(size_t(width)*height*3);
        if(!inputFloat.empty()) {
            auto bytes=read(inputFloat);
            if(bytes.size()!=pixels.size()*4) throw std::invalid_argument("input float image size mismatch");
            std::memcpy(pixels.data(),bytes.data(),bytes.size());
        } else {
            pixels=prepareImage(fastllm::qwen_image::decodePNG(read(input)),height,width);
        }
        Intrinsics k{};
        if(!intrinsics.empty()) {
            std::replace(intrinsics.begin(),intrinsics.end(),',',' ');
            std::istringstream s(intrinsics);
            for(float &v:k) if(!(s>>v)) throw std::invalid_argument("intrinsics need fx,fy,cx,cy");
        }
        std::vector<std::string> commands;
        std::istringstream actionStream(actions); std::string command;
        while(std::getline(actionStream,command,',')) { movePose(identityPose(),command,0,0); commands.push_back(command); }
        if(commands.empty()) throw std::invalid_argument("at least one action required");
        std::filesystem::create_directories(output);
        if(!traceDir.empty()) std::filesystem::create_directories(traceDir);
        Pipeline pipeline(directory,device,bf16Decode);
        auto progress=[](const std::string &stage,int n,int total) { std::cerr<<stage<<" "<<n<<"/"<<total<<std::endl; };
        auto world=pipeline.create(prompt,pixels,height,width,seed,intrinsics.empty()?nullptr:&k,progress);
        Json::array schedule,commandJson,kJson;
        for(auto step:fastSchedule()) schedule.push_back(Json::object{{"timestep",step.timestep},{"flow_sigma",step.flowSigma},{"noise_sigma",step.noiseSigma}});
        for(const auto &a:commands) commandJson.push_back(a);
        for(float v:world->intrinsics) kJson.push_back(v);
        auto metadata=Json(Json::object{{"prompt",prompt},{"width",width},{"height",height},{"seed",double(seed)},
            {"vae_decode_precision",bf16Decode?"bf16":"fp32"},{"fps",16},{"actions",commandJson},{"intrinsics",kJson},{"schedule",schedule}}).dump();
        write(output+"/metadata.json",metadata.data(),metadata.size());
        if(!traceDir.empty()) {
            write(traceDir+"/metadata.json",metadata.data(),metadata.size());
            write(traceDir+"/first_image.f32",pixels.data(),pixels.size()*4);
        }
        int frameNumber=0;
        for(size_t segment=0;segment<commands.size();++segment) {
            Trace trace;
            if(!traceDir.empty()) trace=[&](const std::string &name,const Tensor &x) {
                auto values=pipeline.runtime().download(x);
                write(traceDir+"/"+std::to_string(segment)+"."+name+".f32",values.data(),values.size()*4);
            };
            auto result=pipeline.step(*world,commands[segment],nullptr,progress,trace);
            for(int f=0;f<result.frames;++f) {
                auto png=encodeFramePNG(result,f);
                auto path=output+"/frame_"+std::to_string(frameNumber++)+".png";
                write(path,png.data(),png.size());
            }
        }
        std::cout<<"Generated "<<frameNumber<<" frames in "<<output<<std::endl;
    } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl; return 1; }
}
