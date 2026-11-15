// Real-checkpoint TP regression: two references, pressure on the peer card,
// resident recovery and cancellation. Run with exactly two permitted GPUs.
#include "models/qwen_image/runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using namespace fastllm::qwen_image;
namespace {
void check(cudaError_t code) { if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
size_t freeMemory(int device) {
    check(cudaSetDevice(device)); check(cudaDeviceSynchronize());
    size_t free,total; check(cudaMemGetInfo(&free,&total)); return free;
}
Image read(const char *path) {
    std::ifstream f(path,std::ios::binary); require(bool(f),"cannot open reference PNG");
    return decodePNG(std::string(std::istreambuf_iterator<char>(f),{}));
}
double relativeL2(const std::vector<float> &a,const std::vector<float> &b) {
    require(a.size()==b.size(),"trace shape mismatch"); double error=0,norm=0;
    for (size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite TP trace");
        error+=std::pow(double(a[i])-b[i],2); norm+=double(a[i])*a[i];
    }
    return std::sqrt(error/std::max(norm,1e-30));
}
}
int main(int argc,char **argv) {
    try {
        require(argc==4,"usage: qwen_image_pipeline_parallel_test MODEL REFERENCE1.png REFERENCE2.png");
        int count=0;check(cudaGetDeviceCount(&count));require(count==2,"expose exactly two permitted GPUs");
        GenerateOptions o; o.width=128;o.height=96;o.referenceResolution=288;o.seed=1234;
        o.prompt="Combine the two references into one still life.";o.images={read(argv[2]),read(argv[3])};
        std::map<int,std::map<std::string,std::vector<float>>> expected;
        {
            Pipeline single(argv[1],0);
            for (int steps : {4,40}) {
                o.steps=steps;
                single.generate(o,{},[&](const std::string &name,const std::vector<float> &values) {
                    expected[steps][name]=values;
                });
            }
        }
        Pipeline parallel(argv[1],std::vector<int>{0,1});
        double maximum=0;
        for (int steps : {4,40}) {
            o.steps=steps;size_t seen=0;double error=0;
            parallel.generate(o,{},[&](const std::string &name,const std::vector<float> &values) {
                const auto &ref=expected.at(steps).at(name);++seen;
                auto e=relativeL2(ref,values);error=std::max(error,e);
                if(name=="prompt" || name=="initial_latent" || name.find("condition.")==0)
                    require(ref==values,"TP changed conditioning or random noise");
            });
            require(seen==expected.at(steps).size(),"missing TP trace");maximum=std::max(maximum,error);
            std::cout<<"multi steps="<<steps<<" snapshots="<<seen<<" max_relative_l2="<<error<<std::endl;
        }
        expected.clear();o.images.clear();o.prompt="A red apple on a wooden table.";o.steps=4;
        const auto image=encodePNG(parallel.generate(o));
        size_t resident[2]={freeMemory(0),freeMemory(1)};
        void *pressure=nullptr;
        // Only the secondary card is constrained. The consensus must unload
        // the root's weights as well, without changing generated output.
        require(resident[1]>(size_t(2)<<30),"insufficient free peer memory for pressure test");
        check(cudaMalloc(&pressure,resident[1]-(size_t(1)<<30)));
        try {
            require(encodePNG(parallel.generate(o))==image,"staged TP output changed");
            size_t rootFree=freeMemory(0),peerFree=freeMemory(1);
            std::cout<<"root_reclaimed_bytes="<<rootFree-resident[0]<<" peer_free_bytes="<<peerFree<<std::endl;
            require(rootFree>resident[0]+(size_t(16)<<30) && peerFree>(size_t(2)<<30),
                    "peer pressure did not unload all rank weights");
        } catch (...) { cudaSetDevice(1);cudaFree(pressure);throw; }
        check(cudaSetDevice(1));check(cudaFree(pressure));
        require(encodePNG(parallel.generate(o))==image,"resident recovery changed TP output");
        bool cancelled=false;
        try {
            parallel.generate(o,[](const std::string &stage,int step,int) {
                if(stage=="denoising" && step==1)throw std::runtime_error("cancel-test");
            });
        } catch (const std::exception &e) { if(std::string(e.what())!="cancel-test")throw;cancelled=true; }
        require(cancelled && freeMemory(0)>resident[0]+(size_t(16)<<30) &&
                freeMemory(1)>resident[1]+(size_t(1)<<30),"TP cancellation leaked rank weights");
        require(encodePNG(parallel.generate(o))==image,"cancellation recovery changed TP output");
        std::cout<<"PASS peer pressure, resident recovery and cancellation; maximum TP L2="<<maximum<<std::endl;
        require(maximum<=.05,"TP multireference accuracy exceeds unchanged 5% threshold");
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
