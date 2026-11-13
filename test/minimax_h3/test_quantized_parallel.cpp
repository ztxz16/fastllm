// Explicit real-checkpoint test: fixed packed text/audio/video rows, uneven SP
// shards and Pika adapters. Restore each oracle activation to measure all 50
// blocks on identical inputs, without mixing in cumulative quantization drift.
#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3/model_source.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3;
using fastllm::nomas::ParallelGroup;
using fastllm::nomas::ParallelContext;
static double l2(const std::vector<float> &a,const std::vector<float> &b) {
    if(a.size()!=b.size()) throw std::runtime_error("shape mismatch");
    double e=0,n=0;
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite output");
        e+=std::pow(double(a[i])-b[i],2); n+=double(a[i])*a[i];
    }
    return std::sqrt(e/std::max(n,1e-30));
}
int main(int argc,char **argv) {
    try {
        if(argc!=5 && argc!=6) throw std::invalid_argument("usage: minimax_h3_quantized_parallel VIEW PIKA_OR_DASH fl2va|ref2va tp|sp [256x256_124frame_TRACE]");
        int count=0; cudaGetDeviceCount(&count);
        if(count!=2) throw std::runtime_error("expose exactly two permitted GPUs");
        const bool reference=std::string(argv[3])=="ref2va",sp=std::string(argv[4])=="sp";
        if(std::string(argv[3])!="fl2va" && !reference) throw std::invalid_argument("invalid mode");
        if(std::string(argv[4])!="tp" && !sp) throw std::invalid_argument("invalid parallel mode");
        const auto source=resolveModelSource(argv[1]);
        const auto model=source.baseDirectory+(reference?"/transformer_ref":"/transformer");
        const auto weights=reference?source.referenceWeights:source.transformerWeights;
        const auto adapter=std::string(argv[2])+"/"+argv[3];
        const bool pika=std::string(argv[2])!="-";
        std::mt19937 rng(42); std::normal_distribution<float> normal(0,.4);
        auto values=[&](size_t n) { std::vector<float> x(n); for(auto &v:x) v=normal(rng); return x; };
        int nt=7,na=17,nv=43,rows=nt+na+nv;
        auto text=values(nt*5120),audio=values(na*32),video=values(nv*96);
        PackedLayout layout;
        for(int i=0;i<rows;++i) {
            const int tag=i<nt?1:i<nt+na?2:0;
            layout.tags.push_back(tag); layout.timeIndices.push_back(0);
            (tag==1?layout.text:tag==2?layout.audio:layout.video).push_back(i);
            for(int d=0;d<3;++d) layout.positions.push_back(float((i*3+d)%41));
        }
        if(argc==6) {
            auto read=[&](const char *name) {
                std::ifstream file(std::string(argv[5])+"/"+name+".f32",std::ios::binary|std::ios::ate);
                if(!file || file.tellg()<=0 || size_t(file.tellg())%4) throw std::runtime_error("invalid trace fixture");
                std::vector<float> data(size_t(file.tellg())/4); file.seekg(0);
                file.read(reinterpret_cast<char *>(data.data()),data.size()*4); return data;
            };
            text=read("text"); audio=read("initial_audio"); video=read("initial_video");
            nt=text.size()/5120; na=audio.size()/32; nv=video.size()/96;
            layout=keyframeLayout(std::vector<int>(nt,1),37,16,16,207,{});
            rows=nt+na+nv;
            if(layout.tags.size()!=size_t(rows)) throw std::runtime_error("trace fixture geometry mismatch");
        }
        std::vector<std::vector<float>> plans{{0.f}};
        std::vector<float> expectedVideo,expectedAudio,expectedMod;
        std::vector<std::vector<float>> blocks;
        {
            Runtime rt(0,false),full(0,true); Transformer single(rt,full,model);
            single.setQuantizedWeights(weights);
            if(pika) single.loadPikaAdapter(adapter,reference);
            auto modInput=rt.cast(full.activation(single.timeEmbedding({0.f})),rt.dtype());
            expectedMod=rt.download(single.projectModulation(0,modInput));
            single.prepare(plans);
            const auto result=single.forward(full.upload(video.data(),nv,96),full.upload(audio.data(),na,32),rt.upload(text.data(),nt,5120),layout,
                [&](const std::string &,const Tensor &x){blocks.push_back(rt.download(x));});
            expectedVideo=rt.download(result.first); expectedAudio=rt.download(result.second);
        }
        ParallelGroup group({0,1});
        std::vector<std::unique_ptr<Runtime>> rt(2),full(2);
        std::vector<std::unique_ptr<Transformer>> dit(2);
        double maximum=0;
        group.run([&](int rank){
            rt[rank]=std::make_unique<Runtime>(rank,false); full[rank]=std::make_unique<Runtime>(rank,true);
            dit[rank]=std::make_unique<Transformer>(*rt[rank],*full[rank],model,ParallelContext{&group,rank},sp);
            dit[rank]->setQuantizedWeights(weights);
            if(pika) dit[rank]->loadPikaAdapter(adapter,reference);
            // Bounded host cache and streaming exercise reload after each layer.
            dit[rank]->setHostMemoryBudget(std::make_shared<fastllm::nomas::HostMemoryBudget>(0));
            dit[rank]->setLowMemory(true);
            auto modInput=rt[rank]->cast(full[rank]->activation(dit[rank]->timeEmbedding({0.f})),rt[rank]->dtype());
            auto mod=dit[rank]->projectModulation(0,modInput);
            if(rank==0) std::cout<<"mod_l2="<<l2(expectedMod,rt[rank]->download(mod))<<std::endl;
            dit[rank]->prepare(plans);
        });
        std::vector<float> actualVideo[2],actualAudio[2];
        group.run([&](int rank){
            auto &r=*rt[rank]; auto &f=*full[rank]; size_t layer=0;
            const auto result=dit[rank]->forward(f.upload(video.data(),nv,96),f.upload(audio.data(),na,32),r.upload(text.data(),nt,5120),layout,
                [&](const std::string &,const Tensor &x){
                    if(rank==0) { const auto error=l2(blocks.at(layer),r.download(x)); maximum=std::max(maximum,error); std::cout<<"block="<<layer<<" relative_l2="<<error<<std::endl; }
                    {
                        auto oracle=r.upload(blocks.at(layer).data(),x.rows,x.cols);
                        auto status=cudaMemcpyAsync(x.data,oracle.data,x.bytes(),cudaMemcpyDeviceToDevice,cudaStreamPerThread);
                        if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
                    }
                    ++layer;
                });
            actualVideo[rank]=r.download(result.first); actualAudio[rank]=r.download(result.second);
        });
        if(actualVideo[0]!=actualVideo[1] || actualAudio[0]!=actualAudio[1]) throw std::runtime_error("rank outputs differ");
        const double ve=l2(expectedVideo,actualVideo[0]),ae=l2(expectedAudio,actualAudio[0]);
        std::cout<<argv[3]<<" "<<argv[4]<<" pika="<<pika<<" fixed_block_inputs=1"<<" block_max_l2="<<maximum<<" video_l2="<<ve<<" audio_l2="<<ae<<std::endl;
        if(sp && source.transformerPrecision=="mixed") {
            // 8193 rows produce 4097/4096 token shards: a naive 4096-row
            // loop would issue two calibration collectives on only one rank.
            PackedLayout tail;
            constexpr int total=8193;
            for(int i=0;i<total;++i) {
                const int tag=i<nt?1:i<nt+na?2:0;
                tail.tags.push_back(tag); tail.timeIndices.push_back(0);
                (tag==1?tail.text:tag==2?tail.audio:tail.video).push_back(i);
                for(int d=0;d<3;++d) tail.positions.push_back(float((i*3+d)%41));
            }
            auto tailVideo=values(size_t(total-nt-na)*96);
            struct StopAfterCalibration {};
            group.run([&](int rank) {
                auto &r=*rt[rank];auto &f=*full[rank];bool reached=false;
                try {
                    dit[rank]->forward(f.upload(tailVideo.data(),total-nt-na,96),f.upload(audio.data(),na,32),
                        r.upload(text.data(),nt,5120),tail,[&](const std::string &name,const Tensor &) {
                            if(name=="block.2") {reached=true;throw StopAfterCalibration{};}
                        });
                } catch(const StopAfterCalibration &) {}
                if(!reached) throw std::runtime_error("NVFP4 uneven FFN tail was not exercised");
                dit[rank]->unload();
            });
            std::cout<<"PASS NVFP4 SP 4097/4096 token shards and calibration collectives"<<std::endl;
        }
        group.run([&](int rank){dit[rank].reset(); full[rank].reset(); rt[rank].reset();});
        if(maximum>.05 || ve>.01 || ae>.01) throw std::runtime_error("quantized parallel L2 exceeds 0.05 per block or 0.01 for output heads");
    } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
