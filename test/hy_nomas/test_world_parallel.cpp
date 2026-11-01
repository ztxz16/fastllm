#include "models/hy_nomas/parallel.h"
#include <iostream>
#include <stdexcept>
using namespace fastllm::nomas;
int main(int argc,char **argv) {try {
    if(argc!=4) throw std::invalid_argument("usage: test_world_parallel model text_encoder devices");
    ParallelOptions options{parseParallelDevices(argv[3]),RowReduction::Activation};
    Pipeline pipeline(argv[1],argv[2],options);
    for(int width:{64,256}) {
        auto world=pipeline.createWorld("A room with a wooden table and a window.",width,width,42,8);
        pipeline.stepWorld(*world,0);
        auto saved=*world;
        if(saved.parallelCaches.size()!=options.devices.size())
            throw std::runtime_error("missing per-rank DiT history");
        for(const auto &cache:saved.parallelCaches)
            if(cache.key.size()!=54 || cache.value.size()!=54)
                throw std::runtime_error("incomplete per-rank DiT history");
        auto expected=pipeline.stepWorld(*world,1);
        auto advanced=*world;
        auto branch=saved;
        pipeline.stepWorld(branch,5);
        *world=saved;
        // Abort both while the scratch KV is being reused and after the new cache
        // is complete. Neither may overwrite the shared snapshot's history.
        for(int stage:{2,3}) {
            bool cancelled=false;
            try {
                pipeline.stepWorld(*world,1,[](int currentStage,int current,int,void *ptr) -> int {
                    int stage=*static_cast<int *>(ptr);
                    return currentStage==stage && current==(stage==2?3:1);
                },&stage);
            } catch(const std::runtime_error &e) {
                if(std::string(e.what())!="generation cancelled") throw;
                cancelled=true;
            }
            if(!cancelled || world->frameOffset!=saved.frameOffset || world->generator!=saved.generator ||
               world->decoderLatents!=saved.decoderLatents || world->prompt!=saved.prompt)
                throw std::runtime_error("cancelled step modified world");
            if(world->parallelCaches.size()!=saved.parallelCaches.size())
                throw std::runtime_error("cancelled step modified parallel history");
            for(size_t rank=0;rank<saved.parallelCaches.size();++rank) {
                const auto &a=world->parallelCaches[rank],&b=saved.parallelCaches[rank];
                if(a.key.size()!=b.key.size() || a.value.size()!=b.value.size())
                    throw std::runtime_error("cancelled step modified parallel layers");
                for(size_t i=0;i<b.key.size();++i)
                    if(a.key[i].data!=b.key[i].data || a.value[i].data!=b.value[i].data)
                        throw std::runtime_error("cancelled step modified parallel KV");
            }
        }
        if(pipeline.stepWorld(*world,1)!=expected) throw std::runtime_error("retry differs");
        auto resumed=pipeline.stepWorld(*world,7);
        std::string edit="A room with a blue vase on the wooden table.";
        pipeline.stepWorld(branch,9,nullptr,nullptr,&edit);
        *world=advanced;
        if(pipeline.stepWorld(*world,7)!=resumed) throw std::runtime_error("world switch or prompt edit changed another world");
    }
    std::cout<<"PASS parallel DiT/decode cancellation, world switching and prompt edit\n";
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;} }
