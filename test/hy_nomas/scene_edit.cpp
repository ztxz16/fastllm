// Native state/rollback acceptance for live prompts. Run on the reserved GPU.
#include "models/hy_nomas/runtime.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace fastllm::nomas;
static void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
static void unchanged(const WorldState &a,const WorldState &b) {
    require(a.prompt==b.prompt && a.features.data==b.features.data,"prompt/features changed on cancellation");
    require(a.generator==b.generator && a.frameOffset==b.frameOffset,"RNG/timeline advanced on cancellation");
    require(a.decoderLatents==b.decoderLatents && a.segmentTokens==b.segmentTokens,"history changed on cancellation");
    for(int i=0;i<54;++i)
        require(a.cache.key[i].data==b.cache.key[i].data && a.cache.value[i].data==b.cache.value[i].data,
                "KV cache changed on cancellation");
}
int main(int argc,char **argv) {
    if(argc!=4) { std::cerr<<"model text output\n"; return 2; }
    try {
        std::filesystem::create_directories(argv[3]);
        Pipeline pipeline(argv[1],argv[2],0);
        auto world=pipeline.createWorld("A red ball rolls slowly across a wooden floor.",128,128,42,8);
        auto save=[&](int step,const std::vector<uint8_t> &rgb) {
            std::ofstream file(std::string(argv[3])+"/step_"+std::to_string(step)+"_served.rgb",std::ios::binary);
            file.write(reinterpret_cast<const char *>(rgb.data()),rgb.size());
            require(bool(file),"cannot save RGB");
        };
        save(0,pipeline.stepWorld(*world,0));
        auto initial=*world;
        std::string edited="A blue ball rolls slowly across a wooden floor, illuminated by warm sunlight.";
        bool cancelled=false;
        try {
            pipeline.stepWorld(*world,1,[](int stage,int current,int,void*){return int(stage==2&&current==1);},nullptr,&edited);
        } catch(const std::runtime_error &e) { cancelled=std::string(e.what())=="generation cancelled"; }
        require(cancelled,"edit did not cancel"); unchanged(*world,initial);
        save(1,pipeline.stepWorld(*world,1,nullptr,nullptr,&edited));
        require(world->prompt==edited && world->frameOffset==10 && world->segmentTokens.size()==2,"edit reset the world");
        require(world->segmentTokens.front()==initial.segmentTokens.front(),"edit replaced the old segment");
        require(world->features.rows!=initial.features.rows,"test must cover a changed text token count");
        auto updated=*world; bool encoded=false;
        try {
            pipeline.stepWorld(*world,0,[](int stage,int,int,void *p){if(stage==1)*static_cast<bool*>(p)=true;return 1;},&encoded,&edited);
        } catch(const std::runtime_error &) {}
        require(!encoded,"unchanged prompt was re-encoded"); unchanged(*world,updated);
        int crop=0;
        auto empty=QwenTokenizer(argv[2]).videoPrompt("{\"segments\": [{}]}",crop);
        const int emptyTokens=int(empty.size())-crop, imageTokens=5*4*4;
        require(emptyTokens!=world->features.rows,"test needs distinct continuation and scene token counts");
        auto noEncoding=[](int stage,int,int,void *p){if(stage==1)*static_cast<bool*>(p)=true;return 0;};
        save(2,pipeline.stepWorld(*world,1,noEncoding,&encoded,&edited));
        require(!encoded,"exploring reloaded the text encoder");
        require(world->features.data==updated.features.data && world->prompt==edited,"exploring overwrote the accepted scene description");
        require(world->segmentTokens.back()==imageTokens+emptyTokens,"movement repeated the scene instruction");
        auto exploring=*world;
        bool cancelledMove=false;
        try {
            pipeline.stepWorld(*world,0,[](int stage,int current,int,void*){return int(stage==2&&current==1);});
        } catch(const std::runtime_error &e) {cancelledMove=std::string(e.what())=="generation cancelled";}
        require(cancelledMove,"exploration did not cancel");unchanged(*world,exploring);
        save(3,pipeline.stepWorld(*world,9,noEncoding,&encoded));
        require(!encoded,"explicitly following the prompt reloaded the text encoder");
        require(world->segmentTokens.back()==imageTokens+world->features.rows,"follow_prompt lost the scene instruction");
        for(int action: {0,2,3,4,5,6,7,8}) {
            pipeline.stepWorld(*world,action,noEncoding,&encoded);
            require(world->segmentTokens.back()==imageTokens+emptyTokens,"a camera action repeated the scene instruction");
        }
        require(!encoded,"camera/text switching reloaded the text encoder");
        int keptTokens=0;for(int n:world->segmentTokens)keptTokens+=n;
        require(world->segmentTokens.size()==6,"history bound was lost");
        for(int i=0;i<54;++i)
            require(world->cache.key[i].rows==keptTokens && world->cache.value[i].rows==keptTokens,"variable text length broke cache eviction");
        // A second branch reuses the identical first segment for the official
        // two-segment precision comparison. It does not reuse native KV on the
        // reference side; compare_world.py reconstructs that independently.
        if(const char *traceEnv=std::getenv("FASTLLM_NOMAS_TRACE")) {
            const std::string traceDir=traceEnv;
            const auto branchDir=std::filesystem::path(traceDir)/"exploration";
            std::filesystem::create_directories(branchDir);
            for(const auto &file:std::filesystem::directory_iterator(traceDir))
                if(file.is_regular_file() && file.path().filename().string().rfind("step_0_",0)==0)
                    std::filesystem::copy_file(file.path(),branchDir/file.path().filename(),std::filesystem::copy_options::overwrite_existing);
            setenv("FASTLLM_NOMAS_TRACE",branchDir.c_str(),1);
            auto branch=initial;
            auto rgb=pipeline.stepWorld(branch,1,noEncoding,&encoded);
            std::ofstream out(branchDir/"step_1_served.rgb",std::ios::binary);
            out.write(reinterpret_cast<const char *>(rgb.data()),rgb.size());
            require(bool(out) && !encoded,"exploration precision capture failed");
            setenv("FASTLLM_NOMAS_TRACE",traceDir.c_str(),1);
        }
        std::ofstream report(std::string(argv[3])+"/scene_edit_state.json");
        report<<"{\"passed\":true,\"segments\":"<<world->frameOffset/5<<",\"old_text_tokens\":"<<initial.features.rows
              <<",\"new_text_tokens\":"<<world->features.rows<<",\"continuation_text_tokens\":"<<emptyTokens
              <<",\"cancel_rollback\":true,\"unchanged_prompt_reencoded\":false,\"all_camera_actions_continue\":true,\"follow_prompt_repeats_text\":true,\"bounded_cache\":true}\n";
        std::cout<<"PASS: edit/rollback, exploration without repeated text, explicit text following, switching without re-encoding, variable-length cache eviction\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
