#include "models/minimax_h3/model_source.h"
#include "../../src/models/minimax_h3_image/quantized_partition.h"
#include "json11.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace fastllm::minimax_h3;
using fastllm::minimax_h3_image::detail::ProjectionSlice;
using Json=json11::Json;
namespace fs=std::filesystem;
static void check(bool value,const char *message) { if(!value) throw std::runtime_error(message); }
static void rejects(const std::function<void()> &fn) {
    try { fn(); } catch(const std::exception &) { return; }
    throw std::runtime_error("invalid quantized source accepted");
}
static void write(const fs::path &path,const std::string &text) {
    fs::create_directories(path.parent_path()); std::ofstream(path)<<text;
}
static void fixture(const fs::path &path,bool badShape=false,bool badMarker=false) {
    Json::object header,layers;
    uint64_t offset=0;
    auto tensor=[&](const std::string &name,const std::string &dtype,int n,int k) {
        uint64_t bytes=uint64_t(n)*k*(dtype=="F32"?4:dtype=="BF16"?2:1);
        header[name]=Json::object{{"dtype",dtype},{"shape",Json::array{n,k}},
            {"data_offsets",Json::array{double(offset),double(offset+bytes)}}};
        offset+=bytes;
    };
    for(const auto &name:{"video_patch_proj","audio_patch_proj"}) tensor(std::string(name)+".weight","BF16",5376,name[0]=='v'?96:32);
    tensor("final_layer.video_out.weight","BF16",badShape?6:96,5376);
    tensor("final_layer.audio_out.weight","BF16",32,5376);
    for(int i=0;i<52;++i) {
        const auto prefix=i<50?"blocks."+std::to_string(i):"token_refiner.blocks."+std::to_string(i-50);
        const std::vector<std::pair<std::string,std::pair<int,int>>> shapes={
            {".attn.qkv_proj",{21504,5376}},{".attn.out_proj",{5376,7168}},
            {".mlp.fc1",{28672,5376}},{".mlp.fc2",{5376,14336}}};
        for(const auto &entry:shapes) {
            const auto name=prefix+entry.first;
            tensor(name+".weight",i<50?"F8_E4M3":"BF16",entry.second.first,entry.second.second);
            if(i<50) {
                tensor(name+".weight_scale","F32",1,1);
                layers[name]=Json::object{{"format",badMarker?"unknown":"float8_e4m3fn"}};
            }
        }
    }
    header["__metadata__"]=Json::object{{"_quantization_metadata",Json(Json::object{{"layers",layers}}).dump()}};
    const auto text=Json(header).dump(); const uint64_t n=text.size();
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char *>(&n),8); f<<text;
    // Only metadata is read by discovery. Sparse payloads avoid multi-GB fixtures.
    f.seekp(n+8+offset-1); f.put(0); check(bool(f),"cannot create sparse checkpoint");
}
static void partitions() {
    for(const std::string name:{"blocks.0.attn.qkv_proj","blocks.0.mlp.fc1","blocks.0.attn.out_proj","blocks.0.mlp.fc2"}) {
        const bool qkv=name.find("qkv")!=std::string::npos,fc1=name.find("fc1")!=std::string::npos;
        const int groups=qkv?3:fc1?2:1,n=groups*512,k=1024;
        std::vector<unsigned char> input(size_t(n)*k),reconstructed(input.size(),0);
        for(size_t i=0;i<input.size();++i) input[i]=(i/17+i/k*13)%251;
        for(int rank=0;rank<4;++rank) {
            ProjectionSlice s(name,n,k,{true,rank,4}); std::vector<unsigned char> shard(size_t(s.rows)*s.cols);
            s.transfer([&](size_t pos,void *out,size_t bytes){std::copy_n(input.data()+pos,bytes,static_cast<unsigned char *>(out));},
                [&](size_t pos,const void *data,size_t bytes){std::copy_n(static_cast<const unsigned char *>(data),bytes,shard.data()+pos);},k,s.cols,s.firstColumn());
            for(int r=0;r<s.rows;++r) for(int c=0;c<s.cols;++c) {
                const int globalRow=groups==1?r:(r/(512/4))*512+rank*(512/4)+r%(512/4);
                const int globalCol=groups==1?rank*(k/4)+c:c;
                reconstructed[size_t(globalRow)*k+globalCol]=shard[size_t(r)*s.cols+c];
            }
        }
        check(input==reconstructed,"TP reconstruction changed grouped weight coordinates");
    }
    rejects([]{ProjectionSlice("blocks.0.attn.qkv_proj",21504,5376,{true,0,3});});
    rejects([]{ProjectionSlice("blocks.0.unknown",256,256,{true,0,2});});
}
int main() {
    const auto root=fs::temp_directory_path()/("h3-quant-source-"+std::to_string(std::random_device{}()));
    try {
        fs::create_directory(root); partitions();
        const auto base=root/"base",weights=root/"arbitrary.safetensors",view=root/"view";
        for(const char *part:{"transformer","transformer_ref","text_encoder","vae","audio_vae"}) write(base/part/"config.json","{}");
        write(base/"tokenizer/tokenizer.json","{}");
        fixture(weights);
        auto info=inspectVideoWeights(weights.string()); check(info.precision=="fp8" && info.bytes>(uint64_t(19)<<30),"format or byte estimate");
        Json::object manifest{{"format_version",1},{"base_model","../base"},{"fl2va","../arbitrary.safetensors"}};
        write(view/"h3_quantization.json",Json(manifest).dump());
        auto source=resolveModelSource(view.string());
        check(source.transformerPrecision=="fp8" && source.referencePrecision=="bf16" && source.referenceWeights.empty(),"partial mode must preserve BF16 fallback");
        check(resolveModelSource((view/"h3_quantization.json").string()).transformerWeights==weights.string(),"manifest file selection");
        manifest["unexpected"]=true; write(view/"h3_quantization.json",Json(manifest).dump());
        rejects([&]{resolveModelSource(view.string());});
        fixture(weights,true); rejects([&]{inspectVideoWeights(weights.string());});
        fixture(weights,false,true); rejects([&]{inspectVideoWeights(weights.string());});
        fs::resize_file(weights,32); rejects([&]{inspectVideoWeights(weights.string());});
        fs::remove_all(root); std::cout<<"PASS video format discovery, mode binding and grouped TP byte partitions\n";
    } catch(const std::exception &e) { fs::remove_all(root); std::cerr<<e.what()<<'\n'; return 1; }
}
