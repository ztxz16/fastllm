#include "models/lingbot_world/pipeline.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace fastllm::lingbot;
namespace {
void allocationScopes(Runtime &rt) {
    fastllm::nomas::TensorPool outer(0),inner(0);
    auto allocate=[&](int rows) { Tensor scratch(rows,1024); };
    {
        fastllm::nomas::TensorPool::Scope scope(outer);
        allocate(512);
        {
            fastllm::nomas::TensorPool::Scope nested(inner);
            allocate(512); rt.synchronize();
        }
        // Pool reservation granularity is driver-dependent. Exceed the
        // actual inner reservation instead of assuming a fixed page size.
        allocate(int(inner.reusableBytes()/2048)+4096);
    }
    rt.synchronize();
    if(outer.reusableBytes()<=inner.reusableBytes() || !inner.reusableBytes())
        throw std::runtime_error("nested allocation scope was not restored");
    outer.trim(); inner.trim();
    try {
        fastllm::nomas::TensorPool::Scope scope(outer);
        throw std::runtime_error("allocation scope cancellation");
    } catch(const std::runtime_error &) {}
    allocate(4096); rt.synchronize();
    if(outer.reusableBytes() || inner.reusableBytes())
        throw std::runtime_error("allocation scope leaked into another model");
}
std::vector<float> read(const std::string &path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()%4) throw std::runtime_error("invalid fixture: "+path);
    std::vector<float> v(size_t(f.tellg())/4);f.seekg(0);f.read(reinterpret_cast<char *>(v.data()),v.size()*4);
    if(!f) throw std::runtime_error("truncated fixture");
    return v;
}
void equal(const Segment &a,const Segment &b) {
    if(a.frames!=b.frames || a.height!=b.height || a.width!=b.width || a.pixels.size()!=b.pixels.size() ||
       std::memcmp(a.pixels.data(),b.pixels.data(),a.pixels.size()*4))
        throw std::runtime_error("transactional replay differs");
    for(float x:b.pixels) if(!std::isfinite(x)) throw std::runtime_error("nonfinite replay");
}
}
int main(int argc,char **argv) { try {
    if(argc!=5) throw std::invalid_argument("usage: test_pipeline_state model fixture height width");
    int h=std::stoi(argv[3]),w=std::stoi(argv[4]);
    auto image=read(std::string(argv[2])+"/first_image.f32"),text=read(std::string(argv[2])+"/0.text.f32");
    if(image.size()!=size_t(h)*w*3 || text.empty() || text.size()%4096) throw std::runtime_error("fixture shape");
    Pipeline pipeline(argv[1],0);
    allocationScopes(pipeline.runtime());
    WorldState world;
    world.height=h;world.width=w;world.seed=42;world.intrinsics={float(w),float(w),w*.5f,h*.5f};
    world.firstPixels=pipeline.runtime().fp32.upload(image.data(),h*w,3);
    world.text=pipeline.runtime().upload(text.data(),text.size()/4096,4096);
    pipeline.step(world);
    auto saved=world;
    auto expected=pipeline.step(world,"forward");
    auto advanced=world;
    auto branch=saved;
    pipeline.step(branch,"look_left"); // switches active state while the original snapshot survives
    world=saved;
    pipeline.offloadState(); // force lossless host fallback before rollback
    bool aborted=false;
    try {
        pipeline.step(world,"forward",nullptr,[](const std::string &stage,int current,int total) {
            if(stage=="decode" && current==total) throw std::runtime_error("injected cancellation");
        });
    } catch(const std::runtime_error &e) {
        if(std::string(e.what())!="injected cancellation") throw;
        aborted=true;
    }
    if(!aborted || world.segments!=saved.segments || world.noiseSequence!=saved.noiseSequence || world.pose!=saved.pose ||
       world.cache!=saved.cache || world.encoder!=saved.encoder || world.decoder!=saved.decoder)
        throw std::runtime_error("cancelled step changed world state");
    equal(expected,pipeline.step(world,"forward"));
    // Revisit a different branch, then resume the successful world snapshot.
    pipeline.step(branch,"right");
    auto resumed=pipeline.step(world,"look_up");
    world=advanced;
    pipeline.offloadState(); // revisit an older snapshot after eviction
    equal(resumed,pipeline.step(world,"look_up"));
    std::cout<<"world switching, cancelled-step rollback and deterministic replay: passed\n";
} catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; } }
