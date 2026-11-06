#include "models/hy_nomas/runtime.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>
using namespace fastllm::nomas;
namespace {
void require(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
struct Fixture {
    std::string directory="/tmp/fastllm-host-budget-"+std::to_string(getpid());
    std::string path=directory+"/a.safetensors";
    std::vector<float> values;
    Fixture():values(256*24576) {
        require(std::filesystem::create_directory(directory),"fixture directory already exists");
        for(size_t n=0;n<values.size();++n) values[n]=float(int(n%251)-125)/16;
        write(path,"w");write(directory+"/b.safetensors","v");
        std::ofstream index(directory+"/model.safetensors.index.json");
        index<<"{\"weight_map\":{\"w\":\"a.safetensors\",\"v\":\"b.safetensors\"}}";
        require(bool(index),"fixture index write failed");
    }
    void write(const std::string &file,const std::string &name) const {
        std::string header="{\""+name+"\":{\"dtype\":\"F32\",\"shape\":[256,24576],\"data_offsets\":[0,"+
            std::to_string(values.size()*4)+"]}}";
        uint64_t length=header.size();
        std::ofstream f(file,std::ios::binary);
        f.write(reinterpret_cast<char *>(&length),8);f<<header;
        f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
        require(bool(f),"fixture write failed");
    }
    ~Fixture() { std::error_code error;std::filesystem::remove_all(directory,error); }
};
}
int main() {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    try {
        HostMemoryBudget concurrent(4096);
        std::vector<std::thread> workers;
        for(int n=0;n<8;++n) workers.emplace_back([&] {
            for(int i=0;i<10000;++i) if(concurrent.acquire(1024)) concurrent.release(1024);
        });
        for(auto &worker:workers) worker.join();
        require(concurrent.used()==0 && !concurrent.acquire(4097),"shared quota accounting");
        Fixture f;Runtime rt(0,DType::FP32);
        for(size_t capacity:{size_t(0),f.values.size()*4}) {
            auto quota=std::make_shared<HostMemoryBudget>(capacity);
            {
                Weights a(rt,f.path),b(rt,f.path);
                for(auto *w:{&a,&b}) {w->enableHostCache(size_t(64)<<20);w->setHostMemoryBudget(quota);}
                require(rt.download(a.get("w"))==f.values,"disk upload crossing chunk boundary");
                require(rt.download(b.get("w"))==f.values,"quota exhaustion disk fallback");
                require(quota->used()==capacity,"weight cache exceeded shared budget");
                a.clear();b.clear();
                require(rt.download(a.get("w"))==f.values,"cache/reload mismatch");
            }
            require(quota->used()==0,"destroyed cache leaked quota");
            for(int axis:{0,1}) for(int rank:{0,1}) {
                Weights w(rt,f.path);w.shard("w",axis,rank,2,2);
                w.enableHostCache(size_t(64)<<20);w.setHostMemoryBudget(quota);
                auto actual=rt.download(w.get("w"));
                int rows=axis==0?128:256,cols=axis==1?12288:24576;
                for(int r=0;r<rows;++r) for(int c=0;c<cols;++c) {
                    int sourceRow=axis==0?(r/64)*128+rank*64+r%64:r;
                    int sourceCol=axis==1?(c/6144)*12288+rank*6144+c%6144:c;
                    require(actual[size_t(r)*cols+c]==f.values[size_t(sourceRow)*24576+sourceCol],
                            "grouped partition disk/cache mismatch");
                }
            }
            require(quota->used()==0,"partition cache leaked quota");
            {
                Weights sharded(rt,f.directory),prefetched(rt,f.directory);
                sharded.enableHostCache(size_t(64)<<20);sharded.setHostMemoryBudget(quota);
                prefetched.shareHostCache(sharded);
                require(rt.download(sharded.get("w"))==f.values,"first checkpoint shard");
                require(rt.download(sharded.get("v"))==f.values,"second checkpoint shard disk fallback");
                sharded.clear();
                require(rt.download(prefetched.get("w"))==f.values,"prefetch shared cache reload");
                require(quota->used()==capacity,"checkpoint shards exceeded shared quota");
            }
            require(quota->used()==0,"checkpoint shards leaked quota");
            {
                auto brokenPath=f.directory+"/truncated.safetensors";
                f.write(brokenPath,"w");
                Weights broken(rt,brokenPath);
                broken.enableHostCache(size_t(64)<<20);broken.setHostMemoryBudget(quota);
                // Truncate after reading metadata to exercise read-time cleanup,
                // including any quota acquired before the first disk read.
                std::filesystem::resize_file(brokenPath,4096);
                bool rejected=false;
                try { broken.get("w"); }
                catch(const std::invalid_argument &e) {
                    rejected=std::string(e.what())=="truncated weight";
                }
                require(rejected && quota->used()==0,"failed read leaked host quota");
                f.write(brokenPath,"w");
                require(rt.download(broken.get("w"))==f.values,"failed read poisoned loader retry");
            }
            require(quota->used()==0,"recovered loader leaked quota");
        }
        std::cout<<"PASS shared host budget, disk fallback, partitions, checkpoint shards and failed-read recovery\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
