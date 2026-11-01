#include "models/hy_nomas/parallel.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace fastllm::nomas;
namespace {
void require(bool v,const char *message) { if(!v) throw std::runtime_error(message); }
void near(const std::vector<float> &a,const std::vector<float> &b,double tolerance) {
    require(a.size()==b.size(),"comparison shape mismatch");
    double error=0,norm=0;
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"non-finite result");
        error+=double(a[i]-b[i])*(a[i]-b[i]); norm+=double(b[i])*b[i];
    }
    double relative=std::sqrt(error/std::max(norm,1e-30));
    std::cout<<"relative L2 "<<relative<<" (limit "<<tolerance<<")\n";
    require(relative<=tolerance,"parallel numerical mismatch");
}
std::string fixture() {
    std::string path="/tmp/fastllm-parallel-"+std::to_string(getpid())+".safetensors";
    std::string header="{";
    std::vector<float> values;
    auto tensor=[&](const std::string &name,int rows,int cols,bool bias=false) {
        size_t first=values.size();
        for(int i=0;i<rows*cols;++i) values.push_back(std::sin(float(i)*.17f)*.07f);
        if(header.size()>1) header+=",";
        header+="\""+name+"\":{\"dtype\":\"F32\",\"shape\":["+
            (bias?std::to_string(cols):std::to_string(rows)+","+std::to_string(cols))+
            "],\"data_offsets\":["+std::to_string(first*4)+","+std::to_string(values.size()*4)+"]}";
    };
    tensor("up.weight",512,128); tensor("up.bias",1,512,true);
    tensor("down.weight",128,256); tensor("down.bias",1,128,true);
    tensor("qkv.weight",384,128); tensor("qkv.bias",1,384,true);
    tensor("wide.weight",2048,256); tensor("wide.bias",1,2048,true);
    header+="}";
    uint64_t length=header.size();
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<char *>(&length),8); f<<header;
    f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    require(bool(f),"cannot write checkpoint fixture");
    return path;
}
}
int main() {
#ifndef FASTLLM_TEST_NCCL
    return 77;
#endif
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
    std::string path;
    try {
        require(parseParallelDevices("0,1")==std::vector<int>({0,1}),"device parser");
        for(auto value:{"","0,","0,0","-1","0x1","0,a"}) {
            bool rejected=false;
            try { parseParallelDevices(value); } catch(const std::exception &) { rejected=true; }
            require(rejected,"invalid device list accepted");
        }
        {
            Runtime root(0,DType::FP32),other(1,DType::FP32);
            root.activate();TensorPool pool(0,1u<<20);
            TensorPool::Scope scope(pool);
            std::vector<float> values(257*16,.5f);
            auto saved=root.upload(values.data(),257,16);
            for(int i=0;i<40;++i) {
                root.activate();auto temporary=root.add(saved,saved);
                other.activate();temporary={}; // Free while another device is current.
            }
            root.activate();near(root.download(saved),values,0);root.synchronize();pool.trim();
        }
        path=fixture();
        for(int ranks:{2,4}) {
            if(count<ranks) continue;
            std::vector<int> devices(ranks);
            for(int rank=0;rank<ranks;++rank) devices[rank]=rank;
            ParallelGroup activationGroup(devices);
            for(auto dtype:{DType::BF16,DType::FP16}) {
                Runtime root(0,dtype); Weights full(root,path);
                for(int rows:{67,8193}) {
                    std::vector<float> values(rows*256);
                    for(size_t i=0;i<values.size();++i) values[i]=std::cos(i*.013)*.4;
                    auto input=root.upload(values.data(),rows,256);
                    auto expected=root.download(full.linear(input,"wide"));
                    std::vector<std::vector<float>> actual(ranks);
                    activationGroup.run([&](int rank) {
                        Runtime rt(rank,dtype); Weights w(rt,path);
                        ParallelContext p{&activationGroup,rank,RowReduction::Activation};
                        p.rowWeights(w,"wide");
                        auto x=activationGroup.broadcast(input,rank);
                        x=rt.columns(x,rank*256/ranks,256/ranks);
                        actual[rank]=rt.download(p.row(rt,w,x,"wide"));
                    });
                    // Compare with the unsharded projection, including a single
                    // bias, not a second invocation of the distributed code.
                    near(actual[0],expected,dtype==DType::BF16?.007:.001);
                    for(int rank=1;rank<ranks;++rank) near(actual[rank],actual[0],0);
                }
                activationGroup.run([&](int rank) {
                    Runtime rt(rank,dtype);
                    std::vector<float> values(777*1025,float(rank)+.5f);
                    auto x=rt.upload(values.data(),777,1025);
                    for(int first=0;first<777;first+=257)
                        activationGroup.sumAsync(x.rowSlice(first,std::min(257,777-first)),rank);
                    activationGroup.wait(rank);
                    near(rt.download(x),std::vector<float>(values.size(),ranks*ranks*.5f),0);
                });
            }
            // Token partitions may be uneven; K and V are separate groups.
            // Gather only moves data and must preserve every value on every rank.
            for(auto dtype:{DType::FP32,DType::BF16,DType::FP16})
                for(int rows:{67,128}) for(int groups:{1,2,3}) for(bool async:{false,true}) {
                    activationGroup.run([&](int rank) {
                        Runtime rt(rank,dtype);
                        ParallelContext p{&activationGroup,rank};
                        constexpr int cols=17;
                        int chunk=(rows+ranks-1)/ranks,first=rank*chunk,local=std::min(chunk,rows-first);
                        std::vector<float> expected(groups*rows*cols),values;
                        for(size_t i=0;i<expected.size();++i) expected[i]=float(int(i%113)-56)/16;
                        for(int g=0;g<groups;++g) {
                            auto begin=expected.begin()+(g*rows+first)*cols;
                            values.insert(values.end(),begin,begin+local*cols);
                        }
                        auto input=rt.upload(values.data(),groups*local,cols);
                        auto actual=p.gatherRows(rt,input,rows,groups,async);
                        input={}; // Communication owns inputs until wait().
                        // Reuse allocations and queue independent work before consuming.
                        for(int j=0;j<8;++j) {
                            auto scratch=rt.upload(values.data(),groups*local,cols);
                            scratch=rt.add(scratch,scratch);
                        }
                        if(async) activationGroup.wait(rank);
                        require(actual.rows==groups*rows && actual.cols==cols,"row gather output shape");
                        near(rt.download(actual),expected,0);
                    });
                }
            bool gatherFailed=false;
            try {
                activationGroup.run([&](int rank) {
                    Runtime rt(rank,DType::BF16);
                    TensorPool pool(rank,4u<<20);TensorPool::Scope scope(pool);
                    ParallelContext p{&activationGroup,rank};
                    std::vector<float> values(128*2048,float(rank));
                    auto x=rt.upload(values.data(),128,2048);
                    auto out=p.gatherRows(rt,x,128*ranks,1,true);
                    x={};out={};
                    if(rank==0) throw std::runtime_error("cancel asynchronous gather");
                });
            } catch(const std::exception &) {gatherFailed=true;}
            require(gatherFailed,"asynchronous gather cancellation not propagated");
            activationGroup.run([&](int rank) {
                Runtime rt(rank,DType::BF16);float one=1;auto x=rt.upload(&one,1,1);
                activationGroup.sum(x,rank);
                require(rt.download(x)[0]==ranks,"asynchronous gather cancellation recovery");
            });
            bool invalidRows=false;
            try {
                activationGroup.run([&](int rank) {
                    Runtime rt(rank,DType::BF16);ParallelContext p{&activationGroup,rank};
                    float one=1;auto x=rt.upload(&one,1,1);
                    p.gatherRows(rt,x,128);
                });
            } catch(const std::invalid_argument &) {invalidRows=true;}
            require(invalidRows,"invalid row partition accepted");
            bool failed=false;
            try {
                activationGroup.run([&](int rank) {
                    Runtime rt(rank,DType::BF16); std::vector<float> values(257*1024,.5f);
                    auto x=rt.upload(values.data(),257,1024);
                    activationGroup.sumAsync(x,rank);
                    if(rank==0) throw std::runtime_error("cancel activation reduction");
                });
            } catch(const std::exception &) { failed=true; }
            require(failed,"activation reduction cancellation not propagated");
            activationGroup.run([&](int rank) {
                Runtime rt(rank,true); float one=1;auto x=rt.upload(&one,1,1);
                activationGroup.sum(x,rank);
                require(rt.download(x)[0]==ranks,"activation reduction cancellation recovery");
            });
        }
        {
            const std::string cachePath=path+".bf16";
            const std::string header="{\"weight\":{\"dtype\":\"BF16\",\"shape\":[2,4],\"data_offsets\":[0,16]}}";
            uint64_t length=header.size();
            uint16_t data[8]={0x3f00,0x3f80,0x4000,0x4040,0x4080,0x40a0,0x40c0,0x40e0};
            std::ofstream file(cachePath,std::ios::binary);
            file.write(reinterpret_cast<char *>(&length),8); file<<header;
            file.write(reinterpret_cast<char *>(data),sizeof(data)); file.close();
            try {
                Runtime rt(0,DType::BF16);
                Weights a(rt,cachePath),b(rt,cachePath),wrong(rt,cachePath);
                a.enablePackedCache(size_t(1)<<30);
                b.sharePackedCache(a);
                auto first=a.get("weight");
                a.clear(true);
                bool rejected=false;
                try { a.shard("weight",0,0,2); } catch(const std::invalid_argument &) { rejected=true; }
                require(rejected,"partition changed while a device cache was retained");
                require(a.get("weight").data==first.data,"device cache not retained");
                require(b.get("weight").data==first.data,"device cache not shared");
                wrong.shard("weight",0,0,2);
                rejected=false;
                try { wrong.sharePackedCache(a); } catch(const std::exception &) { rejected=true; }
                require(rejected,"different partitions shared a device cache");
                rejected=false;
                try { wrong.shareHostCache(a); } catch(const std::exception &) { rejected=true; }
                require(rejected,"different partitions shared a host cache");
                a.clear();
                require(a.get("weight").data!=first.data,"device cache not released");
            } catch(...) { std::filesystem::remove(cachePath); throw; }
            std::filesystem::remove(cachePath);
        }
        {
            Runtime rt(0,true); Weights invalid(rt,path);
            invalid.shard("up.weight",0,0,3,2);
            bool rejected=false;
            try { invalid.get("up.weight"); } catch(const std::invalid_argument &) { rejected=true; }
            require(rejected,"non-divisible weight partition accepted");
        }
        ParallelGroup group({0,1});
        std::vector<std::thread::id> threads(2);
        group.run([&](int rank) { threads[rank]=std::this_thread::get_id(); });
        require(threads[0]!=threads[1],"forward ranks share a thread");
        for(auto dtype:{DType::FP32,DType::BF16,DType::FP16})
            for(int cols:{1024,1025}) for(bool biased:{false,true}) {
                group.run([&](int rank) {
                    Runtime rt(rank,true);
                    std::vector<float> values(257*cols),bias(cols);
                    for(size_t j=0;j<values.size();++j)
                        values[j]=std::sin(j*.013+rank)*.7f;
                    for(int j=0;j<cols;++j) bias[j]=std::cos(j*.071)*.03f;
                    auto x=rt.upload(values.data(),257,cols),b=rt.upload(bias.data(),1,cols);
                    auto actual=group.sumCast(rt,x,biased?&b:nullptr,dtype,rank);
                    x=rt.upload(values.data(),257,cols);
                    group.sum(x,rank);
                    auto expected=rt.cast(biased?rt.bias(x,b):x,dtype);
                    // 1024 columns split in the middle of a row; 1025 tests
                    // uneven counts. Keep the old FP32 reduction exactly.
                    near(rt.download(actual),rt.download(expected),0);
                });
            }
        for(auto dtype:{DType::BF16,DType::FP16}) {
            group.run([&](int rank) {
                Runtime rt(rank,true);
                std::vector<float> values(777*1024),bias(1024);
                for(size_t j=0;j<values.size();++j) values[j]=std::sin(j*.013+rank)*.7f;
                for(int j=0;j<1024;++j) bias[j]=std::cos(j*.071)*.03f;
                auto x=rt.upload(values.data(),777,1024),b=rt.upload(bias.data(),1,1024);
                Tensor actual(777,1024,dtype);
                for(int first=0;first<777;first+=257) {
                    int rows=std::min(257,777-first);
                    auto source=x.rowSlice(first,rows),target=actual.rowSlice(first,rows);
                    group.sumCastAsync(source,&b,target,rank);
                }
                // Caller references can retire before the communication stream.
                x={}; b={}; group.wait(rank);
                auto reference=rt.upload(values.data(),777,1024);
                group.sum(reference,rank);
                b=rt.upload(bias.data(),1,1024);
                near(rt.download(actual),rt.download(rt.cast(rt.bias(reference,b),dtype)),0);
            });
        }
        for(auto dtype:{DType::BF16,DType::FP16}) {
            group.run([&](int rank) {
                Runtime rt(rank,dtype); Weights w(rt,path); ParallelContext p{&group,rank};
                p.rowWeights(w,"wide");
                std::vector<float> values(8193*128);
                for(size_t j=0;j<values.size();++j) values[j]=std::sin(j*.013+rank)*.7f;
                auto x=rt.upload(values.data(),8193,128);
                Tensor partial(8193,2048,DType::FP32);
                rt.linearInto(x,w.get("wide.weight"),partial);
                auto expected=rt.download(p.finishRow(rt,w,partial,"wide",dtype));
                auto actual=rt.download(p.row(rt,w,x,"wide"));
                // A different GEMM row count can change FP32 summation order,
                // especially for the one-row tail. Bound that rounding error
                // separately from the asynchronous collective, which is exact.
                near(actual,expected,1e-5);
                for(int first=0;first<x.rows;first+=4096) {
                    int rows=std::min(4096,x.rows-first);
                    auto target=partial.rowSlice(first,rows);
                    rt.linearInto(x.rowSlice(first,rows),w.get("wide.weight"),target);
                }
                near(actual,rt.download(p.finishRow(rt,w,partial,"wide",dtype)),0);
            });
        }
        for(size_t jobs:{size_t(0),size_t(1),size_t(5)}) {
            auto mapped=group.mapToRoot(jobs,[&](int rank,size_t job) {
                Runtime rt(rank,true);
                std::vector<float> values((job+1)*3,float(job+rank*10));
                return rt.upload(values.data(),job+1,3);
            });
            Runtime root(0,true);
            require(mapped.size()==jobs,"tile map count mismatch");
            for(size_t job=0;job<jobs;++job)
                near(root.download(mapped[job]),std::vector<float>((job+1)*3,float(job+(job%2)*10)),0);
        }
        for(auto dtype:{DType::FP32,DType::BF16,DType::FP16}) {
            Runtime root(0,dtype);
            Weights full(root,path);
            std::vector<float> data(67*128);
            for(size_t i=0;i<data.size();++i) data[i]=std::cos(i*.013)*.4;
            auto input=root.upload(data.data(),67,128);
            auto expected=root.download(full.linear(root.siluMultiply(full.linear(input,"up")),"down"));
            auto denseUp=full.linear(input,"up");
            auto expectedUp=root.download(denseUp);
            auto expectedQKV=root.download(full.linear(input,"qkv"));
            std::vector<std::vector<float>> result(2);
            group.run([&](int rank) {
                require(threads[rank]==std::this_thread::get_id(),"worker thread changed");
                Runtime rt(rank,dtype);
                Weights weights(rt,path);
                ParallelContext parallel{&group,rank};
                parallel.columnWeights(weights,"up",2);
                parallel.rowWeights(weights,"down");
                parallel.columnWeights(weights,"qkv",3);
                weights.enableHostCache(1<<20);
                auto x=group.broadcast(input,rank);
                for(int iteration=0;iteration<2;++iteration) {
                    auto up=parallel.column(weights,x,"up");
                    auto upCPU=rt.download(up);
                    std::vector<float> sliced;
                    for(int row=0;row<67;++row)
                        for(int g=0;g<2;++g)
                            for(int c=0;c<128;++c)
                                sliced.push_back(expectedUp[row*512+g*256+rank*128+c]);
                    near(upCPU,sliced,dtype==DType::FP32?2e-6:0.002);
                    near(rt.download(parallel.column(weights,x,"up",true)),expectedUp,dtype==DType::FP32?2e-6:0.002);
                    near(rt.download(parallel.column(weights,x,"qkv",true)),expectedQKV,dtype==DType::FP32?2e-6:0.002);
                    auto out=parallel.row(rt,weights,rt.siluMultiply(up),"down");
                    result[rank]=rt.download(out);
                    weights.clear(); // Retained host cache must contain local shards.
                }
                // Gather must interleave columns per row, not concatenate rank rows.
                auto local=rt.columns(x,rank*64,64);
                near(rt.download(group.gatherColumns(local,rank)),rt.download(x),0);
            });
            near(result[0],result[1],0);
            near(result[0],expected,dtype==DType::FP32?3e-6:0.004);
        }
        bool failed=false;
        try {
            group.run([&](int rank) {
                if(rank==0) throw std::runtime_error("injected rank cancellation");
                Runtime rt(rank,true); float one=1;
                auto x=rt.upload(&one,1,1); group.sum(x,rank);
            });
        } catch(const std::exception &) { failed=true; }
        require(failed,"rank failure not propagated");
        group.run([&](int rank) {
            Runtime rt(rank,true); float value=float(rank+1);
            auto x=rt.upload(&value,1,1); group.sum(x,rank);
            require(rt.download(x)[0]==3,"communicator did not recover");
        });
        failed=false;
        try {
            group.run([&](int rank) {
                Runtime rt(rank,true); std::vector<float> values(257*1024,.5f);
                auto x=rt.upload(values.data(),257,1024); Tensor out(257,1024,DType::BF16);
                group.sumCastAsync(x,nullptr,out,rank);
                if(rank==0) throw std::runtime_error("cancel asynchronous reduction");
            });
        } catch(const std::exception &) { failed=true; }
        require(failed,"asynchronous cancellation not propagated");
        group.run([&](int rank) {
            Runtime rt(rank,true);float one=1;auto x=rt.upload(&one,1,1);
            group.sum(x,rank);require(rt.download(x)[0]==2,"asynchronous cancellation recovery");
        });
        failed=false;
        try {
            group.run([&](int rank) {
                Runtime rt(rank,true); float values[2]={1,2};
                auto x=rt.upload(values,1,rank+1); group.sum(x,rank);
            });
        } catch(const std::exception &) { failed=true; }
        require(failed,"mismatched collective sizes accepted");
        std::filesystem::remove(path);
        std::cout<<"PASS continuous tensor parallel\n";
    } catch(const std::exception &e) {
        if(!path.empty()) std::filesystem::remove(path);
        std::cerr<<e.what()<<"\n"; return 1;
    }
}
