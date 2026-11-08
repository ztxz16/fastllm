#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "json11.hpp"
using Json=json11::Json;
enum class DataType { FLOAT16=3, FP8_E4M3=10 };
static void Check(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
constexpr size_t BufferBytes=8u<<20;
static size_t HostLimit(){return size_t(16)<<30;}
struct FakePool { std::vector<int> dims; size_t unitSize=1; };
struct Pages { FakePool *pool=nullptr; std::vector<std::pair<int,int>> ranges; };
struct FakeModel { int block_cnt=64,num_key_value_heads=4; };
struct FakeReader {
    struct Request { Json chunks; size_t expected; std::function<void(size_t,const char*,size_t)> write; };
    const std::vector<std::string> &bytes;
    explicit FakeReader(const std::vector<std::string>&b):bytes(b){}
    void ReadMany(const std::vector<Request>&requests) {
        for(const auto &r:requests){
            size_t i=size_t(r.chunks["index"].int_value());
            Check(i<bytes.size() && bytes[i].size()==r.expected,"fixture_payload_size");
            r.write(0,bytes[i].data(),bytes[i].size());
        }
    }
};
namespace DiskPrefixCache { struct ReadSession {using Request=FakeReader::Request;}; }
struct TestCase {
    std::map<int,Json::array> bundles;
    std::vector<std::string> bytes;
    std::vector<FakePool> managers;
    std::map<std::tuple<int,int,int>,std::shared_ptr<Pages>> pools;
    int length=0,begin=0,heads=4;
    bool cancel=false;
};
struct Result {
    double wall_ms=0,assemble_ms=0,readmany_ms=0;
    uint64_t memcpy_bytes=0,compare_bytes=0,segment_visits=0,span_copies=0,sample=0;
    std::vector<std::string> outputs;
    std::string error;
};
static void Observe(const char *data,size_t n,Result &r,bool full) {
    asm volatile("" : : "g"(data) : "memory");
    if(full)r.outputs.emplace_back(data,n);
    if(n)r.sample=r.sample*1315423911u+uint8_t(data[0])+uint8_t(data[n/2])+uint8_t(data[n-1]);
}
Result RunBase(TestCase &,bool);
Result RunCandidate(TestCase &,bool);
#include "assembly_base.inc"
#include "assembly_candidate.inc"
static Json ReadJson(const char *path) {
    std::ifstream f(path);std::string text((std::istreambuf_iterator<char>(f)),{}),error;
    auto result=Json::parse(text,error);Check(error.empty()&&result.is_object(),"fixture_json");return result;
}
static void AddPool(TestCase &tc,int layer,int side,int rank,int heads,int dim,
                    std::vector<std::pair<int,int>> ranges) {
    tc.managers.push_back({{0,0,heads,dim},1});
    auto p=std::make_shared<Pages>();p->pool=&tc.managers.back();p->ranges=std::move(ranges);
    tc.pools.emplace(std::make_tuple(layer,side,rank),p);
}
static TestCase FromReal(const Json&block,int repetitions) {
    TestCase tc;tc.managers.reserve(128);
    const int first=block["token_start"].int_value(),step=block["tokens"].int_value();
    Check(step>0,"fixture_block_tokens");
    for(int repeat=0;repeat<repetitions;++repeat) {
        for(const auto &bundle:block["kv"].array_items()) {
            auto object=bundle.object_items();Json::array segs;
            for(const auto &entry:bundle["segments"].array_items()){
                auto e=entry.object_items();e["begin"]=entry["begin"].int_value()-first+repeat*step;
                segs.emplace_back(std::move(e));
            }
            object["segments"]=segs;
            if(repeat==0){
                const size_t bytes=size_t(bundle["bytes"].number_value());
                std::string data(bytes,'\0');
                for(size_t i=0;i<bytes;++i)data[i]=char((i*31+(tc.bytes.size()*17))%251);
                tc.bytes.push_back(std::move(data));
            }
            // Chunk identity is local to the original bundle position.
            const size_t bindex=size_t(&bundle-&block["kv"].array_items()[0]);
            object["chunks"]=Json::object{{"index",int(bindex)}};
            tc.bundles[segs.front()["begin"].int_value()].emplace_back(std::move(object));
        }
    }
    tc.length=repetitions*step;
    std::map<std::tuple<int,int,int>,Json> layout;
    for(const auto &bundle:block["kv"].array_items())
        for(const auto &e:bundle["segments"].array_items())
            layout.emplace(std::make_tuple(e["layer"].int_value(),e["side"].int_value(),e["rank"].int_value()),e);
    for(const auto &item:layout) {
        auto e=item.second;std::vector<std::pair<int,int>> ranges;
        for(auto &r:e["ranges"].array_items())ranges.push_back({r[0].int_value(),r[1].int_value()});
        AddPool(tc,std::get<0>(item.first),std::get<1>(item.first),std::get<2>(item.first),e["heads"].int_value(),e["dim"].int_value(),ranges);
    }
    return tc;
}
static TestCase Synthetic(int control) {
    TestCase tc;tc.managers.reserve(2);tc.length=8;
    const int dim=16,tokens=8;
    Json::array segs;
    for(int i=0;i<2;++i) {
        Json::array ranges={Json::array{0,1},Json::array{1,2}};
        const size_t n=size_t(tokens)*2*dim;
        segs.emplace_back(Json::object{{"begin",0},{"tokens",tokens},{"heads",2},{"dim",dim},{"dtype",10},
         {"bytes",double(n)},{"offset",double(i*n)},{"layer",3},{"side",0},{"rank",i},{"ranges",ranges}});
    }
    const size_t segmentBytes=size_t(tokens)*2*dim;
    std::string data(segmentBytes*2,'x');
    for(size_t i=0;i<segmentBytes;++i)data[i]=data[segmentBytes+i]=char(i%251);
    tc.bytes={data};
    AddPool(tc,3,0,0,2,dim,{{0,2}});
    if(control==1)tc.bytes[0].back()^=1; // replicated bytes conflict
    if(control==2){auto e=segs[0].object_items();e["ranges"]=Json::array{Json::array{0,1}};segs[0]=e;}
    if(control==3){auto e=segs[0].object_items();e["ranges"]=Json::array{Json::array{0}};segs[0]=e;}
    if(control==4){auto e=segs[0].object_items();e["offset"]=1;segs[0]=e;}
    if(control==5)tc.pools.begin()->second->ranges={{0,3}};
    if(control==6)tc.pools.begin()->second->ranges={{3,5}};
    if(control==7)tc.cancel=true;
    if(control==8){segs.clear();} // malformed uncovered bundle fails equally before access
    if(control==9){
        segs.resize(1);auto e=segs[0].object_items();e["heads"]=1;e["bytes"]=double(segmentBytes/2);e["ranges"]=Json::array{Json::array{0,1}};segs[0]=e;
        tc.bytes[0].resize(segmentBytes/2);
    }
    if(control==10){auto e=segs[1].object_items();e["layer"]=7;segs[1]=e;}
    if(control==11){tc.pools.begin()->second->pool->unitSize=2;}
    if(control==12){auto e=segs[0].object_items();e["dtype"]=3;segs[0]=e;}
    if(control==13){
        tc.pools.clear();tc.managers.clear();AddPool(tc,3,0,0,1,dim,{{1,2}});
    }
    tc.bundles[0]={Json::object{{"bytes",double(tc.bytes[0].size())},{"chunks",Json::object{{"index",0}}},{"segments",segs}}};
    return tc;
}
static void Equal(TestCase &tc,const std::string &name) {
    auto a=RunBase(tc,true),b=RunCandidate(tc,true);
    Check(a.error==b.error,("different_error:"+name+":"+a.error+":"+b.error).c_str());
    if(a.error.empty()) {
        Check(a.outputs==b.outputs,("different_output:"+name).c_str());
        Check(a.memcpy_bytes==b.memcpy_bytes && a.compare_bytes==b.compare_bytes,"different_copy_work");
    }
    std::printf("control=%s result=%s output_buffers=%zu memcpy_bytes=%llu baseline_visits=%llu candidate_visits=%llu\n",
      name.c_str(),a.error.empty()?"PASS":a.error.c_str(),a.outputs.size(),
      (unsigned long long)a.memcpy_bytes,(unsigned long long)a.segment_visits,(unsigned long long)b.segment_visits);
}
int main(int argc,char **argv) {
    try {
        Check(argc>=3,"usage: assembly_t0 <metadata.json> controls|base|candidate [blocks]");
        const auto metadata=ReadJson(argv[1]);
        const std::string mode=argv[2];
        if(mode=="controls") {
            auto real=FromReal(metadata,1);Equal(real,"real_layout");
            for(int i=0;i<14;++i){auto tc=Synthetic(i);Equal(tc,"synthetic_"+std::to_string(i));}
            std::puts("PASS assembly exact-source controls");return 0;
        }
        const int blocks=argc>3?std::atoi(argv[3]):416;
        auto tc=FromReal(metadata,blocks);
        auto r=mode=="base"?RunBase(tc,false):RunCandidate(tc,false);
        Check(r.error.empty(),r.error.c_str());
        std::printf("{\"variant\":\"%s\",\"blocks\":%d,\"tokens\":%d,\"wall_ms\":%.3f,\"assemble_ms\":%.3f,"
          "\"readmany_ms\":%.3f,\"memcpy_bytes\":%llu,\"compare_bytes\":%llu,\"segment_visits\":%llu,\"span_copies\":%llu,\"sample\":%llu}\n",
          mode.c_str(),blocks,tc.length,r.wall_ms,r.assemble_ms,r.readmany_ms,(unsigned long long)r.memcpy_bytes,
          (unsigned long long)r.compare_bytes,(unsigned long long)r.segment_visits,(unsigned long long)r.span_copies,(unsigned long long)r.sample);
        return 0;
    }catch(const std::exception&e){std::fprintf(stderr,"FAIL assembly_t0: %s\n",e.what());return 1;}
}
