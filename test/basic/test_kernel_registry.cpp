// Portable selector tests: deterministic timing isolates selection from CPU noise.
#include "utils/kernel_registry.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace fastllm::kernels;
using json11::Json;
#define CHECK(x) do {if(!(x))throw std::runtime_error(#x);}while(0)
struct Context {int runs=0,measures=0,validations=0;bool supported=true;std::string hardware="cpu:test:isa:threads:numa";
    std::vector<float> input{1,2,3,4},output;
    static bool Run(void *p,int i){auto &c=*(Context*)p;++c.runs;c.output.clear();for(float x:c.input)c.output.push_back(x*x+(i==2?.1f:0));return true;}
    static std::string Support(void *p,int i){return i==1&&!((Context*)p)->supported?"missing ISA":"";}
    static double Measure(void *p,int i,int){++((Context*)p)->measures;return i==0?2.:i==1?1.:.1;}
    static bool Validate(void *p,int i,double &l2,std::string &){auto &c=*(Context*)p;++c.validations;Run(p,0);auto ref=c.output;Run(p,i);double sq=0,e=0;for(size_t k=0;k<ref.size();++k){sq+=ref[k]*ref[k];e+=pow(ref[k]-c.output[k],2);}l2=sqrt(e/sq);return true;}
    static std::string Hardware(void*p){return ((Context*)p)->hardware;}
};
int main(){try{
    Family f{90,"cpu.square","test-v1",{{"cpu.reference",Json::object{}},{"cpu.vector",Json::object{}},{"cpu.inaccurate",Json::object{}}}};Register(&f);
    auto config=[](Json::object v){std::string e;v["build_id"]="test-build";CHECK(Configure(Json(v).dump(),e));};
    Context c;Call call{&f,{{90,0,0,5,4,0,0,0,0,0,0,0}},&c,false,Context::Support,Context::Measure,Context::Validate,Context::Hardware};
    config({{"mode","off"}});CHECK(Select(call)==-1&&c.runs==0);
    config({{"mode","quick"}});CHECK(Select(call)==1);CHECK(c.measures==6&&c.validations==3);
    std::string parseError;auto report=Json::parse(Report(),parseError);CHECK(parseError.empty());int ran=c.runs,measured=c.measures;CHECK(Select(call)==1&&ran==c.runs&&measured==c.measures);
    std::thread thread([&]{CHECK(Select(call)==1);});thread.join();CHECK(c.runs==ran);
    Freeze();call.signature.fields[3]=7;CHECK(Select(call)==0&&c.measures==measured);
    config({{"mode","quick"}});call.capturing=true;CHECK(Select(call)==0&&c.runs==ran);call.capturing=false;
    config({{"mode","cache"},{"profile",report}});call.signature.fields[3]=5;CHECK(Select(call)==1&&c.measures==measured&&c.runs>ran);
    config({{"mode","cache"},{"profile",report}});c.hardware="cpu:different";ran=c.runs;CHECK(Select(call)==0&&c.runs==ran);c.hardware="cpu:test:isa:threads:numa";
    config({{"mode","quick"},{"deny",Json::array{"cpu.vector"}}});CHECK(Select(call)==0);
    config({{"mode","quick"},{"force",Json::object{{"cpu.square","cpu.vector"}}}});CHECK(Select(call)==1);
    config({{"mode","quick"},{"force",Json::object{{"cpu.square","cpu.inaccurate"}}}});bool threw=false;try{Select(call);}catch(const std::runtime_error&){threw=true;}CHECK(threw);
    config({{"mode","quick"},{"budget_ms",0}});measured=c.measures;CHECK(Select(call)==0&&measured==c.measures);
    call.fallback=1;
    config({{"mode","quick"},{"budget_ms",0}});ran=c.runs;CHECK(Select(call)==1&&c.runs==ran);
    config({{"mode","quick"}});Freeze();CHECK(Select(call)==1&&c.runs==ran);
    config({{"mode","quick"}});call.capturing=true;CHECK(Select(call)==1&&c.runs==ran);call.capturing=false;
    config({{"mode","quick"},{"budget_ms",0},{"deny",Json::array{"cpu.vector"}}});CHECK(Select(call)==0&&c.runs==ran);
    config({{"mode","quick"},{"budget_ms",0}});c.supported=false;CHECK(Select(call)==0&&c.runs==ran);c.supported=true;
    config({{"mode","quick"},{"budget_ms",0},{"force",Json::object{{"cpu.square","cpu.vector"}}}});
    threw=false;try{Select(call);}catch(const std::runtime_error&){threw=true;}CHECK(threw&&c.runs==ran);
    config({{"mode","quick"}});CHECK(Select(call)==1);call.fallback=0;
    config({{"mode","quick"}});c.supported=false;CHECK(Select(call)==0);c.supported=true;
    {Bypass b;CHECK(Select(call)==-1);}
    std::string error;CHECK(!Configure("{\"deny\":[\"cpu.reference\"]}",error));CHECK(!Configure("{\"deny\":[\"typo\"]}",error));CHECK(!Configure("{\"typo\":1}",error));
    CHECK(!Configure("{\"force\":{\"cpu.square\":\"typo\"}}",error));CHECK(!Configure("{\"max_l2\":2}",error));
    CHECK(Catalog().find("cpu.vector")!=std::string::npos);
    std::cout<<"KERNEL_REGISTRY_PASS immutable plans, numeric rejection, capability, cache, freeze, capture, overrides, thread lookup\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;} }
