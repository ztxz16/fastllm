#include "utils/kernel_registry.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace fastllm::kernels {
using Json = json11::Json;
std::atomic<bool> enabled{false};
thread_local int bypassDepth = 0;
namespace {
using Clock = std::chrono::steady_clock;
struct Hash { size_t operator()(const Signature &s) const {
    size_t h=1469598103934665603ULL;
    for (auto v:s.fields) { h^=std::hash<int64_t>{}(v); h*=1099511628211ULL; }
    return h;
}};
std::mutex &Mutex() { static std::mutex value; return value; }
std::map<int,const Family *> &Families() { static std::map<int,const Family *> value; return value; }
std::unordered_map<Signature,int,Hash> plans;
std::map<std::string,Json> records;
Json configuration=Json::object{};
std::atomic<uint64_t> generation{1};
bool frozen=false;
double spentMs=0, budgetMs=30000, maxL2=.0032;
int repeats=8, samples=3;
std::string buildId;
std::vector<Json> prior;
thread_local std::unordered_map<Signature,int,Hash> local;
thread_local uint64_t localGeneration=0;
std::string Key(const Signature &s) {
    std::string result;
    for (auto value:s.fields) { result+=std::to_string(value); result+=':'; }
    return result;
}
Json Fields(const Signature &s) {
    Json::array out;
    for (auto value:s.fields) out.emplace_back(double(value));
    return out;
}
bool Disabled(const std::string &id) {
    for (const auto &x:configuration["deny"].array_items()) if(x.string_value()==id)return true;
    return false;
}
double Median(std::vector<double> xs) { std::sort(xs.begin(),xs.end()); return xs[xs.size()/2]; }
bool ValidTiming(double x) { return std::isfinite(x)&&x>0; }
void Require(bool ok,const char *text) { if(!ok)throw std::invalid_argument(text); }
}

void Register(const Family *family) {
    std::lock_guard<std::mutex> lock(Mutex());
    auto &all=Families();
    if(!family||family->id<=0||family->implementations.empty()||all.count(family->id))
        throw std::invalid_argument("duplicate or empty kernel family");
    all[family->id]=family;
}

bool Configure(const std::string &text,std::string &error) {
    std::lock_guard<std::mutex> lock(Mutex());
    try {
        auto config=Json::parse(text,error); Require(error.empty()&&config.is_object(),"kernel policy must be an object");
        const auto mode=config["mode"].is_null()?"quick":config["mode"].string_value();
        Require(mode=="off"||mode=="quick"||mode=="cache","kernel mode must be off, quick or cache");
        for(const auto &kv:config.object_items())
            Require(kv.first=="mode"||kv.first=="budget_ms"||kv.first=="repeats"||kv.first=="samples"||
                    kv.first=="max_l2"||kv.first=="build_id"||kv.first=="deny"||kv.first=="force"||kv.first=="profile",
                    "unknown kernel policy option");
        double budget=config["budget_ms"].is_null()?30000:config["budget_ms"].number_value();
        double tolerance=config["max_l2"].is_null()?.0032:config["max_l2"].number_value();
        int repeat=config["repeats"].is_null()?8:config["repeats"].int_value();
        int sample=config["samples"].is_null()?3:config["samples"].int_value();
        Require(std::isfinite(budget)&&budget>=0,"invalid kernel tuning budget");
        Require(std::isfinite(tolerance)&&tolerance>0&&tolerance<=.05,"invalid kernel max_l2");
        Require(repeat>0&&repeat<=1000&&sample>=3&&sample<=31,"invalid kernel measurement repetitions");
        Require(config["deny"].is_null()||config["deny"].is_array(),"deny must be an array");
        Require(config["force"].is_null()||config["force"].is_object(),"force must map family names to implementation IDs");
        for(const auto &x:config["deny"].array_items()) {
            Require(x.is_string(),"deny entries must be IDs");
            bool found=false;
            for(const auto &f:Families()) for(size_t i=0;i<f.second->implementations.size();++i)
                if(x.string_value()==f.second->implementations[i].id) {
                    Require(i!=0,"the reference fallback cannot be disabled");found=true;
                }
            Require(found,"unknown disabled kernel implementation");
        }
        for(const auto &kv:config["force"].object_items()) {
            bool found=false;
            for(const auto &f:Families())if(kv.first==f.second->name)
                for(const auto &impl:f.second->implementations)if(kv.second.string_value()==impl.id)found=true;
            Require(found,"unknown forced kernel family/implementation");
        }
        configuration=config;budgetMs=budget;maxL2=tolerance;repeats=repeat;samples=sample;spentMs=0;
        buildId=config["build_id"].is_null()?std::string(__DATE__ " " __TIME__):config["build_id"].string_value();
        prior=config["profile"]["records"].array_items();
        plans.clear();records.clear();frozen=mode=="cache";generation.fetch_add(1);
        enabled.store(mode!="off",std::memory_order_release);return true;
    } catch(const std::exception &e) { error=e.what();return false; }
}

int Select(const Call &call) {
    if(!Enabled())return -1;
    const auto gen=generation.load(std::memory_order_acquire);
    if(localGeneration!=gen){local.clear();localGeneration=gen;}
    if(auto it=local.find(call.signature);it!=local.end())return it->second;
    std::lock_guard<std::mutex> lock(Mutex());
    if(auto it=plans.find(call.signature);it!=plans.end())return local.emplace(call.signature,it->second).first->second;
    const auto &family=*call.family;
    const std::string key=Key(call.signature), hardware=call.hardware(call.context);
    Json::array candidates;
    std::vector<bool> legal(family.implementations.size(),false);
    const std::string forced=configuration["force"][family.name].string_value();
    for(size_t i=0;i<family.implementations.size();++i) {
        std::string reason=call.unsupported(call.context,int(i));
        if(Disabled(family.implementations[i].id))reason="disabled by policy";
        legal[i]=reason.empty();
        candidates.emplace_back(Json::object{{"id",family.implementations[i].id},{"status",reason.empty()?"eligible":reason}});
    }
    const int fallback=call.fallback>=0 && size_t(call.fallback)<legal.size() && legal[call.fallback]
        ? call.fallback : 0;
    int selected=fallback;std::string source="fallback";
    double caseMs=0;
    // Never select a stale profile based only on an SM number or requested max_batch.
    for(const auto &record:prior) {
        if(record["key"].string_value()!=key||record["hardware"].string_value()!=hardware||
           record["build_id"].string_value()!=buildId||record["version"].string_value()!=family.version||
           record["max_l2"].number_value()>maxL2||call.capturing||
           (record["source"].string_value()!="measured"&&record["source"].string_value()!="forced"&&
            record["source"].string_value()!="cache"))continue;
        for(size_t i=0;i<legal.size();++i)if(legal[i]&&record["selected"].string_value()==family.implementations[i].id&&
                (forced.empty()||forced==family.implementations[i].id)) {
            // Adapters still have to prepare compiled modules before graph capture.
            { Bypass scope;double l2=0;std::string reason;
                if(!call.validate(call.context,int(i),l2,reason)||!std::isfinite(l2)||l2>maxL2)continue;
                auto row=candidates[i].object_items();row["relative_l2"]=l2;row["status"]="cache revalidated";candidates[i]=row;
            }
            selected=int(i);source="cache";
        }
    }
    if(source!="cache" && std::count(legal.begin(),legal.end(),true)==1 && legal[0] && forced.empty())source="only-supported";
    const bool measure=source!="cache"&&source!="only-supported"&&!frozen&&!call.capturing&&spentMs<budgetMs;
    if(measure) {
        const auto start=Clock::now();
        std::vector<std::vector<double>> timings(legal.size());
        for(size_t i=0;i<legal.size();++i)if(legal[i]) {
            double l2=0;std::string reason;
            bool ok=false;
            { Bypass scope;ok=call.validate(call.context,int(i),l2,reason); }
            if(!ok||!std::isfinite(l2)||l2>maxL2){legal[i]=false;if(reason.empty())reason="numeric validation failed";}
            auto row=candidates[i].object_items();row["relative_l2"]=l2;row["status"]=legal[i]?"validated":reason;candidates[i]=row;
        }
        // Alternate order to reduce warm-cache/clock/order bias. Preparation is above this loop.
        for(int pass=0;pass<samples;++pass)for(size_t step=0;step<legal.size();++step) {
            size_t i=pass%2?legal.size()-1-step:step;if(!legal[i])continue;
            double ms;
            { Bypass scope;ms=call.measure(call.context,int(i),repeats); }
            if(ValidTiming(ms))timings[i].push_back(ms);else legal[i]=false;
        }
        double best=1e100;
        for(size_t i=0;i<legal.size();++i)if(legal[i]&&timings[i].size()==size_t(samples)) {
            const double median=Median(timings[i]);
            const auto range=std::minmax_element(timings[i].begin(),timings[i].end());
            auto row=candidates[i].object_items();Json::array values;for(auto t:timings[i])values.emplace_back(t);
            row["samples_ms"]=values;row["median_ms"]=median;row["spread_ms"]=*range.second-*range.first;candidates[i]=row;
            // Keep the earlier stable implementation when the measurements overlap.
            if((forced.empty()&&median<best&& (best==1e100||*range.second<best))||forced==family.implementations[i].id) {
                selected=int(i);best=median;source=forced.empty()?"measured":"forced";
            }
        }
        caseMs=std::chrono::duration<double,std::milli>(Clock::now()-start).count();spentMs+=caseMs;
    }
    if(!legal[selected])throw std::runtime_error("no valid kernel for "+std::string(family.name));
    if(!forced.empty()&&(family.implementations[selected].id!=forced||(source!="forced"&&source!="cache")))
        throw std::runtime_error("forced kernel unavailable, unmeasured or invalid for "+std::string(family.name)+": "+forced);
    Json::object record{{"key",key},{"shape",Fields(call.signature)},{"family",family.name},{"version",family.version},
        {"build_id",buildId},{"hardware",hardware},{"selected",family.implementations[selected].id},
        {"source",source},{"default_candidate",family.implementations[fallback].id},{"max_l2",maxL2},{"tuning_ms",caseMs},{"candidates",candidates}};
    if(source=="fallback")record["reason"]=call.capturing?"unmeasured graph capture":frozen?"unmeasured frozen plan":"tuning budget exhausted";
    // A cold capture cannot measure. Keep its graph on the trusted fallback,
    // but allow a later eager warmup to prepare this signature. Frozen plans
    // and exhausted budgets remain stable, and force still requires validation.
    const bool deferred=source=="fallback"&&call.capturing&&!frozen&&spentMs<budgetMs;
    records[key]=record;
    if(!deferred){plans[call.signature]=selected;local[call.signature]=selected;}
    printf("[kernels] %s rows=%lld dimensions=%lld,%lld groups/topk=%lld selected=%s source=%s\n",family.name,
        static_cast<long long>(call.signature.fields[3]),static_cast<long long>(call.signature.fields[4]),
        static_cast<long long>(call.signature.fields[5]),static_cast<long long>(call.signature.fields[6]),
        family.implementations[selected].id,source.c_str());
    return selected;
}

bool NeedsPreparation(const Signature &signature) {
    if(!Enabled())return false;
    std::lock_guard<std::mutex> lock(Mutex());return plans.find(signature)==plans.end();
}
bool CandidateAllowed(const char *id) {
    if(!enabled.load(std::memory_order_relaxed))return false;
    // Policy is configured before model loading and immutable while a model is active.
    return !Disabled(id);
}
void Freeze(){std::lock_guard<std::mutex> lock(Mutex());frozen=true;}
std::string Catalog() {
    std::lock_guard<std::mutex> lock(Mutex());Json::array out;
    for(const auto &item:Families()) {
        const auto &f=*item.second;Json::array impls;
        for(const auto &impl:f.implementations) {
            auto desc=impl.description.object_items();desc["id"]=impl.id;impls.emplace_back(desc);
        }
        out.emplace_back(Json::object{{"family",f.name},{"version",f.version},{"implementations",impls}});
    }
    return Json(Json::object{{"schema",1},{"families",out}}).dump();
}
std::string Report() {
    std::lock_guard<std::mutex> lock(Mutex());Json::array out;
    for(const auto &entry:records)out.push_back(entry.second);
    return Json(Json::object{{"schema",1},{"build_id",buildId},{"frozen",frozen},{"tuning_ms",spentMs},
        {"budget_ms",budgetMs},{"records",out}}).dump();
}
}

namespace {thread_local std::string kernelResult,kernelError;}
extern "C" int fastllm_kernel_configure(const char *json) {
    kernelError.clear();return fastllm::kernels::Configure(json?json:"",kernelError)?1:0;
}
extern "C" void fastllm_kernel_freeze(){fastllm::kernels::Freeze();}
extern "C" const char *fastllm_kernel_report(){kernelResult=fastllm::kernels::Report();return kernelResult.c_str();}
extern "C" const char *fastllm_kernel_catalog(){kernelResult=fastllm::kernels::Catalog();return kernelResult.c_str();}
extern "C" const char *fastllm_kernel_error(){return kernelError.c_str();}
