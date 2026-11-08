"""Exercise actual production status/capacity helpers with CPU-only owners."""
import datetime
import json
import os
import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(os.environ.get("FASTLLM_TEST_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "fastllm-test")))
(ROOT / 'logs').mkdir(parents=True, exist_ok=True)
REPO=pathlib.Path(__file__).resolve().parents[2]

def record(phase, **fields):
    with (ROOT/"logs/actions.jsonl").open("a") as output:
        output.write(json.dumps({"time_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 "task":"inference-status-repair","phase":phase,**fields})+"\n")
        output.flush();os.fsync(output.fileno())

class NativeRequestStatusT0(unittest.TestCase):
    def test_production_status_capacity_and_public_id_bridge(self):
        run=pathlib.Path(os.environ["FASTLLM_STATUS_ACCEPTANCE_RUN"])
        build=pathlib.Path(os.environ["FASTLLM_STATUS_ACCEPTANCE_BUILD"])/"status-t0"
        disk=(REPO/"src/utils/disk_prefix_cache.cpp").read_text()
        observation=disk[disk.index("Json DiskPrefixCache::CapacityObservation() const {"):
                         disk.index("Json DiskPrefixCache::CapacityStatus() const {")]
        self.assertNotIn("DrainForRead",observation)
        qwen=(REPO/"src/models/qwen3_5.cpp").read_text()
        capacity=qwen[qwen.index("    std::map<std::string, std::string> Qwen3_5Model::GetCacheCapacityObservation() {"):
                      qwen.index("    bool Qwen3_5Model::TryRecordPagedPrefixCacheExtra(")]
        self.assertNotIn("SpillStats",capacity)
        persistent=(REPO/"src/models/qwen3_5_persistent.inc").read_text()
        persistent_observation=persistent[persistent.index("    Json CapacityObservation() const {"):
                                          persistent.index("    std::string Stats() const {")]
        self.assertGreater(persistent.index("    Json CapacityObservation() const {"),persistent.index("public:"))
        self.assertNotIn("persistentPrefixCache->store",capacity)
        bridge=(REPO/"tools/src/pytools.cpp").read_text()
        bridge=bridge[bridge.index("    DLL_EXPORT bool get_request_status_llm_model("):
                      bridge.index("    DLL_EXPORT unsigned long long next_fastllm_observation_seq()")]
        body=r"""
#include <cassert>
#include <cstring>
#include <memory>
#include <set>
#include "utils/prefix_cache_logging.h"
#define DLL_EXPORT
namespace fastllm {
struct Json {
    using object=std::map<std::string,Json>;
    std::string text; object items;
    Json()=default;Json(const char *s):text(s){} Json(std::string s):text(std::move(s)){}
    Json(object o):items(std::move(o)){}
    const Json &operator[](const std::string &key)const{return items.at(key);}
    std::string string_value()const{return text;}
};
struct FakeStore {
    uint64_t physical=10ULL<<30,inflight=1ULL<<30;
    bool statWorks=true;
    uint64_t PhysicalBytes()const{return physical;}
    bool ObservePhysicalBytes(uint64_t &bytes,std::string &error)const{
        if(!statWorks){error="control_log_stat_5";return false;}
        bytes=physical;error.clear();return true;
    }
    uint64_t InFlightBytes()const{return inflight;}
};
struct DiskPrefixCache {
    struct Impl {bool v4Mode=true;uint64_t limit=256ULL<<30;std::shared_ptr<FakeStore> v4=std::make_shared<FakeStore>();};
    std::shared_ptr<Impl> impl=std::make_shared<Impl>();
    Json CapacityObservation()const;
};
""" + observation + r"""
struct PagedCacheManager {
    std::mutex pageIndexLocker;
    int maxPages=10,pageLen=128;
    std::vector<int> freePages{0,1,2},triePages{3,4};
    int FreePageCount()const{return int(freePages.size()+triePages.size());}
    uint64_t GetBytes()const{return 10240;}
};
struct Spill {std::mutex poolMutex;uint64_t bytes=2ULL<<30;};
class Persistent {
    std::shared_ptr<DiskPrefixCache> store=std::make_shared<DiskPrefixCache>();
    std::mutex mutex;
    size_t queuedJobs=1,queuedStateBytes=100,exportProtectedBytes=200;
    int activeWorkers=2;
    std::map<int,int> exports;
    struct Coordinator { uint64_t HostCreditsInUse()const{return 300;} };
    std::shared_ptr<Coordinator> coordinator=std::make_shared<Coordinator>();
    std::atomic<uint64_t> admittedCheckpoints{3},terminalCheckpoints{2};
public:
    std::shared_ptr<DiskPrefixCache> TestStore(){return store;}
""" + persistent_observation + r"""
};
static bool l2Enabled=true;
static bool Qwen35L2HostKVEnabled(){return l2Enabled;}
static uint64_t Qwen35L2HostKVPoolLimitBytes(){return 16ULL<<30;}
struct Qwen3_5Model {
    std::mutex capacityObservationMutex;
    std::vector<PagedCacheManager*> capacityObservationManagers;
    std::shared_ptr<Spill> pagedSpillState=std::make_shared<Spill>();
    std::shared_ptr<Persistent> persistentPrefixCache=std::make_shared<Persistent>();
    std::map<std::string,std::string> GetCacheCapacityObservation();
};
""" + capacity + r"""
}
struct Models {fastllm::Qwen3_5Model *model=nullptr;auto GetModel(int){return model;}} models;
""" + bridge + r"""
int main(){
    using namespace fastllm;
    RegisterFastllmPublicRequestId(7,"req-status");
    RequestStatusFields fields;
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("phase")=="preparing" && fields.at("native_rate_state")=="pending");
    assert(!ReadRequestStatus(7,"wrong",fields));
    const auto now=RequestStatusNowNs();
    BeginRequestPrefillStatus(7,4096);
    RecordRequestPrefillStatus(7,1024,now-300000000ULL,now-200000000ULL);
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("prefill_completed_tokens")=="1024");
    assert(fields.at("native_interval_tokens_s")=="10240.000");
    RecordRequestPrefillStatus(7,1024,now-200000000ULL,now-100000000ULL);
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("native_interval_tokens_s")=="10240.000"); // same completion is not counted twice
    RecordRequestPrefillStatus(7,2048,now-200000000ULL,now);
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("native_interval_tokens_s")=="5120.000");
    RecordRequestDecodeStatus(7,1,0,now,now,false);
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("phase")=="decode" && fields.at("native_rate_state")=="pending");
    assert(!fields.count("native_interval_tokens_s")); // prefill rate cannot leak into decode
    RecordRequestDecodeStatus(7,101,100,now,now+1000000000ULL,false);
    assert(ReadRequestStatus(7,"req-status",fields));
    assert(fields.at("native_interval_tokens_s")=="100.000");
    RegisterFastllmPublicRequestId(7,"req-reused");
    assert(!ReadRequestStatus(7,"req-status",fields));
    assert(ReadRequestStatus(7,"req-reused",fields) && !fields.count("native_output_tokens"));
    BeginRequestPrefillStatus(7,2048);
    RecordRequestPrefillStatus(7,1024,now-5000000000ULL,now-4000000000ULL);
    assert(ReadRequestStatus(7,"req-reused",fields));
    assert(fields.at("native_rate_state")=="stale");

    Qwen3_5Model model;models.model=&model;
    PagedCacheManager key,value;
    value.freePages={0,1};value.triePages={2,3,4};
    model.capacityObservationManagers={&key,&value};
    fields=model.GetCacheCapacityObservation();
    assert(fields.at("l1_total_tokens")=="1280");
    assert(fields.at("l1_used_tokens")=="1024" && fields.at("l1_available_tokens")=="256");
    assert(fields.at("l1_allocatable_tokens")=="640"); // reclaimable trie is not empty space
    assert(fields.at("l1_physical_bytes")=="20480"); // bytes sum, tokens do not
    assert(fields.at("l2_used_gib")=="2.000" && fields.at("l2_available_gib")=="14.000");
    assert(fields.at("ssd_available_gib")=="246.000" && fields.at("ssd_limit_gib")=="256.000");
    model.persistentPrefixCache->TestStore()->impl->limit=128ULL<<30;
    fields=model.GetCacheCapacityObservation();assert(fields.at("ssd_available_gib")=="118.000");
    model.persistentPrefixCache->TestStore()->impl->limit=1014ULL<<30;
    fields=model.GetCacheCapacityObservation();assert(fields.at("ssd_available_gib")=="1004.000");
    model.persistentPrefixCache->TestStore()->impl->v4->physical=2000ULL<<30;
    fields=model.GetCacheCapacityObservation();
    assert(fields.at("ssd_available_bytes")=="0" && fields.at("ssd_capacity_state")=="capacity_unavailable");
    key.pageIndexLocker.lock();
    fields=model.GetCacheCapacityObservation();
    assert(fields.at("l1_capacity_state")=="busy" && !fields.count("l1_total_tokens"));
    key.pageIndexLocker.unlock();
    model.pagedSpillState->poolMutex.lock();
    fields=model.GetCacheCapacityObservation();assert(fields.at("l2_capacity_state")=="busy");
    model.pagedSpillState->poolMutex.unlock();
    l2Enabled=false;fields=model.GetCacheCapacityObservation();assert(fields.at("l2_capacity_state")=="disabled");
    l2Enabled=true;
    char buffer[16384];
    assert(!get_request_status_llm_model(0,7,"req-status",buffer,sizeof(buffer)));
    assert(!get_request_status_llm_model(0,7,"req-reused",buffer,2));
    assert(get_request_status_llm_model(0,7,"req-reused",buffer,sizeof(buffer)));
    assert(std::string(buffer).find("l1_available_tokens=256")!=std::string::npos);
    assert(fields.count("snapshot_begin_ns")==0); // fields above came from the direct getter
    const auto encoded=std::string(buffer);
    assert(encoded.find("snapshot_phase_consistency=stable")!=std::string::npos);
    assert(encoded.find("l1_snapshot_consistency=all_required_pool_locks_held")!=std::string::npos);
    assert(encoded.find("ssd_queued_state_bytes=100")!=std::string::npos);
    model.persistentPrefixCache->TestStore()->impl->v4->statWorks=false;
    fields=model.GetCacheCapacityObservation();
    assert(fields.at("ssd_capacity_state")=="stat_failed" && !fields.count("ssd_available_bytes"));
    model.persistentPrefixCache->TestStore()->impl->v4->statWorks=true;
    UnregisterFastllmPublicRequestId(7);
    assert(!ReadRequestStatus(7,"req-reused",fields));

    BeginFastllmNativeRequest(8);
    BeginRequestPrefillStatus(8,4096);
    RegisterFastllmPublicRequestId(8,"req-before-bind");
    assert(ReadRequestStatus(8,"req-before-bind",fields) && fields.at("phase")=="prefill");
    const auto generation=fields.at("native_generation");
    RegisterFastllmPublicRequestId(8,"req-before-bind");
    assert(ReadRequestStatus(8,"req-before-bind",fields) && fields.at("native_generation")==generation);
    UnregisterFastllmPublicRequestId(8,"req-before-bind"); // detach while native still active
    assert(ReadRequestStatus(8,"req-before-bind",fields) && fields.at("observer_detached")=="1");
    EndFastllmNativeRequest(8); // actual cleanup retires the detached identity
    assert(!ReadRequestStatus(8,"req-before-bind",fields));
    BeginFastllmNativeRequest(8);
    EndFastllmNativeRequest(8); // a short native request may close before public ID bind
    RegisterFastllmPublicRequestId(8,"req-short");
    assert(ReadRequestStatus(8,"req-short",fields) && fields.at("phase")=="native_closed");
    BeginFastllmNativeRequest(8);RegisterFastllmPublicRequestId(8,"req-new-owner");
    UnregisterFastllmPublicRequestId(8,"req-short");
    assert(ReadRequestStatus(8,"req-new-owner",fields) && fields.at("phase")=="queued");
    EndFastllmNativeRequest(8);UnregisterFastllmPublicRequestId(8,"req-new-owner");
    std::puts("REQUEST_STATUS_NATIVE_T0_OK");
}
"""
        record("before",command="prepare status CPU probe and Makefile",cwd=str(REPO),
               build_dir=str(build),log_path=str(run),expected="actual native helper/getter/ABI source")
        build.mkdir(parents=True,exist_ok=False)
        (build/"probe.cpp").write_text(body)
        (build/"Makefile").write_text(
            "CXX := /usr/bin/c++\nCXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -pthread -DFASTLLM_DISK_PREFIX_CACHE -I"+str(REPO/"include")+"\n"+
            "all: status_probe raw_probe\nstatus_probe: probe.cpp\n\t$(CXX) $(CXXFLAGS) $< -o $@\n"+
            "raw_probe: "+str(REPO/"test/api/observability_raw_probe.cpp")+"\n\t$(CXX) $(CXXFLAGS) $< -o $@\n")
        for command,env,name in [
            (["make","-j36"],os.environ.copy(),"status-t0-compile.log"),
            ([str(build/"status_probe")],{**os.environ,"FASTLLM_OBSERVABILITY_MODE":"debug"},"status-t0-run.log")]:
            path=run/name
            record("before",command=command,cwd=str(build),build_dir=str(build),log_path=str(path),
                   gpu_mapping="none",expected="exit 0; no model or CUDA calls")
            with path.open("xb") as output:
                result=subprocess.run(command,cwd=build,env=env,stdout=output,stderr=subprocess.STDOUT)
            record("after",command=command,exit_code=result.returncode,log_path=str(path),
                   resource_cleanup="no runtime resources started")
            text=path.read_text();print(text,end="")
            self.assertEqual(result.returncode,0,text)
        os.environ["FASTLLM_TEST_OBSERVATION_PROBE"]=str(build/"raw_probe")
        self.assertIn("REQUEST_STATUS_NATIVE_T0_OK",(run/"status-t0-run.log").read_text())

if __name__=="__main__":unittest.main()
