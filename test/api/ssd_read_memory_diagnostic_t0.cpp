#include "utils/disk_prefix_cache.h"
#include <atomic>
#include <cstdlib>
#include <cerrno>
#include <unistd.h>
#include <malloc.h>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>
using fastllm::DiskPrefixCache;
using Json=json11::Json;
// Diagnostics only: notify the parent, then wait without allocating.
// The parent freezes this CPU probe and reads /proc/<pid>/smaps externally.
static void MemoryCheckpoint(char stage) {
    const char *out = std::getenv("B21_DIAG_NOTIFY_FD");
    const char *in = std::getenv("B21_DIAG_ACK_FD");
    if (!out || !in) throw std::runtime_error("missing_memory_diagnostic_pipe");
    // Optional allocator counters are written as fixed native uint64 fields;
    // no formatted output or heap allocation is added to the checkpoint.
    if (const char *allocatorFd = std::getenv("B21_DIAG_ALLOC_FD")) {
        const auto m = mallinfo2();
        const uint64_t fields[] = {uint64_t(stage),m.arena,m.ordblks,m.smblks,
            m.hblks,m.hblkhd,m.usmblks,m.fsmblks,m.uordblks,m.fordblks,m.keepcost};
        size_t done = 0;
        while (done < sizeof(fields)) {
            const ssize_t n = write(std::atoi(allocatorFd),
                reinterpret_cast<const char *>(fields) + done, sizeof(fields) - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("memory_diagnostic_allocator");
            done += size_t(n);
        }
    }
    ssize_t count;
    do { count = write(std::atoi(out), &stage, 1); } while (count < 0 && errno == EINTR);
    if (count != 1) throw std::runtime_error("memory_diagnostic_notify");
    char ack;
    do { count = read(std::atoi(in), &ack, 1); } while (count < 0 && errno == EINTR);
    if (count != 1) throw std::runtime_error("memory_diagnostic_ack");
}
static uint64_t Io(const char *name) {
    std::ifstream f("/proc/self/io");std::string key;uint64_t value;
    while(f>>key>>value)if(key==std::string(name)+":")return value;
    return 0;
}
static double CpuMs(const rusage &u) {
    return (u.ru_utime.tv_sec+u.ru_stime.tv_sec)*1000.0+
           (u.ru_utime.tv_usec+u.ru_stime.tv_usec)/1000.0;
}
static uint64_t ResidentKiB() {
    std::ifstream f("/proc/self/status");std::string line;
    while(std::getline(f,line))if(line.rfind("VmRSS:",0)==0)return std::stoull(line.substr(6));
    return 0;
}
int main(int argc,char **argv) {
    if(argc!=5)return 2;
    const std::string dir=argv[1],key=argv[2],identity=argv[3],mode=argv[4];
    Json state;std::string error;uint64_t checked=0;
    uint64_t r0=0,p0=0,r1=0,p1=0,loadPeakKiB=0,residentBefore=0,residentAfter=0;
    double ms=0,cpuMs=0;bool readRejected=false;
    try {
        MemoryCheckpoint('P');
        DiskPrefixCache cache(dir,identity,128ull<<30);
        auto session=cache.BeginRead();
        std::atomic<uint64_t> done{0};
        DiskPrefixCache::ReadProgress progress;
        if(mode=="cancel-start")progress=[](uint64_t,uint64_t){return false;};
        if(mode=="cancel-progress")progress=[&](uint64_t n,uint64_t){return done.fetch_add(n)+n<(1u<<20);};
        r0=Io("rchar");p0=Io("read_bytes");
        rusage cpuBefore{};getrusage(RUSAGE_SELF,&cpuBefore);residentBefore=ResidentKiB();MemoryCheckpoint('B');
        const auto start=std::chrono::steady_clock::now();
        state=mode=="manifest"?session.LoadManifest(key,identity):session.Load(key,identity,true,progress);
        ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        rusage cpuAfter{};getrusage(RUSAGE_SELF,&cpuAfter);
        cpuMs=CpuMs(cpuAfter)-CpuMs(cpuBefore);loadPeakKiB=cpuAfter.ru_maxrss;residentAfter=ResidentKiB();MemoryCheckpoint('L');
        r1=Io("rchar");p1=Io("read_bytes");
        auto read=[&](const Json &chunks) {
            for(const auto &chunk:chunks.array_items()) {
                const size_t n=size_t(chunk["bytes"].number_value());
                std::string bytes(n,'\0');
                session.ReadBytes(Json::array{chunk},n,[&](size_t offset,const char *data,size_t count){
                    if(offset>bytes.size()||count>bytes.size()-offset)throw std::runtime_error("probe_callback_bounds");
                    bytes.replace(offset,count,data,count);
                });
                if(DiskPrefixCache::Digest(bytes)!=chunk["sha256"].string_value())throw std::runtime_error("probe_bad_bytes");
                checked+=n;
            }
        };
        if(mode=="manifest") {
            try {session.ReadBytes(Json::array{},0,[](size_t,const char*,size_t){});}
            catch(const std::exception &e){readRejected=std::string(e.what())=="manifest_only_read_requires_full_load";}
            if(!readRejected)throw std::runtime_error("probe_manifest_guard_missing");
        } else {
            read(state["linear_chunks"]);read(state["speculative_chunks"]);
            for(const auto &bundle:state["kv"].array_items())read(bundle["chunks"]);
        }
    }catch(const std::exception &e){error=e.what();}
    MemoryCheckpoint('C');
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    Json out=Json::object{{"ok",error.empty()},{"error",error},{"mode",mode},{"load_ms",ms},
        {"state_digest",state.is_object()?DiskPrefixCache::Digest(state.dump_compact()):""},
        {"kv_bundles",double(state["kv"].array_items().size())},{"checked_bytes",double(checked)},
        {"rchar",double(r1-r0)},{"physical_read_bytes",double(p1-p0)},
        {"manifest_read_rejected",readRejected},{"maxrss_kb",double(usage.ru_maxrss)},
        {"load_cpu_ms",cpuMs},{"load_peak_kib",double(loadPeakKiB)},
        {"resident_before_kib",double(residentBefore)},{"resident_after_kib",double(residentAfter)}};
    std::puts(out.dump_compact().c_str());return 0;
}
