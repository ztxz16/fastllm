#include "fastllm.h"
#include "utils.h"
#include "gguf.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <unistd.h>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Fixture {
    static constexpr int experts = 8;
    const int hidden, inter;
    std::string path;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> gpu, disk;
    uint64_t bytes = 0;
    Fixture(int gt, int dt, int device, int width=512, int middle=256)
        : hidden(width), inter(middle), gpu(2*(experts+1)), disk(gpu.size()) {
        char name[] = "/tmp/fastllm-disk-v41-XXXXXX";
        int fd = mkstemp(name); Check(fd >= 0, "tempfile"); path = name;
        FILE *file = fdopen(fd, "wb"); std::mt19937 rng(317+gt*17+dt);
        std::normal_distribution<float> normal(0, .04f);
        for (int e=0; e<experts; ++e) for (int part=0; part<2; ++part) {
            int type = part ? dt : gt;
            auto dims = part ? std::vector<int>{hidden,inter} : std::vector<int>{2*inter,hidden};
            auto w = std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            w->disableGGUFRepack = true; w->isGGUFData = true;
            if (type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
                w->Allocate(false);
                for (size_t i=0; i<w->GetBytes(); ++i) w->cpuData[i] = rng()>>24;
                for (size_t i=0; i<w->GetBytes(); i+=ggml_type_size(ggml_type(type))) {
                    uint16_t scale = float_to_half(.002f + .0002f*(rng()%8));
                    if (type == GGML_TYPE_IQ1_M) {
                        auto *sc = reinterpret_cast<uint16_t *>(w->cpuData+i+48);
                        for (int j=0; j<4; ++j) sc[j] = (sc[j]&0xfff) | ((scale>>(4*j)&15)<<12);
                    } else memcpy(w->cpuData+i, &scale, 2);
                }
            } else {
                std::vector<float> values(size_t(dims[0])*dims[1]);
                for (float &v : values) v = normal(rng);
                w->CreateFromOriData(WeightType::LINEAR,FLOAT32,(uint8_t *)values.data(),nullptr,nullptr);
            }
            auto lazy = std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            lazy->isDiskWeight = lazy->isModelWeight = lazy->isGGUFData = true;
            char padding[37] = {}; fwrite(padding,1,sizeof(padding),file);
            DiskWeightPart p; p.fileName=path; p.fileOffset=ftell(file);
            p.bytes=w->GetBytes(); p.sourceDataType=DATA_GGUF_FORMAT; p.dims=dims;
            if (!part && e%2 == 0) {
                // Split gate/up projections with distinct odd file prefixes.
                // The aligned second read must preserve the first one's tail.
                const uint64_t first = p.bytes/2;
                p.bytes = first;
                lazy->diskWeightParts.push_back(p);
                Check(fwrite(w->cpuData,1,first,file)==first,"write split gate");
                fwrite(padding,1,19,file);
                p.fileOffset = ftell(file); p.bytes = w->GetBytes()-first;
                lazy->diskWeightParts.push_back(p);
                Check(fwrite(w->cpuData+first,1,p.bytes,file)==p.bytes,"write split up");
            } else {
                lazy->diskWeightParts.push_back(p);
                Check(fwrite(w->cpuData,1,p.bytes,file)==p.bytes,"write");
            }
            if (e==0) bytes += w->GetBytes();
            w->ToDevice(CUDA,{device},true);
            gpu[2+2*e+part]=w.get(); disk[2+2*e+part]=lazy.get();
            owned.push_back(std::move(w)); owned.push_back(std::move(lazy));
        }
        fclose(file);
    }
    ~Fixture() { owned.clear(); unlink(path.c_str()); }
    std::vector<float> Run(bool lazy, int rows, int offset, int device, int topk=6) {
        Data x(BFLOAT16,{rows,hidden}), y(BFLOAT16,{rows,hidden});
        Data ids(INT32,{rows,topk}), scores(FLOAT32,{rows,topk}),a,b,c;
        x.Allocate(false); ids.Allocate(false); scores.Allocate(false);
        for (int r=0; r<rows; ++r) {
            for (int j=0; j<hidden; ++j) ((uint16_t *)x.cpuData)[r*hidden+j] =
                Float32ToBFloat16RNEBits(std::sin((r*hidden+j)*.71f)*3);
            for (int k=0; k<topk; ++k) {
                ((int32_t *)ids.cpuData)[r*topk+k] = (offset+(k==2 && r%3==0 ? 0 : k)+r%2)%experts;
                ((float *)scores.cpuData)[r*topk+k] = k==0 ? 0 : k==1 ? -.17f : .2f+.13f*k;
            }
        }
        FastllmCudaSetDevice(device);
        x.ToDevice(CUDA,{device},true);
        if (lazy) {
            x.ToDevice(CPU);
            std::vector<Data *> biases(disk.size());
            DataDict data={{"input",&x},{"output",&y},{"index",&ids},{"score",&scores},
                {"weights",(Data *)disk.data()},{"biass",(Data *)biases.data()},{"w1",&a},{"w2",&b},{"w3",&c}};
            IntDict ip={{"weights___batch",int(disk.size())},{"biass___batch",int(disk.size())},
                {"deepSeekV4Mode",1},{"activationQuantBlock",32}};
            DiskMergeMOE op; op.Run("MergeMOE",data,{{"sharedScale",0},{"swigluLimit",.8f}},ip);
        } else {
            std::vector<int32_t> orderIds((int32_t *)ids.cpuData,(int32_t *)ids.cpuData+rows*topk);
            ids.ToDevice(CUDA,{device},true); scores.ToDevice(CUDA,{device},true);
            // One route at a time avoids duplicate-row scatter races in the
            // legacy fallback. Sum in stable expert order, with one final BF16 boundary.
            y.Allocate(false);
            for (int r=0; r<rows; ++r) {
                std::vector<int> order(topk);
                for (int k=0; k<topk; ++k) order[k]=k;
                std::stable_sort(order.begin(),order.end(),[&](int a,int b) {
                    return orderIds[r*topk+a]<orderIds[r*topk+b];
                });
                std::vector<float> sum(hidden,0.f);
                for (int k : order) {
                    Data xi, ii, ss, out;
                    xi.FakeFrom(x,size_t(r)*hidden*2); xi.Resize({1,hidden});
                    ii.FakeFrom(ids,size_t(r*topk+k)*4); ii.Resize({1,1});
                    ss.FakeFrom(scores,size_t(r*topk+k)*4); ss.Resize({1,1});
                    DoCudaMergeMOEV41Fallback(xi,out,ii,ss,gpu.data(),gpu.size(),0,.8f);
                    out.ToDevice(CPU);
                    for (int c=0; c<hidden; ++c) sum[c]+=((float *)out.cpuData)[c];
                }
                for (int c=0; c<hidden; ++c)
                    ((uint16_t *)y.cpuData)[r*hidden+c]=Float32ToBFloat16RNEBits(sum[c]);
            }
        }
        y.ToDevice(CPU);
        std::vector<float> out(size_t(rows)*hidden);
        for (size_t i=0; i<out.size(); ++i) out[i]=BFloat16BitsToFloat32(((uint16_t *)y.cpuData)[i]);
        return out;
    }
};
static void Compare(const std::vector<float> &a, const std::vector<float> &b, double tolerance) {
    Check(a.size()==b.size(),"shape"); double err=0,norm=0;
    for (size_t i=0; i<a.size(); ++i) {
        Check(std::isfinite(a[i]),"finite"); err+=(a[i]-b[i])*(a[i]-b[i]); norm+=b[i]*b[i];
    }
    double rel=std::sqrt(err/std::max(norm,1e-20));
    printf("relative L2 %.8g (limit %.8g)\n",rel,tolerance); fflush(stdout);
    Check(rel<=tolerance,"disk V4.1 GGUF numerics");
}
static uint64_t PreloadRecordBytes(const Fixture &f, bool mapped = true) {
    const long nativePage = sysconf(_SC_PAGESIZE);
    const uint64_t page = nativePage > 0 ? nativePage : 4096;
    uint64_t bytes = 0;
    for (int part=0; part<2; ++part) {
        const uint64_t payload = f.disk[2+part]->GetBytes();
        bytes += mapped ? (payload+8192+page-1)/page*page : payload;
    }
    return bytes;
}
static void TestPreload(Fixture &f, int device, bool mapped = true) {
    SetMoeCudaCacheBytes(0); SetMoeCpuCacheBytes(0); TrimDiskMoeCache();
    auto reference = f.Run(true,1,0,device,8);
    auto before = GetDiskMoeCacheStats();
    Check(PreloadDiskMoeCache({f.disk})==0,"disabled GGUF preload");
    const uint64_t record = PreloadRecordBytes(f,mapped);
    SetMoeCpuCacheBytes(record-1);
    Check(PreloadDiskMoeCache({f.disk})==0,"undersized GGUF preload");
    Check(GetDiskMoeCacheStats().diskBytes==before.diskBytes,"unbudgeted preload read disk");
    SetMoeCpuCacheBytes(record*Fixture::experts-1);
    Check(PreloadDiskMoeCache({f.disk,f.disk})==record*(Fixture::experts-1),"GGUF padding/duplicate preload budget");
    auto partial = GetDiskMoeCacheStats();
    Check(partial.cpuBytes==record*(Fixture::experts-1),"GGUF preload byte accounting");
    Check(partial.diskBytes-before.diskBytes==f.bytes*(Fixture::experts-1),"GGUF preload read accounting");
    Check(partial.cpuHits==before.cpuHits && partial.misses==before.misses,"GGUF preload fabricated routing");
    Check(PreloadDiskMoeCache({f.disk})==0,"GGUF preload not idempotent");
    SetMoeCpuCacheBytes(record*Fixture::experts);
    Check(PreloadDiskMoeCache({f.disk})==record,"GGUF preload did not fill remaining capacity");
    auto full = GetDiskMoeCacheStats();
    Compare(f.Run(true,1,0,device,8),reference,0);
    Check(GetDiskMoeCacheStats().diskBytes==full.diskBytes,"preloaded GGUF reread disk");
    SetMoeCpuCacheBytes(0); TrimDiskMoeCache();
    Check(!GetDiskMoeCacheStats().cpuBytes,"GGUF preload survived disable");
}
int main() { try {
    if (FastllmCudaGetDeviceCount()<1) return 77;
    SetThreads(4);
    for (auto pair : std::vector<std::pair<int,int>>{
            {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},{GGML_TYPE_IQ1_M,GGML_TYPE_Q2_0},
            {GGML_TYPE_Q2_0,GGML_TYPE_Q3_K},{GGML_TYPE_Q3_K,GGML_TYPE_Q4_K},
            {GGML_TYPE_Q4_K,GGML_TYPE_IQ1_S}}) {
        for (int device=0; device<std::min(2,FastllmCudaGetDeviceCount()); ++device) {
            printf("gate=%d down=%d device=%d\n",pair.first,pair.second,device); fflush(stdout);
            Fixture f(pair.first,pair.second,device);
            TestPreload(f,device);
            const uint64_t ramBudget = f.bytes*8 + Fixture::experts*2*4*4096;
            SetMoeCpuCacheBytes(0); SetMoeCudaCacheBytes(0); TrimDiskMoeCache();
            auto cpuReference = f.Run(true,1,0,device,8);
            SetMoeCpuCacheBytes(ramBudget); SetMoeCudaCacheBytes(f.bytes*8);
            auto start=GetDiskMoeCacheStats();
            for (int pass=0; pass<16; ++pass) f.Run(true,1,pass%2,device,8);
            auto warm=GetDiskMoeCacheStats();
            Check(warm.uploads>start.uploads && warm.cudaHits>start.cudaHits,"GPU cache unused");
            Check(warm.cudaBytes==f.bytes*8,"all experts not admitted");
            Check(PreloadDiskMoeCache({f.disk})==0 && GetDiskMoeCacheStats().diskBytes==warm.diskBytes,
                  "preload duplicated GPU-resident GGUF experts");
            // Exactly matches the native decode oracle after all routes hit.
            for (int rows : {1,5,8,9,33}) Compare(f.Run(true,rows,0,device),f.Run(false,rows,0,device),0);
            // Shrink VRAM with RAM headroom: every canonical record must
            // survive eviction without another disk read or requantization.
            auto beforeDemotion = GetDiskMoeCacheStats();
            SetMoeCudaCacheBytes(0); TrimDiskMoeCache();
            auto demoted = GetDiskMoeCacheStats();
            Check(demoted.cudaDemotions == beforeDemotion.cudaDemotions+8 &&
                  demoted.cudaDemotionBytes == beforeDemotion.cudaDemotionBytes+8*f.bytes,
                  "GPU-to-CPU demotion missing");
            Check(!demoted.cudaBytes && demoted.cpuBytes >= f.bytes*8 &&
                  demoted.cpuBytes+demoted.cpuWorkspaceBytes<=ramBudget,"demotion residency and workspace budget");
            Compare(f.Run(true,1,0,device,8),cpuReference,0);
            Check(GetDiskMoeCacheStats().diskBytes == demoted.diskBytes,"demotion reread SSD");
            // Repeated promotion/demotion must reuse backing pages and avoid
            // glibc heap trimming for canonical mapped GGUF records.
            SetMoeCpuCacheBytes(1ULL << 30); // Enable the bounded 64 MiB workspace.
            auto beforeReuse=GetDiskMoeCacheStats();
            for (int cycle=0; cycle<3; ++cycle) {
                SetMoeCudaCacheBytes(f.bytes*8);
                for (int pass=0; pass<16; ++pass) f.Run(true,1,pass%2,device,8);
                SetMoeCudaCacheBytes(0); TrimDiskMoeCache();
                Compare(f.Run(true,1,0,device,8),cpuReference,0);
            }
            auto reused=GetDiskMoeCacheStats();
            Check(reused.cpuBufferReuses>beforeReuse.cpuBufferReuses,"GGUF backing pages not reused");
            Check(reused.asyncMaintenance>beforeReuse.asyncMaintenance,"background maintenance unused");
            Check(reused.cpuReclaims==beforeReuse.cpuReclaims,"mapped GGUF required heap trim");
            Check(reused.cpuBytes+reused.cpuWorkspaceBytes<=GetMoeCpuCacheBytes(),"workspace exceeded RAM budget");
            SetMoeCpuCacheBytes(1); TrimDiskMoeCache();
            Check(GetDiskMoeCacheStats().cpuBytes==0,"tiny RAM budget trim");
            SetMoeCpuCacheBytes(0); TrimDiskMoeCache();
            SetMoeCpuCacheBytes(ramBudget); // Includes aligned direct-I/O padding.
            auto beforeRam = GetDiskMoeCacheStats();
            // Cold streamed prefill and then RAM hits both execute on CUDA.
            for (int pass=0; pass<2; ++pass) Compare(f.Run(true,260,0,device,8),f.Run(false,260,0,device,8),.012);
            auto ram=GetDiskMoeCacheStats(); Check(ram.cpuBytes>=f.bytes*8 && ram.cpuBytes<=GetMoeCpuCacheBytes() && ram.cpuHits>beforeRam.cpuHits,"RAM prefill cache unused");
            SetMoeCpuCacheBytes(f.bytes*2); SetMoeCudaCacheBytes(f.bytes*3); TrimDiskMoeCache();
            auto before=GetDiskMoeCacheStats();
            for (int pass=0; pass<20; ++pass) {
                // CPU Q8_K and CUDA Q8_1 have distinct dot quantization. Keep
                // the existing CUDA boundary exact and bound mixed-tier error.
                Compare(f.Run(true,5,pass%8,device),f.Run(false,5,pass%8,device),.06);
                auto s=GetDiskMoeCacheStats();
                Check(s.cpuBytes+s.cpuWorkspaceBytes<=f.bytes*2 && s.cudaBytes<=f.bytes*3,"cache budgets");
            }
            auto after=GetDiskMoeCacheStats();
            Check(after.cudaHits>before.cudaHits && after.misses>before.misses,"eviction coverage");
            if (pair.first==GGML_TYPE_IQ1_S) {
                // A competing stats/trim caller must drain, rather than race
                // with or deadlock the one pending maintenance job.
                SetMoeCpuCacheBytes(ramBudget*4); SetMoeCudaCacheBytes(f.bytes*5);
                std::atomic<bool> stop{false};
                std::exception_ptr error;
                std::thread observer([&] {
                    try { while (!stop.load()) { GetDiskMoeCacheStats(); TrimDiskMoeCache(); } }
                    catch (...) { error=std::current_exception(); }
                });
                try {
                    for (int pass=0; pass<24; ++pass) f.Run(true,1,pass%8,device,8);
                } catch (...) { stop=true; observer.join(); throw; }
                stop=true; observer.join();
                if (error) std::rethrow_exception(error);
                // Leave the last operation undrained: lazy-weight destruction
                // must finish any queued transfer before invalidating keys.
                f.Run(true,1,3,device,8);
            }
        }
    }
    {
        Fixture a(GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,0), b(GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,0);
        SetMoeCudaCacheBytes(0); SetMoeCpuCacheBytes(0); TrimDiskMoeCache();
        const uint64_t budget = PreloadRecordBytes(a)+PreloadRecordBytes(b);
        SetMoeCpuCacheBytes(budget);
        auto before = GetDiskMoeCacheStats();
        Check(PreloadDiskMoeCache({{a.disk[2],a.disk[3]}})==0,"preload included shared expert");
        ++a.disk[2]->diskWeightParts[0].bytes;
        Check(PreloadDiskMoeCache({{nullptr,nullptr,a.disk[2],a.disk[3]}})==0,"preload accepted oversized GGUF metadata");
        --a.disk[2]->diskWeightParts[0].bytes;
        Check(GetDiskMoeCacheStats().diskBytes==before.diskBytes,"invalid preload performed I/O");
        Check(PreloadDiskMoeCache({a.disk,b.disk,a.disk})==budget,"preload did not distribute across layers");
        auto loaded = GetDiskMoeCacheStats();
        a.Run(true,1,0,0,1); b.Run(true,1,0,0,1);
        Check(GetDiskMoeCacheStats().diskBytes==loaded.diskBytes,"preload starved later layer");
        SetMoeCpuCacheBytes(0); TrimDiskMoeCache();
    }
    {
        // GGUF formats without mapped cache buffers retain exact heap charges.
        Fixture f(GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,0);
        TestPreload(f,0,false);
    }
    {
        // Real decode dimensions exercise the large, huge-page-aligned
        // mappings and their ordinary-page tails during D2H demotion.
        Fixture f(GGML_TYPE_IQ1_S,GGML_TYPE_Q2_0,0,5120,2304);
        SetMoeCpuCacheBytes(0); SetMoeCudaCacheBytes(0); TrimDiskMoeCache();
        auto reference=f.Run(true,1,0,0,8);
        SetMoeCpuCacheBytes(1ULL<<30); SetMoeCudaCacheBytes(f.bytes*8);
        for (int pass=0; pass<16; ++pass) f.Run(true,1,pass%2,0,8);
        auto warm=GetDiskMoeCacheStats();
        Check(warm.cudaBytes==f.bytes*8,"large experts not admitted");
        Compare(f.Run(true,1,0,0,8),f.Run(false,1,0,0,8),0);
        SetMoeCudaCacheBytes(0); TrimDiskMoeCache();
        auto demoted=GetDiskMoeCacheStats();
        Check(demoted.cudaDemotions==warm.cudaDemotions+8,"large demotion missing");
        Compare(f.Run(true,1,0,0,8),reference,0);
        auto after=GetDiskMoeCacheStats();
        Check(after.diskBytes==demoted.diskBytes,"large demotion reread SSD");
        Check(after.cpuBytes+after.cpuWorkspaceBytes<=GetMoeCpuCacheBytes(),"large buffer budget");
    }
    auto s=GetDiskMoeCacheStats(); Check(!s.cpuBytes && !s.cudaBytes && !s.cpuWorkspaceBytes,"unload");
    puts("PASS: disk V4.1 GGUF cache, streamed prefill, mixed tiers, budgets and unload"); return 0;
} catch (const std::exception &e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; } }
