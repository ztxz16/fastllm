#include "fastllm.h"
#include "utils.h"
#include "devices/disk/diskdevice.h"
#include "moe_cache_config.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
static void Check(bool x, const char *s) { if (!x) throw std::runtime_error(s); }
struct Reference : CpuMergeMOE { using CpuMergeMOE::Run; };
struct Fixture {
    static constexpr int hidden=256, inter=256, experts=12;
    std::string path;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<std::vector<Data*>> memory, disk;
    uint64_t bytes=0;
    Fixture() : memory(2, std::vector<Data*>(2*(experts+1))), disk(memory) {
        char name[]="/tmp/fastllm-disk-glm-XXXXXX"; int fd=mkstemp(name); Check(fd>=0,"tempfile");
        path=name; FILE *f=fdopen(fd,"wb");
        for (int layer=0;layer<2;++layer) for(int e=0;e<experts;++e) for(int p=0;p<2;++p) {
            auto w=std::make_unique<Data>(NVFP4_BLOCK_16_E4M3);
            w->blockK=1;w->blockM=16;w->Resize(p?std::vector<int>{hidden,inter}:std::vector<int>{inter*2,hidden});
            w->scales=p?std::vector<float>{.09f}:std::vector<float>{.08f,.11f};w->Allocate(false);
            size_t payload=GetNVFP4WeightBytes(w->dims[0],w->dims[1]);
            for(size_t i=0;i<payload;++i) w->cpuData[i]=(i*37+e*17+p*3+layer*9)^(i>>7);
            for(size_t i=payload;i<w->GetBytes();++i) w->cpuData[i]=45+(i+e)%7;
            auto d=std::make_unique<Data>(w->dataType,w->dims);d->blockK=1;d->blockM=16;
            d->scales=w->scales;d->isDiskWeight=true;
            // Merged gate/up metadata: disjoint payload and scale parts, with
            // deliberately unaligned offsets and distinct tensor multipliers.
            int parts=p?1:2;
            for(int k=0;k<parts;++k) {
                char padding[37]={};fwrite(padding,1,sizeof(padding),f);
                DiskWeightPart data;data.fileName=path;data.fileOffset=ftell(f);
                data.dims={w->dims[0]/parts,w->dims[1]};data.sourceDataType=w->dataType;data.bytes=payload/parts;
                fwrite(w->cpuData+k*data.bytes,1,data.bytes,f);d->diskWeightParts.push_back(data);
                DiskWeightPart scale=data;scale.fileOffset=ftell(f);scale.isScalePart=true;
                scale.bytes=(w->GetBytes()-payload)/parts;scale.scaleOffset=k*scale.bytes;scale.sourceDataType=INT8;
                fwrite(w->cpuData+payload+scale.scaleOffset,1,scale.bytes,f);d->diskWeightParts.push_back(scale);
            }
            memory[layer][2+2*e+p]=w.get();disk[layer][2+2*e+p]=d.get();
            if(layer==0 && e==0) bytes+=GetDataBytes(NVFP4_BLOCK_16_E4M3_PACKED,w->dims[0],w->dims[1]);
            owned.push_back(std::move(w));owned.push_back(std::move(d));
        }
        fclose(f);
    }
    ~Fixture(){owned.clear();unlink(path.c_str());}
    std::vector<uint16_t> Run(bool lazy,int layer,int rows,int offset,int device=-1) {
        Data x(BFLOAT16,{rows,hidden}),y(BFLOAT16,{rows,hidden}),ids(INT32PARAM,{rows,3}),scores(FLOAT32,{rows,3});
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int i=0;i<rows*hidden;++i) {float v=std::sin(i*.71f)*.4f;uint32_t b;memcpy(&b,&v,4);((uint16_t*)x.cpuData)[i]=b>>16;}
        for(int r=0;r<rows;++r) for(int j=0;j<3;++j){
            ((int*)ids.cpuData)[r*3+j]=(offset+(j==2 && r%3==0?0:j*2)+r%2)%experts;
            ((float*)scores.cpuData)[r*3+j]=j==2?(r%3==0?.17f:0):.3f+j*.17f;
        }
#ifdef USE_CUDA
        if(device>=0){FastllmCudaSetDevice(device);x.ToDevice(CUDA,{device},true);x.ToDevice(CPU);}
#endif
        Data a,b,c;std::vector<Data*> bias(memory[layer].size());auto &table=lazy?disk[layer]:memory[layer];
        DataDict datas={{"input",&x},{"output",&y},{"index",&ids},{"score",&scores},{"weights",(Data*)table.data()},
            {"biass",(Data*)bias.data()},{"w1",&a},{"w2",&b},{"w3",&c}};
        IntDict ip={{"weights___batch",int(table.size())},{"biass___batch",int(bias.size())},{"deepSeekV4Mode",1},{"activationQuantBlock",128}};
        FloatDict fp={{"sharedScale",0},{"swigluLimit",2}};
        if(lazy){DiskMergeMOE op;op.Run("MergeMOE",datas,fp,ip);}else{
            y.Allocate(false);
            for(int r=0;r<rows;++r) {
                Data xr(BFLOAT16,{1,hidden},CPU,x.cpuData+size_t(r)*hidden*2);
                Data ir(INT32PARAM,{1,3},CPU,ids.cpuData+r*3*4);
                Data sr(FLOAT32,{1,3},CPU,scores.cpuData+r*3*4), yr(BFLOAT16,{1,hidden});
                auto row=datas;row["input"]=&xr;row["index"]=&ir;row["score"]=&sr;row["output"]=&yr;
                Reference op;op.Run("MergeMOE",row,fp,ip);
                memcpy(y.cpuData+size_t(r)*hidden*2,yr.cpuData,hidden*2);
            }
        }
        y.ToDevice(CPU);return std::vector<uint16_t>((uint16_t*)y.cpuData,(uint16_t*)y.cpuData+rows*hidden);
    }
};
static float Float(uint16_t x){uint32_t b=uint32_t(x)<<16;float f;memcpy(&f,&b,4);return f;}
static void Compare(const std::vector<uint16_t>&a,const std::vector<uint16_t>&b){
    Check(a.size()==b.size(),"shape");double err=0,norm=0;int mismatch=0;
    for(size_t i=0;i<a.size();++i){float x=Float(a[i]),y=Float(b[i]);Check(std::isfinite(x),"finite");err+=(x-y)*(x-y);norm+=y*y;mismatch+=a[i]!=b[i];}
    double rel=std::sqrt(err/std::max(norm,1e-30));
    if(rel>.006){fprintf(stderr,"relative=%.7g mismatch=%d/%zu\n",rel,mismatch,a.size());throw std::runtime_error("numerics");}
}
int main(int argc,char**argv){try{
    int devices=argc>1?atoi(argv[1]):0;
#ifndef USE_NUMAS
    return 77;
#endif
#ifndef USE_CUDA
    if(devices)return 77;
#else
    if(devices>FastllmCudaGetDeviceCount()){puts("FASTLLM_TEST_SKIP_NO_DEVICE");return 77;}
#endif
    SetThreads(8);MoeCacheConfig config;config.halfLife=8;SetMoeCacheConfig(config);
    for(bool tiny:{false,true}){
        Fixture f;SetMoeCpuCacheBytes(tiny?1:f.bytes*8);SetMoeCudaCacheBytes(devices?(tiny?1:f.bytes*4):0);
        Check(PrepareDiskMoeCache(f.disk),"prepare");
        for(int rows:{1,1,2,9,3,40,1,1,1,1,1,1})for(int l=0;l<2;++l){
            auto want=f.Run(false,l,rows,0);auto got=f.Run(true,l,rows,0,devices?l%devices:-1);Compare(got,want);
            auto s=GetDiskMoeCacheStats();Check(s.cpuBytes<=GetMoeCpuCacheBytes(),"CPU budget");
            Check(s.cudaBytes<=GetMoeCudaCacheBytes()*devices,"CUDA budget");
            Check(s.cpuCudaOverlapBytes==0,"host and GPU retained duplicate experts");
        }
        auto before=GetDiskMoeCacheStats();
        for(int i=0;i<45;++i)for(int l=0;l<2;++l)f.Run(true,l,1,6,devices?l%devices:-1);
        auto after=GetDiskMoeCacheStats();
        Check(after.cpuCudaOverlapBytes==0,"hot-set update introduced duplicate ownership");
        if(!tiny) Check(after.cpuHits+after.cudaHits>before.cpuHits+before.cudaHits,"no cache hits");
        for(int l=0;l<2;++l)Compare(f.Run(true,l,1,6,devices?l%devices:-1),f.Run(false,l,1,6));
        if(!tiny)Check(after.cpuEvictions+after.cudaEvictions>0,"no hot-set replacement");
        SetMoeCpuCacheBytes(0);SetMoeCudaCacheBytes(0);
        Check(GetDiskMoeCacheStats().cpuBytes==0 && GetDiskMoeCacheStats().cudaBytes==0,"budget reset retained payload");
        for(int l=0;l<2;++l)Compare(f.Run(true,l,1,6,devices?l%devices:-1),f.Run(false,l,1,6));
    }
    if(devices) {
        {
            Fixture f;SetMoeCpuCacheBytes(0);SetMoeCudaCacheBytes(f.bytes*4);
            Check(PrepareDiskMoeCache(f.disk),"prepare GPU-only cache");
            for(int i=0;i<8;++i)for(int l=0;l<2;++l)
                Compare(f.Run(true,l,1,0,l%devices),f.Run(false,l,1,0));
            auto s=GetDiskMoeCacheStats();
            Check(s.cpuBytes==0 && s.cudaHits>0,"zero RAM prevented GPU adaptation");
        }
        // Every expert fits across the two tiers, but neither tier alone can
        // hold the routed working set. After warmup, cycling that set must not
        // go back to disk, including when the GPU hot set changes.
        Fixture f;SetMoeCpuCacheBytes(f.bytes*20);SetMoeCudaCacheBytes(f.bytes*4);
        Check(PrepareDiskMoeCache(f.disk),"prepare hierarchy");
        for(int i=0;i<12;++i)for(int l=0;l<2;++l)f.Run(true,l,40,i%3*2,l%devices);
        auto warm=GetDiskMoeCacheStats();
        for(int i=0;i<40;++i)for(int l=0;l<2;++l)
            Compare(f.Run(true,l,1,i%3*2,l%devices),f.Run(false,l,1,i%3*2));
        auto after=GetDiskMoeCacheStats();
        Check(after.diskBytes==warm.diskBytes,"warm hierarchy reread experts during cache updates");
        Check(after.cudaDemotions>0 && after.cudaDemotionBytes>0,"GPU victims never reached RAM");
        Check(after.cpuCudaOverlapBytes==0,"hierarchy retains duplicate payloads");
        Check(after.cpuBytes<=GetMoeCpuCacheBytes(),"hierarchy RAM budget");
    }
    Check(GetDiskMoeCacheStats().cpuBytes==0 && GetDiskMoeCacheStats().cudaBytes==0,"unload retained cache");
    puts("PASS: disk GLM compact frequency cache");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
