#include "../../src/models/glm5_next_dsa.h"
#include "../../src/models/glm5_next_hybrid_graph.h"
#include "../../src/devices/cuda/fastllm-hybrid-prefill.cuh"
#include "devices/cuda/fastllm-cuda-hybrid-prefill.h"
#include "devices/cuda/fastllm-cuda-hybrid-kda.h"
#include "devices/cuda/fastllm-cuda-hybrid-dsa.h"
#include "devices/cuda/fastllm-cuda-hybrid-output.h"
#include "gguf.h"
#include "executor.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <fstream>
#include <set>
#include <stdexcept>
using namespace fastllm;
using namespace fastllm::glm5_next_detail;
namespace fastllm {
struct Glm5NextGGUFTestAccess {
    static void HybridGraphs(); static void HybridKda(); static void HybridMla();
    static void HybridDsaPreparation(); static void HybridRouter();
};
}
namespace {
void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
uint16_t Bf(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
float F(uint16_t x) { uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f; }
void Upload(Data &x, const std::vector<int> &dims, const std::vector<float> &v, DataType type = BFLOAT16) {
    x.dataType = type; x.UpdateUnitSize(); x.Resize(dims);
    int device = 0;
    Check(cudaGetDevice(&device) == cudaSuccess, "upload device");
    x.ToDevice(DataDevice::CUDA, {device}, false); x.Allocate();
    Check(x.Count(0) == v.size(), "upload shape");
    if (type == BFLOAT16) {
        std::vector<uint16_t> b(v.size());
        for (size_t i = 0; i < v.size(); ++i) b[i] = Bf(v[i]);
        FastllmCudaCopyFromHostToDevice(x.cudaData, b.data(), b.size()*2);
    } else FastllmCudaCopyFromHostToDevice(x.cudaData, (void*)v.data(), v.size()*4);
}
std::vector<float> Download(Data &x) {
    size_t count=1;for(int dim:x.dims)count*=dim;
    std::vector<float> result(count);
    if(x.dataType==FLOAT32){
        if(x.dataDevice==DataDevice::CUDA) Check(cudaMemcpy(result.data(),x.cudaData,count*4,cudaMemcpyDeviceToHost)==cudaSuccess,"download FP32");
        else std::memcpy(result.data(),x.cpuData,count*4);
    }else{
        Check(x.dataType==BFLOAT16,"download type");std::vector<uint16_t> b(count);
        if(x.dataDevice==DataDevice::CUDA) Check(cudaMemcpy(b.data(),x.cudaData,count*2,cudaMemcpyDeviceToHost)==cudaSuccess,"download BF16");
        else std::memcpy(b.data(),x.cpuData,count*2);
        for(size_t i=0;i<count;++i)result[i]=F(b[i]);
    }
    return result;
}
std::vector<int> Integers(Data &x) {
    std::vector<int> v(x.Count(0));
    FastllmCudaCopyFromDeviceToHost(v.data(), x.cudaData, v.size()*4);
    return v;
}
float Pattern(int t, int d) { return float((t*17+d*31+(t/7)*d)%113-56)/32; }
float E4M3(float x) {
    const float a = std::abs(x); float best = 0, distance = a;
    for (int i = 1; i <= 126; ++i) {
        const int e = i/8, m = i%8;
        const float value = e ? std::ldexp(1.f+m/8.f,e-7) : std::ldexp(float(m),-9);
        const float delta = std::abs(a-value);
        if (delta < distance || (delta == distance && !(i&1))) {
            best = value; distance = delta;
        }
    }
    return std::copysign(best,x);
}
std::vector<float> RotateReference(const std::vector<float> &x) {
    auto y = x;
    for (size_t base=0; base<x.size(); base+=128) {
        float peak=0;
        for (int d=0;d<128;++d) {
            double sum=0;
            for (int j=0;j<128;++j) sum += (__builtin_parity(d&j)?-1:1)*x[base+j];
            y[base+d]=F(Bf(sum/std::sqrt(128.)));
            peak=std::max(peak,std::abs(y[base+d]));
        }
        const float scale=std::exp2(std::ceil(std::log2(std::max(peak,1e-4f)/448)));
        for (int d=0;d<128;++d) y[base+d]=E4M3(y[base+d]/scale)*scale;
    }
    return y;
}
void NormAndQuantization() {
    std::vector<float> x(33*128), gamma(128), beta(128);
    for (int d=0;d<128;++d) { gamma[d]=1+d/256.f; beta[d]=(d%5-2)/128.f; }
    for (int i=0;i<(int)x.size();++i) x[i]=F(Bf(Pattern(i/128,i%128)*std::ldexp(1.f,i/128%8-4)));
    Data input,g,b,n; Upload(input,{33,128},x,FLOAT32); Upload(g,{128},gamma,FLOAT32);
    Upload(b,{128},beta,FLOAT32); Upload(n,{33,128},std::vector<float>(x.size()),FLOAT32);
    Check(FastllmCudaLayerNormWithEpsilon(input,g,b,n,1e-6f),"LayerNorm launch");
    auto got=Download(n); double err=0;
    for(int r=0;r<33;++r) {
        double mean=0,var=0;for(int d=0;d<128;++d)mean+=x[r*128+d]/128.;
        for(int d=0;d<128;++d)var+=std::pow(x[r*128+d]-mean,2)/128.;
        for(int d=0;d<128;++d)err=std::max(err,std::abs(got[r*128+d]-((x[r*128+d]-mean)/std::sqrt(var+1e-6)*gamma[d]+beta[d])));
    }
    Check(err<2e-5,"LayerNorm reference");
    Data bf,h,out;Upload(bf,{33,128},x);RotateAndQuantizeIndexer(bf,h,out);
    auto quant=Download(out),ref=RotateReference(x);
    Check(quant==ref,"Hadamard and UE8M0 FP8 reference");
    std::printf("PASS LayerNorm eps=1e-6 max_error=%.8g; Hadamard/FP8 exact\n",err);
}
void KpoolChunks() {
    constexpr int n=2053;std::vector<float> k(n*128),g(k.size()),a(4*128);
    for(int t=0;t<n;++t)for(int d=0;d<128;++d){k[t*128+d]=F(Bf(Pattern(t,d)));g[t*128+d]=F(Bf(Pattern(t+5,d)/2));}
    for(int i=0;i<512;++i)a[i]=F(Bf(Pattern(i/128,i%128)/4));
    Data ape;Upload(ape,{4,128},a,FLOAT32);
    Glm5NextIndexerCache full,chunked;Data key,gate;Upload(key,{1,n,128},k);Upload(gate,{1,n,128},g);
    AppendIndexerKeys(full,key,gate,ape);
    int start=0;for(int size:{1,2,4,17,1024,1005}){
        Data ck,cg;Upload(ck,{1,size,128},std::vector<float>(k.begin()+start*128,k.begin()+(start+size)*128));
        Upload(cg,{1,size,128},std::vector<float>(g.begin()+start*128,g.begin()+(start+size)*128));
        AppendIndexerKeys(chunked,ck,cg,ape);start+=size;
    }
    Check(start==n&&full.tokens==n&&chunked.tokens==n,"KPool token count");
    Check(Download(full.keys)==Download(chunked.keys),"KPool chunk continuity");
    Check(Download(full.tailKeys)==Download(chunked.tailKeys),"KPool tail continuity");
    std::vector<float> ref(n/4*128);
    for(int group=0;group<n/4;++group)for(int d=0;d<128;++d){
        double num=0,den=0;
        for(int j=0;j<4;++j){double w=std::exp(double(g[(group*4+j)*128+d]+a[j*128+d]));num+=w*k[(group*4+j)*128+d];den+=w;}
        ref[group*128+d]=F(Bf(num/den));
    }
    ref=RotateReference(ref);auto got=Download(full.keys);int different=0;double squared=0;
    for(size_t i=0;i<ref.size();++i){different+=ref[i]!=got[i];squared+=std::pow(ref[i]-got[i],2);}
    std::printf("KPool independent reference: differing=%d/%zu rmse=%.8g\n",different,ref.size(),std::sqrt(squared/ref.size()));
    Check(different<ref.size()/500&&std::sqrt(squared/ref.size())<.002,"KPool reference");
    std::puts("PASS KPool split chunks, incomplete tail, independent reference");
}
// A rejected suffix must be indistinguishable from a request which never saw
// it, including the next append across a KPool group / Top-2048 boundary.
void VerificationRollback() {
    constexpr int total=2080;
    std::vector<float> k(total*128),g(k.size()),a(512);
    for(int t=0;t<total;++t)for(int d=0;d<128;++d){k[t*128+d]=Pattern(t,d);g[t*128+d]=Pattern(t+11,d)/2;}
    for(int i=0;i<512;++i)a[i]=Pattern(i/128,i%128)/4;
    Data key,gate,ape;Upload(key,{1,total,128},k);Upload(gate,{1,total,128},g);Upload(ape,{4,128},a,FLOAT32);
    auto append=[&](Glm5NextIndexerCache &c,int start,int rows){
        if(!rows)return;
        Data x,y;Split(key,1,start,start+rows,x);Split(gate,1,start,start+rows,y);AppendIndexerKeys(c,x,y,ape);
    };
    auto equal=[&](Glm5NextIndexerCache &x,Glm5NextIndexerCache &y){
        Check(x.tokens==y.tokens,"rollback token count");
        for(auto pair:{std::make_pair(&x.keys,&y.keys),std::make_pair(&x.tailKeys,&y.tailKeys),std::make_pair(&x.tailGates,&y.tailGates)}){
            Check(pair.first->dims==pair.second->dims,"rollback tensor shape");
            if(!pair.first->dims.empty())Check(Download(*pair.first)==Download(*pair.second),"rollback changed accepted prefix");
        }
    };
    for(int past:{0,1,2,3,4,2045,2046,2047,2048})for(int rows:{2,4,7,9})for(int accepted=0;accepted<rows;++accepted){
        Glm5NextIndexerCache actual,expected;
        append(actual,0,past);append(expected,0,past+accepted);
        Glm5NextIndexerCheckpoint snapshot;CaptureIndexerCheckpoint(actual,snapshot,true);
        Split(key,1,past,past+rows,actual.replayKeys);Split(gate,1,past,past+rows,actual.replayGates);
        append(actual,past,rows);
        CommitIndexerPrefix(actual,snapshot,accepted,ape);equal(actual,expected);
        // Replace the rejected suffix with unrelated inputs, then cross more groups.
        append(actual,2064,11);append(expected,2064,11);equal(actual,expected);
    }
    std::puts("PASS DSA MTP rollback: all prefixes, 2/4/7/9 rows, KPool and Top-2048 boundaries");
}
void Selection(int past,int rows) {
    const int m=(past+rows)/4;Data scores,groups,indices;std::vector<float> s(rows*m);
    for(int r=0;r<rows;++r)for(int j=0;j<m;++j)s[r*m+j]=(j*37+r*13)%101-50;
    Upload(scores,{1,rows,m},s,FLOAT32);
    Check(FastllmCudaDeepSeekV41IndexerTopK(scores,nullptr,512,4,past,1,groups),"TopK launch");
    if (rows == 1) {
        Data fast;
        Check(FastllmCudaQwen4SelectBlocks(scores,512,past,4,fast),"decode radix selection");
        Check(Integers(fast)==Integers(groups),"decode radix selection order/ties");
    }
    groups.Reshape({rows,groups.dims.back()});
    Check(FastllmCudaQwen4ExpandSelectedBlocks(groups,past+rows,past,4,indices),"expand launch");
    auto got=Integers(indices);const int width=indices.dims.back();
    for(int r=0;r<rows;++r){
        const int visible=(past+r+1)/4;std::vector<int> order(visible);std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](int a,int b){return s[r*m+a]!=s[r*m+b]?s[r*m+a]>s[r*m+b]:a<b;});
        order.resize(std::min(512,visible));std::set<int> expected,actual;
        for(int group:order)for(int d=0;d<4;++d)expected.insert(group*4+d);
        for(int t=visible*4;t<=past+r;++t)expected.insert(t);
        int valid=0;for(int i=0;i<width;++i){int t=got[r*width+i];if(t>=0){Check(t<=past+r,"future token selected");actual.insert(t);++valid;}else Check(t==-1,"bad padding");}
        Check(actual==expected&&valid==(int)actual.size(),"TopK, tie, tail reference");
    }
    std::printf("PASS selection past=%d rows=%d width=%d\n",past,rows,width);
}
void Scoring(bool hybrid = false, int rows = 9) {
    const int m=520,heads=32;std::vector<float> q(rows*heads*128),k(m*128),w(rows*heads);
    for(int r=0;r<rows*heads;++r){w[r]=Pattern(r,3)/32;for(int d=0;d<128;++d)q[r*128+d]=F(Bf(std::ldexp(Pattern(r,d)/8, hybrid ? (r*17+d*13)%7-3 : 0)));}
    for(int j=0;j<m;++j)for(int d=0;d<128;++d)k[j*128+d]=F(Bf(std::ldexp(Pattern(j+8,d)/8, hybrid ? (j*19+d*11)%7-3 : 0)));
    Data qd,kd,wd,s;Upload(qd,{1,rows,heads,128},q);Upload(kd,{1,m,128},k);Upload(wd,{1,rows,heads},w,FLOAT32);
    Check(hybrid ? FastllmCudaGlm5NextHybridIndexerScore(qd,wd,kd,s) :
        FastllmCudaDeepSeekV41IndexerScore(qd,wd,kd,4,2050,s),"score launch");auto got=Download(s);double err=0;
    for(int r=0;r<rows;++r)for(int j=0;j<std::min(m,(2050+r+1)/4);++j){
        double expected=0;for(int h=0;h<heads;++h){double dot=0;for(int d=0;d<128;++d)dot+=q[(r*heads+h)*128+d]*k[j*128+d];expected+=std::max(0.,dot)*w[r*heads+h];}
        err=std::max(err,std::abs(got[r*m+j]-expected));
    }
    std::printf("PASS weighted ReLU scoring max_error=%.8g\n",err);Check(err<2e-5,"score reference");
    if (hybrid) {
        Data reference;
        Check(FastllmCudaDeepSeekV41IndexerScore(qd,wd,kd,4,2050,reference),"legacy score launch");
        Check(got == Download(reference),"hybrid indexer must preserve every FP32 score bit");
    }
}
struct BorrowedCache:Data{~BorrowedCache(){isPagedKVCache=false;}};
bool MlaSupported(DataType type) {
    int device = 0, sharedMemory = 0;
    Check(cudaGetDevice(&device) == cudaSuccess, "MLA device");
    Check(cudaDeviceGetAttribute(&sharedMemory, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                device) == cudaSuccess, "MLA shared memory");
    // FlashInfer's MLA dispatcher requires at least 92672 bytes per SM.
    return FastllmCudaFlashInferDataTypeSupported(type) && sharedMemory >= 92672;
}
void Attention(int past,int rows,bool fragmented,bool hybrid=false) {
    const int device=FastllmCudaGetDevice();
    const int heads=64,rank=512,pageLen=32,tokens=past+rows,pages=(tokens+31)/32;
    std::vector<float> kv((fragmented?pages*2:pages)*pageLen*rank),q(heads*rows*rank);
    std::vector<int> pagesIds(pages);for(int p=0;p<pages;++p)pagesIds[p]=fragmented?(pages-1-p)*2:p;
    for(int t=0;t<tokens;++t)for(int d=0;d<rank;++d)kv[(pagesIds[t/pageLen]*pageLen+t%pageLen)*rank+d]=F(Bf(Pattern(t,d)/8));
    for(int h=0;h<heads;++h)for(int r=0;r<rows;++r)for(int d=0;d<rank;++d)q[(h*rows+r)*rank+d]=F(Bf(Pattern(h+r,d)/8));
    Data scores,groups,idx;std::vector<float> sv(rows*(tokens/4));for(size_t i=0;i<sv.size();++i)sv[i]=Pattern(i/(tokens/4),i%(tokens/4));
    Upload(scores,{1,rows,tokens/4},sv,FLOAT32);
    Check(FastllmCudaDeepSeekV41IndexerTopK(scores,nullptr,512,4,past,1,groups),"attention groups");groups.Reshape({rows,groups.dims.back()});
    Check(FastllmCudaQwen4ExpandSelectedBlocks(groups,tokens,past,4,idx),"attention indices");auto ids=Integers(idx);const int width=idx.dims.back();
    PagedCacheManager pool;Upload(pool,{int(kv.size()/pageLen/rank),pageLen,1,rank},kv);
    BorrowedCache cache;cache.dataDevice=DataDevice::CUDA;cache.dataDeviceIds={device};cache.dataType=BFLOAT16;cache.Resize({1,tokens,rank});cache.isPagedKVCache=true;cache.pageLen=pageLen;cache.lastPageLen=(tokens-1)%pageLen+1;cache.pageIndex=pagesIds;cache.pagedKVCacheData=&pool;
    Data query,out;Upload(query,{heads,rows,rank},q);SparseLatentAttention(query,cache,idx,1.f/16,out,true,nullptr,hybrid);auto got=Download(out);double error=0;
    for(float v:got)Check(std::isfinite(v),"attention nonfinite");
    for(int h:{0,17,63})for(int r=0;r<rows;++r){
        std::vector<double> p(width);double den=0;
        for(int i=0;i<width;++i){int t=ids[r*width+i];if(t<0)continue;double dot=0;for(int d=0;d<rank;++d)dot+=q[(h*rows+r)*rank+d]*kv[(pagesIds[t/pageLen]*pageLen+t%pageLen)*rank+d];den+=(p[i]=std::exp(dot/16));}
        for(int d=0;d<rank;d+=17){double num=0;for(int i=0;i<width;++i){int t=ids[r*width+i];if(t>=0)num+=p[i]*kv[(pagesIds[t/pageLen]*pageLen+t%pageLen)*rank+d];}error=std::max(error,std::abs(got[(h*rows+r)*rank+d]-num/den));}
    }
    if (rows == 1) {
        Data table(INT32), physical, pagedQuery, pagedOutput;
        table.Resize({pages}); table.ToDevice(DataDevice::CUDA,{device},false); table.Allocate(false);
        FastllmCudaCopyFromHostToDevice(table.cudaData,pagesIds.data(),pages*4);
        Check(FastllmCudaQwen4ExpandSelectedBlocks(groups,tokens,past,4,physical,&table,pageLen),"paged expand");
        auto mapped=Integers(physical);
        for(size_t i=0;i<ids.size();++i)
            Check(mapped[i]==(ids[i]<0?-1:pagesIds[ids[i]/pageLen]*pageLen+ids[i]%pageLen),"physical page mapping");
        PagedCacheManager pePool;
        Upload(pePool,{int(kv.size()/pageLen/rank),pageLen,1,64},std::vector<float>(kv.size()/rank*64));
        BorrowedCache peCache;peCache.dataDevice=DataDevice::CUDA;peCache.dataDeviceIds={device};peCache.dataType=BFLOAT16;peCache.Resize({1,tokens,64});
        peCache.isPagedKVCache=true;peCache.pageLen=pageLen;peCache.lastPageLen=cache.lastPageLen;
        peCache.pageIndex=pagesIds;peCache.pagedKVCacheData=&pePool;
        Upload(pagedQuery,{heads,rows,rank},q);
        SparseLatentAttention(pagedQuery,cache,physical,1.f/16,pagedOutput,false,&peCache,hybrid);
        Check(Download(pagedOutput)==got,"paged BF16 fallback differs");
        if (MlaSupported(BFLOAT16)) {
            Upload(pagedQuery,{heads,rows,rank},q);
            Data qPe,direct;
            Upload(qPe,{1,1,heads,64},std::vector<float>(heads*64));
            Upload(direct,{heads,1,rank},std::vector<float>(heads*rank));
            Check(FastllmCudaMLAPaged(pagedQuery,qPe,peCache,cache,direct,
                1.f/16,2048+tokens%4,&physical),"FlashInfer physical-page dispatch");
            SparseLatentAttention(pagedQuery,cache,physical,1.f/16,pagedOutput,true,&peCache);
            auto flash=Download(pagedOutput); double diff=0,sq=0;
            Check(flash==Download(direct),"physical-page wrapper did not use MLA");
            for(size_t i=0;i<got.size();++i){diff=std::max(diff,double(std::abs(flash[i]-got[i])));sq+=(flash[i]-got[i])*(flash[i]-got[i]);}
            Check(diff<.004 && std::sqrt(sq/got.size())<.001,"paged MLA differs");
            std::printf("PASS physical-page MLA max_diff=%.8g rmse=%.8g\n",diff,std::sqrt(sq/got.size()));
        }
    }
    std::printf("PASS sparse latent attention past=%d rows=%d fragmented=%d max_error=%.8g\n",past,rows,fragmented,error);
    Check(error<.003,"attention independent reference");
}
bool Sm75VerificationAttention() {
    cudaDeviceProp prop{};
    int device = 0;
    Check(cudaGetDevice(&device) == cudaSuccess, "attention device");
    Check(cudaGetDeviceProperties(&prop, device) == cudaSuccess, "attention properties");
    if (prop.major != 7 || prop.minor != 5) return false;
    int checks = 0;
    for (int heads : {32, 64}) for (int rows : {1, 2, 3, 4, 8, 9})
        for (int width : {32, 517, 2051, 4096}) {
            const int cap = width + rows + 23;
            std::vector<float> q(rows * heads * 512), kv(cap * 512), sink(heads);
            std::vector<int> ids(rows * width);
            for (int i = 0; i < (int)q.size(); ++i) q[i] = Pattern(i / 512, i % 512) / 8;
            for (int i = 0; i < (int)kv.size(); ++i) kv[i] = Pattern(i / 512 + 11, i % 512) / 8;
            for (int h = 0; h < heads; ++h) sink[h] = Pattern(h, 7);
            for (int r = 0; r < rows; ++r) for (int c = 0; c < width; ++c)
                ids[r * width + c] = c % 19 == 0 ? -1 : (c * 17 + r * 13) % (cap - rows + r + 1);
            // Include fully masked rows as well as per-row tails and duplicate candidates.
            if (rows > 1 && width == 32) std::fill(ids.begin(), ids.begin() + width, -1);
            Data query, cache, chunk, indices(INT32), sinks, reference, actual;
            Upload(query, {1, rows, heads, 512}, q);
            Upload(cache, {1, cap, 512}, kv);
            Upload(chunk, {1, rows, 512}, std::vector<float>(rows * 512));
            Upload(sinks, {heads}, sink, FLOAT32);
            indices.Resize({1, rows, width});
            indices.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            indices.Allocate();
            FastllmCudaCopyFromHostToDevice(indices.cudaData, ids.data(), ids.size() * sizeof(int));
            // Candidate indices already encode causality. Padding unused query
            // rows selects the scalar fallback without a production debug flag.
            const int referenceRows = rows + 8;
            q.resize(referenceRows * heads * 512, 0);
            ids.resize(referenceRows * width, -1);
            Data referenceQuery, referenceChunk, referenceIndices(INT32);
            Upload(referenceQuery, {1, referenceRows, heads, 512}, q);
            Upload(referenceChunk, {1, referenceRows, 512}, std::vector<float>(referenceRows * 512));
            referenceIndices.Resize({1, referenceRows, width});
            referenceIndices.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            referenceIndices.Allocate();
            FastllmCudaCopyFromHostToDevice(referenceIndices.cudaData, ids.data(), ids.size() * sizeof(int));
            Check(FastllmCudaDeepSeekV41SparseAttention(referenceQuery, referenceChunk, nullptr, &cache,
                &referenceIndices, sinks, 0, 0, 1.f / 16, reference), "scalar attention reference");
            auto expected = Download(reference);
            expected.resize(rows * heads * 512);
            Check(FastllmCudaDeepSeekV41SparseAttention(query, chunk, nullptr, &cache,
                &indices, sinks, 0, 0, 1.f / 16, actual), "verification attention launch");
            Check(Download(actual) == expected, "SM75 verification attention changed BF16 output");
            ++checks;
        }
    std::printf("PASS SM75 verification attention device=%d: %d exact BF16 checks\n", device, checks);
    return true;
}
void MlaPlanCache(int device, int heads, int rank, DataType type) {
    FastllmCudaSetDevice(device);
    static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
    if (!MlaSupported(type)) {
        std::printf("SKIP MLA plan cache device=%d dtype=%d: unsupported datatype or shared memory\n",
                    device, int(type));
        return;
    }
    constexpr int pageLen = 16, pages = 8, tokens = pageLen * pages;
    std::vector<int> pageIds(pages), indices(tokens);
    std::vector<float> kv(2 * tokens * rank), pe(2 * tokens * 64), query(heads * rank), queryPe(heads * 64);
    for (int p = 0; p < pages; ++p) pageIds[p] = 2 * (pages - 1 - p);
    for (int i = 0; i < (int)kv.size(); ++i) kv[i] = Pattern(i / rank, i % rank) / 8;
    for (int i = 0; i < (int)pe.size(); ++i) pe[i] = Pattern(i / 64, i % 64) / 16;
    for (int i = 0; i < (int)query.size(); ++i) query[i] = Pattern(i / rank, i % rank) / 8;
    for (int i = 0; i < (int)queryPe.size(); ++i) queryPe[i] = Pattern(i / 64, i % 64) / 16;
    PagedCacheManager kvPool, pePool;
    Data q, qp, output, reference, physical(INT32);
    Upload(kvPool, {2 * pages, pageLen, 1, rank}, kv);
    Upload(pePool, {2 * pages, pageLen, 1, 64}, pe);
    Upload(q, {heads, 1, rank}, query);
    Upload(qp, {1, 1, heads, 64}, queryPe);
    Upload(output, {heads, 1, rank}, std::vector<float>(heads * rank));
    Upload(reference, {heads, 1, rank}, std::vector<float>(heads * rank));
    if (type == FLOAT16) {
        for (Data *x : {static_cast<Data*>(&kvPool), static_cast<Data*>(&pePool), &q, &qp, &output, &reference})
            ToDataType(*x, FLOAT16);
    }
    BorrowedCache cache, peCache;
    for (BorrowedCache *x : {&cache, &peCache}) {
        x->dataType = type;
        x->dataDevice = DataDevice::CUDA;
        x->dataDeviceIds = {device};
        x->Resize({1, tokens, x == &cache ? rank : 64});
        x->isPagedKVCache = true; x->pageLen = pageLen; x->lastPageLen = pageLen;
        x->pagedKVCacheData = x == &cache ? &kvPool : &pePool;
    }
    physical.Resize({tokens}); physical.ToDevice(DataDevice::CUDA, {device}, false); physical.Allocate(false);
    auto readBits = [&](Data &x) {
        std::vector<uint16_t> bits(heads * rank);
        Check(cudaMemcpy(bits.data(), x.cudaData, bits.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess, "MLA bits");
        return bits;
    };
    auto run = [&](int length, bool sparse, Data &out) {
        Check(FastllmCudaMLAPaged(q, qp, peCache, cache, out, 1.f / 16, length,
                                sparse ? &physical : nullptr), "MLA cache dispatch");
        return readBits(out);
    };
    for (int mapping = 0; mapping < 2; ++mapping) {
        if (mapping) std::reverse(pageIds.begin(), pageIds.end());
        cache.pageIndex = peCache.pageIndex = pageIds;
        for (int i = 0; i < tokens; ++i) indices[i] = pageIds[i / pageLen] * pageLen + i % pageLen;
        FastllmCudaCopyFromHostToDevice(physical.cudaData, indices.data(), indices.size() * 4);
        std::vector<std::vector<uint16_t>> expected;
        for (int tail = 0; tail < 4; ++tail) {
            auto first = run(64 + tail, true, output);
            // The non-sparse path overwrites shared integer staging and gives
            // an uncached reference over exactly the same physical tokens.
            Check(first == run(64 + tail, false, reference), "MLA cached vs uncached bits");
            run(17, false, reference);
            Check(first == run(64 + tail, true, output), "MLA plan survived workspace reuse");
            expected.push_back(std::move(first));
        }
        // Exceed the bounded cache, then revisit all four decode tail shapes.
        for (int length = 1; length <= 20; ++length) run(length, true, output);
        for (int tail = 0; tail < 4; ++tail)
            Check(expected[tail] == run(64 + tail, true, output), "MLA plan eviction/rebuild bits");
    }
    std::printf("PASS MLA plan cache device=%d heads=%d rank=%d dtype=%d: tails, changed indices, workspace reuse, eviction exact\n",
                device, heads, rank, int(type));
}
void DecodeRouter() {
    for (int mode=0; mode<4; ++mode) for (bool norm:{false,true}) {
        std::vector<float> x(288),bias(288);
        for(int i=0;i<288;++i){
            x[i]=mode==0?float((i*73)%293)/293: mode==1?.5f:float(i%7)/8;
            bias[i]=mode==2?float(i%3)/16:0;
        }
        if(mode==3) for(int i=0;i<288;i+=17) x[i]=std::numeric_limits<float>::quiet_NaN();
        Data logits,twice,b,id,score,refId,refScore;
        Upload(logits,{1,288},x,FLOAT32);
        auto xx=x;xx.insert(xx.end(),x.begin(),x.end());Upload(twice,{2,288},xx,FLOAT32);
        Upload(b,{288},bias,FLOAT32);Upload(id,{1,8},std::vector<float>(8),INT32);
        Upload(score,{1,8},std::vector<float>(8),FLOAT32);
        Upload(refId,{2,8},std::vector<float>(16),INT32);Upload(refScore,{2,8},std::vector<float>(16),FLOAT32);
        Check(FastllmCudaSelectExpert(logits,&b,id,score,8,norm,1.75f),"decode router");
        Check(FastllmCudaSelectExpert(twice,&b,refId,refScore,8,norm,1.75f),"reference router");
        auto ids=Integers(id),refs=Integers(refId);auto weights=Download(score),expected=Download(refScore);
        Check(std::equal(ids.begin(),ids.end(),refs.begin()),"router ties/indices differ");
        Check(std::equal(weights.begin(),weights.end(),expected.begin()),"router scores differ");
    }
    std::puts("PASS 288-expert decode router: generic equality, ties, bias, NaN, normalization");
}
void DecodeHc() {
    constexpr int dim=4096,flat=4*dim,mix=24;
    std::vector<float> x(flat),fn(mix*flat),scale={.8f,.6f,.7f},base(mix),norm(dim);
    for(int i=0;i<flat;++i)x[i]=Pattern(i/128,i%128);
    for(int i=0;i<mix*flat;++i)fn[i]=Pattern(i/flat,i%flat)/256;
    for(int i=0;i<mix;++i)base[i]=Pattern(i,3)/16;
    for(int i=0;i<dim;++i)norm[i]=1+Pattern(i,4)/16;
    Data input,w,sc,ba,n,mixed,post,comb,reference,out,fastPost,fastComb;
    Upload(input,{1,1,4,dim},x);Upload(w,{mix,flat},fn);Upload(sc,{3},scale,FLOAT32);
    Upload(ba,{mix},base,FLOAT32);Upload(n,{dim},norm,FLOAT32);
    Check(FastllmCudaDeepSeekV4HcPre(input,w,sc,ba,4,20,1e-6f,1e-6f,mixed,post,comb),"HC reference");
    KimiK3RMSNorm(mixed,n,1e-6f,reference);
    Check(FastllmCudaGlm5NextHcPreNorm(input,w,sc,ba,n,4,20,1e-6f,1e-6f,out,fastPost,fastComb),"HC fusion");
    auto a=Download(reference),b=Download(out);
    for(float value:b) Check(std::isfinite(value),"HC nonfinite");
    Check(a==b,"HC fused RMSNorm reference must preserve GLM rounding");
    for(auto pair:{std::make_pair(&post,&fastPost),std::make_pair(&comb,&fastComb)}){
        auto a=Download(*pair.first),b=Download(*pair.second);
        for(size_t i=0;i<a.size();++i)Check(std::abs(a[i]-b[i])<2e-5,"HC mixing differs");
    }
    std::puts("PASS HC fusion preserves GLM RMSNorm rounding exactly");
    const char *flag = "FASTLLM_DSV4_REFERENCE_HC_PRE_FINISH";
    const char *previous = std::getenv(flag);
    const bool hadPrevious = previous != nullptr;
    const std::string previousValue = previous ? previous : "";
    for (int rows = 1; rows <= 8; ++rows) for (auto fnType : {FLOAT32, BFLOAT16}) {
        std::vector<float> values(rows * flat);
        for (int i = 0; i < rows * flat; ++i) values[i] = Pattern(i / 128, i % 128);
        Data caseInput, caseWeight;
        Upload(caseInput, {1, rows, 4, dim}, values);
        Upload(caseWeight, {mix, flat}, fn, fnType);
        for (int iterations : {1, 7, 20}) {
            Data slow[3], fast[3];
            setenv(flag, "1", 1);
            Check(FastllmCudaGlm5NextHcPreNorm(caseInput,caseWeight,sc,ba,n,4,iterations,1e-6f,1e-6f,
                slow[0],slow[1],slow[2]), "HC reference finish");
            unsetenv(flag);
            Check(FastllmCudaGlm5NextHcPreNorm(caseInput,caseWeight,sc,ba,n,4,iterations,1e-6f,1e-6f,
                fast[0],fast[1],fast[2]), "HC shuffle finish");
            for (int i = 0; i < 3; ++i) {
                const auto reference = Download(slow[i]), actual = Download(fast[i]);
                Check(reference.size() == actual.size() &&
                    std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0,
                    "HC shuffle changed rounding");
            }
        }
    }
    if (hadPrevious) setenv(flag, previousValue.c_str(), 1); else unsetenv(flag);
    std::puts("PASS HC shuffle exact: 1..8 rows, FP32/BF16 weights, 1/7/20 Sinkhorn iterations");
}
void DecodeSmallGemv() {
    for(int width:{8,64,128,256}){
        constexpr int rows=8192;
        std::vector<float> x(width),w(rows*width),bias(rows);
        for(int i=0;i<width;++i)x[i]=Pattern(i,3)/4;
        for(int i=0;i<rows*width;++i)w[i]=Pattern(i/width,i%width)/4;
        for(int i=0;i<rows;++i)bias[i]=Pattern(i,7)/16;
        Data input,weight,b,output,refInput,reference;
        Upload(input,{1,width},x);Upload(weight,{rows,width},w);Upload(b,{rows},bias,FLOAT32);
        auto xx=x;xx.insert(xx.end(),x.begin(),x.end());Upload(refInput,{2,width},xx);
        Linear(input,weight,b,output);Linear(refInput,weight,b,reference);
        auto got=Download(output),expected=Download(reference);
        Check(std::equal(got.begin(),got.end(),expected.begin()),"short GEMV changed reduction");
    }
    std::puts("PASS short BF16 GEMV exact against original 256-thread multirow path");
}

void HybridHcPreNorm() {
    constexpr int dim = 4096, flat = 4 * dim, mix = 24;
    std::vector<float> fn(mix * flat), scale = {.8f, .6f, .7f}, base(mix), norm(dim);
    for (int i = 0; i < mix * flat; ++i) fn[i] = Pattern(i / flat, i % flat) / 256;
    for (int i = 0; i < mix; ++i) base[i] = Pattern(i, 3) / 16;
    for (int i = 0; i < dim; ++i) norm[i] = 1 + Pattern(i, 4) / 16;
    for (int rows : {1, 2, 4, 8, 64, 129, 512}) for (auto type : {FLOAT32, BFLOAT16}) {
        for (int variant = 0; variant < (rows == 1 ? 4 : 1); ++variant) {
            const int iterations[] = {20, 1, 2, 37};
            std::vector<float> x(rows * flat);
            for (int i = 0; i < rows * flat; ++i)
                x[i] = variant == 3 ? 0.0f : std::ldexp(Pattern(i / 128, i % 128),
                    variant == 1 ? -7 : variant == 2 ? 7 : 0);
            Data input, w, sc, ba, n, mixed, post, comb, reference, out, fastPost, fastComb;
            Upload(input, {1, rows, 4, dim}, x); Upload(w, {mix, flat}, fn, type);
            Upload(sc, {3}, scale, FLOAT32); Upload(ba, {mix}, base, FLOAT32); Upload(n, {dim}, norm, FLOAT32);
            Check(FastllmCudaDeepSeekV4HcPre(input, w, sc, ba, 4, iterations[variant], 1e-6f, 1e-6f, mixed, post, comb), "HC reference");
            KimiK3RMSNorm(mixed, n, 1e-6f, reference);
            Check(FastllmCudaGlm5NextHybridHcPreNorm(input, w, sc, ba, n, 4, iterations[variant], 1e-6f, 1e-6f, out, fastPost, fastComb), "HC hybrid pre-norm");
            Check(Download(reference) == Download(out), "HC hybrid pre-norm output differs");
            Check(Download(post) == Download(fastPost), "HC hybrid pre-norm post differs");
            Check(Download(comb) == Download(fastComb), "HC hybrid pre-norm comb differs");
        }
    }
    std::puts("PASS hybrid HC pre-norm: output, post and comb exact");
}

bool HybridSm75Attention() {
    int major = 0, minor = 0, device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (major != 7 || minor != 5) return false;
    for (int heads : {32, 64}) for (int rows : {1, 2, 4, 8, 16, 17, 32, 33, 64, 512}) {
        const int count = rows == 2 ? 33 : 2051, cap = 2309;
        std::vector<float> q(rows * heads * 512), kv(cap * 512);
        for (size_t i = 0; i < q.size(); ++i) q[i] = F(Bf(std::ldexp(Pattern(i / 512, i % 512) / 4, int(i * 13 % 7) - 3)));
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = F(Bf(std::ldexp(Pattern(i / 512 + 41, i % 512) / 4, int(i * 19 % 7) - 3)));
        std::vector<int> indices(rows * count);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < count; ++c)
            indices[r * count + c] = c % 29 == 0 ? -1 : (c * 13 + r * 17) % cap;
        Data query, latent, idx(INT32), output, reference, sink;
        Upload(query, {1, rows, heads, 512}, q); Upload(latent, {1, cap, 512}, kv);
        Upload(sink, {heads}, std::vector<float>(heads, -INFINITY), FLOAT32);
        idx.Resize({1, rows, count}); idx.ToDevice(DataDevice::CUDA, {device}, false); idx.Allocate(false);
        FastllmCudaCopyFromHostToDevice(idx.cudaData, indices.data(), indices.size() * sizeof(int));
        Check(FastllmCudaGlm5NextHybridSparseAttention(query, latent, idx, 1.f / 16, output), "hybrid attention launch");
        auto got = Download(output); double maximum = 0, referenceError = 0;
        if (rows == 1) {
            auto padded = kv; padded.insert(padded.begin(), 0.f);
            Data backing, unaligned, fallback;
            Upload(backing, {1, int(padded.size())}, padded);
            unaligned.FakeFrom(backing, sizeof(uint16_t)); unaligned.Resize(latent.dims);
            Check(FastllmCudaGlm5NextHybridSparseAttention(query, unaligned, idx, 1.f / 16, fallback),
                "hybrid unaligned KV fallback launch");
            Check(Download(fallback) == got, "hybrid unaligned KV fallback bits");
        }
        for (float x : got) Check(std::isfinite(x), "hybrid nonfinite");
        for (int r : {0, rows - 1}) for (int h : {0, heads - 1}) {
            std::vector<double> p(count); double den = 0;
            for (int c = 0; c < count; ++c) if (indices[r * count + c] >= 0) {
                double dot = 0;
                for (int d = 0; d < 512; ++d) dot += q[(r * heads + h) * 512 + d] * double(kv[indices[r * count + c] * 512 + d]);
                den += p[c] = std::exp(dot / 16);
            }
            for (int d = 0; d < 512; d += 17) {
                double num = 0;
                for (int c = 0; c < count; ++c) if (indices[r * count + c] >= 0)
                    num += p[c] * kv[indices[r * count + c] * 512 + d];
                const double expected = num / den;
                const double error = std::abs(got[(r * heads + h) * 512 + d] - expected);
                Check(error <= 1e-6 + std::abs(expected) / 250, "hybrid FP64 BF16 rounding bound");
                referenceError = std::max(referenceError, error);
            }
        }
        if (rows <= 64) {
            Check(FastllmCudaDeepSeekV41SparseAttention(query, latent, nullptr, &latent, &idx, sink, 0, 0, 1.f / 16, reference), "reference attention launch");
            auto before = Download(reference);
            for (size_t i = 0; i < got.size(); ++i) maximum = std::max(maximum, std::abs(double(got[i]) - before[i]));
            Check(before == got, "hybrid attention must preserve every BF16 output bit");
        }
        std::printf("PASS hybrid sparse device=%d heads=%d rows=%d old_max=%.8g fp64_max=%.8g\n", device, heads, rows, maximum, referenceError);
    }
    return true;
}

void HybridBf16Prefill() {
    constexpr int rows = 1024, columns = 136, width = 128, stride = 136;
    std::vector<float> x(rows * columns), w(width * columns);
    for (int i = 0; i < int(x.size()); ++i) x[i] = F(Bf(Pattern(i / columns, i % columns)));
    for (int i = 0; i < int(w.size()); ++i) w[i] = F(Bf(Pattern(i / columns + 23, i % columns)));
    Data input, weight, output;
    Upload(input, {rows, columns}, x);
    Upload(weight, {width, columns}, w);
    Upload(output, {rows, stride}, std::vector<float>(rows * stride, 19.f));
    auto handle = getFastllmCublasHandle();
    auto run = [&]() {
        return FastllmCudaHybridBf16Gemm(static_cast<const __nv_bfloat16 *>(input.cudaData),
            static_cast<const __nv_bfloat16 *>(weight.cudaData),
            static_cast<__nv_bfloat16 *>(output.cudaData), rows, columns, width, stride, handle);
    };
    Check(!run(), "hybrid GEMM must not change the default GPU policy");
    {
        FastllmCudaHybridPrefillScope scope(true);
        Check(!FastllmCudaHybridBf16Gemm(static_cast<const __nv_bfloat16 *>(input.cudaData) + 1,
            static_cast<const __nv_bfloat16 *>(weight.cudaData),
            static_cast<__nv_bfloat16 *>(output.cudaData), rows, columns, width, stride, handle),
            "hybrid GEMM unaligned input fallback");
        Check(!FastllmCudaHybridBf16Gemm(static_cast<const __nv_bfloat16 *>(input.cudaData),
            static_cast<const __nv_bfloat16 *>(weight.cudaData) + 1,
            static_cast<__nv_bfloat16 *>(output.cudaData), rows, columns, width, stride, handle),
            "hybrid GEMM unaligned weight fallback");
        Check(run(), "hybrid BF16 GEMM launch");
        auto got = Download(output);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < width; ++col) {
                double expected = 0;
                for (int k = 0; k < columns; ++k)
                    expected += double(x[row * columns + k]) * w[col * columns + k];
                Check(std::abs(got[row * stride + col] - expected) <=
                    1e-5 + std::abs(expected) / 250, "hybrid GEMM FP64 BF16 rounding bound");
            }
            for (int col = width; col < stride; ++col)
                Check(got[row * stride + col] == 19.f, "hybrid GEMM output stride");
        }
        {
            FastllmCudaHybridPrefillScope disabled(false);
            Check(!run(), "nested hybrid scope disable");
        }
        Check(run(), "nested hybrid scope restore");
        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
            "hybrid GEMM capture begin");
        Check(!run(), "hybrid GEMM must leave graph capture to the existing path");
        cudaGraph_t graph;
        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "hybrid GEMM capture end");
        cudaGraphDestroy(graph);
        // One vector spans too wide an exponent range for lossless FP16.
        x[0] = F(Bf(1e-30f));
        Upload(input, {rows, columns}, x);
        Check(!run(), "hybrid GEMM exponent-range fallback");
    }
    Check(!run(), "hybrid scope must restore the default policy");
    std::puts("PASS hybrid BF16 prefill FP64 reference, stride, scope and fallbacks");
}

void HybridHcPostPrefill() {
    constexpr int rows = 1024, dim = 256;
    for (bool wide : {false, true}) {
        std::vector<float> x(rows * dim), r(rows * 4 * dim), p(rows * 4), c(rows * 16);
        auto fill = [&](std::vector<float> &v, int seed) {
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = Pattern(i / dim + seed, i % dim);
                if (wide) v[i] = std::ldexp(v[i], int(i * 17 % 180) - 90);
            }
        };
        fill(x, 11); fill(r, 23); fill(p, 31); fill(c, 47);
        Data input, residual, post, comb, reference, output;
        Upload(input, {1, rows, dim}, x); Upload(residual, {1, rows, 4, dim}, r);
        Upload(post, {rows, 4}, p, FLOAT32); Upload(comb, {rows, 4, 4}, c, FLOAT32);
        Check(FastllmCudaDeepSeekV4HcPostCudaMix(input, residual, post, comb,
            1, rows, 4, dim, reference), "HC post reference");
        {
            FastllmCudaHybridPrefillScope scope(true);
            Check(FastllmCudaDeepSeekV4HcPostCudaMix(input, residual, post, comb,
                1, rows, 4, dim, output), "hybrid HC post");
        }
        std::vector<uint16_t> expected(rows * 4 * dim), actual(expected.size());
        FastllmCudaCopyFromDeviceToHost(expected.data(), reference.cudaData, expected.size() * 2);
        FastllmCudaCopyFromDeviceToHost(actual.data(), output.cudaData, actual.size() * 2);
        Check(actual == expected, "hybrid HC post must preserve every BF16 output bit");
    }
    std::puts("PASS hybrid HC post, FP64-equivalent BF16 output including extreme exponents");
}

void Fixture(const std::string &dir) {
    auto load = [&](const std::string &name, Data &out, DataType type) {
        std::ifstream f(dir+"/"+name+".bin", std::ios::binary);
        Check(bool(f), "fixture file missing"); int ndim=0;f.read((char*)&ndim,4);
        Check(ndim>0&&ndim<5,"fixture ndim");std::vector<int> dims(ndim);f.read((char*)dims.data(),ndim*4);
        size_t count=1;for(int d:dims)count*=d;std::vector<float> values(count);f.read((char*)values.data(),count*4);
        Check(bool(f),"fixture truncated");Upload(out,dims,values,type);
    };
    auto dump = [&](const std::string &name, Data &x) {
        std::ofstream f(dir+"/"+name+".out",std::ios::binary);
        if(x.dataType==INT32){auto v=Integers(x);f.write((char*)v.data(),v.size()*4);}
        else{auto v=Download(x);f.write((char*)v.data(),v.size()*4);}
    };
    WeightMap weight;
    for(const char *name:{"wq_b.weight","wk.weight","k_norm.weight","k_norm.bias","weights_proj.weight","index_kpool_compress_gate","index_kpool_compress_ape"}){
        std::string n(name);weight.weight.try_emplace(n);
        load(n,weight[n],n.find("norm")!=std::string::npos||n.find("weights_proj")!=std::string::npos||n.find("ape")!=std::string::npos?FLOAT32:BFLOAT16);
    }
    Data input,q;load("input",input,BFLOAT16);load("qnorm",q,BFLOAT16);Glm5NextIndexerCache cache;
    for(int begin=0;begin<input.dims[1];begin+=1024){
        int end=std::min(begin+1024,input.dims[1]);Data x,qn,ids;
        Split(input,1,begin,end,x);Split(q,1,begin,end,qn);
        if(begin==0){
            Data key,gate,kf,n,query,h,rot;
            Linear(x,weight["wk.weight"],Data(),key);dump("raw-key",key);
            Linear(x,weight["index_kpool_compress_gate"],Data(),gate);dump("gate",gate);
            ToDataType(key,kf,FLOAT32);Upload(n,key.dims,std::vector<float>(key.Count(0)),FLOAT32);
            Check(FastllmCudaLayerNormWithEpsilon(kf,weight["k_norm.weight"],weight["k_norm.bias"],n,1e-6f),"fixture norm");
            ToDataType(n,BFLOAT16);dump("normalized-key",n);
            Linear(qn,weight["wq_b.weight"],Data(),query);dump("raw-query",query);
            query.Reshape({1,end-begin,32,128});RotateAndQuantizeIndexer(query,h,rot);dump("rotated-query",rot);
        }
        BuildDsaIndices(x,qn,weight,"",begin,2048,cache,ids);
        if(!ids.dims.empty())dump("indices-"+std::to_string(begin),ids);
    }
    dump("pooled-keys",cache.keys);std::puts("PASS real checkpoint fixture execution");
}
}
void fastllm::Glm5NextGGUFTestAccess::HybridRouter() {
    const int device = FastllmCudaGetDevice();
    auto bytes = [&](Data &x) {
        std::vector<unsigned char> b(x.GetBytes());
        Check(cudaMemcpy(b.data(), x.cudaData, b.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "router download");
        return b;
    };
    for (int experts : {64, 128, 256, 512}) {
        // Each shape owns its immutable weight and the Linear reference's
        // cached zero bias, just as model weights do during inference.
        Glm5NextModel model;
        model.executionMode = ExecutionMode::Hybrid;
        model.weight.weight.try_emplace("gate.weight");
        auto &weight = model.weight["gate.weight"];
        std::vector<float> w((size_t)experts * 4096), x(4096), biasValues(experts);
        for (size_t i = 0; i < w.size(); ++i) w[i] = Pattern(int(i / 4096) + 1, int(i % 4096)) * .02f;
        for (int i = 0; i < experts; ++i) biasValues[i] = Pattern(i, 3) * .03f;
        Upload(weight, {experts, 4096}, w, FLOAT32);
        Data input, bias, converted, expected, actual;
        Upload(bias, {experts}, biasValues, FLOAT32);
        Glm5NextModel::HybridDecodeState::Segment segment(device, 1);
        for (int step = 0; step < 6; ++step) {
            for (int i = 0; i < 4096; ++i) x[i] = step == 0 ? 0 :
                std::ldexp(Pattern(i / 128 + step, i % 128), step * 8 - 24);
            Upload(input, {1, 1, 4096}, x);
            segment.Run(input, [&](Data &in, std::vector<Data> &out) {
                Check(model.TryHybridRouterScores("", in, out[0]), "hybrid router launch");
            });
            ToDataType(input, converted, FLOAT32); Linear(converted, weight, Data(), expected);
            Sigmoid(expected, expected);
            Check(bytes(segment.outputs[0]) == bytes(expected), "router probability bits");
            for (bool norm : {false, true}) for (float scale : {.5f, 1.5f}) {
                Data ids, scores, expectedIds, expectedScores;
                SelectExpert(segment.outputs[0], ids, scores, 8, norm, scale, &bias);
                SelectExpert(expected, expectedIds, expectedScores, 8, norm, scale, &bias);
                Check(bytes(ids) == bytes(expectedIds), "router expert IDs");
                Check(bytes(scores) == bytes(expectedScores), "router selected score bits");
            }
            if (step) Check(segment.exec != nullptr, "router graph replay");
        }
        Upload(actual, {1}, {17.f}, FLOAT32); const auto saved = bytes(actual);
        for (auto mode : {ExecutionMode::Gpu, ExecutionMode::Cpu}) {
            model.executionMode = mode;
            Check(!model.TryHybridRouterScores("", input, actual), "router nonhybrid fallback");
        }
        model.executionMode = ExecutionMode::Hybrid;
        model.mtpEnabled = true; Check(!model.TryHybridRouterScores("", input, actual), "router MTP fallback"); model.mtpEnabled = false;
        model.threadTpRank = 0; Check(!model.TryHybridRouterScores("", input, actual), "router TP fallback"); model.threadTpRank = -1;
        weight.lockInCPU = true; Check(!model.TryHybridRouterScores("", input, actual), "router host weight fallback"); weight.lockInCPU = false;
        Upload(input, {1, 2, 4096}, std::vector<float>(8192));
        Check(!model.TryHybridRouterScores("", input, actual), "router multiple-row fallback");
        Upload(input, {1, 1, 4096}, x, FLOAT32);
        Check(!model.TryHybridRouterScores("", input, actual), "router input-type fallback");
        Check(bytes(actual) == saved, "router fallback changed output");
    }
    std::printf("PASS hybrid router GPU%d probabilities/IDs/scores exact, graph and GPU/CPU/TP/MTP/batch fallbacks\n", device);
}

void fastllm::Glm5NextGGUFTestAccess::HybridKda() {
    const int device = FastllmCudaGetDevice();
    auto bytes = [&](Data &x) {
        std::vector<unsigned char> b(x.GetBytes());
        Check(cudaMemcpy(b.data(),x.cudaData,b.size(),cudaMemcpyDeviceToHost)==cudaSuccess,"KDA download");
        return b;
    };
    for (bool gguf : {false, true}) for (int columns : {264, 512, 2048, 4096, 8192, 12288})
            for (int rows : {1, 31, 128}) {
        Data input, fw, bw, gw, f, beta, g, expectedF, expectedBeta, expectedG;
        auto fill = [&](Data &x, const std::vector<int> &dims, int seed, int exponent) {
            size_t count = 1; for (int d : dims) count *= d;
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i)
                values[i] = std::ldexp(Pattern(int(i / 128) + seed, int(i % 128)), exponent);
            Upload(x, dims, values);
        };
        fill(fw, {rows, columns}, 3, -3);
        fill(bw, {rows / 2 + 1, columns}, 7, -3);
        fill(gw, {rows + 5, columns}, 11, -3);
        if (gguf) for (Data *w : {&fw, &bw, &gw}) {
            w->dataType = DATA_GGUF_FORMAT; w->ggmlType = GGML_TYPE_BF16; w->Resize(w->dims);
        }
        auto run = [&](Data &x, Data &fo, Data &bo, Data &go) {
            return FastllmCudaHybridKdaInputProjections(x, fw, bw, gw, fo, bo, go);
        };
        Glm5NextModel::HybridDecodeState::Segment segment(device, 3);
        const bool graphCase = columns == 4096 && rows == 128;
        for (int step = 0; step < 5; ++step) {
            fill(input, {1, 1, columns}, step + 17, step == 1 ? -30 : step == 2 ? 30 : -2);
            if (graphCase) segment.Run(input, [&](Data &x, std::vector<Data> &out) {
                Check(run(x, out[0], out[1], out[2]), "KDA input graph launch");
            });
            else Check(run(input, f, beta, g), "KDA input projections launch");
            Linear(input, fw, Data(), expectedF);
            Linear(input, bw, Data(), expectedBeta);
            Linear(input, gw, Data(), expectedG);
            Check(bytes(graphCase ? segment.outputs[0] : f) == bytes(expectedF), "KDA F projection bits");
            Check(bytes(graphCase ? segment.outputs[1] : beta) == bytes(expectedBeta), "KDA beta projection bits");
            Check(bytes(graphCase ? segment.outputs[2] : g) == bytes(expectedG), "KDA G projection bits");
            if (graphCase && step) Check(segment.exec != nullptr, "KDA input graph replay");
        }
        Data untouched; fill(untouched, {1, 1, 8}, 1, 0);
        const auto saved = bytes(untouched);
        void *original = fw.cudaData; fw.cudaData = (char*)original + 2;
        Check(!run(input, untouched, beta, g), "KDA input unaligned fallback"); fw.cudaData = original;
        Check(bytes(untouched) == saved, "KDA input fallback changed output");
        Check(!run(input, f, f, g), "KDA input aliased outputs fallback");
        const auto shape = input.dims; input.Resize({1, 2, columns / 2});
        Check(!run(input, untouched, beta, g), "KDA input batch fallback"); input.Resize(shape);
        const auto type = bw.dataType; bw.dataType = FLOAT32;
        Check(!run(input, untouched, beta, g), "KDA input mixed-type fallback"); bw.dataType = type;
        Check(bytes(untouched) == saved, "KDA input unsupported fallback changed output");
    }
    std::printf("PASS hybrid KDA input projections device=%d, native/GGUF, 36 shapes x 5 steps, graph and fallback\n", device);
    for (bool gguf : {false, true}) for (int heads : {1, 5, 32, 64, 128, 256}) for (int rank : {8, 24, 128, 256}) for (int taps : {2, 4, 5}) {
        constexpr int dim = 128;
        const int channels = heads * dim;
        Data q,k,v,f,g,beta,fw,gw,qw,kw,vw,a,dt,norm,cache,state,refCache,refState,output,reference;
        auto fill = [&](Data &x,const std::vector<int> &dims,int seed,float scale,DataType type=BFLOAT16) {
            size_t n=1;for(int d:dims)n*=d;std::vector<float> values(n);
            for(size_t i=0;i<n;++i)values[i]=Pattern(int(i/128)+seed,int(i%128))*scale;
            Upload(x,dims,values,type);
        };
        fill(q,{1,1,channels},1,.5f);fill(k,q.dims,2,.5f);fill(v,q.dims,3,.5f);
        fill(f,{1,1,rank},4,.1f);fill(g,f.dims,5,.1f);fill(beta,{1,1,heads},6,.3f);
        fill(fw,{channels,rank},7,.1f);fill(gw,fw.dims,8,.1f);
        if (gguf) for (auto *w : {&fw,&gw}) {
            w->dataType=DATA_GGUF_FORMAT;w->ggmlType=GGML_TYPE_BF16;w->Resize(w->dims);
        }
        fill(qw,{channels,1,taps},9,.1f,FLOAT32);fill(kw,qw.dims,10,.1f,FLOAT32);fill(vw,qw.dims,11,.1f,FLOAT32);
        fill(a,{heads},12,.3f,FLOAT32);fill(dt,{heads,dim},13,.1f,FLOAT32);fill(norm,{dim},14,.2f,FLOAT32);
        fill(cache,{3,taps-1,channels},15,.5f);fill(state,{1,heads,dim,dim},16,.01f,FLOAT32);
        refCache.CopyFrom(cache);refState.CopyFrom(state);
        auto run = [&](Data &input,Data &out) {
            return FastllmCudaHybridKdaDecode(input,k,v,f,g,beta,fw,gw,qw,kw,vw,a,dt,norm,
                cache,state,heads,dim,taps,-5.f,1e-6f,out);
        };
        auto eager = [&] {
            Data rawGate,rawBeta,gate;
            Linear(f,fw,Data(),rawGate);ToDataType(beta,rawBeta,FLOAT32);
            Data convolved[3],views[3];Data *inputs[]{&q,&k,&v};Data *weights[]{&qw,&kw,&vw};
            for(int i=0;i<3;++i) {
                views[i].FakeFrom(refCache,(size_t)i*(taps-1)*channels*2);
                views[i].Resize({1,taps-1,channels});
                KimiK3CausalConv1D(*inputs[i],*weights[i],taps,views[i],convolved[i]);
                convolved[i].Reshape({1,1,heads,dim});
            }
            rawGate.Reshape({1,1,heads,dim});Data attention;
            KimiK3RecurrentKDAOutputOnly(convolved[0],convolved[1],convolved[2],rawGate,rawBeta,
                a,dt,-5.f,refState,attention,true,true);
            Linear(g,gw,Data(),gate);gate.Reshape({1,1,heads,dim});
            KimiK3RMSNormSigmoidGate(attention,gate,norm,1e-6f,reference);
        };
        const bool graphCase = heads == 64 && rank == 128 && taps == 4;
        Glm5NextModel::HybridDecodeState::Segment segment(device,1);
        for (int step=0;step<5;++step) {
            fill(q,q.dims,step+31,.5f);
            if(graphCase)segment.Run(q,[&](Data &in,std::vector<Data> &out){Check(run(in,out[0]),"KDA graph launch");});
            else Check(run(q,output),"KDA launch");
            eager();
            Check(bytes(graphCase?segment.outputs[0]:output)==bytes(reference),"KDA output bits");
            Check(bytes(cache)==bytes(refCache),"KDA convolution cache bits");
            Check(bytes(state)==bytes(refState),"KDA recurrent state bits");
            if(graphCase && step)Check(segment.exec!=nullptr,"KDA graph capture/replay");
            if(graphCase) {
                std::vector<void*> scratch;
                for(int j=0;j<32;++j)scratch.push_back(FastllmCudaMalloc(65536));
                for(void *p:scratch)FastllmCudaFree(p);
            }
        }
        const auto savedCache=bytes(cache),savedState=bytes(state);
        void *saved=f.cudaData;f.cudaData=(char*)f.cudaData+2;
        Check(!run(q,output),"KDA unaligned fallback");f.cudaData=saved;
        Check(bytes(cache)==savedCache && bytes(state)==savedState,"KDA fallback mutated state");
        if(graphCase) {
            auto time = [&](const std::function<void()> &body) {
                cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
                cudaEventRecord(begin,cudaStreamPerThread);
                for(int i=0;i<100;++i)body();
                cudaEventRecord(end,cudaStreamPerThread);cudaEventSynchronize(end);
                float ms=0;cudaEventElapsedTime(&ms,begin,end);cudaEventDestroy(begin);cudaEventDestroy(end);
                return ms*10;
            };
            const float oldUs=time(eager),newUs=time([&]{Check(run(q,output),"KDA timing launch");});
            std::printf("KDA timing device=%d gguf=%d eager_us=%.3f hybrid_us=%.3f\n",device,gguf,oldUs,newUs);
        }
    }
    Check(cudaDeviceSynchronize()==cudaSuccess,"KDA asynchronous error");
    std::printf("PASS hybrid KDA device=%d 144 native/GGUF shapes x 5 steps, output/cache/state bits, graph, fallback\n",device);
}

void fastllm::Glm5NextGGUFTestAccess::HybridMla() {
    const int device = FastllmCudaGetDevice();
    Check(FastllmCudaHybridDsaSupported(), "hybrid MLA SM75 support");
    for (int start : {2048, 4093, 8190, 32766}) for (int heads : {32, 64}) {
        constexpr int steps = 16;
        int pages = (start + steps + 127) / 128 + 7;
        while (pages % 5 == 0) ++pages;
        PagedCacheManager pool;
        BorrowedCache paged;
        Data keyPe, input, unused, absorbed, replacement, output;
        auto fill = [&](Data &x, const std::vector<int> &dims, int seed, DataType type=BFLOAT16) {
            size_t count=1;for(int d:dims)count*=d;std::vector<float> values(count);
            for(size_t i=0;i<count;++i)values[i]=Pattern(int(i/128)+seed,int(i%128))*.125f;
            Upload(x,dims,values,type);
        };
        fill(pool,{pages,128,1,512},1);
        paged.dataType=BFLOAT16;paged.Resize({1,start+1,512});paged.isPagedKVCache=true;
        paged.pageLen=128;paged.pagedKVCacheData=&pool;
        auto updatePages = [&](int tokens) {
            paged.Resize({1,tokens,512});paged.lastPageLen=(tokens-1)%128+1;
            paged.pageIndex.clear();
            for(int p=0;p<(tokens+127)/128;++p)paged.pageIndex.push_back((p*5+3)%pages);
        };
        WeightMap weight;weight.weight.try_emplace("index_kpool_compress_ape");
        auto &ape=weight["index_kpool_compress_ape"];fill(ape,{4,128},2,FLOAT32);
        Glm5NextIndexerCache actual,reference;
        actual.tokens=reference.tokens=start;
        fill(actual.keys,{1,start/4,128},3);reference.keys.CopyFrom(actual.keys);
        if(start%4) {
            fill(actual.tailKeys,{1,start%4,128},4);fill(actual.tailGates,{1,start%4,128},5);
            reference.tailKeys.CopyFrom(actual.tailKeys);reference.tailGates.CopyFrom(actual.tailGates);
        }
        DsaProjections projected;projected.queryQuantized=true;
        fill(absorbed,{heads,1,512},6);fill(input,{1,1,128},7);
        Glm5NextModel::HybridDecodeState::MlaSegment segment;
        auto attend = [&](Data &indices, Data &out) {
            Data q;q.FakeFrom(absorbed,0);q.Resize(absorbed.dims);
            SparseLatentAttention(q,paged,indices,.125f,out,false,&keyPe,true);
        };
        auto run = [&] {
            return segment.Run(actual,projected.keys,projected.gates,projected.query,
                projected.headWeights,ape,paged,{&absorbed},attend,output);
        };
        int replays=0;
        for(int step=0;step<steps;++step) {
            const int past=start+step;updatePages(past+1);
            fill(projected.keys,{1,1,128},step+10);fill(projected.gates,{1,1,128},step+20);
            fill(projected.query,{1,1,32,128},step+30);fill(projected.headWeights,{1,1,32},step+40,FLOAT32);
            if(step==9 && paged.pageIndex.size()>1)std::swap(paged.pageIndex[0],paged.pageIndex[1]);
            if(step==11) {
                fill(replacement,absorbed.dims,60);
                absorbed.CopyFrom(replacement);
                // Force an address replacement while keeping the shape.
                std::swap(absorbed.cudaData,replacement.cudaData);
            }
            if(step==7) {
                actual.captureReplay=true;
                Check(!run() && actual.tokens==past,"hybrid MLA unsupported checkpoint fallback");
                actual.captureReplay=false;
                Data ignored;
                BuildDsaIndices(input,unused,weight,"",past,2048,actual,ignored,&paged,&projected,true);
                attend(ignored,output);
            } else {
                Check(run(),"hybrid MLA decode run");
                replays+=segment.graph && segment.graph->exec;
            }
            Data indices,expected;
            BuildDsaIndices(input,unused,weight,"",past,2048,reference,indices,&paged,&projected,true);
            attend(indices,expected);
            Check(Download(output)==Download(expected),"hybrid MLA graph attention bits");
            Check(actual.tokens==reference.tokens && actual.keys.dims==reference.keys.dims &&
                Download(actual.keys)==Download(reference.keys),"hybrid MLA pooled cache bits");
            Check(actual.tailKeys.dims==reference.tailKeys.dims && actual.tailGates.dims==reference.tailGates.dims,
                "hybrid MLA tail shapes");
            if((past+1)%4) {
                Check(Download(actual.tailKeys)==Download(reference.tailKeys) &&
                    Download(actual.tailGates)==Download(reference.tailGates),"hybrid MLA tail bits");
            }
            if(step!=7) {
                int position=0;FastllmCudaCopyFromDeviceToHost(&position,segment.position.cudaData,4);
                Check(position==actual.tokens,"hybrid MLA device/host position");
            }
        }
        Check(replays>=5,"hybrid MLA must replay across dynamic lengths");
        // The descriptor does not own the pool used by this test.
        paged.pagedKVCacheData=nullptr;
    }
    Check(cudaDeviceSynchronize()==cudaSuccess,"hybrid MLA asynchronous error");
    std::printf("PASS hybrid MLA device=%d dynamic KPool/page/bucket boundaries, graph, fallback, attention/cache bits\n",device);
}

void fastllm::Glm5NextGGUFTestAccess::HybridGraphs() {
    constexpr int n = 4096;
    const int device = FastllmCudaGetDevice();
    Data state;
    Upload(state, {1, 1, n}, std::vector<float>(n));
    std::vector<float> expected(n);
    for (bool disabled : {false, true}) {
        Glm5NextModel::HybridDecodeState::Segment segment(device, 2);
        segment.disabled = disabled;
        for (int step = 0; step < 7; ++step) {
            // Recreate the source allocation; captured kernels must read the
            // copied input, not retain the first caller's tensor address.
            std::vector<float> values(n);
            for (int i = 0; i < n; ++i) values[i] = F(Bf(Pattern(step + 1, i)));
            Data input;
            Upload(input, {1, 1, n}, values);
            segment.Run(input, [&](Data &fixed, std::vector<Data> &out) {
                Mul(fixed, 0.5f, out[0]);
                AddTo(state, out[0]);
                out[1].CopyFrom(state);
            });
            for (int i = 0; i < n; ++i) expected[i] = F(Bf(expected[i] + values[i] * 0.5f));
            Check(Download(state) == expected && Download(segment.outputs[1]) == expected,
                  "hybrid graph recurrent state/input lifetime");
            if (step > 0) Check(disabled ? segment.exec == nullptr : segment.exec != nullptr,
                                "hybrid graph capture/fallback selection");
            std::vector<void *> scratch;
            for (int i = 0; i < 32; ++i) scratch.push_back(FastllmCudaMalloc(65536));
            for (void *p : scratch) FastllmCudaFree(p);
        }
    }
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    if (major == 7 && minor == 5) for (auto type : {FLOAT32, BFLOAT16}) {
        Data fn, scale, base, norm;
        std::vector<float> weights(24 * 4 * n), norms(n), bases(24);
        for (int i = 0; i < int(weights.size()); ++i) weights[i] = Pattern(i / n, i % n) / 256;
        for (int i = 0; i < n; ++i) norms[i] = 1 + Pattern(i, 7) / 16;
        for (int i = 0; i < 24; ++i) bases[i] = Pattern(i, 3) / 16;
        Upload(fn, {24, 4 * n}, weights, type); Upload(norm, {n}, norms, FLOAT32);
        Upload(scale, {3}, {.8f, .6f, .7f}, FLOAT32); Upload(base, {24}, bases, FLOAT32);
        Glm5NextModel::HybridDecodeState::Segment segment(device, 3);
        for (int step = 0; step < 5; ++step) {
            Data input, reference, post, comb;
            std::vector<float> values(4 * n);
            for (int i = 0; i < 4 * n; ++i) values[i] = Pattern(step + i / n, i % n) * .25f;
            Upload(input, {1, 1, 4, n}, values);
            segment.Run(input, [&](Data &fixed, std::vector<Data> &out) {
                Check(FastllmCudaGlm5NextHybridHcPreNorm(fixed, fn, scale, base, norm,
                    4, 20, 1e-6f, 1e-6f, out[0], out[1], out[2]), "HC graph body");
            });
            Check(FastllmCudaGlm5NextHcPreNorm(input, fn, scale, base, norm,
                4, 20, 1e-6f, 1e-6f, reference, post, comb), "HC eager reference");
            Check(Download(reference) == Download(segment.outputs[0]) &&
                  Download(post) == Download(segment.outputs[1]) &&
                  Download(comb) == Download(segment.outputs[2]), "HC graph output/post/comb bits");
            if (step) Check(segment.exec != nullptr, "HC graph replay not captured");
        }
    }
    Check(cudaDeviceSynchronize() == cudaSuccess, "hybrid graph asynchronous error");
    std::printf("PASS hybrid graph device=%d capture/replay, input replacement, pool reuse, eager fallback\n",device);
}
void HybridMoePost() {
    auto bytes = [](Data &x) {
        std::vector<unsigned char> result(x.GetBytes());
        Check(cudaMemcpy(result.data(), x.cudaData, result.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
              "expert post download");
        return result;
    };
    int cases = 0;
    for (int rows : {1, 2, 4, 8, 16, 32}) for (int hc : {1, 4, 8})
    for (int dim : {37, 4096}) for (int seed = 0; seed < 4; ++seed) {
        std::vector<float> a(rows * dim), b(a.size()), residual(rows * hc * dim), post(rows * hc), comb(rows * hc * hc);
        auto value = [&](int i, int offset) {
            float v = Pattern(seed + offset, i);
            if (seed == 1) return std::copysign(0.f, v);
            if (seed == 2) return std::ldexp(v, (i * 7 + offset) % 200 - 100);
            if (seed == 3 && i < 3) return i == 0 ? NAN : i == 1 ? INFINITY : -INFINITY;
            return v;
        };
        for (int i = 0; i < int(a.size()); ++i) { a[i] = value(i, 1); b[i] = value(i, 2); }
        for (int i = 0; i < int(residual.size()); ++i) residual[i] = value(i, 3);
        for (int i = 0; i < int(post.size()); ++i) post[i] = Pattern(seed, i) / 4;
        for (int i = 0; i < int(comb.size()); ++i) comb[i] = Pattern(seed + 5, i) / 8;
        Data shared, routed, res, p, c, sum, reference, output;
        Upload(shared, {rows, dim}, a); Upload(routed, {rows, dim}, b);
        Upload(res, {1, rows, hc, dim}, residual); Upload(p, {1, rows, hc}, post, FLOAT32);
        Upload(c, {1, rows, hc, hc}, comb, FLOAT32);
        auto original = [&] {
            sum.CopyFrom(shared); AddTo(sum, routed);
            Check(FastllmCudaDeepSeekV4HcPostCudaMix(sum, res, p, c, 1, rows, hc, dim, reference),
                  "original expert sum and HC post");
        };
        original();
        Check(FastllmCudaHybridMoeHcPost(shared, routed, res, p, c, output), "fused expert post dispatch");
        Check(bytes(output) == bytes(reference), "expert BF16 sum and FP64 residual bits");
        if (seed == 0 && hc == 4 && dim == 4096) {
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
                  "expert post graph begin");
            Check(FastllmCudaHybridMoeHcPost(shared, routed, res, p, c, output), "expert post capture");
            cudaGraph_t graph; cudaGraphExec_t exec;
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "expert post graph end");
            Check(cudaGraphInstantiateWithFlags(&exec, graph, 0) == cudaSuccess, "expert post instantiate");
            for (int step = 0; step < 3; ++step) {
                for (int i = 0; i < int(a.size()); ++i) a[i] = Pattern(step + 8, i);
                Upload(shared, {rows, dim}, a); original();
                Check(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "expert post replay");
                Check(bytes(output) == bytes(reference), "expert post replay bits");
            }
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
        }
        if (rows == 1 && hc == 4 && dim == 37 && seed == 0) {
            const auto old = bytes(res);
            Check(!FastllmCudaHybridMoeHcPost(shared, routed, res, p, c, res), "expert post alias fallback");
            Check(bytes(res) == old, "expert post fallback must leave output intact");
        }
        ++cases;
    }
    {
        Data x, res, p, c, output;
        Upload(x, {33, 1}, std::vector<float>(33)); Upload(res, {1, 33, 1, 1}, std::vector<float>(33));
        Upload(p, {33}, std::vector<float>(33), FLOAT32); Upload(c, {33}, std::vector<float>(33), FLOAT32);
        Upload(output, {1}, {7}); const auto old = bytes(output);
        Check(!FastllmCudaHybridMoeHcPost(x, x, res, p, c, output), "expert post row limit fallback");
        Check(bytes(output) == old, "expert post row limit must leave output intact");
    }
    Check(cudaDeviceSynchronize() == cudaSuccess, "expert post asynchronous error");
    std::printf("PASS hybrid expert post GPU%d: %d exact cases, rows1..32, graph, nonfinite and alias fallback\n",
                FastllmCudaGetDevice(), cases);
}

void HybridOutput() {
    auto bytes = [](Data &x) {
        std::vector<unsigned char> v(x.GetBytes());
        Check(cudaMemcpy(v.data(),x.cudaData,v.size(),cudaMemcpyDeviceToHost)==cudaSuccess,"output download");
        return v;
    };
    int cases=0;
    for(int hc:{1,2,3,4,8,16}) for(int seed=0;seed<12;++seed) {
        std::vector<float> x(hc*4096),w(4096);
        for(int i=0;i<int(x.size());++i) {
            x[i]=Pattern(i/4096+seed,i%4096);
            if(seed%3==0)x[i]=std::ldexp(x[i],(i*13+seed)%81-40);
            if(seed==1)x[i]=0;
        }
        for(int i=0;i<4096;++i)w[i]=Pattern(seed+17,i)*.25f;
        Data input,weight,reference,output,fp,mean,collapsed;
        Upload(input,{1,1,hc,4096},x);Upload(weight,{4096},w,FLOAT32);
        auto old = [&] {
            ToDataType(input,fp,FLOAT32);
            for(int i=0;i<hc;++i) {
                Data row;Split(fp,2,i,i+1,row);row.Reshape({1,1,4096});
                if(!i)Mul(row,1.f/hc,mean);else AddTo(mean,row,1.f/hc);
            }
            ToDataType(mean,collapsed,BFLOAT16);KimiK3RMSNorm(collapsed,weight,1e-6f,reference);
        };
        old();Check(FastllmCudaHybridHcMeanNorm(input,weight,1e-6f,output),"hybrid output norm dispatch");
        Check(bytes(output)==bytes(reference),"hybrid output normalization bits");
        if(hc==4 && seed==2) {
            Check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"output capture");
            Check(FastllmCudaHybridHcMeanNorm(input,weight,1e-6f,output),"captured norm");
            cudaGraph_t graph;cudaGraphExec_t exec;Check(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"output capture end");
            Check(cudaGraphInstantiateWithFlags(&exec,graph,0)==cudaSuccess,"output instantiate");
            for(int i=0;i<3;++i){Check(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"output replay");Check(bytes(output)==bytes(reference),"norm replay bits");}
            cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
            input.strides[2]++;Check(!FastllmCudaHybridHcMeanNorm(input,weight,1e-6f,output),"strided norm fallback");input.strides[2]--;
            input.Reshape({1,2,2,4096});Check(!FastllmCudaHybridHcMeanNorm(input,weight,1e-6f,output),"batch norm fallback");
        }
        cases++;
    }
    int selections=0;
    for(int n:{1,17,255,256,257,4095,4096,4097,154880,200001}) for(int mode=0;mode<8;++mode) {
        std::vector<float> values(n);
        for(int i=0;i<n;++i)values[i]=Pattern(mode,i);
        if(mode==1)std::fill(values.begin(),values.end(),1.f);
        if(mode>=2) {
            std::fill(values.begin(),values.end(),-INFINITY);
            if(mode==2)for(int i:{1,64,128,256,4096,4097})if(i<n)values[i]=7;
            if(mode==3)for(int i:{1,64})if(i<n)values[i]=7;
            if(mode==4)for(int i:{3,128,192,4097})if(i<n)values[i]=INFINITY;
            if(mode==5)std::fill(values.begin(),values.end(),NAN);
            if(mode==6){std::fill(values.begin(),values.end(),NAN);values.back()=1;}
        }
        Data input,fp,reference,output;Upload(input,{1,1,n},values);
        ToDataType(input,fp,FLOAT32);TopK(fp,reference,1);
        Check(FastllmCudaHybridTop1(input,output),"hybrid top1 dispatch");
        Check(bytes(output)==bytes(reference),"hybrid top1 tie/nonfinite bits");selections++;
        input.Reshape({1,n,1});if(n>1)Check(!FastllmCudaHybridTop1(input,output),"multirow top1 fallback");
    }
    int device=0;cudaGetDevice(&device);
    std::printf("PASS hybrid output device=%d: %d HC mean/norm and %d Top1 cases, exact bits, ties, nonfinite, graph, fallback\n",device,cases,selections);
}
void fastllm::Glm5NextGGUFTestAccess::HybridDsaPreparation() {
    auto bytes = [](Data &x) {
        std::vector<unsigned char> result(x.GetBytes());
        Check(cudaMemcpy(result.data(), x.cudaData, result.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
            "DSA projection download");
        return result;
    };
    int quantCases = 0;
    for (int rows : {1, 2, 7, 32}) {
        Glm5NextModel::HybridDecodeState::Segment segment(FastllmCudaGetDevice(), 1);
        for (int step = 0; step < 7; ++step) {
            std::vector<float> values(rows * 128);
            for (size_t i = 0; i < values.size(); ++i) {
                const float x = Pattern(i / 128, i % 128);
                values[i] = step == 0 ? 0.f : step == 1 ? x :
                    step == 2 ? std::ldexp(x, -80) : step == 3 ? std::ldexp(x, 80) :
                    step == 4 ? (int(i % 127) - 63) * .0625f + .0001f :
                    step == 5 ? F(uint16_t(0x3f00 + i % 256)) + std::ldexp(1.f, -9) : x;
            }
            if (step == 6) {
                values[0] = std::numeric_limits<float>::infinity();
                values[1] = -values[0];
                values[2] = std::numeric_limits<float>::quiet_NaN();
            }
            Data input, rounded, expected, scales, bfScales;
            Upload(input, {rows, 128}, values, FLOAT32);
            ToDataType(input, rounded, BFLOAT16);
            Check(FastllmCudaNaiveQuantizeIndexer(rounded, expected, scales, true), "reference FP8 quantization");
            ToDataType(scales, bfScales, BFLOAT16); MulTo(expected, bfScales);
            segment.Run(input, [&](Data &fixed, std::vector<Data> &out) {
                Check(FastllmCudaHybridDsaQuantizeRotated(fixed, out[0]), "fused FP8 round trip");
            });
            Check(bytes(expected) == bytes(segment.outputs[0]), "FP8 round-trip bits, including scale boundaries");
            if (step) Check(segment.exec != nullptr, "FP8 round-trip graph replay");
            ++quantCases;
        }
    }
    Data unsupported, untouched;
    Upload(unsupported, {1, 129}, std::vector<float>(129), FLOAT32);
    Check(!FastllmCudaHybridDsaQuantizeRotated(unsupported, untouched) && untouched.cudaData == nullptr,
        "unsupported quantization must leave output untouched");
    int cases = 0;
    for (const auto shape : {std::pair<int,int>{37,31}, {257,513}, {4096,1536}}) {
        const int columns = shape.first, rank = shape.second;
        WeightMap weights;
        for (const auto item : {std::pair<const char*,int>{"wk.weight",128},
                {"index_kpool_compress_gate",128}, {"weights_proj.weight",32}, {"wq_b.weight",4096}}) {
            const int width = std::string(item.first) == "wq_b.weight" ? rank : columns;
            std::vector<float> values(size_t(width) * item.second);
            for (size_t i=0;i<values.size();++i) values[i]=Pattern(i/width,i%width)/137;
            Upload(weights[item.first], {item.second,width}, values, FLOAT32);
        }
        std::vector<float> gamma(128),beta(128);
        for(int i=0;i<128;++i){gamma[i]=1+Pattern(i,3)/16;beta[i]=Pattern(i,5)/16;}
        Upload(weights["k_norm.weight"],{128},gamma,FLOAT32);
        Upload(weights["k_norm.bias"],{128},beta,FLOAT32);
        Data input,normalized;
        Glm5NextModel::HybridDecodeState::Segment segment(FastllmCudaGetDevice(),4);
        for(int step=0;step<5;++step) {
            std::vector<float> x(columns),q(rank);
            for(int i=0;i<columns;++i)x[i]=step==4?0.f:std::ldexp(Pattern(step,i),step==2?7:step==3?-7:0);
            for(int i=0;i<rank;++i)q[i]=std::ldexp(Pattern(step+3,i),step==3?-7:0);
            Upload(input,{1,1,columns},x);
            if(!step)Upload(normalized,{1,1,rank},q);
            else {
                std::vector<uint16_t> bits(rank);for(int i=0;i<rank;++i)bits[i]=Bf(q[i]);
                FastllmCudaCopyFromHostToDevice(normalized.cudaData,bits.data(),rank*2);
            }
            DsaProjections reference;
            ProjectDsaKeys(input,weights,"",reference);ProjectDsaQueries(input,normalized,weights,"",reference);
            auto body=[&](Data &fixed,std::vector<Data> &out) {
                Check(FastllmCudaHybridDsaProjections(fixed,normalized,weights["wk.weight"],
                    weights["index_kpool_compress_gate"],weights["wq_b.weight"],weights["weights_proj.weight"],
                    weights["k_norm.weight"],weights["k_norm.bias"],out[0],out[1],out[2],out[3]),"DSA projections dispatch");
            };
            segment.Run(input,body);
            Data *expected[]{&reference.keys,&reference.gates,&reference.query,&reference.headWeights};
            for(int i=0;i<4;++i)Check(bytes(*expected[i])==bytes(segment.outputs[i]),"DSA projection/normalization/scale bits");
            if(step)Check(segment.exec!=nullptr,"DSA projection graph not captured");
            ++cases;
        }
        Data rejected, untouched;Upload(rejected,{1,2,columns},std::vector<float>(2*columns));
        Check(!FastllmCudaHybridDsaProjections(rejected,normalized,weights["wk.weight"],
            weights["index_kpool_compress_gate"],weights["wq_b.weight"],weights["weights_proj.weight"],
            weights["k_norm.weight"],weights["k_norm.bias"],untouched,untouched,untouched,untouched),"DSA multirow fallback");
        Check(untouched.cudaData==nullptr,"DSA rejection mutated output");
    }
    Check(cudaDeviceSynchronize()==cudaSuccess,"DSA projections async error");
    std::printf("PASS hybrid DSA preparation GPU%d: %d projections and %d FP8 round trips exact, graph replay and fallback\n",
        FastllmCudaGetDevice(),cases,quantCases);
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    FastllmCudaSetDevice(0);
    static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:0");
    try {
        const struct HybridCase {
            const char *option;
            void (*run)();
            bool sm75Only;
        } hybridCases[] = {
            {"--hybrid-moe-post", HybridMoePost, true},
            {"--hybrid-dsa-prepare", Glm5NextGGUFTestAccess::HybridDsaPreparation, true},
            {"--hybrid-output", HybridOutput, false},
            {"--hybrid-mla", Glm5NextGGUFTestAccess::HybridMla, true},
            {"--hybrid-router", Glm5NextGGUFTestAccess::HybridRouter, false},
            {"--hybrid-kda", Glm5NextGGUFTestAccess::HybridKda, false},
            {"--hybrid-graph", Glm5NextGGUFTestAccess::HybridGraphs, false},
        };
        for (const auto &test : hybridCases) if (argc == 2 && std::string(argv[1]) == test.option) {
            bool checked = false;
            for (int device = 0; device < devices; ++device) {
                FastllmCudaSetDevice(device);
                static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
                if (test.sm75Only && FastllmCudaRuntimeArch() != 75) continue;
                test.run();
                checked = true;
            }
            return checked ? 0 : 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--hybrid-sm75") {
            bool checked = false;
            for (int device = 0; device < devices; ++device) {
                FastllmCudaSetDevice(device);
                static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
                if (!HybridSm75Attention()) continue;
                checked = true;
                HybridHcPreNorm();
                HybridBf16Prefill();
                HybridHcPostPrefill();
                for (int rows : {1, 2, 4, 8, 9, 17, 33}) Scoring(true, rows);
                Attention(2045, 9, true, true);
                Attention(4098, 1, false, true);
                Attention(32769, 1, true, true);
                Check(cudaDeviceSynchronize() == cudaSuccess, "hybrid CUDA async error");
            }
            return checked ? 0 : 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--sm75-verify") {
            bool checked = false;
            for (int device = 0; device < devices; ++device) {
                FastllmCudaSetDevice(device);
                static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
                checked = Sm75VerificationAttention() || checked;
            }
            return checked ? 0 : 77;
        }
        if (argc == 2 && std::string(argv[1]) == "--rollback") {
            for (int device = 0; device < devices; ++device) {
                FastllmCudaSetDevice(device);
                static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:" + std::to_string(device));
                VerificationRollback();
            }
            std::puts("PASS GLM DSA MTP rollback on all CUDA devices");
            return 0;
        }
        if (argc == 2) { Fixture(argv[1]); return 0; }
        DecodeRouter();
        DecodeHc();
        DecodeSmallGemv();
        NormAndQuantization();
        KpoolChunks();
        VerificationRollback();
        Selection(0, 9);
        Selection(2045, 15);
        Selection(32760, 8);
        for (int past : {16384, 32768, 65535}) Selection(past, 1);
        Scoring();
        Attention(2045, 9, true);
        Attention(4098, 1, false);
        for (int tail = 0; tail < 4; ++tail) Attention(32768 + tail, 1, true);
        MlaPlanCache(0, 64, 512, BFLOAT16);
        MlaPlanCache(0, 32, 512, BFLOAT16);
        MlaPlanCache(0, 64, 128, BFLOAT16);
        MlaPlanCache(0, 64, 128, FLOAT16);
        if (devices > 1) MlaPlanCache(1, 64, 512, BFLOAT16);
        MlaPlanCache(0, 64, 512, BFLOAT16);
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA async error");
        std::puts("PASS GLM DSA");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
