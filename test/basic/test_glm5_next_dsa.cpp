#include "../../src/models/glm5_next_dsa.h"
#include "executor.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <fstream>
#include <set>
#include <stdexcept>
using namespace fastllm;
using namespace fastllm::glm5_next_detail;
namespace {
void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
uint16_t Bf(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
}
float F(uint16_t x) { uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f; }
void Upload(Data &x, const std::vector<int> &dims, const std::vector<float> &v, DataType type = BFLOAT16) {
    x.dataType = type; x.UpdateUnitSize(); x.Resize(dims);
    x.ToDevice(DataDevice::CUDA, {0}, false); x.Allocate();
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
void Scoring() {
    const int rows=9,m=520,heads=32;std::vector<float> q(rows*heads*128),k(m*128),w(rows*heads);
    for(int r=0;r<rows*heads;++r){w[r]=Pattern(r,3)/32;for(int d=0;d<128;++d)q[r*128+d]=F(Bf(Pattern(r,d)/8));}
    for(int j=0;j<m;++j)for(int d=0;d<128;++d)k[j*128+d]=F(Bf(Pattern(j+8,d)/8));
    Data qd,kd,wd,s;Upload(qd,{1,rows,heads,128},q);Upload(kd,{1,m,128},k);Upload(wd,{1,rows,heads},w,FLOAT32);
    Check(FastllmCudaDeepSeekV41IndexerScore(qd,wd,kd,4,2050,s),"score launch");auto got=Download(s);double err=0;
    for(int r=0;r<rows;++r)for(int j=0;j<std::min(m,(2050+r+1)/4);++j){
        double expected=0;for(int h=0;h<heads;++h){double dot=0;for(int d=0;d<128;++d)dot+=q[(r*heads+h)*128+d]*k[j*128+d];expected+=std::max(0.,dot)*w[r*heads+h];}
        err=std::max(err,std::abs(got[r*m+j]-expected));
    }
    std::printf("PASS weighted ReLU scoring max_error=%.8g\n",err);Check(err<2e-5,"score reference");
}
struct BorrowedCache:Data{~BorrowedCache(){isPagedKVCache=false;}};
void Attention(int past,int rows,bool fragmented) {
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
    BorrowedCache cache;cache.dataType=BFLOAT16;cache.Resize({1,tokens,rank});cache.isPagedKVCache=true;cache.pageLen=pageLen;cache.lastPageLen=(tokens-1)%pageLen+1;cache.pageIndex=pagesIds;cache.pagedKVCacheData=&pool;
    Data query,out;Upload(query,{heads,rows,rank},q);SparseLatentAttention(query,cache,idx,1.f/16,out);auto got=Download(out);double error=0;
    for(float v:got)Check(std::isfinite(v),"attention nonfinite");
    for(int h:{0,17,63})for(int r=0;r<rows;++r){
        std::vector<double> p(width);double den=0;
        for(int i=0;i<width;++i){int t=ids[r*width+i];if(t<0)continue;double dot=0;for(int d=0;d<rank;++d)dot+=q[(h*rows+r)*rank+d]*kv[(pagesIds[t/pageLen]*pageLen+t%pageLen)*rank+d];den+=(p[i]=std::exp(dot/16));}
        for(int d=0;d<rank;d+=17){double num=0;for(int i=0;i<width;++i){int t=ids[r*width+i];if(t>=0)num+=p[i]*kv[(pagesIds[t/pageLen]*pageLen+t%pageLen)*rank+d];}error=std::max(error,std::abs(got[(h*rows+r)*rank+d]-num/den));}
    }
    if (rows == 1) {
        Data table(INT32), physical, pagedQuery, pagedOutput;
        table.Resize({pages}); table.ToDevice(DataDevice::CUDA,{0},false); table.Allocate(false);
        FastllmCudaCopyFromHostToDevice(table.cudaData,pagesIds.data(),pages*4);
        Check(FastllmCudaQwen4ExpandSelectedBlocks(groups,tokens,past,4,physical,&table,pageLen),"paged expand");
        auto mapped=Integers(physical);
        for(size_t i=0;i<ids.size();++i)
            Check(mapped[i]==(ids[i]<0?-1:pagesIds[ids[i]/pageLen]*pageLen+ids[i]%pageLen),"physical page mapping");
        PagedCacheManager pePool;
        Upload(pePool,{int(kv.size()/pageLen/rank),pageLen,1,64},std::vector<float>(kv.size()/rank*64));
        BorrowedCache peCache;peCache.dataType=BFLOAT16;peCache.Resize({1,tokens,64});
        peCache.isPagedKVCache=true;peCache.pageLen=pageLen;peCache.lastPageLen=cache.lastPageLen;
        peCache.pageIndex=pagesIds;peCache.pagedKVCacheData=&pePool;
        Upload(pagedQuery,{heads,rows,rank},q);
        SparseLatentAttention(pagedQuery,cache,physical,1.f/16,pagedOutput,false,&peCache);
        Check(Download(pagedOutput)==got,"paged BF16 fallback differs");
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
    std::printf("PASS sparse latent attention past=%d rows=%d fragmented=%d max_error=%.8g\n",past,rows,fragmented,error);
    Check(error<.003,"attention independent reference");
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
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    FastllmCudaSetDevice(0);
    static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:0");
    try {
        if (argc == 2) { Fixture(argv[1]); return 0; }
        DecodeRouter();
        DecodeHc();
        DecodeSmallGemv();
        NormAndQuantization();
        KpoolChunks();
        Selection(0, 9);
        Selection(2045, 15);
        Selection(32760, 8);
        for (int past : {16384, 32768, 65535}) Selection(past, 1);
        Scoring();
        Attention(2045, 9, true);
        Attention(4098, 1, false);
        for (int tail = 0; tail < 4; ++tail) Attention(32768 + tail, 1, true);
        Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA async error");
        std::puts("PASS GLM DSA");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
