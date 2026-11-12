#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include "utils/kernel_registry.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(bool x,const char *s){if(!x)throw std::runtime_error(s);}
static void Cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static float Half(float x){return __half2float(__float2half_rn(x));}
static std::vector<float> Random(size_t n,int seed,bool half=false){std::mt19937 r(seed);std::normal_distribution<float>d(0,.4);std::vector<float>x(n);for(auto &v:x){v=d(r);if(half)v=Half(v);}return x;}
static void Upload(Data &d,const std::vector<float>&x){
    if(d.dataType==FLOAT16){std::vector<half>v(x.size());for(size_t i=0;i<x.size();++i)v[i]=__float2half_rn(x[i]);Cuda(cudaMemcpy(d.cudaData,v.data(),v.size()*2,cudaMemcpyHostToDevice));}
    else Cuda(cudaMemcpy(d.cudaData,x.data(),x.size()*4,cudaMemcpyHostToDevice));
}
static void Allocate(Data &d){d.ToDevice(CUDA,{0},false);d.Allocate(false);}
static double Compare(Data &d,const std::vector<float>&ref,double tolerance,const char *name){
    std::vector<float>x(ref.size());
    if(d.isPagedKVCache){auto &pool=*d.pagedKVCacheData;std::vector<half>v(pool.Count(0));Cuda(cudaMemcpy(v.data(),pool.cudaData,v.size()*2,cudaMemcpyDeviceToHost));const int dim=d.dims[2],heads=d.dims[0],capacity=ref.size()/heads/dim;
        for(int h=0;h<heads;++h)for(int t=0;t<capacity;++t)for(int k=0;k<dim;++k)x[(size_t(h)*capacity+t)*dim+k]=__half2float(v[((size_t(d.pageIndex[t/d.pageLen])*d.pageLen+t%d.pageLen)*heads+h)*dim+k]);}
    else if(d.dataType==FLOAT16){std::vector<half>v(ref.size());Cuda(cudaMemcpy(v.data(),d.cudaData,v.size()*2,cudaMemcpyDeviceToHost));for(size_t i=0;i<v.size();++i)x[i]=__half2float(v[i]);}
    else Cuda(cudaMemcpy(x.data(),d.cudaData,x.size()*4,cudaMemcpyDeviceToHost));
    double err=0,energy=0,maximum=0;
    for(size_t i=0;i<x.size();++i){Check(std::isfinite(x[i]),"nonfinite output");double e=x[i]-ref[i];err+=e*e;energy+=double(ref[i])*ref[i];maximum=std::max(maximum,std::abs(e));}
    double l2=std::sqrt(err/std::max(1e-20,energy));
    if(l2>tolerance&&maximum>2e-6){fprintf(stderr,"%s L2=%g max=%g\n",name,l2,maximum);throw std::runtime_error(name);}return l2;
}
static std::vector<float> NormRope(const float *src,int dim,const std::vector<float>&weight,
        int position,int rotary,bool half){
    float sum=0;for(int d=0;d<dim;++d)sum+=src[d]*src[d];
    float scale=1/std::sqrt(sum/dim+1e-6f);std::vector<float>x(dim),out(dim);
    for(int d=0;d<dim;++d){x[d]=src[d]*scale*weight[d];if(half)x[d]=Half(x[d]);}
    out=x;
    for(int d=0;d<rotary/2;++d){float angle=position/std::pow(1000000.f,float(2*d)/rotary);float sn=std::sin(angle),cs=std::cos(angle);out[d]=x[d]*cs-x[d+rotary/2]*sn;out[d+rotary/2]=x[d]*sn+x[d+rotary/2]*cs;}
    if(half)for(auto &v:out)v=Half(v);return out;
}
struct Request {
    PagedCacheManager keyPool,valuePool;
    Data key{FLOAT16},value{FLOAT16},tail{FLOAT32},tailPos{FLOAT32},blocks{FLOAT32},pos{FLOAT32},meta{INT32},scores{FLOAT32},selected{INT32},capture{FLOAT32};
    std::vector<float> k,v,t,p,b;
    int rows,valid,previous,kcap,vcap,bcap,start;
};
static void Case(int count,int sequence,int heads,int kh,int dim,int ih,int id,int ratio,int budget,bool paged=false){
    const int rotary=std::min(64,id);std::vector<std::unique_ptr<Request>> owned;
    std::vector<FastllmCudaQwen4QSARequest> requests;int total=0;
    auto cache=[&](Data &d,int heads,int capacity,int previous,int width){d.Expansion({heads,capacity,width});d.Resize({heads,previous,width});d.ToDevice(CUDA,{0},false);d.Allocate(false);};
    for(int i=0;i<count;++i){owned.emplace_back(new Request);auto &r=*owned.back();r.rows=sequence+(i%3);r.valid=r.rows-(i%2);r.previous=budget+17+i*3;r.kcap=r.previous+3*r.rows+17;r.vcap=r.kcap+11;r.bcap=(r.kcap+ratio-1)/ratio;r.start=total;total+=r.rows;
        cache(r.key,kh,r.kcap,r.previous,dim);cache(r.value,kh,r.vcap,r.previous,dim);
        r.tail.Resize({ratio,id});r.tailPos.Resize({ratio});r.blocks.Expansion({r.bcap,id});r.blocks.Resize({r.previous/ratio,id});r.pos.Resize({1,r.rows});r.meta.Resize({2});r.scores.Resize({r.rows,r.bcap});r.selected.Resize({r.rows,budget/ratio});r.capture.Resize({r.rows,id});
        for(Data*d:{&r.tail,&r.tailPos,&r.blocks,&r.pos,&r.meta,&r.scores,&r.selected,&r.capture})Allocate(*d);
        r.k=Random(size_t(kh)*r.kcap*dim,11+i,true);r.v=Random(size_t(kh)*r.vcap*dim,29+i,true);r.t=Random(ratio*id,47+i);r.p.resize(ratio);for(int j=0;j<ratio;++j)r.p[j]=r.previous-r.previous%ratio+j;r.b=Random(r.bcap*id,67+i);
        Upload(r.key,r.k);Upload(r.value,r.v);Upload(r.tail,r.t);Upload(r.tailPos,r.p);Upload(r.blocks,r.b);
        if(paged){auto convert=[&](Data &d,PagedCacheManager &pool,const std::vector<float>&x,int capacity){const int pageLen=13,pages=(capacity+pageLen-1)/pageLen;
            pool.dataType=FLOAT16;pool.UpdateUnitSize();pool.pageLen=pageLen;pool.Resize({pages+3,pageLen,kh,dim});Allocate(pool);pool.SetMaxPages(pages+3);
            d.FreeSpace();d.isPagedKVCache=true;d.pagedKVCacheData=&pool;d.pageLen=pageLen;d.pageIndex.resize(pages);for(int j=0;j<pages;++j)d.pageIndex[j]=pages+1-j;pool.Pick(d.pageIndex);
            std::vector<float>physical(pool.Count(0),.125f);for(int h=0;h<kh;++h)for(int t=0;t<capacity;++t)for(int k=0;k<dim;++k)physical[((size_t(d.pageIndex[t/pageLen])*pageLen+t%pageLen)*kh+h)*dim+k]=x[(size_t(h)*capacity+t)*dim+k];Upload(pool,physical);
            d.cudaData=FastllmCudaMalloc(pages*4);Cuda(cudaMemcpy(d.cudaData,d.pageIndex.data(),pages*4,cudaMemcpyHostToDevice));};
            convert(r.key,r.keyPool,r.k,r.kcap);convert(r.value,r.valuePool,r.v,r.vcap);}
        requests.push_back({&r.key,&r.value,&r.tail,&r.tailPos,&r.blocks,&r.pos,&r.meta,&r.scores,&r.selected,&r.capture,r.rows});
    }
    Data qg(FLOAT16,{1,total,2*heads*dim}),key(FLOAT16,{1,total,kh*dim}),value(FLOAT16,key.dims),index(FLOAT16,{1,total,(ih+1)*id}),qn(FLOAT32,{dim}),kn(FLOAT32,{dim}),iqn(FLOAT32,{id}),ikn(FLOAT32,{id}),output(FLOAT16);
    auto qw=Random(dim,101),kw=Random(dim,107),iqw=Random(id,109),ikw=Random(id,113);
    for(auto*v:{&qw,&kw,&iqw,&ikw})for(auto &x:*v)x+=1;
    for(Data*d:{&qg,&key,&value,&index,&qn,&kn,&iqn,&ikn})Allocate(*d);
    Upload(qn,qw);Upload(kn,kw);Upload(iqn,iqw);Upload(ikn,ikw);
    std::vector<float> qv,kv,vv,iv;
    auto inputs=[&](int seed){qv=Random(qg.Count(0),seed,true);kv=Random(key.Count(0),seed+2,true);vv=Random(value.Count(0),seed+4,true);iv=Random(index.Count(0),seed+6,true);Upload(qg,qv);Upload(key,kv);Upload(value,vv);Upload(index,iv);
        for(auto &p:owned){auto&r=*p;std::vector<float>positions(r.rows);for(int j=0;j<r.rows;++j)positions[j]=r.previous+j;Upload(r.pos,positions);int32_t meta[]={r.previous,r.valid};Cuda(cudaMemcpy(r.meta.cudaData,meta,sizeof(meta),cudaMemcpyHostToDevice));}};
    auto run=[&]{return FastllmCudaQwen4QSABatchGraph(requests,qg,key,value,index,qn,kn,iqn,ikn,output,heads,kh,dim,ih,id,ratio,budget,rotary,1000000.f,1e-6f);};
    auto oracle=[&]{std::vector<float> result(size_t(total)*heads*dim),query(size_t(total)*heads*dim),iq(size_t(total)*ih*id);
        for(auto &ptr:owned){auto &r=*ptr;std::vector<float>raw(size_t(r.rows)*id);
            for(int t=0;t<r.rows;++t){int row=r.start+t,pos=r.previous+t;const float *x=iv.data()+size_t(row)*(ih+1)*id+ih*id;std::copy(x,x+id,raw.begin()+size_t(t)*id);if(t>=r.valid)continue;
                for(int h=0;h<heads;++h){auto q=NormRope(qv.data()+(size_t(row)*heads+h)*2*dim,dim,qw,pos,rotary,true);std::copy(q.begin(),q.end(),query.begin()+(size_t(row)*heads+h)*dim);}
                for(int h=0;h<kh;++h){auto k=NormRope(kv.data()+(size_t(row)*kh+h)*dim,dim,kw,pos,rotary,true);std::copy(k.begin(),k.end(),r.k.begin()+(size_t(h)*r.kcap+pos)*dim);std::copy(vv.begin()+(size_t(row)*kh+h)*dim,vv.begin()+(size_t(row)*kh+h+1)*dim,r.v.begin()+(size_t(h)*r.vcap+pos)*dim);}
                for(int h=0;h<ih;++h){auto q=NormRope(iv.data()+size_t(row)*(ih+1)*id+h*id,id,iqw,pos,rotary,true);std::copy(q.begin(),q.end(),iq.begin()+(size_t(row)*ih+h)*id);}
                std::copy(x,x+id,r.t.begin()+size_t(pos%ratio)*id);r.p[pos%ratio]=pos;
                if((pos+1)%ratio==0){std::vector<float>a(id);for(int d=0;d<id;++d){a[d]=r.t[d];for(int j=1;j<ratio;++j)a[d]+=r.t[size_t(j)*id+d];a[d]/=ratio;}auto pooled=NormRope(a.data(),id,ikw,int(r.p[0]),rotary,false);std::copy(pooled.begin(),pooled.end(),r.b.begin()+size_t((pos+1)/ratio-1)*id);}
            }
            Compare(r.capture,raw,0,"raw verifier capture");Compare(r.key,r.k,.001,"key cache and untouched regions");Compare(r.value,r.v,0,"value cache and untouched regions");Compare(r.tail,r.t,0,"tail state");Compare(r.tailPos,r.p,0,"tail positions");Compare(r.blocks,r.b,2e-5,"compressed blocks and untouched regions");
            for(int t=0;t<r.valid;++t){int row=r.start+t,length=r.previous+t+1,blocks=length/ratio,choose=std::min(blocks,budget/ratio);std::vector<float>scores(blocks);std::vector<int>order(blocks);std::iota(order.begin(),order.end(),0);
                for(int b=0;b<blocks;++b){float s=0;for(int h=0;h<ih;++h){float dot=0;for(int d=0;d<id;++d)dot+=iq[(size_t(row)*ih+h)*id+d]*r.b[size_t(b)*id+d];s+=std::max(dot,0.f);}scores[b]=s/std::sqrt(float(id));}
                std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});order.resize(choose);std::sort(order.begin(),order.end());std::vector<int>chosen(choose);Cuda(cudaMemcpy(chosen.data(),(int32_t*)r.selected.cudaData+size_t(t)*(budget/ratio),choose*4,cudaMemcpyDeviceToHost));Check(chosen==order,"top-k set or tie order mismatch");
                std::vector<int>ids;for(int b:order)for(int j=0;j<ratio;++j)ids.push_back(b*ratio+j);for(int j=blocks*ratio;j<length;++j)ids.push_back(j);
                for(int h=0;h<heads;++h){const int khead=h/(heads/kh);std::vector<float>logits(ids.size());float maximum=-INFINITY;for(size_t j=0;j<ids.size();++j){float dot=0;for(int d=0;d<dim;++d)dot+=query[(size_t(row)*heads+h)*dim+d]*r.k[(size_t(khead)*r.kcap+ids[j])*dim+d];logits[j]=dot/std::sqrt(float(dim));maximum=std::max(maximum,logits[j]);}float sum=0;for(auto &x:logits){x=std::exp(x-maximum);sum+=x;}
                    for(int d=0;d<dim;++d){float a=0;for(size_t j=0;j<ids.size();++j)a+=logits[j]*r.v[(size_t(khead)*r.vcap+ids[j])*dim+d];float gate=qv[(size_t(row)*heads+h)*2*dim+dim+d];float p=Half(1.f/Half(1.f+Half(std::exp(-gate))));result[(size_t(row)*heads+h)*dim+d]=Half(Half(a/sum)*p);}
                }
            }
        }return Compare(output,result,.003,"attention output CPU oracle");};
    inputs(127);
    // Warm allocation discovery, then restore mutable state before capture.
    std::vector<size_t> allocations;
    {FastllmCudaAllocationRecorder recorder(&allocations);Check(run(),"batch rejected");}
    double maximum=oracle();
    // A malformed last request must not update any earlier request.
    auto saved=requests.back().value;requests.back().value=requests.back().key;
    Check(!run(),"aliased KV accepted");requests.back().value=saved;
    for(auto &p:owned){Compare(p->key,p->k,.001,"rejection changed state");p->previous+=p->valid;}
    Check(FastllmCudaReserveGraphWorkspace(allocations),"reserve graph scratch");
    Check(FastllmCudaGraphPrepareCaptureDevice(),"prepare graph");
    Check(FastllmCudaGraphMemoryPoolBegin()&&FastllmCudaGraphBeginCapture(),"capture begin");Check(run(),"captured batch rejected");
    void *graph=nullptr,*exec=nullptr;std::vector<void*>pins;
    Check(FastllmCudaGraphEndCapture(&graph)&&FastllmCudaGraphMemoryPoolEnd(pins),"capture end");Check(FastllmCudaGraphInstantiate(graph,&exec),"graph instantiate");
    for(int step=0;step<2;++step){inputs(151+step*23);Check(FastllmCudaGraphLaunch(exec),"graph replay");Cuda(cudaDeviceSynchronize());maximum=std::max(maximum,oracle());for(auto&p:owned)p->previous+=p->valid;}
    FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(pins);
    printf("PASS paged=%d requests=%d sequence=%d rows=%d heads=%d/%d dim=%d index=%d/%d ratio=%d budget=%d L2=%g eager+2 changed-input graph replays\n",paged,count,sequence,total,heads,kh,dim,ih,id,ratio,budget,maximum);fflush(stdout);
}
int main(int argc,char**argv){try{FastllmCudaSetDevice(0);const bool reference=argc>1 && std::string(argv[1])=="--reference";Check(fastllm_kernel_configure(reference?"{\"mode\":\"quick\",\"families\":[\"attention.qsa.requests.fp16\"],\"force\":{\"attention.qsa.requests.fp16\":\"cuda.qsa.requests.reference64\"}}":argc>1?"{\"mode\":\"quick\",\"families\":[\"attention.qsa.requests.fp16\"],\"force\":{\"attention.qsa.requests.fp16\":\"cuda.qsa.requests.chunk128\"}}":"{\"mode\":\"off\"}")==1,"policy");
    Case(1,1,24,2,256,4,128,4,2048);Case(4,1,24,2,256,4,128,4,2048);
    for(int count:{1,2,3,4,8,17})Case(count,count%2?7:2,6,2,128,2,128,4,128);
    Case(2,17,2,1,128,1,64,3,96);Case(3,4,40,2,256,4,256,8,128);
    Case(4,1,24,2,256,4,128,4,2048,true);Case(3,4,6,2,128,2,128,4,128,true);Case(17,2,2,1,128,1,64,3,96,true);
    return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
