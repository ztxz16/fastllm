#include "devices/cuda/fastllm-naive-flex-moe.h"
#include <cuda_bf16.h>
#include "devices/cuda/fastllm-cuda-norm.cuh"
#include "devices/cuda/fastllm-cuda-router-topk.cuh"
#include <cstring>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
constexpr int Ranks=8, Hidden=4096, TopK=8, MaxRows=17;
using BF = __nv_bfloat16;
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void onRanks(F work) {
    std::array<std::thread,Ranks> threads;
    std::array<std::exception_ptr,Ranks> errors{};
    for (int r=0;r<Ranks;++r) threads[r]=std::thread([&,r] {
        try { check(cudaSetDevice(r)); work(r); } catch (...) { errors[r]=std::current_exception(); }
    });
    for (auto &t:threads) t.join();
    for (auto &e:errors) if (e) std::rethrow_exception(e);
}
unsigned __device__ mix(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; return x;
}
// Every E2M1 value is exactly representable in E4M3. Populate two independent
// storage formats from the same logical expert, row, K group and signed nibble.
__global__ void fill(uint8_t *fp4, uint8_t *fp8, uint8_t *fp8Scale, uint8_t *epScale,
                     int rank, int experts, int n, int k, bool gate, bool ep) {
    const size_t elements=size_t(n)*k, groups=elements/32;
    constexpr uint8_t e4m3[]={0,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c};
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<experts*elements/2;i+=size_t(gridDim.x)*blockDim.x) {
        int expert=i/(elements/2), row=(i/(k/2))%n, pair=i%(k/2), group=pair/16;
        int source=gate ? (row/16)*8+row%8+(row%16>=8?n/2:0) : row;
        unsigned x=mix(unsigned(i) ^ (rank*0x982d0171u));
        int lo=x&15, hi=(x>>8)&15;
        fp8[i*2]=e4m3[lo&7]|((lo&8)<<4);
        fp8[i*2+1]=e4m3[hi&7]|((hi&8)<<4);
        size_t base=expert*(elements/2+groups);
        fp4[ep?i:base+size_t(source)*(k/2)+pair]=lo|(hi<<4);
        if (pair%16==0) {
            uint8_t scale=118+(mix(unsigned(i/16)+27*rank)%5);
            int t=(row&~127)+(row%32)*4+(row%128)/32;
            size_t offset=expert*groups+(size_t(group/4)*n+t)*4+group%4;
            fp8Scale[offset]=scale;
            if (ep) epScale[offset]=scale;
            else fp4[base+elements/2+size_t(source)*(k/32)+group]=scale;
        }
    }
}
// Model the ordered attention/router producer independently from FlexMoE.
__global__ void prepareInput(const BF *x, FastllmMoePreparedInput p) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    float v=__bfloat162float(x[i]),mx=fabsf(v);
    for (int d=16;d;d/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
    unsigned exp=max((__float_as_uint(mx)+0x1fffffu)>>23,113u)-8;
    p.values[i]=__nv_fp8_e4m3(v*__uint_as_float((254-exp)<<23)).__x;
    if (i%32==0) p.scales[i/32]=exp;
    if (i<256) p.counters[i]=0;
    if (!i) ++*p.epoch;
}
struct RankData {
    std::vector<void*> buffers;
    template<class T> T *allocate(size_t count) {
        T *ptr=nullptr; check(cudaMalloc(&ptr,count*sizeof(T))); buffers.push_back(ptr); return ptr;
    }
    BF *x, *y[2]; int *ids, *refIds; float *probs, *router, *bias, *refScores, *publishedRouter;
};
// Check the producer in isolation, including graph replay with different inputs.
__global__ void publishDelayedRouter(const float *src, float *dst) {
    cudaTriggerProgrammaticLaunchCompletion();
    auto start = clock64();
    while (clock64() - start < 32000) __nanosleep(64);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i];
}
__global__ void referenceRouterSelection(const float *router, const float *bias, int *ids, float *prob,
                                        int needNorm = 1, float scale = 1.f) {
    __shared__ fastllm::cuda::RouterTopKScratch<8> scratch;
    int row = blockIdx.x;
    fastllm::cuda::SelectRouterTopK<256,8>(router+row*256,bias,ids+row*8,prob+row*8,
        bias!=nullptr,needNorm,scale,scratch);
}
// Capture the same EP producer and consumer used behind both SWA and DSA.
void testEpPrepared(void *fp4,void *fp8,int rank,RankData &a) {
    constexpr int rows=8;
    FastllmMoePreparedInput p{};
    require(!fastshared_ep_prepared_input(fp8,rank,0,rows,&p) &&
            !fastshared_ep_prepared_input(fp4,rank,0,4,&p),"EP prepared fallback");
    require(fastshared_ep_prepared_input(fp4,rank,0,rows,&p),"EP packed input");
    p.indices=a.ids; p.scores=a.probs; p.bias=a.bias; p.routed=true; p.routeScale=1.375f;
    require(p.Valid(rows),"EP descriptor");
    auto *normed=a.allocate<BF>(rows*Hidden);
    auto *weights=a.allocate<float>(256*Hidden),*gamma=a.allocate<float>(Hidden);
    std::vector<float> w(256*Hidden),g(Hidden);
    for(size_t i=0;i<w.size();++i)w[i]=std::sin(i*.037f)*.015f;
    for(int i=0;i<Hidden;++i)g[i]=1.f+.2f*std::cos(i*.087f);
    check(cudaMemcpy(weights,w.data(),w.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(gamma,g.data(),g.size()*4,cudaMemcpyHostToDevice));
    std::vector<BF> x(rows*Hidden),ref(x.size()),got(x.size());
    std::vector<int> ids(rows*TopK),refIds(ids.size());
    std::vector<float> probs(ids.size()),refProbs(ids.size());
    // Selection reads four FP32 values at once from each row of router logits.
    require(fastshared_ep_prepared_norm(a.x,weights,a.router+1,gamma,normed,1.e-6f,p,false)==cudaErrorInvalidValue,
            "Misaligned EP router logits");
    for(bool afterSwa: {false,true}){
        cudaGraph_t graph;cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        require(fastshared_ep_prepared_norm(a.x,weights,a.router,gamma,normed,1.e-6f,p,afterSwa)==0,"EP norm/router");
        require(fastshared_ep_run_prepared(fp4,rank,0,a.ids,a.probs,a.y[1])==0,"EP prepared MoE");
        referenceRouterSelection<<<rows,32,0,cudaStreamPerThread>>>(a.router,a.bias,a.refIds,a.refScores,1,p.routeScale);
        require(fastshared_ep_run(fp8,rank,0,rows,normed,a.refIds,a.refScores,a.y[0])==0,"EP reference MoE");
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
        for(int replay=0;replay<3;++replay){
            for(size_t i=0;i<x.size();++i)x[i]=__float2bfloat16_rn(std::sin(float(i%Hidden)*(.019f+.007f*replay)+i/Hidden));
            check(cudaMemcpy(a.x,x.data(),x.size()*sizeof(BF),cudaMemcpyHostToDevice));
            check(cudaGraphLaunch(exec,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            auto read=[&](auto &host,const auto *device){check(cudaMemcpy(host.data(),device,host.size()*sizeof(host[0]),cudaMemcpyDeviceToHost));};
            read(ids,a.ids);read(probs,a.probs);read(refIds,a.refIds);read(refProbs,a.refScores);
            require(ids==refIds && probs==refProbs,"EP prepared routes");
            read(ref,a.y[0]);read(got,a.y[1]);
            double peak=0;
            for(int row=0;row<rows;++row){
                double err=0,norm=0;
                for(int i=row*Hidden;i<(row+1)*Hidden;++i){double u=float(ref[i]),v=float(got[i]);err+=(u-v)*(u-v);norm+=u*u;}
                require(std::isfinite(err) && norm>0,"EP prepared invalid result");
                peak=std::max(peak,std::sqrt(err/norm));
            }
            require(peak<.003,"EP routed input or schedule mismatch");
        }
        check(cudaGraphExecDestroy(exec));
    }
}
void testResidualRouter(void *fp4, void *fp8, RankData &data) {
    check(cudaSetDevice(0));
    require(!fastshared_tp_residual_router_supported(fp8,0,0,8) &&
            !fastshared_tp_residual_router_supported(fp4,0,0,9), "residual router fallback");
    auto *hidden = data.allocate<BF>(8*Hidden);
    auto *reference = data.allocate<BF>(8*Hidden);
    auto *branch = data.allocate<BF>(8*Hidden);
    auto *normed = data.allocate<BF>(8*Hidden);
    auto *refNorm = data.allocate<BF>(8*Hidden);
    auto *gamma = data.allocate<float>(Hidden);
    auto *weight = data.allocate<float>(256*Hidden);
    auto *biasStorage = data.allocate<float>(257);
    auto *refIds = data.allocate<int>(8*8);
    auto *refProbs = data.allocate<float>(8*8);
    std::vector<float> w(256*Hidden), g(Hidden), biasHost(256);
    std::vector<BF> h(8*Hidden), b(h.size());
    for (int j=0;j<int(w.size());++j) w[j]=std::sin(j*.123f)*.003f;
    for (int j=0;j<Hidden;++j) g[j]=1.f+std::cos(j*.022f)*.1f;
    auto copy=[](auto *dst,const auto &src) {
        check(cudaMemcpy(dst,src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice));
    };
    auto read=[](auto *src,size_t count) {
        using T=std::remove_pointer_t<decltype(src)>;
        std::vector<T> v(count);check(cudaMemcpy(v.data(),src,count*sizeof(T),cudaMemcpyDeviceToHost));return v;
    };
    copy(gamma,g);copy(weight,w);
    for (int rows : {4,5,7,8}) {
        // Exercise the scalar bias fallback with the same graph-replay reference.
        auto *bias = biasStorage + (rows == 7);
        require(fastshared_tp_residual_router_supported(fp4,0,0,rows),"residual router admission");
        FastllmMoePreparedInput buf{};
        require(fastshared_tp_prepared_input(fp4,0,rows,&buf),"prepared buffers");
        cudaGraph_t graph;cudaGraphExec_t exe;
        check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        require(!fastshared_tp_residual_router(fp4,0,0,rows,hidden,branch,gamma,1e-6f,
            normed,weight,data.router,bias,data.ids,data.probs,1,1.f),"residual router launch");
        check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        check(cudaGraphInstantiate(&exe,graph,nullptr,nullptr,0));check(cudaGraphDestroy(graph));
        for(int replay=0;replay<4;++replay){
            for(int j=0;j<int(h.size());++j){
                h[j]=__float2bfloat16_rn(std::sin(j*.025f + replay));
                b[j]=__float2bfloat16_rn(std::cos(j*.014f - replay));
            }
            // Distinct top-k including shared experts on both replays.
            for(int e=0;e<256;++e) biasHost[e]=float((e*71+replay*93)%256)*.05f;
            // A flat row exercises legacy ties; a row with eight tied winners
            // exercises the shortlist tie fallback even when exactly eight qualify.
            if(replay>=2){
                std::fill(h.begin(),h.end(),BF(0.f));std::fill(b.begin(),b.end(),BF(0.f));
                for(int e=0;e<256;++e)biasHost[e]=(replay==3 && e%32==7)?1.f:0.f;
            }
            copy(bias,biasHost);copy(hidden,h);copy(reference,h);copy(branch,b);
            fastllm::cuda_norm::AddRMSNorm<BF,Hidden,true,false><<<rows,256,0,cudaStreamPerThread>>>(
                reference,branch,gamma,refNorm,Hidden,1e-6f,false);
            auto before=read(buf.epoch,1)[0];
            check(cudaGraphLaunch(exe,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
            auto actual=read(normed,rows*Hidden), expected=read(refNorm,rows*Hidden);
            auto sum=read(hidden,rows*Hidden), sumRef=read(reference,rows*Hidden);
            require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(BF)) &&
                    !std::memcmp(sum.data(),sumRef.data(),sum.size()*sizeof(BF)), "residual/norm differs");
            auto q=read(buf.values,rows*Hidden), sc=read(buf.scales,rows*Hidden/32);
            auto router=read(data.router,rows*256);auto ids=read(data.ids,rows*8);
            auto prob=read(data.probs,rows*8);
            require(read(buf.epoch,1)[0]==before+1,"residual router epoch");
            auto counters=read(buf.counters,256);
            require(std::all_of(counters.begin(),counters.end(),[](int n){return n==0;}),"gate counters");
            for(int start=0;start<rows*Hidden;start+=32){
                float mx=0;for(int i=0;i<32;++i) mx=std::max(mx,std::abs(float(expected[start+i])));
                unsigned bits;std::memcpy(&bits,&mx,4);unsigned e=std::max((bits+0x1fffffu)>>23,113u)-8;
                require(sc[start/32]==e,"K32 exponent");
                for(int i=0;i<32;++i)require(q[start+i]==__nv_fp8_e4m3(std::ldexp(float(expected[start+i]),127-int(e))).__x,"K32 value");
            }
            referenceRouterSelection<<<rows,32,0,cudaStreamPerThread>>>(data.router,bias,refIds,refProbs);
            check(cudaStreamSynchronize(cudaStreamPerThread));
            require(ids==read(refIds,rows*8) && prob==read(refProbs,rows*8),"router top-k differs");
            for(int r=0;r<rows;++r)for(int e=0;e<256;++e){
                double dot=0;for(int k=0;k<Hidden;++k)dot+=float(expected[r*Hidden+k])*double(w[e*Hidden+k]);
                require(std::abs(router[r*256+e]-1./(1.+std::exp(-dot)))<2e-6,"router projection");
            }
        }
        check(cudaGraphExecDestroy(exe));
    }
    std::puts("residual router: BF16 residual/norm, K32 bytes, top-k, graph replay PASS");
}
void test(bool ep, bool verifyPages = false) {
    const int experts=ep?32:256, inter=ep?2048:256;
    std::array<RankData,Ranks> data;
    std::array<double,Ranks> peak{};
    int devices[]={0,1,2,3,4,5,6,7};
    void *states[2]{};
    auto create=ep?fastshared_ep_create:fastshared_tp_create;
    auto pack = verifyPages ? fastshared_tp_pack_mxfp4_verify : fastshared_tp_pack_mxfp4;
    auto bind=ep?fastshared_ep_bind:fastshared_tp_bind;
    auto run=ep?fastshared_ep_run:fastshared_tp_run;
    auto destroy=ep?fastshared_ep_destroy:fastshared_tp_destroy;
    for (auto &state:states) require(create(devices,1,&state)==0,"create");
    // Layer packing transfers ownership into a shared host-side state.
    for (int r=0;r<Ranks;++r) {
        check(cudaSetDevice(r)); auto &a=data[r];
        uint8_t *compact[2], *values[2], *scales[2], *epScale[2]{};
        std::vector<uint64_t> pointers(experts*2);
        for (int part=0;part<2;++part) {
            int n=part?Hidden:2*inter,k=part?inter:Hidden;
            size_t elements=size_t(n)*k;
            compact[part]=a.allocate<uint8_t>(experts*(elements/2+(ep?0:elements/32)));
            values[part]=a.allocate<uint8_t>(experts*elements);
            scales[part]=a.allocate<uint8_t>(experts*elements/32);
            if (ep) epScale[part]=a.allocate<uint8_t>(experts*elements/32);
            fill<<<1024,256,0,cudaStreamPerThread>>>(compact[part],values[part],scales[part],
                epScale[part],r,experts,n,k,part==0,ep);
            if (!ep) for (int e=0;e<experts;++e)
                pointers[e*2+part]=reinterpret_cast<uint64_t>(compact[part]+e*(elements/2+elements/32));
        }
        check(cudaStreamSynchronize(cudaStreamPerThread));
        require(bind(states[0],r,0,values[0],values[1],scales[0],scales[1])==0,"reference bind");
        require((ep?fastshared_ep_bind_format(states[1],r,0,compact[0],compact[1],epScale[0],epScale[1],true)
                   :pack(states[1],r,0,pointers.data()))==0,"MXFP4 bind");
        a.x=a.allocate<BF>(MaxRows*Hidden); a.ids=a.allocate<int>(MaxRows*TopK);
        a.probs=a.allocate<float>(MaxRows*TopK);
        a.router=a.allocate<float>(MaxRows*256);
        a.bias=a.allocate<float>(256);
        check(cudaMemset(a.bias,0,256*sizeof(float)));
        if (ep) {
            a.publishedRouter=a.allocate<float>(MaxRows*256);
            a.refIds=a.allocate<int>(MaxRows*TopK);
            a.refScores=a.allocate<float>(MaxRows*TopK);
        }
        for (auto &y:a.y) y=a.allocate<BF>(MaxRows*Hidden);
    }
    // Return to one row after a prepared batch to cover reused counters and mailbox generations.
    for (int variant:{1,2,8,9,17,-4,-8,1}) onRanks([&](int r) {
        const bool prepared=variant<0;
        const int rows=std::abs(variant);
        auto &a=data[r]; cudaGraphExec_t graphs[2]{};
        for (int j=0;j<2;++j) {
            cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            if (!prepared) require(run(states[j],r,0,rows,a.x,a.ids,a.probs,a.y[j])==0,"capture");
            else if (ep) {
                auto *bias = rows == 8 ? a.bias : nullptr;
                const int needNorm = rows == 8;
                const float scale = 1.875f;
                if (j == 0) {
                    referenceRouterSelection<<<rows,32,0,cudaStreamPerThread>>>(a.router,bias,a.refIds,a.refScores,needNorm,scale);
                    require(run(states[j],r,0,rows,a.x,a.refIds,a.refScores,a.y[j])==0,"EP reference");
                } else {
                    publishDelayedRouter<<<rows,256,0,cudaStreamPerThread>>>(a.router,a.publishedRouter);
                    require(fastshared_ep_run_router(states[j],r,0,rows,a.x,a.publishedRouter,bias,a.ids,
                        a.probs,a.y[j],needNorm,scale)==0,"EP routed capture");
                }
            } else if (!j) {
                require(fastshared_tp_run_router(states[j],r,0,rows,a.x,a.router,rows==8 ? a.bias : nullptr,
                    a.ids,a.probs,a.y[j],1,1.f)==0,"reference routing");
            } else {
                FastllmMoePreparedInput input{};
                require(fastshared_tp_prepared_input(states[j],r,rows,&input) && !input.block128,
                    "native K32 prepared input");
                prepareInput<<<rows*Hidden/256,256,0,cudaStreamPerThread>>>(a.x,input);
                require(fastshared_tp_run_prepared(states[j],r,0,rows,a.router,rows==8 ? a.bias : nullptr,
                    a.ids,a.probs,a.y[j],1,1.f)==0,"prepared routing");
            }
            check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            check(cudaGraphInstantiate(&graphs[j],graph,nullptr,nullptr,0)); check(cudaGraphDestroy(graph));
        }
        for (int replay=0;replay<3;++replay) {
            std::vector<int> ids(rows*TopK);
            std::vector<float> probs(rows*TopK);
            std::vector<BF> x(rows*Hidden),ref(x.size()),result(x.size());
            for (size_t i=0;i<x.size();++i) x[i]=__float2bfloat16((int((i*37+replay)%299)-147)*.007f);
            for (int i=0;i<rows*TopK;++i) {
                int row=i/TopK,slot=i%TopK;
                ids[i]=replay==0?slot:replay==1?(row*11+slot*32)%256:(row/4*17+slot*3)%256;
                probs[i]=(slot+1)/36.f;
            }
            if (prepared) {
                std::vector<float> router(rows*256,0.f);
                for (int i=0;i<rows*TopK;++i) router[(i/TopK)*256+ids[i]]=probs[i];
                if (ep && replay == 2) std::fill(router.begin(),router.end(),.5f);
                check(cudaMemcpy(a.router,router.data(),router.size()*sizeof(float),cudaMemcpyHostToDevice));
            }
            check(cudaMemcpy(a.x,x.data(),x.size()*sizeof(BF),cudaMemcpyHostToDevice));
            check(cudaMemcpy(a.ids,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice));
            check(cudaMemcpy(a.probs,probs.data(),probs.size()*sizeof(float),cudaMemcpyHostToDevice));
            for (auto graph:graphs) check(cudaGraphLaunch(graph,cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            if (ep && prepared) {
                std::vector<int> expectedIds(ids.size());
                std::vector<float> expectedScores(probs.size());
                check(cudaMemcpy(expectedIds.data(),a.refIds,ids.size()*sizeof(int),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(expectedScores.data(),a.refScores,probs.size()*sizeof(float),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(ids.data(),a.ids,ids.size()*sizeof(int),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(probs.data(),a.probs,probs.size()*sizeof(float),cudaMemcpyDeviceToHost));
                require(ids == expectedIds && probs == expectedScores,"EP top-k differs from standalone selection");
            }
            check(cudaMemcpy(ref.data(),a.y[0],ref.size()*sizeof(BF),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(result.data(),a.y[1],ref.size()*sizeof(BF),cudaMemcpyDeviceToHost));
            double worst=0;
            for (int row=0;row<rows;++row) {
                double err=0,norm=0;
                for (int i=row*Hidden;i<(row+1)*Hidden;++i) {
                    double u=__bfloat162float(ref[i]),v=__bfloat162float(result[i]);
                    err+=(u-v)*(u-v); norm+=u*u;
                }
                require(std::isfinite(err) && norm>0,"invalid MoE output");
                worst=std::max(worst,std::sqrt(err/norm));
            }
            peak[r]=std::max(peak[r],worst);
            if (r==0) printf("%s rows=%d replay=%d worst_row_l2=%.8g\n",ep?"EP":prepared?"TP-prepared":"TP",rows,replay,worst);
            require(worst<.003,"MXFP4 output differs from equivalent FP8 weights");
        }
        for (auto graph:graphs) check(cudaGraphExecDestroy(graph));
    });
    printf("%s all_rank_max_row_l2=%.8g\n",ep?"EP":verifyPages?"TP-pages":"TP",*std::max_element(peak.begin(),peak.end()));
    if (ep) onRanks([&](int r){testEpPrepared(states[1],states[0],r,data[r]);});
    if (!ep) testResidualRouter(states[1],states[0],data[0]);
    for (auto state:states) destroy(state);
    onRanks([&](int r) { for (auto p:data[r].buffers) check(cudaFree(p)); });
}
}
int main() {
    int devices[]={0,1,2,3,4,5,6,7};
    if (!fastshared_tp_supported(devices,8) || !fastshared_ep_supported(devices,8)) return 77;
    try { test(false); test(false,true); test(true); }
    catch (const std::exception &e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
