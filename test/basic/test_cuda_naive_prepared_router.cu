// Validate BF16/FP8 preparation and the FP32 router error contract.
// Includes unprepared four-row reuse, ragged tiles, guards, routing and changing-input Graph replay.
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../../src/devices/multicuda/fastllm-attention-output-rows.cuh"
#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>
using namespace fastllm_swa_output;
void ck(cudaError_t s){if(s)throw std::runtime_error(cudaGetErrorString(s));}
void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
template<class T>T*alloc(size_t n){T*p;ck(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T>std::vector<T>read(T*p,size_t n){std::vector<T>a(n);ck(cudaMemcpy(a.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return a;}
__global__ void TestRoutes(const float*router,int rows,int*ids,float*scores){
 __shared__ fastllm::cuda::RouterTopKScratch<8> scratch[8];
 const int warp=threadIdx.x/32,row=blockIdx.x*8+warp;
 if(row<rows)fastllm::cuda::SelectRouterTopK<256,8>(router+row*256,nullptr,ids+row*8,scores+row*8,false,1,2.5f,scratch[warp]);
}
template<class T> bool SameBits(const std::vector<T>&a,const std::vector<T>&b) {
 return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);
}
// The direct dot changes FP32 reduction order; BF16 normalization and the
// prepared FP8 image must still match the ordered reference bit for bit.
bool RouterClose(const std::vector<float>&a,const std::vector<float>&b) {
 if(a.size()!=b.size())return false;
 for(size_t i=0;i<a.size();++i)
  if(!std::isfinite(a[i]) || !std::isfinite(b[i]) || fabsf(a[i]-b[i])>1e-5f)return false;
 return true;
}
void CheckBlock128(const FastllmMoePreparedInput &p,int rows) {
 if(!p.block128)return;
 auto q=read(p.values,rows*4096),sc=read(p.scales,rows*128);
 std::vector<uint8_t>expected(33920,0);auto actual=read(p.block128,33920);
 for(int row=0;row<8;++row)for(int k=0;k<32;++k){
  unsigned e=105;
  if(row<rows)for(int j=0;j<4;++j)e=std::max(e,unsigned(sc[row*128+k*4+j]));
  float scale=std::ldexp(1.f,int(e)-127);
  memcpy(expected.data()+32896+(k*8+row)*4,&scale,4);
  for(int j=0;j<128;++j){
   __nv_fp8_e4m3 v;v.__x=row<rows?q[row*4096+k*128+j]:0;
   const int old=row<rows?sc[row*128+k*4+j/32]:105;
   expected[row*4112+k*128+j]=__nv_fp8_e4m3(float(v)*std::ldexp(1.f,old-int(e))).__x;
  }
 }
 need(actual==expected,"block128 differs from independent MXFP8 conversion");
 for(auto v:read(p.block128-64,64))need(v==0x55,"block128 prefix guard");
 for(auto v:read(p.block128+33920,64))need(v==0x55,"block128 suffix guard");
}
#if CUDART_VERSION >= 12000
static __global__ void DelayedResidual(const uint4 *input, uint4 *hidden) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
    const auto start = clock64();
    while (clock64() - start < 40000) __nanosleep(64);
#endif
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    hidden[index] = input[index];
}

template<bool Prepare>
static void CheckRouterDependency(BF16 *input, const float *weights, const float *gamma,
        BF16 *norm[2], float *scores[2], FastllmMoePreparedInput prepared[2]) {
    constexpr int rows = 8, width = 4096;
    const void *consumer = Prepare ? (const void *)NormRowsRouterDirect<8, 2, true, 8, false, true>
                                      : (const void *)NormRowsRouterExperts<8, false, true>;
    cudaFuncAttributes attributes{};
    ck(cudaFuncGetAttributes(&attributes, consumer));
    if (attributes.ptxVersion < 90 || attributes.binaryVersion < 90) return;
    BF16 *hidden = alloc<BF16>(rows * width);
    for (int i = 0; i < 2; ++i) {
        prepared[i].block128 = nullptr;
        prepared[i].overlapVerifyGate = true;
        ck(cudaMemset(prepared[i].epoch, 0, sizeof(uint64_t)));
    }
    auto stream = cudaStreamPerThread;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    DelayedResidual<<<128, 32, 0, stream>>>((const uint4 *)input, (uint4 *)hidden);
    if constexpr (Prepare)
        ck(fastllm::cuda_launch::LaunchProgrammatic<NormRowsRouterDirect<8, 2, true, 8, false, true>>(
            dim3(32, 4), 256, 0, stream, hidden, weights, scores[1], gamma, norm[1], 1e-6f, rows,
            prepared[1], (const BF16 *)nullptr, (BF16 *)nullptr));
    else
        ck(fastllm::cuda_launch::LaunchProgrammatic<NormRowsRouterExperts<8, false, true>>(
            dim3(32, 4), 256, 0, stream, hidden, weights, scores[1], gamma, norm[1], 1e-6f, rows,
            FastllmMoePreparedInput{}));
    ck(cudaStreamEndCapture(stream, &graph));
    ck(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<BF16> values(rows * width);
    for (int replay = 0; replay < 4; ++replay) {
        for (int i = 0; i < rows * width; ++i)
            values[i] = __float2bfloat16_rn(float((i + 7 * replay) % 31 - 15));
        ck(cudaMemcpyAsync(input, values.data(), values.size() * sizeof(BF16), cudaMemcpyHostToDevice, stream));
        ck(cudaMemsetAsync(hidden, 0xff, values.size() * sizeof(BF16), stream));
        if constexpr (!Prepare)
            NormRowsRouterExperts<4><<<dim3(64, 4), 256, 0, stream>>>(input, weights, scores[0],
                gamma, norm[0], 1e-6f, rows);
        else NormRowsRouterDirect<8><<<dim3(32, 4), 256, 0, stream>>>(input, weights, scores[0],
            gamma, norm[0], 1e-6f, rows, prepared[0]);
        ck(cudaGraphLaunch(exec, stream));
        ck(cudaStreamSynchronize(stream));
        need(SameBits(read(scores[0], rows * 256), read(scores[1], rows * 256)), "PDL router scores");
        need(SameBits(read(norm[0], values.size()), read(norm[1], values.size())), "PDL normalized rows");
        if constexpr (!Prepare) continue;
        need(read(prepared[0].values, rows * width) == read(prepared[1].values, rows * width), "PDL FP8 values");
        need(read(prepared[0].scales, rows * 128) == read(prepared[1].scales, rows * 128), "PDL FP8 scales");
        need(read(prepared[0].epoch, 1) == read(prepared[1].epoch, 1), "PDL routing epoch");
    }
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaFree(hidden));
}
#endif

void ValidateWideRouter();
int main(){try{
 int devices=0;cudaDeviceProp prop{};
 if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0 ||
    cudaGetDeviceProperties(&prop,0)!=cudaSuccess || prop.major!=10 || prop.minor!=3){
  std::puts("SKIP: prepared vector router is currently admitted on SM103");return 77;
 }
 ck(cudaSetDevice(0));
 constexpr int rowCounts[]={1,2,3,4,5,6,7,8,9,13,16,17,31,32,33,64,65};
 constexpr size_t rowCount=sizeof(rowCounts)/sizeof(rowCounts[0]);
 const int maxRows=rowCounts[rowCount-1]+1;auto*x=alloc<BF16>(maxRows*4096);auto*w=alloc<float>(256*4096);auto*gamma=alloc<float>(4096);
 int*routeIds[2];float*routeScores[2];for(int i=0;i<2;++i){routeIds[i]=alloc<int>(maxRows*8);routeScores[i]=alloc<float>(maxRows*8);}
 BF16*norm[2];float*out[2];FastllmMoePreparedInput prepared[2];
 for(int i=0;i<2;++i){norm[i]=alloc<BF16>(maxRows*4096);out[i]=alloc<float>(maxRows*256);prepared[i].values=alloc<uint8_t>(maxRows*4096);prepared[i].scales=alloc<uint8_t>(maxRows*128);prepared[i].counters=alloc<int>(256);prepared[i].epoch=alloc<uint64_t>(1);}
 uint8_t *extra[2];for(int j=0;j<2;++j){extra[j]=alloc<uint8_t>(33920+128);ck(cudaMemset(extra[j],0x55,33920+128));}
 std::mt19937 rng(9219);std::normal_distribution<float>dist;double maxL2=0,maxAbs=0;int cases=0;
 enum class Variant { Vector, PreparedDirect, RowsDirect };
 for(auto variant:{Variant::Vector,Variant::PreparedDirect,Variant::RowsDirect}){rng.seed(9219);
 const bool withPrepared=variant!=Variant::RowsDirect, direct=variant!=Variant::Vector;
 for(int rows:rowCounts)for(int seed=0;seed<16;++seed){
  for(int j=0;j<2;++j)prepared[j].block128=(withPrepared&&rows>=4&&rows<=8)?extra[j]+64:nullptr;
  float amp=seed%4==0?0.f:seed%4==1?.01f:seed%4==2?1.f:100.f;
  std::vector<BF16>h(rows*4096);std::vector<float>hw(256*4096),hg(4096);
  for(auto&v:h)v=__float2bfloat16_rn(dist(rng)*amp);for(auto&v:hw)v=dist(rng)*(seed<8?.02f:.2f);for(auto&v:hg)v=1.f+dist(rng)*.2f;
  ck(cudaMemcpy(x,h.data(),h.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(gamma,hg.data(),hg.size()*4,cudaMemcpyHostToDevice));
  auto launch=[&](int mode) {
   if(!withPrepared){
    if(!mode)NormRowsRouterExperts<4><<<dim3(64,(rows+1)/2),256,0,cudaStreamPerThread>>>(x,w,out[0],gamma,norm[0],1e-6f,rows);
    else NormRowsRouterDirect<8,4,false><<<dim3(32,(rows+3)/4),256,0,cudaStreamPerThread>>>(x,w,out[1],gamma,norm[1],1e-6f,rows,{});
   }
   else if(!mode)NormRowsRouterExperts<4,true><<<dim3(64,(rows+1)/2),256,0,cudaStreamPerThread>>>(x,w,out[0],gamma,norm[0],1e-6f,rows,prepared[0]);
   else if(direct)NormRowsRouterDirect<8><<<dim3(32,(rows+1)/2),256,0,cudaStreamPerThread>>>(x,w,out[1],gamma,norm[1],1e-6f,rows,prepared[1]);
   else if(rows<=2)NormRowsRouterPreparedVector<2><<<dim3(128,(rows+1)/2),256,0,cudaStreamPerThread>>>(x,w,out[1],gamma,norm[1],1e-6f,rows,prepared[1]);
   else NormRowsRouterPreparedVector<4><<<dim3(64,(rows+1)/2),256,0,cudaStreamPerThread>>>(x,w,out[1],gamma,norm[1],1e-6f,rows,prepared[1]);
  };
  {
   for(int j=0;j<2;++j){ck(cudaMemset(norm[j],0x55,maxRows*4096*2));ck(cudaMemset(out[j],0x55,maxRows*256*4));
    if(withPrepared){ck(cudaMemset(prepared[j].values,0x55,maxRows*4096));ck(cudaMemset(prepared[j].scales,0x55,maxRows*128));
     ck(cudaMemset(prepared[j].counters,0x55,256*sizeof(int)));
     const uint64_t epoch=9;ck(cudaMemcpy(prepared[j].epoch,&epoch,sizeof(epoch),cudaMemcpyHostToDevice));}
    launch(j);}
   ck(cudaDeviceSynchronize());auto n0=read(norm[0],maxRows*4096),n1=read(norm[1],maxRows*4096);need(memcmp(n0.data(),n1.data(),n0.size()*2)==0,"norm or norm guard changed");
   if(withPrepared){auto p0=read(prepared[0].values,maxRows*4096),p1=read(prepared[1].values,maxRows*4096),s0=read(prepared[0].scales,maxRows*128),s1=read(prepared[1].scales,maxRows*128);need(p0==p1&&s0==s1,"quantized values/scales/guards changed");}
   auto a=read(out[0],maxRows*256),b=read(out[1],maxRows*256);need(memcmp(a.data()+rows*256,b.data()+rows*256,(maxRows-rows)*256*4)==0,"router guard changed");
   need(direct?RouterClose(a,b):SameBits(a,b),"router scores outside contract");
   for(int row=0;row<rows;++row){double diff=0,mag=0;for(int k=row*256;k<(row+1)*256;++k){need(std::isfinite(a[k])&&std::isfinite(b[k]),"nonfinite router");double d=a[k]-b[k];diff+=d*d;mag+=double(a[k])*a[k];maxAbs=fmax(maxAbs,fabs(d));}double l2=sqrt(diff/fmax(mag,1e-30));need(l2<.00001,"router L2 > 1e-5");maxL2=fmax(maxL2,l2);}
   for(int j=0;j<2;++j)TestRoutes<<<(rows+7)/8,256,0,cudaStreamPerThread>>>(out[j],rows,routeIds[j],routeScores[j]);
   ck(cudaStreamSynchronize(cudaStreamPerThread));
   need(read(routeIds[0],rows*8)==read(routeIds[1],rows*8),"top-k expert IDs changed");
   need(direct?RouterClose(read(routeScores[0],rows*8),read(routeScores[1],rows*8)):SameBits(read(routeScores[0],rows*8),read(routeScores[1],rows*8)),"route probability outside contract");
   if(withPrepared)for(int j=0;j<2;++j){
    need(read(prepared[j].epoch,1)[0]==10,"epoch did not advance exactly once");
    for(int count:read(prepared[j].counters,256))need(count==0,"counter was not reset");
    CheckBlock128(prepared[j],rows);
   }
   ++cases;
  }
  if(seed==0){cudaGraph_t g;cudaGraphExec_t ex;ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(0);launch(1);ck(cudaStreamEndCapture(cudaStreamPerThread,&g));ck(cudaGraphInstantiate(&ex,g,0));
   for(int it=0;it<10;++it){for(auto&v:h)v=__float2bfloat16_rn(dist(rng));ck(cudaMemcpyAsync(x,h.data(),h.size()*2,cudaMemcpyHostToDevice,cudaStreamPerThread));ck(cudaGraphLaunch(ex,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));auto a=read(norm[0],rows*4096),b=read(norm[1],rows*4096);need(memcmp(a.data(),b.data(),a.size()*2)==0,"dynamic graph norm");
   if(withPrepared){need(read(prepared[0].values,rows*4096)==read(prepared[1].values,rows*4096),"dynamic graph quant");
    need(read(prepared[0].scales,rows*128)==read(prepared[1].scales,rows*128),"dynamic graph scales");
    for(int j=0;j<2;++j){CheckBlock128(prepared[j],rows);
     need(read(prepared[j].epoch,1)[0]==uint64_t(11+it),"dynamic graph epoch");}}
   need(direct?RouterClose(read(out[0],rows*256),read(out[1],rows*256)):SameBits(read(out[0],rows*256),read(out[1],rows*256)),"dynamic graph router outside contract");
   for(int j=0;j<2;++j)TestRoutes<<<(rows+7)/8,256,0,cudaStreamPerThread>>>(out[j],rows,routeIds[j],routeScores[j]);
   ck(cudaStreamSynchronize(cudaStreamPerThread));need(read(routeIds[0],rows*8)==read(routeIds[1],rows*8),"dynamic graph top-k");
   }
   ck(cudaGraphExecDestroy(ex));ck(cudaGraphDestroy(g));}
 }
 }
#if CUDART_VERSION >= 12000
 CheckRouterDependency<true>(x, w, gamma, norm, out, prepared);
 CheckRouterDependency<false>(x, w, gamma, norm, out, prepared);
#endif
 ValidateWideRouter();
 printf("ROUTER_VALIDATE_PASS cases=%d max_row_router_rel_l2=%.9g max_abs=%.9g norm_fp8_scale_guard_bitwise=1 dynamic_graph_rows=%zu\n",cases,maxL2,maxAbs,rowCount);
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

// Independent reference: quantize each 32-wide block, then rebase four scales.
__global__ void ReferenceWide(const BF16*x,int rows,uint8_t*out,int*counts,uint64_t*epoch){
 int row=blockIdx.x/32,k=blockIdx.x%32,t=threadIdx.x;
 if(blockIdx.x==0){for(int i=t;i<256;i+=128)counts[i]=0;if(t==0)++*epoch;}
 float v=row<rows?float(x[row*4096+k*128+t]):0.f,m=fabsf(v);
 for(int d=16;d;d>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
 unsigned e=max((__float_as_uint(m)+0x7fffffu-0x600000u)>>23,113u)-8;
 __nv_fp8_e4m3 q(v*__uint_as_float((254-e)<<23));
 __shared__ unsigned es[4];if(t%32==0)es[t/32]=e;__syncthreads();
 unsigned c=__reduce_max_sync(0xffffffff,t%32<4?es[t%32]:0);
 out[row*4112+k*128+t]=__nv_fp8_e4m3(float(q)*__uint_as_float(unsigned(max(0,127+int(e)-int(c)))<<23)).__x;
 if(t==0)reinterpret_cast<float*>(out+16*4112)[k*16+row]=__uint_as_float(c<<23);
 if(k==0&&t<16)out[row*4112+4096+t]=0;
}
void ValidateWideRouter(){
 ck(cudaSetDevice(0));constexpr int B=16*4240, Modes=2;
 auto*x=alloc<BF16>(16*4096);auto*w=alloc<float>(256*4096);auto*g=alloc<float>(4096);
 BF16*n[Modes];float*y[Modes];FastllmMoePreparedInput p[Modes];uint8_t*buf[Modes];
 for(int j=0;j<Modes;++j){n[j]=alloc<BF16>(17*4096);y[j]=alloc<float>(17*256);buf[j]=alloc<uint8_t>(B+128);p[j].block128=buf[j]+64;p[j].counters=alloc<int>(256);p[j].epoch=alloc<uint64_t>(1);}
 std::mt19937 gen(78912);std::normal_distribution<float>d;std::vector<BF16>hx(16*4096);std::vector<float>hw(256*4096),hg(4096);
 for(auto&v:hw)v=.02f*d(gen);for(auto&v:hg)v=1.f+.2f*d(gen);
 ck(cudaMemcpy(w,hw.data(),hw.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(g,hg.data(),hg.size()*4,cudaMemcpyHostToDevice));
 auto run=[&](int mode,int rows,const float*weight){
  if(mode==0){NormRowsRouterDirect<8,4,false><<<dim3(32,(rows+3)/4),256,0,cudaStreamPerThread>>>(x,weight,y[0],g,n[0],1e-6f,rows,{});ReferenceWide<<<512,128,0,cudaStreamPerThread>>>(n[0],rows,p[0].block128,p[0].counters,p[0].epoch);}
  if(mode==1)NormRowsRouterDirect<8,4,true,16><<<dim3(32,4),256,0,cudaStreamPerThread>>>(x,weight,y[1],g,n[1],1e-6f,rows,p[1]);
 };
 for(int rows=9;rows<=16;++rows){
  cudaGraph_t graph;cudaGraphExec_t exec;
  ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));for(int m=0;m<Modes;++m)run(m,rows,w);ck(cudaStreamEndCapture(cudaStreamPerThread,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
  for(int seed=0;seed<32;++seed){
   float amplitude=seed%4==0?0.f:seed%4==1?.01f:seed%4==2?1.f:100.f;
   for(auto&v:hx)v=__float2bfloat16_rn(amplitude*d(gen));ck(cudaMemcpy(x,hx.data(),hx.size()*2,cudaMemcpyHostToDevice));
   for(int j=0;j<Modes;++j){ck(cudaMemset(n[j],0x55,17*4096*2));ck(cudaMemset(y[j],0x55,17*256*4));ck(cudaMemset(buf[j],0x55,B+128));ck(cudaMemset(p[j].counters,0x55,1024));uint64_t ep=123+seed;ck(cudaMemcpy(p[j].epoch,&ep,8,cudaMemcpyHostToDevice));}
   if(seed%2)ck(cudaGraphLaunch(exec,cudaStreamPerThread));else for(int j=0;j<Modes;++j)run(j,rows,w);
   ck(cudaStreamSynchronize(cudaStreamPerThread));
   for(int j=1;j<Modes;++j){need(SameBits(read(n[0],17*4096),read(n[j],17*4096)),"wide norm/guard differs");need(SameBits(read(y[0],17*256),read(y[j],17*256)),"wide router/guard differs");need(read(buf[0],B+128)==read(buf[j],B+128),"wide FP8 block128/guard differs");}
   for(int j=0;j<Modes;++j){need(read(p[j].epoch,1)[0]==uint64_t(124+seed),"wide epoch");for(int v:read(p[j].counters,256))need(v==0,"wide counters");}
  }
  ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
 }
 puts("WIDE_ROUTER_PASS rows9..16 seeds32 changing_graph norm_router_block128_guard_bitwise=1");
 for(int j=0;j<Modes;++j){ck(cudaFree(n[j]));ck(cudaFree(y[j]));ck(cudaFree(buf[j]));ck(cudaFree(p[j].counters));ck(cudaFree(p[j].epoch));}
 ck(cudaFree(x));ck(cudaFree(w));ck(cudaFree(g));
}
