// MXFP4 expert groups: compare with the existing SIMT arithmetic across
// repeated/missing/shuffled routes and changed-input CUDA graph replays.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include "moe/fastllm-moe-mxfp4-batch.cuh"
using namespace fastllm::cuda::dsv41_cache;
static bool benchmark = false;
static void C(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void *p=nullptr;Buffer(size_t n){C(cudaMalloc(&p,n));C(cudaMemset(p,0,n));}~Buffer(){cudaFree(p);}};
static void Case(int rows,int topk,int h,int m,int reuse,bool planar){
 int routes=rows*topk,experts=(routes+reuse-1)/reuse;
 size_t gateBytes=size_t(2*m)*h/32*17,downBytes=size_t(h)*m/32*17,stride=gateBytes+downBytes;
 std::mt19937 rng(11+rows+topk);std::vector<uint8_t> records(size_t(experts)*stride);
 for(auto &b:records)b=rng();
 for(int e=0;e<experts;++e)for(int part=0;part<2;++part){int n=part?h:2*m,k=part?m:h;uint8_t *w=records.data()+e*stride+(part?gateBytes:0);
  for(int r=0;r<n;++r)for(int b=0;b<k/32;++b)w[(size_t(r)*k/32+b)*17+16]=116+rng()%5;
  if(planar){std::vector<uint8_t> src(w,w+size_t(n)*k/32*17);for(int r=0;r<n;++r)for(int b=0;b<k/32;++b){
   std::memcpy(w+size_t(r)*k/2+b*16,src.data()+(size_t(r)*k/32+b)*17,16);
   w[size_t(n)*k/2+size_t(r)*k/32+b]=src[(size_t(r)*k/32+b)*17+16];}}
 }
 std::vector<int> slots(routes),map(routes);std::vector<float> scores(routes);std::vector<uint16_t> input(size_t(rows)*h);
 for(int i=0;i<routes;++i){slots[i]=(i%11==10)?-1:i/reuse;map[i]=i;scores[i]=(rng()%100)/100.f;}
 std::shuffle(map.begin(),map.end(),rng);
 Buffer w(records.size()),x(input.size()*2),s(routes*4),r(routes*4),sc(routes*4),a(size_t(routes)*m*2),b(size_t(routes)*m*2),y(size_t(routes)*h*4),z(size_t(routes)*h*4);
 C(cudaMemcpy(w.p,records.data(),records.size(),cudaMemcpyHostToDevice));C(cudaMemcpy(s.p,slots.data(),routes*4,cudaMemcpyHostToDevice));C(cudaMemcpy(r.p,map.data(),routes*4,cudaMemcpyHostToDevice));C(cudaMemcpy(sc.p,scores.data(),routes*4,cudaMemcpyHostToDevice));
 auto launch=[&](bool shared){auto *act=(__nv_bfloat16*)(shared?b.p:a.p);auto *out=(float*)(shared?z.p:y.p);
  if(shared){GateReuse<true><<<dim3(m/32,routes),512,0,cudaStreamPerThread>>>((__nv_bfloat16*)x.p,(int*)s.p,(uint8_t*)w.p,(float*)sc.p,act,h,m,stride,7,topk,(int*)r.p,routes,planar);
   GateMmaBatch<<<dim3(m/32,(routes+15)/16),512,0,cudaStreamPerThread>>>((__nv_bfloat16*)x.p,(int*)s.p,(uint8_t*)w.p,(float*)sc.p,act,h,m,stride,7,topk,(int*)r.p,routes,planar);
   DownReuse<true><<<dim3(h/8,routes),128,0,cudaStreamPerThread>>>(act,(int*)s.p,(uint8_t*)w.p,out,h,m,stride,gateBytes,(int*)r.p,routes,planar);
   DownMmaBatch<<<dim3(h/16,(routes+15)/16),256,0,cudaStreamPerThread>>>(act,(int*)s.p,(uint8_t*)w.p,out,h,m,stride,gateBytes,(int*)r.p,routes,planar);
  }else{GateReuse<<<dim3(m/32,routes),512,0,cudaStreamPerThread>>>((__nv_bfloat16*)x.p,(int*)s.p,(uint8_t*)w.p,(float*)sc.p,act,h,m,stride,7,topk,(int*)r.p,routes,planar);
   DownReuse<<<dim3(h/8,routes),128,0,cudaStreamPerThread>>>(act,(int*)s.p,(uint8_t*)w.p,out,h,m,stride,gateBytes,(int*)r.p,routes,planar);}C(cudaGetLastError());};
 cudaGraph_t graphs[2];cudaGraphExec_t exec[2];
 for(int which=0;which<2;++which){C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(which);C(cudaStreamEndCapture(cudaStreamPerThread,&graphs[which]));C(cudaGraphInstantiate(&exec[which],graphs[which],0));}
 for(int repeat=0;repeat<3;++repeat){for(auto &v:input){auto f=__float2bfloat16_rn((int(rng()%1025)-512)/256.f);std::memcpy(&v,&f,2);}C(cudaMemcpy(x.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  for(int which=0;which<2;++which)C(cudaGraphLaunch(exec[which],cudaStreamPerThread));C(cudaDeviceSynchronize());
  for(auto pair:{std::pair<void*,void*>{a.p,b.p},{y.p,z.p}}){size_t bytes=(pair.first==a.p)?size_t(routes)*m*2:size_t(routes)*h*4;std::vector<uint8_t> u(bytes),v(bytes);
   C(cudaMemcpy(u.data(),pair.first,bytes,cudaMemcpyDeviceToHost));C(cudaMemcpy(v.data(),pair.second,bytes,cudaMemcpyDeviceToHost));double error=0,norm=0;for(size_t i=0;i<bytes/(pair.first==a.p?2:4);++i){
    const double av=pair.first==a.p?__bfloat162float(((__nv_bfloat16*)u.data())[i]):((float*)u.data())[i];
    const double bv=pair.first==a.p?__bfloat162float(((__nv_bfloat16*)v.data())[i]):((float*)v.data())[i];
    if(!std::isfinite(bv))throw std::runtime_error("nonfinite");error+=(av-bv)*(av-bv);norm+=av*av;}
   const double l2=std::sqrt(error/std::max(norm,1e-30));
   printf("CHECK rows=%d topk=%d hidden=%d inter=%d reuse=%d planar=%d part=%s repeat=%d l2=%.9g\n",rows,topk,h,m,reuse,int(planar),pair.first==a.p?"gate":"down",repeat,l2);fflush(stdout);
   if(l2>.004)throw std::runtime_error("MMA reference L2");}
 }
 if(benchmark && h==5120){for(int which=0;which<2;++which)for(int rep=0;rep<3;++rep){
  for(int i=0;i<5;++i)C(cudaGraphLaunch(exec[which],cudaStreamPerThread));cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));C(cudaEventRecord(begin,cudaStreamPerThread));
  for(int i=0;i<30;++i)C(cudaGraphLaunch(exec[which],cudaStreamPerThread));C(cudaEventRecord(end,cudaStreamPerThread));C(cudaEventSynchronize(end));float ms;C(cudaEventElapsedTime(&ms,begin,end));C(cudaEventDestroy(begin));C(cudaEventDestroy(end));
  printf("BENCH rows=%d topk=%d reuse=%d planar=%d which=%d rep=%d us=%.6f\n",rows,topk,reuse,int(planar),which,rep,ms*1000/30);fflush(stdout);}}
 for(int i=0;i<2;++i){C(cudaGraphExecDestroy(exec[i]));C(cudaGraphDestroy(graphs[i]));}
}
int main(int argc, char **argv){benchmark=argc>1 && std::strcmp(argv[1],"--bench")==0;try{int count=0;for(int rows:{3,6,9,12,16,17,24,33})for(int topk:{3,6})for(int reuse:{1,2,3,6,12,17})for(bool planar:{false,true}){Case(rows,topk,256,128,reuse,planar);++count;}
 for(int rows:{6,12,24})for(int reuse:{1,2,3,6,12})for(bool planar:{false,true}){Case(rows,6,5120,2304,reuse,planar);++count;}printf("PASS %d MMA route cases, changed-input graph replay\n",count);return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
