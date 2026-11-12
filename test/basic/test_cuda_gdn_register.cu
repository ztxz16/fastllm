#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>
#include <memory>
#include <stdexcept>
#include <vector>
#include "utils/kernel_registry.h"
using namespace fastllm;

static void Check(bool x,const char*s){if(!x)throw std::runtime_error(s);}
static void Cuda(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void Alloc(Data &d){d.ToDevice(CUDA,{0},false);d.Allocate(false);}
static void Fill(Data &d,int seed,float scale=1.f){std::mt19937 rng(seed);std::uniform_real_distribution<float> f(-scale,scale);
 if(d.dataType==FLOAT32){std::vector<float>x(d.Count(0));for(auto&v:x)v=f(rng);Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
 else {std::vector<half>x(d.Count(0));for(auto&v:x)v=__float2half(f(rng));Cuda(cudaMemcpy(d.cudaData,x.data(),d.GetBytes(),cudaMemcpyHostToDevice));}}
static double L2(Data&a,Data&b){std::vector<float>x(a.Count(0)),y(x.size());
 if(a.dataType==FLOAT32){Cuda(cudaMemcpy(x.data(),a.cudaData,a.GetBytes(),cudaMemcpyDeviceToHost));Cuda(cudaMemcpy(y.data(),b.cudaData,b.GetBytes(),cudaMemcpyDeviceToHost));}
 else {std::vector<half>h(x.size()),k(x.size());Cuda(cudaMemcpy(h.data(),a.cudaData,a.GetBytes(),cudaMemcpyDeviceToHost));Cuda(cudaMemcpy(k.data(),b.cudaData,b.GetBytes(),cudaMemcpyDeviceToHost));for(size_t i=0;i<x.size();++i){x[i]=__half2float(h[i]);y[i]=__half2float(k[i]);}}
 double s=0,e=0;for(size_t i=0;i<x.size();++i){Check(std::isfinite(x[i])&&std::isfinite(y[i]),"nonfinite");s+=double(y[i])*y[i];e+=double(x[i]-y[i])*(x[i]-y[i]);}return sqrt(e/std::max(s,1e-30));}
struct State {
 Data q,a,b,g,s,o,refS,refO;
 State(int h,int k,int seed):q(FLOAT32,{1,1,(2*k+h)*128}),a(FLOAT16,{1,1,h}),b(FLOAT16,a.dims),g(FLOAT16,{1,1,h*128}),s(FLOAT32,{1,h,128,128}),o(FLOAT16,g.dims),refS(FLOAT32,s.dims),refO(FLOAT16,g.dims){
  for(Data*d:{&q,&a,&b,&g,&s,&o,&refS,&refO})Alloc(*d);
  Fill(q,seed);Fill(a,seed+1,2);Fill(b,seed+2,2);Fill(g,seed+3,2);Fill(s,seed+4,.2f);Cuda(cudaMemcpy(refS.cudaData,s.cudaData,s.GetBytes(),cudaMemcpyDeviceToDevice));
 }
};
static void Launch(bool reference,const std::vector<State*>&all,Data&al,Data&bias,Data&norm,int kh,int vh,bool tiled){
 auto run=[&]{
  if(tiled){Check(all.size()==1,"tiled test uses single entry");auto*p=all[0];
   Check(FastllmCudaQwen4GatedDeltaRuleDecodeOutput(p->q,p->a,p->b,al,bias,norm,p->g,
       reference?p->refS:p->s,reference?p->refO:p->o,kh,vh,1e-6f,1e-6f,true),"single rejected");
  }else{std::vector<FastllmCudaQwen4GdnDecodeRequest>rs;for(auto*p:all)rs.push_back({&p->q,&p->a,&p->b,&p->g,reference?&p->refS:&p->s,reference?&p->refO:&p->o});Check(FastllmCudaQwen4GatedDeltaRuleDecodeOutputBatch(rs,al,bias,norm,kh,vh,1e-6f,1e-6f),"batch rejected");}
 };
 if(reference){kernels::Bypass bypass;run();}else run();
}
static void Case(int count,int vh,bool tiled,const std::string&candidate){
 int kh=vh/3;
 std::string error;json11::Json::object config{{"mode","quick"},{"families",json11::Json::array{"gdn.decode.fp16"}},{"budget_ms",30000}};
 if(!candidate.empty())config["force"]=json11::Json::object{{"gdn.decode.fp16",candidate}};
 Check(kernels::Configure(json11::Json(config).dump(),error),error.c_str());
 Data al(FLOAT32,{vh}),bias(FLOAT32,{vh}),norm(FLOAT32,{128});for(Data*d:{&al,&bias,&norm}){Alloc(*d);Fill(*d,7);}
 // A slowly decaying state exposes error accumulation across recurrent steps.
 std::vector<float> slow(vh,-5.f);Cuda(cudaMemcpy(al.cudaData,slow.data(),al.GetBytes(),cudaMemcpyHostToDevice));
 std::vector<std::unique_ptr<State>>own;std::vector<State*>all;for(int i=0;i<count;++i){own.emplace_back(new State(vh,kh,11+i));all.push_back(own.back().get());}
 // First eager call both tunes and executes. Comparing this very first update
 // detects any live-state mutation by validation or repeated timing trials.
 Launch(true,all,al,bias,norm,kh,vh,tiled);Launch(false,all,al,bias,norm,kh,vh,tiled);
 double outL2=0,stateL2=0;
 auto compare=[&]{Cuda(cudaDeviceSynchronize());for(auto*p:all){outL2=std::max(outL2,L2(p->o,p->refO));stateL2=std::max(stateL2,L2(p->s,p->refS));}};
 compare();
 auto report=kernels::Report();Check(report.find(candidate.empty()?"selected":candidate)!=std::string::npos,"missing plan");
 cudaGraph_t graph;cudaGraphExec_t exec;Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));Launch(false,all,al,bias,norm,kh,vh,tiled);Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
 size_t nodes=0;Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));Check(nodes==size_t((count+31)/32),"graph launch count");Cuda(cudaGraphInstantiate(&exec,graph,0,0,0));
 for(int step=0;step<(count==1?128:3);++step){for(int i=0;i<count;++i){Fill(all[i]->q,step*73+i);Fill(all[i]->a,step*79+i,2);Fill(all[i]->b,step*83+i,2);Fill(all[i]->g,step*89+i,2);}
  Launch(true,all,al,bias,norm,kh,vh,tiled);Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));compare();}
 printf("PASS requests=%d heads=%d tiled=%d candidate=%s output_l2=%.9g state_l2=%.9g nodes=%zu\n",count,vh,int(tiled),candidate.c_str(),outL2,stateL2,nodes);fflush(stdout);Check(outL2<.0032&&stateL2<.0032,"L2");
 Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
}
int main(int,char**){try{FastllmCudaSetDevice(0);
 for(const char* candidate:{"cuda.gdn.register.rg1","cuda.gdn.register.rg2","cuda.gdn.register.rg4"}){
  for(int count:{1,2,4,7,33})Case(count,count<=4?48:6,false,candidate);
  Case(1,48,true,candidate);
 }
 Case(3,12,false,"");
 return 0;}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
