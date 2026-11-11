#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/kernel_registry.h"
#include "fastllm.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
using namespace fastllm;
namespace fastllm {void DoCudaAttention(Data &,Data &,Data &,Data &,Data &,int,float,int);}
static void Check(bool v,const char *m){if(!v)throw std::runtime_error(m);}
static void C(cudaError_t e){Check(e==cudaSuccess,cudaGetErrorString(e));}
static void GPU(Data &d){d.dataDevice=CUDA;d.dataDeviceIds={0};d.Allocate(false);}
static void Fill(Data &d,int seed,float scale=1){
 std::mt19937 rng(seed);std::uniform_real_distribution<float> gen(-scale,scale);
 if(d.dataType==FLOAT32){std::vector<float> x(d.Count(0));for(auto &v:x)v=gen(rng);C(cudaMemcpy(d.cudaData,x.data(),x.size()*4,cudaMemcpyHostToDevice));}
 else{std::vector<half>x(d.expansionBytes/2);for(auto &v:x)v=__float2half_rn(gen(rng));C(cudaMemcpy(d.cudaData,x.data(),x.size()*2,cudaMemcpyHostToDevice));}
}
static std::vector<float> Host(Data &d){
 std::vector<float>x(d.Count(0));
 if(d.dataType==FLOAT32)C(cudaMemcpy(x.data(),d.cudaData,x.size()*4,cudaMemcpyDeviceToHost));
 else{std::vector<half>h(x.size());C(cudaMemcpy(h.data(),d.cudaData,h.size()*2,cudaMemcpyDeviceToHost));for(size_t i=0;i<x.size();++i)x[i]=__half2float(h[i]);}
 return x;
}
static double Equal(const std::vector<float>&a,Data&d,const char *label,double limit=.0032){
 auto b=Host(d);Check(a.size()==b.size(),"output shape");double square=0,error=0;float maximum=0;
 for(size_t i=0;i<a.size();++i){Check(std::isfinite(b[i]),"nonfinite");square+=double(a[i])*a[i];error+=double(a[i]-b[i])*(a[i]-b[i]);maximum=std::max(maximum,std::abs(a[i]-b[i]));}
 double l2=std::sqrt(error/std::max(square,1e-30));printf("NUMERIC %s l2=%.9g max=%.9g\n",label,l2,maximum);Check(l2<=limit,label);return l2;
}
static void Policy(const char*family=nullptr,const char*impl=nullptr){
 std::string s=family?std::string("{\"mode\":\"quick\",\"budget_ms\":60000,\"repeats\":4,\"samples\":3,\"force\":{\"")+family+"\":\""+impl+"\"}}":"{\"mode\":\"off\"}";
 Check(fastllm_kernel_configure(s.c_str())==1,fastllm_kernel_error());
}
static void Projection(int rows,int inner,const std::vector<int>&widths){
 Data x(FLOAT16,{1,rows,inner});GPU(x);Fill(x,41,.7f);
 std::vector<Data>w(widths.size()),out(widths.size());std::vector<const Data*>wp;std::vector<Data*>op;
 for(size_t i=0;i<widths.size();++i){w[i]=Data(FLOAT16,{widths[i],inner});out[i]=Data(FLOAT16,{1,rows,widths[i]});GPU(w[i]);GPU(out[i]);Fill(w[i],i+13,1/std::sqrt(float(inner)));wp.push_back(&w[i]);op.push_back(&out[i]);}
 Policy();Check(FastllmCudaHalfMultiLinear(x,wp.data(),op.data(),wp.size()),"projection reference");
 std::vector<std::vector<float>>ref;for(auto &o:out)ref.push_back(Host(o));
 for(const char *id:{"cuda.linear.multi.fp16.cublas","cuda.linear.multi.fp16.tile4","cuda.linear.multi.fp16.tile8"}){
  Policy("linear.multi.fp16",id);Check(FastllmCudaHalfMultiLinear(x,wp.data(),op.data(),wp.size()),"projection candidate");
  printf("KERNEL_REPORT %s\n",fastllm_kernel_report());
  for(size_t i=0;i<out.size();++i)Equal(ref[i],out[i],id);
  Check(FastllmCudaGraphPrepareCaptureDevice(),"projection graph prepare");Check(FastllmCudaGraphBeginCapture(),"projection capture");
  Check(FastllmCudaHalfMultiLinear(x,wp.data(),op.data(),wp.size()),"projection capture dispatch");void *graph=nullptr,*exec=nullptr;
  Check(FastllmCudaGraphEndCapture(&graph)&&FastllmCudaGraphInstantiate(graph,&exec),"projection graph instantiate");
  for(int seed:{73,97}){Fill(x,seed,.6f);Check(FastllmCudaGraphLaunch(exec),"projection replay");std::vector<std::vector<float>>actual;for(auto&o:out)actual.push_back(Host(o));
   {kernels::Bypass b;Check(FastllmCudaHalfMultiLinear(x,wp.data(),op.data(),wp.size()),"projection eager reference");}
   for(size_t i=0;i<out.size();++i)Equal(actual[i],out[i],"projection dynamic input");}
  FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);Fill(x,41,.7f);
 }
 printf("PROJECTION rows=%d inner=%d count=%zu PASS\n",rows,inner,widths.size());
}
static void Hyper(int rows,int inner=10240,int rank=320){
 Data x(FLOAT32,{1,rows,inner}),down(FLOAT16,{rank,inner}),inject(FLOAT16,{4,inner}),a(FLOAT32,{1,rows,rank}),b(FLOAT32,{1,rows,4});
 for(auto*v:{&x,&down,&inject,&a,&b})GPU(*v);Fill(x,47,.7f);Fill(down,53,1/std::sqrt(float(inner)));Fill(inject,59,1/std::sqrt(float(inner)));
 Policy();Check(FastllmCudaQwen4HyperProject(x,down,inject,a,b,4),"HC reference");auto ar=Host(a),br=Host(b);
 for(const char *id:{"cuda.hyper_project.tile4","cuda.hyper_project.tile8","cuda.hyper_project.gemm_epilogue"}){
  Policy("linear.hyper_project.fp16",id);Check(FastllmCudaQwen4HyperProject(x,down,inject,a,b,4),"HC candidate");printf("KERNEL_REPORT %s\n",fastllm_kernel_report());Equal(ar,a,id);Equal(br,b,id);
  Check(FastllmCudaGraphPrepareCaptureDevice()&&FastllmCudaGraphBeginCapture(),"HC capture");
  Check(FastllmCudaQwen4HyperProject(x,down,inject,a,b,4),"HC capture dispatch");void *graph=nullptr,*exec=nullptr;
  Check(FastllmCudaGraphEndCapture(&graph)&&FastllmCudaGraphInstantiate(graph,&exec),"HC graph instantiate");
  Fill(x,83,.4f);Check(FastllmCudaGraphLaunch(exec),"HC replay");auto aa=Host(a),bb=Host(b);
  {kernels::Bypass bypass;Check(FastllmCudaQwen4HyperProject(x,down,inject,a,b,4),"HC eager ref");}
  Equal(aa,a,"HC dynamic input");Equal(bb,b,"HC dynamic gate");FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);Fill(x,47,.7f);
 }
 printf("HYPER rows=%d inner=%d rank=%d PASS\n",rows,inner,rank);
}
static void Sparse(int rows,int width=2051,int dim=256,int heads=24,int keyHeads=2,const char *force=nullptr,float qscale=.3f){
 Check(fastllm_kernel_configure("{\"mode\":\"quick\",\"repeats\":4,\"samples\":3}")==1,"sparse policy");if(force)Policy("attention.indexed.fp16",force);const int capacity=4096+rows,groups=heads/keyHeads;
 Data q(FLOAT16,{heads,rows,dim}),k(FLOAT16,{keyHeads,capacity,dim}),v(FLOAT16,k.dims),ids(INT32,{rows,width}),meta(INT32,{1,2}),actual(FLOAT16,q.dims),ref(FLOAT16,q.dims);
 for(auto*p:{&q,&k,&v,&ids,&meta,&actual,&ref})GPU(*p);Fill(q,61,qscale);Fill(k,67,qscale);Fill(v,71,.7f);
 std::vector<int>ix(size_t(rows)*width);
 for(int row=0;row<rows;++row)for(int i=0;i<width;++i)ix[size_t(row)*width+i]=i%13==0?-1:(i%17==0?capacity:(i*7+row*31)%capacity);
 C(cudaMemcpy(ids.cudaData,ix.data(),ix.size()*4,cudaMemcpyHostToDevice));
 Data pq(FLOAT16,{rows*heads,1,dim}),ck(FLOAT16,{rows*keyHeads,width,dim}),cv(FLOAT16,ck.dims),mask(FLOAT16,{rows,1,width}),packed(FLOAT16,pq.dims);
 for(auto*p:{&pq,&ck,&cv,&mask,&packed})GPU(*p);
 const float scale=1/std::sqrt(float(dim));
 auto reference=[&]{Check(FastllmCudaQwen4PrepareSparseBatchGraph(q,k,v,ids,(int32_t*)meta.cudaData,rows,pq,ck,cv,mask,0,rows),"sparse gather reference");DoCudaAttention(pq,ck,cv,mask,packed,groups,scale,1);Check(FastllmCudaQwen4UnpackSparseBatch(packed,ref,0,rows),"sparse unpack reference");};
 int m[2]={4096,rows};C(cudaMemcpy(meta.cudaData,m,sizeof(m),cudaMemcpyHostToDevice));reference();
 Check(FastllmCudaQwen4SparseBatchGraph(q,k,v,ids,actual,groups,scale,(int32_t*)meta.cudaData),"sparse candidate");printf("KERNEL_REPORT %s\n",fastllm_kernel_report());Equal(Host(ref),actual,"sparse selected");
 Check(FastllmCudaGraphPrepareCaptureDevice()&&FastllmCudaGraphBeginCapture(),"sparse capture");
 Check(FastllmCudaQwen4SparseBatchGraph(q,k,v,ids,actual,groups,scale,(int32_t*)meta.cudaData),"sparse capture dispatch");void *graph=nullptr,*exec=nullptr;
 Check(FastllmCudaGraphEndCapture(&graph)&&FastllmCudaGraphInstantiate(graph,&exec),"sparse graph instantiate");
 for(int previous:{4096,1031,1,4077}){m[0]=previous;C(cudaMemcpy(meta.cudaData,m,sizeof(m),cudaMemcpyHostToDevice));Check(FastllmCudaGraphLaunch(exec),"sparse replay");reference();Equal(Host(ref),actual,"sparse dynamic context");}
 FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);
 printf("SPARSE rows=%d width=%d dim=%d heads=%d groups=%d PASS\n",rows,width,dim,heads,groups);
}
extern "C" int main(int argc,char **argv){try{
 int n=0;C(cudaGetDeviceCount(&n));Check(n>0,"no CUDA device");C(cudaSetDevice(0));
 if(argc>1 && std::string(argv[1])=="--memcheck") {
  Hyper(9,128,24);Sparse(4,2051,256,24,2,"cuda.attention.indexed.cublas_qk",3.0f);
  Policy();puts("PASS focused fusion graph memory check");return 0;
 }
 for(int rows:{1,3,8,9,16,33,129}){Projection(rows,64,{35,17,9});Hyper(rows,128,24);}
 for(int rows:{2,8,16}){Projection(rows,2560,{12288,512,512});Projection(rows,2560,{10240,6144,48,48});Hyper(rows);}
 for(int rows:{1,3,4,9,17,65})Sparse(rows,129,64,6,2);
 Sparse(4);Sparse(7,2051,256,34,2);
 Sparse(4,2051,256,24,2,"cuda.attention.indexed.cublas_qk",3.0f);
 Policy();puts("PASS projection policy, HC arbitrary-row fusion, indexed sparse dynamic graphs");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
