#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/cuda/fastllm-cuda-decode-linear.h"
#include "devices/cuda/low_latency/dense.h"
#include <cuda_bf16.h>
#ifdef FASTLLM_ENABLE_NAIVE_DENSE_SM90
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace fastllm;
namespace ll=fastllm::cuda::low_latency;
void Check(bool x,const char *message){if(!x)throw std::runtime_error(message);}
void C(cudaError_t e){Check(e==cudaSuccess,cudaGetErrorString(e));}
void Init(Data &d,DataType type,std::vector<int> shape){d.dataType=type;d.UpdateUnitSize();d.Resize(shape);d.ToDevice(CUDA,{0},false);d.Allocate(false);}
void Fill(Data &d,int seed){std::vector<__nv_bfloat16> h(d.Count(0));for(auto &x:h){seed=(seed*31+17)%65521;x=__nv_bfloat16((seed%127-63)*.003f);}C(cudaMemcpy(d.cudaData,h.data(),d.GetBytes(),cudaMemcpyHostToDevice));}
std::vector<unsigned short> Read(const Data &d){std::vector<unsigned short> h(d.GetBytes()/2);C(cudaMemcpy(h.data(),d.cudaData,d.GetBytes(),cudaMemcpyDeviceToHost));return h;}
#ifdef FASTLLM_ENABLE_NAIVE_DENSE_SM90
// Binary fractions keep the reference dot products exact for any cuBLAS split.
// Compare every BF16 result including both nonlinear rounding boundaries.
void FillBinary(Data &d, unsigned seed) {
    std::vector<__nv_bfloat16> h(d.Count(0));
    for (auto &x : h) {
        seed = seed * 1664525u + 1013904223u;
        x = __nv_bfloat16((int(seed >> 27) - 16) / 32.f);
    }
    C(cudaMemcpy(d.cudaData, h.data(), d.GetBytes(), cudaMemcpyHostToDevice));
}

void CheckHopperVerifyGateUp(int rows) {
    constexpr int hidden = 4096, width = 2048;
    Data x, gate, up, fused, reference, upResult, bias;
    Init(x, BFLOAT16, {1, rows, hidden});
    Init(gate, BFLOAT16, {width, hidden});
    Init(up, BFLOAT16, gate.dims);
    Init(reference, BFLOAT16, {1, rows, width});
    Init(upResult, BFLOAT16, reference.dims);
    FillBinary(x, 53);
    FillBinary(gate, 11);
    FillBinary(up, 37);
    auto run = [&] { return FastllmCudaNaiveVerifyDenseGateUp(x, gate, up, fused); };
    Check(run(), "SM90 verification gate/up rejected its warmup shape");
    Check(!FastllmCudaNaiveVerifyDenseGateUp(x, gate, up, x), "input/output alias accepted");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    C(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Check(run(), "SM90 verification gate/up rejected capture");
    C(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    size_t nodes = 0;
    C(cudaGraphGetNodes(graph, nullptr, &nodes));
    Check(nodes == 1, "gate/up projection and activation must capture as one kernel");
    C(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (unsigned seed : {53u, 54u, 55u}) {
        FillBinary(x, seed);
        C(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(FastllmCudaBFloat16MatMulBFloat16(x, gate, bias, reference, rows, hidden, width) &&
              FastllmCudaBFloat16MatMulBFloat16(x, up, bias, upResult, rows, hidden, width) &&
              FastllmCudaSilu(reference, reference) && FastllmCudaMulTo(reference, upResult, 1.f),
              "reference gate/up failed");
        C(cudaStreamSynchronize(cudaStreamPerThread));
        Check(Read(fused) == Read(reference), "SM90 gate/up changed BF16 output");
    }
    C(cudaGraphExecDestroy(exec));
    C(cudaGraphDestroy(graph));
}
#endif
int main(){try{
 int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess||!n)return 77;
 C(cudaSetDevice(0));cudaDeviceProp properties{};C(cudaGetDeviceProperties(&properties,0));
 if(properties.major<7)return 77;
#ifdef FASTLLM_ENABLE_NAIVE_DENSE_SM90
 if(properties.major==9 && properties.minor==0) for (int rows : {9, 15, 16, 17, 31, 33, 65, 127, 128}) CheckHopperVerifyGateUp(rows);
#endif
 ll::DensePlan plan;
 const auto admission=ll::PrepareDense({1,1024,256},plan,cudaStreamPerThread);
 const bool mma=admission==cudaSuccess;
 if(!mma) {
  Check(ll::BackendUnavailable(admission),"unexpected error preparing test Dense shape");
  const auto pending=cudaGetLastError();
  Check(pending==cudaSuccess || ll::BackendUnavailable(pending),"unexpected pending CUDA error");
 }
 Data gate,up,down;Init(gate,BFLOAT16,{256,1024});Init(up,BFLOAT16,gate.dims);Init(down,BFLOAT16,{1024,256});
 Fill(gate,13);Fill(up,23);Fill(down,33);
 for(int rows:{1,2,3,8}){
  Data x,mid,out;Init(x,BFLOAT16,{1,rows,1024});Fill(x,17);
  if(!mma){Check(!FastllmCudaDecodeDense(x,gate,up,down,mid,out),"unsupported MMA did not fall back");Check(!out.cudaData&&!mid.cudaData,"fallback allocated outputs");continue;}
  Check(FastllmCudaDecodeDense(x,gate,up,down,mid,out),"portable Dense not admitted");C(cudaStreamSynchronize(cudaStreamPerThread));
  auto expected=Read(out);void *address=out.cudaData;
  cudaGraph_t graph;cudaGraphExec_t exec;
  C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
  Check(FastllmCudaDecodeDense(x,gate,up,down,mid,out),"warm Dense rejected inside capture");
  C(cudaStreamEndCapture(cudaStreamPerThread,&graph));size_t nodes=0;C(cudaGraphGetNodes(graph,nullptr,&nodes));
  Check(nodes==2,"Data adapter did not capture two fused kernels");C(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
  for(int replay=0;replay<4;++replay)C(cudaGraphLaunch(exec,cudaStreamPerThread));
  C(cudaStreamSynchronize(cudaStreamPerThread));Check(out.cudaData==address&&Read(out)==expected,"graph replay changed result or allocation");
  x.dataType=FLOAT16;Check(!FastllmCudaDecodeDense(x,gate,up,down,mid,out),"wrong precision accepted");x.dataType=BFLOAT16;
  Check(!FastllmCudaDecodeDense(x,gate,up,down,mid,x),"output alias accepted");
  auto shape=gate.dims;gate.dims[0]=255;Check(!FastllmCudaDecodeDense(x,gate,up,down,mid,out),"wrong weight shape accepted");gate.dims=shape;
  C(cudaStreamSynchronize(cudaStreamPerThread));Check(out.cudaData==address&&Read(out)==expected,"rejection changed output");
  // Request contents may change without another prepare/capture. Compare the
  // same graph against a fresh eager call using the existing buffer addresses.
  Fill(x,29);C(cudaGraphLaunch(exec,cudaStreamPerThread));C(cudaStreamSynchronize(cudaStreamPerThread));auto changed=Read(out);
  Check(FastllmCudaDecodeDense(x,gate,up,down,mid,out),"rebound eager failed");C(cudaStreamSynchronize(cudaStreamPerThread));
  Check(Read(out)==changed && changed!=expected,"request update/replay mismatch");
  C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(graph));
 }
 // Cold external captures must reject without allocating/preparing and without
 // caching that transient rejection. A later warmup must still admit the shape.
 {
  Data x,mid,out;Init(x,BFLOAT16,{1,4,1024});Fill(x,41);
  cudaGraph_t graph;C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
  Check(!FastllmCudaDecodeDense(x,gate,up,down,mid,out),"cold captured Dense prepared on demand");
  C(cudaStreamEndCapture(cudaStreamPerThread,&graph));C(cudaGraphDestroy(graph));
  Check(!mid.cudaData&&!out.cudaData,"cold captured Dense changed outputs");
  Check(FastllmCudaDecodeDense(x,gate,up,down,mid,out)==mma,"capture rejection poisoned Dense cache");
  C(cudaStreamSynchronize(cudaStreamPerThread));
 }
 {
  Data x,g,w,out;Init(x,BFLOAT16,{1,3,512});Init(g,FLOAT32,{512});Init(w,BFLOAT16,{32,512});
  cudaGraph_t graph;C(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
  Check(!FastllmCudaDecodeNormLinearCanRun(x,g,w,out),"cold captured projection prepared on demand");
  C(cudaStreamEndCapture(cudaStreamPerThread,&graph));C(cudaGraphDestroy(graph));
  Check(!out.cudaData,"cold captured projection changed output");
  Check(FastllmCudaDecodeNormLinearCanRun(x,g,w,out),"capture rejection poisoned projection cache");
 }
 // Capability is independent from performance preference and architecture ID.
 Data x,g,w,out;Init(x,BFLOAT16,{1,9,256});Init(g,FLOAT32,{256});Init(w,BFLOAT16,{32,256});
 Check(FastllmCudaDecodeNormLinearCanRun(x,g,w,out),"multi-tile capability was hidden by preference");
 Check(!FastllmCudaDecodeNormLinearPreferred(x,g,w,out),"GEMM policy changed for >=8 rows");
 puts("PASS portable decode adapters and graph fallback");return 0;
}catch(const std::exception &e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
