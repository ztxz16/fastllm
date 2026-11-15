#include "utils/kernel_registry.h"
#include "devices/cuda/fastllm-kernel-benchmark.cuh"
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cstring>
using namespace fastllm;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
static std::mt19937 rng(481);
static double maxL2=0;
static int cases=0;
static void allocate(Data &d, DataType type, std::vector<int> dims, const void *data) {
    d.dataType=type;d.UpdateUnitSize();d.Resize(dims);d.dataDevice=CUDA;d.dataDeviceIds={0};d.Allocate(false);
    if(data)CHECK(cudaMemcpy(d.cudaData,data,d.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
}
static void run(int rows,int n,int k,int groups,bool fp8,bool graph) {
    std::vector<__nv_bfloat16> x(rows*groups*k),w(groups*n*k),got(rows*groups*n);
    std::vector<uint8_t> q(w.size());
    std::vector<float> scales(groups*n/32*(k/32));
    for(auto &v:x)v=__float2bfloat16_rn(float(int(rng()%1001)-500)/257.f);
    for(auto &v:scales)v=std::ldexp(1.03125f,int(rng()%7)-8);
    for(int r=0;r<groups*n;r++)for(int c=0;c<k;c++) {
        size_t i=size_t(r)*k+c;
        q[i]=uint8_t((rng()%127)|((rng()&1)<<7));
        __nv_fp8_e4m3 f;f.__x=q[i];
        w[i]=__float2bfloat16_rn(fp8?float(f)*scales[(r/32)*(k/32)+c/32]:float(int(rng()%997)-498)/251.f);
    }
    Data input,weight,out;
    allocate(input,BFLOAT16,{rows,1,groups,k},x.data());
    allocate(weight,fp8?FP8_E4M3:BFLOAT16,{groups*n,k},fp8?(void*)q.data():(void*)w.data());
    if(fp8){weight.blockM=weight.blockK=32;weight.scales=scales;Data bias;FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight,bias,groups*n);}
    allocate(out,BFLOAT16,{rows,groups,n},nullptr);
    auto dispatch=[&]() {
        if(groups>1)return FastllmCudaDeepSeekV41CompactWoA(input,weight,groups,n,out);
        Data bias;return FastllmCudaBFloat16MatMulFP8E4M3(input,weight,bias,out,rows,k,n);
    };
    CHECK(dispatch());
    CHECK(cudaDeviceSynchronize()==cudaSuccess);
    if(graph){
        cudaGraph_t g;cudaGraphExec_t e;
        CHECK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
        CHECK(dispatch());
        CHECK(cudaStreamEndCapture(cudaStreamPerThread,&g)==cudaSuccess);
        CHECK(cudaGraphInstantiate(&e,g,0)==cudaSuccess);
        // Replay reads fresh operands, rather than a capture-time result.
        for(auto &v:x)v=__float2bfloat16_rn(float(v)*0.75f+0.125f);
        CHECK(cudaMemcpy(input.cudaData,x.data(),input.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess);
        CHECK(cudaGraphLaunch(e,cudaStreamPerThread)==cudaSuccess);
        CHECK(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess);
        cudaGraphExecDestroy(e);cudaGraphDestroy(g);
    }
    CHECK(cudaMemcpy(got.data(),out.cudaData,out.GetBytes(),cudaMemcpyDeviceToHost)==cudaSuccess);
    double err=0,norm=0;
    for(int t=0;t<rows;t++)for(int g=0;g<groups;g++)for(int j=0;j<n;j++){
        double ref=0;for(int h=0;h<k;h++)ref+=double(float(x[(t*groups+g)*k+h]))*float(w[(g*n+j)*k+h]);
        float actual=float(got[(t*groups+g)*n+j]);CHECK(std::isfinite(actual));
        err+=(actual-ref)*(actual-ref);norm+=ref*ref;
    }
    double l2=std::sqrt(err/std::max(norm,1e-20));maxL2=std::max(maxL2,l2);CHECK(l2<0.004);++cases;

}
__global__ void FillPartial(float *values) {if(threadIdx.x<4)values[threadIdx.x]=2;}
__global__ void FillMixedOutputs(float *projection, half *gate, __nv_bfloat16 *other, int implementation) {
    if (threadIdx.x < 4) {
        projection[threadIdx.x] = implementation == 4 ? NAN : 4096;
        gate[threadIdx.x] = __float2half_rn(implementation == 1 ? 2.f : 1.f);
        other[threadIdx.x] = __float2bfloat16_rn(1.f);
    }
}
static void checkMixedOutputsPreserved() {
    using namespace fastllm::kernels;
    Data projection, gate, other;
    std::vector<float> originalProjection(16, 17), actualProjection(16);
    std::vector<half> originalGate(16, __float2half_rn(19.f)), actualGate(16);
    std::vector<__nv_bfloat16> originalOther(16, __float2bfloat16_rn(23.f)), actualOther(16);
    allocate(projection, FLOAT32, {16}, originalProjection.data());
    allocate(gate, FLOAT16, {16}, originalGate.data());
    allocate(other, BFLOAT16, {16}, originalOther.data());
    struct Outputs { float *projection; half *gate; __nv_bfloat16 *other; } outputs{
        (float *)projection.cudaData, (half *)gate.cudaData, (__nv_bfloat16 *)other.cudaData};
    auto run = [](void *p, int implementation) {
        auto &c = *static_cast<Outputs *>(p);
        FillMixedOutputs<<<1,32,0,cudaStreamPerThread>>>(c.projection, c.gate, c.other, implementation);
        CudaCheck(cudaGetLastError());
        if (implementation == 3) throw std::runtime_error("injected fusion failure");
        return implementation != 2;
    };
    CudaOutputsCase c{{&outputs, run, nullptr, nullptr, 0, false}, {
        {projection.cudaData, 16, FLOAT32}, {gate.cudaData, 16, FLOAT16}, {other.cudaData, 16, BFLOAT16}}};
    for (int implementation : {0, 1, 2, 3, 4}) {
        double l2 = -1;
        std::string reason;
        bool accepted = false, caught = false;
        try { accepted = CudaOutputsValidate(&c, implementation, l2, reason); }
        catch (const std::runtime_error &e) {
            caught = true;
            CHECK(std::string(e.what()) == "injected fusion failure");
        }
        CHECK(caught == (implementation == 3));
        CHECK(accepted == (implementation <= 1));
        // A small gate's error must not be hidden by the large projection.
        if (accepted) CHECK(l2 == double(implementation));
        CHECK(cudaMemcpy(actualProjection.data(), projection.cudaData, projection.GetBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(actualGate.data(), gate.cudaData, gate.GetBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(cudaMemcpy(actualOther.data(), other.cudaData, other.GetBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(actualProjection == originalProjection);
        CHECK(std::memcmp(actualGate.data(), originalGate.data(), gate.GetBytes()) == 0);
        CHECK(std::memcmp(actualOther.data(), originalOther.data(), other.GetBytes()) == 0);
    }
    std::puts("MIXED_OUTPUT_PRESERVED_PASS per-output L2, inactive entries, rejection, exception and non-finite output");
}
static void checkRawColdCapture() {
    using namespace fastllm::kernels;
    static const Family family{95,"test.raw_capture","1",{
        {"test.capture.reference",json11::Json::object{}},
        {"test.capture.candidate",json11::Json::object{}}}};
    Register(&family);
    CHECK(fastllm_kernel_configure("{\"mode\":\"quick\",\"repeats\":1}"));
    float *output=nullptr;CHECK(cudaMalloc(&output,4*sizeof(float))==cudaSuccess);
    auto run=[](void *p,int) {FillPartial<<<1,32,0,cudaStreamPerThread>>>((float*)p);return cudaGetLastError()==cudaSuccess;};
    auto support=[](void *,int) {return std::string();};
    CudaCase c{output,run,support,output,4,false};
    Signature signature{{95,0,0,1,4}};
    // A composite capture can own the scratch pool between raw stream
    // captures. A benchmark must not start a nested graph in that interval.
    CHECK(FastllmCudaGraphMemoryPoolBegin());
    CHECK(SelectCuda(family,signature,c)==0);
    CHECK(NeedsPreparation(signature));
    CHECK(FastllmCudaGraphMemoryPoolActive());
    FastllmCudaGraphMemoryPoolAbort();
    CHECK(!FastllmCudaGraphMemoryPoolActive());
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    CHECK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
    CHECK(SelectCuda(family,signature,c)==0);
    CHECK(LaunchCuda(family,c,0));
    CHECK(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess);
    CHECK(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess);
    CHECK(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess);
    CHECK(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess);
    CHECK(NeedsPreparation(signature));
    CHECK(SelectCuda(family,signature,c)>=0&&!NeedsPreparation(signature));
    float actual[4];CHECK(cudaMemcpy(actual,output,sizeof(actual),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(float value:actual)CHECK(value==2);
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);cudaFree(output);
    CHECK(fastllm_kernel_configure("{\"mode\":\"off\"}"));
    std::puts("RAW_COLD_CAPTURE_PASS deferred measurement and later eager preparation");
}
static void checkPartialOutputPreserved() {
    float *output=nullptr;CHECK(cudaMalloc(&output,16*sizeof(float))==cudaSuccess);
    std::vector<float> original(16),actual(16);for(int i=0;i<16;++i)original[i]=float(i+10);
    CHECK(cudaMemcpy(output,original.data(),64,cudaMemcpyHostToDevice)==cudaSuccess);
    auto run=[](void *p,int) {FillPartial<<<1,32,0,cudaStreamPerThread>>>((float*)p);return cudaGetLastError()==cudaSuccess;};
    fastllm::kernels::CudaCase c{output,run,nullptr,output,16,false};double l2=0;std::string reason;
    CHECK(fastllm::kernels::CudaValidate(&c,1,l2,reason)&&l2==0);
    CHECK(cudaMemcpy(actual.data(),output,64,cudaMemcpyDeviceToHost)==cudaSuccess);
    CHECK(actual==original);
    for(bool throws : {false,true}) {
        struct Failure { float *output; bool throws; } failure{output,throws};
        c.context=&failure;
        c.run=[](void *p,int implementation) {
            const auto &f=*static_cast<Failure *>(p);
            FillPartial<<<1,32,0,cudaStreamPerThread>>>(f.output);
            fastllm::kernels::CudaCheck(cudaGetLastError());
            if(implementation==0)return true;
            if(f.throws)throw std::runtime_error("injected candidate failure");
            return false;
        };
        bool caught=false;
        try {CHECK(!fastllm::kernels::CudaValidate(&c,1,l2,reason));}
        catch(const std::runtime_error &e) {caught=true;CHECK(std::string(e.what())=="injected candidate failure");}
        CHECK(caught==throws);
        CHECK(cudaMemcpy(actual.data(),output,64,cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(actual==original);
        caught=false;
        try {CHECK(fastllm::kernels::CudaMeasure(&c,1,2)<0);}
        catch(const std::runtime_error &e) {caught=true;CHECK(std::string(e.what())=="injected candidate failure");}
        CHECK(caught==throws);
        cudaStreamCaptureStatus status;
        CHECK(cudaStreamIsCapturing(cudaStreamPerThread,&status)==cudaSuccess);
        CHECK(status==cudaStreamCaptureStatusNone);
        CHECK(cudaMemcpy(actual.data(),output,64,cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(actual==original); // Rejected captures must never be replayed.
    }
    cudaFree(output);
    std::printf("PARTIAL_OUTPUT_PRESERVED_PASS success, rejection, exception and capture cleanup\n");
}
static void checkInvalidCaptureCleanup() {
    using namespace fastllm::kernels;
    CudaCase c{nullptr, [](void *, int) {
        // Model an optional implementation that accidentally synchronizes
        // during its timing capture. Its failure must not poison fallback.
        return cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }, nullptr, nullptr, 0, false};
    bool rejected = false;
    try { CudaMeasure(&c, 1, 2); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    CHECK(cudaGetLastError() == cudaSuccess);
    cudaStreamCaptureStatus status;
    CHECK(cudaStreamIsCapturing(cudaStreamPerThread, &status) == cudaSuccess);
    CHECK(status == cudaStreamCaptureStatusNone);
    CHECK(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess);
    std::puts("INVALID_CAPTURE_CLEANUP_PASS no sticky capture error");
}
int main(int argc, char **argv){
    const bool wideOnly = argc == 2 && std::strcmp(argv[1], "--wide") == 0;
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    cudaSetDevice(0);FastllmCudaSetDevice(0);
    checkRawColdCapture();
    checkPartialOutputPreserved();
    checkMixedOutputsPreserved();
    checkInvalidCaptureCleanup();
    if(argc==2&&std::strcmp(argv[1],"--cleanup")==0)return 0;
    if(FastllmCudaRuntimeArch()!=120&&FastllmCudaRuntimeArch()!=121)return 77;
    unsetenv("FASTLLM_CUDA_TRITON");
    CHECK(fastllm_kernel_configure("{\"mode\":\"quick\",\"budget_ms\":120000}"));
    if (!wideOnly) for(bool fp8:{true})for(auto shape:std::vector<std::vector<int>>{{32,32},{96,160},{512,1280}})
        for(int rows:{1,2,3,5,6,8,9,12,16,17,32})for(int groups:{1,2,8})run(rows,shape[0],shape[1],groups,fp8,true);
    // Wide FP8 tiles include a partial N tile and a masked K tail.
    for (int rows : {3, 6, 16}) run(rows, 16416, 96, 1, true, true);
    run(6, 32768, 1280, 1, true, true);
    fastllm_kernel_freeze();
    std::printf("PROFILE %s\n",fastllm_kernel_report());
    std::printf("KERNEL_POLICY_GPU_PASS cases=%d max_l2=%.10g\n",cases,maxL2);
}
