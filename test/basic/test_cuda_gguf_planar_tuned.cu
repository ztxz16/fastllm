#include "cuda_gguf_fusion_test.cuh"
#include "fastllm-cuda-gguf-planar.h"
#include "utils/kernel_registry.h"

static void Case(ggml_type type,int rows,int k,int n,const char *candidate,bool customStream) {
    const int stride=n+13;
    std::string error;
    Check(kernels::Configure(json11::Json(json11::Json::object{{"mode","quick"},
        {"families",json11::Json::array{"gguf.planar.projection"}},{"budget_ms",30000},
        {"force",json11::Json::object{{"gguf.planar.projection",candidate}}}}).dump(),error),error.c_str());
    Projection weight(type,rows,k,n);
    Data x(FLOAT16,{rows,k}),y(FLOAT16,{rows,stride}),ref(FLOAT16,y.dims);
    Allocate(x);Allocate(y);Allocate(ref);
    void *q=nullptr; Cuda(cudaMalloc(&q,FastllmGgufPlanarBytes(rows,k)));
    cudaStream_t stream=cudaStreamPerThread;
    if(customStream)Cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch=[&](Data &out) { Check(FastllmGgufProjectPlanar(type,0,weight.w.cudaData,
        nullptr,q,out.cudaData,rows,k,n,stride,stream),"projection rejected"); };
    auto prepare=[&](int seed) {
        Upload(x,Input(rows,k,seed));Cuda(cudaDeviceSynchronize());
        Cuda(cudaMemsetAsync(y.cudaData,0x3c,y.GetBytes(),stream));
        Cuda(cudaMemsetAsync(ref.cudaData,0x3c,ref.GetBytes(),stream));
        Check(FastllmGgufQuantizePlanar(x.cudaData,1,q,rows,k,stream),"quantization rejected");
        kernels::Bypass bypass;launch(ref);
    };
    auto compare=[&] { Cuda(cudaStreamSynchronize(stream)); Same(y,Download(ref,ref.Count(0)),"strided tuned projection"); };
    prepare(7);launch(y);compare();
    Check(kernels::Report().find(std::string("\"selected\": \"")+candidate+"\"")!=std::string::npos,"forced candidate missing");
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch(y);
    Cuda(cudaStreamEndCapture(stream,&graph));size_t nodes=0;Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));
    Check(nodes==size_t((rows+7)/8),"unexpected graph node count");
    Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int seed:{23,71}) { prepare(seed);Cuda(cudaGraphLaunch(exec,stream));compare(); }
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));Cuda(cudaFree(q));
    if(customStream)Cuda(cudaStreamDestroy(stream));
    printf("PASS type=%d rows=%d k=%d n=%d candidate=%s stream=%d bitwise\n",int(type),rows,k,n,candidate,int(customStream));
}
static void LargeQuantize() {
    constexpr int rows=65537,k=256;
    Data x(FLOAT16,{rows,k});Allocate(x);Upload(x,Input(rows,k,9));
    void *all=nullptr,*one=nullptr;
    Cuda(cudaMalloc(&all,FastllmGgufPlanarBytes(rows,k)));Cuda(cudaMalloc(&one,FastllmGgufPlanarBytes(1,k)));
    Check(FastllmGgufQuantizePlanar(x.cudaData,1,all,rows,k,cudaStreamPerThread),"large quantizer");
    for(int row:{0,65534,65535,65536}) {
        Check(FastllmGgufQuantizePlanar((half*)x.cudaData+size_t(row)*k,1,one,1,k,cudaStreamPerThread),"row quantizer");
        std::vector<unsigned char>a(FastllmGgufPlanarBytes(1,k)),b(a.size());
        Cuda(cudaMemcpy(a.data(),one,a.size(),cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(b.data(),(char*)all+size_t(row)*k,k,cudaMemcpyDeviceToHost));
        for(int plane=0;plane<2;++plane)Cuda(cudaMemcpy(b.data()+k+plane*(k/32)*4,
            (char*)all+size_t(rows)*k+size_t(plane*rows+row)*(k/32)*4,(k/32)*4,cudaMemcpyDeviceToHost));
        Check(a==b,"grid-y tile mismatch");
    }
    Cuda(cudaFree(one));Cuda(cudaFree(all));printf("PASS quantization rows=65537 grid-y tiling\n");
}
int main(int,char**) { try {
    FastllmCudaSetDevice(0);
    for(const char *c:{"cuda.planar.w2.r1.full","cuda.planar.w4.r1.occupancy","cuda.planar.w8.r2.occupancy"})
        for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_IQ4_XS}) {
            for(int rows:{1,2,3,4,5,6,7,8,9,17,33})Case(type,rows,768,73,c,rows==3);
            Case(type,1,2560,6144,c,false);Case(type,8,4096,129,c,false);
        }
    LargeQuantize();return 0;
} catch(const std::exception &e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;} }
