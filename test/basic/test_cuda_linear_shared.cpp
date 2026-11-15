// Shared-input projection regression: independently generated EXL3 storage,
// ordinary linear/CPU references, changed-input graph replay, and atomic fallback.
#define main ExistingExl3OracleMain
#include "test_exl3.cpp"
#undef main
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-shared.h"
#include <memory>
#include <array>

static float Read(const Data &d, size_t at) {
    if (d.dataType == FLOAT32) return reinterpret_cast<const float *>(d.cpuData)[at];
    uint16_t v = reinterpret_cast<const uint16_t *>(d.cpuData)[at];
    if (d.dataType == FLOAT16) return half_to_float(v);
    uint32_t word = uint32_t(v) << 16; float f; std::memcpy(&f, &word, 4); return f;
}
static void Write(Data &d, size_t at, float x) {
    if (d.dataType == FLOAT32) reinterpret_cast<float *>(d.cpuData)[at] = x;
    else reinterpret_cast<uint16_t *>(d.cpuData)[at] = d.dataType == FLOAT16 ? float_to_half(x) : Float32ToBFloat16RNEBits(x);
}
static double Error(const Data &ref, const Data &got) {
    double diff = 0, norm = 0;
    for (size_t i = 0; i < ref.Count(0); ++i) {
        double x = Read(ref, i), y = Read(got, i); diff += (x-y)*(x-y); norm += x*x;
    }
    return std::sqrt(diff / std::max(norm, 1e-30));
}
static void CopyBack(const Data &gpu, Data &cpu) {
    cpu.dataType = gpu.dataType; cpu.UpdateUnitSize(); cpu.Resize(gpu.dims); cpu.Allocate(false);
    FastllmCudaCopyFromDeviceToHost(cpu.cpuData, gpu.cudaData, gpu.GetBytes());
}
static void FillWeight(Data &w, int k, int n, int bits, int parts, std::mt19937 &rng) {
    w.dataType = EXL3; w.UpdateUnitSize(); w.blockM = parts; w.blockK = bits; w.Resize({n,k}); w.Allocate(false);
    const int each = n / parts; const size_t size = Exl3PartBytes(each,k,bits), trellis = size_t(each)*k*bits/16;
    for (int p = 0; p < parts; ++p) {
        auto *b = w.cpuData + p*size;
        for (size_t i = 0; i < trellis; i += 4) { uint32_t v = rng(); std::memcpy(b+i,&v,4); }
        auto *scale = reinterpret_cast<uint16_t *>(b + trellis);
        for (int i = 0; i < k+each; ++i) scale[i] = float_to_half((int(rng()%127)-63)/512.0f);
    }
}
static double maxSeparate = 0, maxCpu = 0;
static void Case(int rows, DataType type, int k, const std::vector<int> &widths,
                 const std::vector<int> &bits, bool multipart, bool portable) {
    std::mt19937 rng(7123+rows+k+int(type)); const int count = widths.size();
    Data host(type,{1,rows,k}), input; host.Allocate(false);
    for (size_t i = 0; i < host.Count(0); ++i) Write(host,i,(int(rng()%127)-63)/32.0f);
    // The two-argument ToDevice(CUDA, {0}) selects the bool overload.
    input.CopyFrom(host); input.ToDevice(CUDA,{0},true);
    Data host32(FLOAT32,host.dims), input32; host32.Allocate(false);
    for (size_t i = 0; i < host.Count(0); ++i) Write(host32,i,Read(host,i));
    input32.CopyFrom(host32); input32.ToDevice(CUDA,{0},true);
    std::vector<std::unique_ptr<Data>> hw, gw, out, ref;
    std::vector<Data *> weights, outputs;
    for (int i = 0; i < count; ++i) {
        hw.emplace_back(new Data()); FillWeight(*hw.back(),k,widths[i],bits[i],multipart && i==0 ? 2 : 1,rng);
        gw.emplace_back(new Data()); gw.back()->CopyFrom(*hw.back()); gw.back()->ToDevice(CUDA,{0},true); weights.push_back(gw.back().get());
        { std::vector<char> bytes(gw.back()->GetBytes()); cudaMemcpy(bytes.data(),gw.back()->cudaData,bytes.size(),cudaMemcpyDeviceToHost); Require(hw.back()->GetBytes()==bytes.size() && !std::memcmp(hw.back()->cpuData,bytes.data(),bytes.size()),"weight transfer differs"); }
        const DataType store = i < 2 ? type : FLOAT32;
        out.emplace_back(new Data(store,{1,rows,widths[i]})); ref.emplace_back(new Data(store,{1,rows,widths[i]}));
        for (auto *d : {out.back().get(),ref.back().get()}) { d->ToDevice(CUDA,{0},false); d->Allocate(false); }
        outputs.push_back(out.back().get());
    }
    // Compressor projections formerly cast their inputs to FLOAT32.
    auto separate = [&] { for (int i = 0; i < count; ++i) Require(FastllmCudaExl3Linear(i < 2 ? input : input32,*weights[i],Data(),*ref[i]),"ordinary EXL3 linear"); };
    auto grouped = [&] {
        bool accepted = FastllmCudaLinearShared(input,weights.data(),outputs.data(),count);
        Require(accepted == !portable,"shared dispatch capability");
        if (!accepted) for (int i = 0; i < count; ++i)
            Require(FastllmCudaExl3Linear(input,*weights[i],Data(),*outputs[i]),"portable fallback");
    };
    separate(); grouped(); Require(cudaDeviceSynchronize()==cudaSuccess,"warmup sync");
    cudaGraph_t graph; cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin"); grouped();
    Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
    Require(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate");
    for (int turn = 0; turn < 3; ++turn) {
        for (size_t i = 0; i < host.Count(0); ++i) Write(host,i,(int(rng()%127)-63)/32.0f);
        FastllmCudaCopyFromHostToDevice(input.cudaData,host.cpuData,host.GetBytes());
        for (size_t i = 0; i < host.Count(0); ++i) Write(host32,i,Read(host,i));
        FastllmCudaCopyFromHostToDevice(input32.cudaData,host32.cpuData,host32.GetBytes()); separate();
        Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"replay");
        Require(cudaDeviceSynchronize()==cudaSuccess,"replay sync");
        for (int i = 0; i < count; ++i) {
            Data a,b; CopyBack(*ref[i],a); CopyBack(*out[i],b); double error = Error(a,b);
            maxSeparate = std::max(maxSeparate,error); Require(std::isfinite(error)&&error<0.008,"shared differs from separate");
            if (k <= 256) {
                Data cpu(out[i]->dataType,out[i]->dims); Exl3LinearCPU(host,*hw[i],Data(),cpu);
                double e = Error(cpu,b); maxCpu = std::max(maxCpu,e);
                Require(std::isfinite(e)&&e<0.008,"shared differs from independently validated CPU decoder");
            }
        }
    }
    // A late invalid member and overlapping outputs cannot partially overwrite earlier outputs.
    std::vector<std::vector<char>> before;
    for (auto *y : outputs) { Data v; CopyBack(*y,v); before.emplace_back(v.cpuData,v.cpuData+v.GetBytes()); }
    auto rejected = [&] { Require(!FastllmCudaLinearShared(input,weights.data(),outputs.data(),count),"invalid group accepted"); };
    Data *last = weights.back(); weights.back() = &input; rejected(); weights.back() = last;
    Data *y = outputs.back(); outputs.back() = outputs[0]; rejected(); outputs.back() = y;
    for (int i = 0; i < count; ++i) { Data v; CopyBack(*outputs[i],v); Require(!std::memcmp(v.cpuData,before[i].data(),v.GetBytes()),"rejection wrote outputs"); }
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
}
static void GroupedCase(int rows, DataType type) {
    const int groups=3,k=128,n=128;
    std::mt19937 rng(813+rows+int(type));
    Data hw;FillWeight(hw,k,groups*n,10,groups,rng);hw.Resize({groups,n,k});
    Data w;w.CopyFrom(hw);w.ToDevice(CUDA,std::vector<int>{0});
    Data host(type,{rows,groups,k}),input,output(type,{rows,groups,n});host.Allocate(false);
    for(size_t i=0;i<host.Count(0);++i)Write(host,i,(int(rng()%127)-63)/32.f);
    input.CopyFrom(host);input.ToDevice(CUDA,std::vector<int>{0});
    output.ToDevice(CUDA,std::vector<int>{0},false);output.Allocate(false);
    auto run=[&]{Require(FastllmCudaExl3Grouped(input,w,output),"strided grouped projection");};
    run();Require(cudaDeviceSynchronize()==cudaSuccess,"grouped warmup");
    cudaGraph_t graph;cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"grouped begin");run();
    Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"grouped end");
    Require(cudaGraphInstantiate(&exec,graph,0)==cudaSuccess,"grouped instantiate");
    for(int repeat=0;repeat<3;++repeat){
        for(size_t i=0;i<host.Count(0);++i)Write(host,i,(int(rng()%127)-63)/32.f);
        Require(cudaMemcpy(input.cudaData,host.cpuData,host.GetBytes(),cudaMemcpyHostToDevice)==cudaSuccess,"grouped changed input");
        Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"grouped replay");
        Data actual,reference(type,output.dims);Exl3GroupedCPU(host,hw,reference);CopyBack(output,actual);
        const double error=Error(reference,actual);maxCpu=std::max(maxCpu,error);
        Require(std::isfinite(error)&&error<.008,"grouped CPU oracle");
    }
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
}
int main(int argc,char **argv) {
    // Test expectation for a library built without the imported CUDA kernels.
    bool portable=argc>1 && std::string(argv[1])=="--portable";
    SetThreads(32); int cases=0;
    for (auto t : {FLOAT32,FLOAT16,BFLOAT16}) for (int n : {1,3,5,6,8,9,12,15,16,17,24,31,32,33,48,63,64,65,97,129}) {
        Case(n,t,256,{256,128,256,128},{10,10,10,10},false,portable); ++cases;
        if (!portable) { Case(n,t,256,{256,128,256},{8,10,12},true,false); ++cases; }
    }
    if (!portable) {
        for (int b : {2,3,4,5,6,7,8,10,12,14,16}) { Case(3,BFLOAT16,256,{128,256},{b,b},false,false);++cases; }
        for (int n : {1,5,6,9,12,17,24,32,33,48,64,65,97,129}) for (int jobs : {2,4}) {
            std::vector<int> ns{1280,512};if(jobs==4){ns.push_back(512);ns.push_back(512);}
            Case(n,BFLOAT16,5120,ns,std::vector<int>(jobs,10),false,false);++cases;
        }
    }
    for(int rows:{3,9,17,24,31,32,33,48,63,65,97,129})for(auto type:{FLOAT32,FLOAT16,BFLOAT16}){GroupedCase(rows,type);++cases;}
    printf("PASS shared projections: %d cases, max relative L2 separate=%.9f CPU=%.9f, portable=%d\n",cases,maxSeparate,maxCpu,int(portable));
}
