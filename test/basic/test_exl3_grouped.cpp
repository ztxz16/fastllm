// Reuse the independent circular-bit-stream decoder and explicit Walsh oracle.
#define main ExistingExl3OracleMain
#include "test_exl3.cpp"
#undef main
#include "devices/cuda/fastllm-cuda.cuh"

static float Value(const Data &d, size_t at) {
    if (d.dataType == FLOAT32) return reinterpret_cast<const float *>(d.cpuData)[at];
    const auto word = reinterpret_cast<const uint16_t *>(d.cpuData)[at];
    if (d.dataType == FLOAT16) return half_to_float(word);
    uint32_t bits = uint32_t(word) << 16; float f; std::memcpy(&f,&bits,4); return f;
}
static void Put(Data &d, size_t at, float f) {
    if (d.dataType == FLOAT32) reinterpret_cast<float *>(d.cpuData)[at] = f;
    else reinterpret_cast<uint16_t *>(d.cpuData)[at] = d.dataType == FLOAT16 ? float_to_half(f) : Float32ToBFloat16RNEBits(f);
}
static void GroupedCase(int bits, int groups, int rows, DataType type, bool graph, int k=256, int n=128) {
    std::mt19937 rng(1191 + bits + groups + rows);
    Data weight(EXL3); weight.blockK=bits; weight.blockM=groups; weight.Resize({groups,n,k}); weight.Allocate(false);
    const size_t bytes=Exl3PartBytes(n,k,bits), trellis=size_t(n)*k*bits/16;
    for (int g=0; g<groups; ++g) {
        auto *p=weight.cpuData+g*bytes;
        for (size_t at=0;at<trellis;at+=4) { uint32_t word=rng(); std::memcpy(p+at,&word,4); }
        auto *scales=reinterpret_cast<uint16_t *>(p+trellis);
        for (int at=0;at<k+n;++at) scales[at]=float_to_half((int(rng()%127)-63)/512.0f);
    }
    Data input(type,{1,rows,groups,k}), cpu(type,{1,rows,groups*n}); input.Allocate(false);
    for (size_t at=0;at<input.Count(0);++at) Put(input,at,(int(rng()%127)-63)/32.0f);
    Data gpuWeight; gpuWeight.CopyFrom(weight);
    Require(gpuWeight.GetBytes()==groups*bytes && gpuWeight.blockM==groups &&
        !std::memcmp(gpuWeight.cpuData,weight.cpuData,groups*bytes),"Grouped copy lost storage or metadata");
    gpuWeight.ToDevice(CUDA,std::vector<int>{0});
    Data gpuInput,gpuOutput(type,{1,rows,groups*n}); gpuInput.CopyFrom(input); gpuInput.ToDevice(CUDA,std::vector<int>{0});
    gpuOutput.ToDevice(CUDA,std::vector<int>{0},false); gpuOutput.Allocate(false);
    auto run=[&]{Require(FastllmCudaExl3Grouped(gpuInput,gpuWeight,gpuOutput),"Grouped GPU dispatch");};
    run(); Require(cudaDeviceSynchronize()==cudaSuccess,"Grouped GPU warmup");
    if (type == BFLOAT16) {
        Data actual;
        Require(FastllmCudaDeepSeekV4WoA(gpuInput,gpuWeight,groups,n,actual,false),"WoA operator integration");
        Require(actual.dims==std::vector<int>({1,rows,groups*n}),"WoA output shape");
        Data ref;ref.CopyFrom(gpuOutput);ref.ToDevice(CPU);actual.ToDevice(CPU);
        Require(ref.GetBytes()==actual.GetBytes() && !std::memcmp(ref.cpuData,actual.cpuData,ref.GetBytes()),"WoA operator changes grouped results");
    }

    cudaGraph_t captured=nullptr; cudaGraphExec_t executable=nullptr;
    if (graph) {
        Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"Grouped capture begin");
        run(); Require(cudaStreamEndCapture(cudaStreamPerThread,&captured)==cudaSuccess,"Grouped capture end");
        Require(cudaGraphInstantiate(&executable,captured,nullptr,nullptr,0)==cudaSuccess,"Grouped graph instantiate");
    }
    for (int turn=0;turn<(graph?3:1);++turn) {
        if (turn) for (size_t at=0;at<input.Count(0);++at) Put(input,at,(int(rng()%127)-63)/32.0f);
        Exl3GroupedCPU(input,weight,cpu);
        // Independent bit/rotation oracle on small shapes; larger shapes use
        // the same already-validated CPU arithmetic, with no GPU scratch reuse.
        if (k<=256) for (int g=0;g<groups;++g) {
            const auto *p=weight.cpuData+g*bytes;
            const auto matrix=Oracle(p,n,k,bits);
            const auto *scales=reinterpret_cast<const uint16_t *>(p+trellis);
            double error=0,norm=0;
            for (int row=0;row<rows;++row) {
                std::vector<double> x(k),y(n,0.0);
                for(int i=0;i<k;++i)x[i]=Value(input,(size_t(row)*groups+g)*k+i)*half_to_float(scales[i]);
                Walsh(x); for(auto &v:x)v=half_to_float(float_to_half(float(v)));
                for(int c=0;c<n;++c)for(int i=0;i<k;++i)y[c]+=matrix[size_t(c)*k+i]*x[i];
                Walsh(y);
                for(int c=0;c<n;++c){double expected=y[c]*half_to_float(scales[k+c]);
                    const double d=Value(cpu,(size_t(row)*groups+g)*n+c)-expected;error+=d*d;norm+=expected*expected;}
            }
            Require(std::sqrt(error/std::max(norm,1e-30))<(type==FLOAT32?1e-4:0.004),"Grouped CPU independent oracle");
        }
        FastllmCudaCopyFromHostToDevice(gpuInput.cudaData,input.cpuData,input.GetBytes());
        if (graph) Require(cudaGraphLaunch(executable,cudaStreamPerThread)==cudaSuccess,"Grouped replay"); else run();
        Require(cudaDeviceSynchronize()==cudaSuccess,"Grouped GPU completion");
        Data actual(type,cpu.dims);actual.Allocate(false);
        FastllmCudaCopyFromDeviceToHost(actual.cpuData,gpuOutput.cudaData,actual.GetBytes());
        double error=0,norm=0;
        for(size_t at=0;at<cpu.Count(0);++at){double a=Value(cpu,at),b=Value(actual,at);error+=(a-b)*(a-b);norm+=a*a;}
        Require(std::isfinite(error)&&std::sqrt(error/std::max(norm,1e-30))<(type==FLOAT32?2e-4:0.006),"Grouped GPU differs from CPU oracle");
    }
    if(executable)cudaGraphExecDestroy(executable);if(captured)cudaGraphDestroy(captured);
}
int main() {
    SetThreads(32);int cases=0;
    for(int bits:{2,3,4,5,6,7,8,10,12,14,16})for(int rows:{1,3,9}) {
        GroupedCase(bits,3,rows,FLOAT32,rows==1);++cases;
    }
    for(auto type:{FLOAT16,BFLOAT16})for(int groups:{1,8,9})for(int rows:{1,9}) {
        GroupedCase(10,groups,rows,type,rows==1);++cases;
    }
    GroupedCase(10,8,1,BFLOAT16,true,4096,1024);++cases;
    std::cout<<"EXL3 GROUPED PASS: "<<cases<<" cases; independent oracle, all bitrates, graph input changes and full WoA shape"<<std::endl;
    return 0;
}
