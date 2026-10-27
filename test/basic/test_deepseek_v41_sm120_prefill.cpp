#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
using namespace fastllm;

static void Check(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t status) { Check(status == cudaSuccess, cudaGetErrorString(status)); }
static Data Random(DataType type, const std::vector<int> &dims, int seed, float scale = .5f) {
    size_t count = 1; for (int d : dims) count *= d;
    std::mt19937 rng(seed); std::normal_distribution<float> normal(0, scale);
    std::vector<float> values(count); for (auto &v : values) v = normal(rng);
    Data out(type, dims, values);
    out.ToDevice(CUDA, std::vector<int>{FastllmCudaGetDevice()}); return out;
}
static std::vector<uint8_t> Bytes(Data &data) {
    Cuda(cudaDeviceSynchronize());
    std::vector<uint8_t> bytes(data.GetBytes());
    Cuda(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost));
    return bytes;
}
static void Equal(const std::vector<uint8_t> &expected, Data &actual, const std::string &label) {
    const auto got = Bytes(actual);
    if (expected == got) return;
    for (size_t i = 0; i < std::min(expected.size(), got.size()); ++i)
        if (expected[i] != got[i]) {
            fprintf(stderr, "%s: byte %zu expected=%u actual=%u\n", label.c_str(), i, expected[i], got[i]);
            break;
        }
    throw std::runtime_error(label + ": bitwise mismatch");
}
struct Case {
    int heads, window, start, width, capacity, ringMode, cmpMode, tokens;
    Data q, chunk, ring, cmp, sink, indices, packedRing, packedCmp, output;
    Case(int heads, int window, int start, int width, int ringMode, int cmpMode, int seed = 1,
         DataType queryType = BFLOAT16, int tokens = 1, int batch = 1)
        : heads(heads), window(window), start(start), width(width), capacity(std::max(64, width)),
          ringMode(ringMode), cmpMode(cmpMode), tokens(tokens),
          q(Random(queryType, {batch,tokens,heads,512}, seed, .1f)),
          chunk(Random(BFLOAT16, {batch,tokens,512}, seed+1)),
          ring(Random(BFLOAT16, {batch,std::max(1,window),512}, seed+2)),
          cmp(Random(BFLOAT16, {batch,capacity,512}, seed+3, seed % 3 ? .5f : 1e-6f)),
          sink(Random(FLOAT32, {heads}, seed+4, seed % 2 ? 1.f : 80.f)),
          indices(INT32, {batch,tokens,width}) {
        if (ringMode) Check(FastllmCudaDeepSeekV41QuantizeKV(ring, packedRing, ringMode,
                            ringMode == 3 ? 16 : 32), "pack ring");
        if (cmpMode) Check(FastllmCudaDeepSeekV41QuantizeKV(cmp, packedCmp, cmpMode,
                            cmpMode == 3 ? 16 : 32), "pack compressed");
        if (width) {
            indices.Allocate();
            std::vector<int32_t> ids(size_t(batch) * tokens * width);
            for (size_t i=0; i<ids.size(); ++i) ids[i] = i % 17 == 0 ? -1 :
                (i % 19 == 0 ? capacity : (i / 2 * 13 + seed + i / width * 997) % capacity);
            memcpy(indices.cpuData, ids.data(), ids.size()*sizeof(int32_t));
            indices.ToDevice(CUDA, std::vector<int>{FastllmCudaGetDevice()});
        }
    }
    void Run() {
        Check(FastllmCudaDeepSeekV41SparseAttention(q, chunk,
            start && window ? (ringMode ? &packedRing : &ring) : nullptr,
            width ? (cmpMode ? &packedCmp : &cmp) : nullptr, width ? &indices : nullptr,
            sink, window, start, 1.f / std::sqrt(512.f), output), "attention dispatch");
    }
    std::vector<uint8_t> Reference() {
        setenv("FASTLLM_DSV41_DISABLE_SM120_PREFILL", "1", 1); Run();
        auto bytes = Bytes(output); unsetenv("FASTLLM_DSV41_DISABLE_SM120_PREFILL"); return bytes;
    }
};


static void Bench(int rows, int width, int mode) {
    Case c(64,128,16384,width,1,mode,7,BFLOAT16,rows);
    auto expected = c.Reference(); c.Run(); Equal(expected, c.output, "production chunk");
    cudaEvent_t begin,end; Cuda(cudaEventCreate(&begin)); Cuda(cudaEventCreate(&end));
    for (int path=0;path<2;++path) {
        setenv("FASTLLM_DSV41_DISABLE_SM120_PREFILL",path ? "0" : "1",1);
        float total=0;
        for (int j=0;j<8;++j) {
            Cuda(cudaEventRecord(begin,cudaStreamPerThread)); c.Run();
            Cuda(cudaEventRecord(end,cudaStreamPerThread)); Cuda(cudaEventSynchronize(end));
            float ms; Cuda(cudaEventElapsedTime(&ms,begin,end)); if(j>=2)total+=ms;
        }
        printf("BENCH rows=%d top=%d format=%d path=%s ms=%.6f\n",rows,width,mode,path?"sm120":"original",total/6);
    }
    Cuda(cudaEventDestroy(begin)); Cuda(cudaEventDestroy(end));
    unsetenv("FASTLLM_DSV41_DISABLE_SM120_PREFILL");
}
int main(int argc,char **argv) {
    try {
        if (FastllmCudaGetDeviceCount()==0) return 77;
        FastllmCudaSetDevice(0); if(FastllmCudaRuntimeArch()!=120) return 77;
        if (argc > 1 && std::string(argv[1]) == "--sanitize") {
            for (int format : {0, 3}) {
                Case c(32,128,127,65,format,format,17,BFLOAT16,128);
                auto expected=c.Reference(); c.Run(); Equal(expected,c.output,"sanitizer case");
            }
            puts("PASS sanitizer cases"); return 0;
        }
        int checks=0;
        const int formats[][2]={{0,0},{1,3},{0,3},{1,0},{3,3},{2,2},{0,2},{3,0},{1,1}};
        const int shapes[][3]={{128,0,0},{128,31,33},{128,127,65},{128,129,512},
                             {4,3,513},{128,8192,0},{0,0,513}};
        for(int rows:{9,128,131}) for(int heads:{8,32,64,65}) for(const auto &fmt:formats) {
            const auto &shape=shapes[checks%7];
            Case c(heads,shape[0],shape[1],shape[2],fmt[0],fmt[1],301+checks,BFLOAT16,rows);
            auto expected=c.Reference(); c.Run(); Equal(expected,c.output,"case="+std::to_string(checks)); ++checks;
        }
        for(int batch:{1,2}) {
            Case c(64,128,16384,1024,1,3,11,BFLOAT16,2048,batch);
            auto expected=c.Reference(); c.Run(); Equal(expected,c.output,"large batch"); ++checks;
        }
        Case masked(64,0,0,65,0,3,13,BFLOAT16,128,2);
        Cuda(cudaMemset(masked.indices.cudaData,0xff,masked.indices.GetBytes()));
        auto expected=masked.Reference(); masked.Run(); Equal(expected,masked.output,"sink only");
        Check(Bytes(masked.output)==std::vector<uint8_t>(masked.output.GetBytes(),0),"sink-only nonzero"); ++checks;
        Case causal(64,128,127,65,1,3,17,BFLOAT16,131);
        causal.Run(); const auto before=Bytes(causal.output); std::vector<uint16_t> future(512,0x4300);
        Cuda(cudaMemcpy(static_cast<char*>(causal.chunk.cudaData)+130*512*2,future.data(),1024,cudaMemcpyHostToDevice));
        causal.Run(); const auto after=Bytes(causal.output);
        Check(std::equal(before.begin(),before.begin()+130*64*512*2,after.begin()),"causal mutation");
        expected=causal.Reference(); causal.Run(); Equal(expected,causal.output,"updated chunk"); ++checks;
        Case graphCase(65,128,127,513,1,3,19,BFLOAT16,128,2);
        expected=graphCase.Reference(); graphCase.Run(); std::vector<void*> reserved;
        Check(FastllmCudaGraphMemoryPoolBegin(),"graph pool"); Check(FastllmCudaGraphBeginCapture(),"graph capture");
        graphCase.Run(); void *graph=nullptr; Check(FastllmCudaGraphEndCapture(&graph),"graph end");
        Check(FastllmCudaGraphMemoryPoolEnd(reserved),"pool end"); cudaGraphExec_t exec;
        Cuda(cudaGraphInstantiate(&exec,(cudaGraph_t)graph,nullptr,nullptr,0));
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); Equal(expected,graphCase.output,"graph replay");
        Cuda(cudaMemset(graphCase.indices.cudaData,0xff,graphCase.indices.GetBytes())); expected=graphCase.Reference();
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); Equal(expected,graphCase.output,"graph updated indices");
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy((cudaGraph_t)graph)); FastllmCudaGraphMemoryPoolRelease(reserved); ++checks;
        Case fp32(32,128,127,65,1,3,23,FLOAT32,128);
        expected=fp32.Reference(); fp32.Run(); Equal(expected,fp32.output,"FP32 fallback"); ++checks;
        printf("PASS %d attention cases, bitwise parity, causal mutation and graph replay\n",checks);
        if(argc>1 && std::string(argv[1])=="--bench") for(int rows:{128,16384}) for(int width:{0,1024}) Bench(rows,width,3);
        return 0;
    }catch(const std::exception &e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
