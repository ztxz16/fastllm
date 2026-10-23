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
        setenv("FASTLLM_DSV41_LEGACY_ATTN", "1", 1); Run();
        auto bytes = Bytes(output); unsetenv("FASTLLM_DSV41_LEGACY_ATTN"); return bytes;
    }
};

static int PrefillChecks() {
    int checks = 0;
    const int formats[][2] = {{0,0},{1,3},{0,3},{1,0},{3,3},{2,2},{0,2},{3,0},{1,1}};
    const int shapes[][3] = {{128,0,0},{128,31,33},{128,127,65},{128,129,512},
                             {4,3,513},{128,8192,0},{0,0,513}};
    for (int tokens : {9,31,128}) for (int heads : {1,7,32,64,65,128,129})
        for (const auto &format : formats) {
            const auto &shape = shapes[checks % 7];
            Case c(heads,shape[0],shape[1],shape[2],format[0],format[1],301+checks,BFLOAT16,tokens);
            auto expected = c.Reference(); c.Run();
            Equal(expected,c.output,"prefill case="+std::to_string(checks));
            ++checks;
        }
    // Production chunk size and independent batch rows, including all-invalid
    // compressed indices and a sink-only softmax (window == 0).
    for (int tokens : {128,2048}) for (int batch : {1,2}) {
        Case c(64,128,2048,513,1,3,503+checks,BFLOAT16,tokens,batch);
        auto expected = c.Reference(); c.Run(); Equal(expected,c.output,"prefill batch/chunk");
        ++checks;
    }
    Case masked(7,0,0,33,0,3,13,BFLOAT16,31,2);
    Cuda(cudaMemset(masked.indices.cudaData,0xff,masked.indices.GetBytes()));
    auto expected = masked.Reference(); masked.Run(); Equal(expected,masked.output,"prefill sink only");
    Check(Bytes(masked.output) == std::vector<uint8_t>(masked.output.GetBytes(),0),"sink-only output");
    ++checks;

    // A later KV row must not change earlier queries across a ring wrap.
    Case causal(7,4,3,65,1,3,17,BFLOAT16,31);
    causal.Run(); const auto before = Bytes(causal.output);
    std::vector<uint16_t> future(512,0x4300);
    Cuda(cudaMemcpy(static_cast<char *>(causal.chunk.cudaData)+30*512*2,
                    future.data(),future.size()*2,cudaMemcpyHostToDevice));
    causal.Run(); const auto after = Bytes(causal.output);
    Check(std::equal(before.begin(),before.begin()+30*7*512*2,after.begin()),"prefill causal mutation");
    expected = causal.Reference(); causal.Run(); Equal(expected,causal.output,"prefill changed KV");
    ++checks;

    // The padded kernel needs no scratch allocation and must replay with live
    // indices rather than values frozen during capture.
    Case graphCase(65,128,127,513,1,3,19,BFLOAT16,128,2);
    expected = graphCase.Reference(); graphCase.Run();
    std::vector<void *> reserved;
    Check(FastllmCudaGraphMemoryPoolBegin(),"prefill graph pool");
    Check(FastllmCudaGraphBeginCapture(),"prefill graph capture");
    graphCase.Run(); void *graph = nullptr;
    Check(FastllmCudaGraphEndCapture(&graph),"prefill graph end");
    Check(FastllmCudaGraphMemoryPoolEnd(reserved),"prefill pool end");
    cudaGraphExec_t exec; Cuda(cudaGraphInstantiate(&exec,(cudaGraph_t)graph,nullptr,nullptr,0));
    Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); Equal(expected,graphCase.output,"prefill graph replay");
    Cuda(cudaMemset(graphCase.indices.cudaData,0xff,graphCase.indices.GetBytes()));
    expected = graphCase.Reference();
    Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); Equal(expected,graphCase.output,"prefill graph updated indices");
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy((cudaGraph_t)graph));
    FastllmCudaGraphMemoryPoolRelease(reserved);
    ++checks;
    return checks;
}

static void GraphChecks(bool capacityMiss, int tokens = 1) {
    Case cases[] = {{32,128,8440,513,1,3,317,BFLOAT16,tokens},
                    {32,128,0,513,3,2,619,BFLOAT16,tokens == 1 ? 1 : tokens - 1}};
    cudaGraph_t graphs[2]; cudaGraphExec_t execs[2]; cudaStream_t streams[2];
    std::vector<void *> reserved[2], held[2];
    std::vector<uint8_t> expected[2];
    for (int i=0; i<2; ++i) {
        expected[i] = cases[i].Reference(); cases[i].Run();
        Equal(expected[i], cases[i].output, "graph warmup");
        Check(FastllmCudaGraphMemoryPoolBegin(), "begin graph pool");
        Check(FastllmCudaGraphBeginCapture(), "begin capture");
        if (capacityMiss) {
            // Exhaust fitting idle blocks; fallback must submit no partial fast path.
            const size_t count = std::min(cases[i].window, cases[i].start + cases[i].tokens) + cases[i].width;
            // These cases all have >=512 candidates and include coefficient scratch.
            const size_t bytes = size_t(cases[i].tokens) * (count * (512+3*32+1) + 32) * sizeof(float);
            for (;;) {
                void *ptr = nullptr; const auto status = FastllmCudaTryMalloc(&ptr, bytes);
                if (status == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE) break;
                Check(status == FASTLLM_CUDA_TRY_MALLOC_SUCCESS, "unexpected pool allocation error");
                held[i].push_back(ptr);
            }
        }
        cases[i].Run();
        void *graph = nullptr;
        Check(FastllmCudaGraphEndCapture(&graph), "end capture"); graphs[i] = (cudaGraph_t)graph;
        Check(FastllmCudaGraphMemoryPoolEnd(reserved[i]), "end pool");
        Check(!FastllmCudaGetThreadError(), "capacity miss poisoned thread error");
        Cuda(cudaGraphInstantiate(&execs[i], graphs[i], nullptr, nullptr, 0));
        Cuda(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        if (!capacityMiss) {
            size_t count=0; Cuda(cudaGraphGetNodes(graphs[i], nullptr, &count));
            Check(count >= 4, "parallel attention/coefficient pass was not captured");
        }
    }
    for (void *a : reserved[0]) for (void *b : reserved[1])
        Check(a != b, "graphs reused a live scratch allocation");
    for (int repeat=0; repeat<20; ++repeat)
        for (int i=0; i<2; ++i) Cuda(cudaGraphLaunch(execs[i], streams[i]));
    Cuda(cudaDeviceSynchronize());
    for (int i=0; i<2; ++i) {
        Equal(expected[i], cases[i].output, "concurrent graphs");
        // The graph must read current indices, not values frozen at capture.
        std::vector<int32_t> invalid(size_t(cases[i].tokens) * cases[i].width, -1);
        Cuda(cudaMemcpy(cases[i].indices.cudaData, invalid.data(), invalid.size()*4, cudaMemcpyHostToDevice));
        expected[i] = cases[i].Reference();
        Cuda(cudaGraphLaunch(execs[i], streams[i]));
        Equal(expected[i], cases[i].output, "updated graph indices");
        Cuda(cudaGraphExecDestroy(execs[i])); Cuda(cudaGraphDestroy(graphs[i]));
        Cuda(cudaStreamDestroy(streams[i])); FastllmCudaGraphMemoryPoolRelease(reserved[i]);
        for (void *ptr : held[i]) FastllmCudaFree(ptr);
    }
}
static int CoefficientChecks() {
    int checks = 0;
    // Odd heads/candidate counts exercise float2 scratch alignment and partial
    // warps. Cover both sides of the coefficient dispatch and its largest list.
    for (int tokens : {1,3,4,5,7,8}) for (int width : {32,127,128,511,512,513,4096})
        for (int pattern : {0,1,2}) {
            Case c(7,0,0,width,0,3,1009+checks,BFLOAT16,tokens);
            if (pattern == 0) {
                std::vector<int32_t> invalid(size_t(tokens)*width,-1);
                Cuda(cudaMemcpy(c.indices.cudaData,invalid.data(),invalid.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
            } else if (pattern == 1) {
                // Large scores, negative/positive sinks, and tied zero scores
                // must retain the scalar online-softmax result exactly.
                Data q = Random(BFLOAT16,{1,tokens,7,512},233+checks,10000.f);
                Data sink = Random(FLOAT32,{7},431+checks,10000.f);
                Cuda(cudaMemcpy(c.q.cudaData,q.cudaData,q.GetBytes(),cudaMemcpyDeviceToDevice));
                Cuda(cudaMemcpy(c.sink.cudaData,sink.cudaData,sink.GetBytes(),cudaMemcpyDeviceToDevice));
            } else {
                Cuda(cudaMemset(c.q.cudaData,0,c.q.GetBytes()));
                Cuda(cudaMemset(c.sink.cudaData,0,c.sink.GetBytes()));
            }
            auto expected=c.Reference(); c.Run();
            Equal(expected,c.output,"coefficient boundary case="+std::to_string(checks));
            ++checks;
        }
    return checks;
}
static int VerifyChecks(bool smoke) {
    int checks = 0;
    const int formats[][2]={{1,3},{0,3},{1,0},{3,3},{2,2},{0,2},{3,0},{1,1}};
    const int shapes[][3]={{128,0,33},{128,125,65},{128,127,512},{128,129,512},
                           {4,0,33},{4,3,65},{128,8192,0},{128,8192,513},{0,0,1024}};
    for (int tokens : {2,3,6,8}) for (int heads : {1,7,32,64})
        for (const auto &shape : shapes) for (const auto &format : formats) {
            if (smoke && (tokens!=6 || heads!=32 || shape[1]!=127)) continue;
            Case c(heads,shape[0],shape[1],shape[2],format[0],format[1],101+checks,BFLOAT16,tokens);
            auto expected=c.Reference(); c.Run();
            Equal(expected,c.output,"verification tokens="+std::to_string(tokens)+" case="+std::to_string(checks));
            ++checks;
        }
    for (int tokens : {2,6,8}) {
        Case c(32,4,3,65,1,3,17,BFLOAT16,tokens);
        c.Run(); const auto before=Bytes(c.output);
        std::vector<uint16_t> future(512, 0x4300);  // BF16 128, only the last KV row changes.
        Cuda(cudaMemcpy(static_cast<char *>(c.chunk.cudaData)+(tokens-1)*512*2,
                        future.data(),future.size()*2,cudaMemcpyHostToDevice));
        c.Run(); const auto after=Bytes(c.output);
        Check(std::equal(before.begin(),before.begin()+(tokens-1)*32*512*2,after.begin()),
              "future chunk KV leaked into an earlier token");
        auto expected=c.Reference(); c.Run(); Equal(expected,c.output,"causal chunk mutation");
        std::vector<int32_t> invalid(c.width,-1);
        Cuda(cudaMemcpy(static_cast<char *>(c.indices.cudaData)+(tokens-1)*c.width*4,
                        invalid.data(),invalid.size()*4,cudaMemcpyHostToDevice));
        c.Run(); const auto changed=Bytes(c.output);
        Check(std::equal(after.begin(),after.begin()+(tokens-1)*32*512*2,changed.begin()),
              "per-token compressed indices leaked into another token");
        expected=c.Reference(); c.Run(); Equal(expected,c.output,"per-token index mutation");
        ++checks;
    }
    if (!smoke) {
        Case seqLimit(32,128,8192,65,1,3,1,BFLOAT16,9),
             fp32(32,128,8192,65,1,3,1,FLOAT32,6),
             candidateLimit(32,128,8192,4097,1,3,1,BFLOAT16,6);
        for (Case *c : {&seqLimit,&fp32,&candidateLimit}) {
            auto expected=c->Reference(); c->Run(); Equal(expected,c->output,"verification fallback"); ++checks;
        }
        for (int tokens : {2,6,8}) { GraphChecks(false,tokens); GraphChecks(true,tokens); }
    }
    return checks;
}
int main(int argc, char **argv) {
    int devices=0; if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        const std::string mode = argc>1 ? argv[1] : "all";
        unsetenv("FASTLLM_DSV41_REFERENCE_MATH"); unsetenv("FASTLLM_DSV41_LEGACY_ATTN");
        int checks=0, tested=0;
        for (int dev=0; dev<devices; ++dev) {
            ApplyDeviceMap({{"cuda:"+std::to_string(dev),1}},0,1); FastllmCudaSetDevice(dev);
            if (FastllmCudaRuntimeArch()!=75) continue;
            ++tested;
            if (mode=="prefill") { checks += PrefillChecks(); continue; }
            const int formats[][2]={{1,3},{0,3},{1,0},{3,3},{2,2},{0,2},{3,0},{1,1}};
            const int shapes[][3]={{128,0,0},{128,31,33},{128,127,65},{128,129,512},
                                   {512,8440,512},{0,0,1024},{128,8192,0},{128,8192,513}};
            for (int heads : {1,7,32,64}) for (const auto &shape : shapes) for (const auto &format : formats) {
                if (mode=="smoke" && (heads!=32 || shape[1]!=129)) continue;
                Case c(heads,shape[0],shape[1],shape[2],format[0],format[1],13+checks);
                const auto expected=c.Reference(); c.Run();
                Equal(expected,c.output,"packed attention dev="+std::to_string(dev)+" case="+std::to_string(checks));
                ++checks;
            }
            checks += VerifyChecks(mode=="smoke");
            if (mode!="smoke") {
                checks += PrefillChecks();
                checks += CoefficientChecks();
                for (int heads : {128,129}) {
                    Case c(heads,128,8192,33,1,3); auto expected=c.Reference();
                    c.Run(); Equal(expected,c.output,"head limit"); ++checks;
                }
                Case tooMany(32,128,8192,4097,1,3), fp32(32,128,8192,65,1,3,1,FLOAT32);
                for (Case *c : {&tooMany,&fp32}) {
                    auto expected=c->Reference(); c->Run(); Equal(expected,c->output,"fallback"); ++checks;
                }
                GraphChecks(false); GraphChecks(true);
            }
        }
        if (!tested) return 77;
        printf("PASS SM75 packed attention: %d bitwise cases on %d devices; mode=%s\n",checks,tested,mode.c_str());
        return 0;
    } catch (const std::exception &e) { fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
