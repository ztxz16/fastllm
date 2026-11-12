#include "cuda_gguf_fusion_test.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"

__global__ void Consume(const half *input, half *output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = __float2half_rn(__half2float(input[i]) * .5f);
}
__global__ void Finish(const half *a, const half *b, float *out) {
    *out = __half2float(*a) + __half2float(*b);
}

static void Case(ggml_type first, ggml_type second, int rows) {
    const int columns = 512;
    context = "hooks " + std::to_string(first) + "/" + std::to_string(second) + " rows=" + std::to_string(rows);
    Data input(FLOAT16, {1, rows, columns}); Allocate(input);
    Projection a(first, rows, columns, 129), b(second, rows, columns, 73);
    if (rows > 8) { a.w.forceGGUFFp32Dequant = false; b.w.forceGGUFFp32Dequant = false; }
    Data ca(FLOAT16, a.y.dims), cb(FLOAT16, b.y.dims); Allocate(ca); Allocate(cb);
    Data finished(FLOAT32, {1}); Allocate(finished);
    Data *weights[] = {&a.w, &b.w}, *outputs[] = {&a.y, &b.y};
    std::vector<int> order;
    Data independent(FLOAT16,input.dims); Allocate(independent);
    int region = 0, outer = 0;
    FastllmCudaGGUFProjectionHooks hooks{
        [&] { order.push_back(-1); region = FastllmCudaGraphBeginIndependentRegion(); },
        [&](int i) {
            order.push_back(i);
            Data &src = *outputs[i], &dst = i ? cb : ca;
            Consume<<<(src.Count(0)+255)/256,256,0,cudaStreamPerThread>>>(
                static_cast<const half *>(src.cudaData), static_cast<half *>(dst.cudaData), src.Count(0));
            if (!i) {
                if (region) { FastllmCudaGraphMarkParallelFirstDone(region); FastllmCudaGraphMarkParallelSecondBegin(region); }
            } else {
                if (region) FastllmCudaGraphMarkParallelJoin(region);
                if (outer) FastllmCudaGraphMarkParallelJoin(outer);
            }
        },
        [&] {
            order.push_back(-2); outer=FastllmCudaGraphBeginIndependentRegion();
            Consume<<<(input.Count(0)+255)/256,256,0,cudaStreamPerThread>>>(
                static_cast<const half *>(input.cudaData),static_cast<half *>(independent.cudaData),input.Count(0));
            if(outer) { FastllmCudaGraphMarkParallelFirstDone(outer); FastllmCudaGraphMarkParallelSecondBegin(outer); }
        }};
    auto run = [&] {
        order.clear();
        Check(FastllmCudaGGUFLinearShared(input, weights, outputs, 2, hooks), "hooks rejected");
        Check(order == std::vector<int>({-2,-1,0,1}), "callback order");
        Finish<<<1,1,0,cudaStreamPerThread>>>(static_cast<const half *>(ca.cudaData),
            static_cast<const half *>(cb.cudaData),static_cast<float *>(finished.cudaData));
    };
    auto prepare = [&](int seed) { Upload(input, Input(rows, columns, seed * 173)); a.Reference(input); b.Reference(input); };
    auto compare = [&] {
        a.Compare(); b.Compare();
        auto independentValues=Download(input,input.Count(0));
        for(auto &v:independentValues)v=__float2half_rn(__half2float(v)*.5f);
        Same(independent,independentValues,"preparation branch");
        for (int i=0;i<2;++i) {
            auto values = Download(i ? b.ref : a.ref, outputs[i]->Count(0));
            for (auto &v : values) v = __float2half_rn(__half2float(v)*.5f);
            Same(i ? cb : ca, values, "parallel consumer");
        }
    };
    prepare(0);
    std::vector<size_t> recorded;
    { FastllmCudaAllocationRecorder recorder(&recorded); run(); }
    Cuda(cudaDeviceSynchronize()); compare();
    Check(FastllmCudaGraphPrepareCaptureDevice(), "prepare capture");
    Check(FastllmCudaReserveGraphWorkspace(recorded), "reserve scratch");
    Check(FastllmCudaGraphMemoryPoolBegin(), "begin pool");
    Check(FastllmCudaGraphBeginCapture(), "begin capture"); run();
    void *graph=nullptr, *exec=nullptr; std::vector<void *> owned;
    Check(FastllmCudaGraphEndCapture(&graph), "end capture");
    Check(FastllmCudaGraphMemoryPoolEnd(owned), "end pool");
    Check(FastllmCudaGraphOptimizeParallelRegions(graph) == 2, "consumer branches did not split");
    Check(FastllmCudaGraphInstantiate(graph,&exec), "instantiate");
    for (int seed=1; seed<=3; ++seed) {
        prepare(seed); Check(FastllmCudaGraphLaunch(exec), "replay");
        Cuda(cudaDeviceSynchronize()); compare();
    }
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(owned);
    auto before = Download(a.y, a.y.Count(0));
    order.clear(); outputs[1] = outputs[0];
    Check(!FastllmCudaGGUFLinearShared(input, weights, outputs, 2, hooks), "output alias accepted");
    Check(order.empty(), "unsupported call invoked callbacks");
    Same(a.y, before, "unsupported call wrote output");
    ++cases; ++graphs;
}
static void Triple(int rows) {
    context = "QSA mixed triple rows=" + std::to_string(rows);
    Data input(FLOAT16,{1,rows,512});Allocate(input);
    Projection q(GGML_TYPE_Q6_K,rows,512,257),k(GGML_TYPE_Q4_K,rows,512,73),v(GGML_TYPE_IQ3_S,rows,512,129);
    std::vector<Projection*> projections{&q,&k,&v};
    for(auto p:projections)if(rows>8)p->w.forceGGUFFp32Dequant=false;
    Data *weights[]={&q.w,&k.w,&v.w},*outputs[]={&q.y,&k.y,&v.y};
    auto run=[&]{Check(FastllmCudaGGUFLinearShared(input,weights,outputs,3),"triple rejected");};
    auto prepare=[&](int seed){Upload(input,Input(rows,512,seed*173));for(auto p:projections)p->Reference(input);};
    prepare(0);std::vector<size_t> recorded;
    {FastllmCudaAllocationRecorder recorder(&recorded);run();}
    Cuda(cudaDeviceSynchronize());for(auto p:projections)p->Compare();
    Check(FastllmCudaGraphPrepareCaptureDevice(),"triple prepare");
    Check(FastllmCudaReserveGraphWorkspace(recorded),"triple reserve");
    Check(FastllmCudaGraphMemoryPoolBegin(),"triple pool");
    Check(FastllmCudaGraphBeginCapture(),"triple capture");run();
    void *graph=nullptr,*exec=nullptr;std::vector<void*>owned;
    Check(FastllmCudaGraphEndCapture(&graph),"triple end");
    Check(FastllmCudaGraphMemoryPoolEnd(owned),"triple pool end");
    Check(FastllmCudaGraphInstantiate(graph,&exec),"triple instantiate");
    for(int seed=1;seed<=3;++seed){prepare(seed);Check(FastllmCudaGraphLaunch(exec),"triple replay");Cuda(cudaDeviceSynchronize());for(auto p:projections)p->Compare();}
    FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);FastllmCudaGraphMemoryPoolRelease(owned);
    ++cases;++graphs;
}
int main(int, char **) {
    try {
        FastllmCudaSetDevice(0);
        for (int rows : {1,2,4,7,8,9,17,33,65}) {
            Triple(rows);
            Case(GGML_TYPE_Q6_K, GGML_TYPE_Q4_K, rows);
            Case(GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, rows);
            Case(GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K, rows);
        }
        printf("PASS %d shared projection hook cases, %d changed-input parallel graphs, bitwise parity\n", cases, graphs);
        return 0;
    } catch (const std::exception &e) { std::cerr << context << ": " << e.what() << '\n'; return 1; }
}
