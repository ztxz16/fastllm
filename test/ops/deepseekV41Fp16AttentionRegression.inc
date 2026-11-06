#include "devices/cuda/fastllm-cuda-graph-workspace.h"
struct AttentionGraph {
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> reserved;
    template<class F> explicit AttentionGraph(F run) {
        std::vector<size_t> allocations;
        { FastllmCudaAllocationRecorder record(&allocations); run(); }
        Cuda(cudaDeviceSynchronize());
        Check(FastllmCudaReserveGraphWorkspace(allocations), "attention reserve");
        Check(FastllmCudaGraphMemoryPoolBegin(), "attention pool begin");
        Check(FastllmCudaGraphBeginCapture(), "attention capture begin"); run();
        Check(FastllmCudaGraphEndCapture(&graph), "attention capture end");
        Check(FastllmCudaGraphMemoryPoolEnd(reserved), "attention pool end");
        Check(FastllmCudaGraphInstantiate(graph, &exec), "attention instantiate");
    }
    bool UsesFp16() const {
#if CUDART_VERSION >= 12080
        size_t count=0; Cuda(cudaGraphGetNodes((cudaGraph_t)graph,nullptr,&count));
        std::vector<cudaGraphNode_t> nodes(count); Cuda(cudaGraphGetNodes((cudaGraph_t)graph,nodes.data(),&count));
        for(auto node:nodes) {
            cudaGraphNodeType type;Cuda(cudaGraphNodeGetType(node,&type));
            if(type!=cudaGraphNodeTypeKernel)continue;
            cudaKernelNodeParams params{};Cuda(cudaGraphKernelNodeGetParams(node,&params));
            const char *name=nullptr;Cuda(cudaFuncGetName(&name,params.func));
            if(name && std::string(name).find("v41_fp16_attention")!=std::string::npos &&
                std::string(name).find("Compute")!=std::string::npos)return true;
        }
        return false;
#else
        return true;
#endif
    }
    void Launch() { Check(FastllmCudaGraphLaunch(exec), "attention replay"); }
    ~AttentionGraph() {
        cudaDeviceSynchronize(); FastllmCudaGraphExecDestroy(exec);
        FastllmCudaGraphDestroy(graph); FastllmCudaGraphMemoryPoolRelease(reserved);
    }
};
// Independent attention oracle: disable FlashInfer only for the existing CUDA
// implementation, then compare before inverse RoPE / output quantization.
static void RunFp16AttentionRegression() {
    fp16Attention = true;
    int checks = 0;
    for (int heads : {1, 8, 17, 64, 128})
    for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 32})
    for (int mode : {0, 1, 2, 3}) {
        const int batch = heads == 17 ? 2 : 1, cap = 1024, width = 512;
        Data q, kv, ring, cmp, rp, cp, indices, sink, actual, expected;
        Fill(q, {batch, tokens, heads, 512}, 1);
        Fill(kv, {batch, tokens, 512}, 2);
        Fill(ring, {batch, 128, 512}, 3);
        Fill(cmp, {batch, cap, 512}, 4);
        Fill(indices, {batch, tokens, width}, 5, INT32);
        Fill(sink, {heads}, 6, FLOAT32);
        std::vector<int> ids(size_t(batch) * tokens * width);
        for (size_t i=0; i<ids.size(); ++i) ids[i] = i % 11 == 0 ? -1 : i % 13 == 0 ? cap+7 : (i*17)%cap;
        Cuda(cudaMemcpy(indices.cudaData, ids.data(), ids.size()*sizeof(int), cudaMemcpyHostToDevice));
        if (mode) {
            Check(FastllmCudaDeepSeekV41QuantizeKV(ring, rp, mode, mode==3?16:32), "pack ring");
            Check(FastllmCudaDeepSeekV41QuantizeKV(cmp, cp, mode, mode==3?16:32), "pack cmp");
        }
        auto run = [&](Data &out, int position, bool compressed) {
            Check(FastllmCudaDeepSeekV41SparseAttention(q, kv, mode?&rp:&ring,
                compressed?(mode?&cp:&cmp):nullptr, compressed?&indices:nullptr,
                sink, 128, position, 1/std::sqrt(512.f), out), "FP16 attention call");
        };
        for (int position : {0, 127, 8192}) {
            const bool compressed = position != 0;
            setenv("FASTLLM_DSV41_FLASHINFER", "0", 1); run(expected, position, compressed);
            setenv("FASTLLM_DSV41_FLASHINFER", "1", 1); run(actual, position, compressed);
            AttentionEqual(actual, expected, "attention eager");
            AttentionGraph graph([&]{ run(actual, position, compressed); });
            Check(graph.UsesFp16(), "FP16 Tensor Core kernel missing from graph");
            for(int replay=0;replay<3;++replay) {
                graph.Launch();
                AttentionEqual(actual, expected, "attention graph"); ++checks;
            }

        }
        std::cout << "attention heads="<<heads<<" rows="<<tokens<<" mode="<<mode<<" worst_l2="<<worstAttentionL2<<std::endl;
    }
    // Short/non-divisible live lists leave some of the eight partitions
    // empty. Exercise those graph paths without padding the index capacity.
    for (int window : {1, 8, 9, 17})
    for (int width : {1, 3, 7, 16, 64})
    for (int position : {0, 7, 128}) {
        Data q, kv, ring, cmp, indices, sink, actual, expected;
        Fill(q, {1, 3, 17, 512}, 1); Fill(kv, {1, 3, 512}, 2);
        Fill(ring, {1, window, 512}, 3); Fill(cmp, {1, 64, 512}, 4);
        Fill(indices, {1, 3, width}, 5, INT32); Fill(sink, {17}, 6, FLOAT32);
        std::vector<int> ids(3 * width);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = i % 3 ? i % 64 : -1;
        Cuda(cudaMemcpy(indices.cudaData, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        auto run = [&](Data &out) {
            Check(FastllmCudaDeepSeekV41SparseAttention(q, kv, &ring, &cmp, &indices,
                sink, window, position, 1 / std::sqrt(512.f), out), "short FP16 attention");
        };
        setenv("FASTLLM_DSV41_FLASHINFER", "0", 1); run(expected);
        setenv("FASTLLM_DSV41_FLASHINFER", "1", 1); run(actual);
        AttentionEqual(actual, expected, "short attention eager");
        AttentionGraph graph([&] { run(actual); });
        Check(graph.UsesFp16(), "short FP16 Tensor Core kernel missing");
        for (int replay = 0; replay < 3; ++replay) {
            graph.Launch(); AttentionEqual(actual, expected, "short attention graph"); ++checks;
        }
    }
    // A BF16 value outside FP16's finite range must use the original math
    // within the captured graph, rather than becoming an infinity.
    Data q,kv,sink,actual,expected;
    Fill(q,{1,1,8,512},1); Fill(kv,{1,1,512},2);Fill(sink,{8},3,FLOAT32);
    __nv_bfloat16 large=__float2bfloat16(131072.f);
    Cuda(cudaMemcpy(q.cudaData,&large,2,cudaMemcpyHostToDevice));
    auto run=[&](Data&out){Check(FastllmCudaDeepSeekV41SparseAttention(q,kv,nullptr,nullptr,nullptr,sink,128,0,.044194f,out),"range fallback");};
    setenv("FASTLLM_DSV41_FLASHINFER","0",1);run(expected);
    setenv("FASTLLM_DSV41_FLASHINFER","1",1);run(actual);Equal(actual,expected,"FP16 range fallback");
    AttentionGraph graph([&]{run(actual);});
    graph.Launch();Equal(actual,expected,"captured FP16 range fallback");
    std::vector<__nv_bfloat16> ordinaryQ(q.Count(0),__float2bfloat16(.3f));
    Cuda(cudaMemcpy(q.cudaData,ordinaryQ.data(),q.GetBytes(),cudaMemcpyHostToDevice));
    for(float value : {1e-7f,1e-20f}) {
        std::vector<__nv_bfloat16> tinyKV(kv.Count(0),__float2bfloat16(value));
        Cuda(cudaMemcpy(kv.cudaData,tinyKV.data(),kv.GetBytes(),cudaMemcpyHostToDevice));
        setenv("FASTLLM_DSV41_FLASHINFER","0",1);run(expected);
        setenv("FASTLLM_DSV41_FLASHINFER","1",1);graph.Launch();
        Equal(actual,expected,"captured FP16 tiny-vector fallback");
    }
    std::cout<<"PASS "<<checks<<" FP16 attention graph checks, worst relative L2="<<worstAttentionL2<<std::endl;
}
