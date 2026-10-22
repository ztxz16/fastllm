// Compare fused GPU routing to the existing per-expert V4.1 CUDA reference.
#include "fastllm.h"
#include "../../src/models/deepseekv41_execution.h"
#include "utils.h"
#include "gguf.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include <cuda_runtime.h>
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
namespace fastllm { void RegisterNumas(Data *data, std::string weightType); }
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
static std::vector<float> Read(const Data &data) {
    Check(data.dataType == FLOAT32, "expert sum must stay FP32");
    std::vector<float> out(data.Count(0));
    Cuda(cudaMemcpy(out.data(), data.cudaData, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

// Exercise the model boundary, not only the resident operator: a hybrid model
// can have GPU-resident GGUF layers whose routed sum is FP32, shared output FP16.
static bool hybridTest = false;
static void HybridAddition(Data &input, Data &indices, Data &scores,
                           std::vector<Data *> &weights, const std::vector<float> &routed,
                           float limit, int device) {
    class Probe : public DeepSeekV41Model {
    public:
        using DeepSeekV41Model::RunHybridMoe;
        void Configure(int hidden, int count, float limit, int device) {
            embed_dim = hidden;
            block_cnt = 1;
            swiglu_limit = limit;
            deviceMap = moeDeviceMap = {{"cuda:" + std::to_string(device), 1}};
            biass = {std::vector<Data *>(count, nullptr)};
            quantizedLinearNames.insert("layers.0.ffn.experts.0.w1.weight");
        }
    } model;
    model.Configure(input.dims.back(), weights.size(), limit, device);
    DeepSeekV41DecodeWorkspace ws;
    const std::vector<int> devices{device};
    // Reuse workspace across shared types/presence to catch stale conversion buffers.
    for (auto sharedType : {FLOAT16, BFLOAT16}) for (bool shared : {true, false}) {
        ws.ffnInput.CopyFrom(input);
        ws.expertIndex.CopyFrom(indices);
        ws.expertScore.CopyFrom(scores);
        std::vector<float> values(routed.size());
        for (size_t i = 0; i < values.size(); ++i) values[i] = (int(i % 31) - 15) / 64.f;
        ws.sharedExpertOut.CopyFrom(Data(FLOAT32, input.dims, values));
        ApplyDeviceMap({{"cuda:" + std::to_string(device), 1}}, 0, 1);
        ToDataType(ws.sharedExpertOut, sharedType);
        ws.sharedExpertOut.ToDevice(CUDA, {device}, true);
        DeepSeekV41MoeContext ctx{ws, weights, devices, 0, input.dims[0],
            false, true, false, false, false, shared, false, [] {}, [] {}};
        Check(!model.RunHybridMoe(ctx, nullptr), "unexpected TP dispatch");
        Check(ws.ffnOut.dataType == input.dataType, "hybrid MoE did not restore activation type");
        for (size_t i = 0; i < values.size(); ++i) values[i] = routed[i] + (shared ? values[i] : 0.f);
        Data expected(FLOAT32, input.dims, values);
        ToDataType(expected, input.dataType);
        ToDataType(expected, FLOAT32);
        ToDataType(ws.ffnOut, FLOAT32);
        auto actual = Read(ws.ffnOut);
        expected.ToDevice(CPU);
        Check(std::equal(actual.begin(), actual.end(), reinterpret_cast<float *>(expected.cpuData)),
              "hybrid MoE rounded before shared addition or changed routed values");
    }
}

static bool cacheTest = false;
static void CacheCompute(Data &input, Data &indices, Data &scores, std::vector<Data *> &weights,
                         const std::vector<float> &expected, int gt, int dt, int inter,
                         float limit, int device) {
    const int experts = weights.size()/2-1, rows = input.dims[0], hidden = input.dims[1], topk = indices.dims[1];
    const size_t gateBytes = size_t(2*inter)*ggml_row_size(ggml_type(gt), hidden);
    const size_t downBytes = size_t(hidden)*ggml_row_size(ggml_type(dt), inter);
    const size_t downOffset = (gateBytes+15)&~size_t(15), stride = (downOffset+downBytes+127)&~size_t(127);
    std::vector<size_t> offsets(experts);
    size_t bytes = 0;
    for (int slot=0; slot<experts; ++slot) { offsets[slot] = bytes; bytes += stride + 128*(slot+1); }
    Data records(INT8, {int(bytes)}); records.Allocate();
    std::memset(records.cpuData, 0, bytes);
    for (int expert=0; expert<experts; ++expert) for (int part=0; part<2; ++part) {
        const auto *w = weights[2*(expert+1)+part];
        if (w) Cuda(cudaMemcpy(records.cpuData+offsets[experts-1-expert]+(part?downOffset:0),
            w->cudaData, part?downBytes:gateBytes, cudaMemcpyDeviceToHost));
    }
    std::vector<int32_t> ids(rows*topk), slots(rows*topk);
    Cuda(cudaMemcpy(ids.data(), indices.cudaData, ids.size()*4, cudaMemcpyDeviceToHost));
    for (size_t i=0; i<ids.size(); ++i) slots[i] = weights[2*(ids[i]+1)] ? experts-1-ids[i] : -1;
    Data routes(INT32, {rows,topk}, CPU, slots.data());
    Data gpuOffsets(INT8, {int(offsets.size()*sizeof(size_t))}, CPU, offsets.data());
    Data scratch(INT8, {int(FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(hidden,inter))});
    Data partial(FLOAT32, {rows*topk,hidden}), activation;
    for (auto *v : {&records,&routes,&gpuOffsets,&scratch,&partial}) {
        v->ToDevice(CUDA,{device},true); v->Allocate(false);
    }
    FastllmCudaMoeGGUFCacheView view{static_cast<const uint8_t *>(records.cudaData),
        static_cast<const int32_t *>(routes.cudaData),stride,downOffset,gt,dt,hidden,inter,
        scratch.cudaData,scratch.GetBytes()};
    view.slotOffsets = static_cast<const size_t *>(gpuOffsets.cudaData);
    auto run = [&] { Check(FastllmCudaMoeV41GGUFCacheCompute(input,activation,view,
        static_cast<const float *>(scores.cudaData),topk,limit,static_cast<float *>(partial.cudaData)),
        "canonical cache compute rejected fixture"); };
    run();
    if (rows == 5) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        run(); Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    }
    const auto values = Read(partial);
    // Reorder and partition original routes, reusing the input Q8 prefix just
    // as resident and asynchronously uploaded experts do in the verifier.
    Cuda(cudaMemsetAsync(partial.cudaData, 0xff, partial.GetBytes(), cudaStreamPerThread));
    for (int partition = 0; partition < 3; ++partition) {
        std::vector<int32_t> selected, selectedSlots;
        for (int r = rows*topk-1; r >= 0; --r) if (r%3 == partition) {
            selected.push_back(r); selectedSlots.push_back(slots[r]);
        }
        if (selected.empty()) continue;
        Data map(INT32,{int(selected.size())},CPU,selected.data());
        Data chosen(INT32,{int(selectedSlots.size())},CPU,selectedSlots.data());
        map.ToDevice(CUDA,{device},true); chosen.ToDevice(CUDA,{device},true);
        view.routeMap = static_cast<const int32_t *>(map.cudaData);
        view.routeSlots = static_cast<const int32_t *>(chosen.cudaData);
        view.routeCount = selected.size(); view.q8InputPrepared = partition > 0;
        run(); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    Check(Read(partial) == values, "sparse canonical route mapping changed expert partials");
    for (int row=0; row<rows; ++row) {
        std::vector<int> order(topk); for(int k=0;k<topk;++k) order[k]=k;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ids[row*topk+a]<ids[row*topk+b];});
        for (int c=0;c<hidden;++c) {
            float sum = 0; for (int k : order) sum += values[size_t(row*topk+k)*hidden+c];
            Check(sum == expected[size_t(row)*hidden+c], "cached canonical experts differ from resident CUDA");
        }
    }
}


#ifdef USE_NUMAS
struct RegisteredFixture {
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> host, gpu;
};
static std::vector<RegisteredFixture> registeredFixtures;
static void CollectRegistrationFixture(std::vector<Data *> &weights, int gt, int dt, int device) {
    if (!((gt==GGML_TYPE_IQ1_S && dt==GGML_TYPE_IQ1_M) ||
          (gt==GGML_TYPE_Q2_0 && dt==GGML_TYPE_Q3_K) ||
          (gt==GGML_TYPE_Q4_K && dt==GGML_TYPE_Q2_0) ||
          (gt==GGML_TYPE_Q2_K && dt==GGML_TYPE_Q4_K))) return;
    RegisteredFixture f;
    f.host.resize(50); f.gpu.resize(50);
    for (int expert=0; expert<24; ++expert) for(int part=0; part<2; ++part) {
        auto *source=weights[2*(expert%8+1)+part];
        for (bool gpu : {false,true}) {
            auto w=std::make_unique<Data>(DATA_GGUF_FORMAT,source->ggmlType,source->dims);
            w->Allocate();
            Cuda(cudaMemcpy(w->cpuData,source->cudaData,source->GetBytes(),cudaMemcpyDeviceToHost));
            // Host weights retain normal NUMA repacking; the resident oracle
            // consumes the original canonical bytes.
            if (gpu) { w->disableGGUFRepack=true; w->ToDevice(CUDA,{device},true); }
            (gpu?f.gpu:f.host)[2*(expert+1)+part]=w.get(); f.owned.push_back(std::move(w));
        }
    }
    registeredFixtures.push_back(std::move(f));
}
static void CheckRegistration() {
    FastllmCudaSetDevice(0); EnableAMX(false);
    std::vector<FastllmCudaMoeCacheLayer> layers;
    size_t maxStride=0;
    for (auto &f : registeredFixtures) {
        layers.push_back({f.host.data(),int(f.host.size()),true,.8f});
        maxStride=std::max(maxStride,((f.host[2]->GetBytes()+15)&~size_t(15))+f.host[3]->GetBytes());
    }
    maxStride=(maxStride+127)&~size_t(127);
    const size_t budget=registeredFixtures.size()*8*maxStride;
    SetMoeCudaCacheBytes(budget);
    Check(FastllmCudaPrepareMoeCache(layers.data(),layers.size(),[&]{
        for(auto &f:registeredFixtures) for(size_t i=2;i<f.host.size();++i)
            RegisterNumas(f.host[i],i%2?"linear":"linearSwiglu");
    }),"mixed V4.1 NUMA GGUF cache registration failed");
    setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS","6",1);
    setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH","0",1);
    const int hidden=512, inter=256, topk=6;
    for(int pass=0;pass<5;++pass) for(size_t layer=0;layer<registeredFixtures.size();++layer) {
        auto &f=registeredFixtures[layer];
        const int rows=pass==4?5:1;
        std::vector<float> x(rows*hidden), scores(rows*topk);
        std::vector<int32_t> ids(rows*topk);
        for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*.031f);
        for(int r=0;r<rows;++r) for(int k=0;k<topk;++k) {
            // Repeated routes hit; disjoint sets overflow each layer's quota.
            ids[r*topk+k]=k+(pass<2?0:12);
            scores[r*topk+k]=k==0?0.f:k==1?-.17f:.25f+.13f*k;
        }
        Data input(FLOAT32,{rows,hidden},x), index(INT32,{rows,topk},CPU,ids.data());
        Data score(FLOAT32,{rows,topk},scores), output, reference, gate, activation, workspace;
        ToDataType(input,BFLOAT16);
        for(auto *v:{&input,&index,&score}) v->ToDevice(CUDA,{0},true);
        Check(FastllmCudaMergeMOEV41GGUFResident(input,gate,activation,workspace,reference,
            f.gpu.data(),f.gpu.size(),index,score,0,.8f),"registered cache reference failed");
        Check(FastllmCudaMergeMOEHybrid(input,index,score,output,f.host.data(),f.host.size(),layer),
            "registered cache dispatch failed");
        // The hybrid boundary rounds the routed sum to BF16.
        ToDataType(reference,BFLOAT16); ToDataType(reference,FLOAT32); ToDataType(output,FLOAT32);
        const auto expected=Read(reference), actual=Read(output);
        Check(actual==expected,"NUMA restoration, mixed layer offsets or verifier cache changed values");
    }
    uint64_t stats[5]={};
    Check(fastllm_moe_cuda_cache_stats(0,stats,false),"cache statistics failed");
    Check(stats[0]>0 && stats[1]>0 && stats[2]<=budget && stats[3]<stats[4],
          "cache test did not exercise bounded hits, admissions and eviction");
    uint64_t routes[8]={}; Check(fastllm_moe_cuda_cache_route_stats(0,routes),"route statistics failed");
    Check(routes[4]>0 && routes[6]>0,"cache did not execute resident GPU routes");
    printf("PASS: mixed V4.1 NUMA GGUF cache hits=%llu misses=%llu bytes=%llu slots=%llu records=%llu gpu_routes=%llu\n",
        (unsigned long long)stats[0],(unsigned long long)stats[1],(unsigned long long)stats[2],
        (unsigned long long)stats[3],(unsigned long long)stats[4],(unsigned long long)routes[4]);
    // Release Data while the executor/NUMA workers are still alive, before
    // process-static teardown reverses their construction order.
    registeredFixtures.clear();
}
#endif
static bool groupedTest = false;
static bool sparseTest = false;
static int Case(int gt, int dt, int hidden, int inter, int device, bool missing) {
    FastllmCudaSetDevice(device);
    const int experts = sparseTest ? 384 : 8, topk = 6;
    std::mt19937 rng(317 + gt*17+dt);
    std::normal_distribution<float> normal(0, .04f);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights(2*(experts+1), nullptr);
    for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
        if (missing && expert == 3) continue;
        const int rows = part ? hidden : 2*inter, cols = part ? inter : hidden;
        auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, part ? dt : gt, std::vector<int>{rows, cols});
        w->disableGGUFRepack = true;
        std::vector<float> values(size_t(rows)*cols);
        for (float &x : values) x = normal(rng);
        const int type = part ? dt : gt;
        if (type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
            // These CUDA-only formats do not all have a CPU encoder. Construct valid
            // packed fixtures directly; the reference consumes the same bytes.
            w->Allocate(false);
            for (size_t i=0; i<w->GetBytes(); ++i) w->cpuData[i] = rng()>>24;
            for (size_t i=0; i<w->GetBytes(); i+=ggml_type_size(ggml_type(type))) {
                const uint16_t scale = float_to_half(.02f + .002f*(rng()%8));
                if (type == GGML_TYPE_IQ1_M) {
                    // Four high nibbles encode one shared FP16 scale. The
                    // lower twelve bits are valid subblock scales.
                    auto *sc = reinterpret_cast<uint16_t *>(w->cpuData+i+48);
                    for (int j=0; j<4; ++j) sc[j] = (sc[j]&0xfff) | ((scale>>(4*j)&15)<<12);
                } else std::memcpy(w->cpuData+i, &scale, sizeof(scale));
            }
        } else w->CreateFromOriData(WeightType::LINEAR, FLOAT32, (uint8_t *)values.data(), nullptr, nullptr);
        w->ToDevice(CUDA, {device}, true);
        weights[2*(expert+1)+part] = w.get();
        owned.push_back(std::move(w));
    }
    Data gate, activation, workspace, output, ref;
    int checks = 0;
    for (int rows : sparseTest ? std::vector<int>{512,1024} : groupedTest ? std::vector<int>{33,64,129} : cacheTest ? std::vector<int>{1,5,8} : std::vector<int>{1,2,8,32}) for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<float> x(size_t(rows)*hidden), scales(rows*topk);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = pattern == 2 ? 0.f : std::ldexp(normal(rng)*30.f, int(i/32%5)-2);
        Data input(FLOAT32, {rows, hidden}, x);
        ToDataType(input, BFLOAT16);
        Data indices(INT32, {rows, topk}); indices.Allocate();
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
            ((int32_t *)indices.cpuData)[row*topk+k] = (7-k+row)%experts;
            scales[row*topk+k] = k == 0 ? 0.f : k == 1 ? -.17f : .25f + .13f*k;
        }
        Data scores(FLOAT32, {rows, topk}, scales);
        for (auto *v : {&input, &indices, &scores}) v->ToDevice(CUDA, {device}, true);
        const float limit = pattern == 0 ? 0.f : .8f;
        const bool accepted = FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
            weights.data(), weights.size(), indices, scores, 0.f, limit);
        if (groupedTest && (gt == GGML_TYPE_Q2_K || dt == GGML_TYPE_Q2_K)) {
            Check(!accepted, "unsupported grouped type did not preserve fallback");
            checks++;
            continue;
        }
        Check(accepted, "resident dispatch rejected fixture");
        const auto actual = Read(output);
        if (cacheTest) CacheCompute(input,indices,scores,weights,actual,gt,dt,inter,limit,device);
        if (hybridTest) HybridAddition(input, indices, scores, weights, actual, limit, device);
        if (groupedTest && gt != GGML_TYPE_Q2_K && dt != GGML_TYPE_Q2_K) {
            const size_t ordinary = (size_t(rows)*(hidden/32) + size_t(rows*topk)*(inter/32))*36 +
                size_t(rows*topk)*hidden*2;
            Check(workspace.GetBytes() > ((ordinary+255)&~size_t(255)), "grouped test did not select MMQ");

        }
        // Per-row reference retains the arithmetic of decode independent of
        // fallback's large-batch GEMM selection and expert occupancy.
        std::vector<float> expected;
        for (int row = 0; row < rows; ++row) {
            Data xi, ii, ss;
            xi.FakeFrom(input, size_t(row)*hidden*2); xi.Resize({1, hidden});
            ii.FakeFrom(indices, size_t(row)*topk*4); ii.Resize({1, topk});
            ss.FakeFrom(scores, size_t(row)*topk*4); ss.Resize({1, topk});
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            DoCudaMergeMOEV41Fallback(xi, ref, ii, ss, weights.data(), weights.size(), 0.f, limit);
            Check(FastllmCudaMergeMOEUsedGraphUnsafeFallback(), "reference did not use per-expert fallback");
            auto values = Read(ref); expected.insert(expected.end(), values.begin(), values.end());
        }
        double err2=0, norm2=0, maxabs=0; int different=0;
        for (size_t i=0;i<actual.size();++i) {
            Check(std::isfinite(actual[i]), "non-finite result");
            const double diff=double(actual[i])-expected[i];
            different += actual[i]!=expected[i]; err2+=diff*diff;
            norm2+=double(expected[i])*expected[i]; maxabs=std::max(maxabs,std::abs(diff));
        }
        printf("gt=%d dt=%d h=%d i=%d device=%d missing=%d rows=%d pattern=%d different=%d maxabs=%.9g rel=%.9g\n",
            gt,dt,hidden,inter,device,missing,rows,pattern,different,maxabs,std::sqrt(err2/std::max(norm2,1e-30)));
        fflush(stdout);
        if (!groupedTest) Check(different == 0, "indexed differs from per-token CUDA reference");
        else Check(std::sqrt(err2/std::max(norm2,1e-30)) < .002, "grouped differs materially from per-token CUDA reference");
        if ((rows == 2 || rows == 64) && pattern == 1) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Check(FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
                weights.data(), weights.size(), indices, scores, 0.f, limit), "graph dispatch");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(Read(output) == actual, "graph replay changed output");
            // Replay must read changing routes, not baked-in host routing.
            std::vector<int32_t> changed(rows*topk);
            Cuda(cudaMemcpy(changed.data(), indices.cudaData, changed.size()*4, cudaMemcpyDeviceToHost));
            for (auto &e : changed) e = (e+3)%experts;
            Cuda(cudaMemcpy(indices.cudaData, changed.data(), changed.size()*4, cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            const auto replayChanged = Read(output);
            Check(FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
                weights.data(), weights.size(), indices, scores, 0.f, limit), "changed-route dispatch");
            Check(Read(output) == replayChanged, "graph replay did not consume changed GPU routes");
            Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        }
        checks++;
    }
#ifdef USE_NUMAS
    if (cacheTest && !missing) CollectRegistrationFixture(weights,gt,dt,device);
#endif
    return checks;
}

int main(int argc, char **argv) {
    try {
        int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        SetThreads(4);
        const std::string mode = argc>1 ? argv[1] : "";
        hybridTest = mode == "--hybrid";
        cacheTest = mode == "--cache";
        if (hybridTest) {
            const int n = Case(GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, 512, 256, 0, false);
            printf("PASS: V4.1 hybrid GGUF FP32/shared FP16/BF16 boundary (%d cases)\n", n);
            return 0;
        }
        groupedTest = mode.find("--grouped") == 0 || mode == "--sparse";
        sparseTest = mode == "--sparse";
        if (sparseTest) {
            int n = 0;
            for (auto pair : std::vector<std::pair<int,int>>{{GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_IQ1_S,GGML_TYPE_Q2_0}, {GGML_TYPE_Q2_0,GGML_TYPE_Q4_K}})
                n += Case(pair.first,pair.second,512,256,0,false);
            printf("PASS: V4.1 sparse routing grouped MMQ (%d checks)\n", n);
            return 0;
        }
        int checks=0;
        for(int gt : {GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M})
        for(int dt : {GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M})
            checks+=Case(gt,dt,512,256,0,false);
        checks+=Case(GGML_TYPE_Q3_K,GGML_TYPE_Q2_0,512,256,devices>1?1:0,true);
        // TP ranks can lack either or both IQ1 projection weights.
        for (auto pair : std::vector<std::pair<int,int>>{
                {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},
                {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_S},
                {GGML_TYPE_IQ1_M,GGML_TYPE_IQ1_M}})
            checks+=Case(pair.first,pair.second,512,256,devices>1?1:0,true);
        if (mode == "--full" || mode == "--grouped-full") {
            for (auto pair : std::vector<std::pair<int,int>>{
                    {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_S}, {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_IQ1_S,GGML_TYPE_Q2_0}, {GGML_TYPE_Q2_0,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_Q2_0,GGML_TYPE_Q3_K}, {GGML_TYPE_Q2_0,GGML_TYPE_Q4_K}})
                checks+=Case(pair.first,pair.second,5120,2304,0,false);
        }
#ifdef USE_NUMAS
        if (cacheTest) CheckRegistration();
#endif
        printf("PASS: V4.1 resident GGUF %s (%d checks)\n", cacheTest ? "canonical cache" : groupedTest ? "grouped prefill" : "indexed decode", checks);
        fflush(stdout);
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
