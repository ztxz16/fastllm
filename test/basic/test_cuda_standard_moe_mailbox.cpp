#include "utils/hybrid_moe.h"
#include "fastllm.h"
#include "models/qwen4_exp.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/numas/numasdevice.h"
#include "gguf.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cuda(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void Gpu(Data &d) { d.ToDevice(CUDA, {FastllmCudaGetDevice()}, false); d.Allocate(false); }

namespace fastllm {
struct Qwen4GGUFTestAccess {
    static void CheckResidentPrefixPolicy(const std::vector<Data *> &hostWeights, int hidden) {
        Qwen4ExpModel model;
        Data resident(FLOAT16, {2, hidden}); Gpu(resident);
        model.block_cnt = 2;
        model.weights = {{nullptr, nullptr, &resident}, hostWeights};
        model.moeDeviceMap = {{"cuda:1", 1}, {"numa", 1}};
        bool correct = true;
        for (int rows : {1, 4}) {
            Data input(FLOAT16, {1, rows, hidden});
            auto policy = model.HybridDecodeGraphPolicy(input, rows > 1);
            correct &= policy.eligible && policy.hybridDense;
        }
        // The fixture owns the cache and weights, not this model instance.
        model.weights.clear();
        Require(correct, "resident prefix hid later NUMA layers from graph policy");
    }
};
}

static std::unique_ptr<Data> Weight(int kind, int rows, int columns, int seed, bool down) {
    std::unique_ptr<Data> w;
    if (kind < 0) {
        w.reset(new Data(NVFP4_BLOCK_16_E4M3, {rows, columns}));
        w->blockK = 1; w->blockM = 16; w->Allocate();
        const size_t packed = GetNVFP4WeightBytes(rows, columns);
        std::memset(w->cpuData, 0, w->GetBytes());
        std::memset(w->cpuData + packed, 0x38, size_t(rows) * columns / 16);
        for (int row = 0; row < rows; ++row) {
            const int col = (row * 17 + seed * 7) % columns;
            w->cpuData[size_t(row) * columns / 2 + col / 2] = (row % 3 ? 2 : 10) << ((col & 1) * 4);
        }
        w->scales = down ? std::vector<float>{.125f} : std::vector<float>{.5f, .25f};
    } else {
        const auto type = static_cast<ggml_type>(kind);
        w.reset(new Data(DATA_GGUF_FORMAT, type, {rows, columns}));
        w->Allocate();
        unsigned random = seed + 937;
        for (size_t i = 0; i < w->GetBytes(); ++i) {
            random = random * 1664525u + 1013904223u;
            w->cpuData[i] = random >> 24;
        }
        // Keep both chained layers in range when exercising the real 2560 x
        // 640 geometry. Random packed IQ weights need fan-in normalization;
        // otherwise the second layer overflows the existing FP16 GEMV too.
        const float fanIn = 256.f / std::max(columns, 256);
        const uint16_t scale = __half_as_ushort(__float2half_rn(
            fanIn * (type == GGML_TYPE_IQ4_XS ? 1.f/8192 : 1.f/256)));
        for (size_t offset = 0; offset < w->GetBytes(); offset += ggml_type_size(type))
            std::memcpy(w->cpuData + offset, &scale, sizeof(scale));
    }
    return w;
}
static void Run(int gateKind, int downKind, int hidden = 256, int inter = 128, int passes = 40,
                bool ownedPp = false, int device = 0, bool qwenTopk10 = false) {
    constexpr int experts = 64, layers = 2;
    const int topk = qwenTopk10 ? 10 : 6;
    Cuda(cudaSetDevice(device));
    const int cacheLayers = layers + int(ownedPp);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights[layers + 1];
    std::vector<std::pair<Data *, std::string>> registration;
    FastllmCudaMoeCacheLayer descriptors[layers + 1];
    size_t stride = 0;
    for (int layer = 0; layer < cacheLayers; ++layer) {
        weights[layer].resize((experts + 1) * 2, nullptr);
        for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
            auto w = Weight(part ? downKind : gateKind, part ? hidden : 2 * inter,
                            part ? inter : hidden, expert + layer * 101 + part * 19, part);
            weights[layer][(expert+1)*2+part] = w.get();
            registration.push_back({w.get(), part ? "linear" : "linearSwiglu"});
            owned.push_back(std::move(w));
        }
        stride = std::max(stride, size_t(weights[layer][2]->GetBytes() + weights[layer][3]->GetBytes() + 256));
        descriptors[layer] = {weights[layer].data(), int(weights[layer].size()), false, 0, false,
            ownedPp ? (layer < layers ? device : 1-device) : -1, true};
    }
    SetMoeCudaCacheBytes(32 * ((stride+127)/128*128));
    Require(FastllmCudaPrepareMoeCache(descriptors, cacheLayers, [&] { RegisterNumasMoeWeightBatch(registration); }),
            "prepare Qwen shared cache");
    Require(FastllmCudaMoeCacheIsQwen38(weights[0].data(), weights[0].size()), "Qwen cache profile missing");
    // The native cache must agree with an independently prepared canonical
    // cache. Reusing the candidate's resident path as its oracle would conceal
    // a shared row-layout bug in admission and graph replay.
    const bool native = gateKind >= 0;
    std::vector<std::unique_ptr<Data>> referenceOwned;
    std::vector<Data *> referenceWeights[layers];
    if (native) {
        FastllmCudaMoePrefillResidents resident;
        Require(FastllmCudaGetMoePrefillResidents(weights[0].data(), experts, resident, true) &&
                resident.nativeGGUF, "native GGUF cache was not selected");
        std::vector<std::pair<Data *, std::string>> referenceRegistration;
        FastllmCudaMoeCacheLayer referenceDescriptors[layers];
        for (int layer = 0; layer < layers; ++layer) {
            referenceWeights[layer].resize((experts + 1) * 2, nullptr);
            for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
                auto w = Weight(part ? downKind : gateKind, part ? hidden : 2*inter,
                    part ? inter : hidden, expert + layer*101 + part*19, part);
                referenceWeights[layer][(expert+1)*2+part] = w.get();
                referenceRegistration.push_back({w.get(), part ? "linear" : "linearSwiglu"});
                referenceOwned.push_back(std::move(w));
            }
            referenceDescriptors[layer] = {referenceWeights[layer].data(), int(referenceWeights[layer].size()),
                false, 0, false, ownedPp ? device : -1};
        }
        Require(FastllmCudaPrepareMoeCache(referenceDescriptors, layers,
            [&] { RegisterNumasMoeWeightBatch(referenceRegistration); }), "prepare canonical oracle cache");
        Require(!FastllmCudaMoeCacheIsQwen38(referenceWeights[0].data(), referenceWeights[0].size()),
                "Qwen cache profile leaked into Standard oracle");
        Require(!FastllmCudaNativeMoeGraphOwner(referenceWeights[0].data(), referenceWeights[0].size(), 1),
                "Qwen graph dispatch leaked into Standard oracle");
        Require(FastllmCudaGetMoePrefillResidents(referenceWeights[0].data(), experts, resident, true) &&
                !resident.nativeGGUF, "oracle cache unexpectedly uses native GGUF");
    }
    if (ownedPp) {
        Cuda(cudaSetDevice(1-device));
        void *scope = FastllmCudaBeginMoeDecode(weights[layers].data(), weights[layers].size(), topk);
        Require(scope, "prepare peer owner policy");
        Require(FastllmCudaNativeMoeGraphOwner(weights[layers].data(), weights[layers].size(), 1),
            "peer graph rejected its own PP layer");
        Require(!FastllmCudaNativeMoeGraphOwner(weights[0].data(), weights[0].size(), 1),
            "peer graph accepted another PP owner's layer");
        FastllmCudaEndMoeDecode(scope);
        Cuda(cudaSetDevice(device));
    }
    if (std::getenv("FASTLLM_TEST_PLACEMENT_POLICY"))
        Qwen4GGUFTestAccess::CheckResidentPrefixPolicy(weights[0], hidden);
    double worstL2 = 0;
    int checks = 0;
    // The exact Qwen shape uses its FP16 activation and all verifier widths,
    // including the non-power-of-two MTP2 case. Keep the broader dtype/row
    // sweep separate so exhaustive placement enumeration remains bounded.
    const std::vector<DataType> types = qwenTopk10 ? std::vector<DataType>{FLOAT16}
                                                 : std::vector<DataType>{FLOAT32, FLOAT16, BFLOAT16};
    const std::vector<int> rowCounts = qwenTopk10 ? std::vector<int>{1, 2, 3, 4, 9}
                                               : std::vector<int>{1, 2, 4, 8};
    for (DataType dtype : types) for (int rows : rowCounts) {
        Data input(dtype, {rows, hidden}), ids[layers], scores[layers], output[layers]; Gpu(input);
        for (int layer = 0; layer < layers; ++layer) {
            ids[layer].dataType = INT32; ids[layer].Resize({rows,topk}); Gpu(ids[layer]);
            scores[layer].dataType = FLOAT32; scores[layer].Resize({rows,topk}); Gpu(scores[layer]);
        }
        void *scope = FastllmCudaBeginMoeDecode(weights[0].data(), weights[0].size(), topk);
        Require(scope, "begin Standard cache policy");
        if (ownedPp) {
            Require(FastllmCudaNativeMoeGraphOwner(weights[0].data(), weights[0].size(), rows),
                "local graph rejected its own PP layer");
            Require(!FastllmCudaNativeMoeGraphOwner(weights[layers].data(), weights[layers].size(), rows),
                "local graph accepted another PP owner's layer");
        }
        FastllmCudaEndMoeDecode(scope);
        void *handles[layers]{}, *children[layers]{}, *graph = nullptr, *exec = nullptr, *replay = nullptr;
        for (int layer = 0; layer < layers; ++layer) {
            handles[layer] = FastllmCudaCreateNativeMoeFullGraph(layer ? output[0] : input,
                ids[layer], scores[layer], output[layer], weights[layer].data(), weights[layer].size(), layer,
                nullptr, &children[layer]);
            Require(handles[layer] && children[layer], "create Standard mailbox child");
        }
        HybridMoeGraph managed;
        if (qwenTopk10) {
            Require(managed.Compose(std::vector<const void *>(children, children + layers),
                std::vector<void *>(handles, handles + layers)), "compose managed Standard mailbox");
        } else {
            Require(FastllmCudaGraphCompose(const_cast<const void *const *>(children), layers, &graph, &exec), "compose Standard mailbox");
            Require(FastllmCudaCreateNativeMoeReplayGraph(graph, handles, layers, &replay), "strip Standard timing");
        }
        for (int pass = 0; pass < passes; ++pass) {
            std::vector<float> source(rows*hidden), actual[layers];
            std::vector<int32_t> routes[layers];
            std::vector<float> scale[layers];
            std::vector<uint16_t> packed(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                source[i] = float(int((i*13+pass*7)%61)-30)/32;
                packed[i] = dtype == FLOAT16 ? __half_as_ushort(__float2half_rn(source[i])) : Float32ToBFloat16RNEBits(source[i]);
            }
            Cuda(cudaMemcpy(input.cudaData, dtype == FLOAT32 ? (void *)source.data() : (void *)packed.data(),
                            source.size() * (dtype == FLOAT32 ? 4 : 2), cudaMemcpyHostToDevice));
            for (int layer = 0; layer < layers; ++layer) {
                routes[layer].resize(rows*topk); scale[layer].resize(rows*topk);
                auto &route = routes[layer]; auto &scoresHost = scale[layer];
                for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
                    const int r = row*topk+k;
                    // Alternate hot routes, eviction, repeated experts and zero/negative weights.
                    route[r] = (pass % 4 == 0 ? pass*7 : 0) + layer*3 + k*5 + row;
                    route[r] %= experts;
                    if (k == 1 && pass % 3 == 0) route[r] = route[row*topk];
                    scoresHost[r] = k == 0 ? 0.f : float((k+row)%5-1)/16;
                }
                Cuda(cudaMemcpy(ids[layer].cudaData, route.data(), route.size()*4, cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(scores[layer].cudaData, scoresHost.data(), scoresHost.size()*4, cudaMemcpyHostToDevice));
            }
            scope = FastllmCudaBeginMoeDecode(weights[0].data(),weights[0].size(),topk);
            Require(scope, "replay policy scope");
            Require(qwenTopk10 ? managed.Launch() :
                FastllmCudaLaunchNativeMoeFullGraph(pass%3 ? replay : exec,handles,layers,pass%3==0), "Standard full replay");
            FastllmCudaEndMoeDecode(scope);
            for (int layer = 0; layer < layers; ++layer) {
                actual[layer].resize(rows*hidden);
                Cuda(cudaMemcpy(actual[layer].data(),output[layer].cudaData,actual[layer].size()*4,cudaMemcpyDeviceToHost));
            }
            for (int layer = 0; layer < layers; ++layer) {
                // Standard NUMA and CUDA retain different activation rounding.
                // Build independent existing CPU/GPU expert references using
                // the actual preceding activation, then require each output row
                // to equal one valid mixture of those experts. Enumerating all
                // 2^topk placements also detects lost/duplicated contributions.
                const auto &activation = layer ? actual[0] : source;
                std::vector<int32_t> cpuRoutes(rows*topk,-1);
                std::vector<float> cpu(size_t(rows)*topk*hidden), gpu(cpu.size());
                NumasMoeDecodeExpertsBatch(activation.data(), cpu.data(), rows, weights[layer].data(),
                    weights[layer].size(), routes[layer].data(), cpuRoutes.data(), scale[layer].data(), topk, layer);
                Data referenceInput(FLOAT32, {rows,hidden}), referenceScores(FLOAT32,{rows,topk}), gate, result;
                Gpu(referenceInput); Gpu(referenceScores);
                Cuda(cudaMemcpy(referenceInput.cudaData,activation.data(),activation.size()*4,cudaMemcpyHostToDevice));
                for (int k=0;k<topk;++k) {
                    std::vector<float> one(rows*topk,0), values(rows*hidden);
                    for (int row=0;row<rows;++row) one[row*topk+k]=1;
                    Cuda(cudaMemcpy(referenceScores.cudaData,one.data(),one.size()*4,cudaMemcpyHostToDevice));
                    auto &oracle = native ? referenceWeights[layer] : weights[layer];
                    // The candidate is already captured; keep the eager GPU
                    // oracle on its original full-warp projection.
                    Require(FastllmCudaMergeMOECache(referenceInput,gate,result,oracle.data(),
                        oracle.size(),static_cast<int32_t*>(ids[layer].cudaData),
                        static_cast<float*>(referenceScores.cudaData),topk),"GPU expert oracle");
                    Cuda(cudaMemcpy(values.data(),result.cudaData,values.size()*4,cudaMemcpyDeviceToHost));
                    for (int row=0;row<rows;++row)
                        std::copy_n(values.data()+row*hidden,hidden,gpu.data()+(row*topk+k)*hidden);
                }
                for (int row=0;row<rows;++row) {
                    double best=1e100;
                    for (int mask=0;mask<(1<<topk);++mask) {
                        double err=0,norm=0;
                        for (int col=0;col<hidden;++col) {
                            float expected=0;
                            for (int k=0;k<topk;++k) {
                                const float value=(mask&(1<<k)?gpu:cpu)[(row*topk+k)*hidden+col];
                                if (rows>1) expected=std::fma(value,scale[layer][row*topk+k],expected);
                                else { volatile float product=value*scale[layer][row*topk+k]; expected+=product; }
                            }
                            const float value=actual[layer][row*hidden+col];
                            if (!std::isfinite(value)) std::fprintf(stderr,
                                "nonfinite kind=%d/%d dtype=%d rows=%d pass=%d layer=%d row=%d col=%d CPU=%g GPU=%g\n",
                                gateKind,downKind,dtype,rows,pass,layer,row,col,cpu[row*topk*hidden+col],gpu[row*topk*hidden+col]);
                            Require(std::isfinite(value),"nonfinite graph output");
                            const double delta=double(value)-expected; err+=delta*delta; norm+=double(expected)*expected;
                        }
                        best=std::min(best,std::sqrt(err/std::max(norm,1e-20)));
                    }
                    worstL2=std::max(worstL2,best);
                    if (best>1e-5) std::fprintf(stderr,"kind=%d/%d dtype=%d rows=%d pass=%d layer=%d row=%d best_mixture_L2=%g\n",gateKind,downKind,dtype,rows,pass,layer,row,best);
                    Require(best<=1e-5,"graph differs from every CPU/GPU expert mixture"); ++checks;
                }
            }
        }
        if (qwenTopk10) {
            managed.Reset();
            Require(!managed && !managed.Launch(), "managed graph reset");
        } else {
            FastllmCudaGraphExecDestroy(replay); FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
            for (void *handle : handles) FastllmCudaReleaseNativeMoeFullGraph(handle);
        }
    }
    if (native) FastllmCudaReleaseMoeCache(referenceWeights[0].data(), referenceWeights[0].size());
    FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size()); ClearNumasMoeRuntimeCache();
    for (const auto &w : referenceOwned) ReleaseNumasMoeWeightArena(w.get());
    for (const auto &w : owned) ReleaseNumasMoeWeightArena(w.get());
    std::printf("PASS Standard mailbox gate=%d down=%d device=%d owned_pp=%d topk=%d checks=%d worst_mixture_L2=%.8g\n",
        gateKind,downKind,device,int(ownedPp),topk,checks,worstL2);
}
int main(int argc, char **argv) {
    try {
        if (!FastllmCudaGetDeviceCount()) return 77;
        SetThreads(4); SetCudaGraph(true);
        const bool qwenTopk10 = argc > 1 && std::strcmp(argv[1], "--qwen-topk10") == 0;
        const bool modelShape = qwenTopk10 || (argc > 1 && std::strcmp(argv[1], "--model-shape") == 0);
        const bool placementPolicy = argc > 1 && std::strcmp(argv[1], "--placement-policy") == 0;
        if (placementPolicy) setenv("FASTLLM_TEST_PLACEMENT_POLICY", "1", 1);
        const bool ownedPp = argc > 1 && std::strcmp(argv[1], "--owned-pp") == 0;
        Require(!ownedPp || FastllmCudaGetDeviceCount() >= 2, "owned PP test requires two GPUs");
        const int hidden = modelShape ? 2560 : 256, inter = modelShape ? 640 : 128;
        for (int device = 0; device < (ownedPp ? 2 : 1); ++device)
            for (auto pair : {std::pair<int,int>{-1,-1},{22,42},{42,42},{21,20},{18,20}})
                Run(pair.first,pair.second,hidden,inter,placementPolicy ? 1 : modelShape || ownedPp ? 4 : 40,ownedPp,device,qwenTopk10);
        std::puts("ALL_PASS"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
