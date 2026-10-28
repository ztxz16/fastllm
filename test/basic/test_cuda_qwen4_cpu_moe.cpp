#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "gguf.h"
#include "models/qwen4_exp.h"
#include "../../src/models/qwen4_exp_graph.h"
#include "../../src/models/qwen4_exp_execution.h"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include "utils.h"
#include "utils/host_moe_prefill.h"
#include "utils/host_moe_decode.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "devices/cpu/cpudevice.h"

using namespace fastllm;

namespace fastllm {
struct Qwen4GGUFTestAccess {
    static void TestDenseBoundaryGraphs();
    static std::pair<bool, bool> GraphPolicy(std::vector<Data *> &hostWeights,
                                            int sequence, bool registeredHost = true,
                                            bool cpuPlacement = true) {
        Qwen4ExpModel model;
        Data residentExpert;
        // Distinct identities model a CUDA prefix, followed by a registered
        // host table. The policy must not infer all layers from weights[0].
        model.weights = {{nullptr, nullptr, &residentExpert, &residentExpert}};
        if (registeredHost) model.weights.push_back(hostWeights);
        model.moeDeviceMap = cpuPlacement
            ? std::map<std::string, int>{{"cuda:0", 1}, {"cpu", 1}}
            : std::map<std::string, int>{{"cuda:0", 1}};
        Data hidden(FLOAT32, {1, sequence, 256});
        const auto policy = model.HybridDecodeGraphPolicy(hidden, false);
        // The test borrows the registration owned by TestDecodeCache.
        model.weights.clear();
        return {policy.hybridDense, policy.eligible};
    }

    static void Run(Qwen4ExpModel &model, const Data &input, Data &output,
                    std::vector<Data *> &weights, std::vector<Data *> &bias) {
        Qwen4ExpModel::MoEExecutionPlan plan;
        plan.mlpPrefix = "model.language_model.layers.0.mlp.";
        plan.weights = &weights;
        plan.bias = &bias;
        model.RunHybridMoE(plan, input, output, false, nullptr, nullptr, {});
    }
};
}

static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void Weight(Data &weight, int rows, int cols, unsigned seed,
                   int type=GGML_TYPE_Q2_0, bool repack=false) {
    weight.dataType = DATA_GGUF_FORMAT;
    weight.ggmlType = type;
    weight.isGGUFData = true;
    weight.isModelWeight = true;
    weight.disableGGUFRepack = !repack;
    weight.Resize({rows, cols});
    weight.Allocate();
    std::mt19937 rng(seed);
    for(size_t i=0;i<weight.GetBytes();++i) weight.cpuData[i]=static_cast<uint8_t>(rng());
    if(type==GGML_TYPE_Q2_0) {
        auto *blocks=reinterpret_cast<block_q2_0 *>(weight.cpuData);
        for(int i=0;i<rows*cols/QK2_0;++i) blocks[i].d=float_to_half(.02f);
    } else if(type==GGML_TYPE_IQ2_S) {
        auto *blocks=reinterpret_cast<block_iq2_s *>(weight.cpuData);
        for(int i=0;i<rows*cols/QK_K;++i) blocks[i].d=float_to_half(.002f);
    } else if(type==GGML_TYPE_IQ2_XXS) {
        auto *blocks=reinterpret_cast<block_iq2_xxs *>(weight.cpuData);
        for(int i=0;i<rows*cols/QK_K;++i) blocks[i].d=float_to_half(.002f);
    } else if(type==GGML_TYPE_IQ1_M) {
        auto *blocks=reinterpret_cast<block_iq1_m *>(weight.cpuData);
        const auto scale=float_to_half(.002f);
        for(int i=0;i<rows*cols/QK_K;++i) {
            auto *scales=reinterpret_cast<uint16_t *>(blocks[i].scales);
            for(int n=0;n<4;++n) scales[n]=(scales[n]&0xfff)|(((scale>>(4*n))&15)<<12);
        }
    }
}

static std::vector<float> Read(Data &data) {
    Data copy;
    copy.CopyFrom(data);
    copy.ToDevice(DataDevice::CPU);
    if(copy.dataType!=FLOAT32) ToDataTypeForceCPU(copy,FLOAT32);
    const float *values = reinterpret_cast<float *>(copy.cpuData);
    return std::vector<float>(values, values + copy.Count(0));
}

static void Test(int tokens) {
    const int hidden = 256;
    Qwen4ExpModel model;
    model.block_cnt = 1;
    model.num_experts = 1;
    model.num_experts_per_tok = 1;
    model.norm_topk_prob = true;
    model.moeDeviceMap = {{"cpu", 1}};
    model.deviceMap = {{"cpu", 1}};
    const std::string prefix = "model.language_model.layers.0.mlp.";
    auto zero = [&](const std::string &name, int rows, int cols) {
        Data data(FLOAT32, {rows, cols}, std::vector<float>(rows * cols, 0));
        model.weight[prefix + name].CopyFrom(data);
    };
    zero("gate.weight", 1, hidden);
    zero("shared_expert.gateup_proj.weight", 2 * hidden, hidden);
    zero("shared_expert.down_proj.weight", hidden, hidden);
    zero("shared_expert_gate.weight", 1, hidden);
    Data gateup, down;
    Weight(gateup, 2 * hidden, hidden, 71);
    Weight(down, hidden, hidden, 72);
    std::vector<Data *> weights = {nullptr, nullptr, &gateup, &down};
    std::vector<Data *> bias(4, nullptr);
    std::vector<float> values(tokens * hidden);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * .17f);
    Data cpuInput(FLOAT32, {1, tokens, hidden}, values), reference;
    ((Executor *)GetExecutor())->SetFirstDevice("cpu");
    Qwen4GGUFTestAccess::Run(model, cpuInput, reference, weights, bias);
    auto expected = Read(reference);

    model.deviceMap = {{"cuda:0", 1}};
    FastllmCudaSetDevice(0);
    ((Executor *)GetExecutor())->SetFirstDevice("cuda:0");
    Data gpuInput(FLOAT32, {1, tokens, hidden}, values), result;
    gpuInput.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    Qwen4GGUFTestAccess::Run(model, gpuInput, result, weights, bias);
    auto actual = Read(result);
    Check(actual.size() == expected.size(), "output size changed");
    double error = 0, magnitude = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        Check(std::isfinite(actual[i]), "non-finite output");
        error += double(actual[i] - expected[i]) * (actual[i] - expected[i]);
        magnitude += double(expected[i]) * expected[i];
    }
    const double relative = std::sqrt(error / std::max(magnitude, 1e-30));
    Check(magnitude > 0, "test did not exercise routed experts");
    Check(relative < .002, "GPU-input CPU experts differ from CPU reference");
    Check(gpuInput.dataDevice == DataDevice::CUDA && Read(gpuInput) == values,
          "borrowed input was modified");
    std::printf("PASS CPU resident Q2_0 experts tokens=%d relative_rms=%.8g\n", tokens, relative);
}

struct CpuReferenceMoe : CpuMergeMOE { using CpuMergeMOE::Run; };
static void SetEnv(const char *key, const char *value) {
#ifdef _WIN32
    _putenv_s(key,value?value:"");
#else
    if(value) setenv(key,value,1); else unsetenv(key);
#endif
}
static void TestSplit(int type=GGML_TYPE_Q2_0, bool repack=false, DataType dtype=FLOAT32) {
    constexpr int rows=64, hidden=256, experts=4, topk=2;
    Data storage[experts*2];
    std::vector<Data *> weights(2*(experts+1),nullptr),bias(weights.size(),nullptr);
    for(int e=0;e<experts;++e) {
        Weight(storage[2*e],2*hidden,hidden,101+2*e,type,repack);
        Weight(storage[2*e+1],hidden,hidden,102+2*e,type,repack);
        weights[2*(e+1)]=&storage[2*e];weights[2*(e+1)+1]=&storage[2*e+1];
    }
    std::vector<float> values(rows*hidden),scores(rows*topk);
    for(size_t i=0;i<values.size();++i) values[i]=std::sin(float(i)*.071f)*.1f;
    Data x(dtype,{rows,hidden},values),index(fastllm::INT32,{rows,topk}),score(FLOAT32,{rows,topk});
    index.Allocate();score.Allocate();
    auto *ids=reinterpret_cast<int32_t *>(index.cpuData);
    auto *prob=reinterpret_cast<float *>(score.cpuData);
    for(int r=0;r<rows;++r) {ids[2*r]=0;ids[2*r+1]=r<32?1:(r<48?2:3);prob[2*r]=.6f;prob[2*r+1]=.4f;}
    auto originalInput=Read(x);
    Data reference(dtype,{rows,hidden}),w1,w2,w3,tmpIn,tmpOut;
    CpuReferenceMoe cpu;
    cpu.Run("MergeMOE",{{"input",&x},{"output",&reference},{"index",&index},{"score",&score},
        {"weights",reinterpret_cast<Data *>(weights.data())},{"biass",reinterpret_cast<Data *>(bias.data())},
        {"w1",&w1},{"w2",&w2},{"w3",&w3},{"temporaryInput",&tmpIn},{"temporaryOutput",&tmpOut}},
        {},{{"weights___batch",int(weights.size())},{"biass___batch",int(bias.size())}});
    auto expected=Read(reference);
    x.ToDevice(CUDA,std::vector<int>{0});index.ToDevice(CUDA,std::vector<int>{0});score.ToDevice(CUDA,std::vector<int>{0});
    SetEnv("FT_EXPERT_LIMIT","32");SetEnv("FT_GPU_PREFILL","1");
    HostMoePrefillStats stats;Data result;
    Check(RunHostMoePrefill(x,index,score,result,weights.data(),int(weights.size()),0,&stats),"split rejected valid Q2_0 prefill");
    Check(stats.cpuExperts==2 && stats.gpuExperts==2 && stats.cpuRoutes==32 && stats.gpuRoutes==96,"split lost or duplicated routes");
    auto actual=Read(result);double err=0,mag=0;
    for(size_t i=0;i<actual.size();++i) {Check(std::isfinite(actual[i]),"split non-finite result");err+=std::pow(double(actual[i]-expected[i]),2);mag+=std::pow(double(expected[i]),2);}
    double relative=std::sqrt(err/std::max(mag,1e-30));
    std::printf("SPLIT type=%d repack=%d dtype=%d relative_rms=%.8g cpu_routes=%d gpu_routes=%d\n",type,int(repack),int(dtype),relative,stats.cpuRoutes,stats.gpuRoutes);
    Check(relative<.03,"CPU/GPU split differs from CPU reference");
    Check(Read(x)==originalInput,"split modified borrowed CUDA input");
    SetEnv("FT_GPU_PREFILL","0");Data untouched(FLOAT32,{1},std::vector<float>{37});
    Check(!RunHostMoePrefill(x,index,score,untouched,weights.data(),int(weights.size()),0),"disable flag ignored");
    Check(Read(untouched)==std::vector<float>{37},"rejected split changed output");
    SetEnv("FT_EXPERT_LIMIT",nullptr);SetEnv("FT_GPU_PREFILL",nullptr);
}

static void TestDecodeCache(int type,bool repack,bool resident) {
    constexpr int hidden=256,experts=32,topk=4;
    Data storage[experts*2];
    std::vector<Data *> weights(2*(experts+1),nullptr),bias(weights.size(),nullptr);
    for(int e=0;e<experts;++e) {
        Weight(storage[2*e],hidden*2,hidden,400+2*e,type,repack);
        Weight(storage[2*e+1],hidden,hidden,401+2*e,type,repack);
        weights[2*e+2]=&storage[2*e];weights[2*e+3]=&storage[2*e+1];
    }
    const uint64_t record=(storage[0].GetBytes()+storage[1].GetBytes()+127)/128*128;
    SetMoeCudaCacheBytes(resident?record*16:0);
    FastllmCudaMoeCacheLayer layer{weights.data(),int(weights.size())};
    Check(FastllmCudaPrepareMoeCache(&layer,1,{},true),"ordinary cache prepare failed");
    struct Release {std::vector<Data *> &w;~Release(){FastllmCudaReleaseMoeCache(w.data(),int(w.size()));SetMoeCudaCacheBytes(0);}} release{weights};
    std::vector<float> values(hidden);for(int i=0;i<hidden;++i) values[i]=std::sin(i*.09f)*.1f;
    Data x(FLOAT32,{1,hidden},values),index(fastllm::INT32,{1,topk}),score(FLOAT32,{1,topk});
    index.Allocate();score.Allocate();
    for(int j=0;j<topk;++j) {reinterpret_cast<int32_t *>(index.cpuData)[j]=j;reinterpret_cast<float *>(score.cpuData)[j]=.1f*(j+1);}
    Data reference(FLOAT32,{1,hidden}),w1,w2,w3,tmpIn,tmpOut;
    CpuReferenceMoe cpu;
    cpu.Run("MergeMOE",{{"input",&x},{"output",&reference},{"index",&index},{"score",&score},
        {"weights",reinterpret_cast<Data *>(weights.data())},{"biass",reinterpret_cast<Data *>(bias.data())},
        {"w1",&w1},{"w2",&w2},{"w3",&w3},{"temporaryInput",&tmpIn},{"temporaryOutput",&tmpOut}},
        {},{{"weights___batch",int(weights.size())},{"biass___batch",int(bias.size())}});
    const auto expected=Read(reference);
    x.ToDevice(CUDA,std::vector<int>{0});index.ToDevice(CUDA,std::vector<int>{0});score.ToDevice(CUDA,std::vector<int>{0});
    uint64_t before[8]{},after[8]{};fastllm_moe_cuda_cache_route_stats(0,before);
    double maxError=0;
    for(int step=0;step<64;++step) {
        void *policy=FastllmCudaBeginMoeDecode(weights.data(),int(weights.size()),topk);
        Check(policy!=nullptr,"host decode policy unavailable");
        if (step == 0) {
            const auto graph = Qwen4GGUFTestAccess::GraphPolicy(weights, 1);
            Check(graph.first && graph.second,
                  "CUDA prefix / ordinary CPU suffix did not select dense graphs");
            const auto unsupported = Qwen4GGUFTestAccess::GraphPolicy(weights, 1, false);
            Check(!unsupported.first && !unsupported.second,
                  "unregistered CPU backend was admitted to full graph capture");
            const auto prefill = Qwen4GGUFTestAccess::GraphPolicy(
                weights, FASTLLM_CUDA_MOE_CACHE_MAX_BATCH + 1);
            Check(!prefill.first && !prefill.second,
                  "prefill was admitted to a decode graph");
            const auto gpu = Qwen4GGUFTestAccess::GraphPolicy(weights, 1, false, false);
            Check(!gpu.first && gpu.second, "CUDA-only graph policy regressed");
        }
        Data output;
        Check(FastllmCudaMergeMOEHybrid(x,index,score,output,weights.data(),int(weights.size()),0),"host dynamic decode rejected");
        FastllmCudaEndMoeDecode(policy);
        auto actual=Read(output);double err=0,mag=0;
        for(size_t j=0;j<actual.size();++j) {Check(std::isfinite(actual[j]),"decode nonfinite");err+=std::pow(double(actual[j]-expected[j]),2);mag+=std::pow(double(expected[j]),2);}
        maxError=std::max(maxError,std::sqrt(err/std::max(mag,1e-30)));
    }
    Check(fastllm_moe_cuda_cache_route_stats(0,after),"decode route statistics unavailable");
    Check(after[0]-before[0]==64 && after[1]-before[1]==64*topk,"decode route conservation failed");
    Check(after[4]>before[4] && after[5]>before[5],"decode did not exercise both CPU and GPU");
    Check(after[4]-before[4]+after[5]-before[5]==64*topk,"decode duplicate or missing route");
    if(resident) Check(after[2]>before[2],"cache never hit");
    Check(maxError<.04,"decode split/cache differs from CPU reference");
    if(resident) {
        constexpr int rows=64;
        std::vector<float> prompt(rows*hidden);
        for(int r=0;r<rows;++r) std::copy(values.begin(),values.end(),prompt.begin()+size_t(r)*hidden);
        Data px(FLOAT32,{rows,hidden},prompt),pi(fastllm::INT32,{rows,topk}),ps(FLOAT32,{rows,topk}),po;
        pi.Allocate();ps.Allocate();
        for(int r=0;r<rows;++r) for(int j=0;j<topk;++j) {
            reinterpret_cast<int32_t *>(pi.cpuData)[r*topk+j]=j;
            reinterpret_cast<float *>(ps.cpuData)[r*topk+j]=.1f*(j+1);
        }
        px.ToDevice(CUDA,std::vector<int>{0});pi.ToDevice(CUDA,std::vector<int>{0});ps.ToDevice(CUDA,std::vector<int>{0});
        SetEnv("FT_EXPERT_LIMIT","1");HostMoePrefillStats prefillStats;
        Check(RunHostMoePrefill(px,pi,ps,po,weights.data(),int(weights.size()),0,&prefillStats),"warm decode cache rejected prefill");
        SetEnv("FT_EXPERT_LIMIT",nullptr);
        const auto actual=Read(po);double error=0,magnitude=0;
        for(size_t j=0;j<actual.size();++j) {
            Check(std::isfinite(actual[j]),"warm cache prefill nonfinite");
            error+=std::pow(double(actual[j]-expected[j%hidden]),2);
            magnitude+=std::pow(double(expected[j%hidden]),2);
        }
        const double relative=std::sqrt(error/std::max(magnitude,1e-30));
        Check(relative<.04,"warm decode cache changed prefill arithmetic");
        std::printf("CACHE_PREFILL type=%d relative_rms=%.8g\n",type,relative);
    }
    SetEnv("FASTLLM_HOST_MOE_DECODE","0");
    Check(!FastllmCudaCanRunMoeCacheSmallBatch(x,index,score,weights.data(),int(weights.size()),MoeGateSwiglu),"decode disable ignored");
    SetEnv("FASTLLM_HOST_MOE_DECODE",nullptr);
    std::printf("DECODE type=%d repack=%d resident=%d max_relative_rms=%.8g hits=%llu gpu=%llu cpu=%llu\n",
        type,int(repack),int(resident),maxError,(unsigned long long)(after[2]-before[2]),(unsigned long long)(after[4]-before[4]),(unsigned long long)(after[5]-before[5]));
}

static void TestDecodeWorkerRecovery() {
    constexpr int hidden=256;
    Data gate,down;
    Weight(gate,2*hidden,hidden,901);
    Weight(down,hidden,hidden,902);
    Data *weights[]={nullptr,nullptr,&gate,&down};
    PrepareHostMoeDecodeWeights(weights,4);
    std::vector<float> input(hidden),output(hidden);
    for(int i=0;i<hidden;++i) input[i]=std::sin(i*.13f)*.1f;
    const int32_t id=0,cpu=-1;
    double elapsed=0;
    bool caught=false;
    try {
        HostMoeDecodeExperts(input.data(),output.data(),weights,&id,&cpu,1,
            [] {throw std::runtime_error("expected GPU submission error");},&elapsed);
    } catch(const std::runtime_error &e) {
        caught=std::strcmp(e.what(),"expected GPU submission error")==0;
    }
    Check(caught && elapsed>0,"GPU submission error did not join CPU worker");
    const auto expected=output;
    for(int step=0;step<8;++step) {
        std::fill(output.begin(),output.end(),0);
        HostMoeDecodeExperts(input.data(),output.data(),weights,&id,&cpu,1,[]{},&elapsed);
        Check(output==expected && elapsed>0,"CPU decode worker failed after submission error");
    }
    std::puts("PASS decode worker recovery and reuse");
}

// Exercise the actual prefix/suffix at the model's decode dimensions. The
// intervening attention output deliberately changes on every replay; attention
// itself stays eager and is not replaced or approximated by this change.
void fastllm::Qwen4GGUFTestAccess::TestDenseBoundaryGraphs() {
    constexpr int hidden=2560, hc=4, lowRank=320;
    Qwen4ExpModel model;
    model.block_cnt=2; model.embed_dim=hidden; model.hcCount=hc;
    model.num_experts=512; model.num_experts_per_tok=10; model.norm_topk_prob=true;
    model.deviceMap={{"cuda:0",1}};
    FastllmCudaSetDevice(0);
    ((Executor *)GetExecutor())->SetFirstDevice("cuda:0");
    auto weight=[&](const std::string &name, std::vector<int> dims, bool norm, int seed) {
        int count=1; for(int d:dims) count*=d;
        std::vector<float> v(count);
        for(int i=0;i<count;++i) v[i]=norm?1.0f:std::sin(float((i+seed)%1021)*.031f)*.004f;
        Data source(norm?FLOAT32:FLOAT16,dims,v);
        model.weight[name].CopyFrom(source);
        model.weight[name].ToDevice(CUDA,std::vector<int>{0});
    };
    for(int layer=0;layer<2;++layer) {
        const auto prefix=model.languagePrefix+"layers."+std::to_string(layer)+".";
        for(const auto &branch:{"attn_hyper_connection.","mlp_hyper_connection."}) {
            const auto path=prefix+branch;
            weight(path+"hc_norm.weight",{hc*hidden},true,0);
            weight(path+"input_mix_weight_down.weight",{lowRank,hc*hidden},false,13+layer);
            weight(path+"block_inject_weight.weight",{hc,hc*hidden},false,23+layer);
            weight(path+"input_mix_weight_up.weight",{hc*hidden,lowRank},false,37+layer);
        }
        weight(prefix+"mlp.gate.weight",{512,hidden},false,51+layer);
    }
    auto fill=[](Data &destination,int width,int seed) {
        std::vector<float> v(width);
        for(int i=0;i<width;++i) v[i]=std::sin(float(i+seed)*.017f)*.1f;
        Data source(FLOAT32,{1,1,width},v);source.ToDevice(CUDA,std::vector<int>{0});
        Check(Qwen4PrepareDecodeGraphTensor(destination,source,0),"boundary input copy failed");
    };
    auto bytes=[](Data &data) {
        ForceDeviceSync();Data copy;copy.CopyFrom(data);copy.ToDevice(CPU);
        return std::vector<uint8_t>(copy.cpuData,copy.cpuData+copy.GetBytes());
    };
    int replayCount=0;
    for(int variant=0;variant<5;++variant) {
        const int layer=variant==0?0:1, mask=variant==0?0:variant-1;
        Qwen4ExpModel::DecodeCudaGraphState eager, captured;
        eager.device=captured.device=0;
        Data peers[2],empty;
        std::vector<std::pair<Data,Data>> kv(2);
        Qwen4ExpModel::RequestState request;
        auto initialize=[&](Qwen4ExpModel::DecodeCudaGraphState &state,Data &peer,int step) {
            fill(state.hiddenStates[0],hc*hidden,step*97);
            fill(state.hiddenStates[1],hc*hidden,step*101);
            auto &previous=state.hybridBoundaries[0];
            fill(previous.output,hidden,step*103);
            fill(previous.injection,hc,step*107);
            fill(previous.moe.result.sharedOutput,hidden,step*109);
            fill(peer,hidden,step*113);
            previous.moe.result.sharedDeferred=(mask&1)!=0;
            previous.moe.result.peerOutput=(mask&2)?&peer:nullptr;
        };
        auto execution=[&](Qwen4ExpModel::DecodeCudaGraphState &state) {
            return Qwen4ExpModel::DecodeGraphExecution{model,&state,0,2,1,0,100,
                0,0,false,false,false,true,true,false,true,true,
                empty,empty,kv,request,nullptr,{}, {}};
        };
        auto a=execution(eager), b=execution(captured);
        auto &ea=eager.hybridBoundaries[layer];auto &cb=captured.hybridBoundaries[layer];
        auto prefix=[&] {b.RunHybridDensePrefix(layer,cb.attentionInput,cb.attentionInjection);};
        auto suffix=[&] {b.RunHybridDenseSuffix(layer,cb.attentionOutput,cb.attentionInjection);};
        auto capture=[&](int key,const std::function<void()> &body) {
            auto &segment=captured.segments[key];
            {FastllmCudaAllocationRecorder recorder(&segment.allocationRequests);body();}
            Check(FastllmCudaGraphPrepareCaptureDevice(),"prepare capture failed");ForceDeviceSync();
            Check(FastllmCudaReserveGraphWorkspace(segment.allocationRequests),"reserve capture workspace failed");
            Check(FastllmCudaGraphMemoryPoolBegin(),"capture pool failed");
            Check(FastllmCudaGraphBeginCapture(),"begin capture failed");body();
            Check(FastllmCudaGraphEndCapture(&segment.graph),"end capture failed");
            Check(FastllmCudaGraphMemoryPoolEnd(segment.reservedPointers),"end pool failed");
            Check(FastllmCudaGraphInstantiate(segment.graph,&segment.exec),"instantiate failed");
        };
        initialize(captured,peers[1],0);capture(0,prefix);
        Check(FastllmCudaGraphLaunch(captured.segments[0].exec),"initial prefix replay failed");
        fill(cb.attentionOutput,hidden,0);capture(1,suffix);
        for(int step=1;step<=8;++step) {
            initialize(eager,peers[0],step);initialize(captured,peers[1],step);
            a.RunHybridDensePrefix(layer,ea.attentionInput,ea.attentionInjection);
            Check(FastllmCudaGraphLaunch(captured.segments[0].exec),"prefix replay failed");
            Check(bytes(ea.attentionInput)==bytes(cb.attentionInput),"prefix replay changed input");
            Check(bytes(ea.attentionInjection)==bytes(cb.attentionInjection),"prefix replay changed injection");
            // Same stable bridge as production, refreshed after eager attention.
            const void *address=cb.attentionOutput.cudaData;
            fill(ea.attentionOutput,hidden,step*127);fill(cb.attentionOutput,hidden,step*127);
            Check(cb.attentionOutput.cudaData==address,"attention bridge address changed");
            a.RunHybridDenseSuffix(layer,ea.attentionOutput,ea.attentionInjection);
            Check(FastllmCudaGraphLaunch(captured.segments[1].exec),"suffix replay failed");
            Check(bytes(eager.hiddenStates[1])==bytes(captured.hiddenStates[1]),"suffix residual differs");
            Check(bytes(ea.input)==bytes(cb.input),"suffix input differs");
            Check(bytes(ea.injection)==bytes(cb.injection),"suffix injection differs");
            Check(bytes(ea.moe.index)==bytes(cb.moe.index),"suffix router indices differ");
            Check(bytes(ea.moe.score)==bytes(cb.moe.score),"suffix router scores differ");
            replayCount+=2;
        }
        ForceDeviceSync();
    }
    std::printf("PASS dense attention boundary graphs: %d replays, first layer + four join masks, bitwise equal\n",replayCount);
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("FASTLLM_TEST_SKIP_NO_DEVICE");
        return 0;
    }
    try {
        setvbuf(stdout,nullptr,_IONBF,0);
        SetThreads(3);
        Test(1);
        Test(4);
        TestSplit();
        TestSplit(GGML_TYPE_IQ2_S,true);
        TestSplit(GGML_TYPE_IQ2_XXS,true);
        TestSplit(GGML_TYPE_IQ1_M,false);
        TestSplit(GGML_TYPE_IQ2_S,true,BFLOAT16);
        TestSplit(GGML_TYPE_IQ1_M,false,BFLOAT16);
        TestSplit(GGML_TYPE_Q2_0,false,FLOAT16);
        TestDecodeCache(GGML_TYPE_Q2_0,false,false);
        TestDecodeCache(GGML_TYPE_Q2_0,false,true);
        TestDecodeCache(GGML_TYPE_IQ2_S,true,true);
        TestDecodeCache(GGML_TYPE_IQ2_XXS,true,true);
        TestDecodeCache(GGML_TYPE_IQ1_M,false,true);
        TestDecodeWorkerRecovery();
        Qwen4GGUFTestAccess::TestDenseBoundaryGraphs();
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
