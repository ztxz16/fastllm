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
#include "utils.h"
#include "utils/host_moe_prefill.h"
#include "devices/cpu/cpudevice.h"

using namespace fastllm;

namespace fastllm {
struct Qwen4GGUFTestAccess {
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

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("FASTLLM_TEST_SKIP_NO_DEVICE");
        return 0;
    }
    try {
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
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
