// PP keeps dense/shared work on one stage; routed EP owns whole experts on
// three GPUs. Exercise loader placement and the actual model MoE boundary.
#include "../../src/models/deepseekv41_execution.h"
#include "gguf.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<float> Read(Data &value) {
    Data copy;
    copy.CopyFrom(value);
    copy.ToDevice(CPU);
    ToDataTypeForceCPU(copy, FLOAT32);
    const auto *p = reinterpret_cast<const float *>(copy.cpuData);
    return {p, p + copy.Count(0)};
}
class Probe : public DeepSeekV41Model {
public:
    using DeepSeekV41Model::RunGpuMoe;
    Probe() {
        block_cnt = 3;
        swiglu_limit = .8f;
        deviceMap = {{"cuda:0", 1}, {"cuda:1", 1}, {"cuda:2", 1}};
        moeDeviceMap = {{"multicuda:0,1,2", 1}};
        biass.resize(3, std::vector<Data *>(20, nullptr));
        for (int layer = 0; layer < 3; ++layer)
            quantizedLinearNames.insert("layers." + std::to_string(layer) + ".ffn.experts.0.w1.weight");
    }
};
static int Run(int gateType, int downType) {
    constexpr int hidden = 512, inter = 256, experts = 9, topk = 6;
    Probe model;
    std::mt19937 rng(319 + gateType * 17 + downType);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> reference(2 * (experts + 1), nullptr), ep(reference);
    for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
        const int type = part ? downType : gateType;
        auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, type,
            std::vector<int>{part ? hidden : 2 * inter, part ? inter : hidden});
        w->disableGGUFRepack = true;
        w->Allocate(false);
        for (size_t i = 0; i < w->GetBytes(); ++i) w->cpuData[i] = rng() >> 24;
        for (size_t i = 0; i < w->GetBytes(); i += ggml_type_size(ggml_type(type))) {
            const uint16_t scale = float_to_half(.02f);
            if (type == GGML_TYPE_IQ1_M) {
                auto *s = reinterpret_cast<uint16_t *>(w->cpuData + i + 48);
                for (int j = 0; j < 4; ++j) s[j] = (s[j] & 0xfff) | ((scale >> (4*j) & 15) << 12);
            } else std::memcpy(w->cpuData + i, &scale, sizeof(scale));
        }
        auto distributed = std::make_unique<Data>();
        distributed->CopyFrom(*w);
        const auto name = "layers.0.ffn.experts." + std::to_string(expert) +
                          (part ? ".w2.weight" : ".gateup.weight");
        distributed->name = name;
        distributed->isModelWeight = true;
        distributed->disableGGUFRepack = true;
        model.AddSpecialWeight(name, part ? "linearColumn" : "linearSwiglu", 0);
        Check(model.MoveSpecialWeightToCudaIfNeeded(name, *distributed), "EP loader rejected weight");
        Check(distributed->multiDeviceData && !distributed->cudaData,
              "EP loader retained unsplit root payload");
        int owners = 0;
        for (const auto &entry : distributed->multiDeviceDatas) if (entry.second->cudaData) {
            Check(entry.first == expert % 3, "wrong EP owner with TP disabled");
            auto *local = entry.second;
            std::vector<uint8_t> actual(w->GetBytes());
            FastllmCudaSetDevice(entry.first);
            Check(cudaMemcpy(actual.data(), local->cudaData, actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
                  "EP readback failed");
            Check(std::memcmp(actual.data(), w->cpuData, actual.size()) == 0, "EP changed packed weight bytes");
            ++owners;
        }
        Check(owners == 1, "EP weight duplicated or missing");
        w->ToDevice(CUDA, {0}, true);
        reference[2*(expert+1)+part] = w.get(); ep[2*(expert+1)+part] = distributed.get();
        owned.push_back(std::move(w)); owned.push_back(std::move(distributed));
    }
    DeepSeekV41DecodeWorkspace ws;
    Data gate, activation, scratch, expected;
    const std::vector<int> noTp;
    int checks = 0;
    // Same workspace, changing PP roots, route ids and token counts, including
    // prefill followed by decode. One-hot routes require exact final bytes;
    // multi-route sums allow only the collective's FP32 reassociation error.
    for (int pass = 0; pass < 3; ++pass) for (int rows : {1, 2, 4, 8, 33, 1}) for (int stage : {0, 1, 2}) {
        std::vector<float> x(rows*hidden), scores(rows*topk), shared(rows*hidden);
        std::vector<int32_t> ids(rows*topk);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = std::sin(float(i + 7*stage + pass)*.031f);
            shared[i] = float(int(i % 17) - 8)/64;
        }
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
            ids[row*topk+k] = (row + 2*stage + k) % experts;
            // Repeated routes with two entirely inactive ranks must produce
            // zeros there, including after buffers held nonzero work.
            if (pass == 2) ids[row*topk+k] = stage + 3*(row%3);
            scores[row*topk+k] = pass ? .125f*(k-2) : (k == stage ? .75f : 0.f);
        }
        Data input(FLOAT32, {rows, hidden}, x), index(INT32, {rows, topk}, CPU, ids.data());
        Data score(FLOAT32, {rows, topk}, scores);
        ToDataTypeForceCPU(input, BFLOAT16);
        ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
        for (auto *v : {&input, &index, &score}) v->ToDevice(CUDA, {0}, true);
        Check(FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, scratch, expected,
            reference.data(), reference.size(), index, score, 0.f, .8f), "reference rejected fixture");
        auto routed = Read(expected);
        for (size_t i = 0; i < routed.size(); ++i) routed[i] += shared[i];
        Data rounded(FLOAT32, {rows, hidden}, routed);
        ToDataTypeForceCPU(rounded, BFLOAT16);
        const auto want = Read(rounded);
        ApplyDeviceMap(model.deviceMap, stage + 1, 3);
        for (auto *v : {&input, &index, &score}) v->ToDevice(CUDA, {stage}, true);
        if (rows <= 8) {
            std::vector<Data *> local(ep.size(), nullptr);
            for (size_t i = 2; i < ep.size(); ++i) {
                const auto it = ep[i]->multiDeviceDatas.find(stage);
                if (it != ep[i]->multiDeviceDatas.end()) local[i] = it->second;
            }
            Data g, a, s, plain, cg, ca, cs, compact;
            Check(FastllmCudaMergeMOEV41GGUFResident(input, g, a, s, plain,
                local.data(), local.size(), index, score, 0.f, .8f), "local reference rejected");
            Check(FastllmCudaMergeMOEV41GGUFResident(input, cg, ca, cs, compact,
                local.data(), local.size(), index, score, 0.f, .8f, true), "compact EP rejected");
            Check(Read(plain) == Read(compact), "compact EP changed FP32 expert result");
        }
        ws.ffnInput.CopyFrom(input); ws.expertIndex.CopyFrom(index); ws.expertScore.CopyFrom(score);
        ws.sharedExpertOut.CopyFrom(Data(BFLOAT16, {rows, hidden}, shared));
        ws.sharedExpertOut.ToDevice(CUDA, {stage}, true);
        DeepSeekV41MoeContext ctx{ws, ep, noTp, stage, rows,
            false, rows == 1, false, false, false, true, false, [] {}, [] {}};
        Check(!model.RunGpuMoe(ctx), "PP incorrectly selected TP post dispatch");
        if (rows <= 8) {
            for (auto *source : {&ws.ffnInput, &ws.expertIndex, &ws.expertScore})
                Check(source->dataDeviceIds == std::vector<int>{stage}, "EP moved PP input/routes to first rank");
        }
        // Queue another use before host readback: a CPU worker join alone
        // must not be mistaken for GPU completion or retire its buffers.
        if (rows <= 8) Check(!model.RunGpuMoe(ctx), "queued PP selected TP");
        Check(!ws.ffnOut.multiDeviceData && ws.ffnOut.dataType == BFLOAT16 &&
              ws.ffnOut.dataDeviceIds == std::vector<int>{stage}, "EP result did not return to PP stage");
        const auto actual = Read(ws.ffnOut);
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            Check(std::isfinite(actual[i]), "nonfinite PP+EP output");
            error += std::pow(double(actual[i]) - want[i], 2); norm += double(want[i])*want[i];
        }
        Check(pass ? std::sqrt(error/(norm+1e-30)) < .003 : actual == want,
              "PP+EP arithmetic or stale route/input regression");
        ++checks;
    }
    return checks;
}
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 3) return 77;
        for (const char *name : {"FASTLLM_TP", "FASTLLM_QWEN3_MOE_TP", "FASTLLM_QWEN3_THREAD_TP", "FASTLLM_STEP3P5_TP"})
            setenv(name, "off", 1);
        SetThreads(4);
        const int checks = Run(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS) +
                           Run(GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS) +
                           Run(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M);
        std::printf("PASS: V4.1 PP3 EP3 (%d cases)\n", checks);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
