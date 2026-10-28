#include "fastllm.h"
#include "models/mimo_tool_grammar.h"
#include "models/speculative_sampling.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        FastllmCudaSetDevice(0);
        // Real-vocabulary mask boundaries and row isolation, CPU versus CUDA.
        const int vocab = 152576;
        std::vector<float> input(3 * vocab, 7.0f);
        Data cpu(FLOAT32, {3, vocab}, input), gpu(FLOAT32, {3, vocab}, input);
        auto mask = std::make_shared<std::vector<uint32_t>>((vocab + 31) / 32, 0);
        for (int id : {0, 31, 32, vocab - 1}) (*mask)[id / 32] |= uint32_t(1) << (id % 32);
        GenerationConfig c; c.tool_call_token_mask = mask;
        ApplyToolTokenMask(cpu, 1, c);
        gpu.ToDevice(DataDevice::CUDA, std::vector<int>{0}); ApplyToolTokenMask(gpu, 1, c);
        gpu.ToDevice(DataDevice::CPU);
        for (int i = 0; i < 3 * vocab; ++i)
            if (((float *)cpu.cpuData)[i] != ((float *)gpu.cpuData)[i]) {
                std::cerr << "MASK_MISMATCH i=" << i << " cpu=" << ((float *)cpu.cpuData)[i]
                          << " gpu=" << ((float *)gpu.cpuData)[i] << '\n';
                Check(false, "CUDA mask differs from CPU");
            }

        const int batch = 10000, v = 64;
        MimoToolGrammar grammar({{"A", {}}, {"B", {}}});
        auto tokens = std::make_shared<std::vector<std::string>>(v, "bad");
        (*tokens)[0] = "A"; (*tokens)[1] = "B"; (*tokens)[2] = ">";
        grammar.Bind(tokens, {63});
        int state = grammar.Advance(0, "<tool_call><function=");
        GenerationConfig cfg; cfg.top_k = v; cfg.top_p = 1; cfg.temperature = 1;
        cfg.tool_call_token_mask = grammar.Mask(state);
        std::vector<float> raw(batch * v, 100.0f);
        for (int b = 0; b < batch; ++b) { raw[b*v] = std::log(.8f); raw[b*v+1] = std::log(.2f); }
        Data draft(FLOAT32, {batch, v}, raw);
        for (int b = 0; b < batch; ++b) ApplyToolTokenMask(draft, b, cfg);
        draft.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        Data q(FLOAT32, {batch, v}); q.ToDevice(DataDevice::CUDA, {0}, false); q.Allocate(false);
        std::vector<float> temperatures(batch * 2, 1), topPs(batch * 2, 1);
        std::vector<int> topKs(batch * 2, v), proposals(batch), sampled(batch * 2), accepted(batch);
        Check(FastllmCudaMtpSampleDraft((float *)draft.cudaData, (float *)q.cudaData,
            temperatures.data(), topKs.data(), topPs.data(), proposals.data(), batch, v), "GPU proposal");
        q.ToDevice(DataDevice::CPU);
        for (int b = 0; b < batch; ++b) {
            Check(proposals[b] == 0 || proposals[b] == 1, "illegal proposal");
            const float *row = (float *)q.cpuData + b*v;
            Check(std::abs(row[0] - .8f) < 1e-6f && std::abs(row[1] - .2f) < 1e-6f, "masked q law");
            for (int i = 2; i < v; ++i) Check(row[i] == 0, "q leaks through mask");
        }
        q.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        std::vector<float> pRaw(batch * 2 * v, 100.0f);
        for (int b = 0; b < batch; ++b) {
            pRaw[b*2*v] = std::log(.2f); pRaw[b*2*v+1] = std::log(.8f);
            pRaw[(b*2+1)*v+2] = 0;
        }
        Data p(FLOAT32, {batch, 2, v}, pRaw);
        for (int b = 0; b < batch; ++b) {
            ApplyToolTokenMask(p, b*2, cfg);
            GenerationConfig next = cfg;
            next.tool_call_token_mask = grammar.Mask(grammar.AdvanceToken(state, proposals[b]));
            ApplyToolTokenMask(p, b*2+1, next);
        }
        p.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        Check(FastllmCudaMtpRejectionSampling((float *)p.cudaData, (float *)q.cudaData,
            temperatures.data(), topKs.data(), topPs.data(), proposals.data(), sampled.data(),
            accepted.data(), batch, 1, v), "GPU rejection");
        int a = 0, accepts = 0;
        for (int b = 0; b < batch; ++b) {
            Check(sampled[b*2] == 0 || sampled[b*2] == 1, "illegal correction or accepted token");
            a += sampled[b*2] == 0; accepts += accepted[b];
            if (accepted[b]) Check(sampled[b*2+1] == 2, "bonus uses wrong grammar state");
        }
        Check(std::abs(a / double(batch) - .2) < .025, "rejection does not reproduce masked p");
        Check(std::abs(accepts / double(batch) - .4) < .025, "unexpected acceptance law");
        // Finite top-k/repetition CPU fallback must filter before top-k as well.
        cfg.top_k = 1; cfg.repeat_penalty = 1.1f;
        Data fallback(FLOAT32, {1, v}, std::vector<float>(raw.begin(), raw.begin()+v));
        Check(LLMSampling(fallback, 0, cfg, LastTokensUnit()) == 0, "CPU constrained fallback");
        auto dist = SpeculativeDistribution(raw.data(), v, cfg, LastTokensUnit());
        Check(dist[0] == 1 && dist[2] == 0, "CPU speculative constrained fallback");
        std::cout << "ALL_PASS MiMo sampling p(A)=" << a/double(batch)
                  << " acceptance=" << accepts/double(batch) << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
