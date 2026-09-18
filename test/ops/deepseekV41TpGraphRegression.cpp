// Run with test/basic/test_deepseek_v41_tp_graph.py (two CUDA devices).
#include "model.h"
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::ForwardSegments;
    using DeepSeekV41Model::DsparkCommitPrefix;
    using DeepSeekV41Model::ApplyDsparkDevice;
    using DeepSeekV41Model::v41DsparkEnabled;
    using DeepSeekV41Model::v41DsparkTargetLayerIds;
    using DeepSeekV41Model::v41IsDsparkTarget;
    using DeepSeekV41Model::v41DsparkTpDevices;
    using DeepSeekV41Model::v41CudaGraphSlots;
};

static void AppendTensor(std::vector<float> &result, const Data &input) {
    if (input.dims.empty() || input.Count(0) == 0) return;
    Data cpu;
    if (input.multiDeviceData) {
        const Data &local = *input.multiDeviceDatas.at(0);
        const int original = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        cpu.CopyFrom(local);
        cpu.ToDevice(DataDevice::CPU);
        FastllmCudaSetDevice(original);
    } else {
        cpu.CopyFrom(input);
        cpu.ToDevice(DataDevice::CPU);
    }
    ToDataTypeForceCPU(cpu, DataType::FLOAT32);
    // The first dimension is one; cache capacity can exceed its logical rows.
    uint64_t count = 1;
    for (int dim : cpu.dims) count *= dim;
    result.push_back(count);
    result.insert(result.end(), (float *)cpu.cpuData, (float *)cpu.cpuData + count);
}

// Exercise every candidate count, fresh request state, ring wrap and rejected
// suffixes. Compare actual target features and committed KV, not only argmax.
static std::vector<float> RunDsparkRequests(DeepSeekV41Model &model) {
    model.*(&Access::v41DsparkEnabled) = true;
    model.*(&Access::v41DsparkTpDevices) = {0, 1};
    // Real requests enter DsparkAdvance after prefill before verifying drafts.
    // Reproduce its device setup without loading synthetic draft weights.
    (model.*(&Access::ApplyDsparkDevice))();
    auto &targets = model.*(&Access::v41DsparkTargetLayerIds);
    targets = {model.block_cnt - 3, model.block_cnt - 2, model.block_cnt - 1};
    auto &mask = model.*(&Access::v41IsDsparkTarget);
    mask.assign(model.block_cnt, false);
    for (int layer : targets) mask[layer] = true;
    std::vector<float> result;
    for (int length : {7, 17, 33, 7}) {
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(model.block_cnt);
        std::vector<Data> dummy(model.block_cnt * 2);
        std::vector<std::pair<Data *, Data *>> past;
        for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i + 1]);
        for (int step = 0; step <= 36; ++step) {
            DeepSeekV41SpecScratch scratch;
            scratch.captureMain = true;
            DeepSeekV41Segment seg;
            seg.state = state;
            seg.startPos = state->totalLen;
            seg.seqlen = step == 0 ? length : 1 + (step - 1) % 6;
            seg.spec = &scratch;
            if (step > 0 && seg.seqlen > 1) {
                scratch.wantAllTokens = scratch.deferWindow = true;
                scratch.windowKV.resize(model.block_cnt);
                scratch.rawKV.resize(model.block_cnt);
                scratch.rawScore.resize(model.block_cnt);
                scratch.prevRawTail.resize(model.block_cnt);
                scratch.prevBlocks.resize(model.block_cnt);
            }
            std::vector<float> tokens(seg.seqlen);
            for (int i = 0; i < seg.seqlen; ++i) tokens[i] = (seg.startPos + i) * 17 % 123 + 3;
            Data ids(DataType::FLOAT32, {1, seg.seqlen}, tokens);
            GenerationConfig config;
            LastTokensManager last(1, 64);
            auto got = (model.*(&Access::ForwardSegments))({seg}, ids, nullptr, nullptr, {config}, last, nullptr, past);
            if (scratch.mainHidden.size() != 3) throw std::runtime_error("missing target features");
            result.insert(result.end(), got.begin(), got.end());
            result.insert(result.end(), scratch.tokens.begin(), scratch.tokens.end());
            for (const Data &hidden : scratch.mainHidden) AppendTensor(result, hidden);
            if (scratch.deferWindow) {
                const int committed = 1 + ((step - 1) / 6) % seg.seqlen;
                (model.*(&Access::DsparkCommitPrefix))(*state, scratch, seg.startPos, committed, seg.seqlen);
            }
            result.push_back(state->totalLen);
            for (const auto &cache : state->layers) {
                for (const Data *data : {&cache.windowKV, &cache.compressedKV, &cache.indexK,
                                         &cache.rawTailKV, &cache.rawTailScore}) AppendTensor(result, *data);
            }
        }
    }
    return result;
}

static std::vector<float> RunRequests(DeepSeekV41Model &model, const std::vector<int> &lengths) {
    std::vector<float> result;
    for (int length : lengths) {
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(model.block_cnt);
        std::vector<Data> dummy(model.block_cnt * 2);
        std::vector<std::pair<Data *, Data *>> past;
        for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i + 1]);
        for (int step = 0; step <= 12; ++step) {
            DeepSeekV41Segment seg;
            seg.state = state; seg.startPos = state->totalLen;
            seg.seqlen = step == 0 ? length : 1; seg.offset = 0;
            std::vector<float> tokens;
            for (int i = 0; i < seg.seqlen; ++i) tokens.push_back((seg.startPos + i) * 17 % 123 + 3);
            Data ids(DataType::FLOAT32, {1, seg.seqlen}, tokens);
            GenerationConfig config; config.output_logits = true;
            LastTokensManager last(1, 64);
            std::vector<float> logits;
            std::vector<std::vector<float> *> outputs{&logits};
            (model.*(&Access::ForwardSegments))({seg}, ids, nullptr, nullptr, {config}, last, &outputs, past);
            if (logits.size() != 256) throw std::runtime_error("fixture vocabulary mismatch");
            result.insert(result.end(), logits.begin(), logits.end());
        }
    }
    return result;
}

static void Compare(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (actual.size() != expected.size()) throw std::runtime_error("missing request logits");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-5f) {
            std::cerr << "logit " << i << ": " << expected[i] << " vs " << actual[i] << '\n';
            throw std::runtime_error("TP decode differs from eager after warmup");
        }
    }
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
    try {
        const bool expertCache = argc == 3 && std::string(argv[2]) == "--expert-cache";
        if (argc != 2 && !expertCache)
            throw std::runtime_error("usage: deepseekV41TpGraphRegression FIXTURE_DIR [--expert-cache]");
        setenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP", "1", 1);
        SetCudaGraph(false);
        SetThreads(2);
        SetCudaSharedExpert(true);
        SetDeviceMap({{"multicuda:0,1", 1}});
        SetMoeDeviceMap({{expertCache ? "numa" : "cpu", 1}});
        if (expertCache) {
            SetMoeCudaCacheBytes(16ULL << 20);
            // Keep the split deterministic while testing graph replay and KV.
            setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "2", 1);
            setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH", "0", 1);
        }
        auto base = CreateLLMModelFromHF(argv[1], DataType::FLOAT16);
        auto &model = dynamic_cast<DeepSeekV41Model &>(*base);
        // Initialize weight shards and kernel plans before comparing dispatch.
        RunRequests(model, {7});
        const auto expected = RunRequests(model, {7, 17, 33, 7});
        unsetenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP");
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(true);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(false);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        SetCudaGraph(true);
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        std::cout << "PASS: eager/graph switching and shared overlap match across 4 requests, 52 steps each\n";
        SetCudaGraph(false);
        const auto speculative = RunDsparkRequests(model);
        SetCudaGraph(true);
        Compare(speculative, RunDsparkRequests(model));
        if ((model.*(&Access::v41CudaGraphSlots)).size() != 6)
            throw std::runtime_error("missing DSpark graph shapes");
        SetCudaGraph(false);
        Compare(speculative, RunDsparkRequests(model));
        SetCudaGraph(true);
        Compare(speculative, RunDsparkRequests(model));
        std::cout << "PASS: DSpark graph shapes 1..6, main features and rollback match across requests\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
