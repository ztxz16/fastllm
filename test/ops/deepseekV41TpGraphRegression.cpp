// Run with test/basic/test_deepseek_v41_tp_graph.py (--tp-size 1/2/4/8).
#include "model.h"
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace fastllm;
static std::vector<int> tpDevices;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::ForwardSegments;
    using DeepSeekV41Model::DsparkCommitPrefix;
    using DeepSeekV41Model::ApplyDsparkDevice;
    using DeepSeekV41Model::v41DsparkEnabled;
    using DeepSeekV41Model::v41DsparkTargetLayerIds;
    using DeepSeekV41Model::v41IsDsparkTarget;
    using DeepSeekV41Model::v41DsparkTpDevices;
    using DeepSeekV41Model::v41CudaGraphSlots;
    using DeepSeekV41Model::v41TpVerifyWorkspace;
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
static std::vector<float> RunDsparkRequests(DeepSeekV41Model &model, int candidates = 8) {
    model.*(&Access::v41DsparkEnabled) = true;
    model.*(&Access::v41DsparkTpDevices) = tpDevices;
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
        for (int step = 0; step <= candidates * 6; ++step) {
            DeepSeekV41SpecScratch scratch;
            scratch.captureMain = true;
            DeepSeekV41Segment seg;
            seg.state = state;
            seg.startPos = state->totalLen;
            seg.seqlen = step == 0 ? length : 1 + (step - 1) % candidates;
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
                const int committed = 1 + ((step - 1) / candidates) % seg.seqlen;
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
            if (logits.size() != model.weight["head.weight"].dims[0])
                throw std::runtime_error("fixture vocabulary mismatch");
            result.insert(result.end(), logits.begin(), logits.end());
        }
    }
    return result;
}

static void Compare(const std::vector<float> &expected, const std::vector<float> &actual) {
    if (actual.size() != expected.size()) throw std::runtime_error("request result size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-5f) {
            std::cerr << "result value " << i << ": " << expected[i] << " vs " << actual[i] << '\n';
            throw std::runtime_error("TP decode differs from eager after warmup");
        }
    }
}

int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return 77;
    try {
        bool expertCache = false, fp8Dense = false, eagerOnly = false, gpuExperts = false;
        bool diskExperts = false, diskEngram = false;
        int tpSize = 2;
        std::string logitsOnly;
        if (argc < 2)
            throw std::runtime_error("usage: deepseekV41TpGraphRegression FIXTURE_DIR "
                "[--expert-cache | --gpu-experts | --disk-experts] [--disk-engram] [--fp8-dense] [--eager-only] "
                "[--tp-size N] [--logits-only FILE]");
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--expert-cache") expertCache = true;
            else if (option == "--gpu-experts") gpuExperts = true;
            else if (option == "--disk-experts") diskExperts = true;
            else if (option == "--disk-engram") diskEngram = true;
            else if (option == "--fp8-dense") fp8Dense = true;
            else if (option == "--eager-only") eagerOnly = true;
            else if (option == "--tp-size" && i + 1 < argc) tpSize = std::stoi(argv[++i]);
            else if (option == "--logits-only" && i + 1 < argc) logitsOnly = argv[++i];
            else throw std::runtime_error("unknown option: " + option);
        }
        if (int(expertCache) + int(gpuExperts) + int(diskExperts) > 1)
            throw std::runtime_error("expert backend options are mutually exclusive");
        if (tpSize != 1 && tpSize != 2 && tpSize != 4 && tpSize != 8)
            throw std::runtime_error("device count must be 1, 2, 4 or 8");
        if (devices < tpSize) return 77;
        std::string mainDevice = tpSize == 1 ? "cuda:0" : "multicuda:";
        for (int i = 0; i < tpSize; ++i) {
            if (tpSize > 1) tpDevices.push_back(i);
            if (tpSize > 1) mainDevice += (i ? "," : "") + std::to_string(i);
        }
        setenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP", "1", 1);
        SetCudaGraph(false);
        SetThreads(2);
        SetCudaSharedExpert(true);
        SetDeviceMap({{mainDevice, 1}});
        const std::string moeDevice = gpuExperts ? mainDevice :
            (expertCache ? "numa" : (diskExperts ? "disk" : "cpu"));
        SetMoeDeviceMap({{moeDevice, 1}});
        SetNgramDevice(diskEngram ? "disk" : "cpu");
        if (expertCache) {
            SetMoeCudaCacheBytes(16ULL << 20);
            // Keep the split deterministic while testing graph replay and KV.
            setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "2", 1);
            setenv("FASTLLM_DSV41_MOE_CACHE_PREFETCH", "0", 1);
        }
        auto base = CreateLLMModelFromHF(argv[1], fp8Dense ? DataType::DATA_AUTO_SOURCE : DataType::FLOAT16);
        auto &model = dynamic_cast<DeepSeekV41Model &>(*base);
        if (model.deviceMap != std::map<std::string, int>{{mainDevice, 1}})
            throw std::runtime_error("requested TP layout silently fell back");
        if (model.GetExecutionMode() != (gpuExperts ? ExecutionMode::Gpu : ExecutionMode::Hybrid)) {
            throw std::runtime_error("wrong execution backend selected");
        }
        // Initialize weight shards and kernel plans before comparing dispatch.
        RunRequests(model, {7});
        const auto expected = RunRequests(model, {7, 17, 33, 7});
        if (!logitsOnly.empty()) {
            std::ofstream file(logitsOnly, std::ios::binary);
            file.write(reinterpret_cast<const char*>(expected.data()), expected.size() * sizeof(float));
            if (!file) throw std::runtime_error("failed to write logits");
            std::cout << "PASS: TP" << tpSize << " wrote " << expected.size() << " logits\n";
            return 0;
        }
        unsetenv("FASTLLM_DSV41_DISABLE_SHARED_OVERLAP");
        Compare(expected, RunRequests(model, {7, 17, 33, 7}));
        if (!eagerOnly) {
            SetCudaGraph(true);
            // Ordinary decode captures only at service startup. Exercise the
            // same warmup here instead of relying on request-time capture.
            model.maxBatch = 1;
            model.WarmUp();
            const size_t ordinaryShapes = diskExperts ? 0 : 1;
            if ((model.*(&Access::v41CudaGraphSlots)).size() != ordinaryShapes)
                throw std::runtime_error("unexpected startup decode graph shapes");
            Compare(expected, RunRequests(model, {7, 17, 33, 7}));
            SetCudaGraph(false);
            Compare(expected, RunRequests(model, {7, 17, 33, 7}));
            SetCudaGraph(true);
            Compare(expected, RunRequests(model, {7, 17, 33, 7}));
            std::cout << "PASS: eager/graph switching and shared overlap match across 4 requests, 52 steps each\n";
        }
        SetCudaGraph(false);
        setenv("FASTLLM_DSV41_DISABLE_TP_VERIFY_ASYNC", "1", 1);
        const auto verifyEight = RunDsparkRequests(model, 8);
        unsetenv("FASTLLM_DSV41_DISABLE_TP_VERIFY_ASYNC");
        Compare(verifyEight, RunDsparkRequests(model, 8));
        auto *workspace = (model.*(&Access::v41TpVerifyWorkspace)).get();
        if (tpSize > 1 && !workspace) throw std::runtime_error("asynchronous verifier was not exercised");
        Compare(verifyEight, RunDsparkRequests(model, 8));
        if ((model.*(&Access::v41TpVerifyWorkspace)).get() != workspace)
            throw std::runtime_error("verifier workspace was not reused");
        std::cout << "PASS: eager verification 1..8, features and rollback match across requests\n";
        if (eagerOnly) return 0;
        SetCudaGraph(true);
        uint64_t cacheBefore[8] = {}, cacheAfter[8] = {};
        if (expertCache && tpSize == 1 && !fastllm_moe_cuda_cache_route_stats(0, cacheBefore))
            throw std::runtime_error("cannot read cache counters before graph verification");
        Compare(verifyEight, RunDsparkRequests(model));
        if (expertCache && tpSize == 1) {
            if (!fastllm_moe_cuda_cache_route_stats(0, cacheAfter))
                throw std::runtime_error("cannot read cache counters after graph verification");
            // 196 steps, top-2: even counting every step as
            // ordinary decode cannot account for verification's route count.
            const uint64_t decodeUpperBound = 4 * (8 * 6 + 1) * model.block_cnt * 2;
            if (cacheAfter[1] - cacheBefore[1] <= decodeUpperBound ||
                cacheAfter[4] - cacheBefore[4] <= decodeUpperBound)
                throw std::runtime_error("graph verification bypassed the expert cache");
            std::cout << "PASS: graph verification cache routes=" << cacheAfter[1] - cacheBefore[1]
                      << " GPU routes=" << cacheAfter[4] - cacheBefore[4] << '\n';
        }
        const size_t expectedGraphShapes = diskExperts ? 0 : 8;
        if ((model.*(&Access::v41CudaGraphSlots)).size() != expectedGraphShapes)
            throw std::runtime_error("unexpected DSpark graph shapes or disk graph fallback");
        SetCudaGraph(false);
        Compare(verifyEight, RunDsparkRequests(model));
        SetCudaGraph(true);
        Compare(verifyEight, RunDsparkRequests(model));
        if (expertCache && tpSize == 1) {
            // A cold/unprofitable cache falls back to NUMA. Switching back
            // to resident GPU experts must preserve every captured buffer.
            setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "0", 1);
            SetCudaGraph(false);
            const auto cpuVerify = RunDsparkRequests(model);
            SetCudaGraph(true);
            Compare(cpuVerify, RunDsparkRequests(model));
            setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "2", 1);
            Compare(verifyEight, RunDsparkRequests(model));
            std::cout << "PASS: graph verification survives NUMA fallback and resident cache transitions\n";
        }
        if (diskExperts) {
            std::cout << "PASS: disk experts preserve eager results with CUDA Graph requested, including verification and rollback\n";
        } else {
            std::cout << "PASS: DSpark graph shapes 1..8, main features and rollback match across requests\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
