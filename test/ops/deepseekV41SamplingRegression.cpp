// Test the V4.1 adapter, conditional multi-token rejection, and cache rollback.
// Uses synthetic p/q so statistical correctness does not depend on model quality
// or comparisons between different floating-point forward kernels.
#include "models/deepseekv41.h"
#include "executor.h"
#include "baseblock.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
struct Probe : DeepSeekV41Model {
    using DeepSeekV41Model::DsparkCommitPrefix;
    using DeepSeekV41Model::DsparkAdvance;
    using DeepSeekV41Model::DsparkProjectHead;
    using DeepSeekV41Model::DsparkAcceptedDraftCount;
    using DeepSeekV41Model::DsparkMaskToolLogits;
    using DeepSeekV41Model::DsparkDraftConfig;
    using DeepSeekV41Model::DsparkSampleVerify;
    using DeepSeekV41Model::DsparkSamplingConfig;
    using DeepSeekV41Model::DsparkSupportsRequest;
    Probe() {
        eos_token_id = -1;
        v41DsparkEnabled = true;
        deviceMap = {{"cuda:0", 1}};
    }
    void HeadSetup() {
        deviceMap = {{"multicuda:0,1", 1}};
        v41DsparkTpDevices = {0, 1};
    }
    void MainFeatureSetup(bool tp, int firstDevice) {
        if (tp) {
            deviceMap = {{"multicuda:" + std::to_string(firstDevice) + "," + std::to_string(1 - firstDevice), 1}};
            v41DsparkTpDevices = {firstDevice, 1 - firstDevice};
        } else {
            deviceMap = {{"cuda:" + std::to_string(firstDevice), 1}};
        }
        embed_dim = head_dim_full = 128;
        qk_rope_head_dim = 32;
        window_size = 8;
        v41DsparkLayers = 1;
        v41DsparkTargetLayerIds = {0, 1, 2};
        std::vector<float> projection(128 * 384), identity(128 * 128);
        for (int i = 0; i < 128; ++i) {
            for (int layer = 0; layer < 3; ++layer)
                projection[i * 384 + layer * 128 + i] = (layer + 1) / 8.0f;
            identity[i * 128 + i] = 1.0f;
        }
        weight["mtp.0.main_proj.weight"].CopyFrom(Data(DataType::FLOAT32, {128, 384}, projection));
        weight["mtp.0.attn.wkv.weight"].CopyFrom(Data(DataType::FLOAT32, {128, 128}, identity));
        for (const char *name : {"mtp.0.main_norm.weight", "mtp.0.attn.kv_norm.weight"})
            weight[name].CopyFrom(Data(DataType::FLOAT32, {128}, std::vector<float>(128, 1.0f)));
    }
    void CacheSetup(bool tp) {
        block_cnt = 2;
        window_size = 8;
        compress_ratios = {4, 16};
        isKvSource = {true, true};
        engram_layer_ids = {0};
        deviceMap = {{tp ? "multicuda:0,1" : "cuda:0", 1}};
        v41DsparkTpDevices = tp ? std::vector<int>{0, 1} : std::vector<int>{};
    }
};
static std::array<double, 4> Target(int prev) {
    std::array<double, 4> p{};
    for (int t = 0; t < 4; ++t)
        p[t] = (1 + (t + prev) % 4) / 10.0;
    return p;
}
static int Draw(const std::array<double, 4> &p, std::mt19937 &rng) {
    return std::discrete_distribution<int>(p.begin(), p.end())(rng);
}
static void Sampling(Probe &model, int drafts, int mode) {
    constexpr int vocab = 128, samples = 6000;
    std::mt19937 rng(12731 + drafts * 37 + mode);
    std::vector<int> counts(64), acceptance(drafts + 1);
    std::vector<float> logits((drafts + 1) * vocab, -1000);
    Data gpu(DataType::FLOAT32, {1, drafts + 1, vocab}, logits);
    gpu.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    GenerationConfig config;
    config.do_sample = true;
    config.top_k = 4;
    config.top_p = 1;
    config.temperature = 1;
    config.repeat_penalty = 1;
    for (int sample = 0; sample < samples; ++sample) {
        DeepSeekV41DsparkState proposal;
        DeepSeekV41SpecScratch scratch;
        proposal.sampledProposal = mode != 0;
        std::vector<float> probabilities(drafts * vocab, 0.0f);
        int previous = 0;
        for (int row = 0; row <= drafts; ++row) {
            auto p = Target(previous);
            for (int t = 0; t < 4; ++t)
                logits[row * vocab + t] = std::log(p[t]);
            if (row == drafts)
                break;
            auto q = mode == 2 ? p : std::array<double, 4>{0.6, 0.3, 0.1, 0.0};
            int token = mode == 0 ? (previous + 1) % 4 : Draw(q, rng);
            proposal.proposalTokens.push_back(token);
            scratch.draftTokens.push_back(token);
            for (int t = 0; t < 4; ++t)
                probabilities[row * vocab + t] = q[t];
            previous = token;
        }
        Check(cudaMemcpy(gpu.cudaData, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice) ==
                  cudaSuccess,
              "logits upload");
        if (proposal.sampledProposal) {
            proposal.proposalProbs.CopyFrom(Data(DataType::FLOAT32, {drafts, vocab}, probabilities));
            proposal.proposalProbs.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        }
        model.DsparkSampleVerify(gpu, proposal, scratch, config);
        ++acceptance.at(scratch.acceptedDrafts);
        // Every emitted prefix is retained. Complete short prefixes with the
        // independent target oracle, then check the three-token joint law.
        std::array<int, 3> output{};
        previous = 0;
        int joint = 0;
        for (int i = 0; i < 3; ++i) {
            output[i] = i <= scratch.acceptedDrafts ? scratch.tokens[i] : Draw(Target(previous), rng);
            previous = output[i];
            joint = joint * 4 + output[i];
        }
        ++counts.at(joint);
    }
    double maxZ = 0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 4; ++c) {
                double p = Target(0)[a] * Target(a)[b] * Target(b)[c];
                double z = std::abs(counts[(a * 4 + b) * 4 + c] - samples * p) / std::sqrt(samples * p * (1 - p));
                maxZ = std::max(maxZ, z);
            }
    Check(maxZ < 7, "rejection changed the conditional joint distribution");
    // log/exp/softmax rounding can make p/q microscopically below one.
    if (mode == 2)
        Check(acceptance[drafts] >= samples - 2, "equal p/q should accept entire chain");
    // Full acceptance of a long, low-overlap chain can be rarer than 1/samples;
    // mode 2 above covers that branch without a probabilistic coverage failure.
    if (mode != 2)
        Check(acceptance[0] > 0 && acceptance[1] > 0, "missing rejection/partial acceptance");
    std::cout << "joint drafts=" << drafts << " mode=" << mode << " samples=" << samples << " max_z=" << maxZ
              << " hist=";
    for (int n : acceptance)
        std::cout << n << ',';
    std::cout << " PASS\n" << std::flush;
}
static std::vector<float> Read(Data &data) {
    if (data.multiDeviceData) {
        Check(data.IsTensorParallelReplicated() && data.multiDeviceDatas.size() == 2, "TP cache layout");
        auto values = Read(*data.multiDeviceDatas.at(0));
        for (auto &replica : data.multiDeviceDatas) {
            Check(replica.second->dims == data.dims, "TP cache shape rollback");
            Check(Read(*replica.second) == values, "TP cache replicas differ");
        }
        return values;
    }
    Data cpu;
    cpu.CopyFrom(data);
    ToDataType(cpu, DataType::FLOAT32);
    cpu.ToDevice(DataDevice::CPU);
    return {(float *)cpu.cpuData, (float *)cpu.cpuData + cpu.Count(0)};
}
// Feed identical, distinct per-layer features through the ordinary and TP
// draft cache update. This detects missing uploads when crossing from TP to
// one GPU, even when target verification still produces correct output tokens.
static void MainFeatures() {
    for (int firstDevice : {0, 1}) {
        Probe single, tp;
        single.MainFeatureSetup(false, firstDevice);
        tp.MainFeatureSetup(true, firstDevice);
        DeepSeekV41RequestState singleState, tpState;
        int start = 0;
        for (auto lengths : {std::pair<int, int>{22, 22}, {1, 1}, {6, 3}, {6, 1}, {6, 6}}) {
            int rows = lengths.first, accepted = lengths.second;
            DeepSeekV41SpecScratch reference, distributed;
            reference.mainHidden.resize(3);
            distributed.mainHidden.resize(3);
            for (int layer = 0; layer < 3; ++layer) {
                std::vector<float> values(rows * 128);
                for (int i = 0; i < rows * 128; ++i)
                    values[i] = ((i * (layer * 2 + 1) + start * 7 + layer * 11) % 53 - 26) / 16.0f;
                Data host(DataType::BFLOAT16, {1, rows, 128}, values);
                reference.mainHidden[layer].CopyFrom(host);
                distributed.mainHidden[layer].CopyFrom(host);
                PrepareMultiCudaReplicatedData(distributed.mainHidden[layer], {0, 1}, true);
            }
            single.DsparkAdvance(singleState, reference, start, start + accepted, -1, GenerationConfig());
            // Include a TP -> single-GPU transition with the other GPU current.
            ((Executor *)GetExecutor())->SetFirstDevice("multicuda:0,1");
            Check(cudaSetDevice(1 - firstDevice) == cudaSuccess, "main feature source device");
            tp.DsparkAdvance(tpState, distributed, start, start + accepted, -1, GenerationConfig());
            Check(tpState.dspark && !tpState.dspark->disabled, "main feature TP draft disabled");
            auto expected = Read(singleState.dspark->layers[0].windowKV);
            auto actual = Read(tpState.dspark->layers[0].windowKV);
            Check(actual == expected, "TP main features changed draft window KV");
            for (float value : actual) Check(std::isfinite(value), "nonfinite draft window KV");
            Check(tpState.dspark->committed == start + accepted, "main feature commit length");
            start += accepted;
        }
    }
    std::cout << "TP main feature transfer, chunking and accepted prefixes PASS\n";
}

// A sparse head has an exact oracle independent of GEMM and gather code.
static void DraftHead() {
    constexpr int vocab = 256, width = 128;
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
        Probe model;
        model.HeadSetup();
        std::vector<float> weights(vocab * width);
        for (int row = 0; row < vocab; ++row)
            weights[row * width + row % width] = (row + 1) / 256.0f;
        auto &head = model.weight["head.weight"];
        head.CopyFrom(Data(type, {vocab, width}, weights));
        head.tpLinearType = TP_LINEAR_ROW;
        for (bool async : {false, true}) for (bool replicated : {false, true}) {
            const bool previousAsync = MultiCudaSetPersistentAsyncDispatch(async);
            Data output;
            for (int tokens : {1, 5, 6, 1}) {
                std::vector<float> values(tokens * width);
                for (int i = 0; i < tokens * width; ++i) values[i] = (i % 4) - 2;
                Data input(DataType::BFLOAT16, {1, tokens, width}, values);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                if (replicated) PrepareMultiCudaReplicatedData(input, {0, 1}, true);
                Check(cudaSetDevice(1) == cudaSuccess, "head starts on remote rank");
                model.DsparkProjectHead(input, output);
                Check(output.dims == std::vector<int>({1, tokens, vocab}) && !output.multiDeviceData,
                      "draft head gather layout");
                auto actual = Read(output);
                std::vector<float> expected(tokens * vocab);
                for (int t = 0; t < tokens; ++t)
                    for (int row = 0; row < vocab; ++row)
                        expected[t * vocab + row] = values[t * width + row % width] * weights[row * width + row % width];
                Data rounded(DataType::BFLOAT16, {1, tokens, vocab}, expected);
                auto reference = Read(rounded);
                Check(actual == reference, "draft head gathered wrong logits");
                Check(head.multiDeviceData && head.multiDeviceDatas.size() == 2,
                      "draft head discarded vocabulary shards");
            }
            Check(MultiCudaSetPersistentAsyncDispatch(previousAsync) == async,
                  "draft head changed async dispatch setting");
        }
    }
    std::cout << "draft TP head PASS\n";
}
static void DeviceSelection(Probe &model, int devices) {
    constexpr int drafts = 7, vocab = 128, token = 17;
    std::vector<float> values((drafts + 1) * vocab, -1000);
    for (int row = 0; row <= drafts; ++row)
        values[row * vocab + token] = 0;
    GenerationConfig config;
    config.do_sample = true;
    config.top_k = 4;
    for (int device = 0; device < std::min(devices, 2); ++device) {
        Data logits(DataType::FLOAT32, {1, drafts + 1, vocab}, values);
        logits.ToDevice(DataDevice::CUDA, std::vector<int>{device});
        DeepSeekV41DsparkState proposal;
        DeepSeekV41SpecScratch scratch;
        proposal.proposalTokens.assign(drafts, token);
        scratch.draftTokens.assign(drafts, token);
        // Simulate a preceding operator leaving a different device current.
        Check(cudaSetDevice((device + 1) % devices) == cudaSuccess, "device switch");
        model.DsparkSampleVerify(logits, proposal, scratch, config);
        // Save more rows than verification consumes, on a different GPU.
        // This covers confidence/output-length truncation and portable staging.
        std::vector<float> q((drafts + 2) * vocab, 0.0f);
        for (int row = 0; row < drafts + 2; ++row) q[row * vocab + token] = 1.0f;
        proposal.sampledProposal = true;
        proposal.proposalProbs.CopyFrom(Data(DataType::FLOAT32, {drafts + 2, vocab}, q));
        proposal.proposalProbs.ToDevice(DataDevice::CUDA, std::vector<int>{(device + 1) % devices});
        model.DsparkSampleVerify(logits, proposal, scratch, config);
        int current = -1;
        Check(cudaGetDevice(&current) == cudaSuccess, "device query");
        Check(current == device, "sampling did not select the logits device");
        Check(scratch.acceptedDrafts == drafts && scratch.tokens == std::vector<int>(drafts + 1, token),
              "cross-device logits transfer/sampling");
    }
    Check(cudaSetDevice(0) == cudaSuccess, "restore device");
    std::cout << "device selection count=" << std::min(devices, 2) << " PASS\n";
}
static void Rollback(Probe &model, bool tp = false) {
    model.CacheSetup(tp);
    ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
    int cases = 0;
    for (int drafts : {1, 3, 4, 5, 7})
        for (int start : {7, 8, 9, 127, 129})
            for (int accepted = 1; accepted <= drafts + 1; ++accepted) {
                int forwarded = drafts + 1, end = start + accepted;
                DeepSeekV41RequestState state;
                DeepSeekV41SpecScratch scratch;
                state.layers.resize(2);
                state.totalLen = start + forwarded;
                state.engramHistory.resize(state.totalLen);
                std::iota(state.engramHistory.begin(), state.engramHistory.end(), 0);
                scratch.windowKV.resize(2);
                scratch.rawKV.resize(2);
                scratch.rawScore.resize(2);
                scratch.prevRawTail.resize(2);
                scratch.prevBlocks.resize(2);
                std::vector<float> newRows(forwarded);
                for (int i = 0; i < forwarded; ++i)
                    newRows[i] = start + i;
                for (int layer = 0; layer < 2; ++layer) {
                    int ratio = layer == 0 ? 4 : 16, tail = start % ratio;
                    auto &cache = state.layers[layer];
                    std::vector<float> ring(8);
                    for (int pos = start - 8; pos < start; ++pos)
                        ring[(pos + 8) % 8] = pos;
                    cache.windowKV.CopyFrom(Data(DataType::FLOAT32, {1, 8, 1}, ring));
                    scratch.windowKV[layer].CopyFrom(Data(DataType::FLOAT32, {1, forwarded, 1}, newRows));
                    scratch.prevRawTail[layer] = tail;
                    scratch.prevBlocks[layer] = start / ratio;
                    std::vector<float> raw(tail + forwarded);
                    for (int i = 0; i < (int)raw.size(); ++i)
                        raw[i] = start - tail + i;
                    scratch.rawKV[layer].CopyFrom(Data(DataType::FLOAT32, {1, (int)raw.size(), 1}, raw));
                    for (float &v : raw)
                        v += 1000;
                    scratch.rawScore[layer].CopyFrom(Data(DataType::FLOAT32, {1, (int)raw.size(), 1}, raw));
                    int blocks = (start + forwarded) / ratio;
                    std::vector<float> compressed(blocks);
                    std::iota(compressed.begin(), compressed.end(), 0);
                    cache.compressedKV.CopyFrom(Data(DataType::FLOAT32, {1, blocks, 1}, compressed));
                    cache.indexK.CopyFrom(Data(DataType::FLOAT32, {1, blocks, 1}, compressed));
                    if (tp) {
                        for (Data *data : {&cache.windowKV, &cache.compressedKV, &cache.indexK,
                                           &scratch.windowKV[layer], &scratch.rawKV[layer], &scratch.rawScore[layer]}) {
                            PrepareMultiCudaReplicatedData(*data, {0, 1}, true);
                        }
                    }
                }
                model.DsparkCommitPrefix(state, scratch, start, accepted, forwarded);
                Check(state.totalLen == end && state.engramHistory.size() == end, "history/total rollback");
                Check(state.engramHistory.back() == end - 1, "history prefix rollback");
                for (int layer = 0; layer < 2; ++layer) {
                    int ratio = layer == 0 ? 4 : 16;
                    auto &cache = state.layers[layer];
                    Check(cache.totalLen == end && cache.compressedBlocks == end / ratio &&
                              cache.rawTail == end % ratio,
                          "cache counters rollback");
                    Check(cache.compressedKV.dims[1] == end / ratio && cache.indexK.dims[1] == end / ratio,
                          "compressed/index rollback");
                    auto compressed = Read(cache.compressedKV), index = Read(cache.indexK);
                    for (int i = 0; i < end / ratio; ++i) {
                        Check(compressed[i] == i && index[i] == i, "compressed/index prefix changed");
                    }
                    auto ring = Read(cache.windowKV);
                    for (int pos = end - 8; pos < end; ++pos)
                        Check(ring[(pos + 8) % 8] == pos, "window rollback");
                    if (end % ratio) {
                        auto raw = Read(cache.rawTailKV), score = Read(cache.rawTailScore);
                        Check(raw.size() == end % ratio && score.size() == raw.size(), "raw tail shape");
                        for (int i = 0; i < (int)raw.size(); ++i) {
                            Check(raw[i] == end - end % ratio + i, "raw tail rollback");
                            Check(score[i] == raw[i] + 1000, "raw score rollback");
                        }
                    }
                }
                ++cases;
            }
    std::cout << "rollback tp=" << tp << " cases=" << cases << " PASS\n";
}
// Cross invoke/parameter boundaries inside a block, including a second
// parameter and a closed invocation. No model weights needed.
static void ToolConstraints(Probe &model) {
    for (const std::string marker : {std::string("｜DSML｜ "), std::string("\\DSML\\ ")}) {
        const std::string invoke = "<" + marker + "invoke name=\"";
        const std::string param = "<" + marker + "parameter name=\"";
        const std::string closeParam = "</" + marker + "parameter>";
        const std::string closeInvoke = "</" + marker + "invoke>";
        std::vector<std::string> pieces = {"read", "\">" + param, "path",
            "\">value" + closeParam + param, "mode", "\">x" + closeParam + closeInvoke,
            "INVALID", "write", "\""};
        model.weight.tokenizer.Clear();
        for (int i = 0; i < (int)pieces.size(); ++i) model.weight.tokenizer.Insert(pieces[i], i);
        GenerationConfig c;
        c.tool_call_name_constraint_enabled = true;
        c.tool_call_parameter_name_constraint_enabled = true;
        c.tool_call_allowed_names = {"read", "write"};
        c.tool_call_allowed_parameter_names = {{"read", {"path", "mode"}}, {"write", {"data"}}};
        c.tool_call_invoke_name_prefixes = {invoke};
        c.tool_call_parameter_name_prefixes = {param};
        c.tool_call_generated_text = invoke;
        c.top_k = 1;
        DeepSeekV41RequestState state;
        Check(model.DsparkSupportsRequest(c, state), "tool constraints disabled DSpark");
        const int vocab = pieces.size();
        std::vector<float> scores(7 * vocab, 0);
        for (int row = 0; row < 7; ++row) {
            scores[row * vocab + row] = 10;
            if (row % 2 == 0) scores[row * vocab + 6] = 100;
        }
        Data logits(DataType::FLOAT32, {1, 7, vocab}, scores);
        model.DsparkMaskToolLogits(logits, c, {0, 1, 2, 3, 4, 5});
        auto *v = (float*)logits.cpuData;
        for (int row : {0, 2, 4}) {
            Check(std::isinf(v[row * vocab + 6]) && v[row * vocab + 6] < 0, "invalid tool token not masked");
            Check(v[row * vocab + row] == 10, "valid tool token masked");
        }
        Check(v[6 * vocab + 6] == 100, "mask leaked past invocation end");
        // Private speculative text includes accepted tokens + correction, not
        // rejected drafts; the scheduler-owned snapshot stays unchanged.
        DeepSeekV41SpecScratch scratch;
        scratch.wantAllTokens = true;
        scratch.draftTokens = {0, 1, 2, 3, 4, 5};
        scratch.tokens = {0, 1, 4, 3, 4, 5, 6};
        for (int accepted : {0, 1, 2, 6}) {
            scratch.acceptedDrafts = accepted;
            auto next = model.DsparkDraftConfig(c, scratch, scratch.tokens[accepted]);
            std::string expected = invoke;
            for (int i = 0; i <= accepted; ++i) expected += pieces[scratch.tokens[i]];
            Check(next.tool_call_generated_text == expected, "accepted prefix snapshot mismatch");
            Check(c.tool_call_generated_text == invoke, "speculation changed emitted prefix");
        }
        scratch.acceptedDrafts = -1;
        Check(model.DsparkDraftConfig(c, scratch, 4).tool_call_generated_text == invoke + pieces[0] + pieces[1] + pieces[4],
              "greedy rejection snapshot mismatch");
        scratch.wantAllTokens = false;
        Check(model.DsparkDraftConfig(c, scratch, 0).tool_call_generated_text == invoke + pieces[0], "anchor duplicated/missing");
        // Invalid forced q must be rejected, including at a constrained bonus.
        c.top_k = 2;
        DeepSeekV41DsparkState proposal;
        proposal.proposalTokens = {6};
        scratch.draftTokens = {6};
        std::vector<float> twoRows(2 * vocab, 0);
        twoRows[6] = 2000;
        twoRows[0] = 1000;
        Data gpu(DataType::FLOAT32, {1, 2, vocab}, twoRows);
        gpu.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        model.DsparkSampleVerify(gpu, proposal, scratch, c);
        Check(scratch.acceptedDrafts == 0 && scratch.tokens[0] == 0, "invalid draft accepted through tool mask");
        proposal.proposalTokens = {0};
        scratch.draftTokens = {0};
        twoRows[vocab + 6] = 2000;
        twoRows[vocab + 8] = 1000;
        Check(cudaMemcpy(gpu.cudaData, twoRows.data(), twoRows.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "bonus logits upload");
        model.DsparkSampleVerify(gpu, proposal, scratch, c);
        Check(scratch.acceptedDrafts == 1 && scratch.tokens[1] == 8, "bonus used stale tool constraint");
    }
    // Nontrivial top-k/top-p law: masked CPU decoding renormalizes the top-k
    // BEFORE top-p. [0.4,0.3,0.2,0.1], k=2,p=.6 must retain both a and b.
    model.weight.tokenizer.Clear();
    for (int i = 0; i < 4; ++i) model.weight.tokenizer.Insert(std::string(1, 'a' + i) + "\"", i);
    GenerationConfig c;
    c.tool_call_name_constraint_enabled = true;
    c.tool_call_allowed_names = {"a", "b", "c", "d"};
    c.tool_call_invoke_name_prefixes = {"<invoke name=\""};
    c.tool_call_generated_text = "<invoke name=\"";
    c.top_k = 2; c.top_p = .6; c.temperature = 1;
    constexpr int vocab = 128, trials = 4000;
    int counts[2] = {0, 0};
    std::vector<float> raw(2 * vocab, -1000);
    for (int i = 0; i < 4; ++i) raw[i] = std::log((4 - i) / 10.0f);
    Data gpu(DataType::FLOAT32, {1, 2, vocab}, raw);
    gpu.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    std::vector<float> draftRaw(vocab, -1000);
    draftRaw[0] = std::log(.55f); draftRaw[1] = std::log(.45f);
    Data draftLogits(DataType::FLOAT32, {1, 1, vocab}, draftRaw);
    draftLogits.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    model.DsparkMaskToolLogits(draftLogits, c, {});
    for (int i = 0; i < trials; ++i) {
        Check(cudaMemcpy(gpu.cudaData, raw.data(), raw.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "masked logits upload");
        DeepSeekV41DsparkState proposal;
        proposal.proposalTokens = {0}; // deterministic q, rejection exercises residual b
        DeepSeekV41SpecScratch scratch;
        scratch.draftTokens = {0};
        if (i % 2) {
            proposal.sampledProposal = true;
            proposal.proposalProbs.CopyFrom(Data(DataType::FLOAT32, {1, vocab}, std::vector<float>(vocab, 0)));
            proposal.proposalProbs.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            int token = -1;
            Check(FastllmCudaMtpSampleDraft((float*)draftLogits.cudaData, (float*)proposal.proposalProbs.cudaData,
                &c.temperature, &c.top_k, &c.top_p, &token, 1, vocab), "constrained draft sampling failed");
            Check(token == 0 || token == 1, "constrained draft support mismatch");
            proposal.proposalTokens = scratch.draftTokens = {token};
        }
        model.DsparkSampleVerify(gpu, proposal, scratch, c);
        Check(scratch.tokens[0] >= 0 && scratch.tokens[0] < 2, "masked top-k/top-p support mismatch");
        ++counts[scratch.tokens[0]];
    }
    Check(std::fabs((double)counts[0] / trials - 4.0 / 7) < .035, "constrained rejection distribution mismatch");
    std::cout << "tool boundary/state/rejection and top-k/top-p distribution PASS\n";
}
static void StopPrefix(Probe &model) {
    const int originalEos = model.eos_token_id;
    const auto originalStops = model.eos_token_ids;
    model.eos_token_id = 99;
    model.eos_token_ids = {100};
    GenerationConfig c;
    c.stop_token_ids = {101};
    for (int stop : {99, 100, 101}) {
        for (int at = 0; at <= 5; ++at) {
            for (bool greedy : {false, true}) {
                DeepSeekV41SpecScratch scratch;
                scratch.tokens = {0, 1, 2, 3, 4, 5};
                scratch.tokens[at] = stop;
                scratch.draftTokens.assign(scratch.tokens.begin(), scratch.tokens.end() - 1);
                scratch.acceptedDrafts = greedy ? -1 : 5;
                Check(model.DsparkAcceptedDraftCount(scratch, c) == at, "committed past stop token");
            }
        }
    }
    DeepSeekV41SpecScratch scratch;
    scratch.tokens = {0, 1, 2, 99, 4, 5};
    scratch.draftTokens = {0, 1, 9, 99, 4};
    scratch.acceptedDrafts = 2;
    Check(model.DsparkAcceptedDraftCount(scratch, c) == 2, "stop in rejected suffix changed prefix");
    scratch.acceptedDrafts = -1;
    Check(model.DsparkAcceptedDraftCount(scratch, c) == 2, "greedy rejected suffix changed prefix");
    model.eos_token_id = originalEos;
    model.eos_token_ids = originalStops;
    std::cout << "EOS/stop prefix truncation PASS\n";
}
static void Configs(Probe &model) {
    GenerationConfig c;
    DeepSeekV41RequestState state;
    Check(model.DsparkSupportsRequest(c, state), "greedy request disabled");
    c.do_sample = true;
    c.top_k = 1;
    Check(model.DsparkSamplingConfig(c).top_k == 5, "ordinary normalization mismatch");
    c.top_k = 40;
    c.top_p = .95;
    c.temperature = .8;
    Check(model.DsparkSupportsRequest(c, state), "sampled request disabled");
    auto good = c;
    c.tool_call_allowed_token_ids = {1, 2};
    Check(!model.DsparkSupportsRequest(c, state), "standalone mask must fall back");
    c.tool_call_name_constraint_enabled = true;
    Check(model.DsparkSupportsRequest(c, state), "active tool mask disabled DSpark");
    Check(model.DsparkSamplingConfig(c).tool_call_allowed_token_ids == c.tool_call_allowed_token_ids,
          "ordinary normalization dropped tool mask");
    c = good;
    c.repeat_penalty = 1.1;
    Check(!model.DsparkSupportsRequest(c, state), "repeat penalty fallback");
    c = good;
    c.output_logits = true;
    Check(!model.DsparkSupportsRequest(c, state), "logits fallback");
    c = good;
    c.output_token_least = 1;
    Check(!model.DsparkSupportsRequest(c, state), "min length fallback");
    c = good;
    c.tool_call_content_sampling_enabled = true;
    Check(!model.DsparkSupportsRequest(c, state), "tool fallback");
    c = good;
    model.deviceMap = {{"cpu", 1}};
    Check(!model.DsparkSupportsRequest(c, state), "CPU sampling fallback");
    model.deviceMap = {{"cuda:0", 1}};
    std::cout << "config/fallback PASS\n";
}
int main() {
#ifdef __linux__
    prctl(PR_SET_DUMPABLE, 0);
#endif
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        return 77;
    Check(cudaSetDevice(0) == cudaSuccess, "CUDA initialization");
    Probe model;
    Configs(model);
    DeviceSelection(model, devices);
    ToolConstraints(model);
    StopPrefix(model);
    for (int drafts : {3, 4, 5, 7})
        for (int mode = 0; mode < 3; ++mode)
            Sampling(model, drafts, mode);
    Rollback(model);
    if (devices >= 2) {
        MainFeatures();
        Rollback(model, true);
        DraftHead();
    }
    std::cout << "DeepSeek V4.1 sampling regression PASS\n";
}
