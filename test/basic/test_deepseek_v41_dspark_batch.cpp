// Usage: test_deepseek_v41_dspark_batch TP_FIXTURE_DIR ATTENTION_DP [bf16|fp4] [requests] [balanced]
// Uses the existing TP fixture (no draft weights): any accidental batched
// draft forward fails. Compare target logits and features against eager math.
#include "model.h"
#include "models/deepseekv41.h"
#include "fastkv/fastkv.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <atomic>
#include <dlfcn.h>
#include <cuda_runtime.h>

// Count actual CUDA launches, not the model's chosen workspace. A newly
// admitted request must not silently send the entire batch through eager math.
static std::atomic<int> graphLaunches{0};
static bool balanced = false;
extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    using Launch = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static const auto launch = reinterpret_cast<Launch>(dlsym(RTLD_NEXT, "cudaGraphLaunch"));
    if (!launch) return cudaErrorNotSupported;
    ++graphLaunches;
    return launch(graph, stream);
}

using namespace fastllm;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::BuildFullDecodeGraph;
    using DeepSeekV41Model::ForwardSegments;
    using DeepSeekV41Model::ForwardSegmentsWithDspark;
    using DeepSeekV41Model::GetOrCreateDsparkState;
    using DeepSeekV41Model::v41CudaGraphWarmingUp;
    using DeepSeekV41Model::v41DsparkEnabled;
    using DeepSeekV41Model::v41DsparkLayers;
    using DeepSeekV41Model::v41DsparkTargetLayerIds;
    using DeepSeekV41Model::v41IsDsparkTarget;
    using DeepSeekV41Model::RegisterState;
    using DeepSeekV41Model::GetOrCreateState;
};
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void ImportRequest(DeepSeekV41Model &model, DeepSeekV41Segment &segment) {
    ResponseContext source, dest;
    source.pastKeyValues.resize(model.block_cnt);
    dest.pastKeyValues.resize(model.block_cnt);
    source.preTokens = dest.preTokens = segment.state->totalLen;
    source.allTokens = dest.allTokens = std::vector<int>(source.preTokens + 1, 3);
    (model.*(&Access::RegisterState))(&source.pastKeyValues, &source.pastKeyValues[0].first, segment.state);
    const auto message = model.ExportPDState(source);
    model.OnResponseContextRemoved(&source);
    model.ImportPDState(dest, message);
    auto restored = (model.*(&Access::GetOrCreateState))(dest.pastKeyValues, false);
    Check(restored->totalLen == segment.state->totalLen, "import advanced request position");
    for (const auto &rank : restored->attentionRanks) {
        Check(rank && rank->totalLen == restored->totalLen, "import did not prepare owner state");
        for (const auto &layer : rank->layers)
            Check(layer.windowKV.cudaData && layer.windowKV.dataDevice == CUDA,
                  "import left the sliding window on CPU");
    }
    const auto exported = model.ExportPDState(dest);
    Check(message.metadata == exported.metadata && message.buffers.size() == exported.buffers.size(),
          "import changed KV metadata");
    for (size_t i = 0; i < message.buffers.size(); ++i) {
        std::vector<uint8_t> before(message.buffers[i].Size()), after(exported.buffers[i].Size());
        message.buffers[i].Read(before.data(), 0, before.size());
        exported.buffers[i].Read(after.data(), 0, after.size());
        Check(before == after, "import changed KV bytes");
    }
    segment.state = std::move(restored);
    model.OnResponseContextRemoved(&dest);
}
struct Requests {
    std::vector<DeepSeekV41Segment> segments;
    std::vector<Data> storage;
    std::vector<std::pair<Data *, Data *>> past;
    std::vector<GenerationConfig> configs;
    Requests(int count, int layers) : segments(count), storage(2 * count * layers), configs(count) {
        for (auto &s : segments) {
            s.state = std::make_shared<DeepSeekV41RequestState>();
            s.state->layers.resize(layers);
        }
        for (size_t i = 0; i < storage.size(); i += 2) past.emplace_back(&storage[i], &storage[i+1]);
        for (auto &c : configs) { c.do_sample = false; c.top_k = 1; c.output_logits = true; }
    }
};
static std::vector<float> Step(DeepSeekV41Model &model, Requests &r, int count, int length,
                             bool wrapper, bool features = false, Requests *padding = nullptr) {
    std::vector<DeepSeekV41Segment> segments(r.segments.begin(), r.segments.begin() + count);
    if (padding && count > 8 && length == 1) {
        std::vector<int> owners(4, 0);
        for (auto &seg : segments) ++owners[seg.state->attentionGroup];
        int bucket = 1;
        while (bucket < *std::max_element(owners.begin(), owners.end())) bucket *= 2;
        const int stride = padding->segments.size() / 4;
        for (int group = 0; group < 4; ++group)
            for (int i = owners[group]; i < bucket; ++i)
                segments.push_back(padding->segments[group * stride + i]);
    }
    std::vector<float> tokens;
    for (int i = 0; i < int(segments.size()); ++i) {
        auto &s = segments[i];
        s.startPos = s.state->totalLen; s.seqlen = length; s.offset = tokens.size();
        for (int j = 0; j < length; ++j) tokens.push_back(3 + (s.startPos + j + i * 19) % 100);
    }
    DeepSeekV41SpecScratch scratch;
    if (features) { Check(count == 1, "single feature step"); scratch.captureMain = true; segments[0].spec = &scratch; }
    Data ids(FLOAT32, {1, int(tokens.size())}, tokens);
    auto configs = std::vector<GenerationConfig>(r.configs.begin(), r.configs.begin() + count);
    auto past = std::vector<std::pair<Data *, Data *>>(r.past.begin(), r.past.begin() + count * model.block_cnt);
    const int rows = segments.size();
    configs.resize(rows);
    for (int i = count; i < rows; ++i) { configs[i].do_sample = false; configs[i].top_k = 1; }
    std::vector<Data> extra((rows-count)*model.block_cnt*2);
    for (size_t i = 0; i < extra.size(); i += 2) past.emplace_back(&extra[i], &extra[i+1]);
    LastTokensManager last(rows, 64);
    std::vector<std::vector<float>> logits(rows);
    std::vector<std::vector<float> *> outputs;
    for (auto &row : logits) outputs.push_back(&row);
    std::vector<int> got;
    const int launches = graphLaunches.load();
    if (wrapper) {
        got = (model.*(&Access::ForwardSegmentsWithDspark))(segments, ids, nullptr, nullptr, configs,
            last, &outputs, past, false);
        for (auto &s : segments) Check(!s.state->dspark, "batch retained stale draft state");
    } else {
        got = (model.*(&Access::ForwardSegments))(segments, ids, nullptr, nullptr, configs, last, &outputs, past);
    }
    if (wrapper && count > 8 && length == 1 && segments[0].state->attentionGroup >= 0) {
        const int launched = graphLaunches.load() - launches;
        if (launched != 8) std::cerr << "batch=" << count << " CUDA launches=" << launched << '\n';
        Check(launched == 8, "decode did not use exactly one full graph per GPU");
    }
    std::vector<float> values;
    for (int i = 0; i < count; ++i) {
        if (!configs[i].output_logits) { values.push_back(got.at(i)); continue; }
        auto &row = logits[i];
        Check(row.size() == size_t(model.weight["head.weight"].dims[0]), "missing target logits");
        for (float x : row) Check(std::isfinite(x), "non-finite logit");
        values.insert(values.end(), row.begin(), row.end());
    }
    if (features) {
        Check(scratch.mainHidden.size() == 3, "plain DSpark step lost target features to full graph");
        for (const auto &hidden : scratch.mainHidden) {
            Data host;
            const Data &local = hidden.multiDeviceData ? *hidden.multiDeviceDatas.begin()->second : hidden;
            host.CopyFrom(local); host.ToDevice(CPU); ToDataTypeForceCPU(host, FLOAT32);
            auto ptr = reinterpret_cast<float *>(host.cpuData);
            values.insert(values.end(), ptr, ptr + host.Count(0));
        }
    }
    return values;
}
static std::vector<float> Run(DeepSeekV41Model &model, bool graph, int dp, int requests, bool padReference = false) {
    // Graphs are enabled at process startup, as in ftllm serve, so cuBLAS
    // reserves its capture workspace before first use. Before warmup there
    // are no graph slots, making the first run an eager reference.
    Requests r(requests, model.block_cnt);
    if (dp == 4) for (int i = 0; i < requests; ++i) {
        auto &state = *r.segments[i].state;
        state.attentionGroup = balanced ? i % 4 : requests == 64 ? (i < 23 ? 0 : i < 41 ? 1 : i < 54 ? 2 : 3) :
            (i < 7 ? 0 : i < 12 ? 1 : i < 15 ? 2 : 3);
        state.attentionRanks.resize(2);
    }
    std::unique_ptr<Requests> padding;
    if (padReference) {
        padding.reset(new Requests(requests*2, model.block_cnt));
        for (int i = 0; i < requests*2; ++i) {
            padding->segments[i].state->attentionGroup = i / (requests/2);
            padding->segments[i].state->attentionRanks.resize(2);
        }
        Step(model, *padding, requests*2, 1, false);
    }
    std::vector<float> result;
    for (int step = 0; step < 16; ++step) {
        const int small[] = {16,15,14,13,12,11,10,9,13,16,7,7,7,15,9,15};
        const int large[] = {64,63,62,61,60,59,58,57,49,48,31,17,9,63,33,61};
        const int count = (requests == 64 ? large : small)[step];
        // Churn at both odd/even compressor positions. Imported requests join
        // an already active irregular batch without an eager initialization step.
        if (dp > 1 && step > 0) {
            ImportRequest(model, r.segments[step % count]);
            ImportRequest(model, r.segments[(step * 7 + 1) % count]);
        }
        // Also exercise a sampler that supports DSpark, without ret-logits
        // disabling the old draft path on this last batch.
        for (auto &config : r.configs) config.output_logits = step != 15;
        if (graph) for (int i = 0; i < count; ++i) {
            auto &draft = r.segments[i].state->dspark;
            draft = std::make_shared<DeepSeekV41DsparkState>();
            draft->committed = r.segments[i].state->totalLen;
            draft->drafts = {91, 92, 93}; draft->anchor = 90;
        }
        auto values = Step(model, r, count, step == 0 ? 17 : 1, graph, false, padding.get());
        result.insert(result.end(), values.begin(), values.end());
    }
    // A request leaving a batch must not recover any proposal or populated
    // draft window from before the intervening ordinary target steps.
    auto &state = *r.segments[0].state;
    auto draft = (model.*(&Access::GetOrCreateDsparkState))(state, state.totalLen);
    Check(draft->committed == state.totalLen && draft->filled == 0 && draft->drafts.empty() &&
          draft->pending.empty(), "draft resumption did not start at the target boundary");
    for (auto &config : r.configs) config.output_logits = true;
    auto values = Step(model, r, 1, 1, false, true);
    result.insert(result.end(), values.begin(), values.end());
    return result;
}
int main(int argc, char **argv) {
    try {
        Check(argc >= 3, "usage: regression TP_FIXTURE_DIR ATTENTION_DP [bf16|fp4] [requests]");
        setenv("FASTLLM_SKIP_WARMUP", "1", 1);
        setenv("FASTLLM_EXECUTION_MODE", "gpu", 1);
        setenv("FASTLLM_TP", "8", 1);
        unsetenv("FASTLLM_DSPARK_TOKENS");
        SetThreads(4); SetCudaGraph(true); SetCudaSharedExpert(true); SetNgramDevice("cuda");
        const std::string devices = "multicuda:0,1,2,3,4,5,6,7";
        SetDeviceMap({{devices, 1}}); SetMoeDeviceMap({{devices, 1}});
        auto base = CreateLLMModelFromHF(argv[1], DATA_AUTO_SOURCE);
        auto &model = dynamic_cast<DeepSeekV41Model &>(*base);
        model.ConfigureAttentionParallel(std::stoi(argv[2]));
        if (argc > 3 && std::string(argv[3]) == "fp4") model.SetKVCacheDataType(FP4_E2M1);
        const int requests = argc > 4 ? std::stoi(argv[4]) : 16;
        balanced = argc > 5 && std::string(argv[5]) == "balanced";
        Check(requests == 16 || requests == 64, "requests must be 16 or 64");
        model.maxBatch = requests;
        model.max_positions = 4096;
        model.*(&Access::v41DsparkEnabled) = true;
        model.*(&Access::v41DsparkLayers) = 3;
        model.*(&Access::v41DsparkTargetLayerIds) = {model.block_cnt-3, model.block_cnt-2, model.block_cnt-1};
        auto &mask = model.*(&Access::v41IsDsparkTarget);
        mask.assign(model.block_cnt, false);
        for (int layer : model.*(&Access::v41DsparkTargetLayerIds)) mask[layer] = true;
        const int dp = std::stoi(argv[2]);
        const auto unpadded = Run(model, false, dp, requests);
        const auto expected = dp == 4 ? Run(model, false, dp, requests, true) : unpadded;
        if (balanced) {
            const auto repeated = Run(model, false, dp, requests, true);
            size_t bad = 0;
            for (size_t i = 0; i < expected.size(); ++i) if (expected[i] != repeated[i]) ++bad;
            std::cout << "eager_repeat_mismatches=" << bad << std::endl;
            Check(bad == 0, "eager execution itself is not repeatable");
        }
        model.*(&Access::v41CudaGraphWarmingUp) = true;
        for (int batch = 1; batch <= requests * 2; batch *= 2) {
            Requests warm(batch, model.block_cnt);
            for (int i = 0; i < 6; ++i) Step(model, warm, batch, 1, false);
            auto segments = warm.segments;
            for (int i = 0; i < batch; ++i) {
                segments[i].startPos = segments[i].state->totalLen;
                segments[i].seqlen = 1; segments[i].offset = i;
            }
            (model.*(&Access::BuildFullDecodeGraph))(batch, segments, -1, -1);
        }
        model.*(&Access::v41CudaGraphWarmingUp) = false;
        const auto actual = Run(model, true, std::stoi(argv[2]), requests);
        if (balanced) for (int repeat = 0; repeat < 2; ++repeat) {
            const auto again = Run(model, true, dp, requests);
            Check(again == actual, "full graph changed output between identical admission traces");
        }
        Check(actual.size() == expected.size(), "result shape changed");
        double maxError = 0;
        double error2 = 0, reference2 = 0;
        size_t worst = 0;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            const double error = std::abs(actual[i] - expected[i]);
            if (error > maxError) { maxError = error; worst = i; }
            error2 += error * error; reference2 += double(expected[i]) * expected[i];
        }
        std::cout << "max_error=" << maxError << " relative_l2=" << std::sqrt(error2 / reference2)
                  << " worst_index=" << worst << " expected=" << expected[worst]
                  << " actual=" << actual[worst] << '\n';
        Check(maxError == 0, "graph/fallback changed same-shape eager target logits or features");
        // Padding can select another BF16/FP8 GEMM shape. Confirm the actual
        // greedy sampler remains identical to the original unpadded path too.
        const int sampled = requests == 64 ? 61 : 15;
        const size_t first = actual.size() - model.weight["head.weight"].dims[0] - 3*model.weight["embed.weight"].dims[1] - sampled;
        for (int i = 0; i < sampled; ++i) Check(actual[first+i] == unpadded[first+i], "padding changed greedy token");
        std::cout << "PASS: DSpark dynamic/uneven batch to single, stale proposals discarded, "
                     "target logits/features bitwise equal; values=" << actual.size() << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
