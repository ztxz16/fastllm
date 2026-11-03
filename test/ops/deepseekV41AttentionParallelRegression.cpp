// Synthetic multi-request fixture: test/basic/test_deepseek_v41_tp_graph.py.
#include "model.h"
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::ForwardSegments;
    using DeepSeekV41Model::SnapshotState;
    using DeepSeekV41Model::RestoreState;
    using DeepSeekV41Model::GetOrCreateState;
    using DeepSeekV41Model::BuildFullDecodeGraph;
    static void WarmGraphs(DeepSeekV41Model &model, bool enabled) {
        model.*(&Access::v41CudaGraphWarmingUp) = enabled;
    }
};
static void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static std::vector<float> Step(DeepSeekV41Model &model,
        const std::vector<std::shared_ptr<DeepSeekV41RequestState>> &states,
        const std::vector<int> &lengths, int seed, int sampleMode = -1) {
    std::vector<DeepSeekV41Segment> segments;
    std::vector<float> tokens;
    std::vector<GenerationConfig> configs(states.size());
    std::vector<std::vector<float>> outputs(states.size());
    std::vector<std::vector<float>*> pointers;
    for (size_t i = 0; i < states.size(); ++i) {
        DeepSeekV41Segment seg;
        seg.state = states[i]; seg.startPos = states[i]->totalLen;
        seg.seqlen = lengths[i]; seg.offset = tokens.size();
        for (int j = 0; j < seg.seqlen; ++j) tokens.push_back((seg.startPos + j + seed + i * 7) * 17 % 123 + 3);
        segments.push_back(seg); configs[i].output_logits = true;
        if (sampleMode >= 0) {
            configs[i].output_logits = false;
            configs[i].top_k = sampleMode ? 0 : 1;
            configs[i].top_p = 1.f;
            configs[i].temperature = sampleMode ? .8f : 0.f;
            configs[i].do_sample = sampleMode != 0;
        }
        pointers.push_back(&outputs[i]);
    }
    Data ids(FLOAT32, {1, int(tokens.size())}, tokens);
    LastTokensManager last(states.size(), 64);
    std::vector<Data> dummy(states.size() * model.block_cnt * 2);
    std::vector<std::pair<Data*,Data*>> past;
    for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i+1]);
    const auto sampled = (model.*(&Access::ForwardSegments))(
        segments, ids, nullptr, nullptr, configs, last, sampleMode < 0 ? &pointers : nullptr, past);
    if (sampleMode >= 0) {
        Require(sampled.size() == states.size(), "missing sampled tokens");
        for (int token : sampled)
            Require(token >= 0 && token < model.weight["head.weight"].dims[0], "invalid sampled token");
        return std::vector<float>(sampled.begin(), sampled.end());
    }
    std::vector<float> result;
    for (auto &output : outputs) {
        Require(output.size() == size_t(model.weight["head.weight"].dims[0]), "missing logits");
        for (float value : output) Require(std::isfinite(value), "nonfinite logits");
        result.insert(result.end(), output.begin(), output.end());
    }
    return result;
}
static void CheckOwnership(const DeepSeekV41RequestState &state, int dp) {
    if (dp == 1) return;
    Require(state.attentionGroup >= 0 && state.attentionGroup < dp, "invalid owner");
    const int tp = 8 / dp;
    Require(state.attentionRanks.size() == size_t(tp), "KV duplicated outside attention TP group");
    for (int r = 0; r < tp; ++r) {
        const auto &local = *state.attentionRanks.at(r);
        Require(local.totalLen == state.totalLen, "request position mismatch");
        const int device = state.attentionGroup * tp + r;
        for (const auto &layer : local.layers) {
            Require(layer.totalLen == state.totalLen, "layer position mismatch");
            Require(layer.windowKV.cudaData != nullptr, "missing owner KV allocation");
            for (const Data *data : {&layer.windowKV, &layer.compressedKV, &layer.indexK}) {
                if (data->dims.empty() || !data->cudaData) continue;
                cudaPointerAttributes attrs{};
                Require(cudaPointerGetAttributes(&attrs, data->cudaData) == cudaSuccess, "invalid KV pointer");
                Require(attrs.device == device && !data->multiDeviceData, "KV on nonowner GPU");
            }
        }
    }
}
int main(int argc, char **argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8) return 77;
    try {
        Require(argc >= 4 && argc <= 11, "usage: regression FIXTURE_DIR ATTENTION_DP OUTPUT [fp4|fp8] [single-group] [reorder] [sampling] [graph|full-graph] [large-batch]");
        const int dp = std::stoi(argv[2]);
        bool fp4 = false, fp8 = false, singleGroup = false, reorder = false, graph = false, full = false;
        bool sampling = false, largeBatch = false;
        for (int i = 4; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "graph") graph = true;
            else if (option == "full-graph") full = graph = true;
            else if (option == "fp4") fp4 = true;
            else if (option == "fp8") fp8 = true;
            else if (option == "single-group") singleGroup = true;
            else if (option == "reorder") reorder = true;
            else if (option == "sampling") sampling = true;
            else if (option == "large-batch") largeBatch = true;
            else throw std::invalid_argument("unknown regression option");
        }
        SetCudaGraph(graph); SetThreads(2); SetCudaSharedExpert(true);
        SetDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}});
        SetMoeDeviceMap({{"multicuda:0,1,2,3,4,5,6,7", 1}});
        SetNgramDevice("cuda");
        auto base = CreateLLMModelFromHF(argv[1], DATA_AUTO_SOURCE);
        auto &model = dynamic_cast<DeepSeekV41Model&>(*base);
        if (fp4) model.kvCacheDataType = FP4_E2M1;
        if (fp8) model.kvCacheDataType = FP8_E4M3;
        model.ConfigureAttentionParallel(dp);
        Access::WarmGraphs(model, graph);
        std::ofstream file(argv[3], std::ios::binary);
        const std::vector<int> batches = largeBatch ? std::vector<int>{17, 32, 64, 128} :
                                                     std::vector<int>{1, 3, 9, 32};
        for (int batch : batches) {
            if (full) {
                // Compose only against disposable warmup KV. The real requests
                // below have different addresses, lengths, and ownership.
                Access::WarmGraphs(model, true);
                std::vector<std::shared_ptr<DeepSeekV41RequestState>> warm(batch);
                for (auto &state : warm) {
                    state = std::make_shared<DeepSeekV41RequestState>();
                    if (singleGroup && dp > 1) {
                        state->attentionGroup = 0;
                        state->attentionRanks.resize(8 / dp);
                    }
                    state->layers.resize(model.block_cnt);
                }
                Step(model, warm, std::vector<int>(batch, 33), 0, sampling ? 0 : -1);
                for (int i = 0; i < 4; ++i) Step(model, warm, std::vector<int>(batch, 1), i, sampling ? 0 : -1);
                std::vector<DeepSeekV41Segment> segments(batch);
                for (int i = 0; i < batch; ++i) {
                    segments[i].state = warm[i]; segments[i].startPos = warm[i]->totalLen;
                    segments[i].seqlen = 1; segments[i].offset = i;
                }
                (model.*(&Access::BuildFullDecodeGraph))(batch, segments, -1, -1);
                Access::WarmGraphs(model, false);
            }
            // Service warmup and serving run on different host threads. The
            // sampling mode also avoids priming a full-logit gather in warmup.
            auto exerciseRequests = [&] {
                std::vector<std::shared_ptr<DeepSeekV41RequestState>> states;
                std::vector<int> lengths;
                for (int i = 0; i < batch; ++i) {
                    auto state = std::make_shared<DeepSeekV41RequestState>();
                    if (singleGroup && dp > 1) {
                        state->attentionGroup = 0;
                        state->attentionRanks.resize(8 / dp);
                    }
                    state->layers.resize(model.block_cnt); states.push_back(state);
                    lengths.push_back(7 + (i % 3) * 13);
                }
                for (int step = 0; step < 9; ++step) {
                    if (reorder && (step == 4 || step == 6))
                        std::rotate(states.begin(), states.begin() + 1, states.end());
                    auto result = Step(model, states, lengths, step, sampling ? (step < 2 ? 0 : 1) : -1);
                    file.write(reinterpret_cast<const char*>(result.data()), result.size() * sizeof(float));
                    for (const auto &state : states) CheckOwnership(*state, dp);
                    lengths.assign(batch, 1);
                }
                // Restore the same snapshot onto another DP group and compare
                // continuation logits against the original request's live KV.
                auto snapshot = (model.*(&Access::SnapshotState))(*states[0], std::vector<int>(states[0]->totalLen, 3));
                Require(bool(snapshot), "prefix snapshot missing");
                auto restored = (model.*(&Access::RestoreState))(*snapshot, snapshot->totalLen);
                if (dp > 1) restored->attentionGroup = (states[0]->attentionGroup + 1) % dp;
                if (dp > 1) restored->attentionRanks.resize(8 / dp);
                auto expected = Step(model, {states[0]}, {1}, 77);
                auto actual = Step(model, {restored}, {1}, 77);
                Require(expected.size() == actual.size(), "restored logits shape");
                float maxError = 0;
                for (size_t i = 0; i < actual.size(); ++i)
                    maxError = std::max(maxError, std::fabs(expected[i] - actual[i]));
                if (maxError > 1e-5f) std::cerr << "prefix batch=" << batch << " max_error=" << maxError << '\n';
                Require(maxError <= 1e-5f, "prefix restore changed logits");
                CheckOwnership(*restored, dp);
                std::cout << "PASS DP" << dp << " batch=" << batch << " mixed prefill, decode, KV ownership, prefix relocation\n";
            };
            if (sampling) std::async(std::launch::async, exerciseRequests).get();
            else exerciseRequests();
        }
        std::vector<std::pair<Data, Data>> past(model.block_cnt);
        auto reused = (model.*static_cast<std::shared_ptr<DeepSeekV41RequestState> (DeepSeekV41Model::*)(
            std::vector<std::pair<Data, Data>>&, bool)>(&Access::GetOrCreateState))(past, false);
        Step(model, {reused}, {7}, 19);
        auto reset = (model.*static_cast<std::shared_ptr<DeepSeekV41RequestState> (DeepSeekV41Model::*)(
            std::vector<std::pair<Data, Data>>&, bool)>(&Access::GetOrCreateState))(past, true);
        Require(reset == reused && reset->attentionGroup == -1 && reset->attentionRanks.empty(),
                "request reset retained attention placement or KV");
        Step(model, {reset}, {7}, 19);
        CheckOwnership(*reset, dp);
        std::cout << "PASS request handle reset and placement reuse\n";
        Require(bool(file), "cannot write logits");
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
