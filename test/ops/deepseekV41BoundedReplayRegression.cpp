// Run via test/basic/test_deepseek_v41_bounded_replay.py (no GPU required).
#include "model.h"
#include "models/deepseekv41.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
// Expose member pointers for this test without changing the model's public API.
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::ForwardSegments;
    using DeepSeekV41Model::SnapshotState;
    using DeepSeekV41Model::RestoreState;
    using DeepSeekV41Model::decoderSwaTailLayer;
    using DeepSeekV41Model::v41DsparkEnabled;
    using DeepSeekV41Model::v41DsparkTargetLayerIds;
    using DeepSeekV41Model::v41IsDsparkTarget;
    using DeepSeekV41Model::quantizedLinearNames;
};
static int checks = 0;
static void Check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
    checks++;
}
static std::vector<float> Read(Data &data) {
    if (data.dims.empty()) return {};
    Data copy;
    copy.CopyFrom(data);
    if (copy.dataType == DataType::INT8) {
        copy.ToDevice(DataDevice::CPU);
        return {(uint8_t *)copy.cpuData, (uint8_t *)copy.cpuData + copy.Count(0)};
    }
    ToDataType(copy, DataType::FLOAT32);
    copy.ToDevice(DataDevice::CPU);
    return {(float *)copy.cpuData, (float *)copy.cpuData + copy.Count(0)};
}
static void Near(const std::vector<float> &a, const std::vector<float> &b) {
    Check(a.size() == b.size(), "tensor size mismatch");
    for (size_t i = 0; i < a.size(); i++) {
        const bool ok = std::isfinite(a[i]) && std::isfinite(b[i]) && std::fabs(a[i] - b[i]) < 1e-4f;
        if (!ok)
            std::cerr << "tensor index " << i << ": " << a[i] << " vs " << b[i] << '\n';
        Check(ok, "tensor value mismatch");
    }
}
using State = std::shared_ptr<DeepSeekV41RequestState>;
static State NewState(int layers) {
    auto state = std::make_shared<DeepSeekV41RequestState>();
    state->layers.resize(layers);
    return state;
}
static void SameState(State a, State b) {
    Check(a->totalLen == b->totalLen && a->engramHistory == b->engramHistory, "history mismatch");
    for (size_t i = 0; i < a->layers.size(); i++) {
        auto &x = a->layers[i]; auto &y = b->layers[i];
        Check(x.totalLen == a->totalLen && y.totalLen == b->totalLen, "layer position mismatch");
        Check(x.compressedBlocks == y.compressedBlocks && x.rawTail == y.rawTail, "compressed length mismatch");
        auto wx = Read(x.windowKV), wy = Read(y.windowKV);
        if (a->totalLen < x.windowKV.dims[1]) {
            wx.resize(a->totalLen * x.windowKV.dims[2]);
            wy.resize(b->totalLen * y.windowKV.dims[2]);
        }
        Near(wx, wy);
        Near(Read(x.compressedKV), Read(y.compressedKV));
        Near(Read(x.indexK), Read(y.indexK));
        if (x.rawTail > 0) Near(Read(x.rawTailKV), Read(y.rawTailKV));
    }
}
static std::vector<std::vector<float>> Forward(DeepSeekV41Model &model, const std::vector<State> &states,
                                              const std::vector<int> &lengths, bool bounded, bool verify = false) {
    model.*(&Access::decoderSwaTailLayer) = bounded ? 4 : -1;
    const int batch = (int)states.size();
    std::vector<DeepSeekV41Segment> segments(batch);
    std::vector<DeepSeekV41SpecScratch> scratch(batch);
    std::vector<float> values;
    std::vector<GenerationConfig> configs(batch);
    std::vector<std::vector<float>> logits(batch);
    std::vector<std::vector<float> *> outputs;
    std::vector<Data> dummy(batch * model.block_cnt * 2);
    std::vector<std::pair<Data *, Data *>> past;
    for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i + 1]);
    for (int s = 0; s < batch; s++) {
        auto &seg = segments[s];
        seg.state = states[s]; seg.startPos = states[s]->totalLen;
        seg.seqlen = lengths[s]; seg.offset = (int)values.size();
        scratch[s].captureMain = true;
        scratch[s].wantAllTokens = verify;
        seg.spec = &scratch[s];
        for (int i = 0; i < lengths[s]; i++) values.push_back((float)((seg.startPos + i) * 17 % 126));
        configs[s].output_logits = !verify;
        outputs.push_back(&logits[s]);
    }
    Data ids(DataType::FLOAT32, {1, (int)values.size()}, values);
    LastTokensManager last(batch, 64);
    (model.*(&Access::ForwardSegments))(segments, ids, nullptr, nullptr, configs, last, &outputs, past);
    for (int s = 0; s < batch; s++) {
        Check(segments[s].seqlen == lengths[s], "caller segment was truncated");
        const int rows = bounded && !verify ? std::min(8, lengths[s]) : lengths[s];
        const int start = segments[s].startPos + lengths[s] - rows;
        Check(states[s]->totalLen == segments[s].startPos + lengths[s], "logical length was truncated");
        for (auto &cache : states[s]->layers) Check(cache.totalLen == states[s]->totalLen, "KV position drift");
        Check(scratch[s].mainHidden.size() == 2, "missing DSpark features");
        for (auto &hidden : scratch[s].mainHidden) Check(hidden.dims[1] == rows, "DSpark feature rows mismatch");
        Check(scratch[s].mainHiddenStartPos == (rows < lengths[s] ? start : -1), "DSpark feature position mismatch");
        if (verify) Check((int)scratch[s].tokens.size() == lengths[s], "verify positions were truncated");
    }
    return logits;
}

static void CompareForward(DeepSeekV41Model &model, const std::vector<State> &full,
                           const std::vector<State> &bounded, const std::vector<int> &lengths) {
    auto expected = Forward(model, full, lengths, false);
    auto actual = Forward(model, bounded, lengths, true);
    for (size_t s = 0; s < full.size(); s++) {
        Near(actual[s], expected[s]);
        SameState(bounded[s], full[s]);
    }
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: deepseekV41BoundedReplayRegression FIXTURE_DIR");
        SetThreads(2);
        SetDeviceMap({{"cpu", 1}});
        SetMoeDeviceMap({{"cpu", 1}});
        auto base = CreateLLMModelFromHF(argv[1], DataType::FLOAT32);
        auto &model = dynamic_cast<DeepSeekV41Model &>(*base);
        Check(model.*(&Access::decoderSwaTailLayer) == 4 && model.block_cnt == 5, "fixture/flag mismatch");
        // Capture targets on both sides of the tail boundary without requiring draft weights.
        model.*(&Access::v41DsparkEnabled) = true;
        model.*(&Access::v41DsparkTargetLayerIds) = {3, 4};
        model.*(&Access::v41IsDsparkTarget) = {0, 0, 0, 1, 1};

        // This fixture has one late layer: its last query and all stored KV must remain
        // equivalent to full prefill, even with Engram and an indexer in that late layer.
        for (int length : {1, 7, 8, 9, 17}) {
            std::cout << "prefill " << length << std::endl;
            auto a = NewState(5), b = NewState(5);
            CompareForward(model, {a}, {b}, {length});
            for (int chunk : {5, 17, 1, 9, 1}) {
                std::cout << "extend " << a->totalLen << " + " << chunk << std::endl;
                CompareForward(model, {a}, {b}, {chunk});
            }
        }
        // Each request keeps its own offset, position and window in mixed prefill/decode.
        std::vector<State> batched{NewState(5), NewState(5)}, full{NewState(5), NewState(5)};
        for (auto lengths : {std::vector<int>{17, 5}, {1, 19}, {11, 1}}) {
            std::cout << "mixed " << lengths[0] << ", " << lengths[1] << std::endl;
            CompareForward(model, full, batched, lengths);
        }
        // Prefix-cache restoration preserves the window's absolute slot positions.
        std::vector<int> history(batched[0]->totalLen);
        for (size_t i = 0; i < history.size(); i++) history[i] = (int)i * 17 % 126;
        auto snapshot = (model.*(&Access::SnapshotState))(*batched[0], history);
        auto restored = (model.*(&Access::RestoreState))(*snapshot, snapshot->totalLen);
        Near(Forward(model, {batched[0]}, {13}, true)[0], Forward(model, {restored}, {13}, true)[0]);
        SameState(batched[0], restored);
        Forward(model, {NewState(5)}, {13}, true, true); // verify larger than the window
        for (DataType dtype : {DataType::FP8_E4M3, DataType::FP4_E2M1}) {
            model.kvCacheDataType = dtype;
            auto a = NewState(5), b = NewState(5);
            for (int length : {17, 11, 1}) {
                CompareForward(model, {a}, {b}, {length});
            }
        }
        // Exercise quantized activation scratch reuse across layers and its
        // release at the tail boundary using the existing unquantized fixture.
        for (int layer = 0; layer < model.block_cnt; layer++) {
            for (const char *suffix : {".attn.wq_a.weight", ".attn.wq_b.weight", ".attn.wkv.weight",
                                       ".attn.wo_b.weight", ".ffn.shared_experts.gateup.weight",
                                       ".ffn.shared_experts.w2.weight"}) {
                (model.*(&Access::quantizedLinearNames)).insert("layers." + std::to_string(layer) + suffix);
            }
        }
        auto a = NewState(5), b = NewState(5);
        for (int length : {40, 17, 1}) CompareForward(model, {a}, {b}, {length});
        std::cout << "PASS: bounded prefill, mixed batch, cache restore, DSpark capture/verify ("
                  << checks << " checks)\n";
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
