#include "models/qwen4_exp.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace fastllm {
// Supply small, valid CPU KV/QSA state to exercise the actual snapshot store
// without loading model weights or running the visual encoder.
struct Qwen4PrefixCacheTestAccess {
    static bool TestNumaWeightsPrepared() {
#ifdef USE_NUMAS
        Qwen4ExpModel model;
        model.block_cnt = 1;
        model.num_experts = 2;
        model.linearLayers = {true};
        model.pleLayer = -1;
        model.deviceMap = {{"cpu", 1}};
        model.moeDeviceMap = {{"numa", 1}};
        std::vector<Data *> experts;
        for (int e = 0; e < model.num_experts; ++e) {
            const std::string prefix = model.languagePrefix +
                "layers.0.mlp.experts." + std::to_string(e) + ".";
            for (const auto &part : {std::make_pair("gateup_proj.weight", "linearSwiglu"),
                                     std::make_pair("down_proj.weight", "linearColumn")}) {
                const std::string name = prefix + part.first;
                Data &weight = model.weight[name];
                weight.CopyFrom(Data(FLOAT32, {64, 16}, std::vector<float>(1024, 1.0f)));
                model.moeLinears.insert(name);
                model.AddSpecialWeight(name, part.second, 0);
                experts.push_back(&weight);
            }
        }
        // No Forward/AutoWarmup or GPU expert cache: source buffers must
        // already be released before a request can visit individual experts.
        model.PrepareWeights();
        for (Data *weight : experts) {
            if (weight->cpuData != nullptr || weight->numasData.empty()) return false;
            for (const uint8_t *shard : weight->numasData)
                if (!shard || reinterpret_cast<const float *>(shard)[0] != 1.0f) return false;
        }
        const auto shards = experts[0]->numasData;
        model.PrepareWeights();
        return experts[0]->numasData == shards;
#else
        return true;
#endif
    }

    static void Configure(Qwen4ExpModel &model) {
        model.block_cnt = model.embed_dim = 1;
        model.deviceMap = {{"cpu", 1}};
        model.linearLayers = {false};
        model.indexerKvHeads = model.indexerHeadDim = 1;
        model.indexerCompressRatio = 2;
    }

    static void Fill(Qwen4ExpModel &model, ResponseContext &context,
                     int length, int firstToken) {
        context.allTokens.resize(length);
        std::iota(context.allTokens.begin(), context.allTokens.end(), firstToken);
        std::vector<float> values(length, 3.0f);
        for (Data *data : {&context.pastKeyValues[0].first,
                          &context.pastKeyValues[0].second}) {
            data->CopyFrom(Data(FLOAT32, {1, length, 1}, values));
        }
        auto &state = model.requestStates[&context.pastKeyValues[0].first];
        state.processedTokens = context.allTokens;
        state.indexerRawKeys[0] = values;
        state.indexerPositions[0].resize(length);
        std::iota(state.indexerPositions[0].begin(), state.indexerPositions[0].end(), 0.0f);
        state.indexerBlockKeys[0].assign(length / 2, 3.0f);
    }

    static size_t Record(Qwen4ExpModel &model,
                         const std::vector<std::pair<Data, Data>> &kv) {
        model.MaybeRecordPrefixSnapshot(kv, model.requestStates[&kv[0].first]);
        return model.prefixSnapshots.size();
    }

    static bool RestoreFirst(Qwen4ExpModel &model, ResponseContext &context) {
        return model.RestorePrefixSnapshot(&context, model.prefixSnapshots.at(0));
    }

    static bool TestRankSnapshots(int length) {
        Qwen4ExpModel rank0, rank1;
        Configure(rank0);
        Configure(rank1);
        rank0.threadTpRank = 0;
        rank1.threadTpRank = 1;
        struct ResetRanks {
            int &a, &b;
            ~ResetRanks() { a = b = -1; }
        } resetRanks{rank0.threadTpRank, rank1.threadTpRank};
        ResponseContext a, b;
        a.Init(1, FLOAT32, FLOAT32);
        b.Init(1, FLOAT32, FLOAT32);
        Fill(rank0, a, length, 1200);
        Fill(rank1, b, length, 1200);
        // The shards deliberately differ: restoring rank 0 on both ranks
        // would pass token-boundary checks but silently corrupt attention.
        reinterpret_cast<float *>(b.pastKeyValues[0].second.cpuData)[0] = 7.0f;
        auto &state1 = rank1.requestStates[&b.pastKeyValues[0].first];
        state1.previousToken1 = 91;
        state1.convHistory = {8.0f, 9.0f};
        if (Record(rank0, a.pastKeyValues) != 1) return false;
        const std::vector<Qwen4ExpModel *> models{&rank0, &rank1};
        auto tokens = a.allTokens;
        tokens.push_back(99);
        if (Qwen4ExpModel::FindCommonPrefixSnapshot(models, tokens, length)) return false;
        if (Record(rank1, b.pastKeyValues) != 1) return false;
        auto pinned = Qwen4ExpModel::FindCommonPrefixSnapshot(models, tokens, length);
        if (!pinned || pinned->ranks.size() != 2 || pinned->cachedLen != length) return false;

        // A newer snapshot on one rank must fall back to the older complete
        // set. Divergent token histories must never match by length alone.
        Fill(rank0, a, length * 2, 1200);
        if (Record(rank0, a.pastKeyValues) != 2) return false;
        auto common = Qwen4ExpModel::FindCommonPrefixSnapshot(models, a.allTokens, length * 2);
        if (!common || common->cachedLen != length) return false;
        auto divergent = a.allTokens;
        divergent[0] = -1;
        if (Qwen4ExpModel::FindCommonPrefixSnapshot(models, divergent, length * 2)) return false;
        rank1.prefixSnapshots.clear();
        if (Qwen4ExpModel::FindCommonPrefixSnapshot(models, tokens, length)) return false;

        // Pending hits pin every shard even if the store evicts them. A new
        // request receives independent writable CPU tensors and PLE history.
        ResponseContext restored0, restored1, again;
        for (auto *context : {&restored0, &restored1, &again}) context->Init(1, FLOAT32, FLOAT32);
        if (!rank0.RestorePrefixSnapshot(restored0.pastKeyValues, GenerationConfig(), pinned->ranks[0]) ||
            !rank1.RestorePrefixSnapshot(restored1.pastKeyValues, GenerationConfig(), pinned->ranks[1])) return false;
        if (reinterpret_cast<float *>(restored0.pastKeyValues[0].second.cpuData)[0] != 3.0f ||
            reinterpret_cast<float *>(restored1.pastKeyValues[0].second.cpuData)[0] != 7.0f) return false;
        auto &restoredState = rank1.requestStates[&restored1.pastKeyValues[0].first];
        if (restoredState.previousToken1 != 91 || restoredState.convHistory != std::vector<float>({8, 9}) ||
            restoredState.indexerRawKeys[0].size() != (size_t)length) return false;
        reinterpret_cast<float *>(restored1.pastKeyValues[0].second.cpuData)[0] = -1.0f;
        if (!rank1.RestorePrefixSnapshot(again.pastKeyValues, GenerationConfig(), pinned->ranks[1]) ||
            reinterpret_cast<float *>(again.pastKeyValues[0].second.cpuData)[0] != 7.0f) return false;

        rank1.autoWarmupRunning.store(true);
        Fill(rank1, b, length * 3, 1200);
        if (Record(rank1, b.pastKeyValues) != 0) return false;
        rank1.autoWarmupRunning.store(false);
        state1.hasMultimodalInput = true;
        if (Record(rank1, b.pastKeyValues) != 0) return false;
        return true;
    }
};
}

using namespace fastllm;

class DirectDecodeFixture : public Qwen4ExpModel {
public:
    size_t records = 0;
    int Forward(const Data &, const Data &, const Data &,
                std::vector<std::pair<Data, Data>> &kv,
                const GenerationConfig &, const LastTokensManager &,
                std::vector<float> *) override {
        records = Qwen4PrefixCacheTestAccess::Record(*this, kv);
        return 7;
    }
};

static void SetEnv(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    SetEnv("FASTLLM_PREFIX_CACHE", "1");
    SetEnv("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", "1");
    SetEnv("FASTLLM_QWEN4_ENABLE_MTP", "0");
    SetEnv("FT_NUMAS", "1");
    SetMoeCudaCacheBytes(0);
    SetDeviceMap({{"cpu", 1}});
    const int length = std::max(2, GetPageLen() * 2);
    int failures = 0;
    auto check = [&](bool condition, const char *message) {
        if (!condition) { std::cerr << message << '\n'; ++failures; }
    };
    check(Qwen4PrefixCacheTestAccess::TestRankSnapshots(length),
          "TP snapshot completeness, shard isolation or eviction recovery failed");
    check(Qwen4PrefixCacheTestAccess::TestNumaWeightsPrepared(),
          "NUMA expert source storage survived preparation without a GPU cache");

    Qwen4ExpModel model;
    Qwen4PrefixCacheTestAccess::Configure(model);
    ResponseContext text;
    text.Init(1, FLOAT32, FLOAT32);
    model.OnResponseContextCreated(&text);
    Qwen4PrefixCacheTestAccess::Fill(model, text, length, 11);
    check(Qwen4PrefixCacheTestAccess::Record(model, text.pastKeyValues) == 1,
          "text snapshot was not recorded");

    ResponseContext restored;
    restored.Init(1, FLOAT32, FLOAT32);
    restored.allTokens = text.allTokens;
    restored.allTokens.push_back(99);
    restored.currentTokens = restored.allTokens;
    check(model.TryRestoreHistoryCache(restored.currentTokens, restored.cacheLen),
          "text snapshot lookup failed");
    model.OnResponseContextCreated(&restored);
    check(restored.cacheLen == length && restored.currentTokens == std::vector<int>{99} &&
          restored.pastKeyValues[0].second.dims == std::vector<int>({1, length, 1}) &&
          restored.pastKeyValues[0].second.cpuData != nullptr &&
          reinterpret_cast<float *>(restored.pastKeyValues[0].second.cpuData)[0] == 3.0f,
          "text snapshot KV was not restored");

    int firstToken = 211;
    for (const char *media : {"image_frames", "video_frames"}) {
        ResponseContext context;
        context.Init(1, FLOAT32, FLOAT32);
        context.multimodalInput[media] = {new Data(FLOAT32)};
        model.OnResponseContextCreated(&context);
        Qwen4PrefixCacheTestAccess::Fill(model, context, length, firstToken);
        const size_t before = Qwen4PrefixCacheTestAccess::Record(model, text.pastKeyValues);
        check(Qwen4PrefixCacheTestAccess::Record(model, context.pastKeyValues) == before,
              "media prefill polluted the independent snapshot store");
        context.cacheLen = length;
        check(!Qwen4PrefixCacheTestAccess::RestoreFirst(model, context),
              "media context accepted a text snapshot");
        context.cacheLen = 0;

        // Simulate later decode with only the persistent request state left.
        delete context.multimodalInput[media][0];
        context.multimodalInput.clear();
        Qwen4PrefixCacheTestAccess::Fill(model, context, length * 2, firstToken);
        check(Qwen4PrefixCacheTestAccess::Record(model, context.pastKeyValues) == before,
              "media decode resumed token-only snapshot recording");

        // Reusing storage for a new text request must not retain the media flag.
        model.OnResponseContextRemoved(&context);
        model.OnResponseContextCreated(&context);
        Qwen4PrefixCacheTestAccess::Fill(model, context, length * 3, firstToken + 200);
        check(Qwen4PrefixCacheTestAccess::Record(model, context.pastKeyValues) > before,
              "new text request retained the previous media restriction");
        model.OnResponseContextRemoved(&context);
        firstToken += 1000;
    }
    model.OnResponseContextRemoved(&text);
    model.OnResponseContextRemoved(&restored);

    DirectDecodeFixture direct;
    Qwen4PrefixCacheTestAccess::Configure(direct);
    ResponseContext context;
    context.Init(1, FLOAT32, FLOAT32);
    Qwen4PrefixCacheTestAccess::Fill(direct, context, length, 611);
    context.multimodalInput["image_frames"] = {new Data(FLOAT32)};
    Data token(FLOAT32, {1, 1}, {99}), position(FLOAT32, {1, 1}, {(float)length}), mask;
    const auto result = direct.ForwardMultimodal(
        token, mask, position, context.pastKeyValues, context.multimodalInput);
    check(result == std::vector<int>{7} && direct.records == 0,
          "direct multimodal decode bypassed snapshot isolation");
    direct.OnResponseContextRemoved(&context);
    if (failures) return 1;
    std::cout << "Qwen4 multimodal snapshot isolation and text reuse: PASS\n";
    return 0;
}
