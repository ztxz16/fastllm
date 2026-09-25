#include "models/qwen4_exp.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace fastllm {
// Supply small, valid CPU KV/QSA state to exercise the actual snapshot store
// without loading model weights or running the visual encoder.
struct Qwen4PrefixCacheTestAccess {
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
    SetDeviceMap({{"cpu", 1}});
    const int length = std::max(2, GetPageLen() * 2);
    int failures = 0;
    auto check = [&](bool condition, const char *message) {
        if (!condition) { std::cerr << message << '\n'; ++failures; }
    };

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
