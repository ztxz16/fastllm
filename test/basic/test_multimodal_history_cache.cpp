#include "models/basellm.h"

#include <iostream>

using namespace fastllm;

// Exercise the real history store and request initialization without inference.
class CacheFixture : public basellm {
public:
    int historyRecords = 0, historyRestores = 0, pagedRecords = 0;
    CacheFixture() {
        block_cnt = 1;
        dataType = kvCacheDataType = FLOAT32;
        saveHistoryChat = true;
    }
    ~CacheFixture() {
        ShutdownRuntime();
        for (auto &entry : pastKVCacheManager.memorys) delete entry.second;
        pastKVCacheManager.memorys.clear();
    }
    std::string MakeInput(const std::string &, int, const std::string &) override { return ""; }
    std::string MakeHistory(const std::string &, int, const std::string &,
                            const std::string &) override { return ""; }
    bool UseModelSpecificScheduler() const override { return true; }
    void RunModelSpecificScheduler() override {}
    void TryRecordHistoryCache(const std::vector<int> &) override { ++historyRecords; }
    bool TryRestoreHistoryCache(std::vector<int> &, int &) override {
        ++historyRestores;
        return false;
    }
    bool TryRecordPagedPrefixCacheExtra(ResponseContext *) override {
        ++pagedRecords;
        return false;
    }
};

int main() {
    SetDeviceMap({{"cpu", 1}});
    CacheFixture model;
    int failures = 0;
    auto check = [&](bool ok, const char *message) {
        if (!ok) { std::cerr << message << '\n'; ++failures; }
    };
    ResponseContext seed;
    seed.Init(1, FLOAT32, FLOAT32);
    seed.allTokens = {11, 12, 13, 14};
    for (Data *cache : {&seed.pastKeyValues[0].first, &seed.pastKeyValues[0].second}) {
        cache->CopyFrom(Data(FLOAT32, {1, 4, 1}, {1, 2, 3, 4}));
    }
    seed.multimodalInput["image_frames"] = {new Data(FLOAT32)};
    seed.TryRecord(&model);
    seed.TryRecordPagedCache(&model);
    check(model.pastKVCacheManager.memorys.empty() && model.historyRecords == 0,
          "media state polluted token-only history");
    check(model.pagedRecords == 0, "media state reached paged prefix snapshot recording");
    // Remove any baseline failure's entry so text behavior can be checked too.
    for (auto &entry : model.pastKVCacheManager.memorys) delete entry.second;
    model.pastKVCacheManager.memorys.clear();
    delete seed.multimodalInput["image_frames"][0];
    seed.multimodalInput.clear();
    model.historyRecords = model.pagedRecords = 0;
    seed.TryRecord(&model);
    seed.TryRecordPagedCache(&model);
    check(model.pastKVCacheManager.memorys.size() == 1 && model.historyRecords == 1 &&
          model.pagedRecords == 1, "text prefix recording stopped working");

    std::vector<int> prompt = {11, 12, 13, 14, 15};
    int text = model.LaunchResponseTokens(prompt, GenerationConfig());
    auto *textContext = model.responseContextDict.GetHandle(text);
    check(textContext->cacheLen == 4 && textContext->currentTokens == std::vector<int>{15},
          "text prefix was not restored");
    model.historyRestores = 0;
    int media = model.LaunchResponseTokens(prompt, GenerationConfig(),
                                          {{"image_frames", {new Data(FLOAT32)}}});
    auto *mediaContext = model.responseContextDict.GetHandle(media);
    check(mediaContext->cacheLen == 0 && mediaContext->currentTokens == prompt &&
          mediaContext->pastKeyValues[0].second.dims.empty() && model.historyRestores == 0,
          "media request inherited token-only history instead of a full visual prefill");
    if (failures) return 1;
    std::cout << "Multimodal history isolation and text reuse: PASS\n";
    return 0;
}
