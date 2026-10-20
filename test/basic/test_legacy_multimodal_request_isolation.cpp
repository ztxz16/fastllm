#include "models/basellm.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
using namespace fastllm;

// Leave the oldest completed handle undrained while two newer requests run.
// The selected KV, media and sampling config must all belong to the same handle.
class IsolationFixture : public basellm {
    std::map<const void *, int> tags;
    std::mutex gate;
    std::condition_variable cv;
    bool entered = false, released = false;
public:
    std::atomic<int> calls{0}, failures{0};
    IsolationFixture() {
        block_cnt = 1; kvCacheId = 0; maxBatch = 1; tokensLimit = 4096;
        max_positions = 512; dataType = kvCacheDataType = FLOAT32;
        eos_token_id = -100; use_new_engine = false;
        canDoBatchForward = false; canDoConcurrentForward = true;
        SetSaveHistoryChat(false);
    }
    ~IsolationFixture() override { ShutdownRuntime(); }
    bool NeedAttentionMask(int, int) override { return false; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    void OnResponseContextCreated(ResponseContext *ctx) override {
        std::lock_guard<std::mutex> lock(gate);
        tags[&ctx->pastKeyValues] = ctx->inputTokens;
    }
    bool WaitFirst() {
        std::unique_lock<std::mutex> lock(gate);
        return cv.wait_for(lock, std::chrono::seconds(5), [&]{return entered;});
    }
    void Release() { std::lock_guard<std::mutex> lock(gate); released = true; cv.notify_all(); }
    int Run(const Data &ids, std::vector<std::pair<Data, Data>> &kv,
            const GenerationConfig &cfg, int mediaTag) {
        int tag;
        {
            std::unique_lock<std::mutex> lock(gate);
            tag = tags.at(&kv);
            if (!entered) { entered = true; cv.notify_all(); cv.wait(lock, [&]{return released;}); }
        }
        const int expectedMedia = tag == 13 ? 0 : tag;
        if (mediaTag != expectedMedia || cfg.top_k != tag) {
            ++failures;
            std::cerr << "request=" << tag << " media=" << mediaTag << " top_k=" << cfg.top_k << '\n';
        }
        for (Data *cache : {&kv[0].first, &kv[0].second}) {
            cache->Resize({1, (int)ids.Count(0), 1}); cache->expansionDims = {1, 512, 1};
        }
        ++calls;
        return 100 + tag;
    }
    int Forward(const Data &ids, const Data &, const Data &,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &cfg,
            const LastTokensManager &, std::vector<float> *) override {
        return Run(ids, kv, cfg, 0);
    }
    std::vector<int> ForwardMultimodal(const Data &ids, const Data &, const Data &,
            std::vector<std::pair<Data, Data>> &kv,
            const std::map<std::string, std::vector<Data *>> &media,
            const GenerationConfig &cfg, const LastTokensManager &,
            std::vector<std::vector<float> *> *) override {
        int tag = (int)((float *)media.at("tag")[0]->cpuData)[0];
        return {Run(ids, kv, cfg, tag)};
    }
};

int main() {
    SetDeviceMap({{"cpu", 1}}); SetMaxTokens(0);
    IsolationFixture model;
    auto launch = [&](int tag) {
        GenerationConfig cfg; cfg.output_token_limit = 1; cfg.add_special_tokens = false; cfg.top_k = tag;
        std::map<std::string, std::vector<Data *>> media;
        if (tag != 13) media["tag"] = {new Data(FLOAT32, std::vector<int>{1}, std::vector<float>{(float)tag})};
        return model.LaunchResponseTokens(std::vector<int>(tag, 1), cfg, media);
    };
    std::vector<int> handles{launch(11)};
    if (!model.WaitFirst()) { model.Release(); std::cerr << "first request stalled\n"; return 1; }
    handles.push_back(launch(12)); handles.push_back(launch(13)); model.Release();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (model.calls < 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (model.calls != 3) { std::cerr << "queued requests stalled\n"; return 1; }
    for (int i = 0; i < 3; ++i) {
        if (model.FetchResponseTokens(handles[i]) != 111 + i || model.FetchResponseTokens(handles[i]) != -1)
            ++model.failures;
    }
    if (model.failures) return 1;
    std::cout << "Queued multimodal/text request isolation: PASS\n";
}
