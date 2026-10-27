#include "models/basellm.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

// Use the real legacy scheduler and sampling routine. A separate rolling
// history checks both sampled tokens and the penalty-adjusted logits.
class Fixture : public basellm {
    std::map<const void *, LastTokensUnit> expectedHistory;
public:
    explicit Fixture(bool fastInput = false) {
        if (fastInput) model_struct = "naive_n05_flash";
        block_cnt = 1; kvCacheId = 0; maxBatch = 4; tokensLimit = 4096;
        max_positions = 512; dataType = kvCacheDataType = FLOAT32;
        eos_token_id = -100; use_new_engine = false;
        canDoBatchForward = false; canDoConcurrentForward = true;
        SetSaveHistoryChat(false);
    }
    ~Fixture() override { ShutdownRuntime(); }
    bool NeedAttentionMask(int, int) override { return false; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    int Forward(const Data &ids, const Data &, const Data &positions,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &cfg,
            const LastTokensManager &history, std::vector<float> *retLogits) override {
        auto &expected = expectedHistory[&kv];
        int past = kv[0].first.dims.empty() ? 0 : kv[0].first.dims[1];
        if (!past) expected.Init(cfg.last_n);
        Check(((float *)positions.cpuData)[0] == past, "position changed");
        Check(history.units.size() == 1 && history.units[0].tot == cfg.last_n, "history slot mismatch");
        Data actual(FLOAT32, {1, 8}, {0.1f, 1.f, .9f, .8f, .7f, .6f, .5f, .4f});
        Data golden(actual);
        int a = LLMSampling(actual, 0, cfg, history.units[0]);
        int b = LLMSampling(golden, 0, cfg, expected);
        Check(a == b, "sampling changed after history copy optimization");
        for (int i = 0; i < 8; ++i)
            Check(((float *)actual.cpuData)[i] == ((float *)golden.cpuData)[i], "penalty logits changed");
        if (retLogits) retLogits->assign((float *)actual.cpuData, (float *)actual.cpuData + 8);
        expected.Push(a);
        for (Data *cache : {&kv[0].first, &kv[0].second}) {
            cache->Resize({1, past + ids.Count(0), 1}); cache->expansionDims = {1, 512, 1};
        }
        return a;
    }
};
int main() {
    try {
        SetDeviceMap({{"cpu", 1}}); SetMaxTokens(0);
        for (bool fastInput : {false, true}) {
        Fixture model(fastInput);
        for (int repeat = 0; repeat < 2; ++repeat) {
            std::vector<int> handles, counts(4), done(4);
            for (int i = 0; i < 4; ++i) {
                GenerationConfig cfg;
                cfg.output_token_limit = 32; cfg.add_special_tokens = false;
                cfg.repeat_penalty = i == 0 ? 1.f : i == 1 ? 1.2f : i == 2 ? .8f : 1.f;
                cfg.last_n = i == 1 ? 0 : 4;
                cfg.output_logits = i == 3;
                handles.push_back(model.LaunchResponseTokens(std::vector<int>(16 + i, 1), cfg));
            }
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (done[0] + done[1] + done[2] + done[3] < 4) {
                Check(std::chrono::steady_clock::now() < deadline, "scheduler stalled");
                for (int i = 0; i < 4; ++i) {
                    if (done[i] || !model.CanFetchResponse(handles[i])) continue;
                    int token = model.FetchResponseTokens(handles[i]);
                    if (token < 0) { Check(token == -1, "request failed"); done[i] = 1; }
                    else ++counts[i];
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            for (int n : counts) Check(n == 32, "request ended early");
        }
        }
        std::cout << "ALL_PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
