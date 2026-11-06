#include "models/basellm.h"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <cstdlib>
#include <cmath>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

// Use the real shared schedulers and sampling routine. A separate rolling
// history checks both sampled tokens and the penalty-adjusted logits.
class Fixture : public basellm {
    std::map<const void *, LastTokensUnit> expectedHistory;
    bool gpuDispatch;
public:
    explicit Fixture(int engine, bool fastInput = false) : gpuDispatch(engine == 2) {
        if (fastInput) model_struct = "naive_n05_flash";
        block_cnt = 1; kvCacheId = 0; maxBatch = 4; tokensLimit = 4096;
        max_positions = 512; dataType = kvCacheDataType = FLOAT32;
        eos_token_id = -100; use_new_engine = engine != 0;
        canDoBatchForward = false; canDoConcurrentForward = true;
        SetSaveHistoryChat(false);
    }
    ~Fixture() override { ShutdownRuntime(); }
    bool NeedAttentionMask(int, int) override { return false; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    bool CanUseGPUForward() const override { return gpuDispatch; }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    int Step(int inputLength, const Data &positions, Data &key, Data &value,
            const GenerationConfig &cfg, const LastTokensUnit &history, std::vector<float> *retLogits) {
        auto &expected = expectedHistory[&key];
        int past = key.dims.empty() ? 0 : key.dims[1];
        if (!past) expected.Init(cfg.last_n);
        Check(((float *)positions.cpuData)[0] == past, "position changed");
        Check(history.tot == cfg.last_n, "history slot mismatch");
        if (cfg.repeat_penalty == 1.0f)
            Check(history.tokenSet.empty() && history.tokenQueue.empty(),
                  "unused repetition history was retained");
        else
            Check(history.tokenSet == expected.tokenSet && history.tokenQueue == expected.tokenQueue,
                  "repetition history lost or assigned to the wrong batch row");
        Data actual(FLOAT32, {1, 8}, {0.1f, 1.f, .9f, .8f, .7f, .6f, .5f, .4f});
        Data golden(actual);
        std::srand(1337 + past);
        int a = LLMSampling(actual, 0, cfg, history);
        std::srand(1337 + past);
        int b = LLMSampling(golden, 0, cfg, expected);
        Check(a == b, "sampling changed after history copy optimization");
        for (int i = 0; i < 8; ++i)
            Check(((float *)actual.cpuData)[i] == ((float *)golden.cpuData)[i], "penalty logits changed");
        if (retLogits) retLogits->assign((float *)actual.cpuData, (float *)actual.cpuData + 8);
        expected.Push(a);
        for (Data *cache : {&key, &value}) {
            cache->Resize({1, past + inputLength, 1}); cache->expansionDims = {1, 512, 1};
        }
        return a;
    }
    int Forward(const Data &ids, const Data &, const Data &positions,
            std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &cfg,
            const LastTokensManager &history, std::vector<float> *retLogits) override {
        Check(history.units.size() == 1, "legacy history row mismatch");
        return Step(ids.Count(0), positions, kv[0].first, kv[0].second, cfg, history.units[0], retLogits);
    }
    // ForwardGPU's default delegation exercises the GPU scheduler's host
    // dispatch contract too; this fixture does not launch CUDA kernels.
    std::vector<int> ForwardV2(int batch, const Data &, const std::vector<Data *> &,
            const std::vector<Data *> &positions, const std::vector<int> &seqLens,
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &configs,
            const LastTokensManager &history, std::vector<std::vector<float> *> *logits) override {
        Check(history.units.empty() || history.units.size() == (size_t)batch, "batch history row mismatch");
        std::vector<int> result;
        for (int row = 0; row < batch; ++row) {
            LastTokensUnit empty(configs[row].last_n);
            Check(!history.units.empty() || configs[row].repeat_penalty == 1.f, "required history omitted");
            result.push_back(Step(seqLens[row], *positions[row], *kv[row].first, *kv[row].second,
                configs[row], history.units.empty() ? empty : history.units[row], logits ? (*logits)[row] : nullptr));
        }
        return result;
    }
};
int main() {
    try {
        SetDeviceMap({{"cpu", 1}}); SetMaxTokens(0);
        for (int mode = 0; mode < 4; ++mode) {
            Fixture model(mode < 2 ? 0 : mode - 1, mode == 1);
            for (int repeat = 0; repeat < 4; ++repeat) {
                std::vector<int> handles, counts(4), done(4);
                for (int i = 0; i < 4; ++i) {
                    GenerationConfig cfg;
                    cfg.output_token_limit = 24 + i * 8; cfg.add_special_tokens = false;
                    cfg.repeat_penalty = i == 0 ? 1.f : i == 1 ? 1.2f : i == 2 ? .8f : 1.f;
                    if (repeat == 2 && (i == 1 || i == 2))
                        cfg.repeat_penalty = std::nextafter(1.f, i == 1 ? 2.f : 0.f);
                    if (repeat == 3) cfg.repeat_penalty = 1.f;
                    cfg.last_n = i == 1 ? (repeat == 1 ? -1 : 0) : 4;
                    cfg.output_logits = i == 3 && repeat != 3;
                    cfg.top_k = i == 0 || i == 3 ? 8 : 1;
                    cfg.top_p = .95f;
                    if (i == 3 && repeat != 3) cfg.tool_call_allowed_token_ids = {1, 3, 5};
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
                for (int i = 0; i < 4; ++i) Check(counts[i] == 24 + i * 8, "request ended early");
            }
        }
        std::cout << "ALL_PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
