#include "models/basellm.h"
#include <atomic>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

// A deterministic speculative model exercises the real scheduler independently
// of floating point rounding and GPU sampling. Its cache includes the whole
// verified block, just as a target verification forward does.
class Fixture : public basellm {
    struct State { int generated = 0; std::deque<int> pending; };
    std::mutex stateMutex;
    std::map<const Data *, State> states;
    int width;
    int Step(int rows, const Data &position, Data &key, Data &value,
             const GenerationConfig &cfg, const LastTokensUnit &history) {
        std::lock_guard<std::mutex> lock(stateMutex);
        auto &s = states[&key];
        Check(s.pending.empty(), "scheduler forwarded a request with an unconsumed verified block");
        const int past = key.dims.empty() ? 0 : key.dims[1];
        Check(((float *)position.cpuData)[0] == past, "position did not advance by published tokens");
        std::multiset<int> expected;
        for (int i = std::max(0, s.generated - cfg.last_n); i < s.generated; ++i) expected.insert(10 + i);
        Check(history.tokenSet == expected, "repetition history missed a published token");
        const int count = std::min(width, cfg.output_token_limit - s.generated);
        Check(count > 0, "forward ran beyond output limit");
        const int first = 10 + s.generated;
        for (int i = 1; i < count; ++i) s.pending.push_back(first + i);
        s.generated += count;
        for (Data *cache : {&key, &value}) {
            cache->Resize({1, past + rows + count - 1, 1});
            cache->expansionDims = {1, 512, 1};
        }
        ++forwards;
        return first;
    }
public:
    std::atomic<int> forwards{0}, popped{0};
    Fixture(int width, bool fastInput, int concurrency) : width(width) {
        if (fastInput) model_struct = "naive_n05_flash";
        block_cnt = 1; kvCacheId = 0; maxBatch = concurrency;
        tokensLimit = concurrency * 1024; max_positions = 512;
        dataType = kvCacheDataType = FLOAT32; eos_token_id = -100;
        use_new_engine = false; canDoBatchForward = false; canDoConcurrentForward = true;
        SetSaveHistoryChat(false);
    }
    ~Fixture() override { ShutdownRuntime(); }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override { return s; }
    bool NeedAttentionMask(int, int) override { return false; }
    bool UseGenericHistoryCache() const override { return false; }
    bool RetainCudaWorkspace() const override { return true; }
    int Forward(const Data &ids, const Data &, const Data &positions,
                std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &cfg,
                const LastTokensManager &history, std::vector<float> *) override {
        return Step(ids.Count(0), positions, kv[0].first, kv[0].second, cfg, history.units.at(0));
    }
    std::vector<int> ForwardBatch(int batch, const Data &, const std::vector<Data *> &,
                const std::vector<Data *> &positions, const std::vector<int> &lengths,
                std::vector<std::pair<Data *, Data *>> &kv,
                const std::vector<GenerationConfig> &configs,
                const LastTokensManager &history, std::vector<std::vector<float> *> *) override {
        std::vector<int> out(batch);
        for (int i = 0; i < batch; ++i)
            out[i] = Step(lengths[i], *positions[i], *kv[i].first, *kv[i].second,
                          configs[i], history.units.at(i));
        return out;
    }
    bool PopPendingResponseToken(ResponseContext *ctx, int previous, int &token) override {
        std::lock_guard<std::mutex> lock(stateMutex);
        auto &s = states.at(&ctx->pastKeyValues[0].first);
        if (s.pending.empty()) return false;
        token = s.pending.front(); s.pending.pop_front();
        Check(token == previous + 1, "verified block was reordered");
        ++popped;
        return true;
    }
    void OnResponseContextRemoved(ResponseContext *ctx) override {
        std::lock_guard<std::mutex> lock(stateMutex);
        states.erase(&ctx->pastKeyValues[0].first);
    }
};

int main() {
    try {
        SetDeviceMap({{"cpu", 1}}); SetMaxTokens(0); SetThreads(2);
        for (bool fastInput : {false, true}) for (int width : {1, 3, 5, 7})
        for (int concurrency : {1, 4, 16, 128}) {
            Fixture model(width, fastInput, concurrency);
            for (int mode : {0, 1, 2}) {
                model.eos_token_id = mode == 2 ? 14 : -100;
                std::vector<int> handles, limits, counts(concurrency), done(concurrency);
                const int before = model.forwards;
                int expectedForwards = 0;
                for (int i = 0; i < concurrency; ++i) {
                    GenerationConfig cfg; cfg.output_token_limit = mode ? 23 : 1 + i % 23;
                    cfg.add_special_tokens = false; cfg.last_n = 8; cfg.repeat_penalty = 1.1f;
                    if (mode == 1) cfg.stop_token_ids = {12};
                    limits.push_back(mode == 1 ? 2 : mode == 2 ? 4 : cfg.output_token_limit);
                    expectedForwards += (limits.back() + (mode ? 1 : 0) + width - 1) / width;
                    handles.push_back(model.LaunchResponseTokens(std::vector<int>(16+i%3,1),cfg));
                }
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
                int remaining = concurrency;
                while (remaining) {
                    Check(std::chrono::steady_clock::now() < deadline, "scheduler stalled");
                    for (int i = 0; i < concurrency; ++i) if (!done[i] && model.CanFetchResponse(handles[i])) {
                        int tokens[64], statistics[3];
                        int count = model.FetchResponseTokensBatch(handles[i], tokens, 64, statistics);
                        for (int j = 0; j < count; ++j) Check(tokens[j] == 10 + counts[i]++, "emitted token sequence differs");
                        if (count < 0) {
                            Check(count == -1 && counts[i] == limits[i], "stop/EOS/limit did not stop inside verified block");
                            Check(statistics[2] == limits[i], "terminal output statistics differ");
                            done[i] = 1; --remaining;
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                Check(model.forwards - before == expectedForwards, "verified tokens required extra forward passes");
            }
        }
        std::cout << "ALL_PASS verified blocks: order, history, positions, EOS, stop, limits, handle reuse, C1..128\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
