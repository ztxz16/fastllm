#include "models/naive_n05_flash.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

// Feed identifiable KV rows through the production archive and request hooks.
// The scheduler is idle: tests cover cache ownership independently of weights.
class HistoryFixture : public NaiveN05FlashModel {
public:
    HistoryFixture() {
        weight.dicts = {{"num_hidden_layers", "3"}, {"hidden_size", "16"},
            {"num_attention_heads", "1"}, {"num_key_value_heads", "1"},
            {"head_dim", "128"}, {"v_head_dim", "8"},
            {"swa_num_attention_heads", "1"}, {"swa_num_key_value_heads", "1"},
            {"swa_head_dim", "128"}, {"swa_v_head_dim", "8"},
            {"hybrid_layer_pattern", "[0,1,0]"}, {"moe_layer_freq", "[0,0,0]"},
            {"scoring_func", "sigmoid"}, {"sliding_window", "128"}};
        InitParams();
        SetSaveHistoryChat(true);
    }
    bool UseModelSpecificScheduler() const override { return true; }
    void RunModelSpecificScheduler() override {}
    static uint16_t Bits(int row, int layer, int part, int col) {
        return (row * 17 + layer * 31 + part * 13 + col) & 0xffff;
    }
    ResponseContext *Create(const std::vector<int> &tokens, bool media = false) {
        std::map<std::string, std::vector<Data *>> multimodal;
        if (media) multimodal["image"] = {new Data(FLOAT32)};
        int handle = LaunchResponseTokens(tokens, GenerationConfig(), multimodal);
        return responseContextDict.GetHandle(handle);
    }
    ResponseContext *CreateDuringForward(const std::vector<int> &tokens) {
        // Host-only restore must not wait for another request's GPU forward.
        std::lock_guard<std::mutex> guard(forwardLocker);
        return Create(tokens);
    }
    void Feed(ResponseContext *context, int past, int count) {
        auto chunk = BeginHistoryChunk(context->pastKeyValues, past, count);
        Require(chunk && chunk->length == count, "History recording was unavailable");
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int part = 0; part < 2; ++part) {
                int width = part ? 8 : layer == 1 ? 128 : 256;
                Data input(BFLOAT16, {1, count, width});
                input.Allocate();
                for (int row = 0; row < count; ++row) for (int col = 0; col < width; ++col)
                    ((uint16_t *)input.cpuData)[row * width + col] = Bits(past + row, layer, part, col);
                CopyHistoryTensor(input, part ? chunk->layers[layer].second : chunk->layers[layer].first, count);
            }
        }
        FinishHistoryChunk(context->pastKeyValues, chunk);
    }
    void CheckBound(ResponseContext *context) {
        auto chunk = BeginHistoryChunk(context->pastKeyValues, 0, 100000000);
        Require(chunk && chunk->length < 100000000 && chunk->bytes <= (1ULL << 30),
                "Archive capacity did not limit an oversized request");
    }
    void CheckRestored(ResponseContext *context, int expected) {
        Require(context->cacheLen == expected, "Incorrect cached prefix length");
        Require(context->preTokens == expected && context->intParams["index"] == -1 &&
                context->intParams["promptLen"] == context->inputTokens,
                "Restored request has incorrect scheduler positions");
        for (int layer = 0; layer < block_cnt; ++layer) {
            int start = layer == 1 ? std::max(0, expected - 127) : 0;
            for (int part = 0; part < 2; ++part) {
                const Data &cache = part ? context->pastKeyValues[layer].second : context->pastKeyValues[layer].first;
                Require(cache.dims[1] == expected - start && cache.isKVCache,
                        "Incorrect restored full/sliding cache shape");
                for (int row = 0; row < cache.dims[1]; ++row) for (int col = 0; col < cache.dims[2]; ++col)
                    Require(((uint16_t *)cache.cpuData)[row * cache.dims[2] + col] == Bits(start + row, layer, part, col),
                            "KV/index key or sliding-window boundary changed during restore");
            }
        }
    }
};

int main() {
    try {
        SetDeviceMap({{"cpu", 1}});
        HistoryFixture model;
        std::vector<int> tokens(300);
        std::iota(tokens.begin(), tokens.end(), 10);
        auto *seed = model.Create(tokens);
        model.CheckBound(seed);
        model.Feed(seed, 0, 150);
        model.Feed(seed, 150, 150);
        seed->allTokens.push_back(900); // sampled but not yet forwarded
        model.TryRecordResponseContext(seed);

        auto *repeat = model.CreateDuringForward(tokens);
        model.CheckRestored(repeat, 299);
        Require(repeat->currentTokens == std::vector<int>{tokens.back()}, "Exact repeat lost final input token");
        ((uint16_t *)repeat->pastKeyValues[0].first.cpuData)[0] ^= 1;
        for (int boundary : {1, 126, 127, 128, 149, 150, 151, 299}) {
            auto branch = tokens;
            branch[boundary] = 999;
            model.CheckRestored(model.Create(branch), boundary);
        }
        model.CheckRestored(model.Create(std::vector<int>(tokens.begin(), tokens.begin() + 80)), 79);
        auto extended = tokens;
        extended.insert(extended.end(), {900, 901, 902});
        auto *child = model.Create(extended);
        model.CheckRestored(child, 300);
        model.Feed(child, 300, 3);
        model.TryRecordResponseContext(child);
        model.CheckRestored(model.Create(extended), 302);
        // Forking and extending a shared chunk must not mutate its parent.
        auto fork = std::vector<int>(tokens.begin(), tokens.begin() + 170);
        fork[149] = 999;
        auto *forked = model.Create(fork);
        model.CheckRestored(forked, 149);
        model.Feed(forked, 149, 21);
        model.TryRecordResponseContext(forked);
        model.CheckRestored(model.Create(fork), 169);
        model.CheckRestored(model.Create(tokens), 299);
        auto *media = model.Create(tokens, true);
        Require(media->cacheLen == 0 && media->currentTokens == tokens, "Multimodal request reused text KV");
        model.TryRecordResponseContext(media);

        // A completed speculative verify can be ahead of the last token
        // emitted by the scheduler. Never publish its unconsumed suffix.
        auto shortRequest = model.Create({101, 102, 103, 104, 105});
        model.Feed(shortRequest, 0, 8);
        model.TryRecordResponseContext(shortRequest);
        auto shortHit = model.Create({101, 102, 103, 104, 105, 900});
        model.CheckRestored(shortHit, 5);
        model.SetSaveHistoryChat(false);
        Require(model.Create(tokens)->cacheLen == 0, "Disabled history cache still hit");
        model.SetSaveHistoryChat(true);
        Require(model.Create(tokens)->cacheLen == 0, "Disabling history did not clear old entries");

        // Hits and recording a covered prefix both refresh the LRU order.
        std::vector<ResponseContext *> recorded;
        for (int i = 0; i < 5; ++i) {
            auto *context = model.Create({1000 + i, 20});
            model.Feed(context, 0, 2);
            model.TryRecordResponseContext(context);
            recorded.push_back(context);
        }
        model.CheckRestored(model.Create({1000, 20, 30}), 2);
        auto *last = model.Create({1005, 20});
        model.Feed(last, 0, 2);
        model.TryRecordResponseContext(last);
        Require(model.Create({1001, 20, 30})->cacheLen == 0, "Least recently used history entry was not evicted");
        model.CheckRestored(model.Create({1000, 20, 30}), 2);
        model.CheckRestored(model.Create({1005, 20, 30}), 2);
        model.TryRecordResponseContext(recorded[2]);
        last = model.Create({1006, 20});
        model.Feed(last, 0, 2);
        model.TryRecordResponseContext(last);
        Require(model.Create({1003, 20, 30})->cacheLen == 0, "Recording a covered prefix did not refresh LRU");
        model.CheckRestored(model.Create({1002, 20, 30}), 2);
        std::cout << "Naive history prefix, fork, sliding, index, isolation and LRU: PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
