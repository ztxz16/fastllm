#include "models/naive_n05_flash.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace fastllm;

// Non-paged execution must safely recompute instead of archiving CPU KV.
class HistoryFixture : public NaiveN05FlashModel {
public:
    HistoryFixture() {
        weight.dicts = {{"num_hidden_layers", "2"}, {"hidden_size", "16"},
            {"num_attention_heads", "1"}, {"num_key_value_heads", "1"},
            {"head_dim", "128"}, {"v_head_dim", "8"},
            {"swa_num_attention_heads", "1"}, {"swa_num_key_value_heads", "1"},
            {"swa_head_dim", "128"}, {"swa_v_head_dim", "8"},
            {"hybrid_layer_pattern", "[0,1]"}, {"moe_layer_freq", "[0,0]"},
            {"scoring_func", "sigmoid"}, {"sliding_window", "128"}};
        InitParams();
        SetSaveHistoryChat(true);
    }
    bool UseModelSpecificScheduler() const override { return true; }
    void RunModelSpecificScheduler() override {}
    void CheckFallback(const std::vector<int> &tokens) {
        int handle = LaunchResponseTokens(tokens, GenerationConfig());
        auto *context = responseContextDict.GetHandle(handle);
        if (context->cacheLen || context->currentTokens != tokens)
            throw std::runtime_error("unsupported prefix layout was reused");
        for (auto &kv : context->pastKeyValues) {
            kv.first.Resize({1, 128, 128}); kv.first.Allocate();
            kv.second.Resize({1, 128, 8}); kv.second.Allocate();
        }
        TryRecordResponseContext(context);
        auto input = tokens;
        int cached = -1;
        if (TryRestoreHistoryCache(input, cached) || cached || input != tokens ||
            TryRecordPagedPrefixCacheExtra(context) || QueryPagedPrefixCacheExtra(context, 128))
            throw std::runtime_error("nonpaged execution created a CPU KV archive");
        RemoveResponseContext(handle);
    }
};

int main() {
    try {
        SetDeviceMap({{"cpu", 1}});
        HistoryFixture model;
        std::vector<int> tokens(300);
        std::iota(tokens.begin(), tokens.end(), 10);
        for (int i = 0; i < 50; ++i) model.CheckFallback(tokens);
        model.SetSaveHistoryChat(false);
        model.CheckFallback(tokens);
        model.SetSaveHistoryChat(true);
        model.CheckFallback(tokens);
        std::cout << "CPU HISTORY DISABLED PASS" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
