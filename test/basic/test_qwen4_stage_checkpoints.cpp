#include "models/qwen4_exp.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Check(bool ok, const char *message) {
        if (!ok) throw std::runtime_error(message);
    }
    static void Stage(Qwen4ExpModel &model, int begin, int end) {
        model.threadTpRank = 0;
        model.stageBegin = begin;
        model.stageEnd = end;
        struct RestoreRank {
            Qwen4ExpModel &model;
            ~RestoreRank() { model.threadTpRank = -1; }
        } restore{model};
        model.linearLayers.assign(48, true);
        for (int i = 3; i < 48; i += 4) model.linearLayers[i] = false;
        std::vector<std::pair<Data, Data>> cache;
        cache.reserve(48);
        for (int layer = 0; layer < 48; ++layer) {
            const bool linear = model.IsLinearAttentionLayer(layer);
            const std::vector<int> shape = linear ? std::vector<int>{2,3}
                : std::vector<int>{1,8,2};
            cache.emplace_back(Data(FLOAT32, shape,
                std::vector<float>(linear ? 6 : 16, float(layer+1))),
                Data(FLOAT32, shape,
                std::vector<float>(linear ? 6 : 16, float(layer+101))));
        }
        Qwen4ExpModel::RequestState state;
        Qwen4ExpModel::TargetRuntimeCheckpoint checkpoint;
        model.CaptureTargetRuntimeCheckpoint(cache, state, checkpoint);
        for (int layer = 0; layer < 48; ++layer) {
            const bool own = layer >= begin && layer < end;
            if (model.IsLinearAttentionLayer(layer)) {
                Check(!checkpoint.linearFirst[layer].dims.empty() == own,
                      "checkpoint captured a different stage's GDN state");
                Check(!checkpoint.linearSecond[layer].dims.empty() == own,
                      "checkpoint lost stage-local recurrent state");
                cache[layer].second.Allocate(float(layer + 501));
            } else {
                Check(checkpoint.keyLengths[layer] == (own ? 8 : 0),
                      "checkpoint crossed the stage's QSA range");
            }
        }
        model.CommitTargetRecurrentState(cache, checkpoint);
        for (int layer = 0; layer < 48; ++layer) {
            if (!model.IsLinearAttentionLayer(layer)) continue;
            const float actual = reinterpret_cast<float *>(cache[layer].second.cpuData)[0];
            const float expected = float(layer + (layer >= begin && layer < end ? 101 : 501));
            Check(actual == expected, "full acceptance modified another stage's cache");
        }
    }
};
}
int main() {
    try {
        unsetenv("FASTLLM_TP");
        fastllm::SetDeviceMap({{"cpu",1}});
        fastllm::SetThreads(2);
        fastllm::Qwen4ExpModel first, second;
        fastllm::Qwen4PrefixCacheTestAccess::Stage(first,0,24);
        fastllm::Qwen4PrefixCacheTestAccess::Stage(second,24,48);
        std::cout << "PASS stage-local checkpoints and recurrent commits\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
