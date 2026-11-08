#include "models/qwen4_exp.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Run() {
        SetDeviceMap({{"cpu", 1}}); SetThreads(1);
        Qwen4ExpModel model;
        model.threadTpRank = 0;
        struct Restore { Qwen4ExpModel &model; ~Restore(){model.threadTpRank=-1;} } restore{model};
        model.stageId = 1;
        model.mtpWeightsStatus.store(1); // Fixture isolates generation admission.
        model.deviceMap = {{"cuda:5",1}};
        model.moeDeviceMap = {{"cuda:5",1}};
        model.layeredMoeDeviceMap.clear();
        for (int drafts : {1,2,4,8}) {
            setenv("FASTLLM_QWEN4_ENABLE_MTP", std::to_string(drafts).c_str(), 1);
            for (int length : {1,16,128}) {
                for (bool sampling : {false,true}) {
                    auto config = Qwen4ExpModel::BuildWarmupGenerationConfig(
                        length,2*(drafts+1),sampling,true);
                    if (!model.MtpSupportsGenerationConfig(config) || config.do_sample ||
                        config.input_token_length != length ||
                        config.output_token_limit != 2*(drafts+1)+1) {
                        throw std::runtime_error("PP2 warmup configuration rejected required TP MTP");
                    }
                }
            }
        }
        auto ordinary = Qwen4ExpModel::BuildWarmupGenerationConfig(16,4,true,false);
        if (!ordinary.do_sample || ordinary.top_k != 10 || ordinary.top_p != 0.8f ||
            model.MtpSupportsGenerationConfig(ordinary)) {
            throw std::runtime_error("ordinary sampling semantics or TP MTP rejection changed");
        }
        model.mtpWeightsStatus.store(0);
        auto greedy = Qwen4ExpModel::BuildWarmupGenerationConfig(16,4,false,true);
        if (model.MtpSupportsGenerationConfig(greedy))
            throw std::runtime_error("warmup accepted a model without MTP weights");
    }
};
}
int main() {
    unsetenv("FASTLLM_TP"); unsetenv("FASTLLM_QWEN4_DUMP_DIR");
    try {
        fastllm::Qwen4PrefixCacheTestAccess::Run();
        std::cout << "PASS PP2 warmup requires executable greedy MTP for k=1/2/4/8\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
