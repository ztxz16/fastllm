#include "model.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace fastllm;

int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "valid";
    const char *ep = mode == "invalid_ep" ? "4" : mode == "ep_without_tp" ? "8" : "1";
#ifdef _WIN32
    _putenv_s("FASTLLM_TP", "false");
    _putenv_s("FASTLLM_EP", ep);
#else
    setenv("FASTLLM_TP", "false", 1);
    setenv("FASTLLM_EP", ep, 1);
#endif
    for (const char *alias : {"mimo_v2", "MiMoV2ForCausalLM", "MiMoV2DSAForCausalLM"}) {
        auto model = CreateEmptyLLMModel(alias);
        model->weight.dicts = {
            {"model_type", "mimo_v2"}, {"enable_dsa", "true"},
            {"attention_projection_layout", "split"}, {"n_shared_experts", "null"},
            {"hidden_size", "4096"}, {"num_hidden_layers", "2"},
            {"hybrid_layer_pattern", "[0,1]"}, {"moe_layer_freq", "[0,1]"},
            {"scoring_func", "sigmoid"}, {"routed_scaling_factor", "null"},
            {"attention_value_scale", "0.707"}
        };
        if (mode == "no_dsa") model->weight.dicts["enable_dsa"] = "false";
        if (mode == "fused_qkv") model->weight.dicts["attention_projection_layout"] = "fused";
        if (mode == "shared_experts") model->weight.dicts["n_shared_experts"] = "1";
        model->InitParams();
        if (mode != "valid" || model->model_type != "naive_n05_flash" ||
            model->model_struct != "naive_n05_flash" || model->dataType != BFLOAT16)
            throw std::runtime_error("MiMo-V2 DSA factory/configuration regression");
    }
    std::cout << "PASS MiMo-V2 DSA configuration" << std::endl;
}
