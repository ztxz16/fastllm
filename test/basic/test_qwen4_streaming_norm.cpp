#include "models/qwen4_exp.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Run() {
        SetDeviceMap({{"cpu",1}}); SetThreads(1);
        const std::vector<std::string> names = {
            "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
            "model.language_model.layers.1.ple.norm_key.weight",
            "model.language_model.layers.1.ple.norm_query.weight",
            "model.language_model.layers.1.ple.norm_conv.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight"};
        for (auto type : {FLOAT32,FLOAT16,BFLOAT16}) {
            for (const auto &name : names) {
                Data norm(type,{3},std::vector<float>{0.0f,-0.5f,1.0f});
                Qwen4ExpModel::ApplyStreamingNormOffset(name,norm);
                if (norm.dataType != FLOAT32 || norm.cpuData == nullptr)
                    throw std::runtime_error("streaming norm was not expanded on CPU: " + name);
                const float *value = reinterpret_cast<const float *>(norm.cpuData);
                if (value[0]!=1.0f || value[1]!=0.5f || value[2]!=2.0f)
                    throw std::runtime_error("streaming norm differs from offset checkpoint convention: " + name);
                Data ggufNorm(type,{3},std::vector<float>{1.0f,0.5f,2.0f});
                ggufNorm.isGGUFData = true;
                Qwen4ExpModel::ApplyStreamingNormOffset(name,ggufNorm);
                Data ggufConverted; ToDataType(ggufNorm,ggufConverted,FLOAT32);
                const float *ggufValues = reinterpret_cast<const float *>(ggufConverted.cpuData);
                if (ggufNorm.dataType != type || ggufValues[0]!=1.0f ||
                    ggufValues[1]!=0.5f || ggufValues[2]!=2.0f)
                    throw std::runtime_error("GGUF effective norm was incorrectly offset: " + name);
            }
            Data linearNorm(type,{3},std::vector<float>{0.0f,-0.5f,1.0f});
            Qwen4ExpModel::ApplyStreamingNormOffset(
                "model.language_model.layers.0.linear_attn.norm.weight",linearNorm);
            Data converted; ToDataType(linearNorm,converted,FLOAT32);
            const float *value = reinterpret_cast<const float *>(converted.cpuData);
            if (linearNorm.dataType != type || value[0]!=0 || value[1]!=-0.5f || value[2]!=1)
                throw std::runtime_error("GDN absolute norm was incorrectly offset");
        }
        Qwen4ExpModel model;
        model.weight.dicts["gguf_architecture"] = "qwen4exp";
        const std::string normName = "mtp.pre_fc_norm_hidden.weight";
        Data &source = model.weight.weight[normName];
        source.CopyFrom(Data(FLOAT32,{3},std::vector<float>{1.0f,0.5f,2.0f}));
        if (model.HasGgufWeightStorage())
            throw std::runtime_error("safetensors architecture name misclassified as GGUF");
        source.isGGUFData = true;
        if (!model.HasGgufWeightStorage())
            throw std::runtime_error("GGUF F32 parameter marker was not recognized");
        // Exercise deferred callbacks without constructing CUDA TP state.
        // A wrong callback would consume/release this CPU load frontier.
        model.deferredGgufPipelineLoad = true;
        model.streamingTpLoad = true;
        model.streamingTpGroup = 1;
        model.streamingTpNames = {normName};
        uint8_t *original = source.cpuData;
        if (model.GetWeightLoadPriority(normName,{}) != 0 ||
            model.ShouldLoadWeightSeriallyBeforeOthers(normName,{}))
            throw std::runtime_error("GGUF PP load was scheduled for eager stage upload");
        model.OnWeightLoadGroupStarted({normName});
        model.OnWeightLoadGroupFinished();
        const float *preserved = reinterpret_cast<const float *>(source.cpuData);
        if (source.cpuData == nullptr || source.cpuData != original || source.dataDevice != DataDevice::CPU ||
            source.multiDeviceData || preserved[0]!=1.0f || preserved[1]!=0.5f ||
            preserved[2]!=2.0f || model.streamingTpGroup != 1 ||
            model.streamingTpNames != std::set<std::string>{normName})
            throw std::runtime_error("GGUF PP callbacks consumed sources before layout restoration");
    }
};
}
int main() {
    unsetenv("FASTLLM_TP");
    try {
        fastllm::Qwen4PrefixCacheTestAccess::Run();
        std::cout << "PASS streamed MTP/PLE norm values match checkpoint offsets without changing GDN norms\n";
        return 0;
    } catch (const std::exception &e) {std::cerr << e.what() << '\n';return 1;}
}
