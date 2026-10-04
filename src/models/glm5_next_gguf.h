#ifndef FASTLLM_GLM5_NEXT_GGUF_H
#define FASTLLM_GLM5_NEXT_GGUF_H

#include "fastllm.h"
#include "utils.h"
#include <cmath>

namespace fastllm {
namespace glm5_next_detail {

// The glm5next GGUF stores -exp(A_log), and K/V projections in separate
// tensors with K transposed per head. Consume the source names so restoration
// is idempotent and ordinary HF checkpoints never enter these conversions.
inline void RestoreGgufWeights(WeightMap &weights, int layers, int heads,
                              int keyDim, int valueDim, int latentDim) {
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        const std::string attn = prefix + "self_attn.";
        auto decay = weights.weight.find(attn + "gguf_decay");
        if (decay != weights.weight.end()) {
            const Data &source = decay->second;
            AssertInFastLLM(source.dataType == FLOAT32 && source.cpuData &&
                source.dataDevice == DataDevice::CPU && source.Count(0) == (uint64_t)heads,
                "GLM-5.3 GGUF KDA decay has an invalid layout.");
            Data &target = weights[attn + "A_log"];
            target.CopyFrom(source);
            float *values = reinterpret_cast<float *>(target.cpuData);
            for (uint64_t i = 0; i < target.Count(0); ++i) {
                AssertInFastLLM(std::isfinite(values[i]) && values[i] < 0,
                    "GLM-5.3 GGUF KDA decay must be finite and negative.");
                values[i] = std::log(-values[i]);
            }
            target.name = attn + "A_log";
            target.isModelWeight = true;
            weights.weight.erase(attn + "gguf_decay");
        }
        auto key = weights.weight.find(attn + "gguf_k_b.weight");
        auto value = weights.weight.find(attn + "gguf_v_b.weight");
        if (key != weights.weight.end() || value != weights.weight.end()) {
            AssertInFastLLM(key != weights.weight.end() && value != weights.weight.end(),
                "GLM-5.3 GGUF requires both K-B and V-B projections.");
            const Data &k = key->second, &v = value->second;
            AssertInFastLLM(k.dataType == FLOAT32 && v.dataType == FLOAT32 &&
                k.dataDevice == DataDevice::CPU && v.dataDevice == DataDevice::CPU &&
                k.cpuData && v.cpuData &&
                k.dims == std::vector<int>({heads, latentDim, keyDim}) &&
                v.dims == std::vector<int>({heads, valueDim, latentDim}),
                "GLM-5.3 GGUF split KV-B projections have invalid layouts.");
            const std::string name = attn + "kv_b_proj.weight";
            AssertInFastLLM(weights.weight.count(name) == 0,
                "GLM-5.3 GGUF mixes combined and split KV-B projections.");
            Data &combined = weights[name];
            combined = Data(BFLOAT16, {heads * (keyDim + valueDim), latentDim});
            combined.Allocate();
            const float *ks = reinterpret_cast<const float *>(k.cpuData);
            const float *vs = reinterpret_cast<const float *>(v.cpuData);
            auto *out = reinterpret_cast<uint16_t *>(combined.cpuData);
            for (int h = 0; h < heads; ++h) {
                for (int d = 0; d < keyDim + valueDim; ++d) {
                    for (int r = 0; r < latentDim; ++r) {
                        const float x = d < keyDim ?
                            ks[((size_t)h * latentDim + r) * keyDim + d] :
                            vs[((size_t)h * valueDim + d - keyDim) * latentDim + r];
                        out[((size_t)h * (keyDim + valueDim) + d) * latentDim + r] =
                            Float32ToBFloat16RNEBits(x);
                    }
                }
            }
            combined.name = name;
            combined.isGGUFData = combined.isModelWeight = true;
            combined.weightType = WeightType::LINEAR;
            weights.weight.erase(attn + "gguf_k_b.weight");
            weights.weight.erase(attn + "gguf_v_b.weight");
        }
    }
}

} // namespace glm5_next_detail
} // namespace fastllm
#endif
