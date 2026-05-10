//
// Step-3.5 text model support.
//

#include "step3p5.h"
#include "utils.h"
#include "json11.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <sstream>

namespace fastllm {
    static const std::string STEP3P5_BOS = "<｜begin▁of▁sentence｜>";
    static const std::string STEP3P5_IM_START = "<|im_start|>";
    static const std::string STEP3P5_IM_END = "<|im_end|>";

    static bool Step3p5IsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return lowered == "1" || lowered == "true" || lowered == "on";
    }

    static bool Step3p5DeviceMapUsesCuda(const std::map<std::string, int> &deviceMap) {
        for (auto &it : deviceMap) {
            if (it.first.rfind("cuda", 0) == 0 || it.first.rfind("multicuda", 0) == 0) {
                return true;
            }
        }
        return false;
    }

    static bool Step3p5DeviceMapUsesDisk(const std::map<std::string, int> &deviceMap) {
        for (auto &it : deviceMap) {
            if (it.first == "disk") {
                return true;
            }
        }
        return false;
    }

    static std::string Step3p5GetDict(const std::map<std::string, std::string> &dict,
                                      const std::string &key,
                                      const std::string &defaultValue = "") {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : it->second;
    }

    static int Step3p5GetInt(const std::map<std::string, std::string> &dict,
                             const std::string &key, int defaultValue) {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : atoi(it->second.c_str());
    }

    static float Step3p5GetFloat(const std::map<std::string, std::string> &dict,
                                 const std::string &key, float defaultValue) {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : atof(it->second.c_str());
    }

    static std::vector<float> Step3p5ParseFloatList(const std::string &value) {
        std::vector<float> ret;
        if (value.empty()) {
            return ret;
        }
        if (value[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(value, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    ret.push_back((float)item.number_value());
                }
            }
            return ret;
        }
        std::stringstream ss(value);
        std::string part;
        while (std::getline(ss, part, ',')) {
            if (!part.empty()) {
                ret.push_back((float)atof(part.c_str()));
            }
        }
        return ret;
    }

    static std::vector<std::string> Step3p5ParseStringList(const std::string &value) {
        std::vector<std::string> ret;
        if (value.empty() || value[0] != '[') {
            return ret;
        }
        std::string error;
        auto arr = json11::Json::parse(value, error);
        if (error.empty() && arr.is_array()) {
            for (auto &item : arr.array_items()) {
                ret.push_back(item.string_value());
            }
        }
        return ret;
    }

    static std::set<int> Step3p5ParseIntSet(const std::string &value) {
        std::set<int> ret;
        if (value.empty()) {
            return ret;
        }
        if (value[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(value, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    ret.insert(item.int_value());
                }
            }
            return ret;
        }
        std::stringstream ss(value);
        std::string part;
        while (std::getline(ss, part, ',')) {
            if (!part.empty()) {
                ret.insert(atoi(part.c_str()));
            }
        }
        return ret;
    }

    static void Step3p5Add1(Data &input) {
        if (input.dims.empty()) {
            return;
        }
        input.ToDevice(DataDevice::CPU);
        if (input.dataType != DataType::FLOAT32) {
            ToDataType(input, DataType::FLOAT32);
            input.ToDevice(DataDevice::CPU);
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    static void Step3p5MakeExpertView(Data &dst, const Data &src, const std::string &name, int expert) {
        AssertInFastLLM(src.dims.size() == 3, "Step3p5 MoE expert source weight should be 3D.");
        AssertInFastLLM(expert >= 0 && expert < src.dims[0], "Step3p5 MoE expert index out of range.");
        int rows = src.dims[1], cols = src.dims[2];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        if (src.isDiskWeight) {
            dst.isDiskWeight = true;
            dst.dataDevice = DataDevice::CPU;
            dst.cpuData = nullptr;
            dst.diskWeightParts.clear();
            for (auto part : src.diskWeightParts) {
                AssertInFastLLM(part.dims.size() == 3 && part.dims[0] == src.dims[0] &&
                                part.dims[1] == rows && part.dims[2] == cols,
                                "Step3p5 disk MoE expert source part should match the fused 3D tensor.");
                uint64_t bytesPerExpert = part.bytes / part.dims[0];
                part.fileOffset += (long long)bytesPerExpert * expert;
                part.bytes = bytesPerExpert;
                part.dims = {rows, cols};
                dst.diskWeightParts.push_back(part);
            }
        } else {
            dst.FakeFrom(src, (size_t)expert * rows * cols * src.unitSize / src.unitSizeDiv);
        }
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int perExpert = ks * ms;
            AssertInFastLLM((expert + 1) * perExpert <= (int)src.scales.size(),
                            "Step3p5 MoE expert scale range is out of bounds.");
            dst.scales.assign(src.scales.begin() + expert * perExpert,
                              src.scales.begin() + (expert + 1) * perExpert);
        }
    }

    static void Step3p5MakeExpertCopy(Data &dst, Data &src, const std::string &name, int expert) {
        AssertInFastLLM(src.dims.size() == 3, "Step3p5 MoE expert source weight should be 3D.");
        AssertInFastLLM(expert >= 0 && expert < src.dims[0], "Step3p5 MoE expert index out of range.");
        src.ToDevice(DataDevice::CPU);
        AssertInFastLLM(src.cpuData != nullptr, "Step3p5 MoE expert source should be in CPU memory.");

        int rows = src.dims[1], cols = src.dims[2];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.Allocate();

        uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expert, bytesPerExpert);
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int perExpert = ks * ms;
            AssertInFastLLM((expert + 1) * perExpert <= (int)src.scales.size(),
                            "Step3p5 MoE expert scale range is out of bounds.");
            dst.scales.assign(src.scales.begin() + expert * perExpert,
                              src.scales.begin() + (expert + 1) * perExpert);
        }
    }

    static void Step3p5MakeGateUpWeight(Data &dst, const Data &gate, const Data &up, const std::string &name) {
        AssertInFastLLM(gate.dims.size() == 2 && up.dims.size() == 2 &&
                        gate.dims[0] == up.dims[0] && gate.dims[1] == up.dims[1],
                        "Step3p5 MoE gate/up expert weights should have the same 2D shape.");
        AssertInFastLLM(gate.dataType == up.dataType,
                        "Step3p5 MoE gate/up expert weights should have the same dtype.");
        dst = Data(gate.dataType, {gate.dims[0] + up.dims[0], gate.dims[1]});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = gate.blockK;
        dst.blockM = gate.blockM;
        dst.group = gate.group;
        dst.groupCnt = gate.groupCnt;
        dst.perChannelAxis = gate.perChannelAxis;
        if (gate.isDiskWeight || up.isDiskWeight) {
            AssertInFastLLM(gate.isDiskWeight && up.isDiskWeight,
                            "Step3p5 disk MoE gate/up weights should both be disk weights.");
            dst.isDiskWeight = true;
            dst.dataDevice = DataDevice::CPU;
            dst.cpuData = nullptr;
            dst.diskWeightParts = gate.diskWeightParts;
            dst.diskWeightParts.insert(dst.diskWeightParts.end(),
                                       up.diskWeightParts.begin(), up.diskWeightParts.end());
        } else {
            dst.Allocate();
            memcpy(dst.cpuData, gate.cpuData, gate.GetBytes());
            memcpy(dst.cpuData + gate.GetBytes(), up.cpuData, up.GetBytes());
        }
        dst.perChannelsConfigs = AppendVector(dst.perChannelsConfigs, gate.perChannelsConfigs);
        dst.perChannelsConfigs = AppendVector(dst.perChannelsConfigs, up.perChannelsConfigs);
        dst.zeros = AppendVector(dst.zeros, gate.zeros);
        dst.zeros = AppendVector(dst.zeros, up.zeros);
        dst.scales = AppendVector(dst.scales, gate.scales);
        dst.scales = AppendVector(dst.scales, up.scales);
        dst.mins = AppendVector(dst.mins, gate.mins);
        dst.mins = AppendVector(dst.mins, up.mins);
        dst.halfScales = AppendVector(dst.halfScales, gate.halfScales);
        dst.halfScales = AppendVector(dst.halfScales, up.halfScales);
        if (!dst.isDiskWeight) {
            dst.CalcWeightSum();
        }
    }

    static void Step3p5MakeGateUpSliceView(Data &dst, const Data &src, const std::string &name,
                                           int rowStart, int rows) {
        AssertInFastLLM(src.dims.size() == 2, "Step3p5 MoE gateup weight should be 2D.");
        AssertInFastLLM(rowStart >= 0 && rows > 0 && rowStart + rows <= src.dims[0],
                        "Step3p5 MoE gateup slice is out of range.");
        int cols = src.dims[1];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.FakeFrom(src, (uint64_t)rowStart * cols * src.unitSize / src.unitSizeDiv);
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            AssertInFastLLM(rowStart % src.blockK == 0,
                            "Step3p5 MoE gateup scale slice should align with blockK.");
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int scaleOffset = (rowStart / src.blockK) * ms;
            int scaleCount = ks * ms;
            AssertInFastLLM(scaleOffset + scaleCount <= (int)src.scales.size(),
                            "Step3p5 MoE gateup scale slice is out of bounds.");
            dst.scales.assign(src.scales.begin() + scaleOffset,
                              src.scales.begin() + scaleOffset + scaleCount);
        }
    }

    static float Step3p5LayerLimit(const std::vector<float> &limits, int layer) {
        return layer >= 0 && layer < (int)limits.size() ? limits[layer] : 0.0f;
    }

    static float Step3p5ReadFloat(const Data &input, int index) {
        if (input.dataType == DataType::FLOAT32) {
            return ((float*)input.cpuData)[index];
        }
        uint16_t v = ((uint16_t*)input.cpuData)[index];
        if (input.dataType == DataType::FLOAT16) {
            return half_to_float(v);
        }
        if (input.dataType == DataType::BFLOAT16) {
            uint32_t raw = ((uint32_t)v) << 16;
            float ret;
            memcpy(&ret, &raw, sizeof(ret));
            return ret;
        }
        ErrorInFastLLM("Step3p5ReadFloat: unsupported data type.");
        return 0.0f;
    }

    static uint16_t Step3p5FloatToBFloat16(float value) {
        uint32_t raw;
        memcpy(&raw, &value, sizeof(raw));
        uint32_t rounding = ((raw >> 16) & 1) + 0x7FFF;
        return (uint16_t)((raw + rounding) >> 16);
    }

    static void Step3p5WriteFloat(Data &input, int index, float value) {
        if (input.dataType == DataType::FLOAT32) {
            ((float*)input.cpuData)[index] = value;
            return;
        }
        if (input.dataType == DataType::FLOAT16) {
            ((uint16_t*)input.cpuData)[index] = float_to_half(value);
            return;
        }
        if (input.dataType == DataType::BFLOAT16) {
            ((uint16_t*)input.cpuData)[index] = Step3p5FloatToBFloat16(value);
            return;
        }
        ErrorInFastLLM("Step3p5WriteFloat: unsupported data type.");
    }

    static void Step3p5Clamp(Data &input, bool hasMin, float minValue, bool hasMax, float maxValue) {
        DataDevice originalDevice = input.dataDevice;
        std::vector<int> originalDeviceIds = input.dataDeviceIds;
        input.ToDevice(DataDevice::CPU);
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            float value = Step3p5ReadFloat(input, i);
            if (hasMin && value < minValue) {
                value = minValue;
            }
            if (hasMax && value > maxValue) {
                value = maxValue;
            }
            Step3p5WriteFloat(input, i, value);
        }
        input.ToDevice(originalDevice, originalDeviceIds);
    }

    static DataType Step3p5ResolvePagedAttentionQType(DataType cacheType, DataType queryType, DataType modelType) {
        if (cacheType == DataType::FLOAT16 || cacheType == DataType::BFLOAT16) {
            return cacheType;
        }
        if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
            return queryType;
        }
        return modelType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
    }

    static Data &Step3p5PreparePagedAttentionQ(Data &src, DataType cacheType, DataType modelType, Data &casted) {
        DataType targetType = Step3p5ResolvePagedAttentionQType(cacheType, src.dataType, modelType);
        if (src.dataType == targetType) {
            return src;
        }
        ToDataType(src, casted, targetType);
        return casted;
    }

    Step3p5Model::Step3p5Model() {
        this->model_type = "step3p5";
        this->model_struct = "step3p5";
        this->use_new_engine = true;

        this->pre_prompt = STEP3P5_BOS;
        this->user_role = STEP3P5_IM_START + std::string("user\n");
        this->bot_role = STEP3P5_IM_END + std::string("\n") +
                         STEP3P5_IM_START + std::string("assistant\n<think>\n");
        this->history_sep = STEP3P5_IM_END + std::string("\n");

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            "model.layers.*.mlp.down_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.share_expert.down_proj.weight",
            "model.layers.*.share_expert.up_proj.weight",
            "model.layers.*.share_expert.gate_proj.weight",
            "model.layers.*.share_expert.gateup_proj.weight",
            "model.layers.*.moe.gate.weight",
            "model.layers.*.moe.gate_proj.weight",
            "model.layers.*.moe.up_proj.weight",
            "model.layers.*.moe.down_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.self_attn.g_proj.weight",
            "model.layers.*.self_attn.mergeqkv.weight"
        };
    }

    void Step3p5Model::InitParams() {
        basellm::InitParams();

        base_attention_heads = Step3p5GetInt(weight.dicts, "num_attention_heads", num_attention_heads);
        base_key_value_heads = Step3p5GetInt(weight.dicts, "num_attention_groups", base_key_value_heads);
        sliding_attention_heads = Step3p5GetInt(weight.dicts, "attention_other_setting.num_attention_heads", base_attention_heads);
        sliding_key_value_heads = Step3p5GetInt(weight.dicts, "attention_other_setting.num_attention_groups", base_key_value_heads);
        num_attention_heads = base_attention_heads;
        num_key_value_heads = base_key_value_heads;
        head_dim = Step3p5GetInt(weight.dicts, "head_dim", head_dim);
        embed_dim = Step3p5GetInt(weight.dicts, "hidden_size", embed_dim);
        dense_intermediate_size = Step3p5GetInt(weight.dicts, "intermediate_size", dense_intermediate_size);
        moe_intermediate_size = Step3p5GetInt(weight.dicts, "moe_intermediate_size", moe_intermediate_size);
        shared_expert_intermediate_size = Step3p5GetInt(weight.dicts, "share_expert_dim", shared_expert_intermediate_size);
        num_experts = Step3p5GetInt(weight.dicts, "moe_num_experts", num_experts);
        num_experts_per_tok = Step3p5GetInt(weight.dicts, "moe_top_k", num_experts_per_tok);
        norm_topk_prob = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "norm_expert_weight", "true"));
        routed_scaling_factor = Step3p5GetFloat(weight.dicts, "moe_router_scaling_factor", 1.0f);
        rms_norm_eps = Step3p5GetFloat(weight.dicts, "rms_norm_eps", rms_norm_eps);
        rope_base = Step3p5GetFloat(weight.dicts, "rope_theta", 10000.0f);
        rope_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.factor", 1.0f);
        sliding_window = Step3p5GetInt(weight.dicts, "sliding_window", sliding_window);
        use_moe_router_bias = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "use_moe_router_bias", "true"));
        need_fp32_gate = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "need_fp32_gate", "true"));
        llama3_original_max_position_embeddings = Step3p5GetFloat(weight.dicts, "rope_scaling.original_max_position_embeddings", 131072.0f);
        llama3_low_freq_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.low_freq_factor", 1.0f);
        llama3_high_freq_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.high_freq_factor", 32.0f);

        layer_types = Step3p5ParseStringList(Step3p5GetDict(weight.dicts, "layer_types", ""));
        if ((int)layer_types.size() < block_cnt) {
            layer_types.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                if (layer_types[i].empty()) {
                    layer_types[i] = (i % 4 == 0) ? "full_attention" : "sliding_attention";
                }
            }
        }

        moe_layers = Step3p5ParseIntSet(Step3p5GetDict(weight.dicts, "moe_layers_enum", ""));
        if (moe_layers.empty()) {
            for (int i = 1; i < block_cnt; i++) {
                moe_layers.insert(i);
            }
        }

        layer_rope_thetas = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "rope_theta", ""));
        if ((int)layer_rope_thetas.size() < block_cnt) {
            layer_rope_thetas.resize(block_cnt, rope_base);
        }

        std::vector<float> partialRotary = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "partial_rotary_factors", ""));
        layer_rotary_dims.resize(block_cnt, head_dim);
        for (int i = 0; i < block_cnt; i++) {
            float factor = (i < (int)partialRotary.size()) ? partialRotary[i] : 1.0f;
            layer_rotary_dims[i] = std::max(0, (int)(head_dim * factor + 1e-5f));
        }
        rotary_dim = layer_rotary_dims.empty() ? head_dim : layer_rotary_dims[0];

        swiglu_limits = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "swiglu_limits", ""));
        swiglu_limits_shared = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "swiglu_limits_shared", ""));

        for (int i = 0; i < block_cnt; i++) {
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );

            std::string denseGateName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string denseUpName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string denseGateupName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (Step3p5LayerLimit(swiglu_limits_shared, i) == 0.0f) {
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({denseGateName, denseUpName}, denseGateupName, std::string("linearSwiglu"))})
                );
            }

            std::string sharedGateName = "model.layers." + std::to_string(i) + ".share_expert.gate_proj.weight";
            std::string sharedUpName = "model.layers." + std::to_string(i) + ".share_expert.up_proj.weight";
            std::string sharedGateupName = "model.layers." + std::to_string(i) + ".share_expert.gateup_proj.weight";
            if (Step3p5LayerLimit(swiglu_limits_shared, i) == 0.0f) {
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({sharedGateName, sharedUpName}, sharedGateupName, std::string("linearSwiglu"))})
                );
            }

            if (IsMoeLayer(i)) {
                std::string moePrefix = "model.layers." + std::to_string(i) + ".moe.";
                this->moeLinears.insert(moePrefix + "gate_proj.weight");
                this->moeLinears.insert(moePrefix + "up_proj.weight");
                this->moeLinears.insert(moePrefix + "down_proj.weight");
            }
        }

        moeGateWeights.clear();
        moeUpWeights.clear();
        moeDownWeights.clear();
        weights.clear();
        biass.clear();
        moeWeightsPrepared = false;
        initialized_add1 = false;
    }

    int Step3p5Model::LayerAttentionHeads(int layer) const {
        return IsFullAttentionLayer(layer) ? base_attention_heads : sliding_attention_heads;
    }

    int Step3p5Model::LayerKeyValueHeads(int layer) const {
        return IsFullAttentionLayer(layer) ? base_key_value_heads : sliding_key_value_heads;
    }

    bool Step3p5Model::IsFullAttentionLayer(int layer) const {
        return layer >= 0 && layer < (int)layer_types.size() && layer_types[layer] == "full_attention";
    }

    bool Step3p5Model::IsMoeLayer(int layer) const {
        return moe_layers.find(layer) != moe_layers.end();
    }

    bool Step3p5Model::UseLlama3Rope(int layer) const {
        return IsFullAttentionLayer(layer) &&
               Step3p5GetDict(weight.dicts, "rope_scaling.rope_type", "") == "llama3";
    }

    void Step3p5Model::PrepareMoeWeights() {
        if (moeWeightsPrepared) {
            return;
        }
        moeGateWeights.resize(block_cnt);
        moeUpWeights.resize(block_cnt);
        moeDownWeights.resize(block_cnt);
        weights.clear();
        biass.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);
        for (int i = 0; i < block_cnt; i++) {
            if (!IsMoeLayer(i)) {
                continue;
            }
            std::string prefix = "model.layers." + std::to_string(i) + ".moe.";
            std::string gateSourceName = prefix + "gate_proj.weight";
            std::string upSourceName = prefix + "up_proj.weight";
            std::string downSourceName = prefix + "down_proj.weight";
            if (weight.weight.find(gateSourceName) == weight.weight.end() ||
                weight.weight.find(upSourceName) == weight.weight.end() ||
                weight.weight.find(downSourceName) == weight.weight.end()) {
                continue;
            }
            Data &gateSource = weight[gateSourceName];
            Data &upSource = weight[upSourceName];
            Data &downSource = weight[downSourceName];
            bool useDiskMergedMoe = gateSource.isDiskWeight || upSource.isDiskWeight || downSource.isDiskWeight;
            moeGateWeights[i].resize(num_experts);
            moeUpWeights[i].resize(num_experts);
            moeDownWeights[i].resize(num_experts);
            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            for (int j = 0; j < num_experts; j++) {
                std::string expertPrefix = prefix + "experts." + std::to_string(j) + ".";
                std::string gateName = expertPrefix + "gate_proj.weight";
                std::string upName = expertPrefix + "up_proj.weight";
                std::string downName = expertPrefix + "down_proj.weight";
                std::string gateupName = expertPrefix + "gateup_proj.weight";
                if (useDiskMergedMoe) {
                    if (weight.weight.find(gateName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[gateName], gateSource, gateName, j);
                    }
                    if (weight.weight.find(upName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[upName], upSource, upName, j);
                    }
                    if (weight.weight.find(downName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[downName], downSource, downName, j);
                    }
                    moeGateWeights[i][j] = &weight[gateName];
                    moeUpWeights[i][j] = &weight[upName];
                    moeDownWeights[i][j] = &weight[downName];
                    if (weight.weight.find(gateupName) == weight.weight.end()) {
                        Step3p5MakeGateUpWeight(weight.weight[gateupName], weight[gateName], weight[upName], gateupName);
                    }
                    weights[i].push_back(&weight[gateupName]);
                    weights[i].push_back(&weight[downName]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                } else {
                    if (weight.weight.find(gateupName) == weight.weight.end()) {
                        Data gateCopy, upCopy;
                        Step3p5MakeExpertCopy(gateCopy, gateSource, gateName, j);
                        Step3p5MakeExpertCopy(upCopy, upSource, upName, j);
                        Step3p5MakeGateUpWeight(weight.weight[gateupName], gateCopy, upCopy, gateupName);
                    }
                    if (weight.weight.find(downName) == weight.weight.end()) {
                        Step3p5MakeExpertCopy(weight.weight[downName], downSource, downName, j);
                    }
                    Data &gateup = weight[gateupName];
                    Data &down = weight[downName];
                    gateup.tpLinearType = TP_LINEAR_ROW;
                    gateup.tpPackType = TP_PACK_GATEUP;
                    down.tpLinearType = TP_LINEAR_COLUMN;
                    if (weight.weight.find(gateName) == weight.weight.end()) {
                        Step3p5MakeGateUpSliceView(weight.weight[gateName], gateup, gateName, 0, gateup.dims[0] / 2);
                    }
                    if (weight.weight.find(upName) == weight.weight.end()) {
                        Step3p5MakeGateUpSliceView(weight.weight[upName], gateup, upName, gateup.dims[0] / 2, gateup.dims[0] / 2);
                    }
                    moeGateWeights[i][j] = &weight[gateName];
                    moeUpWeights[i][j] = &weight[upName];
                    moeDownWeights[i][j] = &weight[downName];
                    weights[i].push_back(&gateup);
                    weights[i].push_back(&down);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
            if (!useDiskMergedMoe) {
                weight.weight.erase(gateSourceName);
                weight.weight.erase(upSourceName);
                weight.weight.erase(downSourceName);
            }
        }
        moeWeightsPrepared = true;
    }

    void Step3p5Model::ApplyStepRotary(Data &input, const Data &positionIds, int layer) {
        int curRotaryDim = layer < (int)layer_rotary_dims.size() ? layer_rotary_dims[layer] : head_dim;
        float theta = layer < (int)layer_rope_thetas.size() ? layer_rope_thetas[layer] : rope_base;
        if (UseLlama3Rope(layer)) {
            fastllm::Llama3RopeEncoding(input, positionIds, curRotaryDim, theta, rope_factor,
                                         llama3_original_max_position_embeddings,
                                         llama3_low_freq_factor, llama3_high_freq_factor);
            return;
        }
        float ropeScale = 1.0f;
        fastllm::RopeEncoding(input, positionIds, curRotaryDim, theta, ropeScale);
    }

    int Step3p5Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Step3p5Model::ForwardV2(int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        AssertInFastLLM(batch > 0, "Step3p5 batch should be positive.");
        AssertInFastLLM((int)seqLens.size() >= batch, "Step3p5 seqLens missing.");
        AssertInFastLLM((int)generationConfigs.size() >= batch, "Step3p5 generation configs missing.");
        bool all1 = true;
        int totalLen = 0;
        for (int b = 0; b < batch; b++) {
            int len = seqLens[b];
            all1 &= (len == 1);
            totalLen += len;
        }

        auto runSplitBatchForward = [&]() -> std::vector<int> {
            std::vector<int> ret;
            ret.reserve(batch);

            int inputOffset = 0;
            for (int b = 0; b < batch; b++) {
                Data curInputIds;
                Split(inputIds, 1, inputOffset, inputOffset + seqLens[b], curInputIds);
                inputOffset += seqLens[b];

                std::vector<Data*> curAttentionMask = {
                    b < (int)attentionMask.size() ? attentionMask[b] : nullptr
                };
                std::vector<Data*> curPositionIds = {
                    b < (int)positionIds.size() ? positionIds[b] : nullptr
                };
                std::vector<int> curSeqLens = {seqLens[b]};
                std::vector<GenerationConfig> curGenerationConfigs = {generationConfigs[b]};

                LastTokensManager curLastTokens;
                if (b < (int)lastTokens.units.size()) {
                    curLastTokens.units.push_back(lastTokens.units[b]);
                } else {
                    int lastN = generationConfigs[b].last_n <= 0 ? max_positions : generationConfigs[b].last_n;
                    curLastTokens = LastTokensManager(1, lastN);
                }

                std::vector<std::pair<Data*, Data*> > curPastKeyValues;
                curPastKeyValues.reserve(block_cnt);
                int pastOffset = b * block_cnt;
                AssertInFastLLM((int)pastKeyValues.size() >= pastOffset + block_cnt,
                                "Step3p5 pastKeyValues missing.");
                for (int i = 0; i < block_cnt; i++) {
                    curPastKeyValues.push_back(pastKeyValues[pastOffset + i]);
                }

                std::vector<std::vector<float>*> curLogits;
                std::vector<std::vector<float>*> *curLogitsPtr = nullptr;
                if (retLogits != nullptr) {
                    curLogits.push_back(b < (int)retLogits->size() ? (*retLogits)[b] : nullptr);
                    curLogitsPtr = &curLogits;
                }

                std::vector<int> curRet = ForwardV2(
                    1, curInputIds, curAttentionMask, curPositionIds, curSeqLens,
                    curPastKeyValues, curGenerationConfigs, curLastTokens, curLogitsPtr
                );
                ret.push_back(curRet[0]);
            }
            return ret;
        };

        auto canRunFusedBatchDecode = [&]() -> bool {
            if (batch <= 1 || !all1 || (int)pastKeyValues.size() < batch * block_cnt) {
                return false;
            }
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                    Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                    if (pastKey == nullptr || pastValue == nullptr ||
                        !pastKey->isPagedKVCache || !pastValue->isPagedKVCache ||
                        pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                        pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                        return false;
                    }
                    Data *cacheStorage = (Data*)pastKey->pagedKVCacheData;
                    if (cacheStorage->dataDevice != DataDevice::CUDA) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (batch > 1 && !canRunFusedBatchDecode()) {
            return runSplitBatchForward();
        }

        PrepareMoeWeights();

        AssertInFastLLM((int)positionIds.size() >= batch, "Step3p5 positionIds missing.");
        std::vector <Data> positionIdsCpu;
        positionIdsCpu.reserve(batch);
        for (int b = 0; b < batch; b++) {
            AssertInFastLLM(positionIds[b] != nullptr, "Step3p5 positionIds should not be null.");
            positionIdsCpu.emplace_back();
            positionIdsCpu.back().CopyFrom(*positionIds[b]);
            positionIdsCpu.back().ToDevice(DataDevice::CPU);
        }

        Data allPositionIds;
        std::vector <float> vPositionIds;
        if (all1) {
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu[b], 0));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu[b], i));
                }
            }
        }
        allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, totalLen}, vPositionIds));

        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
                std::string routerName = "model.layers." + std::to_string(i) + ".moe.gate.weight";
                if (need_fp32_gate && weight.weight.find(routerName) != weight.weight.end()) {
                    ToDataType(this->weight[routerName], DataType::FLOAT32);
                }
            }
            Step3p5Add1(this->weight["model.norm.weight"]);
            initialized_add1 = true;
        }

        Data hiddenStates;
        EmbeddingBlock((Data*)&inputIds, &this->weight["model.embed_tokens.weight"], &hiddenStates, this->dataType);

        Data attenInput, q, k, v, qkv, attenOutput, gate, gateRep, mergedQkv;
        Data w1, w2, w3, routerLogits, routerProb, expertIndex, expertScore;
        Data attenPart, moePart, moeFinal, shareOutput;
        Data tempInput, tempOutput;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            std::string inputRmsName = prefix + "input_layernorm.weight";
            std::string postRmsName = prefix + "post_attention_layernorm.weight";
            int qHeads = LayerAttentionHeads(i);
            int kvHeads = LayerKeyValueHeads(i);
            int qDim = qHeads * head_dim;
            int kvDim = kvHeads * head_dim;

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[mergeQkvWeightName], Data(), mergedQkv);
                Split(mergedQkv, -1, 0, qDim, q);
                Split(mergedQkv, -1, qDim, qDim + kvDim, k);
                Split(mergedQkv, -1, qDim + kvDim, qDim + kvDim * 2, v);
            } else {
                Linear(attenInput, weight[prefix + "self_attn.q_proj.weight"], Data(), q);
                Linear(attenInput, weight[prefix + "self_attn.k_proj.weight"], Data(), k);
                Linear(attenInput, weight[prefix + "self_attn.v_proj.weight"], Data(), v);
            }
            Linear(attenInput, weight[prefix + "self_attn.g_proj.weight"], Data(), gate);

            q.Reshape({bsz, seqlen, qHeads, head_dim});
            k.Reshape({bsz, seqlen, kvHeads, head_dim});
            v.Reshape({bsz, seqlen, kvHeads, head_dim});
            RMSNorm(q, this->weight[prefix + "self_attn.q_norm.weight"], rms_norm_eps, q);
            RMSNorm(k, this->weight[prefix + "self_attn.k_norm.weight"], rms_norm_eps, k);
            ApplyStepRotary(q, allPositionIds, i);
            ApplyStepRotary(k, allPositionIds, i);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});
            q.Reshape({-1, seqlen, head_dim});
            k.Reshape({-1, seqlen, head_dim});
            v.Reshape({-1, seqlen, head_dim});

            if (batch > 1 && all1) {
                for (int b = 0; b < batch; b++) {
                    batchPastKeys[b] = pastKeyValues[b * block_cnt + i].first;
                    batchPastValues[b] = pastKeyValues[b * block_cnt + i].second;
                }

                Data &kCaches = *batchPastKeys[0];
                Data &vCaches = *batchPastValues[0];
                PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                "Step3p5 fused batch decode requires paged KV cache.");

                if (!generatedAppendPagedCacheBatchParams) {
                    GenerateAppendPagedCacheBatchParams(*pagedCacheKManager, batchPastKeys, batch,
                                                        insertIndexs, insertPositions);
                    generatedAppendPagedCacheBatchParams = true;
                }

                Data kAppend, vAppend;
                Permute(k, {1, 0, 2}, kAppend);
                Permute(v, {1, 0, 2}, vAppend);
                AppendPagedCacheBatch(*pagedCacheKManager, batchPastKeys, kAppend, insertIndexs, insertPositions);
                AppendPagedCacheBatch(*pagedCacheVManager, batchPastValues, vAppend, insertIndexs, insertPositions);

                Data qForAttentionHolder;
                Data &qForAttention = Step3p5PreparePagedAttentionQ(q, kCaches.dataType, this->dataType, qForAttentionHolder);
                if (!generatedBatchDecodeParams) {
                    GeneratePagedBatchParams(qForAttention, batchPastKeys, batch,
                                             qSizes, pageSizes, pageIndexs, lastPageLens);
                    generatedBatchDecodeParams = true;
                }
                AttentionPagedBatch(qForAttention, kCaches, vCaches,
                                    qSizes, pageSizes, pageIndexs, lastPageLens,
                                    qkv, qForAttention.dims[0] / kCaches.dims[0],
                                    1.0f / sqrt((float)head_dim), 1, false);
            } else {
                Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
                PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                    i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                    i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                AppendPagedCache(*pagedCacheKManager, pastKey, k);
                AppendPagedCache(*pagedCacheVManager, pastValue, v);
                AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0],
                               1.0f / sqrt((float)head_dim), 1, false);
            }

            if (batch > 1 && all1) {
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            } else {
                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            }

            Sigmoid(gate, gate);
            gate.Reshape({bsz, seqlen, qHeads, 1});
            Repeat(gate, 3, head_dim, gateRep);
            qkv.Reshape({bsz, seqlen, qHeads, head_dim});
            if (gateRep.dataType != qkv.dataType) {
                ToDataType(gateRep, qkv.dataType);
            }
            MulTo(qkv, gateRep);
            qkv.Reshape({bsz, seqlen, qDim});

            Linear(qkv, weight[prefix + "self_attn.o_proj.weight"], Data(), attenInput);
            AddTo(hiddenStates, attenInput);

            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            if (!IsMoeLayer(i)) {
                std::string gateupName = prefix + "mlp.gateup_proj.weight";
                std::string downName = prefix + "mlp.down_proj.weight";
                float denseLimit = Step3p5LayerLimit(swiglu_limits_shared, i);
                if (denseLimit == 0.0f && weight.weight.find(gateupName) != weight.weight.end()) {
                    MLPBlock(&attenInput, &weight[gateupName], &weight[downName], &w3, &w1, &hiddenStates);
                } else {
                    Linear(attenInput, weight[prefix + "mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[prefix + "mlp.up_proj.weight"], Data(), w3);
                    if (denseLimit != 0.0f) {
                        Step3p5Clamp(w1, false, 0.0f, true, denseLimit);
                        Step3p5Clamp(w3, true, -denseLimit, true, denseLimit);
                    }
                    MulTo(w1, w3);
                    Linear(w1, weight[downName], Data(), w2);
                    AddTo(hiddenStates, w2);
                }
            } else {
                std::string sharedGateupName = prefix + "share_expert.gateup_proj.weight";
                std::string sharedDownName = prefix + "share_expert.down_proj.weight";
                float sharedLimit = Step3p5LayerLimit(swiglu_limits_shared, i);
                float expertLimit = Step3p5LayerLimit(swiglu_limits, i);
                if (sharedLimit == 0.0f && weight.weight.find(sharedGateupName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateupName], Data(), w3);
                    Swiglu(w3, w1);
                } else {
                    Linear(attenInput, weight[prefix + "share_expert.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[prefix + "share_expert.up_proj.weight"], Data(), w3);
                    if (sharedLimit != 0.0f) {
                        Step3p5Clamp(w1, false, 0.0f, true, sharedLimit);
                        Step3p5Clamp(w3, true, -sharedLimit, true, sharedLimit);
                    }
                    MulTo(w1, w3);
                }
                Linear(w1, weight[sharedDownName], Data(), shareOutput);

                int flatBatch = attenInput.dims[0];
                int flatLen = attenInput.dims[1];
                attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});
                Linear(attenInput, weight[prefix + "moe.gate.weight"], Data(), routerLogits);
                ToDataType(routerLogits, DataType::FLOAT32);
                Sigmoid(routerLogits, routerProb);
                Data *gateBias = nullptr;
                if (use_moe_router_bias && weight.weight.find(prefix + "moe.router_bias") != weight.weight.end()) {
                    gateBias = &weight[prefix + "moe.router_bias"];
                }
                SelectExpert(routerProb, expertIndex, expertScore, num_experts_per_tok, norm_topk_prob,
                             routed_scaling_factor, gateBias);

                routerProb.ToDevice(DataDevice::CPU);
                expertIndex.ToDevice(DataDevice::CPU);
                expertScore.ToDevice(DataDevice::CPU);
                ToDataType(expertScore, DataType::FLOAT32);
                bool useCudaMoe = Step3p5DeviceMapUsesCuda(this->moeDeviceMap);
                bool useDiskMoe = Step3p5DeviceMapUsesDisk(this->moeDeviceMap);
                if (i < (int)weights.size() && !weights[i].empty() && (useCudaMoe || useDiskMoe)) {
                    Data expertInput;
                    expertInput.CopyFrom(attenInput);
                    ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                    MergeMOE(expertInput, expertIndex, expertScore,
                             weights[i], biass[i],
                             w1, w2, w3, tempInput, tempOutput,
                             1.0f, moeFinal, i);
                    ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                } else {
                int32_t *indexData = (int32_t*)expertIndex.cpuData;
                float *scoreData = (float*)expertScore.cpuData;

                std::map<std::string, int> cpuDeviceMap = {{"cpu", 1}};
                Data expertInput;
                expertInput.CopyFrom(attenInput);
                expertInput.ToDevice(DataDevice::CPU);
                moeFinal.dataType = hiddenStates.dataType;
                moeFinal.dataDevice = expertInput.dataDevice;
                moeFinal.dataDeviceIds = expertInput.dataDeviceIds;
                moeFinal.UpdateUnitSize();
                moeFinal.Resize({0, expertInput.dims[1]});
                moeFinal.Expansion(expertInput.dims);
                ApplyDeviceMap(cpuDeviceMap, 1, 1);
                for (int b = 0; b < flatBatch * flatLen; b++) {
                    Data *currentData = &expertInput;
                    if (flatBatch * flatLen != 1) {
                        Split(expertInput, 0, b, b + 1, attenPart);
                        currentData = &attenPart;
                    }
                    moePart.dataType = hiddenStates.dataType;
                    moePart.dataDevice = currentData->dataDevice;
                    moePart.dataDeviceIds = currentData->dataDeviceIds;
                    moePart.UpdateUnitSize();
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);
                    for (int j = 0; j < num_experts_per_tok; j++) {
                        int expert = indexData[b * num_experts_per_tok + j];
                        float score = scoreData[b * num_experts_per_tok + j];
                        Linear(*currentData, *moeGateWeights[i][expert], Data(), w1);
                        Silu(w1, w1);
                        Linear(*currentData, *moeUpWeights[i][expert], Data(), w3);
                        if (expertLimit != 0.0f) {
                            Step3p5Clamp(w1, false, 0.0f, true, expertLimit);
                            Step3p5Clamp(w3, true, -expertLimit, true, expertLimit);
                        }
                        MulTo(w1, w3);
                        Linear(w1, *moeDownWeights[i][expert], Data(), w2);
                        if (w2.dataType != moePart.dataType) {
                            ToDataType(w2, moePart.dataType);
                        }
                        AddTo(moePart, w2, score);
                    }
                    if (moePart.dataType != moeFinal.dataType) {
                        ToDataType(moePart, moeFinal.dataType);
                    }
                    CatDirect(moeFinal, moePart, 0);
                }
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                }
                moeFinal.expansionDims.clear();
                moeFinal.Reshape(hiddenStates.dims);
                moeFinal.ToDevice(hiddenStates.dataDevice);
                if (moeFinal.dataType != hiddenStates.dataType) {
                    ToDataType(moeFinal, hiddenStates.dataType);
                }
                if (shareOutput.dataType != hiddenStates.dataType) {
                    ToDataType(shareOutput, hiddenStates.dataType);
                }
                AddTo(hiddenStates, moeFinal);
                AddTo(hiddenStates, shareOutput);
            }
        }

        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight["model.norm.weight"], &weight["lm_head.weight"],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool Step3p5Model::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    void Step3p5Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});
        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }

    std::string Step3p5Model::ApplyChatTemplate(const ChatMessages &messages) {
        std::string prompt = STEP3P5_BOS;
        for (auto &message : messages) {
            const std::string &role = message.first;
            const std::string &content = message.second;
            if (role == "system" || role == "user" || role == "assistant") {
                prompt += STEP3P5_IM_START + role + "\n" + content + STEP3P5_IM_END + "\n";
            } else if (role == "tool") {
                prompt += STEP3P5_IM_START + std::string("tool_response\n<tool_response>") +
                          content + "</tool_response>" + STEP3P5_IM_END + "\n";
            }
        }
        prompt += STEP3P5_IM_START + std::string("assistant\n<think>\n");
        return prompt;
    }

    std::string Step3p5Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Step3p5Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input +
               STEP3P5_IM_END + "\n" + STEP3P5_IM_START + "assistant\n" + output + history_sep;
    }
}
