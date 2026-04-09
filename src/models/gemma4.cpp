#include "utils.h"
#include "gemma4.h"

#include <sstream>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    Gemma4Model::Gemma4Model() {
        this->model_struct = "gemma4";
        this->model_type = "gemma4";
        this->use_new_engine = false;

        this->pre_prompt = "";
        this->user_role = "";
        this->bot_role = "";
        this->history_sep = "";

        block_cnt = 60;
        rotary_dim = 256;

        weight.embeddingNames.insert("model.language_model.embed_tokens.weight");
        weight.embeddingNames.insert("model.vision_tower.patch_embedder.position_embedding_table");
        weight.linearNames = {
            "lm_head.weight",
            "model.language_model.layers.*.mlp.down_proj.weight",
            "model.language_model.layers.*.mlp.up_proj.weight",
            "model.language_model.layers.*.mlp.gate_proj.weight",
            "model.language_model.layers.*.router.proj.weight",
            "model.language_model.layers.*.experts.gate_up_proj",
            "model.language_model.layers.*.experts.down_proj",
            "model.language_model.layers.*.self_attn.o_proj.weight",
            "model.language_model.layers.*.self_attn.q_proj.weight",
            "model.language_model.layers.*.self_attn.k_proj.weight",
            "model.language_model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.mlp.down_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.router.proj.weight",
            "model.layers.*.experts.gate_up_proj",
            "model.layers.*.experts.down_proj",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
            "model.vision_tower.patch_embedder.input_proj.weight",
            "model.vision_tower.encoder.layers.*.self_attn.q_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.self_attn.k_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.self_attn.v_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.self_attn.o_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.mlp.gate_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.mlp.up_proj.linear.weight",
            "model.vision_tower.encoder.layers.*.mlp.down_proj.linear.weight",
            "model.embed_vision.embedding_projection.weight"
        };
    }

    static std::string GetDictValue(const std::map<std::string, std::string> &dicts,
                                    const std::string &key1,
                                    const std::string &key2,
                                    const std::string &defaultVal = "") {
        auto it = dicts.find(key1);
        if (it != dicts.end()) return it->second;
        it = dicts.find(key2);
        if (it != dicts.end()) return it->second;
        return defaultVal;
    }

    static bool TryGetGemma4FusedMoeLayerPrefix(const std::string &weightName, std::string &layerPrefix) {
        static const std::string gateupSuffix = "experts.gate_up_proj";
        static const std::string downSuffix = "experts.down_proj";
        if (StringEndWith(weightName, gateupSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - gateupSuffix.size());
            return true;
        }
        if (StringEndWith(weightName, downSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - downSuffix.size());
            return true;
        }
        return false;
    }

    static void SplitGemma4ExpertLinearWeight(Data &dst, const Data &src, const std::string &name, int expertIndex) {
        AssertInFastLLM(src.dims.size() == 3, "Gemma4 fused expert weight should be 3D.");
        AssertInFastLLM(expertIndex >= 0 && expertIndex < src.dims[0], "Gemma4 expert index out of range.");
        AssertInFastLLM(src.dataType == DataType::FLOAT16 || src.dataType == DataType::BFLOAT16 ||
                        src.dataType == DataType::FLOAT32,
                        "Gemma4 fused expert slicing currently supports float16/bfloat16/float32 weights only.");
        AssertInFastLLM(src.dataDevice == DataDevice::CPU && src.cpuData != nullptr,
                        "Gemma4 fused expert slicing expects CPU weight data during load.");

        dst = Data(src.dataType, {src.dims[1], src.dims[2]});
        dst.Allocate();
        const uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expertIndex, bytesPerExpert);
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
    }

    static void MakeGemma4FlatLastDimView(Data &input, Data &output) {
        AssertInFastLLM((input.dims.size() == 2 || input.dims.size() == 3) &&
                        input.strides.size() == input.dims.size() &&
                        input.strides.back() == 1,
                        "Gemma4 flat last-dim view expects a contiguous 2D/3D tensor.");
        if (input.dims.size() == 2) {
            output.dims = input.dims;
            output.strides = input.strides;
        } else {
            output.dims = {input.dims[0] * input.dims[1], input.dims[2]};
            output.strides = {input.strides[1], input.strides[2]};
        }
        output.dataDeviceIds = input.dataDeviceIds;
        output.FakeFrom(input, 0);
    }

    static void MakeGemma4MatrixRowView(Data &input, int row, Data &output) {
        AssertInFastLLM(input.dims.size() == 2 && input.strides.size() == 2 && input.strides[1] == 1,
                        "Gemma4 row view expects a contiguous 2D tensor.");
        AssertInFastLLM(row >= 0 && row < input.dims[0], "Gemma4 row index is out of range.");
        output.dims = {1, input.dims[1]};
        output.strides = input.strides;
        output.dataDeviceIds = input.dataDeviceIds;
        output.FakeFrom(input, (size_t) row * input.strides[0] * input.unitSize);
    }

    static int ParseGemma4LayerIndex(const std::string &layerPrefix) {
        size_t pos = layerPrefix.rfind('.');
        AssertInFastLLM(pos != std::string::npos && pos + 1 < layerPrefix.size(),
                        "Gemma4 layer prefix is invalid.");
        return atoi(layerPrefix.substr(pos + 1).c_str());
    }

    static void BuildGemma4RouterTopKFromLogits(const Data &routerLogits,
                                                const Data &perExpertScale,
                                                int topK,
                                                std::vector<int> &expertIds,
                                                std::vector<float> &expertWeights) {
        Data logitsCpu(routerLogits);
        Data perExpertScaleCpu(perExpertScale);
        logitsCpu.ToDevice(DataDevice::CPU);
        perExpertScaleCpu.ToDevice(DataDevice::CPU);
        if (logitsCpu.dataType != DataType::FLOAT32) {
            ToDataType(logitsCpu, DataType::FLOAT32);
        }
        if (perExpertScaleCpu.dataType != DataType::FLOAT32) {
            ToDataType(perExpertScaleCpu, DataType::FLOAT32);
        }

        AssertInFastLLM(logitsCpu.dims.size() == 2 && perExpertScaleCpu.dims.size() == 1,
                        "Gemma4 router logits/scale tensors should be 2D and 1D.");
        const int tokenCount = logitsCpu.dims[0];
        const int expertCount = logitsCpu.dims[1];
        topK = std::max(0, std::min(topK, expertCount));
        AssertInFastLLM(perExpertScaleCpu.dims[0] == expertCount,
                        "Gemma4 per-expert scale shape is inconsistent with router logits.");

        expertIds.assign(tokenCount * topK, 0);
        expertWeights.assign(tokenCount * topK, 0.0f);
        if (topK == 0) {
            return;
        }

        float *logitsData = (float*) logitsCpu.cpuData;
        float *perExpertScaleData = (float*) perExpertScaleCpu.cpuData;
        std::vector<float> probs(expertCount);
        std::vector<int> order(expertCount);
        for (int token = 0; token < tokenCount; token++) {
            const float *tokenLogits = logitsData + (size_t) token * expertCount;
            float maxLogit = -1e30f;
            for (int expert = 0; expert < expertCount; expert++) {
                maxLogit = std::max(maxLogit, tokenLogits[expert]);
                order[expert] = expert;
            }
            float softmaxSum = 0.0f;
            for (int expert = 0; expert < expertCount; expert++) {
                probs[expert] = expf(tokenLogits[expert] - maxLogit);
                softmaxSum += probs[expert];
            }
            if (softmaxSum <= 0.0f) {
                softmaxSum = 1.0f;
            }
            for (int expert = 0; expert < expertCount; expert++) {
                probs[expert] /= softmaxSum;
            }
            std::partial_sort(order.begin(), order.begin() + topK, order.end(),
                              [&](int a, int b) {
                                  if (probs[a] == probs[b]) {
                                      return a < b;
                                  }
                                  return probs[a] > probs[b];
                              });
            float topProbSum = 0.0f;
            for (int k = 0; k < topK; k++) {
                topProbSum += probs[order[k]];
            }
            if (topProbSum <= 0.0f) {
                topProbSum = 1.0f;
            }
            for (int k = 0; k < topK; k++) {
                const int expert = order[k];
                expertIds[token * topK + k] = expert;
                expertWeights[token * topK + k] = probs[expert] / topProbSum * perExpertScaleData[expert];
            }
        }
    }

    void Gemma4Model::InitParams() {
        basellm::InitParams();
        auto &d = this->weight.dicts;

        std::string val;
        val = GetDictValue(d, "text_config.num_key_value_heads", "num_key_value_heads", "16");
        num_key_value_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_global_key_value_heads", "num_global_key_value_heads", "4");
        global_num_key_value_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.head_dim", "head_dim", "256");
        sliding_head_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.global_head_dim", "global_head_dim", "512");
        global_head_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.hidden_size", "hidden_size", "5376");
        embed_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_attention_heads", "num_attention_heads", "32");
        num_attention_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_hidden_layers", "num_hidden_layers", "60");
        block_cnt = atoi(val.c_str());

        val = GetDictValue(d, "text_config.sliding_window", "sliding_window", "1024");
        sliding_window = atoi(val.c_str());

        val = GetDictValue(d, "text_config.rms_norm_eps", "rms_norm_eps", "1e-6");
        rms_norm_eps = atof(val.c_str());

        val = GetDictValue(d, "text_config.attention_k_eq_v", "attention_k_eq_v", "true");
        attention_k_eq_v = (val == "true" || val == "True" || val == "1");

        val = GetDictValue(d, "text_config.enable_moe_block", "enable_moe_block", "false");
        enable_moe_block = (val == "true" || val == "True" || val == "1");

        val = GetDictValue(d, "text_config.num_experts", "num_experts", "0");
        num_experts = atoi(val.c_str());

        val = GetDictValue(d, "text_config.top_k_experts", "top_k_experts", "0");
        num_experts_per_tok = atoi(val.c_str());

        val = GetDictValue(d, "text_config.moe_intermediate_size", "moe_intermediate_size", "0");
        moe_intermediate_size = atoi(val.c_str());
        routed_scaling_factor = 1.0f;
        norm_topk_prob = true;
        moeWeightsPrepared = false;
        weights.clear();
        biass.clear();
        expertGateupWeights.clear();
        expertDownWeights.clear();

        val = GetDictValue(d, "text_config.final_logit_softcapping", "final_logit_softcapping", "30.0");
        if (val != "None" && val != "null" && !val.empty())
            final_logit_softcapping = atof(val.c_str());

        val = GetDictValue(d, "text_config.max_position_embeddings", "max_position_embeddings", "262144");
        max_positions = atoi(val.c_str());

        val = GetDictValue(d, "text_config.eos_token_id", "eos_token_id", "");
        if (!val.empty() && val != "None") {
            this->eos_token_id = atoi(val.c_str());
        }
        val = GetDictValue(d, "text_config.bos_token_id", "bos_token_id", "");
        if (!val.empty() && val != "None") {
            this->bos_token_id = atoi(val.c_str());
        }

        val = GetDictValue(d, "image_token_id", "image_token_id", "-1");
        image_token_id = atoi(val.c_str());
        val = GetDictValue(d, "boi_token_id", "boi_token_id", "-1");
        boi_token_id = atoi(val.c_str());
        val = GetDictValue(d, "eoi_token_id", "eoi_token_id", "-1");
        eoi_token_id = atoi(val.c_str());

        val = GetDictValue(d, "vision_config.hidden_size", "hidden_size", "1152");
        vision_hidden_size = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.num_hidden_layers", "num_hidden_layers", "27");
        vision_num_layers = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.num_attention_heads", "num_attention_heads", "16");
        vision_num_heads = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.num_key_value_heads", "num_key_value_heads", std::to_string(vision_num_heads));
        vision_num_key_value_heads = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.head_dim", "head_dim", "72");
        vision_head_dim = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.patch_size", "patch_size", "16");
        vision_patch_size = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.pooling_kernel_size", "pooling_kernel_size", "3");
        vision_pooling_kernel_size = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.position_embedding_size", "position_embedding_size", "10240");
        vision_position_embedding_size = atoi(val.c_str());
        val = GetDictValue(d, "vision_soft_tokens_per_image", "vision_soft_tokens_per_image", "280");
        vision_max_soft_tokens = atoi(val.c_str());
        val = GetDictValue(d, "vision_config.standardize", "standardize", "false");
        vision_standardize = (val == "true" || val == "True" || val == "1");

        layer_types.clear();
        val = GetDictValue(d, "text_config.layer_types", "layer_types", "");
        if (!val.empty() && val[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(val, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    if (item.string_value() == "full_attention") {
                        layer_types.push_back(1);
                    } else {
                        layer_types.push_back(0);
                    }
                }
            }
        }
        if ((int)layer_types.size() < block_cnt) {
            layer_types.clear();
            for (int i = 0; i < block_cnt; i++) {
                layer_types.push_back(((i + 1) % 6 == 0) ? 1 : 0);
            }
            layer_types[block_cnt - 1] = 1;
        }

        val = GetDictValue(d, "text_config.rope_parameters.sliding_attention.rope_theta",
                           "rope_parameters.sliding_attention.rope_theta", "10000.0");
        sliding_rope_base = atof(val.c_str());

        val = GetDictValue(d, "text_config.rope_parameters.full_attention.rope_theta",
                           "rope_parameters.full_attention.rope_theta", "1000000.0");
        global_rope_base = atof(val.c_str());

        val = GetDictValue(d, "text_config.rope_parameters.full_attention.partial_rotary_factor",
                           "rope_parameters.full_attention.partial_rotary_factor", "0.25");
        global_partial_rotary_factor = atof(val.c_str());

        head_dim = sliding_head_dim;
        rotary_dim = sliding_head_dim;

        {
            std::vector<float> onesGlobal(global_head_dim, 1.0f);
            weight["__gemma4_v_norm_global.weight"].CopyFrom(
                Data(DataType::FLOAT32, {global_head_dim}, onesGlobal)
            );
            std::vector<float> onesSliding(sliding_head_dim, 1.0f);
            weight["__gemma4_v_norm_sliding.weight"].CopyFrom(
                Data(DataType::FLOAT32, {sliding_head_dim}, onesSliding)
            );
        }

        int maxPos = std::max(max_positions, 16384);
        {
            auto pair = UpdateRotaryPosEmb(sliding_rope_base, 1.0f, maxPos, sliding_head_dim);
            slidingSinData.ToDevice(DataDevice::CPU);
            slidingCosData.ToDevice(DataDevice::CPU);
            slidingSinData.CopyFrom(Data(DataType::FLOAT32,
                {(int)slidingSin.size(), (int)slidingSin[0].size()}, pair.first));
            slidingCosData.CopyFrom(Data(DataType::FLOAT32,
                {(int)slidingCos.size(), (int)slidingCos[0].size()}, pair.second));
        }

        {
            int ropeAngles = (int)(global_partial_rotary_factor * global_head_dim / 2);
            int totalHalf = global_head_dim / 2;
            int nopeAngles = totalHalf - ropeAngles;

            std::vector<float> invFreq;
            for (int i = 0; i < ropeAngles; i++) {
                invFreq.push_back(1.0f / pow(global_rope_base, (float)(2 * i) / (float)global_head_dim));
            }
            for (int i = 0; i < nopeAngles; i++) {
                invFreq.push_back(0.0f);
            }

            int positions = std::max(max_positions, 16384);
            globalSin.resize(positions);
            globalCos.resize(positions);
            for (int p = 0; p < positions; p++) {
                globalSin[p].resize(totalHalf);
                globalCos[p].resize(totalHalf);
                for (int j = 0; j < totalHalf; j++) {
                    float angle = (float)p * invFreq[j];
                    globalSin[p][j] = ::sin(angle);
                    globalCos[p][j] = ::cos(angle);
                }
            }

            std::vector<float> fsin, fcos;
            for (int i = 0; i < (int)globalSin.size(); i++) {
                fsin.insert(fsin.end(), globalSin[i].begin(), globalSin[i].end());
                fcos.insert(fcos.end(), globalCos[i].begin(), globalCos[i].end());
            }
            globalSinData.ToDevice(DataDevice::CPU);
            globalCosData.ToDevice(DataDevice::CPU);
            globalSinData.CopyFrom(Data(DataType::FLOAT32,
                {(int)globalSin.size(), (int)globalSin[0].size()}, fsin));
            globalCosData.CopyFrom(Data(DataType::FLOAT32,
                {(int)globalCos.size(), (int)globalCos[0].size()}, fcos));
        }

        {
            float visionRopeBase = atof(GetDictValue(d,
                "vision_config.rope_parameters.rope_theta",
                "rope_parameters.rope_theta", "100.0").c_str());
            int maxVisionPos = std::max(vision_position_embedding_size, 1);
            int rotaryHalf = vision_head_dim / 4;
            std::vector<float> invFreq;
            for (int i = 0; i < vision_head_dim / 2; i += 2) {
                invFreq.push_back(1.0f / pow(visionRopeBase, (float)i / (vision_head_dim / 2)));
            }

            std::vector<float> visionSin, visionCos;
            visionSin.reserve((size_t)maxVisionPos * rotaryHalf);
            visionCos.reserve((size_t)maxVisionPos * rotaryHalf);
            for (int p = 0; p < maxVisionPos; p++) {
                for (int j = 0; j < rotaryHalf; j++) {
                    float angle = (float)p * invFreq[j];
                    visionSin.push_back(::sin(angle));
                    visionCos.push_back(::cos(angle));
                }
            }
            visionSinData.ToDevice(DataDevice::CPU);
            visionCosData.ToDevice(DataDevice::CPU);
            visionSinData.CopyFrom(Data(DataType::FLOAT32, {maxVisionPos, rotaryHalf}, visionSin));
            visionCosData.CopyFrom(Data(DataType::FLOAT32, {maxVisionPos, rotaryHalf}, visionCos));
        }
    }

    std::pair<std::vector<float>, std::vector<float>> Gemma4Model::UpdateRotaryPosEmb(float base, float factor, int seqLen, int dim) {
        int positions = std::max(max_positions, seqLen);
        int halfDim = dim / 2;
        slidingSin.resize(positions);
        slidingCos.resize(positions);
        std::vector<float> invFreq;
        for (int i = 0; i < dim; i += 2) {
            invFreq.push_back(1.0f / pow(base, (float)i / dim));
        }
        for (int i = 0; i < positions; i++) {
            slidingSin[i].resize(halfDim);
            slidingCos[i].resize(halfDim);
            for (int j = 0; j < halfDim; j++) {
                float angle = (float)i * invFreq[j];
                slidingSin[i][j] = ::sin(angle);
                slidingCos[i][j] = ::cos(angle);
            }
        }
        std::vector<float> fsin, fcos;
        for (int i = 0; i < (int)slidingSin.size(); i++) {
            fsin.insert(fsin.end(), slidingSin[i].begin(), slidingSin[i].end());
            fcos.insert(fcos.end(), slidingCos[i].begin(), slidingCos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    void Gemma4Model::PrepareVision() {
        if (visionPrepared) {
            return;
        }
        AssertInFastLLM(this->weight.weight.find("model.vision_tower.patch_embedder.input_proj.weight") != this->weight.weight.end(),
                        "Gemma4 multimodal needs model.vision_tower.patch_embedder.input_proj.weight.");
        AssertInFastLLM(this->weight.weight.find("model.embed_vision.embedding_projection.weight") != this->weight.weight.end(),
                        "Gemma4 multimodal needs model.embed_vision.embedding_projection.weight.");
        visionPrepared = true;
    }

    void Gemma4Model::ApplyRMSNormNoScale(Data &input, float eps) {
        input.ToDevice(DataDevice::CPU);
        if (input.dataType != DataType::FLOAT32) {
            ToDataType(input, DataType::FLOAT32);
        }
        float *data = (float*) input.cpuData;
        int lastDim = input.dims.back();
        uint64_t total = input.Count(0);
        int outer = (int) (total / lastDim);
        for (int o = 0; o < outer; o++) {
            float *row = data + (uint64_t) o * lastDim;
            float sumSq = 0.0f;
            for (int j = 0; j < lastDim; j++) {
                sumSq += row[j] * row[j];
            }
            float scale = 1.0f / sqrtf(sumSq / lastDim + eps);
            for (int j = 0; j < lastDim; j++) {
                row[j] *= scale;
            }
        }
    }

    void Gemma4Model::BuildVisionPatchPositionIds(const Data &imagePositionIds, Data &posX, Data &posY,
                                                  std::vector<int> &validPatchCounts) {
        Data posCpu(imagePositionIds);
        posCpu.ToDevice(DataDevice::CPU);
        if (posCpu.dataType != DataType::FLOAT32) {
            ToDataType(posCpu, DataType::FLOAT32);
        }
        AssertInFastLLM(posCpu.dims.size() == 3 && posCpu.dims[2] == 2,
                        "Gemma4 image_position_ids should have shape [batch, max_patches, 2].");

        int batch = posCpu.dims[0];
        int seqLen = posCpu.dims[1];
        validPatchCounts.assign(batch, 0);
        std::vector<float> vx(batch * seqLen, 0.0f), vy(batch * seqLen, 0.0f);
        float *posData = (float*) posCpu.cpuData;
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                float x = posData[(b * seqLen + i) * 2];
                float y = posData[(b * seqLen + i) * 2 + 1];
                if (x >= 0.0f && y >= 0.0f) {
                    vx[b * seqLen + i] = x;
                    vy[b * seqLen + i] = y;
                    validPatchCounts[b]++;
                }
            }
        }
        posX.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen}, vx));
        posY.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen}, vy));
    }

    void Gemma4Model::BuildVisionAttentionMask(const Data &imagePositionIds, Data &visionAttentionMask) {
        Data posCpu(imagePositionIds);
        posCpu.ToDevice(DataDevice::CPU);
        if (posCpu.dataType != DataType::FLOAT32) {
            ToDataType(posCpu, DataType::FLOAT32);
        }
        AssertInFastLLM(posCpu.dims.size() == 3 && posCpu.dims[2] == 2,
                        "Gemma4 image_position_ids should have shape [batch, max_patches, 2].");

        int batch = posCpu.dims[0];
        int seqLen = posCpu.dims[1];
        std::vector<char> valid(batch * seqLen, 0);
        float *posData = (float*) posCpu.cpuData;
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < seqLen; i++) {
                float x = posData[(b * seqLen + i) * 2];
                float y = posData[(b * seqLen + i) * 2 + 1];
                valid[b * seqLen + i] = (x >= 0.0f && y >= 0.0f) ? 1 : 0;
            }
        }

        std::vector<float> mask((size_t) batch * seqLen * seqLen, 0.0f);
        for (int b = 0; b < batch; b++) {
            for (int q = 0; q < seqLen; q++) {
                for (int k = 0; k < seqLen; k++) {
                    if (!valid[b * seqLen + q] || !valid[b * seqLen + k]) {
                        mask[((size_t) b * seqLen + q) * seqLen + k] = 1.0f;
                    }
                }
            }
        }
        visionAttentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, seqLen, seqLen}, mask));
    }

    void Gemma4Model::ApplyVisionRotary(Data &input, const Data &posX, const Data &posY) {
        AssertInFastLLM(input.dims.size() == 4 && input.dims.back() % 4 == 0,
                        "Gemma4 vision rotary expects [batch, seq, heads, dim] with dim divisible by 4.");
        int axis = (int) input.dims.size() - 1;
        int half = input.dims.back() / 2;
        int rotaryHalf = input.dims.back() / 4;
        Data xPart, yPart, rotated;
        Split(input, axis, 0, half, xPart);
        Split(input, axis, half, input.dims.back(), yPart);
        LlamaRotatePosition2DPart(xPart, posX, visionSinData, visionCosData, rotaryHalf, half);
        LlamaRotatePosition2DPart(yPart, posY, visionSinData, visionCosData, rotaryHalf, half);
        Cat(xPart, yPart, axis, rotated);
        input.CopyFrom(rotated);
    }

    void Gemma4Model::EncodeImages(const Data &pixelValues, const Data &imagePositionIds, Data &imageFeatures,
                                   std::vector<int> &softTokenCounts) {
        PrepareVision();
        AssertInFastLLM(pixelValues.dims.size() == 3, "Gemma4 pixel_values should have shape [batch, max_patches, patch_dim].");
        AssertInFastLLM(pixelValues.dims[0] == 1, "Gemma4 multimodal MVP currently supports a single image batch only.");

        Data patchPosX, patchPosY, visionMask;
        std::vector<int> validPatchCounts;
        BuildVisionPatchPositionIds(imagePositionIds, patchPosX, patchPosY, validPatchCounts);
        BuildVisionAttentionMask(imagePositionIds, visionMask);

        Data pixelInput(pixelValues);
        pixelInput.ToDevice(DataDevice::CPU);
        if (pixelInput.dataType != DataType::FLOAT32) {
            ToDataType(pixelInput, DataType::FLOAT32);
        }
        float *pixelPtr = (float*) pixelInput.cpuData;
        for (uint64_t i = 0; i < pixelInput.Count(0); i++) {
            pixelPtr[i] = 2.0f * (pixelPtr[i] - 0.5f);
        }

        const std::string patchProjName = "model.vision_tower.patch_embedder.input_proj.weight";
        pixelInput.ToDevice(this->weight[patchProjName].dataDevice);
        Data hiddenStates;
        Linear(pixelInput, this->weight[patchProjName], *GetEmptyData(), hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        Data posTableX, posTableY;
        Split(this->weight["model.vision_tower.patch_embedder.position_embedding_table"], 0, 0, 1, posTableX);
        Split(this->weight["model.vision_tower.patch_embedder.position_embedding_table"], 0, 1, 2, posTableY);
        posTableX.Reshape({vision_position_embedding_size, vision_hidden_size});
        posTableY.Reshape({vision_position_embedding_size, vision_hidden_size});
        Data posEmbedX, posEmbedY, posEmbeddings;
        Embedding(patchPosX, posTableX, posEmbedX);
        Embedding(patchPosY, posTableY, posEmbedY);
        posEmbeddings.CopyFrom(posEmbedX);
        AddTo(posEmbeddings, posEmbedY);

        Data posCpu(imagePositionIds);
        posCpu.ToDevice(DataDevice::CPU);
        if (posCpu.dataType != DataType::FLOAT32) {
            ToDataType(posCpu, DataType::FLOAT32);
        }
        posEmbeddings.ToDevice(DataDevice::CPU);
        if (posEmbeddings.dataType != DataType::FLOAT32) {
            ToDataType(posEmbeddings, DataType::FLOAT32);
        }
        float *posData = (float*) posCpu.cpuData;
        float *embData = (float*) posEmbeddings.cpuData;
        int seqLen = posCpu.dims[1];
        for (int i = 0; i < seqLen; i++) {
            if (posData[i * 2] < 0.0f || posData[i * 2 + 1] < 0.0f) {
                memset(embData + (size_t) i * vision_hidden_size, 0, (size_t) vision_hidden_size * sizeof(float));
            }
        }
        posEmbeddings.ToDevice(hiddenStates.dataDevice);
        if (posEmbeddings.dataType != hiddenStates.dataType) {
            ToDataType(posEmbeddings, hiddenStates.dataType);
        }
        AddTo(hiddenStates, posEmbeddings);

        visionMask.ToDevice(hiddenStates.dataDevice);
        Data attenInput, q, k, v, qkv, output, residual, ffResidual, gate, up, down;
        for (int i = 0; i < vision_num_layers; i++) {
            std::string pre = "model.vision_tower.encoder.layers." + std::to_string(i);
            int batch = hiddenStates.dims[0];
            int curSeqLen = hiddenStates.dims[1];

            Mul(hiddenStates, 1.0f, residual);
            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            Linear(attenInput, this->weight[pre + ".self_attn.q_proj.linear.weight"], *GetEmptyData(), q);
            Linear(attenInput, this->weight[pre + ".self_attn.k_proj.linear.weight"], *GetEmptyData(), k);
            Linear(attenInput, this->weight[pre + ".self_attn.v_proj.linear.weight"], *GetEmptyData(), v);

            q.Reshape({batch, curSeqLen, vision_num_heads, vision_head_dim});
            k.Reshape({batch, curSeqLen, vision_num_key_value_heads, vision_head_dim});
            v.Reshape({batch, curSeqLen, vision_num_key_value_heads, vision_head_dim});

            RMSNorm(q, this->weight[pre + ".self_attn.q_norm.weight"], rms_norm_eps, q);
            RMSNorm(k, this->weight[pre + ".self_attn.k_norm.weight"], rms_norm_eps, k);
            ApplyRMSNormNoScale(v, rms_norm_eps);
            v.ToDevice(q.dataDevice);
            if (v.dataType != q.dataType) {
                ToDataType(v, q.dataType);
            }

            ApplyVisionRotary(q, patchPosX, patchPosY);
            ApplyVisionRotary(k, patchPosX, patchPosY);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            q.Reshape({-1, curSeqLen, vision_head_dim});
            k.Reshape({-1, curSeqLen, vision_head_dim});
            v.Reshape({-1, curSeqLen, vision_head_dim});

            Attention(q, k, v, visionMask, qkv, q.dims[0] / k.dims[0], 1.0f, 2);
            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({curSeqLen, batch, -1});
            PermuteSelf(qkv, {1, 0, 2});

            Linear(qkv, this->weight[pre + ".self_attn.o_proj.linear.weight"], *GetEmptyData(), output);
            RMSNorm(output, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, output);
            AddTo(residual, output);
            hiddenStates.CopyFrom(residual);

            Mul(hiddenStates, 1.0f, ffResidual);
            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);
            Linear(attenInput, this->weight[pre + ".mlp.gate_proj.linear.weight"], *GetEmptyData(), gate);
            Gelu(gate, gate);
            Linear(attenInput, this->weight[pre + ".mlp.up_proj.linear.weight"], *GetEmptyData(), up);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(gate, DataType::FLOAT32);
                ToDataType(up, DataType::FLOAT32);
            }
            MulTo(gate, up);
            Linear(gate, this->weight[pre + ".mlp.down_proj.linear.weight"], *GetEmptyData(), down);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(down, DataType::FLOAT16);
            }
            RMSNorm(down, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, down);
            AddTo(ffResidual, down);
            hiddenStates.CopyFrom(ffResidual);
        }

        Data hiddenCpu(hiddenStates);
        hiddenCpu.ToDevice(DataDevice::CPU);
        if (hiddenCpu.dataType != DataType::FLOAT32) {
            ToDataType(hiddenCpu, DataType::FLOAT32);
        }
        float *hiddenPtr = (float*) hiddenCpu.cpuData;
        float *posRaw = (float*) posCpu.cpuData;
        int outputLength = pixelValues.dims[1] / (vision_pooling_kernel_size * vision_pooling_kernel_size);
        AssertInFastLLM(outputLength > 0, "Gemma4 vision output length must be positive.");

        std::vector<float> pooled;
        int validSoftTokens = 0;
        {
            int maxX = 0;
            for (int i = 0; i < seqLen; i++) {
                if (posRaw[i * 2] >= 0.0f) {
                    maxX = std::max(maxX, (int) posRaw[i * 2] + 1);
                }
            }
            int gridWidth = std::max(1, maxX / vision_pooling_kernel_size);
            std::vector<float> pooledFull((size_t) outputLength * vision_hidden_size, 0.0f);
            std::vector<int> pooledMask(outputLength, 0);
            for (int i = 0; i < seqLen; i++) {
                int x = (int) posRaw[i * 2];
                int y = (int) posRaw[i * 2 + 1];
                if (x < 0 || y < 0) {
                    continue;
                }
                int kernelIdx = (x / vision_pooling_kernel_size) + gridWidth * (y / vision_pooling_kernel_size);
                AssertInFastLLM(kernelIdx >= 0 && kernelIdx < outputLength,
                                "Gemma4 pooled kernel index is out of range.");
                pooledMask[kernelIdx] = 1;
                float *src = hiddenPtr + (size_t) i * vision_hidden_size;
                float *dst = pooledFull.data() + (size_t) kernelIdx * vision_hidden_size;
                for (int j = 0; j < vision_hidden_size; j++) {
                    dst[j] += src[j] / (vision_pooling_kernel_size * vision_pooling_kernel_size);
                }
            }
            float scale = sqrtf((float) vision_hidden_size);
            for (int idx = 0; idx < outputLength; idx++) {
                if (!pooledMask[idx]) {
                    continue;
                }
                validSoftTokens++;
                float *src = pooledFull.data() + (size_t) idx * vision_hidden_size;
                for (int j = 0; j < vision_hidden_size; j++) {
                    pooled.push_back(src[j] * scale);
                }
            }
        }
        AssertInFastLLM(validSoftTokens > 0, "Gemma4 produced no valid vision soft tokens.");
        softTokenCounts = {validSoftTokens};
        AssertInFastLLM(validSoftTokens <= vision_max_soft_tokens,
                        "Gemma4 produced more soft tokens than configured vision_max_soft_tokens.");

        if (vision_standardize &&
            this->weight.weight.find("model.vision_tower.std_bias") != this->weight.weight.end() &&
            this->weight.weight.find("model.vision_tower.std_scale") != this->weight.weight.end()) {
            Data stdBias(this->weight["model.vision_tower.std_bias"]);
            Data stdScale(this->weight["model.vision_tower.std_scale"]);
            stdBias.ToDevice(DataDevice::CPU);
            stdScale.ToDevice(DataDevice::CPU);
            if (stdBias.dataType != DataType::FLOAT32) {
                ToDataType(stdBias, DataType::FLOAT32);
            }
            if (stdScale.dataType != DataType::FLOAT32) {
                ToDataType(stdScale, DataType::FLOAT32);
            }
            float *bias = (float*) stdBias.cpuData;
            float *scale = (float*) stdScale.cpuData;
            for (int i = 0; i < validSoftTokens; i++) {
                float *row = pooled.data() + (size_t) i * vision_hidden_size;
                for (int j = 0; j < vision_hidden_size; j++) {
                    row[j] = (row[j] - bias[j]) * scale[j];
                }
            }
        }

        Data pooledData(DataType::FLOAT32, {1, validSoftTokens, vision_hidden_size}, pooled);
        ApplyRMSNormNoScale(pooledData, rms_norm_eps);
        pooledData.ToDevice(this->weight["model.embed_vision.embedding_projection.weight"].dataDevice);
        Linear(pooledData, this->weight["model.embed_vision.embedding_projection.weight"], *GetEmptyData(), imageFeatures);
        ToDataType(imageFeatures, this->dataType);
    }

    void Gemma4Model::MergeImageFeaturesIntoText(const Data &inputIds, const Data &imageFeatures, Data &hiddenStates) {
        Data idsCpu(inputIds);
        idsCpu.ToDevice(DataDevice::CPU);
        if (idsCpu.dataType != DataType::FLOAT32) {
            ToDataType(idsCpu, DataType::FLOAT32);
        }
        hiddenStates.ToDevice(DataDevice::CPU);
        if (hiddenStates.dataType != DataType::FLOAT32) {
            ToDataType(hiddenStates, DataType::FLOAT32);
        }
        Data imageCpu(imageFeatures);
        imageCpu.ToDevice(DataDevice::CPU);
        if (imageCpu.dataType != DataType::FLOAT32) {
            ToDataType(imageCpu, DataType::FLOAT32);
        }

        int batch = idsCpu.dims[0];
        int seqLen = idsCpu.dims[1];
        int featureCount = imageCpu.dims.size() == 3 ? imageCpu.dims[1] : imageCpu.dims[0];
        AssertInFastLLM(batch == 1, "Gemma4 multimodal MVP currently supports a single text batch only.");
        std::vector<int> imagePositions;
        float *idPtr = (float*) idsCpu.cpuData;
        for (int i = 0; i < seqLen; i++) {
            if ((int) idPtr[i] == image_token_id) {
                imagePositions.push_back(i);
            }
        }
        AssertInFastLLM((int) imagePositions.size() == featureCount,
                        "Gemma4 image feature count does not match image_token placeholders.");

        float *hiddenPtr = (float*) hiddenStates.cpuData;
        float *imagePtr = (float*) imageCpu.cpuData;
        int hiddenSize = hiddenStates.dims[2];
        for (int i = 0; i < featureCount; i++) {
            memcpy(hiddenPtr + (size_t) imagePositions[i] * hiddenSize,
                   imagePtr + (size_t) i * hiddenSize,
                   (size_t) hiddenSize * sizeof(float));
        }
    }

    void Gemma4Model::BuildVisionAwareTextMask(const Data &attentionMask, const Data &mmTokenTypeIds, Data &visionAwareMask) {
        if (attentionMask.dims.size() != 3) {
            visionAwareMask.CopyFrom(attentionMask);
            return;
        }
        Data maskCpu(attentionMask);
        maskCpu.ToDevice(DataDevice::CPU);
        if (maskCpu.dataType != DataType::FLOAT32) {
            ToDataType(maskCpu, DataType::FLOAT32);
        }
        Data mmCpu(mmTokenTypeIds);
        mmCpu.ToDevice(DataDevice::CPU);
        if (mmCpu.dataType != DataType::FLOAT32) {
            ToDataType(mmCpu, DataType::FLOAT32);
        }

        int batch = maskCpu.dims[0];
        int seqLen = maskCpu.dims[1];
        std::vector<float> mask((float*) maskCpu.cpuData, (float*) maskCpu.cpuData + maskCpu.Count(0));
        float *mmPtr = (float*) mmCpu.cpuData;
        for (int b = 0; b < batch; b++) {
            std::vector<int> groupIds(seqLen, -1);
            int group = -1;
            for (int i = 0; i < seqLen; i++) {
                bool isVision = mmPtr[b * seqLen + i] > 0.5f;
                bool isPrevVision = (i > 0 && mmPtr[b * seqLen + i - 1] > 0.5f);
                if (isVision && !isPrevVision) {
                    group++;
                }
                if (isVision) {
                    groupIds[i] = group;
                }
            }
            for (int q = 0; q < seqLen; q++) {
                if (groupIds[q] < 0) {
                    continue;
                }
                for (int k = 0; k < seqLen; k++) {
                    if (groupIds[q] == groupIds[k] && groupIds[k] >= 0) {
                        mask[((size_t) b * seqLen + q) * seqLen + k] = 0.0f;
                    }
                }
            }
        }
        visionAwareMask.CopyFrom(Data(DataType::FLOAT32, maskCpu.dims, mask));
    }

    int Gemma4Model::ForwardTextFromHiddenStates(const Data &inputIds,
                                                 Data &hiddenStates,
                                                 const Data &attentionMask,
                                                 const Data &positionIds,
                                                 std::vector <std::pair <Data, Data> > &pastKeyValues,
                                                 const GenerationConfig &generationConfig,
                                                 const LastTokensManager &lastTokens,
                                                 std::vector <float> *retLogits) {
        int maxLen = hiddenStates.dims[1];
        Data attenInput;
        Data q, k, v, qkv;
        Data attenInputOut;
        Data w1, w2, w3;

        int seqlen = hiddenStates.dims[1];
        std::string layerPrefix = "model.language_model.layers.";
        if (weight.weight.find(layerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            layerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            bool isFullAttn = (layer_types[i] == 1);
            int curHeadDim = isFullAttn ? global_head_dim : sliding_head_dim;
            int curKVHeads = (isFullAttn && attention_k_eq_v) ? global_num_key_value_heads : num_key_value_heads;
            int curRotaryDim = isFullAttn ? global_head_dim / 2 : sliding_head_dim;
            Data *curSinData = isFullAttn ? &globalSinData : &slidingSinData;
            Data *curCosData = isFullAttn ? &globalCosData : &slidingCosData;

            std::string pre = layerPrefix + std::to_string(i);
            std::string qWeightName = pre + ".self_attn.q_proj.weight";
            std::string kWeightName = pre + ".self_attn.k_proj.weight";
            std::string vWeightName = pre + ".self_attn.v_proj.weight";
            std::string oWeightName = pre + ".self_attn.o_proj.weight";
            std::string qNormName = pre + ".self_attn.q_norm.weight";
            std::string kNormName = pre + ".self_attn.k_norm.weight";

            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], *GetEmptyData(), q);
            Linear(attenInput, weight[kWeightName], *GetEmptyData(), k);
            bool useAltAttn = isFullAttn && attention_k_eq_v;
            if (useAltAttn) {
                v.CopyFrom(k);
            } else {
                Linear(attenInput, weight[vWeightName], *GetEmptyData(), v);
            }

            q.Reshape({bsz, seqlen, -1, curHeadDim});
            k.Reshape({bsz, seqlen, -1, curHeadDim});
            v.Reshape({bsz, seqlen, -1, curHeadDim});

            RMSNorm(q, this->weight[qNormName], rms_norm_eps, q);
            RMSNorm(k, this->weight[kNormName], rms_norm_eps, k);
            ApplyRMSNormNoScale(v, rms_norm_eps);
            v.ToDevice(k.dataDevice);
            if (v.dataType != k.dataType) {
                ToDataType(v, k.dataType);
            }

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(k.dataDevice);
                pastValue.ToDevice(k.dataDevice);
            }

            fastllm::LlamaRotatePosition2D(q, positionIds, *curSinData, *curCosData, curRotaryDim);
            fastllm::LlamaRotatePosition2D(k, positionIds, *curSinData, *curCosData, curRotaryDim);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            q.Reshape({-1, seqlen, curHeadDim});
            k.Reshape({-1, seqlen, curHeadDim});
            v.Reshape({-1, seqlen, curHeadDim});

            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0f, 1);
            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});

            Linear(qkv, weight[oWeightName], *GetEmptyData(), attenInputOut);
            RMSNorm(attenInputOut, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, attenInputOut);
            AddTo(hiddenStates, attenInputOut);

            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);
            Linear(attenInput, weight[pre + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
            Gelu(w1, w1);
            Linear(attenInput, weight[pre + ".mlp.up_proj.weight"], *GetEmptyData(), w3);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w1, DataType::FLOAT32);
                ToDataType(w3, DataType::FLOAT32);
            }
            MulTo(w1, w3);
            Linear(w1, weight[pre + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w2, DataType::FLOAT16);
            }
            if (!TryApplyMoeFeedForward(pre, hiddenStates, w2)) {
                RMSNorm(w2, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, w2);
                AddTo(hiddenStates, w2);
            }

            std::string layerScalarName = layerPrefix + std::to_string(i) + ".layer_scalar";
            if (weight.weight.find(layerScalarName) != weight.weight.end()) {
                Data &ls = weight[layerScalarName];
                ls.ToDevice(DataDevice::CPU);
                if (ls.dataType != DataType::FLOAT32) {
                    ToDataType(ls, DataType::FLOAT32);
                }
                float scalar = ((float*) ls.cpuData)[0];
                Mul(hiddenStates, scalar, hiddenStates);
            }
        }

        Data logits, topk, tempHiddenStates;
        Data *lastHiddenStates = &hiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        }

        std::string normName = "model.language_model.norm.weight";
        if (weight.weight.find(normName) == weight.weight.end()) {
            normName = "model.norm.weight";
        }
        RMSNorm(*lastHiddenStates, this->weight[normName], rms_norm_eps, *lastHiddenStates);

        Linear(*lastHiddenStates, weight["lm_head.weight"], *GetEmptyData(), logits);
        ToDataType(logits, DataType::FLOAT32);
        if (final_logit_softcapping > 0.0f) {
            Mul(logits, 1.0f / final_logit_softcapping, logits);
            logits.ToDevice(DataDevice::CPU);
            float *logitsData = (float*) logits.cpuData;
            int total = logits.Count(0);
            for (int j = 0; j < total; j++) {
                logitsData[j] = tanh(logitsData[j]) * final_logit_softcapping;
            }
        }

        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            retLogits->resize(size);
            memcpy((float*) retLogits->data(),
                   ((float*) logits.cpuData) + (logits.dims[1] - 1) * size,
                   (size_t) size * logits.unitSize);
        }

        TopK(logits, topk, 1);
        topk.ToDevice(DataDevice::CPU);
        return (int) (((float *) topk.cpuData)[0] + 1e-3);
    }

    void Gemma4Model::SplitFusedMoeWeightsIfNeeded(const std::string &layerPrefix) {
        if (num_experts <= 0) {
            return;
        }

        const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";
        if (this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end()) {
            return;
        }

        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        auto fusedGateupIt = this->weight.weight.find(fusedGateupName);
        if (fusedGateupIt == this->weight.weight.end()) {
            return;
        }

        auto fusedDownIt = this->weight.weight.find(fusedDownName);
        AssertInFastLLM(fusedDownIt != this->weight.weight.end(), "Gemma4 fused MoE weights are incomplete.");

        Data &fusedGateup = fusedGateupIt->second;
        Data &fusedDown = fusedDownIt->second;
        fusedGateup.ToDevice(DataDevice::CPU);
        fusedDown.ToDevice(DataDevice::CPU);
        AssertInFastLLM(fusedGateup.dims.size() == 3 && fusedDown.dims.size() == 3,
                        "Gemma4 fused expert weights should be 3D.");
        AssertInFastLLM(fusedGateup.dims[0] == num_experts && fusedDown.dims[0] == num_experts,
                        "Gemma4 fused expert count mismatch.");

        for (int expert = 0; expert < num_experts; expert++) {
            const std::string expertGateupName = layerPrefix + "experts." + std::to_string(expert) + ".gateup_proj.weight";
            const std::string expertDownName = layerPrefix + "experts." + std::to_string(expert) + ".down_proj.weight";
            SplitGemma4ExpertLinearWeight(this->weight.weight[expertGateupName], fusedGateup, expertGateupName, expert);
            SplitGemma4ExpertLinearWeight(this->weight.weight[expertDownName], fusedDown, expertDownName, expert);
        }

        this->weight.weight.erase(fusedGateupName);
        this->weight.weight.erase(fusedDownName);
    }

    void Gemma4Model::OnWeightLoaded(const std::string &weightName, const std::set<std::string> &finishedWeightNames) {
        if (num_experts <= 0) {
            return;
        }

        std::string layerPrefix;
        if (!TryGetGemma4FusedMoeLayerPrefix(weightName, layerPrefix) ||
            (!StartWith(layerPrefix, "model.language_model.layers.") &&
             !StartWith(layerPrefix, "model.layers."))) {
            return;
        }

        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        if (finishedWeightNames.find(fusedGateupName) == finishedWeightNames.end() ||
            finishedWeightNames.find(fusedDownName) == finishedWeightNames.end()) {
            return;
        }

        SplitFusedMoeWeightsIfNeeded(layerPrefix);
    }

    void Gemma4Model::PrepareMoeWeights() {
        if (moeWeightsPrepared || num_experts <= 0) {
            moeWeightsPrepared = true;
            return;
        }

        weights.clear();
        biass.clear();
        expertGateupWeights.clear();
        expertDownWeights.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);
        expertGateupWeights.resize(block_cnt);
        expertDownWeights.resize(block_cnt);

        std::string baseLayerPrefix = "model.language_model.layers.";
        if (weight.weight.find(baseLayerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            baseLayerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            const std::string layerPrefix = baseLayerPrefix + std::to_string(i) + ".";
            const std::string routerWeightName = layerPrefix + "router.proj.weight";
            const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
            const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";
            const bool hasMoeLayer =
                this->weight.weight.find(routerWeightName) != this->weight.weight.end() ||
                this->weight.weight.find(fusedGateupName) != this->weight.weight.end() ||
                this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end();
            if (!hasMoeLayer) {
                continue;
            }

            SplitFusedMoeWeightsIfNeeded(layerPrefix);
            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            expertGateupWeights[i].resize(num_experts, nullptr);
            expertDownWeights[i].resize(num_experts, nullptr);
            for (int expert = 0; expert < num_experts; expert++) {
                const std::string expertGateupName = layerPrefix + "experts." + std::to_string(expert) + ".gateup_proj.weight";
                const std::string expertDownName = layerPrefix + "experts." + std::to_string(expert) + ".down_proj.weight";
                auto gateupIt = this->weight.weight.find(expertGateupName);
                auto downIt = this->weight.weight.find(expertDownName);
                AssertInFastLLM(gateupIt != this->weight.weight.end() && downIt != this->weight.weight.end(),
                                "Gemma4 split MoE expert weights are incomplete.");
                expertGateupWeights[i][expert] = &gateupIt->second;
                expertDownWeights[i][expert] = &downIt->second;
                weights[i].push_back(expertGateupWeights[i][expert]);
                weights[i].push_back(expertDownWeights[i][expert]);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
            }
        }

        moeWeightsPrepared = true;
    }

    bool Gemma4Model::TryApplyMoeFeedForward(const std::string &layerPrefix, Data &hiddenStates, Data &denseOutput) {
        if (!enable_moe_block || num_experts <= 0 || num_experts_per_tok <= 0) {
            return false;
        }

        const std::string routerWeightName = layerPrefix + ".router.proj.weight";
        const std::string routerScaleName = layerPrefix + ".router.scale";
        const std::string perExpertScaleName = layerPrefix + ".router.per_expert_scale";
        const std::string postNorm1Name = layerPrefix + ".post_feedforward_layernorm_1.weight";
        const std::string preNorm2Name = layerPrefix + ".pre_feedforward_layernorm_2.weight";
        const std::string postNorm2Name = layerPrefix + ".post_feedforward_layernorm_2.weight";
        const std::string postNormName = layerPrefix + ".post_feedforward_layernorm.weight";
        if (this->weight.weight.find(routerWeightName) == this->weight.weight.end() ||
            this->weight.weight.find(routerScaleName) == this->weight.weight.end() ||
            this->weight.weight.find(perExpertScaleName) == this->weight.weight.end() ||
            this->weight.weight.find(postNorm1Name) == this->weight.weight.end() ||
            this->weight.weight.find(preNorm2Name) == this->weight.weight.end() ||
            this->weight.weight.find(postNorm2Name) == this->weight.weight.end()) {
            return false;
        }

        PrepareMoeWeights();
        const int layerId = ParseGemma4LayerIndex(layerPrefix);
        AssertInFastLLM(layerId >= 0 && layerId < (int) expertGateupWeights.size(),
                        "Gemma4 layer index is out of range.");

        Data denseBranch;
        RMSNorm(denseOutput, this->weight[postNorm1Name], rms_norm_eps, denseBranch);

        Data residualFlat;
        MakeGemma4FlatLastDimView(hiddenStates, residualFlat);

        Data routerInput;
        RMSNorm(residualFlat, this->weight[routerScaleName], rms_norm_eps, routerInput);
        Mul(routerInput, 1.0f / sqrtf((float) residualFlat.dims[1]), routerInput);
        Data routerLogits;
        Linear(routerInput, this->weight[routerWeightName], *GetEmptyData(), routerLogits);

        std::vector<int> expertIds;
        std::vector<float> expertWeights;
        const int topK = std::min(num_experts_per_tok, num_experts);
        BuildGemma4RouterTopKFromLogits(routerLogits,
                                        this->weight[perExpertScaleName],
                                        topK,
                                        expertIds,
                                        expertWeights);

        Data expertInputFlat;
        RMSNorm(residualFlat, this->weight[preNorm2Name], rms_norm_eps, expertInputFlat);
        const int tokenCount = expertInputFlat.dims[0];
        Data moeFlat;
        const bool useMergeMoe = layerId >= 0 &&
                                 layerId < (int) weights.size() &&
                                 !weights[layerId].empty() &&
                                 CanRunMergeMOE(expertInputFlat, biass[layerId]);
        if (useMergeMoe) {
            Data expertIndexData(DataType::INT32, {tokenCount, topK});
            expertIndexData.Allocate();
            memcpy(expertIndexData.cpuData, expertIds.data(), sizeof(int) * expertIds.size());

            Data expertScoreData(DataType::FLOAT32, {tokenCount, topK});
            expertScoreData.Allocate();
            memcpy(expertScoreData.cpuData, expertWeights.data(), sizeof(float) * expertWeights.size());

            Data w1, w2, w3, tempInput, tempOutput;
            Data moeInputTemp, moeOutputTemp;
            ApplyDeviceMap(this->moeDeviceMap, layerId + 1, block_cnt);
            MergeMOEBlock(
                &expertInputFlat, &expertIndexData, &expertScoreData,
                &weights[layerId], &biass[layerId],
                &w1, &w2, &w3, &tempInput, &tempOutput,
                1.0f, &moeFlat, layerId,
                expertInputFlat.dataType, this->moeAtype,
                &moeInputTemp, &moeOutputTemp,
                MoeGateGeglu
            );
            ApplyDeviceMap(this->deviceMap, layerId + 1, block_cnt);
        } else {
            moeFlat.CopyFrom(expertInputFlat);
            Mul(moeFlat, 0.0f, moeFlat);
            for (int token = 0; token < tokenCount; token++) {
                Data tokenInput;
                Data tokenOutput;
                MakeGemma4MatrixRowView(expertInputFlat, token, tokenInput);
                MakeGemma4MatrixRowView(moeFlat, token, tokenOutput);
                for (int k = 0; k < topK; k++) {
                    const int expert = expertIds[token * topK + k];
                    const float expertWeight = expertWeights[token * topK + k];
                    if (expertWeight == 0.0f) {
                        continue;
                    }
                    AssertInFastLLM(expert >= 0 &&
                                    expert < (int) expertGateupWeights[layerId].size() &&
                                    expert < (int) expertDownWeights[layerId].size() &&
                                    expertGateupWeights[layerId][expert] != nullptr &&
                                    expertDownWeights[layerId][expert] != nullptr,
                                    "Gemma4 expert weights are incomplete.");

                    Data gateupOut;
                    Data gatePart;
                    Data upPart;
                    Data expertOut;
                    Linear(tokenInput, *expertGateupWeights[layerId][expert], *GetEmptyData(), gateupOut);
                    Split(gateupOut, 1, 0, moe_intermediate_size, gatePart);
                    Split(gateupOut, 1, moe_intermediate_size, gateupOut.dims[1], upPart);
                    if (gatePart.dataType != DataType::FLOAT32) {
                        ToDataType(gatePart, DataType::FLOAT32);
                    }
                    if (upPart.dataType != DataType::FLOAT32) {
                        ToDataType(upPart, DataType::FLOAT32);
                    }
                    Gelu(gatePart, gatePart);
                    MulTo(gatePart, upPart);
                    Linear(gatePart, *expertDownWeights[layerId][expert], *GetEmptyData(), expertOut);
                    if (expertOut.dataType != tokenOutput.dataType) {
                        ToDataType(expertOut, tokenOutput.dataType);
                    }
                    AddTo(tokenOutput, expertOut, expertWeight);
                }
            }
        }

        moeFlat.Reshape(hiddenStates.dims);
        Data moeBranch;
        RMSNorm(moeFlat, this->weight[postNorm2Name], rms_norm_eps, moeBranch);
        if (moeBranch.dataType != denseBranch.dataType) {
            ToDataType(moeBranch, denseBranch.dataType);
        }
        AddTo(denseBranch, moeBranch);

        Data mergedBranch;
        RMSNorm(denseBranch, this->weight[postNormName], rms_norm_eps, mergedBranch);
        if (mergedBranch.dataType != hiddenStates.dataType) {
            ToDataType(mergedBranch, hiddenStates.dataType);
        }
        AddTo(hiddenStates, mergedBranch);
        return true;
    }

    int Gemma4Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Gemma4Model::ForwardMultimodal(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                                     const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                                                     const std::map <std::string, std::vector <Data*> > &multimodalInput,
                                                     const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                                                     std::vector <std::vector <float>*> *retLogits) {
        std::vector <int> ret;
        std::vector <float> *logits = nullptr;
        if (retLogits != nullptr && !retLogits->empty()) {
            logits = (*retLogits)[0];
        }

        if (pastKeyValues.size() > 0 && pastKeyValues[0].second.dims.size() > 0) {
            ret.push_back(Forward(inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, logits));
            return ret;
        }

        auto pixelIt = multimodalInput.find("pixel_values");
        auto imagePosIt = multimodalInput.find("image_position_ids");
        auto mmTypeIt = multimodalInput.find("mm_token_type_ids");
        AssertInFastLLM(pixelIt != multimodalInput.end() && !pixelIt->second.empty(),
                        "Gemma4 multimodal requires pixel_values.");
        AssertInFastLLM(imagePosIt != multimodalInput.end() && !imagePosIt->second.empty(),
                        "Gemma4 multimodal requires image_position_ids.");
        AssertInFastLLM(mmTypeIt != multimodalInput.end() && !mmTypeIt->second.empty(),
                        "Gemma4 multimodal requires mm_token_type_ids.");

        if (!prepared) {
            Prepare();
            prepared = true;
        }

        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }

        Data imageFeatures;
        std::vector<int> softTokenCounts;
        EncodeImages(*pixelIt->second[0], *imagePosIt->second[0], imageFeatures, softTokenCounts);

        Data hiddenStates;
        Embedding(inputIds, this->weight[embName], hiddenStates);
        Mul(hiddenStates, sqrt((float)embed_dim), hiddenStates);
        MergeImageFeaturesIntoText(inputIds, imageFeatures, hiddenStates);
        hiddenStates.ToDevice(this->weight[embName].dataDevice);
        ToDataType(hiddenStates, this->dataType);

        Data visionAwareMask;
        BuildVisionAwareTextMask(attentionMask, *mmTypeIt->second[0], visionAwareMask);
        visionAwareMask.ToDevice(hiddenStates.dataDevice);

        ret.push_back(ForwardTextFromHiddenStates(inputIds, hiddenStates, visionAwareMask, positionIds,
                                                  pastKeyValues, generationConfig, lastTokens, logits));
        return ret;
    }

    std::vector <int> Gemma4Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;

        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }

        if (!prepared) {
            Prepare();
            prepared = true;
        }

        Embedding(inputIds, this->weight[embName], hiddenStates);
        Mul(hiddenStates, sqrt((float)embed_dim), hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];

        std::string layerPrefix = "model.language_model.layers.";
        if (weight.weight.find(layerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            layerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            bool isFullAttn = (layer_types[i] == 1);
            int curHeadDim = isFullAttn ? global_head_dim : sliding_head_dim;
            int curKVHeads = (isFullAttn && attention_k_eq_v) ? global_num_key_value_heads : num_key_value_heads;
            int curRotaryDim = isFullAttn ? global_head_dim / 2 : sliding_head_dim;
            Data *curSinData = isFullAttn ? &globalSinData : &slidingSinData;
            Data *curCosData = isFullAttn ? &globalCosData : &slidingCosData;

            std::string pre = layerPrefix + std::to_string(i);
            std::string qWeightName = pre + ".self_attn.q_proj.weight";
            std::string kWeightName = pre + ".self_attn.k_proj.weight";
            std::string vWeightName = pre + ".self_attn.v_proj.weight";
            std::string oWeightName = pre + ".self_attn.o_proj.weight";
            std::string qNormName = pre + ".self_attn.q_norm.weight";
            std::string kNormName = pre + ".self_attn.k_norm.weight";

            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], *GetEmptyData(), q);
            Linear(attenInput, weight[kWeightName], *GetEmptyData(), k);

            bool useAltAttn = isFullAttn && attention_k_eq_v;
            if (useAltAttn) {
                v.CopyFrom(k);
            } else {
                Linear(attenInput, weight[vWeightName], *GetEmptyData(), v);
            }

            q.Reshape({bsz, seqlen, -1, curHeadDim});
            k.Reshape({bsz, seqlen, -1, curHeadDim});
            v.Reshape({bsz, seqlen, -1, curHeadDim});

            RMSNorm(q, this->weight[qNormName], rms_norm_eps, q);
            RMSNorm(k, this->weight[kNormName], rms_norm_eps, k);

            {
                v.ToDevice(DataDevice::CPU);
                if (v.dataType != DataType::FLOAT32) {
                    ToDataType(v, DataType::FLOAT32);
                }
                float *vp = (float*)v.cpuData;
                int lastDim = v.dims.back();
                uint64_t total = v.Count(0);
                int outer = (int)(total / lastDim);
                for (int o = 0; o < outer; o++) {
                    float *row = vp + (uint64_t)o * lastDim;
                    float sumSq = 0.0f;
                    for (int j = 0; j < lastDim; j++) {
                        sumSq += row[j] * row[j];
                    }
                    float scale = 1.0f / sqrtf(sumSq / lastDim + rms_norm_eps);
                    for (int j = 0; j < lastDim; j++) {
                        row[j] = row[j] * scale;
                    }
                }
            }
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(k.dataDevice);
                pastValue.ToDevice(k.dataDevice);
            }

            fastllm::LlamaRotatePosition2D(q, positionIds, *curSinData, *curCosData, curRotaryDim);
            fastllm::LlamaRotatePosition2D(k, positionIds, *curSinData, *curCosData, curRotaryDim);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            std::vector<int> qkvSize = {-1, seqlen, curHeadDim};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0f, 1);

            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});

            Linear(qkv, weight[oWeightName], *GetEmptyData(), attenInput);
            RMSNorm(attenInput, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            AddTo(hiddenStates, attenInput);

            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);

            Linear(attenInput, weight[pre + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
            Gelu(w1, w1);
            Linear(attenInput, weight[pre + ".mlp.up_proj.weight"], *GetEmptyData(), w3);

            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w1, DataType::FLOAT32);
                ToDataType(w3, DataType::FLOAT32);
            }
            MulTo(w1, w3);
            Linear(w1, weight[pre + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w2, DataType::FLOAT16);
            }

            if (!TryApplyMoeFeedForward(pre, hiddenStates, w2)) {
                RMSNorm(w2, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, w2);
                AddTo(hiddenStates, w2);
            }

            std::string layerScalarName = layerPrefix + std::to_string(i) + ".layer_scalar";
            if (weight.weight.find(layerScalarName) != weight.weight.end()) {
                Data &ls = weight[layerScalarName];
                ls.ToDevice(DataDevice::CPU);
                if (ls.dataType != DataType::FLOAT32) {
                    ToDataType(ls, DataType::FLOAT32);
                }
                float scalar = ((float*)ls.cpuData)[0];
                Mul(hiddenStates, scalar, hiddenStates);
            }

        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;

            std::string normName = "model.language_model.norm.weight";
            if (weight.weight.find(normName) == weight.weight.end()) {
                normName = "model.norm.weight";
            }
            RMSNorm(hiddenStates, this->weight[normName], rms_norm_eps, hiddenStates);

            std::string lmHeadName = "lm_head.weight";
            Linear(hiddenStates, weight[lmHeadName], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);

            if (final_logit_softcapping > 0.0f) {
                Mul(logits, 1.0f / final_logit_softcapping, logits);
                logits.ToDevice(DataDevice::CPU);
                float *logitsData = (float*)logits.cpuData;
                int total = logits.Count(0);
                for (int j = 0; j < total; j++) {
                    logitsData[j] = tanh(logitsData[j]) * final_logit_softcapping;
                }
            }

            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(),
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size,
                        size * logits.unitSize);
                }
            }

            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        }
        return lastRet;
    }

    std::vector <int> Gemma4Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        if (!prepared) {
            Prepare();
            prepared = true;
        }
        int seqLen = inputIds.dims[1];

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);
        Data allPositionIds;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }
        Embedding(inputIds, this->weight[embName], hiddenStates);
        Mul(hiddenStates, sqrt((float)embed_dim), hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];

        std::string layerPrefix = "model.language_model.layers.";
        if (weight.weight.find(layerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            layerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            bool isFullAttn = (layer_types[i] == 1);
            int curHeadDim = isFullAttn ? global_head_dim : sliding_head_dim;
            int curKVHeads = (isFullAttn && attention_k_eq_v) ? global_num_key_value_heads : num_key_value_heads;
            int curRotaryDim = isFullAttn ? global_head_dim / 2 : sliding_head_dim;
            Data *curSinData = isFullAttn ? &globalSinData : &slidingSinData;
            Data *curCosData = isFullAttn ? &globalCosData : &slidingCosData;

            std::string pre = layerPrefix + std::to_string(i);
            std::string qWeightName = pre + ".self_attn.q_proj.weight";
            std::string kWeightName = pre + ".self_attn.k_proj.weight";
            std::string vWeightName = pre + ".self_attn.v_proj.weight";
            std::string oWeightName = pre + ".self_attn.o_proj.weight";
            std::string qNormName = pre + ".self_attn.q_norm.weight";
            std::string kNormName = pre + ".self_attn.k_norm.weight";

            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], *GetEmptyData(), q);
            Linear(attenInput, weight[kWeightName], *GetEmptyData(), k);

            bool useAltAttn = isFullAttn && attention_k_eq_v;
            if (useAltAttn) {
                v.CopyFrom(k);
            } else {
                Linear(attenInput, weight[vWeightName], *GetEmptyData(), v);
            }

            q.Reshape({bsz, seqlen, -1, curHeadDim});
            k.Reshape({bsz, seqlen, -1, curHeadDim});
            v.Reshape({bsz, seqlen, -1, curHeadDim});

            RMSNorm(q, this->weight[qNormName], rms_norm_eps, q);
            RMSNorm(k, this->weight[kNormName], rms_norm_eps, k);

            {
                v.ToDevice(DataDevice::CPU);
                if (v.dataType != DataType::FLOAT32) {
                    ToDataType(v, DataType::FLOAT32);
                }
                float *vp = (float*)v.cpuData;
                int lastDim = v.dims.back();
                uint64_t total = v.Count(0);
                int outer = (int)(total / lastDim);
                for (int o = 0; o < outer; o++) {
                    float *row = vp + (uint64_t)o * lastDim;
                    float sumSq = 0.0f;
                    for (int j = 0; j < lastDim; j++) {
                        sumSq += row[j] * row[j];
                    }
                    float sc = 1.0f / sqrtf(sumSq / lastDim + rms_norm_eps);
                    for (int j = 0; j < lastDim; j++) {
                        row[j] = row[j] * sc;
                    }
                }
            }

            int cacheOuter = k.dims[2], cacheInner = k.dims[3];
            for (int b = 0; b < batch; b++) {
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(k.dataDevice);
                }

                int curLen = seqLens[b];
                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }

            fastllm::LlamaRotatePosition2D(q, allPositionIds, *curSinData, *curCosData, curRotaryDim);
            fastllm::LlamaRotatePosition2D(k, allPositionIds, *curSinData, *curCosData, curRotaryDim);

            int curEmbedDim = num_attention_heads * curHeadDim;
            Data attenOutput = Data(this->dataType);
            int total = 0;

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            std::vector<int> qkvSize = {-1, seqlen, curHeadDim};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            for (int b = 0; b < batch; b++) {
                Split(k, 1, total, total + seqLens[b], curKs[b]);
                Split(v, 1, total, total + seqLens[b], curVs[b]);
                Split(q, 1, total, total + seqLens[b], curQs[b]);
                total += seqLens[b];
            }

            for (int b = 0; b < batch; b++) {
                keys[b] = (pastKeyValues[b * block_cnt + i].first);
                values[b] = (pastKeyValues[b * block_cnt + i].second);
                pointersK[b] = (&curKs[b]);
                pointersV[b] = (&curVs[b]);
            }
            CatDirectBatch(keys, pointersK, 1);
            CatDirectBatch(values, pointersV, 1);

            attenOutput.ToDevice(curQs[0].dataDevice);
            attenOutput.Resize({1, total, curEmbedDim});
            attenOutput.Allocate();
            int curLen = 0;
            for (int b = 0; b < batch; b++) {
                auto &curQ = curQs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                curAttenOutput.FakeFrom(attenOutput, curLen * curEmbedDim * attenOutput.unitSize);
                curLen += seqLens[b];

                if (attentionMask[b] == nullptr) {
                    Attention(curQ, pastKey, pastValue, *GetEmptyData(), curAttenOutput, curQ.dims[0] / pastKey.dims[0], 1.0f, 1);
                } else {
                    Attention(curQ, pastKey, pastValue, *attentionMask[b], curAttenOutput, curQ.dims[0] / pastKey.dims[0], 1.0f, 1);
                }
                PermuteSelf(curAttenOutput, {1, 0, 2});
            }

            Linear(attenOutput, weight[oWeightName], *GetEmptyData(), attenLastOutput);

            RMSNorm(attenLastOutput, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);

            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);

            Linear(attenInput, weight[pre + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
            Gelu(w1, w1);
            Linear(attenInput, weight[pre + ".mlp.up_proj.weight"], *GetEmptyData(), w3);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w1, DataType::FLOAT32);
                ToDataType(w3, DataType::FLOAT32);
            }
            MulTo(w1, w3);
            Linear(w1, weight[pre + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w2, DataType::FLOAT16);
            }

            if (!TryApplyMoeFeedForward(pre, hiddenStates, w2)) {
                RMSNorm(w2, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, w2);
                AddTo(hiddenStates, w2);
            }

            std::string layerScalarName = layerPrefix + std::to_string(i) + ".layer_scalar";
            if (weight.weight.find(layerScalarName) != weight.weight.end()) {
                Data &ls = weight[layerScalarName];
                ls.ToDevice(DataDevice::CPU);
                if (ls.dataType != DataType::FLOAT32) {
                    ToDataType(ls, DataType::FLOAT32);
                }
                float scalar = ((float*)ls.cpuData)[0];
                Mul(hiddenStates, scalar, hiddenStates);
            }
        }

        std::vector <int> lastRet;
        {
            std::string normName = "model.language_model.norm.weight";
            if (weight.weight.find(normName) == weight.weight.end()) {
                normName = "model.norm.weight";
            }

            std::string lmHeadName = "lm_head.weight";

            RMSNorm(hiddenStates, this->weight[normName], rms_norm_eps, hiddenStates);

            Data logits;
            Linear(hiddenStates, weight[lmHeadName], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);

            if (final_logit_softcapping > 0.0f) {
                Mul(logits, 1.0f / final_logit_softcapping, logits);
                logits.ToDevice(DataDevice::CPU);
                float *logitsData = (float*)logits.cpuData;
                int totalElements = logits.Count(0);
                for (int j = 0; j < totalElements; j++) {
                    logitsData[j] = tanh(logitsData[j]) * final_logit_softcapping;
                }
            }

            if (generationConfigs[0].output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    int offset = 0;
                    for (int s = 0; s < b; s++) offset += seqLens[s];
                    offset += seqLens[b] - 1;
                    memcpy((float*)(*retLogits)[b]->data(),
                        ((float*)logits.cpuData) + offset * size,
                        size * logits.unitSize);
                }
            }

            Data topk;
            int vocabSize = logits.dims.back();
            std::vector<float> lastLogits(batch * vocabSize);
            logits.ToDevice(DataDevice::CPU);
            int pos = 0;
            for (int b = 0; b < batch; b++) {
                int offset = pos + seqLens[b] - 1;
                memcpy(lastLogits.data() + b * vocabSize,
                       ((float*)logits.cpuData) + offset * vocabSize,
                       vocabSize * sizeof(float));
                pos += seqLens[b];
            }
            Data lastLogitsData(DataType::FLOAT32, {batch, 1, vocabSize}, lastLogits);

            TopK(lastLogitsData, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (((float *) topk.cpuData)[b * 2] + 1e-3));
            }
        }
        return lastRet;
    }

    bool Gemma4Model::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void Gemma4Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }
                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string Gemma4Model::MakeInput(const std::string &history, int round, const std::string &input) {
        if (round == 0) {
            return "<bos><|turn>user\n" + input + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        }
        return history + "<|turn>user\n" + input + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    }

    std::string Gemma4Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        if (round == 0) {
            return "<bos><|turn>user\n" + input + "<turn|>\n<|turn>model\n" + output + "<turn|>\n";
        }
        return history + "<|turn>user\n" + input + "<turn|>\n<|turn>model\n" + output + "<turn|>\n";
    }

    void Gemma4Model::Prepare() {
        std::string lmHeadName = "lm_head.weight";
        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }
        bool lmHeadExists = this->weight.weight.find(lmHeadName) != this->weight.weight.end();
        if (!lmHeadExists || this->weight[lmHeadName].dims.size() == 0) {
            this->weight[lmHeadName].CopyFrom(this->weight[embName]);
            ToDataType(this->weight[lmHeadName], this->dataType);
        }

        if (enable_moe_block) {
            PrepareMoeWeights();
        }
        PrepareVision();
    }

    void Gemma4Model::WarmUp() {
        Prepare();

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = 0;
        for (int i = 0; i < block_cnt; i++) {
            elementsInKVCachePerToken +=
                pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
    }
}
