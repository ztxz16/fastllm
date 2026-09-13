//
// DeepSeek-V4.1 视觉编码器（参考官方 inference/vision.py / image_processor.py / model.py）。
//
//   * ViT：patch 线性嵌入 -> vision_n_layers 层（RMSNorm -> 带 bias 的 wqkv -> 2D RoPE -> 全双向注意力
//     -> wo -> 残差 -> RMSNorm -> SwiGLU MLP -> 残差）-> RMSNorm。
//     2D RoPE：head_dim 的前一半维度按 (h, w) 位置旋转，cos/sin = [h * f_0..f_{k-1}, w * f_0..f_{k-1}]，
//     与官方 apply_rotary（前后各一半维度配对）一致，因此可直接用 LlamaRotatePosition2D，
//     只需按 token 顺序预先计算 [n, ropeDim / 2] 的 cos/sin 表。
//   * Aligner：把 [nVitH, nVitW, dim] 的特征按 r x r 块（不足补零）展开成 [L, dim * r * r]（通道优先，
//     与 F.unfold 一致），经 w1 + GELU + w2 得到 LLM 维度的图像嵌入。
//   * 图像 span 在 input_ids 中全部是 image_token_id，布局为
//     [image_start] + ([IMAGE] * nLlmW + [image_newline]) * nLlmH + [image_end]，
//     IMAGE 位置按行主序填入 aligner 输出，其余三种位置使用学习到的分隔符嵌入。
//
// 所有视觉计算以 FLOAT32 激活进行（权重为 float16），既能在 CUDA 也能在 CPU 路径上运行。
//

#include "deepseekv41.h"

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace fastllm {
    namespace {
        // 真实 checkpoint 把视觉参数嵌在 config 的 vision_config 里（HF 命名，展平后为
        // "vision_config.num_hidden_layers" 之类）；测试用的迷你模型写的是扁平的 "vision_xxx"。
        // 两种都要认，优先 HF 命名。
        const std::string *VisionDictFind(const WeightMap &weight, const std::string &hfKey,
                                          const std::string &flatKey) {
            auto it = weight.dicts.find("vision_config." + hfKey);
            if (it != weight.dicts.end()) {
                return &it->second;
            }
            it = weight.dicts.find(flatKey);
            return it == weight.dicts.end() ? nullptr : &it->second;
        }

        int VisionDictInt(const WeightMap &weight, const std::string &hfKey,
                          const std::string &flatKey, int fallback) {
            const std::string *v = VisionDictFind(weight, hfKey, flatKey);
            return v == nullptr ? fallback : atoi(v->c_str());
        }

        float VisionDictFloat(const WeightMap &weight, const std::string &hfKey,
                              const std::string &flatKey, float fallback) {
            const std::string *v = VisionDictFind(weight, hfKey, flatKey);
            return v == nullptr ? fallback : (float)atof(v->c_str());
        }

        // 把张量搬到 CPU 并转成 FLOAT32
        void VisionToCpuFloat(const Data &input, Data &output) {
            Data cpu;
            cpu.CopyFrom(input);
            cpu.ToDevice(DataDevice::CPU);
            if (cpu.dataType == DataType::FLOAT32) {
                output.CopyFrom(cpu);
            } else {
                ToDataType(cpu, output, DataType::FLOAT32);
                output.ToDevice(DataDevice::CPU);
            }
        }

        void VisionDump(const Data &cpuFloat, const std::string &name) {
            const char *dir = std::getenv("FASTLLM_DSV41_DUMP_DIR");
            if (dir == nullptr || dir[0] == '\0' || cpuFloat.cpuData == nullptr) {
                return;
            }
            std::string path = std::string(dir) + "/" + name + ".bin";
            FILE *fo = fopen(path.c_str(), "wb");
            if (fo != nullptr) {
                fwrite(cpuFloat.cpuData, 1, cpuFloat.GetBytes(), fo);
                fclose(fo);
            }
        }

        std::vector<int> VisionReadInts(const Data &input) {
            Data cpu;
            cpu.CopyFrom(input);
            cpu.ToDevice(DataDevice::CPU);
            std::vector<int> ret;
            for (uint64_t i = 0; i < cpu.Count(0); i++) {
                if (cpu.dataType == DataType::INT32) {
                    ret.push_back(((const int32_t*)cpu.cpuData)[i]);
                } else if (cpu.dataType == DataType::FLOAT32) {
                    ret.push_back((int)(((const float*)cpu.cpuData)[i] + (((const float*)cpu.cpuData)[i] >= 0 ? 0.5f : -0.5f)));
                } else {
                    ErrorInFastLLM("DeepSeekV41 vision: unsupported integer tensor dtype.");
                }
            }
            return ret;
        }

        // 全双向注意力，q/k/v 为 [heads, n, headDim] FLOAT32；按 query 分块以限制 [heads, chunk, n] 的分数矩阵大小。
        // 注意 fastllm 的 Attention 在没有掩码且 attentionType == 0 时默认加因果掩码，
        // 这里与 gemma4 的视觉编码器一样传 attentionType = 2 表示不加任何掩码。
        const int kVisionAttentionType = 2;

        void VisionAttention(const Data &q, const Data &k, const Data &v, float scale, Data &output) {
            const int n = q.dims[1];
            const int chunk = 1024;
            Data emptyMask;
            if (n <= chunk) {
                Attention(q, k, v, emptyMask, output, 1, scale, kVisionAttentionType);
                return;
            }
            Data acc;
            for (int st = 0; st < n; st += chunk) {
                int end = std::min(n, st + chunk);
                Data qPart, oPart;
                Split(q, 1, st, end, qPart);
                Attention(qPart, k, v, emptyMask, oPart, 1, scale, kVisionAttentionType);
                if (st == 0) {
                    acc.CopyFrom(oPart);
                } else {
                    Data merged;
                    Cat(acc, oPart, 1, merged);
                    acc.CopyFrom(merged);
                }
            }
            output.CopyFrom(acc);
        }
    }

    void DeepSeekV41Model::InitVisionParams() {
        vision_n_layers = VisionDictInt(this->weight, "num_hidden_layers", "vision_n_layers", 0);
        vision_dim = VisionDictInt(this->weight, "hidden_size", "vision_dim", 1024);
        vision_n_heads = VisionDictInt(this->weight, "num_attention_heads", "vision_n_heads", 16);
        vision_inter_dim = VisionDictInt(this->weight, "intermediate_size", "vision_inter_dim", 2816);
        vision_patch_size = VisionDictInt(this->weight, "patch_size", "vision_patch_size", 14);
        vision_downsample_ratio = VisionDictInt(this->weight, "downsample_ratio", "vision_downsample_ratio", 3);
        vision_rope_theta = VisionDictFloat(this->weight, "rope_theta", "vision_rope_theta", 10000.0f);
        vision_norm_eps = 1e-6f;   // vision.py 的 RMSNorm 默认 eps
        if (!VisionEnabled()) {
            return;
        }
        AssertInFastLLM(vision_dim % vision_n_heads == 0 && (vision_dim / vision_n_heads) % 4 == 0,
                        "DeepSeekV41 vision: vision_dim / vision_n_heads must be a multiple of 4.");
        // 视觉塔权重保持 float16，不参与低比特量化
        this->cantQuantLinears.insert("vision.patch_embed.proj.weight");
        this->cantQuantLinears.insert("aligner.w1.weight");
        this->cantQuantLinears.insert("aligner.w2.weight");
        for (int i = 0; i < vision_n_layers; i++) {
            std::string pre = "vision.blocks." + std::to_string(i);
            this->cantQuantLinears.insert(pre + ".attn.wqkv.weight");
            this->cantQuantLinears.insert(pre + ".attn.wo.weight");
            this->cantQuantLinears.insert(pre + ".mlp.w1.weight");
            this->cantQuantLinears.insert(pre + ".mlp.w2.weight");
        }
    }

    bool DeepSeekV41Model::IsVisionTensor(const std::string &name) const {
        return name.compare(0, 7, "vision.") == 0 || name.compare(0, 8, "aligner.") == 0 ||
               name == "image_start" || name == "image_end" || name == "image_newline";
    }

    void DeepSeekV41Model::EncodeImage(const Data &patches, int nVitH, int nVitW, Data &output,
                                       const std::string &dumpPrefix) {
        AssertInFastLLM(VisionEnabled(), "DeepSeekV41: this checkpoint has no vision tower (vision_n_layers == 0).");
        const int n = nVitH * nVitW;
        const int dim = vision_dim;
        const int heads = vision_n_heads;
        const int headDim = dim / heads;
        const int ropeDim = headDim / 2;          // 旋转维度（前一半 h、后一半 w 各 ropeDim / 2 个频率）
        const int r = vision_downsample_ratio;
        AssertInFastLLM(patches.dims.size() == 2 && patches.dims[0] == n &&
                        patches.dims[1] == 3 * vision_patch_size * vision_patch_size,
                        "DeepSeekV41 vision: patches must be [nVitH * nVitW, 3 * patch * patch].");

        // 1. patch 嵌入
        Data patchInput;
        if (patches.dataType == DataType::FLOAT32) {
            patchInput.CopyFrom(patches);
        } else {
            ToDataType(patches, patchInput, DataType::FLOAT32);
        }
        Data x;
        Linear(patchInput, weight["vision.patch_embed.proj.weight"], weight["vision.patch_embed.proj.bias"], x);
        x.Reshape({1, n, dim});
        auto dumpStage = [&](const Data &data, const std::string &name) {
            if (!dumpPrefix.empty()) {
                Data cpu;
                VisionToCpuFloat(data, cpu);
                VisionDump(cpu, dumpPrefix + name);
            }
        };
        dumpStage(x, "_vit_patch");

        // 2. 2D RoPE 表：freqs[t] = [h(t) * f_0..f_{k-1}, w(t) * f_0..f_{k-1}]，k = ropeDim / 2
        Data posIds, sinData, cosData;
        {
            const int k = ropeDim / 2;
            std::vector<float> invFreq(k);
            for (int i = 0; i < k; i++) {
                invFreq[i] = 1.0f / std::pow(vision_rope_theta, (float)(2 * i) / (float)ropeDim);
            }
            std::vector<float> sinValues((uint64_t)n * ropeDim), cosValues((uint64_t)n * ropeDim), pos(n);
            for (int t = 0; t < n; t++) {
                int h = t / nVitW, w = t % nVitW;
                pos[t] = (float)t;
                for (int i = 0; i < k; i++) {
                    float ah = (float)h * invFreq[i], aw = (float)w * invFreq[i];
                    sinValues[(uint64_t)t * ropeDim + i] = std::sin(ah);
                    cosValues[(uint64_t)t * ropeDim + i] = std::cos(ah);
                    sinValues[(uint64_t)t * ropeDim + k + i] = std::sin(aw);
                    cosValues[(uint64_t)t * ropeDim + k + i] = std::cos(aw);
                }
            }
            posIds.CopyFrom(Data(DataType::FLOAT32, {1, n}, pos));
            sinData.CopyFrom(Data(DataType::FLOAT32, {n, ropeDim}, sinValues));
            cosData.CopyFrom(Data(DataType::FLOAT32, {n, ropeDim}, cosValues));
        }

        // 3. transformer blocks
        const float scale = 1.0f / std::sqrt((float)headDim);
        Data h, qkv, q, k, v, attnOut, proj, mlpHidden, mlpAct, mlpOut;
        for (int layer = 0; layer < vision_n_layers; layer++) {
            std::string pre = "vision.blocks." + std::to_string(layer);
            RMSNorm(x, weight[pre + ".norm1.weight"], vision_norm_eps, h);
            Linear(h, weight[pre + ".attn.wqkv.weight"], weight[pre + ".attn.wqkv.bias"], qkv);
            Split(qkv, 2, 0, dim, q);
            Split(qkv, 2, dim, 2 * dim, k);
            Split(qkv, 2, 2 * dim, 3 * dim, v);
            q.Reshape({1, n, heads, headDim});
            k.Reshape({1, n, heads, headDim});
            LlamaRotatePosition2D(q, posIds, sinData, cosData, headDim);
            LlamaRotatePosition2D(k, posIds, sinData, cosData, headDim);
            q.Reshape({n, heads, headDim});
            k.Reshape({n, heads, headDim});
            v.Reshape({n, heads, headDim});
            PermuteSelf(q, {1, 0, 2});
            PermuteSelf(k, {1, 0, 2});
            PermuteSelf(v, {1, 0, 2});
            VisionAttention(q, k, v, scale, attnOut);
            PermuteSelf(attnOut, {1, 0, 2});
            attnOut.Reshape({1, n, dim});
            Linear(attnOut, weight[pre + ".attn.wo.weight"], weight[pre + ".attn.wo.bias"], proj);
            AddTo(x, proj);

            RMSNorm(x, weight[pre + ".norm2.weight"], vision_norm_eps, h);
            Linear(h, weight[pre + ".mlp.w1.weight"], Data(), mlpHidden);
            Swiglu(mlpHidden, mlpAct);
            Linear(mlpAct, weight[pre + ".mlp.w2.weight"], Data(), mlpOut);
            AddTo(x, mlpOut);
            dumpStage(x, "_vit_block" + std::to_string(layer));
        }
        Data vitOut, vitCpu;
        RMSNorm(x, weight["vision.norm.weight"], vision_norm_eps, vitOut);
        VisionToCpuFloat(vitOut, vitCpu);
        dumpStage(vitCpu, "_vit_out");

        // 4. aligner：r x r 块展开（通道优先，越界补零）-> w1 -> GELU -> w2
        const int nLlmH = (nVitH + r - 1) / r;
        const int nLlmW = (nVitW + r - 1) / r;
        const int blocks = nLlmH * nLlmW;
        const int unfoldDim = dim * r * r;
        Data unfolded(DataType::FLOAT32, {blocks, unfoldDim});
        unfolded.Allocate();
        {
            const float *src = (const float*)vitCpu.cpuData;
            float *dst = (float*)unfolded.cpuData;
            for (int bi = 0; bi < nLlmH; bi++) {
                for (int bj = 0; bj < nLlmW; bj++) {
                    float *row = dst + (uint64_t)(bi * nLlmW + bj) * unfoldDim;
                    for (int ki = 0; ki < r; ki++) {
                        for (int kj = 0; kj < r; kj++) {
                            int hh = bi * r + ki, ww = bj * r + kj;
                            int idx = ki * r + kj;
                            if (hh < nVitH && ww < nVitW) {
                                const float *feat = src + (uint64_t)(hh * nVitW + ww) * dim;
                                for (int c = 0; c < dim; c++) {
                                    row[c * r * r + idx] = feat[c];
                                }
                            } else {
                                for (int c = 0; c < dim; c++) {
                                    row[c * r * r + idx] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
        Data a1, a2;
        Linear(unfolded, weight["aligner.w1.weight"], weight["aligner.w1.bias"], a1);
        Gelu(a1, a1);
        Linear(a1, weight["aligner.w2.weight"], weight["aligner.w2.bias"], a2);
        VisionToCpuFloat(a2, output);
        output.Reshape({blocks, embed_dim});
    }

    void DeepSeekV41Model::EncodeImageSpans(const std::map <std::string, std::vector <Data*> > &multimodalInput,
                                            DeepSeekV41RequestState &state) {
        AssertInFastLLM(VisionEnabled(), "DeepSeekV41: this checkpoint has no vision tower (vision_n_layers == 0).");
        const int dim = embed_dim;
        const int r = vision_downsample_ratio;
        state.imageSpans.clear();
        state.imagesEncoded = true;
        auto pixelIt = multimodalInput.find("pixel_values");
        auto gridIt = multimodalInput.find("image_grid");
        if (pixelIt == multimodalInput.end() || gridIt == multimodalInput.end() || gridIt->second.empty() ||
            gridIt->second[0] == nullptr) {
            return;   // 没有图像数据（例如其它模型格式的 payload），按纯文本处理
        }
        std::vector<int> grid = VisionReadInts(*gridIt->second[0]);
        const int numImages = (int)grid.size() / 3;
        AssertInFastLLM(numImages * 3 == (int)grid.size() && (int)pixelIt->second.size() == numImages,
                        "DeepSeekV41 multimodal: image_grid must be [numImages, 3] and match pixel_values.");

        Data startEmb, endEmb, newlineEmb;
        VisionToCpuFloat(weight["image_start"], startEmb);
        VisionToCpuFloat(weight["image_end"], endEmb);
        VisionToCpuFloat(weight["image_newline"], newlineEmb);
        AssertInFastLLM((int)startEmb.Count(0) == dim && (int)endEmb.Count(0) == dim && (int)newlineEmb.Count(0) == dim,
                        "DeepSeekV41 multimodal: image_start / image_end / image_newline must be [dim].");

        const bool dump = std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr;
        for (int i = 0; i < numImages; i++) {
            const int start = grid[i * 3], nVitH = grid[i * 3 + 1], nVitW = grid[i * 3 + 2];
            const int nLlmH = (nVitH + r - 1) / r, nLlmW = (nVitW + r - 1) / r;
            const int span = nLlmH * (nLlmW + 1) + 2;
            AssertInFastLLM(start >= 0 && nVitH > 0 && nVitW > 0,
                            "DeepSeekV41 multimodal: invalid image_grid entry for image " + std::to_string(i) + ".");
            Data feats;
            EncodeImage(*pixelIt->second[i], nVitH, nVitW, feats, dump ? "fl_image" + std::to_string(i) : "");
            if (dump) {
                VisionDump(feats, "fl_image" + std::to_string(i) + "_embeds");
            }
            DeepSeekV41ImageSpan spanData;
            spanData.start = start;
            spanData.length = span;
            spanData.embeds = Data(DataType::FLOAT32, {span, dim});
            spanData.embeds.Allocate();
            float *dst = (float*)spanData.embeds.cpuData;
            const float *src = (const float*)feats.cpuData;
            int pos = 0;
            memcpy(dst + (uint64_t)pos * dim, startEmb.cpuData, dim * sizeof(float));
            pos++;
            for (int row = 0; row < nLlmH; row++) {
                memcpy(dst + (uint64_t)pos * dim, src + (uint64_t)row * nLlmW * dim, (uint64_t)nLlmW * dim * sizeof(float));
                pos += nLlmW;
                memcpy(dst + (uint64_t)pos * dim, newlineEmb.cpuData, dim * sizeof(float));
                pos++;
            }
            memcpy(dst + (uint64_t)pos * dim, endEmb.cpuData, dim * sizeof(float));
            pos++;
            AssertInFastLLM(pos == span, "DeepSeekV41 multimodal: internal span layout error.");
            state.imageSpans.push_back(std::move(spanData));
        }
    }

    bool DeepSeekV41Model::PrepareImageEmbeds(const Data &inputIds, int startPos, DeepSeekV41RequestState &state,
                                              Data &embeds, std::vector<int> &imageMask) {
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "DeepSeekV41 multimodal: inputIds must be [1, seqlen].");
        const int seqlen = inputIds.dims[1];
        const int dim = embed_dim;
        const int endPos = startPos + seqlen;
        bool overlap = false;
        for (auto &span : state.imageSpans) {
            overlap = overlap || (span.start < endPos && span.start + span.length > startPos);
        }
        if (!overlap) {
            return false;
        }
        std::vector<int> tokens = VisionReadInts(inputIds);
        {
            Data embedOut;
            Embedding(inputIds, weight["embed.weight"], embedOut);
            VisionToCpuFloat(embedOut, embeds);
            embeds.Reshape({1, seqlen, dim});
        }
        imageMask.assign(seqlen, 0);
        float *dst = (float*)embeds.cpuData;
        for (size_t i = 0; i < state.imageSpans.size(); i++) {
            const auto &span = state.imageSpans[i];
            const int lo = std::max(span.start, startPos), hi = std::min(span.start + span.length, endPos);
            for (int p = lo; p < hi; p++) {
                AssertInFastLLM(tokens[p - startPos] == image_token_id,
                                "DeepSeekV41 multimodal: token at " + std::to_string(p) +
                                " is not image_token_id (image " + std::to_string(i) + ").");
                imageMask[p - startPos] = 1;
                memcpy(dst + (uint64_t)(p - startPos) * dim,
                       (const float*)span.embeds.cpuData + (uint64_t)(p - span.start) * dim, dim * sizeof(float));
            }
        }
        if (std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr) {
            VisionDump(embeds, startPos == 0 ? "fl_mm_embeds" : "fl_mm_embeds_p" + std::to_string(startPos));
        }
        return true;
    }

    std::vector<int> DeepSeekV41Model::ForwardMultimodal(const Data &inputIds, const Data &attentionMask,
                                                         const Data &positionIds,
                                                         std::vector<std::pair<Data, Data> > &pastKeyValues,
                                                         const std::map <std::string, std::vector <Data*> > &multimodalInput,
                                                         const GenerationConfig &generationConfig,
                                                         const LastTokensManager &lastTokens,
                                                         std::vector<std::vector<float>*> *retLogits) {
        int startPos = 0;
        if (positionIds.dims.size() >= 1 && positionIds.Count(0) > 0) {
            auto pids = VisionReadInts(positionIds);
            startPos = pids.empty() ? 0 : pids[0];
        }
        // 记录多模态输入（调度器不经过 OnResponseContextCreated 直接调用时也能工作），实际编码在 ForwardSingle
        {
            auto state = GetOrCreateState(pastKeyValues, false);
            if (startPos == 0 && state->totalLen > 0) {
                state = GetOrCreateState(pastKeyValues, true);
            }
            if (!multimodalInput.empty() && state->pendingMultimodal == nullptr) {
                state->pendingMultimodal = &multimodalInput;
            }
        }
        const int seqlen = inputIds.dims[1];
        int chunk = GetChunkedPrefillSize();
        if (startPos > 0 || chunk <= 0 || chunk >= seqlen) {
            return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues,
                                generationConfig, lastTokens, retLogits);
        }
        // 调度器把整个图文 prompt 交给本函数时，按 chunked prefill 大小自行分块
        std::vector<int> ret;
        for (int st = 0; st < seqlen; st += chunk) {
            int end = std::min(seqlen, st + chunk);
            Data curIds, curPos;
            Split(inputIds, 1, st, end, curIds);
            std::vector<float> posValues(end - st);
            for (int i = st; i < end; i++) {
                posValues[i - st] = (float)i;
            }
            curPos.CopyFrom(Data(DataType::FLOAT32, {1, end - st}, posValues));
            ret = ForwardSingle(curIds, curPos, pastKeyValues, generationConfig, lastTokens,
                                end == seqlen ? retLogits : nullptr, nullptr, nullptr);
        }
        return ret;
    }
}
