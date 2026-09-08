#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void Fill(Data &data, int seed, float scale) {
    data.Allocate();
    auto *values = reinterpret_cast<uint16_t *>(data.cpuData);
    std::mt19937 random(seed);
    for (uint64_t i = 0; i < data.Count(0); ++i) {
        values[i] = float_to_half(
            scale * (float(random() % 65537) / 32768.0f - 1.0f));
    }
}

class DraftModel : public Qwen3_5Model {
public:
    using Qwen3_5Model::MtpKvCache;
    using Qwen3_5Model::RunMtpGreedyDraft;
    using Qwen3_5Model::RunMtpGreedyDraftBatch;
    static constexpr int width = 96;
    bool moe;

    explicit DraftModel(int headDim, bool useMoe = false) : moe(useMoe) {
        embed_dim = width;
        num_attention_heads = 12;
        num_key_value_heads = 2;
        head_dim = headDim;
        rotary_dim = 64;
        mtp_num_hidden_layers = 1;
        num_experts = moe ? 2 : 0;
        num_experts_per_tok = 1;
        n_shared_experts = 0;
        dataType = FLOAT16;
        rms_norm_eps = 1e-6f;
        int seed = 0;
        auto add = [&](const std::string &name, std::vector<int> dims, bool norm = false) {
            weight.AddEmptyWeight(name, dims, FLOAT16);
            Data &data = weight[name];
            Fill(data, ++seed, 0.04f);
            if (norm) {
                std::fill_n(reinterpret_cast<uint16_t *>(data.cpuData),
                            data.Count(0), float_to_half(1.0f));
            }
        };
        add(language_prefix + "embed_tokens.weight", {64, width});
        add("lm_head.weight", {64, width});
        add("mtp.fc.weight", {width, 2 * width});
        for (const char *name : {"mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
                                 "mtp.pre_fc_norm_hidden.weight"}) add(name, {width}, true);
        const std::string prefix = "mtp.layers.0.";
        add(prefix + "input_layernorm.weight", {width}, true);
        add(prefix + "post_attention_layernorm.weight", {width}, true);
        add(prefix + "self_attn.q_norm.weight", {headDim}, true);
        add(prefix + "self_attn.k_norm.weight", {headDim}, true);
        add(prefix + "self_attn.mergeqkv.weight", {28 * headDim, width});
        add(prefix + "self_attn.o_proj.weight", {width, 12 * headDim});
        if (moe) {
            add(prefix + "mlp.gate.weight", {2, width});
            for (int expert = 0; expert < 2; ++expert) {
                std::string name = prefix + "mlp.experts." + std::to_string(expert) + ".";
                add(name + "gateup_proj.weight", {256, width});
                add(name + "down_proj.weight", {width, 128});
            }
        } else {
            add(prefix + "mlp.gateup_proj.weight", {256, width});
            add(prefix + "mlp.down_proj.weight", {width, 128});
        }
    }
};

void InitCache(DraftModel::MtpKvCache &cache, int tokens, int headDim) {
    if (tokens == 0) return;
    for (Data *data : {&cache.key, &cache.value}) {
        data->dataType = FLOAT16;
        data->UpdateUnitSize();
        // Cover both spare capacity and expansion across a page boundary.
        data->Expansion({2, ((tokens + 127) / 128) * 128, headDim});
        data->Resize({2, tokens, headDim});
        auto *values = reinterpret_cast<uint16_t *>(data->cpuData);
        std::fill_n(values, data->expansionBytes / sizeof(uint16_t), float_to_half(0.0f));
        for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < tokens; ++t) {
                for (int d = 0; d < headDim; ++d) {
                    values[h * data->strides[0] + t * headDim + d] = float_to_half(
                        float((t * 7 + d * 13 + h * 19) % 97 - 48) / 64.0f);
                }
            }
        }
        data->ToDevice(DataDevice::CUDA, {0}, true);
    }
    cache.tokens = tokens;
}

std::vector<uint16_t> LogicalHalfData(const Data &data) {
    if (data.dims.empty()) return {};
    Require(data.dataType == FLOAT16, "expected FP16 data");
    std::vector<uint16_t> values(data.dims[0] * data.dims[1] * data.dims[2]);
    for (int h = 0; h < data.dims[0]; ++h) {
        size_t count = (size_t)data.dims[1] * data.dims[2];
        FastllmCudaCopyFromDeviceToHost(values.data() + h * count,
            (uint16_t *)data.cudaData + h * data.strides[0], count * sizeof(uint16_t));
    }
    return values;
}

void RunCase(DraftModel &model, int headDim, int context, int length, int sampleRow) {
    DraftModel::MtpKvCache single, cacheOnly, batched, other;
    InitCache(single, context, headDim);
    InitCache(cacheOnly, context, headDim);
    InitCache(batched, context, headDim);
    InitCache(other, 7, headDim);
    Data hidden(FLOAT16, {1, length, DraftModel::width});
    Fill(hidden, 9, 0.6f);
    hidden.ToDevice(DataDevice::CUDA, {0}, true);
    Data otherHidden(FLOAT16, {1, 1, DraftModel::width});
    Fill(otherHidden, 13, 0.6f);
    otherHidden.ToDevice(DataDevice::CUDA, {0}, true);
    std::vector<int> tokens(length);
    std::vector<float> positions(length);
    for (int i = 0; i < length; ++i) {
        tokens[i] = 3 + i;
        positions[i] = context + i;
    }
    Data positionIds(FLOAT32, {1, length}, positions);
    Data otherPosition(FLOAT32, {1, 1}, {7.0f});
    Data sampled;
    int token = model.RunMtpGreedyDraft(0, {0}, single, hidden, tokens,
                                        positionIds, sampleRow, &sampled);
    Require(model.RunMtpGreedyDraft(0, {0}, cacheOnly, hidden, tokens,
                positionIds, sampleRow, nullptr, true) == -1, "cache-only sampled a token");
    Require(single.tokens == context + length && cacheOnly.tokens == single.tokens,
            "sample-row selection truncated the KV append");
    Require(single.key.dims[1] == single.tokens && single.value.dims[1] == single.tokens,
            "KV metadata length differs from appended tokens");
    Require(LogicalHalfData(single.key) == LogicalHalfData(cacheOnly.key) &&
            LogicalHalfData(single.value) == LogicalHalfData(cacheOnly.value),
            "generation changed K/V compared with the full cache-only append");

    // The mixed-length batch keeps the existing full-query attention/MLP
    // path. It supplies an independent reference for the selected output;
    // a batch of one would delegate to the optimized single-request path.
    std::vector<Data> reference;
    auto batchTokens = model.RunMtpGreedyDraftBatch(0, {0}, {&batched, &other},
        {&hidden, &otherHidden}, {tokens, {11}}, {&positionIds, &otherPosition},
        {sampleRow, 0}, &reference);
    auto actual = LogicalHalfData(sampled);
    auto expected = LogicalHalfData(reference[0]);
    Require(actual.size() == DraftModel::width && actual.size() == expected.size(),
            "sampled hidden state has an incorrect shape");
    float maxError = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        float a = half_to_float(actual[i]), b = half_to_float(expected[i]);
        maxError = std::max(maxError, std::fabs(a - b));
        Require(std::isfinite(a) && std::isfinite(b) &&
                    std::fabs(a - b) <= 0.005f + 0.005f * std::fabs(b),
                "selected hidden state differs from full-query reference");
    }
    if (token != batchTokens[0]) {
        Require(headDim == 256 && context + length > 4096 && sampleRow == length - 1,
                "draft token changed outside the split-attention path");
        Data &head = model.weight["lm_head.weight"];
        Require(token >= 0 && token < head.dims[0], "draft token is outside the vocabulary");
        std::vector<uint16_t> weights(head.Count(0));
        FastllmCudaCopyFromDeviceToHost(weights.data(), head.cudaData, weights.size() * sizeof(uint16_t));
        std::vector<double> aScores(head.dims[0]), bScores(head.dims[0]);
        double maxLogitDelta = 0.0;
        for (int row = 0; row < head.dims[0]; ++row) {
            for (int col = 0; col < DraftModel::width; ++col) {
                double w = half_to_float(weights[row * DraftModel::width + col]);
                aScores[row] += w * half_to_float(actual[col]);
                bScores[row] += w * half_to_float(expected[col]);
            }
            maxLogitDelta = std::max(maxLogitDelta, std::abs(aScores[row] - bScores[row]));
        }
        auto rounded = [](double score) { return float_to_half(float(score)); };
        // The full-query reference still uses the old FP16 PV reduction.
        // Retain the hidden-state tolerance, check the new argmax, and only
        // allow a reference ranking change within the measured logit error:
        // two candidate scores can move apart by at most 2 * maxLogitDelta.
        double referenceGap = *std::max_element(bScores.begin(), bScores.end()) - bScores[token];
        Require(referenceGap <= 2 * maxLogitDelta &&
                rounded(aScores[token]) == rounded(*std::max_element(aScores.begin(), aScores.end())),
                "draft token differs beyond the measured reference-logit error");
        std::cout << "near-equal reference logits: token=" << token
                  << " reference=" << batchTokens[0]
                  << " margin=" << referenceGap << " max_logit_delta=" << maxLogitDelta << '\n';
    }
    std::cout << "moe=" << model.moe << " head_dim=" << headDim
              << " context=" << context << " query=" << length
              << " sample_row=" << sampleRow << " max_abs=" << maxError << " PASS\n";
}

void RunCausalCase() {
    DraftModel model(256);
    // Uniform scores and large new V rows make a future row visible
    // in the selected hidden state, even with thousands of cached rows.
    Data &qNorm = model.weight["mtp.layers.0.self_attn.q_norm.weight"];
    std::fill_n(reinterpret_cast<uint16_t *>(qNorm.cpuData),
                qNorm.Count(0), float_to_half(0.0f));
    Data &qkv = model.weight["mtp.layers.0.self_attn.mergeqkv.weight"];
    auto *qkvValues = reinterpret_cast<uint16_t *>(qkv.cpuData);
    for (uint64_t i = 26 * 256 * DraftModel::width; i < qkv.Count(0); ++i) {
        qkvValues[i] = float_to_half(half_to_float(qkvValues[i]) * 1024.0f);
    }
    RunCase(model, 256, 4097, 2, 0);

    DraftModel::MtpKvCache first, changedFuture;
    InitCache(first, 4097, 256);
    InitCache(changedFuture, 4097, 256);
    Data hidden(FLOAT16, {1, 2, DraftModel::width});
    Data otherHidden(FLOAT16, {1, 2, DraftModel::width});
    Fill(hidden, 9, 0.6f);
    Fill(otherHidden, 9, 0.6f);
    auto *values = reinterpret_cast<uint16_t *>(otherHidden.cpuData);
    for (int i = DraftModel::width; i < 2 * DraftModel::width; ++i) {
        values[i] = float_to_half(-half_to_float(values[i]));
    }
    hidden.ToDevice(DataDevice::CUDA, {0}, true);
    otherHidden.ToDevice(DataDevice::CUDA, {0}, true);
    Data positions(FLOAT32, {1, 2}, {4097.0f, 4098.0f}), sampled, otherSampled;
    int token = model.RunMtpGreedyDraft(0, {0}, first, hidden, {3, 4}, positions, 0, &sampled);
    int otherToken = model.RunMtpGreedyDraft(0, {0}, changedFuture, otherHidden,
                                           {3, 5}, positions, 0, &otherSampled);
    // Both calls have identical shapes and the same visible prefix. Changing
    // only a future row must not affect any bit of the selected first row.
    Require(LogicalHalfData(sampled) == LogicalHalfData(otherSampled) && token == otherToken,
            "future MTP row changed a causally earlier output");
    Require(LogicalHalfData(first.value) != LogicalHalfData(changedFuture.value),
            "causal test did not change the future V row");
    std::cout << "non-last MTP output is independent of future rows PASS\n";
}

void RunRollbackCase(DraftModel &model, int headDim, int context,
                     int batch, int accepted) {
    std::vector<DraftModel::MtpKvCache> caches(batch), reference(batch);
    for (int b = 0; b < batch; ++b) {
        // The second request has a different capacity boundary.
        InitCache(caches[b], context + b * 3, headDim);
        InitCache(reference[b], context + b * 3, headDim);
    }
    auto append = [&](std::vector<DraftModel::MtpKvCache> &kv, int step) {
        std::vector<Data> hidden(batch), positions(batch), sampled;
        std::vector<const Data*> hiddenPtrs;
        std::vector<Data*> positionPtrs;
        std::vector<DraftModel::MtpKvCache*> cachePtrs;
        std::vector<std::vector<int>> tokens(batch, {3 + step});
        for (int b = 0; b < batch; ++b) {
            hidden[b].dataType = FLOAT16;
            hidden[b].Resize({1, 1, DraftModel::width});
            Fill(hidden[b], 11 + step + b, 0.6f);
            hidden[b].ToDevice(DataDevice::CUDA, {0}, true);
            Data position(FLOAT32, {1, 1}, {float(kv[b].tokens)});
            positions[b].CopyFrom(position);
            hiddenPtrs.push_back(&hidden[b]);
            positionPtrs.push_back(&positions[b]);
            cachePtrs.push_back(&kv[b]);
        }
        // batch=1 delegates to the single-request production path.
        model.RunMtpGreedyDraftBatch(0, {0}, cachePtrs, hiddenPtrs, tokens,
                                     positionPtrs, std::vector<int>(batch, 0), &sampled);
        std::vector<std::vector<uint16_t>> result;
        for (const Data &data : sampled) result.push_back(LogicalHalfData(data));
        return result;
    };
    append(caches, 0);
    append(caches, 1);
    for (int step = 0; step < accepted; ++step) append(reference, step);

    std::vector<void*> keyPointers(batch), valuePointers(batch);
    for (int b = 0; b < batch; ++b) {
        auto keyCapacity = caches[b].key.expansionDims;
        auto valueCapacity = caches[b].value.expansionDims;
        auto keyStrides = caches[b].key.strides;
        auto valueStrides = caches[b].value.strides;
        auto keyBytes = caches[b].key.expansionBytes;
        auto valueBytes = caches[b].value.expansionBytes;
        keyPointers[b] = caches[b].key.cudaData;
        valuePointers[b] = caches[b].value.cudaData;
        caches[b].Truncate(context + b * 3 + accepted);
        Require(caches[b].tokens == reference[b].tokens &&
                    LogicalHalfData(caches[b].key) == LogicalHalfData(reference[b].key) &&
                    LogicalHalfData(caches[b].value) == LogicalHalfData(reference[b].value),
                "MTP rollback changed the committed K/V prefix");
        Require(caches[b].key.expansionDims == keyCapacity &&
                    caches[b].value.expansionDims == valueCapacity &&
                    caches[b].key.strides == keyStrides &&
                    caches[b].value.strides == valueStrides &&
                    caches[b].key.expansionBytes == keyBytes &&
                    caches[b].value.expansionBytes == valueBytes,
                "MTP rollback restored an obsolete buffer layout");
    }
    Require(append(caches, 2) == append(reference, 2),
            "MTP output after rollback differs from a cache without rejected drafts");
    for (int b = 0; b < batch; ++b) {
        Require(LogicalHalfData(caches[b].key) == LogicalHalfData(reference[b].key) &&
                    LogicalHalfData(caches[b].value) == LogicalHalfData(reference[b].value),
                "MTP append after rollback changed historical K/V");
        if (accepted < 2) {
            Require(caches[b].key.cudaData == keyPointers[b] &&
                        caches[b].value.cudaData == valuePointers[b],
                    "MTP reallocated capacity already reserved by rejected drafts");
        }
    }
    std::cout << "rollback head_dim=" << headDim << " context=" << context
              << " batch=" << batch << " accepted=" << accepted << " PASS\n";
}
}

int main(int argc, char **argv) {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    try {
        SetThreads(2);
        SetCudaEmbedding(false);
        FastllmCudaSetDevice(0);
        const std::string mode = argc == 2 ? argv[1] : "";
        const bool longContext = mode == "--long";
        const bool useMoe = mode == "--moe";
        if (mode == "--causal") {
            RunCausalCase();
            return 0;
        }
        if (mode == "--rollback" || mode == "--rollback-long") {
            const std::vector<int> contexts = mode == "--rollback-long"
                ? std::vector<int>{16383, 196735}
                : std::vector<int>{0, 127, 128, 4095};
            for (int headDim : {128, 256}) {
                DraftModel model(headDim);
                for (int context : contexts) {
                    for (int batch : {1, 2}) {
                        for (int accepted : {0, 1, 2}) {
                            RunRollbackCase(model, headDim, context, batch, accepted);
                        }
                    }
                }
            }
            return 0;
        }
        for (int headDim : {128, 256}) {
            DraftModel model(headDim, useMoe);
            // context=4094 covers appended KV lengths on both sides of 4096.
            for (int context : (longContext ? std::vector<int>{163840, 196608}
                                            : std::vector<int>{0, 127, 4094, 4097})) {
                for (int length : {1, 2, 3, 4, 8}) {
                    RunCase(model, headDim, context, length, length - 1);
                    if (length > 1) RunCase(model, headDim, context, length, 0);
                }
            }
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
