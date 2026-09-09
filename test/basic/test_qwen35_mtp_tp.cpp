#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FASTLLM_TEST_MTP_WIDTH
#define FASTLLM_TEST_MTP_WIDTH 96
#endif

using namespace fastllm;

static void SetEnv(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}


namespace {
    void Require(bool condition, const char *message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void Fill(Data &data, int seed, float scale) {
        data.Allocate();
        auto *values = reinterpret_cast<uint16_t *>(data.cpuData);
        std::mt19937 random(seed);
        for (uint64_t i = 0; i < data.Count(0); ++i) {
            values[i] = float_to_half(scale * (float(random() % 65537) / 32768.0f - 1.0f));
        }
    }

    class DraftModel : public Qwen3_5Model {
    public:
        using Qwen3_5Model::GetMtpPagedCachePool;
        using Qwen3_5Model::MtpKvCache;
        using Qwen3_5Model::mtpPagedCachePools;
        using Qwen3_5Model::mtpTpKvHeadScheme;
        using Qwen3_5Model::PrepareMtpTpWeights;
        using Qwen3_5Model::RestoreMtpPagedSnapshot;
        using Qwen3_5Model::RunMtpGreedyDraft;
        using Qwen3_5Model::RunMtpGreedyDraftBatch;
        using Qwen3_5Model::SnapshotMtpPagedCache;
        static constexpr int width = FASTLLM_TEST_MTP_WIDTH;
        bool moe;

        explicit DraftModel(int headDim, int kvHeads, bool useMoe = false, bool separate = false)
            : moe(useMoe) {
            maxBatch = 8;
            embed_dim = width;
            num_attention_heads = kvHeads * 3;
            num_key_value_heads = kvHeads;
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
                // Native RMSNorm reads norm weights as float, regardless of
                // the activation dtype. Match the real model's FP32 norms.
                weight.AddEmptyWeight(name, dims, norm ? FLOAT32 : FLOAT16);
                Data &data = weight[name];
                ++seed;
                if (norm) {
                    data.Allocate();
                    std::fill_n(reinterpret_cast<float *>(data.cpuData), data.Count(0), 1.0f);
                } else {
                    Fill(data, seed, 0.04f);
                }
            };
            add(language_prefix + "embed_tokens.weight", {64, width});
            add("lm_head.weight", {64, width});
            add("mtp.fc.weight", {width, 2 * width});
            for (const char *name :
                 {"mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"}) {
                add(name, {width}, true);
            }
            const std::string prefix = "mtp.layers.0.";
            add(prefix + "input_layernorm.weight", {width}, true);
            add(prefix + "post_attention_layernorm.weight", {width}, true);
            add(prefix + "self_attn.q_norm.weight", {headDim}, true);
            add(prefix + "self_attn.k_norm.weight", {headDim}, true);
            // Construct merged QKV first so separate and merged forms have exactly
            // the same deterministic weights and exercise independent split paths.
            add(prefix + "self_attn.mergeqkv.weight", {8 * kvHeads * headDim, width});
            if (separate) {
                Data &merged = weight[prefix + "self_attn.mergeqkv.weight"];
                int offset = 0;
                for (auto spec : std::vector<std::pair<std::string, int>>{
                         {"q", 6 * kvHeads * headDim}, {"k", kvHeads * headDim}, {"v", kvHeads * headDim}}) {
                    std::string name = prefix + "self_attn." + spec.first + "_proj.weight";
                    weight.AddEmptyWeight(name, {spec.second, width}, FLOAT16);
                    Data &dst = weight[name];
                    dst.Allocate();
                    std::memcpy(dst.cpuData, merged.cpuData + (size_t)offset * width * 2, dst.GetBytes());
                    offset += spec.second;
                }
                weight.weight.erase(prefix + "self_attn.mergeqkv.weight");
            }
            add(prefix + "self_attn.o_proj.weight", {width, 3 * kvHeads * headDim});
            if (moe) {
                add(prefix + "mlp.gate.weight", {2, width});
                add(prefix + "mlp.shared_expert.gateup_proj.weight", {512, width});
                add(prefix + "mlp.shared_expert.down_proj.weight", {width, 256});
                add(prefix + "mlp.shared_expert_gate.weight", {1, width});
                for (int expert = 0; expert < 2; ++expert) {
                    std::string name = prefix + "mlp.experts." + std::to_string(expert) + ".";
                    add(name + "gateup_proj.weight", {512, width});
                    add(name + "down_proj.weight", {width, 256});
                }
            } else {
                add(prefix + "mlp.gateup_proj.weight", {512, width});
                add(prefix + "mlp.down_proj.weight", {width, 256});
            }
        }
    };

    std::vector<float> ReadHidden(const Data &source) {
        Require(source.dataType == FLOAT16, "sampled hidden must be FP16");
        FastllmCudaSetDevice(source.dataDeviceIds[0]);
        std::vector<uint16_t> values(source.Count(0));
        FastllmCudaCopyFromDeviceToHost(values.data(), source.cudaData, values.size() * 2);
        std::vector<float> result;
        for (auto value : values) {
            result.push_back(half_to_float(value));
        }
        return result;
    }

    float Compare(const std::vector<float> &actual, const std::vector<float> &expected, const char *message) {
        Require(actual.size() == expected.size(), "size mismatch");
        float maxError = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            float error = std::abs(actual[i] - expected[i]);
            maxError = std::max(maxError, error);
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                error > 0.006f + 0.006f * std::abs(expected[i])) {
                std::cerr << message << " index=" << i << " actual=" << actual[i]
                          << " expected=" << expected[i] << '\n';
                throw std::runtime_error(message);
            }
        }
        return maxError;
    }

    std::vector<float> CpuValues(const Data &source) {
        Require(source.cpuData && source.dataType == FLOAT16, "expected compact FP16 snapshot");
        std::vector<float> result(source.Count(0));
        auto *data = reinterpret_cast<uint16_t *>(source.cpuData);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = half_to_float(data[i]);
        }
        return result;
    }

    void Init(DraftModel &model, DraftModel::MtpKvCache &cache, int heads, int dim, int length) {
        if (!length) {
            return;
        }
        Data k(FLOAT16, {heads, length, dim}), v(FLOAT16, k.dims);
        Fill(k, 71, .15f);
        Fill(v, 93, .15f);
        Require(model.RestoreMtpPagedSnapshot(cache, k, v, 0), "restore failed");
    }

    void CheckPages(DraftModel &model, DraftModel::MtpKvCache &cache, int expected) {
        Require(cache.tokens == expected, "parent cache length mismatch");
        for (auto &entry : cache.shards) {
            auto &leaf = *entry.second;
            Require(leaf.tokens == expected, "rank cache length mismatch");
            Require(leaf.key.pageIndex == leaf.value.pageIndex, "rank K/V page tables mismatch");
            Require(leaf.key.pageIndex.size() == (size_t)((expected + GetPageLen() - 1) / GetPageLen()),
                    "rank rollback page leak");
            auto &pool = *model.mtpPagedCachePools.at(entry.first);
            Require(leaf.key.pagedKVCacheData == &pool.key, "wrong rank pool");
        }
    }

    void RunCase(DraftModel &single, DraftModel &tp, int heads, int dim, int context, int length, int row,
                 int &cases) {
        DraftModel::MtpKvCache a, b, only, restored;
        Init(single, a, heads, dim, context);
        Init(tp, b, heads, dim, context);
        Init(tp, only, heads, dim, context);
        Data hidden(FLOAT16, {1, length, DraftModel::width});
        Fill(hidden, 123, .6f);
        hidden.ToDevice(DataDevice::CUDA, {0}, true);
        std::vector<int> tokens(length);
        std::vector<float> positions(length);
        for (int i = 0; i < length; ++i) {
            tokens[i] = i + 3;
            positions[i] = context + i;
        }
        Data pos(FLOAT32, {1, length}, positions), expected, actual;
        FastllmCudaSetDevice(0);
        int ref = single.RunMtpGreedyDraft(0, {0}, a, hidden, tokens, pos, row, &expected);
        int result = tp.RunMtpGreedyDraft(0, {0, 1}, b, hidden, tokens, pos, row, &actual);
        float error = Compare(ReadHidden(actual), ReadHidden(expected), "TP hidden mismatch");
        Require(result >= 0 && result < 64 && ref >= 0 && ref < 64, "invalid draft token");
        // Check the selected token against CPU logits for the returned hidden state.
        auto values = ReadHidden(actual);
        auto &head = single.weight["lm_head.weight"];
        FastllmCudaSetDevice(0);
        std::vector<uint16_t> ws(head.Count(0));
        FastllmCudaCopyFromDeviceToHost(ws.data(), head.cudaData, ws.size() * 2);
        std::vector<float> scores(64);
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < DraftModel::width; ++j) {
                scores[i] += values[j] * half_to_float(ws[i * DraftModel::width + j]);
            }
        }
        Require(*std::max_element(scores.begin(), scores.end()) - scores[result] < 0.002f,
                "TP greedy token is not a valid argmax");
        Require(tp.RunMtpGreedyDraft(0, {0, 1}, only, hidden, tokens, pos, row, nullptr, true) == -1,
                "cache-only produced a token");
        Data ak, av, bk, bv, ck, cv;
        Require(single.SnapshotMtpPagedCache(a, ak, av) && tp.SnapshotMtpPagedCache(b, bk, bv) &&
                    tp.SnapshotMtpPagedCache(only, ck, cv),
                "snapshot failed");
        Compare(CpuValues(bk), CpuValues(ak), "TP K mismatch");
        Compare(CpuValues(bv), CpuValues(av), "TP V mismatch");
        Require(CpuValues(bk) == CpuValues(ck) && CpuValues(bv) == CpuValues(cv), "cache-only changed TP KV");
        Require(tp.RestoreMtpPagedSnapshot(restored, bk, bv, 0), "TP prefix restore failed");
        CheckPages(tp, restored, context + length);
        Data rk, rv;
        Require(tp.SnapshotMtpPagedCache(restored, rk, rv), "restored snapshot failed");
        Require(CpuValues(rk) == CpuValues(bk) && CpuValues(rv) == CpuValues(bv),
                "prefix roundtrip changed KV");
        for (int accepted = length - 1; accepted >= 0; --accepted) {
            b.Truncate(context + accepted);
            CheckPages(tp, b, context + accepted);
        }
        b.Truncate(0);
        CheckPages(tp, b, 0);
        std::cout << "kv_heads=" << heads << " head_dim=" << dim << " context=" << context
                  << " query=" << length << " sample=" << row << " moe=" << tp.moe << " max_abs=" << error
                  << " PASS\n";
        ++cases;
    }

    void RunBatch(DraftModel &single, DraftModel &tp, int heads, int dim, int &cases) {
        std::vector<DraftModel::MtpKvCache> a(2), b(2);
        std::vector<Data> hidden(2), pos(2);
        std::vector<std::vector<int>> tokens = {{3, 4, 5}, {11}};
        for (int i = 0; i < 2; ++i) {
            Init(single, a[i], heads, dim, 127 + i);
            Init(tp, b[i], heads, dim, 127 + i);
            hidden[i].dataType = FLOAT16;
            hidden[i].UpdateUnitSize();
            hidden[i].Resize({1, (int)tokens[i].size(), DraftModel::width});
            Fill(hidden[i], 312 + i, .6f);
            hidden[i].ToDevice(DataDevice::CUDA, {0}, true);
            std::vector<float> positions;
            for (int j = 0; j < (int)tokens[i].size(); ++j) {
                positions.push_back(127 + i + j);
            }
            pos[i].CopyFrom(Data(FLOAT32, {1, (int)positions.size()}, positions));
        }
        std::vector<Data> expected, actual;
        single.RunMtpGreedyDraftBatch(0, {0}, {&a[0], &a[1]}, {&hidden[0], &hidden[1]}, tokens,
                                      {&pos[0], &pos[1]}, {0, 0}, &expected);
        tp.RunMtpGreedyDraftBatch(0, {0, 1}, {&b[0], &b[1]}, {&hidden[0], &hidden[1]}, tokens,
                                  {&pos[0], &pos[1]}, {0, 0}, &actual);
        for (int i = 0; i < 2; ++i) {
            Compare(ReadHidden(actual[i]), ReadHidden(expected[i]), "ragged TP mismatch");
        }
        std::cout << "ragged kv_heads=" << heads << " dim=" << dim << " PASS\n";
        ++cases;
    }
} // namespace

int main(int argc, char **argv) {
    if (FastllmCudaGetDeviceCount() < 2) {
        return 77;
    }
    try {
        SetThreads(2);
        SetCudaEmbedding(false);
        FastllmCudaSetDevice(0);
        SetEnv("FASTLLM_QWEN35_ENABLE_MTP", "3");
        SetEnv("FASTLLM_QWEN35_MTP_TP", "1");
        bool longOnly = argc > 1 && std::string(argv[1]) == "--long";
        const bool smoke = argc > 1 && std::string(argv[1]) == "--smoke";
        SetEnv("FASTLLM_MTP_FP8_DRAFT_HEAD", "0");
        SetMaxTokens(longOnly ? 540672 : 16384);
        int cases = 0;
        for (int dim : {128, 256}) {
            for (int heads : {1, 3, 5}) {
                if (smoke && (dim != 256 || heads != 3)) {
                    continue;
                }
                for (bool moe : {false, true}) {
                    if (longOnly && (heads != 3 || moe)) {
                        continue;
                    }
                    for (bool separate : {false, true}) {
                        if ((longOnly || smoke) && separate) {
                            continue;
                        }
                        DraftModel single(dim, heads, moe, separate), tp(dim, heads, moe, separate);
                        // For one KV head this gives rank 0 an empty attention
                        // shard; it must still own metadata and join collectives.
                        tp.deviceMap = {{"cuda:0", 1}, {"cuda:1", heads == 1 ? 3 : 1}};
                        tp.PrepareMtpTpWeights({0, 1});
                        long long bytes = tp.GetAutoWarmupCudaAdditionalCacheBytesPerToken(0) +
                                          tp.GetAutoWarmupCudaAdditionalCacheBytesPerToken(1);
                        Require(bytes == heads * dim * 4LL, "per-rank MTP KV accounting mismatch");
                        for (int context :
                             (longOnly ? std::vector<int>{131072}
                                       : (smoke ? std::vector<int>{0} : std::vector<int>{0, 127, 4097}))) {
                            for (int length : {1, 4}) {
                                RunCase(single, tp, heads, dim, context, length, length - 1, cases);
                                if (length > 1) {
                                    RunCase(single, tp, heads, dim, context, length, 0, cases);
                                }
                            }
                        }
                        if (!longOnly) {
                            RunBatch(single, tp, heads, dim, cases);
                        }
                        for (auto &entry : tp.mtpPagedCachePools) {
                            Require(entry.second->key.FreePageCount() == entry.second->key.maxPages &&
                                        entry.second->value.FreePageCount() == entry.second->value.maxPages,
                                    "request destruction leaked TP pages");
                        }
                    }
                }
            }
        }
        std::cout << "TOTAL " << cases << " PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
