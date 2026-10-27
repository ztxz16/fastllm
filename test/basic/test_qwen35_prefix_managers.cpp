#include "models/qwen3_5.h"

#include <iostream>
#include <stdexcept>

using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Real manager registry with CPU storage; device IDs identify TP ranks only.
// A CUDA build still calls the model's existing CUDA cleanup on destruction.
class PrefixFixture : public Qwen3_5Model {
public:
    PrefixFixture() {
        block_cnt = 2;
        num_key_value_heads = 4;
        threadTpPagedCacheBase = 1000;
        threadTpPreparedDevices = {3, 1, 7, 5, 2};
        threadTpAttentionKVHeadSchemes.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer) {
            weight[language_prefix + "layers." + std::to_string(layer) +
                   ".self_attn.o_proj.weight"];
            int head = 0;
            for (int rank = 0; rank < 5; ++rank) {
                auto &ranges = threadTpAttentionKVHeadSchemes[layer][
                    threadTpPreparedDevices[rank]];
                if (rank != (layer == 0 ? 0 : 4)) {
                    ranges.push_back({head, head + 1});
                    ++head;
                }
            }
        }
    }

    void Allocate(int missingLayer = -1, int missingRank = -1,
                  int missingKeyFlag = -1) {
        ClearAllPagedCacheManagers();
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int rank = 0; rank < 5; ++rank) {
                int device = threadTpPreparedDevices[rank];
                if (threadTpAttentionKVHeadSchemes[layer][device].empty()) continue;
                for (int keyFlag = 0; keyFlag < 2; ++keyFlag) {
                    if (layer == missingLayer && rank == missingRank &&
                        keyFlag == missingKeyFlag) continue;
                    Data shape(FLOAT32, {1, 0, 8});
                    AllocatePagedCacheManager(
                        (threadTpPagedCacheBase + rank * block_cnt + layer) * 2 + keyFlag,
                        PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
                        shape, 16, 4);
                }
            }
        }
    }

    void Check(bool complete) {
        ResponseContext context;
        context.currentTokens = {1, 2, 3};
        context.intParams["sentinel"] = 17;
        Require(QueryPagedPrefixCacheExtra(&context, 32) == (complete ? 32 : 0),
                "incomplete TP state was accepted, or valid zero-head ranks were rejected");
        Require(context.cacheLen == 0 && context.currentTokens == std::vector<int>({1, 2, 3}) &&
                context.pastKeyValues.empty() && context.intParams.size() == 1 &&
                context.intParams.at("sentinel") == 17,
                "prefix completeness query mutated request state");
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (bool key : {true, false}) {
                for (const auto &ref : GetPagedKVCacheManagers(layer, key)) {
                    Require(ref.second->FreePageCount() == 4,
                            "prefix completeness query acquired cache pages");
                }
            }
        }
    }

    void CheckMtpSnapshot() {
#ifdef USE_CUDA
        mtpTpPrepared = true;
        mtpTpDevices = threadTpPreparedDevices;
        mtpTpKvHeadScheme = threadTpAttentionKVHeadSchemes[0];
        head_dim = 8;
        MtpKvCache cache;
        cache.tokens = 3;
        for (Data *parent : {&cache.key, &cache.value}) {
            parent->dataType = FLOAT32;
            parent->Resize({num_key_value_heads, cache.tokens, head_dim});
        }
        for (const auto &entry : mtpTpKvHeadScheme) {
            if (entry.second.empty()) continue;
            auto &shard = cache.shards[entry.first];
            shard.reset(new MtpKvCache());
            shard->tokens = cache.tokens;
            int head = entry.second.front().first;
            shard->key.CopyFrom(Data(FLOAT32, {1, cache.tokens, head_dim},
                                     std::vector<float>(24, float(head + 1))));
            shard->value.CopyFrom(Data(FLOAT32, {1, cache.tokens, head_dim},
                                       std::vector<float>(24, float(-head - 1))));
        }
        Data key, value;
        Require(SnapshotMtpPagedCache(cache, key, value),
                "complete active MTP shards with a zero-head rank were rejected");
        for (int head = 0; head < num_key_value_heads; ++head) {
            for (int i = 0; i < 24; ++i) {
                Require(((float*)key.cpuData)[head * 24 + i] == float(head + 1) &&
                        ((float*)value.cpuData)[head * 24 + i] == float(-head - 1),
                        "MTP snapshot omitted or reordered a global KV head");
            }
        }
        auto reject = [&] {
            Data rejectedKey, rejectedValue;
            Require(!SnapshotMtpPagedCache(cache, rejectedKey, rejectedValue),
                    "incomplete active MTP shard was accepted");
            Require(rejectedKey.dims.empty() && rejectedValue.dims.empty(),
                    "invalid MTP snapshot changed its destinations");
        };
        auto saved = std::move(cache.shards.at(7));
        cache.shards.erase(7);
        reject();
        cache.shards[7] = std::move(saved);
        auto &shard = *cache.shards.at(7);
        --shard.tokens;
        reject();
        ++shard.tokens;
        shard.value.dims[0] = 2;
        reject();
        shard.value.dims[0] = 1;
        shard.value.dataType = FLOAT16;
        reject();
        shard.value.dataType = FLOAT32;
        mtpTpKvHeadScheme.erase(3);
        reject();
        mtpTpKvHeadScheme = threadTpAttentionKVHeadSchemes[0];
        Require(SnapshotMtpPagedCache(cache, key, value),
                "valid MTP snapshot failed after restoring missing state");
#endif
    }

    void CheckSchemes() {
        auto original = threadTpAttentionKVHeadSchemes;
        threadTpAttentionKVHeadSchemes[0].erase(3); // Unknown is not zero heads.
        Check(false);
        threadTpAttentionKVHeadSchemes = original;
        threadTpAttentionKVHeadSchemes[0][3] = {{1, 0}};
        Check(false);
        threadTpAttentionKVHeadSchemes[0][3] = {{4, 5}};
        Check(false);
        threadTpAttentionKVHeadSchemes = original;
        threadTpAttentionKVHeadSchemes.resize(1);
        Check(false);
        threadTpAttentionKVHeadSchemes = original;
        Check(true);
    }
};

int main() {
    PrefixFixture model;
    model.Allocate();
    model.Check(true);
    Require(model.GetPagedKVCacheManagers(0, true).front().first == 1 &&
            model.GetPagedKVCacheManagers(0, true).size() == 4 &&
            model.GetPagedKVCacheManagers(1, false).front().first == 3 &&
            model.GetPagedKVCacheManagers(1, false).size() == 4,
            "per-layer zero-head rank was not excluded in prepared rank order");
    model.CheckSchemes();
    model.CheckMtpSnapshot();
    for (int layer : {0, 1}) {
        for (int keyFlag : {0, 1}) {
            model.Allocate(layer, 2, keyFlag);
            Require(model.GetPagedKVCacheManagers(layer, keyFlag == 0).empty(),
                    "missing active manager returned a partial rank set");
            model.Check(false);
        }
    }
    model.Allocate();
    auto values = model.GetPagedKVCacheManagers(1, false);
    values.back().second->pageLen = 8;
    model.Check(false);
    values.back().second->pageLen = 16;
    model.Check(true);
    ClearAllPagedCacheManagers();
    std::cout << "Qwen3.5 TP prefix manager completeness: PASS\n";
}
