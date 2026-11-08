#pragma once

// Small, device-independent contracts shared by the Qwen4 TP/PP loader and
// its tests.  The types deliberately contain ranges in global coordinates so
// a shard can be reconstructed without guessing which rank owned a row.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm::qwen4_pipeline {

// A dense CUDA verifier must use the same attention capacity boundary as
// one-token decoding. Apply this before either PP stage consumes the rows.
inline int VerificationDraftCount(int drafts, int previous, int dense_width,
                                  int indexer_budget, bool cuda_graph) {
    if (!cuda_graph || drafts <= 0) return drafts;
    const int remaining = std::min(dense_width, indexer_budget) - previous;
    if (previous < indexer_budget && remaining > 0 && remaining <= drafts) {
        return remaining - 1;
    }
    return drafts;
}

struct Range {
    int begin = 0;
    int end = 0;
    int Size() const { return end - begin; }
};

inline bool operator==(const Range &a, const Range &b) {
    return a.begin == b.begin && a.end == b.end;
}

enum class LayerKind : uint8_t {
    LinearAttention = 0,
    FullAttention = 1,
};

struct HeadSpec {
    int query_heads = 0;
    int kv_heads = 0;
    int linear_key_heads = 0;
    int linear_value_heads = 0;
    int expert_width = 0;
    int head_dim = 0;
    int linear_head_dim = 0;
};

struct RankHeadSpec {
    Range query;
    Range linear_key;
    Range linear_value;
    Range expert;
    std::vector<int> kv_ids;
    // One entry per local query head; the value indexes kv_ids, rather than
    // being a global KV ID. This makes rank 2's A/A+B/B mapping explicit.
    std::vector<int> query_to_local_kv;
};

inline Range BalancedRange(int size, int parts, int rank) {
    if (size < 0 || parts <= 0 || rank < 0 || rank >= parts) {
        throw std::invalid_argument("invalid balanced range arguments");
    }
    // Round each boundary to the nearest complete head. In particular,
    // Q24/TP5 is [0,5,10,14,19,24], without cutting a head in two.
    return {static_cast<int>((static_cast<int64_t>(size) * rank + parts / 2) / parts),
            static_cast<int>((static_cast<int64_t>(size) * (rank + 1) + parts / 2) / parts)};
}

class HeadPartitionPlan {
public:
    static HeadPartitionPlan Build(const HeadSpec &spec, int tp_size, int expert_rotation = 0) {
        if (tp_size <= 0 || tp_size > spec.query_heads || spec.query_heads <= 0 ||
            spec.kv_heads <= 0 || spec.query_heads % spec.kv_heads != 0 ||
            spec.head_dim <= 0 || spec.linear_head_dim <= 0 || spec.expert_width <= 0 ||
            spec.expert_width % 128 != 0 || expert_rotation < 0) {
            throw std::invalid_argument("unsupported Qwen4 TP head specification");
        }
        if (spec.linear_key_heads < 0 || spec.linear_value_heads < 0 ||
            (spec.linear_key_heads == 0) != (spec.linear_value_heads == 0) ||
            (spec.linear_key_heads > 0 && spec.linear_value_heads % spec.linear_key_heads != 0)) {
            throw std::invalid_argument("linear Q/K/V head specification is inconsistent");
        }
        HeadPartitionPlan result;
        result.spec_ = spec;
        result.ranks_.resize(tp_size);
        const int q_per_kv = spec.query_heads / spec.kv_heads;
        int expert_offset = 0;
        for (int rank = 0; rank < tp_size; ++rank) {
            RankHeadSpec &local = result.ranks_[rank];
            local.query = BalancedRange(spec.query_heads, tp_size, rank);
            const int keys = spec.linear_key_heads / tp_size;
            const int extra_keys = spec.linear_key_heads % tp_size;
            local.linear_key = {
                rank * keys + std::min(rank, extra_keys),
                (rank + 1) * keys + std::min(rank + 1, extra_keys),
            };
            const int values_per_key = spec.linear_key_heads == 0
                ? 0 : spec.linear_value_heads / spec.linear_key_heads;
            local.linear_value = {
                local.linear_key.begin * values_per_key,
                local.linear_key.end * values_per_key,
            };
            // Keep Marlin's 128-column alignment. TP8 has three empty expert
            // slices for width640; TP4 has one 256-column slice. The rotation
            // matches the existing layer balancing policy for those degrees.
            const int blocks = spec.expert_width / 128;
            const int local_blocks = blocks / tp_size +
                ((rank + expert_rotation % tp_size) % tp_size < blocks % tp_size ? 1 : 0);
            local.expert = {expert_offset, expert_offset + local_blocks * 128};
            expert_offset = local.expert.end;
            for (int q = local.query.begin; q < local.query.end; ++q) {
                const int global_kv = q / q_per_kv;
                auto it = std::find(local.kv_ids.begin(), local.kv_ids.end(), global_kv);
                if (it == local.kv_ids.end()) {
                    local.kv_ids.push_back(global_kv);
                    it = local.kv_ids.end() - 1;
                }
                local.query_to_local_kv.push_back(
                    static_cast<int>(it - local.kv_ids.begin()));
            }
        }
        return result;
    }

    const RankHeadSpec &Rank(int rank) const {
        if (rank < 0 || rank >= static_cast<int>(ranks_.size())) {
            throw std::out_of_range("Qwen4 TP rank is outside the plan");
        }
        return ranks_[rank];
    }

    int Size() const { return static_cast<int>(ranks_.size()); }
    const HeadSpec &Spec() const { return spec_; }

    std::vector<Range> QueryProjectionRows(int rank) const {
        const auto &r = Rank(rank).query;
        const int stride = spec_.head_dim * 2;
        return {{r.begin * stride, r.end * stride}};
    }

    std::vector<Range> AttentionOutputColumns(int rank) const {
        const auto &r = Rank(rank).query;
        return {{r.begin * spec_.head_dim, r.end * spec_.head_dim}};
    }

    std::vector<Range> LinearQkvRows(int rank) const {
        const auto &r = Rank(rank);
        if (r.linear_key.Size() == 0) return {};
        const int keyRows = spec_.linear_key_heads * spec_.linear_head_dim;
        return {
            {r.linear_key.begin * spec_.linear_head_dim,
             r.linear_key.end * spec_.linear_head_dim},
            {keyRows + r.linear_key.begin * spec_.linear_head_dim,
             keyRows + r.linear_key.end * spec_.linear_head_dim},
            {2 * keyRows + r.linear_value.begin * spec_.linear_head_dim,
             2 * keyRows + r.linear_value.end * spec_.linear_head_dim},
        };
    }

    std::vector<Range> LinearValueRows(int rank) const {
        const auto &r = Rank(rank).linear_value;
        return r.Size() == 0 ? std::vector<Range>{} :
            std::vector<Range>{{r.begin * spec_.linear_head_dim,
                                r.end * spec_.linear_head_dim}};
    }

    std::vector<Range> ExpertGateUpRows(int rank) const {
        const auto &r = Rank(rank).expert;
        return {{r.begin, r.end},
                {spec_.expert_width + r.begin, spec_.expert_width + r.end}};
    }

private:
    HeadSpec spec_;
    std::vector<RankHeadSpec> ranks_;
};

class StageRuntimeLayerMap {
public:
    static StageRuntimeLayerMap Build(int stage, int begin, int end,
                                      const std::vector<LayerKind> &kinds,
                                      const std::vector<int> &ple_layers,
                                      bool owns_mtp) {
        if (stage < 0 || begin < 0 || begin >= end || end > static_cast<int>(kinds.size())) {
            throw std::invalid_argument("invalid Qwen4 PP stage range");
        }
        StageRuntimeLayerMap result;
        result.stage_ = stage;
        result.begin_ = begin;
        result.end_ = end;
        result.global_count_ = static_cast<int>(kinds.size());
        result.kinds_ = kinds;
        result.owns_mtp_ = owns_mtp;
        for (int global : ple_layers) {
            if (global < 0 || global >= result.global_count_) {
                throw std::invalid_argument("PLE layer is outside the model");
            }
            if (global >= begin && global < end) result.owns_ple_ = true;
        }
        for (int global = begin; global < end; ++global) {
            if (kinds[global] == LayerKind::FullAttention) {
                result.full_attention_layers_.push_back(global);
            }
        }
        result.hash_ = result.ComputeHash(ple_layers);
        return result;
    }

    int GlobalLayerCount() const { return global_count_; }
    int LocalLayerCount() const { return end_ - begin_; }
    int FirstFullAttentionGlobal() const {
        if (full_attention_layers_.empty()) return -1;
        return full_attention_layers_.front();
    }
    const std::vector<int> &FullAttentionLayers() const { return full_attention_layers_; }
    bool OwnsPle() const { return owns_ple_; }
    bool OwnsMtp() const { return owns_mtp_; }
    uint64_t LayoutHash() const { return hash_; }

    int GlobalToLocal(int global) const {
        if (global < begin_ || global >= end_) {
            throw std::out_of_range("global layer does not belong to this PP stage");
        }
        return global - begin_;
    }

    int LocalToGlobal(int local) const {
        if (local < 0 || local >= end_ - begin_) {
            throw std::out_of_range("local layer is outside this PP stage");
        }
        return begin_ + local;
    }

    LayerKind Kind(int global) const {
        if (global < 0 || global >= global_count_) {
            throw std::out_of_range("layer is outside the global model");
        }
        return kinds_[global];
    }

private:
    uint64_t ComputeHash(const std::vector<int> &ple_layers) const {
        uint64_t hash = 14695981039346656037ULL;
        const auto add = [&hash](uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ULL;
        };
        add(static_cast<uint64_t>(stage_));
        add(static_cast<uint64_t>(begin_));
        add(static_cast<uint64_t>(end_));
        add(static_cast<uint64_t>(global_count_));
        add(owns_ple_ ? 1 : 0);
        add(owns_mtp_ ? 1 : 0);
        for (auto kind : kinds_) add(static_cast<uint64_t>(kind));
        for (int layer : ple_layers) add(static_cast<uint64_t>(layer));
        return hash;
    }

    int stage_ = 0;
    int begin_ = 0;
    int end_ = 0;
    int global_count_ = 0;
    bool owns_ple_ = false;
    bool owns_mtp_ = false;
    uint64_t hash_ = 0;
    std::vector<LayerKind> kinds_;
    std::vector<int> full_attention_layers_;
};

// Stage ownership is a contract shared by the loader and the future PP
// scheduler. It deliberately uses checkpoint names/global layer numbers so a
// stage cannot accidentally load or execute the other stage's target layers.
struct StageWeightOwnership {
    static int TargetLayer(const std::string &name) {
        const std::string marker = "model.language_model.layers.";
        const size_t begin = 0;
        if (name.rfind(marker, 0) != 0) return -1;
        const size_t start = begin + marker.size();
        size_t end = start;
        while (end < name.size() && name[end] >= '0' && name[end] <= '9') ++end;
        if (end == start || end >= name.size() || name[end] != '.') return -1;
        return std::stoi(name.substr(start, end - start));
    }

    static bool IsMtp(const std::string &name) {
        return name.rfind("mtp.", 0) == 0;
    }

    static bool IsSharedTokenEmbedding(const std::string &name) {
        return name == "model.language_model.embed_tokens.weight";
    }

    static bool IsOutputHead(const std::string &name) {
        return name == "lm_head.weight";
    }

    // Stage 0 owns PLE and the shared input table; Stage 1 owns the output
    // head and MTP. Target-layer weights belong to their 24-layer interval.
    static int Owner(const std::string &name, int pleLayer = 1) {
        if (IsMtp(name) || IsOutputHead(name)) return 1;
        if (IsSharedTokenEmbedding(name)) return 0;
        if (name.rfind("model.visual.", 0) == 0) return 0;
        if (name.rfind("model.language_model.hyper_connection_mixer.", 0) == 0 ||
            name.rfind("model.language_model.norm.", 0) == 0) return 1;
        const int layer = TargetLayer(name);
        if (layer < 0) return -1;
        if (name.find(".ple.") != std::string::npos && layer == pleLayer) return 0;
        if (layer < 24) return 0;
        if (layer < 48) return 1;
        return -1;
    }
};

} // namespace fastllm::qwen4_pipeline
