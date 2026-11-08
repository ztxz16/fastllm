#include "models/qwen4_pipeline_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace fastllm::qwen4_pipeline;

static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F> static void Reject(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception &) { rejected = true; }
    Check(rejected, "invalid plan/index was accepted");
}

static HeadSpec TargetSpec() { return {24, 2, 16, 48, 640, 256, 128}; }

static void Reconstruct(int size, const std::vector<std::vector<Range>> &shards) {
    std::vector<int> source(size), output(size, -1), visits(size, 0);
    std::iota(source.begin(), source.end(), 0);
    for (const auto &ranges : shards) {
        std::vector<int> local;
        for (const auto &range : ranges) {
            Check(range.begin >= 0 && range.end <= size, "slice out of source bounds");
            local.insert(local.end(), source.begin() + range.begin, source.begin() + range.end);
        }
        size_t offset = 0;
        for (const auto &range : ranges) for (int i = range.begin; i < range.end; ++i) {
            output[i] = local.at(offset++);
            ++visits[i];
        }
    }
    Check(output == source, "split/reconstructed weights changed ordering");
    Check(std::all_of(visits.begin(), visits.end(), [](int n) { return n == 1; }),
          "source rows omitted or counted twice");
}

static void ExactTp5() {
    const auto plan = HeadPartitionPlan::Build(TargetSpec(), 5);
    const std::array<int, 6> qb{0,5,10,14,19,24}, kb{0,4,7,10,13,16};
    const std::array<std::vector<int>, 5> kv{{{0},{0},{0,1},{1},{1}}};
    std::vector<std::vector<Range>> qrows, ocols, linear, vrows, scores, gateup, down;
    for (int r = 0; r < 5; ++r) {
        const auto &local = plan.Rank(r);
        Check(local.query.begin == qb[r] && local.query.end == qb[r+1], "Q5/5/4/5/5");
        Check(local.linear_key.begin == kb[r] && local.linear_key.end == kb[r+1], "K4/3/3/3/3");
        Check(local.linear_value.begin == 3*kb[r] && local.linear_value.end == 3*kb[r+1],
              "linear V must stay with its K head");
        Check(local.kv_ids == kv[r], "KV A/A/A+B/B/B");
        Check(local.expert.begin == r*128 && local.expert.end == (r+1)*128, "expert128");
        for (int q = local.query.begin; q < local.query.end; ++q) {
            const int relative = q - local.query.begin;
            Check(local.kv_ids.at(local.query_to_local_kv.at(relative)) == q/12,
                  "query is paired with the wrong KV head");
        }
        qrows.push_back(plan.QueryProjectionRows(r));
        ocols.push_back(plan.AttentionOutputColumns(r));
        linear.push_back(plan.LinearQkvRows(r));
        vrows.push_back(plan.LinearValueRows(r));
        scores.push_back({local.linear_value});
        gateup.push_back(plan.ExpertGateUpRows(r));
        down.push_back({local.expert});
    }
    // Q projection stores adjacent Q/gate vectors within each global head.
    Reconstruct(24*256*2, qrows);
    Reconstruct(24*256, ocols);
    Reconstruct((16*2+48)*128, linear);
    Reconstruct(48*128, vrows);
    Reconstruct(48, scores);
    Reconstruct(640*2, gateup);
    Reconstruct(640, down);
    Check(plan.Rank(2).query_to_local_kv == std::vector<int>({0,0,1,1}),
          "middle rank must pair two Q heads with A and two with B");
    Reject([&] { plan.Rank(-1); });
    Reject([&] { plan.Rank(5); });
}

static void AttentionReference() {
    const auto plan = HeadPartitionPlan::Build(TargetSpec(), 5);
    constexpr int dim = 3, tokens = 7;
    // Independent full-model oracle: global Q0..11 use KV0, Q12..23 use KV1.
    const auto evaluate = [&](int q, int k) {
        std::array<double, tokens> scores{};
        double maxScore = -1e100;
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < dim; ++d)
                scores[t] += (q+1)*(d+1)*0.013 * (k+1)*(t+1)*(d+1)*0.071;
            maxScore = std::max(maxScore, scores[t]);
        }
        double sum = 0, output = 0;
        for (int t = 0; t < tokens; ++t) {
            const double p = std::exp(scores[t]-maxScore);
            sum += p;
            output += p * (100*k + 3*t + 0.125);
        }
        return output/sum;
    };
    for (int r = 0; r < 5; ++r) {
        const auto &local = plan.Rank(r);
        for (int q = local.query.begin; q < local.query.end; ++q) {
            const int k = local.kv_ids.at(local.query_to_local_kv.at(q-local.query.begin));
            Check(std::abs(evaluate(q, q/12)-evaluate(q,k)) < 1e-12,
                  "ragged GQA output differs from full attention");
        }
    }
    Check(std::abs(evaluate(12,0)-evaluate(12,1)) > 50,
          "attention oracle cannot catch the one-KV middle-rank bug");
}

static void OtherDegreesAndMtp() {
    const std::array<std::vector<int>, 3> expected{{{12,12},{6,6,6,6},{3,3,3,3,3,3,3,3}}};
    for (int i = 0; i < 3; ++i) {
        const int degree = std::array<int,3>{2,4,8}[i];
        const auto plan = HeadPartitionPlan::Build(TargetSpec(), degree);
        std::vector<std::vector<Range>> q, linear, expert;
        for (int r = 0; r < degree; ++r) {
            Check(plan.Rank(r).query.Size() == expected[i][r], "even TP Q compatibility");
            q.push_back(plan.AttentionOutputColumns(r));
            linear.push_back(plan.LinearQkvRows(r));
            expert.push_back(plan.ExpertGateUpRows(r));
        }
        Reconstruct(24*256,q);
        Reconstruct((16*2+48)*128,linear);
        Reconstruct(640*2,expert);
    }
    auto spec = TargetSpec(); spec.linear_key_heads = spec.linear_value_heads = 0;
    const auto mtp = HeadPartitionPlan::Build(spec,5);
    Check(mtp.Rank(2).kv_ids == std::vector<int>({0,1}), "MTP middle rank needs both KV");
    for (int r = 0; r < 5; ++r) {
        Check(mtp.Rank(r).linear_key.Size()==0 && mtp.Rank(r).linear_value.Size()==0,
              "MTP must not inherit linear-attention heads");
        Check(mtp.Rank(r).expert.Size()==128, "MTP BF16 expert width");
    }
    Reject([] { HeadPartitionPlan::Build(TargetSpec(),0); });
    Reject([] { HeadPartitionPlan::Build(TargetSpec(),25); });
    spec = TargetSpec(); spec.query_heads = 23;
    Reject([&] { HeadPartitionPlan::Build(spec,5); });
    spec = TargetSpec(); spec.linear_value_heads = 47;
    Reject([&] { HeadPartitionPlan::Build(spec,5); });
    spec = TargetSpec(); spec.expert_width = 639;
    Reject([&] { HeadPartitionPlan::Build(spec,5); });
}

static void TestStageWeightOwnership() {
    using fastllm::qwen4_pipeline::StageWeightOwnership;
    Check(StageWeightOwnership::Owner("model.language_model.layers.0.self_attn.q_proj.weight") == 0,
          "layer 0 must belong to stage 0");
    Check(StageWeightOwnership::Owner("model.language_model.layers.23.mlp.gate.weight") == 0,
          "layer 23 must belong to stage 0");
    Check(StageWeightOwnership::Owner("model.language_model.layers.24.self_attn.q_proj.weight") == 1,
          "layer 24 must belong to stage 1");
    Check(StageWeightOwnership::Owner("model.language_model.layers.47.mlp.gate.weight") == 1,
          "layer 47 must belong to stage 1");
    Check(StageWeightOwnership::Owner(
              "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight") == 0,
          "PLE must stay in stage 0");
    Check(StageWeightOwnership::Owner("model.language_model.embed_tokens.weight") == 0,
          "shared token table owner must be stage 0");
    Check(StageWeightOwnership::Owner("lm_head.weight") == 1,
          "output head must stay in stage 1");
    Check(StageWeightOwnership::Owner("mtp.fc_embedding.weight") == 1 &&
              StageWeightOwnership::IsMtp("mtp.layers.0.self_attn.q_proj.weight"),
          "MTP must stay in stage 1");
    Check(StageWeightOwnership::Owner("model.language_model.layers.48.mlp.gate.weight") == -1,
          "out-of-range target layer accepted");
    Check(StageWeightOwnership::Owner("model.language_model.hyper_connection_mixer.hc_norm.weight") == 1,
          "final hyper connection mixer must stay in stage 1");
    Check(StageWeightOwnership::Owner("model.visual.blocks.0.attn.qkv.weight") == 0,
          "vision ownership must be stage 0 when enabled");
    Check(StageWeightOwnership::Owner("wrong.model.language_model.layers.0.mlp.gate.weight") == -1,
          "unknown prefix treated as a target layer");
}

static void StageMapping() {
    std::vector<LayerKind> kinds(48,LayerKind::LinearAttention);
    for (int g = 3; g < 48; g += 4) kinds[g] = LayerKind::FullAttention;
    const auto s0 = StageRuntimeLayerMap::Build(0,0,24,kinds,{1},false);
    const auto s1 = StageRuntimeLayerMap::Build(1,24,48,kinds,{1},true);
    Check(s0.GlobalLayerCount()==48 && s1.GlobalLayerCount()==48, "global count retained");
    Check(s0.LocalLayerCount()==24 && s1.LocalLayerCount()==24, "stage count24");
    Check(s0.FirstFullAttentionGlobal()==3 && s1.FirstFullAttentionGlobal()==27, "first KV global IDs");
    Check(s0.FullAttentionLayers().size()==6 && s1.FullAttentionLayers().size()==6, "six full layers/stage");
    Check(s0.OwnsPle() && !s1.OwnsPle() && !s0.OwnsMtp() && s1.OwnsMtp(), "stage ownership");
    for (int g = 0; g < 48; ++g) {
        const auto &owner = g<24 ? s0 : s1;
        Check(owner.LocalToGlobal(owner.GlobalToLocal(g))==g, "global/local layer round trip");
        Check(owner.Kind(g)==kinds[g], "global layer type changed");
    }
    Reject([&] { s1.GlobalToLocal(3); });
    Reject([&] { s0.GlobalToLocal(24); });
    Reject([&] { s1.LocalToGlobal(24); });
    Reject([&] { StageRuntimeLayerMap::Build(0,24,49,kinds,{1},false); });
    Reject([&] { StageRuntimeLayerMap::Build(0,0,24,kinds,{48},false); });
    Check(s0.LayoutHash()!=s1.LayoutHash(), "layout hashes do not distinguish stages");
    Check(s1.LayoutHash()==StageRuntimeLayerMap::Build(1,24,48,kinds,{1},true).LayoutHash(),
          "layout hash depends on memory addresses");
    std::vector<std::future<uint64_t>> readers;
    for (int i = 0; i < 4; ++i) readers.push_back(std::async(std::launch::async,[&] {
        for (int n = 0; n < 1000; ++n) Check(s1.GlobalToLocal(27)==3, "concurrent mapping changed");
        return s1.LayoutHash();
    }));
    for (auto &reader : readers) Check(reader.get()==s1.LayoutHash(), "concurrent hash changed");
}

int main() {
    try {
        ExactTp5(); AttentionReference(); OtherDegreesAndMtp(); StageMapping(); TestStageWeightOwnership();
        std::cout << "PASS TP5 projections, KV mapping, expert reconstruction, TP2/4/8, MTP and PP2 layers\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
