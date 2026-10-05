#pragma once

#include "moe_cache_config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastllm {

// Prefill retains already-uploaded records in bulk. Decode observes every layer
// before spending a global admission budget, so early layers cannot monopolize it.
class MoeFrequencyPolicy {
public:
    static constexpr int prefillHalfLife = 128;
    static constexpr float prefillReplacementFactor = 1.25f;
    static constexpr float prefillReplacementMargin = 1.0f;
    struct Admission { int key = -1, slot = -1; };

    MoeFrequencyPolicy(std::vector<int> keyPartitions, std::vector<int> slotPartitions,
                       int layers, MoeCacheConfig config = {},
                       std::vector<uint64_t> recordBytes = {})
        : config(config), keys(keyPartitions.size()), slots(slotPartitions.size(), -1),
          keyPartitions(std::move(keyPartitions)), slotPartitions(std::move(slotPartitions)),
          recordBytes(std::move(recordBytes)), layers(layers) {
        config.Validate();
        if (layers <= 0) throw std::invalid_argument("invalid MoE layer count");
        int count = 0;
        for (int p : this->keyPartitions) {
            if (p < 0) throw std::invalid_argument("invalid MoE key partition");
            count = std::max(count, p + 1);
        }
        residents.resize(count);
        for (int p : this->slotPartitions)
            if (p < 0 || p >= count) throw std::invalid_argument("invalid MoE slot partition");
        if (this->recordBytes.empty() && !config.maxBytes && !config.rankByBytes)
            this->recordBytes.assign(keys.size(), 1);
        if (this->recordBytes.size() != keys.size() ||
            std::any_of(this->recordBytes.begin(), this->recordBytes.end(),
                        [](uint64_t bytes) { return bytes == 0; }))
            throw std::invalid_argument("invalid MoE expert record sizes");
        // Tiny caches may share oversized slots across different record sizes.
        // Separate candidate queues by size, while retaining the real eviction
        // partition, so byte limits/ranking charge each expert's actual payload.
        std::map<std::pair<int, uint64_t>, int> queueIds;
        for (int k = 0; k < int(keys.size()); ++k)
            queueIds.emplace(std::make_pair(this->keyPartitions[k], this->recordBytes[k]), 0);
        for (auto &entry : queueIds) {
            entry.second = groups.size();
            groups.push_back({entry.first.first, entry.first.second});
        }
        for (int k = 0; k < int(keys.size()); ++k)
            keyGroups.push_back(queueIds.at({this->keyPartitions[k], this->recordBytes[k]}));
        SetResidents(slots.data());
    }

    // Reconcile external cache updates without discarding observed frequencies.
    void SetResidents(const int *owners) {
        if (!slots.empty() && !owners) throw std::invalid_argument("missing MoE residency snapshot");
        for (auto &p : residents) p.clear();
        for (auto &key : keys) key.slot = -1;
        for (int s = 0; s < int(slots.size()); ++s) {
            const int key = owners[s];
            if (key < -1 || key >= int(keys.size()) ||
                (key >= 0 && (keyPartitions[key] != slotPartitions[s] || keys[key].slot >= 0)))
                throw std::invalid_argument("invalid MoE residency snapshot");
            slots[s] = key;
            if (key >= 0) keys[key].slot = s;
            residents[slotPartitions[s]].emplace(key < 0 ? -1.0f : keys[key].score, s);
        }
    }

    void BeginStep() {
        ++steps;
        if (prefill) {
            ScaleScores(config.prefillPrior * (config.halfLife > 0 ? config.halfLife : prefillHalfLife) /
                        prefillHalfLife);
            prefill = false;
            decodeSteps = 0;
        }
        ++decodeSteps;
        active = true;
    }

    void Observe(int base, const int *experts, int count) {
        for (int r = 0; r < count; ++r) {
            const int64_t key = int64_t(base) + experts[r];
            if (experts[r] < 0 || key < 0 || key >= int64_t(keys.size())) continue;
            // One observation per token/expert, including duplicate TopK routes.
            bool duplicate = false;
            for (int i = 0; i < r; ++i) duplicate |= experts[i] == experts[r];
            if (duplicate) continue;
            auto &entry = keys[key];
            auto &part = residents[keyPartitions[key]];
            if (entry.slot >= 0) part.erase({entry.score, entry.slot});
            entry.score += 1;
            if (entry.slot >= 0) part.emplace(entry.score, entry.slot);
        }
    }

    // Plan only after the whole token has been observed. Applying these
    // admissions cannot improve the hit count of the token that selected them.
    // Returned slots are reserved in the policy; the caller publishes payload
    // and device residency together before the next cache lookup.
    std::vector<Admission> EndStep() {
        if (!active) return {};
        active = false;
        if (decodeSteps % config.updateInterval != 0) return {};
        std::vector<Admission> plan;
        if (config.maxReplacements > 0 && !slots.empty()) {
            std::vector<std::vector<int>> candidates(groups.size());
            std::vector<size_t> next(groups.size(), 0);
            std::vector<bool> reserved(slots.size(), false);
            for (int key = 0; key < int(keys.size()); ++key)
                if (keys[key].slot < 0 && keys[key].score > 0 && keys[key].score >= config.minHeat)
                    candidates[keyGroups[key]].push_back(key);
            for (auto &part : candidates)
                std::sort(part.begin(), part.end(), [&](int a, int b) {
                    return keys[a].score != keys[b].score ? keys[a].score > keys[b].score : a < b;
                });
            uint64_t bytes = 0;
            while (int(plan.size()) < config.maxReplacements) {
                Admission best;
                int bestGroup = -1;
                double bestGain = 0;
                for (int g = 0; g < int(groups.size()); ++g) {
                    if (next[g] == candidates[g].size() ||
                        (config.maxBytes && groups[g].bytes > config.maxBytes - bytes)) continue;
                    const int candidate = candidates[g][next[g]];
                    for (auto victim : residents[groups[g].partition]) {
                        if (reserved[victim.second]) continue;
                        const int old = slots[victim.second];
                        if (old >= 0 && steps - keys[old].admitted < uint64_t(config.minimumResidence)) continue;
                        double gain = old < 0 ? 1e10 : keys[candidate].score -
                            keys[old].score * config.replacementFactor - config.replacementMargin;
                        if (config.rankByBytes) gain /= double(groups[g].bytes);
                        if (gain > bestGain) {
                            bestGain = gain;
                            best = {candidate, victim.second};
                            bestGroup = g;
                        }
                        break;
                    }
                }
                if (best.key < 0) break;
                ApplyAdmission(best);
                reserved[best.slot] = true;
                plan.push_back(best);
                bytes += groups[bestGroup].bytes;
                ++next[bestGroup];
            }
        }
        // Match the sampling interval: observe, select, then decay. A first
        // decode step therefore retains a complete unit observation.
        if (config.halfLife > 0)
            ScaleScores(std::exp2(-float(config.updateInterval) / config.halfLife));
        return plan;
    }

    // Prefill preserves its existing scale and borrowed-pointer protection.
    // Only the first prefill layer converts preceding decode history.
    void ObservePrefill(int base, const std::vector<int> &counts, int rows) {
        if (rows <= 0 || base < 0 || size_t(base) + counts.size() > keys.size())
            throw std::invalid_argument("invalid MoE prefill histogram");
        if (!prefill) {
            ScaleScores(float(prefillHalfLife) / (config.halfLife > 0 ? config.halfLife : prefillHalfLife));
            prefill = true;
            active = false;
        }
        const float decay = std::exp2(-float(rows) / prefillHalfLife);
        const float scale = (1 - decay) * prefillHalfLife / (std::log(2.0f) * rows);
        for (int e = 0; e < int(counts.size()); ++e) {
            auto &key = keys[base + e];
            auto &part = residents[keyPartitions[base + e]];
            if (key.slot >= 0) part.erase({key.score, key.slot});
            key.score = key.score * decay + counts[e] * scale;
            if (key.slot >= 0) part.emplace(key.score, key.slot);
        }
    }

    Admission SelectPrefill(int key, int base, const std::vector<int> &active) const {
        if (key < 0 || key >= int(keys.size()) || keys[key].slot >= 0 || keys[key].score <= 0)
            return {};
        for (auto victim : residents[keyPartitions[key]]) {
            const int old = slots[victim.second];
            if (old >= base && old < base + int(active.size()) && active[old - base]) continue;
            if (old >= 0 && keys[key].score <= keys[old].score * prefillReplacementFactor +
                prefillReplacementMargin) return {};
            return {key, victim.second};
        }
        return {};
    }

    void Admit(int layer, Admission admission) {
        if (layer < 0 || layer >= layers) throw std::invalid_argument("invalid MoE admission layer");
        ApplyAdmission(admission);
    }

    float Score(int key) const { return keys[key].score; }
    int Slot(int key) const { return keys[key].slot; }

private:
    void ApplyAdmission(Admission admission) {
        const int key = admission.key, slot = admission.slot;
        if (key < 0 || key >= int(keys.size()) || slot < 0 || slot >= int(slots.size()) ||
            keys[key].slot >= 0 || keyPartitions[key] != slotPartitions[slot])
            throw std::invalid_argument("invalid MoE admission");
        auto &part = residents[keyPartitions[key]];
        const int old = slots[slot];
        part.erase({old < 0 ? -1.0f : keys[old].score, slot});
        if (old >= 0) keys[old].slot = -1;
        slots[slot] = key;
        keys[key].slot = slot;
        keys[key].admitted = steps;
        part.emplace(keys[key].score, slot);
    }

    void ScaleScores(float scale) {
        for (auto &key : keys) {
            key.score *= scale;
            if (key.score < 1e-6f) key.score = 0;
        }
        for (auto &p : residents) p.clear();
        for (int s = 0; s < int(slots.size()); ++s)
            residents[slotPartitions[s]].emplace(slots[s] < 0 ? -1.0f : keys[slots[s]].score, s);
    }

    struct Key { float score = 0; uint64_t admitted = 0; int slot = -1; };
    MoeCacheConfig config;
    uint64_t steps = 0, decodeSteps = 0;
    bool prefill = false, active = false;
    std::vector<Key> keys;
    std::vector<int> slots, keyPartitions, slotPartitions;
    std::vector<uint64_t> recordBytes;
    struct Group { int partition; uint64_t bytes; };
    std::vector<Group> groups;
    std::vector<int> keyGroups;
    std::vector<std::set<std::pair<float, int>>> residents;
    int layers;
};
} // namespace fastllm
