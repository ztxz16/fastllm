#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fastllm {

// Decode admission only. Execution stays on the CPU for nonresident experts,
// including the call that admits an expert. Prefill owns its existing policy.
class MoeFrequencyPolicy {
public:
    static constexpr int halfLife = 128;
    static constexpr int decayInterval = 32;
    static constexpr int maxReplacementsPerStep = 8;
    static constexpr int maxFillsPerLayer = 4;
    static constexpr int minimumResidence = 8;
    static constexpr float replacementFactor = 1.25f;
    static constexpr float replacementMargin = 1.0f;

    struct Admission { int key = -1, slot = -1; };

    // Keys and slots are partitioned by their actual packed record size.
    MoeFrequencyPolicy(std::vector<int> keyPartitions, std::vector<int> slotPartitions,
                       int layers)
        : keys(keyPartitions.size()), slots(slotPartitions.size(), -1),
          keyPartitions(std::move(keyPartitions)), slotPartitions(std::move(slotPartitions)),
          layers(layers), admittedLayer(layers, false), filledLayer(layers, 0) {
        if (layers <= 0) throw std::invalid_argument("invalid MoE layer count");
        int count = 0;
        for (int p : this->keyPartitions) {
            if (p < 0) throw std::invalid_argument("invalid MoE key partition");
            count = std::max(count, p + 1);
        }
        residents.resize(count);
        for (int p : this->slotPartitions)
            if (p < 0 || p >= count) throw std::invalid_argument("invalid MoE slot partition");
    }

    // Reconcile after prefill; retained frequencies still describe decode.
    void SetResidents(const int *owners) {
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
        used = 0;
        std::fill(admittedLayer.begin(), admittedLayer.end(), false);
        std::fill(filledLayer.begin(), filledLayer.end(), 0);
        if (steps % decayInterval != 0) return;
        const float decay = std::exp2(-float(decayInterval) / halfLife);
        for (auto &key : keys) {
            key.score *= decay;
            if (key.score < 1e-6f) key.score = 0;
        }
        for (auto &p : residents) p.clear();
        for (int s = 0; s < int(slots.size()); ++s)
            residents[slotPartitions[s]].emplace(slots[s] < 0 ? -1.0f : keys[slots[s]].score, s);
    }

    void Observe(int base, const int *experts, int count) {
        for (int r = 0; r < count; ++r) {
            const int key = base + experts[r];
            if (experts[r] < 0 || key < 0 || key >= int(keys.size())) continue;
            // Top-k is normally unique; repeated routes must not inflate heat.
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

    Admission Select(int layer, int base, const int *experts, int count) const {
        if (steps == 0 || layer < 0 || layer >= layers) return {};
        int candidate = -1;
        for (int r = 0; r < count; ++r) {
            const int key = base + experts[r];
            if (experts[r] < 0 || key < 0 || key >= int(keys.size())) continue;
            if (keys[key].slot < 0 && keys[key].score >= 1 &&
                (candidate < 0 || keys[key].score > keys[candidate].score)) candidate = key;
        }
        if (candidate < 0) return {};
        const auto &part = residents[keyPartitions[candidate]];
        // Empty slots have no eviction cost. Bootstrap them separately so a
        // streamed prefill does not leave decode CPU-bound for hundreds of tokens.
        if (!part.empty() && slots[part.begin()->second] < 0)
            return filledLayer[layer] < maxFillsPerLayer ? Admission{candidate, part.begin()->second} : Admission{};
        const int budget = std::min(layers, maxReplacementsPerStep);
        // Rotate eligibility so early layers cannot exhaust the token budget.
        if (used >= budget || admittedLayer[layer] || keys[candidate].score < 2 ||
            (layer + layers - int(((steps - 1) * budget) % layers)) % layers >= budget) return {};
        for (auto victim : part) {
            const int old = slots[victim.second];
            if (old >= 0 && steps - keys[old].admitted < minimumResidence) continue;
            if (old >= 0 && keys[candidate].score <=
                keys[old].score * replacementFactor + replacementMargin) return {};
            return {candidate, victim.second};
        }
        return {};
    }

    void Admit(int layer, Admission admission) {
        const int key = admission.key, slot = admission.slot;
        if (layer < 0 || layer >= layers || key < 0 || key >= int(keys.size()) ||
            slot < 0 || slot >= int(slots.size()) || keys[key].slot >= 0 ||
            keyPartitions[key] != slotPartitions[slot])
            throw std::invalid_argument("invalid MoE admission");
        auto &part = residents[keyPartitions[key]];
        const int old = slots[slot];
        part.erase({old < 0 ? -1.0f : keys[old].score, slot});
        if (old >= 0) keys[old].slot = -1;
        slots[slot] = key;
        keys[key].slot = slot;
        keys[key].admitted = steps;
        part.emplace(keys[key].score, slot);
        if (old >= 0) {
            admittedLayer[layer] = true;
            ++used;
        } else {
            ++filledLayer[layer];
        }
    }

    float Score(int key) const { return keys[key].score; }

private:
    struct Key { float score = 0; uint64_t admitted = 0; int slot = -1; };
    uint64_t steps = 0;
    std::vector<Key> keys;
    std::vector<int> slots, keyPartitions, slotPartitions;
    int layers, used = 0;
    std::vector<bool> admittedLayer;
    std::vector<int> filledLayer;
    std::vector<std::set<std::pair<float, int>>> residents;
};
} // namespace fastllm
