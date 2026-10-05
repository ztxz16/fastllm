#pragma once

#include "moe_cache_config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
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
          layers(layers) {
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
        if (recordBytes.empty() && !config.maxBytes && !config.rankByBytes)
            recordBytes.assign(keys.size(), 1);
        if (recordBytes.size() != keys.size() ||
            std::any_of(recordBytes.begin(), recordBytes.end(),
                        [](uint64_t bytes) { return bytes == 0; }))
            throw std::invalid_argument("invalid MoE expert record sizes");
        // Tiny caches may share oversized slots across different record sizes.
        // Separate candidate queues by size, while retaining the real eviction
        // partition, so byte limits/ranking charge each expert's actual payload.
        std::map<std::pair<int, uint64_t>, int> queueIds;
        for (int k = 0; k < int(keys.size()); ++k)
            queueIds.emplace(std::make_pair(this->keyPartitions[k], recordBytes[k]), 0);
        for (auto &entry : queueIds) {
            entry.second = groups.size();
            groups.push_back({entry.first.first, entry.first.second});
        }
        for (int k = 0; k < int(keys.size()); ++k)
            keyGroups.push_back(queueIds.at({this->keyPartitions[k], recordBytes[k]}));
        candidates.resize(groups.size());
        candidatePositions.resize(keys.size(), -1);
        residentPositions.resize(slots.size(), -1);
        SetResidents(slots.data());
    }

    // Reconcile external cache updates without discarding observed frequencies.
    void SetResidents(const int *owners) {
        if (!slots.empty() && !owners) throw std::invalid_argument("missing MoE residency snapshot");
        for (auto &p : residents) p.ids.clear();
        for (auto &p : candidates) p.ids.clear();
        std::fill(candidatePositions.begin(), candidatePositions.end(), -1);
        std::fill(residentPositions.begin(), residentPositions.end(), -1);
        for (auto &key : keys) key.slot = -1;
        for (int s = 0; s < int(slots.size()); ++s) {
            const int key = owners[s];
            if (key < -1 || key >= int(keys.size()) ||
                (key >= 0 && (keyPartitions[key] != slotPartitions[s] || keys[key].slot >= 0)))
                throw std::invalid_argument("invalid MoE residency snapshot");
            slots[s] = key;
            if (key >= 0) keys[key].slot = s;
            InsertResident(s);
        }
        for (int key = 0; key < int(keys.size()); ++key)
            if (keys[key].slot < 0) InsertCandidate(key);
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
            entry.score += 1;
            UpdateScore(key);
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
            // The heaps persist across tokens. Only observed experts and
            // changed slots move; no full candidate sort or resident rebuild.
            deferredCandidates.clear();
            deferredResidents.clear();
            uint64_t bytes = 0;
            while (int(plan.size()) < config.maxReplacements) {
                Admission best;
                int bestGroup = -1;
                double bestGain = 0;
                for (int g = 0; g < int(groups.size()); ++g) {
                    if (candidates[g].ids.empty() ||
                        (config.maxBytes && groups[g].bytes > config.maxBytes - bytes)) continue;
                    const int candidate = candidates[g].ids.front();
                    if (keys[candidate].score <= 0 || keys[candidate].score < config.minHeat) continue;
                    auto &part = residents[groups[g].partition];
                    while (!part.ids.empty()) {
                        const int slot = part.ids.front(), old = slots[slot];
                        if (old >= 0 && steps - keys[old].admitted < uint64_t(config.minimumResidence)) {
                            RemoveResident(slot);
                            deferredResidents.push_back(slot);
                            continue;
                        }
                        double gain = old < 0 ? 1e10 : keys[candidate].score -
                            keys[old].score * config.replacementFactor - config.replacementMargin;
                        if (config.rankByBytes) gain /= double(groups[g].bytes);
                        if (gain > bestGain) {
                            bestGain = gain;
                            best = {candidate, slot};
                            bestGroup = g;
                        }
                        break;
                    }
                }
                if (best.key < 0) break;
                // A just-evicted expert was absent from the original candidate
                // set, and a reserved slot must not be overwritten in this batch.
                const int old = slots[best.slot];
                RemoveCandidate(best.key);
                RemoveResident(best.slot);
                if (old >= 0) {
                    keys[old].slot = -1;
                    deferredCandidates.push_back(old);
                }
                slots[best.slot] = best.key;
                keys[best.key].slot = best.slot;
                keys[best.key].admitted = steps;
                deferredResidents.push_back(best.slot);
                plan.push_back(best);
                bytes += groups[bestGroup].bytes;
            }
            for (int key : deferredCandidates) InsertCandidate(key);
            for (int slot : deferredResidents) InsertResident(slot);
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
            key.score = key.score * decay + counts[e] * scale;
            UpdateScore(base + e);
        }
    }

    Admission SelectPrefill(int key, int base, const std::vector<int> &active) const {
        if (key < 0 || key >= int(keys.size()) || keys[key].slot >= 0 || keys[key].score <= 0)
            return {};
        const int slot = FindPrefillVictim(residents[keyPartitions[key]], 0, -1, base, active);
        if (slot < 0) return {};
        const int old = slots[slot];
        if (old >= 0 && keys[key].score <= keys[old].score * prefillReplacementFactor +
            prefillReplacementMargin) return {};
        return {key, slot};
    }

    void Admit(int layer, Admission admission) {
        if (layer < 0 || layer >= layers) throw std::invalid_argument("invalid MoE admission layer");
        ApplyAdmission(admission);
    }

    float Score(int key) const { return keys[key].score; }
    int Slot(int key) const { return keys[key].slot; }

private:
    // Indexed binary heap: identifiers never move outside their partition,
    // and the shared position vector makes score updates/removals O(log N).
    struct Heap {
        std::vector<int> ids;
        void Swap(int a, int b, std::vector<int> &positions) {
            std::swap(ids[a], ids[b]);
            positions[ids[a]] = a;
            positions[ids[b]] = b;
        }
        template<class Before> void Up(int i, std::vector<int> &positions, Before before) {
            while (i > 0 && before(ids[i], ids[(i - 1) / 2])) {
                const int parent = (i - 1) / 2;
                Swap(i, parent, positions);
                i = parent;
            }
        }
        template<class Before> void Down(int i, std::vector<int> &positions, Before before) {
            while (2 * i + 1 < int(ids.size())) {
                int child = 2 * i + 1;
                if (child + 1 < int(ids.size()) && before(ids[child + 1], ids[child])) ++child;
                if (!before(ids[child], ids[i])) break;
                Swap(i, child, positions);
                i = child;
            }
        }
        template<class Before> void Changed(int id, std::vector<int> &positions, Before before) {
            const int i = positions[id];
            if (i > 0 && before(ids[i], ids[(i - 1) / 2])) Up(i, positions, before);
            else Down(i, positions, before);
        }
        template<class Before> void Insert(int id, std::vector<int> &positions, Before before) {
            positions[id] = ids.size();
            ids.push_back(id);
            Up(ids.size() - 1, positions, before);
        }
        template<class Before> void Remove(int id, std::vector<int> &positions, Before before) {
            const int i = positions[id], last = ids.back();
            ids[i] = last;
            positions[last] = i;
            ids.pop_back();
            positions[id] = -1;
            if (i < int(ids.size())) Changed(last, positions, before);
        }
        template<class Before> void RepairTies(std::vector<int> &positions, Before before) {
            // Uniform nonnegative decay preserves score order, but float
            // rounding / the zero cutoff can merge scores. Repair only the
            // resulting ID tie inversions, retaining exact previous semantics.
            for (int i = int(ids.size()) / 2 - 1; i >= 0; --i) Down(i, positions, before);
        }
    };

    bool CandidateBefore(int a, int b) const {
        return keys[a].score != keys[b].score ? keys[a].score > keys[b].score : a < b;
    }
    bool ResidentBefore(int a, int b) const {
        const float sa = slots[a] < 0 ? -1.0f : keys[slots[a]].score;
        const float sb = slots[b] < 0 ? -1.0f : keys[slots[b]].score;
        return sa != sb ? sa < sb : a < b;
    }
    void InsertCandidate(int key) {
        if (config.maxReplacements == 0 || slots.empty()) return;
        candidates[keyGroups[key]].Insert(key, candidatePositions,
            [this](int a, int b) { return CandidateBefore(a, b); });
    }
    void RemoveCandidate(int key) {
        if (candidatePositions[key] < 0) return;
        candidates[keyGroups[key]].Remove(key, candidatePositions,
            [this](int a, int b) { return CandidateBefore(a, b); });
    }
    void InsertResident(int slot) {
        residents[slotPartitions[slot]].Insert(slot, residentPositions,
            [this](int a, int b) { return ResidentBefore(a, b); });
    }
    void RemoveResident(int slot) {
        residents[slotPartitions[slot]].Remove(slot, residentPositions,
            [this](int a, int b) { return ResidentBefore(a, b); });
    }
    void UpdateScore(int key) {
        const int slot = keys[key].slot;
        if (slot < 0) {
            if (candidatePositions[key] >= 0)
                candidates[keyGroups[key]].Changed(key, candidatePositions,
                    [this](int a, int b) { return CandidateBefore(a, b); });
        } else residents[slotPartitions[slot]].Changed(slot, residentPositions,
            [this](int a, int b) { return ResidentBefore(a, b); });
    }
    int FindPrefillVictim(const Heap &part, int i, int best, int base,
                          const std::vector<int> &active) const {
        if (i >= int(part.ids.size())) return best;
        const int slot = part.ids[i], key = slots[slot];
        if (best >= 0 && !ResidentBefore(slot, best)) return best;
        if (!(key >= base && key < int64_t(base) + int64_t(active.size()) && active[key - base]))
            return slot;
        int first = 2 * i + 1, second = first + 1;
        if (second < int(part.ids.size()) && ResidentBefore(part.ids[second], part.ids[first]))
            std::swap(first, second);
        best = FindPrefillVictim(part, first, best, base, active);
        return FindPrefillVictim(part, second, best, base, active);
    }

    void ApplyAdmission(Admission admission) {
        const int key = admission.key, slot = admission.slot;
        if (key < 0 || key >= int(keys.size()) || slot < 0 || slot >= int(slots.size()) ||
            keys[key].slot >= 0 || keyPartitions[key] != slotPartitions[slot])
            throw std::invalid_argument("invalid MoE admission");
        const int old = slots[slot];
        RemoveCandidate(key);
        RemoveResident(slot);
        if (old >= 0) {
            keys[old].slot = -1;
            InsertCandidate(old);
        }
        slots[slot] = key;
        keys[key].slot = slot;
        keys[key].admitted = steps;
        InsertResident(slot);
    }

    void ScaleScores(float scale) {
        for (auto &key : keys) {
            key.score *= scale;
            if (key.score < 1e-6f) key.score = 0;
        }
        for (auto &p : candidates) p.RepairTies(candidatePositions,
            [this](int a, int b) { return CandidateBefore(a, b); });
        for (auto &p : residents) p.RepairTies(residentPositions,
            [this](int a, int b) { return ResidentBefore(a, b); });
    }

    struct Key { float score = 0; uint64_t admitted = 0; int slot = -1; };
    MoeCacheConfig config;
    uint64_t steps = 0, decodeSteps = 0;
    bool prefill = false, active = false;
    std::vector<Key> keys;
    std::vector<int> slots, keyPartitions, slotPartitions;
    struct Group { int partition; uint64_t bytes; };
    std::vector<Group> groups;
    std::vector<int> keyGroups;
    std::vector<Heap> residents, candidates;
    std::vector<int> candidatePositions, residentPositions;
    std::vector<int> deferredCandidates, deferredResidents;
    int layers;
};
} // namespace fastllm
