#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace fastllm {

// Hybrid decode admission for registered expert caches. Configure before loading a model;
// each cache takes a snapshot, so changing this does not mutate a live policy.
struct MoeCacheConfig {
    float halfLife = 128;       // Decode steps; 0 disables frequency decay.
    int updateInterval = 1;
    int maxReplacements = 96;   // Includes filling empty slots; 0 disables admission.
    uint64_t maxBytes = 0;      // Bytes per update, 0 means no additional byte cap.
    float minHeat = 1;
    float replacementMargin = 1;
    float replacementFactor = 1;
    int minimumResidence = 0;
    float prefillPrior = 0;     // Rescaled prefill heat, not cache contents.
    bool rankByBytes = false;

    void Validate() const {
        if (!std::isfinite(halfLife) || halfLife < 0 || updateInterval < 1 ||
            maxReplacements < 0 || !std::isfinite(minHeat) || minHeat < 0 ||
            !std::isfinite(replacementMargin) || replacementMargin < 0 ||
            !std::isfinite(replacementFactor) || replacementFactor < 1 ||
            minimumResidence < 0 || !std::isfinite(prefillPrior) ||
            prefillPrior < 0 || prefillPrior > 1)
            throw std::invalid_argument("invalid MoE cache policy configuration");
    }
};

void SetMoeCacheConfig(const MoeCacheConfig &config);
MoeCacheConfig GetMoeCacheConfig();

} // namespace fastllm
