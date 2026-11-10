#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace fastkv {
struct PrefillRouter::State {
    mutable std::mutex mutex;
    std::vector<size_t> pending;
    std::vector<std::chrono::steady_clock::time_point> overloaded;
    size_t next = 0;
    explicit State(size_t peers) : pending(peers), overloaded(peers) {}
};
PrefillRouter::PrefillRouter(size_t peers) : state(std::make_shared<State>(peers)) {
    Require(peers > 0, "FastKV: routing needs at least one peer");
}
PrefillRouter::Route PrefillRouter::Reserve(size_t tokens, const std::vector<Estimate> &estimates,
                                           size_t preferred, bool owned) {
    Require(tokens > 0 && estimates.size() == state->pending.size(), "FastKV: invalid routing estimates");
    struct Lease {
        std::shared_ptr<State> state;
        size_t peer = 0, tokens = 0;
        ~Lease() {
            if (!state) return;
            std::lock_guard<std::mutex> lock(state->mutex);
            state->pending[peer] -= tokens;
        }
    };
    auto lease = std::make_shared<Lease>();
    std::lock_guard<std::mutex> lock(state->mutex);
    auto useHost = [](const Estimate &e) {
        return e.hostCachedTokens > e.cachedTokens && e.tokensPerSecond > 0 &&
            e.restoreSeconds < double(e.hostCachedTokens - e.cachedTokens) / e.tokensPerSecond;
    };
    auto cached = [&](const Estimate &e) { return useHost(e) ? e.hostCachedTokens : e.cachedTokens; };
    size_t bestMiss = tokens;
    for (const auto &e : estimates) {
        if (e.available && std::isfinite(e.tokensPerSecond) && e.tokensPerSecond > 0)
            bestMiss = std::min(bestMiss, std::max(size_t(1), tokens - std::min(tokens, cached(e))));
    }
    size_t best = estimates.size(), cost = 0;
    double score = std::numeric_limits<double>::infinity();
    double preferredScore = std::numeric_limits<double>::infinity();
    size_t preferredCost = 0;
    for (size_t i = 0; i < estimates.size(); ++i) {
        const size_t peer = (state->next + i) % estimates.size();
        const auto &e = estimates[peer];
        if (!e.available || !std::isfinite(e.tokensPerSecond) || e.tokensPerSecond <= 0 ||
            !std::isfinite(e.restoreSeconds) || e.restoreSeconds < 0 ||
            !std::isfinite(e.pendingRestoreSeconds) || e.pendingRestoreSeconds < 0) continue;
        const size_t miss = std::max(size_t(1), tokens - std::min(tokens, cached(e)));
        // Local leases cover bursts not yet visible in remote queue telemetry.
        // max avoids counting the same submitted work twice.
        // A small queue advantage must not repeatedly duplicate a resident
        // prefix across producers: that work delays other requests and consumes
        // their cache capacity. Charge the extra recomputation once beyond this
        // request's ETA. A sufficiently overloaded warm producer can still spill.
        const double seconds = (double(std::max(e.pendingTokens, state->pending[peer])) + miss + (miss - bestMiss)) /
                               e.tokensPerSecond + (useHost(e) ? e.pendingRestoreSeconds + e.restoreSeconds : 0);
        if (peer == preferred) { preferredScore = seconds; preferredCost = miss; }
        if (seconds < score) { best = peer; cost = miss; score = seconds; }
    }
    if (best == estimates.size()) throw CapacityBusy("FastKV: no available prefill peer");
    // A modest queue difference should not duplicate a long session prefix.
    // Do not prefer a cold owner over a substantially warmer peer or prevent
    // spilling when the owner is overloaded/unavailable.
    if (owned && preferred < estimates.size() && estimates[preferred].available &&
        preferredCost > 0 && std::isfinite(preferredScore)) {
        // Rebuilding/duplicating a whole session costs more than a transient
        // batch's queue imbalance. Require a sustained advantage before moving
        // an established owner; failures can still spill immediately.
        auto &since = state->overloaded[preferred];
        const auto now = std::chrono::steady_clock::now();
        const double rebuild = double(tokens) / estimates[preferred].tokensPerSecond;
        // Migration consumes a full-prefix rebuild and displaces resident KV
        // for future turns. Its benefit must cover both costs, and remain
        // visible for at least one rebuild rather than a single batch wave.
        const double margin = std::max(2.0, 2.0 * rebuild);
        const bool lostPrefix = std::max(estimates[preferred].cachedTokens, estimates[preferred].hostCachedTokens) < tokens / 4 && bestMiss < tokens / 4;
        if (lostPrefix || preferredScore <= score + margin) since = {};
        else if (since == std::chrono::steady_clock::time_point{}) since = now;
        if (!lostPrefix && (since == std::chrono::steady_clock::time_point{} ||
            now - since < std::chrono::duration<double>(std::max(1.0, rebuild)))) {
            best = preferred; cost = preferredCost;
        }
    } else if (preferred < estimates.size() && estimates[preferred].available &&
        preferredCost > 0 && std::isfinite(preferredScore) &&
        preferredCost <= bestMiss + std::max(size_t(128), tokens / 100) &&
        preferredScore <= score + double(std::max(bestMiss / 4, tokens / 64)) / estimates[preferred].tokensPerSecond) {
        best = preferred; cost = preferredCost;
    }
    Require(cost <= SIZE_MAX - state->pending[best], "FastKV: routing work overflow");
    state->pending[best] += cost;
    state->next = (best + 1) % estimates.size();
    lease->state = state; lease->peer = best; lease->tokens = cost;
    return {best, cost, std::move(lease)};
}
std::vector<size_t> PrefillRouter::Pending() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->pending;
}

struct SessionDirectory::Impl {
    struct Entry { Record record; uint64_t used = 0; };
    mutable std::mutex mutex;
    mutable std::map<std::string, Entry> entries;
    mutable uint64_t clock = 0;
    size_t capacity;
    explicit Impl(size_t capacity) : capacity(capacity) {}
};
SessionDirectory::SessionDirectory(size_t capacity) : impl(new Impl(capacity)) {
    Require(capacity > 0, "FastKV: empty session directory capacity");
}
SessionDirectory::~SessionDirectory() = default;
bool SessionDirectory::Newer(const Record &a, const Record &b) {
    return a.version > b.version || (a.version == b.version && a.owner > b.owner);
}
SessionDirectory::Record SessionDirectory::Get(const std::string &key) const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto it = impl->entries.find(key);
    if (it == impl->entries.end()) return {};
    it->second.used = ++impl->clock;
    return it->second.record;
}
bool SessionDirectory::Update(const std::string &key, const Record &record) {
    Require(!key.empty() && key.size() <= 512 && record.version && !record.owner.empty() && record.owner.size() <= 512,
            "FastKV: invalid session placement");
    std::lock_guard<std::mutex> lock(impl->mutex);
    auto it = impl->entries.find(key);
    if (it != impl->entries.end() && !Newer(record, it->second.record)) return false;
    if (it == impl->entries.end() && impl->entries.size() >= impl->capacity) {
        auto oldest = std::min_element(impl->entries.begin(), impl->entries.end(),
            [](const auto &a, const auto &b) { return a.second.used < b.second.used; });
        impl->entries.erase(oldest);
    }
    impl->entries[key] = {record, ++impl->clock};
    return true;
}
} // namespace fastkv
