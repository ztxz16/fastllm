#include "internal.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace fastkv {
struct PrefillRouter::State {
    mutable std::mutex mutex;
    std::vector<size_t> pending;
    size_t next = 0;
    explicit State(size_t peers) : pending(peers) {}
};
PrefillRouter::PrefillRouter(size_t peers) : state(std::make_shared<State>(peers)) {
    Require(peers > 0, "FastKV: routing needs at least one peer");
}
size_t PrefillRouter::SessionPeer(const std::string &key, const std::vector<std::string> &peers) {
    if (key.empty()) return SIZE_MAX;
    uint64_t highest = 0;
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < peers.size(); ++i) {
        uint64_t hash = 14695981039346656037ULL;
        for (unsigned char c : peers[i])
            hash = (hash ^ c) * 1099511628211ULL;
        hash *= 1099511628211ULL; // NUL separator between endpoint and session.
        for (unsigned char c : key)
            hash = (hash ^ c) * 1099511628211ULL;
        // MurmurHash3 fmix64 separates neighbouring ports and sequential session IDs.
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 33;
        hash *= 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> 33;
        if (best == SIZE_MAX || hash > highest) { highest = hash; best = i; }
    }
    return best;
}

bool PrefillRouter::PreferHostPrefix(size_t residentTokens, size_t hostTokens, size_t pageTokens,
                                     double tokensPerSecond, double restoreSeconds) {
    if (hostTokens <= residentTokens) return false;
    const size_t gain = hostTokens - residentTokens;
    // A partial resident page can be recomputed with the suffix. Restoring
    // the whole checkpoint also waits for the model thread to import it.
    return (!residentTokens || gain >= pageTokens) && tokensPerSecond > 0 &&
           restoreSeconds < double(gain) / tokensPerSecond;
}

PrefillRouter::Route PrefillRouter::Reserve(size_t tokens, const std::vector<Estimate> &estimates,
                                           size_t preferred, bool pinPreferred) {
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
    auto cached = [](const Estimate &e) {
        return PreferHostPrefix(e.cachedTokens, e.hostCachedTokens, 1, e.tokensPerSecond, e.restoreSeconds)
            ? e.hostCachedTokens : e.cachedTokens;
    };
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
        const size_t prefix = cached(e);
        const size_t miss = std::max(size_t(1), tokens - std::min(tokens, prefix));
        // Local leases cover bursts not yet visible in remote queue telemetry.
        // max avoids counting the same submitted work twice.
        // A small queue advantage must not repeatedly duplicate a resident
        // prefix across producers: that work delays other requests and consumes
        // their cache capacity. Charge the extra recomputation once beyond this
        // request's ETA. A sufficiently overloaded warm producer can still spill.
        const double seconds = (double(std::max(e.pendingTokens, state->pending[peer])) + miss + (miss - bestMiss)) /
                               e.tokensPerSecond + (prefix > e.cachedTokens ? e.pendingRestoreSeconds + e.restoreSeconds : 0);
        if (peer == preferred) { preferredScore = seconds; preferredCost = miss; }
        if (seconds < score) { best = peer; cost = miss; score = seconds; }
    }
    if (best == estimates.size()) throw CapacityBusy("FastKV: no available prefill peer");
    // A live session keeps its producer even when another producer is idle.
    // An unavailable or invalid estimate never populated preferredScore.
    if (std::isfinite(preferredScore)) {
        const double slack = double(std::max(bestMiss / 4, tokens / 64)) / estimates[preferred].tokensPerSecond;
        if (pinPreferred || (preferredCost <= bestMiss + std::max(size_t(128), tokens / 100) &&
                             preferredScore <= score + slack)) {
            best = preferred; cost = preferredCost;
        }
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
    struct Entry { Record record; uint64_t used = 0; int localGroup = -1; };
    mutable std::mutex mutex;
    mutable std::map<std::string, Entry> entries;
    mutable uint64_t clock = 0;
    uint64_t nextGroup = 0;
    size_t capacity;
    explicit Impl(size_t capacity) : capacity(capacity) {}
    Entry &Touch(const std::string &key) {
        auto it = entries.find(key);
        if (it == entries.end()) {
            if (entries.size() == capacity) {
                auto oldest = std::min_element(entries.begin(), entries.end(),
                    [](const auto &a, const auto &b) { return a.second.used < b.second.used; });
                entries.erase(oldest);
            }
            it = entries.emplace(key, Entry{}).first;
        }
        it->second.used = ++clock;
        return it->second;
    }
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
    impl->Touch(key).record = record;
    return true;
}
int SessionDirectory::LocalGroup(const std::string &key, int groups, bool assign) {
    if (key.empty() || groups <= 1) return -1;
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (!assign) {
        const auto it = impl->entries.find(key);
        return it == impl->entries.end() ? -1 : it->second.localGroup;
    }
    auto &entry = impl->Touch(key);
    if (entry.localGroup < 0) entry.localGroup = impl->nextGroup++ % groups;
    return entry.localGroup;
}
} // namespace fastkv
