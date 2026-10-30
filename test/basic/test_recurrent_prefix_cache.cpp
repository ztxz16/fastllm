#include "utils/recurrent_prefix_cache.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct Snapshot : RecurrentPrefixKey { bool draft = false; };
static void Check(bool ok) { if (!ok) throw std::runtime_error("prefix checkpoint invariant failed"); }

int main() {
    std::vector<std::unique_ptr<Snapshot>> items;
    auto add = [&](std::vector<int> tokens, long long owner, long long time, bool draft = false) {
        auto s = std::make_unique<Snapshot>();
        s->tokens = tokens; s->cachedLen = tokens.size();
        s->requestId = owner; s->timestamp = time; s->draft = draft;
        items.push_back(std::move(s));
    };
    auto find = [&](std::vector<int> tokens, int limit, int exact = -1, bool draft = false) {
        return FindRecurrentPrefix(items, tokens, limit, exact,
            [&](const Snapshot &s) { return !draft || s.draft; });
    };
    add({1, 2}, 1, 1, true);
    add({1, 2, 3, 4}, 1, 2);
    Check(find({1, 2, 3, 4, 5}, 5)->cachedLen == 4);
    Check(find({1, 2, 3, 9}, 4)->cachedLen == 2); // Never slice the four-token state.
    Check(find({1, 2, 3, 4}, 4, -1, true)->cachedLen == 2);
    Check(!find({1, 2, 3, 4}, 3, 3));
    Check(!find({8, 2, 3, 4}, 4));
    Check(!find({1}, 4));
    add({1, 2, 3, 4}, 2, 3);
    Check(find({1, 2, 3, 4}, 4)->timestamp == 3);
    items.back()->cachedLen = 3; // Incomplete metadata must fail closed.
    Check(find({1, 2, 3, 4}, 4)->timestamp == 2);
    items.pop_back();
    add({5, 6}, 1, 3);
    add({7, 8}, 2, 4);
    TrimRecurrentPrefixes(items, 1, 2, 3);
    Check(items.size() == 3 && !find({1, 2}, 2));
    find({1, 2, 3, 4}, 4)->timestamp = 5;
    add({9}, 3, 6);
    TrimRecurrentPrefixes(items, 3, 2, 3);
    Check(items.size() == 3 && !find({5, 6}, 2) && find({1, 2, 3, 4}, 4));
    setenv("FASTLLM_PREFIX_CACHE", "FALSE", 1);
    Check(!RecurrentPrefixCacheEnabled());
    unsetenv("FASTLLM_PREFIX_CACHE");
    Check(RecurrentPrefixCacheEnabled());
    setenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", "16", 1);
    Check(RecurrentPrefixInterval(16) == 256);
    setenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", "999999999999999999", 1);
    Check(RecurrentPrefixInterval(256) > 0);
    std::cout << "PASS recurrent prefix matching, compatibility and eviction" << std::endl;
}
