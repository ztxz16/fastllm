#include "utils/recurrent_prefix_cache.h"
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct Snapshot : RecurrentPrefixKey { bool draft = false; };
static void Check(bool ok) { if (!ok) throw std::runtime_error("prefix checkpoint invariant failed"); }

static std::shared_ptr<Snapshot> Make(std::vector<int> tokens, long long owner, bool draft = false) {
    auto s = std::make_shared<Snapshot>();
    s->cachedLen = tokens.size(); s->tokens = std::move(tokens);
    s->requestId = owner; s->draft = draft;
    return s;
}

static void SharedOwnership() {
    RecurrentPrefixStore<Snapshot> store;
    using Entry = RecurrentPrefixStore<Snapshot>::Entry;
    auto find = [&](std::vector<int> tokens, bool draft = false) {
        return store.Find(tokens, tokens.size(), -1, [&](const Snapshot &s) { return !draft || s.draft; });
    };
    Entry first, second, third;
    Check(store.Publish(Make({1, 2}, 1), first, 4, 2));
    Check(store.Publish(Make({9, 8}, 2), second, 4, 2));
    Check(!store.Publish(Make({3, 4}, 3), third, 4, 2));
    Check(!third && store.Size() == 2 && find({1, 2}) == first && find({9, 8}) == second);
    auto fork = first;
    Check(!store.Publish(Make({1, 2, 3}, 4), first, 4, 2));
    Check(first == fork && first->tokens == std::vector<int>({1, 2}));
    fork.reset();
    Check(store.Publish(Make({1, 2, 3}, 4), first, 4, 2));
    Check(first->cachedLen == 3 && find({9, 8}) == second);
    fork = first;
    Check(store.Publish(Make({1, 2, 3}, 5, true), first, 4, 2));
    Check(find({1, 2, 3}, true) == first && !fork->draft && fork != first);
    Check(!find({1, 2, 7})); // A recurrent checkpoint cannot be sliced.
    second.reset();
    Check(store.Publish(Make({3, 4}, 6), third, 4, 2));
    Check(!find({9, 8}) && find({1, 2, 3}) == first);
}

static void RollingConversations() {
    for (int count : {2, 13, 64, 400}) {
        RecurrentPrefixStore<Snapshot> store;
        using Entry = RecurrentPrefixStore<Snapshot>::Entry;
        std::vector<std::vector<int>> prompts(count);
        long long owner = 0;
        auto find = [&](const std::vector<int> &tokens) {
            return store.Find(tokens, tokens.size(), -1, [](const Snapshot &) { return true; });
        };
        for (int i = 0; i < count; ++i) {
            // Share a system prefix but diverge before the first checkpoint.
            prompts[i] = {1, 2, 100 + i};
            Entry held;
            ++owner;
            for (int chunk = 0; chunk < 4; ++chunk) {
                prompts[i].push_back(7);
                Check(store.Publish(Make(prompts[i], owner), held, 4, count * 4));
            }
        }
        // A fast conversation advances repeatedly while others wait on tools.
        for (int round = 0; round < 16; ++round) {
            auto held = find(prompts.back());
            Check(held && held->cachedLen == int(prompts.back().size()));
            ++owner;
            for (int chunk = 0; chunk < 4; ++chunk) {
                prompts.back().push_back(8);
                Check(store.Publish(Make(prompts.back(), owner), held, 4, count * 4));
            }
            for (int i = 0; i + 1 < count; ++i) {
                auto idle = find(prompts[i]);
                Check(idle && idle->cachedLen == int(prompts[i].size()));
            }
            Check(store.Size() <= size_t(count * 4));
        }
        // Returning requests acquire independent holds before any one advances.
        std::vector<Entry> active;
        for (const auto &prompt : prompts) active.push_back(find(prompt));
        for (int i = 0; i < count; ++i) {
            prompts[i].push_back(11);
            Check(store.Publish(Make(prompts[i], ++owner), active[i], 4, count * 4));
            Check(active[i]->cachedLen == int(prompts[i].size()));
        }
        std::cout << "rolling conversations=" << count << " rounds=16 idle_hits=" << count - 1 << "\n";
    }
}

static void RetainSpareHistory() {
    RecurrentPrefixStore<Snapshot> store;
    RecurrentPrefixStore<Snapshot>::Entry held;
    std::vector<int> tokens{1};
    for (int request = 1; request <= 3; ++request) {
        for (int checkpoint = 0; checkpoint < 2; ++checkpoint) {
            tokens.push_back(2);
            Check(store.Publish(Make(tokens, request), held, 2, 16));
        }
    }
    Check(store.Size() == 6);
    Check(store.Find({1, 2}, 2, 2, [](const Snapshot &) { return true; }) != nullptr);
}

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
    unsetenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS");
    unsetenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST");
    Check(RecurrentPrefixMaxRecords() == 8 && RecurrentPrefixMaxRecords(13) == 52);
    setenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST", "6", 1);
    Check(RecurrentPrefixMaxRecords() == 8 && RecurrentPrefixMaxRecords(13) == 78);
    setenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", "3", 1);
    Check(RecurrentPrefixMaxRecords(13) == 3);
    SharedOwnership();
    RollingConversations();
    RetainSpareHistory();
    std::cout << "PASS recurrent prefix matching, compatibility and eviction" << std::endl;
}
