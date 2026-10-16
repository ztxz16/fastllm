#include "fastllm.h"
#include <numeric>
#include <stdexcept>

using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<int> Tokens(int first, int count = 1024) {
    std::vector<int> tokens(count);
    std::iota(tokens.begin(), tokens.end(), first);
    return tokens;
}

static std::vector<int> Allocate(PagedCacheManager &pool, int count) {
    std::vector<int> pages;
    for (int i = 0; i < count; ++i) pages.push_back(pool.GetUnusedPageIndex(true));
    return pages;
}

static int Match(PagedCacheManager &pool, const std::vector<int> &tokens) {
    std::vector<int> pages;
    pool.Query(tokens, pages);
    return (int)pages.size() * pool.pageLen;
}

static std::vector<int> Publish(PagedCacheManager &pool, const std::vector<int> &tokens) {
    auto pages = Allocate(pool, tokens.size() / pool.pageLen);
    pool.Record(tokens, pages);
    pool.ReleasePageIndices(pages);
    return pages;
}

static void CheckPressure(bool refreshOlder) {
    PagedCacheManager pool;
    pool.pageLen = 128;
    pool.SetMaxPages(16);
    auto a = Tokens(1), b = Tokens(2000);
    auto pa = Publish(pool, a), pb = Publish(pool, b);
    const auto &hot = refreshOlder ? a : b;
    const auto &cold = refreshOlder ? b : a;
    Require(Match(pool, hot) == 1024, "initial hit");
    auto fresh = Allocate(pool, 3);
    Require(Match(pool, hot) == 1024, "hot prefix lost to colder cache");
    Require(Match(pool, cold) == 640, "eviction did not trim cold leaves");
    pool.ReleasePageIndices(fresh);
    pool.EvictTrieSubtree(pool.trieRoot); pool.trieRoot = nullptr;
}

static void CheckPreviewAndReferences() {
    PagedCacheManager pool;
    pool.pageLen = 128; pool.SetMaxPages(16);
    auto a = Tokens(1), b = Tokens(2000);
    auto pa = Publish(pool, a), pb = Publish(pool, b);
    pool.Pick(pa);
    // Referenced pages survive allocation, even if their data is older.
    std::vector<int> preview;
    pool.PreviewUnusedPageIndices(5, preview);
    auto actual = Allocate(pool, 5);
    Require(preview == actual, "batch preview differs from allocation");
    Require(Match(pool, a) == 1024, "active prefix evicted");
    Require(Match(pool, b) == 384, "incorrect pressure accounting");
    pool.ReleasePageIndices(pa); pool.ReleasePageIndices(actual);
    Require(pool.FreePageCount() == 16, "page references leaked");
    int peek = pool.GetUnusedPageIndex(false);
    Require(pool.GetUnusedPageIndex(true) == peek, "single preview differs");
    pool.ReleasePageIndex(peek);
    pool.EvictTrieSubtree(pool.trieRoot); pool.trieRoot = nullptr;
}

static void CheckAnchorAndFork() {
    PagedCacheManager pool;
    pool.pageLen = 128; pool.SetMaxPages(12);
    auto tokens = Tokens(1);
    auto pages = Publish(pool, tokens);
    int endpoint = pages.back();
    long long generation = pool.pageTimestamp[endpoint];
    Require(pool.IsCachedPage(endpoint, generation), "published anchor missing");
    pool.Pick(pages); pool.Record(tokens, pages); pool.ReleasePageIndices(pages);
    Require(pool.IsCachedPage(endpoint, generation), "republish invalidated shared anchor");
    auto fork = tokens; fork.resize(512);
    auto shared = std::vector<int>(pages.begin(), pages.begin() + 4);
    pool.Pick(shared);
    auto tail = Allocate(pool, 4);
    shared.insert(shared.end(), tail.begin(), tail.end());
    auto suffix = Tokens(8000, 512); fork.insert(fork.end(), suffix.begin(), suffix.end());
    pool.Record(fork, shared); pool.ReleasePageIndices(shared);
    Require(Match(pool, tokens) == 1024 && Match(pool, fork) == 1024, "fork damaged shared prefix");
    auto pressure = Allocate(pool, 12);
    Require(!pool.IsCachedPage(endpoint, generation), "stale anchor survived page reuse");
    pool.ReleasePageIndices(pressure);
    pool.SetMaxPages(12);
    auto other = Publish(pool, Tokens(40000));
    Require(!pool.IsCachedPage(endpoint, generation), "pool reset revived an old anchor");
    pool.EvictTrieSubtree(pool.trieRoot); pool.trieRoot = nullptr;
}

static void CheckRepurposedPages() {
    PagedCacheManager pool;
    pool.pageLen = 128; pool.SetMaxPages(8);
    auto old = Tokens(1), replacement = Tokens(2000);
    auto pages = Publish(pool, old);
    long long generation = pool.pageTimestamp[pages.front()];
    long long tailGeneration = pool.pageTimestamp[pages.back()];
    pool.Pick(pages);
    pool.Record(replacement, pages);
    pool.ReleasePageIndices(pages);
    Require(Match(pool, old) == 0, "repurposed page retained old prefix");
    Require(Match(pool, replacement) == 1024, "replacement prefix missing");
    Require(!pool.IsCachedPage(pages.front(), generation), "old generation survived content change");
    Require(!pool.IsCachedPage(pages.back(), tailGeneration), "old suffix generation survived prefix change");
    Require(pool.FreePageCount() == 8, "replacement leaked pages");
    pool.EvictTrieSubtree(pool.trieRoot); pool.trieRoot = nullptr;
}

int main() {
    try {
        CheckPressure(false); CheckPressure(true);
        CheckPreviewAndReferences(); CheckAnchorAndFork();
        CheckRepurposedPages();
        std::cout << "PAGED CACHE LRU PASS" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
