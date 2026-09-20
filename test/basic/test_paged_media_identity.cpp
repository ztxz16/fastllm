#ifdef NDEBUG
#undef NDEBUG
#endif
#include "fastllm.h"
#include "models/basellm.h"
#include "utils/multimodal_prefix_cache.h"
#include <cassert>
#include <iostream>

int main() {
    {
        fastllm::PagedCacheManager textCache;
        textCache.pageLen = 128;
        textCache.type = fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        textCache.SetMaxPages(3);
        {
            fastllm::ResponseContext ctx;
            ctx.Init(1,fastllm::DataType::FLOAT16,fastllm::DataType::FLOAT16);
            ctx.allTokens.assign(256,42);
            ctx.prefixCachePageKeys = {"text-chain-page0", "text-chain-page1"};
            auto &kv = ctx.pastKeyValues[0].first;
            kv.isPagedKVCache = true; kv.pagedKVCacheData = &textCache;
            kv.pageLen = 128; kv.lastPageLen = 128;
            kv.pageIndex = {textCache.GetUnusedPageIndex(true), textCache.GetUnusedPageIndex(true)};
            kv.Resize({1,256,1});
            ctx.TryRecordPagedCache(nullptr);
            std::vector<int> found;
            textCache.Query(ctx.allTokens,found,ctx.PrefixCachePageKeys());
            assert(found == kv.pageIndex);
            textCache.Query(ctx.allTokens,found);
            assert(found.empty());
            auto different = ctx.prefixCachePageKeys;
            different[0] += "-converted";
            textCache.Query(ctx.allTokens,found,&different);
            assert(found.empty());
        }
        textCache.EvictTrieSubtree(textCache.trieRoot);
        textCache.trieRoot = nullptr;
    }
    fastllm::PagedCacheManager cache;
    cache.pageLen = 128;
    cache.type = fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
    cache.SetMaxPages(5);
    std::vector<int> tokens(300, 3);
    std::vector<std::string> a = {"media-A-page0", "media-A-page1"};
    std::vector<std::string> b = {"media-B-page0", "media-B-page1"};
    std::vector<int> pages = {cache.GetUnusedPageIndex(true), cache.GetUnusedPageIndex(true)};
    cache.Record(tokens, pages, &a);
    std::vector<int> found;
    cache.Query(tokens, found, &a);
    assert(found == pages);
    cache.Query(tokens, found, &b);
    assert(found.empty());
    cache.Query(tokens, found);
    assert(found.empty()); // No token-only reuse of a multimodal page.
    std::vector<std::string> extended = a;
    extended.push_back("media-B-new-page");
    cache.Query(std::vector<int>(512, 3), found, &extended);
    assert(found == pages);
    auto partial = a;
    partial[1] = "different image starts here";
    cache.Query(tokens, found, &partial);
    assert(found == std::vector<int>({pages[0]}));
    // Force a bucket collision without relying on a probabilistic hash search.
    auto first = cache.trieRoot->children.begin()->second;
    std::string saved = first->extraKey;
    first->extraKey = "not-the-original-identity";
    cache.Query(tokens, found, &a);
    assert(found.empty());
    first->extraKey = saved;
    cache.ReleasePageIndices(pages);
    // Evict all pages: a stale checkpoint cannot retrieve the old identity.
    for (int i = 0; i < 5; ++i) cache.GetUnusedPageIndex(true);
    cache.Query(tokens, found, &a);
    assert(found.empty());
    cache.EvictTrieSubtree(cache.trieRoot);
    cache.trieRoot = nullptr;
    std::cout << "Paged media identity, collision, extension and eviction: PASS\n";
}
