#include "models/basellm.h"
#include "utils/paged_cache_batch.h"
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}

struct Fixture {
    std::vector<std::unique_ptr<PagedCacheManager>> pools;
    std::vector<std::unique_ptr<Data>> caches;
    std::vector<Data *> leaves;
    std::vector<std::vector<uint64_t>> hashes;
    std::vector<PagedCacheBatchExecutor::Publication> jobs;
    std::vector<PagedCacheBatchExecutor::BorrowedPublication> borrowed;
    Fixture(int n, bool borrow = false) {
        hashes.resize(3);hashes[0].resize(640);
        std::iota(hashes[0].begin(),hashes[0].end(),11);
        hashes[1]=hashes[0];hashes[1].resize(128);
        hashes[2]=hashes[0];hashes[2][33]^=12345;
        for(int m=0;m<n;++m) {
            auto p=std::make_unique<PagedCacheManager>();p->pageLen=128;
            p->type=PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;p->SetMaxPages(2048);
            pools.push_back(std::move(p));
        }
        // Interleave managers, with multiple ordered operations per manager:
        // full path -> physical ancestor replacement -> divergent suffix.
        for(auto &h:hashes)for(auto &p:pools) {
            auto d=std::make_unique<Data>();d->isPagedKVCache=true;d->pagedKVCacheData=p.get();
            for(size_t i=0;i<h.size();++i)d->pageIndex.push_back(p->GetUnusedPageIndex(true));
            // The final allocated page is partial and must never be published.
            d->pageLen=p->pageLen; d->lastPageLen=17;
            if (borrow) borrowed.push_back({p.get(),&h,d.get()});
            else {
                auto complete=d->pageIndex;complete.pop_back();
                jobs.push_back({p.get(),&h,std::move(complete)});
            }
            leaves.push_back(d.get());caches.push_back(std::move(d));
        }
    }
    ~Fixture(){caches.clear();for(auto &p:pools){p->EvictTrieSubtree(p->trieRoot);p->trieRoot=nullptr;}}
};
static void Equal(const Fixture &a,const Fixture &b) {
    for(size_t i=0;i<a.pools.size();++i) {
        auto &x=*a.pools[i],&y=*b.pools[i];
        Require(x.pageRefCount==y.pageRefCount,"page reference mismatch");
        Require(x.pageTimestamp==y.pageTimestamp,"page generation mismatch");
        Require(x.freePages==y.freePages&&x.triePages==y.triePages,"unused page order changed");
        Require(x.FreePageCount()==y.FreePageCount(),"capacity mismatch");
        for(int page=0;page<x.maxPages;++page) {
            auto *u=x.pageToTrieNode[page],*v=y.pageToTrieNode[page];
            Require(bool(u)==bool(v),"page binding presence differs");
            if(!u)continue;
            Require(u->pageId==v->pageId&&u->timestamp==v->timestamp&&u->lastAccess==v->lastAccess&&
                u->nodeId==v->nodeId&&u->edgeHash==v->edgeHash&&u->unusedIndex==v->unusedIndex&&
                u->parent->pageId==v->parent->pageId,"trie identity/LRU differs");
        }
    }
}
static void Case(int managers,int threads,bool diagnostics) {
    Fixture a(managers),b(managers,true);PagedCacheBatchExecutor serial,parallel;
    inference_trace::HostDetail detail;
    serial.Publish(a.jobs,1);
    if(diagnostics)inference_trace::CurrentDetail()=&detail;
    parallel.PublishBorrowed(b.borrowed,threads);
    Require(inference_trace::CurrentDetail()==(diagnostics?&detail:nullptr),"trace TLS changed");
    inference_trace::CurrentDetail()=nullptr;
    Equal(a,b);
    // Repeat publication, including a now-stale physical ancestor path.
    serial.Publish(a.jobs,1);parallel.PublishBorrowed(b.borrowed,threads);Equal(a,b);
    serial.ReleaseReferences(a.leaves,1);
    if(diagnostics)inference_trace::CurrentDetail()=&detail;
    parallel.ReleaseReferences(b.leaves,threads);
    inference_trace::CurrentDetail()=nullptr;
    Require(detail.dropped==0,"diagnostic overflow");
    Equal(a,b);
    for(size_t i=0;i<a.pools.size();++i) {
        Require(a.pools[i]->FreePageCount()==a.pools[i]->maxPages,"references leaked");
        for(auto &h:a.hashes) {
            PagedPrefixMatch ma,mb;std::vector<int> pa,pb;
            a.pools[i]->QueryPrefix(h,ma,pa);b.pools[i]->QueryPrefix(h,mb,pb);
            Require(pa==pb,"restore physical pages changed");
        }
        // Force eviction/recycling. Selection sequence must preserve LRU, not
        // merely the count of free pages.
        std::vector<int> allocated;
        for (int end : {1, 1, 49, 1500}) {
            const size_t first = allocated.size();
            b.pools[i]->ExtendPageIndices(allocated, end);
            for (size_t j = first; j < allocated.size(); ++j) {
                const int page = a.pools[i]->GetUnusedPageIndex(true);
                Require(page == allocated[j], "batched eviction order changed");
                Require(a.pools[i]->pageTimestamp[page] == b.pools[i]->pageTimestamp[page],
                        "batched page generation changed");
            }
        }
    }
    a.caches.clear();b.caches.clear(); // Previously released Data must not release twice.
    Equal(a,b);
}
static void LookaheadPages() {
    PagedCacheManager pool;
    pool.pageLen = 4; pool.SetMaxPages(8);
    std::vector<int> active;
    Require(pool.TryExtendFreePages(active, 3, 5), "free suffix unavailable");
    pool.RecordPrefix({1,2,3}, active);
    const auto generation = pool.pageTimestamp[active.back()];
    Require(pool.MatchesPinnedPrefix(active, 3, generation), "fresh binding rejected");
    std::vector<int> spare;
    Require(!pool.TryExtendFreePages(spare, 2, 4) && spare.empty(), "lookahead consumed headroom");
    const int replacement = pool.GetUnusedPageIndex(true);
    pool.RecordPrefix({1}, {replacement});
    Require(!pool.MatchesPinnedPrefix(active, 3, generation), "rebound ancestor accepted");
    auto rebound = active; rebound.front() = replacement;
    Require(pool.MatchesPinnedPrefix(rebound, 3, generation) &&
        !pool.MatchesPinnedPrefix(rebound, 3, generation + 1), "page generation check failed");
    // The prefix pins remain with the request when the next forward consumes
    // fewer tokens than predicted. Only the unused lookahead suffix is freed.
    Data cache;
    cache.pageLen = pool.pageLen;
    cache.pagedKVCacheData = &pool;
    cache.pageIndex.swap(active);
    PDPrefillPagePlan plan;
    plan.tables.push_back({&cache, generation});
    Require(!plan.Valid(12), "page plan ignored a rebound ancestor");
    Require(pool.TryExtendFreePages(cache.pageIndex, 6, 0), "could not stage the test suffix");
    const int unused = cache.pageIndex.back();
    plan.Trim(17);
    Require(cache.pageIndex.size() == 5 && plan.tables.empty() && pool.pageRefCount[unused] == 0 &&
            pool.pageRefCount[cache.pageIndex.back()] == 1, "shorter batch leaked or freed a live page");
    pool.ReleasePageIndices(cache.pageIndex);
    cache.pageIndex.clear();
    pool.ReleasePageIndex(replacement);
    const auto before = pool.CacheStats();
    Require(!pool.TryExtendFreePages(spare, 8, 0) && pool.CacheStats().pressureEvictedPages == before.pressureEvictedPages,
        "lookahead evicted a cached page");
}

int main() {
    try {
        SetThreads(1);
        LookaheadPages();
        for(int threads:{1,2,4,8,16})Case(16,threads,true);
        Case(1,8,false);Case(4,8,false);
        // Empty/small work retains the serial path without creating workers.
        PagedCacheBatchExecutor e;e.Publish({},8);e.ReleaseReferences({},8);
        std::cout<<"PAGED CACHE BATCH PASS\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
