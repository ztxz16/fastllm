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
    Fixture(int n) {
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
            jobs.push_back({p.get(),&h,d->pageIndex});
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
    Fixture a(managers),b(managers);PagedCacheBatchExecutor serial,parallel;
    inference_trace::HostDetail detail;
    serial.Publish(a.jobs,1);
    if(diagnostics)inference_trace::CurrentDetail()=&detail;
    parallel.Publish(b.jobs,threads);
    Require(inference_trace::CurrentDetail()==(diagnostics?&detail:nullptr),"trace TLS changed");
    inference_trace::CurrentDetail()=nullptr;
    Equal(a,b);
    // Repeat publication, including a now-stale physical ancestor path.
    serial.Publish(a.jobs,1);parallel.Publish(b.jobs,threads);Equal(a,b);
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
        for(int j=0;j<1500;++j)
            Require(a.pools[i]->GetUnusedPageIndex(true)==b.pools[i]->GetUnusedPageIndex(true),"eviction order changed");
    }
    a.caches.clear();b.caches.clear(); // Previously released Data must not release twice.
    Equal(a,b);
}
int main() {
    try {
        SetThreads(1);
        for(int threads:{1,2,4,8,16})Case(16,threads,true);
        Case(1,8,false);Case(4,8,false);
        // Empty/small work retains the serial path without creating workers.
        PagedCacheBatchExecutor e;e.Publish({},8);e.ReleaseReferences({},8);
        std::cout<<"PAGED CACHE BATCH PASS\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
