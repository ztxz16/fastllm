#include "utils/dense_id_set.h"
#include "utils/paged_prefix_preview.h"
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<uint64_t> Hashes(const std::vector<int> &tokens, int page) {
    std::vector<uint64_t> out;
    for (size_t i=0;i+page<=tokens.size();i+=page)
        out.push_back(PagedCacheManager::HashTokenPage(tokens.data()+i,page));
    return out;
}
static void Publish(PagedCacheManager &p, const std::vector<uint64_t> &hashes) {
    std::vector<int> pages;
    for (size_t i=0;i<hashes.size();++i) pages.push_back(p.GetUnusedPageIndex(true));
    p.RecordPrefix(hashes,pages);p.ReleasePageIndices(pages);
}
static void CheckDenseSet() {
    // A small epoch exercises wraparound as well as reused and resized storage.
    DenseIdSet<uint8_t> dense;
    std::mt19937 random(731);
    for (int round=0;round<800;++round) {
        int capacity=1+random()%2048;
        dense.Reset(capacity);
        std::unordered_set<int> reference;
        for (int i=0;i<4096;++i) {
            int id=random()%capacity;
            Require(dense.Contains(id)==(reference.count(id)!=0),"dense membership/reset mismatch");
            dense.Insert(id);reference.insert(id);
        }
    }
}
static void CheckPreview() {
    constexpr int n=16,page=128;
    std::vector<int> tokens(640*page+73);
    std::iota(tokens.begin(),tokens.end(),1);
    auto hashes=Hashes(tokens,page);
    std::vector<std::unique_ptr<PagedCacheManager>> owned;
    std::vector<PagedCacheManager*> managers;
    for(int i=0;i<n;++i) {
        auto p=std::make_unique<PagedCacheManager>();p->pageLen=page;
        p->type=PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE;
        p->SetMaxPages(1024);auto prefix=hashes;prefix.resize(640-i);
        Publish(*p,prefix);managers.push_back(p.get());owned.push_back(std::move(p));
    }
    PagedPrefixPreviewExecutor executor;
    std::map<PagedCacheManager*,PagedPrefixMatch> matches;
    auto check=[&](int threads,bool cold,bool trace) {
        if(cold)matches.clear();
        inference_trace::HostDetail detail;
        auto *before=inference_trace::CurrentDetail();
        if(trace)inference_trace::CurrentDetail()=&detail;
        int first=managers[0]->PreviewPrefix(hashes,matches[managers[0]]);
        int got=executor.Query(managers,hashes,matches,first,threads),expected=first;
        Require(inference_trace::CurrentDetail()==(trace?&detail:before),"worker leaked trace TLS");
        inference_trace::CurrentDetail()=before;
        for(auto *p:managers) {
            std::vector<int> original,planned;
            p->Query(tokens,original);
            p->QueryPrefix(hashes,matches[p],planned);
            Require(original==planned,"parallel preview changed restored physical pages");
            expected=std::min(expected,(int)original.size());
            Require(p->FreePageCount()==p->maxPages,"preview retained page references");
        }
        Require(got==expected,"parallel prefix intersection mismatch");
        if(trace)Require(detail.dropped==0,"parallel trace overflow");
        if(trace&&cold&&threads>1) {
            bool workers=false;
            for(size_t i=0;i<detail.used;++i)
                workers |= std::string(detail.metrics[i].region)=="prefill_preview_workers_sum";
            Require(workers,"parallel fixture did not run workers");
        }
    };
    for(int threads:{1,4,8,2,16,1}) {
        check(threads,true,true);check(threads,false,true);check(threads,false,false);
    }
    // Rebind an ancestor without changing the endpoint. The path must resolve
    // to the new physical pages, not a cached page list from another generation.
    auto shortened=hashes;shortened.resize(64);Publish(*managers[3],shortened);
    check(8,false,false);check(8,true,true);
    // Recycle the full pool and then publish different contents. Old anchors
    // must not revive them, whether the work estimate picks serial or parallel.
    managers[7]->SetMaxPages(1024);
    auto different=hashes;different[0]^=0x1234;Publish(*managers[7],different);
    check(8,false,false);check(8,true,true);
    managers[7]->SetMaxPages(1024);Publish(*managers[7],hashes);
    check(8,true,true);
    // Miss in the probe skips all other managers, as before.
    auto before=matches;auto miss=hashes;miss[0]^=0x9876;
    int got=executor.Query(managers,miss,matches,0,8);
    Require(got==0&&matches.size()==before.size(),"zero probe did not short circuit");
    for(auto &p:owned){p->EvictTrieSubtree(p->trieRoot);p->trieRoot=nullptr;}
}
int main() {
    try {SetThreads(1);CheckDenseSet();CheckPreview();
        std::cout<<"PAGED PREFIX PREVIEW PASS"<<std::endl;
    } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
