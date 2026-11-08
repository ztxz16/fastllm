
#include "utils/prefix_cache_page_ledger.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using namespace fastllm;
static void Require(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
namespace fastllm {
static int GetPageLen() { return 128; }
struct CacheTrieNode { std::shared_ptr<void> producerEvent; };
struct PagedCacheManager {
    std::mutex pageIndexLocker;
    int maxPages = 0;
    std::vector<long long> pageTimestamp;
    std::map<int, std::shared_ptr<CacheTrieNode>> pageToTrieNode;
    std::vector<int> visible;
    uint64_t queries = 0, scanned = 0, restoredHoles = 0, pageLocks = 0;
};
struct ResponseContext {
    uint64_t requestId = 0;
    std::vector<int> currentTokens;
    std::vector<std::string> keys;
    const std::vector<std::string> *PrefixCachePageKeys() const {
        return keys.empty() ? nullptr : &keys;
    }
};
using Qwen35SpillKey = std::tuple<uint64_t,int,int,int>;
struct Qwen35SpilledPage {
    PrefixCachePageIdentity ledgerIdentity;
    std::atomic<uint32_t> promotionRefs{0};
};
struct ProbeModel;
struct Qwen35SpillState {
    ProbeModel *ownerModel = nullptr;
    std::mutex promoteMutex, poolMutex;
    std::atomic<bool> directPublishActive{false};
    std::map<Qwen35SpillKey,std::shared_ptr<Qwen35SpilledPage>> pool;
    size_t bytes = 0;
};
static int rejectIndex = -1;
static int Qwen35InjectedL1RejectIndex() { return rejectIndex; }
static bool PrefixCacheDebugEnabled() { return false; }
template<typename... Args> static void PrefixCacheDebug(const char *, Args...) {}
// 生产发布函数现在输出 cache_publish_profile；T0 只需保证可编译，不影响断言。
template<typename... Args> static void FastllmEvent(const char *, uint64_t, const char *, const char *, Args...) {}
static std::vector<int> Qwen35QuerySpillL1Pages(PagedCacheManager *manager,
        const std::vector<int> &, const std::vector<std::string> *, int) {
    ++manager->queries;
    manager->scanned += manager->visible.size();
    return manager->visible;
}
static void Qwen35DropPromotedGpuPage(PagedCacheManager *manager, int page) {
    for (auto &value : manager->visible) if (value == page) value = -1;
    manager->pageToTrieNode.erase(page);
}
static void Qwen35RestoreSpillTrieHole(PagedCacheManager *manager,
        const std::vector<int> &, const std::vector<std::string> *, int) {
    ++manager->restoredHoles;
}
static bool Qwen35SpillPageUsable(const Qwen35SpilledPage &, PagedCacheManager *) {
    return true;
}
static bool Qwen35RemoveSpilledPageLocked(Qwen35SpillState &state,
        const Qwen35SpillKey &key, const std::shared_ptr<Qwen35SpilledPage> &expected) {
    auto it = state.pool.find(key);
    if (it == state.pool.end() || it->second != expected ||
        expected->promotionRefs.load() != 0) return false;
    state.pool.erase(it);
    return true;
}
struct ProbeModel {
    std::shared_ptr<Qwen35SpillState> pagedSpillState;
    std::shared_ptr<PrefixCachePageLedger> pagedPageLedger =
        std::make_shared<PrefixCachePageLedger>();
    std::vector<std::unique_ptr<PagedCacheManager>> managers;
    mutable uint64_t managerLookups = 0;
    std::vector<std::pair<int,PagedCacheManager*>> GetPagedKVCacheManagers(
            int layer, bool key) const {
        ++managerLookups;
        return {{0,managers.at(layer*2+(key?0:1)).get()}};
    }
    std::vector<PrefixCachePageIdentity> DescribePagedPrefixPages(
            const std::vector<int> &, const std::vector<std::string> &keys, int length) const {
        std::vector<PrefixCachePageIdentity> out;
        for (int p=0;p<length/GetPageLen();++p) {
            for (size_t m=0;m<managers.size();++m) {
                PrefixCachePageIdentity id;
                id.model="probe"; id.family="probe"; id.provenance="runtime";
                id.dtype="fp8"; id.layout="paged"; id.media=keys.at(p);
                id.parentHash=std::to_string(p); id.pageHash=std::to_string(p+1);
                id.pageOrdinal=p; id.layer=m/2; id.side=m%2?"value":"key"; id.rank=0;
                out.push_back(std::move(id));
            }
        }
        return out;
    }
    bool PublishDirectPagedPrefix(ResponseContext *,int,const std::function<bool()> &,
                                  const std::function<void()> &);
};
#include "publication_path.inc"

}
using namespace fastllm;
static void Case(int pages, int kind, bool baseline,int count=8) {
    ProbeModel model;
    ResponseContext ctx;
    const int overlap=kind==1?3:0;
    ctx.currentTokens.resize(pages*128,17);
    ctx.keys.resize(pages,std::string(2048,'x'));
    for (int m=0;m<count;++m) {
        auto manager=std::make_unique<PagedCacheManager>();
        manager->maxPages=pages+8;
        manager->pageTimestamp.resize(pages+8,1);
        manager->visible.resize(overlap);
        for (int p=0;p<overlap;++p) {
            manager->visible[p]=p;
            auto node=std::make_shared<CacheTrieNode>();
            node->producerEvent=std::make_shared<int>(1);
            manager->pageToTrieNode[p]=node;
        }
        model.managers.push_back(std::move(manager));
    }
    model.pagedSpillState=std::make_shared<Qwen35SpillState>();
    model.pagedSpillState->ownerModel=&model;
    auto identities=model.DescribePagedPrefixPages(ctx.currentTokens,ctx.keys,pages*128);
    std::shared_ptr<Qwen35SpilledPage> host;
    const bool hostCase=kind==2||kind==6||kind==7;
    if (hostCase) {
        const auto &id=identities.back();
        auto *manager=model.managers.back().get();
        Require(model.pagedPageLedger->ObserveL1(id,manager,0,1),"seed ledger");
        auto lease=model.pagedPageLedger->BeginReclaim(id,1);
        Require(lease.Valid()&&model.pagedPageLedger->CommitL2(lease,1),"seed L2");
        host=std::make_shared<Qwen35SpilledPage>(); host->ledgerIdentity=id;
        model.pagedSpillState->pool[{std::stoull(id.pageHash),id.layer,1,0}]=host;
        if (kind==6) {
            manager->visible.resize(pages,-1);
            manager->visible.back()=pages-1;
        }
    }
    rejectIndex=kind==2?int(identities.size())-1:-1;
    int calls=0,rollbacks=0;
    auto begin=std::chrono::steady_clock::now();
    const bool ok=model.PublishDirectPagedPrefix(&ctx,pages*128,[&]() {
        ++calls;
        for (auto &manager:model.managers) {
            manager->visible.resize(pages);
            for (int p=0;p<pages;++p) {
                manager->visible[p]=p;
                manager->pageTimestamp[p]=2;
                auto node=std::make_shared<CacheTrieNode>();
                node->producerEvent=std::make_shared<int>(2);
                manager->pageToTrieNode[p]=node;
            }
        }
        if (kind==3) return false;
        if (kind==4) throw std::runtime_error("injected publication error");
        if (kind==5) model.managers.back()->pageToTrieNode[pages-1]->producerEvent.reset();
        return true;
    },[&](){++rollbacks; if(kind==4) throw std::runtime_error("rollback callback");});
    double ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-begin).count();
    Require(ok==(kind==0||kind==1||kind==7),"transaction result");
    Require(!model.pagedSpillState->directPublishActive.load(),"publication flag reset");
    Require(calls==(kind==6?0:1),"callback count");
    uint64_t queries=0,scanned=0,pageLocks=0;
    for (auto &manager:model.managers) { queries+=manager->queries; scanned+=manager->scanned;pageLocks+=manager->pageLocks; }
    if (ok) {
        Require(queries==uint64_t(2*count*(baseline?pages:1)),"whole-prefix query complexity");
        if (host) Require(model.pagedSpillState->pool.empty(),"adopted host owner retained");
        for (const auto &id:identities) {
            auto rec=model.pagedPageLedger->Find(id);
            Require(rec&&rec->runtimeTier==PrefixCacheRuntimeTier::L1&&
                    rec->l1Page==int64_t(id.pageOrdinal)&&rec->producerEvent,
                    "complete L1 adoption");
        }
    } else if (kind!=6) {
        Require(rollbacks==1,"rollback called once");
        if (!baseline) {
            Require(queries<=uint64_t(3*count),"bounded rollback queries");
            for (auto &manager:model.managers)
                Require(manager->restoredHoles<=1,"repeated whole-prefix hole reconstruction");
        }
        for (const auto &id:identities) {
            auto rec=model.pagedPageLedger->Find(id);
            Require(!rec||(host&&id==host->ledgerIdentity&&
                    rec->runtimeTier==PrefixCacheRuntimeTier::L2&&rec->activeLeaseSerial==0),
                    "atomic rollback retained L2 without partial L1 adoption");
        }
        for (auto &manager:model.managers)
            for (int id:manager->visible) Require(id<0,"failed publication left new GPU owner");
    }
    if (host) Require(host->promotionRefs.load()==0,"host promotion pin leaked");
    std::cout<<"case="<<kind<<" tokens="<<pages*128<<" queries="<<queries
             <<" scanned_positions="<<scanned<<" page_locks="<<pageLocks
             <<" manager_lookups="<<model.managerLookups<<" elapsed_ms="<<ms<<"\n";
}
int main(int argc,char**argv) {
    if(argc>=4) {
        Case(std::atoi(argv[2]),0,false,std::atoi(argv[3]));
        return 0;
    }
    const bool baseline=false;
    for (int pages:{56,128,256}) Case(pages,0,baseline);
    for (int kind=1;kind<=7;++kind) Case(8,kind,baseline);
    std::cout<<"production direct publication T0: PASS\n";
}
