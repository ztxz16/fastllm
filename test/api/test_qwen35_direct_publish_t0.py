"""CPU regression for the production SSD-to-L1 publication transaction.

Extracts the full production function unchanged. The trie/GPU adapter is a
metadata fake with query counters; the ownership ledger is the production
header. No model, GPU, user SSD, or inference input is used.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
BUILD = WORKSPACE / "build/request-observability-20261016/host-probes/direct-publish"
LOGS = WORKSPACE / "logs"


def record(**fields):
    with (LOGS / "actions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "task": "ssd-direct-publish-t0",
            **fields}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def extract(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth, state, end = 1, "code", opening + 1
    while depth:
        char, pair = source[end], source[end:end + 2]
        if state == "line":
            if char == "\n":
                state = "code"
        elif state == "comment":
            if pair == "*/":
                state, end = "code", end + 1
        elif state in ('"', "'"):
            if char == "\\":
                end += 1
            elif char == state:
                state = "code"
        elif pair == "//":
            state, end = "line", end + 1
        elif pair == "/*":
            state, end = "comment", end + 1
        elif char in ('"', "'"):
            state = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        end += 1
    return source[start:end]


PREFIX = r"""
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
    uint64_t queries = 0, scanned = 0, restoredHoles = 0;
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
    std::vector<std::pair<int,PagedCacheManager*>> GetPagedKVCacheManagers(
            int layer, bool key) const {
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
"""

CASES = r"""
}
using namespace fastllm;
static void Case(int pages, int kind, bool baseline) {
    ProbeModel model;
    ResponseContext ctx;
    const int count=8, overlap=kind==1?3:0;
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
    uint64_t queries=0,scanned=0;
    for (auto &manager:model.managers) { queries+=manager->queries; scanned+=manager->scanned; }
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
             <<" scanned_positions="<<scanned<<" elapsed_ms="<<ms<<"\n";
}
int main(int argc,char**) {
    const bool baseline=argc>1;
    for (int pages:{56,128,256}) Case(pages,0,baseline);
    for (int kind=1;kind<=7;++kind) Case(8,kind,baseline);
    std::cout<<"production direct publication T0: PASS\n";
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Verify the known old quadratic query count before a fix.")
    args = parser.parse_args()
    tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    folder = BUILD / tag
    record(action="read production publication", path=str(ROOT / "src/models/qwen3_5_spill.inc"))
    if args.baseline:
        # Preserve the original query algorithm as the control. Only the
        # cleanup-guard declaration is corrected, as in the candidate, so GCC
        # can compile both with all warnings fatal. No warning is disabled.
        command = ["git", "show", "c238b4a1:src/models/qwen3_5_spill.inc"]
        record(action="read committed baseline", command=command, cwd=str(ROOT))
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        record(action="read committed baseline", exit_code=completed.returncode)
        if completed.returncode:
            raise SystemExit(completed.stderr)
        source = completed.stdout
    else:
        source = (ROOT / "src/models/qwen3_5_spill.inc").read_text()
    function = extract(source, "bool Qwen3_5Model::PublishDirectPagedPrefix(")
    if args.baseline:
        assert "const std::function<void()> &cleanup;" in function
        function = function.replace("const std::function<void()> &cleanup;",
                                    "decltype(cleanup) &callback;")
        function = function.replace("if (!committed) cleanup();",
                                    "if (!committed) callback();")
    function = function.replace("Qwen3_5Model::", "ProbeModel::")
    record(action="write probe under build", build=str(folder),
           source_sha256=hashlib.sha256(source.encode()).hexdigest(), baseline=args.baseline)
    folder.mkdir(parents=True)
    (folder / "probe.cpp").write_text(PREFIX + function + CASES)
    command = ["/usr/bin/c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-pthread", "-I" + str(ROOT / "include"), str(folder / "probe.cpp"),
               "-o", str(folder / "probe")]
    (folder / "Makefile").write_text("all:\n\t" + " ".join(command) + "\n")
    for label, command in [("compile", ["make", "-j36", "-C", str(folder)]),
                            ("run", [str(folder / "probe")] + (["baseline"] if args.baseline else []))]:
        log = LOGS / ("direct-publish-t0-" + tag + "-" + label + ".log")
        record(action=label, command=command, cwd=str(ROOT), log_path=str(log),
               gpu_mapping="none; CPU metadata only", expected="exit 0")
        with log.open("xb") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        record(action=label, exit_code=completed.returncode, log_path=str(log))
        print(log.read_text(), end="")
        if completed.returncode:
            raise SystemExit(completed.returncode)
    print("T0 evidence:", tag)


if __name__ == "__main__":
    main()
