// T0 ledger commit benchmark (2026-10-17, B21): reproduce the 2.8 s commit at
// the real scale (968 pages x 128 managers = 123904 identities) without GPU.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "utils/prefix_cache_page_ledger.h"

using namespace fastllm;

static std::string Hex64(uint64_t a, uint64_t b) {
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "%016llx%016llx%016llx%016llx",
                  (unsigned long long)a, (unsigned long long)b,
                  (unsigned long long)(a * 0x9e3779b97f4a7c15ull),
                  (unsigned long long)(b * 0xc2b2ae3d27d4eb4full));
    return std::string(buffer);
}
static size_t targetKeyChars=0;
static PrefixCachePageIdentity MakeIdentity(int page, int manager) {
    PrefixCachePageIdentity id;
    id.model = "Qwen3.8-27B-FP8";
    id.family = "qwen3.5-linear-kv-fp8-e4m3-paged-512";
    id.provenance = Hex64(uint64_t(page) + 1, uint64_t(manager) + 1);
    id.dtype = "fp8_e4m3";
    id.layout = "paged";
    id.parentHash = Hex64(uint64_t(page) * 3 + 1, 99);
    id.pageHash = Hex64(uint64_t(page) * 1315423911ull + 7, uint64_t(manager) + 3);
    id.layer = manager / 8;
    id.side = ((manager / 4) % 2) ? "key" : "value";
    id.rank = manager % 4;
    id.pageOrdinal = page;
    if(targetKeyChars) {
        while(id.CanonicalKey().size()!=targetKeyChars) {
            const size_t n=id.CanonicalKey().size();
            if(n<targetKeyChars) id.media.append(targetKeyChars-n,'m');
            else if(id.media.size()>=n-targetKeyChars) id.media.resize(id.media.size()-(n-targetKeyChars));
            else std::abort();
        }
    }
    return id;
}

int main(int argc, char **argv) {
    targetKeyChars=argc>5?size_t(std::atoi(argv[5])):0;
    const int pages = argc > 1 ? std::atoi(argv[1]) : 968;
    const int managers = argc > 2 ? std::atoi(argv[2]) : 128;
    const int prefill = argc > 3 ? std::atoi(argv[3]) : 0;
    PrefixCachePageLedger ledger;
    std::vector<PrefixCachePageIdentity> identities;
    std::vector<PrefixCachePageLedger::L1Adoption> adoptions;
    identities.reserve(size_t(pages) * managers);
    adoptions.reserve(size_t(pages) * managers);
    static int managersStorage[256];
    for (int p = 0; p < pages; ++p) {
        for (int m = 0; m < managers; ++m) {
            identities.push_back(MakeIdentity(p, m));
        }
    }
    if (prefill > 0) {
        const auto pre0 = std::chrono::steady_clock::now();
        for (int i = 0; i < prefill; ++i) {
            PrefixCachePageIdentity id = MakeIdentity(1000000 + i / managers, i % managers);
            ledger.ObserveL1(id, &managersStorage[i % 256], i % 1024, 1);
        }
        const double preMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pre0).count();
        std::printf("prefill=%d prefill_ms=%.1f\n", prefill, preMs);
    }
    size_t index = 0;
    for (int p = 0; p < pages; ++p) {
        for (int m = 0; m < managers; ++m) {
            PrefixCachePageLedger::L1Adoption adoption;
            adoption.identity = identities[index++];
            adoption.manager = &managersStorage[m];
            adoption.page = p;
            adoption.generation = 1;
            adoptions.push_back(std::move(adoption));
        }
    }
    if (argc > 4 && std::atoi(argv[4]) != 0) {
        const auto pre0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < adoptions.size(); ++i)
            ledger.ObserveL1(adoptions[i].identity, adoptions[i].manager, adoptions[i].page, 1);
        std::printf("preobserve_same pre_ms=%.1f\n",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pre0).count());
    }
    const auto start = std::chrono::steady_clock::now();
    const bool ok = ledger.CommitL1Batch(adoptions, &identities);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::printf("identities=%zu ok=%d commit_ms=%.1f per_identity_us=%.2f\n",
                identities.size(), ok ? 1 : 0, ms, ms * 1000.0 / identities.size());
    uint64_t fingerprint=0;
    if(ok) {
        for(size_t i=0;i<adoptions.size();++i) {
            PrefixCachePageSnapshot snapshot;
            const auto &expected=adoptions[i];
            if(!ledger.Snapshot(expected.identity,snapshot) || snapshot.identity!=expected.identity ||
               snapshot.runtimeTier!=PrefixCacheRuntimeTier::L1 ||
               snapshot.runtimeState!=PrefixCacheRuntimeState::RESIDENT ||
               snapshot.l1Manager!=expected.manager || snapshot.l1Page!=expected.page ||
               snapshot.l1Generation!=expected.generation || snapshot.activeLeaseSerial!=0) {
                std::fprintf(stderr,"FAIL ledger output verification index=%zu\n",i);
                return 3;
            }
            fingerprint^=std::hash<std::string>()(snapshot.identity.CanonicalKey())+
                (uint64_t(snapshot.l1Page)*0x9e3779b97f4a7c15ull);
        }
    }
    const auto timing=ledger.LastCommitTiming();
    std::printf("verified=%zu fingerprint=%016llx prepare_ms=%.3f alloc_ms=%.3f "
                "mutate_ms=%.3f trim_ms=%.3f key_avg_chars=%.1f\n",
                ok?adoptions.size():0,(unsigned long long)fingerprint,timing.prepareMs,
                timing.allocMs,timing.mutateMs,timing.trimMs,
                timing.keyCount?double(timing.keyChars)/(2.0*timing.keyCount):0.0);
    return ok ? 0 : 1;
}
