// KVPAIR unit test: K/V paged-cache managers must hand out equal page ids while another thread releases
// K then V pages with a gap (the SSD D2H lease release pattern). Links against libfastllm_tools.so.
// Run without FASTLLM_PAGED_KV_PAIR_ALLOC (expect mismatches > 0: the race exists) and with
// FASTLLM_PAGED_KV_PAIR_ALLOC=1 (expect mismatches == 0 and pools equal at the end).
#include "fastllm.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace fastllm;

int main(int argc, char **argv) {
    const int iterations = argc > 1 ? std::atoi(argv[1]) : 200000;
    PagedCacheManager k, v, k1m, v1m;   // layer 0 (k, v) and layer 1 (k1m, v1m)
    k.pageLen = v.pageLen = k1m.pageLen = v1m.pageLen = 16;
    k.SetMaxPages(64);
    v.SetMaxPages(64);
    k1m.SetMaxPages(64);
    v1m.SetMaxPages(64);
    PairPagedCacheManagers(&k, &v);
    PairPagedCacheManagers(&k1m, &v1m);
    const bool paired = k.pairPartner == &v;

    // Thread B holds a few pages like an SSD lease and releases K before V with a gap.
    std::atomic<bool> stop{false};
    std::atomic<long> leaseCycles{0};
    std::thread releaser([&] {
        while (!stop.load()) {
            int kp = k.GetUnusedPageIndex(true);
            int vp = v.GetUnusedPageIndex(true);
            int kp1 = k1m.GetUnusedPageIndex(true);
            int vp1 = v1m.GetUnusedPageIndex(true);
            std::this_thread::yield();
            k.ReleasePageIndex(kp);          // lease release order of the SSD D2H loop: K layer 0, K layer 1,
            k1m.ReleasePageIndex(kp1);       // then V layer 0, V layer 1, with a gap
            for (volatile int spin = 0; spin < 200; spin = spin + 1) {}
            v.ReleasePageIndex(vp);
            v1m.ReleasePageIndex(vp1);
            ++leaseCycles;
        }
    });

    long mismatches = 0, previewMismatches = 0, crossLayerDiffs = 0;
    for (int i = 0; i < iterations; ++i) {
        int kp = k.GetUnusedPageIndex(true);
        int vp = v.GetUnusedPageIndex(true);
        int kp1 = k1m.GetUnusedPageIndex(true);
        int vp1 = v1m.GetUnusedPageIndex(true);
        if (kp != vp || kp1 != vp1) ++mismatches;
        if (kp != kp1) ++crossLayerDiffs;
        k.ReleasePageIndex(kp);
        k1m.ReleasePageIndex(kp1);
        v.ReleasePageIndex(vp);
        v1m.ReleasePageIndex(vp1);
        // Append-preview pattern (cudadevice CudaGenerateAppendPagedCacheBatchParamsOp): preview two ids on K,
        // let the releaser run, then allocate K,V,K,V; with pairing the handed-out ids must equal the preview.
        if (paired && (i % 4) == 0) {
            // Each layer previews its own pages (as qwen3_5.cpp does with pairing), same calls in both layers.
            for (auto layer : {std::make_pair(&k, &v), std::make_pair(&k1m, &v1m)}) {
                PagedCacheManager &lk = *layer.first, &lv = *layer.second;
                std::vector<int> preview;
                lk.PairReserveForThread(2, preview);
                std::this_thread::yield();
                int k1 = lk.GetUnusedPageIndex(true), v1 = lv.GetUnusedPageIndex(true);
                int k2 = lk.GetUnusedPageIndex(true), v2 = lv.GetUnusedPageIndex(true);
                if (preview.size() != 2 || k1 != preview[0] || v1 != preview[0] || k2 != preview[1] || v2 != preview[1]) ++previewMismatches;
                lk.ReleasePageIndex(k1); lv.ReleasePageIndex(v1); lk.ReleasePageIndex(k2); lv.ReleasePageIndex(v2);
            }
        }
    }
    stop = true;
    releaser.join();

    const std::string diff = k.PairDiff(v);
    const std::string layerDiff = k.PairDiff(k1m);   // layer 0 vs layer 1 K: order must not diverge permanently
    // INFO only: layers diverge in order under concurrent releases also without pairing (pre-existing).
    std::printf("cross_layer_transient_diffs=%ld final_layer_diff='%s'\n", crossLayerDiffs, layerDiff.c_str());
    std::printf("preview_mismatches=%ld\n", previewMismatches);
    std::printf("paired=%d iterations=%d lease_cycles=%ld mismatches=%ld pair_allocs=%llu consumed=%llu cancelled=%llu "
                "fallbacks=%llu reserved_left=%zu/%zu final_diff='%s'\n",
                paired ? 1 : 0, iterations, leaseCycles.load(), mismatches, k.pairAllocs + v.pairAllocs,
                k.pairConsumed + v.pairConsumed, k.pairCancelled + v.pairCancelled, k.pairFallbacks + v.pairFallbacks,
                k.pairReserved.size(), v.pairReserved.size(), diff.c_str());
    if (paired) {
        // K/V of one layer must end with identical pools (order may differ); cross-layer order is INFO only
        const bool ok = mismatches == 0 && previewMismatches == 0 && (diff.empty() || diff == "free_order") && k.pairReserved.empty() && v.pairReserved.empty();
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    std::printf("%s (flag off: race %s)\n", "INFO", mismatches > 0 ? "reproduced" : "not hit");
    return 0;
}
