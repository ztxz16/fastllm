#ifndef FASTLLM_PAGED_PREFIX_PREVIEW_H
#define FASTLLM_PAGED_PREFIX_PREVIEW_H

#include "fastllm.h"
#include "utils/inference_trace.h"
#include "utils/persistent_worker_group.h"
#include <numeric>

namespace fastllm {
    // Owned by one scheduler. The caller holds the model/request lifetime lock;
    // managers are unique and each worker touches only its own matches/managers.
    // No KV is acquired and each manager still validates its anchor generation.
    class PagedPrefixPreviewExecutor {
        PersistentWorkerGroup workers;
        std::vector<int> keys, lengths;
        std::vector<PagedPrefixMatch *> matches;
        std::vector<std::exception_ptr> errors;
    public:
        int Query(const std::vector<PagedCacheManager *> &managers,
                  const std::vector<uint64_t> &hashes,
                  std::map<PagedCacheManager *, PagedPrefixMatch> &cached,
                  int pages, int threadLimit) {
            if (pages <= 0 || managers.size() <= 1) return pages;
            const int count = std::min({threadLimit, 8, (int)managers.size() - 1});
            if (count < 2 || managers.size() < 8 || hashes.size() < 4096 / (managers.size() - 1)) {
                for (size_t i = 1; i < managers.size() && pages > 0; ++i)
                    pages = std::min(pages, managers[i]->PreviewPrefix(hashes, cached[managers[i]]));
                return pages;
            }
            matches.resize(managers.size());
            size_t uncached = 0;
            for (size_t i = 1; i < managers.size(); ++i) {
                matches[i] = &cached[managers[i]];
                uncached += hashes.size() - std::min(hashes.size(), (size_t)std::max(0, matches[i]->pages));
            }
            // Warm anchors and small prompts are cheaper on the scheduler.
            // A stale optimistic estimate only selects the serial fallback.
            if (uncached < 4096) {
                for (size_t i = 1; i < managers.size() && pages > 0; ++i)
                    pages = std::min(pages, managers[i]->PreviewPrefix(hashes, *matches[i]));
                return pages;
            }
            keys.resize(count); std::iota(keys.begin(), keys.end(), 0);
            errors.resize(count); lengths.assign(count, pages);
            auto *parent = inference_trace::CurrentDetail();
            // Per-worker records are allocated only for sampled diagnostics.
            // Their times are SUMS of concurrent worker intervals, not elapsed
            // scheduler time; keep them in a separately named trace region.
            std::unique_ptr<inference_trace::HostDetail[]> details;
            if (parent) details.reset(new inference_trace::HostDetail[count]);
            workers.RunWithCaller(keys, [&](int rank) {
                struct Scope {
                    inference_trace::HostDetail *previous;
                    Scope(inference_trace::HostDetail *current)
                        : previous(inference_trace::CurrentDetail()) {
                        inference_trace::CurrentDetail() = current;
                    }
                    ~Scope() { inference_trace::CurrentDetail() = previous; }
                } scope(parent ? &details[rank] : nullptr);
                inference_trace::DetailRegion region("prefill_preview_workers_sum");
                for (size_t i = 1 + rank; i < managers.size(); i += count)
                    lengths[rank] = std::min(lengths[rank],
                        managers[i]->PreviewPrefix(hashes, *matches[i]));
            }, errors);
            for (auto &error : errors) if (error) std::rethrow_exception(error);
            if (parent) {
                const char *oldRegion = parent->region;
                for (int rank = 0; rank < count; ++rank) {
                    const auto &detail = details[rank];
                    parent->dropped += detail.dropped;
                    for (size_t i = 0; i < detail.used; ++i) {
                        const auto &from = detail.metrics[i];
                        parent->region = from.region;
                        if (auto *to = parent->Get(from.name)) {
                            to->wallMs += from.wallMs; to->cpuMs += from.cpuMs;
                            to->calls += from.calls; to->count += from.count;
                            to->cpuAvailable |= from.cpuAvailable;
                        }
                    }
                }
                parent->region = oldRegion;
                parent->Count("preview.parallel_managers", managers.size() - 1);
            }
            return *std::min_element(lengths.begin(), lengths.end());
        }
    };
}
#endif
