#ifndef FASTLLM_PAGED_CACHE_BATCH_H
#define FASTLLM_PAGED_CACHE_BATCH_H

#include "fastllm.h"
#include "utils/inference_trace.h"
#include "utils/persistent_worker_group.h"
#include <numeric>
#include <unordered_map>

namespace fastllm {
    // CPU metadata only. The caller retains request/manager ownership until
    // completion. Jobs sharing a manager preserve their original order;
    // different managers have independent reference counts, generations and LRU.
    class PagedCacheBatchExecutor {
        PersistentWorkerGroup workers;

        template<class Task, class Work>
        void Run(const std::vector<Task> &tasks, int threadLimit,
                 const char *region, Work work) {
            size_t pages = 0;
            for (const auto &task : tasks) pages += task.PageCount();
            if (threadLimit < 2 || tasks.size() < 8 || pages < 4096) {
                for (const auto &task : tasks) work(task);
                return;
            }
            std::unordered_map<PagedCacheManager *, size_t> indices;
            std::vector<std::vector<size_t>> groups;
            for (size_t i = 0; i < tasks.size(); ++i) {
                auto inserted = indices.emplace(tasks[i].manager, groups.size());
                if (inserted.second) groups.emplace_back();
                groups[inserted.first->second].push_back(i);
            }
            const int count = std::min({threadLimit, 8, (int)groups.size()});
            if (count < 2) {
                for (const auto &task : tasks) work(task);
                return;
            }
            std::vector<int> keys(count);
            std::iota(keys.begin(), keys.end(), 0);
            std::vector<std::exception_ptr> errors(count);
            auto *parent = inference_trace::CurrentDetail();
            std::unique_ptr<inference_trace::HostDetail[]> details;
            if (parent) details.reset(new inference_trace::HostDetail[count]);
            workers.RunWithCaller(keys, [&](int rank) {
                struct Scope {
                    inference_trace::HostDetail *previous;
                    explicit Scope(inference_trace::HostDetail *next)
                        : previous(inference_trace::CurrentDetail()) {
                        inference_trace::CurrentDetail() = next;
                    }
                    ~Scope() { inference_trace::CurrentDetail() = previous; }
                } scope(parent ? &details[rank] : nullptr);
                inference_trace::DetailRegion span(region);
                for (size_t group = rank; group < groups.size(); group += count)
                    for (size_t i : groups[group]) work(tasks[i]);
            }, errors);
            // Concurrent worker intervals are sums, never serial wall time.
            if (parent) {
                const char *oldRegion = parent->region;
                for (int rank = 0; rank < count; ++rank) {
                    parent->dropped += details[rank].dropped;
                    for (size_t i = 0; i < details[rank].used; ++i) {
                        const auto &from = details[rank].metrics[i];
                        parent->region = from.region;
                        if (auto *to = parent->Get(from.name)) {
                            to->wallMs += from.wallMs; to->cpuMs += from.cpuMs;
                            to->calls += from.calls; to->count += from.count;
                            to->cpuAvailable |= from.cpuAvailable;
                        }
                    }
                }
                parent->region = oldRegion;
            }
            for (const auto &error : errors) if (error) std::rethrow_exception(error);
        }
    public:
        struct Publication {
            PagedCacheManager *manager;
            const std::vector<uint64_t> *hashes;
            std::vector<int> pages;
            size_t PageCount() const { return pages.size(); }
        };
        void Publish(const std::vector<Publication> &tasks, int threadLimit) {
            Run(tasks, threadLimit, "paged_publish_workers_sum", [](const Publication &task) {
                inference_trace::DetailSpan trace("record.trie_publish");
                task.manager->RecordPrefix(*task.hashes, task.pages);
            });
        }
        void ReleaseReferences(const std::vector<Data *> &caches, int threadLimit) {
            struct Release {
                PagedCacheManager *manager;
                Data *cache;
                size_t PageCount() const { return cache->pageIndex.size(); }
            };
            std::vector<Release> tasks;
            for (auto *cache : caches)
                // Match Data's destructor. Only leaf reference bookkeeping is
                // parallel; CUDA frees and descriptor destruction stay on caller.
                if (cache && !cache->isFake && !cache->kvMemCache && cache->isPagedKVCache &&
                    !cache->pageIndex.empty())
                    tasks.push_back({cache->pagedKVCacheData, cache});
            Run(tasks, threadLimit, "paged_release_workers_sum", [](const Release &task) {
                task.manager->ReleasePageIndices(task.cache->pageIndex);
                task.cache->pageIndex.clear(); // Prevent a second release in ~Data.
            });
        }
    };
}
#endif
