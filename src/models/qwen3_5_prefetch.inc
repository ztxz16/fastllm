// The worker serves request-scoped SSD reads only.  SSD data is restored
// directly by Prepare/Import(..., false, ...) into L1; this worker must never
// warm L2 or make a background hint look like a request hit.
void Qwen35PersistentCache::PrefetchWork() {
    PrefixCacheInfo("[Prefix SSD] startup/background prefetch disabled=ssd_to_l2_prefetch.\n");

    for (;;) {
        std::shared_ptr<ReadTicket> ticket;
        Json backgroundRequest;
        bool isRequestRead = false;
        {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            prefetchChanged.wait(lock,[&] {
                return prefetchStopping || !requestReads.empty() || !prefetchQueue.empty();
            });
            if (prefetchStopping) break;
            // Request-scoped reads have priority.  A queued hint is consumed
            // and discarded below so an old producer cannot import SSD data
            // into L2 after the three-tier contract was enabled.
            if (!requestReads.empty()) {
                ticket = requestReads.front(); requestReads.pop_front();
                isRequestRead = true;
            } else {
                backgroundRequest = std::move(prefetchQueue.front());
                prefetchQueue.pop_front();
            }
        }

        if (!isRequestRead) {
            const std::string requestKey = backgroundRequest["request_key"].string_value();
            PrefixCacheDebug("[Prefix SSD] prefetch: disabled=ssd_to_l2_prefetch request=%s.\n",
                             requestKey.c_str());
            ++prefetchMisses;
            if (!requestKey.empty()) {
                std::lock_guard<std::mutex> lock(prefetchMutex);
                prefetchPending.erase(requestKey);
            }
            continue;
        }

        if (!ticket->cancelled.load()) {
            RequestDiagnosticOperation diagnosticRead(ticket->diagnosticToken, "ssd_read_worker",
                "DiskPrefixCache.ReadSession.Load");
            const auto begin = Clock::now();
            try {
                ticket->reader = std::make_unique<DiskPrefixCache::ReadSession>(store->BeginRead());
                const auto progress = [ticket](uint64_t bytes,uint64_t discovered) {
                    ticket->discoveredBytes.fetch_add(discovered,std::memory_order_relaxed);
                    ticket->verifiedBytes.fetch_add(bytes,std::memory_order_relaxed);
                    return !ticket->cancelled.load(std::memory_order_acquire) &&
                           Clock::now() < ticket->deadline;
                };
                ticket->state = ticket->reader->Load(ticket->candidate.key,
                                                     ticket->candidate.identity,true,progress);
                ticket->loadedBytes = ticket->reader->LoadedBytesCount();
                ticket->ok = !ticket->cancelled.load() && Clock::now() < ticket->deadline;
            } catch (const std::exception &error) {
                ticket->error = error.what();
            }
            ticket->readMs = std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            if (ticket->error.empty() && !ticket->ok) ticket->error = "ssd_read_cancelled_or_timeout";
            PrefixCacheDebug("[Prefix SSD] request-read-complete: request=%llu bytes=%llu/%llu "
                        "total_state=%s read_ms=%.1f speed_mib_s=%.1f adopted=0 result=%s.\n",
                        (unsigned long long)ticket->id,(unsigned long long)ticket->verifiedBytes.load(),
                        (unsigned long long)ticket->discoveredBytes.load(),ticket->ok ? "complete" : "incomplete",
                        ticket->readMs,ticket->readMs > 0 ?
                            ticket->verifiedBytes.load()/1048576.0*1000/ticket->readMs : 0,
                        ticket->ok ? "ready" : "failed");
            if (ticket->ok) ++prefetchCompleted; else ++prefetchMisses;
        }
        ticket->done.store(true,std::memory_order_release);
        model->dictCV.notify_all();
        // Keep at most one full request payload alive until the scheduler
        // has either adopted it or cancelled/timed it out.
        {
            RequestDiagnosticOperation diagnosticConsume(ticket->diagnosticToken,
                "ssd_read_consumption_wait", RequestDiagnosticResource("persistent.prefetchChanged", &prefetchChanged).c_str());
            std::unique_lock<std::mutex> lock(prefetchMutex);
            prefetchChanged.wait(lock,[&] {
                return prefetchStopping || ticket->consumed.load();
            });
            if (prefetchStopping) break;
        }
    }
}
