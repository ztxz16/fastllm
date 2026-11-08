// Local KV tiering adapter. Reuses PagedCacheManager for allocation/pinning.
// References: vLLM v1/kv_offload/base.py:252-334 (load/store protection),
// SGLang unified_radix_cache.py:880-899 (backup before demotion),
// LMCache storage_manager.py:437-461 (promote a lower-tier hit into the hot tier).
using Qwen35SpillKey = std::tuple<uint64_t, int, int, int>;

// A3 admission contract: these are deliberately small fixed limits.  The
// worker and store stream remain single-owner; a full queue is a terminal
// DEFER, never an invitation to retry forever or grow memory without bound.
static constexpr size_t kQwen35SpillQueueJobs = 16;
static constexpr size_t kQwen35SpillPendingBytes = size_t(128) << 20;
static constexpr size_t kQwen35SpillCopySlots = 1;
static constexpr size_t kQwen35SpillWorkers = 1;
static_assert(kQwen35SpillWorkers == 1, "A3 spill worker contract requires one worker");

enum class Qwen35SpillCopyState : uint8_t {
    Created,
    ProducerRecorded,
    FreezeComplete,
    Queued,
    Copying,
    CopyCommitted,
    QueueFull,
    CopyFailed,
    WorkerExit,
    ShutdownDrained,
    ShutdownTimeout,
};
enum class Qwen35SpillShutdownState : uint8_t {
    Running,
    Stopping,
    Drained,
    Timeout,
};

struct Qwen35SpilledPage {
    PrefixCachePageIdentity ledgerIdentity;
    uint64_t ledgerAllocation = 0;
    uint64_t lastUse = 0;
    DataType dataType = DataType::FLOAT16;
    int pageLen = 0, heads = 0, dim = 0;
    std::vector<uint8_t> hostBytes;
    std::list<Qwen35SpillKey>::iterator lruPosition;
    std::shared_ptr<Qwen35PrefixWriteStatus> writeStatus;
    // A promotion borrows this host owner until the GPU trie has adopted the
    // page.  L2 eviction must not reclaim an in-flight promotion.
    std::atomic<uint32_t> promotionRefs{0};
    // The host object is inserted before the ledger commit so the commit can
    // be performed without holding poolMutex. Keep it invisible to lookup
    // and eviction until that commit succeeds.
    std::atomic<bool> ledgerPublished{false};
    bool fromSSD = false;
};

struct Qwen35SpillTask {
    PagedCacheManager *manager = nullptr;
    Qwen35SpillKey key;
    int pageId = -1;
    int device = -1;
    std::shared_ptr<void> producerReady;
    std::shared_ptr<PagedCacheManager::RecycleTicket> recycleTicket;
    bool completeOnly = false;
    bool reject = false;
    std::shared_ptr<Qwen35PrefixWriteStatus> writeStatus;
    PrefixCachePageLedger::Lease ledgerLease;
    uint64_t ledgerAllocation = 0;
    PrefixCachePageIdentity ledgerIdentity;
    uint64_t save_job_id = 0;        // Recycle ticket serial; never a request id.
    uint64_t request_id = 0;         // Native request that owns this save job.
    uint64_t request_generation = 0; // Request lease generation at admission.
    uint64_t page_generation = 0;    // Page migration generation (ticket).
    uint64_t copy_generation = 0;    // Allocation generation of this copy.
    std::string cache_identity;      // Canonical page cache identity.
    // Stable request owner that keeps the source pages alive until this job
    // reaches a terminal state.  It is deliberately a distinct handle, not a
    // number: request/page/copy generations must never be collapsed into one.
    std::shared_ptr<const void> owner_lease;
    size_t bytes = 0;
    Qwen35SpillCopyState copyState = Qwen35SpillCopyState::Created;
    std::string terminalReason;
    bool HasCompleteIdentity() const noexcept {
        return request_generation != 0 &&
               page_generation != 0 && copy_generation != 0 &&
               !cache_identity.empty() && owner_lease != nullptr &&
               ledgerLease.Valid();
    }
};

// Temporary GPU ownership transferred from an asynchronous restore/promotion
// to the scheduler. The scheduler adds its normal Pick references and then
// drops this holder; abort/remove paths drop it directly.
struct Qwen35GpuPrefixLease {
    std::vector<std::pair<PagedCacheManager *, std::vector<int>>> pages;
    ~Qwen35GpuPrefixLease() {
        for (auto &entry : pages)
            if (entry.first && !entry.second.empty())
                entry.first->ReleasePageIndices(entry.second);
    }
};

struct Qwen35SpillState {
    Qwen3_5Model *ownerModel = nullptr;
    std::mutex poolMutex, queueMutex;
    std::condition_variable poolChanged;
    std::condition_variable queueCond;
    std::map<Qwen35SpillKey, std::shared_ptr<Qwen35SpilledPage>> pool;
    // Host pages selected for a foreground promotion remain protected while
    // deferred L1 victims are copied to L2 and the scheduler retries.
    std::set<Qwen35SpillKey> promotionHolds;
    std::set<Qwen35SpillKey> pending;
    std::deque<Qwen35SpillTask> queue;
    size_t pendingJobs = 0;
    size_t pendingBytes = 0;
    size_t activeCopySlots = 0;
    bool pendingAccountingError = false;
    std::set<PagedCacheManager*> managers;
    std::list<Qwen35SpillKey> lru;
    std::map<uint64_t, std::shared_ptr<Qwen35PrefixWriteStatus>> writeStatuses;
    std::vector<std::shared_ptr<void>> failedWriteOwners;
    std::mutex promoteMutex;
    std::map<int, void*> storeStreams, loadStreams;
    void *storeHost = nullptr, *loadHost = nullptr;
    size_t stagingBytes = 0;
    uint64_t bytes = 0, clock = 0;
    std::atomic<bool> stopping{false};
    std::atomic<bool> prefetchStopping{false};
    std::atomic<Qwen35SpillShutdownState> shutdownState{
        Qwen35SpillShutdownState::Running};
    std::atomic<bool> directPublishActive{false};
    std::thread worker;
    std::map<uint64_t, PrefixCachePageLedger::Lease> ledgerReclaims;
    std::atomic<uint64_t> nextLedgerAllocation{1};
    std::map<ResponseContext *, std::shared_ptr<Qwen35GpuPrefixLease>> adoptedGpuOwners;
    std::map<ResponseContext *, std::vector<Qwen35SpillKey>> retryPromotionHolds;
};

static PrefixCachePageIdentity Qwen35LedgerPageIdentity(
        Qwen3_5Model *model, PagedCacheManager *manager, int layer, int side, int rank,
        uint64_t pageHash, uint64_t parentHash, uint64_t ordinal,
        const std::string &extraKey = std::string()) {
    PrefixCachePageIdentity identity;
    const std::string runtimeNamespace = model ?
        (std::string("qwen3.5@") + std::to_string(reinterpret_cast<uintptr_t>(model))) :
        std::string("qwen3.5@unknown");
    identity.model = model && !model->pagedCacheModelIdentity.empty() ?
        model->pagedCacheModelIdentity : runtimeNamespace;
    identity.family = model && !model->pagedCacheFamilyIdentity.empty() ?
        model->pagedCacheFamilyIdentity : runtimeNamespace;
    // 2026-10-17（B21）：extraKey 是「每页一枚、被该页全部 manager 共享」的长语义键。
    // 实测 CanonicalKey 平均 3966 B（provenance 与 media 各放一次原键），12.8 万身份的
    // 提交准备阶段要 1.39 s。这里把 extraKey 固定成 SHA-256（64 hex）：256-bit 抗碰撞，
    // 不同页/不同多模态 span 仍严格区分，只用于身份相等与键构造，不承载载荷语义。
    // 用 thread_local 记忆化：Describe 里同一页连续 128 次调用只算一次摘要。
    static thread_local std::string cachedExtraKey;
    static thread_local std::string cachedExtraTag;
    if (extraKey != cachedExtraKey) {
        cachedExtraKey = extraKey;
        cachedExtraTag = extraKey.empty() ? std::string() : DiskPrefixCache::Digest(extraKey);
    }
    // Preserve the complete media/provenance key.  A node key is semantic
    // identity metadata, not payload; collapsing it to a node/base bit would
    // allow two multimodal spans to alias the same runtime page.
    identity.provenance = cachedExtraTag.empty() ?
        "runtime-paged-kv:base" : std::string("runtime-paged-kv:") + cachedExtraTag;
    identity.dtype = manager ? GetDataTypeName(manager->dataType) : "unknown";
    if (model && !model->pagedCacheLayout.empty()) {
        identity.layout = model->pagedCacheLayout;
    } else if (manager && manager->dims.size() == 4) {
        identity.layout = "paged-kv-v4:" + std::to_string(manager->pageLen) + ":" +
            std::to_string(manager->dims[2]) + ":" + std::to_string(manager->dims[3]);
    } else {
        identity.layout = "paged-kv-v4:unknown";
    }
    // Empty extraKey is a real base/text namespace, not a missing identity.
    identity.media = cachedExtraTag.empty() ? "__base__" : cachedExtraTag;
    identity.parentHash = std::to_string(parentHash);
    identity.pageHash = std::to_string(pageHash);
    identity.layer = layer;
    identity.side = side == 0 ? "key" : "value";
    identity.rank = rank;
    identity.pageOrdinal = ordinal;
    (void)model;
    return identity;
}

static bool Qwen35SpillPageMatches(const Qwen35SpilledPage &page, PagedCacheManager *manager) {
    return manager != nullptr && manager->maxPages > 0 && manager->dims.size() == 4 &&
        page.dataType == manager->dataType && page.pageLen == manager->pageLen &&
        page.heads == manager->dims[2] && page.dim == manager->dims[3] &&
        page.hostBytes.size() == manager->GetBytes() / (size_t)manager->maxPages;
}

static bool Qwen35SpillPageUsable(const Qwen35SpilledPage &page,
                                   PagedCacheManager *manager) {
    // L2 is a runtime owner even while its durable write is pending; only
    // eviction requires a durable acknowledgement.  SSD read staging never
    // enters this pool, and legacy/import-tagged entries are rejected.
    return !page.fromSSD && page.ledgerPublished.load(std::memory_order_acquire) &&
        Qwen35SpillPageMatches(page, manager);
}

std::vector<PrefixCachePageIdentity> Qwen3_5Model::DescribePagedPrefixPages(
        const std::vector<int> &tokens, const std::vector<std::string> &keys,
        int length) const {
    std::vector<PrefixCachePageIdentity> result;
    const int pageLen = fastllm::GetPageLen();
    if (pageLen <= 0 || length <= 0 || length > (int)tokens.size() ||
        length % pageLen != 0) return result;
    const int pageCount = length / pageLen;
    if (!keys.empty() && (int)keys.size() < pageCount) return result;
    uint64_t hash = 0;
    for (int page = 0; page < pageCount; ++page) {
        const uint64_t parent = hash;
        hash = PagedCacheManager::HashTokenPrefixPage(
            hash, tokens.data() + page * pageLen, pageLen,
            keys.empty() ? std::string() : keys[page]);
        for (int layer = 0; layer < block_cnt; ++layer) {
            for (int side = 0; side < 2; ++side) {
                const auto refs = GetPagedKVCacheManagers(layer, side == 0);
                for (const auto &ref : refs) {
                    if (!ref.second) continue;
                    result.push_back(Qwen35LedgerPageIdentity(
                        const_cast<Qwen3_5Model *>(this), ref.second, layer, side,
                        ref.first, hash, parent, (uint64_t)page,
                        keys.empty() ? std::string() : keys[page]));
                }
            }
        }
    }
    return result;
}

// Caller holds poolMutex.  The expected pointer check prevents a late
// promotion from deleting a newer page with the same canonical key.
static bool Qwen35RemoveSpilledPageLocked(
        Qwen35SpillState &state, const Qwen35SpillKey &key,
        const std::shared_ptr<Qwen35SpilledPage> &expected = nullptr) {
    auto it = state.pool.find(key);
    if (it == state.pool.end() || (expected && it->second != expected)) return false;
    // A foreground promotion owns this logical page even before its GPU
    // adoption lease is committed.  Recycle/duplicate paths must not remove
    // that owner between the capacity preflight and the H2D publication.
    if (state.promotionHolds.count(key) != 0) return false;
    auto page = it->second;
    if (page->promotionRefs.load(std::memory_order_acquire) != 0) return false;
    const uint64_t removed = page->hostBytes.size();
    state.bytes = state.bytes >= removed ? state.bytes - removed : 0;
    state.lru.erase(page->lruPosition);
    state.pool.erase(it);
    if (Qwen35SpillStatsEnabled())
        Qwen35SpillStatsRef().poolBytes.fetch_sub(removed, std::memory_order_relaxed);
    state.poolChanged.notify_all();
    return true;
}

// Caller holds poolMutex. Payloads borrowed by a promotion remain alive via
// shared_ptr even if the LRU index drops them (vLLM prepare_load protection).
static bool Qwen35SpillEvictFor(Qwen35SpillState &state, uint64_t bytes) {
    const uint64_t limit = Qwen35L2HostKVPoolLimitBytes();
    if (limit == 0) return true;
    while (state.bytes + bytes > limit && !state.pool.empty()) {
        size_t blockedRefs = 0, blockedHolds = 0, blockedLedger = 0;
        for (const auto &key : state.lru) {
            const auto &page = state.pool.at(key);
            if (page->promotionRefs.load(std::memory_order_acquire) != 0) {
                ++blockedRefs;
            } else if (state.promotionHolds.count(key) != 0) {
                ++blockedHolds;
            } else if (!state.ownerModel || !state.ownerModel->pagedPageLedger ||
                       !state.ownerModel->pagedPageLedger->CanEvictL2(page->ledgerIdentity)) {
                ++blockedLedger;
            }
        }
        auto victim = std::find_if(state.lru.begin(), state.lru.end(), [&](const auto &key) {
            const auto &page = state.pool.at(key);
            // SGLang backup-before-demotion: an admitted write is not an ack.
            // Keep the recoverable prefix until its existing SSD COMMIT succeeds.
            return page->promotionRefs.load(std::memory_order_acquire) == 0 &&
                state.promotionHolds.count(key) == 0 &&
                state.ownerModel && state.ownerModel->pagedPageLedger &&
                state.ownerModel->pagedPageLedger->CanEvictL2(page->ledgerIdentity);
        });
        if (victim == state.lru.end()) {
            PrefixCacheDebug("[PrefixCache] spill-evict-blocked: bytes=%llu incoming=%llu "
                             "limit=%llu pool_pages=%zu promotion_refs=%zu promotion_holds=%zu "
                             "ledger_veto=%zu.\n",
                             (unsigned long long)state.bytes,
                             (unsigned long long)bytes,
                             (unsigned long long)limit,
                             state.pool.size(), blockedRefs, blockedHolds, blockedLedger);
            return false;
        }
        auto oldest = state.pool.find(*victim);
        const auto eviction = state.ownerModel->pagedPageLedger->BeginEviction(
            oldest->second->ledgerIdentity);
        if (!eviction.Valid()) return false;
        if (!state.ownerModel->pagedPageLedger->CommitEviction(eviction)) return false;
        const uint64_t removed = oldest->second->hostBytes.size();
        state.bytes -= removed;
        state.pool.erase(oldest);
        state.lru.erase(victim);
        if (Qwen35SpillStatsEnabled()) {
            auto &stats = Qwen35SpillStatsRef();
            stats.poolBytes.fetch_sub(removed, std::memory_order_relaxed);
            stats.lruEvictedPages.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return state.bytes + bytes <= limit;
}

static std::vector<std::shared_ptr<Qwen35PrefixWriteStatus>> Qwen35ProtectSpillPrefix(
        Qwen3_5Model *model, const std::vector<int> &tokens,
        const std::vector<std::string> &keys, int begin, int length,
        uint64_t parentHash) {
    std::vector<std::shared_ptr<Qwen35PrefixWriteStatus>> statuses;
    auto spill = model->pagedSpillState;
    if (!spill) return statuses;
    const int pageLen = fastllm::GetPageLen();
    // A bad window falls back to the full walk; short keys still throw in keys.at() as before.
    if (pageLen <= 0 || begin < 0 || begin > length || begin % pageLen != 0) {
        begin = 0;
        parentHash = 0;
    }
    uint64_t hash = parentHash;
    std::lock_guard<std::mutex> lock(spill->poolMutex);
    for (int p = begin / pageLen; p < length / pageLen; ++p) {
        hash = PagedCacheManager::HashTokenPrefixPage(hash, tokens.data() + p * pageLen,
            pageLen, keys.empty() ? std::string() : keys.at(p));
        auto &entry = spill->writeStatuses[hash];
        auto status = entry;
        if (!status) { status = std::make_shared<Qwen35PrefixWriteStatus>(); entry = status; }
        // Record may have already copied this page before the complete linear
        // snapshot existed. Attach the same acknowledgement to that host copy.
        for (auto it = spill->pool.lower_bound({hash, INT_MIN, INT_MIN, INT_MIN});
             it != spill->pool.end() && std::get<0>(it->first) == hash; ++it)
            if (!it->second->writeStatus) it->second->writeStatus = status;
        statuses.push_back(std::move(status));
    }
    if (spill->writeStatuses.size() > 4096) {
        for (auto it = spill->writeStatuses.begin(); it != spill->writeStatuses.end();)
            if (it->second.use_count() == 1 && it->second->state.load() == 1)
                it = spill->writeStatuses.erase(it); else ++it;
    }
    return statuses;
}

static void Qwen35FinishSpillWrite(const std::shared_ptr<Qwen35SpillState> &spill,
        const std::vector<std::shared_ptr<Qwen35PrefixWriteStatus>> &statuses, bool durable,
        std::vector<std::shared_ptr<void>> owners) {
    for (const auto &status : statuses) {
        if (durable) status->state.store(1, std::memory_order_release);
        else { int pending = 0; status->state.compare_exchange_strong(pending, -1); }
    }
    if (!durable && spill && !owners.empty()) {
        std::lock_guard<std::mutex> lock(spill->poolMutex);
        spill->failedWriteOwners.insert(spill->failedWriteOwners.end(), owners.begin(), owners.end());
    }
    if (spill) spill->poolChanged.notify_all();
}

static void Qwen35StopSpillPrefetch(const std::shared_ptr<Qwen35SpillState> &state) {
    if (state) {
        state->prefetchStopping.store(true);
        state->poolChanged.notify_all();
    }
}

// The paged-cache trie uses the hash of the page itself for its edge and
// keeps the complete rolling identity on the node.  Query() intentionally
// stops at a released page; spill lookup must be able to bridge that hole
// with an L2 owner, so walk the trie without taking a read lease.
static uint64_t Qwen35SpillEdgeHash(const int *tokens, int len,
                                    const std::string &extra) {
    uint64_t hash = PagedCacheManager::HashTokenPage(tokens, len);
    for (unsigned char byte : extra) hash = (hash ^ byte) * 1099511628211ULL;
    return hash;
}

static std::vector<int> Qwen35QuerySpillL1Pages(
        PagedCacheManager *manager, const std::vector<int> &tokens,
        const std::vector<std::string> *extraKeys, int pages, bool pick = false) {
    std::vector<int> result;
    if (!manager || pages <= 0 || manager->pageLen <= 0 ||
        (int)tokens.size() < pages * manager->pageLen) return result;
    // QueryPagePositions holds pageIndexLocker across validation and PickLocked.
    // This closes the Query -> Pick window in which a stale page id could be
    // reused by another request.
    manager->QueryPagePositions(tokens, result, extraKeys, pages, pick);
    return result;
}

static uint64_t Qwen35InjectedD2HFailureSerial() {
    static std::atomic<bool> fired{false};
    const char *value = std::getenv("FASTLLM_TEST_FAIL_SPILL_D2H_SERIAL");
    if (!value || !*value || fired.exchange(true, std::memory_order_acq_rel)) return 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && *end == '\0' ? static_cast<uint64_t>(parsed) : 0;
}

static bool Qwen35InjectAllD2HFailures() {
    const char *value = std::getenv("FASTLLM_TEST_FAIL_SPILL_D2H_ALL");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

static int Qwen35InjectedL1RejectIndex() {
    static std::atomic<bool> fired{false};
    const char *value = std::getenv("FASTLLM_TEST_REJECT_L1_BATCH_AT");
    if (!value || !*value || fired.exchange(true, std::memory_order_acq_rel)) return -1;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 0 && parsed <= INT_MAX ?
        static_cast<int>(parsed) : -1;
}

// A-6 fix E (2026-11-01, user-approved, test only): FASTLLM_TEST_SPILL_ADMISSION_FAIL_AT=N turns
// the Nth spill admission into the existing admission-failure path (-> AbortReclaim), so G7 can be
// observed at runtime. Unset: no counting, no behavior change.
static bool Qwen35InjectSpillAdmissionFailure() {
    static const uint64_t target = [] {
        const char *value = std::getenv("FASTLLM_TEST_SPILL_ADMISSION_FAIL_AT");
        if (!value || !*value) return uint64_t(0);
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        return end != value && *end == '\0' && parsed > 0 ? static_cast<uint64_t>(parsed) : uint64_t(0);
    }();
    static std::atomic<uint64_t> admissions{0};
    return target != 0 && admissions.fetch_add(1, std::memory_order_acq_rel) + 1 == target;
}

static void Qwen35DropPromotedGpuPage(PagedCacheManager *manager, int page) {
    if (!manager || page < 0) return;
    std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
    auto it = manager->pageToTrieNode.find(page);
    if (it == manager->pageToTrieNode.end() || !it->second) return;
    CacheTrieNode *node = it->second;
    manager->pageToTrieNode.erase(it);
    if (manager->triePagesSet.erase(page))
        manager->triePages.erase(std::remove(manager->triePages.begin(),
                                              manager->triePages.end(), page),
                                 manager->triePages.end());
    // Keep the identity path as a hole. L2 may still own this logical page or
    // a descendant after rollback; rebuilding it would allocate in a failure
    // destructor and could strand the GPU owner under memory pressure.
    node->pageId = -1;
    node->producerEvent.reset();
    ++manager->evictionEpoch;
}

static void Qwen35RestoreSpillTrieHole(PagedCacheManager *manager,
        const std::vector<int> &tokens, const std::vector<std::string> *extraKeys,
        int pages) {
    std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
    CacheTrieNode *parent = manager->trieRoot;
    uint64_t prefixHash = 0;
    for (int p = 0; p < pages && parent; ++p) {
        const std::string extra = extraKeys ? (*extraKeys)[p] : std::string();
        const auto edge = Qwen35SpillEdgeHash(tokens.data() + p * manager->pageLen,
                                            manager->pageLen, extra);
        prefixHash = PagedCacheManager::HashTokenPrefixPage(
            prefixHash, tokens.data() + p * manager->pageLen, manager->pageLen, extra);
        auto it = parent->children.find(edge);
        if (it == parent->children.end()) {
            auto node = new CacheTrieNode();
            node->parent = parent;
            node->edgeHash = edge;
            node->extraKey = extra;
            node->pageHash = prefixHash;
            parent->children[edge] = node;
            parent = node;
        } else {
            parent = it->second;
        }
    }
}

// A-6 fix E（仅日志）：S1 第 3 关运行时证据——AbortReclaim 的返回值与调用后的账本快照。
// 判据：result=1 found=1 tier=0(NONE) state=0(RESIDENT) l1_page=-1 roots>=1。
static void Qwen35LogAbortReclaim(Qwen35SpillState &state, const Qwen35SpillTask &task,
                                  const char *reason, bool aborted) {
    PrefixCachePageSnapshot after;
    const bool found = state.ownerModel->pagedPageLedger->Snapshot(task.ledgerIdentity, after);
    PrefixCacheDebug("[PrefixCache] abort-reclaim: serial=%llu reason=%s result=%d found=%d "
                     "tier=%d state=%d durable=%d l1_page=%lld roots=%zu.\n",
                     (unsigned long long)(task.recycleTicket ? task.recycleTicket->serial : 0),
                     reason ? reason : "", aborted ? 1 : 0, found ? 1 : 0,
                     found ? (int)after.runtimeTier : -1, found ? (int)after.runtimeState : -1,
                     found ? (int)after.durableState : -1, found ? (long long)after.l1Page : -1LL,
                     found ? after.durableCheckpointRoots.size() : (size_t)0);
}

static void Qwen35CompleteSpillAdmissionFailure(
        Qwen35SpillState &state, Qwen35SpillTask &task, const char *reason,
        Qwen35SpillCopyState terminalState) {
    task.copyState = terminalState;
    task.terminalReason = reason == nullptr ? "spill_admission_failed" : reason;
    PrefixCacheDebug("[PrefixCache] spill-admission: serial=%llu generation=%llu "
                     "bytes=%zu result=%s reason=%s.\n",
                     (unsigned long long)(task.recycleTicket ? task.recycleTicket->serial : 0),
                     (unsigned long long)(task.recycleTicket ? task.recycleTicket->generation : 0),
                     task.bytes,
                     terminalState == Qwen35SpillCopyState::QueueFull ? "DEFER" : "failed",
                     task.terminalReason.c_str());
    if (task.ledgerLease.Valid() && state.ownerModel && state.ownerModel->pagedPageLedger) {
        const bool aborted = state.ownerModel->pagedPageLedger->AbortReclaim(
            task.ledgerLease, task.terminalReason.c_str());
        Qwen35LogAbortReclaim(state, task, task.terminalReason.c_str(), aborted);
    }
    if (task.manager && task.recycleTicket)
        task.manager->CompleteRecyclePage(task.recycleTicket, false,
                                           task.terminalReason.c_str());
}

static void Qwen35SpillWorkerOnce(Qwen35SpillState &state, std::vector<Qwen35SpillTask> &tasks) {
    struct CompleteLeases {
        std::vector<Qwen35SpillTask> &tasks;
        Qwen35SpillState &state;
        std::vector<uint8_t> completed;
        explicit CompleteLeases(std::vector<Qwen35SpillTask> &tasks,
                                Qwen35SpillState &state)
            : tasks(tasks), state(state), completed(tasks.size(), 0) {}
        bool Complete(size_t index, bool success,
                      const char *failureReason = nullptr) {
            if (index >= tasks.size() || completed[index]) return false;
            completed[index] = 1;
            const auto &task = tasks[index];
            bool committedL2 = success && !task.ledgerLease.Valid();
            // Identity/cancel rejection is a clean abort: never publish, but
            // keep the logical prefix cacheable. Only a real copy or commit
            // failure may quarantine the identity.
            bool cleanAbort = false;
            if (task.ledgerLease.Valid()) {
                if (success) {
                    PrefixCachePageSnapshot snapshot;
                    const auto ownerLease = std::static_pointer_cast<
                        const PagedPageOwnerLease>(task.owner_lease);
                    if (!task.HasCompleteIdentity() || !ownerLease ||
                        ownerLease->requestId != task.request_id ||
                        ownerLease->generation != task.request_generation ||
                        ownerLease->cancelled.load(std::memory_order_acquire) ||
                        !state.ownerModel->pagedPageLedger->Snapshot(task.ledgerIdentity, snapshot) ||
                        snapshot.pageGeneration != task.page_generation ||
                        snapshot.cacheIdentity != task.cache_identity) {
                        success = false;
                        cleanAbort = true;
                        failureReason = failureReason ? failureReason : "stale_page_generation";
                    }
                }
                if (success) {
                    committedL2 = state.ownerModel->pagedPageLedger->CommitL2(
                            task.ledgerLease, task.ledgerAllocation);
                    if (!committedL2) {
                        success = false;
                        state.ownerModel->pagedPageLedger->FailMigration(
                            task.ledgerLease, failureReason ? failureReason :
                                "ledger host publication failed");
                    }
                } else if (cleanAbort) {
                    const char *why = failureReason ? failureReason : "identity rejected";
                    const bool aborted = state.ownerModel->pagedPageLedger->AbortReclaim(
                        task.ledgerLease, why);
                    Qwen35LogAbortReclaim(state, task, why, aborted);
                } else {
                    state.ownerModel->pagedPageLedger->FailMigration(
                        task.ledgerLease, failureReason ? failureReason :
                            "spill migration failed");
                }
            }
            auto &terminalTask = tasks[index];
            terminalTask.copyState = success ? Qwen35SpillCopyState::CopyCommitted :
                Qwen35SpillCopyState::CopyFailed;
            terminalTask.terminalReason = success ? std::string() :
                (failureReason ? failureReason : "spill migration failed");
            PrefixCacheDebug("[PrefixCache] recycle-commit-l2: serial=%llu page=%d "
                             "generation=%llu ordinal=%llu ledger_allocation=%llu "
                             "bytes=%zu result=%s reason=%s.\n",
                             (unsigned long long)(task.recycleTicket ? task.recycleTicket->serial : 0),
                             task.pageId,
                             (unsigned long long)(task.recycleTicket ? task.recycleTicket->generation : 0),
                             (unsigned long long)(task.recycleTicket ? task.recycleTicket->pageOrdinal : 0),
                             (unsigned long long)task.ledgerAllocation,
                             task.manager && task.manager->maxPages > 0 ?
                                 task.manager->GetBytes() / (size_t)task.manager->maxPages : 0,
                             committedL2 ? "success" : "failed",
                             failureReason ? failureReason : "");
            if (task.manager && task.recycleTicket)
                task.manager->CompleteRecyclePage(task.recycleTicket, success,
                    failureReason);
            return success;
        }
        ~CompleteLeases() {
            for (size_t i = 0; i < tasks.size(); ++i) Complete(i, false);
        }
        bool IsCompleted(size_t index) const {
            return index < completed.size() && completed[index] != 0;
        }
    } leases{tasks, state};
    if (Qwen35SpillStatsEnabled())
        Qwen35SpillStatsRef().workerTasks.fetch_add(tasks.size(), std::memory_order_relaxed);
#ifdef USE_CUDA
    std::vector<std::shared_ptr<Qwen35SpilledPage>> copies;
    std::vector<size_t> offsets;
    size_t total = 0;
    for (auto &task : tasks) {
        auto manager = task.manager;
        auto copy = std::make_shared<Qwen35SpilledPage>();
        copy->dataType = manager->dataType;
        copy->pageLen = manager->pageLen;
        copy->heads = manager->dims[2];
        copy->dim = manager->dims[3];
        copy->writeStatus = task.writeStatus;
        copy->ledgerIdentity = task.ledgerIdentity;
        copy->ledgerAllocation = task.ledgerAllocation;
        if (!task.completeOnly && !task.reject)
            copy->hostBytes.resize(manager->GetBytes() / (size_t)manager->maxPages);
        offsets.push_back(total);
        total += copy->hostBytes.size();
        copies.push_back(std::move(copy));
    }
        const int oldDevice = FastllmCudaGetDevice();
        struct RestoreDevice { int old; ~RestoreDevice() { FastllmCudaSetDevice(old); } } restore{oldDevice};
        FastllmCudaSetDevice(tasks.front().device);
        void *stream = state.storeStreams.at(tasks.front().device);
    std::set<void*> events;
    std::vector<uint8_t> injected(tasks.size(), 0);
    const uint64_t injectedSerial = Qwen35InjectedD2HFailureSerial();
    for (size_t i = 0; i < tasks.size(); ++i) {
        const auto &task = tasks[i];
        if (task.reject) {
            leases.Complete(i, false);
            continue;
        }
        if (task.completeOnly) {
            leases.Complete(i, true);
            continue;
        }
        if ((Qwen35InjectAllD2HFailures() || injectedSerial != 0) &&
            task.recycleTicket &&
            (Qwen35InjectAllD2HFailures() || task.recycleTicket->serial == injectedSerial)) {
            injected[i] = 1;
            PrefixCacheDebug("[PrefixCache] recycle-d2h-failure-injected: serial=%llu "
                             "page=%d generation=%llu reason=d2h_injected_failure.\n",
                             (unsigned long long)task.recycleTicket->serial, task.pageId,
                             (unsigned long long)task.recycleTicket->generation);
            leases.Complete(i, false, "d2h_injected_failure");
            continue;
        }
        const size_t bytes = copies[i]->hostBytes.size();
        tasks[i].copyState = Qwen35SpillCopyState::Copying;
        const bool producerEventWait = task.producerReady &&
            events.insert(task.producerReady.get()).second;
        if (producerEventWait)
            FastllmCudaStreamWaitEvent(stream, task.producerReady.get());
        PrefixCacheDebug("[PrefixCache] recycle-d2h-submit: serial=%llu page=%d "
                         "generation=%llu ordinal=%llu producer_event_wait=%d bytes=%zu.\n",
                         (unsigned long long)(task.recycleTicket ? task.recycleTicket->serial : 0),
                         task.pageId,
                         (unsigned long long)(task.recycleTicket ? task.recycleTicket->generation : 0),
                         (unsigned long long)(task.recycleTicket ? task.recycleTicket->pageOrdinal : 0),
                         producerEventWait ? 1 : 0, bytes);
        if (!FastllmCudaCopyFromDeviceToPinnedHostAsync(
                (uint8_t *)state.storeHost + offsets[i],
                (uint8_t *)task.manager->cudaData + (uint64_t)task.pageId * bytes, bytes, stream)) {
            { RequestDiagnosticCurrentOperation diagnosticCompletion("l2_cuda_stream_completion_wait",
                    RequestDiagnosticResource("CUDA.stream", stream).c_str()); FastllmCudaStreamSynchronize(stream); }
            throw std::runtime_error("spill async D2H submission failed");
        }
    }
    { RequestDiagnosticCurrentOperation diagnosticCompletion("l2_cuda_stream_completion_wait",
                    RequestDiagnosticResource("CUDA.stream", stream).c_str()); FastllmCudaStreamSynchronize(stream); } // Only this background transfer stream.
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (tasks[i].reject || tasks[i].completeOnly || injected[i] ||
            leases.IsCompleted(i)) continue;
        auto &copy = copies[i];
        const uint64_t bytes = copy->hostBytes.size();
        PrefixCacheDebug("[PrefixCache] recycle-d2h-complete: serial=%llu page=%d "
                         "generation=%llu bytes=%llu.\n",
                         (unsigned long long)(tasks[i].recycleTicket ? tasks[i].recycleTicket->serial : 0),
                         tasks[i].pageId,
                         (unsigned long long)(tasks[i].recycleTicket ? tasks[i].recycleTicket->generation : 0),
                         (unsigned long long)bytes);
        memcpy(copy->hostBytes.data(), (uint8_t *)state.storeHost + offsets[i], bytes);
        std::unique_lock<std::mutex> lock(state.poolMutex);
        if (!copy->writeStatus) {
            auto status = state.writeStatuses.find(std::get<0>(tasks[i].key));
            if (status != state.writeStatuses.end()) copy->writeStatus = status->second;
        }
        auto old = state.pool.find(tasks[i].key);
        if (old != state.pool.end()) {
            if (Qwen35SpillPageMatches(*old->second, tasks[i].manager)) {
                // The verified immutable page is already backed up. Retain a
                // ready SSD import instead of replacing it with a duplicate.
                const auto oldPage = old->second;
                state.lru.splice(state.lru.end(), state.lru, old->second->lruPosition);
                lock.unlock();
                // CompleteRecyclePage takes the manager's page-index lock;
                // never acquire it while poolMutex is held (promotion and
                // restore can take the two locks in the opposite order).
                if (!leases.Complete(i, true)) {
                    // CommitL2 failure quarantines the GPU victim. Do not leave
                    // the duplicate host object advertising a usable L2 copy
                    // for the same failed ledger identity.
                    std::lock_guard<std::mutex> rollbackLock(state.poolMutex);
                    oldPage->ledgerPublished.store(false, std::memory_order_release);
                    if (oldPage->promotionRefs.load(std::memory_order_acquire) == 0)
                        Qwen35RemoveSpilledPageLocked(state, tasks[i].key, oldPage);
                }
                continue;
            }
            if (old->second->promotionRefs.load(std::memory_order_acquire) != 0) {
                // A promotion owns this host copy.  Do not replace it with a
                // concurrent recycle payload; the canonical page remains
                // available to the request until GPU adoption finishes.
                const auto oldPage = old->second;
                lock.unlock();
                if (!leases.Complete(i, true))
                    oldPage->ledgerPublished.store(false, std::memory_order_release);
                continue;
            }
            Qwen35RemoveSpilledPageLocked(state, tasks[i].key, old->second);
        }
        // This worker keeps its GPU page lease while waiting. The scheduler
        // never waits for the SSD ack and an unpersisted last copy is retained.
        // A page with no durable L3 root is intentionally ineligible for L2
        // eviction.  Waiting on poolChanged here cannot make that predicate
        // true and used to deadlock the spill worker (and server shutdown)
        // forever.  Complete the recycle ticket as a terminal, quarantined
        // failure so the caller can fall back or report capacity_unavailable.
        if (!Qwen35SpillEvictFor(state, bytes)) {
            lock.unlock();
            leases.Complete(i, false, "l2_capacity_no_durable_victim");
            lock.lock();
            continue;
        }
        copy->lastUse = ++state.clock;
        state.lru.push_back(tasks[i].key);
        copy->lruPosition = std::prev(state.lru.end());
        copy->promotionRefs.store(1, std::memory_order_release);
        try {
            const auto inserted = state.pool.emplace(tasks[i].key, copy);
            if (!inserted.second) {
                copy->promotionRefs.store(0, std::memory_order_release);
                state.lru.erase(copy->lruPosition);
                lock.unlock();
                leases.Complete(i, false);
                continue;
            }
        } catch (...) {
            copy->promotionRefs.store(0, std::memory_order_release);
            state.lru.erase(copy->lruPosition);
            lock.unlock();
            leases.Complete(i, false);
            throw;
        }
        state.bytes += bytes;
        if (Qwen35SpillStatsEnabled())
            Qwen35SpillStatsRef().poolBytes.fetch_add(bytes, std::memory_order_relaxed);
        lock.unlock();
        // Publish the unified ledger without holding poolMutex. If adoption
        // fails, remove this exact host object and quarantine the GPU victim;
        // no unusable L2 entry remains behind. The temporary promotion ref
        // prevents another worker from evicting the object while the ledger
        // commit runs without poolMutex.
        if (!leases.Complete(i, true)) {
            std::lock_guard<std::mutex> rollbackLock(state.poolMutex);
            copy->promotionRefs.store(0, std::memory_order_release);
            Qwen35RemoveSpilledPageLocked(state, tasks[i].key, copy);
            continue;
        }
        copy->ledgerPublished.store(true, std::memory_order_release);
        copy->promotionRefs.fetch_sub(1, std::memory_order_acq_rel);
        if (Qwen35SpillStatsEnabled()) {
            auto &stats = Qwen35SpillStatsRef();
            stats.spillSuccessPages.fetch_add(1, std::memory_order_relaxed);
        }
    }
#endif
}

static void Qwen35SpillWorkerMain(std::shared_ptr<Qwen35SpillState> state) {
    for (;;) {
        std::vector<Qwen35SpillTask> tasks;
        {
            std::unique_lock<std::mutex> lock(state->queueMutex);
            auto ready = [&] { return state->stopping || !state->queue.empty(); };
            if (Qwen35SpillStatsEnabled()) {
                if (!state->queueCond.wait_for(lock, std::chrono::seconds(1), ready)) {
                    lock.unlock();
                    Qwen35PrintSpillStats("worker_idle");
                    continue;
                }
            } else {
                state->queueCond.wait(lock, ready);
            }
            if (state->queue.empty() && state->stopping) break;
            const int device = state->queue.front().device;
            size_t bytes = 0;
            if (state->activeCopySlots >= kQwen35SpillCopySlots) continue;
            ++state->activeCopySlots;
            while (!state->queue.empty() && state->queue.front().device == device &&
                   tasks.size() < kQwen35SpillQueueJobs) {
                const auto &task = state->queue.front();
                size_t pageBytes = task.manager->GetBytes() / (size_t)task.manager->maxPages;
                if (!tasks.empty() && bytes + pageBytes > state->stagingBytes) break;
                bytes += pageBytes;
                tasks.push_back(std::move(state->queue.front()));
                state->queue.pop_front();
            }
        }
        try {
            Qwen35SpillWorkerOnce(*state, tasks);
        } catch (const std::exception &error) {
            fprintf(stderr, "[PrefixCache] spill failed: %s.\n", error.what());
            for (auto &task : tasks) {
                if (task.copyState != Qwen35SpillCopyState::CopyCommitted) {
                    task.copyState = Qwen35SpillCopyState::WorkerExit;
                    task.terminalReason = "worker_exit";
                }
            }
        } catch (...) {
            fprintf(stderr, "[PrefixCache] spill failed: unknown worker exception.\n");
            for (auto &task : tasks) {
                if (task.copyState != Qwen35SpillCopyState::CopyCommitted) {
                    task.copyState = Qwen35SpillCopyState::WorkerExit;
                    task.terminalReason = "worker_exit";
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(state->queueMutex);
            if (state->activeCopySlots > 0) --state->activeCopySlots;
            for (auto &task : tasks) {
                state->pending.erase(task.key);
                if (!task.bytes || state->pendingJobs == 0 || state->pendingBytes < task.bytes) {
                    state->pendingAccountingError = true;
                    state->pendingJobs = 0;
                    state->pendingBytes = 0;
                } else {
                    --state->pendingJobs;
                    state->pendingBytes -= task.bytes;
                }
            }
        }
        Qwen35PrintSpillStats("worker");
    }
    Qwen35PrintSpillStats("worker_stopped", true);
}

static void Qwen35StopSpillWorker(Qwen3_5Model *model) {
    auto state = model->pagedSpillState;
    if (!state) return;
    state->shutdownState.store(Qwen35SpillShutdownState::Stopping,
                               std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(state->queueMutex);
        state->stopping = true;
    }
    state->queueCond.notify_all();
    state->poolChanged.notify_all();
    if (state->worker.joinable()) state->worker.join();
    {
        std::lock_guard<std::mutex> lock(state->queueMutex);
        const bool drained = state->queue.empty() && state->pendingJobs == 0 &&
            state->pendingBytes == 0 && state->activeCopySlots == 0;
        state->shutdownState.store(drained ? Qwen35SpillShutdownState::Drained :
                                   Qwen35SpillShutdownState::Timeout,
                                   std::memory_order_release);
        PrefixCacheDebug("[PrefixCache] spill-shutdown: result=%s pending_jobs=%zu "
                         "pending_bytes=%zu active_copy_slots=%zu.\n",
                         drained ? "shutdown_drained" : "shutdown_timeout",
                         state->pendingJobs, state->pendingBytes,
                         state->activeCopySlots);
    }
    for (int layer = 0; layer < model->block_cnt; ++layer) {
        for (bool key : {true, false}) {
            for (auto ref : model->GetPagedKVCacheManagers(layer, key)) {
                if (!ref.second || !state->managers.count(ref.second)) continue;
                // The worker has completed (or quarantined) every queued
                // recycle ticket before the callback is disabled.
                ref.second->DisableDeferredRecycle();
                std::lock_guard<std::mutex> lock(ref.second->pageIndexLocker);
                ref.second->onRecordPage = {};
                ref.second->onRecyclePage = {};
            }
        }
    }
    if (Qwen35SpillStatsEnabled())
        Qwen35SpillStatsRef().poolBytes.fetch_sub(state->bytes, std::memory_order_relaxed);
#ifdef USE_CUDA
    const int old = FastllmCudaGetDevice();
    for (auto &entry : state->storeStreams) {
        FastllmCudaSetDevice(entry.first);
        FastllmCudaStreamDestroy(entry.second);
        FastllmCudaStreamDestroy(state->loadStreams.at(entry.first));
    }
    FastllmCudaHostFree(state->storeHost);
    FastllmCudaHostFree(state->loadHost);
    FastllmCudaSetDevice(old);
#endif
    model->pagedSpillState.reset();
}

static void Qwen35EnsureSpillState(Qwen3_5Model *model) {
#ifdef USE_CUDA
    if (!model->pagedSpillState) model->pagedSpillState = std::make_shared<Qwen35SpillState>();
    auto state = model->pagedSpillState;
    state->ownerModel = model;
    if (!state->storeHost && !state->worker.joinable()) {
        size_t maxPageBytes = 0;
        const int old = FastllmCudaGetDevice();
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            for (bool key : {true, false}) for (auto ref : model->GetPagedKVCacheManagers(layer, key)) {
                if (!ref.second || !ref.second->cudaData) continue;
                maxPageBytes = std::max(maxPageBytes, ref.second->GetBytes() / (size_t)ref.second->maxPages);
                const int device = GetPointerDeviceId(ref.second->cudaData);
                if (device < 0 || state->storeStreams.count(device)) continue;
                FastllmCudaSetDevice(device);
                state->storeStreams[device] = FastllmCudaStreamCreateLowestPriority();
                state->loadStreams[device] = FastllmCudaStreamCreateLowestPriority();
            }
        }
        if (maxPageBytes) {
            state->stagingBytes = maxPageBytes * std::max<size_t>(1,
                std::min<size_t>(128, (8 << 20) / maxPageBytes));
            state->storeHost = FastllmCudaHostMalloc(state->stagingBytes);
            state->loadHost = FastllmCudaHostMalloc(state->stagingBytes);
        }
        FastllmCudaSetDevice(old);
    }
#endif
}

static void Qwen35InstallSpillHooks(Qwen3_5Model *model) {
    if (model == nullptr || !Qwen35L2HostKVEnabled()) return;
#ifdef USE_CUDA
    Qwen35EnsureSpillState(model);
    auto state = model->pagedSpillState;
    if (!state->worker.joinable()) state->worker = std::thread(Qwen35SpillWorkerMain, state);
    for (int layer = 0; layer < model->block_cnt; ++layer) {
        for (int side = 0; side < 2; ++side) {
            for (auto ref : model->GetPagedKVCacheManagers(layer, side == 0)) {
                auto manager = ref.second;
                if (!manager || !manager->cudaData || manager->dims.size() != 4) continue;
                if (state->managers.count(manager)) continue;
                // Reserve one replacement page before installing the callback;
                // the callback runs outside the manager lock and only queues a
                // ticket, so the victim cannot be handed to a new writer until
                // the worker acknowledges host ownership.
                manager->EnableDeferredRecycle();
                const int device = GetPointerDeviceId(manager->cudaData);
                std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
                state->managers.insert(manager);
                // Record is metadata only. D2H is performed exclusively after
                // the allocator creates a deferred recycle ticket.
                manager->onRecordPage = [state, model, manager, layer, side, device](
                        int id, uint64_t hash) {
                    if (state->directPublishActive.load(std::memory_order_acquire)) return;
                    auto found = manager->pageToTrieNode.find(id);
                    if (found == manager->pageToTrieNode.end() || !found->second) return;
                    const auto *node = found->second;
                    uint64_t ordinal = 0;
                    for (const auto *cursor = node; cursor && cursor->parent; cursor = cursor->parent)
                        ++ordinal;
                    if (ordinal) --ordinal;
                    const auto identity = Qwen35LedgerPageIdentity(
                        model, manager, layer, side, device, hash,
                        node->parent ? node->parent->pageHash : 0, ordinal, node->extraKey);
                    const uint64_t generation = id >= 0 && id < manager->maxPages ?
                        static_cast<uint64_t>(manager->pageTimestamp[id]) : 0;
                    // This hook is metadata-only. The actual producer event is
                    // attached to the recycle ticket, never a stale scheduler
                    // event sampled here.
                    const bool observed = model->pagedPageLedger->ObserveL1(
                        identity, manager, id, generation, node->producerEvent);
                    if (!observed) {
                        // Record owns the page-index lock and performs the
                        // structural rollback after this callback throws. Do
                        // not mutate/delete the node here: doing so would
                        // leave Record unable to return the picked page to its
                        // free pool without a refcount leak.
                        throw std::runtime_error(
                            "paged cache ledger rejected L1 observation");
                    }
                };
                manager->onRecyclePage = [state, model, manager, layer, side, device](
                        int id, uint64_t hash,
                        const std::shared_ptr<PagedCacheManager::RecycleTicket> &ticket) {
                    if (Qwen35SpillStatsEnabled())
                        Qwen35SpillStatsRef().onRecordPageCalls.fetch_add(1, std::memory_order_relaxed);
                    const Qwen35SpillKey key{hash, layer, side, device};
                    const uint64_t bytes = manager->GetBytes() / (size_t)manager->maxPages;
                    // A recycle ticket must carry the event recorded on the
                    // actual producer stream.  The periodic state event is
                    // intentionally not a fallback: using it can read a page
                    // after the allocator has already reused the stream.
                    const auto producer = ticket ? ticket->producerEvent :
                        std::shared_ptr<void>();
                    const auto ledgerIdentity = Qwen35LedgerPageIdentity(
                        model, manager, layer, side, device, hash,
                        ticket ? ticket->parentHash : 0,
                        ticket ? ticket->pageOrdinal : 0,
                        ticket ? ticket->extraKey : std::string());
                    std::shared_ptr<Qwen35PrefixWriteStatus> writeStatus;
                    {
                        // Keep the lock order consistent with promotion and
                        // eviction: pool state is inspected before the ledger
                        // is changed, never while queueMutex is held.
                        std::lock_guard<std::mutex> host(state->poolMutex);
                        auto found = state->writeStatuses.find(hash);
                        if (found != state->writeStatuses.end()) writeStatus = found->second;
                    }
                    PrefixCachePageLedger::Lease ledgerLease;
                    if (ticket) {
                        ledgerLease = model->pagedPageLedger->BeginReclaim(
                            ledgerIdentity, ticket->serial,
                            producer);
                    }
                    const uint64_t ledgerAllocation = state->nextLedgerAllocation.fetch_add(1);
                    PrefixCacheDebug("[PrefixCache] recycle-ticket: serial=%llu page=%d "
                                     "generation=%llu ordinal=%llu producer_event=%d "
                                     "ledger_allocation=%llu bytes=%llu.\n",
                                     (unsigned long long)(ticket ? ticket->serial : 0),
                                     id,
                                     (unsigned long long)(ticket ? ticket->generation : 0),
                                     (unsigned long long)(ticket ? ticket->pageOrdinal : 0),
                                     producer ? 1 : 0,
                                     (unsigned long long)ledgerAllocation,
                                     (unsigned long long)bytes);
                    // A fixed task-count limit must not silently discard a
                    // GPU->RAM backup. Invalid/oversized pages are queued as
                    // rejection tickets so the worker quarantines them via
                    // CompleteRecyclePage(false), never reusing the page.
                    const bool reject = !ticket || !bytes || !ledgerLease.Valid() ||
                        !producer ||
                        (Qwen35L2HostKVPoolLimitBytes() != 0 &&
                         bytes > Qwen35L2HostKVPoolLimitBytes());
                    // Do not drop a ticket when this key is already queued:
                    // two physical victims can carry the same immutable
                    // logical page, and each migration lease still needs a
                    // terminal CompleteRecyclePage acknowledgement.
                    {
                        Qwen35SpillTask task{manager, key, id, device, producer, ticket, false, reject,
                            std::move(writeStatus), std::move(ledgerLease), ledgerAllocation, ledgerIdentity};
                        task.save_job_id = ticket ? ticket->serial : 0;
                        // A-3: consume the owner identity recorded when the
                        // page was published by its request. A recycle serial
                        // is only the save job id. A missing owner is a hard
                        // admission failure, never a guessed identity.
                        PagedCacheManager::RecycleRequestIdentity ownerIdentity;
                        const bool haveOwner = model != nullptr &&
                            model->PeekPagedPageOwner(manager, id, ownerIdentity);
                        task.request_id = haveOwner ? ownerIdentity.requestId : 0;
                        task.request_generation =
                            haveOwner ? ownerIdentity.requestGeneration : 0;
                        task.owner_lease = haveOwner ? ownerIdentity.ownerLease
                                                     : std::shared_ptr<const void>();
                        task.page_generation = ticket ? ticket->generation : 0;
                        task.copy_generation = ledgerAllocation;
                        task.cache_identity = ledgerIdentity.CanonicalKey();
                        task.bytes = bytes;
                        if (!task.HasCompleteIdentity()) {
                            Qwen35CompleteSpillAdmissionFailure(
                                *state, task, "missing_identity",
                                Qwen35SpillCopyState::CopyFailed);
                            return;
                        }
                        if (producer) task.copyState = Qwen35SpillCopyState::ProducerRecorded;
                        if (producer && task.HasCompleteIdentity())
                            task.copyState = Qwen35SpillCopyState::FreezeComplete;
                        bool admissionFailed = false;
                        bool injectedAdmissionFailure = false;
                        const char *admissionReason = nullptr;
                        Qwen35SpillCopyState admissionState = Qwen35SpillCopyState::QueueFull;
                        {
                            std::lock_guard<std::mutex> guard(state->queueMutex);
                            if (Qwen35InjectSpillAdmissionFailure()) {
                                admissionFailed = true;
                                injectedAdmissionFailure = true;
                                admissionReason = "test_injected_admission_failure";
                                admissionState = Qwen35SpillCopyState::CopyFailed;
                            } else if (state->stopping) {
                                admissionFailed = true;
                                admissionReason = "shutdown";
                                admissionState = Qwen35SpillCopyState::CopyFailed;
                            } else if (!state->stagingBytes || bytes > state->stagingBytes) {
                                admissionFailed = true;
                                admissionReason = "task_too_large";
                                admissionState = Qwen35SpillCopyState::CopyFailed;
                            } else if (state->pendingJobs >= kQwen35SpillQueueJobs ||
                                       bytes > kQwen35SpillPendingBytes ||
                                       state->queue.size() >= kQwen35SpillQueueJobs ||
                                       state->pendingBytes > kQwen35SpillPendingBytes - bytes) {
                                admissionFailed = true;
                                admissionReason = "queue_full";
                                admissionState = Qwen35SpillCopyState::QueueFull;
                            } else if (!model ||
                                       !model->ErasePagedPageOwner(manager, id, ownerIdentity)) {
                                admissionFailed = true;
                                admissionReason = "owner_changed";
                                admissionState = Qwen35SpillCopyState::CopyFailed;
                            } else {
                                task.copyState = Qwen35SpillCopyState::Queued;
                                state->pending.insert(key);
                                ++state->pendingJobs;
                                state->pendingBytes += bytes;
                                state->queue.push_back(std::move(task));
                            }
                        }
                        if (admissionFailed) {
                            if (injectedAdmissionFailure)
                                PrefixCacheDebug("[PrefixCache] test-spill-admission-failure: serial=%llu.\n",
                                                 (unsigned long long)(task.recycleTicket ?
                                                     task.recycleTicket->serial : 0));
                            Qwen35CompleteSpillAdmissionFailure(
                                *state, task, admissionReason, admissionState);
                            return;
                        }
                    }
                    // Admission is complete.  Do not wait or join here: the
                    // producer returns to the scheduler while the worker waits
                    // on the recorded event and performs the copy.
                    state->queueCond.notify_one();
                };
                if (Qwen35SpillStatsEnabled())
                    Qwen35SpillStatsRef().hookRegistrations.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    Qwen35PrintSpillStats("hooks_installed", true);
#endif
}

void Qwen3_5Model::InstallPagedSpillHooks() { Qwen35InstallSpillHooks(this); }

void Qwen3_5Model::ReleasePagedPrefixRestoreOwners(ResponseContext *context) {
    if (!context || !pagedSpillState) return;
    std::lock_guard<std::mutex> lock(pagedSpillState->promoteMutex);
    pagedSpillState->adoptedGpuOwners.erase(context);
}

void Qwen3_5Model::ReleasePagedSpillRetryHolds(ResponseContext *context) {
    if (!context || !pagedSpillState) return;
    std::lock_guard<std::mutex> lock(pagedSpillState->promoteMutex);
    std::lock_guard<std::mutex> poolLock(pagedSpillState->poolMutex);
    auto found = pagedSpillState->retryPromotionHolds.find(context);
    if (found == pagedSpillState->retryPromotionHolds.end()) return;
    for (const auto &key : found->second)
        pagedSpillState->promotionHolds.erase(key);
    pagedSpillState->retryPromotionHolds.erase(found);
}

void Qwen3_5Model::AdoptPagedPrefixRestoreOwners(
        ResponseContext *context,
        std::vector<std::pair<PagedCacheManager *, std::vector<int>>> pages) {
    if (!context || pages.empty() || !pagedSpillState) return;
    auto owner = std::make_shared<Qwen35GpuPrefixLease>();
    owner->pages = std::move(pages);
    std::lock_guard<std::mutex> lock(pagedSpillState->promoteMutex);
    pagedSpillState->adoptedGpuOwners[context] = std::move(owner);
}

void Qwen3_5Model::RecordPagedPrefixProducerEvents() {
#ifdef USE_CUDA
    if (Qwen35L2HostKVEnabled()) InstallPagedSpillHooks();
    auto state = pagedSpillState;
    if (!state) {
        state = std::make_shared<Qwen35SpillState>();
        state->ownerModel = this;
        pagedSpillState = state;
    }
    std::map<int,std::shared_ptr<void>> events;
    std::mutex eventMutex;
    auto record = [&](int device) {
        const int old = FastllmCudaGetDevice();
        FastllmCudaSetDevice(device);
        auto event = std::shared_ptr<void>(FastllmCudaEventCreate(),[device](void *event) {
            const int old = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device); FastllmCudaEventDestroy(event); FastllmCudaSetDevice(old);
        });
        if (!event) {
            FastllmCudaSetDevice(old);
            throw std::runtime_error("paged producer event allocation failed");
        }
        FastllmCudaEventRecordCurrentThread(event.get());
        FastllmCudaSetDevice(old);
        std::lock_guard<std::mutex> lock(eventMutex); events[device] = std::move(event);
    };
    // Copy the existing persistent exporter protocol: record on the actual TP
    // producer threads, never on an unrelated scheduler stream.
    if (threadTpPreparedDevices.size() > 1 && threadTpWorkerGroup.HasWorkers()) {
        std::vector<std::exception_ptr> errors(threadTpPreparedDevices.size());
        threadTpWorkerGroup.RunWithCaller(threadTpPreparedDevices,[&](int rank) {
            record(threadTpPreparedDevices[rank]);
        },errors);
        for (auto &error : errors) if (error) std::rethrow_exception(error);
    } else {
        std::set<int> devices;
        for (int layer = 0; layer < block_cnt; ++layer)
            for (bool key : {true, false})
                for (const auto &ref : GetPagedKVCacheManagers(layer, key))
                    if (ref.second && ref.second->cudaData) {
                        const int device = GetPointerDeviceId(ref.second->cudaData);
                        if (device >= 0) devices.insert(device);
                    }
        for (const auto &stream : state->storeStreams) devices.insert(stream.first);
        for (int device : devices) record(device);
    }
    std::set<PagedCacheManager *> managers;
    for (int layer = 0; layer < block_cnt; ++layer)
        for (bool key : {true, false})
            for (const auto &ref : GetPagedKVCacheManagers(layer, key))
                if (ref.second && ref.second->cudaData) managers.insert(ref.second);
    for (auto *manager : managers) {
        const int device = GetPointerDeviceId(manager->cudaData);
        auto found = events.find(device);
        if (found == events.end() || !found->second)
            throw std::runtime_error("paged producer event missing");
        std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
        manager->recordProducerEvent = found->second;
    }
#endif
}

void Qwen3_5Model::RecordPagedPrefixProducerEventsOnCurrentThread() {
#ifdef USE_CUDA
    if (Qwen35L2HostKVEnabled()) InstallPagedSpillHooks();
    auto state = pagedSpillState;
    if (!state) {
        state = std::make_shared<Qwen35SpillState>();
        state->ownerModel = this;
        pagedSpillState = state;
    }
    std::map<int, std::shared_ptr<void>> events;
    std::set<int> devices;
    std::set<PagedCacheManager *> managers;
    for (int layer = 0; layer < block_cnt; ++layer) {
        for (bool key : {true, false}) {
            for (const auto &ref : GetPagedKVCacheManagers(layer, key)) {
                if (!ref.second || !ref.second->cudaData) continue;
                managers.insert(ref.second);
                const int device = GetPointerDeviceId(ref.second->cudaData);
                if (device >= 0) devices.insert(device);
            }
        }
    }
    for (int device : devices) {
        const int old = FastllmCudaGetDevice();
        FastllmCudaSetDevice(device);
        auto event = std::shared_ptr<void>(FastllmCudaEventCreate(), [device](void *value) {
            const int oldDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            FastllmCudaEventDestroy(value);
            FastllmCudaSetDevice(oldDevice);
        });
        if (!event) {
            FastllmCudaSetDevice(old);
            throw std::runtime_error("paged producer event allocation failed");
        }
        FastllmCudaEventRecordCurrentThread(event.get());
        FastllmCudaSetDevice(old);
        events.emplace(device, std::move(event));
    }
    for (auto *manager : managers) {
        const int device = GetPointerDeviceId(manager->cudaData);
        auto found = events.find(device);
        if (found == events.end())
            throw std::runtime_error("paged current-thread producer event missing");
        std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
        manager->recordProducerEvent = found->second;
    }
#endif
}
static int Qwen35RestoreSpilledPages(Qwen3_5Model *model, ResponseContext *context,
                                    int wantTokens, int maxTokens, bool background) {
    if (Qwen35SpillStatsEnabled())
        Qwen35SpillStatsRef().restoreCalls.fetch_add(1, std::memory_order_relaxed);
    auto state = model ? model->pagedSpillState : nullptr;
    if (!state || !context || wantTokens <= 0 || wantTokens >= maxTokens) return 0;
#ifdef USE_CUDA
    const int pageLen = fastllm::GetPageLen();
    int cappedTokens = std::min(wantTokens, maxTokens);
    cappedTokens = std::min(cappedTokens, (int)context->currentTokens.size());
    const int wantPages = cappedTokens / pageLen;
    if (wantPages <= 0 || cappedTokens != wantTokens) return 0;
    const auto extraKeys = context->PrefixCachePageKeys();
    std::vector<uint64_t> identities;
    uint64_t hash = 0;
    for (int p = 0; p < wantPages; ++p) {
        if (extraKeys && p >= (int)extraKeys->size()) return 0;
        hash = PagedCacheManager::HashTokenPrefixPage(hash,
            context->currentTokens.data() + p * pageLen, pageLen,
            extraKeys ? (*extraKeys)[p] : std::string());
        identities.push_back(hash);
    }
    struct Plan {
        PagedCacheManager *manager = nullptr;
        int layer = -1, side = -1, device = -1;
        std::vector<int> ids, newIds;
        std::vector<std::shared_ptr<Qwen35SpilledPage>> host;
        std::vector<PrefixCachePageLedger::Lease> ledgerLeases;
        std::shared_ptr<void> h2dComplete;
    };
    std::vector<Plan> plans;
    std::vector<Qwen35SpillKey> heldKeys;
    bool pending = false;
    bool adopted = false;
    struct PromotionOwners {
        Qwen35SpillState &state; ResponseContext &context; std::vector<Plan> &plans;
        const std::vector<uint64_t> &identities;
        std::vector<Qwen35SpillKey> &heldKeys;
        bool &adopted; bool &pending;
        ~PromotionOwners() {
            uint64_t poolBefore = 0;
            size_t hostRemoved = 0;
            // Revoke failed GPU publications before returning the ledger and
            // host objects to L2 RESIDENT. A lookup must never see both owners.
            if (!adopted)
                for (auto &plan : plans)
                    for (int id : plan.newIds) Qwen35DropPromotedGpuPage(plan.manager, id);
            {
            std::lock_guard<std::mutex> lock(state.poolMutex);
            // Once the GPU batch has committed, this transaction is the owner
            // that is allowed to remove its host copies.  Drop its holds first
            // so the common remove helper still protects all other pending
            // promotions while permitting this commit's exact pages to leave
            // L2.
            if (adopted)
                for (const auto &key : heldKeys) state.promotionHolds.erase(key);
            poolBefore = state.bytes;
            for (auto &plan : plans) for (size_t p = 0; p < plan.host.size(); ++p) {
                auto &host = plan.host[p]; if (!host) continue;
                uint32_t refs = host->promotionRefs.load(std::memory_order_acquire);
                if (refs) host->promotionRefs.fetch_sub(1, std::memory_order_acq_rel);
                if (adopted && Qwen35RemoveSpilledPageLocked(state,
                    {identities[p], plan.layer, plan.side, plan.device}, host)) {
                    ++hostRemoved;
                }
                else if (p < plan.ledgerLeases.size() && plan.ledgerLeases[p].Valid())
                    state.ownerModel->pagedPageLedger->AbortPromotion(
                        plan.ledgerLeases[p]);
                if (!host->ledgerPublished.load(std::memory_order_acquire) &&
                    host->promotionRefs.load(std::memory_order_acquire) == 0)
                    Qwen35RemoveSpilledPageLocked(
                        state, {identities[p], plan.layer, plan.side, plan.device}, host);
            }
            if (pending) {
                state.retryPromotionHolds[&context] = heldKeys;
            } else if (!adopted) {
                for (const auto &key : heldKeys) state.promotionHolds.erase(key);
            }
            if (adopted) state.retryPromotionHolds.erase(&context);
            }
            for (auto &plan : plans)
                if (!plan.ids.empty()) plan.manager->ReleasePageIndices(plan.ids);
            if (!adopted && !pending) {
                for (const auto &plan : plans) {
                    auto gpu = Qwen35QuerySpillL1Pages(
                        plan.manager, context.currentTokens, context.PrefixCachePageKeys(),
                        (int)identities.size());
                    int visibleFirst = gpu.empty() ? -1 : gpu.front();
                    int visibleLast = gpu.empty() ? -1 : gpu.back();
                    PrefixCacheDebug("[PrefixCache] promotion-rollback: layer=%d side=%d "
                                     "rank=%d gpu_first_after=%d gpu_last_after=%d.\n",
                                     plan.layer, plan.side, plan.device,
                                     visibleFirst, visibleLast);
                }
                // The GPU endpoint lines above prove the trie is empty after
                // rollback.  Also snapshot the unified ledger and the host
                // allocation for every promoted page so a failed batch cannot
                // be mistaken for a partial owner transfer.
                for (const auto &plan : plans) {
                    for (size_t p = 0; p < plan.host.size(); ++p) {
                        if (!plan.host[p]) continue;
                        PrefixCachePageSnapshot snapshot;
                        const bool found = state.ownerModel &&
                            state.ownerModel->pagedPageLedger &&
                            state.ownerModel->pagedPageLedger->Snapshot(
                                plan.host[p]->ledgerIdentity, snapshot);
                        PrefixCacheDebug(
                            "[PrefixCache] promotion-rollback-ledger: layer=%d side=%d "
                            "rank=%d ordinal=%zu host_allocation=%llu found=%d tier=%d "
                            "state=%d durable=%d runtime_ref=%llu lease=%llu quarantined=%d.\n",
                            plan.layer, plan.side, plan.device, p,
                            (unsigned long long)plan.host[p]->ledgerAllocation,
                            found ? 1 : 0,
                            found ? (int)snapshot.runtimeTier : -1,
                            found ? (int)snapshot.runtimeState : -1,
                            found ? (int)snapshot.durableState : -1,
                            found ? (unsigned long long)snapshot.runtimeRefCount : 0ULL,
                            found ? (unsigned long long)snapshot.activeLeaseSerial : 0ULL,
                            found && snapshot.failureQuarantined ? 1 : 0);
                    }
                }
            }
            if (adopted) {
                uint64_t poolAfter = 0;
                {
                    std::lock_guard<std::mutex> lock(state.poolMutex);
                    poolAfter = state.bytes;
                }
                PrefixCacheDebug("[PrefixCache] promotion-commit-l1: host_removed=%zu "
                                 "pool_bytes_before=%llu pool_bytes_after=%llu.\n",
                                 hostRemoved, (unsigned long long)poolBefore,
                                 (unsigned long long)poolAfter);
            }
        }
    } owners{*state, *context, plans, identities, heldKeys, adopted, pending};
    int missingPages = 0; uint64_t transferBytes = 0;
    try {
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model, layer)) continue;
            for (int side = 0; side < 2; ++side) {
                auto refs = model->GetPagedKVCacheManagers(layer, side == 0);
                if (refs.empty()) return 0;
                for (auto ref : refs) {
                    auto manager = ref.second;
                    if (!manager || !manager->cudaData || manager->pageLen != pageLen) return 0;
                    plans.emplace_back();
                    Plan &plan = plans.back();
                    plan.manager = manager; plan.layer = layer;
                    plan.side = side; plan.device = ref.first;
                    plan.host.assign(wantPages, nullptr);
                    plan.ledgerLeases.resize(wantPages);
                    plan.ids.reserve(wantPages);
                    plan.newIds.reserve(wantPages);
                    // Allocate every owner container before the atomic Pick;
                    // from its return onward plan.ids itself owns all pins.
                    manager->QueryPagePositions(context->currentTokens, plan.ids,
                                                extraKeys, wantPages, true);
                    plan.ids.resize(wantPages, -1);
                    const int resident = (int)std::count_if(plan.ids.begin(), plan.ids.end(),
                                                           [](int id) { return id >= 0; });
                    const int missing = wantPages - resident;
                    {
                        std::unique_lock<std::mutex> lock(state->poolMutex, std::try_to_lock);
                        if (!lock.owns_lock()) return 0;
                        for (int p = 0; p < wantPages; ++p) {
                            auto it = state->pool.find({identities[p], layer, side, plan.device});
                            if (plan.ids[p] >= 0) { if (it != state->pool.end()) return 0; continue; }
                            const Qwen35SpillKey key{identities[p], layer, side, plan.device};
                            heldKeys.push_back(key);
                            state->promotionHolds.insert(key);
                            if (it == state->pool.end() ||
                                !Qwen35SpillPageUsable(*it->second, manager)) return 0;
                            auto ledgerLease = state->ownerModel->pagedPageLedger->BeginPromotion(
                                it->second->ledgerIdentity);
                            if (!ledgerLease.Valid()) return 0;
                            plan.ledgerLeases[p] = std::move(ledgerLease);
                            it->second->promotionRefs.fetch_add(1, std::memory_order_acq_rel);
                            state->lru.splice(state->lru.end(), state->lru, it->second->lruPosition);
                            plan.host[p] = it->second; ++missingPages;
                            transferBytes += it->second->hostBytes.size();
                        }
                    }
                }
            }
        }
        if (!missingPages) return 0;

        // All L2 owners are now held before any deferred recycle is dispatched.
        // This prevents FIFO eviction of a later manager's target while the
        // first manager is waiting for its GPU replacement pages.
        for (auto &plan : plans) {
            const int missing = (int)std::count_if(plan.host.begin(), plan.host.end(),
                                                   [](const auto &page) { return page != nullptr; });
            if (missing <= 0) continue;
            int available = 0;
            {
                std::lock_guard<std::mutex> lock(plan.manager->pageIndexLocker);
                available = background ? (int)plan.manager->freePages.size() :
                    plan.manager->FreePageCount();
            }
            if (available >= missing) continue;
            // Dispatch one batch and return immediately. The scheduler owns
            // dictLocker/forwardLocker while calling this function; sleeping
            // here blocks unrelated requests and made retry latency scale with
            // the number of TP managers. Context-scoped retryPromotionHolds
            // keep all target L2 pages safe for the next scheduler pass.
            bool ready = false;
            try { ready = plan.manager->PrepareFreePages(missing); }
            catch (...) { ready = false; }
            if (!ready) {
                if (!background) {
                    bool tickets = false;
                    {
                        std::lock_guard<std::mutex> lock(plan.manager->pageIndexLocker);
                        tickets = !plan.manager->pendingRecycleTickets.empty();
                    }
                    context->intParams["qwen35_l2_promotion_pending"] = tickets ? 1 : 2;
                    context->intParams["qwen35_l2_promotion_tokens"] = wantTokens;
                    context->intParams["qwen35_l2_promotion_reason"] = 1;
                }
                return 0;
            }
        }
        const int oldDevice = FastllmCudaGetDevice();
        struct RestoreDevice { int old; ~RestoreDevice() { FastllmCudaSetDevice(old); } } restore{oldDevice};
        for (auto &plan : plans) {
            FastllmCudaSetDevice(plan.device);
            auto streamIt = state->loadStreams.find(plan.device);
            if (streamIt == state->loadStreams.end() || !state->loadHost) return 0;
            void *stream = streamIt->second;
            const size_t bytes = plan.manager->GetBytes() / (size_t)plan.manager->maxPages;
            if (!bytes || bytes > state->stagingBytes) return 0;
            for (int p = 0; p < wantPages; ++p) {
                if (!plan.host[p]) continue;
                int id = plan.manager->GetUnusedPageIndex(true, background);
                if (id < 0) return 0;
                plan.ids[p] = id; plan.newIds.push_back(id);
                memcpy(state->loadHost, plan.host[p]->hostBytes.data(), bytes);
                FastllmCudaCopyFromPinnedHostToDeviceAsync(
                    (uint8_t *)plan.manager->cudaData + (uint64_t)id * bytes,
                    state->loadHost, bytes, stream);
                { RequestDiagnosticCurrentOperation diagnosticCompletion("l2_cuda_stream_completion_wait",
                    RequestDiagnosticResource("CUDA.stream", stream).c_str()); FastllmCudaStreamSynchronize(stream); }
            }
            void *h2dEvent = FastllmCudaEventCreate();
            if (!h2dEvent) return 0;
            plan.h2dComplete = std::shared_ptr<void>(h2dEvent, [device = plan.device](void *event) {
                const int old = FastllmCudaGetDevice();
                FastllmCudaSetDevice(device);
                FastllmCudaEventDestroy(event);
                FastllmCudaSetDevice(old);
            });
            FastllmCudaEventRecord(plan.h2dComplete.get(), stream);
            { RequestDiagnosticCurrentOperation diagnosticCompletion("l2_cuda_stream_completion_wait",
                    RequestDiagnosticResource("CUDA.stream", stream).c_str()); FastllmCudaStreamSynchronize(stream); }
        }
        // L2 promotion already owns the ledger promotion leases. Suppress the
        // observation hook while Record publishes the GPU trie; the atomic
        // CommitL1Batch below is the sole L1 adoption point.
        struct RecordLedgerGuard {
            Qwen35SpillState &state;
            bool previous;
            explicit RecordLedgerGuard(Qwen35SpillState &value)
                : state(value),
                  previous(state.directPublishActive.exchange(
                      true, std::memory_order_acq_rel)) {}
            ~RecordLedgerGuard() {
                state.directPublishActive.store(
                    previous, std::memory_order_release);
            }
        };
        std::vector<PrefixCachePageLedger::L1Adoption> adoptions;
        std::vector<PrefixCachePageIdentity> expectedAdoptions;
        bool committedBatch = false;
        {
            RecordLedgerGuard recordLedgerGuard(*state);
            for (auto &plan : plans) {
                for (int id : plan.ids) if (id < 0) return 0;
                {
                    std::lock_guard<std::mutex> lock(plan.manager->pageIndexLocker);
                    plan.manager->recordProducerEvent = plan.h2dComplete;
                }
                plan.manager->Record(context->currentTokens, plan.ids, extraKeys);
                if (context->pagedPageOwnerLease &&
                    context->pagedPageOwnerLease->generation != 0) {
                    PagedCacheManager::RecycleRequestIdentity owner;
                    owner.requestId = context->requestId;
                    owner.requestGeneration =
                        context->pagedPageOwnerLease->generation;
                    owner.ownerLease = context->pagedPageOwnerLease;
                    model->RegisterPagedPageOwners(plan.manager, plan.ids, owner);
                }
                auto got = Qwen35QuerySpillL1Pages(
                    plan.manager, context->currentTokens, extraKeys, wantPages);
                if ((int)got.size() < wantPages) return 0;
                for (int p = 0; p < wantPages; ++p) if (got[p] != plan.ids[p]) return 0;
                for (int p = 0; p < wantPages; ++p) {
                    if (!plan.host[p]) continue;
                    uint64_t generation = 0;
                    {
                        std::lock_guard<std::mutex> managerLock(plan.manager->pageIndexLocker);
                        if (plan.ids[p] < 0 || plan.ids[p] >= plan.manager->maxPages) return 0;
                        generation = (uint64_t)plan.manager->pageTimestamp[plan.ids[p]];
                    }
                    expectedAdoptions.push_back(plan.host[p]->ledgerIdentity);
                    adoptions.push_back({plan.host[p]->ledgerIdentity, plan.ledgerLeases[p],
                        plan.manager, plan.ids[p], generation, plan.h2dComplete});
                }
            }
            const int rejectIndex = Qwen35InjectedL1RejectIndex();
            if (rejectIndex >= 0 && rejectIndex < (int)adoptions.size()) {
                adoptions[(size_t)rejectIndex].lease.serial++;
                PrefixCacheDebug("[PrefixCache] commit-l1-failure-injected: index=%d "
                                 "adoptions=%zu reason=stale_lease.\n",
                                 rejectIndex, adoptions.size());
            }
            committedBatch = state->ownerModel->pagedPageLedger->CommitL1Batch(
                adoptions, &expectedAdoptions);
            if (!committedBatch)
                PrefixCacheDebug("[PrefixCache] commit-l1-batch: result=failed "
                                 "adoptions=%zu.\n", adoptions.size());
        }
        if (!committedBatch) return 0;
        adopted = true;
        {
            auto owner = std::make_shared<Qwen35GpuPrefixLease>();
            for (auto &plan : plans) {
                if (!plan.manager || plan.ids.empty()) continue;
                owner->pages.emplace_back(plan.manager, plan.ids);
                plan.ids.clear();
                plan.newIds.clear();
            }
            if (!owner->pages.empty()) state->adoptedGpuOwners[context] = std::move(owner);
        }
        if (!background) {
            context->prefixRestore.tier = PrefixCacheRestoreStatus::Host;
            context->prefixRestore.phase = PrefixCacheRestoreStatus::Ready;
            context->prefixRestore.candidateTokens = wantTokens;
            context->prefixRestore.loadedBytes = transferBytes;
            context->prefixRestore.hostComponents = context->prefixRestore.requiredComponents;
            context->prefixRestore.loadingState = PrefixCacheRecord::Ready;
        }
        if (Qwen35SpillStatsEnabled())
            Qwen35SpillStatsRef().restoreHitPages.fetch_add(missingPages, std::memory_order_relaxed);
        return missingPages;
    } catch (...) { return 0; }
#else
    return 0;
#endif
}

int Qwen3_5Model::QueryPagedSpillPrefix(ResponseContext *context, int maxTokens) const {
    if (!context) return 0;
    auto setQueryStatus = [&](int status, int l1Tokens = 0, int l2Tokens = 0,
                              int combinedTokens = 0, int duplicatePages = 0,
                              int firstGapPage = -1) {
        context->intParams["qwen35_l2_query_status"] = status;
        context->intParams["qwen35_l1_resident_tokens"] = l1Tokens;
        context->intParams["qwen35_l2_host_tokens"] = l2Tokens;
        context->intParams["qwen35_combined_local_tokens"] = combinedTokens;
        context->intParams["qwen35_l2_duplicate_pages"] = duplicatePages;
        context->intParams["qwen35_l2_first_gap_page"] = firstGapPage;
    };
    if (!context->CanUsePagedPrefixCache() || !pagedSpillState ||
        maxTokens <= 0 || context->currentTokens.empty()) {
        setQueryStatus(3);
        return 0;
    }
    std::unique_lock<std::mutex> publication(pagedSpillState->promoteMutex,
                                             std::try_to_lock);
    if (!publication.owns_lock()) {
        setQueryStatus(2);
        return 0;
    }
    const int pageLen = fastllm::GetPageLen();
    const int pages = std::min(maxTokens, (int)context->currentTokens.size()) / pageLen;
    if (pages <= 0) {
        setQueryStatus(3);
        return 0;
    }
    const auto extraKeys = context->PrefixCachePageKeys();
    std::vector<uint64_t> identities;
    uint64_t hash = 0;
    for (int p = 0; p < pages; ++p) {
        if (extraKeys && p >= (int)extraKeys->size()) {
            setQueryStatus(5);
            return 0;
        }
        hash = PagedCacheManager::HashTokenPrefixPage(hash,
            context->currentTokens.data() + p * pageLen, pageLen,
            extraKeys ? (*extraKeys)[p] : std::string());
        identities.push_back(hash);
    }
    std::vector<uint8_t> complete(pages, 1);
    std::vector<uint8_t> pageL1(pages, 1), pageL2(pages, 1);
    std::vector<uint8_t> duplicate(pages, 0);
    bool sawManager = false;
    uint64_t hostHits = 0, hostMisses = 0, hostUnusable = 0, duplicateOwners = 0;
    uint64_t l1Hits = 0;
    for (int layer = 0; layer < block_cnt; ++layer) {
        if (Qwen35LayerIsLinearAttention(this, layer)) continue;
        for (int side = 0; side < 2; ++side) {
            auto refs = GetPagedKVCacheManagers(layer, side == 0);
            if (refs.empty()) {
                setQueryStatus(5);
                return 0;
            }
            for (auto ref : refs) {
                auto manager = ref.second;
                if (!manager || manager->pageLen != pageLen) {
                    setQueryStatus(5);
                    return 0;
                }
                sawManager = true;
                auto gpu = Qwen35QuerySpillL1Pages(
                    manager, context->currentTokens, extraKeys, pages);
                gpu.resize(pages, -1);
                std::lock_guard<std::mutex> lock(pagedSpillState->poolMutex);
                for (int p = 0; p < pages; ++p) {
                    auto it = pagedSpillState->pool.find(
                        {identities[p], layer, side, ref.first});
                    const bool foundHost = it != pagedSpillState->pool.end();
                    const bool hasHost = foundHost &&
                        Qwen35SpillPageUsable(*it->second, manager);
                    if (gpu[p] >= 0) ++l1Hits;
                    if (hasHost) ++hostHits;
                    else if (!foundHost) ++hostMisses;
                    else ++hostUnusable;
                    if (gpu[p] < 0) pageL1[p] = 0;
                    if (!hasHost) pageL2[p] = 0;
                    // A duplicate L1/L2 owner is an invalid publication.  A
                    // missing host payload is also a miss for this component.
                    if (gpu[p] >= 0 && hasHost) {
                        ++duplicateOwners;
                        duplicate[p] = 1;
                        complete[p] = 0;
                    } else if (gpu[p] < 0 && !hasHost) {
                        complete[p] = 0;
                    }
                }
            }
        }
    }
    if (!sawManager) {
        setQueryStatus(5);
        return 0;
    }
    int result = 0;
    while (result < pages && complete[result]) result += 1;
    const int firstGapPage = result < pages ? result : -1;
    result *= pageLen;
    auto recordSummary = [&](int finalTokens, int status) {
        const int summaryPages = std::max(0, finalTokens / pageLen);
        int l1Tokens = 0, l2Tokens = 0;
        for (int p = 0; p < summaryPages; ++p) {
            if (duplicate[p]) continue;
            if (pageL1[p]) l1Tokens += pageLen;
            else if (pageL2[p]) l2Tokens += pageLen;
        }
        setQueryStatus(status, l1Tokens, l2Tokens, finalTokens,
                       (int)duplicateOwners, firstGapPage);
    };
    if (result <= 0 || !Qwen35HasLinearAttentionLayers(this, block_cnt)) {
        recordSummary(result, duplicateOwners > 0 ? 4 : 1);
        PrefixCacheDebug("[PrefixCache] l2-query: pages=%d result_tokens=%d l1_hits=%llu "
                         "host_hits=%llu host_misses=%llu host_unusable=%llu "
                         "duplicate_owners=%llu linear_required=%d.\n",
                         pages, result, (unsigned long long)l1Hits,
                         (unsigned long long)hostHits, (unsigned long long)hostMisses,
                         (unsigned long long)hostUnusable,
                         (unsigned long long)duplicateOwners,
                         Qwen35HasLinearAttentionLayers(this, block_cnt) ? 1 : 0);
        return result;
    }
    const bool requireMtp = RequiresMtpPrefixSnapshot(context);
    const bool requireDFlash = RequiresDFlashPrefixSnapshot(context);
    std::lock_guard<std::mutex> lock(Qwen35LinearPrefixSnapshotsMutex());
    std::map<std::string, std::string> diagnostic;
    const Qwen35LinearPrefixSnapshot *snapshot = Qwen35FindLinearPrefixSnapshotLocked(
        this, context->currentTokens, context, result, -1,
        requireMtp, requireDFlash, dflashLayers, dflashKvHeads, dflashHeadDim,
        GetFastllmObservabilityMode() == FastllmObservabilityMode::Off ? nullptr : &diagnostic);
    if (!diagnostic.empty()) {
        std::string fields;
        for (const auto &entry : diagnostic) fields += entry.first + "=" + entry.second + " ";
        FastllmEvent("cache", context->requestId, "cache_snapshot_match",
                     "kv_prefix_tokens=%d first_gap_page=%d l1_component_hits=%llu host_component_hits=%llu "
                     "host_component_misses=%llu host_component_unusable=%llu %s",
                     result, firstGapPage, (unsigned long long)l1Hits, (unsigned long long)hostHits,
                     (unsigned long long)hostMisses, (unsigned long long)hostUnusable, fields.c_str());
    }
    const int finalResult = snapshot ? std::min(result, snapshot->cachedLen) : 0;
    recordSummary(finalResult, snapshot ? (duplicateOwners > 0 ? 4 : 1) : 5);
    PrefixCacheDebug("[PrefixCache] l2-query: pages=%d result_tokens=%d final_tokens=%d "
                     "l1_hits=%llu host_hits=%llu host_misses=%llu host_unusable=%llu "
                     "duplicate_owners=%llu linear_snapshot=%d.\n",
                     pages, result, finalResult, (unsigned long long)l1Hits,
                     (unsigned long long)hostHits, (unsigned long long)hostMisses,
                     (unsigned long long)hostUnusable,
                     (unsigned long long)duplicateOwners, snapshot ? 1 : 0);
    return finalResult;
}

bool Qwen3_5Model::PromotePagedSpillPrefix(ResponseContext *context, int wantTokens,
                                           bool background) {
    if (!context || !pagedSpillState || wantTokens <= 0) return false;
    std::unique_lock<std::mutex> lock(pagedSpillState->promoteMutex,
                                      std::defer_lock);
    if (background) {
        if (!lock.try_lock()) return false;
    } else {
        lock.lock();
    }
    const int maxTokens = (int)context->currentTokens.size();
    return Qwen35RestoreSpilledPages(this, context, wantTokens, maxTokens,
                                     background) > 0;
}

bool Qwen3_5Model::PublishDirectPagedPrefix(
        ResponseContext *context, int length, const std::function<bool()> &publish,
        const std::function<void()> &rollback) {
    if (!context || length <= 0 || !publish || !pagedPageLedger ||
        context->currentTokens.empty()) return false;
    // SSD direct restore still needs the same publication mutex and host-pool
    // bookkeeping even when L2 is disabled.  Create only the coordination
    // state here; no host staging allocation or spill worker is started.
    if (!pagedSpillState) {
        pagedSpillState = std::make_shared<Qwen35SpillState>();
        pagedSpillState->ownerModel = this;
    }
    auto state = pagedSpillState;
    RequestDiagnosticOperation diagnosticPromoteWait(context->requestId, "l1_publication_promote_lock_wait",
        RequestDiagnosticResource("spill.promoteMutex", &state->promoteMutex).c_str());
    std::unique_lock<std::mutex> publication(state->promoteMutex);
    diagnosticPromoteWait.Finish();
    RequestDiagnosticOperation diagnosticPromoteHeld(context->requestId, "l1_publication_promote_lock_held",
        RequestDiagnosticResource("spill.promoteMutex", &state->promoteMutex).c_str());
    const int pageLen = fastllm::GetPageLen();
    if (pageLen <= 0 || length % pageLen != 0 || length > (int)context->currentTokens.size()) return false;
    // 发布分段计时（2026-10-17，B21）：只观测，不改发布语义、顺序、锁或回滚路径。
    // 真实 l1_publication_ms=5769–5793 无法用 fake 页表 T0 复现（T0 仅约 90 ms），
    // 这里把 identities 构造 / L2 lease / 发布回调 / 采用校验 / ledger 提交 /
    // host 回收 / 末尾逐页诊断拆成原生毫秒字段，成功后随 raw 事件输出。
    struct PublishProfile {
        uint64_t requestId = 0;
        int identities = 0;
        int pages = 0;
        double identitiesMs = 0, leaseMs = 0, callbackMs = 0, adoptMs = 0,
               commitMs = 0, hostMs = 0, diagMs = 0, commitWaitMs = 0, commitWorkMs = 0;
        double commitPrepareMs = 0, commitAllocMs = 0, commitMutateMs = 0, commitTrimMs = 0;
        double commitExpectedMs = 0, commitAdoptionMs = 0;
        size_t commitPagesBefore = 0, commitPagesAfter = 0;
        size_t commitKeyCount = 0, commitKeyChars = 0;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~PublishProfile() {
            const double totalMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            FastllmEvent("cache", requestId, "cache_publish_profile",
                         "measurement=native_publish timing_scope=through_l1_commit "
                         "identities=%d pages=%d identities_ms=%.3f lease_ms=%.3f "
                         "callback_ms=%.3f adopt_ms=%.3f commit_ms=%.3f "
                         "commit_wait_ms=%.3f commit_work_ms=%.3f "
                         "commit_prepare_ms=%.3f commit_alloc_ms=%.3f "
                         "commit_mutate_ms=%.3f commit_trim_ms=%.3f "
                         "commit_expected_ms=%.3f commit_adoption_ms=%.3f "
                         "key_count=%zu key_avg_len=%.1f "
                         "pages_before=%zu pages_after=%zu host_ms=%.3f "
                         "diag_ms=%.3f total_ms=%.3f",
                         identities, pages, identitiesMs, leaseMs, callbackMs, adoptMs,
                         commitMs, commitWaitMs, commitWorkMs, commitPrepareMs,
                         commitAllocMs, commitMutateMs, commitTrimMs, commitExpectedMs,
                         commitAdoptionMs, commitKeyCount,
                         commitKeyCount == 0 ? 0.0
                             : double(commitKeyChars) / double(commitKeyCount) / 2.0,
                         commitPagesBefore, commitPagesAfter, hostMs, diagMs, totalMs);
        }
    } publishProfile;
    publishProfile.requestId = context->requestId;
    auto pubMark = [] { return std::chrono::steady_clock::now(); };
    auto pubMs = [](std::chrono::steady_clock::time_point from,
                    std::chrono::steady_clock::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    };
    auto pubPrev = publishProfile.start;
    const auto extraKeys = context->PrefixCachePageKeys();
    const std::vector<std::string> noKeys;
    const auto &keys = extraKeys ? *extraKeys : noKeys;
    const auto identities = DescribePagedPrefixPages(context->currentTokens, keys, length);
    if (identities.empty()) return false;
    const int pageCount = length / pageLen;
    {
        const auto now = pubMark();
        publishProfile.identitiesMs = pubMs(pubPrev, now);
        publishProfile.identities = (int)identities.size();
        publishProfile.pages = pageCount;
        pubPrev = now;
    }

    struct Plan {
        const PrefixCachePageIdentity *identity = nullptr;
        PagedCacheManager *manager = nullptr;
        std::shared_ptr<Qwen35SpilledPage> host;
        PrefixCachePageLedger::Lease lease;
        int layer = -1, side = -1, rank = -1;
        int ordinal = -1;
        uint64_t hash = 0;
        int previousGpu = -1;
    };
    std::vector<Plan> plans;
    plans.reserve(identities.size());
    std::vector<PrefixCachePageLedger::L1AdoptionView> adoptions;
    adoptions.reserve(identities.size());
    auto rollbackGpu = [&]() {
        std::map<PagedCacheManager *, std::pair<std::vector<int>, bool>> gpuPagesByManager;
        for (auto &plan : plans) {
            try {
                auto found = gpuPagesByManager.find(plan.manager);
                if (found == gpuPagesByManager.end()) {
                    auto gpu = Qwen35QuerySpillL1Pages(
                        plan.manager, context->currentTokens, extraKeys, pageCount);
                    gpu.resize(pageCount, -1);
                    found = gpuPagesByManager.emplace(
                        plan.manager, std::make_pair(std::move(gpu), false)).first;
                }
                const auto &gpu = found->second.first;
                if (plan.ordinal >= 0 && plan.ordinal < (int)gpu.size() &&
                    gpu[plan.ordinal] >= 0 && gpu[plan.ordinal] != plan.previousGpu) {
                    found->second.second = true;
                    Qwen35DropPromotedGpuPage(plan.manager, gpu[plan.ordinal]);
                }
            } catch (...) {
                // Continue cleaning the remaining managers; the ledger abort
                // below still prevents a failed host promotion from staying
                // in PROMOTE_PENDING.
            }
        }
        for (const auto &entry : gpuPagesByManager) {
            if (!entry.second.second) continue;
            try {
                Qwen35RestoreSpillTrieHole(
                    entry.first, context->currentTokens, extraKeys, pageCount);
            } catch (...) {
                // Abort every remaining lease even if hole reconstruction fails.
            }
        }
    };
    auto abortPlansLocked = [&]() {
        for (auto &plan : plans) {
            if (plan.lease.Valid()) {
                state->ownerModel->pagedPageLedger->AbortPromotion(plan.lease, "direct publish failed");
                plan.lease = PrefixCachePageLedger::Lease();
            }
            if (plan.host) {
                const uint32_t refs = plan.host->promotionRefs.load(std::memory_order_acquire);
                if (refs) plan.host->promotionRefs.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    };
    auto abortPlans = [&]() {
        std::lock_guard<std::mutex> lock(state->poolMutex);
        abortPlansLocked();
    };

    bool publicationAttempted = false;
    bool committed = false;
    auto cleanup = [&]() noexcept {
        if (publicationAttempted) {
            rollbackGpu();
            if (rollback) {
                try { rollback(); } catch (...) {}
            }
        }
        try { abortPlans(); } catch (...) {}
    };
    struct DirectTransactionGuard {
        decltype(cleanup) &callback;
        bool &committed;
        ~DirectTransactionGuard() {
            if (!committed) callback();
        }
    } transaction{cleanup, committed};

    // Query each manager once before publication.  Every manager owns all
    // page ordinals for its layer/side/rank; querying the full prefix once per
    // identity made long SSD restores rescan the same trie thousands of times.
    std::map<PagedCacheManager *, std::vector<int>> prePublishGpuPages;
    auto queryPrePublishGpu = [&](PagedCacheManager *manager) -> const std::vector<int> & {
        auto found = prePublishGpuPages.find(manager);
        if (found == prePublishGpuPages.end()) {
            auto gpu = Qwen35QuerySpillL1Pages(
                manager, context->currentTokens, extraKeys, pageCount);
            gpu.resize(pageCount, -1);
            found = prePublishGpuPages.emplace(manager, std::move(gpu)).first;
        }
        return found->second;
    };

    std::map<std::pair<int,int>, std::vector<std::pair<int,PagedCacheManager*>>> managerRefs;
    // Acquire L2 leases before the callback publishes GPU pages.  Every page
    // is also included in the batch so a new SSD-direct page is adopted by the
    // ledger atomically with all layers, K/V sides and TP ranks.
    {
        std::lock_guard<std::mutex> lock(state->poolMutex);
        for (const auto &identity : identities) {
            uint64_t hash = 0;
            try { hash = std::stoull(identity.pageHash); }
            catch (...) { return false; }
            const int side = identity.side == "key" ? 0 : 1;
            PagedCacheManager *manager = nullptr;
            const auto layoutKey = std::make_pair(identity.layer,side);
            auto refs = managerRefs.find(layoutKey);
            if (refs == managerRefs.end())
                refs = managerRefs.emplace(layoutKey,
                    GetPagedKVCacheManagers(identity.layer,side == 0)).first;
            for (const auto &ref : refs->second) {
                if (ref.first == identity.rank) { manager = ref.second; break; }
            }
            if (!manager) return false;
            const auto &gpu = queryPrePublishGpu(manager);
            if (identity.pageOrdinal >= gpu.size()) return false;
            const int previousGpu = gpu[(size_t)identity.pageOrdinal];
            auto it = state->pool.find({hash, identity.layer, side, identity.rank});
            if (it != state->pool.end() && previousGpu >= 0) {
                // A host+GPU duplicate is an invalid candidate, never publish
                // over it or delete either owner.
                return false;
            }
            plans.emplace_back();
            Plan &plan = plans.back();
            // Populate all potentially-allocating fields before acquiring a
            // promotion lease. The placeholder then owns that lease through
            // every subsequent failure path.
            plan.identity = &identity;
            plan.manager = manager;
            plan.layer = identity.layer;
            plan.side = side;
            plan.rank = identity.rank;
            plan.ordinal = (int)identity.pageOrdinal;
            plan.hash = hash;
            plan.previousGpu = previousGpu;
            if (it != state->pool.end()) {
                if (!Qwen35SpillPageUsable(*it->second, manager)) return false;
                plan.lease = state->ownerModel->pagedPageLedger->BeginPromotion(identity);
                if (!plan.lease.Valid()) return false;
                it->second->promotionRefs.fetch_add(1, std::memory_order_acq_rel);
                plan.host = it->second;
            }
        }
    }
    { const auto now = pubMark(); publishProfile.leaseMs = pubMs(pubPrev, now); pubPrev = now; }
    if (plans.size() != identities.size()) return false;

    bool published = false;
    struct DirectPublicationGuard {
        Qwen35SpillState &state;
        explicit DirectPublicationGuard(Qwen35SpillState &value) : state(value) {
            state.directPublishActive.store(true, std::memory_order_release);
        }
        ~DirectPublicationGuard() {
            state.directPublishActive.store(false, std::memory_order_release);
        }
    } publicationGuard(*state);
    publicationAttempted = true;
    try { published = publish(); }
    catch (...) { published = false; }
    { const auto now = pubMark(); publishProfile.callbackMs = pubMs(pubPrev, now); pubPrev = now; }
    if (!published) return false;

    // The publication callback records every page while the promotion mutex
    // is held.  Reuse one post-publication query per manager when collecting
    // producer events and generations for the atomic ledger commit.
    // Query and lock each manager once, keeping the original adoption order.
    // All allocations/validation still precede the single ledger commit.
    std::map<PagedCacheManager *, std::vector<size_t>> plansByManager;
    for (size_t index = 0; index < plans.size(); ++index)
        plansByManager[plans[index].manager].push_back(index);
    adoptions.resize(plans.size());
    for (const auto &group : plansByManager) {
        auto *manager = group.first;
        auto gpu = Qwen35QuerySpillL1Pages(
            manager, context->currentTokens, extraKeys, pageCount);
        gpu.resize(pageCount,-1);
        std::lock_guard<std::mutex> lock(manager->pageIndexLocker);
        for (size_t index : group.second) {
            const auto &plan = plans[index];
            if (plan.ordinal < 0 || plan.ordinal >= (int)gpu.size() ||
                gpu[plan.ordinal] < 0 || gpu[plan.ordinal] >= manager->maxPages) return false;
            const int page = gpu[plan.ordinal];
            const uint64_t generation = static_cast<uint64_t>(manager->pageTimestamp[page]);
            auto node = manager->pageToTrieNode.find(page);
            if (node == manager->pageToTrieNode.end() || !node->second ||
                !node->second->producerEvent) return false;
            auto producerEvent = node->second->producerEvent;
            adoptions[index] = {plan.identity,plan.lease,manager,page,generation,
                                producerEvent,producerEvent};
        }
    }
    { const auto now = pubMark(); publishProfile.adoptMs = pubMs(pubPrev, now); pubPrev = now; }
    const int rejectIndex = Qwen35InjectedL1RejectIndex();
    if (rejectIndex >= 0 && rejectIndex < (int)adoptions.size()) {
        adoptions[(size_t)rejectIndex].lease.serial++;
        PrefixCacheDebug("[PrefixCache] commit-l1-failure-injected: index=%d "
                         "adoptions=%zu reason=stale_lease.\n",
                         rejectIndex, adoptions.size());
    }
    size_t directPoolBytesBefore = 0;
    size_t directHostObjectsBefore = 0;
    {
        std::lock_guard<std::mutex> poolLock(state->poolMutex);
        directPoolBytesBefore = state->bytes;
        directHostObjectsBefore = state->pool.size();
    }
    RequestDiagnosticOperation diagnosticLedger(context->requestId, "l1_ledger_commit_batch",
        RequestDiagnosticResource("PagedPageLedger", state->ownerModel->pagedPageLedger.get()).c_str());
    if (adoptions.size() != identities.size() ||
        !state->ownerModel->pagedPageLedger->CommitL1BatchView(adoptions, &identities)) {
        PrefixCacheDebug("[PrefixCache] commit-l1-batch: result=failed adoptions=%zu.\n",
                         adoptions.size());
        return false;
    }
    diagnosticLedger.Finish();
    committed = true;
    if (context->pagedPageOwnerLease &&
        context->pagedPageOwnerLease->generation != 0) {
        PagedCacheManager::RecycleRequestIdentity owner;
        owner.requestId = context->requestId;
        owner.requestGeneration = context->pagedPageOwnerLease->generation;
        owner.ownerLease = context->pagedPageOwnerLease;
        try {
            std::map<PagedCacheManager *, std::vector<int>> ownerPages;
            for (size_t i = 0; i < adoptions.size() && i < plans.size(); ++i)
                ownerPages[plans[i].manager].push_back((int)adoptions[i].page);
            for (const auto &entry : ownerPages)
                state->ownerModel->RegisterPagedPageOwners(
                    entry.first, entry.second, owner);
        } catch (...) {
            PrefixCacheDebug("[PrefixCache] direct-l1-owner-register: result=failed "
                             "request=%llu reason=exception.\n",
                             (unsigned long long)context->requestId);
        }
    }
    {
        const auto timing = state->ownerModel->pagedPageLedger->LastCommitTiming();
        publishProfile.commitWaitMs = timing.waitMs;
        publishProfile.commitWorkMs = timing.workMs;
        publishProfile.commitPrepareMs = timing.prepareMs;
        publishProfile.commitAllocMs = timing.allocMs;
        publishProfile.commitMutateMs = timing.mutateMs;
        publishProfile.commitTrimMs = timing.trimMs;
        publishProfile.commitPagesBefore = timing.pagesBefore;
        publishProfile.commitPagesAfter = timing.pagesAfter;
        publishProfile.commitExpectedMs = timing.expectedMs;
        publishProfile.commitAdoptionMs = timing.adoptionMs;
        publishProfile.commitKeyCount = timing.keyCount;
        publishProfile.commitKeyChars = timing.keyChars;
    }
    { const auto now = pubMark(); publishProfile.commitMs = pubMs(pubPrev, now); pubPrev = now; }
    for (auto &plan : plans) plan.lease = PrefixCachePageLedger::Lease();

    // Only after every ledger record is committed may the host owners be
    // removed.  The promotion reference prevents the worker from replacing
    // the object while the callback is in flight.
    size_t directHostRemoved = 0;
    {
        std::lock_guard<std::mutex> lock(state->poolMutex);
        for (auto &plan : plans) {
            if (!plan.host) continue;
            const uint32_t refs = plan.host->promotionRefs.load(std::memory_order_acquire);
            if (refs) plan.host->promotionRefs.fetch_sub(1, std::memory_order_acq_rel);
            Qwen35RemoveSpilledPageLocked(
                *state, {plan.hash, plan.layer, plan.side, plan.rank}, plan.host);
            ++directHostRemoved;
        }
        PrefixCacheDebug("[PrefixCache] direct-l1-commit: result=success adoptions=%zu "
                         "pool_bytes_before=%zu pool_bytes_after=%zu "
                         "host_objects_before=%zu host_objects_after=%zu host_removed=%zu.\n",
                         adoptions.size(), directPoolBytesBefore, state->bytes,
                         directHostObjectsBefore, state->pool.size(), directHostRemoved);
    }
    { const auto now = pubMark(); publishProfile.hostMs = pubMs(pubPrev, now); pubPrev = now; }
    // 逐页诊断只在 debug 日志开启时执行（2026-10-17，B21）：这段每页都会
    // CanonicalKey + Find（Find 还会深拷贝 PageRecord），12 万页实测 523 ms，
    // 而 user/info 模式下 PrefixCacheDebug 本来就不会输出。判定条件与
    // PrefixCacheDebug 内部完全一致，因此输出内容/顺序在 debug 下不变。
    for (size_t i = 0; PrefixCacheDebugEnabled() && i < adoptions.size() && i < identities.size(); ++i) {
        const auto &adoption = adoptions[i];
        const auto &identity = identities[i];
        PrefixCacheDebug("[PrefixCache] direct-l1-page: layer=%d side=%d rank=%d "
                         "page=%d generation=%llu producer_event=%d.\n",
                         identity.layer, identity.side == "key" ? 0 : 1, identity.rank,
                         adoption.page, (unsigned long long)adoption.generation,
                         adoption.producerEvent ? 1 : 0);
        const auto ledger = state->ownerModel->pagedPageLedger->Find(identity);
        if (ledger) {
            PrefixCacheDebug("[PrefixCache] direct-l1-ledger: layer=%d side=%d rank=%d "
                             "page=%lld generation=%llu tier=%d state=%d durable=%d "
                             "runtime_ref=%llu lease=%llu quarantined=%d.\n",
                             identity.layer, identity.side == "key" ? 0 : 1, identity.rank,
                             (long long)ledger->l1Page,
                             (unsigned long long)ledger->l1Generation,
                             (int)ledger->runtimeTier, (int)ledger->runtimeState,
                             (int)ledger->durableState,
                             (unsigned long long)ledger->runtimeRefCount,
                             (unsigned long long)ledger->activeLeaseSerial,
                             ledger->failureQuarantined ? 1 : 0);
        }
    }
    { const auto now = pubMark(); publishProfile.diagMs = pubMs(pubPrev, now); }
    return true;
}

void Qwen3_5Model::PreparePagedSpillPrefix(ResponseContext *context, bool background) {
    if (!context || context->cacheLen || !context->CanUsePagedPrefixCache() || !pagedSpillState) return;
    auto state = pagedSpillState;
    // Never make the scheduler wait for an ongoing background promotion.
    std::unique_lock<std::mutex> promotion(state->promoteMutex, std::try_to_lock);
    if (!promotion.owns_lock()) return;
    const int maxTokens = context->currentTokens.size();
    const int pageLen = fastllm::GetPageLen();
    int maxCached = (maxTokens - 1) / pageLen * pageLen;
    while (maxCached > 0) {
        int length = 0;
        {
            std::lock_guard<std::mutex> lock(Qwen35LinearPrefixSnapshotsMutex());
            auto snapshot = Qwen35FindLinearPrefixSnapshotLocked(this, context->currentTokens,
                context, maxCached, -1,
                RequiresMtpPrefixSnapshot(context), RequiresDFlashPrefixSnapshot(context),
                dflashLayers, dflashKvHeads, dflashHeadDim);
            if (!snapshot) return;
            length = snapshot->cachedLen;
        }
        if (Qwen35RestoreSpilledPages(this, context, length, maxTokens, background) > 0) return;
        maxCached = length - pageLen;
    }
}
