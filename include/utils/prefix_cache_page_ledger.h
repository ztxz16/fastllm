#ifndef FASTLLM_UTILS_PREFIX_CACHE_PAGE_LEDGER_H
#define FASTLLM_UTILS_PREFIX_CACHE_PAGE_LEDGER_H

// The page ledger is deliberately independent from CUDA and from the v4 store.
// It records ownership and state only; payload bytes are owned by the GPU/host
// pools or by the SSD store and are never retained here.

#include <algorithm>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fastllm {

struct PrefixCachePageIdentity {
    std::string model;
    std::string family;
    std::string provenance;
    std::string dtype;
    std::string layout;
    std::string media;
    std::string parentHash;
    std::string pageHash;
    int32_t layer = 0;
    std::string side;
    int32_t rank = 0;
    uint64_t pageOrdinal = 0;

    bool operator==(const PrefixCachePageIdentity &other) const {
        return model == other.model && family == other.family &&
               provenance == other.provenance && dtype == other.dtype &&
               layout == other.layout && media == other.media &&
               parentHash == other.parentHash && pageHash == other.pageHash &&
               layer == other.layer && side == other.side && rank == other.rank &&
               pageOrdinal == other.pageOrdinal;
    }
    bool operator!=(const PrefixCachePageIdentity &other) const { return !(*this == other); }

    // Length-prefixing prevents an ambiguous concatenation from becoming an
    // identity collision.  This is intentionally more than a single hash.
    std::string CanonicalKey() const {
        // 2026-10-17（B21）：12 万身份时每次构造要 12 次 append，未预留容量会反复
        // 扩容（1→2→4…→256）。按字段总长一次性预留，内容/字节序列完全不变。
        std::string out;
        out.reserve(model.size() + family.size() + provenance.size() + dtype.size() +
                    layout.size() + media.size() + parentHash.size() + pageHash.size() +
                    side.size() + 96);
        const auto add = [&out](const std::string &value) {
            out += std::to_string(value.size());
            out.push_back(':');
            out += value;
            out.push_back('|');
        };
        add(model);
        add(family);
        add(provenance);
        add(dtype);
        add(layout);
        add(media);
        add(parentHash);
        add(pageHash);
        add(std::to_string(layer));
        add(side);
        add(std::to_string(rank));
        add(std::to_string(pageOrdinal));
        return out;
    }

    bool HasCanonicalFields() const {
        return !model.empty() && !family.empty() && !provenance.empty() &&
               !dtype.empty() && !layout.empty() && !pageHash.empty() &&
               !side.empty();
    }

    bool SameStream(const PrefixCachePageIdentity &other) const {
        return model == other.model && family == other.family &&
               provenance == other.provenance && dtype == other.dtype &&
               layout == other.layout && media == other.media &&
               layer == other.layer && side == other.side && rank == other.rank;
    }
};

using PrefixCachePageIdentityHash = std::hash<std::string>;

enum class PrefixCacheRuntimeTier { NONE, L1, L2, MIGRATING };
enum class PrefixCacheRuntimeState {
    RESIDENT,
    RECLAIM_PENDING,
    PROMOTE_PENDING,
    EVICT_PENDING,
    FAILED,
};
enum class PrefixCacheDurableState { NONE, WRITE_PENDING, DURABLE, FAILED };
enum class PrefixCacheLeaseKind { NONE, RECLAIM, PROMOTION, EVICTION };

struct PrefixCachePageSnapshot {
    PrefixCachePageIdentity identity;
    PrefixCacheRuntimeTier runtimeTier = PrefixCacheRuntimeTier::NONE;
    PrefixCacheRuntimeState runtimeState = PrefixCacheRuntimeState::RESIDENT;
    PrefixCacheDurableState durableState = PrefixCacheDurableState::NONE;
    const void *l1Manager = nullptr;
    int64_t l1Page = -1;
    uint64_t l1Generation = 0;
    // Immutable per-page generation/cache binding used by asynchronous copy
    // and restore validation. l1Generation remains the native GPU generation.
    uint64_t pageGeneration = 0;
    std::string cacheIdentity;
    uint64_t l2HostAllocation = 0;
    uint64_t runtimeRefCount = 0;
    uint64_t activeLeaseSerial = 0;
    PrefixCacheLeaseKind activeLeaseKind = PrefixCacheLeaseKind::NONE;
    uint64_t lastRecycleTicket = 0;
    bool failureQuarantined = false;
    std::string failureReason;
    std::shared_ptr<void> producerEvent;
    std::shared_ptr<void> d2hEvent;
    std::shared_ptr<void> h2dEvent;
    std::vector<std::string> durableCheckpointRoots;
};

class PrefixCachePageLedger {
public:
    struct PageRecord : PrefixCachePageSnapshot {
        // Kept public for diagnostics; callers must use Snapshot/Find and never
        // mutate a record. All mutation in this class is mutex protected.
        // WRITE_PENDING checkpoint rows that list this page (RegisterCheckpoint ++, FinishCheckpoint --).
        // A page shared with an older DURABLE row keeps durableState DURABLE, so durableState alone does
        // not show that a pending row still needs the record.
        uint32_t pendingWriters = 0;
    };
    using RecordHandle = std::shared_ptr<const PageRecord>;

    struct Lease {
        uint64_t ledgerId = 0;
        std::string key;
        uint64_t serial = 0;
        PrefixCacheLeaseKind kind = PrefixCacheLeaseKind::NONE;
        bool Valid() const { return ledgerId != 0 && serial != 0 && kind != PrefixCacheLeaseKind::NONE; }
    };

    struct RuntimeToken {
        uint64_t ledgerId = 0;
        std::string key;
        uint64_t serial = 0;
        bool Valid() const { return ledgerId != 0 && serial != 0; }
    };

    struct L1Adoption {
        PrefixCachePageIdentity identity;
        Lease lease;
        const void *manager = nullptr;
        int64_t page = -1;
        uint64_t generation = 0;
        std::shared_ptr<void> h2dEvent;
        std::shared_ptr<void> producerEvent;
    };

    // Synchronous publication may borrow identities already owned by its
    // complete expected set. Commit still stores owning PageRecord copies;
    // no pointer from this view survives CommitL1BatchView.
    struct L1AdoptionView {
        const PrefixCachePageIdentity *identity = nullptr;
        Lease lease;
        const void *manager = nullptr;
        int64_t page = -1;
        uint64_t generation = 0;
        std::shared_ptr<void> h2dEvent;
        std::shared_ptr<void> producerEvent;
    };

    enum class CheckpointState { WRITE_PENDING, DURABLE, FAILED, ERASED };
    struct CheckpointSnapshot {
        std::string key;
        std::string identity;
        uint64_t generation = 0;
        uint64_t sequence = 0;
        CheckpointState state = CheckpointState::WRITE_PENDING;
        uint64_t activeReaders = 0;
        bool writerActive = true;
        std::vector<PrefixCachePageIdentity> pages;
        std::string failureReason;
    };

    // 最近一次 CommitL1Batch 的“等锁 / 持锁工作”拆分（2026-10-17，B21）。
    // 线上 commit_ms=2796，而 T0 同规模算法只要 655–777 ms；先分辨是等锁还是工作本身。
    struct CommitTiming {
        double waitMs = 0;
        double workMs = 0;
        double prepareMs = 0;
        double allocMs = 0;
        double mutateMs = 0;
        double trimMs = 0;
        size_t pagesBefore = 0;
        size_t pagesAfter = 0;
        size_t trimmed = 0;
        double expectedMs = 0;
        double adoptionMs = 0;
        size_t keyCount = 0;
        size_t keyChars = 0;
    };
    static CommitTiming &LastCommitTiming() {
        static thread_local CommitTiming timing;
        return timing;
    }
    explicit PrefixCachePageLedger(size_t maxRecords = LedgerDefaultMaxRecords())
        : ledgerId_(NextLedgerId()), maxRecords_(std::max<size_t>(1, maxRecords)) {}
    PrefixCachePageLedger(const PrefixCachePageLedger &) = delete;
    PrefixCachePageLedger &operator=(const PrefixCachePageLedger &) = delete;

    // Observe only a real GPU resident page. A failed migration is quarantined
    // and cannot be silently replaced by a later observation.
    bool ObserveL1(const PrefixCachePageIdentity &identity, const void *manager,
                   int64_t page, uint64_t generation,
                   std::shared_ptr<void> producerEvent = std::shared_ptr<void>(),
                   uint64_t runtimeRefs = 0) {
        if (!identity.HasCanonicalFields() || manager == nullptr || page < 0) return false;
        std::lock_guard<std::mutex> lock(mu_);
        const std::string key = identity.CanonicalKey();
        std::shared_ptr<PageRecord> record;
        auto it = pages_.find(key);
        if (it == pages_.end()) {
            record = std::make_shared<PageRecord>();
            record->identity = identity;
            pages_.emplace(key, record);
        } else {
            record = it->second;
            if (record->failureQuarantined || record->runtimeTier == PrefixCacheRuntimeTier::L2 ||
                record->runtimeTier == PrefixCacheRuntimeTier::MIGRATING) {
                return false;
            }
        }
        record->runtimeTier = PrefixCacheRuntimeTier::L1;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        record->l1Manager = manager;
        record->l1Page = page;
        record->l1Generation = generation;
        record->pageGeneration = generation;
        record->cacheIdentity = identity.CanonicalKey();
        record->producerEvent = std::move(producerEvent);
        record->runtimeRefCount = runtimeRefs;
        record->activeLeaseSerial = 0;
        record->activeLeaseKind = PrefixCacheLeaseKind::NONE;
        record->failureReason.clear();
        TrimInactiveBoundedLocked();
        return true;
    }

    RecordHandle Find(const PrefixCachePageIdentity &identity) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        // Return an immutable snapshot copy.  Returning the mutable backing
        // object through shared_ptr<const> would still race with a writer after
        // the mutex is released.
        return it == pages_.end() ? RecordHandle() :
            std::make_shared<PageRecord>(*it->second);
    }

    // Read-only runtime query. It returns only resident, contiguous pages in a
    // single identity stream; no promotion, lease, or SSD import is triggered.
    std::vector<RecordHandle> Query(const PrefixCachePageIdentity &first,
                                    size_t maxPages = std::numeric_limits<size_t>::max()) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<RecordHandle> result;
        if (maxPages == 0) return result;
        PrefixCachePageIdentity expected = first;
        for (size_t count = 0; count < maxPages; ++count) {
            auto it = pages_.find(expected.CanonicalKey());
            if (it == pages_.end()) break;
            const auto &record = *it->second;
            if (record.runtimeState != PrefixCacheRuntimeState::RESIDENT ||
                (record.runtimeTier != PrefixCacheRuntimeTier::L1 &&
                 record.runtimeTier != PrefixCacheRuntimeTier::L2)) break;
            result.emplace_back(std::make_shared<PageRecord>(*it->second));
            PrefixCachePageIdentity next;
            bool found = false;
            for (const auto &entry : pages_) {
                const auto &candidate = *entry.second;
                const bool ordinalContiguous =
                    record.identity.pageOrdinal == 0 || candidate.identity.pageOrdinal == 0 ||
                    candidate.identity.pageOrdinal == record.identity.pageOrdinal + 1;
                if (candidate.identity.SameStream(first) &&
                    ordinalContiguous &&
                    candidate.identity.parentHash == record.identity.pageHash &&
                    candidate.runtimeState == PrefixCacheRuntimeState::RESIDENT &&
                    (candidate.runtimeTier == PrefixCacheRuntimeTier::L1 ||
                     candidate.runtimeTier == PrefixCacheRuntimeTier::L2)) {
                    next = candidate.identity;
                    found = true;
                    break;
                }
            }
            if (!found) break;
            expected = std::move(next);
        }
        return result;
    }

    bool Snapshot(const PrefixCachePageIdentity &identity, PrefixCachePageSnapshot &out) const {
        auto handle = Find(identity);
        if (!handle) return false;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        if (it == pages_.end()) return false;
        out = *it->second;
        return true;
    }

    RuntimeToken AcquireRuntime(const PrefixCachePageIdentity &identity) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        if (it == pages_.end() || it->second->runtimeTier == PrefixCacheRuntimeTier::NONE ||
            it->second->runtimeTier == PrefixCacheRuntimeTier::MIGRATING ||
            it->second->failureQuarantined || it->second->activeLeaseSerial != 0) {
            return RuntimeToken();
        }
        auto &record = *it->second;
        ++record.runtimeRefCount;
        RuntimeToken token;
        token.ledgerId = ledgerId_;
        token.key = identity.CanonicalKey();
        token.serial = ++nextToken_;
        runtimeTokens_[token.serial] = token.key;
        return token;
    }

    bool ReleaseRuntime(const RuntimeToken &token) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!token.Valid() || token.ledgerId != ledgerId_) return false;
        auto tok = runtimeTokens_.find(token.serial);
        if (tok == runtimeTokens_.end() || tok->second != token.key) return false;
        auto it = pages_.find(token.key);
        if (it == pages_.end() || it->second->runtimeRefCount == 0) return false;
        --it->second->runtimeRefCount;
        runtimeTokens_.erase(tok);
        return true;
    }

    Lease BeginReclaim(const PrefixCachePageIdentity &identity, uint64_t ticketSerial,
                       std::shared_ptr<void> producerEvent = std::shared_ptr<void>()) {
        if (ticketSerial == 0) return Lease();
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        if (it == pages_.end()) return Lease();
        auto &record = *it->second;
        if (record.runtimeTier != PrefixCacheRuntimeTier::L1 ||
            record.runtimeState != PrefixCacheRuntimeState::RESIDENT ||
            record.runtimeRefCount != 0 || record.activeLeaseSerial != 0 ||
            record.failureQuarantined || ticketSerial <= record.lastRecycleTicket) {
            return Lease();
        }
        record.lastRecycleTicket = ticketSerial;
        record.runtimeTier = PrefixCacheRuntimeTier::MIGRATING;
        record.runtimeState = PrefixCacheRuntimeState::RECLAIM_PENDING;
        if (producerEvent) record.producerEvent = std::move(producerEvent);
        return MakeLeaseLocked(record, identity.CanonicalKey(), PrefixCacheLeaseKind::RECLAIM);
    }

    bool CommitL2(const Lease &lease, uint64_t hostAllocationId,
                  std::shared_ptr<void> d2hEvent = std::shared_ptr<void>()) {
        if (hostAllocationId == 0) return false;
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::RECLAIM, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::L2;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        record->l2HostAllocation = hostAllocationId;
        record->d2hEvent = std::move(d2hEvent);
        record->l1Manager = nullptr;
        record->l1Page = -1;
        // Keep pageGeneration/cacheIdentity while the durable L2 owner is live.
        record->l1Generation = 0;
        ConsumeLeaseLocked(*record);
        return true;
    }

    bool FailMigration(const Lease &lease, const std::string &reason) {
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::RECLAIM, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::MIGRATING;
        record->runtimeState = PrefixCacheRuntimeState::FAILED;
        record->failureQuarantined = true;
        record->failureReason = reason.empty() ? "migration failed" : reason;
        ConsumeLeaseLocked(*record);
        return true;
    }

    // Admission can fail before a D2H copy is accepted. The victim page was
    // already detached from the trie when its recycle ticket was created
    // (fastllm.cpp:6247-6255 / 6435-6447), so it is not an L1 recovery source
    // any more and must not keep claiming tier==L1: CanEraseCheckpoint() only
    // protects a sole durable root while the tier is NONE. Staying RESIDENT
    // (instead of FAILED) also keeps CanRestoreCheckpoint() from rejecting a
    // healthy archive. An admission miss must not permanently quarantine the
    // logical prefix: a later ObserveL1() re-publishes this identity once it is
    // resident again.
    bool AbortReclaim(const Lease &lease, const std::string &reason = std::string()) {
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::RECLAIM, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::NONE;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        record->l1Manager = nullptr;
        record->l1Page = -1;
        record->l1Generation = 0;
        // durableCheckpointRoots / pageGeneration / cacheIdentity deliberately
        // survive: they are the only remaining recovery evidence for this page.
        record->failureQuarantined = false;
        record->failureReason = reason.empty() ? "reclaim admission aborted" : reason;
        ConsumeLeaseLocked(*record);
        return true;
    }

    Lease BeginPromotion(const PrefixCachePageIdentity &identity) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        if (it == pages_.end()) return Lease();
        auto &record = *it->second;
        if (record.runtimeTier != PrefixCacheRuntimeTier::L2 ||
            record.runtimeState != PrefixCacheRuntimeState::RESIDENT ||
            record.runtimeRefCount != 0 || record.activeLeaseSerial != 0 ||
            record.l2HostAllocation == 0 || record.failureQuarantined) {
            return Lease();
        }
        record.runtimeTier = PrefixCacheRuntimeTier::MIGRATING;
        record.runtimeState = PrefixCacheRuntimeState::PROMOTE_PENDING;
        return MakeLeaseLocked(record, identity.CanonicalKey(), PrefixCacheLeaseKind::PROMOTION);
    }

    bool CommitL1(const Lease &lease, const void *manager, int64_t page,
                  uint64_t generation, std::shared_ptr<void> h2dEvent = std::shared_ptr<void>()) {
        if (manager == nullptr || page < 0) return false;
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::PROMOTION, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::L1;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        record->l1Manager = manager;
        record->l1Page = page;
        record->l1Generation = generation;
        record->pageGeneration = generation;
        record->cacheIdentity = lease.key;
        record->h2dEvent = std::move(h2dEvent);
        record->producerEvent = record->h2dEvent;
        record->l2HostAllocation = 0;
        ConsumeLeaseLocked(*record);
        return true;
    }

    // Validate the complete publication before changing any record.  Multiple
    // pools/layers must not leave the first page committed when a later lease
    // is stale or an H2D/Record verification has failed.
    bool CommitL1Batch(const std::vector<L1Adoption> &adoptions,
                       const std::vector<PrefixCachePageIdentity> *expected = nullptr) {
        return CommitL1BatchImpl(adoptions, expected);
    }

    bool CommitL1BatchView(const std::vector<L1AdoptionView> &adoptions,
                           const std::vector<PrefixCachePageIdentity> *expected) {
        for (const auto &adoption : adoptions)
            if (adoption.identity == nullptr) return false;
        return CommitL1BatchImpl(adoptions, expected);
    }

private:
    static const PrefixCachePageIdentity &AdoptionIdentity(const L1Adoption &adoption) {
        return adoption.identity;
    }
    static const PrefixCachePageIdentity &AdoptionIdentity(const L1AdoptionView &adoption) {
        return *adoption.identity;
    }

    template<class Adoption>
    bool CommitL1BatchImpl(const std::vector<Adoption> &adoptions,
                           const std::vector<PrefixCachePageIdentity> *expected) {
        if (adoptions.empty()) return false;
        const auto commitQueueStart = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mu_);
        const auto commitLockedAt = std::chrono::steady_clock::now();
        {
            CommitTiming &timing = LastCommitTiming();
            timing = CommitTiming();
            timing.pagesBefore = pages_.size();
        }
        const auto commitWorkStart = std::chrono::steady_clock::now();
        auto commitMark = [] { return std::chrono::steady_clock::now(); };
        auto commitSpanMs = [](std::chrono::steady_clock::time_point from,
                               std::chrono::steady_clock::time_point to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        auto commitPrev = commitWorkStart;
        struct CommitTimingGuard {
            std::chrono::steady_clock::time_point queueStart;
            std::chrono::steady_clock::time_point lockedAt;
            ~CommitTimingGuard() {
                CommitTiming &timing = PrefixCachePageLedger::LastCommitTiming();
                timing.waitMs = std::chrono::duration<double, std::milli>(
                    lockedAt - queueStart).count();
                timing.workMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - lockedAt).count();
            }
        } commitTimingGuard{commitQueueStart, commitLockedAt};
        // Prepare every key, new map entry and record copy before modifying an
        // existing owner. Allocation failure must leave all promotion leases
        // and all L1/L2 records unchanged, just like a stale-lease rejection.
        // Each shard only prepares private state while mu_ excludes all other
        // ledger readers/writers. The map, leases and owners are still published
        // once, below, after every shard has validated successfully.
        constexpr size_t kCommitShards = 16;
        const size_t shardCount = adoptions.size() >= 4096 ? kCommitShards : 1;
        std::array<std::vector<size_t>, kCommitShards> adoptionGroups, expectedGroups;
        auto shardOf = [shardCount](const PrefixCachePageIdentity &id) {
            return (std::hash<std::string>()(id.pageHash) ^
                    (std::hash<int32_t>()(id.layer) << 1) ^
                    (std::hash<int32_t>()(id.rank) << 3)) % shardCount;
        };
        if (expected && expected->size() != adoptions.size()) return false;
        for (size_t i = 0; i < adoptions.size(); ++i)
            adoptionGroups[shardOf(AdoptionIdentity(adoptions[i]))].push_back(i);
        if (expected)
            for (size_t i = 0; i < expected->size(); ++i)
                expectedGroups[shardOf((*expected)[i])].push_back(i);
        std::array<std::unordered_set<std::string>, kCommitShards> seen;
        std::array<size_t, kCommitShards> expectedChars{}, adoptionChars{}, newCounts{};
        auto runShards = [shardCount](const auto &task) {
            std::atomic<bool> valid{true};
            auto run = [&](size_t shard) {
                try { if (!task(shard)) valid.store(false); }
                catch (...) { valid.store(false); }
            };
            std::vector<std::thread> threads;
            try {
                threads.reserve(shardCount - 1);
                for (size_t shard = 1; shard < shardCount; ++shard)
                    threads.emplace_back([&, shard] { run(shard); });
                run(0);
            } catch (...) { valid.store(false); }
            for (auto &thread : threads) if (thread.joinable()) thread.join();
            return valid.load();
        };
        if (expected) {
            const auto expectedStart = commitMark();
            if (!runShards([&](size_t shard) {
                auto &unique = seen[shard];
                unique.reserve(expectedGroups[shard].size() * 2 + 1);
                size_t chars = 0;
                for (size_t index : expectedGroups[shard]) {
                    const auto &identity = (*expected)[index];
                    if (!identity.HasCanonicalFields()) return false;
                    auto key = identity.CanonicalKey();
                    chars += key.size();
                    if (!unique.insert(std::move(key)).second) return false;
                }
                expectedChars[shard] = chars;
                return true;
            })) return false;
            LastCommitTiming().expectedMs = commitSpanMs(expectedStart, commitMark());
        }
        struct OwnerHash {
            size_t operator()(const std::pair<const void *, int64_t> &value) const {
                return std::hash<const void *>()(value.first) ^
                       (std::hash<int64_t>()(value.second) << 1);
            }
        };
        const auto adoptionStart = commitMark();
        std::unordered_set<std::pair<const void *, int64_t>, OwnerHash> owners;
        owners.reserve(adoptions.size() * 2 + 1);
        for (const auto &adoption : adoptions) {
            if (!AdoptionIdentity(adoption).HasCanonicalFields() || !adoption.manager || adoption.page < 0 ||
                !owners.insert({adoption.manager, adoption.page}).second) return false;
        }
        std::vector<std::shared_ptr<PageRecord>> records(adoptions.size());
        std::vector<std::string> keys(adoptions.size());
        std::vector<uint8_t> newRecords(adoptions.size(), 0), copied;
        copied.reserve(adoptions.size());
        const auto &readPages = pages_;
        if (!runShards([&](size_t shard) {
            auto &unique = seen[shard];
            if (!expected) unique.reserve(adoptionGroups[shard].size() * 2 + 1);
            size_t chars = 0, fresh = 0;
            for (size_t index : adoptionGroups[shard]) {
                const auto &adoption = adoptions[index];
                auto key = AdoptionIdentity(adoption).CanonicalKey();
                chars += key.size();
                if (expected) {
                    auto found = unique.find(key);
                    if (found == unique.end()) return false;
                    unique.erase(found);
                } else if (!unique.insert(key).second) return false;
                const auto it = readPages.find(key);
                std::shared_ptr<PageRecord> record;
                if (adoption.lease.Valid()) {
                    if (adoption.lease.key != key ||
                        !ValidateLeaseLocked(adoption.lease, PrefixCacheLeaseKind::PROMOTION, record))
                        return false;
                } else if (it != readPages.end()) {
                    record = it->second;
                    if (record->failureQuarantined || record->activeLeaseSerial != 0 ||
                        record->runtimeTier == PrefixCacheRuntimeTier::L2 ||
                        record->runtimeTier == PrefixCacheRuntimeTier::MIGRATING ||
                        (record->runtimeTier == PrefixCacheRuntimeTier::L1 &&
                         (record->l1Manager != adoption.manager || record->l1Page != adoption.page)))
                        return false;
                } else {
                    record = std::make_shared<PageRecord>();
                    record->identity = AdoptionIdentity(adoption);
                }
                keys[index] = std::move(key);
                newRecords[index] = it == readPages.end() ? 1 : 0;
                fresh += newRecords[index];
                records[index] = std::move(record);
            }
            adoptionChars[shard] = chars;
            newCounts[shard] = fresh;
            return true;
        })) return false;
        size_t newCount = 0;
        {
            const auto now = commitMark();
            CommitTiming &timing = LastCommitTiming();
            timing.prepareMs = commitSpanMs(commitPrev, now);
            timing.adoptionMs = commitSpanMs(adoptionStart, now);
            timing.keyCount = adoptions.size();
            for (size_t shard = 0; shard < shardCount; ++shard) {
                newCount += newCounts[shard];
                timing.keyChars += expectedChars[shard] + adoptionChars[shard];
            }
            commitPrev = now;
        }
        auto rollbackNewEntries = [this, &records, &keys, &newRecords]() {
            for (size_t index = 0; index < records.size(); ++index) {
                if (!newRecords[index]) continue;
                auto inserted = pages_.find(keys[index]);
                if (inserted != pages_.end() && inserted->second == records[index])
                    pages_.erase(inserted);
            }
        };
        // reserve/emplace may allocate. No old owner has changed yet, and the
        // mutex prevents readers from observing temporary new entries.
        try {
            pages_.reserve(pages_.size() + newCount);
            for (size_t index = 0; index < records.size(); ++index)
                if (newRecords[index]) pages_.emplace(keys[index], records[index]);
        } catch (...) {
            rollbackNewEntries();
            return false;
        }
        // 写时复制：只复制仍被他人共享的既有记录。仍没有任何既有记录被修改，
        // 因此这里的分配失败只需回滚新条目。
        try {
            copied.assign(records.size(), 0);
            for (size_t index = 0; index < records.size(); ++index) {
                if (newRecords[index] || records[index].use_count() <= 1) continue;
                records[index] = std::make_shared<PageRecord>(*records[index]);
                copied[index] = 1;
            }
        } catch (...) {
            rollbackNewEntries();
            return false;
        }
        {
            const auto now = commitMark();
            CommitTiming &timing = LastCommitTiming();
            timing.allocMs = commitSpanMs(commitPrev, now);
            commitPrev = now;
        }
        // 从这里开始不再分配：原地修改未共享的记录、复制品随后 swap 进 map。
        for (size_t index = 0; index < adoptions.size(); ++index) {
            const auto &adoption = adoptions[index];
            auto &record = *records[index];
            record.runtimeTier = PrefixCacheRuntimeTier::L1;
            record.runtimeState = PrefixCacheRuntimeState::RESIDENT;
            record.l1Manager = adoption.manager;
            record.l1Page = adoption.page;
            record.l1Generation = adoption.generation;
            record.pageGeneration = adoption.generation;
            record.cacheIdentity = keys[index];
            record.h2dEvent = adoption.h2dEvent;
            record.producerEvent = adoption.producerEvent ? adoption.producerEvent : adoption.h2dEvent;
            record.d2hEvent.reset();
            record.l2HostAllocation = 0;
            ConsumeLeaseLocked(record);
        }
        {
            const auto now = commitMark();
            CommitTiming &timing = LastCommitTiming();
            timing.mutateMs = commitSpanMs(commitPrev, now);
            commitPrev = now;
        }
        for (size_t index = 0; index < records.size(); ++index)
            if (copied[index]) pages_.find(keys[index])->second.swap(records[index]);
        TrimInactiveLocked();
        {
            const auto now = commitMark();
            CommitTiming &timing = LastCommitTiming();
            timing.trimMs = commitSpanMs(commitPrev, now);
            timing.pagesAfter = pages_.size();
        }
        return true;
    }

public:

    bool AbortPromotion(const Lease &lease, const std::string &reason = std::string()) {
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::PROMOTION, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::L2;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        if (!reason.empty()) record->failureReason = reason;
        ConsumeLeaseLocked(*record);
        return true;
    }

    bool CanEvictL2(const PrefixCachePageIdentity &identity) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        return it != pages_.end() && CanEvictLocked(*it->second);
    }

    // The storage layer owns object reachability/reader handles.  The ledger
    // only vetoes an ERASE when this root is still actively writing/reading or
    // is the sole durable recovery root for a resident-less page.
    // DESIGN-v3-segments C8 (S9): with FASTLLM_PREFIX_V4_SEGMENTS=1 a durable root this process
    // never tracked (written before a restart) holds no page here and may be retired by FIFO, the
    // same rule as CanRetireCorruptCheckpoint below. Flag off: unchanged (vetoed).
    // Same rule as DiskPrefixV5Enabled() (disk_prefix_v5.cpp); repeated here because the ledger is header-only.
    static bool V5StoreEnabled() {
        static const bool enabled = [] {
            const char *value = std::getenv("FASTLLM_PREFIX_V5");
            return value == nullptr || std::string(value) != "0";
        }();
        return enabled;
    }
    // L2 host KV (FASTLLM_PREFIX_CACHE_L2_HOST_KV, same rule as PublisherPageHashIsChain): with L2 on, pages
    // migrate L1 -> L2 and a reclaim can leave the SSD root as the only copy, so the ledger must keep durable
    // rows and roots ("never delete the last good backup"); the V5 memory defaults below apply only with L2 off.
    static bool L2HostKvEnabled() {
        static const bool enabled = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_L2_HOST_KV");
            return text != nullptr && text[0] != '0';
        }();
        return enabled;
    }
    static bool V5LeanLedger() { return V5StoreEnabled() && !L2HostKvEnabled(); }
    // An untracked root (after a restart, or forgotten by the ledger bound) may be reclaimed by the store.
    // FASTLLM_PREFIX_V4_SEGMENTS=1 / other value decides explicitly; unset = on with V5 (the V5 store keeps
    // in-use files alive with its own pins and writer leases), off with V4 (unchanged).
    static bool UnknownCheckpointErasable() {
        static const bool enabled = [] {
            const char *value = std::getenv("FASTLLM_PREFIX_V4_SEGMENTS");
            if (value == nullptr) return V5LeanLedger();
            return value[0] == '1' && value[1] == '\0';
        }();
        return enabled;
    }
    bool CanEraseCheckpoint(const std::string &key, const std::string &checkpointIdentity) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end()) return UnknownCheckpointErasable();
        // Diagnostic (segments flag only): which guard vetoes; the decision itself is unchanged.
        const bool diag = UnknownCheckpointErasable();
        size_t pageIndex = 0, pagesTotal = it->second.pages.size();
        auto veto = [&](int code, const PageRecord *page) {
            if (!diag) return false;
            ++vetoReasonCounts_[code];
            char detail[384];
            std::snprintf(detail, sizeof(detail),
                "reason=%s cp_state=%d writer=%d readers=%llu page=%zu/%zu tier=%d rstate=%d dstate=%d "
                "refs=%llu lease=%llu lease_kind=%d durable_roots=%zu quarantined=%d",
                VetoReasonName(code), int(it->second.state), int(it->second.writerActive),
                (unsigned long long)it->second.activeReaders, pageIndex, pagesTotal,
                page ? int(page->runtimeTier) : -1, page ? int(page->runtimeState) : -1,
                page ? int(page->durableState) : -1,
                page ? (unsigned long long)page->runtimeRefCount : 0ull,
                page ? (unsigned long long)page->activeLeaseSerial : 0ull,
                page ? int(page->activeLeaseKind) : -1,
                page ? page->durableCheckpointRoots.size() : size_t(0),
                page ? int(page->failureQuarantined) : -1);
            lastVetoDetail_ = detail;
            return false;
        };
        if (it->second.identity != checkpointIdentity) return veto(0, nullptr);
        if (it->second.state != CheckpointState::DURABLE) return veto(1, nullptr);
        if (it->second.writerActive) return veto(2, nullptr);
        if (it->second.activeReaders != 0) return veto(3, nullptr);
        for (const auto &identity : it->second.pages) {
            auto pageIt = pages_.find(identity.CanonicalKey());
            if (pageIt == pages_.end()) { ++pageIndex; continue; }
            const auto &record = *pageIt->second;
            if (record.activeLeaseSerial != 0) return veto(4, &record);
            // Segments (user 2026-11-06: SSD data is disposable): GPU use of a page and "SSD is the
            // last copy" do not block deleting the SSD root; EraseCheckpoint then drops this root from
            // the page, so the page is no longer counted as backed up. Counted, not vetoed.
            if (record.runtimeRefCount != 0) {
                if (!diag) return false;
                ++vetoReasonCounts_[5];
            }
            if (record.runtimeTier == PrefixCacheRuntimeTier::MIGRATING) return veto(6, &record);
            if (record.failureQuarantined) return veto(7, &record);
            if (record.runtimeTier == PrefixCacheRuntimeTier::NONE &&
                record.durableCheckpointRoots.size() <= 1) {
                if (!diag) return false;
                ++vetoReasonCounts_[8];
            }
            ++pageIndex;
        }
        return true;
    }
    static const char *VetoReasonName(int code) {
        static const char *names[] = {"identity", "cp_not_durable", "writer_active", "active_readers",
                                      "page_lease", "page_runtime_ref", "page_migrating",
                                      "page_quarantined", "page_sole_root_nonresident"};   // 5 and 8: counted only (segments)
        return code >= 0 && code < 9 ? names[code] : "unknown";
    }
    std::string VetoDiagnostics() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::string out = "veto_counts=";
        for (int code = 0; code < 9; ++code) {
            if (code) out += ",";
            out += VetoReasonName(code);
            out += ":" + std::to_string(vetoReasonCounts_[code]);
        }
        out += " last={" + lastVetoDetail_ + "}";
        return out;
    }

    bool CanEraseCheckpoint(const std::string &key,
                            const PrefixCachePageIdentity &checkpointIdentity) const {
        return CanEraseCheckpoint(key, checkpointIdentity.CanonicalKey());
    }

    // Proof-gated corrupt retirement keeps every activity/identity guard but
    // omits only the healthy sole-root veto: the confirmed bad root is not a
    // usable recovery backup. An empty ledger is expected after restart.
    bool CanRetireCorruptCheckpoint(const std::string &key,
                                    const std::string &checkpointIdentity) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end()) return true;
        if (it->second.identity != checkpointIdentity ||
            it->second.state != CheckpointState::DURABLE || it->second.writerActive ||
            it->second.activeReaders != 0) return false;
        for (const auto &identity : it->second.pages) {
            auto pageIt = pages_.find(identity.CanonicalKey());
            if (pageIt == pages_.end()) continue;
            const auto &record = *pageIt->second;
            if (record.activeLeaseSerial != 0 || record.runtimeRefCount != 0 ||
                record.runtimeTier == PrefixCacheRuntimeTier::MIGRATING ||
                record.failureQuarantined) return false;
        }
        return true;
    }

    bool CanRetireCorruptCheckpoint(const std::string &key,
                                    const PrefixCachePageIdentity &checkpointIdentity) const {
        return CanRetireCorruptCheckpoint(key, checkpointIdentity.CanonicalKey());
    }

    Lease BeginEviction(const PrefixCachePageIdentity &identity) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = pages_.find(identity.CanonicalKey());
        if (it == pages_.end() || !CanEvictLocked(*it->second)) return Lease();
        auto &record = *it->second;
        record.runtimeTier = PrefixCacheRuntimeTier::MIGRATING;
        record.runtimeState = PrefixCacheRuntimeState::EVICT_PENDING;
        return MakeLeaseLocked(record, identity.CanonicalKey(), PrefixCacheLeaseKind::EVICTION);
    }

    bool CommitEviction(const Lease &lease) {
        std::lock_guard<std::mutex> lock(mu_);
        std::shared_ptr<PageRecord> record;
        if (!ValidateLeaseLocked(lease, PrefixCacheLeaseKind::EVICTION, record)) return false;
        record->runtimeTier = PrefixCacheRuntimeTier::NONE;
        record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
        record->l2HostAllocation = 0;
        ConsumeLeaseLocked(*record);
        return true;
    }

    // Returns zero if the key is already live or if the page list is invalid.
    uint64_t RegisterCheckpoint(const std::string &key, const std::string &checkpointIdentity,
                                const std::vector<PrefixCachePageIdentity> &pages,
                                uint64_t sequence = 0) {
        if (key.empty() || checkpointIdentity.empty() || pages.empty()) return 0;
        std::lock_guard<std::mutex> lock(mu_);
        auto existing = checkpoints_.find(key);
        if (existing != checkpoints_.end() && existing->second.state != CheckpointState::ERASED &&
            existing->second.state != CheckpointState::FAILED) {
            return 0;
        }
        CheckpointRecord root;
        root.key = key;
        root.identity = checkpointIdentity;
        root.generation = existing == checkpoints_.end() ? 1 : existing->second.generation + 1;
        root.sequence = sequence == 0 ? nextCheckpointSequence_++ : sequence;
        if (sequence != 0 && sequence >= nextCheckpointSequence_) nextCheckpointSequence_ = sequence + 1;
        root.state = CheckpointState::WRITE_PENDING;
        root.writerActive = true;
        std::unordered_set<std::string> seen;
        for (const auto &identity : pages) {
            if (!identity.HasCanonicalFields() || !seen.insert(identity.CanonicalKey()).second) return 0;
            root.pages.push_back(identity);
        }
        // Validate the complete list before creating any page records. A failed
        // registration must not leave behind write-pending entries.
        for (const auto &identity : root.pages) {
            auto it = pages_.find(identity.CanonicalKey());
            if (it == pages_.end()) {
                auto record = std::make_shared<PageRecord>();
                record->identity = identity;
                record->runtimeTier = PrefixCacheRuntimeTier::NONE;
                record->runtimeState = PrefixCacheRuntimeState::RESIDENT;
                record->durableState = PrefixCacheDurableState::WRITE_PENDING;
                record->pendingWriters = 1;
                pages_.emplace(identity.CanonicalKey(), record);
            } else {
                ++it->second->pendingWriters;
                if (it->second->durableState == PrefixCacheDurableState::NONE)
                    it->second->durableState = PrefixCacheDurableState::WRITE_PENDING;
            }
        }
        checkpoints_[key] = std::move(root);
        TrimInactiveLocked();
        TrimCheckpointRecordsLocked();
        return checkpoints_[key].generation;
    }

    uint64_t RegisterCheckpoint(const std::string &key,
                                const PrefixCachePageIdentity &checkpointIdentity,
                                const std::vector<PrefixCachePageIdentity> &pages,
                                uint64_t sequence = 0) {
        return RegisterCheckpoint(key, checkpointIdentity.CanonicalKey(), pages, sequence);
    }

    bool FinishCheckpoint(const std::string &key, const std::string &checkpointIdentity,
                          uint64_t generation, bool success,
                          const std::string &reason = std::string()) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end() || it->second.identity != checkpointIdentity ||
            it->second.generation != generation || it->second.state != CheckpointState::WRITE_PENDING) {
            return false;
        }
        auto &root = it->second;
        root.writerActive = false;
        root.state = success ? CheckpointState::DURABLE : CheckpointState::FAILED;
        root.failureReason = success ? std::string() : (reason.empty() ? "checkpoint failed" : reason);
        for (const auto &identity : root.pages) {
            auto pageIt = pages_.find(identity.CanonicalKey());
            if (pageIt == pages_.end()) continue;
            auto &record = *pageIt->second;
            if (record.pendingWriters != 0) --record.pendingWriters;
            if (success) {
                if (std::find(record.durableCheckpointRoots.begin(), record.durableCheckpointRoots.end(), key) ==
                    record.durableCheckpointRoots.end()) {
                    record.durableCheckpointRoots.push_back(key);
                    ++rootRefs_;
                }
            } else if (record.durableCheckpointRoots.empty()) {
                record.durableState = PrefixCacheDurableState::FAILED;
            }
            RefreshDurableStateLocked(record);
        }
        TrimCheckpointRecordsLocked();
        if (success) ForgetOldDurableCheckpointsLocked();
        return true;
    }

    bool FinishCheckpoint(const std::string &key,
                          const PrefixCachePageIdentity &checkpointIdentity,
                          uint64_t generation, bool success,
                          const std::string &reason = std::string()) {
        return FinishCheckpoint(key, checkpointIdentity.CanonicalKey(), generation, success, reason);
    }

    // A-6 fix E (2026-11-01, user-approved): read-side restore failure. Only the in-memory
    // checkpoint state changes; SSD bytes and page ownership are kept. CanRestoreCheckpoint
    // then rejects the root while this row is retained (vLLM: a failed load is a miss and is
    // not retried as the same candidate). Bound: TrimCheckpointRecordsLocked may drop FAILED rows
    // above maxCheckpoints_ (65536); the root may then be tried again, and nothing on SSD is ever
    // erased by this path. FinishCheckpoint(false) is a write-terminal
    // transition from WRITE_PENDING and is not used here.
    bool FailRestoreCheckpoint(const std::string &key, const std::string &checkpointIdentity,
                               const std::string &reason) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end() || it->second.identity != checkpointIdentity ||
            it->second.writerActive || it->second.activeReaders != 0) return false;
        if (it->second.state != CheckpointState::DURABLE &&
            it->second.state != CheckpointState::FAILED) return false;
        it->second.state = CheckpointState::FAILED;
        it->second.failureReason = reason.empty() ? "restore failed" : reason;
        return true;
    }

    // After a restart the ledger has no row for a catalog root; record the failure as a
    // FAILED row (no pages are fabricated) so the next request does not select it again.
    bool RegisterFailedRestoreCheckpoint(const std::string &key,
                                         const std::string &checkpointIdentity,
                                         const std::string &reason) {
        if (key.empty() || checkpointIdentity.empty()) return false;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end()) {
            CheckpointRecord root;
            root.key = key;
            root.identity = checkpointIdentity;
            root.generation = 1;
            root.sequence = nextCheckpointSequence_++;
            root.state = CheckpointState::FAILED;
            root.writerActive = false;
            root.failureReason = reason.empty() ? "restore failed after restart" : reason;
            checkpoints_.emplace(key, std::move(root));
            TrimCheckpointRecordsLocked();
            return true;
        }
        if (it->second.identity != checkpointIdentity ||
            it->second.state == CheckpointState::ERASED || it->second.writerActive) return false;
        it->second.state = CheckpointState::FAILED;
        it->second.failureReason = reason.empty() ? "restore failed after restart" : reason;
        return true;
    }

    bool EraseCheckpoint(const std::string &key, const std::string &checkpointIdentity) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end() || it->second.identity != checkpointIdentity ||
            it->second.state != CheckpointState::DURABLE || it->second.writerActive ||
            it->second.activeReaders != 0) {
            return false;
        }
        it->second.state = CheckpointState::ERASED;
        for (const auto &identity : it->second.pages) {
            auto pageIt = pages_.find(identity.CanonicalKey());
            if (pageIt != pages_.end()) {
                auto &roots = pageIt->second->durableCheckpointRoots;
                const size_t before = roots.size();
                roots.erase(std::remove(roots.begin(), roots.end(), key), roots.end());
                rootRefs_ -= before - roots.size();
                RefreshDurableStateLocked(*pageIt->second);
            }
        }
        TrimCheckpointRecordsLocked();
        return true;
    }

    bool EraseCheckpoint(const std::string &key,
                         const PrefixCachePageIdentity &checkpointIdentity) {
        return EraseCheckpoint(key, checkpointIdentity.CanonicalKey());
    }

    bool SetCheckpointActivity(const std::string &key, const std::string &checkpointIdentity,
                               uint64_t activeReaders, bool writerActive) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end() || it->second.identity != checkpointIdentity ||
            it->second.state != CheckpointState::DURABLE) return false;
        it->second.activeReaders = activeReaders;
        it->second.writerActive = writerActive;
        return true;
    }

    bool Checkpoint(const std::string &key, CheckpointSnapshot &out) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end()) return false;
        out = ToSnapshot(it->second);
        return true;
    }

    // A failed/partial/quarantined root is never a restore source. This check
    // is deliberately separate from Checkpoint() so diagnostics can still
    // inspect failed records without making them usable.
    bool CanRestoreCheckpoint(const std::string &key,
                              const std::string &checkpointIdentity) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = checkpoints_.find(key);
        if (it == checkpoints_.end() || it->second.identity != checkpointIdentity ||
            it->second.state != CheckpointState::DURABLE || it->second.writerActive) return false;
        for (const auto &identity : it->second.pages) {
            auto pageIt = pages_.find(identity.CanonicalKey());
            if (pageIt == pages_.end()) return false;
            const auto &record = *pageIt->second;
            if (record.failureQuarantined || record.runtimeState == PrefixCacheRuntimeState::FAILED ||
                record.durableState == PrefixCacheDurableState::FAILED ||
                record.durableState == PrefixCacheDurableState::WRITE_PENDING) return false;
        }
        return true;
    }

    // FIFO candidate selection is only a read operation; callers still need to
    // call EraseCheckpoint and handle a concurrent failure/retry.
    std::string OldestEvictableCheckpoint() const {
        std::lock_guard<std::mutex> lock(mu_);
        const CheckpointRecord *best = nullptr;
        for (const auto &entry : checkpoints_) {
            const auto &root = entry.second;
            if (root.state != CheckpointState::DURABLE || root.writerActive || root.activeReaders != 0) continue;
            bool safe = true;
            for (const auto &identity : root.pages) {
                auto it = pages_.find(identity.CanonicalKey());
                if (it == pages_.end() || !HasAlternativeRecoveryLocked(*it->second, root.key)) {
                    safe = false;
                    break;
                }
            }
            if (safe && (best == nullptr || root.sequence < best->sequence)) best = &root;
        }
        return best == nullptr ? std::string() : best->key;
    }

    // Drop only records with no runtime owner, lease, reference, durable root,
    // or quarantine.  This keeps the identity index bounded without erasing a
    // page that could still be recovered or is still referenced in flight.
    size_t TrimInactive(size_t target = 0) {
        std::lock_guard<std::mutex> lock(mu_);
        return TrimInactiveLocked(target);
    }

private:
    struct CheckpointRecord : CheckpointSnapshot {};

    static uint64_t NextLedgerId() {
        static std::atomic<uint64_t> next(1);
        return next.fetch_add(1);
    }

    Lease MakeLeaseLocked(PageRecord &record, const std::string &key, PrefixCacheLeaseKind kind) {
        Lease lease;
        lease.ledgerId = ledgerId_;
        // Finish all potentially-throwing fields before publishing the active
        // lease into the page record. An allocation failure must not strand a
        // record in MIGRATING with no Lease object available to abort it.
        lease.key = key;
        lease.serial = ++nextLease_;
        lease.kind = kind;
        record.activeLeaseSerial = lease.serial;
        record.activeLeaseKind = kind;
        return lease;
    }

    bool ValidateLeaseLocked(const Lease &lease, PrefixCacheLeaseKind kind,
                             std::shared_ptr<PageRecord> &record) const {
        if (!lease.Valid() || lease.ledgerId != ledgerId_ || lease.kind != kind) return false;
        auto it = pages_.find(lease.key);
        if (it == pages_.end() || it->second->activeLeaseSerial != lease.serial ||
            it->second->activeLeaseKind != kind) return false;
        record = it->second;
        return true;
    }

    static void ConsumeLeaseLocked(PageRecord &record) {
        record.activeLeaseSerial = 0;
        record.activeLeaseKind = PrefixCacheLeaseKind::NONE;
    }

    static void RefreshDurableStateLocked(PageRecord &record) {
        if (!record.durableCheckpointRoots.empty()) {
            record.durableState = PrefixCacheDurableState::DURABLE;
        } else if (record.durableState != PrefixCacheDurableState::FAILED) {
            record.durableState = PrefixCacheDurableState::NONE;
        }
    }

    static bool HasDurableRootLocked(const PageRecord &record) {
        return !record.durableCheckpointRoots.empty() &&
               record.durableState == PrefixCacheDurableState::DURABLE;
    }

    static bool CanEvictLocked(const PageRecord &record) {
        return record.runtimeTier == PrefixCacheRuntimeTier::L2 &&
               record.runtimeState == PrefixCacheRuntimeState::RESIDENT &&
               record.runtimeRefCount == 0 && record.activeLeaseSerial == 0 &&
               record.l2HostAllocation != 0 && !record.failureQuarantined &&
               HasDurableRootLocked(record);
    }

    bool HasAlternativeRecoveryLocked(const PageRecord &record, const std::string &rootKey) const {
        if (record.runtimeTier == PrefixCacheRuntimeTier::L1 || record.runtimeTier == PrefixCacheRuntimeTier::L2) return true;
        for (const auto &root : record.durableCheckpointRoots) {
            if (root != rootKey) {
                auto it = checkpoints_.find(root);
                if (it != checkpoints_.end() && it->second.state == CheckpointState::DURABLE) return true;
            }
        }
        return false;
    }

    static bool IsInactiveTrimCandidate(const PageRecord &record) {
        return record.runtimeTier == PrefixCacheRuntimeTier::NONE && record.pendingWriters == 0 &&
               record.runtimeRefCount == 0 && record.activeLeaseSerial == 0 &&
               !record.failureQuarantined && record.durableCheckpointRoots.empty() &&
               record.durableState != PrefixCacheDurableState::WRITE_PENDING;
    }

    // ObserveL1 runs once per physical layer/K/V/TP page. A soft limit can be
    // exceeded entirely by live or durable records; an unsuccessful full sweep
    // on every observation then costs O(pages * observations). Resume a bounded
    // sweep instead. A canonical key survives unordered_map rehash; never keep
    // an iterator across calls, insertions, rollback erases or batch commits.
    size_t TrimInactiveBoundedLocked() {
        constexpr size_t kMaxVisits = 64;
        if (pages_.size() <= maxRecords_) {
            trimResumeKey_.clear();
            return 0;
        }
        auto it = trimResumeKey_.empty() ? pages_.begin() : pages_.find(trimResumeKey_);
        if (it == pages_.end()) it = pages_.begin();
        size_t visited = 0, removed = 0;
        while (it != pages_.end() && pages_.size() > maxRecords_ && visited < kMaxVisits) {
            auto candidate = it++;
            ++visited;
            if (IsInactiveTrimCandidate(*candidate->second)) {
                pages_.erase(candidate);
                ++removed;
            }
        }
        // Reaching the end wraps on the next observation. Inactive transitions
        // made after an earlier unsuccessful visit must become visible again.
        if (it == pages_.end() || pages_.size() <= maxRecords_) trimResumeKey_.clear();
        else trimResumeKey_ = it->first;
        return removed;
    }

public:
    // LEDGERBOUND diagnostics: O(1) counters, no map walk (printed by the SSD watermark when enabled).
    struct LedgerStats {
        size_t checkpoints = 0, pages = 0, rootRefs = 0, durablePageRefs = 0, forgotten = 0;
    };
    LedgerStats Stats() const {
        std::lock_guard<std::mutex> lock(mu_);
        LedgerStats out;
        out.checkpoints = checkpoints_.size(); out.pages = pages_.size(); out.rootRefs = rootRefs_;
        out.durablePageRefs = durablePageRefs_; out.forgotten = forgottenCheckpoints_;
        return out;
    }
    // FASTLLM_PREFIX_LEDGER_BOUND=1 (strict "1", default off): bound the in-memory DURABLE checkpoint
    // rows. Each row keeps its full page list and adds its key to every page's root list, and a long
    // request checkpoints every chunk with all pages from the start, so without a bound the ledger
    // grows with the square of request length and never shrinks (measured: logs/a6-memprobe-20261103).
    // The oldest idle DURABLE rows beyond FASTLLM_PREFIX_LEDGER_MAX_PAGE_REFS page references
    // (default 1048576) are forgotten exactly as after a restart: SSD bytes stay, restore of that
    // root is decided as for an untracked root, and FIFO retirement of that root is vetoed (kept).
    // The newest idle row and any row with a page resident in L2 are never forgotten.
    // Default ON since 2026-11-08 (memdiag md2: the unbounded ledger was the main host-memory growth);
    // FASTLLM_PREFIX_LEDGER_BOUND=0 (any value other than "1") turns it off.
    static bool LedgerBoundEnabled() {
        static const bool enabled = [] {
            const char *v = std::getenv("FASTLLM_PREFIX_LEDGER_BOUND");
            return v == nullptr || (v[0] == '1' && v[1] == '\0');
        }();
        return enabled;
    }
    static bool LedgerStatsEnabled() {
        static const bool enabled = [] {
            const char *v = std::getenv("FASTLLM_PREFIX_LEDGER_STATS");
            return LedgerBoundEnabled() || (v != nullptr && v[0] == '1' && v[1] == '\0');
        }();
        return enabled;
    }
    // Soft cap on page records before a full sweep removes inactive ones (FASTLLM_PREFIX_LEDGER_MAX_RECORDS=N,
    // default 262144). The sweep runs inside L1 commits (restore path), so it must stay rare: with cap 1 it
    // visited every live record on every commit (memfix mf2: commit trim 0.02 -> 35 ms, image restore -30 %).
    // With the V5 lean ledger, records freed by forgetting a row are erased right there instead
    // (ForgetOldDurableCheckpointsLocked), per page like vLLM's _maybe_evict_cached_block.
    static size_t LedgerDefaultMaxRecords() {
        static const size_t value = [] {
            const char *v = std::getenv("FASTLLM_PREFIX_LEDGER_MAX_RECORDS");
            const long long parsed = (v != nullptr && *v != '\0') ? std::atoll(v) : 0;
            return parsed > 0 ? size_t(parsed) : size_t(262144);
        }();
        return value;
    }
    // Unset: 0 with V5 (keep only the newest idle durable row; the V5 store has its own index of what is
    // on SSD, like vLLM keeps block-hash metadata only for blocks still in the GPU pool,
    // vllm/v1/core/block_pool.py _maybe_evict_cached_block @83c078d9), 1048576 with V4 (unchanged).
    static size_t LedgerMaxDurablePageRefs() {
        static const size_t value = [] {
            const char *v = std::getenv("FASTLLM_PREFIX_LEDGER_MAX_PAGE_REFS");
            if (v == nullptr || *v == '\0') return V5LeanLedger() ? size_t(0) : size_t(1) << 20;
            const long long parsed = std::atoll(v);
            return parsed >= 0 ? size_t(parsed) : size_t(1) << 20;
        }();
        return value;
    }

private:
    size_t ForgetOldDurableCheckpointsLocked() {
        if (!LedgerBoundEnabled()) return 0;
        std::vector<std::pair<uint64_t, std::string>> idle;
        size_t refs = 0;
        for (const auto &entry : checkpoints_) {
            const auto &row = entry.second;
            if (row.state == CheckpointState::DURABLE && !row.writerActive && row.activeReaders == 0) {
                idle.emplace_back(row.sequence, entry.first);
                refs += row.pages.size();
            }
        }
        const size_t budget = LedgerMaxDurablePageRefs();
        size_t forgotten = 0;
        if (refs > budget && idle.size() > 1) {
            std::sort(idle.begin(), idle.end());
            for (size_t i = 0; i + 1 < idle.size() && refs > budget; ++i) {
                auto it = checkpoints_.find(idle[i].second);
                if (it == checkpoints_.end()) continue;
                bool inL2 = false;
                for (const auto &identity : it->second.pages) {
                    auto pageIt = pages_.find(identity.CanonicalKey());
                    if (pageIt != pages_.end() && pageIt->second->runtimeTier == PrefixCacheRuntimeTier::L2) {
                        inL2 = true;
                        break;
                    }
                }
                if (inL2) continue;
                const bool eraseFreed = V5LeanLedger();
                for (const auto &identity : it->second.pages) {
                    auto pageIt = pages_.find(identity.CanonicalKey());
                    if (pageIt == pages_.end()) continue;
                    auto &roots = pageIt->second->durableCheckpointRoots;
                    const size_t before = roots.size();
                    roots.erase(std::remove(roots.begin(), roots.end(), idle[i].second), roots.end());
                    rootRefs_ -= before - roots.size();
                    RefreshDurableStateLocked(*pageIt->second);
                    // The record now describes nothing tracked (not resident, no refs/lease/roots): drop it here,
                    // O(1), instead of leaving it for a full sweep on the restore path.
                    if (eraseFreed && IsInactiveTrimCandidate(*pageIt->second)) pages_.erase(pageIt);
                }
                refs -= it->second.pages.size();
                checkpoints_.erase(it);
                ++forgotten;
            }
        }
        durablePageRefs_ = refs;
        forgottenCheckpoints_ += forgotten;
        if (forgotten != 0) TrimInactiveLocked();
        return forgotten;
    }

    // Explicit TrimInactive and existing batch/checkpoint call sites keep the
    // full-sweep contract. Visit each entry once rather than restarting at begin
    // after every removal; the same eligible entries are removed in map order.
    size_t TrimInactiveLocked(size_t target = 0) {
        const size_t limit = target == 0 ? maxRecords_ : std::max<size_t>(1, target);
        size_t removed = 0;
        auto it = pages_.begin();
        while (it != pages_.end() && pages_.size() > limit) {
            auto candidate = it++;
            if (IsInactiveTrimCandidate(*candidate->second)) {
                pages_.erase(candidate);
                ++removed;
            }
        }
        trimResumeKey_.clear();
        return removed;
    }

    size_t TrimCheckpointRecordsLocked() {
        size_t removed = 0;
        while (checkpoints_.size() > maxCheckpoints_) {
            auto victim = checkpoints_.end();
            for (auto it = checkpoints_.begin(); it != checkpoints_.end(); ++it) {
                if ((it->second.state == CheckpointState::ERASED ||
                     it->second.state == CheckpointState::FAILED) &&
                    !it->second.writerActive && it->second.activeReaders == 0) {
                    victim = it;
                    break;
                }
            }
            if (victim == checkpoints_.end()) break;
            checkpoints_.erase(victim);
            ++removed;
        }
        return removed;
    }

    static CheckpointSnapshot ToSnapshot(const CheckpointRecord &record) {
        CheckpointSnapshot out;
        out.key = record.key;
        out.identity = record.identity;
        out.generation = record.generation;
        out.sequence = record.sequence;
        out.state = record.state;
        out.activeReaders = record.activeReaders;
        out.writerActive = record.writerActive;
        out.pages = record.pages;
        out.failureReason = record.failureReason;
        return out;
    }

    const uint64_t ledgerId_;
    mutable std::mutex mu_;
    uint64_t nextLease_ = 0;
    uint64_t nextToken_ = 0;
    uint64_t nextCheckpointSequence_ = 1;
    size_t maxRecords_ = 262144;
    std::string trimResumeKey_; // One owned key only; no unbounded candidate queue.
    size_t maxCheckpoints_ = 65536;
    size_t rootRefs_ = 0;            // LEDGERBOUND counters (mutex protected)
    size_t durablePageRefs_ = 0;
    size_t forgottenCheckpoints_ = 0;
    std::unordered_map<std::string, std::shared_ptr<PageRecord>> pages_;
    std::unordered_map<uint64_t, std::string> runtimeTokens_;
    std::unordered_map<std::string, CheckpointRecord> checkpoints_;
    mutable uint64_t vetoReasonCounts_[9] = {};   // diagnostic, guarded by mu_
    mutable std::string lastVetoDetail_;
};

enum class PrefixCacheCandidateTier { L1, L2, L3 };
struct PrefixCacheCandidate {
    PrefixCacheCandidateTier tier = PrefixCacheCandidateTier::L1;
    uint64_t tokens = 0;
    bool complete = false;
    bool contiguous = false;
    bool identityValid = false;
    std::vector<PrefixCachePageIdentity> pages;
};

struct PrefixCacheSelection {
    bool valid = false;
    PrefixCacheCandidateTier tier = PrefixCacheCandidateTier::L1;
    uint64_t tokens = 0;
    bool promoteL2 = false;
    bool directToL1 = false;
    std::vector<PrefixCachePageIdentity> pages;
};

inline bool PrefixCacheCandidateIsLegal(const PrefixCacheCandidate &candidate) {
    if (!candidate.complete || !candidate.contiguous || !candidate.identityValid ||
        candidate.tokens == 0 || candidate.pages.empty()) return false;
    const auto &first = candidate.pages.front();
    if (!first.HasCanonicalFields()) return false;
    for (size_t i = 0; i < candidate.pages.size(); ++i) {
        const auto &page = candidate.pages[i];
        if (!page.HasCanonicalFields() || !page.SameStream(first)) return false;
        if (i > 0) {
            const auto &previous = candidate.pages[i - 1];
            if (!page.parentHash.empty() && page.parentHash != previous.pageHash) return false;
            if (page.pageOrdinal != 0 && previous.pageOrdinal != 0 && page.pageOrdinal <= previous.pageOrdinal) return false;
        }
    }
    return true;
}

inline bool SelectLongestPrefix(const std::vector<PrefixCacheCandidate> &candidates,
                               PrefixCacheSelection &out) {
    out = PrefixCacheSelection();
    int bestPriority = std::numeric_limits<int>::max();
    for (const auto &candidate : candidates) {
        if (!PrefixCacheCandidateIsLegal(candidate)) continue;
        const int priority = candidate.tier == PrefixCacheCandidateTier::L1 ? 0 :
                             candidate.tier == PrefixCacheCandidateTier::L2 ? 1 : 2;
        if (!out.valid || candidate.tokens > out.tokens ||
            (candidate.tokens == out.tokens && priority < bestPriority)) {
            out.valid = true;
            out.tier = candidate.tier;
            out.tokens = candidate.tokens;
            out.promoteL2 = candidate.tier == PrefixCacheCandidateTier::L2;
            out.directToL1 = candidate.tier == PrefixCacheCandidateTier::L3;
            out.pages = candidate.pages;
            bestPriority = priority;
        }
    }
    return out.valid;
}

} // namespace fastllm

#endif // FASTLLM_UTILS_PREFIX_CACHE_PAGE_LEDGER_H
