#ifndef FASTLLM_DISK_PREFIX_WRITE_COORDINATOR_H
#define FASTLLM_DISK_PREFIX_WRITE_COORDINATOR_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Frozen interface contract for the write coordinator (SSD_CACHE_REWRITE_PLAN.md
// §6, §6.1, §7). P1 requires these interfaces to be defined and frozen before the
// storage core is written, so that the model adapter, the commit actor and the
// recovery path agree on ownership and failure semantics.
//
// This header declares the contract only. It deliberately contains no
// implementation and no CUDA/SQLite dependency: everything here is CPU-testable.
namespace fastllm {
namespace prefix_write {

using Digest256 = std::string;  // 64 lowercase hex characters

struct TicketId {
    uint64_t requestId = 0;
    uint64_t sequence = 0;
};

inline bool operator==(const TicketId &left, const TicketId &right) {
    return left.requestId == right.requestId && left.sequence == right.sequence;
}
inline bool operator<(const TicketId &left, const TicketId &right) {
    return left.requestId != right.requestId ? left.requestId < right.requestId
                                             : left.sequence < right.sequence;
}

struct ChunkRef {
    Digest256 hash;
    uint64_t bytes = 0;
    uint32_t kind = 0;   // prefix_v4::StorageObjectKind
};

// Only HashPackPool can create a valid result. The digest and its owned bytes
// move together; a caller cannot attach an arbitrary hash to mutable data.
class HashedChunk {
    friend class HashPackPool;
    std::string digest_;
    std::string payload_;
    bool valid_ = false;
    HashedChunk(std::string digest, std::string payload)
        : digest_(std::move(digest)), payload_(std::move(payload)), valid_(true) {}
public:
    HashedChunk() = default;
    HashedChunk(HashedChunk &&) noexcept = default;
    HashedChunk &operator=(HashedChunk &&) noexcept = default;
    HashedChunk(const HashedChunk &) = delete;
    HashedChunk &operator=(const HashedChunk &) = delete;
    bool Valid() const { return valid_ && digest_.size() == 64 && !payload_.empty(); }
    size_t Bytes() const { return payload_.size(); }
    // Consume exactly once. An already consumed result is invalid.
    bool Take(std::string &digest, std::string &payload) {
        if (!Valid()) return false;
        digest = std::move(digest_);
        payload = std::move(payload_);
        valid_ = false;
        return true;
    }
};

// State-ticket lifecycle (§7.2). DURABLE after the index fails must enter
// RECOVERING rather than being appended again as if it had never been written.
enum class TicketState {
    ADMITTED,
    COPYING,
    HOST_READY,
    HASHING,
    PREPARED,
    DURABLE,
    INDEXED,
    COMMITTED,
    RECOVERING,
    FAILED,
    CANCELLED,
};

inline bool TerminalState(TicketState state) {
    return state == TicketState::COMMITTED || state == TicketState::FAILED ||
           state == TicketState::CANCELLED;
}
// A ticket that reached DURABLE must never be silently retried as a fresh
// append; this is the guard that keeps crash recovery honest.
inline bool RetryableAsNewAppend(TicketState state) {
    return state == TicketState::ADMITTED || state == TicketState::COPYING ||
           state == TicketState::HOST_READY || state == TicketState::HASHING ||
           state == TicketState::PREPARED;
}
inline bool DurableButNotIndexed(TicketState state) {
    return state == TicketState::DURABLE || state == TicketState::RECOVERING;
}

// Result of the non-blocking admission step. TryAdmit never waits for I/O, a
// queue, SQLite or fsync; it only describes identity, consults the dedup table
// and reserves budget.
enum class Admission {
    Accepted,             // budget reserved; caller may add the page lease
    Joined,               // an equivalent in-flight ticket already owns the work
    AlreadyDurable,       // an exact durable key already exists
    CapacityUnavailable,  // budget exhausted; caller must not wait here
    Stopping,             // shutdown in progress; caller must not admit
    Conflict,             // §8.5: same key, different semantics - never "no budget"
};

enum class Terminal { Committed, Failed, Cancelled };

// Host budget windows (§7.3). The base window is 112 MiB and every retained
// snapshot, scratch buffer, manifest node, waiter and staging buffer is charged
// to the same ledger, so no accounting path can silently exceed HostLimit.
constexpr int kHashWorkers = 4;
constexpr int kPinnedSlots = 4;                       // global, not per rank（历史默认，见下面 PinnedSlots()）
// W4 / A2：
// D2 中继的 pinned 槽位数**改为运行期可配**（原硬编码 4）。
// 实测依据（`t2-256k-a1`，256K / 191 检查点）：`slot_acquires=12224 slot_waits=11585`
// ——注意 `PinnedSlotPoolImpl::Acquire()` 在无空槽时是"**明确失败、不等待**"（slot_waits++ 后 return -1），
// 所以 94.8% 的获取**直接失败**；`slot_peak=4` 恒满 ⇒ D2 流水线反复停摆 ⇒
// `commit_ms` p50=25 → p99=4407 ms、读端到端 p99=10.92 s ⇒ 用户感知"开了缓存反而更慢"。
// **不变量**：池容量与"每 lane 分配的 pinned 内存"必须同源（`PinnedSlots()` 同时供两者使用），
// 否则槽号会越界。
// **默认保持 `kPinnedSlots`(=4)：不改变 P1 冻结契约**——契约测试
// `test/basic/test_disk_prefix_coordinator_contract.cpp:370/:875/:935` 断言
// `SlotCapacity()==kPinnedSlots`（"K=4，全局共享"），默认改成 8 会直接判红（已实测 3 处 FAIL）。
// 需要更深槽池时**显式**传 `FASTLLM_PREFIX_CACHE_PINNED_SLOTS`（夹在 [1,64]），
// 例如 A2 对照跑用 8（每 lane 8×8 MiB=64 MiB，4 lane 共 256 MiB pinned 主机内存）。
// "默认值该不该从 4 提高"属于契约变更，不在此处改动。
inline int PinnedSlots() {
    static const int value = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_PINNED_SLOTS");
        const int parsed = text != nullptr ? std::atoi(text) : 0;
        if (parsed <= 0) return kPinnedSlots;
        return parsed > 64 ? 64 : parsed;
    }();
    return value;
}
constexpr uint64_t kSlotBytes = 8ull << 20;
constexpr int kHashStagingSlots = 4;
constexpr uint64_t kRecordStagingBytes = 2ull << 20;  // unchanged PendingBytes order
constexpr uint64_t kMetadataWindowBytes = 16ull << 20;
constexpr uint64_t kBaseWindowBytes = 112ull << 20;

// Same ticket completes its own chunks in sequence order; a later checkpoint of
// the same request must never overtake a missing earlier one (§6.1 step 3).
inline bool MayConsume(const TicketId &next, const TicketId &lastConsumed) {
    if (next.requestId != lastConsumed.requestId) return true;
    return next.sequence == lastConsumed.sequence + 1;
}

// The coordinator owns dedup, budget and completion events. Implementations must
// not perform model-math work, and must not be called while holding a model lock
// that a background thread also needs.
class Coordinator {
public:
    virtual ~Coordinator() = default;
    // Non-blocking. `out` is only meaningful when the result is Accepted.
    // `descriptor` is the normalized semantic description §8 requires
    // (identity+prefix+provenance+layout). Empty means the caller has no
    // semantic information: only "same key already in flight" dedup happens
    // (exactly the pre-wiring behaviour); a non-empty descriptor selects the
    // full §8 protocol (durable reuse, waiters, explicit conflict).
    virtual Admission TryAdmit(const TicketId &ticket, const std::string &key,
                               uint64_t estimatedBytes, TicketId &out,
                               const std::string &descriptor = std::string()) = 0;
    // Registers an already-admitted, fully prepared checkpoint for commit.
    virtual Terminal Submit(const TicketId &ticket, std::vector<HashedChunk> chunks) = 0;
    // Full drain, only for shutdown/maintenance. Returns false on timeout; the
    // caller must report the timeout, never print success.
    virtual bool Drain(uint64_t timeoutMs, std::string &detail) = 0;
    virtual TicketState State(const TicketId &ticket) const = 0;
    // 停止接收新工作并等待已接收的工作结束（关闭顺序见方案 §13）。
    virtual void Stop() = 0;
    // 作业结束（成功或失败）后必须调用：释放该 ticket 的预算与槽位并落到终态。
    // 释放与"发布顺序"无关——失败同样要释放，不能遗留 pin/信用（§7.2）。
    virtual void Complete(const TicketId &ticket, Terminal state) = 0;
    // 当前未归还的 host 信用；drain 必须等到它为 0（§13 第 4 条）。
    virtual uint64_t HostCreditsInUse() const = 0;
};

// Host-budget ledger shared by capture, hash and commit. A release without a
// matching acquire is a bug and must be detectable rather than silently
// unbalancing the ledger.
class BudgetLedger {
public:
    virtual ~BudgetLedger() = default;
    virtual bool Acquire(uint64_t bytes) = 0;
    virtual void Release(uint64_t bytes) = 0;
    virtual uint64_t InUse() const = 0;
    virtual uint64_t Peak() const = 0;
    // Completion workspace is reserved before enqueue so that a fully charged
    // ledger can still finish the work it already accepted (§7.3).
    virtual uint64_t ReservedCompletionBytes() const = 0;
};

// 具体实现（disk_prefix_write_coordinator.cpp）。hostLimitBytes 是持久化总保留
// 预算；K 个 pinned slot 与哈希线程池都是全局的，不按 rank 复制。
std::unique_ptr<Coordinator> CreateCoordinator(uint64_t hostLimitBytes);

// K 个 pinned 拷贝槽（§6.1 第 1 条）：全局共享、不按 rank 复制；无空槽时
// Acquire 返回 -1，**不等待**。
class PinnedSlotPool {
public:
    virtual ~PinnedSlotPool() = default;
    virtual int Acquire() = 0;
    virtual void Release(int slot) = 0;
    virtual int InUse() const = 0;
    virtual int Capacity() const = 0;
};
std::unique_ptr<PinnedSlotPool> CreatePinnedSlotPool(int slots, uint64_t slotBytes);

// 固定哈希线程池（§6.1 第 2–3 条）：每个任务用自己的哈希上下文；完成顺序可以乱，
// 但 TakeOrdered 必须**按提交顺序**交付，且不得越过缺失的前序。
class HashPackPool {
public:
    virtual ~HashPackPool() = default;
protected:
    static HashedChunk MakeChunk(std::string digest, std::string payload) {
        return HashedChunk(std::move(digest), std::move(payload));
    }
public:
    // 非阻塞：超出 staging 预算或已停止时返回 false。
    virtual bool Submit(const TicketId &ticket, std::string payload) = 0;
    // 取回下一个"按提交顺序"已完成的结果；超时返回 false。
    virtual bool TakeOrdered(TicketId &ticket, std::string &digest, uint64_t timeoutMs) = 0;
    virtual bool TakeOrderedChunk(TicketId &ticket, HashedChunk &chunk, uint64_t timeoutMs) = 0;
    // §23（④ 第一步）：**一次任务算多块** —— 把一个区段的所有分块放进同一个任务，
    // 线程交接从"每块一次"降到"每区段一次"（此前 8MiB/次交接导致池化负收益）。
    // 默认实现退化为逐块提交，便于既有实现不改也能编译。
    virtual bool SubmitBatch(const TicketId &ticket, std::vector<std::string> payloads) {
        for (auto &payload : payloads) {
            if (!Submit(ticket, std::move(payload))) return false;
        }
        return true;
    }
    virtual bool TakeOrderedBatch(TicketId &ticket, std::vector<HashedChunk> &chunks,
                                  uint64_t timeoutMs) {
        chunks.clear();
        HashedChunk chunk;
        if (!TakeOrderedChunk(ticket, chunk, timeoutMs)) return false;
        chunks.push_back(std::move(chunk));
        return true;
    }
    virtual uint64_t InFlight() const = 0;
    virtual uint64_t InFlightBytes() const = 0;
    // S4：池在途字节的**历史峰值**（单调不减）。用于回答"账外内存到底多大"——
    // 只在 StoreHashed 调用点采样会漏掉批次内部的高点，必须由池自己记。
    virtual uint64_t InFlightPeak() const = 0;
    virtual void Stop() = 0;
};
std::unique_ptr<HashPackPool> CreateHashPackPool(uint64_t stagingBudgetBytes,
                                                   int workers = kHashWorkers);

// §6.1 D2-stream：把"等 D2H 完成 → 哈希 → 定序交付"流水化，且**不改变
// SHA256 的总计算量**。CUDA 事件与 D2H 都由回调注入，因此同一份实现既能接
// 真机，也能在纯 CPU 测试里被完整驱动（这是 §6.1 在测试中落地的形式）。
//
// 生命周期口径（保守）：槽位从 D2H 开始持有，直到该 ticket 的摘要被取回才
// 释放，绝不出现"还在读这个槽位就把它交给下一份拷贝"的情况。
class D2Stream {
public:
    virtual ~D2Stream() = default;
    // 生产者事件是否已完成；未完成时既不能读 GPU page，也不能释放 page lease。
    using EventReady = std::function<bool(uint64_t eventId)>;
    // 执行 D2H 并返回待哈希的 payload。真机上从 pinned 槽位读出；CPU 测试里返回
    // 确定性的假字节。slot 是槽位号，供调用方定位自己的缓冲区。
    using CopyIntoSlot = std::function<std::string(int slot, const TicketId &ticket, size_t bytes)>;

    // R2：预算不足时不再
    // **立即**返回 false，而是进入**有界推迟**——
    //   ① 每次推迟只等待一个固定切片（env FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS，默认 1 ms）；
    //      等待期间不持 mutex_（条件变量语义），预算释放与 Stop() 都会提前唤醒；
    //   ② 推迟轮次上限 env FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS（默认 32，取自 SGLang
    //      HiCache 的 max_staged_admission_defers）；只有超过上限才返回 false；
    //   ③ 条目本身**绝不**因推迟被部分接收：返回 true 才代表已入队，返回 false 时 pending
    //      不变（与旧契约的“不含半个条目”一致）。
    // R7：预算够但延迟水位在高位时，额外做**一次**切片推迟以软化背压，等完仍会接收。
    // 单线程驱动（同一 lane 的 PumpCopy 与 Submit 在同一线程）不会死锁：切片有超时、轮次有
    // 上限，Submit 内部不调用 PumpCopy（避免重入持锁），控制权必然交还调用方。
    virtual bool Submit(const TicketId &ticket, uint64_t eventId, size_t bytes) = 0;
    // 推进拷贝阶段：有空槽就按提交顺序拷贝 event 已完成的 ticket；一次调用可以
    // 连续拷贝多个（这正是 D2H 与哈希重叠的地方）。事件未完成则**不越过**它。
    virtual void PumpCopy() = 0;
    // 按提交顺序取回已完成摘要；超时返回 false。
    virtual bool TakeOrdered(TicketId &ticket, std::string &digest, uint64_t timeoutMs) = 0;
    virtual bool TakeOrderedChunk(TicketId &ticket, HashedChunk &chunk, uint64_t timeoutMs) = 0;
    // §④ 失败出口：被拒（哈希池预算/停止）的 ticket 不会进入定序队列，若不查这个接口，
    // 调用方只能把"失败"当成"还没好"而白等到超时（生产同族超时是 600000 ms）。
    virtual bool Failed(const TicketId &ticket) const = 0;
    virtual int SlotsInUse() const = 0;
    virtual int SlotCapacity() const = 0;
    virtual int SlotPeak() const = 0;
    virtual uint64_t Pending() const = 0;
    virtual void Stop() = 0;
    // A-5：异常/残留清理。丢弃所有**未取回**的条目（不执行拷贝），
    // 归还本流占用的槽位与暂存预算，并把被丢弃的 ticket 记入 failed_（供 Failed() 查询）。
    // 用途：失败路径确定性收尾，杜绝"旧 ticket 留在队首、下一笔作业撞上它"的串味。
    virtual void Reset() = 0;
};
std::unique_ptr<D2Stream> CreateD2Stream(D2Stream::EventReady ready, D2Stream::CopyIntoSlot copy,
                                        uint64_t stagingBudgetBytes,
                                        HashPackPool *sharedPool = nullptr);

// §8 去重协议机制层。
// 去重键必须由 identity + prefix(前缀键) + provenance + layout/encoding 语义共同派生，
// **不接受** token 长度、请求指针或"字符串存在即复用"这类判断。
class DedupRegistry {
public:
    enum class Registration { Owner, Joined, Durable, Conflict };
    static std::string Key(const std::string &identity, const std::string &prefix,
                           const std::string &provenance, const std::string &layout);
    // descriptor 是规范化语义描述（其摘要参与冲突判定）。ticket 为 owner 票据。
    virtual Registration Register(const std::string &key, const std::string &descriptor,
                                  uint64_t ticket, uint64_t &ownerTicket) = 0;
    // 终态：committed=true 时把该 key 标为 durable；false 时**不伪造成功**，
    // waiters 原样返回由调用方置为 FAILED。
    virtual void Complete(const std::string &key, bool committed, uint64_t ticket,
                          std::vector<uint64_t> &waiters) = 0;
    // 完成缓存绑定 generation 且有容量上限；换 generation 即失效。
    virtual void SetGeneration(uint64_t generation) = 0;
    virtual bool LookupDurable(const std::string &key, std::string &descriptor) = 0;
    virtual uint64_t JoinCount() const = 0;
    virtual uint64_t DurableCount() const = 0;
};
std::unique_ptr<DedupRegistry> CreateDedupRegistry(uint64_t generation, size_t maxDurableEntries);
// §8: the coordinator drives the dedup registry. Injection exists so tests and
// diagnostics can read JoinCount/DurableCount and exercise SetGeneration
// without adding more counters to the frozen interface.
std::unique_ptr<Coordinator> CreateCoordinator(uint64_t hostLimitBytes,
                                               std::unique_ptr<DedupRegistry> dedup);

// S1-4「票/额度租约看门狗」：累计被看门狗强制回收（lease_timeout）的票数，单调不减。
// 这是**新增的诊断读取点**，不属于上面冻结的 Coordinator 接口：不新增虚函数、不改任何
// 既有结构与布局，只有读取时才产生差异（不读也完全不影响行为）。打印点见
// `[Prefix Write] lease_timeout:` 日志；是否把它接进快照由快照侧自行决定。
uint64_t PrefixWriteLeaseExpiredTotal();

// A2（2026-10-02）写路径并发化的诊断读取点。与 PrefixWriteLeaseExpiredTotal 同一模式：
// **不新增虚函数、不改任何既有结构与布局**，只在实现内部累加、在这里读取；不读就不影响
// 任何行为。全部为进程内单调不减的累计值（peak 例外，是全局历史最高水位）。
//
// pinned 槽：D2 中继的 kPinnedSlots×kSlotBytes 槽池（每个 D2Stream 一套，per-lane）。
//   acquires = 成功取到槽位的次数；
//   waits    = 想拷贝但池里没有空槽、本轮只能推迟的次数（**不等待**契约的直接证据）；
//   peak     = **所有槽池合计**同时占用槽位数的历史最高值（per-lane 多池时才有意义；
//   reuses   = 取到的槽号此前被用过（复用同一块 pinned 内存）的次数；
//   releases = 归还槽位次数。
struct PrefixWritePinnedSlotStats {
    uint64_t acquires = 0;
    uint64_t waits = 0;
    uint64_t peak = 0;
    uint64_t reuses = 0;
    uint64_t releases = 0;
};
PrefixWritePinnedSlotStats PrefixWritePinnedSlotStatsSnapshot();

// host 额度账本（BudgetLedger）：acquires/releases 是**次数**，denied 是"字节不够被拒"
// 的次数（准入返回 CapacityUnavailable 的直接来源），peakBytes 是账本历史最高占用字节。
struct PrefixWriteHostCreditStats {
    uint64_t acquires = 0;
    uint64_t releases = 0;
    uint64_t denied = 0;
    uint64_t peakBytes = 0;
};
PrefixWriteHostCreditStats PrefixWriteHostCreditStatsSnapshot();

// §8 去重登记的 owner/waiter 计数：owners=真正承接工作的票，joinedWaiters=登记为等待者的票，
// conflicts=同键不同语义被显式拒绝，durableReuse=命中已完成 durable 描述符而直接复用。
struct PrefixWriteOwnerWaiterStats {
    uint64_t owners = 0;
    uint64_t joinedWaiters = 0;
    uint64_t conflicts = 0;
    uint64_t durableReuse = 0;
};
PrefixWriteOwnerWaiterStats PrefixWriteOwnerWaiterStatsSnapshot();

// R2/R3/R7的诊断读取点。沿用 A2/S1-4 的模式：**不新增虚函数、
// 不改任何既有结构与布局**，只在实现内部累加、在这里读取；不读就不影响任何行为。
//
// R2（D2 暂存准入由“立即拒绝”改为“有界推迟”）：
//   submit_defers            = 因预算不足或延迟水位进入有界推迟的**轮次**（每次一个等待切片）；
//   submit_defer_timeouts    = 推迟轮次耗尽上限、Submit 返回 false 的次数（可直接对照真机
//                              `ssd_d2_submit` 丢弃数）；
//   defer_wait_ms_total      = 推迟等待累计毫秒（单调不减）；
//   max_submit_defers        = 本进程生效过的最大推迟上限（诊断用；多 lane 时取最大值）。
// R7（延迟水位）：
//   latency_high_entries     = 进入“高水位”的次数（单调不减）；
//   latency_ema_us_per_mib   = 最近一次更新的延迟 EMA，单位微秒/MiB（**非单调**，最后写入值）；
//   watermark_defers         = 仅由高水位触发（预算其实够）的推迟轮次。
// R3（准入路径淘汰旧检查点）的计数在存储层 Profile 的 `evicted_for_admission` 字段里
// （见 utils/disk_prefix_cache.h 的 DiskPrefixV4Store::Profile），不在这里重复。
struct PrefixWriteDeferStats {
    uint64_t submit_defers = 0;
    uint64_t submit_defer_timeouts = 0;
    uint64_t defer_wait_ms_total = 0;
    uint64_t max_submit_defers = 0;
    uint64_t latency_high_entries = 0;
    uint64_t latency_ema_us_per_mib = 0;
    uint64_t watermark_defers = 0;
};
PrefixWriteDeferStats PrefixWriteDeferStatsSnapshot();

} // namespace prefix_write
} // namespace fastllm
#endif
