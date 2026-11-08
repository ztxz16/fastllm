#include "utils/disk_prefix_write_coordinator.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>

namespace fastllm {
namespace prefix_write {
namespace {

std::string TicketKey(uint64_t requestId, uint64_t sequence) {
    return std::to_string(requestId) + ":" + std::to_string(sequence);
}

// 全局主机预算账本：所有 retained snapshot / scratch / 提交队列都记在同一本账上，
// 并预留"完成工作区"，避免账本被快照占满后前序任务无法完成（方案 §7.3）。
// §8.6：完成缓存必须有界，不能无限保留旧描述。
constexpr size_t kDurableCacheEntries = 4096;

// S1-4「票/额度租约看门狗」参数（业界依据：LMCache 的 pin_timeout_sec=300 +
// 后台 PinMonitor 周期强制解锁；Mooncake 的 lease TTL）。
// 为什么需要：协调器的额度与 kPinnedSlots 准入票此前只能在显式 Complete() 时归还。
// 真机事故（见 qwen3_5_persistent.inc 的 R3/R6 注释）：4 张票因一次守卫遗漏被泄漏，
// kPinnedSlots=4 被占满后 TryAdmit 恒返 CapacityUnavailable ⇒ 整个会话再也写不进
// 任何检查点，只能重启服务。这里给每张已发票加租约，超时由看门狗强制回收。
// 默认值理由：TTL=300000 ms（=300 s，对齐 LMCache 的 pin_timeout_sec=300，远大于
// 正常写时长，保守），扫描周期=30000 ms（=30 s，最多让泄漏票多占 30 s，又能及时
// 救回会话）。可用 FASTLLM_PREFIX_CACHE_LEASE_TIMEOUT_MS /
// FASTLLM_PREFIX_CACHE_LEASE_CHECK_INTERVAL_MS 覆盖（CPU 用例用来把 TTL 压到毫秒级）。
constexpr uint64_t kLeaseTimeoutDefaultMs = 300000;
constexpr uint64_t kLeaseCheckIntervalDefaultMs = 30000;

// 读正整数环境变量；缺失/不可解析/为 0 一律回退保守默认值——0 会让第一轮扫描
// 立刻回收所有在途票，绝不允许成为（哪怕是误配导致的）实际行为。
uint64_t EnvLeaseMs(const char *name, uint64_t fallback) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || parsed == 0) return fallback;
    return uint64_t(parsed);
}

// R2/R7：D2 暂存推迟与延迟水位的配置项。
// 与 EnvLeaseMs 不同，这里**显式允许 0**：0 是有意义的取值——
//   · max defers = 0 ⇒ 回到旧行为（预算不足立即拒绝，便于一键回退/对比）；
//   · cooldown = 0   ⇒ 允许 EMA 一回落立刻退出高水位（CPU 用例需要，避免真等 1 s）。
// 缺失/非法输入一律回退保守默认值。
uint64_t EnvU64AllowZero(const char *name, uint64_t fallback) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text) return fallback;
    return uint64_t(parsed);
}

double EnvDoubleAllowZero(const char *name, double fallback) {
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0') return fallback;
    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (errno != 0 || end == text) return fallback;
    return parsed;
}

// R2：D2 暂存准入由“立即拒绝”改为“有界推迟（bounded defer）”。
// 业界依据（只读调研）：SGLang HiCache 暂存不足时 break 让位、下一轮重试，上限
// `max_staged_admission_defers = 32`；只有 oversize（永远装不下）或 backlog 到上限才丢。
// 默认上限取 32；**每次推迟的等待切片默认 1 ms**，故单次 Submit 最多推迟 32 ms。
// 为什么要这么小的切片：生产侧每个 D2Stream 由**唯一 lane 线程**驱动，提交阶段不 PumpCopy，
// 暂存预算只能由同一线程稍后 PumpCopy 释放；切片小 + 轮次有界保证“等不到就让位给调用方推进”，
// 与单线程 PumpCopy 驱动严格兼容，绝不互锁。
constexpr uint64_t kMaxSubmitDefersDefault = 32;
constexpr uint64_t kDeferWaitMsDefault = 1;
// R7（W4）：D2 暂存/staging 的**延迟比率水位**（秒/MiB），不替换 HostLedger 字节账本，
// 只是再加一层按延迟的软化背压。常量取自 vLLM v1（KV offloading 的本地水位）：
//   LOCAL_HIGH_WATER_S = 0.010、LOW = 0.002、EMABackpressureDetector::DEFAULT_ALPHA = 0.15、
//   冷却 1.0 s。全部可用 env 覆盖（见 D2StreamImpl 构造函数）。
constexpr double kLatencyHighWaterSPerMiBDefault = 0.010;
constexpr double kLatencyLowWaterSPerMiBDefault = 0.002;
constexpr double kLatencyAlphaDefault = 0.15;
constexpr uint64_t kLatencyCooldownMsDefault = 1000;

// S1-4：看门狗累计回收票数（单调不减）。只用于诊断/快照读取，不参与任何决策。
std::atomic<uint64_t> gLeaseExpiredTotal{0};

// R2/R7（W4）：D2 推迟/水位诊断计数（单调不减，latency EMA 除外——它是最后写入值）。
// 与 A2 同一约定：只在实现内部累加、按 lane 求和或取 max，决不参与准入/落盘决策。
std::atomic<uint64_t> gD2SubmitDefers{0}, gD2SubmitDeferTimeouts{0}, gD2WatermarkDefers{0},
    gD2DeferWaitNs{0}, gD2LatencyHighEntries{0}, gD2LatencyEmaUsPerMib{0}, gD2MaxSubmitDefers{0};

// A2（2026-10-02）：写路径诊断计数（单调不减）。只在实现内部累加，按 lane/实例求和或取 max；
// 决不参与任何准入/落盘决策。读取点见头文件的三个 Snapshot 函数。
std::atomic<uint64_t> gPinnedAcquires{0}, gPinnedWaits{0}, gPinnedPeak{0}, gPinnedReuses{0},
    gPinnedReleases{0}, gPinnedInUseTotal{0};
std::atomic<uint64_t> gCreditAcquires{0}, gCreditReleases{0}, gCreditDenied{0}, gCreditPeak{0};
std::atomic<uint64_t> gOwnerCount{0}, gWaiterJoined{0}, gConflictCount{0}, gDurableReuse{0};

class HostLedger final : public BudgetLedger {
public:
    explicit HostLedger(uint64_t limit) : limit_(limit) {}
    bool Acquire(uint64_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return AcquireLocked(bytes);
    }
    void Release(uint64_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        inUse_ = bytes > inUse_ ? 0 : inUse_ - bytes;
        gCreditReleases.fetch_add(1, std::memory_order_relaxed);
        // peakBytes 用"账本历史最高"而不是当前值；多账本时取全局最大。
        const uint64_t peak = peak_;
        uint64_t observed = gCreditPeak.load(std::memory_order_relaxed);
        while (peak > observed &&
               !gCreditPeak.compare_exchange_weak(observed, peak, std::memory_order_relaxed)) {}
        changed_.notify_all();
    }
    uint64_t InUse() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return inUse_;
    }
    uint64_t Peak() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }
    uint64_t ReservedCompletionBytes() const override { return kMetadataWindowBytes; }
    bool AcquireLocked(uint64_t bytes) {
        if (bytes > limit_ || inUse_ > limit_ - bytes) {
            gCreditDenied.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        inUse_ += bytes;
        if (inUse_ > peak_) peak_ = inUse_;
        gCreditAcquires.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    bool WaitEmpty(uint64_t timeoutMs) {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                                 [this] { return inUse_ == 0; });
    }
private:
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t limit_, inUse_ = 0, peak_ = 0;
};

} // namespace

class CoordinatorImpl final : public Coordinator {
public:
    explicit CoordinatorImpl(uint64_t hostLimitBytes, std::unique_ptr<DedupRegistry> dedup)
        : ledger_(hostLimitBytes),
          dedup_(dedup ? std::move(dedup)
                       : CreateDedupRegistry(1, kDurableCacheEntries)),
          leaseTimeoutMs_(EnvLeaseMs("FASTLLM_PREFIX_CACHE_LEASE_TIMEOUT_MS",
                                     kLeaseTimeoutDefaultMs)),
          leaseCheckIntervalMs_(EnvLeaseMs("FASTLLM_PREFIX_CACHE_LEASE_CHECK_INTERVAL_MS",
                                           kLeaseCheckIntervalDefaultMs)) {
        for (int i = 0; i < kHashWorkers; ++i) workers_.emplace_back([this] { HashLoop(); });
        // S1-4：租约看门狗线程。生命周期与哈希 worker 一样由 Stop() 统一接管。
        leaseWorker_ = std::thread([this] { LeaseLoop(); });
    }
    ~CoordinatorImpl() override { Stop(); }

    Admission TryAdmit(const TicketId &ticket, const std::string &key, uint64_t estimatedBytes,
                       TicketId &out, const std::string &descriptor) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) return Admission::Stopping;
        if (descriptor.empty()) {
            // 兼容路径：调用方没有 §8 的语义描述，只做"在途同键"去重。
            // 这里逐字保持接线前的行为，不产生 durable 复用、不判语义冲突。
            if (inFlight_.count(key)) return Admission::Joined;
            return ReserveLocked(ticket, key, estimatedBytes, out) ? Admission::Accepted
                                                                  : Admission::CapacityUnavailable;
        }
        // §8 完整协议：钥匙由调用方按 identity+prefix+provenance+layout 派生。
        const uint64_t dedupTicket = nextDedupTicket_++;
        uint64_t ownerTicket = 0;
        const auto registration = dedup_->Register(key, descriptor, dedupTicket, ownerTicket);
        if (registration == DedupRegistry::Registration::Conflict) return Admission::Conflict;
        if (registration == DedupRegistry::Registration::Durable) return Admission::AlreadyDurable;
        if (registration == DedupRegistry::Registration::Joined) {
            // §8.2：等待只发生在后台完成图上；这里只登记 waiter，并把 owner 票据
            // 交给调用方，让它能观察 owner 的终态。
            const auto owner = dedupTickets_.find(ownerTicket);
            out = owner == dedupTickets_.end() ? ticket : owner->second;
            dedupTickets_[dedupTicket] = ticket;
            dedupTicketKeys_[dedupTicket] = key;
            dedupTicketOf_[TicketKey(ticket.requestId, ticket.sequence)] = dedupTicket;
            // waiter 已进入完成图（等 owner 的终态），既不是失败也不是立刻完成。
            states_[TicketKey(ticket.requestId, ticket.sequence)] = TicketState::ADMITTED;
            return Admission::Joined;
        }
        if (!ReserveLocked(ticket, key, estimatedBytes, out)) {
            // 预算/槽位不足：必须撤销刚登记的在途条目，否则同 key 的后续请求
            // 会永远拿到 Joined 却等不到任何 owner。
            std::vector<uint64_t> none;
            dedup_->Complete(key, false, dedupTicket, none);
            return Admission::CapacityUnavailable;
        }
        dedupTickets_[dedupTicket] = ticket;
        dedupTicketKeys_[dedupTicket] = key;
        dedupTicketOf_[TicketKey(ticket.requestId, ticket.sequence)] = dedupTicket;
        return Admission::Accepted;
    }

    // 只入队，不做 I/O；哈希在固定线程池上完成。
    Terminal Submit(const TicketId &ticket, std::vector<HashedChunk> chunks) override {
        uint64_t reserved = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto &expected = expected_[ticket.requestId];
            // 同请求必须严格按 sequence 消费：后面的检查点不得越过缺失的前序。
            if (ticket.sequence != expected) return Terminal::Failed;
            const std::string key = TicketKey(ticket.requestId, ticket.sequence);
            // S1-4：只有"仍在租约里"（额度/slot 还没被归还）的票才允许入队。若这张票
            // 已被看门狗回收（已 FAILED），迟到的 Submit 会让 HashLoop 二次归还同一份
            // 额度/slot，把别的在途票的账记少。正常准入路径一定有租约，故不受影响。
            if (leases_.count(key) == 0) return Terminal::Failed;
            ++expected;
            reserved = reserved_[key];
            reserved_.erase(key);
            states_[key] = TicketState::HASHING;
            // S1-4：状态推进到 HASHING ⇒ 刷新"最后一次推进时间"；额度已移交给
            // jobs_（由 HashLoop 归还），看门狗只会回收"连队列里也找不到"的僵尸票。
            const auto lease = leases_.find(key);
            if (lease != leases_.end()) {
                lease->second.submitted = true;
                lease->second.lastProgress = std::chrono::steady_clock::now();
            }
        }
        jobs_.push_back({ticket, std::move(chunks), reserved});
        ready_.notify_one();
        return Terminal::Committed;
    }

    bool Drain(uint64_t timeoutMs, std::string &detail) override {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (std::chrono::steady_clock::now() < deadline) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (jobs_.empty() && activeJobs_ == 0 && !ledger_.InUse()) {
                    detail = "clean";
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        detail = ledger_.InUse() ? "credits_outstanding" : "jobs_outstanding";
        return false;
    }

    TicketState State(const TicketId &ticket) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = states_.find(TicketKey(ticket.requestId, ticket.sequence));
        return found == states_.end() ? TicketState::FAILED : found->second;
    }

    void Complete(const TicketId &ticket, Terminal state) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string id = TicketKey(ticket.requestId, ticket.sequence);
        // S1-4：只有"确实还占着额度/slot"的票才减槽位。看门狗可能已经回收过这张票
        // （Submit 路径下 HashLoop 也可能已经归还），迟到的 Complete 若仍无条件递减，
        // 就会替别的在途票把槽位记少。这一改动只影响"票已不在本层"的调用。
        const bool slotHeld = reserved_.count(id) != 0 || leases_.count(id) != 0;
        const auto found = reserved_.find(id);
        if (found != reserved_.end()) {
            // 释放准入时预留的额度：失败路径同样必须释放，否则账本永远清不了零。
            ledger_.Release(found->second);
            reserved_.erase(found);
        }
        if (slotHeld && activeSlots_ > 0) --activeSlots_;
        const auto key = dedupKeys_.find(id);
        if (key != dedupKeys_.end()) { inFlight_.erase(key->second); dedupKeys_.erase(key); }
        // S1-4：显式归还即租约结束，看门狗不再持有这张票（否则 TTL 后会重复回收）。
        leases_.erase(id);
        FinalizeLocked(id, state);
    }

    uint64_t HostCreditsInUse() const override { return ledger_.InUse(); }

    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) return;
            stopping_ = true;
        }
        ready_.notify_all();
        // S1-4：先唤醒看门狗线程并 join，确保析构后不会再有人访问任何成员；
        // join 顺序只影响退出速度，不引入新的锁序（两者都在 mutex_ 之外被 join）。
        leaseCv_.notify_all();
        if (leaseWorker_.joinable()) leaseWorker_.join();
        for (auto &worker : workers_) if (worker.joinable()) worker.join();
        workers_.clear();
    }

private:
    struct Job {
        TicketId ticket;
        std::vector<HashedChunk> chunks;
        uint64_t reserved = 0;
    };

    // S1-4：一张已发票的租约（参数见文件顶部）。admittedAt 只用于诊断；
    // lastProgress 是判超时的唯一依据——只有"确实超过 TTL 且期间没有任何状态
    // 推进（Submit / 进入哈希）"才回收，正常写期间每次推进都会刷新它。
    struct Lease {
        TicketId ticket;
        uint64_t bytes = 0;      // 该票占用的 host 额度
        bool submitted = false;  // 额度已移交给 jobs_（由 HashLoop 归还）
        std::chrono::steady_clock::time_point admittedAt;
        std::chrono::steady_clock::time_point lastProgress;
    };

    // 共用的资源准入（按**字节账本**计）：两条去重路径都走这里，
    // 保证"占满即拒绝、不等待"的语义只有一份实现。
    //
    // S1-3b 修复：
    // 旧实现在这里用"在途作业数 < kPinnedSlots(=4)"作为准入门槛。它把**整笔检查点作业**当成
    // 一个槽位单位，而长上下文一笔作业要占 121 MB、且要长期占用整条写管道 ⇒ 4 笔在途就把第 5 笔
    // 直接判死。真机 r9 铁证：写跳过 15 次、reason=cancelled_then_still_full、打印
    // queued_jobs=3 queue_limit=16 checkpoint_jobs=4 checkpoint_limit=8（全都远未满），
    // 而字节账本只用了 1.06 GiB / 8 GiB ⇒ 卡住的唯一东西就是这道作业级槽数门槛。
    //
    // 业界依据（只读调研，出处见 evidence 与提交信息）：容量一律按**块/chunk 的字节数**计，
    // 从不按"整笔请求/整段前缀"计——vLLM `num_chunks = cpu_bytes_to_use // aligned_kv_bytes_per_chunk`
    // 且满池先 LRU 淘汰、淘汰不动才丢**那一个 chunk**（vllm/v1/kv_offload/cpu/manager.py
    // `prepare_store`/`_get_num_free_chunks`）；SGLang HiCache 用 page 级 host staging 池
    // （`HICACHE_WRITE_STAGING_POOL_FRACTION=0.2`，读优先、写可延）；LMCache chunk=256 +
    // `max_local_cpu_size` + `cache_policy=LRU`。没有一家给"整笔检查点作业"设并发槽数上限。
    // 我们真正的 pinned 资源是 D2 中继的 kPinnedSlots×kSlotBytes pinned 槽池（D2StreamImpl 持有），
    // 它自己按块节流（无空槽就本轮不拷、不丢整笔），无需在这里重复设卡。因此这里只保留字节账本。
    bool ReserveLocked(const TicketId &ticket, const std::string &key, uint64_t estimatedBytes,
                       TicketId &out) {
        // 始终给"完成工作区"留出余量，避免账本满时已接收的任务无法收尾。
        if (!ledger_.AcquireLocked(estimatedBytes)) return false;
        ++activeSlots_;
        inFlight_.insert(key);
        const std::string id = TicketKey(ticket.requestId, ticket.sequence);
        states_[id] = TicketState::ADMITTED;
        reserved_[id] = estimatedBytes;
        dedupKeys_[id] = key;
        expected_[ticket.requestId] = 0;
        // S1-4：发票即建立租约（admit 时间戳 + 最后推进时间）。此后若调用方漏调
        // Complete()，TTL 到期由看门狗强制回收额度与 slot，而不是拖垮整个会话。
        const auto admittedAt = std::chrono::steady_clock::now();
        leases_[id] = Lease{ticket, estimatedBytes, false, admittedAt, admittedAt};
        out = ticket;
        return true;
    }

    // 票据终态落到状态表与去重登记表（§8.4/§8.6）。正常 Complete 与 S1-4 看门狗
    // 回收共用这一份实现，保证"失败绝不伪造 durable、waiters 一律 FAILED"只有一份语义。
    void FinalizeLocked(const std::string &id, Terminal state) {
        states_[id] = state == Terminal::Committed ? TicketState::COMMITTED
                    : state == Terminal::Cancelled ? TicketState::CANCELLED : TicketState::FAILED;
        // §8.4/§8.6：终态必须落到去重登记表上。owner 成功才写 durable；失败时
        // 绝不伪造成功，waiters 一律落 FAILED；登记表条目在终态后删除。
        const auto dedupEntry = dedupTicketOf_.find(id);
        if (dedupEntry == dedupTicketOf_.end()) return;
        const uint64_t dedupTicket = dedupEntry->second;
        dedupTicketOf_.erase(dedupEntry);
        const auto storedKey = dedupTicketKeys_.find(dedupTicket);
        const std::string keyText = storedKey == dedupTicketKeys_.end() ? std::string() : storedKey->second;
        std::vector<uint64_t> waiters;
        // 非 owner 调用时登记表只把本票据从 waiter 列表里摘掉，不会写入 durable。
        dedup_->Complete(keyText, state == Terminal::Committed, dedupTicket, waiters);
        for (uint64_t waiter : waiters) {
            const auto waiterTicket = dedupTickets_.find(waiter);
            if (waiterTicket != dedupTickets_.end()) {
                const std::string waiterId =
                    TicketKey(waiterTicket->second.requestId, waiterTicket->second.sequence);
                states_[waiterId] = state == Terminal::Committed ? TicketState::COMMITTED
                                                                 : TicketState::FAILED;
                dedupTicketOf_.erase(waiterId);
                dedupTickets_.erase(waiterTicket);
            }
            dedupTicketKeys_.erase(waiter);
        }
        dedupTicketKeys_.erase(dedupTicket);
        dedupTickets_.erase(dedupTicket);
    }

    // S1-4 看门狗主循环：周期扫描，只在 Stop() 时退出。锁序沿用既有约定——
    // 全程只持 mutex_，账本/去重表都在 mutex_ 之内被调用，与 Complete/HashLoop 完全一致。
    void LeaseLoop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            leaseCv_.wait_for(lock,
                              std::chrono::milliseconds(static_cast<long long>(leaseCheckIntervalMs_)),
                              [this] { return stopping_; });
            if (stopping_) return;
            ScanLeasesLocked();
        }
    }

    void ScanLeasesLocked() {
        const auto now = std::chrono::steady_clock::now();
        std::vector<std::string> expired;
        for (const auto &entry : leases_) {
            // "确实超过 TTL 且期间无任何推进"才入选；只收集 id，避免边遍历边改容器。
            if (now - entry.second.lastProgress >=
                std::chrono::milliseconds(static_cast<long long>(leaseTimeoutMs_))) {
                expired.push_back(entry.first);
            }
        }
        for (const auto &id : expired) ExpireLeaseLocked(id, now);
    }

    // 回收一张超时票：释放额度与 slot、复用既有 Terminal::Failed 语义落终态、
    // 通知去重登记表（waiters 一并 FAILED），并打一行可诊断日志。
    void ExpireLeaseLocked(const std::string &id, std::chrono::steady_clock::time_point now) {
        const auto leaseEntry = leases_.find(id);
        if (leaseEntry == leases_.end()) return;
        const Lease lease = leaseEntry->second;
        // 额度归属只有两处：未提交的票在 reserved_，已提交的票在 jobs_（HashLoop 归还）。
        // 两处都找不到说明 HashLoop 正持有该任务（取走后、归还前）——它一定会归还，
        // 这里**不重复回收**，留给 HashLoop 收尾，避免双重 Release / 双重减槽。
        uint64_t bytes = 0;
        bool reclaimed = false;
        const auto reserved = reserved_.find(id);
        if (reserved != reserved_.end()) {
            bytes = reserved->second;
            reserved_.erase(reserved);
            reclaimed = true;
        } else {
            for (auto job = jobs_.begin(); job != jobs_.end(); ++job) {
                if (job->ticket == lease.ticket) {
                    bytes = job->reserved;
                    jobs_.erase(job);   // 摘掉任务，HashLoop 不会再为它归还额度
                    reclaimed = true;
                    break;
                }
            }
        }
        if (!reclaimed) return;
        leases_.erase(leaseEntry);
        ledger_.Release(bytes);
        if (activeSlots_ > 0) --activeSlots_;
        const auto key = dedupKeys_.find(id);
        if (key != dedupKeys_.end()) { inFlight_.erase(key->second); dedupKeys_.erase(key); }
        // 复用既有 Terminal 语义（Failed），不新造状态机。
        FinalizeLocked(id, Terminal::Failed);
        const uint64_t heldMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lease.admittedAt).count());
        const uint64_t idleMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            now - lease.lastProgress).count());
        const uint64_t total = ++gLeaseExpiredTotal;
        // 一行可诊断日志：哪张票被回收、总持有多久、期间多久没有推进、归还多少字节、累计回收数。
        std::fprintf(stderr,
                     "[Prefix Write] lease_timeout: ticket=%llu:%llu held_ms=%llu bytes=%llu "
                     "idle_ms=%llu lease_expired_total=%llu.\n",
                     (unsigned long long)lease.ticket.requestId,
                     (unsigned long long)lease.ticket.sequence,
                     (unsigned long long)heldMs, (unsigned long long)bytes,
                     (unsigned long long)idleMs, (unsigned long long)total);
        std::fflush(stderr);
    }

    void HashLoop() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                if (stopping_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop_front();
                ++activeJobs_;
            }
            // 固定池上完成本任务：每块只在本线程处理，不读模型状态、不写 catalog。
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --activeJobs_;
                // 释放的是**准入时为该 ticket 预留的额度**，不是块字节之和：空块
                // 任务同样必须归还额度，否则账本永远清不了零。
                ledger_.Release(job.reserved);
                --activeSlots_;
                const std::string id = TicketKey(job.ticket.requestId, job.ticket.sequence);
                states_[id] = TicketState::COMMITTED;
                // S1-4：额度与 slot 已归还 ⇒ 租约结束，看门狗不再扫描这张票。
                leases_.erase(id);
            }
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    HostLedger ledger_;
    std::vector<std::thread> workers_;
    std::deque<Job> jobs_;
    std::map<std::string, TicketState> states_;
    std::map<uint64_t, uint64_t> expected_;
    std::map<std::string, uint64_t> reserved_;
    std::map<std::string, std::string> dedupKeys_;
    // 在途同键集合（两条路径共用）；`durable_` 已由 DedupRegistry 的有界完成
    // 缓存取代——旧实现里那个集合从不写入，属于死代码。
    std::set<std::string> inFlight_;
    std::unique_ptr<DedupRegistry> dedup_;
    // §8 票据映射：去重登记表用的是 uint64 票据号，这里双向映射回真实 TicketId。
    std::map<uint64_t, TicketId> dedupTickets_;
    std::map<uint64_t, std::string> dedupTicketKeys_;
    std::map<std::string, uint64_t> dedupTicketOf_;
    uint64_t nextDedupTicket_ = 1;
    int activeSlots_ = 0, activeJobs_ = 0;
    bool stopping_ = false;
    // S1-4 租约看门狗：参数在构造时读一次（环境变量覆盖见文件顶部），线程在
    // Stop()/析构时被唤醒并 join，保证不会在成员析构后访问任何成员。
    uint64_t leaseTimeoutMs_ = kLeaseTimeoutDefaultMs;
    uint64_t leaseCheckIntervalMs_ = kLeaseCheckIntervalDefaultMs;
    std::map<std::string, Lease> leases_;   // 只有"仍占着额度/slot"的票在这里
    std::condition_variable leaseCv_;
    std::thread leaseWorker_;
};

std::unique_ptr<Coordinator> CreateCoordinator(uint64_t hostLimitBytes) {
    return std::unique_ptr<Coordinator>(new CoordinatorImpl(hostLimitBytes, nullptr));
}

std::unique_ptr<Coordinator> CreateCoordinator(uint64_t hostLimitBytes,
                                               std::unique_ptr<DedupRegistry> dedup) {
    return std::unique_ptr<Coordinator>(new CoordinatorImpl(hostLimitBytes, std::move(dedup)));
}

// S1-4：看门狗累计回收的票数。新增的诊断读取点，不改动冻结的 Coordinator 接口
// （无新虚函数、无布局变化）；快照侧可自行决定是否打印。
uint64_t PrefixWriteLeaseExpiredTotal() { return gLeaseExpiredTotal.load(); }

// A2：诊断快照读取点（定义见头文件）。只读、无锁（原子 load），不改变任何行为。
PrefixWritePinnedSlotStats PrefixWritePinnedSlotStatsSnapshot() {
    PrefixWritePinnedSlotStats stats;
    stats.acquires = gPinnedAcquires.load(std::memory_order_relaxed);
    stats.waits = gPinnedWaits.load(std::memory_order_relaxed);
    stats.peak = gPinnedPeak.load(std::memory_order_relaxed);
    stats.reuses = gPinnedReuses.load(std::memory_order_relaxed);
    stats.releases = gPinnedReleases.load(std::memory_order_relaxed);
    return stats;
}
PrefixWriteHostCreditStats PrefixWriteHostCreditStatsSnapshot() {
    PrefixWriteHostCreditStats stats;
    stats.acquires = gCreditAcquires.load(std::memory_order_relaxed);
    stats.releases = gCreditReleases.load(std::memory_order_relaxed);
    stats.denied = gCreditDenied.load(std::memory_order_relaxed);
    stats.peakBytes = gCreditPeak.load(std::memory_order_relaxed);
    return stats;
}
PrefixWriteOwnerWaiterStats PrefixWriteOwnerWaiterStatsSnapshot() {
    PrefixWriteOwnerWaiterStats stats;
    stats.owners = gOwnerCount.load(std::memory_order_relaxed);
    stats.joinedWaiters = gWaiterJoined.load(std::memory_order_relaxed);
    stats.conflicts = gConflictCount.load(std::memory_order_relaxed);
    stats.durableReuse = gDurableReuse.load(std::memory_order_relaxed);
    return stats;
}
// R2/R7（W4）：D2 有界推迟与延迟水位的诊断快照。只读、无锁（原子 load），不改变任何行为。
PrefixWriteDeferStats PrefixWriteDeferStatsSnapshot() {
    PrefixWriteDeferStats stats;
    stats.submit_defers = gD2SubmitDefers.load(std::memory_order_relaxed);
    stats.submit_defer_timeouts = gD2SubmitDeferTimeouts.load(std::memory_order_relaxed);
    stats.defer_wait_ms_total = gD2DeferWaitNs.load(std::memory_order_relaxed) / 1000000ull;
    stats.max_submit_defers = gD2MaxSubmitDefers.load(std::memory_order_relaxed);
    stats.latency_high_entries = gD2LatencyHighEntries.load(std::memory_order_relaxed);
    stats.latency_ema_us_per_mib = gD2LatencyEmaUsPerMib.load(std::memory_order_relaxed);
    stats.watermark_defers = gD2WatermarkDefers.load(std::memory_order_relaxed);
    return stats;
}

} // namespace prefix_write
} // namespace fastllm

// ===========================================================================
// §6.1 机制层：K 个 pinned 拷贝槽 + 固定哈希池（按提交顺序交付）
// ===========================================================================
#include <openssl/evp.h>
namespace fastllm {
namespace prefix_write {
namespace {

std::string Sha256Hex(const std::string &payload) {
    unsigned char raw[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    if (EVP_Digest(payload.data(), payload.size(), raw, &count, EVP_sha256(), nullptr) != 1 ||
        count != 32) {
        return std::string();
    }
    static const char digits[] = "0123456789abcdef";
    std::string text;
    text.reserve(64);
    for (unsigned i = 0; i < count; ++i) { text += digits[raw[i] >> 4]; text += digits[raw[i] & 15]; }
    return text;
}

class PinnedSlotPoolImpl final : public PinnedSlotPool {
public:
    PinnedSlotPoolImpl(int slots, uint64_t slotBytes)
        : slotBytes_(slotBytes), free_(size_t(slots), true), usedOnce_(size_t(slots), false) {}
    int Acquire() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < free_.size(); ++i) {
            if (free_[i]) {
                free_[i] = false; ++inUse_;
                // A2：复用计数 = 该槽号此前被用过（pinned 内存被复用而不是新分配）。
                if (usedOnce_[i]) gPinnedReuses.fetch_add(1, std::memory_order_relaxed);
                else usedOnce_[i] = true;
                gPinnedAcquires.fetch_add(1, std::memory_order_relaxed);
                // peak = **所有槽池合计**同时占用的槽位数（per-lane 多池时才有意义；单池时 <= K）。
                const uint64_t total = gPinnedInUseTotal.fetch_add(1, std::memory_order_relaxed) + 1;
                uint64_t observed = gPinnedPeak.load(std::memory_order_relaxed);
                while (total > observed &&
                       !gPinnedPeak.compare_exchange_weak(observed, total,
                                                          std::memory_order_relaxed)) {}
                return int(i);
            }
        }
        gPinnedWaits.fetch_add(1, std::memory_order_relaxed);   // A2：无空槽（明确失败）
        return -1;   // 无空槽：明确失败，不等待
    }
    void Release(int slot) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot < 0 || size_t(slot) >= free_.size() || free_[slot]) return;   // 防重复释放
        free_[slot] = true; --inUse_;
        gPinnedReleases.fetch_add(1, std::memory_order_relaxed);
        gPinnedInUseTotal.fetch_sub(1, std::memory_order_relaxed);
    }
    int InUse() const override { std::lock_guard<std::mutex> lock(mutex_); return inUse_; }
    int Capacity() const override { return int(free_.size()); }
    uint64_t SlotBytes() const { return slotBytes_; }
private:
    mutable std::mutex mutex_;
    uint64_t slotBytes_ = 0;
    std::vector<bool> free_;
    std::vector<bool> usedOnce_;   // A2：该槽号是否被用过（用于 reuse 计数）
    int inUse_ = 0;
};

struct HashEntry {
    bool ready = false;
    TicketId ticket;
    std::vector<std::string> digests;    // §23：一个任务可含多个分块
    std::vector<std::string> payloads;   // 摘要与字节成对交付
    uint64_t bytes = 0;
};

class HashPackPoolImpl final : public HashPackPool {
public:
    HashPackPoolImpl(uint64_t stagingBudget, int workers) : budget_(stagingBudget) {
        const int count = workers > 0 ? workers : 1;
        for (int i = 0; i < count; ++i) workers_.emplace_back([this] { Loop(); });
    }
    ~HashPackPoolImpl() override { Stop(); }

    bool Submit(const TicketId &ticket, std::string payload) override {
        std::vector<std::string> one;
        one.push_back(std::move(payload));
        return SubmitBatch(ticket, std::move(one));
    }

    // §23：一次任务算多块 —— 线程只被唤醒一次，块数再多也不增加交接次数。
    bool SubmitBatch(const TicketId &ticket, std::vector<std::string> payloads) override {
        uint64_t bytes = 0;
        for (const auto &payload : payloads) bytes += payload.size();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || payloads.empty()) return false;
            if (bytes > budget_ || inFlightBytes_ > budget_ - bytes) return false;
            inFlightBytes_ += bytes;
            if (inFlightBytes_ > peakInFlight_) peakInFlight_ = inFlightBytes_;
            const uint64_t index = next_++;
            entries_[index] = HashEntry{false, ticket, {}, {}, bytes};
            tasks_.push_back({index, std::move(payloads)});
        }
        ready_.notify_one();
        return true;
    }

    bool TakeOrdered(TicketId &ticket, std::string &digest, uint64_t timeoutMs) override {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;) {
            auto head = entries_.find(deliver_);
            if (head != entries_.end() && head->second.ready) {
                ticket = head->second.ticket;
                digest = head->second.digests.empty() ? std::string() : head->second.digests.front();
                inFlightBytes_ -= head->second.bytes;
                entries_.erase(head);
                ++deliver_;
                return true;
            }
            if (tasks_.empty() && entries_.empty()) return false;   // 没有更多在途任务
            if (done_.wait_until(lock, deadline) == std::cv_status::timeout) return false;
        }
    }

    bool TakeOrderedChunk(TicketId &ticket, HashedChunk &chunk, uint64_t timeoutMs) override {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;) {
            auto head = entries_.find(deliver_);
            if (head != entries_.end() && head->second.ready) {
                ticket = head->second.ticket;
                std::string digest = head->second.digests.empty() ? std::string()
                                                                  : std::move(head->second.digests.front());
                std::string payload = head->second.payloads.empty() ? std::string()
                                                                    : std::move(head->second.payloads.front());
                inFlightBytes_ -= head->second.bytes;
                entries_.erase(head);
                ++deliver_;
                chunk = MakeChunk(std::move(digest), std::move(payload));
                return chunk.Valid();
            }
            if (tasks_.empty() && entries_.empty()) return false;
            if (done_.wait_until(lock, deadline) == std::cv_status::timeout) return false;
        }
    }

    // §23：整批取回（按提交顺序）。
    bool TakeOrderedBatch(TicketId &ticket, std::vector<HashedChunk> &chunks,
                          uint64_t timeoutMs) override {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;) {
            auto head = entries_.find(deliver_);
            if (head != entries_.end() && head->second.ready) {
                ticket = head->second.ticket;
                chunks.clear();
                const size_t count = std::min(head->second.digests.size(), head->second.payloads.size());
                for (size_t i = 0; i < count; ++i) {
                    chunks.push_back(MakeChunk(std::move(head->second.digests[i]),
                                               std::move(head->second.payloads[i])));
                }
                inFlightBytes_ -= head->second.bytes;
                entries_.erase(head);
                ++deliver_;
                return !chunks.empty();
            }
            if (tasks_.empty() && entries_.empty()) return false;
            if (done_.wait_until(lock, deadline) == std::cv_status::timeout) return false;
        }
    }

    uint64_t InFlight() const override { std::lock_guard<std::mutex> lock(mutex_); return uint64_t(entries_.size()); }
    uint64_t InFlightBytes() const override { std::lock_guard<std::mutex> lock(mutex_); return inFlightBytes_; }
    uint64_t InFlightPeak() const override { std::lock_guard<std::mutex> lock(mutex_); return peakInFlight_; }
    void Stop() override {
        { std::lock_guard<std::mutex> lock(mutex_); if (stopping_) return; stopping_ = true; }
        ready_.notify_all();
        for (auto &worker : workers_) if (worker.joinable()) worker.join();
        workers_.clear();
    }

private:
    struct Task { uint64_t index; std::vector<std::string> payloads; };
    void Loop() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (tasks_.empty()) { if (stopping_) return; continue; }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            // 每个任务独立的哈希上下文：worker 之间不共享任何 EVP 状态。
            // §23：一个任务可含多块，这里一次唤醒就把整批算完。
            std::vector<std::string> digests;
            digests.reserve(task.payloads.size());
            for (const auto &payload : task.payloads) digests.push_back(Sha256Hex(payload));
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto found = entries_.find(task.index);
                if (found != entries_.end()) {
                    found->second.digests = std::move(digests);
                    found->second.payloads = std::move(task.payloads);
                    found->second.ready = true;
                }
            }
            done_.notify_all();
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_, done_;
    std::vector<std::thread> workers_;
    std::deque<Task> tasks_;
    std::map<uint64_t, HashEntry> entries_;
    uint64_t next_ = 0, deliver_ = 0, inFlightBytes_ = 0, peakInFlight_ = 0, budget_ = 0;
    bool stopping_ = false;
};

} // namespace

std::unique_ptr<PinnedSlotPool> CreatePinnedSlotPool(int slots, uint64_t slotBytes) {
    if (slots <= 0) return nullptr;
    return std::unique_ptr<PinnedSlotPool>(new PinnedSlotPoolImpl(slots, slotBytes));
}
std::unique_ptr<HashPackPool> CreateHashPackPool(uint64_t stagingBudgetBytes, int workers) {
    return std::unique_ptr<HashPackPool>(new HashPackPoolImpl(stagingBudgetBytes, workers));
}

} // namespace prefix_write
} // namespace fastllm

// ===========================================================================
// §8 去重协议机制层
// ===========================================================================
namespace fastllm {
namespace prefix_write {
namespace {

std::string SemanticDigest(const std::string &text) {
    unsigned char raw[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    if (EVP_Digest(text.data(), text.size(), raw, &count, EVP_sha256(), nullptr) != 1 ||
        count != 32) {
        return std::string();
    }
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (unsigned i = 0; i < count; ++i) { out += digits[raw[i] >> 4]; out += digits[raw[i] & 15]; }
    return out;
}

// §6.1 D2-stream 实现。CUDA 事件/D2H 由回调注入，所以这是纯 CPU 可测的调度。
class D2StreamImpl final : public D2Stream {
public:
    D2StreamImpl(EventReady ready, CopyIntoSlot copy, uint64_t stagingBudgetBytes,
                 HashPackPool *sharedPool)
        : ready_(std::move(ready)), copy_(std::move(copy)),
          stagingBudget_(stagingBudgetBytes ? stagingBudgetBytes : 1),
          slots_(CreatePinnedSlotPool(PinnedSlots(), kSlotBytes)),
          shared_(sharedPool),
          pool_(sharedPool ? nullptr : CreateHashPackPool(stagingBudgetBytes ? stagingBudgetBytes : 1, kHashWorkers)) {
        // R2/R7：推迟与水位参数在构造时读一次（env 覆盖）。
        maxSubmitDefers_ = EnvU64AllowZero("FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS",
                                           kMaxSubmitDefersDefault);
        deferWaitMs_ = EnvU64AllowZero("FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS",
                                       kDeferWaitMsDefault);
        latencyHighSPerMiB_ = EnvDoubleAllowZero("FASTLLM_PREFIX_CACHE_D2_LAT_HIGH_S",
                                                 kLatencyHighWaterSPerMiBDefault);
        latencyLowSPerMiB_ = EnvDoubleAllowZero("FASTLLM_PREFIX_CACHE_D2_LAT_LOW_S",
                                                kLatencyLowWaterSPerMiBDefault);
        latencyAlpha_ = EnvDoubleAllowZero("FASTLLM_PREFIX_CACHE_D2_LAT_ALPHA",
                                           kLatencyAlphaDefault);
        latencyCooldownMs_ = EnvU64AllowZero("FASTLLM_PREFIX_CACHE_D2_LAT_COOLDOWN_MS",
                                             kLatencyCooldownMsDefault);
        // 诊断：把生效上限发布到全局快照（多 lane 时取历史最大值，不被更小的值覆盖）。
        uint64_t observed = gD2MaxSubmitDefers.load(std::memory_order_relaxed);
        while (maxSubmitDefers_ > observed &&
               !gD2MaxSubmitDefers.compare_exchange_weak(observed, maxSubmitDefers_,
                                                         std::memory_order_relaxed)) {}
    }

    ~D2StreamImpl() override { Stop(); }

    // R2：预算不足不再立即返回 false，而是**有界推迟**。
    // 旧契约（见 test_disk_prefix_coordinator_contract.cpp 里那条“超预算必须拒绝”的断言）在
    // “永远装不下的 oversize 条目”上仍然成立——推迟上限用尽后依旧返回 false，且 pending 不变；
    // 变化只在于“装得下、只是此刻没空间”的条目：现在会先等一个有界窗口，等不到才失败。
    bool Submit(const TicketId &ticket, uint64_t eventId, size_t bytes) override {
        uint64_t deferred = 0;
        bool watermarkDeferred = false;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stopped_) return false;
            const bool budgetOk = pendingBytes_ + bytes <= stagingBudget_;
            if (budgetOk) {
                // R7：延迟高水位时只额外推迟**一次**切片，给流水线一个压延迟的机会；
                // 等完只要预算仍然够就一定接收——绝不因水位丢掉物理上装得下的新写。
                if (waterHigh_ && !watermarkDeferred) {
                    watermarkDeferred = true;
                    ++watermarkDefers_;
                    gD2WatermarkDefers.fetch_add(1, std::memory_order_relaxed);
                    DeferWaitLocked(lock);
                    continue;
                }
                pending_.push_back(Entry{ticket, eventId, bytes, -1});
                pendingBytes_ += bytes;
                return true;
            }
            // 预算不足：推迟轮次用尽才失败（有界，绝不无限等）。
            if (deferred >= maxSubmitDefers_) {
                ++deferTimeouts_;
                gD2SubmitDeferTimeouts.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            ++deferred;
            ++defers_;
            gD2SubmitDefers.fetch_add(1, std::memory_order_relaxed);
            DeferWaitLocked(lock);
        }
    }

    void PumpCopy() override {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopped_ && !pending_.empty()) {
            Entry &head = pending_.front();
            if (!ready_ || !ready_(head.eventId)) return;      // 事件未完成：绝不越过
            if (slots_->InUse() >= slots_->Capacity()) {
                gPinnedWaits.fetch_add(1, std::memory_order_relaxed);   // A2：无空槽，本轮不拷贝
                return; // 无空槽：不等待
            }
            const int slot = slots_->Acquire();
            if (slot < 0) return;
            if (slot + 1 > slotPeak_) slotPeak_ = slot + 1;
            const TicketId ticket = head.ticket;
            const size_t bytes = head.bytes;
            // INV-1 修复（2026-10-02，只读调研确认为潜在缺陷）：**先把槽位登记进"待取回"队列，
            // 再让哈希池看见这次提交**。旧顺序是先 Submit 后 push：池条目一旦可见就可能被取回，
            // 而取回路径 TakeOrderedChunk 在 `hashSlots_` 为空时直接 `return false` ——
            // 池侧已销账、分块被丢弃、该槽还会在稍后被 push 进队列并永久漏掉
            // （漏满 kPinnedSlots=4 后 PumpCopy 恒无空槽 ⇒ D2 流永久停摆 ⇒ ssd_d2_submit 丢点）。
            // 当前由唯一 worker 线程串行驱动，该窗口不可达；登记前移后窗口彻底消失。
            hashSlots_.push_back(slot);
            lock.unlock();
            const auto copyStart = std::chrono::steady_clock::now();
            std::string payload;
            try {
                payload = copy_ ? copy_(slot, ticket, bytes) : std::string();
            } catch (...) {
                // A-5：copy_ 抛异常时该 ticket 必须**确定性终结**。
                // 旧实现在这里直接让异常穿出去，条目仍留在 pending_ 队首（pop 只在下面两条路径），
                // 而调用方的终结守卫已经把预约结算掉 ⇒ 下一次 PumpCopy 会对同一 ticket 再调一次
                // copy_，调用方随即报 duplicate_or_unadmitted_copy（真因被这条二次错误掩盖，
                // 见 persistent.inc:3340-3356 与 logs/user-deploy-s7-20261031T0826Z/server.log:470）。
                // 现在：弹条目、还槽位、记 failed_（供 Failed() 查询）、唤醒等待者，然后原样抛出。
                const auto copyEnd = std::chrono::steady_clock::now();
                lock.lock();
                ObserveCopyLatencyLocked(bytes, copyEnd - copyStart);
                if (!hashSlots_.empty() && hashSlots_.back() == slot) hashSlots_.pop_back();
                slots_->Release(slot);
                if (!pending_.empty()) {
                    pending_.pop_front();
                    pendingBytes_ -= bytes;
                }
                NotifyBudgetReleasedLocked();
                failed_.insert(TicketKey(ticket.requestId, ticket.sequence));
                throw;
            }
            const auto copyEnd = std::chrono::steady_clock::now();
            const bool accepted = (shared_ ? shared_ : pool_.get())->Submit(ticket, std::move(payload));
            lock.lock();
            // R7：用刚测得的 D2H 延迟更新 EMA 水位（必须在持锁状态下改 waterHigh_）。
            ObserveCopyLatencyLocked(bytes, copyEnd - copyStart);
            if (!accepted) {
                // 哈希池拒收（预算/停止）：撤销预登记（该池条目从未可见，故必在队尾）并归还槽位。
                if (!hashSlots_.empty() && hashSlots_.back() == slot) hashSlots_.pop_back();
                slots_->Release(slot);
                pending_.pop_front();
                pendingBytes_ -= bytes;
                NotifyBudgetReleasedLocked();   // R2：唤醒正在有界推迟的 Submit
                failed_.insert(TicketKey(ticket.requestId, ticket.sequence));
                continue;
            }
            ++copies_;
            pending_.pop_front();
            pendingBytes_ -= bytes;
            NotifyBudgetReleasedLocked();   // R2：唤醒正在有界推迟的 Submit
            // 槽位已在解锁前登记，等到摘要被取走才释放（§6.1 第 2 条）。
        }
    }

    bool TakeOrdered(TicketId &ticket, std::string &digest, uint64_t timeoutMs) override {
        if (!(shared_ ? shared_ : pool_.get())->TakeOrdered(ticket, digest, timeoutMs)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (hashSlots_.empty()) return false;
        slots_->Release(hashSlots_.front());
        hashSlots_.pop_front();
        return true;
    }

    // §22.4.4 甲：把"已哈希且字节不可篡改"的结果整体交出去，供 StoreVerified 使用，
    // 从而不需要在提交路径里重算摘要。
    bool TakeOrderedChunk(TicketId &ticket, HashedChunk &chunk, uint64_t timeoutMs) override {
        if (!(shared_ ? shared_ : pool_.get())->TakeOrderedChunk(ticket, chunk, timeoutMs)) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        if (hashSlots_.empty()) return false;
        slots_->Release(hashSlots_.front());
        hashSlots_.pop_front();
        return true;
    }

    int SlotsInUse() const override { return slots_->InUse(); }
    // §④ 失败出口（见头文件）：只在 failed_ 里查，不等待、不改状态。
    bool Failed(const TicketId &ticket) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_.count(TicketKey(ticket.requestId, ticket.sequence)) != 0;
    }
    int SlotCapacity() const override { return slots_->Capacity(); }
    int SlotPeak() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return slotPeak_;
    }
    uint64_t Pending() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }
    void Stop() override {
        // R2/R7：本 lane 的推迟/水位汇总值在锁内快照，供下面一行诊断使用。
        uint64_t snapDefers = 0, snapTimeouts = 0, snapWaitNs = 0, snapWatermark = 0,
                 snapHigh = 0;
        double snapEma = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return;
            stopped_ = true;
            snapDefers = defers_;
            snapTimeouts = deferTimeouts_;
            snapWaitNs = deferWaitNs_;
            snapWatermark = watermarkDefers_;
            snapHigh = latencyHigh_;
            snapEma = latencyEmaSPerMiB_;
        }
        // R2/R7：把“本 lane 到底推迟了多少次、有没有因为推迟失败、延迟水位进过几次”打一行出来。
        // 既有模型侧 `ssd-watermark` 打印在 src/models/qwen3_5_persistent.inc 里（本批次禁改），
        // 所以新增字段先由协调器自己打印；读取点 PrefixWriteDeferStatsSnapshot() 已就绪，
        // 后续把那行输出追加字段即可（不改名、不删字段）。
        std::fprintf(stderr,
                     "[Prefix Write] d2_defer: defers=%llu timeouts=%llu wait_ms=%llu "
                     "watermark_defers=%llu latency_high_entries=%llu latency_ema_us_per_mib=%llu.\n",
                     (unsigned long long)snapDefers, (unsigned long long)snapTimeouts,
                     (unsigned long long)(snapWaitNs / 1000000ull),
                     (unsigned long long)snapWatermark, (unsigned long long)snapHigh,
                     (unsigned long long)uint64_t(snapEma * 1e6));
        std::fflush(stderr);
        // R2：唤醒正在有界推迟的 Submit。没有这一句它也要等满一个切片才看到 stopped_（仍有界），
        // 但 Stop() 的语义是“立即停止接收”，必须立刻醒。
        budgetCv_.notify_all();
        // 共享哈希池由它的所有者（生产侧/Coordinator）负责停止；这里若不加空判，
        // sharedPool != nullptr 时就是空指针解引用（构造处把 pool_ 置为 nullptr）。
        if (pool_) pool_->Stop();
    }

    // A-5：异常/残留清理，见头文件。调用方必须先保证在途 D2H 已经同步
    // （本类 copy_ 的终结守卫在返回前会同步流），否则释放槽位可能被后续拷贝覆盖。
    void Reset() override {
        // ① 把哈希池里"已拷贝完成、但没人取"的结果取走（有界等待：哈希是异步的，可能还在算），
        //    归还它们占的槽位与暂存预算，并把 ticket 记为失败。
        const auto drainDeadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
        while (std::chrono::steady_clock::now() < drainDeadline) {
            TicketId ticket;
            HashedChunk chunk;
            if (!(shared_ ? shared_ : pool_.get())->TakeOrderedChunk(ticket, chunk, 1)) continue;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!hashSlots_.empty()) { slots_->Release(hashSlots_.front()); hashSlots_.pop_front(); }
            failed_.insert(TicketKey(ticket.requestId, ticket.sequence));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // ② 未拷贝的条目：记 failed_ 后丢弃（绝不保留在队首）。
        for (const auto &entry : pending_)
            failed_.insert(TicketKey(entry.ticket.requestId, entry.ticket.sequence));
        pending_.clear();
        pendingBytes_ = 0;
        // ③ 仍未归还的槽位（已拷完但没取）全部归还。
        for (int slot : hashSlots_) slots_->Release(slot);
        hashSlots_.clear();
        NotifyBudgetReleasedLocked();
    }

private:
    struct Entry { TicketId ticket; uint64_t eventId; size_t bytes; int slot; };

    // R2：在持锁状态下等待一个推迟切片。condition_variable::wait_for 会在等待期间**释放
    // mutex_**，因此等待中不会“持锁等锁”（无锁序倒置）；预算释放（budgetEpoch_ 变化）与
    // Stop() 都会提前唤醒。等待时长累计进诊断计数。超时到点即返回，绝不无限等。
    void DeferWaitLocked(std::unique_lock<std::mutex> &lock) {
        const uint64_t epoch = budgetEpoch_;
        const auto start = std::chrono::steady_clock::now();
        budgetCv_.wait_for(lock, std::chrono::milliseconds(static_cast<long long>(deferWaitMs_)),
                           [this, epoch] { return stopped_ || budgetEpoch_ != epoch; });
        const uint64_t elapsedNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        deferWaitNs_ += elapsedNs;
        gD2DeferWaitNs.fetch_add(elapsedNs, std::memory_order_relaxed);
    }

    // R2：pendingBytes_ 下降（预算释放）后推进 epoch 并唤醒推迟中的 Submit。
    void NotifyBudgetReleasedLocked() {
        ++budgetEpoch_;
        budgetCv_.notify_all();
    }

    // R7：用 D2H 拷贝的**延迟比率**（秒/MiB）做 EMA 水位。字节账本
    // （stagingBudget_/HostLedger）保持不变，这只是再加一层“按延迟”的软化背压：
    //   · EMA 触及高水位 ⇒ waterHigh_=true，并至少保持 cooldown（滞回，防抖）；
    //   · 冷却期满且 EMA 回落到低水位以下 ⇒ 退出高水位；
    //   · EMA 从 0 起步、每个样本按 alpha 混合（与 vLLM EMABackpressureDetector 一致）。
    void ObserveCopyLatencyLocked(size_t bytes, std::chrono::steady_clock::duration elapsed) {
        if (bytes == 0) return;
        const double seconds = std::chrono::duration<double>(elapsed).count();
        const double secondsPerMiB = seconds / (double(bytes) / 1048576.0);
        latencyEmaSPerMiB_ = latencyAlpha_ * secondsPerMiB +
                             (1.0 - latencyAlpha_) * latencyEmaSPerMiB_;
        gD2LatencyEmaUsPerMib.store(uint64_t(latencyEmaSPerMiB_ * 1e6),
                                    std::memory_order_relaxed);
        const auto now = std::chrono::steady_clock::now();
        if (!waterHigh_) {
            if (latencyHighSPerMiB_ > 0.0 && latencyEmaSPerMiB_ >= latencyHighSPerMiB_) {
                waterHigh_ = true;
                waterHighUntil_ = now + std::chrono::milliseconds(
                    static_cast<long long>(latencyCooldownMs_));
                ++latencyHigh_;
                gD2LatencyHighEntries.fetch_add(1, std::memory_order_relaxed);
                budgetCv_.notify_all();
            }
            return;
        }
        if (now >= waterHighUntil_ && latencyEmaSPerMiB_ <= latencyLowSPerMiB_) {
            waterHigh_ = false;
            budgetCv_.notify_all();
        }
    }

    mutable std::mutex mutex_;
    EventReady ready_;
    CopyIntoSlot copy_;
    uint64_t stagingBudget_;
    std::unique_ptr<PinnedSlotPool> slots_;
    HashPackPool *shared_ = nullptr;      // 非拥有：两个 namespace 共用一个哈希池
    std::unique_ptr<HashPackPool> pool_;  // 拥有；shared_ 为空时使用
    std::deque<Entry> pending_;
    std::deque<int> hashSlots_;
    std::set<std::string> failed_;
    uint64_t pendingBytes_ = 0, copies_ = 0;
    // R2：预算释放会推进 epoch，作为“当前等待切片是否值得提前返回”的判据。
    uint64_t budgetEpoch_ = 0;
    // R2/R7：构造时读一次（env 覆盖，见文件顶部常量与构造函数）。
    uint64_t maxSubmitDefers_ = kMaxSubmitDefersDefault;
    uint64_t deferWaitMs_ = kDeferWaitMsDefault;
    double latencyHighSPerMiB_ = kLatencyHighWaterSPerMiBDefault;
    double latencyLowSPerMiB_ = kLatencyLowWaterSPerMiBDefault;
    double latencyAlpha_ = kLatencyAlphaDefault;
    uint64_t latencyCooldownMs_ = kLatencyCooldownMsDefault;
    double latencyEmaSPerMiB_ = 0.0;
    bool waterHigh_ = false;
    std::chrono::steady_clock::time_point waterHighUntil_{};
    std::condition_variable budgetCv_;
    // R2/R7：本 lane 的诊断计数（与全局原子同样累加，供 Stop() 打一行；只读不参与决策）。
    uint64_t defers_ = 0, deferTimeouts_ = 0, watermarkDefers_ = 0, deferWaitNs_ = 0,
             latencyHigh_ = 0;
    int slotPeak_ = 0;
    bool stopped_ = false;
};


class DedupRegistryImpl final : public DedupRegistry {
public:
    DedupRegistryImpl(uint64_t generation, size_t maxDurable)
        : generation_(generation), maxDurable_(maxDurable ? maxDurable : 1) {}

    Registration Register(const std::string &key, const std::string &descriptor, uint64_t ticket,
                          uint64_t &ownerTicket) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string semantic = SemanticDigest(descriptor);
        // 已 durable：语义相同才允许复用，不同则显式冲突，绝不覆盖。
        const auto durable = durable_.find(key);
        if (durable != durable_.end()) {
            ownerTicket = 0;
            if (durable->second == semantic) {
                gDurableReuse.fetch_add(1, std::memory_order_relaxed);   // A2：durable 直接复用
                return Registration::Durable;
            }
            gConflictCount.fetch_add(1, std::memory_order_relaxed);      // A2：同键不同语义
            return Registration::Conflict;
        }
        const auto inflight = inflight_.find(key);
        if (inflight != inflight_.end()) {
            if (inflight->second.semantic != semantic) {
                gConflictCount.fetch_add(1, std::memory_order_relaxed);
                return Registration::Conflict;
            }
            // 在途同 key：登记为 waiter（等待只发生在后台 completion graph）。
            inflight->second.waiters.push_back(ticket);
            ++joins_;
            gWaiterJoined.fetch_add(1, std::memory_order_relaxed);       // A2：登记为 waiter
            ownerTicket = inflight->second.owner;
            return Registration::Joined;
        }
        inflight_[key] = InFlight{ticket, semantic, {}};
        ownerTicket = ticket;
        gOwnerCount.fetch_add(1, std::memory_order_relaxed);             // A2：owner
        return Registration::Owner;
    }

    void Complete(const std::string &key, bool committed, uint64_t ticket,
                  std::vector<uint64_t> &waiters) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = inflight_.find(key);
        if (found == inflight_.end()) return;
        // 只有 owner 能宣告终态，避免 waiter 抢先。
        if (found->second.owner != ticket) {
            auto &list = found->second.waiters;
            list.erase(std::remove(list.begin(), list.end(), ticket), list.end());
            return;
        }
        waiters = found->second.waiters;
        if (committed) {
            InsertDurable(key, found->second.semantic);
        }
        // 失败时**不写入 durable**，也不伪造成功；waiters 由调用方置 FAILED。
        inflight_.erase(found);
        if (committed) ++completed_;
    }

    void SetGeneration(uint64_t generation) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (generation_ == generation) return;
        generation_ = generation;
        // 完成缓存有界且绑定 generation：换代即整体失效，绝不无限保留旧描述。
        durable_.clear();
        order_.clear();
    }

    bool LookupDurable(const std::string &key, std::string &descriptor) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = durable_.find(key);
        if (found == durable_.end()) return false;
        descriptor = found->second;
        return true;
    }

    uint64_t JoinCount() const override { std::lock_guard<std::mutex> lock(mutex_); return joins_; }
    uint64_t DurableCount() const override { std::lock_guard<std::mutex> lock(mutex_); return durable_.size(); }

private:
    struct InFlight { uint64_t owner = 0; std::string semantic; std::vector<uint64_t> waiters; };

    void InsertDurable(const std::string &key, const std::string &semantic) {
        durable_[key] = semantic;
        order_.push_back(key);
        while (order_.size() > maxDurable_) {
            durable_.erase(order_.front());
            order_.pop_front();
        }
    }

    mutable std::mutex mutex_;
    std::map<std::string, InFlight> inflight_;
    std::map<std::string, std::string> durable_;
    std::deque<std::string> order_;
    uint64_t generation_ = 0, joins_ = 0, completed_ = 0;
    size_t maxDurable_ = 1;
};

} // namespace

std::string DedupRegistry::Key(const std::string &identity, const std::string &prefix,
                               const std::string &provenance, const std::string &layout) {
    // 四类语义缺一不可：identity(模型身份)、prefix(前缀键)、provenance(编码来源)、
    // layout/encoding(布局语义)。长度/指针等不可靠信息不参与。
    return SemanticDigest("fastllm-dedup-v1\n" + identity + "\n" + prefix + "\n" + provenance +
                          "\n" + layout);
}

std::unique_ptr<DedupRegistry> CreateDedupRegistry(uint64_t generation, size_t maxDurableEntries) {
    return std::unique_ptr<DedupRegistry>(new DedupRegistryImpl(generation, maxDurableEntries));
}

std::unique_ptr<D2Stream> CreateD2Stream(D2Stream::EventReady ready, D2Stream::CopyIntoSlot copy,
                                        uint64_t stagingBudgetBytes, HashPackPool *sharedPool) {
    return std::unique_ptr<D2Stream>(
        new D2StreamImpl(std::move(ready), std::move(copy), stagingBudgetBytes, sharedPool));
}

} // namespace prefix_write
} // namespace fastllm
