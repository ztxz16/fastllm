// Contract tests for the frozen write-coordinator interface (plan §6/§7).
// These prove the interface is implementable and that the failure/ordering
// rules that later code depends on are pinned before any storage core exists.
#include "utils/disk_prefix_write_coordinator.h"

#include <openssl/sha.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace fastllm::prefix_write;

static int failures = 0;
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL line " << __LINE__ << ": " #condition << "\n"; \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

// A minimal implementation, only to prove the abstract interface is usable and
// that the intended ownership split compiles.
class FakeCoordinator : public Coordinator {
public:
    Admission TryAdmit(const TicketId &ticket, const std::string &, uint64_t estimatedBytes,
                       TicketId &out, const std::string & = std::string()) override {
        if (stopping) return Admission::Stopping;
        if (ledger.InUse() + estimatedBytes > kBaseWindowBytes) {
            return Admission::CapacityUnavailable;
        }
        if (!ledger.Acquire(estimatedBytes)) return Admission::CapacityUnavailable;
        out = ticket;
        states[ticket.requestId] = TicketState::ADMITTED;
        return Admission::Accepted;
    }
    Terminal Submit(const TicketId &, std::vector<HashedChunk>) override {
        return Terminal::Committed;
    }
    bool Drain(uint64_t, std::string &detail) override {
        detail = ledger.InUse() == 0 ? "clean" : "credits_outstanding";
        return ledger.InUse() == 0;
    }
    TicketState State(const TicketId &ticket) const override {
        const auto found = states.find(ticket.requestId);
        return found == states.end() ? TicketState::FAILED : found->second;
    }
    void Stop() override { stopping = true; }
    void Complete(const TicketId &, Terminal) override {}
    uint64_t HostCreditsInUse() const override { return ledger.InUse(); }
    class Ledger : public BudgetLedger {
    public:
        bool Acquire(uint64_t bytes) override {
            if (bytes > kBaseWindowBytes - inUse) return false;
            inUse += bytes;
            if (inUse > peak) peak = inUse;
            return true;
        }
        void Release(uint64_t bytes) override { inUse = bytes > inUse ? 0 : inUse - bytes; }
        uint64_t InUse() const override { return inUse; }
        uint64_t Peak() const override { return peak; }
        uint64_t ReservedCompletionBytes() const override { return kMetadataWindowBytes; }
    private:
        uint64_t inUse = 0, peak = 0;
    };
    Ledger ledger;
    bool stopping = false;
private:
    std::map<uint64_t, TicketState> states;
};

static void orderingRules() {
    // Same request: the next checkpoint must be exactly sequence+1, so a later
    // checkpoint can never overtake a missing earlier one.
    CHECK(MayConsume({7, 3}, {7, 2}));
    CHECK(!MayConsume({7, 4}, {7, 2}));
    CHECK(!MayConsume({7, 2}, {7, 2}));
    // Different requests prepare in parallel.
    CHECK(MayConsume({8, 1}, {7, 5}));
    // Ticket ordering is by request then sequence.
    CHECK((TicketId{1, 2} < TicketId{1, 3}));
    CHECK((TicketId{1, 9} < TicketId{2, 0}));
    CHECK((TicketId{1, 2} == TicketId{1, 2}));
}

static void stateRules() {
    CHECK(TerminalState(TicketState::COMMITTED));
    CHECK(TerminalState(TicketState::FAILED));
    CHECK(TerminalState(TicketState::CANCELLED));
    CHECK(!TerminalState(TicketState::INDEXED));
    // DURABLE must never be retried as a fresh append: that is how a crash
    // between object write and catalog update would duplicate data.
    CHECK(!RetryableAsNewAppend(TicketState::DURABLE));
    CHECK(!RetryableAsNewAppend(TicketState::RECOVERING));
    CHECK(DurableButNotIndexed(TicketState::DURABLE));
    CHECK(DurableButNotIndexed(TicketState::RECOVERING));
    CHECK(RetryableAsNewAppend(TicketState::PREPARED));
    CHECK(RetryableAsNewAppend(TicketState::ADMITTED));
    // Distinct enum values, so a status code cannot be mistaken for another.
    const Admission values[] = {Admission::Accepted, Admission::Joined, Admission::AlreadyDurable,
                                Admission::CapacityUnavailable, Admission::Stopping};
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 5; ++j) CHECK(values[i] != values[j]);
    }
    CHECK(Terminal::Committed != Terminal::Failed);
}

static void budgetRules() {
    // 4 x 8 MiB pinned + 4 x 8 MiB staging + 2 x 16 MiB record staging is the
    // bulk of the documented 112 MiB base window; nothing may be chargeable
    // outside the same ledger.
    const uint64_t bulk = uint64_t(kPinnedSlots) * kSlotBytes +
                          uint64_t(kHashStagingSlots) * kSlotBytes +
                          kRecordStagingBytes + kMetadataWindowBytes;
    CHECK(bulk <= kBaseWindowBytes);
    CHECK(kBaseWindowBytes == (112ull << 20));
}

static void implementationIsUsable() {
    FakeCoordinator coordinator;
    TicketId admitted;
    // A request within budget is accepted and charged to the ledger.
    CHECK((coordinator.TryAdmit({1, 0}, "key", 1 << 20, admitted) == Admission::Accepted));
    CHECK(coordinator.ledger.InUse() == (1ull << 20));
    CHECK(coordinator.ledger.Peak() == (1ull << 20));
    CHECK(coordinator.State({1, 0}) == TicketState::ADMITTED);
    // Over-budget admission fails with CapacityUnavailable and does NOT wait.
    CHECK((coordinator.TryAdmit({2, 0}, "key2", kBaseWindowBytes, admitted) ==
           Admission::CapacityUnavailable));
    CHECK(coordinator.ledger.InUse() == (1ull << 20));  // unchanged by a refusal
    // Stopping beats capacity so shutdown cannot admit new work.
    coordinator.stopping = true;
    CHECK((coordinator.TryAdmit({3, 0}, "key3", 1, admitted) == Admission::Stopping));
    coordinator.stopping = false;
    // Drain must not report clean while credits are outstanding.
    std::string detail;
    CHECK(!coordinator.Drain(1000, detail));
    CHECK(detail == "credits_outstanding");
    coordinator.ledger.Release(1ull << 20);
    CHECK(coordinator.Drain(1000, detail));
    CHECK(detail == "clean");
    CHECK(coordinator.ledger.InUse() == 0);
}


// P3：非阻塞准入 / 生命周期 / 按序消费 / drain 的真实实现验收
static void realCoordinator() {
    auto coordinator = CreateCoordinator(kBaseWindowBytes);
    CHECK(coordinator != nullptr);
    if (!coordinator) return;
    TicketId admitted;
    // 预算内接收
    CHECK(coordinator->TryAdmit({1, 0}, "key-1", 1 << 20, admitted) == Admission::Accepted);
    // 超出总预算：**立即**返回 CapacityUnavailable，不等待
    CHECK(coordinator->TryAdmit({2, 0}, "key-2", kBaseWindowBytes, admitted) ==
          Admission::CapacityUnavailable);
    // 同一 key 在途 → Joined；已 durable → AlreadyDurable
    CHECK(coordinator->TryAdmit({3, 0}, "key-1", 1, admitted) == Admission::Joined);
    // 同请求乱序提交必须被拒绝（后面的检查点不得越过缺失的前序）
    CHECK(coordinator->Submit({1, 1}, {}) == Terminal::Failed);
    // 按序提交：入队后由固定哈希池处理
    CHECK(coordinator->Submit({1, 0}, {}) == Terminal::Committed);
    std::string detail;
    CHECK(coordinator->Drain(5000, detail));
    CHECK(detail == "clean");
    CHECK(coordinator->State({1, 0}) == TicketState::COMMITTED);
    // 关停后不再接收
    coordinator->Stop();
    CHECK(coordinator->TryAdmit({9, 0}, "key-9", 1, admitted) == Admission::Stopping);
}


// §6.1 机制层：K 个 pinned slot + 固定哈希池（按提交顺序交付）
static void hashPoolAndPinnedSlots() {
    auto slots = CreatePinnedSlotPool(kPinnedSlots, kSlotBytes);
    CHECK(slots != nullptr);
    if (slots) {
        CHECK(slots->Capacity() == kPinnedSlots);
        std::vector<int> held;
        for (int i = 0; i < kPinnedSlots; ++i) held.push_back(slots->Acquire());
        CHECK(slots->InUse() == kPinnedSlots);
        // 槽位满时必须立即失败，不等待
        CHECK(slots->Acquire() == -1);
        slots->Release(held[0]);
        CHECK(slots->InUse() == kPinnedSlots - 1);
        slots->Release(held[0]);        // 重复释放同一槽位必须无效
        CHECK(slots->InUse() == kPinnedSlots - 1);
        const int again = slots->Acquire();
        CHECK(again >= 0);
        CHECK(slots->InUse() == kPinnedSlots);
        for (int i = 1; i < kPinnedSlots; ++i) slots->Release(held[i]);
        slots->Release(again);
        CHECK(slots->InUse() == 0);
    }
    auto pool = CreateHashPackPool(size_t(8) << 20);
    CHECK(pool != nullptr);
    if (!pool) return;
    const std::string a(1000, 'a'), b(2000, 'b'), c(3000, 'c');
    CHECK(pool->Submit({1, 0}, a));
    CHECK(pool->Submit({1, 1}, b));
    CHECK(pool->Submit({1, 2}, c));
    CHECK(pool->InFlight() == 3);
    // 完成顺序可以乱，但交付必须按提交顺序，且不得越过缺失的前序。
    const TicketId expected[3] = {{1, 0}, {1, 1}, {1, 2}};
    std::string digests[3];
    for (int i = 0; i < 3; ++i) {
        TicketId ticket;
        std::string digest;
        CHECK(pool->TakeOrdered(ticket, digest, 5000));
        CHECK(ticket == expected[i]);
        CHECK(digest.size() == 64);
        digests[i] = digest;
    }
    CHECK(digests[0] != digests[1] && digests[1] != digests[2]);
    CHECK(pool->InFlight() == 0);
    // 同一内容两次提交必须得到同一摘要（独立上下文仍然确定性）
    CHECK(pool->Submit({1, 3}, a));
    {
        TicketId ticket; std::string digest;
        CHECK(pool->TakeOrdered(ticket, digest, 5000));
        CHECK(digest == digests[0]);
    }
    // 超出 staging 预算必须拒绝，而不是无限缓存
    CHECK(!pool->Submit({1, 4}, std::string(size_t(9) << 20, 'x')));
    pool->Stop();
}


static TicketId &coordinatorTicketScratch() { static TicketId ticket; return ticket; }

// §8 调度层接线：协调器必须真的驱动去重登记表，而不是只看一个在途集合 ——
// waiter 要落到 owner 的终态、owner 失败不得伪造成功、语义冲突必须与"资源不足"
// 分开、durable 复用要有界且换代即失效；没有语义描述的旧路径行为保持不变。
static void dedupSchedulerLayer() {
    auto registry = CreateDedupRegistry(1, 2);      // 完成缓存上限 2，便于验有界
    CHECK(registry != nullptr);
    if (!registry) return;
    DedupRegistry *raw = registry.get();            // 所有权转移给协调器，裸指针仅用于观察
    auto coordinator = CreateCoordinator(kBaseWindowBytes, std::move(registry));
    CHECK(coordinator != nullptr);
    if (!coordinator) return;

    const std::string described = "identity|prefix|provenance|layout-A";
    const std::string conflicting = "identity|prefix|provenance|layout-B";

    // owner 准入，waiter 同键同语义 → Joined，并拿到 owner 票据（等待只在后台图上）
    {
        TicketId admitted;
        CHECK(coordinator->TryAdmit({1, 0}, "key-A", 1 << 20, admitted, described) ==
              Admission::Accepted);
        CHECK(coordinator->State({1, 0}) == TicketState::ADMITTED);
        TicketId joined;
        CHECK(coordinator->TryAdmit({2, 0}, "key-A", 1 << 20, joined, described) ==
              Admission::Joined);
        CHECK(joined.requestId == 1 && joined.sequence == 0);
        CHECK(coordinator->State({2, 0}) == TicketState::ADMITTED);
        CHECK(raw->JoinCount() == 1);
        // 同键不同语义：必须是 Conflict，不能伪装成 CapacityUnavailable。
        TicketId ignored;
        CHECK(coordinator->TryAdmit({3, 0}, "key-A", 1, ignored, conflicting) ==
              Admission::Conflict);
        CHECK(coordinator->HostCreditsInUse() == (1 << 20));   // 冲突不占预算
        // owner 成功 → waiter 一起 COMMITTED（继承 owner 的结果）
        coordinator->Complete({1, 0}, Terminal::Committed);
        CHECK(coordinator->State({2, 0}) == TicketState::COMMITTED);
        CHECK(coordinator->HostCreditsInUse() == 0);
        // 已 durable：同语义可复用
        CHECK(coordinator->TryAdmit({4, 0}, "key-A", 1, ignored, described) ==
              Admission::AlreadyDurable);
        CHECK(raw->DurableCount() == 1);
    }

    // owner 失败：waiter 必须 FAILED（不伪造成功），且该键不得变成 durable
    {
        TicketId admitted, joined;
        CHECK(coordinator->TryAdmit({5, 0}, "key-B", 1 << 20, admitted, described) ==
              Admission::Accepted);
        CHECK(coordinator->TryAdmit({6, 0}, "key-B", 1 << 20, joined, described) ==
              Admission::Joined);
        coordinator->Complete({5, 0}, Terminal::Failed);
        CHECK(coordinator->State({5, 0}) == TicketState::FAILED);
        CHECK(coordinator->State({6, 0}) == TicketState::FAILED);
        CHECK(coordinator->HostCreditsInUse() == 0);
        TicketId ignored;
        // 失败后同键可以重新当 owner（没有被错误地标成 durable）
        CHECK(coordinator->TryAdmit({7, 0}, "key-B", 1 << 20, ignored, described) ==
              Admission::Accepted);
        coordinator->Complete({7, 0}, Terminal::Cancelled);
        CHECK(coordinator->State({7, 0}) == TicketState::CANCELLED);
    }

    // 完成缓存有界（登记表容量 2）：再写入不同键也不会无限增长
    {
        TicketId ignored;
        for (int i = 0; i < 4; ++i) {
            const std::string key = "key-C" + std::to_string(i);
            CHECK(coordinator->TryAdmit({uint64_t(10 + i), 0}, key, 1 << 20, ignored, described) ==
                  Admission::Accepted);
            coordinator->Complete({uint64_t(10 + i), 0}, Terminal::Committed);
        }
        CHECK(raw->DurableCount() <= 2);
    }

    // 换代即失效：旧描述不得无限保留
    raw->SetGeneration(2);
    CHECK(raw->DurableCount() == 0);
    {
        TicketId ignored;
        CHECK(coordinator->TryAdmit({20, 0}, "key-A", 1 << 20, ignored, described) ==
              Admission::Accepted);
        coordinator->Complete({20, 0}, Terminal::Committed);
    }

    // 没有语义描述的旧路径：行为与接线前一致（只在途去重，无 durable 复用）
    {
        TicketId admitted, joined;
        CHECK(coordinator->TryAdmit({30, 0}, "key-legacy", 1 << 20, admitted) ==
              Admission::Accepted);
        CHECK(coordinator->TryAdmit({31, 0}, "key-legacy", 1 << 20, joined) ==
              Admission::Joined);
        coordinator->Complete({30, 0}, Terminal::Committed);
        CHECK(coordinator->State({31, 0}) == TicketState::FAILED);   // 旧路径不传播终态
        TicketId again;
        CHECK(coordinator->TryAdmit({32, 0}, "key-legacy", 1 << 20, again) ==
              Admission::Accepted);                                  // 无 durable 复用
        coordinator->Complete({32, 0}, Terminal::Committed);
        CHECK(coordinator->HostCreditsInUse() == 0);
    }
    coordinator->Stop();
    CHECK(coordinator->TryAdmit({40, 0}, "key-A", 1, coordinatorTicketScratch(), described) ==
          Admission::Stopping);
}

// §6.1 D2-stream：K 个 pinned 槽位 + D2H/哈希重叠 + 定序交付。
// CUDA 事件与 D2H 由回调注入，所以这条流水在纯 CPU 上可以被完整驱动 —— 这正是
// "§6.1 在独立 CPU 测试里落地、不动模型写路径"的形式。
static std::string Sha256Hex(const std::string &bytes) {
    unsigned char out[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size(), out);
    static const char *hex = "0123456789abcdef";
    std::string text;
    text.reserve(SHA256_DIGEST_LENGTH * 2);
    for (unsigned char byte : out) {
        text.push_back(hex[byte >> 4]);
        text.push_back(hex[byte & 0x0F]);
    }
    return text;
}

static void d2StreamPipeline() {
    std::map<uint64_t, bool> ready;
    std::vector<uint64_t> copyOrder;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &ticket, size_t bytes) {
            copyOrder.push_back(ticket.sequence);
            return std::string(bytes, char('a' + int(ticket.sequence % 26)));
        },
        size_t(32) << 20);
    CHECK(stream != nullptr);
    if (!stream) return;
    CHECK(stream->SlotCapacity() == kPinnedSlots);   // K=4，全局共享
    const size_t oneMiB = size_t(1) << 20;

    // 事件都已完成：一次 PumpCopy 连续拷贝 K 份 —— D2H 不必等前一份哈希出结果。
    for (uint64_t i = 0; i < 4; ++i) {
        ready[i] = true;
        CHECK(stream->Submit({1, i}, i, oneMiB));
    }
    stream->PumpCopy();
    CHECK(copyOrder.size() == 4);
    CHECK(stream->SlotsInUse() == 4);
    CHECK(stream->SlotPeak() == 4);
    CHECK(stream->Pending() == 0);
    // 按提交顺序取回，且摘要必须等于 copier 写出的字节（顺序错位会立刻暴露）。
    for (uint64_t i = 0; i < 4; ++i) {
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
        CHECK(ticket.requestId == 1 && ticket.sequence == i);
        CHECK(digest == Sha256Hex(std::string(oneMiB, char('a' + int(i % 26)))));
    }
    CHECK(stream->SlotsInUse() == 0);

    // event 未完成：既不拷贝，也不越过它（严格按提交顺序）。
    ready[4] = false;
    CHECK(stream->Submit({1, 4}, 4, oneMiB));
    stream->PumpCopy();
    CHECK(copyOrder.size() == 4);
    CHECK(stream->Pending() == 1);
    ready[4] = true;
    stream->PumpCopy();
    CHECK(copyOrder.size() == 5 && copyOrder[4] == 4);
    {
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
        CHECK(ticket.sequence == 4);
    }
    CHECK(stream->SlotsInUse() == 0);

    // 前序 event 未完成时，后面的不得抢跑。
    ready[5] = false;
    ready[6] = true;
    CHECK(stream->Submit({1, 5}, 5, oneMiB));
    CHECK(stream->Submit({1, 6}, 6, oneMiB));
    stream->PumpCopy();
    CHECK(copyOrder.size() == 5);
    ready[5] = true;
    stream->PumpCopy();
    CHECK(copyOrder.size() == 7);
    CHECK(copyOrder[5] == 5 && copyOrder[6] == 6);
    for (int i = 0; i < 2; ++i) {
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
    }
    CHECK(stream->SlotsInUse() == 0);

    // staging 预算：**永远装不下**的条目仍然必须失败，且不允许半个条目进队。
    // R2：预算不足不再是“立即 false”，而是有界推迟
    // （默认上限 32 轮 × 1 ms 切片）；oversize（本行 64 MiB > 32 MiB 预算）推迟上限用尽后
    // 依旧返回 false——断言不变，只是耗时从 0 变成 <=32 ms。可变预算/可等待的行为另见
    // `d2SubmitDeferIsBounded` 与 `d2SubmitDeferWakesWhenBudgetFreed`（只增用例）。
    CHECK(!stream->Submit({1, 7}, 7, size_t(64) << 20));
    CHECK(stream->Pending() == 0);
    stream->Stop();
    CHECK(!stream->Submit({1, 8}, 8, oneMiB));
}

// §23（④ 第一步）：一次任务算多块 —— 交接次数从"每块一次"降到"每批一次"，
// 摘要与字节仍按提交顺序成对交付，预算超限仍必须拒绝。
// §④ 宽版前置：共享哈希池路径（旧代码在 Stop/析构时空指针崩溃）+ 失败出口
// （旧接口下"被拒"只能靠超时发现）+ 失败时槽位必须归还。
static void d2StreamSharedPoolAndFailure() {
    auto pool = CreateHashPackPool(size_t(32) << 20, 2);
    CHECK(pool != nullptr);
    if (!pool) return;
    std::map<uint64_t, bool> ready;
    std::vector<uint64_t> copyOrder;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &ticket, size_t bytes) {
            copyOrder.push_back(ticket.sequence);
            return std::string(bytes, 'z');
        },
        size_t(32) << 20, pool.get());
    CHECK(stream != nullptr);
    if (!stream) return;
    // ① 共享池正常路径：交付的摘要必须与拷贝内容一致，Failed 为假。
    const size_t oneMiB = size_t(1) << 20;
    ready[0] = true;
    CHECK(stream->Submit({2, 0}, 0, oneMiB));
    stream->PumpCopy();
    {
        TicketId ticket;
        HashedChunk chunk;
        CHECK(stream->TakeOrderedChunk(ticket, chunk, 5000));
        CHECK(ticket.requestId == 2 && ticket.sequence == 0);
        std::string digest, payload;
        CHECK(chunk.Take(digest, payload));
        CHECK(digest == Sha256Hex(std::string(oneMiB, 'z')));
        CHECK(!stream->Failed({2, 0}));
    }
    CHECK(stream->SlotsInUse() == 0);
    // ② 失败出口：哈希池先停 → PumpCopy 里池拒收 → 该 ticket 必须能被 Failed 查到，
    //    且槽位必须当场归还（否则就是槽位泄漏）。
    pool->Stop();
    ready[1] = true;
    CHECK(stream->Submit({2, 1}, 1, oneMiB));
    stream->PumpCopy();
    CHECK(stream->Failed({2, 1}));
    CHECK(!stream->Failed({2, 0}));
    CHECK(stream->SlotsInUse() == 0);
    // ③ 共享池路径的 Stop/析构：不得停别人的池、不得空指针（旧代码在此崩溃）。
    stream->Stop();
    CHECK(!stream->Submit({2, 2}, 2, oneMiB));
    stream.reset();
    CHECK(pool != nullptr);   // 池的生命周期由所有者负责，未被 D2Stream 提前销毁
}

// R2/R7：推迟与延迟水位参数在 D2Stream **构造时**读一次
// env，所以用例用这个 RAII 小工具在构造前设置、离开作用域（含提前 return）后精确还原，
// 避免污染同进程后续用例。
namespace {
struct ScopedEnv {
    std::string name;
    bool had = false;
    std::string oldValue;
    ScopedEnv(const char *key, const char *value) : name(key) {
        const char *existing = std::getenv(key);
        had = existing != nullptr;
        if (had) oldValue = existing;
        setenv(key, value, 1);
    }
    ~ScopedEnv() {
        if (had) setenv(name.c_str(), oldValue.c_str(), 1);
        else unsetenv(name.c_str());
    }
    ScopedEnv(const ScopedEnv &) = delete;
    ScopedEnv &operator=(const ScopedEnv &) = delete;
};
}   // namespace

// R2（W4）：预算不足时的 defer 必须有界——oversize 条目“永远装不下”，推迟上限用尽后
// 必须返回 false，且不得把半个条目留在 pending；计数必须可观测。
static void d2SubmitDeferIsBounded() {
    ScopedEnv maxDefers("FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS", "3");
    ScopedEnv waitSlice("FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS", "1");
    const auto before = PrefixWriteDeferStatsSnapshot();
    std::map<uint64_t, bool> ready;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &, size_t bytes) { return std::string(bytes, 'd'); },
        size_t(4) << 20);
    CHECK(stream != nullptr);
    if (!stream) return;
    const auto start = std::chrono::steady_clock::now();
    // 8 MiB > 4 MiB 预算：推迟 3 轮（每轮 1 ms 切片）后仍必须失败。
    CHECK(!stream->Submit({1, 0}, 0, size_t(8) << 20));
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    CHECK(stream->Pending() == 0);                       // 不接收半个条目
    CHECK(elapsedMs < 1000);                             // 有界：绝不无限等
    const auto after = PrefixWriteDeferStatsSnapshot();
    CHECK(after.submit_defers >= before.submit_defers + 3);
    CHECK(after.submit_defer_timeouts >= before.submit_defer_timeouts + 1);
    CHECK(after.defer_wait_ms_total >= before.defer_wait_ms_total);
    CHECK(after.max_submit_defers >= 3);
    // 预算够的条目不受影响（仍立即接收）。
    CHECK(stream->Submit({1, 1}, 1, size_t(1) << 20));
    CHECK(stream->Pending() == 1);
    stream->Stop();
}

// R2（W4）：预算被另一个线程（消费者）释放时，推迟中的 Submit 必须被唤醒并成功——
// 这正是“拒绝改推迟”的实际收益：不再因为暂时没空间就丢新检查点。
static void d2SubmitDeferWakesWhenBudgetFreed() {
    ScopedEnv maxDefers("FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS", "200");
    ScopedEnv waitSlice("FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS", "2");
    std::map<uint64_t, bool> ready;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &, size_t bytes) { return std::string(bytes, 'e'); },
        size_t(4) << 20);
    CHECK(stream != nullptr);
    if (!stream) return;
    const size_t oneMiB = size_t(1) << 20;
    for (uint64_t i = 0; i < 4; ++i) {
        ready[i] = true;
        CHECK(stream->Submit({1, i}, i, oneMiB));        // 占满 4 MiB 预算
    }
    const auto before = PrefixWriteDeferStatsSnapshot();
    std::atomic<bool> submitted{false};
    std::thread producer([&] { submitted.store(stream->Submit({2, 0}, 100, oneMiB)); });
    // 等到生产者**确实进入**推迟等待再放空间（轮询计数，避免"还没开始 Submit 就被喂饱"的竞争，
    // 否则本用例会在没有触发 defer 的情况下侥幸通过）。
    bool enteredDefer = false;
    for (int i = 0; i < 2000 && !enteredDefer; ++i) {
        enteredDefer = PrefixWriteDeferStatsSnapshot().submit_defers > before.submit_defers;
        if (!enteredDefer) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(enteredDefer);
    // 消费者推进拷贝：pendingBytes_ 归零 ⇒ 预算释放 ⇒ 必须唤醒推迟中的 Submit。
    // 注意：这里**不能**断言 Pending()==0 —— 生产者被唤醒后会立刻把新条目入队，Pending()
    // 是 0 还是 1 取决于两个线程的先后，属于竞态；预算确实被释放由 submitted==true 证明。
    stream->PumpCopy();
    producer.join();
    CHECK(submitted.load());                             // 等到了空间，新写被接收
    const auto after = PrefixWriteDeferStatsSnapshot();
    CHECK(after.submit_defers >= before.submit_defers + 1);
    CHECK(after.submit_defer_timeouts == before.submit_defer_timeouts);   // 不是靠超时失败
    CHECK(stream->Pending() == 1);
    // 清理：排空结果与 pinned 槽。全局 pinned 槽占用计数只在 Release 时回落，若把占用的槽
    // 留在池里随流析构，会抬高全局 slot_peak，污染后续 `pinnedSlotAndHashInterleavingStress`
    // 用例的 `peak <= kPinnedSlots` 断言（首次运行正是这样红的）。
    ready[100] = true;
    for (int guard = 0; guard < 64 && (stream->SlotsInUse() > 0 || stream->Pending() > 0); ++guard) {
        stream->PumpCopy();
        TicketId ticket;
        HashedChunk chunk;
        stream->TakeOrderedChunk(ticket, chunk, 200);
    }
    CHECK(stream->SlotsInUse() == 0);
    CHECK(stream->Pending() == 0);
    stream->Stop();
}

// R2（W4）：等待期间必须能被 Stop() 唤醒，不能拖到推迟窗口自然结束。
static void d2SubmitDeferStopWakes() {
    // 上限 400 × 5 ms = 2 s 的窗口；Stop() 若能唤醒，这里会立刻返回。
    ScopedEnv maxDefers("FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS", "400");
    ScopedEnv waitSlice("FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS", "5");
    std::map<uint64_t, bool> ready;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &, size_t bytes) { return std::string(bytes, 'f'); },
        size_t(1) << 20);
    CHECK(stream != nullptr);
    if (!stream) return;
    const size_t oneMiB = size_t(1) << 20;
    ready[0] = true;
    CHECK(stream->Submit({1, 0}, 0, oneMiB));            // 占满 1 MiB 预算
    std::atomic<bool> returned{false};
    std::atomic<bool> accepted{true};
    const auto before = PrefixWriteDeferStatsSnapshot();
    std::thread producer([&] {
        accepted.store(stream->Submit({2, 0}, 1, oneMiB));
        returned.store(true);
    });
    // 等到生产者确实进入推迟等待，再 Stop()；否则本用例测不到"唤醒"这件事。
    bool enteredDefer = false;
    for (int i = 0; i < 2000 && !enteredDefer; ++i) {
        enteredDefer = PrefixWriteDeferStatsSnapshot().submit_defers > before.submit_defers;
        if (!enteredDefer) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(enteredDefer);
    const auto start = std::chrono::steady_clock::now();
    stream->Stop();
    producer.join();                                     // 最坏 2 s（窗口上限），不会无限挂
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    CHECK(returned.load());
    CHECK(!accepted.load());                             // Stop() 后必须拒绝
    CHECK(elapsedMs < 1500);                             // 被 Stop() 唤醒，而不是等满 2 s
}

// R7：D2 暂存的“延迟比率”水位——慢拷贝把 EMA 顶到高水位后，Submit 会
// 推迟（可观测），快拷贝把 EMA 压回低水位后恢复“不再推迟”。水位只软化背压，不丢新写。
static void d2LatencyWatermarkDefersThenRecovers() {
    ScopedEnv maxDefers("FASTLLM_PREFIX_CACHE_D2_MAX_SUBMIT_DEFERS", "64");
    ScopedEnv waitSlice("FASTLLM_PREFIX_CACHE_D2_DEFER_WAIT_MS", "1");
    ScopedEnv high("FASTLLM_PREFIX_CACHE_D2_LAT_HIGH_S", "0.010");
    ScopedEnv low("FASTLLM_PREFIX_CACHE_D2_LAT_LOW_S", "0.002");
    ScopedEnv alpha("FASTLLM_PREFIX_CACHE_D2_LAT_ALPHA", "0.15");
    ScopedEnv cooldown("FASTLLM_PREFIX_CACHE_D2_LAT_COOLDOWN_MS", "0");
    std::map<uint64_t, bool> ready;
    size_t copyDelayUs = 0;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &, size_t bytes) {
            if (copyDelayUs != 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(copyDelayUs));
            }
            return std::string(bytes, 'g');
        },
        size_t(64) << 20);
    CHECK(stream != nullptr);
    if (!stream) return;
    const size_t oneMiB = size_t(1) << 20;
    const auto drainOne = [&](uint64_t sequence) {
        ready[sequence] = true;
        CHECK(stream->Submit({1, sequence}, sequence, oneMiB));
        stream->PumpCopy();
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
    };
    const auto baseline = PrefixWriteDeferStatsSnapshot();

    // ① 正常（快拷贝）：没有高水位，不产生 watermark defer。
    drainOne(0);
    const auto quiet = PrefixWriteDeferStatsSnapshot();
    CHECK(quiet.watermark_defers == baseline.watermark_defers);
    CHECK(quiet.latency_high_entries == baseline.latency_high_entries);

    // ② 120 ms / 1 MiB = 0.12 s/MiB ⇒ 首个 EMA 样本 0.018 > 0.010 ⇒ 立刻进入高水位。
    copyDelayUs = 120000;
    drainOne(1);
    const auto highState = PrefixWriteDeferStatsSnapshot();
    CHECK(highState.latency_high_entries > baseline.latency_high_entries);
    CHECK(highState.latency_ema_us_per_mib > 0);

    // ③ 高水位下即使预算够，也必须先推迟一次（watermark defer），随后仍被接收。
    const auto beforeHighSubmit = PrefixWriteDeferStatsSnapshot();
    ready[2] = true;
    CHECK(stream->Submit({1, 2}, 2, oneMiB));            // 预算够：推迟一次后成功
    const auto afterHighSubmit = PrefixWriteDeferStatsSnapshot();
    CHECK(afterHighSubmit.watermark_defers > beforeHighSubmit.watermark_defers);
    stream->PumpCopy();
    {
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
    }

    // ④ 快拷贝把 EMA 以 alpha=0.15 衰减回低水位；之后的新提交不再触发 watermark defer。
    copyDelayUs = 0;
    for (uint64_t sequence = 3; sequence < 60; ++sequence) drainOne(sequence);
    const auto beforeQuietSubmit = PrefixWriteDeferStatsSnapshot();
    ready[999] = true;
    CHECK(stream->Submit({1, 999}, 999, oneMiB));
    const auto afterQuietSubmit = PrefixWriteDeferStatsSnapshot();
    CHECK(afterQuietSubmit.watermark_defers == beforeQuietSubmit.watermark_defers);
    stream->PumpCopy();
    {
        TicketId ticket;
        std::string digest;
        CHECK(stream->TakeOrdered(ticket, digest, 5000));
    }
    stream->Stop();
}

static void hashPoolBatch() {
    auto pool = CreateHashPackPool(size_t(16) << 20, 2);
    CHECK(pool != nullptr);
    if (!pool) return;
    std::vector<std::string> first{"alpha", std::string(1024, 'b'), "gamma"};
    CHECK(pool->SubmitBatch({1, 0}, first));
    std::vector<std::string> second{"delta"};
    CHECK(pool->SubmitBatch({1, 1}, second));
    TicketId ticket;
    std::vector<HashedChunk> chunks;
    CHECK(pool->TakeOrderedBatch(ticket, chunks, 5000));
    CHECK(ticket.sequence == 0);
    CHECK(chunks.size() == 3);
    if (chunks.size() == 3) {
        std::string digest, payload;
        CHECK(chunks[0].Take(digest, payload));
        CHECK(digest == Sha256Hex("alpha") && payload == "alpha");
        CHECK(chunks[1].Take(digest, payload));
        CHECK(digest == Sha256Hex(std::string(1024, 'b')) && payload == std::string(1024, 'b'));
        CHECK(chunks[2].Take(digest, payload));
        CHECK(digest == Sha256Hex("gamma") && payload == "gamma");
    }
    // 第二批按提交顺序在其后取回
    CHECK(pool->TakeOrderedBatch(ticket, chunks, 5000));
    CHECK(ticket.sequence == 1 && chunks.size() == 1);
    // 单块取回与批量混用：批量提交 1 块，用 TakeOrderedChunk 取
    CHECK(pool->SubmitBatch({1, 2}, {"epsilon"}));
    HashedChunk single;
    CHECK(pool->TakeOrderedChunk(ticket, single, 5000));
    CHECK(single.Valid());
    // 超预算整批拒绝，不部分接收
    CHECK(!pool->SubmitBatch({1, 3}, {std::string(size_t(20) << 20, 'x')}));
    CHECK(pool->InFlight() == 0);
    pool->Stop();
}

// §8 去重协议机制层
static void dedupProtocol() {
    auto registry = CreateDedupRegistry(1, 2);   // generation=1, durable 容量上限 2
    CHECK(registry != nullptr);
    if (!registry) return;
    const std::string identity = "identity", prefix = "prefix-key",
                      provenance = "fp16-to-fp8-v1", layout = "layout-A";
    const std::string key = DedupRegistry::Key(identity, prefix, provenance, layout);
    CHECK(key.size() == 64);
    // 四类语义任一变化都必须换键（不接受长度/指针等不可靠信息）
    CHECK(DedupRegistry::Key(identity, prefix, provenance, "layout-B") != key);
    CHECK(DedupRegistry::Key(identity, prefix, "native", layout) != key);
    CHECK(DedupRegistry::Key(identity, "other-prefix", provenance, layout) != key);
    CHECK(DedupRegistry::Key("other-identity", prefix, provenance, layout) != key);

    uint64_t owner = 0;
    CHECK(registry->Register(key, "desc-1", 10, owner) == DedupRegistry::Registration::Owner);
    CHECK(owner == 10);
    // 在途同 key：登记为 waiter，返回 owner 票据（等待只在后台完成图上）
    CHECK(registry->Register(key, "desc-1", 11, owner) == DedupRegistry::Registration::Joined);
    CHECK(owner == 10);
    CHECK(registry->JoinCount() == 1);
    // 同 key 但语义描述不同：显式冲突，绝不覆盖
    CHECK(registry->Register(key, "desc-2", 12, owner) == DedupRegistry::Registration::Conflict);
    // owner 失败：waiters 原样交回，且**不得**变成 durable（不伪造成功）
    std::vector<uint64_t> waiters;
    registry->Complete(key, false, 10, waiters);
    CHECK(waiters.size() == 1 && waiters[0] == 11);
    std::string descriptor;
    CHECK(!registry->LookupDurable(key, descriptor));
    CHECK(registry->DurableCount() == 0);
    // 失败后重新登记仍是 owner
    CHECK(registry->Register(key, "desc-1", 20, owner) == DedupRegistry::Registration::Owner);
    CHECK(registry->Register(key, "desc-1", 21, owner) == DedupRegistry::Registration::Joined);
    registry->Complete(key, true, 20, waiters);
    CHECK(waiters.size() == 1 && waiters[0] == 21);
    CHECK(registry->LookupDurable(key, descriptor));
    // 已 durable：同语义可复用，不同语义冲突
    CHECK(registry->Register(key, "desc-1", 22, owner) == DedupRegistry::Registration::Durable);
    CHECK(registry->Register(key, "desc-3", 23, owner) == DedupRegistry::Registration::Conflict);
    // 完成缓存有界：容量 2，写入第 3 个不同的 key 后仍不超过 2
    for (int i = 0; i < 3; ++i) {
        const std::string other = DedupRegistry::Key(identity, prefix, provenance,
                                                     "layout-" + std::to_string(i));
        CHECK(registry->Register(other, "d", uint64_t(100 + i), owner) ==
              DedupRegistry::Registration::Owner);
        registry->Complete(other, true, uint64_t(100 + i), waiters);
    }
    CHECK(registry->DurableCount() <= 2);
    // 换代即失效，不无限保留旧描述
    registry->SetGeneration(2);
    CHECK(!registry->LookupDurable(key, descriptor));
    CHECK(registry->DurableCount() == 0);
}


// §17.1 调度与资源：交错压力后所有计数必须归零，且槽位不超限、不双重释放
static void resourceCountersReturnToZero() {
    auto coordinator = CreateCoordinator(kBaseWindowBytes);
    auto slots = CreatePinnedSlotPool(kPinnedSlots, kSlotBytes);
    auto pool = CreateHashPackPool(size_t(8) << 20);
    CHECK(coordinator && slots && pool);
    if (!coordinator || !slots || !pool) return;
    for (int round = 0; round < 20; ++round) {
        std::vector<TicketId> tickets;
        std::vector<int> held;
        for (int i = 0; i < kPinnedSlots; ++i) {
            const TicketId ticket{1, uint64_t(round) * kPinnedSlots + i};
            TicketId admitted;
            const auto admission = coordinator->TryAdmit(
                ticket, "k" + std::to_string(round) + ":" + std::to_string(i), 1 << 20, admitted);
            CHECK(admission == Admission::Accepted);
            tickets.push_back(admitted);
            const int slot = slots->Acquire();
            CHECK(slot >= 0);
            held.push_back(slot);
            CHECK(pool->Submit(admitted, std::string(size_t(1000 + i), 'x')));
        }
        // 槽位占满时不得再给（不提前复用）
        CHECK(slots->Acquire() == -1);
        // 哈希结果必须按提交顺序取回
        for (size_t i = 0; i < tickets.size(); ++i) {
            TicketId ticket;
            std::string digest;
            CHECK(pool->TakeOrdered(ticket, digest, 5000));
            CHECK(ticket == tickets[i]);
            CHECK(digest.size() == 64);
        }
        // 归还：槽位 + 协调器信用
        for (int slot : held) slots->Release(slot);
        for (const auto &ticket : tickets) coordinator->Complete(ticket, Terminal::Committed);
        CHECK(slots->InUse() == 0);
        CHECK(coordinator->HostCreditsInUse() == 0);
        CHECK(pool->InFlight() == 0 && pool->InFlightBytes() == 0);
    }
    // 全部归零，且 drain 报 clean
    CHECK(slots->InUse() == 0);
    CHECK(coordinator->HostCreditsInUse() == 0);
    std::string detail;
    CHECK(coordinator->Drain(2000, detail));
    CHECK(detail == "clean");
    pool->Stop();
    coordinator->Stop();
}

// A3-5（2026-10-02）：§17.1"K 个 pinned slot 循环交错"。
// K=4 个 pinned 槽 + 真实哈希池（2 worker）+ D2 中继同时跑，交错"慢事件 / 慢哈希"，
// 断言：①不丢点（提交数 == 交付数）；②不误判（每条摘要/本体严格对应它自己的 ticket）；
// ③槽位不超 K、不提前复用（未取回不得释放）；④最终槽位/在途全部归零，releases<=acquires
// （不双重释放），peak<=K（不无界增长）。只增用例，不改既有断言。
static void pinnedSlotAndHashInterleavingStress() {
    const auto slotsBefore = PrefixWritePinnedSlotStatsSnapshot();
    auto pool = CreateHashPackPool(size_t(16) << 20, 2);
    CHECK(pool != nullptr);
    if (!pool) return;
    const size_t base = size_t(1) << 20;
    std::map<uint64_t, bool> ready;
    std::map<uint64_t, size_t> sizes;
    std::map<uint64_t, std::string> expected;
    auto stream = CreateD2Stream(
        [&](uint64_t eventId) { return ready[eventId]; },
        [&](int, const TicketId &ticket, size_t bytes) {
            return std::string(bytes, char('a' + int(ticket.sequence % 26)));
        },
        size_t(48) << 20, pool.get());
    CHECK(stream != nullptr);
    if (!stream) { pool->Stop(); return; }
    CHECK(stream->SlotCapacity() == kPinnedSlots);   // K=4
    uint64_t next = 0;
    size_t delivered = 0;
    std::set<uint64_t> deliveredTickets;
    const int kRounds = 5;
    const int kPerRound = 4;
    for (int round = 0; round < kRounds; ++round) {
        std::vector<uint64_t> slow;
        for (int i = 0; i < kPerRound; ++i, ++next) {
            const size_t bytes = base + size_t((next % 3) * 4096);
            sizes[next] = bytes;
            expected[next] = std::string(bytes, char('a' + int(next % 26)));
            // 每轮留一个"慢事件"：本轮不就绪，轮末才放开——考"不得越过未完成事件"。
            ready[next] = (i != kPerRound - 1);
            if (i == kPerRound - 1) slow.push_back(next);
            CHECK(stream->Submit({9, next}, next, bytes));
        }
        for (int step = 0; step < 6; ++step) {
            stream->PumpCopy();
            CHECK(stream->SlotsInUse() <= kPinnedSlots);   // 不提前复用/不超限
            for (int drain = 0; drain < 2; ++drain) {
                TicketId ticket;
                HashedChunk chunk;
                if (!stream->TakeOrderedChunk(ticket, chunk, 50)) break;
                std::string digest, payload;
                CHECK(chunk.Take(digest, payload));
                CHECK(deliveredTickets.insert(ticket.sequence).second);   // 不重复交付
                CHECK(sizes.count(ticket.sequence) == 1);
                CHECK(digest == Sha256Hex(expected[ticket.sequence]));     // 不误判
                CHECK(payload == expected[ticket.sequence]);
                ++delivered;
            }
        }
        for (uint64_t id : slow) ready[id] = true;   // 慢事件追上
    }
    // 收尾：把剩下的全部取回。
    for (int i = 0; i < 64; ++i) {
        stream->PumpCopy();
        TicketId ticket;
        HashedChunk chunk;
        if (!stream->TakeOrderedChunk(ticket, chunk, 1000)) {
            if (stream->Pending() == 0 && stream->SlotsInUse() == 0) break;
            continue;
        }
        std::string digest, payload;
        CHECK(chunk.Take(digest, payload));
        CHECK(deliveredTickets.insert(ticket.sequence).second);
        CHECK(digest == Sha256Hex(expected[ticket.sequence]));
        CHECK(payload == expected[ticket.sequence]);
        ++delivered;
    }
    CHECK(delivered == next);                              // 不丢点
    CHECK(deliveredTickets.size() == size_t(next));
    CHECK(stream->SlotsInUse() == 0);
    CHECK(stream->SlotPeak() <= kPinnedSlots);
    CHECK(stream->Pending() == 0);
    stream->Stop();
    pool->Stop();
    CHECK(pool->InFlight() == 0 && pool->InFlightBytes() == 0);
    const auto slotsAfter = PrefixWritePinnedSlotStatsSnapshot();
    CHECK(slotsAfter.peak <= uint64_t(kPinnedSlots));      // 不无界增长
    CHECK(slotsAfter.releases <= slotsAfter.acquires);     // 不双重释放
    CHECK(slotsAfter.acquires >= slotsBefore.acquires);
}

int main() {
    try {
        orderingRules();
        stateRules();
        budgetRules();
        implementationIsUsable();
        realCoordinator();
        hashPoolAndPinnedSlots();
        dedupProtocol();
        dedupSchedulerLayer();
        d2StreamPipeline();
        d2StreamSharedPoolAndFailure();
        // R2/R7：有界推迟 + 延迟水位（只增用例）
        d2SubmitDeferIsBounded();
        d2SubmitDeferWakesWhenBudgetFreed();
        d2SubmitDeferStopWakes();
        d2LatencyWatermarkDefersThenRecovers();
        hashPoolBatch();
        resourceCountersReturnToZero();
        pinnedSlotAndHashInterleavingStress();   // A3-5（§17.1 K 个 pinned slot 交错压力）
    } catch (const std::exception &error) {
        std::cerr << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "FAILED: " << failures << " check(s)\n";
        return 1;
    }
    std::cout << "PASS disk_prefix_write_coordinator_contract\n";
    return 0;
}
