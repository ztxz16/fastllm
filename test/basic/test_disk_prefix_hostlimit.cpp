// P1 补齐：SSD_CACHE_REWRITE_PLAN.md §17.1「调度与资源」条款
//   「HostLimit 接近下限仍有完成窗口，无死锁」
//
// 背景（为什么单独写这个用例）：
//   2026-09-30 起 v4 store 的提交改成了「后台备菜流水线」——请求线程只把
//   「状态 Json + 本次暂存分块」移交进一个**深度上限 2** 的 FIFO，备菜线程按受理
//   顺序做序列化/摘要/预检/装配，再投给唯一写盘 actor。于是主机内存曲线从
//   「一次一笔」变成「最多 2 笔暂存 + 1 笔在途」。本用例就是要在「资源贴近下限」
//   时证明：要么**明确拒绝并报告**，要么**排队后完成**，**不得死锁**。
//
// 【口径如实声明——本文件能测什么、不能测什么】
//   产品里真正读 `FASTLLM_PREFIX_CACHE_HOST_LIMIT_MB` 的准入判断在**模型层**
//   （src/models/qwen3_5_persistent.inc:23-25，超限时打印 `write_skipped ...
//   reason=host_budget`）。DiskPrefixV4Store 的公开 API **不读这个环境变量**，
//   CPU 单测也起不了模型/服务。因此本文件用 store 自己**真实存在**的两个资源下限
//   来覆盖同一条件的两条分支（映射关系写清楚，不含糊）：
//     · 持久化预算下限 = `Open(dir, quotaBytes, ...)` 的 quotaBytes。它是 store
//       真实物理日志预算；贴下限时先按 FIFO 追加 ERASE 退掉最老的检查点，
//       以 `capacity_unavailable` 明确报告本次未持久化。Compact 回收物理字节后
//       调用方显式重试；roots.live=0 不等于物理空间已释放。
//     · 主机暂存下限 = 备菜 FIFO 的深度上限 2（disk_prefix_cache.cpp:2124
//       `kPipelineDepth = 2`、:2288-2310 `EnqueuePipeline` 背压）。队列满时受理方
//       会**等队首完成**（把队首失败原样上报），这正是「排队后完成 / 不静默丢点」
//       的落点，也是死锁最可能出现的地方。
//   模型层的 HostLimit 数值本身（A/B 用的 2048 MiB）**未在本用例中验证**，
//   本文件也不声称验证了它；这是明确列出的缺口，见交付报告。
//
// 全部在单进程内跑完：不起服务、不用 GPU、不写 /tmp（TMPDIR 指向 /tmp 时 REFUSED）。
#include "utils/disk_prefix_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
using fastllm::DiskPrefixCache;
using fastllm::DiskPrefixV4Store;

static int failures = 0;
#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::cerr << "FAIL line " << __LINE__ << ": " #condition << "\n"; \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// 夹具约束（与 test_disk_prefix_store_v4.cpp 同款）：禁止写 /tmp。
// TMPDIR 指向 /tmp 时直接拒绝运行，而不是默默照做。
// ---------------------------------------------------------------------------
static fs::path VerifiedTempBase(const char *tag) {
    const char *tmp = std::getenv("TMPDIR");
    if (tmp != nullptr && std::string(tmp).rfind("/tmp", 0) == 0) {
        std::cerr << "REFUSED: TMPDIR points at " << tmp
                  << " but this project forbids writing under /tmp (" << tag << ")\n";
        std::exit(2);
    }
    return tmp ? fs::path(tmp) : fs::current_path();
}

static fs::path TempRoot() {
    std::string pattern = (VerifiedTempBase("hostlimit") / "fastllm-hostlimit-test-XXXXXX").string();
    char *name = mkdtemp(pattern.data());
    CHECK(name != nullptr);
    return name ? fs::path(name) : fs::current_path();
}

// ---------------------------------------------------------------------------
// 看门狗：需求①「不卡死」。任何一步超过 budget 秒未推进就 _exit(3)，并打印
// 「卡在哪个用例的哪一步」——不是靠 CTest 的整体 TIMEOUT 才知道超时。
// note_ 用 atomic<const char*> 存字符串字面量，避免跨线程读 std::string 的数据竞争。
// ---------------------------------------------------------------------------
class Watchdog {
public:
    Watchdog(const char *name, int seconds) : name_(name) {
        thread_ = std::thread([this, seconds] {
            for (int i = 0; i < seconds * 20; ++i) {
                if (done_.load()) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            if (!done_.load()) {
                std::fprintf(stderr, "WATCHDOG TIMEOUT: case=%s stuck at step %d (%s)\n",
                             name_, step_.load(), note_.load());
                std::fflush(stderr);
                _exit(3);   // 超时即失败，退出码 3 与断言失败(1)区分开
            }
        });
    }
    ~Watchdog() {
        done_ = true;
        if (thread_.joinable()) thread_.join();
    }
    void Step(int n, const char *what) {
        step_ = n;
        note_ = what;
        std::fprintf(stderr, "[step] %s #%d: %s\n", name_, n, what);
        std::fflush(stderr);
    }

private:
    const char *name_;
    std::atomic<bool> done_{false};
    std::atomic<int> step_{0};
    std::atomic<const char *> note_{""};
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Profile 是 JSON 文本；按 "field": <int> 取值（与既有 profileDiagnostics 同口径）。
// 缺字段返回 -1，调用方可据此判失败而不是静默当成 0。
// ---------------------------------------------------------------------------
static long long ProfileInt(const std::string &profile, const char *field) {
    const std::string needle = std::string("\"") + field + "\": ";
    const size_t at = profile.find(needle);
    if (at == std::string::npos) return -1;
    return std::strtoll(profile.c_str() + at + needle.size(), nullptr, 10);
}

static DiskPrefixV4Store::Object MakeObject(const std::string &bytes, uint32_t kind) {
    DiskPrefixV4Store::Object object;
    object.bytes = bytes;
    object.kind = kind;
    object.hash = DiskPrefixCache::Digest(bytes);
    return object;
}

// 一笔「大 state 提交」的素材：chunksPer 个 chunkBytes 的暂存分块 + 一段 padBytes
// 的状态正文（模拟真实状态 Json 随上下文变长）。用 SubmitStaged 直接驱动 store 的
// 备菜流水线（模型层 v3 外壳走的正是这条 API，见 disk_prefix_cache.cpp:1545）。
struct StagedFixture {
    std::string key, identity, prefix;
    std::vector<std::pair<std::string, std::string>> staged;   // (hash, bytes)
    std::vector<std::string> chunkHashes;
    DiskPrefixCache::Json state;
};

static StagedFixture MakeStagedFixture(int index, const std::string &identity,
                                       const std::string &family, size_t chunkBytes,
                                       int chunksPer, size_t padBytes) {
    StagedFixture f;
    f.key = "hostlimit-key-" + std::to_string(index);
    f.identity = identity;
    f.prefix = DiskPrefixCache::Digest("hostlimit-prefix-" + std::to_string(index));
    DiskPrefixCache::Json::array kv;
    for (int j = 0; j < chunksPer; ++j) {
        std::string bytes(chunkBytes, char('A' + (index * 7 + j) % 26));
        const std::string tag = "chunk-" + std::to_string(index) + "-" + std::to_string(j);
        std::memcpy(bytes.data(), tag.data(), std::min(tag.size(), bytes.size()));
        const std::string hash = DiskPrefixCache::Digest(bytes);
        f.chunkHashes.push_back(hash);
        f.staged.emplace_back(hash, std::move(bytes));
        // 状态 Json 按产品同款口径引用这些分块（{sha256, bytes}），
        // 这样 root 清单的「状态本体 + 被引用对象」都要登记齐全。
        kv.push_back(DiskPrefixCache::Json::object{
            {"sha256", hash}, {"bytes", double(chunkBytes)}});
    }
    const std::string padding(padBytes, char('a' + (index % 26)));
    f.state = DiskPrefixCache::Json::object{
        {"family_identity", family},
        {"kind", "prefix"},
        {"length", 2048 + index * 2048},          // 模拟 2048 递增的检查点长度
        {"kv_dtype", "float16"},
        {"provenance", "native"},
        {"prefix", f.prefix},
        {"padding", padding},
        {"kv", kv},
    };
    return f;
}

// 与 store 内部 MetadataText() 完全同口径：状态对象字节 = value.dump_compact()。
// 用它算准预算，也用它做 Load 的逐字节比对。
static std::string StateText(const StagedFixture &f) { return f.state.dump_compact(); }

// 对象载荷与物理日志字节分开。quota 由下面的公开接口实测，
// 不用 total+224/object 代替对齐的 OBJECT/PREPARE/METADATA/COMMIT 记录。
struct Budget {
    uint64_t total = 0;
    uint64_t stateTextBytes = 0;
};

static Budget Measure(const StagedFixture &f, size_t chunkBytes, int chunksPer) {
    Budget b;
    b.stateTextBytes = StateText(f).size();
    b.total = uint64_t(chunksPer) * chunkBytes + b.stateTextBytes;
    return b;
}

static uint64_t MeasurePhysical(const fs::path &root, const StagedFixture &fixture,
                                const std::string &family) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return 0;
    auto copy = fixture;
    CHECK(store->SubmitStaged(copy.key, copy.identity, family, copy.state, false,
                              std::move(copy.staged), error));
    CHECK(store->ReapCheckpoint(nullptr, error));
    CHECK(store->Compact(error));
    return store->PhysicalBytes();
}

// One accepted attempt must acknowledge exactly one terminal result. A capacity
// failure is retryable only after explicit physical reclamation; it is never
// counted as durable merely because the FIFO accepted the job.
static bool SubmitAndReap(DiskPrefixV4Store &store, StagedFixture fixture,
                           const std::string &family, uint64_t *sequence,
                           std::string &error) {
    std::atomic<int> acknowledgements{0}, durable{0};
    error.clear();
    const bool accepted = store.SubmitStaged(fixture.key, fixture.identity, family,
        fixture.state, false, std::move(fixture.staged), error,
        [&](bool ok) { if (ok) ++durable; ++acknowledgements; });
    CHECK(accepted);
    if (!accepted) {
        CHECK(!error.empty());
        CHECK(acknowledgements.load() == 0);
        return false;
    }
    error.clear();
    const bool reaped = store.ReapCheckpoint(sequence, error);
    CHECK(acknowledgements.load() == 1);
    CHECK(durable.load() == (reaped ? 1 : 0));
    CHECK(store.InFlightBytes() == 0);
    return reaped;
}

// Load 必须原样取回「3 个分块 + 状态本体」，一个不少、一个不多、逐字节一致。
static bool LoadExact(DiskPrefixV4Store &store, const StagedFixture &f, std::string &error) {
    std::string identity;
    std::vector<std::string> loaded;
    if (!store.Load(f.key, identity, loaded, error)) return false;
    if (identity != f.identity) return false;
    std::set<std::string> expected(f.chunkHashes.begin(), f.chunkHashes.end());
    bool stateSeen = false;
    for (const auto &bytes : loaded) {
        const std::string digest = DiskPrefixCache::Digest(bytes);
        const auto it = expected.find(digest);
        if (it != expected.end()) { expected.erase(it); continue; }
        if (bytes == StateText(f)) { stateSeen = true; continue; }
        return false;   // 多出一个不属于本检查点的对象
    }
    return expected.empty() && stateSeen;
}

// ===========================================================================
// 用例 1：主机暂存贴下限（备菜 FIFO 深度上限 2）+ 排队后完成 + 无死锁
//
// 设计：物理 quota 先按真实 Compact 后单检查点占用放大到足够覆盖 5 笔；**连续提交 5 笔且中途不
// Reap**，把备菜 FIFO 顶到深度 2，从第 3 笔起受理方必然走 EnqueuePipeline 的
// 「等队首完成」背压分支。若这条路径会死锁（例如队首的备菜需要受理线程做点什么），
// 用例就会在看门狗处超时并打印卡在第几笔。
// ===========================================================================
static void hostPipelineDepthTwoNoDeadlock(const fs::path &root) {
    Watchdog wd("pipeline-depth-2", 45);
    const std::string identity = DiskPrefixCache::Digest("hostlimit-identity");
    const std::string family = DiskPrefixCache::Digest("hostlimit-family");
    const size_t chunkBytes = 1u << 20;      // 1 MiB/分块
    const int chunksPer = 3;                 // 每笔 3 个暂存分块
    const size_t padBytes = 512u << 10;      // 每笔 512 KiB 状态正文
    const int commits = 5;

    const auto probe = MakeStagedFixture(0, identity, family, chunkBytes, chunksPer, padBytes);
    const Budget budget = Measure(probe, chunkBytes, chunksPer);
    // 这一用例只测试 FIFO 深度2和背压，不把物理磁盘配额失败混入调度断言。
    // 先用真实紧凑后的单检查点物理占用给出有界但足够的总配额。
    const uint64_t onePhysical = MeasurePhysical(root / "sizing", probe, family);
    const uint64_t quota = onePhysical * uint64_t(commits) + (1ull << 20);
    std::fprintf(stderr,
                 "[budget] case1 total=%llu one_physical=%llu stateText=%llu quota=%llu (MiB: %.2f)\n",
                 (unsigned long long)budget.total, (unsigned long long)onePhysical,
                 (unsigned long long)budget.stateTextBytes, (unsigned long long)quota,
                 double(quota) / 1048576.0);
    CHECK(onePhysical > budget.total);

    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "open: " << error << "\n"; return; }

    std::vector<StagedFixture> fixtures;
    for (int i = 0; i < commits; ++i)
        fixtures.push_back(MakeStagedFixture(i, identity, family, chunkBytes, chunksPer, padBytes));

    // ① 连续受理，中途不 Reap —— 故意把 FIFO 顶满（深度 2）。
    wd.Step(1, "submit 5 staged commits without reaping (FIFO back-pressure)");
    int accepted = 0;
    for (int i = 0; i < commits; ++i) {
        wd.Step(10 + i, "SubmitStaged");
        error.clear();
        const bool ok = store->SubmitStaged(fixtures[i].key, fixtures[i].identity, family,
                                            fixtures[i].state, /*keepEquivalent=*/false,
                                            std::move(fixtures[i].staged), error);
        if (!ok) {
            // 受理侧若拒绝，必须是**可见失败**（带错误串），绝不静默丢点。
            CHECK(!error.empty());
            std::fprintf(stderr, "[budget] submit #%d rejected visibly: %s\n", i, error.c_str());
        } else {
            ++accepted;
        }
    }
    // store 没有「受理期预算」，只做背压排队：5 笔必须全部被受理。
    CHECK(accepted == commits);
    // 被受理的暂存数据已在 SubmitStaged 里 move 进队列，fixtures 的 staged 已空。

    // ② 收尾：所有已受理的提交必须完成（排队后完成），失败要原样上报。
    wd.Step(30, "ReapCheckpoint drains the whole FIFO");
    uint64_t sequence = 0;
    const bool reaped = store->ReapCheckpoint(&sequence, error);
    if (!reaped) std::fprintf(stderr, "[budget] reap failed: %s\n", error.c_str());
    CHECK(reaped);
    CHECK(sequence > 0);

    // ③ 计数与内存峰值有界：FIFO 观测深度绝不超过 2（= 主机暂存上界 2 笔）。
    wd.Step(31, "Profile(true) bounded counters");
    const std::string profile = store->Profile(true);
    std::fprintf(stderr, "[budget] profile: prep_jobs=%lld prep_max_depth=%lld live=%lld live_bytes=%lld\n",
                 ProfileInt(profile, "prep_jobs"), ProfileInt(profile, "prep_max_depth"),
                 ProfileInt(profile, "live_checkpoints"), ProfileInt(profile, "live_bytes"));
    CHECK(ProfileInt(profile, "prep_jobs") == commits);
    CHECK(ProfileInt(profile, "prep_max_depth") >= 1);
    CHECK(ProfileInt(profile, "prep_max_depth") <= 2);          // 深度上限硬约束
    CHECK(ProfileInt(profile, "live_bytes") >= 0);
    CHECK(uint64_t(ProfileInt(profile, "live_bytes")) <= quota); // 预算不超
    CHECK(profile.find("\"quota_check_ms\"") != std::string::npos);
    // 热路径口径：drain=false 也必须报告同一批已收尾的计数。
    const std::string hot = store->Profile(false);
    CHECK(ProfileInt(hot, "live_checkpoints") == ProfileInt(profile, "live_checkpoints"));

    // ④ 在物理配额足够时，所有已受理作业都必须可读；此用例的配额断言不替代 FIFO 淘汰用例。
    wd.Step(32, "verify accepted jobs remain loadable");
    CHECK(store->LiveCheckpoints() == uint64_t(commits));
    CHECK(store->LiveBytes() <= quota);
    for (int i = 0; i < commits; ++i) {
        std::string identityOut;
        std::vector<std::string> loaded;
        std::string loadError;
        const bool visible = store->Load(fixtures[i].key, identityOut, loaded, loadError);
        std::fprintf(stderr, "[budget] key %d visible=%d error=%s\n", i, int(visible),
                     loadError.c_str());
        CHECK(visible);
        CHECK(LoadExact(*store, fixtures[i], loadError));
        CHECK(loadError.empty());
    }
    // ⑤ 清单完整性（P0 历史缺陷的回归护栏）：被保留的检查点缺失条目必须为 0。
    std::string details;
    const size_t missing = store->SelfCheckManifest(details);
    std::fprintf(stderr, "[budget] selfcheck-manifest missing=%zu %s\n", missing, details.c_str());
    CHECK(missing == 0);
}

// ===========================================================================
// 用例 2：预算「刚好够 1 笔」——仍有完成窗口
// 计划条款的正题：quota 贴到只够一笔（incoming <= quota < total+incoming），
// 每次提交都先把上一笔退掉才能写。断言每笔都 Reap 成功且立刻能逐字节取回，
// 即「贴下限仍有完成窗口」，而不是"卡在等空间/直接失败"。
// ===========================================================================
static void budgetFloorStillHasCompletionWindow(const fs::path &root) {
    Watchdog wd("budget-floor-window", 45);
    const std::string identity = DiskPrefixCache::Digest("floor-identity");
    const std::string family = DiskPrefixCache::Digest("floor-family");
    const size_t chunkBytes = 1u << 20;
    const int chunksPer = 2;
    const size_t padBytes = 256u << 10;
    const int rounds = 4;

    const auto probe = MakeStagedFixture(0, identity, family, chunkBytes, chunksPer, padBytes);
    const Budget budget = Measure(probe, chunkBytes, chunksPer);
    const uint64_t onePhysical = MeasurePhysical(root / "sizing", probe, family);
    const uint64_t quota = onePhysical + (64ull << 10);
    // The extra headroom admits one aligned ERASE record, while the next full
    // checkpoint still requires Compact before it can be retried.
    std::fprintf(stderr, "[budget] case2 total=%llu one_physical=%llu quota=%llu (MiB: %.2f)\n",
                 (unsigned long long)budget.total, (unsigned long long)onePhysical,
                 (unsigned long long)quota, double(quota) / 1048576.0);
    CHECK(onePhysical > budget.total);

    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "open: " << error << "\n"; return; }

    int explicitRetries = 0;
    for (int i = 0; i < rounds; ++i) {
        auto fixture = MakeStagedFixture(i, identity, family, chunkBytes, chunksPer, padBytes);
        wd.Step(1 + i * 3, "SubmitStaged at quota floor");
        uint64_t sequence = 0;
        const bool first = SubmitAndReap(*store, fixture, family, &sequence, error);
        if (!first) {
            std::fprintf(stderr, "[floor] first attempt %d: %s\n", i, error.c_str());
            CHECK(error == "capacity_unavailable");
            CHECK(store->PhysicalBytes() <= quota);
            CHECK(store->Compact(error));
            ++explicitRetries;
            fixture = MakeStagedFixture(i, identity, family, chunkBytes, chunksPer, padBytes);
            wd.Step(2 + i * 3, "Compact then explicitly retry");
            CHECK(SubmitAndReap(*store, fixture, family, &sequence, error));
        }
        CHECK(sequence > 0);
        CHECK(store->LiveCheckpoints() == 1);
        CHECK(store->LiveBytes() <= quota);
        std::string loadError;
        CHECK(LoadExact(*store, fixture, loadError));
        CHECK(loadError.empty());
    }
    CHECK(explicitRetries > 0);
}

// ===========================================================================
// 用例 3：预算连一笔都不够 —— 必须「明确拒绝并报告」，绝不伪造成功
// 反例不能停在"没崩"：要逐条确认
//   (a) 受理后 Reap 明确失败，错误串 = capacity_unavailable；
//   (b) 库里真的没有这个检查点（LiveCheckpoints==0），Load 也失败且带错误串；
//   (c) 拒绝之后 store 未被带坏：一笔装得下的小检查点仍能提交并取回；
//   (d) 同步入口 CommitCheckpoint 同样明确失败（不静默返回 true）。
// ===========================================================================
static void hardRejectIsVisibleNeverFakeSuccess(const fs::path &root) {
    Watchdog wd("hard-reject-visible", 30);
    const std::string identity = DiskPrefixCache::Digest("reject-identity");
    const std::string family = DiskPrefixCache::Digest("reject-family");
    const uint64_t quota = 64ull << 10;              // 64 KiB，远小于一笔
    const auto fixture = MakeStagedFixture(0, identity, family, 1u << 20, 1, 0);
    const Budget budget = Measure(fixture, 1u << 20, 1);

    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "open: " << error << "\n"; return; }

    const uint64_t onePhysical = MeasurePhysical(root / "sizing", fixture, family);
    CHECK(onePhysical > quota);                     // 前提：连一笔都不够
    CHECK(onePhysical > budget.total);
    // (a)+(b) 受理 ≠ 成功：Reap 必须把拒绝原样报出来。
    wd.Step(1, "SubmitStaged a checkpoint larger than the whole budget");
    error.clear();
    auto copy = fixture;
    const bool accepted = store->SubmitStaged(copy.key, copy.identity, family, copy.state, false,
                                              std::move(copy.staged), error);
    CHECK(accepted);
    wd.Step(2, "ReapCheckpoint must fail visibly with capacity_unavailable");
    error.clear();
    uint64_t sequence = 0;
    const bool reaped = store->ReapCheckpoint(&sequence, error);
    std::fprintf(stderr, "[reject] reap=%d error=%s\n", int(reaped), error.c_str());
    CHECK(!reaped);
    CHECK(!error.empty());
    CHECK(error == "capacity_unavailable");
    // Check before Compact: a refused checkpoint must not rescue-flush its
    // unreserved payload and exceed the physical quota even temporarily.
    CHECK(store->PhysicalBytes() <= quota);
    CHECK(store->LiveCheckpoints() == 0);
    CHECK(store->Checkpoints().empty());
    {
        std::string identityOut;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(!store->Load(fixture.key, identityOut, loaded, loadError));
        CHECK(!loadError.empty());          // 读不到必须带原因，不许静默空成功
    }

    // (c) 拒绝不得把 store 带坏（无中毒/无死锁）：装得下的小检查点仍能提交并取回。
    wd.Step(3, "Compact then a checkpoint that does fit must still commit and load");
    CHECK(store->Compact(error));
    CHECK(store->PhysicalBytes() <= quota);
    const auto small = MakeObject(std::string(16u << 10, 'S'), 1);
    const std::string smallMeta = "{\"state\":\"" + DiskPrefixCache::Digest("reject-small") + "\"}";
    error.clear();
    CHECK(store->CommitCheckpoint("reject-small-key", identity, {small}, smallMeta, &sequence, error));
    CHECK(error.empty());
    {
        std::string identityOut;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(store->Load("reject-small-key", identityOut, loaded, loadError));
        CHECK(loaded.size() == 1);
        CHECK(loaded.size() == 1 && loaded[0] == small.bytes);
    }

    // (d) 同步入口同样明确失败（不能出现"返回 true 但没落盘"）。
    wd.Step(4, "synchronous CommitCheckpoint must fail, not fake success");
    const auto tooBig = MakeObject(std::string(1u << 20, 'T'), 1);
    error.clear();
    const bool syncOk = store->CommitCheckpoint("reject-sync-key", identity, {tooBig}, smallMeta,
                                                &sequence, error);
    std::fprintf(stderr, "[reject] sync commit=%d error=%s\n", int(syncOk), error.c_str());
    CHECK(!syncOk);
    CHECK(error == "capacity_unavailable");
    CHECK(store->PhysicalBytes() <= quota);
}

// Four 4 MiB payloads cross the writer's 8 MiB flush threshold twice. The
// synthetic root needs a third OBJECT_BATCH. Admission must account for all
// three aligned records, not just group objects by the 15 MiB payload cap.
static void physicalBatchBoundary(const fs::path &root) {
    Watchdog wd("physical-batch-boundary", 45);
    const auto identity = DiskPrefixCache::Digest("batch-boundary-id");
    std::vector<DiskPrefixV4Store::Object> objects;
    for (int i = 0; i < 4; ++i)
        objects.push_back(MakeObject(std::string(4u << 20, char('a' + i)), 1));
    const std::string metadata = "{\"state\":\"" + objects.front().hash + "\"}";
    std::string error;
    uint64_t physical = 0, sequence = 0;
    {
        auto store = DiskPrefixV4Store::Open((root / "sizing").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->CommitCheckpoint("batch-root", identity,
                                      std::vector<DiskPrefixV4Store::Object>(objects),
                                      metadata, &sequence, error));
        physical = store->PhysicalBytes(); // Actual writer boundaries, before Compact.
    }
    CHECK(physical > 16ull << 20);
    {
        const uint64_t quota = physical + 4096; // One ERASE completion window.
        auto store = DiskPrefixV4Store::Open((root / "fits").string(), quota, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->CommitCheckpoint("batch-root", identity,
                                      std::vector<DiskPrefixV4Store::Object>(objects),
                                      metadata, &sequence, error));
        CHECK(store->PhysicalBytes() == physical);
        CHECK(store->PhysicalBytes() <= quota);
        std::vector<std::string> loaded;
        std::string loadedIdentity;
        CHECK(store->Load("batch-root", loadedIdentity, loaded, error));
        CHECK(loadedIdentity == identity && loaded.size() == objects.size());
        for (size_t i = 0; i < loaded.size() && i < objects.size(); ++i)
            CHECK(loaded[i] == objects[i].bytes);
    }
    {
        const uint64_t quota = physical - 4096;
        auto store = DiskPrefixV4Store::Open((root / "refused").string(), quota, error);
        CHECK(store != nullptr);
        if (!store) return;
        const uint64_t before = store->PhysicalBytes();
        CHECK(!store->CommitCheckpoint("batch-root", identity,
                                       std::vector<DiskPrefixV4Store::Object>(objects),
                                       metadata, &sequence, error));
        CHECK(error == "capacity_unavailable");
        CHECK(store->LiveCheckpoints() == 0);
        CHECK(store->PhysicalBytes() == before);
        CHECK(store->PhysicalBytes() <= quota);
    }
}

// ===========================================================================
// 用例 4：背压路径上的失败必须「点名上报」，不许静默
// 备菜 FIFO 满(2)时受理第三笔，会先等队首；若队首失败，第三笔的受理调用必须
// 返回 false 并把队首的错误串原样带出（disk_prefix_cache.cpp:2288-2310 注释即此
// 语义）。这是「拒绝可见」的另一条真实路径，且同样是死锁高发点。
// ===========================================================================
static void pipelineReportsPriorFailureAtAdmission(const fs::path &root) {
    Watchdog wd("pipeline-failure-report", 30);
    const std::string identity = DiskPrefixCache::Digest("fail-id");
    const std::string goodMeta = "{\"state\":\"" + DiskPrefixCache::Digest("fail-good") + "\"}";
    const auto object = MakeObject(std::string(64u << 10, 'F'), 1);
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;

    // 第一笔：空 metadata → 提交阶段必失败（v4_checkpoint_metadata）。
    wd.Step(1, "submit a job that will fail at commit (empty metadata)");
    error.clear();
    CHECK(store->SubmitCheckpoint("fail-bad-key", identity, {object}, "", error));
    // 第二笔：占满深度 2。
    wd.Step(2, "submit a good job to fill depth 2");
    error.clear();
    CHECK(store->SubmitCheckpoint("fail-good-key", identity, {object}, goodMeta, error));
    // 第三笔：队列满 → 等队首 → 队首失败 → 受理必须可见地失败。
    wd.Step(3, "third submit must surface the head failure, not hang or lie");
    error.clear();
    const bool third = store->SubmitCheckpoint("fail-third-key", identity, {object}, goodMeta, error);
    std::fprintf(stderr, "[backpressure] third submit=%d error=%s\n", int(third), error.c_str());
    CHECK(!third);
    CHECK(!error.empty());
    CHECK(error == "v4_checkpoint_metadata");
    // 第三笔既没被受理，就必须查不到（不许"没受理却像成功"）。
    {
        std::string identityOut;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(!store->Load("fail-third-key", identityOut, loaded, loadError));
        CHECK(!loadError.empty());
    }
    // 已在队列里的好那笔必须完成并可读（背压不丢已受理的工作）。
    wd.Step(4, "the accepted good job must still commit and load");
    error.clear();
    uint64_t sequence = 0;
    const bool reaped = store->ReapCheckpoint(&sequence, error);
    // 队首失败已被上一笔受理吃掉并出队，这里应只收尾好那笔。
    CHECK(reaped);
    {
        std::string identityOut;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(store->Load("fail-good-key", identityOut, loaded, loadError));
        CHECK(loaded.size() == 1);
        CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    }
}

int main() {
    const fs::path tmp = TempRoot();
    std::cout << "hostlimit test root: " << tmp << "\n";
    try {
        hostPipelineDepthTwoNoDeadlock(tmp / "pipeline");
        budgetFloorStillHasCompletionWindow(tmp / "floor");
        hardRejectIsVisibleNeverFakeSuccess(tmp / "reject");
        physicalBatchBoundary(tmp / "batch-boundary");
        pipelineReportsPriorFailureAtAdmission(tmp / "backpressure");
    } catch (const std::exception &error) {
        std::cerr << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "FAILED: " << failures << " check(s)\n";
        return 1;
    }
    std::cout << "PASS disk_prefix_hostlimit\n";
    return 0;
}
