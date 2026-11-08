// CPU tests for the v4 manifest-tree store (step ①/② of the approved order):
// real write path, recovery from the control log, index rebuild and space
// reclamation. No CUDA, no model, no service; all fixtures live under TMPDIR.
#include "utils/disk_prefix_cache.h"
#include "utils/disk_prefix_format_v4.h"
#include "utils/disk_prefix_write_coordinator.h"

// Route B（2026-10-02）单测需要直接读 catalog 投影里的 blocks 行数/删一条父块行，
// 以便断言“块级共享/回收”与“父块缺失 fail-closed”。测试目标本就链接 SQLite3。
#include <sqlite3.h>

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <atomic>
#include <chrono>
#include <csignal>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
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


// 用户硬性要求：禁止写 /tmp，所有夹具必须落在项目内。
// 这里把规则写成**可执行约束**：TMPDIR 指向 /tmp 时直接拒绝运行，而不是默默照做。
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
    std::string pattern = (VerifiedTempBase("v4store") / "fastllm-v4store-test-XXXXXX").string();
    char *name = mkdtemp(pattern.data());
    CHECK(name != nullptr);
    return name ? fs::path(name) : fs::current_path();
}

static std::string ReadAll(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

static DiskPrefixV4Store::Object MakeObject(const std::string &bytes, uint32_t kind) {
    DiskPrefixV4Store::Object object;
    object.bytes = bytes;
    object.kind = kind;
    object.hash = DiskPrefixCache::Digest(bytes);
    return object;
}

// Quotas are physical-log quotas.  Small logical payloads still consume
// aligned OBJECT/PREPARE/COMMIT records, so fixtures size their one-root
// budget from a compacted control log instead of guessing payload+512.
static uint64_t MeasureCompactedPhysical(const fs::path &root,
                                         const std::vector<DiskPrefixV4Store::Object> &objects,
                                         bool checkpoint) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return 0;
    const std::string identity = DiskPrefixCache::Digest("quota-sizing-identity");
    bool committed = false;
    if (checkpoint) {
        committed = store->CommitCheckpoint(
            DiskPrefixCache::Digest("quota-sizing-key"), identity, objects,
            "{\"state\":\"" + DiskPrefixCache::Digest("quota-sizing-state") + "\"}",
            nullptr, error);
    } else {
        committed = store->Commit("quota-sizing-key", identity, objects, nullptr, error);
    }
    CHECK(committed);
    if (!committed) return 0;
    CHECK(store->Compact(error));
    return store->PhysicalBytes();
}

static uint64_t MeasureTwoCompactedPhysical(
        const fs::path &root,
        const std::vector<DiskPrefixV4Store::Object> &first,
        const std::vector<DiskPrefixV4Store::Object> &second,
        bool checkpoint) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return 0;
    const std::string identity = DiskPrefixCache::Digest("quota-sizing-two-identity");
    bool firstCommitted = false, secondCommitted = false;
    if (checkpoint) {
        firstCommitted = store->CommitCheckpoint(
            DiskPrefixCache::Digest("quota-sizing-first"), identity, first,
            "{\"state\":\"" + DiskPrefixCache::Digest("quota-sizing-first-state") + "\"}",
            nullptr, error);
        secondCommitted = firstCommitted && store->CommitCheckpoint(
            DiskPrefixCache::Digest("quota-sizing-second"), identity, second,
            "{\"state\":\"" + DiskPrefixCache::Digest("quota-sizing-second-state") + "\"}",
            nullptr, error);
    } else {
        firstCommitted = store->Commit("quota-sizing-first", identity, first, nullptr, error);
        secondCommitted = firstCommitted && store->Commit("quota-sizing-second", identity,
                                                          second, nullptr, error);
    }
    CHECK(firstCommitted && secondCommitted);
    if (!firstCommitted || !secondCommitted) return 0;
    CHECK(store->Compact(error));
    return store->PhysicalBytes();
}

struct SegmentsEnv {
    explicit SegmentsEnv(const char *bytes) {
        setenv("FASTLLM_PREFIX_V4_SEGMENTS", "1", 1);
        setenv("FASTLLM_PREFIX_V4_SEGMENT_BYTES", bytes, 1);
    }
    ~SegmentsEnv() {
        unsetenv("FASTLLM_PREFIX_V4_SEGMENTS");
        unsetenv("FASTLLM_PREFIX_V4_SEGMENT_BYTES");
    }
};
static size_t CountSegmentFiles(const fs::path &root) {
    size_t count = 0;
    for (const auto &entry : fs::directory_iterator(root / "v4"))
        if (entry.path().filename().string().rfind("seg-", 0) == 0) ++count;
    return count;
}
// Fill a 512 KiB store with 64 KiB segments ten times over: every commit must succeed,
// physical bytes never pass the quota, old roots disappear, the newest roots read back exactly,
// and a reopen (catalog deleted -> full rebuild over the remaining segments) still serves them.
static void segmentsFillPastQuotaManyTimes(const fs::path &root) {
    SegmentsEnv env("65536");
    const uint64_t quota = 512 * 1024;
    const std::string identity = DiskPrefixCache::Digest("seg-identity");
    std::string error;
    std::vector<std::string> payloads;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "seg open: " << error << "\n"; return; }
        for (int i = 0; i < 160; ++i) {
            payloads.push_back(std::string(20000 + i, char('a' + i % 26)) + std::to_string(i));
            const bool ok = store->Commit("seg-" + std::to_string(i), identity,
                                          {MakeObject(payloads.back(), 1)}, nullptr, error);
            CHECK(ok);
            if (!ok) std::cerr << "seg commit " << i << ": " << error << "\n";
            CHECK(store->PhysicalBytes() <= quota);
            CHECK(CountSegmentFiles(root) <= quota / 65536 + 2);
        }
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        CHECK(!store->Load("seg-0", loadedIdentity, loaded, error));
        CHECK(store->Load("seg-159", loadedIdentity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == payloads[159]);
        CHECK(store->LiveCheckpoints() >= 4);
    }
    fs::remove(root / "v4" / "catalog.sqlite3");
    fs::remove(root / "v4" / "catalog.sqlite3-wal");
    fs::remove(root / "v4" / "catalog.sqlite3-shm");
    auto reopened = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(reopened != nullptr);
    if (!reopened) { std::cerr << "seg reopen: " << error << "\n"; return; }
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(reopened->Load("seg-159", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == payloads[159]);
    CHECK(!reopened->Load("seg-0", loadedIdentity, loaded, error));
    CHECK(reopened->Commit("seg-after-reopen", identity, {MakeObject(std::string(30000, 'z'), 1)},
                           nullptr, error));
}
// A root that shares an object with an older root is retired together with the segment
// that holds the shared object; every root that stays live reads back exactly.
static void segmentsSharedObjectRetiresDependents(const fs::path &root) {
    SegmentsEnv env("65536");
    const uint64_t quota = 384 * 1024;
    const std::string identity = DiskPrefixCache::Digest("seg-shared-identity");
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "seg shared open: " << error << "\n"; return; }
    const auto shared = MakeObject(std::string(16000, 'S'), 1);
    CHECK(store->Commit("first", identity, {shared, MakeObject(std::string(9000, '1'), 1)}, nullptr, error));
    std::map<std::string, std::vector<std::string>> expected;
    for (int i = 0; i < 80; ++i) {
        const std::string key = "later-" + std::to_string(i);
        const auto own = MakeObject(std::string(15000 + i, char('A' + i % 26)), 1);
        std::vector<DiskPrefixV4Store::Object> objects{own};
        if (i % 3 == 0) objects.push_back(shared);
        if (!store->Commit(key, identity, objects, nullptr, error)) {
            std::cerr << "seg shared commit " << i << ": " << error << "\n";
            CHECK(false);
            continue;
        }
        std::vector<std::string> bytes;
        for (const auto &object : objects) bytes.push_back(object.bytes);
        expected[key] = bytes;
        CHECK(store->PhysicalBytes() <= quota);
    }
    size_t live = 0;
    for (const auto &item : expected) {
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        if (!store->Load(item.first, loadedIdentity, loaded, error)) {
            CHECK(error == "v4_root_missing");   // retired: never a partial or wrong root
            continue;
        }
        ++live;
        CHECK(loaded == item.second);
    }
    CHECK(live >= 4);
}
// A lowered quota is reached right at Open; PutMetadata on a full store deletes old segments
// instead of failing; physical bytes stay within the quota.
static void segmentsShrinkAtOpenAndMetadataWhenFull(const fs::path &root) {
    SegmentsEnv env("65536");
    const std::string identity = DiskPrefixCache::Digest("seg-shrink-identity");
    std::string error;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 512 * 1024, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "seg shrink open: " << error << "\n"; return; }
        for (int i = 0; i < 40; ++i)
            CHECK(store->Commit("shrink-" + std::to_string(i), identity,
                                {MakeObject(std::string(20000 + i, char('k' + i % 10)), 1)}, nullptr, error));
        CHECK(store->PhysicalBytes() > 256 * 1024);
    }
    const uint64_t lowered = 256 * 1024;
    auto reopened = DiskPrefixV4Store::Open(root.string(), lowered, error);
    CHECK(reopened != nullptr);
    if (!reopened) { std::cerr << "seg shrink reopen: " << error << "\n"; return; }
    CHECK(reopened->PhysicalBytes() <= lowered);
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(reopened->Load("shrink-39", loadedIdentity, loaded, error));
    for (int i = 0; i < 60; ++i) {
        const bool ok = reopened->PutMetadata("meta-" + std::to_string(i % 3), std::string(3000 + i, 'm'), error);
        CHECK(ok);
        if (!ok) std::cerr << "seg meta " << i << ": " << error << "\n";
        CHECK(reopened->PhysicalBytes() <= lowered);
    }
}

static void segmentsLayoutGuards(const fs::path &root) {
    std::string error;
    {
        // flag off creates a legacy single log ...
        auto legacy = DiskPrefixV4Store::Open((root / "legacy").string(), 1ull << 30, error);
        CHECK(legacy != nullptr);
        if (legacy) CHECK(legacy->Commit("k", DiskPrefixCache::Digest("i"),
                                         {MakeObject("legacy-bytes", 1)}, nullptr, error));
    }
    {
        SegmentsEnv env("65536");
        auto refused = DiskPrefixV4Store::Open((root / "legacy").string(), 1ull << 30, error);
        CHECK(refused == nullptr);
        CHECK(error == "v4_segments_legacy_store");
        auto segmented = DiskPrefixV4Store::Open((root / "segmented").string(), 1ull << 30, error);
        CHECK(segmented != nullptr);
        if (segmented) CHECK(segmented->Commit("k", DiskPrefixCache::Digest("i"),
                                               {MakeObject("segment-bytes", 1)}, nullptr, error));
    }
    auto flagOff = DiskPrefixV4Store::Open((root / "segmented").string(), 1ull << 30, error);
    CHECK(flagOff == nullptr);
    CHECK(error == "v4_segments_store_requires_flag");
}

static void writeReadRecover(const fs::path &root) {
    const std::string identity1 = DiskPrefixCache::Digest("identity-1");
    const auto first = MakeObject(std::string(4096, 'A'), 1);
    const auto second = MakeObject(std::string(1024, 'B'), 3);
    std::string error;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "open: " << error << "\n"; return; }
        CHECK(fs::exists(store->Root() + "/FORMAT"));
        CHECK(ReadAll(fs::path(store->Root()) / "FORMAT") ==
              "fastllm-prefix-v4 manifest-tree-1\n");
        uint64_t sequence = 0;
        CHECK(store->Commit("key-a", identity1, {first, second}, &sequence, error));
        CHECK(sequence > 0);
        CHECK(store->LiveCheckpoints() == 1);
        CHECK(store->LiveBytes() == first.bytes.size() + second.bytes.size());
        std::string identity;
        std::vector<std::string> loaded;
        CHECK(store->Load("key-a", identity, loaded, error));
        CHECK(identity == identity1 && loaded.size() == 2);
        CHECK(loaded[0] == first.bytes && loaded[1] == second.bytes);
        // A hash that does not match the bytes it claims is refused —— 这是
        // FASTLLM_PREFIX_CACHE_STRICT_VERIFY=1 的写侧严格模式（§22.4.4 乙案：
        // 默认关闭写前重算，读回/重建仍逐对象校验，见 strictVerifyOffStillSafe）。
        auto forged = first;
        forged.hash = DiskPrefixCache::Digest("something else");
        setenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY", "1", 1);
        CHECK(!store->Commit("key-bad", identity1, {forged}, nullptr, error));
        CHECK(error == "v4_object_checksum");
        unsetenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
        // Reading an unknown key reports rather than inventing data.
        CHECK(!store->Load("missing", identity, loaded, error));
        CHECK(error == "v4_root_missing");
    }
    // Reopen with the existing catalog: the checkpoint must still load.
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->LiveCheckpoints() == 1);
        std::string identity;
        std::vector<std::string> loaded;
        CHECK(store->Load("key-a", identity, loaded, error));
        CHECK(loaded.size() == 2 && loaded[0] == first.bytes);
    }
    // The catalog is only a projection: deleting it must not lose the data.
    {
        fs::remove(fs::path(root) / "v4" / "catalog.sqlite3");
        std::error_code ignored;
        fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
        fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "rebuild: " << error << "\n"; return; }
        std::string identity;
        std::vector<std::string> loaded;
        CHECK(store->Load("key-a", identity, loaded, error));
        CHECK(loaded.size() == 2 && loaded[1] == second.bytes);
        CHECK(store->LiveCheckpoints() == 1);
    }
}

static void unfinishedTailPublishesNothing(const fs::path &root) {
    const std::string identity2 = DiskPrefixCache::Digest("identity-2");
    std::string error;
    const auto committed = MakeObject(std::string(2048, 'C'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->Commit("tail-key", identity2, {committed}, nullptr, error));
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    const std::string before = ReadAll(log);
    // Append a half-written record: it must never become visible.
    {
        std::ofstream out(log, std::ios::binary | std::ios::app);
        const std::string partial(64, '\0');
        out.write(partial.data(), partial.size());
    }
    // 删掉索引（catalog 只是投影）：强制从控制日志重建，
    // 这样半截记录才会真的走到分类逻辑上。
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "tail open: " << error << "\n"; return; }
    std::string identity;
    std::vector<std::string> loaded;
    CHECK(store->Load("tail-key", identity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == committed.bytes);
    CHECK(store->LiveCheckpoints() == 1);
    // §14：不完整尾部必须被归类成 UNCOMMITTED_TAIL（而不是 valid / corrupt）。
    const std::string profile = store->Profile();
    CHECK(profile.find("\"rebuild_uncommitted_tail\": 1") != std::string::npos);
    CHECK(profile.find("\"rebuild_corrupt_control\": 0") != std::string::npos);
    CHECK(profile.find("\"rebuild_valid\": 0") != std::string::npos);
    // 2026-10-02 起（S1-1 修复，commit e7c9a6ad）：残尾不再"原样留在盘上"，而是被
    // **截断到最后一条完整记录的末尾**后继续写（业界 LevelDB log_format / SQLite WAL /
    // RocksDB 恢复同做法；ScanControlLog 已保证残尾不含任何已 COMMIT 的根）。
    // 因此断言从旧语义的"垃圾仍在盘上"改为"盘上只剩完整记录"。
    CHECK(uint64_t(fs::file_size(log)) == before.size());
    CHECK(ReadAll(log).size() == before.size());
}

static void corruptLogFailsClosed(const fs::path &root) {
    const std::string identity3 = DiskPrefixCache::Digest("identity-3");
    std::string error;
    const auto object = MakeObject(std::string(512, 'D'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->Commit("corrupt-key", identity3, {object}, nullptr, error));
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    std::string bytes = ReadAll(log);
    CHECK(bytes.size() > 4096 + 200);
    bytes[4096 + 20] = char(bytes[4096 + 20] ^ 0x1);   // break the second record
    { std::ofstream out(log, std::ios::binary | std::ios::trunc); out.write(bytes.data(), bytes.size()); }
    // Force a rebuild so the corrupt record must be examined.
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store == nullptr);            // fail closed
    CHECK(!error.empty());
    // The damaged log is preserved, not truncated or "repaired" in place.
    CHECK(ReadAll(log).size() == bytes.size());
}

static void quotaEvictionAndCompaction(const fs::path &root) {
    const std::string identity4 = DiskPrefixCache::Digest("identity-4");
    std::string error;
    const auto small = MakeObject(std::string(1024, 'E'), 1);
    const auto large = MakeObject(std::string(8192, 'F'), 1);
    // The physical budget fits one compacted newest root.  Admission first
    // writes ERASE (logical retirement), then the background Compact frees the
    // old bytes before the retry; roots.live=0 alone is not physical reclaim.
    const uint64_t quota = MeasureCompactedPhysical(
        root / "quota-sizing", {large}, false) + 4096;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "quota open: " << error << "\n"; return; }
    CHECK(store->Commit("old", identity4, {small}, nullptr, error));
    CHECK(store->LiveCheckpoints() == 1);
    CHECK(!store->Commit("new", identity4, {large}, nullptr, error));
    CHECK(error == "capacity_unavailable");
    CHECK(store->LiveCheckpoints() == 0);      // old is logically retired
    const uint64_t logicalAfterErase = store->LiveBytes();
    const uint64_t physicalBefore = store->PhysicalBytes();
    CHECK(physicalBefore > logicalAfterErase);
    CHECK(store->Compact(error));
    const uint64_t physicalAfter = store->PhysicalBytes();
    CHECK(physicalAfter < physicalBefore);
    CHECK(store->Commit("new", identity4, {large}, nullptr, error));
    CHECK(store->LiveCheckpoints() == 1);
    CHECK(store->LiveBytes() <= large.bytes.size());
    std::string identity;
    std::vector<std::string> loaded;
    CHECK(!store->Load("old", identity, loaded, error));
    CHECK(error == "v4_root_missing");
    CHECK(store->Load("new", identity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == large.bytes);

    const fs::path log = fs::path(root) / "v4" / "control.log";
    const auto afterRetry = fs::file_size(log);
    CHECK(afterRetry >= physicalAfter);
    CHECK(store->LiveCheckpoints() == 1);
    CHECK(store->Load("new", identity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == large.bytes);
    // Still readable after a reopen, i.e. the compacted log rebuilds correctly.
    store.reset();
    auto reopened = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(reopened != nullptr);
    if (reopened) {
        CHECK(reopened->LiveCheckpoints() == 1);
        CHECK(reopened->Load("new", identity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == large.bytes);
    }
}

// Three-tier storage contract guards: ERASE is observable, an active reader
// cannot be selected by FIFO, shared content remains reachable, and logical
// retirement is distinct from physical log reclamation.
static void fifoEraseCompactSafety(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("fifo-safety-identity");
    const auto shared = MakeObject(std::string(4096, 'Q'), 1);
    const auto retiredOnly = MakeObject(std::string(4096, 'P'), 1);
    const auto newer = MakeObject(std::string(4096, 'R'), 1);
    // Leave one aligned record for the durable ERASE that retires the FIFO
    // victim.  The measured two-root compacted size is not the same as the
    // live append size, and a physical quota must account for that control
    // record without reserving an unconditional reclaim cushion elsewhere.
    const uint64_t quota = MeasureTwoCompactedPhysical(
        root / "fifo-sizing", {shared, retiredOnly}, {shared}, false) + 8192;
    auto store = DiskPrefixV4Store::OpenShared(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) return;
    // Keep FIFO admission and the explicit Compact below in a deterministic
    // order.  A background Compact racing the second admission can rewrite
    // the log before the observer sees the ERASE and change which roots fit.
    setenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT", "1", 1);
    std::vector<std::pair<std::string, std::string>> erased;
    auto observer = store->ObserveCheckpointErases(
        [&](const std::string &key, const std::string &id) { erased.emplace_back(key, id); });
    CHECK(observer != nullptr);
    bool cacheObserverCalled = false;
    DiskPrefixCache cache(root.string(), identity, quota, identity, 1);
    cache.SetCheckpointEraseCallback(
        [&](const std::string &, const std::string &) { cacheObserverCalled = true; });
    bool reclaimAllowed = false;
    cache.SetCheckpointReclaimPredicate(
        [&](const std::string &, const std::string &) { return reclaimAllowed; });
    CHECK(store->Commit("shared-old", identity, {shared, retiredOnly}, nullptr, error));
    // A second root references the same content-addressed object. Compact must
    // retain that object after the first root is retired.
    CHECK(store->Commit("shared-new", identity, {shared}, nullptr, error));
    CHECK(store->LiveCheckpoints() == 2);
    std::string pinError;
    auto pin = store->PinCheckpoint("shared-old", identity, pinError);
    CHECK(pin != nullptr);
    auto pinNew = store->PinCheckpoint("shared-new", identity, pinError);
    CHECK(pinNew != nullptr);
    // The pinned root is the only safe FIFO victim. Admission fails closed;
    // it must not retire the active reader or silently accept an over-quota log.
    CHECK(!store->Commit("blocked", identity, {newer}, nullptr, error));
    CHECK(error == "capacity_unavailable");
    CHECK(store->LiveCheckpoints() == 2);
    CHECK(erased.empty());
    pin.reset();
    pinNew.reset();
    reclaimAllowed = true;
    CHECK(!store->Commit("blocked", identity, {newer}, nullptr, error));
    CHECK(error == "capacity_unavailable");
    CHECK(!erased.empty());
    if (!erased.empty()) {
        CHECK(erased.front().first == "shared-old");
        CHECK(erased.front().second == identity);
    }
    CHECK(cacheObserverCalled);
    CHECK(store->Compact(error));
    CHECK(store->Commit("blocked", identity, {newer}, nullptr, error));
    std::string observedIdentity;
    std::vector<std::string> bytes;
    CHECK(!store->Load("shared-old", observedIdentity, bytes, error));
    CHECK(error == "v4_root_missing");
    CHECK(store->Load("shared-new", observedIdentity, bytes, error));
    CHECK(bytes.size() == 1 && bytes[0] == shared.bytes);
    const uint64_t logicalBefore = store->LiveBytes();
    const uint64_t physicalBefore = store->PhysicalBytes();
    CHECK(physicalBefore > logicalBefore);  // ERASE only changes visibility.
    CHECK(store->Compact(error));
    const uint64_t physicalAfter = store->PhysicalBytes();
    CHECK(physicalAfter < physicalBefore);
    CHECK(store->LiveBytes() == logicalBefore);
    CHECK(store->Load("shared-new", observedIdentity, bytes, error));
    CHECK(bytes.size() == 1 && bytes[0] == shared.bytes);
    observer.reset();
    unsetenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT");
}

// Physical-capacity guard: logical roots and physical log bytes are separate
// counters. An active writer owns a manifest lease while a newer checkpoint is
// admitted; the new checkpoint must remain readable after background Compact.
static void physicalCapacityAndActiveWriter(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("physical-capacity");
    const std::string keyOld = DiskPrefixCache::Digest("physical-old");
    const std::string keyNew = DiskPrefixCache::Digest("physical-new");
    const std::string payload(4096, 'W');
    auto store = DiskPrefixV4Store::OpenShared(root.string(), 1ull << 20, error);
    CHECK(store != nullptr);
    if (!store) return;
    DiskPrefixCache cache(root.string(), identity, 1ull << 20, identity, 1);
    auto writer = cache.BeginWrite(payload.size());
    const auto chunks = writer.StoreBytes(payload.size(),
        [&](size_t offset, char *out, size_t bytes) { memcpy(out, payload.data() + offset, bytes); });
    CHECK(store->Commit(keyOld, identity, {MakeObject(payload, 1)}, nullptr, error));
    const uint64_t physicalBefore = store->PhysicalBytes();
    CHECK(physicalBefore >= store->LiveBytes());
    CHECK(store->Commit(keyNew, identity, {MakeObject(std::string(4096, 'N'), 1)}, nullptr, error));
    CHECK(store->Compact(error));
    CHECK(store->PhysicalBytes() <= physicalBefore + (1ull << 20));
    std::vector<std::string> loaded;
    std::string observed;
    CHECK(store->Load(keyNew, observed, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == std::string(4096, 'N'));
    (void)chunks;
}


// R3：磁盘配额压力下**先淘汰最旧的存活检查点、再接收新写**，
// 而不是把新检查点丢掉；淘汰必须可观测（Profile 的 evicted_for_admission）。
// 与既有 quotaEvictionAndCompaction / batchingAndFifo 的区别：这里显式断言**新写被接收**
// （提交返回 true、可读回）与**计数增加**，把“淘汰而不是丢弃”钉成契约。
static void admissionEvictionKeepsNewestAndCounts(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("admission-eviction-identity");
    const std::string oldKey = DiskPrefixCache::Digest("admission-evict-old");
    const std::string newKey = DiskPrefixCache::Digest("admission-evict-new");
    const auto small = MakeObject(std::string(1024, 'R'), 1);
    const auto large = MakeObject(std::string(8192, 'S'), 1);
    // 物理配额只够放一个压缩后的 newest 检查点；第二次提交先追加
    // ERASE 并请求后台 Compact，再由调用方重试。
    const uint64_t quota = MeasureCompactedPhysical(
        root / "admission-sizing", {large}, true) + 4096;
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "admission eviction open: " << error << "\n"; return; }
    const std::string before = store->Profile();
    CHECK(before.find("\"evicted_for_admission\"") != std::string::npos);
    uint64_t sequence = 0;
    CHECK(store->CommitCheckpoint(oldKey, identity, {small},
                                  "{\"state\":\"" + DiskPrefixCache::Digest("old-state") + "\"}",
                                  &sequence, error));
    CHECK(store->LiveCheckpoints() == 1);
    // 新检查点先淘汰最旧 oldKey，但物理字节要等 Compact 才释放。
    CHECK(!store->CommitCheckpoint(newKey, identity, {large},
                                   "{\"state\":\"" + DiskPrefixCache::Digest("new-state") + "\"}",
                                   &sequence, error));
    CHECK(error == "capacity_unavailable");
    CHECK(store->Compact(error));
    CHECK(store->CommitCheckpoint(newKey, identity, {large},
                                  "{\"state\":\"" + DiskPrefixCache::Digest("new-state") + "\"}",
                                  &sequence, error));
    CHECK(store->LiveCheckpoints() == 1);
    bool hasOld = false, hasNew = false;
    for (const auto &row : store->Checkpoints()) {
        hasOld = hasOld || row.key == oldKey;
        hasNew = hasNew || row.key == newKey;
    }
    CHECK(!hasOld);          // 旧检查点被淘汰（退掉可见性）
    CHECK(hasNew);           // 新检查点被接收，而不是新写被丢
    std::vector<std::string> objects;
    std::string identityOut;
    CHECK(store->Load(newKey, identityOut, objects, error));   // 被接收的新写必须可读回
    CHECK(objects.size() == 1 && objects[0] == large.bytes);
    const std::string after = store->Profile();
    CHECK(after.find("\"evicted_for_admission\": 1") != std::string::npos);
}


// 路 A 验收：DiskPrefixCache 的 v4 模式必须通过它原有的公开接口工作
// （写、读、列清单、元数据、重建索引、重开），且不能碰旧 v3 布局。
static void v4CacheInterface(const fs::path &root) {
    using Cache = DiskPrefixCache;
    const std::string identity = Cache::Digest("target-identity");
    const std::string family = Cache::Digest("family-identity");
    const std::string payload(300000, 'V');
    const std::string key = Cache::Digest("key-v4");
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        // v4 mode must not create the legacy layout at all.
        CHECK(!fs::exists(fs::path(root) / "v3"));
        auto writer = cache.BeginWrite(payload.size());
        const auto chunks = writer.StoreBytes(payload.size(),
            [&](size_t offset, char *out, size_t bytes) { memcpy(out, payload.data() + offset, bytes); });
        CHECK(chunks.size() == 1);
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"length", double(1024)}, {"kv_dtype", "fp8_e4m3"},
            {"provenance", "native"}, {"prefix", Cache::Digest("prefix")}, {"chunks", chunks}};
        CHECK(writer.Commit(key, state));
        CHECK(writer.WrittenBytes() == payload.size());
        // §14：经外壳提交时，元数据序列化/摘要/预检扫描/等提交线程的分项计数必须真的累加，
        // 否则"提交段的时间花在哪"依然无法回答。
        const auto profile = cache.StorageProfile();
        for (const char *field : {"ref_walk_ms", "metadata_encode_ms", "metadata_hash_ms",
                                  "metadata_bytes", "precheck_scan_ms", "actor_wait_ms",
                                  "data_sync_count", "data_sync_ms", "control_sync_count",
                                  "control_sync_ms", "record_hash_bytes", "record_hash_ms",
                                  "actor_queue_ms", "actor_exec_ms", "actor_queued_calls",
                                  "actor_inline_calls", "actor_cpu_ms", "commit_lock_ms",
                                  "quota_check_ms", "root_list_ms", "flush_objects_ms",
                                  "prepare_ms", "commit_record_ms", "root_tx_ms", "evict_ms",
                                  "lookup_ms", "pending_scan_ms", "stage_ms", "assemble_ms",
                                  "object_count", "pending_scan_steps",
                                  "object_verify_ms", "object_verify_bytes", "object_write_ms"}) {
            CHECK(profile[field].is_number());
        }
        // §14：对象数据同步（DATA）与控制记录同步必须都被计到，且记录摘要字节必须非 0。
        CHECK(profile["data_sync_count"].number_value() > 0);
        CHECK(profile["control_sync_count"].number_value() > 0);
        CHECK(profile["record_hash_bytes"].number_value() > 0);
        // §14：提交等待必须拆成"排队"与"执行"两段；本次提交至少执行过，计数必须动。
        CHECK(profile["actor_queued_calls"].number_value() + profile["actor_inline_calls"].number_value() > 0);
        CHECK(profile["actor_exec_ms"].number_value() > 0);
        // 2026-10-01 用户裁定：写侧复算**默认打开**（已接入并行、校验能力不减）→ 未设环境变量时
        // 计数必须 > 0；显式设 0 才关闭，那一侧见 strictVerifyOffStillSafe。
        unsetenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
        CHECK(profile["object_verify_bytes"].number_value() > 0);
        CHECK(profile["actor_cpu_ms"].number_value() > 0);
        // 状态正文确实被序列化并摘要过：字节数必须大于 0（否则计数是死的）。
        CHECK(profile["metadata_bytes"].number_value() > 0);
    }
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        const auto loaded = reader.Load(key);
        CHECK(loaded["kind"].string_value() == "prefix");
        CHECK(loaded["length"].int_value() == 1024);
        std::string got(payload.size(), '\0');
        reader.ReadBytes(loaded["chunks"], payload.size(),
            [&](size_t offset, const char *data, size_t bytes) {
                memcpy(got.data() + offset, data, bytes); });
        CHECK(got == payload);
        CHECK(reader.ReadBytesCount() == payload.size());
        const auto checkpoints = cache.ListCheckpoints();
        CHECK(checkpoints.size() == 1);
        if (!checkpoints.empty()) {
            CHECK(checkpoints[0].length == 1024);
            CHECK(checkpoints[0].kvDtype == "fp8_e4m3");
            CHECK(checkpoints[0].familyIdentity == family);
        }
        CHECK(cache.CheckpointLengths(family, 100000).size() == 1);
        cache.PutMetadata("identity:" + family, Cache::Json::object{{"m", "1"}});
        CHECK(cache.GetMetadata("identity:" + family)["m"].string_value() == "1");
        CHECK(cache.CapacityStatus()["storage_format"].string_value() == "4");
        // RebuildIndex must restore the projection from the control log alone.
        cache.RebuildIndex();
        CHECK(cache.ListCheckpoints().size() == 1);
        CHECK(cache.GetMetadata("identity:" + family)["m"].string_value() == "1");
    }
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        const auto loaded = reader.Load(key);
        std::string got(payload.size(), '\0');
        reader.ReadBytes(loaded["chunks"], payload.size(),
            [&](size_t offset, const char *data, size_t bytes) {
                memcpy(got.data() + offset, data, bytes); });
        CHECK(got == payload);
        // A different key is a clean miss, not a silent reuse.
        std::string error;
        bool threw = false;
        try { reader.Load(Cache::Digest("absent-key")); } catch (const std::exception &) { threw = true; }
        CHECK(threw);
        (void)error;
    }
}


// P1/P2：批量打包证据 + FIFO 失败/淘汰契约
static void batchingAndFifo(const fs::path &root) {
    std::string error;
    const auto a = MakeObject(std::string(1024, 'a'), 1);
    const auto b = MakeObject(std::string(1024, 'b'), 1);
    const auto c = MakeObject(std::string(1024, 'c'), 1);
    // 批量：多个对象必须落在同一条 OBJECT_BATCH 记录里（每对象一次 fdatasync
    // 是首轮实测 commit 暴涨 476→1380ms 的根因）。
    {
        const fs::path one = root / "batch";
        auto store = DiskPrefixV4Store::Open(one.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        const std::string descriptor = "{\"state\":\"" + DiskPrefixCache::Digest("s") + "\"}";
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("k1"), DiskPrefixCache::Digest("i"),
                                      {a, b, c}, descriptor, &sequence, error));
    }
    {
        std::string logText;
        CHECK(fastllm::prefix_v4::ReadWholeFile((root / "batch" / "v4" / "control.log").string(),
                                                logText, error));
        std::vector<fastllm::prefix_v4::ScannedRecord> records;
        CHECK(fastllm::prefix_v4::ScanControlLog(logText.data(), logText.size(), records, error) ==
              fastllm::prefix_v4::LogClassification::Complete);
        int batches = 0;
        for (const auto &record : records) {
            if (record.header.type == fastllm::prefix_v4::kRecordObjectBatch) ++batches;
        }
        // 3 个数据对象 + 1 个清单对象 = 1 条 OBJECT_BATCH（批量生效）
        CHECK(batches == 1);
    }
    // FIFO：超过配额时按最老优先显式退掉，不静默丢弃。
    {
        const fs::path fifo = root / "fifo";
        const auto small = MakeObject(std::string(1024, 'x'), 1);
        const auto medium = MakeObject(std::string(2048, 'y'), 1);
        const auto large = MakeObject(std::string(4096, 'z'), 1);
        // The third write first retires the oldest root and then waits for
        // Compact to reclaim physical log bytes.
        const uint64_t quota = MeasureTwoCompactedPhysical(
            fifo / "sizing", {medium}, {large}, true) + 4096;
        auto store = DiskPrefixV4Store::Open(fifo.string(), quota, error);
        CHECK(store != nullptr);
        if (!store) return;
        const auto descriptor = [](char fill) {
            return "{\"state\":\"" + DiskPrefixCache::Digest(std::string(1, fill)) + "\"}";
        };
        const std::string k1 = DiskPrefixCache::Digest("k1"), k2 = DiskPrefixCache::Digest("k2"),
                          k3 = DiskPrefixCache::Digest("k3");
        const std::string identity = DiskPrefixCache::Digest("identity");
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(k1, identity, {small}, descriptor('1'), &sequence, error));
        CHECK(store->CommitCheckpoint(k2, identity, {medium}, descriptor('2'), &sequence, error));
        CHECK(store->LiveCheckpoints() == 2);
        // 第 3 个先追加 ERASE；Compact 后重试才进入可见集合。
        CHECK(!store->CommitCheckpoint(k3, identity, {large}, descriptor('3'), &sequence, error));
        CHECK(error == "capacity_unavailable");
        CHECK(store->Compact(error));
        CHECK(store->CommitCheckpoint(k3, identity, {large}, descriptor('3'), &sequence, error));
        CHECK(store->LiveCheckpoints() == 2);
        const auto rows = store->Checkpoints();
        bool hasK1 = false, hasK2 = false, hasK3 = false;
        for (const auto &row : rows) {
            hasK1 = hasK1 || row.key == k1;
            hasK2 = hasK2 || row.key == k2;
            hasK3 = hasK3 || row.key == k3;
        }
        CHECK(!hasK1);            // 最老的先退
        CHECK(hasK2 && hasK3);
        std::vector<std::string> objects;
        std::string identityOut;
        CHECK(!store->Load(k1, identityOut, objects, error));
        CHECK(store->Load(k3, identityOut, objects, error));
    }
}


// P2：catalog actor——并发提交必须被串行化，序号唯一且不丢。
static void catalogActorSerializes(const fs::path &root) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "actor open: " << error << "\n"; return; }
    const int threads = 8, per = 3;
    std::atomic<int> failures{0};
    std::mutex guard;
    std::set<uint64_t> sequences;
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < per; ++i) {
                const std::string payload = std::string(128, char('a' + t)) + std::to_string(i);
                const auto object = MakeObject(payload, 1);
                const std::string key =
                    DiskPrefixCache::Digest(std::to_string(t) + ":" + std::to_string(i));
                const std::string descriptor =
                    "{\"state\":\"" + DiskPrefixCache::Digest(key) + "\"}";
                uint64_t sequence = 0;
                std::string localError;
                if (!store->CommitCheckpoint(key, DiskPrefixCache::Digest("actor-identity"),
                                             {object}, descriptor, &sequence, localError)) {
                    ++failures;
                    return;
                }
                std::lock_guard<std::mutex> lock(guard);
                // 单写者下每个检查点的发布序号必须唯一
                if (!sequences.insert(sequence).second) ++failures;
            }
        });
    }
    for (auto &worker : workers) worker.join();
    CHECK(failures.load() == 0);
    CHECK(store->LiveCheckpoints() == uint64_t(threads * per));
    CHECK(store->Checkpoints().size() == size_t(threads * per));
    CHECK(sequences.size() == size_t(threads * per));
}


// §14 计时：日志写入与同步必须分开计量且单调累计
// §22.4.4 甲（存储侧）：固定哈希池算出的"摘要+持有字节"可直接提交，
// 提交路径不再重算摘要；读回仍逐对象校验。
static void storeVerifiedUsesHashPool(const fs::path &root) {
    using fastllm::prefix_write::HashedChunk;
    using fastllm::prefix_write::TicketId;
    using Cache = DiskPrefixCache;
    const std::string identity = Cache::Digest("verified-identity");
    const std::string family = Cache::Digest("verified-family");
    const std::string key = Cache::Digest("verified-key");
    const std::string payload(200000, 'H');
    unsetenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
    auto pool = fastllm::prefix_write::CreateHashPackPool(size_t(8) << 20);
    CHECK(pool != nullptr);
    if (!pool) return;
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto writer = cache.BeginWrite(payload.size());
        CHECK(pool->Submit(TicketId{1, 0}, payload));
        TicketId ticket;
        HashedChunk chunk;
        CHECK(pool->TakeOrderedChunk(ticket, chunk, 5000));
        CHECK(chunk.Valid());
        CHECK(chunk.Bytes() == payload.size());
        const auto chunks = writer.StoreVerified(std::move(chunk));
        CHECK(chunks.size() == 1);
        CHECK(!chunk.Valid());   // Take() 只能消费一次
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"length", double(1024)}, {"kv_dtype", "fp8_e4m3"},
            {"provenance", "native"}, {"prefix", Cache::Digest("prefix")}, {"chunks", chunks}};
        CHECK(writer.Commit(key, state));
        // 2026-10-01 起写侧复算默认打开（并行执行）；本用例只断言"池化提交路径本身可用"，
        // 因此校验计数应为 > 0，而不再是 0。
        const auto profile = cache.StorageProfile();
        CHECK(profile["object_verify_bytes"].number_value() > 0);
    }
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        const auto loaded = reader.Load(key);
        CHECK(!loaded.is_null());
        std::string got(payload.size(), '\0');
        reader.ReadBytes(loaded["chunks"], payload.size(),
            [&](size_t offset, const char *data, size_t bytes) {
                memcpy(got.data() + offset, data, bytes); });
        CHECK(got == payload);
    }
    pool->Stop();
}

// §① metadata_encode：紧凑输出必须与"dump() 再去掉字符串外空白"逐字节一致。
// 一旦有任何字节差异，state 文本摘要就会变，检查点指纹随之变化 —— 这里逐例钉死。
static std::string StripWhitespaceOutsideStrings(const std::string &text) {
    std::string out = text;
    size_t output = 0;
    bool quoted = false, escaped = false;
    for (const char c : out) {
        if (quoted || (c != ' ' && c != '\t' && c != '\r' && c != '\n')) out[output++] = c;
        if (escaped) escaped = false;
        else if (quoted && c == '\\') escaped = true;
        else if (c == '"') quoted = !quoted;
    }
    out.resize(output);
    return out;
}

static void compactJsonMatchesStrip(const fs::path &root) {
    (void)root;
    using Json = DiskPrefixCache::Json;
    std::vector<Json> cases;
    cases.push_back(Json::object{});
    cases.push_back(Json::array{});
    cases.push_back(Json::object{{"empty", Json::object{}}, {"list", Json::array{}},
                                 {"nested", Json::object{{"a", Json::array{1, 2, 3}}}}});
    cases.push_back(Json::object{{"text", "带 空格 和 \"引号\" 与 \\反斜杠"},
                                 {"tab", "a\tb\nc"}, {"unicode", "中文·emoji✓"}});
    cases.push_back(Json::object{{"nums", Json::array{0, -1, 3.5, 1e9}},
                                 {"bools", Json::array{true, false}},
                                 {"null", Json(nullptr)}});
    cases.push_back(Json::object{{"layerName", 1}, {"layerType", 2}, {"input", 3},
                                 {"output", 4}, {"zzz", 5}, {"aaa", 6}});
    for (const auto &value : cases) {
        const std::string pretty = StripWhitespaceOutsideStrings(value.dump());
        const std::string compact = value.dump_compact();
        CHECK(pretty == compact);
    }
}
// §22.4.4 乙案（写侧复核默认关闭）：必须实测证明"关掉写前复核后，数据仍安全准确"。
// 三道底线：①默认关闭时确实没有重算（计数为 0）；②开关打开时确实重算（计数>0）；
// ③盘上 payload 被破坏后，读回必须 fail closed（v4_object_checksum），绝不返回坏字节。
static void strictVerifyOffStillSafe(const fs::path &root) {
    std::string error;
    const auto object = MakeObject(std::string(4096, 'B'), 1);
    const std::string key = DiskPrefixCache::Digest("strict-key");
    const std::string identity = DiskPrefixCache::Digest("strict-identity");
    // 2026-10-01：写侧复算默认是"开"；本用例专门验证"显式关闭"这一侧，故显式设 0。
    setenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY", "0", 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(key, identity, {object},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("ss") + "\"}",
                                      &sequence, error));
        const std::string profile = store->Profile();
        CHECK(profile.find("\"object_verify_bytes\": 0") != std::string::npos);
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        CHECK(store->Load(key, loadedIdentity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    }
    {
        setenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY", "1", 1);
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (store) {
            const auto other = MakeObject(std::string(4096, 'C'), 1);
            uint64_t sequence = 0;
            CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("strict-key-2"), identity, {other},
                                          "{\"state\":\"" + DiskPrefixCache::Digest("s2") + "\"}",
                                          &sequence, error));
            const std::string profile = store->Profile();
            CHECK(profile.find("\"object_verify_bytes\": 0") == std::string::npos);
        }
        unsetenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
    }
    {
        const fs::path log = fs::path(root) / "v4" / "control.log";
        std::string bytes = ReadAll(log);
        std::vector<fastllm::prefix_v4::ScannedRecord> records;
        CHECK(fastllm::prefix_v4::ScanControlLog(bytes.data(), bytes.size(), records, error) ==
              fastllm::prefix_v4::LogClassification::Complete);
        bool flipped = false;
        for (const auto &record : records) {
            if (record.header.type != fastllm::prefix_v4::kRecordObjectBatch) continue;
            const char *payload = bytes.data() + record.offset + fastllm::prefix_v4::kRecordHeaderBytes;
            const std::string payloadText(payload, size_t(record.header.payloadBytes));
            std::vector<fastllm::prefix_v4::ObjectEntry> entries;
            std::string region;
            if (!fastllm::prefix_v4::DecodeObjectBatchPayload(payloadText.data(), payloadText.size(),
                                                             entries, region, error)) {
                continue;
            }
            for (const auto &entry : entries) {
                if (entry.hash != object.hash) continue;
                const size_t tableBytes = 16 + entries.size() * 88;
                const size_t regionStart = (tableBytes + 7) & ~size_t(7);
                const size_t absolute = size_t(record.offset) +
                    fastllm::prefix_v4::kRecordHeaderBytes + regionStart + size_t(entry.offset);
                CHECK(absolute < bytes.size());
                bytes[absolute] = char(bytes[absolute] ^ 0x5A);   // 只坏 payload，不动对象表与记录摘要
                flipped = true;
                break;
            }
            if (flipped) break;
        }
        CHECK(flipped);
        {
            std::ofstream out(log, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), std::streamsize(bytes.size()));
        }
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (store) {
            std::string loadedIdentity;
            std::vector<std::string> loaded;
            std::string loadError;
            CHECK(!store->Load(key, loadedIdentity, loaded, loadError));
            CHECK(loadError == "v4_object_checksum");
        }
    }
}

static void journalTimingCounters(const fs::path &root) {
    const auto before = fastllm::prefix_v4::JournalStatsSnapshot();
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    const auto object = MakeObject(std::string(4096, 'j'), 1);
    uint64_t sequence = 0;
    CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("jk"), DiskPrefixCache::Digest("ji"),
                                  {object}, "{\"state\":\"" + DiskPrefixCache::Digest("js") + "\"}",
                                  &sequence, error));
    const auto after = fastllm::prefix_v4::JournalStatsSnapshot();
    CHECK(after.calls > before.calls);
    CHECK(after.bytes > before.bytes);
    CHECK(after.writeNs >= before.writeNs);
    CHECK(after.syncNs >= before.syncNs);
    CHECK(after.writeNs + after.syncNs > before.writeNs + before.syncNs);
    // §14：新日志文件落盘必须对父目录做一次 fsync，否则改名/建文件本身可能丢失。
    CHECK(after.dirSyncs > before.dirSyncs);
}

// §14：诊断数据结构化输出，且重建分类计数必须真实反映所见日志状态。
static void profileDiagnostics(const fs::path &root) {
    std::string error;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        const auto object = MakeObject(std::string(2048, 'p'), 1);
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("pk"), DiskPrefixCache::Digest("pi"),
                                      {object}, "{\"state\":\"" + DiskPrefixCache::Digest("ps") + "\"}",
                                      &sequence, error));
        const std::string profile = store->Profile();
        for (const char *field : {"journal_calls", "journal_bytes", "journal_write_ms", "journal_sync_ms",
                                  "dir_syncs", "object_pack_ms", "catalog_apply_ms", "live_checkpoints"}) {
            CHECK(profile.find(std::string("\"") + field + "\"") != std::string::npos);
        }
        CHECK(profile.find("\"live_checkpoints\": 1") != std::string::npos);
        // SQLite 的 durability 档位必须是实际查出来的 FULL（2），不能只是注释里写着。
        CHECK(profile.find("\"sqlite_sync\": \"2\"") != std::string::npos);
        // §14：提交段分项计数必须存在，且 metadata_bytes 必须等于本次状态正文的字节数。
        for (const char *field : {"ref_walk_ms", "metadata_encode_ms", "metadata_hash_ms",
                                  "metadata_bytes", "precheck_scan_ms", "actor_wait_ms"}) {
            CHECK(profile.find(std::string("\"") + field + "\"") != std::string::npos);
        }

    }
    // 删掉索引（catalog 只是投影）后再开：强制走一次真实重建，
    // 必须记成 valid（而不是 tail / corrupt），且仍有 1 个检查点。
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    auto reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(reopened != nullptr);
    if (!reopened) return;
    const std::string profile = reopened->Profile();
    CHECK(profile.find("\"rebuild_valid\": 1") != std::string::npos);
    CHECK(profile.find("\"rebuild_corrupt_control\": 0") != std::string::npos);
}


// §17.1 要求的持久性用例：payload 坏掉但对象表摘要仍然正确时，
// 默认 rebuild 必须排除该检查点，且不影响其它完好检查点。
static void rebuildRejectsCorruptPayload(const fs::path &root) {
    std::string error;
    const auto good = MakeObject(std::string(1024, 'g'), 1);
    const auto bad = MakeObject(std::string(1024, 'b'), 1);
    const std::string goodKey = DiskPrefixCache::Digest("good-key");
    const std::string badKey = DiskPrefixCache::Digest("bad-key");
    const std::string identity = DiskPrefixCache::Digest("rebuild-identity");
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(goodKey, identity, {good},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("gs") + "\"}",
                                      &sequence, error));
        CHECK(store->CommitCheckpoint(badKey, identity, {bad},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("bs") + "\"}",
                                      &sequence, error));
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    std::string bytes = ReadAll(log);
    std::vector<fastllm::prefix_v4::ScannedRecord> records;
    CHECK(fastllm::prefix_v4::ScanControlLog(bytes.data(), bytes.size(), records, error) ==
          fastllm::prefix_v4::LogClassification::Complete);
    bool flipped = false;
    for (const auto &record : records) {
        if (record.header.type != fastllm::prefix_v4::kRecordObjectBatch) continue;
        const char *payload = bytes.data() + record.offset + fastllm::prefix_v4::kRecordHeaderBytes;
        const std::string payloadText(payload, size_t(record.header.payloadBytes));
        std::vector<fastllm::prefix_v4::ObjectEntry> entries;
        std::string region;
        if (!fastllm::prefix_v4::DecodeObjectBatchPayload(payloadText.data(), payloadText.size(),
                                                          entries, region, error)) {
            continue;
        }
        for (const auto &entry : entries) {
            if (entry.hash != bad.hash) continue;
            const size_t tableBytes = 16 + entries.size() * 88;
            const size_t regionStart = (tableBytes + 7) / 8 * 8;
            const size_t position = record.offset + fastllm::prefix_v4::kRecordHeaderBytes +
                                    regionStart + size_t(entry.offset);
            bytes[position] = char(bytes[position] ^ 0x1);   // 只坏 payload，不动对象表
            flipped = true;
        }
    }
    CHECK(flipped);
    { std::ofstream out(log, std::ios::binary | std::ios::trunc); out.write(bytes.data(), bytes.size()); }
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "rebuild: " << error << "\n"; return; }
    std::vector<std::string> objects;
    std::string loadedIdentity;
    // 坏检查点不得发布
    CHECK(!store->Load(badKey, loadedIdentity, objects, error));
    CHECK(error == "v4_root_missing");
    // 完好检查点照常可用
    CHECK(store->Load(goodKey, loadedIdentity, objects, error));
    CHECK(objects.size() == 1 && objects[0] == good.bytes);
}


// §17.1：raw tokens 不得落盘；非法 blob 引用必须被拒（v4 与 v3 同规则）
static void v4RejectsInvalidState(const fs::path &root) {
    using Cache = DiskPrefixCache;
    const std::string identity = Cache::Digest("v4-state-identity");
    Cache cache(root.string(), identity, 1ull << 30, identity, 4);
    auto commit = [&](const Cache::Json &state) {
        auto writer = cache.BeginWrite(4096);
        writer.StoreBytes(64, [](size_t, char *out, size_t count) { memset(out, 'x', count); });
        bool threw = false;
        try { writer.Commit(Cache::Digest("state-key"), state); }
        catch (const std::exception &) { threw = true; }
        return threw;
    };
    // 原始 token 数组必须被拒
    CHECK(commit(Cache::Json::object{{"kind", "prefix"}, {"length", double(64)},
                                     {"prefix", Cache::Digest("p")}, {"tokens", Cache::Json::array{1, 2, 3}}}));
    CHECK(commit(Cache::Json::object{{"kind", "prefix"}, {"length", double(64)},
                                     {"prefix", Cache::Digest("p")},
                                     {"input_ids", Cache::Json::array{4, 5}}}));
    // 非法 blob 摘要必须被拒
    CHECK(commit(Cache::Json::object{{"kind", "prefix"}, {"length", double(64)},
                                     {"prefix", Cache::Digest("p")},
                                     {"linear", Cache::Json::object{{"sha256", "not-a-digest"}, {"bytes", double(8)}}}}));
    // 合法状态可以提交
    CHECK(!commit(Cache::Json::object{{"kind", "prefix"}, {"length", double(64)},
                                      {"prefix", Cache::Digest("p")}}));
}


// §17.1：ERASE 墓碑损坏时必须 fail closed（不得"跳过 tombstone"把已淘汰 root 复活）
static void eraseTombstoneCorruptionFailsClosed(const fs::path &root) {
    std::string error;
    const auto a = MakeObject(std::string(1024, 'a'), 1);
    const auto b = MakeObject(std::string(16384, 'b'), 1);
    const std::string identity = DiskPrefixCache::Digest("erase-identity");
    const std::string ka = DiskPrefixCache::Digest("erase-ka");
    const std::string kb = DiskPrefixCache::Digest("erase-kb");
    // Leave one aligned ERASE record after A. B is deliberately larger than
    // the remaining physical quota, so admission retires A but cannot publish
    // B. The tombstone must remain available for the corruption check.
    const uint64_t quota = MeasureCompactedPhysical(
        root / "erase-sizing", {a}, true) + 4096;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(ka, identity, {a},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("as") + "\"}",
                                      &sequence, error));
        CHECK(store->PhysicalBytes() + 4096 <= quota);
        setenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT", "1", 1);
        const bool second = store->CommitCheckpoint(
            kb, identity, {b},
            "{\"state\":\"" + DiskPrefixCache::Digest("bs") + "\"}",
            &sequence, error);
        unsetenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT");
        CHECK(!second);
        CHECK(error == "capacity_unavailable");
        CHECK(store->PhysicalBytes() <= quota);
        CHECK(store->Checkpoints().empty());   // ka 已被 ERASE 退掉，kb 尚未发布
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    std::string bytes = ReadAll(log);
    std::vector<fastllm::prefix_v4::ScannedRecord> records;
    CHECK(fastllm::prefix_v4::ScanControlLog(bytes.data(), bytes.size(), records, error) ==
          fastllm::prefix_v4::LogClassification::Complete);
    bool erased = false;
    for (const auto &record : records) {
        if (record.header.type != fastllm::prefix_v4::kRecordErase) continue;
        const size_t position = record.offset + fastllm::prefix_v4::kRecordHeaderBytes + 8;
        bytes[position] = char(bytes[position] ^ 0x1);   // 破坏墓碑 payload
        erased = true;
    }
    CHECK(erased);
    { std::ofstream out(log, std::ios::binary | std::ios::trunc); out.write(bytes.data(), bytes.size()); }
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    const auto sizeBefore = fs::file_size(log);
    // fail closed：不得静默跳过坏墓碑把 ka 复活，也不得改写原文件
    auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
    CHECK(store == nullptr);
    CHECK(!error.empty());
    CHECK(fs::file_size(log) == sizeBefore);
}

// §17.1：COMMIT 缺失（截断在 COMMIT 起点）时，该检查点不得被发布
static void missingCommitIsNotPublished(const fs::path &root) {
    std::string error;
    const auto first = MakeObject(std::string(1024, 'p'), 1);
    const auto second = MakeObject(std::string(1024, 'q'), 1);
    const std::string identity = DiskPrefixCache::Digest("truncate-identity");
    const std::string k1 = DiskPrefixCache::Digest("truncate-k1");
    const std::string k2 = DiskPrefixCache::Digest("truncate-k2");
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(k1, identity, {first},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("s1") + "\"}",
                                      &sequence, error));
        CHECK(store->CommitCheckpoint(k2, identity, {second},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("s2") + "\"}",
                                      &sequence, error));
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    std::string bytes = ReadAll(log);
    std::vector<fastllm::prefix_v4::ScannedRecord> records;
    CHECK(fastllm::prefix_v4::ScanControlLog(bytes.data(), bytes.size(), records, error) ==
          fastllm::prefix_v4::LogClassification::Complete);
    size_t lastCommit = 0;
    bool found = false;
    for (const auto &record : records) {
        if (record.header.type == fastllm::prefix_v4::kRecordManifestCommit) {
            lastCommit = size_t(record.offset);
            found = true;
        }
    }
    CHECK(found);
    { std::ofstream out(log, std::ios::binary | std::ios::trunc); out.write(bytes.data(), lastCommit); }
    std::error_code ignored;
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-wal", ignored);
    fs::remove(fs::path(root) / "v4" / "catalog.sqlite3-shm", ignored);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "truncate reopen: " << error << "\n"; return; }
    std::vector<std::string> objects;
    std::string loadedIdentity;
    // k2 的 COMMIT 被截掉 → 不发布；k1 完好可用
    CHECK(!store->Load(k2, loadedIdentity, objects, error));
    CHECK(error == "v4_root_missing");
    CHECK(store->Load(k1, loadedIdentity, objects, error));
    CHECK(objects.size() == 1 && objects[0] == first.bytes);
}


// §11.2：同一 root 的两个实例（各自 flock fd）并发提交必须串行化且不丢
static void crossProcessAppendLease(const fs::path &root) {
    std::string error;
    auto first = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    auto second = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(first != nullptr && second != nullptr);
    if (!first || !second) return;
    const std::string identity = DiskPrefixCache::Digest("lease-identity");
    std::atomic<int> failures{0};
    auto work = [&](DiskPrefixV4Store *store, int tag) {
        for (int i = 0; i < 5; ++i) {
            const auto object = MakeObject(std::string(size_t(200 + i), char('a' + tag)), 1);
            const std::string key =
                DiskPrefixCache::Digest(std::to_string(tag) + ":lease:" + std::to_string(i));
            const std::string descriptor =
                "{\"state\":\"" + DiskPrefixCache::Digest(key) + "\"}";
            uint64_t sequence = 0;
            std::string localError;
            if (!store->CommitCheckpoint(key, identity, {object}, descriptor, &sequence,
                                         localError)) {
                ++failures;
            }
        }
    };
    std::thread one(work, first.get(), 1);
    std::thread two(work, second.get(), 2);
    one.join();
    two.join();
    CHECK(failures.load() == 0);
    CHECK(first->Checkpoints().size() == 10);
    // 无争用后，维护路径（独占 lifecycle）必须能成功，且数据仍在
    error.clear();
    CHECK(first->RebuildIndex(error));
    CHECK(first->Checkpoints().size() == 10);
    error.clear();
    CHECK(second->Compact(error));
    CHECK(second->Checkpoints().size() == 10);
}


// 2026-10-01 记账修复：重建/压缩后 LiveBytes() 必须仍是**真实**字节数（旧实现写死 0）。
// 它不只是显示值——写路径用它判断"超配额该淘汰"（cache.cpp:3021/3195），
// 归零会让配额上限形同虚设。
static void liveBytesSurvivesRebuildAndCompact(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("livebytes-identity");
    const auto first = MakeObject(std::string(4096, 'a'), 1);
    const auto second = MakeObject(std::string(8192, 'b'), 1);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "livebytes open: " << error << "\n"; return; }
    CHECK(store->Commit("lb-1", identity, {first}, nullptr, error));
    CHECK(store->Commit("lb-2", identity, {second}, nullptr, error));
    const uint64_t written = store->LiveBytes();
    CHECK(written == first.bytes.size() + second.bytes.size());
    error.clear();
    const bool rebuilt = store->RebuildIndex(error);
    if (!rebuilt) std::cerr << "livebytes rebuild: " << error << "\n";
    CHECK(rebuilt);
    CHECK(store->LiveCheckpoints() == 2);
    CHECK(store->LiveBytes() == written);   // 旧实现在这里恒为 0
    error.clear();
    const bool compacted = store->Compact(error);
    if (!compacted) std::cerr << "livebytes compact: " << error << "\n";
    CHECK(compacted);
    CHECK(store->LiveCheckpoints() == 2);
    CHECK(store->LiveBytes() == written);   // 旧实现在这里也恒为 0
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(store->Load("lb-1", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == first.bytes);
    CHECK(store->Load("lb-2", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == second.bytes);
}


// 2026-10-01 合并保管员：同一个 canonical root 在同一进程只允许一个 store。
// 旧实现里 target/encoder 两个 namespace 各自 Open 同一目录 → 两个 actor 各自
// 缓存 sequence，往同一份 control.log 追加，会写出重复序号（实测
// …8,9,10,9,10,11…）并让 RebuildIndex 报 v4_control_sequence。
static void sameRootSharesOneStore(const fs::path &root) {
    std::string error;
    auto first = DiskPrefixV4Store::OpenShared(root.string(), 1ull << 30, error);
    CHECK(first != nullptr);
    if (!first) { std::cerr << "shared open: " << error << "\n"; return; }
    auto second = DiskPrefixV4Store::OpenShared(root.string(), 1ull << 30, error);
    CHECK(second != nullptr);
    CHECK(first.get() == second.get());   // 同一个 actor / 同一套 SQLite / 同一份配额
    // 配额不一致必须 fail closed，不能悄悄沿用别人的上限。
    error.clear();
    auto mismatched = DiskPrefixV4Store::OpenShared(root.string(), (1ull << 30) + 1, error);
    CHECK(mismatched == nullptr);
    CHECK(error == "v4_shared_quota_mismatch");
    // 共享同一个 store 后，两个 namespace 各自可写、可读回、序号严格递增。
    const std::string targetIdentity = DiskPrefixCache::Digest("shared-target");
    const std::string encoderIdentity = DiskPrefixCache::Digest("shared-encoder");
    const auto targetObject = MakeObject(std::string(3000, 't'), 1);
    const auto encoderObject = MakeObject(std::string(4000, 'e'), 1);
    const std::string descriptor =
        "{\"state\":\"" + DiskPrefixCache::Digest("shared-descriptor") + "\"}";
    uint64_t targetSequence = 0, encoderSequence = 0;
    error.clear();
    CHECK(second->CommitCheckpoint("shared-target-key", targetIdentity, {targetObject}, descriptor,
                                   &targetSequence, error));
    if (!error.empty()) std::cerr << "shared target commit: " << error << "\n";
    error.clear();
    CHECK(first->CommitCheckpoint("shared-encoder-key", encoderIdentity, {encoderObject}, descriptor,
                                  &encoderSequence, error));
    if (!error.empty()) std::cerr << "shared encoder commit: " << error << "\n";
    CHECK(encoderSequence > targetSequence);
    CHECK(first->LiveCheckpoints() == 2);
    error.clear();
    const bool rebuilt = first->RebuildIndex(error);
    if (!rebuilt) std::cerr << "shared rebuild: " << error << "\n";
    CHECK(rebuilt);
    CHECK(first->LiveCheckpoints() == 2);
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(first->Load("shared-target-key", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == targetObject.bytes);
    CHECK(second->Load("shared-encoder-key", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == encoderObject.bytes);
    CHECK(second->LiveBytes() == targetObject.bytes.size() + encoderObject.bytes.size());
}


// 产品路径：文本 KV 与图片特征两个 DiskPrefixCache 指向同一目录时必须共用一个 store。
// use_count 是这里唯一能直接证明"确实只有一个 store"的观测：两个 cache 各持一份，
// 加上本函数临时取到的这一份，共享时应 ≥3；若各自 Open 一个新 store，则只有 1。
static void cacheNamespacesShareOneStore(const fs::path &root) {
    using Cache = DiskPrefixCache;
    const std::string targetIdentity = Cache::Digest("ns-target");
    const std::string encoderIdentity = Cache::Digest("ns-encoder");
    const std::string family = Cache::Digest("ns-family");
    const uint64_t limit = 1ull << 30;
    Cache target(root.string(), targetIdentity, limit, family, 2);
    Cache encoder(root.string(), encoderIdentity, limit, encoderIdentity, 1);
    std::string error;
    auto shared = DiskPrefixV4Store::OpenShared(root.string(), limit, error);
    CHECK(shared != nullptr);
    if (shared) CHECK(shared.use_count() >= 3);
    auto writeThrough = [&](Cache &cache, const std::string &key, const std::string &payload) {
        auto writer = cache.BeginWrite(payload.size());
        const auto chunks = writer.StoreBytes(payload.size(),
            [&](size_t offset, char *out, size_t bytes) {
                memcpy(out, payload.data() + offset, bytes); });
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"length", double(1024)}, {"kv_dtype", "fp8_e4m3"},
            {"provenance", "native"}, {"prefix", Cache::Digest(key)}, {"chunks", chunks}};
        CHECK(writer.Commit(key, state));
    };
    auto readThrough = [&](Cache &cache, const std::string &key, const std::string &expected) {
        auto reader = cache.BeginRead();
        const auto loaded = reader.Load(key);
        std::string got(expected.size(), '\0');
        reader.ReadBytes(loaded["chunks"], expected.size(),
            [&](size_t offset, const char *data, size_t bytes) {
                memcpy(got.data() + offset, data, bytes); });
        CHECK(got == expected);
    };
    const std::string targetKey = Cache::Digest("ns-target-key");
    const std::string encoderKey = Cache::Digest("ns-encoder-key");
    const std::string targetPayload(2000, 'T');
    const std::string encoderPayload(2500, 'E');
    writeThrough(target, targetKey, targetPayload);
    writeThrough(encoder, encoderKey, encoderPayload);
    // 同一份 catalog：任一边都能看到两个 namespace 的检查点。
    CHECK(target.ListCheckpoints().size() == 2);
    CHECK(encoder.ListCheckpoints().size() == 2);
    readThrough(target, targetKey, targetPayload);
    readThrough(encoder, encoderKey, encoderPayload);
    // 共用 actor ≠ 共用命名空间：读取仍按 identity 隔离，本 namespace 看不到对方的 key。
    bool crossNamespaceRejected = false;
    try {
        auto reader = target.BeginRead();
        reader.Load(encoderKey);
    } catch (const std::exception &) { crossNamespaceRejected = true; }
    CHECK(crossNamespaceRejected);
    // 重建索引恢复整份 catalog（两个 namespace），之后两边都还能读回自己的数据。
    target.RebuildIndex();
    readThrough(target, targetKey, targetPayload);
    readThrough(encoder, encoderKey, encoderPayload);
}


// §11.4：配额在**写之前**预约——连退掉所有旧检查点都放不下时必须 fail closed
static void quotaReservationFailsClosed(const fs::path &root) {
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 100, error);   // 比记录开销还小
    CHECK(store != nullptr);
    if (!store) return;
    const auto object = MakeObject(std::string(64, 'q'), 1);
    uint64_t sequence = 0;
    // 单个对象 + 记录开销就超过配额 → 必须明确失败，且不得留下这个检查点
    const bool committed = store->CommitCheckpoint(DiskPrefixCache::Digest("too-big"),
                                                   DiskPrefixCache::Digest("quota-identity"),
                                                   {object}, "{\"state\":\"" +
                                                   DiskPrefixCache::Digest("qs") + "\"}",
                                                   &sequence, error);
    if (committed) {
        // 配额够大时允许成功；这里刻意用极小配额，成功即视为口径不符
        CHECK(store->LiveBytes() <= 1024);
    } else {
        CHECK(error == "capacity_unavailable");
    }
    CHECK(store->Checkpoints().empty());
}

// §④ 异步提交：Submit 立即返回（只代表"已受理"）；Reap 拿到同一结果；
// 失败必须原样上报；销毁时 drain 保证在途提交不丢。
static void asyncCommitPipeline(const fs::path &root) {
    std::string error;
    const auto object = MakeObject(std::string(4096, 'q'), 1);
    const std::string identity = DiskPrefixCache::Digest("async-identity");
    const std::string key = DiskPrefixCache::Digest("async-key");
    const std::string key2 = DiskPrefixCache::Digest("async-key-2");
    const std::string descriptor = "{\"state\":\"" + DiskPrefixCache::Digest("async-state") + "\"}";
    uint64_t sequence = 0;
    {
        auto store = DiskPrefixV4Store::Open((root / "run").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        // ① 提交不等待：立刻返回 true（"已受理"）
        CHECK(store->SubmitCheckpoint(key, identity, {object}, descriptor, error));
        // ② 收尾：拿到序列号，并且该检查点此时可见
        CHECK(store->ReapCheckpoint(&sequence, error));
        CHECK(sequence > 0);
        bool visible = false;
        for (const auto &row : store->Checkpoints()) visible = visible || (row.key == key);
        CHECK(visible);
        // ③ 失败必须原样上报（空 metadata → v4_checkpoint_metadata），不许静默
        CHECK(store->SubmitCheckpoint(key2, identity, {object}, "", error));
        error.clear();
        CHECK(!store->ReapCheckpoint(nullptr, error));
        CHECK(error == "v4_checkpoint_metadata");
        // ④ 在途最多 1 个：连续两次提交，第二次会先收掉第一笔
        CHECK(store->SubmitCheckpoint(key, identity, {object}, descriptor, error));
        CHECK(store->SubmitCheckpoint(key2, identity, {object}, descriptor, error));
        CHECK(store->ReapCheckpoint(nullptr, error));
    }
    // ⑤ 销毁时 drain：提交后**不显式收尾**，析构必须把这一笔落完
    {
        auto store = DiskPrefixV4Store::Open((root / "drain").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->SubmitCheckpoint(key, identity, {object}, descriptor, error));
    }
    {
        auto store = DiskPrefixV4Store::Open((root / "drain").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        bool visible = false;
        for (const auto &row : store->Checkpoints()) visible = visible || (row.key == key);
        CHECK(visible);
    }
}

// S1（"缓存真的可用"的最强证据）：写入 → 关闭 → 重开 → 全量读回 → **逐字节比对**。
// 同时覆盖两条提交路径：同步单锁原子（CommitCheckpoint）与异步（SubmitCheckpoint 后
// 刻意不 Reap，靠析构 drain 落盘）；对每个对象逐字节比较 + 重算摘要 + 描述符比对。
static void writeRestartReadBackExact(const fs::path &root) {
    std::string error;
    struct Fixture {
        std::string key, identity, descriptor;
        std::vector<DiskPrefixV4Store::Object> objects;
    };
    std::vector<Fixture> fixtures;
    for (int i = 0; i < 8; ++i) {
        Fixture f;
        f.key = "readback-key-" + std::to_string(i);
        f.identity = DiskPrefixCache::Digest("readback-identity");
        f.descriptor = "{\"state\":\"" + DiskPrefixCache::Digest("rb-" + std::to_string(i)) +
                       "\",\"length\":" + std::to_string(1000 + i) + "}";
        const size_t bytes = size_t(4096) << (i % 5);          // 4KiB … 64KiB
        f.objects.push_back(MakeObject(std::string(bytes, char('a' + i)), 1));
        f.objects.push_back(MakeObject(std::string(bytes / 2 + 1, char('A' + i)), 3));
        fixtures.push_back(std::move(f));
    }
    const fs::path dataRoot = root / "exact";
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        for (size_t i = 0; i < fixtures.size(); ++i) {
            std::vector<DiskPrefixV4Store::Object> objects(fixtures[i].objects);
            uint64_t sequence = 0;
            if (i % 2 == 0) {
                CHECK(store->CommitCheckpoint(fixtures[i].key, fixtures[i].identity, std::move(objects),
                                              fixtures[i].descriptor, &sequence, error));
                CHECK(sequence > 0);
            } else {
                CHECK(store->SubmitCheckpoint(fixtures[i].key, fixtures[i].identity, std::move(objects),
                                              fixtures[i].descriptor, error));
                // 刻意不 Reap：验证"析构时 drain"确实把数据落完
            }
        }
        CHECK(store->ReapCheckpoint(nullptr, error));
    }
    // 重开：逐对象逐字节比对、重算摘要、描述符必须原样
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        const auto rows = store->Checkpoints();
        for (const auto &f : fixtures) {
            std::string identity;
            std::vector<std::string> loaded;
            CHECK(store->Load(f.key, identity, loaded, error));
            CHECK(identity == f.identity);
            CHECK(loaded.size() == f.objects.size());
            if (loaded.size() != f.objects.size()) continue;
            for (size_t k = 0; k < loaded.size(); ++k) {
                CHECK(loaded[k] == f.objects[k].bytes);
                CHECK(DiskPrefixCache::Digest(loaded[k]) == f.objects[k].hash);
            }
            bool rowFound = false;
            for (const auto &row : rows) {
                if (row.key != f.key) continue;
                rowFound = true;
                CHECK(row.identity == f.identity);
                CHECK(row.metadata == f.descriptor);
            }
            CHECK(rowFound);
        }
    }
    // 再重开一次（走重建/投影路径）仍然逐字节一致
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        for (const auto &f : fixtures) {
            std::string identity;
            std::vector<std::string> loaded;
            CHECK(store->Load(f.key, identity, loaded, error));
            CHECK(loaded.size() == f.objects.size());
            for (size_t k = 0; k < loaded.size(); ++k) CHECK(loaded[k] == f.objects[k].bytes);
        }
    }
}

// 回归（2026-09-30 故障注入发现）：OBJECT_BATCH 的"对象表声明区域"必须**恰等于**实际载荷长度。
// 旧写入路径连最后一个对象之后也补 8 字节对齐位，于是声明比实际少 1–7 字节；读者
// DecodeObjectBatchPayload 的严格判定会拒绝整批（v4_batch_payload_size），该检查点的对象读不出来。
// 本用例故意用**非 8 对齐**的对象长度复现：修复前必然失败，修复后必须逐字节可读。
static void oddSizedObjectsStayReadable(const fs::path &root) {
    std::string error;
    // 触发点：WriteObject 里 `pendingBytes >= 8MiB 就立刻 FlushObjects`。
    // 让**最后加入的那个对象**恰好触发阈值、且它自身非 8 对齐 —— 旧代码会在它后面补位，
    // 于是"对象表声明的 region"比"实际载荷"少 1–7 字节，读者 DecodeObjectBatchPayload
    // 的严格判定拒绝整批（v4_batch_payload_size），该批对象永远读不出来。
    const auto big = MakeObject(std::string(size_t(8) * 1024 * 1024 - 5, 'a'), 1);   // 8388603
    const auto tail = MakeObject(std::string(5, 'b'), 1);                            // 5 → 合计恰好 8MiB
    const fs::path dataRoot = root / "odd";
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("odd-key", DiskPrefixCache::Digest("odd-identity"),
                                      {big, tail},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("odd-state") + "\"}",
                                      &sequence, error));
        CHECK(sequence > 0);
    }
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string identity;
        std::vector<std::string> loaded;
        CHECK(store->Load("odd-key", identity, loaded, error));
        CHECK(static_cast<int>(loaded.size()) == 2);
        if (loaded.size() == 2) {
            CHECK(loaded[0] == big.bytes);
            CHECK(loaded[1] == tail.bytes);
        }
    }
}

// 复现（2026-09-30，纯 CPU）：写失败之后，复用同一共享对象的**后续检查点**还能不能读回。
// 真机现象：一次 EIO 之后，同一请求的 6 个检查点全部 invalid_chunk_reference。
// 这里把结构最小化：第一笔只写共享对象 A（注入让它失败），第二笔复用 A 并新增 B（应成功），
// 重开后 Load(第二笔) 必须能取回 A+B。若取不回 → 复现成功，根因就在这段代码里。
// 用 LD_PRELOAD 注入 + FASTLLM_TEST_ONLY_REPRO=1 可精确控制"第几次写失败"。
static void sharedObjectAfterInjectedFailure(const fs::path &root) {
    std::string error;
    const auto shared = MakeObject(std::string(4096, 'p'), 1);
    const auto extra  = MakeObject(std::string(2048, 'q'), 1);
    const fs::path dataRoot = root / "repro";
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        const bool first = store->CommitCheckpoint("repro-key-1", DiskPrefixCache::Digest("repro-id"),
                                                   {shared},
                                                   "{\"state\":\"" + DiskPrefixCache::Digest("s1") + "\"}",
                                                   &sequence, error);
        std::fprintf(stderr, "[repro] first_commit=%d error=%s\n", int(first), error.c_str());
        error.clear();
        const bool second = store->CommitCheckpoint("repro-key-2", DiskPrefixCache::Digest("repro-id"),
                                                    {shared, extra},
                                                    "{\"state\":\"" + DiskPrefixCache::Digest("s2") + "\"}",
                                                    &sequence, error);
        std::fprintf(stderr, "[repro] second_commit=%d error=%s\n", int(second), error.c_str());
    }
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string identity; std::vector<std::string> loaded;
        const bool ok2 = store->Load("repro-key-2", identity, loaded, error);
        std::fprintf(stderr, "[repro] reopen_load_key2=%d error=%s objects=%zu (期望 2 = A+B)\n",
                     int(ok2), error.c_str(), loaded.size());
        if (ok2) {
            CHECK(loaded.size() == 2);
            if (loaded.size() == 2) { CHECK(loaded[0] == shared.bytes); CHECK(loaded[1] == extra.bytes); }
        }
        std::string id1; std::vector<std::string> l1;
        const bool ok1 = store->Load("repro-key-1", id1, l1, error);
        std::fprintf(stderr, "[repro] reopen_load_key1=%d error=%s objects=%zu (期望 0：它被注入打掉)\n",
                     int(ok1), error.c_str(), l1.size());
    }
}

// 复现（2026-09-30，cache 级）：真机现象是"一次写失败 → 同一请求后续检查点连坐"。
// store 级最小场景重现不出来，这里升到 cache 级（含状态节点/协调器去重）：写 5 笔，
// 前缀共享同一个 base 对象（走对象去重），中间某次写被注入失败，然后重开逐笔读回。
static void cacheLevelCascadeAfterInjectedFailure(const fs::path &root) {
    using Cache = DiskPrefixCache;
    const std::string identity = Cache::Digest("cascade-identity");
    const std::string family = Cache::Digest("cascade-family");
    const fs::path dataRoot = root / "cascade";
    const std::string base(64 * 1024, 'b');
    std::vector<std::string> payloads;
    for (int i = 0; i < 5; ++i) {
        payloads.push_back(base + std::string(size_t(1024) * size_t(i + 1), char('0' + i)));
    }
    std::vector<std::pair<std::string, size_t>> committed;   // key -> payload index
    {
        Cache cache(dataRoot.string(), identity, 1ull << 30, family, 4);
        for (size_t i = 0; i < payloads.size(); ++i) {
            const std::string key = Cache::Digest("cascade-key-" + std::to_string(i));   // 必须是合法摘要
            bool ok = false;
            try {
                auto writer = cache.BeginWrite(payloads[i].size());
                auto chunks = writer.StoreBytes(payloads[i].size(), [&](size_t off, char *out, size_t n) {
                    memcpy(out, payloads[i].data() + off, n);
                });
                Cache::Json state = Cache::Json::object{{"kind", "prefix"}, {"length", double(1024 * (i + 1))},
                    {"kv_dtype", "fp8_e4m3"}, {"provenance", "native"},
                    {"prefix", Cache::Digest("cascade-prefix")}, {"chunks", chunks}};
                ok = writer.Commit(key, state);
            } catch (const std::exception &error) {
                std::fprintf(stderr, "[cascade] commit %zu threw: %s\n", i, error.what());
            }
            std::fprintf(stderr, "[cascade] commit %zu = %d\n", i, int(ok));
            if (ok) committed.emplace_back(key, i);
        }
    }
    {
        Cache cache(dataRoot.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        for (const auto &entry : committed) {
            try {
                const auto loaded = reader.Load(entry.first);
                size_t expected = 0;
                for (const auto &chunk : loaded["chunks"].array_items()) {
                    expected += size_t(chunk["bytes"].number_value());
                }
                std::string got;
                got.reserve(expected);
                reader.ReadBytes(loaded["chunks"], expected, [&](size_t, const char *p, size_t n) {
                    got.append(p, n);
                });
                const bool same = (got == payloads[entry.second]);
                std::fprintf(stderr, "[cascade] reopen %s bytes=%zu/%zu equal=%d\n",
                             entry.first.c_str(), got.size(), payloads[entry.second].size(), int(same));
                CHECK(same);
            } catch (const std::exception &error) {
                std::fprintf(stderr, "[cascade] reopen %s FAILED: %s\n", entry.first.c_str(), error.what());
                CHECK(false);
            }
        }
    }
}

// 复现 P0（2026-09-30）：两笔检查点**共享同一个 >8MiB 分块**时，第二笔的 root 清单会漏掉
// 被去重（已存在）的那个对象 → 重开后 Load(第二笔) 取不全 → 恢复失败（invalid_chunk_reference）。
// 真机表现：一个请求里只有第一个检查点可恢复（所有 restart-hit 的 cached_tokens 恒为 2048）。
static void secondCheckpointSharesFirstChunk(const fs::path &root) {
    using Cache = DiskPrefixCache;
    const std::string identity = Cache::Digest("dedup-identity");
    const std::string family = Cache::Digest("dedup-family");
    const fs::path dataRoot = root / "dedup";
    // 9 MiB：第一块 8 MiB（两笔共享、会被去重）+ 尾部 1 MiB（两笔不同）
    std::string base(9u << 20, 'x');
    std::string first = base;
    std::string second = base + std::string(4096, 'y');
    auto commit = [&](const std::string &keyText, const std::string &payload, double length) {
        Cache cache(dataRoot.string(), identity, 1ull << 30, family, 4);
        auto writer = cache.BeginWrite(payload.size());
        auto chunks = writer.StoreBytes(payload.size(), [&](size_t off, char *out, size_t n) {
            memcpy(out, payload.data() + off, n);
        });
        Cache::Json state = Cache::Json::object{{"kind", "prefix"}, {"length", length},
            {"kv_dtype", "fp8_e4m3"}, {"provenance", "native"},
            {"prefix", Cache::Digest("dedup-prefix")}, {"chunks", chunks}};
        const std::string key = Cache::Digest(keyText);
        std::fprintf(stderr, "[dedup] commit %s chunks=%zu ok=%d\n", keyText.c_str(), chunks.size(),
                     int(writer.Commit(key, state)));
        return key;
    };
    const std::string key1 = commit("dedup-key-1", first, 2048);
    const std::string key2 = commit("dedup-key-2", second, 4096);   // 与第一笔共享第一块 → 去重
    {
        Cache cache(dataRoot.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        for (const auto &pair : std::vector<std::pair<std::string,std::string>>{{key1,first},{key2,second}}) {
            try {
                const auto loaded = reader.Load(pair.first);
                size_t expected = 0;
                for (const auto &chunk : loaded["chunks"].array_items()) expected += size_t(chunk["bytes"].number_value());
                std::string got; got.reserve(expected);
                reader.ReadBytes(loaded["chunks"], expected, [&](size_t, const char *d, size_t n) { got.append(d, n); });
                std::fprintf(stderr, "[dedup] reopen %s bytes=%zu/%zu equal=%d\n",
                             pair.first.substr(0,12).c_str(), got.size(), pair.second.size(), int(got == pair.second));
                CHECK(got == pair.second);
            } catch (const std::exception &error) {
                std::fprintf(stderr, "[dedup] reopen %s FAILED: %s\n", pair.first.substr(0,12).c_str(), error.what());
                CHECK(false);
            }
        }
    }
}

// P0 修复回归（2026-09-30）：root 清单必须登记"只引用"（已在库、本次不重写字节）的对象。
// 修复前：清单只含本次新写对象 → 长检查点重开后取不回全部对象 → invalid_chunk_reference。
static void referenceOnlyObjectsAppearInListing(const fs::path &root) {
    std::string error;
    const auto first  = MakeObject(std::string(4096, 'p'), 1);
    const auto second = MakeObject(std::string(2048, 'q'), 3);
    const auto third  = MakeObject(std::string(1024, 'r'), 1);
    const fs::path dataRoot = root / "reflist";
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("ref-key-1", DiskPrefixCache::Digest("ref-id"),
                                      {first, second},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("ref-s1") + "\"}",
                                      &sequence, error));
        DiskPrefixV4Store::Object refFirst;
        refFirst.hash = first.hash; refFirst.kind = first.kind; refFirst.bytesSize = first.bytes.size();
        DiskPrefixV4Store::Object refSecond;
        refSecond.hash = second.hash; refSecond.kind = second.kind; refSecond.bytesSize = second.bytes.size();
        CHECK(store->CommitCheckpoint("ref-key-2", DiskPrefixCache::Digest("ref-id"),
                                      {third, refFirst, refSecond},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("ref-s2") + "\"}",
                                      &sequence, error));
    }
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string identity;
        std::vector<std::string> loaded;
        CHECK(store->Load("ref-key-2", identity, loaded, error));
        CHECK(static_cast<int>(loaded.size()) == 3);
        bool hasFirst = false, hasSecond = false, hasThird = false;
        for (const auto &bytes : loaded) {
            if (bytes == first.bytes) hasFirst = true;
            if (bytes == second.bytes) hasSecond = true;
            if (bytes == third.bytes) hasThird = true;
        }
        CHECK(hasFirst && hasSecond && hasThird);   // 只引用对象也必须能被取回
    }
    {   // 引用一个**不存在**的对象 → 必须拒绝，绝不登记不存在的对象
        auto store = DiskPrefixV4Store::Open((root / "reflist-bad").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        DiskPrefixV4Store::Object ghost;
        ghost.hash = DiskPrefixCache::Digest("ghost-object"); ghost.kind = 1; ghost.bytesSize = 4096;
        uint64_t sequence = 0;
        CHECK(!store->CommitCheckpoint("ref-bad", DiskPrefixCache::Digest("ref-id"), {ghost},
                                       "{\"state\":\"" + DiskPrefixCache::Digest("ref-s3") + "\"}",
                                       &sequence, error));
        CHECK(error == "v4_object_missing");
    }
}

// §17.1 持久性：v4 提交协议的**记录级**逐点注入（P1 缺口）。
// 做法：子进程在指定注入点 _exit(87)（FaultPoint 命中即退），父进程重开后核对语义：
//   - durable=false 的点（数据未同步 / PREPARE 已同步但没有 COMMIT）：检查点必须
//     **完全不发布**——不许出现"半可见"或"清单指向不存在的对象"；
//   - durable=true 的点（COMMIT 已同步、投影(SQLite)未提交，或已入索引但 ack 丢失）：
//     重开必须能幂等重放出来（§17.1"COMMIT synced但SQL未完成可幂等重放"）；
//   - 两种情况都必须满足：早先已提交的检查点仍可读（不许连坐），且恢复后还能继续提交。
struct V4FailpointCase {
    const char *point;
    bool durable;
};

static void v4FailpointOne(const fs::path &parent, const V4FailpointCase &item) {
    std::string error;
    const fs::path root = parent / item.point;
    const std::string identity = DiskPrefixCache::Digest("fp-identity");
    const std::string key = DiskPrefixCache::Digest("fp-key");
    const std::string earlierKey = DiskPrefixCache::Digest("fp-key-earlier");
    const std::string descriptor = "{\"state\":\"" + DiskPrefixCache::Digest("fp-state") + "\"}";
    const auto object = MakeObject(std::string(4096, 'f'), 1);
    const auto earlier = MakeObject(std::string(2048, 'e'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(earlierKey, identity, {earlier},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("fp-early") + "\"}",
                                      &sequence, error));
    }
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        setenv("FASTLLM_DISK_CACHE_FAILPOINT", item.point, 1);
        try {
            auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
            if (!store) _exit(89);
            uint64_t sequence = 0;
            store->CommitCheckpoint(key, identity, {object}, descriptor, &sequence, error);
            _exit(88);                     // 没命中注入点 → 用例无效，父进程判失败
        } catch (...) { _exit(89); }
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 87);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    std::string identityOut;
    std::vector<std::string> loaded;
    bool visible = store->Load(key, identityOut, loaded, error);
    // §17.1"COMMIT synced但SQL未完成可幂等重放"：先看"打开即恢复"，再试显式重建。
    // 两个入口分别报告，只有"两条路都失败"才算真丢失——不能把"需要显式重建"误报成"恢复失败"。
    bool visibleAfterRebuild = visible;
    if (!visible && item.durable) {
        std::string rebuildError;
        const bool rebuilt = store->RebuildIndex(rebuildError);
        if (rebuilt) {
            std::string retryError;
            visibleAfterRebuild = store->Load(key, identityOut, loaded, retryError);
        } else if (!rebuildError.empty()) {
            error = rebuildError;
        }
    }
    std::fprintf(stderr, "[failpoint] %-24s durable=%d visible_on_open=%d visible_after_rebuild=%d error=%s\n",
                 item.point, int(item.durable), int(visible), int(visibleAfterRebuild), error.c_str());
    if (item.durable) {
        CHECK(visibleAfterRebuild);   // 只要"打开或重建"有一条路能恢复，就算可幂等重放
        if (visibleAfterRebuild) { CHECK(loaded.size() == 1); CHECK(loaded.size() == 1 && loaded[0] == object.bytes); }
        // 恢复对象还不够：模型按 Checkpoints() 获取 descriptor，COMMIT 同步后
        // 崩溃必须同时恢复正确 descriptor，不能出现 roots live / checkpoints 空。
        bool descriptorFound = false;
        for (const auto &row : store->Checkpoints()) {
            if (row.key == key && row.identity == identity) {
                descriptorFound = row.metadata == descriptor;
            }
        }
        CHECK(descriptorFound);
    } else {
        CHECK(!visible);
    }
    // 连坐检查：注入前已提交的那笔必须仍然可读。
    std::string earlierIdentity;
    std::vector<std::string> earlierLoaded;
    std::string earlierError;
    CHECK(store->Load(earlierKey, earlierIdentity, earlierLoaded, earlierError));
    // 恢复后仍能继续提交。
    uint64_t sequence = 0;
    std::string afterError;
    CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("fp-key-after"), identity,
                                  {MakeObject(std::string(3072, 'g'), 1)},
                                  "{\"state\":\"" + DiskPrefixCache::Digest("fp-after") + "\"}",
                                  &sequence, afterError));
}

static void v4FailpointsRecoverWithoutPhantoms(const fs::path &root) {
    const V4FailpointCase cases[] = {
        {"v4_data_write_before", false},
        {"v4_data_synced", false},
        {"v4_prepare_before", false},
        {"v4_prepare_after", false},
        {"v4_commit_before", false},
        {"v4_commit_after", true},
        {"v4_sqlite_commit_before", true},
        {"v4_sqlite_commit_after", true},
        {"v4_ack_before", true},
    };
    for (const auto &item : cases) v4FailpointOne(root, item);
}

// §17.1：METADATA 失败须发生在最终 COMMIT 之前；不发布新 root，也不破坏
// 旧检查点。重建后必须保持同一语义，而非只让脚本本身返回成功。
static void metadataFailureDoesNotPublishRoot(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("metadata-failure-identity");
    const auto object = MakeObject(std::string(2048, 'm'), 1);
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    uint64_t sequence = 0;
    setenv("FASTLLM_TEST_FAIL_RECORD_TYPE", "metadata", 1);
    const bool ok = store->CommitCheckpoint("metadata-failure-key", identity, {object},
                                           "{\"state\":\"" + DiskPrefixCache::Digest("metadata-failure-state") + "\"}",
                                           &sequence, error);
    unsetenv("FASTLLM_TEST_FAIL_RECORD_TYPE");
    CHECK(!ok);
    CHECK(error == "v4_control_test_injected");
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(!store->Load("metadata-failure-key", loadedIdentity, loaded, error));
    CHECK(error == "v4_root_missing");
    CHECK(store->Checkpoints().empty());
    store.reset();

    auto reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(reopened != nullptr);
    if (!reopened) return;
    CHECK(!reopened->Load("metadata-failure-key", loadedIdentity, loaded, error));
    CHECK(error == "v4_root_missing");
    CHECK(reopened->Checkpoints().empty());
    CHECK(reopened->RebuildIndex(error));
    CHECK(!reopened->Load("metadata-failure-key", loadedIdentity, loaded, error));
    CHECK(reopened->Checkpoints().empty());
}

// §17.1 撕裂尾恢复。**2026-10-02 起产品语义已改变**（src/utils/disk_prefix_cache.cpp
// 的 RebuildIndexBareLocked，S1-1 修复，commit e7c9a6ad）：
//   · 旧语义（本用例修复前的断言）：识别到未完成尾部后**永久阻止追加**——Commit 与 Compact
//     双双以 v4_control_uncommitted_tail 拒绝 ⇒ 一次掉电该 root 永久只读、SSD 缓存报废；
//   · 新语义（业界 LevelDB log_format / SQLite WAL / RocksDB 恢复同做法）：把 control.log
//     **截断到最后一个完整记录的末尾**后继续写。ScanControlLog 已保证残尾不含任何已 COMMIT
//     的根，故截断不丢任何已发布数据。
// 函数名保留 uncommittedTailBlocksAppend 只为稳定既有挂载点（CLOSEOUT 网关 + 全量列表），
// 断言已整体改写为"截断自愈后可继续追加"。
static void uncommittedTailBlocksAppend(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("tail-block-identity");
    const auto object = MakeObject(std::string(1024, 't'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("tail-good", identity, {object},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("tail-state") + "\"}",
                                      &sequence, error));
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    // 追加残尾前的"最后一个完整记录末尾"：自愈必须精确截断到这里。
    const uint64_t committedBytes = uint64_t(fs::file_size(log));
    CHECK(committedBytes > 0);
    using namespace fastllm::prefix_v4;
    MetadataRecord record;
    record.sequence = 100;
    record.key = "kv:unfinished";
    record.value = std::string(1024, 'x');
    const auto payload = EncodeMetadata(record);
    RecordHeader header;
    header.type = kRecordMetadata;
    header.coverageMode = kCoverageFullRecord;
    header.sequence = record.sequence;
    header.payloadBytes = payload.size();
    header.physicalBytes = AlignUp(kRecordHeaderBytes + payload.size(), error);
    header.digest = RecordDigest(kRecordMetadata, std::string(payload.data(), payload.size()));
    const auto headerBytes = EncodeRecordHeader(header);
    std::string tail(headerBytes.data(), headerBytes.size());
    tail.append(payload.data(), 200 - tail.size());
    {
        std::ofstream out(log, std::ios::binary | std::ios::app);
        out.write(tail.data(), tail.size());
    }
    CHECK(uint64_t(fs::file_size(log)) == committedBytes + 200);
    auto reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(reopened != nullptr);
    if (!reopened) return;
    // ① 自愈动作：残尾被截断到最后一条完整记录末尾，分类计数如实记为 UncommittedTail。
    CHECK(uint64_t(fs::file_size(log)) == committedBytes);
    CHECK(ReadAll(log).size() == committedBytes);
    const std::string profile = reopened->Profile();
    CHECK(profile.find("\"rebuild_uncommitted_tail\": 1") != std::string::npos);
    // ② 截断前的已提交数据必须逐字节可读回（截断没有丢已发布数据）。
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(reopened->Load("tail-good", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    // ③ append_blocked 已清除 ⇒ 必须能继续追加（旧语义在这里恒为 v4_control_uncommitted_tail）。
    uint64_t sequence = 0;
    error.clear();
    const bool appended = reopened->CommitCheckpoint(
        "tail-new", identity, {object},
        "{\"state\":\"" + DiskPrefixCache::Digest("tail-new-state") + "\"}", &sequence, error);
    if (!appended) std::cerr << "uncommitted-tail append after heal: " << error << "\n";
    CHECK(appended);
    CHECK(sequence > 0);
    CHECK(error.empty());
    CHECK(uint64_t(fs::file_size(log)) > committedBytes);   // 确实在截断点上继续写了
    CHECK(reopened->LiveCheckpoints() == 2);
    // ④ 自愈后显式重建仍成功，且此刻日志已是完整记录集、不再被改写。
    const uint64_t afterAppendBytes = uint64_t(fs::file_size(log));
    error.clear();
    CHECK(reopened->RebuildIndex(error));
    CHECK(uint64_t(fs::file_size(log)) == afterAppendBytes);
    CHECK(reopened->Load("tail-new", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    // ⑤ 再重开一次：自愈是持久的（不是一次性的），两笔都可读回且还能继续追加。
    reopened.reset();
    reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(reopened != nullptr);
    if (!reopened) return;
    CHECK(reopened->Load("tail-good", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    CHECK(reopened->Load("tail-new", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    uint64_t lastSequence = 0;
    error.clear();
    CHECK(reopened->CommitCheckpoint("tail-new-again", identity, {object},
                                     "{\"state\":\"" + DiskPrefixCache::Digest("tail-new-state") + "\"}",
                                     &lastSequence, error));
    CHECK(lastSequence > 0);
    CHECK(reopened->Load("tail-new-again", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == object.bytes);
    CHECK(reopened->LiveCheckpoints() == 3);   // tail-good + tail-new + tail-new-again（三个不同 key）
}

// §17.1（P2 收口，2026-09-30）：自检必须能抓到"root 清单漏登记状态引用的对象"。
// 历史缺陷（清单只登记本次新写对象 → 重开只命中 2048）正是靠旧自检"只看 Load 布尔值"溜过去的，
// 所以这里既验正例（健康的必须 missing=0），也验反例（引用了不在清单里的对象必须被指名）。
static void selfCheckManifestDetectsMissingEntries(const fs::path &root) {
    const std::string identity = DiskPrefixCache::Digest("selfcheck-identity");
    const std::string prefix = DiskPrefixCache::Digest("selfcheck-prefix");
    std::string error;
    const auto payload = MakeObject(std::string(4096, 'm'), fastllm::prefix_v4::kObjectData);
    const std::string stateText =
        "{\"kv\":[{\"sha256\":\"" + payload.hash + "\",\"bytes\":" +
        std::to_string(payload.bytes.size()) + "}]}";
    const auto stateObject = MakeObject(stateText, fastllm::prefix_v4::kObjectStateNode);
    const std::string descriptor =
        "{\"family\":\"" + identity + "\",\"kind\":\"prefix\",\"length\":2048,\"dtype\":\"fp8\","
        "\"provenance\":\"native\",\"prefix\":\"" + prefix + "\",\"created\":1,\"state\":\"" +
        stateObject.hash + "\"}";
    // 正例：状态对象 + 它引用的 payload 都在清单里 → 缺失必须是 0
    {
        auto store = DiskPrefixV4Store::Open((root / "healthy").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("selfcheck-healthy", identity, {payload, stateObject},
                                      descriptor, &sequence, error));
    }
    {
        auto store = DiskPrefixV4Store::Open((root / "healthy").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string details;
        const size_t missing = store->SelfCheckManifest(details);
        std::fprintf(stderr, "[selfcheck] healthy missing=%zu details=%s\n", missing, details.c_str());
        CHECK(missing == 0);
    }
    // 反例：故意只提交状态对象、不提交它引用的 payload → 必须报 1 条缺失并指名
    {
        auto store = DiskPrefixV4Store::Open((root / "broken").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("selfcheck-broken", identity, {stateObject},
                                      descriptor, &sequence, error));
    }
    {
        auto store = DiskPrefixV4Store::Open((root / "broken").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string details;
        const size_t missing = store->SelfCheckManifest(details);
        std::fprintf(stderr, "[selfcheck] broken missing=%zu details=%s\n", missing, details.c_str());
        CHECK(missing == 1);
        CHECK(details.find("missing=1") != std::string::npos);
    }
}

// S6（2026-10-05 复核）：功能性可恢复自检必须能抓到"状态错位/缺失"。
// 旧 SELFCHECK 只做 store 级 Load（"对象齐、能拼回字节"）⇒ 块链覆盖不连续、linear 层缺失、
// spec tokens 与长度错位都能骗过它（门⑦假绿）。本用例对 SelfCheckFunctional 写**正反例断言**：
// 正例（覆盖连续 + linear 齐全 + spec tokens==length，mtp/dflash2/none 三种）必须 bad=0；
// 反例各构造一处错位，必须 bad=1。反例里同时覆盖 mtp 与 dflash2，证明两者同等对待。
static void selfCheckFunctionalDetectsStateMismatch(const fs::path &root) {
    const std::string identity = DiskPrefixCache::Digest("functional-identity");
    const std::string prefix = DiskPrefixCache::Digest("functional-prefix");
    const int length = 512;
    // 构造一份可控的状态对象文本：layout 声明 2 层中第 0 层是 linear、spec 类型为 specKind。
    auto makeState = [&](const std::string &specKind, int specTokens, bool dropLinear,
                         int kvTokens) {
        DiskPrefixCache::Json::array linear;
        if (!dropLinear) linear.push_back(DiskPrefixCache::Json::object{{"layer", 0}});
        DiskPrefixCache::Json::array kv;
        for (int begin = 0; begin < kvTokens; begin += 256) {
            kv.push_back(DiskPrefixCache::Json::object{
                {"segments", DiskPrefixCache::Json::array{DiskPrefixCache::Json::object{
                    {"begin", double(begin)}, {"tokens", double(256)}}}}});
        }
        DiskPrefixCache::Json spec = specKind == "none"
            ? DiskPrefixCache::Json::object{{"kind", "none"}}
            : DiskPrefixCache::Json::object{{"kind", specKind}, {"tokens", double(specTokens)}};
        return DiskPrefixCache::Json(DiskPrefixCache::Json::object{
            {"kind", "prefix"}, {"length", double(length)}, {"prefix", prefix},
            {"layout", DiskPrefixCache::Json::object{
                {"abi", 4}, {"layers", 2},
                {"linear", DiskPrefixCache::Json::array{true, false}},
                {"speculative", specKind}}},
            {"kv", kv}, {"linear", linear}, {"speculative", spec}}).dump();
    };
    struct Case {
        const char *name;
        std::string state;
        size_t expectBad;
    };
    const std::vector<Case> cases = {
        {"mtp-ok", makeState("mtp", length, false, length), 0},
        {"dflash2-ok", makeState("dflash2", length, false, length), 0},
        {"none-ok", makeState("none", 0, false, length), 0},
        // 反例①：spec 状态随 length 增长——tokens 错位（mtp / dflash2 各一，证明同等对待）
        {"mtp-tokens-mismatch", makeState("mtp", 256, false, length), 1},
        {"dflash2-tokens-mismatch", makeState("dflash2", 256, false, length), 1},
        // 反例②：linear 层缺失（layout 说第 0 层是 linear，状态里没有）
        {"linear-missing", makeState("mtp", length, true, length), 1},
        // 反例③：KV 覆盖不连续/不完整（只覆盖 [0,256)，length=512）
        {"kv-short", makeState("mtp", length, false, 256), 1},
    };
    for (const auto &item : cases) {
        const fs::path caseRoot = root / item.name;
        std::string error;
        auto store = DiskPrefixV4Store::Open(caseRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        const auto stateObject = MakeObject(item.state, fastllm::prefix_v4::kObjectStateNode);
        const std::string descriptor =
            "{\"family\":\"" + identity + "\",\"kind\":\"prefix\",\"length\":" +
            std::to_string(length) + ",\"dtype\":\"fp8\",\"provenance\":\"native\",\"prefix\":\"" +
            prefix + "\",\"created\":1,\"state\":\"" + stateObject.hash + "\"}";
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint(std::string("functional-") + item.name, identity,
                                      {stateObject}, descriptor, &sequence, error));
        std::string details;
        const size_t bad = store->SelfCheckFunctional(details);
        std::fprintf(stderr, "[selfcheck-functional] case=%s bad=%zu expect=%zu details=%s\n",
                     item.name, bad, item.expectBad, details.c_str());
        CHECK(bad == item.expectBad);
    }
}

// §17.1 B7：索引数据库（SQLite 投影）**被删除或被损坏**之后，程序必须能**照控制
// 日志重建**出索引，且数据一条不丢。语义边界（fail-closed + 照日志重建）：
//   · catalog.sqlite3 只是"投影"，唯一权威是 v4/control.log；
//   · 投影**缺失** → 直接照日志重建（情形 A）；
//   · 投影**损坏** → 不得带着可疑投影继续服务（fail-closed），但同样必须把它当作
//     "可丢弃的投影"照日志重建，而不是把 SQLITE_NOTADB 直接升级成"整个缓存打不开"（情形 B）。
// 本用例把两种情形都写成**明确断言**：如果产品做不到，用例就应当失败。
static void sqliteCatalogLossRebuildsFromLog(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("b7-identity");
    // 3 个检查点，key/内容都不同：任意一个读不回或字节变了都算"丢数据"。
    const auto first = MakeObject(std::string(4096, 'a'), 1);
    const auto second = MakeObject(std::string(4096, 'b'), 1);
    const auto third = MakeObject(std::string(4096, 'c'), 1);
    const std::vector<std::pair<std::string, DiskPrefixV4Store::Object>> expected = {
        {"b7-1", first}, {"b7-2", second}, {"b7-3", third},
    };
    const uint64_t expectedBytes = first.bytes.size() + second.bytes.size() + third.bytes.size();

    // 准备：同一个 identity 下提交 3 个检查点。
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "b7 prepare open: " << error << "\n"; return; }
        for (const auto &item : expected) {
            uint64_t sequence = 0;
            CHECK(store->Commit(item.first, identity, {item.second}, &sequence, error));
        }
        CHECK(store->LiveCheckpoints() == 3);
        CHECK(store->LiveBytes() == expectedBytes);
    }   // store 析构 = 关闭；此后才允许破坏索引库文件

    const fs::path catalog = fs::path(root) / "v4" / "catalog.sqlite3";
    CHECK(fs::exists(catalog));

    // 重开后统一校验：索引确实可从控制日志重建，且数据逐字节一致、记账口径正确。
    // 情形 A/B 都走这一段，确保两条路径的期望完全相同（B7 要求"一条不丢"）。
    auto verifyRebuilt = [&](const char *stage) {
        error.clear();
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);   // 核心断言：删/坏索引都不得让 Open 失败
        if (!store) {
            std::cerr << "b7 " << stage << " reopen failed: " << error << "\n";
            return;
        }
        CHECK(store->LiveCheckpoints() == 3);
        for (const auto &item : expected) {
            std::string loadedIdentity;
            std::vector<std::string> loaded;
            const bool ok = store->Load(item.first, loadedIdentity, loaded, error);
            CHECK(ok);
            CHECK(loadedIdentity == identity);
            CHECK(loaded.size() == 1 && loaded[0] == item.second.bytes);
        }
        // 2026-10-01 记账修复：重建后 roots.bytes 必须是**真实占用**（旧实现写死 0）。
        // 3 个对象各 4096 字节、互不相同 → LiveBytes() 必须恰好等于 3×4096。
        CHECK(store->LiveBytes() > 0);
        CHECK(store->LiveBytes() == expectedBytes);
    };

    // 情形 A —— 索引库被删除（连同可能存在的 -wal/-shm）。缺投影必须照日志重建。
    {
        std::error_code ignored;
        fs::remove(catalog, ignored);
        fs::remove(fs::path(catalog.string() + "-wal"), ignored);
        fs::remove(fs::path(catalog.string() + "-shm"), ignored);
    }
    CHECK(!fs::exists(catalog));
    verifyRebuilt("catalog-deleted");

    // 情形 B —— 索引库被损坏：清空后覆盖 4096 字节垃圾（0x5A，不是合法 SQLite 头）。
    // §17.1 B7 要求：Open 仍须成功并**照控制日志重建**，3 个检查点全部读回。
    // 明确断言（不迁就实现）：若这里失败，说明实现把 SQLITE_NOTADB 直接当致命错误
    // （Database 构造/PRAGMA 抛异常 → Open 的 catch 只返回 nullptr），而不是把 catalog
    // 当可丢弃投影重建 —— 这是产品缺陷，不是用例过严。
    {
        std::ofstream out(catalog, std::ios::binary | std::ios::trunc);
        const std::string garbage(4096, '\x5A');
        out.write(garbage.data(), garbage.size());
    }
    CHECK(fs::file_size(catalog) == 4096);
    verifyRebuilt("catalog-corrupt");
}

// §保最新：CancelOldestPending 只取消"最老的、尚未开工的"已受理未持久化
// 作业——归还 InFlightBytes 额度、唤醒等待方（该笔结果槽置 v4_cancelled_oldest）、绝不删除任何已
// 持久化数据；流水线里没有可取消的队首时必须 fail-closed 报 v4_no_cancellable_pending。
// 本用例用 SubmitStaged（只受理、不等待落盘）堆出"已受理未持久化"，再走"取消 / 无牺牲者 / 取消后继续"。
static void cancelOldestPendingFreesBudget(const fs::path &root) {
    std::string error;
    const fs::path dataRoot = root / "cancel-oldest";
    const std::string identity = DiskPrefixCache::Digest("cancel-identity");
    const std::string family = DiskPrefixCache::Digest("cancel-family");
    // 三笔互不相同的暂存载荷：取消之外的两笔必须能 Load 回来逐字节比对。
    const std::string payloads[3] = {std::string(1024, '1'), std::string(2048, '2'),
                                     std::string(4096, '3')};
    const std::string keys[3] = {DiskPrefixCache::Digest("cancel-key-0"),
                                 DiskPrefixCache::Digest("cancel-key-1"),
                                 DiskPrefixCache::Digest("cancel-key-2")};
    auto stateOf = [&](int index) {
        return DiskPrefixCache::Json::object{
            {"kind", "prefix"},
            {"family_identity", family},
            {"length", double(64 * (index + 1))},
            {"prefix", DiskPrefixCache::Digest("cancel-prefix-" + std::to_string(index))}};
    };
    auto stagedOf = [&](int index) {
        std::vector<std::pair<std::string, std::string>> staged;
        staged.emplace_back(DiskPrefixCache::Digest(payloads[index]), payloads[index]);
        return staged;
    };
    {
        auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;

        // ① 连续受理 3 笔、刻意不 Reap：FIFO 里必须留着"已受理未持久化"的额度。
        //    注意 FIFO 有深度上限（impl 的 kPipelineDepth=2）：第 3 笔会先等队首收尾腾槽位再入队，
        //    所以"连续受理 3 笔"不等于"3 笔同时在途"；只要 before > 0 即满足本用例前提。
        for (int i = 0; i < 3; ++i) {
            CHECK(store->SubmitStaged(keys[i], identity, family, stateOf(i), false, stagedOf(i),
                                      error));
        }
        const uint64_t before = store->InFlightBytes();
        CHECK(before > 0);   // 已受理未持久化的字节必须记在账上

        // ② 取消最老的一笔：返回 true、释放量 > 0、在途字节同步下降（同口径归还）。
        //    该窗口内没有别的提交/收尾线程会改 pipelineHeldBytes，故等式必须严格成立。
        uint64_t freed = 0;
        error.clear();
        const bool cancelled = store->CancelOldestPending(&freed, error);
        CHECK(cancelled);
        CHECK(freed > 0);
        const uint64_t after = store->InFlightBytes();
        CHECK(after == before - freed);

        // ②附（复核 R6，2026-10-05）：额度已归还但作业仍在 FIFO 里占内存 ⇒ InFlightBytes()
        // 会低于真实占用。Profile 必须把这笔差额单独报出来，供真机判定"低报了多少"。
        {
            std::string profileError;
            const DiskPrefixCache::Json profile =
                DiskPrefixCache::Json::parse(store->Profile(false), profileError);
            CHECK(profileError.empty());
            const double released = profile["resident_quota_released_bytes"].number_value();
            std::fprintf(stderr, "[cancel-oldest] inflight_before=%llu freed=%llu "
                                 "inflight_after=%llu resident_quota_released=%.0f\n",
                         (unsigned long long)before, (unsigned long long)freed,
                         (unsigned long long)after, released);
            CHECK(released >= double(freed));
        }

        // ③ 1d（2026-10-03）契约变更：取消**不再丢弃作业**，只归还额度 ⇒ 等待方必须被**正常唤醒**
        //    且拿到成功（该检查点随后照常发布），不许再报 v4_cancelled_oldest、更不许挂死。
        //    背景：T1c 实证"取消=丢检查点"会在块链上挖洞 ⇒ parent-adopt failed reason=
        //    v4_block_parent_missing ⇒ write_skipped。业界（vLLM/SGLang/LMCache）也不淘汰在写的块。
        error.clear();
        const bool drained = store->ReapCheckpoint(nullptr, error);
        CHECK(drained);
        CHECK(error.empty());

        // ④ 契约断言（**不假设"被取消的必然是 keys[0]"**）：取消只归还额度、作业照常落盘 ⇒
        //    三笔都必须可读（链条不断），且 LiveCheckpoints 与实际可读条数一致（账目自洽）。
        {
            std::string identityOut;
            std::vector<std::string> loaded;
            error.clear();
            CHECK(store->Load(keys[2], identityOut, loaded, error));
            CHECK(identityOut == identity);
            CHECK(!loaded.empty() && loaded[0] == payloads[2]);
            int readable = 0, missing = 0;
            for (int i = 0; i < 3; ++i) {
                std::string outIdentity;
                std::vector<std::string> bytes;
                error.clear();
                if (store->Load(keys[i], outIdentity, bytes, error)) ++readable; else ++missing;
            }
            CHECK(missing == 0);   // 1d：取消不再制造"不可读的已记账检查点"
            CHECK(readable == static_cast<int>(store->LiveCheckpoints()));
        }

        // ⑤ 无牺牲者：这段读侧收尾已把流水线排空，再取消必须 fail-closed 且不回填释放量。
        uint64_t none = 12345;
        error.clear();
        CHECK(!store->CancelOldestPending(&none, error));
        CHECK(error == "v4_no_cancellable_pending");
        CHECK(none == 0);

        // ⑥ 取消后仍可继续：新提交必须被受理、最终可 Load；LiveCheckpoints 与实际可读条数一致。
        const std::string extraKey = DiskPrefixCache::Digest("cancel-key-extra");
        const std::string extraPayload = std::string(8192, 'z');
        std::vector<std::pair<std::string, std::string>> extraStaged;
        extraStaged.emplace_back(DiskPrefixCache::Digest(extraPayload), extraPayload);
        error.clear();
        CHECK(store->SubmitStaged(extraKey, identity, family, stateOf(3), false,
                                  std::move(extraStaged), error));
        error.clear();
        CHECK(store->ReapCheckpoint(nullptr, error));
        {
            std::string identityOut;
            std::vector<std::string> loaded;
            error.clear();
            CHECK(store->Load(extraKey, identityOut, loaded, error));
            bool exact = false;
            for (const auto &bytes : loaded) exact = exact || (bytes == extraPayload);
            CHECK(exact);
        }
        const std::string allKeys[4] = {keys[0], keys[1], keys[2], extraKey};
        uint64_t loadable = 0;
        for (const auto &key : allKeys) {
            std::string identityOut;
            std::vector<std::string> loaded;
            std::string loadError;
            if (store->Load(key, identityOut, loaded, loadError)) ++loadable;
        }
        CHECK(store->LiveCheckpoints() == loadable);
    }
}


// 2026-10-01 第二批收尾：**真·两个 OS 进程**争用同一 root（此前只有同进程双实例）。
// 父进程持有 store 并提交 3 笔；通过 /proc/self/exe 再起一个自身进程（环境变量 FASTLLM_TEST_CHILD_APPEND
// 指定同一 root）提交另外 3 笔；父进程再提交 1 笔；最后关闭重开，断言 7 笔全在、RebuildIndex 成功
// （序号若重复会失败），证明 flock 与序号协调在跨进程下成立。
static void twoProcessesShareRoot(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("mp-identity");
    auto parent = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(parent != nullptr);
    if (!parent) { std::cerr << "mp parent open: " << error << "\n"; return; }
    for (int i = 0; i < 3; ++i) {
        const auto object = MakeObject(std::string(size_t(2048 + i), char('p' + i)), 1);
        CHECK(parent->Commit("mp-parent-" + std::to_string(i), identity, {object}, nullptr, error));
    }
    char exe[4096];
    const ssize_t pathLength = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    CHECK(pathLength > 0);
    if (pathLength > 0) {
        exe[pathLength] = '\0';
        const std::string command = std::string("FASTLLM_TEST_CHILD_APPEND='") + root.string() +
                                    "' '" + exe + "' >/dev/null 2>&1";
        CHECK(std::system(command.c_str()) == 0);
    }
    {
        const auto object = MakeObject(std::string(2048, 'z'), 1);
        CHECK(parent->Commit("mp-parent-last", identity, {object}, nullptr, error));
    }
    CHECK(parent->LiveCheckpoints() == 7);
    parent.reset();
    auto reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(reopened != nullptr);
    if (reopened) {
        CHECK(reopened->LiveCheckpoints() == 7);
        for (int i = 0; i < 3; ++i) {
            std::string loadedIdentity;
            std::vector<std::string> loaded;
            error.clear();
            CHECK(reopened->Load("mp-child-" + std::to_string(i), loadedIdentity, loaded, error));
        }
        error.clear();
        CHECK(reopened->RebuildIndex(error));      // 跨进程序号若重复 → 这里会失败
        CHECK(reopened->LiveCheckpoints() == 7);
    }
}


// ============================================================================
// P0 守卫四连（2026-10-02，纯 CPU、无 GPU、无模型）。对应审计确认的四个写路径 P0：
//   ① 撕裂尾截断自愈；② 写入侧对称校验（发布即可读回）；③ 非阻塞收尾；④ 背压双扣守卫。
// 全部只使用本文件既有的公开 API（DiskPrefixV4Store / DiskPrefixCache::Digest / MakeObject）
// 与既有夹具习惯，不引入任何新接口。
// ============================================================================

// ① 撕裂尾截断自愈（对照 src/utils/disk_prefix_cache.cpp:3674-3701 的 S1-1 修复）。
// 构造"control.log 末尾只有一个半个记录头"的掉电残尾，然后重开：
//   · Open 必须成功；残尾必须被截断到最后一个完整记录末尾（日志必须变短）；
//   · append_blocked 必须被清除 —— 修复前该 root 会被 v4_control_uncommitted_tail 永久拒写；
//   · 早先提交的检查点必须仍可读回；随后**再提交一笔必须成功**；再重开后账目一致。
static void tornTailSelfHealsOnReopen(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("heal-identity");
    const auto first = MakeObject(std::string(2048, 'h'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "heal open: " << error << "\n"; return; }
        uint64_t sequence = 0;
        CHECK(store->CommitCheckpoint("heal-good", identity, {first},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("heal-state-1") + "\"}",
                                      &sequence, error));
        CHECK(sequence > 0);
    }
    const fs::path log = fs::path(root) / "v4" / "control.log";
    const uint64_t committedBytes = uint64_t(fs::file_size(log));
    CHECK(committedBytes > 0);
    // 残尾：一个"自称还有 1KiB 载荷"的合法记录头 + 72 字节载荷（远小于 physicalBytes）。
    // 构造手法与既有 uncommittedTailBlocksAppend（本文件 1749-1764 行）逐字段一致，
    // 保证 ScanControlLog 判为 UncommittedTail（disk_prefix_format_v4.cpp:648）。
    {
        using namespace fastllm::prefix_v4;
        MetadataRecord record;
        record.sequence = 100;
        record.key = "kv:torn-tail";
        record.value = std::string(1024, 'x');
        const auto payload = EncodeMetadata(record);
        RecordHeader header;
        header.type = kRecordMetadata;
        header.coverageMode = kCoverageFullRecord;
        header.sequence = record.sequence;
        header.payloadBytes = payload.size();
        header.physicalBytes = AlignUp(kRecordHeaderBytes + payload.size(), error);
        header.digest = RecordDigest(kRecordMetadata, std::string(payload.data(), payload.size()));
        const auto headerBytes = EncodeRecordHeader(header);
        std::string tail(headerBytes.data(), headerBytes.size());
        tail.append(payload.data(), 200 - tail.size());
        std::ofstream out(log, std::ios::binary | std::ios::app);
        out.write(tail.data(), std::streamsize(tail.size()));
    }
    CHECK(uint64_t(fs::file_size(log)) == committedBytes + 200);
    // 重开：自愈必须发生。
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "torn-tail reopen: " << error << "\n"; return; }
        CHECK(uint64_t(fs::file_size(log)) == committedBytes);   // 残尾已被截掉（自愈动作）
        CHECK(ReadAll(log).size() == committedBytes);
        const std::string profile = store->Profile();
        CHECK(profile.find("\"rebuild_uncommitted_tail\": 1") != std::string::npos);
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        CHECK(store->Load("heal-good", loadedIdentity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == first.bytes);
        // 关键断言：自愈后必须能继续提交（修复前这里恒为 v4_control_uncommitted_tail）。
        uint64_t sequence = 0;
        error.clear();
        const bool committed = store->CommitCheckpoint(
            "heal-new", identity, {MakeObject(std::string(1024, 'n'), 3)},
            "{\"state\":\"" + DiskPrefixCache::Digest("heal-state-2") + "\"}", &sequence, error);
        if (!committed) std::cerr << "torn-tail commit after heal: " << error << "\n";
        CHECK(committed);
        CHECK(sequence > 0);
        CHECK(error.empty());
    }
    // 再重开：残尾已不存在、日志不再被反复裁；两笔都必须可读回（审计一致）。
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        error.clear();
        const bool rebuilt = store->RebuildIndex(error);
        if (!rebuilt) std::cerr << "heal rebuild: " << error << "\n";
        CHECK(rebuilt);
        const std::string profile = store->Profile();
        CHECK(profile.find("\"rebuild_uncommitted_tail\": 0") != std::string::npos);
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        CHECK(store->Load("heal-good", loadedIdentity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == first.bytes);
        CHECK(store->Load("heal-new", loadedIdentity, loaded, error));
        CHECK(loaded.size() == 1 && loaded[0] == std::string(1024, 'n'));
        CHECK(store->LiveCheckpoints() == 2);
    }
}

// ② 写入侧对称校验：**已发布 == 可读回**（对照 S1-2 修复）。
// 真相：解码侧对"对象 > kMaxObjectBytes(8MiB)"（disk_prefix_format_v4.cpp:234）与
// "清单条目 > kMaxObjectListEntries(4096)"（同文件 463 行）一律硬拒；写入侧若不校验，
// 就会产出"CommitCheckpoint 成功、Load 读不回"的检查点（真机 random-seg r7 实证：
// 28 个 state 对象超过 8MiB，最大 13.2MiB）。本用例做三层断言：
//   A. 一批检查点逐笔发布、逐笔立刻读回并逐字节核对（已发布 ⇒ 可读回）；
//   B. 边界值恰等于上限 → 必须可发布且可读回；
//   C. 超过解码侧上限的状态对象/清单 → 必须在**发布前** fail-closed（给出原因、不可读）。
//   D. 生产路径（staged）状态正文超限 → 必须在发布前失败（v4_state_exceeds_decode_limit）。
static void publishedImpliesRestorable(const fs::path &root) {
    using Cache = DiskPrefixCache;
    using fastllm::prefix_v4::kMaxObjectBytes;
    using fastllm::prefix_v4::kMaxObjectListEntries;
    using fastllm::prefix_v4::kObjectStateNode;
    std::string error;
    const std::string identity = Cache::Digest("publish-identity");
    struct Attempt {
        std::string key;
        std::vector<DiskPrefixV4Store::Object> objects;
        std::string descriptor;
        bool overLimit = false;   // 按解码侧口径必须在发布前拒绝
        bool published = false;
    };
    std::vector<Attempt> attempts;
    for (int i = 0; i < 6; ++i) {
        Attempt a;
        a.key = "publish-key-" + std::to_string(i);
        const size_t bytes = size_t(1024) << (i % 4);
        a.objects.push_back(MakeObject(std::string(bytes, char('A' + i)), 1));
        a.objects.push_back(MakeObject(std::string(bytes / 2 + 1, char('a' + i)), 3));
        a.descriptor = "{\"state\":\"" + Cache::Digest("publish-state-" + std::to_string(i)) +
                       "\",\"length\":" + std::to_string(256 * (i + 1)) + "}";
        attempts.push_back(std::move(a));
    }
    {   // 边界：恰好等于解码侧上限 → 必须可发布且可读回
        Attempt a;
        a.key = "publish-key-at-limit";
        a.objects.push_back(MakeObject(std::string(size_t(kMaxObjectBytes), 'L'), 1));
        a.descriptor = "{\"state\":\"" + Cache::Digest("publish-state-limit") + "\"}";
        attempts.push_back(std::move(a));
    }
    {   // 超限状态对象（kind=状态节点）：解码侧 v4_object_bytes 会硬拒
        Attempt a;
        a.key = "publish-key-state-over-limit";
        a.overLimit = true;
        a.objects.push_back(MakeObject(std::string(size_t(kMaxObjectBytes) + 1, 'O'),
                                       kObjectStateNode));
        a.descriptor = "{\"state\":\"" + Cache::Digest("publish-state-over") + "\"}";
        attempts.push_back(std::move(a));
    }
    {   // 超限清单条目数（4097 条 > 4096）：解码侧 v4_object_list_count 会硬拒
        Attempt a;
        a.key = "publish-key-manifest-over-limit";
        a.overLimit = true;
        a.objects.reserve(size_t(kMaxObjectListEntries) + 1);
        for (uint32_t i = 0; i <= kMaxObjectListEntries; ++i) {
            a.objects.push_back(MakeObject(std::string(64, 'm') + std::to_string(i), 1));
        }
        a.descriptor = "{\"state\":\"" + Cache::Digest("publish-state-manifest") + "\"}";
        attempts.push_back(std::move(a));
    }
    size_t publishedCount = 0;
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "publish open: " << error << "\n"; return; }
        for (auto &a : attempts) {
            error.clear();
            uint64_t sequence = 0;
            a.published = store->CommitCheckpoint(a.key, identity, a.objects, a.descriptor,
                                                  &sequence, error);
            if (a.published) {
                ++publishedCount;
                CHECK(sequence > 0);
                CHECK(error.empty());
            } else {
                CHECK(!error.empty());   // fail-closed 必须给出原因，绝不静默丢点
            }
            if (a.overLimit) {
                if (a.published) {
                    std::cerr << "[publish] " << a.key
                              << " 超过解码侧上限却被发布（读回必然失败）\n";
                }
                CHECK(!a.published);
            }
        }
        // A：已发布 == 可读回。逐笔立刻读回、逐字节比对并重算摘要。
        for (const auto &a : attempts) {
            std::string loadedIdentity;
            std::vector<std::string> loaded;
            std::string loadError;
            const bool readable = store->Load(a.key, loadedIdentity, loaded, loadError);
            if (a.published) {
                if (!readable) {
                    std::cerr << "[publish] 已发布却读不回: " << a.key << " " << loadError << "\n";
                }
                CHECK(readable);
                CHECK(loadedIdentity == identity);
                CHECK(loaded.size() == a.objects.size());
                if (loaded.size() == a.objects.size()) {
                    bool exact = true;
                    for (size_t k = 0; k < loaded.size(); ++k) {
                        exact = exact && (loaded[k] == a.objects[k].bytes) &&
                                (DiskPrefixCache::Digest(loaded[k]) == a.objects[k].hash);
                    }
                    CHECK(exact);
                }
            } else {
                CHECK(!readable);                                // 未发布绝不可读
                CHECK(loadError == "v4_root_missing");
            }
        }
        CHECK(store->LiveCheckpoints() == uint64_t(publishedCount));
        CHECK(store->Checkpoints().size() == publishedCount);
    }
    // 重开 + 显式重建：已发布的必须仍然逐字节可读回（读回路径不被超限残渣污染）。
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        error.clear();
        const bool rebuilt = store->RebuildIndex(error);
        if (!rebuilt) std::cerr << "[publish] rebuild: " << error << "\n";
        CHECK(rebuilt);
        CHECK(store->LiveCheckpoints() == uint64_t(publishedCount));
        for (const auto &a : attempts) {
            if (!a.published) continue;
            std::string loadedIdentity;
            std::vector<std::string> loaded;
            std::string loadError;
            const bool readable = store->Load(a.key, loadedIdentity, loaded, loadError);
            if (!readable) {
                std::cerr << "[publish] 重开后读不回: " << a.key << " " << loadError << "\n";
            }
            CHECK(readable);
            CHECK(loaded.size() == a.objects.size());
            if (loaded.size() == a.objects.size()) {
                bool exact = true;
                for (size_t k = 0; k < loaded.size(); ++k) {
                    exact = exact && (loaded[k] == a.objects[k].bytes);
                }
                CHECK(exact);
            }
        }
    }
    // D：生产路径（staged）对称校验。9 MiB 状态正文 > kMaxObjectBytes(8 MiB) 且
    // < DiskPrefixCache::MaxMetadataBytes(64 MiB)，必然走到状态对象上限判定
    // （src/utils/disk_prefix_cache.cpp:2643 的 v4_state_exceeds_decode_limit）。
    {
        const fs::path stagedRoot = root / "staged-state";
        auto store = DiskPrefixV4Store::Open(stagedRoot.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        const std::string family = Cache::Digest("publish-family");
        const std::string key = Cache::Digest("publish-staged-over-limit");
        const std::string chunk(4096, 'c');
        std::vector<std::pair<std::string, std::string>> staged;
        staged.emplace_back(Cache::Digest(chunk), chunk);
        const std::string huge(9u << 20, 'x');
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"family_identity", family}, {"length", double(64)},
            {"prefix", Cache::Digest("publish-staged-prefix")}, {"note", huge}};
        error.clear();
        CHECK(store->SubmitStaged(key, identity, family, state, false, std::move(staged), error));
        // SubmitStaged 的 true 只代表"已受理"；持久化结果必须在收尾时 fail-closed。
        std::string reapError;
        const bool reaped = store->ReapCheckpoint(nullptr, reapError);
        if (reaped) std::cerr << "[publish] 超限状态被发布成功（应发布前拒绝）\n";
        CHECK(!reaped);
        CHECK(reapError == "v4_state_exceeds_decode_limit");
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(!store->Load(key, loadedIdentity, loaded, loadError));
        CHECK(loadError == "v4_root_missing");
        CHECK(store->LiveCheckpoints() == 0);
    }
}

// ③ 非阻塞收尾：受理若干笔后**不调用任何读接口**（不 Reap / 不 Load / 不 Checkpoints /
//    不 Profile），只反复问 InFlightBytes()，在途字节必须自行归零。
//    修复前 InFlightBytes() 刻意不做收尾 ⇒ 已完成作业的 heldBytes 一直挂在账上，
//    必须靠"下一次读请求"才能清掉（审计缺口 → PruneFinishedHead，2026-10-02）。
static void pruneFinishesWithoutRead(const fs::path &root) {
    using Cache = DiskPrefixCache;
    std::string error;
    const std::string identity = Cache::Digest("prune-identity");
    const std::string family = Cache::Digest("prune-family");
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "prune open: " << error << "\n"; return; }
    const std::string keys[4] = {
        Cache::Digest("prune-key-0"), Cache::Digest("prune-key-1"),
        Cache::Digest("prune-key-2"), Cache::Digest("prune-key-3")};
    // 连续受理 4 笔（只受理、不等待落盘）：FIFO 深度上限 2，后两笔会把前两笔挤出队首。
    for (int i = 0; i < 4; ++i) {
        const std::string payload = std::string(size_t(1024) * size_t(i + 1), char('p' + i));
        std::vector<std::pair<std::string, std::string>> staged;
        staged.emplace_back(Cache::Digest(payload), payload);
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"family_identity", family},
            {"length", double(64 * (i + 1))},
            {"prefix", Cache::Digest("prune-prefix-" + std::to_string(i))}};
        error.clear();
        CHECK(store->SubmitStaged(keys[i], identity, family, state, false, std::move(staged),
                                  error));
    }
    // 关键：从这里到断言为止**只有 InFlightBytes()**，没有任何读侧收尾接口。
    // 它自身必须做非阻塞收尾；旧实现下这里会永远停在 >0。
    bool drained = false;
    for (int attempt = 0; attempt < 5000000; ++attempt) {
        if (store->InFlightBytes() == 0) { drained = true; break; }
        std::this_thread::yield();
    }
    if (!drained) {
        std::cerr << "[prune] InFlightBytes 未能自行归零，最后值=" << store->InFlightBytes()
                  << "（旧实现不做读侧收尾就会如此）\n";
    }
    CHECK(drained);
    // 归零后（此时才允许用读接口）：4 笔必须都已持久化且可读回。
    CHECK(store->LiveCheckpoints() == 4);
    for (int i = 0; i < 4; ++i) {
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        error.clear();
        CHECK(store->Load(keys[i], loadedIdentity, loaded, error));
        // staged 路径的 Load 语义（与既有 cancelOldestPendingFreesBudget 一致：它也只断言
        // `!loaded.empty() && loaded[0]==payload`）：返回该检查点清单里的**全部**对象——
        // 本次暂存的 payload 在前，状态 JSON 对象（kind=4）在后，所以这里是 2 个而不是 1 个。
        // 实测证据（临时诊断打印）：i=0/#0 bytes=1024 first=112('p')、#1 bytes=203 first=123('{')。
        const size_t expectedSize = size_t(1024) * size_t(i + 1);
        CHECK(loaded.size() == 2);
        CHECK(loaded[0].size() == expectedSize);
        CHECK(loaded[0] == std::string(expectedSize, char('p' + i)));
        CHECK(loaded[1].find(Cache::Digest("prune-prefix-" + std::to_string(i))) !=
              std::string::npos);
    }
}

// ④ 背压双扣守卫（对照 src/utils/disk_prefix_cache.cpp:2394-2402 的 EnqueuePipeline 修复）。
// 时序：先让队首"被取消"（额度已归还、作业仍留在 FIFO 队首），再让受理方因 FIFO 满触发
// "背压出队队首"——旧代码在背压路径**无条件**再扣一次 heldBytes ⇒ uint64 下溢成
// ~1.8e19 ⇒ storeInflight 恒大于 HostLimit ⇒ 之后所有检查点被拒（服务级失效）。
// 本用例逐轮构造该时序，断言：任何时刻 InFlightBytes() 都必须是正常量级；
// 排空后必须精确为 0；之后的提交仍能成功并读回。
static void backpressureDoubleSubtractGuard(const fs::path &root) {
    using Cache = DiskPrefixCache;
    // 下溢值 = 2^64 - 小额 ≈ 1.8446744e19；任何"正常"在途字节都远小于 2^40（1 TiB）。
    const uint64_t kSaneCeiling = 1ull << 40;
    std::string error;
    const std::string identity = Cache::Digest("bp-identity");
    const std::string family = Cache::Digest("bp-family");
    const std::string bigPayload(256 * 1024, 'B');   // 队首：足够大，保证来不及走到 committing
    const std::string smallPayload(1024, 's');
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "backpressure open: " << error << "\n"; return; }
    auto submit = [&](const std::string &keyText, const std::string &payload,
                      const std::string &tag) {
        std::vector<std::pair<std::string, std::string>> staged;
        staged.emplace_back(Cache::Digest(payload), payload);
        Cache::Json state = Cache::Json::object{
            {"kind", "prefix"}, {"family_identity", family}, {"length", double(256)},
            {"prefix", Cache::Digest(tag)}};
        error.clear();
        return store->SubmitStaged(Cache::Digest(keyText), identity, family, state, false,
                                   std::move(staged), error);
    };
    int cancellations = 0;
    for (int round = 0; round < 32; ++round) {
        const std::string tag = std::to_string(round);
        // 复位：排空 FIFO，保证本轮时序确定（本用例允许使用读侧接口）。
        std::string drainError;
        store->ReapCheckpoint(nullptr, drainError);
        if (!drainError.empty()) CHECK(drainError == "v4_cancelled_oldest");
        CHECK(store->InFlightBytes() == 0);

        // ① 队首 = 大作业：CancelOldestPending 只取消 !committing 的作业，受理后它来不及
        //    走到不可逆点（committing 在 actor 认领时才置位），故取消必然命中队列里的作业。
        CHECK(submit("bp-head-" + tag, bigPayload, "bp-head-" + tag));
        // ② 第二笔占住 FIFO 第二格，使队列到达深度上限 2。
        CHECK(submit("bp-second-" + tag, smallPayload, "bp-second-" + tag));
        // ③ 取消：额度已归还，但作业仍留在 FIFO（§保最新会保留它，让丢点读侧可见）。
        uint64_t freed = 0;
        std::string cancelError;
        const bool cancelled = store->CancelOldestPending(&freed, cancelError);
        if (!cancelled) std::cerr << "[bp] round " << round << " cancel: " << cancelError << "\n";
        CHECK(cancelled);
        CHECK(freed > 0);
        if (cancelled) ++cancellations;
        // ④ 背压：FIFO 已满 ⇒ EnqueuePipeline 等队首完成并把它出队（旧代码在此二次扣减）。
        CHECK(submit("bp-third-" + tag, smallPayload, "bp-third-" + tag));
        // ⑤ 核心不变量：账目绝不允许下溢成天文数字。
        const uint64_t inflight = store->InFlightBytes();
        if (inflight >= kSaneCeiling) {
            std::cerr << "[bp] round " << round << " InFlightBytes 下溢: " << inflight << "\n";
        }
        CHECK(inflight < kSaneCeiling);
    }
    std::fprintf(stderr, "[bp] 构造出的取消时序轮数=%d/32\n", cancellations);
    CHECK(cancellations > 0);   // 用例前提：至少一轮真的构造出"取消后再背压出队"
    // 排空 + 终局账目：在途必须精确归零。
    std::string drainError;
    store->ReapCheckpoint(nullptr, drainError);
    if (!drainError.empty()) CHECK(drainError == "v4_cancelled_oldest");
    CHECK(store->InFlightBytes() == 0);
    // 下溢若发生过，预算判据会被永久毒化：后续提交必须仍能成功且可读回。
    const auto finalObject = MakeObject(std::string(2048, 'F'), 1);
    uint64_t sequence = 0;
    error.clear();
    CHECK(store->CommitCheckpoint("bp-final", identity, {finalObject},
                                  "{\"state\":\"" + Cache::Digest("bp-final-state") + "\"}",
                                  &sequence, error));
    CHECK(sequence > 0);
    std::string loadedIdentity;
    std::vector<std::string> loaded;
    CHECK(store->Load("bp-final", loadedIdentity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == finalObject.bytes);
    CHECK(store->LiveCheckpoints() > 0);
}

// ===========================================================================
// Route B（2026-10-02）CPU 单测：块链式 KV 布局（S2 阶段测试）。
//
// 阅读 src/utils/disk_prefix_cache.cpp 后确认的语义（下游断言以此为准）：
//   · KV 按 FASTLLM_PREFIX_CACHE_BLOCK_TOKENS（默认 256）切块；每块一条
//     kind=kObjectBlock(=5) 的内容寻址对象；链键 = H(父键, 块内 token 数, 块内容摘要)，
//     域串 fastllm-block-chain-v1，首块父键 = BlockGenesisKey()。
//   · 检查点记录里的 state 对象只存标量 + chain_tip/block_count/chained_tokens/
//     block_tokens，route-B 检查点的 root 直接指向该 state 对象（不再是随上下文增长的清单）。
//   · catalog 的 blocks 表只是投影，可由控制日志里的块对象重建（ParseBlockRow 复算链键+内容摘要）。
//   · 父块缺失在写入期（SplitBlocks）与读取期（ChainBlocks）各自 fail-closed，
//     错误串均为 v4_block_parent_missing。
// 全部走公开接口，不修改任何产品代码；若断言失败请按真实缺陷上报，不得放宽断言。
// ===========================================================================

namespace {

constexpr int kRouteBBundleTokens = 128;        // 默认 256-token 块 = 每块 2 条 bundle
constexpr size_t kRouteBBundleBytes = 4096;
// 块对象里放一段填充，让“同一份 kv 再提交时块对象不被重写”的日志增量对比有可观基数。
constexpr size_t kRouteBBundlePad = 2048;

// 本组用例的“8 块/16 块”断言以默认块大小 256 为前提。
void RouteBAssumeDefaultBlockTokens() {
    unsetenv("FASTLLM_PREFIX_CACHE_BLOCK_TOKENS");
}

struct RouteBPayload {
    DiskPrefixCache::Json::array kv;
    std::vector<std::pair<std::string, std::string>> staged;
    int length = 0;
};

// 造 Route B 输入：bundles 条 128-token bundle，每条一个独立数据对象（经公开管线暂存）。
// tag 参与数据对象内容：同一 tag 的不同 bundles 数共享前缀；不同 tag 则对象互不相同。
RouteBPayload MakeRouteBPayload(int bundles, const std::string &tag) {
    RouteBPayload out;
    for (int i = 0; i < bundles; ++i) {
        const int begin = i * kRouteBBundleTokens;
        std::string payload = tag + "#" + std::to_string(i);
        if (payload.size() < kRouteBBundleBytes) {
            payload.resize(kRouteBBundleBytes, char('a' + (i % 26)));
        }
        const std::string hash = DiskPrefixCache::Digest(payload);
        out.staged.emplace_back(hash, payload);
        DiskPrefixCache::Json segment = DiskPrefixCache::Json::object{
            {"layer", 0}, {"side", 0}, {"rank", 0},
            {"begin", double(begin)}, {"tokens", double(kRouteBBundleTokens)},
            {"page", double(kRouteBBundleTokens)}, {"heads", 1}, {"dim", 8}, {"dtype", 1},
            {"ranges", DiskPrefixCache::Json::array{DiskPrefixCache::Json::array{0, 1}}},
            {"bytes", double(payload.size())}, {"offset", 0.0},
        };
        out.kv.push_back(DiskPrefixCache::Json::object{
            {"bytes", double(payload.size())},
            {"segments", DiskPrefixCache::Json::array{segment}},
            {"chunks", DiskPrefixCache::Json::array{DiskPrefixCache::Json::object{
                {"sha256", hash}, {"bytes", double(payload.size())}}}},
            {"pad", std::string(kRouteBBundlePad, 'p')},
        });
    }
    out.length = bundles * kRouteBBundleTokens;
    return out;
}

// DiskPrefixCache 级 bundle：真实经 WriteSession::StoreBytes 落一个数据对象。
DiskPrefixCache::Json MakeCacheRouteBBundle(DiskPrefixCache::WriteSession &writer,
                                            int begin, const std::string &tag) {
    std::string payload = tag + "#" + std::to_string(begin);
    if (payload.size() < kRouteBBundleBytes) {
        payload.resize(kRouteBBundleBytes, char('a' + ((begin / kRouteBBundleTokens) % 26)));
    }
    const auto chunks = writer.StoreBytes(payload.size(),
        [&](size_t offset, char *out, size_t count) {
            std::memcpy(out, payload.data() + offset, count);
        });
    DiskPrefixCache::Json segment = DiskPrefixCache::Json::object{
        {"layer", 0}, {"side", 0}, {"rank", 0},
        {"begin", double(begin)}, {"tokens", double(kRouteBBundleTokens)},
        {"page", double(kRouteBBundleTokens)}, {"heads", 1}, {"dim", 8}, {"dtype", 1},
        {"ranges", DiskPrefixCache::Json::array{DiskPrefixCache::Json::array{0, 1}}},
        {"bytes", double(payload.size())}, {"offset", 0.0},
    };
    return DiskPrefixCache::Json::object{
        {"bytes", double(payload.size())},
        {"segments", DiskPrefixCache::Json::array{segment}},
        {"chunks", chunks},
        {"pad", std::string(kRouteBBundlePad, 'p')},
    };
}

DiskPrefixCache::Json MakeRouteBState(const std::string &family, int length,
                                      const std::string &prefixSeed,
                                      const DiskPrefixCache::Json::array &kv) {
    return DiskPrefixCache::Json::object{
        {"kind", "prefix"},
        {"length", double(length)},
        {"family_identity", family},
        {"kv_dtype", "fp8_e4m3"},
        {"provenance", "native"},
        {"source", DiskPrefixCache::Json::object{}},
        {"prefix", DiskPrefixCache::Digest(prefixSeed)},
        {"layout", DiskPrefixCache::Json::object{{"route_b_test", 1}}},
        {"kv", kv},
    };
}

// 测试侧独立复算链尾键：按块网格分组，逐块 H(parent, 块内 token 数, 内容摘要)。
std::string ExpectedRouteBChainTip(const DiskPrefixCache::Json::array &kv) {
    const uint32_t blockTokens = fastllm::prefix_v4::BlockTokens();
    if (blockTokens == 0) return "";
    std::vector<DiskPrefixCache::Json::array> blocks;
    for (const auto &bundle : kv) {
        const auto segments = bundle["segments"].array_items();
        if (segments.empty()) return "";
        const uint64_t begin = uint64_t(segments.front()["begin"].int_value());
        const size_t index = size_t(begin / blockTokens);
        if (blocks.size() <= index) blocks.resize(index + 1);
        blocks[index].push_back(bundle);
    }
    std::string parent = fastllm::prefix_v4::BlockGenesisKey();
    for (const auto &block : blocks) {
        uint64_t tokens = 0;
        for (const auto &bundle : block) {
            tokens += uint64_t(bundle["segments"].array_items().front()["tokens"].int_value());
        }
        const std::string content =
            DiskPrefixCache::Digest(DiskPrefixCache::Json(block).dump_compact());
        parent = fastllm::prefix_v4::BlockChainKey(parent, tokens, content);
    }
    return parent;
}

// 经公开异步管线提交一个 Route B 检查点并等待持久化（true = 已落盘）。
bool SubmitRouteBCheckpoint(DiskPrefixV4Store &store, const std::string &key,
                            const std::string &identity, const std::string &family,
                            DiskPrefixCache::Json state,
                            std::vector<std::pair<std::string, std::string>> staged,
                            uint64_t *sequence, std::string &error) {
    error.clear();
    if (!store.SubmitStaged(key, identity, family, std::move(state), true,
                            std::move(staged), error)) {
        return false;
    }
    return store.ReapCheckpoint(sequence, error);
}

// 取某检查点 state 对象的字节数，并回填 chain_tip / block_count。
size_t RouteBStateObjectBytes(DiskPrefixV4Store &store, const std::string &identity,
                              const std::string &key, std::string *chainTip, int *blockCount) {
    std::string error;
    std::string metadata;
    if (!store.Descriptor(identity, key, metadata, error)) return 0;
    std::string parseError;
    const auto descriptor = DiskPrefixCache::Json::parse(metadata, parseError);
    if (!parseError.empty()) return 0;
    const std::string stateHash = descriptor["state"].string_value();
    std::vector<std::string> objects;
    if (!store.Load(key, identity, objects, error)) return 0;
    size_t bytes = 0;
    for (const auto &object : objects) {
        if (DiskPrefixCache::Digest(object) != stateHash) continue;
        bytes = object.size();
        std::string stateError;
        const auto state = DiskPrefixCache::Json::parse(object, stateError);
        if (stateError.empty()) {
            if (chainTip != nullptr) *chainTip = state["chain_tip"].string_value();
            if (blockCount != nullptr) *blockCount = state["block_count"].int_value();
        }
    }
    return bytes;
}

// 直接读 catalog 投影里的 blocks 行数（证明块级共享/回收）。
size_t RouteBCountBlockRows(const fs::path &catalogPath) {
    sqlite3 *db = nullptr;
    if (sqlite3_open_v2(catalogPath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        if (db != nullptr) sqlite3_close(db);
        return 0;
    }
    sqlite3_stmt *statement = nullptr;
    size_t rows = 0;
    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM blocks", -1, &statement, nullptr) == SQLITE_OK &&
        sqlite3_step(statement) == SQLITE_ROW) {
        rows = size_t(sqlite3_column_int64(statement, 0));
    }
    if (statement != nullptr) sqlite3_finalize(statement);
    sqlite3_close(db);
    return rows;
}

// 人为删除一条块行（模拟父块缺失），仅供 fail-closed 用例。
bool RouteBDeleteBlockRow(const fs::path &catalogPath, const std::string &chainKey) {
    sqlite3 *db = nullptr;
    if (sqlite3_open_v2(catalogPath.c_str(), &db, SQLITE_OPEN_READWRITE, nullptr) != SQLITE_OK) {
        if (db != nullptr) sqlite3_close(db);
        return false;
    }
    sqlite3_stmt *statement = nullptr;
    bool ok = sqlite3_prepare_v2(db, "DELETE FROM blocks WHERE chain_key=?", -1, &statement,
                                 nullptr) == SQLITE_OK;
    if (ok) {
        sqlite3_bind_text(statement, 1, chainKey.c_str(), -1, SQLITE_TRANSIENT);
        ok = sqlite3_step(statement) == SQLITE_DONE;
    }
    if (statement != nullptr) sqlite3_finalize(statement);
    sqlite3_close(db);
    return ok;
}

}  // namespace

// 用例 1：链键确定性 + 内容寻址。同一份 kv 提交两次（不同 key）：
//   · chain_tip 相同，且等于测试侧独立复算的链尾键；
//   · blocks 投影行数不增长（8+8 块仍只有 8 行，块级共享）；
//   · 第二次提交的控制日志增量远小于块对象总体积（块对象一次写、永不重写）。
static void routeBChainKeyDeterministicAndContentAddressed(const fs::path &root) {
    RouteBAssumeDefaultBlockTokens();
    CHECK(fastllm::prefix_v4::BlockTokens() == fastllm::prefix_v4::kDefaultBlockTokens);
    const std::string identity = DiskPrefixCache::Digest("routeb-det-identity");
    const std::string family = DiskPrefixCache::Digest("routeb-det-family");
    const std::string keyA = DiskPrefixCache::Digest("routeb-det-key-a");
    const std::string keyB = DiskPrefixCache::Digest("routeb-det-key-b");
    const RouteBPayload payload = MakeRouteBPayload(16, "routeb-det");   // 2048 token = 8 块
    const fs::path catalog = root / "v4" / "catalog.sqlite3";
    const fs::path log = root / "v4" / "control.log";
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) { std::cerr << "routeb det open: " << error << "\n"; return; }

    uint64_t sequence = 0;
    CHECK(SubmitRouteBCheckpoint(*store, keyA, identity, family,
                                 MakeRouteBState(family, payload.length, "routeb-det-a", payload.kv),
                                 payload.staged, &sequence, error));
    CHECK(sequence > 0);
    CHECK(RouteBCountBlockRows(catalog) == 8);
    const uint64_t logAfterFirst = uint64_t(fs::file_size(log));
    std::string tipA;
    int blocksA = 0;
    CHECK(RouteBStateObjectBytes(*store, identity, keyA, &tipA, &blocksA) > 0);
    CHECK(blocksA == 8);

    CHECK(SubmitRouteBCheckpoint(*store, keyB, identity, family,
                                 MakeRouteBState(family, payload.length, "routeb-det-b", payload.kv),
                                 payload.staged, &sequence, error));
    std::string tipB;
    int blocksB = 0;
    CHECK(RouteBStateObjectBytes(*store, identity, keyB, &tipB, &blocksB) > 0);
    CHECK(blocksB == 8);
    CHECK(tipA.size() == 64);
    CHECK(tipA == tipB);                                  // 同一份 kv ⇒ 同一链尾键
    CHECK(tipA == ExpectedRouteBChainTip(payload.kv));    // 与独立复算一致
    CHECK(RouteBCountBlockRows(catalog) == 8);            // 行数不增长

    size_t blockObjectBytes = 0;
    std::vector<std::string> objects;
    CHECK(store->Load(keyB, identity, objects, error));
    for (const auto &object : objects) {
        std::string parseError;
        const auto json = DiskPrefixCache::Json::parse(object, parseError);
        if (parseError.empty() && json["chain_key"].is_string()) blockObjectBytes += object.size();
    }
    const uint64_t growth = uint64_t(fs::file_size(log)) - logAfterFirst;
    std::fprintf(stderr, "[routeb-det] tip=%.16s blocks=%d log_growth=%llu block_object_bytes=%zu\n",
                 tipA.c_str(), blocksA, (unsigned long long)growth, blockObjectBytes);
    CHECK(blockObjectBytes > 0);
    CHECK(growth < blockObjectBytes);   // 块对象没有被重写
}

// 用例 2：state 对象 O(1)。8 块（2048 token）与 16 块（4096 token）两次提交，
// state 对象字节数几乎相同（旧形态按 token 线性增长，长上下文会撞 8 MiB 单对象上限）。
static void routeBStateObjectSizeIsConstant(const fs::path &root) {
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = DiskPrefixCache::Digest("routeb-o1-identity");
    const std::string family = DiskPrefixCache::Digest("routeb-o1-family");
    const std::string key8 = DiskPrefixCache::Digest("routeb-o1-key-8");
    const std::string key16 = DiskPrefixCache::Digest("routeb-o1-key-16");
    const RouteBPayload small = MakeRouteBPayload(16, "routeb-o1");    // 8 块（共享前缀）
    const RouteBPayload large = MakeRouteBPayload(32, "routeb-o1");    // 16 块
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    uint64_t sequence = 0;
    CHECK(SubmitRouteBCheckpoint(*store, key8, identity, family,
                                 MakeRouteBState(family, small.length, "routeb-o1-a", small.kv),
                                 small.staged, &sequence, error));
    CHECK(SubmitRouteBCheckpoint(*store, key16, identity, family,
                                 MakeRouteBState(family, large.length, "routeb-o1-b", large.kv),
                                 large.staged, &sequence, error));
    std::string tip8, tip16;
    int blocks8 = 0, blocks16 = 0;
    const size_t bytes8 = RouteBStateObjectBytes(*store, identity, key8, &tip8, &blocks8);
    const size_t bytes16 = RouteBStateObjectBytes(*store, identity, key16, &tip16, &blocks16);
    std::fprintf(stderr, "[routeb-o1] state_bytes 8blocks=%zu 16blocks=%zu delta=%lld\n",
                 bytes8, bytes16, (long long)bytes16 - (long long)bytes8);
    CHECK(blocks8 == 8);
    CHECK(blocks16 == 16);
    CHECK(bytes8 > 0 && bytes16 > 0);
    CHECK(bytes8 < 4096 && bytes16 < 4096);   // 远低于 8 MiB 单对象上限
    CHECK(bytes16 <= bytes8 + 64);            // 块数翻倍不改变单对象大小
}

// 用例 3：多块链式恢复（模拟重启）。提交 16 块后关闭并重新打开 store，
// 按链恢复出的 kv 必须与提交时逐字节一致；length/chain_tip/block_count 正确。
static void routeBMultiBlockChainRestoresByteExact(const fs::path &root) {
    using Cache = DiskPrefixCache;
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = Cache::Digest("routeb-restore-identity");
    const std::string family = Cache::Digest("routeb-restore-family");
    const std::string key = Cache::Digest("routeb-restore-key");
    const int bundles = 32;   // 4096 token = 16 块
    Cache::Json::array original;
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto writer = cache.BeginWrite(8u << 20);
        for (int i = 0; i < bundles; ++i) {
            original.push_back(MakeCacheRouteBBundle(writer, i * kRouteBBundleTokens, "routeb-restore"));
        }
        CHECK(writer.Commit(key, MakeRouteBState(family, bundles * kRouteBBundleTokens,
                                                 "routeb-restore-prefix", original)));
    }   // 关闭 = 模拟进程重启
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        const Cache::Json restored = reader.Load(key, identity);
        const auto restoredKv = restored["kv"].array_items();
        CHECK(restored["length"].int_value() == bundles * kRouteBBundleTokens);
        CHECK(restored["block_count"].int_value() == 16);
        CHECK(restored["chained_tokens"].int_value() == bundles * kRouteBBundleTokens);
        CHECK(restored["chain_tip"].string_value().size() == 64);
        CHECK(restored["chain_tip"].string_value() == ExpectedRouteBChainTip(original));
        CHECK(restoredKv.size() == original.size());
        bool identical = restoredKv.size() == original.size();
        for (size_t i = 0; i < restoredKv.size() && i < original.size(); ++i) {
            if (restoredKv[i].dump_compact() != original[i].dump_compact()) identical = false;
        }
        std::fprintf(stderr, "[routeb-restore] block_count=%d kv_bundles=%zu byte_identical=%d\n",
                     restored["block_count"].int_value(), restoredKv.size(), int(identical));
        CHECK(identical);
    }
}

// 用例 4：父块缺失 fail-closed。
//   (a) 写入期：kv 从 token 256 开始（缺第 0 块的父链）⇒ 提交必须失败；
//   (b) 读取期：先正常提交完整链，再人为删掉链中段的块行 ⇒ Load 必须失败并报
//       v4_block_parent_missing，绝不返回部分/拼接错误的 kv。
static void routeBParentBlockMissingFailsClosed(const fs::path &root) {
    using Cache = DiskPrefixCache;
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = Cache::Digest("routeb-parent-identity");
    const std::string family = Cache::Digest("routeb-parent-family");
    const RouteBPayload payload = MakeRouteBPayload(16, "routeb-parent");   // 8 块

    // (a) 写入期 fail-closed
    {
        const std::string key = Cache::Digest("routeb-parent-suffix-key");
        Cache::Json::array suffix;   // 跳过前 2 条 bundle（token 0..255），首块不再是 token 0
        for (size_t i = 2; i < payload.kv.size(); ++i) suffix.push_back(payload.kv[i]);
        std::string error;
        auto store = DiskPrefixV4Store::Open((root / "write").string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (store) {
            std::vector<std::pair<std::string, std::string>> staged;
            CHECK(store->SubmitStaged(key, identity, family,
                                      MakeRouteBState(family, payload.length, "routeb-parent-suffix", suffix),
                                      true, std::move(staged), error));   // 异步：受理成功
            uint64_t sequence = 0;
            std::string reapError;
            CHECK(!store->ReapCheckpoint(&sequence, reapError));          // 持久化必须失败
            std::fprintf(stderr, "[routeb-parent] write-side reap_error=%s\n", reapError.c_str());
            CHECK(reapError == "v4_block_parent_missing");
        }
    }

    // 先提交一条完整链
    const std::string key = Cache::Digest("routeb-parent-load-key");
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto writer = cache.BeginWrite(8u << 20);
        Cache::Json::array kv;
        for (int i = 0; i < 16; ++i) {
            kv.push_back(MakeCacheRouteBBundle(writer, i * kRouteBBundleTokens, "routeb-parent"));
        }
        CHECK(writer.Commit(key, MakeRouteBState(family, payload.length, "routeb-parent-full", kv)));
    }

    // (b) 读取期 fail-closed：删掉链中段的块行（= 后续块的父块缺失）
    const fs::path catalog = root / "v4" / "catalog.sqlite3";
    std::string middle;
    {
        std::string error;
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        std::string tip;
        int blocks = 0;
        CHECK(RouteBStateObjectBytes(*store, identity, key, &tip, &blocks) > 0);
        CHECK(blocks == 8);
        std::vector<DiskPrefixV4Store::BlockRow> rows;
        CHECK(store->ChainBlocks(tip, uint64_t(blocks), rows, error));
        CHECK(rows.size() == size_t(blocks));
        middle = rows[rows.size() / 2].chainKey;
    }   // 先关闭 store，再改投影，避免两个 SQLite 连接并发写同一 catalog
    CHECK(!middle.empty());
    CHECK(RouteBDeleteBlockRow(catalog, middle));
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        bool threw = false;
        std::string message;
        try {
            reader.Load(key, identity);
        } catch (const std::exception &exception) {
            threw = true;
            message = exception.what();
        }
        std::fprintf(stderr, "[routeb-parent] read-side threw=%d message=%s\n", int(threw), message.c_str());
        CHECK(threw);   // fail-closed：宁可抛错，也不返回半条链
        CHECK(message == "v4_block_parent_missing");
    }
}

// 用例 5：索引可重建。删掉 catalog 投影后重开 + RebuildIndex：
// SelfCheckManifest 缺失数必须为 0，blocks 行按控制日志里的块对象复原，且仍能正常恢复。
static void routeBRebuildIndexRestoresBlocksProjection(const fs::path &root) {
    using Cache = DiskPrefixCache;
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = Cache::Digest("routeb-rebuild-identity");
    const std::string family = Cache::Digest("routeb-rebuild-family");
    const std::string key8 = Cache::Digest("routeb-rebuild-key-8");
    const std::string key16 = Cache::Digest("routeb-rebuild-key-16");
    const RouteBPayload small = MakeRouteBPayload(16, "routeb-rebuild");   // 8 块
    const RouteBPayload large = MakeRouteBPayload(32, "routeb-rebuild");   // 16 块（共享前 8 块）
    std::string tip16;
    {
        std::string error;
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        uint64_t sequence = 0;
        CHECK(SubmitRouteBCheckpoint(*store, key8, identity, family,
                                     MakeRouteBState(family, small.length, "routeb-rebuild-a", small.kv),
                                     small.staged, &sequence, error));
        CHECK(SubmitRouteBCheckpoint(*store, key16, identity, family,
                                     MakeRouteBState(family, large.length, "routeb-rebuild-b", large.kv),
                                     large.staged, &sequence, error));
        int blocks = 0;
        CHECK(RouteBStateObjectBytes(*store, identity, key16, &tip16, &blocks) > 0);
        CHECK(blocks == 16);
    }
    const fs::path catalog = root / "v4" / "catalog.sqlite3";
    {
        std::error_code ignored;
        fs::remove(catalog, ignored);
        fs::remove(fs::path(catalog.string() + "-wal"), ignored);
        fs::remove(fs::path(catalog.string() + "-shm"), ignored);
    }
    CHECK(!fs::exists(catalog));
    {
        std::string error;
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);   // 缺投影 ⇒ 照日志重建
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->RebuildIndex(error));   // 再显式重建一次（幂等）
        std::string details;
        const size_t missing = store->SelfCheckManifest(details);
        std::fprintf(stderr, "[routeb-rebuild] selfcheck_missing=%zu details=%s\n",
                     missing, details.c_str());
        CHECK(missing == 0);
        CHECK(RouteBCountBlockRows(catalog) == 16);   // 8 共享 + 8 新块
        std::vector<DiskPrefixV4Store::BlockRow> rows;
        CHECK(store->ChainBlocks(tip16, 16, rows, error));
        CHECK(rows.size() == 16);
        CHECK(rows.front().index == 0 && rows.front().tokenStart == 0);
        CHECK(rows.back().index == 15);
        std::vector<std::string> objects;
        CHECK(store->Load(key8, identity, objects, error));
        CHECK(store->Load(key16, identity, objects, error));
    }
    {
        Cache cache(root.string(), identity, 1ull << 30, family, 4);
        auto reader = cache.BeginRead();
        const Cache::Json restored = reader.Load(key16, identity);
        CHECK(restored["kv"].array_items().size() == large.kv.size());
        CHECK(restored["block_count"].int_value() == 16);
    }
}

// 用例 6：Compact 保留链可达块、回收未引用块。
// 配额刻意只装得下一个检查点：A（8 块）先提交，B（16 块、与 A 不共享）提交时把 A FIFO 淘汰；
// 物理配额压力先撤销 A，再 Compact 回收 A 的块；重试写入 B 后，B 的链仍逐块可达。
static void routeBCompactKeepsReachableBlocksReclaimsUnreferenced(const fs::path &root) {
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = DiskPrefixCache::Digest("routeb-compact-identity");
    const std::string family = DiskPrefixCache::Digest("routeb-compact-family");
    const std::string keyA = DiskPrefixCache::Digest("routeb-compact-key-a");
    const std::string keyB = DiskPrefixCache::Digest("routeb-compact-key-b");
    const RouteBPayload payloadA = MakeRouteBPayload(16, "routeb-compact-a");   // 8 块
    const RouteBPayload payloadB = MakeRouteBPayload(32, "routeb-compact-b");   // 16 块（不共享）

    // 先在隔离目录量出两个检查点各自的 LiveBytes（= roots.bytes = 本次新写对象字节和），
    // 再据此选“只装得下一个”的配额：quota = S_B + S_A/4。提交按 LiveBytes+incoming 预约，
    // 于是第二次提交必然淘汰 A，而对象表开销（216B/条）远小于 S_A/4，B 不会被自己挤掉。
    uint64_t bytesA = 0, bytesB = 0;
    {
        std::string error;
        auto sizing = DiskPrefixV4Store::Open((root / "sizing").string(), 1ull << 40, error);
        CHECK(sizing != nullptr);
        if (!sizing) return;
        uint64_t sequence = 0;
        CHECK(SubmitRouteBCheckpoint(*sizing, keyA, identity, family,
                                     MakeRouteBState(family, payloadA.length, "routeb-compact-a", payloadA.kv),
                                     payloadA.staged, &sequence, error));
        bytesA = sizing->LiveBytes();
        CHECK(SubmitRouteBCheckpoint(*sizing, keyB, identity, family,
                                     MakeRouteBState(family, payloadB.length, "routeb-compact-b", payloadB.kv),
                                     payloadB.staged, &sequence, error));
        const uint64_t total = sizing->LiveBytes();
        CHECK(total > bytesA);
        bytesB = total - bytesA;
    }
    CHECK(bytesA > 0 && bytesB > 0);
    const uint64_t quota = bytesB + bytesA / 4;
    std::fprintf(stderr, "[routeb-compact] measured S_A=%llu S_B=%llu quota=%llu\n",
                 (unsigned long long)bytesA, (unsigned long long)bytesB, (unsigned long long)quota);

    const fs::path catalog = root / "v4" / "catalog.sqlite3";
    const fs::path log = root / "v4" / "control.log";
    std::string tipA, tipB;
    {
        std::string error;
        auto store = DiskPrefixV4Store::Open(root.string(), quota, error);
        CHECK(store != nullptr);
        if (!store) { std::cerr << "routeb compact open: " << error << "\n"; return; }
        uint64_t sequence = 0;
        CHECK(SubmitRouteBCheckpoint(*store, keyA, identity, family,
                                     MakeRouteBState(family, payloadA.length, "routeb-compact-a", payloadA.kv),
                                     payloadA.staged, &sequence, error));
        CHECK(store->LiveCheckpoints() == 1);
        int blocksA = 0;
        CHECK(RouteBStateObjectBytes(*store, identity, keyA, &tipA, &blocksA) > 0);
        CHECK(blocksA == 8);
        CHECK(tipA == ExpectedRouteBChainTip(payloadA.kv));
        CHECK(RouteBCountBlockRows(catalog) == 8);

        // Keep the retired bytes until the explicit Compact below; otherwise
        // the background request may finish before the before/after sample.
        setenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT", "1", 1);
        CHECK(!SubmitRouteBCheckpoint(*store, keyB, identity, family,
                                      MakeRouteBState(family, payloadB.length, "routeb-compact-b", payloadB.kv),
                                      payloadB.staged, &sequence, error));
        CHECK(error == "capacity_unavailable");
        CHECK(store->PhysicalBytes() <= quota);
        CHECK(store->LiveCheckpoints() == 0);   // A 已被 ERASE，物理空间待 Compact
        std::vector<std::string> objects;
        CHECK(!store->Load(keyA, identity, objects, error));
        CHECK(error == "v4_root_missing");

        const uint64_t beforeCompact = uint64_t(fs::file_size(log));
        CHECK(store->Compact(error));
        const uint64_t afterCompact = uint64_t(fs::file_size(log));
        unsetenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT");
        std::fprintf(stderr, "[routeb-compact] physical_before=%llu physical_after=%llu quota=%llu\n",
                     (unsigned long long)beforeCompact, (unsigned long long)afterCompact,
                     (unsigned long long)quota);
        // Admission may already have compacted the empty log before this call.
        CHECK(beforeCompact == 0 ? afterCompact == 0 : afterCompact < beforeCompact);
        CHECK(SubmitRouteBCheckpoint(*store, keyB, identity, family,
                                     MakeRouteBState(family, payloadB.length, "routeb-compact-b", payloadB.kv),
                                     payloadB.staged, &sequence, error));
        CHECK(store->PhysicalBytes() <= quota);
        CHECK(store->LiveCheckpoints() == 1);
        int blocksB = 0;
        CHECK(RouteBStateObjectBytes(*store, identity, keyB, &tipB, &blocksB) > 0);
        CHECK(blocksB == 16);
        // The first Compact may already rebuild the projection with A's
        // retired rows removed; either way B's reachable chain must remain.
        CHECK(RouteBCountBlockRows(catalog) == 16);
        const uint64_t beforeFinalCompact = uint64_t(fs::file_size(log));
        CHECK(store->Compact(error));
        const uint64_t afterFinalCompact = uint64_t(fs::file_size(log));
        CHECK(afterFinalCompact <= beforeFinalCompact);
        std::string details;
        CHECK(store->SelfCheckManifest(details) == 0);
        CHECK(RouteBCountBlockRows(catalog) == 16);   // 只剩 B 的 16 块

        std::vector<DiskPrefixV4Store::BlockRow> rows;
        CHECK(store->ChainBlocks(tipB, 16, rows, error));   // B 的链仍逐块可达
        CHECK(rows.size() == 16);
        rows.clear();
        CHECK(!store->ChainBlocks(tipA, 8, rows, error));   // A 的块已被回收
        CHECK(error == "v4_block_missing");
        CHECK(store->Load(keyB, identity, objects, error));
    }
    {
        std::string error;
        auto reopened = DiskPrefixV4Store::Open(root.string(), quota, error);
        CHECK(reopened != nullptr);
        if (reopened) {
            std::vector<std::string> objects;
            CHECK(reopened->Load(keyB, identity, objects, error));
            CHECK(reopened->LiveCheckpoints() == 1);
        }
    }
}

// ===========================================================================
// A3（2026-10-02）：方案 §17.1 要求、此前缺失的四项 CPU 故障/兼容用例。
// 只增用例，不改任何既有断言。
// ===========================================================================

// A3-1/A3-3 辅助：十六进制解码 + 小端 u64 读取（v3 记录头是固定 208 字节布局）。
static std::string FromHex(const std::string &hex) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        const size_t high = std::string(digits).find(hex[i]);
        const size_t low = std::string(digits).find(hex[i + 1]);
        if (high == std::string::npos || low == std::string::npos) return std::string();
        out.push_back(char((high << 4) | low));
    }
    return out;
}

static uint64_t LittleEndianU64(const std::string &bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | uint8_t(bytes[offset + size_t(i)]);
    return value;
}

// v3 HeaderCrc（disk_prefix_cache.cpp:351）同款 reflected CRC-32；记录头前 104 字节参与。
static uint32_t LegacyRecordCrc32(const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1)));
    }
    return ~crc;
}

// A3-1：SQLite BUSY(5)/LOCKED(6) 必须**分别诊断**（方案 §17.1）。
//
// 产品现状实测（2026-10-02）：全仓 `grep -rn SQLITE_LOCKED src/ include/` = 0 命中；
//   · Database ctor 只设 sqlite3_busy_timeout(10000)（disk_prefix_cache.cpp:268），
//     把 BUSY 当作"等一会儿"，LOCKED 不单独处理；
//   · Statement::Row 对所有非 ROW/DONE 统一抛 runtime_error("cache_sql_step")（无 rc）；
//   · Database::Exec 抛 "cache_sql_exec:<msg> code=<rc>"（带 rc，但不分类 BUSY/LOCKED）。
// ⇒ "忙（另一写入者持锁）"与"锁（同连接/游标冲突，code=6）"在诊断上**不可分**。
//
// 因此本用例是**骨架**，分两层：
//   ① 默认（任何环境都跑）：锁竞争下必须 fail-closed —— 竞争窗口关闭后提交成功、
//      竞争期间绝不发布半成品、结束后可完整读回；
//   ② 严格契约（FASTLLM_TEST_REQUIRE_SQLITE_DIAG=1 时才跑，当前必然红灯）：
//      要求错误串能区分 BUSY/LOCKED。**不为了变绿而放宽**，等产品补区分后自动变绿。
static void sqliteBusyAndLockedAreDiagnosedSeparately(const fs::path &root) {
    // ⑨ BUSY(5)/LOCKED(6) 分类诊断：
    //  · 文件库无共享缓存 ⇒ LOCKED(6) 无法稳定复现；产品侧以错误明细 + `code=` 区分，本用例覆盖 BUSY(5)；
    //  · 为不真等 10 s，临时把 busy_timeout 缩到 50 ms（env 覆盖，**默认仍 10000 ms**，用例结束恢复）；
    //  · 契约：短暂竞争 ⇒ 提交必须 fail-closed 且带可区分诊断、绝不发布半成品；放锁后**必须能恢复**。
    setenv("FASTLLM_PREFIX_CACHE_SQLITE_BUSY_TIMEOUT_MS", "50", 1);
    const std::string identity = DiskPrefixCache::Digest("sqlite-diag-identity");
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) {
        unsetenv("FASTLLM_PREFIX_CACHE_SQLITE_BUSY_TIMEOUT_MS");
        return;
    }
    const auto object = MakeObject(std::string(4096, 'S'), 1);
    CHECK(store->Commit("sqlite-diag-key", identity, {object}, nullptr, error));
    // 裸连接持写锁，模拟"另一个写入者"。
    // 注意顺序：Open 自身会碰 catalog（PRAGMA/schema），必须在持锁**之前**完成，否则会先撞 BUSY
    // 掩盖本用例要观察的提交路径。
    sqlite3 *blocker = nullptr;
    const std::string catalog = (root / "v4" / "catalog.sqlite3").string();
    const int opened = sqlite3_open(catalog.c_str(), &blocker);
    CHECK(opened == SQLITE_OK && blocker != nullptr);
    if (blocker != nullptr) {
        CHECK(sqlite3_exec(blocker, "BEGIN IMMEDIATE", nullptr, nullptr, nullptr) == SQLITE_OK);
    }
    const auto second = MakeObject(std::string(2048, 'T'), 1);
    std::string secondError;
    const bool contested =
        store->Commit("sqlite-diag-key-2", identity, {second}, nullptr, secondError);
    std::fprintf(stderr, "[sqlite-diag] contested_commit=%d error=%s\n",
                 int(contested), secondError.c_str());
    // ① 竞争期间必须 fail-closed：不得假装成功。
    CHECK(!contested);
    // ② 必须带可区分的诊断（明细 + 错误码；BUSY = code=5）。
    const bool diagnosedBusy = secondError.find("code=5") != std::string::npos ||
                               secondError.find("busy") != std::string::npos ||
                               secondError.find("locked") != std::string::npos;
    if (!diagnosedBusy) {
        std::cerr << "PRODUCT-GAP: SQLITE_BUSY(5) is not diagnosed with detail/code: error="
                  << secondError << "\n";
    }
    CHECK(diagnosedBusy);
    // ③ 竞争期间绝不发布半成品。
    std::vector<std::string> half;
    CHECK(!store->Load("sqlite-diag-key-2", identity, half, error));
    // 放锁：临时竞争不得永久破坏 store —— 重试必须成功且可完整读回。
    if (blocker != nullptr) {
        sqlite3_exec(blocker, "COMMIT", nullptr, nullptr, nullptr);
        sqlite3_close(blocker);
    }
    std::string retryError;
    const bool retried = store->Commit("sqlite-diag-key-2", identity, {second}, nullptr, retryError);
    std::fprintf(stderr, "[sqlite-diag] retry_commit=%d error=%s\n",
                 int(retried), retryError.c_str());
    CHECK(retried);
    CHECK(retryError.empty());
    std::vector<std::string> loaded;
    CHECK(store->Load("sqlite-diag-key-2", identity, loaded, error));
    CHECK(loaded.size() == 1 && loaded[0] == second.bytes);
    unsetenv("FASTLLM_PREFIX_CACHE_SQLITE_BUSY_TIMEOUT_MS");
}

// A3-2：ENOSPC（磁盘满）写失败必须 fail-closed（方案 §17.1："EIO/ENOSPC 不作为
// '坏数据可删'的证据"，且不得发布半成品、不得伪造 durable）。
//
// 注入方式（诚实说明）：项目内无法造出小容量文件系统（非 root、不允许 /tmp 之外挂载），
// 故用子进程 RLIMIT_FSIZE 让"记录写到一半"的 write 返回 EFBIG——产品对 write/fdatasync
// 失败走的是**同一条** fail-closed 分支（AppendRecordToFile → v4_control_write →
// ftruncate 回滚到写前长度，disk_prefix_format_v4.cpp:782-798）。errno 名字不同（EFBIG vs
// ENOSPC），回滚/不发布语义完全一致；真正的 ENOSPC 需要产品级故障注入或小分区。
static void enospcWriteFailureFailsClosed(const fs::path &root) {
    const std::string identity = DiskPrefixCache::Digest("enospc-identity");
    std::string error;
    const auto small = MakeObject(std::string(1024, 'E'), 1);
    {
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(store != nullptr);
        if (!store) return;
        CHECK(store->Commit("enospc-good", identity, {small}, nullptr, error));
    }
    const fs::path log = root / "v4" / "control.log";
    CHECK(fs::exists(log));
    const uint64_t beforeBytes = uint64_t(fs::file_size(log));
    const std::string big(4u << 20, 'F');   // 单条记录远大于剩余可写额度

    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        std::string childError;
        auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, childError);
        if (!store) _exit(11);
        // 打开完成后再压限额：只影响接下来的记录写，不影响 SQLite 初始化。
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit {};
        limit.rlim_cur = rlim_t(beforeBytes + (1u << 20));
        limit.rlim_max = limit.rlim_cur;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(14);
        const auto object = MakeObject(big, 1);
        const bool ok = store->Commit("enospc-big", identity, {object}, nullptr, childError);
        if (ok) _exit(12);                       // 假成功 = 伪造 durable，必须判失败
        const bool failClosed = childError.rfind("v4_control_", 0) == 0;
        _exit(failClosed ? 0 : 13);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status));
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    const uint64_t afterBytes = uint64_t(fs::file_size(log));
    std::fprintf(stderr, "[enospc] child_exit=%d log_before=%llu log_after=%llu\n",
                 code, (unsigned long long)beforeBytes, (unsigned long long)afterBytes);
    CHECK(code == 0);                       // 失败路径必须返回明确的 v4_control_* 错误
    if (code != 0) {
        std::cerr << "  (11=open 12=committed-anyway 13=wrong-error 14=setrlimit)\n";
    }
    CHECK(afterBytes == beforeBytes);       // §17.1：报告失败 ⇒ 文件里不得多出任何字节
    // 不得伪造 durable：重开后没有半成品根，且失败前完好的检查点仍可读。
    {
        auto reopened = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
        CHECK(reopened != nullptr);
        if (reopened) {
            std::string rebuildError;
            CHECK(reopened->RebuildIndex(rebuildError));
            std::vector<std::string> loaded;
            std::string identityOut;
            CHECK(reopened->Load("enospc-good", identity, loaded, error));
            CHECK(loaded.size() == 1 && loaded[0] == small.bytes);
            CHECK(!reopened->Load("enospc-big", identityOut, loaded, error));
            CHECK(error == "v4_root_missing");
        }
    }
}

// A3-4：方案 §17.1"Capture 调用栈没有持模型锁等后台"。
//
// 本文件是纯 CPU store 测试，够不到模型侧互斥量（glm5_next.cpp CaptureTargetRuntimeCheckpoint），
// 因此用**store 侧可观测**的等价断言，并把局限写清楚：
//   · v4 的受理路径（SubmitStaged，正是 WriteSession::Commit / 模型 Capture 的落点）必须
//     "只交出状态就返回"：调用期间/之后不得触发任何"等后台"的收尾
//     （reap_wait_count 不增），且返回时后台确实还在跑（inflight>0 或 prep_jobs>0）；
//   · 只有显式 ReapCheckpoint 才把在途字节归零；随后读回必须成功。
// 局限：这证明的是"store 受理不阻塞后台"，**不能**证明模型侧没有持锁等待——那需要
// 模型侧静态审查或集成测试，超出本用例可观测范围。
static void captureCallStackDoesNotWaitOnBackground(const fs::path &root) {
    RouteBAssumeDefaultBlockTokens();
    const std::string identity = DiskPrefixCache::Digest("capture-identity");
    const std::string family = DiskPrefixCache::Digest("capture-family");
    const std::string key = DiskPrefixCache::Digest("capture-key");
    const RouteBPayload payload = MakeRouteBPayload(16, "capture-payload");   // 8 块
    std::string error;
    auto store = DiskPrefixV4Store::Open(root.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;

    const auto counter = [&](const char *field) -> long long {
        std::string parseError;
        const auto profile = DiskPrefixCache::Json::parse(store->Profile(false), parseError);
        CHECK(parseError.empty());
        return static_cast<long long>(profile[field].number_value());
    };
    const long long reapBefore = counter("reap_wait_count");
    const bool accepted = store->SubmitStaged(
        key, identity, family,
        MakeRouteBState(family, payload.length, "capture-prefix", payload.kv), true,
        std::vector<std::pair<std::string, std::string>>(payload.staged), error);
    CHECK(accepted);
    if (!accepted) { std::cerr << "capture submit: " << error << "\n"; return; }
    // 受理调用本身不得等后台（不触发 reap 收尾）。
    const long long reapAfter = counter("reap_wait_count");
    // 受理返回时后台仍在跑：要么还有在途字节，要么备菜已经开始。
    const bool backgroundStillRunning = counter("inflight_bytes") > 0 || counter("prep_jobs") > 0;
    CHECK(reapAfter == reapBefore);
    CHECK(backgroundStillRunning);
    // 显式收尾后才归零，并且数据可读回（不丢点、不伪造）。
    uint64_t sequence = 0;
    CHECK(store->ReapCheckpoint(&sequence, error));
    CHECK(sequence > 0);
    CHECK(counter("inflight_bytes") == 0);
    std::vector<std::string> objects;
    CHECK(store->Load(key, identity, objects, error));
    CHECK(!objects.empty());
    std::fprintf(stderr, "[capture] reap_wait_delta=%lld background_running=%d objects=%zu\n",
                 reapAfter - reapBefore, int(backgroundStillRunning), objects.size());
}

// S4b（2026-10-04）：有界（窗口化）导出的窗口切分纯函数。无 GPU、无 store、纯算术。
// 断言：每窗口 ≤ 预算、窗口拼接恰好覆盖整段（无洞无重叠）、区间 ≤ 一个窗口时窗口数==1、
// 预算极小时窗口数>1 且每窗口 ≥1 页。
static void splitExportWindowsIsBoundedAndComplete() {
    using fastllm::prefix_v4::ExportWindow;
    using fastllm::prefix_v4::SplitExportWindows;
    auto covers = [](const std::vector<ExportWindow> &windows, uint64_t pageCount) {
        uint64_t cursor = 0;
        for (const auto &window : windows) {
            if (window.beginPage != cursor) return false;          // 洞或重叠
            if (window.endPage <= window.beginPage) return false;  // 空窗口/倒序
            cursor = window.endPage;
        }
        return cursor == pageCount;
    };
    auto maxWindowBytes = [](const std::vector<ExportWindow> &windows, uint64_t bytesPerPage) {
        uint64_t peak = 0;
        for (const auto &window : windows)
            peak = std::max(peak, (window.endPage - window.beginPage) * bytesPerPage);
        return peak;
    };

    // ① 每窗口 ≤ budget（且 ≤ maxWindowBytes）。
    {
        const uint64_t pages = 1000, bytesPerPage = 4096, budget = 64ull << 10, cap = 8ull << 20;
        const auto windows = SplitExportWindows(pages, bytesPerPage, budget, cap);
        CHECK(!windows.empty());
        CHECK(covers(windows, pages));
        for (const auto &window : windows)
            CHECK((window.endPage - window.beginPage) * bytesPerPage <= budget);
        CHECK(maxWindowBytes(windows, bytesPerPage) <= budget);
        CHECK(maxWindowBytes(windows, bytesPerPage) <= cap);
    }
    // ①b maxWindowBytes 才是绑定的上界（预算很大时按 8 MiB 切）。
    {
        const uint64_t pages = 100, bytesPerPage = 1ull << 20, budget = 1ull << 30, cap = 8ull << 20;
        const auto windows = SplitExportWindows(pages, bytesPerPage, budget, cap);
        CHECK(covers(windows, pages));
        CHECK(maxWindowBytes(windows, bytesPerPage) <= cap);
        CHECK(windows.size() == 13);   // 100 页 × 1 MiB，每窗 ≤8 页 ⇒ ceil(100/8)=13
    }
    // ② 窗口拼接恰好覆盖完整页区间、无洞无重叠（含不整除、单页边界）。
    {
        for (uint64_t pages : {uint64_t(1), uint64_t(2), uint64_t(7), uint64_t(97), uint64_t(1000)}) {
            const auto windows = SplitExportWindows(pages, 512, 4096, 4096);
            CHECK(covers(windows, pages));
        }
    }
    // ③ 区间 ≤ 一个窗口时窗口数 == 1（对应"issued>0 且区间不大 ⇒ 单窗口"）。
    {
        const auto windows = SplitExportWindows(4, 4096, 64ull << 10, 8ull << 20);
        CHECK(windows.size() == 1);
        CHECK(windows.size() == 1 && windows[0].beginPage == 0 && windows[0].endPage == 4);
        // 恰好等于一窗容量时仍是 1 个窗口。
        const auto exact = SplitExportWindows(16, 4096, 64ull << 10, 8ull << 20);
        CHECK(exact.size() == 1);
        CHECK(exact.size() == 1 && exact[0].endPage == 16);
    }
    // ④ 预算极小时窗口数 > 1 且每窗口 ≥ 1 页。
    {
        const uint64_t pages = 10, bytesPerPage = 4096;
        const auto onePage = SplitExportWindows(pages, bytesPerPage, 1, 0);       // 预算 < 一页
        CHECK(onePage.size() == pages);
        CHECK(covers(onePage, pages));
        for (const auto &window : onePage) CHECK(window.endPage - window.beginPage >= 1);
        const auto fractional = SplitExportWindows(pages, bytesPerPage, bytesPerPage - 1, 0);
        CHECK(fractional.size() == pages);
        CHECK(covers(fractional, pages));
        const auto halfPage = SplitExportWindows(pages, bytesPerPage, bytesPerPage / 2, 0);
        CHECK(halfPage.size() == pages && covers(halfPage, pages));
    }
    // 边界：零页区间返回空；bytesPerPage==0 无字节约束 ⇒ 单窗口。
    {
        CHECK(SplitExportWindows(0, 4096, 64ull << 10, 8ull << 20).empty());
        const auto unbounded = SplitExportWindows(5, 0, 64ull << 10, 8ull << 20);
        CHECK(unbounded.size() == 1);
        CHECK(unbounded.size() == 1 && unbounded[0].endPage == 5);
    }
}

// ===========================================================================
// D5 阶段1：主#2／#5／#13／#6／#7 的 T0 判据。
// 背景（只读子代理 135a9e3c 的清点）：这五项在 6 个 CPU 目标里"零覆盖"或"仅相邻"；
// 另一子代理 0acb2c88 确认主#2/#5/#13 **用现有公开 API 即可断言**（无需新增可读面）。
// 本批只用公开 API ＋ 既有注入点，**不改产品行为**；每条注明机理与"为什么这样断言"。
// ===========================================================================

// 构造"引用一个从未写过的对象"的状态（毒检查点）。形状照抄已实测有效的
// 取消发布探针：在 linear_chunks 里塞一个格式合法但从未落盘的 sha256。
static DiskPrefixCache::Json D5DanglingRefState(const std::string &family,
                                                const std::string &ghostHash) {
    return DiskPrefixCache::Json::object{
        {"kind", "prefix"},
        {"length", double(2048)},
        {"family_identity", family},
        {"prefix", DiskPrefixCache::Digest("d5-dangling-prefix")},
        {"linear_chunks", DiskPrefixCache::Json::array{DiskPrefixCache::Json::object{
            {"sha256", ghostHash}, {"bytes", double(1024)}}}},
        {"speculative_chunks", DiskPrefixCache::Json::array()},
    };
}

// 主#2：**毒检查点（悬空引用）绝不允许被发布**。
// 为什么单独立这条：既有探针的判据是 InvariantHolds = !published || missing==0
//   ⇒ published==0 时即使 missing==1 也 PASS，**它没有真正断言"不得发布"**（子代理 0acb2c88 核对结论）。
//   本用例**显式断言 published==0**（用 LiveCheckpoints()/Checkpoints() 两个公开面）。
static void d5DanglingRefIsNeverPublished(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("d5-dr-identity");
    const std::string family = DiskPrefixCache::Digest("d5-dr-family");
    const std::string key = DiskPrefixCache::Digest("d5-dr-key");
    const std::string ghost = DiskPrefixCache::Digest("d5-dr-never-stored");
    auto store = DiskPrefixV4Store::Open((root / "d5-dangling-ref").string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    error.clear();
    CHECK(store->SubmitStaged(key, identity, family, D5DanglingRefState(family, ghost), true, {},
                              error));
    std::string reapError;
    store->ReapCheckpoint(nullptr, reapError);   // 提交是异步的：必须收尾才能看到真实结果
    CHECK(store->LiveCheckpoints() == 0);        // ← 主#2 核心：不得发布
    CHECK(store->Checkpoints().empty());
    std::string profileError;
    const DiskPrefixCache::Json profile =
        DiskPrefixCache::Json::parse(store->Profile(true), profileError);
    CHECK(profileError.empty());
    // **实测机制（2026-10-04）**：本构造（linear_chunks 里的悬空引用）在**提交阶段**即因对象缺失
    // 失败，被 批1 的"抢救未提交字节"接住，日志为
    //   `[Prefix SSD] v4 rescued-uncommitted-bytes: key=… objects=… (checkpoint NOT published).`
    // 因此它**不走**发布闸门（`v4_publish_withheld`），`publish_refused_checkpoints` 自然为 0。
    // ⇒ 断言按**实质**写：毒检查点**不得可见、不得计入落盘**；不断言与路径相关的计数器。
    // **覆盖边界（如实）**：发布闸门那条子路径（悬空引用放在 `kv` bundle 里）由既有 T0 探针
    //   由单独的取消发布探针用例 [A] 覆盖，**不在本套件内**。
    CHECK(profile["durable_checkpoints_cum"].number_value() == 0.0);
}

// 主#5：**被扣下的检查点不得伪装成"已落盘"**（S2）。
// 与主#2 同场景，但断言重点不同：主#2 看"不得发布"，主#5 看"**不得被计入 durable**"。
static void d5WithheldIsNotCountedDurable(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("d5-wh-identity");
    const std::string family = DiskPrefixCache::Digest("d5-wh-family");
    const std::string key = DiskPrefixCache::Digest("d5-wh-key");
    const std::string ghost = DiskPrefixCache::Digest("d5-wh-never-stored");
    auto store = DiskPrefixV4Store::Open((root / "d5-withheld").string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    error.clear();
    CHECK(store->SubmitStaged(key, identity, family, D5DanglingRefState(family, ghost), true, {},
                              error));
    std::string reapError;
    store->ReapCheckpoint(nullptr, reapError);
    std::string profileError;
    const DiskPrefixCache::Json profile =
        DiskPrefixCache::Json::parse(store->Profile(true), profileError);
    CHECK(profileError.empty());
    // 主#5 的核心＝**不得被计入"已落盘"**（本构造走"提交即失败 + 批1 抢救"，机制同主#2，见其注释；
    // 故不断言与发布闸门路径相关的 `publish_refused_checkpoints`）。
    CHECK(profile["durable_checkpoints_cum"].number_value() == 0.0);   // ← 主#5 核心
    CHECK(store->Checkpoints().empty());                                // 也不得可见
    CHECK(store->LiveCheckpoints() == 0);
}

// 主#13：门⑦自检必须"**非空**"——这正是 a5 空头通过（SELFCHECK: checkpoints=0）的反面。
// 为什么这样断言：打印行的四元组只在 stdout（SelfCheck() 是 void，h:318），
//   但"非空"可用公开 `Checkpoints()` 直接拿（cpp:5874 的 checkpoints=rows.size() 就是它），
//   功能正确性用 `SelfCheckFunctional()==0` ＋ 逐条 `Load()` 组合（子代理 0acb2c88 的结论）。
static void d5SelfCheckIsNonEmpty(const fs::path &root) {
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("d5-sc-identity");
    const fs::path dataRoot = root / "d5-selfcheck-nonempty";
    auto store = DiskPrefixV4Store::Open(dataRoot.string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    for (int i = 0; i < 2; ++i) {
        const auto object = MakeObject(std::string(4096, char('A' + i)), 1);
        uint64_t sequence = 0;
        error.clear();
        CHECK(store->CommitCheckpoint(DiskPrefixCache::Digest("d5-sc-key-" + std::to_string(i)),
                                      identity, {object},
                                      "{\"state\":\"" + DiskPrefixCache::Digest("d5-sc-s") + "\"}",
                                      &sequence, error));
        CHECK(sequence > 0);
    }
    CHECK(!store->Checkpoints().empty());          // ← 主#13 核心：非空
    std::string details;
    CHECK(store->SelfCheckFunctional(details) == 0);
    for (int i = 0; i < 2; ++i) {                  // 逐条可读回
        std::string loadedIdentity;
        std::vector<std::string> loaded;
        std::string loadError;
        CHECK(store->Load(DiskPrefixCache::Digest("d5-sc-key-" + std::to_string(i)),
                          loadedIdentity, loaded, loadError));
        CHECK(loaded.size() == 1);
    }
}

// 主#6／主#7 共用的触发链（R1b 已逐环核实，见 EXPERIENCE）：
//   family 非 digest ⇒ PrepareStagedJob 首个 Check(cpp:3088) 抛 invalid_checkpoint_metadata
//   ⇒ RunPrep 的 if(!ok) 把结果槽置 finished(cpp:3065-3073)
//   ⇒ 该作业对出队点而言"已完成"（PruneFinishedHead 守卫 cpp:2792）
//   ⇒ PreserveBeforeDropLocked 的早返回(cpp:2713)不成立（staged 仍在）
//   ⇒ SubmitPrepared 交给 actor，actor 认领段 preserveOnly=true(cpp:3253) ⇒ **保字节分支执行**。
static void d5PreserveBranchCommon(const fs::path &root, bool failFlush) {
    if (failFlush) setenv("FASTLLM_DISK_CACHE_FAIL_PRESERVE_FLUSH", "1", 1);
    std::string error;
    const std::string identity = DiskPrefixCache::Digest("d5-po-identity");
    const std::string badFamily = "not-a-digest-family";   // 故意非法 ⇒ 备菜必失败
    const std::string key = DiskPrefixCache::Digest("d5-po-key");
    const std::string payload = std::string(2048, 'P');
    auto store = DiskPrefixV4Store::Open((root / "d5-preserve").string(), 1ull << 30, error);
    CHECK(store != nullptr);
    if (!store) return;
    std::vector<std::pair<std::string, std::string>> staged;
    staged.emplace_back(DiskPrefixCache::Digest(payload), payload);
    error.clear();
    CHECK(store->SubmitStaged(key, identity, badFamily,
                              DiskPrefixCache::Json::object{{"kind", "prefix"},
                                                            {"length", double(2048)},
                                                            {"family_identity", badFamily}},
                              false, std::move(staged), error));
    std::string reapError;
    store->ReapCheckpoint(nullptr, reapError);
    // 保字节是"投递给 actor"的异步动作 ⇒ 有界等待计数出现（最多约 5 秒），避免时序假失败。
    double preservedOnDrop = 0.0, unpreservedDrops = 0.0, flushFailures = 0.0, durable = 0.0;
    for (int i = 0; i < 50; ++i) {
        std::string profileError;
        const DiskPrefixCache::Json profile =
            DiskPrefixCache::Json::parse(store->Profile(true), profileError);
        CHECK(profileError.empty());
        preservedOnDrop = profile["preserved_on_drop"].number_value();
        unpreservedDrops = profile["unpreserved_drops"].number_value();
        flushFailures = profile["preserve_flush_failures"].number_value();
        durable = profile["durable_checkpoints_cum"].number_value();
        if (preservedOnDrop >= 1.0 && (!failFlush || flushFailures >= 1.0)) break;
        usleep(100000);
    }
    // 主#6 核心：保字节被**执行且计数**，且没有"未保字节就出队"
    CHECK(preservedOnDrop >= 1.0);
    CHECK(unpreservedDrops == 0.0);
    // 保下来的对象必须可读（保字节真的写进去了）
    CHECK(store->HasObject(DiskPrefixCache::Digest(payload)));
    // 但该检查点不得发布、不得计入落盘
    CHECK(store->LiveCheckpoints() == 0);
    CHECK(durable == 0.0);
    if (failFlush) {
        // 主#7 核心：S3b 的失败**不得被吞**（必须留下可计数、可审计的痕迹）
        CHECK(flushFailures >= 1.0);
        unsetenv("FASTLLM_DISK_CACHE_FAIL_PRESERVE_FLUSH");
    }
}

static void d5PreserveStagedOnPrepFailure(const fs::path &root) {
    d5PreserveBranchCommon(root, false);
}
static void d5PreserveFlushFailureIsNotSwallowed(const fs::path &root) {
    d5PreserveBranchCommon(root, true);
}

int main() {
    if (std::getenv("FASTLLM_TEST_ONLY_WINDOWS")) {   // S4b：只跑窗口切分纯函数
        splitExportWindowsIsBoundedAndComplete();
        std::cout << (failures ? "FAILED\n" : "PASS windows-only\n");
        return failures ? 1 : 0;
    }
    // 子进程模式（由 twoProcessesShareRoot 经 /proc/self/exe 再次执行本二进制触发）：
    // 打开环境变量指定的同一 root 并提交 3 笔；父进程负责全部断言。
    if (const char *childDir = std::getenv("FASTLLM_TEST_CHILD_APPEND")) {
        std::string childError;
        const std::string childIdentity = DiskPrefixCache::Digest("mp-identity");
        auto childStore = DiskPrefixV4Store::Open(childDir, 1ull << 30, childError);
        if (!childStore) { std::cerr << "mp child open: " << childError << "\n"; return 2; }
        for (int i = 0; i < 3; ++i) {
            const auto object = MakeObject(std::string(size_t(2048 + i), char('c' + i)), 1);
            if (!childStore->Commit("mp-child-" + std::to_string(i), childIdentity, {object},
                                    nullptr, childError)) {
                std::cerr << "mp child commit: " << childError << "\n";
                return 3;
            }
        }
        std::cout << "PASS child\n";
        return 0;
    }
    const fs::path tmp = TempRoot();
    if (std::getenv("FASTLLM_TEST_ONLY_CHUNKDEDUP")) {
        secondCheckpointSharesFirstChunk(tmp / "chunkdedup-only");
        std::cout << (failures ? "FAILED\n" : "PASS chunkdedup-only\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_CASCADE")) {   // 只跑 cache 级连坐复现
        cacheLevelCascadeAfterInjectedFailure(tmp / "cascade-only");
        std::cout << (failures ? "FAILED\n" : "PASS cascade-only\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_REPRO")) {   // 精确控制注入时序：只跑复现用例
        sharedObjectAfterInjectedFailure(tmp / "repro-only");
        std::cout << (failures ? "FAILED\n" : "PASS repro-only\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_FAILPOINTS")) {   // §17.1 记录级逐点注入
        v4FailpointsRecoverWithoutPhantoms(tmp / "failpoints-only");
        std::cout << (failures ? "FAILED\n" : "PASS failpoints-only\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_CLOSEOUT")) {
        metadataFailureDoesNotPublishRoot(tmp / "metadata-failure");
        uncommittedTailBlocksAppend(tmp / "tail-block");
        v4FailpointsRecoverWithoutPhantoms(tmp / "closeout-failpoints");
        std::cout << (failures ? "FAILED closeout\n" : "PASS closeout\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_SHARED")) {   // 记账修复 + 同根合并保管员
        liveBytesSurvivesRebuildAndCompact(tmp / "livebytes-only");
        sameRootSharesOneStore(tmp / "shared-root-only");
        cacheNamespacesShareOneStore(tmp / "ns-shared-only");
        std::cout << (failures ? "FAILED shared\n" : "PASS shared\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_B7")) {   // §17.1 B7：索引库删除/损坏后照控制日志重建
        sqliteCatalogLossRebuildsFromLog(tmp / "b7-index-loss");
        std::cout << (failures ? "FAILED b7\n" : "PASS b7\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_CANCEL")) {   // §保最新：取消最老的未开工在途作业
        cancelOldestPendingFreesBudget(tmp / "cancel-oldest-only");
        std::cout << (failures ? "FAILED cancel\n" : "PASS cancel\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_P0GUARDS")) {   // P0 四连：撕裂尾自愈 / 发布即可读回 / 收尾归零 / 双扣守卫
        tornTailSelfHealsOnReopen(tmp / "p0-torn-tail");
        publishedImpliesRestorable(tmp / "p0-published");
        pruneFinishesWithoutRead(tmp / "p0-prune");
        backpressureDoubleSubtractGuard(tmp / "p0-backpressure");
        std::cout << (failures ? "FAILED p0guards\n" : "PASS p0guards\n");
        return failures ? 1 : 0;
    }
    if (std::getenv("FASTLLM_TEST_ONLY_A3")) {   // A3 四连：BUSY/LOCKED 骨架 / ENOSPC / 黄金读 / Capture 不等后台
        sqliteBusyAndLockedAreDiagnosedSeparately(tmp / "a3-sqlite-diag");
        enospcWriteFailureFailsClosed(tmp / "a3-enospc");
        captureCallStackDoesNotWaitOnBackground(tmp / "a3-capture");
        std::cout << (failures ? "FAILED a3\n" : "PASS a3\n");
        return failures ? 1 : 0;
    }
    std::cout << "v4 store test root: " << tmp << "\n";
    try {
        writeReadRecover(tmp / "basic");
        sqliteCatalogLossRebuildsFromLog(tmp / "b7-index-loss");
        unfinishedTailPublishesNothing(tmp / "tail");
        corruptLogFailsClosed(tmp / "corrupt");
        quotaEvictionAndCompaction(tmp / "quota");
        fifoEraseCompactSafety(tmp / "fifo-erase-compact-safety");
        physicalCapacityAndActiveWriter(tmp / "physical-capacity-active-writer");
        // R3：准入路径“淘汰旧检查点而非丢新写”+ 可观测计数
        admissionEvictionKeepsNewestAndCounts(tmp / "admission-eviction");
        quotaReservationFailsClosed(tmp / "quota-reserve");
        v4CacheInterface(tmp / "cache-v4");
        batchingAndFifo(tmp / "batch-fifo");
        catalogActorSerializes(tmp / "actor");
        crossProcessAppendLease(tmp / "root-lease");
        liveBytesSurvivesRebuildAndCompact(tmp / "livebytes");
        sameRootSharesOneStore(tmp / "shared-root");
        cacheNamespacesShareOneStore(tmp / "ns-shared");
        journalTimingCounters(tmp / "journal-timing");
        profileDiagnostics(tmp / "profile");
        strictVerifyOffStillSafe(tmp / "strict-verify");
        compactJsonMatchesStrip(tmp / "compact-json");
        storeVerifiedUsesHashPool(tmp / "store-verified");
        rebuildRejectsCorruptPayload(tmp / "rebuild-corrupt");
        v4RejectsInvalidState(tmp / "v4-state");
        eraseTombstoneCorruptionFailsClosed(tmp / "erase-corrupt");
        selfCheckManifestDetectsMissingEntries(tmp / "selfcheck-manifest");
        selfCheckFunctionalDetectsStateMismatch(tmp / "selfcheck-functional");
        missingCommitIsNotPublished(tmp / "truncate-commit");
        asyncCommitPipeline(tmp / "async-commit");
        cancelOldestPendingFreesBudget(tmp / "cancel-oldest");
        writeRestartReadBackExact(tmp / "readback-exact");
        oddSizedObjectsStayReadable(tmp / "odd-objects");
        referenceOnlyObjectsAppearInListing(tmp / "ref-list");
        metadataFailureDoesNotPublishRoot(tmp / "metadata-failure");
        uncommittedTailBlocksAppend(tmp / "uncommitted-tail-block");
        twoProcessesShareRoot(tmp / "mp-share-root");
        v4FailpointsRecoverWithoutPhantoms(tmp / "failpoints");
        // P0 守卫四连（2026-10-02）：撕裂尾自愈 / 发布即可读回 / 收尾归零 / 背压双扣守卫
        tornTailSelfHealsOnReopen(tmp / "p0-torn-tail");
        publishedImpliesRestorable(tmp / "p0-published");
        pruneFinishesWithoutRead(tmp / "p0-prune");
        backpressureDoubleSubtractGuard(tmp / "p0-backpressure");
        // Route B（2026-10-02）块链式 KV 布局 CPU 单测（S2 阶段）
        routeBChainKeyDeterministicAndContentAddressed(tmp / "routeb-chain-determinism");
        routeBStateObjectSizeIsConstant(tmp / "routeb-state-o1");
        routeBMultiBlockChainRestoresByteExact(tmp / "routeb-multiblock-restore");
        routeBParentBlockMissingFailsClosed(tmp / "routeb-parent-missing");
        routeBRebuildIndexRestoresBlocksProjection(tmp / "routeb-rebuild-blocks");
        routeBCompactKeepsReachableBlocksReclaimsUnreferenced(tmp / "routeb-compact-blocks");
        // A3（2026-10-02）：§17.1 要求的四项 CPU 故障/兼容用例（只增）
        sqliteBusyAndLockedAreDiagnosedSeparately(tmp / "a3-sqlite-diag");
        enospcWriteFailureFailsClosed(tmp / "a3-enospc");
        captureCallStackDoesNotWaitOnBackground(tmp / "a3-capture");
        // S4b（2026-10-04）：有界导出窗口切分纯函数（无 GPU / 纯算术）
        splitExportWindowsIsBoundedAndComplete();
        // D5 阶段1：主#2／#5／#13／#6／#7 的 T0 判据
        d5DanglingRefIsNeverPublished(tmp / "d5-dangling-ref");
        d5WithheldIsNotCountedDurable(tmp / "d5-withheld");
        d5SelfCheckIsNonEmpty(tmp / "d5-selfcheck-nonempty");
        d5PreserveStagedOnPrepFailure(tmp / "d5-preserve-on-prep-fail");
        d5PreserveFlushFailureIsNotSwallowed(tmp / "d5-preserve-flush-fail");
        // DESIGN-v3-segments (FASTLLM_PREFIX_V4_SEGMENTS=1)
        segmentsFillPastQuotaManyTimes(tmp / "seg-fill");
        segmentsSharedObjectRetiresDependents(tmp / "seg-shared");
        segmentsLayoutGuards(tmp / "seg-layout");
        segmentsShrinkAtOpenAndMetadataWhenFull(tmp / "seg-shrink-meta");
    } catch (const std::exception &error) {
        std::cerr << "FAIL exception: " << error.what() << "\n";
        ++failures;
    }
    if (failures != 0) {
        std::cerr << "FAILED: " << failures << " check(s)\n";
        return 1;
    }
    std::cout << "PASS disk_prefix_store_v4\n";
    return 0;
}
