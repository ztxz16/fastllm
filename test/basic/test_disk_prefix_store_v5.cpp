// V5 store tests (DESIGN-V5-draft3 sec. 9/10): CPU model scenarios (I1-I7 of
// logs/a6-v5-20261107/model) replayed on the real store through the same
// DiskPrefixCache facade the model uses, fault injection at every write step,
// threaded two-writer test, flock, corruption, torn tail and quota checks.
#include "utils/disk_prefix_cache.h"
#include "utils/disk_prefix_v5.h"

#include <openssl/evp.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace fastllm;
namespace fs = std::filesystem;
using Json = DiskPrefixCache::Json;

static int g_failures = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { ++g_failures; std::fprintf(stderr, "FAIL %s:%d: %s ", __FILE__, __LINE__, #cond); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static const int kPage = 128, kChunk = 256;
static const size_t kKvBytes = 8192, kStateBytes = 24576;

static std::string Sha(const std::string &v) { return DiskPrefixCache::Digest(v); }
static std::string TokText(const std::vector<int> &t, int n) {
    std::string s; s.reserve(size_t(n) * 4);
    for (int i = 0; i < n; ++i) { uint32_t x = uint32_t(t[size_t(i)]); s.append(reinterpret_cast<char *>(&x), 4); }
    return s;
}
static std::string PrefixDigest(const std::vector<int> &t, int n) { return Sha("prefix:" + TokText(t, n)); }
static std::string KeyOf(const std::vector<int> &t, int n) { return Sha(PrefixDigest(t, n)); }
static std::string Bytes(const std::string &seed, size_t n) {
    std::seed_seq seq(seed.begin(), seed.end());
    std::mt19937_64 rng(seq);
    std::string out(n, '\0');
    for (size_t i = 0; i + 8 <= n; i += 8) { uint64_t v = rng(); memcpy(&out[i], &v, 8); }
    return out;
}
static std::string KvTruth(const std::vector<int> &t, int chunk) { return Bytes("kv" + PrefixDigest(t, (chunk + 1) * kPage), kKvBytes); }
static std::string StateTruth(const std::vector<int> &t, int n, const char *what) { return Bytes(std::string(what) + PrefixDigest(t, n), kStateBytes); }

struct Env {
    std::string dir, identity = Sha("model-identity"), family = Sha("model-identity");
    uint64_t quota = 1ull << 40;
    std::shared_ptr<DiskPrefixCache> store;
    std::mutex mu;
    std::set<std::string> live;   // durable keys minus erased keys (expected index)
    void Open() {
        store = std::make_shared<DiskPrefixCache>(dir, identity, quota, family, 4);
        store->SetCheckpointEraseCallback([this](const std::string &key, const std::string &) {
            std::lock_guard<std::mutex> lock(mu); live.erase(key);
        });
    }
    void Close() { store.reset(); }
    std::set<std::string> Listed() {
        std::set<std::string> keys;
        for (const auto &c : store->ListCheckpoints("", family, INT32_MAX, "prefix", "")) keys.insert(c.key);
        return keys;
    }
    void Barrier() { std::string error; store->DrainPending(error); }
};

// Lookup exactly like Qwen35PersistentCache::Prepare: deepest stored length <= n-1
// with a matching prefix digest; then a full Load and byte-for-byte check (I1).
static int Restore(Env &env, const std::vector<int> &t, bool verify = true, std::string *errorOut = nullptr) {
    const int n = int(t.size());
    std::vector<std::pair<int, std::string>> wanted;
    for (int len : env.store->CheckpointLengths(env.family, n - 1, "prefix", true)) if (len > 0) wanted.push_back({len, PrefixDigest(t, len)});
    auto candidates = env.store->MatchPrefixes(env.family, wanted, "prefix", true);
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) { return a.length > b.length; });
    for (const auto &c : candidates) {
        if (c.key != KeyOf(t, c.length) || !DiskPrefixCache::IsDigest(c.metadata["state"].string_value())) continue;
        try {
            auto reader = env.store->BeginRead();
            Json state = reader.Load(c.key, c.identity);
            if (!verify) return c.length;
            const auto &kv = state["kv"].array_items();
            EXPECT(int(kv.size()) == c.length / kPage, "kv count %zu len %d", kv.size(), c.length);
            for (size_t j = 0; j < kv.size(); ++j) {
                std::string got(kKvBytes, '\0');
                reader.ReadBytes(kv[j]["chunks"], kKvBytes, [&](size_t off, const char *p, size_t m) { memcpy(&got[off], p, m); });
                EXPECT(got == KvTruth(t, int(j)), "I1 kv mismatch len=%d chunk=%zu", c.length, j);
            }
            for (const char *what : {"linear", "speculative"}) {
                std::string got(kStateBytes, '\0');
                reader.ReadBytes(state[std::string(what) + "_chunks"], kStateBytes, [&](size_t off, const char *p, size_t m) { memcpy(&got[off], p, m); });
                EXPECT(got == StateTruth(t, c.length, what), "I1 state mismatch len=%d %s", c.length, what);
            }
            return c.length;
        } catch (const std::exception &e) {
            if (errorOut) *errorOut = e.what();
            return -1;   // any failure = miss (never partial use)
        }
    }
    return 0;
}

// Save like Qwen35PersistentCache::Export: adopt the parent manifest, append
// chunk-end checkpoints relative to the restored length plus the prompt end.
static void Save(Env &env, const std::vector<int> &t, int restored, std::vector<std::string> *durable = nullptr) {
    const int n = int(t.size());
    const int end = n / kPage * kPage;
    if (end <= restored) return;
    Json::array kv;
    std::unique_ptr<DiskPrefixCache::ReadSession> inherited;
    if (restored > 0) {
        inherited.reset(new DiskPrefixCache::ReadSession(env.store->BeginRead()));
        try { kv = inherited->LoadManifest(KeyOf(t, restored))["kv"].array_items(); }
        catch (...) { return; }
    }
    auto writer = env.store->BeginWrite(0);
    inherited.reset();
    std::vector<int> points;
    for (int p = restored + kChunk; p <= end; p += kChunk) points.push_back(p);
    if (points.empty() || points.back() != end) points.push_back(end);
    int done = restored;
    for (int p : points) {
        for (int c = done / kPage; c < p / kPage; ++c) {
            const std::string bytes = KvTruth(t, c);
            auto chunks = writer.StoreBytes(bytes.size(), [&](size_t off, char *out, size_t m) { memcpy(out, bytes.data() + off, m); });
            kv.push_back(Json::object{{"bytes", double(bytes.size())}, {"segments", 0}, {"chunks", chunks}});
        }
        done = p;
        const std::string lin = StateTruth(t, p, "linear"), spec = StateTruth(t, p, "speculative");
        auto lc = writer.StoreBytes(lin.size(), [&](size_t off, char *out, size_t m) { memcpy(out, lin.data() + off, m); });
        auto sc = writer.StoreBytes(spec.size(), [&](size_t off, char *out, size_t m) { memcpy(out, spec.data() + off, m); });
        const std::string key = KeyOf(t, p);
        Json::object value{{"kind", "prefix"}, {"length", p}, {"family_identity", env.family}, {"kv_dtype", "fp8_e4m3"},
                           {"provenance", ""}, {"prefix", PrefixDigest(t, p)}, {"layout", Json::object{{"tp", 4}}},
                           {"kv", kv}, {"linear_chunks", lc}, {"speculative_chunks", sc},
                           {"v5_prompt_end", p + kPage > n}};
        try {
            writer.CommitTracked(key, Json(value), true, [&env, key, durable](bool ok) {
                if (!ok) return;
                std::lock_guard<std::mutex> lock(env.mu); env.live.insert(key);
                if (durable) durable->push_back(key);
            });
        } catch (const std::exception &) { return; }
    }
}

static std::vector<int> RandomTokens(std::mt19937 &rng, int n) {
    std::vector<int> t(static_cast<size_t>(n));
    for (auto &x : t) x = int(rng() % 50000);
    return t;
}
static std::string FreshDir(const std::string &base, const std::string &name) {
    const std::string dir = base + "/" + name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}
static Json Profile(Env &env) { return env.store->StorageProfile(false); }
static bool SameIndex(Env &env, const char *where) {
    const auto listed = env.Listed();
    std::lock_guard<std::mutex> lock(env.mu);
    if (listed == env.live) return true;
    for (const auto &k : listed) if (!env.live.count(k)) std::fprintf(stderr, "  %s: listed-not-live %s\n", where, k.substr(0, 12).c_str());
    for (const auto &k : env.live) if (!listed.count(k)) std::fprintf(stderr, "  %s: live-not-listed %s\n", where, k.substr(0, 12).c_str());
    return false;
}

// ---------------------------------------------------------------- tests
static void TestMultiTurnSlotsBranch(const std::string &base) {
    Env env; env.dir = FreshDir(base, "multiturn"); env.Open();
    std::mt19937 rng(1);
    auto t1 = RandomTokens(rng, 1000);
    EXPECT(Restore(env, t1) == 0, "cold miss");
    Save(env, t1, 0); env.Barrier();
    auto t2 = t1; auto a = RandomTokens(rng, 300); t2.insert(t2.end(), a.begin(), a.end());
    const int h2 = Restore(env, t2);
    EXPECT(h2 == 1000 / kPage * kPage, "turn2 hits turn1 prompt end, got %d", h2);
    Save(env, t2, h2); env.Barrier();
    auto p = Profile(env);
    EXPECT(p["new_files"].int_value() == 1 && p["appends"].int_value() >= 1, "continuation appends to the same file: %s", p.dump().c_str());
    EXPECT(p["slot_writes"].int_value() == 2, "two prompt-end states in slots: %s", p.dump().c_str());
    // Edit last question: shares turn1 prompt + first 150 answer tokens.
    auto e2 = t2; e2.resize(1150); auto q = RandomTokens(rng, 200); e2.insert(e2.end(), q.begin(), q.end());
    const int he = Restore(env, e2);
    EXPECT(he >= 1000 / kPage * kPage, "edit hits at least turn1 prompt end, got %d", he);
    Save(env, e2, he); env.Barrier();
    p = Profile(env);
    EXPECT(p["branches"].int_value() == 1, "edit becomes a branch file: %s", p.dump().c_str());
    // Turn 3 overwrites the oldest slot (turn1 prompt end) but interval states remain.
    auto t3 = t2; auto b = RandomTokens(rng, 400); t3.insert(t3.end(), b.begin(), b.end());
    const int h3 = Restore(env, t3);
    EXPECT(h3 == int(t2.size()) / kPage * kPage, "turn3 hits turn2 prompt end, got %d", h3);
    Save(env, t3, h3); env.Barrier();
    auto keys = env.Listed();
    EXPECT(!keys.count(KeyOf(t1, 896)), "oldest prompt-end slot displaced");
    EXPECT(keys.count(KeyOf(t1, 768)), "interval state kept");
    EXPECT(keys == env.live, "index == durable - erased (%zu vs %zu)", keys.size(), env.live.size());
    // Restart: everything durable is restorable (I3).
    env.Close(); env.Open();
    EXPECT(env.Listed() == env.live, "I3 index after restart");
    EXPECT(Restore(env, t3) == int(t3.size()) / kPage * kPage, "I3 latest prompt after restart");
    EXPECT(Restore(env, e2) >= he, "I3 branch after restart");
    // Junction (H24): a state-only checkpoint whose KV is all inherited.
    std::vector<int> j = t1; j.resize(640); auto jq = RandomTokens(rng, 50); j.insert(j.end(), jq.begin(), jq.end());
    const int hj = Restore(env, j);
    EXPECT(hj == 512, "junction case restores interval 512, got %d", hj);
    env.Close();
}

static void TestQuotaPinsEviction(const std::string &base) {
    Env env; env.dir = FreshDir(base, "quota"); env.quota = 3ull << 20; env.Open();
    std::mt19937 rng(2);
    std::vector<std::vector<int>> convs;
    for (int i = 0; i < 12; ++i) {
        auto t = RandomTokens(rng, 1100);
        Save(env, t, 0); env.Barrier();
        convs.push_back(t);
        EXPECT(env.store->CapacityStatus()["physical_bytes"].string_value().size() > 0, "capacity");
        const uint64_t used = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
        EXPECT(used <= env.quota, "I2 used %llu > quota", (unsigned long long)used);
    }
    auto p = Profile(env);
    EXPECT(p["deletes"].int_value() > 0, "LRU deletes happened: %s", p.dump().c_str());
    std::fprintf(stderr, "quota: deletes=%d delete_max_ms=%.3f\n", p["deletes"].int_value(), p["delete_max_ms"].number_value());
    EXPECT(env.Listed() == env.live, "index == durable - erased");
    // I5: a pinned (open read session) file survives eviction pressure.
    auto last = convs.back();
    auto reader = env.store->BeginRead();
    const std::string pinnedKey = KeyOf(last, 1024);
    Json pinned = reader.Load(pinnedKey, env.identity);
    for (int i = 0; i < 6; ++i) { Save(env, RandomTokens(rng, 1100), 0); env.Barrier(); }
    std::string got(kKvBytes, '\0');
    reader.ReadBytes(pinned["kv"][0]["chunks"], kKvBytes, [&](size_t off, const char *q, size_t m) { memcpy(&got[off], q, m); });
    EXPECT(got == KvTruth(last, 0), "I5 pinned bytes intact");
    // C5: a save larger than the quota is rejected before any eviction.
    const int deletesBefore = Profile(env)["deletes"].int_value();
    {
        auto writer = env.store->BeginWrite(0);
        const std::string huge(size_t(4) << 20, 'x');
        auto chunks = writer.StoreBytes(huge.size(), [&](size_t off, char *out, size_t m) { memcpy(out, huge.data() + off, m); });
        bool threw = false;
        try { writer.CommitTracked(Sha("huge"), Json(Json::object{{"kind", "prefix"}, {"length", 128}, {"family_identity", env.family},
                                    {"prefix", Sha("hp")}, {"kv", Json::array{}}, {"linear_chunks", chunks}}), true, [](bool) {}); }
        catch (const std::exception &e) { threw = std::string(e.what()) == "v5_save_larger_than_quota"; }
        EXPECT(threw, "C5 oversize rejected");
    }
    env.Barrier();
    EXPECT(Profile(env)["deletes"].int_value() == deletesBefore, "C5 no eviction for oversize save");
    env.Close();
}

static void TestNothingEvictableSkipsAtOnce(const std::string &base) {
    Env env; env.dir = FreshDir(base, "noevict"); env.quota = 300u << 10; env.Open();
    std::mt19937 rng(3);
    auto t = RandomTokens(rng, 600);
    Save(env, t, 0); env.Barrier();
    auto reader = env.store->BeginRead();
    Json held = reader.Load(KeyOf(t, 512), env.identity);   // pins the only file
    const auto start = std::chrono::steady_clock::now();
    Save(env, RandomTokens(rng, 1500), 0); env.Barrier();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    EXPECT(Profile(env)["skipped_quota"].int_value() > 0, "quota skip counted");
    EXPECT(ms < 2000, "skip never waits (%.1f ms)", ms);
    EXPECT(env.Listed().count(KeyOf(t, 512)), "pinned checkpoint kept");
    std::fprintf(stderr, "noevict: skip wall %.1f ms\n", ms);
    env.Close();
}

static void TestFaults(const std::string &base) {
    const char *steps[] = {"data_write", "data_sync", "header_write", "header_sync", "dir_sync",
                           "slot_write", "slot_sync", "slot_header_write", "slot_header_sync"};
    const char *modes[] = {"crash", "torn", "eio"};
    int cases = 0;
    for (const char *step : steps) for (const char *mode : modes) for (int branch = 0; branch < 2; ++branch) {
        Env env; env.dir = FreshDir(base, std::string("fault-") + step + "-" + mode + (branch ? "-b" : "-c")); env.Open();
        std::mt19937 rng(100 + cases);
        auto t1 = RandomTokens(rng, 900);
        Save(env, t1, 0); env.Barrier();
        const std::set<std::string> before = env.Listed();
        std::vector<int> t2;
        if (branch) { t2 = t1; t2.resize(600); auto x = RandomTokens(rng, 500); t2.insert(t2.end(), x.begin(), x.end()); }
        else { t2 = t1; auto x = RandomTokens(rng, 500); t2.insert(t2.end(), x.begin(), x.end()); }
        const int h = Restore(env, t2);
        DiskPrefixV5Store::SetFailpointForTest(step, mode);
        Save(env, t2, h); env.Barrier();
        DiskPrefixV5Store::SetFailpointForTest("", "");
        const std::set<std::string> live = env.live;
        if (std::string(mode) == "eio") {
            // The store keeps running and the next save succeeds.
            auto t3 = t1; auto y = RandomTokens(rng, 300); t3.insert(t3.end(), y.begin(), y.end());
            Save(env, t3, Restore(env, t3)); env.Barrier();
            EXPECT(Restore(env, t3) == int(t3.size()) / kPage * kPage, "%s/%s: save after EIO", step, mode);
        }
        env.Close();
        env.Open();   // start-up recovery (sec. 7)
        const auto listed = env.Listed();
        for (const auto &k : before) EXPECT(listed.count(k), "%s/%s/%d: G1 last good checkpoint lost", step, mode, branch);
        {
            // A header that reached the page cache before the injected failure may be
            // valid after restart: that checkpoint is complete (data synced first) and is
            // verified byte for byte below. Nothing acknowledged durable may be missing.
            std::lock_guard<std::mutex> lock(env.mu);
            for (const auto &k : env.live) EXPECT(listed.count(k), "%s/%s/%d: I3 durable checkpoint missing", step, mode, branch);
        }
        for (const auto &k : listed) {
            for (int len = 128; len <= int(t2.size()); len += 128) if (KeyOf(t2, len) == k) {
                std::vector<int> probe(t2.begin(), t2.begin() + len + 1);
                EXPECT(Restore(env, probe) == len, "%s/%s/%d: extra checkpoint %d not byte-exact", step, mode, branch, len);
            }
        }
        (void)live;
        EXPECT(Restore(env, t1) == 896, "%s/%s/%d: turn1 restorable", step, mode, branch);
        const int h2 = Restore(env, t2);
        EXPECT(h2 >= 0, "%s/%s/%d: no corrupt hit (got %d)", step, mode, branch, h2);
        env.Close();
        ++cases;
    }
    std::fprintf(stderr, "faults: %d cases\n", cases);
}

static void TestTwoWriters(const std::string &base) {
    Env env; env.dir = FreshDir(base, "twowriters"); env.Open();
    std::mt19937 rng(4);
    for (int round = 0; round < 10; ++round) {
        auto t1 = RandomTokens(rng, 800);
        Save(env, t1, 0); env.Barrier();
        std::vector<int> a = t1, b = t1;
        auto x = RandomTokens(rng, 600), y = RandomTokens(rng, 600);
        a.insert(a.end(), x.begin(), x.end()); b.insert(b.end(), y.begin(), y.end());
        const int ha = Restore(env, a), hb = Restore(env, b);
        std::thread ta([&] { Save(env, a, ha); }), tb([&] { Save(env, b, hb); });
        ta.join(); tb.join(); env.Barrier();
        EXPECT(Restore(env, a) == int(a.size()) / kPage * kPage, "round %d writer A", round);
        EXPECT(Restore(env, b) == int(b.size()) / kPage * kPage, "round %d writer B", round);
    }
    env.Close(); env.Open();
    EXPECT(env.Listed() == env.live, "two writers: index after restart");
    std::fprintf(stderr, "two writers: %s\n", Profile(env).dump().c_str());
    env.Close();
}

static void TestLockCorruptTornTail(const std::string &base) {
    Env env; env.dir = FreshDir(base, "lock"); env.Open();
    std::mt19937 rng(5);
    auto t = RandomTokens(rng, 700);
    Save(env, t, 0); env.Barrier();
    // C4: another process cannot open the same root.
    pid_t pid = fork();
    if (pid == 0) {
        int fd = open((env.dir + "/v5/LOCK").c_str(), O_RDWR);
        const bool locked = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0;
        _exit(locked ? 0 : 1);
    }
    int status = 0; waitpid(pid, &status, 0);
    EXPECT(WIFEXITED(status) && WEXITSTATUS(status) == 0, "C4 root flock held");
    env.Close();
    // Torn tail: garbage after the committed length is cut at start-up.
    fs::path file;
    for (auto &e : fs::recursive_directory_iterator(env.dir + "/v5")) if (e.path().extension() == ".v5f") file = e.path();
    { std::ofstream out(file, std::ios::app | std::ios::binary); std::string junk(10000, 'J'); out << junk; }
    env.Open();
    EXPECT(Profile(env)["startup_truncated"].int_value() == 1, "torn tail truncated");
    EXPECT(Restore(env, t) == 640, "data intact after tail cut");
    env.Close();
    // Corruption: flip one payload byte -> checksum miss, classified as stored representation.
    std::string content; { std::ifstream in(file, std::ios::binary); content.assign(std::istreambuf_iterator<char>(in), {}); }
    const std::string needle = KvTruth(t, 0).substr(0, 64);
    const size_t at = content.find(needle);
    EXPECT(at != std::string::npos, "payload located");
    if (at != std::string::npos) {
        std::fstream io(file, std::ios::in | std::ios::out | std::ios::binary);
        io.seekp(std::streamoff(at + 7)); char c = char(content[at + 7] ^ 0x5a); io.write(&c, 1);
    }
    env.Open();
    std::string error;
    const int h = Restore(env, t, true, &error);
    EXPECT(h == -1 && DiskPrefixCache::ClassifyRestoreFailure(error) == DiskPrefixCache::RestoreFailureClass::STORED_REPRESENTATION,
           "corrupt payload -> miss (%d, %s)", h, error.c_str());
    std::string retireError;
    EXPECT(env.store->RetireCorruptCheckpoint(KeyOf(t, 640), env.identity, error, retireError), "retire %s", retireError.c_str());
    env.Close();
}


// two writers on different files cannot both pass the quota.
static void TestQuotaTwoWritersAtomic(const std::string &base) {
    uint64_t one = 0;
    {
        Env cal; cal.dir = FreshDir(base, "cal"); cal.Open();
        std::mt19937 rng(7); Save(cal, RandomTokens(rng, 600), 0); cal.Barrier();
        one = std::stoull(cal.store->CapacityStatus()["physical_bytes"].string_value());
        cal.Close();
    }
    Env env; env.dir = FreshDir(base, "quota2w"); env.quota = 1ull << 40; env.Open();
    std::mt19937 rng(8);
    auto t0 = RandomTokens(rng, 600);
    Save(env, t0, 0); env.Barrier();
    const uint64_t used = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    env.Close();
    env.quota = used + one + one / 2; env.Open();
    auto reader = env.store->BeginRead();
    Json held = reader.Load(KeyOf(t0, 512), env.identity);   // old file pinned: nothing evictable
    auto a = RandomTokens(rng, 600), b = RandomTokens(rng, 600);
    std::thread ta([&] { Save(env, a, 0); }), tb([&] { Save(env, b, 0); });
    ta.join(); tb.join(); env.Barrier();
    const uint64_t after = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    EXPECT(after <= env.quota, "two writers: used %llu > quota %llu", (unsigned long long)after, (unsigned long long)env.quota);
    const auto pq = Profile(env);
    EXPECT(pq["skipped_quota"].int_value() + pq["deletes"].int_value() >= 1, "quota enforced by skip or LRU delete: %s", pq.dump().c_str());
    EXPECT(env.Listed() == env.live, "two writers quota: index == durable - erased");
    env.Close();
}

// a pinned prompt-end slot is never overwritten in place.
static void TestPinnedSlotFallback(const std::string &base) {
    Env env; env.dir = FreshDir(base, "slotpin"); env.Open();
    std::mt19937 rng(9);
    auto t1 = RandomTokens(rng, 700);
    Save(env, t1, 0); env.Barrier();
    auto reader = env.store->BeginRead();
    Json held = reader.Load(KeyOf(t1, 640), env.identity);   // turn1 prompt end lives in slot 0
    auto t2 = t1; auto x = RandomTokens(rng, 300); t2.insert(t2.end(), x.begin(), x.end());
    Save(env, t2, Restore(env, t2)); env.Barrier();          // uses slot 1
    auto t3 = t2; auto y = RandomTokens(rng, 300); t3.insert(t3.end(), y.begin(), y.end());
    Save(env, t3, Restore(env, t3)); env.Barrier();          // would reuse slot 0: must append instead
    EXPECT(Profile(env)["slot_fallbacks"].int_value() >= 1, "pinned slot fallback: %s", Profile(env).dump().c_str());
    std::string got(kStateBytes, '\0');
    reader.ReadBytes(held["linear_chunks"], kStateBytes, [&](size_t off, const char *q, size_t m) { memcpy(&got[off], q, m); });
    EXPECT(got == StateTruth(t1, 640, "linear"), "pinned slot bytes intact");
    EXPECT(Restore(env, t3) == int(t3.size()) / kPage * kPage, "turn3 restorable");
    env.Close(); env.Open();
    EXPECT(Restore(env, t3) == int(t3.size()) / kPage * kPage, "turn3 restorable after restart");
    EXPECT(env.Listed() == env.live, "slot fallback: index after restart");
    env.Close();
}

// a fresh prompt-end save does 4 fdatasync + 1 directory fsync before durable.
static void TestSyncSequence(const std::string &base) {
    Env env; env.dir = FreshDir(base, "syncs"); env.Open();
    std::mt19937 rng(10);
    Save(env, RandomTokens(rng, 200), 0); env.Barrier();   // single checkpoint at 128 = prompt end
    auto p = Profile(env);
    EXPECT(p["syncs"].int_value() == 4 && p["dir_syncs"].int_value() == 1 && p["slot_writes"].int_value() == 1,
           "sync sequence: %s", p.dump().c_str());
    env.Close();
}

// Stop while writes are in flight, and stop while idle.
static void TestDrainAndIdleStop(const std::string &base) {
    {   // never saved / after saves / after an I/O failure: clean, immediate drain
        Env env; env.dir = FreshDir(base, "idle"); env.Open();
        std::string error;
        auto t0 = std::chrono::steady_clock::now();
        EXPECT(env.store->DrainPending(error), "idle drain (never saved): %s", error.c_str());
        std::mt19937 rng(11);
        Save(env, RandomTokens(rng, 700), 0); env.Barrier();
        EXPECT(env.store->DrainPending(error), "idle drain (after saves): %s", error.c_str());
        DiskPrefixV5Store::SetFailpointForTest("data_write", "eio");
        Save(env, RandomTokens(rng, 700), 0);
        env.store->DrainPending(error);   // reports the EIO once
        DiskPrefixV5Store::SetFailpointForTest("", "");
        EXPECT(env.store->DrainPending(error), "idle drain (after failure was reported): %s", error.c_str());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        EXPECT(Profile(env)["abandoned"].int_value() == 0, "idle: abandoned=0");
        std::fprintf(stderr, "idle stop: %.1f ms total\n", ms);
        env.Close();
    }
    {   // saves still queued at a 1 ms drain bound: abandoned, never partial
        setenv("FASTLLM_PREFIX_V5_DRAIN_MS", "1", 1);
        Env env; env.dir = FreshDir(base, "drain"); env.Open();
        setenv("FASTLLM_PREFIX_V5_DRAIN_MS", "20000", 1);
        std::mt19937 rng(12);
        for (int i = 0; i < 6; ++i) Save(env, RandomTokens(rng, 1500), 0);
        std::string error;
        const bool clean = env.store->DrainPending(error);
        const int abandoned = Profile(env)["abandoned"].int_value();
        EXPECT(clean == (abandoned == 0), "drain result matches abandoned=%d (%s)", abandoned, error.c_str());
        std::fprintf(stderr, "drain at 1 ms: clean=%d abandoned=%d error=%s\n", clean ? 1 : 0, abandoned, error.c_str());
        env.Close(); env.Open();
        EXPECT(env.Listed() == env.live, "drain: only durable checkpoints after restart");
        env.Close();
    }
}

// a throwing or cancelling progress observer ends the read
// as a miss without killing the process; pins are released and a retry works.
static void TestReadCancel(const std::string &base) {
    Env env; env.dir = FreshDir(base, "cancel"); env.Open();
    std::mt19937 rng(13);
    auto t = RandomTokens(rng, 1500);
    Save(env, t, 0); env.Barrier();
    for (int mode = 0; mode < 2; ++mode) {
        std::string error;
        try {
            auto reader = env.store->BeginRead();
            reader.Load(KeyOf(t, 1408), env.identity, false, [mode](uint64_t, uint64_t) -> bool {
                if (mode == 0) throw std::runtime_error("observer");
                return false;
            });
        } catch (const std::exception &e) { error = e.what(); }
        EXPECT(error == "ssd_read_cancelled", "mode %d cancel -> %s", mode, error.c_str());
    }
    EXPECT(Restore(env, t) == 1408, "retry after cancel");
    env.Close();
}

// Randomized replay of the CPU model ops (new / continue / edit / restart) with
// invariants I1 (every hit is the truth), I2 (quota), I3 (index == durable - erased
// across restarts) and I4 (unlimited quota: continuation hits the previous prompt end).
static void TestRandomModel(const std::string &base, int seeds) {
    for (int seed = 0; seed < seeds; ++seed) {
        const bool limited = seed % 2 == 1;
        Env env; env.dir = FreshDir(base, "random-" + std::to_string(seed));
        env.quota = limited ? (4ull << 20) : (1ull << 40);
        env.Open();
        std::mt19937 rng(1000 + seed);
        std::vector<std::vector<int>> convs;
        for (int op = 0; op < 40; ++op) {
            const int kind = convs.empty() ? 0 : int(rng() % 10);
            std::vector<int> t; int expect = -1;
            if (kind < 3) t = RandomTokens(rng, 200 + int(rng() % 1200));
            else if (kind < 7) {
                const size_t pick = rng() % convs.size();
                auto &base = convs[pick];
                t = base;
                // Model I4: the latest prompt is a full hit (older prompt-end states may
                // have left the two ping-pong slots by design).
                if (pick + 1 == convs.size()) expect = int(base.size()) / kPage * kPage;
                auto add = RandomTokens(rng, 100 + int(rng() % 500)); t.insert(t.end(), add.begin(), add.end());
            } else if (kind < 9) {
                auto &base = convs[rng() % convs.size()];
                t = base; t.resize(size_t(rng() % base.size()) + 1);
                auto add = RandomTokens(rng, 100 + int(rng() % 500)); t.insert(t.end(), add.begin(), add.end());
            } else { env.Close(); env.Open(); EXPECT(SameIndex(env, "restart"), "seed %d op %d I3 after restart", seed, op); continue; }
            const int h = Restore(env, t);
            EXPECT(h >= 0, "seed %d op %d restore failed", seed, op);
            if (!limited && expect > 0) EXPECT(h >= expect, "seed %d op %d I4 hit %d < %d", seed, op, h, expect);
            Save(env, t, std::max(0, h)); env.Barrier();
            convs.push_back(t);
            const uint64_t used = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
            EXPECT(used <= env.quota, "seed %d I2 used %llu", seed, (unsigned long long)used);
            EXPECT(SameIndex(env, "op"), "seed %d op %d index == durable - erased", seed, op);
        }
        env.Close();
    }
    std::fprintf(stderr, "random model: %d seeds x 40 ops\n", seeds);
}

// ---------------------------------------------------------------- D8 tail truncation (FASTLLM_PREFIX_V5_TRIM=1)
static uint64_t Used(Env &env) { return std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value()); }
static bool AllListedRestorable(Env &env, const std::vector<std::vector<int>> &convs, const char *where) {
    bool ok = true;
    const auto listed = env.Listed();
    for (const auto &t : convs)
        for (int len = 128; len <= int(t.size()); len += 128)
            if (listed.count(KeyOf(t, len))) {
                std::vector<int> probe(t.begin(), t.begin() + len + 1);
                const int got = Restore(env, probe);
                if (got != len) { ok = false; std::fprintf(stderr, "  %s: listed %d restored %d\n", where, len, got); }
            }
    return ok;
}
// A long unused conversation loses only its tail; its head stays restorable byte for byte.
static void TestTrimTailFirst(const std::string &base) {
    std::mt19937 rng(41);
    auto a = RandomTokens(rng, 1100), b = RandomTokens(rng, 1100);
    Env env; env.dir = FreshDir(base, "trim-tail"); env.Open();
    Save(env, a, 0); env.Barrier();
    const uint64_t one = Used(env);
    env.Close();
    env.quota = one * 3 / 2; env.Open();
    Save(env, b, 0); env.Barrier();
    auto p = Profile(env);
    EXPECT(p["trims"].int_value() >= 1 && p["deletes"].int_value() == 0, "D8 tail trimmed, nothing deleted: %s", p.dump().c_str());
    EXPECT(Used(env) <= env.quota, "I2 used %llu > quota %llu", (unsigned long long)Used(env), (unsigned long long)env.quota);
    const auto listed = env.Listed();
    EXPECT(listed.count(KeyOf(a, 256)), "D8 head of the old conversation kept");
    EXPECT(!listed.count(KeyOf(a, 1024)), "D8 tail of the old conversation cut");
    EXPECT(listed.count(KeyOf(b, 1024)), "new save durable");
    EXPECT(SameIndex(env, "trim-tail"), "index == durable - erased");
    EXPECT(AllListedRestorable(env, {a, b}, "trim-tail"), "every listed checkpoint byte-exact after trim");
    std::vector<int> a2 = a; auto x = RandomTokens(rng, 300); a2.insert(a2.end(), x.begin(), x.end());
    EXPECT(Restore(env, a2) >= 256, "old conversation still hits its head");
    env.Close(); env.Open();
    EXPECT(SameIndex(env, "trim-tail restart"), "I3 index after restart");
    EXPECT(AllListedRestorable(env, {a, b}, "trim-tail restart"), "byte-exact after restart");
    EXPECT(Profile(env)["startup_truncated"].int_value() == 0, "trimmed file length == committed on disk");
    env.Close();
}
// C12: a parent is never cut below what a live branch uses; the branch survives restart.
static void TestTrimFloorBranch(const std::string &base) {
    std::mt19937 rng(42);
    auto p = RandomTokens(rng, 1100);
    std::vector<int> c = p; c.resize(600); auto x = RandomTokens(rng, 500); c.insert(c.end(), x.begin(), x.end());
    Env env; env.dir = FreshDir(base, "trim-floor"); env.Open();
    Save(env, p, 0); env.Barrier();
    const int hc = Restore(env, c);
    EXPECT(hc == 512, "branch restores parent interval 512, got %d", hc);
    Save(env, c, hc); env.Barrier();
    EXPECT(Profile(env)["branches"].int_value() == 1, "branch file");
    const uint64_t two = Used(env);
    env.Close();
    env.quota = two * 13 / 10; env.Open();
    std::vector<std::vector<int>> others;
    for (int i = 0; i < 4; ++i) {
        EXPECT(Restore(env, c) >= 1024, "branch hot and whole, round %d", i);   // touches the branch chain
        auto o = RandomTokens(rng, 700); Save(env, o, 0); env.Barrier(); others.push_back(o);
        EXPECT(Used(env) <= env.quota, "I2 round %d", i);
    }
    const auto listed = env.Listed();
    EXPECT(listed.count(KeyOf(p, 512)), "C12 parent kept up to the branch point");
    EXPECT(!listed.count(KeyOf(p, 1024)), "parent tail beyond the branch point cut");
    EXPECT(Profile(env)["trims"].int_value() >= 1, "trims happened");
    EXPECT(Restore(env, c) >= 1024, "branch restorable byte-exact after parent trim");
    EXPECT(SameIndex(env, "trim-floor"), "index == durable - erased");
    env.Close(); env.Open();
    EXPECT(SameIndex(env, "trim-floor restart"), "I3 no orphan after restart");
    EXPECT(Restore(env, c) >= 1024, "branch restorable after restart");
    env.Close();
}
// Crash-safe order: header first, then the cut. A crash or a failed step never serves bad bytes.
static void TestTrimFaults(const std::string &base) {
    const char *cases[][2] = {{"trim_truncate", "crash"}, {"trim_truncate", "eio"}, {"trim_header_write", "eio"},
                              {"trim_header_write", "crash"}, {"trim_header_write", "torn"}, {"trim_header_sync", "eio"},
                              {"trim_header_sync", "crash"}};
    int n = 0;
    for (auto &cs : cases) {
        std::mt19937 rng(43 + n);
        auto a = RandomTokens(rng, 1100), b = RandomTokens(rng, 1100);
        Env env; env.dir = FreshDir(base, std::string("trim-fault-") + cs[0] + "-" + cs[1]); env.Open();
        Save(env, a, 0); env.Barrier();
        const uint64_t one = Used(env);
        const std::set<std::string> before = env.Listed();
        env.Close();
        env.quota = one * 3 / 2; env.Open();
        DiskPrefixV5Store::SetFailpointForTest(cs[0], cs[1]);
        Save(env, b, 0); env.Barrier();
        DiskPrefixV5Store::SetFailpointForTest("", "");
        EXPECT(AllListedRestorable(env, {a, b}, cs[0]), "%s/%s: listed byte-exact in process", cs[0], cs[1]);
        env.Close(); env.Open();
        const auto listed = env.Listed();
        {
            std::lock_guard<std::mutex> lock(env.mu);
            for (const auto &k : env.live) EXPECT(listed.count(k), "%s/%s: durable checkpoint missing after restart", cs[0], cs[1]);
        }
        EXPECT(listed.count(KeyOf(a, 256)), "%s/%s: head of the trimmed file kept", cs[0], cs[1]);
        EXPECT(AllListedRestorable(env, {a, b}, cs[0]), "%s/%s: listed byte-exact after restart", cs[0], cs[1]);
        EXPECT(Used(env) <= env.quota || std::string(cs[1]) != "eio" || std::string(cs[0]) == "trim_truncate",
               "%s/%s: quota", cs[0], cs[1]);
        (void)before;
        env.Close();
        ++n;
    }
    std::fprintf(stderr, "trim faults: %d cases\n", n);
}

// The LRU order survives a restart (seeded from file write times): eviction after a restart
// takes the oldest conversations first, never the newest.
static void TestLruAfterRestart(const std::string &base) {
    std::mt19937 rng(44);
    std::vector<std::vector<int>> convs;
    for (int i = 0; i < 4; ++i) convs.push_back(RandomTokens(rng, 1100));
    Env env; env.dir = FreshDir(base, "lru-restart"); env.Open();
    uint64_t one = 0;
    for (auto &t : convs) {
        Save(env, t, 0); env.Barrier();
        if (!one) one = Used(env);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // distinct write times
    }
    const uint64_t all = Used(env);
    env.Close();
    env.quota = all; env.Open();                      // restart: in-memory touches are gone
    auto extra = RandomTokens(rng, 1100);
    Save(env, extra, 0); env.Barrier();               // needs ~one conversation of space
    const auto listed = env.Listed();
    EXPECT(listed.count(KeyOf(convs[3], 1024)) && listed.count(KeyOf(convs[2], 1024)),
           "newest conversations untouched after restart");
    EXPECT(!listed.count(KeyOf(convs[0], 1024)), "oldest conversation evicted first after restart");
    EXPECT(listed.count(KeyOf(extra, 1024)), "new save durable");
    EXPECT(SameIndex(env, "lru-restart"), "index == durable - erased");
    (void)one;
    env.Close();
}

// A pin protects only the prefix it reads (SGLang lock_ref locks matched nodes): the tail of a pinned
// file beyond that prefix is still cut by D8, and the pinned bytes stay intact.
static void TestTrimPinnedPrefix(const std::string &base) {
    std::mt19937 rng(45);
    auto a = RandomTokens(rng, 1100), b = RandomTokens(rng, 1100);
    Env env; env.dir = FreshDir(base, "trim-pinned"); env.Open();
    Save(env, a, 0); env.Barrier();
    const uint64_t one = Used(env);
    env.Close();
    env.quota = one * 3 / 2; env.Open();
    auto reader = env.store->BeginRead();
    Json pinned = reader.Load(KeyOf(a, 512), env.identity);
    Save(env, b, 0); env.Barrier();
    auto p = Profile(env);
    EXPECT(p["trims"].int_value() >= 1, "pinned file tail trimmed: %s", p.dump().c_str());
    const auto listed = env.Listed();
    EXPECT(listed.count(KeyOf(a, 512)), "pinned checkpoint kept");
    EXPECT(!listed.count(KeyOf(a, 1024)), "tail beyond the pinned prefix cut");
    const auto &kv = pinned["kv"].array_items();
    bool same = int(kv.size()) == 512 / kPage;
    for (size_t j = 0; j < kv.size() && same; ++j) {
        std::string got(kKvBytes, '\0');
        reader.ReadBytes(kv[j]["chunks"], kKvBytes, [&](size_t off, const char *q, size_t m) { memcpy(&got[off], q, m); });
        same = got == KvTruth(a, int(j));
    }
    for (const char *what : {"linear", "speculative"}) {
        std::string got(kStateBytes, '\0');
        reader.ReadBytes(pinned[std::string(what) + "_chunks"], kStateBytes, [&](size_t off, const char *q, size_t m) { memcpy(&got[off], q, m); });
        same = same && got == StateTruth(a, 512, what);
    }
    EXPECT(same, "pinned bytes intact after the tail trim");
    EXPECT(Used(env) <= env.quota, "I2");
    EXPECT(SameIndex(env, "trim-pinned"), "index == durable - erased");
    env.Close();
}


// Regression reproducer A1 (verbatim)
#include <future>
// Replay three page-aligned prompt ends. The first two have an identical
// linear-state blob (a legal digest reuse); their KV and draft state differ.
static void ReproA1SharedSlotDigest(const std::string &base) {
    Env env; env.dir = FreshDir(base, "repro-A1-shared-slot"); env.Open();
    std::mt19937 rng(4101);
    const auto t = RandomTokens(rng, 4 * kPage);
    const std::string common = Bytes("A1-common-linear", kStateBytes);
    Json::array kv;
    for (int turn = 1; turn <= 3; ++turn) {
        const int n = turn * kPage;
        auto writer = env.store->BeginWrite(0);
        const std::string kb = KvTruth(t, turn - 1);
        auto kc = writer.StoreBytes(kb.size(), [&](size_t off, char *p, size_t m) { memcpy(p, kb.data() + off, m); });
        kv.push_back(Json::object{{"bytes", double(kb.size())}, {"segments", 0}, {"chunks", kc}});
        const std::string lin = turn <= 2 ? common : Bytes("A1-new-linear", kStateBytes);
        const std::string spec = StateTruth(t, n, "speculative");
        auto lc = writer.StoreBytes(lin.size(), [&](size_t off, char *p, size_t m) { memcpy(p, lin.data() + off, m); });
        auto sc = writer.StoreBytes(spec.size(), [&](size_t off, char *p, size_t m) { memcpy(p, spec.data() + off, m); });
        Json state = Json::object{{"kind", "prefix"}, {"length", n}, {"family_identity", env.family},
            {"kv_dtype", "fp8_e4m3"}, {"prefix", PrefixDigest(t, n)}, {"layout", Json::object{{"tp", 4}}},
            {"kv", kv}, {"linear_chunks", lc}, {"speculative_chunks", sc}, {"v5_prompt_end", true}};
        std::promise<bool> done; auto result = done.get_future();
        writer.CommitTracked(KeyOf(t, n), state, true, [&](bool ok) { done.set_value(ok); });
        EXPECT(result.get(), "A1 turn %d must be durable", turn);
    }
    env.Barrier();
    const std::string key = KeyOf(t, 2 * kPage);
    EXPECT(env.Listed().count(key), "A1 second/latest-previous prompt end is still indexed");
    bool loaded = false; std::string error;
    try {
        auto reader = env.store->BeginRead();
        const Json state = reader.Load(key, env.identity);
        std::string got(kStateBytes, '\0');
        reader.ReadBytes(state["linear_chunks"], got.size(), [&](size_t off, const char *p, size_t m) { memcpy(&got[off], p, m); });
        loaded = got == common;
    } catch (const std::exception &e) { error = e.what(); }
    std::fprintf(stderr, "A1 OBSERVED indexed=%d byte_exact_load=%d error=%s profile=%s\n",
        int(env.Listed().count(key)), int(loaded), error.c_str(), Profile(env).dump_compact().c_str());
    EXPECT(loaded, "A1 durable slot 1 checkpoint must remain readable after slot 0 reuse; actual=%s", error.c_str());
    env.Close();
}


// Regression reproducer A2 (verbatim)
#include <future>
#include <sys/stat.h>
// Each failed fresh save has synced data but fails before publishing its header.
// Those files still consume blocks and must not disappear from quota accounting.
static void ReproA2FailedHeaderQuota(const std::string &base) {
    Env env; env.dir = FreshDir(base, "repro-A2-failed-header"); env.quota = 64ull << 10; env.Open();
    for (int i = 0; i < 3; ++i) {
        auto writer = env.store->BeginWrite(0);
        const std::string payload = Bytes("A2-payload-" + std::to_string(i), size_t(32) << 10);
        auto chunks = writer.StoreBytes(payload.size(), [&](size_t off, char *p, size_t m) { memcpy(p, payload.data() + off, m); });
        Json state = Json::object{{"kind", "encoder"}, {"length", 0}, {"family_identity", env.family},
            {"kv", Json::array{}}, {"payload_chunks", chunks}};
        std::promise<bool> done; auto result = done.get_future();
        DiskPrefixV5Store::SetFailpointForTest("header_write", "eio");
        writer.CommitTracked(Sha("A2-key-" + std::to_string(i)), state, true, [&](bool ok) { done.set_value(ok); });
        EXPECT(!result.get(), "A2 injected header EIO must fail completion");
        env.Barrier();
    }
    DiskPrefixV5Store::SetFailpointForTest("", "");
    uint64_t actual = 0; size_t files = 0;
    for (const auto &e : fs::recursive_directory_iterator(env.dir + "/v5")) {
        struct stat st{};
        if (e.is_regular_file() && lstat(e.path().c_str(), &st) == 0) actual += uint64_t(st.st_blocks) * 512;
        if (e.path().extension() == ".v5f") ++files;
    }
    const uint64_t reported = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    std::fprintf(stderr, "A2 OBSERVED actual_blocks_bytes=%llu reported_bytes=%llu quota=%llu leftover_files=%zu\n",
        (unsigned long long)actual, (unsigned long long)reported, (unsigned long long)env.quota, files);
    EXPECT(actual <= env.quota, "A2 failed fresh headers must not let physical bytes exceed quota: actual=%llu quota=%llu",
        (unsigned long long)actual, (unsigned long long)env.quota);
    EXPECT(reported == actual, "A2 all failed-save blocks must be counted: reported=%llu actual=%llu",
        (unsigned long long)reported, (unsigned long long)actual);
    env.Close();
}


// Regression reproducer A3 (verbatim)
#include <sys/stat.h>
// Metadata is part of the same cache root and has no exemption from C5.
static void ReproA3MetadataQuota(const std::string &base) {
    Env env; env.dir = FreshDir(base, "repro-A3-metadata-quota"); env.quota = 32ull << 10; env.Open();
    bool accepted = false; std::string error;
    try {
        env.store->PutMetadata("A3-runtime-layout", Json::object{{"padding", std::string(size_t(96) << 10, 'M')}});
        accepted = true;
    } catch (const std::exception &e) { error = e.what(); }
    uint64_t actual = 0;
    for (const auto &e : fs::recursive_directory_iterator(env.dir + "/v5")) {
        struct stat st{};
        if (e.is_regular_file() && lstat(e.path().c_str(), &st) == 0) actual += uint64_t(st.st_blocks) * 512;
    }
    const uint64_t reported = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    std::fprintf(stderr, "A3 OBSERVED accepted=%d actual_blocks_bytes=%llu reported_bytes=%llu quota=%llu error=%s\n",
        int(accepted), (unsigned long long)actual, (unsigned long long)reported, (unsigned long long)env.quota, error.c_str());
    EXPECT(actual <= env.quota, "A3 metadata must obey root quota or reject without writing: actual=%llu quota=%llu",
        (unsigned long long)actual, (unsigned long long)env.quota);
    env.Close();
}


// Regression reproducer A4 (verbatim)
#include <future>
// Preserve the header layout and CRC, but mark it as an unknown future version.
// Startup must count such files without serving OR deleting them (draft3 C5).
static void ReproA4UnknownVersionDeletion(const std::string &base) {
    Env env; env.dir = FreshDir(base, "repro-A4-unknown-version"); env.Open();
    {
        auto writer = env.store->BeginWrite(0);
        const std::string payload = Bytes("A4-future-data", 8192);
        auto chunks = writer.StoreBytes(payload.size(), [&](size_t off, char *p, size_t m) { memcpy(p, payload.data() + off, m); });
        Json state = Json::object{{"kind", "encoder"}, {"length", 0}, {"family_identity", env.family},
            {"kv", Json::array{}}, {"payload_chunks", chunks}};
        std::promise<bool> done; auto result = done.get_future();
        writer.CommitTracked(Sha("A4-key"), state, true, [&](bool ok) { done.set_value(ok); });
        EXPECT(result.get(), "A4 seed file must be durable");
    }
    env.Barrier(); env.Close();
    fs::path file;
    for (const auto &e : fs::recursive_directory_iterator(env.dir + "/v5"))
        if (e.path().extension() == ".v5f") file = e.path();
    EXPECT(!file.empty(), "A4 locate seed file");
    if (file.empty()) return;
    auto crc32c = [](const char *p, size_t n) {
        uint32_t crc = ~uint32_t(0);
        for (size_t i = 0; i < n; ++i) {
            crc ^= static_cast<unsigned char>(p[i]);
            for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78u : 0u);
        }
        return ~crc;
    };
    int changed = 0;
    const int fd = open(file.c_str(), O_RDWR);
    EXPECT(fd >= 0, "A4 open future-version fixture");
    if (fd < 0) return;
    for (int h = 0; h < 2; ++h) {
        std::string block(4096, '\0');
        if (pread(fd, &block[0], block.size(), off_t(h) * 4096) != ssize_t(block.size()) ||
            memcmp(block.data(), "FLV5HDR1", 8) != 0) continue;
        const uint32_t before = crc32c(block.data(), 240);
        uint32_t stored; memcpy(&stored, block.data() + 240, 4);
        EXPECT(before == stored, "A4 original header CRC must be valid");
        const uint32_t futureVersion = 2; memcpy(&block[8], &futureVersion, 4);
        const uint32_t crc = crc32c(block.data(), 240); memcpy(&block[240], &crc, 4);
        EXPECT(pwrite(fd, block.data(), block.size(), off_t(h) * 4096) == ssize_t(block.size()), "A4 write version/CRC fixture");
        ++changed;
    }
    EXPECT(fdatasync(fd) == 0, "A4 sync future-version fixture"); close(fd);
    EXPECT(changed > 0, "A4 at least one future header");
    const uintmax_t beforeBytes = fs::file_size(file);
    env.Open();
    const bool exists = fs::exists(file);
    std::fprintf(stderr, "A4 OBSERVED future_version=2 valid_crc_headers=%d file_exists=%d before_size=%llu profile=%s\n",
        changed, int(exists), (unsigned long long)beforeBytes, Profile(env).dump_compact().c_str());
    EXPECT(env.Listed().empty(), "A4 unknown version must not be served");
    EXPECT(exists, "A4 startup must preserve/count unknown-version file instead of unlinking it");
    env.Close();
}


// Regression reproducer B1 (verbatim)
// H21: a 3072-token interval state remains immutable even when it is a prompt end.
static void ReproB1IntervalPromptEnd(const std::string &base) {
    std::fprintf(stderr, "BEGIN B1 interval-prompt-end\n");
    Env env; env.dir = FreshDir(base, "review-B1"); env.Open();
    std::mt19937 rng(4101);
    const auto all = RandomTokens(rng, 3 * 3072 + 1);
    const std::vector<int> first(all.begin(), all.begin() + 3072 + 1);
    const std::vector<int> second(all.begin(), all.begin() + 2 * 3072 + 1);
    Save(env, first, 0); env.Barrier();
    EXPECT(Restore(env, first) == 3072, "B1 setup: first interval saved");
    Save(env, second, Restore(env, second)); env.Barrier();
    EXPECT(Restore(env, second) == 2 * 3072, "B1 setup: second interval saved");
    Save(env, all, Restore(env, all)); env.Barrier();
    EXPECT(Restore(env, all) == 3 * 3072, "B1 setup: third prompt saved");
    const int got = Restore(env, first);
    std::fprintf(stderr, "B1 RESULT expected_interval=%d actual_hit=%d interval_listed=%d\n",
                 3072, got, int(env.Listed().count(KeyOf(first, 3072))));
    EXPECT(got == 3072, "B1 H21: interval 3072 lost when the oldest prompt-end slot is reused; actual=%d", got);
    env.Close(); env.Open();
    const int restart = Restore(env, first);
    EXPECT(restart == 3072, "B1 H21: interval remains lost after restart; actual=%d", restart);
    env.Close();
    std::fprintf(stderr, "END B1\n");
}


// Regression reproducer B2 (verbatim)
// A fresh-file header sync error leaves allocated bytes outside the in-memory quota ledger.
static void ReproB2FailedHeaderQuota(const std::string &base) {
    std::fprintf(stderr, "BEGIN B2 failed-header-quota\n");
    Env env; env.dir = FreshDir(base, "review-B2"); env.quota = 200ull * 1024; env.Open();
    auto put = [&](const std::string &seed) {
        std::promise<bool> done; auto ready = done.get_future();
        {
            auto writer = env.store->BeginWrite(0);
            const std::string bytes = Bytes(seed, 128u * 1024);
            auto refs = writer.StoreBytes(bytes.size(), [&](size_t off, char *out, size_t n) {
                memcpy(out, bytes.data() + off, n);
            });
            writer.CommitTracked(Sha(seed), Json::object{{"kind", "encoder"}, {"length", 0},
                {"family_identity", env.family}, {"kv", Json::array{}}, {"encoder_chunks", refs}},
                true, [&](bool ok) { done.set_value(ok); });
        }
        std::string error; const bool drained = env.store->DrainPending(error);
        EXPECT(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "B2 completion missing");
        const bool durable = ready.get();
        std::fprintf(stderr, "B2 SAVE seed=%s durable=%d drain=%d error=%s\n",
                     seed.c_str(), int(durable), int(drained), error.c_str());
        return durable;
    };
    DiskPrefixV5Store::SetFailpointForTest("header_sync", "eio");
    EXPECT(!put("B2-failed"), "B2 setup: first save must fail");
    DiskPrefixV5Store::SetFailpointForTest("", "");
    EXPECT(put("B2-good"), "B2 setup: second save must become durable");
    uint64_t actual = 0; int files = 0;
    for (const auto &entry : fs::recursive_directory_iterator(env.dir + "/v5")) {
        if (!entry.is_regular_file()) continue;
        struct stat st{}; EXPECT(::stat(entry.path().c_str(), &st) == 0, "B2 stat");
        actual += uint64_t(st.st_blocks) * 512u;
        if (entry.path().extension() == ".v5f") ++files;
    }
    const uint64_t reported = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    std::fprintf(stderr, "B2 RESULT actual=%llu reported=%llu quota=%llu files=%d\n",
                 (unsigned long long)actual, (unsigned long long)reported, (unsigned long long)env.quota, files);
    EXPECT(actual <= env.quota, "B2 C5/G4: failed fresh-file bytes allow actual quota overrun: %llu > %llu",
           (unsigned long long)actual, (unsigned long long)env.quota);
    EXPECT(actual == reported, "B2 every allocated file byte must remain counted: actual=%llu reported=%llu",
           (unsigned long long)actual, (unsigned long long)reported);
    env.Close();
    std::fprintf(stderr, "END B2\n");
}


// Regression reproducer B3 (verbatim)
// Unknown future versions must be counted but never served or deleted (draft3 sec. 7/C5).
static void ReproB3UnknownVersion(const std::string &base) {
    std::fprintf(stderr, "BEGIN B3 unknown-version\n");
    Env env; env.dir = FreshDir(base, "review-B3"); env.Open();
    std::mt19937 rng(4103); const auto t = RandomTokens(rng, 200);
    Save(env, t, 0); env.Barrier(); env.Close();
    fs::path file;
    for (const auto &entry : fs::recursive_directory_iterator(env.dir + "/v5"))
        if (entry.path().extension() == ".v5f") file = entry.path();
    EXPECT(!file.empty(), "B3 setup: durable conversation file");
    if (file.empty()) return;
    auto crc32c = [](const std::string &bytes, size_t n) {
        uint32_t crc = ~uint32_t(0);
        for (size_t i = 0; i < n; ++i) {
            crc ^= static_cast<unsigned char>(bytes[i]);
            for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78u : 0);
        }
        return ~crc;
    };
    {
        std::fstream io(file, std::ios::in | std::ios::out | std::ios::binary);
        for (int copy = 0; copy < 2; ++copy) {
            std::string header(4096, '\0');
            io.seekg(copy * 4096); io.read(&header[0], header.size());
            EXPECT(io.good(), "B3 setup: read header");
            const uint32_t future = 2; memcpy(&header[8], &future, sizeof(future));
            const uint32_t crc = crc32c(header, 240); memcpy(&header[240], &crc, sizeof(crc));
            io.seekp(copy * 4096); io.write(header.data(), header.size());
        }
        io.flush(); EXPECT(io.good(), "B3 setup: write checksummed future headers");
    }
    struct stat st{}; EXPECT(::stat(file.c_str(), &st) == 0, "B3 setup stat");
    const uint64_t fileBytes = uint64_t(st.st_blocks) * 512u;
    env.Open();
    const bool exists = fs::exists(file);
    const uint64_t counted = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    std::fprintf(stderr, "B3 RESULT exists=%d unknown_file_bytes=%llu counted=%llu listed=%zu\n",
                 int(exists), (unsigned long long)fileBytes, (unsigned long long)counted, env.Listed().size());
    EXPECT(exists, "B3 C5: startup must preserve an unknown-version file, not unlink it");
    EXPECT(counted >= fileBytes, "B3 C5: unknown version must remain counted: counted=%llu file=%llu",
           (unsigned long long)counted, (unsigned long long)fileBytes);
    EXPECT(env.Listed().empty(), "B3 unknown version must never be served");
    env.Close();
    std::fprintf(stderr, "END B3\n");
}


// Regression reproducer B4 (verbatim)
// Valid metadata writes share the root quota; they cannot bypass admission.
static void ReproB4MetadataQuota(const std::string &base) {
    std::fprintf(stderr, "BEGIN B4 metadata-quota\n");
    Env env; env.dir = FreshDir(base, "review-B4"); env.quota = 64ull * 1024; env.Open();
    bool accepted = false; std::string error;
    try {
        env.store->PutMetadata("B4-large", Json::object{{"payload", std::string(96u * 1024, 'm')}});
        accepted = true;
    } catch (const std::exception &e) { error = e.what(); }
    const uint64_t used = std::stoull(env.store->CapacityStatus()["physical_bytes"].string_value());
    std::fprintf(stderr, "B4 RESULT accepted=%d used=%llu quota=%llu error=%s\n", int(accepted),
                 (unsigned long long)used, (unsigned long long)env.quota, error.c_str());
    EXPECT(used <= env.quota, "B4 C5: PutMetadata bypassed root quota: used=%llu quota=%llu accepted=%d",
           (unsigned long long)used, (unsigned long long)env.quota, int(accepted));
    env.Close();
    std::fprintf(stderr, "END B4\n");
}


// Regression reproducer B5 (verbatim)
// Identical state chunks in both prompt-end slots must not lose the surviving slot's lookup.
static void ReproB5SharedSlotDigest(const std::string &base) {
    std::fprintf(stderr, "BEGIN B5 shared-slot-digest\n");
    Env env; env.dir = FreshDir(base, "review-B5"); env.Open();
    std::mt19937 rng(4105); const auto t = RandomTokens(rng, 3 * kPage + 1);
    const std::string shared = Bytes("B5-shared-linear", kStateBytes);
    Json::array kv;
    auto savePoint = [&](int page, const std::string &linear) {
        const int length = page * kPage;
        std::promise<bool> done; auto ready = done.get_future();
        {
            auto writer = env.store->BeginWrite(0);
            const std::string bytes = KvTruth(t, page - 1);
            auto chunks = writer.StoreBytes(bytes.size(), [&](size_t off, char *out, size_t n) {
                memcpy(out, bytes.data() + off, n);
            });
            kv.push_back(Json::object{{"bytes", double(bytes.size())}, {"segments", 0}, {"chunks", chunks}});
            auto lc = writer.StoreBytes(linear.size(), [&](size_t off, char *out, size_t n) {
                memcpy(out, linear.data() + off, n);
            });
            const std::string speculative = StateTruth(t, length, "speculative");
            auto sc = writer.StoreBytes(speculative.size(), [&](size_t off, char *out, size_t n) {
                memcpy(out, speculative.data() + off, n);
            });
            writer.CommitTracked(KeyOf(t, length), Json::object{{"kind", "prefix"}, {"length", length},
                {"family_identity", env.family}, {"prefix", PrefixDigest(t, length)}, {"kv", kv},
                {"linear_chunks", lc}, {"speculative_chunks", sc}, {"v5_prompt_end", true}},
                true, [&](bool ok) { done.set_value(ok); });
        }
        std::string error; env.store->DrainPending(error);
        EXPECT(ready.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "B5 completion missing");
        EXPECT(ready.get(), "B5 setup: point %d must be durable", length);
    };
    auto readSecond = [&]() {
        auto reader = env.store->BeginRead();
        const Json state = reader.Load(KeyOf(t, 2 * kPage), env.identity);
        std::string got(shared.size(), '\0');
        reader.ReadBytes(state["linear_chunks"], got.size(), [&](size_t off, const char *p, size_t n) {
            memcpy(&got[off], p, n);
        });
        return got == shared;
    };
    savePoint(1, shared); savePoint(2, shared);
    EXPECT(readSecond(), "B5 setup: shared bytes restore before reuse");
    savePoint(3, Bytes("B5-different-linear", kStateBytes));
    bool exact = false; std::string error;
    try { exact = readSecond(); } catch (const std::exception &e) { error = e.what(); }
    std::fprintf(stderr, "B5 RESULT surviving_checkpoint_listed=%d restored_exact=%d error=%s\n",
                 int(env.Listed().count(KeyOf(t, 2 * kPage))), int(exact), error.c_str());
    EXPECT(exact, "B5 surviving newer slot lost shared digest on older-slot reuse: %s", error.c_str());
    env.Close(); env.Open();
    EXPECT(readSecond(), "B5 control: surviving on-disk bytes are intact after rebuilding the map at restart");
    env.Close();
    std::fprintf(stderr, "END B5\n");
}


// Regression reproducer A1 (verbatim)
static bool ReproA1ReplayOnce(const std::string &dir, uint64_t *badUsed, int *badOp) {
    Env env; env.dir = FreshDir(dir, "a1"); env.quota = 4ull << 20; env.Open();
    std::mt19937 rng(1001);
    std::vector<std::vector<int>> convs;
    bool ok = true;
    for (int op = 0; op < 40 && ok; ++op) {
        const int kind = convs.empty() ? 0 : int(rng() % 10);
        std::vector<int> t;
        if (kind < 3) t = RandomTokens(rng, 200 + int(rng() % 1200));
        else if (kind < 7) {
            const size_t pick = rng() % convs.size();
            t = convs[pick];
            auto add = RandomTokens(rng, 100 + int(rng() % 500)); t.insert(t.end(), add.begin(), add.end());
        } else if (kind < 9) {
            auto &b = convs[rng() % convs.size()];
            t = b; t.resize(size_t(rng() % b.size()) + 1);
            auto add = RandomTokens(rng, 100 + int(rng() % 500)); t.insert(t.end(), add.begin(), add.end());
        } else { env.Close(); env.Open(); continue; }
        const int h = Restore(env, t);
        Save(env, t, std::max(0, h)); env.Barrier();
        convs.push_back(t);
        const uint64_t used = Used(env);
        if (used > env.quota) { ok = false; *badUsed = used; *badOp = op; }
    }
    env.Close();
    return ok;
}
static void ReproA1TrimQuota(const std::string &base) {
    const uint64_t quota = 4ull << 20;
    bool ok = true; uint64_t badUsed = 0; int badOp = -1, attempt = 0;
    for (; attempt < 40 && ok; ++attempt)
        ok = ReproA1ReplayOnce(base + "/a1-attempt-" + std::to_string(attempt), &badUsed, &badOp);
    std::fprintf(stderr, "A1 RESULT trim=%s attempts=%d used=%llu quota=%llu over=%llu op=%d",
                 getenv("FASTLLM_PREFIX_V5_TRIM") ? getenv("FASTLLM_PREFIX_V5_TRIM") : "unset",
                 attempt, (unsigned long long)badUsed, (unsigned long long)quota,
                 (unsigned long long)(badUsed > quota ? badUsed - quota : 0), badOp);
    EXPECT(ok, "A1: trim-enabled save left used=%llu > quota=%llu at op %d (attempt %d of 40)",
           (unsigned long long)badUsed, (unsigned long long)quota, badOp, attempt);
}

static void TestDefaultsAndOverrides(const std::string &base) {
    // Unset env -> documented defaults; explicit env -> override (the launcher maps deploy flags to these).
    for (const char *k : {"FASTLLM_PREFIX_V5_TRIM", "FASTLLM_PREFIX_V5_READ_THREADS", "FASTLLM_PREFIX_V5_DRAIN_MS",
                          "FASTLLM_PREFIX_V5_INTERVAL_TOKENS"}) unsetenv(k);
    std::string error;
    {
        auto store = DiskPrefixV5Store::OpenShared(base + "/defaults-a", 256ull << 30, error);
        EXPECT(store != nullptr, "defaults open: %s", error.c_str());
        if (store) {
            const std::string got = store->Settings();
            const std::string want = "quota_bytes=274877906944 trim=on read_threads=8 drain_ms=30000 interval_tokens=3072";
            EXPECT(got == want, "defaults: got '%s' want '%s'", got.c_str(), want.c_str());
        }
    }
    setenv("FASTLLM_PREFIX_V5_TRIM", "0", 1); setenv("FASTLLM_PREFIX_V5_READ_THREADS", "4", 1);
    setenv("FASTLLM_PREFIX_V5_DRAIN_MS", "1234", 1); setenv("FASTLLM_PREFIX_V5_INTERVAL_TOKENS", "2048", 1);
    {
        auto store = DiskPrefixV5Store::OpenShared(base + "/defaults-b", 16ull << 30, error);
        EXPECT(store != nullptr, "override open: %s", error.c_str());
        if (store) {
            const std::string got = store->Settings();
            const std::string want = "quota_bytes=17179869184 trim=off read_threads=4 drain_ms=1234 interval_tokens=2048";
            EXPECT(got == want, "overrides: got '%s' want '%s'", got.c_str(), want.c_str());
        }
    }
}

int main(int argc, char **argv) {
    setenv("FASTLLM_PREFIX_V5", "1", 1);
    setenv("FASTLLM_PREFIX_V5_DRAIN_MS", "20000", 1);
    // Trim is on by default now; the default-mode tests cover the whole-file eviction path, so they
    // pin trim off explicitly. "trim" mode turns it on; "defaults" checks the unset defaults.
    if (!(argc > 3 && std::string(argv[3]) == "defaults")) setenv("FASTLLM_PREFIX_V5_TRIM", "0", 1);
    const std::string base = argc > 1 ? argv[1] : (fs::temp_directory_path() / "fastllm-v5-test").string();
    fs::create_directories(base);
    const int seeds = argc > 2 ? std::atoi(argv[2]) : 20;
    if (argc > 3 && std::string(argv[3]) == "review") {
    ReproA1SharedSlotDigest(base);
    ReproA2FailedHeaderQuota(base);
    ReproA3MetadataQuota(base);
    ReproA4UnknownVersionDeletion(base);
    ReproB1IntervalPromptEnd(base);
    ReproB2FailedHeaderQuota(base);
    ReproB3UnknownVersion(base);
    ReproB4MetadataQuota(base);
    ReproB5SharedSlotDigest(base);
        std::printf(g_failures ? "FAIL review failures=%d\n" : "PASS review\n", g_failures); return g_failures ? 1 : 0; }
    if (argc > 3 && std::string(argv[3]) == "defaults") { TestDefaultsAndOverrides(base);
        std::printf(g_failures ? "FAIL defaults failures=%d\n" : "PASS defaults\n", g_failures); return g_failures ? 1 : 0; }
    if (argc > 3 && std::string(argv[3]) == "random") { TestRandomModel(base, seeds); std::printf(g_failures ? "FAIL\n" : "PASS\n"); return g_failures ? 1 : 0; }
    if (argc > 3 && std::string(argv[3]) == "trim") {
        setenv("FASTLLM_PREFIX_V5_TRIM", "1", 1);
        TestTrimTailFirst(base);
        TestTrimFloorBranch(base);
        TestTrimFaults(base);
        TestLruAfterRestart(base);
        TestTrimPinnedPrefix(base);
        ReproA1TrimQuota(base);
        TestRandomModel(base, seeds);   // property test (I1-I5) with tail truncation on
        std::printf(g_failures ? "FAIL trim failures=%d\n" : "PASS trim\n", g_failures); return g_failures ? 1 : 0; }
    TestMultiTurnSlotsBranch(base);
    TestQuotaPinsEviction(base);
    TestNothingEvictableSkipsAtOnce(base);
    TestFaults(base);
    TestTwoWriters(base);
    TestLockCorruptTornTail(base);
    TestQuotaTwoWritersAtomic(base);
    TestPinnedSlotFallback(base);
    TestSyncSequence(base);
    TestDrainAndIdleStop(base);
    TestReadCancel(base);
    TestRandomModel(base, seeds);
    ReproA1SharedSlotDigest(base);
    ReproA2FailedHeaderQuota(base);
    ReproA3MetadataQuota(base);
    ReproA4UnknownVersionDeletion(base);
    ReproB1IntervalPromptEnd(base);
    ReproB2FailedHeaderQuota(base);
    ReproB3UnknownVersion(base);
    ReproB4MetadataQuota(base);
    ReproB5SharedSlotDigest(base);
    if (g_failures) { std::printf("FAIL disk_prefix_store_v5 failures=%d\n", g_failures); return 1; }
    std::printf("PASS disk_prefix_store_v5\n");
    return 0;
}
