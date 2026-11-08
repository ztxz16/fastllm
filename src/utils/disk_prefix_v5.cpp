// SSD prefix cache V5. Design: logs/a6-v5-20261107/DESIGN-V5-draft3.md.
// Section numbers in comments refer to that draft.
#include "utils/disk_prefix_v5.h"

#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <nmmintrin.h>
#endif

namespace fastllm {
namespace fs = std::filesystem;
using Json = json11::Json;

bool DiskPrefixV5Enabled() {
    static const bool enabled = [] {
        // Default ON (user 2026-11-07: V5 replaces V4). FASTLLM_PREFIX_V5=0 = V4 rollback.
        const char *value = std::getenv("FASTLLM_PREFIX_V5");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

namespace {
constexpr uint64_t kBlock = 4096;
constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kSlotBase = 2 * kHeaderBytes;
constexpr char kHeaderMagic[8] = {'F','L','V','5','H','D','R','1'};
constexpr char kChunkMagic[8] = {'F','L','V','5','C','H','K','1'};
constexpr uint32_t kFormatVersion = 1;
constexpr int kDefaultIntervalTokens = 3072;   // draft3 D10; FASTLLM_PREFIX_V5_INTERVAL_TOKENS overrides
constexpr size_t kChunkFixed = 64;
constexpr size_t kEntryBytes = 56;
const char *kFormatText = "fastllm-prefix-v5 1\n";

uint64_t NowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
uint64_t WallNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
uint64_t RoundUp(uint64_t value) { return (value + kBlock - 1) / kBlock * kBlock; }
double Ms(uint64_t ns) { return double(ns) / 1e6; }

// ---- crc32c (Castagnoli). Hardware path when SSE4.2 exists (draft3 G5). ----
uint32_t g_crcTable[256];
bool InitCrcTable() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        g_crcTable[i] = c;
    }
    return true;
}
const bool g_crcTableReady = InitCrcTable();
uint32_t Crc32cSoft(uint32_t crc, const unsigned char *p, size_t n) {
    crc = ~crc;
    while (n--) crc = g_crcTable[(crc ^ *p++) & 0xff] ^ (crc >> 8);
    return ~crc;
}
#if defined(__x86_64__)
__attribute__((target("sse4.2")))
uint32_t Crc32cHw(uint32_t crc, const unsigned char *p, size_t n) {
    uint64_t c = ~uint64_t(crc) & 0xffffffffu;
    while (n >= 8) { uint64_t v; memcpy(&v, p, 8); c = _mm_crc32_u64(c, v); p += 8; n -= 8; }
    uint32_t c32 = uint32_t(c);
    while (n--) c32 = _mm_crc32_u8(c32, *p++);
    return ~c32;
}
#endif
uint32_t Crc32c(const void *data, size_t n, uint32_t crc = 0) {
    (void)g_crcTableReady;
#if defined(__x86_64__)
    static const bool hw = __builtin_cpu_supports("sse4.2");
    if (hw) return Crc32cHw(crc, static_cast<const unsigned char *>(data), n);
#endif
    return Crc32cSoft(crc, static_cast<const unsigned char *>(data), n);
}

std::string Sha256Hex(const void *data, size_t size) {
    unsigned char out[32]; unsigned int length = 0;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, size) != 1 || EVP_DigestFinal_ex(ctx, out, &length) != 1) {
        if (ctx) EVP_MD_CTX_free(ctx);
        throw std::runtime_error("v5_sha256");
    }
    EVP_MD_CTX_free(ctx);
    static const char *hex = "0123456789abcdef";
    std::string text(64, '0');
    for (int i = 0; i < 32; ++i) { text[2*i] = hex[out[i] >> 4]; text[2*i+1] = hex[out[i] & 15]; }
    return text;
}
std::string Sha256Hex(const std::string &value) { return Sha256Hex(value.data(), value.size()); }
bool IsDigest(const std::string &value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
bool HexToRaw(const std::string &hex, unsigned char *out) {
    if (!IsDigest(hex)) return false;
    auto nib = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    for (int i = 0; i < 32; ++i) out[i] = (unsigned char)((nib(hex[2*i]) << 4) | nib(hex[2*i+1]));
    return true;
}
std::string RawToHex(const unsigned char *raw) {
    static const char *hex = "0123456789abcdef";
    std::string text(64, '0');
    for (int i = 0; i < 32; ++i) { text[2*i] = hex[raw[i] >> 4]; text[2*i+1] = hex[raw[i] & 15]; }
    return text;
}

void Put32(std::string &b, size_t off, uint32_t v) { memcpy(&b[off], &v, 4); }
void Put64(std::string &b, size_t off, uint64_t v) { memcpy(&b[off], &v, 8); }
uint32_t Get32(const std::string &b, size_t off) { uint32_t v; memcpy(&v, &b[off], 4); return v; }
uint64_t Get64(const std::string &b, size_t off) { uint64_t v; memcpy(&v, &b[off], 8); return v; }

// A ref is any object {"sha256": <digest>, "bytes": <number>} (same rule as V4 CollectRefs).
void CollectRefs(const Json &value, std::vector<std::pair<std::string, uint64_t>> &out) {
    if (value.is_array()) { for (const auto &item : value.array_items()) CollectRefs(item, out); return; }
    if (!value.is_object()) return;
    const auto &items = value.object_items();
    auto hash = items.find("sha256"), bytes = items.find("bytes");
    if (hash != items.end() && bytes != items.end() && hash->second.is_string() &&
        bytes->second.is_number() && IsDigest(hash->second.string_value())) {
        out.push_back({hash->second.string_value(), uint64_t(bytes->second.number_value())});
        return;
    }
    for (const auto &item : items) CollectRefs(item.second, out);
}

// ---- fault injection (tests only; inert unless set) ----
std::mutex g_failMutex;
std::string g_failStep, g_failMode;
// Returns "" (no fault), "crash", "torn" or "eio" for this step; one-shot.
std::string TakeFault(const char *step) {
    std::lock_guard<std::mutex> lock(g_failMutex);
    if (g_failStep.empty() || g_failStep != step) return "";
    std::string mode = g_failMode; g_failStep.clear(); g_failMode.clear();
    return mode;
}

struct Fd {
    int fd = -1;
    Fd() = default;
    explicit Fd(int value) : fd(value) {}
    Fd(Fd &&other) noexcept : fd(other.fd) { other.fd = -1; }
    Fd &operator=(Fd &&other) noexcept { if (fd >= 0) close(fd); fd = other.fd; other.fd = -1; return *this; }
    ~Fd() { if (fd >= 0) close(fd); }
};

bool PreadAll(int fd, void *out, size_t n, uint64_t offset) {
    char *p = static_cast<char *>(out);
    while (n > 0) {
        const ssize_t r = pread(fd, p, n, off_t(offset));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        p += r; n -= size_t(r); offset += uint64_t(r);
    }
    return true;
}
bool SyncDir(const fs::path &dir) {
    Fd fd(open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    return fd.fd >= 0 && fsync(fd.fd) == 0;
}
uint64_t PhysicalOf(const fs::path &path) {
    struct stat st{};
    if (lstat(path.c_str(), &st) != 0) return 0;
    return uint64_t(st.st_blocks) * 512u;
}
uint64_t PhysicalOfFd(int fd) {
    struct stat st{};
    if (fstat(fd, &st) != 0) return 0;
    return uint64_t(st.st_blocks) * 512u;
}

struct BlobLoc { uint64_t offset = 0, bytes = 0; uint32_t crc = 0; int slot = -1; };
struct Pending { std::string digest; const std::string *payload = nullptr; };

// One chunk = [header region: fixed 64 B + entries + kv JSON + checkpoint JSON,
// padded to 4 KiB][4 KiB-aligned payloads] (sec. 3, G6).
struct ChunkImage {
    std::string header;
    std::vector<Pending> blobs;
    std::vector<uint64_t> rel;
    uint64_t total = 0;
};
ChunkImage BuildChunk(uint64_t incarnation, uint64_t sequence, const std::vector<Pending> &blobs,
                      const std::string &kvJson, const std::string &ckptJson) {
    ChunkImage image;
    image.blobs = blobs;
    const size_t used = kChunkFixed + blobs.size() * kEntryBytes + kvJson.size() + ckptJson.size();
    const uint64_t headerBytes = RoundUp(used);
    image.header.assign(size_t(headerBytes), '\0');
    uint64_t at = headerBytes;
    for (const auto &blob : blobs) { image.rel.push_back(at); at += RoundUp(blob.payload->size()); }
    image.total = at;
    std::string &h = image.header;
    memcpy(&h[0], kChunkMagic, 8);
    Put32(h, 8, kFormatVersion); Put32(h, 12, uint32_t(blobs.size()));
    Put64(h, 16, image.total); Put64(h, 24, headerBytes); Put64(h, 32, sequence); Put64(h, 40, incarnation);
    Put32(h, 48, uint32_t(kvJson.size())); Put32(h, 52, uint32_t(ckptJson.size())); Put32(h, 56, uint32_t(used));
    size_t off = kChunkFixed;
    for (size_t i = 0; i < blobs.size(); ++i) {
        unsigned char raw[32];
        HexToRaw(blobs[i].digest, raw);
        memcpy(&h[off], raw, 32);
        Put64(h, off + 32, image.rel[i]);
        Put64(h, off + 40, blobs[i].payload->size());
        Put32(h, off + 48, Crc32c(blobs[i].payload->data(), blobs[i].payload->size()));
        off += kEntryBytes;
    }
    memcpy(&h[off], kvJson.data(), kvJson.size()); off += kvJson.size();
    memcpy(&h[off], ckptJson.data(), ckptJson.size());
    Put32(h, 60, Crc32c(h.data(), used));   // crc field is still zero here
    return image;
}
struct ParsedChunk {
    uint64_t total = 0, headerBytes = 0, sequence = 0, incarnation = 0;
    struct Entry { std::string digest; uint64_t rel = 0, bytes = 0; uint32_t crc = 0; };
    std::vector<Entry> entries;
    std::string kvJson, ckptJson;
};
bool ReadChunk(int fd, uint64_t offset, uint64_t limit, ParsedChunk &out, std::string &error) {
    std::string h(kBlock, '\0');
    if (offset + kBlock > limit || !PreadAll(fd, &h[0], kBlock, offset)) { error = "cache_short_read"; return false; }
    if (memcmp(h.data(), kChunkMagic, 8) != 0 || Get32(h, 8) != kFormatVersion) { error = "cache_record_header"; return false; }
    out.total = Get64(h, 16); out.headerBytes = Get64(h, 24); out.sequence = Get64(h, 32); out.incarnation = Get64(h, 40);
    const uint32_t blobs = Get32(h, 12), kvBytes = Get32(h, 48), ckptBytes = Get32(h, 52), used = Get32(h, 56);
    if (out.headerBytes < kBlock || out.headerBytes % kBlock || out.total < out.headerBytes || out.total % kBlock ||
        offset + out.total > limit || used > out.headerBytes ||
        uint64_t(kChunkFixed) + uint64_t(blobs) * kEntryBytes + kvBytes + ckptBytes != used) {
        error = "cache_record_bounds"; return false;
    }
    if (out.headerBytes > kBlock) {
        h.resize(size_t(out.headerBytes));
        if (!PreadAll(fd, &h[kBlock], size_t(out.headerBytes - kBlock), offset + kBlock)) { error = "cache_short_read"; return false; }
    }
    const uint32_t stored = Get32(h, 60);
    Put32(h, 60, 0);
    if (Crc32c(h.data(), used) != stored) { error = "cache_record_header_crc"; return false; }
    size_t off = kChunkFixed;
    out.entries.resize(blobs);
    for (uint32_t i = 0; i < blobs; ++i) {
        auto &e = out.entries[i];
        e.digest = RawToHex(reinterpret_cast<const unsigned char *>(&h[off]));
        e.rel = Get64(h, off + 32); e.bytes = Get64(h, off + 40); e.crc = Get32(h, off + 48);
        if (e.rel < out.headerBytes || e.rel % kBlock || e.rel + e.bytes > out.total) { error = "cache_object_bounds"; return false; }
        off += kEntryBytes;
    }
    out.kvJson.assign(h, off, kvBytes); off += kvBytes;
    out.ckptJson.assign(h, off, ckptBytes);
    return true;
}

struct Header {
    uint64_t generation = 0, incarnation = 0, parentInc = 0, parentKv = 0, dataStart = 0,
             committed = 0, slotCap = 0, createdNs = 0;
    std::string identity, parentKvDigest;
    uint64_t slotGen[2] = {0, 0}, slotBytes[2] = {0, 0};
};
std::string EncodeHeader(const Header &value) {
    std::string h(kHeaderBytes, '\0');
    memcpy(&h[0], kHeaderMagic, 8);
    Put32(h, 8, kFormatVersion);
    Put64(h, 16, value.generation); Put64(h, 24, value.incarnation);
    memcpy(&h[32], value.identity.data(), std::min<size_t>(64, value.identity.size()));
    Put64(h, 96, value.parentInc); Put64(h, 104, value.parentKv);
    memcpy(&h[112], value.parentKvDigest.data(), std::min<size_t>(64, value.parentKvDigest.size()));
    Put64(h, 176, value.dataStart); Put64(h, 184, value.committed); Put64(h, 192, value.slotCap);
    Put64(h, 200, value.slotGen[0]); Put64(h, 208, value.slotBytes[0]);
    Put64(h, 216, value.slotGen[1]); Put64(h, 224, value.slotBytes[1]);
    Put64(h, 232, value.createdNs);
    Put32(h, 240, Crc32c(h.data(), 240));
    return h;
}
bool DecodeHeader(const std::string &h, Header &out) {
    if (h.size() < 244 || memcmp(h.data(), kHeaderMagic, 8) != 0 || Get32(h, 8) != kFormatVersion) return false;
    if (Crc32c(h.data(), 240) != Get32(h, 240)) return false;
    out.generation = Get64(h, 16); out.incarnation = Get64(h, 24);
    out.identity.assign(h, 32, 64);
    out.parentInc = Get64(h, 96); out.parentKv = Get64(h, 104);
    out.parentKvDigest.assign(h, 112, 64);
    if (out.parentKvDigest[0] == '\0') out.parentKvDigest.clear();
    out.dataStart = Get64(h, 176); out.committed = Get64(h, 184); out.slotCap = Get64(h, 192);
    out.slotGen[0] = Get64(h, 200); out.slotBytes[0] = Get64(h, 208);
    out.slotGen[1] = Get64(h, 216); out.slotBytes[1] = Get64(h, 224);
    out.createdNs = Get64(h, 232);
    return out.generation > 0 && out.dataStart >= kSlotBase && out.dataStart % kBlock == 0 &&
           out.committed >= out.dataStart && out.committed % kBlock == 0 &&
           out.dataStart == kSlotBase + 2 * out.slotCap && IsDigest(out.identity);
}

std::string IncName(uint64_t inc) {
    char text[32]; snprintf(text, sizeof(text), "%016llx.v5f", (unsigned long long)inc);
    return text;
}

struct Registry { std::mutex mutex; std::map<std::string, std::weak_ptr<DiskPrefixV5Store>> stores; };
Registry &GetRegistry() { static Registry registry; return registry; }
} // namespace

// =========================================================================
struct DiskPrefixV5Store::Impl : std::enable_shared_from_this<DiskPrefixV5Store::Impl> {
    struct Ckpt;
    struct File {
        uint64_t inc = 0;
        std::string identity;
        fs::path path;
        Header header;            // last durable header
        int activeHeader = 0;     // index (0/1) of the durable header copy
        File *parent = nullptr;
        std::vector<Json> ownKv;  // own KV bundles, appended in order
        // digest -> every location holding it (data chunks, slot 0, slot 1). A digest may live in
        // both slots; reusing one slot removes only that slot's locations.
        std::unordered_map<std::string, std::vector<BlobLoc>> blobs;
        std::vector<std::string> slotDigests[2];
        std::shared_ptr<Ckpt> slotCkpt[2];
        uint64_t chunkSeq = 0, physical = 0, lastTouch = 0;
        uint64_t ownTouch = 0;    // touched only as the used file, not via a descendant (D8)
        struct timespec mtime{};  // at start-up: seeds the LRU order (newest write = most recent)
        struct ChunkRec { uint64_t offset = 0, total = 0; size_t kvEnd = 0; };
        std::vector<ChunkRec> chunks;   // data chunks in file order; kvEnd = ownKv.size() after it
        std::multiset<uint64_t> pinExtents;   // byte end each active pin needs in this file (D8)
        int pins = 0, slotPins[2] = {0, 0}, children = 0;
        // Set (under mu) when a writer selects a slot for overwrite, cleared when that commit ends. Load refuses
        // to pin a checkpoint in such a slot, so "not pinned" checked at selection stays true until the bytes
        // change (a reader pinned after selection read overwritten bytes). Same rule as SGLang
        // eviction (lock_ref == 0 checked and the node removed under one lock) and vLLM dropping a block's hash
        // when the block is reallocated.
        bool slotWriting[2] = {false, false};
        uint64_t lease = 0;       // writer id, 0 = none
        std::vector<std::shared_ptr<Ckpt>> ckpts;
        uint64_t KvLength() const { return header.parentKv + ownKv.size(); }
    };
    struct Ckpt {
        std::string key, identity, family, kind, prefix, provenance, kvDtype, stateDigest, metadata;
        int length = 0;
        uint64_t createdNs = 0, kvCount = 0;
        Json value;               // stored commit JSON without "kv"
        File *file = nullptr;
        int slot = -1;            // -1 = in a data chunk
        uint64_t chunkOffset = 0; // data checkpoint: offset of its chunk
        bool retired = false;
    };
    struct Job {
        std::shared_ptr<Writer> writer;
        std::string key, identity, family;
        Json value;
        Staged staged;
        uint64_t bytes = 0, submitNs = 0;
        std::function<void(bool)> completion;
    };

    fs::path base, root;
    uint64_t quota = 0;
    Fd lockFd;
    int readThreads = 8;
    uint64_t drainMs = 30000;
    bool trimEnabled = true;       // D8 tail truncation; default on, FASTLLM_PREFIX_V5_TRIM=0 disables
    int intervalTokens = kDefaultIntervalTokens;
    mutable std::mutex mu;
    std::condition_variable cv, idle;
    std::map<uint64_t, std::unique_ptr<File>> files;
    std::map<std::pair<std::string, std::string>, std::shared_ptr<Ckpt>> ckpts; // (identity,key)
    // identity -> page chain hash -> number of indexed checkpoints that contain it
    std::map<std::string, std::unordered_map<std::string, uint32_t>> pageRefs;
    uint64_t usedBytes = 0, otherBytes = 0, touchClock = 0, nextWriter = 1;
    std::set<const File *> busy;   // target + parent chain of the commit in progress (mu)
    std::deque<std::shared_ptr<Job>> queue;
    std::shared_ptr<Job> active;
    uint64_t inflight = 0;
    bool stopping = false, dead = false;
    uint64_t drainFailures = 0;
    std::string firstFailure;
    std::thread thread;
    EraseObserver eraseObserver;
    Predicate reclaimPredicate, corruptPredicate;
    std::mt19937_64 rng{std::random_device{}() ^ WallNs()};
    // profile counters (mu)
    struct Counters {
        uint64_t commits = 0, commitBytes = 0, dedup = 0, branches = 0, newFiles = 0, appends = 0,
                 slotWrites = 0, slotFallbacks = 0, skippedQuota = 0, skippedTooLarge = 0, failures = 0,
                 cancelled = 0, abandoned = 0, deletes = 0, deleteNs = 0, deleteMaxNs = 0, writeNs = 0,
                 syncNs = 0, syncs = 0, dirSyncs = 0, durableNs = 0, durableMaxNs = 0, crcNs = 0,
                 loads = 0, loadBytes = 0, loadNs = 0, readCrcNs = 0, manifests = 0, retired = 0,
                 parentMissing = 0, startupFiles = 0, startupDropped = 0, startupTruncated = 0,
                 startupNs = 0, trims = 0, trimBytes = 0, trimNs = 0, trimFailures = 0;
    } counters;

    ~Impl() { Stop(); }

    // ------------------------------------------------------------ helpers
    void Touch(File *file) {
        const uint64_t now = ++touchClock;
        if (file) file->ownTouch = now;
        for (File *f = file; f; f = f->parent) f->lastTouch = now;   // chain touch (sec. 5.4)
    }
    Json KvAt(const File *file, uint64_t index) const {
        while (file && index < file->header.parentKv) file = file->parent;
        if (!file) return Json();
        return file->ownKv[size_t(index - file->header.parentKv)];
    }
    void AssembleKv(const File *file, uint64_t count, Json::array &out) const {
        if (!file) return;
        const uint64_t inherited = std::min(count, file->header.parentKv);
        if (inherited > 0) AssembleKv(file->parent, inherited, out);
        for (uint64_t i = file->header.parentKv; i < count; ++i)
            out.push_back(file->ownKv[size_t(i - file->header.parentKv)]);
    }
    std::string KvDigest(const File *file, uint64_t count) const {
        Json::array kv; AssembleKv(file, count, kv);
        std::string text;
        for (const auto &item : kv) { text += item.dump_compact(); text.push_back('\n'); }
        return Sha256Hex(text);
    }
    const BlobLoc *FindBlob(const File *file, const std::string &digest, const File **owner) const {
        for (const File *f = file; f; f = f->parent) {
            auto it = f->blobs.find(digest);
            if (it != f->blobs.end() && !it->second.empty()) { if (owner) *owner = f; return &it->second.front(); }
        }
        return nullptr;
    }
    // Restore lookup: in the checkpoint's file only data chunks or its own (pinned) slot; in
    // ancestors only data chunks. Never a location in an unpinned slot that a writer may reuse.
    const BlobLoc *FindBlobFor(const File *file, int homeSlot, const std::string &digest, uint64_t bytes,
                               const File **owner) const {
        for (const File *f = file; f; f = f->parent) {
            auto it = f->blobs.find(digest);
            if (it == f->blobs.end()) continue;
            for (const auto &loc : it->second)
                if (loc.bytes == bytes && (loc.slot < 0 || (f == file && loc.slot == homeSlot))) { if (owner) *owner = f; return &loc; }
        }
        return nullptr;
    }
    static void DropSlotLocs(File &file, const std::string &digest, int slot) {
        auto it = file.blobs.find(digest);
        if (it == file.blobs.end()) return;
        auto &locs = it->second;
        locs.erase(std::remove_if(locs.begin(), locs.end(), [slot](const BlobLoc &l) { return l.slot == slot; }), locs.end());
        if (locs.empty()) file.blobs.erase(it);
    }
    static bool InChain(const File *file, const File *target) {
        for (const File *f = file; f; f = f->parent) if (f == target) return true;
        return false;
    }
    std::shared_ptr<Ckpt> FindCkpt(const std::string &identity, const std::string &key) const {
        auto it = ckpts.find({identity, key});
        if (it == ckpts.end() || it->second->retired) return nullptr;
        return it->second;
    }
    void PageRefs(const Ckpt &c, int delta) {
        auto &refs = pageRefs[c.identity];
        for (const auto &page : c.value["page_identity"].array_items()) {
            const std::string &hash = page.string_value();
            if (delta > 0) refs[hash]++;
            else { auto it = refs.find(hash); if (it != refs.end() && --it->second == 0) refs.erase(it); }
        }
    }
    void IndexCkpt(const std::shared_ptr<Ckpt> &c) {
        auto &slot = ckpts[{c->identity, c->key}];
        if (slot) PageRefs(*slot, -1);
        slot = c;
        PageRefs(*c, +1);
        c->file->ckpts.push_back(c);
    }
    void UnindexCkpt(const std::shared_ptr<Ckpt> &c, bool notify) {
        auto it = ckpts.find({c->identity, c->key});
        if (it != ckpts.end() && it->second == c) { ckpts.erase(it); PageRefs(*c, -1); }
        auto &list = c->file->ckpts;
        list.erase(std::remove(list.begin(), list.end(), c), list.end());
        if (notify && eraseObserver) {
            try { eraseObserver(c->key, c->identity); } catch (...) {}
        }
    }
    bool ParseCkpt(const std::string &text, File *file, int slot, std::shared_ptr<Ckpt> &out) {
        std::string error;
        Json record = Json::parse(text, error);
        if (!error.empty() || !record.is_object()) return false;
        auto c = std::make_shared<Ckpt>();
        c->key = record["key"].string_value(); c->identity = record["identity"].string_value();
        c->family = record["family"].string_value(); c->kind = record["kind"].string_value();
        c->prefix = record["prefix"].string_value(); c->provenance = record["provenance"].string_value();
        c->kvDtype = record["kv_dtype"].string_value(); c->length = record["length"].int_value();
        c->createdNs = uint64_t(record["created_ns"].number_value());
        c->kvCount = uint64_t(record["kv_count"].number_value());
        c->value = record["value"];
        c->stateDigest = Sha256Hex(text);
        c->metadata = Json(Json::object{{"state", c->stateDigest}, {"layout", c->value["layout"]},
                                        {"storage_format", 5}}).dump_compact();
        c->file = file; c->slot = slot;
        if (!IsDigest(c->key) || !IsDigest(c->identity) || c->kvCount > file->KvLength()) return false;
        out = c;
        return true;
    }
    std::string CkptJson(const Job &job, uint64_t kvCount, const Json &valueNoKv, int length,
                         const std::string &kind) const {
        return Json(Json::object{
            {"key", job.key}, {"identity", job.identity},
            {"family", job.value["family_identity"].is_string() ? job.value["family_identity"].string_value() : job.family},
            {"kind", kind}, {"length", length}, {"prefix", job.value["prefix"].string_value()},
            {"provenance", job.value["provenance"].string_value()},
            {"kv_dtype", job.value["kv_dtype"].string_value()},
            {"created_ns", double(WallNs())}, {"kv_count", double(kvCount)}, {"value", valueNoKv}}).dump_compact();
    }

    // ------------------------------------------------------------ disk writes
    // Every disk mutation goes through these so fault injection covers each step.
    bool WriteStep(const char *step, int fd, const std::vector<iovec> &iov, uint64_t offset, std::string &error) {
        if (dead) { error = "v5_store_dead"; return false; }
        const std::string fault = TakeFault(step);
        size_t total = 0; for (const auto &v : iov) total += v.iov_len;
        size_t limit = total;
        if (fault == "crash") { dead = true; error = "v5_failpoint_crash"; return false; }
        if (fault == "eio") { error = "EIO"; return false; }
        if (fault == "torn") limit = total / 2;
        size_t done = 0, index = 0, inner = 0;
        const uint64_t start = NowNs();
        while (done < limit) {
            std::vector<iovec> batch;
            size_t left = limit - done;
            for (size_t i = index; i < iov.size() && batch.size() < IOV_MAX && left > 0; ++i) {
                iovec v = iov[i];
                if (i == index) { v.iov_base = static_cast<char *>(v.iov_base) + inner; v.iov_len -= inner; }
                if (v.iov_len > left) v.iov_len = left;
                left -= v.iov_len;
                if (v.iov_len) batch.push_back(v);
            }
            const ssize_t w = pwritev(fd, batch.data(), int(batch.size()), off_t(offset + done));
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) { error = errno == ENOSPC ? "ENOSPC" : "EIO"; return false; }
            done += size_t(w);
            size_t advance = size_t(w);
            while (advance > 0 && index < iov.size()) {
                const size_t rest = iov[index].iov_len - inner;
                if (advance >= rest) { advance -= rest; ++index; inner = 0; }
                else { inner += advance; advance = 0; }
            }
        }
        counters.writeNs += NowNs() - start;
        if (fault == "torn") { dead = true; error = "v5_failpoint_torn"; return false; }
        return true;
    }
    bool SyncStep(const char *step, int fd, std::string &error) {
        if (dead) { error = "v5_store_dead"; return false; }
        const std::string fault = TakeFault(step);
        if (fault == "crash" || fault == "torn") { dead = true; error = "v5_failpoint_crash"; return false; }
        if (fault == "eio") { error = "EIO"; return false; }
        const uint64_t start = NowNs();
        if (fdatasync(fd) != 0) { error = "EIO"; return false; }
        counters.syncNs += NowNs() - start; ++counters.syncs;
        return true;
    }
    bool DirSyncStep(const fs::path &dir, std::string &error) {
        if (dead) { error = "v5_store_dead"; return false; }
        const std::string fault = TakeFault("dir_sync");
        if (fault == "crash" || fault == "torn") { dead = true; error = "v5_failpoint_crash"; return false; }
        if (fault == "eio") { error = "EIO"; return false; }
        if (!SyncDir(dir)) { error = "EIO"; return false; }
        ++counters.dirSyncs;
        return true;
    }
    std::vector<iovec> ChunkIov(ChunkImage &image) {
        static const std::string zeros(kBlock, '\0');
        std::vector<iovec> iov;
        iov.push_back({&image.header[0], image.header.size()});
        for (const auto &blob : image.blobs) {
            iov.push_back({const_cast<char *>(blob.payload->data()), blob.payload->size()});
            const size_t pad = size_t(RoundUp(blob.payload->size()) - blob.payload->size());
            if (pad) iov.push_back({const_cast<char *>(zeros.data()), pad});
        }
        return iov;
    }
    bool WriteHeader(const char *writeStep, const char *syncStep, int fd, File &file, Header next,
                     int &slotIndex, std::string &error) {
        next.generation = file.header.generation + 1;
        std::string bytes = EncodeHeader(next);
        const int target = file.header.generation == 0 ? 0 : 1 - file.activeHeader;
        std::vector<iovec> iov{{&bytes[0], bytes.size()}};
        if (!WriteStep(writeStep, fd, iov, uint64_t(target) * kHeaderBytes, error)) return false;
        if (!SyncStep(syncStep, fd, error)) return false;
        slotIndex = target;
        return true;
    }

    // ------------------------------------------------------------ eviction
    // sec. 6.1: unlink + fsync dir + drop index entries + release bytes; O(1)
    // per file. Only leaves (no live child) that are not pinned/leased and
    // whose checkpoints the page ledger allows to erase.
    // D8 (user 2026-11-07 "对很久没用的长对话，只截掉文件尾部，保留开头"; sec. 6.3, C12), with
    // FASTLLM_PREFIX_V5_TRIM=1: the eviction unit is the LAST data chunk of a file, chosen
    // least-recently-used like SGLang 8920463c90 radix_cache.py:563-573 (LRU among evictable leaves;
    // an ancestor only after its children are gone): the tail beyond every live branch point is the
    // leaf, ranked by its own use (ownTouch) when other files branch from this one. A one-chunk leaf
    // file is deleted whole. Never cut below what a live descendant uses (TrimFloor).
    static constexpr uint64_t kTrimLease = ~uint64_t(0);
    // Lowest byte offset of f that must stay: the chunk holding the last KV bundle each direct child
    // inherits, and every data blob any descendant checkpoint's state resolves to inside f.
    uint64_t TrimFloor(const File *f) const {
        uint64_t floor = f->header.dataStart;
        for (const auto &entry : files) {
            const File *d = entry.second.get();
            if (d == f || !InChain(d->parent, f)) continue;
            if (d->parent == f && d->header.parentKv > f->header.parentKv) {
                const size_t own = size_t(d->header.parentKv - f->header.parentKv);
                for (const auto &rec : f->chunks)
                    if (rec.kvEnd >= own) { floor = std::max(floor, rec.offset + rec.total); break; }
            }
            for (const auto &c : d->ckpts) {
                std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(c->value, refs);
                for (const auto &r : refs) {
                    const File *owner = nullptr;
                    const BlobLoc *loc = FindBlobFor(d, c->slot, r.first, r.second, &owner);
                    if (loc && owner == f && loc->slot < 0) floor = std::max(floor, loc->offset + loc->bytes);
                }
            }
        }
        return floor;
    }
    // Byte end of every chain file that checkpoint c needs: KV bundles below its count (and below each
    // child's cut for ancestors), its own data chunk, and its state blobs. A pin protects only this
    // prefix, as SGLang 8920463c90 radix_cache.py:587-614 locks only the matched nodes, so D8 may still
    // cut the tail of a pinned file beyond it.
    // full = the checkpoint value WITH its assembled "kv": every location a Load of it reads is covered.
    std::vector<std::pair<File *, uint64_t>> PinExtents(const Ckpt &c, const Json &full) const {
        std::vector<std::pair<File *, uint64_t>> out;
        std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(full, refs);
        uint64_t kvNeeded = c.kvCount;
        for (File *f = c.file; f; f = f->parent) {
            uint64_t end = f->header.dataStart;
            const uint64_t own = kvNeeded > f->header.parentKv ? kvNeeded - f->header.parentKv : 0;
            if (own > 0)
                for (const auto &rec : f->chunks)
                    if (rec.kvEnd >= own) { end = std::max(end, rec.offset + rec.total); break; }
            if (f == c.file && c.slot < 0)
                for (const auto &rec : f->chunks)
                    if (rec.offset == c.chunkOffset) { end = std::max(end, rec.offset + rec.total); break; }
            for (const auto &r : refs) {
                const File *owner = nullptr;
                const BlobLoc *loc = FindBlobFor(c.file, c.slot, r.first, r.second, &owner);
                if (loc && owner == f && loc->slot < 0) end = std::max(end, loc->offset + loc->bytes);
            }
            out.push_back({f, end});
            kvNeeded = std::min(kvNeeded, f->header.parentKv);
        }
        return out;
    }
    bool EvictOne(const std::set<const File *> &keep, std::string &reason) {
        File *victim = nullptr;
        bool trim = false;
        uint64_t victimPri = 0, cut = 0, cutKv = 0;
        const uint64_t start = NowNs();
        std::vector<std::shared_ptr<Ckpt>> dropped;
        std::unique_ptr<File> owned;
        {
            // Choose and unindex under one lock: PutMetadata may evict from another thread.
            std::lock_guard<std::mutex> lock(mu);
            for (auto &entry : files) {
                File *f = entry.second.get();
                if (keep.count(f) || busy.count(f) || f->slotPins[0] > 0 || f->slotPins[1] > 0 || f->lease != 0) continue;
                if (f->pins > 0 && !trimEnabled) continue;
                const bool whole = f->children == 0 && f->pins == 0 && (!trimEnabled || f->chunks.size() <= 1);
                if (!whole && (!trimEnabled || f->chunks.empty())) continue;
                const uint64_t pri = f->children > 0 ? f->ownTouch : f->lastTouch;
                if (victim && pri >= victimPri) continue;
                const uint64_t from = whole ? 0 : f->chunks.back().offset;
                if (!whole && from < TrimFloor(f)) continue;
                if (!whole && !f->pinExtents.empty() && from < *f->pinExtents.rbegin()) continue;   // pinned prefix
                bool allowed = true;
                if (reclaimPredicate)
                    for (const auto &c : f->ckpts)
                        if ((whole || c->slot >= 0 || c->chunkOffset >= from) && !reclaimPredicate(c->key, c->identity)) {
                            allowed = false; break;
                        }
                if (!allowed) continue;
                victim = f; victimPri = pri; trim = !whole; cut = from;
            }
            if (!victim) { reason = "v5_nothing_evictable"; return false; }
            if (trim) {
                // In memory first (readers can no longer find what goes), bytes after.
                for (const auto &c : std::vector<std::shared_ptr<Ckpt>>(victim->ckpts))
                    if (c->slot >= 0 || c->chunkOffset >= cut) { dropped.push_back(c); UnindexCkpt(c, false); }
                for (int s = 0; s < 2; ++s) {
                    for (const auto &d : victim->slotDigests[s]) DropSlotLocs(*victim, d, s);
                    victim->slotDigests[s].clear();
                    victim->slotCkpt[s].reset();
                }
                for (auto it = victim->blobs.begin(); it != victim->blobs.end();) {
                    auto &locs = it->second;
                    locs.erase(std::remove_if(locs.begin(), locs.end(),
                                              [cut](const BlobLoc &l) { return l.slot < 0 && l.offset >= cut; }), locs.end());
                    if (locs.empty()) it = victim->blobs.erase(it); else ++it;
                }
                victim->chunks.pop_back();
                cutKv = victim->chunks.empty() ? 0 : victim->chunks.back().kvEnd;
                victim->ownKv.resize(size_t(cutKv));
                victim->chunkSeq--;               // the next append reuses the cut chunk's sequence
                victim->lease = kTrimLease;       // no writer and no eviction until the cut is durable
            } else {
                dropped = victim->ckpts;
                for (const auto &c : dropped) UnindexCkpt(c, false);
                if (victim->parent) victim->parent->children--;
                auto it = files.find(victim->inc);
                owned = std::move(it->second);
                files.erase(it);
            }
        }
        if (trim) {
            // Crash-safe order (same as G4): durable header with the shorter committed length and empty
            // slots first, then cut the bytes. A crash between the two leaves bytes after committed,
            // which start-up cuts (Kafka LogSegment.recover).
            std::string error;
            bool headerOk = false, cutOk = false;
            Fd fd(open(victim->path.c_str(), O_RDWR | O_CLOEXEC));
            struct timespec keepMtime{0, UTIME_OMIT};
            if (fd.fd >= 0) { struct stat st{}; if (fstat(fd.fd, &st) == 0) keepMtime = st.st_mtim; }
            if (fd.fd >= 0) {
                Header next = victim->header;
                next.committed = cut;
                next.slotGen[0] = next.slotGen[1] = 0; next.slotBytes[0] = next.slotBytes[1] = 0;
                int headerIndex = victim->activeHeader;
                headerOk = WriteHeader("trim_header_write", "trim_header_sync", fd.fd, *victim, next, headerIndex, error);
                if (headerOk) {
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        next.generation = victim->header.generation + 1;
                        victim->header = next; victim->activeHeader = headerIndex;
                    }
                    const std::string fault = TakeFault("trim_truncate");
                    if (fault == "crash") dead = true;     // test: crash between the header and the cut
                    cutOk = !dead && fault.empty() && ftruncate(fd.fd, off_t(cut)) == 0 && fdatasync(fd.fd) == 0;
                    // A trim is not a use: keep the file's write time so the restart LRU order holds.
                    if (!dead) { const struct timespec times[2] = {{0, UTIME_OMIT}, keepMtime}; futimens(fd.fd, times); }
                }
            }
            const uint64_t phys = fd.fd >= 0 ? PhysicalOfFd(fd.fd) : victim->physical;
            const uint64_t ns = NowNs() - start;
            uint64_t freed = 0, kvLeft = 0;
            std::string name;
            {
                std::lock_guard<std::mutex> lock(mu);
                // once the lease is cleared another thread may evict and free this File;
                // copy what the log needs before that.
                name = victim->path.filename().string();
                kvLeft = victim->header.parentKv + cutKv;
                freed = victim->physical > phys ? victim->physical - phys : 0;
                usedBytes += phys; usedBytes -= std::min(usedBytes, victim->physical); victim->physical = phys;
                // A failed header write leaves the old header valid on disk while memory is already cut:
                // never write this file again in this process (it stays readable below the cut).
                if (headerOk) victim->lease = 0; else ++counters.trimFailures;
                ++counters.trims; counters.trimBytes += freed; counters.trimNs += ns;
            }
            if (eraseObserver) for (const auto &c : dropped) { try { eraseObserver(c->key, c->identity); } catch (...) {} }
            std::fprintf(stderr, "[Prefix SSD V5] trim: file=%s cut_offset=%llu kv_left=%llu freed=%llu checkpoints=%zu "
                         "header=%d truncate=%d ms=%.3f.\n", name.c_str(), (unsigned long long)cut,
                         (unsigned long long)kvLeft, (unsigned long long)freed, dropped.size(),
                         headerOk ? 1 : 0, cutOk ? 1 : 0, Ms(ns));
            return true;
        }
        const fs::path path = owned->path;
        const uint64_t bytes = owned->physical;
        const bool unlinked = unlink(path.c_str()) == 0 || errno == ENOENT;
        SyncDir(path.parent_path());
        const uint64_t ns = NowNs() - start;
        {
            std::lock_guard<std::mutex> lock(mu);
            usedBytes -= std::min(usedBytes, bytes);
            ++counters.deletes; counters.deleteNs += ns;
            counters.deleteMaxNs = std::max(counters.deleteMaxNs, ns);
        }
        if (eraseObserver) for (const auto &c : dropped) { try { eraseObserver(c->key, c->identity); } catch (...) {} }
        std::fprintf(stderr, "[Prefix SSD V5] delete: file=%s bytes=%llu checkpoints=%zu unlink=%d ms=%.3f.\n",
                     path.filename().c_str(), (unsigned long long)bytes, dropped.size(), unlinked ? 1 : 0, Ms(ns));
        return true;
    }

    // ------------------------------------------------------------ commit
    bool Commit(Job &job, std::string &error) {
        const uint64_t commitStart = NowNs();
        const std::string kind = job.value["kind"].is_string() ? job.value["kind"].string_value() : "prefix";
        const int length = job.value["length"].int_value();
        {
            std::lock_guard<std::mutex> lock(mu);
            if (FindCkpt(job.identity, job.key)) { ++counters.dedup; return true; }   // V4 keepEquivalent no-op
        }
        std::unordered_map<std::string, const std::string *> staged;
        for (const auto &item : job.staged) staged[item.first] = &item.second;
        const Json::array kv = job.value["kv"].array_items();
        // m = first bundle that carries a staged ref; all later bundles must be staged.
        size_t m = kv.size();
        for (size_t i = 0; i < kv.size(); ++i) {
            std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(kv[i], refs);
            bool any = false;
            for (const auto &r : refs) if (staged.count(r.first)) { any = true; break; }
            if (any) { m = i; break; }
        }
        for (size_t i = m; i < kv.size(); ++i) {
            std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(kv[i], refs);
            for (const auto &r : refs) {
                auto it = staged.find(r.first);
                if (it == staged.end() || it->second->size() != r.second) { error = "v5_tail_not_staged"; return false; }
            }
        }
        Json::object valueFields = job.value.object_items();
        valueFields.erase("kv");
        const Json valueNoKv(valueFields);
        std::vector<std::pair<std::string, uint64_t>> stateRefs; CollectRefs(valueNoKv, stateRefs);
        // H21 / D10: a prompt end on the 3072 interval grid is an interval state and stays permanent;
        // only off-grid prompt ends use the ping-pong slots.
        const bool promptEnd = job.value["v5_prompt_end"].bool_value() && length % intervalTokens != 0;

        // ---- choose target file (sec. 4.2) ----
        File *target = nullptr, *parent = nullptr;
        bool fresh = false;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (m > 0) {
                // Locate the file that owns bundle m-1 and holds kv[0:m] as its prefix.
                std::vector<std::pair<std::string, uint64_t>> lastRefs; CollectRefs(kv[m - 1], lastRefs);
                File *owner = nullptr;
                for (auto &entry : files) {
                    File *f = entry.second.get();
                    if (f->identity != job.identity || f->KvLength() < m || m - 1 < f->header.parentKv) continue;
                    if (!(f->ownKv[size_t(m - 1 - f->header.parentKv)] == kv[m - 1])) continue;
                    bool same = true;
                    for (size_t i = 0; i < m && same; ++i) same = KvAt(f, i) == kv[i];
                    if (same) { owner = f; break; }
                }
                if (!owner) { ++counters.parentMissing; error = "v5_parent_missing"; return false; }
                const bool leasedByOther = owner->lease != 0 && owner->lease != job.writer->id;
                if (owner->KvLength() == m && !leasedByOther) target = owner;
                else parent = owner;
            }
            if (!target) fresh = true;
            for (const File *f = target ? target : parent; f; f = f->parent) busy.insert(f);
        }
        struct BusyClear {
            Impl *impl;
            ~BusyClear() { std::lock_guard<std::mutex> lock(impl->mu); impl->busy.clear(); }
        } busyClear{this};

        // ---- build chunks ----
        std::vector<Pending> kvBlobs, stateBlobs;
        std::set<std::string> seen;
        Json::array tail(kv.begin() + m, kv.end());
        for (const auto &bundle : tail) {
            std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(bundle, refs);
            for (const auto &r : refs) if (seen.insert(r.first).second) kvBlobs.push_back({r.first, staged[r.first]});
        }
        for (const auto &r : stateRefs) {
            auto it = staged.find(r.first);
            if (it == staged.end()) continue;              // must already exist in the chain (checked below)
            if (it->second->size() != r.second) { error = "v5_state_size"; return false; }
            if (seen.insert(r.first).second) stateBlobs.push_back({r.first, it->second});
        }
        uint64_t stateBytes = 0;
        for (const auto &b : stateBlobs) stateBytes += RoundUp(b.payload->size());
        const std::string tailJson = Json(tail).dump_compact();
        const std::string ckptJson = CkptJson(job, kv.size(), valueNoKv, length, kind);
        const std::string emptyKv = "[]";

        // Slot decision (sec. 4.6, C2): prompt-end state into the inactive slot unless held.
        int useSlot = -1;
        uint64_t slotCap = 0;
        if (fresh) {
            if (kind == "prefix" && stateBytes > 0)
                slotCap = RoundUp((RoundUp(kChunkFixed + stateBlobs.size() * kEntryBytes + ckptJson.size()) + stateBytes) * 5 / 4 + (64u << 10));
        } else {
            slotCap = target->header.slotCap;
        }
        const uint64_t slotNeed = RoundUp(kChunkFixed + stateBlobs.size() * kEntryBytes + emptyKv.size() + ckptJson.size()) + stateBytes;
        std::shared_ptr<Ckpt> displaced;
        File *slotWritingFile = nullptr;
        if (promptEnd && slotCap > 0 && slotNeed <= slotCap) {
            std::lock_guard<std::mutex> lock(mu);
            if (fresh) useSlot = 0;
            else {
                const int inactive = target->header.slotGen[0] <= target->header.slotGen[1] ? 0 : 1;
                displaced = target->slotCkpt[inactive];
                bool held = target->slotPins[inactive] > 0;
                if (!held && displaced && reclaimPredicate && !reclaimPredicate(displaced->key, displaced->identity)) held = true;
                if (!held) { useSlot = inactive; target->slotWriting[inactive] = true; slotWritingFile = target; }
                else { ++counters.slotFallbacks; displaced.reset(); }
            }
        }
        // Clears the overwrite mark on every exit (skip, failure, success); declared before any later lock
        // scope in this function, so it runs after they are released.
        struct SlotWritingGuard {
            Impl *impl; File *&file; int slot;
            ~SlotWritingGuard() { if (file) { std::lock_guard<std::mutex> l(impl->mu); file->slotWriting[slot] = false; } }
        } slotWritingGuard{this, slotWritingFile, useSlot};
        const bool dataHasCkpt = useSlot < 0;
        const bool needData = !tail.empty() || dataHasCkpt;

        // ---- quota (sec. 4.1, C5, G2: bytes are ready in host memory now) ----
        uint64_t needed = 0;
        if (fresh) needed += kSlotBase;
        else needed += kBlock;   // the commit writes the inactive header block, which may be a hole
        if (needData) {
            uint64_t data = RoundUp(kChunkFixed + (kvBlobs.size() + (dataHasCkpt ? stateBlobs.size() : 0)) * kEntryBytes +
                                    tailJson.size() + (dataHasCkpt ? ckptJson.size() : 0));
            for (const auto &b : kvBlobs) data += RoundUp(b.payload->size());
            if (dataHasCkpt) data += stateBytes;
            needed += data;
        }
        if (useSlot >= 0) needed += slotNeed;
        if (needed > quota) { ++counters.skippedTooLarge; error = "v5_save_larger_than_quota"; return false; }
        {
            std::set<const File *> keep;
            if (target) for (const File *f = target; f; f = f->parent) keep.insert(f);
            if (parent) for (const File *f = parent; f; f = f->parent) keep.insert(f);
            for (;;) {
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (usedBytes + needed <= quota) break;
                }
                std::string reason;
                if (!EvictOne(keep, reason)) {
                    std::lock_guard<std::mutex> lock(mu);
                    ++counters.skippedQuota;
                    std::fprintf(stderr, "[Prefix SSD V5] save-skipped: reason=quota_full key=%s used=%llu needed=%llu quota=%llu.\n",
                                 job.key.c_str(), (unsigned long long)usedBytes, (unsigned long long)needed,
                                 (unsigned long long)quota);
                    error = "v5_quota_skip";
                    return false;
                }
            }
        }
        // Every state ref that was not staged must resolve in the chain.
        {
            std::lock_guard<std::mutex> lock(mu);
            const File *chain = target ? target : parent;
            for (const auto &r : stateRefs)
                if (!staged.count(r.first) && !(chain && FindBlob(chain, r.first, nullptr))) { error = "v5_state_ref_missing"; return false; }
        }

        // ---- write (sec. 4 steps 4-6) ----
        std::unique_ptr<File> created;
        File *file = target;
        if (fresh) {
            created.reset(new File());
            created->identity = job.identity;
            {
                std::lock_guard<std::mutex> lock(mu);
                do { created->inc = rng(); } while (created->inc == 0 || files.count(created->inc));
            }
            created->parent = parent;
            created->header.incarnation = created->inc;
            created->header.identity = job.identity;
            created->header.slotCap = slotCap;
            created->header.dataStart = kSlotBase + 2 * slotCap;
            created->header.committed = created->header.dataStart;
            created->header.createdNs = WallNs();
            if (parent) {
                created->header.parentInc = parent->inc;
                created->header.parentKv = m;
                std::lock_guard<std::mutex> lock(mu);
                created->header.parentKvDigest = KvDigest(parent, m);
            }
            const fs::path dir = root / job.identity;
            std::error_code ec; fs::create_directories(dir, ec);
            created->path = dir / IncName(created->inc);
            file = created.get();
        }
        const int openFlags = fresh ? (O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC) : (O_RDWR | O_CLOEXEC);
        Fd fd(open(file->path.c_str(), openFlags, 0600));
        if (fd.fd < 0) { error = "v5_open_failed"; return false; }
        // Any failure after this point: a fresh file that never reached the index is
        // removed (nothing in it was acknowledged); a file already in the index keeps its committed
        // state and its physical bytes are recounted, so the quota sees every allocated block.
        struct FailGuard {
            Impl *impl; File *file; std::unique_ptr<File> *created; int fd; bool ok = false;
            ~FailGuard() {
                if (ok || impl->dead) return;
                if (*created) { unlink(file->path.c_str()); SyncDir(file->path.parent_path()); return; }
                const uint64_t phys = PhysicalOfFd(fd);
                std::lock_guard<std::mutex> lock(impl->mu);
                impl->usedBytes += phys; impl->usedBytes -= std::min(impl->usedBytes, file->physical); file->physical = phys;
            }
        } failGuard{this, file, &created, fd.fd};
        Header next = file->header;
        const uint64_t oldCommitted = file->header.committed;
        ChunkImage dataImage;
        std::vector<Pending> dataBlobs = kvBlobs;
        if (dataHasCkpt) dataBlobs.insert(dataBlobs.end(), stateBlobs.begin(), stateBlobs.end());
        // G1/G4: committed bytes, the valid header and the valid slot are never
        // touched. A failed data write drops the uncommitted tail (frees ENOSPC
        // space). A failed header write keeps the tail: if that header reached the
        // disk anyway, the tail is what it points to. A dead (crashed) store leaves
        // the disk as is; start-up recovery handles it.
        auto dropTail = [&]() {
            if (dead) return;
            if (fresh) { unlink(file->path.c_str()); SyncDir(file->path.parent_path()); return; }
            if (ftruncate(fd.fd, off_t(oldCommitted)) == 0) fdatasync(fd.fd);
        };
        int headerIndex = file->activeHeader;
        bool dirSynced = false;
        if (needData) {
            const uint64_t crcStart = NowNs();
            dataImage = BuildChunk(file->inc, file->chunkSeq + 1, dataBlobs, tailJson, dataHasCkpt ? ckptJson : "");
            counters.crcNs += NowNs() - crcStart;
            auto iov = ChunkIov(dataImage);
            if (!WriteStep("data_write", fd.fd, iov, file->header.committed, error) ||
                !SyncStep("data_sync", fd.fd, error)) { dropTail(); return false; }
            next.committed = file->header.committed + dataImage.total;
            if (!WriteHeader("header_write", "header_sync", fd.fd, *file, next, headerIndex, error)) return false;
            if (fresh) {
                if (!DirSyncStep(file->path.parent_path(), error)) return false;
                dirSynced = true;
            }
        }
        // Publish the data chunk in memory (header is durable now).
        auto publishData = [&](Header &durable) {
            file->header = durable; file->activeHeader = headerIndex;
            if (needData) {
                file->chunkSeq++;
                for (const auto &bundle : tail) file->ownKv.push_back(bundle);
                file->chunks.push_back(File::ChunkRec{oldCommitted, dataImage.total, file->ownKv.size()});
                for (size_t i = 0; i < dataImage.blobs.size(); ++i)
                    file->blobs[dataImage.blobs[i].digest].push_back(
                        BlobLoc{oldCommitted + dataImage.rel[i], dataImage.blobs[i].payload->size(),
                                Crc32c(dataImage.blobs[i].payload->data(), dataImage.blobs[i].payload->size()), -1});
            }
        };
        std::shared_ptr<Ckpt> record;
        if (needData) {
            Header durable = next; durable.generation = file->header.generation + 1;
            std::lock_guard<std::mutex> lock(mu);
            publishData(durable);
            if (fresh) {
                if (parent) parent->children++;
                files[file->inc] = std::move(created);
                ++counters.newFiles;
                if (parent) ++counters.branches;
            } else ++counters.appends;
            if (dataHasCkpt) {
                ParseCkpt(ckptJson, file, -1, record);
                record->chunkOffset = oldCommitted;
                IndexCkpt(record);
            }
            file->lease = job.writer->id;
            Touch(file);
            const uint64_t phys = PhysicalOfFd(fd.fd);
            usedBytes += phys; usedBytes -= std::min(usedBytes, file->physical); file->physical = phys;
        }
        if (useSlot >= 0) {
            // The displaced (older) prompt-end state leaves the index before its bytes change.
            {
                std::lock_guard<std::mutex> lock(mu);
                if (displaced) { UnindexCkpt(displaced, false); file->slotCkpt[useSlot].reset(); }
                for (const auto &d : file->slotDigests[useSlot]) DropSlotLocs(*file, d, useSlot);
                file->slotDigests[useSlot].clear();
            }
            if (displaced && !displaced->retired && eraseObserver) {
                try { eraseObserver(displaced->key, displaced->identity); } catch (...) {}
            }
            const uint64_t slotGen = std::max(file->header.slotGen[0], file->header.slotGen[1]) + 1;
            const uint64_t crcStart = NowNs();
            ChunkImage slotImage = BuildChunk(file->inc, slotGen, stateBlobs, emptyKv, ckptJson);
            counters.crcNs += NowNs() - crcStart;
            const uint64_t slotOffset = kSlotBase + uint64_t(useSlot) * file->header.slotCap;
            // Payload first, slot header block last (so a torn slot never shows a new gen).
            auto iov = ChunkIov(slotImage);
            std::vector<iovec> body(iov.begin() + 1, iov.end()), head(iov.begin(), iov.begin() + 1);
            if (!WriteStep("slot_write", fd.fd, body, slotOffset + slotImage.header.size(), error) ||
                !WriteStep("slot_write", fd.fd, head, slotOffset, error) ||
                !SyncStep("slot_sync", fd.fd, error)) {
                if (!needData) dropTail();
                return false;
            }
            Header slotHeader = file->header;
            if (!needData) slotHeader = next;
            slotHeader.slotGen[useSlot] = slotGen;
            slotHeader.slotBytes[useSlot] = slotImage.total;
            if (!WriteHeader("slot_header_write", "slot_header_sync", fd.fd, *file, slotHeader, headerIndex, error)) return false;
            if (fresh && !dirSynced && !DirSyncStep(file->path.parent_path(), error)) return false;
            Header durable = slotHeader; durable.generation = file->header.generation + 1;
            std::lock_guard<std::mutex> lock(mu);
            file->header = durable; file->activeHeader = headerIndex;
            if (fresh && !needData) {
                if (parent) parent->children++;
                files[file->inc] = std::move(created);
                ++counters.newFiles;
                if (parent) ++counters.branches;
            }
            for (size_t i = 0; i < slotImage.blobs.size(); ++i) {
                file->blobs[slotImage.blobs[i].digest].push_back(
                    BlobLoc{slotOffset + slotImage.rel[i], slotImage.blobs[i].payload->size(),
                            Crc32c(slotImage.blobs[i].payload->data(), slotImage.blobs[i].payload->size()), useSlot});
                file->slotDigests[useSlot].push_back(slotImage.blobs[i].digest);
            }
            ParseCkpt(ckptJson, file, useSlot, record);
            file->slotCkpt[useSlot] = record;
            IndexCkpt(record);
            ++counters.slotWrites;
            file->lease = job.writer->id;
            Touch(file);
            const uint64_t phys = PhysicalOfFd(fd.fd);
            usedBytes += phys; usedBytes -= std::min(usedBytes, file->physical); file->physical = phys;
        }
        failGuard.ok = true;
        // Admission uses computed sizes; the filesystem may allocate a block more (random model,
        // trim mode: +4096 B). Recheck the measured bytes and evict LRU until the quota holds (I2).
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(mu);
                if (usedBytes <= quota) break;
            }
            std::set<const File *> keepChain;
            for (const File *f = file; f; f = f->parent) keepChain.insert(f);
            std::string reason;
            if (!EvictOne(keepChain, reason)) break;
        }
        std::lock_guard<std::mutex> lock(mu);
        const uint64_t ns = NowNs() - commitStart;
        ++counters.commits; counters.commitBytes += needed; counters.durableNs += ns;
        counters.durableMaxNs = std::max(counters.durableMaxNs, ns);
        return true;
    }

    // ------------------------------------------------------------ writer thread
    void Run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) { if (stopping) break; continue; }
                job = queue.front(); queue.pop_front(); active = job;
            }
            std::string error;
            bool ok = false;
            const uint64_t startNs = NowNs();
            try { ok = Commit(*job, error); }
            catch (const std::exception &e) { error = e.what(); ok = false; }
            if (!ok && error != "v5_quota_skip" && error != "v5_save_larger_than_quota") {
                std::fprintf(stderr, "[Prefix SSD V5] save-failed: key=%s reason=%s.\n", job->key.c_str(), error.c_str());
                std::lock_guard<std::mutex> lock(mu);
                ++counters.failures;
                if (error != "v5_parent_missing" && error != "v5_tail_not_staged") {
                    ++drainFailures; if (firstFailure.empty()) firstFailure = error;
                }
            }
            if (ok) {
                const uint64_t endNs = NowNs();
                std::fprintf(stderr, "[Prefix SSD V5] durable: key=%s tokens=%d staged_bytes=%llu queue_ms=%.1f commit_ms=%.1f "
                             "accept_to_durable_ms=%.1f.\n", job->key.c_str(), job->value["length"].int_value(),
                             (unsigned long long)job->bytes, Ms(startNs - job->submitNs), Ms(endNs - startNs),
                             Ms(endNs - job->submitNs));
            }
            auto completion = std::move(job->completion);
            {
                std::lock_guard<std::mutex> lock(mu);
                inflight -= std::min(inflight, job->bytes);
                active.reset();
            }
            job->staged.clear();
            if (completion) { try { completion(ok); } catch (...) {} }
            job.reset();
            idle.notify_all();
        }
    }
    void Stop() {
        {
            std::lock_guard<std::mutex> lock(mu);
            if (stopping) { return; }
        }
        std::string ignored;
        DrainBounded(ignored);
        {
            std::lock_guard<std::mutex> lock(mu);
            stopping = true;
        }
        cv.notify_all();
        if (thread.joinable()) thread.join();
    }
    bool DrainBounded(std::string &error) {
        std::vector<std::shared_ptr<Job>> abandoned;
        {
            std::unique_lock<std::mutex> lock(mu);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(drainMs);
            idle.wait_until(lock, deadline, [&] { return queue.empty() && !active; });
            while (!queue.empty()) { abandoned.push_back(queue.front()); inflight -= std::min(inflight, queue.front()->bytes); queue.pop_front(); }
            counters.abandoned += abandoned.size();
            // The active save finishes or fails on its own (bounded by I/O).
            idle.wait(lock, [&] { return !active; });
        }
        for (auto &job : abandoned) if (job->completion) { try { job->completion(false); } catch (...) {} }
        std::lock_guard<std::mutex> lock(mu);
        bool ok = abandoned.empty() && drainFailures == 0;
        if (!abandoned.empty()) {
            std::fprintf(stderr, "[Prefix SSD V5] drain: abandoned=%zu (not durable, never partial).\n", abandoned.size());
            error = "v5_drain_abandoned:" + std::to_string(abandoned.size());
        } else if (drainFailures) error = firstFailure.empty() ? "v5_commit_failed" : firstFailure;
        drainFailures = 0; firstFailure.clear();
        return ok;
    }

    // ------------------------------------------------------------ start-up (sec. 7)
    bool Open(std::string &error) {
        const uint64_t start = NowNs();
        std::error_code ec;
        fs::create_directories(root, ec);
        if (ec) { error = "v5_root_create"; return false; }
        lockFd = Fd(open((root / "LOCK").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600));
        if (lockFd.fd < 0 || flock(lockFd.fd, LOCK_EX | LOCK_NB) != 0) { error = "v5_lock_held"; return false; }   // C4
        const fs::path format = root / "FORMAT";
        if (!fs::exists(format)) {
            Fd fd(open(format.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
            if (fd.fd < 0 || write(fd.fd, kFormatText, strlen(kFormatText)) != ssize_t(strlen(kFormatText)) ||
                fdatasync(fd.fd) != 0) { error = "v5_format_create"; return false; }
            SyncDir(root);
        }
        fs::create_directories(root / "meta", ec);
        // Count every byte under the root; only valid V5 files are served (C5).
        std::vector<fs::path> candidates;
        for (auto it = fs::recursive_directory_iterator(root, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (!it->is_regular_file(ec)) continue;
            const auto &path = it->path();
            if (path.extension() == ".v5f" && path.parent_path().parent_path() == root) candidates.push_back(path);
            else otherBytes += PhysicalOf(path);
        }
        std::map<uint64_t, std::unique_ptr<File>> loaded;
        std::vector<std::pair<std::shared_ptr<Ckpt>, int>> records;
        for (const auto &path : candidates) {
            std::unique_ptr<File> file(new File());
            std::string why;
            if (!LoadFile(path, *file, why)) {
                if (why == "open" || why == "unknown_version") {   // not ours to judge: count, never serve, never delete
                    otherBytes += PhysicalOf(path);
                    std::fprintf(stderr, "[Prefix SSD V5] startup-skip: file=%s reason=%s.\n", path.c_str(), why.c_str());
                    continue;
                }
                ++counters.startupDropped;
                std::fprintf(stderr, "[Prefix SSD V5] startup-drop: file=%s reason=%s.\n", path.c_str(), why.c_str());
                unlink(path.c_str()); SyncDir(path.parent_path());
                continue;
            }
            ++counters.startupFiles;
            loaded[file->inc] = std::move(file);
        }
        // Resolve parents; a child whose parent is missing or changed is an orphan
        // (never served). Repeat until stable so orphans of orphans go too.
        // Link every parent pointer first: a grandchild may be visited before its
        // parent, and the prefix digest walks the whole chain.
        for (auto &entry : loaded) {
            File *f = entry.second.get();
            auto p = loaded.find(f->header.parentInc);
            f->parent = (f->header.parentInc != 0 && p != loaded.end()) ? p->second.get() : nullptr;
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = loaded.begin(); it != loaded.end();) {
                File *f = it->second.get();
                bool ok = true;
                const char *why = "";
                if (f->header.parentInc != 0) {
                    auto p = loaded.find(f->header.parentInc);
                    ok = p != loaded.end() && p->second->identity == f->identity &&
                         p->second->KvLength() >= f->header.parentKv;
                    why = p == loaded.end() ? "parent_missing" : "parent_short";
                    if (ok) {
                        ok = KvDigest(f->parent, f->header.parentKv) == f->header.parentKvDigest;
                        why = "parent_digest";
                    }
                }
                if (!ok) {
                    std::fprintf(stderr, "[Prefix SSD V5] startup-drop: file=%s reason=orphan:%s parent=%016llx kv=%llu.\n",
                                 f->path.c_str(), why, (unsigned long long)f->header.parentInc,
                                 (unsigned long long)f->header.parentKv);
                    ++counters.startupDropped;
                    unlink(f->path.c_str()); SyncDir(f->path.parent_path());
                    for (auto &other : loaded) if (other.second->parent == f) other.second->parent = nullptr;
                    it = loaded.erase(it); changed = true;
                } else ++it;
            }
        }
        for (auto &entry : loaded) {
            File *f = entry.second.get();
            if (f->parent) f->parent->children++;
            usedBytes += f->physical;
        }
        files = std::move(loaded);
        // LRU order survives a restart: seed it from each file's last write time, oldest first, as
        // SGLang 8920463c90 storage/file/lru_file_evictor.py:281-301 ("Seed LRU index from disk on
        // startup (oldest mtime first)"); a parent is at least as recent as its newest descendant
        // (chain touch, sec. 5.4). Without this every file starts at 0 (stab3 s2 trimmed the newest).
        {
            std::vector<File *> order;
            for (auto &entry : files) order.push_back(entry.second.get());
            std::stable_sort(order.begin(), order.end(), [](const File *a, const File *b) {
                return a->mtime.tv_sec != b->mtime.tv_sec ? a->mtime.tv_sec < b->mtime.tv_sec : a->mtime.tv_nsec < b->mtime.tv_nsec;
            });
            for (File *f : order) { f->ownTouch = ++touchClock; f->lastTouch = f->ownTouch; }
            for (File *f : order)
                for (File *p = f->parent; p; p = p->parent) p->lastTouch = std::max(p->lastTouch, f->lastTouch);
        }
        for (auto &entry : files)
            for (auto &c : entry.second->ckpts) {
                auto key = std::make_pair(c->identity, c->key);
                if (!ckpts.count(key)) { ckpts[key] = c; PageRefs(*c, +1); }
            }
        usedBytes += otherBytes;
        counters.startupNs = NowNs() - start;
        std::fprintf(stderr, "[Prefix SSD V5] startup: root=%s files=%zu checkpoints=%zu used=%llu other=%llu "
                     "dropped=%llu truncated=%llu ms=%.3f.\n", root.c_str(), files.size(), ckpts.size(),
                     (unsigned long long)usedBytes, (unsigned long long)otherBytes,
                     (unsigned long long)counters.startupDropped, (unsigned long long)counters.startupTruncated,
                     Ms(counters.startupNs));
        thread = std::thread([this] { Run(); });
        return true;
    }
    static bool ParseCkptLater(const std::string &text) {
        std::string error; const Json record = Json::parse(text, error);
        return error.empty() && record.is_object();
    }
    bool LoadFile(const fs::path &path, File &file, std::string &why) {
        Fd fd(open(path.c_str(), O_RDWR | O_CLOEXEC));
        if (fd.fd < 0) { why = "open"; return false; }
        Header h[2]; bool valid[2] = {false, false}, otherVersion = false;
        for (int i = 0; i < 2; ++i) {
            std::string bytes(kHeaderBytes, '\0');
            const bool read = PreadAll(fd.fd, &bytes[0], kHeaderBytes, uint64_t(i) * kHeaderBytes);
            valid[i] = read && DecodeHeader(bytes, h[i]);
            // C5: a V5-family header of another format version is counted, never
            // served and never deleted.
            if (read && memcmp(bytes.data(), kHeaderMagic, 7) == 0 && Get32(bytes, 8) != kFormatVersion) otherVersion = true;
        }
        if (!valid[0] && !valid[1]) { why = otherVersion ? "unknown_version" : "no_valid_header"; return false; }
        const int pick = (valid[0] && (!valid[1] || h[0].generation > h[1].generation)) ? 0 : 1;
        file.header = h[pick]; file.activeHeader = pick;
        file.inc = file.header.incarnation; file.identity = file.header.identity; file.path = path;
        if (path.filename() != IncName(file.inc) || path.parent_path().filename() != file.identity) { why = "name_mismatch"; return false; }
        struct stat st{};
        fstat(fd.fd, &st);
        file.mtime = st.st_mtim;
        if (uint64_t(st.st_size) > file.header.committed) {
            // Kafka LogSegment.recover: cut bytes after the committed length.
            if (ftruncate(fd.fd, off_t(file.header.committed)) != 0 || fdatasync(fd.fd) != 0) { why = "truncate"; return false; }
            ++counters.startupTruncated;
        }
        uint64_t offset = file.header.dataStart;
        while (offset < file.header.committed) {
            ParsedChunk chunk; std::string error;
            std::string parseError;
            Json kv;
            const bool good = ReadChunk(fd.fd, offset, file.header.committed, chunk, error) &&
                              chunk.incarnation == file.inc && chunk.sequence == file.chunkSeq + 1 &&
                              (kv = Json::parse(chunk.kvJson, parseError), parseError.empty() && kv.is_array());
            std::shared_ptr<Ckpt> c;
            if (good && !chunk.ckptJson.empty() && !ParseCkptLater(chunk.ckptJson)) error = "chunk_ckpt_json";
            if (!good || !error.empty()) {
                // Serve what precedes the bad chunk; the next append overwrites it.
                std::fprintf(stderr, "[Prefix SSD V5] startup: %s cut at offset %llu (%s).\n", path.filename().c_str(),
                             (unsigned long long)offset, error.empty() ? "chunk" : error.c_str());
                file.header.committed = offset;
                break;
            }
            for (const auto &bundle : kv.array_items()) file.ownKv.push_back(bundle);
            for (const auto &e : chunk.entries) file.blobs[e.digest].push_back(BlobLoc{offset + e.rel, e.bytes, e.crc, -1});
            file.chunkSeq = chunk.sequence;
            file.chunks.push_back(File::ChunkRec{offset, chunk.total, file.ownKv.size()});
            if (!chunk.ckptJson.empty() && ParseCkpt(chunk.ckptJson, &file, -1, c)) { c->chunkOffset = offset; file.ckpts.push_back(c); }
            offset += chunk.total;
        }
        for (int s = 0; s < 2; ++s) {
            if (file.header.slotGen[s] == 0) continue;
            const uint64_t slotOffset = kSlotBase + uint64_t(s) * file.header.slotCap;
            ParsedChunk chunk; std::string error;
            std::shared_ptr<Ckpt> c;
            if (ReadChunk(fd.fd, slotOffset, slotOffset + file.header.slotCap, chunk, error) &&
                chunk.incarnation == file.inc && chunk.sequence == file.header.slotGen[s] &&
                chunk.total == file.header.slotBytes[s] && ParseCkpt(chunk.ckptJson, &file, s, c)) {
                for (const auto &e : chunk.entries) {
                    file.blobs[e.digest].push_back(BlobLoc{slotOffset + e.rel, e.bytes, e.crc, s});
                    file.slotDigests[s].push_back(e.digest);
                }
                file.slotCkpt[s] = c;
                file.ckpts.push_back(c);
            } else {
                std::fprintf(stderr, "[Prefix SSD V5] startup: slot %d of %s not served (%s).\n", s,
                             path.filename().c_str(), error.empty() ? "generation" : error.c_str());
            }
        }
        file.physical = PhysicalOfFd(fd.fd);
        return true;
    }
};

// =========================================================================
DiskPrefixV5Store::Writer::Writer(std::weak_ptr<Impl> owner_, uint64_t id_) : owner(std::move(owner_)), id(id_) {}
DiskPrefixV5Store::Writer::~Writer() {
    auto impl = owner.lock();
    if (!impl) return;
    std::lock_guard<std::mutex> lock(impl->mu);
    for (auto &entry : impl->files) if (entry.second->lease == id) entry.second->lease = 0;
}

DiskPrefixV5Store::DiskPrefixV5Store(std::shared_ptr<Impl> value) : impl(std::move(value)) {}
DiskPrefixV5Store::~DiskPrefixV5Store() { if (impl) impl->Stop(); }

std::shared_ptr<DiskPrefixV5Store> DiskPrefixV5Store::OpenShared(const std::string &directory, uint64_t quotaBytes,
                                                                 std::string &error) {
    error.clear();
    std::error_code ec;
    fs::create_directories(directory, ec);
    const fs::path canonical = fs::weakly_canonical(fs::path(directory), ec);
    const std::string key = (canonical / "v5").string();
    auto &registry = GetRegistry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto found = registry.stores.find(key);
    if (found != registry.stores.end()) if (auto live = found->second.lock()) return live;
    auto impl = std::make_shared<Impl>();
    impl->base = canonical; impl->root = canonical / "v5"; impl->quota = quotaBytes;
    if (const char *text = std::getenv("FASTLLM_PREFIX_V5_READ_THREADS")) {
        const int parsed = std::atoi(text); if (parsed > 0 && parsed <= 64) impl->readThreads = parsed;
    }
    if (const char *text = std::getenv("FASTLLM_PREFIX_V5_TRIM")) impl->trimEnabled = std::string(text) != "0";
    if (const char *text = std::getenv("FASTLLM_PREFIX_V5_INTERVAL_TOKENS")) {
        const int parsed = std::atoi(text); if (parsed > 0) impl->intervalTokens = parsed;
    }
    if (const char *text = std::getenv("FASTLLM_PREFIX_V5_DRAIN_MS")) {
        const long long parsed = std::atoll(text); if (parsed > 0) impl->drainMs = uint64_t(parsed);
    }
    if (quotaBytes == 0 || !impl->Open(error)) { if (error.empty()) error = "v5_invalid_quota"; return nullptr; }
    std::shared_ptr<DiskPrefixV5Store> store(new DiskPrefixV5Store(impl));
    registry.stores[key] = store;
    return store;
}

std::string DiskPrefixV5Store::Root() const { return impl->root.string(); }
std::string DiskPrefixV5Store::Settings() const {
    return "quota_bytes=" + std::to_string(impl->quota) + " trim=" + (impl->trimEnabled ? "on" : "off") +
           " read_threads=" + std::to_string(impl->readThreads) + " drain_ms=" + std::to_string(impl->drainMs) +
           " interval_tokens=" + std::to_string(impl->intervalTokens);
}

std::shared_ptr<DiskPrefixV5Store::Writer> DiskPrefixV5Store::BeginWriter() {
    std::lock_guard<std::mutex> lock(impl->mu);
    return std::shared_ptr<Writer>(new Writer(impl, impl->nextWriter++));
}

bool DiskPrefixV5Store::Submit(const std::shared_ptr<Writer> &writer, const std::string &key,
                               const std::string &identity, const std::string &family, Json state,
                               Staged &&staged, std::function<void(bool)> completion, std::string &error) {
    error.clear();
    if (!writer || !IsDigest(key) || !IsDigest(identity) || !state.is_object()) { error = "invalid_checkpoint_commit"; return false; }
    auto job = std::make_shared<Impl::Job>();
    job->writer = writer; job->key = key; job->identity = identity; job->family = family;
    job->value = std::move(state); job->staged = std::move(staged); job->completion = std::move(completion);
    for (const auto &item : job->staged) job->bytes += item.second.size();
    job->submitNs = NowNs();
    {
        std::lock_guard<std::mutex> lock(impl->mu);
        if (impl->stopping || impl->dead) { error = "v4_actor_stopping"; return false; }
        if (job->bytes > impl->quota) {
            // C5: a save larger than the quota is rejected before any eviction.
            ++impl->counters.skippedTooLarge;
            error = "v5_save_larger_than_quota";
            return false;
        }
        impl->inflight += job->bytes;
        impl->queue.push_back(job);
    }
    impl->cv.notify_all();
    return true;
}

bool DiskPrefixV5Store::Drain(std::string &error) {
    error.clear();
    const bool ok = impl->DrainBounded(error);
    std::fprintf(stderr, "[Prefix SSD V5] drain-profile: ok=%d %s\n", ok ? 1 : 0, Profile().dump_compact().c_str());
    return ok;
}

bool DiskPrefixV5Store::CancelOldestPending(uint64_t *freedBytes, std::string &error) {
    std::shared_ptr<Impl::Job> job;
    {
        std::lock_guard<std::mutex> lock(impl->mu);
        if (impl->queue.empty()) { error = "v4_no_cancellable_pending"; if (freedBytes) *freedBytes = 0; return false; }
        job = impl->queue.front(); impl->queue.pop_front();
        impl->inflight -= std::min(impl->inflight, job->bytes);
        ++impl->counters.cancelled;
    }
    if (freedBytes) *freedBytes = job->bytes;
    if (job->completion) { try { job->completion(false); } catch (...) {} }
    impl->idle.notify_all();
    return true;
}

uint64_t DiskPrefixV5Store::InFlightBytes() const {
    std::lock_guard<std::mutex> lock(impl->mu);
    return impl->inflight;
}

bool DiskPrefixV5Store::Load(const std::string &key, const std::string &identity, bool manifestOnly, Json &value,
                             std::map<std::string, std::string> *objects, std::shared_ptr<void> &pin,
                             std::string &error, const ReadProgress &progress) {
    const uint64_t start = NowNs();
    error.clear();
    struct Piece { int fileIndex; uint64_t offset, bytes; uint32_t crc; std::string digest; std::string *out; };
    std::vector<Piece> pieces;
    std::vector<fs::path> paths;
    std::shared_ptr<Impl::Ckpt> c;
    std::shared_ptr<void> held;   // released only after mu is dropped (its deleter takes mu)
    {
        std::lock_guard<std::mutex> lock(impl->mu);
        c = impl->FindCkpt(identity, key);
        if (!c) { error = "v4_root_missing"; return false; }
        Json::array kv; impl->AssembleKv(c->file, c->kvCount, kv);
        Json::object fields = c->value.object_items();
        fields["kv"] = Json(kv);
        value = Json(fields);
        // A checkpoint whose slot a writer is overwriting is unavailable (miss), checked before any pin.
        if (c->slot >= 0 && c->file->slotWriting[c->slot]) { error = "v5_slot_being_overwritten"; return false; }
        // Pin the chain and the slot (sec. 5.3); released when the read session ends.
        std::vector<Impl::File *> chain;
        for (Impl::File *f = c->file; f; f = f->parent) { f->pins++; chain.push_back(f); }
        const auto extents = impl->PinExtents(*c, value);
        for (const auto &e : extents) e.first->pinExtents.insert(e.second);
        const int slot = c->slot;
        if (slot >= 0) c->file->slotPins[slot]++;
        std::weak_ptr<Impl> weak = impl;
        Impl::File *home = c->file;
        held = std::shared_ptr<void>(nullptr, [weak, chain, extents, slot, home](void *) {
            auto owner = weak.lock();
            if (!owner) return;
            std::lock_guard<std::mutex> guard(owner->mu);
            for (auto *f : chain) f->pins--;
            for (const auto &e : extents) {
                auto it = e.first->pinExtents.find(e.second);
                if (it != e.first->pinExtents.end()) e.first->pinExtents.erase(it);
            }
            if (slot >= 0) home->slotPins[slot]--;
        });
        impl->Touch(c->file);
        if (manifestOnly) ++impl->counters.manifests;
        else {
            std::vector<std::pair<std::string, uint64_t>> refs; CollectRefs(value, refs);
            std::map<const Impl::File *, int> fileIndex;
            for (const auto &r : refs) {
                if (objects->count(r.first)) continue;
                const Impl::File *owner = nullptr;
                const BlobLoc *loc = impl->FindBlobFor(c->file, c->slot, r.first, r.second, &owner);
                if (!loc) { error = "missing_cache_record"; break; }
                auto fi = fileIndex.find(owner);
                int index;
                if (fi == fileIndex.end()) { index = int(paths.size()); fileIndex[owner] = index; paths.push_back(owner->path); }
                else index = fi->second;
                // Only the map node is created under mu; the 1+ GiB of buffers is allocated by
                // the read workers in parallel (a serial resize of 1.1 GiB measured 485 ms).
                std::string &slotOut = (*objects)[r.first];
                pieces.push_back({index, loc->offset, loc->bytes, loc->crc, r.first, &slotOut});
            }
        }
    }
    auto fail = [&](const std::string &why) {
        error = why;
        for (const auto &p : pieces) objects->erase(p.digest);   // never partial use (sec. 5.4)
        held.reset();
        return false;
    };
    if (!error.empty()) return fail(error);
    if (manifestOnly) { pin = std::move(held); return true; }
    std::vector<Fd> fds;
    for (const auto &p : paths) {
        fds.emplace_back(open(p.c_str(), O_RDONLY | O_CLOEXEC));
        if (fds.back().fd < 0) return fail("cache_short_read");
        posix_fadvise(fds.back().fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
    std::sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) {
        return std::tie(a.fileIndex, a.offset) < std::tie(b.fileIndex, b.offset);
    });
    uint64_t total = 0; for (const auto &p : pieces) total += p.bytes;
    const int threads = std::max(1, std::min<int>(impl->readThreads, int(pieces.size())));
    // Contiguous ranges of roughly equal bytes per thread.
    std::vector<size_t> bounds{0};
    { uint64_t acc = 0; const uint64_t share = total / uint64_t(threads) + 1;
      for (size_t i = 0; i < pieces.size(); ++i) { acc += pieces[i].bytes; if (acc >= share * bounds.size() && bounds.size() < size_t(threads)) bounds.push_back(i + 1); }
      if (bounds.back() != pieces.size()) bounds.push_back(pieces.size()); }
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> crcNs{0}, verified{0};
    std::atomic<int> running{0};
    std::mutex errorMutex; std::string firstError;
    auto setError = [&](const char *why) {
        std::lock_guard<std::mutex> g(errorMutex); if (firstError.empty()) firstError = why;
        failed = true;
    };
    // Workers never throw: pread/crc only; buffers were sized before they start.
    auto work = [&](size_t from, size_t to) {
        for (size_t i = from; i < to && !failed.load(); ++i) {
            const Piece &p = pieces[i];
            p.out->resize(size_t(p.bytes));
            if (!PreadAll(fds[size_t(p.fileIndex)].fd, &(*p.out)[0], size_t(p.bytes), p.offset)) { setError("cache_short_read"); break; }
            const uint64_t t = NowNs();
            const bool good = Crc32c(p.out->data(), p.out->size()) == p.crc;
            crcNs += NowNs() - t;
            if (!good) { setError("cache_object_checksum"); break; }
            verified += p.bytes;
        }
        running--;
    };
    std::vector<std::thread> pool;
    running = int(bounds.size()) - 1;
    for (size_t b = 0; b + 1 < bounds.size(); ++b) pool.emplace_back(work, bounds[b], bounds[b + 1]);
    // The caller's progress/cancel observer runs on this thread (V4 ReadProgress contract):
    // returning false or throwing cancels at a blob boundary.
    uint64_t reported = 0;
    bool first = true;
    while (running.load() > 0 || first) {
        if (progress && !failed.load()) {
            const uint64_t now = verified.load();
            bool keep = true;
            try { keep = progress(now - reported, first ? total : 0); } catch (...) { keep = false; }
            reported = now;
            if (!keep) setError("ssd_read_cancelled");
        }
        first = false;
        if (running.load() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    for (auto &t : pool) t.join();
    if (progress && !failed.load() && verified.load() > reported) {
        try { if (!progress(verified.load() - reported, 0)) setError("ssd_read_cancelled"); } catch (...) { setError("ssd_read_cancelled"); }
    }
    if (failed) return fail(firstError);
    pin = std::move(held);
    const uint64_t ns = NowNs() - start;
    std::lock_guard<std::mutex> lock(impl->mu);
    ++impl->counters.loads; impl->counters.loadBytes += total; impl->counters.loadNs += ns;
    impl->counters.readCrcNs += crcNs.load();
    std::fprintf(stderr, "[Prefix SSD V5] load: key=%s tokens=%d bytes=%llu files=%zu threads=%d ms=%.3f MiB_s=%.1f.\n",
                 key.c_str(), c->length, (unsigned long long)total, paths.size(), threads, Ms(ns),
                 ns ? double(total) / 1048576.0 / (double(ns) / 1e9) : 0.0);
    return true;
}

std::vector<int> DiskPrefixV5Store::CheckpointLengths(const std::string &family, int maximum, const std::string &kind) const {
    std::set<int, std::greater<int>> lengths;
    std::lock_guard<std::mutex> lock(impl->mu);
    for (const auto &entry : impl->ckpts) {
        const auto &c = entry.second;
        if (!c->retired && c->family == family && c->kind == kind && c->length <= maximum) lengths.insert(c->length);
    }
    return std::vector<int>(lengths.begin(), lengths.end());
}

namespace {
DiskPrefixV5Store::Row ToRow(const DiskPrefixV5Store::Impl::Ckpt &c) {
    DiskPrefixV5Store::Row row;
    row.key = c.key; row.identity = c.identity; row.familyIdentity = c.family; row.kind = c.kind;
    row.kvDtype = c.kvDtype; row.provenance = c.provenance; row.metadata = c.metadata;
    row.length = c.length; row.createdNs = c.createdNs;
    return row;
}
void SortRows(std::vector<DiskPrefixV5Store::Row> &rows) {
    std::sort(rows.begin(), rows.end(), [](const DiskPrefixV5Store::Row &a, const DiskPrefixV5Store::Row &b) {
        return a.length != b.length ? a.length > b.length : a.createdNs < b.createdNs;
    });
}
}

std::vector<DiskPrefixV5Store::Row> DiskPrefixV5Store::ListCheckpoints(const std::string &exact, const std::string &family,
        int maximum, const std::string &kind, const std::string &prefix) const {
    std::vector<Row> rows;
    std::lock_guard<std::mutex> lock(impl->mu);
    for (const auto &entry : impl->ckpts) {
        const auto &c = *entry.second;
        if (c.retired || c.kind != kind || c.length > maximum) continue;
        if (!exact.empty() && c.identity != exact) continue;
        if (!family.empty() && c.family != family) continue;
        if (!prefix.empty() && c.prefix != prefix) continue;
        rows.push_back(ToRow(c));
    }
    SortRows(rows);
    return rows;
}

std::vector<DiskPrefixV5Store::Row> DiskPrefixV5Store::MatchPrefixes(const std::string &family,
        const std::vector<std::pair<int, std::string>> &prefixes, const std::string &kind) const {
    std::set<std::pair<int, std::string>> wanted(prefixes.begin(), prefixes.end());
    std::vector<Row> rows;
    std::lock_guard<std::mutex> lock(impl->mu);
    for (const auto &entry : impl->ckpts) {
        const auto &c = *entry.second;
        if (c.retired || c.kind != kind || c.family != family) continue;
        if (!wanted.count({c.length, c.prefix})) continue;
        rows.push_back(ToRow(c));
    }
    SortRows(rows);
    return rows;
}

int DiskPrefixV5Store::JunctionPages(const std::string &identity, const std::vector<std::string> &pages) const {
    std::lock_guard<std::mutex> lock(impl->mu);
    auto found = impl->pageRefs.find(identity);
    if (found == impl->pageRefs.end()) return 0;
    const auto &refs = found->second;
    int lo = 0, hi = int(pages.size());           // pages [0, lo) match; answer in [lo, hi]
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (refs.count(pages[size_t(mid - 1)])) lo = mid; else hi = mid - 1;
    }
    return lo;
}

bool DiskPrefixV5Store::PutMetadata(const std::string &key, const std::string &value, std::string &error) {
    const fs::path dir = impl->root / "meta";
    const fs::path path = dir / (Sha256Hex(key) + ".m");
    const fs::path temp = dir / (Sha256Hex(key) + ".tmp");
    const std::string text = key + "\n" + value;
    // C5: metadata obeys the root quota. Reserve the bytes first (evicting LRU
    // files like a save); if nothing can be evicted, reject without writing.
    const uint64_t replaced = PhysicalOf(path);
    const uint64_t need = RoundUp(text.size()) * 2;   // temp copy + final file can coexist before rename
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(impl->mu);
            if (impl->usedBytes + need <= impl->quota) { impl->usedBytes += need; break; }
        }
        std::string reason;
        if (!impl->EvictOne({}, reason)) { error = "v5_metadata_quota"; return false; }
    }
    auto unreserve = [&]() { std::lock_guard<std::mutex> lock(impl->mu); impl->usedBytes -= std::min(impl->usedBytes, need); };
    Fd fd(open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (fd.fd < 0 || write(fd.fd, text.data(), text.size()) != ssize_t(text.size()) || fdatasync(fd.fd) != 0) {
        error = "v4_metadata_failed"; unlink(temp.c_str()); unreserve(); return false;
    }
    if (rename(temp.c_str(), path.c_str()) != 0 || !SyncDir(dir)) { error = "v4_metadata_failed"; unlink(temp.c_str()); unreserve(); return false; }
    const uint64_t after = PhysicalOf(path);
    std::lock_guard<std::mutex> lock(impl->mu);
    impl->usedBytes -= std::min(impl->usedBytes, need + replaced);
    impl->usedBytes += after;
    impl->otherBytes = impl->otherBytes + after - std::min(impl->otherBytes, replaced);
    return true;
}

bool DiskPrefixV5Store::GetMetadata(const std::string &key, std::string &value, std::string &error) const {
    const fs::path path = impl->root / "meta" / (Sha256Hex(key) + ".m");
    Fd fd(open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.fd < 0) { error = "v4_metadata_missing"; return false; }
    std::string text; char buffer[65536];
    for (;;) { const ssize_t r = read(fd.fd, buffer, sizeof(buffer)); if (r < 0 && errno == EINTR) continue; if (r <= 0) break; text.append(buffer, size_t(r)); }
    const size_t newline = text.find('\n');
    if (newline == std::string::npos || text.compare(0, newline, key) != 0) { error = "cache_metadata_descriptor"; return false; }
    value = text.substr(newline + 1);
    return true;
}

bool DiskPrefixV5Store::RetireCorruptCheckpoint(const std::string &key, const std::string &identity,
                                                const std::string &reason, std::string &error) {
    std::shared_ptr<Impl::Ckpt> c;
    {
        std::lock_guard<std::mutex> lock(impl->mu);
        c = impl->FindCkpt(identity, key);
        if (!c) { error = "v4_root_missing"; return false; }
        if (impl->corruptPredicate && !impl->corruptPredicate(key, identity)) { error = "v4_checkpoint_busy"; return false; }
        c->retired = true;
        impl->UnindexCkpt(c, false);
        ++impl->counters.retired;
    }
    if (impl->eraseObserver) { try { impl->eraseObserver(key, identity); } catch (...) {} }
    std::fprintf(stderr, "[Prefix SSD V5] retire: key=%s reason=%s.\n", key.c_str(), reason.c_str());
    return true;
}

void DiskPrefixV5Store::SetEraseObserver(EraseObserver observer) { std::lock_guard<std::mutex> l(impl->mu); impl->eraseObserver = std::move(observer); }
void DiskPrefixV5Store::SetReclaimPredicate(Predicate predicate) { std::lock_guard<std::mutex> l(impl->mu); impl->reclaimPredicate = std::move(predicate); }
void DiskPrefixV5Store::SetCorruptRetirePredicate(Predicate predicate) { std::lock_guard<std::mutex> l(impl->mu); impl->corruptPredicate = std::move(predicate); }

uint64_t DiskPrefixV5Store::LiveCheckpoints() const { std::lock_guard<std::mutex> l(impl->mu); return impl->ckpts.size(); }
uint64_t DiskPrefixV5Store::PhysicalBytes() const { std::lock_guard<std::mutex> l(impl->mu); return impl->usedBytes; }

Json DiskPrefixV5Store::Capacity() const {
    std::lock_guard<std::mutex> lock(impl->mu);
    return Json::object{{"limit_bytes", std::to_string(impl->quota)}, {"physical_bytes", std::to_string(impl->usedBytes)},
                        {"other_bytes", std::to_string(impl->otherBytes)}, {"inflight_bytes", std::to_string(impl->inflight)},
                        {"files", double(impl->files.size())}, {"live_checkpoints", std::to_string(impl->ckpts.size())},
                        {"phase", impl->usedBytes > impl->quota ? "capacity_unavailable" : "ready"},
                        {"policy", "lru-chain-touch"}, {"storage_format", "5"}};
}

Json DiskPrefixV5Store::Profile() const {
    std::lock_guard<std::mutex> lock(impl->mu);
    const auto &k = impl->counters;
    return Json::object{
        {"storage_format", 5}, {"commits", double(k.commits)}, {"commit_bytes", double(k.commitBytes)},
        {"dedup", double(k.dedup)}, {"branches", double(k.branches)}, {"new_files", double(k.newFiles)},
        {"appends", double(k.appends)}, {"slot_writes", double(k.slotWrites)}, {"slot_fallbacks", double(k.slotFallbacks)},
        {"skipped_quota", double(k.skippedQuota)}, {"skipped_too_large", double(k.skippedTooLarge)},
        {"failures", double(k.failures)}, {"cancelled", double(k.cancelled)}, {"abandoned", double(k.abandoned)},
        {"parent_missing", double(k.parentMissing)}, {"deletes", double(k.deletes)},
        {"delete_ms", Ms(k.deleteNs)}, {"delete_max_ms", Ms(k.deleteMaxNs)}, {"write_ms", Ms(k.writeNs)},
        {"trims", double(k.trims)}, {"trim_bytes", double(k.trimBytes)}, {"trim_ms", Ms(k.trimNs)},
        {"trim_failures", double(k.trimFailures)}, {"trim_enabled", impl->trimEnabled},
        {"sync_ms", Ms(k.syncNs)}, {"syncs", double(k.syncs)}, {"dir_syncs", double(k.dirSyncs)},
        {"durable_ms", Ms(k.durableNs)}, {"durable_max_ms", Ms(k.durableMaxNs)}, {"crc_write_ms", Ms(k.crcNs)},
        {"loads", double(k.loads)}, {"load_bytes", double(k.loadBytes)}, {"load_ms", Ms(k.loadNs)},
        {"crc_read_ms", Ms(k.readCrcNs)}, {"manifests", double(k.manifests)}, {"retired", double(k.retired)},
        {"startup_files", double(k.startupFiles)}, {"startup_dropped", double(k.startupDropped)},
        {"startup_truncated", double(k.startupTruncated)}, {"startup_ms", Ms(k.startupNs)},
        {"used_bytes", double(impl->usedBytes)}, {"inflight_bytes", double(impl->inflight)},
        {"files", double(impl->files.size())}, {"checkpoints", double(impl->ckpts.size())}};
}

bool DiskPrefixV5Store::SelfCheck(std::string &details) const {
    // Headers + chunk headers + every blob crc of every live checkpoint.
    std::vector<std::pair<std::string, std::string>> keys;
    {
        std::lock_guard<std::mutex> lock(impl->mu);
        for (const auto &entry : impl->ckpts) keys.push_back(entry.first);
    }
    size_t good = 0, bad = 0;
    for (const auto &k : keys) {
        std::map<std::string, std::string> objects; Json value; std::shared_ptr<void> pin; std::string error;
        if (const_cast<DiskPrefixV5Store *>(this)->Load(k.second, k.first, false, value, &objects, pin, error)) ++good;
        else if (error != "v4_root_missing") { ++bad; details += k.second + ":" + error + "\n"; }
    }
    std::fprintf(stderr, "[Prefix SSD V5] SELFCHECK: checkpoints=%zu good=%zu bad=%zu.\n", keys.size(), good, bad);
    return bad == 0;
}

void DiskPrefixV5Store::SetFailpointForTest(const std::string &step, const std::string &mode) {
    std::lock_guard<std::mutex> lock(g_failMutex);
    g_failStep = step; g_failMode = mode;
}

} // namespace fastllm
