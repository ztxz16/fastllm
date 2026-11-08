#include "utils/disk_prefix_cache.h"
#include "utils/prefix_cache_logging.h"
#ifdef FASTLLM_DISK_PREFIX_CACHE
#include "utils/disk_prefix_format_v4.h"
#include "utils/disk_prefix_v5.h"
#include <openssl/evp.h>
#include "../../third_party/flexkv_ssd/transfer.h"
#include <deque>
#include <future>
#include <functional>
#include <thread>
#include <condition_variable>
#include <cstdio>
#include <tuple>
#include <cstdlib>
#include <sstream>
#include <signal.h>
#include <sqlite3.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastllm {
namespace {
using Json = DiskPrefixCache::Json;
namespace fs = std::filesystem;

// FASTLLM_PREFIX_V4_FAST_ADMISSION=1 (default off): see Impl::ComputeLiveBytesRows.
bool V4FastAdmission() {
    static const bool enabled = [] {
        const char *value = std::getenv("FASTLLM_PREFIX_V4_FAST_ADMISSION");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

// DESIGN-v3-segments (default off): read at every Open so one process can test both layouts.
bool V4SegmentsEnv() {
    const char *value = std::getenv("FASTLLM_PREFIX_V4_SEGMENTS");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}
uint64_t V4SegmentBytesEnv() {
    const char *value = std::getenv("FASTLLM_PREFIX_V4_SEGMENT_BYTES");
    uint64_t bytes = 1ull << 30;
    if (value != nullptr && value[0] != '\0') {
        try { bytes = std::stoull(value); } catch (...) { bytes = 1ull << 30; }
    }
    return std::max<uint64_t>(bytes, 4096);
}

// ===== [store-profile] 写盘路径诊断计数（只计数/计时，不改任何数据路径）=====
// 目的：验证"同一批字节在写盘路径被 SHA256 算了 3 遍"（StoreBytes 逐对象 /
// Flush 整条记录 / IndexPacked 逐对象校验），并量化分配、拷贝、DB、JSON、Flush。
// 计数点为固定几处调用点，不用全局 Digest，避免把读路径的哈希混进来。
struct StoreProfiler {
    std::atomic<uint64_t> storeDigestCalls{0}, storeDigestBytes{0};
    std::atomic<uint64_t> recordDigestCalls{0}, recordDigestBytes{0};
    std::atomic<uint64_t> indexDigestCalls{0}, indexDigestBytes{0};
    std::atomic<uint64_t> indexInserts{0};
    std::atomic<uint64_t> chunks{0}, flushes{0};
    std::atomic<uint64_t> dbOpens{0}, flushDbOpens{0}, findCalls{0}, findHits{0};
    std::atomic<uint64_t> readbacks{0}, readbackBytes{0};
    std::atomic<uint64_t> allocNs{0}, copyNs{0}, hashNs{0}, dbNs{0}, jsonNs{0}, flushNs{0};
    void Reset() {
        storeDigestCalls=0; storeDigestBytes=0; recordDigestCalls=0; recordDigestBytes=0;
        indexDigestCalls=0; indexDigestBytes=0; indexInserts=0; chunks=0; flushes=0;
        dbOpens=0; flushDbOpens=0; findCalls=0; findHits=0; readbacks=0; readbackBytes=0;
        allocNs=0; copyNs=0; hashNs=0; dbNs=0; jsonNs=0; flushNs=0;
    }
};
StoreProfiler g_storeProfiler;
inline uint64_t ProfNs(const std::chrono::steady_clock::time_point &a,
                       const std::chrono::steady_clock::time_point &b) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();
}
// [S1] 写盘严格校验开关：默认关闭"逐对象二次哈希"（对象内容在读回时仍逐个校验），
// 设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY=1 可恢复写盘时的逐对象重算校验。
inline bool StrictVerifyEnabled() {
    // 运行时读取（不再缓存）：既便于 CPU 测试验证开关两侧行为，也允许运行中调整。
    const char *value = std::getenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
    return value != nullptr && value[0] != 0 && value[0] != '0';
}
// [S1] 写侧复算开关：写侧逐对象重算已接入并行（能力不减、串行段摊开），
// 因此**默认打开**；显式设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY=0 可关闭。
// 注意与上面的 StrictVerifyEnabled()（v3 读路径的"整条记录严格校验"）**分开**：读路径默认保持关闭，
// 因为它会把整条 record 重新读一遍并校验（实测 13.4× 读放大），与写侧复算是两回事。
inline bool StrictWriteVerifyEnabled() {
    const char *value = std::getenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
    if (value == nullptr || value[0] == 0) return true;
    return value[0] != '0';
}
// ===== [store-profile] 结束 =====

constexpr size_t Alignment = 4096;
constexpr size_t ContainerHeaderBytes = Alignment;
constexpr size_t RecordHeaderBytes = 128;
constexpr size_t BatchBytes = DiskPrefixCache::BatchBytes;
constexpr size_t PendingBytes = 16 << 20;
constexpr uint64_t CatalogHeadroom = 1 << 20;
constexpr const char *FormatText = "fastllm-prefix-v3 packed-log-1\n";
constexpr uint64_t PackedRecord = 1, CommitRecord = 2, EraseRecord = 3, MetadataRecord = 4;
constexpr size_t MaxMetadataBytes = DiskPrefixCache::MaxMetadataBytes;
const char ContainerMagic[8] = {'F','L','K','V','3','P','0','1'};
const char RecordMagic[8] = {'F','L','K','V','3','L','0','1'};
using Refs = std::map<std::string, uint64_t>;

void Check(bool condition, const char *message) {
    DiskPrefixCache::Require(condition, message);
}

// json11's local dumper pretty-prints every KV segment. At long contexts its
// whitespace alone can exhaust the metadata bound. Keep the same JSON values,
// escaping and numerical representation; existing records remain readable.
std::string MetadataText(const Json &value) {
    // 直接用紧凑输出：与旧的 dump()+去空白**逐字节一致**（由 JSON 单测逐例核对），
    // 省掉"先生成带缩进的整段文本、再整段扫一遍去空白"的第二遍。
    std::string text=value.dump_compact();
    Check(text.size()<=MaxMetadataBytes,"manifest_too_large");
    return text;
}

uint64_t ThreadCpuNs() {
    struct timespec value {};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return 0;
    return uint64_t(value.tv_sec) * 1000000000ull + uint64_t(value.tv_nsec);
}
uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string ProcessStamp(int pid) {
    std::ifstream stream("/proc/"+std::to_string(pid)+"/stat");
    std::string line; std::getline(stream,line); const auto end=line.rfind(')');
    if (end==std::string::npos) return "";
    std::istringstream fields(line.substr(end+1)); std::string value;
    for (int field=3;field<=22;++field) if (!(fields>>value)) return "";
    return value;
}
std::string Unique() {
    static std::atomic<uint64_t> sequence{0};
    return std::to_string(getpid()) + "-" + std::to_string(NowNs()) + "-" +
           std::to_string(sequence.fetch_add(1));
}

uint64_t AlignUp(uint64_t value) {
    Check(value <= UINT64_MAX - (Alignment - 1), "cache_size_overflow");
    return (value + Alignment - 1) / Alignment * Alignment;
}

void PutU64(char *destination, uint64_t value) {
    for (int i = 0; i < 8; ++i) destination[i] = char(value >> (i * 8));
}

uint64_t GetU64(const char *source) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(uint8_t(source[i])) << (i * 8);
    return value;
}

void PrivateDirectory(const fs::path &path) {
    std::error_code error;
    fs::create_directories(path, error);
    Check(!error && !fs::is_symlink(path) && fs::is_directory(path), "cache_directory");
    Check(chmod(path.c_str(), 0700) == 0, "cache_directory_mode");
}

void SyncDirectory(const fs::path &path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    Check(fd >= 0, "cache_directory_open");
    int result = fsync(fd);
    int saved = errno;
    close(fd);
    errno = saved;
    Check(result == 0, "cache_directory_fsync");
}

uint64_t FileSize(const fs::path &path) {
    struct stat status{};
    if (lstat(path.c_str(), &status) != 0) {
        Check(errno == ENOENT, "cache_file_stat");
        return 0;
    }
    Check(S_ISREG(status.st_mode) && status.st_size >= 0, "cache_file_type");
    return uint64_t(status.st_size);
}

uint64_t AllocatedBytes(const fs::path &path) {
    struct stat status{};
    if (lstat(path.c_str(),&status)!=0) { Check(errno==ENOENT,"cache_file_stat"); return 0; }
    Check(S_ISREG(status.st_mode) && status.st_blocks>=0,"cache_file_type");
    return uint64_t(status.st_blocks)*512;
}
void ReadExact(int fd, void *destination, size_t bytes, uint64_t offset = 0) {
    Check(offset <= uint64_t(INT64_MAX) && bytes <= uint64_t(INT64_MAX)-offset,
          "cache_offset_overflow");
    Check(fastllm_flexkv_ssd::transfer_full(fd,destination,bytes,off_t(offset),true),
          "cache_short_read");
}
void WriteExact(int fd, const void *source, size_t bytes, uint64_t offset = 0) {
    Check(offset <= uint64_t(INT64_MAX) && bytes <= uint64_t(INT64_MAX)-offset,
          "cache_offset_overflow");
    Check(fastllm_flexkv_ssd::transfer_full(fd,const_cast<void *>(source),bytes,off_t(offset),false),
          "cache_short_write");
}

struct Lease {
    int fd = -1;
    Lease(const fs::path &path, int operation) {
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        Check(fd >= 0, "cache_lock_open");
        int result;
        do { result = flock(fd, operation); } while (result < 0 && errno == EINTR);
        if (result < 0) {
            close(fd);
            fd = -1;
            throw std::runtime_error("cache_busy");
        }
    }
    ~Lease() {
        if (fd >= 0) {
            flock(fd, LOCK_UN);
            close(fd);
        }
    }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
};

// One locked manifest per active writer, rather than a lifecycle lock held for
// an entire long request or one fd per KV object. Compact reads active manifests
// under the short lifecycle exclusion and preserves exactly their hashes.
struct V4ObjectLeaseState {
    fs::path path;
    int fd = -1;
    std::mutex mutex;
    std::set<std::string> hashes;
    explicit V4ObjectLeaseState(const fs::path &root) {
        path = root / (".object-lease-" + Unique());
        fd = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        Check(fd >= 0, "v4_object_lease_open");
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            close(fd); fd = -1;
            throw std::runtime_error("v4_object_lease_lock");
        }
    }
    ~V4ObjectLeaseState() {
        if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    void Add(const std::vector<std::string> &incoming) {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto &hash : incoming) {
            Check(DiskPrefixCache::IsDigest(hash), "v4_object_lease_digest");
            if (!hashes.insert(hash).second) continue;
            const std::string line = hash + "\n";
            const off_t end = lseek(fd, 0, SEEK_END);
            Check(end >= 0, "v4_object_lease_seek");
            WriteExact(fd, line.data(), line.size(), uint64_t(end));
        }
    }
};

bool V4CollectObjectLeases(const fs::path &root, std::set<std::string> &keep,
                          std::string &error) {
    for (const auto &entry : fs::directory_iterator(root)) {
        if (entry.path().filename().string().rfind(".object-lease-", 0) != 0) continue;
        const int fd = open(entry.path().c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) { if (errno == ENOENT) continue; error = "v4_object_lease_open"; return false; }
        const int locked = flock(fd, LOCK_EX | LOCK_NB);
        if (locked == 0) { flock(fd, LOCK_UN); close(fd); continue; } // no active owner
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            close(fd); error = "v4_object_lease_lock"; return false;
        }
        struct stat info{};
        if (fstat(fd, &info) != 0 || info.st_size < 0) {
            close(fd); error = "v4_object_lease_stat"; return false;
        }
        // Manifest growth is serialized by the lifecycle lease. Read bounded
        // batches so a long writer never needs a checkpoint-sized allocation.
        char block[65 * 256];
        uint64_t offset = 0;
        while (offset < uint64_t(info.st_size)) {
            const size_t count = size_t(std::min<uint64_t>(sizeof(block), uint64_t(info.st_size)-offset));
            if (count % 65 != 0) { close(fd); error = "v4_object_lease_torn"; return false; }
            ReadExact(fd, block, count, offset);
            for (size_t cursor = 0; cursor < count; cursor += 65) {
                const std::string hash(block + cursor, 64);
                if (block[cursor+64] != '\n' || !DiskPrefixCache::IsDigest(hash)) {
                    close(fd); error = "v4_object_lease_digest"; return false;
                }
                keep.insert(hash);
            }
            offset += count;
        }
        close(fd);
    }
    return true;
}

struct Statement {
    sqlite3_stmt *value = nullptr;
    sqlite3 *db = nullptr;
    // A2/⑨：错误必须带明细与**错误码**，
    // 之前 Row() 对所有非 ROW/DONE 一律抛 "cache_sql_step"（无 rc），导致 BUSY(5) 与
    // LOCKED(6) 无法区分（全仓 grep SQLITE_LOCKED = 0 命中）。现在与 Database::Exec 同口径：
    // "<prefix>:<sqlite 明细> code=<rc>"，BUSY 场景即可被观测/断言。
    static std::string StepError(sqlite3 *database, const char *prefix, int result) {
        const char *detail = database ? sqlite3_errmsg(database) : "no database";
        return std::string(prefix) + ":" + detail + " code=" + std::to_string(result);
    }
    Statement(sqlite3 *database, const char *sql) {
        db = database;
        const int result = sqlite3_prepare_v2(database, sql, -1, &value, nullptr);
        if (result != SQLITE_OK) throw std::runtime_error(StepError(database, "cache_sql_prepare", result));
    }
    ~Statement() { sqlite3_finalize(value); }
    void Text(int index, const std::string &value_) {
        Check(sqlite3_bind_text(value, index, value_.data(), int(value_.size()),
                                SQLITE_TRANSIENT) == SQLITE_OK, "cache_sql_bind");
    }
    void Number(int index, uint64_t value_) {
        Check(value_ <= uint64_t(INT64_MAX), "cache_integer_overflow");
        Check(sqlite3_bind_int64(value, index, sqlite3_int64(value_)) == SQLITE_OK,
              "cache_sql_bind");
    }
    bool Row() {
        const int result = sqlite3_step(value);
        if (result != SQLITE_ROW && result != SQLITE_DONE) {
            throw std::runtime_error(StepError(db, "cache_sql_step", result));
        }
        return result == SQLITE_ROW;
    }
    void Run() { Check(!Row(), "cache_sql_unexpected_row"); }
    std::string Text(int column) const {
        const auto *text = sqlite3_column_text(value, column);
        return text ? std::string(reinterpret_cast<const char *>(text),
                                  sqlite3_column_bytes(value, column)) : "";
    }
    uint64_t Number(int column) const {
        return uint64_t(sqlite3_column_int64(value, column));
    }
};

struct Database {
    sqlite3 *value = nullptr;
    explicit Database(const fs::path &path, bool readOnly = false) {
        int result = sqlite3_open_v2(path.c_str(), &value,
            (readOnly ? SQLITE_OPEN_READONLY : SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE) |
                SQLITE_OPEN_FULLMUTEX, nullptr);
        if (result != SQLITE_OK) {
            if (value) sqlite3_close(value);
            value = nullptr;
        }
        Check(result == SQLITE_OK, "cache_sql_open");
        if (readOnly) {
            sqlite3_busy_timeout(value, 0);
            return; // committed WAL snapshot, no chmod/PRAGMA/writer drain
        }
        // ⑨：busy_timeout 可用 env 覆盖，**默认仍是 10000 ms**（行为不变）；
        // 目的：让"BUSY 必须被诊断且不得永久破坏提交"这条契约能被做成快速、可复现的 CPU 用例
        // （测试把窗口设成 50 ms 并短暂持锁即可触发，无需真等 10 s）。
        // 注意：这里按**每次构造**读取（不用函数局部 static），测试才能 setenv/unsetenv 精确控制。
        const char *busyText = std::getenv("FASTLLM_PREFIX_CACHE_SQLITE_BUSY_TIMEOUT_MS");
        const long busyParsed = busyText != nullptr ? std::atol(busyText) : 0;
        sqlite3_busy_timeout(value, busyParsed > 0 ? int(busyParsed) : 10000);
        sqlite3_limit(value, SQLITE_LIMIT_LENGTH, int(MaxMetadataBytes));
        chmod(path.c_str(), 0600);
        Exec("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL; PRAGMA journal_mode=WAL;");
    }
    ~Database() { Close(); }
    // 显式关闭（可重复调用）：用于"先关临时库再原子替换"这类场景，避免二次关闭。
    void Close() {
        if (value) { sqlite3_close(value); value = nullptr; }
    }
    void Exec(const char *sql) {
        const int result = sqlite3_exec(value, sql, nullptr, nullptr, nullptr);
        if (result != SQLITE_OK) {
            const char *detail = value ? sqlite3_errmsg(value) : "no database";
            throw std::runtime_error(std::string("cache_sql_exec:") + detail +
                                     " code=" + std::to_string(result));
        }
    }
    uint64_t Scalar(const char *sql) {
        Statement query(value, sql);
        return query.Row() ? query.Number(0) : 0;
    }
    std::string ScalarText(const char *sql) {
        Statement query(value, sql);
        return query.Row() ? query.Text(0) : "";
    }
};

struct Transaction {
    Database &database;
    bool done = false;
    explicit Transaction(Database &database_) : database(database_) {
        database.Exec("BEGIN IMMEDIATE");
    }
    void Commit() { database.Exec("COMMIT"); done = true; }
    ~Transaction() {
        if (!done) sqlite3_exec(database.value, "ROLLBACK", nullptr, nullptr, nullptr);
    }
};

uint64_t JsonBytes(const Json &value, uint64_t maximum) {
    Check(value.is_number() && std::isfinite(value.number_value()) &&
          value.number_value() > 0 && value.number_value() <= double(maximum) &&
          std::floor(value.number_value()) == value.number_value(), "invalid_chunk_size");
    return uint64_t(value.number_value());
}

void CollectRefs(const Json &node, Refs &refs, int depth = 0) {
    Check(depth <= 64, "cache_metadata_depth");
    if (node.is_array()) {
        for (const auto &child : node.array_items()) CollectRefs(child, refs, depth + 1);
    } else if (node.is_object()) {
        const auto &object = node.object_items();
        if (object.count("sha256") && object.count("bytes")) {
            const auto &hash = node["sha256"].string_value();
            Check(DiskPrefixCache::IsDigest(hash), "invalid_blob_digest");
            const uint64_t bytes = JsonBytes(node["bytes"], DiskPrefixCache::ChunkBytes);
            auto inserted = refs.emplace(hash, bytes);
            Check(inserted.second || inserted.first->second == bytes, "conflicting_blob_size");
            return;
        }
        for (const auto &entry : object) {
            Check(!((entry.first == "tokens" || entry.first == "token_ids" ||
                     entry.first == "input_ids") && entry.second.is_array()),
                  "raw_tokens_in_commit");
            CollectRefs(entry.second, refs, depth + 1);
        }
    }
}

std::string FieldString(const Json &state, const char *field) {
    const auto &value = state[field];
    return value.is_null() ? "" : value.is_string() ? value.string_value() : value.dump();
}

struct ObjectLocation {
    uint64_t container = 0, offset = 0, bytes = 0, physical = 0, payloadOffset = 0;
};
struct PendingRecord { std::string hash; std::vector<char> payload; };
struct RecordHeader {
    uint64_t kind = 0, bytes = 0, sequence = 0, physical = 0;
    std::string hash;
};
uint32_t HeaderCrc(const char *data, size_t bytes) {
    uint32_t crc = ~uint32_t(0);
    for (size_t i = 0; i < bytes; ++i) {
        crc ^= uint8_t(data[i]);
        for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (0xedb88320U & (0U-(crc&1)));
    }
    return ~crc;
}
uint64_t RecordPhysicalBytes(uint64_t bytes) { return AlignUp(RecordHeaderBytes+bytes); }
std::vector<char> MakeHeader(uint64_t kind, uint64_t sequence, const std::string &hash, uint64_t bytes) {
    std::vector<char> header(RecordHeaderBytes,0);
    memcpy(header.data(),RecordMagic,8);
    PutU64(header.data()+8,kind); PutU64(header.data()+16,bytes);
    PutU64(header.data()+24,sequence); memcpy(header.data()+32,hash.data(),64);
    PutU64(header.data()+96,RecordPhysicalBytes(bytes));
    PutU64(header.data()+104,HeaderCrc(header.data(),104));
    return header;
}
RecordHeader ParseHeader(const char *header) {
    Check(memcmp(header,RecordMagic,8)==0 && GetU64(header+104)==HeaderCrc(header,104),
          "cache_record_header_crc");
    RecordHeader result{GetU64(header+8),GetU64(header+16),GetU64(header+24),GetU64(header+96),
                        std::string(header+32,64)};
    Check(result.kind>=PackedRecord && result.kind<=MetadataRecord && result.bytes>0 &&
          result.bytes<=BatchBytes && result.physical==RecordPhysicalBytes(result.bytes) &&
          DiskPrefixCache::IsDigest(result.hash),"cache_record_header");
    return result;
}
void MakeContainerHeader(char *header, uint64_t id) {
    memset(header,0,ContainerHeaderBytes); memcpy(header,ContainerMagic,8);
    PutU64(header+8,id); PutU64(header+16,NowNs());
    const auto hash=DiskPrefixCache::Digest(header,24); memcpy(header+24,hash.data(),64);
}
void VerifyContainerHeader(const char *header, uint64_t id) {
    Check(memcmp(header,ContainerMagic,8)==0 && GetU64(header+8)==id &&
          std::string(header+24,64)==DiskPrefixCache::Digest(header,24),"cache_container_header");
}
uint64_t Add(uint64_t a,uint64_t b) {
    Check(a<=uint64_t(INT64_MAX) && b<=uint64_t(INT64_MAX)-a,"cache_size_overflow");
    return a+b;
}
std::string ReadText(const fs::path &path) {
    std::ifstream stream(path,std::ios::binary);
    Check(bool(stream),"cache_format_open");
    return std::string(std::istreambuf_iterator<char>(stream),{});
}
bool ConfirmedDataFailure(const std::exception &error) {
    const std::string what=error.what();
    return what=="cache_record_checksum" || what=="cache_object_checksum" ||
        what=="cache_record_header_crc" || what=="cache_record_header" ||
        what=="cache_state_json" || what=="cache_metadata_descriptor" ||
        what=="missing_cache_record" || what=="cache_commit_missing_object" ||
        what=="cache_object_bounds" || what=="cache_object_location" || what=="cache_record_bounds";
}
bool EquivalentPrefix(const Json &left,const Json &right) {
    if (left["kind"].string_value()!="prefix" || right["kind"].string_value()!="prefix" ||
        !DiskPrefixCache::IsDigest(left["prefix"].string_value()) ||
        !DiskPrefixCache::IsDigest(left["family_identity"].string_value()) || !left["layout"].is_object()) return false;
    for (const char *field:{"kind","length","family_identity","kv_dtype","provenance","prefix","layout"})
        if (left[field]!=right[field]) return false;
    return true;
}
uint64_t CatalogBudget(size_t rows) {
    Check(rows<=size_t(INT64_MAX/65536)-16,"cache_catalog_budget_overflow");
    return std::max<uint64_t>(CatalogHeadroom,(uint64_t(rows)+16)*65536);
}

// ---------------------------------------------------------------------------
// Route B（2026-10-02）：block-chained KV layout helpers.
//
// 背景（要修的问题）：每个检查点把"从开头到本点的全部 KV 分块描述"重复序列化成一个
// 大 state 对象（实测 177.75 B/token；256K token 时 >8 MiB 单对象上限 ⇒ 长上下文检查点
// "发布成功但读不回"）。Route B 把 KV 描述按块（默认 256 token）拆成**每条一块**的
// 内容寻址对象，状态对象只留链尾键 + 块数，因此不再有"随上下文增长的单一对象"。
struct RouteBBlock {
    uint64_t index = 0, tokenStart = 0, tokenEnd = 0;
    Json::array kv;
};

// bundle 的 token 区间：所有 segments 必须同 begin/tokens（与 Import 的分组口径一致）。
bool BundleSpan(const Json &bundle, uint64_t &begin, uint64_t &tokens, std::string &error) {
    if (!bundle.is_object()) { error = "v4_block_bundle"; return false; }
    const auto &segments = bundle["segments"].array_items();
    if (segments.empty()) { error = "v4_block_bundle_segments"; return false; }
    const Json &first = segments.front();
    if (!first["begin"].is_number() || !first["tokens"].is_number()) {
        error = "v4_block_bundle_range";
        return false;
    }
    if (first["begin"].int_value() < 0 || first["tokens"].int_value() <= 0 ||
        first["tokens"].int_value() > (1 << 20)) {
        error = "v4_block_bundle_range";
        return false;
    }
    begin = uint64_t(first["begin"].int_value());
    tokens = uint64_t(first["tokens"].int_value());
    for (const auto &segment : segments) {
        if (segment["begin"].int_value() != int(begin) ||
            segment["tokens"].int_value() != int(tokens)) {
            error = "v4_block_mixed_bundle_range";
            return false;
        }
    }
    return true;
}

// Route B 只作用于"真正的 KV bundle 描述"数组（每条含非空 segments）。历史形态里
// kv 也可能是扁平的 {sha256,bytes} 引用数组（例如 hostlimit 夹具把分块直接列在 kv 下）；
// 那种形态必须继续走旧扁平路径，不能被误判成块链而拒绝。
bool RouteBEligible(const Json::array &kv) {
    if (kv.empty()) return false;
    for (const auto &bundle : kv) {
        if (!bundle.is_object() || !bundle["segments"].is_array() ||
            bundle["segments"].array_items().empty()) {
            return false;
        }
    }
    return true;
}

// 把完整 kv 数组切成块。块边界按**绝对 token 网格**（blockTokens）对齐，bundle 按
// begin 归块；因此同一前缀在不同检查点里切出的块逐字节相同（可内容寻址去重）。
// fail-closed：
//   · 区间有洞 → v4_block_gap；有重叠 → v4_block_overlap；
//   · 第一块不是从 token 0 开始（缺少父链）→ v4_block_parent_missing；
//   · bundle 跨块（begin 未对齐/块小于 bundle）→ v4_block_aligned。
bool SplitBlocks(const Json::array &kv, uint32_t blockTokens,
                 std::vector<RouteBBlock> &out, std::string &error) {
    out.clear();
    if (kv.empty() || blockTokens == 0) { error = "v4_block_empty_kv"; return false; }
    std::map<uint64_t, uint64_t> spans;   // begin -> end（严格连续校验）
    for (const auto &bundle : kv) {
        uint64_t begin = 0, tokens = 0;
        if (!BundleSpan(bundle, begin, tokens, error)) return false;
        const uint64_t end = begin + tokens;
        auto inserted = spans.emplace(begin, end);
        if (!inserted.second && inserted.first->second != end) { error = "v4_block_overlap"; return false; }
        if (begin % blockTokens + tokens > blockTokens) { error = "v4_block_aligned"; return false; }
    }
    uint64_t cursor = 0;
    for (const auto &span : spans) {
        if (span.first != cursor) {
            error = (cursor == 0) ? "v4_block_parent_missing" : "v4_block_gap";
            return false;
        }
        cursor = span.second;
    }
    const uint64_t total = cursor;
    std::map<uint64_t, size_t> positionByBlock;
    for (const auto &bundle : kv) {
        uint64_t begin = 0, tokens = 0;
        if (!BundleSpan(bundle, begin, tokens, error)) return false;
        const uint64_t blockIndex = begin / blockTokens;
        auto found = positionByBlock.find(blockIndex);
        if (found == positionByBlock.end()) {
            RouteBBlock block;
            block.index = blockIndex;
            block.tokenStart = blockIndex * uint64_t(blockTokens);
            block.tokenEnd = std::min<uint64_t>(total, block.tokenStart + uint64_t(blockTokens));
            const size_t position = out.size();
            out.push_back(std::move(block));
            positionByBlock.emplace(blockIndex, position);
            found = positionByBlock.find(blockIndex);
        }
        out[found->second].kv.push_back(bundle);
    }
    std::sort(out.begin(), out.end(),
              [](const RouteBBlock &a, const RouteBBlock &b) { return a.index < b.index; });
    return true;
}

// 块内容摘要：只绑该块的 KV 分块描述（不含链字段），保证同一前缀切出的块摘要相同。
std::string BlockContentDigest(const Json::array &kv) {
    return DiskPrefixCache::Digest(Json(kv).dump_compact());
}

struct RouteBChain {
    bool enabled = false;
    std::string tip;
    uint64_t blockCount = 0, chainedTokens = 0, blockTokens = 0;
    std::vector<Json> objects;   // 每条块的完整对象文本（转成 Json 便于插入 objects 列表）
};

// 由完整 kv 数组构建块对象 + 链尾键。父键固定种子起步；块键为
// H(parent, 块内 token 数, 块内容摘要)，与 vLLM/SGLang/LMCache 的链式哈希同构。
bool BuildBlockChain(const Json::array &kv, RouteBChain &out, std::string &error) {
    out = RouteBChain();
    std::vector<RouteBBlock> blocks;
    const uint32_t blockTokens = prefix_v4::BlockTokens();
    if (!SplitBlocks(kv, blockTokens, blocks, error)) return false;
    if (blocks.size() > prefix_v4::kMaxBlockCount) { error = "v4_block_count"; return false; }
    std::string parent = prefix_v4::BlockGenesisKey();
    for (const auto &block : blocks) {
        const std::string content = BlockContentDigest(block.kv);
        const uint64_t tokens = block.tokenEnd - block.tokenStart;
        const std::string chainKey = prefix_v4::BlockChainKey(parent, tokens, content);
        if (!prefix_v4::IsDigest(chainKey) || !prefix_v4::IsDigest(content)) {
            error = "v4_block_digest";
            return false;
        }
        const Json object = Json::object{
            {"domain", "fastllm-block-v1"},
            {"index", double(block.index)},
            {"token_start", double(block.tokenStart)},
            {"token_end", double(block.tokenEnd)},
            {"tokens", double(tokens)},
            {"parent", parent},
            {"chain_key", chainKey},
            {"content", content},
            {"kv", Json(block.kv)},
        };
        out.objects.push_back(object);
        parent = chainKey;
    }
    out.enabled = true;
    out.tip = parent;
    out.blockCount = blocks.size();
    out.chainedTokens = blocks.empty() ? 0 : blocks.back().tokenEnd;
    out.blockTokens = blockTokens;
    return true;
}

// 从块对象里解析一条 catalog blocks 行（重建/提交/自检共用同一口径）。
bool ParseBlockRow(const Json &block, DiskPrefixV4Store::BlockRow &row, std::string &error) {
    if (!block.is_object()) { error = "v4_block_object"; return false; }
    const auto &chainKey = block["chain_key"];
    const auto &parent = block["parent"];
    const auto &content = block["content"];
    if (!chainKey.is_string() || !parent.is_string() || !content.is_string() ||
        !prefix_v4::IsDigest(chainKey.string_value()) ||
        !prefix_v4::IsDigest(parent.string_value()) ||
        !prefix_v4::IsDigest(content.string_value())) {
        error = "v4_block_object";
        return false;
    }
    if (!block["index"].is_number() || !block["token_start"].is_number() ||
        !block["token_end"].is_number() || !block["tokens"].is_number() ||
        !block["kv"].is_array()) {
        error = "v4_block_object";
        return false;
    }
    row.chainKey = chainKey.string_value();
    row.parent = parent.string_value();
    row.index = uint64_t(block["index"].number_value());
    row.tokenStart = uint64_t(block["token_start"].number_value());
    row.tokenEnd = uint64_t(block["token_end"].number_value());
    if (row.tokenEnd < row.tokenStart ||
        uint64_t(block["tokens"].number_value()) != row.tokenEnd - row.tokenStart) {
        error = "v4_block_object";
        return false;
    }
    // 链完整性：块键必须能由 (parent, token 数, 内容摘要) 复算出来。
    if (prefix_v4::BlockChainKey(row.parent, row.tokenEnd - row.tokenStart,
                                 content.string_value()) != row.chainKey) {
        error = "v4_block_chain_mismatch";
        return false;
    }
    if (BlockContentDigest(block["kv"].array_items()) != content.string_value()) {
        error = "v4_block_content_mismatch";
        return false;
    }
    return true;
}

void FaultPoint(const char *name) {
#ifdef FASTLLM_DISK_PREFIX_CACHE_TESTING
    const char *point=std::getenv("FASTLLM_DISK_CACHE_FAILPOINT");
    if (point && std::string(point)==name) _exit(87);
#else
    (void)name;
#endif
}
// D5-b1（2026-10-04）**已撤销**：曾计划在 actor"认领前"加一个按 key 阻塞的测试闸门来验
// 主#6（actor 认领条件＝"取消 或 仍持 staged"）。核对后确认该前提**错误**：认领点之前
// `job->staged` 已被 move 进 `objects` 并清空（见 :3222/:3270 一带），故该处 `preserveOnly`
// 恒为 false，闸门**创造不出**保字节条件；真正的触发路径是"**备菜前被丢弃**"——
// `PreserveBeforeDropLocked`（:2750，调用点 :2798 prune-finished-head / :2829 reap-pending /
// **:2868 enqueue-backpressure**）。因此按"最小化改动"删掉闸门（它反而在 actor 路径上引入
// 一个阻塞点），主#6/#7 的用例改挂**背压出队**场景，断言面用 `Profile()` 的
// `preserved_on_drop` / `unpreserved_drops` / `preserve_flush_failures`（:5599-5602）。
thread_local std::string startupManifest;
} // namespace

DiskPrefixCache::RestoreFailureClass DiskPrefixCache::ClassifyRestoreFailure(const std::string &error) {
    const auto has = [&](const char *code) {
        const std::string value(code);
        return error == value || error.rfind(value + ":", 0) == 0 || error.rfind(value + " ", 0) == 0;
    };
    static const char *const stored[] = {
        "cache_record_header_crc", "cache_record_header", "cache_short_read",
        "cache_record_checksum", "cache_object_checksum", "cache_object_bounds",
        "cache_object_location", "cache_record_bounds", "missing_cache_record",
        "cache_state_json", "cache_metadata_descriptor", "v4_object_mismatch",
        "v4_object_size", "v4_root_object", "v4_block_count", "v4_block_object",
        "v4_block_chain_mismatch", "v4_block_content_mismatch", "v4_block_missing",
        "v4_block_parent_missing", "v4_object_missing"};
    for (const char *code : stored) if (has(code)) return RestoreFailureClass::STORED_REPRESENTATION;
    static const char *const logical[] = {
        "ssd_checkpoint_identity", "ssd_import_identity_mismatch",
        "ssd_import_identity_short_context", "layout", "dtype", "provenance",
        "page_alignment", "state_descriptor", "runtime_layout",
        "prefix_provenance_key", "parent_failed_or_quarantined"};
    for (const char *code : logical) if (has(code)) return RestoreFailureClass::LOGICAL;
    static const char *const operational[] = {
        "ssd_restore_cancelled_or_timeout", "ssd_read_cancelled", "cache_busy",
        "capacity_unavailable", "v4_root_missing", "v4_block_chunk_missing",
        "v4_actor_stopping", "v4_checkpoint_pinned", "v4_corrupt_retire_gate_missing",
        "v4_checkpoint_busy", "v4_control_uncommitted_tail", "EIO"};
    for (const char *code : operational) if (has(code)) return RestoreFailureClass::OPERATIONAL;
    return RestoreFailureClass::UNKNOWN;
}

struct DiskPrefixCache::Impl {
    fs::path root,base,containers,dbPath;
    std::string identity,family;
    uint64_t limit,fileLimit;
    int ioThreads;
    bool v4Mode = false;
    std::shared_ptr<DiskPrefixV4Store> v4;
    // V5 (DESIGN-V5-draft3) is the default store; FASTLLM_PREFIX_V5=0 keeps V4.
    bool v5Mode = false;
    std::shared_ptr<DiskPrefixV5Store> v5;
    std::shared_ptr<void> eraseObserver;
    std::shared_ptr<void> reclaimPredicate;
    std::shared_ptr<void> corruptRetirePredicate;
    std::atomic<bool> healthy{true},stopping{false},needsMaintenance{false};
    std::thread maintenance;
    std::mutex maintenanceMutex;
    std::mutex maintenanceStepMutex;
    std::shared_mutex writerGate;
    std::condition_variable maintenanceChanged;

    Impl(const std::string &directory,const std::string &identity_,uint64_t limit_,
         const std::string &family_,uint64_t fileLimit_,int threads,bool v4Mode_)
        :root(directory),base(root/"v3"),containers(base/"containers"),dbPath(base/"catalog.sqlite3"),
         identity(identity_),family(family_.empty()?identity_:family_),limit(limit_),fileLimit(fileLimit_),
         ioThreads(threads),v4Mode(v4Mode_) {
        if (DiskPrefixV5Enabled()) {
            Check(DiskPrefixCache::IsDigest(identity) && DiskPrefixCache::IsDigest(family) && limit>0,
                  "invalid_cache_config");
            // V5 has its own root <directory>/v5 (draft3 G7); V4 data is left untouched.
            v4Mode = false; v5Mode = true;
            std::string v5Error;
            v5 = DiskPrefixV5Store::OpenShared(directory, limit, v5Error);
            Check(v5 != nullptr, v5Error.empty() ? "v5_store_open" : v5Error.c_str());
            std::fprintf(stderr, "[Prefix SSD V5] enabled: root=%s identity=%s %s.\n", v5->Root().c_str(), identity.c_str(),
                         v5->Settings().c_str());
            return;
        }
        Check(v4Mode, "persistent_cache_v4_required");
        Check(DiskPrefixCache::IsDigest(identity) && DiskPrefixCache::IsDigest(family) &&
              limit>0 && limit<=uint64_t(INT64_MAX) && threads>0 && threads<=64,
              "invalid_cache_config");
        if (!v4Mode) {
            Check(fileLimit>=ContainerHeaderBytes+3*Alignment &&
                  fileLimit<=uint64_t(INT64_MAX) && fileLimit%Alignment==0,
                  "invalid_cache_config");
        }
        if (v4Mode) {
            // v4 mode must never create the legacy layout: no PrivateDirectory,
            // no v3 FORMAT, no Recover and no v3 maintenance thread.
            std::string v4Error;
            // §15：同一个 canonical root 在同一进程里只能有一个 CommitActor。
            // target KV 与 encoder/image 两个 namespace 因此共用同一个 store；
            // 各自再 Open 一次会写出重复序号（实测 …8,9,10,9,10,11…）。
            v4 = DiskPrefixV4Store::OpenShared(directory, limit, v4Error);
            Check(v4 != nullptr, v4Error.empty() ? "v4_store_open" : v4Error.c_str());
            return;
        }
        // Inspect before creating/chmod/cleaning anything in an old v3 root.
        if (fs::exists(base)) {
            if (!fs::exists(base/"FORMAT")) {
                Check(fs::is_empty(base),"cache_legacy_v3_requires_explicit_migration");
            } else Check(!fs::is_symlink(base/"FORMAT") && ReadText(base/"FORMAT")==FormatText,"cache_unsupported_format");
        }
        PrivateDirectory(root);
        {
            Lease writers(root/".writers",LOCK_EX);
            Lease lifecycle(root/".lease",LOCK_EX);
            PrivateDirectory(base); PrivateDirectory(containers);
            if (!fs::exists(base/"FORMAT")) {
                int fd=open((base/"FORMAT").c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);
                Check(fd>=0,"cache_format_create");
                try { WriteExact(fd,FormatText,strlen(FormatText)); Check(fdatasync(fd)==0,"cache_format_sync"); close(fd); }
                catch (...) { close(fd); throw; }
                SyncDirectory(base);
            }
            Recover(false);
            Lease metadata(root/".metadata",LOCK_EX);
            Database db(dbPath);
            const auto old=db.ScalarText("SELECT value FROM meta WHERE key='configured_limit'");
            SetMeta(db,"previous_limit",old); SetMeta(db,"configured_limit",std::to_string(limit));
            SetMeta(db,"configured_file_limit",std::to_string(fileLimit));
            const uint64_t used=Used(db);
            SetMeta(db,"shrink_debt",std::to_string(used>limit?used-limit:0));
            needsMaintenance=used>limit;
            if (used>limit) std::fprintf(stderr,
                "[Prefix SSD] shrinking: previous=%s target=%llu physical_reserved=%llu debt=%llu policy=fifo-new-write-first.\n",
                old.c_str(),(unsigned long long)limit,(unsigned long long)used,(unsigned long long)(used-limit));
        }
        maintenance=std::thread([this] {
            std::unique_lock<std::mutex> lock(maintenanceMutex);
            while (!stopping) {
                maintenanceChanged.wait_for(lock,std::chrono::milliseconds(250));
                if (stopping) break;
                if (!needsMaintenance) continue;
                lock.unlock();
                try { needsMaintenance=!MaintainStep(0); } catch (const std::exception &e) {
                    if (std::string(e.what())!="cache_busy") {
                        std::fprintf(stderr,"[Prefix SSD] maintenance deferred: %s.\n",e.what());
                        needsMaintenance=false;
                    }
                }
                lock.lock();
            }
        });
    }
    ~Impl() {
        stopping=true; maintenanceChanged.notify_all();
        if (maintenance.joinable()) maintenance.join();
    }
    static void SetMeta(Database &db,const std::string &key,const std::string &value) {
        Statement put(db.value,"INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)");
        put.Text(1,key); put.Text(2,value); put.Run();
    }
    fs::path ContainerPath(uint64_t id) const {
        char name[64]; snprintf(name,sizeof(name),"container-%012llu.data",(unsigned long long)id);
        return containers/name;
    }
    std::vector<uint64_t> ContainerIds() const {
        std::vector<uint64_t> ids;
        for (const auto &entry:fs::directory_iterator(containers)) {
            const auto name=entry.path().filename().string();
            if (name.size()<16 || name.compare(0,10,"container-") || name.substr(name.size()-5)!=".data") continue;
            Check(!entry.is_symlink() && entry.is_regular_file(),"cache_container_type");
            try {
                const auto id=std::stoull(name.substr(10,name.size()-15));
                if (ContainerPath(id).filename()==entry.path().filename()) ids.push_back(id);
            } catch (const std::invalid_argument &) {} catch (const std::out_of_range &) {}
        }
        std::sort(ids.begin(),ids.end()); return ids;
    }
    void Schema(Database &db) const {
        db.Exec("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS containers(id INTEGER PRIMARY KEY,watermark INTEGER NOT NULL,sealed INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE IF NOT EXISTS records(container INTEGER NOT NULL,offset INTEGER NOT NULL,physical INTEGER NOT NULL,"
            "kind INTEGER NOT NULL,sequence INTEGER NOT NULL,hash TEXT NOT NULL,PRIMARY KEY(container,offset));"
            "CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY,container INTEGER NOT NULL,offset INTEGER NOT NULL,"
            "payload_offset INTEGER NOT NULL,bytes INTEGER NOT NULL,physical INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS events(identity TEXT NOT NULL,key TEXT NOT NULL,kind INTEGER NOT NULL,sequence INTEGER NOT NULL,"
            "container INTEGER NOT NULL,offset INTEGER NOT NULL,physical INTEGER NOT NULL,body TEXT NOT NULL,credit INTEGER NOT NULL,PRIMARY KEY(identity,key));"
            "CREATE TABLE IF NOT EXISTS event_history(identity TEXT NOT NULL,key TEXT NOT NULL,container INTEGER NOT NULL,offset INTEGER NOT NULL,kind INTEGER NOT NULL,PRIMARY KEY(container,offset));"
            "CREATE INDEX IF NOT EXISTS event_history_key ON event_history(identity,key,kind);"
            "CREATE TABLE IF NOT EXISTS checkpoints(identity TEXT NOT NULL,key TEXT NOT NULL,family TEXT NOT NULL,kind TEXT NOT NULL,"
            "length INTEGER NOT NULL,dtype TEXT NOT NULL,provenance TEXT NOT NULL,created INTEGER NOT NULL,bytes INTEGER NOT NULL,"
            "state TEXT NOT NULL,control_bytes INTEGER NOT NULL,prefix TEXT NOT NULL,PRIMARY KEY(identity,key));"
            "CREATE TABLE IF NOT EXISTS refs(identity TEXT NOT NULL,key TEXT NOT NULL,hash TEXT NOT NULL,PRIMARY KEY(identity,key,hash));"
            "CREATE INDEX IF NOT EXISTS ref_hash ON refs(hash);"
            "CREATE INDEX IF NOT EXISTS checkpoint_lookup ON checkpoints(family,kind,length);"
            "CREATE INDEX IF NOT EXISTS checkpoint_prefix ON checkpoints(family,kind,prefix,length);"
            "CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,state TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY,bytes INTEGER NOT NULL,pid INTEGER NOT NULL,start TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS pins(session TEXT NOT NULL,hash TEXT NOT NULL,PRIMARY KEY(session,hash));"
            "CREATE INDEX IF NOT EXISTS pin_hash ON pins(hash);"
            "PRAGMA user_version=301;");
    }
    bool FindObject(Database &db,const std::string &hash,ObjectLocation &out) const {
        Statement query(db.value,"SELECT container,offset,bytes,physical,payload_offset FROM objects WHERE hash=?");
        query.Text(1,hash); if (!query.Row()) return false;
        out={query.Number(0),query.Number(1),query.Number(2),query.Number(3),query.Number(4)};
        Check(out.offset>=ContainerHeaderBytes && out.offset%Alignment==0 && out.physical>=Alignment &&
              out.physical<=RecordPhysicalBytes(BatchBytes) && out.physical%Alignment==0 &&
              out.bytes>0 && out.bytes<=DiskPrefixCache::ChunkBytes && out.payloadOffset>=RecordHeaderBytes &&
              out.payloadOffset<=out.physical && out.bytes<=out.physical-out.payloadOffset,
              "cache_object_location");
        return true;
    }
    std::vector<char> ReadRecord(uint64_t container,uint64_t offset,uint64_t physical,RecordHeader *parsed=nullptr) const {
        Check(physical>=Alignment && physical<=RecordPhysicalBytes(BatchBytes),"cache_record_bounds");
        int fd=open(ContainerPath(container).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
        Check(fd>=0,"cache_container_open");
        std::vector<char> record(static_cast<size_t>(physical));
        try {
            // FlexKV's residual-iovec transfer handles both interruption and short I/O.
            const size_t half=record.size()/2;
            iovec spans[2]={{record.data(),half},{record.data()+half,record.size()-half}};
            Check(fastllm_flexkv_ssd::transfer_iov_full(fd,spans,2,off_t(offset),record.size(),true),"cache_short_read");
            close(fd);
        } catch (...) { close(fd); throw; }
        auto header=ParseHeader(record.data());
        Check(header.physical==physical && DiskPrefixCache::Digest(record.data()+RecordHeaderBytes,header.bytes)==header.hash,
              "cache_record_checksum");
        if (parsed) *parsed=header;
        return record;
    }
    std::vector<char> ReadObject(Database &db,const std::string &hash,uint64_t bytes) const {
#ifdef FASTLLM_DISK_PREFIX_CACHE_TESTING
        const char *readFault=std::getenv("FASTLLM_DISK_CACHE_READ_ERROR");
        if (readFault && std::string(readFault)=="EIO") throw std::runtime_error("cache_injected_io_error");
#endif
        ObjectLocation location;
        Check(FindObject(db,hash,location) && location.bytes==bytes,"missing_cache_record");
        auto record=ReadRecord(location.container,location.offset,location.physical);
        Check(location.payloadOffset>=RecordHeaderBytes && location.payloadOffset<=record.size() &&
              bytes<=record.size()-location.payloadOffset,"cache_object_bounds");
        const char *data=record.data()+location.payloadOffset;
        Check(DiskPrefixCache::Digest(data,bytes)==hash,"cache_object_checksum");
        return std::vector<char>(data,data+bytes);
    }
    std::string ReadChunks(Database &db,const Json &chunks) const {
        Check(chunks.is_array(),"cache_metadata_descriptor");
        std::string result;
        for (const auto &chunk:chunks.array_items()) {
            const uint64_t bytes=JsonBytes(chunk["bytes"],DiskPrefixCache::ChunkBytes);
            Check(bytes<=MaxMetadataBytes-result.size(),"cache_metadata_bound");
            auto payload=ReadObject(db,chunk["sha256"].string_value(),bytes);
            result.append(payload.data(),payload.size());
        }
        return result;
    }
    Json ParseState(Database &db,const Json &descriptor) const {
        std::string error; const auto text=ReadChunks(db,descriptor);
        auto state=Json::parse(text,error);
        Check(error.empty() && state.is_object(),"cache_state_json"); return state;
    }
    void ValidateReferenceHeaders(Database &db,const Json &state) const {
        Refs refs; CollectRefs(state,refs);
        std::map<std::pair<uint64_t,uint64_t>,ObjectLocation> locations;
        for (const auto &ref:refs) {
            ObjectLocation location;
            Check(FindObject(db,ref.first,location) && location.bytes==ref.second,"missing_cache_record");
            locations[{location.container,location.offset}]=location;
        }
        int fd=-1; uint64_t container=UINT64_MAX;
        try {
            for (const auto &item:locations) {
                const auto &location=item.second;
                if (container!=location.container) {
                    if (fd>=0) close(fd);
                    fd=open(ContainerPath(location.container).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
                    Check(fd>=0,"cache_container_open"); container=location.container;
                }
                char bytes[RecordHeaderBytes]; ReadExact(fd,bytes,sizeof(bytes),location.offset);
                const auto header=ParseHeader(bytes);
                Check(header.kind==PackedRecord && header.physical==location.physical &&
                      location.payloadOffset<=RecordHeaderBytes+header.bytes &&
                      location.bytes<=RecordHeaderBytes+header.bytes-location.payloadOffset,"cache_object_bounds");
            }
            if (fd>=0) close(fd);
        } catch (...) { if (fd>=0) close(fd); throw; }
    }
    Json LoadRecord(Database &db,const std::string &id,const std::string &key) const {
        Statement query(db.value,"SELECT state FROM checkpoints WHERE identity=? AND key=?");
        query.Text(1,id); query.Text(2,key); Check(query.Row(),"missing_checkpoint");
        std::string error; auto desc=Json::parse(query.Text(0),error);
        Check(error.empty(),"cache_checkpoint_descriptor");
        const auto state=ParseState(db,desc);
        ValidateReferenceHeaders(db,state); return state;
    }
    uint64_t PhysicalBytes() const {
        uint64_t result=0;
        for (const auto &entry:fs::directory_iterator(base))
            if (entry.is_regular_file() && !entry.is_symlink()) result=Add(result,AllocatedBytes(entry.path()));
        for (auto id:ContainerIds()) result=Add(result,AllocatedBytes(ContainerPath(id)));
        return result;
    }
    uint64_t Used(Database &db) const {
        return Add(Add(PhysicalBytes(),db.Scalar("SELECT COALESCE(SUM(bytes),0) FROM reservations")),
                   Add(db.Scalar("SELECT COALESCE(SUM(control_bytes),0) FROM checkpoints"),
                       db.Scalar("SELECT COALESCE(SUM(credit),0) FROM events")));
    }
    uint64_t Generation(Database &db) const {
        return Add(std::max(NowNs(),std::max(
            db.Scalar("SELECT COALESCE(MAX(sequence),0) FROM records"),
            db.Scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='sequence'"))),1);
    }
    uint64_t NewContainer(Database &db) const {
        const auto ids=ContainerIds();
        const uint64_t id=ids.empty()?0:Add(ids.back(),1);
        int fd=open(ContainerPath(id).c_str(),O_CREAT|O_EXCL|O_WRONLY|O_CLOEXEC|O_NOFOLLOW,0600);
        Check(fd>=0,"cache_container_create");
        char header[ContainerHeaderBytes]; MakeContainerHeader(header,id);
        try { WriteExact(fd,header,sizeof(header)); Check(fdatasync(fd)==0,"cache_container_sync"); close(fd); }
        catch (...) { close(fd); throw; }
        SyncDirectory(containers);
        Statement put(db.value,"INSERT INTO containers(id,watermark,sealed) VALUES(?,?,0)");
        put.Number(1,id); put.Number(2,ContainerHeaderBytes); put.Run();
        SetMeta(db,"active_container",std::to_string(id)); return id;
    }
    void CheckCurrentConfig(Database &db) const {
        const auto quota=db.ScalarText("SELECT value FROM meta WHERE key='configured_limit'");
        const auto file=db.ScalarText("SELECT value FROM meta WHERE key='configured_file_limit'");
        Check((quota.empty() || quota==std::to_string(limit)) && (file.empty() || file==std::to_string(fileLimit)),
              "cache_config_changed_reopen");
    }
    void ReapDeadWriters(Database &db) const {
        std::vector<std::string> dead;
        {
            Statement query(db.value,"SELECT id,pid,start FROM reservations");
            while (query.Row()) {
                const int pid=int(query.Number(1)); const auto stamp=ProcessStamp(pid);
                const bool gone=stamp.empty() && kill(pid,0)<0 && errno==ESRCH;
                if (gone || (!stamp.empty() && stamp!=query.Text(2))) dead.push_back(query.Text(0));
            }
        }
        if (dead.empty()) return;
        Transaction tx(db);
        for (const auto &id:dead) {
            Statement pins(db.value,"DELETE FROM pins WHERE session=?"); pins.Text(1,id); pins.Run();
            Statement reservation(db.value,"DELETE FROM reservations WHERE id=?"); reservation.Text(1,id); reservation.Run();
        }
        tx.Commit();
    }
    uint64_t ActiveContainer(Database &db,uint64_t required) const {
        CheckCurrentConfig(db);
        Check(required<=fileLimit-ContainerHeaderBytes,"cache_record_too_large");
        Statement active(db.value,"SELECT id,watermark FROM containers WHERE sealed=0 ORDER BY id DESC LIMIT 1");
        if (active.Row()) {
            const auto id=active.Number(0),watermark=active.Number(1);
            const uint64_t actual=FileSize(ContainerPath(id));
            // Never append over a crash tail or rewrite an older sealed file.
            if (actual==watermark && actual<=fileLimit && required<=fileLimit-actual) return id;
            Statement seal(db.value,"UPDATE containers SET sealed=1 WHERE id=?"); seal.Number(1,id); seal.Run();
        }
        return NewContainer(db);
    }
    ObjectLocation Append(Database &db,uint64_t kind,uint64_t sequence,const std::vector<char> &payload,
                          std::string *outHash=nullptr) const {
        Check(!payload.empty() && payload.size()<=BatchBytes,"cache_append_bound");
        const auto physical=RecordPhysicalBytes(payload.size());
        const auto container=ActiveContainer(db,physical);
        const auto offset=FileSize(ContainerPath(container));
        // [S1] 记录级摘要只在这里算一次；调用方通过 outHash 复用，避免同一批字节重复 SHA256。
        g_storeProfiler.recordDigestCalls.fetch_add(1,std::memory_order_relaxed);
        g_storeProfiler.recordDigestBytes.fetch_add(uint64_t(payload.size()),std::memory_order_relaxed);
        const auto hash=DiskPrefixCache::Digest(payload.data(),payload.size());
        if (outHash) *outHash=hash;
        auto header=MakeHeader(kind,sequence,hash,payload.size());
        std::vector<char> staging(static_cast<size_t>(physical));
        uint32_t sizes[2]{};
        std::vector<fastllm_flexkv_ssd::PackedSpan> spans{
            {header.data(),uint32_t(header.size()),&sizes[0]},
            {const_cast<char *>(payload.data()),uint32_t(payload.size()),&sizes[1]}};
        int fd=open(ContainerPath(container).c_str(),O_WRONLY|O_CLOEXEC|O_NOFOLLOW);
        Check(fd>=0,"cache_container_open");
        try {
            fastllm_flexkv_ssd::do_transfer_packed_blocks(fd,offset,physical,false,staging.data(),spans);
            Check(fdatasync(fd)==0,"cache_container_sync"); close(fd);
            FaultPoint(kind==PackedRecord?"payload_synced":"commit_synced");
        } catch (...) { close(fd); healthyError(); throw; }
        return {container,offset,uint64_t(payload.size()),physical,RecordHeaderBytes};
    }
    void healthyError() const { const_cast<Impl *>(this)->healthy=false; }
    void IndexRecord(Database &db,uint64_t container,uint64_t offset,const RecordHeader &header) const {
        if (header.sequence>db.Scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='sequence'"))
            SetMeta(db,"sequence",std::to_string(header.sequence));
        Statement put(db.value,"INSERT OR REPLACE INTO records(container,offset,physical,kind,sequence,hash) VALUES(?,?,?,?,?,?)");
        put.Number(1,container); put.Number(2,offset); put.Number(3,header.physical);
        put.Number(4,header.kind); put.Number(5,header.sequence); put.Text(6,header.hash); put.Run();
        Statement watermark(db.value,"INSERT INTO containers(id,watermark,sealed) VALUES(?,?,1) "
            "ON CONFLICT(id) DO UPDATE SET watermark=MAX(watermark,excluded.watermark)");
        watermark.Number(1,container); watermark.Number(2,Add(offset,header.physical)); watermark.Run();
    }
    void IndexPacked(Database &db,uint64_t container,uint64_t offset,const RecordHeader &header,const char *payload) const {
        Check(header.bytes>=8,"cache_packed_table");
        const auto count=GetU64(payload);
        Check(count>0 && count<=512 && 8+count*72<=header.bytes,"cache_packed_table");
        uint64_t cursor=8+count*72;
        for (uint64_t i=0;i<count;++i) {
            const char *entry=payload+8+i*72;
            const std::string hash(entry,64); const auto bytes=GetU64(entry+64);
            // [S1] 结构校验收尾始终执行；**逐对象重算 SHA256 默认开启**（见
            // StrictWriteVerifyEnabled()：未设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY 即返回 true）。
            // 如需关闭，显式设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY=0（内容在读回时仍逐个 SHA256 校验）。
            if (StrictWriteVerifyEnabled()) {
                g_storeProfiler.indexDigestCalls.fetch_add(1,std::memory_order_relaxed);
                g_storeProfiler.indexDigestBytes.fetch_add(bytes,std::memory_order_relaxed);
                Check(DiskPrefixCache::Digest(payload+cursor,bytes)==hash,"cache_packed_object");
            }
            Check(DiskPrefixCache::IsDigest(hash) && bytes>0 && bytes<=DiskPrefixCache::ChunkBytes &&
                  cursor<=header.bytes && bytes<=header.bytes-cursor,"cache_packed_object");
            g_storeProfiler.indexInserts.fetch_add(1,std::memory_order_relaxed);
            Statement put(db.value,"INSERT OR REPLACE INTO objects(hash,container,offset,payload_offset,bytes,physical) VALUES(?,?,?,?,?,?)");
            put.Text(1,hash); put.Number(2,container); put.Number(3,offset);
            put.Number(4,RecordHeaderBytes+cursor); put.Number(5,bytes); put.Number(6,header.physical); put.Run();
            cursor+=bytes;
        }
        Check(cursor==header.bytes,"cache_packed_coverage");
    }
    void IndexEvent(Database &db,uint64_t container,uint64_t offset,const RecordHeader &header,const Json &event) const {
        const std::string id=event["identity"].string_value(),key=event["key"].string_value();
        Check((id=="metadata" || DiskPrefixCache::IsDigest(id)) && !key.empty() && key.size()<=1024,
              "cache_event_identity");
        Statement history(db.value,"INSERT OR REPLACE INTO event_history(identity,key,container,offset,kind) VALUES(?,?,?,?,?)");
        history.Text(1,id); history.Text(2,key); history.Number(3,container); history.Number(4,offset); history.Number(5,header.kind); history.Run();
        Statement put(db.value,"INSERT INTO events(identity,key,kind,sequence,container,offset,physical,body,credit) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(identity,key) DO UPDATE SET kind=excluded.kind,sequence=excluded.sequence,container=excluded.container,"
            "offset=excluded.offset,physical=excluded.physical,body=excluded.body,credit=excluded.credit WHERE excluded.sequence>=events.sequence");
        put.Text(1,id); put.Text(2,key); put.Number(3,header.kind); put.Number(4,header.sequence);
        put.Number(5,container); put.Number(6,offset); put.Number(7,header.physical); put.Text(8,event.dump());
        put.Number(9,header.kind==EraseRecord?std::stoull(event["credit"].string_value()):0); put.Run();
    }
    void ApplyEvent(Database &db,uint64_t kind,const Json &event,bool verifyData) const {
        const std::string id=event["identity"].string_value(),key=event["key"].string_value();
        {
            Statement erase(db.value,"DELETE FROM refs WHERE identity=? AND key=?");
            erase.Text(1,id); erase.Text(2,key); erase.Run();
        }
        if (kind==EraseRecord) {
            Statement erase(db.value,"DELETE FROM checkpoints WHERE identity=? AND key=?");
            erase.Text(1,id); erase.Text(2,key); erase.Run(); return;
        }
        auto state=ParseState(db,event["state"]);
        Refs refs; CollectRefs(event["state"],refs);
        if (kind!=MetadataRecord) CollectRefs(state,refs);
        for (const auto &ref:refs) {
            ObjectLocation loc;
            Check(FindObject(db,ref.first,loc) && loc.bytes==ref.second,"cache_commit_missing_object");
            if (verifyData) ReadObject(db,ref.first,ref.second);
        }
        if (kind==MetadataRecord) {
            Check(id=="metadata","cache_metadata_identity");
            Statement put(db.value,"INSERT OR REPLACE INTO metadata(key,state) VALUES(?,?)");
            put.Text(1,key); put.Text(2,event["state"].dump()); put.Run();
        } else {
            Check(kind==CommitRecord && DiskPrefixCache::IsDigest(event["family"].string_value()),"cache_commit_kind");
            Statement put(db.value,"INSERT OR REPLACE INTO checkpoints(identity,key,family,kind,length,dtype,provenance,created,bytes,state,control_bytes,prefix) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)");
            put.Text(1,id); put.Text(2,key); put.Text(3,event["family"].string_value()); put.Text(4,event["kind"].string_value());
            put.Number(5,uint64_t(std::max(0,event["length"].int_value()))); put.Text(6,event["dtype"].string_value());
            put.Text(7,event["provenance"].string_value());
            put.Number(8,std::stoull(event["created"].string_value()));
            put.Number(9,std::stoull(event["bytes"].string_value())); put.Text(10,event["state"].dump());
            put.Number(11,std::stoull(event["control_bytes"].string_value()));
            put.Text(12,state["prefix"].string_value()); put.Run();
        }
        for (const auto &ref:refs) {
            Statement put(db.value,"INSERT OR IGNORE INTO refs(identity,key,hash) VALUES(?,?,?)");
            put.Text(1,id); put.Text(2,key); put.Text(3,ref.first); put.Run();
        }
    }
    bool Publish(Database &db,uint64_t kind,const Json &event,bool keepEquivalent=false) const {
        CheckCurrentConfig(db);
        Scan(db,false); // another process may have synced a commit before its SQLite transaction failed
        if (kind==CommitRecord) {
            Statement existing(db.value,"SELECT state FROM checkpoints WHERE identity=? AND key=?");
            existing.Text(1,event["identity"].string_value()); existing.Text(2,event["key"].string_value());
            if (existing.Row()) {
                const auto left=LoadRecord(db,event["identity"].string_value(),event["key"].string_value());
                const auto right=ParseState(db,event["state"]);
                Check(left==right || (keepEquivalent && EquivalentPrefix(left,right)),"checkpoint_key_conflict");
                return false;
            }
        }
        const auto sequence=Generation(db);
        auto eventFields=event.object_items();
        if (kind==CommitRecord) eventFields["created"]=std::to_string(sequence);
        const Json durableEvent(eventFields);
        const auto text=durableEvent.dump(); std::vector<char> payload(text.begin(),text.end());
        // [S1] 复用 Append 的记录摘要，去掉同一份文本的第二次 SHA256。
        std::string textHash;
        const auto location=Append(db,kind,sequence,payload,&textHash);
        RecordHeader header{kind,uint64_t(payload.size()),sequence,location.physical,textHash};
        try {
            Transaction tx(db); IndexRecord(db,location.container,location.offset,header);
            IndexEvent(db,location.container,location.offset,header,durableEvent);
            ApplyEvent(db,kind,durableEvent,false); tx.Commit();
            FaultPoint("catalog_published");
        } catch (...) { healthyError(); throw; }
        return true;
    }
    bool CatalogUsable() const {
        if (!fs::exists(dbPath)) return false;
        sqlite3 *raw=nullptr;
        int code=sqlite3_open_v2(dbPath.c_str(),&raw,SQLITE_OPEN_READONLY|SQLITE_OPEN_FULLMUTEX,nullptr);
        if (code!=SQLITE_OK) { if (raw) sqlite3_close(raw); Check(code==SQLITE_CORRUPT || code==SQLITE_NOTADB,"cache_catalog_open_failed"); return false; }
        sqlite3_stmt *statement=nullptr;
        code=sqlite3_prepare_v2(raw,"PRAGMA user_version",-1,&statement,nullptr);
        if (code==SQLITE_OK) code=sqlite3_step(statement);
        const bool usable=code==SQLITE_ROW && sqlite3_column_int(statement,0)==301;
        sqlite3_finalize(statement); sqlite3_close(raw);
        Check(code==SQLITE_ROW || code==SQLITE_CORRUPT || code==SQLITE_NOTADB,"cache_catalog_probe_failed");
        return usable;
    }
    void Scan(Database &db,bool full) const {
        uint64_t scanned=0,lastProgress=0;
        std::set<std::pair<std::string,std::string>> changedEvents;
        const auto started=std::chrono::steady_clock::now();
        if (full) std::fprintf(stderr,"[Prefix SSD] catalog rebuild: scanning self-describing containers; GC disabled.\n");
        for (const auto id:ContainerIds()) {
            const uint64_t actual=FileSize(ContainerPath(id));
            uint64_t offset=ContainerHeaderBytes;
            if (!full) {
                Statement known(db.value,"SELECT watermark FROM containers WHERE id=?"); known.Number(1,id);
                if (known.Row()) offset=known.Number(0);
                Check(actual>=offset,"cache_container_shorter_than_catalog");
                if (actual==offset) continue;
            }
            int fd=open(ContainerPath(id).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
            Check(fd>=0,"cache_container_open");
            try {
                char containerHeader[ContainerHeaderBytes]; ReadExact(fd,containerHeader,sizeof(containerHeader));
                VerifyContainerHeader(containerHeader,id);
                while (offset<actual) {
                    if (actual-offset<RecordHeaderBytes) throw std::runtime_error("cache_recovery_ambiguous_short_header_preserved");
                    char headerBytes[RecordHeaderBytes]; ReadExact(fd,headerBytes,sizeof(headerBytes),offset);
                    RecordHeader header;
                    try { header=ParseHeader(headerBytes); }
                    catch (...) {
                        // An unrecognizable interior header could have been a tombstone.
                        // Refuse ambiguous reconstruction instead of resurrecting data.
                        throw std::runtime_error("cache_recovery_ambiguous_header_preserved");
                    }
                    if (header.physical>actual-offset) {
                        Check(header.kind==PackedRecord,"cache_recovery_incomplete_commit_preserved");
                        break;
                    }
                    std::vector<char> payload(static_cast<size_t>(header.bytes));
                    FaultPoint("recovery_payload_read");
                    ReadExact(fd,payload.data(),payload.size(),offset+RecordHeaderBytes);
                    const bool checksum=DiskPrefixCache::Digest(payload.data(),payload.size())==header.hash;
                    if (!checksum && header.kind!=PackedRecord)
                        throw std::runtime_error("cache_recovery_commit_log_corrupt_preserved");
                    Transaction tx(db);
                    IndexRecord(db,id,offset,header);
                    if (checksum) {
                        if (header.kind==PackedRecord) IndexPacked(db,id,offset,header,payload.data());
                        else {
                            std::string error; auto event=Json::parse(std::string(payload.begin(),payload.end()),error);
                            Check(error.empty() && event.is_object(),"cache_event_json");
                            IndexEvent(db,id,offset,header,event);
                            changedEvents.emplace(event["identity"].string_value(),event["key"].string_value());
                        }
                    } else std::fprintf(stderr,"[Prefix SSD] damaged payload excluded: container=%llu offset=%llu.\n",
                        (unsigned long long)id,(unsigned long long)offset);
                    tx.Commit(); offset+=header.physical; scanned+=header.physical;
                    if (full && scanned-lastProgress>=(uint64_t(256)<<20)) {
                        std::fprintf(stderr,"[Prefix SSD] catalog rebuild: scanned=%.2f GiB elapsed=%.1fs.\n",
                            scanned/double(uint64_t(1)<<30),std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count());
                        lastProgress=scanned;
                    }
                }
                Statement mark(db.value,"INSERT INTO containers(id,watermark,sealed) VALUES(?,?,1) "
                    "ON CONFLICT(id) DO UPDATE SET watermark=excluded.watermark,sealed=1");
                mark.Number(1,id); mark.Number(2,offset); mark.Run();
                close(fd);
            } catch (...) { close(fd); throw; }
        }
        auto replay=[&](uint64_t kind,const std::string &body) {
            std::string error; auto event=Json::parse(body,error);
            Check(error.empty(),"cache_event_json");
            try { Transaction tx(db); ApplyEvent(db,kind,event,false); tx.Commit(); }
            catch (const std::exception &e) {
                const int code=sqlite3_errcode(db.value);
                Check(code==SQLITE_OK || code==SQLITE_ROW || code==SQLITE_DONE,"cache_replay_sql_failed");
                Check(kind==CommitRecord || kind==MetadataRecord,"cache_tombstone_replay_failed");
                if (!ConfirmedDataFailure(e)) throw;
                Transaction tx(db);
                Statement erase(db.value,"DELETE FROM checkpoints WHERE identity=? AND key=?");
                erase.Text(1,event["identity"].string_value()); erase.Text(2,event["key"].string_value()); erase.Run();
                if (kind==MetadataRecord) {
                    Statement metadata(db.value,"DELETE FROM metadata WHERE key=?");
                    metadata.Text(1,event["key"].string_value()); metadata.Run();
                }
                Statement refs(db.value,"DELETE FROM refs WHERE identity=? AND key=?");
                refs.Text(1,event["identity"].string_value()); refs.Text(2,event["key"].string_value()); refs.Run();
                tx.Commit();
                std::fprintf(stderr,"[Prefix SSD] checkpoint unavailable during rebuild: %s.\n",e.what());
            }
        };
        if (full) {
            Statement events(db.value,"SELECT kind,body FROM events ORDER BY sequence");
            while (events.Row()) replay(events.Number(0),events.Text(1));
        } else for (const auto &key:changedEvents) {
            Statement event(db.value,"SELECT kind,body FROM events WHERE identity=? AND key=?");
            event.Text(1,key.first); event.Text(2,key.second);
            if (event.Row()) replay(event.Number(0),event.Text(1));
        }
        if (full) std::fprintf(stderr,"[Prefix SSD] catalog rebuild complete: %.2f GiB; checksums verified.\n",
            scanned/double(uint64_t(1)<<30));
    }
    void Recover(bool force) {
        Lease metadata(root/".metadata",LOCK_EX);
        if (!force && CatalogUsable()) {
            Database db(dbPath);
            try {
                Scan(db,false); // O(container count + unindexed tail), no checkpoint/ref scan
                db.Exec("DELETE FROM reservations; DELETE FROM pins;"); healthy=true; return;
            } catch (const std::exception &e) {
                const int code=sqlite3_errcode(db.value);
                if (code!=SQLITE_CORRUPT && code!=SQLITE_NOTADB) throw;
                std::fprintf(stderr,"[Prefix SSD] catalog requires reconstruction: %s.\n",e.what());
            }
        }
        const fs::path temporary=base/"catalog.rebuild.sqlite3";
        // These are only abandoned rebuild projections, never payload containers.
        for (const auto &suffix:{"","-wal","-shm"}) {
            std::error_code error; fs::remove(temporary.string()+suffix,error);
            Check(!error,"cache_rebuild_temporary_cleanup");
        }
        try {
            Database rebuilt(temporary); Schema(rebuilt); Scan(rebuilt,true);
            SetMeta(rebuilt,"format","packed-log-1");
            rebuilt.Exec("PRAGMA wal_checkpoint(TRUNCATE);");
        } catch (...) {
            // Only this uncommitted projection belongs to the failed rebuild.
            // The original catalog and every container remain untouched.
            for (const auto &suffix:{"","-wal","-shm"}) {
                std::error_code error; fs::remove(temporary.string()+suffix,error);
            }
            throw;
        }
        int fd=open(temporary.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
        Check(fd>=0,"cache_rebuild_open"); const int synced=fsync(fd); close(fd);
        Check(synced==0,"cache_rebuild_sync");
        // Exclusive lifecycle lease prevents readers/writers opening the old WAL.
        // Complete durable log remains authoritative throughout this replacement.
        Check(unlink((dbPath.string()+"-wal").c_str())==0 || errno==ENOENT,"cache_old_wal_remove");
        Check(unlink((dbPath.string()+"-shm").c_str())==0 || errno==ENOENT,"cache_old_shm_remove");
        Check(rename(temporary.c_str(),dbPath.c_str())==0,"cache_catalog_replace");
        SyncDirectory(base); healthy=true;
    }
    void Reserve(Database &db,const std::string &id,uint64_t &remaining,uint64_t required) const {
        CheckCurrentConfig(db);
        if (remaining>=required) return;
        const auto extra=required-remaining,used=Used(db);
        Check(limit>CatalogHeadroom && used<=limit-CatalogHeadroom && extra<=limit-CatalogHeadroom-used,"disk_quota");
        remaining+=extra;
        Statement put(db.value,"INSERT OR REPLACE INTO reservations(id,bytes,pid,start) VALUES(?,?,?,?)");
        put.Text(1,id); put.Number(2,remaining); put.Number(3,getpid()); put.Text(4,ProcessStamp(getpid())); put.Run();
    }
    void Consume(Database &db,const std::string &id,uint64_t &remaining,uint64_t bytes) const {
        Check(bytes<=remaining,"cache_reservation_underflow"); remaining-=bytes;
        Statement put(db.value,"UPDATE reservations SET bytes=? WHERE id=?");
        put.Number(1,remaining); put.Text(2,id); put.Run();
    }
    void DropUnreferenced(Database &db) const {
        // No active session holds a lifecycle read lease. Tombstones were synced
        // before reference removal; database failure stops before physical deletion.
        {
            Transaction tx(db);
            db.Exec("DELETE FROM events WHERE kind=3 AND NOT EXISTS(SELECT 1 FROM event_history h WHERE h.identity=events.identity AND h.key=events.key AND h.kind=2);");
            db.Exec("DELETE FROM objects WHERE NOT EXISTS(SELECT 1 FROM refs WHERE refs.hash=objects.hash) "
                "AND NOT EXISTS(SELECT 1 FROM pins WHERE pins.hash=objects.hash);");
            db.Exec("DELETE FROM records WHERE NOT EXISTS(SELECT 1 FROM objects WHERE objects.container=records.container AND objects.offset=records.offset) "
                "AND NOT EXISTS(SELECT 1 FROM events WHERE events.container=records.container AND events.offset=records.offset);");
            tx.Commit();
        }
        std::vector<uint64_t> dead;
        for (const auto id:ContainerIds()) {
            Statement live(db.value,"SELECT COUNT(*) FROM records WHERE container=?"); live.Number(1,id);
            if (live.Row() && live.Number(0)==0) { dead.push_back(id); continue; }
            Statement others(db.value,"SELECT COUNT(*) FROM records WHERE container=? AND kind<>3"); others.Number(1,id);
            if (!others.Row() || others.Number(0)!=0) continue;
            Statement outside(db.value,"SELECT COUNT(*) FROM events e JOIN event_history h ON h.identity=e.identity AND h.key=e.key "
                "WHERE e.container=? AND e.kind=3 AND h.kind IN (2,4) AND h.container<>?");
            outside.Number(1,id); outside.Number(2,id);
            if (outside.Row() && outside.Number(0)==0) {
                Transaction tx(db);
                Statement events(db.value,"DELETE FROM events WHERE container=? AND kind=3"); events.Number(1,id); events.Run();
                Statement records(db.value,"DELETE FROM records WHERE container=?"); records.Number(1,id); records.Run();
                tx.Commit(); dead.push_back(id);
            }
        }
        for (const auto id:dead) {
            FaultPoint("before_unlink");
            Check(unlink(ContainerPath(id).c_str())==0 || errno==ENOENT,"cache_container_unlink");
            SyncDirectory(containers);
            Statement history(db.value,"DELETE FROM event_history WHERE container=?"); history.Number(1,id); history.Run();
            Statement erase(db.value,"DELETE FROM containers WHERE id=?"); erase.Number(1,id); erase.Run();
        }
    }
    bool BorrowRetiredControl(Database &db,uint64_t needed) const {
        constexpr uint64_t publicationBudget=256*1024;
        struct Credit { Json body; uint64_t bytes; };
        std::vector<Credit> credits; uint64_t available=0;
        {
            Statement query(db.value,"SELECT body,credit FROM events WHERE kind=3 AND credit>262144 ORDER BY sequence");
            while (query.Row() && available<needed) {
                std::string error; auto body=Json::parse(query.Text(0),error); Check(error.empty(),"cache_credit_json");
                const auto bytes=query.Number(1); credits.push_back({body,bytes}); available=Add(available,bytes-publicationBudget);
            }
        }
        if (available<needed) return false;
        for (const auto &credit:credits) {
            if (!needed) break;
            const uint64_t taken=std::min(needed,credit.bytes-publicationBudget);
            auto body=credit.body.object_items(); body["credit"]=std::to_string(credit.bytes-taken-publicationBudget);
            const uint64_t before=PhysicalBytes(); Publish(db,EraseRecord,Json(body));
            const uint64_t after=PhysicalBytes();
            Check(after<=before || after-before<=publicationBudget,"cache_credit_publication_budget");
            needed-=taken;
        }
        return true;
    }
    bool CompactOne(Database &db,uint64_t admission) const {
        // Move one bounded live record from a fragmented file, then release the
        // exclusive maintenance lease. Readers can run between each bounded step.
        Statement source(db.value,"SELECT c.id,c.watermark,COALESCE(SUM(r.physical),0) FROM containers c "
            "LEFT JOIN records r ON r.container=c.id GROUP BY c.id HAVING c.watermark>4096+COALESCE(SUM(r.physical),0) "
            "ORDER BY c.id LIMIT 1");
        if (!source.Row()) return false;
        const uint64_t oldContainer=source.Number(0),fileBytes=source.Number(1),liveBytes=source.Number(2);
        if (liveBytes==0 || liveBytes*4>fileBytes*3) return false;
        Statement next(db.value,"SELECT offset,physical FROM records WHERE container=? ORDER BY offset LIMIT 1");
        next.Number(1,oldContainer); if (!next.Row()) return false;
        const auto offset=next.Number(0),physical=next.Number(1);
        (void)admission;
        RecordHeader header; auto record=ReadRecord(oldContainer,offset,physical,&header);
        std::vector<char> payload; uint64_t catalogBudget=CatalogHeadroom;
        if (header.kind==PackedRecord) {
            struct Member { std::string hash; uint64_t bytes,offset; };
            std::vector<Member> members;
            Statement live(db.value,"SELECT hash,bytes,payload_offset FROM objects WHERE container=? AND offset=? ORDER BY payload_offset");
            live.Number(1,oldContainer); live.Number(2,offset);
            uint64_t total=8;
            while (live.Row()) { members.push_back({live.Text(0),live.Number(1),live.Number(2)}); total=Add(total,72+live.Number(1)); }
            Check(!members.empty() && total<=BatchBytes,"cache_compaction_members");
            catalogBudget=CatalogBudget(members.size()+8);
            payload.resize(static_cast<size_t>(total)); PutU64(payload.data(),members.size());
            size_t cursor=8+72*members.size();
            for (size_t i=0;i<members.size();++i) {
                const auto &member=members[i];
                Check(member.offset<=record.size() && member.bytes<=record.size()-member.offset,"cache_compaction_bounds");
                memcpy(payload.data()+8+72*i,member.hash.data(),64); PutU64(payload.data()+8+72*i+64,member.bytes);
                memcpy(payload.data()+cursor,record.data()+member.offset,member.bytes); cursor+=member.bytes;
            }
        } else payload.assign(record.begin()+RecordHeaderBytes,record.begin()+RecordHeaderBytes+header.bytes);
        Statement seal(db.value,"UPDATE containers SET sealed=1 WHERE id=?"); seal.Number(1,oldContainer); seal.Run();
        const uint64_t budget=Add(Add(RecordPhysicalBytes(payload.size()),ContainerHeaderBytes),catalogBudget);
        const uint64_t used=Used(db);
        if ((used>limit || budget>limit-used) && !BorrowRetiredControl(db,budget)) return false;
        std::string targetHash;
        const auto target=Append(db,header.kind,header.sequence,payload,&targetHash);
        // Full rebuild scans containers in id order. Compaction must always
        // publish the replacement after its source so duplicate hashes/events
        // deterministically resolve to the new physical extent.
        Check(target.container>oldContainer,"cache_compaction_order");
        FaultPoint("compaction_synced");
        header.bytes=payload.size(); header.physical=target.physical;
        header.hash=targetHash;   // [S1] 复用 Append 的摘要
        {
            Transaction tx(db); IndexRecord(db,target.container,target.offset,header);
            if (header.kind==PackedRecord) IndexPacked(db,target.container,target.offset,header,payload.data());
            else {
                std::string error; auto event=Json::parse(std::string(payload.begin(),payload.end()),error);
                Check(error.empty(),"cache_compaction_event"); IndexEvent(db,target.container,target.offset,header,event);
            }
            Statement erase(db.value,"DELETE FROM records WHERE container=? AND offset=?");
            erase.Number(1,oldContainer); erase.Number(2,offset); erase.Run(); tx.Commit();
            FaultPoint("compaction_published");
        }
        DropUnreferenced(db);
        SetMeta(db,"compaction_bytes",std::to_string(Add(target.physical,
            db.Scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='compaction_bytes'"))));
        return true;
    }
    bool MaintainStep(uint64_t required) {
        Check(healthy,"cache_needs_recovery");
        std::unique_lock<std::shared_mutex> writerGuard(writerGate,std::try_to_lock);
        if (!writerGuard.owns_lock()) throw std::runtime_error("cache_busy");
        std::unique_lock<std::mutex> step(maintenanceStepMutex,std::try_to_lock);
        if (!step.owns_lock()) throw std::runtime_error("cache_busy");
        Lease writers(root/".writers",LOCK_EX|LOCK_NB);
        Lease lifecycle(root/".lease",LOCK_EX|LOCK_NB);
        Lease metadata(root/".metadata",LOCK_EX);
        Database db(dbPath);
        CheckCurrentConfig(db); ReapDeadWriters(db);
        Scan(db,false); // replay durable publication tails before any collection
        DropUnreferenced(db);
        uint64_t used=Used(db);
        SetMeta(db,"shrink_debt",std::to_string(used>limit?used-limit:0));
        if (used<=limit && required<=limit-used) return true;
        if (CompactOne(db,required)) return false;
        Statement oldest(db.value,"SELECT identity,key,control_bytes FROM checkpoints ORDER BY created,identity,key LIMIT 1");
        if (!oldest.Row()) return false;
        const auto oldIdentity=oldest.Text(0),oldKey=oldest.Text(1);
        Statement refCount(db.value,"SELECT COUNT(*) FROM refs WHERE identity=? AND key=?");
        refCount.Text(1,oldIdentity); refCount.Text(2,oldKey); Check(refCount.Row(),"cache_ref_count");
        const uint64_t deletionCharge=Add(256*1024,refCount.Number(0)*16384);
        const uint64_t oldControl=oldest.Number(2);
        if (oldControl<deletionCharge) {
            // A checkpoint cannot be safely retired without the control bytes
            // reserved with it.  Keep the durable checkpoint and let the
            // caller's bounded admission loop report a write skip instead of
            // aborting maintenance with an unclassified exception.
            std::fprintf(stderr,
                "[Prefix SSD] write_skipped: reason=missing_deletion_budget "
                "control_bytes=%llu required=%llu identity=%s key=%s.\n",
                (unsigned long long)oldControl,
                (unsigned long long)deletionCharge,
                oldIdentity.c_str(), oldKey.c_str());
            return false;
        }
        const Json tombstone=Json::object{{"identity",oldIdentity},{"key",oldKey},
            {"credit",std::to_string(oldControl-deletionCharge)}};
        // This deletion was budgeted when the checkpoint was accepted. It can
        // consume that existing reservation even after the user lowers quota.
        const uint64_t tombstoneBudget=deletionCharge;
        const uint64_t physicalBefore=PhysicalBytes();
        Publish(db,EraseRecord,tombstone);
        DropUnreferenced(db);
        try {
            db.Exec("PRAGMA wal_checkpoint(TRUNCATE);");
        } catch (const std::exception &error) {
            const std::string detail = error.what();
            if (detail.find("locked") == std::string::npos &&
                detail.find("busy") == std::string::npos) throw;
            std::fprintf(stderr,"[Prefix SSD] WAL compaction deferred: %s.\n",detail.c_str());
        }
        const uint64_t physicalAfter=PhysicalBytes();
        Check(physicalAfter<=physicalBefore || physicalAfter-physicalBefore<=tombstoneBudget,"cache_deletion_budget_exceeded");
        const uint64_t after=Used(db);
        SetMeta(db,"shrink_debt",std::to_string(after>limit?after-limit:0));
        const uint64_t reclaimed=physicalBefore>PhysicalBytes()?physicalBefore-PhysicalBytes():0;
        std::fprintf(stderr,"[Prefix SSD] FIFO: reason=%s evicted_checkpoints=1 reclaimed_bytes=%llu used=%llu quota=%llu debt=%llu.\n",
            used>limit?"shrink":"new_write_admission",(unsigned long long)reclaimed,(unsigned long long)after,
            (unsigned long long)limit,(unsigned long long)(after>limit?after-limit:0));
        return after<=limit && required<=limit-after;
    }
};
struct DiskPrefixCache::WriteSession::State {
    std::shared_ptr<Impl> owner;
    std::unique_ptr<Lease> writerLease,lease;
    std::shared_lock<std::shared_mutex> writerGate;
    std::string id=Unique();
    uint64_t remaining=0,written=0,pendingBytes=0;
    std::vector<PendingRecord> pending;
    std::set<std::string> pendingHashes;
    Refs verified,pinsToPublish;
    std::vector<std::pair<std::string,std::string>> v4Chunks;
    std::shared_ptr<DiskPrefixV4Store::ObjectLease> objectLease;
    std::shared_ptr<DiskPrefixV5Store::Writer> v5Writer;
    explicit State(std::shared_ptr<Impl> owner_):owner(std::move(owner_)) {}
    void Flush();
    void EnsureBudget(uint64_t required);
    ~State() {
        if (writerLease) try {
            Lease metadata(owner->root/".metadata",LOCK_EX); Database db(owner->dbPath);
            Statement erase(db.value,"DELETE FROM reservations WHERE id=?"); erase.Text(1,id); erase.Run();
            Statement pins(db.value,"DELETE FROM pins WHERE session=?"); pins.Text(1,id); pins.Run();
            owner->maintenanceChanged.notify_all();
        } catch (...) {}
    }
};
struct DiskPrefixCache::ReadSession::State {
    struct CachedGroup {
        uint64_t container=0,offset=0,bytes=0;
        std::vector<char> data;
        std::set<uint64_t> verifiedRecords;
    };
    std::shared_ptr<Impl> owner;
    std::unique_ptr<Lease> lease;
    std::unique_ptr<Database> database;
    std::map<std::string,std::string> v4Objects;
    // B1：本会话是否走的是「清单-only」加载。
    // 只有元数据、没有载荷 ⇒ 一旦有人拿它去 ReadMany，必须**当场 fail-closed**，
    // 而不是抛出含义模糊的 invalid_chunk_reference（见 ReadMany 的守卫）。
    bool manifestOnly = false;
    std::shared_ptr<void> v5Pin;   // keeps the V5 file chain/slot of the last Load alive
    uint64_t read=0;
    Refs allowed;
    // The lifecycle lease makes these bytes immutable. Keep at most four
    // bounded groups (128 MiB + 16 KiB), in addition to existing in-flight I/O.
    std::vector<std::shared_ptr<CachedGroup>> cachedGroups;
    // v4 mode has no v3 catalog and no v3 lease file; opening them would fail
    // (or wrongly create the legacy layout), so both are conditional.
    explicit State(std::shared_ptr<Impl> owner_):owner(std::move(owner_)) {
        // v4 readers acquire short lifecycle/root leases through PinCheckpoint
        // only after resolving a root. Keeping a session-wide lifecycle lock
        // would prevent background Compact indefinitely.
        if (!owner->v4Mode && !owner->v5Mode) {
            lease.reset(new Lease(owner->root/".lease", LOCK_SH));
            database.reset(new Database(owner->dbPath));
        }
    }
};
void DiskPrefixCache::Require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
std::string DiskPrefixCache::Digest(const void *data,size_t size) {
    unsigned char digest[EVP_MAX_MD_SIZE]; unsigned int count=0;
    Check(EVP_Digest(data,size,digest,&count,EVP_sha256(),nullptr)==1 && count==32,"sha256_failed");
    static const char hex[]="0123456789abcdef";
    std::string result; result.reserve(64);
    for (unsigned int i=0;i<count;++i) { result+=hex[digest[i]>>4]; result+=hex[digest[i]&15]; }
    return result;
}
std::string DiskPrefixCache::Digest(const std::string &value) { return Digest(value.data(),value.size()); }
// [store-profile] 单调快照：与 StoreProfile 同样的字段，但**永不清零**。
// P3-B（2026-10-01）：把 v4 存储“已受理未持久化”的在途字节暴露给模型侧做主机预算记账。
uint64_t DiskPrefixCache::InFlightBytes() const {
    if (impl->v5Mode) return impl->v5->InFlightBytes();
    return (impl->v4Mode && impl->v4) ? impl->v4->InFlightBytes() : 0;
}
bool DiskPrefixCache::DrainPending(std::string &error) {
    error.clear();
    if (impl->v5Mode) return impl->v5->Drain(error);
    if (!impl->v4Mode || !impl->v4) return true;
    return impl->v4->ReapCheckpoint(nullptr, error);
}

// §保最新、丢最旧：把 store 层"取消最老未开工的已受理作业"透到外壳，
// 供模型层在主机预算不足时先腾额度、再受理最新那笔。非 v4 模式没有该流水线，返回 false。
bool DiskPrefixCache::CancelOldestPending(uint64_t *freedBytes, std::string &error) {
    if (impl->v5Mode) return impl->v5->CancelOldestPending(freedBytes, error);
    if (!impl->v4Mode || !impl->v4) {
        error = "v4_not_enabled";
        if (freedBytes != nullptr) *freedBytes = 0;
        return false;
    }
    return impl->v4->CancelOldestPending(freedBytes, error);
}

Json DiskPrefixCache::StorageProfile(bool drain) const {
    if (impl->v5Mode) return impl->v5->Profile();
    if (!impl->v4Mode || !impl->v4) return Json::object{};
    // Profile() 产出的是 JSON 文本（含缩进），这里解析回对象便于打印数值字段。
    std::string parseError;
    const Json parsed = Json::parse(impl->v4->Profile(drain), parseError);
    return parsed.is_object() ? parsed : Json::object{};
}

Json DiskPrefixCache::StoreProfileSnapshot() {
    return StoreProfile(false);
}
// [store-profile] 导出写盘路径诊断计数（reset=true 时清零，便于按检查点统计）
Json DiskPrefixCache::StoreProfile(bool reset) {
    Json out = Json::object{
        {"store_digest_calls",(double)g_storeProfiler.storeDigestCalls.load()},
        {"store_digest_bytes",(double)g_storeProfiler.storeDigestBytes.load()},
        {"record_digest_calls",(double)g_storeProfiler.recordDigestCalls.load()},
        {"record_digest_bytes",(double)g_storeProfiler.recordDigestBytes.load()},
        {"index_digest_calls",(double)g_storeProfiler.indexDigestCalls.load()},
        {"index_digest_bytes",(double)g_storeProfiler.indexDigestBytes.load()},
        {"index_inserts",(double)g_storeProfiler.indexInserts.load()},
        {"chunks",(double)g_storeProfiler.chunks.load()},
        {"flushes",(double)g_storeProfiler.flushes.load()},
        {"db_opens",(double)g_storeProfiler.dbOpens.load()},
        {"flush_db_opens",(double)g_storeProfiler.flushDbOpens.load()},
        {"find_calls",(double)g_storeProfiler.findCalls.load()},
        {"find_hits",(double)g_storeProfiler.findHits.load()},
        {"readbacks",(double)g_storeProfiler.readbacks.load()},
        {"readback_bytes",(double)g_storeProfiler.readbackBytes.load()},
        {"alloc_ms",(double)g_storeProfiler.allocNs.load()/1e6},
        {"copy_ms",(double)g_storeProfiler.copyNs.load()/1e6},
        {"hash_ms",(double)g_storeProfiler.hashNs.load()/1e6},
        {"db_ms",(double)g_storeProfiler.dbNs.load()/1e6},
        {"json_ms",(double)g_storeProfiler.jsonNs.load()/1e6},
        {"flush_ms",(double)g_storeProfiler.flushNs.load()/1e6}};
    if (reset) g_storeProfiler.Reset();
    return out;
}
bool DiskPrefixCache::IsDigest(const std::string &value) {
    return value.size()==64 && value.find_first_not_of("0123456789abcdef")==std::string::npos;
}
std::string DiskPrefixCache::PrefixKey(const std::vector<int> &tokens,int length,const std::string &media) {
    Check(length>0 && size_t(length)<=tokens.size(),"invalid_prefix_length");
    std::string bytes="fastllm-prefix-v4";
    for (int i=0;i<8;++i) bytes.push_back(char(uint64_t(media.size())>>(i*8)));
    bytes+=media;
    for (int index=0;index<length;++index) {
        const uint32_t token=uint32_t(tokens[index]);
        for (int i=0;i<4;++i) bytes.push_back(char(token>>(i*8)));
    }
    return Digest(bytes);
}
DiskPrefixCache::DiskPrefixCache(const std::string &directory,const std::string &identity,uint64_t limitBytes,
        const std::string &familyIdentity,int ioThreads)
    :impl(std::make_shared<Impl>(directory,identity,limitBytes,familyIdentity,0,ioThreads,true)) {}
DiskPrefixCache::~DiskPrefixCache()=default;
DiskPrefixCache::WriteSession::WriteSession(std::unique_ptr<State> value):state(std::move(value)) {}
DiskPrefixCache::WriteSession::WriteSession(WriteSession &&) noexcept=default;
DiskPrefixCache::WriteSession &DiskPrefixCache::WriteSession::operator=(WriteSession &&) noexcept=default;
DiskPrefixCache::WriteSession::~WriteSession()=default;
DiskPrefixCache::ReadSession::ReadSession(std::unique_ptr<State> value):state(std::move(value)) {}
DiskPrefixCache::ReadSession::ReadSession(ReadSession &&) noexcept=default;
DiskPrefixCache::ReadSession &DiskPrefixCache::ReadSession::operator=(ReadSession &&) noexcept=default;
DiskPrefixCache::ReadSession::~ReadSession()=default;
DiskPrefixCache::WriteSession DiskPrefixCache::BeginWrite(uint64_t estimate) {
    Check(impl->healthy,"cache_needs_recovery");
    auto state=std::make_unique<WriteSession::State>(impl);
    if (impl->v5Mode) {
        // Bytes stay staged in host memory until Commit (G2); the writer owns the file lease.
        state->v5Writer = impl->v5->BeginWriter();
        return WriteSession(std::move(state));
    }
    // A v4 writer must protect referenced bytes with the same lifecycle file
    // used by Compact, and must not create legacy reservations or lease files.
    if (impl->v4Mode) {
        std::string leaseError;
        state->objectLease = impl->v4->BeginObjectLease(leaseError);
        Check(state->objectLease != nullptr,
              leaseError.empty()?"v4_object_lease":leaseError.c_str());
        return WriteSession(std::move(state));
    }
    state->writerGate = std::shared_lock<std::shared_mutex>(impl->writerGate);
    state->writerLease.reset(new Lease(impl->root/".writers",LOCK_SH));
    // A full-context estimate is not admission for a full-context allocation.
    // Reserve a bounded window; every append extends it before touching disk.
    const uint64_t requested=Add(std::min<uint64_t>(estimate?estimate:Alignment*16,ChunkBytes),CatalogHeadroom);
    // v4 mode manages its own quota inside the v4 store and takes no v3 writer
    // lease or reservation row.
    Check(requested<=impl->limit,"disk_quota_no_write_window");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for (;;) {
        state->lease.reset(new Lease(impl->root/".lease",LOCK_SH));
        try {
            Lease metadata(impl->root/".metadata",LOCK_EX); Database db(impl->dbPath);
            impl->Reserve(db,state->id,state->remaining,requested);
            return WriteSession(std::move(state));
        } catch (const std::exception &e) {
            state->lease.reset();
            if (std::string(e.what())!="disk_quota") throw;
            state->writerGate.unlock();
            state->writerLease.reset();
            try { impl->MaintainStep(Add(requested,CatalogHeadroom)); } catch (const std::exception &maintenanceError) {
                if (std::string(maintenanceError.what())!="cache_busy") throw;
            }
            state->writerLease.reset(new Lease(impl->root/".writers",LOCK_SH));
            state->writerGate.lock();
            if (std::chrono::steady_clock::now()>=deadline)
                throw std::runtime_error("write_skipped_quota_or_pinned_timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
}
DiskPrefixCache::ReadSession DiskPrefixCache::BeginRead() const {
    Check(impl->healthy,"cache_needs_recovery");
    return ReadSession(std::make_unique<ReadSession::State>(impl));
}
// §22.4.4 甲：接收固定哈希池产出的"摘要+持有字节"结果，直接进入待提交区，
// **不再重算摘要**（结果只能由 HashPackPool 构造，调用方无法伪造 hash）。
Json::array DiskPrefixCache::WriteSession::StoreVerified(prefix_write::HashedChunk &&chunk) {
    Check(state && state->owner->healthy,"closed_write_session");
    Check(state->owner->v4Mode || state->owner->v5Mode,"store_verified_requires_v4");
    std::string digest;
    std::string payload;
    Check(chunk.Take(digest,payload),"store_verified_invalid_chunk");
    const size_t count = payload.size();
    state->v4Chunks.push_back({digest,std::move(payload)});
    if (state->objectLease) {
        std::string leaseError;
        Check(state->owner->v4->ExtendObjectLease(state->objectLease, {digest}, leaseError),
              leaseError.empty()?"v4_object_lease":leaseError.c_str());
    }
    state->written += count;
    Json::array chunks;
    chunks.emplace_back(Json::object{{"sha256",digest},{"bytes",double(count)}});
    return chunks;
}

Json::array DiskPrefixCache::WriteSession::StoreBytes(size_t bytes,
        const std::function<void(size_t,char *,size_t)> &read,const Json &previous) {
    Check(state && state->owner->healthy,"closed_write_session"); (void)previous;
    if (state->owner->v4Mode || state->owner->v5Mode) {
        Json::array chunks;
        for (size_t offset=0;offset<bytes;) {
            const size_t count=std::min<size_t>(ChunkBytes,bytes-offset);
            std::string payload(count,'\0');
            read(offset,payload.data(),count);
            const std::string hash=Digest(payload.data(),payload.size());
            state->v4Chunks.push_back({hash,std::move(payload)});
            if (state->objectLease) {
                std::string leaseError;
                Check(state->owner->v4->ExtendObjectLease(state->objectLease, {hash}, leaseError),
                      leaseError.empty()?"v4_object_lease":leaseError.c_str());
            }
            state->written+=count;
            chunks.emplace_back(Json::object{{"sha256",hash},{"bytes",double(count)}});
            offset+=count;
        }
        return chunks;
    }
    const auto capacity=state->owner->fileLimit-ContainerHeaderBytes-RecordHeaderBytes-Alignment-80;
    const size_t chunkLimit=static_cast<size_t>(std::min<uint64_t>(ChunkBytes,capacity));
    Json::array chunks;
    for (size_t offset=0;offset<bytes;) {
        const size_t count=std::min(chunkLimit,bytes-offset);
        if (state->pending.size()>=512 || state->pendingBytes+count+72*(state->pending.size()+1)+8>
            std::min<uint64_t>(PendingBytes,capacity)) state->Flush();
        const auto tAlloc=std::chrono::steady_clock::now();
        std::vector<char> payload(count);
        const auto tCopy=std::chrono::steady_clock::now();
        read(offset,payload.data(),count);
        const auto tHash=std::chrono::steady_clock::now();
        const std::string hash=Digest(payload.data(),payload.size());
        const auto tHashEnd=std::chrono::steady_clock::now();
        g_storeProfiler.chunks.fetch_add(1,std::memory_order_relaxed);
        g_storeProfiler.storeDigestCalls.fetch_add(1,std::memory_order_relaxed);
        g_storeProfiler.storeDigestBytes.fetch_add(count,std::memory_order_relaxed);
        g_storeProfiler.allocNs.fetch_add(ProfNs(tAlloc,tCopy),std::memory_order_relaxed);
        g_storeProfiler.copyNs.fetch_add(ProfNs(tCopy,tHash),std::memory_order_relaxed);
        g_storeProfiler.hashNs.fetch_add(ProfNs(tHash,tHashEnd),std::memory_order_relaxed);
        bool exists=state->verified.count(hash)!=0;
        if (!exists) {
            const auto tDb=std::chrono::steady_clock::now();
            Database db(state->owner->dbPath); ObjectLocation location;
            g_storeProfiler.dbOpens.fetch_add(1,std::memory_order_relaxed);
            g_storeProfiler.findCalls.fetch_add(1,std::memory_order_relaxed);
            if (state->owner->FindObject(db,hash,location) && location.bytes==count) {
                try {
                    g_storeProfiler.readbacks.fetch_add(1,std::memory_order_relaxed);
                    g_storeProfiler.readbackBytes.fetch_add(count,std::memory_order_relaxed);
                    state->owner->ReadObject(db,hash,count); exists=true;
                }
                catch (const std::exception &) { exists=false; }
            }
            g_storeProfiler.dbNs.fetch_add(ProfNs(tDb,std::chrono::steady_clock::now()),std::memory_order_relaxed);
        }
        if (exists) g_storeProfiler.findHits.fetch_add(1,std::memory_order_relaxed);
        if (!exists && !state->pendingHashes.count(hash)) {
            state->pendingHashes.insert(hash); state->pendingBytes+=count;
            state->pending.push_back({hash,std::move(payload)});
        }
        if (!state->verified.count(hash)) state->pinsToPublish[hash]=count;
        state->verified[hash]=count;
        const auto tJson=std::chrono::steady_clock::now();
        chunks.emplace_back(Json::object{{"sha256",hash},{"bytes",double(count)}}); offset+=count;
        g_storeProfiler.jsonNs.fetch_add(ProfNs(tJson,std::chrono::steady_clock::now()),std::memory_order_relaxed);
    }
    return chunks;
}
void DiskPrefixCache::WriteSession::State::EnsureBudget(uint64_t required) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for (;;) {
        if (!lease) lease.reset(new Lease(owner->root/".lease",LOCK_SH));
        try {
            Lease metadata(owner->root/".metadata",LOCK_EX); Database db(owner->dbPath);
            if (!pinsToPublish.empty()) {
                Transaction tx(db);
                for (const auto &pin:pinsToPublish) {
                    Statement put(db.value,"INSERT OR IGNORE INTO pins(session,hash) VALUES(?,?)");
                    put.Text(1,id); put.Text(2,pin.first); put.Run();
                }
                tx.Commit(); pinsToPublish.clear();
            }
            owner->Reserve(db,id,remaining,required); return;
        } catch (const std::exception &error) {
            if (std::string(error.what())!="disk_quota") throw;
        }
        // Durable session pins protect unpublished/shared objects while GC
        // reclaims unrelated FIFO checkpoints. The writer lifecycle lease keeps
        // another constructor from mistaking this live session for a crash.
        lease.reset();
        writerGate.unlock();
        writerLease.reset();
        try { owner->MaintainStep(Add(required-std::min(required,remaining),CatalogHeadroom)); }
        catch (const std::exception &error) { if (std::string(error.what())!="cache_busy") throw; }
        writerLease.reset(new Lease(owner->root/".writers",LOCK_SH));
        writerGate.lock();
        if (std::chrono::steady_clock::now()>=deadline) {
            std::fprintf(stderr,"[Prefix SSD] write_skipped: reason=quota_or_pinned_timeout required=%llu.\n",
                (unsigned long long)required);
            throw std::runtime_error("write_skipped_quota_or_pinned_timeout");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
}
void DiskPrefixCache::WriteSession::State::Flush() {
    if (owner->v4Mode || owner->v5Mode) return;
    if (pending.empty()) return;
    Check(owner->healthy,"cache_needs_recovery");
    const auto tFlush=std::chrono::steady_clock::now();
    std::vector<char> payload(static_cast<size_t>(8+pending.size()*72+pendingBytes));
    PutU64(payload.data(),pending.size());
    size_t cursor=8+pending.size()*72;
    for (size_t i=0;i<pending.size();++i) {
        memcpy(payload.data()+8+i*72,pending[i].hash.data(),64);
        PutU64(payload.data()+8+i*72+64,pending[i].payload.size());
        memcpy(payload.data()+cursor,pending[i].payload.data(),pending[i].payload.size()); cursor+=pending[i].payload.size();
    }
    const uint64_t physical=RecordPhysicalBytes(payload.size());
    // Catalog/WAL and a new container header are included, not hidden after I/O.
    const uint64_t charge=Add(Add(physical,ContainerHeaderBytes),CatalogBudget(pending.size()+8));
    EnsureBudget(charge);
    Lease metadata(owner->root/".metadata",LOCK_EX); Database db(owner->dbPath);
    g_storeProfiler.flushDbOpens.fetch_add(1,std::memory_order_relaxed);
    const uint64_t before=owner->PhysicalBytes();
    // [S1] 复用 Append 已算好的记录摘要，不再对同一批字节重复 SHA256（存储值不变）。
    std::string recordHash;
    const auto location=owner->Append(db,PackedRecord,0,payload,&recordHash);
    RecordHeader header{PackedRecord,uint64_t(payload.size()),0,location.physical,recordHash};
    try {
        Transaction tx(db); owner->IndexRecord(db,location.container,location.offset,header);
        owner->IndexPacked(db,location.container,location.offset,header,payload.data()); tx.Commit();
        owner->SetMeta(db,"logical_write_bytes",std::to_string(Add(pendingBytes,
            db.Scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='logical_write_bytes'"))));
        owner->SetMeta(db,"physical_write_bytes",std::to_string(Add(physical,
            db.Scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='physical_write_bytes'"))));
        const uint64_t after=owner->PhysicalBytes();
        Check(after<=before || after-before<=charge,"cache_append_budget_exceeded");
        owner->Consume(db,id,remaining,charge);
        written+=physical;
        pending.clear(); pendingHashes.clear(); pendingBytes=0;
        g_storeProfiler.flushes.fetch_add(1,std::memory_order_relaxed);
        g_storeProfiler.flushNs.fetch_add(ProfNs(tFlush,std::chrono::steady_clock::now()),std::memory_order_relaxed);
    } catch (...) { owner->healthy=false; throw; }
}
bool DiskPrefixCache::WriteSession::Commit(const std::string &key,const Json &value,bool keepEquivalent) {
    return CommitTracked(key, value, keepEquivalent, std::function<void(bool)>());
}
bool DiskPrefixCache::WriteSession::CommitTracked(const std::string &key,const Json &value,
        bool keepEquivalent, const CommitIdentity &identity,
        std::function<void(bool, const CommitIdentity &)> completion) {
    Check(identity.Valid(), "identity_missing");
    Json::object fields = value.object_items();
    fields["request_id"] = double(identity.requestId);
    fields["request_generation"] = double(identity.requestGeneration);
    fields["page_generation"] = double(identity.pageGeneration);
    fields["copy_generation"] = double(identity.copyGeneration);
    fields["cache_identity"] = identity.cacheIdentity;
    fields["owner_lease"] = double(identity.ownerLease);
    return CommitTracked(key, Json(fields), keepEquivalent,
        [identity, completion = std::move(completion)](bool durable) {
            if (completion) completion(durable, identity);
        });
}
bool DiskPrefixCache::WriteSession::CommitTracked(const std::string &key,const Json &value,
        bool keepEquivalent,std::function<void(bool)> completion) {
    Check(state && IsDigest(key) && value.is_object(),"invalid_checkpoint_commit");
    auto owner=state->owner;
    const std::string kind=value["kind"].is_null()?"prefix":value["kind"].string_value();
    const std::string family=value["family_identity"].is_null()?owner->family:value["family_identity"].string_value();
    Check((kind=="prefix" || kind=="encoder") && IsDigest(family),"invalid_checkpoint_metadata");
    if (owner->v5Mode) {
        { Refs validated; CollectRefs(value,validated); }   // same synchronous check as v4
        std::vector<std::pair<std::string,std::string>> staged;
        staged.swap(state->v4Chunks);
        std::string v5Error;
        if (!owner->v5->Submit(state->v5Writer,key,owner->identity,owner->family,value,std::move(staged),
                               std::move(completion),v5Error))
            Check(false,v5Error.empty()?"v5_commit_submit_failed":v5Error.c_str());
        return true;
    }
    if (owner->v4Mode) {
        // §⑥ 备菜流水线（2026-09-30）：请求线程不再做"提交前准备"（实测每检查点约
        // 251.7 ms：状态正文序列化 229.6 + 引用遍历 15.0 + 摘要 7.1；连同描述符构造/
        // 预检扫描/装对象+清单登记合计约 844.9 ms ≈ 29.8%），只把"状态 Json + 本次暂存
        // 分块"移交出去即返回"已受理"。序列化(MetadataText)、引用合法性校验（§17.1
        // raw tokens 不落盘）、摘要(Digest)、描述符、预检扫描、对象装配与 root 清单登记
        // （含只引用对象，P0 修复）全部在 store 的**唯一备菜线程**上按受理顺序完成，
        // 再投递给唯一写盘 actor；磁盘格式/记录布局/清单语义一律不变。
        // 语义：true 只代表"已受理"；持久化/失败结果在下一次 reap、读侧收尾或关服收尾
        // 处原样上报（失败绝不静默）。
        // §17.1「raw tokens 不落盘」+ 引用合法性：这是**同步**校验——调用方必须当场拿到
        // 拒绝（原始 token 数组 / 非法 blob 摘要），这也是既有测试 v4RejectsInvalidState
        // 的硬要求；实测只占约 15ms，故留在请求线程。重活（状态正文序列化 229.6ms、
        // 摘要 7.1ms、描述符构造、预检扫描、装对象 + 清单登记）才移到后台备菜线程。
        const auto refStart = std::chrono::steady_clock::now();
        { Refs validated; CollectRefs(value,validated); }
        const uint64_t refWalkNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - refStart).count());
        owner->v4->AddMetadataTiming(refWalkNs,0,0,0);
        std::vector<std::pair<std::string,std::string>> staged;
        staged.swap(state->v4Chunks);
        std::string v4Error;
        if (!owner->v4->SubmitStaged(key,owner->identity,owner->family,value,keepEquivalent,
                                     std::move(staged),v4Error,std::move(completion),
                                     std::move(state->objectLease))) {
            Check(false,v4Error.empty()?"v4_commit_submit_failed":v4Error.c_str());
        }
        return true;
    }
    Check(kind!="prefix" || (value["length"].is_number() && value["length"].int_value()>0 &&
          value["length"].number_value()==value["length"].int_value()),"invalid_checkpoint_length");
    Refs valueRefs; CollectRefs(value,valueRefs);
    state->pinsToPublish.insert(valueRefs.begin(),valueRefs.end());
    state->Flush();
    {
        Lease metadata(owner->root/".metadata",LOCK_EX); Database db(owner->dbPath);
        Statement query(db.value,"SELECT state FROM checkpoints WHERE identity=? AND key=?");
        query.Text(1,owner->identity); query.Text(2,key);
        if (query.Row()) {
            const auto existing=owner->LoadRecord(db,owner->identity,key);
            Check(existing==value || (keepEquivalent && EquivalentPrefix(existing,value)),"checkpoint_key_conflict");
            if (completion) completion(true);
            return false;
        }
    }
    const auto text=MetadataText(value);
    const auto descriptor=StoreBytes(text.size(),[&](size_t offset,char *out,size_t n) { memcpy(out,text.data()+offset,n); });
    state->Flush();
    Refs commitRefs; CollectRefs(value,commitRefs); CollectRefs(Json(descriptor),commitRefs);
    const uint64_t controlBudget=Add(2*Alignment,CatalogBudget(commitRefs.size()+16));
    const uint64_t required=Add(Add(RecordPhysicalBytes(Json(descriptor).dump().size()+2048)+ContainerHeaderBytes,controlBudget),controlBudget);
    state->EnsureBudget(required);
    Lease metadata(owner->root/".metadata",LOCK_EX); Database db(owner->dbPath);
    // Another process may have published the same immutable key meanwhile.
    Statement exists(db.value,"SELECT state FROM checkpoints WHERE identity=? AND key=?");
    exists.Text(1,owner->identity); exists.Text(2,key);
    if (exists.Row()) {
        const auto existing=owner->LoadRecord(db,owner->identity,key);
        Check(existing==value || (keepEquivalent && EquivalentPrefix(existing,value)),"checkpoint_key_conflict");
        if (completion) completion(true);
        return false;
    }
    const Json event=Json::object{{"identity",owner->identity},{"key",key},{"family",family},{"kind",kind},
        {"length",value["length"].int_value()},{"dtype",FieldString(value,"kv_dtype")},
        {"provenance",FieldString(value,"provenance")},{"created",std::to_string(NowNs())},
        {"bytes",std::to_string(text.size())},{"control_bytes",std::to_string(controlBudget)},
        {"prefix",value["prefix"].string_value()},{"state",descriptor}};
    const uint64_t before=owner->PhysicalBytes();

    const bool published=owner->Publish(db,CommitRecord,event,keepEquivalent);
    const uint64_t after=owner->PhysicalBytes();
    Check(after<=before || after-before<=required-controlBudget,"cache_commit_budget_exceeded");
    owner->Consume(db,state->id,state->remaining,required);
    if (completion) completion(true);
    return published;
}
uint64_t DiskPrefixCache::WriteSession::WrittenBytes() const { return state?state->written:0; }
Json DiskPrefixCache::ReadSession::Load(const std::string &key,const std::string &sourceIdentity,
                                      bool publishedOnly,const ReadProgress &progress) {
    return LoadInternal(key,sourceIdentity,false,publishedOnly,progress);
}
// B1：清单-only 加载。只用于「父检查点继承」（adopt），
// 那条路径只读元数据、从不取载荷（详见头文件声明处的完整说明与证据）。
Json DiskPrefixCache::ReadSession::LoadManifest(const std::string &key,const std::string &sourceIdentity) {
    return LoadInternal(key,sourceIdentity,true);
}
Json DiskPrefixCache::ReadSession::LoadInternal(const std::string &key,const std::string &sourceIdentity,
                                                bool manifestOnly,bool publishedOnly,
                                                const ReadProgress &progress) {
    Check(state && IsDigest(key),"closed_read_session");
    const auto id=sourceIdentity.empty()?state->owner->identity:sourceIdentity;
    Check(IsDigest(id),"invalid_checkpoint_identity");
    if (state->owner->v5Mode) {
        // Only durable checkpoints are indexed, so publishedOnly holds by construction.
        (void)publishedOnly;
        state->v4Objects.clear();
        state->manifestOnly = manifestOnly;
        state->v5Pin.reset();
        Json value; std::string v5Error;
        if (!state->owner->v5->Load(key,id,manifestOnly,value,&state->v4Objects,state->v5Pin,v5Error,progress))
            Check(false,v5Error.empty()?"v4_root_missing":v5Error.c_str());
        return value;
    }
    if (state->owner->v4Mode) {
        std::string leaseError;
        std::shared_ptr<void> checkpointLease =
            state->owner->v4->PinCheckpoint(key,id,leaseError,!publishedOnly);
        Check(checkpointLease != nullptr,
              leaseError.empty()?"v4_checkpoint_lease":leaseError.c_str());
        state->v4Objects.clear();
        // B1：记下本会话的加载口径，供 ReadMany 守卫使用（防止把清单当字节用）。
        state->manifestOnly = manifestOnly;
        std::string v4Error; std::vector<std::string> objects; std::vector<std::string> objectHashes;
        // S2-4：把本次请求的 identity 显式传给 Load。旧签名只按 key 取“最新 sequence”的
        // root，同一 key 属于不同 identity 时会解析到别人的 root（错误命中）。
        // A2：改用多回传摘要的重载，下面直接用它建映射，
        // 不再对同一份字节重算一遍 SHA-256（T0 实测省 1525 ms）。
        // B1：manifestOnly=true 时只取状态对象 + 块对象，
        // 跳过块引用的 KV 分片载荷（实测 463/520 个对象、566.2 MiB 在此处被白读）。
        Json assembledState;
        const auto sessionStart = std::chrono::steady_clock::now();
        if (!state->owner->v4->Load(key,id,objects,objectHashes,v4Error,manifestOnly,publishedOnly,progress,&assembledState)) {
            Check(false,v4Error.empty()?"v4_load_failed":v4Error.c_str());
            return Json();
        }
        const auto sessionStoreDone = std::chrono::steady_clock::now();
        for (size_t index = 0; index < objects.size(); ++index) {
            state->v4Objects[objectHashes[index]] = std::move(objects[index]);
        }
        // S2-4：改为按 (identity,key) 点查描述符，替代旧的“遍历全部 live 检查点、
        // 逐条 Json::parse”全表路径（r7 实测每次 lookup 约 4300 次 parse）。
        std::string metadata,v4MetaError;
        if (!state->owner->v4->Descriptor(id,key,metadata,v4MetaError,publishedOnly)) {
            Check(false,v4MetaError.empty()?"v4_root_missing":v4MetaError.c_str());
            return Json();
        }
        std::string parseError; const auto descriptor=Json::parse(metadata,parseError);
        if (!parseError.empty()) Check(false,"cache_metadata_descriptor");
        const std::string stateHash=descriptor["state"].string_value();
        const auto found=state->v4Objects.find(stateHash);
        Check(found!=state->v4Objects.end(),"missing_cache_record");
        if (found==state->v4Objects.end()) return Json();
        std::string stateError;
        auto value = Json::parse(found->second,stateError);
        if (!stateError.empty() || !value.is_object()) Check(false,"cache_state_json");
        // Route B（2026-10-02）：紧凑状态对象只带 chain_tip/block_count，不含累计 kv 数组。
        // 这里按块链取回各块对象，把它们的 kv 数组按块顺序**合成回旧形态的 state["kv"]**，
        // 使模型侧 Import / 父检查点继承路径零改动。块对象已由 Load 一并取回。
        const bool verifiedChain = assembledState.is_object() &&
            value["chain_tip"] == assembledState["chain_tip"] &&
            value["block_count"] == assembledState["block_count"];
        if (verifiedChain && assembledState["kv"].is_array()) {
            auto fields = value.object_items();
            fields["kv"] = assembledState["kv"];
            value = Json(std::move(fields));
        } else if (value["chain_tip"].is_string() && value["block_count"].is_number()) {
            std::vector<DiskPrefixV4Store::BlockRow> blocks;
            std::string chainError;
            if (!state->owner->v4->ChainBlocks(value["chain_tip"].string_value(),
                    uint64_t(value["block_count"].number_value()), blocks, chainError,publishedOnly)) {
                Check(false,chainError.empty()?"v4_block_missing":chainError.c_str());
                return Json();
            }
            Json::array kv;
            for (const auto &block : blocks) {
                const auto item=state->v4Objects.find(block.objectHash);
                Check(item!=state->v4Objects.end(),"v4_block_missing");
                if (item==state->v4Objects.end()) return Json();
                std::string blockError; const auto blockObject=Json::parse(item->second,blockError);
                if (!blockError.empty() || !blockObject.is_object() || !blockObject["kv"].is_array())
                    Check(false,"v4_block_object");
                DiskPrefixV4Store::BlockRow parsed;
                std::string rowError;
                // A successful same-chain store load already validated these
                // immutable block bytes. Other callers retain the original check.
                if (!verifiedChain && !ParseBlockRow(blockObject,parsed,rowError)) {
                    Check(false,rowError.empty()?"v4_block_object":rowError.c_str());
                    return Json();
                }
                for (const auto &bundle : blockObject["kv"].array_items()) kv.push_back(bundle);
            }
            auto fields=value.object_items();
            fields["kv"]=Json(kv);
            value=Json(fields);
        }
        // A3：这里**原来**还有两句
        //     Refs refs; CollectRefs(value,refs);
        //     for (const auto &ref:refs) state->allowed[ref.first]=ref.second;
        // 已删除。理由（逐处核对过）：`state->allowed` 的**唯一读点**在下面 v3 分支的
        // ReadMany 里（`auto allowed=state->allowed.find(hash);`），而 v4 在本行就 return 了
        // ⇒ 在 v4 下那两句是**只写不读的死代码**；且 `CollectRefs` 要对**整棵合成后的 value**
        // （含刚拼好的 448 个 bundle 的 kv 数组）做递归遍历，纯白干。
        // 删除**不改变任何可观测行为**：v4 的取数一律走 `state->v4Objects`（见 ReadMany 的 v4 分支）。
        const auto sessionDone = std::chrono::steady_clock::now();
        FastllmEvent("cache", 0, "cache_read_session_profile",
                     "mode=%s store_ms=%.3f build_ms=%.3f total_ms=%.3f",
                     manifestOnly ? "manifest" : "full",
                     std::chrono::duration<double,std::milli>(sessionStoreDone-sessionStart).count(),
                     std::chrono::duration<double,std::milli>(sessionDone-sessionStoreDone).count(),
                     std::chrono::duration<double,std::milli>(sessionDone-sessionStart).count());
        return value;
    }
    const auto value=state->owner->LoadRecord(*state->database,id,key);
    Refs refs; CollectRefs(value,refs); state->allowed.insert(refs.begin(),refs.end()); return value;
}
void DiskPrefixCache::ReadSession::ReadBytes(const Json &chunks,size_t expected,
        const std::function<void(size_t,const char *,size_t)> &write) {
    ReadMany({Request{chunks,expected,write}});
}
void DiskPrefixCache::ReadSession::ReadMany(const std::vector<Request> &inputs) {
    if (state && (state->owner->v4Mode || state->owner->v5Mode)) {
        // B1：**清单-only 会话没有载荷**，用它取字节是编程错误。
        // 当场 fail-closed 并给出直白理由，而不是落到下面 invalid_chunk_reference 那种
        // "看着像库坏了"的报错。合法用法只有一种：`Load`（全量）之后才 `ReadMany`。
        Check(!state->manifestOnly,"manifest_only_read_requires_full_load");
        for (const auto &request:inputs) {
            size_t offset=0;
            for (const auto &chunk:request.chunks.array_items()) {
                const std::string hash=chunk["sha256"].string_value();
                const uint64_t count=JsonBytes(chunk["bytes"],ChunkBytes);
                const auto found=state->v4Objects.find(hash);
                if (found==state->v4Objects.end()) {
                    // 诊断（2026-09-30）：一次打印足够判定"清单漏项"还是"对象从未落盘"：
                    //   in_object_table=0 → 该对象根本没写进对象表（写入失败/被跳过）
                    //   in_object_table=1 → 对象在库里，但本次 Load 的清单没带它（清单漏项）
                    const bool inTable = state->owner->v4 && state->owner->v4->HasObject(hash);
                    std::fprintf(stderr,
                        "[Prefix SSD] invalid_chunk_reference: hash=%s loaded_objects=%zu "
                        "request_chunks=%zu in_object_table=%d.\n",
                        hash.substr(0,16).c_str(), state->v4Objects.size(),
                        request.chunks.array_items().size(), int(inTable));
                }
                Check(found!=state->v4Objects.end(),"invalid_chunk_reference");
                if (found==state->v4Objects.end()) continue;
                Check(found->second.size()==count,"cache_object_checksum");
                request.write(offset,found->second.data(),found->second.size());
                offset+=found->second.size(); state->read+=found->second.size();
            }
            Check(offset==request.expected,"cache_chunk_coverage");
        }
        return;
    }
    Check(bool(state),"closed_read_session");
    // 读路径剖析：把 IO / 校验 分段计时，用于定位 68 MB/s 的真正瓶颈
    const auto profileStart = std::chrono::steady_clock::now();
    std::atomic<uint64_t> profileIoNs{0}, profileVerifyNs{0};
    uint64_t profileGroupBytes = 0, profilePayloadBytes = 0;
    struct Piece { size_t input,logical; std::string hash; ObjectLocation location; };
    std::vector<Piece> pieces;
    for (size_t input=0;input<inputs.size();++input) {
        const auto &request=inputs[input]; Check(request.chunks.is_array(),"invalid_chunk_manifest");
        size_t offset=0;
        for (const auto &chunk:request.chunks.array_items()) {
            const auto hash=chunk["sha256"].string_value(); const auto bytes=JsonBytes(chunk["bytes"],ChunkBytes);
            auto allowed=state->allowed.find(hash);
            Check(IsDigest(hash) && allowed!=state->allowed.end() && allowed->second==bytes &&
                  offset<=request.expected && bytes<=request.expected-offset,"invalid_chunk_reference");
            ObjectLocation location;
            Check(state->owner->FindObject(*state->database,hash,location) && location.bytes==bytes,"missing_cache_record");
            profilePayloadBytes += bytes;
            pieces.push_back({input,offset,hash,location}); offset+=bytes;
        }
        Check(offset==request.expected,"incomplete_tensor");
    }
    std::sort(pieces.begin(),pieces.end(),[](const Piece &a,const Piece &b) {
        return std::tie(a.location.container,a.location.offset,a.location.payloadOffset)<
               std::tie(b.location.container,b.location.offset,b.location.payloadOffset);
    });
    struct Group { size_t first,last; uint64_t container,offset,bytes; };
    std::vector<Group> groups;
    for (size_t first=0;first<pieces.size();) {
        size_t last=first+1;
        uint64_t end=Add(pieces[first].location.offset,pieces[first].location.physical);
        const uint64_t start=pieces[first].location.offset,container=pieces[first].location.container;
        while (last<pieces.size() && pieces[last].location.container==container) {
            const auto &next=pieces[last].location;
            const uint64_t newEnd=std::max(end,Add(next.offset,next.physical));
            if (next.offset>end+Alignment || newEnd-start>BatchBytes+Alignment) break;
            end=newEnd; ++last;
        }
        profileGroupBytes += (end-start);
        groups.push_back({first,last,container,start,end-start}); first=last;
    }
    // ── 按需读路径（默认）：只读本次真正需要的 payload 段 ──────────────────
    // 动机（实测）：整段读满足 record 级校验导致 13.4× 读放大（实读 14.08 GiB vs
    // 需要 1.05 GiB），校验累计 13.1 s 占读盘 91.6%。此路径保留 payload 级 SHA256
    // （覆盖 100% 实际使用数据）与 record 头部一致性检查，去掉"整条 record 内容"校验。
    // 设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY=1 可回到下面的严格整段读路径。
    static const bool strictVerify = [] {
        const char *value = std::getenv("FASTLLM_PREFIX_CACHE_STRICT_VERIFY");
        return value != nullptr && value[0] != 0 && value[0] != '0';
    }();
    if (!strictVerify) {
        for (size_t first=0;first<pieces.size();) {
            size_t last=first;
            while (last<pieces.size() && last-first<size_t(state->owner->ioThreads)) ++last;
            // 并行读 payload + 头部一致性 + payload 哈希；写回在主线程串行（与原路径语义一致）
            std::vector<std::future<std::vector<char>>> reads;
            for (size_t i=first;i<last;++i) {
                const Piece piece=pieces[i];
                const std::string containerPath=state->owner->ContainerPath(piece.location.container);
                reads.push_back(std::async(std::launch::async,
                    [piece,containerPath,&profileIoNs,&profileVerifyNs] {
                    char header[RecordHeaderBytes];
                    std::vector<char> payload(static_cast<size_t>(piece.location.bytes));
                    const auto io0=std::chrono::steady_clock::now();
                    int fd=open(containerPath.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
                    Check(fd>=0,"cache_container_open");
                    try {
                        ReadExact(fd,header,RecordHeaderBytes,piece.location.offset);
                        ReadExact(fd,payload.data(),payload.size(),
                                  piece.location.offset+piece.location.payloadOffset);
                        close(fd);
                    } catch (...) { close(fd); throw; }
                    profileIoNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-io0).count();
                    const auto verify0=std::chrono::steady_clock::now();
                    const auto parsed=ParseHeader(header);
                    Check(parsed.kind==PackedRecord && parsed.physical==piece.location.physical,
                          "cache_record_header");
                    Check(DiskPrefixCache::Digest(payload.data(),payload.size())==piece.hash,
                          "cache_object_checksum");
                    profileVerifyNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-verify0).count();
                    return payload;
                }));
            }
            for (size_t i=first;i<last;++i) {
                const Piece &piece=pieces[i];
                const std::vector<char> payload=reads[i-first].get();
                inputs[piece.input].write(piece.logical,payload.data(),payload.size());
                state->read+=payload.size();
            }
            first=last;
        }
        const double wallMs=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-profileStart).count();
        PrefixCacheDebug("[Prefix SSD] read-profile(ondemand): pieces=%zu groups=%zu payload_bytes=%.1fMiB "
               "io_ms=%.1f verify_ms=%.1f wall_ms=%.1f payload_MBps=%.1f.\n",
               pieces.size(),groups.size(),profilePayloadBytes/1048576.0,
               profileIoNs.load()/1e6,profileVerifyNs.load()/1e6,wallMs,
               wallMs>0?profilePayloadBytes/1e6/(wallMs/1000):0.0);
        return;
    }
    for (size_t first=0;first<groups.size();) {
        size_t last=first; uint64_t memory=0;
        while (last<groups.size() && last-first<size_t(state->owner->ioThreads) &&
               (last==first || (memory<=BatchBytes && groups[last].bytes<=BatchBytes-memory))) {
            memory+=groups[last].bytes; ++last;
        }
        std::vector<std::future<std::shared_ptr<State::CachedGroup>>> tasks;
        for (size_t i=first;i<last;++i) {
            const Group group=groups[i]; auto owner=state->owner;
            std::shared_ptr<State::CachedGroup> cached;
            for (const auto &entry:state->cachedGroups)
                if (entry->container==group.container && entry->offset==group.offset && entry->bytes==group.bytes) {
                    cached=entry; break;
                }
            tasks.push_back(std::async(std::launch::async,[owner,group,cached,&pieces,&profileIoNs,&profileVerifyNs] {
                Check(group.bytes<=BatchBytes+Alignment,"cache_read_group_bound");
                auto entry=cached?cached:std::make_shared<State::CachedGroup>();
                if (!cached) {
                    entry->container=group.container; entry->offset=group.offset; entry->bytes=group.bytes;
                    entry->data.resize(static_cast<size_t>(group.bytes));
                    int fd=open(owner->ContainerPath(group.container).c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
                    Check(fd>=0,"cache_container_open");
                    const auto io0=std::chrono::steady_clock::now();
                    try { ReadExact(fd,entry->data.data(),entry->data.size(),group.offset); close(fd); }
                    catch (...) { close(fd); throw; }
                    profileIoNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-io0).count();
                }
                const auto &data=entry->data;
                auto verified=entry->verifiedRecords;
                const auto verify0=std::chrono::steady_clock::now();
                for (size_t j=group.first;j<group.last;++j) {
                    const auto &piece=pieces[j]; const auto &loc=piece.location;
                    const uint64_t relative=loc.offset-group.offset;
                    Check(relative<=data.size() && loc.physical<=data.size()-relative,"cache_read_bounds");
                    const char *record=data.data()+relative;
                    const auto header=ParseHeader(record);
                    Check(header.kind==PackedRecord && header.physical==loc.physical,"cache_record_checksum");
                    if (!verified.count(relative)) {
                        Check(DiskPrefixCache::Digest(record+RecordHeaderBytes,header.bytes)==header.hash,"cache_record_checksum");
                        verified.insert(relative);
                    }
                    Check(loc.payloadOffset>=RecordHeaderBytes && loc.payloadOffset<=loc.physical &&
                          loc.bytes<=loc.physical-loc.payloadOffset &&
                          DiskPrefixCache::Digest(record+loc.payloadOffset,loc.bytes)==piece.hash,"cache_object_checksum");
                }
                profileVerifyNs += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-verify0).count();
                // A group can contain records not selected by the first call.
                // Only publish offsets whose full record checksum was checked.
                entry->verifiedRecords=std::move(verified);
                return entry;
            }));
        }
        // Verification workers never call model/CUDA callbacks concurrently.
        for (size_t i=first;i<last;++i) {
            const auto entry=tasks[i-first].get(); const auto &group=groups[i];
            if (std::find(state->cachedGroups.begin(),state->cachedGroups.end(),entry)==state->cachedGroups.end()) {
                // 组缓存上限可配（默认 4，与原先一致）；FASTLLM_PREFIX_CACHE_GROUP_CACHE 可调大
                static const size_t groupCacheLimit = [] {
                    const char *value=std::getenv("FASTLLM_PREFIX_CACHE_GROUP_CACHE");
                    const int parsed=value!=nullptr?std::atoi(value):0;
                    return size_t(parsed>0?parsed:4);
                }();
                if (state->cachedGroups.size()>=groupCacheLimit) state->cachedGroups.erase(state->cachedGroups.begin());
                state->cachedGroups.push_back(entry);
            }
            const auto &data=entry->data;
            for (size_t j=group.first;j<group.last;++j) {
                const auto &piece=pieces[j]; const auto &loc=piece.location;
                inputs[piece.input].write(piece.logical,data.data()+loc.offset-group.offset+loc.payloadOffset,size_t(loc.bytes));
                state->read+=loc.bytes;
            }
        }
        first=last;
    }
    {
        const double wallMs = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-profileStart).count();
        PrefixCacheDebug("[Prefix SSD] read-profile: pieces=%zu groups=%zu group_bytes=%.1fMiB payload_bytes=%.1fMiB "
               "io_ms=%.1f verify_ms=%.1f wall_ms=%.1f payload_MBps=%.1f.\n",
               pieces.size(),groups.size(),profileGroupBytes/1048576.0,profilePayloadBytes/1048576.0,
               profileIoNs.load()/1e6,profileVerifyNs.load()/1e6,wallMs,
               wallMs>0?profilePayloadBytes/1e6/(wallMs/1000):0.0);
    }
}
uint64_t DiskPrefixCache::ReadSession::ReadBytesCount() const { return state?state->read:0; }
uint64_t DiskPrefixCache::ReadSession::LoadedBytesCount() const {
    uint64_t bytes = 0;
    if (state) for (const auto &object : state->v4Objects) bytes += object.second.size();
    return bytes;
}

std::vector<int> DiskPrefixCache::CheckpointLengths(const std::string &family,int maximum,const std::string &kind,bool publishedOnly) const {
    Check(IsDigest(family) && maximum>=0 && (kind=="prefix" || kind=="encoder"),"invalid_checkpoint_query");
    if (impl->v5Mode) return impl->v5->CheckpointLengths(family,maximum,kind);
    if (impl->v4Mode) {
        // S2-4：v4 此前走 ListCheckpoints 的“全表遍历 + 逐条 Json::parse”。改为与 v3
        // 同款的 SQL 过滤（DISTINCT length + kind/family 索引 + JOIN roots 限定 live）。
        return impl->v4->CheckpointLengths(family,maximum,kind,publishedOnly);
    }
    Lease lease(impl->root/".lease",LOCK_SH); Database db(impl->dbPath);
    Statement query(db.value,"SELECT DISTINCT length FROM checkpoints WHERE family=? AND kind=? AND length<=? ORDER BY length DESC");
    query.Text(1,family); query.Text(2,kind); query.Number(3,maximum);
    std::vector<int> result; while (query.Row()) result.push_back(int(query.Number(0))); return result;
}
std::vector<DiskPrefixCache::Checkpoint> DiskPrefixCache::ListCheckpoints(const std::string &exact,
        const std::string &family,int maximum,const std::string &kind,const std::string &prefix) const {
    Check((exact.empty() || IsDigest(exact)) && (family.empty() || IsDigest(family)) && maximum>=0 &&
          (kind=="prefix" || kind=="encoder") && (prefix.empty() || IsDigest(prefix)),"invalid_checkpoint_query");
    if (impl->v5Mode) {
        std::vector<Checkpoint> result;
        for (const auto &row:impl->v5->ListCheckpoints(exact,family,maximum,kind,prefix)) {
            Checkpoint value;
            value.key=row.key; value.identity=row.identity;
            value.familyIdentity=row.familyIdentity; value.kind=row.kind;
            value.length=row.length; value.kvDtype=row.kvDtype;
            value.provenance=row.provenance; value.createdNs=row.createdNs;
            std::string metadataError;
            value.metadata=Json::parse(row.metadata,metadataError);
            result.push_back(std::move(value));
        }
        return result;
    }
    if (impl->v4Mode) {
        // S2-4：v4 分支照 v3 模板改走 SQL（描述符的 family/kind/length/dtype/provenance/
        // created/prefix 已投影成列），不再全表遍历 + 逐条 Json::parse。过滤与排序语义与
        // 旧实现逐项等价（exact/family/prefix 可选过滤、length 降序 + created 升序）。
        std::vector<Checkpoint> result;
        for (const auto &row:impl->v4->ListCheckpoints(exact,family,maximum,kind,prefix)) {
            Checkpoint value;
            value.key=row.key; value.identity=row.identity;
            value.familyIdentity=row.familyIdentity; value.kind=row.kind;
            value.length=row.length; value.kvDtype=row.kvDtype;
            value.provenance=row.provenance; value.createdNs=row.createdNs;
            // S5（2026-10-05）：把描述符原文带出来，供 Lookup 侧判定"该行有没有状态"
            // （判据见 Qwen35PersistentCache::Prepare 的"无状态行"过滤）。v4 的
            // CheckpointRow 本来就已经从 SQL 取回了 metadata，这里只是解析一次，零额外 I/O。
            // 解析失败 ⇒ metadata 保持 null，调用方按"无法判定"处理（不误跳过）。
            std::string metadataError;
            value.metadata=Json::parse(row.metadata,metadataError);
            result.push_back(std::move(value));
        }
        return result;
    }
    Lease lease(impl->root/".lease",LOCK_SH); Database db(impl->dbPath);
    std::string sql="SELECT key,identity,family,kind,length,dtype,provenance,created FROM checkpoints WHERE kind=? AND length<=?";
    if (!exact.empty()) sql+=" AND identity=?";
    if (!family.empty()) sql+=" AND family=?";
    if (!prefix.empty()) sql+=" AND prefix=?";
    sql+=" ORDER BY length DESC,created ASC";
    Statement query(db.value,sql.c_str()); query.Text(1,kind); query.Number(2,maximum);
    int parameter=3;
    if (!exact.empty()) query.Text(parameter++,exact);
    if (!family.empty()) query.Text(parameter++,family);
    if (!prefix.empty()) query.Text(parameter++,prefix);
    std::vector<Checkpoint> result;
    while (query.Row()) {
        Checkpoint value; value.key=query.Text(0); value.identity=query.Text(1); value.familyIdentity=query.Text(2);
        value.kind=query.Text(3); value.length=int(query.Number(4)); value.kvDtype=query.Text(5);
        value.provenance=query.Text(6); value.createdNs=query.Number(7); result.push_back(std::move(value));
    }
    return result;
}
std::vector<DiskPrefixCache::Checkpoint> DiskPrefixCache::MatchPrefixes(
        const std::string &family,const std::vector<std::pair<int,std::string> > &prefixes,
        const std::string &kind,bool publishedOnly) const {
    // 【业界式「L3 实时问一次」】见头文件说明。
    // 语义与"逐长度 ListCheckpoints"**逐条等价**，由 T0 探针逐条对拍验证。
    Check(IsDigest(family) && (kind=="prefix" || kind=="encoder"),"invalid_checkpoint_query");
    for (size_t i = 0; i < prefixes.size(); ++i)
        Check(prefixes[i].first > 0 && IsDigest(prefixes[i].second),"invalid_checkpoint_query");
    if (prefixes.empty()) return std::vector<Checkpoint>();
    if (impl->v5Mode) {
        std::vector<Checkpoint> result;
        for (const auto &row:impl->v5->MatchPrefixes(family,prefixes,kind)) {
            Checkpoint value;
            value.key=row.key; value.identity=row.identity;
            value.familyIdentity=row.familyIdentity; value.kind=row.kind;
            value.length=row.length; value.kvDtype=row.kvDtype;
            value.provenance=row.provenance; value.createdNs=row.createdNs;
            std::string metadataError;
            value.metadata=Json::parse(row.metadata,metadataError);
            result.push_back(std::move(value));
        }
        return result;
    }
    if (impl->v4Mode) {
        std::vector<Checkpoint> result;
        for (const auto &row:impl->v4->MatchPrefixes(family,prefixes,kind,publishedOnly)) {
            Checkpoint value;
            value.key=row.key; value.identity=row.identity;
            value.familyIdentity=row.familyIdentity; value.kind=row.kind;
            value.length=row.length; value.kvDtype=row.kvDtype;
            value.provenance=row.provenance; value.createdNs=row.createdNs;
            // 与 ListCheckpoints 同口径：把描述符原文带出来（解析失败 ⇒ metadata 为 null）
            std::string metadataError;
            value.metadata=Json::parse(row.metadata,metadataError);
            result.push_back(std::move(value));
        }
        return result;
    }
    Lease lease(impl->root/".lease",LOCK_SH); Database db(impl->dbPath);
    std::string sql="SELECT key,identity,family,kind,length,dtype,provenance,created,prefix FROM checkpoints "
                    "WHERE kind=? AND family=? AND prefix IN (";
    for (size_t i = 0; i < prefixes.size(); ++i) sql += (i==0?"?":",?");
    sql+=") ORDER BY length DESC,created ASC";
    std::set<std::pair<int,std::string> > wanted;
    for (size_t i = 0; i < prefixes.size(); ++i) wanted.insert(prefixes[i]);
    Statement query(db.value,sql.c_str());
    int parameter=1;
    query.Text(parameter++,kind); query.Text(parameter++,family);
    for (size_t i = 0; i < prefixes.size(); ++i) query.Text(parameter++,prefixes[i].second);
    std::vector<Checkpoint> result;
    while (query.Row()) {
        Checkpoint value; value.key=query.Text(0); value.identity=query.Text(1); value.familyIdentity=query.Text(2);
        value.kind=query.Text(3); value.length=int(query.Number(4)); value.kvDtype=query.Text(5);
        value.provenance=query.Text(6); value.createdNs=query.Number(7);
        if (wanted.find(std::make_pair(value.length,query.Text(8)))==wanted.end()) continue;   // 精确对过滤
        result.push_back(std::move(value));
    }
    return result;
}
void DiskPrefixCache::SelfCheckStorage() {
    if (impl->v5Mode) { std::string details; impl->v5->SelfCheck(details); return; }
    if (impl->v4Mode && impl->v4) impl->v4->SelfCheck();
}

int DiskPrefixCache::JunctionLength(const std::vector<std::string> &pageIdentities, int pageLen) const {
    if (!impl->v5Mode) return 0;
    return impl->v5->JunctionPages(impl->identity, pageIdentities) * pageLen;
}
void DiskPrefixCache::RebuildIndex() {
    if (impl->v5Mode) return;   // V5 index is derived from the files at start-up (draft3 sec. 7)
    if (impl->v4Mode) {
        std::string error;
        Check(impl->v4->RebuildIndex(error),error.empty()?"v4_rebuild_failed":error.c_str());
        return;
    }
    Lease writers(impl->root/".writers",LOCK_EX);
    Lease lease(impl->root/".lease",LOCK_EX);
    { Lease metadata(impl->root/".metadata",LOCK_EX); Database db(impl->dbPath); impl->CheckCurrentConfig(db); }
    impl->Recover(true);
    { Lease metadata(impl->root/".metadata",LOCK_EX); Database db(impl->dbPath);
      Impl::SetMeta(db,"configured_limit",std::to_string(impl->limit));
      Impl::SetMeta(db,"configured_file_limit",std::to_string(impl->fileLimit)); }
}
fs::path DiskPrefixCache::Directory() const { if (impl->v5Mode) return fs::path(impl->v5->Root()); return impl->v4Mode?fs::path(impl->v4->Root()):impl->base; }
void DiskPrefixCache::SetCheckpointEraseCallback(
        std::function<void(const std::string &, const std::string &)> callback) {
    if (impl->v5Mode) { impl->v5->SetEraseObserver(std::move(callback)); return; }
    if (!impl->v4Mode || !impl->v4) {
        impl->eraseObserver.reset();
        return;
    }
    impl->eraseObserver=callback
        ? impl->v4->ObserveCheckpointErases(std::move(callback)) : nullptr;
}
void DiskPrefixCache::SetCheckpointReclaimPredicate(CheckpointReclaimPredicate predicate) {
    if (impl->v5Mode) { impl->v5->SetReclaimPredicate(std::move(predicate)); return; }
    if (!impl->v4Mode || !impl->v4) {
        impl->reclaimPredicate.reset();
        return;
    }
    impl->reclaimPredicate=predicate
        ? impl->v4->ObserveCheckpointReclaims(std::move(predicate)) : nullptr;
}
void DiskPrefixCache::SetCheckpointCorruptRetirePredicate(CheckpointCorruptRetirePredicate predicate) {
    if (impl->v5Mode) { impl->v5->SetCorruptRetirePredicate(std::move(predicate)); return; }
    if (!impl->v4Mode || !impl->v4) {
        impl->corruptRetirePredicate.reset();
        return;
    }
    impl->corruptRetirePredicate = predicate
        ? impl->v4->ObserveCheckpointCorruptRetires(std::move(predicate)) : nullptr;
}
bool DiskPrefixCache::RetireCorruptCheckpoint(const std::string &key, const std::string &identity,
                                               const std::string &confirmedReason, std::string &error) {
    if (impl->v5Mode) return impl->v5->RetireCorruptCheckpoint(key, identity, confirmedReason, error);
    if (!impl->v4Mode || !impl->v4) { error = "v4_mode_required"; return false; }
    return impl->v4->RetireCorruptCheckpoint(key, identity, confirmedReason, error);
}
void DiskPrefixCache::PutMetadata(const std::string &key,const Json &value) {
    Check(!key.empty() && key.size()<=1024 && value.is_object(),"invalid_cache_metadata");
    if (impl->v5Mode) {
        const auto text=MetadataText(value); std::string error;
        Check(impl->v5->PutMetadata(key,text,error),error.empty()?"v4_metadata_failed":error.c_str());
        return;
    }
    if (impl->v4Mode) {
        const auto text=MetadataText(value); std::string error;
        Check(impl->v4->PutMetadata(key,text,error),error.empty()?"v4_metadata_failed":error.c_str());
        return;
    }
    const auto text=MetadataText(value);
    auto writer=BeginWrite(text.size());
    const auto chunks=writer.StoreBytes(text.size(),[&](size_t offset,char *out,size_t bytes) { memcpy(out,text.data()+offset,bytes); });
    writer.state->Flush();
    const Json event=Json::object{{"identity","metadata"},{"key",key},{"state",chunks}};
    const uint64_t charge=Add(RecordPhysicalBytes(event.dump().size())+ContainerHeaderBytes,CatalogBudget(chunks.size()+16));
    writer.state->EnsureBudget(charge);
    Lease metadata(impl->root/".metadata",LOCK_EX); Database db(impl->dbPath);
    const auto before=impl->PhysicalBytes();
    impl->Publish(db,MetadataRecord,event);
    const auto after=impl->PhysicalBytes();
    Check(after<=before || after-before<=charge,"cache_metadata_budget_exceeded");
    impl->Consume(db,writer.state->id,writer.state->remaining,charge);
}
Json DiskPrefixCache::GetMetadata(const std::string &key) const {
    if (impl->v5Mode) {
        std::string text,error;
        if (!impl->v5->GetMetadata(key,text,error)) return Json();
        std::string parseError; const auto parsed=Json::parse(text,parseError);
        Check(parseError.empty(),"cache_metadata_descriptor"); return parsed;
    }
    if (impl->v4Mode) {
        std::string text,error;
        if (!impl->v4->GetMetadata(key,text,error)) return Json();
        std::string parseError; const auto parsed=Json::parse(text,parseError);
        Check(parseError.empty(),"cache_metadata_descriptor"); return parsed;
    }
    Lease lease(impl->root/".lease",LOCK_SH); Database db(impl->dbPath);
    Statement query(db.value,"SELECT state FROM metadata WHERE key=?"); query.Text(1,key);
    if (!query.Row()) return Json();
    std::string error; const auto descriptor=Json::parse(query.Text(0),error);
    Check(error.empty(),"cache_metadata_descriptor"); return impl->ParseState(db,descriptor);
}
Json DiskPrefixCache::CapacityObservation() const {
    if (impl->v5Mode) {
        const auto capacity = impl->v5->Capacity();
        return Json::object{{"state", capacity["phase"].string_value() == "ready" ? "ready" : "capacity_unavailable"},
                            {"limit_bytes", capacity["limit_bytes"]}, {"physical_bytes", capacity["physical_bytes"]},
                            {"inflight_bytes", capacity["inflight_bytes"]}, {"accounting", "v5_physical_blocks"}};
    }
    if (!impl->v4Mode) return Json::object{{"state", "unsupported"}};
    uint64_t physical = 0;
    std::string error;
    if (!impl->v4->ObservePhysicalBytes(physical, error))
        return Json::object{{"state", "stat_failed"}, {"error", error}};
    return Json::object{
        {"state", physical > impl->limit ? "capacity_unavailable" : "ready"},
        {"limit_bytes", std::to_string(impl->limit)},
        {"physical_bytes", std::to_string(physical)},
        {"inflight_bytes", std::to_string(impl->v4->InFlightBytes())},
        {"accounting", "physical_control_log"}};
}

Json DiskPrefixCache::CapacityStatus() const {
    if (impl->v5Mode) return impl->v5->Capacity();
    if (impl->v4Mode) {
        // S2-5a：physical_bytes 必须是 control.log 的**真实物理占用**（旧实现把 live 当
        // physical 上报，掩盖了日志里已淘汰/未引用对象的垃圾）。同时新增 live_bytes
        // （沿用原 LiveBytes() 值）与 garbage_bytes（physical−live，负数取 0）。
        // 写路径同时检查 live admission 与真实 control.log physical bytes；
        // physical 超限会请求后台 Compact，并向当前请求返回 capacity_unavailable。
        const uint64_t live=impl->v4->LiveBytes();
        const uint64_t physical=impl->v4->PhysicalBytes();
        return Json::object{{"limit_bytes",std::to_string(impl->limit)},
            {"file_bytes","0"},{"physical_bytes",std::to_string(physical)},
            {"live_bytes",std::to_string(live)},
            {"garbage_bytes",std::to_string(physical>live?physical-live:0)},
            {"reserved_bytes","0"},{"shrink_debt","0"},
            {"phase",physical>impl->limit?"capacity_unavailable":(live>impl->limit?"shrinking":"ready")},
            {"policy","fifo-new-write-first"},
            {"storage_format","4"},{"live_checkpoints",std::to_string(impl->v4->LiveCheckpoints())}};
    }
    Lease lease(impl->root/".lease",LOCK_SH); Database db(impl->dbPath);
    const auto physical=impl->PhysicalBytes(),reserved=Add(db.Scalar("SELECT COALESCE(SUM(bytes),0) FROM reservations"),
        Add(db.Scalar("SELECT COALESCE(SUM(control_bytes),0) FROM checkpoints"),
                       db.Scalar("SELECT COALESCE(SUM(credit),0) FROM events")));
    const auto used=Add(physical,reserved);
    return Json::object{{"limit_bytes",std::to_string(impl->limit)},{"file_bytes",std::to_string(impl->fileLimit)},
        {"physical_bytes",std::to_string(physical)},{"reserved_bytes",std::to_string(reserved)},
        {"shrink_debt",std::to_string(used>impl->limit?used-impl->limit:0)},
        {"phase",used>impl->limit?"shrinking":"ready"},{"policy","fifo-new-write-first"},
        {"logical_write_bytes",db.ScalarText("SELECT value FROM meta WHERE key='logical_write_bytes'")},
        {"physical_write_bytes",db.ScalarText("SELECT value FROM meta WHERE key='physical_write_bytes'")},
        {"compaction_bytes",db.ScalarText("SELECT value FROM meta WHERE key='compaction_bytes'")}};
}
void DiskPrefixCache::SetStartupManifest(const std::string &manifest) { startupManifest=manifest; }
std::string DiskPrefixCache::StartupManifest() { return startupManifest; }

// ===========================================================================
// DiskPrefixV4Store（SSD_CACHE_REWRITE_PLAN.md §10/§11）
// ===========================================================================
namespace {
// v3 already owns the names RecordHeader/CommitRecord/EraseRecord in this
// translation unit's anonymous namespace, so v4 symbols are imported
// selectively and those three are always qualified.
using fastllm::prefix_v4::AlignUp;
using fastllm::prefix_v4::AppendRecordToFile;
using fastllm::prefix_v4::DecodeCommit;
using fastllm::prefix_v4::DecodeErase;
using fastllm::prefix_v4::DecodeMetadata;
using fastllm::prefix_v4::DecodeObjectBatchPayload;
using fastllm::prefix_v4::DecodeObjectBatchPayloadView;
using fastllm::prefix_v4::DecodeObjectList;
using fastllm::prefix_v4::DecodePrepare;
using fastllm::prefix_v4::EncodeCommit;
using fastllm::prefix_v4::EncodeErase;
using fastllm::prefix_v4::EncodeMetadata;
using fastllm::prefix_v4::EncodeObjectBatchPayload;
using fastllm::prefix_v4::EncodeObjectList;
using fastllm::prefix_v4::EncodePrepare;
using fastllm::prefix_v4::LogClassification;
using fastllm::prefix_v4::ObjectEntry;
using fastllm::prefix_v4::ObjectListEntry;
using fastllm::prefix_v4::PrepareRecord;
using fastllm::prefix_v4::RecordDigest;
using fastllm::prefix_v4::ScanControlLog;
using fastllm::prefix_v4::ScannedRecord;
using fastllm::prefix_v4::kFormatText;
using fastllm::prefix_v4::kObjectData;
using fastllm::prefix_v4::kObjectKvNode;
using fastllm::prefix_v4::kRecordErase;
using fastllm::prefix_v4::kRecordHeaderBytes;
using fastllm::prefix_v4::kRecordManifestCommit;
using fastllm::prefix_v4::kRecordMetadata;
using fastllm::prefix_v4::kRecordManifestPrepare;
using fastllm::prefix_v4::kRecordObjectBatch;

// Physical SSD test mode: unsupported direct I/O fails without buffered fallback.
struct V4ReadSplit {
    std::atomic<uint64_t> diskNs{0}, digestNs{0}, extractNs{0};
    std::atomic<uint64_t> records{0}, recordBytes{0}, extractCalls{0};
    std::atomic<uint64_t> directReadCalls{0}, directReadBytes{0};
    std::atomic<int> path{-1};
};
inline V4ReadSplit &V4Split() { static V4ReadSplit split; return split; }

bool V4DirectIOEnabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("FASTLLM_PREFIX_CACHE_DIRECT_IO");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}

// Read into an aligned interior of one owning string, then compact the output
// slice in place. Header/padding checks and on-disk format remain unchanged.
std::string V4ReadDirectRange(const fs::path &path, uint64_t offset, size_t bytes,
                             size_t outputSkip, size_t outputBytes, size_t *actualBytes) {
    constexpr size_t alignment = size_t(prefix_v4::kAlignment);
    static_assert((alignment & (alignment - 1)) == 0, "direct alignment must be a power of two");
    Check(outputSkip <= bytes && outputBytes <= bytes - outputSkip, "v4_direct_io_bounds");
    if (actualBytes != nullptr) *actualBytes = 0;
    if (bytes == 0) return {};
    const size_t leading = size_t(offset % alignment);
    const uint64_t alignedOffset = offset - leading;
    Check(bytes <= std::numeric_limits<size_t>::max() - leading - (alignment - 1),
          "v4_direct_io_bounds");
    const size_t span = (leading + bytes + alignment - 1) & ~(alignment - 1);
    Check(span <= std::numeric_limits<size_t>::max() - (alignment - 1) &&
          span <= size_t(std::numeric_limits<ssize_t>::max()) &&
          alignedOffset <= uint64_t(std::numeric_limits<off_t>::max()) &&
          uint64_t(span) <= uint64_t(std::numeric_limits<off_t>::max()) - alignedOffset,
          "v4_direct_io_bounds");
#ifdef O_DIRECT
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#else
    throw std::runtime_error("v4_direct_io_unsupported");
    const int fd = -1;
#endif
    Check(fd >= 0, "v4_log_open");
    struct CloseFd { int fd; ~CloseFd() { close(fd); } } closeFd{fd};
    std::string storage(span + alignment - 1, '\0');
    const size_t padding = (alignment - (reinterpret_cast<uintptr_t>(storage.data()) % alignment)) % alignment;
    char *aligned = storage.data() + padding;
    size_t done = 0;
    while (done < span) {
        const ssize_t got = pread(fd, aligned + done, span - done, off_t(alignedOffset + done));
        V4Split().directReadCalls.fetch_add(1, std::memory_order_relaxed);
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("v4_log_read");
        }
        if (got == 0) break;
        V4Split().directReadBytes.fetch_add(uint64_t(got), std::memory_order_relaxed);
        done += size_t(got);
        // Nonaligned EOF data cannot be the next direct-I/O buffer/offset.
        if (done % alignment != 0) break;
    }
    const size_t available = done > leading ? std::min(bytes, done - leading) : 0;
    if (actualBytes != nullptr) *actualBytes = available;
    const size_t take = available > outputSkip ? std::min(outputBytes, available - outputSkip) : 0;
    if (take != 0) std::memmove(storage.data(), aligned + leading + outputSkip, take);
    storage.resize(take);
    return storage;
}

std::string V4ReadRange(const fs::path &path, uint64_t offset, size_t bytes) {
    if (V4DirectIOEnabled())
        return V4ReadDirectRange(path, offset, bytes, 0, bytes, nullptr);
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    Check(fd >= 0, "v4_log_open");
    std::string out(bytes, '\0');
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, out.data() + done, bytes - done, off_t(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            close(fd);
            throw std::runtime_error("v4_log_read");
        }
        if (got == 0) break;
        done += size_t(got);
    }
    close(fd);
    out.resize(done);
    return out;
}

// Read the complete physical record directly into its final payload buffer.
// Header and alignment bytes are still consumed, including short-tail checks.
// The validated header bounds both ignored spans to fixed stack storage.
bool V4ReadRecordPayloadRange(const fs::path &path, uint64_t offset,
                             const prefix_v4::RecordHeader &header,
                             std::string &payload, std::string &error) {
    if (V4DirectIOEnabled()) {
        size_t actualBytes = 0;
        std::string fresh = V4ReadDirectRange(path, offset, size_t(header.physicalBytes),
                                             size_t(kRecordHeaderBytes), size_t(header.payloadBytes),
                                             &actualBytes);
        if (actualBytes < header.physicalBytes) { error = "v4_log_short_read"; return false; }
        payload = std::move(fresh);
        return true;
    }
    std::string fresh(size_t(header.payloadBytes), '\0');
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    Check(fd >= 0, "v4_log_open");
    struct CloseFd { int fd; ~CloseFd() { close(fd); } } closeFd{fd};
    char repeatedHeader[size_t(kRecordHeaderBytes)];
    char alignmentTail[size_t(prefix_v4::kAlignment)];
    iovec spans[3] = {
        {repeatedHeader, size_t(kRecordHeaderBytes)},
        {fresh.data(), fresh.size()},
        {alignmentTail, size_t(header.physicalBytes - kRecordHeaderBytes - header.payloadBytes)}
    };
    size_t done = 0;
    int first = 0;
    while (done < header.physicalBytes) {
        const ssize_t got = preadv(fd, spans + first, 3 - first, off_t(offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("v4_log_read");
        }
        if (got == 0) { error = "v4_log_short_read"; return false; }
        done += size_t(got);
        size_t consumed = size_t(got);
        while (consumed != 0) {
            const size_t take = std::min(consumed, spans[first].iov_len);
            spans[first].iov_base = static_cast<char *>(spans[first].iov_base) + take;
            spans[first].iov_len -= take;
            consumed -= take;
            if (spans[first].iov_len == 0) ++first;
        }
    }
    payload = std::move(fresh);
    return true;
}

// Reads one record and returns its payload, verifying header and digest.
// 读路径分段计数（2026-10-18，B21）：把 "pread 等待" 与 "记录/对象 SHA-256" 分开，
// 用来回答 read_ms 里到底多少是盘、多少是哈希。只累加计时，不改变任何读写语义。
bool V4ReadRecord(const fs::path &path, uint64_t offset, prefix_v4::RecordHeader &header,
                  std::string &payload, std::string &error) {
    const auto ioStart = std::chrono::steady_clock::now();
    const std::string head = V4ReadRange(path, offset, size_t(kRecordHeaderBytes));
    if (head.size() < kRecordHeaderBytes) { error = "v4_log_short_read"; return false; }
    if (!DecodeRecordHeader(head.data(), head.size(), header, error)) return false;
    if (!V4ReadRecordPayloadRange(path, offset, header, payload, error)) return false;
    const auto ioEnd = std::chrono::steady_clock::now();
    const auto digestStart = std::chrono::steady_clock::now();
    const bool digestOk = RecordDigest(header.type, payload) == header.digest;
    const auto digestEnd = std::chrono::steady_clock::now();
    auto &split = V4Split();
    split.diskNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(ioEnd - ioStart).count());
    split.digestNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(digestEnd - digestStart).count());
    split.records += 1;
    split.recordBytes += header.payloadBytes;
    if (!digestOk) {
        error = "v4_control_record_digest";
        return false;
    }
    return true;
}

// 记录级读缓存：根因修复。
// 事实：Impl::ObjectBytes() 每查一个对象就调一次 V4ReadRecord()，而后者会
//   ①open()+pread() 读**整条**记录（OBJECT_BATCH 记录上限 kMaxPackedPayloadBytes = 16 MiB）
//   ②对整条 payload 重算 RecordDigest（SHA-256）校验
// 而 Load() 是**按对象逐个**查 hash 的，于是同一条 16 MiB 记录会被反复读、反复哈希。
// 实测指纹（trace-stall-a2 重启自检）：单线程 ~105% CPU、纯用户态；perf 命中
// sha256_block_data_order_avx2 19% / memcpy 9%；/proc/<pid>/io 的 rchar ≈ 1.17 GB/s 且
// read_bytes 不变；58 个检查点累计 rchar ≈ 220 GB，而库实际只有 ~11 GB。
// 修法：以 (路径, 记录偏移) 为键缓存
// **最近 kV4RecordCacheSlots 条**已校验通过的记录，命中即复用（含已校验结论）。
// 为什么必须多槽（T0 实测，2026-10-05）：Load() 按对象逐个查 hash，而一个检查点的对象分布在
// ~82 条记录里（记录均长 7.52 MB）；单槽对「跨记录交替取数」几乎全 miss —— 520 个对象触发
// **467 次整记录读**、rchar **3351 MiB**，而唯一记录只有 **606 MiB**（放大 5.92×）。
// 扩为 16 槽后，只要工作集 ≤16 条记录，每条记录就只被读一次。语义完全不变。
// 安全性：v4 日志是 append-only，旧记录的字节永不被改写（ERASE 只改目录投影），因此
// 同偏移的内容恒定；用 thread_local 免加锁、无跨线程竞态。
// 生命周期：返回的指针指向线程本地缓存，在**同线程下一次调用本函数之前**有效；
// ObjectBytes 取到后当次用完即弃、不再进入本函数（不递归），故零拷贝是安全的。
//  记录级读缓存的**失效机制**。
// 原实现（1512e400）只用 (路径, 偏移) 当键，并假定"v4 日志 append-only ⇒ 同偏移内容恒定"。
// 该假定在**同一路径上的内容被改写**时不成立：Compact 是用 rename 把新文件盖到**同一个路径**
// 上（EXPERIENCE #406：Compact 在生产路径目前无调用者），外部工具与测试也会就地改写
// control.log。一旦内容变了而缓存没变，就会把**旧字节**当新内容返回 ⇒ 记录摘要校验被绕过
// （strictVerifyOffStillSafe 的"盘上 payload 被破坏必须 fail closed"用例曾因此变红）。
// 修法（两件，均为业界通行做法）：
//   (a) 缓存键加入**文件身份**：inode + 大小 + mtime(ns)；取不到身份时**永不命中**（安全侧）。
//   (b) 提供显式失效入口 V4InvalidateRecordCache()，在 Open / 重建 / 压缩 处调用。
// 只清**本线程**的槽：缓存本身是 thread_local，跨线程清空需要全局同步；而 (a) 已覆盖其余线程。
// A1：槽位数。内存上界 = 16 × ≤16 MiB（实测记录均长 7.52 MB）
//   ⇒ 约 120 MiB，线程本地、有界；相对本进程已持有的 ~566 MiB 对象字节可忽略。
constexpr size_t kV4RecordCacheSlots = 16;
struct V4RecordCacheEntry {
    std::string path;
    uint64_t offset = UINT64_MAX;
    uint64_t inode = 0;
    uint64_t size = 0;
    int64_t mtimeNs = 0;
    std::string payload;
};
struct V4RecordCache {
    V4RecordCacheEntry entries[kV4RecordCacheSlots];
    size_t next = 0;   // 环形替换位置
};
inline V4RecordCache &V4RecordCacheSlot() {
    static thread_local V4RecordCache slot;
    return slot;
}
inline void V4InvalidateRecordCache() {
    V4RecordCache &cache = V4RecordCacheSlot();
    for (size_t i = 0; i < kV4RecordCacheSlots; ++i) {
        cache.entries[i].path.clear();
        cache.entries[i].offset = UINT64_MAX;
        cache.entries[i].payload.clear();
    }
    cache.next = 0;
}
inline bool V4FileIdentity(const fs::path &path, uint64_t *inode, uint64_t *size, int64_t *mtimeNs) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return false;
    *inode = uint64_t(st.st_ino);
    *size = uint64_t(st.st_size);
#if defined(__APPLE__)
    *mtimeNs = int64_t(st.st_mtimespec.tv_sec) * 1000000000LL + int64_t(st.st_mtimespec.tv_nsec);
#else
    *mtimeNs = int64_t(st.st_mtim.tv_sec) * 1000000000LL + int64_t(st.st_mtim.tv_nsec);
#endif
    return true;
}
const std::string *V4ReadRecordCached(const fs::path &path, uint64_t offset, std::string &error) {
    V4RecordCache &cache = V4RecordCacheSlot();
    const std::string key = path.string();
    uint64_t inode = 0, size = 0;
    int64_t mtimeNs = 0;
    const bool haveIdentity = V4FileIdentity(path, &inode, &size, &mtimeNs);
    // 命中：路径 + 偏移 + **文件身份**全部一致（见上方 (a)）。多槽线性扫描（≤16 条，开销可忽略）。
    // 返回的指针在**该槽被环形替换之前**一直有效（比原来的「下一次调用前」更强）；
    // ObjectBytes 取到后当次用完即弃、不递归，故零拷贝安全。
    for (size_t i = 0; i < kV4RecordCacheSlots; ++i) {
        const V4RecordCacheEntry &entry = cache.entries[i];
        if (haveIdentity && entry.offset == offset && entry.path == key &&
            entry.inode == inode && entry.size == size && entry.mtimeNs == mtimeNs) {
            return &entry.payload;   // 命中：零磁盘读、零 SHA-256
        }
    }
    prefix_v4::RecordHeader header;
    std::string fresh;
    if (!V4ReadRecord(path, offset, header, fresh, error)) return nullptr;
    V4RecordCacheEntry &entry = cache.entries[cache.next];
    cache.next = (cache.next + 1) % kV4RecordCacheSlots;
    entry.path = key;
    entry.offset = offset;
    entry.inode = inode;
    entry.size = size;
    entry.mtimeNs = mtimeNs;
    entry.payload = std::move(fresh);   // ≤ kMaxPackedPayloadBytes (16 MiB)
    return &entry.payload;
}
} // namespace

struct DiskPrefixV4Store::Impl {
    ~Impl() { StopActor(); }
    fs::path root, logPath, dbPath;
    uint64_t quota = 0;
    bool fastAdmissionStale = true;    // FAST_ADMISSION: roots.bytes may be stale (start, or an ERASE since the last recompute)
    std::unique_ptr<Database> database;
    uint64_t sequence = 0;
    // Runtime readers pin a published checkpoint while its bytes are being
    // assembled. Admission FIFO skips pinned roots; per-root flock files add
    // the cross-process boundary, while lifecycle protects Compact.
    struct PinState {
        std::mutex mutex;
        std::map<std::string, uint64_t> checkpoints;
    };
    struct ObserverState {
        uint64_t next = 1;
        std::mutex mutex;
        std::map<uint64_t, std::function<void(const std::string &, const std::string &)>> callbacks;
        std::map<uint64_t, DiskPrefixCache::CheckpointReclaimPredicate> predicates;
        std::map<uint64_t, DiskPrefixCache::CheckpointCorruptRetirePredicate> corruptPredicates;
    };
    std::shared_ptr<PinState> pins = std::make_shared<PinState>();
    std::shared_ptr<ObserverState> observers = std::make_shared<ObserverState>();
    std::thread compactor;
    std::mutex compactMutex;
    std::condition_variable compactCv;
    bool compactRequested = false;
    bool compactStopping = false;
    // A recoverable short tail may be read after rebuild, but appending after
    // it can turn a failed rollback into a permanently bad control log.
    // Rebuild sets this flag; live-log writers then fail closed.
    bool appendBlocked = false;

    // ===== Segmented log (DESIGN-v3-segments C1-C5; FASTLLM_PREFIX_V4_SEGMENTS=1) =====
    // Kafka layout: one file per segment, named by the global byte offset of its first byte
    // (Kafka 4.1.0 LogFileUtils.java:102-108, 20 digits); lookup = floor of the offset
    // (LogSegments.java:210-221). Every offset stored in the catalog or in a COMMIT stays global.
    bool segmented = false;
    uint64_t segmentBytes = 1ull << 30;
    struct SegmentInfo { fs::path path; uint64_t size = 0; };
    mutable std::mutex segmentsMutex;          // readers resolve offsets while the actor appends/unlinks
    std::map<uint64_t, SegmentInfo> segments;  // base -> file
    bool segmentHoldRoll = false;              // C2: PREPARE..COMMIT never straddle two segments
    bool segmentDeletionAppend = false;        // C5: ERASE/kv records of a deletion use the deletion budget
    uint64_t segmentsUnlinked = 0;
    // root descriptor hash -> minimum offset of every object it reaches (immutable while live).
    std::unordered_map<std::string, uint64_t> rootMinOffset;
    uint64_t rootMinOffsetHits = 0, rootMinOffsetWalks = 0;

    fs::path SegmentPath(uint64_t base) const {
        char name[64];
        std::snprintf(name, sizeof(name), "seg-%020llu.log", (unsigned long long)base);
        return root / name;
    }
    // Port of v3 ContainerIds (c0b93854 :864-876): strict name parse, regular file, no symlink.
    bool ListSegments(std::string &error) {
        std::map<uint64_t, SegmentInfo> found;
        std::error_code listError;
        for (const auto &entry : fs::directory_iterator(root, listError)) {
            const std::string name = entry.path().filename().string();
            if (name.size() != 28 || name.compare(0, 4, "seg-") != 0 || name.compare(24, 4, ".log") != 0) continue;
            const std::string digits = name.substr(4, 20);
            if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
            if (entry.is_symlink() || !entry.is_regular_file()) { error = "v4_segment_type"; return false; }
            const uint64_t base = std::stoull(digits);
            if (SegmentPath(base).filename() != entry.path().filename()) continue;
            std::error_code sizeError;
            const uint64_t size = uint64_t(fs::file_size(entry.path(), sizeError));
            if (sizeError) { error = "v4_segment_stat"; return false; }
            found[base] = SegmentInfo{entry.path(), size};
        }
        if (listError) { error = "v4_segment_list"; return false; }
        std::lock_guard<std::mutex> lock(segmentsMutex);
        segments.swap(found);
        return true;
    }
    uint64_t SegmentEndOffset() const {
        std::lock_guard<std::mutex> lock(segmentsMutex);
        if (segments.empty()) return 0;
        return segments.rbegin()->first + segments.rbegin()->second.size;
    }
    // v3 PhysicalBytes (c0b93854 :998-1003): sum over all files.
    uint64_t SegmentPhysicalBytes() const {
        std::lock_guard<std::mutex> lock(segmentsMutex);
        uint64_t total = 0;
        for (const auto &entry : segments) total += entry.second.size;
        return total;
    }
    // Global offset -> (file, local offset). Flag off: the single control.log.
    bool LogLocation(uint64_t global, fs::path &path, uint64_t &local, std::string &error) const {
        if (!segmented) { path = logPath; local = global; return true; }
        std::lock_guard<std::mutex> lock(segmentsMutex);
        auto it = segments.upper_bound(global);
        if (it == segments.begin()) { error = "v4_segment_deleted"; return false; }
        --it;
        if (global >= it->first + it->second.size) { error = "v4_segment_range"; return false; }
        path = it->second.path;
        local = global - it->first;
        return true;
    }
    // New empty segment at the current end (v3 NewContainer :1011-1023 without the v3 header:
    // the v4 record format is unchanged). Caller holds segmentsMutex.
    bool RollSegmentLocked(std::string &error) {
        const uint64_t next = segments.empty() ? 0
            : segments.rbegin()->first + segments.rbegin()->second.size;
        if (segments.count(next)) return true;   // the last segment is still empty
        const fs::path created = SegmentPath(next);
        const int fd = open(created.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0) { error = "v4_segment_create"; return false; }
        const bool synced = fdatasync(fd) == 0;
        close(fd);
        if (!synced) { error = "v4_segment_sync"; return false; }
        SyncDirectory(root);
        segments[next] = SegmentInfo{created, 0};
        std::fprintf(stderr, "[Prefix SSD] v4 segment-rolled: base=%llu segments=%zu segment_bytes=%llu.\n",
                     (unsigned long long)next, segments.size(), (unsigned long long)segmentBytes);
        return true;
    }
    // C2: append to the newest segment; roll first when the record does not fit (v3 ActiveContainer
    // :1048-1072, Kafka shouldRoll). Quota gate as AppendLiveRecord, except deletion records (C5 step 2,
    // v3 deletion budget :1527-1546).
    bool AppendSegmentRecord(uint32_t type, uint64_t recordSequence, const std::string &payload,
                             uint64_t *outOffset, std::string &error) {
        const uint64_t physical = RecordPhysicalBytes(payload.size());
        if (!segmentDeletionAppend && quota > 0 && Add(SegmentPhysicalBytes(), physical) > quota) {
            error = "capacity_unavailable";
            return false;
        }
        uint64_t base = 0;
        fs::path path;
        {
            std::lock_guard<std::mutex> lock(segmentsMutex);
            bool roll = segments.empty();
            if (!roll && !segmentHoldRoll) {
                const auto &last = segments.rbegin()->second;
                roll = last.size > 0 && Add(last.size, physical) > segmentBytes;
            }
            if (roll && !RollSegmentLocked(error)) return false;
            base = segments.rbegin()->first;
            path = segments.rbegin()->second.path;
        }
        uint64_t local = 0;
        const bool ok = AppendRecordToFile(path.string(), type, recordSequence, payload, &local, error);
        if (!ok && error == "v4_control_rollback") {
            appendBlocked = true;
            SetMeta("append_blocked", "1");
        }
        std::error_code sizeError;
        const uint64_t now = uint64_t(fs::file_size(path, sizeError));
        {
            std::lock_guard<std::mutex> lock(segmentsMutex);
            auto it = segments.find(base);
            if (it != segments.end() && !sizeError) it->second.size = now;
        }
        if (ok && outOffset != nullptr) *outOffset = base + local;
        return ok;
    }
    // C5: delete the oldest whole segment (Kafka retention order, UnifiedLog.java:1847-1889:
    // catalog first, file last). No copy-forward (user 2026-11-06): every live root that reaches
    // the segment is erased with it. Never waits: a pinned / vetoed / leased dependent -> false
    // (LMCache local_disk_backend.py:399-417). Actor thread only.
    bool ReclaimOldestSegment(std::string &reason) {
        const uint64_t startNs = NowNs();
        uint64_t base = 0, size = 0;
        fs::path path;
        {
            std::lock_guard<std::mutex> lock(segmentsMutex);
            if (segments.empty()) { reason = "no_segment"; return false; }
            if (segments.size() == 1) {
                if (segments.begin()->second.size == 0) { reason = "empty_store"; return false; }
                std::string rollError;
                if (!RollSegmentLocked(rollError)) { reason = rollError; return false; }
            }
            base = segments.begin()->first;
            size = segments.begin()->second.size;
            path = segments.begin()->second.path;
        }
        const uint64_t end = base + size;
        // 1. Read-only walk of all live roots (the actor is the only writer).
        struct Dependent { std::string identity, key; };
        std::vector<Dependent> dependents;
        size_t walked = 0;
        {
            struct LiveRow { std::string identity, key, root; };
            std::vector<LiveRow> live;
            {
                Statement roots(database->value,
                                "SELECT identity,key,root,sequence FROM roots WHERE live=1 "
                                "ORDER BY sequence ASC, identity ASC, key ASC");
                while (roots.Row()) live.push_back(LiveRow{roots.Text(0), roots.Text(1),
                                                           roots.Text(2) + "#" + roots.Text(3)});
            }
            static const bool walkCheck = []() {
                const char *v = std::getenv("FASTLLM_PREFIX_V4_SEGMENT_WALK_CHECK");
                return v != nullptr && std::strcmp(v, "1") == 0;
            }();
            auto fullWalkHit = [&](const std::string &root, uint64_t *minOffset) {
                std::set<std::string> reachable;
                std::string walkError;
                if (!ReachableFromRoot(root, reachable, walkError)) return true;  // unreadable: retire
                std::vector<std::string> hashes(reachable.begin(), reachable.end());
                std::vector<uint64_t> offsets;
                std::vector<uint8_t> found;
                ObjectOffsets(hashes, &offsets, &found);
                bool hit = false;
                uint64_t lowest = UINT64_MAX;
                for (size_t i = 0; i < hashes.size(); ++i) {
                    if (!found[i]) continue;
                    lowest = std::min(lowest, offsets[i]);
                    hit = hit || (offsets[i] >= base && offsets[i] < end);
                }
                if (minOffset) *minOffset = lowest;
                return hit;
            };
            {   // memo key = root#sequence (a recommit gets a new sequence); keep it bounded to live roots
                std::unordered_map<std::string, uint64_t> kept;
                for (const auto &row : live) {
                    auto m = rootMinOffset.find(row.root);
                    if (m != rootMinOffset.end()) kept.emplace(m->first, m->second);
                }
                rootMinOffset.swap(kept);
            }
            for (const auto &row : live) {
                ++walked;
                const std::string rootHash = row.root.substr(0, row.root.find('#'));
                bool hit;
                auto memo = rootMinOffset.find(row.root);
                if (memo != rootMinOffset.end()) {
                    ++rootMinOffsetHits;
                    hit = memo->second < end;
                } else {
                    ++rootMinOffsetWalks;
                    uint64_t lowest = UINT64_MAX;
                    hit = fullWalkHit(rootHash, &lowest);
                    if (!hit) rootMinOffset[row.root] = lowest;   // only memoise a readable, surviving root
                }
                if (walkCheck && hit != fullWalkHit(rootHash, nullptr)) {
                    std::fprintf(stderr, "[Prefix SSD] v4 segment-walk-check MISMATCH root=%s base=%llu end=%llu.\n",
                                 rootHash.c_str(), (unsigned long long)base, (unsigned long long)end);
                    std::abort();
                }
                if (hit) dependents.push_back({row.identity, row.key});
            }
        }
        const uint64_t walkNs = NowNs() - startNs;
        // kv rows are re-appended in step 3; bound them before any ERASE so a deletion can never grow
        // the store. Production kv = identity/hints/cost keys, hints capped at 8 MiB.
        {
            Statement kvBytes(database->value, "SELECT COALESCE(SUM(LENGTH(key)+LENGTH(value)),0) FROM kv");
            const uint64_t total = kvBytes.Row() ? uint64_t(kvBytes.Number(0)) : 0;
            if (total > segmentBytes / 4) { reason = "kv_too_large"; return false; }
        }
        // 2. Retire the dependents atomically w.r.t. new pins (same rule as admission).
        std::vector<std::pair<std::string, std::string>> erased;
        bool complete = true;
        {
            std::lock_guard<std::mutex> pinLock(pins->mutex);
            std::vector<std::unique_ptr<Lease>> leases;
            for (const auto &dependent : dependents) {
                if (IsPinnedLocked(dependent.key, dependent.identity)) { reason = "pinned"; return false; }
                if (!ReclaimAllowed(dependent.key, dependent.identity)) { reason = "reclaim_veto"; return false; }
                try {
                    leases.emplace_back(new Lease(RootLeasePath(dependent.key, dependent.identity),
                                                  LOCK_EX | LOCK_NB));
                } catch (const std::exception &) { reason = "root_lease_busy"; return false; }
            }
            segmentDeletionAppend = true;
            struct ResetFlag { bool &flag; ~ResetFlag() { flag = false; } } resetFlag{segmentDeletionAppend};
            for (const auto &dependent : dependents) {
                prefix_v4::EraseRecord erase;
                erase.sequence = sequence + 1;
                erase.key = dependent.key;
                erase.identity = dependent.identity;
                const std::vector<char> eraseBytes = EncodeErase(erase);
                std::string appendError;
                if (!AppendLiveRecord(kRecordErase, ++sequence,
                                      std::string(eraseBytes.data(), eraseBytes.size()), nullptr,
                                      appendError)) {
                    --sequence;
                    reason = "erase_append:" + appendError;
                    complete = false;
                    break;
                }
                Statement retire(database->value, "UPDATE roots SET live=0 WHERE identity=? AND key=?");
                retire.Text(1, dependent.identity); retire.Text(2, dependent.key); retire.Run();
                erased.emplace_back(dependent.key, dependent.identity);
            }
            if (!erased.empty()) {
                SetMeta("sequence", std::to_string(sequence));
                RefreshLogWatermark();
            }
        }
        for (const auto &entry : erased) NotifyCheckpointErased(entry.first, entry.second);
        if (!complete) return false;
        // 3. kv values may have their only record in this segment: carry them forward the way
        //    CompactLocked does (aed65017 :8595-8607). Checkpoint descriptors need no copy: a
        //    descriptor in this segment belongs to a root committed here, which step 2 erased.
        size_t kvRewritten = 0;
        {
            std::vector<std::pair<std::string, std::string>> rows;
            {
                Statement entries(database->value, "SELECT key,value FROM kv");
                while (entries.Row()) rows.emplace_back(entries.Text(0), entries.Text(1));
            }
            segmentDeletionAppend = true;
            struct ResetFlag { bool &flag; ~ResetFlag() { flag = false; } } resetFlag{segmentDeletionAppend};
            for (const auto &row : rows) {
                prefix_v4::MetadataRecord record;
                record.sequence = sequence + 1;
                record.key = "kv:" + row.first;
                record.value = row.second;
                const std::vector<char> payload = EncodeMetadata(record);
                std::string appendError;
                if (!AppendLiveRecord(kRecordMetadata, ++sequence,
                                      std::string(payload.data(), payload.size()), nullptr, appendError)) {
                    --sequence;
                    reason = "kv_append:" + appendError;
                    return false;
                }
                ++kvRewritten;
            }
            if (kvRewritten != 0) {
                SetMeta("sequence", std::to_string(sequence));
                RefreshLogWatermark();
            }
        }
        // 4. Catalog first (Kafka advances logStartOffset before deleting files).
        {
            Transaction transaction(*database);
            Statement blocks(database->value,
                             "DELETE FROM blocks WHERE object_hash IN "
                             "(SELECT hash FROM objects WHERE offset>=? AND offset<?)");
            blocks.Number(1, base); blocks.Number(2, end); blocks.Run();
            Statement rows(database->value, "DELETE FROM objects WHERE offset>=? AND offset<?");
            rows.Number(1, base); rows.Number(2, end); rows.Run();
            transaction.Commit();
        }
        // 5. Unlink the whole file (v3 DropUnreferenced :1383-1406). An fd already open keeps
        //    reading the old bytes (they never change); a reader resolving later gets
        //    v4_segment_deleted (fail closed, never other bytes: base offsets are never reused).
        if (unlink(path.c_str()) != 0 && errno != ENOENT) { reason = "unlink_failed"; return false; }
        SyncDirectory(root);
        {
            std::lock_guard<std::mutex> lock(segmentsMutex);
            segments.erase(base);
        }
        ++segmentsUnlinked;
        fastAdmissionStale = true;
        V4InvalidateRecordCache();
        std::fprintf(stderr,
                     "[Prefix SSD] v4 segment-unlinked: base=%llu bytes=%llu roots_walked=%zu "
                     "roots_erased=%zu kv_rewritten=%zu walk_ms=%.1f total_ms=%.1f "
                     "physical_after=%llu quota=%llu segments=%zu unlinked_total=%llu "
                     "memo_hits=%llu memo_walks=%llu.\n",
                     (unsigned long long)base, (unsigned long long)size, walked, erased.size(),
                     kvRewritten, double(walkNs) / 1e6, double(NowNs() - startNs) / 1e6,
                     (unsigned long long)SegmentPhysicalBytes(), (unsigned long long)quota,
                     segments.size(), (unsigned long long)segmentsUnlinked,
                     (unsigned long long)rootMinOffsetHits, (unsigned long long)rootMinOffsetWalks);
        std::fflush(stderr);
        return true;
    }

    // Lowered quota: delete oldest segments right after Open (PLAN-v3 S2, v3 shrink c0b93854 :1550).
    void ShrinkSegmentsToQuota() {
        if (!segmented || quota == 0 || SegmentPhysicalBytes() <= quota) return;
        std::string runError;
        RunActor([&] {
            RootLocks locks;
            std::string lockError;
            if (!AcquireLocks(locks, false, true, lockError)) return false;
            const uint64_t before = SegmentPhysicalBytes();
            std::string reason;
            size_t deleted = 0;
            while (SegmentPhysicalBytes() > quota && ReclaimOldestSegment(reason)) ++deleted;
            std::fprintf(stderr, "[Prefix SSD] v4 segments shrink-at-open: deleted=%zu physical_before=%llu "
                                 "physical_after=%llu quota=%llu reason=%s.\n",
                         deleted, (unsigned long long)before, (unsigned long long)SegmentPhysicalBytes(),
                         (unsigned long long)quota, deleted ? "ok" : reason.c_str());
            std::fflush(stderr);
            return true;
        }, runError);
    }

    static std::string PinKey(const std::string &key, const std::string &identity) {
        return identity + "\n" + key;
    }
    fs::path RootLeasePath(const std::string &key, const std::string &identity) const {
        return root / (".checkpoint-" + DiskPrefixCache::Digest(PinKey(key, identity)));
    }
    bool IsPinnedLocked(const std::string &key, const std::string &identity) const {
        const std::string pin = PinKey(key, identity);
        auto found = pins->checkpoints.find(pin);
        return found != pins->checkpoints.end() && found->second != 0;
    }
    bool ReclaimAllowed(const std::string &key, const std::string &identity) const {
        std::vector<DiskPrefixCache::CheckpointReclaimPredicate> predicates;
        {
            std::lock_guard<std::mutex> lock(observers->mutex);
            for (const auto &entry : observers->predicates) predicates.push_back(entry.second);
        }
        for (const auto &predicate : predicates) {
            try {
                if (predicate && !predicate(key, identity)) return false;
            } catch (...) {
                // A failed safety observer must fail closed.
                return false;
            }
        }
        return true;
    }
    bool CorruptRetireAllowed(const std::string &key, const std::string &identity) const {
        std::vector<DiskPrefixCache::CheckpointCorruptRetirePredicate> predicates;
        {
            std::lock_guard<std::mutex> lock(observers->mutex);
            for (const auto &entry : observers->corruptPredicates) predicates.push_back(entry.second);
        }
        if (predicates.empty()) return false;
        for (const auto &predicate : predicates) {
            try {
                if (predicate && !predicate(key, identity)) return false;
            } catch (...) {
                return false;
            }
        }
        return true;
    }
    bool OldestUnpinnedRoot(std::string &identity, std::string &key,
                            std::unique_ptr<Lease> &rootLease) {
        Statement oldest(database->value,
                         "SELECT identity,key FROM roots WHERE live=1 "
                         "ORDER BY sequence ASC, identity ASC, key ASC");
        while (oldest.Row()) {
            const std::string candidateIdentity = oldest.Text(0);
            const std::string candidateKey = oldest.Text(1);
            if (IsPinnedLocked(candidateKey, candidateIdentity)) continue;
            if (!ReclaimAllowed(candidateKey, candidateIdentity)) continue;
            try {
                // Cross-process reader protection: keep the exclusive root
                // lease until ERASE has updated the catalog. A reader takes
                // the shared lease before pinning, so it cannot slip between
                // candidate selection and the durable tombstone.
                rootLease.reset(new Lease(RootLeasePath(candidateKey, candidateIdentity),
                                          LOCK_EX | LOCK_NB));
            } catch (const std::exception &) {
                rootLease.reset();
                continue;
            }
            identity = candidateIdentity;
            key = candidateKey;
            return true;
        }
        return false;
    }
    bool ReachableFromRoot(const std::string &rootHash, std::set<std::string> &reachable,
                           std::string &error) {
        std::string rootText;
        if (!ObjectBytes(rootHash, rootText, error)) return false;
        reachable.insert(rootHash);
        std::vector<ObjectListEntry> list;
        std::string listError;
        if (DecodeObjectList(rootText.data(), rootText.size(), list, listError)) {
            for (const auto &entry : list) reachable.insert(entry.hash);
            return true;
        }
        std::string parseError;
        const Json state = Json::parse(rootText, parseError);
        if (!parseError.empty() || !state.is_object() ||
            !state["chain_tip"].is_string() || !state["block_count"].is_number()) {
            error = "v4_root_object";
            return false;
        }
        Refs stateRefs;
        CollectRefs(state, stateRefs);
        for (const auto &ref : stateRefs) reachable.insert(ref.first);
        const uint64_t blockCount = uint64_t(state["block_count"].number_value());
        if (blockCount == 0 || blockCount > prefix_v4::kMaxBlockCount) {
            error = "v4_block_count";
            return false;
        }
        std::string cursor = state["chain_tip"].string_value();
        for (uint64_t step = 0; step < blockCount; ++step) {
            Statement block(database->value,
                            "SELECT object_hash,parent FROM blocks WHERE chain_key=?");
            block.Text(1, cursor);
            if (!block.Row()) {
                error = (step == 0) ? "v4_block_missing" : "v4_block_parent_missing";
                return false;
            }
            const std::string objectHash = block.Text(0);
            const std::string parent = block.Text(1);
            reachable.insert(objectHash);
            std::string blockText;
            if (!ObjectBytes(objectHash, blockText, error)) return false;
            std::string blockError;
            const Json blockObject = Json::parse(blockText, blockError);
            DiskPrefixV4Store::BlockRow parsed;
            if (!blockError.empty() || !ParseBlockRow(blockObject, parsed, blockError)) {
                error = blockError.empty() ? "v4_block_object" : blockError;
                return false;
            }
            Refs blockRefs;
            CollectRefs(blockObject["kv"], blockRefs);
            for (const auto &ref : blockRefs) reachable.insert(ref.first);
            cursor = parent;
        }
        return true;
    }
    // roots.bytes is an admission index, not the physical log size. After an
    // ERASE, an object formerly charged to the retired root may remain
    // reachable through another live root. Recompute this cold-path index by
    // first-live sequence ownership before deciding whether a new write fits.
    bool RecomputeLiveBytesForQuota(std::string &error) {
        std::map<std::string, uint64_t> objectBytes;
        Statement objects(database->value, "SELECT hash,bytes FROM objects");
        while (objects.Row()) objectBytes[objects.Text(0)] = uint64_t(objects.Number(1));
        std::set<std::string> claimed;
        struct RootBytes { std::string identity, key; uint64_t bytes; };
        std::vector<RootBytes> rows;
        Statement roots(database->value,
                        "SELECT identity,key,root FROM roots WHERE live=1 "
                        "ORDER BY sequence ASC, identity ASC, key ASC");
        while (roots.Row()) {
            std::set<std::string> reachable;
            if (!ReachableFromRoot(roots.Text(2), reachable, error)) return false;
            uint64_t bytes = 0;
            for (const auto &hash : reachable) {
                if (!claimed.insert(hash).second) continue;
                auto found = objectBytes.find(hash);
                if (found != objectBytes.end()) bytes += found->second;
            }
            rows.push_back({roots.Text(0), roots.Text(1), bytes});
        }
        Transaction transaction(*database);
        for (const auto &row : rows) {
            Statement update(database->value,
                             "UPDATE roots SET bytes=? WHERE identity=? AND key=?");
            update.Number(1, row.bytes); update.Text(2, row.identity); update.Text(3, row.key);
            update.Run();
        }
        transaction.Commit();
        return true;
    }
    // FAST_ADMISSION C2: the read-only half of RecomputeLiveBytesForQuota, same logic, no transaction.
    struct RootBytesRow { std::string identity, key; uint64_t bytes; };
    bool ComputeLiveBytesRows(std::vector<RootBytesRow> &rows, size_t &walked, std::string &error) {
        rows.clear(); walked = 0;
        std::map<std::string, uint64_t> objectBytes;
        Statement objects(database->value, "SELECT hash,bytes FROM objects");
        while (objects.Row()) objectBytes[objects.Text(0)] = uint64_t(objects.Number(1));
        std::set<std::string> claimed;
        Statement roots(database->value,
                        "SELECT identity,key,root FROM roots WHERE live=1 "
                        "ORDER BY sequence ASC, identity ASC, key ASC");
        while (roots.Row()) {
            std::set<std::string> reachable;
            if (!ReachableFromRoot(roots.Text(2), reachable, error)) return false;
            ++walked;
            uint64_t bytes = 0;
            for (const auto &hash : reachable) {
                if (!claimed.insert(hash).second) continue;
                auto found = objectBytes.find(hash);
                if (found != objectBytes.end()) bytes += found->second;
            }
            rows.push_back({roots.Text(0), roots.Text(1), bytes});
        }
        return true;
    }
    void ApplyLiveBytesRows(const std::vector<RootBytesRow> &rows) {
        Transaction transaction(*database);
        for (const auto &row : rows) {
            Statement update(database->value,
                             "UPDATE roots SET bytes=? WHERE identity=? AND key=? AND live=1");
            update.Number(1, row.bytes); update.Text(2, row.identity); update.Text(3, row.key);
            update.Run();
        }
        transaction.Commit();
    }
    void NotifyCheckpointErased(const std::string &key, const std::string &identity) {
        std::vector<std::function<void(const std::string &, const std::string &)>> callbacks;
        {
            std::lock_guard<std::mutex> lock(observers->mutex);
            for (const auto &entry : observers->callbacks) callbacks.push_back(entry.second);
        }
        // Metadata-only callbacks execute after the catalog mutation and after
        // the observer mutex is released. They must never perform store I/O or
        // re-enter the actor; a throwing observer is isolated from durability.
        for (const auto &callback : callbacks) {
            try { if (callback) callback(key, identity); }
            catch (...) { std::fprintf(stderr,
                "[Prefix SSD] v4 erase observer threw; metadata invalidation ignored.\n"); }
        }
    }

    bool AppendLiveRecord(uint32_t type, uint64_t recordSequence,
                          const std::string &payload, uint64_t *outOffset,
                          std::string &error) {
        if (appendBlocked) {
            error = "v4_control_uncommitted_tail";
            return false;
        }
        if (segmented) return AppendSegmentRecord(type, recordSequence, payload, outOffset, error);
        // Every normal append, including an admission ERASE, must obey the
        // physical quota itself. Higher-level projection is an optimization;
        // this final gate prevents a stale estimate or a record boundary from
        // writing past quota.
        if (quota > 0 && Add(PhysicalBytes(), RecordPhysicalBytes(payload.size())) > quota) {
            error = "capacity_unavailable";
            RequestCompact();
            return false;
        }
        const bool ok = AppendRecordToFile(logPath.string(), type, recordSequence,
                                           payload, outOffset, error);
        if (!ok && error == "v4_control_rollback") {
            appendBlocked = true;
            // Other already-open writers and subsequent Open() calls must see
            // the same restriction, even when the projection watermark matches.
            SetMeta("append_blocked", "1");
        }
        return ok;
    }
    // 写入对象先入队，按"一条记录一批"落盘（方案 §10.1/§10.3）：每个检查点
    // 只需个位数次 fdatasync，而不是每个对象一次。
    struct PendingObject { std::string hash; uint32_t kind; std::string bytes; };
    std::vector<PendingObject> pendingObjects;
    // A2/写路径并发化（2026-10-02）：待写对象的**哈希索引**。旧实现 IsPending 每次线性扫
    // pendingObjects（每对象一次 ⇒ 一个检查点 O(n²) 次字符串比较，实测 p50 254 / max 508 步）。
    // 索引与 pendingObjects 同生共死：入队时插入、FlushObjects swap 出去时清空，语义完全不变。
    std::set<std::string> pendingHashSet;
    uint64_t pendingBytes = 0;
    // §14 计时/计数（只在 actor 线程累加，单调）
    uint64_t packNs = 0, catalogNs = 0;
    // §14：提交段的其余分项（此前完全没有计数，导致"约 1 s 花在哪"无法回答）。
    uint64_t refWalkNs = 0, metadataEncodeNs = 0, metadataHashNs = 0, metadataBytes = 0;
    uint64_t precheckScanNs = 0, actorWaitNs = 0;
    // §14 commit_wait_ms 拆分：排队（入队→出队）与真正执行必须分开，
    // 否则无法区分"actor 慢"和"actor 被别的调用占住/被调度延迟"。
    uint64_t actorQueueNs = 0, actorExecNs = 0, actorQueuedCalls = 0, actorInlineCalls = 0;
    // §22.3：actor 执行内部细分 + 线程 CPU 时间（CPU≪墙钟 = 被抢 CPU）
    uint64_t lockNs = 0, verifyNs = 0, verifyBytes = 0, writeObjectNs = 0, actorCpuNs = 0;
    // §② 对象装配/拷贝的细分（此前只知道"这一段共 ~161ms"，不知道具体在哪一步）
    uint64_t quotaNs = 0, rootListNs = 0, flushNs = 0, prepareNs = 0, commitNs = 0, rootTxNs = 0, evictNs = 0;
    // §② 循环内部细分：每个对象要查一次存在性、扫一次待写表、再把字节拷进队列；
    // 每 8MiB 还要把队列里的字节再拼装成一条记录。
    uint64_t lookupNs = 0, pendingScanNs = 0, stageNs = 0, assembleNs = 0;
    uint64_t objectCount = 0, pendingScanSteps = 0;
    // A2（2026-10-02）：本进程成功落盘（COMMIT 已同步并已入投影）的 **prefix 检查点**累计数
    // （encoder/图片提交不计入，与模型侧 expected_checkpoints 同口径）。
    // 只计数、不参与任何决策，重启后从 0 重新开始（与 live_checkpoints 的"存量"口径不同）。
    uint64_t durableCommits = 0;
    // R3：为**接收新检查点**而在准入路径上按 FIFO 淘汰
    // 最旧检查点的累计次数（对应 Profile 的 evicted_for_admission）。只计数、不参与决策。
    // 注意：这里的“淘汰”只是写 ERASE 记录 + roots.live=0（退掉可见性），**不删除对象字节**，
    // 因此绝不会让在途检查点已引用的对象失效（引用保护见 CommitBareLocked 的配额段注释）。
    uint64_t evictedForAdmission = 0;
    // 4可观测性：**成功取消在途作业**的累计次数与保留下来的字节数。
    // 背景：取消的成功路径此前**一行日志都不打**、profile 也没有计数 ⇒ 真机 r13/r16-a1/r17-a1
    // 无法判定"取消是否发生过"，只能靠推断。现在取消可观测（日志 + profile），且取消时会把
    // 暂存对象写入日志与索引（见 actor 认领段的"保字节"逻辑），不再静默销毁已发布的引用。
    uint64_t cancelledJobs = 0, cancelledPreservedBytes = 0;
    uint64_t publishDanglingRefs = 0;
    // 1b（2026-10-03）：发布前可达性判定改 fail-closed 后，**被拒绝发布的检查点数**。
    // 语义：该检查点引用了永远不可读的分片 ⇒ 按"可见 ⇒ 必可读"不变量**扣下不发**（不是错误，
    // 是降级：上层会回退到更旧的可用父记录）。publishDanglingRefs 仍记录被点名引用的条数。
    uint64_t publishRefusedCheckpoints = 0;
    // 1c（2026-10-03）：出队前保字节的可观测性。preservedOnDrop = 被出队前抢救下来的作业数；
    // unpreservedDrops = **仍持有字节却没抢救成功就丢**的作业数（正常必须恒为 0，>0 即真凶点名）。
    uint64_t preservedOnDrop = 0, unpreservedDrops = 0;
    // S3b（2026-10-03）：保字节写盘失败次数（**必须恒为 0**；>0 说明"已发布记录引用的字节没落地"）。
    uint64_t preserveFlushFailures = 0;
    uint64_t rebuildValid = 0, rebuildTail = 0, rebuildCorrupt = 0, rebuildExcluded = 0;
    // ---- catalog actor（方案 §11.1：每 canonical root 单写者）----
    // 所有会改控制日志/catalog 的操作都在**同一个线程**上执行：sequence 分配、
    // 发布顺序与 SQLite 写因此天然串行，调用方只等待结果，不并发写库。
    std::thread actor;
    std::mutex actorMutex;
    std::condition_variable actorCv;
    std::deque<std::function<void()>> actorQueue;
    bool actorStopping = false;
    std::thread::id actorId;

    // ---- §④ 异步提交（在途最多 1 个）----
    // 目的：把"等 actor 落盘"从"提交后立刻等"改为"下一次提交前收尾"，
    // 使上一张检查点的写盘与本次的 D2H/打包/哈希重叠。单 actor 顺序不变。
    struct CommitResult {
        std::mutex mutex;
        std::condition_variable done;
        bool finished = false;
        bool ok = false;
        std::string error;
        DiskPrefixCache::CommitIdentity identity;
        uint64_t sequence = 0;
        // S2：本笔检查点因"引用不可达"被**扣下未发布**（不是落盘成功）。
        // 用途：让"已落盘"与"被扣下"可区分——不得再打出 `v4-durable`、seq 也不得被当有效值消费。
        bool withheld = false;
        std::function<void(bool)> completion;
        // Called under result.mutex. The acknowledgement uses exactly the
        // existing terminal result; admitting/preparing is never durable.
        void Acknowledge(bool durable) {
            auto callback = std::move(completion);
            if (callback) callback(durable);
        }
    };
    DiskPrefixV4Store *owner = nullptr;   // 回指外壳：备菜线程要调用 store 的私有落盘实现
    std::mutex commitPipeline;                 // 串行化"受理/收尾"这一对操作
    uint64_t commitSubmitNs = 0, commitReapNs = 0, reapWaitNs = 0, reapWaitCount = 0;
    // ---- §⑥ 后台备菜流水线（唯一备菜线程 + FIFO，深度上限 2）----
    // 硬约束（计划 §6.1/§17.1）：投递顺序 = 提交顺序；**写盘者仍然只有 actor 一个**。
    // 请求线程只把"状态 Json + 暂存分块"交出来就返回；序列化/摘要/描述符/预检/装配/
    // 清单登记全部在本线程完成后，再按序投递给 actor。
    static constexpr size_t kPipelineDepth = 2;
    struct PrepJob {
        // 受理时移交的输入
        std::string key, identity, family;
        bool keepEquivalent = false;
        DiskPrefixCache::CommitIdentity commitIdentity;
        uint64_t failureBytes = 0;
        bool failureRetentionRejected = false;
        bool prepared = false;                    // true：objects/metadata 已备好（旧 API 路径）
        Json state;                               // false：状态 Json（在备菜线程上序列化）
        std::vector<std::pair<std::string,std::string>> staged;  // false：本次暂存对象
        std::string metadata;                     // true：描述符 / false：备菜产出
        std::vector<Object> objects;              // true：已装配对象 / false：备菜产出
        std::shared_ptr<DiskPrefixV4Store::ObjectLease> objectLease;
        std::shared_ptr<CommitResult> result;     // 受理时即创建（"已受理 ≠ 已持久化"）
        uint64_t acceptNs = 0;
        bool started = false;                     // 备菜线程已开工
        uint64_t heldBytes = 0;                   // P3-B：受理时算出的在途字节上界（释放时精确扣减）
        // §保最新（2026-10-01）：状态位与**取消边界**必须分清——
        //   posted   = 已交给 actor（仅表示交接完成；**不是**不可取消边界）；
        //   committing = actor 已开始写盘（由 actor 在写第一条记录之前、在同一把
        //                pipelineMutex 下认领）⇒ **这才是不可取消的边界**；
        //   cancelled  = **恒为 false 的防御标志**。1d（2026-10-03）之后全代码已无任何
        //                `cancelled.store(true)`：取消降级为"只归还额度、不丢作业"（见
        //                CancelOldestPending），因此下面的检查分支永远不会命中。保留该标志及
        //                其检查**仅作防御**：防止将来有人重新写入它、把"取消=丢检查点"的
        //                旧语义静默复活（那正是块链挖洞/parent-adopt 失败的根因）。
        //   bytesReleased = 额度已归还（取消 / ReapPending / 背压出队三处共用守卫，防重复扣减）。
        bool posted = false;
        std::atomic<bool> cancelled{false};
        bool bytesReleased = false;
        bool committing = false;
        // 1c（2026-10-03）：出队前"保字节"只允许投递一次（actor 认领时会清空 staged/objects）。
        bool preserveSubmitted = false;
        bool failureRetained = false;
    };
    std::deque<std::shared_ptr<PrepJob>> pipeline;   // 受理顺序，唯一的 FIFO
    // Failed commits retain their canonical state and remaining staged bytes.
    // The model's last hot copy stays protected by the failed acknowledgement.
    std::vector<std::shared_ptr<PrepJob>> failedRetainedJobs;
    struct FailureAudit {
        uint64_t terminalId = 0, requestId = 0, requestGeneration = 0;
        uint64_t pageGeneration = 0, copyGeneration = 0, bytes = 0;
        uint64_t failedAtNs = 0, lastOwnerReleaseAtNs = 0;
        std::string cacheIdentity, key, errorCode, errorText, quarantineReason;
    };
    static constexpr size_t kMaxFailedRetained = 16;
    static constexpr uint64_t kMaxFailedBytes = uint64_t(256) << 20;
    static constexpr uint64_t kMaxRetentionAgeNs = uint64_t(1800000) * 1000000;
    std::deque<FailureAudit> failureAudit;
    uint64_t failedRetainedBytes = 0, nextFailureTerminalId = 0, failedRetentionRejects = 0;
    std::mutex pipelineMutex;
    std::condition_variable pipelineCv;
    std::thread prep;
    std::thread::id prepId;
    bool pipelineStopping = false;
    std::shared_ptr<CommitResult> actorInFlight;     // 备菜线程最近投递的一笔：下一笔先收尾它
    uint64_t prepExecNs = 0, prepQueueNs = 0, prepJobs = 0, prepMaxDepth = 0;
    uint64_t pipelineHeldBytes = 0;   // P3-B：已受理未持久化的在途字节（pipelineMutex 保护）
    std::atomic<uint64_t> reusedCheckpoints{0}, failedCheckpoints{0};

    void ActorLoop() {
        actorId = std::this_thread::get_id();
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(actorMutex);
                actorCv.wait(lock, [this] { return actorStopping || !actorQueue.empty(); });
                if (actorStopping && actorQueue.empty()) return;
                job = std::move(actorQueue.front());
                actorQueue.pop_front();
            }
            job();
        }
    }
    void RequestCompact() {
        if (segmented) return;   // DESIGN-v3-segments C7: segment deletion replaces Compact
        // PLAN-v2 Step 1 (default off): no whole-log rewrite while serving. The rewrite holds the
        // lifecycle lock exclusive on the actor and blocks restores, saves and shutdown for its whole run.
        // Saves stay withheld (capacity_unavailable); existing checkpoints keep serving.
        {
            const char *noCompact = std::getenv("FASTLLM_PREFIX_V4_NO_SERVING_COMPACT");
            if (noCompact != nullptr && noCompact[0] == '1' && noCompact[1] == '\0') {
                static std::atomic<bool> logged{false};
                if (!logged.exchange(true)) {
                    std::fprintf(stderr, "[Prefix SSD] v4 no-serving-compact: compact request skipped "
                                         "(store full; saves withheld, restores served).\n");
                    std::fflush(stderr);
                }
                return;
            }
        }
#ifdef FASTLLM_DISK_PREFIX_CACHE_TESTING
        // Some fail-closed tests need to inspect the durable ERASE record
        // before the asynchronous compactor rewrites the log. Production
        // builds never define this test-only escape hatch.
        const char *disabled = std::getenv("FASTLLM_DISK_CACHE_DISABLE_BACKGROUND_COMPACT");
        if (disabled != nullptr && disabled[0] != '\0' && disabled[0] != '0') return;
#endif
        {
            std::lock_guard<std::mutex> lock(compactMutex);
            compactRequested = true;
        }
        compactCv.notify_one();
    }
    void CompactLoop() {
        std::unique_lock<std::mutex> lock(compactMutex);
        for (;;) {
            compactCv.wait(lock, [this] { return compactStopping || compactRequested; });
            if (compactStopping) return;
            compactRequested = false;
            lock.unlock();
            std::string compactError;
            const uint64_t compactStartNs = NowNs();
            const bool compactOk = RunActor([&] { return owner->CompactLocked(compactError); }, compactError);
            if (V4FastAdmission()) {
                std::fprintf(stderr, "[Prefix SSD] v4 fast-admission: site=compact ms=%.1f ok=%d error=%s.\n",
                             double(NowNs() - compactStartNs) / 1e6, compactOk ? 1 : 0, compactError.c_str());
                std::fflush(stderr);
            }
            lock.lock();
        }
    }
    void StartActor() {
        actor = std::thread([this] { ActorLoop(); });
        prep = std::thread([this] { PrepLoop(); });   // §⑥ 唯一备菜线程
        compactor = std::thread([this] { CompactLoop(); });
    }
    void StopActor() {
        // §⑥/§④ 关闭前必须收尾：备菜线程先把"已受理但还没投递"的作业全部备完并投递给
        // actor，再等全部作业的持久化结果——缓存内容可丢，但失败不许静默。
        // 顺序不能颠倒：actor 必须活到所有已受理作业都投递并执行完。
        StopPipeline();
        {
            std::lock_guard<std::mutex> lock(actorMutex);
            actorStopping = true;
        }
        actorCv.notify_all();
        if (actor.joinable()) actor.join();
        {
            std::lock_guard<std::mutex> lock(compactMutex);
            compactStopping = true;
        }
        compactCv.notify_all();
        if (compactor.joinable()) compactor.join();
    }
    // 已在 actor 线程上就直接执行（例如 Open 里的重建），否则投递并等待。
    bool RunActor(const std::function<bool()> &job, std::string &error) {
        if (std::this_thread::get_id() == actorId) {
            // 调用方本身就是 actor：既没有排队，也要把执行时间计入执行段。
            const uint64_t execStart = NowNs();
            ++actorInlineCalls;
            try {
                const bool done = job();
                actorExecNs += NowNs() - execStart;
                return done;
            } catch (const std::exception &e) {
                actorExecNs += NowNs() - execStart;
                error = e.what();
                return false;
            }
        }
        std::promise<bool> finished;
        auto future = finished.get_future();
        const uint64_t postedNs = NowNs();
        {
            std::lock_guard<std::mutex> lock(actorMutex);
            if (actorStopping) { error = "v4_actor_stopping"; return false; }
            actorQueue.push_back([this, &job, &error, &finished, postedNs] {
                const uint64_t execStart = NowNs();
                actorQueueNs += execStart - postedNs;
                ++actorQueuedCalls;
                try { finished.set_value(job()); }
                catch (const std::exception &e) { error = e.what(); finished.set_value(false); }
                catch (...) { error = "v4_actor_exception"; finished.set_value(false); }
                actorExecNs += NowNs() - execStart;
            });
        }
        actorCv.notify_one();
        return future.get();
    }
    // §④ 异步投递：**按值捕获**（std::function 要求可拷贝，工作体用 shared_ptr 持有），
    // 因此调用方可以立即返回；RunActor 那种引用捕获一旦不等待就会悬垂。
    bool PostActor(std::function<void()> job, std::string &error) {
        if (std::this_thread::get_id() == actorId) {
            const uint64_t execStart = NowNs();
            ++actorInlineCalls;
            try { job(); }
            catch (const std::exception &e) { error = e.what(); actorExecNs += NowNs() - execStart; return false; }
            catch (...) { error = "v4_actor_exception"; actorExecNs += NowNs() - execStart; return false; }
            actorExecNs += NowNs() - execStart;
            return true;
        }
        const uint64_t postedNs = NowNs();
        {
            std::lock_guard<std::mutex> lock(actorMutex);
            if (actorStopping) { error = "v4_actor_stopping"; return false; }
            actorQueue.push_back([this, job = std::move(job), postedNs]() mutable {
                const uint64_t execStart = NowNs();
                actorQueueNs += execStart - postedNs;
                ++actorQueuedCalls;
                try { job(); } catch (...) { /* 结果由 job 自己写进 CommitResult */ }
                actorExecNs += NowNs() - execStart;
            });
        }
        actorCv.notify_one();
        return true;
    }
    // §④ 收尾：等待指定提交完成并**原样上报**其错误（绝不静默丢弃）。
    // 注意：这里只等待、不出队——出队统一由 ReapPending 在 FIFO 队首做，因此备菜线程
    // "先收尾上一笔"不会把别人还没 reap 的结果吃掉。
    bool WaitResult(const std::shared_ptr<CommitResult> &result, std::string &error,
                    uint64_t *outSequence = nullptr) {
        if (!result) return true;
        const uint64_t waitStart = NowNs();
        {
            std::unique_lock<std::mutex> lock(result->mutex);
            const bool alreadyDone = result->finished;
            result->done.wait(lock, [&result] { return result->finished; });
            if (!alreadyDone) ++reapWaitCount;    // 只有真的等到了才计数（证明重叠是否兑现）
        }
        const uint64_t waited = NowNs() - waitStart;
        reapWaitNs += waited;
        commitReapNs += waited;
        if (!result->ok) {
            error = result->error.empty() ? "v4_commit_failed" : result->error;
            return false;
        }
        if (outSequence != nullptr) *outSequence = result->sequence;
        return true;
    }
    // 非阻塞收尾：
    // 把"已经完成但还挂在账上"的队首作业出队并扣减在途字节。
    // 背景（只读调研结论）：pipelineHeldBytes 只在 ReapPending / 背压出队 / CancelOldestPending
    // 三处扣减，而 InFlightBytes() 以前刻意不做读侧收尾 ⇒ 没有新的读请求时，已完成作业的
    // heldBytes 会一直计入主机预算判据（stateBytes + queuedStateBytes + storeInflight > HostLimit）。
    // 实测它没造成拒写（0.5 GB / 8 GiB，占比 6~7%），属"账目不及时"的可改进缺口。
    // 语义：只看队首；用 try_to_lock 拿 result 锁，拿不到立刻返回（**绝不等待**，不抵消异步收益）。
    // 1c（2026-10-03）：**出队前保字节**的统一入口（必须持 pipelineMutex 调用，与
    // SubmitPrepared/PruneFinishedHead 同口径）。语义：任何"仍持有 staged/objects 字节"的作业，
    // 在被移出 FIFO 之前都必须把字节交给 actor（actor 认领时写入日志与索引、不发布 root）。
    // 真机 t1b 实证：取消后仍有 64 个分片永久丢失 ⇒ 说明确实存在"未保字节就出队"的路径；
    // 本函数把这条守卫收敛到唯一入口，并在抢救失败时**点名 + 计数**（unpreservedDrops）。
    void PruneFailureAuditLocked() {
        const uint64_t now = NowNs();
        while (!failureAudit.empty() &&
               now > failureAudit.front().failedAtNs &&
               now - failureAudit.front().failedAtNs > kMaxRetentionAgeNs)
            failureAudit.pop_front();
        while (failureAudit.size() > kMaxFailedRetained) failureAudit.pop_front();
    }
    bool PreserveBeforeDropLocked(const std::shared_ptr<PrepJob> &job, const char *where,
                                  bool failed = false) {
        if (!job) return true;
        PruneFailureAuditLocked();
        // The caller already observed a terminal result, sometimes while
        // owning result.mutex. Never reacquire that non-recursive mutex here.
        if (failed && !job->failureRetained && !job->failureRetentionRejected) {
            const uint64_t bytes = HeldBytesOf(*job);
            FailureAudit audit;
            audit.terminalId = ++nextFailureTerminalId;
            audit.requestId = job->commitIdentity.requestId;
            audit.requestGeneration = job->commitIdentity.requestGeneration;
            audit.pageGeneration = job->commitIdentity.pageGeneration;
            audit.copyGeneration = job->commitIdentity.copyGeneration;
            audit.cacheIdentity = job->commitIdentity.cacheIdentity;
            audit.key = job->key; audit.bytes = bytes; audit.failedAtNs = NowNs();
            audit.errorCode = "commit_failed"; audit.errorText = "failed terminal retained";
            audit.quarantineReason = "failed_cache_not_restorable";
            failureAudit.push_back(std::move(audit));
            while (failureAudit.size() > kMaxFailedRetained) failureAudit.pop_front();
            if (failedRetainedJobs.size() >= kMaxFailedRetained ||
                bytes > kMaxFailedBytes || failedRetainedBytes > kMaxFailedBytes - bytes) {
                job->failureRetentionRejected = true;
                ++failedRetentionRejects;
                std::fprintf(stderr,
                    "[Prefix SSD] failed-retention-rejected: status=FAILED_RETENTION_FULL key=%s bytes=%llu count=%zu.\n",
                    job->key.c_str(), (unsigned long long)bytes, failedRetainedJobs.size());
            } else {
                failedRetainedJobs.push_back(job);
                failedRetainedBytes += bytes;
                job->failureBytes = bytes;
                job->failureRetained = true;
            }
        }
        if (job->staged.empty() && job->objects.empty()) return true;   // 不持字节
        if (job->preserveSubmitted) return true;                       // 已交过 actor，不重复投递
        const size_t staged = job->staged.size(), objects = job->objects.size();
        std::string preserveError;
        if (!SubmitPrepared(job, preserveError)) {
            ++unpreservedDrops;
            std::fprintf(stderr,
                         "[Prefix SSD] v4 PRESERVE-FAILED (%s): handoff error=%s key=%s staged=%zu "
                         "objects=%zu unpreserved_drops=%llu.\n",
                         where, preserveError.c_str(), job->key.c_str(), staged, objects,
                         (unsigned long long)unpreservedDrops);
            return false;
        }
        job->preserveSubmitted = true;
        ++preservedOnDrop;
        std::fprintf(stderr,
                     "[Prefix SSD] v4 preserve-before-drop (%s): staged=%zu objects=%zu key=%s "
                     "preserved_on_drop=%llu.\n",
                     where, staged, objects, job->key.c_str(),
                     (unsigned long long)preservedOnDrop);
        return true;
    }
    // A failed job may retain an object lease while its bytes are being rescued
    // into the content-addressed store. Once the handoff has consumed all
    // staged/owned bytes, keeping the lease would make Compact preserve an
    // otherwise unreachable object forever and turn one capacity failure into
    // permanent capacity_unavailable responses. The pending object queue is
    // already protected independently by CompactLocked, so the job lease can
    // be released at this point.
    void ReleaseFailureRetentionIfEmptyLocked(const std::shared_ptr<PrepJob> &job) {
        if (!job || !job->failureRetained || !job->staged.empty() || !job->objects.empty()) return;
        job->objectLease.reset();
        for (auto it = failedRetainedJobs.begin(); it != failedRetainedJobs.end(); ++it) {
            if (*it == job) {
                failedRetainedJobs.erase(it);
                failedRetainedBytes = failedRetainedBytes >= job->failureBytes
                    ? failedRetainedBytes - job->failureBytes : 0;
                break;
            }
        }
        job->failureRetained = false;
        job->failureBytes = 0;
    }
    void PruneFinishedHead() {
        // RCA 1（2026-10-03，真机 merge-verify-a1 实证）：旧实现**每次只弹 1 个已完成队首**，
        // 而 `InFlight()` 每次只调一次 ⇒ FIFO 深度上限为 2（kPipelineDepth）时，
        // "结束时恰好残留 2 笔已完成作业"这一常见形态会让 `InFlight()` 永远读不到 0
        // ⇒ 关服排空必然**假超时**（打印 `store_inflight_bytes=228850590` 那次即此形态；
        //   455579196 − 228850590 = 226728606 恰为最后两笔作业 heldBytes 之和）。
        // 现改为**循环排空所有已完成队首**；每轮重新取锁，避免长时间持 pipelineMutex。
        // 安全性：循环体内会走 PreserveBeforeDropLocked→SubmitPrepared→PostActor，
        // 而 PruneFinishedHead 的调用方（EnqueuePipeline / InFlight / Profile）**均非
        // actor 线程**，故不存在 PostActor 内联执行导致的递归取锁。
        bool popped = false;
        for (;;) {
            std::shared_ptr<PrepJob> head;
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                if (pipeline.empty()) break;
                head = pipeline.front();
                std::unique_lock<std::mutex> resultLock(head->result->mutex, std::try_to_lock);
                if (!resultLock.owns_lock() || !head->result->finished) break;
                if (!head->bytesReleased) {
                    pipelineHeldBytes -= head->heldBytes;
                    head->bytesReleased = true;
                }
                // 3-c（2026-10-03，真机 r19-a1 实证补漏）+ 1c：出队前一律先保字节（收敛到统一入口）。
                PreserveBeforeDropLocked(head, "prune-finished-head", !head->result->ok);
                ReleaseFailureRetentionIfEmptyLocked(head);
                pipeline.pop_front();
                popped = true;
            }
        }
        if (popped) pipelineCv.notify_all();   // 释放槽位，唤醒可能被背压挡住的受理方
    }
    // §⑥ 收尾（队首语义）：按受理顺序把 FIFO 里**全部**作业等到完成并出队——
    // 连"已受理但还没投递"的作业一起等（关服/读侧排空必须做到这一点）。
    // 失败原样上报（返回 false 并给出第一个错误），但不会中途停下不排空，
    // 否则后面的作业会卡在队列里、槽位永远不释放。
    bool ReapPending(std::string &error, uint64_t *outSequence = nullptr) {
        bool ok = true;
        std::string firstError;
        for (;;) {
            std::shared_ptr<PrepJob> head;
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                if (pipeline.empty()) break;
                head = pipeline.front();
            }
            std::string localError;
            const bool done = WaitResult(head->result, localError, outSequence);
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                if (!pipeline.empty() && pipeline.front() == head) {
                    if (!head->bytesReleased) {   // §保最新：取消时已归还过则不重复扣减
                        pipelineHeldBytes -= head->heldBytes;
                        head->bytesReleased = true;
                    }
                    // 1c（2026-10-03）：收尾出队同样必须先保字节（t1b 实证存在未保字节就出队的路径）。
                    PreserveBeforeDropLocked(head, "reap-pending", !done);
                    ReleaseFailureRetentionIfEmptyLocked(head);
                    pipeline.pop_front();
                }
            }
            pipelineCv.notify_all();   // 释放槽位，唤醒可能被背压挡住的受理方
            // S2：被扣下未发布的检查点（`v4_publish_withheld`）**不算失败**：
            // 原因已由 actor 如实打印（"NOT durable"）并计数 publish_refused_checkpoints，
            // 这里不再把它记为 firstError，避免读侧收尾/关服收尾把它当异常上报。
            if (!done && ok && localError != "v4_publish_withheld" &&
                localError != "v4_preserved_not_published") { ok = false; firstError = localError; }
        }
        if (!ok) error = firstError.empty() ? "v4_commit_failed" : firstError;
        return ok;
    }
    // §⑥ 受理：按受理顺序入 FIFO。深度上限 2 —— 满了就先等队首完成并把它收掉腾槽位
    // （这一步同时把队首的失败原样上报，与旧实现在提交入口处抛错的行为一致）。
    bool EnqueuePipeline(const std::shared_ptr<PrepJob> &job, std::string &error) {
        if (!job || (job->commitIdentity.HasAny() && !job->commitIdentity.Valid())) {
            error = "identity_missing";
            return false;
        }
        if (!job->objectLease) {
            job->objectLease = owner->BeginObjectLease(error);
            if (!job->objectLease) return false;
        }
        std::vector<std::string> leaseHashes;
        for (const auto &object : job->objects) leaseHashes.push_back(object.hash);
        for (const auto &item : job->staged) leaseHashes.push_back(item.first);
        if (!job->state.is_null()) {
            Refs refs; CollectRefs(job->state, refs);
            for (const auto &ref : refs) leaseHashes.push_back(ref.first);
        }
        if (!owner->ExtendObjectLease(job->objectLease, leaseHashes, error)) return false;
        std::unique_lock<std::mutex> lock(pipelineMutex);
        for (;;) {
            if (pipelineStopping) { error = "v4_prep_stopping"; return false; }
            if (pipeline.size() < kPipelineDepth) break;
            const std::shared_ptr<PrepJob> head = pipeline.front();
            lock.unlock();   // 等队首完成时必须放锁：备菜线程要拿这把锁才能开工
            std::string headError;
            const bool headDone = WaitResult(head->result, headError);
            lock.lock();
            if (!pipeline.empty() && pipeline.front() == head) {
                // P0 修复（2026-10-02，对标审计发现）：这里**必须**先看 bytesReleased 守卫。
                // 旧代码无条件扣减：若队首已被 CancelOldestPending 扣过（A 仍在 FIFO 且已 started 时
                // 不会就地 finished），背压出队会**二次扣减** ⇒ uint64 下溢成 ~1.8e19 ⇒
                // storeInflight 恒大于 HostLimit ⇒ 此后所有检查点 reason=host_budget_no_victim，
                // 只能重启（与 R3/R6 同级的服务级失效）。
                if (!head->bytesReleased) {
                    pipelineHeldBytes -= head->heldBytes;   // P3-B：背压出队也扣减
                    head->bytesReleased = true;             // §保最新：与取消路径共用同一守卫
                }
                // 1c（2026-10-03）：**背压出队**也必须先保字节。真机 t1b 实证：取消把"尚未开工"的
                // 作业标成 finished 后，模型线程的下一次受理会在这里把它 pop 掉；旧代码不保字节
                // ⇒ 该作业的 KV 分片永久丢失、后续检查点引用不到（64 个分片 0/64 命中）。
                PreserveBeforeDropLocked(head, "enqueue-backpressure", !headDone);
                ReleaseFailureRetentionIfEmptyLocked(head);
                pipeline.pop_front();
            }
            pipelineCv.notify_all();
            // §保最新：队首是**被我们主动取消**的 ⇒ 那是一次已记账的丢点，不能让它把
            // 新的（最新的）提交顶回去；继续循环等出空槽即可。其余失败仍原样上报。
            if (!headDone && headError != "v4_cancelled_oldest" && headError != "v4_publish_withheld" &&
                headError != "v4_preserved_not_published") {
                error = headError.empty() ? "v4_commit_failed" : headError;
                return false;
            }
        }
        job->acceptNs = NowNs();
        job->heldBytes = HeldBytesOf(*job);          // P3-B：受理时记账
        pipelineHeldBytes += job->heldBytes;
        pipeline.push_back(job);
        if (pipeline.size() > prepMaxDepth) prepMaxDepth = pipeline.size();
        lock.unlock();
        pipelineCv.notify_all();
        PruneFinishedHead();   // 2026-10-02：受理成功后顺手收尾已完成的前序作业
        return true;
    }
    std::shared_ptr<PrepJob> NextUnstartedLocked() const {
        for (const auto &job : pipeline) if (!job->started) return job;
        return nullptr;
    }
    // P3-B（2026-10-01）：受理时的“在途字节保守上界”，供主机预算**如实记账**。
    // 状态 Json **不在请求线程序列化**（那是备菜线程的活，挪回来会抵消流水线收益），
    // 改用“字符串按最坏转义 ×6 + 每节点固定开销”的上界；staged/objects/metadata 精确累加。
    static uint64_t JsonHeldUpperBound(const Json &node) {
        uint64_t total = 16;
        switch (node.type()) {
            case Json::NUL: case Json::BOOL: case Json::NUMBER: return total + 24;
            case Json::STRING: return total + uint64_t(node.string_value().size()) * 6 + 8;
            case Json::ARRAY:
                for (const auto &item : node.array_items()) total += JsonHeldUpperBound(item);
                return total;
            case Json::OBJECT:
                for (const auto &entry : node.object_items()) {
                    total += uint64_t(entry.first.size()) * 6 + 8;
                    total += JsonHeldUpperBound(entry.second);
                }
                return total;
        }
        return total;
    }
    static uint64_t HeldBytesOf(const PrepJob &job) {
        uint64_t total = job.metadata.size();
        for (const auto &object : job.objects) total += object.bytes.size() + object.bytesSize;
        for (const auto &chunk : job.staged) total += chunk.second.size();
        if (!job.state.is_null()) total += JsonHeldUpperBound(job.state);
        return total;
    }
    // 只读：已受理未持久化的在途字节（pipelineMutex 保护）。
    uint64_t InFlight() {
        PruneFinishedHead();   // 2026-10-02：先做非阻塞收尾 —— 账目不再依赖"下一次读请求"
        std::lock_guard<std::mutex> lock(pipelineMutex);
        return pipelineHeldBytes;
    }
    // 复核 R6（2026-10-05）：InFlight() 只统计还在额度账上的字节；被 CancelOldestPending
    // 归还过额度（bytesReleased=true）但仍留在 FIFO 里占内存的作业**不在其内** ⇒ InFlight()
    // 低于真实常驻占用。这里把差额单独量出来（只读观测，不改准入/记账语义）。
    uint64_t ResidentQuotaReleasedBytes() {
        std::lock_guard<std::mutex> lock(pipelineMutex);
        uint64_t total = 0;
        for (const auto &job : pipeline) {
            if (job && job->bytesReleased) total += job->heldBytes;
        }
        return total;
    }
    // §保最新、丢最旧：选中最老的、
    // 尚未过不可逆点（!committing）、且额度尚未归还的已受理作业，把**主机额度**让给更新的检查点。
    // 实测依据：目标负载下丢点发生时模型层队列为空（queued_jobs=0），预算被 store_inflight
    // （398 MiB）占满 → 模型层没有牺牲者，只能在 store 层归还额度。
    // 硬约束：**不丢弃作业、不删除任何已持久化数据**——作业仍照常备菜并由 actor 发布
    // （1d 之前"取消=丢检查点"会挖断块链，已由 T1c 实证否决）。
    bool CancelOldestPending(uint64_t *freedBytes, std::string &error) {
        error.clear();
        if (freedBytes != nullptr) *freedBytes = 0;
        std::lock_guard<std::mutex> lock(pipelineMutex);
        if (pipelineStopping) { error = "v4_prep_stopping"; return false; }
        for (auto entry = pipeline.begin(); entry != pipeline.end(); ++entry) {
            const std::shared_ptr<PrepJob> &job = *entry;
            if (!job) continue;
            // §保最新（子代理复核 §3.1）：**必须 continue 而不是 break**。FIFO 深度只有 2，
            // 真机形态通常是 [A 正在提交, B 已 posted 未认领]；见到 A 就 break 会把明明可取消的
            // B 一起放弃 ⇒ 修复在真机上继续打不中。跳过已过不可逆点的 A 去找更旧的?不——B 比
            // 申请准入的最新那笔**更旧**，取消它仍满足"丢最旧、保最新"；存活者顺序也不变。
            if (job->committing) continue;        // 已过不可逆点，跳过它继续找可取消的更旧作业
            if (job->bytesReleased) continue;     // 额度已归还过（含上一次"取消"），不再重复选中
            // 1d：**已记账的写不许丢 —— 取消只做额度会计，不再丢弃作业。**
            // T1c 实证（死点 14848）：旧实现把牺牲者标成 cancelled/finished ⇒ 该检查点**永不发布**
            // ⇒ 块链被挖出一个洞 ⇒ `parent-adopt failed reason=v4_block_parent_missing` ×5
            // ⇒ 退回整段重导 ⇒ 超 pin 预算 ⇒ `write_skipped` ×5。
            // 与业界一致（vLLM/SGLang/LMCache：绝不淘汰"正在写/在用"的块；vLLM 只在引用计数为 0 时淘汰）：
            // 这里把"丢最旧、保最新"降级为**纯记账释放**——归还主机额度让最新的检查点能被受理，
            // 但作业本身照常备菜、照常由 actor 发布，链条保持连续。
            // 代价（如实）：这次写盘的物理开销不再被省掉；内存压力改由 host 额度与背压吸收。
            pipelineHeldBytes -= job->heldBytes;
            job->bytesReleased = true;
            // 保留在 FIFO（不 erase）：作业必须继续走完 PrepLoop → actor → 发布。
            if (freedBytes != nullptr) *freedBytes = job->heldBytes;
            // 4：取消必须**可观测**（旧实现成功路径零日志，导致三轮真机都无法判定取消是否发生）。
            ++cancelledJobs;
            // 1c（2026-10-03）：把现场一并打出来——真机 t1b/t1c 只看得见"丢了字节/挖了洞"，
            // 这里给出"是否已开工 / 是否已交 actor / 手上还有多少字节"三个关键位。
            std::fprintf(stderr,
                         "[Prefix SSD] v4 cancel: released quota only (job keeps running, checkpoint will "
                         "still be published) held_bytes=%llu cancelled_total=%llu started=%d posted=%d "
                         "preserve_submitted=%d staged=%zu objects=%zu key=%s.\n",
                         (unsigned long long)job->heldBytes, (unsigned long long)cancelledJobs,
                         int(job->started), int(job->posted), int(job->preserveSubmitted),
                         job->staged.size(), job->objects.size(), job->key.c_str());
            pipelineCv.notify_all();
            return true;
        }
        error = "v4_no_cancellable_pending";
        // 诊断（2026-10-01）：三次真机复跑都返回"无牺牲者"，但没留下 FIFO 现场，无法判定是
        // "队列空"还是"全部已过不可逆点"。这里把现场打出来，供下一次真机复跑定位。
        {
            size_t total = 0, committingCount = 0, startedCount = 0, cancelledCount = 0;
            for (const auto &job : pipeline) {
                if (!job) continue;
                ++total;
                if (job->committing) ++committingCount;
                else if (job->cancelled.load()) ++cancelledCount;
                else if (job->started) ++startedCount;
            }
            std::fprintf(stderr,
                         "[Prefix SSD] v4 cancel: no victim; fifo=%zu committing=%zu started=%zu "
                         "cancelled=%zu held_bytes=%llu.\n",
                         total, committingCount, startedCount, cancelledCount,
                         (unsigned long long)pipelineHeldBytes);
        }
        return false;
    }
    // §⑥ 备菜线程主体：**唯一的备菜者**。严格按受理顺序取最早的未开工作业串行执行，
    // 因此"投递顺序 = 受理顺序"（计划 §6.1/§17.1 硬约束）。
    // 关闭时也要把已受理的作业全部备完，绝不丢。
    void PrepLoop() {
        prepId = std::this_thread::get_id();
        for (;;) {
            std::shared_ptr<PrepJob> job;
            {
                std::unique_lock<std::mutex> lock(pipelineMutex);
                pipelineCv.wait(lock, [this] {
                    return pipelineStopping || NextUnstartedLocked() != nullptr;
                });
                job = NextUnstartedLocked();
                if (job) {
                    job->started = true;
                    // 恒 false 的防御分支（2026-10-05 复核 S3c）：1d 之后无任何 cancelled.store(true)，
                    // 本分支永不命中；保留只为防止旧"取消=丢检查点"语义被重新引入。
                    if (job->cancelled.load()) {
                        // §保最新：已取消 ⇒ 不备菜、不投盘。
                        // 3-b（2026-10-03，真机 r19-a1 实证补漏）：**取消绝不能丢字节**。旧实现在这里
                        // 直接丢弃作业 ⇒ 我放在 actor 认领段的"保字节"代码永远不会执行（作业从未到达
                        // actor），于是它的 KV 分片从未落盘；而模型侧已把这些 hash 累计进 state.kv
                        // （issued 已前移、该区间永不重导）⇒ 后续检查点的 root 引用悬空对象——
                        // r13/r16-a1/r17-a1/r19-a1 四轮同点丢点的根因（`v4_object_missing`：该 hash 在
                        // 40GB 控制日志里只作为 JSON 引用出现、没有对象记录）。
                        // 现在改为**把作业照常投给 actor**：actor 认领时会把暂存对象写入日志与索引
                        // （不发布 root、不写检查点记录），再按取消收尾上报。这样 pendingObjects
                        // 只在 actor 线程上被触碰，并发前提与正常提交路径完全一致。
                        std::string postError;
                        if (!SubmitPrepared(job, postError)) {
                            std::fprintf(stderr,
                                         "[Prefix SSD] v4 cancel: handoff for preservation failed: %s.\n",
                                         postError.c_str());
                        }
                        job.reset();
                    }
                }
                else if (pipelineStopping) return;
            }
            if (job) RunPrep(job);
        }
    }
    // Hydrate only retained objects referenced by the next Route-B commit.
    // Capacity failures keep bytes on failedRetainedJobs; treating them as
    // reachable without writing them would publish an unreadable root, while
    // putting all of them in pendingObjects bypasses quota admission.
    void HydrateFailedRetention(std::vector<DiskPrefixV4Store::Object> &objects,
                                std::set<std::string> &hydrated) {
        std::map<std::string, DiskPrefixV4Store::Object> retained;
        {
            std::lock_guard<std::mutex> lock(pipelineMutex);
            for (const auto &job : failedRetainedJobs) {
                if (!job) continue;
                for (const auto &item : job->staged) {
                    if (item.second.empty()) continue;
                    DiskPrefixV4Store::Object object;
                    object.hash = item.first;
                    object.kind = prefix_v4::kObjectData;
                    object.bytes = item.second;
                    retained.emplace(object.hash, std::move(object));
                }
                for (const auto &object : job->objects) {
                    if (!object.bytes.empty()) retained.emplace(object.hash, object);
                }
            }
        }
        if (retained.empty()) return;
        std::set<std::string> needed;
        std::set<std::string> ownedBytes;
        for (const auto &object : objects) {
            if (object.bytes.empty()) {
                needed.insert(object.hash);
                continue;
            }
            ownedBytes.insert(object.hash);
            if (object.kind != prefix_v4::kObjectBlock) continue;
            std::string parseError;
            const Json block = Json::parse(object.bytes, parseError);
            if (parseError.empty() && block.is_object()) {
                Refs refs;
                CollectRefs(block["kv"], refs);
                for (const auto &ref : refs) needed.insert(ref.first);
            }
        }
        for (const auto &hash : needed) {
            if (hash.empty() || ownedBytes.count(hash) || HasObject(hash) || IsPending(hash)) continue;
            const auto found = retained.find(hash);
            if (found == retained.end()) continue;
            bool replaced = false;
            for (auto &object : objects) {
                if (object.hash != hash || !object.bytes.empty()) continue;
                object.kind = found->second.kind;
                object.bytesSize = 0;
                object.bytes = found->second.bytes;
                replaced = true;
                break;
            }
            if (!replaced) objects.push_back(found->second);
            ownedBytes.insert(hash);
            hydrated.insert(hash);
        }
    }

    void ReleaseHydratedRetention(const std::set<std::string> &hydrated) {
        if (hydrated.empty()) return;
        std::lock_guard<std::mutex> lock(pipelineMutex);
        for (auto &job : failedRetainedJobs) {
            if (!job) continue;
            job->staged.erase(std::remove_if(job->staged.begin(), job->staged.end(),
                [&](const auto &item) { return hydrated.count(item.first) != 0; }),
                job->staged.end());
            job->objects.erase(std::remove_if(job->objects.begin(), job->objects.end(),
                [&](const auto &object) { return hydrated.count(object.hash) != 0; }),
                job->objects.end());
        }
        for (auto it = failedRetainedJobs.begin(); it != failedRetainedJobs.end();) {
            const auto &job = *it;
            if (job && job->staged.empty() && job->objects.empty()) {
                job->objectLease.reset();
                job->failureRetained = false;
                it = failedRetainedJobs.erase(it);
            } else {
                ++it;
            }
        }
    }

    void RunPrep(const std::shared_ptr<PrepJob> &job) {
        const uint64_t start = NowNs();
        prepQueueNs += start - job->acceptNs;
        std::string error;
        bool ok = false, noop = false;
        try {
            // ① 先收尾上一笔：预检必须能看到上一次提交的最终状态。
            std::shared_ptr<CommitResult> previous;
            {
                std::lock_guard<std::mutex> lock(pipelineMutex);
                previous = actorInFlight;
            }
            if (previous) {
                std::string previousError;
                if (!WaitResult(previous, previousError)) {
                    // 上一笔的失败由它**自己的结果槽**负责上报（下一次 reap / 停服）；
                    // 这里只打一行便于现场定位，绝不把错误改记到本笔头上。
                    std::fprintf(stderr, "[Prefix SSD] v4 prep: previous commit failed: %s.\n",
                                 previousError.c_str());
                }
            }
            if (!job->prepared) PrepareStagedJob(*job, error, noop);
            if (noop) {
                // 去重命中：内容已在库里，本笔不落盘，直接算成功（请求线程早已知情地
                // 返回了"已受理"）。
                std::lock_guard<std::mutex> lock(job->result->mutex);
                // The identical state already owns these content-addressed
                // bytes. Clear the duplicates before publishing completion:
                // otherwise reap reposts them and overwrites success as a
                // v4_preserved_not_published failure on restart/replay.
                job->staged.clear();
                job->objects.clear();
                ++reusedCheckpoints;
                job->result->ok = true;
                job->result->Acknowledge(true);
                job->result->finished = true;
                job->result->done.notify_all();
                ok = true;
            } else {
                // §保最新：投盘是**不可逆**的，所以必须在同一把锁下先"认领"（posted=true），
                // 让"取消"与"认领"互斥——否则会出现"已取消却仍被投盘"的中间态（实测踩中：
                // 取消返回 true、读侧也报了 v4_cancelled_oldest，但三笔全被持久化）。
                {
                    std::lock_guard<std::mutex> lock(pipelineMutex);
                    if (job->cancelled.load()) throw std::runtime_error("v4_cancelled_oldest");
                    job->posted = true;
                }
                ok = SubmitPrepared(job, error);
            }
        } catch (const std::exception &e) {
            error = e.what(); ok = false;
        } catch (...) { error = "v4_prep_exception"; ok = false; }
        if (ok && job->cancelled.load()) {   // 恒 false 的防御分支（1d 后无人写 cancelled）；保留防旧语义复活
            ok = false;
            error = "v4_cancelled_oldest";
        }
        if (ok) {
            std::lock_guard<std::mutex> lock(pipelineMutex);
            if (job->cancelled.load()) {     // 恒 false 的防御分支（与已废弃的取消互斥）；保留防旧语义复活
                ok = false;
                error = "v4_cancelled_oldest";
            } else {
                job->posted = true;
                actorInFlight = job->result;
            }
        }
        if (!ok) {
            // 备菜阶段失败：写进受理时创建的结果槽（失败绝不静默）。
            std::lock_guard<std::mutex> lock(job->result->mutex);
            if (!job->result->finished) {
                ++failedCheckpoints;
                job->result->ok = false;
                job->result->error = error.empty() ? "v4_prep_failed" : error;
                job->result->Acknowledge(false);
                job->result->finished = true;
                job->result->done.notify_all();
            }
        }
        ++prepJobs;
        prepExecNs += NowNs() - start;
    }
    // §⑥ 备菜：状态 Json →（描述符, 对象列表）。**全部在备菜线程上跑**，请求线程不参与。
    // 顺序与旧请求线程实现一致：收尾上一笔 → 引用遍历 → 序列化 → 摘要 → 描述符 →
    // 预检扫描（需要 stateHash，故必须在摘要之后）→ 装对象 + root 清单登记。
    void PrepareStagedJob(PrepJob &job, std::string &error, bool &noop) {
        noop = false;
        error.clear();
        const Json &value = job.state;
        const std::string kind = value["kind"].is_null() ? "prefix" : value["kind"].string_value();
        const std::string family = value["family_identity"].is_null()
            ? job.family : value["family_identity"].string_value();
        Check((kind=="prefix" || kind=="encoder") && DiskPrefixCache::IsDigest(family),
              "invalid_checkpoint_metadata");
        // 注意：引用合法性 / raw tokens 的校验已在请求线程同步做过（必须当场拒绝），
        // 这里不再重复遍历（§17.1 规则不变，只是把"拒绝"留在调用方能看到的线程上）。
        //
        // Route B（2026-10-02）：把"从开头到本点的全部 KV 分块描述"按 256-token 块拆成
        // **每条一块**的内容寻址对象；状态对象只保留标量 + chain_tip + 块数/token 数 +
        // linear/spec 引用，于是单对象大小不再随上下文增长（旧形态就是那个 177.75 B/token
        // 的大 state 对象）。编码器（无 kv 数组）与已装配对象路径保持旧扁平语义不变。
        RouteBChain chain;
        Json compactValue = value;
        if (!value["kv"].is_null()) {
            Check(value["kv"].is_array(), "v4_block_kv_type");
            const Json::array &kv = value["kv"].array_items();
            if (RouteBEligible(kv)) {
                std::string blockError;
                // 注意：BuildBlockChain 必须**先求值**再取错误串——同一函数调用的实参求值
                // 顺序是未指定的，写成 Check(BuildBlockChain(...), blockError.empty()?...) 会
                // 在部分编译器上先算第二个实参、拿到空错误的兜底串（实测 GCC 即如此）。
                const bool chainOk = BuildBlockChain(kv, chain, blockError);
                Check(chainOk, blockError.empty() ? "v4_block_chain" : blockError.c_str());
                auto fields = value.object_items();
                fields.erase("kv");
                fields["chain_tip"] = chain.tip;
                fields["block_count"] = double(chain.blockCount);
                fields["chained_tokens"] = double(chain.chainedTokens);
                fields["block_tokens"] = double(chain.blockTokens);
                compactValue = Json(fields);
            }
        }
        const auto encodeStart = std::chrono::steady_clock::now();
        const std::string stateText = MetadataText(compactValue);
        // S1-2 修复（2026-10-02，对标审计 P0-1）：**写入侧必须与解码侧对称校验**。
        // 真机实证（random-seg r7，catalog 只读查询）：28 个 state 对象（kind=4）超过解码侧
        // 8 MiB 上限（最大 13,833,448 B ≈ 13.2 MiB）⇒ 检查点"发布成功但读不回"
        // （Load 侧在 v4_object_list_bytes 处直接拒绝），只在运行期表现为 ++failed 与 CACHE-WARN。
        // 这里在**发布前**就 fail-closed：宁可显式丢点，也绝不发布一个读不回来的检查点。
        // Route B 之后状态正文已是 O(1)（标量 + 链尾键），本检查自然长期成立。
        Check(stateText.size() <= prefix_v4::kMaxObjectBytes, "v4_state_exceeds_decode_limit");
        const uint64_t encodeNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - encodeStart).count());
        const auto hashStart = std::chrono::steady_clock::now();
        const std::string stateHash = DiskPrefixCache::Digest(stateText);
        const uint64_t hashNs = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - hashStart).count());
        owner->AddMetadataTiming(0, encodeNs, hashNs, stateText.size());
        Json::object descriptorFields = Json::object{{"family",family},{"kind",kind},
            {"length",double(value["length"].int_value())},{"dtype",FieldString(value,"kv_dtype")},
            {"provenance",FieldString(value,"provenance")},{"prefix",value["prefix"].string_value()},
            {"created",double(NowNs())},{"state",stateHash}};
        if (chain.enabled) {
            descriptorFields["chain_tip"] = chain.tip;
            descriptorFields["block_count"] = double(chain.blockCount);
            descriptorFields["chained_tokens"] = double(chain.chainedTokens);
            descriptorFields["block_tokens"] = double(chain.blockTokens);
        }
        const std::string descriptor = Json(descriptorFields).dump();
        // 预检扫描：只走 Bare 查询——Checkpoints() 会先 DrainForRead，而队首正是本笔
        // （备菜线程等自己 = 自锁，与 ERR 230 同类）。
        const auto scanStart = std::chrono::steady_clock::now();
        for (const auto &row : CheckpointsBare()) {
            if (row.key != job.key || row.identity != job.identity) continue;
            std::string parseError; const auto previous = Json::parse(row.metadata,parseError);
            if (!parseError.empty()) Check(false,"cache_metadata_descriptor");
            if (previous["state"].string_value() == stateHash) { noop = true; return; }
            const bool equivalent = job.keepEquivalent &&
                previous["kind"].string_value()==kind && previous["family"].string_value()==family &&
                previous["dtype"].string_value()==FieldString(value,"kv_dtype") &&
                previous["provenance"].string_value()==FieldString(value,"provenance") &&
                previous["prefix"].string_value()==value["prefix"].string_value() &&
                previous["length"].int_value()==value["length"].int_value();
            Check(equivalent,"checkpoint_key_conflict");
        }
        owner->AddPrecheckTiming(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - scanStart).count()));
        std::vector<Object> objects;
        objects.reserve(job.staged.size() + chain.objects.size() + 1);
        for (auto &chunk : job.staged) {
            Object object;
            object.hash = chunk.first; object.kind = prefix_v4::kObjectData;
            object.bytes = std::move(chunk.second);
            objects.push_back(std::move(object));
        }
        job.staged.clear();
        // Route B：块对象进 objects 列表；root 将直接指向状态对象（不再有随上下文变长的
        // ObjectList 清单），可达性由 blocks 链索引 + 状态里的引用决定。
        for (const auto &block : chain.objects) {
            const std::string blockText = block.dump_compact();
            Object object;
            object.hash = DiskPrefixCache::Digest(blockText);
            object.kind = prefix_v4::kObjectBlock;
            object.bytes = blockText;
            objects.push_back(std::move(object));
        }
        Object stateObject;
        stateObject.hash = stateHash; stateObject.kind = prefix_v4::kObjectStateNode;
        stateObject.bytes = stateText;
        objects.push_back(std::move(stateObject));
        // P0 修复（2026-09-30，**本段必须保留、不得回退**）：root 清单必须登记"状态 JSON
        // 引用的**全部**对象"，包括被去重（已存在于库里）的那些。此前只登记本次新暂存的
        // 对象 → 第 2 个及以后检查点的清单不完整（实测清单恒 80 条，而状态引用 79+64k 条）
        // → 重开后 Load 取不回全部对象 → invalid_chunk_reference → 该检查点永远不可用
        // （缓存只命中 2048）。
        // Route B 下这段仍然必要：块对象引用的 KV 数据对象（本次新写的 + 被去重只引用的）
        // 都必须登记，否则重开后链能走通、块却取不回。
        {
            std::set<std::string> staged;
            for (const auto &object : objects) staged.insert(object.hash);
            std::vector<std::pair<std::string,uint64_t>> references;
            std::function<void(const Json &)> collect = [&](const Json &node) {
                if (node.is_array()) {
                    for (const auto &item : node.array_items()) collect(item);
                } else if (node.is_object()) {
                    const Json &sha = node["sha256"];
                    const Json &bytes = node["bytes"];
                    if (sha.is_string() && bytes.is_number()) {
                        references.emplace_back(sha.string_value(), uint64_t(bytes.number_value()));
                    }
                    for (const char *field : {"chunks","kv","linear","speculative","linear_chunks","speculative_chunks"}) {
                        const Json &child = node[field];
                        if (!child.is_null()) collect(child);
                    }
                }
            };
            for (const char *field : {"kv","linear","speculative","linear_chunks","speculative_chunks"}) {
                // Route B：kv 分块的可达性由块对象承载（Load / Compact / SelfCheck 逐块解析并
                // 惰性校验），不再把 O(context) 条"只引用"条目塞进每次提交——这是"每检查点
                // O(新增块)"而不是 O(context) 的关键。linear/spec 是独立小对象，仍必须登记。
                if (chain.enabled && std::strcmp(field, "kv") == 0) continue;
                const Json &child = value[field];
                if (!child.is_null()) collect(child);
            }
            for (const auto &reference : references) {
                if (reference.first.empty() || staged.count(reference.first)) continue;
                Object entry;
                entry.hash = reference.first;
                entry.kind = prefix_v4::kObjectData;
                entry.bytesSize = reference.second;    // 只引用：不重写字节
                objects.push_back(std::move(entry));
                staged.insert(reference.first);
            }
        }
        job.objects = std::move(objects);
        job.metadata = descriptor;
    }
    // §⑥ 投递给**唯一的写盘 actor**（不等待）：结果写进受理时创建的 CommitResult。
    // lambda 内必须自己兜住异常，否则 PostActor 会吞掉异常且结果永远不完成 → reap 死等。
    bool SubmitPrepared(const std::shared_ptr<PrepJob> &job, std::string &error) {
        const std::shared_ptr<CommitResult> result = job->result;
        const uint64_t submitStart = NowNs();
        const bool posted = PostActor([this, job, result] {
            bool ok = false;
            std::string localError;
            uint64_t sequence = 0;
            // §保最新（2026-10-01）：写第一条记录之前是**唯一**可取消的窗口。在同一把
            // pipelineMutex 下认领（committing=true）或接受取消——与 CancelOldestPending
            // 互斥，保证"要么被取消、要么提交"，不存在"已取消却已落盘"的中间态。
            {
                std::lock_guard<std::mutex> claimLock(pipelineMutex);
                // S3：**"仍持有 staged 字节"的作业必须在这里保字节，而不是被静默丢弃**。
                // 复核证据（R3）：备菜成功会把 `staged` 装配进 `objects` 并清空（`:3040-3047`），
                // 所以"到达 actor 时 staged 非空"只可能是**备菜未跑完/中途抛异常**的作业；
                // 而 1d 之后 `cancelled` 已无人写入 ⇒ 旧分支成了死代码，这类作业的字节会被丢，
                // 且出队侧的 `preserved_on_drop` 计数器会**谎报成功**（它只验证投递成功）。
                // 因此把条件放宽为"取消 或 仍持有未装配的 staged"，并按"已保字节但不发布"收尾。
                const bool preserveOnly = !job->staged.empty();
                // 恒 false 的防御分支：`cancelled` 在 1d 之后无人写入，这里实际只由 preserveOnly 触发；
                // 保留 `cancelled` 项只为防御旧语义复活（取消=丢字节）。
                if (job->cancelled.load() || preserveOnly) {
                    // 3：**取消 ≠ 丢弃字节**。模型侧已把这些分片的
                    // hash 累计进 state.kv、并前移了 issued（该区间永不重导）；若这里"一条记录都不写"，
                    // 后续检查点的 root 就会引用从未落盘的对象——真机 r13/r16-a1/r17-a1 三轮同点丢弃
                    // 的根因正是它（v4_object_missing → 整段重拷 288MiB > 预算 256MiB ⇒ 必丢）。
                    // 故取消时仍把**暂存对象**写入日志与索引（不发布 root、不写检查点记录），
                    // 只丢弃"检查点"本身。本段在 actor 线程执行，pendingObjects 的并发前提与正常提交一致。
                    std::string preserveError;
                    bool preserved = false;
                    auto preserve = [&](const std::string &hash, uint32_t kind, std::string &&bytes) {
                        if (bytes.empty()) return;
                        if (HasObject(hash) || IsPending(hash)) return;
                        pendingBytes += bytes.size();
                        pendingHashSet.insert(hash);
                        cancelledPreservedBytes += bytes.size();
                        pendingObjects.push_back(PendingObject{hash, kind, std::move(bytes)});
                        preserved = true;
                    };
                    // S3：先把"待保字节的规模"记下来（下面会 move/clear，之后就读不到了）。
                    const size_t stagedCount = job->staged.size();
                    const size_t objectsCount = job->objects.size();
                    for (auto &item : job->staged) preserve(item.first, prefix_v4::kObjectData, std::move(item.second));
                    job->staged.clear();
                    for (auto &object : job->objects) preserve(object.hash, object.kind, std::move(object.bytes));
                    job->objects.clear();
                    bool preserveFlushFailed = false;
                    bool preserveFlushOk = !preserved || FlushObjects(preserveError);
#ifdef FASTLLM_DISK_PREFIX_CACHE_TESTING
                    // D5-b2：**确定性**注入"保字节写盘失败"，
                    // 供主#7 断言"S3b 的失败不再被吞"。为什么不用 LD_PRELOAD 注入器：它只 hook
                    // write/fdatasync 且靠 `INJECT_SKIP` 按**调用次数**校准（非按 callsite），确定性弱；
                    // 这里直接把本次判为失败，并走下面**同一条**真实失败分支（计数、日志、错误串都不变）。
                    if (preserveFlushOk && preserved &&
                        std::getenv("FASTLLM_DISK_CACHE_FAIL_PRESERVE_FLUSH") != nullptr) {
                        preserveFlushOk = false;
                        preserveError = "v4_preserve_flush_failed_injected";
                    }
#endif
                    if (preserved && !preserveFlushOk) {
                        // S3b：保字节**必须闭环**——写盘失败不能只打一行日志
                        // 就当保住了。这里置硬失败标志：既不把它当"良性"，也让它可计数、可审计。
                        preserveFlushFailed = true;
                        ++preserveFlushFailures;
                        std::fprintf(stderr, "[Prefix SSD] v4 PRESERVE-FLUSH-FAILED: %s key=%s "
                                             "preserve_flush_failures=%llu.\n",
                                     preserveError.c_str(), job->key.c_str(),
                                     (unsigned long long)preserveFlushFailures);
                    }
                    // S3：把"实际保下来的字节"如实打印出来（供真机核对计数器是否兑现）。
                    if (preserveOnly || preserved) {
                        std::fprintf(stderr,
                                     "[Prefix SSD] v4 preserved-not-published: key=%s bytes=%llu staged=%zu "
                                     "objects=%zu (%s).\n",
                                     job->key.c_str(), (unsigned long long)cancelledPreservedBytes,
                                     stagedCount, objectsCount,
                                     preserveOnly ? "unprepared-staged" : "cancelled");
                    }
                    // claimLock already owns pipelineMutex here; the helper
                    // deliberately assumes that lock and must not reacquire
                    // it on this terminal preservation path.
                    ReleaseFailureRetentionIfEmptyLocked(job);
                    std::lock_guard<std::mutex> doneLock(result->mutex);
                    ++failedCheckpoints;
                    result->finished = true;
                    result->ok = false;
                    result->Acknowledge(false);
                    // S3：不再一律报 v4_cancelled_oldest（1d 后已无取消）；未备菜的作业用专用错误串，
                    // 由流水线两处按"良性"处理，避免把失败传染给无关请求。
                    result->error = preserveFlushFailed ? "v4_preserve_flush_failed"
                                   : (preserveOnly ? "v4_preserved_not_published" : "v4_cancelled_oldest");
                    result->done.notify_all();
                    return;
                }
                job->committing = true;
            }
            // P3 真机测口径（2026-10-01）：本次提交的对象字节（打印用）。必须在 move 之前算。
            uint64_t objectsBytes = 0;
            // 主#9 加固：`owned` 必须活在 try 之外。
            // 机理（T0 探针 [C] 用例已确定性复现 missing=2）：CommitCheckpointLocked 在
            // "字节已移交、尚未写盘"时有多处早返回/抛错（:4150 v4_block_missing、
            // :4178 v4_object_missing、:4244 v4_block_state_missing、:4253-4254 Check、
            // :4257 WriteObject）。旧写法把 owned 声明在 try 内 ⇒ 早返回时**尚未写出的新字节
            // 随作用域销毁**，而 job->objects 已被 move 空 ⇒ 出队侧 PreserveBeforeDropLocked
            // 因 "staged/objects 皆空"(:2659) 既不打印也不保字节 ⇒ **静默丢字节**；模型侧
            // 早已把这些分片计入累计 state.kv（issued 已前移、永不重导）⇒ 后续检查点全部引用
            // 悬空对象 ⇒ 被发布闸门扣下 ⇒ 自激 write_skipped。
            // 修法：提交失败时把"有字节且既不在 objects 表也不在待写表"的对象抢救回
            // pendingObjects 并 Flush；**检查点仍不发布**（保持 ok=false）——语义是
            // "字节必须落地，记录可以不放行"。
            std::vector<Object> owned;
            std::set<std::string> hydratedRetention;
            try {
                for (const auto &object : job->objects) objectsBytes += object.bytes.size() + object.bytesSize;
                owned = std::move(job->objects);
                HydrateFailedRetention(owned, hydratedRetention);
                ok = owner->CommitCheckpointLocked(job->key, job->identity, owned, job->metadata,
                                                   &sequence, localError);
                if (ok) ReleaseHydratedRetention(hydratedRetention);
            } catch (const std::exception &e) { localError = e.what(); ok = false; }
            catch (...) { localError = "v4_commit_exception"; ok = false; }
            if (!ok && localError == "capacity_unavailable") {
                // Admission failure is a terminal failure, not permission to
                // write the unreserved payload through the rescue path. Keep
                // bytes with this failed job in RAM; do not put them in the
                // global pending queue where they could poison a later small
                // commit. No checkpoint or durable acknowledgement is issued.
                for (auto &pending : pendingObjects) {
                    Object retained;
                    retained.hash = pending.hash;
                    retained.kind = pending.kind;
                    retained.bytes = std::move(pending.bytes);
                    owned.push_back(std::move(retained));
                }
                pendingObjects.clear();
                pendingHashSet.clear();
                pendingBytes = 0;
                job->objects = std::move(owned);
                job->preserveSubmitted = true;
                job->objectLease.reset();
                std::fprintf(stderr,
                    "[Prefix SSD] v4 capacity-unavailable: key=%s "
                    "checkpoint withheld without rescue flush.\n", job->key.c_str());
            } else if (!ok) {
                size_t rescued = 0;
                uint64_t rescuedBytes = 0;
                for (auto &object : owned) {
                    if (object.bytes.empty()) continue;                 // 只引用：库里已有
                    if (HasObject(object.hash) || IsPending(object.hash)) continue;
                    pendingBytes += object.bytes.size();
                    pendingHashSet.insert(object.hash);
                    rescuedBytes += object.bytes.size();
                    pendingObjects.push_back(PendingObject{object.hash, object.kind,
                                                           std::move(object.bytes)});
                    ++rescued;
                }
                bool rescueCompleted = (rescued == 0);
                if (rescued > 0) {
                    std::string rescueError;
                    if (!FlushObjects(rescueError)) {
                        ++preserveFlushFailures;
                        std::fprintf(stderr,
                                     "[Prefix SSD] v4 RESCUE-FLUSH-FAILED: %s key=%s rescued=%zu "
                                     "bytes=%llu preserve_flush_failures=%llu.\n",
                                     rescueError.c_str(), job->key.c_str(), rescued,
                                     (unsigned long long)rescuedBytes,
                                     (unsigned long long)preserveFlushFailures);
                    } else {
                        std::fprintf(stderr,
                                     "[Prefix SSD] v4 rescued-uncommitted-bytes: key=%s objects=%zu "
                                     "bytes=%llu (checkpoint NOT published).\n",
                                     job->key.c_str(), rescued, (unsigned long long)rescuedBytes);
                        rescueCompleted = true;
                    }
                }
                // A terminally failed checkpoint has no live root. Once its
                // owned bytes are either already durable or successfully
                // rescued into the canonical object log, its request lease
                // must not keep those unreferenced objects alive forever:
                // Compact must be able to reclaim them before the next
                // physical-quota admission. Keep the lease only when the
                // rescue write itself failed.
                if (rescueCompleted) job->objectLease.reset();
            }
            // S2：**被扣下的检查点不得伪装成"已落盘"**。
            // 复核实证（t1b server.log:824/890）：旧写法在这里 ok=true 收尾，于是控制流落进下面
            // "仅成功时打印"的分支，打出 `v4-durable … seq=0`；而验收脚本（ssd_write_acceptance.py）
            // 只按 `v4-durable` 正则计数 ⇒ **把丢点算成落盘**。现在改为：
            //   · 保持 ok=false，错误串换成专用 `v4_publish_withheld`；
            //   · 结果上置 withheld 标志；durable 打印因此不会发生，seq 也不会被当有效值消费；
            //   · 该错误串在流水线两处（EnqueuePipeline 背压出队、ReapPending 收尾）按"良性"处理，
            //     错误原因仍如实打印，但不会以异常形式传染给无关请求。
            if (!ok && localError == "v4_publish_dangling_ref") {
                std::fprintf(stderr,
                             "[Prefix SSD] v4 checkpoint withheld (dangling kv ref, NOT durable): key=%s "
                             "refused_total=%llu.\n",
                             job->key.c_str(),
                             (unsigned long long)publishRefusedCheckpoints);
                localError = "v4_publish_withheld";
                result->withheld = true;
            }
            // 真正的"落盘完成"时刻：仅成功时打印；不持任何锁，且早于取 result->mutex。
            if (ok && PrefixCacheDebugEnabled()) {
                struct timespec wallTs {};
                clock_gettime(CLOCK_REALTIME, &wallTs);
                struct tm wallTm {};
                localtime_r(&wallTs.tv_sec, &wallTm);
                char wallText[40];
                std::snprintf(wallText, sizeof wallText,
                              "%04d-%02d-%02dT%02d:%02d:%02d.%03d",
                              wallTm.tm_year + 1900, wallTm.tm_mon + 1, wallTm.tm_mday,
                              wallTm.tm_hour, wallTm.tm_min, wallTm.tm_sec,
                              int(wallTs.tv_nsec / 1000000));
                const double acceptToDurableMs = double(NowNs() - job->acceptNs) / 1e6;
                PrefixCacheDebug("[Prefix SSD] v4-durable: key=%s seq=%llu objects_bytes=%llu accept_to_durable_ms=%.2f wall=%s\n",
                            job->key.c_str(), (unsigned long long)sequence,
                            (unsigned long long)objectsBytes, acceptToDurableMs, wallText);
            }
            std::lock_guard<std::mutex> lock(result->mutex);
            if (!ok) ++failedCheckpoints;
            result->ok = ok;
            result->error = localError;
            result->sequence = sequence;
            result->Acknowledge(ok);
            result->finished = true;
            result->done.notify_all();
        }, error);
        commitSubmitNs += NowNs() - submitStart;
        return posted;
    }
    // 预检专用：不带读侧收尾的 Checkpoints（避免备菜线程等自己）。
    std::vector<DiskPrefixV4Store::CheckpointRow> CheckpointsBare() const {
        std::vector<DiskPrefixV4Store::CheckpointRow> rows;
        Statement query(database->value,
                        "SELECT c.identity,c.key,c.metadata,c.sequence FROM checkpoints c "
                        "JOIN roots r ON r.identity=c.identity AND r.key=c.key WHERE r.live=1");
        while (query.Row()) {
            DiskPrefixV4Store::CheckpointRow row;
            row.identity = query.Text(0);
            row.key = query.Text(1);
            row.metadata = query.Text(2);
            row.sequence = query.Number(3);
            rows.push_back(std::move(row));
        }
        return rows;
    }
    // §⑥ 关闭收尾：先把备菜线程上"已受理但还没投递"的作业全部备完并投递，再等全部结果。
    void StopPipeline() {
        {
            std::lock_guard<std::mutex> lock(pipelineMutex);
            pipelineStopping = true;
        }
        pipelineCv.notify_all();
        if (prep.joinable()) prep.join();
        std::string drainError;
        if (!ReapPending(drainError, nullptr)) {
            std::fprintf(stderr, "[Prefix SSD] v4 pending commit failed at stop: %s.\n",
                         drainError.c_str());
        }
    }
    // §④ 读侧收尾：读操作必须看到"已受理"的提交，否则会出现"提交成功却读不到"。
    // 读路径无法处置写失败，故只打印（可见），写路径的下一次收尾还会再报一次。
    // **必须排除 actor 线程与备菜线程本身**：它们可能正是那笔在途提交的执行者，
    // 等自己 = 自锁（CommitBareLocked 内部会调用 LiveBytes()，已实测踩中）。
    void DrainForRead() {
        if (std::this_thread::get_id() == actorId) return;
        if (std::this_thread::get_id() == prepId) return;
        std::string drainError;
        if (!ReapPending(drainError, nullptr)) {
            std::fprintf(stderr, "[Prefix SSD] v4 pending commit failed (drain on read): %s.\n",
                         drainError.c_str());
        }
    }

    // S2-4：SQLite 的 CREATE TABLE IF NOT EXISTS 不会给**已存在**的旧表补列。旧 v4
    // catalog 的 checkpoints 表只有 (identity,key,metadata,sequence)，而新查询/索引依赖
    // family/kind/length/dtype/provenance/prefix/created。若不补列，紧接着的 CREATE INDEX
    // 与后续 SELECT 会直接 "no such column" 报错——而且这个报错发生在 Open 的版本闸门
    // 之前，会先于自动升级把缓存打死。这里逐列检测、缺哪列补哪列（已有列跳过），
    // 保证旧 catalog 在闸门触发重建前也绝不 SQL 报错。
    static void EnsureCheckpointColumn(Database &db, const char *name, const char *definition) {
        {
            Statement query(db.value, "PRAGMA table_info(checkpoints)");
            while (query.Row()) {
                if (query.Text(1) == name) return;   // 列已存在
            }
        }
        db.Exec((std::string("ALTER TABLE checkpoints ADD COLUMN ") + name + " " +
                 definition).c_str());
    }
    static void CreateSchema(Database &db) {
        db.Exec(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY,kind INTEGER NOT NULL,"
            "bytes INTEGER NOT NULL,offset INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS roots(identity TEXT NOT NULL,key TEXT NOT NULL,"
            "root TEXT NOT NULL,sequence INTEGER NOT NULL,bytes INTEGER NOT NULL,"
            "live INTEGER NOT NULL,PRIMARY KEY(identity,key));"
            "CREATE INDEX IF NOT EXISTS roots_live ON roots(live,sequence);"
            // Checkpoint descriptors and durable key/value metadata live in the
            // control log as METADATA records; these tables are only the
            // projection that RebuildIndex restores.
            // S2-4（对标业界：v3 的 checkpoints 表本来就有 family/kind/length/prefix
            // 等索引列，v4 一度只存 metadata JSON，查询被迫全表遍历 + 逐条 Json::parse）：
            // 这里把描述符里的关键字段**投影成列**并建索引，让 CheckpointLengths /
            // ListCheckpoints / Descriptor 都能走 SQL 与索引。投影可随时从 control.log
            // 重建，故新增列不涉及数据迁移（catalog_schema 闸门会强制旧库重建）。
            "CREATE TABLE IF NOT EXISTS checkpoints(identity TEXT NOT NULL,key TEXT NOT NULL,"
            "metadata TEXT NOT NULL,sequence INTEGER NOT NULL,"
            "family TEXT NOT NULL DEFAULT '',kind TEXT NOT NULL DEFAULT '',"
            "length INTEGER NOT NULL DEFAULT 0,dtype TEXT NOT NULL DEFAULT '',"
            "provenance TEXT NOT NULL DEFAULT '',prefix TEXT NOT NULL DEFAULT '',"
            "created INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(identity,key));"
            // Route B：按块索引（每块一条）。权威源仍是控制日志里的块对象
            // （kind=kObjectBlock，内容自带 chain_key/parent/token 区间/object_hash 推导），
            // 本表只是可随时重建的投影。chain_key 本身就是内容寻址的（含块内容摘要），
            // 故全局唯一，不按 identity 区分——同一块被不同检查点复用时只需一行。
            "CREATE TABLE IF NOT EXISTS blocks(chain_key TEXT PRIMARY KEY,"
            "identity TEXT NOT NULL DEFAULT '',key TEXT NOT NULL DEFAULT '',"
            "block_index INTEGER NOT NULL,token_start INTEGER NOT NULL,token_end INTEGER NOT NULL,"
            "parent TEXT NOT NULL,object_hash TEXT NOT NULL,bytes INTEGER NOT NULL);"
            "CREATE INDEX IF NOT EXISTS blocks_object ON blocks(object_hash);"
            "CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY,value TEXT NOT NULL);");
        // 旧表补列（顺序无关；新库/重建库这些列已存在，直接跳过）。
        EnsureCheckpointColumn(db, "family", "TEXT NOT NULL DEFAULT ''");
        EnsureCheckpointColumn(db, "kind", "TEXT NOT NULL DEFAULT ''");
        EnsureCheckpointColumn(db, "length", "INTEGER NOT NULL DEFAULT 0");
        EnsureCheckpointColumn(db, "dtype", "TEXT NOT NULL DEFAULT ''");
        EnsureCheckpointColumn(db, "provenance", "TEXT NOT NULL DEFAULT ''");
        EnsureCheckpointColumn(db, "prefix", "TEXT NOT NULL DEFAULT ''");
        EnsureCheckpointColumn(db, "created", "INTEGER NOT NULL DEFAULT 0");
        // 索引必须在上面的补列之后创建，否则旧 catalog 上会 "no such column"。
        db.Exec(
            "CREATE INDEX IF NOT EXISTS checkpoints_lookup ON checkpoints(kind,family,length);"
            "CREATE INDEX IF NOT EXISTS checkpoints_prefix ON checkpoints(kind,family,prefix,length);");
    }
    void Schema() { CreateSchema(*database); }
    // S2-4：把描述符 JSON 解析**一次**并写入 checkpoints 的全部投影列。全仓 4 处 v4 的
    // "INSERT INTO checkpoints" 都必须走这里；漏一处会导致重建后索引列为空、按
    // kind/family/prefix 过滤的查询静默 miss（本任务最高风险点）。
    static void InsertCheckpointRow(Database &db, const std::string &identity,
                                    const std::string &key, const std::string &metadata,
                                    uint64_t sequence) {
        std::string parseError;
        const Json descriptor = Json::parse(metadata, parseError);
        // 描述符由本模块写入，解析失败只可能是记录损坏；宁可 fail-closed 显式报错，
        // 也不写一行"列为空"的投影让查询静默 miss。
        Check(parseError.empty(), "cache_metadata_descriptor");
        Statement insert(db.value,
            "INSERT OR REPLACE INTO checkpoints(identity,key,metadata,sequence,"
            "family,kind,length,dtype,provenance,prefix,created) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)");
        insert.Text(1, identity);
        insert.Text(2, key);
        insert.Text(3, metadata);
        insert.Number(4, sequence);
        insert.Text(5, descriptor["family"].string_value());
        insert.Text(6, descriptor["kind"].string_value());
        insert.Number(7, uint64_t(std::max(0, descriptor["length"].int_value())));
        insert.Text(8, descriptor["dtype"].string_value());
        insert.Text(9, descriptor["provenance"].string_value());
        insert.Text(10, descriptor["prefix"].string_value());
        // created 是纳秒级 double，必须走 number_value()；int_value() 会溢出截断。
        const double created = descriptor["created"].number_value();
        insert.Number(11, created > 0.0 ? uint64_t(created) : 0);
        // 必须真正执行：Statement 的析构只做 finalize，不 step。漏掉这一行会让
        // checkpoints 永远为空，而 roots 照常写 ⇒ 读侧点查全 miss（实测 v4_root_missing）。
        insert.Run();
    }
    void SetMeta(const std::string &key, const std::string &value) {
        Statement put(database->value, "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)");
        put.Text(1, key); put.Text(2, value); put.Run();
    }
    // Route B：把一条块行写进 catalog 投影（提交/重建/压缩三条路径共用同一口径）。
    static void InsertBlockRow(Database &db, const DiskPrefixV4Store::BlockRow &row) {
        Statement insert(db.value,
            "INSERT OR REPLACE INTO blocks(chain_key,identity,key,block_index,token_start,"
            "token_end,parent,object_hash,bytes) VALUES(?,?,?,?,?,?,?,?,?)");
        insert.Text(1, row.chainKey);
        insert.Text(2, row.identity);
        insert.Text(3, row.key);
        insert.Number(4, row.index);
        insert.Number(5, row.tokenStart);
        insert.Number(6, row.tokenEnd);
        insert.Text(7, row.parent);
        insert.Text(8, row.objectHash);
        insert.Number(9, row.bytes);
        insert.Run();
    }
    // §17.1（2026-09-30 修复③）：控制日志长度 = "投影水位"。凡往 **live control.log** 追加
    // 记录之后，都要在同一步刷新水位；否则下次 Open 会误判"投影落后"而白扫一遍全日志。
    // 反向：崩溃点落在"记录已同步、本事务未提交"时水位故意停在上一笔 → Open 检测到落后，
    // 自动从日志重建（这正是修复③要抓的窗口）。
    void RefreshLogWatermark() {
        std::error_code sizeError;
        const uint64_t logBytes = segmented ? SegmentEndOffset() : (fs::exists(logPath)
            ? uint64_t(fs::file_size(logPath, sizeError)) : 0);
        if (!sizeError) SetMeta("log_bytes", std::to_string(logBytes));
    }
    std::string GetMeta(const std::string &key) {
        Statement query(database->value, "SELECT value FROM meta WHERE key=?");
        query.Text(1, key);
        return query.Row() ? query.Text(0) : "";
    }
    bool HasObject(const std::string &hash, sqlite3 *catalog = nullptr) {
        Statement query(catalog != nullptr ? catalog : database->value,
                        "SELECT 1 FROM objects WHERE hash=?");
        query.Text(1, hash);
        return query.Row();
    }
    bool ObjectBytes(const std::string &hash, std::string &out, std::string &error,
                     sqlite3 *catalog = nullptr) {
        Statement query(catalog != nullptr ? catalog : database->value,
                        "SELECT kind,bytes,offset FROM objects WHERE hash=?");
        query.Text(1, hash);
        if (!query.Row()) { error = "v4_object_missing"; return false; }
        // 根因修复：改用记录级读缓存——同一条 16 MiB 记录内的多个
        // 对象不再各自 open+pread+整条 SHA-256（实测自检 58 检查点累计 rchar ≈220 GB，而库
        // 实际只有 ~11 GB）。语义完全不变：仍校验记录摘要与对象摘要。
        fs::path recordPath;
        uint64_t recordOffset = 0;
        if (!LogLocation(uint64_t(query.Number(2)), recordPath, recordOffset, error)) return false;
        const std::string *payloadPtr = V4ReadRecordCached(recordPath, recordOffset, error);
        if (payloadPtr == nullptr) return false;
        return ExtractObjectFromPayload(*payloadPtr, hash, out, error);
    }
    // A1b：改用**零拷贝的 View 变体** —— 原来每取一个对象都要
    // DecodeObjectBatchPayload 把整条记录 region 拷一遍（T0 实测 520 次 ≈ 3.9 GB memcpy +
    // 约 98 万次缺页）。现在只解析对象表拿到区域偏移，再**只**把目标对象拷出来。
    // 校验与错误串完全不变：仍逐对象重算摘要（fail-closed），未匹配仍报 v4_object_mismatch。
    bool ExtractObjectFromPayload(const std::string &payload, const std::string &hash,
                                  std::string &out, std::string &error) const {
        std::vector<ObjectEntry> entries;
        uint64_t regionStart = 0, regionBytes = 0;
        if (!DecodeObjectBatchPayloadView(payload.data(), payload.size(), entries,
                                          regionStart, regionBytes, error)) {
            return false;
        }
        return ExtractObjectFromView(payload, entries, regionStart, regionBytes, hash, out, error);
    }
    // 一条记录只解码一次，按对象逐个取字节（2026-10-17，B21 记录级预取复用）。
    bool ExtractObjectFromView(const std::string &payload, const std::vector<ObjectEntry> &entries,
                               uint64_t regionStart, uint64_t regionBytes,
                               const std::string &hash, std::string &out, std::string &error) const {
        for (const auto &entry : entries) {
            if (entry.hash != hash) continue;
            // DecodeObjectTable 已保证 offset/bytes 落在 region 内；这里再兜一道，越界即 fail-closed。
            if (entry.offset > regionBytes || entry.bytes > regionBytes - entry.offset) {
                error = "v4_object_mismatch";
                return false;
            }
            const auto extractStart = std::chrono::steady_clock::now();
            std::string body(payload, size_t(regionStart + entry.offset), size_t(entry.bytes));
            const bool extractOk = DiskPrefixCache::Digest(body) == hash;
            auto &split = V4Split();
            split.extractNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - extractStart).count());
            split.extractCalls += 1;
            if (!extractOk) { error = "v4_object_checksum"; return false; }
            out = std::move(body);
            return true;
        }
        error = "v4_object_mismatch";
        return false;
    }
    // catalog 点查只取记录偏移（记录级预取用；列与 ObjectBytes 的 SELECT 同源）。
    bool ObjectOffset(const std::string &hash, uint64_t *offset, std::string &error,
                      sqlite3 *catalog = nullptr) {
        Statement query(catalog != nullptr ? catalog : database->value,
                        "SELECT offset FROM objects WHERE hash=?");
        query.Text(1, hash);
        if (!query.Row()) { error = "v4_object_missing"; return false; }
        *offset = query.Number(0);   // 本查询只有 offset 一列
        return true;
    }
    // 批量点查对象记录偏移（2026-10-18，B21）：逐对象构造 Statement 会把 SQL 解析/准备
    // 摊到每个 hash 上（预取 216 个对象时占掉明显墙钟）。这里按 400 个一批用一条
    // IN 查询，语义与逐条 SELECT offset 完全一致；查不到的项保持 found=0，交给既有路径报错。
    size_t ObjectOffsets(const std::vector<std::string> &hashes,
                         std::vector<uint64_t> *offsets,
                         std::vector<uint8_t> *found,
                         sqlite3 *catalog = nullptr) {
        offsets->assign(hashes.size(), 0);
        found->assign(hashes.size(), 0);
        if (hashes.empty()) return 0;
        std::unordered_map<std::string, size_t> indexOf;
        indexOf.reserve(hashes.size() * 2 + 1);
        for (size_t index = 0; index < hashes.size(); ++index) indexOf.emplace(hashes[index], index);
        sqlite3 *db = catalog != nullptr ? catalog : database->value;
        size_t resolved = 0;
        constexpr size_t kChunk = 400;
        for (size_t begin = 0; begin < hashes.size(); begin += kChunk) {
            const size_t count = std::min(kChunk, hashes.size() - begin);
            std::string sql = "SELECT hash,offset FROM objects WHERE hash IN (";
            for (size_t index = 0; index < count; ++index) sql += (index == 0 ? "?" : ",?");
            sql += ")";
            Statement query(db, sql.c_str());
            for (size_t index = 0; index < count; ++index) query.Text(int(index + 1), hashes[begin + index]);
            while (query.Row()) {
                auto it = indexOf.find(query.Text(0));
                if (it == indexOf.end()) continue;
                (*offsets)[it->second] = query.Number(1);
                (*found)[it->second] = 1;
                ++resolved;
            }
        }
        return resolved;
    }
    // 读一条记录并校验记录摘要；不经过 thread_local 16 槽缓存，避免每个 worker 常驻多份记录。
    bool ReadRecordForLoad(uint64_t offset, std::string &payload, std::string &error) const {
        prefix_v4::RecordHeader header;
        fs::path recordPath;
        uint64_t recordOffset = 0;
        if (!LogLocation(offset, recordPath, recordOffset, error)) return false;
        return V4ReadRecord(recordPath, recordOffset, header, payload, error);
    }
    bool IsPending(const std::string &hash) {
        // 写路径并发化（2026-10-02）：改用与 pendingObjects 同步维护的哈希索引，
        // 单次查找 O(log n)，不再线性扫描。pending_scan_steps 的口径随之变为
        // "一次索引查找"（旧口径是线性比较次数）；字段名保持不变以兼容离线解析。
        ++pendingScanSteps;
        return pendingHashSet.count(hash) != 0;
    }
    // v4 单条记录载荷硬上限（与格式层同一常量，避免两处口径漂移）；留 1 MiB 余量。
    static constexpr uint64_t kBatchPayloadBudget =
        fastllm::prefix_v4::kMaxPackedPayloadBytes - (1ull << 20);

    // 物理预估与 v4 对象表编码保持一致：每项 88 B，表和对象区域按格式对齐。
    static uint64_t ProjectedBatchPayload(uint64_t bytes, size_t count) {
        // EncodeObjectTable uses 16 + 88 * count bytes. Keep the estimate in
        // lockstep with the format encoder; the old 72-byte estimate could
        // admit an over-sized batch and fail only after objects were staged.
        return bytes + 88ull * count + 8ull * count + 32ull;
    }

    // 把 batch[begin, begin+count) 对应的对象写成**一条** OBJECT_BATCH 记录，并登记 catalog。
    // 只在对象**之间**补位、末尾不补：格式层编码器 EncodeObjectBatchPayload 就是这个口径
    // （声明 region = 载荷长度且末尾不补），读者 DecodeObjectBatchPayload 严格要求
    // "表尾 + 声明 region == 载荷长度"（disk_prefix_format_v4.cpp 的 v4_batch_payload_size）。
    bool FlushObjectGroup(std::vector<PendingObject> &batch, size_t begin, size_t count,
                          const std::vector<ObjectEntry> &entries, uint64_t regionBytes,
                          std::string &error) {
        const uint64_t assembleStart = NowNs();
        const auto packStart = std::chrono::steady_clock::now();
        const std::vector<char> table = EncodeObjectTable(entries, regionBytes);
        std::string record;
        record.reserve(table.size() + size_t(regionBytes) + 8);
        record.assign(table.begin(), table.end());
        while (record.size() % 8 != 0) record.push_back('\0');
        for (size_t i = 0; i < count; ++i) {
            record += batch[begin + i].bytes;
            if (i + 1 < count) {
                while (record.size() % 8 != 0) record.push_back('\0');
            }
        }
        packNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - packStart).count());
        assembleNs += NowNs() - assembleStart;
        uint64_t offset = 0;
        if (!AppendLiveRecord(kRecordObjectBatch, ++sequence, record, &offset, error)) {
            --sequence;
            return false;
        }
        {
            // §10.3 第 6 条：索引侧写入必须在**一个事务**里应用，不能逐行隐式提交。
            const auto catalogStart = std::chrono::steady_clock::now();
            Transaction transaction(*database);
            for (const auto &entry : entries) {
                Statement insert(database->value,
                                 "INSERT OR REPLACE INTO objects(hash,kind,bytes,offset) VALUES(?,?,?,?)");
                insert.Text(1, entry.hash);
                insert.Number(2, entry.kind);
                insert.Number(3, entry.bytes);
                insert.Number(4, offset);
                insert.Run();
            }
            transaction.Commit();
            catalogNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - catalogStart).count());
        }
        return true;
    }

    // 一条记录装多个对象；上层在写 PREPARE 之前必须调用 FlushObjects，
    // 保证"数据先落盘、控制记录后落盘"的顺序（方案 §10.3）。
    //
    // R1/R2 修复：
    //   · 旧实现把全部待写对象塞进**一条**记录，载荷超 16 MiB 上限时 AppendRecordToFile 直接失败；
    //     失败分支只回退 sequence、**不清 pendingObjects/pendingBytes** ⇒ 脏批次被之后每次提交
    //     反复带上、越滚越大，整个 store 从此再也写不进任何东西（真机实测：第 77 笔 durable 之后
    //     live_checkpoints 恒 77、journal calls=0，长上下文因此一条也没落盘）。
    //   · 现在按载荷上限**分批写成多条 OBJECT_BATCH 记录**（格式本就允许一个检查点有多条批次记录，
    //     每条记录各自登记自己的对象偏移），任何一批都不会超上限；
    //   · 且无论成败都先把本批从 pending 里 swap 出来，**不再留脏批次**。真正的 IO 失败
    //     （EIO/ENOSPC 等）仍如实返回 false，由调用方中止该检查点（fail-closed）。
    bool FlushObjects(std::string &error) {
        if (pendingObjects.empty()) return true;
        std::vector<PendingObject> batch;
        batch.swap(pendingObjects);
        pendingHashSet.clear();   // 与 pendingObjects 同步：字节已 swap 到本批，索引清零
        pendingBytes = 0;
        size_t index = 0;
        while (index < batch.size()) {
            std::vector<ObjectEntry> entries;
            entries.reserve(batch.size() - index);
            uint64_t regionBytes = 0;
            size_t count = 0;
            while (index + count < batch.size()) {
                const PendingObject &item = batch[index + count];
                // S1-2 通用缺口（2026-10-02，对标审计）：与解码侧 DecodeObjectTable(:234)/
                // DecodeObjectList(:476) 的 `bytes > kMaxObjectBytes` 对称。单个对象超过 8 MiB，
                // 即便能塞进 16 MiB 记录里，读侧也必然拒收（真机 r7：kind=4 有 36 个对象 >8 MiB、
                // 最大 16.7 MB，重启日志出现 15 次 v4_object_list_bytes）⇒ 发布前 fail-closed。
                if (item.bytes.size() > prefix_v4::kMaxObjectBytes) {
                    error = "v4_object_exceeds_decode_limit";
                    return false;
                }
                const uint64_t padded = regionBytes + ((regionBytes % 8 != 0) ? (8 - regionBytes % 8) : 0);
                const uint64_t nextRegion = padded + item.bytes.size();
                if (count > 0 && ProjectedBatchPayload(nextRegion, count + 1) > kBatchPayloadBudget) break;
                ObjectEntry entry;
                entry.kind = item.kind;
                entry.bytes = item.bytes.size();
                entry.offset = padded;
                entry.hash = item.hash;
                entries.push_back(std::move(entry));
                regionBytes = nextRegion;
                ++count;
            }
            if (count == 0) {
                error = "v4_object_exceeds_record_payload_limit";
                return false;
            }
            if (!FlushObjectGroup(batch, index, count, entries, regionBytes, error)) {
                // 2：失败时**不得丢弃**已 swap 出的未写批次。旧实现
                // 直接 return false，未写的对象永久消失；而模型侧已把它们的引用累计进 state.kv
                // （且 issued 已前移、永不重导）⇒ 后续检查点的 root 会引用悬空对象（见上方 1 的根因）。
                // 现在把未写批次退回 pendingObjects：下一次提交会重试写出（内容寻址 ⇒ 重复写幂等）。
                for (size_t rest = index; rest < batch.size(); ++rest) {
                    pendingBytes += batch[rest].bytes.size();
                    pendingHashSet.insert(batch[rest].hash);
                    pendingObjects.push_back(std::move(batch[rest]));
                }
                return false;
            }
            index += count;
        }
        return true;
    }
    // §③：字节移交（move）而不是复制 —— 对象字节在本进程内已经不被使用。
    // `outStaged`（可选）：true = 本次真的把字节排进了待写批次；false = 对象已存在/已在待写
    // 表里（未重写）。Route B 用它保证"块对象一次写、永不重写"，并让 roots.bytes 只计新写。
    bool WriteObjectOwned(const std::string &hash, uint32_t kind, std::string &&bytes,
                          std::string &error, bool *outStaged = nullptr) {
        if (outStaged != nullptr) *outStaged = false;
        ++objectCount;
        const uint64_t lookupStart = NowNs();
        const bool exists = HasObject(hash);
        lookupNs += NowNs() - lookupStart;
        if (exists) return true;
        const uint64_t scanStart = NowNs();
        const bool pending = IsPending(hash);
        pendingScanNs += NowNs() - scanStart;
        if (pending) return true;
        const uint64_t stageStart = NowNs();
        // R2：放进去之前先算载荷——放进去就超上限的话，先把已攒的批次写掉，绝不攒出写不下的批次。
        if (!pendingObjects.empty() &&
            ProjectedBatchPayload(pendingBytes + bytes.size() + 8, pendingObjects.size() + 1)
                > kBatchPayloadBudget) {
            if (!FlushObjects(error)) { stageNs += NowNs() - stageStart; return false; }
        }
        pendingBytes += bytes.size();
        pendingObjects.push_back({hash, kind, std::move(bytes)});
        pendingHashSet.insert(hash);   // 与 pendingObjects 同步维护（IsPending 的索引）
        stageNs += NowNs() - stageStart;
        if (outStaged != nullptr) *outStaged = true;
        if (pendingBytes >= (uint64_t(8) << 20)) return FlushObjects(error);
        return true;
    }
    bool WriteObject(const std::string &hash, uint32_t kind, const std::string &bytes,
                     std::string &error) {
        ++objectCount;
        const uint64_t lookupStart = NowNs();
        const bool exists = HasObject(hash);
        lookupNs += NowNs() - lookupStart;
        if (exists) return true;
        const uint64_t scanStart = NowNs();
        const bool pending = IsPending(hash);
        pendingScanNs += NowNs() - scanStart;
        if (pending) return true;
        const uint64_t stageStart = NowNs();
        // R2：同上，先算载荷再决定放不放。
        if (!pendingObjects.empty() &&
            ProjectedBatchPayload(pendingBytes + bytes.size() + 8, pendingObjects.size() + 1)
                > kBatchPayloadBudget) {
            if (!FlushObjects(error)) { stageNs += NowNs() - stageStart; return false; }
        }
        pendingObjects.push_back({hash, kind, bytes});
        pendingHashSet.insert(hash);   // 与 pendingObjects 同步维护（IsPending 的索引）
        pendingBytes += bytes.size();
        stageNs += NowNs() - stageStart;
        // 保持在 packed payload 上限以内，并让一次 fdatasync 覆盖尽量多对象。
        if (pendingBytes >= (uint64_t(8) << 20)) return FlushObjects(error);
        return true;
    }
    // §11.2：跨进程锁序固定为"先 lifecycle，再 append"，禁止逆序。
    // 正常提交取 lifecycle 共享 + append 独占；GC/rebuild 取 lifecycle 独占
    // （非阻塞尝试，失败即 cache_busy，不持 append 等 reader 释放）。
    struct RootLocks {
        int lifecycle = -1, append = -1;
        ~RootLocks() {
            if (append >= 0) { flock(append, LOCK_UN); close(append); }
            if (lifecycle >= 0) { flock(lifecycle, LOCK_UN); close(lifecycle); }
        }
    };
    bool AcquireLocks(RootLocks &locks, bool exclusiveLifecycle, bool blocking,
                      std::string &error) {
        const int lifecycle = open((root / ".lifecycle").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lifecycle < 0) { error = "v4_lifecycle_open"; return false; }
        const int lifecycleOp = (exclusiveLifecycle ? LOCK_EX : LOCK_SH) | (blocking ? 0 : LOCK_NB);
        if (flock(lifecycle, lifecycleOp) != 0) { close(lifecycle); error = "cache_busy"; return false; }
        locks.lifecycle = lifecycle;
        const int append = open((root / ".append").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (append < 0) { error = "v4_append_open"; return false; }
        if (flock(append, LOCK_EX | (blocking ? 0 : LOCK_NB)) != 0) {
            close(append);
            error = "cache_busy";
            return false;
        }
        locks.append = append;
        // §11.2：进锁后必须刷新尾序号——另一个进程/实例可能已推进控制日志，
        // 过期的 in-memory 计数会分配出重复 sequence，进而让恢复判定为损坏。
        {
            Statement query(database->value, "SELECT value FROM meta WHERE key='sequence'");
            if (query.Row()) {
                const std::string text = query.Text(0);
                const unsigned long long persisted = std::strtoull(text.c_str(), nullptr, 10);
                if (persisted > sequence) sequence = persisted;
            }
        }
        if (GetMeta("append_blocked") == "1") appendBlocked = true;
        return true;
    }
    uint64_t Scalar(const char *sql) {
        Statement query(database->value, sql);
        return query.Row() ? query.Number(0) : 0;
    }
    // S2-5a：control.log 的**真实物理占用字节数**。此前 CapacityStatus 把 live 字节数
    // 当 physical 上报，掩盖了"日志里还留着已被淘汰/未引用对象"的垃圾。这里直接量文件
    // 大小；S2-5b 分段落盘时改成各段求和。只读测量，不做读侧收尾（收尾不改变当前文件
    // 大小，且在 actor 线程上收尾会自锁）。
    uint64_t PhysicalBytes() const {
        if (segmented) return SegmentPhysicalBytes();
        std::error_code sizeError;
        if (!fs::exists(logPath, sizeError)) return 0;
        const uint64_t bytes = uint64_t(fs::file_size(logPath, sizeError));
        return sizeError ? 0 : bytes;
    }
    uint64_t ProjectedPhysicalBytes(const std::vector<DiskPrefixV4Store::Object> &objects,
                                    const std::set<std::string> &existing,
                                    const std::string &key,
                                    const std::string *checkpointMetadata) const {
        uint64_t result = PhysicalBytes();
        std::vector<uint64_t> objectPayloads;
        std::set<std::string> accounted;
        auto addObject = [&](const std::string &hash, uint64_t bytes) {
            if (hash.empty() || bytes == 0 || existing.count(hash) ||
                !accounted.insert(hash).second) return;
            objectPayloads.push_back(bytes);
        };
        for (const auto &pending : pendingObjects)
            addObject(pending.hash, pending.bytes.size());
        for (const auto &object : objects) {
            if (!object.bytes.empty()) addObject(object.hash, object.bytes.size());
        }
        // The normal format stores an object-list root as one more object;
        // Route B's state object is already part of `objects` and needs no
        // synthetic root allocation.
        const bool chainRoot = checkpointMetadata != nullptr &&
            checkpointMetadata->find("\"chain_tip\"") != std::string::npos;
        if (!chainRoot) {
            std::vector<ObjectListEntry> list;
            list.reserve(objects.size());
            for (const auto &object : objects) {
                const uint64_t bytes = object.bytes.empty() ? object.bytesSize : object.bytes.size();
                if (bytes == 0) return std::numeric_limits<uint64_t>::max();
                list.push_back({object.kind, bytes, object.hash});
            }
            const std::vector<char> encoded = EncodeObjectList(list);
            const std::string rootText(encoded.data(), encoded.size());
            addObject(DiskPrefixCache::Digest(rootText), rootText.size());
        }
        uint64_t region = 0, rawBytes = 0;
        size_t count = 0;
        const auto align8 = [](uint64_t value) {
            return value + ((value % 8) ? (8 - value % 8) : 0);
        };
        auto flush = [&] {
            if (count == 0) return;
            const uint64_t table = 16ull + 88ull * count;
            result = Add(result, RecordPhysicalBytes(
                align8(table) + region));
            region = 0; rawBytes = 0; count = 0;
        };
        for (const uint64_t bytes : objectPayloads) {
            // Mirror WriteObjectOwned/WriteObject's admission and 8 MiB
            // automatic flush boundaries, including the synthetic root.
            if (count != 0 && ProjectedBatchPayload(
                    rawBytes + bytes + 8, count + 1) > kBatchPayloadBudget) flush();
            region = region + ((region % 8) ? (8 - region % 8) : 0);
            region += bytes;
            rawBytes += bytes;
            ++count;
            if (rawBytes >= (uint64_t(8) << 20)) flush();
        }
        flush();
        // PREPARE, optional checkpoint metadata, and COMMIT have fixed-format
        // payloads. Estimating their actual sizes avoids rejecting a write
        // that fits the physical quota merely because of a generic 3*256B
        // placeholder.
        result = Add(result, RecordPhysicalBytes(156ull + key.size()));
        if (checkpointMetadata != nullptr)
            result = Add(result, RecordPhysicalBytes(
                92ull + key.size() + checkpointMetadata->size()));
        result = Add(result, RecordPhysicalBytes(152));
        return result;
    }
};

class DiskPrefixV4Store::ObjectLease : public V4ObjectLeaseState {
public:
    explicit ObjectLease(const fs::path &root) : V4ObjectLeaseState(root) {}
};

DiskPrefixV4Store::DiskPrefixV4Store(std::unique_ptr<Impl> value) : impl(std::move(value)) {
    // §⑥ 备菜线程要调用 store 的私有落盘实现（CommitCheckpointLocked），回指外壳。
    impl->owner = this;
}
DiskPrefixV4Store::~DiskPrefixV4Store() = default;
std::string DiskPrefixV4Store::Root() const { return impl->root.string(); }

std::unique_ptr<DiskPrefixV4Store> DiskPrefixV4Store::Open(const std::string &prefixCacheDir,
                                                           uint64_t quotaBytes,
                                                           std::string &error) {
    if (V4DirectIOEnabled())
        std::fprintf(stderr, "[Prefix SSD] V4 read mode: O_DIRECT (4096-byte aligned, buffered fallback disabled).\n");
    error.clear();
    V4InvalidateRecordCache();   // [P1-a] 打开即失效本线程记录缓存（键已含文件身份，此为双保险）
    if (prefixCacheDir.empty()) { error = "v4_empty_prefix_cache_dir"; return nullptr; }
    auto impl = std::make_unique<Impl>();
    impl->root = fs::path(prefixCacheDir).lexically_normal() / "v4";
    impl->logPath = impl->root / "control.log";
    impl->dbPath = impl->root / "catalog.sqlite3";
    impl->quota = quotaBytes;
    try {
        PrivateDirectory(impl->root);
        impl->segmented = V4SegmentsEnv();
        impl->segmentBytes = V4SegmentBytesEnv();
        {
            std::string segmentError;
            if (!impl->ListSegments(segmentError)) { error = segmentError; return nullptr; }
            std::error_code legacyError;
            const bool legacy = fs::exists(impl->logPath, legacyError);
            // DESIGN-v3-segments section 7 (user 2026-11-06): no legacy store support.
            if (impl->segmented && legacy) { error = "v4_segments_legacy_store"; return nullptr; }
            // A flag-off build must never read a segmented store as an empty single log.
            if (!impl->segmented && !impl->segments.empty()) {
                error = "v4_segments_store_requires_flag";
                return nullptr;
            }
            if (impl->segmented) {
                std::fprintf(stderr, "[Prefix SSD] v4 segments: on segment_bytes=%llu segments=%zu "
                                     "physical=%llu quota=%llu.\n",
                             (unsigned long long)impl->segmentBytes, impl->segments.size(),
                             (unsigned long long)impl->SegmentPhysicalBytes(),
                             (unsigned long long)impl->quota);
            }
        }
        const fs::path format = impl->root / "FORMAT";
        if (fs::exists(format)) {
            Check(!fs::is_symlink(format) && ReadText(format) == kFormatText,
                  "v4_format_unsupported");
        } else {
            const int fd = open(format.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
            Check(fd >= 0, "v4_format_create");
            try {
                WriteExact(fd, kFormatText, strlen(kFormatText));
                Check(fdatasync(fd) == 0, "v4_format_sync");
                close(fd);
            } catch (...) { close(fd); throw; }
            SyncDirectory(impl->root);
        }
        // §17.1 B7（2026-10-01 实跑确认缺口）：索引库（SQLite）**自身损坏**时旧实现只有通用
        // catch，直接 return nullptr，缓存"打不开"；而索引库只是**可从控制日志重建的投影**，
        // 正确语义是 fail-closed 地**照日志重建**。这里只对**确认的损坏**降级：
        //   SQLITE_NOTADB(26) / SQLITE_CORRUPT(11) / malformed / file is encrypted
        // 其余错误（EIO/ENOSPC 等，§17.1 B6）**绝不**当"坏数据可删"，保持 fail-closed。
        auto confirmedCatalogCorruption = [](const std::string &message) {
            static const char *kMarkers[] = {"not a database", "code=26", "code=11",
                                             "malformed", "file is encrypted"};
            for (const char *marker : kMarkers) {
                if (message.find(marker) != std::string::npos) return true;
            }
            return false;
        };
        try {
            impl->database = std::make_unique<Database>(impl->dbPath);
            impl->Schema();
        } catch (const std::exception &exception) {
            const std::string catalogError = exception.what();
            if (!confirmedCatalogCorruption(catalogError)) throw;   // EIO/ENOSPC 等：不降级
            // 损坏的投影**改名留证**（不删），随后按"新库"重建；控制日志与对象数据一个字节不动。
            impl->database.reset();
            const std::string stamp = std::to_string(uint64_t(std::time(nullptr)));
            for (const char *suffix : {"", "-wal", "-shm"}) {
                const fs::path source = fs::path(impl->dbPath.string() + suffix);
                std::error_code ignored;
                if (!fs::exists(source, ignored)) continue;
                fs::rename(source, fs::path(source.string() + ".corrupt-" + stamp), ignored);
            }
            SyncDirectory(impl->root);
            std::fprintf(stderr,
                         "[Prefix SSD] v4 open: catalog was corrupt (%s); renamed to *.corrupt-%s "
                         "and rebuilding from the control log.\n",
                         catalogError.c_str(), stamp.c_str());
            impl->database = std::make_unique<Database>(impl->dbPath);
            impl->Schema();
        }
        const std::string initialized = impl->GetMeta("initialized");
        const std::string stored = impl->GetMeta("sequence");
        impl->sequence = stored.empty() ? 0 : std::stoull(stored);
        impl->appendBlocked = impl->GetMeta("append_blocked") == "1";
        // S2-4：内部 catalog 版本闸门。旧 catalog 的 checkpoints 表没有 family/kind/
        // length/dtype/provenance/prefix/created 等索引列，直接跑新 SQL 会 "no such
        // column" 报错；而 catalog 只是 control.log 的投影、可随时重建。故版本号不是
        // "2" 就强制走 RebuildIndex（用新表结构建临时库后原子 rename 覆盖）：旧 catalog
        // 自动升级，不会 SQL 报错。新建库走下面的 initialized.empty() 分支重建，那里也会
        // 由 RebuildIndexBareLocked 写上 catalog_schema=2。
        const bool catalogNeedsUpgrade = impl->GetMeta("catalog_schema") != "2";
        if (initialized.empty()) {
            auto store = std::unique_ptr<DiskPrefixV4Store>(new DiskPrefixV4Store(std::move(impl)));
            store->impl->StartActor();
            if (!store->RebuildIndex(error)) { store->impl->StopActor(); return nullptr; }
            store->impl->ShrinkSegmentsToQuota();
            return store;
        }
        // §17.1（2026-09-30 修复③，故障注入实测）：COMMIT 已同步但 SQLite 投影未提交时重开
        // 原本**不自查**（实测 on_open=0 / after_rebuild=1，error=v4_root_missing）。用持久化的
        // 日志长度水位做 O(1) 检测：水位与控制日志实际长度不符 ⇒ 投影落后于日志（崩溃点落在
        // COMMIT 同步之后、root 事务提交之前），立即自动重建。失败必须 fail-closed，不允许带着
        // 可疑投影继续服务。
        const std::string watermark = impl->GetMeta("log_bytes");
        uint64_t watermarkBytes = 0;
        bool haveWatermark = false;
        if (!watermark.empty()) {
            try { watermarkBytes = std::stoull(watermark); haveWatermark = true; } catch (...) {}
        }
        std::error_code sizeError;
        const uint64_t actualLogBytes = impl->segmented ? impl->SegmentEndOffset()
            : (fs::exists(impl->logPath) ? uint64_t(fs::file_size(impl->logPath, sizeError)) : 0);
        const bool projectionBehind =
            haveWatermark && !sizeError && watermarkBytes != actualLogBytes;
        auto store = std::unique_ptr<DiskPrefixV4Store>(new DiskPrefixV4Store(std::move(impl)));
        store->impl->StartActor();
        if (projectionBehind || catalogNeedsUpgrade) {
            std::string rebuildError;
            if (!store->RebuildIndex(rebuildError)) {
                std::fprintf(stderr,
                             "[Prefix SSD] v4 open: rebuild failed (projection_behind=%d "
                             "catalog_upgrade=%d): %s.\n",
                             int(projectionBehind), int(catalogNeedsUpgrade),
                             rebuildError.c_str());
                store->impl->StopActor();
                error = rebuildError.empty() ? "v4_open_rebuild" : rebuildError;
                return nullptr;
            }
            std::fprintf(stderr,
                         "[Prefix SSD] v4 open: %s; rebuilt.\n",
                         catalogNeedsUpgrade
                             ? "catalog schema upgraded (S2-4 index columns)"
                             : "projection was behind the control log");
        }
        store->impl->ShrinkSegmentsToQuota();
        return store;
    } catch (const std::exception &exception) {
        error = exception.what();
        return nullptr;
    }
}

namespace {
// 一个 canonical root 在一个进程里只允许一个 v4 store（方案 §15：CommitActor 每
// canonical root/进程一个）。同一进程里第二个 DiskPrefixCache 指向同一个目录时
// （文本 KV 与图片特征两个 namespace）必须复用它，否则两个 actor 各自缓存
// sequence 往同一份 control.log 追加，会写出重复序号并让重建失败。
struct SharedV4Stores {
    std::mutex mutex;
    std::map<std::string, std::pair<std::weak_ptr<DiskPrefixV4Store>, uint64_t>> entries;
};
std::mutex &SharedV4OpenMutex() {
    static std::mutex value;
    return value;
}
SharedV4Stores &SharedV4Registry() {
    static SharedV4Stores value;
    return value;
}
std::string SharedV4Key(const std::string &prefixCacheDir) {
    std::error_code ignored;
    fs::path path = fs::weakly_canonical(fs::path(prefixCacheDir), ignored);
    if (ignored) path = fs::absolute(fs::path(prefixCacheDir), ignored).lexically_normal();
    return (path / "v4").string();
}
}   // namespace

std::shared_ptr<DiskPrefixV4Store> DiskPrefixV4Store::OpenShared(const std::string &prefixCacheDir,
                                                                 uint64_t quotaBytes,
                                                                 std::string &error) {
    error.clear();
    V4InvalidateRecordCache();   // [P1-a] 打开即失效本线程记录缓存
    if (prefixCacheDir.empty()) { error = "v4_empty_prefix_cache_dir"; return nullptr; }
    const std::string key = SharedV4Key(prefixCacheDir);
    // Open() 只在每个 cache 实例构造时调用一次、且可能触发全量重建，因此这里用
    // 一把独立的创建锁串行化，而不是在持有注册表锁时做磁盘 I/O。
    std::lock_guard<std::mutex> openLock(SharedV4OpenMutex());
    SharedV4Stores &registry = SharedV4Registry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto found = registry.entries.find(key);
        if (found != registry.entries.end()) {
            if (auto existing = found->second.first.lock()) {
                if (found->second.second != quotaBytes) {
                    error = "v4_shared_quota_mismatch";
                    return nullptr;
                }
                return existing;
            }
            registry.entries.erase(found);
        }
    }
    std::unique_ptr<DiskPrefixV4Store> opened = Open(prefixCacheDir, quotaBytes, error);
    if (opened == nullptr) return nullptr;
    std::shared_ptr<DiskPrefixV4Store> shared(std::move(opened));
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        registry.entries[key] = {std::weak_ptr<DiskPrefixV4Store>(shared), quotaBytes};
    }
    return shared;
}

bool DiskPrefixV4Store::Commit(const std::string &key, const std::string &identity,
                               const std::vector<Object> &objects, uint64_t *outSequence,
                               std::string &error) {
    std::vector<Object> owned(objects);   // 兼容路径：复制一份可变副本交给内部移交逻辑
    return impl->RunActor([&] { return CommitLocked(key, identity, owned, outSequence, error); },
                          error);
}

bool DiskPrefixV4Store::CommitLocked(const std::string &key, const std::string &identity,
                               std::vector<Object> &objects, uint64_t *outSequence,
                               std::string &error) {
    error.clear();
    Impl::RootLocks locks;
    const uint64_t lockStart = NowNs();
    const bool acquired = impl->AcquireLocks(locks, false, true, error);
    impl->lockNs += NowNs() - lockStart;
    if (!acquired) return false;
    return CommitBareLocked(key, identity, objects, outSequence, error);
}

bool DiskPrefixV4Store::CommitBareLocked(const std::string &key, const std::string &identity,
                               std::vector<Object> &objects, uint64_t *outSequence,
                               std::string &error,
                               const std::string *checkpointMetadata) {
    // Never rely on a caller-provided error being empty: a stale message would
    // make a later "if (!error.empty())" check fail for no reason.
    error.clear();
    if (impl->appendBlocked) {
        error = "v4_control_uncommitted_tail";
        return false;
    }
    try {
        if (key.empty() || !prefix_v4::IsDigest(identity) || objects.empty()) {
            // identity is a checkpoint identity digest, not free text: the
            // record layout stores it as a fixed 64-character digest.
            error = "v4_commit_argument";
            return false;
        }
        if (checkpointMetadata != nullptr && checkpointMetadata->empty()) {
            error = "v4_checkpoint_metadata";
            return false;
        }
        // Every declared object must actually hash to its own name before it is
        // written: an unverified hash must never be published as data.
        // 每对象的摘要格式（64 位十六进制）必须始终成立：这是结构约束，代价可忽略。
        for (const auto &object : objects) {
            if (!prefix_v4::IsDigest(object.hash)) { error = "v4_object_checksum"; return false; }
        }
        // 写盘前的"逐对象重算 SHA256"**默认开启**（方案 §22.4.4 乙案原设想默认关闭，
        // 但 2026-10-01 已把默认翻成开启，见 StrictWriteVerifyEnabled()）：对象内容的正确性
        // 另由读回路径 Impl::ObjectBytes() 与重建路径逐对象重算兜底。
        // 如需关闭写盘时的逐对象重算，显式设 FASTLLM_PREFIX_CACHE_STRICT_VERIFY=0。
        if (StrictWriteVerifyEnabled()) {
            const uint64_t verifyStart = NowNs();
            // §407ms 后续（2026-10-01）：写侧复算此前是**逐对象串行**，实测每检查点多约 600 ms
            // （旧库 object_verify_ms=597.3）。这里改成按对象切片并行校验：能力不减（每个对象
            // 仍然逐字节重算 SHA256），只是把串行段摊到多个线程；对象向量只读，各线程只写自己的
            // 失败标记与字节计数，失败即整笔拒绝（与串行版语义一致）。
            const size_t workerCount = std::min<size_t>(4, objects.size());
            std::vector<std::thread> verifyThreads;
            std::vector<char> verifyFailed(workerCount, 0);
            std::atomic<uint64_t> verifiedBytes{0};
            verifyThreads.reserve(workerCount);
            for (size_t worker = 0; worker < workerCount; ++worker) {
                verifyThreads.emplace_back([&, worker]() {
                    const size_t begin = objects.size() * worker / workerCount;
                    const size_t end = objects.size() * (worker + 1) / workerCount;
                    uint64_t bytes = 0;
                    for (size_t i = begin; i < end; ++i) {
                        // 只引用条目（bytes 为空）没有字节可校验：它由前面那条批量存在性查询保证，
                        // 若在这里对空串算摘要就会把合法的"只引用"提交误判为 v4_object_checksum
                        // （2026-10-01 打开默认复算后由 referenceOnlyObjectsAppearInListing 实测抓到）。
                        if (objects[i].bytes.empty()) continue;
                        bytes += objects[i].bytes.size();
                        if (DiskPrefixCache::Digest(objects[i].bytes) != objects[i].hash) {
                            verifyFailed[worker] = 1;
                            break;
                        }
                    }
                    verifiedBytes.fetch_add(bytes, std::memory_order_relaxed);
                });
            }
            for (auto &thread : verifyThreads) {
                if (thread.joinable()) thread.join();
            }
            impl->verifyBytes += verifiedBytes.load(std::memory_order_relaxed);
            impl->verifyNs += NowNs() - verifyStart;
            for (char failed : verifyFailed) {
                if (failed != 0) {
                    error = "v4_object_checksum";
                    return false;
                }
            }
        }
        // §11.4：配额必须**在写之前**预约——把待写入字节连同记录头/对象表/对齐开销
        // 一起算进来；不够就先按 FIFO 持久化 ERASE 退掉最老的检查点，仍不够则 fail
        // closed（绝不允许"写完才发现超配额"）。
        const uint64_t quotaStart = NowNs();
        if (impl->quota > 0) {
            // 写路径并发化（2026-10-02）：旧实现对本检查点的**每个对象**各查一次
            // `HasObject()`（一次预编译 SELECT），32K 检查点约 896 次 prepare+query；这段
            // 代码在唯一 actor 上串行执行，是 object_write_ms 的主要构成之一。现改为
            // **批量**存在性查询（500/批，与下方"只引用对象批量校验"同一口径），配合
            // IsPending 的哈希索引，配额预估从 O(n) 次查询降到 O(n/500) 次。语义不变：
            // "哪些对象算新写"仍然只看 objects 表 + 待写表。
            std::set<std::string> existing;
            {
                constexpr size_t kQuotaLookupBatch = 500;
                for (size_t base = 0; base < objects.size(); base += kQuotaLookupBatch) {
                    const size_t count = std::min(kQuotaLookupBatch, objects.size() - base);
                    std::string sql = "SELECT hash FROM objects WHERE hash IN (";
                    for (size_t i = 0; i < count; ++i) sql += (i == 0 ? "?" : ",?");
                    sql += ")";
                    Statement query(impl->database->value, sql.c_str());
                    for (size_t i = 0; i < count; ++i) query.Text(int(i + 1), objects[base + i].hash);
                    while (query.Row()) existing.insert(query.Text(0));
                }
            }
            uint64_t incoming = 0;
            for (const auto &object : objects) {
                if (existing.count(object.hash) || impl->IsPending(object.hash)) continue;
                incoming += object.bytes.size() + 128 + 88 + 8;   // 记录头 + 表项 + 对齐
            }
            // R3：配额压力下**先按 FIFO 淘汰最旧的存活检查点、
            // 腾出空间再接收新写**，只有在没有安全可回收对象时才
            // fail-closed（capacity_unavailable）。
            // 引用保护（为什么在途检查点不会被误伤）：
            //   · 候选集合是 `roots WHERE live=1`——已经 COMMIT 的存活检查点；本次正在提交的
            //     检查点还没写入 root，根本不在候选里；
            //   · 淘汰动作 = 追加 ERASE 记录 + `roots.live=0`，**不 unlink、不删 objects 行**，
            //     所以被在途检查点“只引用”的对象字节仍然可寻址，提交后的清单不会悬空；
            //   · 物理字节只在 Compact（）这个独立维护路径里按 live root 可达集回收（不在生产
            //     写路径上，且与本次提交同在 actor 上串行）。见报告“R3 残余风险/缺失 API”。
            uint64_t evictedForAdmission = 0;
            std::vector<std::pair<std::string, std::string>> erasedForAdmission;
            bool evictionAppendFailed = false;
            bool evictionNoVictim = false;
            // The quota is physical.  A live-root sum can fit while the
            // append would still exceed the control log quota because ERASE
            // only changes roots.live; retire one safe FIFO root and let the
            // background compactor reclaim its bytes before retrying.
            bool physicalPressure =
                impl->ProjectedPhysicalBytes(objects, existing, key, checkpointMetadata) > impl->quota;
            if (impl->segmented && physicalPressure) {
                // DESIGN-v3-segments C5: free space by deleting the oldest whole segment(s);
                // never wait (LMCache local_disk_backend.py:399-417).
                std::string reclaimReason;
                size_t reclaimedSegments = 0;
                while (physicalPressure && impl->ReclaimOldestSegment(reclaimReason)) {
                    ++reclaimedSegments;
                    // Deleted objects rows changed what this commit must write.
                    existing.clear();
                    constexpr size_t kReclaimLookupBatch = 500;
                    for (size_t first = 0; first < objects.size(); first += kReclaimLookupBatch) {
                        const size_t count = std::min(kReclaimLookupBatch, objects.size() - first);
                        std::string sql = "SELECT hash FROM objects WHERE hash IN (";
                        for (size_t i = 0; i < count; ++i) sql += (i == 0 ? "?" : ",?");
                        sql += ")";
                        Statement query(impl->database->value, sql.c_str());
                        for (size_t i = 0; i < count; ++i) query.Text(int(i + 1), objects[first + i].hash);
                        while (query.Row()) existing.insert(query.Text(0));
                    }
                    incoming = 0;
                    for (const auto &object : objects) {
                        if (existing.count(object.hash) || impl->IsPending(object.hash)) continue;
                        incoming += object.bytes.size() + 128 + 88 + 8;
                    }
                    physicalPressure =
                        impl->ProjectedPhysicalBytes(objects, existing, key, checkpointMetadata) > impl->quota;
                }
                if (physicalPressure) {
                    std::fprintf(stderr,
                        "[Prefix SSD] v4 segment-reclaim-skipped: reason=%s reclaimed=%zu physical=%llu "
                        "quota=%llu (save withheld).\n",
                        reclaimReason.c_str(), reclaimedSegments,
                        (unsigned long long)impl->PhysicalBytes(), (unsigned long long)impl->quota);
                    std::fflush(stderr);
                    error = "capacity_unavailable";
                    return false;
                }
            }
            const bool fastAdmission = V4FastAdmission();
            std::vector<Impl::RootBytesRow> fastRows;
            bool fastRowsReady = false;
            size_t fastWalked = 0;
            uint64_t fastRecomputeNs = 0, fastLockNs = 0;
            const uint64_t fastPassStart = NowNs();
            if (fastAdmission && impl->quota > 0 &&
                impl->fastAdmissionStale && LiveBytes() + incoming > impl->quota) {
                // C2: read-only walk before the reader mutex (actor thread is the only writer).
                std::string accountingError;
                if (!impl->ComputeLiveBytesRows(fastRows, fastWalked, accountingError)) {
                    error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                    return false;
                }
                fastRecomputeNs = NowNs() - fastPassStart;
                fastRowsReady = true;
            }
            {
            // Keep pin selection and the ERASE/catalog mutation atomic with
            // respect to a reader acquiring a new checkpoint pin.
            std::lock_guard<std::mutex> pinLock(impl->pins->mutex);
            const uint64_t fastLockStart = NowNs();
            if (fastRowsReady) {
                impl->ApplyLiveBytesRows(fastRows);
                impl->fastAdmissionStale = false;
            }
            // A retired root can own bytes shared by a newer live root. If
            // the cheap accounting says this write is over quota, repair that
            // cold index before choosing a victim; this prevents shared data
            // from being counted as free merely because its first owner died.
            if (!fastAdmission && impl->quota > 0 && LiveBytes() + incoming > impl->quota) {
                std::string accountingError;
                if (!impl->RecomputeLiveBytesForQuota(accountingError)) {
                    error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                    return false;
                }
            }
            while (LiveBytes() + incoming > impl->quota || physicalPressure) {
                std::string victimIdentity, victimKey;
                std::unique_ptr<Lease> victimLease;
                if (!impl->OldestUnpinnedRoot(victimIdentity, victimKey, victimLease)) {
                    // 淘汰也腾不出空间：按既有契约 fail-closed，绝不“先写后超”。
                    if (evictedForAdmission > 0) {
                        std::fprintf(stderr,
                            "[Prefix SSD] v4 admission eviction: evicted_for_admission=%llu "
                            "still_over_quota live=%llu incoming=%llu quota=%llu.\n",
                            (unsigned long long)evictedForAdmission,
                            (unsigned long long)LiveBytes(), (unsigned long long)incoming,
                            (unsigned long long)impl->quota);
                        std::fflush(stderr);
                    }
                    error = "capacity_unavailable";
                    evictionNoVictim = true;
                    break;
                }
                prefix_v4::EraseRecord erase;
                erase.sequence = impl->sequence + 1;
                erase.key = victimKey;
                erase.identity = victimIdentity;
                const std::vector<char> eraseBytes = EncodeErase(erase);
                if (!impl->AppendLiveRecord(kRecordErase, ++impl->sequence,
                                            std::string(eraseBytes.data(), eraseBytes.size()), nullptr,
                                            error)) {
                    --impl->sequence;
                    evictionAppendFailed = true;
                    break;
                }
                Statement retire(impl->database->value,
                                 "UPDATE roots SET live=0 WHERE identity=? AND key=?");
                retire.Text(1, victimIdentity); retire.Text(2, victimKey); retire.Run();
                impl->SetMeta("sequence", std::to_string(impl->sequence));
                impl->RefreshLogWatermark();
                erasedForAdmission.emplace_back(victimKey, victimIdentity);
                if (fastAdmission) {
                    // C1: LiveBytes() already excludes the retired root; recompute once next pass.
                    impl->fastAdmissionStale = true;
                } else {
                std::string accountingError;
                if (!impl->RecomputeLiveBytesForQuota(accountingError)) {
                    error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                    evictionAppendFailed = true;
                    break;
                }
                }
                ++evictedForAdmission;
                ++impl->evictedForAdmission;
                if (physicalPressure) break;
            }
            fastLockNs = NowNs() - fastLockStart;
            }
            if (fastAdmission && (fastRowsReady || evictedForAdmission > 0 || evictionNoVictim || evictionAppendFailed)) {
                std::fprintf(stderr, "[Prefix SSD] v4 fast-admission: site=admission recompute_ms=%.1f roots_walked=%zu "
                             "victims=%llu locked_ms=%.1f total_ms=%.1f no_victim=%d append_failed=%d physical_pressure=%d.\n",
                             double(fastRecomputeNs) / 1e6, fastWalked, (unsigned long long)evictedForAdmission,
                             double(fastLockNs) / 1e6, double(NowNs() - fastPassStart) / 1e6,
                             evictionNoVictim ? 1 : 0, evictionAppendFailed ? 1 : 0, physicalPressure ? 1 : 0);
                std::fflush(stderr);
            }
            // ERASE is durable and the projection is already retired. Notify
            // only after pin/catalog mutation and the pin mutex are released.
            for (const auto &entry : erasedForAdmission)
                impl->NotifyCheckpointErased(entry.first, entry.second);
            if (evictionAppendFailed || evictionNoVictim) {
                if (physicalPressure) impl->RequestCompact();
                return false;
            }
            if (evictedForAdmission > 0) {
                // R3：一行可观测日志（每次“为新写让路”的准入淘汰打一次），便于真机上把
                // “为什么旧检查点没了”直接归因到准入，而不是事后猜。
                std::fprintf(stderr,
                    "[Prefix SSD] v4 admission eviction: evicted_for_admission=%llu "
                    "incoming=%llu live=%llu quota=%llu.\n",
                    (unsigned long long)evictedForAdmission,
                    (unsigned long long)incoming, (unsigned long long)LiveBytes(),
                    (unsigned long long)impl->quota);
                std::fflush(stderr);
            }
            const uint64_t projectedPhysical =
                impl->ProjectedPhysicalBytes(objects, existing, key, checkpointMetadata);
            if (projectedPhysical > impl->quota) {
                // Logical ERASE above is allowed on the admission path, but
                // physical unlink/rewrite belongs to the background compactor.
                // Do not make the request thread wait for it; retry later after
                // the durable log has been compacted.
                impl->RequestCompact();
                error = "capacity_unavailable";
                return false;
            }
        }
        impl->quotaNs += NowNs() - quotaStart;
        // Route B：描述符带 chain_tip（PrepareStagedJob 写入）时，root 直接指向状态对象，
        // 不再生成"随上下文增长"的 ObjectList 清单——可达性由块链索引 + 状态引用决定。
        bool chainRoot = false;
        Json chainDescriptor;
        if (checkpointMetadata != nullptr &&
            checkpointMetadata->find("\"chain_tip\"") != std::string::npos) {
            std::string descriptorError;
            chainDescriptor = Json::parse(*checkpointMetadata, descriptorError);
            chainRoot = descriptorError.empty() && chainDescriptor.is_object() &&
                chainDescriptor["chain_tip"].is_string() &&
                prefix_v4::IsDigest(chainDescriptor["chain_tip"].string_value()) &&
                chainDescriptor["state"].is_string() &&
                prefix_v4::IsDigest(chainDescriptor["state"].string_value());
        }
        // Route B：catalog 的块行随提交一起写。块对象**一次写、永不重写**——已经存在的块
        // （WriteObjectOwned 会跳过）不再重复写行，也不计入本次新写字节，故每检查点的
        // DB 写入 / roots.bytes 增量都是 O(新增块数)，不是 O(context)。
        bool hasBlockObject = false;
        for (const auto &object : objects) {
            if (object.kind == prefix_v4::kObjectBlock) { hasBlockObject = true; break; }
        }
        if (chainRoot && !hasBlockObject) { error = "v4_block_missing"; return false; }
        // §③ 注意顺序：清单与总字节数必须在**移动字节之前**算好，否则 move 之后 size 变 0。
        std::vector<ObjectListEntry> list;
        for (const auto &object : objects) {
            // 只引用条目：长度来自 bytesSize（对象已在库里）；新写条目：来自实际字节数。
            const uint64_t size = object.bytes.empty() ? object.bytesSize : object.bytes.size();
            if (size == 0) { error = "v4_object_size"; return false; }
            list.push_back({object.kind, size, object.hash});
        }
        // 只引用对象的**批量**存在性校验：一次查询一批 hash（500/批），而不是每个 hash 一次
        // `HasObject()`。后者在 32K 检查点上约 896 次 prepare+query，实测让 commit 段多花
        // 20~35 ms（2026-09-30 实测：commit 321 → 337/350）。批量后同规模只需 2 次查询。
        {
            std::vector<std::string> refHashes;
            for (const auto &object : objects) {
                if (object.bytes.empty()) refHashes.push_back(object.hash);
            }
            constexpr size_t kVerifyBatch = 500;
            for (size_t base = 0; base < refHashes.size(); base += kVerifyBatch) {
                const size_t count = std::min(kVerifyBatch, refHashes.size() - base);
                std::string sql = "SELECT hash FROM objects WHERE hash IN (";
                for (size_t i = 0; i < count; ++i) sql += (i == 0 ? "?" : ",?");
                sql += ")";
                Statement query(impl->database->value, sql.c_str());
                for (size_t i = 0; i < count; ++i) query.Text(int(i + 1), refHashes[base + i]);
                std::set<std::string> found;
                while (query.Row()) found.insert(query.Text(0));
                for (size_t i = 0; i < count; ++i) {
                    if (!found.count(refHashes[base + i])) { error = "v4_object_missing"; return false; }
                }
            }
        }
        // §17.1 持久性：v4 提交协议的记录级注入点（只在 FASTLLM_DISK_PREFIX_CACHE_TESTING
        // 构建里生效；产品库不定义该宏 → FaultPoint 为空操作，行为不变）。
        // 语义分界：v4_data_* 之前 = 数据未同步；v4_data_synced 之后 = 数据已同步但没有
        // PREPARE/COMMIT → 只能作为未被发布的对象存在；v4_commit_after / v4_sqlite_* 之后 =
        // COMMIT 已同步 → 必须能被重放（§17.1"COMMIT synced但SQL未完成可幂等重放"）；
        // v4_ack_before = 已持久化且已入索引，只是调用方没收到确认。
        FaultPoint("v4_data_write_before");
        impl->pendingObjects.reserve(impl->pendingObjects.size() + objects.size());
        const uint64_t writeStart = NowNs();
        uint64_t total = 0;
        std::vector<DiskPrefixV4Store::BlockRow> blockRows;
        // 1b（2026-10-03）：块对象内的 kv 引用**必须在字节被移交之前**收集。
        // 血泪教训：发布前的检测原先放在 FlushObjects 之后从 `objects[i].bytes` 里解析块对象，
        // 而 `WriteObjectOwned` 在**新写**路径上已把 bytes move 走（`std::move(bytes)` 进
        // pendingObjects）⇒ 新块永远解析不到、检测等于没做（只有"去重命中/已在待写"这两条
        // 提前返回的路径才保得住 bytes，r21-a1 的 5760 条点名正是来自这类块）。现改为在此处
        // （字节完整时）收集，FlushObjects 之后只做可达性判定。
        std::vector<std::string> blockRefs;
        for (auto &object : objects) {
            if (object.bytes.empty()) continue;   // 只引用：已在上面的批量校验里确认存在
            const uint64_t size = object.bytes.size();
            // Route B：块行必须在字节被 move 走之前从块对象解析出来（内容自洽校验）。
            DiskPrefixV4Store::BlockRow row;
            bool haveRow = false;
            if (chainRoot && object.kind == prefix_v4::kObjectBlock) {
                std::string parseError;
                const Json block = Json::parse(object.bytes, parseError);
                std::string rowError;
                if (parseError.empty() && ParseBlockRow(block, row, rowError)) {
                    row.identity = identity; row.key = key;
                    row.objectHash = object.hash; row.bytes = size;
                    haveRow = true;
                    Refs refs; CollectRefs(block["kv"], refs);
                    for (const auto &ref : refs) blockRefs.push_back(ref.first);
                } else {
                    error = (!rowError.empty() ? rowError : "v4_block_object");
                    impl->writeObjectNs += NowNs() - writeStart;
                    return false;
                }
            }
            bool staged = false;
            if (!impl->WriteObjectOwned(object.hash, object.kind, std::move(object.bytes), error,
                                        &staged)) {
                impl->writeObjectNs += NowNs() - writeStart;
                return false;
            }
            // 只有本次真的新写的对象才计入 roots.bytes：已存在的块/分片不重复记账，
            // 才能与重建时的"唯一可达对象"口径一致（配额/淘汰判断依赖它）。
            if (!staged) continue;
            total += size;
            if (haveRow) blockRows.push_back(std::move(row));
        }
        impl->writeObjectNs += NowNs() - writeStart;
        const uint64_t listStart = NowNs();
        std::string rootHash;
        if (chainRoot) {
            // 状态对象本体就是 root：它已在 objects 里（新写或只引用），Load 由 ObjectBytes 取回。
            rootHash = chainDescriptor["state"].string_value();
            bool present = false;
            for (const auto &entry : list) {
                if (entry.hash == rootHash) { present = true; break; }
            }
            if (!present) { error = "v4_block_state_missing"; return false; }
            impl->rootListNs += NowNs() - listStart;
        } else {
            const std::vector<char> listBytes = EncodeObjectList(list);
            const std::string listText(listBytes.data(), listBytes.size());
            // S1-2 修复（2026-10-02，对标审计 P0-2）：清单也要与解码侧对称校验。
            // 解码侧在 v4_object_list_count / v4_object_list_bytes 处硬拒（上限 4096 条 / 8 MiB），
            // 而写入侧此前完全不校验 ⇒ 长上下文（约 13 万 token 起）会"发布成功但读不回"。
            // （Route B 的检查点不再走清单，故这条限制只约束旧的扁平路径。）
            Check(list.size() <= prefix_v4::kMaxObjectListEntries, "v4_manifest_exceeds_decode_limit");
            Check(listText.size() <= prefix_v4::kMaxObjectBytes, "v4_manifest_exceeds_decode_limit");
            rootHash = DiskPrefixCache::Digest(listText);
            // The list object is itself stored, so the root hash resolves to bytes.
            if (!impl->WriteObject(rootHash, kObjectKvNode, listText, error)) return false;
            impl->rootListNs += NowNs() - listStart;
        }
        // 数据（含清单）先落盘，再写 PREPARE/COMMIT：一次 fdatasync 覆盖本批对象。
        const uint64_t flushStart = NowNs();
        if (!impl->FlushObjects(error)) return false;
        // 发布前的块链可达性校验（必须在 FlushObjects 之后，否则本批新对象尚未登记进 objects 表，
        // 会被误判为悬空）。
        // 真机根因（r13/r16-a1/r17-a1/r19-a1/r21-a1 五轮同点丢点，已实证到对象级）：模型侧累计的
        // state.kv 引用了被静默取消/丢批的作业分片，而 chain 模式既不登记这些 kv 引用、发布前也不校验
        // ⇒ root 照常发布、却引用一个**从未写入**的对象；之后任何"采用持久父记录"的请求都取不回它
        // （`v4_object_missing` → `v4_block_chunk_missing`）⇒ 退化成整段上下文重拷 288MiB > pin 预算
        // 256MiB ⇒ 必丢（`write_skipped reason=gpu_pin_budget`）。
        //
        // 1b：本判定由"只点名不拦"改为 **fail-closed**。
        // 硬不变量：**可见 ⇒ 必可读**。判据必须精确，否则会打死合法提交：
        //   可达 = ①已在 objects 表；或 ②在待写批次 pendingObjects（IsPending）；或
        //          ③仍被 FIFO 内在途作业持有（job->staged / job->objects 尚未写盘）。
        // ③ 是此前把本判定降级的原因：routeb_probe 的**合法**用法正是"引用仍在途/随后补齐"，
        // 老代码只查 ①② ⇒ 误判。现在三者皆不成立才拒绝发布（该引用永远不会落盘）。
        // 被拒绝时本函数在写 PREPARE/COMMIT 之前返回，属于代码已支持的
        // "v4_data_synced 之后：数据已同步但没有 COMMIT ⇒ 只能作为未被发布的对象存在"状态。
        {
            // blockRefs 已在字节被移交之前收集完毕（见上方块行解析处），此处只做可达性判定。
            // ③ FIFO 内在途作业持有：只对"查不到且非 pending"的少量引用做扫描（冷路径，不进入热路径）。
            auto inFlightStaged = [&](const std::string &hash) -> bool {
                std::lock_guard<std::mutex> lock(impl->pipelineMutex);
                for (const auto &job : impl->pipeline) {
                    if (!job) continue;
                    for (const auto &item : job->staged)
                        if (item.first == hash) return true;
                    for (const auto &object : job->objects)
                        if (!object.bytes.empty() && object.hash == hash) return true;
                }
                return false;
            };
            std::vector<std::string> unreachable;
            constexpr size_t kBlockVerifyBatch = 500;
            for (size_t base = 0; base < blockRefs.size(); base += kBlockVerifyBatch) {
                const size_t count = std::min(kBlockVerifyBatch, blockRefs.size() - base);
                std::string sql = "SELECT hash FROM objects WHERE hash IN (";
                for (size_t i = 0; i < count; ++i) sql += (i == 0 ? "?" : ",?");
                sql += ")";
                Statement query(impl->database->value, sql.c_str());
                for (size_t i = 0; i < count; ++i) query.Text(int(i + 1), blockRefs[base + i]);
                std::set<std::string> found;
                while (query.Row()) found.insert(query.Text(0));
                for (size_t i = 0; i < count; ++i) {
                    const std::string &ref = blockRefs[base + i];
                    if (found.count(ref)) continue;
                    if (impl->IsPending(ref)) continue;
                    if (inFlightStaged(ref)) continue;
                    unreachable.push_back(ref);
                }
            }
            if (!unreachable.empty()) {
                ++impl->publishRefusedCheckpoints;
                for (const auto &hash : unreachable) {
                    ++impl->publishDanglingRefs;
                    std::fprintf(stderr,
                                 "[Prefix SSD] v4 publish refused (dangling kv ref ⇒ checkpoint withheld): "
                                 "hash=%s key=%s total=%llu.\n",
                                 hash.c_str(), key.c_str(),
                                 (unsigned long long)impl->publishDanglingRefs);
                }
                error = "v4_publish_dangling_ref";
                return false;
            }
        }
        impl->flushNs += NowNs() - flushStart;
        FaultPoint("v4_data_synced");   // 数据（含清单对象）已 fdatasync，尚无 PREPARE/COMMIT

        const uint64_t prepareStart = NowNs();
        const uint64_t firstBatch = impl->sequence;
        PrepareRecord prepare;
        prepare.key = key;
        prepare.identity = identity;
        prepare.rootHash = rootHash;
        prepare.firstBatch = firstBatch;
        prepare.sequence = impl->sequence + 1;
        prepare.lastBatch = impl->sequence;
        const std::vector<char> prepareBytes = EncodePrepare(prepare);
        const std::string prepareText(prepareBytes.data(), prepareBytes.size());
        uint64_t prepareOffset = 0;
        FaultPoint("v4_prepare_before");
        struct HoldRoll { bool &flag; ~HoldRoll() { flag = false; } } holdRoll{impl->segmentHoldRoll};
        if (!impl->AppendLiveRecord(kRecordManifestPrepare, ++impl->sequence,
                                    prepareText, &prepareOffset, error)) {
            --impl->sequence;
            return false;
        }
        impl->segmentHoldRoll = impl->segmented;   // released by holdRoll when this commit returns
        FaultPoint("v4_prepare_after");   // PREPARE 已同步但没有 COMMIT → 只能算 UNCOMMITTED_TAIL
        impl->prepareNs += NowNs() - prepareStart;
        // A checkpoint descriptor is durable before the final COMMIT.  If this
        // append fails, the root has never been published and recovery cannot
        // observe a roots/checkpoints split.
        if (checkpointMetadata != nullptr) {
            prefix_v4::MetadataRecord metadataRecord;
            metadataRecord.sequence = impl->sequence + 1;
            metadataRecord.key = "checkpoint:" + key;
            metadataRecord.value = identity + "\n" + *checkpointMetadata;
            if (metadataRecord.value.size() > prefix_v4::kMaxMetadataValueBytes) {
                error = "metadata_value_too_large";
                --impl->sequence;
                return false;
            }
            const std::vector<char> metadataBytes = EncodeMetadata(metadataRecord);
            if (!impl->AppendLiveRecord(kRecordMetadata, ++impl->sequence,
                                        std::string(metadataBytes.data(), metadataBytes.size()),
                                        nullptr, error)) {
                --impl->sequence;
                return false;
            }
        }
        const uint64_t commitStart = NowNs();
        prefix_v4::CommitRecord commit;
        commit.sequence = impl->sequence + 1;
        commit.prepareOffset = prepareOffset;
        commit.prepareBytes = prefix_v4::AlignUp(kRecordHeaderBytes + prepareText.size(), error);
        if (!error.empty()) return false;
        commit.prepareDigest = RecordDigest(kRecordManifestPrepare, prepareText);
        commit.rootHash = rootHash;
        const std::vector<char> commitBytes = EncodeCommit(commit);
        FaultPoint("v4_commit_before");
        if (!impl->AppendLiveRecord(kRecordManifestCommit, ++impl->sequence,
                                    std::string(commitBytes.data(), commitBytes.size()), nullptr,
                                    error)) {
            --impl->sequence;
            return false;
        }
        impl->segmentHoldRoll = false;
        FaultPoint("v4_commit_after");   // COMMIT 已同步：重开必须幂等重放（§17.1）
        impl->commitNs += NowNs() - commitStart;
        const uint64_t rootTxStart = NowNs();
        {
            Transaction transaction(*impl->database);
            Statement upsert(impl->database->value,
                             "INSERT OR REPLACE INTO roots(identity,key,root,sequence,bytes,live) "
                             "VALUES(?,?,?,?,?,1)");
            upsert.Text(1, identity); upsert.Text(2, key); upsert.Text(3, rootHash);
            upsert.Number(4, impl->sequence); upsert.Number(5, total); upsert.Run();
            if (impl->segmented) impl->fastAdmissionStale = true;   // DESIGN-v3-segments C9
            if (checkpointMetadata != nullptr) {
                // S2-4 写入点 1/4（提交路径）：统一走 Impl::InsertCheckpointRow，
                // 保证 metadata 与 family/kind/length/dtype/provenance/prefix/created
                // 索引列在同一事务里同步写入。
                Impl::InsertCheckpointRow(*impl->database, identity, key, *checkpointMetadata,
                                          impl->sequence);
            }
            // Route B 写入点（提交路径）：块行与 root 在同一事务里落进投影。
            for (const auto &row : blockRows) Impl::InsertBlockRow(*impl->database, row);
            impl->SetMeta("sequence", std::to_string(impl->sequence));
            // §17.1（2026-09-30 修复③）：把控制日志长度作为"投影水位"记进**同一个事务**。
            // 崩溃点落在"COMMIT 记录已同步、本事务未提交"时，水位会停在上一笔 → 下次 Open
            // 一比对即可发现投影落后于日志，从而自动重建（实测修前 on_open=0 / after_rebuild=1）。
            impl->RefreshLogWatermark();
            FaultPoint("v4_sqlite_commit_before");   // COMMIT 已同步、投影(SQLite)未提交
            transaction.Commit();
            FaultPoint("v4_sqlite_commit_after");
        }
        impl->rootTxNs += NowNs() - rootTxStart;
        // FIFO eviction: over quota, retire the oldest live checkpoint first, so
        // space is reclaimed by an explicit ERASE record rather than silently.
        const uint64_t evictStart = NowNs();
        std::vector<std::pair<std::string, std::string>> erasedAfterCommit;
        bool evictionAppendFailed = false;
        const bool fastEvict = V4FastAdmission();
        std::vector<Impl::RootBytesRow> fastEvictRows;
        bool fastEvictRowsReady = false;
        size_t fastEvictWalked = 0;
        uint64_t fastEvictRecomputeNs = 0, fastEvictLockNs = 0, fastEvictVictims = 0;
        if (fastEvict && !impl->segmented && impl->quota > 0 && impl->fastAdmissionStale && LiveBytes() > impl->quota) {
            std::string accountingError;
            if (!impl->ComputeLiveBytesRows(fastEvictRows, fastEvictWalked, accountingError)) {
                error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                evictionAppendFailed = true;
            } else {
                fastEvictRowsReady = true;
            }
            fastEvictRecomputeNs = NowNs() - evictStart;
        }
        {
        std::lock_guard<std::mutex> pinLock(impl->pins->mutex);
        const uint64_t fastEvictLockStart = NowNs();
        if (fastEvictRowsReady) {
            impl->ApplyLiveBytesRows(fastEvictRows);
            impl->fastAdmissionStale = false;
        }
        if (!fastEvict && !impl->segmented && impl->quota > 0 && LiveBytes() > impl->quota) {
            std::string accountingError;
            if (!impl->RecomputeLiveBytesForQuota(accountingError)) {
                error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                evictionAppendFailed = true;
            }
        }
        while (!impl->segmented && impl->quota > 0 && LiveBytes() > impl->quota) {
            if (evictionAppendFailed) break;
            std::string victimIdentity, victimKey;
            std::unique_ptr<Lease> victimLease;
            if (!impl->OldestUnpinnedRoot(victimIdentity, victimKey, victimLease)) break;
            prefix_v4::EraseRecord erase;
            erase.sequence = impl->sequence + 1;
            erase.key = victimKey;
            erase.identity = victimIdentity;
            const std::vector<char> eraseBytes = EncodeErase(erase);
            if (!impl->AppendLiveRecord(kRecordErase, ++impl->sequence,
                                        std::string(eraseBytes.data(), eraseBytes.size()), nullptr,
                                        error)) {
                --impl->sequence;
                evictionAppendFailed = true;
                break;
            }
            Statement retire(impl->database->value,
                             "UPDATE roots SET live=0 WHERE identity=? AND key=?");
            retire.Text(1, victimIdentity); retire.Text(2, victimKey); retire.Run();
            impl->SetMeta("sequence", std::to_string(impl->sequence));
            impl->RefreshLogWatermark();   // 修复③：root 事务之后再追加 ERASE，必须刷新水位
            erasedAfterCommit.emplace_back(victimKey, victimIdentity);
            if (fastEvict) {
                impl->fastAdmissionStale = true;   // C1
                ++fastEvictVictims;
            } else {
            std::string accountingError;
            if (!impl->RecomputeLiveBytesForQuota(accountingError)) {
                error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                evictionAppendFailed = true;
            }
            }
        }
        fastEvictLockNs = NowNs() - fastEvictLockStart;
        }
        if (fastEvict && (fastEvictRowsReady || fastEvictVictims > 0 || evictionAppendFailed)) {
            std::fprintf(stderr, "[Prefix SSD] v4 fast-admission: site=post_commit recompute_ms=%.1f roots_walked=%zu "
                         "victims=%llu locked_ms=%.1f total_ms=%.1f append_failed=%d.\n",
                         double(fastEvictRecomputeNs) / 1e6, fastEvictWalked, (unsigned long long)fastEvictVictims,
                         double(fastEvictLockNs) / 1e6, double(NowNs() - evictStart) / 1e6, evictionAppendFailed ? 1 : 0);
            std::fflush(stderr);
        }
        for (const auto &entry : erasedAfterCommit)
            impl->NotifyCheckpointErased(entry.first, entry.second);
        if (evictionAppendFailed) return false;
        impl->evictNs += NowNs() - evictStart;
        FaultPoint("v4_ack_before");   // ack 丢失：已持久化、已入索引，但调用方永远收不到确认
        // A2：只统计带描述符的 prefix 检查点（checkpointMetadata != nullptr）；encoder/图片
        // 走无描述符路径，不计入，保证与模型侧 expected_checkpoints 同口径。
        if (checkpointMetadata != nullptr) ++impl->durableCommits;
        if (outSequence != nullptr) *outSequence = impl->sequence;
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

bool DiskPrefixV4Store::Load(const std::string &key, std::string &identity,
                             std::vector<std::string> &objectBytes, std::string &error) {
    // S2-4：旧签名行为保持不变——不过滤 identity（等价于历史行为），仅供既有调用方
    // 与测试使用；新代码请用带 expectedIdentity 的重载。
    return LoadWithIdentity(key, std::string(), identity, objectBytes, nullptr, error);
}

bool DiskPrefixV4Store::Load(const std::string &key, const std::string &expectedIdentity,
                             std::vector<std::string> &objectBytes, std::string &error) {
    std::string observedIdentity;
    return LoadWithIdentity(key, expectedIdentity, observedIdentity, objectBytes, nullptr, error);
}

bool DiskPrefixV4Store::Load(const std::string &key, const std::string &expectedIdentity,
                             std::vector<std::string> &objectBytes,
                             std::vector<std::string> &objectHashes, std::string &error,
                             bool manifestOnly,bool publishedOnly,
                             const DiskPrefixCache::ReadProgress &progress, Json *assembledState) {
    // A2：多回传对象摘要，供调用方免去重算 SHA-256。
    objectHashes.clear();
    std::string observedIdentity;
    if (!LoadWithIdentity(key, expectedIdentity, observedIdentity, objectBytes, &objectHashes, error,
                          manifestOnly,publishedOnly,progress,assembledState)) {
        objectHashes.clear();
        return false;
    }
    // 一一对应是同序 push 的结构性结果；这里兜一道，防将来改动把两者错位 ⇒ fail-closed。
    if (objectHashes.size() != objectBytes.size()) {
        objectHashes.clear();
        objectBytes.clear();
        error = "v4_object_hash_count_mismatch";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 只读计时埋点：把 Import 的 tLoad（基线实测
// 3580.7 ms）拆成「等盘 / 解析 JSON / catalog 查询」三项，用来判断该并行化读盘还是
// 并行化校验解析。**不改语义、不动数据路径**；若解析不出读数，则不动"该并行化哪一项"这个判断。
//
// 开关复用既有的请求级 trace env（`FASTLLM_PREFIX_CACHE_CAPTURE_TRACE=1`），**默认关**
// ⇒ 不开时下面全部是 4 个静态 bool + 4 次函数调用，零打印、零行为变更。
// 为什么用 thread_local 而非全局：Import 在读侧线程上同步执行（`qwen3_5_persistent.inc:2814`
// 直接 `Import(ctx,candidate)`，进程内无 std::thread/detach）。thread_local 顺带排除
// 背景自检线程混入请求读数。计数按对象**去重后**累计（与 `fetched` 同口径）。
namespace {
struct V4LoadProfile {
    uint64_t objectNs = 0, catalogNs = 0, parseNs = 0, chainNs = 0;
    size_t objectCount = 0, parseCount = 0, catalogCount = 0;
    uint64_t objectBytes = 0;
};
thread_local V4LoadProfile gV4LoadProfile;
// Read the policy on first use, after Python has exported CLI options. A global
// initializer runs during dlopen and would freeze the default before parsing.
static bool V4LoadProfileEnabled() {
    static const bool enabled = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_CAPTURE_TRACE");
        const bool active = PrefixCacheDebugEnabled() && text != nullptr &&
                            text[0] != '\0' && text[0] != '0';
        if (active) {
            std::fprintf(stderr, "[Prefix SSD][load-profile] enabled=1 pid=%d tid=%llu.\n",
                         (int)getpid(), (unsigned long long)std::hash<std::thread::id>()(std::this_thread::get_id()));
            std::fflush(stderr);
        }
        return active;
    }();
    return enabled;
}
// RAII：把一段区间累加到某个计数器上（只在开关打开时计时）。
struct V4Timer {
    uint64_t &target;
    bool active;
    std::chrono::steady_clock::time_point begin;
    V4Timer(uint64_t &t, bool on) : target(t), active(on) {
        if (active) begin = std::chrono::steady_clock::now();
    }
    ~V4Timer() {
        if (!active) return;
        target += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
    }
};
// 并行取对象的"一次在途读"。只被 dispatchObjectReads 的同线程消费（无跨线程访问）。
struct V4LoadInFlight {
    std::string hash;
    std::string bytes;
    std::string error;
    bool ok = false;
    bool claimed = false;     // 已被 fetched 认领（去重后仍需占位，保持顺序）
    bool read = false;        // 是否需要真的读盘
};
}
// 并行对象读取。
// 背景：Import 的 tLoad 基线实测 3580.7 ms，是"单线程顺序读 520 个对象"：每个对象 =
// catalog 点查 + 读盘 + 逐对象 SHA-256 校验（copy_ms 只有 3.4 ms，拷贝不是瓶颈）。
// 这里把这批对象按 N 条线程切片并行读回，**顺序、字节、错误码、去重语义全部不变**：
//   ① 先在调用线程按输入顺序收集 + 去重（与串行版 pushObject 的 fetched 同口径）；
//   ② 再并行读；
//   ③ 最后在调用线程按①的顺序装配 objectBytes / outHashes。
// 线程安全依据：`V4ReadRecordCached` 是 thread_local 的 16 槽记录缓存
// （本文件 :2502 `V4RecordCacheSlot()`），各线程各持一份，无共享可变状态。
// 线程数：`FASTLLM_PREFIX_CACHE_LOAD_THREADS`，默认 16，夹在 [1,32]；=1 时保持原串行行为。
// 默认从 8 提到 16（2026-10-17，B21 T0）：同一 357.5 MB 检查点 load 690 ms -> 538 ms，
// rchar/payload 与内容摘要不变；16/24/32 为 538/522/516 ms，16 已吃到大部分收益且内存更稳。
size_t V4LoadThreads() {
    static const size_t value = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_LOAD_THREADS");
        long parsed = text != nullptr ? std::atol(text) : 0;
        if (parsed <= 0) parsed = 16;
        if (parsed > 32) parsed = 32;
        return size_t(parsed);
    }();
    return value;
}
bool DiskPrefixV4Store::LoadWithIdentity(const std::string &key,
                                         const std::string &expectedIdentity,
                                         std::string &identity,
                                         std::vector<std::string> &objectBytes,
                                         std::vector<std::string> *outHashes,
                                         std::string &error,
                                         bool manifestOnly,bool publishedOnly,
                                         const DiskPrefixCache::ReadProgress &progress, Json *assembledState) {
    if (assembledState != nullptr) *assembledState = Json();
    const bool profile = V4LoadProfileEnabled();
    gV4LoadProfile = V4LoadProfile();
    {
        auto &split = V4Split();
        split.diskNs = 0; split.digestNs = 0; split.extractNs = 0;
        split.records = 0; split.recordBytes = 0; split.extractCalls = 0;
        split.directReadCalls = 0; split.directReadBytes = 0;
        split.path = -1;
    }
    auto emitReadSplit = [](int pathKind) {
        auto &split = V4Split();
        FastllmEvent("cache", 0, "cache_read_profile",
                     "path=%s disk_ms=%.1f digest_ms=%.1f extract_ms=%.1f records=%llu "
                     "record_mb=%.1f extract_calls=%llu io_mode=%s direct_read_calls=%llu direct_read_bytes=%llu",
                     pathKind == 0 ? "flat" : "routeB",
                     double(split.diskNs.load())/1e6, double(split.digestNs.load())/1e6,
                     double(split.extractNs.load())/1e6,
                     (unsigned long long)split.records.load(),
                     double(split.recordBytes.load())/1048576.0,
                     (unsigned long long)split.extractCalls.load(),
                     V4DirectIOEnabled() ? "direct" : "buffered",
                     (unsigned long long)split.directReadCalls.load(),
                     (unsigned long long)split.directReadBytes.load());
    };
    const auto loadWallStart = std::chrono::steady_clock::now();
    // Direct DiskPrefixV4Store::Load callers do not have a ReadSession object
    // carrying the lifecycle lease. Hold a shared lease for the entire read
    // so Compact cannot rename/rewrite the log while an object is in flight.
    std::unique_ptr<Lease> directReadLease;
    if (std::this_thread::get_id() != impl->actorId)
        directReadLease.reset(new Lease(impl->root / ".lifecycle", LOCK_SH));
    if (!publishedOnly) impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    const auto loadWallDrained = std::chrono::steady_clock::now();
    error.clear();
    try {
        std::unique_ptr<Database> published;
        sqlite3 *catalog = impl->database->value;
        if (publishedOnly) {
            published = std::make_unique<Database>(impl->dbPath,true);
            catalog = published->value;
        }
        objectBytes.clear();
        if (outHashes != nullptr) outHashes->clear();
        auto cancelled = [&]() {
            if (progress && !progress(0,0)) { error = "ssd_read_cancelled"; return true; }
            return false;
        };
        if (cancelled()) return false;
        // S2-4：旧实现只按 key 取 "sequence 最新" 的 root，同一 key 属于不同 identity 时
        // 会解析到别人的 root（错误命中）。带 expectedIdentity 时把 identity 加进 WHERE。
        std::string rootHash;
        {
            std::string sql = "SELECT identity,root FROM roots WHERE key=? AND live=1";
            if (!expectedIdentity.empty()) sql += " AND identity=?";
            sql += " ORDER BY sequence DESC LIMIT 1";
            Statement query(catalog, sql.c_str());
            query.Text(1, key);
            if (!expectedIdentity.empty()) query.Text(2, expectedIdentity);
            if (!query.Row()) { error = "v4_root_missing"; return false; }
            identity = query.Text(0);
            rootHash = query.Text(1);
        }
        // Direct DiskPrefixV4Store::Load callers have no ReadSession-owned pin.
        // Pin after resolving the live root and keep the token until every
        // object has been verified, so admission FIFO cannot retire it while
        // this read is in flight. ReadSession loads may hold a second pin;
        // that is intentional and harmless.
        std::shared_ptr<void> directPin;
        directPin = PinCheckpoint(key, identity, error, false);
        if (!directPin) return false;
        std::string rootText;
        { V4Timer catalogTimer(gV4LoadProfile.catalogNs, profile);
          gV4LoadProfile.catalogCount++;
          if (!impl->ObjectBytes(rootHash, rootText, error, catalog)) return false; }
        if (progress && !progress(rootText.size(),rootText.size())) { error = "ssd_read_cancelled"; return false; }
        std::vector<ObjectListEntry> list;
        std::string listError;
        if (DecodeObjectList(rootText.data(), rootText.size(), list, listError)) {
            // 旧扁平路径（已装配对象 / encoder）：语义与字节完全不变。
            for (const auto &entry : list) {
                if (cancelled()) return false;
                if (progress && !progress(0,entry.bytes)) { error = "ssd_read_cancelled"; return false; }
                std::string bytes;
                { V4Timer objectTimer(gV4LoadProfile.objectNs, profile);
                  gV4LoadProfile.objectCount++;
                  if (!impl->ObjectBytes(entry.hash, bytes, error, catalog)) return false; }
                if (progress && !progress(bytes.size(),0)) { error = "ssd_read_cancelled"; return false; }
                if (bytes.size() != entry.bytes) { error = "v4_object_size"; return false; }
                if (outHashes != nullptr) outHashes->push_back(entry.hash);   // A2：同序回传摘要
                gV4LoadProfile.objectBytes += bytes.size();
                objectBytes.push_back(std::move(bytes));
            }
            if (profile) {
                const V4LoadProfile &p = gV4LoadProfile;
                std::fprintf(stderr,"[Prefix SSD][load-profile] path=flat objects=%zu object_mb=%.1f "
                    "read_ms=%.1f catalog_ms=%.1f(%zu) parse_ms=%.1f(%zu) chain_ms=%.1f.\n",
                    p.objectCount, double(p.objectBytes)/1048576.0, double(p.objectNs)/1e6,
                    double(p.catalogNs)/1e6, p.catalogCount, double(p.parseNs)/1e6, p.parseCount,
                    double(p.chainNs)/1e6);
                std::fflush(stderr);
            }
            emitReadSplit(0);
            return true;
        }
        // Route B：root 是紧凑状态对象本身（JSON，含 chain_tip/block_count）。逐块走链取回
        // 块对象，再取回块引用的 KV 分片与状态自身的 linear/spec 分片。任一块/分片缺失都
        // fail-closed（绝不静默返回不完整对象集）。
        std::string parseError;
        const Json state = Json::parse(rootText, parseError);
        if (!parseError.empty() || !state.is_object() || !state["chain_tip"].is_string() ||
            !state["block_count"].is_number()) {
            error = "v4_root_object";
            return false;
        }
        if (outHashes != nullptr) outHashes->push_back(rootHash);   // A2：同序回传摘要
        objectBytes.push_back(rootText);
        std::set<std::string> fetched{rootHash};
        // 记录级预取结果（2026-10-17，B21）：hash → 已校验对象字节。块循环/装配/进度/诊断
        // 语义不变，只把"每个对象各自读整条记录"换成"每条记录读一次"。
        std::map<std::string, std::string> prefetchedBytes;
        std::mutex prefetchedMutex;
        // 方案1：
        // 对象可能正处于"OBJECT_BATCH 记录已追加（FlushObjectGroup :3339）、catalog objects 行
        // 尚未登记（FlushObjectGroup :3346-3356）"的窗口内。旧实现在这里把 ObjectBytes 的**真实
        // 错误覆盖成通用标签**并直接 fail-closed ⇒ 上层（模型侧父采用）退化成"整段上下文重拷"：
        // 真机 r13/r16 实测在 tokens=147456 必丢（需求 288MiB > 预算 256MiB，defer 等满 2s 超时）。
        // 现在：先等在途提交排空，再有界重试；到点仍失败才**如实打印真因**（缺失 hash + 原始错误）
        // 并按原样返回通用错误串（错误码与既有测试口径不变）。绝不无限等待（SGLang #28429 的教训）。
        constexpr uint64_t kObjectReadRetryMs = 2000;   // 对齐 pin defer 默认 2000ms
        auto pushObject = [&](const std::string &hash, std::string &out, const char *missing) {
            if (cancelled()) return false;
            if (!prefix_v4::IsDigest(hash)) { error = missing; return false; }
            if (!fetched.insert(hash).second) { out.clear(); return true; }
            {
                std::lock_guard<std::mutex> lock(prefetchedMutex);
                auto prefetched = prefetchedBytes.find(hash);
                if (prefetched != prefetchedBytes.end()) {
                    out = std::move(prefetched->second);
                    prefetchedBytes.erase(prefetched);
                    if (progress && !progress(out.size(),out.size())) { error = "ssd_read_cancelled"; return false; }
                    if (outHashes != nullptr) outHashes->push_back(hash);
                    gV4LoadProfile.objectBytes += out.size();
                    objectBytes.push_back(out);
                    return true;
                }
            }
            const uint64_t deadline = NowNs() + kObjectReadRetryMs * 1000000ull;
            for (;;) {
                std::string attemptError;
                bool readOk = false;
                { V4Timer objectTimer(gV4LoadProfile.objectNs, profile);
                  gV4LoadProfile.objectCount++;
                  readOk = impl->ObjectBytes(hash, out, attemptError, catalog); }
                if (readOk) {
                    if (progress && !progress(out.size(),out.size())) { error = "ssd_read_cancelled"; return false; }
                    if (outHashes != nullptr) outHashes->push_back(hash);   // A2：同序回传摘要
                    gV4LoadProfile.objectBytes += out.size();
                    objectBytes.push_back(out);
                    return true;
                }
                if (publishedOnly || NowNs() >= deadline) {
                    std::fprintf(stderr, "[Prefix SSD] v4 object unreadable: hash=%s error=%s.\n",
                                 hash.c_str(), attemptError.c_str());
                    error = missing;
                    return false;
                }
                if (!publishedOnly) impl->DrainForRead();
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        };
        // B1：清单-only 模式的**存在性**校验（不取字节）。
        // 为什么必须有它（而不是"反正提交侧还会查一遍"）：提交侧的批量 `HasObject`
        // 发生在**写 worker 的 Commit 里**，失败表现为 `write_skipped: v4_object_missing`
        // ⇒ 那一笔检查点**丢点**。而改动前的语义是 adopt **当场失败** ⇒ 走回退/整段重导
        // ⇒ **不丢点**（这正是 2026-09-30 P0 一轮修出来的性质，见 ERR:242/247）。
        // 所以在 adopt 这一步就把"父分片可读"证出来，保持 fail-closed 的**发生点**不变。
        // 2026-10-05 并行化后，这段校验并入 dispatchObjectReads 的 manifestOnly 分支
        // （同款有界重试：在途提交窗口 + 2000ms 上限，绝不无限等）。
        // 并行取对象（把 3.58 s 的串行读摊到多线程；顺序/字节/错误码/去重语义与串行版逐字一致）。
        // - manifestOnly：只做存在性校验（原 requireObject 语义，`missing` 用调用方给的错误串）；
        // - 否则：读回字节（原 pushObject 语义，不可读时保留原样的 `v4 object unreadable` 诊断行）。
        auto dispatchObjectReads = [&](const Refs &refs, const char *missing) -> bool {
            if (cancelled()) return false;
            std::vector<V4LoadInFlight> items;
            items.reserve(refs.size());
            for (const auto &ref : refs) {
                if (!prefix_v4::IsDigest(ref.first)) { error = missing; return false; }
                V4LoadInFlight item;
                item.hash = ref.first;
                if (fetched.insert(ref.first).second) {
                    item.claimed = true;
                    item.read = !manifestOnly;
                    if (progress && !progress(0,ref.second)) { error = "ssd_read_cancelled"; return false; }
                }
                items.push_back(std::move(item));
            }
            const size_t workerCount = std::min(V4LoadThreads(), items.size());
            // 预取全命中时直接在主线程消费，避免每个块都再起/销毁一批 worker
            // （块循环会调用本函数 370 次量级；线程创建开销本身已成为串行段）。
            bool allPrefetched = !manifestOnly && !items.empty();
            if (allPrefetched) {
                std::lock_guard<std::mutex> lock(prefetchedMutex);
                for (const auto &item : items) {
                    if (!item.read) continue;
                    if (prefetchedBytes.count(item.hash) == 0) { allPrefetched = false; break; }
                }
            }
            if (allPrefetched) {
                std::lock_guard<std::mutex> lock(prefetchedMutex);
                for (auto &item : items) {
                    if (!item.read) continue;
                    auto prefetched = prefetchedBytes.find(item.hash);
                    if (prefetched == prefetchedBytes.end()) { item.ok = false; item.error = "v4_prefetch_unexpected"; continue; }
                    item.bytes = std::move(prefetched->second);
                    prefetchedBytes.erase(prefetched);
                    item.ok = !progress || progress(item.bytes.size(),0);
                    if (!item.ok) item.error = "ssd_read_cancelled";
                }
            } else if (workerCount > 1) {
                std::atomic<size_t> cursor{0};
                std::vector<std::thread> workers;
                workers.reserve(workerCount);
                for (size_t worker = 0; worker < workerCount; ++worker) {
                    workers.emplace_back([&]() {
                        for (;;) {
                            const size_t index = cursor.fetch_add(1, std::memory_order_relaxed);
                            if (index >= items.size()) break;
                            V4LoadInFlight &item = items[index];
                            if (!item.read) continue;   // 去重命中或清单-only：不需要读盘
                            {
                                std::lock_guard<std::mutex> lock(prefetchedMutex);
                                auto prefetched = prefetchedBytes.find(item.hash);
                                if (prefetched != prefetchedBytes.end()) {
                                    item.bytes = std::move(prefetched->second);
                                    prefetchedBytes.erase(prefetched);
                                    item.ok = !progress || progress(item.bytes.size(),0);
                                    if (!item.ok) item.error = "ssd_read_cancelled";
                                    continue;
                                }
                            }
                            try {
                                if (progress && !progress(0,0)) {
                                    item.error = "ssd_read_cancelled";
                                    continue;
                                }
                                const uint64_t deadline = NowNs() + kObjectReadRetryMs * 1000000ull;
                                for (;;) {
                                    std::string attemptError;
                                    bool ok = false;
                                    { V4Timer objectTimer(gV4LoadProfile.objectNs, profile);
                                      gV4LoadProfile.objectCount++;
                                      ok = impl->ObjectBytes(item.hash, item.bytes, attemptError, catalog); }
                                    if (ok) {
                                        item.ok = !progress || progress(item.bytes.size(),0);
                                        if (!item.ok) item.error = "ssd_read_cancelled";
                                        break;
                                    }
                                    if (publishedOnly || NowNs() >= deadline) {
                                        item.error = attemptError;
                                        break;
                                    }
                                    if (!publishedOnly) impl->DrainForRead();
                                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                }
                            } catch (const std::exception &exception) {
                                item.ok = false;
                                item.error = exception.what();
                            } catch (...) {
                                item.ok = false;
                                item.error = "v4_read_worker_exception";
                            }
                        }
                    });
                }
                for (auto &worker : workers) if (worker.joinable()) worker.join();
            } else {
                for (auto &item : items) {
                    if (!item.read) continue;
                    if (cancelled()) return false;
                    {
                        std::lock_guard<std::mutex> lock(prefetchedMutex);
                        auto prefetched = prefetchedBytes.find(item.hash);
                        if (prefetched != prefetchedBytes.end()) {
                            item.bytes = std::move(prefetched->second);
                            prefetchedBytes.erase(prefetched);
                            item.ok = !progress || progress(item.bytes.size(),0);
                            if (!item.ok) item.error = "ssd_read_cancelled";
                            continue;
                        }
                    }
                    const uint64_t deadline = NowNs() + kObjectReadRetryMs * 1000000ull;
                    for (;;) {
                        std::string attemptError;
                        bool ok = false;
                        { V4Timer objectTimer(gV4LoadProfile.objectNs, profile);
                          gV4LoadProfile.objectCount++;
                          ok = impl->ObjectBytes(item.hash, item.bytes, attemptError, catalog); }
                        if (ok) {
                            item.ok = !progress || progress(item.bytes.size(),0);
                            if (!item.ok) item.error = "ssd_read_cancelled";
                            break;
                        }
                        if (publishedOnly || NowNs() >= deadline) { item.error = attemptError; break; }
                        if (!publishedOnly) impl->DrainForRead();
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                }
            }
            // 按调用顺序装配（与串行版逐项一致）
            for (auto &item : items) {
                if (!item.claimed) continue;            // 去重命中：不追加、不报错
                if (manifestOnly) {
                    // 清单-only：claimed 的项只做存在性校验（原 requireObject 语义）。
                    // 注意：items 的 `read` 在 manifestOnly 下为 false，所以这里必须**自己**
                    // 走 HasObject，不能复用 `ok`（`ok` 只在真的读盘时才为 true）。
                    const uint64_t deadline = NowNs() + kObjectReadRetryMs * 1000000ull;
                    bool present = false;
                    for (;;) {
                        if (impl->HasObject(item.hash,catalog)) { present = true; break; }
                        if (publishedOnly || NowNs() >= deadline) break;
                        if (!publishedOnly) impl->DrainForRead();
                        std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    }
                    if (!present) {
                        std::fprintf(stderr, "[Prefix SSD] v4 object missing (manifest-only): hash=%s.\n",
                                     item.hash.c_str());
                        error = missing;
                        return false;
                    }
                    continue;
                }
                if (!item.ok) {
                    std::fprintf(stderr, "[Prefix SSD] v4 object unreadable: hash=%s error=%s.\n",
                                 item.hash.c_str(), item.error.c_str());
                    error = item.error == "ssd_read_cancelled" ? item.error :
                        (!item.error.empty() ? item.error : missing);
                    return false;
                }
                if (outHashes != nullptr) outHashes->push_back(item.hash);
                gV4LoadProfile.objectBytes += item.bytes.size();
                objectBytes.push_back(std::move(item.bytes));
            }
            return true;
        };
        // B1：清单-only 模式**跳过载荷取回**。
        // ①状态对象自己的分片（linear/spec 的 chunks）：adopt 只读 parent["kv"]，不需要它们；
        //   Import 需要 ⇒ Import 走 Load（manifestOnly=false）。
        // 两步都只做存在性校验（不读字节），失败即 fail-closed。
        Refs stateRefs;
        CollectRefs(state, stateRefs);
        const auto setupState0 = std::chrono::steady_clock::now();
        const auto setupState1 = std::chrono::steady_clock::now();
        std::vector<BlockRow> blocks;
        std::string chainError;
        bool chainOk = false;
        { V4Timer chainTimer(gV4LoadProfile.chainNs, profile);
          chainOk = ChainBlocks(state["chain_tip"].string_value(),
                                uint64_t(state["block_count"].number_value()), blocks, chainError,publishedOnly); }
        if (!chainOk) {
            error = chainError.empty() ? "v4_block_missing" : chainError;
            return false;
        }
        const auto setupChain1 = std::chrono::steady_clock::now();
        // 记录级预取（2026-10-17，B21 读取放大修复）。
        // 事实（T0，8 worker，357 MB 检查点）：dispatchObjectReads 按块调用，每次只有 8 个
        // 分片对象且都落在同一条记录里；8 个 worker 各自把这条 7.5 MB 记录读一遍、算一遍
        // 整条 SHA-256，rchar / payload = 5.71x（2.04 GB / 357.5 MB）。这里在块循环前按窗口
        // 把块对象与其 kv 分片按**物理记录偏移**分组，每条记录只读一次、只校验一次记录摘要，
        // 再逐对象取字节放进 prefetchedBytes；块循环、装配顺序、去重、进度回调和错误诊断
        // 全部保持原样，只是消费预取结果。预取失败不改变语义：对象不进入 prefetchedBytes，
        // 仍由既有路径按原样重试并报原始错误。
        const size_t prefetchWindowBlocks = std::max<size_t>(V4LoadThreads() * 3, 8);
        // 块对象解析结果（2026-10-18，B21）：Json::parse + ParseBlockRow + CollectRefs
        // 原本全在主线程块循环里，T0 实测 28 块占 ~138 ms；370 块的真实请求会线性放大。
        // 改为在预取线程内按块并行完成，主循环只消费 refs；解析/校验失败不写 ready，
        // 主线程回退到原解析路径并按原错误串 fail-closed。
        struct PrefetchedBlockPlan {
            bool ready = false;
            Refs refs;
            Json kv;
        };
        std::vector<PrefetchedBlockPlan> blockPlans(blocks.size());
        auto prefetchRecords = [&](size_t begin, size_t end,
                                   const std::vector<std::string> *extraHashes = nullptr) {
            if (manifestOnly) return;
            const auto prefetchWallStart = std::chrono::steady_clock::now();
            struct PrefetchWallPrint {
                std::chrono::steady_clock::time_point start;
                size_t begin, end;
                bool on;
                ~PrefetchWallPrint() {
                    if (!on) return;
                    std::fprintf(stderr, "[Prefix SSD][prefetch-wall] window=[%zu,%zu) wall_ms=%.1f\n",
                                 begin, end,
                                 std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - start).count());
                    std::fflush(stderr);
                }
            } prefetchWallPrint{prefetchWallStart, begin, end,
                                std::getenv("FASTLLM_LOAD_PHASE") != nullptr};
            try {

                struct Want { std::string hash; uint64_t offset = 0; };
                double phaseAMs = 0, phaseBMs = 0;
                const bool phaseOn = std::getenv("FASTLLM_LOAD_PHASE") != nullptr;
                auto fetchGrouped = [&](const std::vector<std::string> &hashes,
                                        std::map<std::string, std::string> &out) {
                    std::vector<Want> wants;
                    wants.reserve(hashes.size());
                    std::vector<size_t> candidates;
                    candidates.reserve(hashes.size());
                    {
                        // 预取运行在独立线程：不能读主线程的 fetched/progress（有竞争）。
                        // 只跳过已经预取过的对象；重复预取最多浪费一次读，语义不受影响。
                        std::lock_guard<std::mutex> lock(prefetchedMutex);
                        for (size_t index = 0; index < hashes.size(); ++index)
                            if (prefetchedBytes.find(hashes[index]) == prefetchedBytes.end())
                                candidates.push_back(index);
                    }
                    std::vector<std::string> lookupHashes;
                    lookupHashes.reserve(candidates.size());
                    for (size_t index : candidates) lookupHashes.push_back(hashes[index]);
                    std::vector<uint64_t> offsets;
                    std::vector<uint8_t> found;
                    impl->ObjectOffsets(lookupHashes, &offsets, &found, catalog);
                    for (size_t slot = 0; slot < candidates.size(); ++slot) {
                        if (!found[slot]) continue;   // 交给既有路径按原样报错
                        Want want;
                        want.hash = hashes[candidates[slot]];
                        want.offset = offsets[slot];
                        wants.push_back(std::move(want));
                    }
                    if (wants.empty()) return;
                    std::map<uint64_t, std::vector<size_t>> byOffset;
                    for (size_t index = 0; index < wants.size(); ++index) byOffset[wants[index].offset].push_back(index);
                    std::vector<std::pair<uint64_t, std::vector<size_t>>> units(byOffset.begin(), byOffset.end());
                    // 两阶段（2026-10-18，B21）：A 并行读记录并校验记录摘要/解析对象表；
                    // B 对象级并行取字节+逐对象摘要。原实现把一个记录里的 8 个对象串在
                    // 同一个 worker 上，长记录时对象摘要无法并行，读取只跑出约 1.4 GB/s，
                    // 而聚合 SHA-256 实测可达 3.3 GB/s。记录字节只保留一份，输出语义不变。
                    const size_t workers = std::min(V4LoadThreads(), std::max<size_t>(units.size(), 1));
                    struct UnitState {
                        bool ok = false;
                        std::string payload;
                        std::vector<ObjectEntry> entries;
                        uint64_t regionStart = 0;
                        uint64_t regionBytes = 0;
                    };
                    std::vector<UnitState> unitsState(units.size());
                    const auto phaseA0 = std::chrono::steady_clock::now();
                    {
                        std::atomic<size_t> readCursor{0};
                        std::vector<std::thread> readers;
                        readers.reserve(workers);
                        for (size_t worker = 0; worker < workers; ++worker) {
                            readers.emplace_back([&]() {
                                for (;;) {
                                    const size_t unitIndex = readCursor.fetch_add(1, std::memory_order_relaxed);
                                    if (unitIndex >= units.size()) break;
                                    UnitState &slot = unitsState[unitIndex];
                                    std::string attemptError;
                                    if (!impl->ReadRecordForLoad(units[unitIndex].first, slot.payload, attemptError)) continue;
                                    if (!DecodeObjectBatchPayloadView(slot.payload.data(), slot.payload.size(),
                                                                      slot.entries, slot.regionStart,
                                                                      slot.regionBytes, attemptError)) {
                                        slot.payload.clear();
                                        slot.entries.clear();
                                        continue;
                                    }
                                    slot.ok = true;
                                }
                            });
                        }
                        for (std::thread &thread : readers) if (thread.joinable()) thread.join();
                    }
                    const auto phaseA1 = std::chrono::steady_clock::now();
                    std::vector<std::map<std::string, std::string>> local(workers);
                    {
                        struct Task { size_t unit; size_t index; };
                        std::vector<Task> tasks;
                        tasks.reserve(wants.size());
                        for (size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
                            if (!unitsState[unitIndex].ok) continue;
                            for (size_t index : units[unitIndex].second) tasks.push_back({unitIndex, index});
                        }
                        std::atomic<size_t> objectCursor{0};
                        std::vector<std::thread> extractors;
                        extractors.reserve(workers);
                        for (size_t worker = 0; worker < workers; ++worker) {
                            extractors.emplace_back([&, worker]() {
                                for (;;) {
                                    const size_t taskIndex = objectCursor.fetch_add(1, std::memory_order_relaxed);
                                    if (taskIndex >= tasks.size()) break;
                                    const Task &task = tasks[taskIndex];
                                    const UnitState &slot = unitsState[task.unit];
                                    std::string body, attemptError;
                                    if (!impl->ExtractObjectFromView(slot.payload, slot.entries, slot.regionStart,
                                                                     slot.regionBytes, wants[task.index].hash, body,
                                                                     attemptError)) {
                                        continue;
                                    }
                                    local[worker].emplace(wants[task.index].hash, std::move(body));
                                }
                            });
                        }
                        for (std::thread &thread : extractors) if (thread.joinable()) thread.join();
                    }
                    if (phaseOn) {
                        phaseAMs += std::chrono::duration<double, std::milli>(
                            phaseA1 - phaseA0).count();
                        phaseBMs += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - phaseA1).count();
                    }
                    for (auto &map : local) {
                        for (auto &entry : map) out.emplace(entry.first, std::move(entry.second));
                    }
                };
                std::vector<std::string> blockHashes;
                blockHashes.reserve(end - begin);
                for (size_t index = begin; index < end; ++index) blockHashes.push_back(blocks[index].objectHash);
                std::map<std::string, std::string> wave;
                fetchGrouped(blockHashes, wave);
                std::vector<std::string> chunkHashes;
                if (extraHashes != nullptr) chunkHashes.insert(chunkHashes.end(),
                                                               extraHashes->begin(), extraHashes->end());
                const size_t parseCount = end - begin;
                if (parseCount > 0 && !wave.empty()) {
                    const size_t parsers = std::min<size_t>(V4LoadThreads(), parseCount);
                    std::atomic<size_t> parseCursor{0};
                    std::vector<std::thread> parseThreads;
                    parseThreads.reserve(parsers);
                    for (size_t worker = 0; worker < parsers; ++worker) {
                        parseThreads.emplace_back([&]() {
                            for (;;) {
                                const size_t offset = parseCursor.fetch_add(1, std::memory_order_relaxed);
                                if (offset >= parseCount) break;
                                const size_t index = begin + offset;
                                auto blockBytes = wave.find(blocks[index].objectHash);
                                if (blockBytes == wave.end()) continue;
                                std::string parseError;
                                Json blockObject = Json::parse(blockBytes->second, parseError);
                                if (!parseError.empty()) continue;
                                BlockRow parsed;
                                std::string blockError;
                                if (!ParseBlockRow(blockObject, parsed, blockError)) continue;
                                Refs refs;
                                CollectRefs(blockObject["kv"], refs);
                                blockPlans[index].refs = std::move(refs);
                                if (assembledState != nullptr)
                                    blockPlans[index].kv = blockObject["kv"];
                                blockPlans[index].ready = true;
                            }
                        });
                    }
                    for (std::thread &thread : parseThreads) if (thread.joinable()) thread.join();
                }
                for (size_t index = begin; index < end; ++index) {
                    if (!blockPlans[index].ready) continue;
                    for (const auto &ref : blockPlans[index].refs) chunkHashes.push_back(ref.first);
                }
                {
                    std::lock_guard<std::mutex> lock(prefetchedMutex);
                    for (auto &entry : wave) prefetchedBytes.emplace(entry.first, std::move(entry.second));
                }
                wave.clear();
                fetchGrouped(chunkHashes, wave);
                std::lock_guard<std::mutex> lock(prefetchedMutex);
                for (auto &entry : wave) prefetchedBytes.emplace(entry.first, std::move(entry.second));
                if (phaseOn) {
                    std::fprintf(stderr,
                                 "[Prefix SSD][prefetch-phase] blocks=%zu A_ms=%.1f B_ms=%.1f\n",
                                 end - begin, phaseAMs, phaseBMs);
                    std::fflush(stderr);
                }
            } catch (...) {
                // 预取只是加速：任何异常都不改变 Load 的语义与错误码。
            }
        };
        // 首窗与 state refs 合并预取（2026-10-18，B21）：state refs 原先在主线程单独
        // 派发，T0 实测 setup 112 ms；现在把它们并入首窗的记录分组预取，一次并行读完。
        // 追加顺序不变：state refs 仍先于块对象进入 objectBytes。
        {
            std::vector<std::string> stateHashes;
            stateHashes.reserve(stateRefs.size());
            for (const auto &ref : stateRefs) stateHashes.push_back(ref.first);
            const size_t firstEnd = std::min(blocks.size(), prefetchWindowBlocks);
            prefetchRecords(0, firstEnd, stateHashes.empty() ? nullptr : &stateHashes);
            if (!stateRefs.empty() &&
                !dispatchObjectReads(stateRefs, "v4_block_chunk_missing")) return false;
        }
        // 预取流水线（2026-10-17，B21）：消费窗口 N 的同时预取窗口 N+1。
        // 原来"预取整窗 -> 消费整窗"串行，读取 6.69 s 里约一半是预取空等。
        // 预取线程只写 prefetchedBytes（互斥保护），不读 fetched/progress。
        std::thread prefetchThread;
        size_t windowStart = std::min(blocks.size(), prefetchWindowBlocks);
        auto schedulePrefetch = [&](size_t begin) {
            if (begin >= blocks.size()) return;
            const size_t end = std::min(blocks.size(), begin + prefetchWindowBlocks);
            prefetchThread = std::thread([&, begin, end]() { prefetchRecords(begin, end); });
        };
        struct PrefetchJoinGuard {
            std::thread &thread;
            ~PrefetchJoinGuard() { if (thread.joinable()) thread.join(); }
        } prefetchJoin{prefetchThread};
        const auto loadWallBeforeLoop = std::chrono::steady_clock::now();
        schedulePrefetch(windowStart);
        const bool loadPhaseTiming = std::getenv("FASTLLM_LOAD_PHASE") != nullptr;
        double phasePushMs = 0, phaseParseMs = 0, phaseDispatchMs = 0, joinWaitMs = 0;
        Json::array assembledKv;
        size_t reusedBlockJson = 0, parsedBlockJson = 0;
        auto phaseMark = [] { return std::chrono::steady_clock::now(); };
        auto phaseMs = [](std::chrono::steady_clock::time_point from,
                          std::chrono::steady_clock::time_point to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
            if (blockIndex == windowStart) {
                if (prefetchThread.joinable()) {
                    const auto join0 = std::chrono::steady_clock::now();
                    prefetchThread.join();   // 本窗数据就绪
                    joinWaitMs += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - join0).count();
                }
                schedulePrefetch(blockIndex + prefetchWindowBlocks);    // 与消费重叠
                windowStart = blockIndex + prefetchWindowBlocks;
            }
            const auto &block = blocks[blockIndex];
            std::string blockText;
            const auto phasePush0 = phaseMark();
            if (!pushObject(block.objectHash, blockText, "v4_block_missing")) return false;
            phasePushMs += phaseMs(phasePush0, phaseMark());
            if (blockText.empty()) continue;   // 已取回（同一对象被多处引用）
            Refs refs;
            Json blockKv;
            if (blockIndex < blockPlans.size() && blockPlans[blockIndex].ready) {
                // 预取线程已并行完成 Json::parse + ParseBlockRow + CollectRefs。
                refs = std::move(blockPlans[blockIndex].refs);
                if (assembledState != nullptr) {
                    blockKv = std::move(blockPlans[blockIndex].kv);
                    ++reusedBlockJson;
                }
            } else {
                std::string blockError;
                Json blockObject;
                const auto phaseParse0 = phaseMark();
                { V4Timer parseTimer(gV4LoadProfile.parseNs, profile);
                  gV4LoadProfile.parseCount++;
                  blockObject = Json::parse(blockText, blockError); }
                phaseParseMs += phaseMs(phaseParse0, phaseMark());
                BlockRow parsed;
                if (!blockError.empty() || !ParseBlockRow(blockObject, parsed, blockError)) {
                    error = blockError.empty() ? "v4_block_object" : blockError;
                    return false;
                }
                CollectRefs(blockObject["kv"], refs);
                if (assembledState != nullptr && !manifestOnly) blockKv = blockObject["kv"];
                ++parsedBlockJson;
            }
            if (assembledState != nullptr && !manifestOnly) {
                // Both producers above already ran ParseBlockRow successfully.
                // Json is immutable: keep its verified KV value, not a second
                // serialization/hash or the consumed block/Refs wrappers.
                if (!blockKv.is_array()) {
                    error = "v4_block_object";
                    return false;
                }
                for (const auto &bundle : blockKv.array_items())
                    assembledKv.push_back(bundle);
            }
            // B1：清单-only 模式**跳过载荷取回**。
            // ②块对象里 kv 数组引用的 KV 分片：这才是 566.2 MiB 的大头（463/520 个对象）。
            //   清单（bundle 的 bytes/segments/chunks 元数据）已经在块对象里拿到了；
            //   adopt 只用元数据。清单-only 下只做**存在性**校验，保持"父分片可读"的
            //   fail-closed 证明发生在 adopt 这一步（不推迟到提交）。
            {
                const auto phaseDispatch0 = phaseMark();
                const bool dispatched = dispatchObjectReads(refs, "v4_block_chunk_missing");
                phaseDispatchMs += phaseMs(phaseDispatch0, phaseMark());
                if (!dispatched) return false;
            }
        }
        if (loadPhaseTiming) {
            std::fprintf(stderr, "[Prefix SSD][load-phase] blocks=%zu push_ms=%.1f parse_ms=%.1f dispatch_ms=%.1f\n",
                         blocks.size(), phasePushMs, phaseParseMs, phaseDispatchMs);
            const auto loadWallEnd = std::chrono::steady_clock::now();
            std::fprintf(stderr,
                         "[Prefix SSD][load-wall] drain_ms=%.1f setup_ms=%.1f root_ms=%.1f state_ms=%.1f chain_ms=%.1f join_wait_ms=%.1f loop_ms=%.1f total_ms=%.1f\n",
                         std::chrono::duration<double, std::milli>(loadWallDrained - loadWallStart).count(),
                         std::chrono::duration<double, std::milli>(loadWallBeforeLoop - loadWallDrained).count(),
                         std::chrono::duration<double, std::milli>(setupState0 - loadWallDrained).count(),
                         std::chrono::duration<double, std::milli>(setupState1 - setupState0).count(),
                         std::chrono::duration<double, std::milli>(setupChain1 - setupState1).count(),
                         joinWaitMs,
                         std::chrono::duration<double, std::milli>(loadWallEnd - loadWallBeforeLoop).count(),
                         std::chrono::duration<double, std::milli>(loadWallEnd - loadWallStart).count());
            std::fflush(stderr);
        }
        if (profile) {
            const V4LoadProfile &p = gV4LoadProfile;
            std::fprintf(stderr,"[Prefix SSD][load-profile] path=chain objects=%zu object_mb=%.1f "
                "read_ms=%.1f catalog_ms=%.1f(%zu) parse_ms=%.1f(%zu) chain_ms=%.1f.\n",
                p.objectCount, double(p.objectBytes)/1048576.0, double(p.objectNs)/1e6,
                double(p.catalogNs)/1e6, p.catalogCount, double(p.parseNs)/1e6, p.parseCount,
                double(p.chainNs)/1e6);
            std::fflush(stderr);
        }
        FastllmEvent("cache", 0, "cache_load_phase",
                     "blocks=%zu push_ms=%.1f parse_ms=%.1f dispatch_ms=%.1f join_ms=%.1f "
                     "loop_ms=%.1f prefetch_window=%zu",
                     blocks.size(), phasePushMs, phaseParseMs, phaseDispatchMs, joinWaitMs,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - loadWallBeforeLoop).count(),
                     prefetchWindowBlocks);
        emitReadSplit(1);
        if (assembledState != nullptr) {
            auto fields = state.object_items();
            // Manifest readers assemble JSON after record/prefetch temporaries
            // are released. The same-chain state carries validation, not payload.
            if (manifestOnly) fields.erase("kv");
            else fields["kv"] = Json(std::move(assembledKv));
            *assembledState = Json(std::move(fields));
            FastllmEvent("cache", 0, "cache_read_json_profile",
                         "reused_blocks=%zu parsed_blocks=%zu",reusedBlockJson,parsedBlockJson);
        }
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

bool DiskPrefixV4Store::ChainBlocks(const std::string &chainTip, uint64_t count,
                                    std::vector<BlockRow> &out, std::string &error,bool publishedOnly) const {
    if (!publishedOnly) impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    error.clear();
    out.clear();
    if (!prefix_v4::IsDigest(chainTip) || count == 0 || count > prefix_v4::kMaxBlockCount) {
        error = "v4_block_argument";
        return false;
    }
    try {
        std::unique_ptr<Database> published;
        sqlite3 *catalog = impl->database->value;
        if (publishedOnly) {
            published = std::make_unique<Database>(impl->dbPath,true);
            catalog = published->value;
        }
        std::vector<BlockRow> reversed;
        reversed.reserve(size_t(count));
        std::string cursor = chainTip;
        for (uint64_t step = 0; step < count; ++step) {
            Statement query(catalog,
                            "SELECT chain_key,parent,object_hash,identity,key,block_index,"
                            "token_start,token_end,bytes FROM blocks WHERE chain_key=?");
            query.Text(1, cursor);
            if (!query.Row()) {
                // 链尾本身缺失 = v4_block_missing；走到一半需要父块而父块缺失 =
                // v4_block_parent_missing（fail-closed，绝不拼出半条链）。
                error = (step == 0) ? "v4_block_missing" : "v4_block_parent_missing";
                return false;
            }
            BlockRow row;
            row.chainKey = query.Text(0);
            row.parent = query.Text(1);
            row.objectHash = query.Text(2);
            row.identity = query.Text(3);
            row.key = query.Text(4);
            row.index = query.Number(5);
            row.tokenStart = query.Number(6);
            row.tokenEnd = query.Number(7);
            row.bytes = query.Number(8);
            if (row.chainKey != cursor || !prefix_v4::IsDigest(row.parent) ||
                !prefix_v4::IsDigest(row.objectHash)) {
                error = "v4_block_object";
                return false;
            }
            reversed.push_back(std::move(row));
            cursor = reversed.back().parent;
        }
        std::reverse(reversed.begin(), reversed.end());
        for (size_t i = 1; i < reversed.size(); ++i) {
            if (reversed[i].index != reversed[i - 1].index + 1 ||
                reversed[i].tokenStart != reversed[i - 1].tokenEnd ||
                reversed[i].parent != reversed[i - 1].chainKey) {
                error = "v4_block_chain_mismatch";
                return false;
            }
        }
        // 链必须从 token 0 的创世块开始：缺父块时明确报 v4_block_parent_missing。
        if (reversed.front().index != 0 || reversed.front().tokenStart != 0) {
            error = "v4_block_parent_missing";
            return false;
        }
        if (reversed.front().parent != prefix_v4::BlockGenesisKey()) {
            error = "v4_block_chain_mismatch";
            return false;
        }
        out = std::move(reversed);
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

bool DiskPrefixV4Store::Descriptor(const std::string &identity, const std::string &key,
                                   std::string &metadata, std::string &error, bool publishedOnly) const {
    std::unique_ptr<Database> published;
    if (publishedOnly) published = std::make_unique<Database>(impl->dbPath,true);
    sqlite3 *catalog = published ? published->value : impl->database->value;

    if (!publishedOnly) impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    error.clear();
    metadata.clear();
    try {
        // S2-4：按 (identity,key) 点查描述符（JOIN roots 保证是 live root），替代
        // ReadSession::Load 里"遍历全部 live 检查点、逐条 Json::parse"的全表路径。
        // 复合主键 (identity,key) 让这是一次索引点查。
        Statement query(catalog,
                        "SELECT c.metadata FROM checkpoints c "
                        "JOIN roots r ON r.identity=c.identity AND r.key=c.key "
                        "WHERE c.identity=? AND c.key=? AND r.live=1");
        query.Text(1, identity);
        query.Text(2, key);
        if (!query.Row()) { error = "v4_root_missing"; return false; }
        metadata = query.Text(0);
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

std::vector<int> DiskPrefixV4Store::CheckpointLengths(const std::string &family,
                                                      int maximum,
                                                      const std::string &kind, bool publishedOnly) const {
    std::unique_ptr<Database> published;
    if (publishedOnly) published = std::make_unique<Database>(impl->dbPath,true);
    sqlite3 *catalog = published ? published->value : impl->database->value;

    if (!publishedOnly) impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    // S2-4：v4 此前走 ListCheckpoints 的"全表遍历 + 逐条 Json::parse"（r7 实测每次
    // lookup ≈ 23 个长度 × 186 行 ≈ 4300 次 parse ≈ 5–10 ms/请求）。改为与 v3 同款的
    // SQL 过滤：DISTINCT 直接用投影列 length，kind/family 走 checkpoints_lookup 索引，
    // JOIN roots 限定 live root（等价于旧 Checkpoints() 的语义）。
    Statement query(catalog,
                    "SELECT DISTINCT c.length FROM checkpoints c "
                    "JOIN roots r ON r.identity=c.identity AND r.key=c.key "
                    "WHERE r.live=1 AND c.kind=? AND c.family=? AND c.length<=? "
                    "ORDER BY c.length DESC");
    query.Text(1, kind);
    query.Text(2, family);
    query.Number(3, uint64_t(maximum));
    std::vector<int> result;
    while (query.Row()) result.push_back(int(query.Number(0)));
    return result;
}

std::vector<DiskPrefixV4Store::CheckpointRow>
DiskPrefixV4Store::ListCheckpoints(const std::string &exact, const std::string &family,
                                   int maximum, const std::string &kind,
                                   const std::string &prefix) const {
    impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    // S2-4：照 v3 的模板改成 SQL（可选 identity/family/prefix 过滤 + 同样的排序），
    // 不再把整张 checkpoints 表读进来逐条 Json::parse。排序与旧实现等价：
    // length 降序、created 升序。
    std::string sql = "SELECT c.key,c.identity,c.metadata,c.sequence,c.family,c.kind,"
                      "c.length,c.dtype,c.provenance,c.created,c.prefix "
                      "FROM checkpoints c "
                      "JOIN roots r ON r.identity=c.identity AND r.key=c.key "
                      "WHERE r.live=1 AND c.kind=? AND c.length<=?";
    if (!exact.empty()) sql += " AND c.identity=?";
    if (!family.empty()) sql += " AND c.family=?";
    if (!prefix.empty()) sql += " AND c.prefix=?";
    sql += " ORDER BY c.length DESC,c.created ASC";
    Statement query(impl->database->value, sql.c_str());
    query.Text(1, kind);
    query.Number(2, uint64_t(maximum));
    int parameter = 3;
    if (!exact.empty()) query.Text(parameter++, exact);
    if (!family.empty()) query.Text(parameter++, family);
    if (!prefix.empty()) query.Text(parameter++, prefix);
    std::vector<CheckpointRow> result;
    while (query.Row()) {
        CheckpointRow row;
        row.key = query.Text(0);
        row.identity = query.Text(1);
        row.metadata = query.Text(2);
        row.sequence = query.Number(3);
        row.familyIdentity = query.Text(4);
        row.kind = query.Text(5);
        row.length = int(query.Number(6));
        row.kvDtype = query.Text(7);
        row.provenance = query.Text(8);
        row.createdNs = query.Number(9);
        row.prefix = query.Text(10);
        result.push_back(std::move(row));
    }
    return result;
}

std::vector<DiskPrefixV4Store::CheckpointRow> DiskPrefixV4Store::MatchPrefixes(
        const std::string &familyIdentity,
        const std::vector<std::pair<int, std::string>> &prefixes,
        const std::string &kind, bool publishedOnly) const {
    std::unique_ptr<Database> published;
    if (publishedOnly) published = std::make_unique<Database>(impl->dbPath,true);
    sqlite3 *catalog = published ? published->value : impl->database->value;

    // 【业界式「L3 实时问一次」】对标 SGLang HiCache 官方原文
    // "it queries the backend in real time"。本函数把"逐长度各发一条 SQL"（1+N≈24 条）
    // 收成**一条 SQL**。
    //
    // 为什么不用行值 `(prefix,length) IN (VALUES (?,?),…)`：那个语法要求 SQLite ≥ 3.15，
    // 而本仓用的是系统 sqlite（实测 CLI 3.53.2，但**未在链接产物上验证过**，见 S3 的不确定项④）。
    // 这里改用**必然可用**的写法：`c.prefix IN (?,?,…)` 仍然命中
    // `checkpoints_prefix(kind,family,prefix,length)` 索引的 (kind,family,prefix) 前缀，
    // 然后在 C++ 里按**精确 (prefix,length) 对**过滤 ⇒ 与旧的"逐长度 ListCheckpoints"
    // **逐条等价**（旧的 maximum 参数是 length<=ℓ，但传进来的 prefix 就是 ℓ 的摘要，
    // 不同长度撞同一摘要不可能，故等价于精确匹配）。
    std::vector<CheckpointRow> result;
    if (prefixes.empty()) return result;
    if (!publishedOnly) impl->DrainForRead();   // 与 ListCheckpoints/CheckpointLengths 同口径：必须先看到已受理的提交
    std::string sql = "SELECT c.key,c.identity,c.metadata,c.sequence,c.family,c.kind,"
                      "c.length,c.dtype,c.provenance,c.created,c.prefix "
                      "FROM checkpoints c "
                      "JOIN roots r ON r.identity=c.identity AND r.key=c.key "
                      "WHERE r.live=1 AND c.kind=? AND c.family=? AND c.prefix IN (";
    for (size_t i = 0; i < prefixes.size(); ++i) sql += (i == 0 ? "?" : ",?");
    sql += ") ORDER BY c.length DESC,c.created ASC";
    std::set<std::pair<int, std::string>> wanted;
    for (const auto &entry : prefixes) wanted.insert(entry);
    Statement query(catalog, sql.c_str());
    int parameter = 1;
    query.Text(parameter++, kind);
    query.Text(parameter++, familyIdentity);
    for (const auto &entry : prefixes) query.Text(parameter++, entry.second);
    while (query.Row()) {
        CheckpointRow row;
        row.key = query.Text(0);
        row.identity = query.Text(1);
        row.metadata = query.Text(2);
        row.sequence = query.Number(3);
        row.familyIdentity = query.Text(4);
        row.kind = query.Text(5);
        row.length = int(query.Number(6));
        row.kvDtype = query.Text(7);
        row.provenance = query.Text(8);
        row.createdNs = query.Number(9);
        row.prefix = query.Text(10);
        if (wanted.find({row.length, row.prefix}) == wanted.end()) continue;   // 精确对过滤
        result.push_back(std::move(row));
    }
    return result;
}

uint64_t DiskPrefixV4Store::PhysicalBytes() const {
    // S2-5a：真实物理占用（control.log 字节数），不再用 live 冒充 physical。
    return impl->PhysicalBytes();
}

bool DiskPrefixV4Store::ObservePhysicalBytes(uint64_t &bytes, std::string &error) const {
    if (impl->segmented) { bytes = impl->SegmentPhysicalBytes(); error.clear(); return true; }
    std::error_code ec;
    const auto size = fs::file_size(impl->logPath, ec);
    if (ec) {
        error = "control_log_stat_" + std::to_string(ec.value());
        return false;
    }
    bytes = uint64_t(size); error.clear(); return true;
}

bool DiskPrefixV4Store::RebuildIndex(std::string &error) {
    return impl->RunActor([&] { return RebuildIndexLocked(error); }, error);
}

bool DiskPrefixV4Store::RebuildIndexLocked(std::string &error) {
    error.clear();
    Impl::RootLocks locks;
    // 维护路径用非阻塞独占：拿不到就报 cache_busy，绝不持 append 等 reader。
    if (!impl->AcquireLocks(locks, true, false, error)) return false;
    return RebuildIndexBareLocked(error);
}

bool DiskPrefixV4Store::RebuildIndexBareLocked(std::string &error) {
    error.clear();
    V4InvalidateRecordCache();   // [P1-a] 重建后日志/投影可能变化 ⇒ 记录缓存必须失效
    // §10.4 第 5 条 / §17.1：重建必须**先在新临时 catalog 里重建全部投影**，
    // 关闭并同步后再原子替换；失败时原 catalog 保持可用，绝不就地 DELETE+INSERT。
    const fs::path temporary = fs::path(impl->dbPath.string() + ".rebuild");
    try {
        {
            std::error_code ignored;
            fs::remove(temporary, ignored);
            fs::remove(fs::path(temporary.string() + "-wal"), ignored);
            fs::remove(fs::path(temporary.string() + "-shm"), ignored);
        }
        Database rebuilt(temporary);
        Impl::CreateSchema(rebuilt);
        rebuilt.Exec("BEGIN IMMEDIATE");
        // 从"临时库 + 控制日志"读取对象（重建期间活动库还没有这些行）。
        auto ReadObjectFromRebuilt = [&](const std::string &hash, std::string &out,
                                         std::string &readError) {
            Statement location(rebuilt.value, "SELECT offset FROM objects WHERE hash=?");
            if (location.value == nullptr) { readError = "v4_object_missing"; return false; }
            location.Text(1, hash);
            if (!location.Row()) { readError = "v4_object_missing"; return false; }
            prefix_v4::RecordHeader header;
            std::string payload;
            fs::path recordPath;
            uint64_t recordOffset = 0;
            if (!impl->LogLocation(uint64_t(location.Number(0)), recordPath, recordOffset, readError))
                return false;
            if (!V4ReadRecord(recordPath, recordOffset, header, payload, readError)) {
                return false;
            }
            std::vector<ObjectEntry> entries;
            std::string region;
            if (!DecodeObjectBatchPayload(payload.data(), payload.size(), entries, region,
                                          readError)) {
                return false;
            }
            for (const auto &entry : entries) {
                if (entry.hash != hash) continue;
                out = region.substr(size_t(entry.offset), size_t(entry.bytes));
                return true;
            }
            readError = "v4_object_mismatch";
            return false;
        };
        // Route B：从紧凑状态对象的 chain_tip 反向走链，收集此 root 的全部可达对象哈希。
        // 每一步都从**临时库**的 blocks 投影取块对象，并复算链键/内容摘要（ParseBlockRow）；
        // 任一环缺失或内容不自洽即返回 false（该 root 不可发布）。
        auto chainReachable = [&](const std::string &rootHash, std::set<std::string> &out) -> bool {
            out.clear();
            std::string stateText;
            if (!ReadObjectFromRebuilt(rootHash, stateText, error)) return false;
            std::string parseError; const Json state = Json::parse(stateText, parseError);
            if (!parseError.empty() || !state.is_object() ||
                !state["chain_tip"].is_string() || !state["block_count"].is_number()) {
                return false;
            }
            out.insert(rootHash);
            Refs stateRefs; CollectRefs(state, stateRefs);
            for (const auto &ref : stateRefs) out.insert(ref.first);
            std::string cursor = state["chain_tip"].string_value();
            const uint64_t count = uint64_t(state["block_count"].number_value());
            if (count == 0 || count > prefix_v4::kMaxBlockCount) return false;
            for (uint64_t step = 0; step < count; ++step) {
                Statement query(rebuilt.value,
                                "SELECT object_hash,parent FROM blocks WHERE chain_key=?");
                query.Text(1, cursor);
                if (!query.Row()) return false;
                const std::string objectHash = query.Text(0);
                const std::string parent = query.Text(1);
                std::string blockText;
                if (!ReadObjectFromRebuilt(objectHash, blockText, error)) return false;
                std::string blockError; const Json block = Json::parse(blockText, blockError);
                DiskPrefixV4Store::BlockRow parsed;
                std::string rowError;
                if (!blockError.empty() || !ParseBlockRow(block, parsed, rowError)) return false;
                out.insert(objectHash);
                Refs blockRefs; CollectRefs(block["kv"], blockRefs);
                for (const auto &ref : blockRefs) out.insert(ref.first);
                cursor = parent;
            }
            return true;
        };
        prefix_v4::ControlLogReader logReader;
        std::vector<ScannedRecord> records;
        // DESIGN-v3-segments C6: scan segments in base order (Kafka LogLoader); record.offset
        // becomes global; sequences must keep increasing across files; a torn tail is only
        // acceptable in the last segment.
        std::vector<std::unique_ptr<prefix_v4::ControlLogReader>> segmentReaders;
        std::vector<uint64_t> segmentBases;
        LogClassification classification = LogClassification::Complete;
        if (!impl->segmented) {
            if (!logReader.Open(impl->logPath.string(), error)) return false;
            classification = logReader.Scan(records, error);
        } else {
            if (!impl->ListSegments(error)) return false;
            std::vector<std::pair<uint64_t, fs::path>> files;
            {
                std::lock_guard<std::mutex> lock(impl->segmentsMutex);
                for (const auto &entry : impl->segments) files.emplace_back(entry.first, entry.second.path);
            }
            uint64_t previousSequence = 0;
            bool havePrevious = false;
            for (size_t index = 0; index < files.size() &&
                                   classification != LogClassification::CorruptControl; ++index) {
                auto reader = std::make_unique<prefix_v4::ControlLogReader>();
                if (!reader->Open(files[index].second.string(), error)) return false;
                std::vector<ScannedRecord> part;
                const LogClassification partClass = reader->Scan(part, error);
                if (partClass == LogClassification::CorruptControl) {
                    classification = partClass;
                    break;
                }
                if (partClass == LogClassification::UncommittedTail) {
                    if (index + 1 != files.size()) {
                        error = "v4_segment_inner_tail";
                        classification = LogClassification::CorruptControl;
                        break;
                    }
                    classification = partClass;
                }
                for (auto &record : part) {
                    if (havePrevious && record.header.sequence <= previousSequence) {
                        error = "v4_control_sequence";
                        classification = LogClassification::CorruptControl;
                        break;
                    }
                    previousSequence = record.header.sequence;
                    havePrevious = true;
                    record.offset += files[index].first;
                    records.push_back(std::move(record));
                }
                segmentBases.push_back(files[index].first);
                segmentReaders.push_back(std::move(reader));
            }
        }
        auto ReadScannedPayload = [&](const ScannedRecord &record, std::string &out,
                                      std::string &readError) -> bool {
            if (!impl->segmented) return logReader.ReadPayload(record, out, readError);
            auto it = std::upper_bound(segmentBases.begin(), segmentBases.end(), record.offset);
            if (it == segmentBases.begin()) { readError = "v4_segment_range"; return false; }
            const size_t index = size_t(it - segmentBases.begin()) - 1;
            ScannedRecord local = record;
            local.offset -= segmentBases[index];
            return segmentReaders[index]->ReadPayload(local, out, readError);
        };
        // A corrupt control log must never be "repaired" by publishing a subset.
        if (classification == LogClassification::CorruptControl) {
            ++impl->rebuildCorrupt;
            return false;
        }
        impl->appendBlocked = classification == LogClassification::UncommittedTail;
        if (impl->appendBlocked) {
            ++impl->rebuildTail;
            // S1-1 修复（2026-10-02，对标审计 P0-3）：撕裂尾不再"永久只读"。
            // 业界（LevelDB log_format / SQLite WAL / RocksDB 恢复）都是"**截断残尾后继续写**"；
            // 我们此前只置 append_blocked，而全仓 8 处引用中没有任何路径能把它清回 false
            // （Commit 与 Compact 双双永久拒绝）⇒ 一次掉电后该 root 永久只读、SSD 缓存报废。
            // 现在：ScanControlLog 已保证残尾不含任何已 COMMIT 的根 ⇒ 把 control.log 截断到
            // 最后一个完整记录的末尾，成功后清掉 append_blocked，使该目录**重开即自愈**。
            // 保守原则：只有确实截断成功才解除；截断失败或无可截断则维持阻塞（fail-closed）。
            const bool segmentTail = impl->segmented && !segmentReaders.empty();
            const uint64_t tailBase = segmentTail ? segmentBases.back() : 0;
            const fs::path tailPath = segmentTail ? impl->SegmentPath(tailBase) : impl->logPath;
            const uint64_t tailSize = segmentTail ? segmentReaders.back()->Size() : logReader.Size();
            uint64_t goodEnd = 0;
            for (const auto &record : records) {
                if (record.offset < tailBase) continue;
                const uint64_t end = record.offset - tailBase + record.header.physicalBytes;
                if (end > goodEnd) goodEnd = end;
            }
            if (goodEnd < tailSize) {
                std::error_code trimError;
                std::filesystem::resize_file(tailPath, goodEnd, trimError);
                if (!trimError && segmentTail) {
                    std::string relistError;
                    if (!impl->ListSegments(relistError)) trimError = std::make_error_code(std::errc::io_error);
                }
                if (!trimError) {
                    impl->appendBlocked = false;
                    std::fprintf(stderr, "[Prefix SSD] control log torn tail truncated: %llu -> %llu bytes (append unblocked).\n",
                                 (unsigned long long)tailSize, (unsigned long long)goodEnd);
                } else {
                    std::fprintf(stderr, "[Prefix SSD] control log truncate failed: %s (append stays blocked).\n",
                                 trimError.message().c_str());
                }
            }
        }
        else ++impl->rebuildValid;
        std::map<uint64_t, PrepareRecord> pending;
        std::map<uint64_t, prefix_v4::MetadataRecord> pendingDescriptors;
        std::set<std::string> badObjects;
        uint64_t excludedRoots = 0;
        uint64_t maximum = 0;
        for (const auto &record : records) {
            std::string payloadText;
            if (!ReadScannedPayload(record, payloadText, error)) return false;
            maximum = std::max(maximum, record.header.sequence);
            if (record.header.type == kRecordObjectBatch) {
                std::vector<ObjectEntry> entries;
                std::string region;
                if (!DecodeObjectBatchPayload(payloadText.data(), payloadText.size(), entries,
                                              region, error)) {
                    error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                    return false;
                }
                for (const auto &entry : entries) {
                    // §10.4 第 2 条：重建时**必须逐对象重算 SHA256**，不受线上 strict
                    // 开关控制。对象表摘要正确但 payload 坏掉时，该对象不得入库，
                    // 引用它的检查点也不得发布。
                    const std::string body = region.substr(size_t(entry.offset), size_t(entry.bytes));
                    if (DiskPrefixCache::Digest(body) != entry.hash) {
                        badObjects.insert(entry.hash);
                        continue;
                    }
                    Statement insert(rebuilt.value,
                                     "INSERT OR REPLACE INTO objects(hash,kind,bytes,offset) "
                                     "VALUES(?,?,?,?)");
                    insert.Text(1, entry.hash); insert.Number(2, entry.kind);
                    insert.Number(3, entry.bytes); insert.Number(4, record.offset);
                    insert.Run();
                    // Route B：块对象的权威内容自带 chain_key/parent/token 区间/内容摘要，
                    // 重建时据此复原 `blocks` 投影；内容不自洽的块一律标记为坏对象，
                    // 引用它的检查点随后不会被发布。
                    if (entry.kind == prefix_v4::kObjectBlock) {
                        std::string blockError; const Json block = Json::parse(body, blockError);
                        DiskPrefixV4Store::BlockRow row;
                        std::string rowError;
                        if (!blockError.empty() || !ParseBlockRow(block, row, rowError)) {
                            badObjects.insert(entry.hash);
                        } else {
                            row.objectHash = entry.hash; row.bytes = entry.bytes;
                            Impl::InsertBlockRow(rebuilt, row);
                        }
                    }
                }
            } else if (record.header.type == kRecordManifestPrepare) {
                PrepareRecord prepare;
                if (!DecodePrepare(payloadText.data(), payloadText.size(), prepare, error)) {
                    error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                    return false;
                }
                pending[record.offset] = prepare;
            } else if (record.header.type == kRecordManifestCommit) {
                prefix_v4::CommitRecord commit;
                if (!DecodeCommit(payloadText.data(), payloadText.size(), commit, error)) {
                    error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                    return false;
                }
                const auto found = pending.find(commit.prepareOffset);
                // A COMMIT without its PREPARE is never published.
                if (found == pending.end()) continue;
                const PrepareRecord &prepare = found->second;
                if (prepare.rootHash != commit.rootHash) {
                    error = "v4_prepare_commit_mismatch";
                    error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                    return false;
                }
                // 发布前检查：root 对象与其清单引用的对象都必须通过逐对象校验，
                // 否则该检查点不可发布（其余完好的检查点不受影响）。
                {
                    bool publishable = badObjects.count(prepare.rootHash) == 0;
                    if (publishable) {
                        // 注意：必须查**临时库**（行尚未换入活动库），不能走
                        // impl->ObjectBytes——否则每次重建都会把 root 误判为不可发布。
                        std::string listText;
                        std::vector<ObjectListEntry> list;
                        std::string listError;
                        if (!ReadObjectFromRebuilt(prepare.rootHash, listText, error)) {
                            publishable = false;
                        }
                        else if (DecodeObjectList(listText.data(), listText.size(), list, listError)) {
                            for (const auto &entry : list)
                                if (badObjects.count(entry.hash)) { publishable = false; break; }
                        }
                        else {
                            // Route B：root 是紧凑状态对象（不是 ObjectList）。沿块链收集全部
                            // 可达对象；缺块/坏块一律不可发布（绝不发"读不回来"的检查点）。
                            std::set<std::string> reachable;
                            if (!chainReachable(prepare.rootHash, reachable)) {
                                publishable = false;
                            } else {
                                for (const auto &hash : reachable)
                                    if (badObjects.count(hash)) { publishable = false; break; }
                            }
                        }
                        error.clear();
                    }
                    if (!publishable) { ++excludedRoots; continue; }
                }
                Statement upsert(rebuilt.value,
                                 "INSERT OR REPLACE INTO roots(identity,key,root,sequence,bytes,live) "
                                 "VALUES(?,?,?,?,?,1)");
                upsert.Text(1, prepare.identity); upsert.Text(2, prepare.key);
                upsert.Text(3, prepare.rootHash); upsert.Number(4, commit.sequence);
                // bytes 先占位 0，扫描结束后由下面的"唯一可达对象"记账统一回填；
                // 绝不能像旧实现那样把 0 留在库里（否则 LiveBytes() 失真、配额失效）。
                upsert.Number(5, 0); upsert.Run();
                const auto descriptor = pendingDescriptors.find(commit.prepareOffset);
                if (descriptor != pendingDescriptors.end()) {
                    const auto &metadata = descriptor->second;
                    const size_t separator = metadata.value.find('\n');
                    // S2-4 写入点 2/4（重建路径：紧随 PREPARE 的描述符）：同样写全索引列，
                    // 否则重建后查询按 kind/family 过滤会漏掉这些行。
                    Impl::InsertCheckpointRow(rebuilt, prepare.identity, prepare.key,
                                              metadata.value.substr(separator + 1),
                                              commit.sequence);
                    pendingDescriptors.erase(descriptor);
                }
                pending.erase(found);
            } else if (record.header.type == kRecordErase) {
                prefix_v4::EraseRecord erase;
                if (!DecodeErase(payloadText.data(), payloadText.size(), erase, error)) return false;
                Statement retire(rebuilt.value,
                                 "UPDATE roots SET live=0 WHERE identity=? AND key=?");
                retire.Text(1, erase.identity); retire.Text(2, erase.key); retire.Run();
            } else if (record.header.type == kRecordMetadata) {
                prefix_v4::MetadataRecord metadata;
                if (!DecodeMetadata(payloadText.data(), payloadText.size(), metadata, error)) {
                    error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                    return false;
                }
                // Two namespaces share the record type: "checkpoint:" carries a
                // checkpoint descriptor, "kv:" carries durable key/value data.
                const std::string checkpointPrefix = "checkpoint:";
                const std::string kvPrefix = "kv:";
                if (metadata.key.compare(0, checkpointPrefix.size(), checkpointPrefix) == 0) {
                    // value = identity '\n' descriptor JSON
                    const size_t separator = metadata.value.find('\n');
                    if (separator == std::string::npos) {
                        error = "v4_checkpoint_metadata";
                        error = "record@" + std::to_string(record.offset) + " type=" + std::to_string(record.header.type) + " seq=" + std::to_string(record.header.sequence) + ": " + error;
                        return false;
                    }
                    bool staged = false;
                    for (auto prepare = pending.rbegin(); prepare != pending.rend(); ++prepare) {
                        if (prepare->second.sequence + 1 == record.header.sequence &&
                            prepare->second.key == metadata.key.substr(checkpointPrefix.size()) &&
                            prepare->second.identity == metadata.value.substr(0, separator)) {
                            pendingDescriptors[prepare->first] = metadata;
                            staged = true;
                            break;
                        }
                    }
                    // New descriptors immediately following PREPARE are not
                    // visible without its COMMIT.  This also preserves an old
                    // descriptor if a replacement attempt fails before COMMIT.
                    if (staged) continue;
                    // S2-4 写入点 3/4（重建路径：非紧随 PREPARE 的描述符，例如被替换/
                    // 较早提交的描述符）：同样写全索引列。
                    Impl::InsertCheckpointRow(rebuilt, metadata.value.substr(0, separator),
                                              metadata.key.substr(checkpointPrefix.size()),
                                              metadata.value.substr(separator + 1),
                                              record.header.sequence);
                } else if (metadata.key.compare(0, kvPrefix.size(), kvPrefix) == 0) {
                    Statement upsert(rebuilt.value,
                                     "INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)");
                    upsert.Text(1, metadata.key.substr(kvPrefix.size()));
                    upsert.Text(2, metadata.value);
                    upsert.Run();
                }
            }
        }
        // 记账修复（2026-10-01）：重建/压缩后 roots.bytes 必须是**真实**的已用字节，
        // 不能是 0（旧实现写死 0）。写入路径按"本次新写的对象"累加该值；重建无法逐条
        // 复原这种归属，于是改成按 sequence 从旧到新、把每个对象只算给第一个引用它的
        // live root——这样 SUM(bytes) 恒等于 live root 可达的唯一对象字节，`LiveBytes()`
        // 才能继续支撑配额/淘汰判断（cache.cpp:3021/3195），否则配额上限形同虚设。
        {
            std::map<std::string, uint64_t> objectBytes;
            {
                Statement all(rebuilt.value, "SELECT hash,bytes FROM objects");
                while (all.Row()) objectBytes[all.Text(0)] = uint64_t(all.Number(1));
            }
            std::set<std::string> claimed;
            std::vector<std::pair<std::string, std::string>> liveRoots;
            std::vector<uint64_t> liveBytes;
            {
                Statement roots(rebuilt.value,
                                "SELECT identity,key,root FROM roots WHERE live=1 "
                                "ORDER BY sequence ASC");
                while (roots.Row()) {
                    const std::string identity = roots.Text(0);
                    const std::string rootKey = roots.Text(1);
                    const std::string rootHash = roots.Text(2);
                    uint64_t bytes = 0;
                    std::string listText, readError;
                    if (ReadObjectFromRebuilt(rootHash, listText, readError)) {
                        std::vector<ObjectListEntry> list;
                        std::string decodeError;
                        if (DecodeObjectList(listText.data(), listText.size(), list, decodeError)) {
                            for (const auto &entry : list) {
                                if (!claimed.insert(entry.hash).second) continue;
                                const auto found = objectBytes.find(entry.hash);
                                if (found != objectBytes.end()) bytes += found->second;
                            }
                        } else {
                            // Route B：按块链收集可达对象集，再把每个对象只算给第一个引用
                            // 它的 live root（口径与旧路径一致，roots.bytes 仍支撑配额/淘汰）。
                            std::set<std::string> reachable;
                            if (chainReachable(rootHash, reachable)) {
                                for (const auto &hash : reachable) {
                                    if (!claimed.insert(hash).second) continue;
                                    const auto found = objectBytes.find(hash);
                                    if (found != objectBytes.end()) bytes += found->second;
                                }
                            }
                        }
                    }
                    liveRoots.emplace_back(identity, rootKey);
                    liveBytes.push_back(bytes);
                }
            }
            for (size_t i = 0; i < liveRoots.size(); ++i) {
                Statement update(rebuilt.value,
                                 "UPDATE roots SET bytes=? WHERE identity=? AND key=?");
                update.Number(1, liveBytes[i]);
                update.Text(2, liveRoots[i].first);
                update.Text(3, liveRoots[i].second);
                update.Run();
            }
        }
        rebuilt.Exec("COMMIT");
        // 关闭临时库前先把 WAL 归并进主文件（TRUNCATE），否则重命名会丢掉 WAL 里的行。
        rebuilt.Exec("PRAGMA wal_checkpoint(TRUNCATE)");
        rebuilt.Close();
        impl->database.reset();
        {
            std::error_code ignored;
            fs::remove(fs::path(impl->dbPath.string() + "-wal"), ignored);
            fs::remove(fs::path(impl->dbPath.string() + "-shm"), ignored);
        }
        std::error_code renameError;
        fs::rename(temporary, impl->dbPath, renameError);
        if (renameError) { error = "v4_rebuild_rename"; return false; }
        SyncDirectory(impl->root);
        impl->database = std::make_unique<Database>(impl->dbPath);
        impl->Schema();
        impl->sequence = maximum;
        impl->SetMeta("sequence", std::to_string(impl->sequence));
        impl->SetMeta("initialized", "1");
        // S2-4：重建出来的 catalog 已带索引列，标记投影版本为 2，Open 的版本闸门据此
        // 判断是否需要再升级（旧库没有这个键 → 强制重建一次）。
        impl->SetMeta("catalog_schema", "2");
        impl->SetMeta("append_blocked", impl->appendBlocked ? "1" : "0");
        // §17.1（修复③）：重建/压缩后刷新日志长度水位，避免下次 Open 误判"投影落后"。
        impl->RefreshLogWatermark();
        impl->rebuildExcluded += excludedRoots;
        if (excludedRoots) {
            std::fprintf(stderr, "[Prefix SSD] v4 rebuild excluded %llu checkpoint(s) with unverifiable objects.\n",
                         (unsigned long long)excludedRoots);
        }
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

uint64_t DiskPrefixV4Store::LiveCheckpoints() const {
    impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    return impl->Scalar("SELECT COUNT(*) FROM roots WHERE live=1");
}
uint64_t DiskPrefixV4Store::LiveBytes() const {
    impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    return impl->Scalar("SELECT COALESCE(SUM(bytes),0) FROM roots WHERE live=1");
}
// P3-B（2026-10-01）：已受理但未持久化的在途字节。**刻意不做读侧收尾**——一旦排空
// 流水线，"在途"就恒为 0，失去记账意义。只读取受理时累计、出队时扣减的当前值。
uint64_t DiskPrefixV4Store::InFlightBytes() const {
    return impl->InFlight();
}

// §保最新、丢最旧：取消最老的"已受理未持久化"作业，把额度让给更新的
// 检查点。成功时返回 true 并回填释放字节数；失败原因见 error（无牺牲者时是
// v4_no_cancellable_pending，调用方必须保持 fail-closed 并如实记录）。绝不删除已持久化数据。
bool DiskPrefixV4Store::CancelOldestPending(uint64_t *freedBytes, std::string &error) {
    return impl->CancelOldestPending(freedBytes, error);
}

bool DiskPrefixV4Store::CommitCheckpoint(const std::string &key, const std::string &identity,
                                         const std::vector<Object> &objects,
                                         const std::string &metadata, uint64_t *outSequence,
                                         std::string &error) {
    std::vector<Object> owned(objects);      // 兼容路径：按值调用者需要一份可变副本
    return CommitCheckpoint(key, identity, std::move(owned), metadata, outSequence, error);
}

bool DiskPrefixV4Store::CommitCheckpoint(const std::string &key, const std::string &identity,
                                         std::vector<Object> &&objects,
                                         const std::string &metadata, uint64_t *outSequence,
                                         std::string &error) {
    // 同步语义：受理 + 立刻收尾（既有调用者与全部测试的行为保持不变）。
    std::lock_guard<std::mutex> pipeline(impl->commitPipeline);
    if (!SubmitPipeline(key, identity, std::move(objects), metadata, error)) return false;
    return ReapPipeline(outSequence, error);
}

bool DiskPrefixV4Store::CommitCheckpointLocked(const std::string &key, const std::string &identity,
                                         std::vector<Object> &objects,
                                         const std::string &metadata, uint64_t *outSequence,
                                         std::string &error) {
    error.clear();
    const uint64_t cpuStart = ThreadCpuNs();
    struct CpuGuard {
        Impl *impl; uint64_t start;
        ~CpuGuard() { impl->actorCpuNs += ThreadCpuNs() - start; }
    } cpuGuard{impl.get(), cpuStart};
    if (metadata.empty()) { error = "v4_checkpoint_metadata"; return false; }
    // 外层一次性持有 root 租约，内部走 Bare 版本，避免同进程 flock 自锁。
    Impl::RootLocks locks;
    const uint64_t lockStart = NowNs();
    const bool acquired = impl->AcquireLocks(locks, false, true, error);
    impl->lockNs += NowNs() - lockStart;
    if (!acquired) return false;
    return CommitBareLocked(key, identity, objects, outSequence, error, &metadata);
}

bool DiskPrefixV4Store::SubmitPipeline(const std::string &key, const std::string &identity,
                                      std::vector<Object> &&objects, const std::string &metadata,
                                      std::string &error) {
    error.clear();
    // §⑥ 所有提交（包括"已备好对象"的旧 API 路径）都走同一条 FIFO，由唯一备菜线程按
    // 受理顺序投递给唯一写盘 actor —— 保证"投递顺序 = 提交顺序"。本函数不再直接占用/
    // 等待在途槽位；深度上限 2 由 EnqueuePipeline 在受理处强制。
    auto job = std::make_shared<Impl::PrepJob>();
    job->key = key;
    job->identity = identity;
    job->prepared = true;
    job->metadata = metadata;
    job->objects = std::move(objects);
    job->result = std::make_shared<Impl::CommitResult>();
    return impl->EnqueuePipeline(job, error);
}

bool DiskPrefixV4Store::SubmitStaged(const std::string &key, const std::string &identity,
                                     const std::string &defaultFamily, DiskPrefixCache::Json state,
                                     bool keepEquivalent,
                                     std::vector<std::pair<std::string,std::string>> &&staged,
                                     std::string &error) {
    return SubmitStaged(key, identity, defaultFamily, std::move(state), keepEquivalent,
                        std::move(staged), error, {});
}
bool DiskPrefixV4Store::SubmitStaged(const std::string &key, const std::string &identity,
                                     const std::string &defaultFamily, DiskPrefixCache::Json state,
                                     bool keepEquivalent,
                                     std::vector<std::pair<std::string,std::string>> &&staged,
                                     std::string &error, std::function<void(bool)> completion) {
    return SubmitStaged(key, identity, defaultFamily, std::move(state), keepEquivalent,
                        std::move(staged), error, std::move(completion), {});
}
bool DiskPrefixV4Store::SubmitStaged(const std::string &key, const std::string &identity,
                                     const std::string &defaultFamily, DiskPrefixCache::Json state,
                                     bool keepEquivalent,
                                     std::vector<std::pair<std::string,std::string>> &&staged,
                                     std::string &error, std::function<void(bool)> completion,
                                     std::shared_ptr<ObjectLease> objectLease,
                                     const DiskPrefixCache::CommitIdentity &suppliedIdentity) {
    DiskPrefixCache::CommitIdentity commitIdentity = suppliedIdentity;
    error.clear();
    // §⑥ 请求线程只做移交：状态 Json 与暂存分块入 FIFO 后立即返回"已受理"。
    std::lock_guard<std::mutex> pipeline(impl->commitPipeline);
    auto job = std::make_shared<Impl::PrepJob>();
    job->key = key;
    job->identity = identity;
    job->family = defaultFamily;
    if (!commitIdentity.HasAny()) {
        auto number = [&](const char *name) -> uint64_t {
            return state[name].is_number() ? uint64_t(state[name].number_value()) : 0;
        };
        commitIdentity.requestId = number("request_id");
        commitIdentity.requestGeneration = number("request_generation");
        commitIdentity.pageGeneration = number("page_generation");
        commitIdentity.copyGeneration = number("copy_generation");
        commitIdentity.cacheIdentity = state["cache_identity"].string_value();
        commitIdentity.ownerLease = number("owner_lease");
    }
    if (commitIdentity.HasAny() && !commitIdentity.Valid()) {
        error = "identity_missing";
        return false;
    }
    job->commitIdentity = commitIdentity;
    job->keepEquivalent = keepEquivalent;
    job->prepared = false;
    job->state = std::move(state);
    job->staged = std::move(staged);
    job->objectLease = std::move(objectLease);
    job->result = std::make_shared<Impl::CommitResult>();
    job->result->identity = job->commitIdentity;
    job->result->completion = std::move(completion);
    return impl->EnqueuePipeline(job, error);
}

bool DiskPrefixV4Store::ReapPipeline(uint64_t *outSequence, std::string &error) {
    return impl->ReapPending(error, outSequence);
}

bool DiskPrefixV4Store::SubmitCheckpoint(const std::string &key, const std::string &identity,
                                         std::vector<Object> &&objects, const std::string &metadata,
                                         std::string &error) {
    std::lock_guard<std::mutex> pipeline(impl->commitPipeline);
    return SubmitPipeline(key, identity, std::move(objects), metadata, error);
}

bool DiskPrefixV4Store::HasObject(const std::string &hash) const {
    try {
        Statement query(impl->database->value, "SELECT 1 FROM objects WHERE hash=? LIMIT 1");
        query.Text(1, hash);
        return query.Row();
    } catch (...) {
        return false;
    }
}

bool DiskPrefixV4Store::ReapCheckpoint(uint64_t *outSequence, std::string &error) {
    std::lock_guard<std::mutex> pipeline(impl->commitPipeline);
    return ReapPipeline(outSequence, error);
}

bool DiskPrefixV4Store::CommitObjectsLocked(const std::string &key, const std::string &identity,
                                       std::vector<Object> &objects, uint64_t *outSequence,
                                       std::string &error) {
    error.clear();
    const uint64_t cpuStart = ThreadCpuNs();
    struct CpuGuard {
        Impl *impl; uint64_t start;
        ~CpuGuard() { impl->actorCpuNs += ThreadCpuNs() - start; }
    } cpuGuard{impl.get(), cpuStart};
    // 外层一次性持有 root 租约，内部走 Bare 版本，避免同进程 flock 自锁。
    Impl::RootLocks locks;
    const uint64_t lockStart = NowNs();
    const bool acquired = impl->AcquireLocks(locks, false, true, error);
    impl->lockNs += NowNs() - lockStart;
    if (!acquired) return false;
    return CommitBareLocked(key, identity, objects, outSequence, error);
}

bool DiskPrefixV4Store::CommitDescriptorLocked(const std::string &key, const std::string &identity,
                                          const std::string &metadata, std::string &error) {
    error.clear();
    if (metadata.empty()) { error = "v4_checkpoint_metadata"; return false; }
    Impl::RootLocks locks;
    const uint64_t lockStart = NowNs();
    const bool acquired = impl->AcquireLocks(locks, false, true, error);
    impl->lockNs += NowNs() - lockStart;
    if (!acquired) return false;
    // The descriptor is appended as a METADATA record so the projection can be
    // rebuilt from the log alone; value = identity '\n' descriptor JSON.
    prefix_v4::MetadataRecord record;
    record.sequence = impl->sequence + 1;
    record.key = "checkpoint:" + key;
    record.value = identity + "\n" + metadata;
    if (record.value.size() > prefix_v4::kMaxMetadataValueBytes) {
        error = "metadata_value_too_large";
        return false;
    }
    const std::vector<char> payload = EncodeMetadata(record);
    if (impl->quota > 0 && Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota) {
        // Segments: make room by deleting the oldest whole segments (same path as a checkpoint save).
        std::string reclaimReason;
        while (impl->segmented &&
               Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota &&
               impl->ReclaimOldestSegment(reclaimReason)) {}
        if (Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota) {
            if (impl->segmented)
                std::fprintf(stderr, "[Prefix SSD] v4 segment-reclaim-skipped: site=metadata reason=%s physical=%llu "
                                     "quota=%llu (write withheld).\n", reclaimReason.c_str(),
                             (unsigned long long)impl->PhysicalBytes(), (unsigned long long)impl->quota);
            impl->RequestCompact();
            error = "capacity_unavailable";
            return false;
        }
    }
    if (!impl->AppendLiveRecord(kRecordMetadata, ++impl->sequence,
                                std::string(payload.data(), payload.size()), nullptr, error)) {
        --impl->sequence;
        return false;
    }
    {
        Transaction transaction(*impl->database);
        // S2-4 写入点 4/4（描述符补写路径）：同样写全索引列。
        Impl::InsertCheckpointRow(*impl->database, identity, key, metadata, impl->sequence);
        transaction.Commit();
    }
    impl->SetMeta("sequence", std::to_string(impl->sequence));
    impl->RefreshLogWatermark();   // 修复③：该路径也往 live 日志追加了 METADATA 记录
    return true;
}

bool DiskPrefixV4Store::CommitObjects(const std::string &key, const std::string &identity,
                                     std::vector<Object> &&objects, uint64_t *outSequence,
                                     std::string &error) {
    const auto waitStart = std::chrono::steady_clock::now();
    const bool done = impl->RunActor([&] {
        return CommitObjectsLocked(key, identity, objects, outSequence, error);
    }, error);
    impl->actorWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - waitStart).count());
    return done;
}

bool DiskPrefixV4Store::CommitDescriptor(const std::string &key, const std::string &identity,
                                        const std::string &descriptor, std::string &error) {
    const auto waitStart = std::chrono::steady_clock::now();
    const bool done = impl->RunActor([&] {
        return CommitDescriptorLocked(key, identity, descriptor, error);
    }, error);
    impl->actorWaitNs += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - waitStart).count());
    return done;
}

void DiskPrefixV4Store::AddMetadataTiming(uint64_t refWalkNs, uint64_t encodeNs, uint64_t hashNs,
                                          size_t bytes) {
    impl->refWalkNs += refWalkNs;
    impl->metadataEncodeNs += encodeNs;
    impl->metadataHashNs += hashNs;
    impl->metadataBytes += bytes;
}
void DiskPrefixV4Store::AddPrecheckTiming(uint64_t scanNs) { impl->precheckScanNs += scanNs; }
void DiskPrefixV4Store::AddActorWaitTiming(uint64_t waitNs) { impl->actorWaitNs += waitNs; }

std::string DiskPrefixV4Store::Profile(bool drain) const {
    // drain=false（模型每检查点的 export-profile 热路径）：**刻意不做读侧收尾**。
    // 一旦在此排空流水线，请求线程就会在提交后立刻等后台备菜+写盘，把刚建立的
    // 重叠全部抵消（"请求线程只交出状态就返回"失效）。代价：计数与
    // live_checkpoints/live_bytes 可能滞后一笔在途提交——诊断口径，可接受。
    // drain=true（默认，测试与离线诊断）：先收尾，保证计数覆盖已受理的提交。
    if (drain) impl->DrainForRead();
    const auto journal = prefix_v4::JournalStatsSnapshot();
    // 计数用整数、耗时才用浮点：json11 的 double 会打成 12 位小数，不适合当计数看。
    return Json(Json::object{
        {"journal_calls", (long long)journal.calls},
        {"journal_bytes", (long long)journal.bytes},
        {"journal_write_ms", double(journal.writeNs) / 1e6},
        {"journal_sync_ms", double(journal.syncNs) / 1e6},
        // §14：data_sync 与 control_sync 必须分开上报。
        {"data_sync_count", (long long)journal.dataSyncs},
        {"data_sync_ms", double(journal.dataSyncNs) / 1e6},
        {"control_sync_count", (long long)journal.controlSyncs},
        {"control_sync_ms", double(journal.controlSyncNs) / 1e6},
        {"record_hash_bytes", (long long)journal.recordHashBytes},
        {"record_hash_ms", double(journal.recordHashNs) / 1e6},
        {"dir_syncs", (long long)journal.dirSyncs},
        {"dir_sync_ms", double(journal.dirSyncNs) / 1e6},
        {"object_pack_ms", double(impl->packNs) / 1e6},
        {"catalog_apply_ms", double(impl->catalogNs) / 1e6},
        // §14：提交段分解（此前缺失，正是"约 1088 ms 花在哪"无法回答的原因）
        {"ref_walk_ms", double(impl->refWalkNs) / 1e6},
        {"metadata_encode_ms", double(impl->metadataEncodeNs) / 1e6},
        {"metadata_hash_ms", double(impl->metadataHashNs) / 1e6},
        {"metadata_bytes", (long long)impl->metadataBytes},
        {"precheck_scan_ms", double(impl->precheckScanNs) / 1e6},
        {"actor_wait_ms", double(impl->actorWaitNs) / 1e6},
        // §14：commit_wait_ms 拆成"排队"与"执行"，避免把调度延迟当成磁盘慢。
        {"actor_queue_ms", double(impl->actorQueueNs) / 1e6},
        {"actor_exec_ms", double(impl->actorExecNs) / 1e6},
        {"actor_cpu_ms", double(impl->actorCpuNs) / 1e6},
        {"commit_lock_ms", double(impl->lockNs) / 1e6},
        // §④ 异步提交：投递耗时、收尾等待耗时，以及"真的等到了"的次数
        // （reap_wait_count 是重叠是否兑现的直接证据；为 0 说明写盘全被藏住了）。
        {"commit_submit_ms", double(impl->commitSubmitNs) / 1e6},
        {"commit_reap_ms", double(impl->commitReapNs) / 1e6},
        {"reap_wait_ms", double(impl->reapWaitNs) / 1e6},
        {"reap_wait_count", (long long)impl->reapWaitCount},
        // §⑥ 备菜流水线：真正的"提交前准备"现在跑在后台唯一备菜线程上。
        // prep_exec_ms = 备菜执行（收尾上一笔+预检+序列化+摘要+描述符+装配+清单+投递）；
        // prep_queue_ms = 受理→开工的排队等待；prep_max_depth = FIFO 观测到的最大深度。
        {"prep_exec_ms", double(impl->prepExecNs) / 1e6},
        {"prep_queue_ms", double(impl->prepQueueNs) / 1e6},
        {"prep_jobs", (long long)impl->prepJobs},
        {"prep_max_depth", (long long)impl->prepMaxDepth},
        {"object_verify_ms", double(impl->verifyNs) / 1e6},
        {"object_verify_bytes", (long long)impl->verifyBytes},
        {"object_write_ms", double(impl->writeObjectNs) / 1e6},
        // §② 装配/拷贝细分
        {"quota_check_ms", double(impl->quotaNs) / 1e6},
        {"root_list_ms", double(impl->rootListNs) / 1e6},
        {"flush_objects_ms", double(impl->flushNs) / 1e6},
        {"prepare_ms", double(impl->prepareNs) / 1e6},
        {"commit_record_ms", double(impl->commitNs) / 1e6},
        {"root_tx_ms", double(impl->rootTxNs) / 1e6},
        {"evict_ms", double(impl->evictNs) / 1e6},
        // R3：为接收新检查点在准入路径上 FIFO 淘汰旧检查点的
        // 累计次数（本进程）。为 0 表示这一段时间没有“为新写让路”的淘汰。
        {"evicted_for_admission", (long long)impl->evictedForAdmission},
        {"cancelled_jobs", (long long)impl->cancelledJobs},
        {"cancelled_preserved_bytes", (long long)impl->cancelledPreservedBytes},
        {"publish_dangling_refs", (long long)impl->publishDanglingRefs},
        {"publish_refused_checkpoints", (long long)impl->publishRefusedCheckpoints},
        {"preserved_on_drop", (long long)impl->preservedOnDrop},
        {"unpreserved_drops", (long long)impl->unpreservedDrops},
        {"preserve_flush_failures", (long long)impl->preserveFlushFailures},
        // §② 循环内部：查存在性 / 扫待写表 / 入队拷贝 / 记录拼装
        {"lookup_ms", double(impl->lookupNs) / 1e6},
        {"pending_scan_ms", double(impl->pendingScanNs) / 1e6},
        {"stage_ms", double(impl->stageNs) / 1e6},
        {"assemble_ms", double(impl->assembleNs) / 1e6},
        {"object_count", (long long)impl->objectCount},
        {"pending_scan_steps", (long long)impl->pendingScanSteps},
        {"actor_queued_calls", (long long)impl->actorQueuedCalls},
        {"actor_inline_calls", (long long)impl->actorInlineCalls},
        {"rebuild_valid", (long long)impl->rebuildValid},
        {"rebuild_uncommitted_tail", (long long)impl->rebuildTail},
        {"rebuild_corrupt_control", (long long)impl->rebuildCorrupt},
        {"rebuild_excluded_roots", (long long)impl->rebuildExcluded},
        // SQLite 实际生效的 durability 档位（PRAGMA synchronous，2=FULL）。
        {"sqlite_sync", impl->database ? impl->database->ScalarText("PRAGMA synchronous") : ""},
        {"live_checkpoints", (long long)impl->Scalar("SELECT COUNT(*) FROM roots WHERE live=1")},
        {"live_bytes", (long long)impl->Scalar("SELECT COALESCE(SUM(bytes),0) FROM roots WHERE live=1")},
        {"physical_bytes", (long long)impl->PhysicalBytes()},
        {"physical_over_quota", impl->quota > 0 && impl->PhysicalBytes() > impl->quota},
        {"durable_checkpoints_cum", (long long)impl->durableCommits},   // A2：本进程累计落盘检查点数
        {"reused_checkpoints_cum", (long long)impl->reusedCheckpoints.load()},
        {"failed_checkpoints_cum", (long long)impl->failedCheckpoints.load()},
        {"inflight_bytes", (long long)impl->InFlight()},   // P3-B：已受理未持久化字节
        // 复核 R6（2026-10-05）：CancelOldestPending 只归还额度、不丢作业 ⇒ 这些作业仍常驻内存，
        // 但已不在 inflight_bytes 里。本值 = 额度已归还（bytesReleased）却仍留在 FIFO 的作业
        // heldBytes 之和；真实常驻 ≈ inflight_bytes + resident_quota_released_bytes。
        // 纯观测：不改变任何准入/取消/记账行为。
        {"resident_quota_released_bytes", (long long)impl->ResidentQuotaReleasedBytes()},
    }).dump();
}

// §17.1（P2 收口，2026-09-30）：清单完整性自检——用**与写方完全相同的引用口径**
// （kv/linear/speculative/linear_chunks/speculative_chunks 及其嵌套的 {sha256,bytes}）校验：
// 状态对象本体 + 它引用的每个对象，都必须能在该 root 的清单里找到。返回缺失条目总数。
// 抽成独立接口的目的是**可被反例断言**：历史缺陷（root 清单漏登记被去重对象 → 重开只命中
// 2048）正是靠"只看 Load 布尔值"的旧自检溜过去的。
// 修复4（2026-10-03，RCA 4）：SELFCHECK 覆盖上限。旧实现三处硬编码 `checked >= 64` ⇒
// 在真机 T2 的规模（上千个检查点）下，门⑦最多只覆盖 **6.4%**，未检出的坏点不会反映在
// `unrestorable` 上（属"看起来通过其实没查"）。现在默认 **0 = 不设上限（全量检查）**，
// 可用 `FASTLLM_PREFIX_CACHE_SELFCHECK_MAX` 显式设上限（例如启动耗时敏感时）。
// 另：覆盖情况由 `SELFCHECK-COVERAGE` 行单独打印，绝不静默降级。
static size_t SelfCheckLimit() {
    static const size_t limit = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_SELFCHECK_MAX");
        const long long parsed = text != nullptr ? std::atoll(text) : 0;
        return parsed > 0 ? (size_t)parsed : (size_t)0;
    }();
    return limit;
}
size_t DiskPrefixV4Store::SelfCheckManifest(std::string &details) {
    details.clear();
    const auto rows = Checkpoints();
    const size_t limit = SelfCheckLimit();
    size_t checked = 0, missing = 0;
    for (const auto &row : rows) {
        if (limit != 0 && checked >= limit) break;   // 0 = 全量（见 SelfCheckLimit）
        ++checked;
        try {
            std::string rootHash;
            {
                Statement rootQuery(impl->database->value,
                                    "SELECT root FROM roots WHERE identity=? AND key=? AND live=1");
                rootQuery.Text(1, row.identity); rootQuery.Text(2, row.key);
                if (!rootQuery.Row()) continue;
                rootHash = rootQuery.Text(0);
            }
            std::string manifestError;
            std::string listText;
            if (!impl->ObjectBytes(rootHash, listText, manifestError)) {
                ++missing;
                if (details.size() < 512) details += row.key.substr(0, 16) + "=>manifest_root;";
                continue;
            }
            std::vector<ObjectListEntry> listed;
            std::string listDecodeError;
            if (!DecodeObjectList(listText.data(), listText.size(), listed, listDecodeError)) {
                // Route B：root 是紧凑状态对象（无扁平清单）。可达性 = 状态本体 + 状态引用
                // （linear/spec 分片）+ 链上每个块对象 + 块引用；逐项确认对象表里确实存在，
                // 链上任一环缺失都算缺失（这正是"清单完整性"在 Route B 下的等价口径）。
                std::string parseError;
                const Json descriptor = Json::parse(row.metadata, parseError);
                const Json state = Json::parse(listText, parseError);
                if (!parseError.empty() || !state.is_object() ||
                    !state["chain_tip"].is_string() || !state["block_count"].is_number()) {
                    ++missing;
                    if (details.size() < 512) details += row.key.substr(0, 16) + "=>chain_decode;";
                    continue;
                }
                std::set<std::string> referenced{descriptor["state"].string_value()};
                Refs stateRefs; CollectRefs(state, stateRefs);
                for (const auto &ref : stateRefs) referenced.insert(ref.first);
                const uint64_t blockCount = uint64_t(state["block_count"].number_value());
                bool chainOk = blockCount > 0 && blockCount <= prefix_v4::kMaxBlockCount;
                std::string cursor = state["chain_tip"].string_value();
                for (uint64_t step = 0; chainOk && step < blockCount; ++step) {
                    Statement blockQuery(impl->database->value,
                                         "SELECT object_hash,parent FROM blocks WHERE chain_key=?");
                    blockQuery.Text(1, cursor);
                    if (!blockQuery.Row()) { chainOk = false; break; }
                    const std::string objectHash = blockQuery.Text(0);
                    const std::string parent = blockQuery.Text(1);
                    referenced.insert(objectHash);
                    std::string blockText;
                    if (!impl->ObjectBytes(objectHash, blockText, parseError)) { chainOk = false; break; }
                    const Json block = Json::parse(blockText, parseError);
                    if (!parseError.empty() || !block.is_object()) { chainOk = false; break; }
                    Refs blockRefs; CollectRefs(block["kv"], blockRefs);
                    for (const auto &ref : blockRefs) referenced.insert(ref.first);
                    cursor = parent;
                }
                if (!chainOk) {
                    ++missing;
                    if (details.size() < 512) details += row.key.substr(0, 16) + "=>chain;";
                    continue;
                }
                size_t rowMissing = 0;
                for (const auto &hash : referenced) if (!impl->HasObject(hash)) ++rowMissing;
                if (rowMissing) {
                    missing += rowMissing;
                    if (details.size() < 512) {
                        details += row.key.substr(0, 16) + "=>missing=" + std::to_string(rowMissing) + ";";
                    }
                }
                continue;
            }
            std::set<std::string> inManifest;
            for (const auto &entry : listed) inManifest.insert(entry.hash);
            std::string parseError;
            const Json descriptor = Json::parse(row.metadata, parseError);
            if (!parseError.empty()) continue;
            const std::string stateHash = descriptor["state"].string_value();
            if (stateHash.empty()) continue;
            std::set<std::string> referenced{stateHash};   // 状态本体也必须能被清单取回
            std::string stateText;
            if (impl->ObjectBytes(stateHash, stateText, parseError)) {
                const Json state = Json::parse(stateText, parseError);
                if (parseError.empty()) {
                    std::function<void(const Json &)> collectRefs = [&](const Json &node) {
                        if (node.is_array()) {
                            for (const auto &item : node.array_items()) collectRefs(item);
                        } else if (node.is_object()) {
                            const Json &sha = node["sha256"];
                            const Json &bytes = node["bytes"];
                            if (sha.is_string() && bytes.is_number()) referenced.insert(sha.string_value());
                            for (const char *field : {"chunks","kv","linear","speculative",
                                                      "linear_chunks","speculative_chunks"}) {
                                const Json &child = node[field];
                                if (!child.is_null()) collectRefs(child);
                            }
                        }
                    };
                    for (const char *field : {"kv","linear","speculative","linear_chunks",
                                              "speculative_chunks"}) {
                        const Json &child = state[field];
                        if (!child.is_null()) collectRefs(child);
                    }
                }
            }
            size_t rowMissing = 0;
            for (const auto &hash : referenced) if (!inManifest.count(hash)) ++rowMissing;
            if (rowMissing) {
                missing += rowMissing;
                if (details.size() < 512) {
                    details += row.key.substr(0, 16) + "=>missing=" + std::to_string(rowMissing) + ";";
                }
            }
        } catch (const std::exception &) {
            ++missing;
            if (details.size() < 512) details += row.key.substr(0, 16) + "=>manifest_exception;";
        }
    }
    return missing;
}

// S6（2026-10-05 复核；门⑦"不再假绿"）：功能性可恢复校验。
// 背景：旧 SELFCHECK 只做 store 级 Load——"对象齐、能拼回字节"就算可恢复。历史故障
// （spec 状态错位 / linear 层缺失 / 块链虽然有对象但 token 覆盖不连续）都能骗过它。
// 本函数不看"对象在不在"（那是 SelfCheckManifest 的职责），只看**读回来能不能真的继续解码**：
//   ① KV 覆盖连续：Route B 块链必须从 token 0 起首尾相接（ChainBlocks 已校验 index/parent/
//      token 邻接），且链尾 token_end == 检查点 length；非链式（无 chain_tip）时按 kv 分块的
//      begin/tokens 拼出 [0,length) 连续覆盖。
//   ② linear 层齐全：layout.linear[layer]==true 的每一层，状态里必须恰好有一条 layer==i 的
//      linear 条目；false 的层不得出现条目（与模型侧 Import 的 ssd_incomplete_linear 同口径）。
//   ③ spec tokens == length：若状态确实记录了 spec（speculative.kind != none），其 kind 必须与
//      layout.speculative 一致、tokens 必须 == length。**dflash2 与 mtp 同等对待**——两者的
//      spec 状态都随 length 增长（复核指出旧方案只覆盖 dflash2，mtp 会漏判）。kind == none 不判红
//      （那是"本次 generation config 不需要 spec"时写下的合法检查点，由 Import 按请求再校验）。
// 只对"状态对象带 layout 字段"的检查点判定：老式/合成状态对象（无 layout）无法判定 ⇒ 跳过，
// 不制造假红。状态对象读不回来时也跳过（那属于 Load 的失败口径，避免与 SelfCheck 重复计数）。
// 返回不合格检查点数（0 = 通过）；details 给出前若干条判据失败说明。只读。
size_t DiskPrefixV4Store::SelfCheckFunctional(std::string &details) {
    details.clear();
    const auto rows = Checkpoints();
    const size_t limit = SelfCheckLimit();
    size_t checked = 0, bad = 0;
    for (const auto &row : rows) {
        if (limit != 0 && checked >= limit) break;   // 0 = 全量（修复4）
        ++checked;
        try {
            std::string parseError;
            const Json descriptor = Json::parse(row.metadata, parseError);
            if (!parseError.empty() || !descriptor.is_object()) continue;
            const std::string stateHash = descriptor["state"].string_value();
            if (!DiskPrefixCache::IsDigest(stateHash)) continue;   // 无状态行：Lookup 侧另处理
            std::string stateText;
            if (!impl->ObjectBytes(stateHash, stateText, parseError)) continue;  // Load 会如实报错
            const Json state = Json::parse(stateText, parseError);
            if (!parseError.empty() || !state.is_object()) continue;
            const Json &layout = state["layout"];
            if (!layout.is_object() || !layout["linear"].is_array()) continue;  // 老式状态：无法判定
            const int length = state["length"].is_number() ? state["length"].int_value()
                                                           : row.length;
            std::string reason;
            // ① KV 覆盖连续
            if (state["chain_tip"].is_string() && state["block_count"].is_number()) {
                const uint64_t count = uint64_t(state["block_count"].number_value());
                std::vector<BlockRow> blocks;
                std::string chainError;
                if (!ChainBlocks(state["chain_tip"].string_value(), count, blocks, chainError)) {
                    reason = "kv_chain:" + (chainError.empty() ? std::string("unknown") : chainError);
                } else if (blocks.empty() || blocks.back().tokenEnd != uint64_t(length)) {
                    reason = "kv_chain_coverage";
                } else if (state["chained_tokens"].is_number() &&
                           uint64_t(state["chained_tokens"].number_value()) != uint64_t(length)) {
                    reason = "kv_chained_tokens";
                }
            } else if (state["kv"].is_array()) {
                // 非链式（Route B 未启用）：按分块 begin/tokens 还原覆盖区间。
                std::map<uint64_t, uint64_t> spans;
                for (const auto &bundle : state["kv"].array_items()) {
                    for (const auto &seg : bundle["segments"].array_items()) {
                        if (!seg["begin"].is_number() || !seg["tokens"].is_number() ||
                            seg["tokens"].int_value() <= 0) {
                            reason = "kv_segment_shape"; break;
                        }
                        const uint64_t begin = uint64_t(seg["begin"].number_value());
                        const uint64_t tokens = uint64_t(seg["tokens"].number_value());
                        auto found = spans.find(begin);
                        if (found == spans.end()) spans[begin] = tokens;
                        else if (found->second != tokens) reason = "kv_segment_ragged";
                    }
                    if (!reason.empty()) break;
                }
                if (reason.empty()) {
                    uint64_t cursor = 0;
                    if (spans.empty()) reason = "kv_empty";
                    for (const auto &span : spans) {
                        if (span.first != cursor) { reason = "kv_gap_or_overlap"; break; }
                        cursor += span.second;
                    }
                    if (reason.empty() && cursor != uint64_t(length)) reason = "kv_short_coverage";
                }
            } else {
                reason = "kv_missing";
            }
            // ② linear 层齐全（true 层不多不少）
            if (reason.empty()) {
                const auto &flags = layout["linear"].array_items();
                const int layers = layout["layers"].is_number() ? layout["layers"].int_value()
                                                                : (int)flags.size();
                if ((int)flags.size() != layers || layers <= 0) {
                    reason = "linear_layout";
                } else {
                    std::set<int> present;
                    for (const auto &entry : state["linear"].array_items()) {
                        if (!entry["layer"].is_number()) { reason = "linear_entry_shape"; break; }
                        const int layer = entry["layer"].int_value();
                        if (layer < 0 || layer >= layers || !present.insert(layer).second) {
                            reason = "linear_duplicate_or_range"; break;
                        }
                    }
                    for (int layer = 0; reason.empty() && layer < layers; ++layer) {
                        const bool need = flags[(size_t)layer].bool_value();
                        const bool have = present.count(layer) != 0;
                        if (need != have) reason = need ? "linear_missing_layer" : "linear_extra_layer";
                    }
                }
            }
            // ③ spec tokens == length（dflash2 与 mtp 同等对待）
            if (reason.empty()) {
                const std::string expected = layout["speculative"].string_value();
                const Json &spec = state["speculative"];
                if (!expected.empty()) {
                    if (!spec.is_object()) {
                        reason = "speculative_missing";
                    } else if (spec["kind"].string_value() != "none") {
                        // 本检查点**确实记录了** spec 状态：kind 必须与布局一致，且 tokens 必须等于
                        // 检查点长度。dflash2 的 window 可短于 length，但它的 tokens 是保存时的
                        // checkpoint 长度；mtp 的 key/value 第 1 维直接等于 length。两者同判据。
                        if (spec["kind"].string_value() != expected) {
                            reason = "speculative_kind";
                        } else if (!spec["tokens"].is_number() ||
                                   spec["tokens"].int_value() != length) {
                            reason = spec["kind"].string_value() + "_tokens_mismatch";
                        }
                    }
                    // kind == "none"：该检查点是在"本次 generation config 不需要 spec"时写的
                    // （Qwen35MtpSupportsGenerationConfig 为假），对不需要 spec 的请求仍然可恢复，
                    // 不能判红；是否需要 spec 由 Import 按当前请求的 require* 再校验。
                }
            }
            if (!reason.empty()) {
                ++bad;
                if (details.size() < 512) details += row.key.substr(0, 16) + "=>" + reason + ";";
            }
        } catch (const std::exception &e) {
            ++bad;
            if (details.size() < 512) {
                details += row.key.substr(0, 16) + "=>functional_exception:";
                details += e.what();
                details += ";";
            }
        }
    }
    return bad;
}

void DiskPrefixV4Store::SelfCheck() {
    static const bool color = (isatty(fileno(stdout)) != 0);
    const char *green = color ? "\033[1;32m" : "";
    const char *red = color ? "\033[1;31m" : "";
    const char *reset = color ? "\033[0m" : "";
    // 定位用：逐检查点进度 + 各子步时间戳。默认关闭，
    // FASTLLM_PREFIX_CACHE_SELFCHECK_TRACE=1 打开。目的：区分"在算"（标记持续前进）
    // 与"在等"（标记停住），并指出是哪一步慢。纯观测，语义不变。
    static const bool trace = [] {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_SELFCHECK_TRACE");
        return PrefixCacheDebugEnabled() && text != nullptr && text[0] != '\0' && text[0] != '0';
    }();
    const uint64_t traceStart = NowNs();
    auto mark = [&](const char *stage, size_t n) {
        if (!trace) return;
        std::fprintf(stderr,"[Prefix SSD][mark] selfcheck_%s checked=%zu elapsed_ms=%.0f\n",
                     stage, n, double(NowNs() - traceStart) / 1e6);
        std::fflush(stderr);
    };
    mark("enter", 0);
    const auto rows = Checkpoints();
    mark("rows_loaded", rows.size());
    const size_t limit = SelfCheckLimit();
    size_t checked = 0, ok = 0, bad = 0;
    std::vector<std::string> failures;
    for (const auto &row : rows) {
        if (limit != 0 && checked >= limit) break;   // 0 = 全量（修复4）
        ++checked;
        if (trace && (checked % 8 == 0)) mark("load_progress", checked);
        std::string identity;
        std::vector<std::string> objects;
        std::string error;
        bool loaded = false;
        try { loaded = Load(row.key, identity, objects, error); }
        catch (const std::exception &e) { error = e.what(); loaded = false; }
        if (loaded) {
            ++ok;
        } else {
            ++bad;
            if (failures.size() < 8) {
                failures.push_back(row.key.substr(0,16) + "=>" + (error.empty() ? "unknown" : error));
            }
        }
    }
    mark("load_done", checked);
    std::string manifestDetails;
    mark("manifest_begin", checked);
    const size_t manifestMissing = SelfCheckManifest(manifestDetails);
    mark("manifest_done", checked);
    // S6（2026-10-05）：功能性校验单独计数，但**并入 unrestorable**——门⑦（extract_verdict.py
    // 的 "restorable=N unrestorable=M"）只认这一行；不并入则"能 Load 但状态错位/缺失"仍会假绿。
    // SelfCheckFunctional 对"状态对象读不回"的行会跳过（那已由上面的 Load 计入 bad），故不重复计数。
    std::string functionalDetails;
    mark("functional_begin", checked);
    const size_t functionalBad = SelfCheckFunctional(functionalDetails);
    mark("functional_done", checked);
    printf("%s[Prefix SSD] SELFCHECK: checkpoints=%zu checked=%zu restorable=%zu unrestorable=%zu.%s\n",
           (bad || functionalBad || manifestMissing) ? red : green, rows.size(), checked, ok,
           bad + functionalBad, reset);
    // 修复4：覆盖情况单独一行（绝不静默降级）。limit=0 表示全量检查；
    // 若 checked < checkpoints（打印 PARTIAL），说明门⑦的覆盖被上限截断，必须人工可见。
    printf("[Prefix SSD] SELFCHECK-COVERAGE: checkpoints=%zu checked=%zu limit=%zu (0=all)%s\n",
           rows.size(), checked, limit, checked < rows.size() ? " PARTIAL" : "");
    printf("%s[Prefix SSD] SELFCHECK-MANIFEST: missing=%zu %s%s\n",
           manifestMissing ? red : green, manifestMissing,
           manifestDetails.empty() ? "" : manifestDetails.c_str(), reset);
    printf("%s[Prefix SSD] SELFCHECK-FUNCTIONAL: bad=%zu %s%s\n",
           functionalBad ? red : green, functionalBad,
           functionalDetails.empty() ? "" : functionalDetails.c_str(), reset);
    for (const auto &item : failures) {
        printf("%s[Prefix SSD] SELFCHECK-BAD: %s.%s\n", red, item.c_str(), reset);
    }
    mark("done", checked);
    fflush(stdout);
}

std::vector<DiskPrefixV4Store::CheckpointRow> DiskPrefixV4Store::Checkpoints() const {
    impl->DrainForRead();   // §④ 读侧收尾：必须看到已受理的提交
    // 只返回 live root：被 ERASE 退休的检查点即使 descriptor 行还在也不能返回。
    // 查询体与 Impl::CheckpointsBare() 共用同一口径（后者给备菜线程预检用，不做收尾）。
    return impl->CheckpointsBare();
}

bool DiskPrefixV4Store::PutMetadata(const std::string &key, const std::string &value,
                                    std::string &error) {
    return impl->RunActor([&] { return PutMetadataLocked(key, value, error); }, error);
}

bool DiskPrefixV4Store::PutMetadataLocked(const std::string &key, const std::string &value,
                                    std::string &error) {
    error.clear();
    if (key.empty()) { error = "v4_metadata_key"; return false; }
    // P3 修复 A（2026-10-01）：与其它写路径一致，**先取 root 租约**再分配 sequence / 追加
    // live 记录。此前本函数不取锁，可在同一 root 的另一实例提交时并发分配 sequence。
    Impl::RootLocks locks;
    if (!impl->AcquireLocks(locks, false, true, error)) return false;
    prefix_v4::MetadataRecord record;
    record.sequence = impl->sequence + 1;
    record.key = "kv:" + key;
    record.value = value;
    if (record.value.size() > prefix_v4::kMaxMetadataValueBytes) {
        error = "metadata_value_too_large";
        return false;
    }
    const std::vector<char> payload = EncodeMetadata(record);
    if (impl->quota > 0 && Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota) {
        // Segments: make room by deleting the oldest whole segments (same path as a checkpoint save).
        std::string reclaimReason;
        while (impl->segmented &&
               Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota &&
               impl->ReclaimOldestSegment(reclaimReason)) {}
        if (Add(impl->PhysicalBytes(), RecordPhysicalBytes(payload.size())) > impl->quota) {
            if (impl->segmented)
                std::fprintf(stderr, "[Prefix SSD] v4 segment-reclaim-skipped: site=metadata reason=%s physical=%llu "
                                     "quota=%llu (write withheld).\n", reclaimReason.c_str(),
                             (unsigned long long)impl->PhysicalBytes(), (unsigned long long)impl->quota);
            impl->RequestCompact();
            error = "capacity_unavailable";
            return false;
        }
    }
    if (!impl->AppendLiveRecord(kRecordMetadata, ++impl->sequence,
                                std::string(payload.data(), payload.size()), nullptr, error)) {
        --impl->sequence;
        return false;
    }
    Statement upsert(impl->database->value, "INSERT OR REPLACE INTO kv(key,value) VALUES(?,?)");
    upsert.Text(1, key);
    upsert.Text(2, value);
    upsert.Run();
    impl->SetMeta("sequence", std::to_string(impl->sequence));
    impl->RefreshLogWatermark();   // 修复③：该路径也往 live 日志追加了 METADATA 记录
    return true;
}

bool DiskPrefixV4Store::GetMetadata(const std::string &key, std::string &value,
                                    std::string &error) {
    error.clear();
    value.clear();
    Statement query(impl->database->value, "SELECT value FROM kv WHERE key=?");
    query.Text(1, key);
    if (!query.Row()) return false;
    value = query.Text(0);
    return true;
}

bool DiskPrefixV4Store::Compact(std::string &error) {
    return impl->RunActor([&] { return CompactLocked(error); }, error);
}

bool DiskPrefixV4Store::RetireCorruptCheckpoint(const std::string &key,
                                                   const std::string &identity,
                                                   const std::string &confirmedReason,
                                                   std::string &error) {
    error.clear();
    if (key.empty() || !prefix_v4::IsDigest(identity)) { error = "v4_checkpoint_argument"; return false; }
    if (DiskPrefixCache::ClassifyRestoreFailure(confirmedReason) !=
        DiskPrefixCache::RestoreFailureClass::STORED_REPRESENTATION) {
        error = "v4_corruption_retire_requires_confirmed_failure";
        return false;
    }
    return impl->RunActor([&] {
        error.clear();
        Impl::RootLocks locks;
        if (!impl->AcquireLocks(locks, false, false, error)) return false;
        bool accountingOk = true;
        {
            std::lock_guard<std::mutex> pinLock(impl->pins->mutex);
            if (!impl->CorruptRetireAllowed(key, identity)) {
                error = "v4_corrupt_retire_gate_missing";
                return false;
            }
            if (impl->IsPinnedLocked(key, identity)) {
                error = "v4_checkpoint_pinned";
                return false;
            }
            std::unique_ptr<Lease> rootLease;
            try { rootLease.reset(new Lease(impl->RootLeasePath(key, identity), LOCK_EX | LOCK_NB)); }
            catch (const std::exception &) { error = "v4_checkpoint_busy"; return false; }
            Statement sameKey(impl->database->value,
                             "SELECT identity FROM roots WHERE key=? AND live=1 ORDER BY sequence DESC LIMIT 1");
            sameKey.Text(1, key);
            if (sameKey.Row() && sameKey.Text(0) != identity) {
                error = "v4_checkpoint_identity_mismatch";
                return false;
            }
            Statement query(impl->database->value,
                            "SELECT 1 FROM roots WHERE identity=? AND key=? AND live=1 LIMIT 1");
            query.Text(1, identity); query.Text(2, key);
            if (!query.Row()) { error = "v4_root_missing"; return false; }
            prefix_v4::EraseRecord erase;
            erase.sequence = impl->sequence + 1; erase.key = key; erase.identity = identity;
            const std::vector<char> bytes = EncodeErase(erase);
            if (!impl->AppendLiveRecord(kRecordErase, ++impl->sequence,
                                        std::string(bytes.data(), bytes.size()), nullptr, error)) {
                --impl->sequence;
                return false;
            }
            Statement retire(impl->database->value,
                             "UPDATE roots SET live=0 WHERE identity=? AND key=? AND live=1");
            retire.Text(1, identity); retire.Text(2, key); retire.Run();
            impl->SetMeta("sequence", std::to_string(impl->sequence));
            impl->RefreshLogWatermark();
            std::string accountingError;
            if (!impl->RecomputeLiveBytesForQuota(accountingError)) {
                error = accountingError.empty() ? "v4_live_accounting" : accountingError;
                accountingOk = false;
            }
        }
        impl->NotifyCheckpointErased(key, identity);
        return accountingOk;
    }, error);
}

std::shared_ptr<void> DiskPrefixV4Store::PinCheckpoint(const std::string &key,
                                                       const std::string &identity,
                                                       std::string &error,
                                                       bool drain) const {
    error.clear();
    if (key.empty() || !prefix_v4::IsDigest(identity)) {
        error = "v4_checkpoint_lease_argument";
        return nullptr;
    }
    // Match the normal read path: a checkpoint admitted to the prep FIFO may
    // not have reached the catalog yet, so finish prior submissions before
    // deciding whether the root can be pinned.
    if (drain) impl->DrainForRead();
    // A read pin is only meaningful for a currently published root. Check the
    // projection before incrementing the count so a failed lookup cannot make
    // a future FIFO admission skip an unrelated checkpoint.
    // Serialize lookup + pin increment with FIFO's candidate selection. The
    // actor may otherwise select an unpinned root immediately before this
    // reader increments its count.
    std::unique_ptr<Lease> lifecycleLease(new Lease(impl->root / ".lifecycle", LOCK_SH));
    std::unique_ptr<Lease> rootLease;
    {
        std::lock_guard<std::mutex> pinLock(impl->pins->mutex);
        rootLease.reset(new Lease(impl->RootLeasePath(key, identity), LOCK_SH));
        Statement query(impl->database->value,
                         "SELECT 1 FROM roots WHERE identity=? AND key=? AND live=1 LIMIT 1");
        query.Text(1, identity); query.Text(2, key);
        if (!query.Row()) { error = "v4_root_missing"; return nullptr; }
        ++impl->pins->checkpoints[Impl::PinKey(key, identity)];
    }
    struct Guard {
        std::shared_ptr<Impl::PinState> pins;
        std::string key, identity;
        std::unique_ptr<Lease> lifecycleLease;
        std::unique_ptr<Lease> rootLease;
    };
    auto guard = std::shared_ptr<Guard>(new Guard{impl->pins, key, identity,
                                                  std::move(lifecycleLease), std::move(rootLease)},
        [](Guard *value) {
            if (value != nullptr) {
                if (value->pins) {
                    std::lock_guard<std::mutex> lock(value->pins->mutex);
                    auto found = value->pins->checkpoints.find(Impl::PinKey(value->key, value->identity));
                    if (found != value->pins->checkpoints.end()) {
                        if (found->second > 1) --found->second;
                        else value->pins->checkpoints.erase(found);
                    }
                }
                delete value;
            }
        });
    return std::static_pointer_cast<void>(guard);
}

std::shared_ptr<void> DiskPrefixV4Store::ObserveCheckpointErases(
        std::function<void(const std::string &, const std::string &)> callback) {
    if (!callback) return nullptr;
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(impl->observers->mutex);
        id = impl->observers->next++;
        impl->observers->callbacks.emplace(id, std::move(callback));
    }
    struct Guard { std::weak_ptr<Impl::ObserverState> observers; uint64_t id; };
    auto guard = std::shared_ptr<Guard>(new Guard{impl->observers, id}, [](Guard *value) {
        if (value != nullptr) {
            if (auto observers = value->observers.lock()) {
                std::lock_guard<std::mutex> lock(observers->mutex);
                observers->callbacks.erase(value->id);
            }
            delete value;
        }
    });
    return std::static_pointer_cast<void>(guard);
}

std::shared_ptr<void> DiskPrefixV4Store::ObserveCheckpointReclaims(
        DiskPrefixCache::CheckpointReclaimPredicate predicate) {
    if (!predicate) return nullptr;
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(impl->observers->mutex);
        id = impl->observers->next++;
        impl->observers->predicates.emplace(id, std::move(predicate));
    }
    struct Guard { std::weak_ptr<Impl::ObserverState> observers; uint64_t id; };
    auto guard = std::shared_ptr<Guard>(new Guard{impl->observers, id}, [](Guard *value) {
        if (value != nullptr) {
            if (auto observers = value->observers.lock()) {
                std::lock_guard<std::mutex> lock(observers->mutex);
                observers->predicates.erase(value->id);
            }
            delete value;
        }
    });
    return std::static_pointer_cast<void>(guard);
}

std::shared_ptr<void> DiskPrefixV4Store::ObserveCheckpointCorruptRetires(
        DiskPrefixCache::CheckpointCorruptRetirePredicate predicate) {
    if (!predicate) return nullptr;
    uint64_t id = 0;
    {
        std::lock_guard<std::mutex> lock(impl->observers->mutex);
        id = impl->observers->next++;
        impl->observers->corruptPredicates.emplace(id, std::move(predicate));
    }
    struct Guard { std::weak_ptr<Impl::ObserverState> observers; uint64_t id; };
    auto guard = std::shared_ptr<Guard>(new Guard{impl->observers, id}, [](Guard *value) {
        if (value != nullptr) {
            if (auto observers = value->observers.lock()) {
                std::lock_guard<std::mutex> lock(observers->mutex);
                observers->corruptPredicates.erase(value->id);
            }
            delete value;
        }
    });
    return std::static_pointer_cast<void>(guard);
}

std::shared_ptr<DiskPrefixV4Store::ObjectLease>
DiskPrefixV4Store::BeginObjectLease(std::string &error) const {
    error.clear();
    try {
        // Only the short manifest mutation excludes Compact. The returned
        // owner pins its hashes, not every byte in the canonical store.
        Lease lifecycle(impl->root / ".lifecycle", LOCK_SH);
        return std::make_shared<ObjectLease>(impl->root);
    } catch (const std::exception &exception) {
        error = exception.what();
        return nullptr;
    }
}

bool DiskPrefixV4Store::ExtendObjectLease(const std::shared_ptr<ObjectLease> &lease,
        const std::vector<std::string> &hashes, std::string &error) const {
    error.clear();
    if (!lease) { error = "v4_object_lease_missing"; return false; }
    try {
        Lease lifecycle(impl->root / ".lifecycle", LOCK_SH);
        lease->Add(hashes);
        return true;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}

bool DiskPrefixV4Store::CompactLocked(std::string &error) {
    if (impl->segmented) { error.clear(); return true; }   // DESIGN-v3-segments C7
    error.clear();
    V4InvalidateRecordCache();   // [P1-a] 压缩会把新文件 rename 到**同一路径** ⇒ 记录缓存必须失效
    if (impl->appendBlocked) {
        error = "v4_control_uncommitted_tail";
        return false;
    }
    Impl::RootLocks locks;
    if (!impl->AcquireLocks(locks, true, false, error)) return false;
    try {
        // Collect every object reachable from a live root.
        std::set<std::string> keep;
        // Keep precisely the existing objects leased by active writers,
        // including jobs with a terminal durable failure. They must not block
        // reclaiming unrelated retired data for the lifetime of the request.
        // Nonexistent hashes remain owned in RAM and need no physical copy.
        if (!V4CollectObjectLeases(impl->root, keep, error)) return false;
        for (const auto &pending : impl->pendingObjects) keep.insert(pending.hash);
        for (auto entry = keep.begin(); entry != keep.end();) {
            if (!impl->HasObject(*entry)) entry = keep.erase(entry);
            else ++entry;
        }
        {
            // S2-5a 缺陷修复：压缩会从 0 重排 sequence（见下方 newSequence），若这里不按
            // 旧 sequence 排序，重排后的 FIFO 淘汰顺序就变成任意顺序（等价于随机淘汰）。
            // 显式 ORDER BY sequence ASC, identity, key 保证“写盘顺序 = 老化顺序”在压缩后
            // 仍然保持。
            Statement roots(impl->database->value,
                            "SELECT root FROM roots WHERE live=1 "
                            "ORDER BY sequence ASC, identity, key");
            while (roots.Row()) {
                const std::string rootHash = roots.Text(0);
                std::string listText;
                if (!impl->ObjectBytes(rootHash, listText, error)) return false;
                std::vector<ObjectListEntry> list;
                std::string listError;
                if (DecodeObjectList(listText.data(), listText.size(), list, listError)) {
                    keep.insert(rootHash);
                    for (const auto &entry : list) keep.insert(entry.hash);
                    continue;
                }
                // Route B：没有扁平清单，可达集 = 状态本体 + 状态引用（linear/spec 分片）
                // + 链上每个块对象 + 块对象引用的 KV 分片。漏掉任何一项都会被当垃圾清掉。
                std::string parseError; const Json state = Json::parse(listText, parseError);
                if (!parseError.empty() || !state.is_object() || !state["chain_tip"].is_string() ||
                    !state["block_count"].is_number()) {
                    error = "v4_root_object";
                    return false;
                }
                keep.insert(rootHash);
                Refs stateRefs; CollectRefs(state, stateRefs);
                for (const auto &ref : stateRefs) keep.insert(ref.first);
                const uint64_t blockCount = uint64_t(state["block_count"].number_value());
                if (blockCount == 0 || blockCount > prefix_v4::kMaxBlockCount) {
                    error = "v4_block_count";
                    return false;
                }
                std::string cursor = state["chain_tip"].string_value();
                for (uint64_t step = 0; step < blockCount; ++step) {
                    Statement block(impl->database->value,
                                    "SELECT object_hash,parent FROM blocks WHERE chain_key=?");
                    block.Text(1, cursor);
                    if (!block.Row()) {
                        error = (step == 0) ? "v4_block_missing" : "v4_block_parent_missing";
                        return false;
                    }
                    const std::string objectHash = block.Text(0);
                    const std::string parent = block.Text(1);
                    keep.insert(objectHash);
                    std::string blockText;
                    if (!impl->ObjectBytes(objectHash, blockText, error)) {
                        error = "v4_block_missing";
                        return false;
                    }
                    std::string blockError; const Json blockObject = Json::parse(blockText, blockError);
                    DiskPrefixV4Store::BlockRow parsed;
                    std::string rowError;
                    if (!blockError.empty() || !ParseBlockRow(blockObject, parsed, rowError)) {
                        error = rowError.empty() ? "v4_block_object" : rowError;
                        return false;
                    }
                    Refs blockRefs; CollectRefs(blockObject["kv"], blockRefs);
                    for (const auto &ref : blockRefs) keep.insert(ref.first);
                    cursor = parent;
                }
            }
        }
        // The compacted log is a fresh log, so it needs its own monotonically
        // increasing sequence; reusing the old numbers would look like a
        // sequence regression to the recovery scanner.
        uint64_t newSequence = 0;
        const fs::path temporary = fs::path(impl->logPath.string() + ".tmp");
        {
            std::error_code ignored;
            fs::remove(temporary, ignored);
        }
        // A store with no live roots may have no OBJECT/PREPARE/COMMIT or
        // METADATA records to append.  Still create an empty replacement log;
        // otherwise the final rename fails with ENOENT and Compact reports a
        // false capacity failure.
        {
            const int fd = open(temporary.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
            Check(fd >= 0, "v4_compact_temp_create");
            Check(fdatasync(fd) == 0, "v4_compact_temp_sync");
            close(fd);
        }
        // Rewrite only reachable data. Records are read back from the live log,
        // so a torn record can never be copied into the compacted log.
        //
        // C-2 修复（2026-10-02，Route B 单测 routeBCompactKeepsReachableBlocksReclaimsUnreferenced 实测发现）：
        // 旧实现给 keep 里**每个对象各写一条** OBJECT_BATCH 记录，而单条记录按 kAlignment=4096 对齐
        // ⇒ 一个几十字节的小对象也要独占 4 KiB。实测压缩前后：360448 → 413696 B（保留 49 个对象被写成
        // 49 条记录，光记录对齐就 49×4 KiB）：未引用对象确实回收了，物理占用却反而变大，压缩等于白做。
        // 现在按正常写路径（FlushObjects / FlushObjectGroup）**同一口径攒批**：一条记录装多个对象、
        // 对象之间 8 B 对齐、累计载荷逼近 16 MiB 上限再落一条，把记录级对齐开销摊薄到可忽略。
        {
            const uint64_t budget = fastllm::prefix_v4::kMaxPackedPayloadBytes - (1ull << 20);
            std::vector<ObjectEntry> entries;
            std::string payload;
            uint64_t regionBytes = 0;
            const auto paddedStart = [](uint64_t cursor) {
                return cursor + ((cursor % 8 != 0) ? (8 - cursor % 8) : 0);
            };
            const auto flushBatch = [&]() -> bool {
                if (entries.empty()) return true;
                const std::vector<char> encoded = EncodeObjectBatchPayload(entries, payload);
                uint64_t offset = 0;
                if (!AppendRecordToFile(temporary.string(), kRecordObjectBatch, ++newSequence,
                                        std::string(encoded.data(), encoded.size()), &offset, error)) {
                    return false;
                }
                entries.clear();
                payload.clear();
                regionBytes = 0;
                return true;
            };
            for (const auto &hash : keep) {
                std::string bytes;
                if (!impl->ObjectBytes(hash, bytes, error)) return false;
                uint32_t kind = kObjectData;
                {
                    Statement query(impl->database->value, "SELECT kind FROM objects WHERE hash=?");
                    query.Text(1, hash);
                    if (query.Row()) kind = uint32_t(query.Number(0));
                }
                // 预算检查与 Impl::ProjectedBatchPayload 同口径（88 B/表项 + 8 B 对齐）。
                if (!entries.empty()) {
                    const uint64_t candidate = paddedStart(regionBytes) + bytes.size();
                    const uint64_t projected = candidate + 88ull * (entries.size() + 1) +
                                               8ull * (entries.size() + 1) + 32ull;
                    if (projected > budget && !flushBatch()) return false;
                }
                const uint64_t offset = paddedStart(regionBytes);
                if (offset > regionBytes) payload.append(size_t(offset - regionBytes), '\0');
                payload.append(bytes);
                regionBytes = offset + bytes.size();
                ObjectEntry entry;
                entry.kind = kind;
                entry.bytes = bytes.size();
                entry.offset = offset;
                entry.hash = hash;
                entries.push_back(std::move(entry));
            }
            if (!flushBatch()) return false;
        }
        // Re-publish every live root with a fresh PREPARE/COMMIT pair.
        {
            // S2-5a 缺陷修复：本循环用 ++newSequence 从 0 重排序列，循环顺序**决定压缩后
            // 的 FIFO 老化顺序**。必须按旧 sequence 升序重放（identity/key 作为稳定次序），
            // 否则老检查点可能拿到更大的新序列，压缩后淘汰顺序变任意。
            Statement roots(impl->database->value,
                            "SELECT identity,key,root FROM roots WHERE live=1 "
                            "ORDER BY sequence ASC, identity, key");
            while (roots.Row()) {
                const std::string identity = roots.Text(0);
                const std::string key = roots.Text(1);
                const std::string rootHash = roots.Text(2);
                PrepareRecord prepare;
                prepare.key = key;
                prepare.identity = identity;
                prepare.rootHash = rootHash;
                uint64_t offset = 0;
                const std::vector<char> prepareBytes = EncodePrepare(prepare);
                const std::string prepareText(prepareBytes.data(), prepareBytes.size());
                if (!AppendRecordToFile(temporary.string(), kRecordManifestPrepare, ++newSequence, prepareText,
                                        &offset, error)) {
                    return false;
                }
                prefix_v4::CommitRecord commit;
                commit.prepareOffset = offset;
                commit.prepareBytes =
                    prefix_v4::AlignUp(kRecordHeaderBytes + prepareText.size(), error);
                if (!error.empty()) return false;
                commit.prepareDigest = RecordDigest(kRecordManifestPrepare, prepareText);
                commit.rootHash = rootHash;
                const std::vector<char> commitBytes = EncodeCommit(commit);
                if (!AppendRecordToFile(temporary.string(), kRecordManifestCommit, ++newSequence,
                                        std::string(commitBytes.data(), commitBytes.size()),
                                        nullptr, error)) {
                    return false;
                }
            }
        }
        // §10.4/§11.4：压缩后的日志必须仍然"自足"——检查点描述与 kv 元数据也要
        // 以 METADATA 记录带过去，否则重建后投影为空（下方投影就丢光了）。
        {
            Statement rows(impl->database->value,
                           "SELECT c.identity,c.key,c.metadata FROM checkpoints c "
                           "JOIN roots r ON r.identity=c.identity AND r.key=c.key WHERE r.live=1");
            while (rows.Row()) {
                prefix_v4::MetadataRecord record;
                record.sequence = ++newSequence;
                record.key = "checkpoint:" + rows.Text(1);
                record.value = rows.Text(0) + "\n" + rows.Text(2);
                const std::vector<char> payload = EncodeMetadata(record);
                if (!AppendRecordToFile(temporary.string(), kRecordMetadata, newSequence,
                                        std::string(payload.data(), payload.size()), nullptr,
                                        error)) {
                    return false;
                }
            }
            Statement entries(impl->database->value, "SELECT key,value FROM kv");
            while (entries.Row()) {
                prefix_v4::MetadataRecord record;
                record.sequence = ++newSequence;
                record.key = "kv:" + entries.Text(0);
                record.value = entries.Text(1);
                const std::vector<char> payload = EncodeMetadata(record);
                if (!AppendRecordToFile(temporary.string(), kRecordMetadata, newSequence,
                                        std::string(payload.data(), payload.size()), nullptr,
                                        error)) {
                    return false;
                }
            }
        }
        // The old log stays intact until the rename succeeds.
        std::error_code renameError;
        fs::rename(temporary, impl->logPath, renameError);
        if (renameError) { error = "v4_compact_rename"; return false; }
        // §17.1（2026-09-30 修复②，故障注入实测）：rename 之后目录 fsync 失败**不得**直接
        // return，否则 RebuildIndexBareLocked 被跳过 → DB 里 offset 仍指旧日志、logPath 已是
        // 新日志 → 所有 live 检查点读不出（v4_object_mismatch / v4_object_table_region_bounds）。
        // 这里降级为"可见失败 + 重建照跑"：重建把投影指向新日志并刷新 log_bytes 水位；若掉电令
        // 目录项未持久（旧日志仍在），下次 Open 会因水位不符而自动重建（修复③）。
        bool dirSynced = true;
        try {
            SyncDirectory(impl->root);
        } catch (const std::exception &syncError) {
            dirSynced = false;
            std::fprintf(stderr, "[Prefix SSD] v4 compact: directory sync failed (%s); "
                                 "continuing with rebuild.\n", syncError.what());
        }
        const bool rebuilt = RebuildIndexBareLocked(error);   // 已持独占租约，不可再次取锁
        if (!dirSynced && rebuilt) { error = "v4_compact_dir_sync"; return false; }
        return rebuilt;
    } catch (const std::exception &exception) {
        error = exception.what();
        return false;
    }
}
} // namespace fastllm
#endif
