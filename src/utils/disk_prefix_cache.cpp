#include "utils/disk_prefix_cache.h"
#ifdef FASTLLM_DISK_PREFIX_CACHE
#include "../../third_party/lmcache_fs/fs/connector.h"
#include <openssl/evp.h>
#include <sqlite3.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace fastllm {
namespace {
using Json = DiskPrefixCache::Json;
namespace fs = std::filesystem;
constexpr size_t BlobHeaderBytes = 80;
constexpr size_t IndexAllowance = 64 << 10;
const char BlobMagic[8] = {'F', 'L', 'K', 'V', '0', '0', '2', '\n'};
using Refs = std::map<std::string, uint64_t>;
void Check(bool value, const char *error) { DiskPrefixCache::Require(value, error); }
uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
std::string Unique() {
    static std::atomic<uint64_t> seq{0};
    return std::to_string(getpid()) + "-" + std::to_string(NowNs()) + "-" + std::to_string(seq++);
}
struct Lease {
    int fd = -1;
    Lease(const fs::path &path, int operation) {
        fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        Check(fd >= 0, "cache_lock_open");
        int result;
        do { result = flock(fd, operation); } while (result < 0 && errno == EINTR);
        if (result < 0) { close(fd); fd = -1; throw std::runtime_error("cache_busy"); }
    }
    ~Lease() { if (fd >= 0) { flock(fd, LOCK_UN); close(fd); } }
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;
};
void PrivateDirectory(const fs::path &path) {
    std::error_code ec;
    fs::create_directories(path, ec);
    Check(!ec && !fs::is_symlink(path) && fs::is_directory(path), "cache_directory");
    Check(chmod(path.c_str(), 0700) == 0, "cache_directory_mode");
}
void SyncDirectory(const fs::path &path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    Check(fd >= 0, "cache_directory_open");
    int result = fsync(fd), saved = errno;
    close(fd); errno = saved;
    Check(result == 0, "cache_directory_fsync");
}
void SyncFile(const fs::path &path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    Check(fd >= 0, "cache_file_open");
    int result = fsync(fd), saved = errno;
    close(fd); errno = saved;
    Check(result == 0, "cache_fsync");
}
uint64_t FileSize(const fs::path &path) {
    struct stat st{};
    if (lstat(path.c_str(), &st) != 0) {
        Check(errno == ENOENT, "cache_file_stat");
        return 0;
    }
    Check(S_ISREG(st.st_mode) && st.st_size >= 0, "cache_file_type");
    return uint64_t(st.st_size);
}
std::string ReadFile(const fs::path &path, size_t maximum) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    Check(fd >= 0, "missing_cache_file");
    try {
        struct stat st{};
        Check(fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 &&
              uint64_t(st.st_size) <= maximum, "cache_file_size");
        std::string bytes(size_t(st.st_size), '\0');
        size_t offset = 0;
        while (offset < bytes.size()) {
            ssize_t count = read(fd, bytes.data() + offset, bytes.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            Check(count > 0, "cache_short_read");
            offset += size_t(count);
        }
        close(fd);
        return bytes;
    } catch (...) { close(fd); throw; }
}
void AtomicWrite(const fs::path &path, const std::string &bytes) {
    std::string pattern = path.string() + ".tmp.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end()); name.push_back(0);
    int fd = mkstemp(name.data());
    Check(fd >= 0, "cache_temp_open");
    try {
        Check(fchmod(fd, 0600) == 0, "cache_file_mode");
        size_t offset = 0;
        while (offset < bytes.size()) {
            ssize_t count = write(fd, bytes.data() + offset, bytes.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            Check(count > 0, "cache_short_write");
            offset += size_t(count);
        }
        Check(fsync(fd) == 0, "cache_fsync");
        int result = close(fd); fd = -1;
        Check(result == 0, "cache_close");
        Check(rename(name.data(), path.c_str()) == 0, "cache_rename");
        SyncDirectory(path.parent_path());
    } catch (...) { if (fd >= 0) close(fd); unlink(name.data()); throw; }
}
struct Statement {
    sqlite3_stmt *value = nullptr;
    Statement(sqlite3 *db, const char *sql) {
        Check(sqlite3_prepare_v2(db, sql, -1, &value, nullptr) == SQLITE_OK, "cache_sql_prepare");
    }
    ~Statement() { sqlite3_finalize(value); }
    void Text(int index, const std::string &text) {
        Check(sqlite3_bind_text(value, index, text.c_str(), int(text.size()), SQLITE_TRANSIENT) == SQLITE_OK,
              "cache_sql_bind");
    }
    void Number(int index, uint64_t n) {
        Check(n <= uint64_t(INT64_MAX), "cache_integer_overflow");
        Check(sqlite3_bind_int64(value, index, sqlite3_int64(n)) == SQLITE_OK, "cache_sql_bind");
    }
    bool Row() {
        int result = sqlite3_step(value);
        Check(result == SQLITE_ROW || result == SQLITE_DONE, "cache_sql_step");
        return result == SQLITE_ROW;
    }
    void Run() { Check(!Row(), "cache_sql_unexpected_row"); }
    std::string Text(int column) const {
        auto p = sqlite3_column_text(value, column);
        return p ? std::string(reinterpret_cast<const char *>(p), sqlite3_column_bytes(value, column)) : "";
    }
    uint64_t Number(int column) const { return uint64_t(sqlite3_column_int64(value, column)); }
};
struct Database {
    sqlite3 *value = nullptr;
    explicit Database(const fs::path &path) {
        int result = sqlite3_open_v2(path.c_str(), &value,
            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
        if (result != SQLITE_OK) { if (value) sqlite3_close(value); value = nullptr; }
        Check(result == SQLITE_OK, "cache_sql_open");
        sqlite3_busy_timeout(value, 10000);
        chmod(path.c_str(), 0600);
        try { Exec("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;"); }
        catch (...) { sqlite3_close(value); value = nullptr; throw; }
    }
    ~Database() { if (value) sqlite3_close(value); }
    void Exec(const char *sql) {
        Check(sqlite3_exec(value, sql, nullptr, nullptr, nullptr) == SQLITE_OK, "cache_sql_exec");
    }
    uint64_t Scalar(const char *sql) {
        Statement query(value, sql);
        return query.Row() ? query.Number(0) : 0;
    }
};
struct Transaction {
    Database &db;
    bool done = false;
    explicit Transaction(Database &db) : db(db) { db.Exec("BEGIN IMMEDIATE"); }
    void Commit() { db.Exec("COMMIT"); done = true; }
    ~Transaction() { if (!done) sqlite3_exec(db.value, "ROLLBACK", nullptr, nullptr, nullptr); }
};
uint64_t JsonBytes(const Json &value, uint64_t maximum) {
    double n = value.number_value();
    Check(value.is_number() && std::isfinite(n) && n > 0 && n <= double(maximum) &&
          std::floor(n) == n, "invalid_chunk_size");
    return uint64_t(n);
}
void CollectRefs(const Json &node, Refs &refs, int depth = 0) {
    Check(depth <= 64, "cache_metadata_depth");
    if (node.is_array()) {
        for (const auto &child : node.array_items()) CollectRefs(child, refs, depth + 1);
    } else if (node.is_object()) {
        const auto &obj = node.object_items();
        if (obj.count("sha256") && obj.count("bytes")) {
            const auto &hash = node["sha256"].string_value();
            Check(DiskPrefixCache::IsDigest(hash), "invalid_blob_digest");
            uint64_t bytes = JsonBytes(node["bytes"], DiskPrefixCache::ChunkBytes);
            auto inserted = refs.emplace(hash, bytes);
            Check(inserted.second || inserted.first->second == bytes, "conflicting_blob_size");
            return;
        }
        for (const auto &entry : obj) {
            Check(!((entry.first == "tokens" || entry.first == "token_ids" ||
                     entry.first == "input_ids") && entry.second.is_array()), "raw_tokens_in_commit");
            CollectRefs(entry.second, refs, depth + 1);
        }
    }
}
std::string FieldString(const Json &state, const char *field) {
    const auto &v = state[field];
    return v.is_null() ? "" : v.is_string() ? v.string_value() : v.dump();
}
std::string BlobKey(const std::string &hash) { return "fastllm@0@0@" + hash; }
// Exact upstream key_to_filename result for the restricted key above. Including
// upstream keys.h here would duplicate its non-inline definitions.
std::string BlobFilename(const std::string &hash) { return "fastllm@0x0@0@" + hash + ".data"; }
std::string StageFilename(const std::string &hash, const std::string &salt) {
    return "stage@0x0@0@" + hash + "@" + salt + ".data";
}
void PutLength(char *header, uint64_t length) {
    for (int i = 0; i < 8; ++i) header[8 + i] = char(length >> (i * 8));
}
uint64_t GetLength(const char *header) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= uint64_t(uint8_t(header[8 + i])) << (i * 8);
    return value;
}
void VerifyHeader(const char *header, const std::string &hash, uint64_t bytes) {
    Check(memcmp(header, BlobMagic, 8) == 0 && GetLength(header) == bytes &&
          std::string(header + 16, 64) == hash, "blob_header");
}
} // namespace

struct DiskPrefixCache::Impl {
    fs::path root, base, transport, commits, dbPath;
    std::string identity, family;
    uint64_t limit;
    std::unique_ptr<lmcache::connector::FSConnector> connector;
    std::mutex completionMutex;
    std::condition_variable completionReady;
    bool draining = false;
    std::map<uint64_t, lmcache::connector::Completion> completions;
    Impl(const std::string &directory, const std::string &identity, uint64_t limit,
         const std::string &family, int workers)
        : root(directory), base(root / "v2"), transport(base / "objects"),
          commits(base / "commits"), dbPath(base / "index.sqlite3"),
          identity(identity), family(family.empty() ? identity : family), limit(limit) {
        Check(IsDigest(identity) && IsDigest(this->family) && limit > 0 &&
              limit <= uint64_t(INT64_MAX) && workers > 0, "invalid_cache_config");
        PrivateDirectory(root);
        Lease lease(root / ".lease", LOCK_EX);
        PrivateDirectory(base); PrivateDirectory(transport); PrivateDirectory(commits);
        SyncDirectory(root); SyncDirectory(base);
        Recover();
        connector.reset(new lmcache::connector::FSConnector(transport.string(), workers, "", false));
    }
    fs::path CommitPath(const std::string &id, const std::string &key) const {
        Check(IsDigest(id) && IsDigest(key), "invalid_checkpoint_key");
        return commits / id / (key + ".commit");
    }
    fs::path BlobPath(const std::string &hash) const {
        Check(IsDigest(hash), "invalid_blob_digest");
        return transport / BlobFilename(hash);
    }
    void Schema(Database &db) {
        db.Exec("PRAGMA journal_mode=WAL;");
        db.Exec("CREATE TABLE IF NOT EXISTS objects(hash TEXT PRIMARY KEY,bytes INTEGER NOT NULL);"
                "CREATE TABLE IF NOT EXISTS checkpoints(identity TEXT NOT NULL,key TEXT NOT NULL,"
                "family TEXT NOT NULL,kind TEXT NOT NULL,length INTEGER NOT NULL,dtype TEXT NOT NULL,"
                "provenance TEXT NOT NULL,created INTEGER NOT NULL,bytes INTEGER NOT NULL,"
                "PRIMARY KEY(identity,key));"
                "CREATE TABLE IF NOT EXISTS refs(identity TEXT NOT NULL,key TEXT NOT NULL,hash TEXT NOT NULL,"
                "PRIMARY KEY(identity,key,hash),FOREIGN KEY(identity,key) REFERENCES checkpoints(identity,key) ON DELETE CASCADE);"
                "CREATE INDEX IF NOT EXISTS checkpoint_lookup ON checkpoints(identity,kind,length);"
                "CREATE INDEX IF NOT EXISTS checkpoint_family ON checkpoints(family,kind,length);"
                "CREATE INDEX IF NOT EXISTS ref_hash ON refs(hash);"
                "CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY,bytes INTEGER NOT NULL);"
                "CREATE TABLE IF NOT EXISTS intents(id TEXT PRIMARY KEY);");
    }
    Json LoadRecord(const fs::path &path, const std::string &id, const std::string &key) const {
        std::string error;
        auto envelope = Json::parse(ReadFile(path, MaxMetadataBytes), error);
        Check(error.empty() && envelope.is_object(), "invalid_manifest");
        auto body = envelope["body"].string_value();
        Check(IsDigest(envelope["sha256"].string_value()) &&
              Digest(body) == envelope["sha256"].string_value(), "manifest_checksum");
        auto record = Json::parse(body, error);
        Check(error.empty() && record["version"].int_value() == Version &&
              record["identity"].string_value() == id && record["key"].string_value() == key &&
              record["state"].is_object() && IsDigest(record["family"].string_value()), "manifest_identity_or_version");
        Refs actual, declared;
        CollectRefs(record["state"], actual); CollectRefs(record["objects"], declared);
        Check(record["objects"].is_array() && actual == declared, "manifest_object_references");
        auto kind = record["kind"].string_value();
        Check(kind == "prefix" || kind == "encoder", "invalid_checkpoint_kind");
        Check(kind != "prefix" || record["state"]["length"].int_value() > 0, "invalid_checkpoint_length");
        const auto &stamp = record["created_ns"].string_value();
        size_t end = 0;
        auto created = std::stoull(stamp, &end);
        Check(end == stamp.size() && created <= uint64_t(INT64_MAX), "invalid_checkpoint_time");
        return record;
    }
    bool HeaderValid(const std::string &hash, uint64_t bytes) const {
        try {
            auto path = BlobPath(hash);
            Check(FileSize(path) == bytes + BlobHeaderBytes, "blob_size");
            int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
            Check(fd >= 0, "missing_blob");
            char header[BlobHeaderBytes]; size_t offset = 0;
            while (offset < BlobHeaderBytes) {
                ssize_t n = read(fd, header + offset, BlobHeaderBytes - offset);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) { close(fd); throw std::runtime_error("blob_header_read"); }
                offset += size_t(n);
            }
            close(fd); VerifyHeader(header, hash, bytes);
            return true;
        } catch (...) { return false; }
    }
    void IndexRecord(Database &db, const Json &record, uint64_t bytes) {
        Refs refs; CollectRefs(record["objects"], refs);
        Transaction transaction(db);
        Statement put(db.value, "INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?)");
        const auto &value = record["state"];
        put.Text(1, record["identity"].string_value()); put.Text(2, record["key"].string_value());
        put.Text(3, record["family"].string_value()); put.Text(4, record["kind"].string_value());
        put.Number(5, uint64_t(std::max(0, value["length"].int_value())));
        put.Text(6, FieldString(value, "kv_dtype")); put.Text(7, FieldString(value, "provenance"));
        put.Number(8, std::stoull(record["created_ns"].string_value())); put.Number(9, bytes); put.Run();
        for (const auto &ref : refs) {
            Statement object(db.value, "INSERT OR REPLACE INTO objects VALUES(?,?)");
            object.Text(1, ref.first); object.Number(2, ref.second + BlobHeaderBytes); object.Run();
            Statement link(db.value, "INSERT INTO refs VALUES(?,?,?)");
            link.Text(1, record["identity"].string_value()); link.Text(2, record["key"].string_value());
            link.Text(3, ref.first); link.Run();
        }
        transaction.Commit();
    }
    uint64_t Used(Database &db) const {
        uint64_t used = db.Scalar("SELECT COALESCE(SUM(bytes),0) FROM objects") +
                        db.Scalar("SELECT COALESCE(SUM(bytes),0) FROM checkpoints") +
                        db.Scalar("SELECT COALESCE(SUM(bytes),0) FROM reservations");
        used += FileSize(dbPath) + FileSize(dbPath.string() + "-wal") + FileSize(dbPath.string() + "-shm");
        return used;
    }
    void Intent(Database &db, const std::string &id, bool add) {
        Statement query(db.value, add ? "INSERT OR IGNORE INTO intents VALUES(?)" : "DELETE FROM intents WHERE id=?");
        query.Text(1, id); query.Run();
    }
    void Recover() {
        // Caller owns exclusive lifecycle lease, so every uncommitted object is
        // unreachable and no live writer's temporary file can be collected.
        Lease metadata(root / ".metadata", LOCK_EX);
        std::unique_ptr<Database> db;
        try {
            db.reset(new Database(dbPath));
            Statement query(db->value, "PRAGMA quick_check");
            Check(query.Row() && query.Text(0) == "ok", "cache_index_corrupt");
        } catch (...) {
            db.reset();
            fs::remove(dbPath); fs::remove(dbPath.string() + "-wal"); fs::remove(dbPath.string() + "-shm");
            db.reset(new Database(dbPath));
        }
        Schema(*db);
        db->Exec("DELETE FROM refs; DELETE FROM checkpoints; DELETE FROM objects; DELETE FROM reservations; DELETE FROM intents;");
        std::set<std::string> reachable;
        for (const auto &directory : fs::directory_iterator(commits)) {
            if (!directory.is_directory() || directory.is_symlink() || !IsDigest(directory.path().filename().string())) continue;
            auto id = directory.path().filename().string();
            for (const auto &file : fs::directory_iterator(directory.path())) {
                if (!file.is_regular_file() || file.is_symlink()) continue;
                if (file.path().extension() != ".commit") { fs::remove(file.path()); continue; }
                auto key = file.path().stem().string();
                Json record;
                Refs refs;
                try {
                    record = LoadRecord(file.path(), id, key);
                    CollectRefs(record["objects"], refs);
                    for (const auto &ref : refs) Check(HeaderValid(ref.first, ref.second), "incomplete_checkpoint");
                } catch (...) { fs::remove(file.path()); continue; }
                // SQL/storage failures abort maintenance, never delete an
                // otherwise valid commit merely because indexing failed.
                IndexRecord(*db, record, FileSize(file.path()));
                for (const auto &ref : refs) reachable.insert(BlobFilename(ref.first));
            }
            SyncDirectory(directory.path());
        }
        for (const auto &file : fs::directory_iterator(transport)) {
            if (file.is_regular_file() && !file.is_symlink() && !reachable.count(file.path().filename().string())) fs::remove(file.path());
        }
        SyncDirectory(transport); SyncDirectory(commits);
        db->Exec("PRAGMA wal_checkpoint(TRUNCATE)");
    }
    bool Maintain(uint64_t required) {
        try {
            Lease lease(root / ".lease", LOCK_EX | LOCK_NB);
            Recover();
            Lease metadata(root / ".metadata", LOCK_EX);
            Database db(dbPath);
            while (true) {
                db.Exec("PRAGMA wal_checkpoint(TRUNCATE)");
                uint64_t used = Used(db);
                if (used <= limit && required <= limit - used) return true;
                std::string id, key;
                { Statement oldest(db.value, "SELECT identity,key FROM checkpoints ORDER BY created,identity,key LIMIT 1");
                  if (!oldest.Row()) return false;
                  id = oldest.Text(0); key = oldest.Text(1); }
                // Withdraw the source of truth first. A crash can leak objects,
                // but cannot leave a published checkpoint pointing at GC'd data.
                fs::remove(CommitPath(id, key)); SyncDirectory(commits / id);
                { Statement erase(db.value, "DELETE FROM checkpoints WHERE identity=? AND key=?");
                  erase.Text(1, id); erase.Text(2, key); erase.Run(); }
                std::vector<std::string> unused;
                { Statement query(db.value, "SELECT hash FROM objects WHERE NOT EXISTS(SELECT 1 FROM refs WHERE refs.hash=objects.hash)");
                  while (query.Row()) unused.push_back(query.Text(0)); }
                for (const auto &hash : unused) {
                    fs::remove(BlobPath(hash));
                    Statement erase(db.value, "DELETE FROM objects WHERE hash=?"); erase.Text(1, hash); erase.Run();
                }
                SyncDirectory(transport);
            }
        } catch (const std::exception &) { return false; }
    }
    lmcache::connector::Completion Wait(uint64_t future) {
        std::unique_lock<std::mutex> lock(completionMutex);
        for (;;) {
            auto ready = completions.find(future);
            if (ready != completions.end()) {
                auto result = std::move(ready->second); completions.erase(ready); return result;
            }
            if (draining) { completionReady.wait(lock); continue; }
            draining = true; lock.unlock();
            struct pollfd fd{connector->event_fd(), POLLIN, 0};
            int result;
            do { result = poll(&fd, 1, -1); } while (result < 0 && errno == EINTR);
            auto completed = connector->drain_completions();
            lock.lock();
            for (auto &item : completed) completions.emplace(item.future_id, std::move(item));
            draining = false; completionReady.notify_all();
            // A live connector owns the descriptor. Never release caller buffers
            // on timeout/cancel: wait until its completion has actually arrived.
            if (result < 0) {
                lock.unlock();
                std::this_thread::yield();
                lock.lock();
            }
        }
    }
    void Transfer(bool write, const std::string &key, std::vector<char> &buffer) {
        std::vector<std::string> keys{key}; std::vector<void *> buffers{buffer.data()};
        std::vector<size_t> lengths{buffer.size()};
        auto future = write ? connector->submit_batch_set(keys, buffers, lengths, buffer.size()) :
                              connector->submit_batch_get(keys, buffers, lengths, buffer.size());
        auto result = Wait(future);
        Check(result.ok, "cache_backend_io");
        // GET records individual failures even when batch.ok remains true.
        Check(write || (result.result_bytes.size() == 1 && result.result_bytes[0] == 1), "cache_backend_read");
    }
    std::vector<char> ReadBlob(const std::string &hash, uint64_t bytes) {
        Check(bytes > 0 && bytes <= ChunkBytes && FileSize(BlobPath(hash)) == bytes + BlobHeaderBytes, "blob_size");
        std::vector<char> buffer(size_t(bytes + BlobHeaderBytes));
        Transfer(false, BlobKey(hash), buffer);
        VerifyHeader(buffer.data(), hash, bytes);
        Check(Digest(buffer.data() + BlobHeaderBytes, size_t(bytes)) == hash, "blob_checksum");
        return buffer;
    }
    DiskPrefixCache::Checkpoint Candidate(const Json &record) const {
        DiskPrefixCache::Checkpoint item;
        item.key = record["key"].string_value(); item.identity = record["identity"].string_value();
        item.familyIdentity = record["family"].string_value(); item.kind = record["kind"].string_value();
        const auto &value = record["state"];
        item.length = value["length"].int_value(); item.kvDtype = FieldString(value, "kv_dtype");
        item.provenance = FieldString(value, "provenance");
        item.createdNs = std::stoull(record["created_ns"].string_value()); item.metadata = value;
        return item;
    }
    std::vector<DiskPrefixCache::Checkpoint> Scan(const std::string &exact, const std::string &family,
                                                int length, const std::string &kind) const {
        std::vector<DiskPrefixCache::Checkpoint> result;
        for (const auto &directory : fs::directory_iterator(commits)) {
            auto id = directory.path().filename().string();
            if (!IsDigest(id) || (!exact.empty() && id != exact) ||
                !directory.is_directory() || directory.is_symlink()) continue;
            for (const auto &file : fs::directory_iterator(directory.path())) {
                if (!file.is_regular_file() || file.is_symlink() || file.path().extension() != ".commit") continue;
                try {
                    auto candidate = Candidate(LoadRecord(file.path(), id, file.path().stem().string()));
                    if (candidate.kind == kind && candidate.length <= length &&
                        (family.empty() || candidate.familyIdentity == family)) result.push_back(std::move(candidate));
                } catch (...) {} // A corrupt commit is a cache miss.
            }
        }
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return a.length != b.length ? a.length > b.length : a.createdNs < b.createdNs;
        });
        return result;
    }
};

struct DiskPrefixCache::WriteSession::State {
    std::shared_ptr<Impl> owner;
    std::unique_ptr<Lease> lease;
    std::string id = Unique();
    uint64_t remaining = 0, written = 0;
    std::vector<fs::path> temporary;
    Refs verified;
    explicit State(std::shared_ptr<Impl> owner) : owner(std::move(owner)) {}
    ~State() {
        // StoreBytes returns only after every submitted LMCache transfer ends.
        for (const auto &path : temporary) {
            std::error_code ignored; fs::remove(path, ignored);
            auto tmp = path; tmp.replace_extension(".tmp"); fs::remove(tmp, ignored);
        }
        if (lease) {
            try {
                Lease metadata(owner->root / ".metadata", LOCK_EX);
                Database db(owner->dbPath);
                Statement erase(db.value, "DELETE FROM reservations WHERE id=?"); erase.Text(1, id); erase.Run();
            } catch (...) {}
        }
    }
    void Reserve(Database &db, uint64_t required) {
        if (remaining >= required) return;
        uint64_t extra = required - remaining, used = owner->Used(db);
        Check(used <= owner->limit && extra <= owner->limit - used, "disk_quota");
        remaining += extra;
        Statement query(db.value, "INSERT OR REPLACE INTO reservations VALUES(?,?)");
        query.Text(1, id); query.Number(2, remaining); query.Run();
    }
    void Consume(Database &db, uint64_t bytes) {
        Check(bytes <= remaining, "cache_reservation_underflow");
        remaining -= bytes;
        Statement query(db.value, "UPDATE reservations SET bytes=? WHERE id=?");
        query.Number(1, remaining); query.Text(2, id); query.Run();
    }
};
struct DiskPrefixCache::ReadSession::State {
    std::shared_ptr<Impl> owner;
    Lease lease;
    uint64_t read = 0;
    Refs allowed;
    explicit State(std::shared_ptr<Impl> owner)
        : owner(std::move(owner)), lease(this->owner->root / ".lease", LOCK_SH) {}
};

void DiskPrefixCache::Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
std::string DiskPrefixCache::Digest(const void *data, size_t size) {
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned int count = 0;
    Check(EVP_Digest(data, size, bytes, &count, EVP_sha256(), nullptr) == 1 && count == 32, "sha256_failed");
    static const char hex[] = "0123456789abcdef";
    std::string result; result.reserve(64);
    for (unsigned int i = 0; i < count; ++i) { result += hex[bytes[i] >> 4]; result += hex[bytes[i] & 15]; }
    return result;
}
std::string DiskPrefixCache::Digest(const std::string &value) { return Digest(value.data(), value.size()); }
bool DiskPrefixCache::IsDigest(const std::string &value) {
    return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::string DiskPrefixCache::PrefixKey(const std::vector<int> &tokens, int length, const std::string &media) {
    Check(length > 0 && size_t(length) <= tokens.size(), "invalid_prefix_length");
    std::string bytes = "fastllm-prefix-v2";
    for (int i = 0; i < 8; ++i) bytes.push_back(char(uint64_t(media.size()) >> (i * 8)));
    bytes += media;
    for (int j = 0; j < length; ++j) {
        uint32_t token = uint32_t(tokens[j]);
        for (int i = 0; i < 4; ++i) bytes.push_back(char(token >> (i * 8)));
    }
    return Digest(bytes);
}
DiskPrefixCache::DiskPrefixCache(const std::string &directory, const std::string &identity,
        uint64_t limitBytes, const std::string &familyIdentity, int ioThreads)
    : impl(std::make_shared<Impl>(directory, identity, limitBytes, familyIdentity, ioThreads)) {}
DiskPrefixCache::~DiskPrefixCache() = default;
DiskPrefixCache::WriteSession::WriteSession(std::unique_ptr<State> state) : state(std::move(state)) {}
DiskPrefixCache::WriteSession::WriteSession(WriteSession &&) noexcept = default;
DiskPrefixCache::WriteSession &DiskPrefixCache::WriteSession::operator=(WriteSession &&) noexcept = default;
DiskPrefixCache::WriteSession::~WriteSession() = default;
DiskPrefixCache::ReadSession::ReadSession(std::unique_ptr<State> state) : state(std::move(state)) {}
DiskPrefixCache::ReadSession::ReadSession(ReadSession &&) noexcept = default;
DiskPrefixCache::ReadSession &DiskPrefixCache::ReadSession::operator=(ReadSession &&) noexcept = default;
DiskPrefixCache::ReadSession::~ReadSession() = default;

DiskPrefixCache::WriteSession DiskPrefixCache::BeginWrite(uint64_t reservationBytes) {
    Check(reservationBytes <= impl->limit, "disk_quota");
    auto state = std::make_unique<WriteSession::State>(impl);
    for (int attempt = 0; attempt < 2; ++attempt) {
        state->lease.reset(new Lease(impl->root / ".lease", LOCK_SH));
        try {
            Lease metadata(impl->root / ".metadata", LOCK_EX);
            Database db(impl->dbPath);
            Check(db.Scalar("SELECT COUNT(*) FROM intents") == 0, "cache_index_needs_rebuild");
            state->Reserve(db, std::max<uint64_t>(reservationBytes, IndexAllowance));
            return WriteSession(std::move(state));
        } catch (...) {
            state->lease.reset();
            if (attempt || !impl->Maintain(std::max<uint64_t>(reservationBytes, IndexAllowance))) throw;
        }
    }
    throw std::runtime_error("disk_quota");
}
DiskPrefixCache::ReadSession DiskPrefixCache::BeginRead() const {
    return ReadSession(std::make_unique<ReadSession::State>(impl));
}
Json::array DiskPrefixCache::WriteSession::StoreBytes(size_t bytes,
        const std::function<void(size_t, char *, size_t)> &read, const Json &previous) {
    Check(bool(state), "closed_write_session");
    (void)previous; // A mutable GDN/draft buffer is never reused by position.
    Json::array chunks;
    for (size_t offset = 0; offset < bytes;) {
        size_t count = std::min(ChunkBytes, bytes - offset);
        std::vector<char> buffer(count + BlobHeaderBytes);
        read(offset, buffer.data() + BlobHeaderBytes, count);
        auto hash = Digest(buffer.data() + BlobHeaderBytes, count);
        // Bounded striped locks protect duplicate writers across processes,
        // without creating one lock file per cached object.
        Lease object(state->owner->root / (".object-" + hash.substr(0, 2)), LOCK_EX);
        bool valid = false;
        try { state->owner->ReadBlob(hash, count); valid = true; } catch (...) {}
        if (!valid) {
            memcpy(buffer.data(), BlobMagic, 8); PutLength(buffer.data(), count);
            memcpy(buffer.data() + 16, hash.data(), 64);
            auto salt = Unique();
            auto stage = state->owner->transport / StageFilename(hash, salt);
            state->temporary.push_back(stage);
            {
                Lease metadata(state->owner->root / ".metadata", LOCK_EX);
                Database db(state->owner->dbPath);
                state->Reserve(db, buffer.size() + IndexAllowance);
            }
            state->owner->Transfer(true, "stage@0@0@" + hash + "@" + salt, buffer);
            Check(FileSize(stage) == buffer.size(), "cache_staging_size");
            Check(chmod(stage.c_str(), 0600) == 0, "cache_file_mode");
            SyncFile(stage);
            {
                Lease metadata(state->owner->root / ".metadata", LOCK_EX);
                Database db(state->owner->dbPath);
                const auto intent = state->id + ":blob:" + hash;
                state->owner->Intent(db, intent, true);
                Check(rename(stage.c_str(), state->owner->BlobPath(hash).c_str()) == 0, "cache_rename");
                SyncDirectory(state->owner->transport);
                Transaction transaction(db);
                Statement put(db.value, "INSERT OR REPLACE INTO objects VALUES(?,?)");
                put.Text(1, hash); put.Number(2, buffer.size()); put.Run();
                state->Consume(db, buffer.size());
                state->owner->Intent(db, intent, false);
                transaction.Commit();
            }
            state->written += buffer.size();
        }
        state->verified[hash] = count;
        chunks.emplace_back(Json::object{{"sha256", hash}, {"bytes", double(count)}});
        offset += count;
    }
    return chunks;
}
void DiskPrefixCache::WriteSession::Commit(const std::string &key, const Json &value) {
    Check(state && IsDigest(key) && value.is_object(), "invalid_checkpoint_commit");
    auto owner = state->owner;
    std::string kind = value["kind"].is_null() ? "prefix" : value["kind"].string_value();
    std::string family = value["family_identity"].is_null() ? owner->family : value["family_identity"].string_value();
    Check((kind == "prefix" || kind == "encoder") && IsDigest(family), "invalid_checkpoint_metadata");
    Check(kind != "prefix" || (value["length"].is_number() && value["length"].int_value() > 0 &&
          value["length"].number_value() == value["length"].int_value()), "invalid_checkpoint_length");
    Refs refs; CollectRefs(value, refs);
    for (const auto &ref : refs) {
        auto found = state->verified.find(ref.first);
        if (found == state->verified.end() || found->second != ref.second) owner->ReadBlob(ref.first, ref.second);
    }
    auto path = owner->CommitPath(owner->identity, key);
    // Serialize same-key publishers without holding the database lock during
    // payload verification. A healthy duplicate never refreshes FIFO age.
    Lease publication(owner->root / (".commit-" + key.substr(0, 2)), LOCK_EX);
    Json existing;
    bool healthy = false;
    try {
        existing = owner->LoadRecord(path, owner->identity, key);
        Refs oldRefs; CollectRefs(existing["objects"], oldRefs);
        for (const auto &ref : oldRefs) owner->ReadBlob(ref.first, ref.second);
        healthy = true;
    } catch (const std::exception &) {}
    if (healthy) {
        Check(existing["kind"].string_value() == kind && existing["family"].string_value() == family &&
              existing["state"] == value, "checkpoint_key_conflict");
        return;
    }
    Json::array objects;
    for (const auto &ref : refs) objects.emplace_back(Json::object{{"sha256", ref.first}, {"bytes", double(ref.second)}});
    Json record = Json::object{{"version", Version}, {"identity", owner->identity}, {"family", family},
        {"kind", kind}, {"key", key}, {"created_ns", std::to_string(NowNs())}, {"objects", objects}, {"state", value}};
    auto body = record.dump();
    auto envelope = Json(Json::object{{"sha256", Digest(body)}, {"body", body}}).dump();
    Check(envelope.size() <= MaxMetadataBytes, "manifest_too_large");
    Lease metadata(owner->root / ".metadata", LOCK_EX);
    Database db(owner->dbPath);
    state->Reserve(db, envelope.size() + IndexAllowance + refs.size() * 512);
    PrivateDirectory(path.parent_path()); SyncDirectory(owner->commits);
    auto intent = state->id + ":commit:" + key;
    owner->Intent(db, intent, true);
    AtomicWrite(path, envelope);
    owner->IndexRecord(db, record, envelope.size());
    state->Consume(db, envelope.size());
    owner->Intent(db, intent, false);
    state->written += envelope.size();
}
uint64_t DiskPrefixCache::WriteSession::WrittenBytes() const { return state ? state->written : 0; }
Json DiskPrefixCache::ReadSession::Load(const std::string &key, const std::string &sourceIdentity) {
    Check(bool(state), "closed_read_session");
    auto id = sourceIdentity.empty() ? state->owner->identity : sourceIdentity;
    auto record = state->owner->LoadRecord(state->owner->CommitPath(id, key), id, key);
    Refs refs; CollectRefs(record["objects"], refs);
    for (const auto &ref : refs) Check(state->owner->HeaderValid(ref.first, ref.second), "incomplete_checkpoint");
    state->allowed.insert(refs.begin(), refs.end());
    return record["state"];
}
void DiskPrefixCache::ReadSession::ReadBytes(const Json &chunks, size_t expected,
        const std::function<void(size_t, const char *, size_t)> &write) {
    Check(state && chunks.is_array(), "invalid_chunk_manifest");
    size_t offset = 0;
    for (const auto &chunk : chunks.array_items()) {
        const auto &hash = chunk["sha256"].string_value();
        uint64_t bytes = JsonBytes(chunk["bytes"], ChunkBytes);
        auto allowed = state->allowed.find(hash);
        Check(IsDigest(hash) && allowed != state->allowed.end() && allowed->second == bytes &&
              offset <= expected && bytes <= expected - offset, "invalid_chunk_reference");
        auto buffer = state->owner->ReadBlob(hash, bytes);
        state->read += buffer.size();
        write(offset, buffer.data() + BlobHeaderBytes, size_t(bytes));
        offset += size_t(bytes);
    }
    Check(offset == expected, "incomplete_tensor");
}
uint64_t DiskPrefixCache::ReadSession::ReadBytesCount() const { return state ? state->read : 0; }
std::vector<DiskPrefixCache::Checkpoint> DiskPrefixCache::ListCheckpoints(const std::string &exactIdentity,
        const std::string &familyIdentity, int maxLength, const std::string &kind) const {
    Check((exactIdentity.empty() || IsDigest(exactIdentity)) &&
          (familyIdentity.empty() || IsDigest(familyIdentity)) &&
          (kind == "prefix" || kind == "encoder") && maxLength >= 0, "invalid_checkpoint_query");
    Lease lease(impl->root / ".lease", LOCK_SH);
    Lease metadata(impl->root / ".metadata", LOCK_EX);
    try {
        Database db(impl->dbPath);
        Check(db.Scalar("SELECT COUNT(*) FROM intents") == 0, "cache_index_needs_rebuild");
        Statement query(db.value, "SELECT key,identity,family,kind,length,dtype,provenance,created FROM checkpoints "
            "WHERE (?='' OR identity=?) AND (?='' OR family=?) AND kind=? AND length<=? ORDER BY length DESC,created ASC");
        query.Text(1, exactIdentity); query.Text(2, exactIdentity); query.Text(3, familyIdentity); query.Text(4, familyIdentity);
        query.Text(5, kind); query.Number(6, uint64_t(maxLength));
        std::vector<Checkpoint> result;
        while (query.Row()) {
            Checkpoint item;
            item.key = query.Text(0); item.identity = query.Text(1); item.familyIdentity = query.Text(2);
            item.kind = query.Text(3); item.length = int(query.Number(4)); item.kvDtype = query.Text(5);
            item.provenance = query.Text(6); item.createdNs = query.Number(7);
            item.metadata = Json::object{{"kind", item.kind}, {"length", item.length},
                {"kv_dtype", item.kvDtype}, {"family_identity", item.familyIdentity}, {"provenance", item.provenance}};
            result.push_back(std::move(item));
        }
        return result;
    } catch (...) {
        // Exact deterministic commit files remain usable without any index.
        return impl->Scan(exactIdentity, familyIdentity, maxLength, kind);
    }
}
void DiskPrefixCache::RebuildIndex() {
    Lease lease(impl->root / ".lease", LOCK_EX);
    impl->Recover();
}
fs::path DiskPrefixCache::Directory() const { return impl->base; }
} // namespace fastllm
#endif
