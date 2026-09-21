#ifndef FASTLLM_DISK_PREFIX_CACHE_H
#define FASTLLM_DISK_PREFIX_CACHE_H

#ifdef FASTLLM_DISK_PREFIX_CACHE
#include "json11.hpp"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fastllm {
// Immutable checksummed objects and complete atomic checkpoints. SQLite is an
// expendable directory; commit files are the source of truth. Payload transport
// uses the unmodified Apache-2.0 LMCache native filesystem connector.
class DiskPrefixCache {
    struct Impl;
public:
    using Json = json11::Json;
    static constexpr int Version = 2;
    static constexpr size_t ChunkBytes = 8 << 20;
    static constexpr size_t MaxMetadataBytes = 64 << 20;
    struct Checkpoint {
        std::string key, identity, familyIdentity, kind, kvDtype, provenance;
        int length = 0;
        uint64_t createdNs = 0;
        Json metadata;
    };
    class WriteSession {
        struct State;
        std::unique_ptr<State> state;
        explicit WriteSession(std::unique_ptr<State> state);
        friend class DiskPrefixCache;
    public:
        WriteSession(WriteSession &&) noexcept;
        WriteSession &operator=(WriteSession &&) noexcept;
        ~WriteSession();
        WriteSession(const WriteSession &) = delete;
        WriteSession &operator=(const WriteSession &) = delete;
        // At most ChunkBytes per callback. A previous descriptor is only a
        // hint: its position never proves unchanged contents.
        Json::array StoreBytes(size_t bytes,
            const std::function<void(size_t, char *, size_t)> &read,
            const Json &previous = Json());
        // State supplies kind, length, family_identity, kv_dtype, provenance.
        // All nested {sha256,bytes} references join the complete commit,
        // including shared KV objects stored by earlier sessions.
        void Commit(const std::string &key, const Json &state);
        uint64_t WrittenBytes() const;
    };
    class ReadSession {
        struct State;
        std::unique_ptr<State> state;
        explicit ReadSession(std::unique_ptr<State> state);
        friend class DiskPrefixCache;
    public:
        ReadSession(ReadSession &&) noexcept;
        ReadSession &operator=(ReadSession &&) noexcept;
        ~ReadSession();
        ReadSession(const ReadSession &) = delete;
        ReadSession &operator=(const ReadSession &) = delete;
        Json Load(const std::string &key, const std::string &sourceIdentity = "");
        // Each chunk is verified before callback. The caller imports into
        // unpublished state and rolls back if a subsequent chunk fails.
        void ReadBytes(const Json &chunks, size_t expected,
            const std::function<void(size_t, const char *, size_t)> &write);
        uint64_t ReadBytesCount() const;
    };
    DiskPrefixCache(const std::string &directory, const std::string &identity,
                    uint64_t limitBytes = uint64_t(256) << 30,
                    const std::string &familyIdentity = "", int ioThreads = 4);
    ~DiskPrefixCache();
    DiskPrefixCache(const DiskPrefixCache &) = delete;
    DiskPrefixCache &operator=(const DiskPrefixCache &) = delete;
    WriteSession BeginWrite(uint64_t reservationBytes = 0);
    ReadSession BeginRead() const;
    // Exact lookup: exactIdentity. Family lookup: empty exactIdentity + family.
    std::vector<Checkpoint> ListCheckpoints(const std::string &exactIdentity = "",
        const std::string &familyIdentity = "", int maxLength = INT_MAX,
        const std::string &kind = "prefix") const;
    // Calling thread must not hold sessions. Reconstructs references from
    // commits and reclaims interrupted writes and unreachable objects.
    void RebuildIndex();
    std::filesystem::path Directory() const;
    static void Require(bool condition, const char *message);
    static std::string Digest(const void *data, size_t size);
    static std::string Digest(const std::string &value);
    static bool IsDigest(const std::string &value);
    static std::string PrefixKey(const std::vector<int> &tokens, int length,
                                 const std::string &media = "");
private:
    std::shared_ptr<Impl> impl;
};
} // namespace fastllm
#endif
#endif
