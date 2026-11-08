#ifndef FASTLLM_DISK_PREFIX_V5_H
#define FASTLLM_DISK_PREFIX_V5_H

// SSD prefix cache V5 (logs/a6-v5-20261107/DESIGN-V5-draft3.md).
// One append-only file per conversation. A file holds the KV bundles of its
// conversation in order; a branch is a new file that points at its parent
// (parent incarnation + number of inherited KV bundles), so prefixes are never
// copied. Prompt-end states live in two ping-pong slots per file; every other
// state is appended. Delete = unlink. The index is derived from the files.
// Default store of DiskPrefixCache; FASTLLM_PREFIX_V5=0 selects the V4 store (rollback).

#include "json11.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {

class DiskPrefixV5Store {
public:
    using Json = json11::Json;
    struct Impl;
    // (sha256 hex digest, payload) pairs staged by one write session.
    using Staged = std::vector<std::pair<std::string, std::string>>;
    using EraseObserver = std::function<void(const std::string &key, const std::string &identity)>;
    using Predicate = std::function<bool(const std::string &key, const std::string &identity)>;
    struct Row {
        std::string key, identity, familyIdentity, kind, kvDtype, provenance, metadata;
        int length = 0;
        uint64_t createdNs = 0;
    };
    // One per write session (one request). Holds the writer lease of the file
    // its checkpoints were appended to (draft3 sec. 4.2).
    class Writer {
    public:
        ~Writer();
        Writer(const Writer &) = delete;
        Writer &operator=(const Writer &) = delete;
    private:
        friend class DiskPrefixV5Store;
        Writer(std::weak_ptr<Impl> owner, uint64_t id);
        std::weak_ptr<Impl> owner;
        uint64_t id = 0;
    };

    // One store per canonical root per process (the target and encoder
    // namespaces share it, like V4 OpenShared). Takes an exclusive flock on
    // <directory>/v5/LOCK; if another process holds it, returns nullptr.
    static std::shared_ptr<DiskPrefixV5Store> OpenShared(const std::string &directory,
                                                         uint64_t quotaBytes, std::string &error);
    ~DiskPrefixV5Store();
    DiskPrefixV5Store(const DiskPrefixV5Store &) = delete;
    DiskPrefixV5Store &operator=(const DiskPrefixV5Store &) = delete;

    std::string Root() const;
    std::string Settings() const;   // effective tunables, for the startup log
    std::shared_ptr<Writer> BeginWriter();
    // Accepts the checkpoint for the single background writer; completion runs
    // once with durable=true only after the last fdatasync of draft3 sec. 4.
    bool Submit(const std::shared_ptr<Writer> &writer, const std::string &key,
                const std::string &identity, const std::string &family, Json state,
                Staged &&staged, std::function<void(bool)> completion, std::string &error);
    // Bounded drain (draft3 G3): waits up to FASTLLM_PREFIX_V5_DRAIN_MS for the
    // queue, then abandons queued saves (never partial). False if any save was
    // abandoned or failed with an I/O error since the previous drain.
    bool Drain(std::string &error);
    bool CancelOldestPending(uint64_t *freedBytes, std::string &error);
    uint64_t InFlightBytes() const;

    // manifestOnly=false reads and crc32c-verifies every referenced blob into
    // *objects before returning (all ranks before any page is published, C7).
    // The pin keeps the file chain (and the slot, if any) alive and unchanged.
    // (verified-byte delta, discovered-byte delta); false cancels at a blob boundary.
    using ReadProgress = std::function<bool(uint64_t, uint64_t)>;
    bool Load(const std::string &key, const std::string &identity, bool manifestOnly, Json &value,
              std::map<std::string, std::string> *objects, std::shared_ptr<void> &pin,
              std::string &error, const ReadProgress &progress = {});
    std::vector<int> CheckpointLengths(const std::string &family, int maximum,
                                       const std::string &kind) const;
    std::vector<Row> ListCheckpoints(const std::string &exact, const std::string &family, int maximum,
                                     const std::string &kind, const std::string &prefix) const;
    std::vector<Row> MatchPrefixes(const std::string &family,
                                   const std::vector<std::pair<int, std::string>> &prefixes,
                                   const std::string &kind) const;
    // Junction (draft3 sec. 4 step 4, H24/H25): number of leading pages of `pages`
    // (chain hashes, so a match at page p implies pages 0..p match) shared with any
    // stored checkpoint of `identity`.
    int JunctionPages(const std::string &identity, const std::vector<std::string> &pages) const;
    bool PutMetadata(const std::string &key, const std::string &value, std::string &error);
    bool GetMetadata(const std::string &key, std::string &value, std::string &error) const;
    bool RetireCorruptCheckpoint(const std::string &key, const std::string &identity,
                                 const std::string &reason, std::string &error);
    void SetEraseObserver(EraseObserver observer);
    void SetReclaimPredicate(Predicate predicate);
    void SetCorruptRetirePredicate(Predicate predicate);
    Json Capacity() const;
    Json Profile() const;
    bool SelfCheck(std::string &details) const;
    uint64_t LiveCheckpoints() const;
    uint64_t PhysicalBytes() const;

    // Test hooks (always compiled, inert unless set). step names: data_write,
    // data_sync, header_write, header_sync, dir_sync, slot_write, slot_sync,
    // slot_header_write, slot_header_sync. mode: "crash" (stop all further disk
    // writes, as a power cut), "torn" (write half of that step, then crash),
    // "eio" (that write fails; the store must roll back and continue).
    static void SetFailpointForTest(const std::string &step, const std::string &mode);

private:
    explicit DiskPrefixV5Store(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl;
};

bool DiskPrefixV5Enabled();

} // namespace fastllm

#endif
