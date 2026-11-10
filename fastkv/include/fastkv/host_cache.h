#pragma once
#include "fastkv.h"

namespace fastkv {

// Complete immutable checkpoints, with optional reusable byte regions. The
// engine supplies representation/namespace identity; no tensor type is decoded
// here. Empty identities mean boundary-local data (e.g. a window/SSM snapshot).
struct CacheRegion {
    size_t offset = 0, bytes = 0;
    std::string identity;
};
struct CachePayload {
    // A range payload may reuse a prefix from a checkpoint actually restored
    // by this request. Boundary payloads never inherit bytes from a parent.
    bool prefixRange = false;
    std::vector<CacheRegion> regions;
};
struct CacheSnapshot {
    Message state;
    std::vector<CachePayload> payloads;
};

class HostCache {
  public:
    struct Options {
        size_t bytes = 0;
        size_t inflightBytes = size_t(4) << 30;
        size_t stagingBytes = size_t(8) << 20;
        size_t queueDepth = 16;
        unsigned workers = 2;
        // Optional physical destination limits; held through consumer release.
        std::map<int, size_t> restoreDeviceBytes;
    };
    struct Entry;
    class Hit {
      public:
        size_t Tokens() const;
        size_t Bytes() const;
        const fastkv::Bytes &Metadata() const;
        size_t Payloads() const;
        explicit operator bool() const { return bool(entry); }
      private:
        friend class HostCache;
        std::shared_ptr<const Entry> entry;
    };
    struct Statistics {
        size_t bytes = 0, entries = 0, inflightBytes = 0, queued = 0;
        uint64_t publications = 0, evictions = 0, skipped = 0, failures = 0;
        uint64_t copiedToHost = 0, reusedBytes = 0, restoredBytes = 0, restores = 0;
        uint64_t restoreMicroseconds = 0;
    };
    explicit HostCache(Options options);
    ~HostCache();
    HostCache(const HostCache &) = delete;
    HostCache &operator=(const HostCache &) = delete;
    // Metadata-only probe. Find pins a complete checkpoint; no materialization.
    size_t ProbeTokens(const std::string &name, const std::vector<int> &tokens) const;
    Hit Find(const std::string &name, const std::vector<int> &tokens) const;
    // Source buffers MUST be immutable, ready and lifetime-owned. On busy the
    // caller can skip this optional backup. Publication is atomic after copy.
    bool PublishAsync(const std::string &name, std::vector<int> tokens,
                      CacheSnapshot snapshot, Hit restoredParent = {});
    // GPU placement is provided by the adapter. Allocation/copy run off the
    // model lane. The future owns buffers through completion even if dropped.
    bool RestoreAsync(Hit hit, std::vector<int> devices, std::future<Message> &result,
                      std::shared_ptr<Cancellation> cancellation = {});
    Statistics Stats() const;
    void Drain();
  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace fastkv
