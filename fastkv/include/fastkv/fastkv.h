#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastkv {

using Bytes = std::vector<uint8_t>;

// The control protocol has explicit byte order and bounds. Never serialize a
// C++ object, native pointer, size_t, or enum representation directly.
class Writer {
  public:
    Bytes bytes;
    void U32(uint32_t v);
    void U64(uint64_t v);
    void String(const std::string &s);
    void Blob(const void *p, size_t n);
    void Ints(const std::vector<int> &v);
};
class Reader {
  public:
    explicit Reader(const Bytes &bytes);
    Reader(const void *data, size_t size);
    uint32_t U32();
    uint64_t U64();
    std::string String(size_t limit = 1 << 20);
    Bytes Blob(size_t limit = 64 << 20);
    std::vector<int> Ints(size_t limit = 1 << 22);
    void End() const;
    size_t Remaining() const { return size - position; }

  private:
    const uint8_t *data;
    size_t size, position = 0;
    const uint8_t *Take(size_t n);
};

enum class MemoryKind : uint32_t { Host = 0, Cuda = 1 };
enum class Transport : uint32_t { Tcp = 0, Rdma = 1, Auto = 2 };

// A Buffer owns (or leases) an allocation; slices retain the same allocation.
// Registration attachments die before the allocation itself. Allocation IDs
// never repeat within a process, even when a device address is recycled.
class Buffer {
  public:
    struct Storage;
    Buffer() = default;
    static Buffer Allocate(size_t bytes, int cudaDevice = -1);
    static Buffer Borrow(void *data, size_t bytes, MemoryKind kind, int device, std::shared_ptr<void> owner);
    Buffer Slice(size_t offset, size_t count) const;
    const void *Data() const;
    void *MutableData() const;
    Buffer ReadOnly() const;
    size_t Size() const { return length; }
    MemoryKind Kind() const;
    int Device() const;
    uint64_t Id() const;
    size_t Offset() const { return offset; }
    void Read(void *host, size_t at, size_t bytes) const;
    void Write(const void *host, size_t at, size_t bytes) const;
    void CopyFrom(const Buffer &source, size_t sourceOffset, size_t targetOffset, size_t bytes) const;
    void MakeRemoteWritesVisible() const;
    std::shared_ptr<void>
    Attachment(const std::string &key,
               const std::function<std::shared_ptr<void>(void *, size_t)> &create) const;

  private:
    std::shared_ptr<Storage> storage;
    size_t offset = 0, length = 0;
    bool readOnly = false;
};

struct Message {
    Bytes metadata;
    std::vector<Buffer> buffers;
};

// Independent immutable copy with the same logical buffer boundaries. Small
// buffers share bounded arenas, so RDMA registers allocations, not tensors.
// storageDevice=-2 preserves each source's memory kind/device; -1 means host.
Message Snapshot(const Message &message, int storageDevice = -2, size_t arenaBytes = 64 << 20);

struct Options {
    Transport transport = Transport::Tcp;
    std::vector<std::string> rdmaDevices;
    int rdmaPort = 1;
    int gidIndex = -1;      // discover an IPv4 RoCE-v2 GID, or use an explicit index
    int receiveDevice = -1; // -1: host; otherwise receive into this CUDA device
    size_t maxPayloadBytes = size_t(4) << 30;
    size_t maxInflightBytes = size_t(8) << 30;
    size_t maxMetadataBytes = 64 << 20;
    size_t maxBuffers = 4096;
    int workers = 4;
    int queueDepth = 64;
    int timeoutMs = 1800000;
};

struct Metrics {
    uint64_t tcpBytes = 0, rdmaBytes = 0, calls = 0, failures = 0;
};

// A service is embedded in an owning process. The bounded worker pool runs
// handlers; buffers are retained until transfer completion or QP teardown.
// No model library, scheduler, or external registry is required.
class Service {
  public:
    using Handler = std::function<Message(const Message &)>;
    explicit Service(Options options = {});
    ~Service();
    Service(const Service &) = delete;
    Service &operator=(const Service &) = delete;
    void Start(const std::string &listenAddress, Handler handler);
    void Stop();
    std::string Address() const;
    Message Call(const std::string &peer, const Message &request);
    std::future<Message> Submit(const std::string &peer, Message request);
    Metrics Stats() const;
    static bool HasRdma();
    static bool HasCuda();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Logical placement planning is independent of transports. Axes are physical
// elements in an agreed wire representation (packed/quantized components use
// their packed axes). Different representation IDs require an explicit codec.
struct Shard {
    std::string component, representation, owner;
    Buffer buffer;
    std::vector<uint64_t> fullShape, begin, shape;
    uint32_t elementBytes = 1;
    uint64_t replica = 0;
};
struct CopyRegion {
    size_t source, target;
    uint64_t sourceOffset, targetOffset, bytes;
};
std::vector<CopyRegion> PlanCopies(const std::vector<Shard> &sources, const std::vector<Shard> &targets);

// Receiver-owned, byte-based credits; reservations survive queued transfers.
class Budget {
  public:
    explicit Budget(size_t capacity);
    std::shared_ptr<void> Reserve(size_t bytes);
    size_t Used() const;

  private:
    struct State;
    std::shared_ptr<State> state;
};

// Complete immutable objects only. A returned lease pins both data and its
// budget; replacement/removal cannot invalidate a concurrent reader.
class ObjectStore {
  public:
    explicit ObjectStore(size_t byteLimit, int storageDevice = -1);
    void Put(const std::string &key, Message value);
    std::shared_ptr<const Message> Get(const std::string &key);
    void Remove(const std::string &key);
    size_t BytesUsed() const;

  private:
    struct Entry {
        std::shared_ptr<const Message> value;
        uint64_t access;
    };
    mutable std::mutex mutex;
    Budget budget;
    int storageDevice;
    uint64_t clock = 0;
    std::map<std::string, Entry> objects;
};

// Exact-boundary prefix checkpoints with shared immutable payload pages.
// Hashes only prune candidates; tokens and page bytes are compared before reuse.
// The caller's namespace includes checkpoint, layout and tenant/media identity.
class PrefixCache {
  public:
    struct Hit {
        size_t tokens = 0;
        std::shared_ptr<const Message> state;
    };
    struct Statistics {
        uint64_t lookups = 0, hits = 0, inputTokens = 0, hitTokens = 0;
        size_t entries = 0, bytes = 0, logicalBytes = 0;
    };
    // Zero maxEntries uses only the byte budget, including checkpoint metadata.
    explicit PrefixCache(size_t byteLimit, size_t maxEntries = 0, int storageDevice = -1);
    ~PrefixCache();
    void Publish(const std::string &nameSpace, const std::vector<int> &tokens, Message state);
    Hit Find(const std::string &nameSpace, const std::vector<int> &tokens, size_t maxTokens);
    size_t BytesUsed() const;
    Statistics Stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace fastkv
