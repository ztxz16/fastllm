#pragma once

#include "fastllm.h"
#include "fastkv/fastkv.h"

namespace fastllm {

struct ResponseContext;
// Prepared by the model scheduler. Transfer workers only fill its private
// ranges; Commit runs under forwardLocker after successful transfer.
class PDReceiveState {
  public:
    virtual ~PDReceiveState() = default;
    virtual std::vector<std::vector<fastkv::Buffer>> Targets(
        const fastkv::Bytes &metadata, const std::vector<size_t> &sizes) = 0;
    virtual void Commit(ResponseContext &context) = 0;
    virtual void CommitShards(ResponseContext &, const fastkv::Message &);
};

// Logical payload metadata contains placement ranks, never allocator pointers.
// The receiving model resolves those ranks to its own device map.
fastkv::Message PDStatePayload(const fastkv::Message &message);
std::vector<int> PDReceivePlacement(const fastkv::Bytes &metadata,
                                   size_t buffers, const std::vector<int> &devices);

// A portable row shard holds consecutive tokens and a contiguous byte range
// within every logical row. Packed encodings use their physical byte axes.
struct PDRowShard {
    std::string component;
    size_t offset, width;
    fastkv::Buffer buffer;
};
struct PDRowDestination {
    size_t rows, rowBytes;
    std::vector<fastkv::Buffer> spans;
};
struct PDRowMessage {
    std::string layout;
    fastkv::Bytes metadata;
    std::vector<PDRowShard> shards;
    fastkv::Message Encode() const;
    explicit PDRowMessage(const fastkv::Message &wire);
    PDRowMessage() = default;
    void Scatter(const std::map<std::string, PDRowDestination> &targets, int device) const;
};

class PDStateWriter {
  public:
    explicit PDStateWriter(fastkv::BufferPool *pool = nullptr);
    // Rank writers share one message's capacity check, including arena padding.
    PDStateWriter Fork() const;
    fastkv::Writer metadata;
    std::vector<fastkv::Buffer> buffers;
    void Tensor(const Data &tensor);
    // Reserve the complete message before issuing GPU copies. A single-owner
    // producer with peer-aware copy kernels may write into several warm GPUs.
    void PlanTensor(const Data &tensor);
    void Reserve(bool allowPeerWrites = false);
    // Allocate a compact GPU payload directly in an owned transport arena.
    // The Data is a temporary borrowed view; Finish retains the allocation.
    void AllocateTensor(Data &tensor);
    fastkv::Message Finish(const std::vector<int> &devices = {});
  private:
    fastkv::Buffer Allocate(size_t bytes, int device);
    fastkv::BufferPool *pool;
    fastkv::Buffer arena;
    size_t arenaOffset = 0;
    size_t nextTensor = 0;
    std::vector<size_t> plannedTensors;
    struct Allocations;
    std::shared_ptr<Allocations> allocations;
};

// Bind a dense incoming tensor to private final storage. A bounded axis may
// shrink for request-specific tails. The returned slice retains the storage.
fastkv::Buffer PDBindTensor(fastkv::Reader &metadata, Data &storage,
                            const std::shared_ptr<void> &lease, int boundedAxis = -1);

class PDStateReader {
    fastkv::Message message;
    size_t next = 0;
  public:
    explicit PDStateReader(const fastkv::Message &message);
    fastkv::Reader metadata;
    // By default produce an independent tensor on the payload's device.
    // A GPU borrowed view is valid only until this reader is destroyed; callers
    // must complete all queued reads before then. CPU payloads are always copied.
    void Tensor(Data &tensor, bool borrowGpu = false);
    void End();
};

} // namespace fastllm
