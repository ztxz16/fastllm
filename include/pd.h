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
};

// Logical payload metadata contains placement ranks, never allocator pointers.
// The receiving model resolves those ranks to its own device map.
fastkv::Message PDStatePayload(const fastkv::Message &message);
std::vector<int> PDReceivePlacement(const fastkv::Bytes &metadata,
                                   size_t buffers, const std::vector<int> &devices);

class PDStateWriter {
  public:
    explicit PDStateWriter(fastkv::BufferPool *pool = nullptr);
    // Rank writers share one message's capacity check, including arena padding.
    PDStateWriter Fork() const;
    fastkv::Writer metadata;
    std::vector<fastkv::Buffer> buffers;
    void Tensor(const Data &tensor);
    // Optional two-pass export: plan every rank before reserving the complete
    // message on the model lane. CapacityBusy must never escape a GPU worker.
    void PlanTensor(const Data &tensor);
    void Reserve();
    // Allocate a compact GPU payload directly in an owned transport arena.
    // The Data is a temporary borrowed view; Finish retains the allocation.
    void AllocateTensor(Data &tensor);
    fastkv::Message Finish(const std::vector<int> &devices = {});
  private:
    fastkv::Buffer Allocate(size_t bytes, int device);
    fastkv::BufferPool *pool;
    fastkv::Buffer arena;
    size_t arenaOffset = 0;
    int plannedDevice = -2;
    size_t plannedCapacity = 0, plannedOffset = 0, nextArena = 0;
    std::vector<size_t> plannedArenas;
    struct Allocations;
    std::shared_ptr<Allocations> allocations;
};

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
