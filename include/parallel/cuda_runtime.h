#ifndef FASTLLM_PARALLEL_CUDA_RUNTIME_H
#define FASTLLM_PARALLEL_CUDA_RUNTIME_H

#include "parallel/topology.h"
#include <cstdint>
#include <functional>
#include <memory>

namespace fastllm {
class Data;
namespace parallel {

// Reconstruct a load-time TP weight before selecting a different topology.
// Supports dense FP32/FP16/BF16 and block FP8; preserves checkpoint bytes.
void RestoreWeightOnHost(Data &weight);

enum class ScalarType { Byte, Float16, BFloat16, Float32, Int32 };

// Uses the existing device worker registry: one persistent submitting thread
// per GPU, shared across all groups. Run joins GPU completion before returning.
// Callbacks use only local operators; recursively dispatching MultiCUDA is invalid.
class CudaRuntime {
public:
    explicit CudaRuntime(Topology topology);
    ~CudaRuntime();
    CudaRuntime(const CudaRuntime&) = delete;
    CudaRuntime& operator=(const CudaRuntime&) = delete;
    const Topology& Layout() const;
    void Run(const std::function<void(int rank, int device)> &callback);
    // Join host submission only, with bidirectional event dependencies between
    // caller and worker streams. Buffers must survive until a later Run/drain.
    // May join a caller-stream capture. Collective members must all capture
    // the same operation and replay its graphs on their respective workers.
    void Enqueue(const std::function<void(int rank, int device)> &callback);
    void AllReduce(int group, int rank, void *data, size_t count, ScalarType type);
    void AllGather(int group, int rank, const void *input, void *output, size_t count, ScalarType type);
    void Broadcast(int group, int rank, void *data, size_t count, ScalarType type, int root = 0);
    uint64_t Generation() const;
private:
    friend class CudaRowTransfer;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Per-rank persistent scratch for one row layout. Methods run on that rank's
// fixed GPU worker, on its per-thread stream. Byte copies preserve exact bits.
class CudaRowTransfer {
public:
    CudaRowTransfer(CudaRuntime &runtime, RowPartition rows);
    ~CudaRowTransfer();
    CudaRowTransfer(const CudaRowTransfer&) = delete;
    CudaRowTransfer& operator=(const CudaRowTransfer&) = delete;
    // Reuse storage across steps; update the permutation at a drained boundary.
    // A captured transfer requires unchanged per-group row counts/capacities;
    // its storage owner must outlive every graph referencing the buffers.
    void Reset(RowPartition rows);
    // Coordinator call at a drained plan boundary, before Collect on any rank.
    void Reserve(size_t maxRowBytes);
    void Select(int rank, const void *global, void *local, size_t rowBytes);
    void Collect(int rank, const void *local, void *global, size_t rowBytes);
    struct Field { const void *local; void *global; size_t rowBytes; };
    // Pack fields into one row record and perform one logical collective.
    void Collect(int rank, const Field *fields, size_t count);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}} // namespace fastllm::parallel
#endif
