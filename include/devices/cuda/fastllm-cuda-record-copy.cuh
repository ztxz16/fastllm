#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace fastllm {
namespace cuda {

// Opaque records: weights, scales and any other payload are copied unchanged.
// The caller owns storage and valid source/destination IDs. Destination IDs
// must be distinct; source and destination ranges must not overlap. Source
// may be mapped pinned host memory or device memory.
struct RecordCopyView {
    const void *source;
    void *destination;
    size_t sourcePitch;
    size_t destinationPitch;
    size_t bytesPerRecord;
};

struct RecordCopyLaunch { int blocks; int threads; };

inline RecordCopyLaunch RecordCopyConfiguration(size_t bytesPerRecord,
                                                int maxRecords,
                                                const cudaDeviceProp &device) {
    int threads = std::max(32, std::min(1024, device.maxThreadsPerBlock) / 32 * 32);
    // Enough outstanding mapped-host reads to cover PCIe latency. Capacity,
    // rather than the SM version, caps concurrency on smaller devices.
    size_t capacity = size_t(std::max(1, device.multiProcessorCount)) *
                      std::max(threads, device.maxThreadsPerMultiProcessor);
    size_t target = std::min(size_t(48 * 1024), capacity);
    size_t units = bytesPerRecord / 16 + (bytesPerRecord % 16 != 0);
    if (maxRecords <= 0 || units == 0) return {1, threads};
    // Saturate at the target before multiplication to avoid size overflow.
    size_t work = units >= target / size_t(maxRecords) + 1
        ? target : std::min(target, units * size_t(maxRecords));
    return {int(std::max(size_t(1), (work + threads - 1) / threads)), threads};
}

namespace record_copy_detail {
template<class Unit, class Index>
__global__ void CopyKernel(RecordCopyView view, const int32_t *sourceIds,
                           const int32_t *destinationIds, const int32_t *count,
                           Index unitsPerRecord) {
    int records = *count;
    if (records <= 0) return;
    size_t total = size_t(records) * unitsPerRecord;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total; flat += stride) {
        // Use 32-bit division for ordinary batches, retaining a 64-bit path
        // for larger payloads. The grid-stride counter cannot wrap at 2^32.
        Index record = Index(flat) / unitsPerRecord;
        Index unit = Index(flat) - record * unitsPerRecord;
        const Unit *source = reinterpret_cast<const Unit *>(
            static_cast<const uint8_t *>(view.source) +
            size_t(sourceIds[record]) * view.sourcePitch);
        Unit *destination = reinterpret_cast<Unit *>(
            static_cast<uint8_t *>(view.destination) +
            size_t(destinationIds[record]) * view.destinationPitch);
        destination[unit] = source[unit];
    }
}

template<class Unit>
inline void Launch(RecordCopyView view, const int32_t *sourceIds,
                   const int32_t *destinationIds, const int32_t *count,
                   int maxRecords, RecordCopyLaunch launch, cudaStream_t stream) {
    size_t units = view.bytesPerRecord / sizeof(Unit);
    if (units <= std::numeric_limits<uint32_t>::max() / size_t(maxRecords)) {
        CopyKernel<Unit, uint32_t><<<launch.blocks, launch.threads, 0, stream>>>(
            view, sourceIds, destinationIds, count, uint32_t(units));
    } else {
        CopyKernel<Unit, uint64_t><<<launch.blocks, launch.threads, 0, stream>>>(
            view, sourceIds, destinationIds, count, uint64_t(units));
    }
}
} // namespace record_copy_detail

// Graph-capturable: count is read on the GPU, with no allocation or host
// synchronization. The caller guarantees 0 <= *count <= maxRecords, valid
// address ranges and a launch supported by the current device. Packed or
// unaligned layouts use a byte-copy fallback; aligned layouts use 16B loads.
inline bool CopyRecords(RecordCopyView view, const int32_t *sourceIds,
                        const int32_t *destinationIds, const int32_t *count,
                        int maxRecords, RecordCopyLaunch launch, cudaStream_t stream) {
    if (maxRecords < 0 || launch.blocks <= 0 || launch.threads <= 0 ||
        view.bytesPerRecord > view.sourcePitch ||
        view.bytesPerRecord > view.destinationPitch) return false;
    if (maxRecords == 0 || view.bytesPerRecord == 0) return true;
    if (!view.source || !view.destination || !sourceIds || !destinationIds || !count ||
        view.bytesPerRecord > std::numeric_limits<size_t>::max() / size_t(maxRecords)) return false;
    uintptr_t alignment = reinterpret_cast<uintptr_t>(view.source) |
                          reinterpret_cast<uintptr_t>(view.destination) |
                          view.sourcePitch | view.destinationPitch | view.bytesPerRecord;
    if ((alignment & 15) == 0)
        record_copy_detail::Launch<uint4>(view, sourceIds, destinationIds, count, maxRecords, launch, stream);
    else
        record_copy_detail::Launch<uint8_t>(view, sourceIds, destinationIds, count, maxRecords, launch, stream);
    return true;
}

} // namespace cuda
} // namespace fastllm
