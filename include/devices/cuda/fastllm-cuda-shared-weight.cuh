#pragma once

#include "fastllm-cuda-record-copy.cuh"

namespace fastllm {
namespace cuda {

// A byte view of a borrowed, row-sharded tensor. No element/quantization type
// is involved: packed integer blocks, floating weights and inline metadata
// use the same gather. The owner retains the source until all copies finish.
// Logical rows may be grouped (e.g. gate/up halves); physical rows interleave
// those groups. Within each shard, row tiles and payload blocks have strides.
struct SharedWeightView {
    uint32_t rows = 0, rowBytes = 0, rowsPerShard = 0, rowGroups = 1;
    uint32_t tileRows = 1, tileStride = 0, rowStride = 0;
    uint32_t blockBytes = 0, blockStride = 0, sourceOffset = 0;
    uint32_t destinationOffset = 0;
    uint32_t groupRows = 0, shardBytes = 0;

    static SharedWeightView Rows(uint32_t rows, uint32_t rowBytes,
                                 uint32_t shards, uint32_t groups = 1) {
        return {rows, rowBytes, shards ? rows / shards : 0, groups,
                1, rowBytes, rowBytes, rowBytes, rowBytes, 0, 0};
    }
    bool Valid(uint32_t shards, uint32_t recordBytes) const {
        return rows && rowBytes && shards && rows % shards == 0 &&
            rowsPerShard == rows / shards && rowGroups && rows % rowGroups == 0 &&
            tileRows && rowsPerShard % tileRows == 0 && blockBytes &&
            blockStride >= blockBytes &&
            uint64_t((rowBytes - 1) / blockBytes) * blockStride +
                (rowBytes - 1) % blockBytes + 1 <= rowStride &&
            uint64_t(tileRows) * rowStride <= tileStride &&
            uint64_t(destinationOffset) + uint64_t(rows) * rowBytes <= recordBytes;
    }
    uint32_t Alignment() const {
        return rowBytes | tileStride | rowStride | blockBytes | blockStride |
               sourceOffset | destinationOffset;
    }
};

struct SharedRecordSpan {
    uint32_t sourceOffset = 0, destinationOffset = 0, bytes = 0;
};

// Two matrices form an expert record. Auxiliary spans retain only metadata
// that cannot be recovered losslessly from the owner's representation.
// Shard pointers are [expert, matrix, shard]; auxiliary records are optional.
struct SharedExpertLayout {
    SharedWeightView weights[2];
    SharedRecordSpan auxiliary[3];
    uint32_t shards = 0, recordBytes = 0, auxiliaryBytes = 0;
    uint32_t sourceBytes[2]{}, readBytes = 0;
    bool scatter = false;

    bool Prepare() {
        if (!recordBytes || !weights[0].Valid(shards, recordBytes) ||
            !weights[1].Valid(shards, recordBytes)) return false;
        for (const auto &span : auxiliary)
            if (uint64_t(span.sourceOffset) + span.bytes > auxiliaryBytes ||
                uint64_t(span.destinationOffset) + span.bytes > recordBytes) return false;
        uint32_t alignment = recordBytes | auxiliaryBytes;
        uint64_t payload = 0, source = 0;
        for (int part = 0; part < 2; ++part) {
            auto &view = weights[part];
            const uint64_t shardBytes = uint64_t(view.sourceOffset) +
                uint64_t(view.rowsPerShard / view.tileRows) * view.tileStride;
            if (shardBytes > UINT32_MAX / shards) return false;
            view.groupRows = view.rows / view.rowGroups;
            view.shardBytes = shardBytes;
            const uint64_t bytes = uint64_t(shards) * shardBytes;
            sourceBytes[part] = bytes;
            alignment |= view.Alignment();
            payload += uint64_t(view.rows) * view.rowBytes;
            source += bytes;
        }
        for (const auto &span : auxiliary) alignment |= span.sourceOffset | span.destinationOffset | span.bytes;
        // Small inline metadata can turn aligned weight loads into many PCIe
        // fragments. Read contiguous 16B host words, then discard that metadata
        // and scatter payload on-device. Avoid reading large planar scale
        // planes (e.g. NVFP4), where gathering only weight bytes is cheaper.
        const uint32_t sourceAlignment = weights[0].shardBytes | weights[1].shardBytes | auxiliaryBytes;
        scatter = (alignment & 15) && !(alignment & 3) && !(sourceAlignment & 15) &&
            source <= payload + payload / 8 && source + auxiliaryBytes <= UINT32_MAX;
        readBytes = scatter ? uint32_t(source + auxiliaryBytes) : 0;
        return true;
    }
};

namespace shared_weight_detail {
template<class Unit>
__device__ Unit Load(const SharedWeightView &view, void *const *pointers,
                     uint32_t offset) {
    uint32_t row = offset / view.rowBytes;
    const uint32_t column = offset - row * view.rowBytes;
    const uint32_t group = row / view.groupRows;
    row = row * view.rowGroups - group * (view.rows - 1);
    const uint32_t shard = row / view.rowsPerShard;
    const auto *source = static_cast<const uint8_t *>(pointers[shard]);
    row -= shard * view.rowsPerShard;
    const uint32_t tile = row / view.tileRows;
    const uint32_t block = column / view.blockBytes;
    const size_t address = view.sourceOffset + size_t(tile) * view.tileStride +
        size_t(row - tile * view.tileRows) * view.rowStride +
        size_t(block) * view.blockStride + column - block * view.blockBytes;
    return *reinterpret_cast<const Unit *>(source + address);
}

template<class Unit>
__global__ void Copy(SharedExpertLayout layout, void *const *pointers,
                     const uint8_t *auxiliary, uint8_t *destination,
                     const int32_t *sourceIds, const int32_t *destinationIds,
                     const int32_t *count) {
    const uint32_t units = layout.recordBytes / sizeof(Unit);
    const int records = *count;
    if (records <= 0) return;
    const uint32_t total = records * units;
    for (size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total; flat += size_t(gridDim.x) * blockDim.x) {
        const uint32_t request = uint32_t(flat) / units, offset = (uint32_t(flat) % units) * sizeof(Unit);
        const int expert = sourceIds[request];
        Unit value{};
        #pragma unroll
        for (int part = 0; part < 2; ++part) {
            const auto &view = layout.weights[part];
            if (offset >= view.destinationOffset &&
                offset - view.destinationOffset < view.rows * view.rowBytes)
                value = Load<Unit>(view, pointers + (expert * 2 + part) * layout.shards,
                                   offset - view.destinationOffset);
        }
        #pragma unroll
        for (const auto &span : layout.auxiliary) {
            if (offset >= span.destinationOffset && offset - span.destinationOffset < span.bytes)
                value = *reinterpret_cast<const Unit *>(auxiliary +
                    size_t(expert) * layout.auxiliaryBytes + span.sourceOffset + offset - span.destinationOffset);
        }
        *reinterpret_cast<Unit *>(destination +
            size_t(destinationIds[request]) * layout.recordBytes + offset) = value;
    }
}
__device__ inline void ScatterWord(const SharedWeightView &view, uint32_t shard,
                                   uint32_t offset, uint32_t value, uint8_t *destination) {
    if (offset < view.sourceOffset) return;
    offset -= view.sourceOffset;
    const uint32_t tile = offset / view.tileStride;
    const uint32_t tileOffset = offset - tile * view.tileStride;
    const uint32_t row = tileOffset / view.rowStride;
    if (row >= view.tileRows) return;
    const uint32_t column = tileOffset - row * view.rowStride;
    const uint32_t block = column / view.blockStride;
    const uint32_t within = column - block * view.blockStride;
    const uint32_t logicalColumn = block * view.blockBytes + within;
    if (within >= view.blockBytes || logicalColumn >= view.rowBytes) return;
    const uint32_t physicalRow = shard * view.rowsPerShard + tile * view.tileRows + row;
    const uint32_t groupRow = physicalRow / view.rowGroups;
    const uint32_t logicalRow = groupRow + (physicalRow - groupRow * view.rowGroups) * view.groupRows;
    *reinterpret_cast<uint32_t *>(destination + view.destinationOffset +
        size_t(logicalRow) * view.rowBytes + logicalColumn) = value;
}

template<class = void>
__global__ void Scatter(SharedExpertLayout layout, void *const *pointers,
                        const uint8_t *auxiliary, uint8_t *destination,
                        const int32_t *sourceIds, const int32_t *destinationIds,
                        const int32_t *count) {
    const int records = *count;
    if (records <= 0) return;
    const uint32_t units = layout.readBytes / 16;
    const uint32_t total = records * units;
    for (size_t flat = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total; flat += size_t(gridDim.x) * blockDim.x) {
        const uint32_t request = uint32_t(flat) / units;
        uint32_t offset = (uint32_t(flat) % units) * 16;
        const int expert = sourceIds[request];
        uint8_t *record = destination + size_t(destinationIds[request]) * layout.recordBytes;
        const int part = offset < layout.sourceBytes[0] ? 0 : 1;
        if (part) offset -= layout.sourceBytes[0];
        if (offset < layout.sourceBytes[part]) {
            const auto &view = layout.weights[part];
            const uint32_t shard = offset / view.shardBytes;
            offset -= shard * view.shardBytes;
            const auto *source = static_cast<const uint8_t *>(pointers[(expert * 2 + part) * layout.shards + shard]);
            const uint4 value = *reinterpret_cast<const uint4 *>(source + offset);
            ScatterWord(view, shard, offset, value.x, record);
            ScatterWord(view, shard, offset + 4, value.y, record);
            ScatterWord(view, shard, offset + 8, value.z, record);
            ScatterWord(view, shard, offset + 12, value.w, record);
        } else {
            offset -= layout.sourceBytes[1];
            const uint4 value = *reinterpret_cast<const uint4 *>(auxiliary +
                size_t(expert) * layout.auxiliaryBytes + offset);
            const uint32_t words[]{value.x, value.y, value.z, value.w};
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                const uint32_t byte = offset + word * 4;
                #pragma unroll
                for (const auto &span : layout.auxiliary)
                    if (byte >= span.sourceOffset && byte - span.sourceOffset < span.bytes)
                        *reinterpret_cast<uint32_t *>(record + span.destinationOffset + byte - span.sourceOffset) = words[word];
            }
        }
    }
}

} // namespace shared_weight_detail

// Prepare() validates layouts and derives source extents once when binding the
// owner. Copy is allocation-free and graph-capturable; count stays on device.
inline bool CopySharedExpertRecords(const SharedExpertLayout &layout,
        void *const *pointers, const uint8_t *auxiliary, uint8_t *destination,
        const int32_t *sourceIds, const int32_t *destinationIds, const int32_t *count,
        int maxRecords, RecordCopyLaunch launch, cudaStream_t stream) {
    if (!pointers || !destination || !sourceIds || !destinationIds || !count ||
        maxRecords <= 0 || layout.recordBytes > UINT32_MAX / uint32_t(maxRecords) ||
        (layout.auxiliaryBytes && !auxiliary)) return false;
    if (layout.scatter && layout.readBytes <= UINT32_MAX / uint32_t(maxRecords)) {
        shared_weight_detail::Scatter<><<<launch.blocks, launch.threads, 0, stream>>>(
            layout, pointers, auxiliary, destination, sourceIds, destinationIds, count);
        return cudaGetLastError() == cudaSuccess;
    }
    uintptr_t alignment = layout.recordBytes | layout.auxiliaryBytes |
        reinterpret_cast<uintptr_t>(auxiliary) | reinterpret_cast<uintptr_t>(destination);
    for (const auto &view : layout.weights) alignment |= view.Alignment();
    for (const auto &span : layout.auxiliary)
        alignment |= span.sourceOffset | span.destinationOffset | span.bytes;
    #define FASTLLM_SHARED_COPY(Unit) \
        shared_weight_detail::Copy<Unit><<<launch.blocks, launch.threads, 0, stream>>>( \
            layout, pointers, auxiliary, destination, sourceIds, destinationIds, count)
    if (!(alignment & 15)) { FASTLLM_SHARED_COPY(uint4); }
    else if (!(alignment & 3)) { FASTLLM_SHARED_COPY(uint32_t); }
    else { FASTLLM_SHARED_COPY(uint8_t); }
    #undef FASTLLM_SHARED_COPY
    return cudaGetLastError() == cudaSuccess;
}

} // namespace cuda
} // namespace fastllm
