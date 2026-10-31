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
    // Destination stores all block32 FP4 payloads, then all UE8M0 scales.
    bool block32Planar = false;

    static SharedWeightView Rows(uint32_t rows, uint32_t rowBytes,
                                 uint32_t shards, uint32_t groups = 1) {
        return {rows, rowBytes, shards ? rows / shards : 0, groups,
                1, rowBytes, rowBytes, rowBytes, rowBytes, 0, 0};
    }
    bool Valid(uint32_t shards, uint32_t recordBytes) const {
        return (!block32Planar || rowBytes % 17 == 0) &&
            rows && rowBytes && shards && rows % shards == 0 &&
            rowsPerShard == rows / shards && rowGroups && rows % rowGroups == 0 &&
            tileRows && rowsPerShard % tileRows == 0 && blockBytes &&
            blockStride >= blockBytes &&
            uint64_t((rowBytes - 1) / blockBytes) * blockStride +
                (rowBytes - 1) % blockBytes + 1 <= rowStride &&
            uint64_t(tileRows) * rowStride <= tileStride &&
            uint64_t(destinationOffset) + uint64_t(rows) * rowBytes <= recordBytes;
    }
    uint32_t Alignment() const {
        // The generic adapter cannot copy a word across a planar boundary.
        return rowBytes | tileStride | rowStride | blockBytes | blockStride |
               sourceOffset | destinationOffset | uint32_t(block32Planar);
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
// A block32 record contains sixteen payload bytes and one scale byte. Resolve
// its NUMA/row address once, then load aligned words and publish one uint4.
// Shard extents are word-aligned, so the final word containing the scale is
// in bounds even when the 17-byte block starts at an unaligned address.
template<bool Batch>
__global__ void RestoreBlock32Planar(SharedExpertLayout layout,
        const uint8_t *gate, const uint8_t *down, uint8_t *destination,
        void *const *pointers, const int32_t *sourceIds,
        const int32_t *destinationIds, const int32_t *count, int stage) {
    const unsigned request = Batch ? blockIdx.y / 2 : 0;
    const unsigned part = Batch ? blockIdx.y % 2 : (stage < 0 ? blockIdx.y : stage);
    if (Batch && int(request) >= *count) return;
    const int expert = Batch ? sourceIds[request] : 0;
    const int slot = Batch ? destinationIds[request] : 0;
    const auto &view = layout.weights[part];
    const unsigned blocksPerRow = view.rowBytes / 17;
    const size_t blocks = size_t(view.rows) * blocksPerRow;
    uint8_t *record = destination + size_t(slot) * layout.recordBytes + view.destinationOffset;
    for (size_t block = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         block < blocks; block += size_t(gridDim.x) * blockDim.x) {
        unsigned row = block / blocksPerRow;
        const unsigned column = (block - size_t(row) * blocksPerRow) * 17;
        const unsigned group = row / view.groupRows;
        row = row * view.rowGroups - group * (view.rows - 1);
        const unsigned shard = row / view.rowsPerShard;
        row -= shard * view.rowsPerShard;
        const unsigned tile = row / view.tileRows;
        const size_t offset = view.sourceOffset + size_t(tile) * view.tileStride +
            size_t(row - tile * view.tileRows) * view.rowStride + column;
        const auto *base = pointers ? static_cast<const uint8_t *>(
            pointers[(expert * 2 + part) * layout.shards + shard]) :
            (part ? down : gate) + size_t(shard) * view.shardBytes;
        // Borrowed views may start inside an allocation. Do not round such a
        // pointer down and read bytes outside the owner's declared extent.
        if (reinterpret_cast<uintptr_t>(base) & 3) {
            #pragma unroll
            for (unsigned byte = 0; byte < 16; ++byte)
                record[block * 16 + byte] = base[offset + byte];
            record[blocks * 16 + block] = base[offset + 16];
            continue;
        }
        const unsigned shift = (offset & 3) * 8;
        const auto *words = reinterpret_cast<const uint32_t *>(base + (offset & ~size_t(3)));
        const uint32_t a = words[0], b = words[1], c = words[2], d = words[3];
        uint4 value = make_uint4(a, b, c, d);
        if (shift) {
            const uint32_t e = words[4];
            value = make_uint4(__funnelshift_r(a, b, shift), __funnelshift_r(b, c, shift),
                              __funnelshift_r(c, d, shift), __funnelshift_r(d, e, shift));
        }
        reinterpret_cast<uint4 *>(record)[block] = value;
        record[blocks * 16 + block] = base[offset + 16];
    }
}

inline bool ContiguousBlock32Planar(const SharedExpertLayout &layout) {
    if (!layout.shards || layout.auxiliaryBytes) return false;
    uint64_t bytes = 0;
    for (const auto &view : layout.weights) {
        if (!view.block32Planar || !view.Valid(layout.shards, layout.recordBytes) ||
            view.destinationOffset != bytes || (view.destinationOffset & 15) ||
            (view.shardBytes & 3) || view.blockBytes != view.rowBytes ||
            view.blockStride != view.rowBytes || view.groupRows != view.rows / view.rowGroups)
            return false;
        bytes += uint64_t(view.rows) * view.rowBytes;
    }
    return bytes == layout.recordBytes && !(bytes & 15);
}

inline unsigned Block32PlanarGrid(const SharedExpertLayout &layout, unsigned blocks, unsigned threads) {
    const size_t work = std::max(size_t(layout.weights[0].rows) * (layout.weights[0].rowBytes / 17),
                                 size_t(layout.weights[1].rows) * (layout.weights[1].rowBytes / 17));
    return unsigned(std::min(size_t(blocks), (work + threads - 1) / threads));
}

// Resolve the same byte view for mapped NUMA shards and concatenated uploads.
__device__ inline size_t SourceOffset(const SharedWeightView &view, uint32_t offset,
                                     uint32_t &shard) {
    if (view.block32Planar) {
        const uint32_t payload = view.rows * (view.rowBytes / 17) * 16;
        offset = offset < payload ? (offset / 16) * 17 + offset % 16 :
                                   (offset - payload) * 17 + 16;
    }
    uint32_t row = offset / view.rowBytes;
    const uint32_t column = offset - row * view.rowBytes;
    const uint32_t group = row / view.groupRows;
    row = row * view.rowGroups - group * (view.rows - 1);
    shard = row / view.rowsPerShard;
    row -= shard * view.rowsPerShard;
    const uint32_t tile = row / view.tileRows;
    const uint32_t block = column / view.blockBytes;
    return view.sourceOffset + size_t(tile) * view.tileStride +
        size_t(row - tile * view.tileRows) * view.rowStride +
        size_t(block) * view.blockStride + column - block * view.blockBytes;
}

template<class Unit>
__device__ Unit Load(const SharedWeightView &view, void *const *pointers,
                     uint32_t offset, const uint8_t *contiguous = nullptr) {
    uint32_t shard;
    const size_t address = SourceOffset(view, offset, shard);
    const auto *source = pointers ? static_cast<const uint8_t *>(pointers[shard]) :
        contiguous + size_t(shard) * view.shardBytes;
    return *reinterpret_cast<const Unit *>(source + address);
}

template<class Unit>
__global__ void CopyRecord(SharedExpertLayout layout, const uint8_t *gate,
        const uint8_t *down, const uint8_t *auxiliary, uint8_t *destination,
        void *const *pointers, int stage) {
    for (size_t offset = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * sizeof(Unit);
         offset < layout.recordBytes; offset += size_t(gridDim.x) * blockDim.x * sizeof(Unit)) {
        Unit value{};
        bool write = stage < 0;
        for (int part = 0; part < 2; ++part) {
            if (stage >= 0 && stage != part) continue;
            const auto &view = layout.weights[part];
            if (offset >= view.destinationOffset && offset - view.destinationOffset < view.rows * view.rowBytes) {
                const auto local = offset - view.destinationOffset;
                value = Load<Unit>(view, pointers ? pointers + part * layout.shards : nullptr,
                    local, part ? down : gate);
                write = true;
            }
        }
        // Gate computation reads both gate/up global scales. Publish all
        // auxiliary metadata before gateReady and never rewrite it during down.
        if (stage != 1) for (const auto &span : layout.auxiliary)
            if (offset >= span.destinationOffset && offset - span.destinationOffset < span.bytes) {
                value = *reinterpret_cast<const Unit *>(auxiliary + span.sourceOffset + offset - span.destinationOffset);
                write = true;
            }
        if (write) *reinterpret_cast<Unit *>(destination + offset) = value;
    }
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

// The uploaded row-sharded NVFP4 record is contiguous within each matrix.
// Gate/up only needs an even/odd row permutation; down is already contiguous.
// Vectorize across down rows as well, even when an individual row is not 16B
// aligned. Other column/tile layouts keep the generic byte-view adapter.
template<int FixedRowBytes = 0>
__global__ void RestoreInterleavedRows(const uint8_t *gate, const uint8_t *down,
        uint8_t *destination, size_t gateBytes, size_t totalBytes,
        unsigned rowBytes, unsigned groupRows) {
    const unsigned pitch = FixedRowBytes ? FixedRowBytes : rowBytes;
    for (size_t byte = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
         byte < totalBytes; byte += size_t(gridDim.x) * blockDim.x * 16) {
        const uint8_t *source;
        if (byte < gateBytes) {
            const size_t row = byte / pitch, column = byte - row * pitch;
            const size_t physical = row < groupRows ? row * 2 : (row - groupRows) * 2 + 1;
            source = gate + physical * pitch + column;
        } else source = down + byte - gateBytes;
        *reinterpret_cast<uint4 *>(destination + byte) = *reinterpret_cast<const uint4 *>(source);
    }
}

inline bool ContiguousInterleavedRows(const SharedExpertLayout &layout) {
    if (!layout.shards || layout.auxiliaryBytes || layout.weights[0].rowGroups != 2 ||
        layout.weights[1].rowGroups != 1 || layout.weights[0].destinationOffset ||
        (layout.weights[0].rowBytes & 15)) return false;
    uint64_t bytes = 0;
    for (const auto &view : layout.weights) {
        if (view.block32Planar || !view.Valid(layout.shards, layout.recordBytes) || view.sourceOffset || view.tileRows != 1 ||
            view.tileStride != view.rowBytes || view.rowStride != view.rowBytes ||
            view.blockBytes != view.rowBytes || view.blockStride != view.rowBytes ||
            view.rowsPerShard % view.rowGroups || view.groupRows != view.rows / view.rowGroups ||
            uint64_t(view.shardBytes) != uint64_t(view.rowsPerShard) * view.rowBytes ||
            view.destinationOffset != bytes) return false;
        bytes += uint64_t(view.rows) * view.rowBytes;
    }
    return bytes == layout.recordBytes && !(bytes & 15);
}

} // namespace shared_weight_detail

inline bool CopySharedExpertRecord(const SharedExpertLayout &layout, void *const *pointers,
        const uint8_t *gate, const uint8_t *down, const uint8_t *auxiliary,
        uint8_t *destination, cudaStream_t stream, int stage = -1) {
    if ((!pointers && (!gate || !down)) || !destination || (layout.auxiliaryBytes && !auxiliary)) return false;
    if (stage >= -1 && stage <= 1 &&
        !((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(down) |
           reinterpret_cast<uintptr_t>(destination)) & 15) &&
        shared_weight_detail::ContiguousBlock32Planar(layout)) {
        const unsigned blocks = shared_weight_detail::Block32PlanarGrid(layout, 256, 256);
        shared_weight_detail::RestoreBlock32Planar<false><<<dim3(blocks, stage < 0 ? 2 : 1), 256, 0, stream>>>(
            layout, gate, down, destination, pointers, nullptr, nullptr, nullptr, stage);
        return cudaGetLastError() == cudaSuccess;
    }
    if (!pointers && stage < 0 &&
        !((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(down) |
           reinterpret_cast<uintptr_t>(destination)) & 15) &&
        shared_weight_detail::ContiguousInterleavedRows(layout)) {
        const auto &view = layout.weights[0];
        const size_t gateBytes = size_t(view.rows) * view.rowBytes;
        if (view.rowBytes == 2720)
            shared_weight_detail::RestoreInterleavedRows<2720><<<256, 256, 0, stream>>>(
                gate, down, destination, gateBytes, layout.recordBytes, view.rowBytes, view.groupRows);
        else
            shared_weight_detail::RestoreInterleavedRows<><<<256, 256, 0, stream>>>(
                gate, down, destination, gateBytes, layout.recordBytes, view.rowBytes, view.groupRows);
        return cudaGetLastError() == cudaSuccess;
    }
    uintptr_t alignment = layout.recordBytes | layout.auxiliaryBytes |
        reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(down) |
        reinterpret_cast<uintptr_t>(auxiliary) | reinterpret_cast<uintptr_t>(destination);
    for (const auto &view : layout.weights) alignment |= view.Alignment() | view.shardBytes;
    for (const auto &span : layout.auxiliary)
        alignment |= span.sourceOffset | span.destinationOffset | span.bytes;
    const int blocks = std::min(256u, (layout.recordBytes + 255) / 256);
    #define FASTLLM_RECORD_COPY(Unit) \
        shared_weight_detail::CopyRecord<Unit><<<blocks, 256, 0, stream>>>(layout, gate, down, auxiliary, destination, pointers, stage)
    if (!(alignment & 15)) { FASTLLM_RECORD_COPY(uint4); }
    else if (!(alignment & 3)) { FASTLLM_RECORD_COPY(uint32_t); }
    else { FASTLLM_RECORD_COPY(uint8_t); }
    #undef FASTLLM_RECORD_COPY
    return cudaGetLastError() == cudaSuccess;
}

// Upload each pinned NUMA shard contiguously, then restore the canonical
// record entirely in VRAM. Reuse staging only on this stream after the gather;
// the caller publishes cache residency after both operations have completed.
// Layouts with separate auxiliary metadata retain their existing adapter.
inline bool UploadSharedExpertRecord(const SharedExpertLayout &layout,
        void *const *hostPointers, uint8_t *staging, size_t stagingBytes,
        uint8_t *destination, cudaStream_t stream) {
    const size_t bytes = size_t(layout.sourceBytes[0]) + layout.sourceBytes[1];
    if (!hostPointers || !staging || !destination || staging == destination ||
        layout.auxiliaryBytes || !bytes || stagingBytes < bytes) return false;
    for (int part = 0; part < 2; ++part) {
        const auto &view = layout.weights[part];
        if (!view.Valid(layout.shards, layout.recordBytes) || !view.shardBytes ||
            uint64_t(view.shardBytes) * layout.shards != layout.sourceBytes[part]) return false;
        for (unsigned shard = 0; shard < layout.shards; ++shard)
            if (!hostPointers[part * layout.shards + shard]) return false;
    }
    size_t offset = 0;
    for (int part = 0; part < 2; ++part) {
        const size_t shardBytes = layout.weights[part].shardBytes;
        for (unsigned shard = 0; shard < layout.shards; ++shard) {
            if (cudaMemcpyAsync(staging + offset, hostPointers[part * layout.shards + shard],
                    shardBytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
            offset += shardBytes;
        }
    }
    return CopySharedExpertRecord(layout, nullptr, staging,
        staging + layout.sourceBytes[0], nullptr, destination, stream);
}

// Prepare() validates layouts and derives source extents once when binding the
// owner. Copy is allocation-free and graph-capturable; count stays on device.
inline bool CopySharedExpertRecords(const SharedExpertLayout &layout,
        void *const *pointers, const uint8_t *auxiliary, uint8_t *destination,
        const int32_t *sourceIds, const int32_t *destinationIds, const int32_t *count,
        int maxRecords, RecordCopyLaunch launch, cudaStream_t stream) {
    if (!pointers || !destination || !sourceIds || !destinationIds || !count ||
        maxRecords <= 0 || launch.blocks <= 0 || launch.threads <= 0 ||
        layout.recordBytes > UINT32_MAX / uint32_t(maxRecords) ||
        (layout.auxiliaryBytes && !auxiliary)) return false;
    if (maxRecords <= 32767 && !(reinterpret_cast<uintptr_t>(destination) & 15) &&
        shared_weight_detail::ContiguousBlock32Planar(layout)) {
        const unsigned blocks = shared_weight_detail::Block32PlanarGrid(layout, launch.blocks, launch.threads);
        shared_weight_detail::RestoreBlock32Planar<true><<<dim3(blocks, 2 * maxRecords), launch.threads, 0, stream>>>(
            layout, nullptr, nullptr, destination, pointers, sourceIds, destinationIds, count, -1);
        return cudaGetLastError() == cudaSuccess;
    }
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
