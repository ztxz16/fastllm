#pragma once

// Lossless inverse of NUMA's R4 packing. Shared by streamed prefill and the
// expert cache; projection kernels continue to receive ordinary GGUF blocks.
// Include gguf.h before this header.
namespace fastllm_gguf_restore {
constexpr size_t kTileBytes = 16 * 1024;
constexpr int kRestoreThreads = 256;
constexpr int kWarpsPerBlock = kRestoreThreads / 32;

__host__ __device__ inline int Ordinary(int type) {
    switch (type) {
        case GGML_TYPE_Q2_K_R4: return GGML_TYPE_Q2_K;
        case GGML_TYPE_Q4_K_R4: return GGML_TYPE_Q4_K;
        case GGML_TYPE_IQ2_XXS_R4: return GGML_TYPE_IQ2_XXS;
        case GGML_TYPE_IQ2_XS_R4: return GGML_TYPE_IQ2_XS;
        case GGML_TYPE_IQ2_S_R4: return GGML_TYPE_IQ2_S;
        case GGML_TYPE_IQ3_XXS_R4: return GGML_TYPE_IQ3_XXS;
        default: return type;
    }
}

__device__ inline unsigned Unsign(unsigned x) { return (x ^ (x << 1)) & 127; }

// A warp restores one block. srcRow is local to a NUMA shard; i addresses
// the canonical destination, so shards never need a second host allocation.
__device__ inline void Block(const uint8_t *source, uint8_t *destination,
        int type, int blocks, int srcRow, int col, int i) {
    const int lane = threadIdx.x % 32;
    const int rlane = srcRow % 4, r4 = (srcRow / 4) * blocks + col;
    if (type == GGML_TYPE_Q2_K_R4) {
        const auto &s = reinterpret_cast<const block_q2_k_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_q2_K *>(destination)[i];
        if (lane == 0) {
            auto *scales = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(&d) + sizeof(d.scales) + sizeof(d.qs));
            scales[0] = reinterpret_cast<const uint16_t *>(s.d)[rlane];
            scales[1] = reinterpret_cast<const uint16_t *>(s.d)[rlane+4];
        }
        if (lane < 16) d.scales[lane] = s.scales[4*lane+rlane];
        for (int b = lane; b < 64; b += 32) {
            unsigned packed = 0;
            for (int k = 0; k < 4; ++k) {
                const int c = (b/32)*128 + b%32 + k*32, p = c%32;
                const unsigned q = s.qs[32*(c/32)+4*rlane+p%4+16*(p/16)];
                packed |= ((q >> (2*((p%16)/4))) & 3) << (2*k);
            }
            d.qs[b] = packed;
        }
    } else if (type == GGML_TYPE_Q4_K_R4) {
        const auto &s = reinterpret_cast<const block_q4_k_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_q4_K *>(destination)[i];
        if (lane == 0) {
            reinterpret_cast<uint16_t *>(&d)[0] = reinterpret_cast<const uint16_t *>(s.d)[rlane];
            reinterpret_cast<uint16_t *>(&d)[1] = reinterpret_cast<const uint16_t *>(s.d)[rlane+4];
        }
        // Each of the first four lanes owns three canonical scale bytes.
        if (lane < 4) {
            const int lo = 4*lane+rlane, hi = lo+16;
            const unsigned a = s.scales_h[lo], l = s.scales_l[lo], h = s.scales_l[hi];
            const unsigned ds0 = (l&15)+16*(a&3), ms0 = (l>>4)+16*((a>>2)&3);
            const unsigned ds1 = (h&15)+16*((a>>4)&3), ms1 = (h>>4)+16*(a>>6);
            d.scales[lane] = ds0 | ((ds1>>4)<<6);
            d.scales[lane+4] = ms0 | ((ms1>>4)<<6);
            d.scales[lane+8] = (ds1&15) | ((ms1&15)<<4);
        }
        for (int b = lane; b < 128; b += 32) {
            unsigned packed = 0;
            for (int k = 0; k < 2; ++k) {
                const int c = (b/32)*64+b%32+k*32, p = c%32;
                const unsigned q = s.qs[64*(c/32)+4*rlane+p%4+32*((p%8)/4)+16*(p/16)];
                packed |= ((q >> (4*((p%16)/8))) & 15) << (4*k);
            }
            d.qs[b] = packed;
        }
    } else if (type == GGML_TYPE_IQ2_XS_R4) {
        const auto &s = reinterpret_cast<const block_iq2_xs_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_iq2_xs *>(destination)[i];
        if (lane == 0) d.d = s.d[rlane];
        if (lane < 8) d.scales[lane] = s.scales[4*lane+rlane];
        const unsigned q = s.qs[16*(lane/4)+4*rlane+lane%4];
        d.qs[lane] = (q & 511) | (Unsign(q>>9)<<9);
    } else if (type == GGML_TYPE_IQ2_XXS_R4) {
        const auto &s = reinterpret_cast<const block_iq2_xxs_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_iq2_xxs *>(destination)[i];
        if (lane == 0) d.d = s.d[rlane];
        reinterpret_cast<uint8_t *>(d.qs)[8*(lane/4)+lane%4] =
            s.qs[16*(lane/4)+4*rlane+lane%4];
        if (lane < 8) {
            uint32_t signs = 0, scale = 0;
            for (int j = 0; j < 4; ++j) {
                const unsigned q = s.sas[16*lane+4*rlane+j];
                signs |= Unsign(q>>1) << (7*j);
                scale |= (q & 1) << j;
            }
            signs |= scale << 28;
            d.qs[4*lane+2] = signs; d.qs[4*lane+3] = signs>>16;
        }
    } else if (type == GGML_TYPE_IQ2_S_R4) {
        const auto &s = reinterpret_cast<const block_iq2_s_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_iq2_s *>(destination)[i];
        if (lane == 0) d.d = s.d[rlane];
        if (lane < 8) {
            d.scales[lane] = s.scales[4*lane+rlane];
            d.qh[lane] = s.qh[4*lane+rlane];
        }
        d.qs[lane] = s.qs[16*(lane/4)+4*rlane+lane%4];
        d.qs[32+lane] = s.signs[16*(lane/4)+4*rlane+lane%4];
    } else if (type == GGML_TYPE_IQ3_XXS_R4) {
        const auto &s = reinterpret_cast<const block_iq3_xxs_r4 *>(source)[r4];
        auto &d = reinterpret_cast<block_iq3_xxs *>(destination)[i];
        if (lane == 0) d.d = s.d[rlane];
        for (int b = lane; b < 64; b += 32)
            d.qs[b] = s.qs[32*(b/8)+8*rlane+b%8];
        if (lane < 8) {
            uint32_t signs = 0, scale = 0;
            for (int j = 0; j < 4; ++j) {
                const unsigned q = s.sas[16*lane+4*rlane+j];
                signs |= Unsign(q>>1) << (7*j);
                scale |= (q & 1) << j;
            }
            const uint32_t packed = signs | (scale << 28);
            for (int b = 0; b < 4; ++b) d.qs[64+4*lane+b] = packed >> (8*b);
        }
    }
}

struct Matrix {
    int type = -1, rows = 0, columns = 0;
    size_t rowBytes = 0;
};
struct Record {
    Matrix weights[2];
    int shards = 0;
    size_t downOffset = 0, stride = 0;
};

template<class Unit>
__device__ void Rows(const Matrix &w, void *const *pointers,
        uint8_t *destination, int shards, bool cross, uint32_t *tile) {
    if (!cross) {
        // Down projections are already in row order. Copy entire shards,
        // including across row boundaries, without per-element row division.
        const size_t units = size_t(w.rows) * w.rowBytes / shards / sizeof(Unit);
        for (int node = 0; node < shards; ++node) {
            const auto *source = static_cast<const Unit *>(pointers[node]);
            for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
                 i < units; i += size_t(gridDim.x) * blockDim.x)
                reinterpret_cast<Unit *>(destination)[size_t(node) * units + i] = source[i];
        }
        return;
    }
    if constexpr (sizeof(Unit) < 16) {
        const size_t shardBytes = size_t(w.rows) * w.rowBytes / shards;
        if (!(shardBytes & 15)) {
            // Read the packed host rows contiguously and transpose in shared memory.
            const size_t rowUnits = w.rowBytes / sizeof(Unit);
            for (int node = 0; node < shards; ++node) {
                const auto *source = static_cast<const uint8_t *>(pointers[node]);
                for (size_t base = size_t(blockIdx.x) * kTileBytes; base < shardBytes;
                     base += size_t(gridDim.x) * kTileBytes) {
                    const size_t bytes = min(kTileBytes, shardBytes - base);
                    for (size_t i = threadIdx.x; i < bytes / 16; i += blockDim.x)
                        reinterpret_cast<uint4 *>(tile)[i] = reinterpret_cast<const uint4 *>(source + base)[i];
                    __syncthreads();
                    const size_t first = (size_t(node) * shardBytes + base) / sizeof(Unit);
                    for (size_t i = threadIdx.x; i < bytes / sizeof(Unit); i += blockDim.x) {
                        const size_t index = first + i;
                        const int row = index / rowUnits;
                        const int outputRow = row / 2 + (row % 2) * (w.rows / 2);
                        reinterpret_cast<Unit *>(destination)[size_t(outputRow) * rowUnits + index % rowUnits] =
                            reinterpret_cast<const Unit *>(tile)[i];
                    }
                    __syncthreads();
                }
            }
            return;
        }
    }
    const size_t units = w.rowBytes / sizeof(Unit);
    const int rowsPerShard = w.rows / shards;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < w.rows * units; i += size_t(gridDim.x) * blockDim.x) {
        const int row = i / units;
        const int physical = 2 * (row % (w.rows / 2)) + row / (w.rows / 2);
        const auto *source = static_cast<const uint8_t *>(pointers[physical / rowsPerShard]);
        const int local = physical % rowsPerShard;
        reinterpret_cast<Unit *>(destination)[i] = reinterpret_cast<const Unit *>(source)[
            size_t(local) * units + i % units];
    }
}

// Read adjacent packed bytes once, then restore all four rows in shared memory.
template<int Type, class Packed>
__device__ void PackedRows(const Matrix &w, void *const *pointers,
        uint8_t *destination, int shards, bool cross, uint32_t *tile) {
    constexpr int words = sizeof(Packed) / sizeof(uint32_t);
    const int columns = w.columns / 256;
    const int groups = (w.rows / 4) * columns;
    const int perShard = groups / shards;
    for (int base = blockIdx.x * kWarpsPerBlock; base < groups; base += gridDim.x * kWarpsPerBlock) {
        const int active = min(kWarpsPerBlock, groups - base);
        if (shards == 1) {
            const auto *source = static_cast<const uint32_t *>(pointers[0]) + size_t(base) * words;
            const int vectors = active * words / 4;
            for (int i = threadIdx.x; i < vectors; i += blockDim.x)
                reinterpret_cast<uint4 *>(tile)[i] = reinterpret_cast<const uint4 *>(source)[i];
            for (int i = 4 * vectors + threadIdx.x; i < active * words; i += blockDim.x)
                tile[i] = source[i];
        } else {
            for (int i = threadIdx.x; i < active * words; i += blockDim.x) {
                const int group = base + i / words;
                const auto *source = static_cast<const uint32_t *>(pointers[group / perShard]);
                tile[i] = source[(group % perShard) * words + i % words];
            }
        }
        __syncthreads();
        const int warp = threadIdx.x / 32;
        if (warp < active) {
            const int group = base + warp;
            for (int row = 0; row < 4; ++row) {
                const int physical = 4 * (group / columns) + row;
                const int outputRow = cross ? physical / 2 + (physical % 2) * (w.rows / 2) : physical;
                Block(reinterpret_cast<const uint8_t *>(tile), destination, Type,
                    columns, row, warp, outputRow * columns + group % columns);
            }
        }
        __syncthreads();
    }
}

// Cache refill and staged execution borrow the same immutable NUMA shards.
// IDs may come from on-device routing or an eager decode decision.
static __global__ void Records(Record layout, void *const *pointers,
        uint8_t *destination, const int32_t *experts, const int32_t *slots,
        const int32_t *count, const uint64_t *slotOffsets, int fixedExpert,
        const uint8_t *promoted, int promotedExpert, const uint8_t *uploaded) {
    // A refill usually admits one expert. Bound the launch independently of
    // top-k so empty requests do not occupy blocks on every decode layer.
    __shared__ __align__(16) uint32_t tile[kTileBytes / sizeof(uint32_t)];
    const int requests = count ? *count : 1;
    for (int request = 0; request < requests; ++request) {
        const int part = blockIdx.z;
        const int expert = experts ? experts[request] : fixedExpert;
        const int slot = slots ? slots[request] : request;
        uint8_t *record = destination + (slotOffsets ? slotOffsets[slot] : size_t(slot) * layout.stride);
        const size_t offset = part ? layout.downOffset : 0;
        const auto &w = layout.weights[part];
        auto *output = record + offset;
        if (promoted && expert == promotedExpert) {
            // Promotion of an already streamed expert never reads host weights again.
            const size_t end = part ? layout.stride : layout.downOffset;
            for (size_t i = offset + (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
                 i < end; i += size_t(gridDim.x) * blockDim.x * 16)
                *reinterpret_cast<uint4 *>(record + i) = *reinterpret_cast<const uint4 *>(promoted + i);
            continue;
        }
        void *deviceSource = uploaded
            ? const_cast<uint8_t *>(uploaded + size_t(request) * layout.stride + offset) : nullptr;
        void *const *sources = uploaded ? &deviceSource : pointers + (expert * 2 + part) * layout.shards;
        const int shards = uploaded ? 1 : layout.shards;
        if (Ordinary(w.type) == w.type) {
            const size_t alignment = part ? size_t(w.rows) * w.rowBytes / shards : w.rowBytes;
            if (!(alignment & 15)) Rows<uint4>(w, sources, output, shards, part == 0, tile);
            else if (!(alignment & 3)) Rows<uint32_t>(w, sources, output, shards, part == 0, tile);
            else Rows<uint8_t>(w, sources, output, shards, part == 0, tile);
        } else {
            switch (w.type) {
                case GGML_TYPE_Q2_K_R4: PackedRows<GGML_TYPE_Q2_K_R4, block_q2_k_r4>(w, sources, output, shards, part == 0, tile); break;
                case GGML_TYPE_Q4_K_R4: PackedRows<GGML_TYPE_Q4_K_R4, block_q4_k_r4>(w, sources, output, shards, part == 0, tile); break;
                case GGML_TYPE_IQ2_XXS_R4: PackedRows<GGML_TYPE_IQ2_XXS_R4, block_iq2_xxs_r4>(w, sources, output, shards, part == 0, tile); break;
                case GGML_TYPE_IQ2_XS_R4: PackedRows<GGML_TYPE_IQ2_XS_R4, block_iq2_xs_r4>(w, sources, output, shards, part == 0, tile); break;
                case GGML_TYPE_IQ2_S_R4: PackedRows<GGML_TYPE_IQ2_S_R4, block_iq2_s_r4>(w, sources, output, shards, part == 0, tile); break;
                case GGML_TYPE_IQ3_XXS_R4: PackedRows<GGML_TYPE_IQ3_XXS_R4, block_iq3_xxs_r4>(w, sources, output, shards, part == 0, tile); break;
            }
        }
        // Slot payloads include alignment padding, just like canonical snapshots.
        if (blockIdx.x == 0) {
            const size_t end = part ? layout.stride : layout.downOffset;
            for (size_t i = offset + w.rows * w.rowBytes + threadIdx.x; i < end; i += blockDim.x)
                record[i] = 0;
        }
    }
}

inline cudaError_t CopyRecords(const Record &layout, void *const *pointers,
        uint8_t *destination, cudaStream_t stream,
        const int32_t *experts = nullptr, const int32_t *slots = nullptr,
        const int32_t *count = nullptr, const uint64_t *slotOffsets = nullptr,
        int fixedExpert = 0,
        const uint8_t *promoted = nullptr, int promotedExpert = -1,
        const uint8_t *uploaded = nullptr) {
    Records<<<dim3(64, 1, 2), kRestoreThreads, 0, stream>>>(layout, pointers, destination,
        experts, slots, count, slotOffsets, fixedExpert, promoted, promotedExpert, uploaded);
    return cudaGetLastError();
}
} // namespace fastllm_gguf_restore
