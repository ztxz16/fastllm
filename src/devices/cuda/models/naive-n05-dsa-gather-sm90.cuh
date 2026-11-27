#pragma once
#include "naive-n05-attention-bulk.cuh"

namespace naive_dsa_gather_sm90 {
using B = __nv_bfloat16;
template <int Heads, int Tile> struct Storage {
    __align__(16) B query[Heads][200];
    B prob[Heads][2056];
    union {
        B key[2][Tile][200];
        B value[2][Tile][136];
    } tile;
    int keyIndex[2048], valueIndex[2048];
};

template <int Width, int Pitch, int Tile, int Threads, bool Fp8KV>
__device__ __forceinline__ void Stage(B (&tile)[Tile][Pitch], const void *src, int stride, const int *indices,
                                      int first, int kvHead) {
    for (int i = threadIdx.x; i < Tile * (Width / 8); i += Threads) {
        const int slot = i / (Width / 8), col = i % (Width / 8) * 8;
        const int idx = indices[first + slot];
        const size_t offset = size_t(max(idx, 0)) * stride + kvHead * Width + col;
        if constexpr (Fp8KV) {
            // Stage eight E4M3 bytes in their final sixteen-byte BF16 slot.
            const unsigned dst = __cvta_generic_to_shared(&tile[slot][col]);
            const uint8_t *bytes = static_cast<const uint8_t *>(src) + offset;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                         :: "r"(dst), "l"(bytes), "r"(idx >= 0 ? 8 : 0) : "memory");
        } else naive_dsa_decode::Copy<true>(&tile[slot][col], static_cast<const B *>(src) + offset, idx >= 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<int Width, int Pitch, int Tile, int Threads>
__device__ __forceinline__ void ExpandTile(B (&tile)[Tile][Pitch]) {
    for (int i = threadIdx.x; i < Tile * (Width / 8); i += Threads) {
        B *slot = &tile[i / (Width / 8)][(i % (Width / 8)) * 8];
        const uint4 expanded = FastllmNaiveKVRow(slot, true).Load8();
        *reinterpret_cast<uint4 *>(slot) = expanded;
    }
    __syncthreads();
}

// Adjacent GQA heads share a gather. Keep K and V page translations distinct,
// and retain the same per-head MMA order and BF16 softmax boundaries as DecodeRow.
template <int Heads, int Tile, class Batch, class Writer, bool Fp8KV = false>
__global__ __launch_bounds__(Heads * 32) void GatherAttention32(const __grid_constant__ Batch batch,
                                                                const B *query, const float *sink, B *out,
                                                                Writer write) {
#if __CUDA_ARCH__ == 900
    using namespace naive_attention_bulk;
    static_assert((Heads == 8 || Heads == 16) && (Tile == 64 || Tile == 128));
    constexpr int threads = Heads * 32;
    extern __shared__ __align__(16) unsigned char shared[];
    auto &s = *reinterpret_cast<Storage<Heads, Tile> *>(shared);
    const int t = threadIdx.x, l = t % 32, warp = t / 32, row = blockIdx.x;
    const int head = blockIdx.y * Heads, kh = head / 16;
    const int length = naive_dsa_decode::Length(batch.rows[row]);
    if (!length) {
        for (int i = t; i < Heads * 16; i += threads)
            write.zero(out, row, head * 128 + i * 8);
        return;
    }
    const auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    for (int i = t; i < 2048; i += threads) {
        const int k = r.indices[i];
        const bool valid = k >= 0 && k < length;
        s.keyIndex[i] = valid ? FastllmNaivePhysicalRow(r, k) : -1;
        s.valueIndex[i] = valid ? FastllmNaivePhysicalRow(r, k, true) : -1;
    }
    if (l < 24)
        *reinterpret_cast<uint4 *>(&s.query[warp][l * 8]) =
            reinterpret_cast<const uint4 *>(query + (size_t(row) * 64 + head + warp) * 192)[l];
    __syncthreads();
    Stage<192, 200, Tile, threads, Fp8KV>(s.tile.key[0], r.key, r.keyStride, s.keyIndex, 0, kh);
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < 2048; first += Tile) {
        const int bank = first / Tile % 2;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (first + Tile < 2048)
            Stage<192, 200, Tile, threads, Fp8KV>(s.tile.key[1 - bank], r.key, r.keyStride, s.keyIndex,
                                           first + Tile, kh);
        if constexpr (Fp8KV) ExpandTile<192, 200, Tile, threads>(s.tile.key[bank]);
        if (warp < (Tile / 16) * (Heads / 8)) {
            const int part = warp / (Tile / 16) * 8, base = (warp % (Tile / 16)) * 16;
            float scores[4]{};
#pragma unroll
            for (int d = 0; d < 192; d += 16) {
                uint32_t a[4], b[2];
                ld4(a, &s.tile.key[bank][base + ar][d + ac]);
                ld2(b, &s.query[part + (l & 7)][d + ((l >> 3) & 1) * 8]);
                mma(scores, a, b);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = first + base + l / 4 + j / 2 * 8;
                s.prob[part + (l % 4) * 2 + j % 2][k] = __float2bfloat16(
                    s.keyIndex[k] < 0
                        ? -INFINITY
                        : naive_dsa_decode::Round(naive_dsa_decode::Round(scores[j]) * rsqrtf(192.f)));
            }
        }
    }
    __syncthreads();
    Stage<128, 136, Tile, threads, Fp8KV>(s.tile.value[0], r.value, 512, s.valueIndex, 0, kh);
    Normalize<2048, false>(&s.prob[0][0], sink, head, 2048, warp, l);
    float acc[4]{};
    for (int first = 0; first < 2048; first += Tile) {
        const int bank = first / Tile % 2;
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (first + Tile < 2048)
            Stage<128, 136, Tile, threads, Fp8KV>(s.tile.value[1 - bank], r.value, 512, s.valueIndex,
                                           first + Tile, kh);
        if constexpr (Fp8KV) ExpandTile<128, 136, Tile, threads>(s.tile.value[bank]);
#pragma unroll
        for (int k = 0; k < Tile; k += 16) {
            uint32_t a[4], b[2];
            ldt4(a, &s.tile.value[bank][k + (l & 7) + (l >> 4) * 8][(warp % 8) * 16 + ((l >> 3) & 1) * 8]);
            ld2(b, &s.prob[(warp / 8) * 8 + (l & 7)][first + k + ((l >> 3) & 1) * 8]);
            mma(acc, a, b);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        s.query[warp / 8 * 8 + l % 4 * 2 + j % 2][warp % 8 * 16 + l / 4 + j / 2 * 8] = __float2bfloat16_rn(acc[j]);
    __syncthreads();
    for (int i = t; i < Heads * 16; i += threads)
        write(out, row, head * 128 + i * 8, *reinterpret_cast<const uint4 *>(&s.query[i / 16][i % 16 * 8]));
#endif
}
} // namespace naive_dsa_gather_sm90
