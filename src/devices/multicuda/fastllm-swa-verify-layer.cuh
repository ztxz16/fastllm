#pragma once
// SM103 TP8 SWA verification. Admission and fixed-Graph preparation are
// handled by the shared attention-output state before launching this kernel.
// Borrow original BF16 weights and existing graph-owned output/mailbox storage.
// Rows are flattened queries, with per-request candidate offsets in [0,7].
#include <cooperative_groups.h>
#include "../cuda/models/naive-n05-mxfp8-projection.cuh"
#include "fastllm-swa-output.cuh"
#include "../cuda/models/naive-n05-swa-mma.cuh"
#include "../cuda/models/naive-n05-swa-ready.cuh"
namespace fastllm_swa_verify {
using namespace fastllm_swa_output;
using fastllm_swa_mma::InputOffset;
using fastllm_swa_mma::QkvColumn;
using fastllm_swa_mma::Step;
constexpr int RowLimit = 64;
constexpr int InputBytes = 8 * 4096 * 2;
__device__ __forceinline__ float RoundBF16(float x) {
    return float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float WarpSum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}
struct Binding {
    BF16 *key, *value;
    const void *const *cache;
    int verifyOffset;
};
struct BatchSwaKeys {
    Binding rows[RowLimit];
    const int *length;
    __device__ int Live(int row) const { return length[row]; }
};
struct Args {
    const BF16 *branch, *weight, *oWeight;
    BF16 *hidden, *q, *attn, *projected, *feature, *normed;
    const float *gamma, *positions, *sink, *outputScales;
    const __nv_bfloat162 *rotary = nullptr;
    float eps, theta, scale;
    int rows, rank;
    bool overlapRouter = false;
    BatchSwaKeys bindings;
    Peers peers;
    uint64_t *epochs;
};
template <int Threads, int Group>
__device__ __forceinline__ void Normalize(const Args &a, int first, unsigned char *tile) {
    constexpr int Groups = Threads / Group, Warps = Group / 32, Pieces = 4096 / (Group * 8);
    const int t = threadIdx.x, lane = t % 32, group = t / Group, local = t % Group;
    __shared__ float sums[Groups][Warps];
    for (int base = 0; base < 8; base += Groups) {
        const int r = base + group, row = first + r;
        uint4 values[Pieces];
        float square = 0;
#pragma unroll
        for (int j = 0; j < Pieces; ++j) {
            int col = (local + j * Group) * 8;
            uint4 h{}, b{};
            if (r < 8 && row < a.rows) {
                h = *reinterpret_cast<const uint4 *>(a.hidden + row * 4096 + col);
                if (a.branch)
                    b = *reinterpret_cast<const uint4 *>(a.branch + row * 4096 + col);
            }
            auto *hp = reinterpret_cast<unsigned *>(&h);
            auto *bp = reinterpret_cast<unsigned *>(&b);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(hp + k));
                float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(bp + k));
                __nv_bfloat162 z = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
                hp[k] = *reinterpret_cast<unsigned *>(&z);
                float2 v = __bfloat1622float2(z);
                square = fmaf(v.x, v.x, square);
                square = fmaf(v.y, v.y, square);
            }
            values[j] = h;
        }
        float part = WarpSum(square);
        if (lane == 0)
            sums[group][local / 32] = part;
        __syncthreads();
        float total = WarpSum(lane < Warps ? sums[group][lane] : 0.f);
        float inv = rsqrtf(__shfl_sync(0xffffffff, total, 0) / 4096 + a.eps);
#pragma unroll
        for (int j = 0; j < Pieces; ++j) {
            int col = (local + j * Group) * 8;
            uint4 value = values[j], result;
            auto *vp = reinterpret_cast<unsigned *>(&value);
            auto *rp = reinterpret_cast<unsigned *>(&result);
            float4 lo = *reinterpret_cast<const float4 *>(a.gamma + col),
                   hi = *reinterpret_cast<const float4 *>(a.gamma + col + 4);
            float gain[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(vp + k));
                __nv_bfloat162 out = __floats2bfloat162_rn(RoundBF16(x.x * inv) * gain[k * 2],
                                                           RoundBF16(x.y * inv) * gain[k * 2 + 1]);
                rp[k] = *reinterpret_cast<unsigned *>(&out);
            }
            if (r < 8)
                *reinterpret_cast<uint4 *>(tile + InputOffset(r, col)) = result;
            if (blockIdx.x == 0 && a.feature && r < 8 && row < a.rows)
                *reinterpret_cast<uint4 *>(a.feature + row * 4096 + col) = value;
        }
        __syncthreads();
    }
}
// For larger batches each row is normalized once by its own CTA. All
// owners reside alongside QKV consumers. Borrow the existing normed output
// as scratch; it is no longer needed after attention enters the O tail.
__device__ __forceinline__ void NormalizeOwner(const Args &a) {
    const int t = threadIdx.x, row = blockIdx.x;
    __shared__ float sums[8];
    float value[16], affine[16], square = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        value[j] = 0;
        if (t < 256) {
            int col = t + j * 256;
            value[j] = RoundBF16(float(a.hidden[row * 4096 + col]) +
                                 (a.branch ? float(a.branch[row * 4096 + col]) : 0.f));
            affine[j] = a.gamma[col];
            square += value[j] * value[j];
        }
    }
    square = WarpSum(square);
    if (t < 256 && t % 32 == 0)
        sums[t / 32] = square;
    __syncthreads();
    float total = WarpSum(t < 8 ? sums[t] : 0.f);
    if (t == 0)
        sums[0] = total;
    __syncthreads();
    float inv = rsqrtf(sums[0] / 4096 + a.eps);
#pragma unroll
    for (int j = 0; j < 16; ++j)
        if (t < 256) {
            int col = t + j * 256;
            a.normed[row * 4096 + col] = __float2bfloat16_rn(RoundBF16(value[j] * inv) * affine[j]);
            if (a.feature)
                a.feature[row * 4096 + col] = __float2bfloat16_rn(value[j]);
        }
    __syncthreads();
    if (t == 0) {
        auto *p = reinterpret_cast<unsigned *>(a.epochs + 128 + row) + 1;
        fastllm_swa::PublishReady(p);
    }
}
// Vector loads/stores shorten the normalization dependency before direct QKV reads.
// Retain the established BF16 residual and normalization rounding boundaries.
__device__ __forceinline__ void NormalizeOwnerVector(const Args &a) {
    const int t = threadIdx.x, row = blockIdx.x, lane = t % 32;
    __shared__ float sums[8];
    extern __shared__ __align__(1024) unsigned char tile[];
    auto *residual = reinterpret_cast<BF16 *>(tile);
    uint4 values[2];
    float square = 0;
    if (t < 256) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = (t + j * 256) * 8;
            uint4 h = *reinterpret_cast<const uint4 *>(a.hidden + row * 4096 + col), b{};
            if (a.branch)
                b = *reinterpret_cast<const uint4 *>(a.branch + row * 4096 + col);
            auto *hp = reinterpret_cast<unsigned *>(&h);
            auto *bp = reinterpret_cast<unsigned *>(&b);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(hp + k));
                float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(bp + k));
                __nv_bfloat162 z = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
                hp[k] = *reinterpret_cast<unsigned *>(&z);
            }
            values[j] = h;
            *reinterpret_cast<uint4 *>(residual + col) = h;
        }
    }
    __syncthreads();
    // Keep the original thread/element reduction order. Reuse the QKV scratch
    // for this local transpose; vector loads must not change RMS rounding.
    if (t < 256) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float value = float(residual[t + j * 256]);
            square += value * value;
        }
    }
    square = WarpSum(square);
    if (t < 256 && lane == 0)
        sums[t / 32] = square;
    __syncthreads();
    float total = WarpSum(t < 8 ? sums[t] : 0.f);
    if (t == 0)
        sums[0] = total;
    __syncthreads();
    float inv = rsqrtf(sums[0] / 4096 + a.eps);
    if (t < 256) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = (t + j * 256) * 8;
            uint4 result;
            auto *vp = reinterpret_cast<unsigned *>(&values[j]);
            auto *rp = reinterpret_cast<unsigned *>(&result);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(vp + k));
                __nv_bfloat162 z = __floats2bfloat162_rn(
                    RoundBF16(v.x * inv) * a.gamma[col + k * 2],
                    RoundBF16(v.y * inv) * a.gamma[col + k * 2 + 1]);
                rp[k] = *reinterpret_cast<unsigned *>(&z);
            }
            *reinterpret_cast<uint4 *>(a.normed + row * 4096 + col) = result;
            if (a.feature)
                *reinterpret_cast<uint4 *>(a.feature + row * 4096 + col) = values[j];
        }
    }
    __syncthreads();
    if (t == 0) {
        auto *p = reinterpret_cast<unsigned *>(a.epochs + 128 + row) + 1;
        fastllm_swa::PublishReady(p);
    }
}
// Borrow original row-major weights. Reuse each weight fragment across N8
// query tiles; each tile retains the original FP32 accumulation order.
template <int Threads, int Group, bool Ordered, int Chunks = 1, bool DirectInput = false>
__device__ __forceinline__ void Project(const Args &a, int first, unsigned char *inputTile, int tile) {
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, g = lane / 4, quad = lane % 4;
    const auto &batch = a.bindings;
    const int *live = batch.length;
    int rows = a.rows;
    auto *weight = a.weight;
    auto *q = a.q;
    auto *position = a.positions;
    float theta = a.theta, valueScale = a.scale;
    constexpr int StagingBytes = Ordered && DirectInput ? 0 : Chunks * InputBytes;
    auto partial = reinterpret_cast<float (*)[Threads / 32][16][8]>(inputTile + StagingBytes);
    __shared__ BF16 projected[Chunks][16][8];
    if constexpr (Ordered) {
        if (t < Chunks * 8 && first + t < rows) {
            auto *p = reinterpret_cast<unsigned *>(a.epochs + 128 + first + t) + 1;
            fastllm_swa::WaitReady(p);
        }
        __syncthreads();
        if constexpr (!DirectInput) {
            for (int i = t; i < Chunks * 8 * 512; i += Threads) {
                int row = i / 512, col = i % 512 * 8;
                unsigned dst = unsigned(__cvta_generic_to_shared(inputTile + (row / 8) * InputBytes + InputOffset(row % 8, col)));
                if (first + row < rows) {
                    const BF16 *src = a.normed + (first + row) * 4096 + col;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst), "l"(src) : "memory");
#endif
                } else
                    *reinterpret_cast<uint4 *>(inputTile + (row / 8) * InputBytes + InputOffset(row % 8, col)) = make_uint4(0, 0, 0, 0);
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile("cp.async.commit_group;cp.async.wait_group 0;" ::: "memory");
#endif
            __syncthreads();
        }
    } else
        Normalize<Threads, Group>(a, first, inputTile);
    float c[Chunks][4] = {};
#pragma unroll 4
    for (int col = warp * 32; col < 4096; col += Threads) {
        const int k = col + quad * 8;
        uint4 wa, wb;
        wa = *reinterpret_cast<const uint4 *>(weight + size_t(QkvColumn(tile, g)) * 4096 + k);
        wb = *reinterpret_cast<const uint4 *>(weight + size_t(QkvColumn(tile, g + 8)) * 4096 + k);
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            uint4 x;
            // Each normalized fragment is consumed once by this warp.
            if constexpr (Ordered && DirectInput) {
                const int row = first + chunk * 8 + g;
                x = row < rows ? *reinterpret_cast<const uint4 *>(a.normed + row * 4096 + k) : make_uint4(0, 0, 0, 0);
            } else
                x = *reinterpret_cast<const uint4 *>(inputTile + chunk * InputBytes + InputOffset(g, k));
            Step(c[chunk], wa, wb, x);
        }
    }
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        partial[chunk][warp][g][quad * 2] = c[chunk][0];
        partial[chunk][warp][g][quad * 2 + 1] = c[chunk][1];
        partial[chunk][warp][g + 8][quad * 2] = c[chunk][2];
        partial[chunk][warp][g + 8][quad * 2 + 1] = c[chunk][3];
    }
    __syncthreads();
    if (t < Chunks * 128) {
        const int chunk = t / 128, i = t % 128;
        float sum = 0;
#pragma unroll
        for (int w = 0; w < Threads / 32; ++w)
            sum += partial[chunk][w][i / 8][i % 8];
        projected[chunk][i / 8][i % 8] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    if (t >= Chunks * 128)
        return;
    const int chunk = t / 128, out = t % 128 / 8, r = t % 8, row = first + chunk * 8 + r;
    if (row >= rows)
        return;
    const int col = QkvColumn(tile, out);
    if (live[row] <= 0) {
        if (col < 1536)
            q[size_t(row) * 1536 + col] = __float2bfloat16_rn(0.f);
        return;
    }
    const auto &binding = batch.rows[row];
    BF16 *key = binding.cache ? (BF16 *)binding.cache[0] : binding.key;
    BF16 *value = binding.cache ? (BF16 *)binding.cache[1] : binding.value;
    const int target = 128 + binding.verifyOffset;
    if (tile < 36) {
        if (out >= 8)
            return;
        const int d = tile % 4 * 8 + out;
        const float angle = position[row] * powf(theta, -2.f * d / 64.f);
        const float cosine = RoundBF16(cosf(angle)), sine = RoundBF16(sinf(angle));
        const float a = float(projected[chunk][out][r]), b = float(projected[chunk][out + 8][r]);
        BF16 *dst = col < 1536 ? q + size_t(row) * 1536 + col : key + target * 192 + col - 1536;
        dst[0] = __float2bfloat16_rn(RoundBF16(a * cosine) - RoundBF16(b * sine));
        dst[32] = __float2bfloat16_rn(RoundBF16(b * cosine) + RoundBF16(a * sine));
    } else if (col < 1536) {
        q[size_t(row) * 1536 + col] = projected[chunk][out][r];
    } else if (col < 1728) {
        key[target * 192 + col - 1536] = projected[chunk][out][r];
    } else {
        value[target * 128 + col - 1728] =
            __float2bfloat16_rn(float(projected[chunk][out][r]) * RoundBF16(valueScale));
    }
}
#include "../cuda/models/naive-n05-swa-core.cuh"

template <int Threads, int Chunks = 1, bool DirectInput = false, bool Mx = false>
__device__ __forceinline__ void Tail(const Args &a, int first, const fastllm_mxfp8_projection::WeightFragment *weights = nullptr) {
    constexpr int KWarps = Threads / 64;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, g = lane / 4, quad = lane % 4;
    const int rows = min(Chunks * 8, a.rows - first), column = blockIdx.x * 32,
              group = warp / KWarps, part = warp % KWarps;
    extern __shared__ __align__(1024) unsigned char tile[];
    auto partial = reinterpret_cast<float (*)[2][KWarps][16][8]>(tile + (DirectInput ? 0 : Chunks * 16384));
    __shared__ BF16 local[Chunks][8][32];
    __shared__ uint32_t received[8][8][16], step[Chunks][8];
    if constexpr (!DirectInput) {
        for (int i = t; i < Chunks * 8 * 128; i += Threads) {
            int r = i / 128, col = i % 128 * 8;
            *reinterpret_cast<uint4 *>(tile + (r / 8) * 16384 + InputOffset(r % 8, col)) =
                r < rows ? *reinterpret_cast<const uint4 *>(a.attn + (first + r) * 1024 + col)
                         : make_uint4(0, 0, 0, 0);
        }
    }
    __nv_bfloat162 prior{};
    if constexpr (Mx) {
        if (t < rows * 16) {
            const int index = (first + t / 16) * 4096 + column + (t % 16) * 2;
            prior = *reinterpret_cast<const __nv_bfloat162 *>(a.hidden + index);
        }
    }
    if (t < min(8, rows)) {
        auto *generation = reinterpret_cast<uint32_t *>(a.epochs + t * 128 + blockIdx.x);
        uint32_t next = *generation;
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            step[chunk][t] = next;
            if (t + chunk * 8 < rows)
                next = uint64_t(next) + 1 == Period ? 0 : next + 1;
        }
        *generation = next;
    }
    __syncthreads();
    float c[Chunks][4] = {};
#pragma unroll
    for (int col = part * 32; col < 1024; col += KWarps * 32) {
        int k = col + quad * 8;
        uint4 w0, w1;
        if constexpr (!Mx) {
            w0 = *reinterpret_cast<const uint4 *>(a.oWeight + size_t(column + group * 16 + g) * 1024 + k);
            w1 = *reinterpret_cast<const uint4 *>(a.oWeight + size_t(column + group * 16 + g + 8) * 1024 + k);
        }
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            uint4 x;
            if constexpr (DirectInput) {
                const int row = chunk * 8 + g;
                x = row < rows ? *reinterpret_cast<const uint4 *>(a.attn + (first + row) * 1024 + k)
                               : make_uint4(0, 0, 0, 0);
            } else {
                x = *reinterpret_cast<const uint4 *>(tile + chunk * 16384 + InputOffset(g, k));
            }
            if constexpr (Mx) {
                fastllm_mxfp8_projection::Step(c[chunk], weights[col / (KWarps * 32)], x);
            } else {
                Step(c[chunk], w0, w1, x);
            }
        }
    }
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        partial[chunk][group][part][g][quad * 2] = c[chunk][0];
        partial[chunk][group][part][g][quad * 2 + 1] = c[chunk][1];
        partial[chunk][group][part][g + 8][quad * 2] = c[chunk][2];
        partial[chunk][group][part][g + 8][quad * 2 + 1] = c[chunk][3];
    }
    __syncthreads();
    for (int i = t; i < rows * 32; i += Threads) {
        int row = i / 32, col = i % 32;
        float sum = 0;
#pragma unroll
        for (int p = 0; p < KWarps; ++p)
            sum += partial[row / 8][col / 16][p][col % 16][row % 8];
        local[row / 8][row % 8][col] = __float2bfloat16_rn(sum);
    }
    __syncthreads();
    // Publish and consume each N8 chunk before reusing its mailbox slots.
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        const int chunkRows = min(8, rows - chunk * 8);
        if (chunkRows <= 0)
            continue;
        {
            for (int i = t; i < chunkRows * 8 * 16; i += Threads) {
                int row = i / 128, peer = i / 16 % 8, pair = i % 16, word = column / 2 + pair;
                size_t frame = (step[chunk][row] % Frames) * FrameBytes;
                uint32_t bits = reinterpret_cast<uint32_t *>(local[chunk][row])[pair];
                Store(a.peers.data[peer] + frame + row * 8 * SlotBytes + a.rank * SlotBytes + word * 8,
                      (uint64_t(bits) << 32) | (step[chunk][row] + 1));
            }
            // MX uses a single N8 frame. Each load waits on its own tagged
            // payload, including the local rank's publication.
            if constexpr (!Mx) __syncthreads();
            for (int i = t; i < chunkRows * 8 * 16; i += Threads) {
                int row = i / 128, peer = i / 16 % 8, pair = i % 16, word = column / 2 + pair;
                size_t frame = (step[chunk][row] % Frames) * FrameBytes;
                uint64_t data;
                do {
                    data = Load(a.peers.data[a.rank] + frame + row * 8 * SlotBytes + peer * SlotBytes + word * 8);
                } while (uint32_t(data) != step[chunk][row] + 1);
                received[row][peer][pair] = uint32_t(data >> 32);
            }
            __syncthreads();
        }
        for (int i = t; i < chunkRows * 16; i += Threads) {
            int row = i / 16, pair = i % 16;
            float x = 0, y = 0;
#pragma unroll
            for (int peer = 0; peer < 8; ++peer) {
                uint32_t bits = received[row][peer][pair];
                float a0 = float(__ushort_as_bfloat16(bits & 65535)),
                      a1 = float(__ushort_as_bfloat16(bits >> 16));
                if (peer == 0) {
                    x = a0;
                    y = a1;
                } else {
                    x += a0;
                    y += a1;
                }
            }
            int index = (first + chunk * 8 + row) * 4096 + column + pair * 2;
            if constexpr (Mx) {
                static_assert(Chunks == 1);
                const auto delta = __floats2bfloat162_rn(x, y);
                *reinterpret_cast<__nv_bfloat162 *>(a.projected + index) = delta;
                if (a.feature) *reinterpret_cast<__nv_bfloat162 *>(a.feature + index) = prior;
                *reinterpret_cast<__nv_bfloat162 *>(a.hidden + index) = __hadd2(prior, delta);
            } else {
                BF16 bx = __float2bfloat16_rn(x), by = __float2bfloat16_rn(y);
                a.projected[index] = bx;
                a.projected[index + 1] = by;
                float h0 = RoundBF16(float(a.hidden[index]) + (a.branch ? float(a.branch[index]) : 0.f)),
                      h1 = RoundBF16(float(a.hidden[index + 1]) + (a.branch ? float(a.branch[index + 1]) : 0.f));
                a.hidden[index] = __float2bfloat16_rn(h0 + float(bx));
                a.hidden[index + 1] = __float2bfloat16_rn(h1 + float(by));
            }
        }
        __syncthreads();
    }
}
// All QKV chunks complete before attention can consume a descriptor. This
// keeps candidate rows correct even when a request spans shuffled row chunks.
// Only the dependencies of the current attention head are polled. Every CTA
// must be resident; the launch adapter checks the complete TP group first.
template <int Threads, int Group, bool Small, int Chunks = 1, bool TensorQK = Small, bool DirectInput = false>
__global__ __launch_bounds__(Threads, 1) void Layer(
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
    __grid_constant__
#endif
        const Args a) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static_assert(!DirectInput || !Small, "direct QKV reads require normalized row owners");
    static_assert(Chunks == 1 || (!Small && Chunks == 2), "unsupported projection tile");
    extern __shared__ __align__(1024) unsigned char tile[];
    if constexpr (!Small) {
        if (blockIdx.x < a.rows) {
            if constexpr (DirectInput)
                NormalizeOwnerVector(a);
            else
                NormalizeOwner(a);
        }
    }
    if (blockIdx.x < 116) {
        for (int first = 0; first < (Small ? 8 : a.rows); first += Chunks * 8) {
            Project<Threads, Group, !Small, Chunks, DirectInput>(a, first, tile, blockIdx.x);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            auto *p = reinterpret_cast<unsigned *>(a.epochs + blockIdx.x + 1) + 1;
            fastllm_swa::PublishReady(p);
        }
    }
    for (int task = blockIdx.x; task < (Small ? 8 : a.rows) * 8; task += 128)
        if (task < a.rows * 8) {
            int t = threadIdx.x, head = task % 8, dependency = -1;
            if (t < 4)
                dependency = head * 4 + t;
            else if (t < 12)
                dependency = 36 + head * 8 + t - 4;
            else if (t < 16)
                dependency = 32 + t - 12;
            else if (t < 32)
                dependency = 100 + t - 16;
            if (dependency >= 0) {
                auto *p = reinterpret_cast<unsigned *>(a.epochs + dependency + 1) + 1;
                fastllm_swa::WaitReady(p);
            }
            __syncthreads();
            SwaAttentionBf16Body<Threads, 128, true, BatchSwaKeys, true, true, TensorQK>(
                a.q, nullptr, nullptr, a.sink, a.attn, a.bindings.length, 8, 1, a.bindings, task / 8, head,
                0);
            __syncthreads();
        }
    // Every input reader has finished before residual mutation or flag reuse.
    cooperative_groups::this_grid().sync();
    if (blockIdx.x == 0) {
        if (threadIdx.x < 116)
            reinterpret_cast<unsigned *>(a.epochs + threadIdx.x + 1)[1] = 0;
        if constexpr (!Small)
            if (threadIdx.x < a.rows)
                reinterpret_cast<unsigned *>(a.epochs + 128 + threadIdx.x)[1] = 0;
    }
    // Reuse the existing eight-row mailbox. Each query tile advances the low-word
    // tags; high words above are independent local QKV readiness flags.
    for (int first = 0; first < (Small ? 8 : a.rows); first += Chunks * 8)
        Tail<Threads, Chunks, DirectInput>(a, first);
#endif
}
// Packed native QKV needs no cross-CTA projection reduction. Stage the
// current head locally, and publish all candidate KV slots once per layer.
__device__ __forceinline__ void RotatePair(const BF16 *src, BF16 *dst, int d, const Args &a, int row) {
    float cosine, sine;
    if (a.rotary) {
        const float2 pair = __bfloat1622float2(a.rotary[row * 32 + d]);
        cosine = pair.x;
        sine = pair.y;
    } else {
        const float angle = a.positions[row] * powf(a.theta, -2.f * d / 64.f);
        cosine = RoundBF16(cosf(angle));
        sine = RoundBF16(sinf(angle));
    }
    const float x = float(src[d]), y = float(src[d + 32]);
    dst[d] = __float2bfloat16_rn(RoundBF16(x * cosine) - RoundBF16(y * sine));
    dst[d + 32] = __float2bfloat16_rn(RoundBF16(y * cosine) + RoundBF16(x * sine));
}

// The QKV producer updates projected, hidden and the rotary table. Its
// ordinary-stream preparation has already ordered the request descriptors
// and committed ring. Only the candidate slots depend on this layer's QKV.
template<int Tile>
__device__ __forceinline__ void StageHistory(const Args &a, int row, int valueTile) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (row >= a.rows) return;
    const int length = a.bindings.Live(row);
    const auto &binding = a.bindings.rows[row];
    const int candidates = binding.verifyOffset + 1;
    if (length <= candidates) return;
    const int committed = min(128, length) - candidates;
    const int first = max(0, length - 128);
    extern __shared__ __align__(1024) unsigned char storage[];
    auto *values = reinterpret_cast<BF16 *>(storage);
    const auto *v = binding.cache ? static_cast<const BF16 *>(binding.cache[1]) : binding.value;
    for (int i = threadIdx.x; i < committed * (Tile / 8); i += blockDim.x) {
        const int slot = i / (Tile / 8), vector = i % (Tile / 8);
        const int physical = (first + slot) & 127;
        const int col = (vector ^ (slot & (Tile / 8 - 1))) * 8;
        unsigned dst = __cvta_generic_to_shared(values + slot * Tile + col);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),
                     "l"(v + physical * 128 + valueTile * Tile + vector * 8) : "memory");
    }
    if (committed >= PackedSwaStagedKeys) {
        auto *keys = values + 128 * Tile;
        const auto *k = binding.cache ? static_cast<const BF16 *>(binding.cache[0]) : binding.key;
        for (int i = threadIdx.x; i < committed * 24; i += blockDim.x) {
            const int slot = i / 24, col = i % 24 * 8;
            unsigned dst = __cvta_generic_to_shared(keys + i * 8);
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),
                         "l"(k + ((first + slot) & 127) * 192 + col) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template<int Threads>
__global__ __launch_bounds__(Threads, 1) void PackedLayer(
#if CUDART_VERSION >= 12010 && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
    __grid_constant__
#endif
        const Args a) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    const int t = threadIdx.x, task = blockIdx.x;
    const int row = task / 16, head = task / 2 % 8, valueTile = task % 2;
    StageHistory<64>(a, row, valueTile);
    constexpr int KWarps = Threads / 64;
    constexpr int Parts = 1024 / (KWarps * 32);
    fastllm_mxfp8_projection::WeightFragment weights[Parts];
    const int warp = t / 32, lane = t % 32;
    const int out = task * 32 + warp / KWarps * 16 + lane / 4;
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
        const int k = (warp % KWarps + part * KWarps) * 32 + lane % 4 * 8;
        weights[part] = fastllm_mxfp8_projection::LoadWeights(a.oWeight, a.outputScales, 1024, out, out + 8, k);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (a.rotary) cudaGridDependencySynchronize();
#endif
    auto flag = [&](int r) { return reinterpret_cast<unsigned *>(a.epochs + 1 + r) + 1; };
    __shared__ __align__(16) BF16 query[192];
    if (row < a.rows) {
        // The first head publishes this row's candidate KV. Separate warps
        // rotate Q and write KV, then share the barrier before the ready flag.
        const bool writesKV = head == 0 && valueTile == 0;
        const int kvLane = t - 192;
        if (writesKV && kvLane >= 0 && kvLane < 288 && a.bindings.length[row] > 0) {
            const auto &binding = a.bindings.rows[row];
            const int slot = 128 + binding.verifyOffset;
            const BF16 *kv = a.projected + row * 1856 + 1536;
            if (kvLane < 160) {
                auto *key = (binding.cache ? (BF16 *)binding.cache[0] : binding.key) + slot * 192;
                if (kvLane < 32) RotatePair(kv, key, kvLane, a, row);
                else key[kvLane + 32] = kv[kvLane + 32];
            } else {
                const int col = kvLane - 160;
                auto *value = binding.cache ? (BF16 *)binding.cache[1] : binding.value;
                value[slot * 128 + col] = __float2bfloat16_rn(float(kv[192 + col]) * RoundBF16(a.scale));
            }
        }
        const BF16 *src = a.projected + row * 1856 + head * 192;
        if (t < 192) {
            if (a.bindings.length[row] <= 0) query[t] = __float2bfloat16_rn(0.f);
            else if (t < 32) RotatePair(src, query, t, a, row);
            else if (t >= 64) query[t] = src[t];
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();
        if (writesKV && t == 0)
            fastllm_swa::PublishReady(flag(row));
        if (!valueTile && t < 24)
            reinterpret_cast<uint4 *>(a.q + (row * 8 + head) * 192)[t] = reinterpret_cast<const uint4 *>(query)[t];
        SwaAttentionBf16Body<Threads, 64, true, BatchSwaKeys, true, true, false, true>(
            a.q, nullptr, nullptr, a.sink, a.attn, a.bindings.length, 8, 1, a.bindings, row, head, valueTile, query, a.epochs, a.rows);
    }
    // All candidate KV and QKV reads finish before Tail reuses projected.
    cooperative_groups::this_grid().sync();
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (a.overlapRouter) cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (task == 127 && t < a.rows) *flag(t) = 0;
    Tail<Threads, 1, true, true>(a, 0, weights);
#endif
}
inline int Threads(int rows) {
    return rows <= 8 ? 1024 : 512;
}
inline int SharedBytes(int rows, bool wide = false, bool mx = false) {
    if (mx) return 128 * (64 + 192) * sizeof(BF16);
    if (wide && rows > 8) {
        // Direct QKV/O reads only need reduction scratch; attention reuses the
        // same storage for its 128x128 BF16 value tile.
        constexpr int attention = 128 * 128 * sizeof(BF16);
        const int partial = 2 * Threads(rows) / 32 * 16 * 8 * sizeof(float);
        return attention > partial ? attention : partial;
    }
    return InputBytes + Threads(rows) / 32 * 16 * 8 * sizeof(float);
}
inline const void *Kernel(int rows, bool wide = false, bool mx = false) {
    if (mx) return (const void *)PackedLayer<1024>;
    // Two N8 tiles share QKV/O weight loads and read graph-owned inputs directly.
    // The original staging schedule remains available when admission fails.
    if (wide && rows > 8)
        return (const void *)Layer<512, 128, false, 2, true, true>;
    return rows <= 8 ? (const void *)Layer<1024, 128, true> : (const void *)Layer<512, 128, false>;
}
static_assert(128 + RowLimit <= MaxRows * MaxBlocks, "readiness flags must fit existing epochs");
} // namespace fastllm_swa_verify
