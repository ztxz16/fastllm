#pragma once
// SM103 TP8 SWA verification. Admission and fixed-Graph preparation are
// handled by the shared attention-output state before launching this kernel.
// Borrow original BF16 weights and existing graph-owned output/mailbox storage.
// Rows are flattened queries, with per-request candidate offsets in [0,7].
#include <cooperative_groups.h>
#include "fastllm-swa-output.cuh"
#include "../cuda/models/naive-n05-swa-mma.cuh"
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
    const float *gamma, *positions, *sink;
    float eps, theta, scale;
    int rows, rank;
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
        asm volatile("st.release.gpu.global.u32 [%0],1;" ::"l"(p) : "memory");
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
        asm volatile("st.release.gpu.global.u32 [%0],1;" ::"l"(p) : "memory");
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
            unsigned flag;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(flag) : "l"(p) : "memory");
            } while (!flag);
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

template <int Threads, int Chunks = 1, bool DirectInput = false>
__device__ __forceinline__ void Tail(const Args &a, int first) {
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
        uint4 w0 = *reinterpret_cast<const uint4 *>(a.oWeight + size_t(column + group * 16 + g) * 1024 + k);
        uint4 w1 = *reinterpret_cast<const uint4 *>(a.oWeight + size_t(column + group * 16 + g + 8) * 1024 + k);
#pragma unroll
        for (int chunk = 0; chunk < Chunks; ++chunk) {
            uint4 x;
            if constexpr (DirectInput) {
                const int row = chunk * 8 + g;
                x = row < rows ? *reinterpret_cast<const uint4 *>(a.attn + (first + row) * 1024 + k) : make_uint4(0, 0, 0, 0);
            } else
                x = *reinterpret_cast<const uint4 *>(tile + chunk * 16384 + InputOffset(g, k));
            Step(c[chunk], w0, w1, x);
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
            __syncthreads();
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
            BF16 bx = __float2bfloat16_rn(x), by = __float2bfloat16_rn(y);
            a.projected[index] = bx;
            a.projected[index + 1] = by;
            float h0 = RoundBF16(float(a.hidden[index]) + (a.branch ? float(a.branch[index]) : 0.f)),
                  h1 = RoundBF16(float(a.hidden[index + 1]) + (a.branch ? float(a.branch[index + 1]) : 0.f));
            a.hidden[index] = __float2bfloat16_rn(h0 + float(bx));
            a.hidden[index + 1] = __float2bfloat16_rn(h1 + float(by));
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
    const __grid_constant__
#endif
        Args a) {
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
            asm volatile("st.release.gpu.global.u32 [%0],1;" ::"l"(p) : "memory");
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
                unsigned flag;
                do {
                    asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(flag) : "l"(p) : "memory");
                } while (!flag);
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
}
inline int Threads(int rows) {
    return rows <= 8 ? 1024 : 512;
}
inline int SharedBytes(int rows, bool wide = false) {
    if (wide && rows > 8) {
        // Direct QKV/O reads only need reduction scratch; attention reuses the
        // same storage for its 128x128 BF16 value tile.
        constexpr int attention = 128 * 128 * sizeof(BF16);
        const int partial = 2 * Threads(rows) / 32 * 16 * 8 * sizeof(float);
        return attention > partial ? attention : partial;
    }
    return InputBytes + Threads(rows) / 32 * 16 * 8 * sizeof(float);
}
inline const void *Kernel(int rows, bool wide = false) {
    // Two N8 tiles share QKV/O weight loads and read graph-owned inputs directly.
    // The original staging schedule remains available when admission fails.
    if (wide && rows > 8)
        return (const void *)Layer<512, 128, false, 2, true, true>;
    return rows <= 8 ? (const void *)Layer<1024, 128, true> : (const void *)Layer<512, 128, false>;
}
static_assert(128 + RowLimit <= MaxRows * MaxBlocks, "readiness flags must fit existing epochs");
} // namespace fastllm_swa_verify
