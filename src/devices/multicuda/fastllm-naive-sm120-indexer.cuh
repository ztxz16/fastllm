#pragma once

// The full Indexer is replicated, while each GPU owns one eighth of every
// logical KV page. Exchange a sorted list per owner through the same registered
// TP mailbox used later by the output projection. No additional collective node.
namespace sm120_verify {
constexpr int IndexProjectionTiles = (2048 + 128 + 16) / 16;
__device__ __forceinline__ void StoreIndexProjection(const Args &a, int tile, int first,
                                                     const float (&acc)[4], float (&part)[16][128]) {
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, quad = t % 4;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        part[warp][(g + j / 2 * 8) * 8 + quad * 2 + j % 2] = acc[j];
    __syncthreads();
    if (t < 128 && first + t % 8 < a.rows) {
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
            sum += part[j][t];
        BF rounded = __float2bfloat16_rn(sum);
        a.indexRaw[(first + t % 8) * 2192 + tile * 16 + t / 8] =
            tile == 136 ? __float2bfloat16_rn(float(rounded) * .25f) : rounded;
    }
}

__device__ void PrepareIndex(const Args &a, float *mem) {
    const int t = threadIdx.x;
    for (int work = blockIdx.x; work < a.rows * 17; work += gridDim.x) {
        const int row = work / 17, head = work % 17;
        const auto &r = a.batch.rows[row];
        const int live = *r.liveKeys, page = r.pageLen, localPage = page / 8;
        if (live <= 0 || live > r.capacity || (head == 16 && (live - 1) % page / localPage != a.rank))
            continue;
        const BF *input = a.indexRaw + row * 2192 + head * 128;
        float x = t < 128 ? float(input[t]) : 0.f;
        if (head == 16) {
            if (t < 64) {
                float z = float(input[t + 64]);
                mem[t] = x + z;
                mem[64 + t] = x * x + z * z;
            }
            __syncthreads();
            for (int d = 32; d; d /= 2) {
                if (t < d) {
                    mem[t] += mem[t + d];
                    mem[64 + t] += mem[64 + t + d];
                }
                __syncthreads();
            }
            const float mean = mem[0] / 128;
            const float deviation = sqrt((mem[64] + mean * mean * 128 - 2 * mean * 128 * mean) / 128 + 1.e-10);
            if (t < 128)
                x = CustomArNormRound((x - mean) / deviation * a.indexNorm[t] + a.indexBias[t]);
            __syncthreads();
        }
        if (t < 128)
            mem[t] = x;
        __syncthreads();
        if (t < 64) {
            const float angle = a.pos[row] * powf(a.theta, -2.f * (t % 32) / 64.f);
            x = CustomArNormRound(CustomArNormRound(x * CustomArNormRound(cosf(angle))) +
                                  (t < 32 ? -1.f : 1.f) *
                                      CustomArNormRound(mem[t ^ 32] * CustomArNormRound(sinf(angle))));
        }
        __syncthreads();
        if (t < 128)
            mem[t] = fabsf(x);
        __syncthreads();
        for (int d = 64; d; d /= 2) {
            if (t < d)
                mem[t] = fmaxf(mem[t], mem[t + d]);
            __syncthreads();
        }
        const float scale = fmaxf(mem[0], 1.e-4f) / 448.f;
        if (head < 16) {
            if (t == 0)
                a.iqScale[row * 16 + head] = scale;
            if (t < 128)
                a.iqBytes[(row * 16 + head) * 128 + t] = __nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x / scale))).__x;
        } else {
            const int logical = live - 1;
            int physical = static_cast<const int *>(r.cache[2])[logical / page] * localPage + logical % localPage;
            auto *dst =
                (uint8_t *)r.cache[4] + size_t(physical) * (a.scaledIndex ? FastllmNaiveIndexerBytes : 128);
            if (t < 128)
                dst[t] = __nv_fp8_e4m3(a.scaledIndex ? fminf(448.f, fmaxf(-448.f, x / scale)) : x).__x;
            if (!t && a.scaledIndex) {
                *reinterpret_cast<float *>(dst + 128) = scale;
                *reinterpret_cast<unsigned *>(dst + 132) = 0;
                *reinterpret_cast<uint64_t *>(dst + 136) = 0;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int ShardCount(int live, int page, int rank) {
    const int step = page / 8;
    return (live / page) * step + max(0, min(step, live % page - rank * step));
}
__device__ __forceinline__ uint4 IndexFragment(const uint8_t *ptr, int k, int quad) {
    FastllmNaiveKVRow row{ptr + k + quad * 2, true};
    auto pair = [&](int off) {
        auto val = __floats2bfloat162_rn(row.LoadFloat(off), row.LoadFloat(off + 1));
        return *reinterpret_cast<unsigned *>(&val);
    };
    return make_uint4(pair(0), pair(8), pair(16), pair(24));
}
__device__ void ScoreIndex(const Args &a) {
    const int lane = threadIdx.x % 32, g = lane / 4, quad = lane % 4, tiles = a.shardWidth / 8;
    for (int work = blockIdx.x * 16 + threadIdx.x / 32; work < a.rows * tiles; work += gridDim.x * 16) {
        int row = work / tiles, first = work % tiles * 8;
        auto &r = a.batch.rows[row];
        int step = r.pageLen / 8;
        int count = ShardCount(max(0, min(r.capacity, *r.liveKeys)), r.pageLen, a.rank);
        if (first >= count)
            continue;
        auto key = [&](int idx) {
            idx = min(idx, count - 1);
            int phys = static_cast<const int *>(r.cache[6])[idx / step] * step + idx % step;
            return static_cast<const uint8_t *>(r.cache[4]) +
                   size_t(phys) * (a.scaledIndex ? FastllmNaiveIndexerBytes : 128);
        };
        float dots[4]{};
#pragma unroll
        for (int k = 0; k < 128; k += 32) {
            auto q = a.iqBytes + (row * 16 + g) * 128;
            fastllm_swa_mma::Step(dots, IndexFragment(q, k, quad), IndexFragment(q + 8 * 128, k, quad),
                                  IndexFragment(key(first + g), k, quad));
        }
        float score[2]{};
#pragma unroll
        for (int h = 0; h < 16; ++h) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float value = __shfl_sync(0xffffffff, dots[h / 8 * 2 + i], (h % 8) * 4 + quad);
                score[i] = fmaf(fmaxf(value * a.iqScale[row * 16 + h], 0.f),
                                float(a.indexRaw[row * 2192 + 2176 + h]), score[i]);
            }
        }
        if (lane < 4)
            for (int i = 0; i < 2; ++i) {
                int idx = first + lane * 2 + i;
                if (idx >= count)
                    continue;
                float scale = a.scaledIndex ? *reinterpret_cast<const float *>(key(idx) + 128) : 1.f;
                float x = score[i] * scale;
                unsigned bits = x == 0.f ? 0 : __float_as_uint(x);
                a.indexScores[size_t(row) * a.shardWidth + idx] = bits >> 31 ? ~bits : (bits ^ 0x80000000u);
            }
    }
}
template <int Items> using ShardSort = cub::BlockRadixSort<unsigned, Threads, Items, int>;

template <int Items, bool LimitStores = false>
__device__ void SortIndexTile(const Args &a, int row, int n, int first, unsigned long long *output,
                              typename ShardSort<Items>::TempStorage &storage) {
    const int page = a.batch.rows[row].pageLen, step = page / 8;
    unsigned keys[Items];
    int positions[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = first + threadIdx.x * Items + j;
        keys[j] = i < n ? a.indexScores[size_t(row) * a.shardWidth + i] : 0;
        positions[j] = i < n ? i / step * page + a.rank * step + i % step : -1;
    }
    // CUB's stable sort preserves ascending logical position for equal scores.
    ShardSort<Items>(storage).SortDescendingBlockedToStriped(keys, positions);
#pragma unroll
    for (int j = 0; j < Items; ++j)
        if (!LimitStores || first + int(threadIdx.x) + j * Threads < min(n, 2048))
            output[threadIdx.x + j * Threads] =
                (static_cast<unsigned long long>(keys[j]) << 32) | ~unsigned(positions[j]);
}

// The sorting CTA publishes one row, then copies that row from its peers.
// Each row uses its own transport slot, allowing ranks to arrive independently.
union IndexSortStorage {
    ShardSort<1>::TempStorage shortList;
    ShardSort<2>::TempStorage fullList;
    ShardSort<4>::TempStorage topK;
    ShardSort<5>::TempStorage overflow;
};
__device__ void PublishIndexRow(const Args &a, int row, int n, IndexSortStorage &sort) {
    auto *dst = reinterpret_cast<unsigned long long *>(a.projection) + row * 1024;
    if (n <= 512)
        SortIndexTile<1>(a, row, n, 0, dst, sort.shortList);
    else
        SortIndexTile<2>(a, row, n, 0, dst, sort.fullList);
}

template<bool CooperativeSearch = false>
__device__ __forceinline__ unsigned long long MergeIndexWarp(const unsigned long long *left, int na,
                                                             const unsigned long long *right, int nb, int first) {
    const int lane = threadIdx.x % 32;
    unsigned long long value = 0;
    if (first < na + nb) {
        int pivot = 0;
        if constexpr (CooperativeSearch) {
            // Probe ordered, evenly spaced diagonals together. A ballot gives
            // the next lower bound with at most three rounds for K <= 2048.
            int lo = max(0, first - nb), hi = min(first, na);
            while (lo < hi) {
                const int step = (hi - lo + 31) / 32;
                const int m = lo + (lane + 1) * step - 1;
                const bool lower = m < hi && left[m] > right[first - m - 1];
                lo += __popc(__ballot_sync(0xffffffff, lower)) * step;
                hi = min(hi, lo + step - 1);
            }
            pivot = lo;
        } else {
            if (!lane) {
                int lo = max(0, first - nb), hi = min(first, na);
                while (lo < hi) {
                    const int m = (lo + hi) / 2;
                    if (left[m] > right[first - m - 1])
                        lo = m + 1;
                    else
                        hi = m;
                }
                pivot = lo;
            }
            pivot = __shfl_sync(0xffffffff, pivot, 0);
        }
        const int ia = pivot + lane, ib = first - pivot + 31 - lane;
        const auto ka = ia < na ? left[ia] : 0ull;
        const auto kb = ib < nb ? right[ib] : 0ull;
        value = max(ka, kb);
#pragma unroll
        for (int d = 16; d; d >>= 1) {
            const auto peer = __shfl_xor_sync(0xffffffff, value, d);
            value = lane & d ? min(value, peer) : max(value, peer);
        }
    }
    return value;
}

template <int Sources, bool Owned = false>
__device__ void MergeIndexPairs(const Args &a, const unsigned long long *input, unsigned long long *output) {
    constexpr int Pairs = 4 / Sources;
    const int inputStride = Sources == 1 ? min(2048, a.shardWidth) : 2048;
    constexpr int OutputStride = 2048, Warps = Threads / 32;
    const int lane = threadIdx.x % 32;
    for (int work = blockIdx.x * Warps + threadIdx.x / 32; work < (Owned ? 1 : a.rows) * Pairs * (OutputStride / 32);
         work += gridDim.x * Warps) {
        const int pair = (work / (OutputStride / 32)) % Pairs;
        const int row = Owned ? a.rank : work / (Pairs * (OutputStride / 32));
        const int first = work % (OutputStride / 32) * 32;
        const auto &r = a.batch.rows[row];
        const int live = max(0, min(r.capacity, *r.liveKeys));
        const int span = (r.pageLen / 8) * Sources, base = (live / r.pageLen) * span;
        const int tail = live % r.pageLen - pair * 2 * span;
        const int na = min(inputStride, base + max(0, min(span, tail)));
        const int nb = min(inputStride, base + max(0, min(span, tail - span)));
        const auto *left = input + row * (Sources == 1 ? 8 * inputStride : 8192) + 2 * pair * inputStride;
        const auto *right = left + inputStride;
        const auto value = MergeIndexWarp<Owned>(left, na, right, nb, first);
        if constexpr (Sources == 4)
            a.selected[row * OutputStride + first + lane] = value ? int(~unsigned(value)) : -1;
        else if (first < na + nb)
            output[row * 8192 + pair * OutputStride + first + lane] = value;
    }
}
} // namespace sm120_verify
