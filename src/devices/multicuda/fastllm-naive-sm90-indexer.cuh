#pragma once

// The full Indexer is replicated, while each GPU owns one eighth of every
// logical KV page. Exchange a sorted list per owner through the same registered
// TP mailbox used later by the output projection. No additional collective node.
namespace hopper_verify {
template <bool Wide, bool KeyOnly = false> __device__ void ProjectIndex(const Args &a, float (&part)[16][128]) {
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, quad = t % 4;
    // Dense attention only needs the cached 128-channel key for later sparse replays.
    constexpr int Tiles = KeyOnly ? 8 : 137; // 2048 query + 128 key + 16 head weights
    constexpr int Groups = Wide ? 2 : 1;
    for (int task = blockIdx.x; task < (Wide ? Tiles : ((a.rows + 7) / 8) * Tiles); task += gridDim.x) {
        const int tile = (KeyOnly ? 128 : 0) + task % Tiles, first = task / Tiles * 8;
        const BF *w = tile < 128 ? a.iqWeight : tile < 136 ? a.ikWeight : a.iwWeight;
        const int base = (tile < 128 ? tile : tile < 136 ? tile - 128 : 0) * 16;
        float acc[Groups][4]{};
        for (int group = warp; group < 128; group += 16) {
            int k = group * 32 + quad * 8;
            const uint4 lo = *reinterpret_cast<const uint4 *>(w + (base + g) * 4096 + k);
            const uint4 hi = *reinterpret_cast<const uint4 *>(w + (base + g + 8) * 4096 + k);
#pragma unroll
            for (int half = 0; half < Groups; ++half) {
                const int row = first + half * 8 + g;
                const uint4 x = row < a.rows ? *reinterpret_cast<const uint4 *>(a.input + row * 4096 + k)
                                             : make_uint4(0, 0, 0, 0);
                fastllm_swa_mma::Step(acc[half], lo, hi, x);
            }
        }
#pragma unroll
        for (int half = 0; half < Groups; ++half) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                part[warp][(g + j / 2 * 8) * 8 + quad * 2 + j % 2] = acc[half][j];
            __syncthreads();
            if (t < 128 && first + half * 8 + t % 8 < a.rows) {
                float sum = 0.f;
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    sum += part[j][t];
                BF rounded = __float2bfloat16_rn(sum);
                a.indexRaw[(first + half * 8 + t % 8) * 2192 + tile * 16 + t / 8] =
                    tile == 136 ? __float2bfloat16_rn(float(rounded) * .25f) : rounded;
            }
            __syncthreads();
        }
    }
}

template<bool Replicated = false, bool KeyOnly = false> __device__ void PrepareIndex(const Args &a, float *mem) {
    const int t = threadIdx.x;
    constexpr int HeadsPerRow = KeyOnly ? 1 : 17;
    for (int work = blockIdx.x; work < a.rows * HeadsPerRow; work += gridDim.x) {
        const int row = work / HeadsPerRow, head = KeyOnly ? 16 : work % HeadsPerRow;
        const auto &r = a.Row(row);
        const int live = *r.liveKeys, page = r.pageLen, localPage = page / 8;
        if (live <= 0 || live > r.capacity || (!Replicated && head == 16 && (live - 1) % page / localPage != a.rank))
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
        if constexpr (Replicated) {
            if (head == 16) {
                if (t < 128) {
                    auto cache = FastllmNaiveResolveDSARow(r);
                    auto *dest = static_cast<BF *>(const_cast<void *>(cache.key)) +
                        size_t(FastllmNaivePhysicalRow(cache, live - 1)) * cache.keyStride + QDim;
                    dest[t] = __float2bfloat16_rn(x);
                }
                __syncthreads();
                continue;
            }
        }
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
            if (t < 128) {
                auto q = __nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x / scale)));
                if constexpr (Replicated)
                    a.indexRaw[row * 2192 + head * 128 + t] = __float2bfloat16_rn(float(q));
                else
                    a.iqBytes[(row * 16 + head) * 128 + t] = q.__x;
            }
        } else {
            const int logical = live - 1;
            int physical =
                static_cast<const int *>(r.cache[2])[logical / page] * localPage + logical % localPage;
            auto *dst = (uint8_t *)r.cache[4] + size_t(physical) * FastllmNaiveIndexerBytes;
            if (t < 128)
                dst[t] = __nv_fp8_e4m3(fminf(448.f, fmaxf(-448.f, x / scale))).__x;
            if (!t) {
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
__device__ __forceinline__ uint2 Fragment(const uint8_t *ptr, int k, int quad) {
    // MMA row-major A and column-major B both use K[quad*4 : +4] and K[+16 : +20].
    return make_uint2(*reinterpret_cast<const unsigned *>(ptr + k + quad * 4),
                      *reinterpret_cast<const unsigned *>(ptr + k + quad * 4 + 16));
}
__device__ __forceinline__ void Dot(float (&c)[4], uint2 lo, uint2 hi, uint2 q) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(lo.x), "r"(hi.x), "r"(lo.y), "r"(hi.y), "r"(q.x), "r"(q.y));
#endif
}
__device__ void ScoreIndex(const Args &a) {
    const int lane = threadIdx.x % 32, g = lane / 4, quad = lane % 4, tiles = ShardWidth / 16;
    for (int work = blockIdx.x * 16 + threadIdx.x / 32; work < a.rows * tiles; work += gridDim.x * 16) {
        const int row = work / tiles, first = work % tiles * 16;
        const auto &r = a.Row(row);
        const int page = r.pageLen, step = page / 8,
                  n = ShardCount(max(0, min(r.capacity, *r.liveKeys)), page, a.rank);
        if (first >= n)
            continue;
        auto key = [&](int idx) {
            idx = min(idx, n - 1);
            int phys = static_cast<const int *>(r.cache[2])[idx / step] * step + idx % step;
            return (const uint8_t *)r.cache[4] + size_t(phys) * FastllmNaiveIndexerBytes;
        };
        auto *lo = key(first + g), *hi = key(first + g + 8);
        float dot[2][4]{};
#pragma unroll
        for (int k = 0; k < 128; k += 32) {
            uint2 l = Fragment(lo, k, quad), h = Fragment(hi, k, quad);
#pragma unroll
            for (int group = 0; group < 2; ++group)
                Dot(dot[group], l, h, Fragment(a.iqBytes + (row * 16 + group * 8 + g) * 128, k, quad));
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            float sum = 0.f;
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                int h = group * 8 + quad * 2;
                const float w0 = float(a.indexRaw[row * 2192 + 2176 + h]),
                            w1 = float(a.indexRaw[row * 2192 + 2176 + h + 1]);
                const float v0 = fmaxf(0.f, dot[group][half * 2] * a.iqScale[row * 16 + h]);
                const float v1 = fmaxf(0.f, dot[group][half * 2 + 1] * a.iqScale[row * 16 + h + 1]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float pair = fmaf(v1, w1, fmaf(v0, w0, sum));
                    sum = __shfl_sync(0xffffffff, pair, (lane & ~3) + j);
                }
            }
            const int idx = first + g + half * 8;
            if (!quad && idx < n) {
                const float keyScale = *reinterpret_cast<const float *>((half ? hi : lo) + 128);
                float s = sum * keyScale;
                unsigned bits = s == 0.f ? 0 : __float_as_uint(s);
                a.indexScores[row * ShardWidth + idx] = (bits >> 31) ? ~bits : (bits ^ 0x80000000u);
            }
        }
    }
}
template <int Items> using ShardSort = cub::BlockRadixSort<unsigned, Threads, Items, int>;

template <int Items>
__device__ void PublishIndexRow(const Args &a, int row, int n,
                                typename ShardSort<Items>::TempStorage &storage) {
    int keysPage = a.Row(row).pageLen;
    unsigned keys[Items];
    int positions[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int i = threadIdx.x * Items + j, step = keysPage / 8;
        keys[j] = i < n ? a.indexScores[row * ShardWidth + i] : 0;
        positions[j] = i < n ? i / step * keysPage + a.rank * step + i % step : -1;
    }
    ShardSort<Items>(storage).SortDescendingBlockedToStriped(keys, positions);
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        int i = threadIdx.x + j * Threads;
        reinterpret_cast<unsigned long long *>(a.projection)[row * 1024 + i] =
            (static_cast<unsigned long long>(keys[j]) << 32) | ~unsigned(positions[j]);
    }
}

// The sorting CTA publishes one row, then copies that row from its peers.
// Each row uses its own transport slot, allowing ranks to arrive independently.
__device__ void ExchangeIndex(const Args &a) {
    __shared__ union {
        ShardSort<1>::TempStorage shortList;
        ShardSort<2>::TempStorage fullList;
    } sort;
    const int slots = a.signalSlots;
    if (blockIdx.x >= slots) return;
    for (int row = blockIdx.x; row < a.rows; row += slots) {
        const auto &r = a.Row(row);
        const int live = max(0, min(r.capacity, *r.liveKeys)), n = ShardCount(live, r.pageLen, a.rank);
        if (n <= 512)
            PublishIndexRow<1>(a, row, n, sort.shortList);
        else
            PublishIndexRow<2>(a, row, n, sort.fullList);
        __threadfence_system();
        __syncthreads();
        CustomArBarrierStart<8>(a.signals, a.signals.signals[a.rank], a.rank);
        const auto peers = a.peers.Load();
        for (int i = threadIdx.x; i < 8192; i += Threads) {
            int owner = i / 1024, pos = i % 1024;
            a.lists[row * 8192 + i] =
                pos < ShardCount(live, r.pageLen, owner)
                    ? reinterpret_cast<const unsigned long long *>(peers.ptrs[owner])[row * 1024 + pos]
                    : 0;
        }
        CustomArBarrierFinal<8>(a.signals, a.signals.signals[a.rank], a.rank);
        __syncthreads();
    }
}

// One warp merges the next 32 items from two sorted lists. The 64-bit key
// carries both the score and the logical position, including the tie order.
template<int Sources>
__device__ void MergeIndexPairs(const Args &a, const unsigned long long *input,
                                unsigned long long *output) {
    constexpr int Pairs = 4 / Sources;
    constexpr int InputStride = Sources == 1 ? 1024 : 2048;
    constexpr int OutputStride = 2048, Warps = Threads / 32;
    const int lane = threadIdx.x % 32;
    for (int work = blockIdx.x * Warps + threadIdx.x / 32;
         work < a.rows * Pairs * (OutputStride / 32); work += gridDim.x * Warps) {
        const int pair = (work / (OutputStride / 32)) % Pairs;
        const int row = work / (Pairs * (OutputStride / 32));
        const int first = work % (OutputStride / 32) * 32;
        const auto &r = a.Row(row);
        const int live = max(0,min(r.capacity,*r.liveKeys));
        const int span = (r.pageLen / 8) * Sources, base = (live / r.pageLen) * span;
        const int tail = live % r.pageLen - pair * 2 * span;
        const int na = min(InputStride,base + max(0,min(span,tail)));
        const int nb = min(InputStride,base + max(0,min(span,tail-span)));
        const auto *left = input + row * 8192 + 2 * pair * InputStride;
        const auto *right = left + InputStride;
        unsigned long long value = 0;
        if (first < na + nb) {
            int pivot = 0;
            if (!lane) {
                int lo = max(0,first - nb), hi = min(first,na);
                while (lo < hi) {
                    const int m = (lo + hi) / 2;
                    if (left[m] > right[first-m-1]) lo = m+1;
                    else hi = m;
                }
                pivot = lo;
            }
            pivot = __shfl_sync(0xffffffff,pivot,0);
            const int ia = pivot+lane, ib = first-pivot+31-lane;
            const auto ka = ia < na ? left[ia] : 0ull;
            const auto kb = ib < nb ? right[ib] : 0ull;
            value = max(ka,kb);
#pragma unroll
            for (int d = 16; d; d >>= 1) {
                const auto peer = __shfl_xor_sync(0xffffffff,value,d);
                value = lane & d ? min(value,peer) : max(value,peer);
            }
        }
        if constexpr (Sources == 4)
            a.selected[row * OutputStride + first + lane] = value ? int(~unsigned(value)) : -1;
        else if (first < na + nb)
            output[row * 8192 + pair * OutputStride + first + lane] = value;
    }
}
} // namespace hopper_verify
