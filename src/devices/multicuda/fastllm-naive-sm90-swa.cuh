#pragma once

// SM90 SWA verification. Included inside hopper_verify after the KV accessors.
__device__ void RingRotary(const Args &a) {
    for (int i = blockIdx.x * Threads + threadIdx.x; i < a.rows * 32; i += gridDim.x * Threads) {
        const int row = i / 32, dim = i % 32;
        const float angle = a.pos[row] * powf(a.theta, -2.f * dim / 64.f);
        reinterpret_cast<float2 *>(a.scores + 2)[i] =
            make_float2(CustomArNormRound(cosf(angle)), CustomArNormRound(sinf(angle)));
    }
}
constexpr int QkvStageBytes = 16 * (4096 + 128 * sizeof(float));
__device__ void StageQkv(const Args &a, uint8_t *stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if (blockIdx.x >= QkvTiles)
        return;
    for (int i = threadIdx.x; i < 16 * 256; i += Threads) {
        const int ch = i / 256, col = i % 256 * 16;
        const uint8_t *src =
            static_cast<const uint8_t *>(a.qkvWeight) + fastllm_swa_mma::QkvColumn(blockIdx.x, ch) * 4096 + col;
        unsigned dst = __cvta_generic_to_shared(stage + ch * 4096 + (col ^ (ch % 8 * 32)));
        asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst), "l"(src) : "memory");
    }
    const int ch = threadIdx.x / 32, col = threadIdx.x % 32 * 4;
    const float *src = a.qkvScales + fastllm_swa_mma::QkvColumn(blockIdx.x, ch) * 128 + col;
    unsigned dst = __cvta_generic_to_shared(stage + 65536 + (ch * 128 + col) * sizeof(float));
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
static_assert(RouterPairBytes + 2 * OutStageBytes <= QkvStageBytes);

// Once QKV is complete its shared arena holds the Router pair followed by
// two 16-channel O tiles. Fetch O during attention. Smaller resident grids
// read any remaining tile directly.
__device__ void StageWideOutput(const Args &a, uint8_t *stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        const int tile = blockIdx.x + slot * gridDim.x;
        if (tile >= 4096 / 16) continue;
        uint8_t *target = stage + slot * OutStageBytes;
        for (int i = threadIdx.x; i < 16 * 64; i += Threads) {
            const int ch = i / 64, col = i % 64 * 16;
            const auto *src = static_cast<const uint8_t *>(a.outWeight) + (tile * 16 + ch) * 1024 + col;
            const unsigned dst = __cvta_generic_to_shared(target + ch * 1024 + (col ^ (ch % 8 * 32)));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
        }
        if (threadIdx.x < 128) {
            const float *src = a.outScales + tile * 16 * 32 + threadIdx.x * 4;
            const unsigned dst = __cvta_generic_to_shared(target + 16 * 1024 + threadIdx.x * 16);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

static_assert(128 * QDim * sizeof(BF) <= QkvStageBytes);
// While the projection CTAs produce this round's suffix, separate resident
// CTAs decode the immutable prefix. The grid join orders the candidate reads.
__device__ void StageRingHistory(const Args &a, int task, BF *keys, BF *values) {
    const int row = task / RingTilesPerRow, first = task % RingTilesPerRow * RingValueTile, t = threadIdx.x;
    auto r = FastllmNaiveResolveDSARow(a.Row(row));
    int live = max(0, min(r.capacity, *r.liveKeys));
    int committed = max(0, min(128, live) - r.verifyOffset - 1);
    FastllmNaiveKV k(r.key, nullptr, 0, r.fp8), v(r.value, nullptr, 0, r.fp8);
    for (int i = t; i < committed * 24; i += Threads) {
        const int slot = i / 24, d = i % 24 * 8;
        reinterpret_cast<uint4 *>(keys)[i] = (k.Row(Token<true>(r, slot, live), r.keyStride) + d).Load8();
    }
    const int slot = t / 4, vec = t % 4;
    if (slot < committed)
        reinterpret_cast<uint4 *>(values)[slot * 4 + (vec ^ (slot % 4))] =
            (v.Row(Token<true>(r, slot, live), VDim) + first + vec * 8).Load8();
}
// Every tagged collective advances all 32 epochs on its rank. Borrow the
// otherwise unused ring score arena for this call's generation.
__device__ void BeginRingReduction(const Args &a) {
    if (blockIdx.x == 0 && threadIdx.x < kTaggedPeerBlocks) {
        const uint64_t step = a.transport.epochs[threadIdx.x];
        if (threadIdx.x == 0)
            *reinterpret_cast<uint64_t *>(a.scores) = step;
        a.transport.epochs[threadIdx.x] = step + 1 == kTaggedPeerEpochPeriod ? 0 : step + 1;
    }
}
// The QKV tiles are idle during attention. Bring their next O tile into L2.
__device__ void PrefetchRingOutput(const Args &a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto *weight = static_cast<const uint8_t *>(a.outWeight);
    for (int tile = blockIdx.x; tile < 128; tile += gridDim.x) {
        const int t = threadIdx.x;
        asm volatile("prefetch.global.L2 [%0];" :: "l"(weight + tile * 32768 + t * 64));
        if (t < 64)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(a.outScales + tile * 1024 + t * 16));
    }
#endif
}
// Preserve each projection's FP32 reduction order. DSA splits K over 16
// warps. SWA uses two groups of eight warps on adjacent output tiles.
template <bool Ring, bool AddResidual = false>
__device__ void ProjectAndReduceOutput(const Args &a, float (&part)[16][128]) {
    constexpr int warps = Ring ? 8 : 16, cols = Ring ? 32 : 16;
    constexpr int pairs = cols / 2, rowWords = pairs * 8, stepRows = Threads / rowWords;
    const auto step = *reinterpret_cast<const uint64_t *>(a.scores);
    const uint32_t tag = uint32_t(step) + 1;
    const size_t frame = (step % kTaggedPeerBuffers) * a.transport.frameBytes;
    if (TaggedPeerClearUnused(tag)) {
        // Share the frame rollover protocol with the ordinary TP collectives.
        for (int word = a.rows * 2048 + blockIdx.x * Threads + threadIdx.x; word < kTaggedPeerMaxElements * 3 / 2;
             word += gridDim.x * Threads) {
            size_t dst = frame + a.rank * a.transport.slotBytes + word * 8;
            a.transport.Publish(dst, 0);
        }
    }
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, quad = t % 4;
    for (int tile = blockIdx.x; tile < 4096 / cols; tile += gridDim.x) {
        const int row = g;
        const int base = tile * cols + (warp / warps) * 16;
        float acc[4]{};
#pragma unroll
        for (int i = 0; i < 32 / warps; ++i) {
            const int group = i * warps + warp % warps, k = group * 32 + quad * 8;
            auto w =
                fastllm_mxfp8_projection::LoadWeights(a.outWeight, a.outScales, 1024, base + g, base + g + 8, k);
            uint2 x{};
            float s = 0;
            if (row < a.rows) {
                const int offset = Ring ? group * a.rows + row : row * 32 + group;
                x = *reinterpret_cast<const uint2 *>(a.packed + offset * 32 + quad * 8);
                s = a.scales[offset];
            }
            fastllm_mxfp8_projection::Accumulate(acc, w, x, __shfl_sync(0xffffffff, s, quad * 8),
                                                 __shfl_sync(0xffffffff, s, quad * 8 + 4));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            part[warp][(quad * 2 + j % 2) * 16 + (j / 2) * 8 + g] = acc[j];
        __syncthreads();
        if (t < 8 * cols) {
            const int r = t / cols, col = t % cols, src = r * 16 + col % 16;
            float sum = 0.f;
#pragma unroll
            for (int k = 0; k < warps; ++k)
                sum += part[col / 16 * warps + k][src];
            // A tagged word publishes two BF16 channels atomically to all ranks.
            float other = __shfl_down_sync(0xffffffff, sum, 1);
            if (!(col & 1) && r < a.rows) {
                const __nv_bfloat162 rounded = __floats2bfloat162_rn(sum, other);
                const unsigned bits = *reinterpret_cast<const unsigned *>(&rounded);
                const int word = (r * 4096 + tile * cols + col) / 2;
                reinterpret_cast<unsigned *>(a.projection)[word] = bits;
                const uint64_t payload = (uint64_t(bits) << 32) | tag;
                const size_t offset = frame + a.rank * a.transport.slotBytes + word * 8;
                a.transport.Publish(offset, payload);
            }
        }
        __syncthreads();
        // Poll local memory in parallel, then sum in the original rank order.
        auto *received = reinterpret_cast<unsigned *>(part);
        // The second word belongs to the same peer four rows later. Issue both
        // reads before testing the tags to keep the two memory requests in flight.
        if (t < a.rows * rowWords) {
            const int r = t / rowWords, peer = t / pairs % 8;
            const int word = r * 2048 + tile * pairs + t % pairs;
            const auto *slot = reinterpret_cast<const uint64_t *>(
                a.transport.peers.data[a.rank] + frame + peer * a.transport.slotBytes) + word;
            const bool second = Ring && r + stepRows < a.rows;
            uint64_t lo, hi;
            do {
                lo = TaggedPeerLoad(slot);
                hi = second ? TaggedPeerLoad(slot + stepRows * 2048) : tag;
            } while (uint32_t(lo) != tag || uint32_t(hi) != tag);
            received[t] = uint32_t(lo >> 32);
            if (second)
                received[t + Threads] = uint32_t(hi >> 32);
        }
        __syncthreads();
        if (t < a.rows * pairs) {
            const int row = t / pairs, pair = t % pairs;
            float2 sum{};
#pragma unroll
            for (int peer = 0; peer < 8; ++peer) {
                unsigned bits = received[(row * 8 + peer) * pairs + pair];
                float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&bits));
                if (!peer)
                    sum = v;
                else {
                    sum.x += v.x;
                    sum.y += v.y;
                }
            }
            const int offset = row * 2048 + tile * pairs + pair;
            const auto reduced = __floats2bfloat162_rn(sum.x, sum.y);
            if constexpr (AddResidual) {
                auto *hidden = reinterpret_cast<__nv_bfloat162 *>(a.hidden);
                hidden[offset] = __hadd2(hidden[offset], reduced);
            } else {
                reinterpret_cast<__nv_bfloat162 *>(a.normed)[offset] = reduced;
            }
        }
        if (tile + gridDim.x < 4096 / cols)
            __syncthreads();
    }
}

// The wide projection writes two rounded channels and their epoch together.
__device__ void FinishPublishedProjection(const Args &a) {
    const auto step = *reinterpret_cast<const uint64_t *>(a.scores);
    const auto tag = uint32_t(step) + 1;
    const size_t frame = (step % kTaggedPeerBuffers) * a.transport.frameBytes;
    const int first = blockIdx.x * Threads + threadIdx.x, stride = gridDim.x * Threads;
    if (TaggedPeerClearUnused(tag)) {
        for (int word = a.rows * 2048 + first; word < kTaggedPeerMaxElements * 3 / 2; word += stride)
            a.transport.Publish(frame + a.rank * a.transport.slotBytes + size_t(word) * 8, 0);
    }
    for (int word = first; word < a.rows * 2048; word += stride) {
        const char *local = a.transport.peers.data[a.rank] + frame + size_t(word) * 8;
        reinterpret_cast<unsigned *>(a.normed)[word] = TaggedPeerReduceBf16Pair(local, a.transport.slotBytes, tag);
    }
}

// One contiguous 512-channel shard per owner. Each source writes only to
// that owner; every GPU reads a single multicast reduction for each pair.
__device__ void FinishShardedProjection(const Args &a, float *scratch) {
    constexpr int pairs = 256;
    const uint64_t step = *reinterpret_cast<const uint64_t *>(a.scores);
    const uint32_t tag = uint32_t(step) + 1;
    const size_t frame = (step % kTaggedPeerBuffers) * a.transport.frameBytes;
    const size_t own = frame + a.rank * a.transport.slotBytes;
    const int live = a.rows * pairs, t = threadIdx.x;
    if (TaggedPeerClearUnused(tag)) {
        // Other tagged collectives reuse these frames with different message
        // sizes. Sweep both unused ranges on epoch rollover.
        for (int i = blockIdx.x * Threads + t; i < kTaggedPeerMaxElements * 3 / 2; i += gridDim.x * Threads)
            if ((i >= live && i < kTaggedPeerMaxElements) || i >= kTaggedPeerMaxElements + live)
                a.transport.Publish(own + size_t(i) * 8, 0);
    }
    // Spread shard reduction over resident CTAs before each row gathers its packets.
    if (t < 64) {
        for (int i = blockIdx.x * 64 + t; i < live; i += gridDim.x * 64) {
            const char *local = a.transport.peers.data[a.rank] + frame + size_t(i) * 8;
            unsigned bits = TaggedPeerReduceBf16Pair(local, a.transport.slotBytes, tag);
            a.transport.Publish(own + size_t(kTaggedPeerMaxElements + i) * 8, (uint64_t(bits) << 32) | tag);
        }
    }
    for (int row = blockIdx.x; row < a.rows; row += gridDim.x) {
        const int owner = t / 64, first = row * pairs + t % 64 * 4;
        const char *local = a.transport.peers.data[a.rank] + frame + owner * a.transport.slotBytes;
        const auto *slot = reinterpret_cast<const uint64_t *>(local) + kTaggedPeerMaxElements + first;
        uint64_t words[4];
        bool ready;
        do {
            ready = true;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                words[i] = TaggedPeerLoad(slot + i);
                ready &= uint32_t(words[i]) == tag;
            }
        } while (!ready);
        CustomArNormPacketType packet;
#pragma unroll
        for (int i = 0; i < 4; ++i) reinterpret_cast<unsigned *>(&packet)[i] = uint32_t(words[i] >> 32);
        CustomArNormPacket(packet, a.hidden + row * 4096, a.gamma, a.normed + row * 4096, a.eps, scratch, t);
        if (row + gridDim.x < a.rows) __syncthreads();
    }
}

// Each CTA computes 32 output dimensions for all eight GQA heads. The
// cache values and the 16-key score tiles are shared by those heads.
template <bool Cached, bool GroupMajor = false>
__device__ void RingAttention(const Args &a, BF *values, float *score, const BF *keys = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int Tile = RingValueTile;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int g = lane / 4, quad = lane % 4;
    __shared__ __align__(16) BF prob[Heads][128];
    for (int task = Cached ? int(blockIdx.x) - QkvTiles : int(blockIdx.x); task < a.rows * (VDim / Tile);
         task += gridDim.x) {
        const int row = task / (VDim / Tile), firstCol = task % (VDim / Tile) * Tile;
        const auto r = FastllmNaiveResolveDSARow(a.Row(row));
        const int live = max(0, min(r.capacity, *r.liveKeys)), count = min(128, live);
        FastllmNaiveKV k(r.key, nullptr, 0, r.fp8), v(r.value, nullptr, 0, r.fp8);
        const int slot = t / (Tile / 8), vec = t % (Tile / 8);
        const int committed = Cached ? max(0, count - r.verifyOffset - 1) : 0;
        if (slot >= committed)
            reinterpret_cast<uint4 *>(values)[slot * (Tile / 8) + (vec ^ (slot % (Tile / 8)))] =
                slot < count ? (v.Row(Token<true>(r, slot, live), VDim) + firstCol + vec * 8).Load8()
                             : make_uint4(0, 0, 0, 0);
        if (warp < 8) {
            const int low = warp * 16 + g, high = low + 8;
            auto l = k.Row(low < count ? Token<true>(r, low, live) : 0, r.keyStride);
            auto h = k.Row(high < count ? Token<true>(r, high, live) : 0, r.keyStride);
            float acc[4]{};
#pragma unroll
            for (int d = quad * 8; d < QDim; d += 32) {
                const uint4 x = *reinterpret_cast<const uint4 *>(a.query + (row * Heads + g) * QDim + d);
                uint4 lo, hi;
                if (Cached && low < committed)
                    lo = *reinterpret_cast<const uint4 *>(keys + low * QDim + d);
                else
                    lo = low < count ? (l + d).Load8() : make_uint4(0, 0, 0, 0);
                if (Cached && high < committed)
                    hi = *reinterpret_cast<const uint4 *>(keys + high * QDim + d);
                else
                    hi = high < count ? (h + d).Load8() : make_uint4(0, 0, 0, 0);
                fastllm_swa_mma::Step(acc, lo, hi, x);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = j < 2 ? low : high;
                score[(quad * 2 + j % 2) * 128 + key] =
                    key < count ? CustomArNormRound(CustomArNormRound(acc[j]) * rsqrtf(192.f)) : -INFINITY;
            }
        }
        __syncthreads();
        if (warp < Heads) {
            const float sink = a.sink ? a.sink[warp] : -INFINITY;
            float largest = sink, p[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = score[warp * 128 + lane + j * 32];
                largest = fmaxf(largest, p[j]);
            }
            for (int d = 16; d; d /= 2)
                largest = fmaxf(largest, __shfl_down_sync(0xffffffff, largest, d));
            largest = __shfl_sync(0xffffffff, largest, 0);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                p[j] = lane + j * 32 < count ? expf(p[j] - largest) : 0.f;
            float first = p[0];
            if (lane == 0 && a.sink)
                first += expf(sink - largest);
            const float denom = __shfl_sync(0xffffffff, CustomArNormWarpSum((first + p[2]) + (p[1] + p[3])), 0);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                prob[warp][lane + j * 32] = __float2bfloat16_rn(denom > 0 ? p[j] / denom : 0.f);
        }
        __syncthreads();
        if (warp < Tile / 16) {
            float acc[4]{};
#pragma unroll
            for (int first = 0; first < 128; first += 16) {
                const int krow = first + lane % 8 + lane / 16 * 8;
                const int vcol = warp * 16 + lane / 8 % 2 * 8;
                const unsigned address = __cvta_generic_to_shared(values + krow * Tile + (vcol ^ (krow % 4 * 8)));
                unsigned m0, m1, m2, m3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                             : "=r"(m0), "=r"(m1), "=r"(m2), "=r"(m3)
                             : "r"(address));
                const unsigned p0 = *reinterpret_cast<const unsigned *>(prob[g] + first + quad * 2);
                const unsigned p1 = *reinterpret_cast<const unsigned *>(prob[g] + first + quad * 2 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                             : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                             : "r"(m0), "r"(m1), "r"(m2), "r"(m3), "r"(p0), "r"(p1));
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                score[(quad * 2 + j % 2) * Tile + warp * 16 + g + j / 2 * 8] = acc[j];
        }
        __syncthreads();
        if (t < Heads * Tile)
            StoreOutput<true, GroupMajor>(a, (row * Heads + warp) * VDim + firstCol + lane, score[t]);
        __syncthreads();
    }
#endif
}
