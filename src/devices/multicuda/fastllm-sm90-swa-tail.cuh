#pragma once

// Publish each BF16 pair directly to the rank that reduces it. An epoch travels
// with the pair, so no rank has to wait for a separate input-ready handshake.
template <bool Small, bool Early = false>
__global__ void HopperTaggedNormRouterInput(const unsigned *input, CustomArNormBF16 *hidden, const float *gamma,
        CustomArNormBF16 *normed, float eps, int rank, int rows, TaggedPeerView transport, const float *routerWeight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    static_assert(16 * 2048 <= kTaggedPeerMaxElements && 16 * 256 <= kTaggedPeerMaxElements / 2);
    const int t = threadIdx.x, row = blockIdx.x;
    if constexpr (Early) {
        if (row < rows && t < 128)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(gamma + t * 32));
        cudaGridDependencySynchronize();
        if constexpr (Small) cudaTriggerProgrammaticLaunchCompletion();
    }
    __shared__ uint64_t step;
    if (!t) {
        step = transport.epochs[row];
        transport.epochs[row] = step + 1 == kTaggedPeerEpochPeriod ? 0 : step + 1;
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    const size_t frame = step % kTaggedPeerBuffers * transport.frameBytes;
    const size_t own = frame + rank * transport.slotBytes;
    if (TaggedPeerClearUnused(tag)) {
        // Changing the row count or alternating with another tagged collective
        // must not leave a matching old tag anywhere after epoch rollover.
        for (int i = row * blockDim.x + t; i < kTaggedPeerMaxElements * 3 / 2; i += gridDim.x * blockDim.x) {
            if (i >= kTaggedPeerMaxElements + rows * 256 || (i >= rows * 2048 && i < kTaggedPeerMaxElements))
                transport.Publish(own + i * 8, 0);
            else if (i < rows * 2048) {
                const int owner = i % 2048 / 256;
                #pragma unroll
                for (int gpu = 0; gpu < 8; ++gpu) if (gpu != owner)
                    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::
                        "l"(transport.peers.data[gpu] + own + i * 8), "l"(uint64_t(0)) : "memory");
            }
        }
    }
    if (row >= rows) {
        if (routerWeight) {
            const int first = (row - rows) * blockDim.x + t, stride = (gridDim.x - rows) * blockDim.x;
            for (int i = first; i < 256 * 4096 / 16; i += stride)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(routerWeight + size_t(i) * 16));
        }
        return;
    }
    __shared__ __align__(16) CustomArNormBF16 values[4096];
    __shared__ float sums[16];
    float affine[16];
    if constexpr (!Small) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) affine[k] = gamma[t + k * 256];
    }
    for (int pair = t; pair < 2048; pair += blockDim.x) {
        const int i = row * 2048 + pair;
        const uint64_t data = (uint64_t(input[i]) << 32) | tag;
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::
            "l"(transport.peers.data[pair / 256] + own + i * 8), "l"(data) : "memory");
    }
    if (t < 256) {
        const int i = row * 2048 + rank * 256 + t;
        const char *local = transport.peers.data[rank] + frame + size_t(i) * 8;
        unsigned bits = TaggedPeerReduceBf16Pair(local, transport.slotBytes, tag);
        const uint64_t result = (uint64_t(bits) << 32) | tag;
        transport.Publish(own + size_t(kTaggedPeerMaxElements + row * 256 + t) * 8, result);
    }
    for (int pair = t; pair < 2048; pair += blockDim.x) {
        const char *base = transport.peers.data[rank] + frame + (pair / 256) * transport.slotBytes;
        const auto *slot = reinterpret_cast<const uint64_t *>(base) + kTaggedPeerMaxElements + row * 256 + pair % 256;
        uint64_t value;
        do { value = TaggedPeerLoad(slot); } while (uint32_t(value) != tag);
        unsigned bits = uint32_t(value >> 32);
        auto reduced = *reinterpret_cast<const __nv_bfloat162 *>(&bits);
        if constexpr (!Small) {
            auto *h = reinterpret_cast<__nv_bfloat162 *>(hidden) + row * 2048 + pair;
            reduced = __hadd2(reduced, *h);
            *h = reduced;
        }
        reinterpret_cast<__nv_bfloat162 *>(values)[pair] = reduced;
    }
    __syncthreads();
    if constexpr (Small) CustomArNormPacket(reinterpret_cast<CustomArNormPacketType *>(values)[t],
        hidden + row * 4096, gamma, normed + row * 4096, eps, sums, t);
    else CustomArNormRow256(values, affine, normed + row * 4096, eps, sums);
#endif
}
