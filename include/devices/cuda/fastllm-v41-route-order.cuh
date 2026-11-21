#pragma once
#include <cuda_runtime.h>

// Sort the six slots by (expert rank, original slot) in each eight-lane
// subgroup. Stable order preserves both FP32 reductions while keeping the
// accumulator in registers when IDs vary across real request rows.
__device__ __forceinline__ int FastllmV41RouteOrder(const int *ids) {
    const int lane = threadIdx.x & 7;
    const int expert = lane < 6 ? ids[lane] : -1;
    int key = ((expert >= 0 ? expert % 8 : 8) << 3) | lane;
#pragma unroll
    for (int span = 2; span <= 8; span *= 2) {
#pragma unroll
        for (int d = span / 2; d; d /= 2) {
            const int other = __shfl_xor_sync(0xffffffff, key, d, 8);
            key = bool(lane & span) == bool(lane & d) ? min(key, other) : max(key, other);
        }
    }
    const int next = __shfl_down_sync(0xffffffff, key, 1, 8);
    return (key & 7) | (key < 64 ? 8 : 0) | (lane == 5 || key / 8 != next / 8 ? 16 : 0);
}

// The low three bits name the slot. Bits 3 and 4 encode validity and the
// last slot of an owner. Six five-bit entries fit in one word per request.
__device__ __forceinline__ void FastllmV41PublishOrder(const int *ids, int *destination) {
    const unsigned lane = threadIdx.x & 7;
    const unsigned entry = FastllmV41RouteOrder(ids);
    unsigned bits = lane < 6 ? entry << (lane * 5) : 0;
#pragma unroll
    for (int mask = 4; mask; mask >>= 1)
        bits |= __shfl_xor_sync(0xffffffff, bits, mask, 8);
    if (threadIdx.x == 0) *destination = static_cast<int>(bits);
}
