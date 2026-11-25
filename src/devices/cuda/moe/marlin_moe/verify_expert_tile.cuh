#pragma once

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace fastllm_marlin_verify {

// Eight tokens with eight routes each can select up to 64 distinct experts.
// The router clears one state word per grouped expert before verification.
constexpr int kRoutes = 64;
constexpr int kExtraLocks = 2 + kRoutes;

__host__ __device__ inline int *ExpertStates(int *locks, int sms) {
    // Gate and Down have independent split-K locks, including a guard each.
    return locks + 4 * sms + 2;
}

struct ExpertTile {
    __nv_bfloat16 *activation = nullptr;
    int *states = nullptr;

    __device__ __forceinline__ void Publish(const int4 *gateUp, int slot, int tile,
                                            const int *routes, int rows) const {
        // Gate and Up each produce two 128-column tiles. The second arrival
        // in each pair computes SwiGLU from the rounded BF16 projections.
        const int half = tile % 2;
        const int pairBit = 1 << half;
        ::cuda::atomic_ref<int, ::cuda::thread_scope_device> state(states[slot]);
        int previous = 0;
        if (threadIdx.x == 0)
            previous = state.fetch_xor(pairBit, ::cuda::memory_order_acq_rel);
        if (!__syncthreads_or(previous & pairBit)) return;

        const int col = half * 128 + threadIdx.x;
        for (int row = 0; row < rows; ++row) {
            const int route = routes[row];
            const auto *pair = reinterpret_cast<const __nv_bfloat16 *>(gateUp) + route * 512 + col;
            const float gate = __bfloat162float(pair[0]);
            const float up = __bfloat162float(pair[256]);
            activation[route * 256 + col] = __float2bfloat16((gate / (1.0f + expf(-gate))) * up);
        }
        __syncthreads();
        if (threadIdx.x == 0)
            state.fetch_or(4 << half, ::cuda::memory_order_acq_rel);
    }

    __device__ __forceinline__ void Wait(int slot) const {
        if (threadIdx.x == 0) {
            ::cuda::atomic_ref<int, ::cuda::thread_scope_device> state(states[slot]);
            // Both arrival bits have toggled back to zero, both activation
            // halves have been released. No other expert needs to be ready.
            while (state.load(::cuda::memory_order_acquire) != 12) {}
        }
        __syncthreads();
    }
};

}  // namespace fastllm_marlin_verify
