#pragma once
#include <cuda/atomic>
#include <cuda/ptx>

namespace naivert {
// state is [arrival count, generation], initialized once to zero. All threads
// of every one-dimensional CTA must participate; the launcher guarantees
// whole-grid residency. Generation is monotone modulo unsigned arithmetic.
__device__ inline void resident_grid_barrier(unsigned* state) {
  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrivals(state[0]);
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> generation(state[1]);
    // Snapshot before arrival; the release RMW orders this read as well as payload writes.
    const unsigned gen = generation.load(cuda::memory_order_relaxed);
    if (arrivals.fetch_add(1u, cuda::memory_order_acq_rel) == unsigned(gridDim.x) - 1u) {
      // The final acquire RMW collects all preceding arrivals before publishing completion.
      arrivals.store(0u, cuda::memory_order_relaxed);
      generation.fetch_add(1u, cuda::memory_order_release);
    } else {
      while (generation.load(cuda::memory_order_relaxed) == gen) {}
    }
    cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_gpu);
  }
  __syncthreads();
}

}  // namespace naivert
