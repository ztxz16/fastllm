#pragma once

#include <cuda_runtime.h>

namespace naivert {

// SM90+: all CTA threads call this with the same 16-byte-aligned addresses.
// Completion and the CTA barrier make the entire input visible on return.
template <int Bytes>
__device__ __forceinline__ void bulk_input_copy(void* destination, const void* source) {
  static_assert(Bytes > 0 && Bytes % 16 == 0);
  __shared__ __align__(8) unsigned long long input_ready;
  if (threadIdx.x == 0) {
    const unsigned barrier = unsigned(__cvta_generic_to_shared(&input_ready));
    const unsigned target = unsigned(__cvta_generic_to_shared(destination));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(Bytes) : "memory");
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];"
        :: "r"(target), "l"(source), "r"(Bytes), "r"(barrier) : "memory");
    unsigned ready;
    do {
      asm volatile(
          "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; "
          "selp.u32 %0, 1, 0, p;}"
          : "=r"(ready) : "r"(barrier) : "memory");
    } while (!ready);
  }
  __syncthreads();
}

}  // namespace naivert
