  __device__ __forceinline__ static void prefetch_l2_bulk(
      const void* source, unsigned bytes) {
    asm volatile(
        "cp.async.bulk.prefetch.L2.global [%0], %1;\n"
        :
        : "l"(source), "r"(bytes)
        : "memory");
  }
