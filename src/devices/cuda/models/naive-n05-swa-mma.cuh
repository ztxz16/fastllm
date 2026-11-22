#pragma once
#include <cuda_runtime.h>

// Register and shared-memory layouts used by the BF16 SWA row kernels.
// Step multiplies a 16x32 weight tile by an eight-row input tile in two
// m16n8k16 instructions. Callers retain their own FP32 reduction order and
// BF16 materialization boundaries; no packed weights or global scratch.
namespace fastllm_swa_mma {
// Pair rotary columns in one QKV tile; all weights remain row-major.
__device__ __forceinline__ int QkvColumn(int tile, int out) {
    if (tile < 36)
        return tile / 4 * 192 + tile % 4 * 8 + out % 8 + out / 8 * 32;
    if (tile < 108)
        return (tile - 36) / 8 * 192 + 64 + (tile - 36) % 8 * 16 + out;
    return 1728 + (tile - 108) * 16 + out;
}
__device__ __forceinline__ int InputOffset(int row, int col) {
    return col / 64 * 1024 + row * 128 + ((col % 64 / 8) ^ row) * 16;
}
__device__ __forceinline__ void Step(float (&c)[4], uint4 a, uint4 b, uint4 x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a.x), "r"(b.x), "r"(a.y), "r"(b.y), "r"(x.x), "r"(x.y));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a.z), "r"(b.z), "r"(a.w), "r"(b.w), "r"(x.z), "r"(x.w));
#endif
}

} // namespace fastllm_swa_mma
