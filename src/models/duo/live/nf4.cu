#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

namespace {
__constant__ float codebook[16] = {
    -1.f,         -.696192801f, -.525073051f, -.394917488f, -.284441382f, -.18477343f, -.0910500363f, 0.f,
    .0795802996f, .160930201f,  .246112302f,  .337915242f,  .440709829f,  .562617004f, .722956836f,   1.f};
__device__ int nearest(float x) {
    int i = 0;
    for (int step = 8; step; step >>= 1) {
        int j = i + step;
        if (x > (codebook[j - 1] + codebook[j]) * .5f)
            i = j;
    }
    return i;
}
__global__ void encode(const __nv_bfloat16 *src, unsigned char *dst, float *scales) {
    // Each warp encodes 64 consecutive weights into 32 packed bytes.
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x % 32;
    float a = __bfloat162float(src[2 * i]), b = __bfloat162float(src[2 * i + 1]);
    float m = fmaxf(fabsf(a), fabsf(b));
    for (int d = 16; d; d /= 2)
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, d));
    if (!lane)
        scales[i / 32] = m;
    float inv = m ? 1.f / m : 0.f;
    dst[i] = (nearest(a * inv) << 4) | nearest(b * inv);
}
// Four packed bytes expand to one coalesced 16-byte BF16 store.
// A warp reads arbitrary codebook entries without serializing constant memory.
__global__ void unpack(const unsigned *q, const float *scales, uint4 *dst, int groups) {
    __shared__ float lut[16];
    if (threadIdx.x < 16)
        lut[threadIdx.x] = codebook[threadIdx.x];
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups)
        return;
    const unsigned bytes = q[i];
    const float scale = scales[i / 8];
    unsigned pair[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned byte = (bytes >> (8 * j)) & 255;
        const auto hi = __float2bfloat16(lut[byte >> 4] * scale);
        const auto lo = __float2bfloat16(lut[byte & 15] * scale);
        pair[j] = __bfloat16_as_ushort(hi) | (unsigned(__bfloat16_as_ushort(lo)) << 16);
    }
    dst[i] = make_uint4(pair[0], pair[1], pair[2], pair[3]);
}
__global__ void gemv(const unsigned char *q, const float *scales, const __nv_bfloat16 *x, __nv_bfloat16 *y,
                     int cols) {
    int row = blockIdx.x * 4 + threadIdx.y, lane = threadIdx.x;
    const float entry = lane < 16 ? codebook[lane] : 0.f;
    float sum = 0;
    for (int i = lane; i < cols / 2; i += 32) {
        int off = row * (cols / 2) + i;
        auto a = q[off];
        float s = scales[off / 32];
        float high = __shfl_sync(0xffffffff, entry, a >> 4);
        float low = __shfl_sync(0xffffffff, entry, a & 15);
        sum = fmaf(__bfloat162float(__float2bfloat16(high * s)), __bfloat162float(x[2 * i]), sum);
        sum = fmaf(__bfloat162float(__float2bfloat16(low * s)), __bfloat162float(x[2 * i + 1]), sum);
    }
    for (int d = 16; d; d /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    if (!lane)
        y[row] = __float2bfloat16(sum);
}
} // namespace
extern "C" void duo_nf4_encode(void *in, void *out, void *scales, int n, cudaStream_t stream) {
    encode<<<n / 256, 128, 0, stream>>>((__nv_bfloat16 *)in, (unsigned char *)out, (float *)scales);
}
extern "C" void duo_nf4_unpack(void *q, void *scales, void *out, int n, cudaStream_t stream) {
    const int groups = n / 8; // NF4 blocks contain 64 elements.
    unpack<<<(groups + 127) / 128, 128, 0, stream>>>((const unsigned *)q, (const float *)scales, (uint4 *)out,
                                                     groups);
}
extern "C" void duo_nf4_gemv(void *q, void *scales, void *x, void *out, int rows, int cols,
                             cudaStream_t stream) {
    gemv<<<rows / 4, dim3(32, 4), 0, stream>>>((unsigned char *)q, (float *)scales, (__nv_bfloat16 *)x,
                                               (__nv_bfloat16 *)out, cols);
}
