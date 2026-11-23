#pragma once
#include "naivert/body/attn_common/rows_gemv_mma.cuh"

namespace naivert::attn_rows {
// Padded shared output with an explicit gamma-readiness callback.
struct StagedNormGammaReady {
  __device__ __forceinline__ bool operator()() const { return true; }
};

template <int kRows, int kDim, int kOutputStride = kDim, bool kPackedGamma = false,
          class GammaReady = StagedNormGammaReady>
__device__ static void normalize_staged_rows(const __nv_bfloat16* residual, const float* gamma,
                                      __nv_bfloat16* x_norm, float eps, GammaReady before_gamma = {}) {
  static_assert(kDim % 256 == 0, "a lane owns kDim / 256 16-byte pieces of a row");
  static_assert(kOutputStride >= kDim && kOutputStride % 8 == 0,
                "output rows must fit and preserve16-byte vector alignment");
  constexpr int kPieces = kDim / 256;
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int warps = int(blockDim.x) >> 5;
  for (int row = warp; row < kRows; row += warps) {
    const uint4* source =
        reinterpret_cast<const uint4*>(residual + std::size_t(row) * kDim);
    uint4 staged[kPieces];
    float sum = 0.0f;
#pragma unroll
    for (int piece = 0; piece < kPieces; ++piece) {
      staged[piece] = __ldg(source + lane + piece * 32);
      const unsigned words[4] = {staged[piece].x, staged[piece].y, staged[piece].z,
                                 staged[piece].w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float2 value =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + w));
        sum = fmaf(value.x, value.x, sum);
        sum = fmaf(value.y, value.y, sum);
      }
    }
    const float inverse_rms = rsqrtf(warp_sum_f32(sum) / float(kDim) + eps);
    const bool gamma_ok = before_gamma();
#pragma unroll
    for (int piece = 0; piece < kPieces; ++piece) {
      const int k = (lane + piece * 32) * 8;
      const unsigned words[4] = {staged[piece].x, staged[piece].y, staged[piece].z,
                                 staged[piece].w};
      uint4 out;
      unsigned* out_words = reinterpret_cast<unsigned*>(&out);
      const int gamma_lo = kPackedGamma ? PackedNormGamma::offset(k) : k;
      const int gamma_hi = kPackedGamma ? PackedNormGamma::offset(k + 4) : k + 4;
      const float4 g_lo = gamma_ok ? *reinterpret_cast<const float4*>(gamma + gamma_lo) : make_float4(0, 0, 0, 0);
      const float4 g_hi = gamma_ok ? *reinterpret_cast<const float4*>(gamma + gamma_hi) : make_float4(0, 0, 0, 0);
      const float g[8] = {g_lo.x, g_lo.y, g_lo.z, g_lo.w, g_hi.x, g_hi.y, g_hi.z, g_hi.w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float2 value =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + w));
        __nv_bfloat162 normalised = __floats2bfloat162_rn(
            value.x * inverse_rms * g[2 * w], value.y * inverse_rms * g[2 * w + 1]);
        out_words[w] = *reinterpret_cast<unsigned*>(&normalised);
      }
      *reinterpret_cast<uint4*>(x_norm + std::size_t(row) * kOutputStride + k) = out;
    }
  }
  __syncthreads();
}

}  // namespace naivert::attn_rows
