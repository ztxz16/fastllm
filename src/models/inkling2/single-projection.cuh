// A single token may have three distinct controller outputs for Q, K and V.
// Pack their activation scales together and select the input by output channel.
struct DecodeActivationBlock {
    int bytes[8];
    float scale;
};

template<class T>
__global__ void packDecodeActivations(const T *q, const T *k, const T *v, int width, DecodeActivationBlock *packed) {
    cuda_launch::DependencyStart();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const int row = blockIdx.y, lane = threadIdx.x % 32;
    const float x = float((row == 0 ? q : row == 1 ? k : v)[i]);
    float maximum = fabsf(x);
#pragma unroll
    for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d));
    const float scale = maximum / 127.f;
    auto &dst = packed[(row * width + i) / 32];
    reinterpret_cast<int8_t *>(dst.bytes)[lane] = maximum > 0.f ? int(roundf(x / scale)) : 0;
    if (!lane) dst.scale = float(half(scale));
}

struct StoreProjectionRow {
    template<class T> __device__ void operator()(T *out, int row, float value) const { out[row] = T(value); }
};

template<int Width, class Out, class Store = StoreProjectionRow>
__global__ void projectDecodeQ8(const DecodeActivationBlock *packed, const uint16_t *weights, Out *output,
                                int rows, int kBegin, int vBegin, Store store = {}) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    constexpr int blocks = Width / 32;
    const int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    cuda_launch::DependencyStart();
    if (row >= rows) return;
    packed += ((row >= kBegin) + (row >= vBegin)) * blocks;
    // Each four-lane subgroup consumes one block of 32 signed coefficients.
    // The 34-byte checkpoint blocks are only guaranteed halfword alignment.
    const int pair = lane % 4 * 2;
    float dot = 0.f;
#pragma unroll
    for (int b = lane / 4; b < blocks; b += 8) {
        const uint16_t *w = weights + (size_t(row) * blocks + b) * 17;
        const float scale = __half2float(__ushort_as_half(w[0])) * packed[b].scale;
        w += 1 + pair * 2;
        const int lo = unsigned(w[0]) | (unsigned(w[1]) << 16);
        const int hi = unsigned(w[2]) | (unsigned(w[3]) << 16);
        const int sum = __dp4a(hi, packed[b].bytes[pair + 1], __dp4a(lo, packed[b].bytes[pair], 0));
        dot += scale * sum;
    }
    dot = cuda::WarpSum(dot);
    if (!lane) store(output, row, dot);
#endif
}

// Four output rows per block. Each lane streams consecutive BF16 pairs from
// its own row and shares activation cache lines with the neighbouring warps.
template<int Width>
__global__ void projectDecodeBf16(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v,
                                  const __nv_bfloat16 *weight, __nv_bfloat16 *out, int rows, int kBegin, int vBegin) {
    constexpr int tiles = Width / 256;
    const int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    const bool valid = row < rows;
    uint4 coefficients[tiles];
    if (valid) {
        auto *w = reinterpret_cast<const uint4 *>(weight + size_t(row) * Width);
#pragma unroll
        for (int t = 0; t < tiles; ++t) coefficients[t] = __ldg(w + t * 32 + lane);
    }
    cuda_launch::DependencyStart();
    if (!valid) return;
    const auto *input = reinterpret_cast<const uint4 *>(row < kBegin ? q : row < vBegin ? k : v);
    float2 acc[2]{};
#pragma unroll
    for (int t = 0; t < tiles; ++t) {
        uint4 x = input[t * 32 + lane], w = coefficients[t];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const auto a = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(&x)[p]);
            const auto b = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(&w)[p]);
            acc[p % 2].x = fmaf(a.x, b.x, acc[p % 2].x);
            acc[p % 2].y = fmaf(a.y, b.y, acc[p % 2].y);
        }
    }
    const float sum = cuda::WarpSum((acc[0].x + acc[1].x) + (acc[0].y + acc[1].y));
    if (!lane) out[row] = __float2bfloat16_rn(sum);
}

template<class F> bool dispatchDecodeWidth(int width, F launch) {
    if (width == 1536) return launch(std::integral_constant<int, 1536>{});
    if (width == 2304) return launch(std::integral_constant<int, 2304>{});
    if (width == 3072) return launch(std::integral_constant<int, 3072>{});
    return false;
}

bool launchDecodeBf16(const __nv_bfloat16 *q, const __nv_bfloat16 *k, const __nv_bfloat16 *v, const void *weights,
                      __nv_bfloat16 *out, int width, int rows, int kBegin, int vBegin) {
    auto launch = [&](auto tag) {
        if (!cuda_launch::Launch<projectDecodeBf16<decltype(tag)::value>>((rows + 3) / 4, 128, 0, cudaStreamPerThread,
                q, k, v, static_cast<const __nv_bfloat16 *>(weights), out, rows, kBegin, vBegin))
            throw std::runtime_error("Inkling BF16 decode projection failed");
        return true;
    };
    return dispatchDecodeWidth(width, launch);
}
