// Q8_0 x Q8_1 singleton GEMV. The segmented row mapping and four-warp
// reduction follow Basalt v0.5.6, native_mmvq.cu (MIT, third_party/basalt/LICENSE).
// https://github.com/jesdga95/basalt/blob/v0.5.6/src/kernels/cuda/native_mmvq.cu
struct InklingQ81 {
    half2 ds;
    int8_t qs[32];
};
struct InklingQ80 {
    half scale;
    uint16_t codes[16];
};
static_assert(sizeof(InklingQ81) == 36 && sizeof(InklingQ80) == 34);

struct Q8RowStore {

    // Eight adjacent threads each own four consecutive channels in one Q8_1 block.
    template<class Round>
    static __device__ void store(InklingQ81 *out, int vector, float4 v) {
        v = make_float4(float(Round(v.x)), float(Round(v.y)), float(Round(v.z)), float(Round(v.w)));
        float mx = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        float sum = (v.x + v.y) + (v.z + v.w);
        const unsigned mask = __activemask();
#pragma unroll
        for (int d = 4; d; d >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(mask, mx, d, 8));
            sum += __shfl_xor_sync(mask, sum, d, 8);
        }
        auto &block = out[vector / 8];
        const float scale = fminf(mx / 127.f, 65504.f);
        auto code = [=](float value) { return static_cast<signed char>(mx == 0.f ? 0.f :
            fminf(127.f, fmaxf(-127.f, roundf(value / scale)))); };
        reinterpret_cast<char4 *>(block.qs)[vector % 8] = make_char4(code(v.x), code(v.y), code(v.z), code(v.w));
        if (!(vector % 8)) block.ds = __floats2half2_rn(scale, fminf(65504.f, fmaxf(-65504.f, sum)));
    }
};

template<class T, class Out, class R> __global__ void normalizeDecodeRow(const R *input, const float *norm, Out *output, int width, float eps) {
    cuda_launch::DependencyStart();
    input += size_t(blockIdx.x) * width;
    if constexpr (std::is_same_v<Out, InklingQ81>) output += size_t(blockIdx.x) * (width / 32);
    else output += size_t(blockIdx.x) * width;
    __shared__ float scratch[16];
    float square = 0.f;
    for (int j = threadIdx.x; j < width / 4; j += blockDim.x)
        square += cuda_attn_res::Square4(cuda_attn_res::LoadChannels4(input, j));
    const float scale = cuda_attn_res::RowRmsScale(square, width, eps, scratch);
    for (int j = threadIdx.x; j < width / 4; j += blockDim.x) {
        float4 x = cuda_attn_res::LoadChannels4(input, j), w = reinterpret_cast<const float4 *>(norm)[j];
        const auto v = make_float4(x.x*scale*w.x, x.y*scale*w.y, x.z*scale*w.z, x.w*scale*w.w);
        if constexpr (std::is_same_v<Out, InklingQ81>) Q8RowStore::store<T>(output, j, v);
        else reinterpret_cast<cuda_attn_res::Channels4<Out> *>(output)[j] = {Out(v.x), Out(v.y), Out(v.z), Out(v.w)};
    }
}

template<class T> __global__ void encodeQ8Row(const T *src, InklingQ81 *out, int width) {
    cuda_launch::DependencyStart();
    src += size_t(blockIdx.y) * width;
    out += size_t(blockIdx.y) * (width / 32);
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < width / 4) {
        const auto x = reinterpret_cast<const cuda_attn_res::Channels4<T> *>(src)[j];
        Q8RowStore::store<T>(out, j, make_float4(float(x.x),float(x.y),float(x.z),float(x.w)));
    }
}

// Keep the checkpoint's interleaved Q8_0 blocks. Each thread reads eight
// signed codes at halfword alignment, so no second device copy is required.
template<int Width, bool Segmented, class T, bool Cap = false>
__global__ void q8DecodeProjection(const InklingQ80 *weights, const InklingQ81 *inputs,
                                   int qEnd, int kEnd, T *output, float limit) {
    const int row = blockIdx.x, tid = threadIdx.x;
    const auto *x = inputs + (Segmented ? (row >= qEnd) + (row >= kEnd) : 0) * (Width / 32);
    const auto *w = weights + size_t(row) * (Width / 32);
    // All immutable Q8_0 weights for this lane fit in at most three fragments.
    constexpr int fragments = (Width / 32 + 31) / 32;
    int lo[fragments], hi[fragments];
    float scale[fragments];
    const int word = (tid % 4) * 4;
#pragma unroll
    for (int f = 0; f < fragments; ++f) {
        const int i = tid / 4 + f * 32;
        if (i >= Width / 32) break;
        const auto *q = w[i].codes + word;
        lo[f] = unsigned(q[0]) | (unsigned(q[1]) << 16);
        hi[f] = unsigned(q[2]) | (unsigned(q[3]) << 16);
        scale[f] = float(w[i].scale);
    }
    cuda_launch::TriggerDependents();
    cuda_launch::DependencyWait();
    float result = 0;
#pragma unroll
    for (int f = 0; f < fragments; ++f) {
        const int i = tid / 4 + f * 32;
        if (i >= Width / 32) break;
        const int *codes = reinterpret_cast<const int *>(x[i].qs) + word / 2;
        result += scale[f] * __low2float(x[i].ds) * __dp4a(hi[f], codes[1], __dp4a(lo[f], codes[0], 0));
    }
    __shared__ float partial[3][32];
    int lane = tid % 32, warp = tid / 32;
    if (warp) partial[warp-1][lane] = result;
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int i = 0; i < 3; ++i) result += partial[i][lane];
        result = cuda::WarpSum(result);
        if (!lane) {
            if constexpr (Cap) result = limit * tanhf(result / limit);
            output[row] = T(result);
        }
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned q8WeightWord(const InklingQ80 &w, int i) {
    return unsigned(w.codes[2*i]) | (unsigned(w.codes[2*i+1]) << 16);
}

__device__ __forceinline__ void q8WindowDot(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                            unsigned x0, unsigned x1, int (&dot)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%10,%10,%10};"
        : "=r"(dot[0]), "=r"(dot[1]), "=r"(dot[2]), "=r"(dot[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(x0), "r"(x1), "r"(0));
}
#endif

// Each warp computes sixteen output channels for eight tokens with one integer
// MMA per quantization block. Scales are applied before the split-K reduction.
// The Q8_0 weights stay in checkpoint order, including at Q/K/V boundaries.
// Fragment layout reference: Basalt native_mmvq.cu (MIT, link above).
template<class Out, bool Cap = false, bool Ordered = false>
__global__ void q8WindowProjection(const InklingQ80 *weights, const InklingQ81 *input, Out *output,
                                   int rows, int blocks, int qEnd, int kEnd, int width, float limit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int warps = Ordered ? 8 : 4, sections = Ordered ? 4 : 1;
    const int tile = blockIdx.x, qTiles = (qEnd + 15) / 16, kTiles = (kEnd - qEnd + 15) / 16;
    const int part = tile < qTiles ? 0 : tile < qTiles + kTiles ? 1 : 2;
    const int begin = part == 0 ? 0 : part == 1 ? qEnd : kEnd;
    const int cols = (part == 0 ? qEnd : part == 1 ? kEnd : width) - begin;
    const int col = 16 * (tile - (part == 0 ? 0 : part == 1 ? qTiles : qTiles + kTiles));
    const int first = blockIdx.y * 8, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int group = lane / 4, word = lane % 4;
    const auto *a = weights + size_t(begin + min(col + group, cols-1)) * blocks;
    const auto *b = weights + size_t(begin + min(col + group + 8, cols-1)) * blocks;
    input += (size_t(part) * rows + first) * blocks;
    cuda_launch::TriggerDependents();
    cuda_launch::DependencyWait();
    float acc[sections][4]{};
    for (int k = warp; k < blocks; k += warps) {
        const auto &x = input[size_t(min(group, rows-first-1))*blocks+k];
        const auto *bytes = reinterpret_cast<const unsigned *>(x.qs);
        const unsigned w0 = q8WeightWord(a[k],word), w1 = q8WeightWord(b[k],word);
        const unsigned w2 = q8WeightWord(a[k],word+4), w3 = q8WeightWord(b[k],word+4);
        const unsigned x0 = bytes[word], x1 = bytes[word+4];
        const float d0 = __low2float(input[size_t(min(2*word,rows-first-1))*blocks+k].ds);
        const float d1 = __low2float(input[size_t(min(2*word+1,rows-first-1))*blocks+k].ds);
        const float scales[4] = {d0 * float(a[k].scale), d1 * float(a[k].scale),
                                 d0 * float(b[k].scale), d1 * float(b[k].scale)};
#pragma unroll
        for (int section = 0; section < sections; ++section) {
            // A reference lane dots eight codes. Mask the other three quarters
            // before tensor-core accumulation, keeping its first rounding point.
            const unsigned lo = !Ordered || word/2 == section ? x0 : 0;
            const unsigned hi = !Ordered || 2+word/2 == section ? x1 : 0;
            int dot[4];
            q8WindowDot(w0, w1, w2, w3, lo, hi, dot);
#pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[section][i] = fmaf(float(dot[i]), scales[i], acc[section][i]);
        }
    }
    output += size_t(begin) * rows + size_t(first) * cols;
    if constexpr (Ordered) {
        // Transpose back to the original 32-lane reduction for each result.
        __shared__ float partial[128][33];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int item = (2*word+(i&1))*16 + group + (i/2)*8;
#pragma unroll
            for (int s = 0; s < 4; ++s) partial[item][warp*4+s] = acc[s][i];
        }
        __syncthreads();
        for (int item = warp; item < 128; item += warps) {
            const int r = item / 16, c = item % 16;
            float v = partial[item][lane];
#pragma unroll
            for (int step = 16; step; step /= 2) v += __shfl_xor_sync(0xffffffffu, v, step);
            if (!lane && first+r < rows && col+c < cols) output[size_t(r)*cols+col+c] = Out(v);
        }
    } else {
        __shared__ float partial[warps][8][16];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            partial[warp][2*word+(i&1)][group+(i/2)*8] = acc[0][i];
        __syncthreads();
        const int row = threadIdx.x / 16, c = threadIdx.x % 16;
        if (first+row < rows && col+c < cols) {
            float v = (partial[0][row][c]+partial[2][row][c])+(partial[1][row][c]+partial[3][row][c]);
            if constexpr (Cap) v = limit*tanhf(v/limit);
            output[size_t(row)*cols+col+c] = Out(v);
        }
    }
#endif
}

template<class Out, bool Cap = false>
void launchQ8Window(const InklingQ80 *weights, const InklingQ81 *input, Out *output, int rows,
                    int inner, int qEnd, int kEnd, int width, float limit = 0.f) {
    int tiles = (qEnd+15)/16 + (kEnd-qEnd+15)/16 + (width-kEnd+15)/16;
    if (!cuda_launch::Launch<q8WindowProjection<Out, Cap>>(dim3(tiles, (rows+7)/8), 128, 0, cudaStreamPerThread,
            weights, input, output, rows, inner/32, qEnd, kEnd, width, limit))
        throw std::runtime_error("Inkling batched Q8 projection launch failed");
}

// The target's recurrent trajectory can cross a routing boundary at a single
// BF16 rounding point. Preserve GGUF MMVQ's eight-code dot and reduction tree
// even when three projections share one PDL launch.
template<class Out, int Tile = 8>
__global__ void q8VerifyProjection(const InklingQ80 *weights, const InklingQ81 *input, Out *output,
                                   int rows, int blocks, int qEnd, int kEnd, int width, int exactThreshold) {
    const int col = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32, first = blockIdx.y * Tile;
    const int live = min(rows - first, Tile);
    const int seg = (col >= qEnd) + (col >= kEnd);
    const int begin = seg == 0 ? 0 : seg == 1 ? qEnd : kEnd;
    const int stride = (seg == 0 ? qEnd : seg == 1 ? kEnd : width) - begin;
    cuda_launch::DependencyStart();
    if (col >= width) return;
    weights += size_t(col) * blocks;
    input += (size_t(seg) * rows + first) * blocks;
    const int parts = live <= 4 || live < exactThreshold ? 4 : 1;
    float sum[Tile]{};
    for (int part = 0; part < parts; ++part) {
        float acc[Tile]{};
        for (int k = lane / 4 + part * 8; k < blocks; k += 8 * parts) {
            const auto &weight = weights[k];
            const uint16_t *w = weight.codes + 4 * (lane % 4);
            const int a = unsigned(w[0]) | unsigned(w[1]) << 16;
            const int b = unsigned(w[2]) | unsigned(w[3]) << 16;
#pragma unroll
            for (int r = 0; r < Tile; ++r) {
                if (r >= live) break;
                const auto &activation = input[size_t(r) * blocks + k];
                const auto *v = reinterpret_cast<const int *>(activation.qs) + 2 * (lane % 4);
                acc[r] += float(weight.scale) * __low2float(activation.ds) * __dp4a(b, v[1], __dp4a(a, v[0], 0));
            }
        }
#pragma unroll
        for (int r = 0; r < Tile; ++r) sum[r] += acc[r];
    }
#pragma unroll
    for (int r = 0; r < Tile; ++r) {
        if (r >= live) break;
        float v = sum[r];
#pragma unroll
        for (int step = 16; step; step /= 2) v += __shfl_xor_sync(0xffffffffu, v, step);
        if (!lane) output[size_t(begin) * rows + size_t(first + r) * stride + col - begin] = Out(v);
    }
}
template<class Out>
void launchQ8Verify(const InklingQ80 *weights, const InklingQ81 *input, Out *out, int rows, int width,
                    int qEnd, int kEnd, int columns) {
    const int threshold = FastllmCudaGetLinearExactBatchThreshold();
    // The one-warp MMVQ reduction applies to each full window and to a final
    // tile of at least five rows. Smaller tails use its four-warp tree.
    const int tail = rows % 8;
    if (threshold <= 8 && (!tail || (tail >= 5 && tail >= threshold))) {
        const int tiles = (qEnd + 15)/16 + (kEnd - qEnd + 15)/16 + (columns - kEnd + 15)/16;
        if (!cuda_launch::Launch<q8WindowProjection<Out, false, true>>(
                dim3(tiles, (rows+7)/8), 256, 0, cudaStreamPerThread, weights, input, out, rows, width/32,
                qEnd, kEnd, columns, 0.f))
            throw std::runtime_error("Inkling ordered Q8 tensor-core projection launch failed");
        return;
    }
    if (!cuda_launch::Launch<q8VerifyProjection<Out>>(dim3((columns + 3)/4, (rows + 7)/8), 128, 0, cudaStreamPerThread,
            weights, input, out, rows, width / 32, qEnd, kEnd, columns, threshold))
        throw std::runtime_error("Inkling verify Q8 projection launch failed");
}
