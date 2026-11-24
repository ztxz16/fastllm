// Controller fusion follows vLLM Inkling ops/loop.py (Apache-2.0).
// The loop controller consumes two independently normalized rows concatenated
// along their channels. Produce the Linear input directly, including for prefill.
struct LoopStream {
    const float *dest, *msg, *norm;
};
struct LoopStreams {
    LoopStream items[3]; // query, key, value (or a single MLP stream)
};
template <typename T> __global__ void PairRMSNorm(LoopStreams streams, T *packed, int rows, int width, float eps) {
    int packedRow = blockIdx.x / 2;
    const LoopStream &s = streams.items[packedRow / rows];
    const float *x = (blockIdx.x % 2 ? s.msg : s.dest) + size_t(packedRow % rows) * width;
    __shared__ float scratch[16];
    float sq = 0.f;
    if (width % 4 == 0) {
        for (int j = threadIdx.x; j < width / 4; j += blockDim.x)
            sq += cuda_attn_res::Square4(reinterpret_cast<const float4 *>(x)[j]);
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) sq += x[j] * x[j];
    }
    float inv = cuda_attn_res::RowRmsScale(sq, width, eps, scratch);
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        packed[size_t(blockIdx.x) * width + j] = x[j] * inv;
}

template <class T, class Out> __global__ void BlendRMSNorm(LoopStreams streams, const float *gates, Out *out, int rows, int width, float eps) {
    const LoopStream &s = streams.items[blockIdx.x / rows];
    const size_t offset = size_t(blockIdx.x % rows) * width;
    out += size_t(blockIdx.x) * width;
    __shared__ float scratch[16];
    extern __shared__ __align__(16) float temp[];
    float *mixed = temp;
    if (size_t(width) * sizeof(float) > 16384) {
        if constexpr (std::is_same_v<Out, float>) mixed = out;
        else mixed = nullptr;
    }
    float sq = 0.f;
    gates += size_t(blockIdx.x) * 2 * width;
    auto blend = [&](int j) {
        size_t i = offset + j;
        return (1.f - tanhf(float(gates[width + j]))) * s.dest[i] + tanhf(float(gates[j])) * s.msg[i];
    };
    if (width % 4 == 0) {
        for (int j = threadIdx.x; j < width / 4; j += blockDim.x) {
            float4 v = make_float4(blend(4*j), blend(4*j+1), blend(4*j+2), blend(4*j+3));
            if (mixed) reinterpret_cast<float4 *>(mixed)[j] = v;
            sq += cuda_attn_res::Square4(v);
        }
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float v = blend(j);
            if (mixed) mixed[j] = v;
            sq += v*v;
        }
    }
    float inv = cuda_attn_res::RowRmsScale(sq, width, eps, scratch);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = mixed ? mixed[j] : blend(j);
        out[j] = Out(T(v * inv * s.norm[j]));
    }
}

// A warp reuses the same coefficient across independent rows. Preserve the
// singleton's per-lane FMA order and final shuffle tree before BF16 rounding.
template<int RowsPerWarp, int ColumnsPerWarp = 1>
__global__ void controllerDown(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output,
                               int rows, int columns) {
    cuda_launch::DependencyStart();
    constexpr int width = 4608, colsPerBlock = RowsPerWarp == 1 ? 2 : 4;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int first = (blockIdx.y * (4 / colsPerBlock) + warp / colsPerBlock) * RowsPerWarp;
    const int col = (blockIdx.x * colsPerBlock + warp % colsPerBlock) * ColumnsPerWarp;
    if (first >= rows) return;
    const size_t row = size_t(blockIdx.z) * rows + first;
    input += row * width;
    weight += (size_t(blockIdx.z) * columns + col) * width;
    float sum[ColumnsPerWarp][RowsPerWarp]{};
#pragma unroll
    for (int base = 0; base < width; base += 256) {
        flashinfer::vec_t<__nv_bfloat16, 8> values[RowsPerWarp];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r)
            values[r].load(input + (first + r < rows ? r : 0) * width + base + lane * 8);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int c = 0; c < ColumnsPerWarp; ++c) {
                const float w = float(weight[c * width + base + j * 32 + lane]);
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ >= 1000
                if constexpr (RowsPerWarp % 2 == 0) {
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; r += 2) {
                        const float2 pair = __ffma2_rn({float(values[r][j]), float(values[r + 1][j])},
                                                       {w, w}, {sum[c][r], sum[c][r + 1]});
                        sum[c][r] = pair.x;
                        sum[c][r + 1] = pair.y;
                    }
                } else
#endif
                {
#pragma unroll
                    for (int r = 0; r < RowsPerWarp; ++r)
                        sum[c][r] = fmaf(float(values[r][j]), w, sum[c][r]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < ColumnsPerWarp; ++c) {
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
            float dot = sum[c][r];
#pragma unroll
            for (int step = 16; step; step /= 2) dot += __shfl_down_sync(0xffffffff, dot, step);
            if (!lane && first + r < rows) output[(row + r) * columns + col + c] = __float2bfloat16_rn(dot);
        }
    }
}

__device__ __forceinline__ float controllerMix(float dest, float msg, float gain, float forget) {
    return fmaf(1.f - tanhf(forget), dest, __fmul_rn(tanhf(gain), msg));
}

// The MLP consumes FP32 storage containing the same rounded values as Q/K/V.
template<class Round, class Out>
__device__ __forceinline__ cuda_attn_res::Channels4<Out> controllerNorm4(float4 x, float inv, float4 w) {
    return {Out(Round(x.x * inv * w.x)), Out(Round(x.y * inv * w.y)),
            Out(Round(x.z * inv * w.z)), Out(Round(x.w * inv * w.w))};
}

using cuda::low_latency::projection::mma_bf16_16816;

// The three-matrix singleton controller uses BF16 tensor cores. Keep its
// K=16 accumulation boundaries and reuse each weight tile across queries.
__global__ void controllerQkvGates(const __nv_bfloat16 *hidden,
        const __nv_bfloat16 *weight, float *out, int rows, int width) {
    cuda_launch::DependencyStart();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int rank = 64;
    const int lane = threadIdx.x % 32, first = blockIdx.y * 8;
    const int channel = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + lane / 4;
    const int pair = (lane % 4) * 2, stream = blockIdx.z;
    const auto *input = hidden + (size_t(stream) * rows + min(first + lane / 4, rows - 1)) * rank + pair;
    weight += (size_t(stream) * 2 * width + channel) * rank + pair;
    float gates[2][4]{};
    auto load = [](const __nv_bfloat16 *p) { return *reinterpret_cast<const unsigned *>(p); };
#pragma unroll
    for (int k = 0; k < rank; k += 16) {
        const unsigned b0 = load(input + k), b1 = load(input + k + 8);
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const auto *w = weight + side * width * rank + k;
            mma_bf16_16816(gates[side], load(w), load(w + 8 * rank), load(w + 8), load(w + 8 * rank + 8), b0, b1);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = first + pair + i % 2;
        if (row >= rows) continue;
        const size_t offset = (size_t(stream) * rows + row) * (2 * width) + channel + (i / 2) * 8;
        out[offset] = gates[0][i];
        out[offset + width] = gates[1][i];
    }
#endif
}

// Reorder the checkpoint's [stream, gain/forget, channel, rank] weights into
// warp fragments. Interleave four 16-channel fragments for float4 residual loads.
struct ControllerGateTile {
    uint4 fragment[4][4][32]; // rank tile, channel tile, warp lane
};
constexpr int controllerWidth = 2304, controllerRank = 64, controllerRows = 8;
constexpr int controllerTiles = controllerWidth / 64, controllerThreads = controllerWidth / 4;
constexpr int controllerPitch = controllerWidth + 4;
constexpr int controllerSharedBytes = controllerRows * controllerPitch * sizeof(float);

__global__ void packControllerGates(const __nv_bfloat16 *weight, ControllerGateTile *tiles) {
    const int lane = threadIdx.x % 32, tile = threadIdx.x / 32;
    const int channel = blockIdx.x * 64 + (lane / 4) * 4 + tile;
    const int k = blockIdx.z * 16 + (lane % 4) * 2;
    const auto *w = weight + (blockIdx.y * controllerWidth + channel) * controllerRank + k;
    auto pair = [](const __nv_bfloat16 *p) { return *reinterpret_cast<const unsigned *>(p); };
    tiles[blockIdx.y * controllerTiles + blockIdx.x].fragment[blockIdx.z][tile][lane] =
        make_uint4(pair(w), pair(w + 32 * controllerRank), pair(w + 8), pair(w + 32 * controllerRank + 8));
}

// One block owns eight complete rows. Gain and forget never leave registers,
// and the mixed FP32 rows stay on chip until the output normalization.
template<class Out>
__global__ void prefillControllerMix(FASTLLM_ROW_PARAMETER LoopStreams streams, const __nv_bfloat16 *hidden,
                                     const ControllerGateTile *weights, Out *out, int rows, float eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(16) float mixed[][controllerPitch];
    __shared__ float partial[controllerRows][16];
    const int first = blockIdx.x * controllerRows, stream = blockIdx.y;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const LoopStream &s = streams.items[stream];
    const size_t hiddenRow = size_t(stream) * rows + min(first + lane / 4, rows - 1);
    const auto *h = hidden + hiddenRow * controllerRank + (lane % 4) * 2;
    uint2 activation[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const auto *p = reinterpret_cast<const unsigned *>(h + k * 16);
        activation[k] = make_uint2(p[0], p[4]);
    }
#pragma unroll
    for (int group = warp; group < controllerTiles; group += controllerThreads / 32) {
        const auto *gate = weights + 2 * stream * controllerTiles + group;
        float gain[4][4]{}, forget[4][4]{};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const uint4 g = gate[0].fragment[k][tile][lane], f = gate[controllerTiles].fragment[k][tile][lane];
                mma_bf16_16816(gain[tile], g.x, g.y, g.z, g.w, activation[k].x, activation[k].y);
                mma_bf16_16816(forget[tile], f.x, f.y, f.z, f.w, activation[k].x, activation[k].y);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = lane % 4 * 2 + i % 2, col = group * 64 + lane / 4 * 4 + i / 2 * 32;
            const size_t offset = size_t(min(first + row, rows - 1)) * controllerWidth + col;
            const float4 x = *reinterpret_cast<const float4 *>(s.dest + offset);
            const float4 m = *reinterpret_cast<const float4 *>(s.msg + offset);
            *reinterpret_cast<float4 *>(&mixed[row][col]) = make_float4(
                controllerMix(x.x, m.x, gain[0][i], forget[0][i]), controllerMix(x.y, m.y, gain[1][i], forget[1][i]),
                controllerMix(x.z, m.z, gain[2][i], forget[2][i]), controllerMix(x.w, m.w, gain[3][i], forget[3][i]));
        }
    }
    __syncthreads();
    // Match the 512-thread ControllerNormRows summation tree and its BF16 rounding.
    float4 values[controllerRows];
#pragma unroll
    for (int r = 0; r < controllerRows; ++r) {
        const auto *v = reinterpret_cast<const float4 *>(mixed[r]);
        values[r] = v[threadIdx.x];
        if (warp < 16) {
            float sq = cuda_attn_res::Square4(values[r]);
            if (threadIdx.x < controllerThreads - 512) sq += cuda_attn_res::Square4(v[threadIdx.x + 512]);
            sq = cuda::WarpSum(sq);
            if (lane == 0) partial[r][warp] = sq;
        }
    }
    __syncthreads();
    const float4 w = reinterpret_cast<const float4 *>(s.norm)[threadIdx.x];
#pragma unroll
    for (int r = 0; r < controllerRows; ++r) {
        if (first + r >= rows) break;
        const float inv = rsqrtf(cuda::WarpSum(lane < 16 ? partial[r][lane] : 0.f) / controllerWidth + eps);
        const size_t offset = (size_t(stream) * rows + first + r) * controllerWidth;
        auto *y = reinterpret_cast<cuda_attn_res::Channels4<Out> *>(out + offset);
        y[threadIdx.x] = controllerNorm4<__nv_bfloat16, Out>(values[r], inv, w);
    }
#endif
}

// The singleton BF16 GEMV reduces eight contiguous segments of the rank dimension.
// Reuse each pair of weight vectors across verification rows. Keep both gates
// in FP32 through the shared residual blend and RMS normalization.
template<int Tile>
__global__ void controllerMlpGates(const __nv_bfloat16 *hidden,
        const __nv_bfloat16 *weight, float *out, int rows, int width) {
    cuda_launch::DependencyStart();
    constexpr int rank = 64, lanes = 8;
    const int lane = threadIdx.x % lanes, channel = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    const int first = blockIdx.y * Tile;
    if (channel >= width) return;
    float gain[Tile]{}, forget[Tile]{};
    using Pack = cuda_attn_res::Channels4<__nv_bfloat16>;
#pragma unroll
    for (int j = 0; j < rank / lanes; j += 4) {
        const int col = lane * (rank / lanes) + j;
        const auto a = *reinterpret_cast<const Pack *>(weight + size_t(channel) * rank + col);
        const auto b = *reinterpret_cast<const Pack *>(weight + size_t(width + channel) * rank + col);
#pragma unroll
        for (int r = 0; r < Tile; ++r) {
            if (first + r >= rows) break;
            const auto x = *reinterpret_cast<const Pack *>(hidden + size_t(first + r) * rank + col);
            auto dot = [&](float sum, Pack v) {
                sum = fmaf(float(x.x), float(v.x), sum);
                sum = fmaf(float(x.y), float(v.y), sum);
                sum = fmaf(float(x.z), float(v.z), sum);
                return fmaf(float(x.w), float(v.w), sum);
            };
            gain[r] = dot(gain[r], a);
            forget[r] = dot(forget[r], b);
        }
    }
    const unsigned mask = __activemask();
#pragma unroll
    for (int r = 0; r < Tile; ++r) {
        if (first + r >= rows) break;
#pragma unroll
        for (int step = lanes / 2; step; step /= 2) {
            gain[r] += __shfl_down_sync(mask, gain[r], step, lanes);
            forget[r] += __shfl_down_sync(mask, forget[r], step, lanes);
        }
        if (!lane) {
            const size_t i = size_t(first + r) * (2 * width) + channel;
            out[i] = gain[r];
            out[i + width] = forget[r];
        }
    }
}

// A controller row fits in registers on the packed path. Pair carries both
// sums through the same block barrier. Blend retains the mixed FP32 channels
// until the scale is available, including for narrowed outputs.
template <bool Pair, class T, class Out, int Width, bool PackDown = false, bool Pdl = false>
__global__ void ControllerNormRows(FASTLLM_ROW_PARAMETER LoopStreams streams, const float *gates, Out *out, float eps) {
    constexpr int threads = 512, cols = Width / 4;
    constexpr int packs = (cols + threads - 1) / threads;
    const LoopStream &stream = streams.items[blockIdx.y];
    const size_t row = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t offset = size_t(blockIdx.x) * cols;
    const auto dest = reinterpret_cast<const float4 *>(stream.dest) + offset;
    const auto msg = reinterpret_cast<const float4 *>(stream.msg) + offset;
    float4 a[packs], b[packs];
    float4 norms[Pdl && !Pair ? packs : 1];
    if constexpr (Pdl && !Pair) {
        // Down consumed these states before producing the gates' input. Only
        // the gates themselves are written by the immediately preceding grid.
#pragma unroll
        for (int i = 0; i < packs; ++i) {
            const int j = threadIdx.x + i * threads;
            if (j < cols) {
                a[i] = dest[j];
                b[i] = msg[j];
                norms[i] = reinterpret_cast<const float4 *>(stream.norm)[j];
            }
        }
    }
    if constexpr (Pdl) cuda_launch::DependencyStart();
    float sa = 0.f, sb = 0.f;
#pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int j = threadIdx.x + i * threads;
        if (j < cols) {
            float4 x = Pdl && !Pair ? a[i] : dest[j], m = Pdl && !Pair ? b[i] : msg[j];
            if constexpr (Pair) {
                a[i] = x;
                b[i] = m;
                sb += cuda_attn_res::Square4(m);
            } else {
                auto g = reinterpret_cast<const float4 *>(gates) + row * 2 * cols + j;
                float4 gain = g[0], forget = g[cols];
                a[i] = make_float4(controllerMix(x.x, m.x, gain.x, forget.x), controllerMix(x.y, m.y, gain.y, forget.y),
                                   controllerMix(x.z, m.z, gain.z, forget.z), controllerMix(x.w, m.w, gain.w, forget.w));
            }
            sa += cuda_attn_res::Square4(a[i]);
        }
    }
    constexpr int warps = threads / 32;
    __shared__ float partial[Pair ? 2 : 1][warps];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    sa = cuda::WarpSum(sa);
    if constexpr (Pair) sb = cuda::WarpSum(sb);
    if (lane == 0) {
        partial[0][warp] = sa;
        if constexpr (Pair) partial[1][warp] = sb;
    }
    __syncthreads();
    auto scale = [&](int which) {
        return rsqrtf(cuda::WarpSum(lane < warps ? partial[which][lane] : 0.f) / Width + eps);
    };
    sa = scale(0);
    if constexpr (Pair) sb = scale(1);
    if constexpr (PackDown) {
        static_assert(Pair && Width % 256 == 0 && std::is_same_v<Out, __nv_bfloat16>);
        // Preserve the reference warp sums above. Adjacent lanes read adjacent
        // source channels, then store eight consecutive dot-product terms per lane.
        for (int v = threadIdx.x; v < 2 * Width / 8; v += threads) {
            const bool message = v >= Width / 8;
            const int slot = v % (Width / 8), start = slot / 32 * 256 + slot % 32;
            const auto *source = (message ? stream.msg : stream.dest) + size_t(blockIdx.x) * Width + start;
            flashinfer::vec_t<Out, 8> values;
#pragma unroll
            for (int j = 0; j < 8; ++j) values[j] = Out(source[32 * j] * (message ? sb : sa));
            values.store(out + row * 2 * Width + v * 8);
        }
    } else {
#pragma unroll
        for (int i = 0; i < packs; ++i) {
            const int j = threadIdx.x + i * threads;
            if (j < cols) {
                float4 v = a[i];
                if constexpr (std::is_same_v<Out, InklingQ81>) {
                    static_assert(!Pair);
                    float4 w = Pdl ? norms[i] : reinterpret_cast<const float4 *>(stream.norm)[j];
                    Q8RowStore::store<T>(out, row * cols + j, make_float4(v.x*sa*w.x, v.y*sa*w.y, v.z*sa*w.z, v.w*sa*w.w));
                } else if constexpr (Pair) {
                    auto y = reinterpret_cast<cuda_attn_res::Channels4<Out> *>(out) + row * 2 * cols;
                    y[j] = {Out(v.x * sa), Out(v.y * sa), Out(v.z * sa), Out(v.w * sa)};
                    v = b[i];
                    y[cols + j] = {Out(v.x * sb), Out(v.y * sb), Out(v.z * sb), Out(v.w * sb)};
                } else {
                    float4 w = Pdl ? norms[i] : reinterpret_cast<const float4 *>(stream.norm)[j];
                    auto y = reinterpret_cast<cuda_attn_res::Channels4<Out> *>(out) + row * cols;
                    y[j] = controllerNorm4<T, Out>(v, sa, w);
                }
            }
        }
    }
}

template<typename Out>
void launchBatchedController(LoopStreams streams, const Rows<__nv_bfloat16> &pair, Rows<Out> &out,
                             const __nv_bfloat16 *down, const __nv_bfloat16 *gate,
                             int count, int rank, int smCount, float eps, InklingQ81 *encoded = nullptr) {
    using T = __nv_bfloat16;
    const int n = pair.rows / count, d = pair.cols / 2;
    auto launch = [](bool success) {
        if (!success) throw std::runtime_error("Inkling batched controller launch failed");
    };
    auto prepare = [&](auto pack) {
        launch(cuda_launch::Launch<ControllerNormRows<true, T, T, 2304, decltype(pack)::value, true>>(
            dim3(n, count), 512, 0, cudaStreamPerThread, streams, static_cast<const float *>(nullptr), pair.p, eps));
    };
    const bool fullBatch = n >= smCount;
    // Preserve the single-row reduction for small verify batches. Full batches
    // feed ordinary [group, row, channel] RMSNorm output directly to GEMM.
    if (count == 1 || fullBatch) prepare(std::false_type{});
    Rows<T> hidden(count * n, rank);
    if (!fullBatch || !FastllmCudaBf16ProjectionGemm(pair.p, down, hidden.p, n, 2 * d, rank, count)) {
        if (count > 1) prepare(std::true_type{});
        if (count == 1)
            FastllmCudaBf16ExactRows(pair.p, down, hidden.p, n, 2 * d, rank);
        else if (rank % 32 == 0 && (int64_t(n) + 7) / 8 * (rank / 32) * count >= smCount)
            launch(cuda_launch::Launch<controllerDown<8, 8>>(
                dim3(rank / 32, (n + 7) / 8, count), 128, 0, cudaStreamPerThread, pair.p, down, hidden.p, n, rank));
        else if (fullBatch)
            launch(cuda_launch::Launch<controllerDown<8>>(dim3(rank / 4, (n + 7) / 8, count), 128, 0, cudaStreamPerThread,
                pair.p, down, hidden.p, n, rank));
        else if (size_t(n / 4) * (rank / 4) * count >= size_t(smCount))
            launch(cuda_launch::Launch<controllerDown<4>>(dim3(rank / 4, (n + 3) / 4, count), 128, 0, cudaStreamPerThread,
                pair.p, down, hidden.p, n, rank));
        else
            launch(cuda_launch::Launch<controllerDown<1>>(dim3(rank / 2, (n + 1) / 2, count), 128, 0, cudaStreamPerThread,
                pair.p, down, hidden.p, n, rank));
    }
    Matrix gates(count * n, 2 * d);
    if (fullBatch && count > 1) {
        // The QKV gates use K=16 BF16 MMA with FP32 accumulation in both paths.
        Data x(BFLOAT16, {count * n, rank}, CUDA, hidden.p);
        Data w(BFLOAT16, {count * 2 * d, rank}, CUDA, const_cast<T *>(gate));
        Data y(FLOAT32, {count * n, 2 * d}, CUDA, gates.p);
        launch(FastllmCudaBatchMatMulTransB(x, w, y, n * rank, 2 * d * rank, n * 2 * d,
                                            rank, rank, count, n, rank, 2 * d, 1.f));
    } else if (count == 1)
        launch(cuda_launch::Launch<controllerMlpGates<4>>(dim3(d / 16, (n + 3) / 4), 128, 0, cudaStreamPerThread,
            hidden.p, gate, gates.p, n, d));
    else
        launch(cuda_launch::Launch<controllerQkvGates>(dim3(d / 64, (n + 7) / 8, count), 128, 0, cudaStreamPerThread,
            hidden.p, gate, gates.p, n, d));
    if (encoded) {
        if (!cuda_launch::Launch<ControllerNormRows<false, T, InklingQ81, 2304, false, true>>(
                dim3(n, count), 512, 0, cudaStreamPerThread, streams, gates.p, encoded, eps))
            throw std::runtime_error("Inkling batched controller quantization failed");
    } else launch(cuda_launch::Launch<ControllerNormRows<false, T, Out, 2304, false, true>>(
        dim3(n, count), 512, 0, cudaStreamPerThread, streams, gates.p, out.p, eps));
}
