// Inkling 2 Mini fixed-loop inference. Architecture and checkpoint layout:
// https://huggingface.co/thinkingmachines/Inkling-2-Mini
// https://vllm-project.github.io/2026/11/20/inkling2-mini.html
#include "devices/cuda/fastllm-inkling-window.h"
#include "runtime.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/persistent_worker_group.h"
#include "executor.h"
#include "gguf.h"
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include "devices/cuda/fastllm-cuda-gguf-projections.h"
#include "utils/recurrent_prefix_cache.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda-attn-res.cuh"
#include "devices/cuda/low_latency/bf16_projection.cuh"
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include "attention/variants.cuh"
#include "attention/decode.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/prefill.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include "devices/cuda/fastllm-fold-sampling.cuh"
#include "devices/cuda/fastllm-bf16-warp-dot.cuh"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>

void LaunchFastllmGemmBf16Bf16(__nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, __nv_bfloat16 *, int, int, int);

bool LaunchFastllmGroupedBf16Projection(const __nv_bfloat16 *const *, const __nv_bfloat16 *, __nv_bfloat16 *,
                                      int, int, const int *);

namespace fastllm {
namespace inkling2 {
thread_local DecodeGraphScratch *DecodeGraphScratch::active = nullptr;

std::shared_ptr<void> DecodeGraphScratch::Acquire(size_t bytes) {
    auto &blocks = *backing;
    if (sealed) {
        if (cursor >= order.size()) throw std::runtime_error("Unexpected Inkling decode temporary");
        const auto &block = blocks[order[cursor++]];
        if (block.bytes < bytes || block.storage.use_count() != 1)
            throw std::runtime_error("Inkling decode scratch lifetime or shape changed");
        return block.storage;
    }
    size_t best = blocks.size();
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto &b = blocks[i];
        if (b.storage.use_count() == 1 && b.bytes >= bytes &&
            (best == blocks.size() || b.bytes < blocks[best].bytes)) best = i;
    }
    if (best == blocks.size()) {
        void *ptr = nullptr;
        check(cudaMalloc(&ptr, bytes));
        blocks.push_back({bytes, std::shared_ptr<void>(ptr, [](void *p) { cudaFree(p); })});
    }
    order.push_back(best);
    return blocks[best].storage;
}
DataType projectionType(int gpu) {
    int major;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu));
    return major >= 8 ? BFLOAT16 : FLOAT16;
}
namespace {
__global__ void normalizeRows(const float *x, const float *scale, float *y, int width, int scaleRows,
                              float eps, float clip) {
    __shared__ float tmp[threads];
    const size_t base = size_t(blockIdx.x) * width;
    float sum = 0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float v = x[base + c];
        if (clip > 0 && scale)
            v *= scale[(blockIdx.x % scaleRows) * width + c];
        sum += v * v;
    }
    sum = blockSum(sum, tmp);
    float factor = clip > 0 ? fminf(1.0f, clip / (sqrtf(sum) + 1e-6f)) : rsqrtf(sum / width + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        y[base + c] = x[base + c] * factor * (scale ? scale[(blockIdx.x % scaleRows) * width + c] : 1.0f);
}
Matrix norm(const Matrix &x, const float *w, float eps, int width = 0, int scaleRows = 1, float clip = 0) {
    auto out = Matrix::temporary(x.rows, x.cols);
    if (!width)
        width = x.cols;
    if (clip <= 0 && w && scaleRows == 1) {
        CudaView src(FLOAT32, {x.rows * (x.cols / width), width}, x.p);
        CudaView dst(FLOAT32, src.dims, out.p);
        CudaView weight(FLOAT32, {width}, const_cast<float *>(w));
        FastllmCudaRMSNorm(src, weight, dst, eps);
        return out;
    }
    normalizeRows<<<x.rows *(x.cols / width), threads, 0, cudaStreamPerThread>>>(x.p, w, out.p, width,
                                                                                 scaleRows, eps, clip);
    return out;
}
template<class T> __global__ void rankEmbedding(const float *ids, const float *weight, T *output, int first, int vocab, int hidden) {
    const int token = int(ids[blockIdx.x]) - first;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x)
        output[size_t(blockIdx.x) * hidden + col] = unsigned(token) < unsigned(vocab) ?
            weight[size_t(token) * hidden + col] : 0.f;
}

template<class T> __global__ void unpackVocabRanks(const T *gathered, T *output, int rows, int vocab, int shard) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * vocab) return;
    const int col = i % vocab;
    output[i] = gathered[(size_t(col / shard) * rows + i / vocab) * shard + col % shard];
}

template <typename T> __global__ void softcap(const T *projected, float *logits, size_t n, float limit) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) logits[i] = limit * tanhf(float(projected[i]) / limit);
}

// Split row-major packed output into contiguous [tokens, channels] tensors in
// the same pass that widens to the attention accumulator type.
struct ProjectionParts { int widths[8]; };
template <typename T, typename U>
__global__ void unpackProjection(const T *src, U *dst, ProjectionParts parts, int rows, int stride) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * stride) return;
    int channel = i % stride, start = 0;
    for (int width : parts.widths) {
        if (channel < start + width) {
            dst[size_t(start) * rows + (i / stride) * width + channel - start] = U(src[i]);
            return;
        }
        start += width;
    }
}

#include "draft-fold.cuh"
#include "loop-controller.cuh"
#include "single-projection.cuh"

template <class T, class Residual> __global__ void conv(const T *x, const float *history, const float *weight, const float *dynamic,
                     Residual *out, int n, int width, int taps, int heads, const Residual *residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int row = i / width, c = i % width;
    float v = x[i];
    for (int t = 0; t < taps; ++t) {
        int r = row - t;
        float a = r < 0 ? history[(taps - 1 + r) * width + c] : float(x[r * width + c]);
        float w = dynamic ? weight[t * width + c] + dynamic[(row * heads + c / (width / heads)) * taps + t]
                          : weight[c * taps + taps - 1 - t];
        v += a * w;
    }
    float y = v * rsqrtf(1.0f + taps * (dynamic ? 2 : 1));
    out[i] = residual ? __fadd_rn(y, float(residual[i])) : y;
}
#include "paged-kv.cuh"
#include "attention.cuh"
#include "prefill.cuh"
#include "paged-verify-attention.cuh"
template <bool Timeline = false, typename T, typename U> __device__ float updateConvChannel(const T *input, const DecodeVisit *visit, int index,
                           const float *weight, const U *dynamic, int d, int taps, int heads, int c, int priorRows, int inputStride) {
    float *history = visit->history[index];
    float v = input[c];
    for (int t = 0; t < taps; ++t) {
        float a;
        if constexpr (Timeline)
            a = t <= priorRows ? float(input[c - ptrdiff_t(t) * inputStride]) : history[(taps - 1 - t) * d + c];
        else
            a = t ? history[(taps - 1 - t) * d + c] : float(input[c]);
        float w = dynamic ? weight[t * d + c] + float(dynamic[(c / (d / heads)) * taps + t])
                          : weight[c * taps + taps - 1 - t];
        v += a * w;
    }
    if constexpr (Timeline) {
        history[(taps - 1) * d + c] = input[c];
    } else {
        for (int t = 0; t + 1 < taps - 1; ++t)
            history[t * d + c] = history[(t + 1) * d + c];
        if (taps > 1)
            history[(taps - 2) * d + c] = input[c];
    }
    return v * rsqrtf(1.0f + taps * (dynamic ? 2 : 1));
}


template <bool Timeline = false, typename T, typename Residual, bool Pdl = false> __global__ void decodeConv(const T *input, const DecodeVisit *visit, int index, const float *weight,
                           const float *dynamic, Residual *output, int d, int taps, int heads, const Residual *residual, int chain = 1) {
    if constexpr (Pdl) cuda_launch::DependencyStart();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d)
        return;
    int row = blockIdx.y;
    size_t offset = size_t(row) * d;
    if (dynamic) dynamic += size_t(row) * heads * taps;
    if (residual) residual += offset;
    output += offset;
    float y = updateConvChannel<Timeline>(input + offset, visit + row, index, weight, dynamic, d, taps, heads, c,
                                         row % chain, d);
    output[c] = residual ? __fadd_rn(y, float(residual[c])) : y;
}
// The W_o warp has finished one channel. Preserve its BF16 rounding before
// updating the four-tap attention history and writing the residual directly.
template<class R>
struct StoreAttentionResidual {
    const DecodeVisit *visit;
    const float4 *weights;
    const R *skip;
    int width, taps;

    __device__ void operator()(R *out, int row, float dot) const {
        const float x = float(__float2bfloat16_rn(dot));
        float *history = visit->history[2] + row;
        const float4 w = weights[row];
        const float h1 = history[width], h2 = history[2 * width];
        const float filtered = fmaf(history[0], w.x, fmaf(h1, w.y, fmaf(h2, w.z, fmaf(x, w.w, x))));
        history[0] = h1;
        history[width] = h2;
        history[2 * width] = x;
        const float y = __fmul_rn(filtered, rsqrtf(float(1 + taps)));
        out[row] = skip ? __fadd_rn(y, float(skip[row])) : y;
    }
};

// Each warp owns adjacent residual channels. The input to the four-tap
// filter is the FP32 sum of the already weighted BF16 expert rows.
template<class R>
__global__ void convolveExpertRows(const __nv_bfloat16 *rows, int count, const DecodeVisit *visit,
                                   const float4 *coeff, int width, R *residual) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool live = c < width;
    const float4 w = live ? coeff[c] : float4{};
    cuda_launch::DependencyStart();
    if (!live) return;
    float x = 0.f;
    for (int r = 0; r < count; ++r) x += float(rows[r * width + c]);
    float *h = visit->history[3] + c;
    float y = x + x * w.w;
    y += h[2 * width] * w.z;
    y += h[width] * w.y;
    y += h[0] * w.x;
    h[0] = h[width];
    h[width] = h[2 * width];
    h[2 * width] = x;
    residual[c] = __fadd_rn(float(residual[c]), __fmul_rn(y, rsqrtf(5.f)));
}
template<class T> __global__ void storeDecodeKv(const DecodeVisit *visit, const T *key, const T *value, int width) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < width) {
        const int row = blockIdx.y;
        visit += row;
        key += size_t(row) * width;
        value += size_t(row) * width;
        const auto &p = visit->pages;
        static_cast<T *>(p.key)[p.offset(visit->past, false, width) + c] = key[c];
        static_cast<T *>(p.value)[p.offset(visit->past, true, width) + c] = value[c];
    }
}
#include "attention-input.cuh"
// A channel's four-tap weights and its last three samples are shared across four rows.
template<class T, class R>
__global__ void prefillResidualConv(const T *input, const float4 *weights, const float *history,
                                   const R *skip, R *output, float *next, int rows, int width, int taps) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x, first = blockIdx.y * 4;
    if (col >= width) return;
    const float4 w = weights[col];
    const size_t start = size_t(first) * width + col;
    float oldest = first ? float(input[start - 3 * width]) : (history ? history[col] : 0.f);
    float middle = first ? float(input[start - 2 * width]) : (history ? history[width + col] : 0.f);
    float recent = first ? float(input[start - width]) : (history ? history[2 * width + col] : 0.f);
    const float scale = rsqrtf(1.f + taps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int row = first + j;
        if (row >= rows) break;
        const size_t idx = start + size_t(j) * width;
        const float x = float(input[idx]);
        float y = fmaf(x, w.w, x);
        y = fmaf(recent, w.z, y);
        y = fmaf(middle, w.y, y);
        y = __fmul_rn(fmaf(oldest, w.x, y), scale);
        output[idx] = skip ? __fadd_rn(y, float(skip[idx])) : y;
        if (row >= rows - 3) next[(row - rows + 3) * width + col] = x;
        oldest = middle;
        middle = recent;
        recent = x;
    }
}

template <class T, class Residual>
void convolution(const Rows<T> &x, const float *w, const Matrix &dyn, int heads, int taps, Matrix &hist,
                 const Residual *residual, Residual *out) {
    const int count = taps - 1, d = x.cols;
    if (taps == 4 && !dyn.p && x.rows >= 4 && x.rows <= 4 * 65535) {
        Matrix next(count, d);
        prefillResidualConv<<<dim3((d + 127) / 128, (x.rows + 3) / 4), 128, 0, cudaStreamPerThread>>>(
            x.p, reinterpret_cast<const float4 *>(w), hist.p, residual, out, next.p, x.rows, d, taps);
        hist = std::move(next);
        return;
    }
    if (!hist.p) {
        hist = Matrix(count, d);
        if (count)
            check(cudaMemsetAsync(hist.p, 0, size_t(count) * d * sizeof(float), cudaStreamPerThread));
    }
    conv<<<(x.rows * d + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, hist.p, w, dyn.p, out, x.rows, d,
                                                                    taps, heads, residual);
    if (count) {
        Matrix next(count, d);
        int keep = std::max(0, count - x.rows);
        if (keep)
            check(cudaMemcpyAsync(next.p, hist.p + size_t(x.rows) * d, size_t(keep) * d * sizeof(float),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        const auto *tail = x.p + size_t(std::max(0, x.rows - count)) * d;
        auto *dst = next.p + size_t(keep) * d;
        const size_t elements = size_t(count - keep) * d;
        if constexpr (std::is_same_v<T, float>)
            check(cudaMemcpyAsync(dst, tail, elements * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        else
            castProjection<<<(elements + 255) / 256, 256, 0, cudaStreamPerThread>>>(tail, dst, elements);
        hist = std::move(next);
    }
}

// FlashInfer loads a vectorized head. Odd widths use zero padding in Q/K/V
// storage; softmax scaling still uses the model's actual head dimension.
int attentionWidth(int d) { return d <= 64 ? 64 : 128; }

Matrix resizeHeads(const Matrix &input, int from, int to) {
    if (from == to) return input;
    const int heads = input.cols / from;
    Matrix out(input.rows, heads * to);
    if (to > from)
        check(cudaMemsetAsync(out.p, 0, size_t(out.rows) * out.cols * sizeof(float), cudaStreamPerThread));
    check(cudaMemcpy2DAsync(out.p, to * sizeof(float), input.p, from * sizeof(float),
                            std::min(from, to) * sizeof(float), size_t(heads) * input.rows,
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    return out;
}
using cuda_attn_res::Query;
template <class Round = float, class Out = float, class In>
Rows<Out> mix(const std::vector<Rows<In>> &states, const Rows<In> &last, const Query *queries, int nq,
           float eps, bool normalizeValues) {
    using Sources = cuda_attn_res::SourceList<In>;
    int count = int(states.size()) + 1;
    Buffer<const In *> table(count > Sources::Inline ? count : 0);
    Sources sources{};
    for (int first = 0; first < count; first += Sources::Inline) {
        int n = std::min(Sources::Inline, count - first);
        for (int i = 0; i < n; ++i)
            sources.items[i] = first + i == count - 1 ? last.p : states[first + i].p;
        if (table.p)
            cuda_attn_res::StoreSources<<<1, Sources::Inline, 0, cudaStreamPerThread>>>(
                table.p + first, sources, n);
    }
    sources.table = table.p;
    auto out = Rows<Out>::temporary(nq * last.rows, last.cols);
    Buffer<float> staging(cuda_attn_res::NeedsStaging<Out>(count, queries, nq, last.cols)
                            ? size_t(out.rows) * out.cols : 0);
    cuda_attn_res::Launch<Round>(sources, count, queries, nq, last.rows, last.cols, eps, normalizeValues,
                          out.p, cudaStreamPerThread, staging.p);
    return out;
}
struct DecodeInput {
    float token = 0;
    int topK = 1;
    float temperature = 1.f, topP = 1.f;
    uint64_t seed = 0, offset = 0;
};
__global__ void unpackDecodeInputs(const DecodeInput *in, int n, float *tokens, int *k, float *temperature,
                                   float *p, uint64_t *seeds, uint64_t *offsets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    tokens[i] = in[i].token;
    k[i] = in[i].topK;
    temperature[i] = in[i].temperature;
    p[i] = in[i].topP;
    seeds[i] = in[i].seed;
    offsets[i] = in[i].offset;
}
// A chained batch retains the same page tables and sampled rows until the next page boundary.
__global__ void advanceBatchDecode(DecodeInput *inputs, DecodeVisit *visits, const int *sampled,
                                   int capacity, int live, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity * slots || i % capacity >= live) return;
    ++visits[i].past;
    if (i < capacity) {
        inputs[i].token = float(sampled[i]);
        ++inputs[i].offset;
    }
}
bool supportsDeviceSampling(const GenerationConfig &s) {
    return ordinarySampling(s) && !s.IsSimpleGreedy() && !s.output_logits && !s.tool_call_grammar &&
           !s.tool_call_content_sampling_enabled && !s.tool_call_name_constraint_enabled &&
           !s.tool_call_parameter_name_constraint_enabled;
}
bool speculativeSampling(const GenerationConfig &s) {
    return s.output_token_least == 0 && supportsDeviceSampling(s);
}
std::mt19937_64 &samplingRng() {
    static thread_local std::mt19937_64 gen(std::random_device{}());
    return gen;
}
uint64_t newSamplingSeed() {
    return samplingRng()();
}
#include "sharded-topk.cuh"
#include "dspark.cuh"
#include "prefix-cache.cuh"
struct Request {
    std::vector<ImageSpan> images;
    std::shared_ptr<InklingPrefix> restored;
    std::vector<int> prefixTokens;
    long long prefixId = 0;
    int64_t nextDecodeCheckpoint = 0;
    std::unique_ptr<DraftState> draft;
    int promptLength = 0, limit = 0;
    int gpu;
    uint64_t seed = newSamplingSeed();
    explicit Request(int device) : gpu(device) {}
    ~Request() {
        int previous = gpu;
        cudaGetDevice(&previous);
        if (previous != gpu)
            cudaSetDevice(gpu);
        if (draft && draft->reader) cudaEventSynchronize(draft->reader->event);
        if (draft && draft->rounds)
            printf("[Inkling DSpark] rounds=%d proposed=%d accepted=%d mean_round_ms=%.3f\n",
                   draft->rounds, draft->proposed, draft->accepted, draft->roundMs / draft->rounds);
        draft.reset();
        images.clear();
        visits.clear();
        if (previous != gpu)
            cudaSetDevice(previous);
    }
    int length = 0;
    std::vector<VisitCache> visits;
};
struct Segment {
    Request *request;
    int offset, length;
    const DecodeVisit *decode = nullptr;
    Verification *verify = nullptr;
    const DecodeVisit *timeline = nullptr;
};
template <typename Residual = float, typename T> Rows<Residual> convolutions(const Rows<T> &input, const float *weight, const Matrix &dynamic, int heads, int taps,
                    const std::vector<Segment> &batch, size_t slot, int index, const Rows<Residual> &residual = {}, bool pdl = false) {
    if (batch.front().timeline) {
        Rows<Residual> out(input.rows, input.cols);
        const auto *visits = batch.front().timeline + slot * input.rows;
        const int chain = batch.front().length;
        auto kernel = chain == 1 ? decodeConv<false, T, Residual> : decodeConv<true, T, Residual>;
        kernel<<<dim3((input.cols + 255) / 256, input.rows), 256, 0, cudaStreamPerThread>>>(
            input.p, visits, index, weight, dynamic.p, out.p, input.cols, taps, heads, residual.p, chain);
        return out;
    }
    if (batch.front().decode) {
        Rows<Residual> out(1, input.cols);
        if (pdl) {
            if (!cuda_launch::Launch<decodeConv<false, T, Residual, true>>((input.cols + 255) / 256, 256, 0, cudaStreamPerThread,
                    input.p, batch.front().decode + slot, index, weight, dynamic.p, out.p, input.cols, taps, heads, residual.p, 1))
                throw std::runtime_error("Inkling residual convolution launch failed");
            return out;
        }
        decodeConv<<<(input.cols + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            input.p, batch.front().decode + slot, index, weight, dynamic.p, out.p, input.cols, taps, heads, residual.p);
        return out;
    }
    auto result = batch.size() == 1 ? Rows<Residual>::temporary(input.rows, input.cols)
                                    : Rows<Residual>(input.rows, input.cols);
    for (const auto &seq : batch) {
        auto rows = input.slice(seq.offset, seq.length);
        auto dyn = dynamic.p ? dynamic.slice(seq.offset, seq.length) : Matrix{};
        auto &history = seq.request->visits[slot].convState[index];
        if (seq.verify) seq.verify->history(history, history, rows, rows.p, rows.cols, rows.cols, rows.rows, taps-1);
        const size_t offset = size_t(seq.offset) * input.cols;
        convolution(rows, weight, dyn, heads, taps, history,
                    residual.p ? residual.p + offset : nullptr, result.p + offset);
    }
    return result;
}
class Decoder final : public Runtime {
    Config c;
    enum class Projection { Default, PdlDecode, DecodeHead, AttentionMma, DraftHead };
    bool decodeArithmetic = false;
    const FastllmBf16MmaProjections *bf16Mma = nullptr;
    Placement placement;
    basellm &model;
    WeightMap &weights;
    DataType kvType = DATA_AUTO_NONE, residualType = DATA_AUTO_NONE;
    ExpertLayers experts;
    uint64_t requestSeed = 0;
    std::vector<uint64_t> promptSeeds;
    std::mutex mutex;
    std::mutex prefixMutex;
    std::atomic<bool> prefixEnabled;
    RecurrentPrefixStore<InklingPrefix> prefixes;
    long long prefixSerial = 0;
    cudaGraphExec_t decodeGraph = nullptr;
    DecodeGraphScratch graphScratch;
    Buffer<DecodeInput> graphInput;
    Buffer<int> graphSample;
    Buffer<DecodeVisit> graphVisits;
    struct Ticket {
        struct Header {
            DecodeInput input;
            int token;
        };
        Header *host = nullptr;
        cudaEvent_t ready = nullptr;
        DecodeVisit *visits() const { return reinterpret_cast<DecodeVisit *>(host + 1); }
        void init(size_t n) {
            static_assert(alignof(Header) >= alignof(DecodeVisit), "unaligned visit staging");
            if (!host) check(cudaMallocHost(&host, sizeof(Header) + n * sizeof(DecodeVisit)));
            if (!ready) check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        }
        ~Ticket() {
            if (ready) cudaEventDestroy(ready);
            if (host) cudaFreeHost(host);
        }
    } tickets[2];
    DecodeInput hostInput{};
    std::vector<DecodeVisit> hostVisits;
    Matrix graphLogits, unitScale, splitScratch;
    bool pagedDecodePdl = false;
    std::optional<InklingWindowKernel> compactWindow, decodeWindow;
    InklingWindowKernel singleWindowKernel;
    int singleWindowParts() const { return splitScratch.rows >= 32 ? 32 : 16; }
    bool useWindowDecodeTile(int rows) const {
        // One CTA scans a full window per request and KV head. Smaller batches
        // benefit from splitting the window over additional SMs.
        return rows > 1 && rows <= 65535 && smCount > 0 && int64_t(rows) * c.kvHeads * 2 >= smCount;
    }
    std::vector<Buffer<__nv_bfloat16>> windowTables;
    FastllmBf16MmaProjections::Project decodeBf16Gemv = nullptr;
    FastllmBf16MmaProjections::Project rankOutputGemv = nullptr;
    bool backboneBf16 = false;
    int smCount = 0, sm120GlobalSplits = 0, verifyAttentionSplits = 0;
    bool tensorPrefill = false, gatedGlobalPrefill = false, gatedWindowPrefill = false;
    bool bf16WindowBias = false, batchedController = false, fusedPrefillController = false, decodePdlSupported = false;

    Data &weight(const std::string &key) const {
        auto &w = weights.weight.at("model.llm." + key);
        if (w.dataType != FLOAT32)
            throw std::runtime_error("Inkling expected FP32 parameter: " + key);
        return w;
    }
    const float *param(const std::string &key, int count) const {
        auto &w = weight(key);
        if (w.Count(0) != count)
            throw std::runtime_error("Inkling tensor shape mismatch: " + key);
        return static_cast<const float *>(w.cudaData);
    }
    static bool q8Projection(const Data &w) {
        return w.dataType == DATA_GGUF_FORMAT && w.ggmlType == GGML_TYPE_Q8_0;
    }
    template<class T, class Out, class Store = StoreProjectionRow>
    bool singleQ8(const T *q, const T *k, const T *v, const Data &weight, Rows<Out> &out,
                  int kBegin, int vBegin, Store store = {}) const {
        if (!decodePdlSupported || !q8Projection(weight) || out.rows != 1 ||
            weight.dims.size() != 2 || weight.dims[0] != out.cols) return false;
        const int width = weight.dims[1];
        if (width != 2304 && width != 3072 && width != 1536) return false;
        const bool shared = q == k && q == v;
        Buffer<DecodeActivationBlock> packed((shared ? 1 : 3) * (width / 32));
        if (!cuda_launch::Launch<packDecodeActivations<T>>(dim3((width + 127) / 128, shared ? 1 : 3), 128,
                0, cudaStreamPerThread, q, k, v, width, packed.p))
            throw std::runtime_error("Inkling activation pack failed");
        auto project = [&](auto columns) {
            return cuda_launch::Launch<projectDecodeQ8<decltype(columns)::value, Out, Store>>((out.cols + 3) / 4, 128, 0,
                cudaStreamPerThread, packed.p, static_cast<const uint16_t *>(weight.cudaData), out.p, out.cols,
                shared ? out.cols : kBegin, shared ? out.cols : vBegin, store);
        };
        const bool ok = dispatchDecodeWidth(width, project);
        if (!ok) throw std::runtime_error("Inkling Q8 decode projection failed");
        return true;
    }
    template<class T, class U> void projectQ8(const Rows<T> &x, const Data &w, int offset, Rows<U> &out, float cap = 0.f) const {
        auto input = castRows<float>(x);
        Matrix tmp;
        float *dst;
        if constexpr (std::is_same_v<U, float>) dst = out.p;
        else { tmp = Matrix(out.rows, out.cols); dst = tmp.p; }
        const size_t bytes = FastllmCudaQ8ProjectionScratchBytes(x.rows, x.cols);
        Buffer<uint8_t> scratch(bytes);
        const auto stride = w.GetBytes() / w.dims[0];
        FastllmCudaProjectQ8_0(input.p, static_cast<const uint8_t *>(w.cudaData) + offset * stride,
                               dst, scratch.p, x.rows, x.cols, out.cols);
        if (cap > 0.f) {
            const size_t n = size_t(out.rows) * out.cols;
            softcap<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(dst, dst, n, cap);
        }
        if constexpr (!std::is_same_v<U, float>) {
            const size_t n = size_t(out.rows) * out.cols;
            castProjection<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(dst, out.p, n);
        }
    }
    template <typename T, typename U> void projectInto(const Rows<T> &x, const std::string &key, Rows<U> &out,
                                            int offset, int groups, Projection mode = Projection::Default, int groupStride = 0) const {
        const int columns = out.cols;
        auto &w = weights.weight.at("model.llm." + key);
        const auto type = std::is_same<T, half>::value ? FLOAT16 : BFLOAT16;
        if (w.dims.size() != 2 || w.dims[1] != x.cols || columns < 1 ||
            offset < 0 || offset + groups * columns > w.dims[0] || x.rows % groups)
            throw std::runtime_error("Inkling packed projection shape or type: " + key);
        const int n = x.rows / groups, m = x.cols;
        if (groups == 1 && q8Projection(w)) {
            if (mode == Projection::PdlDecode && !offset &&
                singleQ8(x.p, x.p, x.p, w, out, columns, columns)) return;
            projectQ8(x, w, offset, out);
            return;
        }
        if constexpr (std::is_same_v<T, U>) {
            if (w.dataType == NVFP4_BLOCK_16 && !offset && groups == 1 && columns == w.dims[0]) {
                CudaView input(type, {n, m}, x.p), result(type, {n, columns}, out.p);
                DoCudaLinear(input, w, *GetEmptyData(), result);
                return;
            }
        }
        if (w.dataType != type) throw std::runtime_error("Unsupported Inkling projection dtype: " + key);
        auto *matrix = static_cast<T *>(w.cudaData) + size_t(offset) * m;
        if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<U, T>) {
            if (mode == Projection::PdlDecode && n == 1 && groups == 1 && decodePdlSupported &&
                launchDecodeBf16(x.p, x.p, x.p, matrix, out.p, m, columns, columns, columns)) return;
            if (mode == Projection::PdlDecode && decodeBf16Gemv && decodeBf16Gemv(x.p, matrix, out.p, n, m, columns))
                return;
            if (mode == Projection::DraftHead && bf16Mma &&
                bf16Mma->draftLogits(x.p, matrix, out.p, n, m, columns))
                return;
            if (mode == Projection::AttentionMma && bf16Mma->attention(x.p, matrix, out.p, n, m, columns))
                return;
            if (mode == Projection::DecodeHead && FastllmCudaRuntimeArch() == 120 &&
                FastllmCudaBf16DecodeProjection(x.p, matrix, out.p, n, m, columns))
                return;
            if (decodeArithmetic && n > 1 && groups == 1 && FastllmCudaRuntimeArch() == 120 && m != 256) {
                if (!FastllmCudaBf16ExactBatchProjection(x.p, matrix, out.p, n, m, columns))
                    LaunchFastllmGemmBf16Bf16(x.p, matrix, out.p, nullptr, n, m, columns);
                return;
            }
        }
        CudaView weightView(type, {groups * columns, m}, matrix);
        CudaView input(type, {groups * n, m}, x.p);
        CudaView result(std::is_same_v<U, float> ? FLOAT32 : type, {groups * n, columns}, out.p);
        if (groups == 1 && std::is_same_v<T, U>)
            DoCudaLinear(input, weightView, *GetEmptyData(), result);
        else {
            const int stride = groupStride ? groupStride : n;
            FastllmCudaBatchMatMulTransB(input, weightView, result, stride * m, columns * m, stride * columns,
                                         m, m, groups, n, m, columns, 1.f);
        }
    }

    // Preserve the single-prompt GEMM shape. Q/K/V controller streams keep
    // their full-batch pitch so each GEMM writes directly into its final rows.
    template <class T, class U> void projectPrompts(const Rows<T> &x, const std::string &key, Rows<U> &out,
            int offset, int groups, Projection mode, int promptRows) const {
        if (!promptRows) {
            projectInto(x, key, out, offset, groups, mode);
            return;
        }
        const int stride = x.rows / groups;
        for (int first = 0; first < stride; first += promptRows) {
            auto dest = out.slice(first, groups * promptRows);
            projectInto(x.slice(first, groups * promptRows), key, dest, offset, groups, mode, stride);
        }
    }

    template <typename T> Rows<T> linear(const Rows<T> &x, const std::string &key, int columns,
                                         int groups = 1, Projection mode = Projection::Default) const {
        auto out = Rows<T>::temporary(x.rows, columns);
        projectInto(x, key, out, 0, groups, mode);
        return out;
    }

    template <typename T, typename Out> Rows<Out> control(const Matrix *dest, const Matrix &messages, const float *const *norms,
                                         int count, const std::string &prefix, bool pdl, int promptRows) const {
        const int n = dest[0].rows, d = dest[0].cols;
        LoopStreams streams{};
        auto out = Rows<Out>::temporary(count * n, d);
        for (int i = 0; i < count; ++i)
            streams.items[i] = {dest[i].p, messages.slice(i * n, n).p, norms[i]};
        const auto path = prefix + (count == 1 ? "controllers.mlp." : "controllers.qkv.");
        auto bf16Weight = [&](const char *name, int rows, int cols) -> const __nv_bfloat16 * {
            const auto &w = weights.weight.at("model.llm." + path + name);
            if (w.dataType != BFLOAT16 || w.dims.size() != 2 || w.dims[0] != rows || w.dims[1] != cols)
                return nullptr;
            return static_cast<const __nv_bfloat16 *>(w.cudaData);
        };
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if (pdl && n == 1 && !decodeArithmetic && c.draft.steps == 0 && d == 2304 && c.rank == 64) {
                const auto *down = bf16Weight("down_weight", count * c.rank, 2 * d);
                const auto *gate = bf16Weight("gates_weight", count * 2 * d, c.rank);
                if (down && gate) {
                    Rows<T> hidden(count, c.rank);
                    Matrix mixed(count, d), squares(count, d / 32);
                    if (!cuda_launch::Launch<projectControllerPair>(dim3(c.rank, count), 256, 0, cudaStreamPerThread,
                            streams, down, hidden.p, c.rank, c.eps) ||
                        !cuda_launch::Launch<blendControllerChannels>(dim3(d / 32, count), 256, 0, cudaStreamPerThread,
                            streams, hidden.p, gate, mixed.p, squares.p) ||
                        !cuda_launch::Launch<normalizeControllerChannels<Out>>(dim3(d / 128, count), 128, 0,
                            cudaStreamPerThread, streams, mixed.p, squares.p, out.p, c.eps))
                        throw std::runtime_error("Inkling single-token controller launch failed");
                    return out;
                }
            }
        }
        auto pair = Rows<T>::temporary(count * n, 2 * d);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if (batchedController && decodeArithmetic && n > 1) {
                const auto *down = bf16Weight("down_weight", count * c.rank, 2 * d);
                const auto *gate = bf16Weight("gates_weight", count * 2 * d, c.rank);
                if (down && gate) {
                    launchBatchedController(streams, pair, out, down, gate, count, c.rank, smCount, c.eps);
                    return out;
                }
            }
        }
        if (pdl) {
            if (!cuda_launch::Launch<ControllerNormRows<true, T, T, 2304, false, true>>(
                    dim3(n, count), 512, 0, cudaStreamPerThread, streams, static_cast<const float *>(nullptr), pair.p, c.eps))
                throw std::runtime_error("Inkling controller pair launch failed");
        } else if (d == 2304)
            ControllerNormRows<true, T, T, 2304><<<dim3(n, count), 512, 0, cudaStreamPerThread>>>(streams, nullptr, pair.p, c.eps);
        else PairRMSNorm<<<count * n * 2, 512, 0, cudaStreamPerThread>>>(streams, pair.p, n, d, c.eps);
        auto hidden = Rows<T>::temporary(count * n, c.rank);
        projectPrompts(pair, path + "down_weight", hidden, 0, count, Projection::Default, promptRows);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // Eight tokens per block need a long prompt to occupy the SMs.
            if (fusedPrefillController && !decodeArithmetic && (promptRows ? promptRows : n) >= 1024) {
                if (const auto *gate = bf16Weight("gates_weight", count * 2 * d, c.rank)) {
                    Buffer<ControllerGateTile> packed(count * 2 * controllerTiles, CurrentPrefillWorkspace());
                    packControllerGates<<<dim3(controllerTiles, count * 2, 4), 128, 0, cudaStreamPerThread>>>(
                        gate, packed.p);
                    const dim3 grid((n + controllerRows - 1) / controllerRows, count);
                    prefillControllerMix<Out><<<grid, controllerThreads, controllerSharedBytes, cudaStreamPerThread>>>(
                        streams, hidden.p, packed.p, out.p, n, c.eps);
                    return out;
                }
            }
        }
        // Controller coefficients feed tanh and the FP32 residual blend directly.
        auto gates = Matrix::temporary(count * n, 2 * d);
        projectPrompts(hidden, path + "gates_weight", gates, 0, count, Projection::Default, promptRows);
        const size_t bytes = size_t(d) * sizeof(float);
        if (pdl) {
            if (!cuda_launch::Launch<ControllerNormRows<false, T, Out, 2304, false, true>>(
                    dim3(n, count), 512, 0, cudaStreamPerThread, streams, gates.p, out.p, c.eps))
                throw std::runtime_error("Inkling controller blend launch failed");
        } else if (d == 2304)
            ControllerNormRows<false, T, Out, 2304><<<dim3(n, count), 512, 0, cudaStreamPerThread>>>(streams, gates.p, out.p, c.eps);
        else BlendRMSNorm<T, Out><<<count * n, 512, bytes <= 16384 ? bytes : 0, cudaStreamPerThread>>>(
            streams, gates.p, out.p, n, d, c.eps);
        return out;
    }

    template<class T> struct AttentionProjection {
        Rows<T> storage;
        std::array<int, 8> widths;
        bool shared;

        template<class F> void groups(F use) const {
            int offset = 0;
            for (int begin = 0; begin < 8;) {
                int end = shared ? 8 : (begin == 0 ? 4 : begin + 2);
                auto part = storage;
                part.p += size_t(offset) * storage.rows;
                part.cols = std::accumulate(widths.begin() + begin, widths.begin() + end, 0);
                use(part, begin, end, offset);
                offset += part.cols;
                begin = end;
            }
        }
        template<class U> std::array<Rows<U>, 8> unpack() const {
            Rows<U> values;
            if (storage.rows == 1) values = castRows<U>(storage);
            else {
                values = Rows<U>::temporary(storage.rows, storage.cols);
                groups([&](const Rows<T> &part, int begin, int end, int offset) {
                    ProjectionParts parts{};
                    std::copy(widths.begin() + begin, widths.begin() + end, parts.widths);
                    const size_t size = size_t(part.rows) * part.cols;
                    unpackProjection<<<(size + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                        part.p, values.p + size_t(offset) * part.rows, parts, part.rows, part.cols);
                });
            }
            std::array<Rows<U>, 8> result;
            for (int i = 0; i < 8; ++i) {
                if (!widths[i]) continue;
                result[i] = values;
                result[i].cols = widths[i];
                values.p += size_t(storage.rows) * widths[i];
            }
            return result;
        }
        PrefillParts<T> views() const {
            PrefillParts<T> result{};
            groups([&](const Rows<T> &part, int begin, int end, int) {
                int col = 0;
                for (int i = begin; i < end; ++i) {
                    result.items[i] = {widths[i] ? part.p + col : nullptr, part.cols};
                    col += widths[i];
                }
            });
            return result;
        }
    };

    bool usesVerifyArithmetic(const std::vector<Segment> &batch) const {
        // A draft-enabled target must agree across singleton decode and multi-row verification.
        return decodeArithmetic || (c.draft.steps > 0 && batch.front().decode);
    }

    Projection attentionProjection(const std::vector<Segment> &batch) const {
        return bf16Mma && usesVerifyArithmetic(batch) ? Projection::AttentionMma : Projection::Default;
    }

    template<class T, class Out, class R> void mixDecodeInto(const std::vector<Rows<R>> &blocks, const Rows<R> &x, const Query *q, int nq,
                                   Out *out, bool normalizeValues) const {
        cuda_attn_res::Queries queries{};
        std::copy_n(q, nq, queries.items);
        cuda_attn_res::SourceList<R> src{};
        int count = 0;
        for (const auto &block : blocks) src.items[count++] = block.p;
        src.items[count++] = x.p;
        cuda_attn_res::WithRegisterSources(count, [&](auto tag) {
            constexpr int N = decltype(tag)::value;
            if (!cuda_launch::Launch<cuda_attn_res::AttnResRegisterRows<N, T, Out, 2304, (N > 4), true, R>>(
                    dim3(x.rows, nq), 512, 0, cudaStreamPerThread, src, queries, x.rows, 2304, c.eps, normalizeValues, out, false))
                throw std::runtime_error("Inkling decode AttnRes launch failed");
        });
    }

    template<class Round, class Out = float, class R>
    Rows<Out> mixLayerInputs(const std::vector<Rows<R>> &blocks, const Rows<R> &last, const Query *queries,
                             int count, bool decodePdl) const {
        if (!decodePdl || blocks.size() >= 8)
            return mix<Round, Out>(blocks, last, queries, count, c.eps, c.normValues);
        auto result = Rows<Out>::temporary(count * last.rows, last.cols);
        mixDecodeInto<Round>(blocks, last, queries, count, result.p, c.normValues);
        return result;
    }

    template<class T> AttentionProjection<T> projectAttention(const std::array<Rows<T>, 3> &inputs,
                                                             const std::string &att, bool local, Projection mode, int rows, int promptRows) const {
        const int q = c.heads * c.headDim, k = c.kvHeads * c.headDim, taps = local ? 0 : c.kvHeads * c.conv;
        AttentionProjection<T> result;
        result.widths = {q, c.heads * c.relative, q, c.heads, k, taps, k, taps};
        // A shared input packs Q, K and V into one matrix. Use the same math when
        // they are separate, including shapes whose individual views need the fallback.
        if (mode == Projection::AttentionMma && rows <= 16 &&
            ((inputs[0].cols | (2 * q + c.heads * (c.relative + 1)) | (k + taps)) & 7))
            mode = Projection::Default;
        result.shared = inputs[0].p == inputs[1].p && inputs[0].p == inputs[2].p;
        result.storage = Rows<T>::temporary(rows, std::accumulate(result.widths.begin(), result.widths.end(), 0));
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if (mode == Projection::AttentionMma && !result.shared) {
                const auto &w = weights.weight.at("model.llm." + att + "packed.weight");
                const void *x[] = {inputs[0].p, inputs[1].p, inputs[2].p};
                const int widths[] = {result.storage.cols - 2 * (k + taps), k + taps, k + taps};
                if (w.dataType == BFLOAT16 && w.dims == std::vector<int>{result.storage.cols, inputs[0].cols} &&
                    bf16Mma->segmentedAttention(x, w.cudaData, result.storage.p, rows, inputs[0].cols, widths))
                    return result;
            }
            if (mode == Projection::PdlDecode && result.storage.rows == 1) {
                const auto &w = weights.weight.at("model.llm." + att + "packed.weight");
                const int width = result.storage.cols, kEnd = width - (k + taps), qEnd = kEnd - (k + taps);
                if (w.dims.size() == 2 && w.dims[1] == inputs[0].cols &&
                    singleQ8(inputs[0].p, inputs[1].p, inputs[2].p, w, result.storage, qEnd, kEnd))
                    return result;
                if (decodePdlSupported && w.dataType == BFLOAT16 && w.dims.size() == 2 && w.dims[0] == width &&
                    w.dims[1] == inputs[0].cols &&
                    launchDecodeBf16(inputs[0].p, inputs[1].p, inputs[2].p, w.cudaData, result.storage.p,
                                     inputs[0].cols, width, qEnd, kEnd)) return result;
                if (w.dataType == BFLOAT16 && w.dims.size() == 2 && w.dims[0] == width &&
                    w.dims[1] == inputs[0].cols && FastllmCudaBf16PdlProjection(inputs[0].p, inputs[1].p,
                        inputs[2].p, w.cudaData, result.storage.p, w.dims[1], qEnd, kEnd, width, 0.f))
                    return result;
            }
            if (mode != Projection::AttentionMma && !result.shared && (inputs[0].rows == 1 || decodeArithmetic)) {
                const auto &w = weights.weight.at("model.llm." + att + "packed.weight");
                const T *x[] = {inputs[0].p, inputs[1].p, inputs[2].p};
                const int widths[] = {result.storage.cols - 2 * (k + taps), k + taps, k + taps};
                if (w.dataType == BFLOAT16 && w.dims.size() == 2 && w.dims[0] == result.storage.cols &&
                    w.dims[1] == inputs[0].cols && LaunchFastllmGroupedBf16Projection(x, static_cast<const T *>(w.cudaData),
                        result.storage.p, result.storage.rows, inputs[0].cols, widths))
                    return result;
            }
        }
        result.groups([&](Rows<T> part, int begin, int, int offset) {
            projectPrompts(inputs[begin == 0 ? 0 : (begin == 4 ? 1 : 2)], att + "packed.weight", part,
                           offset, 1, mode, promptRows);
        });
        return result;
    }

    template<class KV, class T> Rows<T> attend(const std::array<Rows<T>, 3> &inputs, const std::string &att,
                                      bool local, const std::vector<Segment> &batch, size_t slot, int promptRows) {
        const int d = c.headDim, k = c.kvHeads * d;
        const int aligned = attentionWidth(d), kvWidth = c.kvHeads * aligned;
        const int window = local ? c.window : 0;
        const Data &table = weight(att + "rel_logits_proj.proj");
        if (table.Count(0) != c.relative * c.extent)
            throw std::runtime_error("Inkling relative bias table shape");
        const bool singleDecodePdl = decodePdlSupported && batch.size() == 1 && batch.front().decode &&
            !batch.front().timeline && !usesVerifyArithmetic(batch);
        auto projection = projectAttention(inputs, att, local,
            singleDecodePdl ? Projection::PdlDecode : attentionProjection(batch), inputs[0].rows, promptRows);
        if ((batch.front().decode || decodeArithmetic) && aligned == d && c.relative <= threads) {
            const auto projected = projection.views();
            Rows<T> attended(projection.storage.rows, c.heads * d);
            const int queries = batch.front().timeline ? attended.rows : 1;
            const bool usePagedVerify = verifyAttentionSplits > 0 && std::is_same_v<KV, half>;
            // One window query can spread its KV tiles across more SMs. A speculative
            // target keeps the same partitions for both singleton and chain evaluation.
            const int minTiles = window && !usesVerifyArithmetic(batch) ? 1 : VerifyPartition::minTiles;
            const int splits = usePagedVerify ? verifySplitCount(verifyAttentionSplits, window, GetPageLen(), minTiles)
                                              : splitScratch.rows;
            const bool singleWindow = usePagedVerify && window && batch.front().decode && !batch.front().timeline &&
                attended.rows == 1 && !usesVerifyArithmetic(batch);
            const bool splitWindow = singleWindow && singleWindowKernel.handle && std::is_same_v<T, __nv_bfloat16>;
            const bool pipelineAttention = pagedDecodePdl && ((singleWindow && splits <= 32) ||
                (!window && singleDecodePdl && usePagedVerify && attended.rows == 1));
            const int chain = batch.front().length;
            const bool compact = compactWindow && compactWindow->handle && usePagedVerify && window && batch.front().timeline &&
                chain > 1 && std::is_same_v<T, __nv_bfloat16> &&
                batch.size() <= 65535 && int64_t(batch.size()) * c.kvHeads >= smCount &&
                !projected.items[5].data && !projected.items[7].data &&
                !((reinterpret_cast<uintptr_t>(projected.items[2].data) |
                   reinterpret_cast<uintptr_t>(projected.items[3].data)) & 15);
            Matrix q(compact ? 0 : queries, c.heads * d);
            Matrix bias(compact ? 0 : queries * c.heads, c.extent);
            Matrix scratch = queries == 1 || compact ? splitScratch : Matrix(splits, splitScratch.cols * queries);
            auto evaluate = [&](int row, const DecodeVisit *visit) {
                auto sources = projected;
                for (auto &item : sources.items)
                    if (item.data) item.data += size_t(row) * item.stride;
                const auto &sink = sources.items[3], &gate = sources.items[2];
                RelativeInputs<T> in{sources, param(att + "q_l2_clip.scale", c.heads * d),
                                  param(att + "rel_l2_clip.scale", c.heads * c.relative),
                                  param(att + (local ? "k_sconv.weight" : "k_sconv.static_w"), k * c.conv),
                                  param(att + (local ? "v_sconv.weight" : "v_sconv.static_w"), k * c.conv),
                                  static_cast<const float *>(table.cudaData), c.heads, c.kvHeads, d, c.relative,
                                  c.extent, c.conv, c.qClip, c.rClip, c.eps};
                if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<KV, half>) {
                    if (compact) {
                        // The prologue writes BF16 query and key-ordered biases. Each CTA in the
                        // consumer owns its complete window, including sink and output gating.
                        if (launchCompactWindow(in, visit, slot, queries, chain, attended.p)) return;
                        q = Matrix(queries, c.heads * d);
                        bias = Matrix(queries * c.heads, c.extent);
                        scratch = Matrix(splits, splitScratch.cols * queries);
                    }
                }
                const int biasTile = c.relative == 16 && c.extent % 4 == 0 ? 4 * threads : threads;
                int tiles = (c.extent + biasTile - 1) / biasTile;
                if (batch.front().timeline && usePagedVerify && pagedDecodePdl) {
                    auto prepare = [&](auto heads) {
                        constexpr int perTile = decltype(heads)::value;
                        constexpr int headsPerBlock = perTile > 1 ? threads / 32 : 1;
                        const int biasTasks = (c.heads + perTile - 1) / perTile * tiles;
                        const int tasks = (c.heads + headsPerBlock - 1) / headsPerBlock + biasTasks +
                                          (2 * c.kvHeads + headsPerBlock - 1) / headsPerBlock;
                        return cuda_launch::Launch<prepareRelative<KV, true, true, T, true, perTile>>(
                            dim3(tasks, queries), threads, 0, cudaStreamPerThread,
                            visit, in, tiles, q.p, bias.p, batch.front().length);
                    };
                    const bool full = d == 128 && c.relative == 16 && c.extent % 4 == 0 && queries >= smCount;
                    if (!(full ? prepare(std::integral_constant<int, 4>{}) : prepare(std::integral_constant<int, 1>{})))
                        throw std::runtime_error("Inkling verify preparation failed");
                } else if (batch.front().timeline) {
                    auto prepare = usePagedVerify ? prepareRelative<KV, true, true, T> : prepareRelative<KV, true, false, T>;
                    prepare<<<dim3(c.heads * (1 + tiles) + 2 * c.kvHeads, queries), threads, 0, cudaStreamPerThread>>>(
                        visit, in, tiles, q.p, bias.p, batch.front().length);
                } else if (splitWindow) {
                    if (!cuda_launch::Launch<prepareRelative<KV, false, false, T, true>>(
                            c.heads * (1 + tiles) + 2 * c.kvHeads, threads, 0, cudaStreamPerThread,
                            visit, in, tiles, q.p, bias.p, 1))
                        throw std::runtime_error("Inkling split window preparation failed");
                } else if (pipelineAttention) {
                    if (!cuda_launch::Launch<prepareRelative<KV, false, true, T, true>>(
                            c.heads * (1 + tiles) + 2 * c.kvHeads, threads, 0, cudaStreamPerThread,
                            visit, in, tiles, q.p, bias.p, 1))
                        throw std::runtime_error("Inkling paged decode attention preparation failed");
                } else {
                    auto prepare = usePagedVerify ? prepareRelative<KV, false, true, T> : prepareRelative<KV, false, false, T>;
                    prepare<<<c.heads * (1 + tiles) + 2 * c.kvHeads, threads, 0, cudaStreamPerThread>>>(
                        visit, in, tiles, q.p, bias.p, 1);
                }
                AttentionRow args{q.p, nullptr, nullptr, {}, bias.p, nullptr, nullptr,
                                  visit, 0, 0, c.heads, c.kvHeads, d, c.extent,
                                  window, c.logFloor, c.logAlpha, sink.stride, gate.stride};
                if (splitWindow) {
                    auto partial = scratch.slice(0, singleWindowParts());
                    if (FastllmCudaLaunchWindowVerify(singleWindowKernel, partial.rows, c.kvHeads, 1, visit,
                            q.p, bias.p, sink.data, gate.data, partial.p, sizeof(DecodeVisit),
                            offsetof(DecodeVisit, past), 1, sink.stride, gate.stride))
                        launchRelativeMerge<128, false>(args, partial, sink.data, gate.data, attended.p);
                    else
                        launchSplitRelative<128, KV>(args, partial, sink.data, gate.data, attended.p);
                } else if (usePagedVerify) {
                    launchVerifyAttention(args, scratch, sink.data, gate.data, attended.p + size_t(row) * attended.cols,
                                       queries, batch.front().timeline ? batch.front().length : 1, splits, smCount, minTiles, pipelineAttention, batch.front().timeline && pagedDecodePdl);
                } else if (d == 128) launchSplitRelative<128, KV>(args, scratch, sink.data, gate.data,
                                                          attended.p + size_t(row) * attended.cols, sm120GlobalSplits,
                                                          queries, batch.front().timeline ? batch.front().length : 1);
                else launchSplitRelative<64, KV>(args, scratch, sink.data, gate.data, attended.p + size_t(row) * attended.cols, 0, queries);
            };
            if (batch.front().timeline) {
                const auto *visits = batch.front().timeline + slot * attended.rows;
                evaluate(0, visits);
                return attended;
            }
            if (batch.front().decode) {
                evaluate(0, batch.front().decode + slot);
                return attended;
            }
            for (const auto &seg : batch) {
                auto &v = seg.request->visits[slot];
                const int past = seg.request->length;
                // Keep every page visible to the first query. No rejected row can evict its prefix.
                v.prepare(past + seg.length, window ? window + seg.length - 1 : 0);
                if (seg.verify) {
                    for (int i = 0; i < 2 && c.conv > 1; ++i) {
                        auto &history = v.convState[i];
                        auto old = history;
                        history = Matrix(old.rows, old.cols);
                        check(cudaMemcpyAsync(history.p, old.p, size_t(old.rows) * old.cols * sizeof(float),
                                              cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                        const auto raw = projected.items[i ? 6 : 4];
                        seg.verify->history(history, old, projection.storage, raw.data + size_t(seg.offset) * raw.stride,
                                            raw.stride, k, seg.length, c.conv - 1);
                    }
                }
                std::vector<DecodeVisit> host(seg.length);
                Buffer<DecodeVisit> visits(seg.length);
                for (int r = 0; r < seg.length; ++r) {
                    auto &item = host[r];
                    item.pages = v.view();
                    item.past = past + r;
                    for (int i = 0; i < 4; ++i) item.history[i] = v.convState[i].p;
                }
                check(cudaMemcpyAsync(visits.p, host.data(), host.size() * sizeof(DecodeVisit),
                                      cudaMemcpyHostToDevice, cudaStreamPerThread));
                for (int r = 0; r < seg.length; ++r) evaluate(seg.offset + r, visits.p + r);
                if (seg.verify)
                    seg.verify->writes.emplace_back([&v, past, window](int prefix) { v.prepare(past + prefix, window); });
                else
                    v.prepare(past + seg.length, window);
            }
            return attended;
        }
        const int rows = projection.storage.rows;
        const bool queryFits = !std::is_same_v<KV, half> || (c.qClip > 0 && c.qClip <= 65504.f);
        const bool canFuseGate = std::is_same_v<KV, half> && (window ? gatedWindowPrefill : gatedGlobalPrefill);
        if (batch.front().timeline && batch.front().length == 1 && d == 128 && c.relative == 16 &&
            c.extent % 4 == 0 && c.qClip > 0 && c.rClip > 0 && inputs[0].rows <= 65535) {
            const int rows = inputs[0].rows;
            const bool splitBias = sm120GlobalSplits > 0 && rows * c.heads == 768 && c.extent == 1024;
            const int tiles = splitBias ? (c.extent + threads * 4 - 1) / (threads * 4) : 1;
            const auto parts = projection.views();
            const auto &g = parts.items[2], &s = parts.items[3];
            const auto *visits = batch.front().timeline + slot * rows;
            RelativeInputs<T> in{parts, param(att + "q_l2_clip.scale", c.heads * d),
                param(att + "rel_l2_clip.scale", c.heads * c.relative),
                param(att + (local ? "k_sconv.weight" : "k_sconv.static_w"), k * c.conv),
                param(att + (local ? "v_sconv.weight" : "v_sconv.static_w"), k * c.conv),
                static_cast<const float *>(table.cudaData), c.heads, c.kvHeads, d, c.relative,
                c.extent, c.conv, c.qClip, c.rClip, c.eps};
            const bool tiled = std::is_same_v<T, __nv_bfloat16> && std::is_same_v<KV, half> && window &&
                decodeWindow && decodeWindow->handle && useWindowDecodeTile(rows);
            Matrix q(rows, c.heads * d), r(splitBias ? 0 : rows, c.heads * c.relative), bias(rows * c.heads, c.extent);
            Matrix gates(tiled ? rows : 0, c.heads * d), sinks(tiled ? rows : 0, c.heads);
            in.batchGates = gates.p; in.batchSinks = sinks.p; in.splitBias = splitBias;
            const int tasks = ((c.heads + 3) / 4) * (1 + tiles) + (2 * c.kvHeads + 3) / 4;
            prepareRelative<KV, false, false, T, false, 4, false, true><<<dim3(tasks, rows), threads, 0, cudaStreamPerThread>>>(
                visits, in, tiles, q.p, splitBias ? bias.p : r.p, 1);
            if (!splitBias) {
                CudaView relInput(FLOAT32, {rows * c.heads, c.relative}, r.p);
                CudaView relOutput(FLOAT32, {rows * c.heads, c.extent}, bias.p);
                FastllmCudaBatchMatMul(relInput, table, relOutput, 0, 0, 0, c.relative, c.extent,
                                       1, rows * c.heads, c.relative, c.extent, 1.f);
            }
            Rows<T> output(rows, c.heads * d);
            if (tiled && FastllmCudaLaunchWindowVerify(*decodeWindow, 1, c.kvHeads, rows, visits, q.p, bias.p,
                    sinks.p, gates.p, output.p, sizeof(DecodeVisit), offsetof(DecodeVisit, past), 1, c.heads, c.heads * d))
                return output;
            AttentionRow args{q.p, nullptr, nullptr, {}, bias.p, nullptr, nullptr, visits, 0, 0,
                              c.heads, c.kvHeads, d, c.extent, window, c.logFloor, c.logAlpha, s.stride, g.stride};
            launchBatchDecodeAttention<128, KV>(args, rows, splitScratch.rows, s.data, g.data, output.p);
            return output;
        }
        const bool multi = (tensorPrefill || canFuseGate) && queryFits && !std::is_same_v<KV, float> &&
                           rows > 1 && !batch.front().decode &&
                           std::all_of(batch.begin(), batch.end(), [](const Segment &s) { return s.length > 1; });
        const bool compact = multi && canFuseGate;
        // Select the bias precision using the unsharded shape so a prompt
        // follows the same arithmetic when its attention heads are partitioned.
        const bool narrowBias = compact && window && bf16WindowBias && std::is_same_v<T, __nv_bfloat16> &&
                                size_t(rows) * c.heads * placement.world >= size_t(c.extent);
        const int relativeWidth = c.heads * c.relative;
        Buffer<__nv_bfloat16> narrowRelative;
        __nv_bfloat16 *table16 = nullptr;
        Matrix q, r, sink, gates, attended(compact ? 0 : rows, c.heads * aligned);
        Rows<T> gatedOutput(compact ? rows : 0, c.heads * d);
        Rows<KV> typedQuery, typedKey, typedValue;
        auto parts = projection.views();
        if (multi) {
            if (compact) {
                typedQuery.owner = gatedOutput.owner;
                typedQuery.p = reinterpret_cast<KV *>(gatedOutput.p);
                typedQuery.rows = rows;
                typedQuery.cols = gatedOutput.cols;
            } else typedQuery = Rows<KV>(rows, c.heads * (2 * d));
            typedKey = Rows<KV>(rows, k);
            typedValue = Rows<KV>(rows, k);
            if (narrowBias) {
                const size_t inputElements = size_t(rows) * relativeWidth;
                const size_t elements = size_t(c.relative) * c.extent;
                // Reuse the FP32 projection allocation size across layer types
                // to avoid an extra CUDA pool size class.
                narrowRelative = Buffer<__nv_bfloat16>(inputElements + std::max(inputElements, elements));
                table16 = narrowRelative.p + inputElements;
                castProjection<<<(elements + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    static_cast<const float *>(table.cudaData), table16, elements);
            } else {
                r = Matrix(rows, relativeWidth);
            }
            sink = Matrix(rows, c.heads);
            PrefillPreparation prep{param(att + "q_l2_clip.scale", c.heads * d),
                param(att + "rel_l2_clip.scale", c.heads * c.relative),
                {param(att + (local ? "k_sconv.weight" : "k_sconv.static_w"), k * c.conv),
                 param(att + (local ? "v_sconv.weight" : "v_sconv.static_w"), k * c.conv)},
                c.heads, c.kvHeads, d, c.relative, c.conv, c.qClip, c.rClip, c.eps};
            for (const auto &seq : batch) {
                auto &history = seq.request->visits[slot].convState;
                const int count = c.conv - 1, keep = std::max(0, count - seq.length);
                Matrix next[] = {Matrix(count, k), Matrix(count, k)};
                if (seq.verify) for (int i = 0; i < 2; ++i) {
                    const auto raw = parts.items[i ? 6 : 4];
                    seq.verify->history(history[i], history[i], projection.storage, raw.data + size_t(seq.offset)*raw.stride,
                                        raw.stride, k, seq.length, count);
                }
                for (int i = 0; i < 2 && keep; ++i) {
                    const size_t bytes = size_t(keep) * k * sizeof(float);
                    if (history[i].p)
                        check(cudaMemcpyAsync(next[i].p, history[i].p + size_t(seq.length) * k,
                                              bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                    else check(cudaMemsetAsync(next[i].p, 0, bytes, cudaStreamPerThread));
                }
                const int tasks = 2 * (c.heads + c.kvHeads);
                const int groups = (tasks + 3) / 4;
                dim3 grid(seq.length, groups);
                if (compact) {
                    const int rowsY = std::min(seq.length, 65535);
                    grid = dim3(groups, rowsY, 1 + (seq.length - 1) / rowsY);
                }
                auto prepare = [&](auto kernel, auto *relative) {
                    kernel<<<grid, 128, 0, cudaStreamPerThread>>>(
                        parts, prep, seq.offset, seq.length, history[0].p, history[1].p, next[0].p, next[1].p,
                        typedQuery.p, relative, sink.p, typedKey.p, typedValue.p);
                };
                if (narrowBias) prepare(preparePrefill<true, T, KV, __nv_bfloat16>, narrowRelative.p);
                else if (compact) prepare(preparePrefill<true, T, KV>, r.p);
                else prepare(preparePrefill<false, T, KV>, r.p);
                history[0] = std::move(next[0]);
                history[1] = std::move(next[1]);
            }
        } else {
            const auto projected = projection.template unpack<float>();
            projection.storage = {};
            gates = projected[2];
            sink = projected[3];
            q = resizeHeads(norm(projected[0], param(att + "q_l2_clip.scale", c.heads * d),
                                 c.eps, d, c.heads, c.qClip), d, aligned);
            r = norm(projected[1], param(att + "rel_l2_clip.scale", c.heads * c.relative),
                     c.eps, c.relative, c.heads, c.rClip);
            auto convKv = [&](int index, const std::string &name, int which) {
                return convolutions(projected[index], param(att + name + (local ? ".weight" : ".static_w"), k * c.conv),
                                    projected[index + 1], c.kvHeads, c.conv, batch, slot, which);
            };
            auto key = norm(convKv(4, "k_sconv", 0), c.qClip > 0.f ? nullptr : unitScale.p, c.eps, d, 1, c.qClip);
            auto val = norm(convKv(6, "v_sconv", 1), unitScale.p, c.eps, d);
            typedKey = castRows<KV>(resizeHeads(key, d, aligned));
            typedValue = castRows<KV>(resizeHeads(val, d, aligned));
        }
        auto applyGate = [&] {
            attended = resizeHeads(attended, aligned, d);
            CudaView gated(FLOAT32, {attended.rows, attended.cols}, attended.p);
            CudaView gating(FLOAT32, gated.dims, gates.p);
            if (!FastllmCudaSigmoidMulTo(gated, gating)) throw std::runtime_error("Inkling attention gate shape");
            return castRows<T>(attended);
        };
        if (batch.front().timeline && batch.front().length == 1) {
            // Ordinary decode projects an FP32 relative bias indexed by distance.
            const auto *visits = batch.front().timeline + slot * rows;
            Matrix bias(rows * c.heads, c.extent);
            CudaView input(FLOAT32, {rows * c.heads, c.relative}, r.p);
            CudaView projected(FLOAT32, {rows * c.heads, c.extent}, bias.p);
            FastllmCudaBatchMatMul(input, table, projected, 0, 0, 0, c.relative, c.extent,
                                   1, rows * c.heads, c.relative, c.extent, 1.f);
            storeDecodeKv<<<dim3((kvWidth + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
                visits, typedKey.p, typedValue.p, kvWidth);
            if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<KV, half>) {
                if (window && decodeWindow && decodeWindow->handle && useWindowDecodeTile(rows)) {
                    Rows<T> output(rows, c.heads * d);
                    if (FastllmCudaLaunchWindowVerify(*decodeWindow, 1, c.kvHeads, rows, visits, q.p, bias.p,
                            sink.p, gates.p, output.p, sizeof(DecodeVisit), offsetof(DecodeVisit, past),
                            1, c.heads, c.heads * d)) return output;
                }
            }
            AttentionRow args{q.p, nullptr, nullptr, {}, bias.p, sink.p, attended.p, visits, 0, 0,
                              c.heads, c.kvHeads, d, c.extent, window, c.logFloor, c.logAlpha};
            if (aligned == 128) launchBatchDecodeAttention<128, KV>(args, rows, splitScratch.rows, sink.p, static_cast<float *>(nullptr), attended.p);
            else launchBatchDecodeAttention<64, KV>(args, rows, splitScratch.rows, sink.p, static_cast<float *>(nullptr), attended.p);
            return applyGate();
        }
        const auto biasType = narrowBias ? BFLOAT16 : FLOAT32;
        std::optional<CudaView> narrowTable;
        if (narrowBias) narrowTable.emplace(BFLOAT16, table.dims, table16);
        const Data &relativeTable = narrowTable ? *narrowTable : table;
        for (const auto &seq : batch) {
            const DecodeVisit *decode = seq.decode ? seq.decode + slot : nullptr;
            VisitCache *cache = decode ? nullptr : &seq.request->visits[slot];
            int past = decode ? 0 : seq.request->length;
            auto currentKey = typedKey.slice(seq.offset, seq.length),
                 currentValue = typedValue.slice(seq.offset, seq.length);
            const int distances = decode ? c.extent : std::min(c.extent, past + seq.length);
            // Bound temporary relative logits for long prefills. Each row
            // addresses the original complete suffix, including earlier tiles.
            const int baseTile = typedQuery.p ? 512 : 64;
            // Head sharding reduces the number of CTAs per window tile. Batch
            // more queries for the narrow bias GEMM and its attention consumer.
            int tile = narrowBias && placement.world > 1 ? 2048 : baseTile;
            auto allocateBias = [&](int count) {
                const size_t elements = size_t(std::min(seq.length, count)) * c.heads * distances;
                return Buffer<uint8_t>(elements * (narrowBias ? sizeof(__nv_bfloat16) : sizeof(float)),
                                       CurrentPrefillWorkspace());
            };
            Buffer<uint8_t> bias;
            try {
                bias = allocateBias(tile);
            } catch (const CudaWorkspaceExhausted &) {
                if (tile == baseTile || seq.length <= baseTile) throw;
                cudaGetLastError();
                tile = baseTile;
                bias = allocateBias(tile);
            }
            for (int first = 0; first < seq.length; first += tile) {
                int count = std::min(tile, seq.length - first);
                const int rows = count * c.heads;
                const size_t offset = size_t(seq.offset + first) * relativeWidth;
                void *relative = narrowBias ? static_cast<void *>(narrowRelative.p + offset) : r.p + offset;
                CudaView input(biasType, {rows, c.relative}, relative);
                CudaView output(biasType, {rows, distances}, bias.p);
                FastllmCudaBatchMatMul(input, relativeTable, output, 0, 0, 0, c.relative, c.extent,
                                       1, rows, c.relative, distances, 1.f);
                AttentionRow args{multi ? nullptr : q.slice(seq.offset, seq.length).p, currentKey.p, currentValue.p,
                    cache ? cache->view() : PageView{}, narrowBias ? nullptr : reinterpret_cast<float *>(bias.p),
                    sink.slice(seq.offset, seq.length).p, compact ? nullptr : attended.slice(seq.offset, seq.length).p,
                    decode, past, first, c.heads, c.kvHeads, d, distances,
                    window, c.logFloor, c.logAlpha};
                if constexpr (!std::is_same_v<KV, float>) {
                    if (typedQuery.p) {
                        auto *query = typedQuery.slice(seq.offset + first, count).p;
                        if constexpr (std::is_same_v<KV, half>) {
                            if (compact) {
                                const auto gate = parts.items[2];
                                const int row = seq.offset + first;
                                auto launch = [&](auto kernel, auto *relativeBias) {
                                    kernel(args, query, gatedOutput.p + size_t(row) * gatedOutput.cols,
                                           gate.data + size_t(row) * gate.stride, gate.stride, count, relativeBias);
                                };
                                if (narrowBias) {
                                    launch(launchGatedPrefill<true,T,__nv_bfloat16>, reinterpret_cast<__nv_bfloat16 *>(bias.p));
                                } else {
                                    launch(window ? launchGatedPrefill<true,T> : launchGatedPrefill<false,T>, args.relative);
                                }
                                continue;
                            }
                        }
                        if (d == 128) launchPrefill<128, KV>(args, query, count);
                        else launchPrefill<64, KV>(args, query, count);
                        continue;
                    }
                }
                if (aligned == 128) launchRelative<128, KV>(args, count, seq.length == 1, splitScratch);
                else launchRelative<64, KV>(args, count, seq.length == 1, splitScratch);
            }
            if (decode)
                storeDecodeKv<<<(kvWidth + 255) / 256, 256, 0, cudaStreamPerThread>>>(decode, currentKey.p,
                                                                                currentValue.p, kvWidth);
            else {
                auto store = [cache, past, window, currentKey, currentValue, kvWidth](int length) {
                    cache->prepare(past + length, window);
                    int rows = window ? std::min(length, window) : length;
                    appendKv<<<(size_t(rows)*kvWidth+255)/256,256,0,cudaStreamPerThread>>>(
                        cache->view(), currentKey.p, currentValue.p, past, length-rows, kvWidth, rows);
                };
                if (seq.verify) seq.verify->writes.emplace_back(std::move(store));
                else store(seq.length);
            }
        }
        if (compact) return gatedOutput;
        if (multi) {
            Rows<T> out(rows, c.heads * d);
            gatePrefill<<<dim3(rows, (out.cols + 255) / 256), 256, 0, cudaStreamPerThread>>>(
                attended.p, parts.items[2], out.p, out.cols);
            return out;
        }
        return applyGate();
    }

    // The previous pass's sources stay fixed throughout the next core pass.
    // Compute consecutive layers' independent controller messages together.
    template <class R> Matrix crossPassMessages(const std::vector<Rows<R>> &bank, const Rows<R> &previous, int first, int layers, bool pipeline) {
        const int h = c.hidden, count = 2 * (c.end - c.start) * h;
        const int offset = 2 * (first - c.start) * h;
        const float *main = param("middle_cycle_cross_pass_queries", count) + offset;
        const float *kv = param("middle_cycle_cross_pass_kv_queries", count) + offset;
        std::array<Query, cuda_attn_res::Queries::Inline> queries{};
        for (int i = 0; i < layers; ++i) {
            queries[4 * i] = {main};
            queries[4 * i + 1] = {kv};
            queries[4 * i + 2] = {kv + h};
            queries[4 * i + 3] = {main + h};
            main += 2 * h;
            kv += 2 * h;
        }
        if (pipeline && bank.size() < 8) {
            auto out = Matrix::temporary(4 * layers * previous.rows, c.hidden);
            mixDecodeInto<float>(bank, previous, queries.data(), 4 * layers, out.p, true);
            return out;
        }
        return mix(bank, previous, queries.data(), 4 * layers, c.eps, true);
    }

    template<class T, class R> Rows<R> projectAttentionResidual(const Rows<T> &attended, const std::string &key,
            const float *conv, const Rows<R> &skip, const std::vector<Segment> &batch, size_t slot,
            bool overlapProjection, bool decodePdl, int promptRows) const {
        const int h = c.hidden, outWidth = attended.cols;
        const auto &wo = weights.weight.at("model.llm." + key);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if (placement.world == 1 && overlapProjection && c.conv == 4 && q8Projection(wo) &&
                wo.dims.size() == 2 && wo.dims[1] == outWidth) {
                Rows<R> result(1, h);
                StoreAttentionResidual<R> store{batch.front().decode + slot, reinterpret_cast<const float4 *>(conv),
                                                skip.p, h, c.conv};
                if (singleQ8(attended.p, attended.p, attended.p, wo, result, h, h, store)) return result;
            }
        }
        Rows<T> projected;
        if (placement.world > 1 && std::is_same_v<R, float>) {
            Matrix partial(attended.rows, h);
            if (!overlapProjection || !rankOutputGemv ||
                !rankOutputGemv(attended.p, wo.cudaData, partial.p, attended.rows, outWidth, h))
                projectInto(attended, key, partial, 0, 1);
            reduceRankRows(partial);
            projected = castRows<T>(partial);
        } else {
            projected = Rows<T>::temporary(attended.rows, h);
            // For a row-parallel W_o the per-rank K is smaller. Use GEMM once
            // there is enough work along M, writing BF16 directly for the sum.
            const int rows = promptRows ? promptRows : attended.rows;
            for (int first = 0; first < attended.rows; first += rows) {
                auto src = attended.slice(first, rows), dst = projected.slice(first, rows);
                const bool gemm = placement.world > 1 && rows >= 32 && std::is_same_v<T, __nv_bfloat16> &&
                    wo.dataType == BFLOAT16 && FastllmCudaBf16ProjectionGemm(src.p, wo.cudaData, dst.p, rows, outWidth, h, 1);
                if (!gemm) projectInto(src, key, dst, 0, 1,
                                      overlapProjection ? Projection::PdlDecode : attentionProjection(batch));
            }
        }
        if constexpr (std::is_same_v<R, __nv_bfloat16>) {
            if (placement.world > 1) reduceProjection<R>(projected);
        }
        return convolutions(projected, conv, {}, 1, c.conv, batch, slot, 2, skip, decodePdl);
    }

    template <typename T, class R> Rows<R> layer(const Rows<R> &x, std::vector<Rows<R>> &blocks, const Matrix &messages,
                 int physical, const std::vector<Segment> &batch, size_t slot, ExpertPhase phase, int promptRows) {
        const std::string pre = "layers." + std::to_string(physical) + ".", att = pre + "attn.";
        bool local = c.local[physical];
        int h = c.hidden;
        const char *fullNorms[] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
        const int nq = local ? 1 : 3;
        const float *aq = physical ? param("attn_res.attn_queries." + std::to_string(physical - 1), nq * h) : nullptr;
        Query queries[3]{};
        const float *norms[3]{};
        for (int i = 0; i < nq; ++i) {
            norms[i] = param(pre + (local ? "attn_norm.weight" : fullNorms[i]), h);
            queries[i] = {aq ? aq + i * h : nullptr, messages.p ? nullptr : norms[i]};
        }
        const bool singleDecode = x.rows == 1 && batch.front().decode && !batch.front().timeline && !usesVerifyArithmetic(batch);
        const bool bf16Decode = singleDecode && std::is_same_v<T, __nv_bfloat16>;
        const bool verifyWindow = batch.front().timeline && phase == ExpertPhase::Verify && x.rows <= 65535 && decodePdlSupported;
        const bool decodeMix = (bf16Decode || (verifyWindow && std::is_same_v<T, __nv_bfloat16>)) && h == 2304 && blocks.size() < 8;
        const bool decodePdl = decodeMix && decodePdlSupported;
        std::array<Rows<T>, 3> inputs;
        if (messages.p) {
            Matrix dest = aq ? mixLayerInputs<float>(blocks, x, queries, nq, decodePdl) : castRows<float>(x);
            Matrix streams[3];
            for (int i = 0; i < 3; ++i) {
                const int stream = local ? 0 : i;
                streams[i] = dest.slice(aq ? stream * x.rows : 0, x.rows);
                norms[i] = norms[stream];
            }
            auto blended = control<T, T>(streams, messages, norms, 3, pre, decodePdl, promptRows);
            for (int i = 0; i < 3; ++i) inputs[i] = blended.slice(i * x.rows, x.rows);
        } else if (aq) {
            auto dest = mixLayerInputs<T, T>(blocks, x, queries, nq, decodePdl);
            for (int i = 0; i < 3; ++i) inputs[i] = dest.slice((local ? 0 : i) * x.rows, x.rows);
        } else {
            for (int i = 0; i < nq; ++i) {
                inputs[i] = castRows<T>(norm(castRows<float>(x), norms[i], c.eps));
            }
            if (local) inputs[1] = inputs[2] = inputs[0];
        }
        Rows<T> attended;
        if (kvType == FLOAT16) attended = attend<half, T>(inputs, att, local, batch, slot, promptRows);
        else if (kvType == BFLOAT16) attended = attend<__nv_bfloat16, T>(inputs, att, local, batch, slot, promptRows);
        else attended = attend<float, T>(inputs, att, local, batch, slot, promptRows);
        const bool newBlock = physical % c.residualBlock == 0;
        const bool overlapProjection = decodeBf16Gemv && kvType == FLOAT16 && singleDecode;
        const float *attnConv = param(pre + "attn_sconv.weight", h * c.conv);
        if (newBlock)
            blocks.push_back(x);
        const float *mlpNorm = param(pre + "mlp_norm.weight", h);
        Query mlpQuery{param("attn_res.mlp_queries." + std::to_string(physical), h), messages.p ? nullptr : mlpNorm};
        auto residual = projectAttentionResidual(attended, att + "wo_ud.weight", attnConv, newBlock ? Rows<R>{} : x,
                                                 batch, slot, overlapProjection, decodePdl, promptRows);
        auto ffn = messages.p ? mixLayerInputs<float>(blocks, residual, &mlpQuery, 1, decodePdl)
                              : mixLayerInputs<T>(blocks, residual, &mlpQuery, 1, decodePdl);
        if (messages.p)
            ffn = control<T, float>(&ffn, messages.slice(3 * x.rows, x.rows), &mlpNorm, 1, pre, decodePdl, promptRows);
        Rows<__nv_bfloat16> rankSum, routes;
        auto m = experts.layers[physical]->run(c, ffn, param(pre + "mlp.gate.bias", c.experts), phase,
                                               placement.world > 1 && std::is_same_v<R, __nv_bfloat16> ? &rankSum : nullptr,
                                               placement.world == 1 && singleDecode && decodePdl && c.conv == 4 ? &routes : nullptr, promptRows);
        const float *expertConv = param(pre + "mlp_sconv.weight", h * c.conv);
        if (routes.p) {
            if (!cuda_launch::Launch<convolveExpertRows<R>>((h + 127) / 128, 128, 0, cudaStreamPerThread,
                    routes.p, routes.rows, batch.front().decode + slot, reinterpret_cast<const float4 *>(expertConv),
                    h, residual.p))
                throw std::runtime_error("Inkling expert residual launch failed");
            return residual;
        }
        if (placement.world > 1) {
            Rows<R> reduced;
            if constexpr (std::is_same_v<R, __nv_bfloat16>) reduced = std::move(rankSum);
            if (!reduced.p) reduced = castRows<R>(m);
            reduceRankRows(reduced);
            return convolutions(reduced, expertConv, {}, 1, c.conv, batch, slot, 3, residual, decodePdl);
        }
        return convolutions(m, expertConv, {}, 1, c.conv, batch, slot, 3, residual, decodePdl);
    }
    template<class R> void reduceRankRows(Rows<R> &x) const {
        static_assert(std::is_same_v<R, float> || std::is_same_v<R, __nv_bfloat16>);
        constexpr auto type = std::is_same_v<R, float> ? FLOAT32 : BFLOAT16;
        const int count = x.rows * x.cols;
#ifndef USE_ROCM
        // Only graph scratch has stable addresses across warmup and capture.
        // Pass eager operands directly instead of registering each temporary.
        const bool eager = DecodeGraphScratch::active == nullptr;
        TransientAllReduceInputs transient(eager);
        // A large in-place two-rank reduction copies its scratch result back.
        // The prompt consumer can read the out-of-place result directly.
        if (eager && !decodeArithmetic && placement.world == 2 && x.rows >= 512 &&
            FastllmCudaCustomAllReduceCanRun(count, type, placement.gpu)) {
            auto result = Rows<R>::temporary(x.rows, x.cols);
            FastllmNcclAllReduce(x.p, result.p, count, type, placement.gpu);
            x = std::move(result);
            return;
        }
#endif
        FastllmNcclAllReduce(x.p, x.p, count, type, placement.gpu);
    }

    template<class R, class T> void reduceProjection(Rows<T> &output) const {
        auto reduced = castRows<R>(output);
        reduceRankRows(reduced);
        output = castRows<T>(reduced);
    }

    Matrix tokenEmbedding(float *ids, int n) {
        const int h = c.hidden;
        auto &embed = weight("embed.weight");
        if (placement.world == 1) {
            auto embedding = Matrix::temporary(n, h);
            CudaView tokens(FLOAT32, {n}, ids), result(FLOAT32, {n, h}, embedding.p);
            FastllmCudaEmbeddingDirect(tokens, embed, result);
            return embedding;
        }
        auto gather = [&](auto activation) {
            auto embedding = Rows<decltype(activation)>::temporary(n, h);
            rankEmbedding<<<n, 256, 0, cudaStreamPerThread>>>(ids, static_cast<const float *>(embed.cudaData),
                embedding.p, placement.rank * placement.vocabRows, placement.vocabRows, h);
            reduceRankRows(embedding);
            return castRows<float>(embedding);
        };
        return residualType == BFLOAT16 ? gather(__nv_bfloat16{}) : gather(float{});
    }

    void target(float *ids, int n, const std::vector<Segment> &batch, Matrix &logits,
                std::vector<Matrix> *features = nullptr, const std::vector<ImageSpan> &images = {},
                const float *boundedTemperature = nullptr, int *tokenMap = nullptr) {
        // Only eager prompts borrow payload storage. A leased feature may outlive
        // target() until storeFeatures consumes it, still under the decoder mutex.
        std::optional<PrefillWorkspaceScope> workspace;
        if (experts.cache && n >= 256 && !DecodeGraphScratch::active &&
            std::all_of(batch.begin(), batch.end(), [](const Segment &s) {
                return !s.timeline && !s.verify && !s.decode && s.request && s.request->length < s.request->promptLength;
            })) {
            // Size for saved residuals/draft taps, live layer projections, and a relative attention tile.
            const size_t hiddenBytes = size_t(n) * c.hidden * sizeof(float);
            const size_t residualBytes = residualType == BFLOAT16 ? hiddenBytes / 2 : hiddenBytes;
            const size_t savedBytes = c.visits.size() / c.residualBlock * residualBytes;
            const size_t liveBytes = (12 + (features ? c.draft.taps.size() : 0)) * hiddenBytes;
            const size_t biasBytes = size_t(std::min(n, 512)) * c.heads * c.extent * sizeof(float);
            auto arena = experts.cache->borrowPrefillWorkspace(savedBytes + liveBytes + biasBytes, c.topK);
            if (arena) workspace.emplace(std::move(arena));
        }
        Matrix local;
        if (placement.world > 1 && logits.p) local = Matrix(logits.rows, placement.vocabRows);
        auto &head = local.p ? local : logits;
        const auto run = [&](auto projection) {
            using T = decltype(projection);
            if (residualType == BFLOAT16)
                targetTyped<T, __nv_bfloat16>(ids, n, batch, head, features, images);
            else
                targetTyped<T, float>(ids, n, batch, head, features, images);
        };
        if (!backboneBf16) run(half{});
        else run(__nv_bfloat16{});
        if (local.p && boundedTemperature) {
            gatherBoundedLogits(local, logits, placement, boundedTemperature, c.vocab, tokenMap);
        } else if (local.p) {
            const int count = local.rows * local.cols;
            Buffer<float> full(size_t(count) * placement.world);
            if (!FastllmNcclAllGather(local.p, full.p, count, FLOAT32, placement.gpu))
                throw std::runtime_error("Inkling TP vocabulary all-gather failed");
            if (local.rows == 1)
                check(cudaMemcpyAsync(logits.p, full.p, c.vocab * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            else unpackVocabRanks<<<(size_t(local.rows) * c.vocab + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                full.p, logits.p, local.rows, c.vocab, local.cols);
        }
    }

    template<class T> void projectLogits(const Matrix &last, Matrix logits, bool verifyMath, bool timeline) const {
        const auto &unembed = weights.weight.at("model.llm.unembed.weight");
        if (q8Projection(unembed)) {
            projectQ8(last, unembed, 0, logits, c.logitCap);
            return;
        }
        auto headInput = castRows<T>(last);
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if ((verifyMath || timeline) && bf16Mma) {
                if (unembed.dataType == BFLOAT16 && unembed.dims.size() == 2 && unembed.dims[0] >= logits.cols &&
                    unembed.dims[1] == c.hidden &&
                    bf16Mma->cappedLogits(headInput.p, unembed.cudaData, logits.p, last.rows, c.hidden, logits.cols, c.logitCap))
                    return;
            }
        }
        Rows<T> projected(last.rows, logits.cols);
        projectInto(headInput, "unembed.weight", projected, 0, 1,
                    verifyMath ? Projection::DecodeHead : Projection::Default);
        size_t nLogits = size_t(logits.rows) * logits.cols;
        softcap<<<(nLogits + 255) / 256, 256, 0, cudaStreamPerThread>>>(projected.p, logits.p, nLogits, c.logitCap);
    }

    template <typename T, typename R> void targetTyped(float *ids, int n, const std::vector<Segment> &batch, Matrix &logits,
                                                       std::vector<Matrix> *features, const std::vector<ImageSpan> &images) {
        // Incremental verification must use the same arithmetic as ordinary one-token decode.
        // Fresh prompt chunks retain their tensor-core prefill kernels.
        struct Restore { bool &mode; ~Restore() { mode = false; } } restore{decodeArithmetic};
        decodeArithmetic = features && (batch.front().timeline || std::all_of(batch.begin(), batch.end(), [&](const Segment &s) {
            return s.request && s.request->draft && s.request->length >= s.request->promptLength &&
                   s.length <= c.draft.block;
        }));
        ExpertPhase phase = ExpertPhase::Prefill;
        if (decodeArithmetic) phase = ExpertPhase::Verify;
        else if (batch.front().timeline || batch.front().decode || std::all_of(batch.begin(), batch.end(), [](const Segment &s) {
            return s.length == 1 && s.request && s.request->length >= s.request->promptLength;
        })) phase = ExpertPhase::Decode;
        const int promptRows = phase == ExpertPhase::Prefill && batch.size() > 1 && !promptSeeds.empty()
                             ? batch.front().length : 0;
        auto *cacheCycle = phase == ExpertPhase::Prefill || DecodeGraphScratch::active ? nullptr : experts.cache.get();
        if (cacheCycle) cacheCycle->beginStep();
        int h = c.hidden;
        auto embedding = tokenEmbedding(ids, n);
        auto normalized = norm(embedding, param("embed_norm.weight", h), c.eps);
        for (const auto &image : images) {
            check(cudaMemcpyAsync(normalized.p + size_t(image.start) * h, image.features.p,
                                  size_t(image.features.rows) * h * sizeof(float),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
        auto x = castRows<R>(normalized);
        normalized = {};
        embedding = {};
        Rows<R> prelude, previous;
        Matrix messageTile;
        int firstMessage = 0, endMessage = 0;
        const bool pipeline = ((n == 1 && batch.front().decode && !batch.front().timeline && !usesVerifyArithmetic(batch)) ||
                               (batch.front().timeline && decodeArithmetic)) &&
            c.hidden == 2304 && decodePdlSupported;
        std::vector<Rows<R>> blocks, preludeBlocks, bank;
        for (size_t slot = 0; slot < c.visits.size(); ++slot) {
            int i = c.visits[slot];
            if (i == c.start) {
                if (!prelude.p) {
                    prelude = x;
                    preludeBlocks = blocks;
                } else {
                    bank.assign(blocks.begin() + preludeBlocks.size() + 1, blocks.end());
                    previous = x;
                    endMessage = c.start;
                    blocks = preludeBlocks;
                    x = prelude;
                }
            }
            if (i == c.end) {
                previous = {};
                messageTile = {};
                bank.clear();
            }
            Matrix messages;
            if (previous.p) {
                if (i == endMessage) {
                    // Bound temporary storage by the number of query rows in a tile.
                    // A large prefill naturally processes just one layer at a time.
                    const int layers = std::min(c.end - i, std::max(1, cuda_attn_res::Queries::Inline / 4 / n));
                    firstMessage = i;
                    endMessage = i + layers;
                    messageTile = {};
                    messageTile = crossPassMessages(bank, previous, i, layers, pipeline);
                }
                messages = messageTile.slice(4 * (i - firstMessage) * n, 4 * n);
            }
            if (features && std::binary_search(c.draft.taps.begin(), c.draft.taps.end(), int(slot)))
                features->push_back(castRows<float>(x));
            x = layer<T>(x, blocks, messages, i, batch, slot, phase, promptRows);
        }
        if (cacheCycle) cacheCycle->endStep();
        if (!features && !logits.p) return;
        // Ordinary single-request prefill only consumes the final row in the vocabulary head.
        if (!features && n > 1 && batch.size() == 1 && !batch.front().verify && !batch.front().timeline) {
            x = x.slice(n - 1, 1);
            for (auto &block : blocks) block = block.slice(n - 1, 1);
        }
        Query outputQuery{param("attn_res.output_query", h), param("norm.weight", h)};
        if (promptRows) {
            std::vector<Rows<R>> tails(blocks.size());
            for (size_t b = 0; b < batch.size(); ++b) {
                const int last = batch[b].offset + batch[b].length - 1;
                for (size_t i = 0; i < tails.size(); ++i) tails[i] = blocks[i].slice(last, 1);
                auto input = mixLayerInputs<float>(tails, x.slice(last, 1), &outputQuery, 1, false);
                projectLogits<T>(input, logits.slice(b, 1), false, false);
            }
            return;
        }
        const auto &unembed = weights.weight.at("model.llm.unembed.weight");
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            if (pipeline && !features && logits.p && blocks.size() < 8 && unembed.dataType == BFLOAT16 &&
                c.logitCap > 0.f && unembed.dims.size() == 2 && unembed.dims[1] == h && unembed.dims[0] >= logits.cols &&
                !(logits.cols & 1)) {
                Rows<T> input(1, h);
                mixDecodeInto<float>(blocks, x, &outputQuery, 1, input.p, c.normValues);
                if (FastllmCudaBf16PdlProjection(input.p, input.p, input.p, unembed.cudaData, logits.p,
                        h, logits.cols, logits.cols, logits.cols, c.logitCap)) return;
            }
        }
        auto output = mixLayerInputs<float>(blocks, x, &outputQuery, 1, pipeline && blocks.size() < 8);
        if (features) features->push_back(output);
        if (!logits.p) return;
        Matrix last;
        if (batch.front().verify || batch.front().timeline) last = output;
        else if (batch.size() == 1)
            last = output.slice(output.rows - 1, 1);
        else {
            last = Matrix(batch.size(), h);
            for (size_t b = 0; b < batch.size(); ++b) {
                auto row = output.slice(batch[b].offset + batch[b].length - 1, 1);
                check(cudaMemcpyAsync(last.p + b * h, row.p, h * sizeof(float), cudaMemcpyDeviceToDevice,
                                      cudaStreamPerThread));
            }
        }
        projectLogits<T>(last, logits, usesVerifyArithmetic(batch), batch.front().timeline != nullptr);
    }

    #include "dspark-forward.cuh"
    #include "draft-graph.cuh"
    #include "verify-graph.cuh"
    #include "decode-batch.cuh"

    bool sampleDecode(const Matrix &logits) {
        auto *in = graphInput.p;
        Buffer<float> probabilities(c.vocab);
        size_t bytes = FastllmCudaTopKFirstSamplingWorkspaceBytes(1);
        Buffer<uint8_t> scratch(bytes);
        return FastllmCudaTopKFirstSamplingToDeviceWithRng(
                logits.p, probabilities.p, &in->temperature, &in->topK, &in->topP, nullptr, nullptr,
                0, graphSample.p, &in->token, 1, c.vocab, &in->seed, &in->offset, scratch.p, bytes);
    }

    void launchTargetGraph(cudaGraphExec_t graph, DecodeGraphScratch &scratch) {
        if (experts.cache) experts.cache->beginStep();
        check(cudaGraphLaunch(scratch.hybrid ? scratch.hybrid->select(graph) : graph, cudaStreamPerThread));
        scratch.finishHostTasks();
        if (scratch.hybrid) scratch.hybrid->observe();
        if (experts.cache) experts.cache->endStep();
    }

    void decodeBody() {
        target(&graphInput.p->token, 1, {{nullptr, 0, 1, graphVisits.p}}, graphLogits);
        if (!sampleDecode(graphLogits)) throw std::runtime_error("Inkling decode sampling failed");
    }

    void prepareDecodeScratch(DecodeGraphScratch &pool) {
        // Exercise the exact single-row body at position zero. The private KV
        // row and convolution state are discarded before serving any request.
        DecodePadding padding(1, c);
        std::fill(hostVisits.begin(), hostVisits.end(), padding.visits.front());
        hostInput = {};
        check(cudaMemcpyAsync(graphInput.p, &hostInput, sizeof(hostInput), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaMemcpyAsync(graphVisits.p, hostVisits.data(), hostVisits.size() * sizeof(DecodeVisit),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
        pool.Warmup([&] { decodeBody(); });
    }
    void captureDecode() {
        graphLogits = Matrix(1, c.vocab);
        DecodeGraphScratch scratchPool;
        prepareDecodeScratch(scratchPool);
        size_t nodes = 0;
        scratchPool.Capture(decodeGraph, [&] { decodeBody(); }, &nodes);
        graphScratch = std::move(scratchPool);
        printf("[Inkling] captured full single-request decode graph (%zu nodes, %zu visits, %zu scratch bytes)\n",
               nodes, c.visits.size(), graphScratch.Bytes());
    }
    void initializeKV() {
        if (kvType != DATA_AUTO_NONE) {
            if (kvType != model.kvCacheDataType)
                throw std::runtime_error("Inkling KV dtype changed after the cache was initialized");
            if (residualType != model.dataType)
                throw std::runtime_error("Inkling residual dtype changed after the cache was initialized");
            return;
        }
        kvType = model.kvCacheDataType;
        residualType = model.dataType;
        if (!c.draft.steps && backboneBf16 && kvType == FLOAT16 && verifyAttentionSplits > 0 &&
            splitScratch.rows >= 16 && c.window <= 1024) {
            singleWindowKernel = PrepareInklingWindowKernel(c.heads, c.heads / c.kvHeads, c.window,
                                                            c.extent, GetPageLen(), true, singleWindowParts());
        }
        if (GetFastllmEnv().cudaGraph) graphSample = Buffer<int>(1);
        if (model.SupportsDecodePipeline()) {
            try {
                captureDecode();
            } catch (const std::exception &e) {
                fprintf(stderr, "[Inkling] CUDA Graph unavailable, using eager decode: %s\n", e.what());
            }
        }
    }
    DecodeVisit decodeVisit(Request &req, size_t slot) {
        auto &v = req.visits[slot];
        v.prepare(req.length + 1, c.local[c.visits[slot]] ? c.window : 0);
        return {v.view(), {v.convState[0].p, v.convState[1].p, v.convState[2].p, v.convState[3].p}, req.length, 1};
    }
    DecodeInput decodeInput(const Request &req, float token, const GenerationConfig &sampling, bool sampled) const {
        return {token, sampled ? std::min(sampling.top_k, c.vocab) : 1,
                sampled ? sampling.temperature : 1.f, sampled ? sampling.top_p : 1.f,
                req.seed, uint64_t(req.length)};
    }

    void stageDecode(Request &req, int token, const GenerationConfig &sampling, bool sampled,
                     DecodeInput &input, DecodeVisit *visits, bool feedback) {
        for (size_t slot = 0; slot < c.visits.size(); ++slot) visits[slot] = decodeVisit(req, slot);
        input = decodeInput(req, token, sampling, sampled);
        // The previous sampler wrote the next input token into graphInput.
        // Keep the ordinary path's exact per-position Philox offset.
        if (feedback)
            check(cudaMemcpyAsync(&graphInput.p->offset, &input.offset, sizeof(input.offset),
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
        else
            check(cudaMemcpyAsync(graphInput.p, &input, sizeof(input), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaMemcpyAsync(graphVisits.p, visits, c.visits.size() * sizeof(DecodeVisit),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
    Matrix runTarget(const float *tokens, int count, const std::vector<Segment> &batch,
                     const GenerationConfig &sampling, bool &sampledOnDevice, std::vector<Matrix> *features = nullptr, bool needLogits = true) {
        if (needLogits && !features && batch.size() == 1 && count == 1 && batch[0].request->length > 0 && batch[0].request->images.empty()) {
            auto &req = *batch[0].request;
            sampledOnDevice = graphSample.p && ordinarySampling(sampling) && !sampling.IsSimpleGreedy();
            stageDecode(req, tokens[0], sampling, sampledOnDevice, hostInput, hostVisits.data(), false);
            if (!decodeGraph) {
                Matrix logits(1, c.vocab);
                target(&graphInput.p->token, 1, {{&req, 0, 1, graphVisits.p}}, logits);
                if (sampledOnDevice) sampledOnDevice = sampleDecode(logits);
                return logits;
            }
            launchTargetGraph(decodeGraph, graphScratch);
            return graphLogits;
        }
        // Image offsets are absolute within a request. Only pass the portion of
        // each image covered by this prefill chunk to the language backbone.
        std::vector<ImageSpan> images;
        for (const auto &seg : batch) {
            const int past = seg.request->length;
            for (const auto &span : seg.request->images) {
                const int first = std::max(past, span.start);
                const int end = std::min(past + seg.length, span.start + span.features.rows);
                if (first >= end) continue;
                const int offset = seg.offset + (first - past), rows = end - first;
                if (!std::all_of(tokens + offset, tokens + offset + rows,
                                 [](float id) { return id == imagePatchToken; }))
                    throw std::runtime_error("Inkling image span does not match the tokenized prompt");
                images.push_back({offset, span.features.slice(first - span.start, rows)});
            }
        }
        Buffer<float> ids(count);
        check(cudaMemcpyAsync(ids.p, tokens, size_t(count) * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        Matrix logits(needLogits ? batch.size() : 0, c.vocab);
        target(ids.p, count, batch, logits, features, images);
        return logits;
    }

    Data &localCache(Data &root) const {
        return placement.world == 1 ? root : *root.multiDeviceDatas.at(placement.gpu);
    }
    class Pipeline final : public basellm::DecodePipeline {
        Decoder &decoder;
        ResponseContext &context;
        std::unique_lock<std::mutex> lock;
        bool pending[2]{}, incomplete = false;
      public:
        Pipeline(Decoder &d, ResponseContext &ctx)
            : decoder(d), context(ctx), lock(d.mutex, std::defer_lock) {}
        ~Pipeline() override {
            // A caller may abandon a chain without collecting its tickets.
            if (incomplete || pending[0] || pending[1]) cudaStreamSynchronize(cudaStreamPerThread);
        }
        bool Submit(int out, int in) override {
            if (!lock.owns_lock()) {
                lock.lock();
                check(cudaSetDevice(decoder.placement.gpu));
                decoder.finishPublication();
            }
            if (!decoder.decodeGraph) return false;
            auto &cache = context.pastKeyValues;
            auto *req = static_cast<Request *>(decoder.localCache(cache.front().first).kvStorageOwner.get());
            if (!req || req->visits.size() != cache.size()) return false;
            if (in < 0) {
                if (req->length != context.intParams.at("promptLen") + context.intParams.at("index"))
                    return false;
                for (size_t i = 0; i < cache.size(); ++i) {
                    req->visits[i].key = &decoder.localCache(cache[i].first);
                    req->visits[i].value = &decoder.localCache(cache[i].second);
                }
            } else {
                // A page append or sliding-window rotation may reclaim storage
                // still read by the previous replay. Rebind at a drained boundary.
                const int len = decoder.localCache(cache.front().first).pageLen, begin = req->length + 1 - decoder.c.window;
                if (req->length % len == 0 || (begin > 0 && begin % len == 0)) return false;
            }
            auto &t = decoder.tickets[out];
            t.init(cache.size());
            incomplete = true;
            decoder.stageDecode(*req, context.currentTokens.front(), context.generationConfig, true,
                                t.host->input, t.visits(), in >= 0);
            decoder.launchTargetGraph(decoder.decodeGraph, decoder.graphScratch);
            check(cudaMemcpyAsync(&t.host->token, decoder.graphSample.p, sizeof(int),
                                  cudaMemcpyDeviceToHost, cudaStreamPerThread));
            check(cudaEventRecord(t.ready, cudaStreamPerThread));
            pending[out] = true;
            incomplete = false;
            ++req->length;
            return true;
        }
        std::vector<int> Collect(int slot) override {
            auto &t = decoder.tickets[slot];
            check(cudaEventSynchronize(t.ready));
            pending[slot] = false;
            auto &req = *static_cast<Request *>(decoder.localCache(context.pastKeyValues.front().first).kvStorageOwner.get());
            const size_t committed = req.length - int(pending[1 - slot]);
            if (req.prefixId && req.prefixTokens.size() + 1 == committed && context.allTokens.size() == committed) {
                req.prefixTokens.push_back(context.allTokens.back());
                if (!pending[1 - slot]) decoder.checkpointDecode(req);
            }
            return {t.host->token};
        }
        void Drain() override {
            if (incomplete) check(cudaStreamSynchronize(cudaStreamPerThread));
            incomplete = false;
            for (int i = 0; i < 2; ++i) {
                if (pending[i]) check(cudaEventSynchronize(decoder.tickets[i].ready));
                pending[i] = false;
            }
        }
    };

    class BatchPipeline final : public basellm::DecodePipeline {
        Decoder &decoder;
        std::vector<ResponseContext *> contexts;
        std::vector<Request *> requests;
        std::unique_lock<std::mutex> lock;
        BatchDecodePlan *plan = nullptr;
        bool pending[2]{}, submitted = false;
      public:
        BatchPipeline(Decoder &d, const std::vector<ResponseContext *> &c)
            : decoder(d), contexts(c), lock(d.mutex, std::defer_lock) {}
        ~BatchPipeline() override {
            if (submitted || pending[0] || pending[1]) cudaStreamSynchronize(cudaStreamPerThread);
        }
        bool Submit(int out, int in) override {
            auto &d = decoder;
            if (in < 0) {
                lock.lock();
                check(cudaSetDevice(d.placement.gpu));
                d.finishPublication();
                for (auto *ctx : contexts) {
                    auto &cache = ctx->pastKeyValues;
                    auto *req = static_cast<Request *>(d.localCache(cache.front().first).kvStorageOwner.get());
                    if (!req || req->visits.size() != cache.size() ||
                        req->length != ctx->intParams.at("promptLen") + ctx->intParams.at("index")) return false;
                    requests.push_back(req);
                }
                const bool bounded = d.placement.world > 1 && d.placement.world <= 8 &&
                    d.placement.world * shardCandidates < d.c.vocab && d.decodeCapacity(contexts.size()) <= 65535 &&
                    std::all_of(contexts.begin(), contexts.end(), [](const ResponseContext *ctx) {
                        return ctx->generationConfig.top_k <= shardCandidates && std::isfinite(ctx->generationConfig.temperature);
                    });
                plan = &d.batchDecodePlan(d.decodeCapacity(contexts.size()), bounded);
                if (!plan->exec) return false;
                for (int i = 0; i < plan->rows; ++i) {
                    const bool live = i < int(requests.size());
                    plan->hostInputs[i] = live ? d.decodeInput(*requests[i], contexts[i]->currentTokens.front(),
                            contexts[i]->generationConfig, true) : DecodeInput{};
                    for (size_t j = 0; j < d.c.visits.size(); ++j) {
                        if (live) {
                            auto &v = requests[i]->visits[j];
                            auto &kv = contexts[i]->pastKeyValues[j];
                            v.key = &d.localCache(kv.first); v.value = &d.localCache(kv.second);
                        }
                        plan->hostVisits[j * plan->rows + i] = live ? d.decodeVisit(*requests[i], j) : plan->padding.visits[i];
                    }
                }
                plan->upload();
            } else {
                for (auto *req : requests) {
                    const int len = req->visits.front().key->pageLen, begin = req->length + 1 - d.c.window;
                    if (req->length % len == 0 || (begin > 0 && begin % len == 0)) return false;
                }
                const int count = plan->rows * d.c.visits.size();
                advanceBatchDecode<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    plan->inputs.p, plan->visits.p, plan->sampled.p, plan->rows, requests.size(), d.c.visits.size());
                for (auto *req : requests)
                    for (size_t j = 0; j < d.c.visits.size(); ++j)
                        req->visits[j].prepare(req->length + 1, d.c.local[d.c.visits[j]] ? d.c.window : 0);
            }
            submitted = true;
            d.launchTargetGraph(plan->exec, plan->scratch);
            auto &ticket = plan->tickets[out];
            ticket.init(plan->rows);
            check(cudaMemcpyAsync(ticket.tokens.data(), plan->sampled.p, requests.size() * sizeof(int),
                                  cudaMemcpyDeviceToHost, cudaStreamPerThread));
            check(cudaEventRecord(ticket.ready, cudaStreamPerThread));
            for (auto *req : requests) ++req->length;
            pending[out] = true;
            submitted = false;
            return true;
        }
        std::vector<int> Collect(int slot) override {
            auto &ticket = plan->tickets[slot];
            check(cudaEventSynchronize(ticket.ready));
            pending[slot] = false;
            return {ticket.tokens.begin(), ticket.tokens.begin() + requests.size()};
        }
        void Drain() override {
            if (submitted || pending[0] || pending[1]) check(cudaStreamSynchronize(cudaStreamPerThread));
            submitted = pending[0] = pending[1] = false;
        }
    };
  public:
    std::unique_ptr<basellm::DecodePipeline> pipeline(const std::vector<ResponseContext *> &contexts) override {
        for (auto *context : contexts) {
            auto &ctx = *context;
            const auto &s = ctx.generationConfig;
            if (ctx.currentTokens.size() != 1 || ctx.pastKeyValues.size() != c.visits.size() ||
                !ctx.intParams.count("promptLen") || !ctx.intParams.count("index") ||
                !supportsDeviceSampling(s) || s.output_token_least > 0)
                return nullptr;
            const auto *req = static_cast<const Request *>(localCache(ctx.pastKeyValues.front().first).kvStorageOwner.get());
            if (!req || req->draft || (contexts.size() > 1 && req->prefixId)) return nullptr;
        }
        if (contexts.size() == 1) return std::make_unique<Pipeline>(*this, *contexts.front());
        return std::make_unique<BatchPipeline>(*this, contexts);
    }
    void setRequestSeed(uint64_t seed, const std::vector<uint64_t> &seeds = {}) {
        requestSeed = seed;
        promptSeeds = seeds;
    }
    Decoder(const Config &config, const Placement &p, basellm &owner,
            const std::map<std::string, TensorRange> &files, WeightMap *localWeights = nullptr)
        : c(config), placement(p), model(owner), weights(localWeights ? *localWeights : owner.weight), prefixEnabled(owner.saveHistoryChat && RecurrentPrefixCacheEnabled()) {
        if (c.headDim > threads)
            throw std::runtime_error("Inkling attention head exceeds the native kernel width");
        check(cudaSetDevice(p.gpu));
        backboneBf16 = projectionType(p.gpu) == BFLOAT16;
        bf16Mma = FastllmCudaGetBf16MmaProjections();
        graphInput = Buffer<DecodeInput>(1);
        hostVisits.resize(c.visits.size());
        graphVisits = Buffer<DecodeVisit>(hostVisits.size());
#ifndef CUDA_NO_TENSOR_CORE
        const int arch = FastllmCudaRuntimeArch();
#if CUDART_VERSION >= 12000
        if (arch >= 90 && c.hidden == 2304 && c.rank == 64) {
            cudaFuncAttributes attr{};
            check(cudaFuncGetAttributes(&attr, ControllerNormRows<true, __nv_bfloat16, __nv_bfloat16, 2304, false, true>));
            decodePdlSupported = attr.binaryVersion >= 90 && attr.ptxVersion >= 90;
        }
#endif
        if (arch == 120 && c.hidden == controllerWidth && c.rank == controllerRank) {
            batchedController = configureTensorCoreKernel<0>(controllerQkvGates, 128);
            fusedPrefillController =
                configureTensorCoreKernel<controllerSharedBytes>(prefillControllerMix<__nv_bfloat16>, controllerThreads) &&
                configureTensorCoreKernel<controllerSharedBytes>(prefillControllerMix<float>, controllerThreads);
        }
        if (arch >= 80 && (c.headDim == 64 || c.headDim == 128) && c.relative <= threads) {
            tensorPrefill = c.headDim == 128 ? configurePrefill<128, half>() && configurePrefill<128, __nv_bfloat16>()
                                             : configurePrefill<64, half>() && configurePrefill<64, __nv_bfloat16>();
        }
        if (arch >= 75 && c.headDim == 128 && c.relative <= threads) {
            if (weights.weight.at("model.llm.unembed.weight").dataType == FLOAT16) {
                gatedGlobalPrefill = configureGatedPrefill<false, half>();
                gatedWindowPrefill = configureGatedPrefill<true, half>();
            } else if (arch >= 80) {
                gatedGlobalPrefill = configureGatedPrefill<false, __nv_bfloat16>();
                gatedWindowPrefill = configureGatedPrefill<true, __nv_bfloat16>();
                bf16WindowBias = gatedWindowPrefill && configureGatedPrefill<true, __nv_bfloat16, __nv_bfloat16>();
            }
        }
#endif
        check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, p.gpu));
        // The compensated FP16 query operands are finite under this L2 bound.
        if (c.headDim == 128 && c.heads == 3 * c.kvHeads && c.qClip > 0.f && c.qClip <= 65504.f &&
            supportsGlobalRelativeSm120(p.gpu)) {
            // Two waves, each with an integral number of blocks per KV head.
            sm120GlobalSplits = 2 * (smCount / c.kvHeads);
        }
        const int parallelHeads = c.heads == 3 * c.kvHeads ? c.kvHeads : c.heads;
        splitScratch = Matrix(std::max(1, (2 * smCount + parallelHeads - 1) / parallelHeads),
                              c.heads * (attentionWidth(c.headDim) + 1));
        if (tensorPrefill && c.headDim == 128 && c.qClip > 0.f && c.qClip <= 65504.f &&
            configurePrefillKernel<VerifyTraits, VerifyPrefillParams>(pagedVerifyAttention<VerifyTraits>)) {
            verifyAttentionSplits = std::min(splitScratch.rows, std::max(1, 2 * (smCount / c.kvHeads)));
#if CUDART_VERSION >= 12080
            if (decodePdlSupported) {
                decodeBf16Gemv = FastllmCudaGetPdlBf16Gemv();
                if (p.world > 1 && backboneBf16) rankOutputGemv = FastllmCudaGetPdlBf16FloatGemv();
            }
            pagedDecodePdl = sm120GlobalSplits > 0 &&
                configurePrefillKernel<VerifyTraits, VerifyPrefillParams>(pagedSingletonDecode);
#endif
        }
        std::vector<float> ones(std::max(c.headDim, c.hidden), 1.f);
        unitScale = Matrix(1, ones.size());
        check(cudaMemcpy(unitScale.p, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (auto &entry : weights.weight) {
            if (entry.first.compare(0, 10, "model.llm.") == 0)
                entry.second.ToDevice(CUDA, std::vector<int>{p.gpu}, true);
        }
        if (c.draft.steps) {
            initDraft();
            if (c.draft.block * (c.draft.heads / c.draft.kvHeads) <= DraftFlashTraits<2>::CTA_TILE_Q &&
                configurePrefillKernel<DraftFlashTraits<2>, DraftFlashParams>(draftGraphAttention<DraftFlashTraits<2>>))
                draftAttentionWarps = 2;
            else if (configurePrefillKernel<DraftFlashTraits<4>, DraftFlashParams>(draftGraphAttention<DraftFlashTraits<4>>))
                draftAttentionWarps = 4;
        }
        experts = loadExperts(c, p, files, weights);
        getFastllmCublasHandle(); // initialize before graph capture
        int gpu = std::count(p.experts.begin(), p.experts.end(), ExpertDevice::Cuda);
        int numa = std::count(p.experts.begin(), p.experts.end(), ExpertDevice::Numa);
        printf("[Inkling] CUDA:%d backbone; expert layers CUDA=%d CPU=%d NUMA=%d; %zu cache visits\n", p.gpu,
               gpu, c.layers - gpu - numa, numa, c.visits.size());
    }
    ~Decoder() override {
        cudaSetDevice(placement.gpu);
        if (draftQueued) cudaEventSynchronize(publishedPlan->publication.tail->event);
        if (decodeGraph)
            cudaGraphExecDestroy(decodeGraph);
    }
    bool popPending(Data &key, int previous, int &token) override {
        auto *req = static_cast<Request *>(key.kvStorageOwner.get());
        if (!req || !req->draft || req->draft->pending.empty()) return false;
        auto &pending = req->draft->pending;
        if (pending.front().first != previous) throw std::runtime_error("Inkling verified prefix out of sync");
        token = pending.front().second;
        pending.pop_front();
        return true;
    }
#include "prefix-forward.cuh"
    void storeDraftFeatures(const std::vector<Segment> &batch, const std::vector<Matrix> &features, bool bf16) {
        for (const auto &s : batch) if (s.request->draft) {
            if (bf16) storeFeatures<__nv_bfloat16>(*s.request, features, s.offset, s.length);
            else storeFeatures<half>(*s.request, features, s.offset, s.length);
        }
    }

    void warmupDecodeGraph() override {
        std::lock_guard<std::mutex> lock(mutex);
        check(cudaSetDevice(placement.gpu));
        finishPublication();
        initializeKV();
        if (!c.draft.steps && model.maxBatch > 1 && placement.tensorDevices.empty() && placement.world == 1) {
            for (int b = model.maxBatch; b > 1; b = decodeCapacity(b / 2 + b % 2))
                batchDecodePlan(b);
        }
        if (c.draft.steps) {
            const bool bf16 = placement.denseType == BFLOAT16;
            // Each draft plan retains a batch-specific vocabulary distribution.
            // Under TP, prepare additional batch sizes as they are admitted.
            const int warmBatch = placement.world == 1 && placement.tensorDevices.empty() ? std::max(1, model.maxBatch) : 1;
            for (int b = warmBatch; b >= 1; --b) {
                if (bf16)
                    roundPlan<__nv_bfloat16>(b);
                else
                    roundPlan<half>(b);
            }
        }
    }
    std::vector<int> forward(const Data &ids, const std::vector<Data *> &positions,
                             const std::vector<int> &lengths,
                             const std::vector<std::pair<Data *, Data *>> &cache,
                             const std::vector<GenerationConfig> &sampling, const LastTokensManager &history,
                             std::vector<std::vector<float> *> *logits, const ImageInputs *images = nullptr) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(cudaSetDevice(placement.gpu));
        if (publishedPlan && publicationThread != std::this_thread::get_id()) finishPublication();
        initializeKV();
        const_cast<Data &>(ids).ToDevice(CPU);
        if (ids.dataType != FLOAT32 || std::any_of(lengths.begin(), lengths.end(), [](int n) { return n <= 0; }) ||
            std::accumulate(lengths.begin(), lengths.end(), int64_t(0)) != ids.Count(0))
            throw std::runtime_error("Invalid Inkling input token layout");
        const float *values = reinterpret_cast<const float *>(ids.cpuData);
        for (size_t i = 0; i < ids.Count(0); ++i)
            if (!(values[i] >= 0 && values[i] < c.vocab && values[i] == std::floor(values[i])))
                throw std::runtime_error("Invalid Inkling input token");
        // Direct callers may consume one result per forward instead of draining the scheduler queue.
        // Split only those batches that still contain buffered tokens at distinct logical positions.
        if (lengths.size() > 1 && c.draft.steps) {
            bool buffered = false;
            for (size_t b = 0; b < lengths.size(); ++b) {
                auto *key = cache[b * c.visits.size()].first;
                auto *req = key ? static_cast<const Request *>(key->kvStorageOwner.get()) : nullptr;
                buffered |= req && req->draft && !req->draft->pending.empty();
            }
            if (buffered) {
                std::vector<int> result(lengths.size()), ready, sizes;
                std::vector<float> tokens;
                std::vector<Data *> pos;
                std::vector<std::pair<Data *, Data *>> slots;
                std::vector<GenerationConfig> configs;
                std::vector<std::vector<float> *> outputs;
                LastTokensManager histories;
                int offset = 0;
                for (size_t i = 0; i < lengths.size(); ++i) {
                    const int n = lengths[i];
                    auto first = cache.begin() + i * c.visits.size();
                    auto *key = first->first;
                    auto *req = key ? static_cast<Request *>(key->kvStorageOwner.get()) : nullptr;
                    bool pending = n == 1 && req && req->draft && !req->draft->pending.empty();
                    if (pending && positions.at(i)) {
                        auto *position = positions[i];
                        position->ToDevice(CPU);
                        pending = position->dataType == FLOAT32 && position->Count(0) == 1 &&
                                  *reinterpret_cast<const float *>(position->cpuData) ==
                                      req->length - int(req->draft->pending.size());
                    } else
                        pending = false;
                    if (pending) {
                        popPending(*key, int(values[offset]), result[i]);
                    } else {
                        ready.push_back(i);
                        sizes.push_back(n);
                        tokens.insert(tokens.end(), values + offset, values + offset + n);
                        pos.push_back(positions.at(i));
                        configs.push_back(sampling.at(i));
                        slots.insert(slots.end(), first, first + c.visits.size());
                        histories.units.push_back(i < history.units.size() ? history.units[i] : LastTokensUnit{});
                        outputs.push_back(logits && i < logits->size() ? (*logits)[i] : nullptr);
                    }
                    offset += n;
                }
                if (!ready.empty()) {
                    auto emitted = forwardRows(tokens.data(), pos, sizes, slots, configs, histories, &outputs);
                    for (size_t r = 0; r < ready.size(); ++r) result[ready[r]] = emitted[r];
                }
                return result;
            }
        }
        return forwardRows(values, positions, lengths, cache, sampling, history, logits, images);
    }
  private:
    std::vector<int> forwardRows(const float *values, const std::vector<Data *> &positions,
                             const std::vector<int> &lengths,
                             const std::vector<std::pair<Data *, Data *>> &cache,
                             const std::vector<GenerationConfig> &sampling, const LastTokensManager &history,
                             std::vector<std::vector<float> *> *logits, const ImageInputs *images = nullptr) {
        std::vector<Segment> batch;
        int offset = 0;
        for (size_t b = 0; b < lengths.size(); ++b) {
            auto *handle = cache.at(b * c.visits.size()).first;
            if (!handle || !positions.at(b))
                throw std::runtime_error("Invalid Inkling request");
            positions[b]->ToDevice(CPU);
            if (positions[b]->dataType != FLOAT32 || positions[b]->Count(0) != lengths[b])
                throw std::runtime_error("Invalid Inkling position layout");
            const auto *position = reinterpret_cast<const float *>(positions[b]->cpuData);
            int past = static_cast<int>(position[0]);
            for (int i = 0; i < lengths[b]; ++i)
                if (position[i] != past + i || past < 0)
                    throw std::runtime_error("Inkling requires contiguous positions");
            if (!past)
                handle->kvStorageOwner = std::make_shared<Request>(placement.gpu);
            if (!handle->kvStorageOwner)
                throw std::runtime_error("Inkling request state was not initialized");
            auto &req = *static_cast<Request *>(handle->kvStorageOwner.get());
            if (lengths.size() == 1 && lengths[0] == 1 && req.draft && !req.draft->pending.empty() &&
                past == req.length - int(req.draft->pending.size())) {
                int token;
                if (popPending(*handle, int(values[0]), token)) return {token};
            }
            if (req.length != past)
                throw std::runtime_error("Inkling convolution and KV prefix are missing or stale");
            const bool initializing = req.visits.empty();
            if (initializing && placement.world > 1) req.seed = promptSeeds.empty() ? requestSeed + b : promptSeeds.at(b);
            if (initializing) req.visits.resize(c.visits.size());
            for (size_t slot = 0; slot < req.visits.size(); ++slot) {
                const auto &pair = cache[b * c.visits.size() + slot];
                req.visits[slot].key = pair.first;
                req.visits[slot].value = pair.second;
            }
            if (initializing) {
                const int tokenLimit = model.tokensLimit > 0 ? model.tokensLimit : GetMaxTokens();
                Data desc(kvType, {c.kvHeads, 1, attentionWidth(c.headDim)});
                desc.dataDevice = CUDA;
                desc.dataDeviceIds = {placement.gpu};
                const int len = GetPageLen();
                const auto &gen = sampling[b];
                const int64_t prompt = std::max<int64_t>(gen.input_token_length,
                        int64_t(past) + lengths[b] + std::max(0, gen.prefill_chunk_remaining));
                const int64_t limit = gen.output_token_limit > 0
                    ? std::min<int64_t>(model.max_positions, prompt + gen.output_token_limit)
                    : model.max_positions;
                req.limit = int(std::min<int64_t>(limit, tokenLimit > 0 ? tokenLimit : limit));
                req.promptLength = int(prompt);
                if (images && !images->empty()) {
                    if (past || lengths.size() != 1)
                        throw std::runtime_error("Inkling image request must start with a fresh KV cache");
                    req.images = encodeImages(*images, c, weights, placement.gpu, placement.denseType, req.promptLength);
                } else if (prefixEnabled) {
                    req.prefixId = ++prefixSerial;
                    req.nextDecodeCheckpoint = (prompt / len + 1) * len;
                }
                if (c.draft.steps && prompt <= c.draft.maxPositions - c.draft.block && speculativeSampling(gen)) {
                    if (!req.draft) req.draft = std::make_unique<DraftState>();
                    Data draftDesc(FLOAT16, {c.draft.kvHeads, 1, c.draft.dim});
                    draftDesc.dataDevice = CUDA;
                    draftDesc.dataDeviceIds = {placement.gpu};
                    for (int l = 0; l < c.draft.layers; ++l) {
                        if (int(req.draft->kv.size()) <= l) req.draft->kv.push_back(std::make_unique<DraftKV>());
                        auto &owned = req.draft->kv[l];
                        auto &v = owned->cache;
                        v.key = &owned->key;
                        v.value = &owned->value;
                        const int pool = 2 * placement.cacheSlot(int(c.visits.size()) + l);
                        const int64_t extra = c.draft.block - 1;
                        const int maxBatch = std::max(1, model.maxBatch);
                        int64_t pages = tokenLimit > 0 ? (int64_t(tokenLimit) + len - 1) / len +
                                                             maxBatch * ((extra + len - 1) / len)
                                                       : -1;
                        int64_t perRequest =
                            (std::min<int64_t>(limit, c.draft.maxPositions) + extra + len - 1) / len + 1;
                        if (c.draft.local[l]) {
                            int64_t windowPages = (int64_t(c.draft.window) + c.draft.block + len - 2) / len + 1;
                            perRequest = std::min(perRequest, windowPages);
                            if (model.maxBatch > 0)
                                pages = pages < 0 ? windowPages * maxBatch : std::min(pages, windowPages * maxBatch);
                        }
                        int count = int(std::min<int64_t>(pages, INT_MAX));
                        if (!req.restored || c.draft.local[l]) {
                            v.attach(owned->key, pool, draftDesc, count);
                            v.attach(owned->value, pool + 1, draftDesc, count);
                        } else v.shift = pageShift(len);
                        v.reserveTable(std::min<int64_t>(perRequest, owned->key.pagedKVCacheData->maxPages), c.draft.local[l]);
                        if (req.restored) {
                            if (c.draft.local[l]) {
                                v.prepare(past, c.draft.window);
                                req.restored->draftWindows[l].transfer(v, true);
                            } else v.syncTable();
                        }
                    }
                }
                for (size_t slot = 0; slot < c.visits.size(); ++slot) {
                    auto &v = req.visits[slot];
                    int pages = tokenLimit > 0 ? int((int64_t(tokenLimit) + len - 1) / len) : -1;
                    if (c.local[c.visits[slot]] && model.maxBatch > 0) {
                        const int64_t bounded = ((int64_t(c.window) + std::max(0, c.draft.block - 1) + len - 2) / len + 1) * model.maxBatch;
                        pages = int(std::min<int64_t>(bounded, pages > 0 ? pages : std::numeric_limits<int>::max()));
                    }
                    if (!req.restored || c.local[c.visits[slot]]) {
                        v.attach(*v.key, 2 * placement.cacheSlot(slot), desc, pages);
                        v.attach(*v.value, 2 * placement.cacheSlot(slot) + 1, desc, pages);
                    } else v.shift = pageShift(len);
                    // Reserve only page IDs. Physical KV pages remain demand-allocated.
                    int64_t slots = (limit + len - 1) / len;
                    if (c.local[c.visits[slot]])
                        slots = std::min(slots, (int64_t(c.window) + std::max(0, c.draft.block - 1) + len - 2) / len + 1);
                    v.reserveTable(std::min<int64_t>(slots, v.key->pagedKVCacheData->maxPages), c.local[c.visits[slot]]);
                    if (req.restored) {
                        for (int j = 0; j < 4; ++j) v.convState[j] = req.restored->conv[slot][j].load();
                        if (c.local[c.visits[slot]]) {
                            v.prepare(past, c.window);
                            req.restored->windows[slot].transfer(v, true);
                        } else v.syncTable();
                    }
                }
                if (req.restored) {
                    if (req.draft) req.draft->recurrentContext = req.restored->draftContext.load();
                    check(cudaStreamSynchronize(cudaStreamPerThread));
                    req.restored.reset();
                }
            }
            batch.push_back({&req, offset, lengths[b]});
            offset += lengths[b];
        }
        const bool bf16 = backboneBf16;
        bool speculate = c.draft.steps > 0;
        for (size_t b = 0; speculate && b < batch.size(); ++b) {
            const auto &s = batch[b];
            const auto &cfg = sampling[b];
            if (!s.request->draft || !s.request->draft->recurrentContext.p || s.length != 1 ||
                cfg.prefill_chunk_remaining >= 0 || s.request->length < s.request->promptLength ||
                s.request->length > c.draft.maxPositions - c.draft.block ||
                !speculativeSampling(cfg) || (logits && b < logits->size() && (*logits)[b]))
                speculate = false;
        }
        if (speculate)
            return bf16 ? verifyRound<__nv_bfloat16>(batch, values, sampling)
                        : verifyRound<half>(batch, values, sampling);
        finishPublication();
        if (images && !batch.front().request->images.empty()) {
            auto &segment = batch.front();
            auto &req = *segment.request;
            const int chunk = std::max(1, model.GetChunkedPrefillSize());
            while (segment.length > chunk) {
                const std::vector<Segment> part{{&req, 0, chunk}};
                std::vector<Matrix> features;
                bool sampled = false;
                runTarget(values, chunk, part, sampling.front(), sampled, req.draft ? &features : nullptr, false);
                storeDraftFeatures(part, features, bf16);
                req.length += chunk;
                values += chunk;
                offset -= chunk;
                segment.length -= chunk;
            }
        }
        std::vector<float> suffix;
        const bool checkpoint = prefixEnabled &&
            std::any_of(batch.begin(), batch.end(), [](const Segment &s) { return s.request->length < s.request->promptLength; });
        if (checkpoint && offset > 1) checkpointPrompt(values, batch, offset, sampling.front(), bf16, suffix);
        bool collect = std::any_of(batch.begin(),batch.end(),[](const Segment &s){return bool(s.request->draft);});
        std::vector<Matrix> features;
        Matrix out;
        const int *deviceSamples = nullptr;
        if ((GetFastllmEnv().cudaGraph || !placement.tensorDevices.empty()) && batch.size() > 1 &&
            offset == int(batch.size()) && !collect &&
            std::all_of(batch.begin(), batch.end(), [](const Segment &s) {
                return s.request->length > 0 && s.request->length >= s.request->promptLength;
            })) {
            auto &plan = replayBatchDecode(batch, values, sampling);
            out = plan.logits.slice(0, batch.size());
            if (std::all_of(sampling.begin(), sampling.end(), [](const GenerationConfig &cfg) {
                    return ordinarySampling(cfg) || cfg.IsSimpleGreedy();
                }))
                deviceSamples = plan.sampled.p;
        } else {
            bool sampledOnDevice = false;
            out = runTarget(values, offset, batch, sampling.front(), sampledOnDevice, collect ? &features : nullptr);
            if (sampledOnDevice) deviceSamples = graphSample.p;
        }
        storeDraftFeatures(batch, features, bf16);
        std::vector<int> result(batch.size());
        bool greedy = std::all_of(sampling.begin(), sampling.end(),
                                  [](const GenerationConfig &cfg) { return cfg.IsSimpleGreedy(); });
        bool sampled = deviceSamples || greedy;
        if (deviceSamples) {
            check(cudaMemcpy(result.data(), deviceSamples, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        } else if (greedy) {
            Buffer<int> selected(batch.size());
            if (!FastllmCudaGreedySampling(out.p, selected.p, batch.size(), out.cols))
                throw std::runtime_error("Inkling GPU token selection failed");
            check(cudaMemcpy(result.data(), selected.p, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        } else if (std::all_of(sampling.begin(), sampling.end(), ordinarySampling)) {
            std::vector<int> k;
            std::vector<float> p, temperature;
            for (const auto &cfg : sampling) {
                k.push_back(cfg.top_k);
                p.push_back(cfg.top_p);
                temperature.push_back(cfg.temperature);
            }
            // The sampler consumes one RNG draw per launch. Keep each prompt's
            // first draw independent of how many requests the scheduler admitted.
            const int step = promptSeeds.empty() ? int(batch.size()) : 1;
            sampled = true;
            for (int b = 0; b < int(batch.size()); b += step)
                sampled &= FastllmCudaTopKFirstSampling(out.p + size_t(b) * out.cols, temperature.data() + b,
                    k.data() + b, p.data() + b, result.data() + b, step, c.vocab);
        }
        for (size_t b = 0; b < batch.size(); ++b) {
            auto &req = *batch[b].request;
            bool wantLogits = placement.rank == 0 && logits && b < logits->size() && (*logits)[b];
            if (wantLogits || !sampled) {
                Data scores(FLOAT32, {1, c.vocab});
                scores.Allocate(false);
                check(cudaMemcpy(scores.cpuData, out.p + b * out.cols, size_t(c.vocab) * sizeof(float),
                                 cudaMemcpyDeviceToHost));
                auto *data = reinterpret_cast<float *>(scores.cpuData);
                if (wantLogits)
                    (*logits)[b]->assign(data, data + c.vocab);
                if (!sampled)
                    result[b] = LLMSampling(scores, 0, sampling.at(b),
                                            b < history.units.size() ? history.units[b] : LastTokensUnit{});
            }
            const auto &s = batch[b];
            if (req.prefixId && size_t(req.length) == req.prefixTokens.size())
                req.prefixTokens.insert(req.prefixTokens.end(), values + s.offset, values + s.offset + s.length);
            req.length += s.length;
            if (!req.images.empty() && req.length >= req.promptLength) req.images.clear();
            if (checkpoint && req.length <= req.promptLength && req.length % GetPageLen() == 0 &&
                req.prefixTokens.size() == size_t(req.length)) savePrefix(req);
            else checkpointDecode(req);
        }
        return result;
    }
};
#include "rank-runtime.cuh"
} // namespace
std::unique_ptr<Runtime> create(const Config &c, const Placement &p, basellm &model,
                                const std::map<std::string, TensorRange> &ranges) {
    if (p.tensorDevices.size() > 1 &&
        std::all_of(p.experts.begin(), p.experts.end(), [](ExpertDevice e) { return e == ExpertDevice::Cuda; }) &&
        std::all_of(model.weight.weight.begin(), model.weight.weight.end(), [](const auto &entry) {
            const auto type = entry.second.dataType;
            return type == FLOAT32 || type == BFLOAT16 || type == FLOAT16 || entry.second.dims.empty();
        }))
        return std::make_unique<TensorRuntime>(c, p, model, ranges);
    return std::make_unique<Decoder>(c, p, model, ranges);
}
} // namespace inkling2
} // namespace fastllm
