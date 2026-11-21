// Inkling 2 Mini fixed-loop inference. Architecture and checkpoint layout:
// https://huggingface.co/thinkingmachines/Inkling-2-Mini
// https://vllm-project.github.io/2026/11/20/inkling2-mini.html
#include "runtime.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda-attn-res.cuh"
#include <cuda_runtime.h>
#include "attention/variants.cuh"
#include "attention/decode.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace fastllm {
namespace inkling2 {
thread_local DecodeGraphScratch *DecodeGraphScratch::active = nullptr;

std::shared_ptr<void> DecodeGraphScratch::Acquire(size_t bytes) {
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
    Matrix out(x.rows, x.cols);
    if (!width)
        width = x.cols;
    if (clip <= 0 && w && scaleRows == 1) {
        Data src(FLOAT32, {x.rows * (x.cols / width), width}, CUDA, x.p);
        Data dst(FLOAT32, src.dims, CUDA, out.p);
        Data weight(FLOAT32, {width}, CUDA, const_cast<float *>(w));
        FastllmCudaRMSNorm(src, weight, dst, eps);
        return out;
    }
    normalizeRows<<<x.rows *(x.cols / width), threads, 0, cudaStreamPerThread>>>(x.p, w, out.p, width,
                                                                                 scaleRows, eps, clip);
    return out;
}
template <typename T> __global__ void softcap(const T *projected, float *logits, size_t n, float limit) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) logits[i] = limit * tanhf(float(projected[i]) / limit);
}

template <typename T, typename U>
__global__ void castProjection(const T *input, U *output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = U(input[i]);
}

template <typename U, typename T> Rows<U> castRows(const Rows<T> &x) {
    if constexpr (std::is_same<T, U>::value) return x;
    Rows<U> out(x.rows, x.cols);
    const size_t n = size_t(x.rows) * x.cols;
    castProjection<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, out.p, n);
    return out;
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

#include "loop-controller.cuh"

__global__ void conv(const float *x, const float *history, const float *weight, const float *dynamic,
                     float *out, int n, int width, int taps, int heads, const float *residual) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width)
        return;
    int row = i / width, c = i % width;
    float v = x[i];
    for (int t = 0; t < taps; ++t) {
        int r = row - t;
        float a = r < 0 ? history[(taps - 1 + r) * width + c] : x[r * width + c];
        float w = dynamic ? weight[t * width + c] + dynamic[(row * heads + c / (width / heads)) * taps + t]
                          : weight[c * taps + taps - 1 - t];
        v += a * w;
    }
    float y = v * rsqrtf(1.0f + taps * (dynamic ? 2 : 1));
    out[i] = residual ? __fadd_rn(y, residual[i]) : y;
}
#include "paged-kv.cuh"
#include "attention.cuh"
template <typename T, typename U> __device__ float updateConvChannel(const T *input, const DecodeVisit *visit, int index,
                           const float *weight, const U *dynamic, int d, int taps, int heads, int c) {
    float *history = visit->history[index];
    float v = input[c];
    for (int t = 0; t < taps; ++t) {
        float a = t ? history[(taps - 1 - t) * d + c] : float(input[c]);
        float w = dynamic ? weight[t * d + c] + float(dynamic[(c / (d / heads)) * taps + t])
                          : weight[c * taps + taps - 1 - t];
        v += a * w;
    }
    for (int t = 0; t + 1 < taps - 1; ++t)
        history[t * d + c] = history[(t + 1) * d + c];
    if (taps > 1)
        history[(taps - 2) * d + c] = input[c];
    return v * rsqrtf(1.0f + taps * (dynamic ? 2 : 1));
}
template <typename T> __global__ void decodeConv(const T *input, const DecodeVisit *visit, int index, const float *weight,
                           const float *dynamic, float *output, int d, int taps, int heads, const float *residual) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d)
        return;
    float y = updateConvChannel(input, visit, index, weight, dynamic, d, taps, heads, c);
    output[c] = residual ? __fadd_rn(y, residual[c]) : y;
}
template<class T> __global__ void storeDecodeKv(const DecodeVisit *visit, const T *key, const T *value, int width) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < width) {
        const auto &p = visit->pages;
        static_cast<T *>(p.key)[p.offset(visit->past, false, width) + c] = key[c];
        static_cast<T *>(p.value)[p.offset(visit->past, true, width) + c] = value[c];
    }
}
#include "attention-input.cuh"
Matrix convolution(const Matrix &x, const float *w, const Matrix &dyn, int heads, int taps, Matrix &hist, const float *residual) {
    const int count = taps - 1, d = x.cols;
    if (!hist.p) {
        hist = Matrix(count, d);
        if (count)
            check(cudaMemsetAsync(hist.p, 0, size_t(count) * d * sizeof(float), cudaStreamPerThread));
    }
    Matrix out(x.rows, d);
    conv<<<(x.rows * d + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, hist.p, w, dyn.p, out.p, x.rows, d,
                                                                    taps, heads, residual);
    if (count) {
        Matrix next(count, d);
        int keep = std::max(0, count - x.rows);
        if (keep)
            check(cudaMemcpyAsync(next.p, hist.p + size_t(x.rows) * d, size_t(keep) * d * sizeof(float),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        check(cudaMemcpyAsync(next.p + size_t(keep) * d, x.p + size_t(std::max(0, x.rows - count)) * d,
                              size_t(count - keep) * d * sizeof(float), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
        hist = std::move(next);
    }
    return out;
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
template <class Round = float, class Out = float> Rows<Out> mix(const std::vector<Matrix> &states, const Matrix &last, const Query *queries, int nq,
           float eps, bool normalizeValues) {
    using cuda_attn_res::Sources;
    int count = int(states.size()) + 1;
    Buffer<const float *> table(count > Sources::Inline ? count : 0);
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
    Rows<Out> out(nq * last.rows, last.cols);
    Buffer<float> staging(cuda_attn_res::NeedsStaging<Out>(count, queries, nq, last.cols)
                            ? size_t(out.rows) * out.cols : 0);
    cuda_attn_res::Launch<Round>(sources, count, queries, nq, last.rows, last.cols, eps, normalizeValues,
                          out.p, cudaStreamPerThread, staging.p);
    return out;
}
struct DecodeInput {
    float token;
    int topK;
    float temperature, topP;
    uint64_t seed, offset;
};
bool ordinarySampling(const GenerationConfig &cfg) {
    return cfg.repeat_penalty == 1.f && cfg.tool_call_allowed_token_ids.empty() &&
           !cfg.tool_call_token_mask && cfg.top_k > 0 && cfg.temperature > 0.f && cfg.top_p > 0.f &&
           cfg.top_p <= 1.f;
}
uint64_t newSamplingSeed() {
    static thread_local std::mt19937_64 gen(std::random_device{}());
    return gen();
}
struct Request {
    int gpu;
    uint64_t seed = newSamplingSeed();
    explicit Request(int device) : gpu(device) {}
    ~Request() {
        int previous = gpu;
        cudaGetDevice(&previous);
        if (previous != gpu)
            cudaSetDevice(gpu);
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
};
template <typename T> Matrix convolutions(const Rows<T> &input, const float *weight, const Matrix &dynamic, int heads, int taps,
                    const std::vector<Segment> &batch, size_t slot, int index, const Matrix &residual = {}) {
    if (batch.front().decode) {
        Matrix out(1, input.cols);
        decodeConv<<<(input.cols + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            input.p, batch.front().decode + slot, index, weight, dynamic.p, out.p, input.cols, taps, heads, residual.p);
        return out;
    }
    const auto x = castRows<float>(input);
    Matrix result;
    if (batch.size() > 1)
        result = Matrix(x.rows, x.cols);
    for (const auto &seq : batch) {
        auto rows = x.slice(seq.offset, seq.length);
        auto dyn = dynamic.p ? dynamic.slice(seq.offset, seq.length) : Matrix{};
        auto y = convolution(rows, weight, dyn, heads, taps, seq.request->visits[slot].convState[index],
                             residual.p ? residual.p + size_t(seq.offset) * x.cols : nullptr);
        if (batch.size() == 1)
            return y;
        check(cudaMemcpyAsync(result.p + size_t(seq.offset) * x.cols, y.p,
                              size_t(seq.length) * x.cols * sizeof(float), cudaMemcpyDeviceToDevice,
                              cudaStreamPerThread));
    }
    return result;
}
class Decoder final : public Runtime {
    Config c;
    Placement placement;
    basellm &model;
    WeightMap &weights;
    DataType kvType = DATA_AUTO_NONE;
    std::vector<std::unique_ptr<ExpertBackend>> experts;
    std::mutex mutex;
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
    int sm120GlobalSplits = 0;

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
    template <typename T, typename U> void projectInto(const Rows<T> &x, const std::string &key, Rows<U> &out,
                                            int offset, int groups) const {
        const int columns = out.cols;
        const auto &w = weights.weight.at("model.llm." + key);
        const auto type = std::is_same<T, half>::value ? FLOAT16 : BFLOAT16;
        if (w.dataType != type || w.dims.size() != 2 || w.dims[1] != x.cols || columns < 1 ||
            offset < 0 || offset + groups * columns > w.dims[0] || x.rows % groups)
            throw std::runtime_error("Inkling packed projection shape or type: " + key);
        const int n = x.rows / groups, m = x.cols;
        Data input(type, {x.rows, m}, CUDA, x.p), result(std::is_same<U, float>::value ? FLOAT32 : type, {x.rows, columns}, CUDA, out.p);
        Data weightView(type, {groups * columns, m}, CUDA, static_cast<T *>(w.cudaData) + size_t(offset) * m);
        if (groups == 1 && std::is_same<T, U>::value) {
            DoCudaLinear(input, weightView, *GetEmptyData(), result);
        } else {
            FastllmCudaBatchMatMulTransB(input, weightView, result, n * m, columns * m, n * columns,
                                         m, m, groups, n, m, columns, 1.f);
        }
    }

    template <typename T> Rows<T> linear(const Rows<T> &x, const std::string &key, int columns,
                                         int groups = 1) const {
        Rows<T> out(x.rows, columns);
        projectInto(x, key, out, 0, groups);
        return out;
    }

    template <typename T, typename Out> Rows<Out> control(const Matrix *dest, const Matrix &messages, const float *const *norms,
                                         int count, const std::string &prefix) const {
        const int n = dest[0].rows, d = dest[0].cols;
        LoopStreams streams{};
        Rows<T> pair(count * n, 2 * d);
        Rows<Out> out(count * n, d);
        for (int i = 0; i < count; ++i)
            streams.items[i] = {dest[i].p, messages.slice(i * n, n).p, norms[i]};
        if (d == 2304)
            ControllerNormRows<true, T, T, 2304><<<dim3(n, count), 512, 0, cudaStreamPerThread>>>(streams, nullptr, pair.p, c.eps);
        else PairRMSNorm<<<count * n * 2, 512, 0, cudaStreamPerThread>>>(streams, pair.p, n, d, c.eps);
        const auto path = prefix + (count == 1 ? "controllers.mlp." : "controllers.qkv.");
        auto hidden = linear(pair, path + "down_weight", c.rank, count);
        // Controller coefficients feed tanh and the FP32 residual blend directly.
        Matrix gates(count * n, 2 * d);
        projectInto(hidden, path + "gates_weight", gates, 0, count);
        const size_t bytes = size_t(d) * sizeof(float);
        if (d == 2304)
            ControllerNormRows<false, T, Out, 2304><<<dim3(n, count), 512, 0, cudaStreamPerThread>>>(streams, gates.p, out.p, c.eps);
        else BlendRMSNorm<T, Out><<<count * n, 512, bytes <= 16384 ? bytes : 0, cudaStreamPerThread>>>(
            streams, gates.p, out.p, n, d, c.eps);
        return out;
    }

    template <typename T, typename U> std::array<Rows<U>, 8> projectAttention(const std::array<Rows<T>, 3> &inputs,
                                                                 const std::string &att, bool local) const {
        const int q = c.heads * c.headDim, k = c.kvHeads * c.headDim;
        const int taps = local ? 0 : c.kvHeads * c.conv;
        const int widths[] = {q, c.heads * c.relative, q, c.heads, k, taps, k, taps};
        const bool shared = inputs[0].p == inputs[1].p && inputs[0].p == inputs[2].p;
        const int rows = inputs[0].rows, total = std::accumulate(widths, widths + 8, 0);
        Rows<T> packed(rows, total);
        Rows<U> values;
        if (rows > 1) values = Rows<U>(rows, total);
        std::array<Rows<U>, 8> result;
        int offset = 0;
        auto project = [&](const Rows<T> &input, int begin, int end) {
            const int stride = std::accumulate(widths + begin, widths + end, 0);
            auto dest = packed;
            dest.p += size_t(offset) * rows;
            dest.cols = stride;
            projectInto(input, att + "packed.weight", dest, offset, 1);
            if (rows > 1) {
                ProjectionParts parts{};
                std::copy(widths + begin, widths + end, parts.widths);
                const size_t size = size_t(rows) * stride;
                unpackProjection<<<(size + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    dest.p, values.p + size_t(offset) * rows, parts, rows, stride);
            }
            offset += stride;
        };
        if (shared) {
            project(inputs[0], 0, 8);
        } else {
            project(inputs[0], 0, 4);
            project(inputs[1], 4, 6);
            project(inputs[2], 6, 8);
        }
        if (rows == 1) values = castRows<U>(packed);
        for (int i = 0; i < 8; ++i) {
            if (!widths[i]) continue;
            result[i] = values;
            result[i].cols = widths[i];
            values.p += size_t(rows) * widths[i];
        }
        return result;
    }

    template<class KV, class T> Rows<T> attend(const std::array<Rows<T>, 3> &inputs, const std::string &att,
                                      bool local, const std::vector<Segment> &batch, size_t slot) {
        const int d = c.headDim, k = c.kvHeads * d;
        const int aligned = attentionWidth(d), kvWidth = c.kvHeads * aligned;
        const int window = local ? c.window : 0;
        const Data &table = weight(att + "rel_logits_proj.proj");
        if (table.Count(0) != c.relative * c.extent)
            throw std::runtime_error("Inkling relative bias table shape");
        if (batch.front().decode && aligned == d && c.relative <= threads) {
            const auto projected = projectAttention<T, T>(inputs, att, local);
            Rows<T> attended(1, c.heads * d);
            const auto *visit = batch.front().decode + slot;
            Matrix q(1, c.heads * d), bias(c.heads, c.extent);
            RelativeInputs<T> in{projected[0].p, projected[1].p, projected[4].p, projected[6].p,
                              projected[5].p, projected[7].p, param(att + "q_l2_clip.scale", c.heads * d),
                              param(att + "rel_l2_clip.scale", c.heads * c.relative),
                              param(att + (local ? "k_sconv.weight" : "k_sconv.static_w"), k * c.conv),
                              param(att + (local ? "v_sconv.weight" : "v_sconv.static_w"), k * c.conv),
                              static_cast<const float *>(table.cudaData), c.heads, c.kvHeads, d, c.relative,
                              c.extent, c.conv, c.qClip, c.rClip, c.eps};
            int tiles = (c.extent + threads - 1) / threads;
            prepareRelative<KV><<<c.heads * (1 + tiles) + 2 * c.kvHeads, threads, 0, cudaStreamPerThread>>>(
                visit, in, tiles, q.p, bias.p);
            AttentionRow args{q.p, nullptr, nullptr, {}, bias.p, nullptr, nullptr,
                              visit, 0, 0, c.heads, c.kvHeads, d, c.extent,
                              window, c.logFloor, c.logAlpha};
            if (d == 128) launchSplitRelative<128, KV>(args, splitScratch, projected[3].p, projected[2].p,
                                                      attended.p, sm120GlobalSplits);
            else launchSplitRelative<64, KV>(args, splitScratch, projected[3].p, projected[2].p, attended.p);
            return attended;
        }
        const auto projected = projectAttention<T, float>(inputs, att, local);
        const auto &gates = projected[2], &sink = projected[3];
        Matrix attended(projected[0].rows, c.heads * aligned);

        auto q = norm(projected[0], param(att + "q_l2_clip.scale", c.heads * d), c.eps, d, c.heads, c.qClip);
        auto r = norm(projected[1], param(att + "rel_l2_clip.scale", c.heads * c.relative),
                      c.eps, c.relative, c.heads, c.rClip);
        auto convKv = [&](int index, const std::string &name, int which) {
            return convolutions(projected[index], param(att + name + (local ? ".weight" : ".static_w"), k * c.conv),
                                projected[index + 1], c.kvHeads, c.conv, batch, slot, which);
        };
        auto key = norm(convKv(4, "k_sconv", 0), c.qClip > 0.f ? nullptr : unitScale.p, c.eps, d, 1, c.qClip);
        auto val = norm(convKv(6, "v_sconv", 1), unitScale.p, c.eps, d);
        q = resizeHeads(q, d, aligned);
        key = resizeHeads(key, d, aligned);
        val = resizeHeads(val, d, aligned);
        auto typedKey = castRows<KV>(key), typedValue = castRows<KV>(val);
        for (const auto &seq : batch) {
            const DecodeVisit *decode = seq.decode ? seq.decode + slot : nullptr;
            VisitCache *cache = decode ? nullptr : &seq.request->visits[slot];
            int past = decode ? 0 : seq.request->length;
            auto currentKey = typedKey.slice(seq.offset, seq.length),
                 currentValue = typedValue.slice(seq.offset, seq.length);
            const int distances = decode ? c.extent : std::min(c.extent, past + seq.length);
            // Bound temporary relative logits for long prefills. Each row
            // addresses the original complete suffix, including earlier tiles.
            constexpr int tile = 64;
            Matrix bias(std::min(seq.length, tile) * c.heads, distances);
            for (int first = 0; first < seq.length; first += tile) {
                int count = std::min(tile, seq.length - first);
                const int rows = count * c.heads;
                Data input(FLOAT32, {rows, c.relative}, CUDA, r.slice(seq.offset + first, count).p);
                Data output(FLOAT32, {rows, distances}, CUDA, bias.p);
                FastllmCudaBatchMatMul(input, table, output, 0, 0, 0, c.relative, c.extent,
                                       1, rows, c.relative, distances, 1.f);
                AttentionRow args{q.slice(seq.offset, seq.length).p, currentKey.p, currentValue.p,
                    cache ? cache->view() : PageView{}, bias.p,
                    sink.slice(seq.offset, seq.length).p, attended.slice(seq.offset, seq.length).p,
                    decode, past, first, c.heads, c.kvHeads, d, distances,
                    window, c.logFloor, c.logAlpha};
                if (aligned == 128) launchRelative<128, KV>(args, count, seq.length == 1, splitScratch);
                else launchRelative<64, KV>(args, count, seq.length == 1, splitScratch);
            }
            if (decode)
                storeDecodeKv<<<(kvWidth + 255) / 256, 256, 0, cudaStreamPerThread>>>(decode, currentKey.p,
                                                                                currentValue.p, kvWidth);
            else {
                cache->prepare(past + seq.length, window);
                int rows = window ? std::min(seq.length, window) : seq.length;
                appendKv<<<(size_t(rows) * kvWidth + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    cache->view(), currentKey.p, currentValue.p, past, seq.length - rows, kvWidth, rows);
            }
        }
        attended = resizeHeads(attended, aligned, d);
        Data gated(FLOAT32, {attended.rows, attended.cols}, CUDA, attended.p);
        Data gating(FLOAT32, gated.dims, CUDA, gates.p);
        if (!FastllmCudaSigmoidMulTo(gated, gating)) throw std::runtime_error("Inkling attention gate shape");
        return castRows<T>(attended);
    }

    // The previous pass's sources stay fixed throughout the next core pass.
    // Compute consecutive layers' independent controller messages together.
    Matrix crossPassMessages(const std::vector<Matrix> &bank, const Matrix &previous, int first, int layers) {
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
        return mix(bank, previous, queries.data(), 4 * layers, c.eps, true);
    }

    template <typename T> Matrix layer(const Matrix &x, std::vector<Matrix> &blocks, const Matrix &messages,
                 int physical, const std::vector<Segment> &batch, size_t slot) {
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
        std::array<Rows<T>, 3> inputs;
        if (messages.p) {
            Matrix dest = aq ? mix(blocks, x, queries, nq, c.eps, c.normValues) : x;
            Matrix streams[3];
            for (int i = 0; i < 3; ++i) {
                const int stream = local ? 0 : i;
                streams[i] = aq ? dest.slice(stream * x.rows, x.rows) : x;
                norms[i] = norms[stream];
            }
            auto blended = control<T, T>(streams, messages, norms, 3, pre);
            for (int i = 0; i < 3; ++i) inputs[i] = blended.slice(i * x.rows, x.rows);
        } else if (aq) {
            auto dest = mix<T, T>(blocks, x, queries, nq, c.eps, c.normValues);
            for (int i = 0; i < 3; ++i) inputs[i] = dest.slice((local ? 0 : i) * x.rows, x.rows);
        } else {
            for (int i = 0; i < 3; ++i)
                inputs[i] = local && i ? inputs[0] : castRows<T>(norm(x, norms[i], c.eps));
        }
        Rows<T> attended;
        if (kvType == FLOAT16) attended = attend<half, T>(inputs, att, local, batch, slot);
        else if (kvType == BFLOAT16) attended = attend<__nv_bfloat16, T>(inputs, att, local, batch, slot);
        else attended = attend<float, T>(inputs, att, local, batch, slot);
        const bool newBlock = physical % c.residualBlock == 0;
        auto residual =
            convolutions(linear(attended, att + "wo_ud.weight", h), param(pre + "attn_sconv.weight", h * c.conv),
                         {}, 1, c.conv, batch, slot, 2, newBlock ? Matrix{} : x);
        if (newBlock)
            blocks.push_back(x);
        const float *mlpNorm = param(pre + "mlp_norm.weight", h);
        Query mlpQuery{param("attn_res.mlp_queries." + std::to_string(physical), h), messages.p ? nullptr : mlpNorm};
        // Router and expert input share one rounded activation. Round on the
        // final producer only; the loop controller still consumes the FP32 mixture.
        Matrix ffn;
        if (messages.p) {
            auto dest = mix(blocks, residual, &mlpQuery, 1, c.eps, c.normValues);
            ffn = control<T, float>(&dest, messages.slice(3 * x.rows, x.rows), &mlpNorm, 1, pre);
        } else {
            ffn = mix<T>(blocks, residual, &mlpQuery, 1, c.eps, c.normValues);
        }
        auto m = experts[physical]->run(c, ffn, param(pre + "mlp.gate.bias", c.experts));
        return convolutions(m, param(pre + "mlp_sconv.weight", h * c.conv), {}, 1, c.conv,
                                  batch, slot, 3, residual);
    }
    void target(float *ids, int n, const std::vector<Segment> &batch, Matrix &logits) {
        if (weights.weight.at("model.llm.unembed.weight").dataType == FLOAT16)
            targetTyped<half>(ids, n, batch, logits);
        else
            targetTyped<__nv_bfloat16>(ids, n, batch, logits);
    }

    template <typename T> void targetTyped(float *ids, int n, const std::vector<Segment> &batch, Matrix &logits) {
        int h = c.hidden;
        Matrix x(n, h);
        Data tokens(FLOAT32, {n}, CUDA, ids), result(FLOAT32, {n, h}, CUDA, x.p);
        FastllmCudaEmbeddingDirect(tokens, weight("embed.weight"), result);
        x = norm(x, param("embed_norm.weight", h), c.eps);
        Matrix prelude, previous, messageTile;
        int firstMessage = 0, endMessage = 0;
        std::vector<Matrix> blocks, preludeBlocks, bank;
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
                    messageTile = crossPassMessages(bank, previous, i, layers);
                }
                messages = messageTile.slice(4 * (i - firstMessage) * n, 4 * n);
            }
            x = layer<T>(x, blocks, messages, i, batch, slot);
        }
        Query outputQuery{param("attn_res.output_query", h), param("norm.weight", h)};
        auto output = mix(blocks, x, &outputQuery, 1, c.eps, c.normValues);
        Matrix last;
        if (batch.size() == 1)
            last = output.slice(n - 1, 1);
        else {
            last = Matrix(batch.size(), h);
            for (size_t b = 0; b < batch.size(); ++b) {
                auto row = output.slice(batch[b].offset + batch[b].length - 1, 1);
                check(cudaMemcpyAsync(last.p + b * h, row.p, h * sizeof(float), cudaMemcpyDeviceToDevice,
                                      cudaStreamPerThread));
            }
        }
        auto projected = linear(castRows<T>(last), "unembed.weight", c.vocab);
        size_t nLogits = size_t(logits.rows) * logits.cols;
        softcap<<<(nLogits + 255) / 256, 256, 0, cudaStreamPerThread>>>(projected.p, logits.p, nLogits, c.logitCap);
    }

    void decodeBody() {
        auto *in = graphInput.p;
        target(&in->token, 1, {{nullptr, 0, 1, graphVisits.p}}, graphLogits);
        Buffer<float> probabilities(c.vocab);
        size_t bytes = FastllmCudaTopKFirstSamplingWorkspaceBytes(1);
        Buffer<uint8_t> scratch(bytes);
        if (!FastllmCudaTopKFirstSamplingToDeviceWithRng(
                graphLogits.p, probabilities.p, &in->temperature, &in->topK, &in->topP, nullptr, nullptr,
                0, graphSample.p, &in->token, 1, c.vocab, &in->seed, &in->offset, scratch.p, bytes))
            throw std::runtime_error("Inkling decode sampling failed");
    }

    void prepareDecodeScratch(DecodeGraphScratch &pool) {
        // Exercise the exact single-row body at position zero. The private KV
        // row and convolution state are discarded before serving any request.
        const int width = c.kvHeads * attentionWidth(c.headDim);
        Buffer<float> key(width), value(width);
        const size_t stride = size_t(std::max(1, c.conv - 1)) * std::max(width, c.hidden);
        Buffer<float> history(4 * stride);
        Buffer<int2> page(1);
        check(cudaMemsetAsync(page.p, 0, sizeof(int2), cudaStreamPerThread));
        check(cudaMemsetAsync(history.p, 0, 4 * stride * sizeof(float), cudaStreamPerThread));
        DecodeVisit dummy{{key.p, value.p, page.p, 0, 1, 0},
                          {history.p, history.p + stride, history.p + 2 * stride, history.p + 3 * stride}, 0};
        std::fill(hostVisits.begin(), hostVisits.end(), dummy);
        hostInput = {0, 1, 1.f, 1.f, 0, 0};
        check(cudaMemcpyAsync(graphInput.p, &hostInput, sizeof(hostInput), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaMemcpyAsync(graphVisits.p, hostVisits.data(), hostVisits.size() * sizeof(dummy),
                              cudaMemcpyHostToDevice, cudaStreamPerThread));
        DecodeGraphScratch::Scope scope(pool);
        decodeBody();
        check(cudaStreamSynchronize(cudaStreamPerThread));
        pool.sealed = true;
    }
    void captureDecode() {
        graphSample = Buffer<int>(1);
        graphLogits = Matrix(1, c.vocab);
        DecodeGraphScratch scratchPool;
        prepareDecodeScratch(scratchPool);
        cudaGraph_t graph = nullptr;
        check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        bool capturing = true;
        try {
            {
                DecodeGraphScratch::Scope scope(scratchPool);
                decodeBody();
            }
            auto status = cudaStreamEndCapture(cudaStreamPerThread, &graph);
            capturing = false;
            check(status);
            scratchPool.FinishCapture();
            check(cudaGraphInstantiate(&decodeGraph, graph, nullptr, nullptr, 0));
        } catch (...) {
            if (capturing)
                cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (graph)
                cudaGraphDestroy(graph);
            throw;
        }
        graphScratch = std::move(scratchPool);
        size_t nodes = 0;
        cudaGraphGetNodes(graph, nullptr, &nodes);
        cudaGraphDestroy(graph);
        printf("[Inkling] captured full single-request decode graph (%zu nodes, %zu visits, %zu scratch bytes)\n",
               nodes, c.visits.size(), graphScratch.Bytes());
    }
    void initializeKV() {
        if (kvType != DATA_AUTO_NONE) {
            if (kvType != model.kvCacheDataType)
                throw std::runtime_error("Inkling KV dtype changed after the cache was initialized");
            return;
        }
        kvType = model.kvCacheDataType;
        if (model.SupportsDecodePipeline()) {
            try {
                captureDecode();
            } catch (const std::exception &e) {
                fprintf(stderr, "[Inkling] CUDA Graph unavailable, using eager decode: %s\n", e.what());
            }
        }
    }
    void stageDecode(Request &req, int token, const GenerationConfig &sampling, bool sampled,
                     DecodeInput &input, DecodeVisit *visits, bool feedback) {
        for (size_t slot = 0; slot < c.visits.size(); ++slot) {
            auto &v = req.visits[slot];
            v.prepare(req.length + 1, c.local[c.visits[slot]] ? c.window : 0);
            auto &d = visits[slot];
            d.pages = v.view();
            for (int i = 0; i < 4; ++i)
                d.history[i] = v.convState[i].p;
            d.past = req.length;
        }
        input = {float(token), sampled ? std::min(sampling.top_k, c.vocab) : 1,
                 sampled ? sampling.temperature : 1.f, sampled ? sampling.top_p : 1.f,
                 req.seed, uint64_t(req.length)};
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
                     const GenerationConfig &sampling, bool &sampledInGraph) {
        if (batch.size() == 1 && count == 1 && batch[0].request->length > 0) {
            auto &req = *batch[0].request;
            sampledInGraph = decodeGraph && ordinarySampling(sampling) && !sampling.IsSimpleGreedy();
            stageDecode(req, tokens[0], sampling, sampledInGraph, hostInput, hostVisits.data(), false);
            if (!decodeGraph) {
                Matrix logits(1, c.vocab);
                target(&graphInput.p->token, 1, {{&req, 0, 1, graphVisits.p}}, logits);
                return logits;
            }
            check(cudaGraphLaunch(decodeGraph, cudaStreamPerThread));
            return graphLogits;
        }
        Buffer<float> ids(count);
        check(cudaMemcpyAsync(ids.p, tokens, size_t(count) * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        Matrix logits(batch.size(), c.vocab);
        target(ids.p, count, batch, logits);
        return logits;
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
            }
            if (!decoder.decodeGraph) return false;
            auto &cache = context.pastKeyValues;
            auto *req = static_cast<Request *>(cache.front().first.kvStorageOwner.get());
            if (!req || req->visits.size() != cache.size()) return false;
            if (in < 0) {
                if (req->length != context.intParams.at("promptLen") + context.intParams.at("index"))
                    return false;
                for (size_t i = 0; i < cache.size(); ++i) {
                    req->visits[i].key = &cache[i].first;
                    req->visits[i].value = &cache[i].second;
                }
            } else {
                // A page append or sliding-window rotation may reclaim storage
                // still read by the previous replay. Rebind at a drained boundary.
                const int len = cache.front().first.pageLen, begin = req->length + 1 - decoder.c.window;
                if (req->length % len == 0 || (begin > 0 && begin % len == 0)) return false;
            }
            auto &t = decoder.tickets[out];
            t.init(cache.size());
            incomplete = true;
            decoder.stageDecode(*req, context.currentTokens.front(), context.generationConfig, true,
                                t.host->input, t.visits(), in >= 0);
            check(cudaGraphLaunch(decoder.decodeGraph, cudaStreamPerThread));
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

  public:
    std::unique_ptr<basellm::DecodePipeline> pipeline(ResponseContext &ctx) override {
        const auto &s = ctx.generationConfig;
        if (ctx.currentTokens.size() != 1 || ctx.pastKeyValues.size() != c.visits.size() ||
            !ctx.intParams.count("promptLen") || !ctx.intParams.count("index") ||
            !ordinarySampling(s) || s.IsSimpleGreedy() || s.output_logits || s.output_token_least > 0 ||
            s.tool_call_grammar || s.tool_call_content_sampling_enabled ||
            s.tool_call_name_constraint_enabled || s.tool_call_parameter_name_constraint_enabled)
            return nullptr;
        return std::make_unique<Pipeline>(*this, ctx);
    }
    Decoder(const Config &config, const Placement &p, basellm &owner,
            const std::map<std::string, TensorRange> &files)
        : c(config), placement(p), model(owner), weights(owner.weight) {
        if (c.headDim > threads)
            throw std::runtime_error("Inkling attention head exceeds the native kernel width");
        check(cudaSetDevice(p.gpu));
        graphInput = Buffer<DecodeInput>(1);
        hostVisits.resize(c.visits.size());
        graphVisits = Buffer<DecodeVisit>(hostVisits.size());
        int smCount;
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
        std::vector<float> ones(std::max(c.headDim, c.hidden), 1.f);
        unitScale = Matrix(1, ones.size());
        check(cudaMemcpy(unitScale.p, ones.data(), ones.size() * sizeof(float), cudaMemcpyHostToDevice));
        for (auto &entry : weights.weight) {
            if (entry.first.compare(0, 10, "model.llm.") == 0)
                entry.second.ToDevice(CUDA, std::vector<int>{p.gpu}, true);
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
        if (decodeGraph)
            cudaGraphExecDestroy(decodeGraph);
    }
    void warmupDecodeGraph() override {
        std::lock_guard<std::mutex> lock(mutex);
        check(cudaSetDevice(placement.gpu));
        initializeKV();
    }
    std::vector<int> forward(const Data &ids, const std::vector<Data *> &positions,
                             const std::vector<int> &lengths,
                             const std::vector<std::pair<Data *, Data *>> &cache,
                             const std::vector<GenerationConfig> &sampling, const LastTokensManager &history,
                             std::vector<std::vector<float> *> *logits) override {
        std::lock_guard<std::mutex> lock(mutex);
        check(cudaSetDevice(placement.gpu));
        initializeKV();
        const_cast<Data &>(ids).ToDevice(CPU);
        if (ids.dataType != FLOAT32 ||
            std::accumulate(lengths.begin(), lengths.end(), int64_t(0)) != ids.Count(0))
            throw std::runtime_error("Invalid Inkling input token layout");
        const float *values = reinterpret_cast<const float *>(ids.cpuData);
        std::vector<Segment> batch;
        for (size_t i = 0; i < ids.Count(0); ++i)
            if (!(values[i] >= 0 && values[i] < c.vocab && values[i] == std::floor(values[i])))
                throw std::runtime_error("Invalid Inkling input token");
        int offset = 0;
        for (size_t b = 0; b < lengths.size(); ++b) {
            auto *handle = cache.at(b * c.visits.size()).first;
            if (!handle || !positions.at(b) || lengths[b] <= 0)
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
            if (req.length != past)
                throw std::runtime_error("Inkling convolution and KV prefix are missing or stale");
            const bool initializing = req.visits.empty();
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
                const int64_t prompt = int64_t(lengths[b]) + std::max(0, gen.prefill_chunk_remaining);
                const int64_t limit = gen.output_token_limit > 0
                    ? std::min<int64_t>(model.max_positions, prompt + gen.output_token_limit)
                    : model.max_positions;
                for (size_t slot = 0; slot < c.visits.size(); ++slot) {
                    auto &v = req.visits[slot];
                    int pages = tokenLimit > 0 ? int((int64_t(tokenLimit) + len - 1) / len) : -1;
                    if (c.local[c.visits[slot]] && model.maxBatch > 0) {
                        const int64_t bounded = ((int64_t(c.window) + len - 2) / len + 1) * model.maxBatch;
                        pages = int(std::min<int64_t>(bounded, pages > 0 ? pages : std::numeric_limits<int>::max()));
                    }
                    v.attach(*v.key, int(2 * slot), desc, pages);
                    v.attach(*v.value, int(2 * slot + 1), desc, pages);
                    // Reserve only page IDs. Physical KV pages remain demand-allocated.
                    int64_t slots = (limit + len - 1) / len;
                    if (c.local[c.visits[slot]])
                        slots = std::min(slots, (int64_t(c.window) + len - 2) / len + 1);
                    v.reserveTable(int(std::min<int64_t>(slots, v.key->pagedKVCacheData->maxPages)));
                }
            }
            batch.push_back({&req, offset, lengths[b]});
            offset += lengths[b];
        }
        bool sampledInGraph = false;
        auto out = runTarget(values, offset, batch, sampling.front(), sampledInGraph);
        std::vector<int> result(batch.size());
        bool greedy = std::all_of(sampling.begin(), sampling.end(),
                                  [](const GenerationConfig &cfg) { return cfg.IsSimpleGreedy(); });
        bool sampled = sampledInGraph || greedy;
        if (sampledInGraph) {
            check(cudaMemcpy(result.data(), graphSample.p, sizeof(int), cudaMemcpyDeviceToHost));
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
            sampled = FastllmCudaTopKFirstSampling(out.p, temperature.data(), k.data(), p.data(),
                                                       result.data(), batch.size(), c.vocab);
        }
        for (size_t b = 0; b < batch.size(); ++b) {
            auto &req = *batch[b].request;
            bool wantLogits = logits && b < logits->size() && (*logits)[b];
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
            req.length += lengths[b];
        }
        return result;
    }
};
} // namespace
std::unique_ptr<Runtime> create(const Config &c, const Placement &p, basellm &model,
                                const std::map<std::string, TensorRange> &ranges) {
    return std::make_unique<Decoder>(c, p, model, ranges);
}
} // namespace inkling2
} // namespace fastllm
