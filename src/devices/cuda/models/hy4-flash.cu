#include "../../../models/hy4_flash_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
extern "C" cublasHandle_t getFastllmCublasHandle();
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include "hy4-flash-cuda.cuh"
#include "../../../models/hy4_flash_capture.h"

namespace fastllm::hy4 {
void Select(int d) {
    Check(cudaSetDevice(d));
}
void *Allocate(int d, size_t size) {
    Select(d);
    void *p = nullptr;
    Check(cudaMalloc(&p, size));
    return p;
}
void Free(int d, void *p) {
    cudaSetDevice(d);
    cudaFree(p);
}
void CopyToHost(void *dst, const void *src, size_t n) {
    Check(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}
void CopyToGPU(void *dst, const void *src, size_t n) {
    Check(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}
void CopyStage(float *dst, int dd, const float *src, int sd, size_t n) {
    if (dd == sd)
        Check(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice));
    else
        Check(cudaMemcpyPeer(dst, dd, src, sd, n * sizeof(float)));
}
const StepInput *UploadStep(Scratch &s, const std::vector<StepInput> &input) {
    Select(s.device);
    auto dst =
        reinterpret_cast<StepInput *>(s.Get("stepInputs", input.size() * sizeof(StepInput) / sizeof(float)));
    CopyToGPU(dst, input.data(), input.size() * sizeof(StepInput));
    return dst;
}
__global__ static void copyPeerStreams(float *dst, const float *src, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = src[i];
}
void CopyStageCaptured(float *dst, const float *src, size_t count) {
    copyPeerStreams<<<(count + 255) / 256, 256>>>(dst, src, count);
}
void CaptureSegments::Start(int device) {
    if (!enabled)
        return;
    Select(device);
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    activeDevice = device;
}
void *CaptureSegments::Close() {
    if (activeDevice < 0)
        return nullptr;
    Select(activeDevice);
    activeDevice = -1;
    cudaGraph_t graph = nullptr;
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    children.push_back(graph);
    return graph;
}
void CaptureSegments::Abort() {
    if (activeDevice >= 0) {
        cudaSetDevice(activeDevice);
        cudaGraph_t unfinished = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &unfinished);
        if (unfinished)
            cudaGraphDestroy(unfinished);
        activeDevice = -1;
    }
}
CaptureSegments::~CaptureSegments() {
    Abort();
    for (void *g : children)
        cudaGraphDestroy(static_cast<cudaGraph_t>(g));
}

Scratch::~Scratch() {
    for (auto &p : buffers)
        Free(device, p.second.ptr);
}
float *Scratch::Get(const std::string &name, size_t count) {
    auto &b = buffers[name];
    if (b.count < count) {
        auto next = static_cast<float *>(Allocate(device, count * sizeof(float)));
        if (b.ptr)
            Free(device, b.ptr);
        ++generation;
        b.ptr = next;
        b.count = count;
    }
    return b.ptr;
}
__device__ static float bf(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ static float read(const void *p, int type, size_t i) {
    if (type == 0)
        return __bfloat162float(static_cast<const __nv_bfloat16 *>(p)[i]);
    if (type == 1)
        return static_cast<const float *>(p)[i];
    return float(static_cast<const __nv_fp8_e4m3 *>(p)[i]);
}
__device__ static float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
template <bool Maximum = false> __device__ static float warpReduce(float x) {
    for (int delta = 16; delta; delta /= 2) {
        float v = __shfl_down_sync(~0U, x, delta);
        x = Maximum ? fmaxf(x, v) : x + v;
    }
    return x;
}
template <bool Maximum = false> __device__ static float reduce(float x) {
    __shared__ float warp[32];
    x = warpReduce<Maximum>(x);
    if ((threadIdx.x & 31) == 0)
        warp[threadIdx.x >> 5] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
        x = threadIdx.x < (blockDim.x + 31) / 32 ? warp[threadIdx.x] : 0.f;
        x = warpReduce<Maximum>(x);
        if (threadIdx.x == 0)
            warp[0] = x;
    }
    __syncthreads();
    float result = warp[0];
    __syncthreads();
    return result;
}
__global__ static void quantAct(const float *input, float *output, int width) {
    int idx = blockIdx.x * 128 + threadIdx.x, row = blockIdx.y;
    float v = idx < width ? input[size_t(row) * width + idx] : 0;
    float a = fmaxf(reduce<true>(fabsf(v)), 1.e-12f) / 448.f;
    if (idx < width)
        output[size_t(row) * width + idx] = float(__nv_fp8_e4m3(v / a)) * a;
}
__global__ static void gemv(Hy4View w, const float *input, float *out, const int *ids, int routesPerInput,
                            bool roundOutput, Hy4View bias) {
    int r = blockIdx.x, slot = blockIdx.y, e = ids ? ids[slot] : 0;
    size_t first = (size_t(e) * w.rows + r) * w.cols;
    int scalesK = (w.cols + 127) / 128;
    size_t tile = (size_t(e) * ((w.rows + 127) / 128) + r / 128) * scalesK;
    if (routesPerInput)
        input += size_t(slot / routesPerInput) * w.cols;
    float value = 0;
    for (int col = threadIdx.x; col < w.cols; col += blockDim.x) {
        float a = read(w.data, w.type, first + col);
        if (w.scale)
            a *= read(w.scale, w.scaleType, tile + col / 128);
        value = fmaf(a, input[col], value);
    }
    value = reduce(value);
    if (threadIdx.x == 0) {
        float sum = value + (bias.data ? read(bias.data, bias.type, r) : 0.f);
        out[size_t(slot) * w.rows + r] = roundOutput ? bf(sum) : sum;
    }
}
// Each warp owns one output channel. Four adjacent values per lane cover a
// complete 128-column FP8 scale block, with one coalesced matrix transaction.
template <int Type, bool Scaled, bool Dense = false>
__global__ static void warpGemv(Hy4View w, const float *input, float *out, const int *ids, int routesPerInput,
                                bool roundOutput, Hy4View bias) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5),
              slot = blockIdx.y;
    if (row >= w.rows)
        return;
    const int expert = !Dense && ids ? ids[slot] : 0;
    const size_t k0 = (size_t(expert) * w.rows + row) * w.cols;
    if (!Dense && routesPerInput)
        input += size_t(slot / routesPerInput) * w.cols;
    const int blocks = (w.cols + 127) / 128;
    const size_t plane = (size_t(expert) * ((w.rows + 127) / 128) + row / 128) * blocks;
    float sum = 0;
    // Unroll independent weight loads on dense projections. Keep the
    // four-value dot products and their accumulation order unchanged.
    constexpr int unroll = Dense ? 4 : 1;
#pragma unroll (unroll)
    for (int col = 4 * lane; col < w.cols; col += 128) {
        const float4 x = *reinterpret_cast<const float4 *>(input + col);
        float4 weights;
        if constexpr (Type == 2) {
            weights = static_cast<float4>(reinterpret_cast<const __nv_fp8x4_e4m3 *>(w.data)[(k0 + col) / 4]);
        } else if constexpr (Type == 0) {
            const uint2 packed = reinterpret_cast<const uint2 *>(w.data)[(k0 + col) / 4];
            auto a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&packed.x));
            auto b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&packed.y));
            weights = make_float4(a.x, a.y, b.x, b.y);
        } else
            weights = reinterpret_cast<const float4 *>(w.data)[(k0 + col) / 4];
        float dot = fmaf(weights.w, x.w, fmaf(weights.z, x.z, fmaf(weights.y, x.y, weights.x * x.x)));
        if constexpr (Dense && Scaled)
            sum = fmaf(dot, __bfloat162float(static_cast<const __nv_bfloat16 *>(w.scale)[plane + col / 128]), sum);
        else if constexpr (Scaled)
            sum = fmaf(dot, read(w.scale, w.scaleType, plane + col / 128), sum);
        else
            sum += dot;
    }
    sum = warpReduce(sum);
    if (!lane) {
        if constexpr (Dense)
            out[row] = bf(sum + 0.f);
        else {
            sum += bias.data ? read(bias.data, bias.type, row) : 0.f;
            out[size_t(slot) * w.rows + row] = roundOutput ? bf(sum) : sum;
        }
    }
}
static void Linear(Weights &w, Scratch &s, const std::string &name, const float *x, float *y,
                   bool roundOutput = true, const int *ids = nullptr, int slots = 1, int routesPerInput = 0,
                   const std::string &biasName = {}) {
    auto v = w.View(name, s.device);
    if (v.type == 2) {
        const int rows = routesPerInput ? (slots - 1) / routesPerInput + 1 : 1;
        float *qx = s.Get("quantInput", size_t(rows) * v.cols);
        quantAct<<<dim3((v.cols + 127) / 128, rows), 128>>>(x, qx, v.cols);
        x = qx;
    }
    Hy4View bias = biasName.empty() ? Hy4View{} : w.View(biasName, s.device);
    if (v.cols % 4 == 0 && (v.type != 2 || v.scale)) {
        dim3 grid((v.rows + 3) / 4, slots);
        const bool dense = !ids && slots == 1 && !routesPerInput && roundOutput && !bias.data;
        if (dense && v.type == 2 && v.scaleType == 0)
            warpGemv<2, true, true><<<grid, 128>>>(v, x, y, nullptr, 0, true, {});
        else if (dense && v.type == 0)
            warpGemv<0, false, true><<<grid, 128>>>(v, x, y, nullptr, 0, true, {});
        else if (v.type == 2)
            warpGemv<2, true><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
        else if (v.type == 0)
            warpGemv<0, false><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
        else
            warpGemv<1, false><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
    } else
        gemv<<<dim3(v.rows, slots), 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
}
template <int Type> __device__ static float4 loadWeight4(const void *weight, size_t index) {
    if constexpr (Type == 2)
        return static_cast<float4>(reinterpret_cast<const __nv_fp8x4_e4m3 *>(weight)[index]);
    else if constexpr (Type == 0) {
        const uint2 packed = reinterpret_cast<const uint2 *>(weight)[index];
        auto lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&packed.x));
        auto hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&packed.y));
        return make_float4(lo.x, lo.y, hi.x, hi.y);
    } else
        return reinterpret_cast<const float4 *>(weight)[index];
}
__device__ static float dot4(float4 weights, float4 input) {
    return fmaf(weights.w, input.w, fmaf(weights.z, input.z, fmaf(weights.y, input.y, weights.x * input.x)));
}
// Four token vectors share a single read of each weight and scale block.
template <int Type>
__global__ static void batchGemv(Hy4View w, const float *x, float *out, int count, bool roundOutput) {
    int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31, first = blockIdx.y * 4;
    if (row >= w.rows)
        return;
    float sum[4] = {};
    int width = w.cols;
    size_t base = size_t(row) * width;
    for (int col = lane * 4; col < width; col += 128) {
        auto a = loadWeight4<Type>(w.data, (base + col) / 4);
        float scale = 1.f;
        if constexpr (Type == 2)
            scale = read(w.scale, w.scaleType, size_t(row / 128) * ((width + 127) / 128) + col / 128);
#pragma unroll
        for (int r = 0; r < 4; ++r)
            if (first + r < count) {
                auto v = *reinterpret_cast<const float4 *>(x + size_t(first + r) * width + col);
                float dot = dot4(a, v);
                sum[r] = fmaf(dot, scale, sum[r]);
            }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
        if (first + r < count) {
            sum[r] = warpReduce(sum[r]);
            if (!lane)
                out[size_t(first + r) * w.rows + row] = roundOutput ? bf(sum[r]) : sum[r];
        }
}
static const float *QuantizeRows(Scratch &s, const float *x, int rows, int cols) {
    auto q = s.Get("batchQuant", size_t(rows) * cols);
    quantAct<<<dim3((cols + 127) / 128, rows), 128>>>(x, q, cols);
    return q;
}
static void LinearBatch(Weights &w, Scratch &s, const std::string &name, const float *x, float *y, int rows,
                        bool roundOutput = true) {
    if (rows == 1) {
        Linear(w, s, name, x, y, roundOutput);
        return;
    }
    auto v = w.View(name, s.device);
    if (v.type == 2)
        x = QuantizeRows(s, x, rows, v.cols);
    dim3 grid((v.rows + 3) / 4, (rows + 3) / 4);
    if (v.type == 2)
        batchGemv<2><<<grid, 128>>>(v, x, y, rows, roundOutput);
    else if (v.type == 0)
        batchGemv<0><<<grid, 128>>>(v, x, y, rows, roundOutput);
    else
        batchGemv<1><<<grid, 128>>>(v, x, y, rows, roundOutput);
}
// Each warp projects both FFN matrices from the same activation tile and preserves each BF16 rounding.
template <int Type, int Rows>
__global__ static void pairedDraftGemv(Hy4View g, Hy4View u, const float *x, float *gate, float *up) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= g.rows)
        return;
    const size_t offset = size_t(row) * g.cols;
    const int first = blockIdx.y * Rows;
    float gs[Rows] = {}, us[Rows] = {};
    constexpr int unroll = Rows == 1 ? 4 : 1;
#pragma unroll(unroll)
    for (int col = lane * 4; col < g.cols; col += 128) {
        auto gw = loadWeight4<Type>(g.data, (offset + col) / 4);
        auto uw = loadWeight4<Type>(u.data, (offset + col) / 4);
        float gscale = 1.f, uscale = 1.f;
        if constexpr (Type == 2) {
            auto i = (row / 128) * ((g.cols + 127) / 128) + col / 128;
            gscale = __bfloat162float(static_cast<const __nv_bfloat16 *>(g.scale)[i]);
            uscale = __bfloat162float(static_cast<const __nv_bfloat16 *>(u.scale)[i]);
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            auto v = *reinterpret_cast<const float4 *>(x + size_t(first + r) * g.cols + col);
            gs[r] = fmaf(dot4(gw, v), gscale, gs[r]);
            us[r] = fmaf(dot4(uw, v), uscale, us[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        auto a = warpReduce(gs[r]), b = warpReduce(us[r]);
        if (!lane) {
            auto index = size_t(first + r) * g.rows + row;
            gate[index] = bf(a);
            up[index] = bf(b);
        }
    }
}
template <int Type>
static void LaunchGateUp(Hy4View g, Hy4View u, const float *x, float *gate, float *up, int rows) {
    const int blocks = (g.rows + 3) / 4, full = rows / 4 * 4;
    if (full)
        pairedDraftGemv<Type, 4><<<dim3(blocks, full / 4), 128>>>(g, u, x, gate, up);
    if (rows > full)
        pairedDraftGemv<Type, 1><<<dim3(blocks, rows - full), 128>>>(
            g, u, x + size_t(full) * g.cols, gate + size_t(full) * g.rows, up + size_t(full) * g.rows);
}
static void DraftGateUp(Weights &w, Scratch &s, const std::string &prefix, const float *x, float *gate,
                        float *up, int rows) {
    auto g = w.View(prefix + "gate_proj.weight", s.device), u = w.View(prefix + "up_proj.weight", s.device);
    if (g.rows == u.rows && g.cols == u.cols && g.cols % 4 == 0 && g.type == u.type) {
        if (g.type == 2 && g.scaleType == 0 && u.scaleType == 0) {
            LaunchGateUp<2>(g, u, QuantizeRows(s, x, rows, g.cols), gate, up, rows);
            return;
        }
        if (g.type == 0) {
            LaunchGateUp<0>(g, u, x, gate, up, rows);
            return;
        }
    }
    LinearBatch(w, s, prefix + "gate_proj.weight", x, gate, rows);
    LinearBatch(w, s, prefix + "up_proj.weight", x, up, rows);
}
__device__ static void normRow(const float *x, Hy4View w, float *y, int width, float eps, bool roundFirst) {
    float sum = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        sum = fmaf(x[i], x[i], sum);
    float inv = rsqrtf(reduce(sum) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = x[i] * inv;
        y[i] = bf((roundFirst ? bf(v) : v) * (w.data ? read(w.data, w.type, i) : 1.f));
    }
}
__global__ static void norm(const float *x, Hy4View w, float *y, int width, float eps, bool roundFirst,
                            const StepInput *groupEnd = nullptr) {
    if (!groupEnd || (groupEnd->position & 3) == 3)
        normRow(x, w, y, width, eps, roundFirst);
}
struct EngramNormWeights {
    Hy4View head[16];
};
__global__ static void normEngramHeads(float *embedding, EngramNormWeights weights) {
    float *row = embedding + blockIdx.x * 96;
    normRow(row, weights.head[blockIdx.x], row, 96, 1.1920929e-7f, false);
}
static void Norm(Weights &w, Scratch &s, const std::string &name, const float *x, float *y, int width) {
    norm<<<1, 256>>>(x, w.View(name, s.device), y, width, 1e-5f, true);
}
__global__ static void normBatch(const float *input, Hy4View weight, float *output, int width,
                                 int inputStride, int outputStride) {
    normRow(input + size_t(blockIdx.x) * inputStride, weight, output + size_t(blockIdx.x) * outputStride,
            width, 1e-5f, true);
}
static void NormRows(Weights &w, Scratch &s, const std::string &name, const float *input, float *output,
                     int rows, int width, int inputStride, int outputStride) {
    if (rows == 1) {
        Norm(w, s, name, input, output, width);
        return;
    }
    normBatch<<<rows, 256>>>(input, w.View(name, s.device), output, width, inputStride, outputStride);
}
static void NormBatch(Weights &w, Scratch &s, const std::string &name, float *data, int rows, int width,
                      int stride = 0) {
    NormRows(w, s, name, data, data, rows, width, stride ? stride : width, stride ? stride : width);
}
__global__ static void embed(Hy4View w, int token, float *h, const StepInput *step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (step)
        token = step->token;
    if (i < 4 * w.cols)
        h[i] = read(w.data, w.type, size_t(token) * w.cols + i % w.cols);
}
void Embed(Weights &w, Scratch &s, int token, float *h) {
    auto v = w.View("model.embed_tokens.weight", s.device);
    if (token < 0 || token >= v.rows)
        throw std::out_of_range("HY4 token");
    embed<<<(4 * v.cols + 255) / 256, 256>>>(v, token, h, s.step);
}
__global__ static void hc(const float *h, const float *mix, Hy4View bias, Hy4View scale, float *out,
                          float *post, int d, bool head) {
    float square = 0;
    for (int i = threadIdx.x; i < 4 * d; i += blockDim.x)
        square = fmaf(h[i], h[i], square);
    float inv = rsqrtf(reduce(square) / (4 * d) + 1.e-5f);
    float pre[4];
    for (int j = 0; j < 4; j++) {
        pre[j] =
            sigmoid(mix[j] * inv * read(scale.data, scale.type, 0) + read(bias.data, bias.type, j)) + 1.e-6f;
        if (!head && threadIdx.x == 0)
            post[j] = 2.f * sigmoid(mix[j + 4] * inv * read(scale.data, scale.type, 1) +
                                    read(bias.data, bias.type, j + 4)) +
                      1.e-6f;
    }
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = 0;
        for (int j = 0; j < 4; j++)
            v += h[j * d + i] * pre[j];
        out[i] = bf(v);
    }
}
constexpr int HcWidth = 3072, HcInputs = 4 * HcWidth, HcTiles = HcInputs / 512;
__global__ static void hcPartials(const float *weight, const float *h, int rows, float *partial) {
    const int blocks = rows * HcTiles + 8;
    const int token = blockIdx.x / blocks, block = blockIdx.x % blocks;
    h += size_t(token) * HcInputs;
    partial += size_t(token) * blocks;
    if (block < rows * HcTiles) {
        int row = block / HcTiles, start = (block % HcTiles) * 512 + threadIdx.x;
        float sum = 0;
#pragma unroll
        for (int i = 0; i < 512; i += 128)
            sum = fmaf(weight[row * HcInputs + start + i], h[start + i], sum);
        sum = reduce(sum);
        if (threadIdx.x == 0)
            partial[block] = sum;
    } else if (threadIdx.x < 32) {
        // Each of the original eight warps keeps its 48-term FMA chain.
        int warp = block - rows * HcTiles, start = warp * 32 + threadIdx.x;
        float square = 0;
#pragma unroll
        for (int i = 0; i < HcInputs; i += 256) {
            float v = h[start + i];
            square = fmaf(v, v, square);
        }
        square = warpReduce(square);
        if (threadIdx.x == 0)
            partial[block] = square;
    }
}
__device__ static void hcGates(const float *partial, int rows, Hy4View bias, Hy4View scale, float *pre,
                               float *post, bool writePost) {
    int row = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row < rows) {
        float dot = warpReduce(lane < HcTiles ? partial[row * HcTiles + lane] : 0.f);
        float square = warpReduce(lane < 8 ? partial[rows * HcTiles + lane] : 0.f);
        if (lane == 0) {
            float inv = rsqrtf(square / HcInputs + 1.e-5f);
            float gate = sigmoid(dot * inv * read(scale.data, scale.type, row / 4) +
                                 read(bias.data, bias.type, row));
            if (row < 4)
                pre[row] = gate + 1.e-6f;
            else if (writePost)
                post[row - 4] = 2.f * gate + 1.e-6f;
        }
    }
    __syncthreads();
}
__global__ static void hcMixOutput(const float *h, const float *partial, int rows, Hy4View bias,
                                   Hy4View scale, float *out, float *post) {
    __shared__ float pre[4];
    hcGates(partial, rows, bias, scale, pre, post, blockIdx.x == 0);
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;
    for (int j = 0; j < 4; j++)
        value += h[j * HcWidth + col] * pre[j];
    out[col] = bf(value);
}
static void HyperPre(Weights &w, Scratch &s, const std::string &prefix, float *h, float *x, float *post, bool head) {
    auto name = prefix + (head ? "hc_head_fn" : "hc_fn");
    auto weight = w.View(name, s.device);
    auto bias = w.View(prefix + (head ? "hc_head_base" : "hc_base"), s.device),
         scale = w.View(prefix + (head ? "hc_head_scale" : "hc_scale"), s.device);
    if (weight.type == 1 && weight.cols == HcInputs && weight.rows == (head ? 4 : 8)) {
        int blocks = weight.rows * HcTiles + 8;
        float *partial = s.Get("hcPartials", blocks);
        hcPartials<<<blocks, 128>>>(static_cast<const float *>(weight.data), h, weight.rows, partial);
        hcMixOutput<<<HcWidth / 256, 256>>>(h, partial, weight.rows, bias, scale, x, post);
        return;
    }
    float *mix = s.Get("hcMix", 8);
    Linear(w, s, name, h, mix, false);
    hc<<<1, 256>>>(h, mix, bias, scale, x, post, 3072, head);
}
// One CTA owns a token. Keep the BF16-rounded four-stream mixture on chip for RMSNorm.
__global__ static void hcMixNorm(const float *streams, const float *partials, Hy4View bias, Hy4View scale,
                                 Hy4View weight, float *input, float *posts) {
    const int token = blockIdx.x, tid = threadIdx.x, col = tid * 4;
    const float *h = streams + size_t(token) * HcInputs;
    __shared__ float pre[4], mixed[HcWidth];
    hcGates(partials + size_t(token) * (8 * HcTiles + 8), 8, bias, scale, pre, posts + token * 4, true);
    float4 v = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 x = reinterpret_cast<const float4 *>(h + j * HcWidth)[tid];
        v.x = fmaf(x.x, pre[j], v.x);
        v.y = fmaf(x.y, pre[j], v.y);
        v.z = fmaf(x.z, pre[j], v.z);
        v.w = fmaf(x.w, pre[j], v.w);
    }
    float local[4] = {bf(v.x), bf(v.y), bf(v.z), bf(v.w)};
#pragma unroll
    for (int j = 0; j < 4; ++j)
        mixed[col + j] = local[j];
    __syncthreads();
    // Keep the same 256 accumulation chains and warp reduction as standalone Norm.
    float sum = 0;
    if (tid < 256) {
#pragma unroll
        for (int i = tid; i < HcWidth; i += 256)
            sum = fmaf(mixed[i], mixed[i], sum);
    }
    const float inv = rsqrtf(reduce(sum) / HcWidth + 1.e-5f);
    float4 y;
    y.x = bf(bf(local[0] * inv) * read(weight.data, weight.type, col));
    y.y = bf(bf(local[1] * inv) * read(weight.data, weight.type, col + 1));
    y.z = bf(bf(local[2] * inv) * read(weight.data, weight.type, col + 2));
    y.w = bf(bf(local[3] * inv) * read(weight.data, weight.type, col + 3));
    reinterpret_cast<float4 *>(input + size_t(token) * HcWidth)[tid] = y;
}
void HyperPreNorm(Weights &w, Scratch &s, const std::string &prefix, const std::string &normName,
                  float *streams, float *input, float *posts, int rows) {
    const auto weight = w.View(prefix + "hc_fn", s.device);
    if (weight.type != 1 || weight.rows != 8 || weight.cols != HcInputs) {
        for (int row = 0; row < rows; ++row)
            HyperPre(w, s, prefix, streams + size_t(row) * HcInputs, input + size_t(row) * HcWidth,
                     posts + row * 4, false);
        NormBatch(w, s, normName, input, rows, HcWidth);
        return;
    }
    const int blocks = 8 * HcTiles + 8;
    auto partials = s.Get("hcPartials", size_t(rows) * blocks);
    hcPartials<<<rows * blocks, 128>>>(static_cast<const float *>(weight.data), streams, 8, partials);
    hcMixNorm<<<rows, HcWidth / 4>>>(streams, partials, w.View(prefix + "hc_base", s.device),
                                     w.View(prefix + "hc_scale", s.device), w.View(normName, s.device),
                                     input, posts);
}

// Each CTA normalizes one column tile and updates all four residual streams.
// The branch is read-only, so tiles can recompute its small reduction independently.
template <bool Normalize>
__global__ static void hcPostNorm(float *streams, const float *branch, const float *posts, Hy4View weight) {
    constexpr int tiles = HcWidth / 256;
    const int token = blockIdx.x / tiles, col = (blockIdx.x % tiles) * 256 + threadIdx.x;
    const float *x = branch + size_t(token) * HcWidth;
    float value = x[col];
    if constexpr (Normalize) {
        float sum = 0;
#pragma unroll
        for (int i = threadIdx.x; i < HcWidth; i += 256)
            sum = fmaf(x[i], x[i], sum);
        float inv = rsqrtf(reduce(sum) / HcWidth + 1.e-5f);
        value = bf(bf(value * inv) * read(weight.data, weight.type, col));
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float &h = streams[(size_t(token) * 4 + j) * HcWidth + col];
        h = bf(fmaf(posts[token * 4 + j], value, h));
    }
}
void HyperPostNorm(Weights &w, Scratch &s, const std::string &normName, float *streams, const float *branch,
                   const float *posts, int rows, bool optional) {
    const int blocks = rows * (HcWidth / 256);
    if (optional && !w.meta.count(normName))
        hcPostNorm<false><<<blocks, 256>>>(streams, branch, posts, {});
    else
        hcPostNorm<true><<<blocks, 256>>>(streams, branch, posts, w.View(normName, s.device));
}
__global__ static void activate(const float *gate, const float *up, float *out, int width, bool routed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, slot = blockIdx.y;
    if (i >= width)
        return;
    const float *g = gate + size_t(slot) * 2 * width;
    float a = routed ? fminf(g[i], 10.f) : g[i], b = up ? up[i] : fminf(10.f, fmaxf(-10.f, g[width + i]));
    out[size_t(slot) * width + i] = bf(bf(a * sigmoid(a)) * b);
}
static void DenseMlp(Weights &w, Scratch &s, const std::string &p, const float *x, float *out) {
    const int width = w.meta.at(p + "gate_proj.weight").dims[0];
    auto a = s.Get("denseGate", width), b = s.Get("denseUp", width), mid = s.Get("denseMid", width);
    Linear(w, s, p + "gate_proj.weight", x, a);
    Linear(w, s, p + "up_proj.weight", x, b);
    activate<<<(width + 255) / 256, 256>>>(a, b, mid, width, false);
    Linear(w, s, p + "down_proj.weight", mid, out);
}
__global__ static void denseBatchActivation(const float *gate, const float *up, float *out, int size) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < size)
        out[i] = bf(bf(gate[i] * sigmoid(gate[i])) * up[i]);
}
template <bool Draft>
static void DenseMlpRows(Weights &w, Scratch &s, const std::string &p, const float *input, float *output,
                         int rows) {
    if (rows == 1) {
        DenseMlp(w, s, p, input, output);
        return;
    }
    const int intermediate = w.meta.at(p + "gate_proj.weight").dims[0];
    for (int first = 0; first < rows; first += 16) {
        int n = std::min(16, rows - first), size = n * intermediate;
        auto x = input + size_t(first) * 3072;
        auto out = output + size_t(first) * 3072;
        auto gate = s.Get("batchDenseGate", size), up = s.Get("batchDenseUp", size),
             mid = s.Get("batchDenseMid", size);
        if constexpr (Draft) {
            DraftGateUp(w, s, p, x, gate, up, n);
        } else {
            LinearBatch(w, s, p + "gate_proj.weight", x, gate, n);
            LinearBatch(w, s, p + "up_proj.weight", x, up, n);
        }
        denseBatchActivation<<<(size + 255) / 256, 256>>>(gate, up, mid, size);
        LinearBatch(w, s, p + "down_proj.weight", mid, out, n);
    }
}
template <bool Draft>
static void NormalizedDenseMlpRows(Weights &w, Scratch &s, const std::string &p, const float *input,
                                   float *output, int rows) {
    DenseMlpRows<Draft>(w, s, p, input, output, rows);
    if (w.meta.count(p + "post_fc2_norm.weight"))
        NormBatch(w, s, p + "post_fc2_norm.weight", output, rows, HcWidth);
}
void DenseMlpBatch(Weights &w, Scratch &s, const std::string &p, const float *input, float *output,
                   int rows) {
    DenseMlpRows<false>(w, s, p, input, output, rows);
}
__global__ static void pick(const float *logits, Hy4View bias, int *id, float *score, float scale) {
    // Stable sorting retains the lower expert ID on tied corrected scores.
    using Sort = cub::BlockRadixSort<float, 256, 2, int>;
    __shared__ typename Sort::TempStorage scratch;
    __shared__ float probabilities[16];
    float keys[2];
    int indices[2];
    for (int j = 0; j < 2; ++j) {
        int i = 2 * threadIdx.x + j;
        indices[j] = i;
        keys[j] = sigmoid(logits[i]) + read(bias.data, bias.type, i);
    }
    Sort(scratch).SortDescending(keys, indices);
    if (threadIdx.x < 8) {
        for (int j = 0; j < 2; ++j) {
            int rank = 2 * threadIdx.x + j;
            id[rank] = indices[j];
            probabilities[rank] = sigmoid(logits[indices[j]]);
        }
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sum = 0;
        for (int j = 0; j < 16; ++j)
            sum += probabilities[j];
        score[threadIdx.x] = probabilities[threadIdx.x] / sum * scale;
    }
}
static void Route(Weights &w, Scratch &s, int layer, const float *x, int *ids, float *scores, float *latent) {
    auto p = "model.layers." + std::to_string(layer) + ".mlp.";
    auto logits = s.Get("router", 512);
    Linear(w, s, p + "gate.weight", x, logits, false);
    auto bias = w.View(p + "gate.e_score_correction_bias", s.device);
    pick<<<1, 256>>>(logits, bias, ids, scores, w.config["routed_scaling_factor"].number_value());
    Linear(w, s, p + "routed_expert_down_proj.weight", x, latent);
}
void RouteBatch(Weights &w, Scratch &s, int layer, const float *x, int *ids, float *scores, float *latents,
                int rows) {
    if (rows == 1) {
        Route(w, s, layer, x, ids, scores, latents);
        return;
    }
    auto p = "model.layers." + std::to_string(layer) + ".mlp.";
    auto logits = s.Get("batchRouter", rows * 512);
    LinearBatch(w, s, p + "gate.weight", x, logits, rows, false);
    auto bias = w.View(p + "gate.e_score_correction_bias", s.device);
    for (int i = 0; i < rows; ++i)
        pick<<<1, 256>>>(logits + i * 512, bias, ids + i * 16, scores + i * 16,
                         w.config["routed_scaling_factor"].number_value());
    LinearBatch(w, s, p + "routed_expert_down_proj.weight", x, latents, rows);
}
__global__ static void sumExperts(const float *down, const int *id, const float *score, float *out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= 1536)
        return;
    int row = blockIdx.y;
    down += size_t(row) * 16 * 1536;
    id += row * 16;
    score += row * 16;
    out += size_t(row) * 1536;
    int order[16];
    for (int j = 0; j < 16; j++) {
        int t = j;
        while (t && id[order[t - 1]] > id[j]) {
            order[t] = order[t - 1];
            --t;
        }
        order[t] = j;
    }
    float sum = 0;
    for (int i : order)
        sum += bf(down[i * 1536 + k] * bf(score[i]));
    out[k] = bf(sum);
}
void ResidentExperts(Weights &w, Scratch &s, int layer, const float *x, const int *ids, const float *score,
                     float *out, int rows) {
    const auto p = "model.layers." + std::to_string(layer) + ".mlp.experts.";
    // Bound the activation workspace independently of the scheduler's batch.
    constexpr int tileRows = 64;
    int maxRoutes = 16 * std::min(tileRows, rows);
    auto gates = s.Get("residentGU", size_t(maxRoutes) * 3072);
    auto activated = s.Get("residentAct", size_t(maxRoutes) * 1536);
    auto down = s.Get("residentDown", size_t(maxRoutes) * 1536);
    for (int first = 0; first < rows; first += tileRows) {
        int count = std::min(tileRows, rows - first), routes = 16 * count;
        auto expertIds = ids + first * 16;
        Linear(w, s, p + "gate_up_proj", x + size_t(first) * 1536, gates, true, expertIds, routes, 16);
        activate<<<dim3(6, routes), 256>>>(gates, nullptr, activated, 1536, true);
        Linear(w, s, p + "down_proj", activated, down, true, expertIds, routes, 1);
        sumExperts<<<dim3(6, count), 256>>>(down, expertIds, score + first * 16, out + size_t(first) * 1536);
    }
}
__global__ static void add(float *x, const float *y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        x[i] = bf(x[i] + y[i]);
}
void CombineExpertsBatch(Weights &w, Scratch &s, int layer, const float *input, const float *routed,
                         float *out, int rows) {
    auto p = "model.layers." + std::to_string(layer) + ".mlp.";
    auto shared = s.Get("batchShared", size_t(rows) * 3072);
    NormalizedDenseMlpRows<false>(w, s, p + "shared_experts.", input, shared, rows);
    LinearBatch(w, s, p + "routed_expert_up_proj.weight", routed, out, rows);
    add<<<rows * 12, 256>>>(out, shared, rows * 3072);
}
void ProjectHead(Weights &w, Scratch &s, float *h, float *values, Data *draftHidden, int position,
                 int draftWidth, int rows) {
    auto x = s.Get("headX", size_t(rows) * 3072);
    for (int row = 0; row < rows; ++row) {
        float *hidden = x + size_t(row) * 3072;
        HyperPre(w, s, "model.hc_head.", h + size_t(row) * 4 * 3072, hidden, nullptr, true);
    }
    NormBatch(w, s, "model.norm.weight", x, rows, 3072);
    if (draftHidden)
        for (int row = 0; row < rows; ++row)
            SaveDraftHidden(*draftHidden, s.device, position + row, draftWidth, x + size_t(row) * 3072,
                            s.step ? s.step + row : nullptr);
    LinearBatch(w, s, "lm_head.weight", x, values, rows);
}

__device__ static float rope(float a, float b, int j, int pos, float theta) {
    float phase = pos * powf(theta, -2.f * (j / 2) / 64.f);
    return bf(a * bf(cosf(phase)) + ((j & 1) ? b : -b) * bf(sinf(phase)));
}
// Indexed layers retain one normalized token key per slot. The final slot of a
// closed group also owns its pooled, rotated key. Both travel with the request KV.
template <bool Value = false>
__device__ static __nv_bfloat16 *kvRow(const KVPage *const *pages, int slot, int stride) {
    const auto &page = (*pages)[slot / KVPageTokens];
    return static_cast<__nv_bfloat16 *>(Value ? page.value : page.key) + size_t(slot % KVPageTokens) * stride;
}
__global__ static void storeIndexKey(const float *input, int stride, int pos, const StepInput *step,
                                     const KVPage *const *pages) {
    if (step)
        pos = step->position;
    kvRow(pages, pos, stride)[stride - 256 + threadIdx.x] = __float2bfloat16(input[threadIdx.x]);
}
__global__ static void poolIndexKey(int stride, int pos, Hy4View value, Hy4View gate, Hy4View ape,
                                    float *pooled, const StepInput *step, const KVPage *const *pages) {
    if (step)
        pos = step->position;
    if ((pos & 3) != 3)
        return;
    const int channel = blockIdx.x, col = threadIdx.x;
    float values[4], gates[4];
    for (int slot = 0; slot < 4; ++slot) {
        float x = __bfloat162float(kvRow(pages, pos - 3 + slot, stride)[stride - 256 + col]);
        values[slot] = reduce(x * read(value.data, value.type, channel * 128 + col));
        gates[slot] = reduce(x * read(gate.data, gate.type, channel * 128 + col)) +
                      read(ape.data, ape.type, slot * 128 + channel);
    }
    if (col == 0) {
        float peak = fmaxf(fmaxf(gates[0], gates[1]), fmaxf(gates[2], gates[3]));
        float numerator = 0, denominator = 0;
        for (int i = 0; i < 4; ++i) {
            float weight = expf(gates[i] - peak);
            numerator += weight * values[i];
            denominator += weight;
        }
        pooled[channel] = numerator / denominator;
    }
}
__global__ static void storeGroupKey(const float *pooled, int stride, int pos, const StepInput *step,
                                     const KVPage *const *pages) {
    if (step)
        pos = step->position;
    if ((pos & 3) != 3)
        return;
    int i = threadIdx.x;
    float x = i < 64 ? pooled[i] : rope(pooled[i], pooled[i ^ 1], i - 64, pos - 3, 10000000.f);
    kvRow(pages, pos, stride)[stride - 128 + i] = __float2bfloat16(x);
}
__global__ static void rotateIndexQuery(const float *input, float *output, int pos, const StepInput *step) {
    if (step)
        pos = step->position;
    int i = blockIdx.x * 128 + threadIdx.x, d = threadIdx.x;
    output[i] = d < 64 ? input[i] : rope(input[i], input[i ^ 1], d - 64, pos, 10000000.f);
}
__global__ static void scoreIndexGroups(const float *query, const float *weights, int stride, float *scores,
                                        const StepInput *step, const KVPage *const *pages) {
    int groups = step ? step->position / 4 : gridDim.x;
    for (int group = blockIdx.x; group < groups; group += gridDim.x) {
        const int col = threadIdx.x;
        float key = __bfloat162float(kvRow(pages, group * 4 + 3, stride)[stride - 128 + col]);
        float score = 0;
        for (int head = 0; head < 16; ++head) {
            float dot = reduce(query[head * 128 + col] * key);
            score += fmaxf(dot, 0.f) * weights[head] * 0.0220970869f;
        }
        if (!col)
            scores[group] = score;
    }
}
__device__ static unsigned orderedScore(float score) {
    unsigned bits = __float_as_uint(score == 0.f ? 0.f : score);
    return bits ^ ((bits >> 31) ? ~0u : 0x80000000u);
}
// Select the kth score by its sortable bit representation. Stable tie handling
// and a block scan emit group indices chronologically, matching a masked softmax.
__global__ static void selectIndexGroups(const float *scores, int groups, int *chosen,
                                         const StepInput *step) {
    if (step)
        groups = step->position / 4;
    constexpr int count = 511;
    unsigned prefix = 0, mask = 0;
    int remaining = count, tid = threadIdx.x;
    for (int bit = 31; bit >= 0; --bit) {
        unsigned flag = 1u << bit;
        mask |= flag;
        unsigned candidate = prefix | flag;
        int local = 0;
        for (int i = tid; i < groups; i += blockDim.x)
            local += (orderedScore(scores[i]) & mask) == candidate;
        int matches = int(reduce(float(local)));
        if (matches >= remaining)
            prefix = candidate;
        else
            remaining -= matches;
    }
    __shared__ int eqCounts[8], winCounts[8], accepted, equalSeen;
    if (!tid) {
        accepted = 0;
        equalSeen = 0;
    }
    __syncthreads();
    for (int base = 0; base < groups; base += blockDim.x) {
        int i = base + tid, warp = tid >> 5, lane = tid & 31;
        unsigned key = i < groups ? orderedScore(scores[i]) : 0;
        unsigned lower = (1u << lane) - 1;
        unsigned eq = __ballot_sync(~0u, i < groups && key == prefix);
        if (!lane)
            eqCounts[warp] = __popc(eq);
        __syncthreads();
        int rank = equalSeen + __popc(eq & lower);
        for (int w = 0; w < warp; ++w)
            rank += eqCounts[w];
        bool keep = i < groups && (key > prefix || (key == prefix && rank < remaining));
        unsigned wins = __ballot_sync(~0u, keep);
        if (!lane)
            winCounts[warp] = __popc(wins);
        __syncthreads();
        int dst = accepted + __popc(wins & lower);
        for (int w = 0; w < warp; ++w)
            dst += winCounts[w];
        if (keep)
            chosen[dst] = i;
        __syncthreads();
        if (!tid) {
            for (int w = 0; w < 8; ++w) {
                accepted += winCounts[w];
                equalSeen += eqCounts[w];
            }
        }
        __syncthreads();
    }
    if (!tid)
        chosen[count] = groups;
}
static int *IndexPool(Weights &w, Scratch &s, const std::string &attn, const float *hidden,
                      const float *residual, int stride, int pos, const KVPage *const *pages) {
    const auto p = attn + "compressed_indexer.";
    auto key = s.Get("indexTokenKey", 128);
    Linear(w, s, p + "token_key_projection.weight", hidden, key);
    Norm(w, s, p + "token_key_norm.weight", key, key, 128);
    storeIndexKey<<<1, 128>>>(key, stride, pos, s.step, pages);
    if (s.step || (pos & 3) == 3) {
        auto value = w.View(p + "key_value_projection.weight", s.device);
        auto gate = w.View(p + "key_gate_projection.weight", s.device);
        auto ape = w.View(p + "intra_group_ape", s.device);
        auto pooled = s.Get("indexPooled", 128);
        poolIndexKey<<<128, 128>>>(stride, pos, value, gate, ape, pooled, s.step, pages);
        norm<<<1, 128>>>(pooled, w.View(p + "key_norm.weight", s.device), pooled, 128, 1.e-5f, false, s.step);
        storeGroupKey<<<1, 128>>>(pooled, stride, pos, s.step, pages);
    }
    if (pos < 2048)
        return nullptr;
    const int closedGroups = pos / 4;
    const int groupCapacity = ((pos / KVPageTokens * KVPageTokens + KVPageTokens - 1) / 4 + 31) / 32 * 32;
    auto query = s.Get("indexQuery", 16 * 128), rotated = s.Get("indexRotaryQuery", 16 * 128);
    auto weights = s.Get("indexHeadWeights", 16), scores = s.Get("indexGroupScores", groupCapacity);
    auto chosen = reinterpret_cast<int *>(s.Get("indexChosen", 512));
    Linear(w, s, p + "query_projection.weight", residual, query);
    rotateIndexQuery<<<16, 128>>>(query, rotated, pos, s.step);
    Linear(w, s, p + "head_weight_projection.weight", hidden, weights);
    const int grid = s.step ? 512 : closedGroups;
    scoreIndexGroups<<<grid, 128>>>(rotated, weights, stride, scores, s.step, pages);
    selectIndexGroups<<<1, 256>>>(scores, closedGroups, chosen, s.step);
    return chosen;
}

template <bool DraftRows = false>
__global__ static void packKV(const float *kv, const float *rot, int heads, int pos, int slot, float theta,
                              int keyStride, const StepInput *step, int keep, const KVPage *const *pages) {
    if constexpr (DraftRows) {
        const int row = blockIdx.y;
        kv += size_t(row) * heads * 448;
        rot += size_t(row) * 576;
        pos += row;
        slot += row;
    }
    if (step) {
        pos = step->position;
        slot = pos % keep;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * 256)
        return;
    int h = i / 256, d = i % 256;
    float k = d < 192 ? kv[h * 448 + d] : rope(rot[d - 192], rot[(d - 192) ^ 1], d - 192, pos, theta);
    kvRow(pages, slot, keyStride)[i] = __float2bfloat16(k);
    kvRow<true>(pages, slot, heads * 256)[i] = __float2bfloat16(kv[h * 448 + 192 + d]);
}
template <bool DraftRows = false>
__global__ static void rotQ(const float *input, float *query, int size, int pos, float theta,
                            const StepInput *step = nullptr) {
    if constexpr (DraftRows) {
        const int row = blockIdx.y;
        input += size_t(row) * size;
        query += size_t(row) * size;
        pos += row;
    }
    if (step)
        pos = step->position;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    int d = i % 256;
    query[i] = d < 192 ? input[i] : rope(input[i], input[i ^ 1], d - 192, pos, theta);
}
template <bool DraftRows = false>
__global__ static void scoreKV(const float *q, float *score, int start, int length, int heads, int capacity,
                               int keyStride, const int *chosen, bool direct, const StepInput *step,
                               int window, const KVPage *const *pages) {
    if constexpr (DraftRows) {
        const int row = blockIdx.z;
        q += size_t(row) * heads * 256;
        score += size_t(row) * heads * gridDim.x;
        length += row;
    }
    if (step) {
        length = window < 0 ? 2045 + (step->position & 3)
                 : window   ? min(window, step->position + 1)
                            : step->position + 1;
        start = step->position + 1 - length;
    }
    const int h = blockIdx.y, i = threadIdx.x;
    for (int t = blockIdx.x; t < length; t += gridDim.x) {
        int slot = chosen ? (direct ? chosen[t] : chosen[t / 4] * 4 + t % 4) : (start + t) % capacity;
        const auto *k = kvRow(pages, slot, keyStride) + h * 256;
        float dot = reduce(q[h * 256 + i] * __bfloat162float(k[i]));
        if (!i)
            score[h * length + t] = bf(bf(dot) * 0.0625f);
    }
}
template <bool DraftRows = false>
__global__ static void probKV(float *scores, Hy4View sink, int length, const StepInput *step = nullptr,
                              int window = 0) {
    if constexpr (DraftRows) {
        const int row = blockIdx.y;
        scores += size_t(row) * gridDim.x * (length + gridDim.y - 1);
        length += row;
    }
    if (step)
        length = window < 0 ? 2045 + (step->position & 3)
                 : window   ? min(window, step->position + 1)
                            : step->position + 1;
    int h = blockIdx.x;
    float *p = scores + h * length;
    float bound = read(sink.data, sink.type, h);
    for (int i = threadIdx.x; i < length; i += blockDim.x)
        bound = fmaxf(bound, p[i]);
    // The sink can be negative. A zero lower bound still yields the same stable softmax.
    bound = reduce<true>(bound);
    float sum = threadIdx.x == 0 ? expf(read(sink.data, sink.type, h) - bound) : 0.f;
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        p[i] = expf(p[i] - bound);
        sum += p[i];
    }
    sum = reduce(sum);
    for (int i = threadIdx.x; i < length; i += blockDim.x)
        p[i] = bf(p[i] / sum);
}
template <bool DraftRows = false>
__global__ static void applyKV(const float *scores, const float *gate, float *out, int start, int length,
                               int heads, int cap, const int *chosen, bool direct, const StepInput *step,
                               int window, const KVPage *const *pages) {
    if constexpr (DraftRows) {
        const int row = blockIdx.y;
        scores += size_t(row) * heads * (length + gridDim.y - 1);
        gate += size_t(row) * heads * 256;
        out += size_t(row) * heads * 256;
        length += row;
    }
    if (step) {
        length = window < 0 ? 2045 + (step->position & 3)
                 : window   ? min(window, step->position + 1)
                            : step->position + 1;
        start = step->position + 1 - length;
    }
    int h = blockIdx.x, j = threadIdx.x;
    float sum = 0;
    for (int t = 0; t < length; t++) {
        int slot = chosen ? (direct ? chosen[t] : chosen[t / 4] * 4 + t % 4) : (start + t) % cap;
        sum = fmaf(scores[h * length + t],
                   __bfloat162float(kvRow<true>(pages, slot, heads * 256)[h * 256 + j]), sum);
    }
    out[h * 256 + j] = bf(bf(sum) * bf(sigmoid(gate[h * 256 + j])));
}
static void ReserveRecallRing(DraftMemory &draft, int device, int slots) {
    if (draft.recalled.dims.empty()) {
        draft.recalled.dataType = INT32;
        draft.recalled.UpdateUnitSize();
        draft.recalled.dataDevice = CUDA;
        draft.recalled.dataDeviceIds = {device};
        draft.recalled.Resize({slots, 512});
        draft.recalled.Allocate(false);
    }
}
static bool UsesPagedMla(const Weights &w, const Scratch &s, const std::string &prefix) {
    return s.mlaSupported && w.meta.at(prefix + "kv_b_proj.weight").dtype == "BF16";
}

static AttentionCache PrepareAttentionCache(Weights &w, Scratch &s, int layer, int end, Data &key,
                                            Data &value, int draftWidth, DraftMemory *draft) {
    const bool sliding = w.config["layer_types"][layer].string_value() == "sliding_attention";
    const int limit = w.config["max_position_embeddings"].int_value();
    const bool mla = UsesPagedMla(w, s, "model.layers." + std::to_string(layer) + ".self_attn.");
    // An entire MLA query tile appends before attention reads its windows.
    const int keep = sliding ? 512 + (mla ? std::max(15, draftWidth) : draftWidth) : limit;
    const int width = (sliding ? 64 : 32) * 256;
    const int keyWidth = (mla ? 64 : width) + (sliding ? 0 : 256);
    if (!mla)
        s.Get("attScores", 32 * 2048);
    AttentionCache result{
        ReservePages(s.kvPools, s.device, key, value, end, keep, keyWidth, mla ? 512 : width), nullptr};
    if (draft)
        result.draft = ReservePages(s.kvPools, s.device, draft->keys, draft->values, end, keep, 8192, 8192);
    if (!sliding && end > 2048) {
        if (draft)
            ReserveRecallRing(*draft, s.device, draftWidth + 1);
        // The indexer starts using these weights at position 2048. Materialize them before capture.
        const auto p = "model.layers." + std::to_string(layer) + ".self_attn.compressed_indexer.";
        w.View(p + "query_projection.weight", s.device);
        w.View(p + "head_weight_projection.weight", s.device);
        s.Get("indexQuery", 16 * 128);
        s.Get("indexRotaryQuery", 16 * 128);
        s.Get("indexHeadWeights", 16);
        s.Get("indexGroupScores", ((size_t(limit) + 127) / 128) * 32);
        s.Get("indexChosen", 512);
    }
    return result;
}
const AttentionCache *UploadAttentionCaches(Scratch &s, const std::vector<AttentionCache> &caches) {
    // Graph nodes index this stable array by request and layer, including when requests are replaced.
    const size_t bytes = caches.size() * sizeof(AttentionCache);
    auto dst = reinterpret_cast<AttentionCache *>(s.Get("graphAttentionCaches", bytes / sizeof(float)));
    CopyToGPU(dst, caches.data(), bytes);
    return dst;
}
__global__ static void saveChosenRing(int *ring, const int *chosen, const StepInput *step, int slots) {
    int i = blockIdx.x * 256 + threadIdx.x;
    ring[(step->position % slots) * 512 + i] = chosen[i];
}
static void SaveDraftAttentionKV(Weights &w, Scratch &s, DraftMemory *draft, int draftLayer, int draftWidth,
                                 const float *kv, int position, int keep, float theta, const int *chosen,
                                 const AttentionCache *cache) {
    if (!draft)
        return;
    const auto prefix = "model.dfly_layers." + std::to_string(draftLayer) + ".self_attn.";
    auto projected = s.Get("draftHistoricalKV", 32 * 448);
    Linear(w, s, prefix + "kv_b_proj.weight", kv, projected);
    packKV<<<32, 256>>>(projected, kv + 512, 32, position, position % keep, theta, 8192, s.step, keep,
                        &cache->draft);
    if (chosen) {
        auto dst = static_cast<int *>(draft->recalled.cudaData) + size_t(position % (draftWidth + 1)) * 512;
        if (s.step)
            saveChosenRing<<<2, 256>>>(static_cast<int *>(draft->recalled.cudaData), chosen, s.step,
                                       draftWidth + 1);
        else
            Check(cudaMemcpyAsync(dst, chosen, 512 * sizeof(int), cudaMemcpyDeviceToDevice,
                                  cudaStreamPerThread));
    }
}
static void ExpandedAttentionTile(Weights &w, Scratch &s, const std::string &p, int heads,
                                  const AttentionRow *rows, int n, const float *x, const float *qr,
                                  const float *q, const float *kv, const float *gate, float *mix) {
    const bool sliding = heads == 64;
    const int width = heads * 256, keyStride = width + (sliding ? 0 : 256);
    const float theta = sliding ? 10000.f : 10000000.f;
    const float *latent = kv;
    if (n > 1) {
        auto packed = s.Get("batchKvLatent", n * 512);
        for (int i = 0; i < n; ++i)
            CopyStageCaptured(packed + i * 512, kv + i * 576, 512);
        latent = packed;
    }
    auto expanded = s.Get("batchExpanded", n * heads * 448);
    auto rq = s.Get("rotQ", width);
    auto scores = s.Get("attScores", 32 * 2048);
    LinearBatch(w, s, p + "kv_b_proj.weight", latent, expanded, n);
    for (int i = 0; i < n; ++i) {
        const auto &r = rows[i];
        s.step = r.step;
        const auto pages = &r.cache->target;
        const int pos = r.position;
        const int keep = sliding ? 512 + r.draftWidth : w.config["max_position_embeddings"].int_value();
        packKV<<<heads, 256>>>(expanded + i * heads * 448, kv + i * 576 + 512, heads, pos, pos % keep, theta,
                               keyStride, r.step, keep, pages);
        rotQ<<<heads, 256>>>(q + i * width, rq, width, pos, theta, r.step);
        const int *chosen =
            sliding ? nullptr : IndexPool(w, s, p, x + i * 3072, qr + i * 1024, keyStride, pos, pages);
        SaveDraftAttentionKV(w, s, r.draft, r.draftLayer, r.draftWidth, kv + i * 576, pos, keep, theta,
                             chosen, r.cache);
        const int length = chosen ? 2045 + (pos & 3) : (sliding ? std::min(pos + 1, 512) : pos + 1);
        const int start = pos + 1 - length;
        const int window = chosen ? -1 : (sliding ? 512 : 0);
        scoreKV<<<dim3(r.step ? KVPageTokens : length, heads), 256>>>(
            rq, scores, start, length, heads, keep, keyStride, chosen, false, r.step, window, pages);
        probKV<<<heads, 256>>>(scores, w.View(p + "learnable_sink_param", s.device), length, r.step, window);
        applyKV<<<heads, 256>>>(scores, gate + i * width, mix + i * width, start, length, heads, keep, chosen,
                                false, r.step, window, pages);
    }
}

#include "hy4-flash-mla.cuh"

const AttentionCache *PrepareAttentionCaches(Weights &w, Scratch &s, int firstLayer, int endLayer,
                                             int blockCount, const std::vector<int> &lengths,
                                             const std::vector<std::pair<Data *, Data *>> &past,
                                             const std::vector<int> &providers,
                                             const std::vector<DraftRequest *> &states, int draftWidth) {
    Select(s.device);
    PreparePagedMla(s, std::accumulate(lengths.begin(), lengths.end(), 0));
    std::vector<AttentionCache> caches;
    caches.reserve((endLayer - firstLayer) * lengths.size());
    for (int layer = firstLayer; layer < endLayer; ++layer) {
        auto it = std::find(providers.begin(), providers.end(), layer);
        for (size_t i = 0; i < lengths.size(); ++i) {
            auto &kv = past[i * blockCount + layer];
            int end = int(past[i * blockCount].first->cpuIntDatas.size()) + lengths[i];
            auto *draft =
                states[i] && it != providers.end() ? &states[i]->providers[it - providers.begin()] : nullptr;
            caches.push_back(PrepareAttentionCache(w, s, layer, end, *kv.first, *kv.second,
                                                   states[i] ? draftWidth : 0, draft));
        }
    }
    return UploadAttentionCaches(s, caches);
}

struct HashParams {
    int64_t factors[3], sizes[16], offsets[16];
    int pad, mapSize;
};
static HashParams HashConfig(Weights &w, int layer, const std::string &p) {
    // NumPy PCG64 draws for the published vocabulary and engram_seed=0.
    // Check the release geometry before using its frozen hash family.
    static const std::map<int, std::array<int64_t, 3>> factors = {
        {1, {81853575799317LL, 39690085502883LL, 24229915018165LL}},
        {11, {19457948045719LL, 44729556510419LL, 84779352922315LL}},
        {21, {72974230057307LL, 21530602121221LL, 55998057243097LL}},
        {31, {29128509562787LL, 16745155128523LL, 22474588302733LL}}};
    if (w.config["engram_seed"].int_value() != 0)
        throw std::runtime_error("Unsupported Engram hash seed");
    HashParams m{};
    auto &v = factors.at(layer);
    std::copy(v.begin(), v.end(), m.factors);
    auto &orders = w.config["engram_layer_head_vocab_sizes"][std::to_string(layer + 1)];
    int64_t offset = 0;
    for (int j = 0; j < 16; j++) {
        m.sizes[j] = orders[j / 8][j % 8].int_value();
        if (m.sizes[j] <= 0)
            throw std::runtime_error("Invalid Engram head size");
        m.offsets[j] = offset;
        offset += m.sizes[j];
    }
    m.mapSize = w.meta.at(p + "lookup_table").dims[0];
    m.pad = w.config["engram_pad_id"].int_value();
    if (offset != w.meta.at(p + "multi_head_embedding.embedding.weight").dims[0] || m.pad >= m.mapSize)
        throw std::runtime_error("Invalid Engram table geometry");
    return m;
}
__host__ __device__ static int64_t rowId(int head, int a, int b, int c, HashParams m, const int *map) {
    auto ma = map[a >= 0 && a < m.mapSize ? a : m.pad], mb = map[b >= 0 && b < m.mapSize ? b : m.pad];
    int64_t mixed = ma * m.factors[0] ^ mb * m.factors[1];
    if (head >= 8)
        mixed ^= map[c >= 0 && c < m.mapSize ? c : m.pad] * m.factors[2];
    return m.offsets[head] + mixed % m.sizes[head];
}
std::array<int64_t, 16> EngramRows(Weights &w, int layer, const std::array<int, 3> &ids) {
    std::string p = "model.layers." + std::to_string(layer) + ".engram_layer.";
    auto m = HashConfig(w, layer, p);
    auto map = (const int *)w.Host(p + "lookup_table");
    std::array<int64_t, 16> result;
    for (int j = 0; j < 16; j++)
        result[j] = rowId(j, ids[0], ids[1], ids[2], m, map);
    return result;
}
struct HostEngramInput {
    HashParams config;
    const uint16_t *table = nullptr;
    const int *lookup = nullptr;
    uint16_t *staging = nullptr;
    const uint16_t *mapped = nullptr;
    ~HostEngramInput() {
        if (staging)
            cudaFreeHost(staging);
    }
};
void PrepareEngramInput(Weights &w, Scratch &s, int layer, int row, const std::array<int, 3> &ids) {
    auto &entry = s.engramInputs[{layer, row}];
    if (!entry) {
        auto input = std::make_shared<HostEngramInput>();
        const auto p = "model.layers." + std::to_string(layer) + ".engram_layer.";
        input->config = HashConfig(w, layer, p);
        input->table = static_cast<const uint16_t *>(w.Host(p + "multi_head_embedding.embedding.weight"));
        input->lookup = static_cast<const int *>(w.Host(p + "lookup_table"));
        Check(cudaHostAlloc(&input->staging, 16 * 96 * sizeof(uint16_t), cudaHostAllocPortable | cudaHostAllocMapped));
        cudaPointerAttributes attr{};
        Check(cudaPointerGetAttributes(&attr, input->staging));
        input->mapped = static_cast<const uint16_t *>(attr.devicePointer);
        entry = std::move(input);
    }
    // Publish this row with the other replay inputs. The preceding forward has
    // synchronized before this request/layer's pinned storage is reused.
    auto &in = *static_cast<HostEngramInput *>(entry.get());
    for (int head = 0; head < 16; ++head) {
        auto k = rowId(head, ids[0], ids[1], ids[2], in.config, in.lookup);
        std::memcpy(in.staging + head * 96, in.table + k * 96, 96 * sizeof(uint16_t));
    }
}
void RunPreparedEngram(Weights &w, Scratch &s, int layer, int row, const std::array<int, 3> &ids,
                       float *streams) {
    auto *input = static_cast<HostEngramInput *>(s.engramInputs.at({layer, row}).get());
    Engram(w, s, layer, ids, streams, input->mapped ? input->mapped : input->staging,
           input->mapped ? EngramRowStorage::Mapped : EngramRowStorage::Pinned);
}
__global__ static void gather(const __nv_bfloat16 *table, const int *map, HashParams m, int a, int b, int c,
                              bool staged, float *output, const StepInput *step) {
    if (step) {
        a = step->token;
        b = step->previous;
        c = step->older;
    }
    int h = blockIdx.x, i = threadIdx.x;
    int64_t row = staged ? h : rowId(h, a, b, c, m, map);
    float v = i < 96 ? __bfloat162float(table[row * 96 + i]) : 0;
    float scale = bf(fmaxf(reduce<true>(fabsf(v)), 1.e-12f) / 448.f);
    if (i < 96) {
        float quant = float(__nv_fp8_e4m3(bf(v / scale)));
        float restored = bf(quant * scale);
        // The reference uses a straight-through estimator even in eval mode.
        output[h * 96 + i] = bf(v + bf(restored - v));
    }
}
__global__ static void engramGate(const float *a, const float *b, float *out) {
    float dot = 0;
    for (int i = threadIdx.x; i < 3072; i += blockDim.x)
        dot += bf(a[i] * b[i]);
    dot = bf(bf(reduce(dot)) * rsqrtf(3072.f));
    if (threadIdx.x == 0)
        *out = bf(sigmoid(copysignf(bf(sqrtf(fmaxf(fabsf(dot), 1.e-6f))), dot)));
}
__global__ static void engramAdd(float *h, const float *signal, Hy4View gamma, Hy4View scale,
                                 const float *gate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = bf(signal[i] * read(gamma.data, gamma.type, i));
    h[i] = bf(h[i] + bf(bf(*gate * v) + bf(v * read(scale.data, scale.type, i))));
}
void Engram(Weights &w, Scratch &s, int layer, const std::array<int, 3> &ids, float *h,
            const uint16_t *rows, EngramRowStorage storage) {
    if (ids[0] == w.config["engram_pad_id"].int_value() || ids[0] == 120120 || ids[0] == 120124)
        return;
    auto p = "model.layers." + std::to_string(layer) + ".engram_layer.";
    auto m = HashConfig(w, layer, p);
    auto emb = s.Get("engramVectors", 1536);
    const int *map = nullptr;
    const __nv_bfloat16 *table;
    if (storage == EngramRowStorage::Mapped) {
        table = reinterpret_cast<const __nv_bfloat16 *>(rows);
    } else if (rows) {
        auto *staged = s.Get("engramStaged", (16 * 96 + 1) / 2);
        if (storage == EngramRowStorage::Pinned)
            Check(cudaMemcpyAsync(staged, rows, 16 * 96 * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  cudaStreamPerThread));
        else
            CopyToGPU(staged, rows, 16 * 96 * sizeof(uint16_t));
        table = (const __nv_bfloat16 *)staged;
    } else {
        map = (const int *)w.Device(p + "lookup_table", s.device);
        table = (const __nv_bfloat16 *)w.Device(p + "multi_head_embedding.embedding.weight", s.device);
    }
    gather<<<16, 128>>>(table, map, m, ids[0], ids[1], ids[2], rows != nullptr, emb, s.step);
    EngramNormWeights norms;
    for (int j = 0; j < 16; j++)
        norms.head[j] = w.View(p + "lookup_per_head_norms." + std::to_string(j) + ".weight", s.device);
    normEngramHeads<<<16, 128>>>(emb, norms);
    auto signal = s.Get("engramValue", 3072), gates = s.Get("engramGates", 4);
    Linear(w, s, p + "value_proj.weight", emb, signal, true, nullptr, 1, false, p + "value_proj.bias");
    norm<<<1, 256>>>(signal, {}, signal, 3072, 1.1920929e-7f, false);
    for (int j = 0; j < 4; j++) {
        auto id = std::to_string(j);
        float *k = s.Get("engramKey", 3072), *q = s.Get("engramQuery", 3072);
        Linear(w, s, p + "key_projs." + id + ".weight", emb, k, true, nullptr, 1, false,
               p + "key_projs." + id + ".bias");
        norm<<<1, 256>>>(k, w.View(p + "norm_key." + id + ".weight", s.device), k, 3072, 1.1920929e-7f,
                         false);
        norm<<<1, 256>>>(h + 3072 * j, w.View(p + "norm_query." + id + ".weight", s.device), q, 3072,
                         1.1920929e-7f, false);
        engramGate<<<1, 256>>>(k, q, gates + j);
        engramAdd<<<12, 256>>>(h + 3072 * j, signal, w.View(p + "unified_pre_gamma", s.device),
                               w.View(p + "unified_per_hc_scales." + id, s.device), gates + j);
    }
}
__global__ static void saveRing(float *ring, const float *hidden, const StepInput *step, int slots) {
    int i = blockIdx.x * 256 + threadIdx.x;
    ring[(step->position % slots) * 3072 + i] = hidden[i];
}
void SaveDraftHidden(Data &ring, int device, int position, int width, const float *hidden,
                     const StepInput *step) {
    if (ring.dims.empty()) {
        ring.dataType = FLOAT32;
        ring.UpdateUnitSize();
        ring.dataDevice = CUDA;
        ring.dataDeviceIds = {device};
        ring.Resize({width + 1, 3072});
        ring.Allocate(false);
    }
    if (step) {
        saveRing<<<12, 256>>>(static_cast<float *>(ring.cudaData), hidden, step, width + 1);
        return;
    }
    auto dst = static_cast<float *>(ring.cudaData) + size_t(position % (width + 1)) * 3072;
    Check(cudaMemcpyAsync(dst, hidden, 3072 * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
}
__global__ static void draftEmbedding(Hy4View w, int constant, const int *selected, float *output,
                                      int mask = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.cols) {
        int token = blockIdx.y ? mask : (selected ? *selected : constant);
        output[size_t(blockIdx.y) * w.cols + i] = read(w.data, w.type, size_t(token) * w.cols + i);
    }
}
__global__ static void fusionResidual(float *anchor, const float *fused, Hy4View gate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    anchor[i] = bf(anchor[i] + bf(read(gate.data, gate.type, i) * fused[i]));
}
void DraftEmbed(Weights &w, Scratch &s, int anchor, int width, int mask, const float *targetHidden,
                float *block) {
    auto table = w.View("model.embed_tokens.weight", s.device);
    draftEmbedding<<<dim3(12, width), 256>>>(table, anchor, nullptr, block, mask);
    const std::string p = "model.dfly_layers.mtp_fusion.";
    auto concat = s.Get("draftFusionPair", 6144), fused = s.Get("draftFusionVector", 3072);
    Norm(w, s, p + "enorm.weight", block, concat, 3072);
    Norm(w, s, p + "hnorm.weight", targetHidden, concat + 3072, 3072);
    Linear(w, s, p + "eh_proj.weight", concat, fused);
    fusionResidual<<<12, 256>>>(block, fused, w.View(p + "gate", s.device));
}
__global__ static void draftSlots(int *slots, int history, int base, int width, const int *previous) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < history)
        slots[i] = previous ? previous[i / 4] * 4 + i % 4 : i;
    else if (i < history + width)
        slots[i] = base + i - history;
}
void DraftLayer(Weights &w, Scratch &s, int layer, int base, int width, DraftMemory &memory, float *block,
                const KVPage *const *pages) {
    const std::string p = "model.dfly_layers." + std::to_string(layer) + ".", a = p + "self_attn.";
    const int limit = w.config["max_position_embeddings"].int_value();
    auto x = s.Get("draftInput", width * 3072), qr = s.Get("draftQRes", width * 1024),
         q = s.Get("draftQ", width * 8192), rotated = s.Get("draftRotated", width * 8192);
    auto kv = s.Get("draftKV", width * 576), latent = s.Get("draftKVLatent", width * 512),
         expanded = s.Get("draftKVExpand", width * 32 * 448);
    auto gate = s.Get("draftAttGate", width * 8192), mix = s.Get("draftMix", width * 8192),
         branch = s.Get("draftBranch", width * 3072);
    const int history = base <= 2048 ? base : 511 * 4 + (base - 1) % 4 + 1;
    // Recalled history is bounded. Keep draft scratch stable across target verify replays.
    const int capacity = std::min(limit, 2048 + width);
    auto scores = s.Get("draftScores", size_t(width) * 32 * capacity);
    auto slots = reinterpret_cast<int *>(s.Get("draftSlots", capacity));
    auto previous = base <= 2048 ? nullptr
                                 : static_cast<const int *>(memory.recalled.cudaData) +
                                       size_t((base - 1) % memory.recalled.dims[0]) * 512;
    NormRows(w, s, p + "input_layernorm.weight", block, x, width, 3072, 3072, 3072);
    LinearBatch(w, s, a + "q_a_proj.weight", x, qr, width);
    NormBatch(w, s, a + "q_a_layernorm.weight", qr, width, 1024);
    LinearBatch(w, s, a + "q_b_proj.weight", qr, q, width);
    LinearBatch(w, s, a + "kv_a_proj_with_mqa.weight", x, kv, width);
    NormRows(w, s, a + "kv_a_layernorm.weight", kv, latent, width, 512, 576, 512);
    LinearBatch(w, s, a + "kv_b_proj.weight", latent, expanded, width);
    LinearBatch(w, s, a + "linear_gate.weight", x, gate, width);
    const dim3 headRows(32, width);
    packKV<true>
        <<<headRows, 256>>>(expanded, kv + 512, 32, base, base, 10000000.f, 8192, nullptr, limit, pages);
    rotQ<true><<<headRows, 256>>>(q, rotated, 8192, base, 10000000.f);

    // All draft rows share the recalled prefix. Row r sees only the first r + 1 new KV slots.
    const int longest = history + width, firstLength = history + 1;
    draftSlots<<<(longest + 255) / 256, 256>>>(slots, history, base, width, previous);
    scoreKV<true><<<dim3(longest, 32, width), 256>>>(rotated, scores, 0, firstLength, 32, limit, 8192, slots,
                                                     true, nullptr, 0, pages);
    probKV<true><<<headRows, 256>>>(scores, w.View(a + "learnable_sink_param", s.device), firstLength);
    applyKV<true>
        <<<headRows, 256>>>(scores, gate, mix, 0, firstLength, 32, limit, slots, true, nullptr, 0, pages);
    LinearBatch(w, s, a + "o_proj.weight", mix, branch, width);
    NormBatch(w, s, a + "post_proj_norm.weight", branch, width, 3072);
    add<<<12 * width, 256>>>(block, branch, width * 3072);
    NormRows(w, s, p + "post_attention_layernorm.weight", block, x, width, 3072, 3072, 3072);
    NormalizedDenseMlpRows<true>(w, s, p + "mlp.", x, branch, width);
    add<<<12 * width, 256>>>(block, branch, width * 3072);
}
std::vector<int> DraftReadout(Weights &w, Scratch &s, int anchor, int width, float *block,
                              const GenerationConfig &cfg, DraftProposal proposal) {
    const std::string p = "model.dfly_layers.hidden_correction.";
    auto table = w.View("model.embed_tokens.weight", s.device);
    auto embed = s.Get("draftPreviousEmbed", 3072), pair = s.Get("draftCorrectionPair", 6144);
    auto g = s.Get("draftCorrectionGate", 3072), up = s.Get("draftCorrectionUp", 3072);
    auto activated = s.Get("draftCorrectionAct", 3072), delta = s.Get("draftCorrectionDelta", 3072);
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    auto logits = s.Get("draftHeadLogits", vocab);
    auto picks = proposal.tokens ? proposal.tokens : reinterpret_cast<int *>(s.Get("draftPicks", width - 1));
    for (int i = 0; i < width - 1; ++i) {
        auto h = block + size_t(i + 1) * 3072;
        Norm(w, s, "model.dfly_layers.final_norm.weight", h, h, 3072);
        draftEmbedding<<<12, 256>>>(table, anchor, i ? picks + i - 1 : nullptr, embed);
        Norm(w, s, p + "hidden_norm.weight", h, pair, 3072);
        Norm(w, s, p + "embed_norm.weight", embed, pair + 3072, 3072);
        DraftGateUp(w, s, p, pair, g, up, 1);
        activate<<<12, 256>>>(g, up, activated, 3072, false);
        Linear(w, s, p + "down_proj.weight", activated, delta);
        add<<<12, 256>>>(h, delta, 3072);
        Linear(w, s, "lm_head.weight", h, logits);
        if (proposal.tokens) {
            AssertInFastLLM(FastllmCudaMtpSampleDraftLogits(logits, proposal.logits + size_t(i) * vocab,
                                                            proposal.logsumexp + i, picks + i, nullptr,
                                                            &cfg.temperature, 1, vocab),
                            "HY4 DFly2 proposal sampling failed.");
        } else {
            AssertInFastLLM(FastllmCudaGreedySampling(logits, picks + i, 1, vocab),
                            "HY4 DFly2 greedy proposal failed.");
        }
    }
    std::vector<int> result(width - 1);
    CopyToHost(result.data(), picks, result.size() * sizeof(int));
    return result;
}

DraftProposal PrepareProposal(Scratch &s, int request, int batch, int drafts, int vocab) {
    const size_t rows = size_t(batch) * drafts;
    const size_t offset = size_t(request) * drafts;
    return {s.Get("draftSampleLogits", rows * vocab) + offset * vocab, s.Get("draftLogsumexp", rows) + offset,
            reinterpret_cast<int *>(s.Get("draftSampleIds", rows)) + offset};
}

__global__ static void prepareSamplingScores(const float *logits, float *scores, int *ids, int vocab,
                                             float invTemperature) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab) {
        scores[i] = logits[i] * invTemperature;
        ids[i] = i;
    }
}

// Match LLMSampling's top-k before top-p, retaining precisely the first
// crossing token. Stable radix sort breaks equal scores by ascending token ID.
__global__ static void sortedProbabilities(const float *scores, const int *ids, float *probs, int k,
                                           float topP) {
    constexpr int threads = 256;
    using Reduce = cub::BlockReduce<double, threads>;
    using Scan = cub::BlockScan<double, threads>;
    __shared__ union {
        typename Reduce::TempStorage reduce;
        typename Scan::TempStorage scan;
    } tmp;
    __shared__ double total, previous;
    __shared__ int keep;
    double mass = 0;
    const float maximum = scores[0];
    for (int i = threadIdx.x; i < k; i += threads)
        mass += expf(scores[i] - maximum);
    double sum = Reduce(tmp.reduce).Sum(mass);
    if (threadIdx.x == 0) {
        total = sum;
        keep = k;
        previous = 0;
    }
    __syncthreads();
    if (topP < 1) {
        const double threshold = total * topP;
        for (int base = 0; base < k; base += threads) {
            const int i = base + threadIdx.x;
            const double value = i < k ? expf(scores[i] - maximum) : 0;
            double prefix, blockSum;
            Scan(tmp.scan).InclusiveSum(value, prefix, blockSum);
            const double cumulative = previous + prefix;
            __syncthreads();
            if (i < k && cumulative > threshold)
                atomicMin(&keep, i + 1);
            __syncthreads();
            if (keep < k || (base + threads >= k)) {
                if (i + 1 == keep)
                    total = cumulative;
                break;
            }
            if (!threadIdx.x)
                previous += blockSum;
            __syncthreads();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < keep; i += threads)
        probs[ids[i]] = expf(scores[i] - maximum) / total;
}

float *TargetProbabilities(Scratch &s, float *target, int width, int vocab, const GenerationConfig &cfg) {
    auto probabilities = s.Get("verifyProbabilities", size_t(width) * vocab);
    auto scores = s.Get("verifyScores", vocab), sorted = s.Get("verifySortedScores", vocab);
    auto ids = reinterpret_cast<int *>(s.Get("verifyIds", vocab));
    auto order = reinterpret_cast<int *>(s.Get("verifyOrder", vocab));
    size_t bytes = 0;
    Check(cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, scores, sorted, ids, order, vocab, 0, 32,
                                                    cudaStreamPerThread));
    auto temp = s.Get("verifySort", (bytes + sizeof(float) - 1) / sizeof(float));
    Check(cudaMemsetAsync(probabilities, 0, size_t(width) * vocab * sizeof(float), cudaStreamPerThread));
    const int k = std::clamp(cfg.top_k, 1, vocab);
    for (int row = 0; row < width; ++row) {
        prepareSamplingScores<<<(vocab + 255) / 256, 256>>>(target + size_t(row) * vocab, scores, ids, vocab,
                                                            1.f / cfg.temperature);
        Check(cub::DeviceRadixSort::SortPairsDescending(temp, bytes, scores, sorted, ids, order, vocab, 0, 32,
                                                        cudaStreamPerThread));
        sortedProbabilities<<<1, 256>>>(sorted, order, probabilities + size_t(row) * vocab, k, cfg.top_p);
    }
    Check(cudaGetLastError());
    return probabilities;
}

std::vector<int> VerifyDraft(Scratch &s, float *target, int width, int vocab, const GenerationConfig &cfg,
                             DraftProposal proposal, const int *candidates) {
    Select(s.device);
    std::vector<int> tokens(width);
    if (!proposal.tokens) {
        auto picks = reinterpret_cast<int *>(s.Get("verifyGreedyIds", width));
        AssertInFastLLM(FastllmCudaGreedySampling(target, picks, width, vocab),
                        "HY4 DFly2 greedy verification failed.");
        CopyToHost(tokens.data(), picks, width * sizeof(int));
        const auto mismatch = std::mismatch(tokens.begin(), tokens.end() - 1, candidates);
        tokens.resize(mismatch.first - tokens.begin() + 1);
        return tokens;
    }
    auto probabilities = TargetProbabilities(s, target, width, vocab, cfg);
    int offset = 0;
    while (offset < width - 1) {
        // The shared kernel handles up to eight drafts. Continue after a fully
        // accepted segment; only the last segment publishes its bonus token.
        const int n = std::min(8, width - 1 - offset);
        int accepted = 0;
        AssertInFastLLM(FastllmCudaMtpRejectionFromProbs(
                            probabilities + size_t(offset) * vocab, proposal.logits + size_t(offset) * vocab,
                            proposal.logsumexp + offset, proposal.tokens + offset, tokens.data() + offset,
                            &accepted, 1, n, vocab),
                        "HY4 DFly2 rejection sampling failed.");
        if (accepted < n) {
            tokens.resize(offset + accepted + 1);
            break;
        }
        offset += n;
    }
    return tokens;
}

} // namespace fastllm::hy4
