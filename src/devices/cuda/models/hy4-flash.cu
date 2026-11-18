#include "../../../models/hy4_flash_ops.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/block/block_radix_sort.cuh>
#include <cmath>
#include <stdexcept>
#include "utils/hybrid_moe.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"

namespace fastllm::hy4 {
static void CheckAt(cudaError_t err, int line, const char *operation) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("HY4 CUDA line ") + std::to_string(line) + " " + operation +
                                 ": " + cudaGetErrorString(err));
}
#define Check(operation) CheckAt((operation), __LINE__, #operation)
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
void AdvanceKV(Data &data, int end) {
    auto dims = data.dims;
    dims[1] = std::min(data.expansionDims[1], end);
    data.Resize(dims);
}
bool ReplayGpu(GpuGraphs &graphs, const std::vector<uintptr_t> &key, int kind, int rows) {
    auto found = graphs.replays.find(key);
    if (found == graphs.replays.end())
        return false;
    Check(cudaGraphLaunch(static_cast<cudaGraphExec_t>(found->second.executable.get()), cudaStreamPerThread));
    if (std::getenv("FASTLLM_HY4_BATCH_TRACE"))
        fprintf(stderr, "[HY4 GPU replay] verify=%d rows=%d launch=%zu\n", kind, rows, ++graphs.launches);
    return true;
}
struct HybridExecutable {
    std::shared_ptr<DiskMoeV41Graph> disk;
    HybridMoeGraph graph;
    bool Launch() {
        return disk ? disk->Launch([this] { return graph.Launch(); }) : graph.Launch();
    }
};
bool ReplayHybrid(GpuGraphs &graphs, const std::vector<uintptr_t> &key, int kind, int rows) {
    auto found = graphs.replays.find(key);
    if (found == graphs.replays.end())
        return false;
    if (!static_cast<HybridExecutable *>(found->second.executable.get())->Launch())
        throw std::runtime_error("HY4 native MoE graph replay failed");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    if (std::getenv("FASTLLM_HY4_BATCH_TRACE"))
        fprintf(stderr, "[HY4 hybrid replay] verify=%d rows=%d launch=%zu\n", kind, rows, ++graphs.launches);
    return true;
}
void GraphCapture::Start(int device) {
    if (!graphs)
        return;
    Select(device);
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    activeDevice = device;
}
void GraphCapture::Close() {
    if (activeDevice < 0)
        return;
    Select(activeDevice);
    activeDevice = -1;
    cudaGraph_t graph = nullptr;
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    children.push_back(graph);
    hybridChildren.push_back(graph);
}
GraphCapture::~GraphCapture() {
    if (activeDevice >= 0) {
        cudaSetDevice(activeDevice);
        cudaGraph_t unfinished = nullptr;
        cudaStreamEndCapture(cudaStreamPerThread, &unfinished);
        if (unfinished)
            cudaGraphDestroy(unfinished);
    }
    for (void *h : handles)
        FastllmCudaReleaseNativeMoeFullGraph(h);
    for (void *g : children)
        cudaGraphDestroy(static_cast<cudaGraph_t>(g));
}
void GraphCapture::AddNative(void *child, void *handle) {
    hybridChildren.push_back(child);
    handles.push_back(handle);
}
void GraphCapture::FinishHybrid(int kind, int rows) {
    if (!graphs)
        return;
    Close();
    auto compiled = std::make_shared<HybridExecutable>();
    compiled->disk = std::move(disk);
    const size_t count = compiled->disk ? (hybridChildren.size() - 1) / 2 : handles.size();
    if (!compiled->graph.Compose(hybridChildren, std::move(handles)) || !compiled->Launch())
        throw std::runtime_error("HY4-Flash hybrid graph composition or launch failed");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    if (!key.empty()) {
        if (graphs->replays.size() >= 16)
            graphs->replays.clear();
        graphs->replays[key] = {compiled, std::move(outputs)};
    }
    if (std::getenv("FASTLLM_HY4_BATCH_TRACE"))
        fprintf(stderr,
                "[HY4 hybrid full graph] verify=%d rows=%d children=%zu expert_layers=%zu launch=%zu\n", kind,
                rows, hybridChildren.size(), count, ++graphs->launches);
}
void GraphCapture::Finish(int kind, int rows) {
    if (!graphs)
        return;
    Close();
    cudaGraph_t parent = nullptr;
    Check(cudaGraphCreate(&parent, 0));
    struct Owner {
        cudaGraph_t graph;
        ~Owner() {
            cudaGraphDestroy(graph);
        }
    } owner{parent};
    cudaGraphNode_t prior = nullptr;
    size_t nodes = 0;
    for (void *c : children) {
        cudaGraphNode_t next = nullptr;
        auto child = static_cast<cudaGraph_t>(c);
        size_t count = 0;
        Check(cudaGraphGetNodes(child, nullptr, &count));
        nodes += count;
        Check(cudaGraphAddChildGraphNode(&next, parent, prior ? &prior : nullptr, prior ? 1 : 0, child));
        prior = next;
    }
    cudaGraphExec_t executable = nullptr;
    Check(cudaGraphInstantiate(&executable, parent, nullptr, nullptr, 0));
    auto ownerExec = std::shared_ptr<void>(
        executable, [](void *p) { cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(p)); });
    if (graphs->replays.size() >= 16)
        graphs->replays.clear();
    graphs->replays[key] = {std::move(ownerExec), {}};
    Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    // Consumers can use the resulting CUDA tensors immediately on this stream.
    if (std::getenv("FASTLLM_HY4_BATCH_TRACE"))
        fprintf(stderr, "[HY4 full graph] verify=%d rows=%d stages=%zu nodes=%zu capture=1 launch=%zu\n",
                kind, rows, children.size(), nodes, ++graphs->launches);
}

bool EnableGraphPeers(const std::vector<int> &devices) {
    for (size_t i = 1; i < devices.size(); ++i) {
        Select(devices[i]);
        int supported = 0;
        Check(cudaDeviceCanAccessPeer(&supported, devices[i], devices[i - 1]));
        if (!supported)
            return false;
        auto err = cudaDeviceEnablePeerAccess(devices[i - 1], 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled)
            cudaGetLastError();
        else
            Check(err);
    }
    return true;
}
__global__ static void copyPeerStreams(float *dst, const float *src, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = src[i];
}
void CopyStageCaptured(float *dst, int dd, const float *src, int sd, size_t count) {
    copyPeerStreams<<<(count + 255) / 256, 256>>>(dst, src, count);
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
template <bool Maximum = false> __device__ static float reduce(float x) {
    __shared__ float warp[32];
    for (int delta = 16; delta; delta /= 2) {
        float v = __shfl_down_sync(~0U, x, delta);
        x = Maximum ? fmaxf(x, v) : x + v;
    }
    if ((threadIdx.x & 31) == 0)
        warp[threadIdx.x >> 5] = x;
    __syncthreads();
    if (threadIdx.x < 32) {
        x = threadIdx.x < (blockDim.x + 31) / 32 ? warp[threadIdx.x] : 0.f;
        for (int delta = 16; delta; delta /= 2) {
            float v = __shfl_down_sync(~0U, x, delta);
            x = Maximum ? fmaxf(x, v) : x + v;
        }
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
template <int Type, bool Scaled>
__global__ static void warpGemv(Hy4View w, const float *input, float *out, const int *ids, int routesPerInput,
                                bool roundOutput, Hy4View bias) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5),
              slot = blockIdx.y;
    if (row >= w.rows)
        return;
    const int expert = ids ? ids[slot] : 0;
    const size_t k0 = (size_t(expert) * w.rows + row) * w.cols;
    if (routesPerInput)
        input += size_t(slot / routesPerInput) * w.cols;
    const int blocks = (w.cols + 127) / 128;
    const size_t plane = (size_t(expert) * ((w.rows + 127) / 128) + row / 128) * blocks;
    float sum = 0;
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
        if constexpr (Scaled)
            sum = fmaf(dot, read(w.scale, w.scaleType, plane + col / 128), sum);
        else
            sum += dot;
    }
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(~0U, sum, delta);
    if (!lane) {
        sum += bias.data ? read(bias.data, bias.type, row) : 0.f;
        out[size_t(slot) * w.rows + row] = roundOutput ? bf(sum) : sum;
    }
}
// The iHC mixing matrix has only four or eight rows, each 12,288 elements
// wide. Split K so the GPU can read it with many independent thread blocks.
__global__ static void partialMix(Hy4View w, const float *input, float *partial, int tiles) {
    int row = blockIdx.y, tile = blockIdx.x, col = tile * 512 + threadIdx.x;
    const float *weight = static_cast<const float *>(w.data) + size_t(row) * w.cols;
    float sum = 0;
    for (int i = col; i < min((tile + 1) * 512, w.cols); i += blockDim.x)
        sum = fmaf(weight[i], input[i], sum);
    sum = reduce(sum);
    if (!threadIdx.x)
        partial[row * tiles + tile] = sum;
}
__global__ static void finishMix(const float *partial, float *out, int tiles) {
    int row = blockIdx.x, lane = threadIdx.x;
    float sum = lane < tiles ? partial[row * tiles + lane] : 0;
    for (int delta = 16; delta; delta /= 2)
        sum += __shfl_down_sync(~0u, sum, delta);
    if (!lane)
        out[row] = sum;
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
    if (v.type == 1 && v.rows <= 8 && v.cols == 12288 && !ids && !roundOutput && !bias.data && slots == 1) {
        const int tiles = v.cols / 512;
        auto partial = s.Get("hcPartials", v.rows * tiles);
        partialMix<<<dim3(tiles, v.rows), 128>>>(v, x, partial, tiles);
        finishMix<<<v.rows, 32>>>(partial, y, tiles);
    } else if (v.cols % 4 == 0 && (v.type != 2 || v.scale)) {
        dim3 grid((v.rows + 3) / 4, slots);
        if (v.type == 2)
            warpGemv<2, true><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
        else if (v.type == 0)
            warpGemv<0, false><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
        else
            warpGemv<1, false><<<grid, 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
    } else
        gemv<<<dim3(v.rows, slots), 128>>>(v, x, y, ids, routesPerInput, roundOutput, bias);
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
        float4 a;
        if constexpr (Type == 2)
            a = static_cast<float4>(reinterpret_cast<const __nv_fp8x4_e4m3 *>(w.data)[(base + col) / 4]);
        else if constexpr (Type == 0) {
            auto bits = reinterpret_cast<const uint2 *>(w.data)[(base + col) / 4];
            auto lo = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&bits.x));
            auto hi = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&bits.y));
            a = make_float4(lo.x, lo.y, hi.x, hi.y);
        } else
            a = reinterpret_cast<const float4 *>(w.data)[(base + col) / 4];
        float scale = 1.f;
        if constexpr (Type == 2)
            scale = read(w.scale, w.scaleType, size_t(row / 128) * ((width + 127) / 128) + col / 128);
#pragma unroll
        for (int r = 0; r < 4; ++r)
            if (first + r < count) {
                auto v = *reinterpret_cast<const float4 *>(x + size_t(first + r) * width + col);
                float dot = fmaf(a.w, v.w, fmaf(a.z, v.z, fmaf(a.y, v.y, a.x * v.x)));
                sum[r] = fmaf(dot, scale, sum[r]);
            }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
        if (first + r < count) {
            for (int delta = 16; delta; delta /= 2)
                sum[r] += __shfl_down_sync(~0u, sum[r], delta);
            if (!lane)
                out[size_t(first + r) * w.rows + row] = roundOutput ? bf(sum[r]) : sum[r];
        }
}
static void LinearBatch(Weights &w, Scratch &s, const std::string &name, const float *x, float *y, int rows,
                        bool roundOutput = true) {
    if (rows == 1) {
        Linear(w, s, name, x, y, roundOutput);
        return;
    }
    auto v = w.View(name, s.device);
    if (v.type == 2) {
        auto q = s.Get("batchQuant", size_t(rows) * v.cols);
        quantAct<<<dim3((v.cols + 127) / 128, rows), 128>>>(x, q, v.cols);
        x = q;
    }
    dim3 grid((v.rows + 3) / 4, (rows + 3) / 4);
    if (v.type == 2)
        batchGemv<2><<<grid, 128>>>(v, x, y, rows, roundOutput);
    else if (v.type == 0)
        batchGemv<0><<<grid, 128>>>(v, x, y, rows, roundOutput);
    else
        batchGemv<1><<<grid, 128>>>(v, x, y, rows, roundOutput);
}
__global__ static void norm(const float *x, Hy4View w, float *y, int width, float eps, bool roundFirst) {
    float sum = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        sum = fmaf(x[i], x[i], sum);
    float inv = rsqrtf(reduce(sum) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = x[i] * inv;
        y[i] = bf((roundFirst ? bf(v) : v) * (w.data ? read(w.data, w.type, i) : 1.f));
    }
}
void Norm(Weights &w, Scratch &s, const std::string &name, const float *x, float *y, int width) {
    norm<<<1, 256>>>(x, w.View(name, s.device), y, width, 1e-5f, true);
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
void HyperPre(Weights &w, Scratch &s, const std::string &prefix, float *h, float *x, float *post, bool head) {
    float *mix = s.Get("hcMix", 8);
    Linear(w, s, prefix + (head ? "hc_head_fn" : "hc_fn"), h, mix, false);
    auto bias = w.View(prefix + (head ? "hc_head_base" : "hc_base"), s.device),
         scale = w.View(prefix + (head ? "hc_head_scale" : "hc_scale"), s.device);
    hc<<<1, 256>>>(h, mix, bias, scale, x, post, 3072, head);
}
__global__ static void hcPost(float *h, const float *x, const float *post, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * 4)
        h[i] = bf(fmaf(post[i / width], x[i % width], h[i]));
}
void HyperPost(float *h, const float *x, const float *post, int d) {
    hcPost<<<(d * 4 + 255) / 256, 256>>>(h, x, post, d);
}
__global__ static void activate(const float *gate, const float *up, float *out, int width, bool routed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, slot = blockIdx.y;
    if (i >= width)
        return;
    const float *g = gate + size_t(slot) * 2 * width;
    float a = routed ? fminf(g[i], 10.f) : g[i], b = up ? up[i] : fminf(10.f, fmaxf(-10.f, g[width + i]));
    out[size_t(slot) * width + i] = bf(bf(a * sigmoid(a)) * b);
}
void DenseMlp(Weights &w, Scratch &s, const std::string &p, const float *x, float *out) {
    const int width = w.meta.at(p + "gate_proj.weight").dims[0];
    auto a = s.Get("denseGate", width), b = s.Get("denseUp", width), mid = s.Get("denseMid", width);
    Linear(w, s, p + "gate_proj.weight", x, a);
    Linear(w, s, p + "up_proj.weight", x, b);
    activate<<<(width + 255) / 256, 256>>>(a, b, mid, width, false);
    Linear(w, s, p + "down_proj.weight", mid, out);
    if (w.meta.count(p + "post_fc2_norm.weight"))
        Norm(w, s, p + "post_fc2_norm.weight", out, out, 3072);
}
__global__ static void denseBatchActivation(const float *gate, const float *up, float *out, int size) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < size)
        out[i] = bf(bf(gate[i] * sigmoid(gate[i])) * up[i]);
}
void DenseMlpBatch(Weights &w, Scratch &s, const std::string &p, const float *input, float *output,
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
        LinearBatch(w, s, p + "gate_proj.weight", x, gate, n);
        LinearBatch(w, s, p + "up_proj.weight", x, up, n);
        denseBatchActivation<<<(size + 255) / 256, 256>>>(gate, up, mid, size);
        LinearBatch(w, s, p + "down_proj.weight", mid, out, n);
        if (w.meta.count(p + "post_fc2_norm.weight"))
            for (int r = 0; r < n; ++r)
                Norm(w, s, p + "post_fc2_norm.weight", out + r * 3072, out + r * 3072, 3072);
    }
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
void Route(Weights &w, Scratch &s, int layer, const float *x, int *ids, float *scores, float *latent) {
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
void CombineExperts(Weights &w, Scratch &s, int layer, const float *x, const float *routed, float *out) {
    auto p = "model.layers." + std::to_string(layer) + ".mlp.";
    float *shared = s.Get("shared", 3072);
    DenseMlp(w, s, p + "shared_experts.", x, shared);
    Linear(w, s, p + "routed_expert_up_proj.weight", routed, out);
    add<<<12, 256>>>(out, shared, 3072);
    Norm(w, s, p + "combine_output_norm.weight", out, out, 3072);
}
void CombineExpertsBatch(Weights &w, Scratch &s, int layer, const float *input, const float *routed,
                         float *out, int rows) {
    if (rows == 1) {
        CombineExperts(w, s, layer, input, routed, out);
        return;
    }
    auto p = "model.layers." + std::to_string(layer) + ".mlp.";
    auto shared = s.Get("batchShared", size_t(rows) * 3072);
    DenseMlpBatch(w, s, p + "shared_experts.", input, shared, rows);
    LinearBatch(w, s, p + "routed_expert_up_proj.weight", routed, out, rows);
    add<<<rows * 12, 256>>>(out, shared, rows * 3072);
    for (int i = 0; i < rows; ++i)
        Norm(w, s, p + "combine_output_norm.weight", out + i * 3072, out + i * 3072, 3072);
}
void ProjectHead(Weights &w, Scratch &s, float *h, float *values, Data *draftHidden, int position,
                 int draftWidth) {
    auto x = s.Get("headX", 3072);
    HyperPre(w, s, "model.hc_head.", h, x, nullptr, true);
    Norm(w, s, "model.norm.weight", x, x, 3072);
    if (draftHidden)
        SaveDraftHidden(*draftHidden, s.device, position, draftWidth, x, s.step);
    int n = w.meta.at("lm_head.weight").dims[0];
    Linear(w, s, "lm_head.weight", x, values);
}
std::vector<float> Head(Weights &w, Scratch &s, float *h, Data *draftHidden, int position, int draftWidth) {
    int n = w.meta.at("lm_head.weight").dims[0];
    auto values = s.Get("logits", n);
    ProjectHead(w, s, h, values, draftHidden, position, draftWidth);
    std::vector<float> result(n);
    CopyToHost(result.data(), values, n * sizeof(float));
    return result;
}

__device__ static float rope(float a, float b, int j, int pos, float theta) {
    float phase = pos * powf(theta, -2.f * (j / 2) / 64.f);
    return bf(a * bf(cosf(phase)) + ((j & 1) ? b : -b) * bf(sinf(phase)));
}
// Indexed layers retain one normalized token key per slot. The final slot of a
// closed group also owns its pooled, rotated key. Both travel with the request KV.
__global__ static void storeIndexKey(const float *input, __nv_bfloat16 *cache, int stride, int pos,
                                     const StepInput *step) {
    if (step)
        pos = step->position;
    cache[size_t(pos) * stride + stride - 256 + threadIdx.x] = __float2bfloat16(input[threadIdx.x]);
}
__global__ static void poolIndexKey(const __nv_bfloat16 *cache, int stride, int pos, Hy4View value,
                                    Hy4View gate, Hy4View ape, float *pooled, const StepInput *step) {
    if (step)
        pos = step->position;
    const int channel = blockIdx.x, col = threadIdx.x;
    float values[4], gates[4];
    for (int slot = 0; slot < 4; ++slot) {
        float x = __bfloat162float(cache[size_t(pos - 3 + slot) * stride + stride - 256 + col]);
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
__global__ static void storeGroupKey(const float *pooled, __nv_bfloat16 *cache, int stride, int pos,
                                     const StepInput *step) {
    if (step)
        pos = step->position;
    int i = threadIdx.x;
    float x = i < 64 ? pooled[i] : rope(pooled[i], pooled[i ^ 1], i - 64, pos - 3, 10000000.f);
    cache[size_t(pos) * stride + stride - 128 + i] = __float2bfloat16(x);
}
__global__ static void rotateIndexQuery(const float *input, float *output, int pos, const StepInput *step) {
    if (step)
        pos = step->position;
    int i = blockIdx.x * 128 + threadIdx.x, d = threadIdx.x;
    output[i] = d < 64 ? input[i] : rope(input[i], input[i ^ 1], d - 64, pos, 10000000.f);
}
__global__ static void scoreIndexGroups(const float *query, const float *weights, const __nv_bfloat16 *cache,
                                        int stride, float *scores, const StepInput *step) {
    if (step && blockIdx.x >= step->position / 4)
        return;
    int group = blockIdx.x, col = threadIdx.x;
    float key = __bfloat162float(cache[(size_t(group) * 4 + 3) * stride + stride - 128 + col]);
    float score = 0;
    for (int head = 0; head < 16; ++head) {
        float dot = reduce(query[head * 128 + col] * key);
        score += fmaxf(dot, 0.f) * weights[head] * 0.0220970869f;
    }
    if (!col)
        scores[group] = score;
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
                      const float *residual, __nv_bfloat16 *keyCache, int stride, int pos) {
    const auto p = attn + "compressed_indexer.";
    auto key = s.Get("indexTokenKey", 128);
    Linear(w, s, p + "token_key_projection.weight", hidden, key);
    Norm(w, s, p + "token_key_norm.weight", key, key, 128);
    storeIndexKey<<<1, 128>>>(key, keyCache, stride, pos, s.step);
    if ((pos & 3) == 3) {
        auto value = w.View(p + "key_value_projection.weight", s.device);
        auto gate = w.View(p + "key_gate_projection.weight", s.device);
        auto ape = w.View(p + "intra_group_ape", s.device);
        auto pooled = s.Get("indexPooled", 128);
        poolIndexKey<<<128, 128>>>(keyCache, stride, pos, value, gate, ape, pooled, s.step);
        norm<<<1, 128>>>(pooled, w.View(p + "key_norm.weight", s.device), pooled, 128, 1.e-5f, false);
        storeGroupKey<<<1, 128>>>(pooled, keyCache, stride, pos, s.step);
    }
    if (pos < 2048)
        return nullptr;
    const int closedGroups = pos / 4;
    const int groupCapacity = ((pos / 32 * 32 + 31) / 4 + 31) / 32 * 32;
    auto query = s.Get("indexQuery", 16 * 128), rotated = s.Get("indexRotaryQuery", 16 * 128);
    auto weights = s.Get("indexHeadWeights", 16), scores = s.Get("indexGroupScores", groupCapacity);
    auto chosen = reinterpret_cast<int *>(s.Get("indexChosen", 512));
    Linear(w, s, p + "query_projection.weight", residual, query);
    rotateIndexQuery<<<16, 128>>>(query, rotated, pos, s.step);
    Linear(w, s, p + "head_weight_projection.weight", hidden, weights);
    scoreIndexGroups<<<s.step ? groupCapacity : closedGroups, 128>>>(rotated, weights, keyCache, stride,
                                                                     scores, s.step);
    selectIndexGroups<<<1, 256>>>(scores, closedGroups, chosen, s.step);
    return chosen;
}

__global__ static void packKV(const float *kv, const float *rot, float *q, __nv_bfloat16 *key,
                              __nv_bfloat16 *val, int heads, int pos, int slot, float theta, int keyStride,
                              const StepInput *step = nullptr, int keep = 1) {
    if (step) {
        pos = step->position;
        slot = pos % keep;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * 256)
        return;
    int h = i / 256, d = i % 256;
    float k = d < 192 ? kv[h * 448 + d] : rope(rot[d - 192], rot[(d - 192) ^ 1], d - 192, pos, theta);
    key[size_t(slot) * keyStride + i] = __float2bfloat16(k);
    val[size_t(slot) * heads * 256 + i] = __float2bfloat16(kv[h * 448 + 192 + d]);
}
__global__ static void rotQ(const float *input, float *query, int size, int pos, float theta,
                            const StepInput *step = nullptr) {
    if (step)
        pos = step->position;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    int d = i % 256;
    query[i] = d < 192 ? input[i] : rope(input[i], input[i ^ 1], d - 192, pos, theta);
}
__global__ static void scoreKV(const float *q, const __nv_bfloat16 *key, float *score, int start, int length,
                               int heads, int capacity, int keyStride, const int *chosen, bool direct = false,
                               const StepInput *step = nullptr, int window = 0) {
    if (step) {
        length = window < 0 ? 2045 + (step->position & 3)
                 : window   ? min(window, step->position + 1)
                            : step->position + 1;
        start = step->position + 1 - length;
    }
    if (blockIdx.x >= length)
        return;
    int h = blockIdx.y, t = blockIdx.x, i = threadIdx.x,
        slot = chosen ? (direct ? chosen[t] : chosen[t / 4] * 4 + t % 4) : (start + t) % capacity;
    const auto *k = key + size_t(slot) * keyStride + h * 256;
    float dot = reduce(q[h * 256 + i] * __bfloat162float(k[i]));
    if (!i)
        score[h * length + t] = bf(bf(dot) * 0.0625f);
}
__global__ static void probKV(float *scores, Hy4View sink, int length, const StepInput *step = nullptr,
                              int window = 0) {
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
__global__ static void applyKV(const float *scores, const __nv_bfloat16 *val, const float *gate, float *out,
                               int start, int length, int heads, int cap, const int *chosen,
                               bool direct = false, const StepInput *step = nullptr, int window = 0) {
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
        sum = fmaf(scores[h * length + t], __bfloat162float(val[(size_t(slot) * heads + h) * 256 + j]), sum);
    }
    out[h * 256 + j] = bf(bf(sum) * bf(sigmoid(gate[h * 256 + j])));
}
static void ReserveKV(Data &data, int device, int seq, int width, int retained) {
    const int size = std::min(seq, retained);
    if (data.dims.empty()) {
        data.dataType = BFLOAT16;
        data.UpdateUnitSize();
        data.dataDevice = CUDA;
        data.dataDeviceIds = {device};
    }
    if (data.expansionDims.empty() || data.expansionDims[1] < size)
        data.Expansion({1, std::min(retained, (size + 31) / 32 * 32), width});
    data.Resize({1, size, width});
}
__global__ static void saveChosenRing(int *ring, const int *chosen, const StepInput *step, int slots) {
    int i = blockIdx.x * 256 + threadIdx.x;
    ring[(step->position % slots) * 512 + i] = chosen[i];
}
struct ProjectedAttention {
    float *qr, *q, *kv, *expanded, *gate, *mix;
};
static void AttentionRowForward(Weights &w, Scratch &s, int layer, int position, const float *x, float *out,
                                Data &kc, Data &vc, int draftWidth, DraftMemory *draft, int draftLayer,
                                const ProjectedAttention *batch) {
    const std::string p = "model.layers." + std::to_string(layer) + ".self_attn.";
    bool sliding = w.config["layer_types"][layer].string_value() == "sliding_attention";
    int heads = sliding ? 64 : 32, width = heads * 256;
    const int keep = sliding ? 512 + draftWidth : w.config["max_position_embeddings"].int_value();
    const int keyStride = width + (sliding ? 0 : 256);
    float *qr, *q, *kv, *expanded, *gate;
    auto rq = s.Get("rotQ", width);
    if (batch) {
        qr = batch->qr;
        q = batch->q;
        kv = batch->kv;
        expanded = batch->expanded;
        gate = batch->gate;
    } else {
        qr = s.Get("qResid", 1024);
        q = s.Get("q", width);
        kv = s.Get("kvA", 576);
        expanded = s.Get("kvB", heads * 448);
        gate = s.Get("attGate", width);
        Linear(w, s, p + "q_a_proj.weight", x, qr);
        Norm(w, s, p + "q_a_layernorm.weight", qr, qr, 1024);
        Linear(w, s, p + "q_b_proj.weight", qr, q);
        Linear(w, s, p + "kv_a_proj_with_mqa.weight", x, kv);
        Norm(w, s, p + "kv_a_layernorm.weight", kv, kv, 512);
        Linear(w, s, p + "kv_b_proj.weight", kv, expanded);
        Linear(w, s, p + "linear_gate.weight", x, gate);
    }
    int length = sliding ? std::min(position + 1, 512) : position + 1, start = position + 1 - length;
    ReserveKV(kc, s.device, position + 1, keyStride, keep);
    ReserveKV(vc, s.device, position + 1, width, keep);
    auto key = (__nv_bfloat16 *)kc.cudaData, val = (__nv_bfloat16 *)vc.cudaData;
    float theta = sliding ? 10000.f : 10000000.f;
    packKV<<<width / 256, 256>>>(expanded, kv + 512, q, key, val, heads, position, position % keep, theta,
                                 keyStride, s.step, keep);
    rotQ<<<width / 256, 256>>>(q, rq, width, position, theta, s.step);
    const int *chosen = sliding ? nullptr : IndexPool(w, s, p, x, qr, key, keyStride, position);
    if (draft) {
        const auto prefix = "model.dfly_layers." + std::to_string(draftLayer) + ".self_attn.";
        auto projected = s.Get("draftHistoricalKV", 32 * 448);
        Linear(w, s, prefix + "kv_b_proj.weight", kv, projected);
        ReserveKV(draft->keys, s.device, position + 1, 32 * 256, keep);
        ReserveKV(draft->values, s.device, position + 1, 32 * 256, keep);
        packKV<<<32, 256>>>(projected, kv + 512, nullptr, static_cast<__nv_bfloat16 *>(draft->keys.cudaData),
                            static_cast<__nv_bfloat16 *>(draft->values.cudaData), 32, position, position,
                            theta, 32 * 256, s.step, keep);
        if (chosen) {
            if (draft->recalled.dims.empty()) {
                draft->recalled.dataType = INT32;
                draft->recalled.UpdateUnitSize();
                draft->recalled.dataDevice = CUDA;
                draft->recalled.dataDeviceIds = {s.device};
                draft->recalled.Resize({draftWidth + 1, 512});
                draft->recalled.Allocate(false);
            }
            auto dst =
                static_cast<int *>(draft->recalled.cudaData) + size_t(position % (draftWidth + 1)) * 512;
            if (s.step)
                saveChosenRing<<<2, 256>>>(static_cast<int *>(draft->recalled.cudaData), chosen, s.step,
                                           draftWidth + 1);
            else
                Check(cudaMemcpyAsync(dst, chosen, 512 * sizeof(int), cudaMemcpyDeviceToDevice,
                                      cudaStreamPerThread));
        }
    }
    if (chosen)
        length = 511 * 4 + (position % 4) + 1;
    auto scores = s.Get("attScores", size_t(heads) * ((length + 127) / 128 * 128)),
         mix = batch ? batch->mix : s.Get("attMix", width);
    int launchLength = s.step ? (length + 31) / 32 * 32 : length;
    scoreKV<<<dim3(launchLength, heads), 256>>>(rq, key, scores, start, length, heads, keep, keyStride,
                                                chosen, false, s.step,
                                                chosen    ? -1
                                                : sliding ? 512
                                                          : 0);
    auto sink = w.View(p + "learnable_sink_param", s.device);
    probKV<<<heads, 256>>>(scores, sink, length, s.step, chosen ? -1 : sliding ? 512 : 0);
    applyKV<<<heads, 256>>>(scores, val, gate, mix, start, length, heads, keep, chosen, false, s.step,
                            chosen    ? -1
                            : sliding ? 512
                                      : 0);
    if (batch)
        return;
    Linear(w, s, p + "o_proj.weight", mix, out);
    Norm(w, s, p + "post_proj_norm.weight", out, out, 3072);
}

void Attention(Weights &w, Scratch &s, int layer, int position, const float *x, float *out, Data &kc,
               Data &vc, int draftWidth, DraftMemory *draft, int draftLayer) {
    AttentionRowForward(w, s, layer, position, x, out, kc, vc, draftWidth, draft, draftLayer, nullptr);
}
void AttentionBatch(Weights &w, Scratch &s, int layer, const std::vector<AttentionRow> &rows,
                    const float *inputs, float *outputs) {
    const std::string p = "model.layers." + std::to_string(layer) + ".self_attn.";
    const int heads = w.config["layer_types"][layer].string_value() == "sliding_attention" ? 64 : 32;
    const int width = heads * 256;
    if (rows.size() == 1) {
        const auto &r = rows.front();
        s.step = r.step;
        Attention(w, s, layer, r.position, inputs, outputs, *r.key, *r.value, r.draftWidth, r.draft,
                  r.draftLayer);
        return;
    }
    for (int first = 0; first < int(rows.size()); first += 16) {
        int n = std::min(16, int(rows.size()) - first);
        auto x = inputs + size_t(first) * 3072;
        auto y = outputs + size_t(first) * 3072;
        auto qr = s.Get("batchQr", n * 1024), q = s.Get("batchQ", n * width), kv = s.Get("batchKv", n * 576);
        auto expanded = s.Get("batchExpanded", n * heads * 448), gate = s.Get("batchGate", n * width),
             mix = s.Get("batchMix", n * width);
        LinearBatch(w, s, p + "q_a_proj.weight", x, qr, n);
        LinearBatch(w, s, p + "kv_a_proj_with_mqa.weight", x, kv, n);
        // kv_a begins with 512 normalized channels followed by 64 rotary channels.
        auto normKv = s.Get("batchKvLatent", n * 512);
        for (int i = 0; i < n; ++i) {
            Norm(w, s, p + "q_a_layernorm.weight", qr + i * 1024, qr + i * 1024, 1024);
            Norm(w, s, p + "kv_a_layernorm.weight", kv + i * 576, kv + i * 576, 512);
            CopyStageCaptured(normKv + i * 512, s.device, kv + i * 576, s.device, 512);
        }
        LinearBatch(w, s, p + "q_b_proj.weight", qr, q, n);
        LinearBatch(w, s, p + "kv_b_proj.weight", normKv, expanded, n);
        LinearBatch(w, s, p + "linear_gate.weight", x, gate, n);
        for (int i = 0; i < n; ++i) {
            const auto &r = rows[first + i];
            s.step = r.step;
            ProjectedAttention projections{qr + i * 1024,    q + i * width,
                                           kv + i * 576,     expanded + i * heads * 448,
                                           gate + i * width, mix + i * width};
            AttentionRowForward(w, s, layer, r.position, x + i * 3072, y + i * 3072, *r.key, *r.value,
                                r.draftWidth, r.draft, r.draftLayer, &projections);
        }
        LinearBatch(w, s, p + "o_proj.weight", mix, y, n);
        for (int i = 0; i < n; ++i)
            Norm(w, s, p + "post_proj_norm.weight", y + i * 3072, y + i * 3072, 3072);
    }
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
    std::array<int, 3> ids;
    uint16_t *staging = nullptr;
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
        Check(cudaHostAlloc(&input->staging, 16 * 96 * sizeof(uint16_t), cudaHostAllocPortable));
        entry = std::move(input);
    }
    static_cast<HostEngramInput *>(entry.get())->ids = ids;
}
static void CUDART_CB readEngramRows(void *opaque) {
    auto &in = *static_cast<HostEngramInput *>(opaque);
    for (int head = 0; head < 16; ++head) {
        auto k = rowId(head, in.ids[0], in.ids[1], in.ids[2], in.config, in.lookup);
        std::memcpy(in.staging + head * 96, in.table + k * 96, 96 * sizeof(uint16_t));
    }
}
void RunPreparedEngram(Weights &w, Scratch &s, int layer, int row, const std::array<int, 3> &ids,
                       float *streams) {
    auto *input = static_cast<HostEngramInput *>(s.engramInputs.at({layer, row}).get());
    Check(cudaLaunchHostFunc(cudaStreamPerThread, readEngramRows, input));
    Engram(w, s, layer, ids, streams, input->staging, true);
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
            const uint16_t *hostRows, bool asyncCopy) {
    if (ids[0] == w.config["engram_pad_id"].int_value() || ids[0] == 120120 || ids[0] == 120124)
        return;
    auto p = "model.layers." + std::to_string(layer) + ".engram_layer.";
    auto m = HashConfig(w, layer, p);
    auto emb = s.Get("engramVectors", 1536);
    const int *map = nullptr;
    const __nv_bfloat16 *table;
    if (hostRows) {
        auto *staged = s.Get("engramStaged", (16 * 96 + 1) / 2);
        if (asyncCopy)
            Check(cudaMemcpyAsync(staged, hostRows, 16 * 96 * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                  cudaStreamPerThread));
        else
            CopyToGPU(staged, hostRows, 16 * 96 * sizeof(uint16_t));
        table = (const __nv_bfloat16 *)staged;
    } else {
        map = (const int *)w.Device(p + "lookup_table", s.device);
        table = (const __nv_bfloat16 *)w.Device(p + "multi_head_embedding.embedding.weight", s.device);
    }
    gather<<<16, 128>>>(table, map, m, ids[0], ids[1], ids[2], hostRows != nullptr, emb, s.step);
    for (int j = 0; j < 16; j++)
        norm<<<1, 128>>>(emb + j * 96,
                         w.View(p + "lookup_per_head_norms." + std::to_string(j) + ".weight", s.device),
                         emb + j * 96, 96, 1.1920929e-7f, false);
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
__global__ static void draftEmbedding(Hy4View w, int constant, const int *selected, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.cols) {
        int token = selected ? *selected : constant;
        output[i] = read(w.data, w.type, size_t(token) * w.cols + i);
    }
}
__global__ static void fusionResidual(float *anchor, const float *fused, Hy4View gate) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    anchor[i] = bf(anchor[i] + bf(read(gate.data, gate.type, i) * fused[i]));
}
void DraftEmbed(Weights &w, Scratch &s, int anchor, int width, int mask, const float *targetHidden,
                float *block) {
    auto table = w.View("model.embed_tokens.weight", s.device);
    for (int row = 0; row < width; ++row)
        draftEmbedding<<<12, 256>>>(table, row ? mask : anchor, nullptr, block + size_t(row) * 3072);
    const std::string p = "model.dfly_layers.mtp_fusion.";
    auto concat = s.Get("draftFusionPair", 6144), fused = s.Get("draftFusionVector", 3072);
    Norm(w, s, p + "enorm.weight", block, concat, 3072);
    Norm(w, s, p + "hnorm.weight", targetHidden, concat + 3072, 3072);
    Linear(w, s, p + "eh_proj.weight", concat, fused);
    fusionResidual<<<12, 256>>>(block, fused, w.View(p + "gate", s.device));
}
__global__ static void draftSlots(int *slots, int history, int base, int pos, const int *previous) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < history)
        slots[i] = previous ? previous[i / 4] * 4 + i % 4 : i;
    else if (i < history + pos - base + 1)
        slots[i] = base + i - history;
}
void DraftLayer(Weights &w, Scratch &s, int layer, int base, int width, DraftMemory &memory, float *block) {
    const std::string p = "model.dfly_layers." + std::to_string(layer) + ".", a = p + "self_attn.";
    const int limit = w.config["max_position_embeddings"].int_value();
    ReserveKV(memory.keys, s.device, base + width, 32 * 256, limit);
    ReserveKV(memory.values, s.device, base + width, 32 * 256, limit);
    auto key = static_cast<__nv_bfloat16 *>(memory.keys.cudaData);
    auto value = static_cast<__nv_bfloat16 *>(memory.values.cudaData);
    auto x = s.Get("draftInput", 3072), qr = s.Get("draftQRes", 1024), q = s.Get("draftQ", 8192);
    auto rotated = s.Get("draftRotated", 8192), kv = s.Get("draftKV", 576),
         expanded = s.Get("draftKVExpand", 32 * 448);
    auto gate = s.Get("draftAttGate", 8192), mix = s.Get("draftMix", 8192),
         branch = s.Get("draftBranch", 3072);
    const int history = base <= 2048 ? base : 511 * 4 + (base - 1) % 4 + 1;
    auto scores = s.Get("draftScores", size_t(32) * (history + width));
    auto slots = reinterpret_cast<int *>(s.Get("draftSlots", history + width));
    auto previous = base <= 2048 ? nullptr
                                 : static_cast<const int *>(memory.recalled.cudaData) +
                                       size_t((base - 1) % memory.recalled.dims[0]) * 512;
    for (int row = 0; row < width; ++row) {
        const int pos = base + row, length = history + row + 1;
        float *h = block + size_t(row) * 3072;
        Norm(w, s, p + "input_layernorm.weight", h, x, 3072);
        Linear(w, s, a + "q_a_proj.weight", x, qr);
        Norm(w, s, a + "q_a_layernorm.weight", qr, qr, 1024);
        Linear(w, s, a + "q_b_proj.weight", qr, q);
        Linear(w, s, a + "kv_a_proj_with_mqa.weight", x, kv);
        Norm(w, s, a + "kv_a_layernorm.weight", kv, kv, 512);
        Linear(w, s, a + "kv_b_proj.weight", kv, expanded);
        Linear(w, s, a + "linear_gate.weight", x, gate);
        packKV<<<32, 256>>>(expanded, kv + 512, nullptr, key, value, 32, pos, pos, 10000000.f, 8192);
        rotQ<<<32, 256>>>(q, rotated, 8192, pos, 10000000.f);
        draftSlots<<<(length + 255) / 256, 256>>>(slots, history, base, pos, previous);
        scoreKV<<<dim3(length, 32), 256>>>(rotated, key, scores, 0, length, 32, limit, 8192, slots, true);
        probKV<<<32, 256>>>(scores, w.View(a + "learnable_sink_param", s.device), length);
        applyKV<<<32, 256>>>(scores, value, gate, mix, 0, length, 32, limit, slots, true);
        Linear(w, s, a + "o_proj.weight", mix, branch);
        Norm(w, s, a + "post_proj_norm.weight", branch, branch, 3072);
        add<<<12, 256>>>(h, branch, 3072);
        Norm(w, s, p + "post_attention_layernorm.weight", h, x, 3072);
        DenseMlp(w, s, p + "mlp.", x, branch);
        add<<<12, 256>>>(h, branch, 3072);
    }
}
__global__ static void argmaxLogits(const float *logits, int size, int *token) {
    __shared__ float values[256];
    __shared__ int indices[256];
    int tid = threadIdx.x, best = tid;
    for (int i = tid + 256; i < size; i += 256)
        if (logits[i] > logits[best])
            best = i;
    values[tid] = logits[best];
    indices[tid] = best;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride && (values[tid + stride] > values[tid] ||
                             (values[tid + stride] == values[tid] && indices[tid + stride] < indices[tid]))) {
            values[tid] = values[tid + stride];
            indices[tid] = indices[tid + stride];
        }
        __syncthreads();
    }
    if (!tid)
        *token = indices[0];
}
std::vector<int> DraftReadout(Weights &w, Scratch &s, int anchor, int width, float *block) {
    const std::string p = "model.dfly_layers.hidden_correction.";
    auto table = w.View("model.embed_tokens.weight", s.device);
    auto embed = s.Get("draftPreviousEmbed", 3072), pair = s.Get("draftCorrectionPair", 6144);
    auto g = s.Get("draftCorrectionGate", 3072), up = s.Get("draftCorrectionUp", 3072);
    auto activated = s.Get("draftCorrectionAct", 3072), delta = s.Get("draftCorrectionDelta", 3072);
    const int vocab = w.meta.at("lm_head.weight").dims[0];
    auto logits = s.Get("draftHeadLogits", vocab);
    auto picks = reinterpret_cast<int *>(s.Get("draftPicks", width - 1));
    for (int i = 0; i < width - 1; ++i) {
        auto h = block + size_t(i + 1) * 3072;
        Norm(w, s, "model.dfly_layers.final_norm.weight", h, h, 3072);
        draftEmbedding<<<12, 256>>>(table, anchor, i ? picks + i - 1 : nullptr, embed);
        Norm(w, s, p + "hidden_norm.weight", h, pair, 3072);
        Norm(w, s, p + "embed_norm.weight", embed, pair + 3072, 3072);
        Linear(w, s, p + "gate_proj.weight", pair, g);
        Linear(w, s, p + "up_proj.weight", pair, up);
        activate<<<12, 256>>>(g, up, activated, 3072, false);
        Linear(w, s, p + "down_proj.weight", activated, delta);
        add<<<12, 256>>>(h, delta, 3072);
        Linear(w, s, "lm_head.weight", h, logits);
        argmaxLogits<<<1, 256>>>(logits, vocab, picks + i);
    }
    std::vector<int> result(width - 1);
    CopyToHost(result.data(), picks, result.size() * sizeof(int));
    return result;
}

} // namespace fastllm::hy4
