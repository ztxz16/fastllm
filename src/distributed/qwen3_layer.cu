#include "distributed/qwen3_layer.h"
#include "process_protocol.h"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace distributed {
namespace {
/**
 * 从复制的词嵌入表查找输入激活，token ID 已在 CPU 验证。
 * @param ids GPU uint32 [tokens]。
 * @param embedding FP16 行主序 [vocab,hidden]。
 * @param x FP16 行主序 [tokens,hidden] 输出。
 * @param count tokens*hidden，已检查不超出 int。
 * @param hidden 每个词嵌入的宽度。
 */
__global__ void Embed(const uint32_t *ids, const half *embedding, half *x, int count, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        x[i] = embedding[uint64_t(ids[i / hidden]) * hidden + i % hidden];
    }
}

void Cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen3 layer CUDA: ") + cudaGetErrorString(result));
    }
}
void Blas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Qwen3 layer cuBLAS: " + std::to_string(int(result)));
    }
}
void Nccl(ncclResult_t result) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("Qwen3 layer NCCL: ") + ncclGetErrorString(result));
    }
}

/**
 * 对每行执行 RMSNorm；FP32 求和，归一化后转 FP16，再乘 FP16 权重并转 FP16。
 * @param x 行主序 FP16 输入。
 * @param weight 长度 width 的 FP16 scale。
 * @param output FP16 输出，可与 x 相同。
 * @param width 归一化维度。
 * @param rowStride token 间跨度，包含 QKV 融合行中的其他头。
 * @param heads 每 token 的待处理头数。
 * @param offset 融合行内待处理部分的起点。
 * @param epsilon 分母中的正数常量。
 */
__global__ void Norm(const half *x, const half *weight, half *output, int width,
                     int rowStride, int heads, int offset, float epsilon) {
    __shared__ float sums[256];
    int base = (blockIdx.x / heads) * rowStride + (blockIdx.x % heads) * width + offset;
    float sum = 0;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = __half2float(x[base + i]);
        sum += v * v;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (threadIdx.x < step) {
            sums[threadIdx.x] += sums[threadIdx.x + step];
        }
        __syncthreads();
    }
    float scale = rsqrtf(sums[0] / width + epsilon);
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        half normalized = __float2half_rn(__half2float(x[base + i]) * scale);
        output[base + i] = __float2half_rn(__half2float(normalized) * __half2float(weight[i]));
    }
}

/**
 * 按 Qwen3 split-half 约定旋转 Q/K；FP32 三角函数，最终转 FP16，V 保持原样。
 * @param qkv FP16 [T,(qHeads+2*kvHeads)*dim]，原地更新 Q/K。
 * @param qHeads 本 rank 的 Q 头数。
 * @param kvHeads 本 rank 的 KV 头数。
 * @param dim 偶数头维度。
 * @param stride 每 token 的元素跨度。
 * @param positionBegin 序列的连续位置起点。
 * @param theta RoPE 基数。
 */
__global__ void Rope(half *qkv, int qHeads, int kvHeads, int dim, int stride,
                     int positionBegin, float theta) {
    int head = blockIdx.x % (qHeads + kvHeads), token = blockIdx.x / (qHeads + kvHeads);
    int base = token * stride + head * dim;
    for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) {
        float angle = (positionBegin + token) / powf(theta, float(2 * i) / dim);
        // cos/sin 与 eager FP16 激活保持同精度；每个乘法分别转 FP16 后再加减。
        float c = __half2float(__float2half_rn(cosf(angle)));
        float s = __half2float(__float2half_rn(sinf(angle)));
        float a = __half2float(qkv[base + i]), b = __half2float(qkv[base + i + dim / 2]);
        float ac = __half2float(__float2half_rn(a * c)), bs = __half2float(__float2half_rn(b * s));
        float bc = __half2float(__float2half_rn(b * c)), as = __half2float(__float2half_rn(a * s));
        qkv[base + i] = __float2half_rn(ac - bs);
        qkv[base + i + dim / 2] = __float2half_rn(bc + as);
    }
}

/**
 * 将当前层旋转后的 K 与原始 V 追加到请求缓存，不写历史 token。
 * @param qkv FP16 [tokens,(qHeads+2*kvHeads)*dim]，K 已执行 Norm/RoPE。
 * @param cache FP16 [capacity,2,kvHeads,dim]，第二维依次 K、V。
 * @param count tokens*2*kvHeads*dim，调用方已检查边界。
 * @param width 2*kvHeads*dim。
 * @param qWidth qHeads*dim。
 * @param past 已缓存 token 数；写入 [past,past+tokens)。
 */
__global__ void AppendKV(const half *qkv, half *cache, int count, int width, int qWidth, int past) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        cache[size_t(past) * width + i] = qkv[(i / width) * (qWidth + width) + qWidth + i % width];
    }
}

/**
 * 计算一个 token/head 的 causal attention，支持请求内 KV 与无缓存重算。
 *
 * QK 点积、softmax、加权 V 均 FP32 累加，输出 FP16。按全局 Q 头编号映射 KV，
 * 支持 KV 头复制及不均匀 Q 分片。共享内存只存 <=2048 个概率，不创建 T*T*H 张量。
 * @param qkv FP16 [T,(qHeads+2*kvHeads)*dim]，Q/K 已归一化并旋转。
 * @param context FP16 [T,qHeads*dim] 输出。
 * @param qHeads 本 rank 的 Q 头数。
 * @param kvHeads 本 rank 的 KV 头数。
 * @param dim 头维度。
 * @param qBegin 本 rank 首个 Q 头的全局编号。
 * @param kvBegin 本 rank 首个 KV 头的全局编号。
 * @param groupSize 全局 Q 头数除以全局 KV 头数。
 * @param cache 可空的 FP16 [capacity,2,kvHeads,dim]；非空时当前 token 的 KV 已追加。
 * @param past 当前输入之前已缓存的长度；无缓存时为零。
 */
__global__ void Attention(const half *qkv, half *context, int qHeads, int kvHeads, int dim,
                          int qBegin, int kvBegin, int groupSize, const half *cache, int past) {
    __shared__ float probabilities[2048];
    int head = blockIdx.x % qHeads, token = blockIdx.x / qHeads;
    int kv = (qBegin + head) / groupSize - kvBegin;
    int stride = (qHeads + 2 * kvHeads) * dim;
    const half *keys = cache ? cache : qkv + qHeads * dim;
    int kvStride = cache ? 2 * kvHeads * dim : stride;
    int last = past + token;
    for (int key = threadIdx.x; key <= last; key += blockDim.x) {
        float dot = 0;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(qkv[token * stride + head * dim + d]) *
                   __half2float(keys[key * kvStride + kv * dim + d]);
        }
        probabilities[key] = dot * rsqrtf(float(dim));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = probabilities[0], total = 0;
        for (int i = 1; i <= last; ++i) {
            maximum = fmaxf(maximum, probabilities[i]);
        }
        for (int i = 0; i <= last; ++i) {
            probabilities[i] = expf(probabilities[i] - maximum);
            total += probabilities[i];
        }
        for (int i = 0; i <= last; ++i) {
            probabilities[i] /= total;
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0;
        for (int key = 0; key <= last; ++key) {
            value += probabilities[key] * __half2float(keys[key * kvStride + (kvHeads + kv) * dim + d]);
        }
        context[(token * qHeads + head) * dim + d] = __float2half_rn(value);
    }
}

template<class T>
__global__ void Bias(T *values, const half *bias, int count, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[i] = T(float(values[i]) + __half2float(bias[i % width]));
    }
}

__global__ void Residual(half *x, const float *partial, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        half projected = __float2half_rn(partial[i]);
        x[i] = __float2half_rn(__half2float(x[i]) + __half2float(projected));
    }
}

/** Gate/Up 按行拼接为 [T,2*M]，SiLU 先转 FP16，再与 Up 相乘转 FP16。 */
__global__ void Swiglu(const half *gateup, half *output, int count, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int base = (i / width) * 2 * width + i % width;
        float gate = __half2float(gateup[base]);
        half activated = __float2half_rn(gate / (1 + expf(-gate)));
        output[i] = __float2half_rn(__half2float(activated) * __half2float(gateup[base + width]));
    }
}
} // namespace

struct Qwen3LayerExecutor::Impl {
    const Qwen3DeviceWeights &weights;
    cudaStream_t stream;
    ncclComm_t comm;
    cublasHandle_t blas = nullptr;
    std::vector<void*> buffers;
    size_t allocated = 0;
    std::vector<half*> kvCache;
    Qwen3RequestInfo request;

    void *Allocate(size_t bytes) {
        if (bytes > 256ULL * 1024 * 1024 - allocated) {
            throw std::runtime_error("Qwen3 layer workspace exceeds 256 MiB");
        }
        buffers.push_back(nullptr);
        Cuda(cudaMalloc(&buffers.back(), bytes));
        allocated += bytes;
        return buffers.back();
    }

    const half *Weight(const std::string &name, const std::vector<uint64_t> &shape, bool optional = false) {
        auto tensor = weights.Find(name);
        if (!tensor && optional) {
            return nullptr;
        }
        if (!tensor || tensor->shape != shape) {
            throw std::runtime_error("Qwen3 layer missing/mismatched weight: " + name);
        }
        return static_cast<const half*>(tensor->data);
    }

    /** 执行 Y[T,N]=X[T,K]*W[N,K]^T；FP16 输入/权重，FP32 累加，输出类型由调用方指定。 */
    void Linear(const half *x, const half *w, void *y, int t, int n, int k, bool fp32) {
        const float alpha = 1, beta = 0;
        Blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
            w, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, y, fp32 ? CUDA_R_32F : CUDA_R_16F,
            n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    }

    struct Workspace {
        half *norm = nullptr, *qkv = nullptr, *context = nullptr, *gateup = nullptr, *activated = nullptr;
        float *partial = nullptr;
    };

    /** 在分配和算子执行前检查单层与完整 prefill 共用的布局约束。 */
    void Validate(uint32_t layer, uint32_t tokens, uint32_t hidden, uint32_t positionBegin,
                  Qwen3LayerStage stage) const {
        const auto &p = weights.Plan();
        if (!blas || layer >= p.layerCount || !tokens || tokens > 128 || hidden != p.hiddenSize ||
            !hidden || hidden > 65536 || positionBegin > 1048576 ||
            uint32_t(stage) > 2 || p.headDim % 2 || p.headDim > 256 || !p.headDim ||
            (p.queryHeadCount + 2 * p.kvHeadCount) * p.headDim > 65536 || p.intermediateCount > 262144) {
            throw std::runtime_error("invalid/unsupported Qwen3 layer shape, stage or position");
        }
        if (!p.forwardUnsupportedReason.empty() || p.ropeTheta <= 1) {
            throw std::runtime_error("unsupported Qwen3 forward configuration: " + p.forwardUnsupportedReason);
        }
    }

    /** 按请求一次性分配工作区；同一模型各层复用，成功结束后统一释放。 */
    Workspace Prepare(uint32_t tokens, uint32_t hidden, Qwen3LayerStage stage) {
        const auto &p = weights.Plan();
        Workspace w;
        w.norm = static_cast<half*>(Allocate(size_t(tokens) * hidden * 2));
        w.partial = static_cast<float*>(Allocate(size_t(tokens) * hidden * 4));
        if (stage != Qwen3LayerStage::Mlp) {
            w.qkv = static_cast<half*>(Allocate(size_t(tokens) * (p.queryHeadCount + 2 * p.kvHeadCount) * p.headDim * 2));
            w.context = static_cast<half*>(Allocate(size_t(tokens) * p.queryHeadCount * p.headDim * 2));
        }
        if (stage != Qwen3LayerStage::Attention) {
            w.gateup = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 4));
            w.activated = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 2));
        }
        return w;
    }

    /**
     * 在 GPU 原地更新一层隐藏状态；单层检查与全模型共用此计算和通信路径。
     * @param layer 层号，调用方已验证。
     * @param tokens 序列长度，范围 1..128。
     * @param positionBegin 连续位置起点。
     * @param stage 指定 Attention、MLP 或完整层。
     * @param x FP16 [tokens,hidden] 行主序输入/输出，不读回 CPU。
     * @param w 当前请求的 GPU 工作区，同 stream 顺序复用。
     * @param cached true 表示写入当前层 KV，并在 Attention 中读取有效历史。
     */
    void RunLayer(uint32_t layer, uint32_t tokens, uint32_t positionBegin, Qwen3LayerStage stage,
                  half *x, const Workspace &w, bool cached = false) {
        auto &r = *this;
        const auto &p = weights.Plan();
        const uint32_t hidden = p.hiddenSize;
        auto norm = w.norm, qkv = w.qkv, context = w.context, gateup = w.gateup, activated = w.activated;
        auto partial = w.partial;
        const int t = tokens, h = hidden, d = p.headDim, qh = p.queryHeadCount, kh = p.kvHeadCount;
        const int qw = qh * d, kw = kh * d, qkvWidth = qw + 2 * kw, m = p.intermediateCount;
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        const bool attention = stage != Qwen3LayerStage::Mlp, mlp = stage != Qwen3LayerStage::Attention;
        if (attention) {
            Norm<<<t,256,0,r.stream>>>(x, r.Weight(prefix + "input_layernorm.weight", {hidden}), norm, h, h, 1, 0, p.rmsNormEps);
            r.Linear(norm, r.Weight(prefix + "self_attn.mergeqkv.weight", {uint64_t(qkvWidth), hidden}), qkv, t, qkvWidth, h, false);
            if (auto bias = r.Weight(prefix + "self_attn.mergeqkv.bias", {uint64_t(qkvWidth)}, true)) {
                Bias<<<(t*qkvWidth+255)/256,256,0,r.stream>>>(qkv, bias, t*qkvWidth, qkvWidth);
            }
            Norm<<<t*qh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.q_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, qh, 0, p.rmsNormEps);
            Norm<<<t*kh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.k_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, kh, qw, p.rmsNormEps);
            Rope<<<t*(qh+kh),256,0,r.stream>>>(qkv, qh, kh, d, qkvWidth, positionBegin, p.ropeTheta);
            half *cache = cached ? kvCache.at(layer) : nullptr;
            const int past = cached ? request.tokens : 0;
            if (cached) {
                AppendKV<<<(t*2*kw+255)/256,256,0,r.stream>>>(qkv, cache, t*2*kw, 2*kw, qw, past);
            }
            Attention<<<t*qh,256,0,r.stream>>>(qkv, context, qh, kh, d, p.queryHeadBegin, p.kvHeadBegin,
                                              p.queryHeads / p.kvHeads, cache, past);
            r.Linear(context, r.Weight(prefix + "self_attn.o_proj.weight", {hidden, uint64_t(qw)}), partial, t, h, qw, true);
            if (auto bias = r.Weight(prefix + "self_attn.o_proj.bias", {hidden}, true)) {
                Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
            }
            Cuda(cudaGetLastError());
            Nccl(ncclAllReduce(partial, partial, size_t(t) * h, ncclFloat, ncclSum, r.comm, r.stream));
            Residual<<<(t*h+255)/256,256,0,r.stream>>>(x, partial, t*h);
        }
        if (mlp) {
            Norm<<<t,256,0,r.stream>>>(x, r.Weight(prefix + "post_attention_layernorm.weight", {hidden}), norm, h, h, 1, 0, p.rmsNormEps);
            r.Linear(norm, r.Weight(prefix + "mlp.gateup_proj.weight", {uint64_t(2*m), hidden}), gateup, t, 2*m, h, false);
            Swiglu<<<(t*m+255)/256,256,0,r.stream>>>(gateup, activated, t*m, m);
            r.Linear(activated, r.Weight(prefix + "mlp.down_proj.weight", {hidden, uint64_t(m)}), partial, t, h, m, true);
            if (auto bias = r.Weight(prefix + "mlp.down_proj.bias", {hidden}, true)) {
                Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
            }
            Cuda(cudaGetLastError());
            Nccl(ncclAllReduce(partial, partial, size_t(t) * h, ncclFloat, ncclSum, r.comm, r.stream));
            Residual<<<(t*h+255)/256,256,0,r.stream>>>(x, partial, t*h);
        }
        Cuda(cudaGetLastError());
    }

    /** 同步当前 stream 并检查 NCCL 异步错误；失败交给 worker 整组退出。 */
    void Synchronize() {
        Cuda(cudaStreamSynchronize(stream));
        ncclResult_t asynchronous;
        Nccl(ncclCommGetAsyncError(comm, &asynchronous));
        Nccl(asynchronous);
    }

    void ReleaseBuffers() {
        Cuda(cudaStreamSynchronize(stream));
        for (auto &buffer : buffers) {
            Cuda(cudaFree(buffer));
            buffer = nullptr;
        }
        buffers.clear();
        allocated = 0;
    }
};

Qwen3LayerExecutor::Qwen3LayerExecutor(const Qwen3DeviceWeights &weights, cudaStream_t stream, ncclComm_t comm)
    : impl(new Impl{weights, stream, comm}) {
    if (!stream || !comm) {
        throw std::runtime_error("Qwen3 layer requires initialized stream/NCCL");
    }
    Blas(cublasCreate(&impl->blas));
    Blas(cublasSetStream(impl->blas, stream));
    Blas(cublasSetMathMode(impl->blas, CUBLAS_DEFAULT_MATH));
}

// 异常析构不进入可能阻塞的 CUDA 清理；worker 的失败路径退出进程回收上下文。
Qwen3LayerExecutor::~Qwen3LayerExecutor() = default;

std::vector<float> Qwen3LayerExecutor::Forward(uint32_t layer, uint32_t tokens, uint32_t hidden,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input) {
    auto &r = *impl;
    r.Validate(layer, tokens, hidden, positionBegin, stage);
    if (input.size() != uint64_t(tokens) * hidden) {
        throw std::runtime_error("invalid Qwen3 layer input size");
    }
    std::vector<half> host(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i]) || std::fabs(input[i]) > 65504) {
            throw std::runtime_error("Qwen3 layer input outside finite FP16 range");
        }
        host[i] = __float2half_rn(input[i]);
    }
    // Find 会检查设备绑定；任何算子开始前先验证当前设备与权重一致。
    r.Weight("model.layers." + std::to_string(layer) + ".input_layernorm.weight", {hidden});
    auto x = static_cast<half*>(r.Allocate(input.size() * 2));
    auto workspace = r.Prepare(tokens, hidden, stage);
    Cuda(cudaMemcpyAsync(x, host.data(), host.size() * 2, cudaMemcpyHostToDevice, r.stream));
    r.RunLayer(layer, tokens, positionBegin, stage, x, workspace);
    Cuda(cudaMemcpyAsync(host.data(), x, host.size() * 2, cudaMemcpyDeviceToHost, r.stream));
    r.Synchronize();
    std::vector<float> result(host.size());
    for (size_t i = 0; i < host.size(); ++i) {
        result[i] = __half2float(host[i]);
        if (!std::isfinite(result[i])) {
            throw std::runtime_error("Qwen3 layer produced nonfinite output");
        }
    }
    r.ReleaseBuffers();
    return result;
}

std::vector<float> Qwen3LayerExecutor::Prefill(const std::vector<uint32_t> &tokenIds,
                                             uint32_t positionBegin) {
    return RunTokens(tokenIds, positionBegin, false);
}

std::vector<float> Qwen3LayerExecutor::RunTokens(const std::vector<uint32_t> &tokenIds,
                                               uint32_t positionBegin, bool cached) {
    auto &r = *impl;
    const auto &p = r.weights.Plan();
    if (tokenIds.empty() || tokenIds.size() > 128) {
        throw std::runtime_error("invalid Qwen3 prefill token count");
    }
    r.Validate(0, tokenIds.size(), p.hiddenSize, positionBegin, Qwen3LayerStage::Full);
    if (cached && (!r.request.active || r.request.tokens > r.request.capacity ||
        tokenIds.size() > r.request.capacity - r.request.tokens ||
        positionBegin != r.request.positionBegin + r.request.tokens)) {
        throw std::runtime_error("invalid Qwen3 request cache append");
    }
    if (!p.vocabSize || p.vocabSize > 1048576 || !p.vocabCount ||
        p.vocabCount > (protocol::kMaxPayload - 24) / 4 ||
        p.vocabBegin > p.vocabSize || p.vocabCount > p.vocabSize - p.vocabBegin) {
        throw std::runtime_error("unsupported Qwen3 prefill vocabulary size/slice");
    }
    for (auto id : tokenIds) {
        if (id >= p.vocabSize) {
            throw std::runtime_error("Qwen3 prefill token ID outside vocabulary");
        }
    }

    // 权重查找同时验证设备绑定；不在任何 Decoder 层之间读回或上传隐藏状态。
    const auto embedding = r.Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize});
    const auto finalNorm = r.Weight("model.norm.weight", {p.hiddenSize});
    const auto lmHead = r.Weight("lm_head.weight", {p.vocabCount, p.hiddenSize});
    const int t = tokenIds.size(), h = p.hiddenSize;
    auto ids = static_cast<uint32_t*>(r.Allocate(tokenIds.size() * sizeof(uint32_t)));
    auto x = static_cast<half*>(r.Allocate(size_t(t) * h * 2));
    auto workspace = r.Prepare(t, h, Qwen3LayerStage::Full);
    auto logits = static_cast<float*>(r.Allocate(p.vocabCount * sizeof(float)));

    Cuda(cudaMemcpyAsync(ids, tokenIds.data(), tokenIds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, r.stream));
    Embed<<<(t*h+255)/256,256,0,r.stream>>>(ids, embedding, x, t*h, h);
    Cuda(cudaGetLastError());
    for (uint32_t layer = 0; layer < p.layerCount; ++layer) {
        r.RunLayer(layer, t, positionBegin, Qwen3LayerStage::Full, x, workspace, cached);
    }
    Norm<<<1,256,0,r.stream>>>(x + size_t(t - 1) * h, finalNorm, workspace.norm,
                              h, h, 1, 0, p.rmsNormEps);
    r.Linear(workspace.norm, lmHead, logits, 1, p.vocabCount, h, true);
    Cuda(cudaGetLastError());

    std::vector<float> result(p.vocabCount);
    Cuda(cudaMemcpyAsync(result.data(), logits, result.size() * sizeof(float), cudaMemcpyDeviceToHost, r.stream));
    r.Synchronize();
    for (float value : result) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Qwen3 prefill produced nonfinite logits");
        }
    }
    r.ReleaseBuffers();
    if (cached) {
        r.request.tokens += tokenIds.size();
    }
    return result;
}

std::vector<float> Qwen3LayerExecutor::BeginRequest(const std::vector<uint32_t> &tokenIds,
                                                  uint32_t capacity, uint32_t positionBegin) {
    auto &r = *impl;
    const auto &p = r.weights.Plan();
    if (r.request.active) {
        throw std::runtime_error("Qwen3 request already active");
    }
    if (!capacity || capacity > 2048 || tokenIds.empty() || tokenIds.size() > 128 ||
        tokenIds.size() > capacity || positionBegin > 1048576 || capacity - 1 > 1048576 - positionBegin) {
        throw std::runtime_error("invalid Qwen3 request capacity/position");
    }
    r.Validate(0, tokenIds.size(), p.hiddenSize, positionBegin, Qwen3LayerStage::Full);
    r.Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize});
    for (auto id : tokenIds) {
        if (id >= p.vocabSize) {
            throw std::runtime_error("Qwen3 prefill token ID outside vocabulary");
        }
    }
    const uint64_t layerBytes = uint64_t(capacity) * 2 * p.kvHeadCount * p.headDim * sizeof(half);
    if (!layerBytes || layerBytes > 256ULL * 1024 * 1024 || p.layerCount > (256ULL * 1024 * 1024) / layerBytes) {
        throw std::runtime_error("Qwen3 request KV exceeds 256 MiB");
    }

    // KV 生命周期跨调用，不能放入每次前向结束就释放的临时 buffers。
    r.kvCache.resize(p.layerCount, nullptr);
    for (auto &cache : r.kvCache) {
        Cuda(cudaMalloc(reinterpret_cast<void**>(&cache), layerBytes));
        r.request.deviceBytes += layerBytes;
    }
    r.request.active = true;
    r.request.capacity = capacity;
    r.request.positionBegin = positionBegin;
    return RunTokens(tokenIds, positionBegin, true);
}

std::vector<float> Qwen3LayerExecutor::Decode(uint32_t tokenId) {
    const auto &request = impl->request;
    if (!request.active) {
        throw std::runtime_error("Qwen3 decode requires active request");
    }
    if (request.tokens >= request.capacity) {
        throw std::runtime_error("Qwen3 request capacity exhausted");
    }
    return RunTokens({tokenId}, request.positionBegin + request.tokens, true);
}

void Qwen3LayerExecutor::EndRequest() {
    auto &r = *impl;
    r.Synchronize();
    for (auto &cache : r.kvCache) {
        Cuda(cudaFree(cache));
        cache = nullptr;
    }
    r.kvCache.clear();
    r.request = {};
}

Qwen3RequestInfo Qwen3LayerExecutor::RequestInfo() const {
    return impl->request;
}

void Qwen3LayerExecutor::Clear() {
    EndRequest();
    impl->ReleaseBuffers();
    if (impl->blas) {
        Blas(cublasDestroy(impl->blas));
        impl->blas = nullptr;
    }
}

} // namespace distributed
} // namespace fastllm
