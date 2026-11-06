#include "distributed/qwen3_layer.h"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm {
namespace distributed {
namespace {
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
 * 计算一个 token/head 的 causal attention，无 KV cache。
 *
 * QK 点积、softmax、加权 V 均 FP32 累加，输出 FP16。按全局 Q 头编号映射 KV，
 * 支持 KV 头复制及不均匀 Q 分片。共享内存只存 <=128 个概率，不创建 T*T*H 张量。
 * @param qkv FP16 [T,(qHeads+2*kvHeads)*dim]，Q/K 已归一化并旋转。
 * @param context FP16 [T,qHeads*dim] 输出。
 * @param qHeads 本 rank 的 Q 头数。
 * @param kvHeads 本 rank 的 KV 头数。
 * @param dim 头维度。
 * @param qBegin 本 rank 首个 Q 头的全局编号。
 * @param kvBegin 本 rank 首个 KV 头的全局编号。
 * @param groupSize 全局 Q 头数除以全局 KV 头数。
 */
__global__ void Attention(const half *qkv, half *context, int qHeads, int kvHeads, int dim,
                          int qBegin, int kvBegin, int groupSize) {
    __shared__ float probabilities[128];
    int head = blockIdx.x % qHeads, token = blockIdx.x / qHeads;
    int kv = (qBegin + head) / groupSize - kvBegin;
    int stride = (qHeads + 2 * kvHeads) * dim;
    for (int key = threadIdx.x; key <= token; key += blockDim.x) {
        float dot = 0;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(qkv[token * stride + head * dim + d]) *
                   __half2float(qkv[key * stride + (qHeads + kv) * dim + d]);
        }
        probabilities[key] = dot * rsqrtf(float(dim));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = probabilities[0], total = 0;
        for (int i = 1; i <= token; ++i) {
            maximum = fmaxf(maximum, probabilities[i]);
        }
        for (int i = 0; i <= token; ++i) {
            probabilities[i] = expf(probabilities[i] - maximum);
            total += probabilities[i];
        }
        for (int i = 0; i <= token; ++i) {
            probabilities[i] /= total;
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0;
        for (int key = 0; key <= token; ++key) {
            value += probabilities[key] * __half2float(qkv[key * stride + (qHeads + kvHeads + kv) * dim + d]);
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
    const auto &p = r.weights.Plan();
    if (!r.blas || layer >= p.layerCount || !tokens || tokens > 128 || hidden != p.hiddenSize ||
        !hidden || hidden > 65536 || input.size() != uint64_t(tokens) * hidden || positionBegin > 1048576 ||
        uint32_t(stage) > 2 || p.headDim % 2 || p.headDim > 256 || !p.headDim ||
        (p.queryHeadCount + 2 * p.kvHeadCount) * p.headDim > 65536 || p.intermediateCount > 262144) {
        throw std::runtime_error("invalid/unsupported Qwen3 layer shape, stage or position");
    }
    if (!p.forwardUnsupportedReason.empty() || p.ropeTheta <= 1) {
        throw std::runtime_error("unsupported Qwen3 forward configuration: " + p.forwardUnsupportedReason);
    }
    const int t = tokens, h = hidden, d = p.headDim, qh = p.queryHeadCount, kh = p.kvHeadCount;
    const int qw = qh * d, kw = kh * d, qkvWidth = qw + 2 * kw, m = p.intermediateCount;
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    const bool attention = stage != Qwen3LayerStage::Mlp, mlp = stage != Qwen3LayerStage::Attention;
    std::vector<half> host(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i]) || std::fabs(input[i]) > 65504) {
            throw std::runtime_error("Qwen3 layer input outside finite FP16 range");
        }
        host[i] = __float2half_rn(input[i]);
    }
    // Find 会检查设备绑定；任何算子开始前先验证当前设备与权重一致。
    r.Weight(prefix + "input_layernorm.weight", {hidden});
    auto x = static_cast<half*>(r.Allocate(input.size() * 2));
    auto norm = static_cast<half*>(r.Allocate(input.size() * 2));
    auto partial = static_cast<float*>(r.Allocate(input.size() * 4));
    Cuda(cudaMemcpyAsync(x, host.data(), host.size() * 2, cudaMemcpyHostToDevice, r.stream));

    if (attention) {
        auto qkv = static_cast<half*>(r.Allocate(size_t(t) * qkvWidth * 2));
        auto context = static_cast<half*>(r.Allocate(size_t(t) * qw * 2));
        Norm<<<t,256,0,r.stream>>>(x, r.Weight(prefix + "input_layernorm.weight", {hidden}), norm, h, h, 1, 0, p.rmsNormEps);
        r.Linear(norm, r.Weight(prefix + "self_attn.mergeqkv.weight", {uint64_t(qkvWidth), hidden}), qkv, t, qkvWidth, h, false);
        if (auto bias = r.Weight(prefix + "self_attn.mergeqkv.bias", {uint64_t(qkvWidth)}, true)) {
            Bias<<<(t*qkvWidth+255)/256,256,0,r.stream>>>(qkv, bias, t*qkvWidth, qkvWidth);
        }
        Norm<<<t*qh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.q_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, qh, 0, p.rmsNormEps);
        Norm<<<t*kh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.k_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, kh, qw, p.rmsNormEps);
        Rope<<<t*(qh+kh),256,0,r.stream>>>(qkv, qh, kh, d, qkvWidth, positionBegin, p.ropeTheta);
        Attention<<<t*qh,256,0,r.stream>>>(qkv, context, qh, kh, d, p.queryHeadBegin, p.kvHeadBegin, p.queryHeads / p.kvHeads);
        r.Linear(context, r.Weight(prefix + "self_attn.o_proj.weight", {hidden, uint64_t(qw)}), partial, t, h, qw, true);
        if (auto bias = r.Weight(prefix + "self_attn.o_proj.bias", {hidden}, true)) {
            Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
        }
        Cuda(cudaGetLastError());
        Nccl(ncclAllReduce(partial, partial, input.size(), ncclFloat, ncclSum, r.comm, r.stream));
        Residual<<<(t*h+255)/256,256,0,r.stream>>>(x, partial, t*h);
    }
    if (mlp) {
        auto gateup = static_cast<half*>(r.Allocate(size_t(t) * 2 * m * 2));
        auto activated = static_cast<half*>(r.Allocate(size_t(t) * m * 2));
        Norm<<<t,256,0,r.stream>>>(x, r.Weight(prefix + "post_attention_layernorm.weight", {hidden}), norm, h, h, 1, 0, p.rmsNormEps);
        r.Linear(norm, r.Weight(prefix + "mlp.gateup_proj.weight", {uint64_t(2*m), hidden}), gateup, t, 2*m, h, false);
        Swiglu<<<(t*m+255)/256,256,0,r.stream>>>(gateup, activated, t*m, m);
        r.Linear(activated, r.Weight(prefix + "mlp.down_proj.weight", {hidden, uint64_t(m)}), partial, t, h, m, true);
        if (auto bias = r.Weight(prefix + "mlp.down_proj.bias", {hidden}, true)) {
            Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
        }
        Cuda(cudaGetLastError());
        Nccl(ncclAllReduce(partial, partial, input.size(), ncclFloat, ncclSum, r.comm, r.stream));
        Residual<<<(t*h+255)/256,256,0,r.stream>>>(x, partial, t*h);
    }
    Cuda(cudaGetLastError());
    Cuda(cudaMemcpyAsync(host.data(), x, host.size() * 2, cudaMemcpyDeviceToHost, r.stream));
    Cuda(cudaStreamSynchronize(r.stream));
    ncclResult_t asynchronous;
    Nccl(ncclCommGetAsyncError(r.comm, &asynchronous));
    Nccl(asynchronous);
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

void Qwen3LayerExecutor::Clear() {
    impl->ReleaseBuffers();
    if (impl->blas) {
        Blas(cublasDestroy(impl->blas));
        impl->blas = nullptr;
    }
}

} // namespace distributed
} // namespace fastllm
