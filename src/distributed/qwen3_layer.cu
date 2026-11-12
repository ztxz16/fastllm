#include "distributed/qwen3_limits.h"
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
 * @param decodeMeta 可空的 GPU uint32[2]，依次为历史长度和本次 RoPE 位置；Graph 重放时读取。
 */
__global__ void Rope(half *qkv, int qHeads, int kvHeads, int dim, int stride,
                     int positionBegin, float theta, const uint32_t *decodeMeta = nullptr) {
    if (decodeMeta) positionBegin = decodeMeta[1];
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
 * @param decodeMeta 可空的 GPU uint32[2]；非空时以第零项覆盖 past。
 */
__global__ void AppendKV(const half *qkv, half *cache, int count, int width, int qWidth, int past, const uint32_t *decodeMeta = nullptr) {
    if (decodeMeta) past = decodeMeta[0];
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
 * @param decodeMeta 可空的 GPU uint32[2]；Graph 重放时读取实际历史长度。
 */
__global__ void Attention(const half *qkv, half *context, int qHeads, int kvHeads, int dim,
                          int qBegin, int kvBegin, int groupSize, const half *cache, int past, const uint32_t *decodeMeta = nullptr) {
    if (decodeMeta) past = decodeMeta[0];
    __shared__ float probabilities[kQwen3MaxContextTokens];
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
    ProcessAllReduce &allReduce;
    cublasHandle_t blas = nullptr;
    std::vector<void*> buffers;
    size_t allocated = 0;
    std::vector<half*> kvCache;
    Qwen3RequestInfo request;
    Qwen3GraphInfo graphInfo;
    Qwen3PrefixInfo prefixInfo;
    struct PrefixEntry {
        std::vector<uint32_t> ids;
        uint32_t position = 0;
        std::vector<half*> layers;
    };
    std::vector<PrefixEntry> prefixes; // 最久未使用在前；快照永不被 Decode 写入。
    std::vector<void*> graphBuffers;
    size_t graphAllocated = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    uint32_t *graphIds = nullptr, *graphMeta = nullptr;
    half *graphX = nullptr;
    float *graphLogits = nullptr;

    void *Allocate(size_t bytes, bool persistent = false) {
        auto &pool = persistent ? graphBuffers : buffers;
        auto &used = persistent ? graphAllocated : allocated;
        if (bytes > 256ULL * 1024 * 1024 - used) {
            throw std::runtime_error("Qwen3 layer workspace exceeds 256 MiB");
        }
        pool.push_back(nullptr);
        Cuda(cudaMalloc(&pool.back(), bytes));
        used += bytes;
        return pool.back();
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
    Workspace graphWorkspace;

    /** @return 每层每 token 的 rank-local FP16 [2,kvHeads,headDim] 字节数。 */
    uint64_t KVTokenBytes() const {
        const auto &p = weights.Plan();
        return 2 * p.kvHeadCount * p.headDim * sizeof(half);
    }

    /**
     * 同步释放一条快照；调用方须确保拷贝和前向已经完成。
     * @param index LRU 容器中的有效下标；只改变快照资源，不改变活动请求。
     */
    void ErasePrefix(size_t index) {
        auto &entry = prefixes.at(index);
        for (auto &layer : entry.layers) {
            Cuda(cudaFree(layer));
            layer = nullptr;
        }
        prefixInfo.deviceBytes -= entry.ids.size() * KVTokenBytes() * weights.Plan().layerCount;
        prefixes.erase(prefixes.begin() + index);
        prefixInfo.entries = prefixes.size();
    }

    /**
     * 恢复最长公共前缀，至少留下一个 token 重算末位置 logits。
     *
     * 各 rank 在进入层间规约前用 NCCL 核对复用长度；不一致即失败，不能静默降级。
     * 模型/布局由执行器生命周期隔离，键另外包含完整 ID 前缀和 RoPE 起点。
     * @param ids 已验证的 1..2048 个 Prompt ID。
     * @param position 当前 Prompt 的 RoPE 起点。
     * @return 实际复制的 token 数；复制到独立请求 KV，不共享可写地址。
     */
    uint32_t RestorePrefix(const std::vector<uint32_t> &ids, uint32_t position) {
        if (!prefixInfo.enabled) return 0;
        uint32_t reused = 0;
        size_t selected = 0;
        for (size_t i = 0; i < prefixes.size(); ++i) {
            const auto &entry = prefixes[i];
            if (entry.position != position) continue;
            uint32_t common = 0;
            const size_t limit = std::min(entry.ids.size(), ids.size() - 1);
            while (common < limit && entry.ids[common] == ids[common]) ++common;
            if (common > reused) { reused = common; selected = i; }
        }

        const auto &p = weights.Plan();
        auto local = static_cast<uint32_t*>(Allocate(sizeof(uint32_t)));
        auto peers = static_cast<uint32_t*>(Allocate(p.worldSize * sizeof(uint32_t)));
        Cuda(cudaMemcpyAsync(local, &reused, sizeof(reused), cudaMemcpyHostToDevice, stream));
        Nccl(ncclAllGather(local, peers, 1, ncclUint32, comm, stream));
        std::vector<uint32_t> decisions(p.worldSize);
        Cuda(cudaMemcpyAsync(decisions.data(), peers, decisions.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        Synchronize();
        ReleaseBuffers();
        for (auto decision : decisions) {
            if (decision != reused) throw std::runtime_error("inconsistent Qwen3 prefix cache ranks");
        }

        if (reused) {
            auto &entry = prefixes[selected];
            for (size_t layer = 0; layer < kvCache.size(); ++layer) {
                Cuda(cudaMemcpyAsync(kvCache[layer], entry.layers[layer], reused * KVTokenBytes(), cudaMemcpyDeviceToDevice, stream));
            }
            std::rotate(prefixes.begin() + selected, prefixes.begin() + selected + 1, prefixes.end());
        }
        ++prefixInfo.lookups;
        prefixInfo.hits += reused != 0;
        prefixInfo.reusedTokens += reused;
        prefixInfo.lastReusedTokens = reused;
        prefixInfo.lastComputedTokens = ids.size() - reused;
        return reused;
    }

    /**
     * 在成功 Prefill 后保存不可变 Prompt KV，后续生成取消仍可复用该有效快照。
     *
     * 每条各层布局为 FP16 [promptTokens,2,localKVHeads,headDim]。最多 8 条；
     * 使用全局 KV 头数保守计费，保证所有 rank 的 64 MiB 淘汰决策完全相同。
     * 超过单条预算时不插入；分配或复制错误终止 worker，不假装缓存成功。
     * @param ids 已完成前向的原始 Prompt ID，不包含 Decode token。
     * @param position Prompt 的 RoPE 起点。
     */
    void SavePrefix(const std::vector<uint32_t> &ids, uint32_t position) {
        if (!prefixInfo.enabled) return;
        for (size_t i = 0; i < prefixes.size(); ++i) {
            if (prefixes[i].position == position && prefixes[i].ids == ids) {
                std::rotate(prefixes.begin() + i, prefixes.begin() + i + 1, prefixes.end());
                return;
            }
        }
        const auto &p = weights.Plan();
        constexpr uint64_t budget = 64ULL * 1024 * 1024;
        const uint64_t charge = 2 * p.kvHeads * p.headDim * sizeof(half) * p.layerCount;
        if (ids.size() > budget / charge) return;
        uint64_t tokens = ids.size();
        for (const auto &entry : prefixes) tokens += entry.ids.size();
        while (!prefixes.empty() && (prefixes.size() >= 8 || tokens > budget / charge)) {
            tokens -= prefixes.front().ids.size();
            ErasePrefix(0);
            ++prefixInfo.evictions;
        }

        prefixes.push_back({ids, position, std::vector<half*>(p.layerCount, nullptr)});
        auto &entry = prefixes.back();
        const uint64_t bytes = ids.size() * KVTokenBytes();
        for (size_t layer = 0; layer < entry.layers.size(); ++layer) {
            Cuda(cudaMalloc(reinterpret_cast<void**>(&entry.layers[layer]), bytes));
            Cuda(cudaMemcpyAsync(entry.layers[layer], kvCache[layer], bytes, cudaMemcpyDeviceToDevice, stream));
        }
        Synchronize();
        prefixInfo.deviceBytes += bytes * p.layerCount;
        prefixInfo.entries = prefixes.size();
    }

    /** 在分配和算子执行前检查单层与完整 prefill 共用的布局约束。 */
    void Validate(uint32_t layer, uint32_t tokens, uint32_t hidden, uint32_t positionBegin,
                  Qwen3LayerStage stage) const {
        const auto &p = weights.Plan();
        if (!blas || layer >= p.layerCount || !tokens || tokens > kQwen3MaxPromptTokens || hidden != p.hiddenSize ||
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
    Workspace Prepare(uint32_t tokens, uint32_t hidden, Qwen3LayerStage stage, bool persistent = false) {
        const auto &p = weights.Plan();
        Workspace w;
        w.norm = static_cast<half*>(Allocate(size_t(tokens) * hidden * 2, persistent));
        w.partial = static_cast<float*>(Allocate(size_t(tokens) * hidden * 4, persistent));
        if (stage != Qwen3LayerStage::Mlp) {
            w.qkv = static_cast<half*>(Allocate(size_t(tokens) * (p.queryHeadCount + 2 * p.kvHeadCount) * p.headDim * 2, persistent));
            w.context = static_cast<half*>(Allocate(size_t(tokens) * p.queryHeadCount * p.headDim * 2, persistent));
        }
        if (stage != Qwen3LayerStage::Attention) {
            w.gateup = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 4, persistent));
            w.activated = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 2, persistent));
        }
        return w;
    }

    /**
     * 在 GPU 原地更新一层隐藏状态；单层检查与全模型共用此计算和通信路径。
     * @param layer 层号，调用方已验证。
     * @param tokens 序列长度，范围 1..2048。
     * @param positionBegin 连续位置起点。
     * @param stage 指定 Attention、MLP 或完整层。
     * @param x FP16 [tokens,hidden] 行主序输入/输出，不读回 CPU。
     * @param w 当前请求的 GPU 工作区，同 stream 顺序复用。
     * @param cached true 表示写入当前层 KV，并在 Attention 中读取有效历史。
     * @param decodeMeta Graph 的 GPU uint32[2] 动态位置；为空时沿用 Eager 的值参数。
     */
    void RunLayer(uint32_t layer, uint32_t tokens, uint32_t positionBegin, Qwen3LayerStage stage,
                  half *x, const Workspace &w, bool cached = false, const uint32_t *decodeMeta = nullptr) {
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
            Rope<<<t*(qh+kh),256,0,r.stream>>>(qkv, qh, kh, d, qkvWidth, positionBegin, p.ropeTheta, decodeMeta);
            half *cache = cached ? kvCache.at(layer) : nullptr;
            const int past = cached ? request.tokens : 0;
            if (cached) {
                AppendKV<<<(t*2*kw+255)/256,256,0,r.stream>>>(qkv, cache, t*2*kw, 2*kw, qw, past, decodeMeta);
            }
            Attention<<<t*qh,256,0,r.stream>>>(qkv, context, qh, kh, d, p.queryHeadBegin, p.kvHeadBegin,
                                              p.queryHeads / p.kvHeads, cache, past, decodeMeta);
            r.Linear(context, r.Weight(prefix + "self_attn.o_proj.weight", {hidden, uint64_t(qw)}), partial, t, h, qw, true);
            if (auto bias = r.Weight(prefix + "self_attn.o_proj.bias", {hidden}, true)) {
                Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
            }
            Cuda(cudaGetLastError());
            r.allReduce.Sum(partial, size_t(t) * h);
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
            r.allReduce.Sum(partial, size_t(t) * h);
            Residual<<<(t*h+255)/256,256,0,r.stream>>>(x, partial, t*h);
        }
        Cuda(cudaGetLastError());
    }

    /**
     * 提交全模型计算，供 Eager 与 Graph 共用；不分配、同步或读回 CPU。
     * @param ids GPU uint32[t] token ID。
     * @param x FP16 [t,hidden] 隐藏状态工作区。
     * @param workspace 与 stream 同生命周期的算子工作区。
     * @param logits FP32 [local_vocab] 最后一个 token 的分片 logits。
     * @param t 本轮 token 数；Graph 固定为 1。
     * @param positionBegin Eager 的绝对位置起点。
     * @param cached 是否读写请求 KV。
     * @param decodeMeta Graph 的 GPU uint32[2]={历史长度,绝对位置}，Eager 为空。
     */
    void EnqueueTokens(const uint32_t *ids, half *x, const Workspace &workspace,
                       float *logits, int t, uint32_t positionBegin, bool cached,
                       const uint32_t *decodeMeta = nullptr) {
        const auto &p = weights.Plan();
        const int h = p.hiddenSize;
        auto embedding = Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize});
        auto finalNorm = Weight("model.norm.weight", {p.hiddenSize});
        auto lmHead = Weight("lm_head.weight", {p.vocabCount, p.hiddenSize});
        Embed<<<(t*h+255)/256,256,0,stream>>>(ids, embedding, x, t*h, h);
        Cuda(cudaGetLastError());
        for (uint32_t layer = 0; layer < p.layerCount; ++layer) {
            RunLayer(layer, t, positionBegin, Qwen3LayerStage::Full, x, workspace, cached, decodeMeta);
        }
        Norm<<<1,256,0,stream>>>(x + size_t(t - 1) * h, finalNorm, workspace.norm,
                                h, h, 1, 0, p.rmsNormEps);
        Linear(workspace.norm, lmHead, logits, 1, p.vocabCount, h, true);
        Cuda(cudaGetLastError());
    }

    /** 销毁请求图后释放其固定地址；调用方须先同步 stream，失败由 worker 整组退出。 */
    void ReleaseGraph() {
        if (graphExec) {
            Cuda(cudaGraphExecDestroy(graphExec));
            graphExec = nullptr;
        }
        if (graph) {
            Cuda(cudaGraphDestroy(graph));
            graph = nullptr;
        }
        if (!graphBuffers.empty()) {
            // SetStream 将 cuBLAS 的用户工作区恢复为默认值，避免释放后继续引用。
            Blas(cublasSetStream(blas, stream));
        }
        for (auto &buffer : graphBuffers) {
            Cuda(cudaFree(buffer));
            buffer = nullptr;
        }
        graphBuffers.clear();
        graphAllocated = 0;
        graphIds = graphMeta = nullptr;
        graphX = nullptr;
        graphLogits = nullptr;
        graphWorkspace = {};
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

Qwen3LayerExecutor::Qwen3LayerExecutor(const Qwen3DeviceWeights &weights, cudaStream_t stream, ncclComm_t comm, ProcessAllReduce &allReduce, bool cudaGraph, bool prefixCache)
    : impl(new Impl{weights, stream, comm, allReduce}) {
    if (!stream || !comm) {
        throw std::runtime_error("Qwen3 layer requires initialized stream/NCCL");
    }
    impl->graphInfo.enabled = cudaGraph;
    impl->prefixInfo.enabled = prefixCache;
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
    if (tokenIds.empty() || tokenIds.size() > kQwen3MaxPromptTokens) {
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

    // 在分配前验证权重绑定设备；共享提交路径内继续核对全部布局。
    r.Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize});
    const int t = tokenIds.size(), h = p.hiddenSize;
    auto ids = static_cast<uint32_t*>(r.Allocate(tokenIds.size() * sizeof(uint32_t)));
    auto x = static_cast<half*>(r.Allocate(size_t(t) * h * 2));
    auto workspace = r.Prepare(t, h, Qwen3LayerStage::Full);
    auto logits = static_cast<float*>(r.Allocate(p.vocabCount * sizeof(float)));

    Cuda(cudaMemcpyAsync(ids, tokenIds.data(), tokenIds.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, r.stream));
    r.EnqueueTokens(ids, x, workspace, logits, t, positionBegin, cached);

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
    if (!capacity || capacity > kQwen3MaxContextTokens || tokenIds.empty() || tokenIds.size() > kQwen3MaxPromptTokens ||
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
    r.request.tokens = r.RestorePrefix(tokenIds, positionBegin);
    std::vector<uint32_t> suffix(tokenIds.begin() + r.request.tokens, tokenIds.end());
    auto result = RunTokens(suffix, positionBegin + r.request.tokens, true);
    r.SavePrefix(tokenIds, positionBegin);
    return result;
}

std::vector<float> Qwen3LayerExecutor::Decode(uint32_t tokenId) {
    const auto &request = impl->request;
    if (!request.active) {
        throw std::runtime_error("Qwen3 decode requires active request");
    }
    if (request.tokens >= request.capacity) {
        throw std::runtime_error("Qwen3 request capacity exhausted");
    }
    if (tokenId >= impl->weights.Plan().vocabSize) {
        throw std::runtime_error("Qwen3 prefill token ID outside vocabulary");
    }
    if (impl->graphInfo.enabled) return DecodeGraph(tokenId);
    auto result = RunTokens({tokenId}, request.positionBegin + request.tokens, true);
    ++impl->graphInfo.eagerDecodes;
    return result;
}

std::vector<float> Qwen3LayerExecutor::DecodeGraph(uint32_t tokenId) {
    auto &r = *impl;
    const auto &p = r.weights.Plan();
    const uint32_t meta[] = {r.request.tokens, r.request.positionBegin + r.request.tokens};
    r.Validate(0, 1, p.hiddenSize, meta[1], Qwen3LayerStage::Full);
    r.Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize});

    if (!r.graphExec) {
        r.graphIds = static_cast<uint32_t*>(r.Allocate(sizeof(uint32_t), true));
        r.graphMeta = static_cast<uint32_t*>(r.Allocate(sizeof(meta), true));
        r.graphX = static_cast<half*>(r.Allocate(size_t(p.hiddenSize) * sizeof(half), true));
        r.graphWorkspace = r.Prepare(1, p.hiddenSize, Qwen3LayerStage::Full, true);
        r.graphLogits = static_cast<float*>(r.Allocate(p.vocabCount * sizeof(float), true));
        constexpr size_t blasWorkspaceBytes = 4 * 1024 * 1024;
        Blas(cublasSetWorkspace(r.blas, r.Allocate(blasWorkspaceBytes, true), blasWorkspaceBytes));
    }
    Cuda(cudaMemcpyAsync(r.graphIds, &tokenId, sizeof(tokenId), cudaMemcpyHostToDevice, r.stream));
    Cuda(cudaMemcpyAsync(r.graphMeta, meta, sizeof(meta), cudaMemcpyHostToDevice, r.stream));

    if (!r.graphExec) {
        // warmup 只覆盖当前 KV 槽位，不推进有效长度；重放重新从 embedding 开始计算。
        r.EnqueueTokens(r.graphIds, r.graphX, r.graphWorkspace, r.graphLogits, 1, meta[1], true, r.graphMeta);
        r.Synchronize();
        ++r.graphInfo.warmups;
        Cuda(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeThreadLocal));
        try {
            r.EnqueueTokens(r.graphIds, r.graphX, r.graphWorkspace, r.graphLogits, 1, meta[1], true, r.graphMeta);
        } catch (...) {
            // 即便捕获失效，也必须结束捕获；失败不允许退回 Eager。
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(r.stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            throw;
        }
        Cuda(cudaStreamEndCapture(r.stream, &r.graph));
        Cuda(cudaGraphInstantiate(&r.graphExec, r.graph, nullptr, nullptr, 0));
        ++r.graphInfo.captures;
    }

    Cuda(cudaGraphLaunch(r.graphExec, r.stream));
    std::vector<float> result(p.vocabCount);
    Cuda(cudaMemcpyAsync(result.data(), r.graphLogits, result.size() * sizeof(float), cudaMemcpyDeviceToHost, r.stream));
    r.Synchronize();
    for (float value : result) {
        if (!std::isfinite(value)) throw std::runtime_error("Qwen3 CUDA Graph produced nonfinite logits");
    }
    ++r.graphInfo.replays;
    ++r.request.tokens;
    return result;
}

void Qwen3LayerExecutor::EndRequest() {
    auto &r = *impl;
    r.Synchronize();
    r.ReleaseGraph();
    for (auto &cache : r.kvCache) {
        Cuda(cudaFree(cache));
        cache = nullptr;
    }
    r.kvCache.clear();
    r.request = {};
}

Qwen3RequestInfo Qwen3LayerExecutor::RequestInfo() const {
    auto info = impl->request;
    info.graph = impl->graphInfo;
    info.graph.active = impl->graphExec != nullptr;
    info.graph.workspaceBytes = impl->graphAllocated;
    info.prefix = impl->prefixInfo;
    for (const auto &entry : impl->prefixes) {
        info.prefix.snapshots.emplace_back(entry.position, entry.ids);
    }
    return info;
}

void Qwen3LayerExecutor::ClearPrefixCache() {
    auto &r = *impl;
    r.Synchronize();
    while (!r.prefixes.empty()) r.ErasePrefix(0);
    if (r.prefixInfo.enabled) ++r.prefixInfo.clears;
}

void Qwen3LayerExecutor::Clear() {
    EndRequest();
    ClearPrefixCache();
    impl->ReleaseBuffers();
    if (impl->blas) {
        Blas(cublasDestroy(impl->blas));
        impl->blas = nullptr;
    }
}

} // namespace distributed
} // namespace fastllm
