#include "distributed/qwen3_limits.h"
#include "distributed/qwen3_layer.h"
#include "process_protocol.h"
#include "qwen_moe_qkv.cuh"
#include "qwen_moe_trace.cuh"
#include "qwen_moe_block_trace.cuh"

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include "qwen3_paged_cache.cuh"

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

/**
 * 将独立连续 Q/K/V 投影逐位拷贝到 Attention 使用的行交错布局。
 * @param q FP16 [T,qWidth]；投影完成后只读，不执行额外类型转换。
 * @param k FP16 [T,kvWidth]。
 * @param v FP16 [T,kvWidth]。
 * @param qkv FP16 [T,qWidth+2*kvWidth] 输出，与三个输入不重叠。
 * @param count T*(qWidth+2*kvWidth)，已由调用方限制在 int 范围内。
 * @param qWidth Q 通道数。
 * @param kvWidth K 和 V 各自的通道数，允许与 qWidth 不同。
 */
__global__ void PackMoeQkv(const half *q, const half *k, const half *v, half *qkv,
                          int count, int qWidth, int kvWidth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int width = qWidth + 2 * kvWidth, row = i / width, col = i % width;
    if (col < qWidth) qkv[i] = q[row * qWidth + col];
    else if (col < qWidth + kvWidth) qkv[i] = k[row * kvWidth + col - qWidth];
    else qkv[i] = v[row * kvWidth + col - qWidth - kvWidth];
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
 * QK 点积、softmax、加权 V 均 FP32 累加，输出 FP16。MoE 另保留 HF Eager 的
 * 点积、缩放和概率 FP16 舍入边界，避免更高精度的中间值改变后续专家路由。
 * 按全局 Q 头编号映射 KV，
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
 * @param fp16Attention true 对齐 Qwen2 MoE Eager 的 FP16 中间张量；false 保留 dense 路径。
 * @param decodeMeta 可空的 GPU uint32[2]；Graph 重放时读取实际历史长度。
 */
__global__ void Attention(const half *qkv, half *context, int qHeads, int kvHeads, int dim,
                          int qBegin, int kvBegin, int groupSize, const half *cache, int past,
                          bool fp16Attention, const uint32_t *decodeMeta = nullptr) {
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
        if (fp16Attention) {
            // HF 先得到 FP16 matmul 结果，再执行独立缩放；不能合并两次舍入。
            dot = __half2float(__float2half_rn(dot));
            probabilities[key] = __half2float(__float2half_rn(dot * rsqrtf(float(dim))));
        } else {
            probabilities[key] = dot * rsqrtf(float(dim));
        }
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
            if (fp16Attention) probabilities[i] = __half2float(__float2half_rn(probabilities[i]));
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

/**
 * 将 bias 广播到 GEMM 的 C 矩阵，由 beta=1 在累加器中相加后统一转换输出类型。
 *
 * 此处只复制 FP16 bias，不读取旧输出；随后同 stream 的 GEMM 覆盖整个矩阵。
 * 避免先将点积转 FP16、再单独加 bias 的二次舍入，支持 Graph 重放复用缓冲。
 * @param values 行主序 [tokens,width]，T 为 half 或 float。
 * @param bias FP16 [width]，只读且不与 values 重叠。
 * @param count tokens*width，已由调用方检查边界。
 * @param width 输出通道数。
 */
template<class T>
__global__ void BroadcastBias(T *values, const half *bias, int count, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        values[i] = T(__half2float(bias[i % width]));
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
#include "qwen_moe_kernels.cuh"
} // namespace

struct Qwen3LayerExecutor::Impl {
    const Qwen3DeviceWeights &weights;
    cudaStream_t stream;
    ncclComm_t comm;
    ProcessAllReduce &allReduce;
    cublasHandle_t blas = nullptr;
    cublasLtHandle_t qkvLt = nullptr;
    std::vector<std::unique_ptr<detail::MoeQkvLtPlan>> qkvPlans, graphQkvPlans;
    std::string forwardTraceDirectory;
    std::unique_ptr<detail::MoeForwardTrace> forwardTrace;
    uint64_t forwardTraceRound = 0;
    std::string blockTraceDirectory;
    std::unique_ptr<detail::MoeBlockTrace> blockTrace;
    uint64_t blockTraceRound = 0;
    std::vector<void*> buffers;
    size_t allocated = 0;
    std::vector<half*> kvCache; // MoE保留原有连续缓存。
    std::unique_ptr<detail::ProcessPagedKV> paged;
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
    uint64_t reservedKVBytes = 0;

    void *Allocate(size_t bytes, bool persistent = false) {
        auto &pool = persistent ? graphBuffers : buffers;
        auto &used = persistent ? graphAllocated : allocated;
        const size_t budget = (weights.Plan().model == ShardModel::Qwen2Moe ? 512ULL : 256ULL) * 1024 * 1024;
        const size_t total = weights.Plan().model == ShardModel::Qwen2Moe ? graphAllocated + allocated : used;
        if (total > budget || bytes > budget - total) {
            throw std::runtime_error(weights.Plan().model == ShardModel::Qwen2Moe ?
                "MoE layer workspace exceeds 512 MiB" : "Qwen3 layer workspace exceeds 256 MiB");
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

    /**
     * 执行行主序线性计算 Y[T,N]=X[T,K]*W[N,K]^T+bias[N]。
     *
     * Qwen2 MoE 使用与 HF/PyTorch 常规 FP16 GEMM 一致的 FP32 计算类型和
     * Tensor Op 算法选项。PEDANTIC 的不同累加路径会令 router 分数产生一个
     * FP16 间隔的差异，并改变 Top-K 边界。此策略覆盖一般 MoE Linear；
     * 有偏置的 QKV 改用独立 Lt 投影，以保留 HF 的矩阵形状和 epilogue。
     * Dense Qwen3 保留 PEDANTIC 策略。复用执行器的 handle、stream 和工作区，
     * 可选 bias 先广播到输出，再以 beta=1 在 GEMM 中相加，避免点积提前转
     * FP16；无 bias 仍使用 beta=0。工作区在同 stream 内覆盖，不分配或同步；
     * cuBLAS 失败直接上抛，不切换算法重试。
     *
     * @param x 连续 FP16 激活 [T,K]。
     * @param w 连续 FP16 权重 [N,K]。
     * @param y 连续输出 [T,N]，类型由 fp32 指定。
     * @param t Token 行数 T，由调用方验证为正数。
     * @param n 输出通道数 N。
     * @param k 输入通道数 K。
     * @param fp32 true 输出 FP32，用于投影中间值或 logits；false 输出 FP16。
     * @param bias 可空的 FP16 [N]；非空时在最终输出转换前相加，不与 x/w/y 重叠。
     */
    void Linear(const half *x, const half *w, void *y, int t, int n, int k, bool fp32,
                const half *bias = nullptr) {
        const bool moe = weights.Plan().model == ShardModel::Qwen2Moe;
        const auto computeType = moe ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_PEDANTIC;
        const auto algorithm = moe ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
        const float alpha = 1, beta = bias ? 1 : 0;

        if (bias) {
            if (fp32) {
                BroadcastBias<<<(t*n+255)/256,256,0,stream>>>(static_cast<float*>(y), bias, t*n, n);
            } else {
                BroadcastBias<<<(t*n+255)/256,256,0,stream>>>(static_cast<half*>(y), bias, t*n, n);
            }
            Cuda(cudaGetLastError());
        }

        Blas(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, t, k, &alpha,
            w, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, y, fp32 ? CUDA_R_32F : CUDA_R_16F,
            n, computeType, algorithm));
    }

    struct Workspace {
        half *norm = nullptr, *qkv = nullptr, *context = nullptr, *gateup = nullptr, *activated = nullptr;
        float *partial = nullptr;
        half *qkvParts[3]{};
        void *qkvLtWorkspace = nullptr;
        bool persistent = false;
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
     * dense TP 用 NCCL 核对复用长度；Attention DP 的目录与命中长度彼此独立。
     * 模型/布局由执行器生命周期隔离，键另外包含完整 ID 前缀和 RoPE 起点。
     * @param ids 已验证的 1..2048 个 Prompt ID。
     * @param position 当前 Prompt 的 RoPE 起点。
     * @return 实际复用的token数；dense绑定完整只读页，MoE沿用独立请求KV的快照复制。
     */
    uint32_t RestorePrefix(const std::vector<uint32_t> &ids, uint32_t position) {
        if (!prefixInfo.enabled) return 0;
        if (paged) {
            std::vector<int> tokens(ids.begin(), ids.end());
            uint32_t local = paged->Query(tokens, position), reused = 0;
            auto values = static_cast<uint32_t*>(Allocate(2 * sizeof(uint32_t)));
            Cuda(cudaMemcpyAsync(values, &local, sizeof(local), cudaMemcpyHostToDevice, stream));
            Nccl(ncclAllReduce(values, values + 1, 1, ncclUint32, ncclMin, comm, stream));
            Cuda(cudaMemcpyAsync(&reused, values + 1, sizeof(reused), cudaMemcpyDeviceToHost, stream));
            Synchronize();
            ReleaseBuffers();
            paged->Restore(tokens, position, reused);
            ++prefixInfo.lookups;
            prefixInfo.hits += reused != 0;
            prefixInfo.reusedTokens += reused;
            prefixInfo.lastReusedTokens = reused;
            prefixInfo.lastComputedTokens = ids.size() - reused;
            return reused;
        }
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
        if (p.model == ShardModel::DenseQwen3) {
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
     * 在成功Prefill后登记有效Prompt前缀，后续生成取消仍可复用已计算的部分。
     *
     * Dense仅向FT索引登记完整页面，容量和淘汰沿用FT页池。
     * MoE保留FP16 [promptTokens,2,localKVHeads,headDim]快照及原有条数/字节预算。
     * 分配、登记或复制错误终止worker。
     * @param ids 已完成前向的原始 Prompt ID，不包含 Decode token。
     * @param position Prompt 的 RoPE 起点。
     */
    void SavePrefix(const std::vector<uint32_t> &ids, uint32_t position) {
        if (!prefixInfo.enabled) return;
        if (paged) {
            paged->Record(std::vector<int>(ids.begin(), ids.end()), position);
            return;
        }
        for (size_t i = 0; i < prefixes.size(); ++i) {
            if (prefixes[i].position == position && prefixes[i].ids == ids) {
                std::rotate(prefixes.begin() + i, prefixes.begin() + i + 1, prefixes.end());
                return;
            }
        }
        const auto &p = weights.Plan();
        const uint64_t budget = (p.model == ShardModel::Qwen2Moe ? 512ULL : 64ULL) * 1024 * 1024;
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
        w.persistent = persistent;
        w.norm = static_cast<half*>(Allocate(size_t(tokens) * hidden * 2, persistent));
        w.partial = static_cast<float*>(Allocate(size_t(tokens) * hidden * 4, persistent));
        if (stage != Qwen3LayerStage::Mlp) {
            w.qkv = static_cast<half*>(Allocate(size_t(tokens) * (p.queryHeadCount + 2 * p.kvHeadCount) * p.headDim * 2, persistent));
            w.context = static_cast<half*>(Allocate(size_t(tokens) * p.queryHeadCount * p.headDim * 2, persistent));
            if (qkvLt) {
                const uint64_t widths[] = {p.queryHeadCount * p.headDim, p.kvHeadCount * p.headDim,
                                          p.kvHeadCount * p.headDim};
                for (int i = 0; i < 3; ++i) {
                    w.qkvParts[i] = static_cast<half*>(Allocate(size_t(tokens) * widths[i] * sizeof(half), persistent));
                }
                w.qkvLtWorkspace = Allocate(detail::MoeQkvLtPlan::kWorkspaceBytes, persistent);
            }
        }
        if (stage != Qwen3LayerStage::Attention) {
            w.gateup = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 4, persistent));
            w.activated = static_cast<half*>(Allocate(size_t(tokens) * p.intermediateCount * 2, persistent));
        }
        return w;
    }

    /**
     * 执行 MoE 有偏置的独立 Q/K/V Lt 投影，再恢复现有 Attention 的 QKV 布局。
     *
     * 权重仍按行合并存储，仅切片读取；三个输出各自连续且独立 cudaMalloc 对齐。
     * 按形状和实际地址对齐缓存算法，临时与 Graph 缓存分开。Graph warmup 建立
     * 所有计划，捕获时缺计划直接失败；同 stream 复用 1 MiB 工作区，无中途同步。
     *
     * @param layer 层号，仅用于失败定位。
     * @param tokens Token 数，必须与 prepared 工作区行数相同。
     * @param weight 连续 FP16 [qWidth+2*kvWidth,hidden] 合并权重。
     * @param bias 连续 FP16 [qWidth+2*kvWidth] 合并偏置，非空。
     * @param prepared 当前 Eager 或 Graph 池拥有的缓冲；输入为 norm，输出为 qkv。
     */
    void ProjectMoeQkv(uint32_t layer, int tokens, const half *weight, const half *bias,
                       const Workspace &prepared) {
        const auto &p = weights.Plan();
        const int h = p.hiddenSize, qw = p.queryHeadCount * p.headDim, kw = p.kvHeadCount * p.headDim;
        const int widths[] = {qw, kw, kw}, offsets[] = {0, qw, qw + kw};
        auto &plans = prepared.persistent ? graphQkvPlans : qkvPlans;
        if (!qkvLt || !bias || !prepared.qkvLtWorkspace) {
            throw std::runtime_error("MoE QKV Lt workspace/handle/bias missing");
        }

        for (int i = 0; i < 3; ++i) {
            const auto partWeight = weight + size_t(offsets[i]) * h, partBias = bias + offsets[i];
            auto output = prepared.qkvParts[i];
            try {
                detail::MoeQkvLtPlan *plan = nullptr;
                for (const auto &candidate : plans) {
                    if (candidate->Matches(tokens, widths[i], h, prepared.norm, partWeight, output, partBias)) {
                        plan = candidate.get();
                        break;
                    }
                }
                if (!plan) {
                    cudaStreamCaptureStatus capture;
                    Cuda(cudaStreamIsCapturing(stream, &capture));
                    if (capture != cudaStreamCaptureStatusNone) {
                        throw std::runtime_error("QKV Lt plan missing during CUDA Graph capture");
                    }
                    plans.push_back(std::make_unique<detail::MoeQkvLtPlan>(qkvLt, tokens, widths[i], h,
                        prepared.norm, partWeight, output, partBias));
                    plan = plans.back().get();
                }
                plan->Run(qkvLt, stream, prepared.norm, partWeight, output, partBias, prepared.qkvLtWorkspace);
            } catch (const std::exception &error) {
                throw std::runtime_error("MoE layer=" + std::to_string(layer) + " projection=" +
                    std::string(1, "QKV"[i]) + ": " + error.what());
            }
        }

        const int count = tokens * (qw + 2 * kw);
        PackMoeQkv<<<(count+255)/256,256,0,stream>>>(prepared.qkvParts[0], prepared.qkvParts[1],
            prepared.qkvParts[2], prepared.qkv, count, qw, kw);
        Cuda(cudaGetLastError());
    }

    struct MoeWorkspace {
        half *logits = nullptr, *prob = nullptr, *shared = nullptr, *sharedGate = nullptr;
        int *ids = nullptr, *counts = nullptr;
        float *routed = nullptr;
        half *norm = nullptr, *gathered = nullptr, *allProb = nullptr;
        half *packed = nullptr, *gateup = nullptr, *activated = nullptr, *expertOutput = nullptr;
        int *allIds = nullptr, *map = nullptr;
        half *contributions = nullptr, *localContributions = nullptr;
    };
    MoeWorkspace graphMoe;

    /**
     * 分配 FP16 激活、独立专家贡献及路由索引，不执行前向。
     *
     * 贡献布局为 [全组行数,Top-K,hidden]，每槽仅一个 owner 写入；本地接收
     * 同布局的 [capacity,Top-K,hidden]。routed 以 FP32 保存最终 FP16 累加值。
     * 所有缓冲计入原 512 MiB 预算；Graph 使用固定地址且不在捕获中分配。
     * @param capacity 全组最大行数，至少一；等于一时使用设备路由专家内核。
     * @param persistent true 使用 Graph 固定地址池，false 使用本轮临时池。
     * @return 缓冲视图，生命周期由对应池管理；超出预算或分配失败抛异常。
     */
    MoeWorkspace PrepareMoe(int capacity, bool persistent = false) {
        const auto &p = weights.Plan();
        const int h = p.hiddenSize, e = p.expertCount, k = p.expertsPerToken;
        const int m = p.intermediateCount, sharedWidth = p.sharedIntermediateSize;
        const int rows = capacity * p.worldSize;
        auto allocHalf = [&](size_t count) { return static_cast<half*>(Allocate(count * sizeof(half), persistent)); };
        auto norm = allocHalf(size_t(capacity) * h);
        auto gathered = allocHalf(size_t(rows) * h);
        auto logits = allocHalf(size_t(capacity) * e);
        auto prob = allocHalf(size_t(capacity) * k);
        auto allProb = allocHalf(size_t(rows) * k);
        auto ids = static_cast<int*>(Allocate(size_t(capacity) * k * sizeof(int), persistent));
        auto allIds = static_cast<int*>(Allocate(size_t(rows) * k * sizeof(int), persistent));
        auto counts = static_cast<int*>(Allocate(size_t(e) * sizeof(int), persistent));
        auto map = static_cast<int*>(Allocate(size_t(e) * rows * sizeof(int), persistent));
        auto contributions = allocHalf(size_t(rows) * k * h);
        auto localContributions = allocHalf(size_t(capacity) * k * h);
        auto routed = static_cast<float*>(Allocate(size_t(capacity) * h * sizeof(float), persistent));
        const size_t channels = std::max(size_t(rows) * m, size_t(capacity) * sharedWidth);
        auto packed = allocHalf(size_t(rows) * h);
        auto gateup = allocHalf(2 * channels);
        auto activated = allocHalf(channels);
        auto expertOutput = allocHalf(size_t(rows) * h);
        auto shared = allocHalf(size_t(capacity) * h);
        auto sharedGate = allocHalf(capacity);

        return {logits, prob, shared, sharedGate, ids, counts, routed, norm, gathered,
                allProb, packed, gateup, activated, expertOutput, allIds, map, contributions, localContributions};
    }

    /**
     * 复用工作区执行 MoE，单层、Eager 和 Graph 共用本计算及通信路径。
     * @param layer 模型层号。
     * @param tokens 本地 Token 数；Graph 固定为一并通过 meta 屏蔽空闲 rank。
     * @param capacity 整组共同的 padded 行数。
     * @param x FP16 [tokens,hidden] 隐藏状态，原地加残差。
     * @param prepared 可空的预分配空间；单层诊断为空时分配临时空间。
     * @param meta 可空的设备 [past,position,active]，用于 Graph 动态路由。
     * @param traceBlock 仅正式Eager Prefill第0层采集内部张量；不改变计算或通信。
     * @return 工作区视图；CUDA/NCCL 失败须退出 worker。
     */
    MoeWorkspace RunMoe(uint32_t layer, int tokens, int capacity, half *x,
                        const MoeWorkspace *prepared = nullptr, const uint32_t *meta = nullptr,
                        bool traceBlock = false) {
        const auto &p = weights.Plan();
        const int h = p.hiddenSize, e = p.expertCount, k = p.expertsPerToken;
        const int m = p.intermediateCount, sharedWidth = p.sharedIntermediateSize;
        const int rows = capacity * p.worldSize;
        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        auto w = prepared ? *prepared : PrepareMoe(capacity);
        auto norm = w.norm, gathered = w.gathered, logits = w.logits, prob = w.prob, allProb = w.allProb;
        auto packed = w.packed, gateup = w.gateup, activated = w.activated, expertOutput = w.expertOutput;
        auto shared = w.shared, sharedGate = w.sharedGate;
        auto ids = w.ids, allIds = w.allIds, counts = w.counts, map = w.map;
        auto contributions = w.contributions, localContributions = w.localContributions;
        auto routed = w.routed;

        Cuda(cudaMemsetAsync(norm, 0, size_t(capacity) * h * sizeof(half), stream));
        Cuda(cudaMemsetAsync(logits, 0, size_t(capacity) * e * sizeof(half), stream));
        Cuda(cudaMemsetAsync(contributions, 0, size_t(rows) * k * h * sizeof(half), stream));
        if (tokens) {
            Norm<<<tokens,256,0,stream>>>(x, Weight(prefix + "post_attention_layernorm.weight", {p.hiddenSize}),
                norm, h, h, 1, 0, p.rmsNormEps);
            Linear(norm, Weight(prefix + "mlp.gate.weight", {p.expertCount, p.hiddenSize}), logits, tokens, e, h, false);
        }
        moe::Route<<<capacity,1,0,stream>>>(logits, ids, prob, tokens, capacity, e, k, p.normTopkProb, meta);
        Nccl(ncclAllGather(norm, gathered, size_t(capacity) * h, ncclHalf, comm, stream));
        Nccl(ncclAllGather(ids, allIds, size_t(capacity) * k, ncclInt32, comm, stream));
        Nccl(ncclAllGather(prob, allProb, size_t(capacity) * k, ncclHalf, comm, stream));
        moe::Index<<<(e+255)/256,256,0,stream>>>(allIds, map, counts, rows, k, e);

        std::vector<int> hostCounts(e);
        if (capacity > 1) {
            // Prefill 压紧后以实际 Token 数调用 GEMM；Decode 不读取 CPU 路由。
            Cuda(cudaMemcpyAsync(hostCounts.data(), counts, e * sizeof(int), cudaMemcpyDeviceToHost, stream));
            Synchronize();
        }
        detail::MoeBlockTrace *trace = nullptr;
        if (traceBlock) {
            if (layer != 0 || meta || capacity <= 1 || blockTraceDirectory.empty())
                throw std::runtime_error("MoE block trace requires layer zero Eager Prefill");
            blockTrace.reset(new detail::MoeBlockTrace(blockTraceDirectory, tokens, capacity, h, m,
                sharedWidth, e, k, p.rank, p.worldSize, hostCounts));
            blockTrace->Attach(Allocate(blockTrace->Bytes()));
            trace = blockTrace.get();
            if (tokens) trace->Capture("input", x, stream);
            trace->Capture("norm", norm, stream);
            trace->Capture("router_logits", logits, stream);
            trace->Capture("selected_experts", ids, stream);
            trace->Capture("routing_weights", prob, stream);
            trace->Capture("gathered_norm", gathered, stream);
            trace->Capture("gathered_experts", allIds, stream);
            trace->Capture("gathered_weights", allProb, stream);
            trace->Capture("expert_counts", counts, stream);
        }

        for (int expert = p.rank; expert < e; expert += p.worldSize) {
            std::string name = prefix + "mlp.experts." + std::to_string(expert) + ".";
            auto gate = Weight(name + "gateup_proj.weight", {uint64_t(2*m), p.hiddenSize});
            auto down = Weight(name + "down_proj.weight", {p.hiddenSize, uint64_t(m)});
            if (capacity == 1) {
                moe::DecodeGate<<<(rows*m+255)/256,256,0,stream>>>(gathered, gate, allIds, activated, rows, h, m, k, expert);
                moe::DecodeDown<<<(rows*h+255)/256,256,0,stream>>>(activated, down, allIds, allProb, contributions, rows, h, m, k, expert);
            } else if (hostCounts[expert]) {
                int count = hostCounts[expert];
                const int *indices = map + expert * rows;
                const std::string entry = trace ? "expert_" + std::to_string(expert) + "_" : "";
                if (trace) trace->Capture(entry + "indices", indices, stream);
                moe::Pack<<<(count*h+255)/256,256,0,stream>>>(gathered, indices, packed, count, h, k);
                if (trace) trace->Capture(entry + "packed", packed, stream);
                Linear(packed, gate, gateup, count, 2*m, h, false);
                if (trace) trace->Capture(entry + "gateup", gateup, stream);
                Swiglu<<<(count*m+255)/256,256,0,stream>>>(gateup, activated, count*m, m);
                if (trace) trace->Capture(entry + "activated", activated, stream);
                Linear(activated, down, expertOutput, count, h, m, false);
                if (trace) trace->Capture(entry + "down", expertOutput, stream);
                moe::Scatter<<<(count*h+255)/256,256,0,stream>>>(expertOutput, indices, allProb, contributions, count, h);
            }
        }
        // 每槽位仅一个 rank 写入有效值，通信不进行跨专家的浮点求和。
        Nccl(ncclReduceScatter(contributions, localContributions, size_t(capacity) * k * h,
                               ncclHalf, ncclSum, comm, stream));
        if (trace) trace->Capture("local_contributions", localContributions, stream);
        moe::OrderedSum<<<capacity,256,0,stream>>>(localContributions, ids, routed, h, k);
        if (trace) trace->Capture("routed", routed, stream);
        if (tokens) {
            std::string name = prefix + "mlp.shared_expert.";
            Linear(norm, Weight(name + "gateup_proj.weight", {uint64_t(2*sharedWidth), p.hiddenSize}),
                gateup, tokens, 2*sharedWidth, h, false);
            if (trace) trace->Capture("shared_gateup", gateup, stream);
            Swiglu<<<(tokens*sharedWidth+255)/256,256,0,stream>>>(gateup, activated, tokens*sharedWidth, sharedWidth);
            if (trace) trace->Capture("shared_activated", activated, stream);
            Linear(activated, Weight(name + "down_proj.weight", {p.hiddenSize, uint64_t(sharedWidth)}),
                shared, tokens, h, sharedWidth, false);
            if (trace) trace->Capture("shared_down", shared, stream);
            Linear(norm, Weight(prefix + "mlp.shared_expert_gate.weight", {1, p.hiddenSize}), sharedGate, tokens, 1, h, false);
            if (trace) trace->Capture("shared_gate", sharedGate, stream);
            moe::Finish<<<(tokens*h+255)/256,256,0,stream>>>(x, sharedGate, shared, routed, tokens*h, h, prob, k);
            if (trace) trace->Capture("shared_weighted", shared, stream);
            if (trace) trace->Capture("output", x, stream);
        }
        Cuda(cudaGetLastError());
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
            const auto bias = r.Weight(prefix + "self_attn.mergeqkv.bias", {uint64_t(qkvWidth)}, true);
            const auto weight = r.Weight(prefix + "self_attn.mergeqkv.weight", {uint64_t(qkvWidth), hidden});
            if (bias && p.model == ShardModel::Qwen2Moe) {
                r.ProjectMoeQkv(layer, t, weight, bias, w);
            } else {
                r.Linear(norm, weight, qkv, t, qkvWidth, h, false);
            }
            if (bias && p.model == ShardModel::DenseQwen3) {
                Bias<<<(t*qkvWidth+255)/256,256,0,r.stream>>>(qkv, bias, t*qkvWidth, qkvWidth);
            }
            if (p.model == ShardModel::DenseQwen3) {
                Norm<<<t*qh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.q_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, qh, 0, p.rmsNormEps);
                Norm<<<t*kh,256,0,r.stream>>>(qkv, r.Weight(prefix + "self_attn.k_norm.weight", {uint64_t(d)}), qkv, d, qkvWidth, kh, qw, p.rmsNormEps);
            }
            Rope<<<t*(qh+kh),256,0,r.stream>>>(qkv, qh, kh, d, qkvWidth, positionBegin, p.ropeTheta, decodeMeta);
            if (paged) {
                paged->Run(layer, t, qkv, context, cached, decodeMeta != nullptr);
            } else {
                half *cache = cached ? kvCache.at(layer) : nullptr;
                const int past = cached ? request.tokens : 0;
                if (cached) {
                    AppendKV<<<(t*2*kw+255)/256,256,0,r.stream>>>(qkv, cache, t*2*kw, 2*kw, qw, past, decodeMeta);
                }
                Attention<<<t*qh,256,0,r.stream>>>(qkv, context, qh, kh, d, p.queryHeadBegin, p.kvHeadBegin,
                                                  p.queryHeads / p.kvHeads, cache, past,
                                                  p.model == ShardModel::Qwen2Moe, decodeMeta);
            }
            r.Linear(context, r.Weight(prefix + "self_attn.o_proj.weight", {hidden, uint64_t(qw)}), partial, t, h, qw, true);
            if (auto bias = r.Weight(prefix + "self_attn.o_proj.bias", {hidden}, true)) {
                Bias<<<(t*h+255)/256,256,0,r.stream>>>(partial, bias, t*h, h);
            }
            Cuda(cudaGetLastError());
            if (p.model == ShardModel::DenseQwen3) r.allReduce.Sum(partial, size_t(t) * h);
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
     * 提交完整 MoE 模型；激活始终留在设备，同层专家缓冲在各层复用。
     * @param ids GPU uint32[t] 输入，空闲 Eager 不读取。
     * @param x FP16 [max(t,1),hidden] 工作区。
     * @param attention 预分配 Attention 工作区。
     * @param experts 预分配整组 EP 工作区。
     * @param logits FP32 [vocab] 输出；空闲 Eager 不写。
     * @param t 本地计算行数；Graph 为一，空闲掩码来自 meta。
     * @param capacity 整组统一的 EP 行数。
     * @param position 本轮绝对位置。
     * @param meta 可空的 GPU [past,position,active]。
     * @param trace 可空的独立数值诊断快照；仅 Eager Begin 允许，逐阶段同 stream 复制。
     */
    void EnqueueMoe(const uint32_t *ids, half *x, const Workspace &attention,
                    const MoeWorkspace &experts, float *logits, int t, int capacity,
                    uint32_t position, const uint32_t *meta = nullptr,
                    detail::MoeForwardTrace *trace = nullptr) {
        const auto &p = weights.Plan();
        const int h = p.hiddenSize;
        if (t) Embed<<<(t*h+255)/256,256,0,stream>>>(ids,
            Weight("model.embed_tokens.weight", {p.vocabSize, p.hiddenSize}), x, t*h, h);
        if (trace) trace->Capture(x, size_t(t)*h, stream);
        for (uint32_t layer = 0; layer < p.layerCount; ++layer) {
            if (t) RunLayer(layer, t, position, Qwen3LayerStage::Attention, x, attention, true, meta);
            if (trace) trace->Capture(x, size_t(t)*h, stream);
            RunMoe(layer, t, capacity, x, &experts, meta, layer == 0 && !blockTraceDirectory.empty());
            if (trace) trace->Capture(x, size_t(t)*h, stream);
        }
        if (t) {
            Norm<<<1,256,0,stream>>>(x + size_t(t-1)*h, Weight("model.norm.weight", {p.hiddenSize}),
                attention.norm, h, h, 1, 0, p.rmsNormEps);
            if (trace) trace->Capture(attention.norm, h, stream);
            Linear(attention.norm, Weight("lm_head.weight", {p.vocabSize, p.hiddenSize}),
                   logits, 1, p.vocabSize, h, true);
        }
        Cuda(cudaGetLastError());
    }

    /** 固定地址 KV 按模型预留，供 Graph 跨请求复用；分配失败由 worker 退出回收。 */
    void ReserveMoeKV() {
        if (!kvCache.empty()) return;
        const auto &p = weights.Plan();
        const uint64_t bytes = kQwen3MaxContextTokens * KVTokenBytes();
        if (!bytes || bytes > 512ULL*1024*1024 || p.layerCount > (512ULL*1024*1024)/bytes) {
            throw std::runtime_error("MoE reserved KV exceeds 512 MiB");
        }
        kvCache.resize(p.layerCount, nullptr);
        for (auto &cache : kvCache) {
            Cuda(cudaMalloc(reinterpret_cast<void**>(&cache), bytes));
            Cuda(cudaMemsetAsync(cache, 0, bytes, stream));
            reservedKVBytes += bytes;
        }
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
        graphQkvPlans.clear();
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
        if (paged) paged->ReleaseGraphScratch();
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
        forwardTrace.reset();
        blockTrace.reset();
        qkvPlans.clear();
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
    if (weights.Plan().model == ShardModel::DenseQwen3) {
        if (stream != cudaStreamPerThread) throw std::runtime_error("FT paged TP requires the worker thread default CUDA stream");
        const auto &p = weights.Plan();
        impl->paged.reset(new detail::ProcessPagedKV(p.layerCount, p.queryHeadCount, p.kvHeadCount, p.headDim));
        impl->prefixInfo.directoryVersion = 2;
    }
    if (const char *trace = std::getenv("FASTLLM_MOE_TRACE_DIR")) {
        if (*trace) {
            if (weights.Plan().model != ShardModel::Qwen2Moe || cudaGraph || prefixCache) {
                throw std::runtime_error("MoE forward trace requires Eager with Prefix disabled");
            }
            impl->forwardTraceDirectory = trace;
        }
    }
    if (const char *trace = std::getenv("FASTLLM_MOE_BLOCK_TRACE_DIR")) {
        if (*trace) {
            if (weights.Plan().model != ShardModel::Qwen2Moe || cudaGraph || prefixCache ||
                !impl->forwardTraceDirectory.empty()) {
                throw std::runtime_error("MoE block trace requires Eager, no Prefix and no forward trace");
            }
            impl->blockTraceDirectory = trace;
        }
    }
    Blas(cublasCreate(&impl->blas));
    Blas(cublasSetStream(impl->blas, stream));
    Blas(cublasSetMathMode(impl->blas, CUBLAS_DEFAULT_MATH));
    if (weights.Plan().model == ShardModel::Qwen2Moe) {
        for (const auto &tensor : weights.Plan().tensors) {
            if (tensor.name.find(".self_attn.mergeqkv.bias") != std::string::npos) {
                Blas(cublasLtCreate(&impl->qkvLt));
                break;
            }
        }
    }
}

/** 正常卸载由Clear释放；异常析构不调用可能阻塞的CUDA清理，由worker退出回收上下文。 */
Qwen3LayerExecutor::~Qwen3LayerExecutor() {
    if (impl) (void)impl->paged.release();
}

std::vector<float> Qwen3LayerExecutor::Forward(uint32_t layer, uint32_t tokens, uint32_t hidden,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input) {
    auto &r = *impl;
    if (r.weights.Plan().model != ShardModel::DenseQwen3) throw std::runtime_error("dense layer requires Qwen3 weights");
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

QwenMoeLayerResult Qwen3LayerExecutor::ForwardMoe(uint32_t layer, uint32_t tokens, uint32_t capacity,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input) {
    auto &r = *impl;
    const auto &p = r.weights.Plan();
    if (p.model != ShardModel::Qwen2Moe || tokens > capacity || input.size() != uint64_t(tokens) * p.hiddenSize ||
        r.request.active || r.graphInfo.enabled || r.prefixInfo.enabled) {
        throw std::runtime_error("invalid MoE layer model/input/state");
    }
    r.Validate(layer, capacity, p.hiddenSize, positionBegin, stage);
    const uint64_t responseElements = uint64_t(tokens) * (3*p.hiddenSize + p.expertCount + 2*p.expertsPerToken) + p.expertCount;
    if (responseElements > (protocol::kMaxPayload - 40) / 4) throw std::runtime_error("MoE diagnostic response exceeds protocol limit");
    r.Weight("model.layers." + std::to_string(layer) + ".input_layernorm.weight", {p.hiddenSize});
    std::vector<half> host(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i]) || std::fabs(input[i]) > 65504) throw std::runtime_error("invalid MoE FP16 input");
        host[i] = __float2half_rn(input[i]);
    }
    auto x = static_cast<half*>(r.Allocate(size_t(capacity) * p.hiddenSize * sizeof(half)));
    Cuda(cudaMemsetAsync(x, 0, size_t(capacity) * p.hiddenSize * sizeof(half), r.stream));
    if (tokens) Cuda(cudaMemcpyAsync(x, host.data(), host.size() * sizeof(half), cudaMemcpyHostToDevice, r.stream));
    if (tokens && stage != Qwen3LayerStage::Mlp) {
        auto w = r.Prepare(tokens, p.hiddenSize, Qwen3LayerStage::Attention);
        r.RunLayer(layer, tokens, positionBegin, Qwen3LayerStage::Attention, x, w);
    }
    QwenMoeLayerResult result;
    Impl::MoeWorkspace w{};
    if (stage != Qwen3LayerStage::Attention) {
        w = r.RunMoe(layer, tokens, capacity, x);
        result.allGatherCalls = 3;
        result.reduceScatterCalls = 1;
        result.indexedDecode = capacity == 1;
    }

    // 所有诊断读回在同一计算流完成后进行，不参与后端选择或路由决策。
    auto readHalf = [&](const half *device, size_t count) {
        std::vector<half> values(count);
        if (count) Cuda(cudaMemcpyAsync(values.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost, r.stream));
        r.Synchronize();
        std::vector<float> output;
        for (auto value : values) {
            float f = __half2float(value);
            if (!std::isfinite(f)) throw std::runtime_error("nonfinite MoE intermediate/output");
            output.push_back(f);
        }
        return output;
    };
    result.output = readHalf(x, input.size());
    if (stage != Qwen3LayerStage::Attention) {
        result.routerLogits = readHalf(w.logits, size_t(tokens) * p.expertCount);
        result.routingWeights = readHalf(w.prob, size_t(tokens) * p.expertsPerToken);
        result.sharedOutput = readHalf(w.shared, input.size());
        result.routedOutput.resize(input.size());
        result.selectedExperts.resize(size_t(tokens) * p.expertsPerToken);
        result.expertTokens.resize(p.expertCount);
        if (tokens) {
            Cuda(cudaMemcpyAsync(result.routedOutput.data(), w.routed, input.size() * sizeof(float), cudaMemcpyDeviceToHost, r.stream));
            Cuda(cudaMemcpyAsync(result.selectedExperts.data(), w.ids, result.selectedExperts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, r.stream));
        }
        Cuda(cudaMemcpyAsync(result.expertTokens.data(), w.counts, p.expertCount * sizeof(uint32_t), cudaMemcpyDeviceToHost, r.stream));
        r.Synchronize();
        for (size_t e = 0; e < result.expertTokens.size(); ++e) {
            if (e % p.worldSize != uint64_t(p.rank)) result.expertTokens[e] = 0;
        }
        for (float value : result.routedOutput) if (!std::isfinite(value)) throw std::runtime_error("nonfinite MoE routed output");
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
    if (p.model != ShardModel::DenseQwen3) throw std::runtime_error("dense generation requires Qwen3 weights");
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
    if (r.paged) {
        if (capacity > uint64_t(r.paged->PageLen()) * r.paged->MaxPages()) {
            throw std::runtime_error("Qwen3 request exceeds FT paged pool capacity");
        }
        r.paged->End();
    } else {
        const uint64_t layerBytes = uint64_t(capacity) * 2 * p.kvHeadCount * p.headDim * sizeof(half);
        if (!layerBytes || layerBytes > 256ULL * 1024 * 1024 || p.layerCount > (256ULL * 1024 * 1024) / layerBytes) {
            throw std::runtime_error("Qwen3 request KV exceeds 256 MiB");
        }
        r.kvCache.resize(p.layerCount, nullptr);
        for (auto &cache : r.kvCache) {
            Cuda(cudaMalloc(reinterpret_cast<void**>(&cache), layerBytes));
            r.request.deviceBytes += layerBytes;
        }
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
    if (r.paged) r.paged->PrepareDecode(r.request.tokens);
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

std::vector<float> Qwen3LayerExecutor::StepMoe(const QwenMoeStep &step, bool graph) {
    auto &r = *impl;
    const auto &p = r.weights.Plan();
    const bool begin = step.action == QwenMoeAction::Begin;
    const bool decode = step.action == QwenMoeAction::Decode;
    if ((!r.forwardTraceDirectory.empty() || !r.blockTraceDirectory.empty()) &&
        (decode || graph || (begin && step.positionBegin != 0))) {
        throw std::runtime_error("MoE forward trace supports only fresh Eager Begin at position zero");
    }
    if (p.model != ShardModel::Qwen2Moe || uint32_t(step.action) > 3 ||
        (graph && (!r.graphInfo.enabled || begin)) || p.vocabSize > (protocol::kMaxPayload - 4) / 4) {
        throw std::runtime_error("invalid MoE round mode/model/vocabulary");
    }
    r.Validate(0, 1, p.hiddenSize, step.positionBegin, Qwen3LayerStage::Attention);
    std::vector<uint32_t> ids = step.tokenIds;
    for (auto id : ids) if (id >= p.vocabSize) throw std::runtime_error("MoE token outside vocabulary");
    if ((!begin && !decode && !ids.empty()) || (decode && ids.size() != 1)) {
        throw std::runtime_error("invalid MoE round token count");
    }
    if (step.action == QwenMoeAction::End) EndRequest();
    if (step.action == QwenMoeAction::Idle && r.request.active) {
        throw std::runtime_error("active MoE request cannot silently idle");
    }

    if (begin) {
        if (r.request.active || ids.empty() || ids.size() > kQwen3MaxPromptTokens ||
            !step.capacity || step.capacity > kQwen3MaxContextTokens || ids.size() > step.capacity ||
            step.positionBegin > 1048576 || step.capacity - 1 > 1048576 - step.positionBegin) {
            throw std::runtime_error("invalid MoE request admission");
        }
        r.ReserveMoeKV();
        r.request.active = true;
        r.request.capacity = step.capacity;
        r.request.positionBegin = step.positionBegin;
        r.request.deviceBytes = uint64_t(step.capacity) * r.KVTokenBytes() * p.layerCount;
        r.request.tokens = r.RestorePrefix(ids, step.positionBegin);
        ids.erase(ids.begin(), ids.begin() + r.request.tokens);
    }
    if (decode && (!r.request.active || r.request.tokens >= r.request.capacity)) {
        throw std::runtime_error("MoE decode requires active request with capacity");
    }

    // 前缀命中后，各 rank 的后缀长度不同；统一集合通信尺寸，空闲 rank 也参与。
    const uint32_t local = ids.size();
    auto counts = static_cast<uint32_t*>(r.Allocate((p.worldSize + 1) * sizeof(uint32_t)));
    Cuda(cudaMemcpyAsync(counts, &local, sizeof(local), cudaMemcpyHostToDevice, r.stream));
    Nccl(ncclAllGather(counts, counts + 1, 1, ncclUint32, r.comm, r.stream));
    std::vector<uint32_t> peers(p.worldSize);
    Cuda(cudaMemcpyAsync(peers.data(), counts + 1, peers.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, r.stream));
    r.Synchronize();
    r.ReleaseBuffers();
    const auto capacity = *std::max_element(peers.begin(), peers.end());
    if (!capacity) return {};
    if (capacity > kQwen3MaxPromptTokens || (graph && capacity != 1)) {
        throw std::runtime_error("invalid joint MoE capacity");
    }
    r.ReserveMoeKV();
    const uint32_t position = local ? r.request.positionBegin + r.request.tokens : 0;
    std::vector<float> result(local ? p.vocabSize : 0);

    if (graph) {
        const uint32_t token = local ? ids[0] : 0;
        const uint32_t meta[] = {local ? r.request.tokens : 0, position, local};
        if (!r.graphExec) {
            r.graphIds = static_cast<uint32_t*>(r.Allocate(sizeof(token), true));
            r.graphMeta = static_cast<uint32_t*>(r.Allocate(sizeof(meta), true));
            r.graphX = static_cast<half*>(r.Allocate(p.hiddenSize * sizeof(half), true));
            r.graphLogits = static_cast<float*>(r.Allocate(p.vocabSize * sizeof(float), true));
            r.graphWorkspace = r.Prepare(1, p.hiddenSize, Qwen3LayerStage::Attention, true);
            r.graphMoe = r.PrepareMoe(1, true);
            constexpr size_t bytes = 4 * 1024 * 1024;
            Blas(cublasSetWorkspace(r.blas, r.Allocate(bytes, true), bytes));
        }
        Cuda(cudaMemcpyAsync(r.graphIds, &token, sizeof(token), cudaMemcpyHostToDevice, r.stream));
        Cuda(cudaMemcpyAsync(r.graphMeta, meta, sizeof(meta), cudaMemcpyHostToDevice, r.stream));
        auto enqueue = [&]() {
            r.EnqueueMoe(r.graphIds, r.graphX, r.graphWorkspace, r.graphMoe,
                         r.graphLogits, 1, 1, 0, r.graphMeta);
        };
        if (!r.graphExec) {
            enqueue();
            r.Synchronize();
            ++r.graphInfo.warmups;
            Cuda(cudaStreamBeginCapture(r.stream, cudaStreamCaptureModeThreadLocal));
            try { enqueue(); }
            catch (...) {
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
        if (local) Cuda(cudaMemcpyAsync(result.data(), r.graphLogits, result.size() * sizeof(float), cudaMemcpyDeviceToHost, r.stream));
        r.Synchronize();
        ++r.graphInfo.replays;
    } else {
        const size_t rows = std::max<uint32_t>(1, local);
        auto input = static_cast<uint32_t*>(r.Allocate(rows * sizeof(uint32_t)));
        auto x = static_cast<half*>(r.Allocate(rows * p.hiddenSize * sizeof(half)));
        auto logits = static_cast<float*>(r.Allocate(p.vocabSize * sizeof(float)));
        auto attention = r.Prepare(rows, p.hiddenSize, Qwen3LayerStage::Attention);
        auto experts = r.PrepareMoe(capacity);
        if (local && !r.forwardTraceDirectory.empty()) {
            r.forwardTrace.reset(new detail::MoeForwardTrace(r.forwardTraceDirectory, ids,
                                                           p.hiddenSize, p.layerCount, p.rank));
            r.forwardTrace->Attach(r.Allocate(r.forwardTrace->Bytes()));
        }
        if (local) Cuda(cudaMemcpyAsync(input, ids.data(), ids.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, r.stream));
        r.EnqueueMoe(input, x, attention, experts, logits, local, capacity, position, nullptr, r.forwardTrace.get());
        if (local) Cuda(cudaMemcpyAsync(result.data(), logits, result.size() * sizeof(float), cudaMemcpyDeviceToHost, r.stream));
        if (r.forwardTrace) r.forwardTrace->Readback(r.stream);
        if (r.blockTrace) r.blockTrace->Readback(r.stream);
        r.Synchronize();
        if (r.forwardTrace) r.forwardTrace->Save(++r.forwardTraceRound);
        if (r.blockTrace) r.blockTrace->Save(++r.blockTraceRound);
        r.ReleaseBuffers();
        if (decode) ++r.graphInfo.eagerDecodes;
    }
    for (float value : result) if (!std::isfinite(value)) throw std::runtime_error("nonfinite MoE logits");
    r.request.tokens += local;
    if (begin) r.SavePrefix(step.tokenIds, step.positionBegin);
    return result;
}

void Qwen3LayerExecutor::EndRequest() {
    auto &r = *impl;
    r.Synchronize();
    if (r.weights.Plan().model == ShardModel::Qwen2Moe) {
        r.request = {};
        return;
    }
    r.ReleaseGraph();
    if (r.paged) r.paged->End();
    for (auto &cache : r.kvCache) {
        Cuda(cudaFree(cache));
        cache = nullptr;
    }
    r.kvCache.clear();
    r.request = {};
}

Qwen3KVUsage Qwen3LayerExecutor::KVUsage() const {
    if (!impl->paged) throw std::runtime_error("KV pressure requires dense paged cache");
    return impl->paged->Usage();
}

Qwen3RequestInfo Qwen3LayerExecutor::RequestInfo() const {
    auto info = impl->request;
    info.graph = impl->graphInfo;
    info.graph.active = impl->graphExec != nullptr;
    info.graph.workspaceBytes = impl->graphAllocated + (impl->paged ? impl->paged->GraphBytes() : 0);
    info.prefix = impl->paged ? impl->paged->Info(impl->prefixInfo) : impl->prefixInfo;
    if (impl->paged && info.active) {
        info.deviceBytes = info.prefix.activePages * info.prefix.pageLen * impl->weights.Plan().kvHeadCount * impl->weights.Plan().headDim * sizeof(half);
    }
    info.reservedKVBytes = impl->reservedKVBytes;
    for (const auto &entry : impl->prefixes) {
        info.prefix.snapshots.emplace_back(entry.position, entry.ids);
    }
    return info;
}

void Qwen3LayerExecutor::ClearPrefixCache(int layer) {
    auto &r = *impl;
    if (r.weights.Plan().model == ShardModel::Qwen2Moe && r.request.active) {
        throw std::runtime_error("cannot clear MoE prefix cache during an active request");
    }
    r.Synchronize();
    if (layer >= 0 && !r.paged) throw std::runtime_error("selective prefix clear requires dense paged TP");
    if (r.paged) r.paged->ClearPrefix(layer);
    while (!r.prefixes.empty()) r.ErasePrefix(0);
    if (r.prefixInfo.enabled) ++r.prefixInfo.clears;
}

void Qwen3LayerExecutor::Clear() {
    EndRequest();
    ClearPrefixCache();
    impl->ReleaseGraph();
    for (auto cache : impl->kvCache) Cuda(cudaFree(cache));
    impl->kvCache.clear();
    impl->reservedKVBytes = 0;
    impl->paged.reset();
    impl->ReleaseBuffers();
    if (impl->qkvLt) {
        Blas(cublasLtDestroy(impl->qkvLt));
        impl->qkvLt = nullptr;
    }
    if (impl->blas) {
        Blas(cublasDestroy(impl->blas));
        impl->blas = nullptr;
    }
}

} // namespace distributed
} // namespace fastllm
