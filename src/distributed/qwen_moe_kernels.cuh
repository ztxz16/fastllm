#ifndef FASTLLM_QWEN_MOE_KERNELS_CUH
#define FASTLLM_QWEN_MOE_KERNELS_CUH

// 仅由 qwen3_layer.cu 包含；使用同一 worker stream，FP16 输入与 FP32 累加。
namespace moe {
/**
 * 计算 FP32 softmax 与 Top-K，最终路由权重转 FP16；相等概率按专家 ID 升序。
 * @param logits FP16 [capacity,experts]，只读取有效 Token。
 * @param ids int32 [capacity,topk] 输出；空位 ID=-1。
 * @param prob FP16 [capacity,topk] 输出；空位概率为零。
 * @param valid 有效 Token 数，可以为零。
 * @param capacity 预留 Token 数，至少为 valid 且不小于一。
 * @param experts 专家数，1..256。
 * @param topk 每 Token 选择数，1..experts。
 * @param normalize 是否归一化被选中的概率；不能忽略模型 norm_topk_prob。
 */
__global__ void Route(const half *logits, int *ids, half *prob, int valid, int capacity,
                      int experts, int topk, bool normalize) {
    int row = blockIdx.x;
    if (threadIdx.x || row >= capacity) return;
    if (row >= valid) {
        for (int k = 0; k < topk; ++k) {
            ids[row * topk + k] = -1;
            prob[row * topk + k] = __float2half_rn(0);
        }
        return;
    }

    float scores[256], selected[256];
    float maximum = -INFINITY, total = 0, sum = 0;
    for (int e = 0; e < experts; ++e) {
        float value = __half2float(logits[row * experts + e]);
        if (!isfinite(value)) {
            // 有限输入仍可能在 FP16 GEMM 溢出；禁止 NaN 生成重复 ID 后越界写专家索引。
            // 保留 NaN 路由证据供本次同步读回拒绝，同时所有 rank 完成相同集合通信。
            for (int k = 0; k < topk; ++k) {
                ids[row * topk + k] = -1;
                prob[row * topk + k] = __float2half_rn(NAN);
            }
            return;
        }
        maximum = fmaxf(maximum, value);
    }
    for (int e = 0; e < experts; ++e) {
        scores[e] = expf(__half2float(logits[row * experts + e]) - maximum);
        total += scores[e];
    }
    for (int e = 0; e < experts; ++e) scores[e] /= total;

    for (int k = 0; k < topk; ++k) {
        int best = 0;
        for (int e = 1; e < experts; ++e) if (scores[e] > scores[best]) best = e;
        ids[row * topk + k] = best;
        selected[k] = scores[best];
        sum += selected[k];
        scores[best] = -1;
    }
    for (int k = 0; k < topk; ++k) {
        prob[row * topk + k] = __float2half_rn(selected[k] / (normalize ? sum : 1));
    }
}

/**
 * 为每个专家建立稳定的有效 Token 索引，不把 padding 计入专家工作量。
 * @param ids int32 [rows,topk]，-1 表示无效。
 * @param map int32 [experts,rows]；每个专家的前 counts[e] 项是路由槽位索引。
 * @param counts int32 [experts]，全部由本 kernel 写入。
 * @param rows 全组预留 Token 数。
 * @param topk 每 Token 的路由槽位数。
 * @param experts 专家总数。
 */
__global__ void Index(const int *ids, int *map, int *counts, int rows, int topk, int experts) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= experts) return;
    int count = 0;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < topk; ++k) {
            if (ids[row * topk + k] == expert) {
                map[expert * rows + count++] = row * topk + k;
                break;
            }
        }
    }
    counts[expert] = count;
}

/** 从全组 FP16 [rows,hidden] 激活按路由槽位聚合为 FP16 [count,hidden]，供 Prefill GEMM 使用。 */
__global__ void Pack(const half *input, const int *map, half *output, int count, int hidden, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * hidden) output[i] = input[(map[i / hidden] / topk) * hidden + i % hidden];
}

/** 将单个专家 FP16 [count,hidden] 输出乘 FP16 路由概率后舍入，累加到全组 FP32 贡献。 */
__global__ void Scatter(const half *values, const int *map, const half *prob, float *sum,
                        int count, int hidden, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * hidden) {
        int slot = map[i / hidden];
        sum[(slot / topk) * hidden + i % hidden] += __half2float(__float2half_rn(
            __half2float(values[i]) * __half2float(prob[slot])));
    }
}

/** 查询当前 Token 对指定专家的路由槽位；未命中（包括空闲 padding）返回 -1。 */
__device__ int Slot(const int *ids, int row, int topk, int expert) {
    for (int k = 0; k < topk; ++k) if (ids[row * topk + k] == expert) return row * topk + k;
    return -1;
}

/**
 * 单 Token 轮次使用设备路由索引计算专家 Gate/Up，避免读取 CPU 路由或改变 Graph 拓扑。
 * @param x FP16 [rows,hidden] 激活。
 * @param weight FP16 [2*width,hidden] 融合 Gate/Up 权重。
 * @param ids int32 [rows,topk]；只为命中 expert 的行计算。
 * @param activated FP16 [rows,width]；未命中行不会被后续 Down 读取。
 * @param rows 全组预留 Token 数。
 * @param hidden 输入隐藏维度。
 * @param width 专家中间维度。
 * @param topk 每 Token 的路由槽位数。
 * @param expert 本次计算的全局专家 ID。
 */
__global__ void DecodeGate(const half *x, const half *weight, const int *ids, half *activated,
                           int rows, int hidden, int width, int topk, int expert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    int row = i / width, col = i % width;
    if (Slot(ids, row, topk, expert) < 0) return;

    float gate = 0, up = 0;
    for (int k = 0; k < hidden; ++k) {
        float value = __half2float(x[row * hidden + k]);
        gate += value * __half2float(weight[col * hidden + k]);
        up += value * __half2float(weight[(width + col) * hidden + k]);
    }
    gate = __half2float(__float2half_rn(gate));
    up = __half2float(__float2half_rn(up));
    float silu = __half2float(__float2half_rn(gate / (1 + expf(-gate))));
    activated[i] = __float2half_rn(silu * up);
}

/** 按同一设备路由索引执行 FP16 专家 Down，舍入加权结果后累加 FP32 全组贡献，无原子竞争。 */
__global__ void DecodeDown(const half *activated, const half *weight, const int *ids, const half *prob,
                           float *sum, int rows, int hidden, int width, int topk, int expert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    int row = i / hidden, col = i % hidden;
    int slot = Slot(ids, row, topk, expert);
    if (slot < 0) return;
    float value = 0;
    for (int k = 0; k < width; ++k) value += __half2float(activated[row * width + k]) * __half2float(weight[col * width + k]);
    value = __half2float(__float2half_rn(value));
    sum[i] += __half2float(__float2half_rn(value * __half2float(prob[slot])));
}

/** 共享专家按 sigmoid(gate) 门控；与 ReduceScatter 返回的路由贡献相加，再按 FP16 语义加残差。 */
__global__ void Finish(half *x, const half *gate, half *shared, const float *routed, int count, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sigmoid = __half2float(__float2half_rn(1 / (1 + expf(-__half2float(gate[i / hidden])))));
    shared[i] = __float2half_rn(sigmoid * __half2float(shared[i]));
    float delta = __half2float(__float2half_rn(__half2float(__float2half_rn(routed[i])) + __half2float(shared[i])));
    x[i] = __float2half_rn(__half2float(x[i]) + delta);
}
} // namespace moe
#endif
