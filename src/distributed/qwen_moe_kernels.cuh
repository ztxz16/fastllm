#ifndef FASTLLM_QWEN_MOE_KERNELS_CUH
#define FASTLLM_QWEN_MOE_KERNELS_CUH

// 仅由 qwen3_layer.cu 包含；使用同一 worker stream，专家贡献按全局专家顺序逐次舍入为 FP16。
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
 * @param meta 可空的设备 [past,position,active]；Graph 用 active 掩蔽空闲 rank。
 */
__global__ void Route(const half *logits, int *ids, half *prob, int valid, int capacity,
                      int experts, int topk, bool normalize, const uint32_t *meta = nullptr) {
    if (meta) valid = meta[2];
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

/**
 * 将加权专家输出写入独占路由槽位，不在专家所在 rank 提前求和。
 *
 * 每个槽位仅有一个专家及一个 owner rank；其他 rank 的槽位保持零，供
 * ReduceScatter 返回请求所属 rank 后按全局专家顺序累加。
 * @param values FP16 [count,hidden] 专家输出。
 * @param map int32 [count]，元素为全组 Token 的扁平路由槽位索引。
 * @param prob FP16 [rows,topk] 路由概率。
 * @param contributions FP16 [rows,topk,hidden]，调用前整组置零。
 * @param count 当前专家的有效 Token 数。
 * @param hidden 隐藏维度。
 */
__global__ void Scatter(const half *values, const int *map, const half *prob, half *contributions,
                        int count, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * hidden) {
        int slot = map[i / hidden];
        contributions[size_t(slot) * hidden + i % hidden] = __float2half_rn(
            __half2float(values[i]) * __half2float(prob[slot]));
    }
}

/**
 * 按设备端有效行数聚合专家输入，尾部填零，复用 Graph 固定地址及路由索引。
 * @param input 连续 FP16 [rows,hidden] 全组激活。
 * @param map int32 [rows] 专家路由槽位，只有前 count 行有效。
 * @param count 设备端单个 int32 有效行数，由正式 Index 生成。
 * @param output FP16 [rows,hidden] 固定容量输出。
 * @param rows 全组固定容量，不随本轮专家命中数变化。
 * @param hidden 输入宽度。
 * @param topk 每 Token 的路由槽位数。
 */
__global__ void PackMasked(const half *input, const int *map, const int *count, half *output,
                           int rows, int hidden, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * hidden) {
        int row = i / hidden;
        output[i] = row < *count ? input[(map[row] / topk) * hidden + i % hidden] : __float2half_rn(0);
    }
}

/**
 * 仅将设备端有效行写回独占路由槽位；保持正式 Scatter 的 FP16 加权舍入边界。
 * @param values FP16 [rows,hidden] 专家结果，尾部不得参与贡献。
 * @param map int32 [rows] 路由槽位，仅前 count 行有效。
 * @param count 设备端 int32 有效行数。
 * @param prob FP16 [rows,topk] 原始路由概率。
 * @param contributions FP16 [rows,topk,hidden]，调用前整组置零。
 * @param rows 全组固定容量。
 * @param hidden 输出宽度。
 */
__global__ void ScatterMasked(const half *values, const int *map, const int *count,
                              const half *prob, half *contributions, int rows, int hidden) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * hidden && i / hidden < *count) {
        int slot = map[i / hidden];
        contributions[size_t(slot) * hidden + i % hidden] = __float2half_rn(
            __half2float(values[i]) * __half2float(prob[slot]));
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

/**
 * 通过设备路由执行单 Token 专家 Down，将加权结果写入独占槽位。
 *
 * Down 和乘路由概率分别舍入为 FP16，与 Prefill Scatter 的输出布局相同。
 * 未选中及空闲行不写入，不读取 CPU 路由，不改变 CUDA Graph 拓扑。
 * @param activated FP16 [rows,width] 专家激活。
 * @param weight FP16 [hidden,width] Down 权重。
 * @param ids int32 [rows,topk] 全组专家 ID，padding 为 -1。
 * @param prob FP16 [rows,topk] 路由概率。
 * @param contributions FP16 [rows,topk,hidden]，调用前整组置零。
 * @param rows 全组预留 Token 数。
 * @param hidden 隐藏维度。
 * @param width 专家中间维度。
 * @param topk 每 Token 的路由槽位数。
 * @param expert 当前全局专家 ID。
 */
__global__ void DecodeDown(const half *activated, const half *weight, const int *ids, const half *prob,
                           half *contributions, int rows, int hidden, int width, int topk, int expert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * hidden) return;
    int row = i / hidden, col = i % hidden;
    int slot = Slot(ids, row, topk, expert);
    if (slot < 0) return;
    float value = 0;
    for (int k = 0; k < width; ++k) value += __half2float(activated[row * width + k]) * __half2float(weight[col * width + k]);
    value = __half2float(__float2half_rn(value));
    contributions[size_t(slot) * hidden + col] = __float2half_rn(value * __half2float(prob[slot]));
}

/**
 * 按全局专家 ID 升序逐项做 FP16 加法，复现 HF 的专家 index_add_ 舍入边界。
 *
 * ReduceScatter 只搬运每个槽位的单一非零贡献，不合并不同专家；本 kernel
 * 在请求所属 rank 统一合并，结果不依赖 EP 分片或 Top-K 槽位排序。设备排序
 * 在每次调用重做，支持 Graph 重放时路由改变；空闲行输出零。
 * @param contributions FP16 [capacity,topk,hidden]，已完成 ReduceScatter。
 * @param ids int32 [capacity,topk] 本 rank 专家 ID，padding 为 -1。
 * @param routed FP32 [capacity,hidden]，保存 FP16 累加值的无损提升，兼容诊断协议。
 * @param hidden 隐藏维度。
 * @param topk 路由槽位数，1..256；由模型配置校验保证范围及 ID 唯一性。
 */
__global__ void OrderedSum(const half *contributions, const int *ids, float *routed,
                           int hidden, int topk) {
    __shared__ int order[256];
    const int row = blockIdx.x;
    const int *experts = ids + row * topk;
    if (threadIdx.x == 0) {
        for (int slot = 0; slot < topk; ++slot) {
            int position = slot;
            while (position > 0 && experts[order[position - 1]] > experts[slot]) {
                order[position] = order[position - 1];
                --position;
            }
            order[position] = slot;
        }
    }
    __syncthreads();

    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        half sum = __float2half_rn(0.0f);
        for (int i = 0; i < topk; ++i) {
            const int slot = order[i];
            if (experts[slot] >= 0) {
                sum = __hadd(sum, contributions[(size_t(row) * topk + slot) * hidden + col]);
            }
        }
        routed[size_t(row) * hidden + col] = __half2float(sum);
    }
}

/** 共享专家按 sigmoid(gate) 门控；与有序累加的路由贡献相加，再按 FP16 语义加残差。 */
__global__ void Finish(half *x, const half *gate, half *shared, const float *routed,
                       int count, int hidden, const half *prob, int topk) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    for (int k = 0; k < topk; ++k) {
        if (!isfinite(__half2float(prob[(i / hidden) * topk + k]))) {
            x[i] = __float2half_rn(NAN);
            return;
        }
    }
    float sigmoid = __half2float(__float2half_rn(1 / (1 + expf(-__half2float(gate[i / hidden])))));
    shared[i] = __float2half_rn(sigmoid * __half2float(shared[i]));
    float delta = __half2float(__float2half_rn(__half2float(__float2half_rn(routed[i])) + __half2float(shared[i])));
    x[i] = __float2half_rn(__half2float(x[i]) + delta);
}
} // namespace moe
#endif
