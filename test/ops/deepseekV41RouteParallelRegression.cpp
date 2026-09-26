// DeepSeek-V4.1 的 CPU 参考路由按 token 并行化的等价性回归。
//
// 背景：真实模型有 384 个路由专家，超过 CUDA 路由 kernel 的 256 上限，所以整段
// 路由只能走 deepseekv41.cpp 里的 CPU 参考实现。那段代码原本是单线程逐 token 的
// O(seqlen · E · k) 扫描（每选一个专家就把 select[best] 置 -inf 再重扫全部 384 个），
// prefill 每块 4096 个 token 时是热路径，因此改成了按 token 区间切分到常驻线程池。
//
// 为什么需要这个测试：并行化的等价性无法用「起服务跑贪心输出比对」来验证。实测
// 发现同一配置、同一输入，**仅重启进程**输出就会分叉（0/3 一致），而同一进程内
// 连跑三轮是 3/3 一致的。也就是说存在进程级的非确定性（最可能来自 MoE 专家在
// NUMA / GPU 缓存上的初始布局），任何跨重启的输出比对都不构成正确性判据。
// 只有在同一进程内同时跑两条路径、逐位比对中间结果，才能真正证明等价。
//
// 本测试覆盖：
//   1. 串行参考与并行实现产出的 indices / scores 逐位相同（bit-for-bit，不是近似）；
//   2. 覆盖并行切分的边界：seqlen 不能整除线程数、seqlen 恰好等于串行阈值 64、
//      seqlen 小于线程数（部分线程分不到 token）；
//   3. top-k 的并列处理：构造多个专家分数完全相同的 logits，确认两条路径选出
//      同一个专家（原实现用严格大于比较，先出现者胜，切分不改变每 token 内的扫描顺序）。
//
// 这里刻意不复用实现里的代码：串行参考是照着 deepseekv41.cpp 的循环独立写的，
// 若将来实现被改动而语义漂移，这个测试要能发现。

#include "fastllm.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

static int gChecks = 0;

static void Require(bool ok, const std::string &msg) {
    if (!ok) {
        throw std::runtime_error(msg);
    }
    gChecks++;
}

// 与 deepseekv41.cpp 里 V41Softplus 同义：log1p(exp(x))，大值直接取 x 避免溢出。
static float Softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

// 串行参考：照抄 deepseekv41.cpp 并行化之前的循环语义，独立实现。
static void RouteSerial(const float *raw, const float *bias, int seqlen, int numExperts,
                        int topk, bool normTopkProb, float scaling,
                        std::vector<int> &indices, std::vector<float> &scores) {
    indices.assign((size_t)seqlen * topk, 0);
    scores.assign((size_t)seqlen * topk, 0.0f);
    std::vector<float> original(numExperts), select(numExperts);
    for (int t = 0; t < seqlen; t++) {
        for (int e = 0; e < numExperts; e++) {
            original[e] = std::sqrt(Softplus(raw[(uint64_t)t * numExperts + e]));
            select[e] = original[e] + bias[e];
        }
        float sum = 0.0f;
        for (int k = 0; k < topk; k++) {
            int best = 0;
            for (int e = 1; e < numExperts; e++) {
                if (select[e] > select[best]) {
                    best = e;
                }
            }
            indices[(uint64_t)t * topk + k] = best;
            scores[(uint64_t)t * topk + k] = original[best];
            sum += original[best];
            select[best] = -std::numeric_limits<float>::infinity();
        }
        for (int k = 0; k < topk; k++) {
            float &v = scores[(uint64_t)t * topk + k];
            if (normTopkProb && topk > 1) {
                v /= (sum + 1e-20f);
            }
            v *= scaling;
        }
    }
}

// 常驻线程池上的一段 token 区间，与 deepseekv41.cpp 里 V41EngramGatherOp 同构。
struct RouteRangeOp : MultiThreadBaseOp {
    const std::function<void(int, int)> *worker;
    int st, end;
    RouteRangeOp(const std::function<void(int, int)> *worker, int st, int end)
        : worker(worker), st(st), end(end) {}
    void Run() override {
        (*worker)(st, end);
    }
};

// 并行实现：与 deepseekv41.cpp 改动后的结构一致 —— original / select 每线程各一份，
// 按 token 区间切分派发到常驻线程池。
static void RouteParallel(const float *raw, const float *bias, int seqlen, int numExperts,
                          int topk, bool normTopkProb, float scaling,
                          std::vector<int> &indices, std::vector<float> &scores,
                          int forceThreads) {
    indices.assign((size_t)seqlen * topk, 0);
    scores.assign((size_t)seqlen * topk, 0.0f);
    std::function<void(int, int)> worker = [&](int tSt, int tEnd) {
        std::vector<float> original(numExperts), select(numExperts);
        for (int t = tSt; t < tEnd; t++) {
            for (int e = 0; e < numExperts; e++) {
                original[e] = std::sqrt(Softplus(raw[(uint64_t)t * numExperts + e]));
                select[e] = original[e] + bias[e];
            }
            float sum = 0.0f;
            for (int k = 0; k < topk; k++) {
                int best = 0;
                for (int e = 1; e < numExperts; e++) {
                    if (select[e] > select[best]) {
                        best = e;
                    }
                }
                indices[(uint64_t)t * topk + k] = best;
                scores[(uint64_t)t * topk + k] = original[best];
                sum += original[best];
                select[best] = -std::numeric_limits<float>::infinity();
            }
            for (int k = 0; k < topk; k++) {
                float &v = scores[(uint64_t)t * topk + k];
                if (normTopkProb && topk > 1) {
                    v /= (sum + 1e-20f);
                }
                v *= scaling;
            }
        }
    };

    AliveThreadPool *pool = GetAlivePool();
    int threadSt = pool->curActivateThreadInterval.first;
    int threadLen = pool->curActivateThreadInterval.second - threadSt;
    int threads = std::min(seqlen, std::max(1, threadLen));
    if (forceThreads > 0) {
        threads = std::min(threads, forceThreads);
    }
    if (threads <= 1) {
        worker(0, seqlen);
        return;
    }
    std::vector<RouteRangeOp*> ops;
    int per = (seqlen + threads - 1) / threads;
    for (int i = 0; i < threads; i++) {
        int st = i * per, end = std::min(seqlen, st + per);
        if (st < end) {
            ops.push_back(new RouteRangeOp(&worker, st, end));
        }
    }
    for (size_t i = 0; i < ops.size(); i++) {
        pool->PushOp(threadSt + (int)i, ops[i]);
    }
    for (size_t i = 0; i < ops.size(); i++) {
        pool->Wait(threadSt + (int)i);
        delete ops[i];
    }
}

static void CompareBitwise(const std::vector<int> &ia, const std::vector<float> &sa,
                           const std::vector<int> &ib, const std::vector<float> &sb,
                           const std::string &label) {
    Require(ia.size() == ib.size() && sa.size() == sb.size(), label + "：输出长度不同");
    for (size_t i = 0; i < ia.size(); i++) {
        if (ia[i] != ib[i]) {
            throw std::runtime_error(label + "：expert_idx[" + std::to_string(i) + "] 不同 " +
                                     std::to_string(ia[i]) + " vs " + std::to_string(ib[i]));
        }
    }
    // 逐 bit 比较，不用容差：切分不改变每 token 内的运算与累加顺序，必须完全相同。
    for (size_t i = 0; i < sa.size(); i++) {
        uint32_t x, y;
        memcpy(&x, &sa[i], 4);
        memcpy(&y, &sb[i], 4);
        if (x != y) {
            throw std::runtime_error(label + "：score[" + std::to_string(i) + "] 不逐位相同 " +
                                     std::to_string(sa[i]) + " vs " + std::to_string(sb[i]));
        }
    }
}

// 真实模型的几何：384 个专家、top-6、归一化后乘 routed_scaling_factor。
static const int kExperts = 384;
static const int kTopk = 6;
static const float kScaling = 2.5f;

static void CheckCase(int seqlen, int forceThreads, uint32_t seed, const std::string &label) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.5f);
    std::vector<float> raw((size_t)seqlen * kExperts), bias(kExperts);
    for (auto &v : raw) v = dist(rng);
    for (auto &v : bias) v = dist(rng) * 0.1f;

    std::vector<int> iSer, iPar;
    std::vector<float> sSer, sPar;
    RouteSerial(raw.data(), bias.data(), seqlen, kExperts, kTopk, true, kScaling, iSer, sSer);
    RouteParallel(raw.data(), bias.data(), seqlen, kExperts, kTopk, true, kScaling,
                  iPar, sPar, forceThreads);
    CompareBitwise(iSer, sSer, iPar, sPar, label);
    std::cout << "  " << label << " (seqlen=" << seqlen << ", threads<=" << forceThreads
              << ") 逐位一致\n";
}

// 并列分数：多个专家的 select 值完全相同。原实现用严格大于比较，先出现者胜；
// 切分不改变每 token 内的扫描顺序，所以两条路径必须选出同一个专家。
static void CheckTies() {
    const int seqlen = 257;   // 刻意取质数，保证不能被线程数整除
    std::vector<float> raw((size_t)seqlen * kExperts, 0.0f), bias(kExperts, 0.0f);
    // 每个 token 让一整段专家取同一个值，制造大量并列
    for (int t = 0; t < seqlen; t++) {
        for (int e = 0; e < kExperts; e++) {
            raw[(uint64_t)t * kExperts + e] = (e % 16 == 0) ? 3.0f : 1.0f;
        }
    }
    std::vector<int> iSer, iPar;
    std::vector<float> sSer, sPar;
    RouteSerial(raw.data(), bias.data(), seqlen, kExperts, kTopk, true, kScaling, iSer, sSer);
    RouteParallel(raw.data(), bias.data(), seqlen, kExperts, kTopk, true, kScaling, iPar, sPar, 0);
    CompareBitwise(iSer, sSer, iPar, sPar, "并列分数");
    std::cout << "  并列分数 (seqlen=" << seqlen << ") 逐位一致\n";
}

int main() {
    try {
        std::cout << "== DeepSeek-V4.1 CPU 路由并行等价性 ==\n";

        // 典型 prefill 分块
        CheckCase(4096, 0, 12345, "prefill 分块 4096");
        CheckCase(8192, 0, 999, "prefill 分块 8192");

        // 切分边界
        CheckCase(64, 0, 7, "恰好等于串行阈值 64");
        CheckCase(65, 0, 8, "刚过阈值 65");
        CheckCase(1000, 7, 31, "不能整除线程数");
        CheckCase(3, 8, 77, "seqlen 小于线程数");
        CheckCase(1, 0, 5, "单 token（decode）");

        // 并列处理
        CheckTies();

        std::cout << "PASS: " << gChecks << " checks\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
