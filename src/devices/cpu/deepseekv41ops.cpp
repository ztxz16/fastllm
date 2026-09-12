//
// DeepSeek-V4.1 专用算子的 CPU 参考实现。
//
// 这些算子同时作为 CUDA 实现的数值参考。所有算子均以 FP32 计算，
// 输入/输出支持 FLOAT32 / FLOAT16 / BFLOAT16。
//
//   DeepSeekV41HcMix           : Hyper-Connections 系数（pre / post / comb），不做混合
//   DeepSeekV41HcApplyPre      : 用给定 pre 系数把 hc 份隐藏状态折叠成一份
//   DeepSeekV41EngramApply     : Engram 门控写回 residual stream（原地）
//   DeepSeekV41RotaryQuant     : RoPE（可逆向）+ 可选 FP8 / FP4 伪量化（原地）
//   DeepSeekV41Compress        : compressor 池化（ratio 1 / 2）+ RMSNorm，输出 pre-RoPE latent
//   DeepSeekV41IndexerScore    : indexer 打分 score[t, j] = sum_h relu(q_h . k_j) * w_h
//   DeepSeekV41CandidateBlocks : 两级 top-k 的第一级（按块选候选）
//   DeepSeekV41IndexerTopK     : 因果 / 候选掩码后的 top-k（输出升序，-1 表示无效）
//   DeepSeekV41SparseAttention : 滑窗 + 压缩 top-k 的稀疏注意力（MQA，含 attention sink）
//   DeepSeekV41WindowStore     : 把当前 chunk 的 KV 写入环形滑窗缓存
//

#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"
#include "utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace fastllm {
    namespace {
        int V41Int(const IntDict &params, const char *name, int fallback) {
            auto it = params.find(name);
            return it == params.end() ? fallback : it->second;
        }

        float V41Float(const FloatDict &params, const char *name, float fallback) {
            auto it = params.find(name);
            return it == params.end() ? fallback : it->second;
        }

        Data *V41Optional(const DataDict &datas, const char *name) {
            auto it = datas.find(name);
            if (it == datas.end() || it->second == nullptr) {
                return nullptr;
            }
            return it->second;
        }

        inline float V41ToFloat(const uint8_t *base, DataType type, uint64_t index) {
            if (type == DataType::FLOAT32) {
                return ((const float*)base)[index];
            } else if (type == DataType::BFLOAT16) {
                return BFloat16BitsToFloat32(((const uint16_t*)base)[index]);
            } else if (type == DataType::FLOAT16) {
                return half_to_float(((const uint16_t*)base)[index]);
            }
            return 0.0f;
        }

        inline void V41FromFloat(uint8_t *base, DataType type, uint64_t index, float value) {
            if (type == DataType::FLOAT32) {
                ((float*)base)[index] = value;
            } else if (type == DataType::BFLOAT16) {
                ((uint16_t*)base)[index] = Float32ToBFloat16RNEBits(value);
            } else if (type == DataType::FLOAT16) {
                ((uint16_t*)base)[index] = float_to_half(value);
            }
        }

        bool V41IsFloatType(DataType type) {
            return type == DataType::FLOAT32 || type == DataType::BFLOAT16 || type == DataType::FLOAT16;
        }

        void V41ReadFloat(const Data &data, std::vector<float> &out) {
            uint64_t n = data.Count(0);
            out.resize(n);
            if (data.dataType == DataType::FLOAT32) {
                memcpy(out.data(), data.cpuData, n * sizeof(float));
                return;
            }
            for (uint64_t i = 0; i < n; i++) {
                out[i] = V41ToFloat(data.cpuData, data.dataType, i);
            }
        }

        void V41WriteFloat(const std::vector<float> &values, Data &data) {
            uint64_t n = data.Count(0);
            if (data.dataType == DataType::FLOAT32) {
                memcpy(data.cpuData, values.data(), n * sizeof(float));
                return;
            }
            for (uint64_t i = 0; i < n; i++) {
                V41FromFloat(data.cpuData, data.dataType, i, values[i]);
            }
        }

        // 简单的多线程并行 for：把 [0, count) 切成若干段
        struct V41RangeTask : MultiThreadBaseOp {
            std::function<void(int, int)> fn;
            int st, end;
            V41RangeTask(std::function<void(int, int)> fn, int st, int end) : fn(std::move(fn)), st(st), end(end) {}
            void Run() override { fn(st, end); }
        };

        void V41ParallelFor(int count, const std::function<void(int, int)> &fn, int minPerThread = 1) {
            if (count <= 0) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            int firstThread = pool->curActivateThreadInterval.first;
            int available = std::max(1, pool->curActivateThreadInterval.second - firstThread);
            int threads = std::min(available, std::max(1, count / std::max(1, minPerThread)));
            if (threads <= 1) {
                fn(0, count);
                return;
            }
            std::vector<V41RangeTask*> tasks;
            int per = (count + threads - 1) / threads;
            for (int i = 0; i < threads; i++) {
                int st = i * per, end = std::min(count, st + per);
                if (st >= end) {
                    break;
                }
                tasks.push_back(new V41RangeTask(fn, st, end));
            }
            for (int i = 0; i < (int)tasks.size(); i++) {
                pool->PushOp(firstThread + i, tasks[i]);
            }
            for (int i = 0; i < (int)tasks.size(); i++) {
                pool->Wait(firstThread + i);
                delete tasks[i];
            }
        }

        inline float V41Sigmoid(float x) {
            if (x >= 0.0f) {
                return 1.0f / (1.0f + std::exp(-x));
            }
            float z = std::exp(x);
            return z / (1.0f + z);
        }

        // 与 model.py::precompute_freqs_cis 一致的 YaRN 频率
        std::vector<float> V41InvFreq(int ropeDim, float base, int originalSeqLen,
                                      float factor, int betaFast, int betaSlow) {
            std::vector<float> invFreq;
            for (int i = 0; i < ropeDim; i += 2) {
                invFreq.push_back(1.0f / std::pow(base, (float)i / ropeDim));
            }
            if (originalSeqLen > 0) {
                auto correctedDim = [&](float rotations) {
                    return ropeDim * std::log((float)originalSeqLen / (rotations * 2.0f * (float)M_PI)) /
                           (2.0f * std::log(base));
                };
                int low = std::max((int)std::floor(correctedDim((float)betaFast)), 0);
                int high = std::min((int)std::ceil(correctedDim((float)betaSlow)), ropeDim - 1);
                float denom = std::max((float)(high - low), 1e-3f);
                for (int i = 0; i < (int)invFreq.size(); i++) {
                    float ramp = std::max(0.0f, std::min(1.0f, ((float)i - low) / denom));
                    float smooth = 1.0f - ramp;
                    invFreq[i] = invFreq[i] / factor * (1.0f - smooth) + invFreq[i] * smooth;
                }
            }
            return invFreq;
        }
    }

    // ---------------- FP8 / FP4 伪量化 ----------------

    static float V41Pow2Ceil(float x) {
        // fast_round_scale: 2^ceil(log2(x))
        if (!(x > 0.0f)) {
            return 1.0f;
        }
        uint32_t bits;
        memcpy(&bits, &x, sizeof(bits));
        int exponent = (int)((bits >> 23) & 0xFF) - 127 + ((bits & ((1u << 23) - 1)) != 0 ? 1 : 0);
        return std::ldexp(1.0f, exponent);
    }

    static float V41FP8RoundTrip(float v) {
        static const FP8E4M3ToFP32Manager fp8;
        return fp8.dict[fp8.quantization(v)];
    }

    static float V41FP4RoundTrip(float v) {
        // E2M1 (无 inf / nan)：0, 0.5, 1, 1.5, 2, 3, 4, 6，RNE
        static const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        float a = std::fabs(v);
        if (a >= 6.0f) {
            return std::copysign(6.0f, v);
        }
        int lower = 0;
        while (lower + 1 < 8 && grid[lower + 1] <= a) {
            lower++;
        }
        float lo = grid[lower], hi = grid[std::min(lower + 1, 7)];
        float r;
        if (a == lo) {
            r = lo;
        } else {
            float mid = 0.5f * (lo + hi);
            if (a < mid) {
                r = lo;
            } else if (a > mid) {
                r = hi;
            } else {
                r = ((lower & 1) == 0) ? lo : hi; // 平局取偶数编码
            }
        }
        return std::copysign(r, v);
    }

    // quantMode: 1 = FP8 E4M3 + UE8M0 scale（act_quant, 每 blockSize 一组）
    //            2 = FP4 E2M1 + UE8M0 scale（fp4_act_quant scale_dtype=e8m0）
    //            3 = FP4 E2M1 + E4M3 scale（fp4_act_quant scale_dtype=e4m3, 压缩 KV）
    float DeepSeekV41QuantMax(int quantMode) {
        return quantMode == 1 ? 448.0f : 6.0f;
    }

    // 由块内 amax 推出该块的 scale。FP4 存储与伪量化共用这一份推导，
    // 保证「先伪量化再按 FP4 存储」是幂等的（存储无损）。
    float DeepSeekV41BlockScale(float amax, int quantMode) {
        if (quantMode == 1) {
            amax = std::max(amax, 1e-4f);
            return V41Pow2Ceil(amax * (1.0f / 448.0f));
        } else if (quantMode == 2) {
            amax = std::max(amax, 6.0f * std::ldexp(1.0f, -126));
            return V41Pow2Ceil(amax * (1.0f / 6.0f));
        }
        amax = std::max(amax, 6.0f * std::ldexp(1.0f, -9));
        float scale = V41FP8RoundTrip(amax / 6.0f);
        return scale > 0.0f ? scale : std::ldexp(1.0f, -9);
    }

    void DeepSeekV41FakeQuantRow(float *row, int len, int quantMode, int blockSize) {
        if (quantMode <= 0) {
            return;
        }
        const float qmax = DeepSeekV41QuantMax(quantMode);
        for (int start = 0; start < len; start += blockSize) {
            int end = std::min(start + blockSize, len);
            float amax = 0.0f;
            for (int i = start; i < end; i++) {
                amax = std::max(amax, std::fabs(row[i]));
            }
            float scale = DeepSeekV41BlockScale(amax, quantMode);
            for (int i = start; i < end; i++) {
                float q = std::max(-qmax, std::min(qmax, row[i] / scale));
                if (quantMode == 1) {
                    q = V41FP8RoundTrip(q);
                } else {
                    q = V41FP4RoundTrip(q);
                }
                row[i] = q * scale;
            }
        }
    }

    // ---------------- 量化 KV 存储 ----------------
    //
    // 缓存行统一放在 INT8 的 Data 里，[b, rows, rowBytes]；三种行布局按 (quantMode, blockSize)
    // 区分，与写入 cache 前的伪量化 (DeepSeekV41FakeQuantRow) 严格同构：
    //
    //   quantMode 1, block 32 : [dim 个 FP8 E4M3][dim/32 个 UE8M0]        滑窗 KV，dim + dim/32 字节
    //   quantMode 3, block 16 : [dim 个 FP4 E2M1（打包）][dim/16 个 E4M3]  压缩 KV，dim/2 + dim/16 字节
    //   quantMode 2, block 32 : [dim 个 FP4 E2M1（打包）][dim/32 个 UE8M0] indexer key，dim/2 + dim/32 字节
    //
    // FP4 打包：一个字节放两个 code，低 4 位是偶数下标。UE8M0 的值 = 2^(byte - 127)。

    inline float V41DecodeFp8E4M3(uint8_t c) {
        int e = (c >> 3) & 0xF, m = c & 7;
        float v;
        if (e == 0) {
            v = std::ldexp((float)m, -9);
        } else if (e == 15 && m == 7) {
            v = std::numeric_limits<float>::quiet_NaN();
        } else {
            v = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
        }
        return (c & 0x80) ? -v : v;
    }

    // r 必须已经在 E4M3 网格上（V41FP8RoundTrip 的输出）
    inline uint8_t V41EncodeFp8E4M3(float r) {
        if (std::isnan(r)) {
            return 0x7F;
        }
        uint8_t sign = std::signbit(r) ? 0x80 : 0;
        float a = std::fabs(r);
        if (a == 0.0f) {
            return sign;
        }
        if (a >= 448.0f) {
            return sign | 0x7E;
        }
        if (a < std::ldexp(1.0f, -6)) {
            int m = (int)std::lround(std::ldexp(a, 9));
            return sign | (uint8_t)std::min(m, 7);
        }
        int e;
        float f = std::frexp(a, &e);          // a = f * 2^e, f in [0.5, 1)
        int E = e - 1;
        int m = (int)std::lround((2.0f * f - 1.0f) * 8.0f);
        if (m == 8) {
            m = 0;
            E++;
        }
        if (E > 8 || (E == 8 && m > 6)) {
            return sign | 0x7E;
        }
        return sign | (uint8_t)(((E + 7) << 3) | m);
    }

    // E2M1 编码：bit3 符号，bit2..0 是网格下标（值表见下）。r 必须已经在 E2M1 网格上。
    static const float kV41Fp4Grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    inline uint8_t V41EncodeFp4E2M1(float r) {
        uint8_t sign = std::signbit(r) ? 8 : 0;
        float a = std::fabs(r);
        int idx = 0;
        for (int i = 7; i >= 1; i--) {
            if (kV41Fp4Grid[i] <= a) {
                idx = i;
                break;
            }
        }
        return sign | (uint8_t)idx;
    }

    inline float V41DecodeFp4E2M1(uint8_t c) {
        float v = kV41Fp4Grid[c & 7];
        return (c & 8) ? -v : v;
    }

    inline int V41Fp8RowBytes(int dim) {
        return dim + dim / 32;
    }

    // (quantMode, blockSize) -> 每行字节数
    inline int V41KvRowBytes(int dim, int quantMode, int blockSize) {
        return (quantMode == 1 ? dim : dim / 2) + dim / blockSize;
    }

    // 由行宽反推布局；dim >= 32 时三种布局的行宽两两不同
    inline bool V41ParseKvRow(int dim, int rowBytes, int *quantMode, int *blockSize) {
        if (rowBytes == V41KvRowBytes(dim, 1, 32)) {
            *quantMode = 1; *blockSize = 32; return true;
        }
        if (rowBytes == V41KvRowBytes(dim, 3, 16)) {
            *quantMode = 3; *blockSize = 16; return true;
        }
        if (rowBytes == V41KvRowBytes(dim, 2, 32)) {
            *quantMode = 2; *blockSize = 32; return true;
        }
        return false;
    }

    // 把一行 float 按 (quantMode, blockSize) 量化进缓存行。scale 的推导与 DeepSeekV41FakeQuantRow
    // 共用 DeepSeekV41BlockScale，因此对已伪量化过的行是幂等的（存储无损）。
    void V41QuantizeKvRow(const float *row, int dim, int quantMode, int blockSize, uint8_t *dst) {
        const bool fp8 = quantMode == 1;
        const float qmax = DeepSeekV41QuantMax(quantMode);
        uint8_t *scales = dst + (fp8 ? dim : dim / 2);
        for (int start = 0; start < dim; start += blockSize) {
            float amax = 0.0f;
            for (int i = start; i < start + blockSize; i++) {
                amax = std::max(amax, std::fabs(row[i]));
            }
            float scale = DeepSeekV41BlockScale(amax, quantMode);
            if (quantMode == 3) {
                scales[start / blockSize] = V41EncodeFp8E4M3(scale);
            } else {
                int e;
                std::frexp(scale, &e);        // scale = 0.5 * 2^e = 2^(e - 1)
                scales[start / blockSize] = (uint8_t)(e - 1 + 127);
            }
            for (int i = start; i < start + blockSize; i++) {
                float q = std::max(-qmax, std::min(qmax, row[i] / scale));
                if (fp8) {
                    dst[i] = V41EncodeFp8E4M3(V41FP8RoundTrip(q));
                } else {
                    uint8_t code = V41EncodeFp4E2M1(V41FP4RoundTrip(q));
                    if ((i & 1) == 0) {
                        dst[i >> 1] = code;
                    } else {
                        dst[i >> 1] |= (uint8_t)(code << 4);
                    }
                }
            }
        }
    }

    void V41DequantKvRow(const uint8_t *src, int dim, int quantMode, int blockSize, float *dst) {
        const uint8_t *scales = src + (quantMode == 1 ? dim : dim / 2);
        for (int d = 0; d < dim; d++) {
            uint8_t sb = scales[d / blockSize];
            float scale = quantMode == 3 ? V41DecodeFp8E4M3(sb) : std::ldexp(1.0f, (int)sb - 127);
            if (quantMode == 1) {
                dst[d] = V41DecodeFp8E4M3(src[d]) * scale;
            } else {
                uint8_t byte = src[d >> 1];
                dst[d] = V41DecodeFp4E2M1((d & 1) ? (byte >> 4) : (byte & 0xF)) * scale;
            }
        }
    }

    // 把整块量化缓存解成 float [rows, dim]，布局由行宽自动识别
    void V41DequantKvRows(const Data &data, int dim, std::vector<float> &out) {
        const int rowBytes = data.dims.back();
        int quantMode = 1, blockSize = 32;
        AssertInFastLLM(V41ParseKvRow(dim, rowBytes, &quantMode, &blockSize),
                        "DeepSeekV41: unknown quantized KV row layout.\n");
        uint64_t rows = data.Count(0) / rowBytes;
        out.resize(rows * dim);
        V41ParallelFor((int)rows, [&](int st, int end) {
            for (int r = st; r < end; r++) {
                V41DequantKvRow(data.cpuData + (uint64_t)r * rowBytes, dim, quantMode, blockSize,
                                out.data() + (uint64_t)r * dim);
            }
        }, 8);
    }

    void CpuDeepSeekV41QuantizeKVOp::Reshape(const std::string &opType, const DataDict &datas,
                                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int quantMode = V41Int(intParams, "quantMode", 1);
        int blockSize = V41Int(intParams, "quantBlock", 32);
        AssertInFastLLM(input.dims.size() == 3 && V41IsFloatType(input.dataType) &&
                        blockSize > 0 && input.dims[2] % blockSize == 0 &&
                        (quantMode == 1 || ((quantMode == 2 || quantMode == 3) && input.dims[2] % 2 == 0)),
                        "DeepSeekV41QuantizeKV error: input should be float [b, s, d] with d % block == 0.\n");
        output.dataType = DataType::INT8;
        output.Resize({input.dims[0], input.dims[1], V41KvRowBytes(input.dims[2], quantMode, blockSize)});
    }

    void CpuDeepSeekV41QuantizeKVOp::Run(const std::string &opType, const DataDict &datas,
                                         const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int quantMode = V41Int(intParams, "quantMode", 1);
        int blockSize = V41Int(intParams, "quantBlock", 32);
        output.Allocate();
        const int dim = input.dims[2];
        const int rows = input.dims[0] * input.dims[1];
        const int rowBytes = V41KvRowBytes(dim, quantMode, blockSize);
        std::vector<float> values;
        V41ReadFloat(input, values);
        V41ParallelFor(rows, [&](int st, int end) {
            for (int r = st; r < end; r++) {
                V41QuantizeKvRow(values.data() + (uint64_t)r * dim, dim, quantMode, blockSize,
                                 output.cpuData + (uint64_t)r * rowBytes);
            }
        });
    }

    // ---------------- HcMix ----------------

    void CpuDeepSeekV41HcMixOp::Reshape(const std::string &opType, const DataDict &datas,
                                        const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        int hcMult = V41Int(intParams, "hcMult", 1);
        AssertInFastLLM(input.dims.size() == 4 && input.dims[2] == hcMult,
                        "DeepSeekV41HcMix error: input should be [b, s, hc, d].\n");
        int bsz = input.dims[0], seqlen = input.dims[1];
        pre.dataType = DataType::FLOAT32;
        pre.Resize({bsz, seqlen, hcMult});
        post.dataType = DataType::FLOAT32;
        post.Resize({bsz, seqlen, hcMult});
        comb.dataType = DataType::FLOAT32;
        comb.Resize({bsz, seqlen, hcMult, hcMult});
    }

    // 单 token 的 mixes -> pre / post / comb（与 kernel.py::hc_split_sinkhorn 一致）
    void DeepSeekV41SplitSinkhorn(const float *mixes, const float *scale, const float *base,
                                  int hcMult, int sinkhornIters, float eps,
                                  float *pre, float *post, float *comb) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = V41Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[h] = 2.0f * V41Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }
        for (int r = 0; r < hcMult; r++) {
            float rowMax = -FLT_MAX;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c + 2 * hcMult;
                comb[r * hcMult + c] = mixes[idx] * scale[2] + base[idx];
                rowMax = std::max(rowMax, comb[r * hcMult + c]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                float v = std::exp(comb[r * hcMult + c] - rowMax);
                comb[r * hcMult + c] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                comb[r * hcMult + c] = comb[r * hcMult + c] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += comb[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                comb[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += comb[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    comb[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += comb[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    comb[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
    }

    void CpuDeepSeekV41HcMixOp::Run(const std::string &opType, const DataDict &datas,
                                    const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &hcFn = *(datas.find("hcFn")->second);
        Data &hcScale = *(datas.find("hcScale")->second);
        Data &hcBase = *(datas.find("hcBase")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        int hcMult = V41Int(intParams, "hcMult", 1);
        int sinkhornIters = V41Int(intParams, "sinkhornIters", 20);
        float eps = V41Float(floatParams, "eps", 1e-6f);
        float normEps = V41Float(floatParams, "normEps", 1e-6f);

        int bsz = input.dims[0], seqlen = input.dims[1], dim = input.dims[3];
        int tokens = bsz * seqlen;
        int flatDim = hcMult * dim;
        int mixHc = (2 + hcMult) * hcMult;
        AssertInFastLLM(V41IsFloatType(input.dataType) && V41IsFloatType(hcFn.dataType) &&
                        hcScale.dataType == DataType::FLOAT32 && hcBase.dataType == DataType::FLOAT32 &&
                        hcFn.Count(0) == (uint64_t)mixHc * flatDim && hcScale.Count(0) >= 3 &&
                        hcBase.Count(0) >= (uint64_t)mixHc,
                        "DeepSeekV41HcMix error: invalid inputs.\n");
        std::vector<float> fnStorage;
        const float *fn;
        if (hcFn.dataType == DataType::FLOAT32) {
            fn = (const float*)hcFn.cpuData;
        } else {
            V41ReadFloat(hcFn, fnStorage);
            fn = fnStorage.data();
        }
        const float *scale = (const float*)hcScale.cpuData;
        const float *base = (const float*)hcBase.cpuData;
        pre.Allocate();
        post.Allocate();
        comb.Allocate();
        float *preData = (float*)pre.cpuData;
        float *postData = (float*)post.cpuData;
        float *combData = (float*)comb.cpuData;

        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> x(flatDim), mixes(mixHc);
            for (int t = st; t < end; t++) {
                for (int k = 0; k < flatDim; k++) {
                    x[k] = V41ToFloat(input.cpuData, input.dataType, (uint64_t)t * flatDim + k);
                }
                double ss = 0.0;
                for (int k = 0; k < flatDim; k++) {
                    ss += (double)x[k] * x[k];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / flatDim) + normEps);
                for (int m = 0; m < mixHc; m++) {
                    const float *w = fn + (uint64_t)m * flatDim;
                    double v = 0.0;
                    for (int k = 0; k < flatDim; k++) {
                        v += (double)x[k] * w[k];
                    }
                    mixes[m] = (float)v * rsqrt;
                }
                DeepSeekV41SplitSinkhorn(mixes.data(), scale, base, hcMult, sinkhornIters, eps,
                                         preData + (uint64_t)t * hcMult,
                                         postData + (uint64_t)t * hcMult,
                                         combData + (uint64_t)t * hcMult * hcMult);
            }
        }, 4);
    }

    // ---------------- HcApplyPre ----------------

    void CpuDeepSeekV41HcApplyPreOp::Reshape(const std::string &opType, const DataDict &datas,
                                             const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(input.dims.size() == 4, "DeepSeekV41HcApplyPre error: input should be [b, s, hc, d].\n");
        output.dataType = input.dataType;
        output.Resize({input.dims[0], input.dims[1], input.dims[3]});
    }

    void CpuDeepSeekV41HcApplyPreOp::Run(const std::string &opType, const DataDict &datas,
                                         const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &pre = *(datas.find("pre")->second);
        Data &output = *(datas.find("output")->second);
        int bsz = input.dims[0], seqlen = input.dims[1], hcMult = input.dims[2], dim = input.dims[3];
        int tokens = bsz * seqlen;
        AssertInFastLLM(pre.dataType == DataType::FLOAT32 && pre.Count(0) == (uint64_t)tokens * hcMult,
                        "DeepSeekV41HcApplyPre error: invalid pre.\n");
        output.Allocate();
        const float *preData = (const float*)pre.cpuData;
        V41ParallelFor(tokens, [&](int st, int end) {
            for (int t = st; t < end; t++) {
                for (int d = 0; d < dim; d++) {
                    float v = 0.0f;
                    for (int h = 0; h < hcMult; h++) {
                        v += preData[(uint64_t)t * hcMult + h] *
                             V41ToFloat(input.cpuData, input.dataType, ((uint64_t)t * hcMult + h) * dim + d);
                    }
                    V41FromFloat(output.cpuData, output.dataType, (uint64_t)t * dim + d, v);
                }
            }
        }, 8);
    }

    // ---------------- EngramApply ----------------

    void CpuDeepSeekV41EngramApplyOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &hidden = *(datas.find("hidden")->second);
        Data &kv = *(datas.find("kv")->second);
        Data &qWeight = *(datas.find("qWeight")->second);
        Data &kWeight = *(datas.find("kWeight")->second);
        Data *mask = V41Optional(datas, "mask");
        float eps = V41Float(floatParams, "eps", 1e-20f);
        float clampValue = V41Float(floatParams, "clampValue", 1e-6f);

        AssertInFastLLM(hidden.dims.size() == 4, "DeepSeekV41EngramApply error: hidden should be [b, s, hc, d].\n");
        int bsz = hidden.dims[0], seqlen = hidden.dims[1], hcMult = hidden.dims[2], dim = hidden.dims[3];
        int tokens = bsz * seqlen;
        AssertInFastLLM(kv.Count(0) == (uint64_t)tokens * (hcMult + 1) * dim &&
                        qWeight.Count(0) == (uint64_t)hcMult * dim && kWeight.Count(0) == (uint64_t)hcMult * dim &&
                        qWeight.dataType == DataType::FLOAT32 && kWeight.dataType == DataType::FLOAT32,
                        "DeepSeekV41EngramApply error: shape mismatch.\n");
        const float *qw = (const float*)qWeight.cpuData;
        const float *kw = (const float*)kWeight.cpuData;
        std::vector<float> maskValues;
        if (mask != nullptr) {
            V41ReadFloat(*mask, maskValues);
        }
        uint64_t kvStride = (uint64_t)(hcMult + 1) * dim;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> h(dim), key(dim), value(dim);
            for (int t = st; t < end; t++) {
                for (int d = 0; d < dim; d++) {
                    value[d] = V41ToFloat(kv.cpuData, kv.dataType, (uint64_t)t * kvStride + (uint64_t)hcMult * dim + d);
                }
                for (int c = 0; c < hcMult; c++) {
                    uint64_t hOff = ((uint64_t)t * hcMult + c) * dim;
                    double hss = 0.0, kss = 0.0, dot = 0.0;
                    for (int d = 0; d < dim; d++) {
                        h[d] = V41ToFloat(hidden.cpuData, hidden.dataType, hOff + d);
                        key[d] = V41ToFloat(kv.cpuData, kv.dataType, (uint64_t)t * kvStride + (uint64_t)c * dim + d);
                        hss += (double)h[d] * h[d];
                        kss += (double)key[d] * key[d];
                        dot += (double)h[d] * (qw[c * dim + d] * kw[c * dim + d]) * key[d];
                    }
                    float rstd = (1.0f / std::sqrt((float)(hss / dim) + eps)) *
                                 (1.0f / std::sqrt((float)(kss / dim) + eps));
                    float score = (float)dot * rstd * (1.0f / std::sqrt((float)dim));
                    float mag = std::sqrt(std::max(std::fabs(score), clampValue));
                    float gate = V41Sigmoid(std::copysign(mag, score));
                    if (mask != nullptr && maskValues[t] == 0.0f) {
                        gate = 0.0f;
                    }
                    for (int d = 0; d < dim; d++) {
                        V41FromFloat(hidden.cpuData, hidden.dataType, hOff + d, h[d] + gate * value[d]);
                    }
                }
            }
        }, 4);
    }

    // ---------------- RotaryQuant ----------------

    void DeepSeekV41RotaryQuantRows(float *values, int rows, int rowsPerToken, int dim,
                                    int ropeDim, float ropeBase, int startPos, int posStep, bool inverse,
                                    int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                    int quantMode, int quantDim, int quantBlock) {
        auto invFreq = V41InvFreq(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        int pairs = ropeDim / 2;
        int off = dim - ropeDim;
        V41ParallelFor(rows, [&](int st, int end) {
            for (int r = st; r < end; r++) {
                int token = r / rowsPerToken;
                float pos = (float)(startPos + (int64_t)token * posStep);
                float *row = values + (uint64_t)r * dim + off;
                for (int p = 0; p < pairs; p++) {
                    float ang = pos * invFreq[p];
                    float c = std::cos(ang), s = std::sin(ang);
                    if (inverse) {
                        s = -s;
                    }
                    float a = row[2 * p], b = row[2 * p + 1];
                    row[2 * p] = a * c - b * s;
                    row[2 * p + 1] = a * s + b * c;
                }
                if (quantMode > 0) {
                    DeepSeekV41FakeQuantRow(values + (uint64_t)r * dim, quantDim, quantMode, quantBlock);
                }
            }
        }, 16);
    }

    void CpuDeepSeekV41RotaryQuantOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int ropeDim = V41Int(intParams, "ropeDim", 64);
        int startPos = V41Int(intParams, "startPos", 0);
        int posStep = V41Int(intParams, "posStep", 1);
        bool inverse = V41Int(intParams, "inverse", 0) != 0;
        int originalSeqLen = V41Int(intParams, "originalSeqLen", 0);
        int betaFast = V41Int(intParams, "betaFast", 32);
        int betaSlow = V41Int(intParams, "betaSlow", 1);
        int quantMode = V41Int(intParams, "quantMode", 0);
        int quantBlock = V41Int(intParams, "quantBlock", 32);
        float ropeBase = V41Float(floatParams, "ropeBase", 10000.0f);
        float ropeFactor = V41Float(floatParams, "ropeFactor", 1.0f);

        AssertInFastLLM(input.dims.size() == 3 || input.dims.size() == 4,
                        "DeepSeekV41RotaryQuant error: input should be [b, s, d] or [b, s, h, d].\n");
        int dim = input.dims.back();
        int quantDim = V41Int(intParams, "quantDim", dim);
        int rowsPerToken = input.dims.size() == 4 ? input.dims[2] : 1;
        int rows = (int)(input.Count(0) / dim);
        AssertInFastLLM(ropeDim > 0 && ropeDim <= dim && ropeDim % 2 == 0 && quantDim <= dim,
                        "DeepSeekV41RotaryQuant error: invalid params.\n");
        std::vector<float> values;
        V41ReadFloat(input, values);
        DeepSeekV41RotaryQuantRows(values.data(), rows, rowsPerToken, dim, ropeDim, ropeBase, startPos, posStep,
                                   inverse, originalSeqLen, ropeFactor, betaFast, betaSlow,
                                   quantMode, quantDim, quantBlock);
        V41WriteFloat(values, input);
    }

    // ---------------- Compress ----------------

    void CpuDeepSeekV41CompressOp::Reshape(const std::string &opType, const DataDict &datas,
                                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &kv = *(datas.find("kv")->second);
        Data &output = *(datas.find("output")->second);
        int ratio = V41Int(intParams, "compressRatio", 1);
        AssertInFastLLM(kv.dims.size() == 3 && ratio > 0 && kv.dims[1] % ratio == 0,
                        "DeepSeekV41Compress error: kv should be [b, n, d] with n % ratio == 0.\n");
        output.dataType = DataType::BFLOAT16;
        output.Resize({kv.dims[0], kv.dims[1] / ratio, kv.dims[2]});
    }

    void CpuDeepSeekV41CompressOp::Run(const std::string &opType, const DataDict &datas,
                                       const FloatDict &floatParams, const IntDict &intParams) {
        Data &kv = *(datas.find("kv")->second);
        Data *score = V41Optional(datas, "score");
        Data &normWeight = *(datas.find("normWeight")->second);
        Data &output = *(datas.find("output")->second);
        int ratio = V41Int(intParams, "compressRatio", 1);
        float normEps = V41Float(floatParams, "normEps", 1e-20f);
        int bsz = kv.dims[0], n = kv.dims[1], dim = kv.dims[2];
        int blocks = n / ratio;
        AssertInFastLLM(ratio == 1 || (score != nullptr && score->dims == kv.dims),
                        "DeepSeekV41Compress error: score is required when ratio > 1.\n");
        std::vector<float> normValues;
        V41ReadFloat(normWeight, normValues);
        AssertInFastLLM((int)normValues.size() >= dim, "DeepSeekV41Compress error: norm weight mismatch.\n");
        output.Allocate();
        int total = bsz * blocks;
        V41ParallelFor(total, [&](int st, int end) {
            std::vector<float> pooled(dim);
            for (int idx = st; idx < end; idx++) {
                int b = idx / blocks, j = idx % blocks;
                for (int d = 0; d < dim; d++) {
                    if (ratio == 1) {
                        pooled[d] = V41ToFloat(kv.cpuData, kv.dataType, ((uint64_t)b * n + j) * dim + d);
                    } else {
                        float mx = -FLT_MAX;
                        for (int r = 0; r < ratio; r++) {
                            uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                            mx = std::max(mx, V41ToFloat(score->cpuData, score->dataType, off));
                        }
                        double sum = 0.0, value = 0.0;
                        for (int r = 0; r < ratio; r++) {
                            uint64_t off = ((uint64_t)b * n + (uint64_t)j * ratio + r) * dim + d;
                            double e = std::exp((double)V41ToFloat(score->cpuData, score->dataType, off) - mx);
                            sum += e;
                            value += e * V41ToFloat(kv.cpuData, kv.dataType, off);
                        }
                        pooled[d] = (float)(value / sum);
                    }
                    // kv.to(dtype): 归一化前先转 BF16
                    pooled[d] = BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(pooled[d]));
                }
                double ss = 0.0;
                for (int d = 0; d < dim; d++) {
                    ss += (double)pooled[d] * pooled[d];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / dim) + normEps);
                for (int d = 0; d < dim; d++) {
                    ((uint16_t*)output.cpuData)[(uint64_t)idx * dim + d] =
                        Float32ToBFloat16RNEBits(normValues[d] * pooled[d] * rsqrt);
                }
            }
        }, 4);
    }

    // ---------------- IndexerScore ----------------

    void CpuDeepSeekV41IndexerScoreOp::Reshape(const std::string &opType, const DataDict &datas,
                                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &output = *(datas.find("output")->second);
        int qm = 1, qb = 32;
        AssertInFastLLM(q.dims.size() == 4 && k.dims.size() == 3 && q.dims[0] == k.dims[0] &&
                        (k.dataType == DataType::INT8 ? V41ParseKvRow(q.dims[3], k.dims[2], &qm, &qb)
                                                      : q.dims[3] == k.dims[2]),
                        "DeepSeekV41IndexerScore error: q should be [b, s, h, d], k should be [b, m, d].\n");
        output.dataType = DataType::FLOAT32;
        output.Resize({q.dims[0], q.dims[1], k.dims[1]});
    }

    void CpuDeepSeekV41IndexerScoreOp::Run(const std::string &opType, const DataDict &datas,
                                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &weights = *(datas.find("weights")->second);
        Data &k = *(datas.find("k")->second);
        Data &output = *(datas.find("output")->second);
        int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
        int m = k.dims[1];
        AssertInFastLLM(weights.Count(0) == (uint64_t)bsz * seqlen * heads,
                        "DeepSeekV41IndexerScore error: weights should be [b, s, h].\n");
        std::vector<float> qv, wv, kvv;
        V41ReadFloat(q, qv);
        V41ReadFloat(weights, wv);
        if (k.dataType == DataType::INT8) {
            V41DequantKvRows(k, dim, kvv);      // indexer key 可以是打包 FP4 的缓存行
        } else {
            V41ReadFloat(k, kvv);
        }
        output.Allocate();
        float *out = (float*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            for (int t = st; t < end; t++) {
                int b = t / seqlen;
                const float *qrow = qv.data() + (uint64_t)t * heads * dim;
                const float *w = wv.data() + (uint64_t)t * heads;
                float *orow = out + (uint64_t)t * m;
                for (int j = 0; j < m; j++) {
                    const float *krow = kvv.data() + ((uint64_t)b * m + j) * dim;
                    float total = 0.0f;
                    for (int h = 0; h < heads; h++) {
                        const float *qh = qrow + (uint64_t)h * dim;
                        float dot = 0.0f;
                        for (int d = 0; d < dim; d++) {
                            dot += qh[d] * krow[d];
                        }
                        total += std::max(dot, 0.0f) * w[h];
                    }
                    orow[j] = total;
                }
            }
        });
    }

    // ---------------- CandidateBlocks ----------------

    void CpuDeepSeekV41CandidateBlocksOp::Reshape(const std::string &opType, const DataDict &datas,
                                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int blockSize = V41Int(intParams, "blockSize", 8);
        AssertInFastLLM(score.dims.size() == 3 && blockSize > 0,
                        "DeepSeekV41CandidateBlocks error: score should be [b, s, m].\n");
        int numBlocks = (score.dims[2] + blockSize - 1) / blockSize;
        output.dataType = DataType::INT8;
        output.Resize({score.dims[0], score.dims[1], numBlocks});
    }

    void CpuDeepSeekV41CandidateBlocksOp::Run(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int blockSize = V41Int(intParams, "blockSize", 8);
        int topkBlocks = V41Int(intParams, "topkBlocks", 0);
        int compressRatio = V41Int(intParams, "compressRatio", 1);
        int startPos = V41Int(intParams, "startPos", 0);
        int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
        int numBlocks = (m + blockSize - 1) / blockSize;
        AssertInFastLLM(score.dataType == DataType::FLOAT32, "DeepSeekV41CandidateBlocks error: score should be float32.\n");
        output.Allocate();
        const float *sc = (const float*)score.cpuData;
        uint8_t *out = (uint8_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<float> blockScore(numBlocks);
            std::vector<int> order(numBlocks);
            for (int t = st; t < end; t++) {
                int i = t % seqlen;
                int visible = std::min(m, (startPos + i + 1) / compressRatio);
                const float *row = sc + (uint64_t)t * m;
                for (int k = 0; k < numBlocks; k++) {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int j = k * blockSize; j < std::min(m, (k + 1) * blockSize); j++) {
                        if (j < visible) {
                            mx = std::max(mx, row[j]);
                        }
                    }
                    blockScore[k] = mx;
                }
                if (visible > 0) {
                    blockScore[(visible - 1) / blockSize] = std::numeric_limits<float>::infinity();
                }
                uint8_t *orow = out + (uint64_t)t * numBlocks;
                memset(orow, 0, numBlocks);
                int keep = std::min(topkBlocks, numBlocks);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                                  [&](int a, int b) {
                                      return blockScore[a] > blockScore[b] || (blockScore[a] == blockScore[b] && a < b);
                                  });
                for (int k = 0; k < keep; k++) {
                    if (blockScore[order[k]] > -std::numeric_limits<float>::infinity()) {
                        orow[order[k]] = 1;
                    }
                }
            }
        });
    }

    // ---------------- IndexerTopK ----------------

    void CpuDeepSeekV41IndexerTopKOp::Reshape(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data &output = *(datas.find("output")->second);
        int topK = V41Int(intParams, "topK", 0);
        AssertInFastLLM(score.dims.size() == 3 && topK > 0, "DeepSeekV41IndexerTopK error: score should be [b, s, m].\n");
        output.dataType = DataType::INT32;
        output.Resize({score.dims[0], score.dims[1], std::min(topK, score.dims[2])});
    }

    void CpuDeepSeekV41IndexerTopKOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &score = *(datas.find("score")->second);
        Data *candidates = V41Optional(datas, "candidates");
        Data &output = *(datas.find("output")->second);
        int topK = V41Int(intParams, "topK", 0);
        int compressRatio = V41Int(intParams, "compressRatio", 1);
        int startPos = V41Int(intParams, "startPos", 0);
        int blockSize = V41Int(intParams, "blockSize", 8);
        int bsz = score.dims[0], seqlen = score.dims[1], m = score.dims[2];
        int width = std::min(topK, m);
        AssertInFastLLM(score.dataType == DataType::FLOAT32, "DeepSeekV41IndexerTopK error: score should be float32.\n");
        int numBlocks = candidates == nullptr ? 0 : candidates->dims.back();
        output.Allocate();
        const float *sc = (const float*)score.cpuData;
        int32_t *out = (int32_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<int> order;
            std::vector<float> masked(m);
            for (int t = st; t < end; t++) {
                int i = t % seqlen;
                int visible = std::min(m, (startPos + i + 1) / compressRatio);
                const float *row = sc + (uint64_t)t * m;
                const uint8_t *cand = candidates == nullptr ? nullptr :
                    (const uint8_t*)candidates->cpuData + (uint64_t)t * numBlocks;
                order.clear();
                for (int j = 0; j < visible; j++) {
                    if (cand != nullptr && (j / blockSize >= numBlocks || cand[j / blockSize] == 0)) {
                        continue;
                    }
                    order.push_back(j);
                }
                int keep = std::min(width, (int)order.size());
                std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                                  [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
                std::sort(order.begin(), order.begin() + keep);
                int32_t *orow = out + (uint64_t)t * width;
                for (int k = 0; k < width; k++) {
                    orow[k] = k < keep ? order[k] : -1;
                }
            }
        });
    }

    // ---------------- SparseAttention ----------------

    void CpuDeepSeekV41SparseAttentionOp::Reshape(const std::string &opType, const DataDict &datas,
                                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(q.dims.size() == 4, "DeepSeekV41SparseAttention error: q should be [b, s, h, d].\n");
        output.dataType = DataType::BFLOAT16;
        output.Resize(q.dims);
    }

    void CpuDeepSeekV41SparseAttentionOp::Run(const std::string &opType, const DataDict &datas,
                                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &chunkKV = *(datas.find("chunkKV")->second);
        Data *ringKV = V41Optional(datas, "ringKV");
        Data *compressedKV = V41Optional(datas, "compressedKV");
        Data *cmpIdx = V41Optional(datas, "cmpIdx");
        Data &attnSink = *(datas.find("attnSink")->second);
        Data &output = *(datas.find("output")->second);
        int windowSize = V41Int(intParams, "windowSize", 128);
        int startPos = V41Int(intParams, "startPos", 0);
        float softmaxScale = V41Float(floatParams, "softmaxScale", 1.0f);

        int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
        AssertInFastLLM(chunkKV.dims.size() == 3 && chunkKV.dims[0] == bsz && chunkKV.dims[1] == seqlen &&
                        chunkKV.dims[2] == dim && attnSink.Count(0) >= (uint64_t)heads,
                        "DeepSeekV41SparseAttention error: chunkKV / sink shape mismatch.\n");
        bool hasRing = ringKV != nullptr && ringKV->dims.size() == 3 && ringKV->Count(0) > 0;
        bool hasCompressed = compressedKV != nullptr && cmpIdx != nullptr && compressedKV->Count(0) > 0 &&
                             cmpIdx->Count(0) > 0;
        int ringRows = hasRing ? ringKV->dims[1] : 0;
        int cap = hasCompressed ? compressedKV->dims[1] : 0;
        int topWidth = hasCompressed ? cmpIdx->dims[2] : 0;
        // 缓存行可以是 BF16/FP32 [.., dim]，也可以是量化的 INT8 行（FP8 或打包 FP4，见上面的布局说明）
        bool ringQuant = hasRing && ringKV->dataType == DataType::INT8;
        bool compQuant = hasCompressed && compressedKV->dataType == DataType::INT8;
        int qm = 1, qb = 32;
        AssertInFastLLM(!hasRing || (ringKV->dims[0] == bsz && ringRows == windowSize &&
                                     (ringQuant ? V41ParseKvRow(dim, ringKV->dims[2], &qm, &qb)
                                                : ringKV->dims[2] == dim)),
                        "DeepSeekV41SparseAttention error: ring shape mismatch.\n");
        AssertInFastLLM(!hasCompressed || (cmpIdx->dataType == DataType::INT32 && cmpIdx->dims.size() == 3 &&
                                           cmpIdx->dims[0] == bsz && cmpIdx->dims[1] == seqlen &&
                                           (compQuant ? V41ParseKvRow(dim, compressedKV->dims[2], &qm, &qb)
                                                      : compressedKV->dims[2] == dim)),
                        "DeepSeekV41SparseAttention error: compressed shape mismatch.\n");
        std::vector<float> qv, chunk, ring, comp, sink;
        V41ReadFloat(q, qv);
        V41ReadFloat(chunkKV, chunk);
        if (hasRing) {
            if (ringQuant) {
                V41DequantKvRows(*ringKV, dim, ring);
            } else {
                V41ReadFloat(*ringKV, ring);
            }
        }
        if (hasCompressed) {
            if (compQuant) {
                V41DequantKvRows(*compressedKV, dim, comp);
            } else {
                V41ReadFloat(*compressedKV, comp);
            }
        }
        V41ReadFloat(attnSink, sink);
        output.Allocate();
        uint16_t *out = (uint16_t*)output.cpuData;
        int tokens = bsz * seqlen;
        V41ParallelFor(tokens, [&](int st, int end) {
            std::vector<const float*> rows;
            std::vector<float> scores;
            std::vector<float> acc(dim);
            for (int t = st; t < end; t++) {
                int b = t / seqlen, i = t % seqlen;
                int pos = startPos + i;
                rows.clear();
                for (int p = std::max(0, pos - windowSize + 1); p <= pos; p++) {
                    if (p >= startPos) {
                        rows.push_back(chunk.data() + ((uint64_t)b * seqlen + (p - startPos)) * dim);
                    } else {
                        AssertInFastLLM(hasRing, "DeepSeekV41SparseAttention error: ring cache is missing.\n");
                        rows.push_back(ring.data() + ((uint64_t)b * ringRows + (p % windowSize)) * dim);
                    }
                }
                if (hasCompressed) {
                    const int32_t *idx = (const int32_t*)cmpIdx->cpuData + (uint64_t)t * topWidth;
                    for (int k = 0; k < topWidth; k++) {
                        if (idx[k] >= 0 && idx[k] < cap) {
                            rows.push_back(comp.data() + ((uint64_t)b * cap + idx[k]) * dim);
                        }
                    }
                }
                scores.resize(rows.size());
                for (int h = 0; h < heads; h++) {
                    const float *qrow = qv.data() + ((uint64_t)t * heads + h) * dim;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (size_t k = 0; k < rows.size(); k++) {
                        double dot = 0.0;
                        for (int d = 0; d < dim; d++) {
                            dot += (double)qrow[d] * rows[k][d];
                        }
                        scores[k] = (float)dot * softmaxScale;
                        mx = std::max(mx, scores[k]);
                    }
                    float safeMx = std::isfinite(mx) ? mx : 0.0f;
                    double denom = std::exp((double)sink[h] - safeMx);
                    std::fill(acc.begin(), acc.end(), 0.0f);
                    for (size_t k = 0; k < rows.size(); k++) {
                        double w = std::exp((double)scores[k] - safeMx);
                        denom += w;
                        for (int d = 0; d < dim; d++) {
                            acc[d] += (float)w * rows[k][d];
                        }
                    }
                    float inv = (float)(1.0 / std::max(denom, 1e-30));
                    uint16_t *orow = out + ((uint64_t)t * heads + h) * dim;
                    for (int d = 0; d < dim; d++) {
                        orow[d] = Float32ToBFloat16RNEBits(acc[d] * inv);
                    }
                }
            }
        });
    }

    // ---------------- WindowStore ----------------

    void CpuDeepSeekV41WindowStoreOp::Run(const std::string &opType, const DataDict &datas,
                                          const FloatDict &floatParams, const IntDict &intParams) {
        Data &chunk = *(datas.find("chunk")->second);
        Data &ring = *(datas.find("ring")->second);
        int startPos = V41Int(intParams, "startPos", 0);
        int windowSize = V41Int(intParams, "windowSize", 128);
        AssertInFastLLM(chunk.dims.size() == 3, "DeepSeekV41WindowStore error: chunk should be [b, s, d].\n");
        int bsz = chunk.dims[0], seqlen = chunk.dims[1], dim = chunk.dims[2];
        if (ring.dims.size() != 3 || ring.dims[0] != bsz || ring.dims[1] != windowSize || ring.dims[2] != dim ||
            ring.cpuData == nullptr) {
            ring.dataType = chunk.dataType;
            ring.Resize({bsz, windowSize, dim});
            ring.Allocate();
        }
        AssertInFastLLM(ring.dataType == chunk.dataType, "DeepSeekV41WindowStore error: dtype mismatch.\n");
        int unit = chunk.unitSize;
        for (int b = 0; b < bsz; b++) {
            for (int i = std::max(0, seqlen - windowSize); i < seqlen; i++) {
                int slot = (startPos + i) % windowSize;
                memcpy(ring.cpuData + ((uint64_t)b * windowSize + slot) * dim * unit,
                       chunk.cpuData + ((uint64_t)b * seqlen + i) * dim * unit,
                       (uint64_t)dim * unit);
            }
        }
    }
}
