// Reference numerical profile reused from the preserved local V4.1 implementation.
#ifndef FASTLLM_DEEPSEEKV41_REFERENCE_MATH_H
#define FASTLLM_DEEPSEEKV41_REFERENCE_MATH_H

#include "fastllm.h"
namespace fastllm {
    enum class V41ReferenceOp {
        Linear,
        Embedding,
        RMSNorm,
        HCMixes,
        HCPre,
        HCPost,
        Rotary,
        Quantize,
        Compress,
        IndexScores,
        IndexSelect,
        SparseAttention,
        QuantizedLinear,
        HCPreNorm,
        CacheUpdate
    };
}
#include <cmath>
#include <cstdint>
#include <cstddef>

#ifdef __CUDACC__
#define V41_HD __host__ __device__
#else
#define V41_HD
#endif

namespace fastllm {
    namespace v41ref {
        // Shared scalar specification. CUDA assigns independent rows/elements to
        // threads, preserving quantization and reduction boundaries on both devices.
        enum Buffer {
            Input,
            Weight,
            Scale,
            Base,
            Mix,
            Post,
            Comb,
            Residual,
            Query,
            Keys,
            HeadWeights,
            Candidates,
            Indices,
            Sink,
            Scores,
            Values,
            TailKV,
            TailScore,
            Output,
            Output1,
            Output2,
            Output3,
            Output4,
            BufferCount
        };
        struct Task {
            V41ReferenceOp op;
            float *p[BufferCount] = {};
            int rows = 0, cols = 0, out = 0, width = 0, groups = 1, dtype = 0;
            int heads = 0, dim = 0, h = 4, iterations = 20, block = 32, mode = 0;
            int start = 0, stride = 1, ropeDim = 64, original = 65536, inverse = 0;
            int ratio = 0, visible = 0, offset = 0, window = 128, topk = 512;
            int candidateBlocks = 2048, candidateBlock = 8, buildCandidates = 0, useCandidates = 0;
            float eps = 1e-20f, hcEps = 1e-6f, theta = 10000, factor = 16, low = 0, high = 0;
            size_t work = 0;
        };
        V41_HD inline float Min(float a, float b) { return a < b ? a : b; }
        V41_HD inline float Max(float a, float b) { return a > b ? a : b; }
        V41_HD inline int MinI(int a, int b) { return a < b ? a : b; }
        V41_HD inline int MaxI(int a, int b) { return a > b ? a : b; }
        // Evaluate transcendentals at higher precision before the explicit FP32
        // rounding boundary; CUDA's fast expf can move BF16 halfway cases.
        V41_HD inline float Exp(float value) { return float(exp(double(value))); }
        V41_HD inline uint32_t Bits(float v) {
            union {
                float f;
                uint32_t u;
            } b;
            b.f = v;
            return b.u;
        }
        V41_HD inline float Float(uint32_t v) {
            union {
                float f;
                uint32_t u;
            } b;
            b.u = v;
            return b.f;
        }
        // FP32 standalone exp profile: range reduction on a 1/128 ln(2)
        // grid, a quadratic residual, and a split power-of-two reconstruction.
        // Generate the grid values mathematically, without a library data table.
        // Separate multiply/add rounding is intentional (VML's SSE2 HA profile).
        V41_HD inline float TensorExp(float x) {
            if (!(x >= -80 && x <= 80))
                return Exp(x);
            float n = nearbyintf(x * float(128.0 / 0.69314718055994530942));
            int k = int(n), j = k & 127;
            const float hi = 0.0054168701171875f;
            const float lo = float(0.69314718055994530942 / 128.0 - double(hi));
            float r = (x - n * hi) - n * lo;
            double grid = exp2(double(j) / 128.0);
            float upper = float(grid), lower = float(grid - double(upper));
            float residual = r + r * r * 0.5f;
            return ldexpf(upper + (upper * residual + lower), k >> 7);
        }
        // Adapted from Arm optimized-routines math/sincosf{,_data}.c and
        // sincosf.h, Copyright (c) 2018-2024 Arm Limited. MIT license:
        // third_party/arm-optimized-routines/LICENSE.txt.
        V41_HD inline void SinCos(float angle, float &sine, float &cosine) {
            uint32_t bits = Bits(angle), top = (bits >> 20) & 0x7ff;
            double x = angle;
            int quadrant = 0, sign = 0;
            if (top < ((Bits(0x1.921FB6p-1f) >> 20) & 0x7ff)) {
                if (top < ((Bits(0x1p-12f) >> 20) & 0x7ff)) {
                    sine = angle;
                    cosine = 1;
                    return;
                }
            } else if (top < ((Bits(120.0f) >> 20) & 0x7ff)) {
                quadrant = int(nearbyint(x * 0x1.45F306DC9C883p-1));
                x -= quadrant * 0x1.921FB54442D18p0;
            } else if (top < 0x7f8) {
                const uint32_t inv[24] = {0xa2,       0xa2f9,     0xa2f983,   0xa2f9836e, 0xf9836e4e,
                                          0x836e4e44, 0x6e4e4415, 0x4e441529, 0x441529fc, 0x1529fc27,
                                          0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd,
                                          0xf534ddc0, 0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599,
                                          0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041};
                const uint32_t *table = inv + ((bits >> 26) & 15);
                uint32_t mantissa = ((bits & 0xffffff) | 0x800000) << ((bits >> 23) & 7);
                uint64_t a = uint32_t(mantissa * table[0]);
                uint64_t b = uint64_t(mantissa) * table[4], c = uint64_t(mantissa) * table[8];
                uint64_t remainder = ((c >> 32) | (a << 32)) + b;
                uint64_t n = (remainder + (uint64_t(1) << 61)) >> 62;
                remainder -= n << 62;
                x = double(int64_t(remainder)) * 0x1.921FB54442D18p-62;
                quadrant = int(n);
                sign = int(bits >> 31);
            } else {
                sine = cosine = angle - angle;
                return;
            }
            const double signs[4] = {1, -1, -1, 1};
            double x2 = x * x;
            x *= signs[(quadrant + sign) & 3];
            double x3 = x * x2, x4 = x2 * x2, x5 = x3 * x2, x6 = x4 * x2;
            double s =
                (x + x3 * -0x1.555545995a603p-3) + x5 * (0x1.1107605230bc4p-7 + x2 * -0x1.994eb3774cf24p-13);
            double c = ((1 + x2 * -0x1.ffffffd0c621cp-2) + x4 * 0x1.55553e1068f19p-5) +
                       x6 * (-0x1.6c087e89a359dp-10 + x2 * 0x1.99343027bf8c3p-16);
            if ((quadrant + sign) & 2)
                c = -c;
            sine = float((quadrant & 1) ? c : s);
            cosine = float((quadrant & 1) ? s : c);
        }
        // Scalar adaptation of SLEEF xexpf.
        // Copyright Naoki Shibata and contributors 2010 - 2025.
        // Boost Software License 1.0; see third_party/sleef/LICENSE.txt.
        V41_HD inline float ReductionExp(float d) {
            if (d < -104)
                return 0;
            if (d > 100)
                return INFINITY;
            if (d != d)
                return d;
            int q = int(nearbyintf(d * 1.4426950408889634074f));
            float s = fmaf(float(q), -0.693145751953125f, d);
            s = fmaf(float(q), -1.428606765330187045e-6f, s);
            float u = 0.000198527617612853646278381f;
            u = fmaf(u, s, 0.00139304355252534151077271f);
            u = fmaf(u, s, 0.00833336077630519866943359f);
            u = fmaf(u, s, 0.0416664853692054748535156f);
            u = fmaf(u, s, 0.166666671633720397949219f);
            u = fmaf(u, s, 0.5f);
            u = 1.0f + fmaf(s * s, u, s);
            int half = q >> 1;
            return (u * Float(uint32_t(half + 127) << 23)) * Float(uint32_t(q - half + 127) << 23);
        }
        V41_HD inline float BF(float v) {
            uint32_t u = Bits(v);
            if ((u & 0x7fffffff) > 0x7f800000)
                return Float((u & 0xffff0000) | 0x00400000);
            return Float((u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000);
        }
        V41_HD inline float FP8(int code) {
            int sign = code & 128, e = (code >> 3) & 15, m = code & 7;
            float v = e ? ldexpf(1.0f + m * .125f, e - 7) : m * (1.0f / 512);
            return sign ? -v : v;
        }
        V41_HD inline float RoundFP8(float x) {
            float a = Min(fabsf(x), 448.0f);
            int e = int((Bits(a) >> 23) & 255) - 127;
            float step = e < -6 ? 1.0f / 512 : ldexpf(1.0f, e - 3);
            float v = nearbyintf(a / step) * step;
            return copysignf(v, x);
        }
        V41_HD inline float FP4(int code) {
            const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
            return (code & 8) ? -values[code & 7] : values[code & 7];
        }
        V41_HD inline float Value(const Task &t, size_t i) {
            if (t.dtype == 0)
                return t.p[Weight][i];
            if (t.dtype == 1)
                return Float(uint32_t(reinterpret_cast<const uint16_t *>(t.p[Weight])[i]) << 16);
            auto bytes = reinterpret_cast<const uint8_t *>(t.p[Weight]);
            return t.dtype == 2 ? FP8(bytes[i]) : FP4((bytes[i / 2] >> ((i & 1) * 4)) & 15);
        }
        struct Load {
            const float *p;
            bool square = false;
            V41_HD float operator()(size_t i) const {
                float x = p[i];
                return square ? x * x : x;
            }
        };
        // PyTorch CPU cascade sum (BSD, third_party/pytorch/LICENSE).
        template <class F> V41_HD float Sum(size_t count, const F &load) {
            if (count < 8) {
                float partial[4] = {};
                size_t i = 0;
                for (; i + 4 <= count; i += 4)
                    for (int j = 0; j < 4; ++j)
                        partial[j] += load(i + j);
                for (; i < count; ++i)
                    partial[0] += load(i);
                for (int j = 1; j < 4; ++j)
                    partial[0] += partial[j];
                return partial[0];
            }
            size_t groups = count / 32, bits = 0;
            for (size_t v = groups ? groups - 1 : 0; v; v >>= 1)
                ++bits;
            size_t power = bits / 4 > 4 ? bits / 4 : 4, step = size_t(1) << power, group = 0;
            float sums[4][4][8] = {};
            while (group + step <= groups) {
                for (size_t end = group + step; group < end; ++group)
                    for (int s = 0; s < 4; ++s)
                        for (int l = 0; l < 8; ++l)
                            sums[0][s][l] += load(group * 32 + s * 8 + l);
                for (int level = 1; level < 4; ++level) {
                    for (int s = 0; s < 4; ++s)
                        for (int l = 0; l < 8; ++l) {
                            sums[level][s][l] += sums[level - 1][s][l];
                            sums[level - 1][s][l] = 0;
                        }
                    if (group & ((step - 1) << (level * power)))
                        break;
                }
            }
            for (; group < groups; ++group)
                for (int s = 0; s < 4; ++s)
                    for (int l = 0; l < 8; ++l)
                        sums[0][s][l] += load(group * 32 + s * 8 + l);
            for (int level = 1; level < 4; ++level)
                for (int s = 0; s < 4; ++s)
                    for (int l = 0; l < 8; ++l)
                        sums[0][s][l] += sums[level][s][l];
            for (size_t i = groups * 32; i + 8 <= count; i += 8)
                for (int l = 0; l < 8; ++l)
                    sums[0][0][l] += load(i + l);
            for (int s = 1; s < 4; ++s)
                for (int l = 0; l < 8; ++l)
                    sums[0][0][l] += sums[0][s][l];
            float total = 0;
            for (size_t i = count / 8 * 8; i < count; ++i)
                total += load(i);
            for (int l = 0; l < 8; ++l)
                total += sums[0][0][l];
            return total;
        }
        V41_HD inline void HCMixesFromInv(const Task &t, int row, float inv) {
            int h = t.h, m = 2 * h + h * h;
            float *y = t.p[Output];
            t.p[Output4][row] = inv;
            float *mix = t.p[Output3] + row * m, *comb = t.p[Output2] + row * h * h;
            for (int j = 0; j < m; ++j)
                mix[j] = t.p[Mix][row * m + j] * inv;
            for (int j = 0; j < h; ++j) {
                float pre = -(mix[j] * t.p[Scale][0] + t.p[Base][j]);
                float post = -(mix[h + j] * t.p[Scale][1] + t.p[Base][h + j]);
                bool vector = row * h + j < (t.rows * h) / 32 * 32;
                y[row * h + j] = 1.0f / (1.0f + (vector ? ReductionExp(pre) : Exp(pre))) + t.hcEps;
                t.p[Output1][row * h + j] = 2.0f / (1.0f + (vector ? ReductionExp(post) : Exp(post)));
            }
            for (int j = 0; j < h * h; ++j)
                comb[j] = mix[2 * h + j] * t.p[Scale][2] + t.p[Base][2 * h + j];
            for (int i = 0; i < h; ++i) {
                float maximum = -INFINITY, total = 0;
                for (int j = 0; j < h; ++j)
                    maximum = Max(maximum, comb[i * h + j]);
                for (int j = 0; j < h; ++j) {
                    comb[i * h + j] = ReductionExp(comb[i * h + j] - maximum);
                    total += comb[i * h + j];
                }
                float reciprocal = 1.0f / total;
                for (int j = 0; j < h; ++j)
                    comb[i * h + j] = comb[i * h + j] * reciprocal + t.hcEps;
            }
            for (int it = 0; it < t.iterations; ++it) {
                if (it)
                    for (int i = 0; i < h; ++i) {
                        float total = 0;
                        for (int j = 0; j < h; ++j)
                            total += comb[i * h + j];
                        for (int j = 0; j < h; ++j)
                            comb[i * h + j] /= total + t.hcEps;
                    }
                for (int j = 0; j < h; ++j) {
                    float total = 0;
                    for (int i = 0; i < h; ++i)
                        total += comb[i * h + j];
                    for (int i = 0; i < h; ++i)
                        comb[i * h + j] /= total + t.hcEps;
                }
            }
        }
        V41_HD inline float HCPreValue(const Task &t, size_t row, int j) {
            float sum = 0;
            for (int h = 0; h < t.h; ++h)
                sum += t.p[Mix][row * t.h + h] * t.p[Input][row * t.cols + h * t.dim + j];
            return BF(sum);
        }
        struct HCPreSquares {
            const Task &task;
            size_t row;
            V41_HD float operator()(size_t j) const {
                float v = HCPreValue(task, row, int(j));
                return v * v;
            }
        };
        V41_HD inline void Execute(const Task &t, size_t item) {
            float *y = t.p[Output];
            const float *x = t.p[Input];
            if (t.op == V41ReferenceOp::Embedding) {
                int token = int(item / t.cols), j = int(item % t.cols);
                int id = reinterpret_cast<const int *>(t.p[Indices])[token];
                y[item] = id >= 0 && id < t.out ? Value(t, size_t(id) * t.dim + j % t.dim) : Float(0x7fc00000);
            } else if (t.op == V41ReferenceOp::Quantize) {
                size_t base = item * t.block;
                float maximum = t.mode == 0 ? 1e-4f : (t.mode == 1 ? 6.0f / 512 : ldexpf(6.0f, -126));
                for (int j = 0; j < t.block; ++j)
                    maximum = Max(maximum, fabsf(x[base + j]));
                float normalized = maximum * (t.mode == 0 ? 1.0f / 448 : 1.0f / 6), scale;
                if (t.mode == 1)
                    scale = RoundFP8(maximum / 6);
                else {
                    uint32_t bits = Bits(normalized);
                    scale = ldexpf(1.0f, int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0));
                }
                const float boundaries[7] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5};
                for (int j = 0; j < t.block; ++j) {
                    float v = x[base + j] / scale;
                    if (t.mode == 0)
                        y[base + j] = BF(RoundFP8(v) * scale);
                    else {
                        float mag = Min(6.0f, fabsf(v));
                        int code = 0;
                        while (code < 7 && (mag > boundaries[code] || (mag == boundaries[code] && (code & 1))))
                            ++code;
                        if (Bits(v) >> 31)
                            code |= 8;
                        y[base + j] = BF(FP4(code) * scale);
                    }
                }
            } else if (t.op == V41ReferenceOp::Linear) {
                int row = int(item % t.out), token = int(item / t.out);
                const float *a = x + size_t(token) * t.cols + (row / (t.out / t.groups)) * t.width;
                float value = 0;
                if (t.dtype >= 2) {
                    int sr = t.dtype == 2 ? row / 32 : row;
                    for (int b = 0; b < t.width / 32; ++b) {
                        float dot = 0;
                        for (int j = 0; j < 32; ++j) {
                            int col = b * 32 + j;
                            dot += a[col] * Value(t, size_t(row) * t.width + col);
                        }
                        value += dot * t.p[Scale][size_t(sr) * (t.width / 32) + b];
                    }
                } else if (t.dtype == 1 && t.mode) {
                    int block = t.rows > 1 && t.width > 1024 ? 1024 : 512;
                    for (int b = 0; b < t.width; b += block) {
                        float partial = 0;
                        for (int k = b; k < MinI(b + block, t.width); k += 2) {
                            // VDPBF16PS accumulates the high pair before the low pair.
                            partial = fmaf(a[k + 1], Value(t, size_t(row) * t.width + k + 1), partial);
                            partial = fmaf(a[k], Value(t, size_t(row) * t.width + k), partial);
                        }
                        value += partial;
                    }
                } else {
                    // Match the reference LP64 SGEMM profile: complete 16x2/4
                    // tiles use FMA with K=192; fringe rows/columns use a
                    // four-lane non-fused dot over the entire K dimension.
                    if (row < t.out / 16 * 16 && token < t.rows / 2 * 2) {
                        for (int start = 0; start < t.width; start += 192) {
                            float partial = 0;
                            for (int k = start; k < MinI(start + 192, t.width); ++k)
                                partial = fmaf(a[k], Value(t, size_t(row) * t.width + k), partial);
                            value += partial;
                        }
                    } else {
                        float partial[4] = {};
                        // The singleton GEMV starts with four scalar products
                        // before its SIMD loop (aligned native parameter storage).
                        int prefix = t.rows == 1 ? MinI(4, t.width) : 0;
                        for (int k = 0; k < prefix; ++k)
                            partial[0] += a[k] * Value(t, size_t(row) * t.width + k);
                        for (int k = prefix; k < t.width; ++k)
                            partial[k % 4] += a[k] * Value(t, size_t(row) * t.width + k);
                        value = (partial[0] + partial[2]) + (partial[1] + partial[3]);
                    }
                }
                y[item] = t.mode || t.dtype >= 2 ? BF(value) : value;
            } else if (t.op == V41ReferenceOp::RMSNorm) {
                int row = int(item);
                float inv = 1.0f / sqrtf(Sum(t.cols, Load{x + size_t(row) * t.cols, true}) / t.cols + t.eps);
                for (int j = 0; j < t.cols; ++j)
                    y[size_t(row) * t.cols + j] = BF((x[size_t(row) * t.cols + j] * inv) * Value(t, j));
            } else if (t.op == V41ReferenceOp::HCPreNorm) {
                // Keep the rounded HCPre value before the exact RMS cascade.
                // This scalar path also supplies the portable CPU implementation.
                float inv = 1.0f / sqrtf(Sum(t.dim, HCPreSquares{t, item}) / t.dim + t.eps);
                for (int j = 0; j < t.dim; ++j)
                    y[item * t.dim + j] = BF((HCPreValue(t, item, j) * inv) * Value(t, j));
            } else if (t.op == V41ReferenceOp::HCPre) {
                int row = int(item / t.dim), j = int(item % t.dim);
                float sum = 0;
                for (int h = 0; h < t.h; ++h)
                    sum += t.p[Mix][row * t.h + h] * x[size_t(row) * t.cols + h * t.dim + j];
                y[item] = BF(sum);
            } else if (t.op == V41ReferenceOp::HCPost) {
                int row = int(item / (t.h * t.dim)), h = int(item / t.dim) % t.h, j = int(item % t.dim);
                float sum = 0;
                for (int in = 0; in < t.h; ++in)
                    sum += t.p[Comb][row * t.h * t.h + in * t.h + h] *
                           t.p[Residual][size_t(row) * t.h * t.dim + in * t.dim + j];
                y[item] = BF(t.p[Post][row * t.h + h] * x[size_t(row) * t.dim + j] + sum);
            } else if (t.op == V41ReferenceOp::HCMixes) {
                int row = int(item);
                float inv = 1.0f / sqrtf(Sum(t.cols, Load{x + size_t(row) * t.cols, true}) / t.cols + t.eps);
                HCMixesFromInv(t, row, inv);
            } else if (t.op == V41ReferenceOp::Rotary) {
                int pairs = t.ropeDim / 2, head = int(item / pairs), i = int(item % pairs),
                    token = head / (t.cols / t.dim);
                float freq = 1.0f / float(pow(double(t.theta), double(float(2 * i) / t.ropeDim)));
                if (t.mode && t.original > 0) {
                    float smooth = 1.0f - Min(1.0f, Max(0.0f, (i - t.low) / Max(t.high - t.low, 1e-3f)));
                    freq = freq / t.factor * (1 - smooth) + freq * smooth;
                }
                float angle = (t.start + token * t.stride) * freq;
                float c, s;
                SinCos(angle, s, c);
                if (t.inverse)
                    s = -s;
                size_t base = size_t(head) * t.dim + t.dim - t.ropeDim + 2 * i;
                float re = x[base], im = x[base + 1];
                y[base] = BF(re * c - im * s);
                y[base + 1] = BF(im * c + re * s);
            } else if (t.op == V41ReferenceOp::CacheUpdate) {
                int row = int(item / t.cols), col = int(item % t.cols);
                if (t.mode == 1 && t.visible == 0) {
                    // Initialize every ring slot and retain only the last write
                    // when a prefill wraps around the ring multiple times.
                    int source = (row - t.start % t.window + t.window) % t.window;
                    if (source < t.rows) {
                        source += (t.rows - 1 - source) / t.window * t.window;
                        y[item] = x[size_t(source) * t.cols + col];
                    } else
                        y[item] = 0;
                } else {
                    int source = t.mode == 1 ? t.rows - MinI(t.rows, t.window) + row : row;
                    int dest = t.mode == 1 ? (t.start + source) % t.window : t.start + source;
                    y[size_t(dest) * t.cols + col] = x[size_t(source) * t.cols + col];
                }
            } else if (t.op == V41ReferenceOp::Compress) {
                int j = int(item), produced = 0;
                for (int i = 0; i < t.rows; ++i) {
                    int slot = (t.start + i) % t.ratio;
                    t.p[TailKV][slot * t.cols + j] = x[i * t.cols + j];
                    t.p[TailScore][slot * t.cols + j] = t.p[Scores][i * t.cols + j];
                    if (slot + 1 == t.ratio) {
                        float maximum = -INFINITY, total = 0, v = 0;
                        for (int k = 0; k < t.ratio; ++k)
                            maximum = Max(maximum, t.p[TailScore][k * t.cols + j]);
                        for (int k = 0; k < t.ratio; ++k)
                            total += Exp(t.p[TailScore][k * t.cols + j] - maximum);
                        for (int k = 0; k < t.ratio; ++k)
                            v += t.p[TailKV][k * t.cols + j] *
                                 (Exp(t.p[TailScore][k * t.cols + j] - maximum) / total);
                        y[produced * t.cols + j] = BF(v);
                        ++produced;
                    }
                }
            } else if (t.op == V41ReferenceOp::IndexScores) {
                int token = int(item / t.visible), pos = int(item % t.visible);
                float score = 0;
                for (int head = 0; head < t.heads; ++head) {
                    const float *q = x + (size_t(token) * t.heads + head) * t.dim;
                    double dot = 0;
                    for (int j = 0; j < t.dim; ++j)
                        dot += double(q[j]) * t.p[Keys][size_t(pos) * t.dim + j];
                    float w = BF(t.p[HeadWeights][token * t.heads + head] * t.factor);
                    score += BF(Max(0.0f, BF(float(dot))) * w);
                }
                int length = t.start == 0 ? (token + 1) / t.ratio : t.visible;
                y[item] = pos < length ? BF(score) : -INFINITY;
            } else if (t.op == V41ReferenceOp::IndexSelect) {
                // One row per thread; top-k ordering is descending score, then
                // ascending position. Selected positions are returned sorted.
                int token = int(item), n = t.visible, length = t.start == 0 ? (token + 1) / t.ratio : n;
                float *mask = t.p[Output1] + size_t(token) * n;
                const float *scores = x + size_t(token) * n;
                int blocks = (n + t.candidateBlock - 1) / t.candidateBlock;
                for (int i = 0; i < n; ++i)
                    mask[i] = t.useCandidates ? t.p[Candidates][size_t(token) * n + i] : 1;
                if (t.buildCandidates && blocks > t.candidateBlocks) {
                    // Rank block maxima without materializing an unbounded local array.
                    for (int b = 0; b < blocks; ++b) {
                        float v = -INFINITY;
                        for (int j = b * t.candidateBlock; j < MinI(n, (b + 1) * t.candidateBlock); ++j)
                            v = Max(v, scores[j]);
                        if (length > 0 && b == (length - 1) / t.candidateBlock)
                            v = INFINITY;
                        int rank = 0;
                        for (int c = 0; c < blocks; ++c) {
                            float w = -INFINITY;
                            for (int j = c * t.candidateBlock; j < MinI(n, (c + 1) * t.candidateBlock); ++j)
                                w = Max(w, scores[j]);
                            if (length > 0 && c == (length - 1) / t.candidateBlock)
                                w = INFINITY;
                            if (w > v || (w == v && c < b))
                                ++rank;
                        }
                        for (int j = b * t.candidateBlock; j < MinI(n, (b + 1) * t.candidateBlock); ++j)
                            mask[j] = rank < t.candidateBlocks && v > -INFINITY;
                    }
                }
                auto dest = reinterpret_cast<int *>(y) + size_t(token) * t.topk;
                int count = 0;
                for (int pos = 0; pos < n; ++pos) {
                    float v = mask[pos] ? scores[pos] : -INFINITY;
                    int rank = 0;
                    if (t.topk < n)
                        for (int j = 0; j < n; ++j) {
                            float w = mask[j] ? scores[j] : -INFINITY;
                            if (w > v || (w == v && j < pos))
                                ++rank;
                        }
                    if (rank < t.topk)
                        dest[count++] = pos < length ? pos + t.offset : -1;
                }
            } else if (t.op == V41ReferenceOp::SparseAttention) {
                // Per-query/head online softmax; 64-slot tiles and BF16
                // probabilities are part of the native V4.1 numerical contract.
                int token = int(item / t.heads), head = int(item % t.heads), hd = t.dim;
                const float *q = x + (size_t(token) * t.heads + head) * hd;
                float *num = y + (size_t(token) * t.heads + head) * hd;
                for (int j = 0; j < hd; ++j)
                    num[j] = 0;
                float maximum = -1e30f, denominator = 0;
                const int *positions = reinterpret_cast<const int *>(t.p[Indices]) + size_t(token) * t.width;
                for (int first = 0; first < t.width; first += 64) {
                    float scores[64], raw[64], probs[64];
                    int indices[64];
                    float next = maximum;
                    for (int i = 0; i < 64; ++i) {
                        int pos = first + i < t.width ? positions[first + i] : -1;
                        if (pos < 0 || pos >= t.offset + t.visible)
                            pos = -1;
                        indices[i] = pos;
                        float dot = 0;
                        if (pos >= 0) {
                            const float *k = pos < t.offset ? t.p[Keys] + size_t(pos) * hd
                                                            : t.p[Values] + size_t(pos - t.offset) * hd;
                            for (int firstK = 0; firstK < hd; firstK += 192) {
                                float partial = 0;
                                for (int j = firstK; j < MinI(firstK + 192, hd); ++j)
                                    partial = fmaf(q[j], k[j], partial);
                                dot += partial;
                            }
                        }
                        scores[i] = pos >= 0 ? dot * t.factor : -INFINITY;
                        next = Max(next, scores[i]);
                    }
                    float correction = TensorExp(maximum - next);
                    for (int i = 0; i < 64; ++i) {
                        raw[i] = TensorExp(scores[i] - next);
                        probs[i] = BF(raw[i]);
                    }
                    denominator = denominator * correction + Sum(64, Load{raw});
                    maximum = next;
                    for (int j = 0; j < hd; ++j) {
                        float dot = 0;
                        for (int i = 0; i < 64; ++i)
                            if (indices[i] >= 0) {
                                int pos = indices[i];
                                float v = pos < t.offset ? t.p[Keys][size_t(pos) * hd + j]
                                                         : t.p[Values][size_t(pos - t.offset) * hd + j];
                                dot = fmaf(probs[i], v, dot);
                            }
                        num[j] = num[j] * correction + dot;
                    }
                }
                denominator += TensorExp(t.p[Sink][head] - maximum);
                for (int j = 0; j < hd; ++j)
                    num[j] = BF(num[j] / denominator);
            }
        }
    } // namespace v41ref
} // namespace fastllm
#undef V41_HD
#endif
