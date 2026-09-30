#include "fastllm.h"
#include "executor.h"
#include "devices/numas/numasdevice.h"
#include "utils.h"
#include "devices/cpu/computeutils.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "devices/cuda/naive-n05-cuda.cuh"
#endif
using namespace fastllm;
namespace fastllm { CPUInstructInfo *GetCPUInstructInfo(); }

static float BF(float x) { return RoundFloat32ToBFloat16RNE(x); }
static int Column(int expert, int part, int row, int width) {
    return (row * 17 + expert * 13 + part * 7) % width;
}
static uint8_t WeightBits(int row) {
    return row % 5 == 0 ? 7 : (row % 3 == 0 ? 0xb8 : 0x38);
}
static float WeightValue(int row) {
    return row % 5 == 0 ? 7.0f / 512 : (row % 3 == 0 ? -1.0f : 1.0f);
}
static float WeightScale(int expert, int part, int row, int col) {
    return std::ldexp(part ? .25f : .5f, (expert + row / 128 * 3 + col / 128) % 5 - 2);
}
static float Quantize(float x) {
    // Scalar finite E4M3 oracle, independent of the production encoder.
    float magnitude = std::abs(x), nearest = 0, distance = magnitude;
    int previous = 0;
    for (int bits = 1; bits <= 126; bits++) {
        int exp = bits >> 3, mantissa = bits & 7;
        float value = exp ? std::ldexp(1 + mantissa / 8.0f, exp - 7) : std::ldexp(mantissa / 8.0f, -6);
        float delta = std::abs(value - magnitude);
        if (delta < distance || (delta == distance && !(bits & 1) && (previous & 1))) {
            nearest = value; distance = delta; previous = bits;
        }
    }
    return std::copysign(nearest, x);
}
static void RoundActivation(std::vector<float> &values, std::vector<float> *scales = nullptr) {
    if (scales) scales->resize((values.size() + 127) / 128);
    for (size_t start = 0; start < values.size(); start += 128) {
        float maximum = 0;
        size_t end = std::min(values.size(), start + 128);
        for (size_t i = start; i < end; i++) maximum = std::max(maximum, std::abs(values[i]));
        float scale = maximum * (1.0f / 448);
        if (scales) (*scales)[start / 128] = scale;
        for (size_t i = start; i < end; i++) {
            float normalized = values[i] / std::max(scale, 1e-12f);
            int exponent;
            std::frexp(normalized, &exponent);
            float truncated = std::ldexp(std::floor(std::ldexp(std::abs(normalized), 11 - exponent)), exponent - 11);
            values[i] = Quantize(std::copysign(truncated, normalized));
            if (!scales) values[i] *= scale;
        }
    }
}
static void TestVectorQuantization() {
    if (!GetCPUInstructInfo()->hasAVX512BF16) return;
    // Exhaust every finite FP16 value in the activation range, including
    // signed zero, FP8 subnormals and halfway ties. Each block has amax=448,
    // so its packed scale is exactly one and the independent oracle applies.
    std::vector<float> values;
    for (int sign = 0; sign < 2; ++sign) for (int bits = 0; bits <= 0x5f00; ++bits) {
        int exp = bits >> 10, mantissa = bits & 1023;
        float value = exp ? std::ldexp(1.0f + mantissa / 1024.0f, exp - 15) : std::ldexp((float)mantissa, -24);
        values.push_back(sign ? -value : value);
        if (values.size() % 128 == 127) values.push_back(448.0f);
    }
    values.resize((values.size() + 127) / 128 * 128, 448.0f);
    std::vector<uint8_t> packed(values.size() / 128 * 132);
    if (!QuantizeEagerFP8_AVX512BF16(values.data(), packed.data(), values.size() / 128, 128))
        throw std::runtime_error("SIMD FP8 quantizer declined aligned input");
    for (size_t i = 0; i < values.size(); ++i) {
        int code = packed[i / 128 * 132 + i % 128];
        int exp = (code & 127) >> 3, mantissa = code & 7;
        float got = exp ? std::ldexp(1.0f + mantissa / 8.0f, exp - 7) : std::ldexp(mantissa / 8.0f, -6);
        if (code & 128) got = -got;
        float want = Quantize(values[i]);
        if (got != want || std::signbit(got) != std::signbit(want))
            throw std::runtime_error("SIMD FP8 halfway/subnormal/sign mismatch");
    }
    // Arbitrary per-block scales and partial final blocks, with the actual
    // intermediate FP16 RTZ conversion performed independently of the kernel.
    std::mt19937 rng(12345);
    FP8E4M3ToFP32Manager decode;
    for (int columns : {16, 96, 128, 160, 4096}) {
        std::vector<float> input(17 * columns);
        for (auto &v : input) v = std::ldexp((int)(rng() % 20001) - 10000.0f, -20 + (int)(rng() % 22));
        size_t stride = GetDataBytes(FP8_E4M3_BLOCK_128, 1, columns);
        packed.resize(17 * stride);
        QuantizeEagerFP8_AVX512BF16(input.data(), packed.data(), 17, columns);
        for (int row = 0; row < 17; ++row) {
            std::vector<float> expected(input.begin() + row * columns, input.begin() + (row + 1) * columns);
            RoundActivation(expected);
            for (int start = 0; start < columns; start += 128) {
                int count = std::min(128, columns - start);
                uint8_t *block = packed.data() + row * stride + start / 128 * 132;
                float scale; std::memcpy(&scale, block + count, 4);
                for (int i = 0; i < count; ++i) {
                    if (decode.dict[block[i]] * scale != expected[start + i])
                        throw std::runtime_error("SIMD FP8 scale/RTZ/tail mismatch");
                }
            }
        }
    }
}

static void TestTiledGemm() {
    if (!GetCPUInstructInfo()->hasAVX512BF16) return;
    std::mt19937 rng(9271);
    FP8E4M3ToFP32Manager fp8;
    // Exercise both dispatch paths and every 1..7-row packed-kernel tail.
    for (int m : {32, 96, 128, 160, 384, 4096}) for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 64}) {
        const int k = 43, st = 3, end = 37;
        size_t stride = GetDataBytes(FP8_E4M3_BLOCK_128, 1, m);
        std::vector<uint8_t> a(n * stride), b(k * stride);
        for (auto *data : {&a, &b}) {
            for (size_t row = 0; row < data->size() / stride; ++row) for (int start = 0; start < m; start += 128) {
                uint8_t *block = data->data() + row * stride + start / 128 * 132;
                int count = std::min(128, m - start);
                for (int i = 0; i < count; ++i) block[i] = (rng() % 127) | ((rng() & 1) << 7);
                float scale = std::ldexp(float(1 + rng() % 100), -14);
                std::memcpy(block + count, &scale, 4);
            }
        }
        std::vector<float> out(n * k, 12345.0f);
        FastllmGemmFP8Block128_AVX512BF16(a.data(), stride, b.data(), stride, out.data(), k * 4, n, m, st, end);
        for (int row = 0; row < n; ++row) for (int col = 0; col < k; ++col) {
            if (col < st || col >= end) {
                if (out[row * k + col] != 12345.0f) throw std::runtime_error("GEMM wrote outside output slice");
                continue;
            }
            double ref = 0;
            for (int start = 0; start < m; start += 128) {
                int count = std::min(128, m - start);
                auto *pa = a.data() + row * stride + start / 128 * 132;
                auto *pb = b.data() + col * stride + start / 128 * 132;
                float sa, sb; std::memcpy(&sa, pa + count, 4); std::memcpy(&sb, pb + count, 4);
                double dot = 0;
                for (int i = 0; i < count; ++i) dot += (double)fp8.dict[pa[i]] * fp8.dict[pb[i]];
                ref += (dot * sa) * sb;
            }
            if (std::abs(out[row * k + col] - ref) > 2e-5 * std::max(1.0, std::abs(ref)))
                throw std::runtime_error("Tiled FP8 GEMM differs from double precision oracle");
        }
    }
}

int main(int argc, char **argv) {
    try {
        const bool hybrid = argc == 2 && std::strcmp(argv[1], "--hybrid") == 0;
        if (hybrid) {
#ifdef USE_CUDA
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
#else
            return 77;
#endif
        }
        if (argc == 2 && std::strcmp(argv[1], "--avx2") == 0) {
            auto *info = GetCPUInstructInfo();
            if (!info->hasAVX2) return 77;
            info->hasAVX512F = info->hasAVX512BF16 = info->hasAVX512VNNI = false;
            info->hasAMX = false;
        }
        TestVectorQuantization();
        TestTiledGemm();
        constexpr int hidden = 256, inter = 128, experts = 3, topk = 2;
        auto &executor = *(Executor *)GetExecutor();
        executor.SetFirstDevice("numa");
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<Data *> weights(2 * (experts + 1), nullptr), biases(weights);
        for (int e = 1; e <= experts; e++) for (int part = 0; part < 2; part++) {
            int rows = part ? hidden : 2 * inter, cols = part ? inter : hidden;
            auto w = std::make_unique<Data>(FP8_E4M3, std::vector<int>{rows, cols});
            w->blockK = w->blockM = 128;
            w->scales.assign((rows / 128) * (cols / 128), part ? .25f : .5f);
            for (int r = 0; r < rows / 128; ++r) for (int c = 0; c < cols / 128; ++c)
                w->scales[r * (cols / 128) + c] = WeightScale(e, part, r * 128, c * 128);
            w->Allocate(); std::memset(w->cpuData, 0, w->GetBytes());
            for (int r = 0; r < rows; r++)
                w->cpuData[r * cols + Column(e, part, r, cols)] = WeightBits(r);
            weights[e * 2 + part] = w.get(); owned.push_back(std::move(w));
        }
        Data x(BFLOAT16), ids(INT32), scores(FLOAT32), output(BFLOAT16), w1, w2, w3, ti, to;
        const std::vector<int> rowCases = hybrid ? std::vector<int>{256, 257, 65, 2} :
                                                  std::vector<int>{1, 3, 65, 2, 255, 256, 257};
        for (int rows : rowCases) {
            x.Resize({rows, hidden}); x.Allocate();
            ids.Resize({rows, topk}); ids.Allocate();
            scores.Resize({rows, topk}); scores.Allocate();
            std::vector<std::vector<float>> inputs(rows, std::vector<float>(hidden));
            std::vector<std::vector<float>> inputScales(rows);
            for (int row = 0; row < rows; row++) {
                for (int c = 0; c < hidden; c++) {
                    float value = BF(std::sin((row * hidden + c) * .739f) *
                                     std::ldexp(2.1f, (row + c / 128) % 5 - 2));
                    inputs[row][c] = value;
                    ((uint16_t *)x.cpuData)[row * hidden + c] = Float32ToBFloat16RNEBits(value);
                }
                for (int j = 0; j < topk; j++) {
                    ((int *)ids.cpuData)[row * topk + j] = (row + j) % experts;
                    ((float *)scores.cpuData)[row * topk + j] = j ? .6837f : .3163f;
                }
                RoundActivation(inputs[row], &inputScales[row]);
            }
            executor.Run("MergeMOE", {
                {"input", &x}, {"index", &ids}, {"score", &scores},
                {"weights", (Data *)weights.data()}, {"biass", (Data *)biases.data()},
                {"w1", &w1}, {"w2", &w2}, {"w3", &w3}, {"curInput", &ti}, {"curOutput", &to}, {"output", &output}
            }, {{"sharedScale", 0}}, {{"weights___batch", (int)weights.size()},
                {"biass___batch", (int)biases.size()}, {"fp8EagerMode", 1}});
            output.ToDevice(DataDevice::CPU);
            std::vector<float> gpuRoutes;
#ifdef USE_CUDA
            if (hybrid && rows >= 256) {
                // Call CUDA directly as well: an accidental CPU fallback in
                // the mixed executor must not make this test pass silently.
                std::vector<float> silu(65536);
                for (int i = 0; i < 65536; ++i) {
                    float v = BFloat16BitsToFloat32(i);
                    silu[i] = v / (1 + std::exp(-v));
                }
                std::vector<FastllmNaiveFP8ExpertTask> tasks;
                for (int e = 1; e <= experts; ++e) {
                    FastllmNaiveFP8ExpertTask task{weights[e * 2]->numasData[0],
                                                  weights[e * 2 + 1]->numasData[0], {}};
                    for (int r = 0; r < rows * topk; ++r)
                        if (((int *)ids.cpuData)[r] + 1 == e) task.routes.push_back(r);
                    tasks.push_back(std::move(task));
                }
                gpuRoutes.resize((size_t)rows * topk * hidden);
                if (!FastllmCudaNaiveExpertPrefill(0, (uint16_t *)x.cpuData,
                    (float *)scores.cpuData, rows, topk, hidden, inter, tasks,
                    silu.data(), gpuRoutes.data()))
                    throw std::runtime_error("CUDA FP8 expert path unavailable");
            }
#endif
            for (int row = 0; row < rows; row++) {
                std::vector<float> expected(hidden, 0);
                for (int j = 0; j < topk; j++) {
                    int expert = ((int *)ids.cpuData)[row * topk + j] + 1;
                    std::vector<float> mid(inter);
                    for (int c = 0; c < inter; c++) {
                        int gc = Column(expert, 0, c, hidden), uc = Column(expert, 0, c + inter, hidden);
                        // Dot on unscaled FP8 values, then A scale and B scale.
                        // Scaling A before its dot can round differently.
                        float gate = BF((inputs[row][gc] * WeightValue(c)) * inputScales[row][gc / 128] * WeightScale(expert, 0, c, gc));
                        float up = BF((inputs[row][uc] * WeightValue(c + inter)) * inputScales[row][uc / 128] * WeightScale(expert, 0, c + inter, uc));
                        mid[c] = BF(BF(gate / (1 + std::exp(-gate))) * up);
                    }
                    std::vector<float> midScales;
                    RoundActivation(mid, &midScales);
                    float route = BF(((float *)scores.cpuData)[row * topk + j]);
                    for (int c = 0; c < hidden; c++) {
                        int dc = Column(expert, 1, c, inter);
                        float routed = BF(BF((mid[dc] * WeightValue(c)) * midScales[dc / 128] * WeightScale(expert, 1, c, dc)) * route);
                        expected[c] += routed;
                        if (!gpuRoutes.empty() && gpuRoutes[((size_t)row * topk + j) * hidden + c] != routed)
                            throw std::runtime_error("CUDA FP8 routed output differs from scalar oracle");
                    }
                }
                for (int c = 0; c < hidden; c++) {
                    float actual = BFloat16BitsToFloat32(((uint16_t *)output.cpuData)[row * hidden + c]);
                    if (!std::isfinite(actual) || std::abs(actual - BF(expected[c])) > 1e-6f) {
                        std::fprintf(stderr, "rows=%d row=%d col=%d actual=%g expected=%g\n", rows, row, c, actual, BF(expected[c]));
                        throw std::runtime_error("NUMA W8A8 MoE differs from scalar BF16/FP8 oracle");
                    }
                }
            }
        }
        ClearNumasMoeRuntimeCache();
        std::puts("NUMA FP8 eager MoE regression passed");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
    return 0;
}
