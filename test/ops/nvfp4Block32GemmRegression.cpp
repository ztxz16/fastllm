// NVFP4_BLOCK_32_E8M0 fused CPU GEMM regression + micro-benchmark.
//
// Checks that the AVX2 fused kernel, the AVX512-BF16 fused kernel (when the
// CPU has one) and the generic dequantise-then-GEMM fallback all agree, and
// measures the routed-expert decode bandwidth that DeepSeek-V4/V4.1 MoE
// depends on.
//
//   ./nvfp4Block32GemmRegression            correctness only
//   ./nvfp4Block32GemmRegression --bench    correctness + benchmark
//   ./nvfp4Block32GemmRegression --avx2 --bench --rows 4 --threads 40
//   ./nvfp4Block32GemmRegression --avx2 --bench --input-dim 2304 --output-dim 5120
#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace fastllm { CPUInstructInfo *GetCPUInstructInfo(); }

using namespace fastllm;

static int failures = 0;

static void Check(bool ok, const std::string &what) {
    if (!ok) {
        printf("[FAIL] %s\n", what.c_str());
        failures++;
    } else {
        printf("[ OK ] %s\n", what.c_str());
    }
}

static const float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static float E8M0ToFloat(uint8_t v) {
    uint32_t bits = v == 0 ? 0x00400000u : ((uint32_t)v << 23);
    float ret;
    memcpy(&ret, &bits, sizeof(ret));
    return ret;
}

static float Bf16ToFloat(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float ret;
    memcpy(&ret, &bits, sizeof(ret));
    return ret;
}

static uint16_t FloatToBf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

// Packs k rows of m NVFP4 values into the fastllm block-32 layout:
// 16 packed nibble bytes + 1 UE8M0 scale byte for every 32 values.
static size_t Block32RowBytes(int m) {
    return (size_t)((m + 31) / 32) * 17;
}

static void BuildWeights(int k, int m, std::vector<uint8_t> &out, std::mt19937 &rng) {
    const size_t rowBytes = Block32RowBytes(m);
    out.assign(rowBytes * (size_t)k, 0);
    for (int row = 0; row < k; row++) {
        uint8_t *p = out.data() + (size_t)row * rowBytes;
        for (size_t block = 0; block < rowBytes / 17; block++) {
            for (int b = 0; b < 16; b++) {
                p[block * 17 + b] = (uint8_t)(rng() & 0xFF);
            }
            // Keep the exponents around 2^0 so the reference and the kernels
            // stay in the same magnitude range.
            p[block * 17 + 16] = (uint8_t)(120 + (rng() % 15));
        }
    }
}

// Also returns, per output element, the sum of the magnitudes of the terms.
// A dot product of 5120 random signed terms cancels almost completely, so the
// only meaningful accuracy bound for an FP32 accumulator is relative to that
// magnitude, not to the (near-zero) result.
static void Reference(const uint16_t *A, long ldaElems, const uint8_t *B, size_t rowBytes,
                      float *C, float *Scale, int ldcElems, int n, int m, int st, int end) {
    const int blocks = (m + 31) / 32;
    for (int i = 0; i < n; i++) {
        for (int j = st; j < end; j++) {
            const uint8_t *rowStart = B + (size_t)j * rowBytes;
            double total = 0.0;
            double magnitude = 0.0;
            for (int block = 0; block < blocks; block++) {
                const uint8_t *bs = rowStart + (size_t)block * 17;
                const float scale = E8M0ToFloat(bs[16]);
                const int base = block * 32;
                const int elems = std::min(32, m - base);
                double now = 0.0;
                for (int o = 0; o < elems; o++) {
                    const uint8_t byte = bs[o >> 1];
                    const uint8_t code = (o & 1) ? (byte >> 4) : (byte & 0xF);
                    const double term =
                        (double)Bf16ToFloat(A[(size_t)i * ldaElems + base + o]) *
                        (double)kE2M1[code];
                    now += term;
                    magnitude += std::fabs(term) * (double)scale;
                }
                total += now * (double)scale;
            }
            C[(size_t)i * ldcElems + j] = (float)total;
            Scale[(size_t)i * ldcElems + j] = (float)magnitude;
        }
    }
}

static void Correctness() {
    std::mt19937 rng(20260911);
    const int shapes[][2] = {{5120, 128}, {2304, 96}, {4608, 64}, {160, 33}, {96, 7}};
    for (auto &shape : shapes) {
        const int m = shape[0], k = shape[1];
        std::vector<uint8_t> weights;
        BuildWeights(k, m, weights, rng);
        const size_t rowBytes = Block32RowBytes(m);
        for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 13, 32}) {
            std::vector<uint16_t> input((size_t)n * m);
            for (auto &v : input) {
                v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);
            }
            std::vector<float> ref((size_t)n * k, 0.0f), got((size_t)n * k, 0.0f);
            std::vector<float> mag((size_t)n * k, 0.0f);
            Reference(input.data(), m, weights.data(), rowBytes, ref.data(), mag.data(), k, n, m, 0, k);
            bool ran = FastllmGemmBFloat16NVFP4Block32E8M0_AVX2(
                input.data(), (long)m * sizeof(uint16_t),
                weights.data(), (long)rowBytes,
                got.data(), (long)k * sizeof(float),
                n, m, k, 0, k);
            if (!ran) {
                printf("[SKIP] AVX2 kernel unavailable in this build\n");
                return;
            }
            double maxRel = 0.0;
            bool finite = true;
            for (size_t i = 0; i < ref.size(); i++) {
                finite &= std::isfinite(got[i]);
                const double denom = std::max(1e-6, (double)mag[i]);
                maxRel = std::max(maxRel, std::fabs((double)got[i] - ref[i]) / denom);
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "AVX2 fused m=%d k=%d n=%d maxErr/|terms|=%.3e", m, k, n, maxRel);
            Check(finite && maxRel < 2e-5, buf);

            // Same shape through the public dispatcher, which also exercises
            // the generic fallback when the AVX2 path is disabled.
            std::vector<float> viaGemm((size_t)n * k, 0.0f);
            FastllmGemm(n, m, k,
                        input.data(), (long)m * sizeof(uint16_t),
                        weights.data(), (long)rowBytes,
                        viaGemm.data(), (long)k * sizeof(float),
                        0, k,
                        DataType::BFLOAT16, DataType::NVFP4_BLOCK_32_E8M0, DataType::FLOAT32);
            double gemmRel = 0.0;
            bool gemmFinite = true;
            for (size_t i = 0; i < ref.size(); i++) {
                gemmFinite &= std::isfinite(viaGemm[i]);
                const double denom = std::max(1e-6, (double)mag[i]);
                gemmRel = std::max(gemmRel, std::fabs((double)viaGemm[i] - ref[i]) / denom);
            }
            snprintf(buf, sizeof(buf), "FastllmGemm  m=%d k=%d n=%d maxErr/|terms|=%.3e", m, k, n, gemmRel);
            Check(gemmFinite && gemmRel < 2e-5, buf);
        }
    }
}

// Splitting the output range must preserve results for every scale,
// including encodings that produce subnormals, overflow, or NaN.
static void ScaleCorrectness() {
    constexpr int m = 1024, k = 16;
    const size_t rowBytes = Block32RowBytes(m);
    std::mt19937 rng(20260920);
    std::vector<uint8_t> weights;
    BuildWeights(k, m, weights, rng);
    std::vector<uint16_t> input(8 * m);
    for (auto &v : input) {
        v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);
    }
#ifdef __SSE__
    const unsigned original = _mm_getcsr();
    constexpr int modes = 16;
#else
    constexpr int modes = 1;
#endif
    for (int mode = 0; mode < modes; mode++) {
#ifdef __SSE__
        // Four rounding modes, each with all FTZ/DAZ combinations.
        _mm_setcsr((original & ~0xe040u) | ((mode / 4) << 13) |
                   ((mode & 1) ? 0x8000u : 0) | ((mode & 2) ? 0x40u : 0));
#endif
        bool valid = true;
        for (int scale = 0; scale < 256; scale++) {
            for (size_t offset = 16; offset < weights.size(); offset += 17) {
                weights[offset] = (uint8_t)scale;
            }
            for (int n : {1, 4, 8}) {
                std::vector<float> whole(n * k), split(n * k);
                auto run = [&](std::vector<float> &out, int st, int end) {
                    return FastllmGemmBFloat16NVFP4Block32E8M0_AVX2(
                        input.data(), m * 2, weights.data(), (long)rowBytes,
                        out.data(), k * 4, n, m, k, st, end);
                };
                if (!run(whole, 0, k) || !run(split, 0, k / 2) ||
                    !run(split, k / 2, k)) {
#ifdef __SSE__
                    _mm_setcsr(original);
#endif
                    printf("[SKIP] AVX2 scale checks unavailable in this build\n");
                    return;
                }
                for (size_t i = 0; i < whole.size(); i++) {
                    valid &= memcmp(&whole[i], &split[i], sizeof(float)) == 0 ||
                        (std::isnan(whole[i]) && std::isnan(split[i]));
                }
            }
        }
        char description[128];
        snprintf(description, sizeof(description),
                 "AVX2 whole/split range all 256 scales, FP mode=%d", mode);
        Check(valid, description);
    }
#ifdef __SSE__
    _mm_setcsr(original);
#endif
}

// NUMA workers write a slice of the expert's full output matrix. Exercise
// nonzero/odd column ranges, byte strides, and full/partial blocks;
// untouched columns/padding are guards.
static void StridedCorrectness() {
    std::mt19937 rng(20260919);
    const int shapes[][4] = {
        {5120, 211, 3, 210}, {2304, 133, 1, 131},
        {5120, 35, 7, 23}, {5120, 35, 7, 22},
        {1056, 21, 1, 20}, {1024, 21, 1, 20},
        {1023, 21, 1, 20}, {5121, 21, 1, 20},
        {2305, 21, 1, 20}, {31, 21, 1, 20}, {1, 21, 1, 20}
    };
    for (const auto &shape : shapes) {
        const int m = shape[0], k = shape[1], st = shape[2], end = shape[3];
        const int lda = m + 11, ldc = k + 7;
        const size_t rowBytes = Block32RowBytes(m), ldb = rowBytes + 13;
        std::vector<uint8_t> denseWeights;
        BuildWeights(k, m, denseWeights, rng);
        std::vector<uint8_t> weights((size_t)k * ldb, 0xcd);
        for (int col = 0; col < k; col++) {
            memcpy(weights.data() + col * ldb, denseWeights.data() + col * rowBytes, rowBytes);
            // Also exercise E8M0's special subnormal scale encoding.
            if (col % 5 == 0) weights[col * ldb + 16] = 0;
        }
        for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 13, 32}) {
            std::vector<uint16_t> input((size_t)n * lda, 0x7fc1);
            for (int row = 0; row < n; row++) {
                for (int d = 0; d < m; d++) {
                    input[(size_t)row * lda + d] = FloatToBf16(
                        ((int)(rng() % 2001) - 1000) / 1000.0f);
                }
            }
            constexpr float guard = -1234567.0f;
            std::vector<float> got((size_t)n * ldc, guard);
            std::vector<float> ref(got), mag(got.size(), 0.0f);
            Reference(input.data(), lda, weights.data(), ldb,
                      ref.data(), mag.data(), ldc, n, m, st, end);
            if (!FastllmGemmBFloat16NVFP4Block32E8M0_AVX2(
                    input.data(), (long)lda * 2, weights.data(), (long)ldb,
                    got.data(), (long)ldc * 4, n, m, k, st, end)) return;
            bool valid = true;
            for (int row = 0; row < n; row++) {
                for (int col = 0; col < ldc; col++) {
                    const size_t i = (size_t)row * ldc + col;
                    if (col < st || col >= end) valid &= got[i] == guard;
                    else valid &= std::isfinite(got[i]) &&
                        std::fabs((double)got[i] - ref[i]) <= 2e-5 * std::max(1e-6, (double)mag[i]);
                }
            }
            char description[160];
            snprintf(description, sizeof(description),
                     "AVX2 strided m=%d n=%d columns=[%d,%d) padding preserved", m, n, st, end);
            Check(valid, description);
        }
    }
}

// Streams a DeepSeek-V4.1-sized pile of routed-expert weights through many
// threads. Bandwidth counts nominal weight bytes; multi-row tiles may read
// the same weights again when a batch is split into smaller row groups.
static void Bench(int threads, bool useAvx2, double gigabytes, int n, int m, int k) {
    const size_t rowBytes = Block32RowBytes(m);
    const size_t expertBytes = rowBytes * (size_t)k;
    const int experts = std::max(1, (int)(gigabytes * 1e9 / expertBytes));

    std::mt19937 rng(7);
    std::vector<std::vector<uint8_t>> weights(experts);
    for (int e = 0; e < experts; e++) {
        BuildWeights(k, m, weights[e], rng);
    }
    std::vector<uint16_t> input((size_t)n * m);
    for (auto &v : input) v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);

    const int rowsPerTask = 208;
    const int chunks = (k + rowsPerTask - 1) / rowsPerTask;
    const int tasks = experts * chunks;
    std::vector<std::vector<float>> outputs(threads, std::vector<float>((size_t)n * k, 0.0f));

    for (int warm = 0; warm < 2; warm++) {
        std::atomic<int> next(0);
        auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) {
            pool.emplace_back([&, t]() {
                while (true) {
                    int task = next.fetch_add(1, std::memory_order_relaxed);
                    if (task >= tasks) break;
                    const int e = task / chunks;
                    const int chunk = task - e * chunks;
                    const int st = chunk * rowsPerTask;
                    const int end = std::min(st + rowsPerTask, k);
                    if (useAvx2) {
                        FastllmGemmBFloat16NVFP4Block32E8M0_AVX2(
                            input.data(), (long)m * sizeof(uint16_t),
                            weights[e].data(), (long)rowBytes,
                            outputs[t].data(), (long)k * sizeof(float),
                            n, m, k, st, end);
                    } else {
                        FastllmGemm(n, m, k,
                                    input.data(), (long)m * sizeof(uint16_t),
                                    weights[e].data(), (long)rowBytes,
                                    outputs[t].data(), (long)k * sizeof(float),
                                    st, end,
                                    DataType::BFLOAT16, DataType::NVFP4_BLOCK_32_E8M0,
                                    DataType::FLOAT32);
                    }
                }
            });
        }
        for (auto &th : pool) th.join();
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double bytes = (double)expertBytes * experts;
        if (warm == 1) {
            printf("threads=%3d kernel=%-8s n=%d m=%d k=%d experts=%d bytes=%.2f GB  %.1f ms  %.1f GB/s"
                   "  (%.1f ms per 4.5 GB of weights)\n",
                   threads, useAvx2 ? "avx2" : "dispatch", n, m, k, experts, bytes / 1e9,
                   seconds * 1000.0, bytes / 1e9 / seconds,
                   4.5e9 / (bytes / seconds) * 1000.0);
        }
    }
}

int main(int argc, char **argv) {
    bool bench = false;
    bool forceAvx2 = false;
    double gigabytes = 8.0;
    int rows = 1, inputDim = 5120, outputDim = 2304 * 2;
    std::vector<int> threadList;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bench") bench = true;
        else if (arg == "--avx2") forceAvx2 = true;
        else if (arg == "--gb" && i + 1 < argc) gigabytes = atof(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threadList.push_back(atoi(argv[++i]));
        else if (arg == "--rows" && i + 1 < argc) rows = atoi(argv[++i]);
        else if (arg == "--input-dim" && i + 1 < argc) inputDim = atoi(argv[++i]);
        else if (arg == "--output-dim" && i + 1 < argc) outputDim = atoi(argv[++i]);
    }
    if (rows <= 0 || inputDim <= 0 || outputDim <= 0 || gigabytes <= 0 ||
        std::any_of(threadList.begin(), threadList.end(), [](int n) { return n <= 0; })) {
        fprintf(stderr, "Benchmark dimensions, threads and size must be positive.\n");
        return 1;
    }
    auto *info = GetCPUInstructInfo();
    if (forceAvx2) {
        info->hasAVX512F = false;
        info->hasAVX512BF16 = false;
        info->hasAVX512VNNI = false;
        info->hasAMX = false;
    }
    printf("hasAVX2=%d hasAVX512BF16=%d\n", (int)info->hasAVX2, (int)info->hasAVX512BF16);
    Correctness();
    ScaleCorrectness();
    StridedCorrectness();
    if (bench) {
        if (threadList.empty()) {
            threadList = {(int)std::thread::hardware_concurrency()};
        }
        for (int t : threadList) {
            Bench(t, false, gigabytes, rows, inputDim, outputDim);
            Bench(t, true, gigabytes, rows, inputDim, outputDim);
        }
    }
    printf(failures == 0 ? "ALL PASS\n" : "FAILED %d\n", failures);
    return failures == 0 ? 0 : 1;
}
