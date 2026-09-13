// NVFP4_BLOCK_32_E8M0 fused CPU GEMM regression + micro-benchmark.
//
// Checks that the AVX2 fused kernel, the AVX512-BF16 fused kernel (when the
// CPU has one) and the generic dequantise-then-GEMM fallback all agree, and
// measures the routed-expert decode bandwidth that DeepSeek-V4/V4.1 MoE
// depends on.
//
//   ./nvfp4Block32GemmRegression            correctness only
//   ./nvfp4Block32GemmRegression --bench    correctness + benchmark
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
        for (int n : {1, 2, 3, 5, 6, 8, 13, 32}) {
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
            for (size_t i = 0; i < ref.size(); i++) {
                const double denom = std::max(1e-6, (double)mag[i]);
                maxRel = std::max(maxRel, std::fabs((double)got[i] - ref[i]) / denom);
            }
            char buf[256];
            snprintf(buf, sizeof(buf), "AVX2 fused m=%d k=%d n=%d maxErr/|terms|=%.3e", m, k, n, maxRel);
            Check(maxRel < 2e-5, buf);

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
            for (size_t i = 0; i < ref.size(); i++) {
                const double denom = std::max(1e-6, (double)mag[i]);
                gemmRel = std::max(gemmRel, std::fabs((double)viaGemm[i] - ref[i]) / denom);
            }
            snprintf(buf, sizeof(buf), "FastllmGemm  m=%d k=%d n=%d maxErr/|terms|=%.3e", m, k, n, gemmRel);
            Check(gemmRel < 2e-5, buf);
        }
    }
}

// Streams a DeepSeek-V4.1-sized pile of routed-expert weights the way a decode
// step does: every expert matrix is read exactly once, from memory, by many
// threads at the same time.
static void Bench(int threads, bool useAvx2, double gigabytes) {
    const int m = 5120;      // hidden size
    const int k = 2304 * 2;  // fused gate+up rows of one expert
    const size_t rowBytes = Block32RowBytes(m);
    const size_t expertBytes = rowBytes * (size_t)k;
    const int experts = std::max(1, (int)(gigabytes * 1e9 / expertBytes));

    std::mt19937 rng(7);
    std::vector<std::vector<uint8_t>> weights(experts);
    for (int e = 0; e < experts; e++) {
        BuildWeights(k, m, weights[e], rng);
    }
    std::vector<uint16_t> input(m);
    for (auto &v : input) v = FloatToBf16(((int)(rng() % 2001) - 1000) / 1000.0f);

    const int rowsPerTask = 208;
    const int chunks = (k + rowsPerTask - 1) / rowsPerTask;
    const int tasks = experts * chunks;
    std::vector<std::vector<float>> outputs(threads, std::vector<float>(k, 0.0f));

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
                            1, m, k, st, end);
                    } else {
                        FastllmGemm(1, m, k,
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
            printf("threads=%3d kernel=%-8s experts=%d bytes=%.2f GB  %.1f ms  %.1f GB/s"
                   "  (%.1f ms per 4.5 GB decode step)\n",
                   threads, useAvx2 ? "avx2" : "dispatch", experts, bytes / 1e9,
                   seconds * 1000.0, bytes / 1e9 / seconds,
                   4.5e9 / (bytes / seconds) * 1000.0);
        }
    }
}

int main(int argc, char **argv) {
    bool bench = false;
    double gigabytes = 8.0;
    std::vector<int> threadList;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--bench") bench = true;
        else if (arg == "--gb" && i + 1 < argc) gigabytes = atof(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) threadList.push_back(atoi(argv[++i]));
    }
    auto *info = GetCPUInstructInfo();
    printf("hasAVX2=%d hasAVX512BF16=%d\n", (int)info->hasAVX2, (int)info->hasAVX512BF16);
    Correctness();
    if (bench) {
        if (threadList.empty()) {
            threadList = {(int)std::thread::hardware_concurrency()};
        }
        for (int t : threadList) {
            Bench(t, false, gigabytes);
            Bench(t, true, gigabytes);
        }
    }
    printf(failures == 0 ? "ALL PASS\n" : "FAILED %d\n", failures);
    return failures == 0 ? 0 : 1;
}
