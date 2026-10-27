#include "fastllm.h"
#include "devices/numas/numasdevice.h"
#include "devices/cpu/computeutils.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace fastllm { void RegisterNumas(Data *, std::string); }
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

static std::unique_ptr<Data> Weight(ggml_type type, int rows, int columns, unsigned seed) {
    auto w = std::make_unique<Data>(DATA_GGUF_FORMAT);
    w->isGGUFData = true; w->ggmlType = type; w->Resize({rows, columns}); w->Allocate(false);
    for (size_t i = 0; i < w->GetBytes(); ++i) {
        seed = seed * 1664525U + 1013904223U;
        w->cpuData[i] = seed >> 24;
    }
    const size_t block = ggml_type_size(type);
    for (size_t i = 0; i < w->GetBytes(); i += block) {
        const uint16_t scale = type == GGML_TYPE_IQ4_XS ? 0x0800 : 0x2000;
        std::memcpy(w->cpuData + i, &scale, sizeof(scale));
    }
    return w;
}

// Whole-shard GEMM and activation/quantization provide a serial reference
// independent of the decode queue's slices, route mapping and synchronization.
static void Reference(const float *input, float *output, Data &gate, Data &down) {
    const int hidden = gate.dims[1], inter = down.dims[1], nodes = gate.numasData.size();
    const auto gateType = gate.GetLinearActDataType(1), downType = down.GetLinearActDataType(1);
    std::vector<uint8_t> x(GetDataBytes(gateType, 1, hidden)), y(GetDataBytes(downType, 1, inter));
    std::vector<float> gu(2 * inter), mid(inter);
    ConvertFromFloat32(x.data(), gateType, input, 1, hidden);
    for (int node = 0; node < nodes; ++node)
        MultiThreadGemmOp(x.data(), gateType, gate.numasData[node], gate.GetDataType(),
            reinterpret_cast<uint8_t *>(gu.data() + node * 2 * inter / nodes), FLOAT32,
            1, hidden, 2 * inter, 0, 2 * inter / nodes).Run();
    MultiThreadCrossSwigluOp(gu.data(), inter, inter, mid.data(), 1, 2 * inter, inter).Run();
    ConvertFromFloat32(y.data(), downType, mid.data(), 1, inter);
    for (int node = 0; node < nodes; ++node)
        MultiThreadGemmOp(y.data(), downType, down.numasData[node], down.GetDataType(),
            reinterpret_cast<uint8_t *>(output + node * hidden / nodes), FLOAT32,
            1, inter, hidden, 0, hidden / nodes).Run();
}

static void TestScoredSmallBlocks() {
    // Q4_0 has 32-value blocks: legal intermediate widths need not meet
    // NVFP4's 128-value per-shard activation constraint.
    constexpr int hidden = 256, experts = 3, topk = 4;
    for (int inter : {64, 192}) {
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<Data *> weights(2 * (experts + 1), nullptr);
        for (int e = 1; e <= experts; ++e) for (int part = 0; part < 2; ++part) {
            auto w = Weight(GGML_TYPE_Q4_0, part ? hidden : 2 * inter,
                            part ? inter : hidden, 31 * e + part);
            RegisterNumas(w.get(), part ? "linearColumn" : "linearSwiglu");
            weights[2 * e + part] = w.get(); owned.push_back(std::move(w));
        }
        Check(CanRunNumasMoeDecodeExperts(weights.data(), weights.size()), "small-block fixture rejected");
        std::vector<float> x(hidden), expected(topk * hidden), actual(topk * hidden);
        std::vector<uint16_t> bits(hidden);
        for (int c = 0; c < hidden; ++c) {
            bits[c] = Float32ToBFloat16RNEBits(std::sin(c * .17f) * .125f);
            x[c] = BFloat16BitsToFloat32(bits[c]);
        }
        const int32_t ids[topk] = {2, 0, 2, 1};
        const float scores[topk] = {.375f, 0, -.125f, .25f};
        for (int cpuCount = 1; cpuCount <= experts; ++cpuCount) {
            int32_t gpu[topk];
            for (int r = 0; r < topk; ++r) gpu[r] = ids[r] >= experts - cpuCount ? -1 : 0;
            std::fill(expected.begin(), expected.end(), 123456.f);
            std::fill(actual.begin(), actual.end(), 123456.f);
            NumasMoeVerifyExperts(bits.data(), expected.data(), 1, weights.data(), weights.size(),
                ids, gpu, scores, topk, 0, .125f, true, 128);
            int submitted = 0;
            double cpuUs = 0;
            NumasMoeDecodeExpertsWithOverlap(x.data(), actual.data(), weights.data(), ids, gpu,
                topk, 0, [&] { ++submitted; }, scores, .125f, 128, &cpuUs);
            Check(submitted == 1 && cpuUs > 0, "small-block callback or timing missing");
            for (size_t i = 0; i < actual.size(); ++i)
                Check(actual[i] == expected[i], "scored small-block decode differs from verifier");
        }
        ClearNumasMoeRuntimeCache();
    }
}

static void BatchCases(std::vector<Data *> &weights, int hidden, int experts) {
    constexpr int topk = 5;
    for (int rows : {1, 2, 3, 4, 8, 9}) for (int pattern = 0; pattern < 4; ++pattern) {
        std::vector<float> x(rows * hidden), scores(rows * topk, .125f);
        std::vector<int32_t> ids(rows * topk), gpu(rows * topk);
        std::vector<float> expected(rows * topk * hidden, 123456.f), actual(expected);
        for (int i = 0; i < rows * hidden; ++i) x[i] = std::sin(float(i * 7 + pattern)) * .03125f;
        for (int r = 0; r < rows * topk; ++r) {
            // Shared experts, duplicates within a row, and ownership that
            // differs between rows must all preserve the original route slots.
            ids[r] = pattern == 3 ? 0 : (r % topk + (r / topk) / 2) % experts;
            gpu[r] = pattern == 0 ? 0 : pattern == 1 ? -1 : (r % 3 == 0 ? 0 : -1);
            if (gpu[r] < 0) Reference(x.data() + (r / topk) * hidden,
                expected.data() + r * hidden, *weights[2 * (ids[r] + 1)], *weights[3 + 2 * ids[r]]);
        }
        int submitted = 0;
        auto run = [&] {
            NumasMoeDecodeExpertsBatchWithOverlap(x.data(), actual.data(), rows,
                weights.data(), weights.size(), ids.data(), gpu.data(), scores.data(), topk,
                pattern % 2, [&] { ++submitted; });
        };
        run(); Check(submitted == 1, "batch callback count changed");
        for (size_t i = 0; i < actual.size(); ++i) {
            if (gpu[i / hidden] >= 0) Check(actual[i] == 123456.f, "batch wrote a GPU-owned route");
            else if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) >=
                         3e-5f * (1 + std::abs(expected[i]))) {
                std::fprintf(stderr, "batch mismatch rows=%d pattern=%d at=%zu actual=%g expected=%g\n",
                    rows, pattern, i, actual[i], expected[i]);
                Check(false, "batch differs from serial reference");
            }
        }
        if (rows == 4 && pattern == 2) {
            bool caught = false;
            try {
                NumasMoeDecodeExpertsBatchWithOverlap(x.data(), actual.data(), rows,
                    weights.data(), weights.size(), ids.data(), gpu.data(), scores.data(), topk,
                    pattern % 2, [] { throw std::runtime_error("expected batch callback failure"); });
            } catch (const std::runtime_error &) { caught = true; }
            Check(caught, "batch callback failure swallowed");
            std::fill(actual.begin(), actual.end(), 123456.f);
            run();
            for (size_t i = 0; i < actual.size(); ++i)
                Check(std::abs(actual[i] - expected[i]) < 3e-5f * (1 + std::abs(expected[i])),
                      "batch callback recovery changed output");
        }
    }
}

int main(int argc, char **argv) {
    try {
        SetThreads(argc > 1 ? std::atoi(argv[1]) : 4);
#ifdef __linux__
        cpu_set_t allowed;
        Check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "cannot read test CPU mask");
#endif
        TestScoredSmallBlocks();
        constexpr int experts = 8, topk = 9;
        for (auto gateType : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                             GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
                             GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL})
        for (auto downType : {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL})
        for (const auto dims : {std::pair<int, int>{256, 256}, {768, 384}, {2560, 640}}) {
            const int hidden = dims.first, inter = dims.second;
            std::vector<std::unique_ptr<Data>> owned;
            std::vector<Data *> weights(2 * (experts + 1), nullptr);
            for (int e = 1; e <= experts; ++e) for (int part = 0; part < 2; ++part) {
                auto w = Weight(part ? downType : gateType, part ? hidden : 2 * inter,
                                part ? inter : hidden, e * 13 + part);
                RegisterNumas(w.get(), part ? "linearColumn" : "linearSwiglu");
                weights[2 * e + part] = w.get(); owned.push_back(std::move(w));
            }
#ifdef __linux__
            // NUMA initialization may choose machine-wide CPU ids. Keep this
            // test inside the caller's mask even on a shared test host.
            for (auto *thread : GetAlivePool()->threads)
                Check(pthread_setaffinity_np(thread->native_handle(), sizeof(allowed), &allowed) == 0,
                      "cannot restrict test worker affinity");
#endif
            Check(CanRunNumasMoeDecodeExperts(weights.data(), weights.size()), "decode fixture rejected");
            std::vector<float> x(hidden), expected(topk * hidden), actual(topk * hidden);
            int32_t ids[topk], gpu[topk];
            for (int pass = 0; pass < 12; ++pass) {
                for (int c = 0; c < hidden; ++c) x[c] = std::sin(float(c * 17 + pass)) * .03125f;
                const int count = pass <= topk ? pass : 1;
                for (int r = 0; r < topk; ++r) {
                    ids[r] = (r * 3 + pass) % experts;
                    gpu[r] = (r + pass) % topk < count ? -1 : 0;
                }
                std::fill(actual.begin(), actual.end(), 123456.f);
                std::fill(expected.begin(), expected.end(), 123456.f);
                for (int r = 0; r < topk; ++r)
                    if (gpu[r] < 0) Reference(x.data(), expected.data() + r * hidden,
                        *weights[2 * (ids[r] + 1)], *weights[2 * (ids[r] + 1) + 1]);
                int submitted = 0;
                auto run = [&] {
                    NumasMoeDecodeExpertsWithOverlap(x.data(), actual.data(), weights.data(),
                        ids, gpu, topk, pass % 2, [&] { ++submitted; });
                };
                run(); Check(submitted == 1, "GPU submission callback count changed");
                auto check = [&] {
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (gpu[i / hidden] >= 0) Check(actual[i] == 123456.f, "wrote a GPU-owned route");
                        else Check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <
                              3e-5f * (1 + std::abs(expected[i])), "decode differs from serial reference");
                    }
                };
                check();
                if (pass == 3 && gateType == GGML_TYPE_IQ2_XXS && downType == GGML_TYPE_Q2_0) {
                    for (int rows : {2,3,7,9}) {
                        std::vector<float> bx(rows*hidden), by(rows*topk*hidden,123456.f), scores(rows*topk,1.f);
                        std::vector<int32_t> bi(rows*topk), bg(rows*topk);
                        for (int row=0;row<rows;++row) {
                            std::copy(x.begin(),x.end(),bx.begin()+row*hidden);
                            std::copy_n(ids,topk,bi.begin()+row*topk);
                            std::copy_n(gpu,topk,bg.begin()+row*topk);
                        }
                        submitted=0;double elapsed=0;
                        NumasMoeDecodeExpertsBatchWithOverlap(bx.data(),by.data(),rows,weights.data(),weights.size(),
                            bi.data(),bg.data(),scores.data(),topk,0,[&]{++submitted;},&elapsed);
                        Check(submitted==1 && elapsed>0,"batch CPU submission or elapsed missing");
                        for (int row=0;row<rows;++row) for (size_t i=0;i<actual.size();++i)
                            Check(by[row*topk*hidden+i]==actual[i],"unscored batch changed decode arithmetic");
                    }
                }
                if (pass == 1) {
                    bool caught = false;
                    try {
                        NumasMoeDecodeExpertsWithOverlap(x.data(), actual.data(), weights.data(), ids, gpu,
                            topk, pass % 2, [] { throw std::runtime_error("expected callback failure"); });
                    } catch (const std::runtime_error &) { caught = true; }
                    Check(caught, "callback failure swallowed");
                    run(); check();
                }
            }
            BatchCases(weights, hidden, experts);
            ClearNumasMoeRuntimeCache();
            std::printf("PASS gate=%d down=%d hidden=%d inter=%d\n", gateType, downType, hidden, inter);
        }
        std::puts("PASS: GGUF CPU decode slices, route masks, repeated experts and callback recovery");
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    return 0;
}
