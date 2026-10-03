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

int main(int argc, char **argv) {
    try {
        SetThreads(argc > 1 ? std::atoi(argv[1]) : 4);
#ifdef __linux__
        cpu_set_t allowed;
        Check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "cannot read test CPU mask");
#endif
        constexpr int experts = 8, topk = 9;
        for (auto gateType : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                             GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS})
        for (auto downType : {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL})
        for (int inter : {256, 640}) {
            const int hidden = inter == 640 ? 2560 : 256;
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
            ClearNumasMoeRuntimeCache();
            std::printf("PASS gate=%d down=%d hidden=%d inter=%d\n", gateType, downType, hidden, inter);
        }
        std::puts("PASS: GGUF CPU decode slices, route masks, repeated experts and callback recovery");
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    return 0;
}
