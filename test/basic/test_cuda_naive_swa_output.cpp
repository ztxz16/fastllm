#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include "naive_n05_mxfp8_test.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace fastllm;
void TestAttentionOutputMailboxWrap(bool mma = false);
namespace {
void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
template<class F> void Parallel(F fn) {
    std::array<std::exception_ptr, 8> errors{};
    std::vector<std::thread> workers;
    for (int r = 0; r < 8; ++r) workers.emplace_back([&, r] {
        try { FastllmCudaSetDevice(r); fn(r); } catch (...) { errors[r] = std::current_exception(); }
    });
    for (auto &worker : workers) worker.join();
    for (auto &error : errors) if (error) std::rethrow_exception(error);
}
void Init(Data &x, const std::vector<int> &shape, int device) {
    x.Resize(shape); x.ToDevice(DataDevice::CUDA, {device}, false); x.Allocate(false);
}
void Fill(Data &x, unsigned seed, float scale = 1.f) {
    std::vector<uint8_t> bytes(x.GetBytes());
    for (size_t i = 0; i < x.Count(0); ++i) {
        seed = seed * 1664525u + 1013904223u;
        float value = (int(seed >> 16) - 32768) / 32768.f * scale;
        if (x.dataType == BFLOAT16) {
            uint16_t bits = Float32ToBFloat16RNEBits(value);
            std::memcpy(bytes.data() + i * 2, &bits, 2);
        } else std::memcpy(bytes.data() + i * 4, &value, 4);
    }
    Cuda(cudaMemcpy(x.cudaData, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
}
std::vector<uint8_t> Read(const Data &x) {
    std::vector<uint8_t> bytes(x.GetBytes());
    Cuda(cudaMemcpy(bytes.data(), x.cudaData, bytes.size(), cudaMemcpyDeviceToHost)); return bytes;
}
struct Tensors {
    Data x{BFLOAT16}, w{BFLOAT16}, p{BFLOAT16}, h{BFLOAT16}, n{FLOAT32}, y{BFLOAT16};
    Data referenceP{BFLOAT16}, referenceH{BFLOAT16}, referenceY{BFLOAT16};
    Data routerWeight{FLOAT32}, router{FLOAT32}, referenceRouter{FLOAT32};
    bool withRouter = false;
    bool ordered = false, allowMma = false;
    int rows = 1;
    cudaGraphExec_t graph = nullptr;
    double maxL2 = 0;
    explicit Tensors(int rank, int count, bool exact = false, bool mx = false) : w(mx ? FP8_E4M3 : BFLOAT16), ordered(exact), rows(count) {
        // Two logical requests also use the same flattened-row contract.
        auto shape = [&](int width) { return rows == 8 ? std::vector<int>{2, 4, width} : std::vector<int>{1, rows, width}; };
        Init(x, shape(1024), rank); Init(w, {4096, 1024}, rank); Init(n, {4096}, rank);
        for (auto *a : {&p, &h, &y, &referenceP, &referenceH, &referenceY}) Init(*a, shape(4096), rank);
        Init(routerWeight, {256, 4096}, rank); Init(router, shape(256), rank); Init(referenceRouter, shape(256), rank);
        Fill(routerWeight, 89, .01f);
        if (mx) FillVerifyMxWeight(w, 19 + rank);
        else Fill(w, 19 + rank, .02f);
        Fill(n, 731, 1.f);
    }
    void Reset(int seed, int rank) {
        Fill(x, seed * 17 + rank, seed % 3 == 0 ? 10.f : 1.f);
        Fill(h, seed + 801);
        Cuda(cudaMemcpy(referenceH.cudaData, h.cudaData, h.GetBytes(), cudaMemcpyDeviceToDevice));
        Cuda(cudaMemset(p.cudaData, 0x3d, p.GetBytes())); Cuda(cudaMemset(y.cudaData, 0x27, y.GetBytes()));
    }
    void Legacy(bool reference, int rank) {
        FastllmCudaSetLinearExactBatchThreshold(rows + 1); // Match verification/decode policy.
        Data empty;
        auto &out = reference ? referenceP : p;
        auto &hidden = reference ? referenceH : h;
        auto &normed = reference ? referenceY : y;
        Require(w.dataType == FP8_E4M3 ? FastllmCudaBFloat16MatMulFP8E4M3(x,w,empty,out,rows,1024,4096) :
                FastllmCudaBFloat16MatMulBFloat16(x, w, empty, out, rows, 1024, 4096), "reference projection failed");
        Require(FastllmCudaCustomAllReduceRows(out.cudaData, out.cudaData, rows * 4096, 4096, BFLOAT16, rank), "reference collective failed");
        FastllmCudaNaiveAddDecodeRMSNorm(hidden, out, n, 1e-6f, normed);
        if (withRouter) Require(FastllmCudaNaiveRouterSigmoid(normed, routerWeight, reference ? referenceRouter : router), "reference router");
    }
    bool Fused(int rank, int kind = 0) {
        FastllmCudaSetLinearExactBatchThreshold(kind == 17 && rank == 3 ? 1 : rows + 1);
        Data a, b, c, d, e, f, rw, ro;
        auto view = [](Data &dst, const Data &src) { dst.Resize(src.dims); dst.FakeFrom(src, 0); dst.strides = src.strides; };
        view(a, x); view(b, w); view(c, p); view(d, h); view(e, n); view(f, y);
        view(rw, routerWeight); view(ro, router);
        float eps = 1e-6f;
        if (rank == 3) {
            if (kind == 1) a.strides.back() = 2;
            if (kind == 2) b.dataType = FLOAT32;
            if (kind == 3) e.cudaData = (char *)e.cudaData + 4;
            if (kind == 4) f.cudaData = d.cudaData;
            if (kind == 5) a.Resize({1, rows + 1, 1024});
            if (kind == 6) d.cudaData = (char *)d.cudaData + 2;
            if (kind == 7) eps = -1.f;
            if (kind == 8) a.dataDevice = DataDevice::CPU;
            if (kind == 9) d.Resize({1, rows + 1, 4096});
            if (kind == 10) f.Resize({16}); // A borrowed output cannot be enlarged.
            if (kind == 11) a.cudaData = nullptr;
            if (kind == 15) rw.dataType = BFLOAT16;
            if (kind == 16) ro.cudaData = d.cudaData;
            if (kind == 12) b.dims = {2147483647, 2147483647, 4};
            if (kind == 19) {
                // An owned destination may need allocation even when the
                // communication tuple is already registered.
                f.isFake = false; f.cudaDataBorrowed = false;
                f.cudaData = nullptr; f.dims.clear(); f.strides.clear();
            }
        }
        Data &outputWeight = rank == 3 && (kind == 2 || kind == 12 || kind == 18) ? b : w;
        bool accepted = kind == 13 && rank == 3
            ? FastllmCudaSwaOutputAddRMSNorm(a, outputWeight, d, d, e, f, eps, rank)
            : kind == 14 && rank == 3
            ? FastllmCudaSwaOutputAddRMSNorm(a, outputWeight, c, d, e, c, eps, rank)
            : FastllmCudaSwaOutputAddRMSNormWithProjection(a, outputWeight, c, d, e, f, eps, rank, withRouter ? &rw : nullptr, withRouter ? &ro : nullptr, ordered, allowMma);
        if (kind == 13 && rank == 3) Require(d.dims == h.dims && d.dataType == h.dataType,
                                                    "alias rejection changed source metadata");
        if (kind == 9 && rank == 3) Require(c.dims == p.dims && f.dims == y.dims,
                                                   "bad input changed output metadata");
        if (kind == 10 && rank == 3) Require(f.dims == std::vector<int>{16}, "borrowed output was resized");
        if (kind == 18 && rank == 3) Require(b.extraCudaData.empty(), "fusion allocated a borrowed weight cache");
        if (kind == 19 && rank == 3) Require(!f.cudaData && f.dims.empty(), "cold capture prepared an output");
        return accepted;
    }
    void ExpectCaptureReject(int rank, int kind = 0) {
        const auto beforeP = Read(p), beforeH = Read(h), beforeY = Read(y), beforeRouter = Read(router);
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        const bool accepted = Fused(rank, kind);
        cudaGraph_t graph{}; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        size_t nodes = 0; Cuda(cudaGraphGetNodes(graph, nullptr, &nodes));
        Cuda(cudaGraphDestroy(graph));
        if (accepted || nodes) std::fprintf(stderr, "cold capture: rows=%d kind=%d mma=%d ordered=%d rank=%d accepted=%d nodes=%zu\n", rows, kind, allowMma, ordered, rank, accepted, nodes);
        Require(!accepted && nodes == 0 && beforeP == Read(p) && beforeH == Read(h) &&
            beforeY == Read(y) && beforeRouter == Read(router), "cold capture was not a clean fallback");
    }
    void Compare() {
        if (!withRouter && rows == 1 && w.dataType == BFLOAT16) {
            Require(Read(p) == Read(referenceP), "projection/reduction differs");
            Require(Read(h) == Read(referenceH), "residual differs");
            Require(Read(y) == Read(referenceY), "norm differs");
            return;
        }
        std::vector<std::pair<Data *, Data *>> pairs = {
            {&p, &referenceP}, {&h, &referenceH}, {&y, &referenceY}};
        if (withRouter) pairs.push_back({&router, &referenceRouter});
        if ((rows > 1 && !allowMma) || ordered) {
            for (auto pair : pairs)
                Require(Read(*pair.first) == Read(*pair.second), "multi-row tail lost reference rounding");
            return;
        }
        for (auto pair : pairs) {
            auto a = Read(*pair.first), b = Read(*pair.second);
            const size_t width = pair.first->Count(0) / rows;
            for (int row = 0; row < rows; ++row) {
                double error = 0, magnitude = 0;
                for (size_t i = row * width; i < (row + 1) * width; ++i) {
                    float x, z;
                    if (pair.first->dataType == FLOAT32) {
                        std::memcpy(&x, a.data() + i * 4, 4); std::memcpy(&z, b.data() + i * 4, 4);
                    } else {
                        uint16_t u, v;
                        std::memcpy(&u, a.data() + i * 2, 2); std::memcpy(&v, b.data() + i * 2, 2);
                        x = BFloat16BitsToFloat32(u); z = BFloat16BitsToFloat32(v);
                    }
                    Require(std::isfinite(x) && std::isfinite(z), "nonfinite attention output");
                    error += double(x - z) * (x - z); magnitude += double(z) * z;
                }
                maxL2 = std::max(maxL2, std::sqrt(error / std::max(magnitude, 1e-30)));
                Require(std::sqrt(error) <= .003 * std::sqrt(magnitude) + 1e-7,
                        "fused attention output row L2");
            }
        }
    }
};
}
int main(int argc, char **argv) {
    try {
        bool disabled = false, benchmark = false, mma = false, mx = false;
        for (int i = 1; i < argc; ++i) {
            disabled |= std::strcmp(argv[i], "--disabled") == 0;
            benchmark |= std::strcmp(argv[i], "--bench") == 0;
            mx |= std::strcmp(argv[i], "--mxfp8") == 0;
            mma |= std::strcmp(argv[i], "--mma") == 0;
        }
#ifdef _WIN32
        _putenv_s("FASTLLM_NAIVE_SWA_FUSION", "");
#else
        unsetenv("FASTLLM_NAIVE_SWA_FUSION");
#endif
        if (disabled) {
#ifdef _WIN32
            _putenv_s("FASTLLM_NAIVE_SWA_FUSION", "0");
#else
            setenv("FASTLLM_NAIVE_SWA_FUSION", "0", 1);
#endif
        }
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8) return 77;
        bool supported = true;
        for (int r = 0; r < 8; ++r) {
            cudaDeviceProp p{}; Cuda(cudaGetDeviceProperties(&p, r));
            supported = supported && p.major == 10 && p.minor == 3;
        }
        const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        if (!FastllmInitNccl(devices) || !FastllmCudaCustomAllReduceInit(devices)) return 77;
        bool firstCase = true, firstEightRows = true;
        const std::vector<int> shapes = mx ? std::vector<int>{1} :
            std::vector<int>{1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 8, -3, -8, 1, 0};
        for (int shape : shapes) {
            const int rows = shape ? std::abs(shape) : 1;
            const bool ordered = shape <= 0;
            std::array<std::unique_ptr<Tensors>, 8> data;
            Parallel([&](int r) { data[r] = std::make_unique<Tensors>(r, rows, ordered, mx); data[r]->allowMma = mma; });
            std::array<int, 8> accepted{};
            const bool expect = supported && !disabled && rows <= 8;
            if (firstCase) {
                // No warm call for this tuple. Capture must contain no kernel or
                // tensor write; a serving miss uses the ordinary prepared fallback.
                Parallel([&](int r) {
                    auto &d = *data[r]; d.Reset(0, r);
                    d.ExpectCaptureReject(r);
                });
            }
            firstCase = false;
            if (expect && rows == 8 && firstEightRows && (!mma || ordered)) {
                // Register p through the staged path, without warming the
                // group's fused eight-row GEMM. A tuple hit is insufficient.
                Parallel([&](int r) {
                    auto &d = *data[r]; d.Reset(0, r); d.Legacy(false, r);
                    Cuda(cudaDeviceSynchronize()); d.ExpectCaptureReject(r);
                });
                firstEightRows = false;
            }
            for (bool withRouter : {false, true}) {
                for (auto &d : data) d->withRouter = withRouter;
                for (int seed = 0; seed < 6; ++seed) {
                    Parallel([&](int r) { data[r]->Reset(seed, r); data[r]->Legacy(true, r); Cuda(cudaDeviceSynchronize()); });
                    Parallel([&](int r) {
                        accepted[r] = data[r]->Fused(r);
                        if (!accepted[r]) data[r]->Legacy(false, r);
                        Cuda(cudaDeviceSynchronize()); data[r]->Compare();
                    });
                    for (int ok : accepted) Require(bool(ok) == expect, "default/disabled/SM dispatch mismatch");
                }
                if (expect) {
                    if (rows > 1 || ordered) Parallel([&](int r) { data[r]->ExpectCaptureReject(r, 19); });
                    if (rows == 8 && (!mma || ordered)) Parallel([&](int r) {
                        auto &d = *data[r];
                        const auto p = Read(d.p), h = Read(d.h), y = Read(d.y), router = Read(d.router);
                        Require(!d.Fused(r, 18), "cold borrowed weight was admitted");
                        Cuda(cudaDeviceSynchronize());
                        Require(p == Read(d.p) && h == Read(d.h) && y == Read(d.y) && router == Read(d.router),
                            "cold weight rejection wrote tensor data");
                        d.ExpectCaptureReject(r, 18);
                    });
                    if (mma && !ordered && rows > 1) Parallel([&](int r) {
                        auto &d = *data[r]; d.Reset(27, r); d.Legacy(true, r);
                        Require(d.Fused(r, 18), "MMA rejected borrowed row-major weights");
                        Cuda(cudaDeviceSynchronize()); d.Compare();
                        cudaGraph_t graph{}; cudaGraphExec_t exec{};
                        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                        Require(d.Fused(r, 18), "borrowed MMA capture rejected");
                        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                        Cuda(cudaGraphInstantiate(&exec, graph, 0)); Cuda(cudaGraphDestroy(graph));
                        d.Reset(28, r); d.Legacy(true, r);
                        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                        Cuda(cudaDeviceSynchronize()); d.Compare(); Cuda(cudaGraphExecDestroy(exec));
                    });
                    // Use the same registered tuple for every rejection: a cache hit
                    // must not bypass a later single-rank eligibility failure.
                    for (int kind = 1; kind <= (withRouter ? (rows > 1 ? 17 : 16) : 14); ++kind) {
                        Parallel([&](int r) {
                            auto &d = *data[r]; d.Reset(10 + kind, r);
                            auto p = Read(d.p), h = Read(d.h), y = Read(d.y);
                            accepted[r] = d.Fused(r, kind); Cuda(cudaDeviceSynchronize());
                            Require(p == Read(d.p) && h == Read(d.h) && y == Read(d.y), "rejection wrote tensor data");
                        });
                        for (int ok : accepted) Require(!ok, "one bad rank did not reject the whole group");
                    }
                    Parallel([&](int r) {
                        auto &d = *data[r];
                        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                        accepted[r] = d.Fused(r);
                        cudaGraph_t graph = nullptr; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                        Cuda(cudaGraphInstantiate(&d.graph, graph, 0)); Cuda(cudaGraphDestroy(graph));
                    });
                    for (int ok : accepted) Require(ok, "warm graph capture rejected");
                    for (int seed = 30; seed < 40; ++seed) {
                        Parallel([&](int r) { data[r]->Reset(seed, r); data[r]->Legacy(true, r); Cuda(cudaDeviceSynchronize()); });
                        Parallel([&](int r) {
                            Cuda(cudaGraphLaunch(data[r]->graph, cudaStreamPerThread));
                            Cuda(cudaDeviceSynchronize()); data[r]->Compare();
                            if (mma) {
                                auto &d = *data[r];
                                const auto p = Read(d.p), h = Read(d.h), y = Read(d.y), router = Read(d.router);
                                d.Reset(seed, r); Require(d.Fused(r), "eager MMA replay rejected");
                                Cuda(cudaDeviceSynchronize());
                                Require(p == Read(d.p) && h == Read(d.h) && y == Read(d.y) && router == Read(d.router),
                                    "graph/eager output differs");
                            }
                        });
                    }
                }
                if (benchmark && rows <= 8 && supported) {
                    // One fixed graph per implementation, replayed on all ranks.
                    for (bool fused : {false, true}) {
                        std::array<float, 8> milliseconds{};
                        Parallel([&](int r) {
                            auto &d = *data[r];
                            if (d.graph) { Cuda(cudaGraphExecDestroy(d.graph)); d.graph = nullptr; }
                            d.Reset(71, r);
                            if (fused) Require(d.Fused(r), "benchmark fusion rejected"); else d.Legacy(false, r);
                            Cuda(cudaDeviceSynchronize());
                        });
                        Parallel([&](int r) {
                            auto &d = *data[r];
                            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                            if (fused) Require(d.Fused(r), "benchmark capture rejected"); else d.Legacy(false, r);
                            cudaGraph_t g{}; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
                            Cuda(cudaGraphInstantiate(&d.graph, g, 0)); Cuda(cudaGraphDestroy(g));
                        });
                        Parallel([&](int r) {
                            auto &d = *data[r];
                            cudaEvent_t begin{}, end{}; Cuda(cudaEventCreate(&begin)); Cuda(cudaEventCreate(&end));
                            Cuda(cudaEventRecord(begin, cudaStreamPerThread));
                            for (int j = 0; j < 500; ++j) Cuda(cudaGraphLaunch(d.graph, cudaStreamPerThread));
                            Cuda(cudaEventRecord(end, cudaStreamPerThread)); Cuda(cudaEventSynchronize(end));
                            Cuda(cudaEventElapsedTime(&milliseconds[r], begin, end));
                            Cuda(cudaEventDestroy(begin)); Cuda(cudaEventDestroy(end));
                        });
                        float slowest = 0;
                        for (float value : milliseconds) slowest = std::max(slowest, value);
                        std::printf("BENCH rows=%d router=%d fused=%d us=%.6f\n", rows, withRouter, fused, slowest * 1000 / 500);
                    }
                }
                Parallel([&](int r) { if (data[r]->graph) { Cuda(cudaGraphExecDestroy(data[r]->graph)); data[r]->graph = nullptr; } });
            }
            double maxL2 = 0;
            for (const auto &d : data) maxL2 = std::max(maxL2, d->maxL2);
            Parallel([&](int r) { data[r].reset(); });
            std::printf("Attention output PASS: rows=%d enabled=%d max_row_L2=%.9g; numerical/collective rejection/Graph replay\n", rows, expect, maxL2);
        }
        if (supported && !disabled && !mx) {
            // Queue different row counts without intervening host/GPU waits.
            // All graphs share the same epoch/mailbox, including OutputNorm's
            // one-row completion counter. Keep their tensors alive throughout.
            std::array<std::array<std::unique_ptr<Tensors>, 3>, 8> mixed;
            const int counts[] = {1, 3, 8};
            for (int i = 0; i < 3; ++i) {
                Parallel([&](int r) {
                    auto &d = mixed[r][i]; d = std::make_unique<Tensors>(r, counts[i]);
                    d->allowMma = mma; d->withRouter = i == 1; d->Reset(91 + i, r);
                    d->Legacy(true, r); Require(d->Fused(r), "mixed graph warmup rejected");
                    Cuda(cudaDeviceSynchronize()); d->Compare();
                });
                Parallel([&](int r) {
                    auto &d = *mixed[r][i];
                    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    Require(d.Fused(r), "mixed graph capture rejected");
                    cudaGraph_t graph{}; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    Cuda(cudaGraphInstantiate(&d.graph, graph, 0)); Cuda(cudaGraphDestroy(graph));
                });
            }
            Parallel([&](int r) {
                for (int j = 0; j < 12; ++j)
                    for (auto &d : mixed[r]) Cuda(cudaGraphLaunch(d->graph, cudaStreamPerThread));
                Cuda(cudaDeviceSynchronize());
            });
            Parallel([&](int r) {
                for (int j = 0; j < 12; ++j)
                    for (auto &d : mixed[r]) d->Legacy(true, r);
                Cuda(cudaDeviceSynchronize());
                for (auto &d : mixed[r]) { d->Compare(); Cuda(cudaGraphExecDestroy(d->graph)); d.reset(); }
            });
            std::printf("Attention output PASS: queued 1/3/8-row graphs share one mailbox\n");
        }
        if (mma && supported && !disabled) {
            // MMA neither needs a BLAS handle nor attaches a bias cache to weights.
            std::array<std::unique_ptr<Tensors>, 8> cold;
            Parallel([&](int r) {
                auto &d = cold[r]; d = std::make_unique<Tensors>(r, 8); d->allowMma = true;
                d->Reset(71, r); Require(d->w.extraCudaData.empty(), "cold weight already prepared");
                Require(d->Fused(r), "cold MMA projection rejected"); Cuda(cudaDeviceSynchronize());
                Require(d->w.extraCudaData.empty(), "MMA created a weight auxiliary cache");
                d->Legacy(true, r); Cuda(cudaDeviceSynchronize()); d->Compare();
            });
            Parallel([&](int r) { cold[r].reset(); });
        }
        if (supported && !disabled) TestAttentionOutputMailboxWrap(mma);
        FastllmCudaCustomAllReduceReset();
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "SWA output FAIL: %s\n", e.what()); return 1; }
}
