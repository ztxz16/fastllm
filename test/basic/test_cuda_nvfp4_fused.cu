#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-nvfp4-fused.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using namespace fastllm;
static bool expectFused = true;
void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool b, const char *m) {
    if (!b)
        throw std::runtime_error(m);
}
std::vector<half> Read(Data &d) {
    std::vector<half> v(d.Count(0));
    Check(cudaMemcpy(v.data(), d.cudaData, v.size() * 2, cudaMemcpyDeviceToHost));
    return v;
}
void Upload(Data &d, const std::vector<half> &v) {
    Check(cudaMemcpy(d.cudaData, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
}
uint16_t Bits(half h) {
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}
half Half(uint16_t b) {
    half h;
    memcpy(&h, &b, 2);
    return h;
}
void Run(bool gate, int n, int k, bool profitable = true, int batch = 1) {
    const bool useFusion = expectFused && profitable;
    int out = gate ? n / 2 : n, groups = k / 16;
    printf("SHAPE gate=%d N=%d K=%d batch=%d\n", gate, n, k, batch);
    const int outputCount = batch * out, inputCount = batch * k;
    const float common = .000371f;
    std::vector<uint8_t> raw(size_t(n) * groups * 12);
    for (int r = 0; r < n; ++r)
        for (int g = 0; g < groups; ++g) {
            auto p = raw.data() + (size_t(r) * groups + g) * 12;
            for (int j = 0; j < 8; ++j)
                p[j] = uint8_t((r * 31 + g * 43 + j * 59 + (r * g) % 113) & 255);
            __nv_fp8_e4m3 sc;
            sc.__x = uint8_t((r + g * 7) % 127);
            float effective = common * float(sc);
            memcpy(p + 8, &effective, 4);
        }
    Data w(NVFP4_BLOCK_16, {n, k});
    w.blockM = 16;
    w.blockK = 1;
    w.weightType = WeightType::LINEAR;
    w.scales = {common};
    w.Allocate();
    memcpy(w.cpuData, raw.data(), raw.size());
    Data x(FLOAT16, {batch, 1, k}, std::vector<float>(inputCount, .01f)),
        y(FLOAT16, {batch, 1, out}, std::vector<float>(outputCount, 0)),
        ref(FLOAT16, {batch, 1, out}, std::vector<float>(outputCount, 0)), middle(FLOAT16), empty;
    middle.dataDevice = DataDevice::CUDA;
    middle.dataDeviceIds = {0};
    for (Data *d : {&w, &x, &y, &ref})
        d->ToDevice(CUDA, std::vector<int>{0});
    auto can = [&]() { return FastllmCudaNvfp4FusedCanRun(x, w, empty, y, gate); };
    auto block = [&](Data &o) {
        return gate ? CudaNvfp4LinearSwigluBlock(x, w, empty, middle, o)
                    : CudaNvfp4LinearAddBlock(x, w, empty, middle, o);
    };
    Require(!can(), "unprepared weight admitted");
    FastllmCudaSetNcclForceSync(true);
    if (batch > 8) { x.Resize({1, 1, k}); y.Resize({1, 1, out}); }
    Require(!block(y), "cold weight unexpectedly fused");
    Check(cudaDeviceSynchronize());
    if (batch > 8) {
        x.Resize({batch, 1, k}); y.Resize({batch, 1, out});
        Require(!block(y), "large batch fused during warmup");
        Check(cudaDeviceSynchronize());
    }
    FastllmCudaSetNcclForceSync(false);
    Require(FastllmCudaHasNVFP4MarlinLayout(w), "repack failed");
    Require(can() == useFusion, "prepared weight availability incorrect");
    auto flag = gate ? "FASTLLM_CUDA_NVFP4_SWIGLU" : "FASTLLM_CUDA_NVFP4_ADD";
    setenv(flag, "0", 1);
    Require(!can(), "disable flag ignored");
    setenv(flag, "1", 1);
    auto old = x.dataType;
    x.dataType = BFLOAT16;
    Require(!can(), "BF16 admitted");
    x.dataType = old;
    x.Resize({9, k});
    Require(!can(), "batch9 admitted");
    x.Resize({batch, 1, k});
    w.blockM = 32;
    Require(!can(), "wrong block size admitted");
    w.blockM = 16;
    Data bias(FLOAT32, {n});
    Require(!FastllmCudaNvfp4FusedCanRun(x, w, bias, y, gate), "bias admitted");
    // CanRun must neither touch residual storage nor admit aliased buffers.
    auto before = Read(y);
    void *saved = y.cudaData;
    y.cudaData = x.cudaData;
    Require(!can(), "input/output alias admitted");
    y.cudaData = saved;
    if (batch > 1) {
        y.cudaData = static_cast<char *>(saved) + 2;
        Require(!can(), "unaligned residual admitted");
        y.cudaData = saved;
    }
    auto afterCan = Read(y);
    Require(memcmp(before.data(), afterCan.data(), outputCount * 2) == 0, "CanRun changed output");
    std::vector<uint8_t> packed(size_t(n) * k * 9 / 16), packedAfter(packed.size());
    Check(cudaMemcpy(packed.data(), w.cudaData, packed.size(), cudaMemcpyDeviceToHost));
    std::vector<half> hx(inputCount), res(outputCount);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(block(y) == useFusion, "graph dispatch incorrect");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < inputCount; ++i)
            hx[i] = half(.15f * sinf(i * .017f + step * .7f));
        for (int i = 0; i < outputCount; ++i)
            res[i] = half(.2f * cosf(i * .013f + step));
        Upload(x, hx);
        Upload(y, res);
        Upload(ref, res);
        setenv(flag, "0", 1);
        Require(!block(ref), "disabled block fused");
        setenv(flag, "1", 1);
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaDeviceSynchronize());
        auto a = Read(y), b = Read(ref);
        double se = 0, sr = 0;
        float max = 0;
        for (int i = 0; i < outputCount; ++i) {
            float av = float(a[i]), bv = float(b[i]);
            Require(std::isfinite(av) && std::isfinite(bv), "nonfinite output");
            se += (av - bv) * (av - bv);
            sr += bv * bv;
            max = fmaxf(max, fabsf(av - bv));
        }
        double rel = sqrt(se / (sr + 1e-30));
        printf("gate=%d replay=%d fallback_rel_rms=%.9g max_abs=%.9g\n", gate, step, rel, max);
        Require(rel < .003, "fallback disagreement");
        // Independently reconstruct selected logical rows from original bytes, using the
        // same representable normalized half weights as the Marlin conversion.
        auto dot = [&](int r) {
            const int row = gate ? 0 : r / out;
            if (!gate) r %= out;
            double sum = 0;
            for (int g = 0; g < groups; ++g) {
                auto p = raw.data() + (size_t(r) * groups + g) * 12;
                float eff;
                memcpy(&eff, p + 8, 4);
                half shifted = half(float(half(eff / common)) * 128);
                uint8_t enc = float(shifted) < 2 ? 0 : uint8_t(Bits(shifted) >> 7);
                half sc = Half(uint16_t(enc) << 7);
                for (int j = 0; j < 16; ++j) {
                    int code = (p[j / 2] >> ((j & 1) * 4)) & 15;
                    half v = Half(uint16_t(((code & 8) << 12) | ((code & 7) << 9)));
                    half weight = half(float(v) * float(sc));
                    sum += double(float(weight)) * float(hx[row * k + g * 16 + j]);
                }
            }
            return float(half(float(sum) * (common * 128)));
        };
        double ce = 0, cr = 0;
        for (int i = 0; i < 100; ++i) {
            int r = (i * 7919 + 7) % outputCount;
            float v = dot(r);
            if (gate) {
                float e = expf(-fabsf(v));
                v = (v >= 0 ? v : v * e) / (1 + e) * dot(r + n / 2);
            } else
                v += float(res[r]);
            float expected = float(half(v)), err = float(a[r]) - expected;
            ce += err * err;
            cr += expected * expected;
        }
        printf("gate=%d replay=%d cpu_rel_rms=%.9g\n", gate, step, sqrt(ce / (cr + 1e-30)));
        Require(sqrt(ce / (cr + 1e-30)) < .002, "CPU reference disagreement");
        if (!gate && n == 5120 && k == 8704) {
            setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", "0", 1);
            Upload(ref, res);
            Require(block(ref) == (useFusion && batch == 1), "shape tuning changed Block admission");
            Check(cudaDeviceSynchronize());
            auto generic = Read(ref);
            Require(batch > 1 || !memcmp(a.data(), generic.data(), outputCount * sizeof(half)),
                    "shape specialization changed rounding");
            setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", "1", 1);
            int major = 0, minor = 0;
            Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
            Check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
            Require(FastllmCudaNvfp4ShapeGemvCanRun(x, w, empty, y) ==
                    (useFusion && batch == 1 && major * 10 + minor >= 75), "plain GEMV admission incorrect");
            Executor linear;
            linear.SetFirstDevice("cuda:0");
            auto runLinear = [&](Data &o) {
                linear.RunOnDevice("cuda", "Linear",
                    {{"input", &x}, {"weight", &w}, {"bias", &empty}, {"output", &o}}, {}, {});
            };
            setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", "0", 1);
            Require(!FastllmCudaNvfp4ShapeGemvCanRun(x, w, empty, y), "plain disable ignored");
            Upload(ref, res);
            runLinear(ref);
            Check(cudaDeviceSynchronize());
            setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", "1", 1);
            Upload(y, res); // Plain epilogue must overwrite these nonzero values.
            cudaGraph_t lg;
            cudaGraphExec_t le;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            runLinear(y);
            Check(cudaStreamEndCapture(cudaStreamPerThread, &lg));
            Check(cudaGraphInstantiate(&le, lg, nullptr, nullptr, 0));
            Check(cudaGraphLaunch(le, cudaStreamPerThread));
            Check(cudaDeviceSynchronize());
            auto plain = Read(y), marlin = Read(ref);
            double pe = 0, pr = 0;
            for (int i = 0; i < outputCount; ++i) {
                float err = float(plain[i]) - float(marlin[i]);
                Require(std::isfinite(float(plain[i])), "plain GEMV nonfinite");
                pe += err * err;
                pr += float(marlin[i]) * float(marlin[i]);
            }
            printf("plain replay=%d fallback_rel_rms=%.9g\n", step, sqrt(pe / (pr + 1e-30)));
            Require(sqrt(pe / (pr + 1e-30)) < .003, "plain GEMV disagreement");
            Check(cudaGraphExecDestroy(le));
            Check(cudaGraphDestroy(lg));
        }
        auto after = Read(x);
        Require(memcmp(after.data(), hx.data(), inputCount * 2) == 0, "input changed");
    }
    if (std::getenv("BENCH_SMALL_BATCH") && !gate && n == 5120 && k == 8704) {
        Executor linear;
        linear.SetFirstDevice("cuda:0");
        for (bool add : {false, true}) {
            for (bool enabled : {false, true}) {
                setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", enabled ? "1" : "0", 1);
                setenv("FASTLLM_CUDA_NVFP4_ADD", enabled ? "1" : "0", 1);
                cudaGraph_t bg;
                cudaGraphExec_t be;
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                for (int j = 0; j < 32; ++j) {
                    if (add) block(y);
                    else linear.RunOnDevice("cuda", "Linear",
                        {{"input", &x}, {"weight", &w}, {"bias", &empty}, {"output", &y}}, {}, {});
                }
                Check(cudaStreamEndCapture(cudaStreamPerThread, &bg));
                Check(cudaGraphInstantiate(&be, bg, nullptr, nullptr, 0));
                for (int j = 0; j < 3; ++j) Check(cudaGraphLaunch(be, cudaStreamPerThread));
                cudaEvent_t start, end;
                Check(cudaEventCreate(&start)); Check(cudaEventCreate(&end));
                Check(cudaEventRecord(start, cudaStreamPerThread));
                for (int j = 0; j < 20; ++j) Check(cudaGraphLaunch(be, cudaStreamPerThread));
                Check(cudaEventRecord(end, cudaStreamPerThread)); Check(cudaEventSynchronize(end));
                float ms; Check(cudaEventElapsedTime(&ms, start, end));
                printf("BENCH batch=%d add=%d fused=%d us=%.6f\n", batch, add, enabled, ms*1000/(20*32));
                Check(cudaEventDestroy(start)); Check(cudaEventDestroy(end));
                Check(cudaGraphExecDestroy(be)); Check(cudaGraphDestroy(bg));
            }
        }
        setenv("FASTLLM_CUDA_NVFP4_ADD", "1", 1);
        setenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING", "1", 1);
    }
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
    Check(cudaMemcpy(packedAfter.data(), w.cudaData, packedAfter.size(), cudaMemcpyDeviceToHost));
    Require(packed == packedAfter, "packed weights or scales changed");
}
int main() {
    try {
        const char *flag = std::getenv("EXPECT_FUSED");
        expectFused = !flag || std::strcmp(flag, "0");
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            return 77;
        int major = 0, minor = 0;
        Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
        Check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
        if (major * 10 + minor < 75)
            return 77; // This regression exercises the Marlin layout, not the SM70 backend.
        Executor executor;
        executor.SetFirstDevice("cuda:0");
        if (std::getenv("SMALL_BATCH_ONLY")) {
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8}) Run(false, 5120, 8704, true, batch);
            puts("PASS"); return 0;
        }
        for (int batch : {2, 3, 4, 5, 6, 7, 8}) Run(false, 5120, 8704, true, batch);
        Run(false, 5120, 8704, false, 9);
        for (int n : {34816, 17408, 8704, 1536}) Run(true, n, 5120);
        Run(true, 11776, 5120, false);
        Run(true, 11520, 5120, false);
        Run(true, 4096, 3072);
        Run(true, 8192, 4352);
        for (int k : {17408, 8704, 5888, 5760, 4352, 3072}) Run(false, 5120, k);
        puts("PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
