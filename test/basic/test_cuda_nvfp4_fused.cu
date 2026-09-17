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
void Run(bool gate) {
    int n = gate ? 34816 : 5120, k = gate ? 5120 : 17408, out = gate ? n / 2 : n, groups = k / 16;
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
    Data x(FLOAT16, {1, 1, k}, std::vector<float>(k, .01f)),
        y(FLOAT16, {1, 1, out}, std::vector<float>(out, 0)),
        ref(FLOAT16, {1, 1, out}, std::vector<float>(out, 0)), middle(FLOAT16), empty;
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
    Require(!block(y), "cold weight unexpectedly fused");
    Check(cudaDeviceSynchronize());
    FastllmCudaSetNcclForceSync(false);
    Require(FastllmCudaHasNVFP4MarlinLayout(w), "repack failed");
    Require(can() == expectFused, "prepared weight availability incorrect");
    auto flag = gate ? "FASTLLM_CUDA_NVFP4_SWIGLU" : "FASTLLM_CUDA_NVFP4_ADD";
    setenv(flag, "0", 1);
    Require(!can(), "disable flag ignored");
    setenv(flag, "1", 1);
    auto old = x.dataType;
    x.dataType = BFLOAT16;
    Require(!can(), "BF16 admitted");
    x.dataType = old;
    x.Resize({2, k});
    Require(!can(), "batch2 admitted");
    x.Resize({1, 1, k});
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
    auto afterCan = Read(y);
    Require(memcmp(before.data(), afterCan.data(), out * 2) == 0, "CanRun changed output");
    std::vector<uint8_t> packed(size_t(n) * k * 9 / 16), packedAfter(packed.size());
    Check(cudaMemcpy(packed.data(), w.cudaData, packed.size(), cudaMemcpyDeviceToHost));
    std::vector<half> hx(k), res(out);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(block(y) == expectFused, "graph dispatch incorrect");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < k; ++i)
            hx[i] = half(.15f * sinf(i * .017f + step * .7f));
        for (int i = 0; i < out; ++i)
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
        for (int i = 0; i < out; ++i) {
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
                    sum += double(float(weight)) * float(hx[g * 16 + j]);
                }
            }
            return float(half(float(sum) * (common * 128)));
        };
        double ce = 0, cr = 0;
        for (int i = 0; i < 100; ++i) {
            int r = (i * 173 + 7) % out;
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
        auto after = Read(x);
        Require(memcmp(after.data(), hx.data(), k * 2) == 0, "input changed");
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
        Run(true);
        Run(false);
        puts("PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
