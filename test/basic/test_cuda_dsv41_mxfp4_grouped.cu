#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cmath>
#include "../../src/devices/cuda/moe/fastllm-moe-deepseekv41-cache.cuh"
#include "fastllm-cuda.cuh"
#define CUDA(x)                                                                                              \
    do {                                                                                                     \
        cudaError_t e = (x);                                                                                 \
        if (e != cudaSuccess) {                                                                              \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                        \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
namespace ref = fastllm::cuda::dsv41_cache;

// Optional SM12x test: enable FASTLLM_CUDA_TRITON and select a compiler/cache.
// The CUDA GEMV oracle preserves the original MXFP4 and activation rounding.
int main(int argc, char **argv) {
    bool planar = false, draft = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--planar") planar = true;
        else if (option == "--draft") draft = true;
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (draft && !planar) {
        std::fprintf(stderr, "--draft requires --planar\n");
        return 1;
    }
    if (!fastllm::CudaTritonMxfp4GroupedEnabled())
        return 77;
    const char *flag = std::getenv("FASTLLM_CUDA_TRITON_MXFP4_GROUPED");
    const std::string original = flag ? flag : "";
    const bool hadFlag = flag != nullptr;
    for (const char *value : {"0", "false", "off", "FALSE", "OFF"}) {
        setenv("FASTLLM_CUDA_TRITON_MXFP4_GROUPED", value, 1);
        if (fastllm::CudaTritonMxfp4GroupedEnabled())
            return 5;
    }
    setenv("FASTLLM_CUDA_TRITON_MXFP4_GROUPED", "true", 1);
    if (!fastllm::CudaTritonMxfp4GroupedEnabled())
        return 5;
    if (hadFlag)
        setenv("FASTLLM_CUDA_TRITON_MXFP4_GROUPED", original.c_str(), 1);
    else
        unsetenv("FASTLLM_CUDA_TRITON_MXFP4_GROUPED");
    constexpr int H = 5120, I = 2304, S = 32;
    const int K = draft ? 3 : 6;
    const size_t down = size_t(2) * I * (H / 32) * 17, stride = down + size_t(H) * (I / 32) * 17;
    std::mt19937 gen(42);
    std::vector<uint8_t> w(stride * S);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = i % 17 == 16 ? 113 + gen() % 10 : gen() % 256;
    uint8_t *dw;
    CUDA(cudaMalloc(&dw, w.size()));
    CUDA(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    uint8_t *dwPlanar = nullptr;
    if (planar) {
        std::vector<uint8_t> packed(w.size());
        for (int slot = 0; slot < S; ++slot) for (int part = 0; part < 2; ++part) {
            const size_t offset = part ? down : 0, bytes = part ? stride - down : down;
            const auto *src = w.data() + size_t(slot) * stride + offset;
            auto *dst = packed.data() + size_t(slot) * stride + offset;
            for (size_t block = 0; block < bytes / 17; ++block) {
                std::memcpy(dst + block * 16, src + block * 17, 16);
                dst[bytes / 17 * 16 + block] = src[block * 17 + 16];
            }
        }
        CUDA(cudaMalloc(&dwPlanar, packed.size()));
        CUDA(cudaMemcpy(dwPlanar, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    }
    w.clear();
    w.shrink_to_fit();
    __nv_bfloat16 *x, *a, *b;
    float *y, *z, *scores;
    int *slots, *map;
    CUDA(cudaMalloc(&x, 8 * H * 2));
    CUDA(cudaMalloc(&a, 8 * K * I * 2));
    CUDA(cudaMalloc(&b, 8 * K * I * 2));
    CUDA(cudaMalloc(&y, 8 * K * H * 4));
    CUDA(cudaMalloc(&z, 8 * K * H * 4));
    CUDA(cudaMalloc(&scores, 8 * K * 4));
    CUDA(cudaMalloc(&slots, 8 * K * 4));
    CUDA(cudaMalloc(&map, 8 * K * 4));
    std::vector<__nv_bfloat16> hx(8 * H);
    for (auto &v : hx)
        v = __float2bfloat16(float(int(gen() % 2048) - 1024) / 1024);
    CUDA(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hs(8 * K);
    for (auto &v : hs)
        v = float(gen() % 1024) / 2048;
    CUDA(cudaMemcpy(scores, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&end));
    for (int rows : {2, 3, 4, 6, 8})
        for (int shift : {0, 2, 6})
            for (int active : {0, 4}) {
                int n = rows * K;
                const int count = active ? active : n;
                CUDA(cudaMemset(y, 0, n * H * sizeof(float)));
                CUDA(cudaMemset(z, 0, n * H * sizeof(float)));
                std::vector<int> order(n), ss(n);
                for (int i = 0; i < n; ++i)
                    order[i] = i;
                auto slot = [&](int r) { return (r % K + (r / K) * shift) % S; };
                std::sort(order.begin(), order.end(), [&](int l, int r) { return slot(l) < slot(r); });
                for (int i = 0; i < n; ++i)
                    ss[i] = slot(order[i]);
                std::vector<int> grouped, unique;
                for (int first = 0; first < n;) {
                    int end = first + 1;
                    while (end < n && ss[end] == ss[first])
                        ++end;
                    auto &target = end - first > 1 ? grouped : unique;
                    for (int at = first; at < end; ++at)
                        target.push_back(order[at]);
                    first = end;
                }
                grouped.insert(grouped.end(), unique.begin(), unique.end());
                order = grouped;
                for (int i = 0; i < n; ++i)
                    ss[i] = slot(order[i]);
                CUDA(cudaMemcpy(slots, ss.data(), n * 4, cudaMemcpyHostToDevice));
                CUDA(cudaMemcpy(map, order.data(), n * 4, cudaMemcpyHostToDevice));
                void *raw = nullptr;
                if (!planar) CUDA(cudaMalloc(&raw, 8 * K * 2 * I * 2));
                fastllm::Data input(fastllm::BFLOAT16, {rows, H}), activation(fastllm::BFLOAT16, {rows * K, I});
                for (auto *d : {&input, &activation}) {
                    d->isFake = true;
                    d->dataDevice = fastllm::CUDA;
                    d->dataDeviceIds = {0};
                }
                input.cudaData = x;
                activation.cudaData = b;
                auto admitted = [&](int count) {
                    return fastllm::TryCudaTritonMxfp4Grouped(input, activation, planar ? dwPlanar : dw, slots, map, scores, z, raw,
                                                              count, H, I, K, stride, down, 10, planar);
                };
                if (admitted(n + 1) || admitted(3))
                    return 6;
                activation.cudaData = x;
                if (admitted(n))
                    return 6;
                activation.cudaData = b;
                activation.dataDeviceIds = {1};
                if (admitted(n))
                    return 6;
                activation.dataDeviceIds = {0};
                input.multiDeviceData = true;
                if (admitted(n))
                    return 6;
                input.multiDeviceData = false;
                auto run = [&](bool candidate) {
                    if (!candidate) {
                        ref::Gate<true><<<dim3(I / 32, count), 512, 0, cudaStreamPerThread>>>(
                            x, slots, dw, scores, a, H, I, stride, 10, K, map);
                        ref::Down<<<dim3(H / 8, count), 128, 0, cudaStreamPerThread>>>(a, slots, dw, y, H, I, stride,
                                                                                   down, map);
                    } else if (!fastllm::TryCudaTritonMxfp4Grouped(input, activation, planar ? dwPlanar : dw, slots, map, scores, z,
                                                                  raw, count, H, I, K, stride, down, 10, planar)) {
                        fprintf(stderr, "Triton path rejected\n");
                        exit(2);
                    }
                };
                run(false);
                run(true);
                CUDA(cudaDeviceSynchronize());
                std::vector<float> yy(n * H), zz(n * H);
                CUDA(cudaMemcpy(yy.data(), y, n * H * 4, cudaMemcpyDeviceToHost));
                CUDA(cudaMemcpy(zz.data(), z, n * H * 4, cudaMemcpyDeviceToHost));
                double err = 0, denom = 0;
                int unequal = 0;
                for (int j = 0; j < n * H; ++j) {
                    if (!std::isfinite(zz[j]))
                        return 3;
                    double d = yy[j] - zz[j];
                    err += d * d;
                    denom += double(yy[j]) * yy[j];
                    unequal += yy[j] != zz[j];
                }
                double rel = std::sqrt(err / denom);
                if (rel >= 0.003163) {
                    fprintf(stderr, "FAIL rel=%g\n", rel);
                    return 4;
                }
                float times[2];
                for (int v = 0; v < 2; ++v) {
                    for (int j = 0; j < 3; ++j)
                        run(v);
                    CUDA(cudaEventRecord(start, cudaStreamPerThread));
                    for (int j = 0; j < 20; ++j)
                        run(v);
                    CUDA(cudaEventRecord(end, cudaStreamPerThread));
                    CUDA(cudaEventSynchronize(end));
                    CUDA(cudaEventElapsedTime(&times[v], start, end));
                }
                printf("rows=%d shift=%d active=%d rel_l2=%g mismatch=%d/%d baseline_ms=%.4f triton_ms=%.4f\n", rows, shift,
                       count, rel, unequal, n * H, times[0] / 20, times[1] / 20);
                fflush(stdout);
                if (rows == 8 && shift == 2 && active == 0) {
                    cudaGraph_t graph;
                    cudaGraphExec_t executable;
                    CUDA(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                    run(true);
                    CUDA(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    for (int replay = 0; replay < 3; ++replay) {
                        for (auto &value : hx)
                            value = __float2bfloat16(float(int(gen() % 2048) - 1024) / 1024);
                        for (auto &route : order)
                            route = n - 1 - route;
                        CUDA(cudaMemcpyAsync(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice,
                                             cudaStreamPerThread));
                        CUDA(cudaMemcpyAsync(map, order.data(), n * 4, cudaMemcpyHostToDevice,
                                             cudaStreamPerThread));
                        CUDA(cudaGraphLaunch(executable, cudaStreamPerThread));
                        run(false);
                        CUDA(cudaStreamSynchronize(cudaStreamPerThread));
                        CUDA(cudaMemcpy(yy.data(), y, n * H * 4, cudaMemcpyDeviceToHost));
                        CUDA(cudaMemcpy(zz.data(), z, n * H * 4, cudaMemcpyDeviceToHost));
                        double error = 0, norm = 0;
                        for (int j = 0; j < n * H; ++j) {
                            if (!std::isfinite(zz[j]))
                                return 7;
                            double delta = yy[j] - zz[j];
                            error += delta * delta;
                            norm += double(yy[j]) * yy[j];
                        }
                        double relative = std::sqrt(error / norm);
                        if (relative >= 0.003163)
                            return 7;
                        printf("GRAPH_PASS replay=%d rel_l2=%g\n", replay, relative);
                    }
                    CUDA(cudaGraphExecDestroy(executable));
                    CUDA(cudaGraphDestroy(graph));
                }
                if (raw) CUDA(cudaFree(raw));
            }
    puts("ALL_PASS: 30 full/partial route layouts through engine compiler metadata and CUDA wrapper");
    if (dwPlanar) CUDA(cudaFree(dwPlanar));
    for (void *p : {(void *)x, (void *)a, (void *)b, (void *)y, (void *)z, (void *)scores, (void *)slots,
                    (void *)map, (void *)dw})
        CUDA(cudaFree(p));
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(end));
    return 0;
}
