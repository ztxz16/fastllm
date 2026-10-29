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
int main() {
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
    constexpr int H = 5120, I = 2304, K = 6, S = 32;
    const size_t down = size_t(2) * I * (H / 32) * 17, stride = down + size_t(H) * (I / 32) * 17;
    std::mt19937 gen(42);
    std::vector<uint8_t> w(stride * S);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = i % 17 == 16 ? 113 + gen() % 10 : gen() % 256;
    uint8_t *dw;
    CUDA(cudaMalloc(&dw, w.size()));
    CUDA(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
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
    for (int rows : {2, 4, 6, 8})
        for (int shift : {0, 2, 6}) {
            int n = rows * K;
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
            int repeated = grouped.size();
            grouped.insert(grouped.end(), unique.begin(), unique.end());
            order = grouped;
            for (int i = 0; i < n; ++i)
                ss[i] = slot(order[i]);
            CUDA(cudaMemcpy(slots, ss.data(), n * 4, cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(map, order.data(), n * 4, cudaMemcpyHostToDevice));
            void *raw;
            CUDA(cudaMalloc(&raw, 8 * K * 2 * I * 2));
            fastllm::Data input(fastllm::BFLOAT16, {rows, H}), activation(fastllm::BFLOAT16, {rows * K, I});
            for (auto *d : {&input, &activation}) {
                d->isFake = true;
                d->dataDevice = fastllm::CUDA;
                d->dataDeviceIds = {0};
            }
            input.cudaData = x;
            activation.cudaData = b;
            auto admitted = [&](int count) {
                return fastllm::TryCudaTritonMxfp4Grouped(input, activation, dw, slots, map, scores, z, raw,
                                                          count, H, I, K, stride, down, 10);
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
                    ref::Gate<true><<<dim3(I / 32, n), 512, 0, cudaStreamPerThread>>>(
                        x, slots, dw, scores, a, H, I, stride, 10, K, map);
                    ref::Down<<<dim3(H / 8, n), 128, 0, cudaStreamPerThread>>>(a, slots, dw, y, H, I, stride,
                                                                               down, map);
                } else {
                    int done = 0;
                    if (repeated >= 4) {
                        if (!fastllm::TryCudaTritonMxfp4Grouped(input, activation, dw, slots, map, scores, z,
                                                                raw, repeated, H, I, K, stride, down, 10)) {
                            fprintf(stderr, "Triton path rejected\n");
                            exit(2);
                        }
                        done = repeated;
                    }
                    if (done < n) {
                        ref::Gate<true><<<dim3(I / 32, n - done), 512, 0, cudaStreamPerThread>>>(
                            x, slots + done, dw, scores, b, H, I, stride, 10, K, map + done);
                        ref::Down<<<dim3(H / 8, n - done), 128, 0, cudaStreamPerThread>>>(
                            b, slots + done, dw, z, H, I, stride, down, map + done);
                    }
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
            printf("rows=%d shift=%d rel_l2=%g mismatch=%d/%d baseline_ms=%.4f triton_ms=%.4f\n", rows, shift,
                   rel, unequal, n * H, times[0] / 20, times[1] / 20);
            fflush(stdout);
            if (rows == 8 && shift == 2) {
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
            CUDA(cudaFree(raw));
        }
    puts("ALL_PASS: 12 route/reuse layouts through engine compiler metadata and CUDA wrapper");
    for (void *p : {(void *)x, (void *)a, (void *)b, (void *)y, (void *)z, (void *)scores, (void *)slots,
                    (void *)map, (void *)dw})
        CUDA(cudaFree(p));
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(end));
    return 0;
}
