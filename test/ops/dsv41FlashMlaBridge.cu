// Native test entry for the optional full attention/output adapter.
#include "fastllm.h"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "deepseekv41-flashmla.cuh"
#include <cstdlib>
#include <cstring>
using namespace fastllm;

extern "C" int test_v41_mla(int m, int width, int ratio, float scale, void *q, void *kv, void *angles, void *req,
                            void *idx, void *sink, void *output, void *codes, void *scales, void *currentPacked) {
    FastllmV41FullDecodeConfig c;
    c.batch = m;
    c.heads = 64;
    c.ratio = ratio;
    c.window = 128;
    c.ringMode = 1;
    c.ringRowBytes = 528;
    c.cmpMode = 3;
    c.cmpRowBytes = 288;
    c.topK = width;
    c.softmaxScale = scale;
    auto w = FastllmV41CreateMla(c);
    if (!w)
        return 1;
    cudaStream_t s = cudaStreamPerThread;
    cudaMemcpyAsync(w->query, q, size_t(m) * 32768 * 2, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(w->angles, angles, size_t(m) * 64 * 4, cudaMemcpyDeviceToDevice, s);
    FastllmV41FullDecodeBuffers b;
    b.savedKV = currentPacked;
    b.kv = kv;
    b.output = output;
    b.topk = idx;
    b.sink = sink;
    if (FastllmV41RunMla(*w, c, b, static_cast<const FastllmV41FullDecodeRequest *>(req), s, true) != cudaSuccess)
        return 2;
    cudaMemcpyAsync(codes, w->codes, size_t(m) * 32768, cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(scales, w->scales, size_t(w->scalePitch) * 256 * 4, cudaMemcpyDeviceToDevice, s);
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : 3;
}

extern "C" int test_v41_output(int m, void *aCode, float *aSf, void *bCode, float *bSf, void *codes, void *scales,
                               void *out) {
    auto fill = [](Data &w, int rows, int cols, void *bytes, const float *sf) {
        w.blockM = w.blockK = 32;
        w.Allocate();
        std::memcpy(w.cpuData, bytes, size_t(rows) * cols);
        w.scales.assign(sf, sf + size_t(rows / 32) * (cols / 32));
        w.ToDevice(CUDA, {0}, true);
    };
    Data a(FP8_E4M3, {8192, 4096}), b(FP8_E4M3, {5120, 8192}), y(BFLOAT16, {m, 5120});
    fill(a, 8192, 4096, aCode, aSf);
    fill(b, 5120, 8192, bCode, bSf);
    y.ToDevice(CUDA, {0}, false);
    y.Allocate(false);
    const char *order = std::getenv("FASTLLM_DSV41_MLA_HEAD_MAJOR");
    auto weight = FastllmCudaV41PackOutputWeight(a, order && std::strcmp(order, "1") == 0);
    if (!weight)
        return 1;
    auto p = FastllmCudaV41PlanOutput(weight, b, y, m, (m + 3) / 4 * 4, codes, scales);
    if (!p)
        return 2;
    if (!FastllmCudaV41RunOutput(p))
        return 3;
    return int(cudaMemcpyAsync(out, y.cudaData, size_t(m) * 5120 * 2, cudaMemcpyDeviceToDevice,
                               cudaStreamPerThread) != cudaSuccess ||
               cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess);
}
