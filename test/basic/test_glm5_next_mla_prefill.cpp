#include "../../src/models/glm5_next_mla_prefill.h"
#include "executor.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fastllm;
namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
uint16_t Bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float Float(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
void Upload(Data &x, const std::vector<int> &dims,
            const std::vector<uint16_t> &values) {
    x.dataType = BFLOAT16;
    x.UpdateUnitSize();
    x.dataDevice = DataDevice::CUDA;
    x.dataDeviceIds = {0};
    x.Resize(dims);
    x.Allocate();
    Require(x.Count(0) == values.size(), "fixture shape mismatch");
    FastllmCudaCopyFromHostToDevice(x.cudaData, (void *)values.data(), values.size() * 2);
}
struct Cache : Data {
    ~Cache() { isPagedKVCache = false; }
};
float Latent(int t, int d) {
    return float((t * 7 + d * 11 + (t / 13) * d) % 31 - 15) / 16;
}
float Projected(int h, int t, int d, bool value) {
    const int a = (d * 7 + h * 11) % 512;
    const int b = (d * 13 + h * 17 + 1) % 512;
    return Float(Bf16(Latent(t, a) * (value ? -.125f : .5f) +
                      Latent(t, b) * (value ? .75f : .25f)));
}
float Query(int h, int r, int d, bool uniform) {
    return uniform ? 0 : float((h * 7 + r * 11 + d * 13) % 17 - 8) / 16;
}
void Run(int heads, int tokens, int rows, int headBudget, bool fragmented,
         bool uniform) {
    constexpr int dim = 256, rank = 512, pageLen = 32;
    const int pages = (tokens + pageLen - 1) / pageLen;
    const int capacity = fragmented ? pages * 2 + 3 : pages;
    std::vector<int> ids(pages);
    for (int i = 0; i < pages; ++i)
        ids[i] = fragmented ? (pages - 1 - i) * 2 + 1 : i;
    std::vector<uint16_t> backing(capacity * pageLen * rank, 0x7fc0);
    for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < rank; ++d)
            backing[(ids[t / pageLen] * pageLen + t % pageLen) * rank + d] =
                Bf16(Latent(t, d));
    PagedCacheManager pool;
    Upload(pool, {capacity, pageLen, 1, rank}, backing);
    Cache cache;
    cache.dataType = BFLOAT16;
    cache.UpdateUnitSize();
    cache.dataDevice = DataDevice::CUDA;
    cache.dataDeviceIds = {0};
    cache.Resize({1, tokens, rank});
    cache.isPagedKVCache = true;
    cache.pagedKVCacheData = &pool;
    cache.pageLen = pageLen;
    cache.lastPageLen = (tokens - 1) % pageLen + 1;
    cache.pageIndex = ids;
    Data q, wk, wv, out;
    std::vector<uint16_t> qv(heads * rows * dim);
    std::vector<uint16_t> kw(heads * dim * rank, 0), vw(kw.size(), 0);
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < dim; ++d) {
            const size_t offset = size_t(h * dim + d) * rank;
            const int a = (d * 7 + h * 11) % rank;
            const int b = (d * 13 + h * 17 + 1) % rank;
            kw[offset + a] = Bf16(.5f);
            kw[offset + b] = Bf16(Float(kw[offset + b]) + .25f);
            vw[offset + a] = Bf16(-.125f);
            vw[offset + b] = Bf16(Float(vw[offset + b]) + .75f);
        }
        for (int r = 0; r < rows; ++r)
            for (int d = 0; d < dim; ++d)
                qv[(h * rows + r) * dim + d] = Bf16(Query(h, r, d, uniform));
    }
    Upload(q, {heads, rows, dim}, qv);
    Upload(wk, {heads, dim, rank}, kw);
    Upload(wv, {heads, dim, rank}, vw);
    const size_t perHead = size_t(pages) * pageLen * dim * 2;
    Require(glm5_next_detail::TryMhaPrefill(q, cache, wk, wv, out, 1.f / 16,
                                           2 * perHead * headBudget), "prefill rejected");
    Require(cache.pageIndex == ids && cache.pagedKVCacheData == &pool,
            "prefill changed the persistent cache");
    std::vector<uint16_t> result(qv.size());
    FastllmCudaCopyFromDeviceToHost(result.data(), out.cudaData, result.size() * 2);
    std::vector<uint16_t> cacheAfter(backing.size());
    FastllmCudaCopyFromDeviceToHost(cacheAfter.data(), pool.cudaData, cacheAfter.size() * 2);
    Require(cacheAfter == backing, "prefill modified persistent cache contents");
    double maxError = 0, squared = 0;
    for (int h = 0; h < heads; ++h) {
        for (int r = 0; r < rows; ++r) {
            const int visible = tokens - rows + r + 1;
            std::vector<double> scores(visible);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < visible; ++t) {
                double dot = 0;
                for (int d = 0; d < dim; ++d)
                    dot += Query(h, r, d, uniform) * Projected(h, t, d, false);
                scores[t] = dot / 16;
                maximum = std::max(maximum, scores[t]);
            }
            double denominator = 0;
            for (double &s : scores) {
                s = std::exp(s - maximum);
                denominator += s;
            }
            for (int d = 0; d < dim; ++d) {
                double expected = 0;
                for (int t = 0; t < visible; ++t)
                    expected += scores[t] * Projected(h, t, d, true);
                expected /= denominator;
                const float actual = Float(result[(h * rows + r) * dim + d]);
                Require(std::isfinite(actual), "nonfinite attention output");
                const double error = actual - expected;
                maxError = std::max(maxError, std::abs(error));
                squared += error * error;
            }
        }
    }
    const double rmse = std::sqrt(squared / result.size());
    std::printf("heads=%d kv=%d q=%d head_budget=%d fragmented=%d uniform=%d max=%.7f rmse=%.7f\n",
                heads, tokens, rows, headBudget, fragmented, uniform, maxError, rmse);
    Require(maxError < .008 && rmse < .002, "independent reference mismatch");
    Data rejected;
    Require(!glm5_next_detail::TryMhaPrefill(q, cache, wk, wv, rejected, 1.f / 16, 0),
            "zero workspace must fall back");
    Require(rejected.dims.empty(), "fallback modified output");
    Data decode;
    decode.FakeFrom(q, 0);
    decode.Resize({heads, 1, dim});
    Require(!glm5_next_detail::TryMhaPrefill(decode, cache, wk, wv, rejected, 1.f / 16),
            "decode must fall back");
}
} // namespace
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
    if (prop.major < 8) return 77;
    FastllmCudaSetDevice(0);
    if (!FastllmCudaFlashInferSupported()) return 77;
    static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:0");
    try {
        Run(3, 64, 64, 4, false, false);
        Run(5, 73, 17, 2, true, false);
        Run(3, 129, 65, 1, true, true);
        Run(4, 193, 128, 2, false, false);
        Require(cudaDeviceSynchronize() == cudaSuccess, "CUDA execution failed");
        std::puts("PASS GLM MLA prefill");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
