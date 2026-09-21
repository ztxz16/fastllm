#include "fastllm.h"
#include "devices/cuda/attention/fastllm-paged-attention-native.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;
namespace {
constexpr int heads = 12, kvHeads = 2, dim = 256, pageLen = 128, group = 6;
void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void Require(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
float Half(float x) {
    return half_to_float(float_to_half(x));
}
float Key(int h, int t, int d) {
    return Half(float((t % 31 * 13 + d * 17 + h * 19) % 31 - 15) / 32);
}
float Value(int h, int t, int d) {
    return Half(.25f + float((t % 61 * 7 + d * 11 + h * 23) % 61 - 30) / 256);
}
float Query(int h, int r, int d) {
    return Half(float((h * 23 + r * 13 + d * 37) % 127 - 63) / 64);
}
void Init(Data &a, const std::vector<int> &shape) {
    a.dataDevice = DataDevice::CUDA;
    a.dataDeviceIds = {0};
    a.Resize(shape);
    a.Allocate();
}
void Ints(Data &a, const std::vector<int> &v) {
    Init(a, {int(v.size())});
    a.cpuIntDatas = v;
    Check(cudaMemcpy(a.cudaData, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
}
struct CacheView : Data {
    CacheView(PagedCacheManager &pool, int length, const std::vector<int> &pages) : Data(FLOAT16) {
        Resize({kvHeads, length, dim});
        isPagedKVCache = true;
        pagedKVCacheData = &pool;
        pageLen = ::pageLen;
        pageIndex = pages;
        lastPageLen = (length - 1) % pageLen + 1;
    }
    ~CacheView() { isPagedKVCache = false; } // Synthetic pages have no allocator refs.
};

void Run(int length, int rows, int layout, bool tail) {
    int pages = (length + pageLen - 1) / pageLen;
    std::vector<int> ids(pages);
    std::iota(ids.begin(), ids.end(), 0);
    if (layout == 1)
        std::reverse(ids.begin(), ids.end());
    if (layout == 2)
        for (int i = 0; i < pages; ++i)
            ids[i] += i * 9 / pages;
    if (layout >= 3) {
        // Logical fragmentation can still coalesce physically (layout 3).
        // Isolated physical pages (layout 4) must retain the gather fallback.
        if (layout == 4)
            for (int &p : ids)
                p *= 2;
        std::mt19937 random(1234);
        std::shuffle(ids.begin(), ids.end(), random);
    }
    int capacity = *std::max_element(ids.begin(), ids.end()) + 1;
    PagedCacheManager kp, vp;
    kp.dataType = vp.dataType = FLOAT16;
    kp.UpdateUnitSize();
    vp.UpdateUnitSize();
    Init(kp, {capacity, pageLen, kvHeads, dim});
    Init(vp, kp.dims);
    std::vector<uint16_t> buffer(kp.Count(0), float_to_half(std::numeric_limits<float>::quiet_NaN()));
    for (int which = 0; which < 2; ++which) {
        // Materialize each input period once, keeping long tests dominated
        // by the attention operation rather than repeated host conversion.
        int period = which == 0 ? 31 : 61;
        std::vector<uint16_t> pattern(period * kvHeads * dim);
        for (int t = 0; t < period; ++t)
            for (int h = 0; h < kvHeads; ++h)
                for (int d = 0; d < dim; ++d) {
                    pattern[(t * kvHeads + h) * dim + d] =
                        float_to_half(which == 0 ? Key(h, t, d) : Value(h, t, d));
                }
        for (int t = 0; t < length; ++t) {
            auto *dst = buffer.data() + (size_t(ids[t / pageLen]) * pageLen + t % pageLen) * kvHeads * dim;
            if (!tail) {
                std::copy_n(pattern.data() + t % period * kvHeads * dim, kvHeads * dim, dst);
            } else {
                for (int h = 0; h < kvHeads; ++h) {
                    float v = which == 1 && t >= length - rows
                                  ? float((t - (length - rows) + 1) * 4000 + h * 500)
                                  : 0.f;
                    std::fill_n(dst + h * dim, dim, float_to_half(v));
                }
            }
        }
        Data &pool = which == 0 ? static_cast<Data &>(kp) : static_cast<Data &>(vp);
        Check(cudaMemcpy(pool.cudaData, buffer.data(), pool.GetBytes(), cudaMemcpyHostToDevice));
    }
    buffer.clear();
    buffer.shrink_to_fit();
    CacheView k(kp, length, ids), v(vp, length, ids);
    Data q(FLOAT16), out(FLOAT16), qs(INT32), ps(INT32), pi(INT32), ls(INT32);
    Init(q, {heads, rows, dim});
    Init(out, {rows, heads, dim});
    std::vector<uint16_t> queries(q.Count(0));
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < rows; ++r)
            for (int d = 0; d < dim; ++d)
                queries[(h * rows + r) * dim + d] = float_to_half(tail ? 0.f : Query(h, r, d));
    Check(cudaMemcpy(q.cudaData, queries.data(), q.GetBytes(), cudaMemcpyHostToDevice));
    Ints(qs, {0, rows});
    Ints(ps, {0, pages});
    Ints(pi, ids);
    Ints(ls, {k.lastPageLen});
    Require(FastllmCudaHalfPagedAttentionBatchFastllmFallback(q, k, v, qs, ps, pi, ls, out, group, 1.f / 16),
            "attention failed");
    Check(cudaDeviceSynchronize());
    std::vector<uint16_t> actual(out.Count(0));
    Check(cudaMemcpy(actual.data(), out.cudaData, out.GetBytes(), cudaMemcpyDeviceToHost));
    double maxError = 0;
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h) {
            const int visible = length - rows + r + 1;
            double weights[61] = {}, denominator = 0;
            if (!tail) {
                double scores[31], maximum = -std::numeric_limits<double>::infinity();
                for (int t = 0; t < 31; ++t) {
                    scores[t] = 0;
                    for (int d = 0; d < dim; ++d)
                        scores[t] += double(Query(h, r, d)) * Key(h / group, t, d) / 16;
                    maximum = std::max(maximum, scores[t]);
                }
                // Independent double reference, using only the input periods to
                // count repeated keys/values; no dependence on the page planner.
                for (int t = 0; t < 31 * 61; ++t) {
                    int count = visible / (31 * 61) + (t < visible % (31 * 61));
                    double weight = count * std::exp(scores[t % 31] - maximum);
                    weights[t % 61] += weight;
                    denominator += weight;
                }
            }
            for (int d = 0; d < dim; ++d) {
                double expected = 0;
                if (tail) {
                    for (int i = 0; i <= r; ++i)
                        expected += Half(float((i + 1) * 4000 + h / group * 500));
                    expected /= visible;
                } else {
                    for (int t = 0; t < 61; ++t)
                        expected += weights[t] * Value(h / group, t, d);
                    expected /= denominator;
                }
                double value = half_to_float(actual[(r * heads + h) * dim + d]);
                double error = std::abs(value - expected);
                maxError = std::max(maxError, error);
                if (!std::isfinite(value) || error > .0015 + .003 * std::abs(expected)) {
                    std::fprintf(
                        stderr,
                        "FAIL len=%d rows=%d layout=%d tail=%d r=%d h=%d d=%d actual=%.9g expected=%.9g\n",
                        length, rows, layout, tail, r, h, d, value, expected);
                    throw std::runtime_error("paged attention differs from independent reference");
                }
            }
        }
    std::printf("PASS len=%d rows=%d layout=%d tail=%d max_abs=%.9g\n", length, rows, layout, tail, maxError);
    std::fflush(stdout);
}
} // namespace
int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return 77;
    Check(cudaSetDevice(0));
    // Exercise SM70's direct-KV path even when the test GPU is newer.
    setenv("FASTLLM_PAGED_CUBLAS_LINEAR_KV", "1", 1);
    try {
        bool longTest = argc > 1 && std::string(argv[1]) == "--long";
        for (int length : longTest ? std::vector<int>{179908, 179969, 208897} : std::vector<int>{129, 4099})
            for (int rows : {2, 4, 8})
                for (int layout = 0; layout < 5; ++layout)
                    for (bool tail : {false, true})
                        Run(length, rows, layout, tail);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
