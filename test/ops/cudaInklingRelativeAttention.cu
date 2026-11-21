// Independent softmax reference for paged/grouped Inkling attention with a
// learned sink, absolute-position temperature and an uncommitted prefill suffix.
#include "../../src/models/inkling2/runtime.cuh"
#include "attention/variants.cuh"
#include "attention/decode.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace fastllm::inkling2 {
namespace {
#include "../../src/models/inkling2/paged-kv.cuh"
#include "../../src/models/inkling2/attention.cuh"

template<class T>
Buffer<T> upload(const std::vector<T> &v) {
    Buffer<T> gpu(v.size());
    if (!v.empty()) check(cudaMemcpy(gpu.p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return gpu;
}

template<int D, class T>
void compare(int group, int past, int rows, int window, int pageLen, bool committed, int splits = 28, bool deviceVisit = false, int sm120Splits = 0) {
    const int kv = 2, heads = group * kv, stride = D * kv, extent = 125;
    const int begin = window ? std::max(0, past - window) : 0;
    const int firstPage = begin / pageLen, end = past + rows;
    const int pageCount = (end - 1) / pageLen - firstPage + 1;
    std::mt19937 rng(923 + past);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    auto fill = [&](auto &v) { for (auto &x : v) x = uniform(rng); };
    std::vector<T> k(end * stride), v(end * stride), kp(pageCount * pageLen * stride), vp(kp.size());
    std::vector<int2> ids(pageCount);
    fill(k); fill(v);
    for (int page = 0; page < pageCount; ++page)
        ids[page] = make_int2(pageCount - page - 1, (page + 2) % pageCount);
    for (int t = begin; t < (committed ? end : past); ++t) {
        int2 id = ids[t / pageLen - firstPage];
        std::copy_n(k.data() + t * stride, stride, kp.data() + (id.x * pageLen + t % pageLen) * stride);
        std::copy_n(v.data() + t * stride, stride, vp.data() + (id.y * pageLen + t % pageLen) * stride);
    }
    std::vector<float> q(rows * heads * D), rel(rows * heads * extent), sink(rows * heads), gate(heads * D);
    fill(q); fill(rel); fill(sink); fill(gate);
    auto dk = upload(k), dv = upload(v), dkp = upload(kp), dvp = upload(vp);
    auto di = upload(ids);
    auto dq = upload(q), dr = upload(rel), ds = upload(sink), dg = upload(gate);
    Matrix output(rows, heads * D), scratch(splits, heads * (D + 1));
    PageView p{dkp.p, dvp.p, di.p, firstPage, pageLen, pageShift(pageLen)};
    DecodeVisit visit{};
    visit.past = past;
    visit.pages = p;
    auto descriptor = upload(std::vector<DecodeVisit>{visit});
    bool split = rows == 1;
    AttentionRow a{dq.p, committed ? nullptr : dk.p + past * stride, committed ? nullptr : dv.p + past * stride,
                   p, dr.p, ds.p, output.p, (committed || deviceVisit) ? descriptor.p : nullptr,
                   past, 0, heads, kv, D, extent, window, 16, 0.3f};
    if (split)
        launchSplitRelative<D,T>(a, scratch, ds.p, dg.p, output.p, sm120Splits);
    else
        launchRelative<D,T>(a, rows, split, scratch);
    std::vector<float> result(rows * heads * D);
    check(cudaMemcpy(result.data(), output.p, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (split && !sm120Splits) {
        if (group == 3) launchRelativeKernel<D,3,true,T>(a, 1, scratch.p, splits);
        else launchRelativeKernel<D,1,true,T>(a, 1, scratch.p, splits);
        mergeRelative<D><<<heads,D,splits*sizeof(float),cudaStreamPerThread>>>(a, scratch.p, splits, ds.p, dg.p, output.p);
        std::vector<float> generic(result.size());
        check(cudaMemcpy(generic.data(), output.p, generic.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(generic.data(), result.data(), result.size() * sizeof(float)))
            throw std::runtime_error("Split attention differs from the generic FP32 accumulation");
    }
    double diff = 0, norm = 0, maxAbs = 0;
    for (int r = 0; r < rows; ++r) {
        const int pos = past + r, first = window ? std::max(0, pos + 1 - window) : 0;
        const double tau = window ? 1. : 1. + a.alpha * std::log(std::max(1., (pos + 1.) / a.floor));
        for (int h = 0; h < heads; ++h) {
            int base = (r * heads + h) * D;
            double denom = std::exp(double(sink[r * heads + h]));
            std::array<double, D> sum{};
            for (int t = first; t <= pos; ++t) {
                int i = t * stride + (h / group) * D;
                double dot = 0;
                for (int j = 0; j < D; ++j) dot += q[base + j] * double(float(k[i + j]));
                double bias = pos - t < extent ? rel[(r * heads + h) * extent + pos - t] : 0.;
                double weight = std::exp((dot / std::sqrt(double(D)) + bias) * tau);
                denom += weight;
                for (int j = 0; j < D; ++j) sum[j] += weight * float(v[i + j]);
            }
            for (int j = 0; j < D; ++j) {
                double ref = sum[j] / denom;
                if (split) ref /= 1. + std::exp(-double(gate[h * D + j]));
                double delta = result[base + j] - ref;
                diff += delta * delta;
                norm += ref * ref;
                maxAbs = std::max(maxAbs, std::abs(delta));
            }
        }
    }
    const double error = std::sqrt(diff / norm);
    printf("D=%d group=%d past=%d rows=%d W=%d page=%d committed=%d sm120Splits=%d relL2=%.3g maxAbs=%.3g\n",
           D, group, past, rows, window, pageLen, committed, sm120Splits, error, maxAbs);
    const double l2Tolerance = sm120Splits ? 3e-6 : 1e-4;
    const double absTolerance = sm120Splits ? 1e-6 : 1e-5;
    if (!(error < l2Tolerance && maxAbs < absTolerance)) throw std::runtime_error("Relative attention mismatch");
}
} // namespace
int testRelative() {
    for (int past : {0, 16, 1023, 1024, 1025, 4096}) {
        compare<128, nv_bfloat16>(3, past, 1, 1024, 128, true);
        compare<128, nv_bfloat16>(3, past, 1, 0, 128, true);
    }
    compare<64, half>(2, 2047, 1, 1001, 63, true);
    compare<64, half>(1, 31, 1, 0, 16, false);
    compare<128, nv_bfloat16>(3, 2047, 9, 1024, 128, false);
    compare<64, half>(2, 2047, 9, 0, 63, false);
    compare<128, float>(3, 1024, 1, 0, 127, true);
    compare<64, half>(2, 1031, 1, 1001, 63, true, 33);
    compare<128, half>(3, 4101, 1, 0, 127, true, 32);
    compare<64, half>(2, 31, 1, 0, 16, false, 1, true);
    // Exercise both sides of the single-tile bound, including split tails
    // that occupy a partial warp and independent non-power-of-two KV pages.
    for (int window : {143, 144, 145})
        compare<64, float>(3, 197, 1, window, 23, true, 3);
    compare<128, half>(1, 91, 1, 59, 17, true, 4);
    // Oversized windows share the streaming kernel with global attention.
    compare<128, nv_bfloat16>(3, 2057, 1, 1025, 61, true, 7);
    compare<64, half>(2, 3001, 1, 2047, 63, true, 4);
    // Test global splits one row below, at and above a two-pass KV stripe.
    // FP16 fits 30 rows with three query heads at D=128, and 64 at D=64/G=1.
    for (int past : {86, 89, 90})
        compare<128, half>(3, past, 1, 0, 17, true, 3);
    for (int past : {188, 191, 192})
        compare<64, half>(2, past, 1, 0, 23, true, 3);
    compare<128, half>(3, 8190, 1, 0, 127, true, 35);
    compare<64, float>(1, 4099, 1, 0, 63, true, 21);
    int gpu;
    check(cudaGetDevice(&gpu));
    if (supportsGlobalRelativeSm120(gpu)) {
        // Partial warp tiles, a partially filled last split, odd page sizes,
        // long contexts, and independently permuted K/V.
        for (int past : {0, 1, 14, 15, 16, 62, 63, 64, 1023, 4095, 4100, 8191, 8317})
            compare<128, half>(3, past, 1, 0, 37, true, 28, false, 26);
        for (int pageLen : {48, 128})
            compare<128, half>(3, 4098, 1, 0, pageLen, true, 28, false, 26);
    }
    return 0;
}
} // namespace fastllm::inkling2
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    return fastllm::inkling2::testRelative();
}
