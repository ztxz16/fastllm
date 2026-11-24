// Independent softmax reference for paged/grouped Inkling attention with a
// learned sink, absolute-position temperature and an uncommitted prefill suffix.
#include "../../src/models/inkling2/runtime.cuh"
#include "attention/variants.cuh"
#include "attention/decode.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/prefill.cuh"
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
#include "../../src/models/inkling2/prefill.cuh"
#include "../../src/models/inkling2/paged-verify-attention.cuh"

struct DisabledPrefillHooks {
    static constexpr bool custom_output_store = false;
    static constexpr bool custom_score_tile = false;
    static constexpr bool defer_softmax_rescale = false;
};
static_assert(!flashinfer::prefill_custom_output_store<DisabledPrefillHooks>::value &&
              !flashinfer::prefill_custom_score_tile<DisabledPrefillHooks>::value &&
              !flashinfer::prefill_defer_softmax_rescale<DisabledPrefillHooks>::value);

template<class T>
Buffer<T> upload(const std::vector<T> &v) {
    Buffer<T> gpu(v.size());
    if (!v.empty()) check(cudaMemcpy(gpu.p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
    return gpu;
}

// Two requests have different page mappings. End one chain early and compare
// every verification row (including padding) to a standalone decode launch.
template<int D, class T>
void compareVerifyChain(AttentionRow a, PageView pages, std::vector<int2> ids, int past,
                        int chain, int splits, int sm120Splits) {
    const int rows = 2 * chain, width = a.heads * D;
    std::reverse(ids.begin(), ids.end());
    auto otherPages = upload(ids);
    std::vector<DecodeVisit> descriptors(rows);
    for (int r = 0; r < rows; ++r) {
        auto &v = descriptors[r];
        v.pages = pages;
        v.past = past + r % chain;
        v.verifyRemaining = chain - r % chain;
        if (r >= chain) {
            v.pages.ids = otherPages.p;
            v.verifyRemaining = chain / 2 + 1 - r % chain;
            v.past = past + 1 + std::min(r % chain, chain / 2);
        }
    }
    auto visits = upload(descriptors);
    a.visit = visits.p;
    Matrix batchScratch(splits, rows * a.heads * (D + 1)), singleScratch(splits, a.heads * (D + 1));
    Matrix batched(rows, width), single(rows, width);
    // Gates and sinks share a packed projection row in the production graph.
    // The standalone launches below use the original contiguous reference inputs.
    a.sinkStride = a.gateStride = width + a.heads + 3;
    Matrix packedGates(rows, a.gateStride);
    float *gate = packedGates.p, *sink = gate + width;
    check(cudaMemcpy2DAsync(gate, a.gateStride * sizeof(float), a.q, width * sizeof(float),
                            width * sizeof(float), rows, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    check(cudaMemcpy2DAsync(sink, a.sinkStride * sizeof(float), a.sink, a.heads * sizeof(float),
                            a.heads * sizeof(float), rows, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    launchSplitRelative<D,T>(a, batchScratch, sink, gate, batched.p, sm120Splits, rows, chain);
    for (int r = 0; r < rows; ++r) {
        AttentionRow one = a;
        one.visit += r;
        one.q += r * width;
        one.relative += r * a.heads * a.distances;
        launchSplitRelative<D,T>(one, singleScratch, a.sink + r * a.heads, one.q, single.p + r * width, sm120Splits);
    }
    std::vector<float> actual(rows * width), expected(actual.size());
    check(cudaMemcpy(actual.data(), batched.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(expected.data(), single.p, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)))
        throw std::runtime_error("Verification attention differs from standalone decode");
    if constexpr (D == 128 && std::is_same_v<T, half>) {
        if (!a.window && configurePrefillKernel<VerifyTraits, VerifyPrefillParams>(pagedVerifyAttention)) {
            int device, smCount;
            check(cudaGetDevice(&device));
            check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
            const float *referenceGate = a.q;
            std::vector<float> query(size_t(rows) * width);
            check(cudaMemcpy(query.data(), a.q, query.size() * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<half> packed(query.size() * 2);
            for (size_t j = 0; j < query.size(); ++j) {
                const size_t at = j / D * (2 * D) + j % D;
                packed[at] = half(query[j]);
                packed[at + D] = half(query[j] - float(packed[at]));
            }
            auto q = upload(packed);
            a.q = reinterpret_cast<const float *>(q.p);
            launchVerifyAttention(a, batchScratch, sink, gate, batched.p, rows, chain, splits, smCount, VerifyPartition::minTiles);
            for (int r = 0; r < rows; ++r) {
                auto one = a;
                one.q += size_t(r) * width;
                one.relative += size_t(r) * a.heads * a.distances;
                one.visit += r;
                launchVerifyAttention(one, singleScratch, a.sink + size_t(r) * a.heads, referenceGate + size_t(r) * width,
                                      single.p + size_t(r) * width, 1, 1, splits, smCount, VerifyPartition::minTiles);
            }
            const auto baseline = expected;
            check(cudaMemcpy(actual.data(), batched.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(expected.data(), single.p, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double diff = 0., norm = 0.;
            for (size_t j = 0; j < actual.size(); ++j) {
                const bool live = descriptors[j / width].verifyRemaining > 0;
                if (actual[j] != (live ? expected[j] : 0.f))
                    throw std::runtime_error("Paged verification differs from singleton attention");
                if (live) {
                    diff += std::pow(double(actual[j]) - baseline[j], 2);
                    norm += std::pow(double(baseline[j]), 2);
                }
            }
            if (!(diff <= norm * 1e-8)) throw std::runtime_error("Paged attention differs from the decode reference");
        }
    }
    printf("verification attention exact: window=%d past=%d chain=%d requests=2\n", a.window, past, chain);
}

template<int D, class T, class Gate = void, class Bias = float>
void compare(int group, int past, int rows, int window, int pageLen, bool committed, int splits = 28, bool deviceVisit = false, int sm120Splits = 0, bool prefill = false, int chain = 0) {
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
    if (chain) {
        if (rows != 2 * chain || !committed) throw std::logic_error("Invalid verification fixture");
        return compareVerifyChain<D,T>(a, p, ids, past, chain, splits, sm120Splits);
    }
    std::vector<float> result(rows * heads * D);
    std::vector<float> compactGate;
    const int gateStride = heads * D + 13;
    if constexpr (!std::is_void_v<Gate>) {
        static_assert(D == 128 && std::is_same_v<T, half>);
        if (!(window ? configureGatedPrefill<true, Gate, Bias>() : configureGatedPrefill<false, Gate, Bias>())) return;
        std::vector<half> query(q.begin(), q.end());
        compactGate.resize(rows * gateStride);
        fill(compactGate);
        std::vector<Gate> typedGate(compactGate.begin(), compactGate.end());
        Buffer<Bias> typedBias;
        if constexpr (std::is_same_v<Bias, float>) typedBias = dr;
        else typedBias = upload(std::vector<Bias>(rel.begin(), rel.end()));
        auto fused = upload(query);
        auto gating = upload(typedGate);
        for (int first = 0; first < rows; first += 137) {
            a.first = first;
            half *input = fused.p + size_t(first) * heads * D;
            auto launch = window ? launchGatedPrefill<true, Gate, Bias> : launchGatedPrefill<false, Gate, Bias>;
            launch(a, input, reinterpret_cast<Gate *>(input), gating.p + size_t(first) * gateStride,
                   gateStride, std::min(137, rows - first), typedBias.p + size_t(first) * heads * extent);
        }
        std::vector<Gate> values(q.size());
        check(cudaMemcpy(values.data(), fused.p, values.size() * sizeof(Gate), cudaMemcpyDeviceToHost));
        std::copy(values.begin(), values.end(), result.begin());
    } else if (prefill) {
        if constexpr (!std::is_same_v<T, float>) {
            if (!configurePrefill<D,T>()) throw std::runtime_error("Insufficient prefill shared memory");
            std::vector<T> packed(q.size() * 2);
            for (size_t i = 0; i < q.size(); ++i) {
                const size_t index = (i / D) * (2 * D) + i % D;
                packed[index] = T(q[i]);
                packed[index + D] = T(q[i] - float(packed[index]));
            }
            auto typed = upload(packed);
            // A second query tile must see the first tile as part of its suffix.
            for (int first = 0; first < rows; first += 35) {
                a.first = first;
                a.relative = dr.p + size_t(first) * heads * extent;
                launchPrefill<D,T>(a, typed.p + size_t(first) * heads * (2 * D), std::min(35, rows - first));
            }
        }
    } else if (split)
        launchSplitRelative<D,T>(a, scratch, ds.p, dg.p, output.p, sm120Splits);
    else
        launchRelative<D,T>(a, rows, split, scratch);
    if constexpr (std::is_void_v<Gate>)
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
                double bias = pos - t < extent ? float(Bias(rel[(r * heads + h) * extent + pos - t])) : 0.;
                double weight = std::exp((dot / std::sqrt(double(D)) + bias) * tau);
                denom += weight;
                for (int j = 0; j < D; ++j) sum[j] += weight * float(v[i + j]);
            }
            for (int j = 0; j < D; ++j) {
                double ref = sum[j] / denom;
                if (split) ref /= 1. + std::exp(-double(gate[h * D + j]));
                if constexpr (!std::is_void_v<Gate>)
                    ref /= 1. + std::exp(-double(float(Gate(compactGate[r * gateStride + h * D + j]))));
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
    // The fused operator returns the model activation type. Compare that stored
    // output, including its rounding, against the independently gated FP64 result.
    const double l2Tolerance = std::is_same_v<Gate, half> ? 5e-4 :
                               std::is_same_v<Gate, nv_bfloat16> ? 2e-3 : prefill ? 1e-5 : sm120Splits ? 3e-6 : 1e-4;
    const double absTolerance = std::is_same_v<Gate, half> ? 5e-4 :
                                std::is_same_v<Gate, nv_bfloat16> ? 2e-3 : prefill ? 2e-6 : sm120Splits ? 1e-6 : 1e-5;
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
    int major;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu));
    // Each gated case checks its own compiled kernel and shared-memory limit.
    compare<128, half, half>(3, 0, 513, 0, 128, false);
    compare<128, half, half>(5, 8190, 39, 0, 127, false);
    compare<128, half, half>(1, 0, 149, 1, 37, false);
    compare<128, half, half>(3, 1017, 513, 1024, 31, false);
    compare<128, half, half>(5, 8190, 145, 97, 37, false);
    if (major >= 8) {
        compare<128, half, nv_bfloat16>(3, 8190, 39, 0, 127, false);
        compare<128, half, nv_bfloat16>(3, 1017, 159, 1024, 31, false);
    }
    if (major >= 8 && configurePrefill<128, half>() && configurePrefill<128, nv_bfloat16>()) {
        for (int group : {1, 3, 5}) {
            compare<128, half, half>(group, 125, 151, 0, 37, false);
            compare<128, half, nv_bfloat16>(group, 0, 143, 0, 128, false);
        }
        compare<128, half, half>(3, 8190, 19, 0, 127, false);
        compare<128, half, nv_bfloat16, nv_bfloat16>(3, 8173, 157, 1024, 31, false);
        compare<128, half>(3, 65531, 5, 0, 127, false, 1, false, 0, true);
        compare<128, half>(3, 8189, 17, 1024, 128, false, 1, false, 0, true);
        for (int window : {0, 1024}) {
            compare<128, half>(3, 0, 71, window, 37, false, 1, false, 0, true);
            compare<128, nv_bfloat16>(3, 1051, 71, window, 31, false, 1, false, 0, true);
            compare<64, half>(2, 2050, 39, window, 128, false, 1, false, 0, true);
        }
    }
    const bool sm120 = supportsGlobalRelativeSm120(gpu);
    for (int chain : {3, 7, 8}) {
        // Cross a partition-size boundary, then the point where the window starts sliding.
        for (int past : {446, 1019})
            compare<128, half>(3, past, 2 * chain, 1024, 37, true, 28, true, 0, false, chain);
        for (int past : {14, 399, 4093})
            compare<128, half>(3, past, 2 * chain, 0, 37, true, 28, true, sm120 ? 26 : 0, false, chain);
    }
    // Span different split grains and more than one query tile, including clipped padding.
    compare<128, half>(3, 28 * 128 - 4, 46, 0, 37, true, 28, true, sm120 ? 26 : 0, false, 23);
    if (sm120) {
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
