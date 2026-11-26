// Independent softmax reference for paged/grouped Inkling attention with a
// learned sink, absolute-position temperature and an uncommitted prefill suffix.
#include "../../src/models/inkling2/runtime.cuh"
#include "devices/cuda/fastllm-cuda-launch.cuh"
#include "devices/cuda/fastllm-kvmem-page-selection.cuh"
#include "devices/cuda/fastllm-cuda-attn-res.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-bf16-warp-dot.cuh"
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
    constexpr bool fp8KV = std::is_same_v<T, __nv_fp8_e4m3>;
    if constexpr (D == 128 && (std::is_same_v<T, half> || fp8KV)) {
        if ((fp8KV || !a.window) && configurePrefillKernel<VerifyTraits, VerifyPrefillParams>(pagedVerifyAttention<VerifyTraits>)) {
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
            if constexpr (fp8KV) {
#if CUDART_VERSION >= 12080
                if (supportsGlobalRelativeSm120(device)) {
                    launchVerifyAttention<true>(a, batchScratch, sink, gate, batched.p, rows, chain, splits,
                                                smCount, VerifyPartition::minTiles, false, true);
                    check(cudaMemcpy(actual.data(), batched.p, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
                    double diff = 0., norm = 0.;
                    for (size_t j = 0; j < actual.size(); ++j) {
                        if (descriptors[j / width].verifyRemaining <= 0) {
                            if (actual[j] != 0.f) throw std::runtime_error("FP8 padded verification row is nonzero");
                            continue;
                        }
                        diff += std::pow(double(actual[j]) - expected[j], 2);
                        norm += std::pow(double(expected[j]), 2);
                    }
                    if (!(diff <= norm * 1e-8)) {
                        fprintf(stderr, "FP8 tiled verification: window=%d past=%d chain=%d relL2=%.4g\n",
                                a.window, past, chain, std::sqrt(diff / norm));
                        throw std::runtime_error("FP8 tiled verification differs from decode");
                    }
                }
#endif
                return;
            }
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
#if CUDART_VERSION >= 12080
            if (supportsGlobalRelativeSm120(device)) {
                // Force the full-batch dispatch on this small two-request fixture.
                launchVerifyAttention(a, batchScratch, sink, gate, batched.p, rows, chain, splits,
                                      1, VerifyPartition::minTiles, false, true);
                std::vector<float> tiled(actual.size());
                check(cudaMemcpy(tiled.data(), batched.p, tiled.size() * sizeof(float), cudaMemcpyDeviceToHost));
                if (tiled != actual) throw std::runtime_error("Tiled global verification differs from standard paged attention");
            }
#endif
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
    auto fill = [&](auto &v) { for (auto &x : v) x = std::decay_t<decltype(x)>(uniform(rng)); };
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
        static_assert(D == 128 && sizeof(PrefillQuery<T>) == 2);
        if (!(window ? configureGatedPrefill<true, Gate, T, Bias>() : configureGatedPrefill<false, Gate, T, Bias>())) return;
        std::vector<PrefillQuery<T>> query(q.begin(), q.end());
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
            auto *input = fused.p + size_t(first) * heads * D;
            auto launch = window ? launchGatedPrefill<true, Gate, T, Bias> : launchGatedPrefill<false, Gate, T, Bias>;
            launch(a, input, reinterpret_cast<Gate *>(input), gating.p + size_t(first) * gateStride,
                   gateStride, std::min(137, rows - first), typedBias.p + size_t(first) * heads * extent);
        }
        std::vector<Gate> values(q.size());
        check(cudaMemcpy(values.data(), fused.p, values.size() * sizeof(Gate), cudaMemcpyDeviceToHost));
        std::copy(values.begin(), values.end(), result.begin());
    } else if (prefill) {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
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
// Two live requests cycle through independent device rings. Re-read every byte
// after eviction and after overwriting a rejected proposal on the next page.
template<class T> void checkMappedPages() {
    constexpr int len = 16, width = 8, count = 10, hot = 2, span = len * width;
    const size_t sideBytes = span * sizeof(T);
    const MappedPageCounts capacity{count, hot, count};
    MappedKvPages a(capacity, 2 * sideBytes), b(capacity, 2 * sideBytes);
    int request = 0;
    for (auto *storage : {&a, &b}) {
        PageView view{nullptr, nullptr, storage->addresses(), 0, len, pageShift(len)};
        std::vector<int2> initial(count);
        check(cudaMemcpyAsync(initial.data(), view.ids, count * sizeof(int2), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        for (int2 id : initial) {
            auto address = (uintptr_t(uint32_t(id.y)) << 32) | uint32_t(id.x);
            cudaPointerAttributes attr{};
            check(cudaPointerGetAttributes(&attr, reinterpret_cast<void *>(address)));
            if (attr.type != cudaMemoryTypeHost)
                throw std::runtime_error("KVMem host page missing at request admission");
        }
        std::vector<T> keys(count * span), values(keys.size());
        for (int i = 0; i < int(keys.size()); ++i) {
            keys[i] = T(float((i + request) % 19));
            values[i] = T(float((i + 3 * request) % 31));
        }
        auto key = upload(keys), value = upload(values);
        auto write = [&](int start, int rows) {
            appendKv<<<(rows * width + 127) / 128, 128, 0, cudaStreamPerThread>>>(
                view, key.p + start * width, value.p + start * width, start, 0, width, rows);
        };
        write(0, 2 * len + 3);
        storage->prepare(3, 1); // Reject the tail; only the first page is immutable.
        keys[2 * span] = T(77.f);
        check(cudaMemcpyAsync(key.p + 2 * span, &keys[2 * span], sizeof(T), cudaMemcpyHostToDevice, cudaStreamPerThread));
        write(2 * len, (count - 2) * len);
        for (int sealed : {2, 5, count - 1}) {
            storage->prepare(count, sealed);
            check(cudaStreamSynchronize(cudaStreamPerThread));
            std::vector<int2> table(count);
            check(cudaMemcpy(table.data(), view.ids, count * sizeof(int2), cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                const auto bits = uintptr_t(uint32_t(table[i].y)) << 32 | uint32_t(table[i].x);
                auto *ptr = reinterpret_cast<const T *>(bits);
                cudaPointerAttributes attr{};
                check(cudaPointerGetAttributes(&attr, ptr));
                const bool resident = i >= sealed - hot && i < sealed;
                if (attr.type != (resident ? cudaMemoryTypeDevice : cudaMemoryTypeHost))
                    throw std::runtime_error("KVMem exceeded the per-request device ring or lost a host page");
                std::vector<T> pair(2 * span);
                check(cudaMemcpy(pair.data(), ptr, 2 * sideBytes, cudaMemcpyDefault));
                if (memcmp(pair.data(), keys.data() + i * span, sideBytes) ||
                    memcmp(pair.data() + span, values.data() + i * span, sideBytes))
                    throw std::runtime_error("KVMem cold/hot page bytes differ from committed KV");
            }
        }
        ++request;
    }
}

// Absolute positions keep working across host and device ring wraps, including
// an unaligned prefill larger than the ring and overwrites after rejection.
__global__ void readMappedWindow(PageView view, const int *end, int window, int2 *out) {
    const int t = *end - window + threadIdx.x;
    if (t >= 0) out[threadIdx.x] = make_int2(*view.row<int>(t, false, 1), *view.row<int>(t, true, 1));
}

void checkSlidingHostPages() {
    constexpr int len = 16, window = 32, block = 8, pages = 32;
    Config model{};
    model.window = window;
    model.draft.block = block;
    KvMemConfig kv;
    kv.residentTokens = len;
    const auto capacity = model.mappedKvPages(kv, pages * len, len, false, true);
    MappedKvPages storage(capacity, 2 * len * sizeof(int));
    PageView view{nullptr, nullptr, storage.addresses(), 0, len, pageShift(len)};
    std::vector<int2> initial(pages);
    check(cudaMemcpyAsync(initial.data(), view.ids, pages * sizeof(int2), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    if (capacity.host != 4 || memcmp(&initial[0], &initial[capacity.host], sizeof(int2)))
        throw std::runtime_error("Sliding window did not reuse its host backing");
    Buffer<int> end(1);
    Buffer<int2> actual(window);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    readMappedWindow<<<1,window,0,cudaStreamPerThread>>>(view, end.p, window, actual.p);
    check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::vector<int> reference;
    int serial = 0;
    constexpr std::pair<int, int> steps[] = {{47, 47}, {8, 1}, {8, 0}, {8, 8}, {103, 103},
                                             {8, 3}, {8, 1}, {8, 8}, {127, 127}, {8, 2}};
    for (auto [rows, accepted] : steps) {
        const int past = int(reference.size());
        storage.prepare((past + rows + len - 1) / len, past / len);
        std::vector<int> key(rows), value(rows);
        for (int i = 0; i < rows; ++i) { key[i] = ++serial; value[i] = -serial; }
        auto k = upload(key), v = upload(value);
        int keep = storage.appendRows(rows, len);
        appendKv<<<1,128,0,cudaStreamPerThread>>>(view, k.p, v.p, past, rows - keep, 1, keep);
        reference.insert(reference.end(), key.begin(), key.begin() + accepted);
        const int length = int(reference.size());
        storage.prepare((length + len - 1) / len, length / len);
        check(cudaMemcpyAsync(end.p, &length, sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaGraphLaunch(exec, cudaStreamPerThread));
        std::vector<int2> observed(window);
        check(cudaMemcpyAsync(observed.data(), actual.p, window * sizeof(int2), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        for (int i = 0; i < window; ++i) {
            int expected = reference[length - window + i];
            if (observed[i].x != expected || observed[i].y != -expected)
                throw std::runtime_error("Sliding host ring lost a committed token");
        }
    }
    check(cudaGraphExecDestroy(exec));
    check(cudaGraphDestroy(graph));
}

// Two queries reverse the ranking, while replay uses identical graph addresses.
// Check selected logical IDs, chronological order and the absolute bias/sink/tau
// arithmetic against an independent scalar attention over those IDs.
template<class T> void checkSelectedPages(DataType type) {
    constexpr int D = 128, len = 16, pages = 11, hot = 4, past = 10 * len + 3, span = D * len;
    MappedKvPages storage({pages, hot, pages}, 2 * span * sizeof(T));
    KvMemConfig cfg;
    cfg.sinkTokens = len; cfg.recentTokens = len; cfg.retrievalTokens = 2 * len;
    storage.selectPages(cfg, type, len, 1, 1, D);
    storage.prepare(pages, 0);
    PageView view{nullptr, nullptr, storage.addresses(), 0, len, pageShift(len), storage.selector()};
    std::vector<T> keys(pages * span), values(keys.size());
    for (int p = 0; p < pages; ++p) {
        for (int t = 0; t < len; ++t) {
            keys[p * span + t * D] = T(float(p / 2));
            for (int c = 0; c < D; ++c) values[p * span + t * D + c] = T(float(p));
        }
    }
    auto key = upload(keys), value = upload(values);
    appendKv<<<(keys.size() + 255) / 256,256,0,cudaStreamPerThread>>>(view, key.p, value.p, 0, 0, D, pages * len);
    storage.prepare(pages, past / len);
    DecodeVisit visit{}; visit.pages = view; visit.past = past;
    auto visits = upload(std::vector<DecodeVisit>{visit});
    std::vector<float> bias(past + 1), q(D);
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = float((i / 16) % 3) * 0.5f;
    auto b = upload(bias), query = upload(q), sink = upload(std::vector<float>{1.f});
    Matrix out(1, D);
    AttentionRow a{query.p,nullptr,nullptr,{},b.p,sink.p,out.p,visits.p,0,0,1,1,D,past+1,0,32,0.45f};
    cudaGraph_t graph; cudaGraphExec_t exec;
    check(cudaStreamSynchronize(cudaStreamPerThread));
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    selectAttentionPages(a,1,1);
    launchRelativeKernel<D,1,false,T>(a,1,nullptr,1);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (float sign : {1.f, -1.f}) {
        q[0] = sign;
        check(cudaMemcpyAsync(query.p,q.data(),D*sizeof(float),cudaMemcpyHostToDevice,cudaStreamPerThread));
        check(cudaGraphLaunch(exec,cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        kv_selection::State state;
        check(cudaMemcpy(&state,storage.selector(),sizeof(state),cudaMemcpyDeviceToHost));
        const std::vector<int> expected = sign > 0 ? std::vector<int>{0,6,8,9} : std::vector<int>{0,1,2,9};
        std::vector<int> selected(state.kept);
        check(cudaMemcpy(selected.data(),state.selected,selected.size()*sizeof(int),cudaMemcpyDeviceToHost));
        if (selected != expected) throw std::runtime_error("KVMem query ranking/tie or hot page budget");
        const double tau = 1. + a.alpha * std::log(double(past + 1) / a.floor);
        double sum = std::exp(1.), numerator = 0.;
        for (int t = 0; t <= past; ++t) {
            int page = t / len;
            if (page < state.sealed && std::find(expected.begin(),expected.end(),page) == expected.end()) continue;
            double w = std::exp(tau * (sign * float(keys[t * D]) / std::sqrt(double(D)) + bias[past - t]));
            sum += w; numerator += w * float(values[t * D]);
        }
        float actual;
        check(cudaMemcpy(&actual,out.p,sizeof(float),cudaMemcpyDeviceToHost));
        if (!(std::abs(actual - numerator / sum) < 2.e-4))
            throw std::runtime_error("KVMem selected attention differs from absolute-position reference");
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_fp8_e4m3>) {
            if (!configureGatedPrefill<false,half,T>()) continue;
            std::vector<half> packed(2 * D), gates(2 * D, half(0.f));
            packed[0] = packed[D] = half(sign);
            auto pq = upload(packed), pg = upload(gates);
            auto sinks = upload(std::vector<float>{1.f, 1.f});
            auto biases = bias;
            biases.insert(biases.end(), bias.begin(), bias.end());
            auto pb = upload(biases);
            Buffer<half> po(2 * D);
            auto pre = a;
            pre.visit = nullptr; pre.pages = view; pre.past = past;
            pre.key = key.p + size_t(past) * D; pre.value = value.p + size_t(past) * D;
            pre.sink = sinks.p; pre.relative = pb.p;
            launchGatedPrefill<false,half,T>(pre,pq.p,po.p,pg.p,D,2,pb.p);
            std::vector<half> actual(2 * D);
            check(cudaMemcpy(actual.data(),po.p,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
            for (int row = 0; row < 2; ++row) {
                double den = std::exp(1.), acc = 0.;
                const int pos = past + row;
                const double scale = 1. + a.alpha * std::log(double(pos + 1) / a.floor);
                for (int t = 0; t <= pos; ++t) {
                    const int page = t / len, distance = pos - t;
                    if (page < state.sealed && std::find(expected.begin(),expected.end(),page) == expected.end()) continue;
                    const double add = distance < int(bias.size()) ? bias[distance] : 0.;
                    const double mass = std::exp(scale * (sign * float(keys[t * D]) / std::sqrt(double(D)) + add));
                    den += mass; acc += mass * float(values[t * D]);
                }
                if (!(std::abs(float(actual[row * D]) - .5 * acc / den) < .008))
                    throw std::runtime_error("KVMem selected tiled prefill differs from causal sparse reference");
            }
        }
    }
    check(cudaGraphExecDestroy(exec)); check(cudaGraphDestroy(graph));
}

int testRelative() {
    checkSelectedPages<float>(FLOAT32);
    checkSelectedPages<half>(FLOAT16);
    checkSelectedPages<__nv_bfloat16>(BFLOAT16);
    checkSelectedPages<__nv_fp8_e4m3>(FP8_E4M3);
    checkSlidingHostPages();
    checkMappedPages<float>();
    checkMappedPages<half>();
    checkMappedPages<nv_bfloat16>();
    checkMappedPages<__nv_fp8_e4m3>();
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
    // FP8 pages and suffixes share the same reference, including window wrap and clipped verification.
    for (int window : {0, 1024}) {
        compare<128, __nv_fp8_e4m3>(3, 1153, 1, window, 37, true);
        compare<128, __nv_fp8_e4m3>(3, 1019, 16, window, 37, true, 28, true, 0, false, 8);
        compare<128, __nv_fp8_e4m3, half>(3, 1017, 159, window, 31, false);
    }
    compare<64, __nv_fp8_e4m3>(2, 137, 9, 0, 31, false);
    // Each gated case checks its own compiled kernel and shared-memory limit.
    compare<128, half, half>(3, 0, 513, 0, 128, false);
    compare<128, half, half>(5, 8190, 39, 0, 127, false);
    compare<128, half, half>(1, 0, 149, 1, 37, false);
    compare<128, half, half>(3, 1017, 513, 1024, 31, false);
    compare<128, half, half>(5, 8190, 145, 97, 37, false);
    if (major >= 8) {
        compare<128, half, nv_bfloat16>(3, 8190, 39, 0, 127, false);
        compare<128, half, nv_bfloat16>(3, 1017, 159, 1024, 31, false);
        for (int past : {0, 1025, 8190})
            compare<128, nv_bfloat16, nv_bfloat16, nv_bfloat16>(3, past, 159, 1024, 127, false);
        // Short prompt chunks use FP32 relative bias.
        compare<128, nv_bfloat16, nv_bfloat16>(3, 1017, 9, 1024, 31, false);
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
        compare<128, __nv_fp8_e4m3>(3, 6013, 16, 0, 16, true, 54, true, 0, false, 8);
        compare<128, __nv_fp8_e4m3>(3, 1039, 46, 1024, 16, true, 28, true, 0, false, 23);
        // Force more than 32 active splits and replay split slots past the launch grid.
        compare<128, half>(3, 6013, 16, 0, 16, true, 54, true, 26, false, 8);
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
