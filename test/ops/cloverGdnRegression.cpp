#include "devices/cuda/fastllm-clover.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void Check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Buffer {
    void *p = nullptr;
    explicit Buffer(size_t bytes) { Check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    template<class T> void Put(const std::vector<T> &x) { Check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice)); }
    template<class T> std::vector<T> Get(size_t n) const {
        std::vector<T> x(n); Check(cudaMemcpy(x.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return x;
    }
};
uint16_t BF16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4); bits += 0x7fff + ((bits >> 16) & 1); return bits >> 16;
}
float Float(uint16_t x) { uint32_t bits = uint32_t(x) << 16; float y; std::memcpy(&y, &bits, 4); return y; }
std::vector<uint16_t> Pack(const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
        const std::vector<uint16_t> &v, int rows, int kh, int vh, int stride) {
    std::vector<uint16_t> packed((size_t)rows * stride + 128, BF16(123.f));
    for (int row = 0; row < rows; ++row) {
        auto *dst = packed.data() + 64 + (size_t)row * stride;
        std::copy_n(q.data() + (size_t)row * kh * 128, kh * 128, dst);
        std::copy_n(k.data() + (size_t)row * kh * 128, kh * 128, dst + kh * 128);
        std::copy_n(v.data() + (size_t)row * vh * 128, vh * 128, dst + kh * 256);
    }
    return packed;
}
void Close(float actual, float expected, float atol, float rtol) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
        std::cerr << "actual=" << actual << " expected=" << expected << std::endl;
        throw std::runtime_error("GDN reference differs");
    }
}
void Convolution(int batch, int tokens, int channels) {
    std::mt19937 rng(911 + tokens + channels);
    std::normal_distribution<float> normal(0.f, .7f);
    const size_t count = (size_t)batch * tokens * channels, historyCount = (size_t)batch * channels * 3;
    auto random = [&](size_t n) { std::vector<uint16_t> x(n); for (auto &v : x) v = BF16(normal(rng)); return x; };
    auto input = random(count), history = random(historyCount), weight = random((size_t)channels * 4);
    std::vector<uint16_t> outputGuard(count + 128, BF16(123.f));
    std::vector<uint16_t> snapshotGuard(count * 3 + 128, BF16(123.f));
    Buffer x(count * 2), h(historyCount * 2), w(weight.size() * 2),
        y(outputGuard.size() * 2), snapshots(snapshotGuard.size() * 2);
    x.Put(input); w.Put(weight); y.Put(outputGuard); snapshots.Put(snapshotGuard);
    auto launch = [&](bool withSnapshots) {
        Require(FastllmCudaCloverCausalConv(x.p, h.p, w.p, (uint16_t *)y.p + 64,
            withSnapshots ? (uint16_t *)snapshots.p + 64 : nullptr, batch, tokens, channels), "causal convolution launch");
    };
    // A non-null snapshot buffer selects the original token-wise kernel.
    // Check it against the published BF16-product/FP32-sum contract before
    // using its exact bits to compare the optimized path and graph replay.
    auto reference = [&]() {
        h.Put(history); launch(true);
        auto actual = y.Get<uint16_t>(outputGuard.size());
        auto actualSnapshots = snapshots.Get<uint16_t>(snapshotGuard.size());
        auto actualHistory = h.Get<uint16_t>(historyCount);
        for (int i = 0; i < 64; ++i) {
            Require(actual[i] == outputGuard[i] && actual[count + 64 + i] == outputGuard[i], "conv output guard");
            Require(actualSnapshots[i] == snapshotGuard[i] && actualSnapshots[count * 3 + 64 + i] == snapshotGuard[i], "conv snapshot guard");
        }
        auto source = [&](int b, int t, int c) {
            return t < 0 ? history[((size_t)b * channels + c) * 3 + t + 3]
                         : input[((size_t)b * tokens + t) * channels + c];
        };
        for (int b = 0; b < batch; ++b) for (int t = 0; t < tokens; ++t) for (int c = 0; c < channels; ++c) {
            float sum = 0.f;
            for (int tap = 0; tap < 4; ++tap)
                sum += Float(BF16(Float(source(b, t + tap - 3, c)) * Float(weight[c * 4 + tap])));
            float expected = Float(BF16(sum / (1.f + std::exp(-sum))));
            Close(Float(actual[64 + ((size_t)b * tokens + t) * channels + c]), expected, .0005f, .008f);
            for (int tap = 0; tap < 3; ++tap)
                Require(actualSnapshots[64 + (((size_t)t * batch + b) * channels + c) * 3 + tap] ==
                    source(b, t + tap - 2, c), "conv snapshot contents");
        }
        for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) for (int tap = 0; tap < 3; ++tap)
            Require(actualHistory[((size_t)b * channels + c) * 3 + tap] == source(b, tokens + tap - 3, c), "conv committed history");
        return std::make_pair(actual, actualHistory);
    };
    auto expected = reference();
    auto check = [&]() {
        Require(y.Get<uint16_t>(outputGuard.size()) == expected.first, "tiled convolution differs from token-wise path");
        Require(h.Get<uint16_t>(historyCount) == expected.second, "tiled convolution history differs");
        Require(x.Get<uint16_t>(input.size()) == input, "convolution wrote input");
    };
    h.Put(history); y.Put(outputGuard); launch(false); check();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch(false);
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    input = random(count); history = random(historyCount); x.Put(input);
    expected = reference(); h.Put(history); y.Put(outputGuard);
    Check(cudaGraphLaunch(executable, cudaStreamPerThread)); check();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void Run(int batch, int tokens, int kh, int vh, bool cpuReference) {
    std::mt19937 rng(789 + tokens); std::normal_distribution<float> normal(0.f, 1.f);
    auto random = [&](size_t n, float scale) { std::vector<uint16_t> x(n); for (auto &v : x) v = BF16(normal(rng) * scale); return x; };
    auto q = random((size_t)batch * tokens * kh * 128, 1.f), k = random(q.size(), 1.f);
    auto v = random((size_t)batch * tokens * vh * 128, .4f);
    std::vector<uint16_t> beta(batch * tokens * vh);
    std::vector<float> g(beta.size()), initial((size_t)batch * vh * 128 * 128);
    for (auto &x : beta) x = BF16(float(rng() % 1000) / 1000.f);
    for (auto &x : g) x = -.2f * float(rng() % 1000) / 1000.f;
    for (auto &x : initial) x = normal(rng) * .03f;
    const size_t snapCount = tokens <= 32 ? initial.size() * tokens : 1;
    const size_t workCount = (size_t)batch * tokens * (2 * kh * 128 + vh);
    const int stride = (2 * kh + vh) * 128 + 17;
    auto packed = Pack(q, k, v, batch * tokens, kh, vh, stride);
    std::vector<float> sentinel(workCount + 128, 123456.f);
    Buffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), dg(g.size() * 4), db(beta.size() * 2),
        state(initial.size() * 4), output(v.size() * 2), snaps(snapCount * 4), workspace(sentinel.size() * 4), fused(packed.size() * 2);
    dq.Put(q); dk.Put(k); dv.Put(v); dg.Put(g); db.Put(beta); workspace.Put(sentinel);
    fused.Put(packed);
    auto *base = (uint16_t *)fused.p + 64;
    auto launch = [&](bool prepared, bool strided = false) {
        Require(FastllmCudaCloverCausalGdn(strided ? base : dq.p, strided ? base + kh * 128 : dk.p,
            strided ? base + kh * 256 : dv.p, (float *)dg.p, db.p, (float *)state.p,
            output.p, tokens <= 32 ? (float *)snaps.p : nullptr, batch, tokens, kh, vh, nullptr,
            prepared ? (float *)workspace.p + 64 : nullptr, strided ? stride : 0), "GDN launch");
    };
    state.Put(initial); launch(false);
    auto expected = output.Get<uint16_t>(v.size());
    auto expectedState = state.Get<float>(initial.size());
    auto expectedSnaps = tokens <= 32 ? snaps.Get<float>(snapCount) : std::vector<float>();
    auto checkPrepared = [&]() {
        Require(output.Get<uint16_t>(v.size()) == expected, "prepared GDN output is not bitwise equal");
        Require(state.Get<float>(initial.size()) == expectedState, "prepared GDN state is not bitwise equal");
        if (tokens <= 32) Require(snaps.Get<float>(snapCount) == expectedSnaps, "prepared GDN snapshots differ");
        auto guard = workspace.Get<float>(sentinel.size());
        for (int i = 0; i < 64; ++i) Require(guard[i] == sentinel[i] && guard[workCount + 64 + i] == sentinel[i], "GDN workspace overrun");
    };
    state.Put(initial); launch(true); checkPrepared();
    for (bool prepared : {false, true}) { state.Put(initial); launch(prepared, true); checkPrepared(); }
    Require(fused.Get<uint16_t>(packed.size()) == packed, "causal GDN wrote its strided inputs");
    for (int badStride : {-1, vh * 128 - 1})
        Require(!FastllmCudaCloverCausalGdn(dq.p, dk.p, dv.p, (float *)dg.p, db.p,
            (float *)state.p, output.p, nullptr, batch, tokens, kh, vh, nullptr, nullptr, badStride), "invalid causal stride accepted");
    if (cpuReference) {
        // Independent scalar form of the published recurrent_commit_torch:
        // normalize Q/K, decay S, apply the beta-scaled delta, then query S.
        auto refState = initial;
        std::vector<uint16_t> refOutput(v.size());
        for (int b = 0; b < batch; ++b) for (int t = 0; t < tokens; ++t) for (int h = 0; h < vh; ++h) {
            const size_t row = (size_t)b * tokens + t, qk = (row * kh + h / (vh / kh)) * 128;
            float qn[128], kn[128], qs = 0.f, ks = 0.f;
            for (int d = 0; d < 128; ++d) { qn[d] = Float(q[qk + d]); kn[d] = Float(k[qk + d]); qs += qn[d] * qn[d]; ks += kn[d] * kn[d]; }
            float qi = 1.f / std::sqrt(qs + 1.e-6f) / std::sqrt(128.f), ki = 1.f / std::sqrt(ks + 1.e-6f);
            for (int d = 0; d < 128; ++d) { qn[d] *= qi; kn[d] *= ki; }
            float decay = std::exp(g[row * vh + h]), rate = Float(beta[row * vh + h]);
            for (int d = 0; d < 128; ++d) {
                float *s = refState.data() + (((size_t)b * vh + h) * 128 + d) * 128;
                float remembered = 0.f;
                for (int j = 0; j < 128; ++j) { s[j] *= decay; remembered += s[j] * kn[j]; }
                float delta = (Float(v[(row * vh + h) * 128 + d]) - remembered) * rate;
                float y = 0.f;
                for (int j = 0; j < 128; ++j) { s[j] += kn[j] * delta; y += s[j] * qn[j]; }
                refOutput[(row * vh + h) * 128 + d] = BF16(y);
            }
        }
        for (size_t i = 0; i < expected.size(); ++i) Close(Float(expected[i]), Float(refOutput[i]), .002f, .008f);
        for (size_t i = 0; i < refState.size(); ++i) Close(expectedState[i], refState[i], 2.e-5f, 2.e-4f);
    }
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch(true, true);
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &x : q) x = BF16(normal(rng));
    packed = Pack(q, k, v, batch * tokens, kh, vh, stride); fused.Put(packed);
    dq.Put(q); state.Put(initial); launch(false);
    expected = output.Get<uint16_t>(v.size()); expectedState = state.Get<float>(initial.size()); if (tokens <= 32) expectedSnaps = snaps.Get<float>(snapCount);
    state.Put(initial); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); checkPrepared();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void BlockPrefill(int batch, int tokens, int kh, int vh, bool weakDecay) {
    size_t bytes = FastllmCudaCloverPrefillGdnWorkspaceBytes(batch, tokens, kh, vh);
    if (!bytes) return; // The scalar path above covers unsupported architectures.
    std::mt19937 rng(7103 + tokens + batch);
    std::normal_distribution<float> normal(0.f, .3f);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);
    const size_t rows = (size_t)batch * tokens, states = (size_t)batch * vh * 128 * 128;
    const int stride = (2 * kh + vh) * 128 + 3;
    auto random = [&](size_t n) { std::vector<uint16_t> x(n); for (auto &v : x) v = BF16(normal(rng)); return x; };
    auto q = random(rows * kh * 128), k = random(q.size()), v = random(rows * vh * 128);
    std::vector<float> gates(rows * vh), initial(states + 64, 123.f);
    std::vector<uint16_t> beta(rows * vh), outputGuard(v.size() + 128, BF16(123.f));
    for (auto &x : gates) x = -uniform(rng) * (weakDecay ? .001f : .3f);
    for (auto &x : beta) x = BF16(uniform(rng));
    for (size_t i = 32; i < states + 32; ++i) initial[i] = normal(rng) * .1f;
    auto packed = Pack(q, k, v, rows, kh, vh, stride);
    std::vector<unsigned char> workGuard(bytes + 256, 173);
    Buffer input(packed.size() * 2), g(gates.size() * 4), b(beta.size() * 2),
        state(initial.size() * 4), output(outputGuard.size() * 2), workspace(workGuard.size()),
        referenceState(initial.size() * 4), referenceOutput(outputGuard.size() * 2),
        scalarWork(rows * (2 * kh * 128 + vh) * 4);
    input.Put(packed); g.Put(gates); b.Put(beta); workspace.Put(workGuard);
    auto *qp = (uint16_t *)input.p + 64;
    auto *kp = qp + kh * 128;
    auto *vp = qp + kh * 256;
    auto launch = [&](size_t capacity, void *work) {
        return FastllmCudaCloverPrefillGdn(qp, kp, vp, (float *)g.p, b.p,
            (float *)state.p + 32, (uint16_t *)output.p + 64, batch, tokens, kh, vh,
            work, capacity, nullptr, stride);
    };
    auto *work = (unsigned char *)workspace.p + 128;
    state.Put(initial); output.Put(outputGuard);
    Require(!launch(bytes - 1, work), "undersized block workspace accepted");
    Require(!launch(bytes, work + 1), "unaligned block workspace accepted");
    Require(state.Get<float>(initial.size()) == initial && output.Get<uint16_t>(outputGuard.size()) == outputGuard,
        "invalid prefill changed state/output");
    auto checkReference = [&]() {
        referenceState.Put(initial); referenceOutput.Put(outputGuard);
        Require(FastllmCudaCloverCausalGdn(qp, kp, vp, (float *)g.p, b.p,
            (float *)referenceState.p + 32, (uint16_t *)referenceOutput.p + 64, nullptr,
            batch, tokens, kh, vh, nullptr, (float *)scalarWork.p, stride), "scalar block reference launch");
        auto y = output.Get<uint16_t>(outputGuard.size()), ref = referenceOutput.Get<uint16_t>(outputGuard.size());
        auto s = state.Get<float>(initial.size()), rs = referenceState.Get<float>(initial.size());
        double yd = 0., yn = 0., sd = 0., sn = 0.;
        for (size_t i = 64; i < v.size() + 64; ++i) {
            double a = Float(y[i]), z = Float(ref[i]); Require(std::isfinite(a), "nonfinite prefill output");
            yd += (a-z)*(a-z); yn += z*z;
        }
        for (size_t i = 32; i < states + 32; ++i) {
            double a = s[i], z = rs[i]; Require(std::isfinite(a), "nonfinite prefill state");
            sd += (a-z)*(a-z); sn += z*z;
        }
        // BF16 block products are checked against the independently validated
        // scalar recurrence; a relative aggregate bound avoids zero-denominator
        // failures for individual components that cancel.
        Require(std::sqrt(yd / std::max(yn, 1.e-30)) < .02 &&
            std::sqrt(sd / std::max(sn, 1.e-30)) < .02, "block/scalar recurrence differs");
        for (int i = 0; i < 64; ++i)
            Require(y[i] == outputGuard[i] && y[v.size()+64+i] == outputGuard[i], "block output guard");
        for (int i = 0; i < 32; ++i)
            Require(s[i] == initial[i] && s[states+32+i] == initial[i], "block state guard");
        auto wg = workspace.Get<unsigned char>(workGuard.size());
        for (int i = 0; i < 128; ++i)
            Require(wg[i] == 173 && wg[bytes+128+i] == 173, "block workspace guard");
        Require(input.Get<uint16_t>(packed.size()) == packed, "prefill modified QKV");
    };
    Require(launch(bytes, work), "block prefill launch"); checkReference();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(launch(bytes, work), "block prefill capture");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &x : q) x = BF16(normal(rng));
    for (size_t i = 32; i < states + 32; ++i) initial[i] = normal(rng) * .1f;
    packed = Pack(q, k, v, rows, kh, vh, stride); input.Put(packed); state.Put(initial);
    Require(launch(bytes, work), "refreshed block prefill"); checkReference();
    auto expectedY = output.Get<uint16_t>(outputGuard.size());
    auto expectedState = state.Get<float>(initial.size());
    state.Put(initial); output.Put(outputGuard);
    Check(cudaGraphLaunch(executable, cudaStreamPerThread));
    Require(output.Get<uint16_t>(outputGuard.size()) == expectedY && state.Get<float>(initial.size()) == expectedState,
        "block graph replay differs from eager");
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}

void ResidualNormalization(int rows, int width, bool centered, bool unaligned = false) {
    const int offset = unaligned || width % 2 ? 1 : 64;
    const size_t count = (size_t)rows * width, size = count + 2 * offset;
    std::mt19937 rng(1201 + rows + width);
    std::normal_distribution<float> normal(0.f, .7f);
    std::vector<uint16_t> input(size, BF16(123.f)), residual = input, guard = input;
    std::vector<float> weights(width);
    Buffer x(size * 2), h(size * 2), expectedH(size * 2), w(width * 4), y(size * 2), expectedY(size * 2);
    auto refresh = [&]() {
        for (size_t i = offset; i < offset + count; ++i) {
            input[i] = BF16(normal(rng)); residual[i] = BF16(normal(rng));
            if (i % 7 == 0) residual[i] = BF16(-Float(input[i]));
            if (i % 13 == 0) { input[i] = BF16(1.e-20f); residual[i] = BF16(-1.e-20f); }
            if (i % 29 == 0) { input[i] = BF16(1.e10f); residual[i] = BF16(-1.e10f); }
        }
        for (auto &v : weights) v = normal(rng);
        x.Put(input); h.Put(residual); w.Put(weights); y.Put(guard); expectedY.Put(guard);
        auto rounded = residual;
        for (size_t i = offset; i < offset + count; ++i) rounded[i] = BF16(Float(residual[i]) + Float(input[i]));
        expectedH.Put(rounded);
        Require(FastllmCudaCloverNorm((uint16_t *)expectedH.p + offset, (float *)w.p, nullptr,
            (uint16_t *)expectedY.p + offset, rows, width, 1.e-6f, centered), "residual norm reference launch");
        auto expected = expectedY.Get<uint16_t>(size);
        // Independent RMS equation checks the original normalization path;
        // exact comparison below additionally preserves its reduction tree.
        for (int row = 0; row < rows; ++row) {
            double sum = 0.;
            for (int d = 0; d < width; ++d) { double v = Float(rounded[offset + (size_t)row * width + d]); sum += v * v; }
            double inv = 1. / std::sqrt(sum / width + 1.e-6);
            for (int d = 0; d < width; ++d) {
                size_t i = offset + (size_t)row * width + d;
                float reference = Float(rounded[i]) * inv * (weights[d] + (centered ? 1. : 0.));
                Close(Float(expected[i]), Float(BF16(reference)), .0005f, .008f);
            }
        }
        return std::make_pair(rounded, expected);
    };
    auto launch = [&]() { return FastllmCudaCloverAddNorm((uint16_t *)h.p + offset,
        (uint16_t *)x.p + offset, (float *)w.p, (uint16_t *)y.p + offset, rows, width, 1.e-6f, centered); };
    auto expected = refresh();
    auto check = [&]() {
        Require(h.Get<uint16_t>(size) == expected.first, "fused residual rounding or guard differs");
        Require(y.Get<uint16_t>(size) == expected.second, "fused residual normalization or guard differs");
        Require(x.Get<uint16_t>(size) == input, "fused normalization changed addition input");
    };
    Require(launch(), "residual norm launch"); check();
    Require(!FastllmCudaCloverAddNorm(h.p, x.p, (float *)w.p, h.p, rows, width, 1.e-6f, centered),
        "residual/output alias accepted");
    Require(!FastllmCudaCloverAddNorm(h.p, x.p, (float *)w.p, y.p, 0, width, 1.e-6f, centered),
        "invalid residual norm shape accepted");
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(launch(), "residual norm capture launch");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    expected = refresh(); Check(cudaGraphLaunch(executable, cudaStreamPerThread)); check();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void Normalization(int rows, int width, bool gated, bool centered) {
    std::mt19937 rng(rows + width);
    std::normal_distribution<float> normal(0.f, 1.f);
    const size_t count = (size_t)rows * width;
    const uint16_t sentinel = BF16(123.f);
    std::vector<uint16_t> input(count), gate(count), guarded(count + 128, sentinel);
    std::vector<float> weight(width);
    for (auto &x : input) x = BF16(normal(rng));
    for (auto &x : gate) x = BF16(normal(rng));
    for (auto &x : weight) x = normal(rng);
    Buffer x(count * 2), g(count * 2), w(width * 4), out(guarded.size() * 2), ref(count * 2);
    x.Put(input); g.Put(gate); w.Put(weight); out.Put(guarded);
    auto launch = [&](bool bulk) {
        for (int row = 0; row < rows; row += bulk ? rows : 511) {
            size_t offset = (size_t)row * width;
            Require(FastllmCudaCloverNorm((uint16_t *)x.p + offset, (float *)w.p,
                gated ? (uint16_t *)g.p + offset : nullptr,
                bulk ? (uint16_t *)out.p + 64 : (uint16_t *)ref.p + offset,
                bulk ? rows : std::min(511, rows - row), width, 1.e-6f, centered, nullptr), "norm launch");
        }
    };
    auto check = [&]() {
        auto actual = out.Get<uint16_t>(guarded.size()), expected = ref.Get<uint16_t>(count);
        Require(std::equal(expected.begin(), expected.end(), actual.begin() + 64), "bulk norm differs from CTA path");
        for (int i = 0; i < 64; ++i)
            Require(actual[i] == sentinel && actual[count + 64 + i] == sentinel, "norm output overrun");
        Require(x.Get<uint16_t>(count) == input && g.Get<uint16_t>(count) == gate, "norm wrote inputs");
        for (int row = 0; row < rows; ++row) {
            double sum = 0.;
            for (int d = 0; d < width; ++d) { float v = Float(input[(size_t)row * width + d]); sum += double(v) * v; }
            double inv = 1. / std::sqrt(sum / width + 1.e-6);
            for (int d = 0; d < width; ++d) {
                size_t pos = (size_t)row * width + d;
                double value = Float(input[pos]) * inv * (weight[d] + (centered ? 1.f : 0.f));
                if (gated) { double z = Float(gate[pos]); value *= z / (1. + std::exp(-z)); }
                Close(Float(actual[pos + 64]), Float(BF16(value)), .001f, .008f);
            }
        }
    };
    launch(false); launch(true); check();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)); launch(true);
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &v : input) v = BF16(normal(rng));
    for (auto &v : gate) v = BF16(normal(rng));
    x.Put(input); g.Put(gate); launch(false);
    Check(cudaGraphLaunch(executable, cudaStreamPerThread)); check();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void AttentionPreparation(int tokens, int qh, int kh, int rotaryDim, int position, bool fused = false) {
    std::mt19937 rng(271 + tokens + position);
    std::normal_distribution<float> normal(0.f, .7f);
    const size_t qcount = (size_t)tokens * qh * 256, kcount = (size_t)tokens * kh * 256;
    std::vector<uint16_t> qgate(qcount * 2), key(kcount), query(qcount), gate(qcount);
    std::vector<float> qw(256), kw(256);
    for (auto &x : qgate) x = BF16(normal(rng));
    for (auto &x : key) x = BF16(normal(rng));
    for (int i = 0; i < 256; ++i) { qw[i] = normal(rng); kw[i] = normal(rng); }
    for (size_t row = 0; row < qcount / 256; ++row) for (int d = 0; d < 256; ++d) {
        query[row * 256 + d] = qgate[row * 512 + d];
        gate[row * 256 + d] = qgate[row * 512 + 256 + d];
    }
    Buffer qg(qgate.size() * 2), k(kcount * 2), q(qcount * 2), wq(1024), wk(1024),
        refQ(qcount * 2), refK(kcount * 2), outQ((qcount + 128) * 2), outK((kcount + 128) * 2), outG((qcount + 128) * 2);
    qg.Put(qgate); k.Put(key); q.Put(query); wq.Put(qw); wk.Put(kw);
    std::vector<uint16_t> qGuard(qcount + 128, BF16(123.f)), kGuard(kcount + 128, BF16(123.f));
    outQ.Put(qGuard); outG.Put(qGuard); outK.Put(kGuard);
    Require(FastllmCudaCloverNorm(q.p, (float *)wq.p, nullptr, refQ.p, tokens * qh, 256, 1.e-6f, true), "reference Q norm");
    Require(FastllmCudaCloverNorm(k.p, (float *)wk.p, nullptr, refK.p, tokens * kh, 256, 1.e-6f, true), "reference K norm");
    Require(FastllmCudaCloverRotary(refQ.p, tokens, qh, 256, rotaryDim, position, 1.e7f), "reference Q rotary");
    Require(FastllmCudaCloverRotary(refK.p, tokens, kh, 256, rotaryDim, position, 1.e7f), "reference K rotary");
    const int stride = (qh + kh) * 512 + 17;
    std::vector<uint16_t> packed((size_t)tokens * stride + 128, BF16(123.f)), value(kcount);
    for (auto &x : value) x = BF16(normal(rng));
    for (int row = 0; row < tokens; ++row) {
        auto dst = packed.begin() + 64 + (size_t)row * stride;
        std::copy_n(qgate.begin() + (size_t)row * qh * 512, qh * 512, dst);
        std::copy_n(key.begin() + (size_t)row * kh * 256, kh * 256, dst + qh * 512);
        std::copy_n(value.begin() + (size_t)row * kh * 256, kh * 256, dst + qh * 512 + kh * 256);
    }
    Buffer fusedInput(packed.size() * 2), outV(kGuard.size() * 2);
    fusedInput.Put(packed); outV.Put(kGuard);
    auto launch = [&]() {
        return FastllmCudaCloverPrepareAttention(fused ? (uint16_t *)fusedInput.p + 64 : qg.p,
            fused ? (uint16_t *)fusedInput.p + 64 + qh * 512 : k.p, (float *)wq.p, (float *)wk.p,
            (uint16_t *)outQ.p + 64, (uint16_t *)outK.p + 64, (uint16_t *)outG.p + 64,
            tokens, qh, kh, rotaryDim, position, 1.e7f, 1.e-6f, nullptr, fused ? stride : 0,
            fused ? (uint16_t *)outV.p + 64 : nullptr);
    };
    auto check = [&]() {
        auto a = outQ.Get<uint16_t>(qGuard.size()), b = outK.Get<uint16_t>(kGuard.size()), c = outG.Get<uint16_t>(qGuard.size());
        auto eq = refQ.Get<uint16_t>(qcount), ek = refK.Get<uint16_t>(kcount);
        Require(std::equal(eq.begin(), eq.end(), a.begin() + 64), "fused Q norm/rotary differs");
        Require(std::equal(ek.begin(), ek.end(), b.begin() + 64), "fused K norm/rotary differs");
        Require(std::equal(gate.begin(), gate.end(), c.begin() + 64), "fused gate split differs");
        for (int i = 0; i < 64; ++i) {
            Require(a[i] == qGuard[i] && a[qcount + 64 + i] == qGuard[i] &&
                c[i] == qGuard[i] && c[qcount + 64 + i] == qGuard[i] &&
                b[i] == kGuard[i] && b[kcount + 64 + i] == kGuard[i], "attention preparation output guard");
        }
        if (fused) {
            auto actualV = outV.Get<uint16_t>(kGuard.size());
            Require(std::equal(value.begin(), value.end(), actualV.begin() + 64), "fused V unpack differs");
            for (int i = 0; i < 64; ++i)
                Require(actualV[i] == kGuard[i] && actualV[kcount + 64 + i] == kGuard[i], "V output guard");
            Require(fusedInput.Get<uint16_t>(packed.size()) == packed, "fused QKV input modified");
        }
        Require(qg.Get<uint16_t>(qgate.size()) == qgate && k.Get<uint16_t>(key.size()) == key,
            "attention preparation changed inputs");
    };
    Require(launch(), "fused attention preparation launch"); check();
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(launch(), "fused attention preparation capture");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    outQ.Put(qGuard); outG.Put(qGuard); outK.Put(kGuard);
    Check(cudaGraphLaunch(executable, cudaStreamPerThread)); check();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
void Preview(int batch, int tokens, int kh, int vh, bool masked) {
    std::mt19937 rng(42 + tokens); std::normal_distribution<float> normal(0.f, .2f);
    auto random = [&](size_t n) { std::vector<uint16_t> x(n); for (auto &v : x) v = BF16(normal(rng)); return x; };
    const int rows = batch * tokens, stride = (2 * kh + vh) * 128 + 17;
    auto q = random((size_t)rows * kh * 128), k = random(q.size()), v = random((size_t)rows * vh * 128);
    auto packed = Pack(q, k, v, rows, kh, vh, stride);
    std::vector<float> state((size_t)batch * vh * 128 * 128), g(rows * vh), scale(vh), bias(vh), hc(vh);
    for (auto &x : state) x = normal(rng);
    for (auto &x : g) x = -std::abs(normal(rng));
    for (int h = 0; h < vh; ++h) { scale[h] = 2.f + normal(rng); bias[h] = normal(rng); hc[h] = normal(rng); }
    std::vector<int> slots(batch);
    std::vector<unsigned char> valid(rows, 1);
    for (int b = 0; b < batch; ++b) slots[b] = b == batch - 1 && masked ? -1 : batch - 1 - b;
    if (masked) for (int i = 0; i < rows; ++i) valid[i] = i % 3 != 0;
    Buffer dq(q.size() * 2), dk(k.size() * 2), dv(v.size() * 2), fused(packed.size() * 2),
        ds(state.size() * 4), dg(g.size() * 4), dscale(vh * 4), dbias(vh * 4), dhc(vh * 4),
        dslots(batch * 4), dvalid(rows), out(v.size() * 2);
    dq.Put(q); dk.Put(k); dv.Put(v); fused.Put(packed); ds.Put(state); dg.Put(g);
    dscale.Put(scale); dbias.Put(bias); dhc.Put(hc); dslots.Put(slots); dvalid.Put(valid);
    auto *base = (uint16_t *)fused.p + 64;
    auto launch = [&](bool strided, int rowStride) {
        return FastllmCudaCloverPreviewGdn(strided ? base : dq.p, strided ? base + kh * 128 : dk.p,
            strided ? base + kh * 256 : dv.p, (float *)dg.p, (float *)ds.p, (float *)dscale.p,
            (float *)dbias.p, (float *)dhc.p, masked ? (int *)dslots.p : nullptr,
            masked ? (unsigned char *)dvalid.p : nullptr, out.p, batch, tokens, kh, vh, nullptr, rowStride);
    };
    Require(launch(false, 0), "preview contiguous launch");
    auto expected = out.Get<uint16_t>(v.size());
    auto check = [&]() {
        Require(out.Get<uint16_t>(v.size()) == expected, "strided preview differs from contiguous");
        Require(ds.Get<float>(state.size()) == state, "preview modified recurrent state");
        Require(fused.Get<uint16_t>(packed.size()) == packed, "preview wrote its strided inputs");
        if (masked) for (int row = 0; row < rows; ++row) if (slots[row / tokens] < 0 || !valid[row])
            for (int d = 0; d < vh * 128; ++d) Require(expected[(size_t)row * vh * 128 + d] == 0, "inactive preview row nonzero");
        // Independent scalar form of the published bidirectional local-memory
        // equation: normalize Q/K, softmax local and history logits together,
        // then mix local values with the read-only FP32 recurrent state.
        for (int b = 0; b < batch; ++b) for (int t = 0; t < tokens; ++t) for (int h = 0; h < vh; ++h) {
            const int row = b * tokens + t, slot = masked ? slots[b] : b, keyHead = h / (vh / kh);
            if (slot < 0 || (masked && !valid[row])) continue;
            float qn[128], kn[128]; double qs = 0.;
            for (int d = 0; d < 128; ++d) {
                qn[d] = Float(q[((size_t)row * kh + keyHead) * 128 + d]);
                qs += double(qn[d]) * qn[d];
            }
            for (auto &x : qn) x = float(x / std::sqrt(qs + 1.e-6));
            int count = 0; for (int j = 0; j < tokens; ++j) count += !masked || valid[b * tokens + j];
            std::vector<double> weights(tokens + 1);
            weights[tokens] = bias[h] + hc[h] * std::log(std::max(count, 1) / 32.);
            double maximum = weights[tokens];
            for (int j = 0; j < tokens; ++j) {
                if (masked && !valid[b * tokens + j]) { weights[j] = -INFINITY; continue; }
                double ks = 0.;
                for (int d = 0; d < 128; ++d) {
                    kn[d] = Float(k[(((size_t)b * tokens + j) * kh + keyHead) * 128 + d]);
                    ks += double(kn[d]) * kn[d];
                }
                double dot = 0.;
                for (int d = 0; d < 128; ++d) dot += qn[d] * (kn[d] / std::sqrt(ks + 1.e-6));
                weights[j] = dot * scale[h]; maximum = std::max(maximum, weights[j]);
            }
            double denominator = 0.;
            for (auto &x : weights) { x = std::exp(x - maximum); denominator += x; }
            for (auto &x : weights) x /= denominator;
            for (int d = 0; d < 128; ++d) {
                double history = 0., local = 0.;
                const float *s = state.data() + (((size_t)slot * vh + h) * 128 + d) * 128;
                for (int j = 0; j < 128; ++j) history += double(qn[j]) * s[j];
                for (int j = 0; j < tokens; ++j)
                    local += weights[j] * Float(v[(((size_t)b * tokens + j) * vh + h) * 128 + d]);
                float reference = float(weights[tokens] * history / std::sqrt(128.) * std::exp(g[row * vh + h]) + local);
                Close(Float(expected[((size_t)row * vh + h) * 128 + d]), Float(BF16(reference)), .001f, .01f);
            }
        }
    };
    Require(launch(true, stride), "preview strided launch"); check();
    Require(!launch(true, -1) && !launch(true, vh * 128 - 1), "invalid preview stride accepted");
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(launch(true, stride), "preview capture launch");
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph)); Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (auto &x : q) x = BF16(normal(rng));
    for (auto &x : v) x = BF16(normal(rng));
    if (masked) { std::fill(valid.begin(), valid.end(), 0); valid[0] = 1; slots[0] = 0; }
    packed = Pack(q, k, v, rows, kh, vh, stride);
    dq.Put(q); dv.Put(v); fused.Put(packed); dslots.Put(slots); dvalid.Put(valid);
    Require(launch(false, 0), "refreshed contiguous preview"); expected = out.Get<uint16_t>(v.size());
    Check(cudaGraphLaunch(executable, cudaStreamPerThread)); check();
    Check(cudaGraphExecDestroy(executable)); Check(cudaGraphDestroy(graph));
}
}
int main() {
    try {
        int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Check(cudaSetDevice(0));
        for (int tokens : {1, 2, 3, 127, 128, 129}) Convolution(2, tokens, 2048);
        Convolution(2, 129, 2051);
        Convolution(2, 129, 2047);
        Convolution(1, 512, 8192);
        for (int tokens : {1, 2, 4, 16, 32, 65}) Run(3, tokens, 2, 4, true);
        for (int tokens : {1, 16, 32, 512, 2048, 4096}) Run(1, tokens, 16, 32, tokens == 16);
        Run(1, 128, 16, 16, true);
        Run(2, 129, 4, 16, true);
        Run(1, 513, 16, 32, false);
        Run(1, 4096, 1, 1, true);
        Require(!FastllmCudaCloverPrefillGdnWorkspaceBytes(1, 0, 2, 4) &&
            !FastllmCudaCloverPrefillGdnWorkspaceBytes(1, 64, 3, 4) &&
            !FastllmCudaCloverPrefillGdnWorkspaceBytes(65535, 65535, 16, 32), "invalid prefill geometry accepted");
        for (int tokens : {1, 63, 64, 65, 129, 1025}) BlockPrefill(2, tokens, 2, 4, tokens == 129);
        BlockPrefill(1, 4096, 16, 32, false);
        BlockPrefill(1, 4096, 16, 32, true);
        for (int width : {128, 256}) for (int rows : {31, 512, 1023, 1024, 1025, 4099})
            for (bool gated : {false, true}) for (bool centered : {false, true})
                Normalization(rows, width, gated, centered);
        Normalization(1025, 131, true, false);
        Normalization(1025, 2048, false, true);
        for (int rows : {1, 2, 16, 32, 64, 128, 256, 4096}) for (bool centered : {false, true})
            ResidualNormalization(rows, 2048, centered);
        ResidualNormalization(17, 2048, true, true);
        for (int rows : {1, 16, 32, 64, 128, 256})
            for (bool gated : {false, true}) for (bool centered : {false, true})
                Normalization(rows, 2048, gated, centered);
        for (int width : {1, 127, 128, 256, 517, 4096}) for (bool centered : {false, true})
            ResidualNormalization(17, width, centered);
        ResidualNormalization(1025, 128, true);
        ResidualNormalization(1025, 256, false);
        for (bool fused : {false, true})
        for (int tokens : {1, 2, 16, 32, 257}) for (int rotary : {2, 64, 128, 256}) {
            AttentionPreparation(tokens, 16, 2, rotary, 4096, fused);
            AttentionPreparation(tokens, 1, 1, rotary, 0, fused);
        }
        for (int tokens : {1, 2, 4, 8, 16, 32}) for (bool masked : {false, true}) {
            Preview(3, tokens, 2, 4, masked); Preview(1, tokens, 16, 32, masked);
        }
        std::cout << "Clover GDN: CPU convolution/recurrence, bitwise normalization reuse and strided QKV, snapshots, read-only preview, masks, refreshed graphs and guards PASS" << std::endl;
    } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
