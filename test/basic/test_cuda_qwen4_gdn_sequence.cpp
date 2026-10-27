#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Init(Data &data, const std::vector<int> &shape,
                 const std::vector<float> &values) {
    data.Resize(shape);
    data.Allocate();
    for (size_t i = 0; i < values.size(); ++i) {
        if (data.dataType == FLOAT32) ((float *)data.cpuData)[i] = values[i];
        else if (data.dataType == FLOAT16)
            ((uint16_t *)data.cpuData)[i] = float_to_half(values[i]);
        else {
            uint32_t bits;
            std::memcpy(&bits, &values[i], sizeof(bits));
            ((uint16_t *)data.cpuData)[i] = bits >> 16;
        }
    }
    data.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, true);
}

static std::vector<float> Read(const Data &data) {
    std::vector<float> values(data.Count(0));
    Require(cudaMemcpy(values.data(), data.cudaData, values.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "GDN copy failed");
    return values;
}

static float Compare(const std::vector<float> &actual,
                     const std::vector<float> &expected, const char *message,
                     float absolute = 1e-7f, float relative = 1e-6f) {
    Require(actual.size() == expected.size(), message);
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        // The established out-of-place and in-place kernels can differ by
        // a few FP32 ULPs due to contraction of the state update expression.
        Require(std::isfinite(actual[i]) &&
                error <= absolute + relative * std::abs(expected[i]), message);
        maximum = std::max(maximum, error);
    }
    return maximum;
}

static void Run(int batch, int keyHeads, int valueHeads, int sequence,
                DataType gateType, bool outOfPlace, int passes = 1, bool graph = false) {
    const int channels = (2 * keyHeads + valueHeads) * 128;
    std::vector<float> x(batch * sequence * channels);
    std::vector<float> a(batch * sequence * valueHeads), beta(a.size());
    std::vector<float> state(batch * valueHeads * 128 * 128);
    std::vector<float> log(valueHeads), bias(valueHeads);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(i * 1.37f) * .13f;
    for (size_t i = 0; i < state.size(); ++i) state[i] = std::cos(i * .73f) * .02f;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (int(i % 17) - 8) / 8.f;
        beta[i] = (int(i % 13) - 6) / 7.f;
    }
    for (int i = 0; i < valueHeads; ++i) {
        log[i] = (i % 5 - 2) * .17f;
        bias[i] = (i % 7 - 3) * .11f;
    }
    if (sequence >= 64) {
        // Slow decay stresses error accumulation over successive chunks;
        // extreme gates cover both softplus branches in all input dtypes.
        for (int i = 0; i < valueHeads; ++i) log[i] = -float(i % 7);
        a[0] = -30.f; a[1] = 30.f;
        beta[0] = -20.f; beta[1] = 20.f;
    }
    Data q(FLOAT32), alpha(gateType), b(gateType), alog(FLOAT32), dt(FLOAT32);
    Data st(FLOAT32), next(FLOAT32), referenceState(FLOAT32), output(FLOAT32);
    const std::vector<int> stateShape{batch, valueHeads, 128, 128};
    Init(q, {batch, sequence, channels}, x);
    Init(alpha, {batch, sequence, valueHeads}, a);
    Init(b, {batch, sequence, valueHeads}, beta);
    Init(alog, {valueHeads}, log); Init(dt, {valueHeads}, bias);
    Init(st, stateShape, state); Init(referenceState, stateShape, state);
    Init(next, stateShape, std::vector<float>(state.size(), -123.f));
    Init(output, {batch, sequence, valueHeads, 128},
         std::vector<float>(batch * sequence * valueHeads * 128));
    Data reference(FLOAT32);
    Init(reference, output.dims, std::vector<float>(output.Count(0)));
    auto candidate = [&]() {
        return FastllmCudaQwen4GatedDeltaRuleDecode(q, alpha, b, alog, dt, st, output,
            keyHeads, valueHeads, 128, 128, 1e-6f, outOfPlace ? &next : nullptr);
    };
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)
                == cudaSuccess, "GDN sequence capture start failed");
        Require(candidate(), "GDN sequence capture rejected");
        Require(cudaStreamEndCapture(cudaStreamPerThread, &captured) == cudaSuccess,
                "GDN sequence capture end failed");
        Require(cudaGraphInstantiateWithFlags(&executable, captured, 0) == cudaSuccess,
                "GDN sequence graph instantiate failed");
    }
    for (int pass = 0; pass < passes; ++pass) {
        const auto before = outOfPlace ? Read(st) : std::vector<float>();
        if (graph) Require(cudaGraphLaunch(executable, cudaStreamPerThread) == cudaSuccess,
                           "GDN sequence graph launch failed");
        else Require(candidate(), "GDN sequence rejected");
        if (outOfPlace) Require(Read(st) == before, "GDN modified input state");

        // One-token calls use the unchanged decode kernel. Borrow rows from
        // the same rounded inputs to avoid testing the prefill against itself.
        const size_t gateBytes = gateType == FLOAT32 ? sizeof(float) : sizeof(uint16_t);
        for (int item = 0; item < batch; ++item) {
            for (int token = 0; token < sequence; ++token) {
                const size_t row = size_t(item) * sequence + token;
                Data sq(FLOAT32, {1, 1, channels}, CUDA,
                        (float *)q.cudaData + row * channels);
                Data sa(gateType, {1, 1, valueHeads}, CUDA,
                        (char *)alpha.cudaData + row * valueHeads * gateBytes);
                Data sb(gateType, {1, 1, valueHeads}, CUDA,
                        (char *)b.cudaData + row * valueHeads * gateBytes);
                Data ss(FLOAT32, {1, valueHeads, 128, 128}, CUDA,
                        (float *)referenceState.cudaData + size_t(item) * valueHeads * 128 * 128);
                Data so(FLOAT32, {1, 1, valueHeads, 128}, CUDA,
                        (float *)reference.cudaData + row * valueHeads * 128);
                for (auto *alias : {&sq, &sa, &sb, &ss, &so}) {
                    alias->cudaDataBorrowed = true;
                    alias->dataDeviceIds = {FastllmCudaGetDevice()};
                }
                Require(FastllmCudaQwen4GatedDeltaRuleDecode(sq, sa, sb, alog, dt,
                    ss, so, keyHeads, valueHeads, 128, 128, 1e-6f),
                    "GDN single token rejected");
            }
        }
        // Long prefill sums four FP32 state tiles in a different order. Keep
        // the stricter existing decode tolerance below the prefill boundary.
        const float absolute = sequence >= 64 ? 1e-6f : 1e-7f;
        const float relative = sequence >= 64 ? 1e-5f : 1e-6f;
        const float outputError = Compare(Read(output), Read(reference),
            "GDN sequence output differs from single tokens", absolute, relative);
        const float stateError = Compare(Read(outOfPlace ? next : st), Read(referenceState),
            "GDN sequence final state differs from single tokens", absolute, relative);
        std::printf("PASS batch=%d key_heads=%d value_heads=%d sequence=%d dtype=%d out_of_place=%d pass=%d graph=%d output_error=%g state_error=%g\n",
                    batch, keyHeads, valueHeads, sequence, int(gateType), outOfPlace,
                    pass, graph, outputError, stateError);
        if (outOfPlace) Require(cudaMemcpy(st.cudaData, next.cudaData, st.GetBytes(),
                cudaMemcpyDeviceToDevice) == cudaSuccess, "GDN state continuation failed");
    }
    if (executable) cudaGraphExecDestroy(executable);
    if (captured) cudaGraphDestroy(captured);
}

static std::vector<uint16_t> ReadHalf(const Data &data) {
    std::vector<uint16_t> values(data.Count(0));
    Require(cudaMemcpy(values.data(), data.cudaData, values.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "GDN half copy failed");
    return values;
}

static void RunFusedOutput(int batch, int keyHeads, int valueHeads,
                           bool tiled, bool graph) {
    const int channels = (2 * keyHeads + valueHeads) * 128;
    std::vector<float> x(batch * channels), initial(batch * valueHeads * 128 * 128);
    std::vector<float> a(batch * valueHeads), beta(a.size()), z(a.size() * 128);
    std::vector<float> log(valueHeads), bias(valueHeads), norm(128);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(i * 1.37f) * .13f;
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = std::cos(i * .73f) * .02f;
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (int(i % 17) - 8) / 8.f;
        beta[i] = (int(i % 13) - 6) / 7.f;
    }
    for (size_t i = 0; i < z.size(); ++i) z[i] = std::sin(i * .731f) * 2.f;
    for (int i = 0; i < valueHeads; ++i) {
        log[i] = (i % 5 - 2) * .17f;
        bias[i] = (i % 7 - 3) * .11f;
    }
    for (int i = 0; i < 128; ++i) norm[i] = 1.f + std::sin(i * .13f) * .1f;
    Data q(FLOAT32), alpha(FLOAT16), b(FLOAT16), alog(FLOAT32), dt(FLOAT32);
    Data st(FLOAT32), referenceState(FLOAT32), core(FLOAT32), gate(FLOAT16);
    Data nw(FLOAT32), out(FLOAT16), reference(FLOAT16);
    Init(q, {batch, 1, channels}, x);
    Init(alpha, {batch, 1, valueHeads}, a); Init(b, alpha.dims, beta);
    Init(alog, {valueHeads}, log); Init(dt, {valueHeads}, bias);
    Init(st, {batch, valueHeads, 128, 128}, initial);
    Init(referenceState, st.dims, initial);
    Init(core, {batch * valueHeads, 128}, std::vector<float>(z.size()));
    Init(gate, core.dims, z); Init(nw, {128}, norm);
    Init(reference, core.dims, std::vector<float>(z.size()));
    Init(out, {batch, 1, valueHeads * 128}, std::vector<float>(z.size(), -9.f));
    auto fused = [&]() {
        return FastllmCudaQwen4GatedDeltaRuleDecodeOutput(q, alpha, b, alog, dt,
            nw, gate, st, out, keyHeads, valueHeads, 1e-6f, 1e-6f, tiled);
    };
    // Failed preconditions must leave the recurrent state available to fallback.
    Require(!FastllmCudaQwen4GatedDeltaRuleDecodeOutput(q, alpha, b, alog, dt,
        nw, gate, st, out, keyHeads, valueHeads, 1e-6f, 0.f, tiled),
        "fused GDN accepted invalid epsilon");
    Require(!FastllmCudaQwen4GatedDeltaRuleDecodeOutput(q, alpha, b, alog, dt,
        nw, gate, st, gate, keyHeads, valueHeads, 1e-6f, 1e-6f, tiled),
        "fused GDN accepted aliased gate/output");
    const auto rejectedState = Read(st);
    Require(std::memcmp(rejectedState.data(), initial.data(),
                        initial.size() * sizeof(float)) == 0,
            "rejected fused GDN modified state");
    cudaGraph_t captured = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal)
                == cudaSuccess, "GDN graph capture start failed");
        Require(fused(), "GDN graph fused call rejected");
        Require(cudaStreamEndCapture(cudaStreamPerThread, &captured) == cudaSuccess,
                "GDN graph capture end failed");
        Require(cudaGraphInstantiateWithFlags(&executable, captured, 0)
                == cudaSuccess, "GDN graph instantiate failed");
    }
    for (int token = 0; token < 32; ++token) {
        Require(FastllmCudaQwen4GatedDeltaRuleDecode(q, alpha, b, alog, dt,
            referenceState, core, keyHeads, valueHeads, 128, 128, 1e-6f),
            "reference GDN rejected");
        Require(FastllmCudaQwen4GdnOutputGateExact(core, nw, gate, reference, 1e-6f),
                "reference GDN gate rejected");
        if (graph) Require(cudaGraphLaunch(executable, cudaStreamPerThread) == cudaSuccess,
                           "GDN graph launch failed");
        else Require(fused(), "fused GDN rejected");
        const auto grouped = ReadHalf(reference);
        std::vector<uint16_t> expected(grouped.size());
        const int perKey = valueHeads / keyHeads;
        for (int item = 0; item < batch; ++item)
            for (int head = 0; head < valueHeads; ++head) {
                const int destination = tiled
                    ? (head % perKey) * keyHeads + head / perKey : head;
                std::copy_n(grouped.data() + (item * valueHeads + head) * 128, 128,
                            expected.data() + (item * valueHeads + destination) * 128);
            }
        Require(ReadHalf(out) == expected, "fused GDN output/layout differs bitwise");
        const auto actualState = Read(st), expectedState = Read(referenceState);
        Require(std::memcmp(actualState.data(), expectedState.data(),
                            actualState.size() * sizeof(float)) == 0,
                "fused GDN state differs bitwise");
    }
    if (executable) cudaGraphExecDestroy(executable);
    if (captured) cudaGraphDestroy(captured);
    std::printf("FUSED_PASS device=%d batch=%d key_heads=%d value_heads=%d tiled=%d graph=%d steps=32 state_exact=1 output_exact=1\n",
                FastllmCudaGetDevice(), batch, keyHeads, valueHeads, tiled, graph);
}

int main() {
    try {
        if (FastllmCudaGetDeviceCount() < 1) return 77;
        FastllmCudaSetDevice(0); SetThreads(2);
        for (DataType dtype : {FLOAT32, FLOAT16, BFLOAT16})
            for (bool outOfPlace : {false, true})
                for (int sequence : {1, 2, 3, 5, 9, 17, 33})
                    Run(sequence % 2 + 1, 4, 12, sequence, dtype, outOfPlace);
        Run(1, 16, 48, 5, FLOAT16, true);
        Run(8, 8, 32, 5, FLOAT32, false);
        for (DataType dtype : {FLOAT32, FLOAT16, BFLOAT16}) {
            for (bool outOfPlace : {false, true}) {
                for (int length : {63, 64, 65, 1024})
                    Run(2, 4, 12, length, dtype, outOfPlace, 3);
                Run(1, 16, 48, 4096, dtype, outOfPlace, 3);
            }
        }
        Run(2, 4, 12, 65, FLOAT16, true, 3, true);

        for (bool tiled : {false, true})
            for (bool graph : {false, true}) {
                RunFusedOutput(1, 16, 48, tiled, graph);
                RunFusedOutput(2, 8, 24, tiled, graph);
            }
        if (FastllmCudaGetDeviceCount() >= 2) {
            // Exercise the actual second device with the per-rank TP shape.
            FastllmCudaSetDevice(1);
            Run(1, 16, 48, 1024, FLOAT16, false, 3);
            RunFusedOutput(1, 8, 24, false, false);
            RunFusedOutput(1, 8, 24, true, true);
            FastllmCudaSetDevice(0);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
