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
    data.ToDevice(DataDevice::CUDA, {0}, true);
}

static std::vector<float> Read(const Data &data) {
    std::vector<float> values(data.Count(0));
    Require(cudaMemcpy(values.data(), data.cudaData, values.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess, "GDN copy failed");
    return values;
}

static float Compare(const std::vector<float> &actual,
                     const std::vector<float> &expected, const char *message) {
    Require(actual.size() == expected.size(), message);
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::abs(actual[i] - expected[i]);
        // The established out-of-place and in-place kernels can differ by
        // a few FP32 ULPs due to contraction of the state update expression.
        Require(std::isfinite(actual[i]) &&
                error <= 1e-7f + 1e-6f * std::abs(expected[i]), message);
        maximum = std::max(maximum, error);
    }
    return maximum;
}

static void Run(int batch, int keyHeads, int valueHeads, int sequence,
                DataType gateType, bool outOfPlace) {
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
    Require(FastllmCudaQwen4GatedDeltaRuleDecode(q, alpha, b, alog, dt, st, output,
        keyHeads, valueHeads, 128, 128, 1e-6f, outOfPlace ? &next : nullptr),
        "GDN sequence rejected");
    const auto actual = Read(output);
    if (outOfPlace) Require(Read(st) == state, "GDN modified input state");

    // Repeated one-token calls are independent of the sequence dispatch and
    // check every intermediate output as well as the final recurrent state.
    std::vector<float> expected(actual.size());
    for (int token = 0; token < sequence; ++token) {
        std::vector<float> sx(batch * channels), sa(batch * valueHeads), sb(sa.size());
        for (int item = 0; item < batch; ++item) {
            std::copy_n(x.data() + (item * sequence + token) * channels,
                        channels, sx.data() + item * channels);
            std::copy_n(a.data() + (item * sequence + token) * valueHeads,
                        valueHeads, sa.data() + item * valueHeads);
            std::copy_n(beta.data() + (item * sequence + token) * valueHeads,
                        valueHeads, sb.data() + item * valueHeads);
        }
        Data sq(FLOAT32), salpha(gateType), sbet(gateType), so(FLOAT32);
        Init(sq, {batch, 1, channels}, sx);
        Init(salpha, {batch, 1, valueHeads}, sa);
        Init(sbet, {batch, 1, valueHeads}, sb);
        Init(so, {batch, 1, valueHeads, 128}, std::vector<float>(batch * valueHeads * 128));
        Require(FastllmCudaQwen4GatedDeltaRuleDecode(sq, salpha, sbet, alog, dt,
            referenceState, so, keyHeads, valueHeads, 128, 128, 1e-6f),
            "GDN single token rejected");
        auto row = Read(so);
        for (int item = 0; item < batch; ++item)
            std::copy_n(row.data() + item * valueHeads * 128, valueHeads * 128,
                        expected.data() + (item * sequence + token) * valueHeads * 128);
    }
    const float outputError = Compare(actual, expected,
        "GDN sequence output differs from single tokens");
    const float stateError = Compare(Read(outOfPlace ? next : st), Read(referenceState),
        "GDN sequence final state differs from single tokens");
    std::printf("PASS batch=%d key_heads=%d value_heads=%d sequence=%d dtype=%d out_of_place=%d output_error=%g state_error=%g\n",
                batch, keyHeads, valueHeads, sequence, int(gateType), outOfPlace,
                outputError, stateError);
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
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
