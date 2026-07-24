#include "test_utils.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "cutlass/float8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

struct W4A8Case {
    int n = 1;
    int m = 128;
    int k = 128;
    bool bf16Input = false;
    bool withBias = false;
};

class ScopedEnv {
public:
    ScopedEnv(const char *key, const char *value) : key(key) {
        const char *old = std::getenv(key);
        if (old != nullptr) {
            hadOldValue = true;
            oldValue = old;
        }
        if (value == nullptr) {
            unsetenv(key);
        } else {
            setenv(key, value, 1);
        }
    }

    ~ScopedEnv() {
        if (hadOldValue) {
            setenv(key.c_str(), oldValue.c_str(), 1);
        } else {
            unsetenv(key.c_str());
        }
    }

private:
    std::string key;
    bool hadOldValue = false;
    std::string oldValue;
};

void CheckCuda(cudaError_t state, const char *message) {
    if (state != cudaSuccess) {
        std::printf("[W4A8 CUTLASS] CUDA error: %s: %s\n", message, cudaGetErrorString(state));
        std::exit(1);
    }
}

bool RuntimeSm90() {
    int arch = FastllmCudaRuntimeArch();
    return arch == 90;
}

std::vector<float> DataToFloat(fastllm::Data &data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    size_t count = (size_t)data.Count(0);
    std::vector<float> result(count);
    const uint16_t *src = (const uint16_t*)data.cpuData;
    for (size_t i = 0; i < count; ++i) {
        if (data.dataType == fastllm::DataType::FLOAT16) {
            half v;
            std::memcpy(&v, &src[i], sizeof(v));
            result[i] = __half2float(v);
        } else if (data.dataType == fastllm::DataType::BFLOAT16) {
            __nv_bfloat16 v;
            std::memcpy(&v, &src[i], sizeof(v));
            result[i] = __bfloat162float(v);
        } else if (data.dataType == fastllm::DataType::FLOAT32) {
            const float *fp = (const float*)data.cpuData;
            result[i] = fp[i];
        } else {
            fastllm::cuda_test::Expect(false, "unsupported output dtype.");
        }
    }
    return result;
}

std::vector<float> RoundInputForDtype(const std::vector<float> &input, fastllm::DataType dtype) {
    std::vector<float> rounded(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (dtype == fastllm::DataType::FLOAT16) {
            rounded[i] = __half2float(__float2half(input[i]));
        } else if (dtype == fastllm::DataType::BFLOAT16) {
            rounded[i] = __bfloat162float(__float2bfloat16(input[i]));
        } else {
            rounded[i] = input[i];
        }
    }
    return rounded;
}

float RoundFp8E4M3(float value) {
    return (float)cutlass::float_e4m3_t(value);
}

std::vector<float> QuantizeActivationReference(const std::vector<float> &input,
                                               const W4A8Case &shape) {
    std::vector<float> result(input.size());
    for (int token = 0; token < shape.n; ++token) {
        size_t rowOffset = (size_t)token * shape.m;
        float absMax = 0.0f;
        for (int ic = 0; ic < shape.m; ++ic) {
            absMax = std::max(absMax, std::fabs(input[rowOffset + ic]));
        }
        float scale = std::max(absMax, 1.0e-10f) * (1.0f / 448.0f);
        for (int ic = 0; ic < shape.m; ++ic) {
            float q = std::max(-448.0f, std::min(448.0f, input[rowOffset + ic] / scale));
            result[rowOffset + ic] = RoundFp8E4M3(q) * scale;
        }
    }
    return result;
}

std::vector<float> QuantizeScaleReference(const std::vector<float> &scales,
                                          const W4A8Case &shape) {
    int groupCnt = 128;
    int groups = shape.m / groupCnt;
    std::vector<float> result(scales.size());
    for (int oc = 0; oc < shape.k; ++oc) {
        float channelAbsMax = 0.0f;
        size_t rowOffset = (size_t)oc * groups;
        for (int group = 0; group < groups; ++group) {
            channelAbsMax = std::max(channelAbsMax, std::fabs(scales[rowOffset + group]));
        }
        float channelScale = std::max(channelAbsMax, 1.0e-10f) / 56.0f;
        for (int group = 0; group < groups; ++group) {
            result[rowOffset + group] = RoundFp8E4M3(scales[rowOffset + group] / channelScale) * channelScale;
        }
    }
    return result;
}

std::vector<uint8_t> MakeSignedCompatibleInt4Weights(int outChannels, int inChannels, uint32_t seed) {
    return fastllm::cuda_test::MakeRandomInt4Weights(outChannels, inChannels, seed);
}

std::vector<uint8_t> ConvertFastllmInt4ToUint4B8(
    const std::vector<uint8_t> &fastllmPacked) {
    std::vector<uint8_t> uint4b8(fastllmPacked.size());
    for (size_t i = 0; i < fastllmPacked.size(); ++i) {
        uint4b8[i] = (fastllmPacked[i] >> 4) | (fastllmPacked[i] << 4);
    }
    return uint4b8;
}

std::vector<uint16_t> ConvertScalesToBFloat16(
    const std::vector<float> &scales) {
    std::vector<uint16_t> result(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        __nv_bfloat16 value = __float2bfloat16(scales[i]);
        std::memcpy(&result[i], &value, sizeof(value));
    }
    return result;
}

std::vector<float> RoundScalesToBFloat16(const std::vector<float> &scales) {
    std::vector<float> result(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        result[i] = __bfloat162float(__float2bfloat16(scales[i]));
    }
    return result;
}

void MakeSignedCompatibleScalesAndMins(size_t count, uint32_t seed,
                                       std::vector<float> &scales,
                                       std::vector<float> &mins) {
    scales = fastllm::cuda_test::MakeRandomFloats(count, 0.0005f, 0.02f, seed);
    mins.resize(count);
    for (size_t i = 0; i < count; ++i) {
        mins[i] = -8.0f * scales[i];
    }
}

std::vector<float> CpuReference(const std::vector<float> &input,
                                const std::vector<uint8_t> &qweight,
                                const std::vector<float> &scales,
                                const std::vector<float> *bias,
                                const W4A8Case &shape) {
    int groupCnt = 128;
    int groups = shape.m / groupCnt;
    std::vector<float> output((size_t)shape.n * shape.k, 0.0f);
    for (int token = 0; token < shape.n; ++token) {
        for (int oc = 0; oc < shape.k; ++oc) {
            float acc = bias == nullptr ? 0.0f : (*bias)[oc];
            for (int ic = 0; ic < shape.m; ++ic) {
                uint8_t packed = qweight[(size_t)oc * (shape.m / 2) + ic / 2];
                int q = (ic & 1) ? (packed & 0xF) : (packed >> 4);
                int signedQ = q - 8;
                int group = ic / groupCnt;
                float weight = signedQ * scales[(size_t)oc * groups + group];
                acc += input[(size_t)token * shape.m + ic] * weight;
            }
            output[(size_t)token * shape.k + oc] = acc;
        }
    }
    return output;
}

fastllm::Data MakeInputData(const W4A8Case &shape, const std::vector<float> &input) {
    fastllm::Data data(shape.bf16Input ? fastllm::DataType::BFLOAT16 : fastllm::DataType::FLOAT16,
                       {shape.n, shape.m}, input);
    data.ToDevice(fastllm::DataDevice::CUDA);
    return data;
}

fastllm::Data MakeWeightData(const W4A8Case &shape,
                             fastllm::DataType dtype,
                             const std::vector<uint8_t> &qweight,
                             const std::vector<float> &scales,
                             const std::vector<float> &mins,
                             int groupCnt = 128) {
    fastllm::Data weight(dtype, {shape.k, shape.m});
    if (dtype == fastllm::DataType::INT4_GROUP) {
        weight.groupCnt = groupCnt;
        weight.group = (shape.m + groupCnt - 1) / groupCnt;
        weight.scales = scales;
        weight.mins = mins;
    } else if (dtype == fastllm::DataType::INT4_W4A8) {
        weight.InitW4A8Weight(
            fastllm::W4A8WeightEncoding::COMPRESSED_TENSORS_UINT4B8);
        std::vector<uint16_t> bf16Scales = ConvertScalesToBFloat16(scales);
        weight.SetW4A8GroupScales(bf16Scales.data(), bf16Scales.size());
    }
    weight.Allocate(false);
    std::vector<uint8_t> uint4b8;
    const std::vector<uint8_t> *packedSource = &qweight;
    if (dtype == fastllm::DataType::INT4_W4A8) {
        uint4b8 = ConvertFastllmInt4ToUint4B8(qweight);
        packedSource = &uint4b8;
    }
    std::memcpy(weight.cpuData, packedSource->data(),
                std::min((size_t)weight.GetBytes(), packedSource->size()));
    weight.ToDevice(fastllm::DataDevice::CUDA);
    return weight;
}

fastllm::Data MakeOutputData(const W4A8Case &shape) {
    fastllm::Data output(shape.bf16Input ? fastllm::DataType::BFLOAT16 : fastllm::DataType::FLOAT16,
                         {shape.n, shape.k});
    output.Allocate(false);
    return output;
}

bool TryCase(W4A8Case shape,
             fastllm::DataType inputTypeOverride,
             fastllm::DataType weightTypeOverride,
             const fastllm::Data *biasOverride,
             int groupCntOverride,
             bool enableGemm) {
    int groups = (shape.m + 127) / 128;
    std::vector<float> input = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.n * shape.m, -1.0f, 1.0f, 1001);
    std::vector<uint8_t> qweight = MakeSignedCompatibleInt4Weights(shape.k, shape.m, 1002);
    std::vector<float> scales;
    std::vector<float> mins;
    MakeSignedCompatibleScalesAndMins((size_t)shape.k * groups, 1003, scales, mins);
    std::vector<float> biasValues = fastllm::cuda_test::MakeRandomFloats(shape.k, -0.1f, 0.1f, 1004);

    fastllm::Data inputData(inputTypeOverride, {shape.n, shape.m}, input);
    inputData.ToDevice(fastllm::DataDevice::CUDA);
    fastllm::Data weightData = MakeWeightData(shape, weightTypeOverride, qweight, scales, mins, groupCntOverride);
    fastllm::Data outputData(inputTypeOverride, {shape.n, shape.k});
    outputData.Allocate(false);

    fastllm::Data emptyBias;
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.k}, biasValues);
    if (shape.withBias) {
        biasData.ToDevice(fastllm::DataDevice::CUDA);
    }
    const fastllm::Data &bias = biasOverride != nullptr ? *biasOverride : (shape.withBias ? biasData : emptyBias);

    ScopedEnv validateEnv("FASTLLM_CUDA_W4A8_VALIDATE", "1");
    ScopedEnv traceEnv("FASTLLM_CUDA_W4A8_TRACE", "1");
    ScopedEnv gemmEnv("FASTLLM_CUDA_W4A8_ENABLE_GEMM", enableGemm ? "1" : nullptr);
    return TryCudaCutlassW4A8(inputData, weightData, bias, outputData, shape.n, shape.m, shape.k);
}

bool RunEntryBehaviorTests() {
    using fastllm::cuda_test::Expect;

    W4A8Case valid;
    bool ok = true;

#ifndef FASTLLM_ENABLE_VLLM_CUTLASS_W4A8
    ok = !TryCase(valid, fastllm::DataType::FLOAT16, fastllm::DataType::INT4_GROUP,
                  nullptr, 128, true) && ok;
    std::printf("[W4A8 CUTLASS] compile macro disabled skip PASS\n");
    return ok;
#else
    if (!RuntimeSm90()) {
        ok = !TryCase(valid, fastllm::DataType::FLOAT16, fastllm::DataType::INT4_GROUP,
                      nullptr, 128, true) && ok;
        std::printf("[W4A8 CUTLASS] non-SM90 skip %s\n", ok ? "PASS" : "FAIL");
    }

    W4A8Case badShape = valid;
    badShape.m = 96;
    ok = !TryCase(badShape, fastllm::DataType::FLOAT16, fastllm::DataType::INT4_GROUP,
                  nullptr, 128, true) && ok;
    std::printf("[W4A8 CUTLASS] m/k non-128-aligned skip checked\n");

    ok = !TryCase(valid, fastllm::DataType::FLOAT32, fastllm::DataType::INT4_GROUP,
                  nullptr, 128, true) && ok;
    std::printf("[W4A8 CUTLASS] input dtype skip checked\n");

    ok = !TryCase(valid, fastllm::DataType::FLOAT16, fastllm::DataType::FLOAT16,
                  nullptr, 128, true) && ok;
    std::printf("[W4A8 CUTLASS] weight dtype skip checked\n");

    std::vector<float> badBiasValues(valid.k, 0.0f);
    fastllm::Data badBias(fastllm::DataType::FLOAT16, {valid.k}, badBiasValues);
    badBias.ToDevice(fastllm::DataDevice::CUDA);
    ok = !TryCase(valid, fastllm::DataType::FLOAT16, fastllm::DataType::INT4_GROUP,
                  &badBias, 128, true) && ok;
    std::printf("[W4A8 CUTLASS] bias dtype skip checked\n");

    ok = !TryCase(valid, fastllm::DataType::FLOAT16, fastllm::DataType::INT4_GROUP,
                  nullptr, 64, true) && ok;
    std::printf("[W4A8 CUTLASS] groupCnt skip checked\n");

    Expect(ok, "W4A8 entry behavior tests failed.");
    return ok;
#endif
}

bool RunNumericalCase(const W4A8Case &shape, uint32_t seedBase, const std::string &name) {
    using namespace fastllm::cuda_test;

    int groups = shape.m / 128;
    std::vector<float> input = MakeRandomFloats((size_t)shape.n * shape.m, -1.0f, 1.0f, seedBase + 1);
    std::vector<uint8_t> qweight = MakeSignedCompatibleInt4Weights(shape.k, shape.m, seedBase + 2);
    std::vector<float> scales;
    std::vector<float> mins;
    MakeSignedCompatibleScalesAndMins((size_t)shape.k * groups, seedBase + 3, scales, mins);
    std::vector<float> biasValues = MakeRandomFloats(shape.k, -0.1f, 0.1f, seedBase + 4);

    fastllm::Data inputData = MakeInputData(shape, input);
    fastllm::Data weightData = MakeWeightData(shape, fastllm::DataType::INT4_W4A8,
                                              qweight, scales, mins);
    fastllm::Data outputData = MakeOutputData(shape);
    fastllm::Data emptyBias;
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.k}, biasValues);
    if (shape.withBias) {
        biasData.ToDevice(fastllm::DataDevice::CUDA);
    }
    fastllm::Data &bias = shape.withBias ? biasData : emptyBias;

    ScopedEnv validateEnv("FASTLLM_CUDA_W4A8_VALIDATE", "1");
    ScopedEnv traceEnv("FASTLLM_CUDA_W4A8_TRACE", "1");
    ScopedEnv gemmEnv("FASTLLM_CUDA_W4A8_ENABLE_GEMM", "1");
    bool usedW4A8 = TryCudaCutlassW4A8(inputData, weightData, bias, outputData,
                                       shape.n, shape.m, shape.k);
    Expect(usedW4A8, "TryCudaCutlassW4A8 returned false for numerical case.");
    CheckCuda(cudaDeviceSynchronize(), "sync TryCudaCutlassW4A8");

    std::vector<float> expectedInput = RoundInputForDtype(
        input, shape.bf16Input ? fastllm::DataType::BFLOAT16 : fastllm::DataType::FLOAT16);
    expectedInput = QuantizeActivationReference(expectedInput, shape);
    std::vector<float> expectedScales = QuantizeScaleReference(
        RoundScalesToBFloat16(scales), shape);
    const std::vector<float> *biasPtr = shape.withBias ? &biasValues : nullptr;
    std::vector<float> expected = CpuReference(expectedInput, qweight, expectedScales, biasPtr, shape);
    std::vector<float> actual = DataToFloat(outputData);

    constexpr float maxAbsTol = 2.5e-1f;
    constexpr float meanAbsTol = 3.0e-2f;
    const float maxRelTol = shape.bf16Input ? 5.0f : 3.0e-1f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult(name, result, maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
}

bool RunNumericalTests() {
#ifndef FASTLLM_ENABLE_VLLM_CUTLASS_W4A8
    std::printf("[W4A8 CUTLASS] numerical tests skipped: FASTLLM_ENABLE_VLLM_CUTLASS_W4A8 is off\n");
    return true;
#else
    if (!RuntimeSm90()) {
        std::printf("[W4A8 CUTLASS] numerical tests skipped: runtime GPU is not SM90\n");
        return true;
    }

    std::vector<W4A8Case> cases = {
        {1, 128, 128, false, false},
        {4, 128, 128, false, true},
        {8, 256, 128, true, false},
        {16, 256, 256, true, true},
        {64, 128, 128, false, false},
        {128, 128, 256, false, true},
    };

    bool ok = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        const W4A8Case &shape = cases[i];
        std::string name = "W4A8 output vs CPU baseline n=" + std::to_string(shape.n) +
                           " m=" + std::to_string(shape.m) +
                           " k=" + std::to_string(shape.k) +
                           (shape.bf16Input ? " bf16" : " fp16") +
                           (shape.withBias ? " bias" : " no_bias");
        ok = RunNumericalCase(shape, 2000 + (uint32_t)i * 20, name) && ok;
    }
    return ok;
#endif
}

}  // namespace

int main() {
    try {
        bool ok = RunEntryBehaviorTests();
        ok = RunNumericalTests() && ok;
        std::printf("[W4A8 CUTLASS] %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception &e) {
        std::printf("[W4A8 CUTLASS] exception: %s\n", e.what());
        return 1;
    }
}
