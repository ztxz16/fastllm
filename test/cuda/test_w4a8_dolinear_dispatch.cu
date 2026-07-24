#include "test_utils.h"
#include "fastllm.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
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

struct W4A8DispatchCase {
    int n = 4;
    int m = 128;
    int k = 128;
    int groupCnt = 128;
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
        std::printf("[W4A8 DoCudaLinear] CUDA error: %s: %s\n", message, cudaGetErrorString(state));
        std::exit(1);
    }
}

bool RuntimeSm90() {
    return FastllmCudaRuntimeArch() == 90;
}

float RoundFp8E4M3(float value) {
    return (float)cutlass::float_e4m3_t(value);
}

std::vector<float> RoundFp16(const std::vector<float> &input) {
    std::vector<float> rounded(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        rounded[i] = __half2float(__float2half(input[i]));
    }
    return rounded;
}

std::vector<float> QuantizeActivationReference(const std::vector<float> &input,
                                               const W4A8DispatchCase &shape) {
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

void MakeSignedCompatibleScalesAndMins(size_t count, uint32_t seed,
                                       std::vector<float> &scales,
                                       std::vector<float> &mins) {
    scales = fastllm::cuda_test::MakeRandomFloats(count, 0.0005f, 0.02f, seed);
    mins.resize(count);
    for (size_t i = 0; i < count; ++i) {
        mins[i] = -8.0f * scales[i];
    }
}

std::vector<uint8_t> ConvertFastllmInt4ToUint4B8(
    const std::vector<uint8_t> &fastllmPacked) {
    std::vector<uint8_t> uint4b8(fastllmPacked.size());
    for (size_t i = 0; i < fastllmPacked.size(); ++i) {
        uint4b8[i] = (fastllmPacked[i] >> 4) | (fastllmPacked[i] << 4);
    }
    return uint4b8;
}

std::vector<float> RoundScalesToBFloat16(const std::vector<float> &scales) {
    std::vector<float> result(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        result[i] = __bfloat162float(__float2bfloat16(scales[i]));
    }
    return result;
}

std::vector<float> QuantizeScaleReference(const std::vector<float> &scales,
                                          const W4A8DispatchCase &shape) {
    int groups = shape.m / shape.groupCnt;
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

std::vector<float> CpuReference(const std::vector<float> &input,
                                const std::vector<uint8_t> &qweight,
                                const std::vector<float> &scales,
                                const std::vector<float> &bias,
                                const W4A8DispatchCase &shape) {
    int groups = shape.m / shape.groupCnt;
    std::vector<float> output((size_t)shape.n * shape.k, 0.0f);
    for (int token = 0; token < shape.n; ++token) {
        for (int oc = 0; oc < shape.k; ++oc) {
            float acc = bias[oc];
            for (int ic = 0; ic < shape.m; ++ic) {
                uint8_t packed = qweight[(size_t)oc * (shape.m / 2) + ic / 2];
                int q = (ic & 1) ? (packed & 0xF) : (packed >> 4);
                int signedQ = q - 8;
                int group = ic / shape.groupCnt;
                float weight = signedQ * scales[(size_t)oc * groups + group];
                acc += input[(size_t)token * shape.m + ic] * weight;
            }
            output[(size_t)token * shape.k + oc] = acc;
        }
    }
    return output;
}

fastllm::Data MakeInputData(const W4A8DispatchCase &shape,
                            const std::vector<float> &input) {
    fastllm::Data data(fastllm::DataType::FLOAT16, {shape.n, shape.m}, input);
    data.ToDevice(fastllm::DataDevice::CUDA);
    return data;
}

fastllm::Data MakeWeightData(const W4A8DispatchCase &shape,
                             const std::vector<uint8_t> &qweight,
                             const std::vector<float> &scales,
                             const std::vector<float> &mins) {
    fastllm::Data weight(fastllm::DataType::INT4_GROUP, {shape.k, shape.m});
    weight.groupCnt = shape.groupCnt;
    weight.group = shape.m / weight.groupCnt;
    weight.scales = scales;
    weight.mins = mins;
    weight.Allocate(false);
    std::memcpy(weight.cpuData, qweight.data(), std::min((size_t)weight.GetBytes(), qweight.size()));
    weight.ToDevice(fastllm::DataDevice::CUDA);
    return weight;
}

fastllm::Data MakeCompressedW4A8WeightData(
    const W4A8DispatchCase &shape,
    const std::vector<uint8_t> &qweight,
    const std::vector<float> &scales) {
    fastllm::Data weight(fastllm::DataType::INT4_W4A8, {shape.k, shape.m});
    weight.InitW4A8Weight(
        fastllm::W4A8WeightEncoding::COMPRESSED_TENSORS_UINT4B8);
    std::vector<uint16_t> bf16Scales(scales.size());
    for (size_t i = 0; i < scales.size(); ++i) {
        __nv_bfloat16 value = __float2bfloat16(scales[i]);
        std::memcpy(&bf16Scales[i], &value, sizeof(value));
    }
    weight.SetW4A8GroupScales(bf16Scales.data(), bf16Scales.size());
    std::vector<uint8_t> uint4b8 = ConvertFastllmInt4ToUint4B8(qweight);
    weight.Allocate(false);
    std::memcpy(weight.cpuData, uint4b8.data(), uint4b8.size());
    weight.ToDevice(fastllm::DataDevice::CUDA);
    return weight;
}

fastllm::Data MakeOutputData(const W4A8DispatchCase &shape) {
    fastllm::Data output(fastllm::DataType::FLOAT16, {shape.n, shape.k});
    output.Allocate(false);
    return output;
}

std::vector<float> DataToFloat(fastllm::Data &data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    size_t count = (size_t)data.Count(0);
    std::vector<float> result(count);
    const uint16_t *src = (const uint16_t*)data.cpuData;
    for (size_t i = 0; i < count; ++i) {
        half v;
        std::memcpy(&v, &src[i], sizeof(v));
        result[i] = __half2float(v);
    }
    return result;
}

bool RunFallbackCase(const W4A8DispatchCase &shape, uint32_t seedBase,
                     const std::string &name) {
#ifndef FASTLLM_ENABLE_VLLM_CUTLASS_W4A8
    (void)shape;
    (void)seedBase;
    std::printf("[W4A8 DoCudaLinear] %s compile macro disabled skip PASS\n", name.c_str());
    return true;
#else
    using namespace fastllm::cuda_test;
    int groups = shape.m / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.n * shape.m, -1.0f, 1.0f, seedBase + 1);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.k, shape.m, seedBase + 2);
    std::vector<float> scales;
    std::vector<float> mins;
    MakeSignedCompatibleScalesAndMins((size_t)shape.k * groups, seedBase + 3, scales, mins);
    std::vector<float> biasValues = MakeRandomFloats(shape.k, -0.1f, 0.1f, seedBase + 4);

    fastllm::Data inputData = MakeInputData(shape, input);
    fastllm::Data weightData = MakeWeightData(shape, qweight, scales, mins);
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.k}, biasValues);
    fastllm::Data outputData = MakeOutputData(shape);
    biasData.ToDevice(fastllm::DataDevice::CUDA);

    ScopedEnv gemmEnv("FASTLLM_CUDA_W4A8_ENABLE_GEMM", "1");
    ScopedEnv traceEnv("FASTLLM_CUDA_W4A8_TRACE", "1");
    ScopedEnv validateEnv("FASTLLM_CUDA_W4A8_VALIDATE", "1");
    fastllm::DoCudaLinear(inputData, weightData, biasData, outputData);
    CheckCuda(cudaDeviceSynchronize(), "sync DoCudaLinear fallback case");

    std::vector<float> expected = CpuReference(RoundFp16(input), qweight,
                                               RoundFp16(scales),
                                               RoundFp16(biasValues), shape);
    std::vector<float> actual = DataToFloat(outputData);

    constexpr float maxAbsTol = 2.5e-1f;
    constexpr float meanAbsTol = 3.0e-2f;
    constexpr float maxRelTol = 5.0e-1f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult(name, result, maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
#endif
}

bool RunNoSwitchCase() {
#ifndef FASTLLM_ENABLE_VLLM_CUTLASS_W4A8
    std::printf("[W4A8 DoCudaLinear] no-switch compile macro disabled skip PASS\n");
    return true;
#else
    using namespace fastllm::cuda_test;
    W4A8DispatchCase shape;
    int groups = shape.m / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.n * shape.m, -1.0f, 1.0f, 3301);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.k, shape.m, 3302);
    std::vector<float> scales;
    std::vector<float> mins;
    MakeSignedCompatibleScalesAndMins((size_t)shape.k * groups, 3303, scales, mins);
    std::vector<float> biasValues = MakeRandomFloats(shape.k, -0.1f, 0.1f, 3304);

    fastllm::Data inputData = MakeInputData(shape, input);
    fastllm::Data weightData = MakeWeightData(shape, qweight, scales, mins);
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.k}, biasValues);
    fastllm::Data outputData = MakeOutputData(shape);
    biasData.ToDevice(fastllm::DataDevice::CUDA);

    ScopedEnv gemmEnv("FASTLLM_CUDA_W4A8_ENABLE_GEMM", nullptr);
    ScopedEnv traceEnv("FASTLLM_CUDA_W4A8_TRACE", nullptr);
    ScopedEnv validateEnv("FASTLLM_CUDA_W4A8_VALIDATE", nullptr);
    fastllm::DoCudaLinear(inputData, weightData, biasData, outputData);
    CheckCuda(cudaDeviceSynchronize(), "sync DoCudaLinear no-switch case");

    std::vector<float> expected = CpuReference(RoundFp16(input), qweight,
                                               RoundFp16(scales),
                                               RoundFp16(biasValues), shape);
    std::vector<float> actual = DataToFloat(outputData);

    constexpr float maxAbsTol = 2.5e-1f;
    constexpr float meanAbsTol = 3.0e-2f;
    constexpr float maxRelTol = 5.0e-1f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult("W4A8 DoCudaLinear no-switch fallback", result,
                       maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
#endif
}

bool RunW4A8EnterCase() {
#ifndef FASTLLM_ENABLE_VLLM_CUTLASS_W4A8
    std::printf("[W4A8 DoCudaLinear] compile macro disabled skip PASS\n");
    return true;
#else
    if (!RuntimeSm90()) {
        std::printf("[W4A8 DoCudaLinear] runtime GPU is not SM90 skip PASS\n");
        return true;
    }

    using namespace fastllm::cuda_test;
    W4A8DispatchCase shape;
    int groups = shape.m / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.n * shape.m, -1.0f, 1.0f, 3001);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.k, shape.m, 3002);
    std::vector<float> scales;
    std::vector<float> mins;
    MakeSignedCompatibleScalesAndMins((size_t)shape.k * groups, 3003, scales, mins);
    std::vector<float> biasValues = MakeRandomFloats(shape.k, -0.1f, 0.1f, 3004);

    fastllm::Data inputData = MakeInputData(shape, input);
    fastllm::Data weightData = MakeCompressedW4A8WeightData(shape, qweight, scales);
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.k}, biasValues);
    fastllm::Data outputData = MakeOutputData(shape);
    biasData.ToDevice(fastllm::DataDevice::CUDA);

    ScopedEnv gemmEnv("FASTLLM_CUDA_W4A8_ENABLE_GEMM", "1");
    ScopedEnv traceEnv("FASTLLM_CUDA_W4A8_TRACE", "1");
    ScopedEnv validateEnv("FASTLLM_CUDA_W4A8_VALIDATE", "1");
    fastllm::DoCudaLinear(inputData, weightData, biasData, outputData);
    CheckCuda(cudaDeviceSynchronize(), "sync DoCudaLinear W4A8 enter case");

    std::vector<float> expectedInput = QuantizeActivationReference(RoundFp16(input), shape);
    std::vector<float> expectedScales = QuantizeScaleReference(
        RoundScalesToBFloat16(scales), shape);
    std::vector<float> expected = CpuReference(expectedInput, qweight, expectedScales, biasValues, shape);
    std::vector<float> actual = DataToFloat(outputData);

    constexpr float maxAbsTol = 2.5e-1f;
    constexpr float meanAbsTol = 3.0e-2f;
    constexpr float maxRelTol = 3.0e-1f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult("W4A8 DoCudaLinear enter case", result, maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
#endif
}

}  // namespace

int main() {
    try {
        bool ok = RunW4A8EnterCase();
        ok = RunFallbackCase({4, 128, 128, 64}, 3100,
                             "W4A8 DoCudaLinear fallback groupCnt=64") && ok;
        ok = RunFallbackCase({4, 96, 128, 32}, 3200,
                             "W4A8 DoCudaLinear fallback shape m=96") && ok;
        ok = RunNoSwitchCase() && ok;
        std::printf("[W4A8 DoCudaLinear] %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception &e) {
        std::printf("[W4A8 DoCudaLinear] exception: %s\n", e.what());
        return 1;
    } catch (...) {
        std::printf("[W4A8 DoCudaLinear] unknown exception\n");
        return 1;
    }
}
