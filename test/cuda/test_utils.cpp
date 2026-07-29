#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace fastllm::cuda_test {

std::vector<float> MakeRandomFloats(size_t count, float minValue, float maxValue, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    std::vector<float> values(count);
    for (float &value : values) value = dist(rng);
    return values;
}

std::vector<uint8_t> MakeRandomInt4Weights(int outChannels, int inChannels, uint32_t seed) {
    Expect(outChannels > 0, "outChannels must be positive.");
    Expect(inChannels > 0, "inChannels must be positive.");
    Expect((inChannels & 1) == 0, "inChannels must be even for packed int4 weights.");
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    std::vector<uint8_t> qweight((size_t)outChannels * (inChannels / 2));
    for (int oc = 0; oc < outChannels; ++oc) {
        for (int ic = 0; ic < inChannels; ic += 2) {
            int hi = dist(rng), lo = dist(rng);
            qweight[(size_t)oc * (inChannels / 2) + ic / 2] =
                (uint8_t)((hi << 4) | lo);
        }
    }
    return qweight;
}

CompareResult CompareVectors(const std::vector<float> &expected,
                             const std::vector<float> &actual,
                             float maxAbsTol, float meanAbsTol, float maxRelTol) {
    Expect(expected.size() == actual.size(), "CompareVectors size mismatch.");
    CompareResult result;
    if (expected.empty()) {
        result.passed = true;
        return result;
    }
    double absSum = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        float absError = std::fabs(expected[i] - actual[i]);
        float relError = absError / std::max(std::fabs(expected[i]), 1.0e-2f);
        absSum += absError;
        if (absError > result.maxAbsError) {
            result.maxAbsError = absError;
            result.maxAbsIndex = i;
        }
        result.maxRelError = std::max(result.maxRelError, relError);
    }
    result.meanAbsError = (float)(absSum / expected.size());
    result.passed = result.maxAbsError <= maxAbsTol &&
                    result.meanAbsError <= meanAbsTol &&
                    result.maxRelError <= maxRelTol;
    return result;
}

void PrintCompareResult(const std::string &name, const CompareResult &result,
                        float maxAbsTol, float meanAbsTol, float maxRelTol) {
    std::printf("[%s] max_abs=%g/%g mean_abs=%g/%g max_rel=%g/%g max_abs_index=%zu %s\n",
                name.c_str(), result.maxAbsError, maxAbsTol,
                result.meanAbsError, meanAbsTol, result.maxRelError, maxRelTol,
                result.maxAbsIndex, result.passed ? "PASS" : "FAIL");
}

void Expect(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace fastllm::cuda_test
