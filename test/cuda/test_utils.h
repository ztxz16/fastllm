#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm::cuda_test {

struct CompareResult {
    float maxAbsError = 0.0f;
    float meanAbsError = 0.0f;
    float maxRelError = 0.0f;
    size_t maxAbsIndex = 0;
    bool passed = false;
};

std::vector<float> MakeRandomFloats(size_t count, float minValue, float maxValue, uint32_t seed);
std::vector<uint8_t> MakeRandomInt4Weights(int outChannels, int inChannels, uint32_t seed);
CompareResult CompareVectors(const std::vector<float> &expected,
                             const std::vector<float> &actual,
                             float maxAbsTol, float meanAbsTol, float maxRelTol);
void PrintCompareResult(const std::string &name, const CompareResult &result,
                        float maxAbsTol, float meanAbsTol, float maxRelTol);
void Expect(bool condition, const std::string &message);

}  // namespace fastllm::cuda_test
