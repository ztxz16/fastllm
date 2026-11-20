#pragma once
#include <string>

namespace fastllm::duo::live {
std::string referencePreset(int width, int height);
// RGB8 at the reply resolution -> Realization's packed BF16 reference tokens.
std::string encodeReference(const std::string &model, const std::string &rgb, int width, int height);
}
