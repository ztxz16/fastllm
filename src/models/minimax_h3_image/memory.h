#pragma once
#include <cstddef>
#include <string>

namespace fastllm::minimax_h3_image::detail {
// procRoot is injectable for CPU-only tests of host/container layouts.
size_t availableHostMemory(const std::string &procRoot = "/proc");
} // namespace fastllm::minimax_h3_image::detail
