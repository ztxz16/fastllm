#pragma once
#include <cstddef>
#include <string>

namespace fastllm::minimax_h3_image::detail {
// procRoot is injectable for CPU-only tests of host/container layouts.
size_t availableHostMemory(const std::string &procRoot = "/proc");
// Headroom for a resident quantized DiT, auxiliary weights and request workspaces.
size_t textCacheReserve(size_t quantizedBytes, size_t imageRows, size_t textRows, bool references);
} // namespace fastllm::minimax_h3_image::detail
