#pragma once

#include <cstdint>

namespace fastllm {
namespace distributed {
// Attention 的共享概率缓冲及连续 KV 使用同一容量；不放宽显存预算。
constexpr uint32_t kQwen3MaxContextTokens = 2048;
constexpr uint32_t kQwen3MaxPromptTokens = kQwen3MaxContextTokens;
}
}
