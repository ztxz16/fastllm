#pragma once
#include <cstddef>
namespace fastllm {
class Data;
}
// Owned block-32 BF16 FP8 layouts; all other quantization caches are untouched.
extern "C" void FastllmCudaCacheBFloat16FP8Weight(fastllm::Data &weight, const void *dequantized,
                                                  size_t bytes);
extern "C" void FastllmCudaReleaseBFloat16FP8Auxiliary(fastllm::Data &weight);
