#pragma once
#include <cstddef>
namespace fastllm {
class Data;
}
// Owned block-32 BF16 FP8 layouts; all other quantization caches are untouched.
extern "C" void FastllmCudaCacheBFloat16FP8Weight(fastllm::Data &weight, const void *dequantized,
                                                  size_t bytes);
extern "C" void FastllmCudaReleaseBFloat16FP8Auxiliary(fastllm::Data &weight);

// Owned native FP8 output-channel slice. Cached with its source weight and
// released on source mutation/destruction; creation is forbidden in capture.
fastllm::Data *FastllmCudaFP8RowShard(fastllm::Data &weight, int start, int count);
// Preserve complete block-32 groups; zero-pad each local intermediate width
// to 64. Paired rows select matching channels from the two matrix halves.
fastllm::Data *FastllmCudaFP8PairedRowShard(fastllm::Data &weight, int start, int count);
fastllm::Data *FastllmCudaFP8ColumnShard(fastllm::Data &weight, int start, int count);

// Check numerical eligibility without allocating or depending on warmup state.
extern "C" bool FastllmCudaCanUseMarlinBFloat16FP8(const fastllm::Data &weight,
    int rows, int inner, int columns);
// Prepare the immutable native MXFP8 layout before graph capture.
extern "C" bool FastllmCudaPrepareMarlinBFloat16FP8(fastllm::Data &weight,
    int rows, int inner, int columns);

// Immutable, exactly representable FP32 gate copy; source Data owns its lifetime.
fastllm::Data *FastllmCudaExactRouterBFloat16Weight(fastllm::Data &weight);
