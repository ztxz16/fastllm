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

// Hopper W8A16 decode. Layouts/scratch are prepared only in synchronized model
// warmup; unsupported shapes, devices and unprepared workers use the generic path.
bool FastllmCudaPerChannelW8A16(const void *input, fastllm::Data &weight, void *output,
                              int rows, int inner, int channels);
void FastllmCudaReleasePerChannelW8A16(fastllm::Data &weight);
// Reserve the owned layout and one unbound worker scratch slot before KV sizing.
// Unsupported architectures/formats are a no-op and keep their generic fallback.
bool FastllmCudaPreparePerChannelW8A16(fastllm::Data &weight);

// Hopper native row-wise block32 W8A8, keeping original FP32 weight scales.
// Unsupported devices/alignment return false without writing output. Scratch
// reuses the existing bounded Linear lease, retained by captured graphs.
bool FastllmCudaBlock32W8A8SM90(const void *input, const void *weight, const float *scales,
                              void *output, int rows, int inner, int channels);

// Blackwell uses the same BF16 Linear lease and original FP8 weight as GEMV.
bool FastllmCudaMxfp8LinearSM103(const void *input, fastllm::Data &weight, void *output,
                                 int rows, int inner, int channels);
void FastllmCudaRegisterMxfp8Scales(fastllm::Data &weight, void *allocation);
bool FastllmCudaHasExactMxfp8Scales(const void *allocation, int device);

constexpr size_t FastllmCompactMxfp8Slot = 8; // Slots 0–7 hold legacy FP8 layouts.
const void *FastllmCudaCompactMxfp8(const fastllm::Data &weight);

// Graph-safe attention projection admission. Return borrowed K32 scales only when
// the unfused Linear would also use block-quantized activations on this worker.
const float *FastllmCudaVerifyMxfp8Scales(const fastllm::Data &weight, int channels, int inner, int device);

// Small-row SWA projection; false leaves the residual and scratch outputs untouched.
bool FastllmCudaMxfp8SwaQkv(void *hidden, const void *branch, const float *gamma, float eps,
                            const fastllm::Data &weight, const float *scales, void *output, int rows,
                            const float *positions, float theta, void *rotary);
