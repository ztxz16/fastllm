#pragma once
namespace fastllm { class Data; }

// Bias-free projections of the same resident, contiguous input. Outputs are
// preallocated and may have different widths and floating-point storage types.
// Each backend chooses grouped kernels or shared activation preparation.
// Unsupported layouts return false before writes; ordinary Linear is the fallback.
bool FastllmCudaLinearShared(const fastllm::Data &input, fastllm::Data *const *weights,
                            fastllm::Data *const *outputs, int count);
