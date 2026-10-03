#pragma once

namespace fastllm {
class Data;
}

// In-place FP16 residual epilogue for 1..8 input rows. A false result leaves
// output untouched, so the existing Linear + AddTo fallback remains valid.
// Supported weights: IQ3_S, IQ3_XXS, IQ4_XS, Q4_K, Q2_K, IQ2_S and IQ2_XS.
bool FastllmCudaGGUFLinearAdd(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
                              fastllm::Data &output);

// Internal extended-MMVQ entry; uses the same activation quantization and
// reduction as the ordinary FP16 MMVQ path.
bool FastllmCudaHalfMatMulGGUFMMVQAddTo(const void *input, const void *weight, void *output, int weightType,
                                        int rows, int columns, int outputRows, void *stream);
