#ifndef FASTLLM_CUDA_VISION_H
#define FASTLLM_CUDA_VISION_H
#include "fastllm.h"

// Dense, bias-free vision TP partial projection. Both operands use the
// same floating type; accumulation and output remain FP32 until reduction.
bool FastllmCudaVisionLinearFloat32(const fastllm::Data &input,
                                    const fastllm::Data &weight,
                                    fastllm::Data &output);
// Convert a reduced vision projection with the round-to-nearest-even mode
// used by native half GEMV and tensor-core GEMM outputs. Generic ToDataType
// intentionally truncates toward zero and is not equivalent to this epilogue.
bool FastllmCudaVisionFloat32ToHalf(const fastllm::Data &input,
                                  fastllm::Data &output);
#endif
