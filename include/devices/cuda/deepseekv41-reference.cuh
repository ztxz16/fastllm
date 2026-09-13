#pragma once
#include "fastllm.h"
extern "C" {
bool FastllmCudaV41ReferenceRotary(fastllm::Data &, int, float, int, int, bool, int, float, int, int, int, int,
                                   int);
bool FastllmCudaV41ReferenceEngram(fastllm::Data &, const fastllm::Data &, fastllm::Data &, fastllm::Data &,
                                   const fastllm::Data *, float, float);
bool FastllmCudaV41ReferenceNorm(const fastllm::Data &, fastllm::Data &, float, fastllm::Data &);
bool FastllmCudaV41ReferencePreNorm(const fastllm::Data &, const fastllm::Data &, fastllm::Data &, float,
                                    fastllm::Data &);
bool FastllmCudaV41ReferenceMix(const fastllm::Data &, fastllm::Data &, fastllm::Data &, fastllm::Data &, int,
                                int, float, float, fastllm::Data &, fastllm::Data &, fastllm::Data &);
bool FastllmCudaV41ReferenceLinear(const fastllm::Data &, fastllm::Data &, fastllm::Data &, int, bool);
bool FastllmCudaV41ReferenceAttention(const fastllm::Data &, const fastllm::Data &, const fastllm::Data *,
                                      const fastllm::Data *, const fastllm::Data *, fastllm::Data &, int, int,
                                      float, fastllm::Data &);
}
