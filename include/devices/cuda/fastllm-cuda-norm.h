#pragma once

// Raw CUDA overloads with a fixed 256-lane FP32 reduction. Default stream:
// cudaStreamPerThread. The Data API retains its existing reduction selection.
// Selected 16-bit input/output, FP32 norm weights. The optional gate is SiLU-gated
// before the single output cast, as required by the GDN output norm.
bool FastllmCudaRMSNorm(const void *input, const float *weight, const void *gate,
    void *output, int rows, int width, float eps, bool zeroCentered,
    void *stream = nullptr, bool fp16 = false);
// Update 16-bit residual += input before normalization, including its
// residual rounding. Output must not overlap either input allocation.
bool FastllmCudaAddRMSNorm(void *residual, const void *input, const float *weight,
    void *output, int rows, int width, float eps, bool zeroCentered,
    void *stream = nullptr, bool fp16 = false);
