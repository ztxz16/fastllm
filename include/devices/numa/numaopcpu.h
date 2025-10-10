//
// NUMA-aware CPU operator interface
// Complete support for all instruction sets and data formats
//

#ifndef FASTLLM_NUMAOPCPU_H
#define FASTLLM_NUMAOPCPU_H

#ifdef USE_NUMA

#include "fastllm.h"
#include "devices/numa/numathreadpool.h"

namespace fastllm {

    // Check if NUMA should be activated
    bool ShouldActivateNuma();

    // Initialize NUMA system
    void InitializeNumaIfNeeded();

    // ===========================================
    // NUMA-aware Linear Operations
    // ===========================================

    // Float32 x Float32
    void RunNumaLinearFloat32Float32(float *inputData, float *weightData,
                                    float *outputData, float *biasData,
                                    int n, int m, int k);

    // Float32 x Float16
    void RunNumaLinearFloat32Float16(float *inputData, uint16_t *weightData,
                                    float *outputData, float *biasData,
                                    int n, int m, int k);

    // Float32 x BFloat16
    void RunNumaLinearFloat32BFloat16(float *inputData, uint16_t *weightData,
                                     float *outputData, float *biasData,
                                     int n, int m, int k);

    // INT8 quantized
    void RunNumaLinearInt8(float *inputData, Data &weight,
                          float *outputData, float *biasData,
                          int n, int m, int k);

    // INT4 group quantized
    void RunNumaLinearInt4Group(float *inputData, Data &weight,
                               float *outputData, float *biasData,
                               int n, int m, int k, int group, int groupCnt);

    // INT4 no-zero quantized
    void RunNumaLinearInt4NoZero(float *inputData, Data &weight,
                                float *outputData, float *biasData,
                                int n, int m, int k);

    // GGUF format
    void RunNumaLinearGGUF(float *inputData, Data &weight,
                          float *outputData, float *biasData,
                          int n, int m, int k);

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMAOPCPU_H
