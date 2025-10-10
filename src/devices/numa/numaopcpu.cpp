//
// NUMA-aware CPU operator implementations
// Complete implementation supporting all instruction sets and data formats
//

#ifdef USE_NUMA

#include "devices/numa/numaopcpu.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"
#include <cstring>
#include <cstdlib>
#include <iostream>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#ifdef __AVX2__
#include "immintrin.h"
#include "avxMath.h"
#endif

#include "gguf.h"

namespace fastllm {

    extern FP16ToFP32Manager fp16tofp32;
    extern BF16ToFP32Manager bf16tofp32;
    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);
    extern void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);

    // NUMA-aware Float32 x Float32 linear operation with full SIMD support
    void RunNumaLinearFloat32Float32(float *inputData, float *weightData,
                                    float *outputData, float *biasData,
                                    int n, int m, int k) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        // Use do_k_work_stealing_job for NUMA-aware distribution
        pool.DoNumaWork(1, k,
            nullptr,  // init_func
            [&](int task_id) {
                int j = task_id;
                for (int i = 0; i < n; i++) {
                    float sum = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    
#ifdef __aarch64__
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    for (; l + 3 < m; l += 4) {
                        float32x4_t vi = vld1q_f32(inputData + i * m + l);
                        float32x4_t vw = vld1q_f32(weightData + j * m + l);
                        vsum = vmlaq_f32(vsum, vi, vw);
                    }
                    sum += vaddvq_f32(vsum);
#elif defined(__AVX2__)
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    sum += Floatsum(vsum);
#endif
                    for (; l < m; l++) {
                        sum += inputData[i * m + l] * weightData[j * m + l];
                    }
                    outputData[i * k + j] = sum;
                }
            },
            nullptr  // finalize_func
        );
    }

    // NUMA-aware Float32 x Float16 linear
    void RunNumaLinearFloat32Float16(float *inputData, uint16_t *weightData,
                                    float *outputData, float *biasData,
                                    int n, int m, int k) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        pool.DoNumaWork(1, k,
            nullptr,
            [&](int task_id) {
                int j = task_id;
                for (int i = 0; i < n; i++) {
                    float sum = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    
#ifdef __aarch64__
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    for (; l + 3 < m; l += 4) {
                        float32x4_t vi = vld1q_f32(inputData + i * m + l);
                        float16x4_t vw16 = vld1_f16((const __fp16*)(weightData + j * m + l));
                        float32x4_t vw = vcvt_f32_f16(vw16);
                        vsum = vmlaq_f32(vsum, vi, vw);
                    }
                    sum += vaddvq_f32(vsum);
#endif
#elif defined(__AVX2__)
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m128i vw16 = _mm_loadu_si128((__m128i*)(weightData + j * m + l));
                        __m256 vw = fp16tofp32.convert(vw16);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    sum += Floatsum(vsum);
#endif
                    for (; l < m; l++) {
                        sum += inputData[i * m + l] * fp16tofp32.convert(weightData[j * m + l]);
                    }
                    outputData[i * k + j] = sum;
                }
            },
            nullptr
        );
    }

    // NUMA-aware Float32 x BFloat16 linear
    void RunNumaLinearFloat32BFloat16(float *inputData, uint16_t *weightData,
                                     float *outputData, float *biasData,
                                     int n, int m, int k) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        pool.DoNumaWork(1, k,
            nullptr,
            [&](int task_id) {
                int j = task_id;
                for (int i = 0; i < n; i++) {
                    float sum = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    
#ifdef __aarch64__
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    for (; l + 3 < m; l += 4) {
                        float32x4_t vi = vld1q_f32(inputData + i * m + l);
                        float32x4_t vw = bf16tofp32.convert(vld1_u16(weightData + j * m + l));
                        vsum = vmlaq_f32(vsum, vi, vw);
                    }
                    sum += vaddvq_f32(vsum);
#elif defined(__AVX2__)
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m128i vw16 = _mm_loadu_si128((__m128i*)(weightData + j * m + l));
                        __m256 vw = bf16tofp32.convert(vw16);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    sum += Floatsum(vsum);
#endif
                    for (; l < m; l++) {
                        sum += inputData[i * m + l] * bf16tofp32.convert(weightData[j * m + l]);
                    }
                    outputData[i * k + j] = sum;
                }
            },
            nullptr
        );
    }

    // NUMA-aware INT8 quantized linear
    void RunNumaLinearInt8(float *inputData, Data &weight,
                          float *outputData, float *biasData,
                          int n, int m, int k) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        uint8_t *weightData = weight.cpuData;
        
        pool.DoNumaWork(1, k,
            nullptr,
            [&](int task_id) {
                int j = task_id;
                LowBitConfig config = weight.perChannelsConfigs[j];
                float minValue = config.min;
                float maxValue = config.max;
                float scale = (maxValue - minValue) / 255.0f;
                
                for (int i = 0; i < n; i++) {
                    float sum = biasData ? biasData[j] : 0.0f;
                    for (int l = 0; l < m; l++) {
                        float w = minValue + weightData[j * m + l] * scale;
                        sum += inputData[i * m + l] * w;
                    }
                    outputData[i * k + j] = sum;
                }
            },
            nullptr
        );
    }

    // NUMA-aware INT4 quantized linear (group quantization)
    void RunNumaLinearInt4Group(float *inputData, Data &weight,
                               float *outputData, float *biasData,
                               int n, int m, int k, int group, int groupCnt) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        uint8_t *weightData = weight.cpuData;
        
        pool.DoNumaWork(1, k,
            nullptr,
            [&](int task_id) {
                int j = task_id;
                for (int i = 0; i < n; i++) {
                    float sum = biasData ? biasData[j] : 0.0f;
                    
                    for (int g = 0; g < group; g++) {
                        int gid = j * group + g;
                        float minValue = weight.mins[gid];
                        float scale = weight.scales[gid];
                        int st = g * groupCnt;
                        int end = std::min((g + 1) * groupCnt, m);
                        
                        for (int l = st; l < end; l++) {
                            int idx = j * m + l;
                            uint8_t byte = weightData[idx / 2];
                            int w_int = (idx % 2 == 0) ? (byte & 0xF) : (byte >> 4);
                            float w = minValue + w_int * scale;
                            sum += inputData[i * m + l] * w;
                        }
                    }
                    outputData[i * k + j] = sum;
                }
            },
            nullptr
        );
    }

    // NUMA-aware INT4 no-zero quantized linear
    void RunNumaLinearInt4NoZero(float *inputData, Data &weight,
                                float *outputData, float *biasData,
                                int n, int m, int k) {
        return RunNumaLinearInt4Group(inputData, weight, outputData, biasData, n, m, k, 1, m);
    }

    // NUMA-aware GGUF format linear
    void RunNumaLinearGGUF(float *inputData, Data &weight,
                          float *outputData, float *biasData,
                          int n, int m, int k) {
        NumaThreadPool& pool = NumaThreadPool::GetInstance();
        
        weight.Repack();
        ggml_tensor *tensor = (ggml_tensor*)weight.ggmlTensor;
        uint8_t *weightData = weight.cpuData;
        
        auto vec_dot_type = ggml_type_vec_dot_type(tensor->type);
        int rowCount = ggml_row_size(vec_dot_type, m);
        
        // Quantize input to Q8_K format
        std::vector<uint8_t> q8kInputs(n * rowCount);
        for (int i = 0; i < n; i++) {
            iqk_quantize_row_q8_K(
                inputData + i * m, q8kInputs.data() + i * rowCount, m,
                vec_dot_type, tensor->type
            );
        }
        
        // NUMA-aware matmul
        pool.DoNumaWork(1, k,
            nullptr,
            [&](int task_id) {
                int j = task_id;
                auto vec_dot = ggml_type_vec_dot(tensor->type);
                
                for (int i = 0; i < n; i++) {
                    float sum = 0.0f;
                    vec_dot(m,
                           &sum,
                           0,
                           weightData + j * ggml_row_size(tensor->type, m),
                           0,
                           q8kInputs.data() + i * rowCount,
                           0,
                           1);
                    outputData[i * k + j] = sum + (biasData ? biasData[j] : 0.0f);
                }
            },
            nullptr
        );
    }

}  // namespace fastllm

#endif  // USE_NUMA
