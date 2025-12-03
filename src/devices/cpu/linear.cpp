//
// Created by huangyuyang on 8/14/24.
//

#include "devices/cpu/computeutils.h"
#include "devices/cpu/cpudevice.h"

#include <cstring>
#include <thread>
#include <cfloat>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"
#include "computeutils.h"
#include <array>  // For std::array
#include "gguf.h"
#include <assert.h>

namespace fastllm {
    extern CPUInstructInfo cpuInstructInfo;
    extern FP16ToFP32Manager fp16tofp32;
    extern BF16ToFP32Manager bf16tofp32;
    extern FP8E4M3ToFP32Manager fp8e4m3tofp32;
    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);
    extern void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);
    extern void Float16ToBFloat16(uint16_t *float16, uint16_t *bfloat16, int len);
    extern void OnlineQuantization(float *inputData, std::vector<uint8_t> &uinput, std::vector<LowBitConfig> &inputConfigs, 
                                int n, int m, int group, int groupCnt,
                                std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros, 
                                int permuteType);
#ifdef __AVX2__
    extern int DotU4U8(uint8_t *a, uint8_t *b, int n);
#endif

    CPUInstructInfo *GetCPUInstructInfo() {
        return &cpuInstructInfo;
    }

    void AddBias(float *outputData, float *biasData, int n, int k, int st, int end) {
        if (biasData) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    outputData[i * k + j] += biasData[j];
                }
            }
        }
    }

    bool LinearBFloat16_FP8E4M3BLOCK128_Base_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        static int block_size = 128;
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        
        // 计算总共有多少个完整的block和最后一个不完整block的大小
        int num_blocks = (m + block_size - 1) / block_size;
        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
        
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
                            
            for (int j = st; j < end; j++) {
                uint8_t *rowStart = (uint8_t*)weightData + j * perRow;
                float sum = 0.0f;
                
                // 按block进行处理
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    // 计算当前block的大小（最后一个block可能不完整）
                    int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                    
                    // 计算当前block的起始位置
                    // 每个block占用 128字节(fp8) + 4字节(float scale)
                    uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                    uint8_t *fp8_ptr = block_start;
                    float *scale_ptr = (float*)(block_start + block_size);
                    
                    // 先计算block内的点积，最后再乘以scale
                    float block_sum = 0.0f;
                    int base_idx = block_idx * block_size;
                    
                    for (int l = 0; l < current_block_size; l++) {
                        // 将bf16的A转换为fp32
                        float valA = bf16tofp32.dict[bf16A[base_idx + l]];
                        
                        // 将fp8的B转换为fp32
                        float valB = fp8e4m3tofp32.dict[fp8_ptr[l]];
                        
                        block_sum += valA * valB;
                    }
                    
                    // 整个block的结果乘以scale
                    sum += block_sum * (*scale_ptr);
                }
                
                floatC[j] = sum;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    
    extern bool LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearBFloat16_FP8E4M3BLOCK128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512BF16) {
            return LinearBFloat16_FP8E4M3BLOCK128_AVX512BF16_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } if (GetCPUInstructInfo()->hasAVX2) {
            return LinearBFloat16_FP8E4M3BLOCK128_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else { 
            return LinearBFloat16_FP8E4M3BLOCK128_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
        return false;
    }

    bool LinearBFloat16_FP8E4M3PERCHANNEL_Base_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        size_t perRow = GetDataBytes(DataType::FP8_E4M3_PERCHANNEL, 1, m);
        for (int i = 0; i < n; i++) {
            uint16_t *bf16A = inputData + i * m;
            float *floatC = outputData + i * k;
                            
            for (int j = st; j < end; j++) {
                uint8_t *rowStart = (uint8_t*)weightData + j * perRow;
                float sum = 0.0f;
                
                uint8_t *block_start = rowStart;
                uint8_t *fp8_ptr = block_start;
                float *scale_ptr = (float*)(block_start + m);
                                
                float block_sum = 0.0f;
                for (int l = 0; l < m; l++) {
                    // 将bf16的A转换为fp32
                    float valA = bf16tofp32.dict[bf16A[l]];
                        
                    // 将fp8的B转换为fp32
                    float valB = fp8e4m3tofp32.dict[fp8_ptr[l]];
                        
                    block_sum += valA * valB;
                }
                    
                sum += block_sum * (*scale_ptr);
                floatC[j] = sum;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    
    extern bool LinearBFloat16_FP8E4M3PERCHANNEL_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16_FP8E4M3PERCHANNEL_AVX2_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearBFloat16_FP8E4M3PERCHANNEL_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512BF16) {
            return LinearBFloat16_FP8E4M3PERCHANNEL_AVX512BF16_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } if (GetCPUInstructInfo()->hasAVX2) {
            return LinearBFloat16_FP8E4M3PERCHANNEL_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else { 
            return LinearBFloat16_FP8E4M3PERCHANNEL_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
        return false;
    }

    bool LinearINT8PERCHANNEL_INT4PERCHANNEL_Base_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，InfInt8PerChannel格式
            uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
            int8_t *quantizedA = (int8_t*)infInt8A;
            float scaleA = *(float*)(infInt8A + m);
            int sumA = *(int*)(infInt8A + m + sizeof(float));
            
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            
            for (int j = st; j < end; j++) {
                // B矩阵的第j行，INT4_PERCHANNEL格式
                uint8_t *int4B = (uint8_t*)weightData + j * ldb;
                float minB = *(float*)(int4B + (m + 1) / 2);
                float scaleB = *(float*)(int4B + (m + 1) / 2 + sizeof(float));
                
                int sum = 0;
                // 一次处理两个int4值（假设m是偶数）
                for (int l = 0; l < m; l += 2) {
                    uint8_t packedValue = int4B[l / 2];
                    // 提取高4位和低4位
                    uint8_t int4Value0 = (packedValue >> 4) & 0x0F;  // 第一个int4
                    uint8_t int4Value1 = packedValue & 0x0F;         // 第二个int4
                    
                    // 同时计算两个乘积并累加
                    sum += quantizedA[l] * int4Value0 + quantizedA[l + 1] * int4Value1;
                }
                
                floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    extern bool LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8PERCHANNEL_INT4PERCHANNEL_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512VNNI) {
            return LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX512VNNI_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else if (GetCPUInstructInfo()->hasAVX2) {
            return LinearINT8PERCHANNEL_INT4PERCHANNEL_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else { 
            return LinearINT8PERCHANNEL_INT4PERCHANNEL_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
    }

    bool LinearINT8PERCHANNEL_INT8PERCHANNEL_Base_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        size_t lda = GetDataBytes(DataType::INF_INT8_PERCHANNEL, 1, m);
        size_t ldb = GetDataBytes(DataType::INT8_PERCHANNEL, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，InfInt8PerChannel格式
            uint8_t *infInt8A = (uint8_t*)inputData + i * lda;
            int8_t *quantizedA = (int8_t*)infInt8A;
            float scaleA = *(float*)(infInt8A + m);
            int sumA = *(int*)(infInt8A + m + sizeof(float));
            
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            
            for (int j = st; j < end; j++) {
                // B矩阵的第j行，INT8_PERCHANNEL格式
                uint8_t *int8B = (uint8_t*)weightData + j * ldb;
                float minB = *(float*)(int8B + m);
                float scaleB = *(float*)(int8B + m + sizeof(float));
                
                int sum = 0;
                for (int l = 0; l < m; l++) {
                    sum += quantizedA[l] * int8B[l];
                }
                
                floatC[j] = sum * scaleA * scaleB + minB * scaleA * sumA;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }
    extern bool LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8PERCHANNEL_INT8PERCHANNEL_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512VNNI) {
            return LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX512VNNI_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else if (GetCPUInstructInfo()->hasAVX2) {
            return LinearINT8PERCHANNEL_INT8PERCHANNEL_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else { 
            return LinearINT8PERCHANNEL_INT8PERCHANNEL_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
    }

    bool LinearINT8GROUP128_INT4GROUP128_Base_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        size_t lda = GetDataBytes(DataType::INF_INT8_GROUP128, 1, m);
        size_t ldb = GetDataBytes(DataType::INT4_GROUP128, 1, m);
        size_t ldc = GetDataBytes(DataType::FLOAT32, 1, k);
        
        int groupCnt = 128;
        int groups = m / 128;
        size_t astride = groupCnt + sizeof(float) + sizeof(int);
        size_t bstride = (groupCnt / 2) + sizeof(float) * 2;
        for (int i = 0; i < n; i++) {
            // A矩阵的第i行，INF_INT8_GROUP128格式
            float *floatC = (float*)((uint8_t*)outputData + i * ldc);
            for (int j = st; j < end; j++) {
                float fsum = 0.0f;
                for (int g = 0; g < groups; g++) {
                    uint8_t *infInt8A = (uint8_t*)inputData + i * lda + g * astride;
                    int8_t *quantizedA = (int8_t*)infInt8A;
                    
                    float scaleA = *(float*)(infInt8A + groupCnt);
                    int sumA = *(int*)(infInt8A + groupCnt + sizeof(float));

                    uint8_t *int4B = (uint8_t*)weightData + j * ldb + g * bstride;
                    float minB = *(float*)(int4B + (groupCnt + 1) / 2);
                    float scaleB = *(float*)(int4B + (groupCnt + 1) / 2 + sizeof(float));

                    int sum = 0;
                    // 一次处理两个int4值（假设m是偶数）
                    for (int l = 0; l < groupCnt; l += 2) {
                        uint8_t packedValue = int4B[l / 2];
                        // 提取高4位和低4位
                        uint8_t int4Value0 = (packedValue >> 4) & 0x0F;  // 第一个int4
                        uint8_t int4Value1 = packedValue & 0x0F;         // 第二个int4
                        
                        // 同时计算两个乘积并累加
                        sum += quantizedA[l] * int4Value0 + quantizedA[l + 1] * int4Value1;
                    }

                    fsum += sum * scaleA * scaleB + minB * scaleA * sumA;
                }
                floatC[j] = fsum;
            }
        }
        AddBias(outputData, biasData, n, k, st, end);
        return true;
    }

    extern bool LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearINT8GROUP128_INT4GROUP128_AVX2_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8GROUP128_INT4GROUP128_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
        if (GetCPUInstructInfo()->hasAVX512VNNI) {
            return LinearINT8GROUP128_INT4GROUP128_AVX512VNNI_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else if (GetCPUInstructInfo()->hasAVX2) {
            return LinearINT8GROUP128_INT4GROUP128_AVX2_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        } else { 
            return LinearINT8GROUP128_INT4GROUP128_Base_Kernel(inputData, weightData, biasData, outputData, n, m, k, st, end);
        }
    }

    void MultiThreadLinearFloat32Float32Op::Run() {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t sum = {0, 0, 0, 0};
                for (; l + 3 < m; l += 4) {
                    sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vld1q_f32(weightData + j * m + l)));
                }
                now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; l + 7 < m; l += 8) {
                    __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                    __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
#endif
                for (; l < m; l++) {
                    now += inputData[i * m + l] * weightData[j * m + l];
                }
                outputData[i * k + j] = now;
            }
        }
    }

    
    void MultiThreadLinearFloat32GGUFOp::Run() {
        ggml_tensor *tensor = (ggml_tensor*)this->ggmlTensor;
        int rowCount = m / QK_K; // 每行有多少个block
        auto vec_dot_type = ggml_type_vec_dot_type(tensor->type);
        auto vec_dot = ggml_type_vec_dot(tensor->type);
        if (GetMulMatFunction(tensor->type, 1) != nullptr) {
            int part = (n == 1 ? (end - st) : 64);
            int oldSt = st, oldEnd = end;

            int maxRows = 8;
            std::vector <mul_mat_t> mats;
            mats.resize(maxRows + 1);
            for (int i = 1; i <= maxRows; i++) {
                mats[i] = GetMulMatFunction(tensor->type, i);
            }
            while (st < oldEnd) {
                end = std::min(st + part, oldEnd);
                int i = 0;

                for (; i + 7 < n; i += 8) {
                    DataInfo info{&outputData[i * k + st], 
                        (const char*)q8kInputData + i * ggml_row_size(vec_dot_type, m), 
                        (size_t)k, ggml_row_size(vec_dot_type, m), 
                        0, 1, nullptr, 0};
                    mats[8](m, weightData + st * ggml_row_size(tensor->type, m), ggml_row_size(tensor->type, m), info, end - st);
                }

                if (i < n) {
                    DataInfo info{&outputData[i * k + st], 
                        (const char*)q8kInputData + i * ggml_row_size(vec_dot_type, m), 
                        (size_t)k, ggml_row_size(vec_dot_type, m), 
                        0, 1, nullptr, 0};
                    mats[n - i](m, weightData + st * ggml_row_size(tensor->type, m), ggml_row_size(tensor->type, m), info, end - st);
                }

                if (biasData) {
                    for (int i = 0; i < n; i++) {
                        for (int j = st; j < end; j++) {
                            outputData[i * k + j] += biasData[j];
                        }
                    }
                }
                st = end;
            }
        } else if (vec_dot != nullptr) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = 0.0f;
                    vec_dot (
                        m, &outputData[i * k + j], 0, 
                        weightData + j * ggml_row_size(tensor->type, m), 0, 
                        q8kInputData + i * ggml_row_size(vec_dot_type, m), 0, 
                        1
                    );
                    outputData[i * k + j] += (biasData ? biasData[j] : 0.0f);
                } 
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport GGUF's dataType " + std::string(ggml_type_name(tensor->type)) + ".\n");
        }
    }

    bool LinearQ8K_GGUF_Kernel(uint8_t *q8kInputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, DataType AType, DataType BType) {
        ggml_type inputType = (ggml_type)((int)AType - (int)DataType::DATA_GGUF_FORMAT);
        ggml_type weightType = (ggml_type)((int)BType - (int)DataType::DATA_GGUF_FORMAT);

        auto vec_dot_type = ggml_type_vec_dot_type(weightType);
        auto vec_dot = ggml_type_vec_dot(weightType);
        if (GetMulMatFunction(weightType, 1) != nullptr) {
            int part = (n == 1 ? (end - st) : 64);
            int oldSt = st, oldEnd = end;

            int maxRows = 8;
            std::vector <mul_mat_t> mats;
            mats.resize(maxRows + 1);
            for (int i = 1; i <= maxRows; i++) {
                mats[i] = GetMulMatFunction(weightType, i);
            }
            while (st < oldEnd) {
                end = std::min(st + part, oldEnd);
                int i = 0;

                for (; i + 7 < n; i += 8) {
                    DataInfo info{&outputData[i * k + st], 
                        (const char*)q8kInputData + i * ggml_row_size(vec_dot_type, m), 
                        (size_t)k, ggml_row_size(vec_dot_type, m), 
                        0, 1, nullptr, 0};
                    mats[8](m, weightData + st * ggml_row_size(weightType, m), ggml_row_size(weightType, m), info, end - st);
                }

                if (i < n) {
                    DataInfo info{&outputData[i * k + st], 
                        (const char*)q8kInputData + i * ggml_row_size(vec_dot_type, m), 
                        (size_t)k, ggml_row_size(vec_dot_type, m), 
                        0, 1, nullptr, 0};
                    mats[n - i](m, weightData + st * ggml_row_size(weightType, m), ggml_row_size(weightType, m), info, end - st);
                }

                if (biasData) {
                    for (int i = 0; i < n; i++) {
                        for (int j = st; j < end; j++) {
                            outputData[i * k + j] += biasData[j];
                        }
                    }
                }
                st = end;
            }
        } else if (vec_dot != nullptr) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = 0.0f;
                    vec_dot (
                        m, &outputData[i * k + j], 0, 
                        weightData + j * ggml_row_size(weightType, m), 0, 
                        q8kInputData + i * ggml_row_size(vec_dot_type, m), 0, 
                        1
                    );
                    outputData[i * k + j] += (biasData ? biasData[j] : 0.0f);
                } 
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport GGUF's dataType " + std::string(ggml_type_name(weightType)) + ".\n");
            return false;
        }
        return true;
    }

    extern bool LinearFloat32Float16_AVX512F_Kernel(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearFloat32Float16_AVX2_Kernel(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);

    void MultiThreadLinearFloat32Float16Op::Run() {
        if (cpuInstructInfo.hasAVX512F) {
            if (LinearFloat32Float16_AVX512F_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end
                )) {
                return;
            }
        }
        if (cpuInstructInfo.hasAVX2 && n > 1) {
            if (LinearFloat32Float16_AVX2_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end
                )) {
                return;
            }
        }   

        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                float16x8_t sum_vec = vdupq_n_f16(0.0f);
                for (; l + 7 < m; l += 8) {
                    // 加载 8 个 FP32 输入：分成低 4 和高 4
                    float32x4_t vin_low = vld1q_f32(inputData + i * m + l);
                    float32x4_t vin_high = vld1q_f32(inputData + i * m + l + 4);
                    
                    // 转换为 FP16
                    float16x4_t vin16_low = vcvt_f16_f32(vin_low);
                    float16x4_t vin16_high = vcvt_f16_f32(vin_high);
                    
                    // 合并为 FP16x8
                    float16x8_t vin16 = vcombine_f16(vin16_low, vin16_high);
                    
                    // 加载 8 个 FP16 权重
                    float16x8_t vweight = vld1q_f16((const float16_t*)(weightData + j * m + l));
                    
                    // FP16 FMA: sum += vin * vweight
                    sum_vec = vfmaq_f16(sum_vec, vin16, vweight);
                }
                
                // 将 sum_vec 转换为 FP32 并求和
                float16x4_t slow = vget_low_f16(sum_vec);
                float16x4_t shigh = vget_high_f16(sum_vec);
                float32x4_t flow = vcvt_f32_f16(slow);
                float32x4_t fhigh = vcvt_f32_f16(shigh);
                now += vaddvq_f32(flow) + vaddvq_f32(fhigh);
#else
#ifdef __aarch64__
                float32x4_t sum = {0, 0, 0, 0};
                for (; l + 3 < m; l += 4) {
                    float32x4_t vcur = {fp16tofp32.dict[weightData[j * m + l]], fp16tofp32.dict[weightData[j * m + l + 1]],
                                        fp16tofp32.dict[weightData[j * m + l + 2]], fp16tofp32.dict[weightData[j * m + l + 3]]};
                    sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vcur));
                }
                now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; l + 7 < m; l += 8) {
                    __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                    __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + l)));
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
#endif
#endif
                for (; l < m; l++) {
                    now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = now;
            }
        }
    }

    extern bool LinearBFloat16BFloat16_AVX512BF16_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16BFloat16_AVX2_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    extern bool LinearBFloat16BFloat16_AMX_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    void MultiThreadLinearBFloat16BFloat16Op::Run() {
        if (cpuInstructInfo.hasAMX && GetEnableAMX() && n > 7) {
            if (LinearBFloat16BFloat16_AMX_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end
                )) {
                return;
            }
        }
        
        if (cpuInstructInfo.hasAVX512BF16) {
            if (LinearBFloat16BFloat16_AVX512BF16_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end
                )) {
                return;
            }
        }

        if (cpuInstructInfo.hasAVX2) {
            if (LinearBFloat16BFloat16_AVX2_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end
                )) {
                return;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __AVX2__
                __m256 vsum = _mm256_setzero_ps();
                for (; l + 7 < m; l += 8) {
                    // 从内存加载 8 个 bfloat16 值（16 字节）
                    __m128i vi_bf16 = _mm_loadu_si128((__m128i *) (inputData + i * m + l));
                    
                    // 将 bfloat16 转换为 float32
                    // bfloat16 转换需要将每个 16 位值左移 16 位到 float32 的高位
                    __m256i vi_shifted = _mm256_cvtepu16_epi32(vi_bf16);
                    vi_shifted = _mm256_slli_epi32(vi_shifted, 16);
                    __m256 vi = _mm256_castsi256_ps(vi_shifted);

                    // 从内存加载 8 个 bfloat16 值（16 字节）
                    __m128i vw_bf16 = _mm_loadu_si128((__m128i *) (weightData + j * m + l));
                    
                    // 将 bfloat16 转换为 float32
                    // bfloat16 转换需要将每个 16 位值左移 16 位到 float32 的高位
                    __m256i vw_shifted = _mm256_cvtepu16_epi32(vw_bf16);
                    vw_shifted = _mm256_slli_epi32(vw_shifted, 16);
                    __m256 vw = _mm256_castsi256_ps(vw_shifted);
                    
                    vsum = _mm256_fmadd_ps(vi, vw, vsum);
                }
                now += Floatsum(vsum);
#endif
                for (; l < m; l++) {
                    now += bf16tofp32.dict[inputData[i * m + l]] * bf16tofp32.dict[weightData[j * m + l]];
                }
                outputData[i * k + j] = now;
            }
        }
    }

#ifdef __AVX2__
    // 2. Vectorized FP8 E4M3 to FP32 conversion (AVX2)
    //    Input: 8 uint8_t values packed into the lower 64 bits of an __m128i
    //    Output: 8 float values in an __m256 vector
    //    This is the most complex part to implement correctly and efficiently.
    inline __m256 _mm256_fp8e4m3_to_fp32_ps(__m128i v_u8) {
        // --- Implementation Sketch ---
        // a. Unpack 8 uint8_t values into 8 32-bit integers (__m256i)
        __m256i v_u32 = _mm256_cvtepu8_epi32(v_u8); // Zero-extends uint8 -> int32
        // b. Define masks and constants for AVX2 registers
        const __m256i sign_mask = _mm256_set1_epi32(0x80);
        const __m256i exp_mask  = _mm256_set1_epi32(0x78); // E4
        const __m256i mant_mask = _mm256_set1_epi32(0x07); // M3
        const __m256i exp_shift = _mm256_set1_epi32(3);
        const __m256i mant_shift = _mm256_set1_epi32(23 - 3); // Shift M3 to FP32 mantissa position
        const __m256i fp32_sign_shift = _mm256_set1_epi32(24); // Shift sign bit
        const __m256i fp32_exp_shift = _mm256_set1_epi32(23); // Shift exponent bits
        const __m256i bias_delta = _mm256_set1_epi32(127 - 7); // FP32 bias - FP8 bias
        const __m256i zero = _mm256_setzero_si256();
        // Mask for checking if exp and mantissa are both zero (value is zero)
        const __m256i non_sign_mask = _mm256_set1_epi32(0x7F);
        // c. Extract components using bitwise operations
        __m256i signs = _mm256_and_si256(v_u32, sign_mask);
        __m256i exp8  = _mm256_srli_epi32(_mm256_and_si256(v_u32, exp_mask), 3);
        __m256i mant8 = _mm256_and_si256(v_u32, mant_mask);
        // d. Convert exponent bias
        __m256i exp32 = _mm256_add_epi32(exp8, bias_delta);
        // e. Shift components to their FP32 positions
        __m256i fp32_sign = _mm256_sllv_epi32(signs, fp32_sign_shift); // Shift sign bit
        __m256i fp32_exp  = _mm256_sllv_epi32(exp32, fp32_exp_shift); // Shift exponent
        __m256i fp32_mant = _mm256_sllv_epi32(mant8, mant_shift);  // Shift mantissa
        // f. Combine components (assuming normal numbers for now)
        __m256i fp32_bits = _mm256_or_si256(fp32_sign, _mm256_or_si256(fp32_exp, fp32_mant));
        // g. Handle zeros (where exponent and mantissa bits are 0)
        //    Check if bits 0-6 are zero
        __m256i is_zero_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_u32, non_sign_mask), zero);
        //    Select 0.0f where the input was zero, otherwise keep calculated bits
        fp32_bits = _mm256_andnot_si256(is_zero_mask, fp32_bits); // Bitwise SELECT(mask, 0, fp32_bits)
        // h. Handle other special cases (NaN, Inf, subnormals) based on E4M3 spec - SKIPPED IN THIS SKETCH
        // i. Cast the integer bits representation to float vector
        return _mm256_castsi256_ps(fp32_bits);
        // --- End Implementation Sketch ---
    }

    // 2. Vectorized FP8 E4M3 to FP32 conversion (AVX2)
    //    Input: 8 uint8_t values packed into the lower 64 bits of an __m128i
    //    Output: 8 float values in an __m256 vector
    //    This is the most complex part to implement correctly and efficiently.
    inline __m256 _mm256_fp8e4m3_to_fp32_fast_ps(__m128i v_u8) {
        // --- Implementation Sketch ---
        // a. Unpack 8 uint8_t values into 8 32-bit integers (__m256i)
        __m256i v_u32 = _mm256_cvtepu8_epi32(v_u8); // Zero-extends uint8 -> int32
        // b. Define masks and constants for AVX2 registers
        const __m256i sign_mask = _mm256_set1_epi32(0x80);
        const __m256i last_mask  = _mm256_set1_epi32(0x7F); // last
        const __m256i fp32_sign_shift = _mm256_set1_epi32(24); // Shift sign bit
        const __m256i fp32_last_shift = _mm256_set1_epi32(20); // Shift last bits
        // c. Extract components using bitwise operations
        __m256i signs = _mm256_and_si256(v_u32, sign_mask);
        __m256i lasts = _mm256_and_si256(v_u32, last_mask);

        __m256i fp32_sign = _mm256_sllv_epi32(signs, fp32_sign_shift); // Shift sign bit
        __m256i fp32_last  = _mm256_sllv_epi32(lasts, fp32_last_shift); // Shift last

        // f. Combine components (assuming normal numbers for now)
        __m256i fp32_bits = _mm256_or_si256(fp32_sign, fp32_last);
        return _mm256_castsi256_ps(fp32_bits);
        // --- End Implementation Sketch ---
    }
#endif

    extern bool LinearBFloat16FP8E4M3_AVX512BF16_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
        int n, int m, int k, int st, int end, int blockK, int blockM, float *scales, 
        int ks, int ms, float magicScale);

    void MultiThreadLinearBFloat16FP8E4M3Op::Run() {
        static struct FP8E4M3ToFP32Manager fp8e4m3tofp32;
        static float magicScale = pow(2, 120);
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;

        if (cpuInstructInfo.hasAVX512BF16) {
            if (LinearBFloat16FP8E4M3_AVX512BF16_Kernel(
                inputData, weightData, biasData, outputData, n, m, k, st, end, blockK, blockM, scales, ks, ms, magicScale
            )) {
                return;
            }
        }
#ifdef __AVX2__
        if (m % blockM == 0 && blockM % 8 == 0) {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    __m256 lastSum = _mm256_setzero_ps();
                    for (int midx = 0; midx < ms; midx++) {
                        float curScale = scales[j / blockK * ms + midx];
                        int l = midx * blockM;

                        __m256 vsum = _mm256_setzero_ps();
                        for (; l + 7 < (midx + 1) * blockM; l += 8) {
                            // __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                            __m128i bf16_vec_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputData + i * m + l));
                            __m256i bf16_extended_to_32 = _mm256_cvtepu16_epi32(bf16_vec_128);
                            __m256i fp32_bits_int = _mm256_slli_epi32(bf16_extended_to_32, 16);
                            __m256 vi = _mm256_castsi256_ps(fp32_bits_int);
                            
                            __m128i vw_u8 = _mm_loadl_epi64((const __m128i*)(weightData + j * m + l));
                            __m256 vw = _mm256_fp8e4m3_to_fp32_fast_ps(vw_u8);
                            vsum = _mm256_fmadd_ps(vi, vw, vsum);
                        }
                        // now += Floatsum(vsum) * curScale;
                        lastSum = _mm256_fmadd_ps(vsum, _mm256_set1_ps(curScale), lastSum);                        
                    }
                    now += Floatsum(lastSum) * pow(2, 120);
                    outputData[i * k + j] = now;
                }
            }
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    for (int midx = 0; midx < ms; midx++) {
                        float curScale = scales[j / blockK * ms + midx] * magicScale;
                        int l = midx * blockM;
                        __m256 vsum = _mm256_setzero_ps();
                        for (; l + 7 < m && l + 7 < (midx + 1) * blockM; l += 8) {
                            // __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                            __m128i bf16_vec_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(inputData + i * m + l));
                            __m256i bf16_extended_to_32 = _mm256_cvtepu16_epi32(bf16_vec_128);
                            __m256i fp32_bits_int = _mm256_slli_epi32(bf16_extended_to_32, 16);
                            __m256 vi = _mm256_castsi256_ps(fp32_bits_int);

                            __m128i vw_u8 = _mm_loadl_epi64((const __m128i*)(weightData + j * m + l));
                            __m256 vw = _mm256_fp8e4m3_to_fp32_fast_ps(vw_u8);
                            vsum = _mm256_fmadd_ps(vi, vw, vsum);
                        }
                        now += Floatsum(vsum) * curScale;
                        for (; l < m && l < (midx + 1) * blockM; l++) {
                            now += curScale * inputData[i * m + l] * fp8e4m3tofp32.dict[weightData[j * m + l]];
                        }
                    }
                    outputData[i * k + j] = now;
                }
            }
        }
#else
        ErrorInFastLLM("Unsupport MultiThreadLinearBFloat16FP8E4M3Op");
#endif
    }

    void MultiThreadLinearInt8Int8Op::Run() {
        MatMulInt8Int8(a, b, c, n, m, k, kstride);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                float value = ((int32_t *) c)[i * kstride + j];
#ifdef __AVX2__
                value += (128 * weightSums[j]);
                value += (128 * inputSums[i]);
                value -= m * 128 * 128;
#endif
                value -= weightSums[j] * izeros[i];
                value -= inputSums[i] * weightZeros[j];
                value += (int) izeros[i] * weightZeros[j] * m;
                ((float*)c)[i * kstride + j] = scales[j] * iscales[i] * value + (bias == nullptr ? 0.0 : bias[j]);
            }
        }
    }

    void MultiThreadLinearFloat16Float16Op::Run() {
        MatMulFloat16Float16(
                inputData, weightData, biasData, outputData, 
                n, m, k, st, end
        );
    }

    extern bool MatMulInt8Int4_AVX512VNNI(uint8_t *a, uint8_t *b, float *c, int n, int m, int k);
    extern bool MatMulInt8Int4Group_AVX512VNNI(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
        int group, int realGroup, int groupCnt, float *iscales, float *scales, float *izeros, float *weightMins);
    
    void MultiThreadLinearInt8Int4GroupOp::Run() {
#ifdef __AVX2__
        if (group == 1) {
            int block = 0;
            int realGroup = (m - 1) / groupCnt + 1;            
            std::vector <float> tempValue, values;
            tempValue.resize(n);
            values.resize(n * k);
            for (; block < n; block++) {
                tempValue[block] = (inputSums[block] - izeros[block] * groupCnt) * iscales[block];
            }

             if (cpuInstructInfo.hasAVX512VNNI && 
                MatMulInt8Int4_AVX512VNNI(a, b, values.data(), n, m, k)) {
            } else  {
                block = 0;
                for (; block + 3 < n; block += 4) {
                    uint8_t *weightWalk = b;
                    uint8_t *inputStart = a + block * m;

                    for (int i = 0; i < k; i++) {
                        uint8_t *a = weightWalk + (i * m) / 2;
                        uint8_t *b = inputStart;

                        __m256i acc0 = _mm256_setzero_si256();
                        __m256i acc1 = _mm256_setzero_si256();
                        __m256i acc2 = _mm256_setzero_si256();
                        __m256i acc3 = _mm256_setzero_si256();

                        const __m256i lowMask = _mm256_set1_epi8(0xf);
                        const __m256i ones = _mm256_set1_epi16(1);
                        int j = 0, ans = 0;
                        for (; j + 31 < m; j += 32) {
                            __m128i orix = _mm_loadu_si128((const __m128i *) (a + j / 2));
                            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                            __m256i bx = _mm256_and_si256(lowMask, bytex);
                            __m256i by0 = _mm256_loadu_si256((const __m256i *) (b + j));
                            __m256i by1 = _mm256_loadu_si256((const __m256i *) (b + m * 1 + j));
                            __m256i by2 = _mm256_loadu_si256((const __m256i *) (b + m * 2 + j));
                            __m256i by3 = _mm256_loadu_si256((const __m256i *) (b + m * 3 + j));

                            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(by2, bx), ones));
                            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(by3, bx), ones));
                        }
                        values[block * k + i] = I32sum(acc0);
                        values[(block + 1) * k + i] = I32sum(acc1);
                        values[(block + 2) * k + i] = I32sum(acc2);
                        values[(block + 3) * k + i] = I32sum(acc3);
                    }
                }

                for (; block + 2 < n; block += 3) {
                    uint8_t *weightWalk = b;
                    uint8_t *inputStart = a + block * m;

                    for (int i = 0; i < k; i++) {
                        uint8_t *a = weightWalk + (i * m) / 2;
                        uint8_t *b = inputStart;

                        __m256i acc0 = _mm256_setzero_si256();
                        __m256i acc1 = _mm256_setzero_si256();
                        __m256i acc2 = _mm256_setzero_si256();

                        const __m256i lowMask = _mm256_set1_epi8(0xf);
                        const __m256i ones = _mm256_set1_epi16(1);
                        int j = 0, ans = 0;
                        for (; j + 31 < m; j += 32) {
                            __m128i orix = _mm_loadu_si128((const __m128i *) (a + j / 2));
                            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                            __m256i bx = _mm256_and_si256(lowMask, bytex);
                            __m256i by0 = _mm256_loadu_si256((const __m256i *) (b + j));
                            __m256i by1 = _mm256_loadu_si256((const __m256i *) (b + m * 1 + j));
                            __m256i by2 = _mm256_loadu_si256((const __m256i *) (b + m * 2 + j));                        

                            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(by2, bx), ones));
                        }
                        values[block * k + i] = I32sum(acc0);
                        values[(block + 1) * k + i] = I32sum(acc1);
                        values[(block + 2) * k + i] = I32sum(acc2);
                    }
                }

                for (; block + 1 < n; block += 2) {
                    uint8_t *weightWalk = b;
                    uint8_t *inputStart = a + block * m;

                    for (int i = 0; i < k; i++) {
                        uint8_t *a = weightWalk + (i * m) / 2;
                        uint8_t *b = inputStart;

                        __m256i acc0 = _mm256_setzero_si256();
                        __m256i acc1 = _mm256_setzero_si256();

                        const __m256i lowMask = _mm256_set1_epi8(0xf);
                        const __m256i ones = _mm256_set1_epi16(1);
                        int j = 0, ans = 0;
                        for (; j + 31 < m; j += 32) {
                            __m128i orix = _mm_loadu_si128((const __m128i *) (a + j / 2));
                            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                            __m256i bx = _mm256_and_si256(lowMask, bytex);
                            __m256i by0 = _mm256_loadu_si256((const __m256i *) (b + j));
                            __m256i by1 = _mm256_loadu_si256((const __m256i *) (b + m * 1 + j));

                            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(by0, bx), ones));
                            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(by1, bx), ones));
                        }
                        values[block * k + i] = I32sum(acc0);
                        values[(block + 1) * k + i] = I32sum(acc1);
                    }
                }

                for (; block < n; block++) {
                    uint8_t *weightWalk = b;
                    uint8_t *inputStart = a + block * m;

                    for (int i = 0; i < k; i++) {
                        uint8_t *a = weightWalk + (i * m) / 2;
                        uint8_t *b = inputStart;

                        __m256i acc = _mm256_setzero_si256();
                        const __m256i lowMask = _mm256_set1_epi8(0xf);
                        const __m256i ones = _mm256_set1_epi16(1);
                        int j = 0, ans = 0;
                        for (; j + 31 < m; j += 32) {
                            __m128i orix = _mm_loadu_si128((const __m128i *) (a + j / 2));
                            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                            __m256i bx = _mm256_and_si256(lowMask, bytex);
                            __m256i by = _mm256_loadu_si256((const __m256i *) (b + j));
                            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
                        }
                        values[block * k + i] = I32sum(acc);
                    }
                }
            }

            block = 0;
            for (; block < n; block++) {
                int i = 0;
                for (; i < k; i++) {
                    const float vv = (float)values[block * k + i] - weightSums[i] * izeros[block];
                    float sum = scales[i] * iscales[block] * vv + weightMins[i] * tempValue[block];
                    ((float*)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
                }
            }
            return;
        } else if (true) {
            int block = 0;
            int realGroup = (m - 1) / groupCnt + 1;            
            std::vector <float> values;
            values.resize(n * k);

            if (cpuInstructInfo.hasAVX512VNNI && 
                MatMulInt8Int4Group_AVX512VNNI(a, b, values.data(), n, m, k, group, realGroup, groupCnt, iscales, scales, izeros, weightMins)) {
                    
            } else  {
                const __m256i lowMask = _mm256_set1_epi8(0xf);
                const __m256i ones = _mm256_set1_epi16(1);

                block = 0;
                for (; block < n; block++) {
                    uint8_t *weightWalk = b;
                    uint8_t *inputStart = a + block * m;

                    for (int i = 0; i < k; i++) {
                        uint8_t *a = weightWalk + (i * m) / 2;
                        uint8_t *b = inputStart;
                        __m256 lastSum = _mm256_setzero_ps();
                        for (int g = 0; g < realGroup; g++) {
                            const int iid = block * group + g;
                            const int gid = i * group + g;
                            int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);

                            __m256i acc = _mm256_setzero_si256();
                            __m256i sub = _mm256_setzero_si256();
                            __m256i zeros = _mm256_set1_epi8((uint8_t)izeros[iid]);

                            for (int j = st; j + 31 < end; j += 32) {
                                __m128i orix = _mm_loadu_si128((const __m128i *) (a + j / 2));
                                __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                                __m256i bx = _mm256_and_si256(lowMask, bytex);
                                __m256i by = _mm256_loadu_si256((const __m256i *) (b + j));

                                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
                                sub = _mm256_add_epi32(sub, _mm256_madd_epi16(_mm256_maddubs_epi16(zeros, bx), ones));
                            }

                            lastSum = _mm256_add_ps (
                                    lastSum, 
                                    _mm256_mul_ps(
                                        _mm256_cvtepi32_ps(_mm256_sub_epi32(acc, sub)), 
                                        _mm256_set1_ps(iscales[iid] * scales[gid])
                                    ));
                        }

                        values[block * k + i] = Floatsum(lastSum);
                    }
                }
            }

            std::vector <float> tempValue;
            tempValue.resize(realGroup);
            block = 0;
            for (; block < n; block++) {
                for (int g = 0; g < realGroup; g++) {
                    int iid = block * group + g;
                    tempValue[g] = (inputSums[iid] - izeros[iid] * groupCnt) * iscales[iid];
                }

                int i = 0;
                for (; i < k; i++) {
                    float sum = (float)values[block * k + i];
                    int g = 0;
                    __m256 sum_vec = _mm256_setzero_ps();
                    for (; g + 7 < realGroup; g += 8) {
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(
                            _mm256_loadu_ps(&tempValue[g]), 
                            _mm256_loadu_ps(&weightMins[i * group + g])
                        ));
                    }
                    sum += Floatsum(sum_vec);

                    for (; g < realGroup; g++) {
                        const int iid = block * group + g;
                        const int gid = i * group + g;
                        sum += tempValue[g] * weightMins[gid];
                    }

                    ((float*)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
                }
            }
            return;
        }
#endif
        std::vector <float> values;
        values.resize(group);

        int block = 0;
        int realGroup = (m - 1) / groupCnt + 1;
        std::vector <float> tempValue;
        tempValue.resize(realGroup);
        for (; block < n; block++) {
            for (int g = 0; g < realGroup; g++) {
                int iid = block * group + g;
                tempValue[g] = (inputSums[iid] - izeros[iid] * groupCnt) * iscales[iid];
            }

            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                std::fill(values.begin(), values.end(), 0.0f);
                uint8_t *inputWalk = inputStart;
                float sum = 0.0;

                for (int g = 0; g < realGroup; g++) {
                    int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                    float &value = values[g];
                    int j = st;
#ifdef __ARM_FEATURE_DOTPROD
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x2_t sum0 = {0, 0};

                    for (; j + 15 < end; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vdot_u32(sum0, va, in.val[1]);
                        sum0 = vdot_u32(sum0, vb, in.val[0]);
                    }
                    value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < end; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                        sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                    }
                    value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                    value += DotU4U8(weightWalk + (i * m + st) / 2, inputWalk + st, end - st);
                    j += (end - st);
#endif
                    for (; j + 1 < end; j += 2) {
                        int id = (i * m + j) / 2;
                        value += (weightWalk[id] >> 4) * inputWalk[j];
                        value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                    }
                }

                int g = 0;
#ifdef __aarch64__
                float32x4_t vSum = vdupq_n_f32(0.0f);
                float32x4_t vGroupCnt = vdupq_n_f32(groupCnt);
                for (; g + 3 < realGroup; g += 4) {
                    int iid = block * group + g;
                    int gid = i * group + g;
                    float32x4_t vValue = vld1q_f32(values.data() + g);
                    float32x4_t vWeightSum = vcvtq_f32_s32(vld1q_s32(weightSums + gid));
                    float32x4_t vWeightMin = vld1q_f32(weightMins + gid);
                    float32x4_t vScale = vld1q_f32(scales + gid);
                    float32x4_t vIzero = vld1q_f32(izeros + iid);
                    float32x4_t vIscale = vld1q_f32(iscales + iid);
                    float32x4_t vInputSum = vld1q_f32(inputSums + iid);
                    float32x4_t vMiddle = vsubq_f32(vInputSum, vmulq_f32(vIzero, vGroupCnt));
                    vValue = vsubq_f32(vValue, vmulq_f32(vWeightSum, vIzero));
                    vSum = vaddq_f32(vSum, vmulq_f32(vScale, vmulq_f32(vIscale, vValue)));
                    vSum = vaddq_f32(vSum, vmulq_f32(vWeightMin, vmulq_f32(vMiddle, vIscale)));
                }
                sum += vSum[0] + vSum[1] + vSum[2] + vSum[3];
#endif
                // 处理剩余元素（标量处理）
                for (; g < realGroup; g++) {
                    const int iid = block * group + g;
                    const int gid = i * group + g;
                    
                    // 修正value为float类型
                    const float value = (float)values[g] - weightSums[gid] * izeros[iid];
                    sum += scales[gid] * iscales[iid] * value + weightMins[gid] * tempValue[g];
                }

                if (group * groupCnt > m) {
                    int iid = block * group + group - 1;
                    int gid = i * group + group - 1;
                    sum += weightMins[gid] * izeros[iid] * (group * groupCnt - m) * iscales[iid];
                }

                ((float*)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
            }
        }
    }

    void RunLinearFloat32Float32(float *inputData, float *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearFloat32Float32Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearFloat32Float32Op(inputData, weightData, biasData, outputData,
                                                    n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void RunLinearFloat16Float32(uint16_t *inputData, float *weightData, uint16_t *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <float> floatInput, floatOutput;
        floatInput.resize(n * m);
        floatOutput.resize(n * k);
        Float16ToFloat32(inputData, floatInput.data(), n * m);
        RunLinearFloat32Float32(floatInput.data(), weightData, floatOutput.data(), biasData, n, m, k, pool, startTid, threadNum);
        Float32ToFloat16(floatOutput.data(), outputData, n * k);
    }

    void RunLinearFloat32Float16(float *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearFloat32Float16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops.push_back(new MultiThreadLinearFloat32Float16Op(inputData, weightData, biasData, outputData,
                                                n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    struct FastllmBF16Manager {
        std::vector <uint16_t> bf16Input;
        std::vector <uint16_t> bf16Weight;
    } fastllmBf16Manager;

    void RunLinearFloat32BFloat16(float *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <uint16_t> &bf16Input = fastllmBf16Manager.bf16Input;
        if (bf16Input.size() < n * m) {
            bf16Input.resize(n * m);
        }
        if (n > 4) {
            int per = n * m / threadNum;
            int cur = 0;
            std::vector<fastllm::MultiThreadFloat32ToBFloat16Op*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < n * m);
                if (i == threadNum - 1) {
                    end = n * m;
                }
                ops.push_back(new MultiThreadFloat32ToBFloat16Op(inputData + cur, bf16Input.data() + cur, end - cur));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(startTid + i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(startTid + i);
                delete ops[i];
            }
        } else {
            Float32ToBFloat16(inputData, bf16Input.data(), n * m);
        }
/*
        int stride = 64;
        std::vector<fastllm::MultiThreadLinearBFloat16BFloat16Op*> ops;
        for (int i = 0; i < k; i += stride) {
            ops.push_back(new MultiThreadLinearBFloat16BFloat16Op(bf16Input.data(), weightData, biasData, outputData,
                                                n, m, k, i, i + stride));
        }

        int per = ops.size() / threadNum, remain = ops.size() % threadNum;
        std::vector <std::deque <int> > tasks;
        tasks.resize(threadNum);
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int now = per + (i < remain);
            for (int j = cur; j < cur + now; j++) {
                tasks[i].push_back(j);
            }
            cur += now;
        }

        int pushTasks = 0;
        while (pushTasks < ops.size()) {
            for (int i = 0; i < threadNum; i++) {
                if (pool->TryWait(startTid + i)) {
                    if (tasks[i].size() > 0) {
                        int taskId = tasks[i].front();
                        tasks[i].pop_front();
// printf("thread %d do task [%d, %d)\n", i, ops[taskId]->st, ops[taskId]->end);
                        pool->PushOp(startTid + i, ops[taskId]);
                        pushTasks++;
                    } else {
                        int sel = -1, maxS = 0;
                        for (int j = 0; j < threadNum; j++) {
                            if (tasks[j].size() > maxS) {
                                maxS = tasks[j].size();
                                sel = j;
                            }
                        }
                        if (sel != -1) {
                            int taskId = tasks[sel].back();
                            tasks[sel].pop_back();
// printf("thread %d do task [%d, %d)\n", i, ops[taskId]->st, ops[taskId]->end);
                            pool->PushOp(startTid + i, ops[taskId]);
                            pushTasks++;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
        }
// printf("finish...\n");
        for (int i = 0; i < ops.size(); i++) {
            delete ops[i];
        }
*/
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearBFloat16BFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops.push_back(new MultiThreadLinearBFloat16BFloat16Op(bf16Input.data(), weightData, biasData, outputData,
                                                n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void RunLinearBFloat16BFloat16(uint16_t *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearBFloat16BFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops.push_back(new MultiThreadLinearBFloat16BFloat16Op(inputData, weightData, biasData, outputData,
                                                n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }
    
    void LaunchLinearInt8Int8(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
        int *weightSums, int *weightZeros, float *scales, float *bias,
        float *inputSums, float *iscales, float *izeros,
        std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops[startTid + i] = new MultiThreadLinearInt8Int8Op(a, b + cur * m, (int32_t*)c + cur, n, m, end - cur, k, 
                                                        weightSums + cur, weightZeros + cur, scales + cur, 
                                                        (bias == nullptr ? (float *) nullptr : bias + cur), 
                                                        iscales, izeros, inputSums);
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void RunLinearInt8Int8(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
                            int *weightSums, int *weightZeros, float *scales, float *bias,
                            float *inputSums, float *iscales, float *izeros,
                            AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearInt8Int8Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops.push_back(new MultiThreadLinearInt8Int8Op(a, b + cur * m, (int32_t*)c + cur, n, m, end - cur, k, 
                                                        weightSums + cur, weightZeros + cur, scales + cur, 
                                                        (bias == nullptr ? (float *) nullptr : bias + cur), 
                                                        iscales, izeros, inputSums));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void RunLinearInt8Int4Group(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, int group, int groupCnt,
                                int *weightSums, float *weightMins, float *scales, float *bias,
                                float *inputSums, float *iscales, float *izeros,
                                AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearInt8Int4GroupOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
            ops.push_back(new MultiThreadLinearInt8Int4GroupOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                (bias == nullptr ? (float *) nullptr : bias + cur), iscales, izeros,
                                inputSums, group, groupCnt));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void RunLinearFloat32Int8(float *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        weight.CalcWeightSum();
        std::vector<LowBitConfig> inputConfigs;
        std::vector<uint8_t> uinput;
        std::vector <float> inputSums, iscales, izeros;
        OnlineQuantization(inputData, uinput, inputConfigs, n, m, 1, m, inputSums, iscales, izeros, 0);

        RunLinearInt8Int8(uinput.data(), (uint8_t*)weight.cpuData, outputData, n, m, k, 
                weight.weightSum.data(), weight.zeros.data(), weight.scales.data(), biasData, 
                inputSums.data(), iscales.data(), izeros.data(),
                pool, startTid, threadNum);
        /*
        这部分是float输入，float输出
        int threadNum = threads;
        int per = k / threadNum;
        int cur = 0;
        std::vector<std::thread *> threads;
        for (int i = 0; i < threadNum - 1; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            threads.push_back(new std::thread(&Int8LinearPart, inputData, weightData, biasData, outputData,
                                                weight.perChannelsConfigs.data(), n, m, k, cur, end));
            cur = end;
        }
        Int8LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
        for (int i = 0; i < threadNum - 1; i++) {
            threads[i]->join();
            delete threads[i];
        }
        */
    }

    void RunLinearFloat32FP8E4M3(float *inputData, Data &weight, float *outputData, float *biasData, 
                    int n, int m, int k, 
                    AliveThreadPool *pool, int startTid, int threadNum) {
        // std::vector <uint16_t> bf16Input;
        // bf16Input.resize(n * m);
        // Float32ToBFloat16(inputData, bf16Input.data(), n * m);

        std::vector <uint16_t> &bf16Input = fastllmBf16Manager.bf16Input;
        if (bf16Input.size() < n * m) {
            bf16Input.resize(n * m);
        }
        
        if (n > 4) {
            int per = n * m / threadNum;
            int cur = 0;
            std::vector<fastllm::MultiThreadFloat32ToBFloat16Op*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < n * m);
                if (i == threadNum - 1) {
                    end = n * m;
                }
                ops.push_back(new MultiThreadFloat32ToBFloat16Op(inputData + cur, bf16Input.data() + cur, end - cur));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(startTid + i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(startTid + i);
                delete ops[i];
            }
        } else {
            Float32ToBFloat16(inputData, bf16Input.data(), n * m);
        }

        std::vector <uint16_t> &bf16Weight = fastllmBf16Manager.bf16Weight;
        if (bf16Weight.size() < k * m) {
            bf16Weight.resize(k * m);
        }

        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearBFloat16FP8E4M3Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearBFloat16FP8E4M3Op(bf16Input.data(), weight.cpuData, biasData, outputData,
                                                    n, m, k, cur, end, weight.scales.data(), weight.blockK, weight.blockM));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void LaunchLinearBFloat16FP8E4M3(uint16_t *inputData, Data &weight, float *outputData, float *biasData, 
        int n, int m, int k, 
        std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops[startTid + i] = new MultiThreadLinearBFloat16FP8E4M3Op(inputData, weight.cpuData, biasData, outputData,
                                    n, m, k, cur, end, weight.scales.data(), weight.blockK, weight.blockM);
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    void LaunchLinearBFloat16BFloat16(uint16_t *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops[startTid + i] = new MultiThreadLinearBFloat16BFloat16Op(inputData, (uint16_t*)weight.cpuData, biasData, outputData,
                                    n, m, k, cur, end);
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    void LaunchLinearFloat32Float16(float *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum) {
                                    int per = k / threadNum;
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            if (i == threadNum - 1) {
                end = k;
            }
            ops[startTid + i] = new MultiThreadLinearFloat32Float16Op(inputData, (uint16_t*)weight.cpuData, biasData, outputData,
                                    n, m, k, cur, end);
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    struct FastllmQuantManager {
        std::vector <uint8_t> uinput;
    } fastllmQuantManager;

    void RunLinearFloat32Int4Group(float *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, int group, int groupCnt,
                                AliveThreadPool *pool, int startTid, int threadNum) {
        weight.CalcWeightSum();
        std::vector<LowBitConfig> inputConfigs;
        std::vector <float> inputSums, iscales, izeros;
        std::vector <uint8_t> &uinput = fastllmQuantManager.uinput;
        OnlineQuantization(inputData, uinput, inputConfigs, n, m, group, groupCnt, inputSums, iscales, izeros, 1);
        RunLinearInt8Int4Group(uinput.data(), (uint8_t*)weight.cpuData, outputData, n, m, k,
                                group, groupCnt, weight.weightSum.data(), weight.mins.data(), weight.scales.data(), 
                                biasData, inputSums.data(), iscales.data(), izeros.data(),
                                pool, startTid, threadNum);
    }

    struct MultiThreadLinearFloat32Int2GroupOp : MultiThreadBaseOp {
        float *inputData;
        Data *weight;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadLinearFloat32Int2GroupOp(float *inputData, Data *weight, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weight(weight), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            int group = weight->group, groupCnt = weight->groupCnt;
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    for (; l < m; l++) {
                        int gid = j * group + (l / groupCnt);
                        float scale = weight->scales[gid];
                        float min = weight->mins[gid];
                        uint8_t w = weight->cpuData[(j * m + l) / 4];
                        w = (w >> ((3 - l % 4) * 2)) & 3;
                        now += inputData[i * m + l] * (min + scale * w);
                    }
                    outputData[i * k + j] = now;
                }
            }
        }
    };

    void RunLinearFloat32Int2Group(float *inputData, Data &weight, float *outputData, float *biasData, 
        int n, int m, int k, int group, int groupCnt,
        AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearFloat32Int2GroupOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearFloat32Int2GroupOp(inputData, &weight, biasData, outputData,
                                                    n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void RunLinearFloat16Float16(uint16_t *inputData, uint16_t *weightData, uint16_t *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearFloat16Float16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearFloat16Float16Op(inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
    }

    void RunLinearFloat16Int8(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
                            int n, int m, int k, AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <float> floatInput, floatOutput;
        floatInput.resize(n * m);
        floatOutput.resize(n * k);
        Float16ToFloat32(inputData, floatInput.data(), n * m);
        RunLinearFloat32Int8(floatInput.data(), weight, floatOutput.data(), biasData, n, m, k, pool, startTid, threadNum);
        Float32ToFloat16(floatOutput.data(), outputData, n * k);
    }

    void RunLinearFloat16Int4Group(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
                            int n, int m, int k, int group, int groupCnt,
                            AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <float> floatInput, floatOutput;
        floatInput.resize(n * m);
        floatOutput.resize(n * k);
        Float16ToFloat32(inputData, floatInput.data(), n * m);
        RunLinearFloat32Int4Group(floatInput.data(), weight, floatOutput.data(), biasData, n, m, k, group, groupCnt, pool, startTid, threadNum);
        Float32ToFloat16(floatOutput.data(), outputData, n * k);
    }

    void RunLinearFloat16FP8E4M3(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
        int n, int m, int k, 
        AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <float> floatOutput;
        floatOutput.resize(n * k);
        
        std::vector <uint16_t> bf16Input;
        bf16Input.resize(n * m);
        Float16ToBFloat16(inputData, bf16Input.data(), n * m);

        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearBFloat16FP8E4M3Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearBFloat16FP8E4M3Op(bf16Input.data(), weight.cpuData, biasData, floatOutput.data(), 
                                                    n, m, k, cur, end, weight.scales.data(), weight.blockK, weight.blockM));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }

        Float32ToFloat16(floatOutput.data(), outputData, n * k);
    }

    void RunLinearFloat16GGUF(uint16_t *inputData, uint8_t *weightData, uint16_t *outputData, float *biasData, 
                                Data *weight, int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        std::vector <float> floatInput, floatOutput;
        floatInput.resize(n * m);
        floatOutput.resize(n * k);
        Float16ToFloat32(inputData, floatInput.data(), n * m);
        RunLinearFloat32GGUF(floatInput.data(), weightData, floatOutput.data(), biasData, weight, n, m, k, pool, startTid, threadNum);
        Float32ToFloat16(floatOutput.data(), outputData, n * k);
    }

    void LaunchLinearQ8KGGUF(uint8_t *a, uint8_t *b, float *c, float *bias, Data *weight, 
                            int n, int m, int k,
                            std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum) {
        weight->Repack();
        int rows = 8;
        int ks = (k / rows);
        int per = ks / threadNum;
        int cur = 0;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < ks);
            if (i == threadNum - 1) {
                end = ks;
            }
            ops[startTid + i] = new MultiThreadLinearFloat32GGUFOp(a, b, bias, c, weight->ggmlTensor, n, m, k, cur * rows, end * rows);
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    struct GGUFMemoryManager {
        std::vector <uint8_t> q8kInputs;
    } ggufMemoryManager;

    void RunLinearFloat32GGUF(float *inputData, uint8_t *weightData, float *outputData, float *biasData, 
                                Data *weight, int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum) {
        weight->Repack();
        ggml_tensor *tensor = (ggml_tensor*)weight->ggmlTensor;
        // printf("gguf tensor %s: %s\n", tensor->name.c_str(), ggml_type_name(tensor->type));
        
        std::vector <uint8_t> &q8kInputs = ggufMemoryManager.q8kInputs;
        int rowCount = ggml_row_size(ggml_type_vec_dot_type(tensor->type), m);
        if (ggml_is_quantized(tensor->type)) {
            if (q8kInputs.size() < n * rowCount) {
                q8kInputs.resize(n * rowCount);
            }
            if (n > 1) {
                std::vector<fastllm::MultiThreadFloat32ToQ8KOp*> ops;
                int per = n / threadNum;
                int cur = 0;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < n);
                    ops.push_back(new MultiThreadFloat32ToQ8KOp(
                        inputData + cur * m, (uint8_t*)(q8kInputs.data() + cur * rowCount), (end - cur) * m, tensor->type));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }
            } else {
                for (int i = 0; i < n; i++) {
                    iqk_quantize_row_q8_K (
                        inputData + i * m, q8kInputs.data() + i * rowCount, m, 
                        ggml_type_vec_dot_type(tensor->type), tensor->type
                    );
                }
            }

            int rows = 8;
            int ks = (k / rows);
            int per = ks / threadNum;
            int cur = 0;
            std::vector<fastllm::MultiThreadLinearFloat32GGUFOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < ks);
                ops.push_back(new MultiThreadLinearFloat32GGUFOp((uint8_t*)q8kInputs.data(), weightData, biasData, outputData,
                                                        (void*)tensor, n, m, k, cur * rows, i == threadNum - 1 ? k : end * rows));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(startTid + i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(startTid + i);
                delete ops[i];
            }        
            return;
        }
/*
// 下面这段是反量化再计算的代码
        std::vector <float> floatWeight;
        floatWeight.resize(k * m);
        dequantize_row_q4_K((block_q4_K*)weightData, floatWeight.data(), k * m);
        
        int per = k / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadLinearFloat32Float32Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            ops.push_back(new MultiThreadLinearFloat32Float32Op(inputData, floatWeight.data(), biasData, outputData,
                                                    n, m, k, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(startTid + i);
            delete ops[i];
        }
*/
    }

#ifdef __AVX2__
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        int ans = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i xors = _mm256_set1_epi8(-128);
        for (; i + 31 < n; i += 32) {
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

            by = _mm256_xor_si256(by, xors);
            by = _mm256_add_epi8(by, _mm256_and_si256(_mm256_cmpeq_epi8(by, xors), ones8));

            by = _mm256_sign_epi8(by, bx);
            bx = _mm256_sign_epi8(bx, bx);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
        }
        for (; i < n; i++) {
            ans += ((int8_t*)a)[i] * ((int)b[i] - 128);
        }

        return ans + I32sum(acc);
    };
#endif

    void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, 
                            int n, int m, int k, int st, int end) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (n > 1 && n < 8) {
            const int BLOCKA = 7;
            float16x8_t va[BLOCKA], vb, vc[BLOCKA];
        
            for (int i = 0; i < n; i += BLOCKA) {
                int cur = std::min(BLOCKA, n - i);
                for (int j = st; j < end; j++) {
                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = vdupq_n_f16(0.0f);
                    }
                        
                    for (int k = 0; k < m; k += 8) {
                        for (int l = 0; l < cur; l++) {
                            va[l] = vld1q_f16((float16_t*)inputData + (i + l) * m + k);
                        }
                        vb = vld1q_f16((float16_t*)weightData + j * m + k);
                                    
                        for (int l1 = 0; l1 < cur; l1++) {
                            vc[l1] = vfmaq_f16(vc[l1], va[l1], vb);
                        }
                    }

                    for (int l0 = 0; l0 < cur; l0++) {
                        float now = vc[l0][0] + vc[l0][1] + vc[l0][2] + vc[l0][3] + 
                                    vc[l0][4] + vc[l0][5] + vc[l0][6] + vc[l0][7];
                        if (biasData != nullptr) {
                            now += biasData[j];
                        }
                        outputData[(i + l0) * k + j] = float_to_half(now);
                    }
                }
            }
        } else if (n > 1) {
            const int BN = 64, BM = 64, BK = 64;
            uint16_t *a = new uint16_t[BN * BM];
            uint16_t *b = new uint16_t[BK * BM];
            uint16_t *c = new uint16_t[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(uint16_t));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                        }

                        const int BLOCKB = 8;
                        float16x8_t va, vb[BLOCKB], vc[BLOCKB];

                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j += BLOCKB) {
                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = vdupq_n_f16(0.0f);
                                }

                                for (int k = 0; k < BM; k += 8) {
                                    va = vld1q_f16((float16_t*)a + i * BM + k);
                                    for (int l = 0; l < BLOCKB; l++) {
                                        vb[l] = vld1q_f16((float16_t*)b + (j + l) * BM + k);
                                    }
                                    
                                    for (int l1 = 0; l1 < BLOCKB; l1++) {
                                        vc[l1] = vfmaq_f16(vc[l1], va, vb[l1]);
                                    }
                                }

                                float16x8x2_t temp0 = vtrnq_f16(vc[0], vc[1]);
                                float16x8x2_t temp1 = vtrnq_f16(vc[2], vc[3]);
                                float16x8x2_t temp2 = vtrnq_f16(vc[4], vc[5]);
                                float16x8x2_t temp3 = vtrnq_f16(vc[6], vc[7]);

                                vc[0] = vaddq_f16(temp0.val[0], temp0.val[1]);
                                vc[1] = vaddq_f16(temp1.val[0], temp1.val[1]);
                                vc[2] = vaddq_f16(temp2.val[0], temp2.val[1]);
                                vc[3] = vaddq_f16(temp3.val[0], temp3.val[1]);

                                float32x4x2_t temp4 = vtrnq_f32(vreinterpretq_f32_f16(vc[0]), vreinterpretq_f32_f16(vc[1]));
                                float32x4x2_t temp5 = vtrnq_f32(vreinterpretq_f32_f16(vc[2]), vreinterpretq_f32_f16(vc[3]));

                                vc[0] = vaddq_f16(vreinterpretq_f16_f32(temp4.val[0]), vreinterpretq_f16_f32(temp4.val[1]));
                                vc[1] = vaddq_f16(vreinterpretq_f16_f32(temp5.val[0]), vreinterpretq_f16_f32(temp5.val[1]));

                                vst1q_f16((float16_t*)c + i * BK + j, vcombine_f16(vadd_f16(vget_high_f16(vc[0]), vget_low_f16(vc[0])), vadd_f16(vget_high_f16(vc[1]), vget_low_f16(vc[1]))));

                                /*for (int l0 = 0; l0 < BLOCKA; l0++) {
                                        for (int l1 = 0; l1 < BLOCKB; l1++) {
                                            float now = vc[l0][l1][0] + vc[l0][l1][1] + vc[l0][l1][2] + vc[l0][l1][3] + 
                                                vc[l0][l1][4] + vc[l0][l1][5] + vc[l0][l1][6] + vc[l0][l1][7];
                                            c[(i + l0) * block + (j + l1)] = float_to_half(now);
                                        }
                                }*/
                            }
                        }
/*
                            for (int i = 0; i < block && i < (nend - nst); i++) {
                                for (int j = 0; j < block; j++) {
                                    float now = 0.0;
                                    int k = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                                    for (; k + 7 < block; k += 8) {
                                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)a + i * block + k),
                                                            vld1q_f16((float16_t*)b + j * block + k));
                                    }
                                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif
                                    for (; k < block; k++) {
                                        now += fp16tofp32.dict[a[i * block + k]] * fp16tofp32.dict[b[j * block + k]];
                                    }
                                    c[i * block + j] = float_to_half(now);
                                }
                            }
*/

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                            for (; j + 7 < kend; j += 8) {
                                vst1q_f16((float16_t*)outputData + i * k + j, vaddq_f16(vld1q_f16((float16_t*)outputData + i * k + j), vld1q_f16((float16_t*)c + (i - nst) * BK + (j - kst))));
                            }
#endif
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
/*
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        float now = biasData ? biasData[j] : 0.0f;
                        for (int l = 0; l < m; l++) {
                            now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                        }
{
    float a = half_to_float(outputData[i * k + j]);
    if (st == 0 && fabs(a - now) > 1e-1) {
        printf("wrong %d %d %f %f\n", i, j, a, now);
        exit(0);
    }
}
                        outputData[i * k + j] = float_to_half(now);
                    }
                }
*/
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (; l + 7 < m; l += 8) {
                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)inputData + i * m + l),
                                            vld1q_f16((float16_t*)weightData + j * m + l));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif
                    for (; l < m; l++) {
                        now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = float_to_half(now);
                }
            }
        }
#elif defined(__AVX2__)
        if (n > 8) {
            const int BN = 64, BM = 128, BK = 64;
            uint16_t *a = new uint16_t[BN * BM];
            uint16_t *b = new uint16_t[BK * BM];
            uint16_t *c = new uint16_t[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(uint16_t));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        memcpy(a + (i - nst) * BM, inputData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            memcpy(b + (i - kst) * BM, weightData + i * m + mst, (mend - mst) * sizeof(uint16_t));
                        }

                        const int BLOCKB = 8;
                        __m256 va, vb[BLOCKB], vc[BLOCKB];

                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j += BLOCKB) {
                                for (int l1 = 0; l1 < BLOCKB; l1++) {
                                    vc[l1] = _mm256_setzero_ps();
                                }

                                for (int k = 0; k < BM; k += 8) {
                                    va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (a + i * BM + k)));
                                    for (int l = 0; l < BLOCKB; l++) {
                                        vb[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (b + (j + l) * BM + k)));
                                    }
                                    
                                    for (int l1 = 0; l1 < BLOCKB; l1++) {
                                        vc[l1] = _mm256_fmadd_ps(va, vb[l1], vc[l1]);
                                    }
                                }

                                __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
                                __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;

                                __t0 = _mm256_unpacklo_ps(vc[0], vc[1]);
                                __t1 = _mm256_unpackhi_ps(vc[0], vc[1]);
                                __t2 = _mm256_unpacklo_ps(vc[2], vc[3]);
                                __t3 = _mm256_unpackhi_ps(vc[2], vc[3]);
                                __t4 = _mm256_unpacklo_ps(vc[4], vc[5]);
                                __t5 = _mm256_unpackhi_ps(vc[4], vc[5]);
                                __t6 = _mm256_unpacklo_ps(vc[6], vc[7]);
                                __t7 = _mm256_unpackhi_ps(vc[6], vc[7]);
                                __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
                                __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
                                __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
                                __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
                                __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
                                __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
                                __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
                                __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
                                
                                __m256 sum = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x20));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt0, __tt4, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt1, __tt5, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt2, __tt6, 0x31));
                                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(__tt3, __tt7, 0x31));
                                
                                _mm_storeu_si128 ((__m128i*)(c + i * BK + j), _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT));
                            }
                        }

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
                            for (; j + 7 < kend; j += 8) {
                                __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (outputData + i * k + j)));
                                __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (c + (i - nst) * BK + (j - kst))));
                                __m256 vsum = _mm256_add_ps(va, vb);
                                _mm_storeu_si128 ((__m128i*)(outputData + i * k + j), _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
                            }
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + fp16tofp32.dict[c[(i - nst) * BK + (j - kst)]]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
        } else if (n > 0) {
            const int BLOCKA = 8;
            __m256 va[BLOCKA], vb, vc[BLOCKA];
        
            for (int i = 0; i < n; i += BLOCKA) {
                int cur = std::min(BLOCKA, n - i);
                for (int j = st; j < end; j++) {
                    for (int l1 = 0; l1 < cur; l1++) {
                        vc[l1] = _mm256_setzero_ps();
                    }
                        
                    for (int k = 0; k < m; k += 8) {
                        for (int l = 0; l < cur; l++) {
                            va[l] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (inputData + (i + l) * m + k)));
                        }
                        vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + k)));
                                    
                        for (int l1 = 0; l1 < cur; l1++) {
                            vc[l1] = _mm256_fmadd_ps(va[l1], vb, vc[l1]);
                        }
                    }

                    for (int l0 = 0; l0 < cur; l0++) {
                        float now = Floatsum(vc[l0]);
                        if (biasData != nullptr) {
                            now += biasData[j];
                        }
                        outputData[(i + l0) * k + j] = float_to_half(now);
                    }
                }
            }
        }
#else
        if (n > 3) {
            const int BN = 64, BM = 64, BK = 64;
            float *a = new float[BN * BM];
            float *b = new float[BK * BM];
            float *c = new float[BN * BK];
            if (biasData == nullptr) {
                for (int i = 0; i < n; i++) {
                    memset(outputData + i * k + st, 0, (end - st) * sizeof(uint16_t));
                }
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = st; j < end; j++) {
                        outputData[i * k + j] = float_to_half(biasData[j]);
                    }
                }
            }

            for (int mst = 0; mst < m; mst += BM) {
                int mend = std::min(mst + BM, m);
                memset(c, 0, BN * BK * sizeof(float));
                for (int nst = 0; nst < n; nst += BN) {
                    int nend = std::min(nst + BN, n);
                    for (int i = nst; i < nend; i++) {
                        Float16ToFloat32(inputData + i * m + mst, a + (i - nst) * BM, mend - mst);
                    }
                    for (int kst = st; kst < end; kst += BK) {
                        int kend = std::min(kst + BK, end);
                        for (int i = kst; i < kend; i++) {
                            Float16ToFloat32(weightData + i * m + mst, b + (i - kst) * BM, mend - mst);
                        }
                        
                        for (int i = 0; i < BN && i < (nend - nst); i++) {
                            for (int j = 0; j < BK; j++) {
                                float now = 0.0;
                                for (int k = 0; k < BM; k++) {
                                    now += a[i * BM + k] * b[j * BM + k];
                                }
                                c[i * BK + j] = now;
                            }
                        }

                        for (int i = nst; i < nend; i++) {
                            int j = kst;
                            for (; j < kend; j++) {
                                outputData[i * k + j] = (float_to_half)(fp16tofp32.dict[outputData[i * k + j]] + c[(i - nst) * BK + (j - kst)]);
                            }
                        }
                    }
                }
            }
            delete[] a;
            delete[] b;
            delete[] c;
        } else {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
                    for (; l < m; l++) {
                        now += fp16tofp32.dict[inputData[i * m + l]] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = float_to_half(now);
                }
            }
        }
#endif
    }

#ifdef __ARM_FEATURE_DOTPROD
        inline static void MatMulInt8Int8RunSomeBlock(uint8_t *weightWalk, uint8_t *inputStart, int32_t *c, 
                            int curBlock, uint32x4_t *sum, uint8x16_t *vi, 
                            int block, int k, int m, int kstride) {
                for (int i = 0; i < k; i++) {
                    std::vector <int> values = std::vector <int> (curBlock, 0);
                    uint8_t *inputWalk = inputStart;
                    int j = 0;

                    for (int j = 0; j < curBlock; j++) {
                        sum[j][0] = sum[j][1] = sum[j][2] = sum[j][3] = 0;
                    }
                    for (; j + 15 < m; j += 16) {
                        for (int x = 0; x < curBlock; x++) {
                            vi[x] = vld1q_u8(inputWalk + m * x);
                        }
                        uint8x16_t vw = vld1q_u8(weightWalk);
                        for (int x = 0; x < curBlock; x++) {
                            sum[x] = vdotq_u32(sum[x], vi[x], vw);
                        }
                        inputWalk += 16;
                        weightWalk += 16;
                    }
                    for (int x = 0; x < curBlock; x++) {
                        values[x] += sum[x][0] + sum[x][1] + sum[x][2] + sum[x][3];
                    }

                    for (; j < m; j++) {
                        int curWeight = (int)(*(weightWalk++));
                        for (int x = 0; x < curBlock; x++) {
                            values[x] += curWeight * (*(inputWalk + x * m));
                        }
                        inputWalk++;
                    }

                    for (int x = 0; x < curBlock; x++) {
                        c[(block + x) * kstride + i] = values[x];
                    }
                }
        }
#endif

    void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x) for (; block + (x - 1) < n; block += (x)) MatMulInt8Int8RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
            int block = 0;
            uint32x4_t sum[16];
            uint8x16_t vi[16];
            RUNBLOCK(16);
            RUNBLOCK(8);RUNBLOCK(7);RUNBLOCK(6);RUNBLOCK(5);
            RUNBLOCK(4);RUNBLOCK(3);RUNBLOCK(2);RUNBLOCK(1);
#undef RUNBLOCK
#elif defined(__aarch64__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;

                    int per = 64;
                    int cnt = m / per;
                    int sur = m % per;

                    uint32x4_t sum = {0};
                    uint16x8_t temp = {0};
                    uint16x8_t temp1 = {0};
                    uint16x8_t temp2 = {0};
                    uint16x8_t temp3 = {0};
                    uint16x8_t temp4 = {0};
                    uint16x8_t temp5 = {0};
                    uint16x8_t temp6 = {0};
                    uint16x8_t temp7 = {0};

                    while (cnt--) {
                        temp = vmull_u8(vld1_u8(inputWalk), vld1_u8(weightWalk));
                        temp1 = vmull_u8(vld1_u8(inputWalk + 8), vld1_u8(weightWalk + 8));
                        temp2 = vmull_u8(vld1_u8(inputWalk + 16), vld1_u8(weightWalk + 16));
                        temp3 = vmull_u8(vld1_u8(inputWalk + 24), vld1_u8(weightWalk + 24));
                        temp4 = vmull_u8(vld1_u8(inputWalk + 32), vld1_u8(weightWalk + 32));
                        temp5 = vmull_u8(vld1_u8(inputWalk + 40), vld1_u8(weightWalk + 40));
                        temp6 = vmull_u8(vld1_u8(inputWalk + 48), vld1_u8(weightWalk + 48));
                        temp7 = vmull_u8(vld1_u8(inputWalk + 56), vld1_u8(weightWalk + 56));

                        sum = vpadalq_u16(sum, temp);
                        sum = vpadalq_u16(sum, temp1);
                        sum = vpadalq_u16(sum, temp2);
                        sum = vpadalq_u16(sum, temp3);
                        sum = vpadalq_u16(sum, temp4);
                        sum = vpadalq_u16(sum, temp5);
                        sum = vpadalq_u16(sum, temp6);
                        sum = vpadalq_u16(sum, temp7);

                        inputWalk += per;
                        weightWalk += per;
                    }

                    value += (sum[0] + sum[1] + sum[2] + sum[3]);
                    while (sur--) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#elif defined(__AVX2__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *inputWalk = inputStart;

                    c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                    weightWalk += m;
                }
            }
#else
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    for (int j = 0; j < m; j++) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#endif
    }
}