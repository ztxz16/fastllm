//
// Created by huangyuyang on 8/14/24.
//

#ifndef FASTLLM_COMPUTE_LINEAR_H
#define FASTLLM_COMPUTE_LINEAR_H

#include <cstdint>
#include "fastllm.h"

namespace fastllm {
    struct MultiThreadLinearFloat32Float32Op : MultiThreadBaseOp {
        float *inputData;
        float *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadLinearFloat32Float32Op(float *inputData, float *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run();
    };

    struct MultiThreadLinearFloat32Float16Op : MultiThreadBaseOp {
        float *inputData;
        uint16_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadLinearFloat32Float16Op(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run();
    };

    struct MultiThreadLinearInt8Int8Op : MultiThreadBaseOp {
        uint8_t *a;
        uint8_t *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums, *weightZeros;
        float *scales, *bias;
        float *iscales, *izeros, *inputSums;

        MultiThreadLinearInt8Int8Op(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride, 
                                    int *weightSums, int *weightZeros, float *scales, float *bias,
                                    float *iscales, float *izeros, float *inputSums) : 
            a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride),
            weightSums(weightSums), weightZeros(weightZeros), scales(scales), bias(bias),
            iscales(iscales), izeros(izeros), inputSums(inputSums) {}

        void Run();
    };

    struct MultiThreadLinearInt8Int4GroupOp : MultiThreadBaseOp {
        uint8_t *a, *b;
        float *c;
        int n, m, k, kstride;
        int *weightSums;
        float *weightMins;
        float *scales;
        float *bias;
        float *iscales, *izeros;
        float *inputSums;
        int group, groupCnt;

        MultiThreadLinearInt8Int4GroupOp(
                uint8_t *a, uint8_t *b, float *c, int n, int m, int k, int kstride,
                int *weightSums, float *weightMins, float *scales, float *bias,
                float *iscales, float *izeros, float *inputSums, int group, int groupCnt
        ) :
                a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride),
                weightSums(weightSums), weightMins(weightMins), scales(scales), bias(bias),
                iscales(iscales), izeros(izeros), inputSums(inputSums), group(group), groupCnt(groupCnt) {}

        void Run();
    };

    struct MultiThreadLinearFloat16Float16Op : MultiThreadBaseOp {
        uint16_t *inputData;
        uint16_t *weightData;
        float *biasData;
        uint16_t *outputData;
        int n, m, k, st, end;

        MultiThreadLinearFloat16Float16Op(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run();
    };

    void RunLinearFloat32Float32(float *inputData, float *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16Float32(uint16_t *inputData, float *weightData, uint16_t *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Float16(float *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearInt8Int8(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
                            int *weightSums, int *weightZeros, float *scales, float *bias,
                            float *inputSums, float *iscales, float *izeros,
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearInt8Int4Group(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, int group, int groupCnt,
                                int *weightSums, float *weightMins, float *scales, float *bias,
                                float *inputSums, float *iscales, float *izeros,
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Int8(float *inputData, Data &weight, float *outputData, float *biasData, 
                            int n, int m, int k, 
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Int4Group(float *inputData, Data &weight, float *outputData, float *biasData, 
                            int n, int m, int k, int group, int groupCnt,
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16Float16(uint16_t *inputData, uint16_t *weightData, uint16_t *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16Int8(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
                            int n, int m, int k, 
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16Int4Group(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
                            int n, int m, int k, int group, int groupCnt,
                            AliveThreadPool *pool, int startTid, int threadNum);

    void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, 
                            int n, int m, int k, int st, int end);

    void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride);
}

#endif // FASTLLM_COMPUTE_LINEAR_H