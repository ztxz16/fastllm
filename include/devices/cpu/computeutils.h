//
// Created by huangyuyang on 8/14/24.
//

#ifndef FASTLLM_COMPUTE_LINEAR_H
#define FASTLLM_COMPUTE_LINEAR_H

#include <cstdint>
#include <atomic>

#include "fastllm.h"

namespace fastllm {
    void FastllmGemm (int n, int m, int k, 
        const void *A, long lda, // A [n * m], lda = bytes for 1 row in A
        const void *B, long ldb, // B [k * m], ldb = bytes for 1 row in B
        void *C, long ldc, // C[n * k], ldc = bytes for 1 row in C
        int st, int end, // calc C[0 : n, st : end]
        DataType AType, DataType BType, DataType CType);
    
    struct MultiThreadGemmOp : MultiThreadBaseOp {
        uint8_t *inputData;   // [n * m]
        uint8_t *weightData;  // [k * m]
        uint8_t *outputData;  // [n * k]
        DataType inputDataType, weightDataType, outputDataType;
        int n, m, k, st, end;
        MultiThreadGemmOp(uint8_t *inputData, DataType inputDataType,
                        uint8_t *weightData, DataType weightDataType,
                        uint8_t *outputData, DataType outputDataType,
                        int n, int m, int k, int st, int end) :
            inputData(inputData), inputDataType(inputDataType),
            weightData(weightData), weightDataType(weightDataType),
            outputData(outputData), outputDataType(outputDataType),
            n(n), m(m), k(k), st(st), end(end) {}
        void Run();
    };

    struct MultiThreadReduceBatchOp : MultiThreadBaseOp {
        uint8_t *downOutData;
        DataType downOutDataType;
        float *weights;
        float *lastOutput;
        int *pos;
        int bsz, k;
        int hidden_size;
        int batch_st, batch_end;  // batch维度的范围
        int hidden_st, hidden_end; // hidden维度的范围
        
        MultiThreadReduceBatchOp(uint8_t *downOutData, DataType downOutDataType,
            float *weights, float *lastOutput,
            int *pos, int bsz, int k, 
            int hidden_size, 
            int batch_st, int batch_end,
            int hidden_st, int hidden_end) : 
            downOutData(downOutData), downOutDataType(downOutDataType),
            weights(weights), lastOutput(lastOutput),
            pos(pos), bsz(bsz), k(k),
            hidden_size(hidden_size), 
            batch_st(batch_st), batch_end(batch_end),
            hidden_st(hidden_st), hidden_end(hidden_end) {}
        
        void Run();
    };

    struct MultiThreadRepackWeightsOp : MultiThreadBaseOp {
        Data **weights;
        int st, end;      

        MultiThreadRepackWeightsOp (Data **weights, int st, int end) : 
            weights(weights), st(st), end(end) {}

        void Run();
    };

    void MultiThreadReduceBatch(uint8_t *downOutData, DataType downOutDataType,
                    float *weights, float *lastOutput, int *pos, int bsz, int k, int hidden_size);

    void ConvertFromFloat32(void *dstData, DataType dstDataType, const float *floatData, size_t rows, size_t columns);

    // 新增一个专门的Op来处理数据类型转换
    struct MultiThreadConvertFromFloat32Op : MultiThreadBaseOp {
        void *dstData;
        DataType dstDataType;
        const float *floatData;
        size_t columns;
        size_t startRow, endRow;  // 处理的行范围 [startRow, endRow)
        MultiThreadConvertFromFloat32Op(void *dstData, DataType dstDataType, 
                                    const float *floatData, size_t columns,
                                    size_t startRow, size_t endRow) :
            dstData(dstData), dstDataType(dstDataType), 
            floatData(floatData), columns(columns),
            startRow(startRow), endRow(endRow) {}

        void Run();
    };

    // 对应的多线程运行函数
    void RunMultiThreadConvertFromFloat32(void *dstData, DataType dstDataType, 
                                                const float *floatData, size_t rows, 
                                                size_t columns, AliveThreadPool *pool);

    struct WorkStealingOp : MultiThreadBaseOp {
        struct alignas(64) TaskState {
            std::atomic<int> curr;
            int end;
            std::vector<MultiThreadBaseOp*> tasks;
            std::atomic<bool> completed;
        };
        
        int threadId;
        std::vector<TaskState*>* allStates;
        TaskState* myState;
        int totalThreads;
        
        WorkStealingOp(int tid, std::vector<TaskState*>* states, 
                    TaskState* state, int numThreads) 
            : threadId(tid), allStates(states), 
            myState(state), totalThreads(numThreads) {}
        
        void Run() override;
        
    private:
        void processOwnTasks();
        
        void stealFromOthers();
    };

    // 重构的动态任务调度函数，支持work-stealing
    void DynamicScheduleTasks(std::vector<MultiThreadBaseOp*>& ops);

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

    struct MultiThreadLinearFloat32GGUFOp : MultiThreadBaseOp {
        void *ggmlTensor;
        uint8_t *q8kInputData, *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadLinearFloat32GGUFOp(uint8_t *q8kInputData, uint8_t *weightData, float *biasData, float *outputData, void *ggmlTensor,
                           int n, int m, int k, int st, int end) : 
            q8kInputData(q8kInputData), weightData(weightData), biasData(biasData), outputData(outputData), ggmlTensor(ggmlTensor),
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

    struct MultiThreadLinearBFloat16BFloat16Op : MultiThreadBaseOp {
        uint16_t *inputData;
        uint16_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadLinearBFloat16BFloat16Op(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run();
    };
    
    struct MultiThreadLinearBFloat16FP8E4M3Op : MultiThreadBaseOp {
        uint16_t *inputData;
        uint8_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;
        int blockK, blockM;
        float *scales;

        MultiThreadLinearBFloat16FP8E4M3Op(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                int n, int m, int k, int st, int end, 
                float *scales, int blockK, int blockM) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end), 
            scales(scales), blockK(blockK), blockM(blockM) {}

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

    void LaunchLinearQ8KGGUF(uint8_t *a, uint8_t *b, float *c, float *bias, Data *weight, 
                            int n, int m, int k,
                            std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum);
    void LaunchLinearInt8Int8(uint8_t *a, uint8_t *b, float *c, int n, int m, int k, 
        int *weightSums, int *weightZeros, float *scales, float *bias,
        float *inputSums, float *iscales, float *izeros,
        std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum);
    void LaunchLinearBFloat16FP8E4M3(uint16_t *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum);
    void LaunchLinearBFloat16BFloat16(uint16_t *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum);
    void LaunchLinearFloat32Float16(float *inputData, Data &weight, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool, int startTid, int threadNum);

    void RunLinearFloat32Float32(float *inputData, float *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16Float32(uint16_t *inputData, float *weightData, uint16_t *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Float16(float *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32BFloat16(float *inputData, uint16_t *weightData, float *outputData, float *biasData, 
                                int n, int m, int k, 
                                AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearBFloat16BFloat16(uint16_t *inputData, uint16_t *weightData, float *outputData, float *biasData, 
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
    void RunLinearFloat32FP8E4M3(float *inputData, Data &weight, float *outputData, float *biasData, 
                            int n, int m, int k, 
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Int4Group(float *inputData, Data &weight, float *outputData, float *biasData, 
                            int n, int m, int k, int group, int groupCnt,
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32Int2Group(float *inputData, Data &weight, float *outputData, float *biasData, 
                            int n, int m, int k, int group, int groupCnt,
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat32GGUF(float *inputData, uint8_t *weightData, float *outputData, float *biasData, 
                            Data *weight, int n, int m, int k, 
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
    void RunLinearFloat16FP8E4M3(uint16_t *inputData, Data &weight, uint16_t *outputData, float *biasData, 
                            int n, int m, int k, 
                            AliveThreadPool *pool, int startTid, int threadNum);
    void RunLinearFloat16GGUF(uint16_t *inputData, uint8_t *weightData, uint16_t *outputData, float *biasData, 
                            Data *weight, int n, int m, int k, 
                            AliveThreadPool *pool, int startTid, int threadNum);

    void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, 
                            int n, int m, int k, int st, int end);

    void MatMulInt8Int8(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride);

    bool LinearBFloat16_FP8E4M3PERCHANNEL_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearBFloat16_FP8E4M3BLOCK128_Kernel(uint16_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8PERCHANNEL_INT8PERCHANNEL_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8PERCHANNEL_INT4PERCHANNEL_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearINT8GROUP128_INT4GROUP128_Kernel(uint8_t *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end);
    bool LinearQ8K_GGUF_Kernel(uint8_t *q8kInputData, uint8_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end, DataType AType, DataType BType);
}

#endif // FASTLLM_COMPUTE_LINEAR_H