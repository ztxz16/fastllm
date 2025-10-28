//
// Created by huangyuyang on 6/14/23.
//

#ifndef FASTLLM_CUDADEVICE_H
#define FASTLLM_CUDADEVICE_H

#include "device.h"

namespace fastllm {
    void DoCudaAttentionReshape(Data &q, Data &v, Data &output);
    void DoCudaAttention(Data &q, Data &k, Data &v, Data &mask, Data &output, int group, float scale, int maskType);
    void DoCudaAttentionBatchReshape(Data **qs, Data **vs, Data **outputs, int batch);
    void DoCudaAttentionBatch(Data **qs, Data **ks, Data **vs, Data **masks, Data **outputs, int group, float scale, int batch);
    void DoCudaLinearReshape(Data &input, Data &weight, Data &output);
    void DoCudaLinear(Data &input, Data &weight, const Data &bias, Data &output);
    void DoCudaSwigluReshape(Data &input, Data &output);
    void DoCudaSwiglu(Data &input, Data &output);
    void DoCudaSplitReshape(Data &input, int axis, int start, int end, Data &output);
    void DoCudaSplit(Data &input, int axis, int start, int end, Data &output);
    void DoCudaCatDirect(Data &input0, Data &input1, int axis);
    void DoCudaCatDirectBatch(Data **input0s, Data **input1s, int batch, int axis);
    void DoCudaPermuteSelf(Data &input, const std::vector <int> &axis);
    void DoCudaMergeMOE(Data &input, Data &output, Data &gateBias, Data &logits, Data &w1, Data &w2, Data &w3, 
        Data **weights, Data **biass, int topk, int needNorm, float sharedScale, float routeScale);
    
    class CudaDevice : BaseDevice {
    public:
        CudaDevice ();

        bool Malloc (void **ret, size_t size); // 分配尺寸为size的空间
        bool Free(void *ret); // 释放ret

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class CudaToFloat16 : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaToFloat32 : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaConvertToFloat16 : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaConvertToFloat32 : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAttention : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMergeAttention : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCopyKVCacheOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaLayerNormOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaEmbedding : CpuEmbedding {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaRMSNormOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaConv1DPerChannel : CpuConv1DPerChannel {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaConv2DOp : CpuConv2DOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaRecurrentGatedDeltaRuleOp : CpuRecurrentGatedDeltaRuleOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaLinearOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSplitOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaRepeatOp : CpuRepeatOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCatOp : CpuCatOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCatDirectOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMatMulOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMatMulTransBOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSoftMaxOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaExpOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaReluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaGeluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaGeluNewOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSiluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSigmoidOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMambaSoftplusOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSwigluOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAddOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMulOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAddToOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMulToOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCausalMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaTransferAttnOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCumSumLastDimOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMakeDecayMaskOp : CpuMakeDecayMaskOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAttentionMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAlibiMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaTopKOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaPermuteSelfOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaLlamaRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaLlamaRotatePosition2DPartOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaNearlyRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaRepeatPenaltyOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaApplyLognAttnOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMergeMOE : CpuMergeMOE {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMergeMLA : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSplitBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCatBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMulBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMatMulBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaMatMulTransBBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaSoftmaxBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaCatDirectBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAppendKVCacheBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CudaAttentionBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif //FASTLLM_CUDADEVICE_H
