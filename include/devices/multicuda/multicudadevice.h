//
// Created by huangyuyang on 8/2/24.
//

#ifndef FASTLLM_MULTICUDADEVICE_H
#define FASTLLM_MULTICUDADEVICE_H

#include "device.h"

namespace fastllm {
    class MultiCudaDevice : BaseDevice {
        
    private:
        CudaDevice *cudaDevice;

    public:
        MultiCudaDevice (CudaDevice *cudaDevice);

        bool Malloc (void **ret, size_t size); // 分配尺寸为size的空间
        bool Free(void *ret); // 释放ret

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);

        // 是否可以运行某一个算子
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行形状推理
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行推理
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLinearOp : CudaLinearOp {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaToFloat16 : CudaToFloat16 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaToFloat32 : CudaToFloat32 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaConvertToFloat16 : CudaConvertToFloat16 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaConvertToFloat32 : CudaConvertToFloat32 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaToBFloat16 : CudaToBFloat16 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaConvertToBFloat16 : CudaConvertToBFloat16 {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaSoftMaxOp : CudaSoftMaxOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaSigmoidOp : CudaSigmoidOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaSelectExpertOp : CudaSelectExpertOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLinearAddOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLinearSwigluOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaRMSNormOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaRMSNormPartOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaAddToOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaCatOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaSplitOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaSwigluOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaPermuteSelfOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLlamaRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLlamaRotatePosition2DPartOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaRopeEncodingOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaYarnRopeEncodingOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaQwen35InterleavedRopeOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaDeepSeekV4Op : BaseOperator {
    public:
        explicit MultiCudaDeepSeekV4Op(BaseOperator *cudaOp) : cudaOp(cudaOp) {}

    private:
        BaseOperator *cudaOp;
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    // DeepSeek-V4.1 专用算子的多卡包装。除注意力（按 query head 切分）之外，
    // V4.1 的压缩 KV、两级 indexer 与 Hyper-Connections 系数在每张卡上各算一份
    // （复制），因此绝大多数算子只需要把输入/输出换成本卡副本后原样下发。
    class MultiCudaDeepSeekV41Op : BaseOperator {
    public:
        explicit MultiCudaDeepSeekV41Op(BaseOperator *cudaOp) : cudaOp(cudaOp) {}

    private:
        BaseOperator *cudaOp;
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        // V4.1 有若干算子只有 output 没有 input（IndexerScore / CandidateBlocks /
        // SparseAttention ...），BaseOperator 的默认 Reshape 会解引用不存在的
        // "input" 项。形状由每卡下发时的 cudaOp->Reshape 决定，这里留空。
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    // 复制布局下的 CatDirect：V4.1 的压缩 KV / indexer key / 滑窗 KV 缓存都是
    // 每卡一份，追加必须落到每张卡的本地缓存上，否则副本会与 root 不一致。
    class MultiCudaCatDirectOp : CudaCatDirectOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    // 标量乘：复制布局下逐卡就地执行，切分布局下每卡只处理本地分片。
    class MultiCudaMulOp : CudaMulOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaAppendPagedCacheOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaAttentionPagedOp : CudaAttentionPagedOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaAttentionPagedBatchOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaQKVRMSNormRopeSplitAppendPagedCacheOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMLPOp : CudaLinearOp {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMergeMOE : CpuMergeMOE {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMergeAttention : CudaMergeAttention {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif //FASTLLM_MULTICUDADEVICE_H
