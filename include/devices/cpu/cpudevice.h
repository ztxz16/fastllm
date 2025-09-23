//
// Created by huangyuyang on 6/13/23.
//

#ifndef FASTLLM_CPUDEVICE_H
#define FASTLLM_CPUDEVICE_H

#include "device.h"
#include "alivethreadpool.h"

namespace fastllm {
    void DoCpuLinearReshape(Data &input, Data &weight, Data &output);
    void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output);

    void DoCpuSwigluReshape(Data &input, Data &output);
    void DoCpuSwiglu(Data &input, Data &output);
    
    void DoCpuCatDirect(Data &input0, Data &input1, int axis);

    struct MultiThreadFloat32ToBFloat16Op : MultiThreadBaseOp {
        float *input;
        uint16_t *output;
        int len;

        MultiThreadFloat32ToBFloat16Op (float *input, uint16_t *output, int len) :
                input(input), output(output), len(len) {}

        void Run();
    };

    struct MultiThreadFloat32ToQ8KOp : MultiThreadBaseOp {
        float *input;
        uint8_t *output;
        int len;
        int ggmlType;

        MultiThreadFloat32ToQ8KOp (float *input, uint8_t *output, int len, int ggmlType) :
                input(input), output(output), len(len), ggmlType(ggmlType) {}

        void Run();
    };

    struct MultiThreadOnlineQuantizationOp : MultiThreadBaseOp {
        float *input;
        uint8_t *output;
        LowBitConfig *configs;
        int n, m, group, groupCnt;
        float *inputSums, *iscales, *izeros;
        int permuteType;

        MultiThreadOnlineQuantizationOp (float *input, uint8_t *output, LowBitConfig *configs, int n, int m, int group, int groupCnt,
                                        float *inputSums, float *iscales, float *izeros, 
                                        int permuteType) :
                input(input), output(output), configs(configs), n(n), m(m), group(group), groupCnt(groupCnt), 
                inputSums(inputSums), iscales(iscales), izeros(izeros),
                permuteType(permuteType) {} ;

        void Run();
    };

    struct MultiThreadSwigluGptOssOp : MultiThreadBaseOp {
        float *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadSwigluGptOssOp (float *input, int mid, int len, float *output,
                             int n, int inputStride, int outputStride) :
            input(input), mid(mid), len(len), output(output),
            n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run();
    };

    struct MultiThreadSwigluOp : MultiThreadBaseOp {
        float *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadSwigluOp (float *input, int mid, int len, float *output,
                             int n, int inputStride, int outputStride) :
            input(input), mid(mid), len(len), output(output),
            n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run();
    };

    struct MultiThreadSwigluFloat16Op : MultiThreadBaseOp {
        uint16_t *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadSwigluFloat16Op (uint16_t *input, int mid, int len, uint16_t *output,
                             int n, int inputStride, int outputStride) :
            input(input), mid(mid), len(len), output(output),
            n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run();
    };

    struct MultiThreadInt4GroupLinearOp : MultiThreadBaseOp {
        float *inputData;
        uint8_t *weightData;
        float *biasData, *outputData;
        uint16_t *mins, *scales;
        int n, m, k, st, end, group, groupCnt;

        MultiThreadInt4GroupLinearOp(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                                uint16_t *mins, uint16_t *scales, int n, int m, int k, int st, int end, int group, int groupCnt) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData), mins(mins), scales(scales),
            n(n), m(m), k(k), st(st), end(end), group(group), groupCnt(groupCnt) {}

        void Run();
    };
    
    // Causal Mask: mask[i][j] = (lastlen + i < j)
    struct MultiThreadSoftmaxOp : MultiThreadBaseOp {
        float *value;
        int n, m, lastlen;

        MultiThreadSoftmaxOp(float *value, int n, int m, int lastlen) : value(value), n(n), m(m), lastlen(lastlen) {}

        void Run();
    };

    struct MultiThreadSiluOp : MultiThreadBaseOp {
        float *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadSiluOp (float *input, int len, float *output,
                           int n, int inputStride, int outputStride) :
                input(input), len(len), output(output),
                n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run();
    };

    struct MultiThreadGeluOp : MultiThreadBaseOp {
        float *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadGeluOp (float *input, int len, float *output,
                           int n, int inputStride, int outputStride) :
                input(input), len(len), output(output),
                n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run();
    };

    // q布局: [qlen, qdim]
    // k布局: [klen, qdim]
    // v布局: [klen, vdim]
    // output布局: [qlen, vdim]
    // Causal Mask: mask[i][j] = (lastlen + i < j)
    struct MultiThreadSingleAttentionCausalOp : MultiThreadBaseOp {
        float *qd, *kd, *vd, *od;
        float scale;
        int qlen, qdim, lastlen, klen, vdim;

        MultiThreadSingleAttentionCausalOp(float *qd, float *kd, float *vd, float *od,
                                           float scale, int qlen, int qdim, int lastlen, int klen, int vdim) :
                qd(qd), kd(kd), vd(vd), od(od),
                scale(scale), qlen(qlen), qdim(qdim),
                lastlen(lastlen), klen(klen), vdim(vdim) {}

        void Run();
    };

    void SiluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool);

    void GeluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool);

    void SoftmaxMultiThread(float *input, int n, int m, int lastlen, AliveThreadPool *pool);
    void SwigluMultiThread(float *input, int mid, int len, float *output,
        int n, int inputStride, int outputStride, AliveThreadPool *pool);
    void SwigluMultiThreadFloat16(uint16_t *input, int mid, int len, uint16_t *output,
        int n, int inputStride, int outputStride, AliveThreadPool *pool);
    void SwigluGptOssMultiThread(float *input, int mid, int len, float *output,
        int n, int inputStride, int outputStride, AliveThreadPool *pool);

    void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, float *c, int n, int m, int k,
        int *weightSums, float *weightMins, float *scales, float *bias,
        std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros,
        std::vector <LowBitConfig> &configs, int startTid, int threadNum, int group, int groupCnt,
        std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool);


    class CpuDevice : BaseDevice {
    public:
        CpuDevice ();

        bool Malloc (void **ret, size_t size); // 分配尺寸为size的空间
        using BaseDevice::Malloc;
        bool Free(void *ret); // 释放ret

        bool CopyDataToCPU(void *dst, void *src, size_t size); // 不重要, cpu device不会进行这个操作
        using BaseDevice::CopyDataToCPU;
        bool CopyDataFromCPU(void *dst, void *src, size_t size); // 不重要, cpu device不会进行这个操作
        using BaseDevice::CopyDataFromCPU;

        int threads = 4;
    };

    class CpuToFloat16 : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuToFloat32 : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuConvertToFloat16 : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuConvertToFloat32 : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAttention : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    protected:
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMergeMOE : BaseOperator {
    protected:
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMergeMLA : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuEmbedding : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuLayerNormOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRMSNormOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRMSNormExOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuLinearOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    protected:
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    // inputChannels = outputChannels = group的conv1d
    // 逐通道conv, 首用于Qwen3_next中的GDN结构
    class CpuConv1DPerChannel : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    protected:
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuConv2DOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    protected:
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSplitOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCatOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRepeatOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCatDirectOp : BaseOperator {
        protected:
            void  Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMatMulOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMatMulTransBOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSoftMaxOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuNormalizeOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSiluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuTanHOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuReluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSigmoidOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuExpOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuGeluOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuGeluNewOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSwigluOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSwigluGptOssOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMambaSoftplusOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMulOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMulToOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAddOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAddToOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuTransferAttnOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRecurrentGatedDeltaRuleOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCausalMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
    
    class CpuAttentionMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAttentionExtendedMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAlibiMaskOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuTopKOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuPermuteOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuPermuteSelfOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuNearlyRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuLlamaRotatePosition2DOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuLlamaRotatePosition2DPartOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuRepeatPenaltyOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuApplyLognAttnOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCumSumLastDimOp : BaseOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMakeDecayMaskOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCopyKVCacheOp : BaseOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSplitBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCatBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMulBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMatMulBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuMatMulTransBBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuSoftmaxBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuCatDirectBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAppendKVCacheBatchOp : BaseBatchOperator {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class CpuAttentionBatchOp : BaseBatchOperator {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif //FASTLLM_CPUDEVICE_H
