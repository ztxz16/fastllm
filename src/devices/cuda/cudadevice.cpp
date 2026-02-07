//
// Created by huangyuyang on 6/14/23.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"

#include "fastllm-cuda.cuh"

#include "utils.h"

namespace fastllm {
    CudaDevice::CudaDevice() {
        this->deviceType = "cuda";
        this->ops["ToFloat16"] = (BaseOperator*)(new CudaToFloat16());
        this->ops["ToFloat32"] = (BaseOperator*)(new CudaToFloat32());
        this->ops["ConvertToFloat16"] = (BaseOperator*)(new CudaConvertToFloat16());
        this->ops["ConvertToFloat32"] = (BaseOperator*)(new CudaConvertToFloat32());

        this->ops["Attention"] = (BaseOperator*)(new CudaAttention());
        // this->ops["MergeAttention"] = (BaseOperator*)(new CudaMergeAttention());
        this->ops["CopyKVCache"] = (BaseOperator*)(new CudaCopyKVCacheOp());
        this->ops["Embedding"] = (BaseOperator*)(new CudaEmbedding());
        this->ops["LayerNorm"] = (BaseOperator*)(new CudaLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CudaRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CudaLinearOp());
        this->ops["Conv1DPerChannel"] = (BaseOperator*)(new CudaConv1DPerChannel());
        this->ops["Conv2D"] = (BaseOperator*)(new CudaConv2DOp());
        this->ops["Split"] = (BaseOperator*)(new CudaSplitOp());
        this->ops["Repeat"] = (BaseOperator*)(new CudaRepeatOp());
        this->ops["Cat"] = (BaseOperator*)(new CudaCatOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CudaCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CudaMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CudaMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CudaSoftMaxOp());
        this->ops["Exp"] = (BaseOperator*)(new CudaExpOp());
        this->ops["Relu"] = (BaseOperator*)(new CudaReluOp());
        this->ops["Gelu"] = (BaseOperator*)(new CudaGeluOp());
        this->ops["GeluNew"] = (BaseOperator*)(new CudaGeluNewOp());
        this->ops["Silu"] = (BaseOperator*)(new CudaSiluOp());
        this->ops["Sigmoid"] = (BaseOperator*)(new CudaSigmoidOp());
        this->ops["MambaSoftplus"] = (BaseOperator*)(new CudaMambaSoftplusOp());
        this->ops["Swiglu"] = (BaseOperator*)(new CudaSwigluOp());
        this->ops["Add"] = (BaseOperator*)(new CudaAddOp());
        this->ops["Mul"] = (BaseOperator*)(new CudaMulOp());
        this->ops["AddTo"] = (BaseOperator*)(new CudaAddToOp());
        this->ops["MulTo"] = (BaseOperator*)(new CudaMulToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CudaAttentionMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new CudaAlibiMaskOp());
        this->ops["TransferAttn"] = (BaseOperator*)(new CudaTransferAttnOp());
        this->ops["CumSumLastDim"] = (BaseOperator*)(new CudaCumSumLastDimOp());
        this->ops["TopK"] = (BaseOperator*)(new CudaTopKOp());
        this->ops["SelectExpert"] = (BaseOperator*)(new CudaSelectExpertOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CudaPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new CudaRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new CudaNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CudaLlamaRotatePosition2DOp());
        this->ops["LlamaRotatePosition2DPart"] = (BaseOperator*)(new CudaLlamaRotatePosition2DPartOp());
        this->ops["RepeatPenalty"] = (BaseOperator*)(new CudaRepeatPenaltyOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new CudaApplyLognAttnOp());
        this->ops["MergeMOE"] = (BaseOperator*)(new CudaMergeMOE());
        this->ops["MergeMLA"] = (BaseOperator*)(new CudaMergeMLA());
        this->ops["MergeMLAPaged"] = (BaseOperator*)(new CudaMergeMLAPaged());
        this->ops["RecurrentGatedDeltaRule"] = (BaseOperator*)(new CudaRecurrentGatedDeltaRuleOp());
        this->ops["CausalMask"] = (BaseOperator*)(new CudaCausalMaskOp());
        this->ops["MakeDecayMask"] = (BaseOperator*)(new CudaMakeDecayMaskOp());

        this->ops["SplitBatch"] = (BaseOperator*)(new CudaSplitBatchOp());
        this->ops["CatBatch"] = (BaseOperator*)(new CudaCatBatchOp());
        this->ops["MulBatch"] = (BaseOperator*)(new CudaMulBatchOp());
        this->ops["MatMulBatch"] = (BaseOperator*)(new CudaMatMulBatchOp());
        this->ops["MatMulTransBBatch"] = (BaseOperator*)(new CudaMatMulTransBBatchOp());
        this->ops["SoftMaxBatch"] = (BaseOperator*)(new CudaSoftmaxBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator*)(new CudaCatDirectBatchOp());
        this->ops["AppendKVCachebatch"] = (BaseOperator*)(new CudaAppendKVCacheBatchOp());
        this->ops["AttentionBatch"] = (BaseOperator*)(new CudaAttentionBatchOp());

        this->ops["AttentionPaged"] = (BaseOperator*)(new CudaAttentionPagedOp());
        this->ops["AppendPagedCache"] = (BaseOperator*)(new CudaAppendPagedCacheOp());
        this->ops["AppendPagedCacheBatch"] = (BaseOperator*)(new CudaAppendPagedCacheBatchOp());
        this->ops["AttentionPagedBatch"] = (BaseOperator*)(new CudaAttentionPagedBatchOp());
        this->ops["GeneratePagedBatchParams"] = (BaseOperator*)(new CudaGeneratePagedBatchParamsOp());
        this->ops["GenerateAppendPagedCacheBatchParams"] = (BaseOperator*)(new CudaGenerateAppendPagedCacheBatchParamsOp());
    }

    bool CudaDevice::Malloc(void **ret, size_t size) {
        *ret = FastllmCudaMalloc(size);
        return true;
    }

    bool CudaDevice::Free(void *ret) {
        FastllmCudaFree(ret);
        return true;
    }

    bool CudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool CudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    void CudaToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT16) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT32) {
            float *old = (float*)data.cudaData;
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            data.cudaData = FastllmCudaMalloc(data.GetBytes());
            int len = data.Count(0);
            FastllmFloatToHalf(old, data.cudaData, len);
            FastllmCudaFree(old);
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
    }

    void CudaToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT32) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT16) {
            uint16_t *old = (uint16_t*)data.cudaData;
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            data.cudaData = FastllmCudaMalloc(data.GetBytes());
            int len = data.Count(0);
            FastllmHalfToFloat(old, data.cudaData, len);
            FastllmCudaFree(old);
        } else {
            ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
        }
    }

    void CudaConvertToFloat16::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT16;
        output->Resize(input->dims);
        output->UpdateUnitSize();
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CudaConvertToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT16) {
            FastllmCudaCopyFromDeviceToDevice(output.cudaData, input.cudaData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT32) {
            FastllmFloatToHalf(input.cudaData, output.cudaData, input.Count(0));
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
    }

    void CudaConvertToFloat32::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT32;
        output->Resize(input->dims);
        output->UpdateUnitSize();
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CudaConvertToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT32) {
            FastllmCudaCopyFromDeviceToDevice(output.cudaData, input.cudaData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT16) {
            FastllmHalfToFloat(input.cudaData, output.cudaData, input.Count(0));
        } else {
            ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
        }
    }

    void DoCudaAttentionReshape(Data &q, Data &v, Data &output) {
        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    void CudaAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];

        AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
        AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
        AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
        AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
        AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");

        AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType,
                        "Attention: q, k, v's datatype should be same.\n");
        AssertInFastLLM(q.dataType == DataType::FLOAT32 ||
                        q.dataType == DataType::FLOAT16, 
                        "Attention's input's type should be float32 or float16.\n");

        DoCudaAttentionReshape(q, v, output);
    }

    void DoCudaAttention(Data &q, Data &k, Data &v, Data &mask, Data &output, int group, float scale, int maskType) {
        output.Allocate();
        if (q.dataType == DataType::FLOAT32) {
            FastllmCudaAttention(q, k, v, mask, output, group, scale, maskType);
        } else if (q.dataType == DataType::FLOAT16) {
#ifdef CUDA_NO_TENSOR_CORE
            Data q32, k32, v32, mask32, output32;
            ToDataType(q, q32, DataType::FLOAT32);
            ToDataType(k, k32, DataType::FLOAT32);
            ToDataType(v, v32, DataType::FLOAT32);
            ToDataType(output, output32, DataType::FLOAT32);
            if (mask.dims.size() > 0)
                ToDataType(mask, mask32, DataType::FLOAT32);
            FastllmCudaAttention(q32, k32, v32, mask32, output32, group, scale, maskType);
            ToDataType(output32, output, DataType::FLOAT16);
#else
            FastllmCudaHalfAttention(q, k, v, mask, output, group, scale, maskType);
#endif
        }
    }

    void CudaAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data emptyData;
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &mask = datas.find("mask")->second ? *(datas.find("mask")->second) : emptyData;
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        int maskType = intParams.find("maskType") != intParams.end() ? intParams.find("maskType")->second : 0;

        DoCudaAttention(q, k, v, mask, output, group, scale, maskType);
    }

    void CudaCopyKVCacheOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return;
    }

    void CudaCopyKVCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &oldCache = *(datas.find("oldCache")->second);
        Data &newCache = *(datas.find("newCache")->second);

        int oldBsStart = intParams.find("oldBsStart") != intParams.end() ? intParams.find("oldBsStart")->second : -1;
        int newBsStart = intParams.find("newBsStart") != intParams.end() ? intParams.find("newBsStart")->second : -1;
        int bs = intParams.find("bs") != intParams.end() ? intParams.find("bs")->second : -1;
        int offset = intParams.find("offset") != intParams.end() ? intParams.find("offset")->second : -1;

        int unitSize = oldCache.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) newCache.cudaData + newBsStart * newCache.strides[0] * unitSize
                                                                          + offset * newCache.strides[1] * unitSize,
                                          newCache.strides[0] * unitSize,
                                          (uint8_t *) oldCache.cudaData + oldBsStart * oldCache.strides[0] * unitSize,
                                          oldCache.strides[0] * unitSize,
                                          oldCache.dims[1] * oldCache.dims[2] * unitSize, bs);
    }

    bool CudaRMSNormOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    void CudaRMSNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16,
                        "RMSNorm error: datatype should be float32 or float16.");

        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        FastllmCudaRMSNorm(input, weight, output, eps);
    }

    bool CudaLayerNormOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.strides[axis];
        return inner == 1;
    }

    void CudaLayerNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        output.Allocate();
        FastllmCudaLayerNorm(input, gamma, beta, output, axis);
    }

    bool CudaEmbedding::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (GetLowMemMode() || !GetCudaEmbedding()) {
            return false;
        }
        Data &input = *(datas.find("input")->second);
        if (input.dataType != DataType::FLOAT32) {
            return false;
        }
        return true;
    }

    void CudaEmbedding::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);;
        output.Allocate();

        FastllmCudaEmbedding(input, weight, output);
    }

    void CudaConv1DPerChannel::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        int inputChannels = intParams.find("inputChannels")->second;    
        int outputChannels = intParams.find("outputChannels")->second; 
        int kernelSize = intParams.find("kernel")->second;
        int padding = intParams.find("pad")->second; 
        int stride = intParams.find("stride")->second;
        int groups = inputChannels;  // 组数等于通道数，实现逐通道卷积

        output.Allocate();
        FastllmCudaConv1DPerChannelFloat32(input, weight, bias, inputChannels, outputChannels, kernelSize, stride, padding, output);
    }

    void CudaConv2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelH = intParams.find("kernelH")->second;
        int kernelW = intParams.find("kernelW")->second;
        int padH = intParams.find("padH")->second;
        int padW = intParams.find("padW")->second;
        int strideH = intParams.find("strideH")->second;
        int strideW = intParams.find("strideW")->second;
        
        std::vector <int> dims = input.dims;
        int inputHeight = dims[2], inputWidth = dims[3];
        int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
        int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

        FastllmCudaConv2DFloat32(input, weight, bias, inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, output);
    }

    void DoCudaLinearReshape(Data &input, Data &weight, Data &output) {
        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");
        DoCudaLinearReshape(input, weight, output);
    }

    bool CudaLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        return true;
    }

    void DoCudaLinear(Data &input, Data &weight, const Data &bias, Data &output) {
        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        if (bias.dataType != DataType::FLOAT32) {
            ErrorInFastLLM("Linear error: unsupport bias' dataType.\n");
        } else if (input.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT32) {
                FastllmCudaHalfMatMulFloat32(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FLOAT16) {
                FastllmCudaHalfMatMulFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::BFLOAT16) {
                FastllmCudaHalfMatMulBFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT8) {
                FastllmCudaHalfMatMulFloatInt8(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_GROUP) {
                FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_NOZERO) {
                FastllmCudaHalfMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FP8_E4M3) {
                FastllmCudaHalfMatMulFloatFP8E4M3(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                FastllmCudaHalfMatMulGGUF(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                FastllmCudaMatMulFloat32(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FLOAT16) {
                FastllmCudaMatMulFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::BFLOAT16) {
                FastllmCudaMatMulBFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT8) {
                FastllmCudaMatMulFloatInt8(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4) {
                FastllmCudaMatMulFloatInt4(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_NOZERO) {
                FastllmCudaMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_GROUP) {
                FastllmCudaMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FP8_E4M3) {
                FastllmCudaMatMulFloatFP8E4M3(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                FastllmCudaMatMulFloatGGUF(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FP8_E4M3_BLOCK_128) {
                FastllmCudaMatMulFloatFP8E4M3Block128(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            if (weight.dataType == DataType::BFLOAT16) {
                FastllmCudaBFloat16MatMulBFloat16(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType for BFLOAT16 input.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
        }
    }

    void CudaLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        DoCudaLinear(input, weight, bias, output);
    }

    void DoCudaSplitReshape(Data &input, int axis, int start, int end, Data &output) {
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaSplitOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;
        DoCudaSplitReshape(input, axis, start, end, output);
    }

    void DoCudaSplit(Data &input, int axis, int start, int end, Data &output) {
        output.Allocate();

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t*)output.cudaData, outputStride * unitSize,
                                          (uint8_t*)input.cudaData + start * inner * unitSize, inputStride * unitSize,
                                          (end - start) * inner * unitSize, outer);
    }

    void CudaSplitOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;
        DoCudaSplit(input, axis, start, end, output);
    }

    void CudaRepeatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        output.Allocate();

        int outer = output.Count(0) / output.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;
        FastllmCudaRepeat(input.cudaData, output.cudaData, outer, repeatTimes, inputStride * unitSize, outputStride * unitSize, channels * inner * unitSize, channels * inner * unitSize);
    }

    void CudaCatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.CopyFrom(input1);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.CopyFrom(input0);
            return;
        }

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = output.Count(0) / output.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) output.cudaData, outputStride * unitSize,
                                            (uint8_t *) input0.cudaData, input0Stride * unitSize,
                                            input0.dims[axis] * inner * unitSize, outer);
        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) output.cudaData + input0.dims[axis] * inner * unitSize, outputStride * unitSize,
                                            (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                            input1.dims[axis] * inner * unitSize, outer);
    }

    void DoCudaCatDirect(Data &input0, Data &input1, int axis) {
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                                (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "Cat's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expansionDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData, input0Stride * unitSize,
                                              (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                              input1.dims[axis] * inner * unitSize, outer);
            return;
        }

        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmCudaMemcpy2DDeviceToDevice((uint8_t *) input0.cudaData + oldDims[axis] * inner * unitSize,
                                          input0Stride * unitSize,
                                          (uint8_t *) input1.cudaData, input1Stride * unitSize,
                                          input1.dims[axis] * inner * unitSize, outer);
    }

    void CudaCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        DoCudaCatDirect(input0, input1, axis);
    }

    void CudaMatMulOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMul error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                        "MatMul's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMul's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims[input1.dims.size() - 2],
                        "MatMul's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMul: input0.dims[1] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMul's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 1];

        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CudaMatMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        FastllmCudaBatchMatMul(input0, input1, output,
                               input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                               batch1, n, m, k, alpha);
    }

    void CudaMatMulTransBOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMulTransB error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
                        "MatMulTransB's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMulTransB's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims.back(),
                        "MatMulTransB's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMulTransB: input0.dims[0] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMulTransB's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 2];
        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CudaMatMulTransBOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        FastllmCudaBatchMatMulTransB(input0, input1, output,
                                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                                     batch1, n, m, k, alpha);
    }

    bool CudaSoftMaxOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int inner = input.Count(axis + 1);
        if (inner != 1) {
            return false;
        }
        return true;
    }

    void CudaSoftMaxOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16, 
                        "Softmax error: Data's type should be float32 or float16.\n");
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        FastllmCudaSoftmax(input, output, axis);
    }

    void CudaGeluNewOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");
        FastllmCudaGeluNew(input, output);
    }

    void CudaGeluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Gelu error: Data's type should be float32 or float16.\n");
        FastllmCudaGelu(input, output);
    }

    void CudaExpOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16, 
                "Exp error: Data's type should be float32 or float16\n");
        FastllmCudaExp(input, output);
    }

    void CudaReluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Relu error: Data's type should be float32\n");
        FastllmCudaRelu(input, output);
    }

    void DoCudaSwigluReshape(Data &input, Data &output) {
        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        DoCudaSwigluReshape(input, output);
    }

    void DoCudaSwiglu(Data &input, Data &output) {
        output.Allocate();
        FastllmCudaSwiglu(input, output);
    }

    void CudaSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Swiglu error: Data's type should be float32 or float16.\n");
        DoCudaSwiglu(input, output);
    }

    void CudaSiluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Silu error: Data's type should be float32 or float16.\n");
        FastllmCudaSilu(input, output);
    }

    void CudaSigmoidOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Sigmoid error: Data's type should be float32 or float16.\n");
        FastllmCudaSigmoid(input, output);
    }

    void CudaMambaSoftplusOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &aLogData = *(datas.find("aLog")->second);
        Data &dtBiasData = *(datas.find("dtBias")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "CudaMambaSoftplusOp error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(aLogData.dataType == DataType::FLOAT32 && dtBiasData.dataType == DataType::FLOAT32,
                        "CudaMambaSoftplusOp error: alog's type and dtbias's type should be float32.\n");
        
        FastllmCudaMambaSoftplus(input, output, aLogData, dtBiasData);
    }

    void CudaAddOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Mul error: Data's type should be float32 or float16.\n");
        FastllmCudaAdd(input, v, output);
    }

    void CudaMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Mul error: Data's type should be float32 or float16.\n");
        FastllmCudaMul(input, v, output);
    }

    void CudaAddToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "AddTo error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");
        FastllmCudaAddTo(input0, input1, alpha);
    }

    void CudaMulToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "MulTo error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims == input1.dims || input1.Count(0) == 1
                        || input0.Count(0) % input1.Count(0) == 0, "MulTo error: input's shape should be same.\n");
        FastllmCudaMulTo(input0, input1, alpha);
    }

    void CudaAttentionMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        FastllmCudaAttentionMask(input, mask, maskValue);
    }

    void CudaAlibiMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        FastllmCudaAlibiMask(input, mask, maskValue);
    }

    void CudaTransferAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        FastllmCudaTransferAttn(input);
    }

    void CudaCumSumLastDimOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        FastllmCudaCumSumLastDim(input);
    }

    void CudaTopKOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "TopK error: Data's type should be float32.\n");
        
        int dimsLen = input.dims.size();
        std::vector<int> dims = input.dims;
        dims[dimsLen - 1] = topk * 2;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    bool CudaTopKOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        if (topk > 50) {
            return false;
        }
        return true;
    }

    void CudaTopKOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : -1;
        FastllmCudaTopK(input, output, topk);
    }

    void CudaSelectExpertOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &logits = *(datas.find("logits")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

        AssertInFastLLM(logits.dataType == DataType::FLOAT32, "SelectExpert error: logits's type should be float32.\n");
        
        int dimsLen = logits.dims.size();
        int n = logits.Count(0) / logits.dims[dimsLen - 1]; // number of tokens
        int numExperts = logits.dims[dimsLen - 1]; // number of experts

        // Index output: [n, topk]
        index.dataType = DataType::INT32;
        index.Resize({n, topk});
        
        // Score output: [n, topk]
        score.dataType = DataType::FLOAT32;
        score.Resize({n, topk});
    }

    bool CudaSelectExpertOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        if (topk > 50) {
            return false; // 回退到 CPU 实现
        }
        return true;
    }

    void CudaSelectExpertOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &logits = *(datas.find("logits")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data *gateBias = datas.find("gateBias") != datas.end() ? datas.find("gateBias")->second : nullptr;
        
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        bool needNorm = intParams.find("needNorm") != intParams.end() ? (intParams.find("needNorm")->second != 0) : false;
        float routeScale = floatParams.find("routeScale") != floatParams.end() ? floatParams.find("routeScale")->second : 1.0f;
        
        index.Allocate();
        score.Allocate();
        
        bool success = FastllmCudaSelectExpert(logits, gateBias, index, score, topk, needNorm, routeScale);
        if (!success) {
            ErrorInFastLLM("CudaSelectExpert failed, topk may be too large (> 50).\n");
        }
    }

    void DoCudaPermuteSelf(Data &input, const std::vector <int> &axis) {
        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[0] == 1 && input.dims[1] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        same |= ((axis == std::vector <int>{1, 0, 2, 3}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{1, 2, 0, 3}) && input.dims[1] == 1 && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1}) && (input.dims[1] == 1 || input.dims[2] == 1));
        if (same) {
            std::vector<int> new_dims;
            for (int i = 0; i < axis.size(); i++) {
                new_dims.push_back(input.dims[axis[i]]);
            }
            input.Resize(new_dims);
            return;
        }
        FastllmCudaPermute(input, axis);
    }

    void CudaPermuteSelfOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                                input.dataType == DataType::FLOAT16,
                                "Permute error: datatype should be float32 or float16.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
        DoCudaPermuteSelf(input, axis);
    }

    void CudaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

        FastllmCudaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }

    void CudaNearlyRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;
        int positionStride = intParams.find("positionStride") != intParams.end() ? intParams.find("positionStride")->second : 1;

        FastllmCudaNearlyRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim, positionStride);
    }

    void CudaLlamaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        FastllmCudaLlamaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }

    void CudaLlamaRotatePosition2DPartOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int part = intParams.find("part") != intParams.end() ? intParams.find("part")->second : 128;

        FastllmCudaLlamaRotatePosition2DPart(data, positionIds, sinData, cosData, rotaryDim, part);
    }

    void CudaRepeatPenaltyOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &penalty = *(datas.find("penalty")->second);
        Data &penaltyScale = *(datas.find("penaltyScale")->second);
        AssertInFastLLM(input.dataType == DataType::FLOAT32 && penalty.dataType == DataType::FLOAT32 && penaltyScale.dataType == DataType::FLOAT32,
                        "Repeat Penalty error: Data's type should be float32.\n");
        FastllmCudaRepeatPenalty(input, penalty, penaltyScale);
    }

    void CudaApplyLognAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &lognAttn = *(datas.find("lognAttn")->second);
        Data &positionIds = *(datas.find("positionIds")->second);

        FastllmCudaApplyLognAttn(input, lognAttn, positionIds);
    }

    void CudaRecurrentGatedDeltaRuleOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &g = *(datas.find("g")->second);
        Data &b = *(datas.find("b")->second);
        Data &last_recurrent_state = *(datas.find("last_recurrent_state")->second);
        Data &core_attn_out = *(datas.find("core_attn_out")->second);
        core_attn_out.Allocate();
        FastllmRecurrentGatedDeltaRule(q, k, v, g, b, last_recurrent_state, core_attn_out);
    }

    void CudaCausalMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int base = intParams.find("base") != intParams.end() ? intParams.find("base")->second : 0;
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        FastllmCudaCausalMask(input, base, maskValue);
    }

    void CudaMakeDecayMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        FastllmCudaMakeDecayMask(input, output);
    }

    void CudaMergeMLA::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &output = *(datas.find("output")->second);
        // int b = qNope.dims[0], s = q_nope.dims[1], h = q_nope.dims[2], d = q_nope.dims[3], c = qNope.dims.back();
        output.dataType = qNope.dataType;
        output.Resize(qNope.dims);
    }

    void CudaMergeMLA::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &qPe = *(datas.find("qPe")->second);
        Data &kvCache = *(datas.find("kvCache")->second);
        Data &peCache = *(datas.find("peCache")->second);
        Data &mask = *(datas.find("mask")->second);
        Data &output = *(datas.find("output")->second);
        float softmaxScale = floatParams.find("softmaxScale") != floatParams.end() ? floatParams.find("softmaxScale")->second : 1.0f;        
        int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
        output.Allocate();

        // qNope: {b * s, h, c}
        // qPe: {b, s, h, r}
        // kvCache : {1, t, r}
        // peCache : {1, t, c}
        // output : {b * s, h, c}

        Data score0, score1;
        if (b == 1 && s == 1) {
            FastllmCudaMLA(qNope, qPe, kvCache, peCache, score0, output, softmaxScale);
        } else {
            if ((double)b * s * h * t * 2 * 4 > 1e9) {
                int parth = 1;
                Data qNopePart, qPePart;
                std::vector <Data> outputParts;
                std::vector <Data*> outputPartPointers;
                outputParts.resize((h - 1) / parth + 1);
                for (int i = 0; i < outputParts.size(); i++) {
                    outputPartPointers.push_back(&outputParts[i]);
                }
                for (int sth = 0; sth < h; sth += parth) {
                    int idx = sth / parth;
                    int curh = std::min(parth, h - sth);
                    Split(qNope, 1, sth, sth + curh, qNopePart);
                    Split(qPe, 2, sth, sth + curh, qPePart);
                    qNopePart.Reshape({b, s * curh, c});
                    MatMulTransB(qNopePart, peCache, score0);
                    score0.Reshape({b, s, curh, t});
                    qPePart.Reshape({b, s * curh, r});
                    MatMulTransB(qPePart, kvCache, score1);
                    score1.Reshape({b, s, curh, t});
                    AddTo(score1, score0);
                    Mul(score1, softmaxScale, score0);
                    if (mask.dims.size() > 0) {
                        score0.Reshape({b * s, curh, t});
                        ToDataType(mask, qNope.dataType);
                        AttentionMask(score0, mask, -10000);
                    }

                    Softmax(score0, score0, -1);
                    score0.Reshape({b, s * curh, t});
                    MatMul(score0, peCache, outputParts[idx]);
                    outputParts[idx].Reshape({b, s, curh, c});
                }
                CatBatch(outputPartPointers, 2, output);
                output.Reshape({b * s, h, c});
            } else {
                qNope.Reshape({b, s * h, c});
                MatMulTransB(qNope, peCache, score0);
                score0.Reshape({b, s, h, t});

                qPe.Reshape({qPe.dims[0], -1, qPe.dims[3]});
                MatMulTransB(qPe, kvCache, score1);
                score1.Reshape({b, s, h, t});
                AddTo(score1, score0);
                Mul(score1, softmaxScale, score0);

                if (mask.dims.size() > 0) {
                    score0.Reshape({b * s, h, t});
                    ToDataType(mask, qNope.dataType);
                    AttentionMask(score0, mask, -10000);
                }

                Softmax(score0, score0, -1);
                score0.Reshape({b, s * h, t});
                MatMul(score0, peCache, output);
            }
        }
    }

    void CudaMergeMLAPaged::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &output = *(datas.find("output")->second);
        output.dataType = qNope.dataType;
        output.Resize(qNope.dims);
    }

    void CudaMergeMLAPaged::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &qPe = *(datas.find("qPe")->second);
        Data &kvCachePaged = *(datas.find("kvCachePaged")->second);
        Data &peCachePaged = *(datas.find("peCachePaged")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float softmaxScale = floatParams.find("softmaxScale") != floatParams.end() ? floatParams.find("softmaxScale")->second : 1.0f;

        AssertInFastLLM(kvCachePaged.isPagedKVCache && peCachePaged.isPagedKVCache,
            "CudaMergeMLAPaged: kvCachePaged and peCachePaged must be paged KV cache (isPagedKVCache=true).\n");
        if (qNope.dataType != DataType::FLOAT16) {
            printf("CudaMergeMLAPaged: qNope must be FLOAT16 to use FastllmCudaMLAPaged.\n");
            return;
        }

        bool ok = FastllmCudaMLAPaged(qNope, qPe, kvCachePaged, peCachePaged, output, softmaxScale);
        AssertInFastLLM(ok, "CudaMergeMLAPaged: FastllmCudaMLAPaged failed.\n");
    }

    void DoCudaMergeMOEFromCPU (Data &input, Data &output, Data &index, Data &score, Data &w1, Data &w2, Data &w3, 
                        Data **weights, Data **biass, float sharedScale, bool setZero, int expertLimit) {
// static std::map <std::string, float> timeCnt;
// auto st = std::chrono::system_clock::now();
        if (setZero) {
            output.Allocate();
            output.ToCudaTemporary({}, false);
            FastllmCudaMemset0(output.cudaData, output.GetBytes());
        } else {
            output.ToCudaTemporary({}, true);
        }
        input.ToCudaTemporary({}, true);
// ForceDeviceSync(); timeCnt["io"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        int batch = input.dims[0];
        
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;
        int n = index.dims[0];
        int topk = index.dims[1];
        
        // 计算最大专家数量
        int maxExpert = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < topk; j++) {
                int expertIdx = indexData[i * topk + j];
                if (expertIdx > maxExpert) {
                    maxExpert = expertIdx;
                }
            }
        }
        int m = maxExpert + 1; // 专家数量
        
        std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
        expertTasks.resize(m + 1);
        for (int b = 0; b < batch; b++) {
            expertTasks[0].push_back(std::make_pair(b, sharedScale));
            for (int j = 0; j < topk; j++) {
                int expertIdx = indexData[b * topk + j];
                float value = scoreData[b * topk + j];
                expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
            }
        }

        std::vector <int> indexVec;
        std::vector <float> scales;
        std::vector <int> startIdx;                
        for (int i = 0; i < expertTasks.size(); i++) {
            startIdx.push_back(indexVec.size());
            for (int j = 0; j < expertTasks[i].size(); j++) {
                indexVec.push_back(expertTasks[i][j].first);
                scales.push_back(expertTasks[i][j].second);
            }
        }

// ForceDeviceSync(); timeCnt["get experts"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        int *cudaIndex = (int*)FastllmCudaMalloc(indexVec.size() * sizeof(int));
        float *cudaScales = (float*)FastllmCudaMalloc(scales.size() * sizeof(float));
// ForceDeviceSync(); timeCnt["malloc index"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        FastllmCudaCopyFromHostToDevice(cudaIndex, indexVec.data(), indexVec.size() * sizeof(int));
        FastllmCudaCopyFromHostToDevice(cudaScales, scales.data(), scales.size() * sizeof(float));
// ForceDeviceSync(); timeCnt["copy index"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        static Data tempInput, tempMiddle, tempSwiglu, tempOutput;
        tempInput.Resize(input.dims);
        tempInput.dataType = input.dataType;
        tempInput.ToDevice(input.dataDevice);
        tempInput.Allocate();

        tempMiddle.Resize({input.dims[0], weights[2]->dims[0]});
        tempMiddle.dataType = input.dataType;
        tempMiddle.ToDevice(input.dataDevice);
        tempMiddle.Allocate();

        tempSwiglu.Resize({input.dims[0], weights[2]->dims[0] / 2});
        tempSwiglu.dataType = input.dataType;
        tempSwiglu.ToDevice(input.dataDevice);
        tempSwiglu.Allocate();

        tempOutput.Resize(output.dims);
        tempOutput.dataType = input.dataType;
        tempOutput.ToDevice(output.dataDevice);
        tempOutput.Allocate();
// ForceDeviceSync(); timeCnt["alloc data"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
static float total = 0.0f;
std::map <int, int> eeCnt;
for (int e = 0; e < expertTasks.size(); e++) {
    if (weights[e * 2] != nullptr) {
        eeCnt[expertTasks[e].size()]++;
    }
}
for (auto &it : eeCnt) {
    // printf("%d: %d\n", it.first, it.second);
}
        for (int i = 0; i < expertTasks.size(); i++) {
            if (expertTasks[i].size() == 0 || expertTasks[i].size() < expertLimit || weights[i * 2] == nullptr) {
                continue;
            }
// ForceDeviceSync(); timeCnt["expert start"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            weights[i * 2]->ToCudaTemporary({}, true);
            weights[i * 2 + 1]->ToCudaTemporary({}, true);
total += weights[i * 2]->GetBytes();
total += weights[i * 2 + 1]->GetBytes();
// ForceDeviceSync(); timeCnt["copy weight"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            tempInput.Resize({(int)expertTasks[i].size(), tempInput.dims[1]});
            FastllmCudaPickInput (
                (uint8_t*)input.cudaData, 
                (uint8_t*)tempInput.cudaData, 
                expertTasks[i].size(), 
                GetDataBytes(input.dataType, 1, input.dims[1]), 
                cudaIndex + startIdx[i]
            );
// ForceDeviceSync(); timeCnt["pick input"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            DoCudaLinearReshape(tempInput, *weights[i * 2], tempMiddle);
            DoCudaLinear(tempInput, *weights[i * 2], *GetEmptyData(), tempMiddle);
// ForceDeviceSync(); timeCnt["linear"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            DoCudaSwigluReshape(tempMiddle, tempSwiglu);
            DoCudaSwiglu(tempMiddle, tempSwiglu);
// ForceDeviceSync(); timeCnt["swiglu"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            DoCudaLinearReshape(tempSwiglu, *weights[i * 2 + 1], tempOutput);
            DoCudaLinear(tempSwiglu, *weights[i * 2 + 1], *GetEmptyData(), tempOutput);
// ForceDeviceSync(); timeCnt["linear"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            FastllmCudaPickOutput (
                (uint8_t*)tempOutput.cudaData, 
                (uint8_t*)output.cudaData, 
                expertTasks[i].size(), 
                output.dims[1], 
                cudaIndex + startIdx[i],
                cudaScales + startIdx[i], 
                tempOutput.dataType
            );
// ForceDeviceSync(); timeCnt["pick output"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            weights[i * 2]->FreeCudaTemporary({}, false);
            weights[i * 2 + 1]->FreeCudaTemporary({}, false);
// ForceDeviceSync(); timeCnt["FreeCudaTemporary"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        }

        FastllmCudaFree(cudaIndex);
        FastllmCudaFree(cudaScales);
// printf("copy weight %f G.\n", total / 1e9);

        input.FreeCudaTemporary({}, false);
        output.FreeCudaTemporary({}, true);
// ForceDeviceSync(); timeCnt["last free"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
float totalTime = 0.0f;
// for (auto &it : timeCnt) {
    // printf("%s: %f s.\n", it.first.c_str(), it.second);
    // totalTime += it.second;
// }
// printf("total time = %f\n", totalTime);
    }

    void DoCudaMergeMOE(Data &input, Data &output, Data &index, Data &score, Data &w1, Data &w2, Data &w3, 
                        Data **weights, Data **biass, float sharedScale) {
// static std::map<std::string, float> mergeMoeTimeCnt;
// auto st = std::chrono::system_clock::now();
        output.Allocate();
// ForceDeviceSync(); mergeMoeTimeCnt["allocate"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
        {
            int batch = input.dims[0];
            
            // 确保 index 和 score 在 CPU 上
            index.ToDevice(DataDevice::CPU);
            score.ToDevice(DataDevice::CPU);
            ToDataType(index, DataType::INT32);
            ToDataType(score, DataType::FLOAT32);
            
            int32_t *indexData = (int32_t*)index.cpuData;
            float *scoreData = (float*)score.cpuData;
            int n = index.dims[0];
            int topk = index.dims[1];
            
            if (batch == 1) {
                std::vector <std::pair <int, float> > v;
                v.resize(topk + 1);
                for (int j = 0; j < topk; j++) {
                    // index 存储的是专家索引（从0开始），需要+1因为0表示shared expert
                    int expertIdx = indexData[j];
                    float expertScore = scoreData[j];
                    v[j] = std::make_pair(expertIdx + 1, expertScore);
                }
                v.back() = (std::make_pair(0, sharedScale));
// ForceDeviceSync(); mergeMoeTimeCnt["get_experts"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    float value = v[j].second;
                    if (weights[idx * 2] == nullptr) {
                        continue;
                    }

                    DoCudaLinearReshape(input, *weights[idx * 2], w3);
                    DoCudaLinear(input, *weights[idx * 2], *GetEmptyData(), w3);
// ForceDeviceSync(); mergeMoeTimeCnt["linear1"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    DoCudaSwigluReshape(w3, w1);
                    DoCudaSwiglu(w3, w1);
// ForceDeviceSync(); mergeMoeTimeCnt["swiglu"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    DoCudaLinearReshape(w1, *weights[idx * 2 + 1], w2);
                    DoCudaLinear(w1, *weights[idx * 2 + 1], *GetEmptyData(), w2);
// ForceDeviceSync(); mergeMoeTimeCnt["linear2"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    if (j == 0) {
                        output.dataType = w2.dataType;
                        output.Resize(w2.dims);
                        FastllmCudaMul(w2, value, output);
                    } else {
                        FastllmCudaAddTo(output, w2, value);
                    }
// ForceDeviceSync(); mergeMoeTimeCnt["mul_add"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                }
            } else {
                FastllmCudaMemset0(output.cudaData, output.GetBytes());
                
                // 计算最大专家数量
                int maxExpert = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = indexData[i * topk + j];
                        if (expertIdx > maxExpert) {
                            maxExpert = expertIdx;
                        }
                    }
                }
                int m = maxExpert + 1; // 专家数量
                
                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                for (int b = 0; b < batch; b++) {
                    expertTasks[0].push_back(std::make_pair(b, sharedScale));
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = indexData[b * topk + j];
                        float value = scoreData[b * topk + j];
                        expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
                    }
                }

                std::vector <int> indexVec2;
                std::vector <float> scales;
                std::vector <int> startIdx;                
                for (int i = 0; i < expertTasks.size(); i++) {
                    startIdx.push_back(indexVec2.size());
                    for (int j = 0; j < expertTasks[i].size(); j++) {
                        indexVec2.push_back(expertTasks[i][j].first);
                        scales.push_back(expertTasks[i][j].second);
                    }
                }
// ForceDeviceSync(); mergeMoeTimeCnt["get_experts"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                int *cudaIndex = (int*)FastllmCudaMalloc(indexVec2.size() * sizeof(int));
// ForceDeviceSync(); mergeMoeTimeCnt["malloc_index"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                FastllmCudaCopyFromHostToDevice(cudaIndex, indexVec2.data(), indexVec2.size() * sizeof(int));

                float *cudaScales = (float*)FastllmCudaMalloc(scales.size() * sizeof(float));
                FastllmCudaCopyFromHostToDevice(cudaScales, scales.data(), scales.size() * sizeof(float));
// ForceDeviceSync(); mergeMoeTimeCnt["copy_index"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                Data tempInput, tempMiddle, tempSwiglu, tempOutput;
                tempInput.Resize(input.dims);
                tempInput.ToDevice(input.dataDevice);
                tempInput.Allocate();

                tempMiddle.Resize({input.dims[0], weights[2]->dims[0]});
                tempMiddle.ToDevice(input.dataDevice);
                tempMiddle.Allocate();

                tempSwiglu.Resize({input.dims[0], weights[2]->dims[0] / 2});
                tempSwiglu.ToDevice(input.dataDevice);
                tempSwiglu.Allocate();

                tempOutput.Resize(output.dims);
                tempOutput.ToDevice(output.dataDevice);
                tempOutput.Allocate();
// ForceDeviceSync(); mergeMoeTimeCnt["alloc_data"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                for (int i = 0; i < expertTasks.size(); i++) {
                    if (expertTasks[i].size() == 0 || weights[i * 2] == nullptr) {
                        continue;
                    }
// ForceDeviceSync(); mergeMoeTimeCnt["expert_start"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    tempInput.Resize({(int)expertTasks[i].size(), tempInput.dims[1]});
                    FastllmCudaPickInput (
                        (uint8_t*)input.cudaData, 
                        (uint8_t*)tempInput.cudaData, 
                        expertTasks[i].size(), 
                        GetDataBytes(input.dataType, 1, input.dims[1]), 
                        cudaIndex + startIdx[i]
                    );
// ForceDeviceSync(); mergeMoeTimeCnt["pick_input"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    DoCudaLinearReshape(tempInput, *weights[i * 2], tempMiddle);
                    DoCudaLinear(tempInput, *weights[i * 2], *GetEmptyData(), tempMiddle);
// ForceDeviceSync(); mergeMoeTimeCnt["linear1"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    DoCudaSwigluReshape(tempMiddle, tempSwiglu);
                    DoCudaSwiglu(tempMiddle, tempSwiglu);
// ForceDeviceSync(); mergeMoeTimeCnt["swiglu"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    DoCudaLinearReshape(tempSwiglu, *weights[i * 2 + 1], tempOutput);
                    DoCudaLinear(tempSwiglu, *weights[i * 2 + 1], *GetEmptyData(), tempOutput);
// ForceDeviceSync(); mergeMoeTimeCnt["linear2"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();

                    FastllmCudaPickOutput (
                        (uint8_t*)tempOutput.cudaData, 
                        (uint8_t*)output.cudaData, 
                        expertTasks[i].size(), 
                        output.dims[1], 
                        cudaIndex + startIdx[i],
                        cudaScales + startIdx[i], 
                        output.dataType
                    );
// ForceDeviceSync(); mergeMoeTimeCnt["pick_output"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                }

                FastllmCudaFree(cudaIndex);
                FastllmCudaFree(cudaScales);
// ForceDeviceSync(); mergeMoeTimeCnt["free"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            }
        }
// float totalTime = 0.0f;
// for (auto &it : mergeMoeTimeCnt) {
    // printf("[DoCudaMergeMOE] %s: %f s.\n", it.first.c_str(), it.second);
    // totalTime += it.second;
// }
// printf("[DoCudaMergeMOE] total: %f s.\n", totalTime);
    }

    void CudaMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;        

        DoCudaMergeMOE (
            input, output, index, score, w1, w2, w3, weights, biass, sharedScale
        );
    }

    void CudaMergeAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaMergeAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sinData")->second);
        Data &cosData = *(datas.find("cosData")->second);
        Data &output = *(datas.find("output")->second);
        Data &qkv = *(datas.find("qkv")->second);
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        int qNum = intParams.find("qNum")->second;
        int kvNum = intParams.find("kvNum")->second;
        int headDim = intParams.find("headDim")->second;
        int rotDim = intParams.find("rotDim")->second;
        float attentionScale = floatParams.find("attentionScale")->second;
        Data **keys = (Data**)(datas.find("keys")->second);
        Data **values = (Data**)(datas.find("values")->second);
        Data **masks = (Data**)(datas.find("masks")->second);
        output.Allocate();

        int bsz = input.dims[0], seqlen = input.dims[1];
        DoCudaLinearReshape(input, weight0, qkv);
        DoCudaLinear(input, weight0, bias0, qkv);

        int per = qkv.dims.back() / (qNum / kvNum + 2);
        int qdim = per * (qNum / kvNum);
        
        DoCudaSplitReshape(qkv, -1, 0, qdim, q);
        DoCudaSplitReshape(qkv, -1, qdim, qdim + per, k);
        DoCudaSplitReshape(qkv, -1, qdim + per, qdim + per * 2, v);
        DoCudaSplit(qkv, -1, 0, qdim, q);
        DoCudaSplit(qkv, -1, qdim, qdim + per, k);
        DoCudaSplit(qkv, -1, qdim + per, qdim + per * 2, v);

        std::vector <int> qkvSize = {bsz, seqlen, -1, headDim};
        q.Reshape(qkvSize);
        k.Reshape(qkvSize);
        v.Reshape(qkvSize);

        FastllmCudaLlamaRotatePosition2D(q, positionIds, sinData, cosData, rotDim);
        FastllmCudaLlamaRotatePosition2D(k, positionIds, sinData, cosData, rotDim);

        DoCudaPermuteSelf(q, {0, 2, 1, 3});
        DoCudaPermuteSelf(k, {0, 2, 1, 3});
        DoCudaPermuteSelf(v, {0, 2, 1, 3});

        qkvSize = {-1, seqlen, headDim};
        q.Reshape(qkvSize);
        k.Reshape(qkvSize);
        v.Reshape(qkvSize);

        int unitLen = 128;
        Data &pastKey = *keys[0];
        Data &pastValue = *values[0];
        while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || seqlen > pastKey.expansionDims[1]))
            || (pastKey.dims.size() > 0 && pastKey.dims[1] + seqlen > pastKey.expansionDims[1])) {
            std::vector <int> newDims;
            if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
            } else {
                newDims = pastKey.dims;
                newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
            }
            pastKey.Expansion(newDims);
        }
        while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || seqlen > pastValue.expansionDims[1]))
            || (pastValue.dims.size() > 0 && pastValue.dims[1] + seqlen > pastValue.expansionDims[1])) {
            std::vector <int> newDims;
            if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                newDims = std::vector <int> {kvNum, ((seqlen - 1) / unitLen + 1) * unitLen, headDim};
            } else {
                newDims = pastValue.dims;
                newDims[1] += ((seqlen - 1) / unitLen + 1) * unitLen;
            }
            pastValue.Expansion(newDims);
        }

        DoCudaCatDirect(pastKey, k, 1);
        DoCudaCatDirect(pastValue, v, 1);

        DoCudaAttentionReshape(q, pastValue, qkv);
        DoCudaAttention(q, pastKey, pastValue, *masks[0], qkv, q.dims[0] / pastKey.dims[0], attentionScale, 0);

        DoCudaPermuteSelf(qkv, {1, 0, 2});
        qkv.Reshape({seqlen, bsz, -1});
        DoCudaPermuteSelf(qkv, {1, 0, 2});

        DoCudaLinearReshape(qkv, weight1, output);
        DoCudaLinear(qkv, weight1, bias1, output);
    }

    void CudaAppendPagedCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &cache = *(datas.find("cache")->second);
        Data &input = *(datas.find("input")->second);

        // CUDA实现分页缓存追加
        int numHeads = input.dims[0];
        int seqLen = input.dims[1];
        int headDim = input.dims[2];
        int unitSize = input.unitSize;
        int tokensToAppend = seqLen;
        int inputOffset = 0;

        // pagedKVCacheData shape: [maxPages, pageLen, numHeads, headDim]
        Data *pagedKVCache = cache.pagedKVCacheData;
        int maxPages = pagedKVCache->dims[0];
        int pageLen = cache.pageLen;
        uint8_t *pagedData = (uint8_t*)pagedKVCache->cudaData;
        uint8_t *inputData = (uint8_t*)input.cudaData;
        
        // 计算当前page的剩余空间
        int remainingInCurrentPage = 0;
        if (cache.pageIndex.size() > 0) {
            remainingInCurrentPage = pageLen - cache.lastPageLen;
        }
        
        // 先填充当前page的剩余空间
        if (remainingInCurrentPage > 0 && tokensToAppend > 0) {
            int currentPageIdx = cache.pageIndex.back();
            int copyLen = std::min(remainingInCurrentPage, tokensToAppend);
            
            // kernel复制 input 到 pagedKVCacheData 的当前 page 的剩余空间
            // input: [numHeads, seqLen, headDim], pagedKVCacheData: [maxPages, pageLen, numHeads, headDim]
            FastllmCudaPagedCacheCopy(
                pagedData, // dst: pagedKVCache->cudaData
                currentPageIdx, // page idx
                pageLen,
                numHeads,
                headDim,
                unitSize,
                inputData, // src: input.cudaData
                seqLen,    // input sequence length
                inputOffset,
                copyLen,
                cache.lastPageLen // page offset: 从当前page的lastPageLen位置开始写入
            );
            
            cache.lastPageLen += copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }
        
        // 如果还有剩余的数据，需要分配新的 pages
        while (tokensToAppend > 0) {
            int newPageIdx = cache.pagedKVCacheData->GetUnusedPageIndex(true);
            cache.pageIndex.push_back(newPageIdx);
            int copyLen = std::min(pageLen, tokensToAppend);

            // kernel复制 input 的一个 page 到 pagedKVCacheData 的新 page
            // input: [numHeads, seqLen, headDim], pagedKVCacheData: [maxPages, pageLen, numHeads, headDim]
            FastllmCudaPagedCacheCopy(
                pagedData, // dst: pagedKVCache->cudaData
                newPageIdx, // page idx
                pageLen,
                numHeads,
                headDim,
                unitSize,
                inputData, // src: input.cudaData
                seqLen,    // input sequence length
                inputOffset,
                copyLen,
                0 // page offset: 新page从0开始写入
            );

            cache.lastPageLen = copyLen;
            tokensToAppend -= copyLen;
            inputOffset += copyLen;
        }
    }

    void DoCudaAttentionPaged(Data &q, Data &k, Data &v, Data &output, int group, float scale) {
        FastllmCudaHalfPagedAttention(q, k, v, output, group, scale);
    }

    void CudaAttentionPagedOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        output.Allocate();
        DoCudaAttentionPaged(q, k, v, output, group, scale);
    }

    void CudaAppendPagedCacheBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second); // batch, num_heads, head_dim
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        PagedCacheManager &manager = *(PagedCacheManager*)datas.find("pagedCacheManager")->second;

        int batch = input.dims[0], numHeads = input.dims[1], headDim = input.dims[2];
        int pageLen = ((Data*)&manager)->dims[1];
        int unitSize = input.unitSize;
        uint8_t *pagedData = (uint8_t*)((Data*)&manager)->cudaData;
        uint8_t *inputData = (uint8_t*)input.cudaData;
        int32_t *idxDataCuda = (int32_t*)insertIndexs.cudaData;
        int32_t *posDataCuda = (int32_t*)insertPositions.cudaData;
        // 使用batch函数一次性处理所有batch
        FastllmCudaPagedCacheCopyBatch(
            pagedData,
            idxDataCuda,
            posDataCuda,
            pageLen,
            batch,
            numHeads,
            headDim,
            unitSize,
            inputData
        );
    }

    void CudaGenerateAppendPagedCacheBatchParamsOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        PagedCacheManager &manager = *(PagedCacheManager*)datas.find("pagedCacheManager")->second;
        Data **pastKeys = (Data**)(datas.find("pastKeys")->second);
        Data &insertIndexs = *(datas.find("insertIndexs")->second);
        Data &insertPositions = *(datas.find("insertPositions")->second);
        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        if (batch <= 0 && intParams.find("batch") != intParams.end()) {
            batch = intParams.find("batch")->second;
        }
        AssertInFastLLM(batch > 0, "CudaGenerateAppendPagedCacheBatchParamsOp: batch must be positive.\n");

        insertIndexs.Allocate();
        insertPositions.Allocate();
        
        // 先在CPU上准备数据
        auto &idxDataHost = insertIndexs.cpuIntDatas;
        auto &posDataHost = insertPositions.cpuIntDatas;
        idxDataHost.resize(batch);
        posDataHost.resize(batch);

        for (int b = 0; b < batch; b++) {
            Data *pk = pastKeys[b];
            int pageLen = pk->pageLen;
            int insertIdx, insertPos;
            if (pk->pageIndex.empty()) {
                insertIdx = manager.GetUnusedPageIndex(false);
                insertPos = 0;
            } else if (pk->lastPageLen < pageLen) {
                insertIdx = pk->pageIndex.back();
                insertPos = pk->lastPageLen;
            } else {
                insertIdx = manager.GetUnusedPageIndex(false);
                insertPos = 0;
            }
            idxDataHost[b] = insertIdx;
            posDataHost[b] = insertPos;
        }
        
        // 拷贝到CUDA设备
        FastllmCudaCopyFromHostToDevice(insertIndexs.cudaData, idxDataHost.data(), batch * sizeof(int32_t));
        FastllmCudaCopyFromHostToDevice(insertPositions.cudaData, posDataHost.data(), batch * sizeof(int32_t));
    }

    void DoCudaAttentionPagedBatch(Data &q, Data &kCaches, Data &vCaches, Data &qSizes, Data &pageSizes, Data &pageIndexs, Data &lastPageLens, Data &output, int group, float scale, int attentionType) {
        FastllmCudaHalfPagedAttentionBatch(q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens, output, group, scale, attentionType);
    }

    void CudaAttentionPagedBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &kCaches = *(datas.find("kCaches")->second);
        Data &vCaches = *(datas.find("vCaches")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / kCaches.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        int attentionType = intParams.find("attentionType") != intParams.end() ? intParams.find("attentionType")->second : 0;
        output.Allocate();
        DoCudaAttentionPagedBatch(q, kCaches, vCaches, qSizes, pageSizes, pageIndexs, lastPageLens, output, group, scale, attentionType);  
    }

    void CudaGeneratePagedBatchParamsOp::Run(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data **pastKeys = (Data**)(datas.find("pastKeys")->second);
        Data &qSizes = *(datas.find("qSizes")->second);
        Data &pageSizes = *(datas.find("pageSizes")->second);
        Data &pageIndexs = *(datas.find("pageIndexs")->second);
        Data &lastPageLens = *(datas.find("lastPageLens")->second);
        
        int batch = intParams.find("pastKeys___batch") != intParams.end() ? intParams.find("pastKeys___batch")->second : 1;
        
        // 分配输出内存
        qSizes.Allocate();
        pageSizes.Allocate();
        lastPageLens.Allocate();
        pageIndexs.Allocate();
        
        // 先在CPU上计算，然后拷贝到GPU
        auto &qSizesHost = qSizes.cpuIntDatas;
        qSizesHost.resize(batch + 1);

        auto &pageSizesHost = pageSizes.cpuIntDatas;
        pageSizesHost.resize(batch + 1);

        auto &lastPageLensHost = lastPageLens.cpuIntDatas;
        lastPageLensHost.resize(batch);

        auto &pageIndexsHost = pageIndexs.cpuIntDatas;
        
        // 计算总的page数量
        int totalPages = 0;
        for (int b = 0; b < batch; b++) {
            totalPages += pastKeys[b]->pageIndex.size();
        }
        pageIndexsHost.resize(totalPages);
        
        // 生成qSizes: [0, 1, 2, ... batch]
        qSizesHost[0] = 0;
        for (int b = 0; b < batch; b++) {
            qSizesHost[b + 1] = b + 1;
        }
        
        // 生成pageSizes, pageIndexs, lastPageLens
        int pageOffset = 0;
        pageSizesHost[0] = 0;
        for (int b = 0; b < batch; b++) {
            int numPages = pastKeys[b]->pageIndex.size();
            pageSizesHost[b + 1] = pageSizesHost[b] + numPages;
            
            // 复制pageIndex
            for (int i = 0; i < numPages; i++) {
                pageIndexsHost[pageOffset + i] = pastKeys[b]->pageIndex[i];
            }
            pageOffset += numPages;
            
            // 设置lastPageLen
            lastPageLensHost[b] = pastKeys[b]->lastPageLen;
        }
        
        // 拷贝到对应设备
        FastllmCudaCopyFromHostToDevice(qSizes.cudaData, qSizesHost.data(), (batch + 1) * sizeof(int32_t));
        FastllmCudaCopyFromHostToDevice(pageSizes.cudaData, pageSizesHost.data(), (batch + 1) * sizeof(int32_t));
        FastllmCudaCopyFromHostToDevice(lastPageLens.cudaData, lastPageLensHost.data(), batch * sizeof(int32_t));
        FastllmCudaCopyFromHostToDevice(pageIndexs.cudaData, pageIndexsHost.data(), totalPages * sizeof(int32_t));
    }
}
