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
        this->ops["Attention"] = (BaseOperator*)(new CudaAttention());
        this->ops["CopyKVCache"] = (BaseOperator*)(new CudaCopyKVCacheOp());
        this->ops["Embedding"] = (BaseOperator*)(new CudaEmbedding());
        this->ops["LayerNorm"] = (BaseOperator*)(new CudaLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CudaRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CudaLinearOp());
        this->ops["Conv2D"] = (BaseOperator*)(new CudaConv2DOp());
        this->ops["Split"] = (BaseOperator*)(new CudaSplitOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CudaCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CudaMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CudaMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CudaSoftMaxOp());
        this->ops["Gelu"] = (BaseOperator*)(new CudaGeluOp());
        this->ops["GeluNew"] = (BaseOperator*)(new CudaGeluNewOp());
        this->ops["Silu"] = (BaseOperator*)(new CudaSiluOp());
        this->ops["Swiglu"] = (BaseOperator*)(new CudaSwigluOp());
        this->ops["Mul"] = (BaseOperator*)(new CudaMulOp());
        this->ops["AddTo"] = (BaseOperator*)(new CudaAddToOp());
        this->ops["MulTo"] = (BaseOperator*)(new CudaMulToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CudaAttentionMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new CudaAlibiMaskOp());
        this->ops["TopK"] = (BaseOperator*)(new CudaTopKOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CudaPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new CudaRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new CudaNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CudaLlamaRotatePosition2DOp());
        this->ops["RepeatPenalty"] = (BaseOperator*)(new CudaRepeatPenaltyOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new CudaApplyLognAttnOp());

        this->ops["SplitBatch"] = (BaseOperator*)(new CudaSplitBatchOp());
        this->ops["CatBatch"] = (BaseOperator*)(new CudaCatBatchOp());
        this->ops["MulBatch"] = (BaseOperator*)(new CudaMulBatchOp());
        this->ops["MatMulBatch"] = (BaseOperator*)(new CudaMatMulBatchOp());
        this->ops["MatMulTransBBatch"] = (BaseOperator*)(new CudaMatMulTransBBatchOp());
        this->ops["SoftMaxBatch"] = (BaseOperator*)(new CudaSoftmaxBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator*)(new CudaCatDirectBatchOp());
        this->ops["AppendKVCachebatch"] = (BaseOperator*)(new CudaAppendKVCacheBatchOp());
        this->ops["AttentionBatch"] = (BaseOperator*)(new CudaAttentionBatchOp());
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

        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
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
        output.Allocate();

        if (q.dataType == DataType::FLOAT32) {
            FastllmCudaAttention(q, k, v, mask, output, group, scale, maskType);
        } else if (q.dataType == DataType::FLOAT16) {
            FastllmCudaHalfAttention(q, k, v, mask, output, group, scale, maskType);
        }
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

    void CudaLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    bool CudaLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        return true;
    }

    void CudaLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        if (input.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16) {
                FastllmCudaHalfMatMulFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT8) {
                FastllmCudaHalfMatMulFloatInt8(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_GROUP) {
                FastllmCudaHalfMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_NOZERO) {
                FastllmCudaHalfMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                FastllmCudaMatMulFloat32(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FLOAT16) {
                FastllmCudaMatMulFloat16(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT8) {
                FastllmCudaMatMulFloatInt8(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4) {
                FastllmCudaMatMulFloatInt4(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_NOZERO) {
                FastllmCudaMatMulFloatInt4NoZero(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4_GROUP) {
                FastllmCudaMatMulFloatInt4Group(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
        }
    }

    void CudaSplitOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaSplitOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

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

    void CudaCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

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

    void CudaMatMulOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMul error: inputs should use same device.\n");
        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "MatMul's input's type should be float32.\n");
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
        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "MatMulTransB's input's type should be float32.\n");
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

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Softmax error: Data's type should be float32.\n");
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

    void CudaSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CudaSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Swiglu error: Data's type should be float32 or float16.\n");
        FastllmCudaSwiglu(input, output);
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
        AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");
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

        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
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

        FastllmCudaNearlyRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
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
}