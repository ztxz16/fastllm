//
// Created by huangyuyang on 7/19/23.
//

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
#include "device.h"

namespace fastllm {
    void CpuSplitBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = input.dims[axis];
        std::vector <int> dims = input.dims;
        dims[axis] = 1;
        for (int i = 0; i < part; i++) {
            outputs[i]->dataType = input.dataType;
            outputs[i]->Resize(dims);
        }
    }

    void CpuSplitBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = input.dims[axis];

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = outputs[0]->Count(axis);
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        for (int i = 0; i < part; i++) {
            int start = i, end = i + 1;
            outputs[i]->Allocate();
            for (int o = 0; o < outer; o++) {
                memcpy(outputs[i]->cpuData + o * outputStride * unitSize,
                       input.cpuData + (o * inputStride + start * inner) * unitSize,
                       (end - start) * inner * unitSize);
            }
        }
    }

    void CpuCatBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = inputs[0]->dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("input___batch")->second;
        std::vector <int> dims = inputs[0]->dims;
        dims[axis] = part;
        output.dataType = inputs[0]->dataType;
        output.Resize(dims);
    }

    void CpuCatBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = inputs[0]->dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = intParams.find("input___batch")->second;

        int outer = inputs[0]->Count(0) / inputs[0]->Count(axis);
        int inputStride = inputs[0]->Count(axis);
        int outputStride = output.Count(axis);
        int inner = inputs[0]->strides[axis];
        int unitSize = inputs[0]->unitSize;

        output.Allocate();
        for (int i = 0; i < part; i++) {
            int start = i, end = i + 1;
            for (int o = 0; o < outer; o++) {
                memcpy(output.cpuData + (o * outputStride + start * inner) * unitSize,
                       inputs[i]->cpuData + (o * inputStride) * unitSize,
                       (end - start) * inner * unitSize);
            }
        }
    }

    void CpuMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        int batch = intParams.find("input___batch")->second;
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
            AssertInFastLLM(inputs[i]->dataType == DataType::FLOAT32, "Mul error: Data's type should be float32.\n");

            float *inputData = (float *) inputs[i]->cpuData;
            float *outputData = (float *) outputs[i]->cpuData;
            int len = inputs[i]->Count(0);
            for (int i = 0; i < len; i++) {
                outputData[i] = inputData[i] * v;
            }
        }
    }

    void CpuMatMulTransBBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuMatMulTransBOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Reshape("MatMulTransB", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuMatMulTransBBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuMatMulTransBOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Run("MatMulTransB", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuMatMulBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuMatMulOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Reshape("MatMulTransB", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuMatMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuMatMulOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Run("MatMulTransB", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuSoftmaxBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuSoftMaxOp());
        int batch = intParams.find("input___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input"] = ((Data**)datas.find("input")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Run("Softmax", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuCatDirectBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuCatDirectOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            op->Run("CatDirect", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void CpuAttentionBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **qs = (Data**)(datas.find("q")->second);
        Data **ks = (Data**)(datas.find("k")->second);
        Data **vs = (Data**)(datas.find("v")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int batch = intParams.find("q___batch")->second;

        Data &q = *qs[0], &k = *ks[0], &v = *vs[0];
        AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
        AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
        AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
        AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
        AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");
        AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType,
                        "Attention: q, k, v's datatype should be same.\n");
        AssertInFastLLM(q.dataType == DataType::FLOAT32, "Attention's input's type should be float32.\n");

        for (int i = 0; i < batch; i++) {
            outputs[i]->dataType = qs[i]->dataType;
            outputs[i]->Resize({qs[i]->dims[0], qs[i]->dims[1], vs[i]->dims[2]});
        }
    }

    void CpuAttentionBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuAttention());
        int batch = intParams.find("q___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["q"] = ((Data**)datas.find("q")->second)[i];
            tempDatas["k"] = ((Data**)datas.find("k")->second)[i];
            tempDatas["v"] = ((Data**)datas.find("v")->second)[i];
            tempDatas["mask"] = ((Data**)datas.find("mask")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Run("Attention", tempDatas, floatParams, intParams);
        }
        delete op;
    }
}