//
// Created by huangyuyang on 7/19/23.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"

#include "fastllm-cuda.cuh"
#include "utils.h"

namespace fastllm {
    void CudaSplitBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
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

    void CudaSplitBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = input.dims[axis];
        for (int i = 0; i < part; i++) {
            outputs[i]->Allocate();
        }
        FastllmCudaSplitBatch(input, outputs, axis);
    }

    void CudaMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data **inputs = (Data**)(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        int batch = intParams.find("input___batch")->second;
        for (int i = 0; i < batch; i++) {
            outputs[i]->Allocate();
            AssertInFastLLM(inputs[i]->dataType == DataType::FLOAT32, "Mul error: Data's type should be float32.\n");
        }

        FastllmCudaMulBatch(inputs, v, batch, outputs);
    }

    void CudaMatMulBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CudaMatMulOp());
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

    void CudaMatMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CudaMatMulOp());
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

    void CudaMatMulTransBBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CudaMatMulTransBOp());
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

    void CudaMatMulTransBBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CudaMatMulTransBOp());
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
}