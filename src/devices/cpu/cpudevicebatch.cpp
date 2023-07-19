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

namespace fastllm {
    void CpuSplitBatchOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data *outputs = (datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = input.dims[axis];
        std::vector <int> dims = input.dims;
        dims[axis] = 1;
        for (int i = 0; i < part; i++) {
            outputs[i].dataType = input.dataType;
            outputs[i].Resize(dims);
        }
    }

    void CpuSplitBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data *outputs = (datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int part = input.dims[axis];

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = outputs[0].Count(axis);
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        for (int i = 0; i < part; i++) {
            int start = i, end = i + 1;
            outputs[i].Allocate();
            for (int o = 0; o < outer; o++) {
                memcpy(outputs[i].cpuData + o * outputStride * unitSize,
                       input.cpuData + (o * inputStride + start * inner) * unitSize,
                       (end - start) * inner * unitSize);
            }
        }
    }

    void CpuMulBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *inputs = (datas.find("input")->second);
        Data *outputs = (datas.find("output")->second);

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        int batch = intParams.find("input___batch")->second;
        for (int i = 0; i < batch; i++) {
            outputs[i].Allocate();
            AssertInFastLLM(inputs[i].dataType == DataType::FLOAT32, "Mul error: Data's type should be float32.\n");

            float *inputData = (float *) inputs[i].cpuData;
            float *outputData = (float *) outputs[i].cpuData;
            int len = inputs[i].Count(0);
            for (int i = 0; i < len; i++) {
                outputData[i] = inputData[i] * v;
            }
        }
    }
}