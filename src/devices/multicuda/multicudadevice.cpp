//
// Created by huangyuyang on 8/2/24.
//

#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/multicudadevice.h"

#include "fastllm-multicuda.cuh"

#include "utils.h"

namespace fastllm {
    MultiCudaDevice::MultiCudaDevice(CudaDevice *cudaDevice) {
        this->cudaDevice = cudaDevice;
        this->deviceType = "multicuda";

        this->ops["MLP"] = (BaseOperator*)(new MultiCudaMLPOp());
        this->ops["Linear"] = (BaseOperator*)(new MultiCudaLinearOp());
    }

    bool MultiCudaDevice::Malloc(void **ret, size_t size) {
        *ret = FastllmCudaMalloc(size);
        return true;
    }

    bool MultiCudaDevice::Free(void *ret) {
        FastllmCudaFree(ret);
        return true;
    }

    bool MultiCudaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        FastllmCudaCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    bool MultiCudaDevice::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            if (((BaseDevice*)this->cudaDevice)->ops.find(opType) == ((BaseDevice*)this->cudaDevice)->ops.end()) {
                return false;
            } else {
                return ((BaseDevice*)this->cudaDevice)->CanRun(opType, datas, floatParams, intParams);
            }
        } else {
            return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行形状推理
    void MultiCudaDevice::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Reshape(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
        }
    }

    // 对某一个算子进行推理
    void MultiCudaDevice::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            ((BaseDevice*)this->cudaDevice)->Run(opType, datas, floatParams, intParams);
        } else {
            this->ops[opType]->Run(opType, datas, floatParams, intParams);
        }
    }

    void MultiCudaMLPOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &weight1 = *(datas.find("weight1")->second);

        AssertInFastLLM(weight0.dims.size() == 2 && weight1.dims.size() == 2, "MLP's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight0.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dims[0] / 2 == weight1.dims[1], "MLP's weight's shape error.\n");
        AssertInFastLLM(weight0.dataType == weight1.dataType, "MLP's weight's data type error.\n");

        weight0.weightType = WeightType::LINEAR;
        weight1.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight1.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void MultiCudaMLPOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight0 = *(datas.find("weight0")->second);
        Data &bias0 = *(datas.find("bias0")->second);
        Data &weight1 = *(datas.find("weight1")->second);
        Data &bias1 = *(datas.find("bias1")->second);

        output.Allocate();
        FastllmMultiCudaMLP(input, weight0, weight1, output);
    }

    bool MultiCudaLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        Data &weight = *(datas.find("weight")->second);
        return weight.dims[0] > 10000 || weight.dims[1] > 10000;
    }

    void MultiCudaLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
// auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        if (input.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16 ||
                weight.dataType == DataType::INT8 ||
                weight.dataType == DataType::INT4_NOZERO ||
                weight.dataType == DataType::INT4_GROUP) {
                FastllmMultiCudaMatMul(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                FastllmCudaMatMulFloat32(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::FLOAT16 ||
                        weight.dataType == DataType::INT8 ||
                        weight.dataType == DataType::INT4_NOZERO ||
                        weight.dataType == DataType::INT4_GROUP) {
                FastllmMultiCudaMatMul(input, weight, bias, output, n, m, k);
            } else if (weight.dataType == DataType::INT4) {
                FastllmCudaMatMulFloatInt4(input, weight, bias, output, n, m, k);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport input's dataType.\n");
        }
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }
}