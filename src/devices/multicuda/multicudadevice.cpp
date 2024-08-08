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
    MultiCudaDevice::MultiCudaDevice() {
        this->deviceType = "multicuda";

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
                FastllmMultiCudaHalfMatMul(input, weight, bias, output, n, m, k);
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