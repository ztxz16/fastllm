//
// Created by fastllm-numa on 12/25/24.
//

#include "devices/numa/numadevice.h"
#include "devices/numa/numaopcpu.h"
#include "devices/cpu/cpudevice.h"
#include "devices/numa/numamoe.h"
#include "utils.h"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace fastllm {
    NumaDevice::NumaDevice() {
        this->deviceType = "numa";
        this->ops["Linear"] = (BaseOperator *) (new NumaLinearOp());
        this->ops["MergeMOE"] = (BaseOperator *) (new NumaMergeMOEOp());
        this->ops["Attention"] = (BaseOperator *) (new NumaAttentionOp());
        this->ops["CatDirect"] = (BaseOperator *) (new NumaCatDirectOp());
        
        // 初始化NUMA线程池
        NumaThreadPool::GetInstance().Initialize(GetThreads());
    }

    bool NumaDevice::Malloc(void **ret, size_t size) {
        // 使用CPU内存分配，避免CUDA内存分配
        *ret = (void*)new uint8_t[size];
        return true;
    }

    bool NumaDevice::Free(void *ret) {
        delete[] (uint8_t *)ret;
        return true;
    }

    bool NumaDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        // NUMA设备使用CPU内存，直接返回true
        return true;
    }

    bool NumaDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        // NUMA设备使用CPU内存，直接返回true
        return true;
    }

    // NumaLinearOp实现
    void NumaLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
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

    void NumaLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        // 确保数据在CPU上
        input.ToDevice(DataDevice::CPU);
        weight.ToDevice(DataDevice::CPU);
        output.ToDevice(DataDevice::CPU);
        if (bias.dims.size() > 0) {
            bias.ToDevice(DataDevice::CPU);
        }

        // 计算维度
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = weight.dims[0];

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        float *biasData = bias.dims.size() > 0 ? (float*)bias.cpuData : nullptr;

        // 根据权重数据类型选择合适的NUMA操作
        if (weight.dataType == DataType::FLOAT32) {
            RunNumaLinearFloat32Float32(inputData, (float*)weight.cpuData, outputData, biasData, n, m, k);
        } else if (weight.dataType == DataType::FLOAT16) {
            RunNumaLinearFloat32Float16(inputData, (uint16_t*)weight.cpuData, outputData, biasData, n, m, k);
        } else if (weight.dataType == DataType::BFLOAT16) {
            RunNumaLinearFloat32BFloat16(inputData, (uint16_t*)weight.cpuData, outputData, biasData, n, m, k);
        } else if (weight.dataType == DataType::INT8) {
            RunNumaLinearInt8(inputData, weight, outputData, biasData, n, m, k);
        } else if (weight.dataType == DataType::INT4_GROUP) {
            int group = weight.group;
            int groupCnt = weight.groupCnt;
            RunNumaLinearInt4Group(inputData, weight, outputData, biasData, n, m, k, group, groupCnt);
        } else if (weight.dataType == DataType::INT4_NOZERO) {
            RunNumaLinearInt4NoZero(inputData, weight, outputData, biasData, n, m, k);
        } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
            RunNumaLinearGGUF(inputData, weight, outputData, biasData, n, m, k);
        }
    }

    bool NumaLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        
        // 检查数据类型支持
        if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT8) {
            return true;
        }
        if (input.dataType == DataType::FLOAT32 && weight.dataType == DataType::FLOAT32) {
            return true;
        }
        return false;
    }

    // NumaMergeMOEOp实现
    void NumaMergeMOEOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.dataType = input.dataType;
        output.Resize(input.dims);
    }

    void NumaMergeMOEOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &routerLogits = *(datas.find("routerLogits")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);

        // 确保所有数据在CPU上，避免CUDA内存分配
        input.ToDevice(DataDevice::CPU);
        output.ToDevice(DataDevice::CPU);
        routerLogits.ToDevice(DataDevice::CPU);

        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        int batch = intParams.find("input___batch") != intParams.end() ? intParams.find("input___batch")->second : 1;
        int expertNum = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : 1;

        // 计算维度
        int n = input.Count(0) / input.dims.back();
        int hidden_size = input.dims.back();

        // 准备专家输出数组
        std::vector<float*> expert_outputs(expertNum);
        for (int i = 0; i < expertNum; i++) {
            if (weights[i]) {
                weights[i]->ToDevice(DataDevice::CPU);
                expert_outputs[i] = (float*)weights[i]->cpuData;
            }
        }

        // 使用NUMA优化的MoE合并操作
        RunNumaMoeMerge(expert_outputs, (float*)routerLogits.cpuData, (float*)output.cpuData, 
                       n, expertNum, hidden_size);
    }

    bool NumaMergeMOEOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // NUMA MoE操作支持所有数据类型
        return true;
    }

    // NumaAttentionOp实现
    void NumaAttentionOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 使用CPU Attention的Reshape逻辑
        CpuAttentionOp cpuOp;
        cpuOp.Reshape(opType, datas, floatParams, intParams);
    }

    void NumaAttentionOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 确保所有数据在CPU上
        for (auto &it : datas) {
            if (it.second) {
                it.second->ToDevice(DataDevice::CPU);
            }
        }
        
        // 使用CPU Attention操作（NUMA优化在线程池层面实现）
        CpuAttentionOp cpuOp;
        cpuOp.Run(opType, datas, floatParams, intParams);
    }

    bool NumaAttentionOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    // NumaCatDirectOp实现
    void NumaCatDirectOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 使用CPU CatDirect的Reshape逻辑
        CpuCatDirectOp cpuOp;
        cpuOp.Reshape(opType, datas, floatParams, intParams);
    }

    void NumaCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 确保所有数据在CPU上
        for (auto &it : datas) {
            if (it.second) {
                it.second->ToDevice(DataDevice::CPU);
            }
        }
        
        // 使用CPU CatDirect操作（NUMA优化在线程池层面实现）
        CpuCatDirectOp cpuOp;
        cpuOp.Run(opType, datas, floatParams, intParams);
    }

    bool NumaCatDirectOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }
}