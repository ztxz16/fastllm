//
// Created by huangyuyang on 6/13/23.
//

#include "utils.h"
#include "device.h"

namespace fastllm {
    bool BaseDevice::Malloc(void **ret, Data &data) {
        return Malloc(ret, data.expansionBytes);
    }

    bool BaseDevice::CopyDataFromCPU(Data &data) {
        AssertInFastLLM(data.cpuData != nullptr, "Copy data to " + this->deviceName + " from cpu failed: cpu's data is null.\n");
        AssertInFastLLM(data.deviceData == nullptr, "Copy data to " + this->deviceName + " from cpu failed: device's data is not null.\n");
        Malloc(&data.deviceData, data.expansionBytes);
        bool ret = CopyDataFromCPU(data.cudaData, data.cpuData, data.expansionBytes);
        delete[] data.cpuData;
        data.cpuData = nullptr;
        return ret;
    }

    bool BaseDevice::CopyDataToCPU(Data &data) {
        AssertInFastLLM(data.cpuData == nullptr, "Copy data from " + this->deviceName + " to cpu failed: cpu's data is not null.\n");
        AssertInFastLLM(data.deviceData != nullptr, "Copy data from " + this->deviceName + " to cpu failed: device's data is null.\n");
        data.cpuData = new uint8_t [data.expansionBytes];
        bool ret = CopyDataToCPU(data.cpuData, data.deviceData, data.expansionBytes);
        this->Free(data.deviceData);
        data.deviceData = nullptr;
        return ret;
    }

    bool BaseDevice::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (this->ops.find(opType) == this->ops.end()) {
            return false;
        }
        return this->ops[opType]->CanRun(opType, datas, floatParams, intParams);
    }

    void BaseDevice::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        this->ops[opType]->Reshape(opType, datas, floatParams, intParams);
    }

    void BaseDevice::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        this->ops[opType]->Run(opType, datas, floatParams, intParams);
    }

    bool BaseOperator::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                              const IntDict &intParams) {
        return true;
    }

    void BaseOperator::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams,
                               const IntDict &intParams) {
        if (datas.find("output") == datas.end()) {
            return;
        }
        // 默认的Reshape，把output和input变成一样的形状
        Data *inputs = (datas.find("input")->second);
        Data *outputs = (datas.find("output")->second);
        if (inputs == outputs) {
            return;
        }
        outputs[0].dataType = inputs[0].dataType;
        outputs[0].Resize(inputs[0].dims);
    }

    void BaseBatchOperator::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (datas.find("output") == datas.end()) {
            return;
        }
        // 默认的Reshape，把output和input变成一样的形状
        Data **inputs = (Data**)(datas.find("input")->second);
        Data **outputs = (Data**)(datas.find("output")->second);
        if (inputs == outputs) {
            return;
        }

        int batch = 1;
        if (intParams.find("input___batch") != intParams.end()) {
            batch = intParams.find("input___batch")->second;
        }

        for (int i = 0; i < batch; i++) {
            outputs[i]->dataType = inputs[i]->dataType;
            outputs[i]->Resize(inputs[i]->dims);
        }
    }
}