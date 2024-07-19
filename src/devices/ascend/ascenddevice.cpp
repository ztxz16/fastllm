#include "ascenddevice.h"

#include "fastllm-acl.h"

#include <acl/acl.h>

namespace fastllm {
    AscendNpuDevice::AscendNpuDevice() {
        this->deviceType = "acl";
        npu::FastllmAclInit();
    }

    AscendNpuDevice::~AscendNpuDevice() {
        npu::FastllmAclFinalize();
    }

    bool AscendNpuDevice::Malloc(void **ret, size_t size) {
        *ret = npu::FastllmAclMalloc(size);
        return (*ret != nullptr);
    }

    bool AscendNpuDevice::Free(void *ret) {
        npu::FastllmAclFree(ret);
        return true;
    }

    bool AscendNpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromHostToDevice(dst, src, size);
        return result == 0;
    }

    bool AscendNpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        int result = npu::FastllmAclCopyFromDeviceToHost(dst, src, size);
        return result == 0;
    }

    std::map<DataType, aclDataType> BaseAscendOperator::dataTypes = {
            {DataType::FLOAT32, aclDataType::ACL_FLOAT},
            {DataType::FLOAT16, aclDataType::ACL_FLOAT16},
            {DataType::INT8, aclDataType::ACL_INT8},
            {DataType::INT32PARAM, aclDataType::ACL_INT32},
            {DataType::INT16, aclDataType::ACL_INT16},
            {DataType::BIT, aclDataType::ACL_BOOL}
        };

    void BaseAscendOperator::ToAclTensor(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
                                         std::vector<aclDataBuffer *> buffers) {
        std::vector<int64_t> expandDims64(data.second->dims.size(), 0);
        for (int i = 0; i < data.second->dims.size(); i++)
            expandDims64[i] = (int64_t) data.second->expansionDims[i];
        aclTensorDesc* tensor = aclCreateTensorDesc(dataTypes[data.second->dataType], data.second->expansionDims.size(),
                                           expandDims64.data(), aclFormat::ACL_FORMAT_ND);
        tensors.emplace_back(tensor);
        aclDataBuffer* buffer = aclCreateDataBuffer(data.second->deviceData, data.second->expansionBytes);
        buffers.emplace_back(buffer);
    }

    void BaseAscendOperator::ToAclOpAttr(const FloatDict &floatParams, const IntDict &intParams, aclopAttr *opAttr) {

    }

    void BaseAscendOperator::DestoryAclTensors(std::vector<aclTensorDesc *> &inputTensors, std::vector<aclDataBuffer *> &inputBuffers,
        std::vector<aclTensorDesc *> &outputTensors, std::vector<aclDataBuffer *> outputBuffers) {
        for (size_t i = 0; i < inputTensors.size(); ++i) {
            aclDestroyTensorDesc(inputTensors[i]);
        }
        inputTensors.clear();
        for (size_t i = 0; i < inputBuffers.size(); ++i) {
            aclDestroyDataBuffer(inputBuffers[i]);
        }
        inputBuffers.clear();
        for (size_t i = 0; i < outputTensors.size(); ++i) {
            aclDestroyTensorDesc(outputTensors[i]);
        }
        outputTensors.clear();
        for (size_t i = 0; i < outputBuffers.size(); ++i) {
            aclDestroyDataBuffer(outputBuffers[i]);
        }
        outputBuffers.clear();
    }

}