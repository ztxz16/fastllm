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

    AscendLinearOp::AscendLinearOp() :
        BaseAscendOperator("FullyConnection") {}

    void AscendLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
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
        output.Allocate();
    }

    bool AscendLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return false;
    }

}