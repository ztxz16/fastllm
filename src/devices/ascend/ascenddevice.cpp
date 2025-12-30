//
// Created by Claude on 2024-12-30.
// Ascend NPU Device Support for fastllm
//

#include "devices/ascend/ascenddevice.h"
#include "devices/cpu/cpudevice.h"
#include "utils/utils.h"

namespace fastllm {
    // Global Ascend device instance
    static AscendDevice *gAscendDevice = nullptr;

    AscendDevice::AscendDevice() : initialized(false), deviceId(0), context(nullptr), stream(nullptr) {
        this->deviceType = "ascend";
        this->deviceName = "Ascend NPU";

        // Initialize ACL
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclInit failed, ret = %d\n", ret);
            return;
        }

        // Set device
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclrtSetDevice failed, ret = %d\n", ret);
            aclFinalize();
            return;
        }

        // Create context
        ret = aclrtCreateContext(&context, deviceId);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclrtCreateContext failed, ret = %d\n", ret);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return;
        }

        // Create stream
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclrtCreateStream failed, ret = %d\n", ret);
            aclrtDestroyContext(context);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return;
        }

        // Register operators
        this->ops["Linear"] = (BaseOperator *)(new AscendLinearOp());
        this->ops["Attention"] = (BaseOperator *)(new AscendAttention());
        this->ops["LayerNorm"] = (BaseOperator *)(new AscendLayerNorm());

        initialized = true;
        fprintf(stderr, "Ascend NPU initialized successfully on device %d\n", deviceId);
    }

    AscendDevice::~AscendDevice() {
        if (stream) {
            aclrtDestroyStream(stream);
        }
        if (context) {
            aclrtDestroyContext(context);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    }

    bool AscendDevice::Malloc(void **ret, size_t size) {
        if (!initialized) {
            return false;
        }
        aclError err = aclrtMalloc(ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            fprintf(stderr, "aclrtMalloc failed, size = %zu, ret = %d\n", size, err);
            return false;
        }
        return true;
    }

    bool AscendDevice::Free(void *ret) {
        if (!initialized) {
            return false;
        }
        aclError err = aclrtFree(ret);
        if (err != ACL_SUCCESS) {
            fprintf(stderr, "aclrtFree failed, ret = %d\n", err);
            return false;
        }
        return true;
    }

    bool AscendDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        if (!initialized) {
            return false;
        }
        aclError err = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
        if (err != ACL_SUCCESS) {
            fprintf(stderr, "aclrtMemcpy D2H failed, ret = %d\n", err);
            return false;
        }
        return true;
    }

    bool AscendDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        if (!initialized) {
            return false;
        }
        aclError err = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            fprintf(stderr, "aclrtMemcpy H2D failed, ret = %d\n", err);
            return false;
        }
        return true;
    }

    // AscendLinearOp implementation
    bool AscendLinearOp::CanRun(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        if (datas.find("weight") == datas.end()) {
            return true;
        }
        Data *weight = datas.at("weight");
        return weight == nullptr ||
               weight->dataType == DataType::INT4_NOZERO ||
               weight->dataType == DataType::INT8 ||
               weight->dataType == DataType::INT4_GROUP ||
               weight->dataType == DataType::FLOAT32 ||
               weight->dataType == DataType::FLOAT16;
    }

    void AscendLinearOp::Reshape(const std::string &opType, const DataDict &datas,
                                 const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        if (intParams.find("exType") != intParams.end()) {
            LinearExType type = (LinearExType)intParams.find("exType")->second;
            if (type == LinearExType::ExSwiglu) {
                dims.back() /= 2;
            }
        }

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void AscendLinearOp::Run(const std::string &opType, const DataDict &datas,
                             const FloatDict &floatParams, const IntDict &intParams) {
        // TODO: Implement ACL MatMul operator using aclnnMatmul
        // For now, this is a placeholder that will use CPU fallback via the executor
        Data &output = *(datas.find("output")->second);
        output.Allocate();
    }

    // AscendAttention implementation
    void AscendAttention::Run(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        // TODO: Implement ACL Attention operator using aclnnMatmul, aclnnSoftmaxV2, etc.
        // For now, this is a placeholder that will use CPU fallback via the executor
        Data &output = *(datas.find("output")->second);
        output.Allocate();
    }

    // AscendLayerNorm implementation
    void AscendLayerNorm::Run(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        // TODO: Implement ACL LayerNorm operator using aclnnLayerNorm
        // For now, this is a placeholder that will use CPU fallback via the executor
        Data &output = *(datas.find("output")->second);
        output.Allocate();
    }

} // namespace fastllm
