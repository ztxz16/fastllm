//
// Created by Claude on 2024-12-30.
// Ascend NPU Device Support for fastllm
//

#ifndef FASTLLM_ASCENDDEVICE_H
#define FASTLLM_ASCENDDEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

#ifdef USE_ASCEND
#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

namespace fastllm {
    class AscendDevice : public BaseDevice {
    public:
        AscendDevice();
        ~AscendDevice();

        // Memory management
        bool Malloc(void **ret, size_t size) override;
        bool Free(void *ret) override;

        // Data transfer between CPU and NPU
        bool CopyDataToCPU(void *dst, void *src, size_t size) override;
        bool CopyDataFromCPU(void *dst, void *src, size_t size) override;

        // Check if NPU is initialized
        bool IsInitialized() const { return initialized; }

    private:
        bool initialized;
        int deviceId;
        aclrtContext context;
        aclrtStream stream;
    };

    // Linear operator using Ascend NPU
    class AscendLinearOp : public BaseOperator {
    public:
        bool CanRun(const std::string &opType, const DataDict &datas,
                    const FloatDict &floatParams, const IntDict &intParams) override;
        void Reshape(const std::string &opType, const DataDict &datas,
                     const FloatDict &floatParams, const IntDict &intParams) override;
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams) override;
    };

    // Attention operator using Ascend NPU
    class AscendAttention : public BaseOperator {
    public:
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams) override;
    };

    // LayerNorm operator using Ascend NPU
    class AscendLayerNorm : public BaseOperator {
    public:
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams) override;
    };

} // namespace fastllm

#endif // USE_ASCEND

#endif // FASTLLM_ASCENDDEVICE_H
