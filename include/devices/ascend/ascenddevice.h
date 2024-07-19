//
// Created by TylunasLi on 7/15/24.
//

#ifndef FASTLLM_ASCEND_DEVICE_H
#define FASTLLM_ASCEND_DEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"
#include <acl/acl_base.h>
#include <acl/acl_op.h>

namespace fastllm {
    class AscendNpuDevice : BaseDevice {
    public:
        AscendNpuDevice();
        virtual ~AscendNpuDevice();

        bool Malloc(void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    /**
     * 对于简单的算子，每个AscendOperator负责算子的转换，FastAclExecuteOp负责算子的执行
     */
    class BaseAscendOperator : BaseOperator {
    public:
        /**
         * 将张量转换为Ascend CL张量信息
         * @param[in] datas fastllm 张量对
         * @param[out] tensors Ascend CL张量定义
         * @param[out] buffers Ascend CL张量shape定义
         */
        void ToAclTensor(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
            std::vector<aclDataBuffer *> buffers);

        /**
         * 将算子參數转换为Ascend CL算子參數信息
         * @param floatParams float类型參數
         * @param intParams int类型參數
         */
        void ToAclOpAttr(const FloatDict &floatParams, const IntDict &intParams, aclopAttr *opAttr);

        /**
         *
         * @param inputTensors Ascend CL张量定义
         * @param inputBuffers Ascend CL张量shape定义
         * @param outputTensors Ascend CL张量定义
         * @param outputBuffers Ascend CL张量shape定义
         */
        void DestoryAclTensors(std::vector<aclTensorDesc *> &inputTensors, std::vector<aclDataBuffer *> &inputBuffers,
                               std::vector<aclTensorDesc *> &outputTensors, std::vector<aclDataBuffer *> outputBuffers);

    protected:
        std::string name;

        static std::map<DataType, aclDataType> dataTypes;
    };

    class AscendLinearOp : BaseAscendOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif // FASTLLM_ASCEND_DEVICE_H