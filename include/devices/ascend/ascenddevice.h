//
// Created by TylunasLi on 7/15/24.
//

#ifndef FASTLLM_ASCEND_DEVICE_H
#define FASTLLM_ASCEND_DEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

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
    class BaseAscendOperator : public BaseOperator {
    public:
        BaseAscendOperator() {}
        BaseAscendOperator(std::string name) : name(name) {}
    protected:
        bool warmUpMode;
        std::string name;
    };

    class AscendLinearOp : public BaseAscendOperator {
    public:
        AscendLinearOp();
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

}

#endif // FASTLLM_ASCEND_DEVICE_H