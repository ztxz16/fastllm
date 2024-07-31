//
// Created by TylunasLi on 7/15/24.
//

#ifndef FASTLLM_ASCEND_DEVICE_H
#define FASTLLM_ASCEND_DEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

namespace fastllm {

    typedef std::map<std::string, std::pair<std::vector<int>, std::vector<std::vector<int64_t>>>> DynamicShapeDict;
    typedef std::vector<std::pair<std::string, Data*>> OrderedData;
    typedef std::map <std::string, bool> BoolDict;

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
        // 是否可以运行某一个算子
        virtual bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        // 不编译，直接运行一个算子
        bool RunSingleOp(const std::string &opType, const OrderedData &inputData,
                         const DataDict &outputData, const FloatDict &floatParams,
                         const IntDict &intParams, const BoolDict &boolParams);
        // 动态shape编译，并运行一个算子
        bool CompileAndRunSingleOp(const std::string &opType, const OrderedData &inputData, const fastllm::DataDict &outputData,
                                   const DynamicShapeDict &dynamicShapes, const FloatDict &floatParams,
                                   const IntDict &intParams, const BoolDict &boolParams);
    protected:
        bool warmUpMode;
        bool deviceOk = true;
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