//
// Created by huangyuyang on 4/11/24.
//

#ifndef FASTLLM_NUMADEVICE_H
#define FASTLLM_NUMADEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

namespace fastllm {
    class NumaDevice : BaseDevice {
    public:
        NumaDevice();

        // numa use cpu DDR
        bool Malloc (void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class NumaLinearOp : CpuLinearOp {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        long long int Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumaMergeMOE : CpuMergeMOE {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumaCatDirectOp : CpuCatDirectOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumaAttention : CpuAttention {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumaAttentionBatchOp : CpuAttentionBatchOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumaCatDirectBatchOp : CpuCatDirectBatchOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif