//
// Created by huangyuyang on 2/24/225.
//

#ifndef FASTLLM_TOPSDEVICE_H
#define FASTLLM_TOPSDEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

namespace fastllm {
    class TopsDevice : BaseDevice {
    public:
        TopsDevice();

        bool Malloc (void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class TopsLinearOp : CpuLinearOp {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        long long int Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif