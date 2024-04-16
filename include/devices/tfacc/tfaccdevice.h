//
// Created by huangyuyang on 4/11/24.
//

#ifndef FASTLLM_TFACCDEVICE_H
#define FASTLLM_TFACCDEVICE_H

#include "device.h"

namespace fastllm {
    class TfaccDevice : BaseDevice {
    public:
        TfaccDevice();

        // tfacc use cpu DDR
        bool Malloc (void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class TfaccLinearOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        long long int Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif