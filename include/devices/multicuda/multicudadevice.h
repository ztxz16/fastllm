//
// Created by huangyuyang on 8/2/24.
//

#ifndef FASTLLM_MULTICUDADEVICE_H
#define FASTLLM_MULTICUDADEVICE_H

#include "device.h"

namespace fastllm {
    class MultiCudaDevice : BaseDevice {
    public:
        MultiCudaDevice ();

        bool Malloc (void **ret, size_t size); // 分配尺寸为size的空间
        bool Free(void *ret); // 释放ret

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class MultiCudaLinearOp : CudaLinearOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif //FASTLLM_MULTICUDADEVICE_H
