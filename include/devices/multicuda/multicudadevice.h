//
// Created by huangyuyang on 8/2/24.
//

#ifndef FASTLLM_MULTICUDADEVICE_H
#define FASTLLM_MULTICUDADEVICE_H

#include "device.h"

namespace fastllm {
    class MultiCudaDevice : BaseDevice {
        
    private:
        CudaDevice *cudaDevice;

    public:
        MultiCudaDevice (CudaDevice *cudaDevice);

        bool Malloc (void **ret, size_t size); // 分配尺寸为size的空间
        bool Free(void *ret); // 释放ret

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);

        // 是否可以运行某一个算子
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行形状推理
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);

        // 对某一个算子进行推理
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaLinearOp : CudaLinearOp {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMLPOp : CudaLinearOp {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMergeMOE : CpuMergeMOE {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class MultiCudaMergeAttention : CudaMergeAttention {
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif //FASTLLM_MULTICUDADEVICE_H
