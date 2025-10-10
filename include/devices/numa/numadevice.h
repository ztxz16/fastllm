//
// Created by fastllm-numa on 12/25/24.
//

#ifndef FASTLLM_NUMADEVICE_H
#define FASTLLM_NUMADEVICE_H

#include "device.h"
#include "numathreadpool.h"
#include "numamoe.h"

namespace fastllm {
    class NumaDevice : public BaseDevice {
    public:
        NumaDevice();

        bool Malloc(void **ret, size_t size); // 在NUMA节点上分配内存

        bool Free(void *ret); // 释放NUMA内存

        bool CopyDataToCPU(void *dst, void *src, size_t size); // 从NUMA内存复制到CPU

        bool CopyDataFromCPU(void *dst, void *src, size_t size); // 从CPU复制到NUMA内存
    };

    // NUMA Linear操作
    class NumaLinearOp : public BaseOperator {
    public:
        void Reshape(const std::string &opType, const fastllm::DataDict &datas,
                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        void Run(const std::string &opType, const fastllm::DataDict &datas,
                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        bool CanRun(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);
    };

    // NUMA MoE Merge操作
    class NumaMergeMOEOp : public BaseOperator {
    public:
        void Reshape(const std::string &opType, const fastllm::DataDict &datas,
                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        void Run(const std::string &opType, const fastllm::DataDict &datas,
                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        bool CanRun(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);
    };

    // NUMA Attention操作
    class NumaAttentionOp : public BaseOperator {
    public:
        void Reshape(const std::string &opType, const fastllm::DataDict &datas,
                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        void Run(const std::string &opType, const fastllm::DataDict &datas,
                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        bool CanRun(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);
    };

    // NUMA CatDirect操作
    class NumaCatDirectOp : public BaseOperator {
    public:
        void Reshape(const std::string &opType, const fastllm::DataDict &datas,
                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        void Run(const std::string &opType, const fastllm::DataDict &datas,
                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);

        bool CanRun(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams);
    };
}

#endif //FASTLLM_NUMADEVICE_H