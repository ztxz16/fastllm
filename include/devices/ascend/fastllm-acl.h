// Created By TylunasLi 2024-07-15

#ifndef FASTLLM_ASCEND_ADAPTER_H
#define FASTLLM_ASCEND_ADAPTER_H

#include "fastllm.h"
#include "device.h"
#include <acl/acl_base.h>
#include <acl/acl_op.h>

namespace fastllm {

    namespace npu {

        void FastllmAclInit(void);
        void FastllmAclFinalize(void);

        void FastllmAclSetDevice(int32_t device_id);

        void* FastllmAclMalloc(size_t size);
        void FastllmAclFree(void *ret);
        void* FastllmAclDirectMalloc(size_t size);
        void FastllmAclDirectFree(void *ret);
        void FastllmAclMallocBigBuffer(size_t size);
        void FastllmAclClearBigBuffer();
        void FastllmAclClearBuffer();

        int FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size);
        int FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size);
        void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size);
        void FastllmAclMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);

        /**
         * 将张量转换为Ascend CL张量信息
         * @param[in] datas fastllm 张量对
         * @param[out] tensors Ascend CL张量shape定义
         * @param[out] buffers Ascend CL张量内存地址
         */
        void FastllmAclToTensor(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
                                std::vector<aclDataBuffer *> &buffers);

        /**
         * 将张量转换为Ascend CL张量信息
         * @param[in] datas fastllm 张量对
         * @param[out] tensors Ascend CL张量shape定义
         * @param[in] dynamicDimension 设置为动态的维度
         * @param[in] dynamicRange 各维度对应的动态的范围
         */
        void FastllmAclCreateShape(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
                                   std::vector<int> dynamicDimension = {}, std::vector<std::vector<int64_t>> dynamicRange = {});


        /**
         * 将算子属性转换为Ascend CL算子属性信息
         * @param floatParams float类型參數
         * @param intParams int类型參數
         * @param boolParams bool类型參數
         * @param[out] opAttr AscendCL算子属性
         */
        void FastllmAclToOpAttribute(const FloatDict &floatParams, const IntDict &intParams,
                                     const std::map <std::string, bool> &boolParams, aclopAttr **opAttr);

        /**
         * 释放算子输入输出定义（但不释放内存）
         * @param inputTensors Ascend CL输入张量shape定义
         * @param inputBuffers Ascend CL输入张量内存地址
         * @param outputTensors Ascend CL输出张量shape定义
         * @param outputBuffers Ascend CL输出张量内存地址
         */
        void FastllmAclDestoryTensors(std::vector<aclTensorDesc *> &inputTensors, std::vector<aclDataBuffer *> &inputBuffers,
                                      std::vector<aclTensorDesc *> &outputTensors, std::vector<aclDataBuffer *> &outputBuffers, aclopAttr **opAttr);
        /**
         * 仅释放 tensorShapes
         */
        void FastllmAclDestroyShape(std::vector<aclTensorDesc *> &tensorShapes);

        /**
         * 初始化（编译）可以复用的算子
         */
        bool FastllmAclInitOp(std::string name, std::vector<aclTensorDesc *> &inputTensorShapes,
                              std::vector<aclTensorDesc *> &outputTensorShapes, aclopAttr *opAttr);

        /**
         * 执行已编译过的算子
         */
        bool FastllmAclExecuteAfterInit(std::string name, std::vector<aclTensorDesc *> &inputTensors,
                                        std::vector<aclDataBuffer *> &inputBuffers,
                                        std::vector<aclTensorDesc *> &outputTensors,
                                        std::vector<aclDataBuffer *> &outputBuffers, aclopAttr *opAttr);

        /**
         * 执行没有编译过的算子
         */
        bool FastllmAclExecute(std::string name, std::vector<aclTensorDesc *> &inputTensors,
                               std::vector<aclDataBuffer *> &inputBuffers,
                               std::vector<aclTensorDesc *> &outputTensors,
                               std::vector<aclDataBuffer *> &outputBuffers, aclopAttr *opAttr);

    }
}

#endif // FASTLLM_ASCEND_ADAPTER_H