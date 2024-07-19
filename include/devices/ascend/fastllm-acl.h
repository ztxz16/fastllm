#include "fastllm.h"
// Created By TylunasLi 2024-07-15

#ifndef FASTLLM_ASCEND_ADAPTER_H
#define FASTLLM_ASCEND_ADAPTER_H

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
    }
}

#endif // FASTLLM_ASCEND_ADAPTER_H