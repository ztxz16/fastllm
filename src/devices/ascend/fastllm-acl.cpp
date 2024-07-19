#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-acl.h"
#include <acl/acl_base.h>
#include <acl/acl.h>
#include <acl/ops/acl_cblas.h>

#define checkAclError(message, val) showError(val, message, __FILE__, __LINE__)

void showError(aclError state, char const* const message, const char* const file,
           int const line) {
    if (ACL_SUCCESS != state) {
        printf("%s\n  AsecndCL error = %d at %s:%d\n", message, state, file, line);
        if (state >= 500000)
            exit(state);
    }
}

namespace fastllm {
namespace npu {

void FastllmAclInit() {
    aclError state = aclInit(NULL);
    checkAclError("Error: Ascend CL error when init!", state);
}

static std::map<int32_t, aclrtContext> FastllmAclContextMap;
aclrtContext getFastllmAclContextHandle() {
    int32_t id = -1;
    aclrtGetDevice(&id);
    auto it = FastllmAclContextMap.find(id);
    if (it != FastllmAclContextMap.end()) {
        return it->second;
    }
    aclrtContext context = nullptr;
    aclError state = aclrtCreateContext(&context, id);
    if (state != ACL_SUCCESS) {
        printf("Error: Acl Create Context Failed state %d.\n", state);
        exit(state);
    } else {
        FastllmAclContextMap[id] = context;
    }
    return context;
}

void FastllmAclFinalize() {
    for (auto &it : FastllmAclContextMap) {
        aclError state = aclrtDestroyContext(it.second);
        checkAclError("Error: Ascend CL error when destory context!", state);
        state = aclrtResetDevice(it.first);
        checkAclError("Error: Ascend CL reset device failed", state);
    }
    aclError state = aclFinalize();
    checkAclError("Error: Ascend CL error when finalize!", state);
}

void FastllmAclSetDevice(int32_t device_id) {
    aclError state = aclrtSetDevice(device_id);
    checkAclError("Error: Ascend CL error when set device!", state);
}

void DeviceSync() {
    //aclrtSynchronizeDevice();
}

void * FastllmAclPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::NPU) {
        ret = (void*)input.deviceData;
    } else {
        ret = (void*)(input.expansionBytes);
        auto state = aclrtMemcpy(ret, input.expansionBytes, input.cpuData, input.expansionBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ACL_SUCCESS != state) {
            checkAclError("Error: AscendCL error when copy from memory to NPU!", state);
            return nullptr;
        }
    }
    return ret;
}

void FastllmAclFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::NPU) {
        FastllmAclFree(data);
    }
}

void * FastllmAclPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::NPU) {
        ret = (float*)output.deviceData;
    } else {
        ret = (float*)FastllmAclMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmAclFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::NPU) {
        auto state = aclrtMemcpy(output.cpuData, output.expansionBytes, data, output.expansionBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        checkAclError("Error: AscendCL error when copy from NPU to memory!", state);
        FastllmAclFree(data);
    }
    DeviceSync();
}

struct AscendMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    AscendMemoryBuffer () {}

    AscendMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};

std::map<int32_t, std::vector <AscendMemoryBuffer>> buffersMap;
std::map<int32_t, size_t> noBusyCnt;
std::map<int32_t, std::vector <AscendMemoryBuffer>> bigBuffersMap;

void * FastllmAclDirectMalloc(size_t size) {
    void * ret;
    aclError state = aclrtMalloc(&ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_SUCCESS != state) {
        printf("Error: AscendCL error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkAclError("", state);
        return nullptr;
    }
    return ret;
}

void FastllmAclDirectFree(void *ret) {
    aclError state = aclrtFree(ret);
    //checkAclError("Error: AscendCL error when release memory!", state);
}

void * FastllmAclMalloc(size_t size) {
    int32_t id = -1;
    aclError state = ACL_SUCCESS;
    state = aclrtGetDevice(&id);
    checkAclError("Error: AscendCL error when find device!", state);
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 1 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void * ret;
        state = aclrtMalloc(&ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ACL_SUCCESS != state) {
            printf("Error: AscendCL error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
            checkAclError("", state);
            return nullptr;
        }
        bigBuffers.push_back(AscendMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &buffers = buffersMap[id];
    for (int i = 0; i < buffers.size(); i++) {
        if (buffers[i].size >= size && !buffers[i].busy) {
            buffers[i].busy = true;
            noBusyCnt[id] -= buffers[i].size;
            return buffers[i].data;
        }
    }
    void * ret;
    state = aclrtMalloc(&ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_SUCCESS != state) {
        printf("Error: AscendCL error when allocating %lu KB memory! maybe there's no enough memory left on device.", size >> 10);
        checkAclError("", state);
        return nullptr;
    }
    buffers.push_back(AscendMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmAclFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (buffersMap.empty())
        return;
    aclError state = ACL_SUCCESS;
    for (auto &it: buffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &buffers = it.second;
            std::vector <AscendMemoryBuffer> temp;
            for (int i = 0; i < buffers.size(); i++) {
                if (!buffers[i].busy) {
                    state = aclrtSetDevice(it.first);
                    state = aclrtFree(buffers[i].data);
                    if (ACL_SUCCESS != state)
                        printf("Error: AscendCL error when release memory on device %d!", it.first);
                    checkAclError("", state);
                } else {
                    temp.push_back(buffers[i]);
                }
            }
            buffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    for (auto &it: buffersMap) {
        auto &buffers = it.second;
        for (int i = 0; i < buffers.size(); i++) {
            if (buffers[i].data == ret) {
                noBusyCnt[it.first] += buffers[i].size;
                buffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    state = aclrtFree(ret);
    checkAclError("AscendCL error when release memory!", state);
}

void FastllmAclMallocBigBuffer(size_t size) {
    void * ret;
    int32_t id = -1;
    aclrtGetDevice(&id);
    auto &bigBuffers = bigBuffersMap[id];
    aclError state = aclrtMalloc(&ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_SUCCESS != state)
        printf("Error: AscendCL error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
    checkAclError("", state);
    bigBuffers.push_back(AscendMemoryBuffer(ret, size, false));
}

void FastllmAclClearBigBuffer() {
    int32_t id = -1;
    aclrtGetDevice(&id);
    if (bigBuffersMap.empty())
        return;
    aclError state = ACL_SUCCESS;
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector <AscendMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                state = aclrtSetDevice(it.first);
                state = aclrtFree(bigBuffers[i].data);
                if (ACL_SUCCESS != state)
                    printf("Error: AscendCL error when release memory on device %d!", it.first);
                checkAclError("", state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    aclrtSetDevice(id);
}


void FastllmAclClearBuffer() {
    int32_t id = -1;
    aclrtGetDevice(&id);
    if (bigBuffersMap.empty())
        return;
    aclError state = ACL_SUCCESS;
    for (auto &it : buffersMap) {
        auto &buffers = it.second;
        std::vector <AscendMemoryBuffer> temp;
        for (int i = 0; i < buffers.size(); i++) {
            if (!buffers[i].busy) {
                state = aclrtSetDevice(it.first);
                state = aclrtFree(buffers[i].data);
                if (ACL_SUCCESS != state)
                    printf("Error: AscendCL error when release memory on device %d!", it.first);
                checkAclError("", state);
            } else {
                temp.push_back(buffers[i]);
            }
        }
        buffers.clear();
        buffers = temp;
    }
    aclrtSetDevice(id);
}

int FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size) {
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError("Error: AscendCL error when copy from memory to NPU!", state);
    return (int) state;
}

int FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError("Error: AscendCL error when copy from NPU to memory!", state);
    return (int) state;
}

void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    checkAclError("Error: AscendCL error when copy on NPU!", state);
}

void FastllmAclMemcpyBetweenDevices(int32_t dstId, void *dst, int32_t srcId, void *src, size_t size) {
    int canAccessPeer = 0;
    aclError state = aclrtDeviceCanAccessPeer(&canAccessPeer, srcId, dstId);
    if (canAccessPeer && state == ACL_SUCCESS) {
        state = aclrtDeviceEnablePeerAccess(dstId, 0);
        state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    } else {
        uint8_t *cpuData = new uint8_t[size];
        state = aclrtSetDevice(srcId);
        state = aclrtMemcpy(cpuData, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);

        state = aclrtSetDevice(dstId);
        state = aclrtMemcpy(dst, size, cpuData, size, ACL_MEMCPY_HOST_TO_DEVICE);
        delete[] cpuData;
    }
    checkAclError("Error: AscendCL error when copy Between NPUs!", state);
}

/*
void FastllmAclMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
}
*/

}
}
