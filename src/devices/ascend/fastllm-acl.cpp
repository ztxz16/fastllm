#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-acl.h"
#include <acl/acl_base.h>
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>

#define checkAclErrorFormat(message, state, args...)                                 \
    if (ACL_SUCCESS != state) {                                                      \
        printf(message, ##args);                                                     \
        printf("AscendCL error = %d at %s:%d\n", state, "fastllm-acl.cpp", __LINE__);\
        if (state >= 500000) {                                                       \
            aclFinalize();                                                           \
            exit(state);                                                             \
        }                                                                            \
    }

#define checkAclError(message, val) showError(val, message, "fastllm-acl.cpp", __LINE__)

void showError(aclError state, char const* const message, const char* const file,
           int const line) {
    if (ACL_SUCCESS != state) {
        printf("%s  AscendCL error = %d at %s:%d\n", message, state, file, line);
        if (state >= 500000) {
            aclFinalize();
            exit(state);
        }
    }
}

namespace fastllm {
namespace npu {

void FastllmAclInit() {
    aclError state = aclInit(NULL);
    checkAclError("Error: AscendCL error when init!", state);
}

static std::map<int32_t, aclrtContext> FastllmAclContextMap;
aclrtContext getFastllmAclContextHandle(int32_t id) {
    // aclrtGetDevice(&id);
    auto it = FastllmAclContextMap.find(id);
    if (it != FastllmAclContextMap.end()) {
        return it->second;
    }
    aclrtContext context = nullptr;
    aclError state = aclrtCreateContext(&context, id);
    if (state != ACL_SUCCESS) {
        printf("Error: AscendCL Create Context Failed state %d.\n", state);
        exit(state);
    } else {
        FastllmAclContextMap[id] = context;
    }
    return context;
}

static int32_t curDeviceId = -1;

static std::map<int32_t, aclrtStream> FastllmAclStreamMap;
aclrtStream getFastllmAclStream(int32_t id) {
    // aclrtGetDevice(&id);
    auto it = FastllmAclStreamMap.find(id);
    if (it != FastllmAclStreamMap.end()) {
        return it->second;
    }
    aclrtStream stream = nullptr;
    aclError state = aclrtCreateStream(&stream);
    if (state != ACL_SUCCESS) {
        printf("Error: AscendCL error when create stream! state %d.\n", state);
    } else {
        FastllmAclStreamMap[id] = stream;
    }
    return stream;
}

static thread_local bool aclInitialized = false;

static inline void EnsureAclContext() {
    // 如果当前线程还没初始化过，且主线程已经初始化过
    if (!aclInitialized && curDeviceId != -1) {
        // 在新线程中调用 SetDevice，这是 ACL 的强制要求
        aclError state = aclrtSetDevice(curDeviceId);
        checkAclError("Error: AscendCL error when set device in new thread!", state);
        state = aclrtSetCurrentContext(getFastllmAclContextHandle(curDeviceId));
        checkAclError("Error: AscendCL error when switch context!", state);
        aclInitialized = true; // 标记新线程已初始化，后续不再重复调用
    }
}

void FastllmAclFinalize() {
    aclError state = ACL_SUCCESS;
    for (auto &it : FastllmAclStreamMap) {
        state = aclrtSetCurrentContext(getFastllmAclContextHandle(it.first));
        state = aclrtSynchronizeStream(it.second);
        state = aclrtDestroyStream(it.second);
        checkAclError("Error: AscendCL error when destroy stream!", state);
    }
    FastllmAclStreamMap.clear();
    for (auto &it : FastllmAclContextMap) {
        state = aclrtDestroyContext(it.second);
        checkAclError("Error: AscendCL error when destory context!", state);
        // state = aclrtResetDevice(it.first);
        // checkAclError("Error: Ascend CL reset device failed", state);
    }
    FastllmAclContextMap.clear();
    state = aclFinalize();
    checkAclError("Error: AscendCL error when finalize!", state);
}

void FastllmAclSetDevice(int32_t device_id) {
    //aclError state = aclrtSetDevice(device_id);
    aclError state;
    if (curDeviceId == device_id)
        return;
    if (curDeviceId != -1 && device_id != -1 && aclInitialized) {
        state = aclrtSynchronizeStream(FastllmAclStreamMap[curDeviceId]);
        checkAclError("Error: AscendCL error when Synchronize stream!", state);
    }
    if (device_id == -1)
        device_id = curDeviceId;
    aclrtContext context = getFastllmAclContextHandle(device_id);
    state = aclrtSetCurrentContext(context);
    checkAclError("Error: AscendCL error when set device!", state);
    aclrtStream stream = getFastllmAclStream(device_id);
    curDeviceId = device_id;
    aclInitialized = true;
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
    EnsureAclContext();
    aclError state = aclrtMalloc(&ret, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ACL_SUCCESS != state) {
        printf("Error: AscendCL error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkAclError("", state);
        return nullptr;
    }
    return ret;
}

void FastllmAclDirectFree(void *ret) {
    EnsureAclContext();
    aclError state = aclrtFree(ret);
    //checkAclError("Error: AscendCL error when release memory!", state);
}

void * FastllmAclMalloc(size_t size) {
    int32_t id = -1;
    aclError state = ACL_SUCCESS;
    EnsureAclContext();
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
    EnsureAclContext();
    for (auto &it: buffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &buffers = it.second;
            std::vector <AscendMemoryBuffer> temp;
            for (int i = 0; i < buffers.size(); i++) {
                if (!buffers[i].busy) {
                    FastllmAclSetDevice(it.first);
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
    EnsureAclContext();
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
    EnsureAclContext();
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
                FastllmAclSetDevice(it.first);
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
    FastllmAclSetDevice(id);
}


void FastllmAclClearBuffer() {
    EnsureAclContext();
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
                FastllmAclSetDevice(it.first);
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
    FastllmAclSetDevice(id);
}

int FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size) {
    EnsureAclContext();
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError("Error: AscendCL error when copy from memory to NPU!", state);
    return (int) state;
}

int FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    EnsureAclContext();
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError("Error: AscendCL error when copy from NPU to memory!", state);
    return (int) state;
}

void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    EnsureAclContext();
    aclError state = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    checkAclError("Error: AscendCL error when copy on NPU!", state);
}

void FastllmAclMemcpyBetweenDevices(int32_t dstId, void *dst, int32_t srcId, void *src, size_t size) {
    int canAccessPeer = 0;
    aclError state = aclrtDeviceCanAccessPeer(&canAccessPeer, srcId, dstId);
    if (canAccessPeer && state == ACL_SUCCESS) {
        state = aclrtDeviceEnablePeerAccess(srcId, 0);
        FastllmAclSetDevice(srcId);
        state = aclrtDeviceEnablePeerAccess(dstId, 0);
        FastllmAclSetDevice(dstId);
        checkAclError("Error: AscendCL error when enabling peer access!", state);
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


std::map<DataType, aclDataType> dataTypes = {
        {DataType::FLOAT32, aclDataType::ACL_FLOAT},
        {DataType::FLOAT16, aclDataType::ACL_FLOAT16},
        {DataType::INT8, aclDataType::ACL_INT8},
        {DataType::INT32PARAM, aclDataType::ACL_INT32},
        {DataType::INT16, aclDataType::ACL_INT16},
        {DataType::BIT, aclDataType::ACL_BOOL}
    };

void FastllmAclToTensor(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
                        std::vector<aclDataBuffer *> &buffers) {
    std::vector<int64_t> expandDims64(data.second->dims.size(), 0);
    std::vector<int> *dimensions = &data.second->dims;
    if (data.second->expansionDims.size() == data.second->dims.size())
        dimensions = &data.second->expansionDims;
    for (int i = 0; i < data.second->dims.size(); i++) {
        expandDims64[i] = (int64_t) (*dimensions)[i];
    }
    aclTensorDesc* tensor = aclCreateTensorDesc(dataTypes[data.second->dataType], expandDims64.size(),
                                       expandDims64.data(), aclFormat::ACL_FORMAT_ND);
    // printf("%s: type: %d dims: %zu : ", data.first.c_str(), dataTypes[data.second->dataType], expandDims64.size());
    // for (int j = 0; j < expandDims64.size(); j++)
    //     printf("%ld ", expandDims64[j]);
    aclSetTensorDescName(tensor, data.first.c_str());
    tensors.emplace_back(tensor);
    uint64_t size = data.second->expansionBytes == 0 ? (data.second->Count(0) * data.second->unitSize - 1) / data.second->unitSizeDiv + 1 : data.second->expansionBytes;
    // printf(" %ld\n", size);
    aclDataBuffer* buffer = aclCreateDataBuffer(data.second->deviceData, size);
    buffers.emplace_back(buffer);
}

void FastllmAclCreateShape(const std::pair<std::string, Data*> &data, std::vector<aclTensorDesc *> &tensors,
                           std::vector<int> dynamicDimension, std::vector<std::vector<int64_t>> dynamicRange) {
    std::vector<int> *dimensions = &data.second->dims;
    if (data.second->expansionDims.size() == data.second->dims.size())
        dimensions = &data.second->expansionDims;
    std::vector<int64_t> expandDims64(dimensions->size(), 0);
    int64_t (* range_info)[ACL_TENSOR_SHAPE_RANGE_NUM] = new int64_t[dimensions->size()][ACL_TENSOR_SHAPE_RANGE_NUM];
    int range_pos = 0;
    for (int i = 0; i < data.second->dims.size(); i++) {
        if (range_pos < dynamicDimension.size() && i == dynamicDimension[range_pos]) {
            expandDims64[i] = -1L;
            range_info[i][0] = dynamicRange[range_pos][0], range_info[i][1] = dynamicRange[range_pos][1];
            ++range_pos;
        } else {
            expandDims64[i] = (int64_t) (*dimensions)[i];
            range_info[i][0] = range_info[i][1] = expandDims64[i];
        }
    }
    // printf("%s: type: %d dims: %zu : ", data.first.c_str(), dataTypes[data.second->dataType], expandDims64.size());
    // for (int j = 0; j < expandDims64.size(); j++)
    //     printf("%ld ", expandDims64[j]);
    // printf("\n");
    aclTensorDesc* tensor = aclCreateTensorDesc(dataTypes[data.second->dataType], expandDims64.size(),
                                       expandDims64.data(), aclFormat::ACL_FORMAT_ND);
    aclSetTensorDescName(tensor, data.first.c_str());
    if (dynamicDimension.size() > 0) {
        aclError state = aclSetTensorShapeRange(tensor, dimensions->size(), range_info);
        checkAclError("Error: AscendCL tensor dynamic shape setting error. ", state);
    } else {
        delete[] range_info;
    }
    tensors.emplace_back(tensor);
}

void FastllmAclToOpAttribute(const FloatDict &floatParams, const IntDict &intParams,
                             const std::map <std::string, bool> &boolParams, aclopAttr **opAttr) {
    *opAttr = aclopCreateAttr();
    aclError state = ACL_SUCCESS;
    for (auto & it : intParams)
        state = aclopSetAttrInt(*opAttr, it.first.c_str(), it.second);
    for (auto & it : floatParams)
        state = aclopSetAttrFloat(*opAttr, it.first.c_str(), it.second);
    for (auto & it : boolParams)
        state = aclopSetAttrBool(*opAttr, it.first.c_str(), it.second);
    checkAclError("Error: AscendCL operator attribute setting error. ", state);
}

void FastllmAclDestroyShape(std::vector<aclTensorDesc *> &tensorShapes) {
    for (size_t i = 0; i < tensorShapes.size(); ++i) {
        aclDestroyTensorDesc(tensorShapes[i]);
    }
    tensorShapes.clear();
}

void FastllmAclDestoryTensors(std::vector<aclTensorDesc *> &inputTensors, std::vector<aclDataBuffer *> &inputBuffers,
                              std::vector<aclTensorDesc *> &outputTensors, std::vector<aclDataBuffer *> &outputBuffers, aclopAttr **opAttr) {
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        aclDestroyTensorDesc(inputTensors[i]);
    }
    inputTensors.clear();
    for (size_t i = 0; i < inputBuffers.size(); ++i) {
        aclDestroyDataBuffer(inputBuffers[i]);
    }
    inputBuffers.clear();
    for (size_t i = 0; i < outputTensors.size(); ++i) {
        aclDestroyTensorDesc(outputTensors[i]);
    }
    outputTensors.clear();
    for (size_t i = 0; i < outputBuffers.size(); ++i) {
        aclDestroyDataBuffer(outputBuffers[i]);
    }
    outputBuffers.clear();
    aclopDestroyAttr(const_cast<const aclopAttr *>(*opAttr));
}

bool FastllmAclInitOp(std::string name, std::vector<aclTensorDesc *> &inputTensorShapes,
                      std::vector<aclTensorDesc *> &outputTensorShapes, aclopAttr *opAttr) {
    aclError state = aclopCompile(name.c_str(),
                                  inputTensorShapes.size(),
                                  inputTensorShapes.data(),
                                  outputTensorShapes.size(),
                                  outputTensorShapes.data(),
                                  opAttr,
                                  ACL_ENGINE_SYS,
                                  ACL_COMPILE_SYS,
                                  nullptr);
    checkAclErrorFormat("Error: AscendCL error: compile op [%s] failed.", state, name.c_str());
    return (state == ACL_SUCCESS);
}

bool FastllmAclExecuteAfterInit(std::string name, std::vector<aclTensorDesc *> &inputTensors,
                                std::vector<aclDataBuffer *> &inputBuffers,
                                std::vector<aclTensorDesc *> &outputTensors,
                                std::vector<aclDataBuffer *> &outputBuffers, aclopAttr *opAttr) {
    aclError state = ACL_SUCCESS;
    EnsureAclContext();
    aclrtStream stream = getFastllmAclStream(curDeviceId);
    state = aclopExecuteV2(name.c_str(),
                           inputTensors.size(),
                           inputTensors.data(),
                           inputBuffers.data(),
                           outputTensors.size(),
                           outputTensors.data(),
                           outputBuffers.data(),
                           opAttr, stream);
    if (state == ACL_ERROR_OP_TYPE_NOT_MATCH || state == ACL_ERROR_OP_INPUT_NOT_MATCH ||
        state == ACL_ERROR_OP_OUTPUT_NOT_MATCH || state == ACL_ERROR_OP_ATTR_NOT_MATCH) {
        printf("Error: AscendCL op [%s] with the given description is not compiled. Please run atc first", name.c_str());
        checkAclError("", state);
        return false;
    }
    checkAclErrorFormat("Error: AscendCL error: execute op [%s] failed.", state, name.c_str());
    if (state != ACL_SUCCESS)
        return false;
    state = aclrtSynchronizeStream(stream);
    checkAclError("Error: AscendCL error when synchronize stream.", state);
    return (state == ACL_SUCCESS);
}

bool FastllmAclExecute(std::string name, std::vector<aclTensorDesc *> &inputTensors,
                       std::vector<aclDataBuffer *> &inputBuffers,
                       std::vector<aclTensorDesc *> &outputTensors,
                       std::vector<aclDataBuffer *> &outputBuffers, aclopAttr* opAttr) {
    aclError state = ACL_SUCCESS;
    EnsureAclContext();
    aclrtStream stream = getFastllmAclStream(curDeviceId);
    state = aclopCompileAndExecuteV2(name.c_str(),
                                     inputTensors.size(),
                                     inputTensors.data(),
                                     inputBuffers.data(),
                                     outputTensors.size(),
                                     outputTensors.data(),
                                     outputBuffers.data(),
                                     opAttr,
                                     ACL_ENGINE_SYS,
                                     ACL_COMPILE_SYS,
                                     nullptr,
                                     stream);
    checkAclErrorFormat("Error: AscendCL error when comnpile & execute op [%s].", state, name.c_str());
    state = aclrtSynchronizeStream(stream);
    checkAclError("Error: AscendCL error when synchronize stream.", state);
    return (state == ACL_SUCCESS);
}

}
}
