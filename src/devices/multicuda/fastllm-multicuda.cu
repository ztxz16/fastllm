//
// Created by huangyuyang on 8/2/24.
//

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"
#include "fastllm.h"
#include "utils.h"
#include "gguf.h"

#include "devices/cpu/alivethreadpool.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"

#include <cuda_runtime.h>
#include <nccl.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);

void FastllmCudaMemcpy2D(void* dst, size_t dpitch, const void* src,
    size_t spitch, size_t width, size_t height, cudaMemcpyKind type, 
    int dstDeviceId, int srcDeviceId) {
#if defined(USE_ROCM) && defined(USE_MI50_WORKAROUND)
    // MI50 的跨卡 2D D2D 拷贝可能出错，保留旧的 host staging 兼容路径。
    if (type == cudaMemcpyDeviceToDevice && srcDeviceId != dstDeviceId) {
        std::vector <uint8_t> hostBuffer(height * width);
        cudaSetDevice(srcDeviceId);
        cudaMemcpy2D(hostBuffer.data(), width, src, spitch, width, height, cudaMemcpyDeviceToHost);

        cudaSetDevice(dstDeviceId);
        cudaMemcpy2D(dst, dpitch, hostBuffer.data(), width, width, height, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        return;
    }
#endif

    cudaError_t state = cudaSuccess;
    if (type == cudaMemcpyDeviceToDevice) {
        cudaSetDevice(dstDeviceId);
    }
    state = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, type);

#ifdef USE_ROCM
    cudaDeviceSynchronize();
#endif

    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when memcpy2D!", state);
    }
}

void FastllmCudaMemcpy2DDeviceToDeviceAuto(void * 	dst, size_t 	dpitch, const void * 	src,
    size_t 	spitch, size_t 	width, size_t 	height, int dstDeviceId, int srcDeviceId) {
    FastllmCudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, dstDeviceId, srcDeviceId);
}

void FastllmCudaSyncDevice(int deviceId) {
    cudaSetDevice(deviceId);
    cudaError_t state = cudaDeviceSynchronize();
    checkCudaErrors("Error: CUDA sync failed!", state);
}

std::map <int, std::string> specialDeviceIds = {
    {99999, "cpu"}
};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType) {
    specialId = "";
    if (specialDeviceIds.find(deviceId) == specialDeviceIds.end()) {
        cudaSetDevice(deviceId);
    } else {
        specialId = specialDeviceIds[deviceId];
    }
    mallocType = 1;
    if (specialId == "cpu") {
        mallocType = 0;
    }
}

/*
type: device type (0 for cpu, 1 for cuda)
*/
void *AutoMalloc(size_t size, int type) {
    if (type == 0) {
        return (void*)(new uint8_t[size]);
    } else {
        return (void*)FastllmCudaMalloc(size);
    }
}

cudaError_t AutoMemset(void *a, int value, size_t size, int type) {
    if (type == 0) {
        memset(a, value, size);
        return cudaSuccess;
    } else {
        return cudaMemset(a, value, size);
    }
}

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType) {
    if (srcType == 0) {
        if (dstType == 0) {
            return cudaMemcpyHostToHost;
        } else {
            return cudaMemcpyHostToDevice;
        }
    } else {
        if (dstType == 0) {
            return cudaMemcpyDeviceToHost;
        } else {
            return cudaMemcpyDeviceToDevice;
        }
    }
}

std::vector <int> multiCudaCurrentDevices;
std::map <int, int> multiCudaCurrentRatios;

void FastllmMultiCudaSetDevice(std::vector <int> ids) {
    multiCudaCurrentDevices = ids;
}

void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio) {
    multiCudaCurrentRatios = deviceRatio;
}

void FastllmGetMulticudaDeviceAndRatio(std::vector <int> &devices, std::map <int, int> &ratios, bool noSpecial) {
    devices.clear();
    ratios.clear();
    for (int i : multiCudaCurrentDevices) {
        if (noSpecial == false || specialDeviceIds.find(i) == specialDeviceIds.end()) {
            devices.push_back(i);
            ratios[i] = multiCudaCurrentRatios.find(i) != multiCudaCurrentRatios.end() ? multiCudaCurrentRatios[i] : 1;
        }
    }
}

// 将total个计算任务切分
// 若当前有x个设备，返回一个长度为(x + 1)的vector，第i个设备执行任务[ret[i], ret[i + 1])
std::vector <int> FastllmMultiCudaGetSplitPoints(std::vector <int> &multiCudaCurrentDevices, 
                                std::map <int, int> &multiCudaCurrentRatios, int total, int unit = 1) {
    int deviceNum = multiCudaCurrentDevices.size();
    unit = std::max(1, total > 0 ? std::min(unit, total) : unit);
    int nodes = total / unit;
    int totalRatio = 0;
    if (multiCudaCurrentRatios.size() > 0) {
        for (auto &it : multiCudaCurrentRatios) {
            totalRatio += it.second;
        }
    } else {
        totalRatio = deviceNum;
    }
    std::vector <int> ret;
    int cur = 0;
    for (int i = 0; i < deviceNum; i++) {
        int curRatio = 1;
        if (multiCudaCurrentRatios.find(multiCudaCurrentDevices[i]) != multiCudaCurrentRatios.end()) {
            curRatio = multiCudaCurrentRatios[multiCudaCurrentDevices[i]];
        }
        int now = std::max(1, nodes * curRatio / totalRatio) * unit;
        int end = (i == deviceNum - 1 ? total : cur + now);
        ret.push_back(cur);
        if (i == deviceNum - 1) {
            ret.push_back(end);
        }
        cur = end;
    }
    return ret;
}

static int GcdInt(int a, int b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

static int LcmInt(int a, int b) {
    a = std::max(1, a);
    b = std::max(1, b);
    return a / GcdInt(a, b) * b;
}

static bool IsGGUFTensor(const fastllm::Data &data) {
    return data.dataType == fastllm::DataType::DATA_GGUF_FORMAT && data.ggmlType >= 0;
}

static int GetGGUFBlockSize(const fastllm::Data &data) {
    return IsGGUFTensor(data) ? (int)ggml_blck_size((ggml_type)data.ggmlType) : 1;
}

static size_t GetGGUFRowBytes(const fastllm::Data &data, int columns) {
    int blockSize = GetGGUFBlockSize(data);
    fastllm::AssertInFastLLM(blockSize > 0 && columns % blockSize == 0,
                             "GGUF split requires aligned columns, got " + std::to_string(columns) +
                             " for block size " + std::to_string(blockSize) + ".\n");
    return ggml_row_size((ggml_type)data.ggmlType, columns);
}

static int GetMultiCudaSplitUnit(const fastllm::Data &data) {
    int unit = data.groupCnt <= 0 ? 128 : data.groupCnt;
    if (data.dataType == fastllm::DataType::FP8_E4M3) {
        unit = data.blockM;
    }
    if (IsGGUFTensor(data)) {
        unit = LcmInt(unit, GetGGUFBlockSize(data));
    }
    return std::max(1, unit);
}

static int GetGGUFHeadSplitUnit(const fastllm::Data &data, int widthPerHead) {
    if (!IsGGUFTensor(data)) {
        return 1;
    }
    int blockSize = GetGGUFBlockSize(data);
    int gcd = GcdInt(blockSize, std::max(1, widthPerHead));
    return std::max(1, blockSize / gcd);
}

static void InitMultiCudaLocalTensorMeta(const fastllm::Data &src, fastllm::Data &dst) {
    dst.name = src.name;
    dst.isModelWeight = src.isModelWeight;
    dst.group = src.group;
    dst.groupCnt = src.groupCnt;
    dst.blockK = src.blockK;
    dst.blockM = src.blockM;
    dst.perChannelAxis = src.perChannelAxis;
    dst.isGGUFData = src.isGGUFData || src.dataType == fastllm::DataType::DATA_GGUF_FORMAT;
    dst.ggmlType = src.ggmlType;
    dst.IsRepacked = src.IsRepacked;
    dst.tpLinearType = src.tpLinearType;
    dst.tpPackType = src.tpPackType;
    dst.tpQHeads = src.tpQHeads;
    dst.tpKVHeads = src.tpKVHeads;
    dst.tpHeadDim = src.tpHeadDim;
}

static fastllm::Data *CreateMultiCudaLocalTensor(const fastllm::Data &src, const std::vector <int> &dims) {
    fastllm::Data *local = nullptr;
    if (dims.empty()) {
        local = new fastllm::Data(src.dataType);
    } else if (src.dataType == fastllm::DataType::DATA_GGUF_FORMAT) {
        fastllm::AssertInFastLLM(src.ggmlType >= 0,
                                 "GGUF tensor \"" + src.name + "\" is missing ggmlType when creating multicuda local tensor.\n");
        local = new fastllm::Data(src.dataType, src.ggmlType, dims);
    } else {
        local = new fastllm::Data(src.dataType, dims);
    }
    InitMultiCudaLocalTensorMeta(src, *local);
    return local;
}

static void ClearGGUFExtraCudaCaches(fastllm::Data &data) {
#ifdef USE_CUDA
    if (!IsGGUFTensor(data)) {
        return;
    }
    for (void *ptr : data.extraCudaData) {
        if (ptr != nullptr) {
            FastllmCudaFree(ptr);
        }
    }
    for (void *ptr : data.extraCudaHalfData) {
        if (ptr != nullptr) {
            FastllmCudaFree(ptr);
        }
    }
#endif
    data.extraCudaData.clear();
    data.extraCudaHalfData.clear();
}

static void ResetMultiDeviceData(fastllm::Data &data) {
    if (!data.multiDeviceData) {
        data.ClearTensorParallelLayout();
        return;
    }
    for (auto &it : data.multiDeviceDatas) {
        delete it.second;
    }
    data.multiDeviceDatas.clear();
    data.multiDeviceData = false;
    data.ClearTensorParallelLayout();
}

void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData) {
    if (data.multiDeviceData) {
        return;
    }
    data.multiDeviceData = true;
    int oriId = FastllmCudaGetDevice();

    if (copyData) {
        bool srcOnGpu = (data.dataDevice == fastllm::DataDevice::CUDA
                         && data.cudaData != nullptr
                         && !data.dataDeviceIds.empty());
        int srcDevice = srcOnGpu ? data.dataDeviceIds[0] : -1;

        if (srcOnGpu) {
            uint64_t bytes = data.GetBytes();
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                auto *local = CreateMultiCudaLocalTensor(data, data.dims);
                local->dataDevice = fastllm::DataDevice::CUDA;
                local->dataDeviceIds = {device};
                local->Allocate();

                if (device == srcDevice) {
                    cudaMemcpy(local->cudaData, data.cudaData, bytes, cudaMemcpyDeviceToDevice);
                } else {
                    cudaMemcpyPeer(local->cudaData, device, data.cudaData, srcDevice, bytes);
                }

                local->group = data.group;
                local->groupCnt = data.groupCnt;
                local->scales = data.scales;
                local->mins = data.mins;
                local->zeros = data.zeros;
                local->halfScales = data.halfScales;
                data.multiDeviceDatas[device] = local;
            }
        } else {
            data.ToDevice(fastllm::DataDevice::CPU);
            for (int device : devices) {
                int mallocType = 0;
                std::string specialId = "";
                SwitchDeviceAndGetInfos(device, specialId, mallocType);
                fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

                data.multiDeviceDatas[device] = new fastllm::Data();
                data.multiDeviceDatas[device]->CopyFrom(data);
                InitMultiCudaLocalTensorMeta(data, *data.multiDeviceDatas[device]);
                data.multiDeviceDatas[device]->ToDevice(dataDevice, std::vector <int> {device});
                data.multiDeviceDatas[device]->dataDeviceIds = {device};

                data.multiDeviceDatas[device]->group = data.group;
                data.multiDeviceDatas[device]->groupCnt = data.groupCnt;
                data.multiDeviceDatas[device]->scales = data.scales;
                data.multiDeviceDatas[device]->mins = data.mins;
                data.multiDeviceDatas[device]->zeros = data.zeros;
                data.multiDeviceDatas[device]->halfScales = data.halfScales;
            }
        }
    } else {
        for (int device : devices) {
            int mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);
            data.multiDeviceDatas[device] = CreateMultiCudaLocalTensor(data, data.dims);
            data.multiDeviceDatas[device]->dataDevice = dataDevice;
            data.multiDeviceDatas[device]->dataDeviceIds = {device};
        }
    }
    FastllmCudaSetDevice(oriId);
}

void PrepareMultiCudaReplicatedData(fastllm::Data &data, std::vector <int> devices, bool copyData) {
    if (data.IsTensorParallelReplicated() && data.multiDeviceData) {
        return;
    }
    auto oriDevice = data.dataDevice;
    auto oriDeviceIds = data.dataDeviceIds;
    ResetMultiDeviceData(data);
    CopyToMultiDevices(data, devices, copyData);
    if (copyData && oriDevice == fastllm::DataDevice::CUDA) {
        data.ToDevice(oriDevice, oriDeviceIds, true);
    }
    data.tpLayout = fastllm::TP_LAYOUT_REPLICATED;
    data.tpAxis = -1;
    data.tpGlobalDims = data.dims;
}

void PrepareMultiCudaShardedData(fastllm::Data &data, std::vector <int> devices,
    const std::vector <int> &globalDims, int axis, DivisionScheme divisionScheme) {
    ResetMultiDeviceData(data);
    data.multiDeviceData = true;
    data.tpLayout = fastllm::TP_LAYOUT_SHARDED;
    data.tpAxis = axis;
    data.tpGlobalDims = globalDims;
    data.tpRanges = divisionScheme;
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = devices;
    data.Resize(globalDims);
    int oriId = FastllmCudaGetDevice();
    for (int device : devices) {
        int mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(device, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU : fastllm::DataDevice::CUDA);
        std::vector <int> localDims = globalDims;
        int len = 0;
        for (auto &range : divisionScheme[device]) {
            len += range.second - range.first;
        }
        localDims[axis] = len;
        data.multiDeviceDatas[device] = CreateMultiCudaLocalTensor(data, localDims);
        data.multiDeviceDatas[device]->dataDevice = dataDevice;
        data.multiDeviceDatas[device]->dataDeviceIds = {device};
    }
    data.cudaData = nullptr;
    FastllmCudaSetDevice(oriId);
}

DivisionScheme BuildMultiCudaRowSplitScheme(fastllm::Data &weight, std::vector <int> &devices, std::map <int, int> &ratios) {
    DivisionScheme divisionScheme;
    if (weight.tpPackType == fastllm::TP_PACK_QKV) {
        int kvHeads = weight.tpKVHeads;
        int qHeads = weight.tpQHeads;
        int headDim = weight.tpHeadDim;
        int group = qHeads / kvHeads;
        int qWidth = qHeads * headDim;
        int kvWidth = kvHeads * headDim;
        int headUnit = GetGGUFHeadSplitUnit(weight, group * headDim);
        fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || kvHeads % headUnit == 0,
                                 "GGUF QKV tensor parallel requires kv heads aligned to head unit " +
                                 std::to_string(headUnit) + ".\n");
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvHeads, headUnit);
        for (int i = 0; i < devices.size(); i++) {
            int deviceId = devices[i];
            int st = points[i], end = points[i + 1];
            divisionScheme[deviceId].push_back({st * group * headDim, end * group * headDim});
            divisionScheme[deviceId].push_back({qWidth + st * headDim, qWidth + end * headDim});
            divisionScheme[deviceId].push_back({qWidth + kvWidth + st * headDim, qWidth + kvWidth + end * headDim});
        }
        return divisionScheme;
    }

    if (weight.tpPackType == fastllm::TP_PACK_GATEUP) {
        int mid = weight.dims[0] / 2;
        int unit = GetMultiCudaSplitUnit(weight);
        fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || mid % unit == 0,
                                 "GGUF GateUp tensor parallel requires aligned split unit " +
                                 std::to_string(unit) + ".\n");
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, mid, unit);
        for (int i = 0; i < devices.size(); i++) {
            int deviceId = devices[i];
            int st = points[i], end = points[i + 1];
            divisionScheme[deviceId].push_back({st, end});
            divisionScheme[deviceId].push_back({mid + st, mid + end});
        }
        return divisionScheme;
    }

    int unit = GetMultiCudaSplitUnit(weight);
    fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || weight.dims[0] % unit == 0,
                             "GGUF row split requires aligned split unit " + std::to_string(unit) + ".\n");
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, weight.dims[0], unit);
    for (int i = 0; i < devices.size(); i++) {
        divisionScheme[devices[i]].push_back({points[i], points[i + 1]});
    }
    return divisionScheme;
}

bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
                    std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis) {
    int deviceNum = multiCudaCurrentDevices.size();
    int rootDevice = deviceNum > 0 ? multiCudaCurrentDevices[0] : 0;
    fastllm::AssertInFastLLM(weight.dataType != fastllm::DataType::DATA_GGUF_FORMAT || weight.ggmlType >= 0,
                             "GGUF weight \"" + weight.name + "\" is missing ggmlType before multicuda split.\n");
    if (weight.dataDevice != fastllm::DataDevice::CUDA || weight.cudaData == nullptr ||
        weight.dataDeviceIds.size() == 0 || weight.dataDeviceIds[0] != rootDevice) {
        weight.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
    }
    if (bias.dims.size() > 0 &&
        (bias.dataDevice != fastllm::DataDevice::CUDA || bias.cudaData == nullptr ||
         bias.dataDeviceIds.size() == 0 || bias.dataDeviceIds[0] != rootDevice)) {
        bias.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
    }

    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    int canPeerAccess = 0;
                    cudaError_t peerState = cudaDeviceCanAccessPeer(&canPeerAccess,
                                                                    multiCudaCurrentDevices[i],
                                                                    multiCudaCurrentDevices[j]);
                    if (peerState == cudaSuccess && canPeerAccess) {
                        peerState = cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                        if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
            }
        }
    }
    cudaSetDevice(rootDevice);

    if (weight.multiDeviceData) {
        return true;
    }
    weight.multiDeviceData = true;
    bias.multiDeviceData = true;
    int k = weight.dims[0], m = weight.dims[1];
    cudaError_t state = cudaSuccess;
    float *cudaBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
    if (bias.dims.size() > 0) {
        state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        if (splitAxis == 0) {
            weight.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(weight, {len, m});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(bias, {len});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            weight.multiDeviceDatas[deviceId]->Allocate();
            bias.multiDeviceDatas[deviceId]->Allocate();

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);
            int curLen = 0;
            if (IsGGUFTensor(weight)) {
                size_t rowBytes = GetGGUFRowBytes(weight, m);
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    state = cudaMemcpy((uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                                       (uint8_t*)weight.cudaData + (size_t)it.first * rowBytes,
                                       (size_t)copyLen * rowBytes,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                       (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    curLen += copyLen;
                }
            } else {
                for (auto &it : div) {
                    state = cudaMemcpy((uint8_t*)deviceWeightData + curLen * m * weight.unitSize / weight.unitSizeDiv, 
                                        (uint8_t*)weight.cudaData + it.first * m * weight.unitSize / weight.unitSizeDiv, 
                                        (it.second - it.first) * m * weight.unitSize / weight.unitSizeDiv, GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    curLen += (it.second - it.first);
                }
            }
        } else {
            weight.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(weight, {k, len});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(bias, {k});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            weight.multiDeviceDatas[deviceId]->Allocate();
            bias.multiDeviceDatas[deviceId]->Allocate();

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);

            int curLen = 0;
            if (IsGGUFTensor(weight)) {
                size_t srcRowBytes = GetGGUFRowBytes(weight, m);
                size_t dstRowBytes = GetGGUFRowBytes(*weight.multiDeviceDatas[deviceId], len);
                size_t dstOffsetBytes = 0;
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    size_t srcOffsetBytes = GetGGUFRowBytes(weight, it.first);
                    size_t copyBytes = GetGGUFRowBytes(weight, copyLen);
                    FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstOffsetBytes,
                                        dstRowBytes,
                                        (uint8_t*)weight.cudaData + srcOffsetBytes,
                                        srcRowBytes,
                                        copyBytes,
                                        k, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    dstOffsetBytes += copyBytes;
                    curLen += copyLen;
                }
            } else {
                for (auto &it : div) {
                    if (mallocType == 0) {
                        cudaSetDevice(rootDevice);
                    }
                    FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + curLen * weight.unitSize / weight.unitSizeDiv,
                                        (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                        (uint8_t*)weight.cudaData + it.first * weight.unitSize / weight.unitSizeDiv, 
                                        m * weight.unitSize / weight.unitSizeDiv, 
                                        (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                        k, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    curLen += (it.second - it.first);
                }
            }
            if (i == 0) {
                state = cudaMemcpy(deviceBiasData, cudaBiasData, k * sizeof(float), GetCudaMemcpyType(mallocType, 1));
            } else {
                state = AutoMemset(deviceBiasData, 0, k * sizeof(float), mallocType);
            }
        }

        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }

    if (weight.dataType == fastllm::DataType::FP8_E4M3) {
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
            
            auto &div = divisionScheme[deviceId];
            int len = 0;
            for (auto &it : div) {
                len += it.second - it.first;
            }

            float *cudaScales;
            float *cudaMins;
            uint8_t *cudaZeropoints;
            auto curDevice = weight.multiDeviceDatas[deviceId];
            curDevice->blockK = weight.blockK;
            curDevice->blockM = weight.blockM;
            int ks = (curDevice->dims[0] - 1) / curDevice->blockK + 1;
            int ms = (curDevice->dims[1] - 1) / curDevice->blockM + 1;
            curDevice->scales.resize(ks * ms);
            if (splitAxis == 0) {
                int curLen = 0;
                for (auto &it : div) {
                    memcpy(curDevice->scales.data() + curLen * ms, weight.scales.data() + it.first / curDevice->blockM * ms, 
                        (it.second - it.first) / curDevice->blockM * ms * sizeof(float));
                    curLen += (it.second - it.first) / curDevice->blockM;
                }
            } else {
                int oriMs = weight.scales.size() / ks;
                for (int i = 0; i < ks; i++) {
                    int curLen = 0;
                    for (auto &it : div) {
                        memcpy(curDevice->scales.data() + i * ms, weight.scales.data() + i * oriMs + it.first / curDevice->blockM, 
                            (it.second - it.first) / curDevice->blockM * sizeof(float));
                        curLen += (it.second - it.first) / curDevice->blockM;
                    }   
                }
            }
        }
    } else {
        // 1. mins, scales
        if (weight.mins.size() > 0) {
            int weightGroup = weight.group < 0 ? 1 : weight.group;
            std::vector <int> zeropoints = std::vector <int> (k * weightGroup, 0);
            if (weight.perChannelsConfigs.size() > 0) {
                for (int i = 0; i < k * weightGroup; i++) {
                    zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
                }
            } else if (weight.zeros.size() > 0) {
                for (int i = 0; i < k * weightGroup; i++) {
                    zeropoints[i] = weight.zeros[i];
                }
            } else {
                for (int i = 0; i < k * weightGroup; i++) {
                    zeropoints[i] = 0;
                }
            }
            for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
                int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
                std::string specialId = "";
                SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
                
                auto &div = divisionScheme[deviceId];
                int len = 0;
                for (auto &it : div) {
                    len += it.second - it.first;
                }

                float *cudaScales;
                float *cudaMins;
                uint8_t *cudaZeropoints;
                auto curDevice = weight.multiDeviceDatas[deviceId];
                if (splitAxis == 0) {
                    curDevice->group = weight.group;
                    curDevice->groupCnt = weight.groupCnt;
                    curDevice->scales.resize(len * weightGroup);
                    curDevice->mins.resize(len * weightGroup);
                    if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                        int curLen = 0;
                        for (auto &it : div) {
                            memcpy(curDevice->scales.data() + curLen * weightGroup, weight.scales.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float));
                            memcpy(curDevice->mins.data() + curLen * weightGroup, weight.mins.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float));
                            curLen += (it.second - it.first);
                        }
                    } else {
                        curDevice->zeros.resize(len * weightGroup);
                        int curLen = 0;
                        for (auto &it : div) {
                            memcpy(curDevice->scales.data() + curLen * weightGroup, weight.scales.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float));
                            memcpy(curDevice->mins.data() + curLen * weightGroup, weight.mins.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float));
                            memcpy(curDevice->zeros.data() + curLen * weightGroup, zeropoints.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(int));
                            curLen += (it.second - it.first);
                        }
                    }
                } else {
                    curDevice->scales.resize(k * weightGroup);
                    curDevice->mins.resize(k * weightGroup);
                    curDevice->group = weight.group;
                    curDevice->groupCnt = weight.groupCnt;
                    if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                        int base = div[0].first / weight.groupCnt;
                        std::vector <float> scales, mins;
                        for (int i = 0; i < weight.scales.size(); i++) {
                            scales.push_back((i + base < weight.scales.size() ? weight.scales[i + base] : 0.0f));
                        }
                        for (int i = 0; i < weight.mins.size(); i++) {
                            mins.push_back((i + base < weight.mins.size() ? weight.mins[i + base] : 0.0f));
                        }
                        memcpy(curDevice->scales.data(), scales.data(), k * weightGroup * sizeof(float));
                        memcpy(curDevice->mins.data(), mins.data(), k * weightGroup * sizeof(float));
                    } else {
                        curDevice->zeros.resize(k * weightGroup);
                        memcpy(curDevice->scales.data(), weight.scales.data(), k * weightGroup * sizeof(float));
                        memcpy(curDevice->mins.data(), weight.mins.data(), k * weightGroup * sizeof(float));
                        memcpy(curDevice->zeros.data(), zeropoints.data(), k * weightGroup * sizeof(int));
                    }
                }
            }
        }
    }

    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(rootDevice);
    FastllmCudaFree(weight.cudaData);
    FastllmCudaFree(cudaBiasData);
    ClearGGUFExtraCudaCaches(weight);
    weight.cudaData = nullptr;
    weight.weightSum.clear();
    return true;
}

bool SplitMultiCudaWeight1D(fastllm::Data &bias, std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme) {
    int deviceNum = multiCudaCurrentDevices.size();
    int rootDevice = deviceNum > 0 ? multiCudaCurrentDevices[0] : 0;
    if (bias.dims.size() > 0 &&
        (bias.dataDevice != fastllm::DataDevice::CUDA || bias.cudaData == nullptr ||
         bias.dataDeviceIds.size() == 0 || bias.dataDeviceIds[0] != rootDevice)) {
        bias.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
    }
    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    int canPeerAccess = 0;
                    cudaError_t peerState = cudaDeviceCanAccessPeer(&canPeerAccess,
                                                                    multiCudaCurrentDevices[i],
                                                                    multiCudaCurrentDevices[j]);
                    if (peerState == cudaSuccess && canPeerAccess) {
                        peerState = cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                        if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
            }
        }
    }
    cudaSetDevice(0);

    if (bias.multiDeviceData) {
        return true;
    }
    bias.multiDeviceData = true;
    int k = bias.dims[0];
    cudaError_t state = cudaSuccess;
    float *cudaBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
    if (bias.dims.size() > 0) {
        state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        {
            bias.multiDeviceDatas[deviceId] = new fastllm::Data(bias.dataType, {len});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId]->name = bias.name;
            bias.multiDeviceDatas[deviceId]->isModelWeight = bias.isModelWeight;
            bias.multiDeviceDatas[deviceId]->Allocate();
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
        }

        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }
    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(0);
    FastllmCudaFree(cudaBiasData);
    return true;
}

std::vector <bool> streamInits = std::vector <bool> (4, 0);
cudaStream_t streams[4];

cudaStream_t *GetFastllmStream(int id) {
    if (!streamInits[id]) {
        streamInits[id] = true;
        cudaSetDevice(id);
        cudaStreamCreate(&streams[id]);
        cudaSetDevice(0);
    }
    return &streams[id];
}

// 全局变量存储通信器
// Key: deviceId, Value: ncclComm_t
static std::map<int, ncclComm_t> g_ncclComms;
static bool g_ncclInitialized = false;

bool FastllmInitNccl(const std::vector<int>& devices) {
    if (devices.empty()) return false;

    bool ready = g_ncclInitialized;
    if (ready) {
        for (int device : devices) {
            auto it = g_ncclComms.find(device);
            if (it == g_ncclComms.end() || it->second == nullptr) {
                ready = false;
                break;
            }
        }
    }
    if (ready) {
        return true;
    }

    for (auto &it : g_ncclComms) {
        if (it.second != nullptr) {
            ncclCommDestroy(it.second);
        }
    }
    g_ncclComms.clear();
    g_ncclInitialized = false;

    int numGPUs = devices.size();
    std::vector<ncclComm_t> comms(numGPUs);

    for (int device : devices) {
        cudaSetDevice(device);
        cudaFree(0);
    }

    // ncclCommInitAll 会在这一组设备之间建立通信域
    // 注意：这会阻塞，直到所有卡都就绪
    ncclResult_t initRes = ncclCommInitAll(comms.data(), numGPUs, devices.data());
    if (initRes != ncclSuccess) {
        printf("Error: ncclCommInitAll failed: %s\n", ncclGetErrorString(initRes));
        return false;
    }

    // 将生成的 comms 存入 map，方便后续通过 deviceId 查找
    for(int i = 0; i < numGPUs; ++i) {
        g_ncclComms[devices[i]] = comms[i];
    }
        
    g_ncclInitialized = true;
    printf("NCCL Initialized for %d devices.\n", numGPUs);
    return true;
}

ncclComm_t GetNcclComm(int deviceId) {
    if (g_ncclComms.find(deviceId) != g_ncclComms.end()) {
        return g_ncclComms[deviceId];
    }
    printf("Error: No NCCL comm found for device %d\n", deviceId);
    return nullptr;
}

// 功能：将 root 设备上的 data 数据广播到所有卡 (In-place 操作)
// data: 也就是发送/接收缓冲区的首地址
// count: 元素个数
// dataType: fastllm 数据类型
// root: 源数据的 deviceId
// deviceId: 当前调用线程所属的 deviceId
void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId) {
    // 1. 获取当前设备的通信器
    ncclComm_t comm = GetNcclComm(deviceId);
    if (comm == nullptr) {
        std::vector<int> devices;
        std::map<int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        FastllmInitNccl(devices);
        comm = GetNcclComm(deviceId);
    }
    if (comm == nullptr) {
        printf("Error: FastllmNcclBroadcast failed, comm is null for device %d\n", deviceId);
        return;
    }

    // 2. 映射数据类型
    ncclDataType_t ncclType = ncclFloat; 
    if (dataType == 7) { // float16
        ncclType = ncclHalf;
    } else if (dataType == 0) { // float32
        ncclType = ncclFloat;
    } else {
        printf("Error: Unknown dataType %d for NCCL Broadcast\n", dataType);
        return;
    }

    // 3. 执行 Broadcast
    // ncclBroadcast(sendbuff, recvbuff, ...);
    // 对于 In-place 操作，sendbuff 和 recvbuff 传同一个地址即可
    cudaStream_t stream = 0; // 使用默认流
    
    ncclResult_t res = ncclBroadcast(data, data, count, ncclType, root, comm, stream);
    
    if (res != ncclSuccess) {
        printf("Error: ncclBroadcast failed on device %d: %s\n", deviceId, ncclGetErrorString(res));
    }
}

// 功能：将所有卡上的 data 数据进行 Sum 求和，结果保存在 dest 中 (支持 in-place，即 data == dest)
void FastllmNcclAllReduce(void* data, void* dest, int count, int dataType, int deviceId) {
    // 1. 获取当前设备的通信器
    ncclComm_t comm = GetNcclComm(deviceId);
    if (comm == nullptr) {
        std::vector<int> devices;
        std::map<int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        FastllmInitNccl(devices);
        comm = GetNcclComm(deviceId);
    }
    if (comm == nullptr) {
        printf("Error: FastllmNcclAllReduce failed, comm is null for device %d\n", deviceId);
        return;
    }

    // 2. 映射数据类型 (Fastllm DataType -> NCCL type)
    // 假设 dataType: 0 -> float32, 1 -> float16/half (具体需根据 fastllm 的 DataType 枚举调整)
    ncclDataType_t ncclType = ncclFloat; 
    if (dataType == 7) { // 假设 1 代表 float16
        ncclType = ncclHalf;
    } else if (dataType == 0) { // 假设 0 代表 float32
        ncclType = ncclFloat;
    } else {
        // 如果有其他类型，需在此补充
        printf("Error: Unknown dataType %d for NCCL\n", dataType);
        return;
    }

    // 3. 执行 AllReduce
    // op: ncclSum (求和)
    // stream: 使用当前 CUDA流，通常 fastllm 默认使用 0 流。
    // 如果 fastllm 使用了特定流，需通过 cudaStream_t 参数传入。
    cudaStream_t stream = 0; 
    
    // 注意：NCCL调用是异步的
    ncclResult_t res = ncclAllReduce(data, dest, count, ncclType, ncclSum, comm, stream);
    
    if (res != ncclSuccess) {
        printf("Error: ncclAllReduce failed on device %d: %s\n", deviceId, ncclGetErrorString(res));
    }
}
