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

#include "devices/cpu/alivethreadpool.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"

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
    // MI50 的 2D D2D 拷贝会导致数据错误
    if (type == cudaMemcpyDeviceToDevice && srcDeviceId != dstDeviceId) {
        std::vector<uint8_t> hostBuffer(height * width);
        
        cudaSetDevice(srcDeviceId);
        cudaMemcpy2D(hostBuffer.data(), width, src, spitch, width, height, cudaMemcpyDeviceToHost);
        
        cudaSetDevice(dstDeviceId);
        cudaMemcpy2D(dst, dpitch, hostBuffer.data(), width, width, height, cudaMemcpyHostToDevice);
        
        cudaDeviceSynchronize();
        return;
    }
#endif

    auto state = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, type);
    
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
            curRatio = multiCudaCurrentRatios[i];
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

void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData) {
    if (data.multiDeviceData) {
        return;
    }
    data.multiDeviceData = true;
    int oriId = FastllmCudaGetDevice();

    if (copyData) {
        data.ToDevice(fastllm::DataDevice::CPU);
        for (int device : devices) {
            int mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

            data.multiDeviceDatas[device] = new fastllm::Data();
            data.multiDeviceDatas[device]->CopyFrom(data);
            data.multiDeviceDatas[device]->ToDevice(dataDevice);

            data.multiDeviceDatas[device]->group = data.group;
            data.multiDeviceDatas[device]->groupCnt = data.groupCnt;
            data.multiDeviceDatas[device]->scales = data.scales;
            data.multiDeviceDatas[device]->mins = data.mins;
            data.multiDeviceDatas[device]->zeros = data.zeros;
            data.multiDeviceDatas[device]->halfScales = data.halfScales;
        }
    } else {
        for (int device : devices) {
            int mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);
            if (data.dims.size() == 0) {
                data.multiDeviceDatas[device] = new fastllm::Data(data.dataType);    
            } else {
                data.multiDeviceDatas[device] = new fastllm::Data(data.dataType, data.dims);
            }
            data.multiDeviceDatas[device]->dataDevice = dataDevice;
        }
    }
    FastllmCudaSetDevice(oriId);
}

bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
                    std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis) {
    int deviceNum = multiCudaCurrentDevices.size();
    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                }
            }
        }
    }
    cudaSetDevice(0);

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
            weight.multiDeviceDatas[deviceId] = new fastllm::Data(weight.dataType, {len, m});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId] = new fastllm::Data(bias.dataType, {len});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->Allocate();
            bias.multiDeviceDatas[deviceId]->Allocate();

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy((uint8_t*)deviceWeightData + curLen * m * weight.unitSize / weight.unitSizeDiv, 
                                    (uint8_t*)weight.cudaData + it.first * m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * m * weight.unitSize / weight.unitSizeDiv, GetCudaMemcpyType(mallocType, 1));
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
        } else {
            weight.multiDeviceDatas[deviceId] = new fastllm::Data(weight.dataType, {k, len});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId] = new fastllm::Data(bias.dataType, {k});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->Allocate();
            bias.multiDeviceDatas[deviceId]->Allocate();

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);

            int curLen = 0;
            for (auto &it : div) {
                if (mallocType == 0) {
                    cudaSetDevice(0);
                }
                FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + curLen * weight.unitSize / weight.unitSizeDiv,
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    (uint8_t*)weight.cudaData + it.first * weight.unitSize / weight.unitSizeDiv, 
                                    m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    k, GetCudaMemcpyType(mallocType, 1), deviceId, 0);
                curLen += (it.second - it.first);
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

    cudaSetDevice(0);
    FastllmCudaFree(weight.cudaData);
    FastllmCudaFree(cudaBiasData);
    weight.cudaData = nullptr;
    weight.weightSum.clear();
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