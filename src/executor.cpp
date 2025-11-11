//
// Created by huangyuyang on 6/13/23.
//

#include "utils.h"

#include "executor.h"

#include "devices/cpu/cpudevice.h"

#ifdef USE_CUDA
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/multicudadevice.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#ifdef USE_TFACC
#include "devices/tfacc/tfaccdevice.h"
#endif

#ifdef USE_TOPS
#include "devices/tops/topsdevice.h"
#endif

#ifdef USE_NUMA
#include "devices/numa/numadevice.h"
#endif

#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

namespace fastllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        if (FastllmCudaGetDeviceCount() > 0) {
            this->devices.push_back((BaseDevice*) new CudaDevice());
            this->devices.push_back((BaseDevice*) new MultiCudaDevice((CudaDevice*)this->devices.back()));
        }
#endif
#ifdef USE_TOPS
        this->devices.push_back((BaseDevice*) new TopsDevice());
#endif
#ifdef USE_TFACC
        this->devices.push_back((BaseDevice*) new TfaccDevice());
#endif
#ifdef USE_NUMA
        try {
            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
            if (s != "" && s != "OFF") {
                printf("ACTIVATE NUMA = ON\n");
                this->devices.push_back((BaseDevice*) new NumaDevice());
            }
        } catch (...) {
        }
#endif

#ifdef USE_NUMAS
        this->devices.push_back((BaseDevice*) new NumasDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::ClearDevices() {
        this->devices.clear();
    }

    void Executor::AddDevice(fastllm::BaseDevice *device) {
        this->devices.push_back(device);
    }

    std::string Executor::GetFirstDeviceType() {
        return this->devices[0]->deviceType;
    }

    void Executor::SetFirstDevice(const std::string &device) {
        auto temp = this->devices;
        this->devices.clear();
        for (int i = 0; i < temp.size(); i++) {
            if (StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
                this->devices.back()->deviceIds = ParseDeviceIds(device, temp[i]->deviceType, this->devices.back()->deviceIdsRatio);
            }
        }
        for (int i = 0; i < temp.size(); i++) {
            if (!StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
            }
        }

        this->firstDevice = device;
    }

    std::vector <int> Executor::GetDeviceIds(const std::string &device) {
        for (int i = 0; i < devices.size(); i++) {
            if (StartWith(devices[i]->deviceType, device)) {
                return devices[i]->deviceIds;
            }
        }
        return {0};
    }

    bool Executor::CanRunOnFirstDevice(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {     
        return this->devices[0]->CanRun(opType, datas, floatParams, intParams);
    }

    void Executor::Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {
        auto st = std::chrono::system_clock::now();
        bool lockInCPU = false;
        if (GetKVCacheInCPU() || GetHistoryCacheInCPU()) {
            // 暂时只有kvcache可能lock在CPU上
            for (auto &it: datas) {
                if (intParams.find(it.first + "___batch") != intParams.end()) {
                    int batch = intParams.find(it.first + "___batch")->second;
                    for (int i = 0; i < batch; i++) {
                        lockInCPU |= (((Data**)it.second)[i] && ((Data**)it.second)[i]->lockInCPU);
                    }
                } else {
                    lockInCPU |= (it.second && it.second->lockInCPU);
                }
            }
        }

        bool run = false;
        for (auto device: devices) {
            if (lockInCPU && device->deviceType != "cpu") {
                continue;
            }
            if (device->CanRun(opType, datas, floatParams, intParams)) {
#ifdef USE_CUDA
                if (device->deviceType == "cuda" && device->deviceIds.size() > 0) {
                    FastllmCudaSetDevice(device->deviceIds[0]);
                }
                if (device->deviceType == "multicuda" && device->deviceIds.size() > 0) {
                    FastllmMultiCudaSetDevice(device->deviceIds);
                    if (device->deviceIdsRatio.size() > 0) {
                        FastllmMultiCudaSetDeviceRatio(device->deviceIdsRatio);
                    }
                }
#endif
                bool intParamsSize = intParams.size();
                for (auto &it: datas) {
                    if (intParamsSize > 0 && intParams.find(it.first + "___batch") != intParams.end()) {
                        int batch = intParams.find(it.first + "___batch")->second;
                        if ((it.first == "weights" || it.first == "biass") && ((Data**)it.second)[2]) {
                            if ((device->deviceType == "cpu" || device->deviceType == "numa" || device->deviceType == "tfacc") && 
                                ((Data**)it.second)[2]->dataDevice == DataDevice::CPU) {
                                continue;
                            }
                            if ((device->deviceType == "cuda" || device->deviceType == "multicuda") && ((Data**)it.second)[2]->dataDevice == DataDevice::CUDA) {
                                continue;
                            }
                        }
                        if ((it.first == "biass") && !((Data**)it.second)[2]) {
                            continue;
                        }
                        for (int i = 0; i < batch; i++) {
                            if (((Data**)it.second)[i]) {
                                ((Data**)it.second)[i]->ToDevice((void *) device);
                            }
                        }
                    } else {
                        if (it.second) {
                            it.second->ToDevice((void *) device);
                        }
                    }
                }
                device->Reshape(opType, datas, floatParams, intParams);
                device->Run(opType, datas, floatParams, intParams);
                run = true;
                break;
            }
        }
        if (!run) {
            ErrorInFastLLM("Can't run " + opType + " in any device.");
        }
        float spend = GetSpan(st, std::chrono::system_clock::now());
        profiler[opType] += spend;
    }

    void Executor::ClearProfiler() {
        profiler.clear();
    }

    void Executor::PrintProfiler() {
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %f\n", it.first.c_str(), it.second);
            sum += it.second;
        }
        printf("total spend %f\n", sum);
    }
}