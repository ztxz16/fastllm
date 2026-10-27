// Shared tensor storage helpers; execution policies live in their own files.
#include "deepseekv41_execution.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <set>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    bool V41EnvFlag(const char *name) {
        const char *v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
    }

    bool V41DeviceSpecUsesType(const std::string &spec, const std::string &type) {
        std::string normalized = spec;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return normalized == type || normalized.rfind(type + ":", 0) == 0;
    }

    bool V41DeviceMapUsesCuda(const std::map<std::string, int> &deviceMap) {
        for (const auto &it : deviceMap) {
            if (V41DeviceSpecUsesType(it.first, "cuda") ||
                V41DeviceSpecUsesType(it.first, "multicuda")) {
                return true;
            }
        }
        return false;
    }

    int V41DeviceMapCudaDeviceCount(const std::map<std::string, int> &deviceMap) {
        std::set<int> devices;
        for (const auto &it : deviceMap) {
            const std::string &spec = it.first;
            if (!V41DeviceSpecUsesType(spec, "cuda")) {
                continue;
            }
            size_t pos = spec.find(':');
            if (pos == std::string::npos) {
                devices.insert(0);
            } else {
                devices.insert(atoi(spec.c_str() + pos + 1));
            }
        }
        return (int)devices.size();
    }

#ifdef USE_CUDA
    void V41ResetMultiDevice(Data &data) {
        if (!data.multiDeviceData) {
            return;
        }
        for (auto &it : data.multiDeviceDatas) {
            delete it.second;
        }
        data.multiDeviceDatas.clear();
        data.multiDeviceData = false;
        data.ClearTensorParallelLayout();
    }

    void V41SyncRootFromReplica(Data &data, const std::vector<int> &devices) {
        if (!data.multiDeviceData || devices.empty()) {
            return;
        }
        auto it = data.multiDeviceDatas.find(devices[0]);
        if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
            return;
        }
        Data *first = it->second;
        data.dataType = first->dataType;
        data.UpdateUnitSize();
        if (data.dims != first->dims) {
            data.Resize(first->dims);
        }
        data.expansionDims = first->expansionDims;
        data.strides = first->strides;
        data.expansionSize = first->expansionSize;
        data.expansionBytes = first->expansionBytes;
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = devices;
        data.tpLayout = TP_LAYOUT_REPLICATED;
        data.tpAxis = -1;
        data.tpGlobalDims = data.dims;
    }

    static void V41CopyTensor(Data &dst, const Data &src, const std::vector<int> &devices) {
        if (devices.empty() || !src.multiDeviceData || !src.IsTensorParallelReplicated()) {
            dst.CopyFrom(src);
            return;
        }
        V41ResetMultiDevice(dst);
        dst.FreeSpace();
        dst.dataType = src.dataType;
        dst.UpdateUnitSize();
        dst.Resize(src.dims);
        dst.dataDevice = DataDevice::CUDA;
        dst.dataDeviceIds = devices;
        dst.multiDeviceData = true;
        const int oriDevice = FastllmCudaGetDevice();
        for (int device : devices) {
            auto it = src.multiDeviceDatas.find(device);
            AssertInFastLLM(it != src.multiDeviceDatas.end() && it->second != nullptr,
                            "DeepSeekV41: missing TP replica while copying a tensor.");
            FastllmCudaSetDevice(device);
            Data *local = new Data();
            local->CopyFrom(*it->second);
            local->dataDeviceIds = {device};
            dst.multiDeviceDatas[device] = local;
        }
        FastllmCudaSetDevice(oriDevice);
        dst.tpLayout = TP_LAYOUT_REPLICATED;
        dst.tpAxis = -1;
        dst.tpGlobalDims = dst.dims;
        V41SyncRootFromReplica(dst, devices);
    }

#endif
            void V41Assign(Data &dst, const Data &src, const std::vector<int> &tpDevices) {
    #ifdef USE_CUDA
                if (!tpDevices.empty() && src.multiDeviceData && src.IsTensorParallelReplicated()) {
                    V41CopyTensor(dst, src, tpDevices);
                    return;
                }
    #endif
                dst.CopyFrom(src);
            }

            void V41ReplicaToCpu(Data &dst, const Data &src, const std::vector<int> &tpDevices) {
    #ifdef USE_CUDA
                // Preserve prefill's allocation / staging behavior. In particular,
                // retaining its large CPU buffers changes dynamic expert staging.
                // The optimization targets one activation vector during decode.
                if (!src.dims.empty() && src.Count(0) == (uint64_t)src.dims.back() &&
                    MultiCudaCopyReplicaToCpu(dst, src, tpDevices)) return;
                if (!tpDevices.empty() && src.multiDeviceData && src.IsTensorParallelReplicated()) {
                    for (int device : tpDevices) {
                        auto it = src.multiDeviceDatas.find(device);
                        if (it == src.multiDeviceDatas.end() || it->second == nullptr ||
                            it->second->cudaData == nullptr) {
                            continue;
                        }
                        const int oriDevice = FastllmCudaGetDevice();
                        FastllmCudaSetDevice(device);
                        dst.CopyFrom(*it->second);
                        dst.ToDevice(DataDevice::CPU);
                        FastllmCudaSetDevice(oriDevice);
                        return;
                    }
                }
    #endif
                dst.CopyFrom(src);
                dst.ToDevice(DataDevice::CPU);
            }

} // namespace fastllm
