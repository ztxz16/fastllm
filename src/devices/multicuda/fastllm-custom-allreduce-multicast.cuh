#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <memory>
#include <limits>
#include <vector>

#if defined(__linux__) && CUDA_VERSION >= 12010
#include <dlfcn.h>

namespace {
// Optional driver entry points keep older drivers and non-NVSwitch machines
// on the peer-write path without adding a libcuda link-time dependency.
struct TaggedMulticastDriver {
#define TAGGED_MC_API(X) \
    X(cuDeviceGetAttribute) X(cuMulticastGetGranularity) X(cuMulticastCreate) \
    X(cuMulticastAddDevice) X(cuMulticastBindMem) X(cuMulticastUnbind) \
    X(cuMemCreate) X(cuMemRelease) X(cuMemAddressReserve) X(cuMemAddressFree) \
    X(cuMemMap) X(cuMemUnmap) X(cuMemSetAccess)
#define TAGGED_MC_DECLARE(name) decltype(&name) name##Fn = nullptr;
    TAGGED_MC_API(TAGGED_MC_DECLARE)
#undef TAGGED_MC_DECLARE
    bool available = false;
    TaggedMulticastDriver() {
        void *driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!driver) return;
        available = true;
#define TAGGED_MC_LOAD(name) \
        name##Fn = reinterpret_cast<decltype(name##Fn)>(dlsym(driver, #name)); \
        available = available && name##Fn;
        TAGGED_MC_API(TAGGED_MC_LOAD)
#undef TAGGED_MC_LOAD
#undef TAGGED_MC_API
        // Retain the module: buffers can outlive other CUDA singleton objects.
    }
};

class TaggedMulticastMemory {
public:
    static std::unique_ptr<TaggedMulticastMemory> Create(const std::vector<int> &devices, size_t bytes) {
        auto result = std::unique_ptr<TaggedMulticastMemory>(new TaggedMulticastMemory);
        if (!result->Initialize(devices, bytes)) return nullptr;
        return result;
    }
    TaggedMulticastMemory(const TaggedMulticastMemory &) = delete;
    TaggedMulticastMemory &operator=(const TaggedMulticastMemory &) = delete;
    ~TaggedMulticastMemory() { Release(); }
    char *Local(int rank) const { return reinterpret_cast<char *>(local[rank]); }
    char *Broadcast(int rank) const { return reinterpret_cast<char *>(broadcast[rank]); }

private:
    TaggedMulticastMemory() = default;
    static TaggedMulticastDriver &Driver() {
        static TaggedMulticastDriver api;
        return api;
    }
    std::vector<int> devices;
    CUmemGenericAllocationHandle group = 0;
    std::array<CUmemGenericAllocationHandle, 8> physical{};
    std::array<CUdeviceptr, 8> local{}, broadcast{};
    std::array<bool, 8> localMapped{}, broadcastMapped{}, bound{};
    size_t size = 0;

    bool Initialize(const std::vector<int> &requested, size_t bytes) {
        auto &api = Driver();
        if (!api.available || requested.size() != 8 || bytes == 0) return false;
        for (size_t i = 0; i < requested.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (requested[i] == requested[j]) return false;
        for (int device : requested) {
            int supported = 0;
            if (api.cuDeviceGetAttributeFn(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                                           device) != CUDA_SUCCESS || !supported) return false;
        }
        devices = requested;
        CUmulticastObjectProp prop{};
        prop.numDevices = devices.size();
        prop.size = bytes;
        size_t granularity = 0;
        if (api.cuMulticastGetGranularityFn(&granularity, &prop,
                CU_MULTICAST_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS || !granularity) return false;
        if (bytes > std::numeric_limits<size_t>::max() - (granularity - 1)) return false;
        size = (bytes + granularity - 1) / granularity * granularity;
        prop.size = size;
        if (api.cuMulticastCreateFn(&group, &prop) != CUDA_SUCCESS) return false;
        // All devices must join before BindMem or Map; these operations can
        // otherwise wait for a missing peer indefinitely.
        for (int device : devices)
            if (api.cuMulticastAddDeviceFn(group, device) != CUDA_SUCCESS) return false;
        std::array<CUmemAccessDesc, 8> access{};
        for (int rank = 0; rank < 8; ++rank) {
            access[rank].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access[rank].location.id = devices[rank];
            access[rank].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        for (int rank = 0; rank < 8; ++rank) {
            if (cudaSetDevice(devices[rank]) != cudaSuccess) return false;
            CUmemAllocationProp allocation{};
            allocation.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            allocation.location = access[rank].location;
            if (api.cuMemCreateFn(&physical[rank], size, &allocation, 0) != CUDA_SUCCESS ||
                api.cuMemAddressReserveFn(&local[rank], size, granularity, 0, 0) != CUDA_SUCCESS)
                return false;
            if (api.cuMemMapFn(local[rank], size, 0, physical[rank], 0) != CUDA_SUCCESS) return false;
            localMapped[rank] = true;
            if (api.cuMemSetAccessFn(local[rank], size, access.data(), access.size()) != CUDA_SUCCESS ||
                api.cuMulticastBindMemFn(group, 0, physical[rank], 0, size, 0) != CUDA_SUCCESS) return false;
            bound[rank] = true;
            if (api.cuMemAddressReserveFn(&broadcast[rank], size, granularity, 0, 0) != CUDA_SUCCESS ||
                api.cuMemMapFn(broadcast[rank], size, 0, group, 0) != CUDA_SUCCESS) return false;
            broadcastMapped[rank] = true;
            if (api.cuMemSetAccessFn(broadcast[rank], size, &access[rank], 1) != CUDA_SUCCESS) return false;
        }
        return true;
    }

    void Release() {
        if (devices.empty()) return;
        // The owning collective drains all device streams before destruction.
        // Initialization failures have never published these mappings.
        auto &api = Driver();
        int original = 0;
        cudaGetDevice(&original);
        for (int rank = 0; rank < int(devices.size()); ++rank) {
            cudaSetDevice(devices[rank]);
            if (broadcastMapped[rank]) api.cuMemUnmapFn(broadcast[rank], size);
            if (broadcast[rank]) api.cuMemAddressFreeFn(broadcast[rank], size);
            if (bound[rank]) api.cuMulticastUnbindFn(group, devices[rank], 0, size);
            if (localMapped[rank]) api.cuMemUnmapFn(local[rank], size);
            if (local[rank]) api.cuMemAddressFreeFn(local[rank], size);
            if (physical[rank]) api.cuMemReleaseFn(physical[rank]);
        }
        if (group) api.cuMemReleaseFn(group);
        devices.clear();
        cudaSetDevice(original);
    }
};
} // namespace
#else
namespace {
class TaggedMulticastMemory {
public:
    static std::unique_ptr<TaggedMulticastMemory> Create(const std::vector<int> &, size_t) { return nullptr; }
    char *Local(int) const { return nullptr; }
    char *Broadcast(int) const { return nullptr; }
};
}
#endif
