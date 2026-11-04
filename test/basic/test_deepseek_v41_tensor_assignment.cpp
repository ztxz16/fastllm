#include "../../src/models/deepseekv41_execution.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <iostream>
#include <stdexcept>

using namespace fastllm;

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        const std::vector<int> devices{0, 1};
        for (bool gpuSource : {false, true}) {
            Data destination(FLOAT32, {1, 1, 4}, std::vector<float>{1, 2, 3, 4});
            PrepareMultiCudaReplicatedData(destination, devices, true);
            V41SyncRootFromReplica(destination, devices);
            Data replacement(FLOAT32, {1, 1, 4}, std::vector<float>{5, 6, 7, 8});
            if (gpuSource) replacement.ToDevice(CUDA, std::vector<int>{0});
            V41Assign(destination, replacement, devices);
            Check(!destination.multiDeviceData && destination.multiDeviceDatas.empty() &&
                      !destination.IsTensorParallel(), "stale replicas after ordinary assignment");
            destination.ToDevice(CPU);
            for (int i = 0; i < 4; ++i)
                Check(((float *)destination.cpuData)[i] == i + 5, "ordinary assignment values");

            PrepareMultiCudaReplicatedData(destination, devices, true);
            V41SyncRootFromReplica(destination, devices);
            V41Assign(destination, destination, devices);
            Check(destination.IsTensorParallelReplicated(), "self assignment lost replicas");
            Data copy;
            V41Assign(copy, destination, devices);
            Check(copy.IsTensorParallelReplicated(), "replicated assignment layout");
            for (int device : devices) {
                Check(copy.multiDeviceDatas.at(device)->cudaData !=
                          destination.multiDeviceDatas.at(device)->cudaData,
                      "replicated assignment must own its storage");
                Data host;
                host.CopyFrom(*copy.multiDeviceDatas.at(device));
                host.ToDevice(CPU);
                for (int i = 0; i < 4; ++i)
                    Check(((float *)host.cpuData)[i] == i + 5, "replicated assignment values");
            }
        }
        std::cout << "PASS V41 ordinary, replicated and self assignment\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
