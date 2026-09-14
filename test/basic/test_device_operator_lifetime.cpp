#include "device.h"
#include "executor.h"
#include "devices/cpu/cpudevice.h"
#include "devices/disk/diskdevice.h"
#ifdef USE_CUDA
#include "devices/cuda/cudadevice.h"
#include "devices/multicuda/multicudadevice.h"
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
class CountingOperator : public fastllm::BaseOperator {
public:
    explicit CountingOperator(int &live) : live(live) { ++live; }
    ~CountingOperator() override { --live; }
    void Run(const std::string &, const fastllm::DataDict &,
             const fastllm::FloatDict &, const fastllm::IntDict &) override {}
private:
    int &live;
};

class TestDevice : public fastllm::BaseDevice {
public:
    explicit TestDevice(int &live) {
        ops["first"] = new CountingOperator(live);
        ops["second"] = new CountingOperator(live);
    }
    bool Malloc(void **, size_t) override { return false; }
    bool Free(void *) override { return false; }
    bool CopyDataToCPU(void *, void *, size_t) override { return false; }
    bool CopyDataFromCPU(void *, void *, size_t) override { return false; }
};

static_assert(!std::is_copy_constructible<TestDevice>::value,
              "owning device registries must not be shallow copied");
static_assert(!std::is_copy_assignable<TestDevice>::value,
              "owning device registries must not be shallow assigned");

void CheckConcreteRegistries() {
    using namespace fastllm;
    for (int iteration = 0; iteration < 1000; ++iteration) {
        std::vector<std::unique_ptr<BaseDevice>> devices;
#ifdef USE_CUDA
        auto *cuda = new CudaDevice();
        devices.emplace_back((BaseDevice*)cuda);
        devices.emplace_back((BaseDevice*)new MultiCudaDevice(cuda));
#endif
#ifdef USE_NUMAS
        devices.emplace_back((BaseDevice*)new NumasDevice());
#endif
        devices.emplace_back((BaseDevice*)new DiskDevice());
        devices.emplace_back((BaseDevice*)new CpuDevice());
        std::set<const void*> ownedOperators;
        for (const auto &device : devices) {
            for (const auto &entry : device->ops) {
                if (!entry.second || !ownedOperators.insert(
                        dynamic_cast<const void*>(entry.second)).second) {
                    throw std::runtime_error("operator ownership is null or shared across registries");
                }
            }
            if (iteration == 0) {
                std::cout << device->deviceType << " registry: "
                          << device->ops.size() << " unique operators\n";
            }
        }
        // MultiCUDA wrappers borrow CUDA operators. Neither destruction order
        // may delete those borrowed pointers or dereference them at teardown.
        if (iteration % 2 == 0) {
            for (auto &device : devices) device.reset();
        } else {
            for (auto it = devices.rbegin(); it != devices.rend(); ++it) it->reset();
        }
    }
    for (int iteration = 0; iteration < 100; ++iteration) {
        Executor executor;
        executor.SetFirstDevice(iteration % 2 == 0 ? "cpu" : "cuda:0");
    }
}
}

int main() {
    int live = 0;
    for (int iteration = 0; iteration < 1000; ++iteration) {
        {
            std::unique_ptr<fastllm::BaseDevice> device(new TestDevice(live));
            if (live != 2) {
                std::cerr << "operator ownership count is wrong\n";
                return 1;
            }
        }
        if (live != 0) {
            std::cerr << "device destruction leaked " << live << " operators\n";
            return 1;
        }
    }
    try {
        TestDevice device(live);
        throw std::runtime_error("exercise stack unwinding");
    } catch (const std::runtime_error &) {
        if (live != 0) {
            std::cerr << "exception unwinding leaked operators\n";
            return 1;
        }
    }
    CheckConcreteRegistries();
    std::cout << "Device operator lifetime: PASS\n";
}
