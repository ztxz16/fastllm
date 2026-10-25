#include "../../src/devices/multicuda/fastllm-custom-allreduce-multicast.cuh"
#include <cstdint>
#include <iostream>
#include <stdexcept>

static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static void Cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void Publish(uint64_t *output, int rank, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 257) output[rank * 512 + i] = (uint64_t(step + 1) << 32) | uint64_t(rank * 1000 + i);
}
int main() {
    try {
        int count = 0;
        Cuda(cudaGetDeviceCount(&count));
        if (count < 8) return 77;
        std::vector<int> devices;
        for (int d = 0; d < 8; ++d) {
            cudaDeviceProp prop{};
            Cuda(cudaGetDeviceProperties(&prop, d));
            if (prop.major != 9) return 77;
            Cuda(cudaSetDevice(d));
            Cuda(cudaFree(nullptr));
            devices.push_back(d);
        }
        Check(!TaggedMulticastMemory::Create({}, 32768), "empty group accepted");
        Check(!TaggedMulticastMemory::Create({0, 1}, 32768), "unsupported group accepted");
        Check(!TaggedMulticastMemory::Create(devices, 0), "empty allocation accepted");
        for (int repeat = 0; repeat < 3; ++repeat) {
            // Reject duplicate devices before creating an incomplete group.
            Check(!TaggedMulticastMemory::Create(std::vector<int>(8, 0), 32768), "duplicate devices accepted");
            auto memory = TaggedMulticastMemory::Create(devices, 32768);
            Check(bool(memory), "multicast allocation unavailable");
            for (int d : devices) {
                Cuda(cudaSetDevice(d));
                Cuda(cudaMemset(memory->Local(d), 0, 32768));
                Cuda(cudaDeviceSynchronize());
            }
            for (int step = 0; step < 8; ++step) {
                for (int d : devices) {
                    Cuda(cudaSetDevice(d));
                    Publish<<<2, 256>>>(reinterpret_cast<uint64_t *>(memory->Broadcast(d)), d, step);
                    Cuda(cudaGetLastError());
                }
                for (int d : devices) { Cuda(cudaSetDevice(d)); Cuda(cudaDeviceSynchronize()); }
                for (int d : devices) {
                    Cuda(cudaSetDevice(d));
                    std::vector<uint64_t> result(4096);
                    Cuda(cudaMemcpy(result.data(), memory->Local(d), 32768, cudaMemcpyDeviceToHost));
                    for (int rank : devices)
                        for (int i = 0; i < 512; ++i) {
                            uint64_t expected = i < 257 ? (uint64_t(step + 1) << 32) | uint64_t(rank * 1000 + i) : 0;
                            Check(result[rank * 512 + i] == expected, "multicast payload mismatch");
                        }
                }
            }
        }
        std::cout << "PASS multicast allocation, invalid groups, broadcast and repeated teardown\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
