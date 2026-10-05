#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
static void Require(bool value, const char *what) { if (!value) throw std::runtime_error(what); }
static Data BF(const std::vector<int> &dims, int seed) {
    Data x(BFLOAT16, dims); x.Allocate();
    for (int i = 0; i < x.Count(0); ++i)
        ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
    x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    return x;
}
static std::vector<uint16_t> Bits(const Data &x) {
    Data host(x); host.ToDevice(DataDevice::CPU);
    return std::vector<uint16_t>((uint16_t *)host.cpuData, (uint16_t *)host.cpuData + host.Count(0));
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        ApplyDeviceMap({{"cuda:0", 1}}, 1, 1);
        int checks = 0;
        // The unchanged two-row GEMV is an exact oracle for the new one-row
        // layout, including bias conversion, tail output tiles and cancellation.
        for (int outputs : {1, 7, 8, 9, 255, 513, 19072, 152576}) {
            Data x = BF({1, 256}, 71);
            Data twice, bias = BF({outputs}, 31);
            ToDataType(bias, FLOAT32);
            Cat(x, x, 0, twice);
            for (bool withBias : {false, true}) {
                Data w = BF({outputs, 256}, outputs);
                Data actual, reference;
                Linear(x, w, withBias ? bias : Data(), actual);
                Linear(twice, w, withBias ? bias : Data(), reference);
                auto a = Bits(actual), b = Bits(reference);
                Require(std::equal(a.begin(), a.end(), b.begin()), "small-K GEMV differs from multirow");
                ++checks;
            }
        }
        std::cout << "DRAFT OPS PASS checks=" << checks << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
