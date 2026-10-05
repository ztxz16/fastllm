#include "fastllm.h"
#include "executor.h"
#include "utils/utils.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
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
        for (int vocab : {1, 7, 255, 256, 257, 1023, 1025, 152576}) {
            for (int pattern = 0; pattern < 5; ++pattern) {
                Data base(BFLOAT16, {1, 2, vocab}), bias(BFLOAT16, {1, 1, vocab});
                base.Allocate(); bias.Allocate();
                for (int i = 0; i < vocab; ++i) {
                    float a = std::sin(i * .137f), b = std::cos(i * .213f);
                    if (pattern == 1) { a = -1; b = 0; if (i == 1 || i == 2 || i == 256) a = 2; }
                    if (pattern == 2) { a = std::numeric_limits<float>::quiet_NaN(); b = 0; }
                    if (pattern == 3) { a = -std::numeric_limits<float>::infinity(); b = 0; }
                    if (pattern == 4) { a = i % 7 ? 0 : std::numeric_limits<float>::infinity(); b = 0; }
                    ((uint16_t *)base.cpuData)[i] = Float32ToBFloat16RNEBits(a);
                    ((uint16_t *)base.cpuData)[vocab + i] = Float32ToBFloat16RNEBits(-a);
                    ((uint16_t *)bias.cpuData)[i] = Float32ToBFloat16RNEBits(b);
                }
                base.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                bias.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                Data ids(FLOAT32, {1, 3}, {0, 0, 0}), partial;
                ids.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                for (int row = 0; row < 2; ++row) {
                    Data logits, top;
                    Split(base, 1, row, row + 1, logits);
                    AddTo(logits, bias); ToDataType(logits, FLOAT32); TopK(logits, top, 1);
                    top.ToDevice(DataDevice::CPU);
                    FastllmCudaNaiveDraftArgmax(base, bias, row, partial, ids);
                    Data host(ids); host.ToDevice(DataDevice::CPU);
                    Require(((float *)host.cpuData)[row + 1] == ((float *)top.cpuData)[0], "fused argmax differs from Top1");
                    ++checks;
                }
            }
        }
        for (int rows : {1, 5, 8}) {
            std::vector<Data> sources;
            for (int i = 0; i < 8; ++i) sources.push_back(BF({1, 8, 257}, i * 91));
            std::vector<const Data *> pointers;
            Data reference;
            for (auto &x : sources) {
                pointers.push_back(&x);
                Data slice, joined; Split(x, 1, 0, rows, slice);
                if (reference.dims.empty()) Copy(slice, reference);
                else { Cat(reference, slice, -1, joined); Copy(joined, reference); }
            }
            Data output;
            Require(FastllmCudaNaiveDraftConcat(pointers, rows, output), "concat fast path rejected valid inputs");
            Require(Bits(output) == Bits(reference), "concat changed feature order");
            Require(!FastllmCudaNaiveDraftConcat(pointers, 9, output), "concat accepted an invalid row count");
            ++checks;
        }
        {
            Data padded = BF({1, 8, 258}, 31), output = BF({1, 1, 7}, 17);
            // A view keeps the 258-element row stride while exposing only 257
            // columns. The dense fast path must reject it without touching output.
            padded.dims[2] = 257;
            auto before = Bits(output);
            Require(!FastllmCudaNaiveDraftConcat({&padded}, 5, output), "concat accepted padded rows");
            Require(output.dims == std::vector<int>({1, 1, 7}) && Bits(output) == before,
                    "concat fallback changed output");
            ++checks;
        }
        std::cout << "DRAFT OPS PASS checks=" << checks << '\n';
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
