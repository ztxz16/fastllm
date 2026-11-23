#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(what);
}
template<class T> struct DeviceBuffer {
    T *data = nullptr;
    explicit DeviceBuffer(const std::vector<T> &source) {
        const size_t bytes = source.size() * sizeof(T);
        check(cudaMalloc(&data, bytes) == cudaSuccess, "allocate");
        check(cudaMemcpy(data, source.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess, "upload");
    }
    ~DeviceBuffer() { cudaFree(data); }
};

template<class T> void append(std::vector<uint8_t> &out, T v) {
    const size_t start = out.size();
    out.resize(start + sizeof(T));
    std::memcpy(out.data() + start, &v, sizeof(T));
}

void verify(int ngram, int heads, int layers, bool positions, int anchor, int start) {
    constexpr int rows = FastllmV41OwnerVerifyRows, image = 7, pad = 5;
    const int ids[rows] = {anchor, 3, image, 2, 0, 6};
    const std::vector<int> map{3, -1, 9, 10, 13, 15, 19, 23};
    FastllmV41VerifyHistory history{{3, -1, 13, 10, 3, 19, 10, 9}};
    for (int i = start; i < 8; ++i) history.previous[i] = -1;
    const int cols = (ngram - 1) * heads;
    std::vector<uint8_t> expected;
    for (int id : ids) append(expected, id);
    if (positions) for (int row = 0; row < rows; ++row) append(expected, start + row);
    std::vector<int64_t> params;
    auto compress = [&](int i) { return ids[i] == image ? -1 : map[ids[i]]; };
    for (int layer = 0; layer < layers; ++layer) {
        const size_t base = params.size();
        for (int j = 0; j < ngram; ++j) params.push_back(INT64_MAX - 2 * j - 16 * layer);
        for (int col = 0; col < cols; ++col) params.push_back(17 + 2 * col);
        for (int col = 0; col < cols; ++col) params.push_back(193 * col);
        for (int row = 0; row < rows; ++row) {
            uint64_t rolling = 0;
            bool blocked = false;
            for (int j = 0; j < ngram; ++j) {
                const int i = row - j, token = i < 0 ? history.previous[-1-i] : compress(i);
                blocked |= token < 0;
                rolling ^= uint64_t(blocked ? pad : token) * uint64_t(params[base + j]);
                if (!j) continue;
                for (int head = 0; head < heads; ++head) {
                    const int col = (j - 1) * heads + head;
                    const int64_t prime = params[base + ngram + col], remainder = int64_t(rolling) % prime;
                    append(expected, params[base + ngram + cols + col] + (remainder < 0 ? remainder + prime : remainder));
                }
            }
        }
        for (int row = 0; row < rows; ++row) append(expected, compress(row) < 0 ? 0.f : 1.f);
    }
    std::vector<uint8_t> actual(expected.size(), 0xab);
    // Row zero deliberately contains a stale ID. Only the scheduler owns the anchor.
    std::memcpy(actual.data() + sizeof(int), ids + 1, (rows - 1) * sizeof(int));
    DeviceBuffer<uint8_t> packet(actual);
    DeviceBuffer<int> tokenMap(map);
    DeviceBuffer<int64_t> parameters(params);
    check(FastllmCudaV41BuildOwnerVerifyPacket(packet.data, positions, start, anchor, tokenMap.data,
            int(map.size()), pad, image, parameters.data, layers, ngram, heads, history), "launch");
    check(cudaDeviceSynchronize() == cudaSuccess, "execute");
    check(cudaMemcpy(actual.data(), packet.data, actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read");
    check(actual == expected, "owner packet differs from CPU reference");
}
}

int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
        return 77;
    check(status == cudaSuccess, cudaGetErrorString(status));
    for (int ngram : {2, 4, 9})
        for (int heads : {1, 8, 17})
            for (int layers : {1, 2})
                for (bool positions : {false, true})
                    for (int anchor : {1, 4, 7})
                        for (int start : {0, 37}) verify(ngram, heads, layers, positions, anchor, start);
    std::cout << "216 owner packets matched the CPU reference\n";
}
