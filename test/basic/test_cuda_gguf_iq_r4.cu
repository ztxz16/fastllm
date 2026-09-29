#include <cublas_v2.h>
#include "fastllm-gguf-dequant.cuh"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }

template<typename T> static float Round(float value);
template<> float Round<float>(float value) { return value; }
template<> float Round<half>(float value) { return __half2float(__float2half(value)); }
template<> float Round<__nv_bfloat16>(float value) { return __bfloat162float(__float2bfloat16(value)); }
template<typename T> static float Float(T value) { return static_cast<float>(value); }

template<typename T>
static void TestType(ggml_type original, ggml_type packedType, int rows, int columns,
                     to_t_cuda_t<T> dequant) {
    Check(dequant != nullptr, "missing IQ R4 CUDA dequantizer");
    const size_t rowBytes = ggml_row_size(original, columns);
    std::vector<uint8_t> plain(rows * rowBytes), packed(plain.size());
    uint32_t random = 1919;
    for (auto &byte : plain) { random = random * 1664525U + 1013904223U; byte = random >> 24; }
    // Keep scales finite and exactly representable. Random payloads exercise
    // all sign indices, sub-block scales and codebook entries.
    const size_t blockBytes = ggml_type_size(original);
    for (size_t offset = 0; offset < plain.size(); offset += blockBytes) {
        const uint16_t scale = float_to_half(float((offset / blockBytes) % 7 + 1) / 32);
        std::memcpy(plain.data() + offset, &scale, sizeof(scale));
    }
    const auto *repack = get_repack_info(original);
    Check(repack && repack->new_type == packedType, "missing CPU IQ repacker");
    repack->repack(rows, columns, reinterpret_cast<const char *>(plain.data()),
                   reinterpret_cast<char *>(packed.data()), false);
    std::vector<float> reference(rows * columns);
    ggml_type_to_float(original)(plain.data(), reference.data(), reference.size());
    void *deviceWeight = nullptr; T *deviceOutput = nullptr;
    Cuda(cudaMalloc(&deviceWeight, packed.size()));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&deviceOutput), reference.size() * sizeof(T)));
    Cuda(cudaMemcpy(deviceWeight, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    Data weight(DATA_GGUF_FORMAT, int(packedType), {rows, columns});
    const int rowGroup = FastllmGGUFDequantRowGroup(packedType);
    Check(rowGroup == 4, "IQ R4 workspace split is not aligned to four rows");
    // A workspace fitting 7 rows must split into groups of 4, not split R4 blocks.
    const int chunkRows = FastllmGGUFCalcChunkRows(7 * columns * sizeof(T), columns,
        rows, sizeof(T), 0, rowGroup, weight, "IQ R4 test");
    Check(chunkRows == 4, "IQ R4 chunk alignment");
    for (int start = 0; start < rows; start += chunkRows) {
        const int count = std::min(chunkRows, rows - start);
        dequant(static_cast<uint8_t *>(deviceWeight) + start * rowBytes,
                deviceOutput + start * columns, count, columns, cudaStreamPerThread);
    }
    Cuda(cudaGetLastError()); Cuda(cudaDeviceSynchronize());
    std::vector<T> output(reference.size());
    Cuda(cudaMemcpy(output.data(), deviceOutput, output.size() * sizeof(T), cudaMemcpyDeviceToHost));
    Cuda(cudaFree(deviceWeight)); Cuda(cudaFree(deviceOutput));
    for (size_t i = 0; i < reference.size(); ++i) {
        const float actual = Float(output[i]), expected = Round<T>(reference[i]);
        if (actual != expected) {
            std::cerr << ggml_type_name(packedType) << " index=" << i << " actual=" << actual << " expected=" << expected << '\n';
            throw std::runtime_error("IQ R4 CUDA/CPU dequantization mismatch");
        }
    }
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        if (!get_repack_info(GGML_TYPE_IQ2_XS)) return 77;
        Cuda(cudaSetDevice(0));
        for (const auto types : {std::make_pair(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS_R4),
                                  std::make_pair(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS_R4)}) {
            for (int columns : {256, 2560}) {
                TestType<float>(types.first, types.second, 12, columns, ggml_get_to_fp32_cuda(types.second));
                TestType<half>(types.first, types.second, 12, columns, ggml_get_to_fp16_cuda(types.second));
                TestType<__nv_bfloat16>(types.first, types.second, 12, columns, ggml_get_to_bf16_cuda(types.second));
            }
        }
        std::cout << "PASS: IQ2_XS/IQ3_XXS R4 CUDA FP32/FP16/BF16 match CPU, including chunk boundaries\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
