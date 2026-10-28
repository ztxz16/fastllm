// Bitwise regression against the original scalar R4 layout. Cover both IQ2
// encodings, destination types, unaligned fallback and special FP16 scales.
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "fastllm-gguf-dequant.cuh"
#include "fastllm-gguf-gemv.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<ggml_type type> struct Format;
template<> struct Format<GGML_TYPE_IQ2_XS_R4> {
    using Block = block_iq2_xs_r4;
    static constexpr const char *name = "IQ2_XS_R4";
    static constexpr int exhaustiveBlocks = 8192;
};
template<> struct Format<GGML_TYPE_IQ2_XXS_R4> {
    using Block = block_iq2_xxs_r4;
    static constexpr const char *name = "IQ2_XXS_R4";
    static constexpr int exhaustiveBlocks = 4096;
};

template<class T, ggml_type type>
__global__ void Reference(const typename Format<type>::Block *src, T *output, int64_t columns) {
    const int64_t block = blockIdx.x, blocksPerRow = columns / QK_K;
    const int row = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ib = lane % 8, il = lane / 8;
    const auto &x = src[block];
    T *y = output + (block / blocksPerRow * 4 + row) * columns +
        block % blocksPerRow * QK_K + 32 * ib + 8 * il;
    uint64_t grid;
    uint8_t packed;
    int scale;
    if constexpr (type == GGML_TYPE_IQ2_XS_R4) {
        const uint16_t q2 = x.qs[16 * ib + 4 * row + il];
        grid = iq2xs_grid[q2 & 511];
        packed = q2 >> 9;
        scale = (x.scales[4 * ib + row] >> (4 * (il / 2))) & 15;
    } else {
        const uint8_t *sas = x.sas + 16 * ib + 4 * row;
        scale = (sas[0] & 1) | ((sas[1] & 1) << 1) |
                ((sas[2] & 1) << 2) | ((sas[3] & 1) << 3);
        grid = iq2xxs_grid[x.qs[16 * ib + 4 * row + il]];
        packed = sas[il] >> 1;
    }
    const uint8_t signs = ksigns_iq2xs[(packed ^ (packed << 1)) & 127];
    const float d = __half2float(x.d[row]) * (0.5f + scale) * 0.25f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (grid >> (8 * j)) & 255;
        y[j] = DequantizeCast<T>::cast(d * q * ((signs & (1 << j)) ? -1.f : 1.f));
    }
}

template<class T> to_t_cuda_t<T> Dispatch(ggml_type type);
template<> to_t_cuda_t<float> Dispatch<float>(ggml_type type) { return ggml_get_to_fp32_cuda(type); }
template<> to_t_cuda_t<half> Dispatch<half>(ggml_type type) { return ggml_get_to_fp16_cuda(type); }
template<> to_t_cuda_t<__nv_bfloat16> Dispatch<__nv_bfloat16>(ggml_type type) { return ggml_get_to_bf16_cuda(type); }

template<class T, ggml_type type>
void Run(const char *dtype, int rows, int columns, int offset, bool exhaustive = false) {
    using F = Format<type>;
    using Block = typename F::Block;
    std::mt19937 rng(1027 + rows + columns);
    std::vector<Block> source(size_t(rows) * columns / (4 * QK_K));
    for (auto &block : source) {
        auto *bytes = reinterpret_cast<uint8_t *>(&block);
        for (size_t i = 0; i < sizeof(block); ++i) bytes[i] = rng() & 255;
        for (int row = 0; row < 4; ++row) {
            uint16_t bits = (rng() % 0x7c00) | ((rng() & 1) << 15);
            std::memcpy(&block.d[row], &bits, sizeof(bits));
        }
    }
    const uint16_t special[] = {0, 0x8000, 1, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0x7e00};
    for (size_t i = 0; i < source.size() && i < sizeof(special) / sizeof(special[0]); ++i)
        std::memcpy(&source[i].d[0], special + i, sizeof(special[0]));
    if (exhaustive) {
        // Enumerate every grid/sign combination at each of the 16 scale nibbles.
        if (source.size() != F::exhaustiveBlocks) throw std::runtime_error("Wrong exhaustive shape");
        for (size_t b = 0; b < source.size(); ++b) {
            auto &block = source[b];
            for (int row = 0; row < 4; ++row) block.d[row] = __float2half(1.f);
            if constexpr (type == GGML_TYPE_IQ2_XS_R4) {
                for (int i = 0; i < 32; ++i) block.scales[i] = (b / 512) * 0x11;
                for (int i = 0; i < 128; ++i) block.qs[i] = (b * 128 + i) & 65535;
            } else {
                const int scale = b / 256;
                for (int i = 0; i < 128; ++i) {
                    const int packed = (b * 128 + i) & 32767;
                    block.qs[i] = packed & 255;
                    block.sas[i] = ((packed >> 8) << 1) | ((scale >> (i % 4)) & 1);
                }
            }
        }
    }
    Block *x;
    T *a, *b;
    const size_t count = size_t(rows) * columns, bytes = (count + 32) * sizeof(T);
    Check(cudaMalloc(&x, source.size() * sizeof(Block)));
    Check(cudaMalloc(&a, bytes)); Check(cudaMalloc(&b, bytes));
    Check(cudaMemcpy(x, source.data(), source.size() * sizeof(Block), cudaMemcpyHostToDevice));
    Check(cudaMemset(a, 0x5a, bytes)); Check(cudaMemset(b, 0x5a, bytes));
    Reference<T, type><<<source.size(), 128>>>(x, a + offset, columns);
    Dispatch<T>(type)(x, b + offset, rows, columns, 0);
    Check(cudaGetLastError());
    std::vector<uint8_t> expected(bytes), actual(bytes);
    Check(cudaMemcpy(expected.data(), a, bytes, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(), b, bytes, cudaMemcpyDeviceToHost));
    if (expected != actual) throw std::runtime_error(std::string(F::name) + " " + dtype + " bitwise/guard mismatch");
    std::printf("PASS format=%s dtype=%s rows=%d columns=%d offset=%d exhaustive=%d values=%zu bitwise=1\n",
                F::name, dtype, rows, columns, offset, int(exhaustive), count);
    Check(cudaFree(x)); Check(cudaFree(a)); Check(cudaFree(b));
}

template<class T, ggml_type type> void Cases(const char *dtype) {
    for (int rows : {4, 12}) for (int columns : {256, 768, 1024, 3072, 5120})
        for (int offset : {0, 1}) Run<T, type>(dtype, rows, columns, offset);
    Run<T, type>(dtype, 2304, 5120, 0);
    Run<T, type>(dtype, 4608, 5120, 0);
    const int exhaustiveRows = Format<type>::exhaustiveBlocks / 2;
    for (int offset : {0, 1}) Run<T, type>(dtype, exhaustiveRows, 2048, offset, true);
    auto convert = Dispatch<T>(type);
    convert(nullptr, nullptr, 0, 256, 0);
    convert(nullptr, nullptr, 4, 0, 0);
    Check(cudaGetLastError());
}

int main() {
    try {
        Cases<__nv_bfloat16, GGML_TYPE_IQ2_XS_R4>("bf16");
        Cases<half, GGML_TYPE_IQ2_XS_R4>("fp16");
        Cases<float, GGML_TYPE_IQ2_XS_R4>("fp32");
        Cases<__nv_bfloat16, GGML_TYPE_IQ2_XXS_R4>("bf16");
        Cases<half, GGML_TYPE_IQ2_XXS_R4>("fp16");
        Cases<float, GGML_TYPE_IQ2_XXS_R4>("fp32");
        std::puts("ALL PASS: IQ2_XS_R4 and IQ2_XXS_R4 values and output guards are bitwise equal");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
