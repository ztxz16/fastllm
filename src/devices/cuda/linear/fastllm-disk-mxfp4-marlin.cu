#include "fastllm-cuda.cuh"

#include <algorithm>
#include <map>
#include <stdexcept>

#define MARLIN_NAMESPACE_NAME fastllm_disk_mxfp4
#include "marlin_dense_fp8/kernel.h"
#include "marlin_dense_fp8/marlin_template.h"

namespace {
using Kernel = void (*)(MARLIN_KERNEL_PARAMS);
thread_local FastllmCudaDiskMxfp4Scope *diskScope = nullptr;

// Canonical RAM records contain packed values followed by E8M0 scales.
// Convert only the streamed GPU copy, including the gate/up row permutation.
template<int width>
__global__ void PackNative(const uint8_t *source, uint8_t *destination,
                           int rows, int blocks, bool cross) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * width;
    int pitch = blocks * 17;
    if (offset >= pitch) return;
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        int original = cross ? row / 2 + (row % 2) * (rows / 2) : row;
        int record = offset / 17, byte = offset % 17;
        uint32_t value = 0;
#pragma unroll
        for (int j = 0; j < width; ++j) {
            int b = byte + j, block = record;
            if (b >= 17) { b -= 17; ++block; }
            size_t src = size_t(original) * blocks + block;
            uint8_t x = b < 16 ? source[src * 16 + b] :
                source[size_t(rows) * blocks * 16 + src];
            value |= uint32_t(x) << (j * 8);
        }
        if constexpr (width == 4)
            reinterpret_cast<uint32_t *>(destination + size_t(row) * pitch)[offset / 4] = value;
        else destination[size_t(row) * pitch + offset] = value;
    }
}

// Routes of one expert are ordered by token. One thread owns each output
// element and handles duplicate routes in order, without atomics or races.
__global__ void ReduceRoutes(const __nv_bfloat16 *values, float *sum,
                             const int *indices, int rows, int columns) {
    int row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns || (row > 0 && indices[row - 1] == indices[row])) return;
    size_t dst = size_t(indices[row]) * columns + col;
    float value = sum[dst];
    for (int r = row; r < rows && indices[r] == indices[row]; ++r)
        value += __bfloat162float(values[size_t(r) * columns + col]);
    sum[dst] = value;
}

void Check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("disk MXFP4 Marlin: ") + cudaGetErrorString(status));
}

Kernel Select(int rows, bool narrow) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
    return nullptr;
#else
#define MXFP4(TM, M8, TN, TK) \
    return MARLIN_NAMESPACE_NAME::Marlin< \
        vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), \
        vllm::kFE8M0fnu.id(), 128, TM, TN, TK, M8, 2, 2, false, true, false, false, true>
    if (narrow) {
        if (rows <= 8) MXFP4(1, true, 4, 8);
        if (rows <= 16) MXFP4(1, false, 4, 8);
        if (rows <= 32) MXFP4(2, false, 4, 8);
        if (rows <= 48) MXFP4(3, false, 4, 8);
        MXFP4(4, false, 4, 8);
    }
    if (rows <= 8) MXFP4(1, true, 8, 4);
    if (rows <= 16) MXFP4(1, false, 8, 4);
    if (rows <= 32) MXFP4(2, false, 8, 4);
    if (rows <= 48) MXFP4(3, false, 8, 4);
    MXFP4(4, false, 8, 4);
#undef MXFP4
#endif
}

struct DeviceConfig {
    int sms = 0, shared = 0;
};

const DeviceConfig &Config() {
    static thread_local std::map<int, DeviceConfig> devices;
    int device = 0;
    Check(cudaGetDevice(&device));
    auto found = devices.find(device);
    if (found != devices.end()) return found->second;
    cudaDeviceProp prop{};
    Check(cudaGetDeviceProperties(&prop, device));
    DeviceConfig config;
#ifndef CUDA_NO_TENSOR_CORE
    if (prop.major == 7 && prop.minor == 5) {
        config.sms = prop.multiProcessorCount;
        config.shared = prop.sharedMemPerBlockOptin;
        for (bool narrow : {false, true}) for (int rows : {8, 16, 32, 48, 64})
            Check(cudaFuncSetAttribute(Select(rows, narrow),
                cudaFuncAttributeMaxDynamicSharedMemorySize, config.shared));
    }
#endif
    return devices.emplace(device, config).first->second;
}

bool Supported(const fastllm::Data &weight) {
    return weight.dataType == fastllm::DataType::NVFP4_BLOCK_32_E8M0 &&
        weight.dataDevice == fastllm::DataDevice::CUDA && weight.cudaData &&
        weight.dims.size() == 2 && weight.dims[0] > 0 && weight.dims[1] > 0 &&
        weight.dims[0] % 64 == 0 && weight.dims[1] % 64 == 0 &&
        (weight.dims[0] % 128 == 0 || weight.dims[1] % 128 == 0) &&
        Config().sms > 0;
}

// Permute a transposed scale index into the Marlin 64-element tile.
__device__ int ScaleIndex(int flat) {
    int p = ((flat & 7) << 3) | ((flat & 63) >> 3);
    p = (p & ~3) | ((p & 1) << 1) | ((p & 2) >> 1);
    return (flat & ~63) + p;
}

// One thread owns a whole Marlin uint32, so no nibble stores can race.
__global__ void Pack(const uint8_t *source, uint32_t *packed,
                     int columns, int inner) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= columns * (inner / 8)) return;
    int tile = id / 128, word = id % 128, lane = word / 4, warp = word % 4;
    int n = (tile % (columns / 64)) * 64 + warp * 16 + lane / 4;
    int k = (tile / (columns / 64)) * 16 + (lane % 4) * 2;
    uint32_t value = 0;
    for (int i = 0; i < 8; ++i) {
        int v = (i % 4) * 2 + i / 4;
        int row = n + (v >= 4 ? 8 : 0);
        int col = k + (v % 4 >= 2 ? 8 : 0) + (v & 1);
        uint8_t byte = source[((size_t)row * (inner / 32) + col / 32) * 17 + (col % 32) / 2];
        value |= uint32_t((byte >> ((col & 1) * 4)) & 15) << (i * 4);
    }
    packed[id] = value;
}

__global__ void PackScales(const uint8_t *source, uint8_t *scales,
                           int columns, int groups, int *invalid) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= columns * groups) return;
    int row = id / groups, group = id % groups;
    uint8_t scale = source[(size_t)id * 17 + 16];
    // Conservative lossless FP16 range, including every FP4 magnitude.
    // Out-of-range weights retain the ordinary BF16 implementation.
    if (scale < 113 || scale > 140) atomicExch(invalid, 1);
    scales[ScaleIndex(group * columns + row)] = scale;
}

__global__ void Unpack(const uint32_t *packed, const uint8_t *scales,
                       uint8_t *destination, int columns, int groups) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= columns * groups) return;
    int row = id / groups, group = id % groups;
    for (int j = 0; j < 16; ++j) {
        int col = group * 32 + j * 2;
        int tile = (col / 16) * (columns / 64) + row / 64;
        int lane = (row % 8) * 4 + (col % 8) / 2;
        uint32_t value = packed[tile * 128 + lane * 4 + (row % 64) / 16];
        int shift = ((row % 16 >= 8 ? 2 : 0) + (col % 16 >= 8 ? 1 : 0)) * 4;
        destination[(size_t)id * 17 + j] =
            ((value >> shift) & 15) | (((value >> (shift + 16)) & 15) << 4);
    }
    destination[(size_t)id * 17 + 16] = scales[ScaleIndex(group * columns + row)];
}

__global__ void ConvertInput(const __nv_bfloat16 *input, half *output,
                             size_t count, int *invalid) {
    size_t id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= count) return;
    float x = __bfloat162float(input[id]);
    half h = __float2half_rn(x);
    // A representable BF16 value converts exactly to FP16. Reject overflow,
    // underflow, NaN and infinity rather than silently changing the operands.
    if (!isfinite(x) || __half2float(h) != x) atomicExch(invalid, 1);
    output[id] = h;
}

struct Scratch {
    void *data;
    explicit Scratch(size_t bytes) : data(FastllmCudaMalloc(bytes)) {
        if (!data) FastllmCudaClearThreadError();
    }
    Scratch(const Scratch &) = delete;
    Scratch &operator=(const Scratch &) = delete;
    ~Scratch() {
        // The GEMM can still be using its reduction buffer when the host
        // returns. Do not expose it to another stream through the free pool.
        if (data && !FastllmCudaFreeAfterCurrentThreadStream(data)) {
            cudaStreamSynchronize(cudaStreamPerThread);
            FastllmCudaFree(data);
        }
    }
};

bool Invalid(int *flag) {
    int invalid = 0;
    Check(cudaMemcpyAsync(&invalid, flag, sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    return invalid != 0;
}
} // namespace

void FastllmCudaPackDiskMxfp4(const void *source, void *destination,
        int rows, int columns, bool cross, void *stream) {
    const int pitch = (columns / 32) * 17;
    if (pitch % 4 == 0)
        PackNative<4><<<dim3((pitch / 4 + 255) / 256, std::min(rows, 65535)), 256, 0,
            static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t *>(source),
                static_cast<uint8_t *>(destination), rows, columns / 32, cross);
    else
        PackNative<1><<<dim3((pitch + 255) / 256, std::min(rows, 65535)), 256, 0,
            static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t *>(source),
                static_cast<uint8_t *>(destination), rows, columns / 32, cross);
    Check(cudaGetLastError());
}

void FastllmCudaReduceDiskMxfp4(const void *values, float *sum,
        const int *indices, int rows, int columns) {
    ReduceRoutes<<<dim3((columns + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(values), sum, indices, rows, columns);
    Check(cudaGetLastError());
}

FastllmCudaDiskMxfp4Scope::FastllmCudaDiskMxfp4Scope(
        const fastllm::Data *gate, const fastllm::Data *down)
    : gate(gate), down(down), previous(diskScope) { diskScope = this; }

FastllmCudaDiskMxfp4Scope::~FastllmCudaDiskMxfp4Scope() { diskScope = previous; }

extern "C" bool FastllmCudaDiskMxfp4Enabled(const fastllm::Data &weight) {
    return diskScope && (diskScope->gate == &weight || diskScope->down == &weight);
}

extern "C" bool FastllmCudaSetDiskMxfp4MarlinLayout(fastllm::Data &weight, bool enable) {
    if (weight.dataType != fastllm::DataType::NVFP4_BLOCK_32_E8M0) return false;
    if (weight.IsRepacked == enable) return enable;
    if (enable && !Supported(weight)) return false;
    const int columns = weight.dims[0], inner = weight.dims[1], groups = inner / 32;
    const size_t bytes = weight.GetBytes(), weightBytes = size_t(columns) * inner / 2;
    Scratch scratch(bytes + sizeof(int));
    if (!scratch.data) {
        if (enable) return false; // Repacking is optional; native data is intact.
        throw std::runtime_error("disk MXFP4: insufficient scratch to restore native layout");
    }
    auto *temporary = static_cast<uint8_t *>(scratch.data);
    auto *flag = reinterpret_cast<int *>(temporary + bytes);
    if (enable) {
        Check(cudaMemsetAsync(flag, 0, sizeof(int), cudaStreamPerThread));
        Pack<<<(columns * (inner / 8) + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            static_cast<const uint8_t *>(weight.cudaData), reinterpret_cast<uint32_t *>(temporary), columns, inner);
        PackScales<<<(columns * groups + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            static_cast<const uint8_t *>(weight.cudaData), temporary + weightBytes, columns, groups, flag);
        Check(cudaGetLastError());
        if (Invalid(flag)) return false; // The original allocation is untouched.
    } else {
        Unpack<<<(columns * groups + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            static_cast<const uint32_t *>(weight.cudaData),
            static_cast<const uint8_t *>(weight.cudaData) + weightBytes, temporary, columns, groups);
        Check(cudaGetLastError());
    }
    Check(cudaMemcpyAsync(weight.cudaData, temporary, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    weight.IsRepacked = enable;
    return enable;
}

extern "C" bool FastllmCudaTryDiskMxfp4Marlin(
        const void *input, fastllm::Data &weight, void *output, int rows) {
    auto fallback = [&]() {
        FastllmCudaSetDiskMxfp4MarlinLayout(weight, false);
        return false;
    };
    if (rows < 32 || !input || !output || !Supported(weight)) return fallback();
    const int columns = weight.dims[0], inner = weight.dims[1];
    const auto &config = Config();
    const size_t inputBytes = (size_t(rows) * inner * sizeof(half) + 15) & ~size_t(15);
    const size_t reduceBytes = size_t(config.sms) * 64 * 256 * sizeof(float);
    const size_t lockBytes = size_t(config.sms) * 4 * sizeof(int);
    Scratch scratch(inputBytes + reduceBytes + lockBytes + sizeof(int));
    if (!scratch.data) return fallback();
    auto *a = static_cast<half *>(scratch.data);
    auto *reduce = reinterpret_cast<int4 *>(static_cast<uint8_t *>(scratch.data) + inputBytes);
    auto *locks = reinterpret_cast<int *>(reinterpret_cast<uint8_t *>(reduce) + reduceBytes);
    auto *flag = reinterpret_cast<int *>(reinterpret_cast<uint8_t *>(locks) + lockBytes);
    Check(cudaMemsetAsync(locks, 0, lockBytes + sizeof(int), cudaStreamPerThread));
    ConvertInput<<<(size_t(rows) * inner + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(input), a, size_t(rows) * inner, flag);
    Check(cudaGetLastError());
    if (Invalid(flag)) return fallback();
    if (!FastllmCudaSetDiskMxfp4MarlinLayout(weight, true)) return false;
    for (int row = 0; row < rows;) {
        int count = rows - row;
        if (count >= 64) count = std::min(count / 64, 128) * 64;
        Select(count, columns % 128 != 0)<<<config.sms, 128, config.shared, cudaStreamPerThread>>>(
            reinterpret_cast<const int4 *>(a + size_t(row) * inner),
            static_cast<const int4 *>(weight.cudaData),
            reinterpret_cast<int4 *>(static_cast<uint16_t *>(output) + size_t(row) * columns),
            reduce, nullptr, nullptr,
            reinterpret_cast<const int4 *>(static_cast<const uint8_t *>(weight.cudaData) + size_t(columns) * inner / 2),
            nullptr, nullptr, nullptr, inner / 32, count, columns, inner, inner, locks,
            false, false, true, config.shared);
        Check(cudaGetLastError());
        row += count;
    }
    return true;
}
