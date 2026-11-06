#include "distributed/qwen3_device_weights.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace fastllm {
namespace distributed {
namespace {
void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Qwen3 weights CUDA: ") + cudaGetErrorString(result));
    }
}

/**
 * 按 safetensors 小端字节解码单个浮点数，不进行别名重解释或未对齐读取。
 * @param data 已由调用方验证长度的 F16/BF16/F32 元素。
 * @param dtype 分片读取器验证过的数据类型。
 * @return FP32 数值，后续转换负责拒绝非有限值及 FP16 溢出。
 */
float Decode(const uint8_t *data, const std::string &dtype) {
    uint32_t bits = uint32_t(data[0]) | (uint32_t(data[1]) << 8);
    if (dtype == "F16") {
        uint16_t shortBits = static_cast<uint16_t>(bits);
        __half value;
        memcpy(&value, &shortBits, sizeof(shortBits));
        return __half2float(value);
    }
    if (dtype == "BF16") {
        bits <<= 16;
    } else if (dtype == "F32") {
        bits |= (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
    } else {
        throw std::runtime_error("unsupported Qwen3 source dtype");
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
} // namespace

Qwen3DeviceWeights::Qwen3DeviceWeights(const Qwen3ShardLoader &loader, cudaStream_t stream,
        uint64_t maxTensorBytes, uint64_t maxDeviceBytes) : stream(stream) {
    if (!stream || !maxTensorBytes || !maxDeviceBytes) {
        throw std::invalid_argument("invalid Qwen3 device loading limits/stream");
    }
    CheckCuda(cudaGetDevice(&device));
    const auto &plan = loader.Plan();
    if (plan.tensors.empty() || plan.tensors.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid Qwen3 tensor count");
    }

    // 在任何 payload I/O 或 cudaMalloc 前验证全部配额，避免已知不足时部分加载。
    for (const auto &tensor : plan.tensors) {
        const uint64_t width = tensor.dtype == "F32" ? 4 : 2;
        const uint64_t bytes = (tensor.bytes / width) * 2;
        if (tensor.bytes > maxTensorBytes) {
            throw std::runtime_error("Qwen3 source tensor limit: " + tensor.name);
        }
        if (bytes > maxDeviceBytes - deviceBytes || bytes > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Qwen3 device weight limit: " + tensor.name);
        }
        deviceBytes += bytes;
    }
    size_t freeBytes = 0, totalBytes = 0;
    CheckCuda(cudaMemGetInfo(&freeBytes, &totalBytes));
    if (deviceBytes > freeBytes) {
        throw std::runtime_error("Qwen3 device weights exceed free GPU memory");
    }

    tensors.reserve(plan.tensors.size());
    std::vector<__half> converted(512 * 1024);
    for (const auto &tensor : plan.tensors) {
        try {
            auto source = loader.LoadTensor(tensor.name, maxTensorBytes);
            const size_t width = tensor.dtype == "F32" ? 4 : 2;
            const size_t elements = source.data.size() / width;
            tensors.push_back({tensor.name, tensor.shape, nullptr, uint64_t(elements) * 2});
            auto &target = tensors.back();
            CheckCuda(cudaMalloc(&target.data, static_cast<size_t>(target.bytes)));
            for (size_t begin = 0; begin < elements;) {
                const size_t count = std::min(converted.size(), elements - begin);
                for (size_t i = 0; i < count; ++i) {
                    float value = Decode(source.data.data() + (begin + i) * width, tensor.dtype);
                    if (!std::isfinite(value) || std::fabs(value) > 65504.0f) {
                        throw std::runtime_error("nonfinite/out-of-range FP16 value at element " +
                                                 std::to_string(begin + i));
                    }
                    converted[i] = __float2half_rn(value);
                }
                CheckCuda(cudaMemcpyAsync(static_cast<uint8_t*>(target.data) + begin * 2,
                    converted.data(), count * 2, cudaMemcpyHostToDevice, stream));
                CheckCuda(cudaStreamSynchronize(stream));
                begin += count;
            }
            sourceBytesRead += source.bytesRead;
        } catch (const std::exception &error) {
            throw std::runtime_error("Qwen3 tensor " + tensor.name + ": " + error.what());
        }
    }
}

void Qwen3DeviceWeights::CheckDevice() const {
    int current;
    CheckCuda(cudaGetDevice(&current));
    if (current != device) {
        throw std::runtime_error("Qwen3 weight device migration is not supported");
    }
}

const Qwen3DeviceTensor &Qwen3DeviceWeights::Tensor(uint32_t index) const {
    CheckDevice();
    if (index >= tensors.size()) {
        throw std::runtime_error("Qwen3 tensor index out of range");
    }
    return tensors[index];
}

std::vector<uint8_t> Qwen3DeviceWeights::Read(uint32_t index, uint64_t offset, uint32_t bytes) const {
    const auto &tensor = Tensor(index);
    if (!bytes || bytes > 1024 * 1024 || bytes % 2 || offset % 2 ||
        offset > tensor.bytes || bytes > tensor.bytes - offset) {
        throw std::runtime_error("Qwen3 weight readback out of range");
    }
    std::vector<uint8_t> result(bytes);
    CheckCuda(cudaMemcpyAsync(result.data(), static_cast<const uint8_t*>(tensor.data) + offset,
                              bytes, cudaMemcpyDeviceToHost, stream));
    CheckCuda(cudaStreamSynchronize(stream));
    return result;
}

void Qwen3DeviceWeights::Clear() {
    CheckDevice();
    CheckCuda(cudaStreamSynchronize(stream));
    for (auto &tensor : tensors) {
        CheckCuda(cudaFree(tensor.data));
        tensor.data = nullptr;
    }
    tensors.clear();
    deviceBytes = sourceBytesRead = 0;
}

} // namespace distributed
} // namespace fastllm
