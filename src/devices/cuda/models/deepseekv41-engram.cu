#include "devices/cuda/fastllm-v41-engram.h"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>

namespace {
constexpr int kMaxDevices = 16;
struct EngramView {
    const uint8_t *weights[kMaxDevices]{};
    const uint8_t *scales[kMaxDevices]{};
    int64_t rowsPerShard = 0;
    int dim = 0, scaleBlock = 0;
};
struct EngramTable {
    EngramView view;
    std::vector<int> devices;
    int64_t rows = 0;
    ~EngramTable() {
        int previous = 0;
        cudaGetDevice(&previous);
        // A gather on any rank can still read another rank's shard.
        for (int device : devices) {
            cudaSetDevice(device);
            cudaDeviceSynchronize();
        }
        for (size_t i = 0; i < devices.size(); ++i) {
            cudaSetDevice(devices[i]);
            if (view.weights[i])
                cudaFree(const_cast<uint8_t *>(view.weights[i]));
            if (view.scales[i])
                cudaFree(const_cast<uint8_t *>(view.scales[i]));
        }
        cudaSetDevice(previous);
    }
};

struct RowScratch {
    int device = 0;
    int64_t *rows = nullptr;
    size_t capacity = 0;
    ~RowScratch() {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaStreamSynchronize(cudaStreamPerThread);
        if (rows)
            cudaFree(rows);
        cudaSetDevice(previous);
    }
};

__global__ void GatherEngram(EngramView table, const int64_t *rows, __nv_bfloat16 *output,
                             size_t count) {
    const size_t lookup = blockIdx.x;
    if (lookup >= count)
        return;
    const int64_t row = rows[lookup];
    const int shard = row / table.rowsPerShard;
    const int64_t local = row - shard * table.rowsPerShard;
    const uint8_t *weight = table.weights[shard] + local * table.dim;
    const uint8_t *scale = table.scales[shard] + local * (table.dim / table.scaleBlock);
    for (int col = threadIdx.x; col < table.dim; col += blockDim.x) {
        // Match the existing CPU table decoder, including its finite 0x7f
        // endpoint convention; avoid the hardware E4M3 NaN interpretation.
        const uint8_t packed = weight[col];
        const int exponent4 = (packed >> 3) & 15, mantissa = packed & 7;
        float value = exponent4 ? ldexpf(1.f + mantissa * 0.125f, exponent4 - 7)
                                : ldexpf(float(mantissa), -9);
        if (packed & 128)
            value = -value;
        const uint8_t exponent = scale[col / table.scaleBlock];
        const float factor =
            exponent == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.f, int(exponent) - 127);
        output[lookup * table.dim + col] = __float2bfloat16_rn(value * factor);
    }
}
} // namespace

void *FastllmCudaV41CreateEngram(const std::vector<int> &devices, const uint8_t *weights,
                                 const uint8_t *scales, int64_t rows, int dim, int scaleBlock) {
    if (devices.empty() || devices.size() > kMaxDevices || !weights || !scales || rows <= 0 ||
        dim <= 0 || dim > 65536 || scaleBlock <= 0 || dim % scaleBlock ||
        rows > std::numeric_limits<int64_t>::max() / dim ||
        std::set<int>(devices.begin(), devices.end()).size() != devices.size() ||
        FastllmCudaGraphIsCapturingFast())
        return nullptr;
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess ||
        std::any_of(devices.begin(), devices.end(),
                    [deviceCount](int d) { return d < 0 || d >= deviceCount; })) {
        cudaGetLastError();
        return nullptr;
    }
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess)
        return nullptr;
    auto table = std::make_unique<EngramTable>();
    table->devices = devices;
    table->rows = rows;
    table->view.dim = dim;
    table->view.scaleBlock = scaleBlock;
    table->view.rowsPerShard = (rows + devices.size() - 1) / devices.size();
    bool ok = true;
    // Validate all ranks before allocating. Keep room for weights, KV and graphs.
    for (int device : devices) {
        size_t available = 0, total = 0;
        const size_t bytes = size_t(table->view.rowsPerShard) * (dim + dim / scaleBlock);
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaMemGetInfo(&available, &total) != cudaSuccess || bytes > available ||
            available - bytes < (size_t(16) << 30)) {
            ok = false;
            break;
        }
        for (int peer : devices) {
            if (peer == device)
                continue;
            int accessible = 0;
            if (cudaDeviceCanAccessPeer(&accessible, device, peer) != cudaSuccess || !accessible) {
                ok = false;
                break;
            }
            const auto status = cudaDeviceEnablePeerAccess(peer, 0);
            if (status == cudaErrorPeerAccessAlreadyEnabled)
                cudaGetLastError();
            else if (status != cudaSuccess) {
                cudaGetLastError();
                ok = false;
                break;
            }
        }
        if (!ok)
            break;
    }
    for (size_t i = 0; ok && i < devices.size(); ++i) {
        cudaSetDevice(devices[i]);
        const int64_t first = i * table->view.rowsPerShard;
        const int64_t count =
            std::min(table->view.rowsPerShard, std::max(int64_t(0), rows - first));
        if (!count)
            continue;
        void *w = nullptr, *s = nullptr;
        ok = cudaMalloc(&w, size_t(count) * dim) == cudaSuccess;
        table->view.weights[i] = static_cast<uint8_t *>(w);
        if (ok)
            ok = cudaMalloc(&s, size_t(count) * (dim / scaleBlock)) == cudaSuccess;
        table->view.scales[i] = static_cast<uint8_t *>(s);
        if (ok)
            ok = cudaMemcpy(w, weights + first * dim, size_t(count) * dim,
                            cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok)
            ok = cudaMemcpy(s, scales + first * (dim / scaleBlock),
                            size_t(count) * (dim / scaleBlock),
                            cudaMemcpyHostToDevice) == cudaSuccess;
    }
    cudaSetDevice(previous);
    if (!ok) {
        cudaGetLastError();
        return nullptr;
    }
    return table.release();
}

void FastllmCudaV41FreeEngram(void *handle) {
    delete static_cast<EngramTable *>(handle);
}

bool FastllmCudaV41GatherEngramDevice(void *handle, const int64_t *rows, size_t count,
                                      fastllm::Data &output) {
    const auto *table = static_cast<const EngramTable *>(handle);
    const int device = FastllmCudaGetDevice();
    if (!table || !rows || !count || count > 1024 * 1024 ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        output.dataDevice != fastllm::DataDevice::CUDA || !output.cudaData ||
        output.Count(0) != count * table->view.dim ||
        output.dataDeviceIds != std::vector<int>{device} ||
        std::find(table->devices.begin(), table->devices.end(), device) == table->devices.end())
        return false;
    GatherEngram<<<count, 256, 0, cudaStreamPerThread>>>(
        table->view, rows, static_cast<__nv_bfloat16 *>(output.cudaData), count);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool FastllmCudaV41GatherEngram(void *handle, const std::vector<int64_t> &rows,
                                fastllm::Data &output) {
    const auto *table = static_cast<const EngramTable *>(handle);
    if (!table || rows.empty() || rows.size() > 1024 * 1024 ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        output.dataDevice != fastllm::DataDevice::CUDA || !output.cudaData ||
        output.Count(0) != rows.size() * table->view.dim || FastllmCudaGraphIsCapturingFast())
        return false;
    const int device = FastllmCudaGetDevice();
    if (std::find(table->devices.begin(), table->devices.end(), device) == table->devices.end() ||
        output.dataDeviceIds != std::vector<int>{device})
        return false;
    for (int64_t row : rows)
        if (row < 0 || row >= table->rows)
            return false;
    static thread_local std::map<int, std::unique_ptr<RowScratch>> scratch;
    auto &buffer = scratch[device];
    if (!buffer) {
        buffer = std::make_unique<RowScratch>();
        buffer->device = device;
    }
    if (buffer->capacity < rows.size()) {
        size_t capacity = 1024;
        while (capacity < rows.size())
            capacity *= 2;
        int64_t *replacement = nullptr;
        if (cudaMalloc(&replacement, capacity * sizeof(int64_t)) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (buffer->rows) {
            cudaStreamSynchronize(cudaStreamPerThread);
            cudaFree(buffer->rows);
        }
        buffer->rows = replacement;
        buffer->capacity = capacity;
    }
    // Synchronous host upload gives the temporary vector a clear lifetime.
    // Table/output accesses remain ordered on the dedicated rank stream.
    if (cudaMemcpy(buffer->rows, rows.data(), rows.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    GatherEngram<<<rows.size(), 256, 0, cudaStreamPerThread>>>(
        table->view, buffer->rows, static_cast<__nv_bfloat16 *>(output.cudaData), rows.size());
    return cudaPeekAtLastError() == cudaSuccess;
}
