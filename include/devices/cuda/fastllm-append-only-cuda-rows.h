#ifndef FASTLLM_APPEND_ONLY_CUDA_ROWS_H
#define FASTLLM_APPEND_ONLY_CUDA_ROWS_H

#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>

namespace fastllm {
// Request-owned, read-only replicas of a contiguous [1, rows, width] append-only
// tensor. The writer retains the original Data and its device. Call Invalidate
// before rewriting any existing row (including speculative rollback). A fresh
// request/history restore must own a fresh instance, never share reader state.
struct CudaAppendOnlyRows {
    struct Reader {
        int device;
        size_t capacity = 0;
        int validRows = 0;
        void *allocation = nullptr;
        Data view;
        explicit Reader(int device) : device(device) { view.isFake = true; }
        ~Reader() { Release(allocation); view.cudaData = nullptr; }
        void Release(void *ptr) {
            if (!ptr) return;
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr);
            FastllmCudaSetDevice(previous);
        }
    };
    std::map<int, std::unique_ptr<Reader>> readers;
    const Data *owner = nullptr;
    DataType type = DataType::FLOAT32;
    size_t rowBytes = 0;
    int width = 0, observedRows = 0;
    uint64_t copiedBytes = 0, allocations = 0;

    void Invalidate() {
        for (auto &item : readers) item.second->validRows = 0;
        observedRows = 0;
    }

    // nullptr means unsupported/capture/OOM: the caller retains its established
    // migration path. OOM is optional-allocation failure, not a CUDA error.
    Data *Get(Data &source, int target) {
        if (source.dataDevice != DataDevice::CUDA || !source.cudaData ||
            source.multiDeviceData || source.isGGUFData || source.isFake ||
            source.dataDeviceIds.size() != 1 || source.dims.size() != 3 ||
            source.dims[0] != 1 || source.dims[1] <= 0 || source.dims[2] <= 0 ||
            source.strides.size() != 3 || source.strides[2] != 1 ||
            source.strides[1] != (uint64_t)source.dims[2] || source.unitSizeDiv != 1 ||
            (source.dataType != DataType::INT8 && source.dataType != DataType::BFLOAT16 &&
             source.dataType != DataType::FLOAT16 && source.dataType != DataType::FLOAT32) ||
            target < 0 || FastllmCudaGraphIsCapturingFast()) return nullptr;
        const int sourceDevice = source.dataDeviceIds[0];
        if (sourceDevice == target) return &source;
        const int rows = source.dims[1];
        const size_t bytesPerRow = (size_t)source.dims[2] * source.unitSize;
        if (source.expansionBytes < (size_t)rows * bytesPerRow) return nullptr;
        if (owner != &source || type != source.dataType || width != source.dims[2] ||
            rowBytes != bytesPerRow) {
            readers.clear(); owner = &source; type = source.dataType;
            width = source.dims[2]; rowBytes = bytesPerRow; observedRows = 0;
        }
        if (rows < observedRows) Invalidate();
        observedRows = rows;
        auto &slot = readers[target];
        if (!slot) slot = std::make_unique<Reader>(target);
        Reader &r = *slot;
        const int previous = FastllmCudaGetDevice();
        struct Restore { int device; ~Restore() { FastllmCudaSetDevice(device); } } restore{previous};
        FastllmCudaSetDevice(target);
        if (r.capacity < (size_t)rows) {
            const size_t capacity = ((std::max((size_t)rows, r.capacity + r.capacity / 2) + 255) / 256) * 256;
            void *next = nullptr;
            const auto result = FastllmCudaTryMalloc(&next, capacity * rowBytes);
            if (result != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return nullptr;
            // Growth is uncommon. Refresh the live prefix from the canonical
            // writer, rather than transferring unused allocation capacity.
            void *old = r.allocation;
            r.allocation = next; r.capacity = capacity; r.validRows = 0;
            r.Release(old); ++allocations;
        }
        if (r.validRows < rows) {
            const size_t offset = (size_t)r.validRows * rowBytes;
            const size_t bytes = (size_t)(rows - r.validRows) * rowBytes;
            FastllmCudaMemcpyBetweenDevices(target, (uint8_t*)r.allocation + offset,
                                           sourceDevice, (uint8_t*)source.cudaData + offset, bytes);
            if (FastllmCudaGetThreadError()) return nullptr;
            copiedBytes += bytes; r.validRows = rows;
        }
        r.view.dataType = source.dataType; r.view.UpdateUnitSize();
        r.view.dataDevice = DataDevice::CUDA; r.view.dataDeviceIds = {target};
        r.view.cudaData = r.allocation;
        r.view.expansionDims = {1, (int)r.capacity, width};
        r.view.expansionSize = r.capacity * width;
        r.view.expansionBytes = r.capacity * rowBytes;
        // Resize preserves strides for expanded tensors. Initialize the same
        // capacity strides as Data::Expansion, including on the first view.
        r.view.strides = {(uint64_t)r.capacity * width, (uint64_t)width, 1};
        r.view.Resize({1, rows, width});
        return &r.view;
    }
};
}
#endif
