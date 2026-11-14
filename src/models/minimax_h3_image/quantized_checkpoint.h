#pragma once
#include "json11.hpp"
#include "quantized_partition.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fastllm::minimax_h3_image::detail {
inline void requireCheckpoint(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
// Bounded header parsing and payload transfers shared by the native INT8 and
// NVFP4/FP8 readers. Validate dtype, integral shapes/offsets and exact byte counts
// before allocating device storage or seeking into an untrusted checkpoint.
class QuantizedCheckpoint {
  public:
    std::ifstream file;
    json11::Json header;
    uint64_t base = 0, length = 0;
    explicit QuantizedCheckpoint(const std::string &path) : file(path, std::ios::binary) {
        requireCheckpoint(bool(file), "cannot open quantized checkpoint: " + path);
        file.seekg(0, std::ios::end);
        const auto end = file.tellg();
        requireCheckpoint(end >= 8, "truncated quantized checkpoint");
        length = uint64_t(end);
        file.seekg(0);
        uint64_t size = 0;
        file.read(reinterpret_cast<char *>(&size), 8);
        requireCheckpoint(size > 0 && size <= (16u << 20) && size < length - 8,
                          "invalid quantized header size");
        std::string data(size, '\0'), error;
        file.read(data.data(), size);
        header = json11::Json::parse(data, error);
        requireCheckpoint(bool(file) && error.empty() && header.is_object(), "invalid quantized header JSON");
        base = size + 8;
    }
    json11::Json descriptor(const std::string &name, const std::string &dtype) const {
        auto d = header[name];
        requireCheckpoint(d.is_object() && d["dtype"].string_value() == dtype,
                          "invalid quantized tensor: " + name);
        auto offsets = d["data_offsets"].array_items();
        requireCheckpoint(offsets.size() == 2 && integer(offsets[0], length - base) &&
                              integer(offsets[1], length - base) &&
                              offsets[1].number_value() >= offsets[0].number_value(),
                          "invalid quantized offsets: " + name);
        requireCheckpoint(d["shape"].is_array() && d["shape"].array_items().size() <= 8,
                          "invalid tensor rank: " + name);
        uint64_t bytes = dtype == "F32" ? 4 : (dtype == "BF16" ? 2 : 1);
        for (const auto &dimension : d["shape"].array_items()) {
            requireCheckpoint(integer(dimension, std::numeric_limits<int>::max()) &&
                                  dimension.number_value() > 0 &&
                                  bytes <= (length - base) / uint64_t(dimension.number_value()),
                              "invalid tensor shape: " + name);
            bytes *= uint64_t(dimension.number_value());
        }
        requireCheckpoint(double(bytes) == offsets[1].number_value() - offsets[0].number_value(),
                          "quantized tensor byte size mismatch: " + name);
        return d;
    }
    json11::Json desc(const std::string &name, const std::string &dtype, size_t bytes) const {
        auto d = descriptor(name, dtype);
        requireCheckpoint(payloadBytes(d) == bytes, "quantized tensor byte size mismatch: " + name);
        return d;
    }
    static size_t payloadBytes(const json11::Json &d) {
        return size_t(d["data_offsets"][1].number_value() - d["data_offsets"][0].number_value());
    }
    void readRange(const json11::Json &d, size_t offset, void *dest, size_t bytes) {
        requireCheckpoint(offset <= payloadBytes(d) && bytes <= payloadBytes(d) - offset,
                          "invalid tensor read range");
        file.clear();
        file.seekg(base + uint64_t(d["data_offsets"][0].number_value()) + offset);
        file.read(static_cast<char *>(dest), bytes);
        requireCheckpoint(bool(file), "truncated quantized payload");
    }
    void read(const json11::Json &d, void *dest, size_t bytes) {
        requireCheckpoint(bytes == payloadBytes(d), "quantized tensor byte size mismatch");
        readRange(d, 0, dest, bytes);
    }
    json11::Json marker(const std::string &name) {
        auto d = descriptor(name, "U8");
        size_t bytes = payloadBytes(d);
        requireCheckpoint(bytes > 0 && bytes < 4096, "invalid quantization marker size");
        std::string text(bytes, '\0'), error;
        read(d, text.data(), bytes);
        auto result = json11::Json::parse(text, error);
        requireCheckpoint(error.empty() && result.is_object(), "invalid quantization marker JSON");
        return result;
    }
    // Slice the logical [Q, K, V] layout before uploading. Image checkpoints
    // can store three tensors or interleave whole heads in a single tensor.
    void uploadByteMatrices(const std::vector<json11::Json> &matrices, void *dest,
                            const ProjectionSlice &slice, bool interleavedQkv = false) {
        if (slice.size > 1) {
            if (matrices.size() == 1 && !interleavedQkv) {
                uploadSlice(matrices.front(), dest, slice, size_t(slice.fullCols), size_t(slice.cols),
                            size_t(slice.firstColumn()));
                return;
            }
            std::vector<size_t> offsets{0};
            for (const auto &matrix : matrices) {
                requireCheckpoint(matrix["shape"][1].int_value() == slice.fullCols,
                                  "inconsistent quantized matrix width");
                offsets.push_back(offsets.back() + payloadBytes(matrix));
            }
            requireCheckpoint(offsets.back() == size_t(slice.fullRows) * slice.fullCols,
                              "quantized matrix partition size mismatch");
            requireCheckpoint(!interleavedQkv ||
                                  (matrices.size() == 1 && slice.groups == 3 && slice.fullRows % 384 == 0),
                              "invalid interleaved QKV partition");
            slice.transfer([&](size_t offset, void *host, size_t bytes) {
                if (interleavedQkv) {
                    const size_t row = offset / slice.fullCols, width = slice.fullRows / 3;
                    const size_t source = (row % width / 128) * 384 + row / width * 128 + row % 128;
                    offset = source * slice.fullCols + offset % slice.fullCols;
                }
                const size_t index = std::upper_bound(offsets.begin(), offsets.end(), offset) - offsets.begin() - 1;
                readRange(matrices.at(index), offset - offsets[index], host, bytes);
            }, [&](size_t offset, const void *host, size_t bytes) {
                uploadBuffer(static_cast<char *>(dest) + offset, host, bytes);
            }, size_t(slice.fullCols), size_t(slice.cols), size_t(slice.firstColumn()));
            return;
        }
        size_t offset = 0;
        for (const auto &matrix : matrices) {
            const size_t bytes = size_t(matrix["shape"][0].int_value()) * slice.cols;
            upload(matrix, static_cast<char *>(dest) + offset, bytes);
            offset += bytes;
        }
    }
    void uploadSlice(const json11::Json &d, void *dest, const ProjectionSlice &slice,
                     size_t sourceStride, size_t targetStride, size_t firstByte,
                     int rowUnit = 1, bool positiveE4m3 = false) {
        if (slice.size == 1) {
            upload(d, dest, size_t(slice.rows/rowUnit)*targetStride, positiveE4m3);
            return;
        }
        slice.transfer([&](size_t offset, void *host, size_t bytes) { readRange(d,offset,host,bytes); },
            [&](size_t offset, const void *host, size_t bytes) {
                uploadBuffer(static_cast<char *>(dest) + offset, host, bytes);
            }, sourceStride,targetStride,firstByte,rowUnit,positiveE4m3);
    }
    void upload(const json11::Json &d, void *dest, size_t bytes, bool positiveE4m3 = false) {
        requireCheckpoint(payloadBytes(d) == bytes, "quantized matrix byte size mismatch");
        std::vector<char> buffer(std::min(bytes, size_t(8) << 20));
        for (size_t offset = 0; offset < bytes;) {
            const size_t count = std::min(bytes - offset, buffer.size());
            readRange(d, offset, buffer.data(), count);
            if (positiveE4m3)
                for (size_t i = 0; i < count; ++i)
                    requireCheckpoint(uint8_t(buffer[i]) <= 126, "invalid NVFP4 block scale");
            uploadBuffer(static_cast<char *>(dest) + offset, buffer.data(), count);
            offset += count;
        }
    }

  private:
    // Synchronize before reusing the bounded host staging buffer.
    static void uploadBuffer(void *dest, const void *host, size_t bytes) {
        auto status = cudaMemcpyAsync(dest, host, bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
        if (status == cudaSuccess)
            status = cudaStreamSynchronize(cudaStreamPerThread);
        if (status != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(status));
    }
    static bool integer(const json11::Json &v, uint64_t upper) {
        return v.is_number() && std::isfinite(v.number_value()) && v.number_value() >= 0 &&
               std::floor(v.number_value()) == v.number_value() && v.number_value() <= double(upper);
    }
};
} // namespace fastllm::minimax_h3_image::detail
