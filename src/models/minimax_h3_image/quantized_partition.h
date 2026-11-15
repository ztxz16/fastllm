#pragma once
#include "models/minimax_h3_image/quantized.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace fastllm::minimax_h3_image::detail {
// Logical projection slicing is shared by byte, nibble and GGUF block readers.
// QKV and gate/value must split each group, not the whole packed output.
struct ProjectionSlice {
    int rows, cols, fullRows, fullCols, groups = 1, rank = 0, size = 1;
    bool rowParallel = false;
    ProjectionSlice(const std::string &name, int n, int k, const ProjectionOptions &o)
        : rows(n), cols(k), fullRows(n), fullCols(k), rank(o.rank), size(o.size) {
        if (n < 1 || k < 1 || size < 1 || rank < 0 || rank >= size)
            throw std::invalid_argument("invalid quantized matrix partition");
        if (size == 1) return;
        auto ends = [&](const char *suffix) {
            const std::string s(suffix);
            return name.size() >= s.size() && name.compare(name.size()-s.size(), s.size(), s) == 0;
        };
        rowParallel = ends(".attn.out_proj") || ends(".mlp.fc2");
        if (ends(".attn.qkv_proj")) groups = 3;
        else if (ends(".mlp.fc1")) groups = 2;
        else if (!rowParallel) throw std::invalid_argument("unknown quantized TP projection: " + name);
        if ((rowParallel ? k : n / groups) % size || n % groups)
            throw std::invalid_argument("quantized projection does not divide TP size");
        if (rowParallel) cols /= size;
        else rows /= size;
    }
    int sourceRow(int row) const {
        return rowParallel || size == 1 ? row :
            (row / (rows/groups)) * (fullRows/groups) + rank * (rows/groups) + row % (rows/groups);
    }
    int firstColumn() const { return rowParallel ? rank * cols : 0; }
    std::vector<float> scales(const std::vector<float> &input) const {
        if (input.size() != size_t(fullRows)) throw std::invalid_argument("projection scale count");
        std::vector<float> result(rows);
        for (int i=0;i<rows;++i) result[i]=input[sourceRow(i)];
        return result;
    }
    // read(offset, host, bytes); offsets and strides are in payload bytes.
    // Transfers remain bounded even when the entire source matrix is huge.
    template<class Read, class Upload>
    void transfer(Read read, Upload upload, size_t sourceStride, size_t targetStride,
                  size_t firstByte, int rowUnit = 1, bool positiveE4m3 = false) const {
        if (rowUnit < 1 || !targetStride || rows % rowUnit || fullRows % rowUnit || sourceRow(0) % rowUnit ||
            firstByte > sourceStride || targetStride > sourceStride-firstByte)
            throw std::invalid_argument("unaligned quantized block partition");
        const size_t maxRows = std::max(size_t(1), (size_t(8)<<20)/targetStride);
        std::vector<char> host(std::min(size_t(rows/rowUnit), maxRows)*targetStride);
        for (int first=0;first<rows/rowUnit;) {
            int count = int(std::min(maxRows, size_t(rows/rowUnit-first)));
            for (int r=0;r<count;++r) {
                const int source = sourceRow((first+r)*rowUnit);
                if (source % rowUnit) throw std::invalid_argument("quantized scale tile alignment");
                read(size_t(source/rowUnit)*sourceStride+firstByte, host.data()+size_t(r)*targetStride,
                     targetStride);
            }
            const size_t bytes=size_t(count)*targetStride;
            if (positiveE4m3)
                for (size_t i=0;i<bytes;++i)
                    if (uint8_t(host[i])>126) throw std::invalid_argument("invalid NVFP4 block scale");
            upload(size_t(first)*targetStride, host.data(), bytes);
            first += count;
        }
    }
};
} // namespace fastllm::minimax_h3_image::detail
