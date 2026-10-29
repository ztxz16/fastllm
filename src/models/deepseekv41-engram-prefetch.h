#pragma once

#include "devices/disk/diskdevice.h"
#include "utils.h"
#include <cmath>
#include <limits>

namespace fastllm {
namespace v41_engram_detail {

// CPU-only gather for a request-owned prefetch worker. It never enters the
// executor or the shared NUMA MoE conversion pool. Tickets own both I/O buffers.
inline void GatherNativeRows(const uint8_t *values, const uint8_t *scales,
        DiskEmbeddingRowReader *valueReader, DiskEmbeddingRowReader *scaleReader,
        int64_t tableRows, int dim, int scaleBlock,
        const std::vector<int64_t> &rows, uint16_t *dst) {
    AssertInFastLLM(dim > 0 && scaleBlock > 0 && dim % scaleBlock == 0,
                    "DeepSeekV41: invalid native Engram row shape.");
    const bool disk = valueReader != nullptr && scaleReader != nullptr;
    AssertInFastLLM(disk || (values != nullptr && scales != nullptr),
                    "DeepSeekV41: native Engram prefetch storage unavailable.");
    const int scaleCols = dim / scaleBlock;
    std::vector<int32_t> indices;
    indices.reserve(rows.size());
    for (int64_t row : rows) {
        AssertInFastLLM(row >= 0 && row < tableRows && row <= INT32_MAX,
                        "DeepSeekV41: native Engram hash out of range.");
        indices.push_back(int32_t(row));
    }
    DiskEmbeddingRowReader::Ticket valueRead, scaleRead;
    if (disk) {
        // Submit both reads before waiting; they have separate bounded pools.
        valueRead = valueReader->ReadAsync(indices);
        scaleRead = scaleReader->ReadAsync(indices);
        const auto &rawValues = valueRead.values.get();
        const auto &rawScales = scaleRead.values.get();
        AssertInFastLLM(rawValues.size() == rows.size() * dim &&
                        rawScales.size() == rows.size() * scaleCols,
                        "DeepSeekV41: native Engram prefetch row size mismatch.");
        values = rawValues.data();
        scales = rawScales.data();
    }
    static const FP8E4M3ToFP32Manager fp8;
    for (size_t i = 0; i < rows.size(); ++i) {
        const size_t sourceRow = disk ? i : size_t(rows[i]);
        const uint8_t *src = values + sourceRow * dim;
        const uint8_t *sc = scales + sourceRow * scaleCols;
        uint16_t *out = dst + i * dim;
        for (int block = 0; block < scaleCols; ++block) {
            // Match the synchronous FP8 multiply and BF16 RNE exactly, including
            // the scale-zero subnormal and reserved UE8M0 NaN code.
            const float scale = sc[block] == 0xff
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(sc[block]) - 127);
            for (int j = block * scaleBlock; j < (block + 1) * scaleBlock; ++j) {
                out[j] = Float32ToBFloat16RNEBits(fp8.dict[src[j]] * scale);
            }
        }
    }
}

} // namespace v41_engram_detail
} // namespace fastllm
