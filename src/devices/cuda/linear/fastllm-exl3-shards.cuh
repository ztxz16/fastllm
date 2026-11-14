#pragma once
// Lossless row-shard restore. The CPU arena stores each trellis and scale byte
// once; cached GPU kernels continue to consume the original EXL3 layout.
namespace fastllm_exl3_shards {
__device__ inline void Restore(int rows, int columns, int bits, int parts,
        void *const *pointers, int shards, const uint8_t *contiguous, uint8_t *output) {
    const int n = rows / parts;
    const size_t trellis = size_t(n) * columns * bits / 16;
    const size_t partBytes = trellis + size_t(n + columns) * 2;
    const size_t line = size_t(n) * bits, shardLine = line / shards;
    const size_t shardPart = partBytes / shards, shardBytes = shardPart * parts;
    const size_t inputScales = size_t(columns) * 2, outputScales = size_t(n) * 2;
    for (size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 16;
         i < partBytes * parts; i += size_t(gridDim.x) * blockDim.x * 16) {
        const int part = i / partBytes;
        const size_t byte = i % partBytes;
        int node; size_t local;
        if (byte < trellis) {
            node = (byte % line) / shardLine;
            local = (byte / line) * shardLine + byte % shardLine;
        } else if (byte < trellis + inputScales) {
            const size_t at = byte - trellis, perNode = inputScales / shards;
            node = at / perNode; local = trellis / shards + at % perNode;
        } else {
            const size_t at = byte - trellis - inputScales, perNode = outputScales / shards;
            node = at / perNode; local = (trellis + inputScales) / shards + at % perNode;
        }
        const auto *source = contiguous ? contiguous + node * shardBytes :
            static_cast<const uint8_t *>(pointers[node]);
        *reinterpret_cast<uint4 *>(output + i) =
            *reinterpret_cast<const uint4 *>(source + part * shardPart + local);
    }
}
struct Matrix {
    int rows, columns, bits, parts, shards;
    void *pointers[32];
};
static __global__ void MatrixRestore(Matrix matrix, uint8_t *destination) {
    Restore(matrix.rows, matrix.columns, matrix.bits, matrix.parts,
        matrix.pointers, matrix.shards, nullptr, destination);
}
}
