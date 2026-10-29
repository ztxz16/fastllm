#pragma once
namespace {
__global__ void WriteContextIndexer(FastllmNaiveKV cache, const BF16 *input,
        int width, int count, int start, const int *live, int rank, int ranks, int capacity) {
    int token = blockIdx.x;
    int first = live ? *live - count : start;
    if (first < 0 || first + count > capacity) return;
    const int global = first + token, page = cache.pageLen * ranks;
    if (global % page / cache.pageLen != rank) return;
    const int local = global / page * cache.pageLen + global % cache.pageLen;
    auto target = cache.Row(local, width);
    if (cache.scaledIndexer) {
        for (int i = threadIdx.x; i < 36; i += blockDim.x)
            ((unsigned *)target.data)[i] = ((const unsigned *)input)[token * 36 + i];
    } else {
        FastllmNaiveKVWriteRow out{(void *)target.data, cache.fp8};
        for (int i = threadIdx.x; i < width; i += blockDim.x) out[i] = input[token * width + i];
    }
}
__global__ void GatherContextIndexer(FastllmNaiveKV cache, uint4 *target,
        int width, int keys, int rank, int ranks, int localRows) {
    int vectors = width * (cache.fp8 ? 1 : 2) / 16;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localRows * vectors) return;
    int row = i / vectors, page = cache.pageLen * ranks;
    int global = row / cache.pageLen * page + rank * cache.pageLen + row % cache.pageLen;
    uint4 value{};
    if (global < keys) value = ((const uint4 *)cache.Row(row, width).data)[i % vectors];
    target[(size_t)rank * localRows * vectors + i] = value;
}
__global__ void UnpackContextIndexer(const uint4 *source, uint4 *target,
        int keys, int vectors, int localPage, int ranks, int localRows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= keys * vectors) return;
    int row = i / vectors, page = localPage * ranks;
    int rank = row % page / localPage;
    int local = row / page * localPage + row % localPage;
    target[i] = source[((size_t)rank * localRows + local) * vectors + i % vectors];
}
}
void FastllmCudaNaiveContextIndexerWrite(fastllm::Data &cache, const fastllm::Data &input,
        int start, const fastllm::Data *live, int rank, int ranks) {
    using namespace fastllm;
    AssertInFastLLM(cache.isPagedKVCache && cache.pageLen > 0 && ranks > 1 && rank >= 0 && rank < ranks &&
        input.dataType == BFLOAT16 && input.dims.size() == 3 && cache.dims.size() == 3 &&
        input.kvCacheFp8Indexer == cache.kvCacheFp8Indexer &&
        input.dims[2] == (cache.kvCacheFp8Indexer ? 72 : cache.dims[2]), "Invalid context-sharded Indexer write.");
    WriteContextIndexer<<<input.dims[1], 128>>>(FastllmNaiveKV(cache), (const BF16 *)input.cudaData,
        cache.dims[2], input.dims[1], start, live ? (const int *)live->cudaData : nullptr,
        rank, ranks, cache.expansionDims[1] * ranks);
    CheckLaunch();
}
void FastllmCudaNaiveContextIndexerGather(const fastllm::Data &cache, int keys,
        int rank, int ranks, fastllm::Data &exchange) {
    using namespace fastllm;
    const int page = cache.pageLen * ranks;
    const int localRows = (keys + page - 1) / page * cache.pageLen;
    Output(exchange, cache.dataType, {ranks, localRows, cache.dims[2]});
    exchange.kvCacheFp8Indexer = cache.kvCacheFp8Indexer;
    const int vectors = cache.dims[2] * cache.unitSize / 16;
    GatherContextIndexer<<<(localRows * vectors + 255) / 256, 256>>>(FastllmNaiveKV(cache),
        (uint4 *)exchange.cudaData, cache.dims[2], keys, rank, ranks, localRows);
    CheckLaunch();
}
void FastllmCudaNaiveContextIndexerUnpack(const fastllm::Data &exchange, int keys,
        int page, int ranks, fastllm::Data &full) {
    Output(full, exchange.dataType, {1, keys, exchange.dims[2]});
    full.kvCacheFp8Indexer = exchange.kvCacheFp8Indexer;
    const int vectors = exchange.dims[2] * exchange.unitSize / 16;
    UnpackContextIndexer<<<(keys * vectors + 255) / 256, 256>>>((const uint4 *)exchange.cudaData,
        (uint4 *)full.cudaData, keys, vectors, page / ranks, ranks, exchange.dims[1]);
    CheckLaunch();
}
