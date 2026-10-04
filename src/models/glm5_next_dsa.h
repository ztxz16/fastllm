#ifndef FASTLLM_GLM5_NEXT_DSA_H
#define FASTLLM_GLM5_NEXT_DSA_H

#include "models/glm5_next.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/glm5-next-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cmath>
#include <limits>

namespace fastllm { namespace glm5_next_detail {

inline void ClearIndexerTensor(Data &data) {
    data.FreeSpace();
    data.expansionDims.clear();
    data.dims.clear();
    data.strides.clear();
}

// All arithmetic below uses existing kernels. FP8 operands are stored exactly
// in BF16 after power-of-two dequantization, for the existing V4.1 BF16 scorer.
inline void RotateAndQuantizeIndexer(Data &input, Data &hadamard, Data &output) {
    if (hadamard.dims.empty()) {
        hadamard.dataType = DataType::FLOAT32;
        hadamard.Resize({128, 128});
        hadamard.Allocate();
        auto *p = reinterpret_cast<float *>(hadamard.cpuData);
        for (int i = 0; i < 128; ++i)
            for (int j = 0; j < 128; ++j) {
                unsigned bits = unsigned(i & j), parity = 0;
                while (bits) { parity ^= bits & 1; bits >>= 1; }
                p[i * 128 + j] = parity ? -1.0f : 1.0f;
            }
    }
    Data fp32, rotated, scales, bf16Scales;
    ToDataType(input, fp32, DataType::FLOAT32);
    const auto dims = input.dims;
    fp32.Reshape({int(input.Count(0) / 128), 128});
    MatMulTransB(fp32, hadamard, rotated, 1.0f / std::sqrt(128.0f));
    ToDataType(rotated, DataType::BFLOAT16);
    rotated.Reshape(dims);
    AssertInFastLLM(FastllmCudaNaiveQuantizeIndexer(
        rotated, output, scales, true), "GLM DSA FP8 quantization failed.");
    ToDataType(scales, bf16Scales, DataType::BFLOAT16);
    MulTo(output, bf16Scales);
}

inline void AppendIndexerRows(Data &cache, Data &rows) {
    const int old = cache.dims.empty() ? 0 : cache.dims[1];
    const int wanted = old + rows.dims[1];
    cache.dataType = rows.dataType;
    cache.UpdateUnitSize();
    if (cache.expansionDims.empty() || cache.expansionDims[1] < wanted) {
        int capacity = std::max(256, cache.expansionDims.empty() ? 0 : cache.expansionDims[1]);
        while (capacity < wanted) capacity *= 2;
        cache.Expansion({1, capacity, 128});
    }
    CatDirect(cache, rows, 1);
}

inline void AppendIndexerKeys(Glm5NextIndexerCache &cache,
        Data &key, Data &gate, Data &ape) {
    const int sequence = key.dims[1];
    const int tail = cache.tokens % 4;
    for (Data *data : {&cache.keys, &cache.tailKeys, &cache.tailGates}) {
        data->lockInCPU = false;
        data->ToDevice(key.dataDevice, key.dataDeviceIds);
    }
    Data keys, gates;
    if (tail) {
        AssertInFastLLM(cache.tailKeys.dims == std::vector<int>({1, tail, 128}) &&
            cache.tailGates.dims == cache.tailKeys.dims,
            "GLM DSA incomplete KPool tail.");
        Cat(cache.tailKeys, key, 1, keys);
        Cat(cache.tailGates, gate, 1, gates);
    } else {
        keys.FakeFrom(key, 0); keys.Resize(key.dims);
        gates.FakeFrom(gate, 0); gates.Resize(gate.dims);
    }
    const int complete = (tail + sequence) / 4;
    if (complete) {
        Data pooled, quantized;
        ape.ToDevice(key.dataDevice, key.dataDeviceIds);
        AssertInFastLLM(FastllmCudaDeepSeekV4BuildCompressedKV(
            keys, gates, ape, 0, tail + sequence, 0, complete,
            4, 128, 128, false, pooled), "GLM DSA KPool compression failed.");
        ToDataType(pooled, DataType::BFLOAT16);
        RotateAndQuantizeIndexer(pooled, cache.hadamard, quantized);
        AppendIndexerRows(cache.keys, quantized);
    }
    if ((tail + sequence) % 4) {
        Split(keys, 1, complete * 4, tail + sequence, cache.tailKeys);
        Split(gates, 1, complete * 4, tail + sequence, cache.tailGates);
    } else {
        ClearIndexerTensor(cache.tailKeys);
        ClearIndexerTensor(cache.tailGates);
    }
    cache.tokens += sequence;
}

struct DsaProjections {
    Data keys, gates, query, headWeights;
};

// Stateless projections can share weight loads across independent requests.
// Pooling, rotation/scoring, and page tables retain their request-local cache.
inline void ProjectDsaKeys(Data &input, WeightMap &weight,
        const std::string &prefix, DsaProjections &projected) {
    const int sequence = input.Count(0) / input.dims.back();
    Data key, keyFloat;
    Linear(input, weight[prefix + "wk.weight"], Data(), key);
    ToDataType(key, keyFloat, DataType::FLOAT32);
    auto &gamma = weight[prefix + "k_norm.weight"];
    auto &beta = weight[prefix + "k_norm.bias"];
    gamma.ToDevice(input.dataDevice, input.dataDeviceIds);
    beta.ToDevice(input.dataDevice, input.dataDeviceIds);
    auto &normalized = projected.keys;
    normalized.dataType = DataType::FLOAT32;
    normalized.Resize(keyFloat.dims);
    normalized.ToDevice(input.dataDevice, input.dataDeviceIds, false);
    normalized.Allocate(false);
    AssertInFastLLM(FastllmCudaLayerNormWithEpsilon(
        keyFloat, gamma, beta, normalized, 1e-6f), "GLM DSA LayerNorm failed.");
    ToDataType(normalized, DataType::BFLOAT16);
    normalized.Reshape({1, sequence, 128});
    Linear(input, weight[prefix + "index_kpool_compress_gate"], Data(), projected.gates);
    projected.gates.Reshape(normalized.dims);
}

inline void ProjectDsaQueries(Data &input, Data &qNormalized, WeightMap &weight,
        const std::string &prefix, DsaProjections &projected) {
    const int sequence = input.Count(0) / input.dims.back();
    Linear(qNormalized, weight[prefix + "wq_b.weight"], Data(), projected.query);
    projected.query.Reshape({1, sequence, 32, 128});
    Data floatInput;
    ToDataType(input, floatInput, DataType::FLOAT32);
    Linear(floatInput, weight[prefix + "weights_proj.weight"], Data(), projected.headWeights);
    Mul(projected.headWeights, 1.0f / std::sqrt(32.0f), projected.headWeights);
    Mul(projected.headWeights, 1.0f / std::sqrt(128.0f), projected.headWeights);
}

inline void BuildDsaIndices(Data &input, Data &qNormalized,
        WeightMap &weight, const std::string &prefix, int past, int topK,
        Glm5NextIndexerCache &cache, Data &indices,
        const Data *pagedCache = nullptr, DsaProjections *projected = nullptr) {
    AssertInFastLLM(input.dataDevice == DataDevice::CUDA &&
        input.dataType == DataType::BFLOAT16 && topK == 2048,
        "GLM DSA currently requires CUDA BF16 and Top-2048.");
    if (past == 0 && cache.tokens != 0) {
        for (Data *data : {&cache.keys, &cache.tailKeys, &cache.tailGates}) {
            ClearIndexerTensor(*data);
        }
        cache.tokens = 0;
    }
    AssertInFastLLM(cache.tokens == past, "GLM DSA Indexer cache is out of sync.");
    const int sequence = input.Count(0) / input.dims.back();
    DsaProjections local;
    if (projected == nullptr) {
        ProjectDsaKeys(input, weight, prefix, local);
        projected = &local;
    }
    AppendIndexerKeys(cache, projected->keys, projected->gates,
        weight[prefix + "index_kpool_compress_ape"]);
    if (past + sequence <= topK) return;
    if (projected == &local) ProjectDsaQueries(input, qNormalized, weight, prefix, local);
    AssertInFastLLM(!projected->query.dims.empty() && !projected->headWeights.dims.empty(),
        "GLM DSA query projections are missing.");
    Data quantized, scores, groups;
    RotateAndQuantizeIndexer(projected->query, cache.hadamard, quantized);
    auto &headWeights = projected->headWeights;
    AssertInFastLLM(FastllmCudaDeepSeekV41IndexerScore(
        quantized, headWeights, cache.keys, 4, past, scores),
        "GLM DSA Indexer scoring failed.");
    const bool selected = sequence == 1 && FastllmCudaQwen4SelectBlocks(
        scores, topK / 4, past, 4, groups);
    AssertInFastLLM(selected || FastllmCudaDeepSeekV41IndexerTopK(
        scores, nullptr, topK / 4, 4, past, 1, groups),
        "GLM DSA group selection failed.");
    groups.Reshape({sequence, groups.dims.back()});
    const Data *pageTable = nullptr;
    if (pagedCache != nullptr) {
        AssertInFastLLM(sequence == 1 && pagedCache->isPagedKVCache &&
            pagedCache->dims[1] == past + sequence && pagedCache->pageLen > 0,
            "GLM DSA decode page table is out of sync.");
        if (cache.pageTableIds != pagedCache->pageIndex ||
            cache.pageTable.dataDeviceIds != input.dataDeviceIds ||
            cache.pageTable.cudaData == nullptr) {
            cache.pageTable.dataType = DataType::INT32;
            cache.pageTable.UpdateUnitSize();
            cache.pageTable.Resize({int(pagedCache->pageIndex.size())});
            cache.pageTable.ToDevice(input.dataDevice, input.dataDeviceIds, false);
            cache.pageTable.Allocate(false);
            FastllmCudaCopyFromHostToDevice(
                cache.pageTable.cudaData, (void*)pagedCache->pageIndex.data(),
                pagedCache->pageIndex.size() * sizeof(int32_t));
            cache.pageTableIds = pagedCache->pageIndex;
        }
        pageTable = &cache.pageTable;
    }
    AssertInFastLLM(FastllmCudaQwen4ExpandSelectedBlocks(
        groups, past + sequence, past, 4, indices,
        pageTable, pagedCache == nullptr ? 0 : pagedCache->pageLen),
        "GLM DSA group expansion failed.");
}

inline void SparseLatentAttention(Data &query, const Data &cache,
        Data &indices, float scale, Data &output, bool useFlashInfer = true,
        const Data *keyPeCache = nullptr) {
    const int heads = query.dims[0], sequence = query.dims[1];
    const int tokens = cache.dims[1], rank = query.dims[2];
    const auto *pool = cache.pagedKVCacheData;
    AssertInFastLLM(pool && rank == 512 && pool->dims[2] == 1 &&
        pool->dims[3] == rank, "GLM DSA latent cache layout mismatch.");
    Data latent(DataType::BFLOAT16);
    if (keyPeCache != nullptr) {
        // A paged decode cache means indices already address physical tokens.
        AssertInFastLLM(sequence == 1, "GLM DSA physical indices require decode.");
        if (useFlashInfer) {
            Data queryPe(DataType::BFLOAT16);
            queryPe.Resize({1, 1, heads, 64});
            queryPe.ToDevice(query.dataDevice, query.dataDeviceIds, false);
            queryPe.Allocate(0.0f);
            output.dataType = DataType::BFLOAT16;
            output.UpdateUnitSize();
            output.Resize(query.dims);
            output.ToDevice(query.dataDevice, query.dataDeviceIds, false);
            output.Allocate(false);
            const int selectedTokens = std::min(tokens / 4, 512) * 4 + tokens % 4;
            if (FastllmCudaMLAPaged(query, queryPe, *keyPeCache, cache, output,
                    scale, selectedTokens, &indices)) return;
        }
        // The BF16 fallback reads the same pool without gathering logical KV.
        latent.FakeFrom(*pool, 0);
        latent.dataDeviceIds = pool->dataDeviceIds;
        latent.Resize({1, int(pool->Count(0) / rank), rank});
    } else {
        latent.Resize({1, tokens, rank});
        latent.ToDevice(query.dataDevice, query.dataDeviceIds, false);
        latent.Allocate(false);
        for (size_t first = 0; first < cache.pageIndex.size();) {
            size_t end = first + 1;
            while (end < cache.pageIndex.size() &&
                cache.pageIndex[end] == cache.pageIndex[end - 1] + 1) ++end;
            const size_t begin = first * cache.pageLen;
            const size_t count = std::min(end * cache.pageLen, size_t(tokens)) - begin;
            AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                static_cast<uint8_t *>(latent.cudaData) + begin * rank * 2,
                static_cast<uint8_t *>(pool->cudaData) +
                    size_t(cache.pageIndex[first]) * cache.pageLen * rank * 2,
                count * rank * 2), "GLM DSA latent gather failed.");
            first = end;
        }
    }
    PermuteSelf(query, {1, 0, 2});
    query.Reshape({1, sequence, heads, rank});
    indices.Reshape({1, sequence, indices.dims.back()});
    if (keyPeCache != nullptr || !useFlashInfer ||
        !FastllmCudaGlm5NextDsaPrefill(query, latent, indices, scale, output)) {
        Data sink(DataType::FLOAT32);
        sink.Resize({heads});
        sink.ToDevice(query.dataDevice, query.dataDeviceIds, false);
        sink.Allocate(-std::numeric_limits<float>::infinity());
        // Candidate construction already enforces causal positions.
        AssertInFastLLM(FastllmCudaDeepSeekV41SparseAttention(
            query, latent, nullptr, &latent, &indices, sink, 0, 0, scale, output),
            "GLM DSA sparse MLA failed.");
    }
    output.Reshape({sequence, heads, rank});
    PermuteSelf(output, {1, 0, 2});
}

} } // namespace fastllm::glm5_next_detail
#endif
#endif
