#include "devices/cuda/naive-n05-swa-settings.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/multicuda/fastllm-naive-context-indexer.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/utils.h"
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "parallel/cuda_runtime.h"
#ifndef USE_ROCM
#include "devices/multicuda/fastllm-naive-verify.h"
#endif

namespace fastllm {
NaiveAttentionCapabilities FastllmCudaNaiveAttentionCapabilities() {
    const int arch = FastllmCudaRuntimeArch();
    static const bool swaQkv = fastllm_swa_settings::QkvEnabled();
    return {arch == 103, arch == 103 && swaQkv, arch == 90, arch == 100 || arch == 103,
            FastllmCudaHasBlackwellIndexerAttention()};
}

bool FastllmCudaStablePrefillReduction(const std::vector<int> &archs) {
    return !archs.empty() && std::all_of(archs.begin(), archs.end(), [](int arch) { return arch == 120; });
}
bool FastllmCudaStablePrefillReduction(parallel::CudaRuntime &runtime) {
    std::vector<int> archs(runtime.Layout().Size());
    runtime.Run([&](int rank, int) { archs[rank] = FastllmCudaRuntimeArch(); });
    return FastllmCudaStablePrefillReduction(archs);
}

int FastllmCudaOwnerDraftFeatureLimit(const std::vector<int> &devices) {
    return FastllmCudaNaiveBatchedDraftHeadGroup(devices) ? kFastllmMtpSmallRejectionMaxBatch : 1;
}

bool FastllmCudaVerifyReduceNormSupported(int rows, const std::vector<int> &devices) {
    return rows > 0 && (rows <= 128 || FastllmCudaNaiveHopperGroup(devices));
}

NaiveVerifyAttentionPlan FastllmCudaPlanVerifyAttention(int device, int rows, int tpRanks,
        const std::vector<int> *requests, bool ringCache, bool integratedExperts, bool localDsa,
        int capacity, int indexHeads, int indexDim, int topK, int localAttentionRanks) {
    NaiveVerifyAttentionPlan plan{};
#ifndef USE_ROCM
    // TP attention consumes all verification rows even when EP distributes the experts.
    plan.packed = requests && requests->size() == 1 && FastllmCudaPackedVerifySupported(rows, tpRanks);
    if (localAttentionRanks == 1) {
        plan.replicaArch = FastllmCudaRuntimeArch();
        plan.replicaLocalDsa = localDsa && (plan.replicaArch == 90 || plan.replicaArch == 103 || plan.replicaArch == 120);
    }
#ifdef FASTLLM_ENABLE_NAIVE_HOPPER_VERIFY
    const bool hopper = FastllmCudaRuntimeArch() == 90;
    const bool index = indexHeads == 16 && indexDim == 128 && topK == 2048;
    // Independent attention ranks project the padded verification group in one GEMM.
    // Their local Q and head weights are consumed even in the short-context buckets.
    plan.packedIndexer = hopper && index && !requests && !integratedExperts &&
                         capacity > 0 && rows > 0 && rows <= 256;
    // Hopper TP attention materializes its residual independently of the expert layout.
    if (requests && hopper && !requests->empty() &&
        std::all_of(requests->begin(), requests->end(), [](int n) { return n > 0; })) {
        plan.hopperRing = ringCache;
        plan.exportsRingFeature = ringCache && rows <= 8 && requests->size() == 1;
        plan.hopperReplicated = index && localDsa && requests->size() == 1 && rows <= 8;
        plan.hopperContext = index && capacity > 2048 && capacity <= 8192;
        plan.packedIndexer = index && capacity > topK &&
            std::all_of(requests->begin(), requests->end(), [](int n) { return n <= 8; });
        return plan;
    }
#endif
    if (requests && integratedExperts) {
        plan.swa = ringCache && rows <= 16 && FastllmCudaSwaVerifyLayerSupported(device, rows);
        plan.dsa = localDsa && FastllmCudaLocalDsaLayerSupported(*requests, rows, capacity, indexHeads, indexDim, topK);
    }
#endif
    return plan;
}
NaiveVerifyAttentionPlan::Layer NaiveVerifyAttentionPlan::ForLayer(bool sliding, bool residual, bool localIndexer,
        int cacheRequests, bool ringCache, bool contextIndex) const {
    const bool packedCache = cacheRequests == 1 || (cacheRequests > 1 && replicaArch == 90);
    const bool replicaSwa = replicaArch == 90 || replicaArch == 100 || replicaArch == 103 || replicaArch == 120;
    const bool replicaDsa = replicaLocalDsa || (replicaArch == 120 && contextIndex);
    if (!residual && (sliding ? hopperRing && ringCache :
        localIndexer && (hopperReplicated || (hopperContext && contextIndex)))) return {true, false, false, false};
    return {packed && packedCache && (sliding ? ringCache : !residual && localIndexer && contextIndex),
            sliding && residual && swa, !sliding && localIndexer && dsa,
            packedCache && !residual && (sliding ? ringCache && replicaSwa : localIndexer && replicaDsa)};
}

static void GatherContextCandidates(Data &exchange, int part, int rank, int ranks, int gpu,
        const IndexerGather &gather, int rowWords) {
    if (gather) { gather((int *)exchange.cudaData + rank * part, exchange.cudaData, part); return; }
    if (!FastllmCudaCustomAllGatherRows(exchange.cudaData, ranks * part, rowWords, FLOAT32, gpu))
        AssertInFastLLM(FastllmNcclAllGather((int *)exchange.cudaData + rank * part,
            exchange.cudaData, part, INT32, gpu), "Indexer context all-gather failed.");
}
static int ContextCandidateWidth(int capacity, int page, int ranks) {
    const int pages = capacity / page + (capacity % page != 0);
    return std::min(2048, pages * (page / ranks));
}
void RunContextVerifyIndexer(const Data &query, const Data &weights,
        const std::vector<FastllmNaiveVerifyIndexerItem> &items,
        FastllmNaiveDecodeScratch &scratch, Data &live, Data &exchange,
        int rank, int ranks, int page, int gpu, bool fp8Query, const std::vector<int> &devices, const IndexerGather &gather) {
    int rows = 0, capacity = 0;
    for (const auto &item : items) {
        rows += item.rows;
        capacity = std::max(capacity, item.capacity ? item.capacity : item.past + item.rows);
    }
    // Every rank derives the same exchange width from the graph capacity,
    // including ranks whose final page is empty during replay.
    const int width = ranks > 0 && page > 0 ? ContextCandidateWidth(capacity, page, ranks) : 0;
    const bool hopper = FastllmCudaNaiveHopperGroup(devices);
    const int directWidth = (hopper ? (width < 2048 || rows < 16) : rows == 8) && ranks == 8 && page > 0 && page % ranks == 0 &&
        (hopper || capacity <= 2048 * ranks) && (hopper || FastllmCudaNaiveIndexerContextSupported(devices, true))
        ? width : 0;
    const int rowWords = (directWidth ? directWidth : 2048) * 2, part = rows * rowWords;
    exchange.dataType = INT32; exchange.Resize({ranks * part});
    exchange.ToDevice(query.dataDevice, query.dataDeviceIds, false); exchange.Allocate(false);
    AssertInFastLLM(FastllmCudaNaiveBatchVerifyIndexer(query, weights, items, fp8Query,
        scratch, live, rank, ranks, &exchange, page, directWidth), "Unsupported context-sharded verification layout.");
    GatherContextCandidates(exchange, part, rank, ranks, gpu, gather, rowWords);
    FastllmCudaNaiveMergeVerifyIndexer(items, scratch, live, exchange, ranks, page, directWidth);
}
void RunContextDecodeIndexer(const Data &query, const Data &weights, const Data &live,
        const std::vector<FastllmNaiveIndexerRow> &rows, Data &exchange,
        int rank, int ranks, int page, int gpu, bool fp8Query, const IndexerGather &gather) {
    // A graph replays different live lengths within this capacity. Size the
    // dedicated exchange for the largest possible shard on any rank.
    int directWidth = 0;
    if (!rows.empty() && rows.size() < 16) {
        int capacity = 0;
        for (const auto &row : rows) capacity = std::max(capacity, row.capacity);
        directWidth = ContextCandidateWidth(capacity, page, ranks);
    }
    const int rowWords = (directWidth ? directWidth : 2048) * 2;
    const int part = rows.size() * rowWords;
    exchange.dataType = INT32;
    exchange.Resize({ranks * part});
    exchange.ToDevice(query.dataDevice, query.dataDeviceIds, false);
    exchange.Allocate(false);
    AssertInFastLLM(FastllmCudaNaiveBatchDecodeIndexer(query, weights, live, rows, fp8Query,
        rank, ranks, &exchange, page, directWidth), "Unsupported context-sharded decode Indexer layout.");
    GatherContextCandidates(exchange, part, rank, ranks, gpu, gather, rowWords);
    FastllmCudaNaiveMergeDecodeIndexer(live, rows, exchange, ranks, page, directWidth);
}
}
