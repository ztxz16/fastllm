#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-moe-stream.cuh"
#include "../moe/fastllm-moe-gguf-common.cuh"
#include "../moe/fastllm-moe-gguf-restore.cuh"

// Host NUMA shards remain immutable. Selected experts reuse resident records
// or pass through bounded GPU scratch before optional cache admission.
namespace fastllm_gguf_stream {
struct WeightCopy {
    const uint8_t *source;
    uint8_t *destination;
    int type, rows, columns, cross, blockSize, blockBytes;
};

static int Ordinary(int type) {
    type = fastllm_gguf_restore::Ordinary(type);
    switch (type) {
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_Q2_0:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q8_0:
            return type;
        default: return -1;
    }
}

__global__ void Restore(const WeightCopy *copies) {
    const WeightCopy w = copies[blockIdx.y];
    const int blocks = w.columns / w.blockSize;
    const int lane = threadIdx.x%32;
    // A warp restores one quantized block with contiguous field stores.
    for (int i = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
         i < w.rows*blocks; i += gridDim.x*(blockDim.x/32)) {
        const int row = i/blocks, col = i%blocks;
        const int srcRow = w.cross ? 2*(row%(w.rows/2)) + row/(w.rows/2) : row;
        if (fastllm_gguf_restore::Ordinary(w.type) != w.type) {
            fastllm_gguf_restore::Block(w.source, w.destination, w.type, blocks, srcRow, col, i);
        } else {
            const int bytes = w.blockBytes;
            for (int b = lane; b < bytes; b += 32)
                w.destination[size_t(i)*bytes+b] = w.source[size_t(srcRow*blocks+col)*bytes+b];
        }
    }
}

static size_t Align(size_t bytes) { return (bytes+255)&~size_t(255); }

static void UploadWeight(const fastllm::Data &weight, uint8_t *target, cudaStream_t stream) {
    if (weight.cpuData) {
        CUDA_CHECK(cudaMemcpyAsync(target, weight.cpuData, weight.GetBytes(), cudaMemcpyHostToDevice, stream));
    } else {
        const size_t shardBytes = weight.GetBytes() / weight.numasData.size();
        for (size_t node = 0; node < weight.numasData.size(); ++node)
            CUDA_CHECK(cudaMemcpyAsync(target + node * shardBytes, weight.numasData[node],
                shardBytes, cudaMemcpyHostToDevice, stream));
    }
}

struct DownUpload {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, ready = nullptr;
    DownUpload() {
        // Restore must run between MMQ CTAs so it can release staging for
        // the next transfer. Equal-priority MMQ can otherwise starve uploads.
        int leastPriority = 0, greatestPriority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
        CUDA_CHECK(cudaEventCreateWithFlags(&begin, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
    ~DownUpload() {
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(ready));
    }
};
// The caller serializes model layers; keep a small reusable transfer ring per
// device rather than creating streams/events for every expert or layer.
struct Pipeline {
    static constexpr int groups = 3, experts = 16;
    std::mutex mutex;
    cudaStream_t dma = nullptr, restore = nullptr;
    cudaEvent_t metadata = nullptr, copied[groups]{}, restored[groups]{}, released[groups]{};
    Pipeline() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&dma, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&restore, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&metadata, cudaEventDisableTiming));
        for (int i = 0; i < groups; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&copied[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&restored[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&released[i], cudaEventDisableTiming));
        }
    }
};
static Pipeline &GetPipeline(int device) {
    static std::mutex mutex;
    static auto *pipelines = new std::map<int, std::unique_ptr<Pipeline>>;
    std::lock_guard<std::mutex> lock(mutex);
    auto &pipeline = (*pipelines)[device];
    if (!pipeline) pipeline.reset(new Pipeline());
    return *pipeline;
}

static bool RunPipelined(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &workspace, fastllm::Data &output, fastllm::Data **weights,
        int expertCount, const int32_t *indices, const float *scores, int topk,
        const std::unordered_set<int> &experts, bool cross, int gt, int dt, int inter) {
    using namespace fastllm_gguf_mmq;
    using fastllm_gguf_moe::AllocateTensor;
    const int device = FastllmCudaGetDevice(), rows = input.dims[0], hidden = input.dims[1];
    auto &pipeline = GetPipeline(device);
    std::lock_guard<std::mutex> lock(pipeline.mutex);
    FastllmCudaMoeGGUFResidents resident;
    FastllmCudaGetMoeGGUFResidents(weights, expertCount, resident);
    std::vector<std::vector<int>> routes(expertCount);
    for (int r = 0; r < rows * topk; ++r) {
        const int e = indices[r];
        if (e >= 0 && e < expertCount && experts.count(e + 1)) routes[e].push_back(r);
    }
    auto cached = [&](int e) {
        return resident.gateType == gt && resident.downType == dt && resident.hidden == hidden &&
            resident.inter == inter && 2 * e + 1 < int(resident.weights.size()) && resident.weights[2 * e];
    };
    std::vector<int> order;
    for (int e = 0; e < expertCount; ++e) if (!routes[e].empty()) order.push_back(e);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cached(a) > cached(b); });
    if (order.empty()) return false;
    struct Batch {
        std::vector<int> experts, counts, offsets{0}, tiles, routes;
        size_t pointersOffset, countsOffset, offsetsOffset, tilesOffset, routesOffset;
        int copiesBegin = 0, copiesCount = 0;
    };
    std::vector<Batch> batches;
    int capacity = 0;
    size_t metaBytes = 0;
    auto reserve = [&](size_t bytes) { const size_t at = metaBytes; metaBytes += Align(bytes); return at; };
    for (size_t begin = 0; begin < order.size(); begin += Pipeline::experts) {
        Batch batch;
        const size_t end = std::min(order.size(), begin + Pipeline::experts);
        for (size_t j = begin; j < end; ++j) {
            const int e = order[j], local = j - begin;
            batch.experts.push_back(e); batch.counts.push_back(routes[e].size());
            batch.routes.insert(batch.routes.end(), routes[e].begin(), routes[e].end());
            const int padded = (routes[e].size() + 15) / 16 * 16;
            batch.routes.resize(batch.offsets.back() + padded, -1);
            batch.offsets.push_back(batch.routes.size());
            batch.tiles.insert(batch.tiles.end(), padded / 16, local);
        }
        capacity = std::max(capacity, int(batch.routes.size()));
        batch.pointersOffset = reserve(2 * batch.experts.size() * sizeof(void *));
        batch.countsOffset = reserve(batch.counts.size() * sizeof(int));
        batch.offsetsOffset = reserve(batch.offsets.size() * sizeof(int));
        batch.tilesOffset = reserve(batch.tiles.size() * sizeof(int));
        batch.routesOffset = reserve(batch.routes.size() * sizeof(int));
        batches.push_back(std::move(batch));
    }
    const auto *gu = weights[2 * (order[0] + 1)], *down = weights[2 * (order[0] + 1) + 1];
    const size_t gateWeightBytes = Align(gu->GetBytes());
    const size_t stride = gateWeightBytes + Align(down->GetBytes());
    const bool misses = std::any_of(order.begin(), order.end(), [&](int e) { return !cached(e); });
    const size_t ringBytes = misses ? Pipeline::groups * Pipeline::experts * stride : 0;
    const size_t rawOffset = 0, canonicalOffset = ringBytes, metaOffset = 2 * ringBytes;
    const size_t descOffset = reserve(2 * order.size() * sizeof(WeightCopy));
    const size_t scoreOffset = reserve(size_t(rows) * topk * sizeof(float));
    const size_t computeOffset = metaOffset + metaBytes;
    const size_t bytes = Align(computeOffset + StreamedMoeWorkspaceBytes(rows, hidden, inter, topk, capacity));
    const size_t gateBytes = size_t(rows) * topk * inter * (input.dataType == fastllm::FLOAT32 ? 4 : 2);
    // Query the driver only when either persistent allocation must grow.
    if (!workspace.cudaData || workspace.expansionBytes < bytes || !gate.cudaData || gate.expansionBytes < gateBytes) {
        size_t freeBytes = 0, totalBytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        const size_t reusable = (workspace.cudaData ? workspace.expansionBytes : 0) +
            (gate.cudaData ? gate.expansionBytes : 0);
        if (bytes + gateBytes + 256 * size_t(1024 * 1024) > freeBytes + reusable) return false;
    }
    if (bytes / 256 > size_t(INT32_MAX)) return false;
    AllocateTensor(workspace, fastllm::INT8, {int(bytes / 256), 256}, device);
    AllocateTensor(gate, input.dataType, {rows * topk, inter}, device);
    AllocateTensor(output, input.dataType, {rows, hidden}, device);
    auto *base = static_cast<uint8_t *>(workspace.cudaData);
    FastllmCudaMoeGGUFPrefillPlan admission;
    FastllmCudaPlanMoeGGUFPrefill(weights, expertCount, indices, scores, rows, topk, experts, admission);
    struct Upload { const fastllm::Data *weight; uint8_t *target; };
    std::vector<std::vector<Upload>> uploads(batches.size());
    std::vector<WeightCopy> copies;
    std::vector<uint8_t> metadata(metaBytes, 0);
    auto put = [&](size_t offset, const void *source, size_t size) { std::memcpy(metadata.data() + offset, source, size); };
    for (size_t g = 0; g < batches.size(); ++g) {
        auto &batch = batches[g];
        std::vector<const void *> pointers;
        batch.copiesBegin = copies.size();
        for (size_t j = 0; j < batch.experts.size(); ++j) {
            const int e = batch.experts[j];
            for (int part = 0; part < 2; ++part) {
                const auto &w = *weights[2 * (e + 1) + part];
                if (cached(e)) { pointers.push_back(resident.weights[2 * e + part]); continue; }
                const size_t offset = ((g % Pipeline::groups) * Pipeline::experts + j) * stride +
                    (part ? gateWeightBytes : 0);
                auto *destination = 2 * e + part < int(admission.weights.size()) && admission.weights[2 * e + part]
                    ? static_cast<uint8_t *>(admission.weights[2 * e + part]) : base + canonicalOffset + offset;
                pointers.push_back(destination);
                const bool restore = (cross && part == 0) || Ordinary(w.ggmlType) != w.ggmlType;
                auto *target = restore ? base + rawOffset + offset : destination;
                uploads[g].push_back({&w, target});
                if (restore) copies.push_back({target, destination, w.ggmlType, w.dims[0], w.dims[1],
                    int(cross && part == 0), int(ggml_blck_size((ggml_type)Ordinary(w.ggmlType))),
                    int(ggml_type_size((ggml_type)Ordinary(w.ggmlType)))});
            }
        }
        batch.copiesCount = copies.size() - batch.copiesBegin;
        put(batch.pointersOffset, pointers.data(), pointers.size() * sizeof(void *));
        put(batch.countsOffset, batch.counts.data(), batch.counts.size() * sizeof(int));
        put(batch.offsetsOffset, batch.offsets.data(), batch.offsets.size() * sizeof(int));
        put(batch.tilesOffset, batch.tiles.data(), batch.tiles.size() * sizeof(int));
        put(batch.routesOffset, batch.routes.data(), batch.routes.size() * sizeof(int));
    }
    if (!copies.empty()) put(descOffset, copies.data(), copies.size() * sizeof(WeightCopy));
    put(scoreOffset, scores, size_t(rows) * topk * sizeof(float));
    const auto stream = cudaStreamPerThread;
    CUDA_CHECK(cudaMemcpyAsync(base + metaOffset, metadata.data(), metaBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(pipeline.metadata, stream));
    CUDA_CHECK(cudaStreamWaitEvent(pipeline.restore, pipeline.metadata, 0));
    auto run = [&](StreamedMoePhase phase, const StreamedMoeBatch &batch) {
        fastllm::AssertInFastLLM(RunStreamedMoe(phase, input, gate, output, base + computeOffset,
            capacity, hidden, inter, topk, gt, dt, batch,
            reinterpret_cast<const float *>(base + metaOffset + scoreOffset)), "Streamed GGUF prefill failed.");
    };
    run(StreamedMoePhase::Prepare, {});
    const int maxBlocks = std::max(gu->dims[0] * (hidden / int(ggml_blck_size((ggml_type)gt))),
        down->dims[0] * (inter / int(ggml_blck_size((ggml_type)dt))));
    for (size_t g = 0; g < batches.size(); ++g) {
        const auto &batch = batches[g];
        const int slot = g % Pipeline::groups;
        if (g >= Pipeline::groups) CUDA_CHECK(cudaStreamWaitEvent(pipeline.dma, pipeline.released[slot], 0));
        for (const auto &upload : uploads[g]) UploadWeight(*upload.weight, upload.target, pipeline.dma);
        CUDA_CHECK(cudaEventRecord(pipeline.copied[slot], pipeline.dma));
        CUDA_CHECK(cudaStreamWaitEvent(pipeline.restore, pipeline.copied[slot], 0));
        if (batch.copiesCount) Restore<<<dim3(std::min(64, (maxBlocks + 7) / 8), batch.copiesCount),
            256, 0, pipeline.restore>>>(reinterpret_cast<const WeightCopy *>(base + metaOffset + descOffset) + batch.copiesBegin);
        CUDA_CHECK(cudaEventRecord(pipeline.restored[slot], pipeline.restore));
        CUDA_CHECK(cudaStreamWaitEvent(stream, pipeline.restored[slot], 0));
        StreamedMoeBatch work;
        work.experts = batch.experts.size(); work.rows = batch.routes.size();
        work.weights = reinterpret_cast<const uint8_t *const *>(base + metaOffset + batch.pointersOffset);
        work.counts = reinterpret_cast<const int *>(base + metaOffset + batch.countsOffset);
        work.offsets = reinterpret_cast<const int *>(base + metaOffset + batch.offsetsOffset);
        work.tileExperts = reinterpret_cast<const int *>(base + metaOffset + batch.tilesOffset);
        work.routes = reinterpret_cast<const int *>(base + metaOffset + batch.routesOffset);
        run(StreamedMoePhase::Compute, work);
        CUDA_CHECK(cudaEventRecord(pipeline.released[slot], stream));
    }
    run(StreamedMoePhase::Finish, {});
    FastllmCudaPublishMoeGGUFPrefill(admission);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}
} // namespace fastllm_gguf_stream

bool FastllmCudaMergeMOEGGUFHost(const fastllm::Data &input,
        fastllm::Data &gate, fastllm::Data &workspace, fastllm::Data &output,
        fastllm::Data **weights, int expertCount, const int32_t *indices,
        const float *scores, int topk, const std::unordered_set<int> &experts,
        bool crossSwiglu, bool deepSeekV4Mode, float swigluLimit,
        int activationQuantBlock) {
    using namespace fastllm_gguf_stream;
    using fastllm_gguf_moe::AllocateTensor;
    if (!weights || !indices || !scores || input.dims.size() != 2 ||
        input.dims[0] <= 32 || input.dims[0] > 4096 || topk <= 0 || topk > 32 ||
        expertCount <= 0 || expertCount > 1024 || experts.empty() || experts.count(0) ||
        input.dataDevice != fastllm::CUDA || !input.cudaData ||
        !fastllm_gguf_moe::IsActivationType(input.dataType)) return false;
    // V4.1's FP8 block-32 and BF16 boundaries are implemented for the
    // Q2_K gate / Q4_K down pair. V4 block-128 retains its existing fallback.
    if (deepSeekV4Mode && (activationQuantBlock != 32 ||
        input.dataType != fastllm::BFLOAT16)) return false;
    const int device = FastllmCudaGetDevice(), rows = input.dims[0], hidden = input.dims[1];
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    int gt = -1, dt = -1, inter = 0, maxBlocks = 0;
    size_t packedBytes = 0, stagingBytes = 0;
    struct Source {
        const fastllm::Data *weight;
        size_t offset;
        int slot;
        bool restore;
    };
    std::vector<Source> sources;
    // Validate the entire subset before allocating, copying, or changing output.
    // The selected expert IDs use NUMA's +1 convention (slot 0 is shared).
    for (int e : experts) {
        if (e <= 0 || e > expertCount) return false;
        auto *gu = weights[2*e], *down = weights[2*e+1];
        if (!gu || !down || gu->dims.size() != 2 || down->dims.size() != 2 ||
            gu->dims[1] != hidden || gu->dims[0] != 2*down->dims[1] ||
            down->dims[0] != hidden) return false;
        const int g = Ordinary(gu->ggmlType), d = Ordinary(down->ggmlType);
        if (gt < 0) { gt = g; dt = d; inter = down->dims[1]; }
        if (g < 0 || d < 0 || g != gt || d != dt || inter != down->dims[1]) return false;
        for (int part = 0; part < 2; ++part) {
            const auto *w = weights[2*e+part];
            if (w->dataType != fastllm::DATA_GGUF_FORMAT || w->dims[0]%4 ||
                w->dims[1]%ggml_blck_size((ggml_type)Ordinary(w->ggmlType)) ||
                (!w->cpuData && (w->numasData.empty() ||
                 w->dims[0]%(4*w->numasData.size()) ||
                 std::any_of(w->numasData.begin(), w->numasData.end(),
                    [](const uint8_t *p) { return p == nullptr; })))) return false;
            // Ordinary GGUF builds only its small streamed batches below.
            // These full-subset descriptors belong to the V4.1 path.
            if (!deepSeekV4Mode) continue;
            const bool restore = (crossSwiglu && part == 0) || Ordinary(w->ggmlType) != w->ggmlType;
            const size_t weightBytes = Align(w->GetBytes());
            const int slot = 2*(e-1)+part;
            sources.push_back({w, packedBytes, slot, restore});
            packedBytes += weightBytes;
            stagingBytes = std::max(stagingBytes, weightBytes);
            maxBlocks = std::max(maxBlocks, w->dims[0] *
                (w->dims[1]/int(ggml_blck_size((ggml_type)Ordinary(w->ggmlType)))));
        }
    }
    // IQ1_M gate/up currently uses per-route DP4A in grouped MMQ. On long
    // prefill, the existing per-expert GEMM reuses these weights much better.
    // Keep that implementation until a matrix IQ1_M gate kernel is available.
    if (gt == GGML_TYPE_IQ1_M) return false;
    const size_t mmqBytes = FastllmCudaMoeGGUFGroupedWorkspaceBytes(
        gt, dt, rows, hidden, inter, expertCount, topk, deepSeekV4Mode);
    if (!mmqBytes) return false;
    if (!deepSeekV4Mode) return RunPipelined(input, gate, workspace, output, weights,
        expertCount, indices, scores, topk, experts, crossSwiglu, gt, dt, inter);
    // Two bounded upload slots: gate and down restore on separate streams.
    // Canonical weights and routing live through both projections.
    // Never alias in-flight DMA with MMQ scratch.
    const size_t tableOffset = packedBytes + 2*stagingBytes;
    const size_t descOffset = tableOffset+Align(2*expertCount*sizeof(void *));
    const size_t indexOffset = descOffset+Align(sources.size()*sizeof(WeightCopy));
    const size_t scoreOffset = indexOffset+Align(size_t(rows)*topk*sizeof(int32_t));
    const size_t mmqOffset = scoreOffset+Align(size_t(rows)*topk*sizeof(float));
    const size_t bytes = Align(mmqOffset + mmqBytes);
    if (bytes/256 > size_t(INT32_MAX)) return false;
    const size_t gateBytes = size_t(rows)*topk*inter*(input.dataType == fastllm::FLOAT32 ? 4 : 2);
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t reusable = (workspace.cudaData ? workspace.expansionBytes : 0) +
        (gate.cudaData ? gate.expansionBytes : 0);
    if (bytes+gateBytes+256*size_t(1024*1024) > freeBytes+reusable) return false;
    // Full V4.1 worker subsets can exceed 2 GiB even while fitting VRAM.
    // A two-dimensional byte tensor keeps each dimension in the int range.
    AllocateTensor(workspace, fastllm::INT8, {int(bytes/256), 256}, device);
    AllocateTensor(gate, input.dataType, {rows*topk, inter}, device);
    AllocateTensor(output, input.dataType, {rows, hidden}, device);
    auto *base = static_cast<uint8_t *>(workspace.cudaData);
    const auto stream = cudaStreamPerThread;
    DownUpload downUpload;
    std::vector<const void *> table(2*expertCount, nullptr);
    std::vector<WeightCopy> copies;
    std::vector<int> copyIndices(sources.size(), -1);
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto &src = sources[i];
        const auto &w = *src.weight;
        auto *target = base+src.offset;
        table[src.slot] = target;
        if (src.restore) {
            auto *upload = base+packedBytes+(src.slot%2)*stagingBytes;
            copyIndices[i] = copies.size();
            copies.push_back({upload, target, w.ggmlType, w.dims[0], w.dims[1],
                int(crossSwiglu && src.slot%2 == 0),
                int(ggml_blck_size((ggml_type)Ordinary(w.ggmlType))),
                int(ggml_type_size((ggml_type)Ordinary(w.ggmlType)))});
        }
    }
    if (!copies.empty()) CUDA_CHECK(cudaMemcpyAsync(base+descOffset, copies.data(),
        copies.size()*sizeof(WeightCopy), cudaMemcpyHostToDevice, stream));
    auto upload = [&](size_t sourceIndex, cudaStream_t uploadStream) {
        const auto &src = sources[sourceIndex];
        const auto &w = *src.weight;
        auto *target = src.restore ? const_cast<uint8_t *>(copies[copyIndices[sourceIndex]].source) : base+src.offset;
        UploadWeight(w, target, uploadStream);
        if (src.restore) {
            const bool kR4 = w.ggmlType == GGML_TYPE_Q2_K_R4 || w.ggmlType == GGML_TYPE_Q4_K_R4;
            const int gridLimit = kR4 ? 16*fastllm_gguf_mmq::ggml_cuda_info().devices[device].nsm : 64;
            Restore<<<std::min(gridLimit, (maxBlocks+7)/8), 256, 0, uploadStream>>>(
                reinterpret_cast<const WeightCopy *>(base+descOffset)+copyIndices[sourceIndex]);
        }
    };
    for (size_t i = 0; i < sources.size(); ++i)
        if (sources[i].slot%2 == 0) upload(i, stream);
    CUDA_CHECK(cudaMemcpyAsync(base+tableOffset, table.data(), table.size()*sizeof(void *), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(base+indexOffset, indices, size_t(rows)*topk*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(base+scoreOffset, scores, size_t(rows)*topk*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(downUpload.begin, stream));
    CUDA_CHECK(cudaStreamWaitEvent(downUpload.stream, downUpload.begin, 0));
    for (size_t i = 0; i < sources.size(); ++i)
        if (sources[i].slot%2) upload(i, downUpload.stream);
    CUDA_CHECK(cudaEventRecord(downUpload.ready, downUpload.stream));
    CUDA_CHECK(cudaGetLastError());
    const bool ok = fastllm_gguf_mmq::RunGrouped(input, gate, output, base+tableOffset,
        reinterpret_cast<const int32_t *>(base+indexOffset),
        reinterpret_cast<const float *>(base+scoreOffset), base+mmqOffset,
        gt, dt, hidden, inter, expertCount, topk, deepSeekV4Mode, swigluLimit, downUpload.ready);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // A CUDA launch failure is an error, not permission to execute a second path.
    fastllm::AssertInFastLLM(ok, "GGUF NUMA grouped prefill launch failed.");
    return true;
}
