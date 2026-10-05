#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-moe-stream.cuh"
#include "../moe/fastllm-moe-gguf-common.cuh"
#include "../moe/fastllm-moe-gguf-restore.cuh"

// Host NUMA shards remain immutable; only the current worker's selected
// experts occupy GPU scratch.
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

struct DownUpload {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, ready = nullptr;
    explicit DownUpload(bool enabled) {
        if (!enabled) return;
        // Restore must run between MMQ CTAs so it can release staging for
        // the next transfer. Equal-priority MMQ can otherwise starve uploads.
        int leastPriority = 0, greatestPriority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPriority));
        CUDA_CHECK(cudaEventCreateWithFlags(&begin, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    }
    ~DownUpload() {
        if (!stream) return;
        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaEventDestroy(begin));
        CUDA_CHECK(cudaEventDestroy(ready));
    }
};
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
    FastllmCudaMoeGGUFResidents resident;
    if (!deepSeekV4Mode) FastllmCudaGetMoeGGUFResidents(weights, expertCount, resident);
    struct Source {
        const fastllm::Data *weight;
        size_t offset;
        int slot;
        bool restore;
        const void *resident;
    };
    std::vector<Source> sources;
    // Validate the entire subset before allocating, copying, or changing output.
    // The selected expert IDs use NUMA's +1 convention (slot 0 is shared).
    for (int e = 1; e <= expertCount; ++e) {
        if (!experts.count(e)) continue;
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
            const bool restore = (crossSwiglu && part == 0) || Ordinary(w->ggmlType) != w->ggmlType;
            const size_t weightBytes = Align(w->GetBytes());
            const int slot = 2*(e-1)+part;
            const void *cached = resident.gateType == g && resident.downType == d &&
                resident.hidden == hidden && resident.inter == inter &&
                slot < int(resident.weights.size()) ? resident.weights[slot] : nullptr;
            sources.push_back({w, packedBytes, slot, restore, cached});
            if (cached) continue;
            packedBytes += weightBytes;
            stagingBytes = std::max(stagingBytes, weightBytes);
            maxBlocks = std::max(maxBlocks, w->dims[0] *
                (w->dims[1]/int(ggml_blck_size((ggml_type)Ordinary(w->ggmlType)))));
        }
    }
    if (sources.empty() || sources.size() != experts.size()*2) return false;
    // IQ1_M gate/up currently uses per-route DP4A in grouped MMQ. On long
    // prefill, the existing per-expert GEMM reuses these weights much better.
    // Keep that implementation until a matrix IQ1_M gate kernel is available.
    if (gt == GGML_TYPE_IQ1_M) return false;
    const size_t mmqBytes = FastllmCudaMoeGGUFGroupedWorkspaceBytes(
        gt, dt, rows, hidden, inter, expertCount, topk, deepSeekV4Mode);
    if (!mmqBytes) return false;
    // Two bounded upload slots: gate and down restore on separate streams.
    // Canonical misses and routing live through both projections; resident
    // records are borrowed directly. Never alias in-flight DMA with MMQ scratch.
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
    DownUpload downUpload(stagingBytes != 0);
    std::vector<const void *> table(2*expertCount, nullptr);
    std::vector<WeightCopy> copies;
    std::vector<int> copyIndices(sources.size(), -1);
    for (size_t i = 0; i < sources.size(); ++i) {
        const auto &src = sources[i];
        const auto &w = *src.weight;
        auto *target = base+src.offset;
        table[src.slot] = src.resident ? src.resident : target;
        if (src.resident) continue;
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
        if (src.resident) return;
        const auto &w = *src.weight;
        auto *target = src.restore ? const_cast<uint8_t *>(copies[copyIndices[sourceIndex]].source) : base+src.offset;
        if (w.cpuData) {
            CUDA_CHECK(cudaMemcpyAsync(target, w.cpuData, w.GetBytes(), cudaMemcpyHostToDevice, uploadStream));
        } else {
            const size_t shardBytes = w.GetBytes()/w.numasData.size();
            for (size_t node = 0; node < w.numasData.size(); ++node)
                CUDA_CHECK(cudaMemcpyAsync(target+node*shardBytes, w.numasData[node],
                    shardBytes, cudaMemcpyHostToDevice, uploadStream));
        }
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
    if (downUpload.stream) {
        CUDA_CHECK(cudaEventRecord(downUpload.begin, stream));
        CUDA_CHECK(cudaStreamWaitEvent(downUpload.stream, downUpload.begin, 0));
        for (size_t i = 0; i < sources.size(); ++i)
            if (sources[i].slot%2) upload(i, downUpload.stream);
        CUDA_CHECK(cudaEventRecord(downUpload.ready, downUpload.stream));
    }
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
