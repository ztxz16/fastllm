#include "fastllm-gguf-mmq-common.cuh"
#include "../moe/fastllm-moe-gguf-common.cuh"

// Host NUMA shards remain immutable; only the current worker's selected
// experts occupy GPU scratch.
namespace fastllm_gguf_stream {
struct WeightCopy {
    const uint8_t *source;
    uint8_t *destination;
    int type, rows, columns, cross;
};

static int Ordinary(int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS_R4: return GGML_TYPE_IQ2_XXS;
        case GGML_TYPE_IQ2_XS_R4: return GGML_TYPE_IQ2_XS;
        case GGML_TYPE_IQ2_S_R4: return GGML_TYPE_IQ2_S;
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ1_M: case GGML_TYPE_Q2_0:
            return type;
        default: return -1;
    }
}

// Inverse of the CPU R4 repacker's prefix-XOR sign permutation.
__device__ unsigned Unsign(unsigned x) { return (x ^ (x << 1)) & 127; }

__global__ void Restore(const WeightCopy *copies) {
    const WeightCopy w = copies[blockIdx.y];
    const int blocks = w.columns / (w.type == GGML_TYPE_Q2_0 ? 64 : 256);
    const int lane = threadIdx.x%32;
    // A warp restores one quantized block with contiguous field stores. Keep
    // the grid bounded: one CTA per block would waste scheduling bandwidth.
    for (int i = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
         i < w.rows*blocks; i += gridDim.x*(blockDim.x/32)) {
        const int row = i/blocks, col = i%blocks;
        const int srcRow = w.cross ? 2*(row%(w.rows/2)) + row/(w.rows/2) : row;
        const int rlane = srcRow%4, r4 = (srcRow/4)*blocks+col;
        if (w.type == GGML_TYPE_IQ2_XS_R4) {
            const auto &s = reinterpret_cast<const block_iq2_xs_r4 *>(w.source)[r4];
            auto &d = reinterpret_cast<block_iq2_xs *>(w.destination)[i];
            if (lane == 0) d.d = s.d[rlane];
            if (lane < 8) d.scales[lane] = s.scales[4*lane+rlane];
            const unsigned q = s.qs[16*(lane/4)+4*rlane+lane%4];
            d.qs[lane] = (q & 511) | (Unsign(q>>9)<<9);
        } else if (w.type == GGML_TYPE_IQ2_XXS_R4) {
            const auto &s = reinterpret_cast<const block_iq2_xxs_r4 *>(w.source)[r4];
            auto &d = reinterpret_cast<block_iq2_xxs *>(w.destination)[i];
            if (lane == 0) d.d = s.d[rlane];
            reinterpret_cast<uint8_t *>(d.qs)[8*(lane/4)+lane%4] =
                s.qs[16*(lane/4)+4*rlane+lane%4];
            if (lane < 8) {
                uint32_t signs = 0, scale = 0;
                for (int j = 0; j < 4; ++j) {
                    const unsigned q = s.sas[16*lane+4*rlane+j];
                    signs |= Unsign(q>>1) << (7*j);
                    scale |= (q & 1) << j;
                }
                signs |= scale << 28;
                d.qs[4*lane+2] = signs; d.qs[4*lane+3] = signs>>16;
            }
        } else if (w.type == GGML_TYPE_IQ2_S_R4) {
            const auto &s = reinterpret_cast<const block_iq2_s_r4 *>(w.source)[r4];
            auto &d = reinterpret_cast<block_iq2_s *>(w.destination)[i];
            if (lane == 0) d.d = s.d[rlane];
            if (lane < 8) {
                d.scales[lane] = s.scales[4*lane+rlane];
                d.qh[lane] = s.qh[4*lane+rlane];
            }
            d.qs[lane] = s.qs[16*(lane/4)+4*rlane+lane%4];
            d.qs[32+lane] = s.signs[16*(lane/4)+4*rlane+lane%4];
        } else {
            const int bytes = w.type == GGML_TYPE_Q2_0 ? sizeof(block_q2_0) :
                w.type == GGML_TYPE_IQ1_M ? sizeof(block_iq1_m) :
                w.type == GGML_TYPE_IQ2_XS ? sizeof(block_iq2_xs) :
                w.type == GGML_TYPE_IQ2_S ? sizeof(block_iq2_s) : sizeof(block_iq2_xxs);
            for (int b = lane; b < bytes; b += 32)
                w.destination[size_t(i)*bytes+b] = w.source[size_t(srcRow*blocks+col)*bytes+b];
        }
    }
}

static size_t Align(size_t bytes) { return (bytes+255)&~size_t(255); }
} // namespace fastllm_gguf_stream

bool FastllmCudaMergeMOEGGUFHost(const fastllm::Data &input,
        fastllm::Data &gate, fastllm::Data &workspace, fastllm::Data &output,
        fastllm::Data **weights, int expertCount, const int32_t *indices,
        const float *scores, int topk, const std::unordered_set<int> &experts,
        bool crossSwiglu) {
    using namespace fastllm_gguf_stream;
    using fastllm_gguf_moe::AllocateTensor;
    if (!weights || !indices || !scores || input.dims.size() != 2 ||
        input.dims[0] <= 32 || input.dims[0] > 4096 || topk <= 0 || topk > 32 ||
        expertCount <= 0 || expertCount > 1024 || experts.empty() || experts.count(0) ||
        input.dataDevice != fastllm::CUDA || !input.cudaData ||
        !fastllm_gguf_moe::IsActivationType(input.dataType)) return false;
    const int device = FastllmCudaGetDevice(), rows = input.dims[0], hidden = input.dims[1];
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;
    int gt = -1, dt = -1, inter = 0, maxBlocks = 0;
    size_t packedBytes = 0;
    struct Source { const fastllm::Data *weight; size_t offset; int slot; };
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
                w->dims[1]%(w->ggmlType == GGML_TYPE_Q2_0 ? 64 : 256) ||
                (!w->cpuData && (w->numasData.empty() ||
                 w->dims[0]%w->numasData.size() ||
                 std::any_of(w->numasData.begin(), w->numasData.end(),
                    [](const uint8_t *p) { return p == nullptr; })))) return false;
            sources.push_back({w, packedBytes, 2*(e-1)+part});
            packedBytes += Align(w->GetBytes());
            maxBlocks = std::max(maxBlocks, w->dims[0] *
                (w->dims[1]/(w->ggmlType == GGML_TYPE_Q2_0 ? 64 : 256)));
        }
    }
    if (sources.empty() || sources.size() != experts.size()*2) return false;
    // IQ1_M gate/up currently uses per-route DP4A in grouped MMQ. On long
    // prefill, the existing per-expert GEMM reuses these weights much better.
    // Keep that implementation until a matrix IQ1_M gate kernel is available.
    if (gt == GGML_TYPE_IQ1_M) return false;
    const size_t mmqBytes = FastllmCudaMoeGGUFGroupedWorkspaceBytes(
        gt, dt, rows, hidden, inter, expertCount, topk);
    if (!mmqBytes) return false;
    const size_t tableOffset = 2*packedBytes;
    const size_t descOffset = tableOffset+Align(2*expertCount*sizeof(void *));
    const size_t indexOffset = descOffset+Align(sources.size()*sizeof(WeightCopy));
    const size_t scoreOffset = indexOffset+Align(size_t(rows)*topk*sizeof(int32_t));
    const size_t mmqOffset = scoreOffset+Align(size_t(rows)*topk*sizeof(float));
    const size_t bytes = mmqOffset+mmqBytes;
    if (bytes > size_t(INT32_MAX)) return false;
    const size_t gateBytes = size_t(rows)*topk*inter*(input.dataType == fastllm::FLOAT32 ? 4 : 2);
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t reusable = (workspace.cudaData ? workspace.expansionBytes : 0) +
        (gate.cudaData ? gate.expansionBytes : 0);
    if (bytes+gateBytes+256*size_t(1024*1024) > freeBytes+reusable) return false;
    AllocateTensor(workspace, fastllm::INT8, {int(bytes)}, device);
    AllocateTensor(gate, input.dataType, {rows*topk, inter}, device);
    AllocateTensor(output, input.dataType, {rows, hidden}, device);
    auto *base = static_cast<uint8_t *>(workspace.cudaData);
    const auto stream = cudaStreamPerThread;
    std::vector<const void *> table(2*expertCount, nullptr);
    std::vector<WeightCopy> copies;
    for (const auto &src : sources) {
        const auto &w = *src.weight;
        auto *target = base+src.offset;
        if (w.cpuData) {
            CUDA_CHECK(cudaMemcpyAsync(target, w.cpuData, w.GetBytes(), cudaMemcpyHostToDevice, stream));
        } else {
            const size_t shardBytes = w.GetBytes()/w.numasData.size();
            for (size_t node = 0; node < w.numasData.size(); ++node)
                CUDA_CHECK(cudaMemcpyAsync(target+node*shardBytes, w.numasData[node],
                    shardBytes, cudaMemcpyHostToDevice, stream));
        }
        const bool cross = crossSwiglu && src.slot%2 == 0;
        const bool restore = cross || Ordinary(w.ggmlType) != w.ggmlType;
        table[src.slot] = restore ? target+packedBytes : target;
        if (restore) copies.push_back({target, target+packedBytes, w.ggmlType,
            w.dims[0], w.dims[1], int(cross)});
    }
    CUDA_CHECK(cudaMemcpyAsync(base+tableOffset, table.data(), table.size()*sizeof(void *), cudaMemcpyHostToDevice, stream));
    if (!copies.empty()) CUDA_CHECK(cudaMemcpyAsync(base+descOffset, copies.data(),
        copies.size()*sizeof(WeightCopy), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(base+indexOffset, indices, size_t(rows)*topk*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(base+scoreOffset, scores, size_t(rows)*topk*sizeof(float), cudaMemcpyHostToDevice, stream));
    if (!copies.empty()) Restore<<<dim3(std::min(64, (maxBlocks+7)/8), copies.size()), 256, 0, stream>>>(
        reinterpret_cast<const WeightCopy *>(base+descOffset));
    CUDA_CHECK(cudaGetLastError());
    const bool ok = FastllmCudaMoeGGUFGrouped(input, gate, output, base+tableOffset,
        reinterpret_cast<const int32_t *>(base+indexOffset),
        reinterpret_cast<const float *>(base+scoreOffset), base+mmqOffset,
        gt, dt, hidden, inter, expertCount, topk);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // A CUDA launch failure is an error, not permission to execute a second path.
    fastllm::AssertInFastLLM(ok, "GGUF NUMA grouped prefill launch failed.");
    return true;
}
