#pragma once
#include "devices/cuda/fastllm-cuda-disk-v41-graph.h"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include <cuda_bf16.h>
#include "fastllm-moe-deepseekv41-cache.cuh"
namespace disk_v41_graph {
struct alignas(64) Control {
    unsigned published;
    alignas(64) unsigned completed;
};
__global__ void Publish(Control *box, float *host, int *hostIds, float *hostScores, int *hostResident,
                        int *slots, const uint8_t *const *records, const __nv_bfloat16 *input, const int *ids,
                        const float *scores, int rows, int hidden, int topk, int experts) {
#if __CUDA_ARCH__ >= 700
    for (int c = threadIdx.x; c < rows * hidden; c += blockDim.x)
        host[c] = __bfloat162float(input[c]);
    for (int r = threadIdx.x; r < rows * topk; r += blockDim.x) {
        const int e = max(0, min(ids[r], experts - 1));
        hostIds[r] = e;
        hostScores[r] = scores[r];
        hostResident[r] = slots[r] = records[e] ? e : -1;
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x)
        atomicExch_system(&box->published, 1u);
#endif
}
__global__ void Wait(const Control *box) {
#if __CUDA_ARCH__ >= 700
    unsigned done;
    do {
        asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(done) : "l"(&box->completed) : "memory");
        if (!done)
            __nanosleep(128);
    } while (!done);
#endif
}
__global__ void Reduce(const float *gpu, const float *host, const int *slots, const int *ids,
                       __nv_bfloat16 *out, int hidden, int topk) {
    __shared__ int order[16];
    const int row = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (!threadIdx.x)
        for (int r = 0; r < topk; ++r) {
            int at = r;
            while (at && ids[row * topk + order[at - 1]] > ids[row * topk + r]) {
                order[at] = order[at - 1];
                --at;
            }
            order[at] = r;
        }
    __syncthreads();
    if (c < hidden) {
        float sum = 0;
        for (int j = 0; j < topk; ++j) {
            const int r = row * topk + order[j];
            sum = __fadd_rn(sum, slots[r] >= 0 ? gpu[size_t(r) * hidden + c] : host[size_t(r) * hidden + c]);
        }
        out[size_t(row) * hidden + c] = __float2bfloat16_rn(sum);
    }
}
} // namespace disk_v41_graph
struct FastllmCudaDiskV41Graph::Impl {
    int device, rows, hidden, topk;
    void *host = nullptr, *mapped = nullptr;
    size_t inputAt, outputAt, idsAt, scoresAt, residentAt, recordsAt;
    fastllm::Data slots{fastllm::INT32}, partial{fastllm::FLOAT32}, scratch{fastllm::INT8}, activation,
        quantized;
    cudaGraph_t graph = nullptr;
    ~Impl() {
        const int previous = FastllmCudaGetDevice();
        FastllmCudaSetDevice(device);
        if (graph)
            cudaGraphDestroy(graph);
        if (host)
            cudaFreeHost(host);
        slots.FreeSpace();
        partial.FreeSpace();
        scratch.FreeSpace();
        activation.FreeSpace();
        quantized.FreeSpace();
        FastllmCudaSetDevice(previous);
    }
    Host View(bool gpu = false) const {
        auto *p = static_cast<uint8_t *>(gpu ? mapped : host);
        return {reinterpret_cast<float *>(p + inputAt),  reinterpret_cast<float *>(p + outputAt),
                reinterpret_cast<int32_t *>(p + idsAt),  reinterpret_cast<int32_t *>(p + residentAt),
                reinterpret_cast<float *>(p + scoresAt), reinterpret_cast<const uint8_t **>(p + recordsAt)};
    }
};
FastllmCudaDiskV41Graph::FastllmCudaDiskV41Graph(std::unique_ptr<Impl> p) : impl(std::move(p)) {}
FastllmCudaDiskV41Graph::~FastllmCudaDiskV41Graph() = default;
void *FastllmCudaDiskV41Graph::Graph() const {
    return impl->graph;
}
FastllmCudaDiskV41Graph::Host FastllmCudaDiskV41Graph::Request() const {
    return impl->View();
}
void FastllmCudaDiskV41Graph::Reset() {
    auto *c = static_cast<disk_v41_graph::Control *>(impl->host);
    __atomic_store_n(&c->published, 0u, __ATOMIC_RELEASE);
    __atomic_store_n(&c->completed, 0u, __ATOMIC_RELEASE);
}
bool FastllmCudaDiskV41Graph::Published() const {
    return __atomic_load_n(&static_cast<disk_v41_graph::Control *>(impl->host)->published,
                           __ATOMIC_ACQUIRE) != 0;
}
void FastllmCudaDiskV41Graph::Complete() {
    __atomic_store_n(&static_cast<disk_v41_graph::Control *>(impl->host)->completed, 1u, __ATOMIC_RELEASE);
}
std::unique_ptr<FastllmCudaDiskV41Graph>
FastllmCudaDiskV41Graph::Create(const fastllm::Data &input, const fastllm::Data &index,
                                const fastllm::Data &score, fastllm::Data &output,
                                fastllm::DataType weightType, int gateType, int downType, int inter,
                                int experts, size_t gateBytes, size_t recordBytes, float limit) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    const bool native = weightType == NVFP4;
    if (weightType != DATA_GGUF_FORMAT && !native)
        return {};
    if (input.dataType != BFLOAT16 || input.dims.size() != 2 || input.dims[0] < 1 || input.dims[0] > 32 ||
        index.dataType != INT32 || score.dataType != FLOAT32 || index.dims != score.dims ||
        index.dims.size() != 2 || index.dims[0] != input.dims[0] || index.dims[1] < 1 || index.dims[1] > 16 ||
        (!native && !FastllmCudaV41GGUFSupported(gateType, downType)) || experts < 1 || inter < 1 ||
        input.dims[1] < 1 || input.dims[1] % 32 || inter % 32 || FastllmCudaGraphIsCapturingFast())
        return {};
    for (const Data *d : {&input, &index, &score})
        if (d->multiDeviceData || d->dataDevice != CUDA || !d->cudaData ||
            d->dataDeviceIds != std::vector<int>{device})
            return {};
    if (&output == &input || &output == &index || &output == &score || output.multiDeviceData)
        return {};
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || prop.major < 7 || !prop.canMapHostMemory)
        return {};
    auto p = std::make_unique<Impl>();
    p->device = device;
    p->rows = input.dims[0];
    p->hidden = input.dims[1];
    p->topk = index.dims[1];
    size_t bytes = 256;
    auto take = [&](size_t n) {
        auto at = bytes;
        bytes += (n + 255) & ~size_t(255);
        return at;
    };
    p->inputAt = take(size_t(p->rows) * p->hidden * 4);
    p->outputAt = take(size_t(p->rows) * p->topk * p->hidden * 4);
    p->idsAt = take(p->rows * p->topk * 4);
    p->scoresAt = take(p->rows * p->topk * 4);
    p->residentAt = take(p->rows * p->topk * 4);
    p->recordsAt = take(experts * sizeof(void *));
    if (cudaHostAlloc(&p->host, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    if (cudaHostGetDevicePointer(&p->mapped, p->host, 0) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    memset(p->host, 0, bytes);
    p->slots.Resize({p->rows, p->topk});
    p->partial.Resize({p->rows, p->topk, p->hidden});
    if (native) {
        p->activation.dataType = BFLOAT16;
        p->activation.UpdateUnitSize();
        p->activation.Resize({p->rows * p->topk, inter});
        p->quantized.dataType = BFLOAT16;
        p->quantized.UpdateUnitSize();
        p->quantized.Resize(input.dims);
        for (Data *d : {&p->activation, &p->quantized}) {
            d->ToDevice(CUDA, {device}, false);
            d->Allocate(false);
        }
    } else {
        const auto scratchBytes = FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(p->hidden, inter);
        if (!scratchBytes)
            return {};
        p->scratch.Resize({int(scratchBytes)});
        p->scratch.ToDevice(CUDA, {device}, false);
        p->scratch.Allocate(false);
    }
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(input.dims);
    for (Data *d : {&p->slots, &p->partial, &output}) {
        d->ToDevice(CUDA, {device}, false);
        d->Allocate(false);
    }
    const auto host = p->View(true);
    FastllmCudaMoeGGUFCacheView view{nullptr,
                                     static_cast<const int32_t *>(p->slots.cudaData),
                                     recordBytes,
                                     gateBytes,
                                     gateType,
                                     downType,
                                     p->hidden,
                                     inter,
                                     p->scratch.cudaData,
                                     p->scratch.GetBytes()};
    view.recordPointers = host.records;
    auto launch = [&] {
        disk_v41_graph::Publish<<<1, 256, 0, cudaStreamPerThread>>>(
            static_cast<disk_v41_graph::Control *>(p->mapped), host.input, host.ids, host.scores,
            host.resident, static_cast<int *>(p->slots.cudaData), host.records,
            static_cast<const __nv_bfloat16 *>(input.cudaData), static_cast<const int *>(index.cudaData),
            static_cast<const float *>(score.cudaData), p->rows, p->hidden, p->topk, experts);
        if (native) {
            // Preserve the eager block-32 FP8 activation boundary. Disk GPU
            // records have interleaved Gate/Up rows and native 17-byte blocks.
            if (!FastllmCudaDeepSeekV41QuantizeActivation(input, p->quantized))
                return false;
            using namespace fastllm::cuda::dsv41_cache;
            Gate<true, true><<<dim3(inter / 32, p->rows * p->topk), 512, 0, cudaStreamPerThread>>>(
                static_cast<const __nv_bfloat16 *>(p->quantized.cudaData),
                static_cast<const int32_t *>(p->slots.cudaData), nullptr,
                static_cast<const float *>(score.cudaData),
                static_cast<__nv_bfloat16 *>(p->activation.cudaData), p->hidden, inter, recordBytes, limit,
                p->topk, nullptr, false, host.records);
            Down<true><<<dim3(p->hidden / 8, p->rows * p->topk), 128, 0, cudaStreamPerThread>>>(
                static_cast<const __nv_bfloat16 *>(p->activation.cudaData),
                static_cast<const int32_t *>(p->slots.cudaData), nullptr,
                static_cast<float *>(p->partial.cudaData), p->hidden, inter, recordBytes, gateBytes, nullptr,
                false, host.records);
        } else
            for (int row = 0; row < p->rows; row += 8) {
                Data tile(BFLOAT16, {std::min(8, p->rows - row), p->hidden}, CUDA,
                          static_cast<uint8_t *>(input.cudaData) + size_t(row) * p->hidden * 2);
                tile.cudaDataBorrowed = true;
                tile.dataDeviceIds = {device};
                auto v = view;
                v.routeSlots += row * p->topk;
                if (!FastllmCudaMoeV41GGUFCacheCompute(
                        tile, p->activation, v, static_cast<const float *>(score.cudaData) + row * p->topk,
                        p->topk, limit,
                        static_cast<float *>(p->partial.cudaData) + size_t(row) * p->topk * p->hidden))
                    return false;
            }
        disk_v41_graph::Wait<<<1, 1, 0, cudaStreamPerThread>>>(
            static_cast<disk_v41_graph::Control *>(p->mapped));
        disk_v41_graph::Reduce<<<dim3((p->hidden + 255) / 256, p->rows), 256, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(p->partial.cudaData), host.output,
            static_cast<const int *>(p->slots.cudaData), host.ids,
            static_cast<__nv_bfloat16 *>(output.cudaData), p->hidden, p->topk);
        return cudaGetLastError() == cudaSuccess;
    };
    // Warm all dynamic allocations and lazy kernels before capture. Empty
    // residency and an already-completed producer make this a bounded warmup.
    static_cast<disk_v41_graph::Control *>(p->host)->completed = 1;
    if (!launch() || cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
        return {};
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        return {};
    const bool ok = launch();
    const auto ended = cudaStreamEndCapture(cudaStreamPerThread, &p->graph);
    if (!ok || ended != cudaSuccess || !p->graph) {
        cudaGetLastError();
        return {};
    }
    return std::unique_ptr<FastllmCudaDiskV41Graph>(new FastllmCudaDiskV41Graph(std::move(p)));
}
