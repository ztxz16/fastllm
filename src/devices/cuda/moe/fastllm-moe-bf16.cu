#include "utils/hybrid_moe.h"
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <algorithm>
#include "moe_cache_config.h"
#include "devices/moe_frequency_policy.h"
#include "devices/moe_decode_scheduler.h"
#include <stdexcept>
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

namespace fastllm {
namespace {
void CheckBF16(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("BF16 hybrid MoE: ") + message);
}
void CudaBF16(cudaError_t error) { CheckBF16(error == cudaSuccess, cudaGetErrorString(error)); }
__global__ void ReduceBF16Routes(const float *parts, const float *gpuParts, const int *owners, const int *ids, __nv_bfloat16 *output,
                                  int rows, int hidden, int topk) {
    int d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    extern __shared__ int order[];
    // Stable expert-id order is prepared once per CTA, including tied ids.
    for (int r = threadIdx.x; r < topk; r += blockDim.x) {
        int rank = 0, expert = ids[row * topk + r];
        for (int j = 0; j < topk; ++j)
            rank += ids[row * topk + j] < expert || (ids[row * topk + j] == expert && j < r);
        order[rank] = r;
    }
    __syncthreads();
    if (d >= hidden || row >= rows) return;
    float sum = 0;
    for (int rank = 0; rank < topk; ++rank) {
        int r = order[rank];
        sum += (owners && owners[row * topk + r] >= 0 ? gpuParts : parts)[((size_t)row * topk + r) * hidden + d];
    }
    output[(size_t)row * hidden + d] = __float2bfloat16(sum);
}
// Small routed blocks are bandwidth dominated. Submit all resident routes
// together; the packed route order keeps repeated experts adjacent in L2.
// FP32 accumulation and explicit BF16 boundaries work without Tensor Cores.
__device__ float SumBF16Warp(float value) {
    for (int shift = 16; shift; shift /= 2) value += __shfl_down_sync(0xffffffff, value, shift);
    return value;
}
__global__ void GateBF16Routes(const __nv_bfloat16 *input, const uint8_t *records,
        const int *owners, const int *routes, __nv_bfloat16 *middle,
        int hidden, int inter, int topk, size_t recordBytes) {
    int route = routes[blockIdx.y], column = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (column >= inter) return;
    const auto *x = input + size_t(route / topk) * hidden;
    const auto *gate = (const __nv_bfloat16 *)(records + size_t(owners[route]) * recordBytes) + size_t(column) * hidden;
    const auto *up = gate + size_t(inter) * hidden;
    float g = 0, u = 0;
    for (int k = lane; k < hidden; k += 32) {
        float a = __bfloat162float(x[k]);
        g += a * __bfloat162float(gate[k]); u += a * __bfloat162float(up[k]);
    }
    g = SumBF16Warp(g); u = SumBF16Warp(u);
    if (lane == 0) {
        g = __bfloat162float(__float2bfloat16(g)); u = __bfloat162float(__float2bfloat16(u));
        middle[size_t(route) * inter + column] = __float2bfloat16(g / (1.f + expf(-g)) * u);
    }
}
__global__ void DownBF16Routes(const __nv_bfloat16 *middle, const uint8_t *records,
        const int *owners, const int *routes, const float *scores, float *parts,
        int hidden, int inter, size_t gateBytes, size_t recordBytes) {
    int route = routes[blockIdx.y], column = blockIdx.x * 8 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (column >= hidden) return;
    const auto *x = middle + size_t(route) * inter;
    const auto *weight = (const __nv_bfloat16 *)(records + size_t(owners[route]) * recordBytes + gateBytes) + size_t(column) * inter;
    float sum = 0;
    for (int k = lane; k < inter; k += 32) sum += __bfloat162float(x[k]) * __bfloat162float(weight[k]);
    sum = SumBF16Warp(sum);
    if (lane == 0) parts[size_t(route) * hidden + column] =
        __bfloat162float(__float2bfloat16(sum * scores[route]));
}
void PrepareBF16(Data &data, DataType type, const std::vector<int> &dims, int device) {
    data.dataType = type; data.UpdateUnitSize(); data.Resize(dims);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false); data.Allocate(false);
}
using ClockBF16 = std::chrono::steady_clock;
double MicrosecondsBF16(ClockBF16::time_point start) {
    return std::chrono::duration<double, std::micro>(ClockBF16::now() - start).count();
}

}
struct HybridMoeBF16::Impl {
    int device, hidden = 0, inter = 0, experts = 0;
    std::vector<std::vector<std::unique_ptr<Data>>> owned;
    std::vector<std::vector<Data *>> tables;
    std::vector<uint16_t> input;
    std::vector<int32_t> indices, owners;
    std::vector<float> scores, parts;
    Data gpuParts, gpuResults, gpuOwners, gpuRoutes, middle;
    std::vector<std::pair<Data *, Data *>> sources;
    std::unique_ptr<MoeFrequencyPolicy> frequency;
    std::vector<MoeDecodeOverlapScheduler> timings;
    MoeFrequencyPolicy::RuntimeBudget admissionBudget;
    int slots = 0, staging = 0, firstLayer = -1, lastLayer = -1;
    bool prefill = false;
    size_t gateBytes = 0, recordBytes = 0;
    uint8_t *records = nullptr, *upload = nullptr;
    cudaStream_t copyStream = nullptr;
    cudaEvent_t metadata = nullptr, copyStart = nullptr, copyDone = nullptr, residentStart = nullptr, residentDone = nullptr;
    std::vector<cudaEvent_t> ready, stagedStart, stagedDone;
    HybridMoeBF16::Stats stats;
    explicit Impl(int device) : device(device) {}
    ~Impl() {
        int previous; cudaGetDevice(&previous); cudaSetDevice(device); cudaDeviceSynchronize();
        for (auto event : {metadata, copyStart, copyDone, residentStart, residentDone}) if (event) cudaEventDestroy(event);
        for (const auto &events : {ready, stagedStart, stagedDone}) for (auto event : events) if (event) cudaEventDestroy(event);
        if (copyStream) cudaStreamDestroy(copyStream);
        cudaFree(records); if (upload) cudaFreeHost(upload);
        cudaSetDevice(previous);
    }
    // Immutable checkpoint rows are copied through a bounded pinned buffer.
    // Each staged expert is uploaded once for the entire request block.
    void Pack(int key, uint8_t *destination) {
        int layer = key / experts, expert = key % experts;
        std::memcpy(destination, sources[layer].first->cpuData + size_t(expert) * gateBytes, gateBytes);
        std::memcpy(destination + gateBytes, sources[layer].second->cpuData + size_t(expert) * (recordBytes - gateBytes), recordBytes - gateBytes);
    }
    void Copy(int key, int slot, int stagingSlot, cudaStream_t stream) {
        Pack(key, upload + size_t(stagingSlot) * recordBytes);
        CudaBF16(cudaMemcpyAsync(records + size_t(slot) * recordBytes,
            upload + size_t(stagingSlot) * recordBytes, recordBytes, cudaMemcpyHostToDevice, stream));
        ++stats.uploads;
    }
    void Compute(const Data &input, const Data &scores, int offset, int count, int topk) {
        if (!count) return;
        const int *routes = (const int *)gpuRoutes.cudaData + offset;
        // A bounded y grid supports large prefill blocks without changing
        // whole-expert admission or duplicating weight transfers.
        for (int first = 0; first < count; first += 65535) {
            int n = std::min(65535, count - first);
            GateBF16Routes<<<dim3((inter + 7) / 8, n), 256, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)input.cudaData, records, (const int *)gpuOwners.cudaData,
                routes + first, (__nv_bfloat16 *)middle.cudaData, hidden, inter, topk, recordBytes);
            DownBF16Routes<<<dim3((hidden + 7) / 8, n), 256, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)middle.cudaData, records, (const int *)gpuOwners.cudaData,
                routes + first, (const float *)scores.cudaData, (float *)gpuResults.cudaData,
                hidden, inter, gateBytes, recordBytes);
        }
        CudaBF16(cudaGetLastError());
    }
};
HybridMoeBF16::HybridMoeBF16(const std::vector<std::pair<Data *, Data *>> &layers, int device)
        : impl(new Impl(device)) {
#ifdef USE_NUMAS
    CheckBF16(!layers.empty(), "missing layers");
    impl->owned.resize(layers.size()); impl->tables.resize(layers.size());
    impl->sources = layers; impl->timings.resize(layers.size());
    for (size_t l = 0; l < layers.size(); ++l) {
        Data *gate = layers[l].first, *down = layers[l].second;
        if (!gate && !down) continue;
        CheckBF16(gate && down && gate->dataType == BFLOAT16 && down->dataType == BFLOAT16 &&
            gate->dims.size() == 3 && down->dims.size() == 3 && gate->cpuData && down->cpuData &&
            gate->dataDevice == DataDevice::CPU && down->dataDevice == DataDevice::CPU,
            "requires resident packed BF16 weights");
        int e = gate->dims[0], h = gate->dims[2], i = gate->dims[1] / 2;
        CheckBF16(e > 0 && h > 0 && i > 0 && gate->dims[1] == 2 * i && down->dims == std::vector<int>({e, h, i}) &&
            (!impl->experts || (impl->experts == e && impl->hidden == h && impl->inter == i)),
            "inconsistent expert shapes");
        impl->experts = e; impl->hidden = h; impl->inter = i;
        if (impl->firstLayer < 0) impl->firstLayer = l; impl->lastLayer = l;
        auto &table = impl->tables[l]; table.resize(2 * (e + 1), nullptr);
        std::vector<std::pair<Data *, std::string>> pending;
        for (int expert = 0; expert < e; ++expert) for (int projection = 0; projection < 2; ++projection) {
            auto data = std::make_unique<Data>(BFLOAT16,
                projection ? std::vector<int>{h, i} : std::vector<int>{2 * i, h});
            data->Allocate(); data->name = (projection ? down : gate)->name + "." + std::to_string(expert);
            data->weightType = WeightType::LINEAR; data->isModelWeight = true;
            std::memcpy(data->cpuData, (projection ? down : gate)->cpuData + expert * data->GetBytes(), data->GetBytes());
            table[2 * (expert + 1) + projection] = data.get();
            pending.emplace_back(data.get(), projection ? "linear" : "linearSwiglu");
            impl->owned[l].push_back(std::move(data));
        }
        RegisterNumasMoeWeightBatch(pending);
    }
    auto &w = *impl;
    CheckBF16(w.firstLayer >= 0, "missing host experts");
    CudaBF16(cudaSetDevice(device));
    w.gateBytes = size_t(w.hidden) * w.inter * 4;
    w.recordBytes = size_t(w.hidden) * w.inter * 6;
    const int capacity = std::min<uint64_t>(GetMoeCudaCacheDeviceBytes(device) / w.recordBytes, layers.size() * w.experts + 16);
    if (capacity > 0) {
        w.staging = std::min(16, std::max(1, capacity / 8)); w.slots = capacity - w.staging;
        CudaBF16(cudaMalloc(&w.records, size_t(capacity) * w.recordBytes));
        CudaBF16(cudaMallocHost(&w.upload, size_t(w.staging) * w.recordBytes));
        CudaBF16(cudaStreamCreateWithFlags(&w.copyStream, cudaStreamNonBlocking));
        for (auto *event : {&w.metadata, &w.copyStart, &w.copyDone, &w.residentStart, &w.residentDone})
            CudaBF16(cudaEventCreate(event));
        w.ready.resize(w.staging); w.stagedStart.resize(w.staging); w.stagedDone.resize(w.staging);
        for (int i = 0; i < w.staging; ++i) for (auto *event : {&w.ready[i], &w.stagedStart[i], &w.stagedDone[i]})
            CudaBF16(cudaEventCreate(event));
        w.admissionBudget.savedUsPerKey.resize(layers.size() * w.experts);
        w.frequency = std::make_unique<MoeFrequencyPolicy>(std::vector<int>(layers.size() * w.experts, 0),
            std::vector<int>(w.slots, 0), layers.size(), GetMoeCacheConfig(),
            std::vector<uint64_t>(layers.size() * w.experts, w.recordBytes));
    }

#else
    throw std::runtime_error("BF16 hybrid MoE requires a NUMA-enabled build");
#endif
}
HybridMoeBF16::~HybridMoeBF16() = default;
HybridMoeBF16::Stats HybridMoeBF16::GetStats() const { return impl->stats; }
void HybridMoeBF16::SetPrefill(bool prefill) { impl->prefill = prefill; }
void HybridMoeBF16::Run(const Data &input, const Data &index, const Data &score,
                        Data &output, int layer, const std::function<void()> &launchParallel) {
#ifdef USE_NUMAS
    auto &w = *impl;
    CheckBF16(layer >= 0 && layer < int(w.tables.size()) && !w.tables[layer].empty(), "invalid layer");
    CheckBF16(input.dataDevice == DataDevice::CUDA && input.dataType == BFLOAT16 && input.dims.size() == 2 &&
        input.dims[1] == w.hidden && index.dataDevice == DataDevice::CUDA && index.dataType == INT32 &&
        index.dims.size() == 2 && index.dims[0] == input.dims[0] && score.dataDevice == DataDevice::CUDA &&
        score.dataType == FLOAT32 && score.dims == index.dims, "invalid input");
    CudaBF16(cudaSetDevice(w.device));
    CudaBF16(cudaStreamSynchronize(cudaStreamPerThread));
    const int rows = input.dims[0], topk = index.dims[1], routes = rows * topk;
    CheckBF16(rows > 0 && topk > 0 && topk <= std::min(w.experts, 1024), "invalid route geometry");
    w.input.resize((size_t)rows * w.hidden); w.indices.resize(routes); w.scores.resize(routes);
    w.owners.assign(routes, -1); w.parts.resize((size_t)routes * w.hidden);
    CudaBF16(cudaMemcpyAsync(w.input.data(), input.cudaData, input.GetBytes(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CudaBF16(cudaMemcpyAsync(w.indices.data(), index.cudaData, index.GetBytes(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CudaBF16(cudaMemcpyAsync(w.scores.data(), score.cudaData, score.GetBytes(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CudaBF16(cudaStreamSynchronize(cudaStreamPerThread));
    for (int e : w.indices) CheckBF16(e >= 0 && e < w.experts, "invalid expert id");
    std::vector<std::vector<int>> grouped(w.experts);
    for (int r = 0; r < routes; ++r) grouped[w.indices[r]].push_back(r);
    std::vector<int> resident, misses, reuse, packedRoutes;
    auto &timing = w.timings[layer];
    int selected = 0, residentRoutes = 0, stagedRoutes = 0;
    if (w.frequency) {
        if (!w.prefill && layer == w.firstLayer) w.frequency->BeginStep();
        std::vector<int> unique;
        for (int e = 0; e < w.experts; ++e) if (!grouped[e].empty()) unique.push_back(e);
        if (w.prefill) {
            std::vector<int> counts(w.experts);
            for (int e : unique) counts[e] = grouped[e].size();
            w.frequency->ObservePrefill(layer * w.experts, counts, rows, true);
        } else w.frequency->Observe(layer * w.experts, unique.data(), unique.size());
        for (int e : unique) {
            if (w.frequency->Slot(layer * w.experts + e) >= 0) resident.push_back(e);
            else misses.push_back(e);
        }
        std::stable_sort(misses.begin(), misses.end(), [&](int a, int b) { return grouped[a].size() > grouped[b].size(); });
        for (int e : resident) residentRoutes += grouped[e].size();
        for (int e : misses) reuse.push_back(grouped[e].size());
        selected = std::min(w.staging, timing.SelectMisses(misses.size(), residentRoutes, reuse.data()));
        for (int i = 0; i < selected; ++i) stagedRoutes += grouped[misses[i]].size();
        auto append = [&](int e, int slot) {
            for (int r : grouped[e]) { w.owners[r] = slot; packedRoutes.push_back(r); }
        };
        for (int e : resident) append(e, w.frequency->Slot(layer * w.experts + e));
        for (int i = 0; i < selected; ++i) append(misses[i], w.slots + i);
        PrepareBF16(w.gpuResults, FLOAT32, {routes, w.hidden}, w.device);
        PrepareBF16(w.gpuOwners, INT32, {routes}, w.device);
        PrepareBF16(w.gpuRoutes, INT32, {std::max(1, int(packedRoutes.size()))}, w.device);
        PrepareBF16(w.middle, BFLOAT16, {routes, w.inter}, w.device);
        CudaBF16(cudaMemcpyAsync(w.gpuOwners.cudaData, w.owners.data(), routes * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
        if (!packedRoutes.empty()) CudaBF16(cudaMemcpyAsync(w.gpuRoutes.cudaData, packedRoutes.data(), packedRoutes.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
    double submitUs = 0;
    auto submit = [&] {
        auto start = ClockBF16::now();
        if (launchParallel) launchParallel();
        if (w.frequency) {
            CudaBF16(cudaEventRecord(w.metadata, cudaStreamPerThread));
            CudaBF16(cudaStreamWaitEvent(w.copyStream, w.metadata, 0));
            CudaBF16(cudaEventRecord(w.copyStart, w.copyStream));
            for (int i = 0; i < selected; ++i) {
                w.Copy(layer * w.experts + misses[i], w.slots + i, i, w.copyStream);
                CudaBF16(cudaEventRecord(w.ready[i], w.copyStream));
            }
            CudaBF16(cudaEventRecord(w.copyDone, w.copyStream));
            int offset = 0;
            CudaBF16(cudaEventRecord(w.residentStart, cudaStreamPerThread));
            w.Compute(input, score, offset, residentRoutes, topk);
            offset += residentRoutes;
            CudaBF16(cudaEventRecord(w.residentDone, cudaStreamPerThread));
            for (int i = 0; i < selected; ++i) {
                CudaBF16(cudaStreamWaitEvent(cudaStreamPerThread, w.ready[i], 0));
                CudaBF16(cudaEventRecord(w.stagedStart[i], cudaStreamPerThread));
                w.Compute(input, score, offset, grouped[misses[i]].size(), topk);
                offset += grouped[misses[i]].size();
                CudaBF16(cudaEventRecord(w.stagedDone[i], cudaStreamPerThread));
            }
        }
        submitUs = MicrosecondsBF16(start);
    };
    auto &table = w.tables[layer];
    HostMoePlacementScope placement(table[2]);
    auto cpuStart = ClockBF16::now();
    NumasMoeBF16Experts(w.input.data(), w.parts.data(), rows, table.data(), table.size(),
        w.indices.data(), w.owners.data(), w.scores.data(), topk, layer, submit);
    // GPU submission now overlaps the NUMA gate queue. Subtracting its full
    // duration would also subtract CPU work that ran concurrently.
    double cpuUs = MicrosecondsBF16(cpuStart);
    int cpuRoutes = routes - residentRoutes - stagedRoutes;
    w.stats.cpuRoutes += cpuRoutes; w.stats.gpuRoutes += residentRoutes + stagedRoutes; w.stats.cacheHits += residentRoutes;
    PrepareBF16(w.gpuParts, FLOAT32, {routes, w.hidden}, w.device);
    PrepareBF16(output, BFLOAT16, input.dims, w.device);
    CudaBF16(cudaMemcpyAsync(w.gpuParts.cudaData, w.parts.data(), w.parts.size() * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    ReduceBF16Routes<<<dim3((w.hidden + 255) / 256, rows), 256, topk * sizeof(int), cudaStreamPerThread>>>(
        (const float *)w.gpuParts.cudaData, (const float *)w.gpuResults.cudaData, (const int *)w.gpuOwners.cudaData,
        (const int *)index.cudaData, (__nv_bfloat16 *)output.cudaData, rows, w.hidden, topk);
    CudaBF16(cudaGetLastError());
    if (w.frequency) {
        CudaBF16(cudaStreamSynchronize(cudaStreamPerThread));
        CudaBF16(cudaStreamSynchronize(w.copyStream));
        auto elapsed = [&](cudaEvent_t a, cudaEvent_t b) { float ms; CudaBF16(cudaEventElapsedTime(&ms, a, b)); return ms * 1000.; };
        if (cpuRoutes) timing.cpuExpert.Observe(cpuUs / cpuRoutes);
        if (residentRoutes) timing.residentExpert.Observe(elapsed(w.residentStart, w.residentDone) / residentRoutes);
        if (selected) {
            timing.copiedExpert.Observe(elapsed(w.copyStart, w.copyDone) / selected);
            double compute = 0;
            for (int i = 0; i < selected; ++i) compute += elapsed(w.stagedStart[i], w.stagedDone[i]);
            timing.stagedExpert.Observe(compute / stagedRoutes);
        }
        timing.dispatch.Observe(submitUs); ++timing.calls;
        if (timing.copiedExpert.initialized)
            w.admissionBudget.uploadUsPerByte = timing.copiedExpert.us / w.recordBytes;
        if (timing.cpuExpert.initialized && (timing.residentExpert.initialized || timing.stagedExpert.initialized)) {
            const double gpuUs = timing.residentExpert.initialized ? timing.residentExpert.us : timing.stagedExpert.us;
            for (int e = 0; e < w.experts; ++e) if (!grouped[e].empty()) {
                const double saved = std::max(0.0, timing.cpuExpert.us - gpuUs) * grouped[e].size();
                auto &estimate = w.admissionBudget.savedUsPerKey[layer * w.experts + e];
                estimate = estimate > 0 ? estimate * .9 + saved * .1 : saved;
            }
        }
        if (w.prefill) {
            std::vector<int> active;
            for (int e = 0; e < w.experts; ++e) if (!grouped[e].empty()) active.push_back(e);
            for (int i = 0; i < selected; ++i) {
                const auto admission = w.frequency->SelectPrefill(layer * w.experts + misses[i],
                    layer * w.experts, active, w.frequency->HasDecodeHistory());
                if (admission.slot < 0) continue;
                // Promote a record already uploaded for useful GPU work;
                // CPU-only prefill experts incur no speculative PCIe copies.
                CudaBF16(cudaMemcpyAsync(w.records + size_t(admission.slot) * w.recordBytes,
                    w.records + size_t(w.slots + i) * w.recordBytes, w.recordBytes,
                    cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                w.frequency->Admit(layer, admission);
            }
        } else if (layer == w.lastLayer) {
            // The shared policy sees all layers before spending one global
            // admission budget. No record is overwritten while an expert kernel reads it.
            auto admissions = w.frequency->EndStep(&w.admissionBudget);
            for (size_t first = 0; first < admissions.size(); first += w.staging) {
                size_t end = std::min(admissions.size(), first + w.staging);
                for (size_t i = first; i < end; ++i) w.Copy(admissions[i].key, admissions[i].slot, i - first, w.copyStream);
                CudaBF16(cudaStreamSynchronize(w.copyStream));
            }
        }
    }

#else
    throw std::runtime_error("BF16 hybrid MoE requires a NUMA-enabled build");
#endif
}
}
