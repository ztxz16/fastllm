// BF16 activations x per-channel FP8 weights, without a global BF16 weight copy.
// CUTLASS converts FP8 * FP32 scale to BF16 inside the WGMMA mainloop, preserving
// the weight rounding of the generic BF16 GEMM path. Split-K only changes the
// order of FP32 accumulation; it does not quantize activations.
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>

namespace fastllm_perchannel_sm90 {
using namespace cute;
constexpr int kRowsPerChunk = 128;

// This CUTLASS revision's mixed-input builder also queries bytes for fixed stages.
template<int Stages>
struct FixedStages : cutlass::gemm::collective::StageCount<Stages> {
    static constexpr int bytes = Stages;
};

template<int Columns, bool Partial>
struct Projection {
    using Tile = Shape<_64, Int<Columns>, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Element = std::conditional_t<Partial, float, cutlass::bfloat16_t>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        Element, cutlass::layout::ColumnMajor, Partial ? 4 : 8,
        Element, cutlass::layout::ColumnMajor, Partial ? 4 : 8,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        cute::tuple<cutlass::float_e4m3_t, float>, cutlass::layout::RowMajor, 16,
        cutlass::bfloat16_t, cutlass::layout::ColumnMajor, 8, float, Tile, Cluster,
        FixedStages<3>, cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static bool Configure() {
        return cudaFuncSetAttribute(cutlass::device_kernel<Kernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel::SharedStorageSize) == cudaSuccess;
    }
    static bool Run(const void *input, const uint8_t *weight, const float *scales,
                    void *output, int rows, int inner, int channels, int split) {
        typename Mainloop::Arguments mainloop;
        mainloop.ptr_A = reinterpret_cast<const cutlass::float_e4m3_t *>(weight);
        mainloop.dA = make_stride(int64_t(inner), _1{}, int64_t(inner / split));
        mainloop.ptr_B = reinterpret_cast<const cutlass::bfloat16_t *>(input);
        mainloop.dB = make_stride(int64_t(inner), _1{}, int64_t(inner / split));
        mainloop.ptr_S = scales;
        mainloop.dS = make_stride(_1{}, int64_t(channels), int64_t(0));
        mainloop.group_size = inner / split;
        auto stride = make_stride(_1{}, int64_t(channels), int64_t(channels) * rows);
        auto *result = reinterpret_cast<Element *>(output);
        typename Epilogue::Arguments epilogue{{1.f, 0.f}, result, stride, result, stride};
        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
            {channels, rows, inner / split, split}, mainloop, epilogue};
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess || gemm.get_workspace_size(args))
            return false;
        // Attributes were configured before capture. update() only builds host
        // parameters/TMA descriptors, so replay needs no allocation or setup.
        if (gemm.update(args, nullptr) != cutlass::Status::kSuccess) return false;
        if (gemm.run(cudaStreamPerThread) != cutlass::Status::kSuccess)
            throw std::runtime_error("FP8 per-channel W8A16 launch failed");
        return true;
    }
};

__global__ void PackPerChannel(const uint8_t *source, uint8_t *packed,
                               float *scales, int inner) {
    const size_t row = blockIdx.x;
    const auto *src = reinterpret_cast<const uint32_t *>(source + row * (inner + sizeof(float)));
    auto *dst = reinterpret_cast<uint32_t *>(packed + row * inner);
    for (int i = threadIdx.x; i < inner / 4; i += blockDim.x) dst[i] = src[i];
    if (!threadIdx.x) scales[row] = __uint_as_float(src[inner / 4]);
}

__global__ void ReducePartial(const float *partial, __nv_bfloat16 *output, size_t count, int split) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = 0.f;
        for (int s = 0; s < split; ++s) sum += partial[size_t(s) * count + i];
        output[i] = __float2bfloat16_rn(sum);
    }
}

int TileColumns(int rows) { return rows <= 16 ? 16 : rows <= 32 ? 32 : 64; }
int SplitCount(int rows, int inner, int channels, int sms) {
    const int channelTiles = (channels + 63) / 64;
    if (channelTiles >= 2 * sms) return 1;
    const int tiles = channelTiles * ((rows + TileColumns(rows) - 1) / TileColumns(rows));
    const int target = sms * (rows <= 16 ? 4 : 2);
    int split = 1;
    while (split < 8 && tiles * split < target && inner % (split * 2 * 128) == 0 &&
           inner / (split * 2) >= 512) split *= 2;
    return split;
}

struct PerChannelLayout {
    int device = -1, sms = 0, inner = 0, channels = 0;
    const void *source = nullptr;
    uint8_t *packed = nullptr;
    float *scales = nullptr;
    size_t scratchBytes = 0;
    float *unassignedScratch = nullptr;
    std::mutex mutex;
    std::map<std::thread::id, float *> scratch;
    ~PerChannelLayout() {
        int previous;
        if (device < 0 || cudaGetDevice(&previous) != cudaSuccess) return;
        cudaSetDevice(device);
        for (auto &item : scratch) cudaFree(item.second);
        cudaFree(unassignedScratch);
        cudaFree(packed);
        cudaFree(scales);
        cudaSetDevice(previous);
    }
};
struct Registry {
    std::mutex mutex;
    std::map<const fastllm::Data *, std::shared_ptr<PerChannelLayout>> weights;
};
Registry &Layouts() {
    // Data owns entries. Avoid CUDA calls during static registry destruction.
    static auto *registry = new Registry;
    return *registry;
}

bool CanPrepare() {
    return !FastllmCudaGraphIsCapturingFast() && FastllmCudaGetNcclForceSync();
}

bool SupportsWeight(const fastllm::Data &weight) {
    return !weight.isFake && weight.dataDevice == fastllm::DataDevice::CUDA &&
        weight.dataType == fastllm::DataType::FP8_E4M3_PERCHANNEL && weight.cudaData &&
        weight.dims.size() == 2 && weight.dims[1] >= 512 && weight.dims[1] % 128 == 0 &&
        weight.dims[0] > 0 && weight.dims[0] % 8 == 0 && FastllmCudaRuntimeArch() == 90;
}

std::shared_ptr<PerChannelLayout> GetLayout(fastllm::Data &weight, int inner, int channels) {
    const int device = FastllmCudaGetDevice();
    auto &registry = Layouts();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto it = registry.weights.find(&weight);
        if (it != registry.weights.end()) {
            auto entry = it->second;
            return entry->source == weight.cudaData && entry->device == device &&
                entry->inner == inner && entry->channels == channels ? entry : nullptr;
        }
    }
    if (!CanPrepare()) return nullptr;
    // No registry mutex across allocation/synchronization: other TP workers
    // can be waiting on this rank's stream or preparing their own weights.
    auto entry = std::make_shared<PerChannelLayout>();
    entry->device = device;
    entry->source = weight.cudaData;
    entry->inner = inner;
    entry->channels = channels;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || prop.major != 9 || prop.minor != 0)
        return nullptr;
    entry->sms = prop.multiProcessorCount;
    for (int rows = 1; rows <= kRowsPerChunk; ++rows) {
        int split = SplitCount(rows, inner, channels, entry->sms);
        if (split > 1) entry->scratchBytes = std::max(entry->scratchBytes,
            size_t(rows) * channels * split * sizeof(float));
    }
    if (!Projection<16, false>::Configure() || !Projection<32, false>::Configure() ||
        !Projection<64, false>::Configure() || !Projection<16, true>::Configure() ||
        !Projection<32, true>::Configure() || !Projection<64, true>::Configure()) return nullptr;
    auto allocate = [](void **p, size_t bytes) {
        auto status = cudaMalloc(p, bytes);
        if (status == cudaErrorMemoryAllocation) { cudaGetLastError(); return false; }
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return true;
    };
    if (!allocate(reinterpret_cast<void **>(&entry->packed), size_t(inner) * channels) ||
        !allocate(reinterpret_cast<void **>(&entry->scales), size_t(channels) * sizeof(float)) ||
        (entry->scratchBytes && !allocate(reinterpret_cast<void **>(&entry->unassignedScratch), entry->scratchBytes)))
        return nullptr;
    PackPerChannel<<<channels, 256, 0, cudaStreamPerThread>>>(
        static_cast<const uint8_t *>(weight.cudaData), entry->packed, entry->scales, inner);
    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess)
        throw std::runtime_error("FP8 per-channel packing failed");
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        // Mark ownership before publishing: if vector allocation throws, the
        // local entry still releases every allocation without a registry leak.
        if (weight.extraCudaData.size() < 7) weight.extraCudaData.resize(7, nullptr);
        auto inserted = registry.weights.emplace(&weight, entry);
        if (!inserted.second) return inserted.first->second;
    }
    return entry;
}

float *GetScratch(PerChannelLayout &entry) {
    if (!entry.scratchBytes) return nullptr;
    const auto thread = std::this_thread::get_id();
    {
        std::lock_guard<std::mutex> lock(entry.mutex);
        auto it = entry.scratch.find(thread);
        if (it != entry.scratch.end()) return it->second;
        if (entry.unassignedScratch) {
            float *pointer = entry.unassignedScratch;
            entry.unassignedScratch = nullptr;
            entry.scratch.emplace(thread, pointer);
            return pointer;
        }
    }
    if (!CanPrepare()) return nullptr;
    float *pointer = nullptr;
    auto status = cudaMalloc(&pointer, entry.scratchBytes);
    if (status == cudaErrorMemoryAllocation) { cudaGetLastError(); return nullptr; }
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    {
        std::lock_guard<std::mutex> lock(entry.mutex);
        entry.scratch.emplace(thread, pointer);
    }
    return pointer;
}

template<bool Partial>
bool Launch(const void *input, PerChannelLayout &layout, void *output, int rows, int split) {
    switch (TileColumns(rows)) {
        case 16: return Projection<16, Partial>::Run(input, layout.packed, layout.scales,
            output, rows, layout.inner, layout.channels, split);
        case 32: return Projection<32, Partial>::Run(input, layout.packed, layout.scales,
            output, rows, layout.inner, layout.channels, split);
        default: return Projection<64, Partial>::Run(input, layout.packed, layout.scales,
            output, rows, layout.inner, layout.channels, split);
    }
}
} // namespace

bool FastllmCudaPreparePerChannelW8A16SM90(fastllm::Data &weight) {
    using namespace fastllm_perchannel_sm90;
    if (!CanPrepare() || !SupportsWeight(weight)) return false;
    return bool(GetLayout(weight, weight.dims[1], weight.dims[0]));
}

bool FastllmCudaPerChannelW8A16(const void *input, fastllm::Data &weight, void *output,
                              int rows, int inner, int channels) {
    using namespace fastllm_perchannel_sm90;
    if (!input || !output || rows <= 0 || !SupportsWeight(weight) ||
        weight.dims[0] != channels || weight.dims[1] != inner) return false;
    auto layout = GetLayout(weight, inner, channels);
    if (!layout) return false;
    float *scratch = GetScratch(*layout);
    if (layout->scratchBytes && !scratch) return false;
    for (int offset = 0; offset < rows; offset += kRowsPerChunk) {
        const int count = std::min(kRowsPerChunk, rows - offset);
        const int split = SplitCount(count, inner, channels, layout->sms);
        const auto *x = static_cast<const __nv_bfloat16 *>(input) + size_t(offset) * inner;
        auto *y = static_cast<__nv_bfloat16 *>(output) + size_t(offset) * channels;
        if (split == 1) {
            if (!Launch<false>(x, *layout, y, count, 1)) return false;
        } else {
            if (!Launch<true>(x, *layout, scratch, count, split)) return false;
            const size_t elements = size_t(count) * channels;
            ReducePartial<<<(elements + 255) / 256, 256, 0, cudaStreamPerThread>>>(scratch, y, elements, split);
        }
    }
    return true;
}

void FastllmCudaReleasePerChannelW8A16(fastllm::Data &weight) {
    using namespace fastllm_perchannel_sm90;
    std::shared_ptr<PerChannelLayout> owned;
    auto &registry = Layouts();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto it = registry.weights.find(&weight);
        if (it == registry.weights.end()) return;
        owned = std::move(it->second);
        registry.weights.erase(it);
    }
    // cudaFree may synchronize; never hold the registry lock here.
}
