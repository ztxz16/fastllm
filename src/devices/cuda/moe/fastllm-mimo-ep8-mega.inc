#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mxfp4-ep-graph.h"
#include "devices/cuda/fastllm-mxfp4-ep-generic.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "fastllm-mimo-ep8-builtin.inc"
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>

#ifdef USE_ROCM
bool FastllmCudaMxfp4EpGraphSupported(const std::vector<int> &) { return false; }
bool FastllmCudaMxfp4EpGraphCapturing() { return false; }
struct FastllmMxfp4EpNativeState {};
std::shared_ptr<FastllmMxfp4EpNativeState> FastllmCudaCreateMxfp4EpNative(const std::vector<int> &,
    const std::vector<std::vector<std::vector<uint64_t>>> &) { return {}; }
void FastllmCudaBindMxfp4EpNativeWeights(FastllmMxfp4EpNativeState &, int) {}
std::array<void *, 2> FastllmCudaMxfp4EpPackedWeights(const FastllmMxfp4EpNativeState &, int, int) { return {}; }
std::vector<FastllmMxfp4EpGraph> FastllmCudaPrepareMxfp4EpNative(FastllmMxfp4EpNativeState &, int, int) { return {}; }
bool FastllmCudaReplayMxfp4EpGraph(const FastllmMxfp4EpGraph &, const void *, const int32_t *,
    const float *, void *, int, int, int, const int *) { return false; }
#endif

#ifndef FASTLLM_ENABLE_DSV41_MXFP4_SM120
bool FastllmGenericEpSupported(const std::vector<int> &) { return false; }
std::shared_ptr<FastllmGenericEp> FastllmCreateGenericEp(const std::vector<int> &, int, int, int, int) {
    throw std::runtime_error("SM120 grouped MXFP4 EP was not compiled.");
}
void FastllmBindGenericEp(FastllmGenericEp &, int, int, const std::vector<uint64_t> &) {
    throw std::runtime_error("SM120 grouped MXFP4 EP was not compiled.");
}
bool FastllmGenericEpTunedEp8(const std::vector<int> &) { return false; }
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8Decode(const std::vector<int> &) { return {}; }
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8(const std::vector<int> &, bool &densePrefill) {
    densePrefill = false;
    return {};
}
void FastllmBindGenericEp(FastllmGenericEp &, int, int, const std::vector<uint64_t> &, const std::array<void *,2> &) {
    throw std::runtime_error("SM120 grouped MXFP4 EP was not compiled.");
}
void FastllmRunGenericEp(FastllmGenericEp &, int, int, const void *, const int32_t *, const float *,
        void *, int, int, const int *) {
    throw std::runtime_error("SM120 grouped MXFP4 EP was not compiled.");
}
void FastllmRunGenericEp(FastllmGenericEp &, int, int, const void *, const int32_t *, const float *,
        void *, int, int, const int *, int) {
    throw std::runtime_error("SM120 grouped MXFP4 EP was not compiled.");
}
#endif

struct FastllmCudaMimoEp8Mega {
    int hidden = 0, topk = 0;
    std::shared_ptr<FastllmCudaMimoEp8BuiltinMega> builtin;
    std::shared_ptr<FastllmMxfp4EpNativeState> native;
    // The generic weight views are destroyed before the owning native state.
    std::shared_ptr<FastllmGenericEp> generic;
    static constexpr int SharedDecodeRows = 128;
    bool groupedDensePrefill = false;

    bool useGrouped(int rows, int densePrefixRows) const {
        if (!generic || !native) return bool(generic);
        if (rows <= SharedDecodeRows) return true;
        // All ranks must make the same choice even for ragged or masked graphs.
        if (!groupedDensePrefill || densePrefixRows != rows) return false;
        // In the 320–479 crossover, native wins several neighboring shapes.
        return rows <= 288 || rows == 384 || rows == 448 || (rows >= 480 && rows <= 768) || rows == 2048;
    }
    std::vector<int> devices;
    std::vector<int> layers;
    int preparedRows = 0;
    std::mutex prepareMutex;
    std::map<std::pair<int, int>, std::vector<FastllmMxfp4EpGraph>> graphs;
    FastllmMxfp4EpGraph Graph(int rank, int layer, int rows) {
        int capacity = 1;
        while (capacity < rows) {
            if (capacity > INT_MAX / 2) throw std::runtime_error("SM120 EP row count is too large.");
            capacity *= 2;
        }
        std::lock_guard<std::mutex> lock(prepareMutex);
        auto key = std::make_pair(layer, capacity);
        auto found = graphs.find(key);
        if (found == graphs.end()) {
            if (FastllmCudaMxfp4EpGraphCapturing())
                throw std::runtime_error("SM120 MXFP4 EP graph was not prepared before capture.");
            found = graphs.emplace(key, FastllmCudaPrepareMxfp4EpNative(*native, layer, capacity)).first;
        }
        return found->second.at(rank);
    }
};

bool FastllmCudaMimoEp8MegaWeights(FastllmCudaMimoEp8Mega &state, int rank, int layer, const void **out, bool *mxfp4) {
    return state.builtin && FastllmCudaMimoEp8BuiltinMegaWeights(*state.builtin, rank, layer, out, mxfp4);
}
bool FastllmCudaMimoEp8MegaWeights(FastllmCudaMimoEp8Mega &state, int rank, int layer, const void **out) {
    return FastllmCudaMimoEp8MegaWeights(state, rank, layer, out, nullptr);
}
bool FastllmCudaMimoEp8MegaOwnsPackedWeights(int device) {
    return FastllmCudaMimoEp8BuiltinMegaOwnsPackedWeights(device);
}

void FastllmCudaPrepareMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int maxRows) {
    if (!state.native || maxRows <= state.preparedRows) return;
    int capacity = 1;
    while (capacity < maxRows) {
        if (capacity > INT_MAX / 2) throw std::runtime_error("SM120 EP row count is too large.");
        capacity *= 2;
    }
    struct RestoreDevice {
        int device = FastllmCudaGetDevice();
        ~RestoreDevice() { FastllmCudaSetDevice(device); }
    } restoreDevice;
    // Instantiation may synchronize a device. Doing it under Graph's lock
    // after one rank has entered a peer collective can block every other rank.
    // Prepare on the coordinator, while all earlier peer work can still drain.
    for (int rows = state.preparedRows ? state.preparedRows * 2 : 1; ; rows *= 2) {
        // The same bucket may serve masked replays even when dense prefill uses grouped.
        if (!state.generic || rows > state.SharedDecodeRows)
            for (int layer : state.layers) state.Graph(0, layer, rows);
        state.preparedRows = rows;
        if (rows == capacity) break;
    }
}

bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &devices) {
    return FastllmCudaMimoEp8BuiltinMegaSupported(devices) ||
        FastllmCudaMxfp4EpGraphSupported(devices) || FastllmGenericEpSupported(devices);
}

size_t FastllmCudaMimoEp8PrefillWorkspaceBytes(int rows, int ranks) {
    // SM120 owns its CUDA workspace. Keep the borrowed prefill arena for SM90.
    return FastllmCudaMimoEp8BuiltinPrefillWorkspaceBytes(rows, ranks);
}

std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
        const std::vector<int> &devices,
        const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights) {
    return FastllmCudaCreateMimoEp8Mega(devices, weights, 8);
}

std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
        const std::vector<int> &devices,
        const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights, int topk) {
    using namespace fastllm;
    auto state = std::make_shared<FastllmCudaMimoEp8Mega>();
    state->devices = devices;
    const bool package = FastllmCudaMxfp4EpGraphSupported(devices);
    if (!package && !FastllmGenericEpSupported(devices)) {
        state->builtin = FastllmCudaCreateMimoEp8BuiltinMega(devices, weights, topk);
        return state;
    }
    if (weights.size() != devices.size()) throw std::runtime_error("Invalid SM120 EP weight rank count.");
    int h = 0, mid = 0, experts = 0;
    for (const auto &table : weights.front()) if (!table.empty()) {
        if (table.size() < 4 || table.size() % 2 || !table[3] || table[3]->dims.size() != 2)
            throw std::runtime_error("Invalid SM120 EP expert weight table");
        experts = (table.size() - 2) / 2;
        h = table[3]->dims[0]; mid = table[3]->dims[1];
        break;
    }
    if (!experts || experts % devices.size() || h <= 0 || mid <= 0)
        throw std::runtime_error("Invalid SM120 EP expert geometry");
    const char *option = std::getenv("FASTLLM_SM120_EP_BACKEND");
    const std::string mode = option ? option : "auto";
    if (mode != "auto" && mode != "grouped" && mode != "native")
        throw std::runtime_error("FASTLLM_SM120_EP_BACKEND accepts auto, grouped or native");
    const bool compatible = package && h == 4096 && mid == 2048 && experts == 256 && topk == 8;
    if (mode == "native" && !compatible) throw std::runtime_error("No matching SM120 EP kernel package");
    const bool usePackage = compatible && mode != "grouped";
    const bool shareWeights = usePackage && mode == "auto" && FastllmGenericEpTunedEp8(devices);
    state->hidden = h; state->topk = topk;
    std::vector<std::vector<std::vector<uint64_t>>> sources;
    for (size_t r = 0; r < devices.size(); ++r) {
        std::vector<std::vector<uint64_t>> layers;
        for (const auto &table : weights[r]) {
            std::vector<uint64_t> pointers;
            if (!table.empty()) {
                if (table.size() != size_t(2 + 2 * experts)) throw std::runtime_error("Invalid SM120 EP expert table.");
                for (int e = r * (experts / devices.size()); e < (int)(r + 1) * (experts / devices.size()); ++e) {
                    for (int part = 0; part < 2; ++part) {
                        const Data *w = table[2 + 2*e + part];
                        const int n = part ? h : 2 * mid, k = part ? mid : h;
                        if (!w || w->dataType != NVFP4 || w->dims != std::vector<int>({n, k}) ||
                            !w->cudaData || w->dataDeviceIds != std::vector<int>{devices[r]} ||
                            w->GetBytes() != (uint64_t)n * k / 32 * 17)
                            throw std::runtime_error("SM120 EP requires complete planar MXFP4/E8M0 block32 experts.");
                        pointers.push_back(reinterpret_cast<uint64_t>(w->cudaData));
                    }
                }
            }
            layers.push_back(std::move(pointers));
        }
        sources.push_back(std::move(layers));
    }
    if (usePackage) {
        state->native = FastllmCudaCreateMxfp4EpNative(devices, sources);
        if (!state->native) throw std::runtime_error("Native SM120 MXFP4 EP is unavailable in this build.");
        if (shareWeights) state->generic = FastllmCreateSharedEp8(devices, state->groupedDensePrefill);
    } else {
        state->generic = FastllmCreateGenericEp(devices, h, mid, experts, topk);
    }
    // The grouped and native kernels use the same FP4 value order on tuned EP8.
    // Build their scale layouts before releasing the original expert tensors.
    struct RestoreDevice {
        int device = FastllmCudaGetDevice();
        ~RestoreDevice() { FastllmCudaSetDevice(device); }
    } restoreDevice;
    for (size_t layer = 0; layer < weights[0].size(); ++layer) {
        if (weights[0][layer].empty()) continue;
        if (state->native) {
            state->layers.push_back(layer);
            FastllmCudaBindMxfp4EpNativeWeights(*state->native, layer);
        }
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            if (state->generic) {
                const auto packed = shareWeights ? FastllmCudaMxfp4EpPackedWeights(*state->native, rank, layer)
                                                 : std::array<void *, 2>{};
                FastllmBindGenericEp(*state->generic, rank, layer, sources[rank][layer], packed);
            }
            for (Data *weight : weights[rank][layer])
                if (weight) weight->FreeSpace();
        }
    }
    printf("MIMO_EP_MEGA ranks=%zu experts=%zu inter=%d W4A8 backend=%s\n",
           devices.size(), experts / devices.size(), mid,
           state->groupedDensePrefill ? "sm120-auto-grouped-and-native" :
           shareWeights ? "sm120-grouped-decode-native-prefill" :
           usePackage ? "flashinfer-sm120-native-cubin" : "sm120-grouped-ep");
    return state;
}

namespace {
void MimoReplayRows(FastllmCudaMimoEp8Mega &state, int rank, int layer, const void *input, const int32_t *indices,
        const float *scores, void *output, int rows, int collectiveRows, const int *liveKeys, int densePrefixRows = 0) {
    if (state.useGrouped(collectiveRows, densePrefixRows)) {
        FastllmRunGenericEp(*state.generic, rank, layer, input, indices, scores, output,
                            rows, collectiveRows, liveKeys, densePrefixRows);
        return;
    }
    auto graph = state.Graph(rank, layer, collectiveRows);
    // Every EP rank executes the same rounds, including empty local slices.
    // The artifact's largest tile is a per-round capacity, not a request limit.
    if (graph.capacity <= 0) throw std::runtime_error("Invalid native EP graph capacity.");
    for (int64_t start = 0; start < collectiveRows; start += graph.capacity) {
        int first = std::min<int64_t>(start, rows);
        int count = std::min(graph.capacity, rows - first);
        if (!FastllmCudaReplayMxfp4EpGraph(graph,
                (const uint16_t *)input + (size_t)first * 4096,
                indices + (size_t)first * 8, scores + (size_t)first * 8,
                (uint16_t *)output + (size_t)first * 4096,
                count, 4096, 8, liveKeys ? liveKeys + first : nullptr))
            throw std::runtime_error("Native SM120 EP graph replay failed.");
    }
}
void MimoGraphOutput(fastllm::Data &output, const fastllm::Data &input, int device) {
    output.dataType = fastllm::BFLOAT16;
    output.Resize({std::max(1, input.dims[0]), input.dims[1]});
    output.ToDevice(fastllm::DataDevice::CUDA, {device}, false);
    output.Allocate(false);
    output.Resize(input.dims);
}
void MimoGraphGather(FastllmCudaMimoEp8Mega &state, int rank, fastllm::Data &output) {
    using namespace fastllm;
    const int total = output.dims.at(0), width = output.dims.at(1), ranks = state.devices.size();
    if (total % ranks == 0) {
        const int count = total / ranks * width;
        if (!FastllmNcclAllGather((char *)output.cudaData + (size_t)rank * count * output.unitSize,
                output.cudaData, count, output.dataType, state.devices[rank]))
            throw std::runtime_error("SM120 EP output AllGather failed.");
    } else {
        for (int peer = 0; peer < ranks; ++peer) {
            const int first = (int64_t)total * peer / ranks;
            const int count = (int64_t)total * (peer + 1) / ranks - first;
            if (count) FastllmNcclBroadcast((char *)output.cudaData + (size_t)first * width * output.unitSize,
                    count * width, output.dataType, state.devices[peer], state.devices[rank]);
        }
    }
}
}

void FastllmCudaMimoEp8PrefillAllGather(FastllmCudaMimoEp8Mega &state, int rank, fastllm::Data &output) {
    if (state.builtin) return FastllmCudaMimoEp8BuiltinPrefillAllGather(*state.builtin, rank, output);
    MimoGraphGather(state, rank, output);
}

void FastllmCudaMimoEp8PrefillReduceScatter(FastllmCudaMimoEp8Mega &state, int rank, fastllm::Data &output) {
    if (state.builtin) return FastllmCudaMimoEp8BuiltinPrefillReduceScatter(*state.builtin, rank, output);
    if (output.Count(0) % state.devices.size()) throw std::runtime_error("Unequal SM120 EP ReduceScatter slices.");
    int count = output.Count(0) / state.devices.size();
    if (!FastllmNcclReduceScatter(output.cudaData, (char *)output.cudaData + (size_t)rank * count * output.unitSize,
            count, output.dataType, state.devices[rank])) throw std::runtime_error("SM120 EP ReduceScatter failed.");
}

bool FastllmCudaTryMimoEp8RoutedDecode(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &probabilities, const fastllm::Data &bias,
    fastllm::Data &output, const int *liveKeys, bool normalize, float scale, int capacity, bool logits) {
#ifdef FASTLLM_ENABLE_DSV41_MXFP4_SM120
    using namespace fastllm;
    if (capacity < 1 || capacity > 128 || !state.useGrouped(capacity, 0) || state.builtin || input.dataType != BFLOAT16 || !input.cudaData ||
        input.dims.empty() || input.dims.back() != 4096 || !input.Count(0) || input.Count(0) > size_t(capacity) * 4096 || !liveKeys ||
        probabilities.dataType != FLOAT32 || probabilities.Count(0) != input.Count(0) / 16 || !probabilities.cudaData ||
        bias.dataType != FLOAT32 || bias.dims != std::vector<int>{256} || !bias.cudaData ||
        probabilities.dataDeviceIds != input.dataDeviceIds || bias.dataDeviceIds != input.dataDeviceIds ||
        (uintptr_t(probabilities.cudaData) & 15)) return false;
    const int rows = input.Count(0) / 4096;
    Data row(BFLOAT16, {rows, 4096});
    row.FakeFrom(input, 0);
    MimoGraphOutput(output, row, state.devices.at(rank));
    return FastllmTryGenericEpRoutedDecode(*state.generic, rank, layer, row.cudaData,
        static_cast<const float *>(probabilities.cudaData), static_cast<const float *>(bias.cudaData),
        output.cudaData, liveKeys, normalize, scale, rows, capacity, logits);
#else
    return false;
#endif
}

void FastllmCudaRunMimoEp8MegaLocal(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, int maxRows, const int *liveKeys, fastllm::Data *prefillScratch) {
    FastllmCudaRunMimoEp8MegaLocal(state, rank, layer, input, index, score, output, maxRows, liveKeys, prefillScratch, 0);
}

void FastllmCudaRunMimoEp8MegaLocal(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, int maxRows, const int *liveKeys, fastllm::Data *prefillScratch, int densePrefixRows) {
    FastllmCudaRunMimoEp8MegaLocal(state, rank, layer, input, index, score, output, maxRows, liveKeys,
                                  prefillScratch, densePrefixRows, false);
}

void FastllmCudaRunMimoEp8MegaLocal(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, int maxRows, const int *liveKeys, fastllm::Data *prefillScratch, int densePrefixRows, bool graphVerify) {
    if (state.builtin) return FastllmCudaRunMimoEp8BuiltinMegaLocal(*state.builtin, rank, layer,
        input, index, score, output, maxRows, liveKeys, prefillScratch, graphVerify);
    if (input.dataType != fastllm::BFLOAT16 || input.dims.size() != 2 || input.dims[1] != state.hidden ||
        input.dims[0] < 0 || input.dims[0] > maxRows || maxRows <= 0 ||
        index.dims != std::vector<int>({input.dims[0], state.topk}) || score.dims != index.dims)
        throw std::runtime_error("Invalid rank-local SM120 EP input.");
    MimoGraphOutput(output, input, state.devices.at(rank));
    MimoReplayRows(state, rank, layer, input.cudaData, (int32_t *)index.cudaData,
        (float *)score.cudaData, output.cudaData, input.dims[0], maxRows, liveKeys, densePrefixRows);
}

bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, bool graphDecode, fastllm::Data *residual,
        const fastllm::Data *normWeight, fastllm::Data *normOutput, float eps,
        fastllm::Data *prefillScratch, const int *liveKeys) {
    if (state.builtin) return FastllmCudaRunMimoEp8BuiltinMega(*state.builtin, rank, layer, input,
        index, score, output, graphDecode, residual, normWeight, normOutput, eps, prefillScratch, liveKeys);
    if (input.dataType != fastllm::BFLOAT16 || input.dims.size() != 2 || input.dims[1] != state.hidden)
        throw std::runtime_error("Invalid SM120 EP input.");
    const int total = input.dims[0], ranks = state.devices.size();
    const bool sharded = total >= ranks;
    const int first = sharded ? (int64_t)total * rank / ranks : 0;
    const int rows = sharded ? (int64_t)total * (rank + 1) / ranks - first : total;
    const bool localRouting = sharded && index.dims == std::vector<int>({rows, state.topk});
    if ((!localRouting && index.dims != std::vector<int>({total, state.topk})) || score.dims != index.dims)
        throw std::runtime_error("Invalid SM120 EP routing shape.");
    MimoGraphOutput(output, input, state.devices.at(rank));
    const int commonRows = sharded ? (total + ranks - 1) / ranks : total;
    const int densePrefix = !graphDecode && !liveKeys && state.groupedDensePrefill
        ? (sharded ? total / ranks : total) : 0;
    MimoReplayRows(state, rank, layer,
        (uint16_t *)input.cudaData + (size_t)first * state.hidden,
        (int32_t *)index.cudaData + (localRouting ? 0 : first * state.topk),
        (float *)score.cudaData + (localRouting ? 0 : first * state.topk),
        (uint16_t *)output.cudaData + (size_t)first * state.hidden,
        rows, commonRows, liveKeys ? liveKeys + first : nullptr, densePrefix);
    if (!graphDecode && localRouting && residual && normWeight && normOutput) {
        // Prefill deliberately retains only the owning rank's residual rows.
        // Update those rows, then publish normalized activations for the next
        // TP projection. Gathering raw MoE output and adding the full residual
        // would read stale nonlocal rows after the first sharded layer.
        if (residual->dataType != fastllm::BFLOAT16 || residual->Count(0) != total * state.hidden ||
            normWeight->dataType != fastllm::FLOAT32 || normWeight->Count(0) != state.hidden)
            throw std::runtime_error("Invalid SM120 EP residual/RMSNorm shape.");
        normOutput->dataType = fastllm::BFLOAT16;
        normOutput->Resize(residual->dims);
        normOutput->ToDevice(output.dataDevice, output.dataDeviceIds, false);
        normOutput->Allocate(false);
        const size_t offset = (size_t)first * state.hidden * sizeof(uint16_t);
        fastllm::Data localResidual(fastllm::BFLOAT16, {rows, state.hidden}), localOutput, localNorm;
        localResidual.FakeFrom(*residual, offset);
        localOutput.FakeFrom(output, offset);
        localNorm.FakeFrom(*normOutput, offset);
        FastllmCudaNaiveAddDecodeRMSNorm(localResidual, localOutput, *normWeight, eps, localNorm);
        fastllm::Data gathered(fastllm::BFLOAT16, {total, state.hidden});
        gathered.FakeFrom(*normOutput, 0);
        MimoGraphGather(state, rank, gathered);
        return true;
    }
    if (sharded) MimoGraphGather(state, rank, output);
    return false;
}
