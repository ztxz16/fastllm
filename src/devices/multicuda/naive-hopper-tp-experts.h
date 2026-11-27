#pragma once
#include "devices/cuda/fastllm-mxfp4-ep.h"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "utils/persistent_worker_group.h"
#include <array>
#include <set>

namespace fastllm {
struct HopperTpExperts {
    struct Rank {
        Data workspace{INT8}, staging{INT8};
        std::vector<std::shared_ptr<void>> layers;
        std::vector<Data> views;
        std::vector<Data *> prefillWeights, biases;
        Data w2, w3, tempInput, tempOutput;
    };
    std::vector<Rank> ranks;
    static std::shared_ptr<HopperTpExperts> Prepare(const std::vector<int> &devices,
        const FastllmTpExperts::Weights &shards, PersistentWorkerGroup &workers, bool &packing);
    std::vector<Data *> &Stage(int rank, int layer);
    void Run(int rank, int layer, const Data &input, const Data &index,
             const Data &score, Data &output, const Data *routerBias);
};

inline std::shared_ptr<HopperTpExperts> HopperTpExperts::Prepare(const std::vector<int> &devices,
        const FastllmTpExperts::Weights &shards, PersistentWorkerGroup &workers, bool &packing) {
    constexpr int experts = 256, hidden = 4096, intermediate = 256;
    if (devices.size() != 8 || shards.size() != devices.size() || shards.front().empty()) return {};
    const size_t layers = shards.front().size();
    for (const auto &rank : shards) if (rank.size() != layers) return {};
    size_t workspaceBytes = 0;
    for (int device : devices) {
        workspaceBytes = fastllm_mxfp4_tp8_workspace_bytes(device);
        if (!workspaceBytes) return {};
    }

    struct Source {
        std::array<const void *, experts> gate, down;
        std::array<int, 2 * experts> shifts;
    };
    std::vector<std::vector<Source>> sources(devices.size(), std::vector<Source>(layers));
    std::vector<unsigned char> supported(devices.size());
    std::vector<std::exception_ptr> errors(devices.size());
    workers.Run(devices, [&](int rank) {
        const int device = devices[rank];
        FastllmCudaSetDevice(device);
        std::set<const void *> owners;
        for (size_t layer = 0; layer < layers; ++layer) if (!shards[rank][layer].empty()) {
            const auto &weights = shards[rank][layer];
            if (weights.size() != 2 + 2 * experts || weights[0] || weights[1]) return;
            auto &source = sources[rank][layer];
            for (int e = 0; e < experts; ++e) for (int leg = 0; leg < 2; ++leg) {
                const Data *w = weights[2 + 2 * e + leg];
                if (!w || !FastllmCudaMimoIsMxfp4Expert(*w) || !w->cudaData || w->isFake ||
                    w->cudaDataBorrowed || !w->extraCudaData.empty() || !w->extraCudaHalfData.empty() ||
                    w->dims[0] != (leg ? hidden : 2 * intermediate) || w->dims[1] != (leg ? intermediate : hidden) ||
                    w->dataDevice != DataDevice::CUDA || w->dataDeviceIds != std::vector<int>{device} ||
                    !owners.insert(w->cudaData).second) return;
                (leg ? source.down : source.gate)[e] = w->cudaData;
            }
            int valid = fastllm_mxfp4_tp8_validate(source.gate.data(), source.down.data(), source.shifts.data());
            if (valid < 0) throw std::runtime_error(fastllm_mxfp4_ep_error());
            if (!valid) return;
        }
        supported[rank] = true;
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    if (std::find(supported.begin(), supported.end(), 0) != supported.end()) return {};

    // Every source is valid and exclusively owned. Pack and retire one layer
    // at a time. After the first transfer a failed preparation requires reload.
    packing = true;
    auto next = std::make_shared<HopperTpExperts>();
    next->ranks.resize(devices.size());
    workers.Run(devices, [&](int rank) {
        const int device = devices[rank];
        FastllmCudaSetDevice(device);
        auto &local = next->ranks[rank];
        auto allocate = [&](Data &buffer, size_t bytes) {
            buffer.Resize({static_cast<int>(bytes)});
            buffer.ToDevice(DataDevice::CUDA, {device}, false);
            buffer.Allocate(false);
        };
        allocate(local.workspace, workspaceBytes);
        FastllmCudaMemset0(local.workspace.cudaData, workspaceBytes);
        const size_t gateBytes = GetNVFP4StorageBytes(2 * intermediate, hidden, 1, 32);
        const size_t downBytes = GetNVFP4StorageBytes(hidden, intermediate, 1, 32);
        allocate(local.staging, experts * (gateBytes + downBytes));
        local.views.resize(2 * experts);
        local.prefillWeights.resize(2 + 2 * experts, nullptr);
        local.biases.resize(local.prefillWeights.size(), nullptr);
        for (int e = 0; e < experts; ++e) for (int leg = 0; leg < 2; ++leg) {
            Data &view = local.views[2 * e + leg];
            view.FakeFrom(local.staging, leg ? experts * gateBytes + e * downBytes : e * gateBytes);
            view.dataType = NVFP4;
            view.blockK = 1;
            view.blockM = 32;
            view.UpdateUnitSize();
            view.Resize({leg ? hidden : 2 * intermediate, leg ? intermediate : hidden});
            local.prefillWeights[2 + 2 * e + leg] = &view;
        }

        local.layers.resize(layers);
        for (size_t layer = 0; layer < layers; ++layer) if (!shards[rank][layer].empty()) {
            const auto &source = sources[rank][layer];
            void *handle = fastllm_mxfp4_tp8_create(source.gate.data(), source.down.data(),
                source.shifts.data(), local.workspace.cudaData, workspaceBytes);
            if (!handle) throw std::runtime_error(fastllm_mxfp4_ep_error());
            local.layers[layer] = std::shared_ptr<void>(handle, fastllm_mxfp4_ep_destroy);
            const auto &weights = shards[rank][layer];
            for (size_t i = 2; i < weights.size(); ++i) {
                Data &w = *weights[i];
                if (w.directMemory) FastllmCudaDirectFree(w.cudaData);
                else FastllmCudaForceFree(w.cudaData);
                w.cudaData = nullptr;
                w.FreeSpace();
            }
        }
    }, errors);
    for (auto error : errors) if (error) std::rethrow_exception(error);
    return next;
}

inline std::vector<Data *> &HopperTpExperts::Stage(int rank, int layer) {
    auto &local = ranks[rank];
    if (!fastllm_mxfp4_tp8_restore(local.layers[layer].get(), local.staging.cudaData))
        throw std::runtime_error(fastllm_mxfp4_ep_error());
    return local.prefillWeights;
}

inline void HopperTpExperts::Run(int rank, int layer, const Data &input, const Data &index,
                                        const Data &score, Data &output, const Data *routerBias) {
    const int rows = input.dims[0];
    AssertInFastLLM(rows > 0 && input.dataType == BFLOAT16 && input.dims == std::vector<int>({rows, 4096}) &&
        (routerBias ? (routerBias->dataType == FLOAT32 && routerBias->Count(0) == 256 && routerBias->cudaData)
                    : (index.dataType == INT32 && index.Count(0) == rows * 8 && index.cudaData)) &&
        score.dataType == FLOAT32 && score.Count(0) == rows * (routerBias ? 256 : 8) && score.cudaData,
        "Unexpected TP-local MegaMoE decode inputs.");
    output.dataType = BFLOAT16;
    output.Resize({rows, 4096});
    output.ToDevice(DataDevice::CUDA, input.dataDeviceIds, false);
    output.Allocate(false);
    const int tileRows = routerBias ? kFastllmMxfp4LocalRoutedMaxRows : kFastllmMxfp4LocalMaxRows;
    for (int first = 0; first < rows; first += tileRows) {
        const int count = std::min(tileRows, rows - first);
        void *handle = ranks[rank].layers[layer].get();
        const auto *x = static_cast<const uint16_t *>(input.cudaData) + (size_t)first * 4096;
        auto *y = static_cast<uint16_t *>(output.cudaData) + (size_t)first * 4096;
        const auto *p = static_cast<const float *>(score.cudaData);
        const bool ok = routerBias
            ? fastllm_mxfp4_tp8_run_routed(handle, x, y, p + first * 256,
                                          static_cast<const float *>(routerBias->cudaData), count)
            : fastllm_mxfp4_ep_run(handle, x, y, static_cast<const int32_t *>(index.cudaData) + first * 8,
                                  p + first * 8, count, count, nullptr);
        if (!ok)
            throw std::runtime_error(std::string("TP-local MegaMoE forward: ") + fastllm_mxfp4_ep_error());
    }
}
}  // namespace fastllm
