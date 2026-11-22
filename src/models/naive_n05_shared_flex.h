#pragma once
#include "fastllm.h"
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-naive-flex-moe.h"
#include "devices/cuda/fastllm-moe-weight-owner.h"
#include <cstdlib>
#include <cstdio>

namespace fastllm {
struct NaiveSharedFlexMoe {
    using Weights = std::vector<std::vector<std::vector<Data *>>>;
    using Run = int (*)(void *, int, int, int, const void *, const void *, const void *, void *);
    void *state = nullptr;
    bool ep = false, routeSupported = false;
    Run run = nullptr;
    void (*destroy)(void *) = nullptr;

    static bool Requested() {
        const char *value = std::getenv("FASTLLM_NAIVE_FLEX_SHARED");
        return value && std::string(value) == "1";
    }
    static bool DeviceSupported(const std::vector<int> &devices) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        return fastshared_tp_supported(devices.data(), devices.size()) &&
               fastshared_ep_supported(devices.data(), devices.size());
#else
        return false;
#endif
    }
    static bool TPWeightsSupported(const Weights &weights, const std::vector<int> &devices) {
        if (weights.size() != 8 || devices.size() != 8) return false;
        for (size_t rank = 0; rank < weights.size(); ++rank) for (const auto &ws : weights[rank]) {
            if (ws.empty()) continue;
            if (ws.size() != 514 || ws[0] || ws[1]) return false;
            for (int i = 2; i < 514; ++i) {
                const Data *w = ws[i];
                if (!w || !w->cudaData || w->isFake || w->cudaDataBorrowed ||
                    !(w->directMemory || (w->cudaRepackSource && FastllmCudaIsWeightSlabPointer(w->cudaData))) ||
                    w->dataDevice != DataDevice::CUDA || w->dataDeviceIds != std::vector<int>{devices[rank]} ||
                    !w->extraCudaData.empty() || !w->extraCudaHalfData.empty() ||
                    w->dataType != FP8_E4M3 || w->blockK != 128 || w->blockM != 128 ||
                    w->dims != (i % 2 ? std::vector<int>{4096, 256} : std::vector<int>{512, 4096}) ||
                    w->scales.size() != (i % 2 ? 64u : 128u)) return false;
            }
        }
        return true;
    }
    // EP uses MegaMoE's same packed allocation for larger row counts. TP's
    // ordered eight-row adapter covers all positive row counts itself.
    bool Supports(int rows) const { return rows > 0 && (!ep || rows <= 8); }

    NaiveSharedFlexMoe(const std::vector<int> &devices, const Weights &weights,
                       FastllmCudaMimoEp8Mega *mega) : ep(mega != nullptr) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        AssertInFastLLM(DeviceSupported(devices) && weights.size() == 8 &&
            (ep || TPWeightsSupported(weights, devices)), "Shared Flex admission was not completed.");
        auto create = ep ? fastshared_ep_create : fastshared_tp_create;
        run = ep ? fastshared_ep_run : fastshared_tp_run;
        destroy = ep ? fastshared_ep_destroy : fastshared_tp_destroy;
        AssertInFastLLM(create(devices.data(), weights[0].size(), &state) == 0,
                       "Shared Flex workspace initialization failed.");
        routeSupported = !ep && fastshared_tp_router_supported(state);
        size_t released = 0;
        int layers = 0;
        try {
            for (size_t layer = 0; layer < weights[0].size(); ++layer) {
                if (weights[0][layer].empty()) continue;
                ++layers;
                for (int rank = 0; rank < 8; ++rank) {
                    if (ep) {
                        const void *p[4]{};
                        AssertInFastLLM(FastllmCudaMimoEp8MegaWeights(*mega, rank, layer, p) &&
                            fastshared_ep_bind(state, rank, layer, p[0], p[1], p[2], p[3]) == 0,
                            "Shared Flex cannot borrow Mega packed weights.");
                    } else {
                        const auto &ws = weights[rank][layer];
                        std::vector<uint64_t> pointers;
                        std::vector<float> gs, ds;
                        for (int i = 2; i < 514; ++i) {
                            pointers.push_back(reinterpret_cast<uint64_t>(ws[i]->cudaData));
                            auto &scales = i % 2 ? ds : gs;
                            scales.insert(scales.end(), ws[i]->scales.begin(), ws[i]->scales.end());
                        }
                        AssertInFastLLM(fastshared_tp_pack(state, rank, layer,
                            pointers.data(), gs.data(), ds.data()) == 0, "Shared Flex TP packing failed.");
                        for (int i = 2; i < 514; ++i) released += ReleasePackedMoeSource(*ws[i]);
                    }
                }
            }
        } catch (...) { destroy(state); state = nullptr; throw; }
        fprintf(stderr, "[SharedFlex] TP8 EP%d: layers=%d workspace_per_gpu=%zu "
            "source_bytes_released=%zu original_payload_retained=0 packed_owner=%s\n",
            ep ? 8 : 1, layers, (ep ? fastshared_ep_workspace_bytes : fastshared_tp_workspace_bytes)(),
            released, ep ? "MegaMoE" : "FlexMoE");
#else
        AssertInFastLLM(false, "Shared Flex requires a compiled SM103 backend.");
#endif
    }
    ~NaiveSharedFlexMoe() { if (state) destroy(state); }
    bool CanRoute(int rows, const Data &router, const Data &bias) const {
        auto dense = [](const Data &x) {
            if (x.strides.empty()) return true;
            if (x.strides.size() != x.dims.size()) return false;
            int64_t stride = 1;
            for (int i = (int)x.dims.size() - 1; i >= 0; --i) {
                if (x.strides[i] != stride) return false;
                stride *= x.dims[i];
            }
            return true;
        };
        // Every positive batch is tiled with independent row offsets; the backend
        // retains its small-row and unsupported-format fallbacks.
        return routeSupported && rows > 1 && router.dataType == FLOAT32 &&
            router.dataDevice == DataDevice::CUDA && router.cudaData &&
            !router.dims.empty() && router.dims.back() == 256 && router.Count(0) == (size_t)rows * 256 &&
            (reinterpret_cast<uintptr_t>(router.cudaData) & 15) == 0 && dense(router) &&
            bias.dataType == FLOAT32 && bias.dataDevice == DataDevice::CUDA && bias.cudaData &&
            bias.Count(0) == 256 && bias.dataDeviceIds == router.dataDeviceIds && dense(bias);
    }
    bool PreparedInput(int rank, int rows, FastllmMoePreparedInput &input,
            Data &ids, Data &scores, const Data &bias, const Data &hidden,
            bool needNorm, float routeScale) const {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        // Keep small-row PDL and batches beyond the prepared tile on their existing paths.
        if (ep || rows < 4 || rows > 16 ||
            !fastshared_tp_prepared_input(state, rank, rows, &input)) return false;
        // A grouped consumer can select routes locally without a global
        // router barrier. Keep cooperative routing for the existing consumer.
        if (rows != 8 || input.preferFusedSelection) return true;
        if (bias.dataType != FLOAT32 || bias.dataDevice != DataDevice::CUDA ||
            !bias.cudaData || bias.Count(0) != 256 || bias.dataDeviceIds != hidden.dataDeviceIds)
            return true; // Preserve the original prepared-activation path.
        if (!bias.strides.empty()) {
            if (bias.strides.size() != bias.dims.size()) return true;
            size_t stride = 1;
            for (int i = (int)bias.dims.size() - 1; i >= 0; --i) {
                if (bias.dims[i] <= 0 || bias.strides[i] != stride) return true;
                stride *= bias.dims[i];
            }
        }
        for (auto *p : {&ids, &scores}) {
            p->dataType = p == &ids ? INT32 : FLOAT32;
            p->UpdateUnitSize(); p->Resize({rows, 8});
            p->ToDevice(hidden.dataDevice, hidden.dataDeviceIds, false); p->Allocate(false);
        }
        input.indices = static_cast<int *>(ids.cudaData);
        input.scores = static_cast<float *>(scores.cudaData);
        input.bias = static_cast<const float *>(bias.cudaData);
        input.needNorm = needNorm; input.routeScale = routeScale; input.routed = true;
        if (!input.Valid(rows)) input.routed = false;
        return true;
#else
        return false;
#endif
    }
    bool PrepareDense(int rank,int rows,Data &input,Data &gate,Data &up,Data &down,
                      Data &gamma,float eps,Data &silu,Data &out) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        if(ep)return false; // State belongs to the EP backend, not TP.
        bool eligible=rows>=1 && rows<=64 && fastshared_tp_dense_supported(state) &&
            gate.dims==std::vector<int>{2048,4096} && up.dims==gate.dims &&
            down.dims==std::vector<int>{4096,2048} && input.Count(0)==size_t(rows)*4096 &&
            gamma.Count(0)==4096 && std::isfinite(eps) && eps>0;
        for(const Data *p:{&input,&gate,&up,&down,&gamma}) {
            eligible=eligible && p->dataType==(p==&gamma?FLOAT32:BFLOAT16) &&
                p->dataDevice==DataDevice::CUDA && p->cudaData &&
                p->dataDeviceIds==input.dataDeviceIds && uintptr_t(p->cudaData)%16==0;
            size_t stride=1;
            if(!p->strides.empty()) {
                eligible=eligible && p->strides.size()==p->dims.size();
                for(int i=int(p->dims.size())-1;eligible && i>=0;--i) {
                    eligible=p->strides[i]==stride;stride*=p->dims[i];
                }
            }
        }
        if(eligible) for(Data *p:{&silu,&out}) {
            eligible=eligible && p!=&input && p!=&gate && p!=&up && p!=&down && p!=&gamma && &silu!=&out;
            if(!eligible)break;
            // Borrowed output buffers are not resized by this specialization.
            if(p->isFake || p->cudaDataBorrowed){eligible=false;break;}
            auto dims=input.dims;dims.back()=p==&silu?2048:4096;
            p->dataType=BFLOAT16;p->UpdateUnitSize();p->Resize(dims);
            p->ToDevice(input.dataDevice,input.dataDeviceIds,false);p->Allocate(false);
        }
        if(eligible) {
            const Data *buffers[]={&input,&gate,&up,&down,&gamma,&silu,&out};
            for(int i=0;i<7;++i) {
                const auto *p=buffers[i];const size_t bytes=p->Count(0)*p->unitSize;
                eligible=eligible && p->cudaData && uintptr_t(p->cudaData)%16==0 && uintptr_t(p->cudaData)<=UINTPTR_MAX-bytes;
                for(int j=0;j<i;++j)eligible=eligible && FastllmMoePreparedInput::Disjoint(
                    {p->cudaData,bytes},{buffers[j]->cudaData,buffers[j]->Count(0)*buffers[j]->unitSize});
            }
        }
        return fastshared_tp_dense_admit(state,rank,rows,eligible)!=0;
#else
        return false;
#endif
    }
    void Dense(int rank,int rows,Data &input,Data &gate,Data &up,Data &down,
               Data &gamma,float eps,Data &silu,Data &out) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        AssertInFastLLM(fastshared_tp_run_dense(state,rank,rows,input.cudaData,gate.cudaData,
            up.cudaData,down.cudaData,static_cast<const float*>(gamma.cudaData),eps,silu.cudaData,out.cudaData)==0,
            "Shared Flex dense launch failed.");
#endif
    }
    void ForwardRouted(int rank, int layer, int rows, Data &x, Data &router, Data &bias,
                       Data &ids, Data &scores, Data &out, bool needNorm, float routeScale, const FastllmMoePreparedInput *prepared = nullptr) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        AssertInFastLLM(CanRoute(rows, router, bias) && x.dataType == BFLOAT16 &&
            x.dataDevice == router.dataDevice && x.dataDeviceIds == router.dataDeviceIds,
            "Invalid Shared Flex routing inputs.");
        for (auto *p : {&ids, &scores, &out}) {
            p->dataType = p == &ids ? INT32 : p == &scores ? FLOAT32 : BFLOAT16;
            p->UpdateUnitSize(); p->Resize({rows, p == &out ? 4096 : 8});
            p->ToDevice(x.dataDevice, x.dataDeviceIds, false); p->Allocate(false);
        }
        if (prepared && prepared->routed) {
            AssertInFastLLM(fastshared_tp_run_preselected(state, rank, layer, rows,
                static_cast<const int *>(ids.cudaData), static_cast<const float *>(scores.cudaData),
                out.cudaData) == 0, "Shared Flex preselected launch failed.");
            return;
        }
        if (prepared) {
            AssertInFastLLM(fastshared_tp_run_prepared(state, rank, layer, rows,
                static_cast<const float *>(router.cudaData), static_cast<const float *>(bias.cudaData),
                static_cast<int *>(ids.cudaData), static_cast<float *>(scores.cudaData), out.cudaData,
                needNorm, routeScale) == 0, "Shared Flex prepared launch failed.");
            return;
        }
        AssertInFastLLM(fastshared_tp_run_router(state, rank, layer, rows, x.cudaData,
            static_cast<const float *>(router.cudaData), static_cast<const float *>(bias.cudaData),
            static_cast<int *>(ids.cudaData), static_cast<float *>(scores.cudaData), out.cudaData,
            needNorm, routeScale) == 0, "Shared Flex routed launch failed.");
#else
        AssertInFastLLM(false, "Shared Flex routing requires a compiled SM103 backend.");
#endif
    }
    void Forward(int rank, int layer, int rows, Data &x, Data &ids, Data &scores, Data &out) {
        AssertInFastLLM(Supports(rows) && x.dataType == BFLOAT16 &&
            ids.dataType == INT32 && scores.dataType == FLOAT32, "Invalid Shared Flex inputs.");
        out.dataType = BFLOAT16;
        out.Resize({rows, 4096});
        out.ToDevice(x.dataDevice, x.dataDeviceIds, false);
        out.Allocate(false);
        // A CUDA launch failure is not an unsupported-shape signal: do not
        // retry a partially executed collective against retired source data.
        AssertInFastLLM(run(state, rank, layer, rows, x.cudaData, ids.cudaData,
            scores.cudaData, out.cudaData) == 0, "Shared Flex launch failed.");
    }
};
}
