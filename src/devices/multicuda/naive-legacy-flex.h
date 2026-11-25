#pragma once
#include "fastllm.h"
#include <cstdlib>
#include <cstdio>
#include "devices/cuda/fastllm-naive-flex-moe.h"
#include "devices/cuda/fastllm-moe-weight-owner.h"

namespace fastllm {
// Legacy Flex arithmetic remains available. Its packed weights become the
// sole payload; all row counts use the existing eight-row adapter.
struct NaiveFlexMoe {
    void* state = nullptr;
    using Create = int(*)(const int*, int, void**);
    using Pack = int(*)(void*, int, int, const uint64_t*, const float*, const float*);
    using Run = int(*)(void*, int, int, int, const void*, const void*, const void*, void*);
    using Destroy = void(*)(void*);
    Run run = nullptr;
    Destroy destroy = nullptr;
    static bool Requested() {
        const char* value = std::getenv("FASTLLM_NAIVE_FLEX_MOE");
        return value && std::string(value) == "1";
    }
    static bool DeviceSupported(const std::vector<int> &devices) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        return fastflex_supported(devices.data(), devices.size());
#else
        return false;
#endif
    }
    NaiveFlexMoe(const std::vector<int>& devices,
                const std::vector<std::vector<std::vector<Data*>>>& weights) {
        AssertInFastLLM(devices.size() == 8 && weights.size() == 8,
                       "Experimental FlexMoE requires TP8 EP1.");
        for (int rank = 0; rank < 8; ++rank) for (auto& ws : weights[rank]) {
            if (ws.empty()) continue;
            AssertInFastLLM(ws.size() == 514 && !ws[0] && !ws[1], "FlexMoE requires 256 routed experts.");
            for (int i = 2; i < 514; ++i) {
                Data* w = ws[i];
                AssertInFastLLM(w && w->cudaData && w->dataType == FP8_E4M3 &&
                    w->blockK == 128 && w->blockM == 128 &&
                    w->dims == (i%2 == 0 ? std::vector<int>{512,4096} : std::vector<int>{4096,256}) &&
                    w->scales.size() == (i%2 == 0 ? 128u : 64u),
                    "FlexMoE requires original block128 FP8 TP8 shards (H4096, I256).");
            }
        }
        Create create = nullptr;
        Pack pack = nullptr;
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
        create = fastflex_create;
        pack = fastflex_pack;
        run = fastflex_run;
        destroy = fastflex_destroy;
#else
        AssertInFastLLM(false, "FlexMoE requires a build containing the native SM103 backend.");
#endif
        AssertInFastLLM(create(devices.data(), weights[0].size(), &state) == 0, "FlexMoE initialization failed.");
        try {
        for (size_t layer = 0; layer < weights[0].size(); ++layer) {
            if (weights[0][layer].empty()) continue;
            for (int rank = 0; rank < 8; ++rank) {
                auto& ws = weights[rank][layer];
                std::vector<uint64_t> pointers; std::vector<float> gs, ds;
                for (int i = 2; i < 514; ++i) {
                    pointers.push_back((uint64_t)ws[i]->cudaData);
                    auto& scales = i%2 == 0 ? gs : ds;
                    scales.insert(scales.end(), ws[i]->scales.begin(), ws[i]->scales.end());
                }
                AssertInFastLLM(pack(state,rank,layer,pointers.data(),gs.data(),ds.data()) == 0,
                               "FlexMoE weight packing failed.");
                for (int i = 2; i < 514; ++i) ReleasePackedMoeSource(*ws[i]);
            }
            if (layer%8 == 7) fprintf(stderr,"[FlexMoE] packed layer %zu / %zu\n",layer+1,weights[0].size());
        }
        } catch (...) { destroy(state); state = nullptr; throw; }
        fprintf(stderr,"[FlexMoE] enabled: SM103, TP8 EP1, W8A8, rows tiled by 8, original_payload_retained=0, Graph compatible, backend=builtin\n");
    }
    ~NaiveFlexMoe() { if (state) destroy(state);  }
    void Forward(int rank,int layer,int rows,Data& x,Data& probabilities,const Data& bias,Data& hidden) {
        AssertInFastLLM(x.dataType == BFLOAT16 && hidden.dataType == BFLOAT16 &&
                       probabilities.dataType == FLOAT32 && bias.dataType == FLOAT32,
                       "FlexMoE input type mismatch.");
        if (rows <= 8) {
            AssertInFastLLM(run(state,rank,layer,rows,x.cudaData,probabilities.cudaData,bias.cudaData,hidden.cudaData)==0,
                           "FlexMoE kernel launch failed.");
            return;
        }
        // Attention and the FP32 router retain their original full-row shape.
        // Only MoE is tiled; rank-local streams serialize workspace/mailbox reuse.
        // All ranks visit the same tiles, including the final short tile.
        for (int first = 0; first < rows; first += 8) {
            const int count = std::min(8, rows - first);
            const size_t hiddenBytes = (size_t)first * 4096 * sizeof(uint16_t);
            const size_t probabilityBytes = (size_t)first * 256 * sizeof(float);
            AssertInFastLLM(run(state,rank,layer,count,
                (const char*)x.cudaData + hiddenBytes,
                (const char*)probabilities.cudaData + probabilityBytes,bias.cudaData,
                (char*)hidden.cudaData + hiddenBytes)==0, "FlexMoE prefill tile launch failed.");
        }
    }
};
}
