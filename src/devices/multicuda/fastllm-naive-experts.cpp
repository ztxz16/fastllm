#include "utils/utils.h"
#include "devices/multicuda/fastllm-naive-experts.h"
#include "naive-legacy-flex.h"
#include "naive-shared-flex.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/persistent_worker_group.h"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-mxfp4-marlin-tp.h"
#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/multicuda/fastllm-naive-verify.h"
#endif
#include <cstdlib>
#include <cstring>
#include <optional>
#ifdef FASTLLM_ENABLE_NAIVE_TP_MEGAMOE
#include "naive-hopper-tp-experts.h"
#endif

namespace fastllm {
namespace {
enum class FusionMode { Automatic, Disabled, Enabled };

FusionMode ReadFusionMode() {
    const char *value = std::getenv("FASTLLM_NAIVE_MOE_FUSION");
    if (value && std::strcmp(value, "0") == 0) return FusionMode::Disabled;
    if (value && std::strcmp(value, "1") == 0) return FusionMode::Enabled;
    return FusionMode::Automatic;
}
}

bool FastllmTpExperts::FusionEnabled() {
    return ReadFusionMode() != FusionMode::Disabled;
}

FastllmDenseResult FastllmRunNaiveDense(FastllmExpertHandoff *prepared, bool decode, int rank, int rows,
        Data &hidden, Data &normed, Data &gate, Data &up, Data &down, Data &gamma,
        float eps, Data &scratch, Data &output) {
    if (prepared) {
        prepared->Dense(rank, rows, hidden, gate, up, down, gamma, eps, scratch, output);
        return FastllmDenseResult::Reduced;
    }
#ifndef USE_ROCM
#ifdef FASTLLM_ENABLE_NAIVE_DENSE_SM90
    if (decode && rows > 8 && FastllmCudaNaiveVerifyDenseGateUp(normed, gate, up, scratch)) {
        Linear(scratch, down, Data(), output);
        return FastllmDenseResult::Local;
    }
#endif
    if (decode && FastllmCudaDecodeDense(normed, gate, up, down, scratch, output)) return FastllmDenseResult::Local;
#endif
    return FastllmDenseResult::Unsupported;
}

NaiveMoeBackend FastllmTpExperts::Plan(const std::vector<int> &devices, int ep, bool enabled,
        bool flexGeometry, bool attentionReplicated, bool megaAvailable, bool &packsSources) {
    const bool shared = NaiveSharedFlexMoe::Requested(), old = NaiveFlexMoe::Requested();
    const bool flex = attentionReplicated && enabled && flexGeometry && (shared ?
        NaiveSharedFlexMoe::DeviceSupported(devices) : old && NaiveFlexMoe::DeviceSupported(devices));
    const auto layout = SelectNaiveMoeBackend(ep, shared, old, megaAvailable, flex,
                                              flexGeometry || megaAvailable, true, enabled);
    packsSources = layout == NaiveMoeBackend::FlexTP || layout == NaiveMoeBackend::LegacyFlexTP ||
        (NaiveMoeUsesEP(layout) && FastllmCudaMimoEp8MegaOwnsPackedWeights(devices.front()));
    return layout;
}

std::shared_ptr<FastllmTpExperts> FastllmTpExperts::Prepare(NaiveMoeBackend &layout, bool &packsSources,
        const std::vector<int> &devices, const Weights &weights, int hidden, int experts, int topk,
        bool allowPackedTp, int maxBatch, bool fixedRouting, bool preferVerifyPages, PersistentWorkerGroup &workers, bool &packing,
        std::shared_ptr<FastllmCudaMimoEp8Mega> &mega) {
    if ((layout == NaiveMoeBackend::FlexTP || layout == NaiveMoeBackend::LegacyFlexTP) &&
        !NaiveSharedFlexMoe::TPWeightsSupported(weights, devices, layout == NaiveMoeBackend::FlexTP)) {
        layout = NaiveMoeBackend::LegacyTP;
        packsSources = false;
        fprintf(stderr, "[Naive MoE fallback] unsupported TP shards; keep original weights and MoE.\n");
    }
    auto state = std::make_shared<FastllmTpExperts>();
    if (NaiveMoeUsesEP(layout)) mega = FastllmCudaCreateMimoEp8Mega(devices, weights, topk);
    if (layout == NaiveMoeBackend::FlexEP && !NaiveSharedFlexMoe::CanBorrowEpWeights(*mega, weights)) {
        layout = NaiveMoeBackend::MegaEP;
        fprintf(stderr, "[Naive MoE fallback] Flex cannot use the packed EP layout; retaining MegaMoE.\n");
    }
    if (layout == NaiveMoeBackend::FlexTP || layout == NaiveMoeBackend::FlexEP) {
        state->shared = std::make_shared<NaiveSharedFlexMoe>(devices, weights, mega.get(), preferVerifyPages);
    } else if (layout == NaiveMoeBackend::LegacyFlexTP) {
        state->legacy = std::make_shared<NaiveFlexMoe>(devices, weights);
#ifndef USE_ROCM
    } else if (layout == NaiveMoeBackend::LegacyTP && allowPackedTp) {
        const FusionMode fusion = ReadFusionMode();
        if (fusion == FusionMode::Disabled) return nullptr;
        struct RestoreDevice {
            int device = FastllmCudaGetDevice();
            ~RestoreDevice() { FastllmCudaSetDevice(device); }
        } restore;
#ifdef FASTLLM_ENABLE_NAIVE_TP_MEGAMOE
        // Restaging expert weights trades long-prefill throughput for faster verification.
        const bool twoTiles = preferVerifyPages && fusion == FusionMode::Enabled;
        const int requests = (kFastllmMxfp4LocalRoutedMaxRows / 8) * (twoTiles ? 2 : 1);
        if (fixedRouting && hidden == 4096 && experts == 256 && topk == 8 && maxBatch > 0 && maxBatch <= requests &&
            (state->hopper = HopperTpExperts::Prepare(devices, weights, workers, packing))) {
            packsSources = true;
            return state;
        }
#endif
        for (size_t r = 0; r < devices.size(); ++r) {
            FastllmCudaSetDevice(devices[r]);
            auto rank = PrepareMxfp4MarlinRank(devices[r], weights[r], hidden, experts, topk);
            if (!rank) return nullptr;
            state->ranks.push_back(std::move(rank));
        }
        state->wideMarlin = devices.size() == 8 && FastllmCudaNaiveHopperGroup(devices);
        // No source may be retired until every rank has admitted its shards.
        packing = true;
        std::vector<std::exception_ptr> errors(devices.size());
        workers.Run(devices, [&](int r) {
            FastllmCudaSetDevice(devices[r]);
            PackMxfp4MarlinRank(*state->ranks[r], weights[r]);
        }, errors);
        for (auto error : errors) if (error) std::rethrow_exception(error);
#endif
    } else return nullptr;
    return state;
}

const char *FastllmTpExperts::Name() const {
    if (hopper) return "mega-tp/packed-stage";
    if (shared) return NaiveMoeBackendName(shared->ep ? NaiveMoeBackend::FlexEP : NaiveMoeBackend::FlexTP);
    return legacy ? NaiveMoeBackendName(NaiveMoeBackend::LegacyFlexTP) : "marlin-tp/mxfp4-w4a16";
}
void FastllmTpExperts::Validate(bool bf16, bool fixedRouting) const {
    AssertInFastLLM(bf16 && (!(shared || legacy || hopper) || fixedRouting),
                    "Prepared expert configuration changed; reload the model.");
}
const FastllmTpExperts *FastllmTpExperts::ForRows(int rows, bool replicated) const {
    return shared && (!replicated || !shared->Supports(rows)) ? nullptr : this;
}
FastllmExpertHandoff *FastllmTpExperts::Handoff() const { return shared.get(); }
bool FastllmTpExperts::PackedVerify(int rows, int count) const {
#ifndef USE_ROCM
    return !ranks.empty() && FastllmCudaPackedVerifySupported(rows, count);
#else
    return false;
#endif
}

bool FastllmExpertRouter::CanRouteInExpert(int rows) const {
    if (experts && experts->hopper)
        return rows > 1 && rows < FastllmCudaGetLinearExactBatchThreshold() && topk == 8 && normalize && scale == 1.f &&
               probabilities.dataType == FLOAT32 && !probabilities.dims.empty() && probabilities.dims.back() == 256 &&
               bias.dataType == FLOAT32 && bias.Count(0) == 256 && bias.cudaData;
    return topk == 8 && experts && experts->shared && experts->shared->CanRoute(rows, probabilities, bias);
}

bool FastllmExpertRouter::Project(Data &input, Data &weight, Data *verifyScratch,
        bool probabilitiesReady, const Data *residual, const Data *gamma, float eps, int firstShardRow) {
    if (probabilitiesReady && firstShardRow >= 0) {
        // TP attention produces all router rows. Each EP source dispatches just its own rows.
        const int rows = input.dims.at(0);
        const int width = probabilities.dims.back();
        AssertInFastLLM(width > 0 && (size_t(firstShardRow) + rows) * width <= probabilities.Count(0),
                        "EP router shard exceeds the fused attention output.");
        Data local(probabilities.dataType, {rows, width});
        local.FakeFrom(probabilities, size_t(firstShardRow) * width * probabilities.unitSize);
        SelectExpert(local, ids, scores, topk, normalize, scale, &bias);
        return true;
    }
    if (experts && experts->legacy) {
        if (!probabilitiesReady && !FastllmCudaNaiveRouterSigmoid(input, weight, probabilities)) {
            ToDataType(input, projectionScratch, FLOAT32);
            Linear(projectionScratch, weight, Data(), probabilities);
            Sigmoid(probabilities, probabilities);
        }
        return true;
    }
    if (!gamma && (!experts || (!experts->shared && !experts->hopper)) && verifyScratch &&
        FastllmCudaNaiveVerifyRouter(input, weight, bias, *verifyScratch, ids, scores, topk, normalize, scale))
        return true;
#ifndef USE_ROCM
    if (!probabilitiesReady && packedVerify && experts && !experts->ranks.empty()) {
        Mxfp4MarlinRouter router{experts->ranks.at(rank).get(), ids, scores, bias, topk, normalize, scale};
        preparedRoutes = router.Project(input, weight, residual, gamma, eps);
    }
#endif
    return preparedRoutes;
}
bool FastllmExpertRouter::Select(const Data &probabilities) {
#ifndef USE_ROCM
    if (packedVerify && experts && !experts->ranks.empty()) {
        Mxfp4MarlinRouter router{experts->ranks.at(rank).get(), ids, scores, bias, topk, normalize, scale};
        preparedRoutes = router.Select(probabilities);
    }
#endif
    return preparedRoutes;
}

FastllmExpertRows FastllmPrepareExpertRows(const Data &input, Data &gate, Data &activation, Data &output,
        std::vector<Data *> &weights, const Data &ids, const Data &scores, int rows, int topk, bool allowGrouped) {
    if (weights.size() <= 2 || !weights[2] || weights[2]->dataType != NVFP4_BLOCK_16_E4M3_PACKED ||
        !FastllmCudaNVFP4E4M3GroupedMoeSupported(input.dataDeviceIds.at(0)) ||
        !FastllmCudaPrepareNVFP4E4M3Moe(weights.data(), weights.size())) return FastllmExpertRows::Owning;
    if (allowGrouped && FastllmCudaMergeMOENVFP4E4M3MarlinRows(input, gate, activation, output, weights.data(),
        weights.size(), static_cast<const int32_t *>(ids.cudaData), static_cast<const float *>(scores.cudaData),
        rows, topk)) return FastllmExpertRows::Complete;
    return FastllmExpertRows::Resident;
}

FastllmExpertResult FastllmTpExperts::Forward(int layer, int rows, Data &input, Data &hidden,
        Data &scratch, Data &output, const FastllmExpertRouter &router, bool routeInExpert, bool verifyReduce,
        const FastllmMoePreparedInput *prepared, const FastllmExpertNorm *norm) const {
    FastllmExpertResult result;
#ifdef FASTLLM_ENABLE_NAIVE_TP_MEGAMOE
    if (hopper) {
        if (verifyReduce || rows == 1 || rows < FastllmCudaGetLinearExactBatchThreshold()) {
            hopper->Run(router.rank, layer, input, router.ids, routeInExpert ? router.probabilities : router.scores,
                        output, routeInExpert ? &router.bias : nullptr);
        } else {
            auto &local = hopper->ranks.at(router.rank);
            MergeMOE(input, router.ids, router.scores, hopper->Stage(router.rank, layer), local.biases,
                     scratch, local.w2, local.w3, local.tempInput, local.tempOutput, 0.f, output, layer);
        }
        return result;
    }
#endif
    if (legacy) {
        legacy->Forward(router.rank, layer, rows, input, router.probabilities, router.bias, hidden);
        result.residualUpdated = true;
        return result;
    }
    if (shared) {
        if (routeInExpert) shared->ForwardRouted(router.rank, layer, rows, input, router.probabilities, router.bias,
            router.ids, router.scores, output, router.normalize, router.scale, prepared);
        else shared->Forward(router.rank, layer, rows, input, router.ids, router.scores, output);
        result.reduced = true;
        return result;
    }
#ifndef USE_ROCM
    std::optional<Mxfp4MarlinNorm> fused;
    if (norm && norm->weight.dataType == FLOAT32 && norm->weight.Count(0) == 4096) {
        fused.emplace(Mxfp4MarlinNorm{norm->hidden, norm->weight, norm->output, norm->eps, norm->next});
        fused->BindOutputs(input, norm->feature, norm->featureIndex, norm->featureCount);
    }
    AssertInFastLLM(RunMxfp4MarlinTP(*ranks.at(router.rank), layer, input, router.ids, router.scores,
        scratch, output, verifyReduce ? &result.reduced : nullptr, fused ? &*fused : nullptr, router.preparedRoutes),
        "Prepared MXFP4 Marlin TP execution failed.");
    result.trackReduction = result.reduced;
    result.normalized = fused && fused->ready;
    result.featureWritten = result.normalized && fused->feature;
#endif
    return result;
}
}
