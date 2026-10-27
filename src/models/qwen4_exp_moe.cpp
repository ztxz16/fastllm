// Expert execution plans are separate from the model-wide forward policy.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_tp.h"
#include "utils.h"
#include "executor.h"
namespace fastllm {
    void Qwen4ExpModel::PrepareMoEExecutionPlans() {
        auto prepare = [&](int layer, const std::string &prefix, std::vector<Data *> &weights,
                           std::vector<Data *> &bias) {
            auto device =
                SelectSpecialWeightDevice(prefix + "experts.0.gateup_proj.weight", layer);
            if (device.empty()) device = ((Executor *)GetExecutor())->GetFirstDeviceType();
            const bool gpu = ExecutionDeviceIsGpu(device);
            return MoEExecutionPlan{gpu ? ExecutionMode::Gpu : ExecutionMode::Hybrid, layer, prefix,
                                    &weights, &bias};
        };
        targetMoePlans.clear();
        targetMoePlans.reserve(block_cnt);
        for (int layer = 0; layer < block_cnt; ++layer) {
            targetMoePlans.push_back(
                prepare(layer, languagePrefix + "layers." + std::to_string(layer) + ".mlp.",
                        weights[layer], biass[layer]));
        }
        draftMoePlan = prepare(block_cnt - 1, "mtp.layers.0.mlp.", mtpMoeWeights, mtpMoeBiass);
    }

    void Qwen4ExpModel::RunMoE(int layer, const Data &input, Data &output, bool reduceOutput,
                               PrefillStageContext *prefill, MoEPrepared *prepared,
                               const std::function<void()> &sharedGraph) {
        RunMoE(targetMoePlans[layer], input, output, reduceOutput, prefill, prepared, sharedGraph);
    }

    void Qwen4ExpModel::RunMoE(const MoEExecutionPlan &plan, const Data &input, Data &output,
                               bool reduceOutput, PrefillStageContext *prefill,
                               MoEPrepared *prepared, const std::function<void()> &sharedGraph) {
        if (plan.backend == ExecutionMode::Gpu) {
            RunGpuMoE(plan, input, output, reduceOutput, prefill, prepared, sharedGraph);
        } else {
            RunHybridMoE(plan, input, output, reduceOutput, prefill, prepared, sharedGraph);
        }
    }
    void Qwen4ExpModel::SelectMoEExperts(Data &routerLogits, Data &expertIndex, Data &expertScore) {
#ifdef USE_CUDA
        if (routerLogits.dataDevice == DataDevice::CUDA &&
            routerLogits.dataType == DataType::FLOAT32 && !routerLogits.dims.empty() &&
            routerLogits.dims.back() == 512 && this->num_experts_per_tok == 10) {
            FusedSoftmaxSelectExpert(routerLogits, expertIndex, expertScore,
                                     this->num_experts_per_tok, this->norm_topk_prob,
                                     this->routed_scaling_factor, nullptr);
            return;
        }
#endif
        Softmax(routerLogits, routerLogits, -1);
        SelectExpert(routerLogits, expertIndex, expertScore, this->num_experts_per_tok,
                     this->norm_topk_prob, this->routed_scaling_factor, nullptr);
    }
    void Qwen4ExpModel::RunSharedExpert(const std::string &mlp, Data &flattened, Data &sharedGateUp,
                                        Data &sharedHidden, Data &sharedGate, Data &sharedResult) {
#ifdef USE_CUDA
        if (FastllmCudaQwen4SharedExpert(flattened,
                                         this->weight[mlp + "shared_expert.gateup_proj.weight"],
                                         this->weight[mlp + "shared_expert.down_proj.weight"],
                                         this->weight[mlp + "shared_expert_gate.weight"],
                                         sharedGateUp, sharedHidden, sharedGate, sharedResult)) {
            return;
        }
#endif
        Linear(flattened, this->weight[mlp + "shared_expert.gateup_proj.weight"], Data(),
               sharedGateUp);
        Swiglu(sharedGateUp, sharedHidden);
        Linear(sharedHidden, this->weight[mlp + "shared_expert.down_proj.weight"], Data(),
               sharedResult);
        Linear(flattened, this->weight[mlp + "shared_expert_gate.weight"], Data(), sharedGate);
        SigmoidMulTo(sharedResult, sharedGate);
    }
    void Qwen4ExpModel::PrepareMoERouting(int layer, const Data &input, MoEPrepared &prepared) {
        const std::string mlp = languagePrefix + "layers." + std::to_string(layer) + ".mlp.";
        Data flattened;
        flattened.FakeFrom(input, 0);
        flattened.Resize(input.dims);
        flattened.Reshape({input.dims[0] * input.dims[1], input.dims.back()});
        const bool hostMoe = threadTpRank >= 0 && threadTpOwner->hybrid.hostMoeLayers[layer];
        if (!hostMoe || threadTpRank == 0) {
            Data routerLogits;
            Linear(flattened, this->weight[mlp + "gate.weight"], Data(), routerLogits);
            ToDataType(routerLogits, DataType::FLOAT32);
            SelectMoEExperts(routerLogits, prepared.index, prepared.score);
        }
    }
} // namespace fastllm
