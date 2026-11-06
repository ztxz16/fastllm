#include "blocks/attentionexecutor.h"
#include "utils.h"

namespace fastllm {
    namespace {
        /** 返回模型KV Cache布局的配置名称。 */
        const char *LayoutName(ModelKVCacheLayout layout) {
            return layout == ModelKVCacheLayout::PAGED ? "paged" : "continuous";
        }
    }

    ModelAttentionExecutionPlan KVCacheController::Resolve(
            const ModelAttentionCapabilities &capabilities,
            int batch,
            bool isPrefill,
            bool usesAlibi,
            bool kvCacheInCPU) {
        AssertInFastLLM(batch > 0,
                        "Attention execution requires a positive batch.\n");

        const std::string requested = GetKVCacheLayout();
        ModelKVCacheLayout layout = capabilities.defaultLayout;
        if (requested == "continuous") {
            layout = ModelKVCacheLayout::CONTINUOUS;
        } else if (requested == "paged") {
            layout = ModelKVCacheLayout::PAGED;
        } else {
            AssertInFastLLM(requested == "auto",
                            "Unknown KV Cache layout '" + requested + "'.\n");
        }

        if (layout == ModelKVCacheLayout::CONTINUOUS) {
            AssertInFastLLM(capabilities.supportsContinuous,
                            capabilities.modelName +
                            " doesn't support continuous KV cache.\n");
        } else {
            AssertInFastLLM(capabilities.supportsPaged,
                            capabilities.modelName +
                            " doesn't support paged KV cache.\n");
            AssertInFastLLM(!usesAlibi || capabilities.supportsPagedAlibi,
                            capabilities.modelName +
                            " paged KV cache doesn't support ALiBi attention.\n");
            AssertInFastLLM(!kvCacheInCPU || capabilities.supportsPagedCpuCache,
                            capabilities.modelName +
                            " paged KV cache doesn't support CPU KV cache.\n");
        }

        TraceKVCacheLayout(capabilities.modelName, LayoutName(layout));
        return {layout, isPrefill, batch};
    }

    void KVCacheController::BindLayerCaches(
            std::vector<std::pair<Data*, Data*>> &pastKeyValues,
            int batch,
            int blockCount,
            int layerIndex,
            std::vector<Data*> &keys,
            std::vector<Data*> &values) {
        AssertInFastLLM(batch > 0 && blockCount > 0,
                        "KV Cache binding requires positive batch and block count.\n");
        AssertInFastLLM(layerIndex >= 0 && layerIndex < blockCount,
                        "KV Cache layer index is out of range.\n");
        AssertInFastLLM((int)pastKeyValues.size() >= batch * blockCount,
                        "KV Cache descriptors should use batch * block_count layout.\n");

        keys.resize(batch);
        values.resize(batch);
        for (int requestIndex = 0; requestIndex < batch; requestIndex++) {
            const int cacheIndex = requestIndex * blockCount + layerIndex;
            Data *key = pastKeyValues[cacheIndex].first;
            Data *value = pastKeyValues[cacheIndex].second;
            AssertInFastLLM(key != nullptr && value != nullptr,
                            "KV Cache descriptor contains a null K/V pointer.\n");
            keys[requestIndex] = key;
            values[requestIndex] = value;
        }
    }

    AttentionExecutor AttentionExecutor::Create(
            const ModelAttentionCapabilities &capabilities,
            int batch,
            bool isPrefill,
            bool usesAlibi,
            bool kvCacheInCPU) {
        return AttentionExecutor(KVCacheController::Resolve(
            capabilities, batch, isPrefill, usesAlibi, kvCacheInCPU));
    }

    AttentionExecutor::AttentionExecutor(
            const ModelAttentionExecutionPlan &plan) : plan(plan) {}

    bool AttentionExecutor::UsesPagedKVCache() const {
        return plan.cacheLayout == ModelKVCacheLayout::PAGED;
    }

    bool AttentionExecutor::IsPrefill() const {
        return plan.isPrefill;
    }

    int AttentionExecutor::Batch() const {
        return plan.batch;
    }

    void AttentionExecutor::BindLayerCaches(
            std::vector<std::pair<Data*, Data*>> &pastKeyValues,
            int blockCount,
            int layerIndex,
            std::vector<Data*> &keys,
            std::vector<Data*> &values) const {
        KVCacheController::BindLayerCaches(
            pastKeyValues, plan.batch, blockCount, layerIndex, keys, values);
    }
}
