#ifndef FASTLLM_ATTENTION_EXECUTOR_H
#define FASTLLM_ATTENTION_EXECUTOR_H

#include "fastllm.h"

#include <string>
#include <utility>
#include <vector>

namespace fastllm {

    enum class ModelKVCacheLayout {
        CONTINUOUS,
        PAGED,
    };

    /** 描述模型前向路径支持的KV Cache能力。 */
    struct ModelAttentionCapabilities {
        std::string modelName;
        ModelKVCacheLayout defaultLayout = ModelKVCacheLayout::CONTINUOUS;
        bool supportsContinuous = true;
        bool supportsPaged = false;
        bool supportsPagedAlibi = false;
        bool supportsPagedCpuCache = false;
    };

    /** 保存一次模型Attention前向已经确定的静态执行计划。 */
    struct ModelAttentionExecutionPlan {
        ModelKVCacheLayout cacheLayout = ModelKVCacheLayout::CONTINUOUS;
        bool isPrefill = false;
        int batch = 0;
    };

    /**
     * 统一解析模型请求的KV Cache布局并检查模型能力。
     *
     * auto使用模型声明的默认布局；显式布局不允许静默回退。Paged布局还会
     * 检查ALiBi和CPU KV Cache能力。成功后记录实际布局，供正式测试确认。
     */
    class KVCacheController {
    public:
        /**
         * 生成一次模型Attention前向的执行计划。
         *
         * @param capabilities 模型声明的Continuous/Paged能力。
         * @param batch 本次合批的请求数量。
         * @param isPrefill true表示至少一个请求包含多个新Token。
         * @param usesAlibi true表示模型使用ALiBi位置偏置。
         * @param kvCacheInCPU true表示KV Cache固定在CPU。
         * @return 已完成能力检查的布局和阶段计划；不兼容时直接报错。
         */
        static ModelAttentionExecutionPlan Resolve(
            const ModelAttentionCapabilities &capabilities,
            int batch,
            bool isPrefill,
            bool usesAlibi,
            bool kvCacheInCPU);

        /**
         * 取得指定模型层在每个请求中的K/V Cache描述符。
         *
         * pastKeyValues按request-major顺序保存：下标为
         * requestIndex * blockCount + layerIndex。本函数只定位Cache描述符，
         * Token到连续偏移或物理Page的二级寻址仍由具体Cache Backend完成。
         *
         * @param pastKeyValues batch * blockCount个K/V Cache描述符。
         * @param batch 请求数量。
         * @param blockCount Transformer层数。
         * @param layerIndex 当前层编号。
         * @param keys 返回每个请求当前层的K Cache。
         * @param values 返回每个请求当前层的V Cache。
         */
        static void BindLayerCaches(
            std::vector<std::pair<Data*, Data*>> &pastKeyValues,
            int batch,
            int blockCount,
            int layerIndex,
            std::vector<Data*> &keys,
            std::vector<Data*> &values);
    };

    /**
     * 持有模型一次Forward期间固定不变的Attention执行计划。
     *
     * 模型负责QKV投影、Norm、RoPE和残差；本组件负责统一布局决策以及
     * request-major KV Cache绑定。设备侧具体Kernel继续由CUDA Attention
     * Backend注册器选择，避免模型层重复实现Backend调度。
     */
    class AttentionExecutor {
    public:
        static AttentionExecutor Create(
            const ModelAttentionCapabilities &capabilities,
            int batch,
            bool isPrefill,
            bool usesAlibi,
            bool kvCacheInCPU);

        bool UsesPagedKVCache() const;
        bool IsPrefill() const;
        int Batch() const;

        void BindLayerCaches(
            std::vector<std::pair<Data*, Data*>> &pastKeyValues,
            int blockCount,
            int layerIndex,
            std::vector<Data*> &keys,
            std::vector<Data*> &values) const;

    private:
        explicit AttentionExecutor(const ModelAttentionExecutionPlan &plan);

        ModelAttentionExecutionPlan plan;
    };

} // namespace fastllm

#endif
