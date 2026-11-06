#ifndef FASTLLM_ATTENTION_BACKEND_H
#define FASTLLM_ATTENTION_BACKEND_H

#include "fastllm.h"

#include <string>

namespace fastllm {

    enum class CudaAttentionCacheLayout {
        CONTIGUOUS,
        PAGED,
    };

    enum class CudaAttentionPhase {
        PREFILL,
        DECODE,
    };

    /**
     * 描述一次CUDA Attention调用的统一逻辑参数。
     *
     * 连续KV Cache路径使用qs、ks、vs、masks和outputs；分页KV Cache路径
     * 使用q、kCaches、vCaches及分页元数据。Backend必须先检查cacheLayout，
     * 不得把不兼容的Cache布局静默转换为另一种布局。
     */
    struct CudaAttentionInvocation {
        CudaAttentionCacheLayout cacheLayout = CudaAttentionCacheLayout::CONTIGUOUS;
        CudaAttentionPhase phase = CudaAttentionPhase::DECODE;
        DataType dataType = DataType::FLOAT32;
        int deviceId = -1;
        int batch = 0;
        int group = 1;
        int headDim = 0;
        float scale = 1.0f;
        int attentionType = 0;
        bool inited = false;
        bool sync = true;
        bool enableCudaGraph = false;
        int flashInferCudaGraph = -1;
        int windowLeft = -1;

        Data **qs = nullptr;
        Data **ks = nullptr;
        Data **vs = nullptr;
        Data **masks = nullptr;
        Data **outputs = nullptr;

        Data *q = nullptr;
        Data *kCaches = nullptr;
        Data *vCaches = nullptr;
        Data *qSizes = nullptr;
        Data *pageSizes = nullptr;
        Data *pageIndexs = nullptr;
        Data *lastPageLens = nullptr;
        Data *output = nullptr;
    };

    struct CudaAttentionSupportResult {
        bool supported = false;
        std::string reason;
    };

    using CudaAttentionSupport = CudaAttentionSupportResult (*)(
        const CudaAttentionInvocation &invocation);
    using CudaAttentionRun = bool (*)(CudaAttentionInvocation &invocation);

    struct CudaAttentionBackendRegistration {
        std::string name;
        int priority = 0;
        CudaAttentionSupport supports = nullptr;
        CudaAttentionRun run = nullptr;
    };

    /**
     * 注册一个CUDA Attention Backend。
     *
     * Backend名称在进程内必须唯一，并且必须在首次Attention执行前完成注册；
     * 注册完成后由统一选择器根据命令行配置、Cache布局、数据类型和运行阶段
     * 选择。重复名称或空回调会返回false。
     *
     * @param registration Backend名称、优先级、能力检查和执行回调。
     * @return true表示注册成功；false表示参数无效或名称已经存在。
     */
    bool RegisterCudaAttentionBackend(
        const CudaAttentionBackendRegistration &registration);

    /**
     * 通过已注册Backend执行一次Attention。
     *
     * auto模式按优先级选择兼容Backend；显式模式只允许指定Backend。选择
     * 结果按静态调用签名缓存，正式推理不会重复遍历注册表。失败原因写入
     * lastError；严格模式禁止显式Backend不兼容时静默fallback，非严格模式
     * 允许回退到auto选择结果。
     *
     * @param invocation 统一Attention调用参数。
     * @param actualBackend 返回实际执行的Backend名称。
     * @param lastError 返回选择或执行失败原因。
     * @return true表示Backend执行成功；false表示选择或执行失败。
     */
    bool RunCudaAttentionBackend(CudaAttentionInvocation &invocation,
                                 std::string &actualBackend,
                                 std::string &lastError);

    /** 注册当前CUDA构建提供的连续与分页Attention Backend。 */
    void RegisterBuiltinCudaAttentionBackends();
    void RegisterCudaContiguousAttentionBackends();
    void RegisterCudaPagedAttentionBackends();

} // namespace fastllm

#endif
