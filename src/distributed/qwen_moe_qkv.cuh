#ifndef FASTLLM_DISTRIBUTED_QWEN_MOE_QKV_CUH
#define FASTLLM_DISTRIBUTED_QWEN_MOE_QKV_CUH

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace fastllm {
namespace distributed {
namespace detail {

/** MoE 有偏置 Q/K/V 共用的单投影 Lt 计划；仅拥有 CPU 描述，不拥有设备内存。 */
class MoeQkvLtPlan {
public:
    static constexpr size_t kWorkspaceBytes = 1024 * 1024;

    /**
     * 为连续 FP16 Y[T,N]=X[T,K]*W[N,K]^T+bias[N] 固定一个 Lt 算法。
     *
     * 使用 FP32 计算、FP16 输出、beta=0 与 BIAS epilogue；列主序视图为
     * W[K,N]^T * X[K,T]。独立 Q/K/V 的 N 和输出步长不能换成合并 QKV 的宽度。
     * 本函数只在 Eager 或 Graph warmup 调用，捕获期间禁止创建计划。
     *
     * @param handle 当前 worker 的 Lt handle，须比计划存活更久。
     * @param t Token 数 T，必须为正。
     * @param n 单个投影的输出宽度 N，必须为正。
     * @param k 隐藏宽度 K，必须为正。
     * @param x 连续 FP16 输入 [T,K]，仅用于确定地址对齐。
     * @param weight 连续 FP16 权重 [N,K]，允许来自合并权重的连续行切片。
     * @param output 连续 FP16 输出 [T,N]，不得与输入或偏置重叠。
     * @param bias 非空 FP16 [N]，与权重、输入、输出位于当前设备。
     */
    MoeQkvLtPlan(cublasLtHandle_t handle, int t, int n, int k, const half *x,
                 const half *weight, half *output, const half *bias)
        : tokens(t), width(n), hidden(k), alignments(Alignments(x, weight, output, bias)) {
        if (!handle || t <= 0 || n <= 0 || k <= 0 || !x || !weight || !output || !bias) {
            throw std::runtime_error(Context() + "invalid shape/pointer/handle");
        }
        try {
            Check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "create descriptor");
            const cublasOperation_t transA = CUBLAS_OP_T, transB = CUBLAS_OP_N;
            const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            Attribute(CUBLASLT_MATMUL_DESC_TRANSA, transA);
            Attribute(CUBLASLT_MATMUL_DESC_TRANSB, transB);
            Attribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
            Attribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
            Check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, n, k), "layout A");
            Check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, t, k), "layout B");
            Check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16F, n, t, n), "layout C/D");

            Check(cublasLtMatmulPreferenceCreate(&preference), "create preference");
            Check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &kWorkspaceBytes, sizeof(kWorkspaceBytes)), "workspace preference");
            const cublasLtMatmulPreferenceAttributes_t keys[] = {
                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
            for (size_t i = 0; i < alignments.size(); ++i) {
                Check(cublasLtMatmulPreferenceSetAttribute(preference, keys[i], &alignments[i],
                    sizeof(alignments[i])), "alignment preference");
            }

            int count = 0;
            Check(cublasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c, preference, 1, &chosen, &count),
                  "heuristic");
            if (count != 1 || chosen.state != CUBLAS_STATUS_SUCCESS || chosen.workspaceSize > kWorkspaceBytes) {
                throw std::runtime_error(Context() + "no valid algorithm; count=" + std::to_string(count) +
                    " state=" + std::to_string(int(chosen.state)) +
                    " workspace=" + std::to_string(chosen.workspaceSize));
            }
        } catch (...) {
            Release();
            throw;
        }
    }

    ~MoeQkvLtPlan() { Release(); }
    MoeQkvLtPlan(const MoeQkvLtPlan&) = delete;
    MoeQkvLtPlan &operator=(const MoeQkvLtPlan&) = delete;

    /**
     * 检查新调用是否可复用既有算法，不查询启发式或读取设备数据。
     * @param t Token 数 T。
     * @param n 输出宽度 N。
     * @param k 输入宽度 K。
     * @param x 连续 FP16 [T,K]。
     * @param weight 连续 FP16 [N,K]。
     * @param output 连续 FP16 [T,N]。
     * @param bias 非空 FP16 [N]。
     * @return 形状与地址对齐全部一致时为 true；地址可不同，便于各层复用。
     */
    bool Matches(int t, int n, int k, const half *x, const half *weight,
                 const half *output, const half *bias) const {
        return t == tokens && n == width && k == hidden && x && weight && output && bias &&
            alignments == Alignments(x, weight, output, bias);
    }

    /**
     * 在 worker stream 提交既定投影；不分配、不同步、不查询算法，也不 fallback。
     * @param handle 创建计划时的 worker Lt handle。
     * @param stream 正式算子、KV 与 NCCL 共用的 CUDA stream。
     * @param x 已通过 Matches 的 FP16 [T,K]。
     * @param weight 已通过 Matches 的 FP16 [N,K]。
     * @param output 已通过 Matches 的 FP16 [T,N]。
     * @param bias 已通过 Matches 的 FP16 [N]；每次更新以适配不同层。
     * @param workspace 至少 kWorkspaceBytes 字节、256 字节对齐的设备空间；
     *                  由调用方的临时或 Graph 池拥有，并保证执行完成前不释放。
     */
    void Run(cublasLtHandle_t handle, cudaStream_t stream, const half *x, const half *weight,
             half *output, const half *bias, void *workspace) {
        Attribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
        const float alpha = 1, beta = 0;
        Check(cublasLtMatmul(handle, desc, &alpha, weight, a, x, b, &beta, output, c, output, c,
            &chosen.algo, workspace, kWorkspaceBytes, stream), "matmul");
    }

private:
    int tokens, width, hidden;
    std::array<uint32_t, 4> alignments;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t chosen{};

    /** @return 地址的实际二次幂对齐，最多 256 字节；空地址用于初始化时按 256 处理。 */
    static uint32_t Alignment(const void *pointer) {
        const auto address = reinterpret_cast<uintptr_t>(pointer);
        uint32_t alignment = 256;
        while (address % alignment) alignment /= 2;
        return alignment;
    }

    /** @return 与 PyTorch 2.7.1 gemm_and_bias 一致的 A/B/C/D 偏好；D 使用 bias 对齐。 */
    static std::array<uint32_t, 4> Alignments(const half *x, const half *weight,
                                              const half *output, const half *bias) {
        return {Alignment(weight), Alignment(x), Alignment(output), Alignment(bias)};
    }

    /** @return 包含实际后端、FP16/BIAS 语义及逻辑矩阵形状的错误前缀。 */
    std::string Context() const {
        return "MoE QKV cublasLt separate FP16 BIAS T=" + std::to_string(tokens) +
            " N=" + std::to_string(width) + " K=" + std::to_string(hidden) + ": ";
    }

    /** 带阶段和矩阵形状传播 Lt 失败；调用方须终止当前 worker，不重试其他算法。 */
    void Check(cublasStatus_t status, const char *phase) const {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(Context() + phase + " status=" + std::to_string(int(status)));
        }
    }

    /** 按属性实际类型设置描述；偏置为设备指针，传递指针值而非设备数据。 */
    template<typename T> void Attribute(cublasLtMatmulDescAttributes_t key, const T &value) {
        Check(cublasLtMatmulDescSetAttribute(desc, key, &value, sizeof(value)), "descriptor attribute");
    }

    /** 无异常释放 CPU 描述；不同步或释放设备资源，异常退出仍由 worker 回收 CUDA 上下文。 */
    void Release() noexcept {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (desc) cublasLtMatmulDescDestroy(desc);
        preference = nullptr;
        a = b = c = nullptr;
        desc = nullptr;
    }
};

} // namespace detail
} // namespace distributed
} // namespace fastllm

#endif
