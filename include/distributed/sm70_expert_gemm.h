#ifndef FASTLLM_DISTRIBUTED_SM70_EXPERT_GEMM_H
#define FASTLLM_DISTRIBUTED_SM70_EXPERT_GEMM_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastllm {
namespace distributed {

/** 在加载阶段检查当前设备为 SM70；检查编译产物及 kernel 资源，不分配内存，失败抛异常。 */
void ValidateSm70ExpertGemmDevice();

/**
 * 使用固定 CUTLASS SM70 配置提交 Y[M,N]=X[M,K]*W[N,K]^T+bias[N]。
 *
 * 输入/权重为连续 FP16，起始地址按 16 字节对齐，K 是 8 的倍数。输出连续，
 * N 可以为 1；固定 FP32 累加，无 Split-K，不按 M 调优。bias 先广播到输出，
 * epilogue 在最终类型转换前相加。复用调用方的 stream/缓冲区，无分配、同步或回退。
 * 加载阶段必须先调用设备验证；运行错误向上抛出，由进程组终止清理。
 * @param input FP16 [M,K] 激活，不与输出重叠。
 * @param weight FP16 [N,K] 权重，不与输出重叠。
 * @param bias 可空的 FP16 [N] 偏置，不与输出重叠。
 * @param output FP16 或 FP32 [M,N] 输出，按元素大小对齐。
 * @param m 非负 Token 行数；0 为合法空专家，不启动 kernel。
 * @param n 正数输出列数。
 * @param k 正数归约维度，必须为 8 的倍数。
 * @param fp32 true 输出 FP32，否则输出 FP16。
 * @param stream 正式 worker 的 CUDA stream，不能为默认空指针。
 */
void Sm70ExpertGemm(const half *input, const half *weight, const half *bias, void *output,
                    int m, int n, int k, bool fp32, cudaStream_t stream);

} // namespace distributed
} // namespace fastllm
#endif
