#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>

namespace fastllm_machete {

// 仅限宿主侧接口；不会引入PyTorch或改变FastLLM的张量ABI。
struct Weight {
    const char *encoding = "";
    float maxScaleError = 0;
    float maxOffsetError = 0;
    size_t residentBytes = 0;
    virtual ~Weight() = default;
    virtual void Run(const void *input, void *output, const float *bias,
                     int rows, cudaStream_t stream) = 0;
};

/**
 * 为一个设备上的不可变量化权重创建独立的Machete预打包副本。
 *
 * 不释放或修改源权重。输入为[N,K]行主序，INT4偶数元素在高半字节；
 * scales/offsets为CPU FP32 [N,K/groupSize]，语义为q*scale+offset。
 * 转换至激活类型并记录最大舍入误差。调用须在CUDA Graph捕获之外进行。
 * @param bits 权重位数，4或8。
 * @param bf16 true使用BF16，否则FP16。
 * @param source 源设备权重指针。
 * @param n 输出通道数，须为128的倍数。
 * @param k 输入通道数，须为64的倍数。
 * @param groupSize 分组大小，须整除k且为64的倍数，或等于k。
 * @param scales 原始FP32比例系数。
 * @param offsets 原始FP32加法偏移量，不是未乘scale的zero-point。
 * @param stream 正式推理使用的CUDA stream。
 * @return 拥有设备资源的句柄；失败抛出异常，不自动回退。
 */
std::unique_ptr<Weight> Create(int bits, bool bf16, const void *source,
                             int n, int k, int groupSize,
                             const float *scales, const float *offsets,
                             cudaStream_t stream);
}
