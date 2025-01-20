//
// Created by huangyuyang on 8/2/24.
//

#include "fastllm.h"

std::vector <long long> FastllmCudaGetFreeSizes();

#ifdef  __cplusplus
extern "C" {
#endif

void FastllmMultiCudaSetDevice(std::vector <int> ids);

bool FastllmMultiCudaHalfMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmMultiCudaMLP(const fastllm::Data &input, fastllm::Data &weight0, fastllm::Data &weight1, fastllm::Data &output);

#ifdef  __cplusplus
}
#endif
