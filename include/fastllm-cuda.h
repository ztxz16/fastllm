#include "fastllm.h"

#ifdef  __cplusplus
extern "C" {
#endif
void FastllmInitCublas(void);

bool FastllmMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

void *FastllmCudaMalloc(size_t size);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaFree(void *ret);

bool FastllmGelu(const fastllm::Data &input, fastllm::Data &output);

#ifdef  __cplusplus
}
#endif
