#include "fastllm.h"

#ifdef  __cplusplus
extern "C" {
#endif
void FastllmInitCublas(void);

void FastllmMatMulInt8(int8_t *a, int8_t *b, int32_t *c, int n, int m, int k);

void *FastllmCudaMalloc(size_t size);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaFree(void *ret);

#ifdef  __cplusplus
}
#endif
