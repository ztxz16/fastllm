#include "fastllm.h"

#ifdef  __cplusplus
extern "C" {
#endif
void FastllmInitCublas(void);

void FastllmCudaMallocBigBuffer(size_t size);
void FastllmCudaClearBigBuffer();
void *FastllmCudaMalloc(size_t size);
void FastllmCudaFree(void *ret);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);

void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height);
void FastllmCudaMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                       size_t *	spitchs, size_t *widths, size_t *	heights,
                                       int batch);

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool FastllmCudaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds);

bool FastllmCudaSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis);
bool FastllmCudaCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis);
bool FastllmCudaMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs);
bool FastllmCudaSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch);
bool FastllmCudaBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                      int *ns, int *ms, int *ks,
                                      int *i0Strides, int *i1Strides, float alpha, int batch);
bool FastllmCudaBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch);
void FastllmCudaSetDevice(int gpu_id);
#ifdef  __cplusplus
}
#endif
