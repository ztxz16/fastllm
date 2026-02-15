#include "fastllm.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <map>
#include <memory>

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
void showError(cudaError_t result, char const* const message, const char* const file, int const line);

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define CUDA_MAX(a, b) ((a) > (b) ? (a) : (b))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

typedef union __align__(16) {
    uint2 in;
    uint8_t out[8];
} union_char8;

typedef union __align__(16) {
    uint32_t in;
    uint8_t out[4];
} union_char4;

typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {
      // Do nothing
    }
} union_half4;

typedef union __align__(16) _union_half_8 {
    uint4 in;
    half out[8];
    half2 out2[4];
    __device__ _union_half_8() {
      // Do nothing
    }
} union_half8;
#else
typedef void* cublasHandle_t;
#endif

std::vector <long long> FastllmCudaGetFreeSizes();

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

#ifdef  __cplusplus
extern "C" {
#endif

struct CudaInfos {
    int cudaArch;
    bool hasTensorCore;

    CudaInfos ();
};

const size_t ST128_FP16_COUNT = 8;

CudaInfos *getCudaInfos();

void *FastllmCudaPrepareInput(const fastllm::Data &input);
void *FastllmCudaPrepareOutput(fastllm::Data &output);
void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
void FastllmCudaFinishOutput(fastllm::Data &output, void *data);
cublasHandle_t getFastllmCublasHandle();

void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *cudaIndex);
void FastllmCudaPickOutput(uint8_t *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, fastllm::DataType dataType);

void DeviceSync();
void ForceDeviceSync();
void FastllmInitCublas(void);

void FastllmCudaMallocBigBuffer(size_t size);
void FastllmCudaClearBigBuffer();
void *FastllmCudaMalloc(size_t size);
void FastllmCudaFree(void *ret);
void * FastllmCudaDirectMalloc(size_t size);
void FastllmCudaDirectFree(void *ret);
void FastllmCudaMemset0(void *ret, size_t size);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);

// 将 host 端数据拷到 GPU 临时缓冲区，按数据类型加到 dst（GPU）上，len 为元素个数
void FastllmCudaAddHostToDevice(void *dst, void *hostSrc, int len, fastllm::DataType dataType);
void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);

void FastllmCudaMemcpy2DDeviceToDeviceAuto(void * 	dst, size_t 	dpitch, const void * 	src,
    size_t 	spitch, size_t 	width, size_t 	height, int dstDeviceId, int srcDeviceId);
    
void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height);
void FastllmCudaMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                       size_t *	spitchs, size_t *widths, size_t *	heights,
                                       int batch);
void FastllmCudaRepeat(void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen);
void FastllmCudaPagedCacheCopy(uint8_t *pagedData, int pageIdx, int pageLen, int numHeads, int headDim, int unitSize, uint8_t *inputData, int seqLen, int inputOffset, int copyLen, int pageOffset);
void FastllmCudaPagedCacheCopyBatch(uint8_t *pagedData, int32_t *pageIdxArray, int32_t *pageOffsetArray, int pageLen, int batch, int numHeads, int headDim, int unitSize, uint8_t *inputData);

bool FastllmFloatToHalf(void *a, void *b, int len);
bool FastllmHalfToFloat(void *a, void *b, int len);
bool FastllmBF16ToFloat(void *a, void *b, int len);
bool FastllmFloatToBF16(void *a, void *b, int len);
bool FastllmBF16ToHalf(void *a, void *b, int len);
bool FastllmHalfToBF16(void *a, void *b, int len);

void FastllmReduce(uint8_t *output, uint8_t* partOutput, int len, int threadNum, fastllm::DataType dataType);

bool FastllmCudaMLA(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCache, const fastllm::Data &peCache, 
                    fastllm::Data &score, fastllm::Data &output, float softmaxScale);

bool FastllmCudaMLAPaged(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCachePaged, const fastllm::Data &peCachePaged,
                         fastllm::Data &output, float softmaxScale);

bool FastllmCudaEmbedding(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType);
bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData);
bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaAdd(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaTransferAttn(fastllm::Data &input);
bool FastllmCudaCumSumLastDim(fastllm::Data &input);
bool FastllmCudaCausalMask(fastllm::Data &input, int base, float maskValue);
bool FastllmCudaMakeDecayMask(fastllm::Data &input, fastllm::Data &output);

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaHalfMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaConv1DPerChannelFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernel, int stride, int pad, fastllm::Data &output);

bool FastllmCudaConv2DFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, fastllm::Data &output);

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
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int positionStride);
bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
bool FastllmCudaLlamaRotatePosition2DPart(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int part);
bool FastllmCudaRopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim, float ropeTheta, float ropeScale);
bool FastllmCudaQKVRMSNormRope(fastllm::Data &qkv, fastllm::Data &qNormWeight, fastllm::Data &kNormWeight,
                                const fastllm::Data &positionIds,
                                int q_heads, int k_heads, int head_dim,
                                int rotateDim, float eps, float ropeTheta, float ropeScale);
// 融合 QKVRMSNormRope + Split KV + AppendPagedCacheBatch
// qkv: [bs, seqlen, total_dim], qOutput: [bs*q_heads, seqlen, head_dim] (permuted)
// K/V 直接写入 paged cache
bool FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qkv, fastllm::Data &qNormWeight, fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    uint8_t *pagedKData, uint8_t *pagedVData,
    int32_t *insertIndexs, int32_t *insertPositions,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale,
    int pageLen, int unitSize, int batch);
bool FastllmCudaRepeatPenalty (fastllm::Data &input, fastllm::Data &penalty, fastllm::Data &penaltyScale);
bool FastllmCudaApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds);

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                          fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch);
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

bool FastllmCudaHalfAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType);
bool FastllmCudaHalfPagedAttention(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &output, int group, float scale, bool inited = false);
bool FastllmCudaHalfPagedAttentionBatch(fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches, fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs, fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale, int attentionType, bool inited = false);
bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo = false);
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaBFloat16MatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lenght, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids);

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out);

void FastllmCudaSetDevice(int gpu_id);
int FastllmCudaGetDevice();
int GetPointerDeviceId(void *ptr);
int FastllmCudaGetDeviceCount();
#ifdef  __cplusplus
}
#endif

#ifdef __CUDACC__
/* CUDA kernel declarations (shared by linear/ggml/attention .cu files) */
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);
extern __global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len);
extern __global__ void FastllmCudaBF162FloatKernel(uint16_t* a, float *b, int len);
extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2Bf16Kernel(float* a, __nv_bfloat16* b, int len);
extern __global__ void FastllmCudaBF162HalfKernel(uint16_t* a, half *b, int len);
extern __global__ void FastllmCudaHalf2BF16Kernel(half* a, __nv_bfloat16 *b, int len);
extern __global__ void FastllmCudaBiasKernel(__nv_bfloat16* a, __nv_bfloat16* bias, int k);
#endif
