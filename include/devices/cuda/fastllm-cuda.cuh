#ifndef FASTLLM_CUDA_CUH
#define FASTLLM_CUDA_CUH

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
std::vector <long long> FastllmCudaGetTotalSizes();

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

// FlashInfer attention requires compute capability >= 7.5 (Turing+).
bool FastllmCudaFlashInferSupported();

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

void *FastllmCudaStreamCreate(bool nonBlocking = true);
void FastllmCudaStreamDestroy(void *stream);
void FastllmCudaStreamSynchronize(void *stream);

void *FastllmCudaEventCreate();
void FastllmCudaEventDestroy(void *event);
void FastllmCudaEventRecord(void *event, void *stream = nullptr);
void FastllmCudaEventSynchronize(void *event);
void FastllmCudaStreamWaitEvent(void *stream, void *event);

bool FastllmCudaGraphBeginCapture();
bool FastllmCudaGraphEndCapture(void **graph);
bool FastllmCudaGraphInstantiate(void *graph, void **exec);
bool FastllmCudaGraphLaunch(void *exec);
void FastllmCudaGraphDestroy(void *graph);
void FastllmCudaGraphExecDestroy(void *exec);
const char *FastllmCudaGraphLastError();
bool FastllmCudaGraphCaptureInvalidated();

// Qwen3.5 MoE graph markers are emitted only while the per-thread stream is
// being captured. After capture, the optimizer rewires the sequential region
// into shared/routed expert branches and removes every marker node. It returns
// the number of parallelized MoE layers, or -1 on an invalid graph/error.
void FastllmCudaGraphMarkQwen35MoeFork(int layer);
void FastllmCudaGraphMarkQwen35MoeSharedDone(int layer);
void FastllmCudaGraphMarkQwen35MoeRoutedBegin(int layer);
void FastllmCudaGraphMarkQwen35MoeJoin(int layer);
int FastllmCudaGraphOptimizeQwen35Moe(void *graph);
bool FastllmCudaGraphQwen35MoeSelfTest();

// 线程级 CUDA 错误标志：showError 报错时置位；graph 捕获路径用于错误熔断。
void FastllmCudaClearThreadError();
void FastllmCudaSetThreadError();
bool FastllmCudaGetThreadError();

void FastllmCudaMallocBigBuffer(size_t size);
void FastllmCudaClearBigBuffer();
#ifdef __CUDACC__
cudaError_t FastllmCudaCheckedMalloc(void **ret, size_t size, const char *file, int line);
#endif
void *FastllmCudaMalloc(size_t size);
void FastllmCudaForceFree(void *ret);
void FastllmCudaFree(void *ret);
void DisableCudaMalloc();
// 由 multicuda 在 NCCL 初始化成功后置位；置位后真实 cudaMalloc 前会先排空在途 NCCL 集合通信，
// 规避 cudaMalloc 与 NCCL 主机 proxy 争用 CUDA 驱动锁导致的跨 rank 死锁。
void FastllmCudaSetNcclActive(bool value);
// 控制 NCCL 集合通信是否「发射后立即同步」。默认 true（权重加载/warmup 阶段防死锁），
// warmup 成功结束后由 basellm 置 false 以恢复稳态解码吞吐。
void FastllmCudaSetNcclForceSync(bool value);
bool FastllmCudaGetNcclForceSync();
void FastllmCudaSetWeightSlabBytes(size_t bytes);
size_t FastllmCudaGetWeightSlabBytes();
void *FastllmCudaMallocModelWeight(size_t size);
void FastllmCudaMemPoolStats();
void * FastllmCudaDirectMalloc(size_t size);
void FastllmCudaDirectFree(void *ret);
void FastllmCudaMemset0(void *ret, size_t size);

// Borrow a per-device temporary CUDA buffer for short-lived intermediate data.
// Small requests reuse the existing FlashInfer float workspace when it has
// already been created and is large enough. Larger requests use one persistent
// grow-only temp buffer per device, so warmup can reserve the max scratch size
// and serving does not need to create another CUDA allocation.
//
// outOwn is kept for compatibility with older scratch users. Current borrowed
// buffers are cache-owned, so callers should still pair Release with Borrow but
// should not assume Release frees the CUDA memory.
void *FastllmBorrowCudaTempBuffer(size_t needBytes, size_t *outBytes, bool *outOwn);
void FastllmReleaseCudaTempBuffer(void *ptr, bool own);

// 借用 FlashInfer 的 d_float_workspace 作为临时 scratch（例如 INT4 反量化为 FP16 的临时缓冲）。
// 语义：
//   - 当前 device 的 workspace 指针 + 字节大小通过出参返回；
//   - 仅在两次 attention 调用之间使用是安全的，因为下一次 attention 会重新 plan 并覆盖里面的 tmp_v/tmp_s；
//   - 调用方需自行保证调用本身是串行的（同一个 stream），且不要在 attention kernel 还在跑时使用；
//   - 如果 workspace 还没有创建，会按默认大小（FT_FLOAT_WORKSPACE_SIZE 或 256MB）惰性分配。
// 注意：返回的指针只是借用，不需要 free。
void *FastllmCudaGetFlashInferFloatWorkspace(size_t *outSize);

// 借/还 dequant 用的临时 scratch buffer。
// FastllmBorrowDequantScratch:
//   - needBytes: 期望大小（字节）；如果为 0，按 workspace 大小返回。
//   - outBytes:  实际可用字节数（>= 1，可能小于 needBytes，表示需要分块）。
//   - outOwn:    兼容旧调用方；当前返回的 scratch 由缓存持有，调用方只需要配对 Release。
// 行为：优先借用已有 FlashInfer workspace；不足时使用每设备一个 grow-only temp buffer。
void *FastllmBorrowDequantScratch(size_t needBytes, size_t *outBytes, bool *outOwn);
// 与 Borrow 配对。
void FastllmReleaseDequantScratch(void *ptr, bool own);

bool FastllmCudaGptqMarlinRepack(const uint32_t *b_q_weight, uint32_t *out,
                                 int size_k, int size_n);
bool FastllmCudaMarlinHalfInt4Gemm(const void *a, const uint32_t *b_q_weight,
                                   const void *b_scales, const uint32_t *b_zeros,
                                   void *c, int size_m, int size_n, int size_k,
                                   int group_size, int *workspace);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromPinnedHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromPinnedHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);

void *FastllmCudaHostMalloc(size_t size);
void FastllmCudaHostFree(void *ptr);
bool FastllmCudaHostRegister(void *ptr, size_t size);
void FastllmCudaHostUnregister(void *ptr);

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
void FastllmCudaShiftAppendWindow(uint8_t *cache, const uint8_t *newToken, int channels, int window, int unitSize);
void FastllmCudaRepeat(void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen);
void FastllmCudaPagedCacheCopy(uint8_t *pagedData, int pageIdx, int pageLen, int numHeads, int headDim,
                               fastllm::DataType dstType, uint8_t *inputData, fastllm::DataType srcType,
                               int seqLen, int inputOffset, int copyLen, int pageOffset);
void FastllmCudaPagedCacheCopyBatch(uint8_t *pagedData, int32_t *pageIdxArray, int32_t *pageOffsetArray,
                                    int pageLen, int batch, int numHeads, int headDim,
                                    fastllm::DataType dstType, uint8_t *inputData, fastllm::DataType srcType,
                                    bool sync = true);

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
bool FastllmCudaEmbeddingDirect(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType);
bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaGeglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaClamp(fastllm::Data &input, bool hasMin, float minValue, bool hasMax, float maxValue);
bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData);
bool FastllmCudaSigmoidMambaSoftplus(fastllm::Data &sigmoidInputOutput, const fastllm::Data &softplusInput, fastllm::Data &softplusOutput, const fastllm::Data &aLogData, const fastllm::Data &dtBiasData);
bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCopy(const fastllm::Data &input, fastllm::Data &output);
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
bool FastllmCudaApplyChunkDecayByLastLogG(fastllm::Data &input, const fastllm::Data &g);

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
// Benchmark/validation entry. threadCount == 0 selects the legacy launch;
// threadCount == 32 selects the exact FP16 channel-128 specialization.
bool FastllmCudaRMSNormFloat16WithThreadCount(const fastllm::Data &input, fastllm::Data &weight,
                                              fastllm::Data &output, float eps, int threadCount);
bool FastllmCudaRMSNormPart(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps, int start, int end);
bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase, int startPos,
                                       int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                       float eps);
bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                      int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                      int quantDim, int blockSize, int posStep);
bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode);
bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits, fastllm::Data &tid2eid,
                                         const int *inputIds, int tokens, int topk,
                                         int scoreFuncMode, float routeScale,
                                         fastllm::Data &expertIndex, fastllm::Data &expertScore);
bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                fastllm::Data &hcScale, fastllm::Data &hcBase,
                                int hcMult, int sinkhornIters, float eps, float normEps,
                                fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb);
bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                    int hcMult, fastllm::Data &dotsFloat);
bool FastllmCudaDeepSeekV4StoreWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4BuildWindowKVPrefix(const fastllm::Data &windowKV, int startPos,
                                             int windowSize, int prefixLen, fastllm::Data &output);
bool FastllmCudaDeepSeekV4BuildCompressedKV(const fastllm::Data &kv, const fastllm::Data &score,
                                            const fastllm::Data &ape, int rawTokenBase, int rawLen,
                                            int blockStart, int blockCount, int compressRatio,
                                            int headDim, int wideDim, bool overlap,
                                            fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q, const fastllm::Data &windowKV,
                                                      const fastllm::Data &compressedKV, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, int compressedCount,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedBatch(
                                                      const std::vector<fastllm::Data*> &q,
                                                      const std::vector<fastllm::Data*> &windowKV,
                                                      const std::vector<fastllm::Data*> &compressedKV,
                                                      fastllm::Data &attnSink,
                                                      int windowSize,
                                                      const std::vector<int> &startPositions,
                                                      const std::vector<int> &compressedCounts,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionPrefill(const fastllm::Data &q, const fastllm::Data &kv,
                                                 fastllm::Data &attnSink, int windowSize, int startPos,
                                                 int compressRatio, int ropeDim, float ropeBase,
                                                 int originalSeqLen, float ropeFactor, int betaFast,
                                                 int betaSlow, float softmaxScale, fastllm::Data &output,
                                                 int prefixLen = 0);
bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA, int groups, int oRank, fastllm::Data &output);
bool FastllmCudaDeepSeekV4HcPost(const fastllm::Data &x, const fastllm::Data &residual, const float *post,
                                 const float *comb, int bsz, int seqlen, int hcMult, int dim,
                                 fastllm::Data &output);
bool FastllmCudaDeepSeekV4HcPostCudaMix(const fastllm::Data &x, const fastllm::Data &residual,
                                        const fastllm::Data &post, const fastllm::Data &comb,
                                        int bsz, int seqlen, int hcMult, int dim,
                                        fastllm::Data &output);
// 计算每个 outer 行在 [start, end) 范围内的 sum(x^2) (FP32)，用于多卡 RMSNorm 的跨卡归约。
// outer 与通道的物理布局来自 input；output sumOut 长度为 outer。
// 同时如果 copyInput == true 且 input != outputBuffer，会把 input 完整内容拷到 outputBuffer（用于后续 apply 阶段就地写回）。
bool FastllmCudaRMSNormPartSum2(const fastllm::Data &input, float *sumOut, int start, int end);
// 给定外部已经聚合好的 sumIn（长度 outer，FP32），按 partChannelsGlobal 计算 scale，并对 input[start:end) 做 weight * scale 写到 output。
// input == output 时为 in-place 操作；start/end 可以是 input 局部坐标，weight 物理上是与 partLocal 对齐的局部权重。
bool FastllmCudaRMSNormPartApply(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, const float *sumIn, float eps, int start, int end, int partChannelsGlobal);
bool FastllmCudaRMSNormSiluMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &gateInput, fastllm::Data &output, float eps);
bool FastllmCudaRMSNormSiluMulFloat16WithThreadCount(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &gateInput,
    fastllm::Data &output, float eps, int threadCount);
bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaFusedSoftmaxSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias,
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaMaskAndRemapExpertsForLocalRange(fastllm::Data &index, fastllm::Data &score,
                                                 int expertStart, int expertEnd);
bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmCudaPermuteTo(const fastllm::Data &input, fastllm::Data &output,
                          const std::vector<int> &axis);
bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaQuantizeLinearWeightFP8E4M3Block128(
    const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMatMulFloatGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaFloatMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                        fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaHalfMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

bool FastllmCudaConv1DPerChannelFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernel, int stride, int pad, fastllm::Data &output);
bool FastllmCudaConv1DPerChannelSiluSingleTokenFloat16(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(fastllm::Data &cache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluTwoTokenFloat16(fastllm::Data &cache, const fastllm::Data &newTokens, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output, fastllm::Data *firstTokenCache = nullptr);
bool FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(fastllm::Data &cache, const fastllm::Data &newTokens, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output, fastllm::Data **tokenCaches, int numTokenCaches);
bool FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16BatchPointers(
    const std::vector<fastllm::Data*> &caches, const fastllm::Data &newTokens,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output,
    const std::vector<fastllm::Data*> &tokenCaches, int numTokenCaches);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(const std::vector<fastllm::Data*> &caches, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(void *cudaCachePointers, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots(void *cudaCachePool, void *cudaSlotIds, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);

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
bool FastllmCudaLlama3RopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                   float ropeTheta, float factor, float originalMaxPosition,
                                   float lowFreqFactor, float highFreqFactor);
bool FastllmCudaQwen35InterleavedRope(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                      int sectionT, int sectionH, int sectionW,
                                      float ropeTheta, float ropeScale);
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
    int32_t *lastPageLens,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale,
    int pageLen, int maxPages, fastllm::DataType pagedDataType, int batch,
    int doQKNorm,
    int useLlama3 = 0, float llama3Factor = 1.0f,
    float llama3OriginalMaxPosition = 131072.0f,
    float llama3LowFreqFactor = 1.0f,
    float llama3HighFreqFactor = 32.0f);
bool FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qgatekv, fastllm::Data &qNormWeight, fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput, fastllm::Data &gateOutput,
    uint8_t *pagedKData, uint8_t *pagedVData,
    int32_t *insertIndexs, int32_t *insertPositions,
    int32_t *lastPageLens,
    int qHeads, int kHeads, int headDim,
    int rotaryDim, int sectionT, int sectionH, int sectionW,
    float eps, float ropeTheta, float ropeScale,
    int pageLen, fastllm::DataType pagedDataType, int batch,
    int doQKNorm);
bool FastllmCudaAdvanceDecodeMeta(
    int32_t *insertPositions, int32_t *lastPageLens, int batch);
bool FastllmCudaRepeatPenalty (fastllm::Data &input, fastllm::Data &penalty, fastllm::Data &penaltyScale);
bool FastllmCudaTopKTopPSampling(float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize);
bool FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
                                  float *logits, float *temperatures,
                                  int *topKArr, float *topPArr,
                                  int *output,
                                  int batch, int vocabSize,
                                  const int *typicalCandidateIds,
                                  const int *typicalCandidateRows,
                                  unsigned char *typicalAccepted,
                                  int *typicalRecoveredIds,
                                  int typicalCount,
                                  float typicalPosteriorThreshold,
                                  float typicalPosteriorAlpha);
bool FastllmCudaGreedySampling(float *logits, int *output,
                               int batch, int vocabSize);
bool FastllmCudaGreedySamplingWithScores(float *logits, int *output,
                                         float *scores, int batch,
                                         int vocabSize);
bool FastllmCudaSampleTopK(float *topk, float *temperatures,
                           int *topKArr, float *topPArr, float *randoms,
                           int *output,
                           int batch, int maxTopK);
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
bool FastllmCudaHalfPagedAttentionBatch(fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches, fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs, fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale, int attentionType, bool inited = false, bool sync = true, bool enableCudaGraph = false, int flashInferCudaGraph = -1);
bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo = false);
bool FastllmCudaHalfMatMulFloat16WithRouterSpecialization(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo, bool allowRouterSpecialization);
bool FastllmCudaHalfMatMulFloat16AddToNoBias(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
void FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaRegisterMoeFp8ExpertTableFromPacked(fastllm::Data **weights, int weightsBatch, int hidden, int inter,
                                                    void *packedGateWeights, void *packedGateScales,
                                                    void *packedDownWeights, void *packedDownScales,
                                                    int gateBlockM, int gateBlockK, int downBlockM, int downBlockK);
bool FastllmCudaHalfMergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                          fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                          bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                 fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                 const float *scores, int topk, int hidden, int inter,
                                                 bool allowWarpSpecialization = true);
bool FastllmCudaHalfMergeMOEFP8E4M3Block128Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int topk, int hidden, int inter);
bool FastllmCudaHalfFusedMOEFP8E4M3(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
                                    fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
                                    fastllm::Data &w1, fastllm::Data &output,
                                    int batch, int topk, int hidden, int inter, int experts, float swigluLimit,
                                    bool allowWarpSpecialization = true);
bool FastllmCudaHalfFusedMOEFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
                                            fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
                                            fastllm::Data &w1, fastllm::Data &output,
                                            int batch, int topk, int hidden, int inter, int experts, float swigluLimit);
bool FastllmCudaHalfMergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                  fastllm::Data **weights, int weightsBatch,
                                                  const int *routeRows, const float *routeScales,
                                                  const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                  int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                        fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                               fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                               const float *scores, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                fastllm::Data **weights, int weightsBatch,
                                                const int *routeRows, const float *routeScales,
                                                const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                       fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                       bool scoresOnCuda, int topk, int hidden, int inter);

bool FastllmCudaBFloat16MatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                              fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                              bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOEFP8E4M3Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                     fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                     const float *scores, int topk, int hidden, int inter,
                                                     bool allowWarpSpecialization = true);
bool FastllmCudaBFloat16MergeMOEFP8E4M3Block128Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                             fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                             const float *scores, int topk, int hidden, int inter);
bool FastllmCudaBFloat16FusedMOEFP8E4M3(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
                                        fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
                                        fastllm::Data &w1, fastllm::Data &output,
                                        int batch, int topk, int hidden, int inter, int experts, float swigluLimit,
                                        bool allowWarpSpecialization = true);
bool FastllmCudaBFloat16FusedMOEFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
                                                fastllm::Data &down, const fastllm::Data &index, const fastllm::Data &score,
                                                fastllm::Data &w1, fastllm::Data &output,
                                                int batch, int topk, int hidden, int inter, int experts, float swigluLimit);
bool FastllmCudaBFloat16MergeMOEFP8E4M3SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                         fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                         const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOEFP8E4M3GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                      fastllm::Data **weights, int weightsBatch,
                                                      const int *routeRows, const float *routeScales,
                                                      const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                      int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4Batch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                            fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                            bool scoresOnCuda, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4Batch1Indexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                       fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                       const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                    fastllm::Data **weights, int weightsBatch,
                                                    const int *routeRows, const float *routeScales,
                                                    const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                    int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4Block16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulNVFP4Block16E8M0(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                           fastllm::Data **gateups, fastllm::Data **downs, const float *scores,
                                           bool scoresOnCuda, int topk, int hidden, int inter);

bool FastllmCudaTritonLinearFP8E4M3Block128(
    const char *quantCubitPath, const char *quantKernelName, int quantNumWarps, int quantShared,
    const char *matmulCubitPath, const char *matmulKernelName, int matmulNumWarps, int matmulShared,
    int blockM, int blockN, int blockK, int groupSizeM, bool packedWeight, bool stridedMatmul,
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output,
    int n, int m, int k);

bool FastllmCudaTritonMergeMOEFP8E4M3Indexed(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN, int groupBlockK, int groupSizeM,
    const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
    fastllm::Data **weights, int weightsBatch, const int32_t *indices, const float *scores,
    int batch, int topk, int hidden, int inter);
bool FastllmCudaTritonMergeMOEFP8E4M3IndexedIsPacked(
    fastllm::Data **weights, int weightsBatch, int hidden, int inter);

bool FastllmCudaTritonFusedMOEFP8E4M3(
    const char *const *cubinPaths, const char *const *kernelNames,
    const int *numWarps, const int *shared,
    int routeBlockT, int maxExperts, int groupBlockM, int groupBlockN, int groupBlockK, int groupSizeM,
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up, fastllm::Data &down,
    const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &w1, fastllm::Data &output,
    int batch, int topk, int hidden, int inter, int experts);

bool FastllmCudaHalfMatMulFloat16Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lenght, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids);
void FastllmResetLogitsOfEOSAll(int batch, fastllm::Data *logits, const std::vector<int> &eos_ids);

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float qScale = 1.0f);
bool FastllmLinearAttentionStateTransposeKVToVKFloat16(fastllm::Data &last_recurrent_state);
bool FastllmLinearAttentionStateTransposeVKToKVFloat16(fastllm::Data &last_recurrent_state);
bool FastllmRecurrentGatedDeltaRuleNormTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &a, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &aLog, fastllm::Data &dtBias, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
// Benchmark/validation entry for the single-token transposed recurrent kernel.
bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16WithConfig(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale, int tileV, bool exactNorm128);
bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    fastllm::Data **tokenStates, int numTokenStates,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16BatchSnapshots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    const std::vector<fastllm::Data*> &lastRecurrentStates,
    fastllm::Data &coreAttnOut,
    const std::vector<fastllm::Data*> &tokenStates, int numTokenStates,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
void FastllmRecurrentGatedDeltaRuleBatch(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out, float qScale = 1.0f);
void FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposed(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    void *cudaStatePool, void *cudaSlotIds, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale = 1.0f);
void FastllmChunkGatedDeltaRulePrefill(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn, fastllm::Data &k_cumdecay,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out);

void FastllmCudaSetDevice(int gpu_id);
int FastllmCudaGetDevice();
int FastllmCudaRuntimeArch();
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

#ifndef FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#define cudaMalloc(ptr, size) FastllmCudaCheckedMalloc((void **)(ptr), (size), __FILE__, __LINE__)
#endif
#endif

#endif // FASTLLM_CUDA_CUH
