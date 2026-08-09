#ifndef FASTLLM_CUDA_CUH
#define FASTLLM_CUDA_CUH

#include "fastllm.h"

// Device-resident request/chunk offsets shared by the ragged GDN frontend,
// recurrent kernels, and output layout conversion.  The backing storage is
// owned by a per-worker CUDA cache; callers must only retain this view until
// the next call made by the same worker with different sequence lengths.
struct FastllmCudaRaggedGdnMetadataView {
    const int *tokenOffsets = nullptr;
    const int *chunkOffsets = nullptr;
    const int *chunkTokenBases = nullptr;
    const int *chunkValidTokens = nullptr;
    int batch = 0;
    int totalTokens = 0;
    int totalChunks = 0;
    int maxChunks = 0;
    int maxPaddedTokens = 0;
};

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
long long FastllmCudaGetFreeSize();
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
// BF16 FlashInfer kernels additionally require SM80; FP16 remains available
// from SM75. Other compute types are rejected.
bool FastllmCudaFlashInferDataTypeSupported(fastllm::DataType dataType);

void *FastllmCudaPrepareInput(const fastllm::Data &input);
void *FastllmCudaPrepareOutput(fastllm::Data &output);
void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
void FastllmCudaFinishOutput(fastllm::Data &output, void *data);
cublasHandle_t getFastllmCublasHandle();

void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *cudaIndex);
void FastllmCudaPickOutput(uint8_t *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, fastllm::DataType dataType);

void DeviceSync();
void ForceDeviceSync();
void FastllmCudaSyncCurrentThreadStream();
void FastllmInitCublas(void);

void *FastllmCudaStreamCreate(bool nonBlocking = true);
void FastllmCudaStreamDestroy(void *stream);
void FastllmCudaStreamSynchronize(void *stream);

void *FastllmCudaEventCreate();
void *FastllmCudaEventCreateTiming();
void FastllmCudaEventDestroy(void *event);
void FastllmCudaEventRecord(void *event, void *stream = nullptr);
void FastllmCudaEventRecordCurrentThread(void *event);
void FastllmCudaEventSynchronize(void *event);
float FastllmCudaEventElapsedTime(void *start, void *end);
void FastllmCudaStreamWaitEvent(void *stream, void *event);
void FastllmCudaCurrentThreadStreamWaitEvent(void *event);

bool FastllmCudaGraphBeginCapture();
// Resolve per-device process-lifetime graph sentinels before any stream in a
// multi-GPU capture has begun. Calling cudaMalloc after the first rank starts
// capture invalidates that rank even when the allocation targets another GPU.
bool FastllmCudaGraphPrepareCaptureDevice();
bool FastllmCudaGraphEndCapture(void **graph);
bool FastllmCudaGraphInstantiate(void *graph, void **exec);
bool FastllmCudaTensorParallelGreedyGatherGraphCreate(
        int rootDevice, int ranks,
        const void *const *ids, const void *const *scores,
        void *cudaGather, void *hostGather,
        size_t rankBytes, size_t scoreBase, size_t totalBytes,
        void **exec);
bool FastllmCudaGraphLaunch(void *exec);
void FastllmCudaGraphDestroy(void *graph);
void FastllmCudaGraphExecDestroy(void *exec);
const char *FastllmCudaGraphLastError();
bool FastllmCudaGraphIsCapturing();
bool FastllmCudaGraphCaptureInvalidated();

// Qwen3.5 MoE graph markers are emitted only while the per-thread stream is
// being captured. After capture, the optimizer rewires the sequential region
// into shared/routed expert branches and removes every marker node. CUDA does
// not allow node removal from graphs containing allocation/free nodes; in that
// case the optimizer leaves the graph untouched and asks the caller to capture
// again with markers disabled.
static constexpr int FASTLLM_CUDA_GRAPH_MOE_RECAPTURE_WITHOUT_MARKERS = -2;
bool FastllmCudaGraphSetQwen35MoeMarkersEnabled(bool enabled);
void FastllmCudaGraphMarkQwen35MoeFork(int layer);
void FastllmCudaGraphMarkQwen35MoeSharedDone(int layer);
void FastllmCudaGraphMarkQwen35MoeRoutedBegin(int layer);
void FastllmCudaGraphMarkQwen35MoeJoin(int layer);
// Returns the number of parallelized layers, the recapture status above, or
// -1 on an invalid graph/runtime error.
int FastllmCudaGraphOptimizeQwen35Moe(void *graph);
bool FastllmCudaGraphQwen35MoeSelfTest();
// CUDA graph 捕获期间，算子内部仍会从 FastLLM 内存池申请并归还临时块。
// 捕获结束后 Graph 会长期保存这些地址，因此需要把已归还、但被 Graph 引用的
// 内存池块保留到 Graph 销毁。Begin/End 之间允许正常复用，以保持逐层 workspace
// 的峰值而不是把整步所有临时张量累加起来。
bool FastllmCudaGraphMemoryPoolBegin();
bool FastllmCudaGraphMemoryPoolEnd(std::vector<void*> &reservedPointers);
void FastllmCudaGraphMemoryPoolAbort();
void FastllmCudaGraphMemoryPoolRelease(const std::vector<void*> &reservedPointers);
// Returns a valid device address only after an allocation failure in a managed
// whole-step capture. Kernels may retain this address while the failed capture
// is completed and discarded, but it must never be launched or freed by Data.
bool FastllmCudaGraphGetAllocationFailurePlaceholder(void **ptr);

// 线程级 CUDA 错误标志：showError 报错时置位；graph 捕获路径用于错误熔断。
void FastllmCudaClearThreadError();
void FastllmCudaSetThreadError();
bool FastllmCudaGetThreadError();
void FastllmCudaClearGraphError();
bool FastllmCudaGetGraphError();

// Best-effort warmup reserve allocation.  Returns the number of blocks added
// and preserves every existing pool entry when a later block cannot be
// allocated.  Capacity failures are reported to the caller without poisoning
// the CUDA thread/graph error flags, so the caller can restore its state and
// fail startup cleanly.
int FastllmCudaTryMallocBigBuffers(size_t size, int count);
void FastllmCudaMallocBigBuffer(size_t size);
void FastllmCudaClearBigBuffer();
#ifdef __CUDACC__
cudaError_t FastllmCudaCheckedMalloc(void **ret, size_t size, const char *file, int line);
#endif
void *FastllmCudaMalloc(size_t size);
void FastllmCudaForceFree(void *ret);
// Return a pooled allocation after all work already submitted to this host
// thread's per-thread stream has completed. The allocation is detached from
// its Data owner immediately but is not eligible for pool reuse until the
// recorded stream event is ready. Returns false for non-pooled allocations or
// while CUDA Graph capture owns the ordinary free path.
bool FastllmCudaFreeAfterCurrentThreadStream(void *ret);
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
void *FastllmCudaMallocModelWeight(size_t size, const std::string &name);
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
bool FastllmCudaGptqMarlinRepackStream(const uint32_t *b_q_weight, uint32_t *out,
                                       int size_k, int size_n, void *stream);
bool FastllmCudaGptqMarlinRepackBits(const uint32_t *b_q_weight, uint32_t *out,
                                     int size_k, int size_n, int num_bits);
bool FastllmCudaGptqMarlinRepackBitsStream(const uint32_t *b_q_weight, uint32_t *out,
                                           int size_k, int size_n, int num_bits,
                                           void *stream);
bool FastllmCudaMarlinHalfInt4Gemm(const void *a, const uint32_t *b_q_weight,
                                   const void *b_scales, const uint32_t *b_zeros,
                                   void *c, int size_m, int size_n, int size_k,
                                   int group_size, int *workspace);
// SM75+ weight-only FP8 Marlin (W8A16), for small-batch / MTP verify (n<=8).
// Returns false to fall back to native FP8 GEMV.
bool FastllmCudaMarlinHalfFP8Gemm(const void *a, const uint32_t *b_q_weight,
                                  const void *b_scales, void *c,
                                  int size_m, int size_n, int size_k,
                                  int group_size, int *workspace);
// SM75+ weight-only NVFP4 Marlin (W4A16, group size 16).  SM75 selects the
// two-stage Turing specialization; SM80+ selects the four-stage kernel.
bool FastllmCudaMarlinHalfNVFP4Gemm(const void *a,
                                    const uint32_t *b_q_weight,
                                    const void *b_scales,
                                    const float *global_scale, void *c,
                                    int size_m, int size_n, int size_k,
                                    int *workspace, void *c_tmp);
bool FastllmCudaMarlinNVFP4Supported(int size_n, int size_k);
bool FastllmCudaHasFp8MarlinLayout(const fastllm::Data &weight);
bool FastllmCudaTryMarlinHalfMatMulFloatFP8E4M3(const fastllm::Data &input,
                                                fastllm::Data &weight,
                                                const fastllm::Data &bias,
                                                 fastllm::Data &output,
                                                 int n, int m, int k);
bool FastllmCudaHasNVFP4MarlinLayout(const fastllm::Data &weight);
bool FastllmCudaTryMarlinHalfMatMulFloatNVFP4Block16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k);

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromPinnedHostToDevice(void *dst, void *src, size_t size);
void FastllmCudaCopyFromHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromPinnedHostToDeviceAsync(void *dst, void *src, size_t size, void *stream);
void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size);
bool FastllmCudaCopyFromDeviceToPinnedHostAsync(
    void *dst, const void *src, size_t size, void *stream);
bool FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
    void *dst, const void *src, size_t size);
bool FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
    void *dst, const void *src, size_t size);
void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size);
bool FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
    void *dst, const void *src, size_t size);
bool FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
    void *const *dsts, const void *const *srcs, const size_t *sizes, int count);

void *FastllmCudaHostMalloc(size_t size);
void FastllmCudaHostFree(void *ptr);
bool FastllmCudaHostRegister(void *ptr, size_t size);
void FastllmCudaHostUnregister(void *ptr);

// 将 host 端数据拷到 GPU 临时缓冲区，按数据类型加到 dst（GPU）上，len 为元素个数
void FastllmCudaAddHostToDevice(void *dst, void *hostSrc, int len, fastllm::DataType dataType);
void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size);
bool FastllmCudaMemcpyPeerAsyncCurrentThread(
    int dstId, void *dst, int srcId, const void *src, size_t size);

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
bool FastllmCudaPagedCacheCopyMultiPage(uint8_t *pagedData, const int *pageIdxHost, int pageCount,
                                        int firstPageOffset, int pageLen, int numHeads, int headDim,
                                        fastllm::DataType dstType, uint8_t *inputData,
                                        fastllm::DataType srcType, int seqLen);
bool FastllmCudaPreparePagedBatchParamsSingle(
    int32_t *qSizes, int32_t *pageSizes, int32_t *pageIndexs,
    int32_t *lastPageLens, const int *pageIdxHost, int pageIndexCount,
    int totalPages, int qSize, int lastPageLen);
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
                         fastllm::Data &output, float softmaxScale, int kvLen = -1);

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
bool FastllmCudaSigmoidMulTo(fastllm::Data &input,
                             const fastllm::Data &gate);
bool FastllmCudaClamp(fastllm::Data &input, bool hasMin, float minValue, bool hasMax, float maxValue);
bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData, float outputScale = 1.0f);
bool FastllmCudaSigmoidMambaSoftplus(fastllm::Data &sigmoidInputOutput, const fastllm::Data &softplusInput, fastllm::Data &softplusOutput, const fastllm::Data &aLogData, const fastllm::Data &dtBiasData);
bool FastllmCudaSigmoidMambaSoftplusCombinedFloat16(
        const fastllm::Data &input,
        const fastllm::Data &aLogData,
        const fastllm::Data &dtBiasData,
        int batch, int seqLen, int inputChannels,
        int baOffset, int channels,
        fastllm::Data &sigmoidOutput,
        fastllm::Data &softplusOutput);
bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCopy(const fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaAdd(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output);
bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha);
bool FastllmCudaMulToCausalMask(fastllm::Data &input0,
                                const fastllm::Data &input1,
                                float alpha, int base, float maskValue);
bool FastllmCudaMulCausalMask(const fastllm::Data &input0,
                              const fastllm::Data &input1,
                              fastllm::Data &output,
                              float alpha, int base, float maskValue);
// Merge the routed and shared Qwen3.5 MoE branches into destination. When
// sharedGate is non-null it contains the per-token gate. It is pre-sigmoid
// unless sharedGateAlreadySigmoid is true. Setting addResidual preserves
// destination and adds the merged local result to it; otherwise destination
// is overwritten. The FLOAT16 path deliberately keeps the same intermediate
// FP16 rounding as Sigmoid + MulTo + AddTo (+ AddTo).
bool FastllmCudaQwen35FusedMoeJoin(
        fastllm::Data &destination,
        const fastllm::Data &routedOutput,
        const fastllm::Data &sharedOutput,
        const fastllm::Data *sharedGate,
        bool addResidual,
        bool sharedGateAlreadySigmoid = false);
bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue);
bool FastllmCudaTransferAttn(fastllm::Data &input);
bool FastllmCudaCumSumLastDim(fastllm::Data &input);
bool FastllmCudaCausalMask(fastllm::Data &input, int base, float maskValue);
bool FastllmCudaMakeDecayMask(fastllm::Data &input, fastllm::Data &output);
bool FastllmCudaCumSumLastDimMakeDecayMask(fastllm::Data &input,
                                           fastllm::Data &output);
bool FastllmCudaCumSumDecayMaskNegMulCausal(
        fastllm::Data &input, const fastllm::Data &matrix,
        fastllm::Data &decayMask, fastllm::Data &output);
bool FastllmCudaApplyChunkDecayByLastLogG(fastllm::Data &input, const fastllm::Data &g);

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmCudaRMSNormCombinedQKFloat16(
        const fastllm::Data &qkvInput, const fastllm::Data &weight,
        int batch, int seqLen, int keyHeads, int valueHeads,
        int kDim, int vDim, float eps,
        fastllm::Data &q, fastllm::Data &k);
bool FastllmCudaQwen35GdnPostConvExactFloat16(
        const fastllm::Data &qkvInput, const fastllm::Data &normWeight,
        const fastllm::Data &gInput, const fastllm::Data &betaInput,
        int batch, int seqLen, int keyHeads, int valueHeads,
        int kDim, int vDim, float normEps, float qScale,
        fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
        fastllm::Data &g, fastllm::Data &beta,
        fastllm::Data &kBeta, fastllm::Data &vBeta);
bool FastllmCudaQwen35GdnPostConvRaggedExactFloat16(
        const fastllm::Data &qkvInput,
        const fastllm::Data &normWeight,
        const fastllm::Data &combinedBaInput,
        const fastllm::Data &aLog,
        const fastllm::Data &dtBias,
        int baOffset, const std::vector<int> &seqLens,
        int chunkSize, int keyHeads, int valueHeads,
        int kDim, int vDim, float normEps, float qScale,
        fastllm::Data &q, fastllm::Data &k,
        fastllm::Data &g, fastllm::Data &kBeta,
        fastllm::Data &vBeta);
bool FastllmCudaKimiK3RMSNorm(const fastllm::Data &input,
                              const fastllm::Data &weight,
                              fastllm::Data &output, float eps);
bool FastllmCudaKimiK3CausalConv1D(const fastllm::Data &input,
                                   const fastllm::Data &weight,
                                   fastllm::Data *cache,
                                   fastllm::Data &output, int kernelSize,
                                   bool initializeCache);
bool FastllmCudaKimiK3UpdatePackedConvCache(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, fastllm::Data &cache,
        int history, int tokens);
bool FastllmCudaKimiK3L2Norm(const fastllm::Data &input,
                             fastllm::Data &output, float eps);
bool FastllmCudaKimiK3RecurrentKDA(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &rawGate,
        const fastllm::Data &rawBeta, const fastllm::Data &aLog,
        const fastllm::Data &dtBias, fastllm::Data &state,
        fastllm::Data &output, fastllm::Data &decay,
        fastllm::Data &beta, float lowerBound, bool initializeState,
        int tokenLimit = -1, bool stateOnly = false,
        bool outputAux = true);
bool FastllmCudaKimiK3RMSNormSigmoidGate(
        const fastllm::Data &input, const fastllm::Data &gate,
        const fastllm::Data &weight, fastllm::Data &output, float eps);
bool FastllmCudaKimiK3AttnRes(
        const fastllm::Data &prefixSum,
        const fastllm::Data &blockResidual,
        const fastllm::Data &projection, const fastllm::Data &norm,
        fastllm::Data &output, float eps);
bool FastllmCudaKimiK3SiTUAndMul(
        const fastllm::Data &gate, const fastllm::Data &up,
        fastllm::Data &output, float beta, float linearBeta);
bool FastllmCudaKimiK3CausalAttention(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, fastllm::Data &output, float scale);
// Benchmark/validation entry. threadCount == 0 selects the legacy launch;
// threadCount == 32 selects the exact FP16 channel-128 specialization.
bool FastllmCudaRMSNormFloat16WithThreadCount(const fastllm::Data &input, fastllm::Data &weight,
                                              fastllm::Data &output, float eps, int threadCount);
// Benchmark/validation entry. threadCount == 0 selects the legacy launch;
// threadCount == 256 selects the exact BF16 channel-3072 specialization.
bool FastllmCudaRMSNormBFloat16WithThreadCount(const fastllm::Data &input, fastllm::Data &weight,
                                               fastllm::Data &output, float eps, int threadCount);
bool FastllmCudaRMSNormPart(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps, int start, int end);
bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase, int startPos,
                                       int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                       float eps);
bool FastllmCudaDeepSeekV4ScaleQRotaryGraph(fastllm::Data &q, int ropeDim, float ropeBase,
                                            const int32_t *decodeMeta, int originalSeqLen,
                                            float ropeFactor, int betaFast, int betaSlow, float eps);
bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase, int startPos,
                                      int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                      int quantDim, int blockSize, int posStep);
bool FastllmCudaDeepSeekV4RotaryQuantGraph(fastllm::Data &x, int ropeDim, float ropeBase,
                                           const int32_t *decodeMeta, int originalSeqLen,
                                           float ropeFactor, int betaFast, int betaSlow,
                                           int quantDim, int blockSize, int posStep);
// Decode-only horizontal fusion matching DeepSeek-V4's production shape:
// per-head Q RMSNorm + RoPE, weighted KV RMSNorm + RoPE + quant round-trip,
// and the FLOAT32 sliding-window cache write. Unsupported layouts return false.
bool FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                                            fastllm::Data &q, fastllm::Data &kv,
                                            fastllm::Data &kvNormWeight,
                                            const int32_t *decodeMeta,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow, float eps,
                                            int quantDim, int quantBlockSize,
                                            int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode);
// Fused DeepSeek-V4 sqrt-softplus router for the production 256-expert,
// top-6 shape. The built-in CUDA kernel is architecture-generic; when
// allowTriton is true an eligible SM120 device may use the faster Triton path.
bool FastllmCudaDeepSeekV4SqrtSoftplusRouter(const fastllm::Data &logits,
                                             const fastllm::Data &gateBias,
                                             float routeScale,
                                             fastllm::Data &expertIndex,
                                             fastllm::Data &expertScore,
                                             bool allowTriton = true);
bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits, fastllm::Data &tid2eid,
                                         const int *inputIds, int tokens, int topk,
                                         int scoreFuncMode, float routeScale,
                                         fastllm::Data &expertIndex, fastllm::Data &expertScore);
bool FastllmCudaDeepSeekV4HashRouteScoreGraph(const fastllm::Data &logits, fastllm::Data &tid2eid,
                                              const int32_t *decodeMeta, int tokens, int topk,
                                              int scoreFuncMode, float routeScale,
                                              fastllm::Data &expertIndex, fastllm::Data &expertScore);
// Drop every per-device CUDA copy owned by a CPU route table. Data destruction
// calls this before the object's address can be reused by another model.
void FastllmCudaReleaseDeepSeekV4RouteTableCache(const fastllm::Data *routeTable);
bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                fastllm::Data &hcScale, fastllm::Data &hcBase,
                                int hcMult, int sinkhornIters, float eps, float normEps,
                                fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb);
// Decode-only fused mHC pre + RMSNorm.  Unsupported layouts return false so
// callers can fall back to DeepSeekV4HcPre followed by RMSNorm.
bool FastllmCudaDeepSeekV4HcPreNorm(const fastllm::Data &x, fastllm::Data &hcFn,
                                    fastllm::Data &hcScale, fastllm::Data &hcBase,
                                    fastllm::Data &normWeight, int hcMult,
                                    int sinkhornIters, float eps, float normEps,
                                    fastllm::Data &normOutput, fastllm::Data &post,
                                    fastllm::Data &comb);
// Decode-only fusion across an mHC transition: previous HcPost, next HcPre,
// and the next block's RMSNorm. Unsupported layouts return false.
bool FastllmCudaDeepSeekV4HcPostPreNorm(
                                    const fastllm::Data &x,
                                    const fastllm::Data &residual,
                                    const fastllm::Data &previousPost,
                                    const fastllm::Data &previousComb,
                                    fastllm::Data &nextHcFn,
                                    fastllm::Data &nextHcScale,
                                    fastllm::Data &nextHcBase,
                                    fastllm::Data &nextNormWeight,
                                    int hcMult, int sinkhornIters,
                                    float eps, float normEps,
                                    fastllm::Data &residualOutput,
                                    fastllm::Data &normOutput,
                                    fastllm::Data &nextPost,
                                    fastllm::Data &nextComb);
bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                    int hcMult, fastllm::Data &dotsFloat);
bool FastllmCudaDeepSeekV4HcHead(const fastllm::Data &x, const fastllm::Data &hcFn,
                                 const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                 int hcMult, float eps, float normEps, fastllm::Data &output);
// FP32-accumulating mean over the mHC axis. Unsupported layouts return false
// so CPU and older/general execution retain the operator-composed fallback.
bool FastllmCudaDeepSeekV4HcMean(const fastllm::Data &x,
                                 fastllm::Data &output);
bool FastllmCudaDeepSeekV4DsparkMarkovLocalArgmax(
    const float *baseLogits, const float *markovBias,
    int *packedCandidate, int vocabSize);
// SM120 graph path: copy the root rank's tiny FP32 Markov latent once to each
// TP rank, then compute the local FP16 weight shard from that local replica.
// Unsupported devices return false and retain the operator-composed fallback.
bool FastllmCudaDeepSeekV4DsparkMarkovPeerAvailable();
bool FastllmCudaDeepSeekV4DsparkMarkovLinearPeer(
    const float *peerLatent, const fastllm::Data &localWeight,
    float *localOutput, int hiddenSize, int localVocabSize);
bool FastllmCudaDeepSeekV4DsparkMarkovSignal(
    uint32_t *signal, int step);
bool FastllmCudaDeepSeekV4DsparkMarkovCopyPeer(
    const uint32_t *peerSignal, uint32_t *localSeen, int step,
    const float *peerLatent, float *localLatent, int hiddenSize);
bool FastllmCudaDeepSeekV4DsparkMarkovWaitPeer(
    const uint32_t *peerSignal, uint32_t *localSeen, int step);
bool FastllmCudaDeepSeekV4DsparkMarkovSelect(
    const int *packedCandidates, const int *globalOffsets,
    int ranks, int *proposalIds, float *previousId, int step);
bool FastllmCudaDeepSeekV4DsparkMarkovSelectPeer(
    const uint64_t *peerCandidatePointers,
    const uint64_t *peerSignalPointers, uint32_t *localSeen,
    const int *globalOffsets, int ranks, int steps,
    int *proposalIds, float *previousId, int step);
// SM120 steady-state DSpark handoff.  Wait for the root draft proposal and
// populate this rank's target decode metadata without a GPU-to-host boundary.
bool FastllmCudaDeepSeekV4DsparkPrepareTargetPeer(
    const uint32_t *peerSignal, uint32_t *localSeen,
    const int *peerProposalIds, int proposalCount,
    int anchorToken, int startPos, int32_t *decodeMeta,
    float *inputIds);
// SM120 verifier postprocess: reduce the TP-local greedy candidates, compare
// them with the draft proposal, and publish the accepted prefix on the root.
bool FastllmCudaDeepSeekV4DsparkAcceptPeer(
    const int *candidateIds, const float *candidateScores,
    const int *globalOffsets, const int *proposalIds,
    int ranks, int rows, int *result, uint32_t *readySignal);
// SM120 next-draft preamble.  Every TP rank waits for the root acceptance,
// commits the dynamic prefix into its three chronological draft KV windows,
// and fills the stable draft graph metadata/input allocations.
bool FastllmCudaDeepSeekV4DsparkPrepareDraftPeer(
    const uint32_t *peerSignal, uint32_t *localSeen,
    const int *peerResult, int baseCommittedTokens,
    const void *stageKv0, void *windowKv0,
    const void *stageKv1, void *windowKv1,
    const void *stageKv2, void *windowKv2,
    int rows, int windowSize, int headDim,
    int noiseTokenId, int proposalCount,
    int32_t *decodeMeta, float *inputIds);
bool FastllmCudaDeepSeekV4StoreWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                             int windowSize, fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(const fastllm::Data &kv,
                                                   const int32_t *decodeMeta,
                                                   int windowSize, fastllm::Data &windowKV);
// Append the first appendTokens rows of kv to an already-full chronological
// window in place. Appends longer than the window retain the trailing window
// rows of that committed prefix. This keeps the cache address captured by the
// draft CUDA graph stable and needs no temporary allocation.
bool FastllmCudaDeepSeekV4AppendFullWindowKVCache(const fastllm::Data &kv,
                                                  int appendTokens,
                                                  fastllm::Data &windowKV);
bool FastllmCudaDeepSeekV4BuildWindowKVPrefix(const fastllm::Data &windowKV, int startPos,
                                             int windowSize, int prefixLen, fastllm::Data &output);
bool FastllmCudaDeepSeekV4BuildCompressedKV(const fastllm::Data &kv, const fastllm::Data &score,
                                            const fastllm::Data &ape, int rawTokenBase, int rawLen,
                                            int blockStart, int blockCount, int compressRatio,
                                            int headDim, int wideDim, bool overlap,
                                            fastllm::Data &output);
// Preserve the eager pipeline's FP32 -> BF16 -> RMSNorm -> BF16 -> RoPE /
// quantization boundaries, but finish each compressed row in one launch and
// write it directly into an already-reserved cache.
bool FastllmCudaDeepSeekV4FinalizeCompressedKV(
                                            const fastllm::Data &compressed,
                                            const fastllm::Data &normWeight,
                                            int blockStart, int compressRatio,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &cache);
// Drop a disjoint prefix from the two raw compressor caches in place.  The
// allocation and its reserve stride remain stable; only logical dimensions
// are shortened after the byte-identical device copy is enqueued.
bool FastllmCudaDeepSeekV4CompactCompressorRaw(fastllm::Data &kv,
                                               fastllm::Data &score,
                                               int dropLen);
bool FastllmCudaDeepSeekV4InitGraphRawRing(const fastllm::Data &raw, int rawTokenBase,
                                           fastllm::Data &ring);
bool FastllmCudaDeepSeekV4UpdateCompressedKVGraph(
                                            const fastllm::Data &kv, const fastllm::Data &score,
                                            const fastllm::Data &ape, const fastllm::Data &normWeight,
                                            const int32_t *decodeMeta, int compressRatio,
                                            int headDim, int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &kvRing, fastllm::Data &scoreRing,
                                            fastllm::Data &compressedKV);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q, const fastllm::Data &windowKV,
                                                      const fastllm::Data &compressedKV, fastllm::Data &attnSink,
                                                      int windowSize, int startPos, int compressedCount,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                                                      const fastllm::Data &q,
                                                      const fastllm::Data &windowKV,
                                                      const fastllm::Data &compressedKV,
                                                      const fastllm::Data *compressedIndices,
                                                      const fastllm::Data *compressedLengths,
                                                      fastllm::Data &attnSink, int windowSize,
                                                      int compressRatio, const int32_t *decodeMeta,
                                                      int ropeDim, float ropeBase, int originalSeqLen,
                                                      float ropeFactor, int betaFast, int betaSlow,
                                                      float softmaxScale, fastllm::Data &output,
                                                      bool allowTriton = true);
// SM120-only optimized path. The two cache tensors use FlashInfer's packed
// DSv4 ABI (64-token pages, 584 logical bytes/token). These helpers return
// false on unsupported devices/layouts so callers retain the generic kernel.
bool FastllmCudaDeepSeekV4SparseMlaSm120Available();
bool FastllmCudaDeepSeekV4PrepareSparseMlaSm120Cache(
                                                      const fastllm::Data &windowKV,
                                                      int totalLen, int windowSize,
                                                      const fastllm::Data &compressedKV,
                                                      int compressedCount,
                                                      fastllm::Data &packedWindowKV,
                                                      fastllm::Data &packedCompressedKV);
// Build C4 learned-indexer candidates.  The function uses the exact SM120
// DeepGEMM MQA scorer when available and an architecture-independent CUDA
// scorer otherwise; both preserve vLLM's ascending shortcut for <=512 rows.
bool FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                                                      const fastllm::Data &q,
                                                      const fastllm::Data &weights,
                                                      const fastllm::Data &compressedKV,
                                                      const int32_t *decodeMeta,
                                                      int compressRatio,
                                                      float ropeBase,
                                                      int originalSeqLen,
                                                      float ropeFactor,
                                                      int betaFast,
                                                      int betaSlow,
                                                      fastllm::Data &indices,
                                                      fastllm::Data &lengths);
size_t FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120ScratchBytes(
                                                      int seqlen, int heads,
                                                      int compressRatio);
bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120(
                                                      const fastllm::Data &q,
                                                      const fastllm::Data &windowKV,
                                                      const fastllm::Data &compressedKV,
                                                      const fastllm::Data *compressedIndices,
                                                      const fastllm::Data *compressedLengths,
                                                      fastllm::Data &packedWindowKV,
                                                      fastllm::Data &packedCompressedKV,
                                                      fastllm::Data *scratch,
                                                      fastllm::Data &attnSink,
                                                      int windowSize, int compressRatio,
                                                      const int32_t *decodeMeta,
                                                      int ropeDim, float ropeBase,
                                                      int originalSeqLen, float ropeFactor,
                                                      int betaFast, int betaSlow,
                                                      float softmaxScale,
                                                      fastllm::Data &output);
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
                                                 int prefixLen = 0,
                                                 bool nonCausalBlock = false,
                                                 const int32_t *decodeMeta = nullptr);
bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA,
                              int groups, int oRank, fastllm::Data &output,
                              bool allowTriton = true);
bool FastllmCudaDeepSeekV4PrepareMoeDownInput(
                              const fastllm::Data &gateUp,
                              fastllm::Data &downInput,
                              const float *routeScales,
                              float swigluLimit,
                              bool quantize);
#ifdef FASTLLM_ENABLE_DSV4_WOA_DEEPGEMM_SM120
extern "C" bool FastllmCudaDeepSeekV4WoADeepGemmSm120(
                              const fastllm::Data &o,
                              const fastllm::Data &woA,
                              int groups, int oRank,
                              fastllm::Data &output);
#endif
namespace fastllm {
bool FastllmCudaTryTritonDeepSeekV4WoA(const Data &o, Data &woA,
                                       int groups, int oRank, Data &output);
bool FastllmCudaTryCombinedBaSigmoidMambaSoftplus(
        const Data &input, const Data &aLog, const Data &dtBias,
        int batch, int seqLen, int inputChannels,
        int baOffset, int channels,
        Data &sigmoidOutput, Data &softplusOutput);
bool FastllmCudaTryCombinedGdnConvInput(
        const std::vector<Data*> &caches,
        const Data &combinedInput,
        Data &weight, Data &bias, Data &output);
bool FastllmCudaTryCombinedGdnZGate(
        const Data &input, Data &weight,
        const Data &combinedGateInput,
        int gateOffset, int gateHeads,
        Data &output, float eps);
bool FastllmCudaTryCombinedGdnOutputGate(
        const Data &headMajorInput, Data &weight,
        const Data &combinedGateInput,
        int batch, int seqLen,
        int gateOffset, int gateHeads,
        Data &output, float eps);
bool FastllmCudaTryTritonChunkGdnPostConv(
        const Data &qkvInput, const Data &normWeight,
        const Data &gInput, const Data &betaInput,
        int batch, int seqLen, int keyHeads, int valueHeads,
        int kDim, int vDim, float normEps, float qScale,
        Data &normalizedQ, Data &normalizedK,
        Data &q, Data &k, Data &v, Data &g, Data &beta,
        Data &kBeta, Data &vBeta);
bool FastllmCudaTryChunkGdnRaggedPostConv(
        const Data &qkvInput, const Data &normWeight,
        const Data &combinedBaInput, const Data &aLog,
        const Data &dtBias, int baOffset,
        const std::vector<int> &seqLens, int chunkSize,
        int keyHeads, int valueHeads, int kDim, int vDim,
        float normEps, float qScale,
        Data &q, Data &k, Data &g,
        Data &kBeta, Data &vBeta);
bool FastllmCudaMappedGdnKkt(
        const Data &kBeta, const Data &k,
        int headGroup, Data &output);
bool FastllmCudaTryTritonChunkGdnRecompute(
        const Data &attn, const Data &vBeta,
        const Data &kBeta, const Data &gExp, const Data &g,
        Data &vOutput, Data &kOutput);
bool FastllmCudaChunkGatedDeltaRuleVarlenPrefill(
        Data &q, Data &k, Data &v, Data &g, Data &attn,
        Data &decayMask, Data &kCumdecay, Data &lastRecurrentState,
        bool fuseDecayMask, bool directOutputQk,
        const std::vector<int> &seqLens, Data &coreAttnOut);
// Runs only the optional Triton implementation. Unlike the availability-first
// wrapper above, this returns false without entering the native CUDA fallback.
bool FastllmCudaTryTritonChunkGdnVarlenPrefill(
        Data &q, Data &k, Data &v, Data &g, Data &attn,
        Data &decayMask, Data &kCumdecay, Data &lastRecurrentState,
        bool fuseDecayMask, bool directOutputQk,
        const std::vector<int> &seqLens, Data &coreAttnOut);
bool FastllmCudaTryTritonDeepSeekV4SparseAttentionDecodeGraph(
        const Data &q, const Data &windowKV, const Data &compressedKV,
        const Data &attnSink, int windowSize, int compressRatio,
        const int32_t *decodeMeta, float softmaxScale, float *output);
bool FastllmCudaTryTritonDeepSeekV4SqrtSoftplusRouter(
        const Data &logits, const Data &gateBias,
        float routeScale, Data &expertIndex, Data &expertScore);
}
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
bool FastllmCudaRMSNormSiluMulFloat16CombinedGate(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &combinedGateInput,
    int gateOffset, int gateHeads,
    fastllm::Data &output, float eps);
bool FastllmCudaRMSNormSiluMulFloat16HeadMajorCombinedGate(
    const fastllm::Data &headMajorInput, fastllm::Data &weight,
    const fastllm::Data &combinedGateInput,
    int batch, int seqLen,
    int gateOffset, int gateHeads,
    fastllm::Data &output, float eps);
bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis);
bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaFusedSoftmaxSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias,
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
bool FastllmCudaFusedSigmoidSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias,
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale);
// Remap global expert ids in [expertStart, expertEnd) to local ids. Remote slots
// become index -1 with score 0 so fused MoE kernels can skip them entirely.
bool FastllmCudaMaskAndRemapExpertsForLocalRange(fastllm::Data &index, fastllm::Data &score,
                                                 int expertStart, int expertEnd);
bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);
bool FastllmCudaPermuteTo(const fastllm::Data &input, fastllm::Data &output,
                          const std::vector<int> &axis);
bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaMatMulFloatInt4Group32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
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
bool FastllmCudaMatMulFloatFP8E4M3PerChannel(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
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
    const std::vector<fastllm::Data*> &tokenCaches, int numTokenCaches,
    int tokenMajorInputOffset = 0);
bool FastllmCudaShiftAppendConv1DPerChannelSiluRaggedPrefillFloat16BatchPointers(
    const std::vector<fastllm::Data*> &caches, const fastllm::Data &newTokens,
    const std::vector<int> &seqLens, fastllm::Data &weight,
    fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(const std::vector<fastllm::Data*> &caches, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(void *cudaCachePointers, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);
bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots(void *cudaCachePool, void *cudaSlotIds, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output);

bool FastllmCudaPackRaggedGdnPrefillFloat16(
    const fastllm::Data &q, const fastllm::Data &k,
    const fastllm::Data &v, const fastllm::Data &b,
    const fastllm::Data &g, const std::vector<int> &seqLens,
    int paddedSeqLen, float qScale,
    fastllm::Data &qPadded, fastllm::Data &kPadded,
    fastllm::Data &vPadded, fastllm::Data &bPadded,
    fastllm::Data &gPadded);
bool FastllmCudaUnpackRaggedGdnPrefillFloat16(
    const fastllm::Data &padded, const std::vector<int> &seqLens,
    fastllm::Data &ragged);
bool FastllmCudaPackRaggedGdnPrefillChunksFloat16(
    const fastllm::Data &q, const fastllm::Data &k,
    const fastllm::Data &v, const fastllm::Data &b,
    const fastllm::Data &g, const std::vector<int> &seqLens,
    int chunkSize, float qScale,
    fastllm::Data &qPacked, fastllm::Data &kPacked,
    fastllm::Data &vPacked, fastllm::Data &bPacked,
    fastllm::Data &gPacked);
bool FastllmCudaUnpackRaggedGdnPrefillChunksFloat16(
    const fastllm::Data &packed, const std::vector<int> &seqLens,
    int chunkSize, fastllm::Data &ragged);
bool FastllmCudaGetRaggedGdnMetadata(
    const std::vector<int> &seqLens, int chunkSize,
    FastllmCudaRaggedGdnMetadataView &view);

bool FastllmCudaConv2DFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, fastllm::Data &output);

bool FastllmCudaBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                                  int input0Spatial, int input1Spatial, int outputSpatial,
                                  int input0Stride, int input1Stride,
                                  int batch, int n, int m, int k, float alpha);
bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha);
bool FastllmCudaBatchMatMulTransBHeadMapped(
    const fastllm::Data &input0, const fastllm::Data &input1,
    fastllm::Data &output, int headGroup, float alpha = 1.0f);
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
bool FastllmCudaYarnRopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                 float ropeTheta, float factor, float attentionFactor,
                                 float correctionLow, float correctionHigh);
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
    float llama3HighFreqFactor = 32.0f,
    int useYarn = 0, float yarnFactor = 1.0f,
    float yarnAttentionFactor = 1.0f,
    float yarnCorrectionLow = 0.0f,
    float yarnCorrectionHigh = 1.0f);
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
bool FastllmCudaQwen35QGateKVPrefill(
    const fastllm::Data &qgatekv,
    const fastllm::Data &qNormWeight,
    const fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput, fastllm::Data &gateOutput,
    fastllm::Data &kOutput, fastllm::Data &vOutput,
    int qHeads, int kHeads, int headDim,
    int rotaryDim, int sectionT, int sectionH, int sectionW,
    float eps, float ropeTheta, float ropeScale);
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
bool FastllmCudaTopKTopPSamplingToDevice(
                                  float *logits, float *probs,
                                  float *temperatures, int *topKArr,
                                  float *topPArr,
                                  int *penaltyIds, float *penaltyFactors,
                                  int penaltyTokens,
                                  int *output, float *floatOutput,
                                  int batch, int vocabSize);
bool FastllmCudaGreedySampling(float *logits, int *output,
                               int batch, int vocabSize);
bool FastllmCudaGreedySamplingWithFloatOutput(float *logits, int *output,
                                              float *floatOutput,
                                              int batch, int vocabSize);
bool FastllmCudaGreedySamplingWithScores(float *logits, int *output,
                                         float *scores, int batch,
                                         int vocabSize);
bool FastllmCudaGreedySamplingPackedCandidateWithIdOffset(
                                         float *logits,
                                         void *packedCandidates,
                                         int batch, int vocabSize,
                                         int idOffset);
bool FastllmCudaMergeShardedGreedyCandidates(
                                         const void *packedCandidates,
                                         int *output, float *floatOutput,
                                         int ranks, int batch);
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
bool FastllmCudaHalfPagedAttentionBatch(fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches, fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs, fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale, int attentionType, bool inited = false, bool sync = true, bool enableCudaGraph = false, int flashInferCudaGraph = -1, int windowLeft = -1);
bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo = false);
enum FastllmCudaLinearFp16Path {
    FASTLLM_CUDA_LINEAR_FP16_PATH_AUTO = 0,
    FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE = 1,
    FASTLLM_CUDA_LINEAR_FP16_PATH_CUBLAS = 2,
};
// AUTO keeps B1-B7 on the native kernels, uses the verified Qwen3.6 B8 GDN
// specialization for 5120x48 without bias, and retains cuBLAS otherwise.
// Exposed so correctness tests can verify the production dispatch policy.
FastllmCudaLinearFp16Path FastllmCudaResolveLinearFp16AutoPath(int n, int m, int k, bool addTo, bool hasBias);
// Explicit path selection is intended for correctness tests and benchmarks.
// Production callers should use AUTO through FastllmCudaHalfMatMulFloat16.
// NATIVE supports one to eight rows; CUBLAS bypasses AUTO dispatch.
bool FastllmCudaHalfMatMulFloat16WithPath(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo, bool allowRouterSpecialization, FastllmCudaLinearFp16Path path);
bool FastllmCudaHalfMatMulFloat16WithRouterSpecialization(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k, bool addTo, bool allowRouterSpecialization);
// Fuses the Qwen3.5 FP16 router and shared-gate projections for one to seven
// flattened decode rows. Larger batches keep the cuBLAS GEMM path.
bool FastllmCudaQwen35RouterSharedGateFloat16(const fastllm::Data &input, fastllm::Data &routerWeight, fastllm::Data &sharedGateWeight, fastllm::Data &routerOutput, fastllm::Data &sharedGateOutput, bool sigmoidSharedGate = false);
bool FastllmCudaHalfMatMulFloat16AddToNoBias(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulBFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMergeMOEInt8Batch1Indexed(const fastllm::Data &input,
                                              fastllm::Data &scratch,
                                              fastllm::Data &output,
                                              fastllm::Data **weights,
                                              int weightsBatch,
                                              const int32_t *indices,
                                              const float *scores,
                                              int topk);
bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatInt4Group32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMergeMOEInt4GroupBatch1Indexed(const fastllm::Data &input,
                                                   fastllm::Data &scratch,
                                                   fastllm::Data &output,
                                                   fastllm::Data **weights,
                                                   int weightsBatch,
                                                   const int32_t *indices,
                                                   const float *scores,
                                                   int topk);
bool FastllmCudaHalfMergeMOEInt4GroupSmallBatchIndexed(const fastllm::Data &input,
                                                       fastllm::Data &scratch,
                                                       fastllm::Data &output,
                                                       fastllm::Data **weights,
                                                       int weightsBatch,
                                                       const int32_t *indices,
                                                       const float *scores,
                                                       int batch,
                                                       int topk);
bool FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &activation, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int batch, int topk);
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
bool FastllmCudaHalfMergeMOENVFP4Batch1IndexedSharedFP8(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                        fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                        const float *scores, float sharedScale,
                                                        int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                   fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                   const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaHalfMergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                fastllm::Data **weights, int weightsBatch,
                                                const int *routeRows, const float *routeScales,
                                                const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaHalfMatMulFloatFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaHalfMatMulFloatFP8E4M3PerChannel(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
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
bool FastllmCudaBFloat16MatMulInt4Group32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
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
bool FastllmCudaHalfMergeMOENVFP4Batch1ExpertParallel(const fastllm::Data &input, fastllm::Data &w1,
                                                      fastllm::Data &output, fastllm::Data **weights,
                                                      int weightsBatch, const int32_t *globalIndices,
                                                      const float *scores, int topk,
                                                      int ownerRank, int ownerCount);
bool FastllmCudaBFloat16MergeMOENVFP4Batch1ExpertParallel(const fastllm::Data &input, fastllm::Data &w1,
                                                           fastllm::Data &output, fastllm::Data **weights,
                                                           int weightsBatch, const int32_t *globalIndices,
                                                           const float *scores, int topk,
                                                           int ownerRank, int ownerCount);
bool FastllmCudaHalfMergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount);
bool FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount);
void FastllmCudaReleaseMergeMOEVllmMarlinCache(const fastllm::Data *layerKey);
#ifdef FASTLLM_ENABLE_DSV4_MOE_DEEPGEMM_SM120
bool FastllmCudaBFloat16MergeMOEDeepGemmSm120ExpertParallel(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores,
        int topk, int ownerRank, int ownerCount, float swigluLimit);
bool FastllmCudaBFloat16MergeMOEDeepGemmSm120TensorParallel(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores, int topk,
        float swigluLimit);
void FastllmCudaReleaseMergeMOEDeepGemmSm120Cache(
        const fastllm::Data *layerKey);
#endif
bool FastllmCudaBFloat16MergeMOENVFP4Batch1IndexedSharedFP8(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                            fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                            const float *scores, float sharedScale,
                                                            int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
                                                       fastllm::Data **weights, int weightsBatch, const int32_t *indices,
                                                       const float *scores, int batch, int topk, int hidden, int inter);
bool FastllmCudaBFloat16MergeMOENVFP4GroupedIndexed(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2, fastllm::Data &output,
                                                    fastllm::Data **weights, int weightsBatch,
                                                    const int *routeRows, const float *routeScales,
                                                    const int *routePositions, const int *expertStarts, const int *expertCounts,
                                                    int batch, int topk, int totalTasks, int maxExpertTasks, int hidden, int inter);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3PerChannel(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128Swiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaBFloat16MatMulFP8E4M3Block128AddTo(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float alpha, bool overwrite, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3PerChannel(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128Add(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNorm(const fastllm::Data &input, fastllm::Data &normWeight, float eps, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormMaterialize(
        const fastllm::Data &input, fastllm::Data &normWeight, float eps,
        fastllm::Data &normOutput,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGate(
        const fastllm::Data &headMajorInput, fastllm::Data &normWeight,
        const fastllm::Data &combinedGateInput,
        int batch, int seqLen, int gateOffset, int gateHeads, float eps,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateAdd(
        const fastllm::Data &headMajorInput, fastllm::Data &normWeight,
        const fastllm::Data &combinedGateInput,
        int batch, int seqLen, int gateOffset, int gateHeads, float eps,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluAdd(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
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

bool FastllmCudaTritonDeepSeekV4WoA(
    const char *cubinPath, const char *kernelName, int numWarps, int shared,
    int blockTokens, int blockOut, int blockHidden,
    const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output,
    int numTokens, int groups, int outRank, int hiddenSize);

bool FastllmCudaTritonDeepSeekV4SparseAttentionDecodeGraph(
    const char *splitCubinPath, const char *splitKernelName,
    int splitNumWarps, int splitShared,
    const char *mergeCubinPath, const char *mergeKernelName,
    int mergeNumWarps, int mergeShared,
    int compressedCapacity, int numSplits, int splitSize,
    int splitHeadBlock, int blockD, int mergeBlockD, const fastllm::Data &q,
    const fastllm::Data &windowKV, const fastllm::Data &compressedKV,
    const fastllm::Data &attnSink, int windowSize, int compressRatio,
    const int32_t *decodeMeta, float softmaxScale, float *output);

bool FastllmCudaTritonDeepSeekV4SqrtSoftplusRouter(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared, int numExperts, int topk, int blockN,
    const fastllm::Data &logits, const fastllm::Data &gateBias,
    float routeScale, fastllm::Data &expertIndex,
    fastllm::Data &expertScore);

bool FastllmCudaTritonChunkGatedDeltaRulePrefill(
    const char *hCubinPath, const char *hKernelName, int hNumWarps, int hShared,
    const char *oCubinPath, const char *oKernelName, int oNumWarps, int oShared,
    const char *oFusedDecayCubinPath,
    const char *oFusedDecayKernelName,
    int oFusedDecayNumWarps, int oFusedDecayShared,
    const char *hPrecomputedScaleCubinPath,
    const char *hPrecomputedScaleKernelName,
    int hPrecomputedScaleNumWarps, int hPrecomputedScaleShared,
    bool precomputeScale, bool fuseDecayMask,
    int chunks, int chunkSize, int kDim, int vDim,
    int hBlockV, int oBlockV,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &decayMask, fastllm::Data &kCumdecay,
    fastllm::Data &lastRecurrentState, fastllm::Data &coreAttnOut);

bool FastllmCudaTritonChunkGatedDeltaRuleVarlenPrefill(
    const char *hCubinPath, const char *hKernelName,
    int hNumWarps, int hShared,
    const char *oCubinPath, const char *oKernelName,
    int oNumWarps, int oShared,
    const char *oFusedDecayCubinPath,
    const char *oFusedDecayKernelName,
    int oFusedDecayNumWarps, int oFusedDecayShared,
    const char *hPrecomputedScaleCubinPath,
    const char *hPrecomputedScaleKernelName,
    int hPrecomputedScaleNumWarps, int hPrecomputedScaleShared,
    const char *oDirectQkCubinPath,
    const char *oDirectQkKernelName,
    int oDirectQkNumWarps, int oDirectQkShared,
    bool precomputeScale, bool fuseDecayMask, bool directOutputQk,
    int maxChunks, int chunkSize, int kDim, int vDim,
    int hBlockV, int oBlockV, const std::vector<int> &seqLens,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &decayMask, fastllm::Data &kCumdecay,
    fastllm::Data &lastRecurrentState,
    fastllm::Data &coreAttnOut);

bool FastllmCudaTritonChunkGdnKkt(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared,
    const fastllm::Data &kBeta, const fastllm::Data &k,
    int headGroup, fastllm::Data &output);

bool FastllmCudaTritonChunkGdnPostConv(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared, int blockT,
    const fastllm::Data &qInput, const fastllm::Data &kInput,
    const fastllm::Data &qkvInput, const fastllm::Data &gInput,
    const fastllm::Data &betaInput,
    int batch, int seqLen, int keyHeads, int valueHeads,
    int kDim, int vDim, float qScale,
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &beta,
    fastllm::Data &kBeta, fastllm::Data &vBeta);

bool FastllmCudaTritonChunkGdnRecompute(
    const char *cubinPath, const char *kernelName,
    int numWarps, int shared,
    bool precomputeScale, bool internalExp, int blockD,
    const fastllm::Data &attn, const fastllm::Data &vBeta,
    const fastllm::Data &kBeta, const fastllm::Data &gExp,
    const fastllm::Data &g,
    fastllm::Data &vOutput, fastllm::Data &kOutput);

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
bool FastllmChunkGatedDeltaRuleVarlenPrefillNative(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &g, fastllm::Data &attn,
    fastllm::Data &k_cumdecay, fastllm::Data &last_recurrent_state,
    const std::vector<int> &seqLens, fastllm::Data &core_attn_out,
    const fastllm::Data *decay_mask = nullptr,
    bool apply_decay_mask = false);

void FastllmCudaSetDevice(int gpu_id);
int FastllmCudaGetDevice();
int FastllmCudaRuntimeArch();
int GetPointerDeviceId(void *ptr);
bool FastllmCudaValidatePointerRange(const void *ptr, size_t bytes,
                                     int expectedDevice);
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
