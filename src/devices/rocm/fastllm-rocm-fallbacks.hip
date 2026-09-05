// CUDA-only optional accelerators. Returning false preserves native HIP,
// hipBLAS, dequantization, and RCCL fallbacks without changing any output.
#include "fastllm-hip.cuh"
#include "fastllm-multicuda.cuh"

#ifndef USE_ROCM
#error "This file is only part of the ROCm backend"
#endif

bool FastllmCudaBFloat16MatMulGGUFMMQ(const void *input, const void *weight,
                                     void *output, int ggmlType,
                                     int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaBFloat16MatMulGGUFMMVQ(const void *input, const void *weight,
                                      void *output, int ggmlType,
                                      int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return false;
}

bool FastllmCudaCustomAllReduce(void* data, void* dest, int count,
                                int dataType, int deviceId) {
    return false;
}

bool FastllmCudaCustomAllReduceAdd(void* data, void* dest, int count,
                                   int dataType, int deviceId) {
    return false;
}

bool FastllmCudaCustomAllReduceCanRun(int count, int dataType, int deviceId) {
    return false;
}

bool FastllmCudaCustomAllReduceEnabled() {
    return false;
}

bool FastllmCudaCustomAllReduceInit(const std::vector<int>& devices) {
    return false;
}

bool FastllmCudaCustomAllReducePairAdd(void* first, void* second, void* dest,
                                      int count, int dataType, int deviceId) {
    return false;
}

void FastllmCudaCustomAllReduceReset() {
    // This backend never allocates the corresponding CUDA-only cache.
}

bool FastllmCudaCutlassLinearFP8E4M3Block128Add(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGate(
        const fastllm::Data &headMajorInput, fastllm::Data &normWeight,
        const fastllm::Data &combinedGateInput,
        int batch, int seqLen, int gateOffset, int gateHeads, float eps,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateAdd(
        const fastllm::Data &headMajorInput, fastllm::Data &normWeight,
        const fastllm::Data &combinedGateInput,
        int batch, int seqLen, int gateOffset, int gateHeads, float eps,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNorm(const fastllm::Data &input, fastllm::Data &normWeight, float eps, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormMaterialize(
        const fastllm::Data &input, fastllm::Data &normWeight, float eps,
        fastllm::Data &normOutput,
        fastllm::Data &weight, const fastllm::Data &bias,
        fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluAdd(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluAvailable() {
    return false;
}

bool FastllmCudaFloatMatMulGGUFMMVQ(const void *input, const void *weight,
                                   void *output, int ggmlType,
                                   int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaGptqMarlinRepack(const uint32_t *b_q_weight, uint32_t *out,
                                 int size_k, int size_n) {
    return false;
}

bool FastllmCudaHalfGgufGateUpSiluMulMMVQ(
        const void *input, const void *gateWeight, const void *upWeight,
        void *output, int ggmlType, int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaHalfMatMulGGUFMMQ(const void *input, const void *weight,
                                 void *output, int ggmlType,
                                 int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaHalfMatMulGGUFMMVQ(const void *input, const void *weight,
                                  void *output, int ggmlType,
                                  int n, int m, int k, void *stream) {
    return false;
}

bool FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &activation, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int batch, int topk) {
    return false;
}

bool FastllmCudaHalfMergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return false;
}

bool FastllmCudaMarlinHalfInt4Gemm(const void *a, const uint32_t *b_q_weight,
                                   const void *b_scales, const uint32_t *b_zeros,
                                   void *c, int size_m, int size_n, int size_k,
                                   int group_size, int *workspace) {
    return false;
}

bool FastllmCudaMarlinNVFP4Supported(int size_n, int size_k) {
    return false;
}

bool FastllmCudaMergeMOENVFP4E4M3MarlinIndexed(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &activation, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int batch, int topk) {
    return false;
}

bool FastllmCudaNVFP4E4M3GroupedMoeSupported(int device) {
    return false;
}

void FastllmCudaReleaseMergeMOEVllmMarlinCache(const fastllm::Data *layerKey) {
    // This backend never allocates the corresponding CUDA-only cache.
}

int FastllmCudaRuntimeArch() {
    // No NVIDIA compute capability is available on a HIP device.
    return 0;
}

bool FastllmCudaTryMarlinHalfMatMulFloatFP8E4M3(const fastllm::Data &input,
                                                fastllm::Data &weight,
                                                const fastllm::Data &bias,
                                                 fastllm::Data &output,
                                                 int n, int m, int k) {
    return false;
}

bool FastllmCudaTryMarlinHalfMatMulFloatNVFP4Block16(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3Block128(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaCutlassLinearFP8E4M3PerChannel(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    return false;
}

bool FastllmCudaHasFp8MarlinLayout(const fastllm::Data &weight) {
    return false;
}

namespace fastllm {
namespace awq_sm70 {
// TurboMind SM70 kernels are NVIDIA-only. Preserve the unpacked weights so
// the native INT4/FP8/NVFP4 implementation can consume them.
bool Supported() { return false; }
bool Fp8Supported() { return false; }
bool Nvfp4Supported() { return false; }
void *Prepare(const uint16_t *, const half *, const half *,
              int, int, int, int, hipStream_t) { return nullptr; }
bool Gemm(void *, const half *, half *, int, hipStream_t) { return false; }
void Free(void *) {}
bool PrepareFp8InPlace(uint8_t *, const float *, half **,
                       int, int, int, int, hipStream_t) { return false; }
bool GemmFp8(const uint8_t *, const half *, const half *, half *,
             int, int, int, int, hipStream_t) { return false; }
bool PrepareNvfp4InPlace(uint8_t *, size_t, int, int, hipStream_t) { return false; }
bool GemmNvfp4(const uint8_t *, const half *, half *, int, int, int,
               hipStream_t) { return false; }
} // namespace awq_sm70
} // namespace fastllm

size_t FastllmCudaDFlashTopKScratchBytes(int rows) {
    return 0;
}

bool FastllmCudaDFlashTopK(
        const fastllm::Data &logits,
        fastllm::Data &packedCandidates,
        fastllm::Data &scratch,
        int topk, int globalIdOffset) {
    return false;
}

bool FastllmCudaDFlashRejectionSampling(
                                  float *logits,
                                  const float *temperatures,
                                  const int *topKArr,
                                  const float *topPArr,
                                  const int *draftTokenIds,
                                  const int *draftCandidateIds,
                                  const float *draftCandidateProbs,
                                  int *outputTokenIds,
                                  int *acceptedDraftTokens,
                                  int batch, int draftTokens,
                                  int selectorTopK, int vocabSize) {
    return false;
}
