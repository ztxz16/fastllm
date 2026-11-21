#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Native, opt-in SM100/SM103 routed experts for DeepSeek V4.1 Flash.
// All eight ranks must call Run in identical order, including empty shards.
// Workspace and packed layer storage must outlive every captured CUDA graph.
struct FastllmDsv41MegaLayer;
constexpr int FastllmDsv41MegaMaxRows = 256;
bool FastllmDsv41MegaCanPrepare(const int *devices, int count);
size_t FastllmDsv41MegaWorkspaceBytes();
void FastllmDsv41MegaInitWorkspace(void *workspace, const int *devices, int rank);
FastllmDsv41MegaLayer *FastllmDsv41MegaCreate(
    const void *const *gate, const void *const *down,
    void *const *workspaces, int rank, const void *cacheKey = nullptr);
bool FastllmDsv41MegaGraphReady(const void *cacheKey, int rows);
void FastllmDsv41MegaDestroy(FastllmDsv41MegaLayer *layer);
void FastllmDsv41MegaRun(FastllmDsv41MegaLayer *layer, const void *inputBf16,
    const int32_t *globalIndices, const float *scores, float *output,
    int rows, void *stream, bool ownerOutput = false);

// Shared weight bytes are device resident; block-32 UE8M0 scales are host
// resident and consumed synchronously during cold model preparation.
void FastllmDsv41MegaEnableShared(FastllmDsv41MegaLayer *layer,
    const void *gate, const uint8_t *gateScales, const void *down, const uint8_t *downScales);
void FastllmDsv41MegaRunShared(FastllmDsv41MegaLayer *layer, const void *inputBf16,
    const int32_t *globalIndices, const float *scores, void *localBf16Rows, int rows, void *stream);

// Each attention owner supplies its own independent token rows. Empty ranks
// launch the same expert collective and need no operand buffers.
void FastllmDsv41MegaRunLocal(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, void *output, int localRows, void *stream);
const void *FastllmDsv41OwnerPostKernel();
const void *FastllmDsv41PrepareKernel();
void FastllmDsv41MegaRunLocalPost(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int localRows, void *stream);
void FastllmDsv41MegaRunLocalPostAfterMix(FastllmDsv41MegaLayer *layer, const void *input,
    const int32_t *ids, const float *scores, const void *residual, const float *post,
    const float *comb, void *hidden, int localRows, void *stream, void *mixDone);
namespace fastllm {
class Data;
std::vector<FastllmDsv41MegaLayer *> MultiCudaDsv41SharedLayers(const Data *key,
    const std::vector<int> &devices);
bool MultiCudaDsv41CombinedExperts(Data &input, Data &ids, Data &scores,
    std::vector<Data *> &routed, Data &gate, Data &down, Data &output,
    const std::vector<int> &devices, bool runDecode);
}
