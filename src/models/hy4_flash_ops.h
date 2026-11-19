#pragma once
#include "models/hy4_flash.h"
#include <array>
#include <deque>
#ifdef FASTLLM_ENABLE_HY4_CUDA
#include "devices/cuda/fastllm-mla-plan.h"
#endif

namespace fastllm {
// A projection can consume an ordinary matrix or the contiguous matrix of a
// selected expert. Scale blocks use the checkpoint's original storage type.
struct Hy4View {
    const void *data = nullptr, *scale = nullptr;
    int type = 0, scaleType = 0, rows = 0, cols = 0;
};
struct Hy4Source {
    std::string file, dtype;
    uint64_t offset = 0, bytes = 0;
    std::vector<int> dims;
};
struct Hy4FlashModel::Weights {
    std::map<std::string, Hy4Source> meta;
    std::map<std::string, std::vector<uint8_t>> host;
    struct Allocation {
        int device;
        void *ptr;
        ~Allocation();
    };
    std::map<std::pair<int, std::string>, std::unique_ptr<Allocation>> gpu;
    json11::Json config;
    const void *Host(const std::string &name);
    const void *Device(const std::string &name, int device);
    Hy4View View(const std::string &name, int device);
};
#ifdef FASTLLM_ENABLE_HY4_CUDA
namespace hy4 {
inline constexpr int KVPageTokens = 128;
// Updated once per step and GPU, before launching a reusable decode graph.
struct StepInput {
    int token, previous, older, position;
};
class KVPagePool;
struct KVPools {
    std::map<std::pair<int, int>, std::shared_ptr<KVPagePool>> layouts;
};
struct Scratch {
    const StepInput *step = nullptr;
    size_t generation = 0;
    int device;
    // Each intermediate is allocated at its maximum live size for this step.
    // std::map nodes keep entries stable while independent ops request space.
    struct Buffer {
        float *ptr = nullptr;
        size_t count = 0;
    };
    std::map<std::string, Buffer> buffers;
    KVPools kvPools;
    const bool mlaSupported;
    std::map<std::pair<int, int>, std::shared_ptr<FastllmCudaMLAPlan>> mlaPlans; // (heads, query rows)

    std::map<std::pair<int, int>, std::shared_ptr<void>> engramInputs;
    explicit Scratch(int device) : device(device), mlaSupported(FastllmCudaMLAPagedPlanSupported(device)) {}
    ~Scratch();
    float *Get(const std::string &name, size_t count);
};
using Weights = Hy4FlashModel::Weights;
const StepInput *UploadStep(Scratch &, const std::vector<StepInput> &);
void CopyStageCaptured(float *dst, const float *src, size_t count);
void ProjectHead(Weights &, Scratch &, float *streams, float *logits, Data *hidden, int pos, int width,
                 int rows = 1);
struct DraftMemory {
    Data keys, values, recalled;
};
struct DraftRequest {
    std::array<DraftMemory, 5> providers;
    Data hidden;
    std::deque<int> pending;
    int expected = -1;
};
void DraftEmbed(Weights &, Scratch &, int anchor, int width, int mask, const float *targetHidden,
                float *block);
struct KVPage;
void DraftLayer(Weights &, Scratch &, int draftLayer, int base, int width, DraftMemory &, float *block,
                const KVPage *const *pages);
std::vector<int> DraftReadout(Weights &, Scratch &, int anchor, int width, float *block);
void SaveDraftHidden(Data &ring, int device, int position, int width, const float *hidden,
                     const StepInput *step = nullptr);
void Embed(Weights &, Scratch &, int token, float *streams);
void HyperPre(Weights &, Scratch &, const std::string &prefix, float *streams, float *input, float *post,
              bool finalHead = false);
void HyperPost(float *streams, const float *branch, const float *post, int width);
void Norm(Weights &, Scratch &, const std::string &name, const float *input, float *output, int width);
void NormBatch(Weights &, Scratch &, const std::string &name, float *data, int rows, int width, int stride = 0);
void Engram(Weights &, Scratch &, int layer, const std::array<int, 3> &ids, float *streams,
            const uint16_t *hostRows = nullptr, bool asyncCopy = false);
void PrepareEngramInput(Weights &, Scratch &, int layer, int row, const std::array<int, 3> &ids);
void RunPreparedEngram(Weights &, Scratch &, int layer, int row, const std::array<int, 3> &ids,
                       float *streams);
struct KVPage {
    void *key, *value;
};
struct AttentionCache {
    const KVPage *target, *draft;
};
void TruncateKV(Data &key, Data &value, int end);
const KVPage *ReservePages(KVPools &, int device, Data &key, Data &value, int end, int retained, int keyWidth,
                           int valueWidth);
const AttentionCache *PrepareAttentionCaches(Weights &w, Scratch &s, int firstLayer, int endLayer,
                                             int blockCount, const std::vector<int> &lengths,
                                             const std::vector<std::pair<Data *, Data *>> &past,
                                             const std::vector<int> &providers,
                                             const std::vector<DraftRequest *> &states, int draftWidth);
const AttentionCache *UploadAttentionCaches(Scratch &, const std::vector<AttentionCache> &);
struct AttentionRow {
    int position;
    int draftWidth;
    DraftMemory *draft;
    int draftLayer;
    const StepInput *step;
    const AttentionCache *cache;
};
void PagedAttentionBatch(Weights &, Scratch &, int layer, const std::vector<AttentionRow> &, const float *,
                         float *);
void DenseMlpBatch(Weights &, Scratch &, const std::string &, const float *, float *, int rows);
void RouteBatch(Weights &, Scratch &, int layer, const float *, int *, float *, float *, int rows);
void CombineExpertsBatch(Weights &, Scratch &, int layer, const float *, const float *, float *, int rows);
void ResidentExperts(Weights &, Scratch &, int layer, const float *latent, const int *ids,
                     const float *scores, float *result, int rows);
void *Allocate(int device, size_t bytes);
void Free(int device, void *ptr);
void Select(int device);
void CopyToHost(void *dst, const void *src, size_t bytes);
void CopyToGPU(void *dst, const void *src, size_t bytes);
void CopyStage(float *dst, int dstDevice, const float *src, int srcDevice, size_t count);
std::array<int64_t, 16> EngramRows(Weights &, int layer, const std::array<int, 3> &ids);
} // namespace hy4
#endif
} // namespace fastllm
