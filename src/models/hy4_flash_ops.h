#pragma once
#include "models/hy4_flash.h"
#include <array>
#include <deque>

namespace fastllm {
class DiskMoeV41Graph;
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
// Updated once per step and GPU, before launching a reusable decode graph.
struct StepInput {
    int token, previous, older, position;
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
    std::map<std::pair<int, int>, std::shared_ptr<void>> engramInputs;
    explicit Scratch(int device) : device(device) {}
    ~Scratch();
    float *Get(const std::string &name, size_t count);
};
using Weights = Hy4FlashModel::Weights;
struct GpuGraphs {
    struct Replay {
        std::shared_ptr<void> executable;
        std::vector<std::shared_ptr<Data>> outputs;
    };
    std::map<std::vector<uintptr_t>, Replay> replays;
    size_t launches = 0;
    size_t warmedRows = 0;
    bool warmedSingle = false;
    bool peerChecked = false, peerReady = false;
};
struct GraphCapture {
    explicit GraphCapture(GpuGraphs *graphs) : graphs(graphs) {}
    std::shared_ptr<DiskMoeV41Graph> disk;
    std::vector<uintptr_t> key;
    std::vector<std::shared_ptr<Data>> outputs;
    ~GraphCapture();
    void Start(int device);
    void Close();
    void Finish(int kind, int rows);
    void AddNative(void *child, void *handle);
    void FinishHybrid(int kind, int rows);
    std::vector<void *> handles;
    std::vector<const void *> hybridChildren;
    GpuGraphs *graphs;
    int activeDevice = -1;
    std::vector<void *> children;
};
const StepInput *UploadStep(Scratch &, const std::vector<StepInput> &);
bool ReplayGpu(GpuGraphs &, const std::vector<uintptr_t> &, int kind, int rows);
bool ReplayHybrid(GpuGraphs &, const std::vector<uintptr_t> &, int kind, int rows);
void AdvanceKV(Data &, int end);
bool EnableGraphPeers(const std::vector<int> &devices);
void CopyStageCaptured(float *dst, int dstDevice, const float *src, int srcDevice, size_t count);
void ProjectHead(Weights &, Scratch &, float *streams, float *logits, Data *hidden, int pos, int width);
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
void DraftLayer(Weights &, Scratch &, int draftLayer, int base, int width, DraftMemory &, float *block);
std::vector<int> DraftReadout(Weights &, Scratch &, int anchor, int width, float *block);
void SaveDraftHidden(Data &ring, int device, int position, int width, const float *hidden,
                     const StepInput *step = nullptr);
void Embed(Weights &, Scratch &, int token, float *streams);
void HyperPre(Weights &, Scratch &, const std::string &prefix, float *streams, float *input, float *post,
              bool finalHead = false);
void HyperPost(float *streams, const float *branch, const float *post, int width);
void Norm(Weights &, Scratch &, const std::string &name, const float *input, float *output, int width);
void Engram(Weights &, Scratch &, int layer, const std::array<int, 3> &ids, float *streams,
            const uint16_t *hostRows = nullptr, bool asyncCopy = false);
void PrepareEngramInput(Weights &, Scratch &, int layer, int row, const std::array<int, 3> &ids);
void RunPreparedEngram(Weights &, Scratch &, int layer, int row, const std::array<int, 3> &ids,
                       float *streams);
struct AttentionRow {
    int position;
    Data *key, *value;
    int draftWidth;
    DraftMemory *draft;
    int draftLayer;
    const StepInput *step;
};
void AttentionBatch(Weights &, Scratch &, int layer, const std::vector<AttentionRow> &, const float *inputs,
                    float *outputs);
void Attention(Weights &, Scratch &, int layer, int position, const float *input, float *output,
               Data &keyCache, Data &valueCache, int draftWidth = 0, DraftMemory *draft = nullptr,
               int draftLayer = -1);
void DenseMlpBatch(Weights &, Scratch &, const std::string &, const float *, float *, int rows);
void RouteBatch(Weights &, Scratch &, int layer, const float *, int *, float *, float *, int rows);
void CombineExpertsBatch(Weights &, Scratch &, int layer, const float *, const float *, float *, int rows);
void DenseMlp(Weights &, Scratch &, const std::string &prefix, const float *input, float *output);
void Route(Weights &, Scratch &, int layer, const float *input, int *ids, float *scores, float *latent);
void ResidentExperts(Weights &, Scratch &, int layer, const float *latent, const int *ids,
                     const float *scores, float *result, int rows);
void CombineExperts(Weights &, Scratch &, int layer, const float *input, const float *routed, float *out);
std::vector<float> Head(Weights &, Scratch &, float *streams, Data *draftHidden = nullptr, int position = 0,
                        int draftWidth = 0);
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
