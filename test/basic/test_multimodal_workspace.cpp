#include "models/qwen3_5.h"
#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "utils/utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void CheckArena() {
    constexpr size_t capacity = 1 << 20;
    auto arena = std::make_shared<CudaWorkspace>(0, capacity);
    std::weak_ptr<CudaWorkspace> weak = arena;
    void *retained;
    {
        CudaWorkspaceScope scope(arena);
        void *a = FastllmCudaMalloc(capacity / 4);
        void *b = FastllmCudaMalloc(capacity / 4);
        void *c = FastllmCudaMalloc(capacity / 2);
        Require(a && b && c && arena->LiveBytes() == capacity, "arena capacity accounting");
        void *overflow = nullptr;
        Require(FastllmCudaTryMalloc(&overflow, 256) == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE,
                "exhausted workspace must not allocate outside its reservation");
        Require(!overflow && !FastllmCudaGetThreadError(), "optional allocation poisoned CUDA error state");
        FastllmCudaFree(b);
        FastllmCudaForceFree(a);
        void *ab = FastllmCudaMalloc(capacity / 2);
        Require(ab == a, "adjacent free ranges did not coalesce");
        FastllmCudaDirectFree(c);
        FastllmCudaFree(ab);
        {
            auto inner = std::make_shared<CudaWorkspace>(0, 4096);
            CudaWorkspaceScope nested(inner);
            void *p = FastllmCudaMalloc(4096);
            Require(inner->LiveBytes() == 4096 && arena->LiveBytes() == 0, "nested scope selection");
            FastllmCudaFree(p);
        }
        if (FastllmCudaGetDeviceCount() > 1) {
            FastllmCudaSetDevice(1);
            void *otherDevice = FastllmCudaMalloc(4096);
            Require(otherDevice && !IsCudaWorkspacePointer(otherDevice), "workspace crossed CUDA devices");
            FastllmCudaFree(otherDevice);
            FastllmCudaSetDevice(0);
        }
        retained = FastllmCudaMalloc(capacity);
        std::vector<float> source(1024, 3.25f), result(1024);
        FastllmCudaCopyFromHostToDevice(retained, source.data(), source.size() * sizeof(float));
        FastllmCudaCopyFromDeviceToHost(result.data(), retained, result.size() * sizeof(float));
        Require(source == result, "workspace storage data roundtrip");
        arena.reset();
    }
    Require(!weak.expired(), "outstanding tensor did not retain its workspace");
    Require(FastllmCudaFreeAfterCurrentThreadStream(retained), "deferred release missed workspace");
    Require(weak.expired(), "workspace leaked after last allocation was released");
    void *normal = FastllmCudaMalloc(4096);
    Require(normal && !IsCudaWorkspacePointer(normal), "scope leaked into ordinary CUDA allocations");
    FastllmCudaFree(normal);
}

class TestModel : public Qwen3_5Model {
public:
    using Qwen3_5Model::BuildMultimodalTextEmbeddings;
    using Qwen3_5Model::SplitMultimodalTextEmbeddings;
    using Qwen3_5Model::MergeMultimodalFeaturesIntoText;
};

void CheckCpuStaging(bool cudaEmbedding, DataType dtype, bool replicated = false) {
    constexpr int width = 64, vocab = 32, tokens = 10001;
    TestModel model;
    model.dataType = dtype;
    model.SetChunkedPrefillSize(512);
    std::vector<float> values(vocab * width), ids(tokens), types(tokens, 1.0f);
    for (int i = 0; i < vocab * width; ++i) values[i] = (i % 128 - 64) / 64.0f;
    for (int i = 0; i < tokens; ++i) ids[i] = i % vocab;
    types[0] = types[tokens - 1] = 0;
    types[1] = 2;
    model.weight.AddEmptyWeight(model.language_prefix + "embed_tokens.weight", {vocab, width}, FLOAT16);
    auto &weight = model.weight[model.language_prefix + "embed_tokens.weight"];
    weight.CopyFrom(Data(FLOAT16, {vocab, width}, values));
    SetCudaEmbedding(cudaEmbedding);
    if (replicated) {
        auto *replica = new Data(weight);
        replica->ToDevice(DataDevice::CUDA, std::vector<int>{0});
        weight.FreeSpace();
        weight.multiDeviceData = true;
        weight.multiDeviceDatas[0] = replica;
        weight.tpLayout = TP_LAYOUT_REPLICATED;
        weight.tpGlobalDims = {vocab, width};
    } else if (cudaEmbedding) {
        weight.ToDevice(DataDevice::CUDA, std::vector<int>{0});
    }
    Executor executor;
    executor.SetFirstDevice("cuda:0");
    struct RestoreExecutor {
        void *previous = GetExecutor();
        ~RestoreExecutor() { SetCurrentThreadExecutor(previous); }
    } restore;
    SetCurrentThreadExecutor(&executor);
    auto arena = std::make_shared<CudaWorkspace>(0, 1 << 20);
    CudaWorkspaceScope scope(arena);
    Data input(FLOAT32, {1, tokens}, ids), hidden;
    model.BuildMultimodalTextEmbeddings(input, hidden);
    Require(hidden.dataDevice == DataDevice::CPU && hidden.cudaData == nullptr,
            "full prompt must stay on CPU");
    Require(arena->LiveBytes() == 0, "embedding chunk storage leaked");
    if (cudaEmbedding) Require(arena->PeakBytes() > 0, "CUDA embedding was not exercised");
    Require(arena->PeakBytes() < 512 * 1024, "embedding did not stay bounded by chunk size");
    Data expected(FLOAT32, {1, tokens, width});
    expected.Allocate();
    auto *reference = reinterpret_cast<float*>(expected.cpuData);
    for (int row = 0; row < tokens; ++row) {
        for (int col = 0; col < width; ++col) reference[row * width + col] = values[(int)ids[row] * width + col];
    }
    Data image(FLOAT32, {1, tokens - 3, width});
    image.Allocate();
    std::fill_n(reinterpret_cast<float*>(image.cpuData), image.Count(0), 0.25f);
    Data video(FLOAT32, {1, width}, std::vector<float>(width, -0.5f));
    Data mm(FLOAT16, {1, tokens}, types);
    size_t embeddingPeak = arena->PeakBytes();
    model.MergeMultimodalFeaturesIntoText(mm, &image, &video, hidden);
    Require(arena->LiveBytes() == 0 && arena->PeakBytes() == embeddingPeak,
            "feature conversion/merge unexpectedly used CUDA");
    // Chunk splitting must also avoid uploading the complete CPU prompt.
    Data chunk;
    model.SplitMultimodalTextEmbeddings(hidden, tokens - 3, tokens, chunk);
    Require(chunk.dataDevice == DataDevice::CPU && !chunk.lockInCPU,
            "prefill slice must start on CPU and remain movable to its compute device");
    Require(arena->PeakBytes() == embeddingPeak, "Split uploaded the complete prompt");
    ToDataTypeForceCPU(hidden, FLOAT32);
    auto *actual = reinterpret_cast<float*>(hidden.cpuData);
    for (int row = 0; row < tokens; ++row) {
        for (int col = 0; col < width; ++col) {
            float wanted = types[row] == 0 ? reference[row * width + col] : types[row] == 1 ? 0.25f : -0.5f;
            Require(actual[row * width + col] == wanted, "multimodal embedding/merge value mismatch");
        }
    }
    SetCudaEmbedding(false);
}
}

int main() {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    FastllmCudaSetDevice(0);
    FastllmCudaClearThreadError();
    SetThreads(4);
    try {
        CheckArena();
        for (bool cudaEmbedding : {false, true}) {
            for (DataType dtype : {FLOAT32, FLOAT16, BFLOAT16}) CheckCpuStaging(cudaEmbedding, dtype);
        }
        CheckCpuStaging(true, FLOAT16, true);
        Require(!FastllmCudaGetThreadError(), "unexpected CUDA thread error");
        std::cout << "Multimodal workspace and CPU staging: PASS\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
