#include "models/qwen3_5.h"
#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "utils/utils.h"

#include <cuda_runtime_api.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace fastllm;

#ifdef FASTLLM_TEST_WRAP_WORKSPACE
namespace {
size_t allocationLimit = 0;
bool allocationError = false;
int allocationCalls = 0;
}
extern "C" FastllmCudaTryMallocResult __real_FastllmCudaTryDirectMalloc(void **pointer, size_t bytes);
extern "C" FastllmCudaTryMallocResult __wrap_FastllmCudaTryDirectMalloc(void **pointer, size_t bytes) {
    if (allocationLimit || allocationError) {
        ++allocationCalls;
        *pointer = nullptr;
        if (allocationError) return FASTLLM_CUDA_TRY_MALLOC_ERROR;
        if (bytes > allocationLimit) return FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE;
    }
    return __real_FastllmCudaTryDirectMalloc(pointer, bytes);
}
#endif

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

void CheckBorrowedArena() {
    constexpr size_t bytes = 4 << 20;
    void *base = FastllmCudaDirectMalloc(bytes + 512);
    bool returned = false;
    auto arena = std::make_shared<CudaWorkspace>(0,
        std::vector<CudaWorkspace::Region>{{static_cast<char *>(base) + 256, bytes}},
        [&] { returned = true; });
    std::weak_ptr<CudaWorkspace> weak = arena;
    std::thread worker([&] {
        FastllmCudaSetDevice(0);
        Data scratch(INT8, {int(bytes)});
        scratch.cudaWorkspace = arena;
        scratch.ToDevice(CUDA, {0}, false);
        scratch.Allocate(false);
        Require(IsCudaWorkspacePointer(scratch.cudaData), "explicit tensor arena was ignored");
        FastllmCudaMemset0(scratch.cudaData, bytes);
        // Destruction publishes a GPU fence, not a host-side stream wait.
    });
    worker.join();
    void *again = AllocateCudaWorkspace(arena, bytes);
    Require(again == static_cast<char *>(base) + 256, "borrowed range not reused across streams");
    std::vector<unsigned char> values(bytes, 1);
    FastllmCudaCopyFromDeviceToHost(values.data(), again, bytes);
    Require(std::all_of(values.begin(), values.end(), [](unsigned char x) { return x == 0; }),
        "borrowed range reused before previous stream completed");
    arena.reset();
    Require(!weak.expired() && !returned, "owner returned with a live allocation");
    FastllmCudaFree(again);
    Require(weak.expired() && returned, "borrowed owner leaked after release");
    // Disjoint format partitions cannot satisfy one allocation across a gap.
    returned = false;
    arena = std::make_shared<CudaWorkspace>(0,
        std::vector<CudaWorkspace::Region>{
            {static_cast<char *>(base) + 256, bytes / 4},
            {static_cast<char *>(base) + bytes / 2 + 256, bytes / 4}},
        [&] { returned = true; });
    Require(!AllocateCudaWorkspace(arena, bytes / 2, false),
        "borrowed allocation crossed a partition gap");
    void *left = AllocateCudaWorkspace(arena, bytes / 4);
    void *right = AllocateCudaWorkspace(arena, bytes / 4);
    Require(left == static_cast<char *>(base) + 256 &&
            right == static_cast<char *>(base) + bytes / 2 + 256,
        "borrowed partition ranges overlap or moved");
    FastllmCudaFree(left); FastllmCudaFree(right);
    arena.reset();
    Require(returned, "partitioned workspace did not return its owner");
    // The allocator must not cudaFree a borrowed address or its parent.
    FastllmCudaMemset0(base, bytes + 512);
    FastllmCudaDirectFree(base);
}

// One release may be split between two subsequent allocating streams. Each
// must wait for the original writer, even after the first slice was reused.
void CheckBorrowedArenaSplitStreams() {
    constexpr size_t bytes = 512;
    void *base = FastllmCudaDirectMalloc(bytes);
    auto arena = std::make_shared<CudaWorkspace>(0,
        std::vector<CudaWorkspace::Region>{{base, bytes}}, [] {});
    void *original = arena->Allocate(bytes);
    void *writer = FastllmCudaStreamCreate();
    std::atomic<bool> release{false};
    Require(cudaLaunchHostFunc(static_cast<cudaStream_t>(writer), [](void *p) {
        auto &ready = *static_cast<std::atomic<bool> *>(p);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!ready.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
    }, &release) == cudaSuccess, "cannot delay workspace writer");
    Require(cudaMemsetAsync(original, 0x11, bytes, static_cast<cudaStream_t>(writer)) == cudaSuccess,
            "cannot queue workspace write");
    arena->FreeAfterStream(original, writer);
    void *left = arena->Allocate(bytes / 2), *right = nullptr;
    void *done = FastllmCudaEventCreate();
    std::thread consumer([&] {
        FastllmCudaSetDevice(0);
        right = arena->Allocate(bytes / 2);
        if (right) cudaMemsetAsync(right, 0x22, bytes / 2, cudaStreamPerThread);
        FastllmCudaEventRecordCurrentThread(done);
    });
    consumer.join();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    bool premature = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (cudaEventQuery(static_cast<cudaEvent_t>(done)) == cudaSuccess) {
            premature = true;
            break;
        }
        std::this_thread::yield();
    }
    release.store(true);
    FastllmCudaStreamSynchronize(writer);
    FastllmCudaEventSynchronize(done);
    FastllmCudaSyncCurrentThreadStream();
    std::vector<unsigned char> values(bytes);
    FastllmCudaCopyFromDeviceToHost(values.data(), base, bytes);
    arena->Free(left); arena->Free(right); arena.reset();
    FastllmCudaEventDestroy(done); FastllmCudaStreamDestroy(writer);
    FastllmCudaDirectFree(base);
    Require(!premature, "second allocating stream skipped the original release fence");
    Require(std::all_of(values.begin(), values.begin() + bytes / 2, [](unsigned char x) { return x == 0x11; }) &&
            std::all_of(values.begin() + bytes / 2, values.end(), [](unsigned char x) { return x == 0x22; }),
            "split workspace slices raced with their previous writer");
}

void CheckArenaFallback() {
    for (size_t minimum : {size_t(4097), std::numeric_limits<size_t>::max()}) {
        bool rejected = false;
        try { CudaWorkspace invalid(0, 4096, minimum); }
        catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "invalid arena minimum accepted");
    }
#ifdef FASTLLM_TEST_WRAP_WORKSPACE
    struct Restore {
        int device = FastllmCudaGetDevice();
        ~Restore() { allocationLimit = 0; allocationError = false; FastllmCudaSetDevice(device); }
    } restore;
    const int previous = FastllmCudaGetDeviceCount() > 1 ? 1 : 0;
    FastllmCudaSetDevice(previous);
    allocationLimit = 4096;
    {
        auto arena = std::make_shared<CudaWorkspace>(0, 16384, 2048);
        Require(arena->Capacity() >= 2048 && arena->Capacity() <= allocationLimit,
                "arena did not shrink within requested bounds");
        Require(FastllmCudaGetDevice() == previous, "arena allocation changed current device");
        void *pointer = AllocateCudaWorkspace(arena, 2048);
        Require(pointer && TryFreeCudaWorkspace(pointer), "shrunken arena is unusable");
    }
    allocationLimit = 1024;
    bool exhausted = false;
    try { CudaWorkspace arena(0, 16384, 2048); }
    catch (const std::runtime_error &) { exhausted = true; }
    Require(exhausted && FastllmCudaGetDevice() == previous && !FastllmCudaGetThreadError(),
            "arena exhaustion changed device or poisoned CUDA state");
    allocationError = true;
    allocationCalls = 0;
    bool failed = false;
    try { CudaWorkspace arena(0, 16384, 2048); }
    catch (const std::runtime_error &) { failed = true; }
    Require(failed && allocationCalls == 1 && FastllmCudaGetDevice() == previous,
            "non-capacity allocation error was retried or changed device");
#endif
}

void CheckConcurrentArena() {
    auto arena = std::make_shared<CudaWorkspace>(0, 2 << 20);
    std::array<std::vector<void *>, 8> blocks;
    auto parallel = [&](const auto &task) {
        std::array<std::exception_ptr, 8> errors;
        std::vector<std::thread> threads;
        for (int rank = 0; rank < 8; ++rank) threads.emplace_back([&, rank] {
            try { task(rank); } catch (...) { errors[rank] = std::current_exception(); }
        });
        for (auto &thread : threads) thread.join();
        for (auto error : errors) if (error) std::rethrow_exception(error);
    };
    parallel([&](int rank) {
        for (int i = 0; i < 1024; ++i) {
            auto *pointer = AllocateCudaWorkspace(arena, 256, false);
            Require(pointer && IsCudaWorkspacePointer(pointer), "concurrent arena allocation failed");
            blocks[rank].push_back(pointer);
        }
    });
    Require(arena->LiveBytes() == arena->Capacity(), "concurrent arena accounting mismatch");
    Require(!AllocateCudaWorkspace(arena, 256, false), "full arena allocation must fail");
    parallel([&](int rank) {
        for (auto *pointer : blocks[(rank + 1) % 8])
            Require(TryFreeCudaWorkspace(pointer), "cross-thread release lost arena ownership");
    });
    Require(arena->LiveBytes() == 0, "concurrent arena release leaked storage");
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
        CheckBorrowedArena();
        CheckBorrowedArenaSplitStreams();
        CheckArenaFallback();
        CheckConcurrentArena();
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
