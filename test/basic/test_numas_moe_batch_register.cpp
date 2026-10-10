#include "fastllm.h"
#include "gguf.h"
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include "utils/hostbufferreclaim.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <cstdio>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
using namespace fastllm;
namespace fastllm { void RegisterNumas(Data *weight, std::string type); }
static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
// Exercise page-aligned, unaligned and sub-page ranges while live canaries
// occupy both neighbours. A locked mapping must take the fallback path.
static void TestSourcePageReclaim() {
#if defined(__linux__) && defined(__GLIBC__)
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t total = 8 * page;
    auto *mapping = static_cast<uint8_t*>(mmap(nullptr, total, PROT_READ | PROT_WRITE,
                                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    Check(mapping != MAP_FAILED, "reclaim test mmap failed");
    for (size_t offset : {size_t(0), size_t(1), size_t(63), page - 1}) {
        for (size_t bytes : {size_t(1), page - 1, page, 3 * page, 3 * page + 17}) {
            std::memset(mapping, 0x5a, total);
            uint8_t *buffer = mapping + page + offset;
            const size_t discarded = detail::DiscardUnusedHostBufferPages(buffer, bytes);
            size_t zeros = 0;
            for (size_t i = 0; i < total; ++i) {
                if (mapping + i < buffer || mapping + i >= buffer + bytes)
                    Check(mapping[i] == 0x5a, "reclaim changed a live neighbour");
                else {
                    Check(mapping[i] == 0 || mapping[i] == 0x5a, "unexpected reclaimed byte");
                    zeros += mapping[i] == 0;
                }
            }
            Check(buffer[0] == 0x5a, "reclaim destroyed allocation prefix");
            Check(zeros == discarded, "reclaimed byte accounting mismatch");
            Check(discarded % page == 0, "reclaim must operate on whole pages");
            if (bytes >= 3 * page) Check(discarded >= 2 * page, "large buffer was not reclaimed");
        }
    }
    // Locked anonymous pages reject MADV_DONTNEED. The caller must be able to
    // fall back to ordinary deletion/trim without losing or changing data.
    std::memset(mapping, 0x5a, total);
    if (mlock(mapping, total) == 0) {
        Check(detail::DiscardUnusedHostBufferPages(mapping, total) == 0, "locked page reclaim should fail");
        for (size_t i = 0; i < total; ++i) Check(mapping[i] == 0x5a, "failed reclaim changed data");
        Check(munlock(mapping, total) == 0, "reclaim fallback test munlock failed");
    } else {
        std::puts("SKIP: locked-page reclaim check (mlock unavailable)");
    }
    Check(munmap(mapping, total) == 0, "reclaim test munmap failed");
#endif
    Check(detail::DiscardUnusedHostBufferPages(nullptr, 0) == 0, "empty reclaim failed");
}
static std::unique_ptr<Data> MakeWeight(int seed) {
    const ggml_type types[] = {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0};
    // Cover both gate/up and down layouts for every quantization type.
    const ggml_type type = types[(seed / 2) % 6];
    auto weight = std::make_unique<Data>(DATA_GGUF_FORMAT, type, std::vector<int>{64, 256});
    weight->Allocate(false);
    for (size_t i = 0; i < weight->GetBytes(); ++i) weight->cpuData[i] = (i * 17 + seed * 29) & 255;
    return weight;
}
static void TestChunkBoundaries() {
    const int nodes = GetNumasMoeShardCount();
    std::vector<std::unique_ptr<Data>> reference, batched;
    std::vector<std::pair<Data*, std::string>> pending;
    // Three 102 MiB weights must span two 256 MiB arenas.
    for (int seed = 0; seed < 3; ++seed) {
        auto make = [&]() {
            auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, GGML_TYPE_Q8_0,
                                            std::vector<int>{32768, 3072});
            w->Allocate(false);
            for (size_t i = 0; i < w->GetBytes(); ++i) w->cpuData[i] = (i * 17 + seed * 29) & 255;
            return w;
        };
        reference.push_back(make()); batched.push_back(make());
        RegisterNumas(reference.back().get(), "linearColumn");
        pending.push_back({batched.back().get(), "linearColumn"});
    }
    RegisterNumasMoeWeightBatch(pending);
    for (int i = 0; i < 3; ++i) {
        Check(batched[i]->GetBytes() == reference[i]->GetBytes(), "chunk changed packed byte count");
        for (int node = 0; node < nodes; ++node)
            Check(std::memcmp(batched[i]->numasData[node], reference[i]->numasData[node],
                              reference[i]->GetBytes() / nodes) == 0, "chunk boundary corrupted weight bytes");
    }
    auto *firstArena = batched[0]->numasData[0];
    auto *secondArena = batched[2]->numasData[0];
    batched[0].reset();
#ifdef USE_CUDA
    unsigned int flags = 0;
    if (NumasMoeWeightsArePinned())
        Check(cudaHostGetFlags(&flags, firstArena) == cudaSuccess, "first arena lost its remaining owner");
#endif
    batched[1].reset();
#ifdef USE_CUDA
    if (NumasMoeWeightsArePinned()) {
        Check(cudaHostGetFlags(&flags, firstArena) != cudaSuccess, "completed first arena was not retired");
        cudaGetLastError();
        Check(cudaHostGetFlags(&flags, secondArena) == cudaSuccess, "retiring first arena invalidated second arena");
    }
#endif
    for (auto &weight : reference) {
        for (auto *ptr : weight->numasData) {
            if (weight->isPinned) free_pinned_numa(ptr, weight->GetBytes() / nodes + 63);
            else free_aligned_numa(ptr, weight->GetBytes() / nodes + 63);
        }
        weight->numasData.clear();
    }
}
int main(int argc, char **argv) {
    try {
        TestSourcePageReclaim();
#ifdef USE_CUDA
        int devices = 0;
        if (NumasMoeWeightsArePinned() &&
            (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE");
            return 77;
        }
#endif
        if (argc == 2 && std::string(argv[1]) == "--chunk-boundary") {
            SetThreads(4);
            TestChunkBoundaries();
            std::puts("PASS: bounded NUMA arenas across chunk boundaries");
            return 0;
        }
        SetThreads(4);
        const int nodes = GetNumasMoeShardCount();
        std::vector<std::unique_ptr<Data>> reference, batched;
        std::vector<std::pair<Data*, std::string>> pending;
        for (int i = 0; i < 24; ++i) {
            reference.push_back(MakeWeight(i)); batched.push_back(MakeWeight(i));
            const std::string type = i % 2 ? "linearColumn" : "linearSwiglu";
            RegisterNumas(reference.back().get(), type);
            pending.push_back({batched.back().get(), type});
        }
        // Duplicate and null entries must not repack or register a tensor twice.
        pending.push_back(pending[0]); pending.push_back({nullptr, "linearColumn"});
        size_t completed = 0;
        RegisterNumasMoeWeightBatch(pending, [&](size_t count) {
            Check(count >= completed && count <= pending.size(), "non-monotonic batch progress");
            completed = count;
        });
        Check(completed == pending.size(), "incomplete batch progress");
        for (int i = 0; i < 24; ++i) {
            auto &a = *reference[i]; auto &b = *batched[i];
            Check(b.cpuData == nullptr && b.numasData.size() == (size_t)nodes, "invalid batch storage");
            Check(a.ggmlType == b.ggmlType && a.GetBytes() == b.GetBytes(), "batch changed packed format");
            const size_t shardBytes = a.GetBytes() / nodes;
            for (int node = 0; node < nodes; ++node) {
                Check(std::memcmp(a.numasData[node], b.numasData[node], shardBytes) == 0,
                      "batch changed interleaved/repacked expert bytes");
#ifdef USE_CUDA
                if (NumasMoeWeightsArePinned()) {
                    Check(b.isPinned, "batch unexpectedly lost pinned status");
                    cudaPointerAttributes attrs{};
                    Check(cudaPointerGetAttributes(&attrs, b.numasData[node]) == cudaSuccess &&
                          attrs.type == cudaMemoryTypeHost, "arena slice is not CUDA registered");
                    void *device = nullptr;
                    Check(cudaMalloc(&device, shardBytes) == cudaSuccess, "CUDA test allocation failed");
                    std::vector<uint8_t> copied(shardBytes);
                    Check(cudaMemcpyAsync(device, b.numasData[node], shardBytes, cudaMemcpyHostToDevice) == cudaSuccess,
                          "async arena slice upload failed");
                    Check(cudaMemcpy(copied.data(), device, shardBytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                          "arena roundtrip failed");
                    Check(std::memcmp(copied.data(), a.numasData[node], shardBytes) == 0,
                          "arena GPU roundtrip changed weight bytes");
                    cudaFree(device);
                } else {
                    Check(!b.isPinned, "unpinned mode falsely reports pinned storage");
                }
#endif
            }
        }
        auto *survivor = batched.back()->numasData[0];
        const size_t lastBytes = batched.back()->GetBytes() / nodes;
        // Destroy the first owner and all intervening owners; the final slice
        // must retain both its allocation and its CUDA registration.
        for (int i = 0; i < 23; ++i) batched[i].reset();
        Check(std::memcmp(survivor, reference.back()->numasData[0], lastBytes) == 0,
              "destroying an arena peer invalidated a live weight");
#ifdef USE_CUDA
        if (NumasMoeWeightsArePinned()) {
            unsigned int flags = 0;
            Check(cudaHostGetFlags(&flags, survivor) == cudaSuccess, "arena unregistered before last owner");
        }
#endif
        batched.back().reset();
#ifdef USE_CUDA
        if (NumasMoeWeightsArePinned()) {
            unsigned int flags = 0;
            Check(cudaHostGetFlags(&flags, survivor) != cudaSuccess, "arena still registered after last owner");
            cudaGetLastError();
        }
#endif
        for (auto &weight : reference) {
            for (auto *ptr : weight->numasData) {
                if (weight->isPinned) free_pinned_numa(ptr, weight->GetBytes() / nodes + 63);
                else free_aligned_numa(ptr, weight->GetBytes() / nodes + 63);
            }
            weight->numasData.clear();
        }
        // Re-register after the first arena was released, including address reuse.
        auto next = MakeWeight(19);
        RegisterNumasMoeWeightBatch({{next.get(), "linearColumn"}});
        Check(next->cpuData == nullptr && next->numasData.size() == (size_t)nodes,
              "second batch registration failed");
        std::puts("PASS: bounded NUMA arenas, exact GGUF layouts, CUDA transfers and shared lifetime");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
