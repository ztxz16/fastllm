#include "devices/cuda/fastllm-cuda-host-upload.h"
#include "devices/cuda/fastllm-cuda-mapped-host.h"
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main() {
    try {
        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        if (!devices) return 77;
        // Exercise the last owner on a transient submitting thread before
        // the main thread retains any upload ring or shared copy helper.
        for (bool immutable : {false, true}) {
            std::exception_ptr failure;
            std::thread uploader([&] {
                try {
                    Check(cudaSetDevice(0));
                    std::vector<unsigned char> source(1ULL << 20, 37), actual(source.size());
                    unsigned char *target = nullptr;
                    cudaStream_t stream = nullptr;
                    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
                    Check(cudaMalloc(&target, source.size()));
                    Check(immutable
                        ? fastllm::cuda::UploadImmutableHostWeight(target, source.data(), source.size(), false, stream)
                        : fastllm::cuda::UploadHostWeight(target, source.data(), source.size(), false, stream));
                    Check(cudaStreamSynchronize(stream));
                    Check(cudaMemcpy(actual.data(), target, source.size(), cudaMemcpyDeviceToHost));
                    if (actual != source) throw std::runtime_error("transient uploader corrupted data");
                    Check(cudaFree(target));
                    Check(cudaStreamDestroy(stream));
                } catch (...) { failure = std::current_exception(); }
            });
            uploader.join();
            if (failure) std::rethrow_exception(failure);
        }
        std::puts("PASS transient uploader thread lifetime");
        // Cross more than four ring slots, include an unaligned tail, and
        // immediately overwrite each ordinary source after Copy returns.
        const size_t bytes = (21ULL << 20) + 37;
        std::vector<unsigned char> source(bytes), expected(bytes), actual(bytes);
        for (size_t i = 0; i < bytes; ++i) expected[i] = (i * 17 + i / 65537) % 251;
        for (int device = 0; device < std::min(devices, 2); ++device) {
            Check(cudaSetDevice(device));
            cudaStream_t streams[2];
            unsigned char *targets[2];
            for (int i = 0; i < 2; ++i) {
                Check(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
                Check(cudaMalloc(&targets[i], bytes));
            }
            for (int repeat = 0; repeat < 4; ++repeat) {
                for (int i = 0; i < 2; ++i) {
                    source = expected;
                    Check(fastllm::cuda::UploadHostWeight(targets[i], source.data(), bytes, false, streams[i]));
                    std::fill(source.begin(), source.end(), 0);
                }
                for (int i = 0; i < 2; ++i) {
                    Check(cudaStreamSynchronize(streams[i]));
                    Check(cudaMemcpy(actual.data(), targets[i], bytes, cudaMemcpyDeviceToHost));
                    if (actual != expected) throw std::runtime_error("pageable upload reused a live source/slot");
                }
            }
            // Borrowed immutable weights may return before their CPU copy.
            // More than four chunks exercise wraparound without a host wait.
            for (int repeat = 0; repeat < 4; ++repeat) {
                for (int i = 0; i < 2; ++i)
                    Check(fastllm::cuda::UploadImmutableHostWeight(targets[i], expected.data(), bytes, false, streams[i]));
            }
            for (int i = 0; i < 2; ++i) {
                Check(cudaStreamSynchronize(streams[i]));
                Check(cudaMemcpy(actual.data(), targets[i], bytes, cudaMemcpyDeviceToHost));
                if (actual != expected) throw std::runtime_error("immutable upload reused a live ring slot");
            }
            // A rejected CUDA copy must still retire its CPU producer job.
            // Reuse that slot after wraparound and destroy the ring safely.
            {
                fastllm::cuda::ImmutableHostUploadRing rejected;
                if (rejected.Init(device)) {
                    if (rejected.Copy(nullptr, expected.data(), 4096, streams[0]) == cudaSuccess)
                        throw std::runtime_error("invalid upload unexpectedly succeeded");
                    cudaGetLastError();
                    Check(rejected.Copy(targets[0], expected.data(), bytes, streams[0]));
                    Check(cudaStreamSynchronize(streams[0]));
                    Check(cudaMemcpy(actual.data(), targets[0], bytes, cudaMemcpyDeviceToHost));
                    if (actual != expected) throw std::runtime_error("rejected upload poisoned the ring");
                }
            }
            fastllm::cuda::MappedHostBuffer metadata;
            Check(metadata.Allocate(bytes, false));
            if (metadata.Device()) throw std::runtime_error("ordinary metadata unexpectedly mapped");
            std::memcpy(metadata.Host(), expected.data(), bytes);
            Check(metadata.CopyToDeviceAsync(targets[0], metadata.Host(), bytes, streams[0]));
            Check(cudaStreamSynchronize(streams[0]));
            Check(cudaMemcpy(actual.data(), targets[0], bytes, cudaMemcpyDeviceToHost));
            if (actual != expected) throw std::runtime_error("ordinary metadata upload corrupted");
            // Destroy streams with uploads still queued, then exceed the
            // registry bound. Eviction must drain its own last DMA safely.
            cudaStream_t transient[16];
            for (auto &stream : transient) Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (int i = 0; i < 16; ++i) {
                const size_t offset = size_t(i) << 20;
                Check(fastllm::cuda::UploadImmutableHostWeight(targets[0] + offset,
                    expected.data() + offset, 1ULL << 20, false, transient[i]));
                Check(cudaStreamDestroy(transient[i]));
            }
            Check(cudaDeviceSynchronize());
            Check(cudaMemcpy(actual.data(), targets[0], bytes, cudaMemcpyDeviceToHost));
            if (actual != expected) throw std::runtime_error("transient stream eviction corrupted an upload");
            for (int i = 0; i < 2; ++i) { Check(cudaFree(targets[i])); Check(cudaStreamDestroy(streams[i])); }
        }
        std::puts("PASS pageable upload lifetime and device/stream isolation");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
