#include "fastllm.h"
#include "utils.h"
#include "devices/disk/diskdevice.h"
#include "devices/cpu/alivethreadpool.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <cmath>
#include <future>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unistd.h>

using namespace fastllm;
namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
struct CpuReference : CpuMergeMOE { using CpuMergeMOE::Run; };
struct Fixture {
    static constexpr int hidden = 256, middle = 128, experts = 12;
    std::vector<std::unique_ptr<Data>> storage;
    std::vector<Data*> memory, disk, biases;
    std::string path;
    uint64_t expertBytes = 0;
    Fixture(bool shared) : memory((experts + 1) * 2), disk(memory.size()), biases(memory.size()) {
        char name[] = "/tmp/fastllm-disk-moe-XXXXXX";
        int fd = mkstemp(name);
        Require(fd >= 0, "fixture file");
        FILE *file = fdopen(fd, "wb");
        path = name;
        for (int e = shared ? 0 : 1; e <= experts; ++e) {
            for (int part = 0; part < 2; ++part) {
                auto w = std::make_unique<Data>(DataType::NVFP4);
                w->blockK = 1; w->blockM = 32;
                w->Resize(part ? std::vector<int>{hidden, middle} : std::vector<int>{2 * middle, hidden});
                w->Allocate(false);
                size_t payload = GetNVFP4WeightBytes(w->dims[0], w->dims[1]);
                for (size_t i = 0; i < payload; ++i) w->cpuData[i] = uint8_t((i * 37 + e * 13 + part * 17) ^ (i >> 7));
                for (size_t i = payload; i < w->GetBytes(); ++i) w->cpuData[i] = 122 + (i + e + part) % 3;
                auto lazy = std::make_unique<Data>(DataType::NVFP4);
                lazy->blockK = 1; lazy->blockM = 32; lazy->Resize(w->dims);
                lazy->isDiskWeight = true; lazy->isModelWeight = true;
                DiskWeightPart source;
                source.fileName = path; source.fileOffset = ftell(file);
                source.bytes = w->GetBytes(); source.sourceDataType = DataType::NVFP4; source.dims = w->dims;
                lazy->diskWeightParts.push_back(source);
                Require(fwrite(w->cpuData, 1, w->GetBytes(), file) == w->GetBytes(), "fixture write");
                if (e == 1) expertBytes += w->GetBytes();
                memory[e * 2 + part] = w.get(); disk[e * 2 + part] = lazy.get();
                storage.push_back(std::move(w)); storage.push_back(std::move(lazy));
            }
        }
        fclose(file);
    }
    ~Fixture() { storage.clear(); unlink(path.c_str()); }
    std::vector<uint16_t> Run(bool cached, int batch, int offset, int device = -1,
                              bool borrowedInput = false, int topk = 3, bool quantShared = true) {
        Data x(DataType::BFLOAT16, {batch, hidden}), y(DataType::BFLOAT16, {batch, hidden});
        Data ids(DataType::INT32PARAM, {batch, topk}), scores(DataType::FLOAT32, {batch, topk});
        Data a, b, c;
        x.Allocate(false); ids.Allocate(false); scores.Allocate(false);
        for (int row = 0; row < batch; ++row) {
            for (int col = 0; col < hidden; ++col) {
                float value = RoundFloat32ToBFloat16RNE(std::sin(float(row * hidden + col) * .71f) * .4f);
                uint32_t bits; memcpy(&bits, &value, 4);
                ((uint16_t*)x.cpuData)[row * hidden + col] = bits >> 16;
            }
            for (int j = 0; j < topk; ++j) {
                ((int32_t*)ids.cpuData)[row * topk + j] = (offset + (topk - 1 - j) * 2 + row % 2) % experts;
                ((float*)scores.cpuData)[row * topk + j] = .17f + j * .21f;
            }
        }
#ifdef USE_CUDA
        if (device >= 0) {
            FastllmCudaSetDevice(device);
            x.ToDevice(DataDevice::CUDA, std::vector<int>{device}, true);
            x.ToDevice(DataDevice::CPU);
        }
#endif
        auto &table = cached ? disk : memory;
        DataDict data = {{"input", &x}, {"output", &y}, {"index", &ids}, {"score", &scores},
            {"weights", (Data*)table.data()}, {"biass", (Data*)biases.data()}, {"w1", &a}, {"w2", &b}, {"w3", &c}};
#ifdef USE_CUDA
        Data source, view;
        if (borrowedInput) {
            source.CopyFrom(x);
            source.ToDevice(DataDevice::CUDA, {0}, true);
            view.FakeFrom(source, 0);
            view.Resize(source.dims);
            view.ToDevice(DataDevice::CPU);  // Executor's disk placement
            Require(view.cpuData == nullptr, "fixture must exercise a borrowed CUDA view");
            data["input"] = &view;
        }
#endif
        IntDict params = {{"weights___batch", (int)table.size()}, {"biass___batch", (int)biases.size()},
            {"deepSeekV4Mode", 1}, {"activationQuantBlock", 32}, {"quantizeSharedExpert", quantShared ? 1 : 0}};
        FloatDict floats = {{"sharedScale", 1.0f}, {"swigluLimit", 2.0f}};
        if (cached) { DiskMergeMOE op; op.Run("MergeMOE", data, floats, params); }
        else { CpuReference op; op.Run("MergeMOE", data, floats, params); }
#ifdef USE_CUDA
        if (borrowedInput) {
            source.ToDevice(DataDevice::CPU);
            Require(std::memcmp(source.cpuData, x.cpuData, x.GetBytes()) == 0,
                "disk materialization modified or freed the borrowed input");
        }
#endif
        y.ToDevice(DataDevice::CPU);
        return std::vector<uint16_t>((uint16_t*)y.cpuData, (uint16_t*)y.cpuData + batch * hidden);
    }
};
float Float(uint16_t value) { uint32_t bits = uint32_t(value) << 16; float v; memcpy(&v, &bits, 4); return v; }
void Compare(const std::vector<uint16_t> &actual, const std::vector<uint16_t> &expected, bool exact) {
    Require(actual.size() == expected.size(), "output shape");
    double diff = 0, norm = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        float a = Float(actual[i]), b = Float(expected[i]);
        Require(std::isfinite(a), "nonfinite output");
        diff += double(a - b) * (a - b); norm += double(b) * b;
        mismatches += actual[i] != expected[i];
    }
    double relative = std::sqrt(diff / std::max(1e-12, norm));
    if ((exact && mismatches) || relative > .004) {
        std::fprintf(stderr, "mismatches=%zu/%zu relative_l2=%.8g exact=%d\n", mismatches, actual.size(), relative, exact);
        throw std::runtime_error("disk cache numerical mismatch");
    }
}
}
int main(int argc, char **argv) {
    try {
        int devices = argc > 1 ? std::stoi(argv[1]) : 0;
#ifndef USE_CUDA
        Require(devices == 0, "CUDA not built");
#else
        if (devices > 0 && devices > FastllmCudaGetDeviceCount()) return 77;
        for (int device = 0; device < devices; ++device) {
            FastllmCudaSetDevice(device);
            if (FastllmCudaRuntimeArch() < 80) return 77;
        }
#endif
        setenv("FT_GPU_PREFILL", argc > 2 ? "1" : "0", 1);
        SetThreads(8);
        for (bool shared : {false, true}) {
            Fixture f(shared);
            if (!shared && CPUInstructInfo().hasAVX512BF16 &&
                !std::getenv("FASTLLM_DSV4_DISABLE_CPU_MOE_FAST")) {
                auto expected = f.Run(false, 1, 0);
                auto *pool = GetAlivePool();
                auto previous = pool->curActivateThreadInterval;
                std::atomic<bool> release{false};
                struct Hold : MultiThreadBaseOp {
                    std::atomic<bool> &release;
                    explicit Hold(std::atomic<bool> &flag) : release(flag) {}
                    void Run() override {
                        while (!release.load()) std::this_thread::yield();
                    }
                } hold(release);
                // Occupied inactive workers make use of the historical pool
                // capacity observable, including an active interval not at 0.
                pool->curActivateThreadInterval = {3, 5};
                for (int i = 0; i < int(pool->threads.size()); ++i)
                    if (i < 3 || i >= 5) pool->PushOp(i, &hold);
                auto job = std::async(std::launch::async, [&] { return f.Run(false, 1, 0); });
                bool ready = job.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
                release.store(true);
                for (int i = 0; i < int(pool->threads.size()); ++i)
                    if (i < 3 || i >= 5) pool->Wait(i);
                auto actual = job.get();
                pool->curActivateThreadInterval = previous;
                Require(ready, "CPU MoE used inactive workers");
                Compare(actual, expected, true);
            }
            SetMoeCudaCacheBytes(0);
            SetMoeCpuCacheBytes(f.expertBytes * 7);
#ifdef USE_CUDA
            if (FastllmCudaGetDeviceCount() > 0) {
                for (int batch : {1, 7})
                    Compare(f.Run(true, batch, 0, -1, true), f.Run(false, batch, 0), true);
            }
#endif
            for (int batch : {1, 7, 8, 40, 260, 520}) {
                auto expected = f.Run(false, batch, 0);
                Compare(f.Run(true, batch, 0), expected, true);
                auto before = GetDiskMoeCacheStats();
                Compare(f.Run(true, batch, 0), expected, true);
                auto after = GetDiskMoeCacheStats();
                Require(after.cpuHits > before.cpuHits && after.diskBytes == before.diskBytes, "RAM hit read disk");
                Require(after.cpuBytes <= GetMoeCpuCacheBytes(), "RAM budget");
            }
            // A capacity smaller than one expert must remain usable without residency.
            SetMoeCpuCacheBytes(1);
            Compare(f.Run(true, 1, 0), f.Run(false, 1, 0), true);
            Require(GetDiskMoeCacheStats().cpuBytes == 0, "undersized cache retained payload");
            SetMoeCpuCacheBytes(f.expertBytes * 4);
            for (int i = 0; i < 6; ++i) f.Run(true, 1, 0);
            for (int offset = 1; offset < 8; ++offset) f.Run(true, 1, offset);
            auto warm = GetDiskMoeCacheStats();
            Compare(f.Run(true, 1, 0), f.Run(false, 1, 0), true);
            Require(GetDiskMoeCacheStats().cpuHits - warm.cpuHits == (shared ? 4 : 3), "scan displaced hot experts");
            Require(GetDiskMoeCacheStats().cpuBytes <= GetMoeCpuCacheBytes(), "eviction exceeded budget");
            if (!shared) {
                // Saturated old hot counters must eventually decay, otherwise
                // strict admission would pin the initial working set forever.
                for (int i = 0; i < 100; ++i) f.Run(true, 1, 0);
                for (int i = 0; i < 1500; ++i) f.Run(true, 1, 6);
                auto adapted = GetDiskMoeCacheStats();
                f.Run(true, 1, 6);
                Require(GetDiskMoeCacheStats().cpuHits - adapted.cpuHits == 3, "cache did not adapt to new hot set");
            }
            for (int device = 0; device < devices; ++device) {
                SetMoeCudaCacheBytes(0);
                SetMoeCpuCacheBytes(f.expertBytes * (Fixture::experts + 1));
                f.Run(true, 520, 0);
                auto ramBefore = GetDiskMoeCacheStats();
                Compare(f.Run(true, 520, 0, device), f.Run(false, 520, 0), false);
                auto ramAfter = GetDiskMoeCacheStats();
                Require(ramAfter.cpuHits > ramBefore.cpuHits && ramAfter.diskBytes == ramBefore.diskBytes,
                    "warm RAM prefill read disk");
                Require(ramAfter.cudaBytes == 0, "temporary prefill upload became permanent");
                SetMoeCpuCacheBytes(f.expertBytes * 4);
                SetMoeCudaCacheBytes(f.expertBytes * 4);
                for (int repeat = 0; repeat < 3; ++repeat) Compare(f.Run(true, 1, 0, device), f.Run(false, 1, 0), false);
                auto before = GetDiskMoeCacheStats();
                for (int batch : {1, 7, 8, 40, 260, 520}) Compare(f.Run(true, batch, 0, device), f.Run(false, batch, 0), false);
                auto after = GetDiskMoeCacheStats();
                Require(after.cudaHits > before.cudaHits, "no CUDA cache hits");
                Require(after.cudaBytes <= GetMoeCudaCacheBytes() * (device + 1), "CUDA budget");
                // GPU-only entries must remain valid after evicting their RAM copy.
                SetMoeCpuCacheBytes(0);
                before = GetDiskMoeCacheStats();
                Compare(f.Run(true, 1, 0, device), f.Run(false, 1, 0), false);
                after = GetDiskMoeCacheStats();
                Require(after.cpuBytes == 0 && after.cudaHits > before.cudaHits, "GPU-only residency");
                // Heat a disjoint routed set enough to replace CUDA records,
                // then return to the original set. Exercise borrowed views
                // after exact-size record reuse, with no RAM backup.
                auto replacing = GetDiskMoeCacheStats();
                for (int offset : {6, 0}) {
                    const auto start = GetDiskMoeCacheStats();
                    // The cache clock is shared across devices. Allow a full
                    // aging epoch instead of depending on the previous case
                    // ending within 96 calls of an epoch boundary.
                    for (int repeat = 0; repeat < 1600; ++repeat) {
                        f.Run(true, 1, offset, device);
                        if (repeat >= 95 && GetDiskMoeCacheStats().cudaEvictions > start.cudaEvictions) break;
                    }
                    Compare(f.Run(true, 1, offset, device), f.Run(false, 1, offset), false);
                }
                auto replaced = GetDiskMoeCacheStats();
                Require(replaced.cudaEvictions > replacing.cudaEvictions && replaced.uploads > replacing.uploads,
                    "CUDA replacement path was not exercised");
                Require(replaced.cpuBytes == 0 && replaced.cudaBytes <= GetMoeCudaCacheBytes() * (device + 1),
                    "CUDA replacement exceeded cache budget");
                // Duplicate routes, shared-input quantization and mixed GPU /
                // disk residency must survive deferred promotion and a single
                // gather. The tiny RAM tier also exercises read-ahead teardown.
                SetMoeCpuCacheBytes(1);
                for (bool quantShared : {false, true}) for (int batch : {1, 7, 8}) {
                    auto start = GetDiskMoeCacheStats();
                    Compare(f.Run(true, batch, 0, device, false, 8, quantShared),
                            f.Run(false, batch, 0, -1, false, 8, quantShared), false);
                    auto end = GetDiskMoeCacheStats();
                    Require(end.cudaHits > start.cudaHits && end.misses > start.misses,
                        "duplicate-route fixture did not mix GPU hits and disk reads");
                    Require(end.cpuBytes == 0, "read-ahead retained an over-budget RAM entry");
                }
            }
            SetMoeCudaCacheBytes(1);
            for (int device = 0; device < devices; ++device)
                Compare(f.Run(true, 520, 0, device), f.Run(false, 520, 0), false);
            Require(GetDiskMoeCacheStats().cudaBytes == 0, "undersized CUDA budget retained payload");
            SetMoeCudaCacheBytes(0); SetMoeCpuCacheBytes(0);
            auto stats = GetDiskMoeCacheStats();
            Require(stats.cpuBytes == 0 && stats.cudaBytes == 0, "disable did not release cache");
            Compare(f.Run(true, 1, 0), f.Run(false, 1, 0), true);
            SetMoeCpuCacheBytes(f.expertBytes * 4);
            f.Run(true, 1, 0);
        }
        auto s = GetDiskMoeCacheStats();
        Require(s.cpuBytes == 0 && s.cudaBytes == 0, "model unload retained cache");
        std::printf("PASS disk MoE cache: cpu_hits=%llu cuda_hits=%llu misses=%llu uploads=%llu\n",
            (unsigned long long)s.cpuHits, (unsigned long long)s.cudaHits, (unsigned long long)s.misses, (unsigned long long)s.uploads);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
