#ifndef FASTLLM_DISKDEVICE_H
#define FASTLLM_DISKDEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/kimi_k3_ops.h"
#include <future>

namespace fastllm {
    // Immutable disk embedding rows, with a bounded host cache and dedicated
    // sleeping I/O workers. Tickets own their buffers until every read finishes.
    class DiskEmbeddingRowReader {
    public:
        struct Ticket {
            std::vector<int32_t> rows;
            std::shared_future<std::vector<uint8_t>> values;
        };
        struct Stats {
            uint64_t requests = 0, hits = 0, reads = 0, cacheBytes = 0;
        };
        DiskEmbeddingRowReader(const Data &weight, size_t cacheBytes, int threads = 4);
        ~DiskEmbeddingRowReader();
        DiskEmbeddingRowReader(const DiskEmbeddingRowReader&) = delete;
        DiskEmbeddingRowReader &operator=(const DiskEmbeddingRowReader&) = delete;
        Ticket ReadAsync(const std::vector<int32_t> &rows);
        size_t RowBytes() const;
        Stats GetStats() const;
    private:
        struct Impl;
        std::unique_ptr<Impl> impl;
    };

    struct DiskMoeCacheStats {
        uint64_t cpuBytes = 0, cudaBytes = 0;
        uint64_t cpuHits = 0, cudaHits = 0, misses = 0;
        uint64_t diskBytes = 0, uploads = 0, cpuEvictions = 0, cudaEvictions = 0;
        // Compact hierarchy: retained host/GPU overlap and selected GPU-to-RAM moves.
        uint64_t cpuCudaOverlapBytes = 0, cudaDemotions = 0, cudaDemotionBytes = 0;
        uint64_t cpuWorkspaceBytes = 0, cpuBufferReuses = 0, cpuBufferAllocations = 0;
        uint64_t cpuReclaims = 0, asyncMaintenance = 0;
        uint64_t cpuBufferMmaps = 0, cpuBufferMunmaps = 0, cpuBufferSplits = 0, cpuBufferCoalesces = 0;
        uint64_t canonicalPromotionChecks = 0, canonicalPromotionRejects = 0, canonicalResidenceRejects = 0;
        // Host wall time including allocation/page preparation and transfer waits.
        uint64_t canonicalUploadBytes = 0, canonicalUploadUs = 0;
        uint64_t canonicalDownloadUs = 0; // Bytes are already counted by cudaDemotionBytes.
        uint64_t decodeHandoffD2HCopies = 0, decodeHandoffD2HBytes = 0, decodeHandoffWaitUs = 0;
        uint64_t decodeOutputH2DCopies = 0, decodeOutputH2DBytes = 0;
        uint64_t residentReadbackCopies = 0, residentReadbackBytes = 0, residentReadbackWaitUs = 0;
    };
    // CUDA bytes are summed across devices; the configured CUDA budget is per GPU.
    DiskMoeCacheStats GetDiskMoeCacheStats();
    void TrimDiskMoeCache();
    void ReleaseDiskMoeCache(const Data *weight);
    // Fill unused RAM budget across layers with native block32 or canonical GGUF experts.
    // Entries remain evictable; loading alone does not count as a route/hit.
    uint64_t PreloadDiskMoeCache(const std::vector<std::vector<Data*>> &layers);
    // Register metadata only. Compact experts stay in their checkpoint files;
    // both bounded cache tiers use the public MoE frequency configuration.
    bool PrepareDiskMoeCache(const std::vector<std::vector<Data*>> &layers);

    // Whole disk layers may share a decode step with whole CUDA-resident
    // layers. Reject CPU/NUMA and partially placed layers; an empty result
    // also keeps models without disk experts on their existing paths.
    inline std::vector<int> DiskMoeLayerIndices(
            const std::vector<std::vector<Data *>> &layers, int cudaDevice = -1) {
        std::vector<int> indices;
        for (size_t l = 0; l < layers.size(); ++l) {
            const auto &table = layers[l];
            if (table.size() < 4 || table.size() % 2 || !table[2]) return {};
            const bool disk = table[2]->isDiskWeight;
            for (size_t i = 2; i < table.size(); ++i) {
                const Data *weight = table[i];
                if (!weight || weight->isDiskWeight != disk) return {};
                if (!disk && (weight->dataDevice != DataDevice::CUDA || !weight->cudaData ||
                        weight->multiDeviceData || (cudaDevice >= 0 &&
                        (weight->dataDeviceIds.size() != 1 || weight->dataDeviceIds[0] != cudaDevice))))
                    return {};
            }
            if (disk) indices.push_back(int(l));
        }
        return indices;
    }

    // Bounded prediction from previous prefill routes. Hints only overlap I/O;
    // selected expert IDs and output math are always determined by the router.
    void PrefetchDiskMoeV41(const std::vector<Data *> &weights, int rows);

    // Disk V4.1 single-token handoff only; unsupported layouts leave outputs
    // unchanged and use the ordinary executor's staging.
    bool StageDiskMoeV41Inputs(const Data &input, const Data &index, const Data &score,
                               Data &hostInput, Data &hostIndex, Data &hostScore);

    // The TP owner submits one host MoE operation, sharing its RAM tier while
    // allowing resident experts on every participating CUDA device to run.
    class DiskMoeCudaDeviceScope {
        std::vector<int> previous;
    public:
        explicit DiskMoeCudaDeviceScope(const std::vector<int> &devices);
        ~DiskMoeCudaDeviceScope();
        DiskMoeCudaDeviceScope(const DiskMoeCudaDeviceScope &) = delete;
        DiskMoeCudaDeviceScope &operator=(const DiskMoeCudaDeviceScope &) = delete;
    };

    // A single ordinary GGUF disk decode step may queue GPU work ahead of
    // CPU experts. Finish drains all posted work and reports CPU failures.
    // Ineligible builds/layouts keep the existing synchronous disk adapter.
    class DiskMoeAsyncScope {
        struct Impl;
        std::unique_ptr<Impl> impl;
    public:
        DiskMoeAsyncScope(const std::vector<std::vector<Data *>> &layers, int device, bool enabled,
                          bool retainPrefillWorkspace = false);
        ~DiskMoeAsyncScope();
        void Finish();
        static bool TryRun(int layer, const Data &input, const Data &index, const Data &score,
                           Data &output, const std::function<void()> &sharedExpert = {});
    };

    class DiskDevice : BaseDevice {
    public:
        DiskDevice();

        bool Malloc(void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class DiskMergeMOE : CpuMergeMOE {
        enum class CachedMode { Cpu, GpuPrefill, GpuDecode };
        bool RunCached(const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams,
                       CachedMode mode = CachedMode::Cpu);
    public:
        // Leaves input/output on CUDA when the complete routed prefill fits.
        // A false result leaves the caller free to use the ordinary executor.
        bool RunGpuPrefill(Data &input, Data &index, Data &score, Data &output,
                           std::vector<Data *> &weights, float swigluLimit);
        bool RunGpuDecode(Data &input, Data &index, Data &score, Data &output,
                          std::vector<Data *> &weights, float swigluLimit);
        // Canonical ordinary GGUF decode. Routing and activations are staged
        // together; the optional independent shared branch follows readback.
        bool RunGpuGGUFDecode(Data &input, Data &index, Data &score, Data &output,
                              std::vector<Data *> &weights,
                              const std::function<void()> &sharedExpert = {});

        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class DiskKimiK3RoutedExpertsOp : public CpuKimiK3RoutedExpertsOp {
    public:
        bool CanRun(const std::string &opType, const DataDict &datas,
                    const FloatDict &floatParams,
                    const IntDict &intParams) override;
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams,
                 const IntDict &intParams) override;
    };

    class DiskLinearOp : public BaseOperator {
    public:
        void Reshape(const std::string &opType, const DataDict &datas,
                     const FloatDict &floatParams, const IntDict &intParams) override;
        bool CanRun(const std::string &opType, const DataDict &datas,
                    const FloatDict &floatParams, const IntDict &intParams) override;
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams) override;
    };

    class DiskEmbeddingOp : public BaseOperator {
    public:
        explicit DiskEmbeddingOp(bool direct) : direct(direct) {}

        void Reshape(const std::string &opType, const DataDict &datas,
                     const FloatDict &floatParams, const IntDict &intParams) override;
        bool CanRun(const std::string &opType, const DataDict &datas,
                    const FloatDict &floatParams, const IntDict &intParams) override;
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams) override;

    private:
        bool direct;
    };
}

#endif
