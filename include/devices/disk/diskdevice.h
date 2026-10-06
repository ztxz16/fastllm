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
    };
    // CUDA bytes are summed across devices; the configured CUDA budget is per GPU.
    DiskMoeCacheStats GetDiskMoeCacheStats();
    void TrimDiskMoeCache();
    void ReleaseDiskMoeCache(const Data *weight);
    // Register metadata only. Compact experts stay in their checkpoint files;
    // both bounded cache tiers use the public MoE frequency configuration.
    bool PrepareDiskMoeCache(const std::vector<std::vector<Data*>> &layers);

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

    class DiskDevice : BaseDevice {
    public:
        DiskDevice();

        bool Malloc(void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class DiskMergeMOE : CpuMergeMOE {
        void RunCached(const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    public:
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
