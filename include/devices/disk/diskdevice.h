#ifndef FASTLLM_DISKDEVICE_H
#define FASTLLM_DISKDEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/kimi_k3_ops.h"

namespace fastllm {
    struct DiskMoeCacheStats {
        uint64_t cpuBytes = 0, cudaBytes = 0;
        uint64_t cpuHits = 0, cudaHits = 0, misses = 0;
        uint64_t diskBytes = 0, uploads = 0, cpuEvictions = 0, cudaEvictions = 0;
    };
    // CUDA bytes are summed across devices; the configured CUDA budget is per GPU.
    DiskMoeCacheStats GetDiskMoeCacheStats();
    void TrimDiskMoeCache();
    void ReleaseDiskMoeCache(const Data *weight);

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
