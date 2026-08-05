//
// Created by huangyuyang on 10/15/25.
//

#ifndef FASTLLM_NUMASDEVICE_H
#define FASTLLM_NUMASDEVICE_H

#include "device.h"
#include "devices/cpu/cpudevice.h"

namespace fastllm {
    class NumasDevice : BaseDevice {
    public:
        NumasDevice();

        // numa use cpu DDR
        bool Malloc (void **ret, size_t size);
        bool Free(void *ret);

        bool CopyDataToCPU(void *dst, void *src, size_t size);
        bool CopyDataFromCPU(void *dst, void *src, size_t size);
    };

    class NumasLinearOp : CpuLinearOp {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        long long int Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumasMergeMOE : CpuMergeMOE {
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    class NumasDeepSeekV4WoAOp : public CpuDeepSeekV4WoAOp {
    protected:
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams,
                 const IntDict &intParams) override;
    };

    class NumasFusedMOE : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams);
    };

    // Register a large set of ordinary row-major linear weights in a small
    // number of per-node arenas.  GGUF weights are repacked in place; the
    // other formats supported by MergeMOE reuse RegisterNumas' conversion
    // path before their NUMA shards are consolidated into the arenas.  This
    // is intended for checkpoints that keep every routed expert as an
    // individual tensor: allocating one NUMA mmap per tensor would otherwise
    // exhaust vm.max_map_count.
    void RegisterNumasLinearWeightBatch(const std::vector<Data*> &weights);
    bool IsNumasLinearWeightSupported(const Data *weight);
    bool IsNumasLinearWeightRegistered(const Data *weight);

    // NUMA MoE keeps reusable host/CUDA staging buffers outside the model.
    // Release them explicitly while the CUDA allocator is still alive.
    void ClearNumasMoeRuntimeCache();

    class NumasKimiK3RoutedExpertsOp : BaseOperator {
        bool CanRun(const std::string &opType, const DataDict &datas,
                    const FloatDict &floatParams, const IntDict &intParams);
        void Reshape(const std::string &opType, const DataDict &datas,
                     const FloatDict &floatParams, const IntDict &intParams);
        void Run(const std::string &opType, const DataDict &datas,
                 const FloatDict &floatParams, const IntDict &intParams);
    };
}

#endif
