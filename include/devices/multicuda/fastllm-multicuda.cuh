//
// Created by huangyuyang on 8/2/24.
//

#pragma once

#include "fastllm.h"

#include <functional>

std::vector <long long> FastllmCudaGetFreeSizes();
std::vector <long long> FastllmCudaGetTotalSizes();

#ifdef  __cplusplus
extern "C" {
#endif

// deviceId -> [[l0, r0), [l1, r1), ...]
using DivisionScheme = std::map <int, std::vector <std::pair <int, int> > >;

bool FastllmInitNccl(const std::vector<int>& devices);
// Graph-safe small-tensor all-reduce backed by direct peer reads and a GPU-side
// barrier. It is opt-in through FASTLLM_CUDA_CUSTOM_ALLREDUCE=1; all functions
// return false when disabled or unsupported so NCCL remains the fallback.
bool FastllmCudaCustomAllReduceEnabled();
bool FastllmCudaCustomAllReduceInit(const std::vector<int>& devices);
bool FastllmCudaCustomAllReduce(void* data, void* dest, int count,
                                int dataType, int deviceId);
// TP=2 graph-safe fused residual path: dest = allreduce(data) + dest.
// Returns false without modifying dest when the topology/type is unsupported.
bool FastllmCudaCustomAllReduceAdd(void* data, void* dest, int count,
                                   int dataType, int deviceId);
// CUDA Graph 中的跨卡点对点搬运使用独立的双 rank 通信域，避免与模型的
// TP/EP 集合通信共享 NCCL 操作序列。通信域必须在开始 stream capture 前创建。
bool FastllmInitNcclGraphPeer(int srcDevice, int dstDevice);
bool FastllmNcclGraphPeerCopy(int dstDevice, void *dst,
                              int srcDevice, const void *src, size_t bytes);
void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId);
void FastllmNcclBroadcastFrom(void* send, void* recv, int count, int dataType, int root, int deviceId);
void FastllmNcclAllReduce(void* data, void* dest, int count, int dataType, int deviceId);
// Returns whether the TP=2 peer-access fast path can be used for this tensor.
// Callers use this preflight to preserve their existing NCCL fallback without
// first changing the reduction's compute or accumulation order.
bool FastllmCanUseTP2P2PAllReduceAdd(int count, int dataType, int deviceId);
// Computes dest = allreduce(data) + dest over CUDA peer access. This keeps TP
// ranks on an identical pre-reduction path and folds the replicated residual
// addition into the communication kernel. Returns false without modifying the
// tensors when the topology, data type or runtime state is not suitable.
bool FastllmTryTP2P2PAllReduceAdd(void* data, void* dest, int count, int dataType, int deviceId);
void FastllmNcclReduce(void* data, void* dest, int count, int dataType, int root, int deviceId);
void FastllmCudaPackMoeEpPacket(void *packet, const void *hidden, size_t hiddenBytes,
                               const int32_t *indices, const float *scores, int topk);
void FastllmCudaSyncDevice(int deviceId);

std::vector <int> FastllmMultiCudaGetSplitPoints(std::vector <int> &multiCudaCurrentDevices, std::map <int, int> &multiCudaCurrentRatios, int total, int unit);
void FastllmGetMulticudaDeviceAndRatio(std::vector <int> &devices, std::map <int, int> &ratios, bool noSpecial);
void BalanceMultiCudaDivisionSchemeByLayer(const std::string &weightName,
    const std::vector <int> &multiCudaCurrentDevices, DivisionScheme &divisionScheme,
    bool explicitDeviceRatios = false);
void BalanceMultiCudaPairedHalfDivisionSchemeSizesByLayer(const std::string &weightName,
    const std::vector <int> &multiCudaCurrentDevices, DivisionScheme &divisionScheme,
    int mid, bool explicitDeviceRatios = false);
bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
    std::vector <int> &multiCudaCurrentDevices, DivisionScheme &divisionScheme, int splitAxis,
    bool explicitDeviceRatios = false);
bool SplitMultiCudaWeight1D(fastllm::Data &bias, std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme); // 1维的多卡切分
bool PlaceMultiCudaWeightOnDevice(fastllm::Data &weight, std::vector <int> &multiCudaCurrentDevices, int targetDevice);
void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData);
void PrepareMultiCudaReplicatedData(fastllm::Data &data, std::vector <int> devices, bool copyData);
void PrepareMultiCudaShardedData(fastllm::Data &data, std::vector <int> devices,
    const std::vector <int> &globalDims, int axis, DivisionScheme divisionScheme);
DivisionScheme BuildMultiCudaRowSplitScheme(fastllm::Data &weight, std::vector <int> &devices, std::map <int, int> &ratios);
DivisionScheme BuildMultiCudaColumnSplitScheme(fastllm::Data &weight, std::vector <int> &devices, std::map <int, int> &ratios);

void FastllmMultiCudaSetDevice(std::vector <int> ids);
void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio);

bool FastllmMultiCudaHalfMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

#ifdef  __cplusplus
}
#endif

namespace fastllm {
    // Retire EP caches keyed by a routed-expert Data object before the object's
    // address can be reused by a subsequently loaded model.
    void MultiCudaReleaseMoeWeightCaches(const Data *layerKey);

    bool MultiCudaLinearRow(Data &input, Data &weight, Data &bias, Data &output);
    bool MultiCudaLinearColumn(Data &input, Data &weight, Data &bias, Data &output);

    // 将常驻 MultiCUDA worker 的 per-thread stream 接入/并回调用线程正在捕获的
    // 每卡 CUDA Graph。events 必须与 devices 一一对应并归属相同设备。
    bool MultiCudaGraphWorkersWaitEvents(const std::vector<int> &devices,
                                         const std::vector<void*> &events);
    bool MultiCudaGraphWorkersRecordEvents(const std::vector<int> &devices,
                                           const std::vector<void*> &events);

    // 单 token、固定 workspace 的模型路径可让连续 MultiCUDA 算子复用每卡
    // 常驻 worker stream，省去每个算子边界的 device synchronize。调用方必须
    // 保证参与计算的 tensor 生命周期覆盖整个异步区间，并在回到调用线程的
    // CUDA stream 前显式等待对应 worker event。
    bool MultiCudaSetPersistentAsyncDispatch(bool enabled);
    bool MultiCudaRunDeviceCallbacks(
        const std::vector<int> &devices,
        const std::function<void(int, int)> &callback);
    bool MultiCudaCurrentThreadWaitForWorker(int device);

    // Broadcast a root CUDA tensor into its fixed per-device replicas, then
    // execute Repeat independently on every device.  This keeps the output
    // replicas current without rebuilding their storage, so the whole handoff
    // can be captured by a multi-device CUDA Graph.
    bool MultiCudaRepeatToReplicated(Data &input, int axis, int repeatTimes,
                                     Data &output);
}
