//
// Created by huangyuyang on 8/2/24.
//

#include "fastllm.h"

std::vector <long long> FastllmCudaGetFreeSizes();
std::vector <long long> FastllmCudaGetTotalSizes();

#ifdef  __cplusplus
extern "C" {
#endif

// deviceId -> [[l0, r0), [l1, r1), ...]
using DivisionScheme = std::map <int, std::vector <std::pair <int, int> > >;

bool FastllmInitNccl(const std::vector<int>& devices);
void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId);
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
    bool MultiCudaLinearRow(Data &input, Data &weight, Data &bias, Data &output);
    bool MultiCudaLinearColumn(Data &input, Data &weight, Data &bias, Data &output);
}
