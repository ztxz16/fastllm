//
// Created by huangyuyang on 8/2/24.
//

#include "fastllm.h"

std::vector <long long> FastllmCudaGetFreeSizes();

#ifdef  __cplusplus
extern "C" {
#endif

// deviceId -> [[l0, r0), [l1, r1), ...]
using DivisionScheme = std::map <int, std::vector <std::pair <int, int> > >;

void FastllmInitNccl(const std::vector<int>& devices);
void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId);
void FastllmNcclAllReduce(void* data, void* dest, int count, int dataType, int deviceId);

std::vector <int> FastllmMultiCudaGetSplitPoints(std::vector <int> &multiCudaCurrentDevices, std::map <int, int> &multiCudaCurrentRatios, int total, int unit);
void FastllmGetMulticudaDeviceAndRatio(std::vector <int> &devices, std::map <int, int> &ratios, bool noSpecial);
bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
    std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis);
bool SplitMultiCudaWeight1D(fastllm::Data &bias, std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme); // 1维的多卡切分
void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData);

void FastllmMultiCudaSetDevice(std::vector <int> ids);
void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio);

bool FastllmMultiCudaHalfMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

#ifdef  __cplusplus
}
#endif
