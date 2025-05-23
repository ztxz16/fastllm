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

std::vector <int> FastllmMultiCudaGetSplitPoints(std::vector <int> &multiCudaCurrentDevices, std::map <int, int> &multiCudaCurrentRatios, int total, int unit);
void FastllmGetMulticudaDeviceAndRatio(std::vector <int> &devices, std::map <int, int> &ratios, bool noSpecial);
bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
    std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme, int splitAxis);
void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData);

void FastllmMultiCudaSetDevice(std::vector <int> ids);
void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio);

bool FastllmMultiCudaHalfMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);
bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

#ifdef  __cplusplus
}
#endif
