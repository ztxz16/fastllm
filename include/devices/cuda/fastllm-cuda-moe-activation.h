#pragma once

#include "fastllm.h"

namespace fastllm {
// Both projections and route reduction belong to MergeMOE. A model can
// supply the intermediate activation when it includes quantization and
// must consume route scores before the down projection.
struct CudaMoeActivation {
    virtual void Run(const Data &gateUp, const Data &scores, Data &middle,
                     int rows, int intermediate) const = 0;
    virtual ~CudaMoeActivation() = default;
};

// A prepared layout owns the grouped expert matrices. Scratch belongs to the
// call so graphs and concurrent requests never share mutable routing state.
struct CudaGroupedMxfp8Moe {
    virtual size_t WorkspaceInts(int rows, int topk) const = 0;
    virtual void Run(const Data &input, Data &output, const Data &indices, const Data &scores,
                     Data &gateUp, Data &middle, int *scratch, const CudaMoeActivation &activation) const = 0;
    virtual ~CudaGroupedMxfp8Moe() = default;
};
std::unique_ptr<CudaGroupedMxfp8Moe> PrepareCudaGroupedMxfp8Moe(Data **weights, int count);

bool PrepareCudaMergeMOE(Data **weights, int weightsBatch);
void ReleaseCudaMergeMOE(Data **weights, int weightsBatch);
void DoCudaMergeMOE(const Data &input, Data &output, const Data &indices, const Data &scores,
                    Data &gateUp, Data &middle, Data **weights, int weightsBatch,
                    const CudaMoeActivation &activation, const CudaGroupedMxfp8Moe *grouped = nullptr,
                    int *workspace = nullptr);

bool RunCudaMarlinMoeWithActivation(const Data &input, Data &output, const Data &indices, const Data &scores,
                                   Data &gateUp, Data &middle, Data **weights, int weightsBatch,
                                   const CudaMoeActivation &activation);
bool RunCudaIndexedMoeWithActivation(const Data &input, Data &output, const Data &indices,
                                     const Data &scores, Data &gateUp, Data &middle, Data **weights,
                                     int weightsBatch, const CudaMoeActivation &activation);
bool PrepareCudaIndexedMoeWeights(Data **weights, int weightsBatch);
void ReleaseCudaIndexedMoeWeights(Data **weights);
}
