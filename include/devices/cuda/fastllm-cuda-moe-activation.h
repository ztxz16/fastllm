#pragma once

#include "fastllm.h"

struct CUstream_st;

namespace fastllm {
// Both projections and route reduction belong to MergeMOE. A model can
// supply the intermediate activation when it includes quantization and
// must consume route scores before the down projection.
struct CudaMoeActivation {
    virtual void Run(const Data &gateUp, const Data &scores, Data &middle,
                     int rows, int intermediate) const = 0;
    virtual ~CudaMoeActivation() = default;
};

// Run accepts FP32 gate/up in expert order with equally permuted scores.
// Writing the result to FP16 matches a later round-to-nearest gather.
struct CudaExpertOrderedFp16Activation : public CudaMoeActivation {};

struct CudaMarlinMxfp8Weights {
    const uint32_t *codes;
    const uint8_t *scales;
};

// A prepared layout owns the grouped expert matrices. Scratch belongs to the
// call so graphs and concurrent requests never share mutable routing state.
struct CudaGroupedMxfp8Moe {
    virtual size_t WorkspaceInts(int rows, int topk) const = 0;
    virtual void Run(const Data &input, Data &output, const Data &indices, const Data &scores,
                     Data &gateUp, Data &middle, int *scratch, const CudaMoeActivation &activation) const = 0;
    virtual ~CudaGroupedMxfp8Moe() = default;
    // Backends with grouped Marlin weights can share them with native prefill.
    virtual CudaMarlinMxfp8Weights MarlinWeights(bool downProjection) const { return {}; }
    // Leaves contiguous [token, route, hidden] BF16 down-projection rows in gateUp.
    // Callers can combine route reduction with the next operation.
    virtual void ProjectRoutes(const Data &input, const Data &indices, const Data &scores, Data &gateUp,
                               Data &middle, int *scratch, const CudaMoeActivation &activation) const = 0;
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
// Stable slot tables for indexed FP32 projection. Owners may refill the payload
// between calls. NVFP4 uses block-16 records; FP8 uses separate block scales.
struct CudaIndexedMoeSlots {
    uint8_t **gate, **down;
    float **gateScales, **downScales;
    int gateBlockM, gateBlockK, downBlockM, downBlockK;
};
void CudaMoeGraphSlotGate(const CudaIndexedMoeSlots &, float *input, const int *slots, float *output,
                          int rows, int top, int hidden, int intermediate, CUstream_st *stream);
void CudaMoeGraphSlotDown(const CudaIndexedMoeSlots &, float *middle, const int *slots, float *output,
                          int routes, int hidden, int intermediate, CUstream_st *stream);
bool RunCudaIndexedMoeSlots(const CudaIndexedMoeSlots &slots, const Data &input, const Data &indices,
                            const Data &scores, Data &gateUp, Data &middle, Data &output,
                            int intermediate, const CudaMoeActivation &activation);
bool RunCudaIndexedMoeWithActivation(const Data &input, Data &output, const Data &indices,
                                     const Data &scores, Data &gateUp, Data &middle, Data **weights,
                                     int weightsBatch, const CudaMoeActivation &activation);
bool GetCudaIndexedMoeSlots(Data **weights, int weightsBatch, CudaIndexedMoeSlots &slots);
bool PrepareCudaIndexedMoeWeights(Data **weights, int weightsBatch);
void ReleaseCudaIndexedMoeWeights(Data **weights);
}
