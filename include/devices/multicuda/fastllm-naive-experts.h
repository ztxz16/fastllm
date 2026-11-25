#pragma once

#include "fastllm.h"
#include "models/naive_moe_dispatch.h"
#include "devices/cuda/fastllm-mxfp8-projection.h"
#include "devices/cuda/fastllm-naive-flex-moe.h"

struct FastllmCudaMimoEp8Mega;

namespace fastllm {
class PersistentWorkerGroup;
struct NaiveFlexMoe;
struct NaiveSharedFlexMoe;
struct Mxfp4MarlinRank;

// Optional producer hooks for a backend that accepts prepared activations or routes.
struct FastllmExpertHandoff {
    virtual ~FastllmExpertHandoff() = default;
    virtual bool HasVerifyReduce(int rank, int layer, int rows) const = 0;
    virtual bool CanRoute(int rows, const Data &router, const Data &bias) const = 0;
    virtual bool PreparedInput(int rank, int rows, FastllmMoePreparedInput &input,
            Data &ids, Data &scores, const Data &bias, const Data &hidden,
            bool needNorm, float routeScale) const = 0;
    virtual bool PrepareResidualRouter(int rank, int layer, int rows, Data &hidden, const Data &branch,
            const Data &gamma, float eps, Data &normed, const Data &weight, Data &router,
            const Data &bias, Data &ids, Data &scores, bool needNorm, float routeScale) const = 0;
    virtual bool PrepareDense(int rank, int rows, Data &input, Data &gate, Data &up, Data &down,
                              Data &gamma, float eps, Data &silu, Data &out) = 0;
    virtual void Dense(int rank, int rows, Data &input, Data &gate, Data &up, Data &down,
                       Data &gamma, float eps, Data &silu, Data &out) = 0;
};

enum class FastllmDenseResult { Unsupported, Local, Reduced };
FastllmDenseResult FastllmRunNaiveDense(FastllmExpertHandoff *prepared, bool decode, int rank, int rows,
        Data &hidden, Data &normed, Data &gate, Data &up, Data &down, Data &gamma,
        float eps, Data &scratch, Data &output);

struct FastllmTpExperts;
struct FastllmExpertRouter {
    const FastllmTpExperts *experts;
    int rank;
    Data &ids, &scores, &probabilities, &projectionScratch;
    const Data &bias;
    int topk;
    bool normalize;
    float scale;
    bool packedVerify, preparedRoutes = false;

    // A pending gamma means that the producer handed off an unnormalized residual.
    bool Project(Data &input, Data &weight, Data *verifyScratch, bool probabilitiesReady,
                 const Data *residual, const Data *gamma, float eps);
    bool Select(const Data &probabilities);
    bool CanRouteInExpert(int rows) const;
};

struct FastllmExpertNorm {
    Data &hidden;
    const Data &weight;
    Data &output;
    float eps;
    FastllmMxfp8Activation next;
    Data *feature;
    int featureIndex, featureCount;
};

// Some ordinary TP weights admit resident row views before grouped execution is eligible.
enum class FastllmExpertRows { Owning, Resident, Complete };
FastllmExpertRows FastllmPrepareExpertRows(const Data &input, Data &gate, Data &activation, Data &output,
        std::vector<Data *> &weights, const Data &ids, const Data &scores, int rows, int topk, bool allowGrouped);

struct FastllmExpertResult {
    bool residualUpdated = false, reduced = false, trackReduction = false, normalized = false, featureWritten = false;
};

struct FastllmTpExperts {
    using Weights = std::vector<std::vector<std::vector<Data *>>>;
    static NaiveMoeBackend Plan(const std::vector<int> &devices, int ep, bool enabled,
                                bool flexGeometry, bool attentionReplicated, bool megaAvailable, bool &packsSources);
    static std::shared_ptr<FastllmTpExperts> Prepare(NaiveMoeBackend &layout, bool &packsSources,
            const std::vector<int> &devices, const Weights &weights, int hidden, int experts, int topk,
            bool allowPackedTp, bool preferVerifyPages, PersistentWorkerGroup &workers, bool &packing,
            std::shared_ptr<FastllmCudaMimoEp8Mega> &mega);
    const char *Name() const;
    bool UpdatesResidual() const { return bool(legacy); }
    bool AllowsLargeBatch() const { return bool(shared); }
    void Validate(bool bf16, bool fixedRouting) const;
    const FastllmTpExperts *ForRows(int rows, bool replicated) const;
    FastllmExpertHandoff *Handoff() const;
    bool PackedVerify(int rows, int ranks) const;
    FastllmExpertResult Forward(int layer, int rows, Data &input, Data &hidden, Data &scratch,
            Data &output, const FastllmExpertRouter &router, bool routeInExpert, bool verifyReduce,
            const FastllmMoePreparedInput *prepared, const FastllmExpertNorm *norm) const;
private:
    friend struct FastllmExpertRouter;
    std::shared_ptr<NaiveFlexMoe> legacy;
    std::shared_ptr<NaiveSharedFlexMoe> shared;
    std::vector<std::shared_ptr<Mxfp4MarlinRank>> ranks;
};
}
