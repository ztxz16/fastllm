#pragma once
#include "fastllm-swa-output.cuh"
#include "fastllm-swa-layer.cuh"
#include "fastllm-swa-verify-layer.cuh"
#include "fastllm-cuda-launch.cuh"
#include "fastllm-attention-output-rows.cuh"
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
#include "devices/cuda/naive-n05-dsa-layer.cuh"
#endif
#include <atomic>
#include <memory>
#include <vector>
#include "devices/cuda/naive-n05-swa-settings.h"

namespace {
class SwaOutputState {
    std::vector<int> devices;
    fastllm_swa_output::Peers peers;
    uint64_t *epochs[8]{};
    const void *const kernel = (const void *)fastllm_swa_output::OutputNorm<32>;
    bool verifySupported[4]{};
    bool layerSupported = false, mxLayerSupported = false, mxOutputSupported = false, outputRouterSupported = false, rowsSupported = false;
    bool rowsRouterExpertsSupported = false, rowsMmaSupported = false;
    bool preparedRouterSupported = false, routedRouterSupported = false;
    bool swaRouterPdlSupported = false, unpreparedRouterPdlSupported = false;
    bool preparedVectorRouterSupported = false, rowsDirectRouterSupported = false;
    bool widePreparedRouterSupported = false;
    // A registered output tuple alone does not establish that the eight-row
    // GEMM has warmed its process-lifetime BLAS handle outside capture.
    std::atomic<bool> rowsGemmPrepared[8]{};
public:
    static bool Enabled() {
        return fastllm_swa_settings::Enabled("FASTLLM_NAIVE_SWA_FUSION");
    }
    static std::unique_ptr<SwaOutputState> Create(const std::vector<int> &requested, bool enabled = Enabled()) {
        if (requested.size() != 8 || !enabled) return {};
        auto state = std::unique_ptr<SwaOutputState>(new SwaOutputState);
        if (!state->Initialize(requested)) { cudaGetLastError(); return {}; }
        return state;
    }
    ~SwaOutputState() { Release(true); }
    void ReleaseAfterDrain() { Release(false); }
    cudaError_t Launch(const void *input, const void *weight, void *projected,
            void *hidden, const float *normWeight, void *normed, float eps, int rank) {
        void *args[] = {&input, &weight, &projected, &hidden, &normWeight, &normed,
                        &eps, &peers, &epochs[rank], &rank};
        return cudaLaunchCooperativeKernel(kernel,
            dim3(128), dim3(256), args, 0, cudaStreamPerThread);
    }
    bool MxOutputSupported() const { return mxOutputSupported; }
    cudaError_t LaunchMxOutput(const void *input, const void *weight, const float *scales,
            void *projected, void *hidden, const float *gamma, void *normed, float eps,
            const float *routerWeight, float *router, int rank) {
        using namespace fastllm_swa_layer;
        void *standaloneNorm = routerWeight ? nullptr : normed;
        void *args[] = {&input, &weight, &scales, &projected, &hidden, &peers,
                        &epochs[rank], &rank, &gamma, &standaloneNorm, &eps};
        auto status = cudaLaunchCooperativeKernel((const void *)MxOutput,
            dim3(LayerBlocks), dim3(LayerThreads), args, 0, cudaStreamPerThread);
        if (status != cudaSuccess || !routerWeight) return status;
        NormRouter4096<<<Experts, RouterThreads, 0, cudaStreamPerThread>>>(
            (const BF16 *)hidden, routerWeight, router, gamma, (BF16 *)normed, eps, nullptr);
        return cudaGetLastError();
    }
    bool OutputRouterSupported() const { return outputRouterSupported; }
    cudaError_t LaunchOutputRouter(const void *input, const void *weight, void *projected,
            void *hidden, const float *normWeight, void *normed, float eps,
            const float *routerWeight, float *router, int rank) {
        using namespace fastllm_swa_layer;
        void *args[] = {&input, &weight, &projected, &hidden, &peers, &epochs[rank], &rank};
        cudaError_t status = cudaLaunchCooperativeKernel((const void *)OutputResidual,
            dim3(LayerBlocks), dim3(LayerThreads), args, 0, cudaStreamPerThread);
        if (status != cudaSuccess) return status;
        NormRouter4096<<<Experts, RouterThreads, 0, cudaStreamPerThread>>>(
            (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, nullptr);
        return cudaGetLastError();
    }
    bool PreparedRouterSupported(bool routed = false, int rows = 8) const {
        if (rows > 8) return rows <= 16 && !routed && widePreparedRouterSupported;
        return preparedRouterSupported && (!routed || routedRouterSupported);
    }
    bool PreparedVectorRouterSupported() const { return preparedVectorRouterSupported; }
    bool UseRowsDirectRouter(const void *hidden, const void *normWeight,
            const void *normed, const void *routerWeight, int rows) const {
        // Select only measured token-row sizes. The tiled kernel itself also
        // handles ragged/larger tiles; other shapes retain the established path.
        return rowsDirectRouterSupported && rows >= 9 && rows <= 16 &&
            (reinterpret_cast<uintptr_t>(hidden) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(normWeight) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(normed) % 16 == 0) &&
            (reinterpret_cast<uintptr_t>(routerWeight) % 16 == 0);
    }
    bool RowsSupported() const { return rowsSupported; }
    bool RowsMmaSupported() const { return rowsMmaSupported; }
    bool RowsGemmPrepared(int rank) const {
        return rowsGemmPrepared[rank].load(std::memory_order_acquire);
    }
    void MarkRowsGemmPrepared(int rank) {
        rowsGemmPrepared[rank].store(true, std::memory_order_release);
    }
    cudaError_t LaunchRows(const void *input, const void *weight, void *projected,
            void *hidden, const float *normWeight, void *normed, float eps,
            const float *routerWeight, float *router, int rank, int rows, bool mma = false, const FastllmMoePreparedInput *prepared = nullptr) {
        using namespace fastllm_swa_output;
        void *args[] = {&input, &weight, &projected, &hidden, &peers, &epochs[rank], &rank, &rows};
        cudaError_t status = cudaLaunchCooperativeKernel(OutputRowsKernel(rows, mma),
            dim3(MaxBlocks), dim3(RowThreads), args, 0, cudaStreamPerThread);
        if (status != cudaSuccess) return status;
        return LaunchRowsNorm(hidden, normWeight, normed, eps, routerWeight, router, rows, prepared);
    }
    cudaError_t LaunchRowsNorm(const void *hidden, const float *normWeight, void *normed,
            float eps, const float *routerWeight, float *router, int rows,
            const FastllmMoePreparedInput *prepared = nullptr, bool afterSwa = false) {
        using namespace fastllm_swa_output;
#if CUDART_VERSION >= 12000
        if (afterSwa && !prepared)
            return fastllm::cuda_launch::LaunchProgrammatic<NormRowsRouterExperts<8, false, true>>(
                dim3(32, 4), 256, 0, cudaStreamPerThread, (const BF16 *)hidden, routerWeight,
                router, normWeight, (BF16 *)normed, eps, rows, FastllmMoePreparedInput{});
#endif
        if (prepared && prepared->epRank>=0) {
#ifdef FASTLLM_ENABLE_NAIVE_FLEX_SM103
            return static_cast<cudaError_t>(fastshared_ep_prepared_norm(hidden,routerWeight,router,normWeight,normed,eps,*prepared,afterSwa));
#else
            return cudaErrorNotSupported;
#endif
        }
        if (routerWeight && prepared && prepared->routed) {
            FastllmMoePreparedInput input = *prepared;
            void *params[] = {&hidden, &routerWeight, &router, &normWeight,
                              &normed, &eps, &rows, &input};
            return cudaLaunchCooperativeKernel((const void *)NormRowsRouterRoutes,
                dim3(64, (rows + 1) / 2), dim3(256), params, 0, cudaStreamPerThread);
        }
        if (routerWeight) {
            if (prepared && rows > 8) {
                // Emit all sixteen rows, including zero padding for ragged tiles.
                NormRowsRouterDirect<8, 4, true, 16><<<dim3(32, 4), 256, 0, cudaStreamPerThread>>>(
                    (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, *prepared);
            } else if (prepared && preparedVectorRouterSupported && rows >= 2 && rows <= 8 &&
                (reinterpret_cast<uintptr_t>(hidden) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(normed) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(normWeight) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(routerWeight) % 16 == 0) &&
                (reinterpret_cast<uintptr_t>(prepared->values) % 8 == 0)) {
#if CUDART_VERSION >= 12000
                if (afterSwa)
                    return fastllm::cuda_launch::LaunchProgrammatic<NormRowsRouterDirect<8, 2, true, 8, false, true>>(
                        dim3(32, 4), 256, 0, cudaStreamPerThread, (const BF16 *)hidden, routerWeight, router,
                        normWeight, (BF16 *)normed, eps, rows, *prepared, (const BF16 *)nullptr, (BF16 *)nullptr);
#endif
                if (rows >= 7)
                    NormRowsRouterDirect<8><<<dim3(32, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                        (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, *prepared);
                else if (rows == 2)
                    NormRowsRouterPreparedVector<2><<<dim3(128, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                        (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, *prepared);
                else
                    NormRowsRouterPreparedVector<4><<<dim3(64, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                        (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, *prepared);
            } else if (prepared)
                NormRowsRouterExperts<4, true><<<dim3(64, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                    (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, *prepared);
            else if (UseRowsDirectRouter(hidden, normWeight, normed, routerWeight, rows))
                NormRowsRouterDirect<8, 4, false><<<dim3(32, (rows + 3) / 4), 256, 0, cudaStreamPerThread>>>(
                    (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows, {});
            else if (rows >= 3 && rowsRouterExpertsSupported)
                NormRowsRouterExperts<4><<<dim3(64, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                    (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows);
            else
                NormRowsRouter<<<dim3(256, (rows + 1) / 2), 256, 0, cudaStreamPerThread>>>(
                    (const BF16 *)hidden, routerWeight, router, normWeight, (BF16 *)normed, eps, rows);
        } else {
            NormRows<<<rows, 256, 0, cudaStreamPerThread>>>(
                (const BF16 *)hidden, normWeight, (BF16 *)normed, eps);
        }
        return cudaGetLastError();
    }
    bool VerifySupported(int rows, bool mx = false) const {
        if (mx) return rows >= 2 && rows <= 8 && verifySupported[3];
        return rows >= 1 && rows <= fastllm_swa_verify::RowLimit &&
            (rows <= 8 ? verifySupported[0] : verifySupported[1] || verifySupported[2]);
    }
    cudaError_t LaunchVerify(fastllm_swa_verify::Args args, int rank,
            const float *normWeight, const float *routerWeight, float *router,
            const FastllmMoePreparedInput *prepared = nullptr) {
        args.peers = peers; args.epochs = epochs[rank]; args.rank = rank;
        args.overlapRouter = args.rotary && args.rows == 8 && routerWeight &&
            (prepared ? swaRouterPdlSupported && prepared->overlapVerifyGate && (!prepared->routed || prepared->epRank>=0)
                      : unpreparedRouterPdlSupported);
        void *params[] = {&args};
        const bool wide = args.rows > 8 && verifySupported[2];
        const bool mx = args.outputScales != nullptr;
        const auto *kernel = fastllm_swa_verify::Kernel(args.rows, wide, mx);
        const int threads = fastllm_swa_verify::Threads(args.rows);
        const int shared = fastllm_swa_verify::SharedBytes(args.rows, wide, mx);
        cudaError_t status;
#if CUDART_VERSION >= 12000
        if (args.rotary) {
            cudaLaunchAttribute attrs[2]{};
            attrs[0].id = cudaLaunchAttributeCooperative;
            attrs[0].val.cooperative = 1;
            attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[1].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg{};
            cfg.gridDim = dim3(128); cfg.blockDim = dim3(threads);
            cfg.dynamicSmemBytes = shared; cfg.stream = cudaStreamPerThread;
            cfg.attrs = attrs; cfg.numAttrs = 2;
            status = cudaLaunchKernelExC(&cfg, kernel, params);
        } else
#endif
            status = cudaLaunchCooperativeKernel(kernel, dim3(128), dim3(threads), params, shared, cudaStreamPerThread);
        if (status != cudaSuccess) return status;
        return LaunchRowsNorm(args.hidden, normWeight, args.normed, args.eps,
                              routerWeight, router, args.rows, prepared, args.overlapRouter);
    }
#ifdef FASTLLM_ENABLE_NAIVE_DSA_FUSED
    cudaError_t LaunchDsa(FastllmNaiveDSALayerArgs args, int rank, const float *postNorm,
                          const float *routerWeight, float *router, const FastllmMoePreparedInput *prepared, bool deferPostNorm, const std::vector<FastllmNaiveDSALayerArgs> &tiles) {
        auto launch = [&](FastllmNaiveDSALayerArgs tile) {
            tile.rank = rank;
            tile.epochs = epochs[rank];
            for (int i = 0; i < 8; ++i) tile.peers[i] = peers.data[i];
            return FastllmCudaNaiveDSALayerLaunch(tile);
        };
        if (tiles.empty()) {
            if (auto status = launch(args); status != cudaSuccess) return status;
        } else for (const auto &tile : tiles) {
            if (auto status = launch(tile); status != cudaSuccess) return status;
        }
        if(deferPostNorm)return cudaSuccess;
        return LaunchRowsNorm(args.hidden, postNorm, args.normed, args.eps, routerWeight, router, args.rows,
                              prepared);
    }
#endif
    bool LayerSupported(bool mx = false) const { return mx ? mxLayerSupported : layerSupported; }
    cudaError_t LaunchLayer(fastllm_swa_layer::Args args, int rank) {
        using namespace fastllm_swa_layer;
        args.peers = peers;
        args.epochs = epochs[rank];
        args.rank = rank;
        void *params[] = {&args};
        cudaError_t status = cudaLaunchCooperativeKernel(
            args.qkvScales ? (const void *)Layer<true> : (const void *)Layer<false>, dim3(LayerBlocks), dim3(LayerThreads),
            params, args.qkvScales ? MxStagingBytes : StagingBytes, cudaStreamPerThread);
        if (status != cudaSuccess) return status;
        NormRouter4096<<<Experts, RouterThreads, 0, cudaStreamPerThread>>>(
            args.hidden, args.routerWeight, args.routerOut, args.postNorm,
            args.normed, args.eps, args.epochs);
        return cudaGetLastError();
    }
private:
    bool Initialize(const std::vector<int> &requested) {
        int original = 0;
        if (cudaGetDevice(&original) != cudaSuccess) return false;
        struct Restore { int device; ~Restore() { cudaSetDevice(device); } } restore{original};
        // Check every rank before allocating or publishing a backend. Compiling
        // a kernel is insufficient: the entire cooperative grid must be resident.
        for (int device : requested) {
            if (cudaSetDevice(device) != cudaSuccess) return false;
            cudaDeviceProp p{};
            int active = 0;
            cudaFuncAttributes a{};
            if (cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                p.major != 10 || p.minor != 3 || !p.cooperativeLaunch ||
                cudaFuncGetAttributes(&a, kernel) != cudaSuccess ||
                a.ptxVersion < 80 ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
                    kernel, 256, 0) != cudaSuccess ||
                active * p.multiProcessorCount < 128) return false;
            for (int peer : requested) if (peer != device) {
                int atomics = 0;
                if (cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                        device, peer) != cudaSuccess || !atomics) return false;
            }
        }
        // This optional implementation reuses the output mailbox. A capability
        // miss must not disable the established output-only backend.
        const bool stagesEnabled = fastllm_swa_settings::QkvEnabled() && fastllm_swa_settings::CoreEnabled();
        auto canLaunchLayer = [&](const void *full, int shared, int minPtx) {
            using namespace fastllm_swa_layer;
            if (!stagesEnabled) return false;
            for (int device : requested) {
                cudaDeviceProp p{}; cudaFuncAttributes attributes{}; int active = 0;
                if (cudaSetDevice(device) != cudaSuccess || cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                    cudaFuncSetAttribute(full, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess ||
                    cudaFuncGetAttributes(&attributes, full) != cudaSuccess || attributes.ptxVersion < minPtx ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, full, LayerThreads, shared) != cudaSuccess ||
                    active * p.multiProcessorCount < LayerBlocks) {
                    cudaGetLastError();
                    return false;
                }
            }
            return true;
        };
        layerSupported = canLaunchLayer((const void *)fastllm_swa_layer::Layer<false>, fastllm_swa_layer::StagingBytes, 80);
        mxLayerSupported = canLaunchLayer((const void *)fastllm_swa_layer::Layer<true>, fastllm_swa_layer::MxStagingBytes, 89);
        mxOutputSupported = canLaunchLayer((const void *)fastllm_swa_layer::MxOutput, 0, 89);
        // Every schedule reuses the established mailbox and must be resident
        // on all ranks. The direct-input query tile falls back to the original
        // N8 staging schedule when its required resources are unavailable.
        for (int variant = 0; variant < 4; ++variant) {
            verifySupported[variant] = stagesEnabled;
            if (!stagesEnabled) continue;
            const int rows = variant == 1 || variant == 2 ? 64 : 8;
            const void *entry = fastllm_swa_verify::Kernel(rows, variant == 2, variant == 3);
            const int threads = fastllm_swa_verify::Threads(rows);
            const int shared = fastllm_swa_verify::SharedBytes(rows, variant == 2, variant == 3);
            for (int device : requested) {
                cudaDeviceProp p{}; cudaFuncAttributes a{}; int active = 0;
                if (cudaSetDevice(device) != cudaSuccess ||
                    cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                    cudaFuncSetAttribute(entry, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess ||
                    cudaFuncGetAttributes(&a, entry) != cudaSuccess || a.ptxVersion < 80 ||
                    (variant == 3 && (p.major != 10 || p.minor != 3 || a.ptxVersion < 89)) ||
                    a.maxThreadsPerBlock < threads ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, entry, threads, shared) != cudaSuccess ||
                    active * p.multiProcessorCount < 128) {
                    verifySupported[variant] = false; cudaGetLastError();
                }
            }
        }
        outputRouterSupported = true;
        for (int device : requested) {
            using namespace fastllm_swa_layer;
            cudaDeviceProp p{}; int active = 0;
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, (const void *)OutputResidual,
                    LayerThreads, 0) != cudaSuccess || active * p.multiProcessorCount < LayerBlocks) {
                outputRouterSupported = false;
                cudaGetLastError();
            }
        }
        rowsSupported = true;
        for (int device : requested) for (int rows : {2, 4, 7, 8}) {
            using namespace fastllm_swa_output;
            cudaDeviceProp p{}; cudaFuncAttributes a{}; int active = 0;
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                cudaFuncGetAttributes(&a, OutputRowsKernel(rows)) != cudaSuccess ||
                a.ptxVersion < 80 ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, OutputRowsKernel(rows),
                    RowThreads, 0) != cudaSuccess || active * p.multiProcessorCount < MaxBlocks) {
                rowsSupported = false;
                cudaGetLastError();
            }
        }

        // This is optional and group-wide: a capability miss must keep the
        // ordered/GEMM path available, with the original mailbox allocation.
        rowsMmaSupported = rowsSupported;
        for (int device : requested) {
            using namespace fastllm_swa_output;
            cudaDeviceProp p{}; cudaFuncAttributes a{}; int active = 0;
            const void *mma = OutputRowsKernel(8, true);
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaGetDeviceProperties(&p, device) != cudaSuccess ||
                cudaFuncGetAttributes(&a, mma) != cudaSuccess || a.ptxVersion < 80 ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, mma, RowThreads, 0) != cudaSuccess ||
                active * p.multiProcessorCount < MaxBlocks) {
                rowsMmaSupported = false;
                cudaGetLastError();
            }
        }
        rowsRouterExpertsSupported = true;
        for (int device : requested) {
            cudaFuncAttributes attributes{};
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaFuncGetAttributes(&attributes, (const void *)fastllm_swa_output::NormRowsRouterExperts<4>) != cudaSuccess ||
                attributes.maxThreadsPerBlock < 256) {
                rowsRouterExpertsSupported = false;
                cudaGetLastError();
            }
        }
        rowsDirectRouterSupported = rowsRouterExpertsSupported;
        // Initialize already admitted SM103 on every rank. This optional
        // image needs its own probe so a miss leaves prepared paths intact.
        for (int device : requested) {
            cudaFuncAttributes attributes{};
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaFuncGetAttributes(&attributes,
                    (const void *)fastllm_swa_output::NormRowsRouterDirect<8, 4, false>) != cudaSuccess ||
                attributes.ptxVersion < 80 || attributes.maxThreadsPerBlock < 256) {
                rowsDirectRouterSupported = false;
                cudaGetLastError();
            }
        }
        widePreparedRouterSupported = rowsDirectRouterSupported;
        for (int device : requested) {
            cudaFuncAttributes attributes{};
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaFuncGetAttributes(&attributes,
                    (const void *)fastllm_swa_output::NormRowsRouterDirect<8, 4, true, 16>) != cudaSuccess ||
                attributes.ptxVersion < 80 || attributes.maxThreadsPerBlock < 256) {
                widePreparedRouterSupported = false;
                cudaGetLastError();
            }
        }
        preparedRouterSupported = rowsRouterExpertsSupported;
        for (int device : requested) {
            cudaFuncAttributes attributes{};
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaFuncGetAttributes(&attributes, (const void *)fastllm_swa_output::NormRowsRouterExperts<4, true>) != cudaSuccess ||
                attributes.ptxVersion < 80 || attributes.maxThreadsPerBlock < 256) {
                preparedRouterSupported = false;
                cudaGetLastError();
            }
        }
        routedRouterSupported = preparedRouterSupported;
        for (int device : requested) {
            cudaDeviceProp prop{}; cudaFuncAttributes attributes{}; int active = 0;
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaGetDeviceProperties(&prop, device) != cudaSuccess || !prop.cooperativeLaunch ||
                cudaFuncGetAttributes(&attributes, (const void *)fastllm_swa_output::NormRowsRouterRoutes) != cudaSuccess ||
                attributes.ptxVersion < 80 || attributes.maxThreadsPerBlock < 256 ||
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
                    (const void *)fastllm_swa_output::NormRowsRouterRoutes, 256, 0) != cudaSuccess ||
                active * prop.multiProcessorCount < 256) {
                routedRouterSupported = false; cudaGetLastError();
            }
        }
        // A missing optional kernel on any rank leaves the original prepared
        // router available. Limit performance selection to the measured SM;
        // no workspace, cooperative launch or request-time capability probe.
        preparedVectorRouterSupported = preparedRouterSupported;
        for (int device : requested) {
            cudaDeviceProp prop{};
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
                prop.major != 10 || prop.minor != 3) {
                preparedVectorRouterSupported = false;
                cudaGetLastError();
                continue;
            }
            for (const void *entry : {(const void *)fastllm_swa_output::NormRowsRouterPreparedVector<2>,
                    (const void *)fastllm_swa_output::NormRowsRouterPreparedVector<4>,
                    (const void *)fastllm_swa_output::NormRowsRouterDirect<8>}) {
                cudaFuncAttributes attributes{};
                if (cudaFuncGetAttributes(&attributes, entry) != cudaSuccess ||
                    attributes.ptxVersion < 80 || attributes.maxThreadsPerBlock < 256) {
                    preparedVectorRouterSupported = false;
                    cudaGetLastError();
                }
            }
        }
#if CUDART_VERSION >= 12000
        // Each rank must have PDL instructions in both producer and consumer.
        auto pdlKernel = [](const void *entry, int threads) {
            cudaFuncAttributes attr{};
            if (cudaFuncGetAttributes(&attr, entry) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            return attr.binaryVersion >= 90 && attr.ptxVersion >= 90 && attr.maxThreadsPerBlock >= threads;
        };
        swaRouterPdlSupported = verifySupported[3] && preparedVectorRouterSupported;
        unpreparedRouterPdlSupported = verifySupported[3] && rowsRouterExpertsSupported;
        for (int device : requested) {
            if (!swaRouterPdlSupported && !unpreparedRouterPdlSupported) break;
            if (cudaSetDevice(device) != cudaSuccess ||
                !pdlKernel(fastllm_swa_verify::Kernel(8, false, true), fastllm_swa_verify::Threads(8))) {
                swaRouterPdlSupported = unpreparedRouterPdlSupported = false;
                cudaGetLastError();
                break;
            }
            swaRouterPdlSupported = swaRouterPdlSupported &&
                pdlKernel((const void *)fastllm_swa_output::NormRowsRouterDirect<8, 2, true, 8, false, true>, 256);
            unpreparedRouterPdlSupported = unpreparedRouterPdlSupported &&
                pdlKernel((const void *)fastllm_swa_output::NormRowsRouterExperts<8, false, true>, 256);
        }
#endif
        devices = requested;
        for (int r = 0; r < 8; ++r) {
            cudaSetDevice(devices[r]);
            constexpr size_t bytes = fastllm_swa_output::Frames * fastllm_swa_output::FrameBytes;
            constexpr size_t counterBytes = fastllm_swa_output::MaxBlocks * fastllm_swa_output::MaxRows * sizeof(uint64_t);
            if (cudaMalloc(&peers.data[r], bytes) != cudaSuccess ||
                cudaMalloc(&epochs[r], counterBytes) != cudaSuccess ||
                cudaMemset(peers.data[r], 0, bytes) != cudaSuccess ||
                cudaMemset(epochs[r], 0, counterBytes) != cudaSuccess) return false;
        }
        // Worker PTDS streams must observe the coordinator's initialization.
        for (int device : devices)
            if (cudaSetDevice(device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return false;
        return true;
    }
    void Release(bool synchronize) {
        if (devices.empty()) return;
        int original = 0; cudaGetDevice(&original);
        if (synchronize) for (int device : devices) { cudaSetDevice(device); cudaDeviceSynchronize(); }
        for (size_t r = 0; r < devices.size(); ++r) {
            cudaSetDevice(devices[r]);
            cudaFree(peers.data[r]); cudaFree(epochs[r]);
            peers.data[r] = nullptr; epochs[r] = nullptr;
        }
        devices.clear(); cudaSetDevice(original);
    }
};
} // namespace
