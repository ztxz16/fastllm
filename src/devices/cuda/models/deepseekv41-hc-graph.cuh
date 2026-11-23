#pragma once
// The caller supplies only mHC segments whose pre/post/comb outputs are first
// consumed at join[i], and the hidden input remains unchanged until that join.
// Each side branch owns its mix scratch; graph-pool temporaries may be reused.
namespace fastllm_v41_hc_graph {
// Keep the combined kernel ABI so its captured argument storage can be reused.
// The weight/mix arguments are unused by the normalization-only replacement.
template <bool Quantize>
__global__ void NormOnly(const __nv_bfloat16 *x, const float *, float *, const float *previousPre,
                         const float *normWeight, __nv_bfloat16 *out, __nv_bfloat16 *quantized, float eps) {
    V41HcPreNormBody<1024, 3, false, Quantize>(x, previousPre, normWeight, out, 4, 5120, eps, 0, nullptr,
                                               nullptr, nullptr, 0, 0.f, nullptr, nullptr, nullptr,
                                               quantized);
}
// Only an attention input segment can defer the current layer's mixing
// coefficients. Q/KV consume the previous coefficients and this norm. The
// attention residual consumes the new coefficients after the child has joined.
inline void *CloneSingleAttentionMix(void *source, float *scratch) {
    cudaGraph_t graph = nullptr;
    if (!source || !scratch || cudaGraphClone(&graph, static_cast<cudaGraph_t>(source)) != cudaSuccess)
        return nullptr;
    auto reject = [&]() -> void * { cudaGraphDestroy(graph); return nullptr; };
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return reject();
    cudaGraphNode_t dots = nullptr, mixNorm = nullptr;
    cudaKernelNodeParams dotParams{}, normParams{};
    bool quantized = false;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess) return reject();
        if (p.func == reinterpret_cast<void *>(V41HcWarpDots5120)) {
            if (dots || p.gridDim.x != 25 || p.gridDim.y != 1 || p.gridDim.z != 8) return reject();
            dots = node; dotParams = p;
        }
        const bool q = p.func == reinterpret_cast<void *>(V41HcPreNormKernel<1024, 3, true, true, 8>);
        if (q || p.func == reinterpret_cast<void *>(V41HcPreNormKernel<1024, 3, true, false, 8>)) {
            if (mixNorm || p.gridDim.x != 2 || p.blockDim.x != 1024) return reject();
            mixNorm = node; normParams = p; quantized = q;
        }
    }
    if (!dots || !mixNorm || !dotParams.kernelParams || !normParams.kernelParams ||
        *static_cast<void **>(dotParams.kernelParams[2]) != *static_cast<void **>(normParams.kernelParams[7]))
        return reject();
    cudaGraphNode_t successor = nullptr;
    if (FastllmCudaGraphNodeGetDependentNodes(dots, nullptr, &count) != cudaSuccess || count != 1 ||
        FastllmCudaGraphNodeGetDependentNodes(dots, &successor, &count) != cudaSuccess || successor != mixNorm ||
        FastllmCudaGraphNodeGetDependencies(mixNorm, nullptr, &count) != cudaSuccess || count != 1 ||
        FastllmCudaGraphNodeGetDependencies(dots, nullptr, &count) != cudaSuccess) return reject();
    std::vector<cudaGraphNode_t> before(count);
    if (count && FastllmCudaGraphNodeGetDependencies(dots, before.data(), &count) != cudaSuccess) return reject();
    // Each child is serialized with adjacent layers; its private storage
    // cannot alias the graph allocator's later projection scratch.
    void *dotArgs[] = {dotParams.kernelParams[0], dotParams.kernelParams[1], &scratch};
    dotParams.kernelParams = dotArgs;
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
    V41SelectStagedDots(1, dotParams);
#endif
    cudaGraphNode_t parallelDots = nullptr, finish = nullptr;
    if (cudaGraphAddKernelNode(&parallelDots, graph, before.data(), count, &dotParams) != cudaSuccess)
        return reject();
    void *finishArgs[16];
    std::copy_n(normParams.kernelParams, 16, finishArgs);
    finishArgs[7] = &scratch;
    auto finishParams = normParams;
    finishParams.kernelParams = finishArgs;
    finishParams.gridDim = dim3(1);
    finishParams.blockDim = dim3(32);
    if (cudaGraphAddKernelNode(&finish, graph, &parallelDots, 1, &finishParams) != cudaSuccess) return reject();
    normParams.func = quantized ? reinterpret_cast<void *>(V41HcPreNormKernel<1024, 3, false, true>) :
                                  reinterpret_cast<void *>(V41HcPreNormKernel<1024, 3, false, false>);
    normParams.gridDim = dim3(1);
    if (cudaGraphKernelNodeSetParams(mixNorm, &normParams) != cudaSuccess ||
        cudaGraphDestroyNode(dots) != cudaSuccess) return reject();
    for (auto dependency : before)
        if (FastllmCudaGraphAddDependencies(graph, &dependency, &mixNorm, 1) != cudaSuccess) return reject();
    return graph;
}
#ifdef FASTLLM_ENABLE_MXFP8_LINEAR_SM103
// The row normalization reads previousPre. The current Sinkhorn result is
// first consumed by the attention residual or by the post-MoE residual.
template <int Splits>
__global__ void FinishOwnerMix(const float *mix, const float *scale, const float *base,
        int iterations, float mixEps, float normEps, float *pre, float *post, float *comb) {
    const int row = blockIdx.x;
    if constexpr (Splits == 1) {
        V41HcFinishRow(mix, scale, base, 20480, iterations, mixEps, normEps, pre, post, comb, row);
    } else {
        __shared__ float sums[25];
        const int lane = threadIdx.x;
        if (lane < 25) {
            float value = 0;
#pragma unroll
            for (int s = 0; s < Splits; ++s) value += mix[(row * Splits + s) * 25 + lane];
            sums[lane] = value;
        }
        __syncwarp();
        V41HcFinishRow(sums, scale, base, 20480, iterations, mixEps, normEps,
                       pre + row * 4, post + row * 4, comb + row * 16, 0);
    }
}

// CUDA 13 reports lossy introspection when a programmatic edge is queried
// without its metadata. Child nodes retain those edges on clone. New producer
// edges use full completion, including when replacing a cuBLAS subgraph.
template <bool Parents>
inline cudaError_t OwnerNeighbors(cudaGraphNode_t node, cudaGraphNode_t *out, size_t *count) {
#if CUDART_VERSION >= 13000
    auto query = Parents ? &cudaGraphNodeGetDependencies : &cudaGraphNodeGetDependentNodes;
    if (out) {
        std::vector<cudaGraphEdgeData> edges(*count);
        return query(node, out, edges.data(), count);
    }
    size_t capacity = 4;
    for (;;) {
        std::vector<cudaGraphNode_t> nodes(capacity);
        std::vector<cudaGraphEdgeData> edges(capacity);
        *count = capacity;
        const auto status = query(node, nodes.data(), edges.data(), count);
        if (status != cudaSuccess || *count < capacity) return status;
        capacity *= 2;
    }
#else
    return Parents ? FastllmCudaGraphNodeGetDependencies(node, out, count)
                   : FastllmCudaGraphNodeGetDependentNodes(node, out, count);
#endif
}

struct OwnerMixGemm {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    void *workspace = nullptr;
    static constexpr size_t workspaceBytes = 32u << 20;
    ~OwnerMixGemm() {
        if (handle) cublasDestroy(handle);
        if (workspace) cudaFree(workspace);
        if (stream) cudaStreamDestroy(stream);
    }
    bool Prepare(bool needsGemm = true) {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (!needsGemm) return true;
        return cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS &&
            cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS &&
            cublasSetMathMode(handle, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH |
                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)) == CUBLAS_STATUS_SUCCESS &&
            cudaMalloc(&workspace, workspaceBytes) == cudaSuccess &&
            cublasSetWorkspace(handle, workspace, workspaceBytes) == CUBLAS_STATUS_SUCCESS;
    }
    cudaGraph_t Capture(const __nv_bfloat16 *hidden, const void *packed, int rows, float *mix) {
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return nullptr;
        float *projected = mix + (rows * 25 + 3) / 4 * 4;
        const bool ok = V41HcTensorGemm(handle, hidden, packed, rows, projected) == CUBLAS_STATUS_SUCCESS;
        if (ok) V41HcJoinTensorDots<<<rows, kHcThreads, 0, stream>>>(hidden, projected, mix);
        cudaGraph_t graph = nullptr;
        if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || !ok) {
            if (graph) cudaGraphDestroy(graph);
            return nullptr;
        }
        return graph;
    }
};

using OwnerGraph = std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)>;

// A marker precedes this operator's opaque cuBLAS nodes in the source graph.
// Replace precisely that serial interval. No cuBLAS argument ABI is inspected.
inline bool ForkTensorProjection(cudaGraph_t graph, cudaGraphNode_t marker,
        cudaGraphNode_t join, cudaGraphNode_t norm, int rows, float *scratch,
        void **ownerGemm, cudaGraphNode_t *producer, OwnerGraph *standalone) {
    cudaKernelNodeParams markerParams{};
    if (cudaGraphKernelNodeGetParams(marker, &markerParams) != cudaSuccess || !markerParams.kernelParams)
        return false;
    const auto *x = *static_cast<const __nv_bfloat16 **>(markerParams.kernelParams[0]);
    const auto *weight = *static_cast<const void **>(markerParams.kernelParams[1]);
    cudaKernelNodeParams joinParams{};
    if (cudaGraphKernelNodeGetParams(join, &joinParams) != cudaSuccess || !joinParams.kernelParams ||
        *static_cast<const __nv_bfloat16 **>(joinParams.kernelParams[0]) != x) return false;
    std::vector<cudaGraphNode_t> interval{marker};
    size_t n = 0;
    for (cudaGraphNode_t node = marker; node != join; ) {
        cudaGraphNode_t next = nullptr;
        if (OwnerNeighbors<false>(node, nullptr, &n) != cudaSuccess || n != 1 ||
            OwnerNeighbors<false>(node, &next, &n) != cudaSuccess ||
            OwnerNeighbors<true>(next, nullptr, &n) != cudaSuccess || n != 1 ||
            next == norm) return false;
        interval.push_back(next);
        node = next;
    }
    if (OwnerNeighbors<true>(marker, nullptr, &n) != cudaSuccess) return false;
    std::vector<cudaGraphNode_t> before(n);
    if (n && OwnerNeighbors<true>(marker, before.data(), &n) != cudaSuccess) return false;
    if (!*ownerGemm) {
        auto plan = std::make_unique<OwnerMixGemm>();
        if (!plan->Prepare()) return false;
        *ownerGemm = plan.release();
    }
    cudaGraph_t projection = static_cast<OwnerMixGemm *>(*ownerGemm)->Capture(x, weight, rows, scratch);
    if (!projection) return false;
    const cudaError_t status = cudaGraphAddChildGraphNode(producer, graph, before.data(), n, projection);
    if (standalone) standalone->reset(projection);
    else cudaGraphDestroy(projection);
    if (status != cudaSuccess) return false;
    for (auto node : interval)
        if (cudaGraphDestroyNode(node) != cudaSuccess) return false;
    for (auto parent : before)
        if (FastllmCudaGraphAddDependencies(graph, &parent, &norm, 1) != cudaSuccess) return false;
    return true;
}

#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
// FastLLM assembles request pages after Q-B. Attach the stats branch to the
// page producer inside the core graph so FlashMLA and mHC become siblings.
inline cudaGraph_t ForkAtMlaPages(cudaGraph_t source, cudaGraphNode_t stats, cudaGraphNode_t sinkhorn,
                                  int rows, cudaGraph_t standalone, bool verifying) {
    cudaGraph_t graph = nullptr;
    if (cudaGraphClone(&graph, source) != cudaSuccess) return nullptr;
    auto reject = [&]() -> cudaGraph_t { cudaGraphDestroy(graph); return nullptr; };
    size_t count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &count) != cudaSuccess || !count) return reject();
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(graph, nodes.data(), &count) != cudaSuccess) return reject();
    cudaGraphNode_t pages = nullptr;
    bool fusedVerify = false;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams params{};
        auto status = cudaGraphKernelNodeGetParams(node, &params);
        if (status == cudaErrorInvalidDeviceFunction) { cudaGetLastError(); continue; }
        if (status != cudaSuccess) return reject();
        if (verifying) {
            const bool packed = params.func == (void *)V41FullDecodePrepareKernel<true, false, true> ||
                                params.func == (void *)V41FullDecodePrepareKernel<false, false, true>;
            fusedVerify |= packed;
            if (!packed && params.func != (void *)V41FullVerifyPackWindow) continue;
            if ((packed ? params.gridDim.y : params.gridDim.x) != rows || !params.kernelParams || params.extra ||
                !static_cast<const FastllmV41FullDecodeConfig *>(params.kernelParams[0])->verifying)
                return reject();
        } else {
            if (!FastllmV41IsMlaPageKernel(params.func)) continue;
            if (params.gridDim.y != rows) return reject();
        }
        if (pages) return reject();
        pages = node;
    }
    if (!pages || OwnerNeighbors<false>(pages, nullptr, &count) != cudaSuccess || count != 1) return reject();
    // The stats branch reads the prelude residual, independently of KV.
    // Give it the page writer's predecessors in the cooperative verifier:
    // the small page writer can admit the dots before the resident MMA grid.
    std::vector<cudaGraphNode_t> after{pages};
    if (fusedVerify && rows == 6) {
        if (OwnerNeighbors<true>(pages, nullptr, &count) != cudaSuccess) return reject();
        after.resize(count);
        if (count && OwnerNeighbors<true>(pages, after.data(), &count) != cudaSuccess) return reject();
    }
    const auto parents = after.empty() ? nullptr : after.data();
    cudaGraphNode_t copied = nullptr;
    cudaGraphNodeType type;
    if (cudaGraphNodeGetType(stats, &type) != cudaSuccess) return reject();
    if (type == cudaGraphNodeTypeGraph) {
        // CUDA does not allow reparenting the read-only graph owned by a child
        // node. Keep the original cuBLAS capture alive until both copies exist.
        if (!standalone || cudaGraphAddChildGraphNode(&copied, graph, parents, after.size(), standalone) != cudaSuccess)
            return reject();
    } else {
        cudaKernelNodeParams params{};
        if (type != cudaGraphNodeTypeKernel || cudaGraphKernelNodeGetParams(stats, &params) != cudaSuccess ||
            cudaGraphAddKernelNode(&copied, graph, parents, after.size(), &params) != cudaSuccess) return reject();
    }
    cudaKernelNodeParams epilogue{};
    cudaGraphNode_t finish = nullptr;
    if (cudaGraphKernelNodeGetParams(sinkhorn, &epilogue) != cudaSuccess ||
        cudaGraphAddKernelNode(&finish, graph, &copied, 1, &epilogue) != cudaSuccess) return reject();
    return graph;
}
#endif

// The consumer expects the block-32 codes and transposed scale words in its
// graph-owned scratch. Fuse only an activation-only pack of this exact BF16 row.
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
// Non-source layers consume an immutable selection from the previous layer.
// Only the current ring record depends on Q/KV preparation. Its writer owns
// slot 127, so the other records may be gathered alongside the Q projections.
inline cudaGraph_t MoveHistoryToPrelude(cudaGraph_t core, cudaGraph_t prelude, cudaGraphNode_t *early) {
    cudaGraph_t raw = nullptr;
    if (cudaGraphClone(&raw, core) != cudaSuccess) return nullptr;
    OwnerGraph graph(raw, &cudaGraphDestroy);
    size_t n = 0;
    if (cudaGraphGetNodes(raw, nullptr, &n) != cudaSuccess) return nullptr;
    std::vector<cudaGraphNode_t> nodes(n);
    if (cudaGraphGetNodes(raw, nodes.data(), &n) != cudaSuccess) return nullptr;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess) { cudaGetLastError(); continue; }
        if (!FastllmV41IsMlaHistoryKernel(p.func)) continue;
        size_t parents = 0, users = 0;
        if (OwnerNeighbors<true>(node, nullptr, &parents) != cudaSuccess || parents != 1 ||
            OwnerNeighbors<false>(node, nullptr, &users) != cudaSuccess || !users) return nullptr;
        std::vector<cudaGraphNode_t> next(users);
        cudaGraphNode_t parent = nullptr;
        if (OwnerNeighbors<true>(node, &parent, &parents) != cudaSuccess ||
            OwnerNeighbors<false>(node, next.data(), &users) != cudaSuccess) return nullptr;
        for (auto user : next)
            if (FastllmCudaGraphAddDependencies(raw, &parent, &user, 1) != cudaSuccess) return nullptr;
        // Add the external root last so failure leaves the original graphs intact.
        if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess ||
            cudaGraphAddKernelNode(early, prelude, nullptr, 0, &p) != cudaSuccess) return nullptr;
        if (cudaGraphDestroyNode(node) != cudaSuccess) {
            cudaGraphDestroyNode(*early); *early = nullptr; return nullptr;
        }
        return graph.release();
    }
    return nullptr;
}
#endif

#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
// The direct-query kernel only consumes the normalized K value and immutable per-step
// request metadata. Its published ring slot is disjoint from the early history gather.
inline cudaGraph_t ForkNativeKV(cudaGraph_t core, cudaGraph_t prelude, cudaGraphNode_t *early, bool verify) {
    auto uniqueKernel = [](cudaGraph_t g, const void *symbol, cudaKernelNodeParams &params) {
        size_t count = 0;
        cudaGraphNode_t found = nullptr;
        if (cudaGraphGetNodes(g, nullptr, &count) != cudaSuccess) return found;
        std::vector<cudaGraphNode_t> nodes(count);
        if (cudaGraphGetNodes(g, nodes.data(), &count) != cudaSuccess) return found;
        for (auto node : nodes) {
            cudaGraphNodeType type;
            cudaKernelNodeParams candidate{};
            if (cudaGraphNodeGetType(node, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) continue;
            if (cudaGraphKernelNodeGetParams(node, &candidate) != cudaSuccess) { cudaGetLastError(); continue; }
            if (candidate.func == symbol && !candidate.extra && candidate.kernelParams) {
                if (found) return cudaGraphNode_t(nullptr);
                found = node;
                params = candidate;
            }
        }
        return found;
    };
    cudaKernelNodeParams norm{}, kv{};
    auto parent = uniqueKernel(prelude, verify ? (void *)V41QKVNormPackedBf16Kernel :
                                               (void *)V41QKVNormQuantKernel<2,true>, norm);
    if (!parent) return nullptr;
    cudaGraph_t raw = nullptr;
    if (cudaGraphClone(&raw, core) != cudaSuccess) return nullptr;
    OwnerGraph copy(raw, &cudaGraphDestroy);
    // A native-query verifier reads K and per-row metadata here. Q-B and
    // the index projections write independent buffers, consumed in the child.
    auto node = uniqueKernel(raw, verify ? (void *)V41FullDecodePrepareKernel<true,false,true> :
                                           (void *)V41FullDecodePrepareKernel<true>, kv);
    size_t count = 0;
    if (!node || OwnerNeighbors<true>(node, nullptr, &count) != cudaSuccess || count ||
        *static_cast<void **>(norm.kernelParams[verify ? 4 : 6]) != static_cast<FastllmV41FullDecodeBuffers *>(kv.kernelParams[1])->kv)
        return nullptr;
    if (cudaGraphAddKernelNode(early, prelude, &parent, 1, &kv) != cudaSuccess) return nullptr;
    if (cudaGraphDestroyNode(node) != cudaSuccess) {
        cudaGraphDestroyNode(*early);
        return nullptr;
    }
    return copy.release();
}
#endif

inline cudaGraph_t PackAttentionInput(cudaGraph_t source, cudaGraphNode_t original, int rows, bool quantized) {
    cudaGraph_t graph = nullptr;
    if (cudaGraphClone(&graph, source) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    auto reject = [&]() -> cudaGraph_t { cudaGraphDestroy(graph); cudaGetLastError(); return nullptr; };
    cudaGraphNode_t norm = nullptr, pack = nullptr, next = nullptr;
    size_t count = 0;
    cudaKernelNodeParams a{}, b{};
    if (cudaGraphNodeFindInClone(&norm, original, graph) != cudaSuccess ||
        OwnerNeighbors<false>(norm, nullptr, &count) != cudaSuccess || !count) return reject();
    std::vector<cudaGraphNode_t> readers(count);
    if (OwnerNeighbors<false>(norm, readers.data(), &count) != cudaSuccess) return reject();
    for (auto reader : readers) {
        cudaKernelNodeParams params{};
        if (cudaGraphKernelNodeGetParams(reader, &params) != cudaSuccess) { cudaGetLastError(); continue; }
        if (params.func == FastllmCudaV41PrepareBlock32Kernel()) {
            if (pack) return reject();
            pack = reader;
        }
    }
    if (!pack ||
        OwnerNeighbors<true>(pack, nullptr, &count) != cudaSuccess || count != 1 ||
        OwnerNeighbors<false>(pack, nullptr, &count) != cudaSuccess || count != 1 ||
        OwnerNeighbors<false>(pack, &next, &count) != cudaSuccess ||
        cudaGraphKernelNodeGetParams(norm, &a) != cudaSuccess ||
        cudaGraphKernelNodeGetParams(pack, &b) != cudaSuccess ||
        b.func != FastllmCudaV41PrepareBlock32Kernel() || !a.kernelParams || !b.kernelParams ||
        a.extra || b.extra || *static_cast<int *>(b.kernelParams[5]) != rows ||
        *static_cast<int *>(b.kernelParams[7]) != 5120 || *static_cast<int *>(b.kernelParams[10]) != 0 ||
        *static_cast<void **>(b.kernelParams[2]) != *static_cast<void **>(a.kernelParams[3])) return reject();
    void *args[19];
    std::copy_n(a.kernelParams, 16, args);
    args[16] = b.kernelParams[3]; args[17] = b.kernelParams[4]; args[18] = b.kernelParams[8];
    a.kernelParams = args;
    a.func = quantized ? (void *)V41HcPreNormPacked<true> : (void *)V41HcPreNormPacked<false>;
    if (cudaGraphKernelNodeSetParams(norm, &a) != cudaSuccess ||
        FastllmCudaGraphAddDependencies(graph, &norm, &next, 1) != cudaSuccess ||
        cudaGraphDestroyNode(pack) != cudaSuccess) return reject();
    return graph;
}

#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
inline cudaGraph_t ForkKeyNorm(cudaGraph_t source, cudaGraphNode_t original, bool keepQuery) {
    cudaGraph_t copy = nullptr;
    if (cudaGraphClone(&copy, source) != cudaSuccess) return nullptr;
    OwnerGraph graph(copy, &cudaGraphDestroy);
    cudaGraphNode_t norm = nullptr, key = nullptr;
    cudaKernelNodeParams q{}, k{};
    size_t size = 0;
    if (cudaGraphNodeFindInClone(&norm, original, copy) != cudaSuccess ||
        cudaGraphKernelNodeGetParams(norm, &q) != cudaSuccess || !q.kernelParams || q.extra ||
        OwnerNeighbors<false>(norm, nullptr, &size) != cudaSuccess) return nullptr;
    std::vector<cudaGraphNode_t> list(size);
    if (OwnerNeighbors<false>(norm, list.data(), &size) != cudaSuccess) return nullptr;
    for (auto child : list) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(child, &type) != cudaSuccess) return nullptr;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams candidate{};
        if (cudaGraphKernelNodeGetParams(child, &candidate) != cudaSuccess) return nullptr;
        if (candidate.func != (void *)V41FullDecodePrepareKernel<true>) continue;
        if (key) return nullptr;
        key = child; k = candidate;
    }
    if (!key || !k.kernelParams || k.extra || k.blockDim.x != 512 || k.gridDim.x != 1 ||
        OwnerNeighbors<true>(key, nullptr, &size) != cudaSuccess || size != 1) return nullptr;
    auto qa = q.kernelParams, ka = k.kernelParams;
    const auto &buffers = *static_cast<FastllmV41FullDecodeBuffers *>(ka[1]);
    if (*static_cast<void **>(qa[6]) != buffers.kv || *static_cast<int *>(qa[7]) != 1280 ||
        *static_cast<int *>(qa[8]) != 512 || k.gridDim.y * 2 != q.gridDim.x || *static_cast<int *>(ka[13]) != 0)
        return nullptr;
    V41DecodeKeyNorm args{*static_cast<const __nv_bfloat16 **>(qa[1]), *static_cast<const float **>(qa[3]),
                          *static_cast<int *>(qa[10]), *static_cast<float *>(qa[11])};
    void *parameters[19];
    std::copy_n(ka, 18, parameters); parameters[18] = &args;
    k.func = (void *)V41FullDecodePrepareKernel<true,true>; k.gridDim.x = 2; k.kernelParams = parameters;
    q.func = keepQuery ? (void *)V41QKVNormPack1280<false,true> : (void *)V41QKVNormPack1280<true,true>;
    if (cudaGraphKernelNodeSetParams(key, &k) != cudaSuccess || cudaGraphKernelNodeSetParams(norm, &q) != cudaSuccess ||
        OwnerNeighbors<true>(norm, nullptr, &size) != cudaSuccess || !size) return nullptr;
    list.resize(size);
    if (OwnerNeighbors<true>(norm, list.data(), &size) != cudaSuccess) return nullptr;
    for (auto producer : list)
        if (FastllmCudaGraphAddDependencies(copy, &producer, &key, 1) != cudaSuccess) return nullptr;
#if CUDART_VERSION >= 13000
    if (cudaGraphRemoveDependencies(copy, &norm, &key, nullptr, 1) != cudaSuccess) return nullptr;
#else
    if (cudaGraphRemoveDependencies(copy, &norm, &key, 1) != cudaSuccess) return nullptr;
#endif
    return graph.release();
}
#endif

inline cudaGraph_t PackQueryInput(cudaGraph_t source, int rows, bool keepQuery) {
    cudaGraph_t raw = nullptr;
    if (cudaGraphClone(&raw, source) != cudaSuccess) return nullptr;
    OwnerGraph graph(raw, &cudaGraphDestroy);
    size_t count = 0;
    if (cudaGraphGetNodes(raw, nullptr, &count) != cudaSuccess) return nullptr;
    std::vector<cudaGraphNode_t> nodes(count);
    if (cudaGraphGetNodes(raw, nodes.data(), &count) != cudaSuccess) return nullptr;
    for (auto norm : nodes) {
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(norm, &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams a{}, b{};
        if (cudaGraphKernelNodeGetParams(norm, &a) != cudaSuccess) { cudaGetLastError(); continue; }
        if (a.func != (void *)V41QKVNormQuantKernel<2,true> || !a.kernelParams || a.extra ||
            a.gridDim.x != rows * 2 || a.blockDim.x != 512) continue;
        cudaGraphNode_t pack = nullptr, next = nullptr;
        if (OwnerNeighbors<false>(norm, nullptr, &count) != cudaSuccess || !count) return nullptr;
        std::vector<cudaGraphNode_t> consumers(count);
        if (OwnerNeighbors<false>(norm, consumers.data(), &count) != cudaSuccess) return nullptr;
        for (auto child : consumers) {
            cudaKernelNodeParams candidate{};
            if (cudaGraphKernelNodeGetParams(child, &candidate) != cudaSuccess) { cudaGetLastError(); continue; }
            if (candidate.func == FastllmCudaV41PrepareBlock32Kernel()) {
                if (pack) return nullptr;
                pack = child;
            }
        }
        if (!pack ||
            OwnerNeighbors<true>(pack, nullptr, &count) != cudaSuccess || count != 1 ||
            OwnerNeighbors<false>(pack, nullptr, &count) != cudaSuccess || count != 1 ||
            OwnerNeighbors<false>(pack, &next, &count) != cudaSuccess ||
            cudaGraphKernelNodeGetParams(pack, &b) != cudaSuccess || !b.kernelParams || b.extra ||
            b.func != FastllmCudaV41PrepareBlock32Kernel()) return nullptr;
        const int dim = *static_cast<int *>(a.kernelParams[7]);
        void *input = *static_cast<void **>(b.kernelParams[2]);
        bool rounded = input == *static_cast<void **>(a.kernelParams[5]);
        if ((!rounded && input != *static_cast<void **>(a.kernelParams[4])) || dim % 128 ||
            dim != *static_cast<int *>(b.kernelParams[7]) || rows != *static_cast<int *>(b.kernelParams[5]) ||
            *static_cast<int *>(b.kernelParams[10]) != 0) return nullptr;
        void *args[16];
        std::copy_n(a.kernelParams, 12, args);
        args[12] = b.kernelParams[3]; args[13] = b.kernelParams[4]; args[14] = b.kernelParams[8]; args[15] = &rounded;
        a.func = (void *)V41QKVNormPackForB; a.kernelParams = args;
        if (dim == 1280 && *static_cast<int *>(a.kernelParams[8]) == 512 &&
            V41EnvOn("FASTLLM_DSV41_FIXED_QNORM"))
            a.func = keepQuery ? (void *)V41QKVNormPack1280<false> : (void *)V41QKVNormPack1280<true>;
        if (cudaGraphKernelNodeSetParams(norm, &a) != cudaSuccess ||
            FastllmCudaGraphAddDependencies(raw, &norm, &next, 1) != cudaSuccess ||
            cudaGraphDestroyNode(pack) != cudaSuccess) return nullptr;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
        if (V41EnvOn("FASTLLM_DSV41_FORK_KEY_NORM")) {
            if (auto fused = ForkKeyNorm(raw, norm, keepQuery)) return fused;
            cudaGetLastError();
        }
#endif
        return graph.release();
    }
    return nullptr;
}

inline void *ForkOwnerMix(const void *const *children, int count, int rows,
                          float *scratch, const void *postKernel, const void *prepareKernel, void **ownerGemm, bool keepQuery) {
    if (!children || count < 1 || rows < 1 || !scratch) return nullptr;
    cudaGraph_t graph = nullptr;
    if (cudaGraphClone(&graph, (cudaGraph_t)children[0]) != cudaSuccess) {
        cudaGetLastError(); // Captures containing allocation nodes cannot be cloned.
        return nullptr;
    }
    auto reject = [&]() -> void * {
        cudaGraphDestroy(graph);
        cudaGetLastError(); // optional composition must leave the captured fallback usable
        return nullptr;
    };
    size_t n = 0;
    if (cudaGraphGetNodes(graph, nullptr, &n) != cudaSuccess || !n) return reject();
    std::vector<cudaGraphNode_t> nodes(n), leaves;
    if (cudaGraphGetNodes(graph, nodes.data(), &n) != cudaSuccess) return reject();
    cudaGraphNode_t norm = nullptr, consumer = nullptr, marker = nullptr, dispatch = nullptr;
    cudaKernelNodeParams normal{};
    bool quantize = false, split = false;
    for (auto node : nodes) {
        size_t successors = 0;
        if (OwnerNeighbors<false>(node, nullptr, &successors) != cudaSuccess) return reject();
        if (!successors) leaves.push_back(node);
        cudaGraphNodeType type;
        if (cudaGraphNodeGetType(node, &type) != cudaSuccess) return reject();
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        const auto status = cudaGraphKernelNodeGetParams(node, &p);
        // cuBLAS may capture driver-loaded kernels with no CUDA Runtime host
        // stub. Their identities are outside our contract; retain them opaque.
        if (status == cudaErrorInvalidDeviceFunction) {
            cudaGetLastError();
            continue;
        }
        if (status != cudaSuccess) return reject();
        if (prepareKernel && p.func == prepareKernel) {
            if (dispatch) return reject();
            dispatch = node;
        }
        if (p.func == (void *)V41HcTensorMarker) {
            if (marker) return reject();
            marker = node;
        }
        if (postKernel && p.func == postKernel) {
            if (consumer) return reject();
            consumer = node;
        }
        const bool q8 = p.func == (void *)V41HcPreNormKernel<1024, 3, true, true, 8>;
        const bool s8 = q8 || p.func == (void *)V41HcPreNormKernel<1024, 3, true, false, 8>;
        const bool q1 = p.func == (void *)V41HcPreNormKernel<1024, 3, true, true>;
        if (s8 || q1 || p.func == (void *)V41HcPreNormKernel<1024, 3, true, false>) {
            if (norm || !p.kernelParams || p.extra || p.gridDim.x != 2 * rows ||
                p.blockDim.x != 1024 || *static_cast<int *>(p.kernelParams[5]) != 5120) return reject();
            norm = node; normal = p; quantize = q8 || q1; split = s8;
        }
    }
    if (!norm || (postKernel && !consumer) || leaves.empty()) return reject();
    cudaGraphNode_t producer = nullptr;
    if (OwnerNeighbors<true>(norm, nullptr, &n) != cudaSuccess || n != 1 ||
        OwnerNeighbors<true>(norm, &producer, &n) != cudaSuccess) return reject();
    cudaKernelNodeParams prod{};
    cudaGraphNodeType producerType;
    if (cudaGraphNodeGetType(producer, &producerType) != cudaSuccess || producerType != cudaGraphNodeTypeKernel ||
        cudaGraphKernelNodeGetParams(producer, &prod) != cudaSuccess || !prod.kernelParams || prod.extra)
        return reject();
    const bool warpDots = prod.func == (void *)V41HcWarpDots5120;
    // A small multi-row verifier uses one CTA per (row, dot), writing
    // the same 25 reduced FP32 values consumed by FinishOwnerMix<1>.
    const bool reducedDots = !postKernel && !split &&
        prod.func == (void *)V41HcDecodeDots<__nv_bfloat16> && prod.gridDim.x == 25 &&
        prod.gridDim.y == rows && prod.blockDim.x == kHcThreads &&
        *static_cast<int *>(prod.kernelParams[2]) == 4 * 5120;
    const bool plainDots = warpDots || reducedDots;
    const bool postDots = prod.func == (void *)V41HcPostWarpDots<5120, 2> ||
                          prod.func == (void *)V41HcPostWarpDots<5120, 4>;
    if (split ? !(postKernel ? postDots : plainDots) : !reducedDots && prod.func != (void *)V41HcJoinTensorDots) return reject();
    const int outputArg = postDots ? 6 : (reducedDots ? 3 : 2);
    if (*static_cast<void **>(prod.kernelParams[outputArg]) != *static_cast<void **>(normal.kernelParams[7]) ||
        OwnerNeighbors<false>(producer, nullptr, &n) != cudaSuccess || n != 1) return reject();

    // The original graph allocator could reuse mix storage immediately after
    // normalization. The detached branch instead owns a graph-lifetime buffer.
    const bool atFlash = !postKernel && count > 1 && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_FLASH_FORK", rows);
    OwnerGraph tensorSource(nullptr, &cudaGraphDestroy);
    if (marker && !split && ownerGemm) {
        cudaGraphNode_t separate = nullptr;
        if (!ForkTensorProjection(graph, marker, producer, norm, rows, scratch, ownerGemm, &separate, atFlash ? &tensorSource : nullptr))
            return reject();
        producer = separate;
        // Graph mutation may invalidate the prior node-parameter view.
        if (cudaGraphKernelNodeGetParams(norm, &normal) != cudaSuccess) return reject();
    } else {
        void *producerArgs[7];
        std::copy_n(prod.kernelParams, outputArg + 1, producerArgs);
        producerArgs[outputArg] = &scratch;
        prod.kernelParams = producerArgs;
        if (warpDots) V41SelectStagedDots(rows, prod);
        if (reducedDots && rows == 6 && V41IsSm<10, 3>())
            prod.func = (void *)V41HcDecodeDotsTiled<40>;
        if (cudaGraphKernelNodeSetParams(producer, &prod) != cudaSuccess) return reject();
    }

    void **a = normal.kernelParams;
    void *args[] = {&scratch, a[8], a[9], a[10], a[11], a[6], a[14], a[12], a[13]};
    cudaKernelNodeParams epilogue{};
    epilogue.func = split ? (void *)FinishOwnerMix<8> : (void *)FinishOwnerMix<1>;
    epilogue.gridDim = dim3(rows); epilogue.blockDim = dim3(32); epilogue.kernelParams = args;
    cudaGraphNode_t finish = nullptr;
    if (cudaGraphAddKernelNode(&finish, graph, &producer, 1, &epilogue) != cudaSuccess) return reject();
    normal.func = quantize ? (void *)V41HcPreNormKernel<1024, 3, false, true> :
                             (void *)V41HcPreNormKernel<1024, 3, false, false>;
    normal.gridDim = dim3(rows);
    if (cudaGraphKernelNodeSetParams(norm, &normal) != cudaSuccess) return reject();

    if (plainDots) {
        if (OwnerNeighbors<true>(producer, nullptr, &n) != cudaSuccess) return reject();
        std::vector<cudaGraphNode_t> before(n);
        if (n && OwnerNeighbors<true>(producer, before.data(), &n) != cudaSuccess) return reject();
#if CUDART_VERSION >= 13000
        cudaGraphEdgeData edge{};
        cudaGraphNode_t parent{};
        size_t edges = 1;
        if (cudaGraphNodeGetDependencies(norm, &parent, &edge, &edges) != cudaSuccess ||
            edges != 1 || parent != producer ||
            cudaGraphRemoveDependencies(graph, &producer, &norm, &edge, 1) != cudaSuccess) return reject();
#else
        if (cudaGraphRemoveDependencies(graph, &producer, &norm, 1) != cudaSuccess) return reject();
#endif
        for (auto parent : before)
            if (FastllmCudaGraphAddDependencies(graph, &parent, &norm, 1) != cudaSuccess) return reject();
    }
    if (FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_LATE_JOIN", rows)) {
        // Launch the independent branch beside the expensive kernel rather
        // than contesting SM admission with the short Q and router projections.
        // Normalization consumes postDots' residual and may never wait on
        // the delayed finish. A detached tensor GEMM has no such dependency.
        const auto delayed = (plainDots || (marker && !split && ownerGemm)) ? producer : finish;
        if (postKernel) {
            if (!dispatch || FastllmCudaGraphAddDependencies(graph, &dispatch, &delayed, 1) != cudaSuccess)
                return reject();
        } else {
            for (auto node : leaves)
                if (node == producer || node == finish ||
                    FastllmCudaGraphAddDependencies(graph, &node, &delayed, 1) != cudaSuccess) return reject();
        }
    }
    if (consumer && FastllmCudaGraphAddDependencies(graph, &finish, &consumer, 1) != cudaSuccess) return reject();
    // Attention's remaining child graphs consume only normalized Q/KV. Join
    // implicitly at the end of this composite, before the caller reads mix.
    for (int i = 1; i < count; ++i) {
        cudaGraphNode_t next = nullptr;
        cudaGraph_t part = (cudaGraph_t)children[i], moved = nullptr;
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
        if (i == 1 && atFlash) {
            // Norm must already be independent of these statistics. Only their
            // finish may consume the old producer, and finish has no reader here.
            cudaGraphNode_t only = nullptr;
            if (!(plainDots || (marker && !split && ownerGemm)) ||
                OwnerNeighbors<false>(producer, nullptr, &n) != cudaSuccess || n != 1 ||
                OwnerNeighbors<false>(producer, &only, &n) != cudaSuccess || only != finish ||
                OwnerNeighbors<false>(finish, nullptr, &n) != cudaSuccess || n != 0) return reject();
            moved = ForkAtMlaPages(part, producer, finish, rows, tensorSource.get(), reducedDots);
            if (moved) {
                if (cudaGraphDestroyNode(finish) != cudaSuccess || cudaGraphDestroyNode(producer) != cudaSuccess) {
                    cudaGraphDestroy(moved);
                    return reject();
                }
                part = moved;
            } else if (reducedDots) {
                cudaGetLastError(); // Retain the independent prelude branch for other verify layouts.
            } else return reject();
        }
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
        if (!reducedDots && i == 1 && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_EARLY_HISTORY", rows)) {
            cudaGraphNode_t history = nullptr;
            if (auto shortened = MoveHistoryToPrelude(part, graph, &history)) {
                if (moved) cudaGraphDestroy(moved);
                part = moved = shortened;
                leaves.push_back(history);
            } else cudaGetLastError();
        }
#endif
#ifdef FASTLLM_ENABLE_DSV41_FLASHMLA
        if (i == 1 && ((!reducedDots && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_DIRECT_QUERY", rows)) ||
                       (reducedDots && rows == 6))) {
            cudaGraphNode_t early = nullptr;
            if (auto shortened = ForkNativeKV(part, graph, &early, reducedDots)) {
                if (moved) cudaGraphDestroy(moved);
                part = moved = shortened;
                leaves.push_back(early);
            } else cudaGetLastError();
        }
#endif
        const auto status = cudaGraphAddChildGraphNode(&next, graph, leaves.data(), leaves.size(), part);
        if (moved) cudaGraphDestroy(moved);
        if (status != cudaSuccess) return reject();
        leaves.assign(1, next);
    }
    if (!reducedDots && !postKernel && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_PACK_ATTENTION", rows)) {
        if (auto packed = PackAttentionInput(graph, norm, rows, quantize)) {
            cudaGraphDestroy(graph);
            graph = packed;
        }
    }
    if (!reducedDots && !postKernel && FastllmV41HcEnabledAtRows("FASTLLM_DSV41_PACK_QUERY", rows)) {
        if (auto packed = PackQueryInput(graph, rows, keepQuery)) {
            cudaGraphDestroy(graph); graph = packed;
        } else cudaGetLastError();
    }
    return graph;
}
#endif

inline bool Compose(const void *const *children, int count, const int *start, const int *join, int branches,
                    float *saved, void **outGraph, void **outExec) {
    if (!children || count <= 0 || !start || !join || branches <= 0 || !saved || !outGraph || !outExec)
        return false;
    *outGraph = nullptr;
    *outExec = nullptr;
    std::vector<int> branchAt(count, -1);
    for (int b = 0; b < branches; ++b) {
        if (start[b] < 0 || start[b] >= count || join[b] <= start[b] || join[b] >= count ||
            branchAt[start[b]] >= 0)
            return false;
        branchAt[start[b]] = b;
    }
    cudaGraph_t graph = nullptr, cloned = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<cudaGraphNode_t> finishNodes(branches, nullptr);
    cudaGraphNode_t previous = nullptr;
    auto fail = [&] {
        if (cloned)
            cudaGraphDestroy(cloned);
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
        return false;
    };
    if (cudaGraphCreate(&graph, 0) != cudaSuccess)
        return false;
    for (int i = 0; i < count; ++i) {
        auto child = reinterpret_cast<cudaGraph_t>(const_cast<void *>(children[i]));
        const int b = branchAt[i];
        cudaGraphNode_t dotNode = nullptr;
        if (b >= 0) {
            if (cudaGraphClone(&cloned, child) != cudaSuccess) {
                return fail();
            }
            size_t n = 0;
            if (cudaGraphGetNodes(cloned, nullptr, &n) != cudaSuccess) {
                return fail();
            }
            std::vector<cudaGraphNode_t> nodes(n);
            if (cudaGraphGetNodes(cloned, nodes.data(), &n) != cudaSuccess) {
                return fail();
            }
            cudaGraphNode_t target = nullptr, combined = nullptr;
            cudaKernelNodeParams params{}, combinedParams{};
            bool quantize = false;
            for (auto node : nodes) {
                cudaGraphNodeType type;
                if (cudaGraphNodeGetType(node, &type) != cudaSuccess) {
                    return fail();
                }
                if (type != cudaGraphNodeTypeKernel)
                    continue;
                cudaKernelNodeParams candidate{};
                if (cudaGraphKernelNodeGetParams(node, &candidate) != cudaSuccess) {
                    return fail();
                }
                if (candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<false>) ||
                    candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<true>)) {
                    if (combined) {
                        return fail();
                    }
                    combined = node;
                    combinedParams = candidate;
                    quantize = candidate.func == reinterpret_cast<void *>(V41HcDotsPreNormKernel<true>);
                }
                if (candidate.func == reinterpret_cast<void *>(V41HcDecodeFinish)) {
                    if (target) {
                        return fail();
                    }
                    target = node;
                    params = candidate;
                }
            }
            if (!target || !combined || !combinedParams.kernelParams || combinedParams.extra ||
                combinedParams.gridDim.x != 26 || combinedParams.gridDim.y != 1 ||
                combinedParams.gridDim.z != 1 || combinedParams.blockDim.x != 1024 ||
                combinedParams.blockDim.y != 1 || combinedParams.blockDim.z != 1 || !params.kernelParams ||
                params.extra || params.gridDim.x != 1 || params.gridDim.y != 1 || params.gridDim.z != 1 ||
                params.blockDim.x != 32 || params.blockDim.y != 1 || params.blockDim.z != 1 ||
                params.sharedMemBytes != 0) {
                return fail();
            }
            const float *source = *static_cast<const float **>(params.kernelParams[0]);
            if (source != *static_cast<float **>(combinedParams.kernelParams[2])) {
                return fail();
            }
            size_t dependencies = 0;
            if (FastllmCudaGraphNodeGetDependencies(target, nullptr, &dependencies) != cudaSuccess ||
                dependencies != 1) {
                return fail();
            }
            cudaGraphNode_t dependency;
            if (FastllmCudaGraphNodeGetDependencies(target, &dependency, &dependencies) != cudaSuccess ||
                dependency != combined) {
                return fail();
            }
            float *destination = saved + b * 32;
            const auto *x = *static_cast<const __nv_bfloat16 **>(combinedParams.kernelParams[0]);
            const float *fn = *static_cast<const float **>(combinedParams.kernelParams[1]);
            int flatDim = 20480;
            void *dotArgs[] = {&x, &fn, &flatDim, &destination};
            cudaKernelNodeParams dot{};
            dot.func = reinterpret_cast<void *>(V41HcDecodeDotsTiled<16>);
            dot.gridDim = dim3(25);
            dot.blockDim = dim3(256);
            dot.kernelParams = dotArgs;
            if (cudaGraphAddKernelNode(&dotNode, graph, nullptr, 0, &dot) != cudaSuccess) {
                return fail();
            }
            void *args[10];
            for (int a = 0; a < 10; ++a)
                args[a] = params.kernelParams[a];
            args[0] = &destination;
            params.kernelParams = args;
            // AddKernelNode copies argument values before SetParams invalidates
            // the original node-owned argument storage.
            if (cudaGraphAddKernelNode(&finishNodes[b], graph, &dotNode, 1, &params) != cudaSuccess) {
                return fail();
            }
            combinedParams.func = quantize ? reinterpret_cast<void *>(NormOnly<true>)
                                           : reinterpret_cast<void *>(NormOnly<false>);
            combinedParams.gridDim = dim3(1);
            if (cudaGraphKernelNodeSetParams(combined, &combinedParams) != cudaSuccess) {
                return fail();
            }
            size_t successors = 0;
            if (FastllmCudaGraphNodeGetDependentNodes(target, nullptr, &successors) != cudaSuccess) {
                return fail();
            }
            std::vector<cudaGraphNode_t> dependent(successors);
            if (FastllmCudaGraphNodeGetDependentNodes(target, dependent.data(), &successors) != cudaSuccess) {
                return fail();
            }
            // All work following Finish still waits for normalization. The
            // caller joins the detached mix before any use of its outputs.
            if (cudaGraphDestroyNode(target) != cudaSuccess) {
                return fail();
            }
            for (auto node : dependent) {
                if (FastllmCudaGraphAddDependencies(cloned, &combined, &node, 1) != cudaSuccess) {
                    return fail();
                }
            }
            child = cloned;
        }
        std::vector<cudaGraphNode_t> deps;
        if (previous)
            deps.push_back(previous);
        for (int j = 0; j < branches; ++j)
            if (join[j] == i)
                deps.push_back(finishNodes[j]);
        cudaGraphNode_t next = nullptr;
        if (cudaGraphAddChildGraphNode(&next, graph, deps.data(), deps.size(), child) != cudaSuccess) {
            return fail();
        }
        if (cloned) {
            cudaGraphDestroy(cloned);
            cloned = nullptr;
        }
        if (b >= 0 && FastllmCudaGraphAddDependencies(graph, &next, &dotNode, 1) != cudaSuccess) {
            return fail();
        }
        previous = next;
    }
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
        return fail();
    }
    *outGraph = graph;
    *outExec = exec;
    return true;
}
} // namespace fastllm_v41_hc_graph
