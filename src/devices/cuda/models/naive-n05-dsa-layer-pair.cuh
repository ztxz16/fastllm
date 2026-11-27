// Two eight-row request tiles share projection weights, but have independent
// KV bindings, causal lengths, selection and partials. The score arena is
// reused for attention after the cluster selector completes.
struct PairedScratch : SplitLayout {
    static constexpr int kMaxSeqLen = 32768;
    static constexpr size_t Control = fastllm_dsa_fused::Align(kIndexLogitsOffset + size_t(8) * kMaxSeqLen * sizeof(float));
    static constexpr size_t Bytes = fastllm_dsa_fused::Align(Control + (9 + 8 + 16) * sizeof(unsigned));
};
static_assert(2 * PairedScratch::Bytes <= SplitLayout::kBytes);
static_assert(8 * fastllm_dsa_fused::native_attention::Scratch * sizeof(float) <= 8 * PairedScratch::kMaxSeqLen * sizeof(float));

static_assert(2 * 128 * sizeof(PackedGroup) <= Layout::Bytes - Layout::Indices);

struct PairedArgs { Args tile[2]; };

__global__ __launch_bounds__(Threads, 1) void PairedScores(const __grid_constant__ PairedArgs pair) {
    using Body = fastllm_dsa_fused::Body<true, false, PairedScratch>;
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &s = *reinterpret_cast<Storage<true> *>(bytes);
    int c = blockIdx.x, t = threadIdx.x;
    auto grid = cooperative_groups::this_grid();
    if (c < 16) {
        int row = c % 8;
        const auto &a = pair.tile[c / 8];
        if (row < a.rows) {
            if (c < 8) NormalizeOwner<true>(a, row, s.projection.input);
            else NormalizeOwner<true,8>(a, row, s.projection.input);
        }
    }
    if (c < 2 && t < 8) {
        const auto &a = pair.tile[c];
        const auto &binding = a.bindings[t];
        int n = t < a.rows ? max(0, min(binding.capacity, binding.liveKeys ? *binding.liveKeys : binding.length)) : 0;
        auto *lengths = reinterpret_cast<int *>(a.coreWorkspace + PairedScratch::Control);
        lengths[t] = n;
        const int longest = __reduce_max_sync(0xffu,n);
        if (t == 0) lengths[8] = longest;
        lengths[17 + t] = lengths[25 + t] = 0;
    }
    grid.sync();
    if (c < 137) ProjectPair<2>(pair.tile[0], reinterpret_cast<float *>(bytes), &pair.tile[1]);
    grid.sync();
    if (c < 48) {
        const auto &a = pair.tile[c / 24];
        int part = c % 24, row = part % 8;
        const int first = c < 24 ? 0 : pair.tile[0].rows;
        if (row < a.rows) {
            if (part < 8) Publish<false>(a, row, PublishPart::Qkv, first);
            else if (part < 16) RotateAndPack<true>(a, row, first);
            else WriteIndexKeyWarp(a, row, first);
        }
    }
    grid.sync();
    const auto &a = pair.tile[c % 2];
    const int stride = gridDim.x / 2, first = c / 2;
    auto *lengths = reinterpret_cast<const int *>(a.coreWorkspace + PairedScratch::Control);
    const int tiles = (lengths[8] + ScoreTileRows - 1) / ScoreTileRows;
    if (first < tiles) {
        auto core = CoreArguments(a, 0, 0);
        core.weights += (c % 2) * pair.tile[0].rows * 16;
        static_assert(sizeof(s.core.sweep) == sizeof(typename Body::SweepStorage));
        auto &sweep = *reinterpret_cast<typename Body::SweepStorage *>(&s.core.sweep);
        Body::score_tiles(core, lengths, sweep, 0, bytes + sizeof(Storage<true>),
                          first, stride, (tiles - first + stride - 1) / stride);
    }
}

__global__ __launch_bounds__(1024, 1) void PairedSelect(const __grid_constant__ PairedArgs pair) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    const auto &a = pair.tile[blockIdx.z];
    if (blockIdx.x >= a.rows) return; // All CTAs in this query cluster share x.
    const auto *scores = reinterpret_cast<const float *>(a.coreWorkspace + PairedScratch::kIndexLogitsOffset);
    int *ids = const_cast<int *>(a.bindings[0].indices);
    auto *lengths = reinterpret_cast<int *>(a.coreWorkspace + PairedScratch::Control);
    auto *retry = reinterpret_cast<unsigned *>(lengths + 9);
    cluster_sel::Select::run<PairedScratch::kMaxSeqLen>(scores, ids, retry, lengths,
        *reinterpret_cast<cluster_sel::Select::Shared *>(bytes));
    cooperative_groups::this_cluster().sync();
    const int q = blockIdx.x;
    if (blockIdx.y == 0 && retry[q])
        cluster_sel::RetryQuery(scores + size_t(q) * PairedScratch::kMaxSeqLen, ids + q * 2048,
                                lengths[q], *reinterpret_cast<cluster_sel::ExactRetry *>(bytes));
}

using PairOutputStorage = Storage<false,16,true>;
__global__ __launch_bounds__(Threads, 1) void PairedAttentionOutput(const __grid_constant__ PairedArgs pair) {
    extern __shared__ __align__(1024) unsigned char bytes[];
    auto &s = *reinterpret_cast<PairOutputStorage *>(bytes);
    const int c = blockIdx.x, t = threadIdx.x;
    if (c < 128 && c / 8 % 8 < pair.tile[c / 64].rows) {
        const auto &a = pair.tile[c / 64];
        const int row = c / 8 % 8;
        auto *part = reinterpret_cast<float *>(a.coreWorkspace + PairedScratch::kIndexLogitsOffset);
        auto *completion = reinterpret_cast<unsigned *>(a.coreWorkspace + PairedScratch::Control) + 17;
        namespace attn = fastllm_dsa_fused::native_attention;
        attn::Attend<true, attn::Phase::Fused, 8, 128, 256, true, true, PackOutputGroup, 8, false, 8>(
            a.query, a.bindings[row], a.sink, a.attention, part, completion, 8, 1, row, 0, c % 8,
            s.attention, PackOutputGroup{OutputGroups(a)});
    }
    cooperative_groups::this_grid().sync();
    if (c >= 128) return;
    int warp = t / 32, g = t % 32 / 4, q = t % 4;
    float acc[2][4]{};
    if (warp < 8) {
        int column = c * 32 + (warp / 4) * 16 + g;
        for (int kk = warp % 4 * 256; kk < (warp % 4 + 1) * 256; kk += 32) {
            auto w = fastllm_mxfp8_projection::LoadWeights(pair.tile[0].outputWeight, pair.tile[0].outputScales,
                                                            1024, column, column + 8, kk + q * 8);
            #pragma unroll
            for (int j = 0; j < 2; ++j) PreparedStep(acc[j], w, OutputGroups(pair.tile[j])[kk / 32], pair.tile[j].rows);
        }
    }
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (warp < 8) {
            auto *p = s.tail.partial + warp * 128 + g * 8 + q * 2;
            p[0] = acc[j][0]; p[1] = acc[j][1]; p[64] = acc[j][2]; p[65] = acc[j][3];
        }
        __syncthreads();
        if (t < 256 && t % 8 < pair.tile[j].rows) {
            int column = t / 128 * 16 + (t % 128) / 8;
            float v = 0.f;
            #pragma unroll
            for (int split = 0; split < 4; ++split)
                v += s.tail.partial[(t / 128 * 4 + split) * 128 + t % 128];
            s.tail.local[(j ? pair.tile[0].rows : 0) + t % 8][column] = __float2bfloat16_rn(v);
        }
        __syncthreads();
    }
    ReduceTail<16,true>(pair.tile[0], 0, s.tail, pair.tile[0].rows + pair.tile[1].rows);
}

static int PairedGrid(int device) {
    static thread_local std::unordered_map<int,int> cached;
    const auto found = cached.find(device);
    if (found != cached.end()) return found->second;
    if (FastllmCudaGraphIsCapturingFast() || FastllmCudaRuntimeArch() != 103) return 0;
    cudaDeviceProp props{};
    auto resident = [](const void *kernel, int bytes) {
        int blocks = 0;
        return cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes) == cudaSuccess &&
               cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Threads,bytes) == cudaSuccess && blocks > 0;
    };
    cudaLaunchAttribute attr{};
    auto cfg = cluster_sel::Config(attr, cudaStreamPerThread);
    cfg.gridDim.z = 2;
    int clusters = 0;
    const bool ok = cudaGetDeviceProperties(&props,device) == cudaSuccess && props.cooperativeLaunch &&
        props.multiProcessorCount >= 144 && props.multiProcessorCount % 2 == 0 &&
        resident((const void *)PairedScores, SplitScoreBytes<true>) &&
        resident((const void *)PairedAttentionOutput, sizeof(PairOutputStorage)) &&
        cudaFuncSetAttribute(PairedSelect,cudaFuncAttributeMaxDynamicSharedMemorySize,cfg.dynamicSmemBytes) == cudaSuccess &&
        cudaOccupancyMaxActiveClusters(&clusters,PairedSelect,&cfg) == cudaSuccess && clusters > 0;
    if (!ok) cudaGetLastError();
    return cached[device] = ok ? props.multiProcessorCount : 0;
}

static cudaError_t RunPairedSplit(const Args &a) {
    const int grid = PairedGrid(FastllmCudaGetDevice());
    if (!grid || a.requestCount != 2 || a.requests[0] < 1 || a.requests[0] > 8 ||
        a.requests[1] < 1 || a.requests[1] > 8 || a.rows != a.requests[0] + a.requests[1] ||
        a.phaseStop || a.fp8Key || !a.bindings[0].fp8) return cudaErrorNotSupported;
    PairedArgs p{{a,a}};
    for (int j = 0; j < 2; ++j) {
        auto &v = p.tile[j]; int off = j ? a.requests[0] : 0;
        v.requestCount = 1; v.rows = v.requests[0] = a.requests[j];
        v.positions += off; v.hidden += off * 4096; v.normed += off * 4096;
        v.projected += off * 4096; v.query += off * 1536; v.attention += off * 1024;
        if (v.feature) v.feature += off * 4096;
        if (v.branch) v.branch += off * 4096;
        v.coreWorkspace += j * PairedScratch::Bytes;
        for (int i = 0; i < v.rows; ++i) {
            v.bindings[i] = a.bindings[off + i];
            if (!v.bindings[i].fp8 || !v.bindings[0].indices || v.bindings[i].selectedCount != 2048 ||
                v.bindings[i].indices != v.bindings[0].indices + i * 2048 ||
                v.bindings[i].capacity > PairedScratch::kMaxSeqLen) return cudaErrorNotSupported;
        }
    }
    void *params[] = {&p};
    auto stream = cudaStreamPerThread;
    auto status = cudaLaunchCooperativeKernel((const void *)PairedScores,grid,Threads,params,SplitScoreBytes<true>,stream);
    if (status != cudaSuccess) return status;
    cudaLaunchAttribute attr{};
    auto cfg = cluster_sel::Config(attr,stream);
    cfg.gridDim.z = 2;
    status = cudaLaunchKernelEx(&cfg,PairedSelect,p);
    if (status != cudaSuccess) return status;
    return cudaLaunchCooperativeKernel((const void *)PairedAttentionOutput,grid,Threads,params,sizeof(PairOutputStorage),stream);
}
