// SM103 attention output path. Included by the native MXFP8 translation
// unit so O-B uses the same Tensor Core implementation as ordinary Linear.
namespace {
constexpr int outputGroups = 8, groupColumns = 4096, groupRank = 1024;
constexpr int attentionWidth = outputGroups * groupColumns, projectionWidth = outputGroups * groupRank;
using GroupedMaps = std::array<Desc, 5>;

auto GroupedOutputKernel() {
    return &dg::sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K, 32, 32, 32, 0, groupRank,
                                             groupColumns, 16, 128, 128, outputGroups, 128, 128, 128, 12, 2, 128,
                                             128, 2, true, kSMs, true, true, dg::GemmType::Batched, false,
                                             cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                             dg::epilogue::operators::CastToFP8WithClamp>;
}

bool GroupedDescriptor(Desc &d, void *p, int inner, int rows, int rowStride, int groupStride, int tileRows) {
    const cuuint64_t sizes[] = {uint64_t(inner), uint64_t(rows), outputGroups};
    const cuuint64_t strides[] = {uint64_t(rowStride), uint64_t(groupStride)};
    const cuuint32_t box[] = {128, uint32_t(tileRows), 1}, step[] = {1, 1, 1};
    return cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&d), CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, p, sizes,
                                  strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// Permute the eight heads of each group into FlashMLA's 32-channel order.
// Rebuilding from BF16 is permitted only after checking every value against
// its decoded FP8 representation. Native checkpoint codes keep their scale.
__global__ void PackOutputWeight(const __nv_bfloat16 *bf, const uint8_t *fp8, const float *sf, uint8_t *out,
                                 uint8_t *outSf, int *inexact, bool headMajor) {
    const int block = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    const int source = headMajor ? block : (block % 8) * 16 + block / 8;
    const int k = source * 32 + lane;
    uint8_t code, exponent;
    if (bf) {
        const float v = float(bf[size_t(row) * groupColumns + k]);
        float mx = fabsf(v);
        for (int offset = 16; offset; offset >>= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, offset));
        const uint32_t bits = __float_as_uint(mx / 448.f);
        exponent = min(254u, max(1u, (bits >> 23) + ((bits & 0x7fffff) != 0)));
        const float scale = __uint_as_float(uint32_t(exponent) << 23);
        __nv_fp8_e4m3 packed(v / scale);
        code = packed.__x;
        if (float(packed) * scale != v)
            atomicExch(inexact, 1);
    } else {
        code = fp8[size_t(row) * groupColumns + k];
        exponent = __float_as_uint(sf[(row / 32) * 128 + source]) >> 23;
    }
    out[size_t(row) * groupColumns + block * 32 + lane] = code;
    if (!lane) {
        const int group = row / groupRank, within = row % groupRank;
        outSf[((group * 32 + block / 4) * groupRank + within) * 4 + block % 4] = exponent;
    }
}

struct OutputPlan {
    std::shared_ptr<fastllm::Data> weight;
    fastllm::Data scratch;
    GroupedMaps grouped;
    int rows, pitch;
    bool panelMajor = false;
    void *codes, *scales, *bWeight, *bScales, *out;
};
} // namespace

std::shared_ptr<fastllm::Data> FastllmCudaV41PackOutputWeight(fastllm::Data &a, bool headMajor) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    if (a.dims != std::vector<int>{projectionWidth, groupColumns} || !a.cudaData || a.multiDeviceData ||
        a.dataDevice != CUDA || a.dataDeviceIds != std::vector<int>{device} || FastllmCudaGraphIsCapturing() ||
        FastllmCudaRuntimeArch() != 103 || (a.dataType != BFLOAT16 && a.dataType != FP8_E4M3) || a.IsRepacked)
        return {};
    const bool native = a.dataType == FP8_E4M3;
    if (native) {
        if (a.blockK != 32 || a.blockM != 32 || a.scales.size() != projectionWidth / 32 * 128)
            return {};
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(a, *GetEmptyData(), projectionWidth);
        if (a.extraCudaData.empty() || !FastllmCudaHasExactMxfp8Scales(a.extraCudaData[0], device))
            return {};
    }
    constexpr int bytes = projectionWidth * groupColumns, sfBytes = bytes / 32;
    auto packed = std::make_shared<Data>(INT8, std::vector<int>{bytes + sfBytes + 4});
    packed->ToDevice(CUDA, {device}, false);
    packed->Allocate(false);
    auto *raw = static_cast<uint8_t *>(packed->cudaData);
    auto *flag = reinterpret_cast<int *>(raw + bytes + sfBytes);
    cudaStream_t stream = cudaStreamPerThread;
    if (cudaMemsetAsync(flag, 0, sizeof(int), stream) != cudaSuccess)
        return {};
    PackOutputWeight<<<dim3(128, projectionWidth), 32, 0, stream>>>(
        native ? nullptr : static_cast<const __nv_bfloat16 *>(a.cudaData),
        native ? static_cast<const uint8_t *>(a.cudaData) : nullptr,
        native ? static_cast<const float *>(a.extraCudaData[0]) : nullptr, raw, raw + bytes, flag, headMajor);
    int error = 1;
    if (cudaGetLastError() != cudaSuccess ||
        cudaMemcpyAsync(&error, flag, sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess || error)
        return {};
    return packed;
}

std::shared_ptr<void> FastllmCudaV41PlanOutput(const std::shared_ptr<fastllm::Data> &a, fastllm::Data &b,
                                               fastllm::Data &output, int m, int pitch, void *q, void *sf) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    if (!a || !q || !sf || m < 1 || m > 128 || pitch < m || pitch % 4 || !b.cudaData || b.dataType != FP8_E4M3 ||
        b.dims != std::vector<int>{5120, projectionWidth} || b.blockK != 32 || b.blockM != 32 ||
        b.scales.size() != (5120 / 32) * (projectionWidth / 32) || b.IsRepacked || b.multiDeviceData ||
        b.dataDeviceIds != std::vector<int>{device} || b.dataDevice != CUDA || !output.cudaData ||
        output.dataDeviceIds != b.dataDeviceIds || output.dataType != BFLOAT16 ||
        output.GetBytes() < size_t(m) * 5120 * 2 || !Available(device, false))
        return {};
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(b, *GetEmptyData(), 5120);
    if (b.extraCudaData.empty() || !b.extraCudaData[0])
        return {};
    auto *bs = const_cast<uint32_t *>(FastllmCudaBlock32PackedScales(b, device, false));
    if (!bs || cudaFuncSetAttribute(GroupedOutputKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, 230204) !=
                   cudaSuccess)
        return {};
    auto p = std::make_shared<OutputPlan>();
    p->weight = a;
    p->rows = m;
    p->pitch = pitch;
    p->bWeight = b.cudaData;
    if (m <= 16 && UseDecodeProjections() && Projection<16, 1, false, true>::Configure()) {
        if (void *panels = FastllmCudaV41OutputPanels(b, device)) {
            p->bWeight = panels;
            p->panelMajor = true;
        }
    }
    p->bScales = bs;
    p->out = output.cudaData;
    p->scratch.dataType = INT8;
    p->scratch.UpdateUnitSize();
    p->scratch.Resize({pitch * (projectionWidth + projectionWidth / 32)});
    p->scratch.ToDevice(CUDA, {device}, false);
    p->scratch.Allocate(false);
    p->codes = p->scratch.cudaData;
    p->scales = static_cast<uint8_t *>(p->codes) + pitch * projectionWidth;
    auto *w = static_cast<uint8_t *>(a->cudaData);
    auto &d = p->grouped;
    if (!GroupedDescriptor(d[0], q, groupColumns, m, attentionWidth, groupColumns, 8) ||
        !GroupedDescriptor(d[1], w, groupColumns, groupRank, groupColumns, groupColumns * groupRank, 128) ||
        !Descriptor(d[2], sf, CU_TENSOR_MAP_DATA_TYPE_INT32, pitch, 32 * outputGroups, 4, 128, 1, false) ||
        !Descriptor(d[3], w + projectionWidth * groupColumns, CU_TENSOR_MAP_DATA_TYPE_INT32, groupRank,
                    32 * outputGroups, 4, 128, 1, false) ||
        !GroupedDescriptor(d[4], p->codes, groupRank, m, projectionWidth, groupRank, 16))
        return {};
    return p;
}

bool FastllmCudaV41RunOutput(const std::shared_ptr<void> &handle) {
    auto &p = *static_cast<OutputPlan *>(handle.get());
    auto &d = p.grouped;
    dg::EpilogueOperatorArgs ep{};
    ep.sfd = static_cast<uint32_t *>(p.scales);
    ep.sfd_stride = p.pitch;
    ep.shape_m = p.rows;
    ep.shape_n = groupRank;
    cudaLaunchAttribute cluster{};
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {2, 1, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kSMs);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 230204;
    cfg.stream = cudaStreamPerThread;
    cfg.attrs = &cluster;
    cfg.numAttrs = 1;
    if (cudaLaunchKernelEx(&cfg, GroupedOutputKernel(), (int *)nullptr, uint32_t(p.rows), uint32_t(groupRank),
                           uint32_t(groupColumns), ep, d[0], d[1], d[2], d[3], d[4]) != cudaSuccess)
        return false;
    auto run = V41SingleCtaProjection(p.rows, 5120, projectionWidth) ? Projection<16, 1>::Run : Projection<16, 2>::Run;
    if (p.panelMajor) run = Projection<16, 1, false, true>::Run;
    return run(p.codes, p.bWeight, p.scales, p.bScales, p.out, p.rows, 5120, projectionWidth, p.pitch);
}
