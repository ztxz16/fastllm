/* NVFP4 routed MoE integration around vLLM's SM100/SM120 grouped W4A4 CUTLASS GEMM. */
#include "fastllm-cuda.cuh"
#define NVFP4_ENABLE_ELTS16 1
#include "../fp4/fastllm-nvfp4-utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

/** GPU并行统计每个expert收到的有效路由数量。 */
__global__ void CountRoutes(const int32_t *indices, int *expertCounts,
                            int routes, int experts) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) return;
    const int expert = indices[route];
    if (expert >= 0 && expert < experts) {
        atomicAdd(expertCounts + expert, 1);
    }
}

/** 在GPU上生成路由前缀和、128行scale前缀和及scatter游标。 */
__global__ void BuildRouteOffsets(const int *expertCounts, int *expertOffsets,
                                  int *blockscaleOffsets, int *expertCursors,
                                  int experts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int routeOffset = 0;
    int scaleOffset = 0;
    expertOffsets[0] = 0;
    blockscaleOffsets[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
        expertCursors[expert] = routeOffset;
        routeOffset += expertCounts[expert];
        scaleOffset += (expertCounts[expert] + 127) / 128 * 128;
        expertOffsets[expert + 1] = routeOffset;
        blockscaleOffsets[expert + 1] = scaleOffset;
    }
}

/** 按expert连续重排[token,topk]路由，同时保留反向位置和路由权重。 */
__global__ void ScatterRoutes(const int32_t *indices, const float *scores,
                              int *expertCursors, int *routeRows,
                              float *routeScales, int *routePositions,
                              int routes, int topk, int experts) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) return;
    const int expert = indices[route];
    if (expert < 0 || expert >= experts) return;
    const int position = atomicAdd(expertCursors + expert, 1);
    routeRows[position] = route / topk;
    routeScales[position] = scores[route];
    routePositions[route] = position;
}

/**
 * 用一个kernel量化全部expert的连续路由行。
 *
 * expertOffsets定位每行所属expert；blockscaleOffsets保证每个expert的
 * E4M3 scale区域独立按128行对齐。FusedSiluMul为true时输入采用
 * [gate,up]布局并在量化前融合SwiGLU。
 */
template <typename T, bool FusedSiluMul>
__global__ void __launch_bounds__(512)
QuantizeExpertRows(const T *input, uint8_t *output, uint8_t *outputScales,
                   const int *expertOffsets,
                   const int *blockscaleOffsets, int rows, int columns,
                   int experts) {
    const int pack = blockIdx.y * blockDim.x + threadIdx.x;
    const int packs = columns / fastllm_nvfp4::kElementsPerThread;
    if (pack >= packs) return;
    const int roundedScaleColumns = ((columns / 16) + 3) / 4 * 4;
    const int kTiles = (columns + 63) / 64;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        int left = 0, right = experts - 1, expert = 0;
        while (left <= right) {
            const int middle = (left + right) / 2;
            const int begin = expertOffsets[middle];
            const int end = expertOffsets[middle + 1];
            if (row < begin) right = middle - 1;
            else if (row >= end) left = middle + 1;
            else { expert = middle; break; }
        }
        const int rowInExpert = row - expertOffsets[expert];
        const int element = pack * fastllm_nvfp4::kElementsPerThread;
        fastllm_nvfp4::PackedVec<T> values;
        if constexpr (FusedSiluMul) {
            fastllm_nvfp4::PackedVec<T> gate, up;
            const T *rowInput = input + (size_t)row * columns * 2;
            fastllm_nvfp4::LoadOrZero(gate, rowInput + element, true);
            fastllm_nvfp4::LoadOrZero(up, rowInput + columns + element, true);
            values = fastllm_nvfp4::SiluMul(gate, up);
        } else {
            fastllm_nvfp4::LoadOrZero(
                values, input + (size_t)row * columns + element, true);
        }
        uint8_t *expertScales = outputScales +
            (size_t)blockscaleOffsets[expert] * roundedScaleColumns;
        uint8_t *scale = fastllm_nvfp4::QuantScaleAddress(
            rowInExpert, pack, kTiles, expertScales);
        const auto packed = fastllm_nvfp4::Quantize(values, 1.0f, scale);
        fastllm_nvfp4::StoreFp4(
            output + (size_t)row * (columns / 2) + element / 2, packed);
    }
}

template <typename T>
__global__ void GatherRows(const T *input, const int *routeRows, T *output,
                           int rows, int columns) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)rows * columns;
    if (index < count) {
        int row = int(index / columns);
        int col = int(index - (size_t)row * columns);
        output[index] = input[(size_t)routeRows[row] * columns + col];
    }
}

template <typename T>
__global__ void ReduceRoutes(const T *parts, const int *routePositions,
                             const float *routeScales, T *output,
                             int batch, int topk, int hidden) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * hidden;
    if (index >= count) return;
    int token = int(index / hidden);
    int col = int(index - (size_t)token * hidden);
    float value = 0.0f;
    for (int slot = 0; slot < topk; ++slot) {
        int route = routePositions[token * topk + slot];
        if constexpr (std::is_same_v<T, half>) {
            value += __half2float(parts[(size_t)route * hidden + col]) * routeScales[route];
        } else {
            value += __bfloat162float(parts[(size_t)route * hidden + col]) * routeScales[route];
        }
    }
    if constexpr (std::is_same_v<T, half>) output[index] = __float2half_rn(value);
    else output[index] = __float2bfloat16_rn(value);
}

/**
 * 对grouped MoE第一层输出执行GeGLU，供后续独立NVFP4量化使用。
 *
 * vLLM仅为SwiGLU提供“激活+FP4量化”融合kernel；GeGLU走普通门控激活
 * 后再独立量化。本kernel实现其中的普通GeGLU阶段，并与grouped GEMM使用
 * 同一CUDA流，避免借用Data高级接口时改变流边界或产生额外同步。
 *
 * 输入每行按[gate, up]连续存放，数学语义为：
 * output[row,col] = GELU(input[row,col]) * input[row,hidden+col]。
 * GELU采用FastLLM现有GeGLU相同的erf精确形式。
 *
 * @tparam T      输入输出类型，仅支持half或__nv_bfloat16。
 * @param input   第一层输出，逻辑形状为[rows, 2 * hidden]。
 * @param output  GeGLU结果，逻辑形状为[rows, hidden]。
 * @param rows    当前expert分到的路由行数。
 * @param hidden  GeGLU输出宽度。
 */
template <typename T>
__global__ void GegluRows(const T *input, T *output, int rows, int hidden) {
    const size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (size_t)rows * hidden;
    if (index >= count) return;
    const int row = int(index / hidden);
    const int column = int(index - (size_t)row * hidden);
    const size_t inputOffset = (size_t)row * hidden * 2 + column;
    float gate, up;
    if constexpr (std::is_same_v<T, half>) {
        gate = __half2float(input[inputOffset]);
        up = __half2float(input[inputOffset + hidden]);
        output[index] = __float2half_rn(
            gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    } else {
        gate = __bfloat162float(input[inputOffset]);
        up = __bfloat162float(input[inputOffset + hidden]);
        output[index] = __float2bfloat16_rn(
            gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

/**
 * Grouped NVFP4 MoE在一张GPU、一个调用线程上的持久临时缓冲区。
 *
 * 路由表、gather结果以及两层激活量化结果均按需扩容并跨调用复用，避免
 * 每个token步、每个活跃expert重复cudaMalloc/cudaFree。scratch使用
 * thread_local隔离CPU调用线程；同一线程再按GPU编号隔离，和
 * cudaStreamPerThread的生命周期保持一致。
 */
struct MoeScratch {
    int *expertCounts = nullptr;
    int *expertOffsets = nullptr;
    int *blockscaleOffsets = nullptr;
    int *expertCursors = nullptr;
    int *routeRows = nullptr;
    int *routePositions = nullptr;
    float *routeScales = nullptr;
    uint8_t *gathered = nullptr;
    uint8_t *gateValues = nullptr;
    uint8_t *gateScales = nullptr;
    uint8_t *downValues = nullptr;
    uint8_t *downScales = nullptr;
    uint8_t *activated = nullptr;
    size_t expertCountsCapacity = 0;
    size_t expertOffsetsCapacity = 0;
    size_t blockscaleOffsetsCapacity = 0;
    size_t expertCursorsCapacity = 0;
    size_t routeRowsCapacity = 0;
    size_t routePositionsCapacity = 0;
    size_t routeScalesCapacity = 0;
    size_t gatheredCapacity = 0;
    size_t gateValueCapacity = 0;
    size_t gateScaleCapacity = 0;
    size_t downValueCapacity = 0;
    size_t downScaleCapacity = 0;
    size_t activatedCapacity = 0;
};

static thread_local std::map<int, MoeScratch> moeScratch;

struct MoeLayerWeightMetadata {
    std::vector<const uint8_t *> gateWeights;
    std::vector<const uint8_t *> gateScales;
    std::vector<const float *> gateAlphas;
    std::vector<const uint8_t *> downWeights;
    std::vector<const uint8_t *> downScales;
    std::vector<const float *> downAlphas;
};

using MoeLayerWeightKey = std::pair<const fastllm::Data *, int>;
static thread_local std::map<MoeLayerWeightKey, MoeLayerWeightMetadata>
    moeLayerWeightMetadata;

/**
 * 获取本层routed expert的CUTLASS权重指针表，并缓存稳定热路径结果。
 *
 * 每次调用只重新查询首个gate权重以检测GPU迁移或cache重建；首指针保持
 * 不变时直接返回整层指针表，避免每个token对全部expert重复执行全局
 * weight cache map和mutex查询。首指针变化时重新扫描整层，保证不会使用
 * 已释放的重排权重地址。
 *
 * @param weights       MoE权重数组。
 * @param weightsBatch  权重指针数量。
 * @param hidden        模型隐藏维度。
 * @param inter         expert中间维度。
 * @return 成功时返回当前GPU上的整层指针表，否则返回nullptr。
 */
static MoeLayerWeightMetadata *GetLayerWeightMetadata(
        fastllm::Data **weights, int weightsBatch, int hidden, int inter) {
    const int experts = weightsBatch / 2 - 1;
    const int device = FastllmCudaGetDevice();
    if (!weights || experts <= 0 || device < 0 || !weights[2]) return nullptr;
    const uint8_t *firstWeight = nullptr, *firstScale = nullptr;
    const float *firstAlpha = nullptr;
    if (!FastllmCudaPrepareNvfp4W4A4Weight(
            *weights[2], hidden, inter * 2,
            &firstWeight, &firstScale, &firstAlpha)) return nullptr;
    MoeLayerWeightMetadata &metadata =
        moeLayerWeightMetadata[{weights[2], device}];
    if ((int)metadata.gateWeights.size() == experts &&
        metadata.gateWeights[0] == firstWeight) {
        return &metadata;
    }
    metadata = {};
    metadata.gateWeights.resize(experts);
    metadata.gateScales.resize(experts);
    metadata.gateAlphas.resize(experts);
    metadata.downWeights.resize(experts);
    metadata.downScales.resize(experts);
    metadata.downAlphas.resize(experts);
    for (int expert = 0; expert < experts; ++expert) {
        fastllm::Data *gate = weights[2 + expert * 2];
        fastllm::Data *down = weights[3 + expert * 2];
        if (!gate || !down ||
            !FastllmCudaPrepareNvfp4W4A4Weight(
                *gate, hidden, inter * 2, &metadata.gateWeights[expert],
                &metadata.gateScales[expert], &metadata.gateAlphas[expert]) ||
            !FastllmCudaPrepareNvfp4W4A4Weight(
                *down, inter, hidden, &metadata.downWeights[expert],
                &metadata.downScales[expert], &metadata.downAlphas[expert])) {
            metadata = {};
            return nullptr;
        }
    }
    return &metadata;
}

template <typename T>
static bool EnsureScratchBuffer(T *&buffer, size_t &capacity, size_t bytes) {
    if (buffer != nullptr && capacity >= bytes) return true;
    T *replacement = static_cast<T *>(FastllmCudaMalloc(bytes));
    if (replacement == nullptr) return false;
    if (buffer != nullptr) FastllmCudaFree(buffer);
    buffer = replacement;
    capacity = bytes;
    return true;
}

/** 获取并扩容当前GPU/线程的MoE scratch；正式计算阶段不再释放。 */
static MoeScratch *GetMoeScratch(size_t routeCount, size_t gatheredBytes,
                                 size_t gateValueBytes, size_t gateScaleBytes,
                                 size_t downValueBytes, size_t downScaleBytes,
                                 size_t activatedBytes, int experts) {
    const int device = FastllmCudaGetDevice();
    MoeScratch &scratch = moeScratch[device];
    const size_t routeBytes = routeCount * sizeof(int);
    const size_t expertBytes = (size_t)(experts + 1) * sizeof(int);
    if (FastllmCudaGraphIsCapturing() &&
        (scratch.expertCountsCapacity < expertBytes ||
         scratch.expertOffsetsCapacity < expertBytes ||
         scratch.blockscaleOffsetsCapacity < expertBytes ||
         scratch.expertCursorsCapacity < expertBytes ||
         scratch.routeRowsCapacity < routeBytes ||
         scratch.routePositionsCapacity < routeBytes ||
         scratch.routeScalesCapacity < routeCount * sizeof(float) ||
         scratch.gatheredCapacity < gatheredBytes ||
         scratch.gateValueCapacity < gateValueBytes ||
         scratch.gateScaleCapacity < gateScaleBytes ||
         scratch.downValueCapacity < downValueBytes ||
         scratch.downScaleCapacity < downScaleBytes ||
         scratch.activatedCapacity < activatedBytes)) {
        return nullptr;
    }
    if (!EnsureScratchBuffer(scratch.expertCounts, scratch.expertCountsCapacity, expertBytes) ||
        !EnsureScratchBuffer(scratch.expertOffsets, scratch.expertOffsetsCapacity, expertBytes) ||
        !EnsureScratchBuffer(scratch.blockscaleOffsets, scratch.blockscaleOffsetsCapacity, expertBytes) ||
        !EnsureScratchBuffer(scratch.expertCursors, scratch.expertCursorsCapacity, expertBytes) ||
        !EnsureScratchBuffer(scratch.routeRows, scratch.routeRowsCapacity, routeBytes) ||
        !EnsureScratchBuffer(scratch.routePositions, scratch.routePositionsCapacity, routeBytes) ||
        !EnsureScratchBuffer(scratch.routeScales, scratch.routeScalesCapacity,
                             routeCount * sizeof(float)) ||
        !EnsureScratchBuffer(scratch.gathered, scratch.gatheredCapacity, gatheredBytes) ||
        !EnsureScratchBuffer(scratch.gateValues, scratch.gateValueCapacity, gateValueBytes) ||
        !EnsureScratchBuffer(scratch.gateScales, scratch.gateScaleCapacity, gateScaleBytes) ||
        !EnsureScratchBuffer(scratch.downValues, scratch.downValueCapacity, downValueBytes) ||
        !EnsureScratchBuffer(scratch.downScales, scratch.downScaleCapacity, downScaleBytes) ||
        !EnsureScratchBuffer(scratch.activated, scratch.activatedCapacity, activatedBytes)) {
        return nullptr;
    }
    return &scratch;
}

static bool Enabled() {
    const char *value = std::getenv("FASTLLM_CUDA_MOE_NVFP4_W4A4");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool StrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_MOE_NVFP4_W4A4_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/**
 * 执行一次NVFP4 W4A4 CUTLASS grouped MoE计算。
 *
 * 本函数只负责单次设备计算，不选择或固定MoE后端，也不处理失败后的
 * CUTLASS cache销毁和原始权重恢复。执行流程为：校验输入、路由和全部
 * routed expert权重；将input按expert聚集；动态量化激活；执行gate/up
 * grouped W4A4 GEMM；完成SwiGLU或GeGLU；再次动态量化并执行down
 * grouped W4A4 GEMM；最后按照routePositions和routeScales归并top-k结果。
 *
 * 数学语义为：对每个token x及其选中的expert e，计算
 * down_e(activation(gate_up_e(x)))，乘以对应路由权重后在top-k维求和。
 * input和output逻辑形状为[batch, hidden]；gate/up权重形状为
 * [2 * inter, hidden]；down权重形状为[hidden, inter]。权重必须采用
 * NVFP4_BLOCK_16，hidden和inter必须按32对齐。SM120仅接受BF16，
 * SM100路径可接受FP16或BF16。CUDA Graph捕获期间不在此路径首次执行。
 *
 * @tparam T              激活及输出的CUDA元素类型，仅支持half或
 *                        __nv_bfloat16，并且必须与input.dataType一致。
 * @param input           CUDA输入张量，逻辑形状为[batch, hidden]。
 * @param w1              gate/up grouped GEMM临时输出；函数内调整为
 *                        [totalTasks, 2 * inter]。
 * @param w2              down grouped GEMM临时输出；函数内调整为
 *                        [totalTasks, hidden]。
 * @param output          最终加权归并结果，逻辑形状与input一致。
 * @param weights         MoE权重指针数组；前两个槽位保留给shared expert，
 *                        routed expert e使用weights[2 + 2 * e]作为gate/up，
 *                        weights[3 + 2 * e]作为down。
 * @param weightsBatch    weights中的指针数量，routed expert数量为
 *                        weightsBatch / 2 - 1。
 * @param routeRows       按expert连续分组后，每条任务对应的input行号，
 *                        长度为totalTasks。
 * @param routeScales     每条分组任务的路由权重，长度为totalTasks。
 * @param routePositions  原始[token, topk]槽位到分组任务位置的反向映射，
 *                        长度为batch * topk。
 * @param expertStarts    每个routed expert在分组任务数组中的起始偏移，
 *                        长度为expert数量。
 * @param expertCounts    每个routed expert本次处理的任务行数，长度为
 *                        expert数量。
 * @param batch           输入token行数，即总的MoE输入M维。
 * @param topk            每个token选中的routed expert数量。
 * @param totalTasks      路由任务总数，必须等于batch * topk。
 * @param hidden          模型隐藏维度，也是gate/up GEMM的K维和down GEMM的N维。
 * @param inter           单个expert中间维度，也是gate/up输出半宽和down GEMM的K维。
 * @param gateType        门控激活类型，支持MoeGateSwiglu和MoeGateGeglu。
 * @return true表示聚集、两次量化、两次grouped GEMM及归并均成功；false
 *         表示语义、布局、资源申请或CUDA启动失败，由外层生命周期入口
 *         决定首次回退或固定后报错。
 */
template <typename T>
bool RunHostRoutes(const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
         fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
         const int *routeRows, const float *routeScales,
         const int *routePositions, const int *expertStarts,
         const int *expertCounts, int batch, int topk, int totalTasks,
         int hidden, int inter, fastllm::MoeGateType gateType) {
    const fastllm::DataType dtype = std::is_same_v<T, half>
        ? fastllm::DataType::FLOAT16 : fastllm::DataType::BFLOAT16;
    const int experts = weightsBatch / 2 - 1;
    const int arch = FastllmCudaRuntimeArch();
    // 先验证架构、输入类型、路由规模和张量布局；任一条件不满足时交由
    // 外层决定是否进入Legacy，不在本函数中改变后端生命周期。
    // vLLM 的 SM120 grouped kernel 固定输出 BF16；SM100 同时支持 FP16/BF16。
    if (input.dataType != dtype || input.dataDevice != fastllm::DataDevice::CUDA ||
        (arch >= 120 && dtype != fastllm::DataType::BFLOAT16) ||
        (gateType != fastllm::MoeGateSwiglu &&
         gateType != fastllm::MoeGateGeglu) ||
        batch <= 0 || topk <= 0 || totalTasks != batch * topk ||
        experts <= 0 || experts > 256 || hidden % 32 != 0 || inter % 32 != 0 ||
        routeRows == nullptr || routeScales == nullptr || routePositions == nullptr ||
        expertStarts == nullptr || expertCounts == nullptr ||
        FastllmCudaGraphIsCapturing()) return false;

    for (int e = 0; e < experts; ++e) {
        fastllm::Data *gate = weights[(e + 1) * 2];
        fastllm::Data *down = weights[(e + 1) * 2 + 1];
        if (!gate || !down || gate->dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            down->dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            gate->blockM != 16 || down->blockM != 16 ||
            gate->dims != std::vector<int>({inter * 2, hidden}) ||
            down->dims != std::vector<int>({hidden, inter})) return false;
    }

    // 为两次grouped GEMM准备正式推理使用的中间张量：w1保存gate/up，
    // w2保存down结果；output保持与输入相同的batch/token布局。
    w1.dataDevice = fastllm::DataDevice::CUDA;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = dtype;
    w1.UpdateUnitSize();
    w1.Resize({totalTasks, inter * 2});
    w1.Allocate(false);
    w2.dataDevice = fastllm::DataDevice::CUDA;
    w2.dataDeviceIds = input.dataDeviceIds;
    w2.dataType = dtype;
    w2.UpdateUnitSize();
    w2.Resize({totalTasks, hidden});
    w2.Allocate(false);
    output.dataDevice = fastllm::DataDevice::CUDA;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = dtype;
    output.UpdateUnitSize();
    output.Resize(input.dims);
    output.Allocate(false);

    // 每个expert的scale布局都会把自身M补齐到128，因此不能只按totalTasks
    // 计算一块scale大小；先计算各expert在持久scale缓冲区中的独立偏移。
    std::vector<int> activeExperts;
    std::vector<size_t> gateScaleOffsets, downScaleOffsets;
    activeExperts.reserve(experts);
    gateScaleOffsets.reserve(experts);
    downScaleOffsets.reserve(experts);
    size_t gateScaleBytes = 0, downScaleBytes = 0;
    for (int e = 0; e < experts; ++e) if (expertCounts[e] > 0) {
        activeExperts.push_back(e);
        gateScaleOffsets.push_back(gateScaleBytes);
        downScaleOffsets.push_back(downScaleBytes);
        gateScaleBytes += FastllmCudaNvfp4SwizzledScaleBytes(expertCounts[e], hidden);
        downScaleBytes += FastllmCudaNvfp4SwizzledScaleBytes(expertCounts[e], inter);
    }
    if (activeExperts.empty()) return false;

    MoeScratch *scratch = GetMoeScratch(
        totalTasks, (size_t)totalTasks * hidden * sizeof(T),
        (size_t)totalTasks * hidden / 2, gateScaleBytes,
        (size_t)totalTasks * inter / 2, downScaleBytes,
        gateType == fastllm::MoeGateGeglu
            ? (size_t)totalTasks * inter * sizeof(T) : sizeof(T), experts);
    if (scratch == nullptr) return false;

    // 路由元数据由CPU按expert连续排列；异步上传后，gather kernel把原始
    // token行重排为grouped GEMM要求的expert分组输入。
    int *cudaRouteRows = scratch->routeRows;
    int *cudaRoutePositions = scratch->routePositions;
    float *cudaRouteScales = scratch->routeScales;
    cudaStream_t stream = cudaStreamPerThread;
    bool ok = cudaMemcpyAsync(cudaRouteRows, routeRows, sizeof(int) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(cudaRoutePositions, routePositions, sizeof(int) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess &&
              cudaMemcpyAsync(cudaRouteScales, routeScales, sizeof(float) * (size_t)totalTasks,
                              cudaMemcpyHostToDevice, stream) == cudaSuccess;
    T *cudaInput = static_cast<T *>(FastllmCudaPrepareInput(input));
    if (ok && cudaInput) {
        size_t elements = (size_t)totalTasks * hidden;
        GatherRows<<<(elements + 255) / 256, 256, 0, stream>>>(
            cudaInput, cudaRouteRows, reinterpret_cast<T *>(scratch->gathered),
            totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    } else ok = false;

    // 第一层：逐个活跃expert量化输入、取得对应CUTLASS重排权重，然后
    // 将各expert的指针和实际行数组成一次grouped W4A4 GEMM参数。
    std::vector<const uint8_t *> a, b, scaleA, scaleB;
    std::vector<const float *> alpha;
    std::vector<void *> d;
    std::vector<int> rows;
    a.reserve(activeExperts.size()); b.reserve(activeExperts.size());
    scaleA.reserve(activeExperts.size()); scaleB.reserve(activeExperts.size());
    alpha.reserve(activeExperts.size()); d.reserve(activeExperts.size());
    rows.reserve(activeExperts.size());
    for (size_t i = 0; ok && i < activeExperts.size(); ++i) {
        const int e = activeExperts[i];
        int count = expertCounts[e], start = expertStarts[e];
        uint8_t *quantValues = scratch->gateValues + (size_t)start * hidden / 2;
        uint8_t *quantScales = scratch->gateScales + gateScaleOffsets[i];
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &gate = *weights[(e + 1) * 2];
        ok = FastllmCudaNvfp4QuantizeActivation(
                 reinterpret_cast<T *>(scratch->gathered) + (size_t)start * hidden,
                 dtype, quantValues, quantScales, count, hidden, 1.0f, (void *)stream) &&
             FastllmCudaPrepareNvfp4W4A4Weight(
                 gate, hidden, inter * 2, &weight, &weightScales, &weightAlpha);
        if (ok) {
            rows.push_back(count);
            a.push_back(quantValues); b.push_back(weight);
            scaleA.push_back(quantScales); scaleB.push_back(weightScales);
            alpha.push_back(weightAlpha);
            d.push_back(static_cast<T *>(w1.cudaData) + (size_t)start * inter * 2);
        }
    }
    if (ok) {
        ok = arch >= 120
            ? FastllmCudaNvfp4GroupedGemmSm120(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), inter * 2, hidden, dtype, (void *)stream)
            : FastllmCudaNvfp4GroupedGemmSm100(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), inter * 2, hidden, dtype, (void *)stream);
    }

    // 第二层：复用同一组参数容器和持久scratch。SwiGLU走融合激活量化，
    // GeGLU先计算门控激活再独立量化，最终调用down grouped W4A4 GEMM。
    a.clear(); b.clear(); scaleA.clear(); scaleB.clear(); alpha.clear(); d.clear(); rows.clear();
    for (size_t i = 0; ok && i < activeExperts.size(); ++i) {
        int e = activeExperts[i], count = expertCounts[e], start = expertStarts[e];
        uint8_t *quantValues = scratch->downValues + (size_t)start * inter / 2;
        uint8_t *quantScales = scratch->downScales + downScaleOffsets[i];
        const uint8_t *weight = nullptr, *weightScales = nullptr;
        const float *weightAlpha = nullptr;
        fastllm::Data &down = *weights[(e + 1) * 2 + 1];
        const T *gateInput = static_cast<T *>(w1.cudaData) +
                             (size_t)start * inter * 2;
        if (gateType == fastllm::MoeGateSwiglu) {
            // SwiGLU沿用vLLM对应的激活+FP4量化融合路径。
            ok = FastllmCudaSiluMulNvfp4Quantize(
                gateInput, dtype, quantValues, quantScales,
                count, inter, 1.0f, (void *)stream);
        } else {
            // vLLM的GeGLU路径并不融合量化：先完成门控激活，再调用普通
            // NVFP4动态量化。activated按路由顺序复用持久scratch。
            T *activated = reinterpret_cast<T *>(scratch->activated) +
                           (size_t)start * inter;
            const size_t elements = (size_t)count * inter;
            GegluRows<<<(elements + 255) / 256, 256, 0, stream>>>(
                gateInput, activated, count, inter);
            ok = cudaGetLastError() == cudaSuccess &&
                 FastllmCudaNvfp4QuantizeActivation(
                     activated, dtype, quantValues, quantScales,
                     count, inter, 1.0f, (void *)stream);
        }
        ok = ok && FastllmCudaPrepareNvfp4W4A4Weight(
            down, inter, hidden, &weight, &weightScales, &weightAlpha);
        if (ok) {
            rows.push_back(count); a.push_back(quantValues); b.push_back(weight);
            scaleA.push_back(quantScales); scaleB.push_back(weightScales);
            alpha.push_back(weightAlpha);
            d.push_back(static_cast<T *>(w2.cudaData) + (size_t)start * hidden);
        }
    }
    if (ok) {
        ok = arch >= 120
            ? FastllmCudaNvfp4GroupedGemmSm120(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), hidden, inter, dtype, (void *)stream)
            : FastllmCudaNvfp4GroupedGemmSm100(
                  a.data(), b.data(), scaleA.data(), scaleB.data(), alpha.data(), d.data(),
                  rows.data(), (int)rows.size(), hidden, inter, dtype, (void *)stream);
    }
    if (ok) {
        // grouped GEMM输出按路由顺序排列；根据原始token/topk位置和路由
        // 分数做加权归并，恢复为output[batch, hidden]。
        size_t elements = (size_t)batch * hidden;
        ReduceRoutes<<<(elements + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(w2.cudaData), cudaRoutePositions,
            cudaRouteScales, static_cast<T *>(output.cudaData), batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }

    if (cudaInput) FastllmCudaFinishInput(input, cudaInput);
    return ok;
}

/**
 * 在SM120上使用GPU常驻路由执行一次NVFP4 grouped MoE。
 *
 * 本路径忠实采用vLLM的设备侧流程：GPU统计每个expert的路由数量并生成
 * 前缀和，按expert重排输入，再用单个expert-aware kernel量化全部路由行。
 * 两次grouped GEMM都直接读取GPU端expertOffsets，不把index、score或
 * expert计数复制回CPU。当前函数只负责单次计算，后端固定与首次失败回退
 * 仍由外层FastllmCudaMergeMoeNvfp4W4A4Grouped处理。
 *
 * @tparam T          激活类型，SM120当前仅实例化__nv_bfloat16。
 * @param input       BF16输入，逻辑形状为[batch, hidden]。
 * @param index       CUDA INT32路由索引，形状为[batch, topk]。
 * @param score       CUDA FP32路由权重，形状为[batch, topk]。
 * @param w1          gate/up grouped GEMM输出，形状为[batch*topk,2*inter]。
 * @param w2          down grouped GEMM输出，形状为[batch*topk,hidden]。
 * @param output      加权归并后的BF16输出，形状为[batch,hidden]。
 * @param weights     MoE权重数组；routed expert从下标2开始成对存放。
 * @param weightsBatch 权重指针数量。
 * @param batch       token行数。
 * @param topk        每个token的routed expert数。
 * @param hidden      模型隐藏维度。
 * @param inter       expert中间维度。
 * @param gateType    SwiGLU或GeGLU。
 * @return 全部CUDA阶段成功启动时返回true，否则返回false。
 */
template <typename T>
bool RunDeviceRoutesSm120(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data &w1, fastllm::Data &w2,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        int batch, int topk, int hidden, int inter,
        fastllm::MoeGateType gateType) {
    const int experts = weightsBatch / 2 - 1;
    const int totalTasks = batch * topk;
    if (!std::is_same_v<T, __nv_bfloat16> ||
        (gateType != fastllm::MoeGateSwiglu &&
         gateType != fastllm::MoeGateGeglu) ||
        FastllmCudaRuntimeArch() < 120 || FastllmCudaRuntimeArch() >= 130 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        index.dataDevice != fastllm::DataDevice::CUDA ||
        index.dataType != fastllm::DataType::INT32 ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataType != fastllm::DataType::FLOAT32 ||
        batch <= 0 || topk <= 0 || experts <= 0 || experts > 256 ||
        totalTasks <= 0 || hidden % 32 != 0 || inter % 32 != 0) return false;

    w1.dataDevice = fastllm::DataDevice::CUDA;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = fastllm::DataType::BFLOAT16;
    w1.UpdateUnitSize();
    w1.Resize({totalTasks, inter * 2});
    w1.Allocate(false);
    w2.dataDevice = fastllm::DataDevice::CUDA;
    w2.dataDeviceIds = input.dataDeviceIds;
    w2.dataType = fastllm::DataType::BFLOAT16;
    w2.UpdateUnitSize();
    w2.Resize({totalTasks, hidden});
    w2.Allocate(false);
    output.dataDevice = fastllm::DataDevice::CUDA;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = fastllm::DataType::BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(input.dims);
    output.Allocate(false);
    if (!w1.cudaData || !w2.cudaData || !output.cudaData) return false;

    // scale按每个expert独立补齐到128行；总容量取最坏上界，真实偏移由
    // BuildRouteOffsets在GPU上计算，无需把expertCounts同步回CPU。
    const size_t paddedRows = (size_t)totalTasks +
        (size_t)std::min(experts, totalTasks) * 127;
    const size_t gateScaleBytes = paddedRows * (((hidden / 16) + 3) / 4 * 4);
    const size_t downScaleBytes = paddedRows * (((inter / 16) + 3) / 4 * 4);
    MoeScratch *scratch = GetMoeScratch(
        totalTasks, (size_t)totalTasks * hidden * sizeof(T),
        (size_t)totalTasks * hidden / 2, gateScaleBytes,
        (size_t)totalTasks * inter / 2, downScaleBytes,
        gateType == fastllm::MoeGateGeglu
            ? (size_t)totalTasks * inter * sizeof(T) : sizeof(T), experts);
    if (scratch == nullptr) return false;

    cudaStream_t stream = cudaStreamPerThread;
    constexpr int threads = 256;
    bool ok = cudaMemsetAsync(scratch->expertCounts, 0,
                              sizeof(int) * (size_t)(experts + 1), stream) == cudaSuccess;
    if (ok) {
        CountRoutes<<<(totalTasks + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<const int32_t *>(index.cudaData), scratch->expertCounts,
            totalTasks, experts);
        BuildRouteOffsets<<<1, 1, 0, stream>>>(
            scratch->expertCounts, scratch->expertOffsets,
            scratch->blockscaleOffsets, scratch->expertCursors, experts);
        ScatterRoutes<<<(totalTasks + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<const int32_t *>(index.cudaData),
            static_cast<const float *>(score.cudaData), scratch->expertCursors,
            scratch->routeRows, scratch->routeScales, scratch->routePositions,
            totalTasks, topk, experts);
        ok = cudaGetLastError() == cudaSuccess;
    }
    T *cudaInput = static_cast<T *>(FastllmCudaPrepareInput(input));
    if (ok && cudaInput != nullptr) {
        const size_t elements = (size_t)totalTasks * hidden;
        GatherRows<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
            cudaInput, scratch->routeRows, reinterpret_cast<T *>(scratch->gathered),
            totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    } else {
        ok = false;
    }

    MoeLayerWeightMetadata *weightMetadata = ok
        ? GetLayerWeightMetadata(weights, weightsBatch, hidden, inter) : nullptr;
    ok = ok && weightMetadata != nullptr;

    const int gatePacks = hidden / fastllm_nvfp4::kElementsPerThread;
    if (ok) {
        const int quantThreads = std::min(512, gatePacks);
        QuantizeExpertRows<T, false>
            <<<dim3(std::min(totalTasks, 128),
                    (gatePacks + quantThreads - 1) / quantThreads),
               quantThreads, 0, stream>>>(
                reinterpret_cast<const T *>(scratch->gathered),
                scratch->gateValues, scratch->gateScales,
                scratch->expertOffsets, scratch->blockscaleOffsets,
                totalTasks, hidden, experts);
        ok = cudaGetLastError() == cudaSuccess &&
             FastllmCudaNvfp4GroupedGemmSm120DeviceRoutes(
                 scratch->gateValues, scratch->gateScales, w1.cudaData,
                 weightMetadata->gateWeights.data(),
                 weightMetadata->gateScales.data(),
                 weightMetadata->gateAlphas.data(),
                 scratch->expertOffsets, scratch->blockscaleOffsets,
                 experts, inter * 2, hidden, fastllm::DataType::BFLOAT16,
                 (void *)stream);
    }

    const int downPacks = inter / fastllm_nvfp4::kElementsPerThread;
    if (ok) {
        const int quantThreads = std::min(512, downPacks);
        const dim3 quantGrid(
            std::min(totalTasks, 128),
            (downPacks + quantThreads - 1) / quantThreads);
        if (gateType == fastllm::MoeGateSwiglu) {
            QuantizeExpertRows<T, true><<<quantGrid, quantThreads, 0, stream>>>(
                static_cast<const T *>(w1.cudaData), scratch->downValues,
                scratch->downScales, scratch->expertOffsets,
                scratch->blockscaleOffsets, totalTasks, inter, experts);
        } else {
            const size_t elements = (size_t)totalTasks * inter;
            GegluRows<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<const T *>(w1.cudaData),
                reinterpret_cast<T *>(scratch->activated), totalTasks, inter);
            QuantizeExpertRows<T, false><<<quantGrid, quantThreads, 0, stream>>>(
                reinterpret_cast<const T *>(scratch->activated),
                scratch->downValues, scratch->downScales,
                scratch->expertOffsets, scratch->blockscaleOffsets,
                totalTasks, inter, experts);
        }
        ok = cudaGetLastError() == cudaSuccess &&
             FastllmCudaNvfp4GroupedGemmSm120DeviceRoutes(
                 scratch->downValues, scratch->downScales, w2.cudaData,
                 weightMetadata->downWeights.data(),
                 weightMetadata->downScales.data(),
                 weightMetadata->downAlphas.data(),
                 scratch->expertOffsets, scratch->blockscaleOffsets,
                 experts, hidden, inter, fastllm::DataType::BFLOAT16,
                 (void *)stream);
    }
    if (ok) {
        const size_t elements = (size_t)batch * hidden;
        ReduceRoutes<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<const T *>(w2.cudaData), scratch->routePositions,
            scratch->routeScales, static_cast<T *>(output.cudaData),
            batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (cudaInput) FastllmCudaFinishInput(input, cudaInput);
    return ok;
}

static bool TraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_NVFP4_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/** 返回代表本层routed experts后端生命周期的首个gate/up权重。 */
static fastllm::Data *BackendAnchor(fastllm::Data **weights, int weightsBatch) {
    return weights != nullptr && weightsBatch >= 4 ? weights[2] : nullptr;
}

/**
 * 在首次grouped GEMM前确认全部routed expert权重均已完成重排。
 *
 * 首批token通常只命中部分expert，不能仅提交活跃权重，否则尚未执行的
 * expert会被错误标成Cutlass。这里遍历整层routed权重；正常模型加载路径
 * 已逐权重预重排，因此这里只命中cache。直接调用路径则补做重排，并把
 * Uninitialized更新为Prepared，仍保留首次真实GEMM的一次性验证机会。
 *
 * @param weights      MoE权重数组。
 * @param weightsBatch 权重指针数量。
 * @param device       当前CUDA设备号。
 * @return true表示全部routed权重都具备CUTLASS表示。
 */
static bool PrepareRoutedWeights(fastllm::Data **weights, int weightsBatch,
                                 int device) {
    for (int i = 2; i < weightsBatch; ++i) {
        fastllm::Data *weight = weights[i];
        const FastllmCudaNvfp4BackendState state = weight == nullptr
            ? FastllmCudaNvfp4BackendState::Rejected
            : FastllmCudaGetNvfp4W4A4BackendState(*weight, device);
        if (weight == nullptr ||
            weight->dataType != fastllm::DataType::NVFP4_BLOCK_16 ||
            weight->dims.size() != 2 ||
            (state != FastllmCudaNvfp4BackendState::Uninitialized &&
             state != FastllmCudaNvfp4BackendState::Prepared &&
             state != FastllmCudaNvfp4BackendState::CutlassW4A4)) {
            return false;
        }
        const uint8_t *packed = nullptr, *scales = nullptr;
        const float *alpha = nullptr;
        if (!FastllmCudaPrepareNvfp4W4A4Weight(
                *weight, weight->dims[1], weight->dims[0],
                &packed, &scales, &alpha)) {
            return false;
        }
        if (FastllmCudaGetNvfp4W4A4BackendState(*weight, device) ==
            FastllmCudaNvfp4BackendState::Uninitialized) {
            FastllmCudaSetNvfp4W4A4BackendState(
                *weight, device, FastllmCudaNvfp4BackendState::Prepared);
        }
    }
    return true;
}

/**
 * 将本层全部routed expert权重提交为固定CUTLASS后端。
 *
 * grouped GEMM首次同步验证成功后统一提交状态。shared expert位于前两个
 * 槽位，由独立Dense Linear维护自己的生命周期，不在此处修改。
 *
 * @param weights      MoE权重数组。
 * @param weightsBatch 权重指针数量。
 * @param device       当前CUDA设备号。
 */
static void CommitRoutedWeights(fastllm::Data **weights, int weightsBatch,
                                int device) {
    for (int i = 2; i < weightsBatch; ++i) {
        fastllm::Data *weight = weights[i];
        if (weight != nullptr &&
            weight->dataType == fastllm::DataType::NVFP4_BLOCK_16) {
            FastllmCudaSetNvfp4W4A4BackendState(
                *weight, device, FastllmCudaNvfp4BackendState::CutlassW4A4);
        }
    }
}

/**
 * 把本层全部routed expert权重一次性固定为Legacy后端。
 *
 * 首次grouped GEMM验证失败时先释放整层在当前GPU上的CUTLASS重排cache，
 * 再从CPU/NUMA/mmap原始表示逐个恢复CUDA权重，最后统一写入Rejected。
 * 这种两阶段顺序避免整层原始权重与整层重排权重同时占用显存。shared
 * expert不属于grouped计算，继续由Dense Linear独立管理。
 *
 * @param weights      MoE权重数组。
 * @param weightsBatch 权重指针数量。
 * @param device       当前CUDA设备号。
 * @return true表示全部原始CUDA权重恢复成功并已固定为Rejected。
 */
static bool RejectRoutedWeights(fastllm::Data **weights, int weightsBatch,
                                int device) {
    // 第一阶段只清cache，不立即恢复，先降低整层回退时的显存峰值。
    for (int i = 2; i < weightsBatch; ++i) {
        fastllm::Data *weight = weights[i];
        if (weight != nullptr &&
            weight->dataType == fastllm::DataType::NVFP4_BLOCK_16) {
            FastllmCudaReleaseNvfp4W4A4CacheForDevice(weight, device);
        }
    }

    // 第二阶段恢复Legacy需要的原始设备表示。
    for (int i = 2; i < weightsBatch; ++i) {
        fastllm::Data *weight = weights[i];
        if (weight != nullptr &&
            weight->dataType == fastllm::DataType::NVFP4_BLOCK_16 &&
            weight->cudaData == nullptr &&
            !weight->RestoreCudaDataForRepackedWeight(device)) {
            return false;
        }
    }

    // 资源全部就绪后再提交状态，正式推理不再重复尝试grouped CUTLASS。
    for (int i = 2; i < weightsBatch; ++i) {
        fastllm::Data *weight = weights[i];
        if (weight != nullptr &&
            weight->dataType == fastllm::DataType::NVFP4_BLOCK_16) {
            FastllmCudaSetNvfp4W4A4BackendState(
                *weight, device, FastllmCudaNvfp4BackendState::Rejected);
        }
    }
    return true;
}
} // namespace

/**
 * 执行一次NVFP4 W4A4 grouped MoE计算。
 *
 * 本函数是CUDA grouped W4A4 MoE的公开入口，负责首次选择并固定本层
 * routed experts后端，再按输入类型分派到内部Run实现。内部流程依次完成路由行gather、
 * gate/up激活动态NVFP4量化、第一层grouped GEMM、SwiGLU或GeGLU门控、
 * down激活动态NVFP4量化、第二层grouped GEMM，以及top-k路由加权归并。
 * SM120直接接收CUDA端index/score并在GPU上构造路由表；SM100暂时保留
 * host-routes兼容实现，等待对应device metadata launcher完成后再切换。
 *
 * 数学语义为：对每个token选中的top-k expert分别计算
 * down(activation(gate_up(input)))，再乘以对应routeScales并求和。
 * gate/up权重逻辑形状为[2 * inter, hidden]，down权重逻辑形状为
 * [hidden, inter]，两者均为NVFP4_BLOCK_16布局。
 *
 * Prepared或Uninitialized状态下会对首次真实GEMM强制同步验证。成功后
 * 全部routed权重固定为Cutlass；失败则先清整层cache、再恢复整层原始
 * CUDA权重并固定为Rejected。正式推理进入Cutlass后禁止运行时切换；
 * 此后若kernel失败直接抛错。STRICT模式下首次失败也禁止静默降级。
 *
 * @param input           FP16或BF16输入，逻辑形状为[batch, hidden]；SM120
 *                        当前仅接受BF16。
 * @param w1              第一层grouped GEMM临时输出，函数内调整为
 *                        [totalTasks, 2 * inter]。
 * @param w2              第二层grouped GEMM临时输出，函数内调整为
 *                        [totalTasks, hidden]。
 * @param output           最终MoE输出，逻辑形状与input一致。
 * @param weights          expert权重指针数组；第0对为可选shared expert，
 *                        routed expert e使用第e+1对gate/up和down权重。
 * @param weightsBatch     weights数组中的Data指针数量，必须为偶数且至少为4。
 * @param index            CUDA INT32 expert索引，形状为[batch,topk]。
 * @param score            CUDA FP32路由权重，形状为[batch,topk]。
 * @param batch            输入token行数，即grouped MoE的M总规模。
 * @param topk             每个token选中的routed expert数量。
 * @param hidden           模型隐藏维度，必须按32对齐。
 * @param inter            单个expert的中间维度，必须按32对齐。
 * @param gateType         门控激活类型，当前支持MoeGateSwiglu和
 *                        MoeGateGeglu。
 * @return true表示两次grouped W4A4 GEMM和路由归并均成功；false表示条件
 *         不满足或执行失败，具体路径可通过FASTLLM_CUDA_NVFP4_TRACE查看。
 */
bool FastllmCudaMergeMoeNvfp4W4A4Grouped(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const fastllm::Data &index, const fastllm::Data &score,
        int batch, int topk, int hidden, int inter,
        fastllm::MoeGateType gateType) {
    const int arch = FastllmCudaRuntimeArch();
    const int device = FastllmCudaGetDevice();
    fastllm::Data *anchor = BackendAnchor(weights, weightsBatch);
    if (device < 0 || anchor == nullptr) return false;

    const FastllmCudaNvfp4BackendState state =
        FastllmCudaGetNvfp4W4A4BackendState(*anchor, device);
    if (state == FastllmCudaNvfp4BackendState::Rejected ||
        state == FastllmCudaNvfp4BackendState::MarlinW4A16Fallback ||
        state == FastllmCudaNvfp4BackendState::NativeW4A16Fallback) {
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict NVFP4 grouped MoE backend was already rejected");
        }
        return false;
    }
    const bool selecting =
        state == FastllmCudaNvfp4BackendState::Uninitialized ||
        state == FastllmCudaNvfp4BackendState::Prepared;
    if ((!Enabled() && selecting) || arch < 100 || arch >= 130 ||
        weights == nullptr || weightsBatch < 4 || (weightsBatch & 1)) {
        if (selecting) {
            if (!RejectRoutedWeights(weights, weightsBatch, device)) {
                throw std::runtime_error(
                    "NVFP4 grouped MoE fallback cannot restore original weights");
            }
            if (StrictEnabled()) {
                throw std::runtime_error(
                    "strict NVFP4 grouped MoE requires the CUTLASS backend");
            }
            return false;
        }
        throw std::runtime_error(
            "NVFP4 grouped MoE fixed CUTLASS backend became unavailable");
    }

    // 首次选择时必须先验证整层所有routed权重，而不是只检查本批活跃expert。
    bool ok = !selecting || PrepareRoutedWeights(weights, weightsBatch, device);
    if (ok && arch >= 120 && input.dataType == fastllm::DataType::BFLOAT16) {
        ok = RunDeviceRoutesSm120<__nv_bfloat16>(
            input, index, score, w1, w2, output, weights, weightsBatch,
            batch, topk, hidden, inter, gateType);
    } else if (ok && arch < 120) {
        // SM100兼容路径暂时保留原实现；仅该架构仍发生一次D2H同步。
        const int totalTasks = batch * topk;
        std::vector<int32_t> hostIndex(totalTasks);
        std::vector<float> hostScore(totalTasks);
        ok = index.dataDevice == fastllm::DataDevice::CUDA &&
             index.dataType == fastllm::DataType::INT32 &&
             score.dataDevice == fastllm::DataDevice::CUDA &&
             score.dataType == fastllm::DataType::FLOAT32 &&
             cudaMemcpyAsync(hostIndex.data(), index.cudaData,
                             sizeof(int32_t) * (size_t)totalTasks,
                             cudaMemcpyDeviceToHost, cudaStreamPerThread) == cudaSuccess &&
             cudaMemcpyAsync(hostScore.data(), score.cudaData,
                             sizeof(float) * (size_t)totalTasks,
                             cudaMemcpyDeviceToHost, cudaStreamPerThread) == cudaSuccess &&
             cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
        const int experts = weightsBatch / 2 - 1;
        std::vector<int> counts(experts, 0), starts(experts, 0);
        for (int route = 0; ok && route < totalTasks; ++route) {
            const int expert = hostIndex[route];
            if (expert < 0 || expert >= experts) ok = false;
            else counts[expert]++;
        }
        int total = 0;
        for (int expert = 0; expert < experts; ++expert) {
            starts[expert] = total;
            total += counts[expert];
        }
        std::vector<int> cursors = starts, rows(totalTasks), positions(totalTasks);
        std::vector<float> scales(totalTasks);
        for (int route = 0; ok && route < totalTasks; ++route) {
            const int position = cursors[hostIndex[route]]++;
            rows[position] = route / topk;
            scales[position] = hostScore[route];
            positions[route] = position;
        }
        if (ok && input.dataType == fastllm::DataType::FLOAT16)
            ok = RunHostRoutes<half>(
                input, w1, w2, output, weights, weightsBatch,
                rows.data(), scales.data(), positions.data(), starts.data(),
                counts.data(), batch, topk, totalTasks, hidden, inter, gateType);
        else if (ok && input.dataType == fastllm::DataType::BFLOAT16)
            ok = RunHostRoutes<__nv_bfloat16>(
                input, w1, w2, output, weights, weightsBatch,
                rows.data(), scales.data(), positions.data(), starts.data(),
                counts.data(), batch, topk, totalTasks, hidden, inter, gateType);
    } else {
        ok = false;
    }
    if (ok && selecting) {
        // CUDA kernel异步返回；首次必须同步验证后才能释放回退机会并固定后端。
        ok = cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess;
    }

    if (ok && selecting) {
        CommitRoutedWeights(weights, weightsBatch, device);
    } else if (!ok && selecting) {
        // 一次性兜底只发生在正式后端固定前。
        if (!RejectRoutedWeights(weights, weightsBatch, device)) {
            throw std::runtime_error(
                "NVFP4 grouped MoE fallback cannot restore original weights");
        }
        if (StrictEnabled()) {
            throw std::runtime_error(
                "strict NVFP4 grouped MoE requires the CUTLASS backend");
        }
    } else if (!ok) {
        // 已固定CUTLASS后不得混推或临时切回Legacy。
        throw std::runtime_error(
            "NVFP4 grouped MoE CUTLASS failed after backend selection");
    }
    if (TraceEnabled()) {
        std::fprintf(stderr,
            "[fastllm][nvfp4] path=%s state=%s route=%s batch=%d topk=%d hidden=%d inter=%d sm=%d\n",
            ok ? "grouped-moe-w4a4-cutlass" : "grouped-moe-fallback",
            selecting ? "selected" : "fixed",
            arch >= 120 ? "gpu" : "host",
            batch, topk, hidden, inter, arch);
    }
    return ok;
}
