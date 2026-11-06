#include "fastllm-cuda.cuh"
#include <cstdlib>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <thread>

#ifdef FASTLLM_ENABLE_MACHETE
#include "machete/fastllm-machete-kernel.cuh"
#endif

namespace {
using fastllm::Data;
using fastllm::DataType;
std::mutex cacheMutex;

struct State {
    const void *source = nullptr;
    int n = 0, k = 0, groupSize = 0;
    std::string policy;
#ifdef FASTLLM_ENABLE_MACHETE
    std::unique_ptr<fastllm_machete::Weight> weight;
#endif
    std::mutex submissionMutex;
};
using Key = std::tuple<const Data*, int, int, std::thread::id>;
std::map<Key, std::shared_ptr<State>> states;
std::set<std::tuple<const Data*, int, int, std::string>> traces;

/** 读取正式Linear后端策略；默认native以保持旧模型行为不变。 */
std::string Policy() {
    const char *value = std::getenv("FASTLLM_LINEAR_BACKEND");
    std::string policy = value && *value ? value : "native";
    if (policy != "native" && policy != "machete" && policy != "auto") {
        throw std::runtime_error("FASTLLM_LINEAR_BACKEND must be native, machete or auto");
    }
    return policy;
}

/** 判断是否为本次W4/W8权重量化后端选择范围，不包含W4A8/W8A8/FP8/NVFP4。 */
bool Target(DataType type) {
    return type == DataType::INT8 || type == DataType::INT4 ||
           type == DataType::INT4_NOZERO || type == DataType::INT4_GROUP ||
           type == DataType::INT4_GROUP32 || type == DataType::INT4_GROUP128;
}

/** 在路径确认阶段记录选择和量化参数舍入误差，每个权重/设备/类型仅记录一次。 */
void Trace(const Data &weight, int device, int dtype, const std::string &requested,
           const std::string &actual, const std::string &reason, const State *state = nullptr) {
    const char *enabled = std::getenv("FASTLLM_LINEAR_BACKEND_TRACE");
    if (!enabled || std::string(enabled) != "1") return;
    if (!traces.emplace(&weight, device, dtype, requested).second) return;
    float scaleError = 0, offsetError = 0;
    size_t bytes = 0;
    const char *encoding = "native";
#ifdef FASTLLM_ENABLE_MACHETE
    if (state && state->weight) {
        scaleError = state->weight->maxScaleError;
        offsetError = state->weight->maxOffsetError;
        bytes = state->weight->residentBytes;
        encoding = state->weight->encoding;
    }
#endif
    fprintf(stderr, "[fastllm][linear-backend] requested=%s actual=%s device=%d "
            "dtype=%d weight=%s reason=%s scale_max_abs_error=%.9g "
            "offset_max_abs_error=%.9g prepack_bytes=%zu encoding=%s n=%d k=%d group_size=%d\n",
            requested.c_str(), actual.c_str(), device, dtype, weight.name.c_str(),
            reason.c_str(), scaleError, offsetError, bytes, encoding,
            weight.dims.size() == 2 ? weight.dims[0] : 0,
            weight.dims.size() == 2 ? weight.dims[1] : 0,
            state ? state->groupSize : weight.groupCnt);
}

/** 检查FastLLM原生量化元数据能否映射为Machete unsigned+scale+offset。 */
std::string Support(const Data &input, const Data &weight, const Data &bias,
                    const Data &output, int &groupSize, int device) {
#ifndef FASTLLM_ENABLE_MACHETE
    return "build_without_machete_sm90a";
#else
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "device_query_failed";
    if (prop.major != 9 || prop.minor != 0) return "requires_sm90_hopper";
    if (input.dataType != DataType::FLOAT16 && input.dataType != DataType::BFLOAT16) return "requires_a16";
    if (weight.dataType != DataType::INT8 && weight.dataType != DataType::INT4_NOZERO &&
        weight.dataType != DataType::INT4_GROUP) return "unsupported_native_weight_encoding";
    if (weight.dims.size() != 2 || input.dims.empty()) return "invalid_rank";
    int n = weight.dims[0], k = weight.dims[1];
    if (n <= 0 || k <= 0 || n % 128 || k % 64) return "requires_N_multiple128_K_multiple64";
    if (input.dims.back() != k || output.dataType != input.dataType ||
        output.dims.empty() || output.dims.back() != n ||
        output.Count(0) != input.Count(0) / k * n) return "invalid_io_shape_or_dtype";
    if (!input.cudaData || !weight.cudaData || !output.cudaData) return "requires_resident_cuda_tensors";
    if (input.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA) return "requires_cuda_tensors";
    if (weight.strides.size() != 2 || weight.strides[1] != 1 || weight.strides[0] != uint64_t(k)) return "requires_contiguous_weight";
    for (const Data *data : {&input, &output}) {
        uint64_t stride = 1;
        if (data->strides.size() != data->dims.size()) return "invalid_io_strides";
        for (int d = int(data->dims.size()) - 1; d >= 0; --d) {
            if (data->strides[d] != stride) return "requires_contiguous_io";
            stride *= data->dims[d];
        }
    }
    if (!bias.dims.empty() && (bias.dataType != DataType::FLOAT32 || !bias.cudaData ||
                              bias.Count(0) != uint64_t(n))) return "requires_fp32_bias_N";
    for (const Data *data : {&input, &weight, &output, &bias}) {
        if (data == &bias && bias.dims.empty()) continue;
        cudaPointerAttributes attributes;
        if (cudaPointerGetAttributes(&attributes, data->cudaData) != cudaSuccess) {
            cudaGetLastError();
            return "invalid_device_pointer";
        }
        if (attributes.device != device) return "tensor_on_wrong_cuda_device";
    }
    groupSize = weight.dataType == DataType::INT4_GROUP ? weight.groupCnt : k;
    if (groupSize <= 0 || k % groupSize || (groupSize != k && groupSize % 64)) return "unsupported_group_size";
    const size_t count = size_t(n) * (k / groupSize);
    if (weight.scales.size() != count) return "invalid_scale_count";
    if (weight.dataType == DataType::INT4_GROUP && weight.group != k / groupSize) return "invalid_group_count";
    const bool zero = weight.dataType == DataType::INT8 ||
                      (weight.dataType == DataType::INT4_GROUP && weight.zeros.size() == count);
    if (zero ? weight.zeros.size() != count : weight.mins.size() != count) return "invalid_zero_or_min_count";
    for (size_t i = 0; i < count; i++) {
        if (weight.dataType == DataType::INT8 && (weight.zeros[i] < 0 || weight.zeros[i] > 255 ||
            std::floor(weight.zeros[i]) != weight.zeros[i])) return "int8_zero_must_be_uint8";
        float offset = zero ? -weight.scales[i] * weight.zeros[i] : weight.mins[i];
        if (!std::isfinite(weight.scales[i]) || !std::isfinite(offset)) return "nonfinite_quant_metadata";
    }
    return "";
#endif
}
}

/**
 * 在正式Linear入口执行Machete后端选择，native分支不改变旧实现。
 * @return true表示已执行；false只表示选择了原生路径。显式拒绝或运行错误抛异常。
 */
bool FastllmCudaTryMacheteLinear(const Data &input, Data &weight, const Data &bias, Data &output) {
    if (!Target(weight.dataType)) return false;
    const std::string policy = Policy();
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("Machete cudaGetDevice failed");
    const int dtype = int(input.dataType);
    std::shared_ptr<State> state;
    {
        std::lock_guard<std::mutex> guard(cacheMutex);
        const Key key(&weight, device, dtype, std::this_thread::get_id());
        auto found = states.find(key);
        if (found != states.end()) {
            state = found->second;
            if (state->policy != policy || state->source != weight.cudaData ||
                weight.dims != std::vector<int>({state->n, state->k})) {
                throw std::runtime_error("Machete backend/source changed after selection; reload the model");
            }
        } else {
            if (policy == "native") {
                Trace(weight, device, dtype, policy, "native", "explicit");
                return false;
            }
            int groupSize = 0;
            std::string reason = Support(input, weight, bias, output, groupSize, device);
            if (!reason.empty()) {
                Trace(weight, device, dtype, policy, policy == "machete" ? "none" : "native", reason);
                if (policy == "machete") throw std::runtime_error("Machete rejected weight '" + weight.name + "': " + reason);
                return false;
            }
#ifdef FASTLLM_ENABLE_MACHETE
            state = std::make_shared<State>();
            state->source = weight.cudaData;
            state->n = weight.dims[0];
            state->k = weight.dims[1];
            state->groupSize = groupSize;
            state->policy = policy;
            std::vector<float> offsets(weight.scales.size());
            bool zero = weight.dataType == DataType::INT8 ||
                        (weight.dataType == DataType::INT4_GROUP && weight.zeros.size() == offsets.size());
            for (size_t i = 0; i < offsets.size(); i++) {
                offsets[i] = zero ? -weight.scales[i] * weight.zeros[i] : weight.mins[i];
            }
            state->weight = fastllm_machete::Create(weight.dataType == DataType::INT8 ? 8 : 4,
                input.dataType == DataType::BFLOAT16, weight.cudaData, state->n, state->k,
                groupSize, weight.scales.data(), offsets.data(), cudaStreamPerThread);
            states.emplace(key, state);
            Trace(weight, device, dtype, policy, "machete", "a16_scale_offset_rounding", state.get());
#endif
        }
    }
#ifdef FASTLLM_ENABLE_MACHETE
    // 与正式CUDA编译的per-thread默认stream一致；不同线程和设备不共享workspace。
    std::lock_guard<std::mutex> submission(state->submissionMutex);
    state->weight->Run(input.cudaData, output.cudaData,
        bias.dims.empty() ? nullptr : static_cast<const float*>(bias.cudaData),
        input.Count(0) / state->k, cudaStreamPerThread);
    return true;
#else
    return false;
#endif
}

/** 在Data释放、复制或设备迁移前退休预打包资源，防止地址复用命中旧缓存。 */
void FastllmCudaReleaseMacheteCache(const Data *weight) {
    if (!weight || !Target(weight->dataType)) return;
    std::lock_guard<std::mutex> guard(cacheMutex);
    for (auto it = states.begin(); it != states.end();) {
        if (std::get<0>(it->first) == weight) it = states.erase(it);
        else ++it;
    }
    for (auto it = traces.begin(); it != traces.end();) {
        if (std::get<0>(*it) == weight) it = traces.erase(it);
        else ++it;
    }
}
