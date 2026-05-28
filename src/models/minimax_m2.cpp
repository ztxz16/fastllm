//
// Created by huangyuyang on 2/14/26.
//

#include "utils.h"

#include "minimax_m2.h"
#include "baseblock.h"
#include "executor.h"

#include <sstream>

#include <unordered_map>

#include <cstring>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <set>

#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

#ifdef USE_CUDA
    namespace {
        using namespace qwen3cuda;

        static std::atomic<int> minimaxM2ThreadTpNextPagedCacheBase(6000000);

        static std::string MinimaxM2TrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool MinimaxM2IsDisabledTpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool MinimaxM2IsPositiveIntegerSpec(const std::string &value) {
            if (value.empty()) {
                return false;
            }
            for (char c : value) {
                if (!std::isdigit((unsigned char)c)) {
                    return false;
                }
            }
            return std::atoi(value.c_str()) > 0;
        }

        static bool AppendMinimaxM2CudaDevicesFromSpec(const std::string &spec,
                                                       const std::string &type,
                                                       int defaultRatio,
                                                       std::vector<int> &devices,
                                                       std::map<int, int> &ratios) {
            std::map<int, int> parsedRatios;
            std::vector<int> parsed = ParseDeviceIds(spec, type, parsedRatios);
            if (parsed.empty() && (spec == "cuda" || spec == "multicuda")) {
                parsed.push_back(0);
            }
            bool added = false;
            for (int device : parsed) {
                if (device < 0 || device == 99999) {
                    continue;
                }
                int ratio = defaultRatio;
                auto ratioIt = parsedRatios.find(device);
                if (ratioIt != parsedRatios.end() && ratioIt->second > 0) {
                    ratio = ratioIt->second;
                }
                if (ratio <= 0) {
                    ratio = 1;
                }
                if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
                    devices.push_back(device);
                }
                ratios[device] += ratio;
                added = true;
            }
            return added;
        }

        static bool ParseMinimaxM2GPUForwardSpec(const std::string &rawSpec,
                                                 std::vector<int> &devices,
                                                 std::map<int, int> &ratios) {
            std::string spec = MinimaxM2TrimString(rawSpec);
            if (MinimaxM2IsDisabledTpSpec(spec)) {
                return false;
            }

            std::string lower = spec;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (lower == "1" || lower == "true" || lower == "on" || lower == "auto") {
                int count = FastllmCudaGetDeviceCount();
                for (int i = 0; i < count; i++) {
                    devices.push_back(i);
                    ratios[i] = 1;
                }
                return !devices.empty();
            }
            if (MinimaxM2IsPositiveIntegerSpec(lower)) {
                int requested = std::atoi(lower.c_str());
                int count = FastllmCudaGetDeviceCount();
                if (count > 0) {
                    requested = std::min(requested, count);
                }
                for (int i = 0; i < requested; i++) {
                    devices.push_back(i);
                    ratios[i] = 1;
                }
                return !devices.empty();
            }

            std::string parseSpec = spec;
            std::string type = "cuda";
            if (StartWith(lower, "multicuda")) {
                type = "multicuda";
            } else if (!StartWith(lower, "cuda")) {
                parseSpec = "cuda:" + spec;
            }
            return AppendMinimaxM2CudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetMinimaxM2GPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                                  std::vector<int> &devices,
                                                  std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env == nullptr || MinimaxM2IsDisabledTpSpec(MinimaxM2TrimString(env))) {
                env = std::getenv("FASTLLM_MINIMAX_M2_TP");
            }
            if (env != nullptr) {
                ParseMinimaxM2GPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        AppendMinimaxM2CudaDevicesFromSpec(it.first, "multicuda", it.second, devices, ratios);
                    }
                }
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "cuda")) {
                        AppendMinimaxM2CudaDevicesFromSpec(it.first, "cuda", it.second, devices, ratios);
                    }
                }
            }

            std::vector<int> uniqueDevices;
            std::set<int> seen;
            for (int device : devices) {
                if (device >= 0 && seen.insert(device).second) {
                    uniqueDevices.push_back(device);
                    if (ratios.find(device) == ratios.end() || ratios[device] <= 0) {
                        ratios[device] = 1;
                    }
                }
            }
            devices.swap(uniqueDevices);
            return !devices.empty();
        }

        static bool MinimaxM2DeviceSpecStartsWith(const std::string &device, const std::string &prefix) {
            std::string lower = device;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return lower == prefix || lower.rfind(prefix + ":", 0) == 0;
        }

        static bool MinimaxM2DeviceSpecIsCuda(const std::string &device) {
            return MinimaxM2DeviceSpecStartsWith(device, "cuda") ||
                   MinimaxM2DeviceSpecStartsWith(device, "multicuda");
        }

        static bool MinimaxM2DeviceMapAllCuda(const std::map<std::string, int> &deviceMap) {
            bool hasCuda = false;
            for (auto &it : deviceMap) {
                if (it.second <= 0) {
                    continue;
                }
                if (!MinimaxM2DeviceSpecIsCuda(it.first)) {
                    return false;
                }
                hasCuda = true;
            }
            return hasCuda;
        }

        static bool MinimaxM2MoeDeviceMapAllowsCudaOnly(const std::map<std::string, int> &moeDeviceMap) {
            return moeDeviceMap.empty() || MinimaxM2DeviceMapAllCuda(moeDeviceMap);
        }

        static bool MinimaxM2CanUseGPUForward(const std::map<std::string, int> &deviceMap,
                                              const std::map<std::string, int> &moeDeviceMap) {
            (void)moeDeviceMap;
            std::vector<int> devices;
            std::map<int, int> ratios;
            return GetMinimaxM2GPUForwardDevices(deviceMap, devices, ratios);
        }

        static bool GetMinimaxM2ThreadTpDevices(const std::map<std::string, int> &deviceMap,
                                                std::vector<int> &devices,
                                                std::map<int, int> &ratios) {
            if (!GetMinimaxM2GPUForwardDevices(deviceMap, devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static Executor &MinimaxM2ThreadLocalGenericExecutor() {
            static thread_local std::unique_ptr<Executor> executor;
            if (executor == nullptr) {
                executor.reset(new Executor());
            }
            return *executor;
        }

        class MinimaxM2ScopedGenericExecutor {
        public:
            explicit MinimaxM2ScopedGenericExecutor(const std::string &firstDevice)
                    : oldExecutor(GetExecutor()) {
                Executor &executor = MinimaxM2ThreadLocalGenericExecutor();
                if (!firstDevice.empty()) {
                    executor.SetFirstDevice(firstDevice);
                }
                SetCurrentThreadExecutor(&executor);
            }

            ~MinimaxM2ScopedGenericExecutor() {
                SetCurrentThreadExecutor(oldExecutor);
            }

            MinimaxM2ScopedGenericExecutor(const MinimaxM2ScopedGenericExecutor&) = delete;
            MinimaxM2ScopedGenericExecutor &operator=(const MinimaxM2ScopedGenericExecutor&) = delete;

        private:
            void *oldExecutor;
        };

        static void MinimaxM2ResetCpuScratch(Data &data) {
            if (data.isFake) {
                data.isFake = false;
                data.cpuData = nullptr;
                data.cudaData = nullptr;
                data.deviceData = nullptr;
                data.expansionSize = 0;
                data.expansionBytes = 0;
            } else {
                data.FreeSpace();
            }
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
            data.dataDevice = DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.lockInCPU = false;
            data.expansionDims.clear();
        }

        static DivisionScheme ExtractMinimaxM2FirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                for (int r = 0; r < (int)it.second.size(); r += 3) {
                    ret[it.first].push_back(it.second[r]);
                }
            }
            return ret;
        }

        static DivisionScheme ExtractMinimaxM2KRangeScheme(const DivisionScheme &qkvScheme,
                                                           int qWidth,
                                                           int kvWidth) {
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                for (int r = 1; r < (int)it.second.size(); r += 3) {
                    auto range = it.second[r];
                    if (range.first >= qWidth && range.second <= qWidth + kvWidth) {
                        ret[it.first].push_back({range.first - qWidth, range.second - qWidth});
                    }
                }
            }
            return ret;
        }

        static DivisionScheme ExtractMinimaxM2KVHeadScheme(const DivisionScheme &qkvScheme,
                                                           int qWidth,
                                                           int kvWidth,
                                                           int headDim) {
            (void)kvWidth;
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                for (int r = 1; r < (int)it.second.size(); r += 3) {
                    auto range = it.second[r];
                    int st = (range.first - qWidth) / headDim;
                    int end = (range.second - qWidth) / headDim;
                    ret[it.first].push_back({st, end});
                }
            }
            return ret;
        }

        static DataType ResolveMinimaxM2ThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT16 || modelType == DataType::BFLOAT16) {
                return modelType;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveMinimaxM2ThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FLOAT16 ||
                cacheType == DataType::BFLOAT16 ||
                cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static void PrepareMinimaxM2EmbeddingWeightType(Data &embedWeight,
                                                        DataType outputType,
                                                        bool requireCpu) {
            if (requireCpu || embedWeight.dataType != outputType) {
                if (embedWeight.multiDeviceData) {
                    embedWeight.ResetMultiDeviceState();
                }
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
            }
            if (embedWeight.dataType != outputType) {
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static void MinimaxM2CpuEmbeddingDirect(Data &inputIds, Data &embedWeight,
                                                Data &hiddenStates, DataType outputType) {
            PrepareMinimaxM2EmbeddingWeightType(embedWeight, outputType, true);
            inputIds.ToDevice(DataDevice::CPU);
            Executor *executor = (Executor*)GetExecutor();
            executor->RunOnDevice("cpu", "EmbeddingDirect",
                                  DataDict{{"input", &inputIds},
                                           {"weight", &embedWeight},
                                           {"output", &hiddenStates}},
                                  FloatDict(), IntDict());
        }

        static void PrepareMinimaxM2CudaEmbeddingWeightType(Data &embedWeight,
                                                            DataType outputType) {
            if (embedWeight.dataType != outputType) {
                embedWeight.ResetMultiDeviceState();
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static Data *CreateMinimaxM2CudaReplicaLike(const Data &source, int device) {
            Data *local = new Data(source.dataType);
            local->Resize(source.dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            FastllmCudaSetDevice(device);
            local->Allocate(false);
            return local;
        }

        static void PrepareMinimaxM2CpuEmbeddingHiddenStates(Data &hiddenStates,
                                                             const std::vector<int> &devices,
                                                             PersistentWorkerGroup &workerGroup) {
            AssertInFastLLM(!devices.empty(),
                            "MiniMax-M2 ForwardGPU CPU embedding got empty CUDA devices.\n");
            hiddenStates.ToDevice(DataDevice::CPU);
            AssertInFastLLM(hiddenStates.cpuData != nullptr,
                            "MiniMax-M2 ForwardGPU CPU embedding has no CPU data.\n");
            if (devices.size() == 1) {
                hiddenStates.ToDevice(DataDevice::CUDA, {devices[0]}, true);
                return;
            }

            uint64_t count = hiddenStates.Count(0);
            AssertInFastLLM(count <= (uint64_t)INT_MAX,
                            "MiniMax-M2 ForwardGPU CPU embedding result is too large for NCCL broadcast.\n");
            hiddenStates.ResetMultiDeviceState();
            hiddenStates.multiDeviceData = true;
            hiddenStates.tpLayout = TP_LAYOUT_REPLICATED;
            hiddenStates.tpAxis = -1;
            hiddenStates.tpGlobalDims = hiddenStates.dims;
            hiddenStates.dataDevice = DataDevice::CUDA;
            hiddenStates.dataDeviceIds = devices;

            int rootDevice = devices[0];
            for (int device : devices) {
                hiddenStates.multiDeviceDatas[device] =
                    CreateMinimaxM2CudaReplicaLike(hiddenStates, device);
            }
            FastllmCudaSetDevice(rootDevice);
            FastllmCudaCopyFromHostToDevice(hiddenStates.multiDeviceDatas[rootDevice]->cudaData,
                                            hiddenStates.cpuData,
                                            hiddenStates.GetBytes());

            std::vector<std::exception_ptr> errors(devices.size());
            workerGroup.Run(devices, [&](int r) {
                int device = devices[r];
                auto it = hiddenStates.multiDeviceDatas.find(device);
                AssertInFastLLM(it != hiddenStates.multiDeviceDatas.end() && it->second != nullptr,
                                "MiniMax-M2 ForwardGPU CPU embedding missing local CUDA replica.\n");
                FastllmCudaSetDevice(device);
                FastllmNcclBroadcast(it->second->cudaData, (int)count,
                                     (int)hiddenStates.dataType,
                                     rootDevice, device);
                ForceDeviceSync();
            }, errors);
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }

        static Data *EnsureMinimaxM2ThreadTpLocalCache(Data &root, int device, DataType localDataType) {
            root.multiDeviceData = true;
            root.dataDevice = DataDevice::CUDA;
            auto &local = root.multiDeviceDatas[device];
            if (local == nullptr) {
                local = new Data(localDataType);
                local->SetKVCache();
                local->cacheUid = root.cacheUid;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
            } else if (local->dataType != localDataType && local->dims.empty()) {
                local->dataType = localDataType;
                local->UpdateUnitSize();
            }
            return local;
        }

        static void PrepareMinimaxM2SingleCudaCache(Data &cache, int device, DataType localDataType) {
            cache.isKVCache = true;
            cache.lockInCPU = false;
            if (cache.dataType != localDataType && cache.dims.empty()) {
                cache.dataType = localDataType;
                cache.UpdateUnitSize();
            }
            cache.ToDevice(DataDevice::CUDA, {device}, false);
        }

        static void SyncMinimaxM2ThreadTpRootCacheMetaFromLocal(Data &root,
                                                                Data *firstLocal,
                                                                const std::vector<int> &devices,
                                                                const DivisionScheme &kvHeadScheme,
                                                                int globalKVHeads,
                                                                int headDim) {
            if (firstLocal == nullptr || firstLocal->dims.size() < 3) {
                return;
            }

            AssertInFastLLM(firstLocal->pageIndex.size() < 1000000,
                            "MiniMax-M2 ForwardGPU got invalid local paged cache pageIndex metadata.\n");

            std::vector<int> globalDims = {globalKVHeads, firstLocal->dims[1], headDim};
            bool samePageIndex = root.pageIndex.size() == firstLocal->pageIndex.size() &&
                (root.pageIndex.empty() || root.pageIndex.back() == firstLocal->pageIndex.back());
            bool fastMetaUpdate =
                root.multiDeviceData &&
                root.dataDevice == DataDevice::CUDA &&
                root.tpLayout == TP_LAYOUT_SHARDED &&
                root.tpAxis == 0 &&
                root.isKVCache &&
                root.isPagedKVCache == firstLocal->isPagedKVCache &&
                root.pageLen == firstLocal->pageLen &&
                root.pagedKVCacheData == firstLocal->pagedKVCacheData &&
                root.tpGlobalDims == globalDims &&
                root.dims == globalDims &&
                samePageIndex;
            if (fastMetaUpdate) {
                root.lastPageLen = firstLocal->lastPageLen;
                return;
            }

            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.multiDeviceData = true;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = 0;
            root.tpRanges = kvHeadScheme;
            root.tpGlobalDims = globalDims;
            if (root.dims != globalDims) {
                root.Resize(globalDims);
            }
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
        }

        static void MinimaxM2CudaClearMultiDeviceState(Data &data) {
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
        }

        static void MinimaxM2ZeroCudaLike(Data &dst, const Data &like, int device) {
            bool needReset = dst.isFake || dst.dataType != like.dataType ||
                             dst.dataDevice != DataDevice::CUDA || dst.dims != like.dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (!dst.isFake) {
                    dst.FreeSpace();
                } else {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                }
                MinimaxM2CudaClearMultiDeviceState(dst);
                dst.dataType = like.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(like.dims);
            }
            dst.Allocate();
            if (dst.cudaData != nullptr) {
                FastllmCudaMemset0(dst.cudaData, dst.GetBytes());
            }
        }

        static bool MinimaxM2HasLocalMoeShard(const std::vector<Data*> &localWeights) {
            for (int i = 2; i + 1 < (int)localWeights.size(); i += 2) {
                Data *gateup = localWeights[i];
                Data *down = localWeights[i + 1];
                if (gateup != nullptr && down != nullptr &&
                    gateup->dims.size() == 2 && down->dims.size() == 2 &&
                    gateup->dims[0] > 0 && down->dims[1] > 0) {
                    return true;
                }
            }
            return false;
        }

        static void MinimaxM2CudaSigmoid(Qwen3CudaDirectRunner &runner,
                                         Data &input, Data &output) {
            runner.Run("Sigmoid",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }
    }
#endif

    MinimaxM2Model::MinimaxM2Model() {
        this->model_type = "minimax_m2";
        this->model_struct = "minimax_m2";
        this->use_new_engine = true;

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.w2.weight", "model.layers.*.w3.weight",
            "model.layers.*.w1.weight",  "model.layers.*.w1.weight", "model.layers.*.w1w3.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight",
            "model.layers.*.block_sparse_moe.*.weight"
        };
    }

    void MinimaxM2Model::InitParams() {
        basellm::InitParams();
        num_experts = atoi(this->weight.dicts["num_local_experts"].c_str());
        num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        norm_topk_prob = false;

        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        embed_dim = head_dim * num_attention_heads;
        if (this->weight.dicts.find("rotary_dim") != this->weight.dicts.end()) {
            rotary_dim = atoi(this->weight.dicts["rotary_dim"].c_str());
        }

        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("use_qk_norm") != this->weight.dicts.end()) {
            use_qk_norm = (this->weight.dicts["use_qk_norm"] == "true");
        }
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));

        for (int i = 0; i < block_cnt; i++) {
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );
        }

        for (int i = 0; i < block_cnt; i++) {
            for (int j = 0; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w1.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w3.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                this->AddSpecialWeight(swigluWeightName, "linearSwiglu", i);
                this->AddSpecialWeight(downWeightName, "linearColumn", i);
                
                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }
        }
    }

    Data &MinimaxM2Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    bool MinimaxM2Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetMinimaxM2ThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    std::pair<std::vector<float>, std::vector<float>> MinimaxM2Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }
        float scale = rope_type == RoPEType::LINEAR_SCALE ? factor : 1.0;
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int MinimaxM2Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (MinimaxM2CanUseGPUForward(this->deviceMap, this->moeDeviceMap) &&
            GetMinimaxM2GPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens,
                              &batchLogits)[0];
        }
#endif
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    void MinimaxM2Model::ForwardSingleGPU(
            int gpuId,
            std::map <int, int> ratios,
            int batch,
            const Data &inputIds,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            bool all1,
            bool isPrefill,
            bool tensorParallel,
            bool firstTensorParallelRank,
            int pagedCacheLayerOffset,
            Data &logits,
            Data *precomputedHiddenStates) {
#ifndef USE_CUDA
        ErrorInFastLLM("MiniMax-M2 ForwardSingleGPU requires CUDA.\n");
#else
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "MiniMax-M2 ForwardSingleGPU got invalid GPU ratio.\n");
        FastllmCudaSetDevice(gpuId);

        Qwen3CudaDirectRunner cudaRunner(gpuId);
        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("MiniMax-M2 ForwardSingleGPU missing local tensor: " + name + ".\n");
            return nullptr;
        };
        auto requireLocalBias = [&](Data &data, const std::string &name) -> Data* {
            if (data.dims.empty()) {
                return GetEmptyData();
            }
            return requireLocal(data, name);
        };

        const DataType computeType = ResolveMinimaxM2ThreadTpComputeType(this->dataType);
        const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
        Data localHiddenStates;
        Data *hiddenStatesPtr = nullptr;
        if (precomputedHiddenStates != nullptr) {
            hiddenStatesPtr = requireLocal(*precomputedHiddenStates, "precomputedHiddenStates");
        } else {
            Qwen3CudaEmbeddingDirect(cudaRunner,
                                     *requireLocal((Data&)inputIds, "inputIds"),
                                     *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                     localHiddenStates);
            hiddenStatesPtr = &localHiddenStates;
        }
        Data &hiddenStates = *hiddenStatesPtr;
        if (hiddenStates.dataType != computeType) {
            Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
        }

        Data attenInput, qkv, q, attenOutput, attenLastOutput;
        Data routerLogits, routerLogitsTemp, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, moeFinal;
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        bool generatedAppendParams = false;
        bool generatedDecodeParams = false;
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;
        bool useMappedNonCudaMoe = !MinimaxM2MoeDeviceMapAllowsCudaOnly(this->moeDeviceMap);

        for (int i = 0; i < block_cnt; i++) {
            std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string gateWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.gate.weight";
            std::string gateBiasName = "model.layers." + std::to_string(i) + ".block_sparse_moe.e_score_correction_bias";

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[inputRmsName], inputRmsName),
                             rms_norm_eps, attenInput);
            Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
            int group = num_attention_heads / num_key_value_heads;
            int localKVHeads = localMergeW->tpKVHeads > 0 ?
                localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
            int localQHeads = localMergeW->tpQHeads > 0 ?
                localMergeW->tpQHeads : localKVHeads * group;
            AssertInFastLLM(localQHeads > 0 && localKVHeads > 0,
                            "MiniMax-M2 ForwardSingleGPU got empty local attention shard.\n");
            Qwen3CudaAttentionPagedBlock(
                cudaRunner,
                &attenInput,
                localMergeW, requireLocalBias(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                GetEmptyData(), GetEmptyData(),
                GetEmptyData(), GetEmptyData(),
                GetEmptyData(), GetEmptyData(),
                requireLocal(weight[qNormName], qNormName),
                requireLocal(weight[kNormName], kNormName),
                GetEmptyData(), GetEmptyData(),
                requireLocal(weight[oWeightName], oWeightName),
                requireLocalBias(GetThreadTensorParallelBias(oBiasName), oBiasName),
                requireLocal((Data&)positionIds, "positionIds"),
                &pastKeyValues,
                &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                nullptr,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendParams, &generatedDecodeParams,
                batch, block_cnt, i,
                seqLens,
                localQHeads, localKVHeads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                false,
                true,
                pagedCacheLayerOffset,
                true,
                false,
                false,
                -1,
                num_attention_heads * head_dim,
                num_key_value_heads * head_dim
            );
            if (tensorParallel) {
                DataType residualType = hiddenStates.dataType;
                if (firstTensorParallelRank) {
                    if (attenOutput.dataType == residualType) {
                        Qwen3CudaLinearAddBlock(cudaRunner, &attenOutput,
                                               requireLocal(weight[oWeightName], oWeightName),
                                               requireLocalBias(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                               &attenLastOutput, &hiddenStates);
                    } else {
                        Qwen3CudaLinear(cudaRunner, attenOutput,
                                        *requireLocal(weight[oWeightName], oWeightName),
                                        *requireLocalBias(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                        attenLastOutput);
                        if (attenLastOutput.dataType != residualType) {
                            Qwen3CudaToDataType(cudaRunner, attenLastOutput, residualType);
                        }
                        Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
                    }
                } else {
                    Qwen3CudaLinear(cudaRunner, attenOutput,
                                    *requireLocal(weight[oWeightName], oWeightName),
                                    *requireLocalBias(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                    hiddenStates);
                    if (hiddenStates.dataType != residualType) {
                        Qwen3CudaToDataType(cudaRunner, hiddenStates, residualType);
                    }
                }
                FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                     hiddenStates.Count(0), hiddenStates.dataType, gpuId);
            } else {
                Qwen3CudaLinear(cudaRunner, attenOutput,
                                *requireLocal(weight[oWeightName], oWeightName),
                                *requireLocalBias(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                attenLastOutput);
                if (attenLastOutput.dataType != hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, attenLastOutput, hiddenStates.dataType);
                }
                Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
            }

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[postRmsName], postRmsName),
                             rms_norm_eps, attenInput);
            int localBatch = attenInput.dims[0];
            int localLen = attenInput.dims[1];
            attenInput.Reshape({localBatch * localLen, attenInput.dims[2]});
            Qwen3CudaLinear(cudaRunner, attenInput,
                            *requireLocal(weight[gateWeightName], gateWeightName),
                            *GetEmptyData(), routerLogits, true);
            Qwen3CudaConvertToDataType(cudaRunner, routerLogits, routerLogitsTemp, DataType::FLOAT32);
            MinimaxM2CudaSigmoid(cudaRunner, routerLogitsTemp, routerLogitsTemp);
            Data *localGateBias = nullptr;
            if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
            }
            Qwen3CudaSelectExpert(cudaRunner, routerLogitsTemp, expertIndex, expertScore,
                                  this->num_experts_per_tok, true,
                                  this->routed_scaling_factor, localGateBias);
            if (useMappedNonCudaMoe) {
                if (!tensorParallel || firstTensorParallelRank) {
                    const auto &effectiveMoeMap = this->moeDeviceMap.empty() ? this->deviceMap : this->moeDeviceMap;
                    std::string selectedMoeDevice = SelectDeviceFromMap(effectiveMoeMap, i + 1, block_cnt);
                    MinimaxM2ResetCpuScratch(moeFinal);
                    FastllmCudaSetDevice(gpuId);
                    MinimaxM2ScopedGenericExecutor scopedExecutor(selectedMoeDevice);
                    MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                                  &weights[i], &biass[i],
                                  &w1, &w2, &w3,
                                  &tempInput, &tempOutput,
                                  1.0f, &moeFinal, i,
                                  computeType, threadTpMoeAtype,
                                  &moeInputTemp, &moeOutputTemp);
                    FastllmCudaSetDevice(gpuId);
                    if (moeFinal.dataDevice != DataDevice::CUDA || moeFinal.cudaData == nullptr ||
                        (!moeFinal.dataDeviceIds.empty() && moeFinal.dataDeviceIds[0] != gpuId)) {
                        moeFinal.ToDevice(DataDevice::CUDA, {gpuId}, true);
                    }
                } else {
                    MinimaxM2ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
            } else {
                auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                if (MinimaxM2HasLocalMoeShard(localWeights)) {
                    Qwen3CudaMergeMOEBlock(cudaRunner, &attenInput, &expertIndex, &expertScore,
                                           &localWeights, &localBiass,
                                           &w1, &w2, &w3,
                                           &tempInput, &tempOutput,
                                           1.0f, &moeFinal, i,
                                           computeType, threadTpMoeAtype,
                                           &moeInputTemp, &moeOutputTemp);
                } else {
                    MinimaxM2ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
            }
            moeFinal.Reshape(hiddenStates.dims);
            if (moeFinal.dataType != hiddenStates.dataType) {
                Qwen3CudaToDataType(cudaRunner, moeFinal, hiddenStates.dataType);
            }
            if (tensorParallel) {
                if (firstTensorParallelRank) {
                    Qwen3CudaAddTo(cudaRunner, moeFinal, hiddenStates);
                }
                FastllmNcclAllReduce(moeFinal.cudaData, hiddenStates.cudaData,
                                     moeFinal.Count(0), moeFinal.dataType, gpuId);
            } else {
                Qwen3CudaAddTo(cudaRunner, hiddenStates, moeFinal);
            }
        }

        Data lastHiddenStates;
        Data *headInput = &hiddenStates;
        if (!all1) {
            int total = 0;
            std::vector<Data> lastToks(seqLens.size());
            std::vector<Data*> lastTokPointers;
            lastTokPointers.reserve(seqLens.size());
            for (int b = 0; b < (int)seqLens.size(); b++) {
                Qwen3CudaSplit(cudaRunner, hiddenStates, 1,
                               total + seqLens[b] - 1, total + seqLens[b],
                               lastToks[b]);
                total += seqLens[b];
                lastTokPointers.push_back(&lastToks[b]);
            }
            Qwen3CudaCatBatch(cudaRunner, lastTokPointers, 1, lastHiddenStates);
            headInput = &lastHiddenStates;
        }

        Qwen3CudaRMSNorm(cudaRunner, *headInput,
                         *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                         rms_norm_eps, *headInput);
        Qwen3CudaLinear(cudaRunner, *headInput,
                        *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                        *requireLocalBias(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                          "lm_head.weight.tp_bias"),
                        logits);
        Qwen3CudaToDataType(cudaRunner, logits, DataType::FLOAT32);
#endif
    }

    std::vector <int> MinimaxM2Model::ForwardGPU(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
#ifndef USE_CUDA
        return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                         pastKeyValues, generationConfigs, lastTokens, retLogits);
#else
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!MinimaxM2CanUseGPUForward(this->deviceMap, this->moeDeviceMap) ||
            !GetMinimaxM2GPUForwardDevices(this->deviceMap, devices, ratios)) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;
        bool useMappedNonCudaMoe = !MinimaxM2MoeDeviceMapAllowsCudaOnly(this->moeDeviceMap);
        bool useCpuEmbedding = !GetCudaEmbedding() || GetLowMemMode();
        const DataType computeType = ResolveMinimaxM2ThreadTpComputeType(this->dataType);
        if (!useCpuEmbedding) {
            PrepareMinimaxM2CudaEmbeddingWeightType(weight["model.embed_tokens.weight"], computeType);
        }

        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "MiniMax-M2 ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "MiniMax-M2 ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "MiniMax-M2 ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "MiniMax-M2 ForwardGPU doesn't support CPU KV cache.\n");
        if (tensorParallel) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "MiniMax-M2 ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = minimaxM2ThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }

        if (weights.empty()) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(nullptr);
                weights[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    std::string gateupWeightName = "model.layers." + std::to_string(i) +
                        ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight";
                    std::string downWeightName = "model.layers." + std::to_string(i) +
                        ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight";
                    AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                    "MiniMax-M2 ForwardGPU requires merged expert w1w3 weight.\n");
                    AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                    "MiniMax-M2 ForwardGPU requires expert w2 weight.\n");
                    Data &gateup = weight[gateupWeightName];
                    Data &down = weight[downWeightName];
                    gateup.tpLinearType = TP_LINEAR_ROW;
                    gateup.tpPackType = TP_PACK_GATEUP;
                    down.tpLinearType = TP_LINEAR_COLUMN;
                    weights[i].push_back(&gateup);
                    weights[i].push_back(&down);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
        }

        int seqLen = inputIds.dims[1];
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;

        Data allPositionIds;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector<float> vPositionIds;
            vPositionIds.reserve(batch);
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector<float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                AssertInFastLLM(positionIds[b] != nullptr,
                                "MiniMax-M2 ForwardGPU: null positionIds.\n");
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        if (tensorParallel) {
            PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
            PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
        }

        std::vector<DivisionScheme> localKvHeadSchemes;
        DivisionScheme localLmHeadScheme;
        const std::vector<DivisionScheme> *kvHeadSchemes = &localKvHeadSchemes;
        const DivisionScheme *lmHeadScheme = &localLmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        auto hasMoeCache = [&](const std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                               const std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache) {
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto weightIt = weightCache.find(device);
                auto biasIt = biasCache.find(device);
                if (weightIt == weightCache.end() || biasIt == biasCache.end() ||
                    (int)weightIt->second.size() != block_cnt || (int)biasIt->second.size() != block_cnt) {
                    return false;
                }
                for (int i = 0; i < block_cnt; i++) {
                    if ((int)weightIt->second[i].size() != expectedSize ||
                        (int)biasIt->second[i].size() != expectedSize) {
                        return false;
                    }
                    for (int j = 2; j < expectedSize; j++) {
                        if (weightIt->second[i][j] == nullptr) {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        auto fillMoeCache = [&](std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                                std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache,
                                bool useLocalShards) {
            weightCache.clear();
            biasCache.clear();
            for (int device : devices) {
                auto &deviceWeights = weightCache[device];
                auto &deviceBiass = biasCache[device];
                deviceWeights.resize(block_cnt);
                deviceBiass.resize(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    auto &layerWeights = deviceWeights[i];
                    auto &layerBiass = deviceBiass[i];
                    layerWeights.reserve(this->num_experts * 2 + 2);
                    layerBiass.reserve(this->num_experts * 2 + 2);
                    layerWeights.push_back(nullptr);
                    layerWeights.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight";
                        auto getLocalOrRoot = [&](Data &data) -> Data* {
                            if (useLocalShards) {
                                auto it = data.multiDeviceDatas.find(device);
                                if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                                    return it->second;
                                }
                            }
                            return &data;
                        };
                        layerWeights.push_back(getLocalOrRoot(weight[gateupWeightName]));
                        layerWeights.push_back(getLocalOrRoot(weight[downWeightName]));
                        layerBiass.push_back(nullptr);
                        layerBiass.push_back(nullptr);
                    }
                }
            }
        };

        if (tensorParallel) {
            auto usePreparedThreadTpSchemes = [&]() {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "MiniMax-M2 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                (useMappedNonCudaMoe ||
                                 hasMoeCache(threadTpMoeWeights, threadTpMoeBiass)),
                                "MiniMax-M2 ForwardGPU thread TP cached weight schemes are incomplete.\n");
                kvHeadSchemes = &threadTpKVHeadSchemes;
                lmHeadScheme = &threadTpLmHeadScheme;
            };

            if (threadTpWeightsPrepared.load(std::memory_order_acquire)) {
                usePreparedThreadTpSchemes();
            } else {
                std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
                if (!threadTpWeightsPrepared.load(std::memory_order_relaxed)) {
                    auto prepareReplicated = [&](const std::string &name) {
                        PrepareMultiCudaReplicatedData(this->weight[name], devices, true);
                    };
                    if (!useCpuEmbedding) {
                        prepareReplicated("model.embed_tokens.weight");
                    }
                    prepareReplicated("model.norm.weight");

                    threadTpKVHeadSchemes.assign(block_cnt, DivisionScheme());
                    for (int i = 0; i < block_cnt; i++) {
                        std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                        std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                        std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                        std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                        std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                        std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                        std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                        std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                        std::string gateWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.gate.weight";
                        std::string gateBiasName = "model.layers." + std::to_string(i) + ".block_sparse_moe.e_score_correction_bias";

                        AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires merged qkv weight.\n");
                        AssertInFastLLM(weight.weight.find(qNormName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires q_norm weight.\n");
                        AssertInFastLLM(weight.weight.find(kNormName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires k_norm weight.\n");
                        AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires router gate weight.\n");

                        prepareReplicated(inputRmsName);
                        prepareReplicated(postRmsName);
                        prepareReplicated(gateWeightName);
                        if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                            prepareReplicated(gateBiasName);
                        }

                        Data &mergeW = weight[mergeQkvWeightName];
                        Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                        mergeW.tpPackType = TP_PACK_QKV;
                        mergeW.tpQHeads = num_attention_heads;
                        mergeW.tpKVHeads = num_key_value_heads;
                        mergeW.tpHeadDim = head_dim;
                        std::vector<int> devCopy = devices;
                        DivisionScheme qkvScheme = BuildMultiCudaRowSplitScheme(mergeW, devCopy, ratios);
                        AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                        "MiniMax-M2 ForwardGPU failed to split " + mergeQkvWeightName + ".\n");

                        int qWidth = num_attention_heads * head_dim;
                        int kvWidth = num_key_value_heads * head_dim;
                        DivisionScheme qScheme = ExtractMinimaxM2FirstRangeScheme(qkvScheme);
                        DivisionScheme kScheme = ExtractMinimaxM2KRangeScheme(qkvScheme, qWidth, kvWidth);
                        threadTpKVHeadSchemes[i] =
                            ExtractMinimaxM2KVHeadScheme(qkvScheme, qWidth, kvWidth, head_dim);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight1D(weight[qNormName], devCopy, qScheme),
                                        "MiniMax-M2 ForwardGPU failed to split " + qNormName + ".\n");
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight1D(weight[kNormName], devCopy, kScheme),
                                        "MiniMax-M2 ForwardGPU failed to split " + kNormName + ".\n");

                        Data &oB = GetThreadTensorParallelBias(oBiasName);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, qScheme, 1),
                                        "MiniMax-M2 ForwardGPU failed to split " + oWeightName + ".\n");

                        if (useMappedNonCudaMoe) {
                            continue;
                        }

                        DivisionScheme gateScheme;
                        for (int j = 0; j < this->num_experts; j++) {
                            std::string gateupWeightName = "model.layers." + std::to_string(i) +
                                ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight";
                            std::string downWeightName = "model.layers." + std::to_string(i) +
                                ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight";
                            AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                            "MiniMax-M2 ForwardGPU requires merged expert w1w3 weight.\n");
                            AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                            "MiniMax-M2 ForwardGPU requires expert w2 weight.\n");

                            Data &gateup = weight[gateupWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            devCopy = devices;
                            gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                            "MiniMax-M2 ForwardGPU failed to split " + gateupWeightName + ".\n");

                            Data &down = weight[downWeightName];
                            Data &downBias = GetThreadTensorParallelBias(downWeightName + ".tp_bias");
                            down.tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractMinimaxM2FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1),
                                            "MiniMax-M2 ForwardGPU failed to split " + downWeightName + ".\n");
                        }
                    }

                    if (useMappedNonCudaMoe) {
                        threadTpMoeWeights.clear();
                        threadTpMoeBiass.clear();
                    } else {
                        fillMoeCache(threadTpMoeWeights, threadTpMoeBiass, true);
                    }

                    Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                    std::vector<int> devCopy = devices;
                    threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0),
                                    "MiniMax-M2 ForwardGPU failed to split lm_head.weight.\n");

                    threadTpPreparedDevices = devices;
                    threadTpPreparedRatios = ratios;
                    threadTpWeightsPrepared.store(true, std::memory_order_release);
                }
                usePreparedThreadTpSchemes();
            }
        } else {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            if (!singleGpuWeightsPrepared.load(std::memory_order_relaxed) ||
                (!useMappedNonCudaMoe && !hasMoeCache(singleGpuMoeWeights, singleGpuMoeBiass))) {
                int device = devices[0];
                for (int i = 0; i < block_cnt; i++) {
                    std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                    std::string gateWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.gate.weight";
                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "MiniMax-M2 ForwardGPU requires merged qkv weight.\n");
                    AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                    "MiniMax-M2 ForwardGPU requires router gate weight.\n");

                    Data &mergeW = weight[mergeQkvWeightName];
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = num_attention_heads;
                    mergeW.tpKVHeads = num_key_value_heads;
                    mergeW.tpHeadDim = head_dim;

                    if (useMappedNonCudaMoe) {
                        continue;
                    }
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight";
                        AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires merged expert w1w3 weight.\n");
                        AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                        "MiniMax-M2 ForwardGPU requires expert w2 weight.\n");
                        Data &gateup = weight[gateupWeightName];
                        Data &down = weight[downWeightName];
                        gateup.tpLinearType = TP_LINEAR_ROW;
                        gateup.tpPackType = TP_PACK_GATEUP;
                        down.tpLinearType = TP_LINEAR_COLUMN;
                        gateup.ToDevice(DataDevice::CUDA, {device}, true);
                        down.ToDevice(DataDevice::CUDA, {device}, true);
                    }
                }
                if (!useMappedNonCudaMoe) {
                    fillMoeCache(singleGpuMoeWeights, singleGpuMoeBiass, false);
                } else {
                    singleGpuMoeWeights.clear();
                    singleGpuMoeBiass.clear();
                }
                singleGpuWeightsPrepared.store(true, std::memory_order_release);
            }
            localKvHeadSchemes.assign(block_cnt, DivisionScheme());
            for (int i = 0; i < block_cnt; i++) {
                localKvHeadSchemes[i][devices[0]].push_back({0, num_key_value_heads});
            }
            localLmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        if (tensorParallel && !useCpuEmbedding) {
            PrepareMultiCudaReplicatedData(weight["model.embed_tokens.weight"], devices, true);
        }
        Data cpuEmbeddingHiddenStates;
        Data *precomputedHiddenStates = nullptr;
        if (useCpuEmbedding) {
            Data cpuInputIds;
            cpuInputIds.CopyFrom(inputIds);
            MinimaxM2CpuEmbeddingDirect(cpuInputIds, weight["model.embed_tokens.weight"],
                                        cpuEmbeddingHiddenStates, computeType);
            PrepareMinimaxM2CpuEmbeddingHiddenStates(cpuEmbeddingHiddenStates, devices, threadTpWorkerGroup);
            precomputedHiddenStates = &cpuEmbeddingHiddenStates;
        }

        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues;
        if (tensorParallel) {
            localPastKeyValues.resize(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                localPastKeyValues[r].resize(pastKeyValues.size());
                for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                    DataType keyCacheType = ResolveMinimaxM2ThreadTpCacheType(
                        pastKeyValues[i].first->dataType, computeType);
                    DataType valueCacheType = ResolveMinimaxM2ThreadTpCacheType(
                        pastKeyValues[i].second->dataType, computeType);
                    localPastKeyValues[r][i].first = EnsureMinimaxM2ThreadTpLocalCache(
                        *pastKeyValues[i].first, device, keyCacheType);
                    localPastKeyValues[r][i].second = EnsureMinimaxM2ThreadTpLocalCache(
                        *pastKeyValues[i].second, device, valueCacheType);
                }
            }
        } else {
            int device = devices[0];
            for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                DataType keyCacheType = ResolveMinimaxM2ThreadTpCacheType(
                    pastKeyValues[i].first->dataType, computeType);
                DataType valueCacheType = ResolveMinimaxM2ThreadTpCacheType(
                    pastKeyValues[i].second->dataType, computeType);
                PrepareMinimaxM2SingleCudaCache(*pastKeyValues[i].first, device, keyCacheType);
                PrepareMinimaxM2SingleCudaCache(*pastKeyValues[i].second, device, valueCacheType);
            }
        }

        std::vector<std::exception_ptr> errors(devices.size());
        std::vector<Data> localLogits(devices.size());
        if (devices.size() == 1) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            ForwardSingleGPU(devices[0], ratios, batch, gpuInputIds, allPositionIds,
                             seqLens, pastKeyValues, all1, isPrefill,
                             false, true, threadTpPagedCacheBase, localLogits[0],
                             precomputedHiddenStates);
        } else {
            threadTpWorkerGroup.Run(devices, [&](int r) {
                ForwardSingleGPU(devices[r], ratios, batch, gpuInputIds, allPositionIds,
                                 seqLens, localPastKeyValues[r], all1, isPrefill,
                                 tensorParallel, r == 0,
                                 threadTpPagedCacheBase + r * block_cnt,
                                 localLogits[r], precomputedHiddenStates);
                FastllmCudaSetDevice(devices[r]);
                ForceDeviceSync();
            }, errors);
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }

        if (tensorParallel) {
            auto validLocalMeta = [](Data *data) {
                return data != nullptr && data->dims.size() >= 3;
            };
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    int idx = b * block_cnt + i;
                    Data *localKeyMeta = !localPastKeyValues.empty() &&
                        idx < (int)localPastKeyValues[0].size() ? localPastKeyValues[0][idx].first : nullptr;
                    Data *localValueMeta = !localPastKeyValues.empty() &&
                        idx < (int)localPastKeyValues[0].size() ? localPastKeyValues[0][idx].second : nullptr;
                    if ((!validLocalMeta(localKeyMeta) || !validLocalMeta(localValueMeta)) &&
                         localPastKeyValues.size() > 1) {
                        for (auto &rankPastKeyValues : localPastKeyValues) {
                            if (idx < (int)rankPastKeyValues.size()) {
                                Data *candidateKey = rankPastKeyValues[idx].first;
                                Data *candidateValue = rankPastKeyValues[idx].second;
                                if (!validLocalMeta(localKeyMeta) && validLocalMeta(candidateKey)) {
                                    localKeyMeta = candidateKey;
                                }
                                if (!validLocalMeta(localValueMeta) && validLocalMeta(candidateValue)) {
                                    localValueMeta = candidateValue;
                                }
                                if (validLocalMeta(localKeyMeta) && validLocalMeta(localValueMeta)) {
                                    break;
                                }
                            }
                        }
                    }
                    SyncMinimaxM2ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].first, localKeyMeta,
                                                                devices, (*kvHeadSchemes)[i],
                                                                num_key_value_heads, head_dim);
                    SyncMinimaxM2ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].second, localValueMeta,
                                                                devices, (*kvHeadSchemes)[i],
                                                                num_key_value_heads, head_dim);
                }
            }
        }

        int vocabSize = lmHead.dims[0];
        Data fullLogits(DataType::FLOAT32);
        fullLogits.Resize({batch, vocabSize});
        fullLogits.Allocate();
        std::fill((float*)fullLogits.cpuData,
                  (float*)fullLogits.cpuData + fullLogits.Count(0), -1.0e30f);

        for (int r = 0; r < (int)devices.size(); r++) {
            int device = devices[r];
            localLogits[r].ToDevice(DataDevice::CPU);
            int localVocab = localLogits[r].dims.back();
            int rows = localLogits[r].Count(0) / localVocab;
            AssertInFastLLM(rows == batch,
                            "MiniMax-M2 ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            auto schemeIt = lmHeadScheme->find(device);
            AssertInFastLLM(schemeIt != lmHeadScheme->end(),
                            "MiniMax-M2 ForwardGPU: missing lm_head split scheme.\n");
            for (auto &range : schemeIt->second) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "MiniMax-M2 ForwardGPU: invalid lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    memcpy(dst + (long long)b * vocabSize + range.first,
                           src + (long long)b * localVocab + localOffset,
                           (size_t)len * sizeof(float));
                }
                localOffset += len;
            }
        }
        ResetLogitsOfEOS(batch, &fullLogits, pastKeyValues, generationConfigs);

        std::vector<int> lastRet;
        LastTokensUnit emptyLastTokens;
        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].output_logits && retLogits != nullptr &&
                b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                (*retLogits)[b]->resize(vocabSize);
                memcpy((float*)(*retLogits)[b]->data(),
                       (float*)fullLogits.cpuData + (long long)b * vocabSize,
                       (size_t)vocabSize * sizeof(float));
            }
            const LastTokensUnit &unit = b < (int)lastTokens.units.size() ?
                lastTokens.units[b] : emptyLastTokens;
            lastRet.push_back(LLMSampling(fullLogits, b, generationConfigs[b], unit));
        }
        return lastRet;
#endif
    }

    std::vector <int> MinimaxM2Model::ForwardV2(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        if (IsThreadTensorParallelEnabled()) {
            return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, generationConfigs, lastTokens, retLogits);
        }

        int seqLen = inputIds.dims[1];

        Data qkv, q, k, v;
        Data embeddingResult, hiddenStates, attenInput, attenLastOutput;
        Data w1, w2, w3, routerLogits, routerLogitsTemp, attenPart, moePart, moeFinal;
        Data tempInput, tempOutput;
        Data moeInputTemp, moeOutputTemp;
        std::vector <Data*> pointersK;
        pointersK.resize(batch);



        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        Data qSizes, pageSizes, pageIndexs, lastPageLens;
        Data insertIndexs, insertPositions;
        Data attenOutput;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;
        Data expertIndex, expertScore;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], embeddingResult);
        ToDataType(embeddingResult, hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];

        if (weights.size() == 0) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(nullptr);
                weights[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w1w3.weight"]);
                    weights[i].push_back(&weight["model.layers." + std::to_string(i) + ".block_sparse_moe.experts." + std::to_string(j) + ".w2.weight"]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";

            bool hasMergeQkv = (weight.weight.find(mergeQkvWeightName) != weight.weight.end());
            Data *mergeW = hasMergeQkv ? &weight[mergeQkvWeightName] : GetEmptyData();
            Data *mergeB = hasMergeQkv ? &weight[mergeQkvBiasName] : GetEmptyData();
            Data *qW = hasMergeQkv ? GetEmptyData() : &weight[qWeightName];
            Data *qB = hasMergeQkv ? GetEmptyData() : &weight[qBiasName];
            Data *kW = hasMergeQkv ? GetEmptyData() : &weight[kWeightName];
            Data *kB = hasMergeQkv ? GetEmptyData() : &weight[kBiasName];
            Data *vW = hasMergeQkv ? GetEmptyData() : &weight[vWeightName];
            Data *vB = hasMergeQkv ? GetEmptyData() : &weight[vBiasName];

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            AttentionPagedBlock(
                &attenInput,
                mergeW, mergeB,
                qW, qB, kW, kB, vW, vB,
                &weight[qNormName], &weight[kNormName],
                GetEmptyData(), GetEmptyData(),
                &weight[oWeightName], &weight[oBiasName],
                &allPositionIds,
                &pastKeyValues, &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendPagedCacheBatchParams, &generatedBatchDecodeParams,
                batch, block_cnt, i,
                seqLens,
                num_attention_heads, num_key_value_heads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                false,
                true
            );
            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);

            {
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".block_sparse_moe.gate.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".block_sparse_moe.e_score_correction_bias";

                int curBatch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({curBatch * len, attenInput.dims[2]});

                Linear(attenInput, weight[gateWeightName], *GetEmptyData(), routerLogits, true);
                ToDataType(routerLogits, routerLogitsTemp, DataType::FLOAT32);
                bool needNorm = true;
                Sigmoid(routerLogitsTemp, routerLogitsTemp);

                if (weight.weight.find("model.layers." + std::to_string(i) + ".block_sparse_moe.experts.0.w1w3.weight") != weight.weight.end()
                    && CanRunMergeMOE(attenInput, biass[i])) {
                    SelectExpert(routerLogitsTemp, expertIndex, expertScore, this->num_experts_per_tok, needNorm,
                                this->routed_scaling_factor, weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr);
                }
                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                                  &weights[i], &biass[i],
                                  &w1, &w2, &w3, &tempInput, &tempOutput,
                                  1.0f, &moeFinal, i,
                                  this->dataType, this->moeAtype,
                                  &moeInputTemp, &moeOutputTemp);
                moeFinal.Reshape(hiddenStates.dims);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, moeFinal);
            }
        }

        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight["model.norm.weight"], &weight["lm_head.weight"],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool MinimaxM2Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    void MinimaxM2Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }

                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string MinimaxM2Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string MinimaxM2Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void MinimaxM2Model::WarmUp() {
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        this->num_experts_per_tok = this->num_experts;

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->num_experts_per_tok = oldTopk;
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
