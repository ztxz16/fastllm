#ifndef FASTLLM_GLM5_NEXT_TP_H
#define FASTLLM_GLM5_NEXT_TP_H

#include "utils/persistent_worker_group.h"
#ifdef USE_CUDA
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif
#include <condition_variable>
#include <chrono>
#include <stdexcept>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace fastllm {
    struct Glm5NextModel::ThreadTpState {
        using Cache = std::vector<std::pair<Data, Data>>;
        std::vector<int> devices;
        std::vector<std::unique_ptr<Glm5NextModel>> ranks;
        std::map<const Cache *, std::vector<Cache>> requests;
        std::mutex forwardMutex, barrierMutex;
        std::condition_variable barrierCv;
        unsigned generation = 0;
        int arrived = 0;
        bool failed = false;
        PersistentWorkerGroup workers;

        ~ThreadTpState() { workers.Stop(); }

        void Abort() {
            std::lock_guard<std::mutex> lock(barrierMutex);
            failed = true;
            barrierCv.notify_all();
        }

        void Barrier() {
            std::unique_lock<std::mutex> lock(barrierMutex);
            if (failed) throw std::runtime_error("GLM TP peer failed");
            unsigned previous = generation;
            if (++arrived == (int)devices.size()) {
                arrived = 0;
                ++generation;
                barrierCv.notify_all();
            } else if (!barrierCv.wait_for(lock, std::chrono::seconds(120), [&] {
                    return failed || generation != previous;
                })) {
                failed = true;
                barrierCv.notify_all();
                throw std::runtime_error("GLM TP rank rendezvous timed out");
            }
            if (failed) throw std::runtime_error("GLM TP peer failed");
        }
    };

    void Glm5NextModel::InitThreadTp() {
#ifdef USE_CUDA
        if (threadTpRank >= 0 || threadTpState) return;
        const char *env = std::getenv("FASTLLM_TP");
        if (!env || !*env) return;
        std::string spec(env);
        std::vector<int> devices;
        std::map<int, int> ratios;
        const int available = FastllmCudaGetDeviceCount();
        if (spec == "auto" || spec == "true" || spec == "on" || spec == "1") {
            for (int d = 0; d < available; ++d) devices.push_back(d);
        } else {
            const std::string type = spec.find("multicuda") == 0 ? "multicuda" : "cuda";
            if (spec.find(type) != 0) spec = type + ":" + spec;
            devices = ParseDeviceIds(spec, type, ratios);
        }
        for (int d : devices) AssertInFastLLM(d >= 0 && d < available &&
            (!ratios.count(d) || ratios[d] == 1), "GLM TP requires valid equal CUDA ranks.");
        AssertInFastLLM(std::set<int>(devices.begin(), devices.end()).size() == devices.size(),
            "GLM TP device IDs must be unique.");
        if (devices.size() <= 1) return;
        const int count = devices.size();
        AssertInFastLLM(kdaHeads % count == 0 && num_attention_heads % count == 0,
            "GLM TP degree must divide KDA and MLA heads.");
        AssertInFastLLM(!mtpEnabled, "GLM TP currently requires --mtp 0.");
        AssertInFastLLM(GetCudaSharedExpert(), "GLM TP requires --cuda_shared_expert true.");
        AssertInFastLLM(!GetKVCacheInCPU(), "GLM TP requires CUDA KV caches.");
        for (const auto &item : deviceMap) if (item.second > 0)
            AssertInFastLLM(item.first == "cuda" || item.first.rfind("cuda:", 0) == 0,
                "GLM TP requires CUDA dense layers.");
        for (int layer = 0; layer < block_cnt; ++layer) if (!denseMlpLayers[layer]) {
            const auto device = SelectMoeDeviceForLayer(layer);
            AssertInFastLLM(device == "disk" || device == "cpu" || device == "numa" ||
                device.rfind("numa:", 0) == 0, "GLM hybrid TP requires disk/CPU/NUMA experts.");
        }
        threadTpState = std::make_unique<ThreadTpState>();
        threadTpState->devices = std::move(devices);
#endif
    }

    bool Glm5NextModel::ShouldDelaySpecialWeightCudaMove(const std::string &) const {
        return threadTpState != nullptr;
    }

    void Glm5NextModel::PrepareThreadTp() {
#ifdef USE_CUDA
        auto &tp = *threadTpState;
        if (!tp.ranks.empty()) return;
        auto &devices = tp.devices;
        const int count = devices.size();
        AssertInFastLLM(FastllmInitNccl(devices), "GLM TP NCCL initialization failed.");
        for (int d : devices) {
            FastllmCudaSetDevice(d);
            AssertInFastLLM(FastllmCudaGraphPrepareCaptureDevice(),
                "GLM TP CUDA device initialization failed.");
        }
        for (int rank = 0; rank < count; ++rank) {
            auto model = std::make_unique<Glm5NextModel>();
            model->threadTpRank = rank;
            model->threadTpOwner = &tp;
            model->weight.dicts = weight.dicts;
            model->InitParams();
            model->eos_token_id = eos_token_id;
            model->eos_token_ids = eos_token_ids;
            model->dataType = dataType;
            model->moeAtype = moeAtype;
            model->kvCacheDataType = kvCacheDataType;
            model->kdaHeads = kdaHeads / count;
            model->num_attention_heads = num_attention_heads / count;
            model->num_key_value_heads = num_key_value_heads / count;
            model->deviceMap = {{"cuda:" + std::to_string(devices[rank]), 1}};
            model->moeDeviceMap = moeDeviceMap;
            model->layeredMoeDeviceMap = layeredMoeDeviceMap;
            model->moeDeviceLayers = moeDeviceLayers;
            model->expertWeights = expertWeights;
            model->expertBiases = expertBiases;
            tp.ranks.push_back(std::move(model));
        }
        auto equalScheme = [&](int width, int parts = 1) {
            AssertInFastLLM(width % count == 0, "GLM TP weight dimension is not divisible.");
            DivisionScheme scheme;
            for (int r = 0; r < count; ++r) for (int part = 0; part < parts; ++part)
                scheme[devices[r]].push_back({part * width + width * r / count,
                                             part * width + width * (r + 1) / count});
            return scheme;
        };
        std::vector<std::string> names;
        for (const auto &item : weight.weight) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) {
            auto &source = weight.weight.at(name);
            if (source.dims.empty() || name.find(".mlp.experts.") != std::string::npos) continue;
            if (name == languagePrefix + "embed_tokens.weight") {
                source.ToDevice(DataDevice::CPU);
                auto &local = tp.ranks[0]->weight[name];
                local = source;
                local.isFake = true;
                local.lockInCPU = true;
                continue;
            }
            int axis = -1, parts = 1;
            const auto attn = name.find(".self_attn.");
            if (attn != std::string::npos && name.find(".indexer.") == std::string::npos) {
                const auto suffix = name.substr(attn + std::string(".self_attn.").size());
                if (suffix == "o_proj.weight") axis = 1;
                else if (suffix == "q_proj.weight" || suffix == "k_proj.weight" || suffix == "v_proj.weight" ||
                    suffix == "q_b_proj.weight" || suffix == "kv_b_proj.weight" ||
                    suffix == "q_conv1d.weight" || suffix == "k_conv1d.weight" || suffix == "v_conv1d.weight" ||
                    suffix == "f_b_proj.weight" || suffix == "g_b_proj.weight" || suffix == "b_proj.weight" ||
                    suffix == "A_log" || suffix == "dt_bias") axis = 0;
            } else if (name.find(".mlp.") != std::string::npos) {
                if (name.find("gateup_proj.weight") != std::string::npos) { axis = 0; parts = 2; }
                else if (name.find("down_proj.weight") != std::string::npos) axis = 1;
            }
            const auto originalDims = source.dims;
            if (axis >= 0) {
                if (source.dims.size() != 2) source.Reshape({source.dims[0], (int)source.Count(1)});
                Data bias;
                auto scheme = equalScheme(source.dims[axis] / parts, parts);
                AssertInFastLLM(SplitMultiCudaWeight(source, bias, devices, scheme, axis, true,
                    source.dataType == DataType::NVFP4_BLOCK_16_E4M3), "GLM TP failed to split " + name);
                for (int r = 0; r < count; ++r) {
                    Data *shard = source.multiDeviceDatas.at(devices[r]);
                    auto &local = tp.ranks[r]->weight[name];
                    local = *shard;
                    shard->isFake = true;
                    local.ClearTensorParallelLayout();
                    if (originalDims.size() != 2) {
                        auto dims = originalDims;
                        dims[0] = local.dims[0];
                        local.Reshape(dims);
                    }
                }
            } else {
                source.ToDevice(DataDevice::CPU);
                // Only the sampling rank needs the vocabulary head.
                for (int r = 0; r < (name == "lm_head.weight" ? 1 : count); ++r) {
                    FastllmCudaSetDevice(devices[r]);
                    auto &local = tp.ranks[r]->weight[name];
                    local.CopyFrom(source);
                    local.isModelWeight = true;
                    local.weightType = source.weightType;
                    local.scales = source.scales;
                    local.blockK = source.blockK;
                    local.blockM = source.blockM;
                    local.ToDevice(DataDevice::CUDA, std::vector<int>{devices[r]});
                }
                source.FreeSpace();
            }
        }
        FastllmCudaSetDevice(devices.front());
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        std::printf("[GLM TP] %d ranks ready: %d KDA heads, %d MLA heads per rank; shared host experts.\n",
            count, kdaHeads / count, num_attention_heads / count);
        std::fflush(stdout);
#endif
    }

    void Glm5NextModel::ThreadTpAllReduce(Data &data) {
#ifdef USE_CUDA
        if (threadTpRank < 0) return;
        AssertInFastLLM(data.dataDevice == DataDevice::CUDA && data.cudaData &&
            data.Count(0) <= std::numeric_limits<int>::max(), "GLM TP invalid reduction tensor.");
        FastllmNcclAllReduceNoCustom(data.cudaData, data.cudaData, data.Count(0), data.dataType,
            threadTpOwner->devices[threadTpRank]);
#endif
    }

    void Glm5NextModel::RemoveThreadTpRequest(const std::vector<std::pair<Data, Data>> *key) {
        if (!threadTpState) return;
        auto &tp = *threadTpState;
        std::lock_guard<std::mutex> lock(tp.forwardMutex);
        auto it = tp.requests.find(key);
        if (it == tp.requests.end()) return;
        for (size_t r = 0; r < tp.ranks.size(); ++r)
            tp.ranks[r]->indexerCaches.erase(r == 0 ? key : &it->second[r]);
        tp.requests.erase(it);
    }

    int Glm5NextModel::ForwardThreadTp(const Data &inputIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens, std::vector<float> *logits) {
#ifdef USE_CUDA
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1 && inputIds.dims[1] > 0,
            "GLM TP requires a non-empty request.");
        const bool fresh = std::all_of(pastKeyValues.begin(), pastKeyValues.end(), [](const auto &kv) {
            return kv.first.dims.empty() && kv.second.dims.empty();
        });
        if (fresh) RemoveThreadTpRequest(&pastKeyValues);
        auto &tp = *threadTpState;
        std::lock_guard<std::mutex> lock(tp.forwardMutex);
        AssertInFastLLM(!tp.failed, "GLM TP cannot reuse a failed execution group.");
        try { PrepareThreadTp(); }
        catch (...) { tp.Abort(); throw; }
        pastKeyValues.resize(block_cnt);
        auto &caches = tp.requests[&pastKeyValues];
        if (caches.empty()) {
            caches.resize(tp.devices.size());
            for (size_t r = 1; r < caches.size(); ++r) caches[r].resize(block_cnt);
        }
        std::vector<Data> hidden(tp.devices.size());
        std::vector<std::exception_ptr> errors(tp.devices.size());
        int result = 0;
        tp.workers.Run(tp.devices, [&](int r) {
            FastllmCudaSetDevice(tp.devices[r]);
            static thread_local Executor executor;
            struct RestoreExecutor {
                void *old;
                ~RestoreExecutor() { SetCurrentThreadExecutor(old); }
            } restore{GetExecutor()};
            SetCurrentThreadExecutor(&executor);
            executor.SetFirstDevice("cuda:" + std::to_string(tp.devices[r]));
            try {
                auto &model = *tp.ranks[r];
                auto &cache = r == 0 ? pastKeyValues : caches[r];
                // Host embedding lookup uses the shared CPU pool exactly once.
                if (r == 0) {
                    Data ids(inputIds);
                    model.ForwardEmbedding(ids, hidden[0]);
                    FastllmCudaSyncCurrentThreadStream();
                }
                tp.Barrier();
                if (r != 0) {
                    hidden[r].CopyFrom(hidden[0]);
                    hidden[r].ToDevice(DataDevice::CUDA, std::vector<int>{tp.devices[r]});
                    FastllmCudaSyncCurrentThreadStream();
                }
                tp.Barrier();
                FastllmCudaSetDevice(tp.devices[r]);
                model.ForwardLayers(hidden[r], 0, block_cnt, {&cache});
                if (r == 0) result = model.ForwardOutput(hidden[r], generationConfig,
                    lastTokens, logits, true, nullptr);
                FastllmCudaSyncCurrentThreadStream();
            } catch (...) { tp.Abort(); throw; }
        }, errors);
        for (auto &error : errors) if (error) std::rethrow_exception(error);
        return result;
#else
        throw std::runtime_error("GLM TP requires CUDA");
#endif
    }
}
#endif
