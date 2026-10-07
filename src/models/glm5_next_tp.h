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
        std::vector<std::vector<int>> workerCpus;
        std::vector<unsigned char> workerBoundToSpareCores;
        std::vector<std::unique_ptr<Glm5NextModel>> ranks;
        std::map<const Cache *, std::vector<Cache>> requests;
        std::map<const Cache *, std::vector<std::vector<KdaReplayCapture>>> replays;
        std::mutex forwardMutex, barrierMutex;
        std::condition_variable barrierCv;
        unsigned generation = 0;
        int arrived = 0;
        bool failed = false;
        PersistentWorkerGroup workers;
        std::vector<void *> moeReady;
        std::vector<bool> shardedMoeLayers;
        int loadingLayer = -1;

        ~ThreadTpState() {
            workers.Stop();
#ifdef USE_CUDA
            const int previous = FastllmCudaGetDevice();
            for (size_t r = 0; r < moeReady.size(); ++r) if (moeReady[r]) {
                FastllmCudaSetDevice(devices[r]);
                FastllmCudaEventDestroy(moeReady[r]);
            }
            FastllmCudaSetDevice(previous);
#endif
        }

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
        AssertInFastLLM(GetCudaSharedExpert(), "GLM TP requires --cuda_shared_expert true.");
        AssertInFastLLM(!GetKVCacheInCPU(), "GLM TP requires CUDA KV caches.");
        for (const auto &item : deviceMap) if (item.second > 0)
            AssertInFastLLM(item.first == "cuda" || item.first.rfind("cuda:", 0) == 0,
                "GLM TP requires CUDA dense layers.");
        std::vector<bool> shardedMoeLayers(block_cnt, false);
        for (int layer = 0; layer < block_cnt; ++layer) if (!denseMlpLayers[layer]) {
            const auto device = SelectMoeDeviceForLayer(layer);
            if (device == "cuda" || device.rfind("cuda:", 0) == 0) {
                std::map<int, int> moeRatios;
                const auto owners = ParseDeviceIds(device, "cuda", moeRatios);
                const bool singleOwner = owners.size() == 1 &&
                    std::find(devices.begin(), devices.end(), owners[0]) != devices.end();
                AssertInFastLLM(singleOwner || owners == devices,
                    "GLM TP resident experts require one CUDA owner or the full TP group.");
                for (int d : owners) AssertInFastLLM(!moeRatios.count(d) || moeRatios[d] == 1,
                    "GLM TP resident expert shards require equal device ratios.");
                shardedMoeLayers[layer] = !singleOwner;
            } else {
                AssertInFastLLM(device == "disk" || device == "cpu" || device == "numa" ||
                    device.rfind("numa:", 0) == 0, "GLM TP requires CUDA/disk/CPU/NUMA experts.");
            }
        }
        threadTpState = std::make_unique<ThreadTpState>();
        threadTpState->devices = std::move(devices);
        threadTpState->shardedMoeLayers = std::move(shardedMoeLayers);
#endif
    }

    int Glm5NextModel::ThreadTpExpertLayer(const std::string &name) const {
        const std::string prefix = languagePrefix + "layers.";
        if (name.rfind(prefix, 0) != 0) return -1;
        const char *start = name.c_str() + prefix.size();
        char *end = nullptr;
        const long layer = std::strtol(start, &end, 10);
        return end != start && std::string(end).rfind(".mlp.experts.", 0) == 0 &&
            layer >= 0 && layer < block_cnt ? int(layer) : -1;
    }

    bool Glm5NextModel::ShouldDelaySpecialWeightCudaMove(const std::string &name) const {
        if (!threadTpState) return false;
        const int layer = ThreadTpExpertLayer(name);
        // Single-owner experts can upload immediately. Sharded experts wait
        // for their load group to finish merging gate/up, then split once.
        return layer < 0 || threadTpState->shardedMoeLayers[layer];
    }

    void Glm5NextModel::StageThreadTpWeight(const std::string &name) {
#ifdef USE_CUDA
        auto &source = weight.weight.at(name);
        const int expertLayer = ThreadTpExpertLayer(name);
        if (source.dims.empty() || source.multiDeviceData ||
            name == languagePrefix + "embed_tokens.weight" ||
            name.rfind(languagePrefix + "layers." + std::to_string(block_cnt) + ".", 0) == 0 ||
            (expertLayer >= 0 && !threadTpState->shardedMoeLayers[expertLayer])) return;
        auto &devices = threadTpState->devices;
        const int count = devices.size();
        struct RestoreDevice {
            int previous = FastllmCudaGetDevice();
            ~RestoreDevice() { FastllmCudaSetDevice(previous); }
        } restore;
        auto equalScheme = [&](int width, int parts = 1) {
            AssertInFastLLM(width % count == 0, "GLM TP weight dimension is not divisible.");
            DivisionScheme scheme;
            for (int r = 0; r < count; ++r) for (int part = 0; part < parts; ++part)
                scheme[devices[r]].push_back({part * width + width * r / count,
                                             part * width + width * (r + 1) / count});
            return scheme;
        };
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
            source.Reshape(originalDims);
        } else {
            source.ToDevice(DataDevice::CPU);
            source.multiDeviceData = true;
            // Only the sampling rank needs the vocabulary head. Own each
            // completed replica even if a later allocation fails.
            for (int r = 0; r < (name == "lm_head.weight" ? 1 : count); ++r) {
                FastllmCudaSetDevice(devices[r]);
                auto local = std::make_unique<Data>();
                local->CopyFrom(source);
                local->isModelWeight = true;
                local->weightType = source.weightType;
                local->scales = source.scales;
                local->blockK = source.blockK;
                local->blockM = source.blockM;
                local->ToDevice(DataDevice::CUDA, std::vector<int>{devices[r]});
                source.multiDeviceDatas.emplace(devices[r], local.get());
                local.release();
            }
            source.FreeSpace();
        }
#endif
    }

    int Glm5NextModel::StreamingThreadTpLayer(const std::string &name) const {
        const auto arch = weight.dicts.find("gguf_architecture");
        if (!threadTpState || threadTpRank >= 0 || arch == weight.dicts.end() ||
            arch->second != "glm5next") return -1;
        if (name == "lm_head.weight" || name == languagePrefix + "norm.weight") return block_cnt;
        const std::string prefix = languagePrefix + "layers.";
        if (name.rfind(prefix, 0) != 0) return -1;
        const int expertLayer = ThreadTpExpertLayer(name);
        if (expertLayer >= 0 && !threadTpState->shardedMoeLayers[expertLayer]) return -1;
        const char *start = name.c_str() + prefix.size();
        char *end = nullptr;
        const long layer = std::strtol(start, &end, 10);
        return end != start && *end == '.' && layer >= 0 && layer < block_cnt ? int(layer) : -1;
    }

    int Glm5NextModel::GetWeightLoadPriority(const std::string &name,
            const std::vector<std::pair<std::string, DataType>> &) const {
        const int layer = StreamingThreadTpLayer(name);
        return layer < 0 ? 0 : layer - block_cnt - 1;
    }

    bool Glm5NextModel::ShouldLoadWeightSeriallyBeforeOthers(const std::string &name,
            const std::vector<std::pair<std::string, DataType>> &) const {
        return StreamingThreadTpLayer(name) >= 0;
    }

    void Glm5NextModel::OnWeightLoadGroupStarted(const std::set<std::string> &names) {
        if (!threadTpState) return;
        int &layer = threadTpState->loadingLayer;
        layer = -1;
        for (const auto &name : names) {
            const int current = StreamingThreadTpLayer(name);
            if (current < 0) continue;
            AssertInFastLLM(layer < 0 || layer == current, "GLM TP load group contains multiple layers.");
            layer = current;
        }
    }

    void Glm5NextModel::OnWeightLoadGroupFinished() {
#ifdef USE_CUDA
        if (!threadTpState || threadTpState->loadingLayer < 0) return;
        const int layer = threadTpState->loadingLayer;
        threadTpState->loadingLayer = -1;
        // Join all loader workers before converting paired K/V tensors and
        // uploading merged gate/up weights. CPU/NUMA experts stay resident.
        if (layer < block_cnt) {
            glm5_next_detail::RestoreGgufWeights(weight, layer + 1,
                num_attention_heads, qkNopeHeadDim, valueHeadDim, kvLoraRank, layer);
        }
        std::vector<std::string> names;
        for (const auto &item : weight.weight)
            if (StreamingThreadTpLayer(item.first) == layer) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) StageThreadTpWeight(name);
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
#endif
    }

    void Glm5NextModel::PrepareThreadTp() {
#ifdef USE_CUDA
        auto &tp = *threadTpState;
        if (!tp.ranks.empty()) return;
        auto &devices = tp.devices;
        const int count = devices.size();
        tp.workerCpus.resize(count);
        tp.workerBoundToSpareCores.resize(count, false);
#ifdef USE_NUMAS
        for (int layer = 0; layer < block_cnt; ++layer) if (!denseMlpLayers[layer]) {
            const auto device = SelectMoeDeviceForLayer(layer);
            if (device == "numa" || device.rfind("numa:", 0) == 0) {
                tp.workerCpus = GetNumasCudaWorkerCpuSets(devices);
                break;
            }
        }
#endif
        AssertInFastLLM(FastllmInitNccl(devices), "GLM TP NCCL initialization failed.");
        for (int d : devices) {
            FastllmCudaSetDevice(d);
            tp.moeReady.push_back(d == devices[0] ? nullptr : FastllmCudaEventCreate());
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
        std::vector<std::string> names;
        for (const auto &item : weight.weight) names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const auto &name : names) {
            auto &source = weight.weight.at(name);
            const int expertLayer = ThreadTpExpertLayer(name);
            if (source.dims.empty() ||
                name.rfind(languagePrefix + "layers." + std::to_string(block_cnt) + ".", 0) == 0 ||
                (expertLayer >= 0 && !tp.shardedMoeLayers[expertLayer])) continue;
            if (name == languagePrefix + "embed_tokens.weight") {
                source.ToDevice(DataDevice::CPU);
                auto &local = tp.ranks[0]->weight[name];
                local = source;
                local.isFake = true;
                local.lockInCPU = true;
                continue;
            }
            StageThreadTpWeight(name);
            // Streamed weights already own their final allocations. Transfer
            // them to ranks without another host copy or TP split.
            for (int r = 0; r < (name == "lm_head.weight" ? 1 : count); ++r) {
                Data *shard = source.multiDeviceDatas.at(devices[r]);
                auto &local = tp.ranks[r]->weight[name];
                local = *shard;
                shard->isFake = true;
                local.ClearTensorParallelLayout();
                if (source.dims.size() != local.dims.size()) {
                    auto dims = source.dims;
                    dims[0] = local.dims[0];
                    local.Reshape(dims);
                }
            }
        }
        for (int layer = 0; layer < block_cnt; ++layer) if (tp.shardedMoeLayers[layer]) {
            for (int expert = 0; expert < num_experts; ++expert) {
                const std::string prefix = languagePrefix + "layers." + std::to_string(layer) +
                    ".mlp.experts." + std::to_string(expert) + ".";
                for (int r = 0; r < count; ++r) {
                    auto &model = *tp.ranks[r];
                    model.expertWeights[layer][2 * (expert + 1)] = &model.weight.weight.at(prefix + "gateup_proj.weight");
                    model.expertWeights[layer][2 * (expert + 1) + 1] = &model.weight.weight.at(prefix + "down_proj.weight");
                }
            }
        }
        FastllmCudaSetDevice(devices.front());
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
        std::printf("[GLM TP] %d ranks ready: %d KDA heads, %d MLA heads per rank; shared expert storage.\n",
            count, kdaHeads / count, num_attention_heads / count);
        for (int device : devices) {
            uint64_t bytes = 0;
            int layers = 0;
            for (const auto &experts : expertWeights) {
                if (experts.size() < 4 || !experts[2] || experts[2]->dataDevice != DataDevice::CUDA ||
                    experts[2]->dataDeviceIds != std::vector<int>{device}) continue;
                ++layers;
                for (size_t i = 2; i < experts.size(); ++i) if (experts[i]) {
                    AssertInFastLLM(experts[i]->cpuData == nullptr && experts[i]->numasData.empty(),
                        "GLM TP resident expert still owns host weight storage.");
                    bytes += experts[i]->GetBytes();
                }
            }
            if (layers) std::printf("[GLM TP] cuda:%d owns %d resident MoE layers (%.3f GiB); no host weight copy.\n",
                device, layers, double(bytes) / (1ULL << 30));
        }
        for (int r = 0; r < count; ++r) {
            uint64_t bytes = 0;
            int layers = 0;
            for (int layer = 0; layer < block_cnt; ++layer) if (tp.shardedMoeLayers[layer]) {
                ++layers;
                const auto &experts = tp.ranks[r]->expertWeights[layer];
                for (size_t i = 2; i < experts.size(); ++i) if (experts[i]) {
                    AssertInFastLLM(!expertWeights[layer][i]->cpuData &&
                        expertWeights[layer][i]->numasData.empty() &&
                        !experts[i]->cpuData && experts[i]->numasData.empty() &&
                        experts[i]->dataDeviceIds == std::vector<int>{devices[r]},
                        "GLM TP expert shard retained host storage or has the wrong CUDA owner.");
                    bytes += experts[i]->GetBytes();
                }
            }
            if (layers) std::printf("[GLM TP] cuda:%d owns shards of %d MoE layers (%.3f GiB); no host weight copy.\n",
                devices[r], layers, double(bytes) / (1ULL << 30));
        }
        std::fflush(stdout);
#endif
    }

    Data &Glm5NextModel::OutputHead() {
        return threadTpState ? threadTpState->ranks.at(0)->weight["lm_head.weight"] : weight["lm_head.weight"];
    }

    void Glm5NextModel::ThreadTpAllReduce(Data &data) {
#ifdef USE_CUDA
        if (threadTpRank < 0) return;
        AssertInFastLLM(data.dataDevice == DataDevice::CUDA && data.cudaData &&
            data.Count(0) <= std::numeric_limits<int>::max(), "GLM TP invalid reduction tensor.");
        // A sleeping peer otherwise adds a wakeup after each CPU MoE phase.
        // Decode and exact multi-row verification use the same short wait
        // budget, only on submission cores isolated from NUMA workers.
        const uint64_t rows = data.Count(0) / embed_dim;
        const bool decodeOrVerify = rows == 1 ||
            (rows > 1 && rows < (uint64_t)FastllmCudaGetLinearExactBatchThreshold());
        const int hostSpinUs = decodeOrVerify &&
            threadTpOwner->workerBoundToSpareCores[threadTpRank] ? 1000 : 0;
        FastllmNcclAllReduceNoCustomWithSpin(data.cudaData, data.cudaData, data.Count(0), data.dataType,
            threadTpOwner->devices[threadTpRank], hostSpinUs);
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
        tp.replays.erase(key);
        tp.requests.erase(it);
    }

    int Glm5NextModel::ForwardThreadTp(const Data &inputIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens, std::vector<float> *logits,
            bool sampleOutput, Data *targetHiddenStates,
            std::vector<KdaReplayCapture> *kdaReplay) {
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
        auto &replays = tp.replays[&pastKeyValues];
        if (kdaReplay) {
            replays.resize(tp.devices.size());
            for (auto &replay : replays) { replay.clear(); replay.resize(block_cnt); }
        }
        const int exactThreshold = FastllmCudaGetLinearExactBatchThreshold();
        std::vector<Data> hidden(tp.devices.size());
        std::vector<std::exception_ptr> errors(tp.devices.size());
        int result = 0;
        tp.workers.Run(tp.devices, [&](int r) {
#ifdef USE_NUMAS
            // The persistent submission threads must not compete with the
            // pinned NUMA expert workers or their SMT siblings.
            static thread_local bool placementAttempted = false;
            if (!placementAttempted) {
                tp.workerBoundToSpareCores[r] = BindNumasWorkerCpuSet(tp.workerCpus[r]);
                placementAttempted = true;
            }
#endif
            FastllmCudaSetDevice(tp.devices[r]);
            struct RestoreExactThreshold {
                int old = FastllmCudaGetLinearExactBatchThreshold();
                ~RestoreExactThreshold() { FastllmCudaSetLinearExactBatchThreshold(old); }
            } restoreExact;
            FastllmCudaSetLinearExactBatchThreshold(exactThreshold);
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
                model.ForwardLayers(hidden[r], 0, block_cnt, {&cache}, kdaReplay ? &replays[r] : nullptr);
                if (r == 0) result = model.ForwardOutput(hidden[r], generationConfig,
                    lastTokens, logits, sampleOutput, targetHiddenStates);
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
