// Shared TP workers, collectives, weight shards and request lifetimes.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_tp.h"
#include "qwen4_exp_graph.h"
#include "qwen4_exp_qsa.h"
#include "qwen4_tp_sampling.h"
#include "models/qwen4_paged_cache.h"
#include "executor.h"
#include "gguf.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-multicuda.cuh"
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
namespace fastllm {
    void Qwen4ExpModel::ThreadTpState::ReleaseCache(const Data *key,
                                                    std::unique_ptr<RequestCache> &cache) {
        if (!cache)
            return;
        for (size_t r = 0; r < ranks.size(); ++r) {
            const Data *localKey =
                r == 0 && !cache->ownsRankZero ? key : &cache->ranks[r].front().first;
            ranks[r]->decodeCudaGraphStates.erase(localKey);
            ranks[r]->requestStates.erase(localKey);
        }
        cache.reset();
    }

    Qwen4ExpModel::ThreadTpState::~ThreadTpState() {
        workers.Stop();
        for (auto &item : caches)
            ReleaseCache(item.first, item.second);
        ReleaseCache(nullptr, idleCache);
    }

    void Qwen4ExpModel::ThreadTpState::Barrier() {
        std::unique_lock<std::mutex> lock(barrierMutex);
        unsigned previous = generation.load(std::memory_order_acquire);
        if (++arrived == (int)devices.size()) {
            arrived = 0;
            generation.fetch_add(1, std::memory_order_release);
            barrierCv.notify_all();
        } else {
            // Short eager collectives usually arrive a few microseconds
            // apart. Avoid a scheduler round trip while retaining the
            // blocking wait for long prefill and host-expert work.
            lock.unlock();
            for (int spin = 0; spin < 2048; ++spin) {
                if (generation.load(std::memory_order_acquire) != previous)
                    return;
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield");
#endif
            }
            lock.lock();
            if (!barrierCv.wait_for(lock, std::chrono::seconds(60), [&] {
                    return generation.load(std::memory_order_acquire) != previous;
                })) {
                std::fprintf(stderr, "Qwen4 TP rank rendezvous timed out.\n");
                std::abort();
            }
        }
    }

#ifdef USE_CUDA
    static DivisionScheme Qwen4ExpertTpDivision(const Data &source, const Data &down,
                                                const std::vector<int> &devices, int axis,
                                                int layer) {
        const int count = devices.size();
        const int width = axis == 0 ? source.dims[0] / 2 : source.dims[1];
        // NVFP4 uses 128-column tiles. Packed GGUF only needs its down
        // projection's block alignment and the 32-value Q8 activation blocks.
        int splitUnit = 128;
        if (source.dataType == DataType::DATA_GGUF_FORMAT) {
            AssertInFastLLM(down.dataType == DataType::DATA_GGUF_FORMAT && down.ggmlType >= 0,
                            "Qwen4 TP GGUF expert requires a packed down projection.");
            splitUnit = std::max(32, int(ggml_blck_size((ggml_type)down.ggmlType)));
        }
        AssertInFastLLM(width > 0 && width % splitUnit == 0,
                        "Qwen4 TP expert width cannot satisfy its packed-column alignment.");
        DivisionScheme scheme;
        int offset = 0;
        for (int r = 0; r < count; ++r) {
            const int blocks = width / splitUnit / count +
                               ((r + layer) % count < (width / splitUnit) % count ? 1 : 0);
            const int end = offset + blocks * splitUnit;
            scheme[devices[r]] = {{offset, end}};
            if (axis == 0)
                scheme[devices[r]].push_back({width + offset, width + end});
            offset = end;
        }
        return scheme;
    }
#endif

#ifdef USE_CUDA
    static void Qwen4CopyTpReplica(const Data &source, Data &local, int device) {
        FastllmCudaSetDevice(device);
        local.CopyFrom(source);
        local.isModelWeight = true;
        local.weightType = source.weightType;
        local.scales = source.scales;
        local.blockK = source.blockK;
        local.blockM = source.blockM;
        local.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    }
#endif

#ifdef USE_CUDA
    static void Qwen4StreamTpReplica(Data &source, const std::vector<int> &devices) {
        AssertInFastLLM(source.cpuData && !source.multiDeviceData &&
                            source.multiDeviceDatas.empty() && source.dims.size() == 2,
                        "Qwen4 TP streaming replica has an invalid source: " + source.name);
        // Own each completed replica even if a later allocation fails.
        source.multiDeviceData = true;
        for (int device : devices) {
            auto local = std::make_unique<Data>();
            Qwen4CopyTpReplica(source, *local, device);
            source.multiDeviceDatas.emplace(device, local.get());
            local.release();
        }
        source.FreeSpace();
    }
#endif

    int Qwen4ExpModel::StreamingThreadTpExpertLayer(const std::string &name) const {
#ifdef USE_CUDA
        if (!threadTpState || threadTpRank >= 0)
            return -1;
        if (Qwen4StartsWith(name, kQwen4MtpExpertPrefix)) {
            return threadTpState->hybrid.hostMoeLayers.back() && !threadTpState->mtpGpuExperts
                ? -1 : block_cnt;
        }
        const std::string prefix = languagePrefix + "layers.";
        if (!Qwen4StartsWith(name, prefix))
            return -1;
        const char *start = name.c_str() + prefix.size();
        char *end = nullptr;
        const long layer = std::strtol(start, &end, 10);
        if (end == start || layer < 0 || layer >= block_cnt ||
            !Qwen4StartsWith(end, ".mlp.experts.") || threadTpState->hybrid.hostMoeLayers[layer])
            return -1;
        return int(layer);
#else
        return -1;
#endif
    }

    int Qwen4ExpModel::StreamingThreadTpReplicaLayer(const std::string &name) const {
#ifdef USE_CUDA
        if (!threadTpState || threadTpRank >= 0 ||
            (!Qwen4StartsWith(name, languagePrefix) && !Qwen4StartsWith(name, "mtp.")))
            return -1;
        // These projections are replicated unchanged on every TP rank. Keep
        // norms and lookup tables on the host for PrepareWeights and sharing.
        if (!Qwen4EndsWith(name, ".input_mix_weight_down.weight") &&
            !Qwen4EndsWith(name, ".input_mix_weight_up.weight") &&
            !Qwen4EndsWith(name, ".block_inject_weight.weight") &&
            !Qwen4EndsWith(name, ".mlp.gate.weight") &&
            !Qwen4EndsWith(name, ".ple.key_proj.weight") &&
            !Qwen4EndsWith(name, ".ple.value_proj.weight") && name != "mtp.fc_embedding.weight" &&
            name != "mtp.fc_hidden.weight")
            return -1;
        const std::string prefix = languagePrefix + "layers.";
        if (!Qwen4StartsWith(name, prefix))
            return block_cnt;
        return std::atoi(name.c_str() + prefix.size());
#else
        return -1;
#endif
    }

    int Qwen4ExpModel::GetWeightLoadPriority(
        const std::string &tensorName,
        const std::vector<std::pair<std::string, DataType>> &) const {
        // TP retains its existing expert/replica groups (priorities <= 0).
        // This last serial group still completes before parallel host experts.
        if (IsStreamingHybridBackboneWeight(tensorName))
            return threadTpState ? 1 : -block_cnt - 2;
        const int layer = StreamingThreadTpExpertLayer(tensorName);
        if (layer >= 0)
            return layer - block_cnt;
        const int replicaLayer = StreamingThreadTpReplicaLayer(tensorName);
        if (replicaLayer < 0)
            return 0;
        return replicaLayer == block_cnt ? -block_cnt - 1 : replicaLayer - block_cnt;
    }

    bool Qwen4ExpModel::ShouldLoadWeightSeriallyBeforeOthers(
        const std::string &tensorName,
        const std::vector<std::pair<std::string, DataType>> &) const {
        return IsStreamingHybridBackboneWeight(tensorName) ||
               StreamingThreadTpExpertLayer(tensorName) >= 0 ||
               StreamingThreadTpReplicaLayer(tensorName) >= 0;
    }

    void Qwen4ExpModel::OnWeightLoadGroupStarted(const std::set<std::string> &names) {
        loadingHybridBackbone = !names.empty() &&
            IsStreamingHybridBackboneWeight(*names.begin());
        if (loadingHybridBackbone || !threadTpState) return;
        int &current = threadTpState->loadingExpertLayer;
        current = -1;
        threadTpState->loadingReplicas.clear();
        for (const auto &name : names) {
            if (StreamingThreadTpReplicaLayer(name) >= 0) {
                threadTpState->loadingReplicas.push_back(name);
            }
            const int layer = StreamingThreadTpExpertLayer(name);
            if (layer < 0)
                continue;
            AssertInFastLLM(current < 0 || current == layer,
                            "Qwen4 TP expert load group contains multiple layers.");
            current = layer;
        }
    }

    void Qwen4ExpModel::OnWeightLoadGroupFinished() {
        if (loadingHybridBackbone) {
            FinishHybridBackboneLoad();
            return;
        }
        if (!threadTpState) return;
#ifdef USE_CUDA
        if (threadTpState->loadingExpertLayer < 0 && threadTpState->loadingReplicas.empty())
            return;
        const int layer = std::exchange(threadTpState->loadingExpertLayer, -1);
        std::vector<std::string> replicas;
        replicas.swap(threadTpState->loadingReplicas);
        auto &devices = threadTpState->devices;
        Qwen4CudaDeviceGuard deviceGuard;
        // The loader has joined all readers and finished gate/up merging.
        // Upload one layer before reading the next; the parent owns the shards
        // until PrepareThreadTp transfers them to rank models. Splitting again
        // there reuses these shards without retaining or rereading CPU payloads.
        for (int expert = 0; layer >= 0 && expert < num_experts; ++expert) {
            const std::string prefix =
                (layer == block_cnt
                     ? kQwen4MtpExpertPrefix
                     : languagePrefix + "layers." + std::to_string(layer) + ".mlp.experts.") +
                std::to_string(expert) + ".";
            Data &gate = weight.weight.at(prefix + "gateup_proj.weight");
            Data &down = weight.weight.at(prefix + "down_proj.weight");
            AssertInFastLLM(gate.dims.size() == 2 && down.dims.size() == 2,
                            "Qwen4 TP streaming expert has an invalid shape.");
            for (int axis : {0, 1}) {
                Data &source = axis == 0 ? gate : down;
                auto scheme = Qwen4ExpertTpDivision(source, down, devices, axis,
                                                    std::min(layer, block_cnt - 1));
                Data bias;
                AssertInFastLLM(SplitMultiCudaWeight(source, bias, devices, scheme, axis, true),
                                "Qwen4 TP failed to stream " + source.name);
            }
        }
        for (const std::string &name : replicas) {
            Data &source = weight.weight.at(name);
            Qwen4StreamTpReplica(source, devices);
        }
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
#endif
    }

    void Qwen4ExpModel::InitThreadTp() {
#ifdef USE_CUDA
        if (threadTpRank >= 0 || threadTpState)
            return;
        const char *env = std::getenv("FASTLLM_TP");
        if (env == nullptr || *env == 0)
            return;
        std::string spec(env);
        std::vector<int> devices;
        std::map<int, int> ratios;
        const int deviceCount = FastllmCudaGetDeviceCount();
        if (spec == "auto" || spec == "true" || spec == "on" || spec == "1") {
            for (int d = 0; d < deviceCount; ++d)
                devices.push_back(d);
        } else {
            const std::string type = spec.find("multicuda") == 0 ? "multicuda" : "cuda";
            if (spec.find(type) != 0)
                spec = type + ":" + spec;
            devices = ParseDeviceIds(spec, type, ratios);
        }
        for (int d : devices) {
            AssertInFastLLM(d >= 0 && d < deviceCount, "Qwen4 TP has an invalid CUDA device.");
            AssertInFastLLM(!ratios.count(d) || ratios[d] == 1,
                            "Qwen4 TP currently requires equal device ratios.");
        }
        AssertInFastLLM(std::set<int>(devices.begin(), devices.end()).size() == devices.size(),
                        "Qwen4 TP device IDs must be unique.");
        if (devices.size() <= 1)
            return;
        AssertInFastLLM(Qwen4CudaOnlyDeviceMap(deviceMap), "Qwen4 TP requires CUDA dense weights.");
        const int count = devices.size();
        AssertInFastLLM(num_k_heads % count == 0 && num_v_heads % count == 0 &&
                            num_attention_heads % count == 0 &&
                            (num_key_value_heads % count == 0 || count % num_key_value_heads == 0),
                        "Qwen4 TP degree must divide query and linear attention heads.");
        AssertInFastLLM(!GetKVCacheInCPU(), "Qwen4 TP requires CUDA KV caches.");
        threadTpState.reset(new ThreadTpState());
        threadTpState->devices = std::move(devices);
        threadTpState->hybrid.Init(*this, threadTpState->devices);
#ifndef USE_ROCM
        threadTpState->mtpGpuExperts = Qwen4MtpDraftsPerStep() > 0;
#endif
#endif
    }

    bool Qwen4ExpModel::ShouldDelaySpecialWeightCudaMove(const std::string &) const {
        // Splitting directly from host storage avoids loading all experts on rank 0.
        return threadTpState != nullptr;
    }

    int Qwen4ExpModel::SplitThreadTpWeight(const std::string &name, Data &source) {
#ifdef USE_CUDA
        auto &tp = *threadTpState;
        auto &devices = tp.devices;
        const int count = devices.size();
        const bool expertWeight = name.find(".mlp.experts.") != std::string::npos;
        const bool mtpExpert = expertWeight && name.rfind("mtp.", 0) == 0;
        const int expertLayer = expertWeight ? (mtpExpert ? block_cnt - 1 :
            std::atoi(name.c_str() + name.find("layers.") + 7)) : -1;
        auto equalScheme = [&](int width, int parts = 1) {
            AssertInFastLLM(width % count == 0, "Qwen4 TP weight dimension is not divisible.");
            DivisionScheme result;
            for (int r = 0; r < count; ++r) {
                for (int part = 0; part < parts; ++part)
                    result[devices[r]].push_back(
                        {part * width + width * r / count, part * width + width * (r + 1) / count});
            }
            return result;
        };
        DivisionScheme scheme;
        int axis = -1;
        const bool linear = name.find(".linear_attn.") != std::string::npos;
        const bool full = name.find(".self_attn.") != std::string::npos &&
                          name.find(".indexer.") == std::string::npos;
        if (name == "lm_head.weight" && source.dims.size() == 2 &&
            (source.dataType == DataType::FLOAT16 ||
             (source.dataType == DataType::DATA_GGUF_FORMAT && source.ggmlType >= 0 &&
              !source.IsRepacked)) &&
            source.dims[0] / 256 >= count && source.dims[0] <= (1 << 24)) {
            // Row sharding preserves each logit's native GEMV arithmetic,
            // including packed GGUF rows, without replicating the head.
            axis = 0;
            tp.vocabSize = source.dims[0];
            tp.topCandidates.resize(count);
            tp.topProbabilities.resize(count);
            tp.shardLogits.resize(count);
            // Select the draft rows while the canonical full target head
            // is still available. Partition by subset row, preserving
            // TopK lane order and balancing even a skewed token list.
            if (Qwen4MtpDraftsPerStep() > 0) PrepareMtpDraftSubsetHead();
            for (int r = 0; r < count; ++r) {
                scheme[devices[r]] = {qwen4_tp::VocabRange(tp.vocabSize, count, r)};
            }
        } else if (linear) {
            const int vd = num_v_heads * head_v_dim;
            if (name.find("in_proj_qkv.weight") != std::string::npos ||
                name.find("conv1d.weight") != std::string::npos) {
                const int kd = num_k_heads * head_k_dim;
                for (int r = 0; r < count; ++r) {
                    scheme[devices[r]] = {{kd * r / count, kd * (r + 1) / count},
                                         {kd + kd * r / count, kd + kd * (r + 1) / count},
                                         {2 * kd + vd * r / count, 2 * kd + vd * (r + 1) / count}};
                }
                axis = 0;
            } else if (name.find("out_proj.weight") != std::string::npos) {
                // Packed GGUF columns retain tiled [value, key] heads.
                // Each rank takes its key-head interval from every value group.
                scheme = ggufWeightsRestored && source.dataType == DataType::DATA_GGUF_FORMAT &&
                                 source.isGGUFData
                             ? equalScheme(num_k_heads * head_v_dim, num_v_heads / num_k_heads)
                             : equalScheme(vd);
                axis = 1;
            } else if (name.find("in_proj_z.weight") != std::string::npos) {
                scheme = equalScheme(vd);
                axis = 0;
            } else if (name.find("in_proj_a.weight") != std::string::npos ||
                       name.find("in_proj_b.weight") != std::string::npos ||
                       name.find("A_log") != std::string::npos ||
                       name.find("dt_bias") != std::string::npos) {
                scheme = equalScheme(num_v_heads);
                axis = 0;
            }
        } else if (full) {
            if (name.find("q_proj.weight") != std::string::npos) {
                scheme = equalScheme(num_attention_heads * head_dim * 2);
                axis = 0;
            } else if (name.find("o_proj.weight") != std::string::npos) {
                scheme = equalScheme(num_attention_heads * head_dim);
                axis = 1;
            } else if (name.find("k_proj.weight") != std::string::npos ||
                       name.find("v_proj.weight") != std::string::npos) {
                for (int r = 0; r < count; ++r) {
                    int first = r * num_attention_heads / count /
                                (num_attention_heads / num_key_value_heads);
                    int heads = std::max(1, num_key_value_heads / count);
                    scheme[devices[r]] = {{first * head_dim, (first + heads) * head_dim}};
                }
                axis = 0;
            }
        } else if (name.find(".mlp.") != std::string::npos) {
            if (name.find("gateup_proj.weight") != std::string::npos) {
                axis = 0;
            } else if (name.find("down_proj.weight") != std::string::npos) {
                axis = 1;
            }
            if (axis >= 0 && expertWeight) {
                const std::string downName =
                    axis == 1 ? name
                              : name.substr(0, name.size() -
                                                   std::string("gateup_proj.weight").size()) +
                                    "down_proj.weight";
                scheme = Qwen4ExpertTpDivision(source, weight.weight.at(downName), devices,
                                               axis, expertLayer);
            } else if (axis >= 0) {
                scheme = axis == 0 ? equalScheme(source.dims[0] / 2, 2)
                                   : equalScheme(source.dims[1]);
            }
        }
        if (axis >= 0) {
            if (source.dims.size() != 2)
                source.Reshape({source.dims[0], (int)source.Count(1)});
            Data bias;
            // Explicit head ranges must not be rotated by layer balancing.
            bool compact = source.dataType == DataType::NVFP4_BLOCK_16_E4M3;
            AssertInFastLLM(
                SplitMultiCudaWeight(source, bias, devices, scheme, axis, true, compact),
                "Qwen4 TP failed to split " + name);
        }
        return axis;
#else
        return -1;
#endif
    }

    void Qwen4ExpModel::StreamHybridThreadTpWeight(const std::string &name, Data &source) {
#ifdef USE_CUDA
        // The MTP subset head needs the full host head after rank creation.
        if (name == "lm_head.weight" && Qwen4MtpDraftsPerStep() > 0) return;
        if (SplitThreadTpWeight(name, source) < 0)
            Qwen4StreamTpReplica(source, threadTpState->devices);
#endif
    }

    void Qwen4ExpModel::PrepareThreadTp() {
#ifdef USE_CUDA
        ThreadTpState &tp = *threadTpState;
        if (!tp.ranks.empty())
            return;
        auto &devices = tp.devices;
        const int count = devices.size();
        PrepareWeights();
        tp.hybrid.workerCpus = HybridWorkerCpuSets(devices);
        tp.votes.resize(count);
        tp.draftCandidates.resize(count);
        tp.topProbabilities.resize(count);
        AssertInFastLLM(FastllmInitNccl(devices), "Qwen4 TP NCCL initialization failed.");
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            AssertInFastLLM(FastllmCudaGraphPrepareCaptureDevice(),
                            "Qwen4 TP could not prepare CUDA graph device storage.");
        }
        for (int rank = 0; rank < count; ++rank) {
            auto model = std::unique_ptr<Qwen4ExpModel>(new Qwen4ExpModel());
            model->threadTpRank = rank;
            model->threadTpOwner = &tp;
            model->weight.dicts = weight.dicts;
            model->InitParams();
            // The loader also reads stop IDs from generation/tokenizer config.
            // Rank-local InitParams alone does not recover those IDs.
            model->eos_token_id = eos_token_id;
            model->eos_token_ids = eos_token_ids;
            model->dataType = dataType;
            model->kvCacheDataType = kvCacheDataType;
            model->kvMemConfig = kvMemConfig;
            model->num_k_heads = num_k_heads / count;
            model->num_v_heads = num_v_heads / count;
            model->num_attention_heads = num_attention_heads / count;
            model->num_key_value_heads = std::max(1, num_key_value_heads / count);
            model->deviceMap = {{"cuda:" + std::to_string(devices[rank]), 1}};
            // Preserve the layer boundaries, but execute every CUDA expert
            // shard on its own rank's device. Host layers retain their policy.
            auto localMoeMap = [&](const std::map<std::string, int> &source) {
                std::map<std::string, int> result;
                for (const auto &item : source) {
                    if (item.second <= 0)
                        continue;
                    const bool cuda = item.first == "cuda" || item.first.rfind("cuda:", 0) == 0;
                    result[cuda ? model->deviceMap.begin()->first : item.first] += item.second;
                }
                return result;
            };
            model->moeDeviceMap = localMoeMap(moeDeviceMap);
            model->layeredMoeDeviceMap = localMoeMap(layeredMoeDeviceMap);
            model->moeDeviceLayers = moeDeviceLayers;
            model->InitExecutionMode();
            model->ngramDevice = ngramDevice;
            model->pleNgramDiskWeight = pleNgramDiskWeight;
            model->pleNgramDiskWeight.isFake = true;
            model->pleDiskReader = pleDiskReader;
            model->qsaKeyNormValues = qsaKeyNormValues;
            model->preparedWeights = true; // Parent already applied norm offsets.
            model->weights.assign(block_cnt, std::vector<Data *>(2 + 2 * num_experts, nullptr));
            model->biass = model->weights;
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (tp.hybrid.hostMoeLayers[layer]) {
                    // The parent outlives its ranks and owns NUMA registration.
                    // EP shares the unsplit tables; its NUMA subset runs once.
                    model->weights[layer] = weights[layer];
                    model->biass[layer] = biass[layer];
                    continue;
                }
                for (int expert = 0; expert < num_experts; ++expert) {
                    const std::string prefix = languagePrefix + "layers." + std::to_string(layer) +
                                               ".mlp.experts." + std::to_string(expert) + ".";
                    model->weights[layer][2 + 2 * expert] =
                        &model->weight[prefix + "gateup_proj.weight"];
                    model->weights[layer][3 + 2 * expert] =
                        &model->weight[prefix + "down_proj.weight"];
                }
            }
            if (Qwen4MtpDraftsPerStep() > 0) {
                model->mtpWeightsStatus.store(mtpWeightsStatus.load());
                model->mtpMoeWeights.assign(2 + 2 * num_experts, nullptr);
                model->mtpMoeBiass = model->mtpMoeWeights;
                if (tp.hybrid.hostMoeLayers.back() && !tp.mtpGpuExperts) {
                    model->mtpMoeWeights = mtpMoeWeights;
                    model->mtpMoeBiass = mtpMoeBiass;
                } else {
                    for (int expert = 0; expert < num_experts; ++expert) {
                        const std::string prefix =
                            kQwen4MtpExpertPrefix + std::to_string(expert) + ".";
                        model->mtpMoeWeights[2 + 2 * expert] =
                            &model->weight[prefix + "gateup_proj.weight"];
                        model->mtpMoeWeights[3 + 2 * expert] =
                            &model->weight[prefix + "down_proj.weight"];
                    }
                }
            }
            tp.ranks.push_back(std::move(model));
        }
        std::map<int, int> compactLayerCounts;
        // WeightMap uses an unordered_map. Sorting is essential for the per-layer
        // consolidation below to bound the number of live source allocations.
        std::vector<std::string> names;
        for (const auto &item : weight.weight)
            names.push_back(item.first);
        std::sort(names.begin(), names.end());
        for (const std::string &name : names) {
            Data &source = weight.weight.at(name);
            if (name.find(languagePrefix) != 0 && name != "lm_head.weight" &&
                name.find("mtp.") != 0)
                continue;
            if (source.dims.empty())
                continue;
            const bool expertWeight = name.find(".mlp.experts.") != std::string::npos;
            const bool mtpExpert = expertWeight && name.rfind("mtp.", 0) == 0;
            const int expertLayer = expertWeight ? (mtpExpert ? block_cnt - 1 :
                std::atoi(name.c_str() + name.find("layers.") + 7)) : -1;
            if (expertWeight && tp.hybrid.hostMoeLayers.at(expertLayer) &&
                !(mtpExpert && tp.mtpGpuExperts)) {
                // Keep the original host layout and NUMA weight registrations.
                continue;
            }
            // Host lookup weights are immutable and can be shared by ranks.
            // Respect the embedding policy before allocating any GPU copies:
            // moving them back during the first forward is too late to avoid
            // exhausting memory while preparing large resident checkpoints.
            const bool hostEmbedding = name == languagePrefix + "embed_tokens.weight" &&
                                       (GetLowMemMode() || !GetCudaEmbeddingRequested());
            if (hostEmbedding)
                source.ToDevice(DataDevice::CPU);
            if (hostEmbedding ||
                name.find(".ple.ple_embedding.ngram_embedding.") != std::string::npos) {
                for (auto &model : tp.ranks) {
                    Data &local = model->weight[name];
                    local = source;
                    local.isFake = true;
                    local.lockInCPU = true;
                }
                continue;
            }
            const std::vector<int> originalDims = source.dims;
            const int axis = SplitThreadTpWeight(name, source);
            if (source.multiDeviceData) {
                // Both split weights and streamed replicas transfer their
                // allocations to ranks without a CPU round trip.
                for (int r = 0; r < count; ++r) {
                    Data *shard = source.multiDeviceDatas.at(devices[r]);
                    Data &local = tp.ranks[r]->weight[name];
                    local = *shard; // Transfer allocation ownership to the rank.
                    shard->isFake = true;
                    local.ClearTensorParallelLayout();
                    if (axis >= 0 && originalDims.size() != 2) {
                        std::vector<int> dims = originalDims;
                        dims[0] = local.dims[0];
                        local.Reshape(dims);
                    }
                }
            } else {
                source.ToDevice(DataDevice::CPU);
                for (int r = 0; r < count; ++r) {
                    Qwen4CopyTpReplica(source, tp.ranks[r]->weight[name], devices[r]);
                }
                // Rank-local copies own their CUDA storage. Host lookup tables
                // shared by reference took the early continue above; no rank
                // needs this replicated weight's parent CPU payload anymore.
                source.FreeSpace();
            }
            if (source.dataType == DataType::NVFP4_BLOCK_16_E4M3 && expertWeight) {
                const int layer = mtpExpert ? block_cnt : expertLayer;
                if (++compactLayerCounts[layer] == num_experts * 2) {
                    // Repack each layer immediately to release small source
                    // allocations before their CUDA page overhead exhausts memory.
                    for (int r = 0; r < count; ++r) {
                        FastllmCudaSetDevice(devices[r]);
                        auto &experts =
                            mtpExpert ? tp.ranks[r]->mtpMoeWeights : tp.ranks[r]->weights[layer];
                        if (experts[2]->dims[0] == 0)
                            continue;
                        AssertInFastLLM(
                            FastllmCudaPrepareNVFP4E4M3Moe(experts.data(), experts.size()),
                            "Qwen4 TP could not prepare compact NVFP4 layer " +
                                std::to_string(layer));
                    }
                }
            }
        }
        for (auto &rank : tp.ranks)
            rank->PrepareMoEExecutionPlans();
        FastllmCudaSetDevice(devices.front());
#if defined(__GLIBC__)
        // Return released TP source allocations before prefill warmup peaks.
        malloc_trim(0);
#endif
        std::printf("[Qwen4 TP] %d ranks ready (linear heads %d/%d, Q/KV heads %d/%d per rank).\n",
                    count, num_k_heads / count, num_v_heads / count, num_attention_heads / count,
                    std::max(1, num_key_value_heads / count));
        std::fflush(stdout);
#endif
    }

    void Qwen4ExpModel::ThreadTpAllReduce(Data &data) {
#ifdef USE_CUDA
        if (threadTpRank < 0)
            return;
        AssertInFastLLM(data.dataDevice == DataDevice::CUDA && data.cudaData != nullptr &&
                            data.Count(0) <= std::numeric_limits<int>::max(),
                        "Qwen4 TP reduction has invalid storage.");
        const int device = threadTpOwner->devices[threadTpRank];
        if (!threadTpOwner->hybrid.hostMoe &&
            FastllmTryTP2P2PAllReduce(data.cudaData, data.cudaData, data.Count(0), data.dataType,
                                      device)) {
            return;
        }
        // The NCCL wrapper owns both host submission boundaries and bypasses
        // them during capture. Repeating those waits here adds another rank
        // wakeup without extending the protected submission interval.
        // Captured execution and unsupported topologies retain NCCL. The
        // eager peer path exchanges current pointers instead of caching pooled
        // temporary addresses across requests.
        FastllmNcclAllReduceNoCustom(data.cudaData, data.cudaData, data.Count(0), data.dataType,
                                     device);
#endif
    }

    bool Qwen4ExpModel::ThreadTpAllTrue(bool value) {
        if (threadTpRank < 0)
            return value;
        auto &tp = *threadTpOwner;
        tp.votes[threadTpRank] = value;
        tp.Barrier();
        const bool result =
            std::all_of(tp.votes.begin(), tp.votes.end(), [](int vote) { return vote != 0; });
        tp.Barrier();
        return result;
    }

    bool Qwen4ExpModel::ThreadTpState::TrySampleLogits(int rank, Data &logits,
                                                       const GenerationConfig &config,
                                                       int cacheLength, int &token,
                                                       std::vector<int> *tokens,
                                                       float *topProbability) {
#ifdef USE_CUDA
        if (vocabSize == 0)
            return false;
        const auto range = qwen4_tp::VocabRange(vocabSize, devices.size(), rank);
        const int width = range.second - range.first;
        const int rows = logits.dims.empty() ? 0 : (int)(logits.Count(0) / width);
        AssertInFastLLM(logits.dataDevice == DataDevice::CUDA &&
                            logits.dataType == DataType::FLOAT32 && rows > 0 &&
                            logits.dims.back() == width && (rows == 1 || tokens != nullptr),
                        "Qwen4 TP vocabulary shard has invalid logits.");
        if (config.IsSimpleGreedy() && !config.output_logits &&
            std::getenv("FASTLLM_QWEN4_DUMP_DIR") == nullptr) {
            if (config.output_token_least > 0) {
                std::vector<int> localEos;
                auto append = [&](int id) {
                    if (id >= range.first && id < range.second)
                        localEos.push_back(id - range.first);
                };
                append(ranks[rank]->eos_token_id);
                for (int id : ranks[rank]->eos_token_ids)
                    append(id);
                for (int id : config.stop_token_ids)
                    append(id);
                if (rows == 1) {
                    if (config.output_token_least > cacheLength - config.input_token_length)
                        FastllmResetLogitsOfEOSAll(1, &logits, localEos);
                } else if (!localEos.empty()) {
                    std::vector<int> lengths(rows), counts(rows, localEos.size()), ids;
                    for (int row = 0; row < rows; ++row) {
                        lengths[row] = config.output_token_least - (cacheLength - rows + row + 1) +
                                       config.input_token_length;
                        ids.insert(ids.end(), localEos.begin(), localEos.end());
                    }
                    FastllmResetLogitsOfEOS(rows, &logits, lengths, counts, ids);
                }
            }
            Data top;
            TopK(logits, top, 1);
            top.ToDevice(DataDevice::CPU);
            const float *values = reinterpret_cast<const float *>(top.cpuData);
            auto &local = topCandidates[rank];
            local.resize(rows);
            for (int row = 0; row < rows; ++row)
                local[row] = {(int)(values[row * 2] + 1e-3f) + range.first, values[row * 2 + 1]};
            if (topProbability != nullptr) {
                AssertInFastLLM(rows == 1, "MTP confidence requires one draft row.");
                topProbabilities[rank] = {values[1], Qwen4TopProbability(logits, values[1])};
            }
            Barrier();
            if (topProbability != nullptr)
                *topProbability = qwen4_tp::MergeTopProbability(topProbabilities);
            if (tokens)
                tokens->resize(rows);
            for (int row = 0; row < rows; ++row) {
                auto best = topCandidates.front()[row];
                for (size_t r = 1; r < topCandidates.size(); ++r) {
                    const auto &candidate = topCandidates[r][row];
                    if (qwen4_tp::Top1Before(candidate.first, candidate.second, best.first,
                                             best.second))
                        best = candidate;
                }
                token = best.first;
                if (tokens)
                    (*tokens)[row] = token;
            }
            // A following draft or verifier may resize the same candidates.
            // All ranks must finish reading before their next sampling call.
            if (Qwen4MtpDraftsPerStep() > 0)
                Barrier();
            return true;
        }

        // Sampling penalties, constraints, returned logits and dumps retain the
        // original full-vocabulary CUDA sampling path on rank zero. Producers
        // remain alive until all peer copies complete; forwardMutex serializes
        // use of this shared gather buffer across requests.
        shardLogits[rank] = &logits;
        FastllmCudaSyncCurrentThreadStream();
        Barrier();
        if (rank == 0) {
            auto dims = logits.dims;
            dims.back() = vocabSize;
            AssertInFastLLM(Qwen4PrepareDecodeGraphWorkspace(gatheredLogits, DataType::FLOAT32,
                                                             dims, devices[0]),
                            "Qwen4 TP could not allocate full sampling logits.");
            for (int r = 0; r < (int)devices.size(); ++r) {
                const auto part = qwen4_tp::VocabRange(vocabSize, devices.size(), r);
                FastllmCudaMemcpyBetweenDevices(
                    devices[0], (float *)gatheredLogits.cudaData + part.first, devices[r],
                    shardLogits[r]->cudaData, (size_t)(part.second - part.first) * sizeof(float));
            }
        }
        // The copy helper completes both peer and staged transfers before return.
        Barrier();
        if (rank != 0) {
            token = -1; // Only rank zero's sampled token is returned by the parent.
            return true;
        }
        logits.FreeSpace();
        Qwen4BorrowCudaStorage(logits, gatheredLogits);
#endif
        return false;
    }

    std::vector<int> Qwen4ExpModel::ForwardThreadTp(
        int batch, const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues, const GenerationConfig &generationConfig,
        const LastTokensManager &lastTokens, std::vector<std::vector<float> *> *logits,
        const Data *precomputedEmbedding) {
#ifdef USE_CUDA
        ThreadTpState &tp = *threadTpState;
        const Data *key = &pastKeyValues[0].first;
        bool hasMultimodalInput = precomputedEmbedding != nullptr;
        {
            // Preserve the request-wide restriction even on a later decode
            // call (or a direct one-token ForwardMultimodal call).
            std::lock_guard<std::mutex> lock(stateMutex);
            const auto found = requestStates.find(key);
            hasMultimodalInput = hasMultimodalInput ||
                                 (found != requestStates.end() && found->second.hasMultimodalInput);
        }
        std::lock_guard<std::mutex> guard(tp.forwardMutex);
        // Prefix lookup also takes this mutex: publish the rank vector only
        // after its weights and workers are fully initialized.
        PrepareThreadTp();
        auto &entry = tp.caches[key];
        bool resetRequest = false;
        if (!entry) {
            // Fresh TP requests can reuse storage without reusing contents.
            // Host experts cannot enter a whole CUDA graph, but their KV and
            // pinned QSA buffers have the same reusable lifetime.
            const bool reuseMtpGraphs = Qwen4MtpDraftsPerStep() > 0 &&
                GetFastllmEnv().cudaGraph &&
                !autoWarmupRunning.load() && maxBatch == 1 &&
                !GetKVCacheInCPU() && !saveHistoryChat && !Qwen4PrefixCacheEnabled() &&
                !hasMultimodalInput && generationConfig.IsSimpleGreedy() &&
                generationConfig.input_token_length >= indexerBudget &&
                std::all_of(tp.ranks.begin(), tp.ranks.end(),
                    [](const auto &rank) { return rank->servingCache != nullptr; });
            const bool reusable = (Qwen4MtpDraftsPerStep() == 0 || reuseMtpGraphs) &&
                batch == 1 && (GetFastllmEnv().cudaGraph || tp.hybrid.hostMoe) &&
                Qwen4TpDecodeFitsCapacityClass(generationConfig, indexerBudget) &&
                std::all_of(pastKeyValues.begin(), pastKeyValues.end(),
                            [](const std::pair<Data, Data> &kv) {
                                return kv.first.dims.empty() && kv.second.dims.empty();
                            });
            if (!reusable || tp.caches.size() > 1) {
                tp.ReleaseCache(nullptr, tp.idleCache);
            } else if (tp.idleCache && !tp.idleCache->startupReserved) {
                // Keep reuse within a capacity class. A much shorter request
                // must not pin a long-context KV allocation through its own
                // prefill, cancellation or admission of a concurrent request.
                for (int layer = 0; layer < block_cnt; ++layer) {
                    if (!IsLinearAttentionLayer(layer) &&
                        Qwen4AxisCapacity(tp.idleCache->ranks[0][layer].first, 1) >
                            2LL * Qwen4RoundUpCacheCapacity(generationConfig.input_token_length,
                                                            128)) {
                        tp.ReleaseCache(nullptr, tp.idleCache);
                        break;
                    }
                }
            }
            if (tp.idleCache) {
                entry = std::move(tp.idleCache);
                resetRequest = true;
            } else {
                entry = std::make_unique<ThreadTpState::RequestCache>();
                entry->ownsRankZero = reusable;
                entry->reuseMtpGraphs = reusable && reuseMtpGraphs;
                entry->ranks.resize(tp.devices.size());
                for (size_t r = reusable ? 0 : 1; r < tp.devices.size(); ++r) {
                    entry->ranks[r].resize(block_cnt);
                }
            }
        }
        // Visual replacement embeddings cannot initialize the text draft KV.
        // Keep subsequent decode calls on the same ordinary TP path as prefill.
        if (precomputedEmbedding)
            entry->mtpDisabled = true;
        auto &caches = entry->ranks;
        std::vector<int> hostInputTokens;
        if (batch == 1 && inputIds.dims == std::vector<int>({1, 1})) {
            // Embedding dispatch moves each rank's ids to CUDA. Preserve the
            // token now so PLE does not read it back after the layer-0 graph.
            Data convertedIds;
            const Data *hostIds = &inputIds;
            if (inputIds.dataDevice != DataDevice::CPU || inputIds.dataType != DataType::FLOAT32) {
                ToDataType(inputIds, convertedIds, DataType::FLOAT32);
                convertedIds.ToDevice(DataDevice::CPU);
                hostIds = &convertedIds;
            }
            hostInputTokens.push_back(
                (int)(reinterpret_cast<const float *>(hostIds->cpuData)[0] + 0.01f));
        }
        // Each worker owns an executor, input tensors and a stable per-thread CUDA
        // stream. Operator dispatch must never mutate the shared global executor.
        std::vector<std::exception_ptr> errors(tp.devices.size());
        std::vector<int> result;
        tp.workers.Run(
            tp.devices,
            [&](int r) {
                tp.hybrid.BindWorker(r);
                FastllmCudaSetDevice(tp.devices[r]);
                static thread_local Executor executor;
                struct Restore {
                    void *old;
                    bool oldTp;
                    bool oldCaptureMode;
                    ~Restore() {
                        SetCurrentThreadExecutor(old);
                        qwen4ThreadTpExecution = oldTp;
                        FastllmCudaGraphSetManagedCaptureOnly(oldCaptureMode);
                    }
                } restore{GetExecutor(), qwen4ThreadTpExecution,
                          FastllmCudaGraphSetManagedCaptureOnly(true)};
                qwen4ThreadTpExecution = true;
                SetCurrentThreadExecutor(&executor);
                executor.SetFirstDevice("cuda:" + std::to_string(tp.devices[r]));
                Data ids(inputIds), mask(attentionMask), positions(positionIds);
                Data embedding;
                if (precomputedEmbedding) {
                    embedding.CopyFrom(*precomputedEmbedding);
                    embedding.ToDevice(DataDevice::CUDA, std::vector<int>{tp.devices[r]});
                }
                Qwen4ExpModel &model = *tp.ranks[r];
                model.autoWarmupRunning.store(autoWarmupRunning.load());
                auto &cache = r == 0 && !entry->ownsRankZero ? pastKeyValues : caches[r];
                if (resetRequest) {
                    // No prefix or recurrent contents survive a request boundary.
                    // Preserve only reusable storage, after the previous worker's
                    // stream has completed, so captured addresses stay valid.
                    auto &state = model.requestStates[&cache.front().first];
                    if (entry->reuseMtpGraphs) {
                        model.ResetDecodeServingState(state, true);
                    } else {
                        RequestState fresh;
                        fresh.indexerTailKeyTensors.swap(state.indexerTailKeyTensors);
                        fresh.indexerTailPositionTensors.swap(state.indexerTailPositionTensors);
                        fresh.indexerBlockKeyTensors.swap(state.indexerBlockKeyTensors);
                        fresh.indexerHostMirrorTransfers.swap(state.indexerHostMirrorTransfers);
                        for (auto *tensors : {&fresh.indexerTailKeyTensors,
                                              &fresh.indexerTailPositionTensors,
                                              &fresh.indexerBlockKeyTensors}) {
                            for (auto &item : *tensors) {
                                if (item.second && !item.second->dims.empty()) {
                                    auto dims = item.second->dims;
                                    dims[0] = 0;
                                    item.second->Resize(dims);
                                }
                            }
                        }
                        for (auto &item : fresh.indexerHostMirrorTransfers) {
                            if (item.second) {
                                item.second->MarkDeviceSynchronized();
                                item.second->Rollback(0);
                            }
                        }
                        state = std::move(fresh);
                    }
                    for (int layer = 0; layer < block_cnt; ++layer) {
                        for (Data *tensor : {&cache[layer].first, &cache[layer].second}) {
                            if (model.IsLinearAttentionLayer(layer)) {
                                tensor->Allocate(0.0f);
                            } else if (!tensor->dims.empty()) {
                                Qwen4ResizeKVCache(*tensor, 0);
                            }
                        }
                    }
                }
                if (entry->reuseMtpGraphs) {
                    model.requestStates[&cache.front().first].reusableGraphRequest = true;
                }
                if (hasMultimodalInput) {
                    model.requestStates[&cache.front().first].hasMultimodalInput = true;
                }
                // Every rank runs the same proposal/acceptance state machine.
                // Global vocabulary sampling keeps rollback and collectives aligned.
                auto tokens =
                    Qwen4MtpDraftsPerStep() > 0 && !entry->mtpDisabled
                        ? model.ForwardBatch(batch, ids, mask, positions, cache, generationConfig,
                                             lastTokens, r == 0 ? logits : nullptr)
                        : model.ForwardTarget(batch, ids, mask, positions, cache, generationConfig,
                                              lastTokens, r == 0 ? logits : nullptr, nullptr,
                                              nullptr, nullptr, nullptr, true, true, false,
                                              hostInputTokens.empty() ? nullptr : &hostInputTokens,
                                              false, precomputedEmbedding ? &embedding : nullptr);
                if (r == 0)
                    result = std::move(tokens);
                FastllmCudaSyncCurrentThreadStream();
            },
            errors);
        for (auto &error : errors)
            if (error) {
                entry->failed = true;
                std::rethrow_exception(error);
            }
        if (entry->ownsRankZero) {
            // The scheduler observes rank-0 cache metadata. Ownership remains
            // in the TP slot until graphs are destroyed or the slot is reused.
            for (int layer = 0; layer < block_cnt; ++layer) {
                const bool linear = IsLinearAttentionLayer(layer);
                Qwen4BorrowCudaTensor(caches[0][layer].first, pastKeyValues[layer].first, linear,
                                      false);
                Qwen4BorrowCudaTensor(caches[0][layer].second, pastKeyValues[layer].second, linear,
                                      false);
            }
        }
        entry->startupReserved = true;
        for (size_t rank = 0; rank < tp.ranks.size(); ++rank) {
            const Data *localKey =
                rank == 0 && !entry->ownsRankZero ? key : &caches[rank].front().first;
            const auto state = tp.ranks[rank]->requestStates.find(localKey);
            if (state == tp.ranks[rank]->requestStates.end() || !state->second.servingCache) {
                entry->startupReserved = false;
            }
        }
        return result;
#else
        return {};
#endif
    }

    void Qwen4ExpModel::RemoveThreadTpRequest(const Data *key, bool completed) {
        if (!threadTpState)
            return;
        ThreadTpState &tp = *threadTpState;
        std::lock_guard<std::mutex> guard(tp.forwardMutex);
        auto found = tp.caches.find(key);
        if (found == tp.caches.end())
            return;
        auto &cache = found->second;
        // Admission of another request takes priority over an idle graph.
        // Retaining a completed long-context slot during a peer's prefill
        // would otherwise pin its large KV and graph allocations unnecessarily.
        bool reusable = completed && tp.caches.size() == 1 &&
            cache->ownsRankZero && !cache->failed;
        for (size_t r = 0; reusable && !cache->reuseMtpGraphs && !tp.hybrid.hostMoe && r < tp.ranks.size(); ++r) {
            const auto &graphs = tp.ranks[r]->decodeCudaGraphStates;
            auto graph = graphs.find(&cache->ranks[r].front().first);
            reusable = graph != graphs.end() && graph->second && graph->second->wholeGraphMode &&
                       std::any_of(graph->second->segments.begin(), graph->second->segments.end(),
                    [](const auto &segment) { return segment.second.captured; });
        }
        if (cache->reuseMtpGraphs) {
#ifdef USE_CUDA
            Qwen4CudaDeviceGuard deviceGuard;
            for (size_t r = 0; r < tp.ranks.size(); ++r) {
                FastllmCudaSetDevice(tp.devices[r]);
                ForceDeviceSync();
                auto &model = *tp.ranks[r];
                auto &local = cache->ranks[r];
                const Data *localKey = &local.front().first;
                if (reusable) {
                    model.ResetDecodeServingState(model.requestStates[localKey], true);
                    // Return logical KV pages before the scheduler admits the
                    // next request, retaining all captured allocations.
                    for (int layer = 0; layer < block_cnt; ++layer) {
                        if (model.IsLinearAttentionLayer(layer)) continue;
                        for (Data *tensor : {&local[layer].first, &local[layer].second})
                            if (!tensor->dims.empty()) Qwen4ResizeKVCache(*tensor, 0);
                    }
                }
            }
#endif
        }
        if (reusable) {
            tp.ReleaseCache(nullptr, tp.idleCache);
            tp.idleCache = std::move(cache);
        } else {
            tp.ReleaseCache(key, cache);
        }
        tp.caches.erase(found);
    }
} // namespace fastllm
