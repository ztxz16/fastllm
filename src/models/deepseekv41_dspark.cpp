//
// DeepSeek-V4.1 的 DSpark 投机解码。
//
// 结构（对应官方 inference/model.py 的 DSparkBlock / DSparkAttention /
// DSparkMarkovHead / DSparkConfidenceHead 与 Transformer.forward_spec）：
//
//   * mtp.0 / mtp.1 / mtp.2 是三个与主干同构的草稿层（compress_ratio 0，纯滑窗注意力，
//     128 专家 top-3）。mtp.0 额外有 main_proj / main_norm，mtp.2 额外有 norm /
//     markov_head（embed + head）/ confidence_head；embedding 与 lm_head 与主干共享。
//   * 草稿层的滑窗 KV 不来自草稿 token，而来自目标模型 dspark_target_layer_ids
//     （[37, 38, 39]）各层 attention 输入（对 hc 份取均值）拼接后经
//     main_proj / main_norm 得到的 main_x：每个已提交位置在草稿侧只有一行 KV。
//   * 一次 proposal：把 block_size 个位置的输入置为 noise token（第 0 个位置放锚点
//     token），一次前向产出 block_size 组 logits；再用 markov head 对每个位置做
//     bigram 修正后逐位置生成固定或随机候选，得到 block_size 个 token 和置信度。
//
// 校验：候选 token 与锚点拼成一个 block_size + 1 长度的片段喂给目标模型
// （ForwardSegments 天然支持一次多 token）。贪心请求在第一个 argmax 不匹配处截断。
// 随机采样请求保留实际 proposal q，以目标 p / q 接受，拒绝时采样归一化的
// max(p-q, 0)，全部接受时从额外目标行采样。固定草稿是 q 为 one-hot 的特例。
// 分布等价以目标条件概率相同为前提；不同 batch/kernel 的浮点结果仍可能有差异。
//
// 回滚：校验前向按完整 block 更新缓存，接受长度确定后需要把多算的部分退回：
//   * 滑窗环形缓冲：写入延后到接受长度确定之后（对本次前向的注意力没有影响，
//     因为片段内的位置一律从 chunkKV 读取），因此不需要保存旧值；
//   * 压缩 KV / indexer key：按接受后的长度重新截断行数，并用保存下来的
//     compressor 原始输入流重建 rawTail；
//   * Engram 历史与各层 totalLen：直接截断到接受后的长度。
//

#include "deepseekv41.h"

#include "baseblock.h"
#include "executor.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
#include "utils.h"
#include "utils/hybrid_moe.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#endif
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "fastllm-cuda-graph-workspace.h"
#ifdef FASTLLM_ENABLE_DSV41_SM100_MEGA
#include "devices/cuda/fastllm-dsv41-mega.h"
#endif
#include "devices/cuda/fastllm-cuda-dspark.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/cuda/fastllm-draft-head.cuh"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-v41-full-decode.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include <algorithm>
#include "devices/disk/diskdevice.h"
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>

namespace fastllm {
    // Per-request inputs and outputs have stable addresses. Temporary tensors
    // created while capturing are owned by the graph allocator until teardown.
    struct DeepSeekV41DsparkWorkspace {
        int block = 0, width = 0;
        bool disabled = false, parallelExperts = true, deviceInputs = false, parallelAttention = false;
        bool sampledMarkov = false;
        float temperature = 1.0f;
        int topK = 0;
        float topP = 1.0f;
        std::vector<int> devices;
        std::vector<void*> cachePointers;
        Data input, preMix, indices, position, headHidden, normalized, logits;
        Data markovTokens, markovValues, markovIndices, markovEmbeds;
        Data samplingWorkspace;
        struct GreedyWorkspace { Data values, indices; };
        std::vector<GreedyWorkspace> greedyWorkspaces;
        std::shared_ptr<void> markovWorkspace;
        // The graph optimizer may run shared and routed branches concurrently.
        // Their scratch must remain disjoint throughout every replay.
        struct SharedExpertWorkspace {
            Data output, gateup, activated, floatOutput;
        };
        Data confidenceLogits;
        std::vector<SharedExpertWorkspace> sharedExperts;
        struct RoutedExpertWorkspace {
            Data input, index, score, output;
            void *graph = nullptr, *exec = nullptr;
        };
        std::vector<RoutedExpertWorkspace> routedExperts;
        std::vector<void *> moeHandles;
        bool hybridExperts = false;
        std::unique_ptr<DiskMoeV41Graph> diskExperts;
        // Resolve immutable packed weights during warmup and keep them alive
        // until ClearGraphs has destroyed every graph referencing them.
        std::map<int, Data> packedQkvWeights;
        std::map<int, std::shared_ptr<void>> attentionTp;
#ifdef USE_CUDA
        std::vector<void*> graphs, execs, starts, ends, reserved;
        FastllmDsparkHcBranch hcBranch;
        void *replayReady = nullptr;
        std::vector<void *> replayDone;
        int *hostProposal = nullptr;
        void *proposalReady = nullptr, *mirrorStream = nullptr;
        int pendingCount = 0, packetPosition = -1;
        Data verifyTokenMap, verifyHash, verifyPacket, verifyDecision;
        int nextMetadataAt = -1, nextMetadataAnchor = -1, fullStreak = 0;
        int reusableDraftAt = -1, reusableDraftAnchor = -1;
        float *hostTop1 = nullptr;
        void *top1Ready = nullptr, *kvReady = nullptr;
        void *verifyReady = nullptr, *packetStream = nullptr;
        bool rootWinnerRecorded = false;

        void FinishProposal(std::vector<int> *output = nullptr) {
            if (!pendingCount) return;
            FastllmCudaEventSynchronize(proposalReady);
            if (output) output->assign(hostProposal, hostProposal + pendingCount);
            pendingCount = 0;
        }
        bool QueueProposal(int count) {
            if (!proposalReady) proposalReady = FastllmCudaEventCreate();
            if (!hostProposal) hostProposal = static_cast<int *>(FastllmCudaHostMalloc(block * sizeof(int)));
            if (!hostProposal || !proposalReady || pendingCount || count != block) return false;
            const int *source = static_cast<const int *>(markovTokens.cudaData) + 1;
            bool copied = false;
            if (rootWinnerRecorded && !replayDone.empty()) {
                if (!mirrorStream) mirrorStream = FastllmCudaStreamCreate();
                if (mirrorStream) {
                    FastllmCudaStreamWaitEvent(mirrorStream, replayDone.front());
                    copied = FastllmCudaCopyFromDeviceToPinnedHostAsync(hostProposal, source, count * sizeof(int), mirrorStream);
                    if (copied) FastllmCudaEventRecord(proposalReady, mirrorStream);
                }
            }
            if (!copied) {
                if (!FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(hostProposal, source, count * sizeof(int))) return false;
                FastllmCudaEventRecordCurrentThread(proposalReady);
            }
            pendingCount = count;
            packetPosition = -1;
            return true;
        }
        void ClearGraphs() {
            const int original = FastllmCudaGetDevice();
            FinishProposal();
            for (size_t i = 0; i < devices.size(); ++i) {
                FastllmCudaSetDevice(devices[i]);
                if (i < execs.size() && execs[i]) {
                    FastllmCudaSyncDevice(devices[i]);
                    FastllmCudaGraphExecDestroy(execs[i]);
                }
                if (i < graphs.size() && graphs[i]) FastllmCudaGraphDestroy(graphs[i]);
                if (i < starts.size() && starts[i]) FastllmCudaEventDestroy(starts[i]);
                if (i < ends.size() && ends[i]) FastllmCudaEventDestroy(ends[i]);
                if (i < replayDone.size() && replayDone[i]) FastllmCudaEventDestroy(replayDone[i]);
            }
            replayDone.clear();
            rootWinnerRecorded = false;
            if (!devices.empty()) FastllmCudaSetDevice(devices.front());
            if (hcBranch.stream) { FastllmCudaStreamDestroy(hcBranch.stream); hcBranch.stream = nullptr; }
            if (hcBranch.begin) { FastllmCudaEventDestroy(hcBranch.begin); hcBranch.begin = nullptr; }
            if (hcBranch.done) { FastllmCudaEventDestroy(hcBranch.done); hcBranch.done = nullptr; }
            if (proposalReady && !devices.empty()) FastllmCudaSetDevice(devices.front());
            if (kvReady) { FastllmCudaEventDestroy(kvReady); kvReady = nullptr; }
            if (top1Ready) { FastllmCudaEventDestroy(top1Ready); top1Ready = nullptr; }
            if (hostTop1) { FastllmCudaHostFree(hostTop1); hostTop1 = nullptr; }
            if (verifyReady) {
                FastllmCudaEventSynchronize(verifyReady);
                FastllmCudaEventDestroy(verifyReady);
                verifyReady = nullptr;
            }
            if (mirrorStream) { FastllmCudaStreamDestroy(mirrorStream); mirrorStream = nullptr; }
            if (packetStream) { FastllmCudaStreamDestroy(packetStream); packetStream = nullptr; }
            packetPosition = -1;
            if (proposalReady) { FastllmCudaEventDestroy(proposalReady); proposalReady = nullptr; }
            if (hostProposal) { FastllmCudaHostFree(hostProposal); hostProposal = nullptr; }
            if (replayReady) {
                FastllmCudaSetDevice(devices.front());
                FastllmCudaEventDestroy(replayReady);
                replayReady = nullptr;
            }
            // Parent executables must die before their native child buffers.
            for (size_t i = 0; i < routedExperts.size(); ++i) {
                FastllmCudaSetDevice(devices.front());
                if (routedExperts[i].exec) FastllmCudaGraphExecDestroy(routedExperts[i].exec);
                if (i < moeHandles.size()) FastllmCudaReleaseNativeMoeFullGraph(moeHandles[i]);
            }
            diskExperts.reset();
            routedExperts.clear(); moeHandles.clear();
            FastllmCudaGraphMemoryPoolRelease(reserved);
            graphs.clear(); execs.clear(); starts.clear(); ends.clear(); reserved.clear();
            FastllmCudaSetDevice(original);
        }
        ~DeepSeekV41DsparkWorkspace() { ClearGraphs(); }
#endif
    };

    DeepSeekV41DsparkState::~DeepSeekV41DsparkState() {
        // Graph executables retain pointers into layers and proposalProbs.
        // Retire them before member destruction returns those buffers to pools.
        workspace.reset();
        contextWorkspaces.clear();
    }

    void DeepSeekV41DsparkState::ClearProposal() {
#ifdef USE_CUDA
        if (workspace) workspace->FinishProposal();
#endif
        drafts.clear();
        confidence.clear();
        proposalTokens.clear();
        sampledProposal = false;
        anchor = -1;
        anchorPos = -1;
    }

    void DeepSeekV41DsparkState::Reset(int startPos) {
#ifdef USE_CUDA
        if (workspace) {
            workspace->fullStreak = 0;
            workspace->reusableDraftAt = workspace->nextMetadataAt = workspace->packetPosition = -1;
        }
#endif
        committed = startPos;
        filled = 0;
        disabled = false;
        contextRequests.clear();
        nextVerifyPosition = -1;
        ClearProposal();
        pending.clear();
        rounds = proposed = accepted = 0;
        // Retain storage, not validity: only the freshly filled prefix can be
        // attended, and proposalProbs is unreadable until a new draft publishes q.
    }

    void DeepSeekV41Model::RetireDsparkState(std::shared_ptr<DeepSeekV41DsparkState> state) {
        if (v41DsparkEnabled && state && state.use_count() == 1 &&
            v41DsparkFreeStates.size() < size_t(std::max(1, maxBatch)))
            v41DsparkFreeStates.push_back(std::move(state));
    }

    namespace {
#ifdef USE_CUDA
        thread_local bool dsparkCapturing = false;

        bool DsRunDraftGraph(DeepSeekV41DsparkWorkspace &state, const std::function<void()> &body, bool streamJoin = false) {
            const int root = state.devices.front();
            state.rootWinnerRecorded = false;
            auto launch = [&]() {
                if (state.devices.size() > 1) {
                    const auto replay = [&] {
                        std::vector<int> ok(state.devices.size(), 0);
                        FastllmCudaSetDevice(root);
                        if (!state.replayReady) state.replayReady = FastllmCudaEventCreate();
                        // A rejected or early proposal can reuse the same token buffer.
                        // Only the next writer of that buffer must wait for its CPU mirror.
                        if (state.pendingCount && state.mirrorStream)
                            FastllmCudaCurrentThreadStreamWaitEvent(state.proposalReady);
                        // The caller has updated draft context and input metadata.
                        // Join that producer before replay, then drain only the
                        // graph streams, leaving target cache admissions running.
                        FastllmCudaEventRecordCurrentThread(state.replayReady);
                        const bool events = streamJoin && !state.diskExperts && !state.hybridExperts &&
                            state.replayDone.size() == state.devices.size() &&
                            std::all_of(state.replayDone.begin(), state.replayDone.end(), [](void *e) { return e != nullptr; });
                        auto runRank = [&](int rank, int) {
                            FastllmCudaCurrentThreadStreamWaitEvent(state.replayReady);
                            ok[rank] = rank == 0 && !state.moeHandles.empty()
                                ? FastllmCudaLaunchNativeMoeFullGraph(state.execs[rank],
                                    state.moeHandles.data(), int(state.moeHandles.size()))
                                : FastllmCudaGraphLaunch(state.execs[rank]);
                            if (events) FastllmCudaEventRecordCurrentThread(state.replayDone[rank]);
                        };
                        const bool queued = MultiCudaRunDeviceCallbacksEnqueueOnly(state.devices, runRank);
                        if (queued && events) {
                            state.rootWinnerRecorded = true;
                            // The caller can submit the next graph while the workers run.
                            // Peer-local consumers and the root proposal copy both retain their dependencies.
                            for (size_t r = 1; r < state.devices.size(); ++r) {
                                FastllmCudaSetDevice(state.devices[r]);
                                FastllmCudaCurrentThreadStreamWaitEvent(state.replayDone[r]);
                            }
                            FastllmCudaSetDevice(root);
                            for (void *event : state.replayDone) FastllmCudaCurrentThreadStreamWaitEvent(event);
                            return std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; });
                        }
                        const bool dispatched = queued
                            ? MultiCudaSynchronizeDeviceCallbacks(state.devices)
                            : MultiCudaRunDeviceCallbacks(state.devices, runRank);
                        return dispatched && std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; });
                    };
                    AssertInFastLLM(state.diskExperts ? state.diskExperts->Launch(replay) : replay(),
                                    "DeepSeekV41 DSpark graph launch failed.");
                } else {
                    FastllmCudaSetDevice(root);
                    const auto replay = [&] {
                        return state.moeHandles.empty() ? FastllmCudaGraphLaunch(state.execs.front()) :
                            FastllmCudaLaunchNativeMoeFullGraph(state.execs.front(), state.moeHandles.data(),
                                                              int(state.moeHandles.size()));
                    };
                    AssertInFastLLM(state.diskExperts ? state.diskExperts->Launch(replay) : replay(),
                                    "DeepSeekV41 DSpark graph launch failed.");
                }
                FastllmCudaSetDevice(root);
            };
            if (!state.execs.empty()) { launch(); return true; }
            if (state.disabled) return false;
            FastllmCudaGraphPreparation preparation;
            body(); // Resolve weights, collective buffers and all kernel workspaces.
            if (state.disabled) return true; // Warmup already produced valid eager outputs.
            const int count = state.devices.size();
            std::vector<size_t> allocations;
            {
                // The backbone remains root-owned with TP attention/head too.
                // Record its temporary allocations after persistent weights are
                // ready: eager reuse does not provision every captured address.
                FastllmCudaAllocationRecorder record(&allocations);
                body();
            }
            if (state.disabled) return true;
            {
                FastllmCudaSetDevice(root);
                FastllmCudaSyncDevice(root);
                if (!FastllmCudaReserveGraphWorkspace(allocations)) {
                    state.disabled = true;
                    fprintf(stderr, "[Fastllm] DSpark graph workspace unavailable; using eager.\n");
                    return true; // Warmup has already produced the requested outputs.
                }
            }
            state.graphs.assign(count, nullptr); state.execs.assign(count, nullptr);
            state.starts.assign(count, nullptr); state.ends.assign(count, nullptr);
            if (streamJoin) state.replayDone.assign(count, nullptr);
            bool ok = true;
            for (int i = 0; i < count; ++i) {
                FastllmCudaSetDevice(state.devices[i]);
                FastllmCudaSyncDevice(state.devices[i]);
                ok = FastllmCudaGraphPrepareCaptureDevice() && ok;
                state.starts[i] = FastllmCudaEventCreate();
                state.ends[i] = FastllmCudaEventCreate();
                if (streamJoin) state.replayDone[i] = FastllmCudaEventCreate();
                ok = state.starts[i] && state.ends[i] && ok;
            }
            const bool pool = ok && FastllmCudaGraphMemoryPoolBegin();
            ok = pool;
            int begun = 0;
            bool joined = false;
            std::exception_ptr error;
            if (pool) {
                FastllmCudaClearThreadError();
                for (int device : state.devices) {
                    FastllmCudaSetDevice(device);
                    if (!FastllmCudaGraphBeginCapture()) { ok = false; break; }
                    ++begun;
                }
                if (ok && count > 1) {
                    for (int i = 0; i < count; ++i) {
                        FastllmCudaSetDevice(state.devices[i]);
                        FastllmCudaEventRecordCurrentThread(state.starts[i]);
                    }
                    joined = MultiCudaGraphWorkersWaitEvents(state.devices, state.starts);
                    ok = joined;
                }
                if (ok) {
                    FastllmCudaSetDevice(root);
                    dsparkCapturing = true;
                    try { body(); } catch (...) { error = std::current_exception(); }
                    dsparkCapturing = false;
                    ok = !error && !state.disabled && !FastllmCudaGetThreadError() && !FastllmCudaGetGraphError();
                }
                if (joined) {
                    ok = MultiCudaGraphWorkersRecordEvents(state.devices, state.ends) && ok;
                    for (int i = 0; i < count; ++i) {
                        FastllmCudaSetDevice(state.devices[i]);
                        FastllmCudaCurrentThreadStreamWaitEvent(state.ends[i]);
                    }
                }
                for (int i = 0; i < begun; ++i) {
                    FastllmCudaSetDevice(state.devices[i]);
                    ok = FastllmCudaGraphEndCapture(&state.graphs[i]) && ok;
                }
                ok = FastllmCudaGraphMemoryPoolEnd(state.reserved) && ok;
                if (ok) for (int i = 0; i < count && ok; ++i) {
                    FastllmCudaSetDevice(state.devices[i]);
                    // Only the root core graph contains parallel expert regions.
                    if (i == 0 && state.parallelExperts && !state.sharedExperts.empty()) {
                        const int parallel = FastllmCudaGraphOptimizeParallelRegions(state.graphs[i]);
                        if (parallel == FASTLLM_CUDA_GRAPH_RECAPTURE_WITHOUT_PARALLEL_MARKERS) {
                            // Allocation nodes cannot be rewritten. Capture a
                            // serial graph instead, retaining the graph speedup.
                            state.ClearGraphs();
                            state.parallelExperts = false;
                            FastllmCudaSetDevice(root);
                            return DsRunDraftGraph(state, body, streamJoin);
                        }
                        ok = parallel >= 0;
                    }
                    if (ok) ok = FastllmCudaGraphInstantiate(state.graphs[i], &state.execs[i]);
                }
            }
            FastllmCudaSetDevice(root);
            if (!ok) {
                const std::string reason = FastllmCudaGraphLastError();
                state.ClearGraphs(); state.disabled = true;
                FastllmCudaClearLastError(); FastllmCudaClearThreadError(); FastllmCudaClearGraphError();
                if (error) std::rethrow_exception(error);
                fprintf(stderr, "[Fastllm] DeepSeekV41 DSpark graph capture failed (%s); using eager.\n", reason.c_str());
                return false;
            }
            launch();
            return true;
        }
#endif

        // ---- 与 deepseekv41.cpp 中同名的算子薄封装。那些封装位于匿名命名空间，
        // ---- 跨文件不可复用，这里按同样的参数重新封装一份（只是 Executor::Run 的转发）。

        Executor &DsExecutor() {
            return *((Executor*)GetExecutor());
        }

        bool DsEnvFlag(const char *name) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return false;
            }
            std::string v(value);
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            return v == "1" || v == "on" || v == "true" || v == "yes";
        }

#ifdef USE_CUDA
        bool DsDraftGraphEnabled() {
            const auto &env = GetFastllmEnv();
            return env.cudaGraph && !env.cudaSync && !env.printProfile &&
                std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
                !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && !DsEnvFlag("FASTLLM_DSV41_LEGACY_ROTARY");
        }
#endif

        float DsEnvFloat(const char *name, float fallback) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return fallback;
            }
            char *end = nullptr;
            float ret = std::strtof(value, &end);
            if (end == value || *end != '\0' || !std::isfinite(ret)) {
                return fallback;
            }
            return ret;
        }

        int DsEnvInt(const char *name, int fallback) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return fallback;
            }
            return atoi(value);
        }

        void DsRMSNormBF16(const Data &input, Data &weight, float eps, Data &output) {
            RMSNorm(input, weight, eps, output);
            ToDataType(output, DataType::BFLOAT16);
        }

        void DsHcMix(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                     int hcMult, int sinkhornIters, float eps, float normEps,
                     Data &pre, Data &post, Data &comb) {
            DsExecutor().Run("DeepSeekV41HcMix", {
                {"input", (Data*)&input}, {"hcFn", &hcFn}, {"hcScale", &hcScale}, {"hcBase", &hcBase},
                {"pre", &pre}, {"post", &post}, {"comb", &comb}
            }, {{"eps", eps}, {"normEps", normEps}}, {{"hcMult", hcMult}, {"sinkhornIters", sinkhornIters}});
        }

        void DsHcApplyPre(const Data &input, const Data &pre, Data &output) {
            DsExecutor().Run("DeepSeekV41HcApplyPre", {
                {"input", (Data*)&input}, {"pre", (Data*)&pre}, {"output", &output}
            }, {}, {});
        }

        void DsHcPost(Data &input, Data &residual, Data &post, Data &comb, Data &output) {
            DsExecutor().Run("DeepSeekV41HcPost", {{"input", &input}, {"residual", &residual},
                {"post", &post}, {"comb", &comb}, {"output", &output}}, {}, {});
        }

#ifdef USE_CUDA
        bool DsFusedHcEnabled() {
            return !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                !DsEnvFlag("FASTLLM_DSV41_LEGACY_HCMIX") &&
                !DsEnvFlag("FASTLLM_DSV41_LEGACY_HCMIX_DECODE") &&
                !DsEnvFlag("FASTLLM_DSV41_DISABLE_HCPRENORM");
        }
#endif

        void DsHcApplyPreNorm(const Data &input, const Data &pre, Data &normWeight,
                             float eps, Data &scratch, Data &output, bool retainHidden = false) {
#ifdef USE_CUDA
            if (DsFusedHcEnabled() &&
                FastllmCudaDeepSeekV41HcPreNorm(input, pre, normWeight, eps, output)) {
                // Confidence consumes the pre-normalization BF16 hidden state.
                // The fused norm only writes output; materialize it on demand.
                if (retainHidden) DsHcApplyPre(input, pre, scratch);
                return;
            }
#endif
            DsHcApplyPre(input, pre, scratch);
            DsRMSNormBF16(scratch, normWeight, eps, output);
        }

        // Normalized output is always produced; return whether its quantized copy is ready.
        bool DsHcMixPreNorm(const Data &input, const Data &previousPre,
                           Data &fn, Data &scale, Data &base, Data &normWeight,
                           int hcMult, int iters, float mixEps, float normEps,
                           Data &pre, Data &post, Data &comb, Data &scratch, Data &output,
                           Data *quantizedOutput = nullptr) {
#ifdef USE_CUDA
            if (DsFusedHcEnabled() && hcMult == 4 &&
                FastllmCudaDeepSeekV41HcMixPreNorm(input, previousPre, fn, scale, base, normWeight,
                    iters, mixEps, normEps, pre, post, comb, output, quantizedOutput))
                return quantizedOutput != nullptr;
#endif
            DsHcMix(input, fn, scale, base, hcMult, iters, mixEps, normEps, pre, post, comb);
            DsHcApplyPreNorm(input, previousPre, normWeight, normEps, scratch, output);
            return false;
        }

#ifdef USE_CUDA
        bool DsHcPostMixPreNorm(const Data &input, const Data &residual,
                const Data &previousPre, const Data &previousPost, const Data &previousComb,
                Data &fn, Data &scale, Data &base, Data &normWeight,
                int hcMult, int iters, float mixEps, float normEps,
                Data &hidden, Data &pre, Data &post, Data &comb, Data &output) {
            if (!DsFusedHcEnabled() || hcMult != 4 || residual.dataDevice != DataDevice::CUDA ||
                residual.multiDeviceData || residual.dims.size() != 4 || residual.dims[0] <= 0 ||
                residual.dims[1] <= 0 || residual.dataType != DataType::BFLOAT16 ||
                residual.dataDeviceIds != std::vector<int>{FastllmCudaGetDevice()}) return false;
            const int64_t tokens = (int64_t)residual.dims[0] * residual.dims[1];
            const int d = residual.dims[3];
            if (tokens < 2 || tokens > 8 || d < 4096 || d > 8192 || d % 1024 ||
                (d > 6144 && tokens <= 6)) return false;
            for (Data *w : {&fn, &scale, &base, &normWeight}) {
                if (w->multiDeviceData || w->dataType != DataType::FLOAT32) return false;
                w->ToDevice(DataDevice::CUDA, residual.dataDeviceIds, true);
            }
            return FastllmCudaDeepSeekV41HcPostMixPreNorm(input, residual, previousPre,
                previousPost, previousComb, fn, scale, base, normWeight,
                iters, mixEps, normEps, hidden, pre, post, comb, output);
        }
#endif

        struct DsRopeParams {
            int ropeDim;
            float base;
            int originalSeqLen;
            float factor;
            int betaFast;
            int betaSlow;
        };

#ifdef USE_CUDA
        struct DsAttentionTpWeights {
            Data wq, woA, woB;
            std::vector<int> devices;
        };

        struct DsAttentionTpWorkspace {
            struct Rank {
                Data packet, query, table, attention, woA, partial, floatSum, sink;
            };
            std::shared_ptr<DsAttentionTpWeights> weights;
            std::vector<std::unique_ptr<Rank>> ranks;
            std::vector<std::vector<int>> inputShapes;
            std::vector<int> devices;
            std::vector<size_t> offsets;
            Data windowShadow, windowValid;
            bool residentWindow = false;
            int bytes = 0;
            bool ready = false;
        };

        bool DsParallelAttention(const std::vector<int> &devices, Data &queryInput, Data &kv,
                                 const Data &window, Data &indices, Data &position, Data &wq, Data &woA,
                                 Data &woB, Data &sink, int heads, int groups, int oRank,
                                 const DsRopeParams &rope, int startPos, float scale,
                                 std::shared_ptr<void> &weightCache, std::shared_ptr<void> &cache,
                                 Data &output) {
            const int count = (int)devices.size();
            if (count < 2 || groups <= 0 || groups % count || heads <= 0 || heads % groups || oRank <= 0 ||
                heads / count > 128 || rope.ropeDim < 2 || rope.ropeDim > 128 || rope.ropeDim % 2 ||
                queryInput.dataType != DataType::BFLOAT16 || queryInput.dims.size() != 3 ||
                queryInput.dims[0] != 1 || queryInput.dims[1] < 2 || queryInput.dims[1] > 8 ||
                kv.dataType != DataType::BFLOAT16 || window.dataType != DataType::BFLOAT16 ||
                indices.dataType != DataType::INT32 || position.dataType != DataType::INT32 ||
                wq.dataType != DataType::FP8_E4M3 || woB.dataType != DataType::FP8_E4M3 ||
                woA.dataType != DataType::BFLOAT16 || sink.dataType != DataType::FLOAT32 ||
                wq.multiDeviceData || woA.multiDeviceData || woB.multiDeviceData || sink.multiDeviceData ||
                wq.dims.size() != 2 || woA.dims.size() != 2 || woB.dims.size() != 2 ||
                kv.dims != std::vector<int>({1, queryInput.dims[1], 512}) || window.dims.size() != 3 ||
                window.dims[0] != 1 || window.dims[1] < 1 || window.dims[1] > 1024 || window.dims[2] != 512 ||
                indices.dims.size() != 3 || indices.dims[0] != 1 || indices.dims[1] != queryInput.dims[1] ||
                wq.dims != std::vector<int>({heads * 512, queryInput.dims[2]}) ||
                woA.dims != std::vector<int>({groups * oRank, heads / groups * 512}) ||
                woB.dims[1] != groups * oRank || sink.Count(0) != (uint64_t)heads || position.Count(0) < 1 ||
                DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") || DsEnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
                DsEnvFlag("FASTLLM_DISABLE_NCCL"))
                return false;
            // Warmup copies raw checkpoint storage; unsupported/lazy layouts
            // must fall back before any weight or activation is changed.
            for (const Data *x : {&wq, &woA, &woB, &sink}) {
                if (x->dataDevice == DataDevice::CPU ? !x->cpuData
                                                     : x->dataDevice != DataDevice::CUDA || !x->cudaData)
                    return false;
            }
            const Data *sources[] = {&queryInput, &kv, &window, &indices, &position};
            for (const Data *x : sources) {
                if (x->dataDevice != DataDevice::CUDA || !x->cudaData || x->multiDeviceData ||
                    x->dataDeviceIds != std::vector<int>{devices.front()} ||
                    x->strides.size() != x->dims.size())
                    return false;
                uint64_t stride = 1;
                for (int axis = (int)x->dims.size() - 1; axis >= 0; --axis) {
                    if (x->strides[axis] != stride) return false;
                    stride *= x->dims[axis];
                }
            }
            const int rows = queryInput.dims[1], inner = queryInput.dims[2];
            const int localHeads = heads / count, localGroups = groups / count;
            if (!FastllmCudaCanUseMarlinBFloat16FP8(wq, rows, inner, heads * 512) ||
                !FastllmCudaCanUseMarlinBFloat16FP8(woB, rows, groups * oRank, woB.dims[0]) ||
                (localGroups * oRank) % 64)
                return false;
            auto alloc = [](Data &x, DataType type, const std::vector<int> &dims, int device) {
                x.dataType = type;
                x.UpdateUnitSize();
                x.Resize(dims);
                x.ToDevice(DataDevice::CUDA, {device}, false);
                x.Allocate();
            };
            if (!cache) {
                if (FastllmCudaGraphIsCapturingFast()) return false;
                auto w = std::make_shared<DsAttentionTpWorkspace>();
                cache = w;
                w->devices = devices;
                auto weights = std::static_pointer_cast<DsAttentionTpWeights>(weightCache);
                if (weights && weights->devices != devices) weights.reset();
                if (!weights) {
                    weights = std::make_shared<DsAttentionTpWeights>();
                    weights->devices = devices;
                    std::vector<int> mutableDevices = devices;
                    auto split = [&](Data &source, Data &target, int axis) {
                        target.CopyFrom(source);
                        // Split owns its source buffer; stage the immutable clone before peer copies.
                        target.ToDevice(DataDevice::CPU);
                        target.blockM = source.blockM;
                        target.blockK = source.blockK;
                        target.scales = source.scales;
                        target.isModelWeight = true;
                        DivisionScheme ranges;
                        const int width = source.dims[axis] / count;
                        for (int i = 0; i < count; ++i)
                            ranges[devices[i]].push_back({i * width, (i + 1) * width});
                        Data bias;
                        return SplitMultiCudaWeight(target, bias, mutableDevices, ranges, axis, true);
                    };
                    if (!FastllmInitNccl(devices) || !split(wq, weights->wq, 0) ||
                        !split(woA, weights->woA, 0) || !split(woB, weights->woB, 1))
                        return false;
                    weightCache = weights;
                }
                w->weights = std::move(weights);
                w->inputShapes = {queryInput.dims, kv.dims, window.dims, indices.dims, position.dims};
                w->residentWindow = count == 8 && rows == 5 && window.dims[1] == 128 &&
                    getCudaInfos()->cudaArch == 1030 && FastllmCudaPeerAccessInit(devices);
                if (w->residentWindow) {
                    alloc(w->windowShadow, DataType::BFLOAT16, window.dims, devices.front());
                    alloc(w->windowValid, DataType::INT32, {window.dims[1]}, devices.front());
                    FastllmCudaMemset0(w->windowValid.cudaData, w->windowValid.GetBytes());
                }
                size_t bytes = 0;
                for (size_t size : {queryInput.GetBytes(), kv.GetBytes(), w->residentWindow ? w->windowValid.GetBytes() : window.GetBytes(),
                                    indices.GetBytes(), sizeof(int32_t)}) {
                    w->offsets.push_back(bytes);
                    bytes = (bytes + size + 255) & ~size_t(255);
                }
                if (bytes > size_t(std::numeric_limits<int>::max())) return false;
                w->bytes = (int)bytes;
                sink.ToDevice(DataDevice::CUDA, {devices.front()}, true);
                bool supported = true;
                for (int i = 0; i < count; ++i) {
                    const int device = devices[i];
                    FastllmCudaSetDevice(device);
                    auto r = std::make_unique<DsAttentionTpWorkspace::Rank>();
                    alloc(r->packet, DataType::INT8, {w->bytes}, device);
                    alloc(r->query, DataType::BFLOAT16, {1, rows, localHeads, 512}, device);
                    alloc(r->partial, DataType::BFLOAT16, {1, rows, woB.dims[0]}, device);
                    alloc(r->floatSum, DataType::FLOAT32, r->partial.dims, device);
                    alloc(r->sink, DataType::FLOAT32, {localHeads}, device);
                    FastllmCudaMemcpyBetweenDevices(device, r->sink.cudaData, devices.front(),
                                                    (char *)sink.cudaData + i * localHeads * sizeof(float),
                                                    localHeads * sizeof(float));
                    supported =
                        supported &&
                        FastllmCudaCanUseMarlinBFloat16FP8(*w->weights->wq.multiDeviceDatas.at(device), rows,
                                                           inner, localHeads * 512) &&
                        FastllmCudaCanUseMarlinBFloat16FP8(*w->weights->woB.multiDeviceDatas.at(device), rows,
                                                           localGroups * oRank, woB.dims[0]);
                    w->ranks.push_back(std::move(r));
                }
                FastllmCudaSetDevice(devices.front());
                w->ready = supported;
            }
            auto &w = *std::static_pointer_cast<DsAttentionTpWorkspace>(cache);
            if (!w.ready || w.devices != devices ||
                w.inputShapes != std::vector<std::vector<int>>{queryInput.dims, kv.dims, window.dims,
                                                               indices.dims, position.dims})
                return false;
            FastllmCudaSetDevice(devices.front());
            if (w.residentWindow) {
                FastllmCudaDsparkPackAttention(queryInput, kv, window, indices, position, w.windowShadow,
                    w.windowValid, w.offsets.data(), w.ranks[0]->packet);
            } else for (int i = 0; i < 5; ++i)
                FastllmCudaCopyFromDeviceToDevice((char *)w.ranks[0]->packet.cudaData + w.offsets[i],
                                                  sources[i]->cudaData,
                                                  i == 4 ? sizeof(int32_t) : sources[i]->GetBytes());
            struct DispatchScope {
                bool async = MultiCudaSetPersistentAsyncDispatch(true);
                bool ordered = MultiCudaSetWorkerOrderedCapture(false);
                ~DispatchScope() {
                    MultiCudaSetWorkerOrderedCapture(ordered);
                    MultiCudaSetPersistentAsyncDispatch(async);
                }
            } scope;
            const bool dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                auto &r = *w.ranks[rank];
                FastllmNcclBroadcast(r.packet.cudaData, w.bytes, DataType::INT8, devices.front(), device);
                // Non-owning stack views only describe the packed inputs. Wq_b
                // consumes its raw pointer directly; no temporary heap owners.
                auto data = [&](int i) { return (char *)r.packet.cudaData + w.offsets[i]; };
                Data localKv(DataType::BFLOAT16, kv.dims, DataDevice::CUDA, data(1));
                Data localWindow(DataType::BFLOAT16, window.dims, DataDevice::CUDA, data(2));
                Data localIndices(DataType::INT32, indices.dims, DataDevice::CUDA, data(3));
                for (Data *view : {&localKv, &localWindow, &localIndices})
                    view->dataDeviceIds = {device};
                const int *pos = (const int *)((char *)r.packet.cudaData + w.offsets[4]);
                AssertInFastLLM(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(
                                    data(0), *w.weights->wq.multiDeviceDatas.at(device), r.query.cudaData,
                                    rows, inner, localHeads * 512),
                                "Draft TP Wq_b failed.");
                AssertInFastLLM(FastllmCudaDsparkDraftRotaryTable(r.query, localKv,
                                    w.residentWindow ? window : localWindow, rope.ropeDim, rope.base, startPos,
                                    rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow, pos, r.table,
                                    w.residentWindow ? reinterpret_cast<const int *>(data(2)) : nullptr),
                                "Draft TP rotary failed.");
                // The rank-local reduction scratch is idle until after O-B on this stream.
                AssertInFastLLM(FastllmCudaDsparkDraftSparseAttention(r.query, localKv, r.table,
                                    localIndices, r.sink, scale, r.floatSum, r.attention),
                                "Draft TP attention failed.");
                AssertInFastLLM(
                    FastllmCudaDeepSeekV41RotaryQuantPosition(r.attention, rope.ropeDim, rope.base, startPos,
                                                              1, true, rope.originalSeqLen, rope.factor,
                                                              rope.betaFast, rope.betaSlow, 0, 512, 32, pos),
                    "Draft TP inverse rotary failed.");
                AssertInFastLLM(FastllmCudaDeepSeekV4WoA(r.attention,
                                                         *w.weights->woA.multiDeviceDatas.at(device),
                                                         localGroups, oRank, r.woA),
                                "Draft TP WoA failed.");
                AssertInFastLLM(FastllmCudaTryMarlinBFloat16MatMulFP8E4M3(
                                    r.woA.cudaData, *w.weights->woB.multiDeviceDatas.at(device),
                                    r.partial.cudaData, rows, localGroups * oRank, woB.dims[0]),
                                "Draft TP Wo_b failed.");
                const int elements = rows * woB.dims[0];
                // Match row-parallel BF16 output: local GEMM rounds once,
                // reduction accumulates in FP32, then rounds its final sum.
                const bool reduced = w.residentWindow && FastllmCudaCustomReduceOwnerBF16(
                    r.partial.cudaData, elements, devices.front(), device);
                if (!reduced && !FastllmCudaCustomAllReduce(r.partial.cudaData, r.partial.cudaData, elements,
                                                DataType::BFLOAT16, device)) {
                    AssertInFastLLM(FastllmBF16ToFloat(r.partial.cudaData, r.floatSum.cudaData, elements),
                                    "Draft TP cast failed.");
                    FastllmNcclAllReduceNoCustom(r.floatSum.cudaData, r.floatSum.cudaData, elements,
                                                 DataType::FLOAT32, device);
                    AssertInFastLLM(FastllmFloatToBF16(r.floatSum.cudaData, r.partial.cudaData, elements),
                                    "Draft TP cast failed.");
                }
            });
            AssertInFastLLM(dispatched && !FastllmCudaGetThreadError(), "Draft TP dispatch failed.");
            FastllmCudaSetDevice(devices.front());
            output.CopyFrom(w.ranks[0]->partial);
            return true;
        }
#endif

        void DsRotaryQuant(Data &x, const DsRopeParams &rope, int startPos, int posStep,
                           bool inverse, int quantMode, int quantBlock, int quantDim = -1,
                           const int *devicePosition = nullptr) {
            static const bool disableFakeQuant = DsEnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            if (disableFakeQuant) {
                quantMode = 0;
            }
#ifdef USE_CUDA
            if (devicePosition) {
                AssertInFastLLM(FastllmCudaDeepSeekV41RotaryQuantPosition(x, rope.ropeDim, rope.base,
                    startPos, posStep, inverse, rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow,
                    quantMode, quantDim > 0 ? quantDim : x.dims.back(), quantBlock, devicePosition),
                    "DeepSeekV41 DSpark dynamic rotary rejected input.");
                return;
            }
#endif
            IntDict ints = {
                {"ropeDim", rope.ropeDim}, {"startPos", startPos}, {"posStep", posStep},
                {"inverse", inverse ? 1 : 0}, {"originalSeqLen", rope.originalSeqLen},
                {"betaFast", rope.betaFast}, {"betaSlow", rope.betaSlow},
                {"quantMode", quantMode}, {"quantBlock", quantBlock}
            };
            if (quantDim > 0) {
                ints["quantDim"] = quantDim;
            }
            DsExecutor().Run("DeepSeekV41RotaryQuant", {{"input", &x}},
                             {{"ropeBase", rope.base}, {"ropeFactor", rope.factor}}, ints);
        }

        void DsSparseAttention(const Data &q, const Data &chunkKV, const Data *ringKV,
                               const Data *compressedKV, const Data *cmpIdx, Data &attnSink,
                               int windowSize, int startPos, float softmaxScale, Data &output) {
            DataDict datas = {
                {"q", (Data*)&q}, {"chunkKV", (Data*)&chunkKV}, {"attnSink", &attnSink}, {"output", &output}
            };
            if (ringKV != nullptr && ringKV->dims.size() == 3) {
                datas["ringKV"] = (Data*)ringKV;
            }
            if (compressedKV != nullptr && cmpIdx != nullptr && compressedKV->dims.size() == 3 &&
                compressedKV->dims[1] > 0 && cmpIdx->dims.size() == 3) {
                datas["compressedKV"] = (Data*)compressedKV;
                datas["cmpIdx"] = (Data*)cmpIdx;
            }
            DsExecutor().Run("DeepSeekV41SparseAttention", datas, {{"softmaxScale", softmaxScale}},
                             {{"windowSize", windowSize}, {"startPos", startPos}});
        }

        void DsWindowStore(const Data &chunkKV, Data &ring, int startPos, int windowSize, int rowCount = -1) {
            DsExecutor().Run("DeepSeekV41WindowStore", {
                {"chunk", (Data*)&chunkKV}, {"ring", &ring}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}, {"rowCount", rowCount}});
        }

        float DsSoftplus(float x) {
            return x > 20.0f ? x : std::log1p(std::exp(x));
        }

        float DsSigmoid(float x) {
            return x >= 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
        }

        // ---------------- 接受率与分段计时 ----------------
        // FASTLLM_DSPARK_STATS 默认 1，打印总体及逐位置接受率；=0 关闭，=2 打印详细统计及逐轮耗时。
        // FASTLLM_DSPARK_STATS_EVERY=N 控制中途汇总的频率（默认 32，0 表示只在退出时打印）。
        //
        // 草稿阶段（三个草稿层 + markov head + confidence head）与校验阶段分开计时。
        // 草稿层的路由专家跑在 moe_device 上：放 cpu / numa 时是同步的，计时准确；
        // 注意力等 GPU 上的部分是异步下发的，要拿到真实耗时需要同时设 FASTLLM_CUDA_SYNC=1，
        // 否则这些项只反映 kernel launch 的时间。
        //
        // "校验前向"与"普通单 token 前向"分别累计，两者之差就是多校验 N 个候选的边际代价；
        // 路由专家在 CPU 上时代价与"选中专家的权重字节数"成正比而不是与 token 数成正比，
        // 所以这个差值通常远小于 N 倍。

        double DsNowMs() {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        struct DsparkStat {
            uint64_t verifyRounds = 0;      // 带候选的校验前向
            uint64_t plainRounds = 0;       // 没有候选的单 token 前向（dspark 已开启）
            uint64_t pendingHits = 0;       // 直接从待发队列出队、完全没有前向的轮
            uint64_t proposed = 0;          // 实际参与校验的候选数
            uint64_t accepted = 0;
            uint64_t generatedByDraft = 0;  // 由草稿模型产出、还没经过置信度筛选的候选数
            std::vector<uint64_t> verifyOfferHist;
            std::vector<uint64_t> acceptHist;   // 接受长度 0..blockSize
            std::vector<uint64_t> offerHist;    // 置信度截断后送去校验的候选数 0..blockSize
            uint64_t noProposal = 0;        // 该轮没有可用候选（还没生成 / 锚点不匹配）
            double verifyForward = 0.0, plainForward = 0.0, commit = 0.0;
            double draftTotal = 0.0, draftLayers = 0.0;
            double draftHead = 0.0, draftMarkov = 0.0, draftConf = 0.0;
            double draftMarkovBias = 0.0, draftMarkovArgmax = 0.0;
            uint64_t draftMarkovFused = 0;
            uint64_t draftCalls = 0;        // 草稿前向批数，独立请求回退按一请求一批统计
            double contextMs = 0.0;         // 按请求更新草稿滑窗；前向与采样按兼容请求合批统计
            uint64_t contextCalls = 0;
        };

        // DsparkRunDraft 记录当前批的分段耗时，由 DsparkGenerateDrafts 汇总。
        struct DsDraftTiming {
            double layers = 0.0, head = 0.0, markov = 0.0, conf = 0.0;
            // markov 内部再拆：bias 是 embed + [vocab, rank] 投影，argmax 含 TopK 与同步回主机。
            // 后者每个位置一次，block_size 个位置就是 block_size 次设备同步。
            double markovBias = 0.0, markovArgmax = 0.0;
            bool markovFused = false;      // 走了融合 kernel（此时不再拆投影 / argmax）
            void Reset() {
                layers = head = markov = conf = markovBias = markovArgmax = 0.0;
                markovFused = false;
            }
        };

        DsDraftTiming &DsDraftTimingSlot() {
            static thread_local DsDraftTiming timing;
            return timing;
        }

        struct DsparkProfiler {
            int level = 1;
            uint64_t reportEvery = 32;
            std::mutex mutex;
            DsparkStat stat;
            uint64_t sinceReport = 0;
            int blockSize = 0;
            // 头几轮包含 CUDA context、显存池、权重量化缓存的一次性开销，
            // 会把均值拉得没法看（实测第一次 decode 比稳态慢一个数量级），默认跳过。
            uint64_t warmupLeft = 3;
            uint64_t warmupSkipped = 0;

            DsparkProfiler() {
                const char *v = std::getenv("FASTLLM_DSPARK_STATS");
                if (v != nullptr && v[0] != '\0') {
                    level = strcmp(v, "0") == 0 ? 0 : std::max(1, atoi(v));
                }
                const char *e = std::getenv("FASTLLM_DSPARK_STATS_EVERY");
                if (e != nullptr && e[0] != '\0') {
                    long long n = atoll(e);
                    reportEvery = n > 0 ? (uint64_t)n : 0;
                }
                const char *w = std::getenv("FASTLLM_DSPARK_STATS_WARMUP");
                if (w != nullptr && w[0] != '\0') {
                    long long n = atoll(w);
                    warmupLeft = n > 0 ? (uint64_t)n : 0;
                }
            }

            // 调用方已经持有 mutex
            bool WarmedLocked() const { return warmupLeft == 0; }

            ~DsparkProfiler() {
                if (level > 0) {
                    Report("汇总");
                }
            }

            bool On() const { return level > 0; }

            void SetBlockSize(int n) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (blockSize != n) {
                    blockSize = n;
                    stat.acceptHist.assign((size_t)n + 1, 0);
                    stat.verifyOfferHist.assign((size_t)n + 1, 0);
                    stat.offerHist.assign((size_t)n + 1, 0);
                }
            }

            void AddPendingHit() {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.pendingHits++;
            }

            void AddNoProposal() {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.noProposal++;
            }

            // 草稿模型产出了 generated 个候选，置信度筛选后送去校验 offered 个
            void AddProposal(int generated, int offered) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.generatedByDraft += (uint64_t)generated;
                if (offered >= 0 && offered < (int)stat.offerHist.size()) {
                    stat.offerHist[offered]++;
                }
            }

            void AddDraft(double total, const DsDraftTiming &parts) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.draftCalls++;
                stat.draftTotal += total;
                stat.draftLayers += parts.layers;
                stat.draftHead += parts.head;
                stat.draftMarkov += parts.markov;
                stat.draftMarkovBias += parts.markovBias;
                stat.draftMarkovArgmax += parts.markovArgmax;
                stat.draftMarkovFused += parts.markovFused ? 1 : 0;
                stat.draftConf += parts.conf;
            }

            void AddContext(double ms) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.contextCalls++;
                stat.contextMs += ms;
            }

            void AddPlainForward(double ms) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (warmupLeft > 0) {
                    warmupLeft--;
                    warmupSkipped++;
                    return;
                }
                stat.plainRounds++;
                stat.plainForward += ms;
            }

            void AddVerify(int drafts, int accepted, double forwardMs, double commitMs) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (warmupLeft > 0) {
                    warmupLeft--;
                    warmupSkipped++;
                    return;
                }
                stat.verifyRounds++;
                if (drafts >= 0 && drafts < (int)stat.verifyOfferHist.size()) ++stat.verifyOfferHist[drafts];
                stat.proposed += (uint64_t)drafts;
                stat.accepted += (uint64_t)accepted;
                stat.verifyForward += forwardMs;
                stat.commit += commitMs;
                if (accepted >= 0 && accepted < (int)stat.acceptHist.size()) {
                    stat.acceptHist[accepted]++;
                }
                if (level >= 2) {
                    printf("[DSpark] round %llu：候选 %d 接受 %d，校验前向 %.3f ms，回滚 %.3f ms\n",
                           (unsigned long long)stat.verifyRounds, drafts, accepted, forwardMs, commitMs);
                    fflush(stdout);
                }
                if (reportEvery > 0 && ++sinceReport >= reportEvery) {
                    sinceReport = 0;
                    ReportLocked("进行中");
                }
            }

            void Report(const char *tag) {
                std::lock_guard<std::mutex> guard(mutex);
                ReportLocked(tag);
            }

            void ReportLocked(const char *tag) {
                const DsparkStat &s = stat;
                if (s.verifyRounds == 0 && s.plainRounds == 0 && s.draftCalls == 0) {
                    return;
                }
                if (level == 1) {
                    printf("[DeepSeek-V4.1 DSpark] accept_rate=%.2f%% (%llu/%llu), pos_accept_rate=[",
                           s.proposed > 0 ? 100.0 * (double)s.accepted / (double)s.proposed : 0.0,
                           (unsigned long long)s.accepted, (unsigned long long)s.proposed);
                    for (int i = 0; i < blockSize; ++i) {
                        uint64_t attempts = 0, accepts = 0;
                        for (int j = i + 1; j <= blockSize; ++j) {
                            attempts += s.verifyOfferHist[j];
                            accepts += s.acceptHist[j];
                        }
                        printf("%s%.2f%%", i == 0 ? "" : ", ",
                               attempts ? 100.0 * (double)accepts / attempts : 0.0);
                    }
                    printf("].\n");
                    fflush(stdout);
                    return;
                }
                const double verify = (double)std::max<uint64_t>(s.verifyRounds, 1);
                const double plain = (double)std::max<uint64_t>(s.plainRounds, 1);
                const double draft = (double)std::max<uint64_t>(s.draftCalls, 1);
                const uint64_t forwards = s.verifyRounds + s.plainRounds;
                const uint64_t outputs = forwards + s.pendingHits;

                printf("[DSpark %s] 前向 %llu 次（校验 %llu + 普通 %llu），出队 %llu，共产出 %llu 个 token"
                       "；每次前向 %.3f 个 token（已跳过 %llu 轮预热）\n",
                       tag, (unsigned long long)forwards, (unsigned long long)s.verifyRounds,
                       (unsigned long long)s.plainRounds, (unsigned long long)s.pendingHits,
                       (unsigned long long)outputs,
                       forwards > 0 ? (double)outputs / (double)forwards : 0.0,
                       (unsigned long long)warmupSkipped);

                if (s.verifyRounds > 0) {
                    printf("[DSpark %s] 候选 %llu 接受 %llu（接受率 %.1f%%），平均每轮接受 %.2f 个"
                           "；草稿产出 %llu 个候选，置信度筛掉 %.1f%%\n",
                           tag, (unsigned long long)s.proposed, (unsigned long long)s.accepted,
                           s.proposed > 0 ? 100.0 * (double)s.accepted / (double)s.proposed : 0.0,
                           (double)s.accepted / verify,
                           (unsigned long long)s.generatedByDraft,
                           s.generatedByDraft > 0 ?
                               100.0 * (double)(s.generatedByDraft - s.proposed) / (double)s.generatedByDraft : 0.0);
                    PrintHist(tag, "接受长度", s.acceptHist, s.verifyRounds);
                }
                if (!s.offerHist.empty()) {
                    uint64_t total = 0;
                    for (uint64_t v : s.offerHist) {
                        total += v;
                    }
                    if (total > 0) {
                        PrintHist(tag, "送检候选数", s.offerHist, total);
                    }
                }
                if (s.noProposal > 0) {
                    printf("[DSpark %s] 没有可用候选的轮次 %llu\n", tag, (unsigned long long)s.noProposal);
                }

                printf("[DSpark %s] 每次前向：校验 %.3f ms（%llu 次）/ 普通 %.3f ms（%llu 次），"
                       "差 %.3f ms；回滚 %.3f ms\n",
                       tag, s.verifyForward / verify, (unsigned long long)s.verifyRounds,
                       s.plainForward / plain, (unsigned long long)s.plainRounds,
                       s.verifyRounds > 0 && s.plainRounds > 0 ?
                           s.verifyForward / verify - s.plainForward / plain : 0.0,
                       s.commit / verify);
                printf("[DSpark %s] 每批草稿 %.3f ms（三层 %.3f + head %.3f + markov %.3f"
                       " + confidence %.3f；%llu 批）\n",
                       tag, s.draftTotal / draft, s.draftLayers / draft,
                       s.draftHead / draft, s.draftMarkov / draft, s.draftConf / draft,
                       (unsigned long long)s.draftCalls);
                if (s.draftCalls > 0 && s.draftMarkovFused == s.draftCalls) {
                    printf("[DSpark %s] 其中 markov %.3f ms（融合 kernel，%d 个位置的链留在设备上，"
                           "主机只取回一次）\n", tag, s.draftMarkov / draft, blockSize);
                } else {
                    printf("[DSpark %s] 其中 markov %.3f ms = 投影 %.3f + argmax/同步 %.3f"
                           "（%d 个位置逐位置串行，融合 kernel 用上 %llu / %llu 次）\n",
                           tag, s.draftMarkov / draft, s.draftMarkovBias / draft,
                           s.draftMarkovArgmax / draft, blockSize,
                           (unsigned long long)s.draftMarkovFused, (unsigned long long)s.draftCalls);
                }
                if (s.contextCalls > 0) {
                    printf("[DSpark %s] 草稿滑窗更新（按请求）%llu 次，每次 %.3f ms\n",
                           tag, (unsigned long long)s.contextCalls,
                           s.contextMs / (double)s.contextCalls);
                }
                // 每产出一个 token 的实际开销：目标前向 + 草稿 + 回滚
                const double totalMs = s.verifyForward + s.plainForward + s.draftTotal + s.contextMs + s.commit;
                if (outputs > 0) {
                    printf("[DSpark %s] 每个输出 token 合计 %.3f ms（目标前向 %.3f + 草稿 %.3f + 回滚 %.3f）\n",
                           tag, totalMs / (double)outputs,
                           (s.verifyForward + s.plainForward) / (double)outputs,
                           (s.draftTotal + s.contextMs) / (double)outputs,
                           s.commit / (double)outputs);
                }
                fflush(stdout);
            }

            static void PrintHist(const char *tag, const char *name,
                                  const std::vector<uint64_t> &hist, uint64_t total) {
                if (hist.empty() || total == 0) {
                    return;
                }
                std::string line;
                char buffer[64];
                for (size_t i = 0; i < hist.size(); i++) {
                    snprintf(buffer, sizeof(buffer), "%s%d:%llu(%.0f%%)", i == 0 ? "" : " ",
                             (int)i, (unsigned long long)hist[i], 100.0 * (double)hist[i] / (double)total);
                    line += buffer;
                }
                printf("[DSpark %s] %s分布 %s\n", tag, name, line.c_str());
            }
        };

        DsparkProfiler &DsProfiler() {
            static DsparkProfiler profiler;
            return profiler;
        }

        // FASTLLM_DSPARK_PROBE_EVERY=N：每 N 轮故意不带候选走一次普通单 token 前向，
        // 给"校验 N 个候选"提供一个同等缓存状态下的对照基线。N=0（默认）关闭。
        // 只在诊断时打开：这些轮次不投机，会按比例损失一点吞吐。
        bool DsProbeThisRound() {
            static int every = DsEnvInt("FASTLLM_DSPARK_PROBE_EVERY", 0);
            if (every <= 0) {
                return false;
            }
            static std::atomic<long long> counter{0};
            return (counter.fetch_add(1) % every) == 0;
        }

        // 计时开销可以忽略，但关掉统计时连时钟都不读
        inline double DsTick() {
            return DsProfiler().On() ? DsNowMs() : 0.0;
        }

        inline double DsElapsed(double start) {
            return DsProfiler().On() ? DsNowMs() - start : 0.0;
        }

        // ---- 测试用的调试钩子 ----
        // FASTLLM_DSPARK_FORCE_DRAFTS：每行一组候选 token（空格分隔），按顺序替换模型
        //   自己产生的候选，用来构造"接受 0 个 / 接受一部分 / 全部接受"三种回滚场景。
        // FASTLLM_DSPARK_STATS_FILE：每轮校验追加一行 "round drafts accepted"。
        struct DsForcedDrafts {
            std::vector<std::vector<int> > lines;
            size_t cursor = 0;
            bool loaded = false;
            std::mutex locker;
        };

        DsForcedDrafts &DsForced() {
            static DsForcedDrafts forced;
            return forced;
        }

        bool DsNextForcedDrafts(std::vector<int> &out) {
            DsForcedDrafts &forced = DsForced();
            std::lock_guard<std::mutex> guard(forced.locker);
            if (!forced.loaded) {
                forced.loaded = true;
                const char *path = std::getenv("FASTLLM_DSPARK_FORCE_DRAFTS");
                if (path != nullptr && path[0] != '\0') {
                    std::ifstream file(path);
                    std::string line;
                    while (std::getline(file, line)) {
                        std::istringstream stream(line);
                        std::vector<int> row;
                        int value = 0;
                        while (stream >> value) {
                            row.push_back(value);
                        }
                        if (!row.empty()) {
                            forced.lines.push_back(row);
                        }
                    }
                    printf("[Fastllm] DeepSeek-V4.1 DSpark: loaded %d forced draft rows from %s\n",
                           (int)forced.lines.size(), path);
                    fflush(stdout);
                }
            }
            if (forced.cursor >= forced.lines.size()) {
                return false;
            }
            out = forced.lines[forced.cursor++];
            return true;
        }

        void DsRecordStats(long long round, int drafts, int accepted) {
            static const char *path = std::getenv("FASTLLM_DSPARK_STATS_FILE");
            if (path == nullptr || path[0] == '\0') {
                return;
            }
            static std::mutex locker;
            std::lock_guard<std::mutex> guard(locker);
            FILE *file = fopen(path, "a");
            if (file != nullptr) {
                fprintf(file, "%lld %d %d\n", round, drafts, accepted);
                fclose(file);
            }
        }

        std::vector<int> DsReadIds(const Data &data) {
            Data cpu;
            const Data *src = &data;
            if (data.dataDevice != DataDevice::CPU) {
                cpu.CopyFrom(data);
                cpu.ToDevice(DataDevice::CPU);
                src = &cpu;
            }
            std::vector<int> ret;
            uint64_t n = src->Count(0);
            ret.reserve(n);
            for (uint64_t i = 0; i < n; i++) {
                if (src->dataType == DataType::FLOAT32) {
                    ret.push_back((int)(((const float*)src->cpuData)[i] + 1e-6));
                } else if (src->dataType == DataType::INT32) {
                    ret.push_back(((const int32_t*)src->cpuData)[i]);
                } else {
                    ErrorInFastLLM("DeepSeekV41 DSpark: unsupported id dtype.");
                }
            }
            return ret;
        }

        // 取一批 logits（[1, rows, vocab]）每行的 argmax
        std::vector<int> DsArgmaxRows(const Data &logits) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            const int rows = topk.dims[topk.dims.size() - 2];
            const int stride = topk.dims[topk.dims.size() - 1];   // (id, score)
            const float *data = (const float*)topk.cpuData;
            std::vector<int> ret(rows);
            for (int i = 0; i < rows; i++) {
                ret[i] = (int)(data[(uint64_t)i * stride] + 1e-3);
            }
            return ret;
        }
    }

    // ==================== 参数与权重 ====================

    std::string DeepSeekV41Model::SelectSpecialWeightDevice(const std::string &name, int layerId) const {
        // Preserve target shared/dense placement before applying draft-only overrides.
        const std::string device = DeepSeekV4Model::SelectSpecialWeightDevice(name, layerId);
        // External GGUF MXFP4 is converted to the native compact layout once.
        // It cannot be lazily read as native FP4 directly from the GGUF file.
        if (name.compare(0, 4, "mtp.") == 0 && weight.dicts.count("dspark_external_gguf") &&
            (device == "disk" || device.compare(0, 5, "disk:") == 0)) {
            return GetFastllmEnv().activateNuma ? "numa" : "cpu";
        }
        return device;
    }

    void DeepSeekV41Model::InitDsparkParams() {
        v41DsparkTokens = std::max(0, DsEnvInt("FASTLLM_DSPARK_TOKENS", 0));
        v41DsparkBlockSize = 0;
        v41DsparkEnabled = false;
        auto dictInt = [&](const std::string &key, int fallback) {
            auto it = this->weight.dicts.find(key);
            return it == this->weight.dicts.end() ? fallback : atoi(it->second.c_str());
        };
        if (v41DsparkTokens <= 0) {
            return;
        }
        v41DsparkBlockSize = dictInt("dspark_block_size", 0);
        v41DsparkLayers = dictInt("num_nextn_predict_layers", dictInt("n_mtp_layers", 0));
        v41DsparkNoiseTokenId = dictInt("dspark_noise_token_id", -1);
        v41DsparkMarkovRank = dictInt("dspark_markov_rank", 0);
        v41DsparkExperts = dictInt("dspark_n_routed_experts", 0);
        v41DsparkTopk = dictInt("dspark_num_experts_per_tok", 0);
        v41DsparkTargetLayerIds.clear();
        auto it = this->weight.dicts.find("dspark_target_layer_ids");
        if (it != this->weight.dicts.end()) {
            std::string err;
            auto parsed = json11::Json::parse(it->second, err);
            if (err.empty() && parsed.is_array()) {
                for (const auto &item : parsed.array_items()) {
                    v41DsparkTargetLayerIds.push_back(item.int_value());
                }
            }
        }
        if (v41DsparkBlockSize <= 0 || v41DsparkLayers <= 0 || v41DsparkNoiseTokenId < 0 ||
            v41DsparkMarkovRank <= 0 || v41DsparkExperts <= 0 || v41DsparkTopk <= 0 ||
            v41DsparkTargetLayerIds.empty()) {
            printf("[Fastllm] DeepSeek-V4.1: the checkpoint has no usable DSpark configuration, "
                   "speculative decoding stays off.\n");
            fflush(stdout);
            v41DsparkTokens = 0;
            return;
        }
        const int trainedBlockSize = v41DsparkBlockSize;
        // The block is a runtime sequence length, not a weight dimension.
        // Keep the trained shape when verifying a shorter prefix, and extend
        // the entire draft forward (including its bidirectional attention)
        // when more candidates are requested. Every candidate is verified.
        v41DsparkBlockSize = std::max(trainedBlockSize, v41DsparkTokens);
        for (int layer : v41DsparkTargetLayerIds) {
            AssertInFastLLM(layer >= 0 && layer < block_cnt,
                            "DeepSeekV41 DSpark: dspark_target_layer_ids is out of range.");
        }
        // 草稿层的 compress_ratio 必须是 0（纯滑窗注意力）
        for (int stage = 0; stage < v41DsparkLayers; stage++) {
            const int idx = block_cnt + stage;
            AssertInFastLLM((int)compress_ratios.size() <= idx || compress_ratios[idx] == 0,
                            "DeepSeekV41 DSpark: draft layers must use pure sliding-window attention.");
        }
        v41DsparkEnabled = true;
        const std::string device = SelectDeviceFromMap(this->deviceMap, block_cnt, block_cnt);
        v41DsparkTpDevices.clear();
        if (device == "multicuda" || device.find("multicuda:") == 0) {
            std::map<int, int> ratios;
            v41DsparkTpDevices = ParseDeviceIds(device, "multicuda", ratios);
#ifdef USE_CUDA
            if (v41DsparkTpDevices.empty()) {
                for (int i = 0; i < FastllmCudaGetDeviceCount(); i++) v41DsparkTpDevices.push_back(i);
            }
#endif
        }
        v41DsparkConfidenceThreshold =
            DsEnvFloat("FASTLLM_DSPARK_CONFIDENCE_THRESHOLD", 0.0f);
        if (!(v41DsparkConfidenceThreshold >= 0.0f && v41DsparkConfidenceThreshold <= 1.0f)) {
            v41DsparkConfidenceThreshold = 0.0f;
        }
        v41IsDsparkTarget.assign(block_cnt, 0);
        for (int layer : v41DsparkTargetLayerIds) {
            v41IsDsparkTarget[layer] = 1;
        }

        // 草稿层的权重：专家合并规则、设备选择、量化白名单
        for (int stage = 0; stage < v41DsparkLayers; stage++) {
            const std::string pre = "mtp." + std::to_string(stage);
            const int layerId = std::max(0, block_cnt - v41DsparkLayers + stage);
            for (int expert = -1; expert < v41DsparkExperts; expert++) {
                std::string expertPre = pre + ".ffn.";
                expertPre += expert < 0 ? "shared_experts" : ("experts." + std::to_string(expert));
                const std::string w1 = expertPre + ".w1.weight";
                const std::string w3 = expertPre + ".w3.weight";
                const std::string gateup = expertPre + ".gateup.weight";
                const std::string down = expertPre + ".w2.weight";
                this->weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle({w1, w3}, gateup, std::string("linearSwiglu"))}));
                if (expert >= 0 || !GetCudaSharedExpert()) {
                    this->AddSpecialWeight(gateup, "linearSwiglu", layerId);
                    this->AddSpecialWeight(down, "linearColumn", layerId);
                }
                this->moeLinears.insert(w1);
                this->moeLinears.insert(w3);
                this->moeLinears.insert(down);
            }
            this->cantQuantLinears.insert(pre + ".attn.wkv.weight");
            this->cantQuantLinears.insert(pre + ".attn.wo_a.weight");
            this->cantQuantLinears.insert(pre + ".ffn.gate.weight");
        }
        DsProfiler().SetBlockSize(v41DsparkTokens);
        printf("[Fastllm] DeepSeek-V4.1 DSpark: %d draft layers, trained block %d, draft block %d (verifying %d), "
               "%d experts (top-%d), target layers = [", v41DsparkLayers, trainedBlockSize, v41DsparkBlockSize,
               v41DsparkTokens, v41DsparkExperts, v41DsparkTopk);
        for (size_t i = 0; i < v41DsparkTargetLayerIds.size(); i++) {
            printf("%s%d", i == 0 ? "" : ", ", v41DsparkTargetLayerIds[i]);
        }
        printf("], confidence threshold %.3f\n", v41DsparkConfidenceThreshold);
        fflush(stdout);
    }

    void DeepSeekV41Model::ApplyDsparkDevice(bool targetComplete) {
        if (v41DsparkTpDevices.empty()) {
            ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
            return;
        }
        // The draft backbone owns ordinary single-device buffers. Only its
        // shared output head uses the target model's existing TP shards.
        // Leave the CUDA executor's device list empty: TP worker kernels use
        // their own current device when moving rank-local weights to CUDA.
#ifdef USE_CUDA
        if (DsExecutor().GetFirstDeviceType() == "multicuda" && !dsparkCapturing && !targetComplete) {
            FastllmCudaSyncDevice(v41DsparkTpDevices.front());
        }
#endif
        DsExecutor().SetFirstDevice("cuda");
#ifdef USE_CUDA
        FastllmCudaSetDevice(v41DsparkTpDevices.front());
#endif
    }

    void DeepSeekV41Model::DsparkEmbedInput(const Data &inputIds, Data &hiddenStates) {
        Data &embedding = weight["embed.weight"];
        // Target prefill has placed CUDA embeddings on the first PP stage.
        // Query there and transfer only selected rows before expanding HC.
        // CPU/GGUF lookups and single-device/TP drafts need no device switch.
        const bool ppCudaEmbedding = v41DsparkTpDevices.empty() && deviceMap.size() > 1 &&
                                     embedding.dataDevice == DataDevice::CUDA;
        if (ppCudaEmbedding) {
            ApplyDeviceMap(this->deviceMap, 1, block_cnt);
        }
        Data embedOut;
        Embedding(inputIds, embedding, embedOut);
        ToDataType(embedOut, DataType::BFLOAT16);
        if (ppCudaEmbedding) {
            ApplyDsparkDevice();
        }
        embedOut.Reshape({1, static_cast<int>(inputIds.Count(0)), 1, embed_dim});
        Repeat(embedOut, 2, hc_mult, hiddenStates);
    }

    void DeepSeekV41Model::DsparkProjectHead(Data &input, Data &output, bool persistent, bool sharded) {
        if (sharded && !v41DsparkTpDevices.empty())
            ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
        if (v41DsparkTpDevices.empty() && TryBFloat16HeadLinear(input, output)) return;
#ifdef USE_CUDA
        if (persistent && !sharded && input.dataType == DataType::BFLOAT16 && !v41DsparkTpDevices.empty() &&
            !quantizedLinearNames.count("head.weight")) {
            ApplyDsparkDevice();
            FastllmMultiCudaSetDevice(v41DsparkTpDevices);
            if (MultiCudaLinearOutputGatherPersistent(input, weight["head.weight"], output,
                                                     v41DsparkHeadWorkspace)) return;
        }
#endif
        Data floatInput;
        Data *headInput = &input;
        if (weight["head.weight"].dataType == DataType::FLOAT32 && input.dataType != DataType::FLOAT32) {
            ToDataType(input, floatInput, DataType::FLOAT32);
            headInput = &floatInput;
        }
        if (v41DsparkTpDevices.empty()) {
            LinearWithActivationQuant(*headInput, "head.weight", output);
            return;
        }
        if (!sharded) ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
        Data quantized;
        Data *source = headInput;
        if (quantizedLinearNames.count("head.weight")) {
            DsExecutor().Run("DeepSeekV41QuantizeActivation",
                {{"input", &input}, {"output", &quantized}}, {}, {});
            source = &quantized;
        }
        if (sharded) {
            // Preserve the same casts and quantization as the gathered draft head,
            // while keeping the target executor and vocabulary layout for capture.
            Linear(*source, weight["head.weight"], *GetEmptyData(), output);
            return;
        }
        // Quantization and target operators can leave another rank current.
        // Drain their streams and allocate the gather destination on rank 0.
        ApplyDsparkDevice();
        ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
#ifdef USE_CUDA
        if (persistent && MultiCudaLinearOutputGatherPersistent(*source, weight["head.weight"], output,
                                                  v41DsparkHeadWorkspace)) {
            ApplyDsparkDevice();
            return;
        }
#endif
        // Keep the vocabulary shards intact and gather complete logits on the
        // first GPU, as in Qwen's draft output-gather linear path.
        DsExecutor().Run("Linear",
            {{"input", source}, {"weight", &weight["head.weight"]},
             {"bias", GetEmptyData()}, {"output", &output}}, {}, {{"forceOutputGather", 1}});
        ApplyDsparkDevice();
    }

    bool DeepSeekV41Model::DsparkTensorNeeded(const std::string &name) const {
        if (!v41DsparkEnabled) {
            return false;
        }
        // 只需要前 v41DsparkLayers 个 stage 的权重
        size_t dot = name.find('.', 4);
        if (dot == std::string::npos) {
            return false;
        }
        const int stage = atoi(name.substr(4, dot - 4).c_str());
        return stage >= 0 && stage < v41DsparkLayers;
    }

    void DeepSeekV41Model::DsparkBuildMoeWeights() {
        if (!v41DsparkMoeWeights.empty()) {
            return;
        }
        auto getWeightPtr = [&](const std::string &name) -> Data* {
            auto it = weight.weight.find(name);
            return it == weight.weight.end() ? nullptr : &it->second;
        };
        v41DsparkMoeWeights.resize(v41DsparkLayers);
        v41DsparkMoeBiass.resize(v41DsparkLayers);
        for (int stage = 0; stage < v41DsparkLayers; stage++) {
            const std::string pre = "mtp." + std::to_string(stage) + ".ffn";
            v41DsparkMoeWeights[stage].push_back(getWeightPtr(pre + ".shared_experts.gateup.weight"));
            v41DsparkMoeWeights[stage].push_back(getWeightPtr(pre + ".shared_experts.w2.weight"));
            v41DsparkMoeBiass[stage].push_back(nullptr);
            v41DsparkMoeBiass[stage].push_back(nullptr);
            for (int expert = 0; expert < v41DsparkExperts; expert++) {
                v41DsparkMoeWeights[stage].push_back(
                        getWeightPtr(pre + ".experts." + std::to_string(expert) + ".gateup.weight"));
                v41DsparkMoeWeights[stage].push_back(
                        getWeightPtr(pre + ".experts." + std::to_string(expert) + ".w2.weight"));
                v41DsparkMoeBiass[stage].push_back(nullptr);
                v41DsparkMoeBiass[stage].push_back(nullptr);
            }
        }
    }

    // ==================== 请求状态 ====================

    GenerationConfig DeepSeekV41Model::DsparkSamplingConfig(const GenerationConfig &config) {
        GenerationConfig normalized = config;
        // Match ForwardSegments' ordinary sampling normalization exactly.
        if (normalized.do_sample && normalized.top_k <= 1 && normalized.temperature > 1e-6f) {
            normalized.top_k = 5;
        }
        return normalized;
    }

    bool DeepSeekV41Model::DsparkSupportsRequest(const GenerationConfig &config,
                                                 const DeepSeekV41RequestState &state) const {
        if (!v41DsparkEnabled) {
            return false;
        }
        // Repetition penalties and content-dependent sampling remain unsupported.
        if (!std::isfinite(config.repeat_penalty) || std::fabs(config.repeat_penalty - 1.0f) > 1e-8f ||
            (!config.tool_call_allowed_token_ids.empty() && !config.tool_call_name_constraint_enabled &&
             !config.tool_call_parameter_name_constraint_enabled) || config.tool_call_content_sampling_enabled ||
            config.tool_call_content_sampling_active || config.output_logits || config.output_token_least > 0) {
            return false;
        }
        const GenerationConfig sampling = DsparkSamplingConfig(config);
        if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p)) {
            return false;
        }
        if (sampling.top_k > 1) {
#ifdef USE_CUDA
            // The shared Qwen rejection/sampling kernels operate on CUDA logits.
            const std::string device = SelectDeviceFromMap(this->deviceMap, block_cnt, block_cnt);
            if (device.find("cuda") == std::string::npos) return false;
#else
            return false;
#endif
        }
        return true;
    }

    static int DsPublishableDrafts(int accepted, const GenerationConfig &config, int startPos) {
        if (startPos >= 0 && config.output_token_limit > 0 && config.input_token_length > 0) {
            const int64_t emitted = int64_t(startPos) - config.input_token_length + 1;
            accepted = int(std::min<int64_t>(accepted, std::max<int64_t>(0, int64_t(config.output_token_limit) - emitted - 1)));
        }
        return accepted;
    }

    int DeepSeekV41Model::DsparkAcceptedDraftCount(const DeepSeekV41SpecScratch &scratch,
            const GenerationConfig &config, int startPos) const {
        int accepted = scratch.acceptedDrafts;
        if (accepted < 0) {
            accepted = 0;
            while (accepted < (int)scratch.draftTokens.size() &&
                   scratch.tokens[accepted] == scratch.draftTokens[accepted]) ++accepted;
        }
        // Keep the verifier's existing proposal shape at the output limit.
        // Only the publishable prefix may advance KV/history or enter pending;
        // evaluating later causal rows must not extend the request's state.
        accepted = DsPublishableDrafts(accepted, config, startPos);
        // EOS/stop is returned to the scheduler, but is not appended to
        // allTokens. Commit only the inputs preceding that output. Otherwise
        // the cache can run beyond allTokens and cannot be snapshotted.
        for (int i = 0; i <= accepted; ++i) {
            const int token = scratch.tokens[i];
            if (token == eos_token_id || eos_token_ids.count(token) || config.stop_token_ids.count(token))
                return i;
        }
        return accepted;
    }

    GenerationConfig DeepSeekV41Model::DsparkDraftConfig(const GenerationConfig &config,
            const DeepSeekV41SpecScratch &scratch, int anchorToken) {
        GenerationConfig next = DsparkSamplingConfig(config);
        if (next.tool_call_name_constraint_enabled || next.tool_call_parameter_name_constraint_enabled) {
            // The scheduler has not emitted this round yet. Include the entire
            // accepted output (including the correction/bonus), exactly once.
            if (scratch.wantAllTokens) {
                const int accepted = DsparkAcceptedDraftCount(scratch, config);
                for (int i = 0; i <= accepted; ++i)
                    AdvanceToolCallConstraintText(next.tool_call_generated_text, scratch.tokens[i]);
            } else {
                AdvanceToolCallConstraintText(next.tool_call_generated_text, anchorToken);
            }
        }
        return next;
    }

    void DeepSeekV41Model::DsparkMaskToolLogits(Data &logits, const GenerationConfig &config,
            const std::vector<int> &prefixTokens) {
        if (!config.tool_call_name_constraint_enabled && !config.tool_call_parameter_name_constraint_enabled) return;
        GenerationConfig step = config;
        const int vocab = logits.dims.back(), rows = logits.Count(0) / vocab;
        AssertInFastLLM((int)prefixTokens.size() == rows - 1, "DSpark: invalid constraint prefix.");
        std::vector<float> values;
        for (int row = 0; row < rows; ++row) {
            PrepareToolCallConstraint(step);
            const auto &allowed = step.tool_call_allowed_token_ids;
            if (!allowed.empty()) {
                values.resize(vocab);
                float *cpuRow = logits.cpuData ? (float*)logits.cpuData + (size_t)row * vocab : nullptr;
#ifdef USE_CUDA
                float *gpuRow = logits.cudaData ? (float*)logits.cudaData + (size_t)row * vocab : nullptr;
                if (logits.dataDevice == DataDevice::CUDA) {
                    if (!logits.dataDeviceIds.empty()) FastllmCudaSetDevice(logits.dataDeviceIds[0]);
                    FastllmCudaCopyFromDeviceToHost(values.data(), gpuRow, vocab * sizeof(float));
                } else
#endif
                std::copy(cpuRow, cpuRow + vocab, values.begin());
                std::vector<int> candidates;
                for (int id : allowed) if (id >= 0 && id < vocab) candidates.push_back(id);
                if (!candidates.empty()) {
                    // Ordinary constrained LLMSampling applies top-p after
                    // renormalizing top-k. Prune first so CUDA uses that same law.
                    const int k = std::min((int)candidates.size(), std::max(1, config.top_k));
                    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                        [&](int a, int b) { return values[a] > values[b] || (values[a] == values[b] && a < b); });
                    std::vector<float> masked(vocab, -std::numeric_limits<float>::infinity());
                    for (int i = 0; i < k; ++i) masked[candidates[i]] = values[candidates[i]];
#ifdef USE_CUDA
                    if (logits.dataDevice == DataDevice::CUDA)
                        FastllmCudaCopyFromHostToDevice(gpuRow, masked.data(), vocab * sizeof(float));
                    else
#endif
                    std::copy(masked.begin(), masked.end(), cpuRow);
                }
            }
            if (row + 1 < rows) AdvanceToolCallConstraintText(step.tool_call_generated_text, prefixTokens[row]);
        }
    }

    bool DeepSeekV41Model::DsparkSampleVerifyBatch(Data &logits,
            const std::vector<DeepSeekV41Segment> &segments, const std::vector<GenerationConfig> &configs) {
#ifdef USE_CUDA
        const int batch = segments.size();
        if (!batch || configs.size() != segments.size() || logits.dataType != FLOAT32 ||
            logits.dataDevice != CUDA || !logits.cudaData || logits.dims.empty() ||
            logits.multiDeviceData || logits.dataDeviceIds.size() != 1) return false;
        const int rows = segments.front().seqlen, drafts = rows - 1, vocab = logits.dims.back();
        if (drafts <= 0 || uint64_t(batch) * rows > INT_MAX ||
            logits.Count(0) != uint64_t(batch) * rows * vocab) return false;
        std::vector<GenerationConfig> sampling;
        sampling.reserve(batch);
        for (int b = 0; b < batch; ++b) {
            const auto &s = segments[b];
            if (!s.spec || !s.state || !s.state->dspark || s.seqlen != rows || s.offset != b * rows)
                return false;
            sampling.push_back(DsparkSamplingConfig(configs[b]));
            const auto &c = sampling.back(); const auto &proposal = *s.state->dspark;
            const auto &q = proposal.proposalProbs;
            if (c.top_k <= 1 || c.tool_call_name_constraint_enabled || c.tool_call_parameter_name_constraint_enabled ||
                !proposal.sampledProposal || q.dataType != FLOAT32 || q.dataDevice != CUDA || !q.cudaData ||
                q.multiDeviceData || q.dataDeviceIds != logits.dataDeviceIds || q.dims.size() != 2 ||
                q.dims[0] < drafts || q.dims[1] != vocab || s.spec->draftTokens.size() != size_t(drafts) ||
                proposal.proposalTokens.size() < size_t(drafts)) return false;
            for (int i = 0; i < drafts; ++i) {
                const int token = s.spec->draftTokens[i];
                AssertInFastLLM(token >= 0 && token < vocab && token == proposal.proposalTokens[i],
                    "DeepSeekV41 DSpark: batched proposal is out of sync.");
            }
        }
        struct Workspace {
            void *plan = FastllmCudaMtpCreateVerifier();
            Data ids{INT32};
            std::vector<int> hostIds, topKs, output, accepted;
            std::vector<float> temperatures, topPs;
            std::vector<FastllmMtpProposalView> proposals;
            ~Workspace() { FastllmCudaMtpDestroyVerifier(plan); }
        };
        if (!v41DsparkVerifier) v41DsparkVerifier = std::make_shared<Workspace>();
        auto &w = *std::static_pointer_cast<Workspace>(v41DsparkVerifier);
        FastllmCudaSetDevice(logits.dataDeviceIds.front());
        w.ids.Resize({batch, drafts}); w.ids.ToDevice(CUDA, logits.dataDeviceIds, false); w.ids.Allocate(false);
        w.hostIds.resize(size_t(batch) * drafts); w.proposals.resize(batch);
        w.temperatures.resize(size_t(batch) * rows); w.topPs.resize(w.temperatures.size());
        w.topKs.resize(w.temperatures.size()); w.output.resize(w.temperatures.size()); w.accepted.resize(batch);
        for (int b = 0; b < batch; ++b) {
            const auto &proposal = *segments[b].state->dspark;
            std::copy_n(segments[b].spec->draftTokens.data(), drafts, w.hostIds.data() + b * drafts);
            w.proposals[b] = {(const float *)proposal.proposalProbs.cudaData, nullptr,
                             (const int *)w.ids.cudaData + b * drafts};
            std::fill_n(w.temperatures.data() + b * rows, rows, sampling[b].temperature);
            std::fill_n(w.topKs.data() + b * rows, rows, sampling[b].top_k);
            std::fill_n(w.topPs.data() + b * rows, rows, sampling[b].top_p);
        }
        FastllmCudaCopyFromHostToDevice(w.ids.cudaData, w.hostIds.data(), w.hostIds.size() * sizeof(int));
        if (!FastllmCudaMtpVerifierEnqueueProbabilities(w.plan, (float *)logits.cudaData, w.proposals.data(),
                w.temperatures.data(), w.topKs.data(), w.topPs.data(), batch, drafts, vocab,
                GetFastllmEnv().cudaGraph)) return false;
        AssertInFastLLM(FastllmCudaMtpVerifierResolve(w.plan, w.output.data(), w.accepted.data()),
            "DeepSeekV41 DSpark: batched rejection sampling failed.");
        for (int b = 0; b < batch; ++b) {
            auto &scratch = *segments[b].spec;
            scratch.tokens.assign(w.output.begin() + b * rows, w.output.begin() + (b + 1) * rows);
            scratch.acceptedDrafts = w.accepted[b];
            for (int i = 0; i < w.accepted[b]; ++i)
                AssertInFastLLM(scratch.tokens[i] == scratch.draftTokens[i],
                    "DeepSeekV41 DSpark: invalid batched accepted token.");
        }
        return true;
#endif
        return false;
    }

    void DeepSeekV41Model::DsparkSampleVerify(Data &logits, const DeepSeekV41DsparkState &proposal,
            DeepSeekV41SpecScratch &scratch, const GenerationConfig &config) {
#ifdef USE_CUDA
        const GenerationConfig sampling = DsparkSamplingConfig(config);
        const int drafts = (int)scratch.draftTokens.size();
        const int rows = drafts + 1;
        AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.cudaData &&
                        logits.dataType == DataType::FLOAT32 && !logits.dims.empty() &&
                        logits.Count(0) == (uint64_t)rows * logits.dims.back() &&
                        (int)proposal.proposalTokens.size() >= drafts,
                        "DeepSeekV41 DSpark: invalid rejection sampling proposal/logits shape.");
        DsparkMaskToolLogits(logits, sampling, scratch.draftTokens);
        for (int i = 0; i < drafts; ++i) {
            AssertInFastLLM(scratch.draftTokens[i] == proposal.proposalTokens[i],
                            "DeepSeekV41 DSpark: rejection proposal is out of sync.");
        }
        std::vector<float> temperatures(rows, sampling.temperature), topPs(rows, sampling.top_p);
        std::vector<int> topKs(rows, sampling.top_k);
        scratch.tokens.assign(rows, -1);
        scratch.acceptedDrafts = -1;
        if (!logits.dataDeviceIds.empty()) FastllmCudaSetDevice(logits.dataDeviceIds[0]);
        const int device = FastllmCudaGetDevice(), vocab = logits.dims.back();
        Data localProbs(DataType::FLOAT32);
        if (drafts == 0) {
            // An anchor-only verification step has no proposal distribution.
            // Sample its target row with the same filtering law as rejection
            // verification, then commit exactly one input through the graph path.
            localProbs.Resize({1, vocab});
            localProbs.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
            localProbs.Allocate(false);
            AssertInFastLLM(FastllmCudaMtpSampleDraft((float*)logits.cudaData,
                (float*)localProbs.cudaData, temperatures.data(), topKs.data(), topPs.data(),
                scratch.tokens.data(), 1, vocab),
                "DeepSeekV41 DSpark: anchor-only target sampling failed.");
            scratch.acceptedDrafts = 0;
            return;
        }
        float *probs = (float*)proposal.proposalProbs.cudaData;
        if (proposal.sampledProposal) {
            const Data &q = proposal.proposalProbs;
            AssertInFastLLM(q.dataType == DataType::FLOAT32 && q.dataDevice == DataDevice::CUDA &&
                q.cudaData && q.dims.size() == 2 && q.dims[0] >= drafts && q.dims[1] == vocab &&
                q.dataDeviceIds.size() == 1,
                "DeepSeekV41 DSpark: invalid saved proposal probabilities.");
            if (q.dataDeviceIds[0] != device) {
                // Layer placement can put the draft head on another GPU.
                // Stage through CPU so peer access is not required.
                localProbs.Resize({drafts, vocab});
                localProbs.Allocate();
                FastllmCudaSetDevice(q.dataDeviceIds[0]);
                FastllmCudaCopyFromDeviceToHost(localProbs.cpuData, q.cudaData, localProbs.GetBytes());
                localProbs.ToDevice(DataDevice::CUDA, std::vector<int>{device});
                probs = (float*)localProbs.cudaData;
            }
        } else {
            std::vector<float> oneHot((size_t)drafts * vocab, 0.0f);
            for (int i = 0; i < drafts; ++i) {
                const int token = scratch.draftTokens[i];
                AssertInFastLLM(token >= 0 && token < vocab, "DeepSeekV41 DSpark: invalid forced token.");
                oneHot[(size_t)i * vocab + token] = 1.0f;
            }
            localProbs.CopyFrom(Data(DataType::FLOAT32, {drafts, vocab}, oneHot));
            localProbs.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            probs = (float*)localProbs.cudaData;
        }
        AssertInFastLLM(FastllmCudaMtpRejectionSampling((float*)logits.cudaData, probs,
            temperatures.data(), topKs.data(), topPs.data(), scratch.draftTokens.data(),
            scratch.tokens.data(), &scratch.acceptedDrafts, 1, drafts, vocab),
            "DeepSeekV41 DSpark: CUDA rejection sampling failed.");
        AssertInFastLLM(scratch.acceptedDrafts >= 0 && scratch.acceptedDrafts <= drafts,
                        "DeepSeekV41 DSpark: invalid accepted prefix length.");
        for (int i = 0; i <= scratch.acceptedDrafts; ++i) {
            AssertInFastLLM(scratch.tokens[i] >= 0 && scratch.tokens[i] < logits.dims.back() &&
                            (i == scratch.acceptedDrafts || scratch.tokens[i] == scratch.draftTokens[i]),
                            "DeepSeekV41 DSpark: invalid rejection sampling output.");
        }
#else
        ErrorInFastLLM("DeepSeekV41 DSpark: random sampling requires CUDA.");
#endif
    }

    std::shared_ptr<DeepSeekV41DsparkState> DeepSeekV41Model::GetOrCreateDsparkState(
            DeepSeekV41RequestState &state, int startPos) {
        if (!state.dspark) {
            if (!v41DsparkFreeStates.empty()) {
                state.dspark = std::move(v41DsparkFreeStates.back());
                v41DsparkFreeStates.pop_back();
            } else state.dspark = std::make_shared<DeepSeekV41DsparkState>();
            state.dspark->layers.resize(v41DsparkLayers);
            // 前缀缓存恢复出来的前缀没有草稿侧的滑窗（main hidden 无法从目标缓存反推），
            // committed 从恢复长度开始，filled 从 0 重新累积。
            state.dspark->Reset(startPos);
        }
        return state.dspark;
    }

    bool DeepSeekV41Model::PopPendingResponseToken(ResponseContext *context,
                                                    int previousToken, int &token) {
        if (!v41DsparkEnabled || !context || context->pastKeyValues.empty()) return false;
        auto state = GetStateByFirstKey(&context->pastKeyValues[0].first);
        if (!state) return false;
        const int next = DsparkTakePending(*state, previousToken);
        if (next < 0) return false;
        token = next;
        return true;
    }

    int DeepSeekV41Model::DsparkTakePending(DeepSeekV41RequestState &state, const Data &inputIds,
                                            int seqlen) {
        if (!state.dspark || state.dspark->pending.empty()) return -1;
        const auto ids = seqlen == 1 ? DsReadIds(inputIds) : std::vector<int>{};
        return DsparkTakePending(state, ids.size() == 1 ? ids[0] : -1);
    }

    int DeepSeekV41Model::DsparkTakePending(DeepSeekV41RequestState &state, int previousToken) {
        if (!state.dspark || state.dspark->pending.empty()) return -1;
        auto &pending = state.dspark->pending;
        // Publish through the serving loop's ordinary stop/grammar/history
        // checks before scheduling another shared verification forward.
        if (previousToken == pending.front().first) {
            const int token = pending.front().second;
            pending.pop_front();
            DsProfiler().AddPendingHit();
            return token;
        }
        // 不匹配：丢弃剩余的已校验 token，退回普通解码。缓存里多出来的 token
        // 位置在 state.totalLen 里，下一次前向会因为位置不一致而报错，因此这里
        // 也把草稿状态标为失效，由调用方走完整前向。
        pending.clear();
        state.dspark->disabled = true;
        return -1;
    }

    bool DeepSeekV41Model::ReadPublishedDsparkIds(const DeepSeekV41Segment &seg) {
#ifdef USE_CUDA
        if (!seg.spec->deviceDecision) return false;
        auto &w = *seg.state->dspark->workspace;
        FastllmCudaEventSynchronize(w.top1Ready);
        for (int i = 0; i < seg.seqlen; ++i) seg.spec->tokens[i] = int(w.hostTop1[2*i] + 1e-3f);
        return true;
#else
        return false;
#endif
    }

    void DeepSeekV41Model::PublishDsparkDecision(const DeepSeekV41Segment &seg,
            const GenerationConfig &config, const Data &topk) {
#ifdef USE_CUDA
        auto &scratch = *seg.spec;
        if (!scratch.verifyPacket || !seg.state->dspark) return;
        auto &ds = *seg.state->dspark;
        auto w = ds.workspace;
        if (!w || !w->verifyDecision.cudaData || !w->deviceInputs ||
            topk.dataType != FLOAT32 || topk.dataDevice != CUDA || !topk.cudaData || topk.dims.empty() ||
            topk.Count(0) != size_t(seg.seqlen) * topk.dims.back() || ds.committed != seg.startPos ||
            topk.dims.back() != 2 || !w->hostTop1 || !w->top1Ready ||
            w->devices.front() != FastllmCudaGetDevice() || scratch.attentionRanks.size() != 1 ||
            !scratch.attentionRanks.front()->commitDeviceSpan) return;
        FastllmV41VerifyStops stops{};
        auto addStop = [&](int id) {
            if (std::find(stops.ids, stops.ids + stops.size, id) != stops.ids + stops.size) return true;
            if (stops.size == 16) return false;
            stops.ids[stops.size++] = id;
            return true;
        };
        addStop(eos_token_id);
        for (int id : eos_token_ids) if (!addStop(id)) return;
        for (int id : config.stop_token_ids) if (!addStop(id)) return;
        int *packet = static_cast<int *>(w->verifyDecision.cudaData);
        if (!FastllmCudaV41ResolveDraft(static_cast<const float *>(topk.cudaData), topk.dims.back(),
                static_cast<const int *>(w->markovTokens.cudaData), seg.seqlen, seg.startPos, 0,
                DsPublishableDrafts(seg.seqlen-1, config, seg.startPos), stops, packet)) return;
        // The readback event precedes reusable context/draft work on the root
        // stream. Sampling waits just for the verifier IDs, never the next draft.
        if (!FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(w->hostTop1, topk.cudaData, topk.GetBytes())) return;
        FastllmCudaEventRecordCurrentThread(w->top1Ready);
        scratch.deviceDecision = true;
        auto &local = *scratch.attentionRanks.front();
        local.kvCommitted = w->kvReady && local.commitDeviceSpan(w->devices.front(), packet);
        if (local.kvCommitted) {
            FastllmCudaEventRecordCurrentThread(w->kvReady);
            local.deviceCommitReady = w->kvReady;
        }
        // Only reuse an existing, exact-shape context graph. The full-width
        // writes are predicated on accepting all rows; partial prefixes replay
        // the ordinary graph for their actual row count and reduction order.
        auto found = ds.contextWorkspaces.find(seg.seqlen);
        if (found == ds.contextWorkspaces.end() || !found->second || ds.disabled) return;
        auto &ctx = *found->second;
        if (ctx.devices.size() != 1 || ctx.devices.front() != w->devices.front() || ctx.execs.size() != 1 ||
            ctx.disabled || !ctx.execs.front() || ctx.cachePointers.size() != ds.layers.size() ||
            ctx.position.Count(0) != 2 || !ctx.position.cudaData || !ctx.input.cudaData) return;
        for (size_t i = 0; i < ds.layers.size(); ++i)
            if (ctx.cachePointers[i] != ds.layers[i].windowKV.cudaData) return;
        const auto &source = scratch.packedMainHidden;
        if (source.dims.size() == 3 && source.cudaData && source.dims == ctx.input.dims &&
            source.dataType == ctx.input.dataType) {
            if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(ctx.input.cudaData, source.cudaData, ctx.input.GetBytes()))
                return;
        } else {
            std::vector<const Data *> inputs;
            for (const auto &hidden : scratch.mainHidden) {
                const Data *h = &hidden;
                if (h->multiDeviceData) {
                    auto it = h->multiDeviceDatas.find(w->devices.front());
                    if (it == h->multiDeviceDatas.end()) return;
                    h = it->second;
                }
                inputs.push_back(h);
            }
            if (inputs.empty() || !FastllmCudaNaiveDraftConcat(inputs, seg.seqlen, ctx.input)) return;
        }
        if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(ctx.position.cudaData, packet + 2, 2*sizeof(int))) return;
        AssertInFastLLM(FastllmCudaGraphLaunch(ctx.execs.front()), "DSpark early context graph launch failed");
        scratch.earlyContext = true;
        // The resident full-window greedy graph has a request-private output.
        // If acceptance is partial its context stores were masked and the output
        // will be discarded. Skip this work after a recent rejection or near EOS.
        if (w->fullStreak < 2 || DsPublishableDrafts(seg.seqlen-1,config,seg.startPos) != seg.seqlen-1 ||
            ds.filled != window_size || !w->deviceInputs || w->sampledMarkov || w->hybridExperts ||
            w->diskExperts || w->disabled || w->execs.empty() || !w->position.cudaData ||
            config.tool_call_name_constraint_enabled || config.tool_call_parameter_name_constraint_enabled) return;
        if (!FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(w->position.cudaData, packet + 4, 2*sizeof(int))) return;
        DsRunDraftGraph(*w, [] {}, true);
        scratch.earlyDraft = true;
#endif
    }

    void DeepSeekV41Model::StageDsparkVerify(DeepSeekV41RequestState &state, const DeepSeekV41SpecScratch &previous) {
#ifdef USE_CUDA
        if (!state.dspark || !state.dspark->workspace || executionMode != ExecutionMode::Gpu ||
            v41AttentionDP <= 1 || !engramMeta.loaded || engram_layer_ids.empty()) return;
        auto &ds = *state.dspark;
        auto &w = *ds.workspace;
        const int layers = engram_layer_ids.size(), ngram = engram_max_ngram_size, heads = engram_n_heads;
        if (w.pendingCount != 5 || ds.committed != state.totalLen || w.devices.empty() ||
            int(state.engramHistory.size()) != ds.committed || ngram < 2 || ngram > 9 || heads <= 0 ||
            !std::is_sorted(engram_layer_ids.begin(), engram_layer_ids.end())) return;
        const int rows = w.pendingCount + 1, cols = (ngram - 1) * heads;
        const int root = w.devices.front();
        FastllmCudaSetDevice(root);
        auto allocate = [&](Data &data, size_t bytes) {
            if (data.cudaData) return;
            data.dataType = INT8; data.UpdateUnitSize(); data.Resize({int(bytes)});
            data.ToDevice(CUDA, {root}, false); data.Allocate(false);
        };
        if (!w.verifyHash.cudaData) {
            std::vector<int64_t> params;
            if (engramMeta.multipliers.size() != layers || engramMeta.offsets.size() != layers ||
                engramMeta.primes.size() != layers || engramMeta.tokenMap.empty()) return;
            for (int l = 0; l < layers; ++l) {
                const auto &m = engramMeta.multipliers[l], &p = engramMeta.primes[l], &o = engramMeta.offsets[l];
                if (m.size() != ngram || p.size() != cols || o.size() != cols ||
                    std::any_of(p.begin(), p.end(), [](int64_t v) { return v <= 0; })) return;
                params.insert(params.end(), m.begin(), m.end()); params.insert(params.end(), p.begin(), p.end());
                params.insert(params.end(), o.begin(), o.end());
            }
            allocate(w.verifyTokenMap, engramMeta.tokenMap.size() * sizeof(int));
            allocate(w.verifyHash, params.size() * sizeof(int64_t));
            FastllmCudaCopyFromHostToDevice(w.verifyTokenMap.cudaData, engramMeta.tokenMap.data(), w.verifyTokenMap.GetBytes());
            FastllmCudaCopyFromHostToDevice(w.verifyHash.cudaData, params.data(), w.verifyHash.GetBytes());
        }
        allocate(w.verifyDecision, 6 * sizeof(int));
        if (!w.hostTop1) w.hostTop1 = static_cast<float *>(FastllmCudaHostMalloc(rows * 2 * sizeof(float)));
        if (!w.top1Ready) w.top1Ready = FastllmCudaEventCreate();
        if (!w.kvReady) w.kvReady = FastllmCudaEventCreate();
        const bool packetAllocated = w.verifyPacket.cudaData != nullptr;
        allocate(w.verifyPacket, rows * (sizeof(int64_t) * layers * cols + sizeof(float) * (layers + 1)));
        if (!w.verifyReady) w.verifyReady = FastllmCudaEventCreate();
        FastllmV41VerifyHistory prefix{};
        for (int j = 0; j < ngram - 1; ++j)
            prefix.previous[j] = ds.committed > j ? state.engramHistory[ds.committed-1-j] : -1;
        // Only the owner's final Markov token is read. Keep every rank's existing
        // graph completion dependency for reusing collective buffers.
        void *packetStream = nullptr;
        if (packetAllocated && w.rootWinnerRecorded && !w.replayDone.empty()) {
            if (!w.packetStream) w.packetStream = FastllmCudaStreamCreate();
            packetStream = w.packetStream;
            if (packetStream) FastllmCudaStreamWaitEvent(packetStream, w.replayDone.front());
        }
        if (w.verifyReady && FastllmCudaV41DraftVerifyPacketOnStream(static_cast<const int *>(w.markovTokens.cudaData),
                ds.anchor, rows, static_cast<const int *>(w.verifyTokenMap.cudaData), engramMeta.tokenMap.size(),
                engramMeta.padCompressedId, image_token_id, static_cast<const int64_t *>(w.verifyHash.cudaData),
                layers, ngram, heads, prefix, w.verifyPacket.cudaData, packetStream)) {
            if (packetStream) {
                FastllmCudaEventRecord(w.verifyReady, packetStream);
                // A discarded proposal or a change of batch may reuse markovTokens.
                FastllmCudaCurrentThreadStreamWaitEvent(w.verifyReady);
            } else FastllmCudaEventRecordCurrentThread(w.verifyReady);
            w.packetPosition = ds.committed;
            ds.nextVerifyPosition = -1;
            if (previous.attentionRanks.size() == 1 && previous.attentionRanks.front()->stageNextPages &&
                previous.attentionRanks.front()->stageNextPages(ds.nextVerifyPages, ds.committed))
                ds.nextVerifyPosition = ds.committed;
        }
#endif
    }

    int DeepSeekV41Model::DsparkBuildVerifyInput(DeepSeekV41RequestState &state, int anchor,
                                                 int startPos, std::vector<int> &drafts) {
        drafts.clear();
#ifdef USE_CUDA
        if (state.dspark && state.dspark->workspace && state.dspark->workspace->pendingCount) {
            auto &proposal = *state.dspark;
            proposal.workspace->FinishProposal(&proposal.drafts);
            proposal.proposalTokens = proposal.drafts;
        }
#endif
        if (!state.dspark || state.dspark->disabled || state.dspark->drafts.empty()) {
            DsProfiler().AddNoProposal();
            return 0;
        }
        DeepSeekV41DsparkState &dspark = *state.dspark;
        if (dspark.anchorPos != startPos || dspark.committed != startPos) {
            dspark.drafts.clear();
            DsProfiler().AddNoProposal();
            return 0;
        }
        if (anchor != dspark.anchor) {
            dspark.drafts.clear();
            DsProfiler().AddNoProposal();
            return 0;
        }
        // 置信度：conditional survival 低于阈值处截断本轮校验的候选数
        const int generated = std::min((int)dspark.drafts.size(), v41DsparkTokens);
        int count = generated;
        if (v41DsparkConfidenceThreshold > 0.0f && (int)dspark.confidence.size() >= count) {
            for (int i = 0; i < count; i++) {
                if (dspark.confidence[i] < v41DsparkConfidenceThreshold) {
                    count = i;
                    break;
                }
            }
        }
        if (DsProbeThisRound()) {
            // 对照轮：丢掉候选走普通前向（下一轮 DsparkAdvance 会重新生成）
            dspark.drafts.clear();
            return 0;
        }
        DsProfiler().AddProposal(generated, count);
        if (count <= 0) {
            dspark.drafts.clear();
            return 0;
        }
        drafts.assign(dspark.drafts.begin(), dspark.drafts.begin() + count);
        {
            // 测试钩子：用外部给定的候选替换模型的候选（构造指定的接受长度）
            std::vector<int> forced;
            if (DsNextForcedDrafts(forced)) {
                if ((int)forced.size() > v41DsparkTokens) {
                    forced.resize(v41DsparkTokens);
                }
                if (forced.empty()) {
                    dspark.drafts.clear();
                    return 0;
                }
                drafts = forced;
                count = (int)forced.size();
                // A forced test proposal is deterministic, hence its actual
                // q is one-hot even when the request uses random sampling.
                dspark.sampledProposal = false;
                dspark.proposalTokens = forced;
            }
        }
        dspark.drafts.clear();
        return count;
    }

    std::vector<int> DeepSeekV41Model::ForwardSegmentsWithDspark(
            std::vector<DeepSeekV41Segment> &segments,
            const Data &inputIds,
            const Data *inputEmbeds,
            const std::vector<int> *imageMask,
            const std::vector<GenerationConfig> &generationConfigs,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float>*> *retLogits,
            std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues,
            bool allowSpeculate) {
        const int numSegments = (int)segments.size();
        if (!v41DsparkEnabled) {
            return ForwardSegments(segments, inputIds, inputEmbeds, imageMask,
                                   generationConfigs, lastTokens, retLogits,
                                   samplingPastKeyValues);
        }
        std::vector<DeepSeekV41SpecScratch> scratches(numSegments);
        std::vector<char> active(numSegments, 0);
        for (int i = 0; i < numSegments; i++) {
            DeepSeekV41RequestState &state = *segments[i].state;
            if (!DsparkSupportsRequest(generationConfigs[i], state) ||
                (state.dspark && state.dspark->disabled)) {
                continue;
            }
            active[i] = 1;
            scratches[i].captureMain = true;
            segments[i].spec = &scratches[i];
        }

        // Flatten all eligible requests into one target forward. Request-local
        // offsets are consumed by attention, sampling, KV commit and feature views.
        const auto originalSegments = segments;
        struct RestoreSegments {
            std::vector<DeepSeekV41Segment> &segments;
            const std::vector<DeepSeekV41Segment> &original;
            ~RestoreSegments() { segments = original; for (auto &s : segments) s.spec = nullptr; }
        } restore{segments, originalSegments};
        std::vector<int> draftCounts(numSegments, 0);
        Data verifyIds;
        const Data *idsPtr = &inputIds;
        const bool textOnly = inputEmbeds == nullptr && imageMask == nullptr;
        bool verifyStep = allowSpeculate && textOnly;
        for (int i = 0; i < numSegments; ++i)
            verifyStep &= active[i] && segments[i].seqlen == 1 && segments[i].startPos > 0 &&
                segments[i].state->dspark != nullptr;
        if (verifyStep) {
            const auto anchors = DsReadIds(inputIds);
            std::vector<float> tokens;
            for (int i = 0; i < numSegments; ++i) {
                auto &segment = segments[i];
                auto &scratch = scratches[i];
                const int anchor = anchors[segment.offset];
                auto &proposal = *segment.state->dspark;
                int drafts = 0;
#ifdef USE_CUDA
                const auto w = proposal.workspace;
                if (numSegments == 1 && w && w->pendingCount == 5 && w->packetPosition == segment.startPos &&
                    proposal.anchor == anchor && proposal.anchorPos == segment.startPos && proposal.committed == segment.startPos &&
                    !proposal.disabled && !std::getenv("FASTLLM_DSPARK_FORCE_DRAFTS") && !std::getenv("FASTLLM_DSPARK_PROBE_EVERY") &&
                    (max_positions <= 0 || int64_t(segment.startPos) + 6 <= max_positions) &&
                    (tokensLimit <= 0 || int64_t(segment.startPos) + 6 <= tokensLimit)) {
                    drafts = w->pendingCount;
                    scratch.draftTokens.resize(drafts);
                    scratch.verifyPacket = w->verifyPacket.cudaData; scratch.verifyPacketBytes = w->verifyPacket.GetBytes();
                    scratch.verifyProducer = w->devices.front(); scratch.verifyReady = w->verifyReady;
                    scratch.resolveDraftIds = [w, &proposal, &scratch, &verifyIds]() {
                        w->FinishProposal(&scratch.draftTokens);
                        proposal.proposalTokens = scratch.draftTokens;
                        auto *ids = static_cast<float *>(static_cast<void *>(verifyIds.cpuData));
                        for (size_t i = 0; i < scratch.draftTokens.size(); ++i) ids[1+i] = float(scratch.draftTokens[i]);
                    };
                    DsProfiler().AddProposal(drafts, drafts);
                } else
#endif
                    drafts = DsparkBuildVerifyInput(*segment.state, anchor, segment.startPos, scratch.draftTokens);
                // Output limits constrain commit, not the verification shape.
                // Context capacity is still a hard bound on speculative reads.
                int64_t capacity = max_positions;
                if (tokensLimit > 0 && (capacity <= 0 || tokensLimit < capacity)) capacity = tokensLimit;
                if (capacity > 0) {
                    drafts = int(std::min<int64_t>(drafts, std::max<int64_t>(0, capacity - segment.startPos - 1)));
                    scratch.draftTokens.resize(drafts);
                }
                draftCounts[i] = drafts;
                segment.offset = tokens.size();
                segment.seqlen = drafts + 1;
                tokens.push_back(float(anchors[originalSegments[i].offset]));
                for (int token : scratch.draftTokens) tokens.push_back(float(token));
                scratch.deferWindow = scratch.wantAllTokens = true;
                scratch.windowKV.resize(block_cnt);
                scratch.rawKV.resize(block_cnt);
                scratch.rawScore.resize(block_cnt);
                scratch.prevRawTail.assign(block_cnt, 0);
                scratch.prevBlocks.assign(block_cnt, 0);
            }
            verifyIds.CopyFrom(Data(DataType::FLOAT32, {1, int(tokens.size())}, tokens));
            idsPtr = &verifyIds;
        }

        const double forwardStart = DsTick();
        std::vector<int> ret = ForwardSegments(segments, *idsPtr, inputEmbeds, imageMask, generationConfigs,
                                               lastTokens, retLogits, samplingPastKeyValues);
        const double forwardMs = DsElapsed(forwardStart);
        if (numSegments == 1 && draftCounts[0] == 0 && originalSegments[0].seqlen == 1 && active[0])
            DsProfiler().AddPlainForward(forwardMs);

        if (verifyStep) {
            ret.resize(numSegments);
            for (int i = 0; i < numSegments; ++i) {
                AssertInFastLLM((int)scratches[i].tokens.size() == draftCounts[i] + 1,
                    "DeepSeekV41 DSpark: the verifier did not return per-position tokens.");
                scratches[i].acceptedDrafts = DsparkAcceptedDraftCount(scratches[i], generationConfigs[i], segments[i].startPos);
            }
            DsparkCommitBatch(segments);
            // Commit every request before starting any draft. Each scratch view
            // borrows a disjoint slice of the one target graph's persistent data.
            for (int i = 0; i < numSegments; ++i) {
                auto &state = *segments[i].state;
                auto &scratch = scratches[i];
                const int drafts = draftCounts[i], startPos = segments[i].startPos;
                const int accepted = scratch.acceptedDrafts;
                const double commitStart = DsTick();
                DsparkCommitPrefix(state, scratch, startPos, accepted + 1, drafts + 1);
                const double commitMs = DsElapsed(commitStart);
                if (drafts > 0) DsProfiler().AddVerify(drafts, accepted, forwardMs / numSegments, commitMs);
                ret[i] = scratch.tokens[0];
                for (int j = 1; j <= accepted; ++j)
                    state.dspark->pending.emplace_back(scratch.tokens[j - 1], scratch.tokens[j]);
                if (drafts > 0) {
                    const long long round = v41DsparkVerifyRounds.fetch_add(1) + 1;
                    v41DsparkProposedTokens.fetch_add(drafts);
                    v41DsparkAcceptedTokens.fetch_add(accepted);
                    DsRecordStats(round, drafts, accepted);
                    state.dspark->rounds++;
                    state.dspark->proposed += drafts;
                    state.dspark->accepted += accepted;
                }
            }
            std::vector<DeepSeekV41DsparkState*> draftStates;
            std::vector<int> draftAnchors;
            std::vector<GenerationConfig> draftConfigs;
            const bool advancedBatch = DsparkAdvanceBatch(segments);
            for (int i = 0; i < numSegments; ++i) {
                auto &scratch = scratches[i];
                const int anchor = scratch.tokens[scratch.acceptedDrafts];
                const auto &config = generationConfigs[i];
                const bool stopped = anchor == eos_token_id || eos_token_ids.count(anchor) || config.stop_token_ids.count(anchor);
                if (!advancedBatch) DsparkAdvance(*segments[i].state, scratch, segments[i].startPos,
                    segments[i].startPos + scratch.acceptedDrafts + 1);
                if (!stopped) {
                    draftStates.push_back(segments[i].state->dspark.get());
                    draftAnchors.push_back(anchor);
                    draftConfigs.push_back(DsparkDraftConfig(config, scratch, anchor));
                }
            }
            DsparkGenerateDrafts(draftStates, draftAnchors, draftConfigs);
            if (numSegments == 1 && draftStates.size() == 1) StageDsparkVerify(*segments.front().state, scratches.front());
            return ret;
        }

        std::vector<DeepSeekV41DsparkState*> draftStates;
        std::vector<int> draftAnchors;
        std::vector<GenerationConfig> draftConfigs;
        for (int i = 0; i < numSegments; i++) {
            if (!active[i]) {
                continue;
            }
            // prefill 分块的中间结果不是真正的下一个 token，这时只更新滑窗、不生成候选
            const int anchor = allowSpeculate && !isIntermediateChunkedPrefill && textOnly &&
                segments[i].seqlen == 1 && (int)ret.size() > i ? ret[i] : -1;
            DsparkAdvance(*segments[i].state, scratches[i], segments[i].startPos,
                          segments[i].startPos + segments[i].seqlen);
            if (anchor >= 0) {
                draftStates.push_back(segments[i].state->dspark.get());
                draftAnchors.push_back(anchor);
                draftConfigs.push_back(DsparkDraftConfig(generationConfigs[i], scratches[i], anchor));
            }
        }
        DsparkGenerateDrafts(draftStates, draftAnchors, draftConfigs);
        if (numSegments == 1 && draftStates.size() == 1) StageDsparkVerify(*segments.front().state, scratches.front());
        return ret;
    }

    // ==================== 回滚 / 提交 ====================

    bool DeepSeekV41Model::DsparkCommitBatch(const std::vector<DeepSeekV41Segment> &segments) {
        if (segments.size() < 2 || !segments.front().spec || !segments.front().spec->commitBatchKV)
            return false;
        auto &first = *segments.front().spec;
        std::vector<std::pair<int, int>> spans;
        int end = 0;
        for (const auto &s : segments) {
            if (!s.spec || s.spec->commitGroup != first.commitGroup || !s.spec->commitKV ||
                s.spec->kvCommitted || s.spec->acceptedDrafts < 0 ||
                s.spec->acceptedDrafts >= s.seqlen || s.offset < end) return false;
            spans.emplace_back(s.spec->acceptedDrafts + 1, s.offset);
            end = s.offset + s.seqlen;
        }
        first.commitBatchKV(spans);
        for (const auto &s : segments) s.spec->kvCommitted = true;
        return true;
    }

    void DeepSeekV41Model::DsparkCommitPrefix(DeepSeekV41RequestState &state,
                                              DeepSeekV41SpecScratch &scratch,
                                              int startPos, int accept, int forwarded) {
        AssertInFastLLM(accept >= 1 && accept <= forwarded,
                        "DeepSeekV41 DSpark: invalid accepted prefix length.");
        if (v41AttentionDP > 1) {
            CommitAttentionParallel(state, scratch, startPos, accept, forwarded);
            state.totalLen = startPos + accept;
            if ((int)state.engramHistory.size() > state.totalLen) state.engramHistory.resize(state.totalLen);
            return;
        }
        ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
        const int newLen = startPos + accept;
        bool batchedWindows = false, batchedTails = false;
#ifdef USE_CUDA
        ScopedTpDispatch commitDispatch(v41DsparkTpDevices.size() > 1, v41DsparkTpDevices);
        if (scratch.commitKV) {
            if (!scratch.kvCommitted) scratch.commitKV(accept);
            batchedWindows = batchedTails = true;
        } else if (v41DsparkTpDevices.size() > 1 && startPos >= 0 && window_size > 0 && accept <= window_size) {
            std::vector<MultiCudaReplicatedCopyRange> ranges;
            ranges.reserve(block_cnt * 4);
            bool valid = true;
            for (int layer = 0; layer < block_cnt; ++layer) {
                if (layer >= (int)scratch.windowKV.size() || scratch.windowKV[layer].dims.size() != 3)
                    continue;
                const Data &input = scratch.windowKV[layer];
                Data &ring = state.layers[layer].windowKV;
                if (input.dims[0] != 1 || input.dims[1] < accept || input.dims[2] <= 0 ||
                    ring.dims != std::vector<int>({1, window_size, input.dims[2]}) ||
                    ring.dataType != input.dataType) {
                    valid = false;
                    break;
                }
                const size_t rowBytes = (size_t)input.dims[2] * input.unitSize;
                const int offset = startPos % window_size;
                const int first = std::min(accept, window_size - offset);
                ranges.push_back({&ring, &input, offset * rowBytes, 0, first * rowBytes});
                if (first < accept)
                    ranges.push_back({&ring, &input, 0, first * rowBytes, (accept - first) * rowBytes});
            }
            const bool validWindows = valid;
            const size_t windowRanges = ranges.size();
            // Window rows and compressor tails are independent rank-local
            // writes. Prepare all tails before one shared worker dispatch.
            for (int layer = 0; valid && layer < block_cnt; ++layer) {
                if (!isKvSource[layer] || compress_ratios[layer] <= 0) continue;
                const int total = scratch.prevRawTail[layer] + accept;
                const int rem = total % compress_ratios[layer];
                if (rem == 0) continue;
                auto prepareTail = [&](Data &source, Data &tail) {
                    MultiCudaReplicatedCopyRange range;
                    if (!MultiCudaPrepareReplicatedRowCopy(tail, source, total - rem, rem,
                            v41DsparkTpDevices, range)) return false;
                    ranges.push_back(range);
                    return true;
                };
                auto &cache = state.layers[layer];
                valid = layer < (int)scratch.rawKV.size() &&
                    prepareTail(scratch.rawKV[layer], cache.rawTailKV);
                if (valid && compress_ratios[layer] > 1)
                    valid = layer < (int)scratch.rawScore.size() &&
                        prepareTail(scratch.rawScore[layer], cache.rawTailScore);
            }
            if (valid)
                batchedWindows = batchedTails = MultiCudaCopyReplicatedRanges(ranges, v41DsparkTpDevices);
            // An unsupported tail must not turn the already supported window
            // batch back into one dispatch per layer.
            if (!batchedWindows && validWindows) {
                ranges.resize(windowRanges);
                batchedWindows = MultiCudaCopyReplicatedRanges(ranges, v41DsparkTpDevices);
            }
        }
#endif
        const bool pipelineCommit = v41DsparkTpDevices.empty() &&
            deviceMap.size() > 1;
        for (int layer = 0; layer < block_cnt; layer++) {
            // Eager fallback and partially captured PP stages retain local KV.
            if (pipelineCommit && !scratch.commitKV) ApplyDeviceMap(deviceMap, layer + 1, block_cnt);
            DeepSeekV41LayerCache &cache = state.layers[layer];
            // 1) 滑窗环形缓冲：只写入被接受的行
            if (!batchedWindows && layer < (int)scratch.windowKV.size() &&
                scratch.windowKV[layer].dims.size() == 3) {
                // Read the accepted prefix directly from each rank's scratch.
                // Split would collapse replicas and broadcast them again.
                DsWindowStore(scratch.windowKV[layer], cache.windowKV, startPos, window_size, accept);
            }
            cache.totalLen = newLen;
            // 2) 压缩 KV / indexer key：截断行数并重建 rawTail
            if (!isKvSource[layer] || compress_ratios[layer] <= 0) {
                continue;
            }
            const int ratio = compress_ratios[layer];
            const int prevTail = scratch.prevRawTail[layer];
            const int prevBlocks = scratch.prevBlocks[layer];
            const int total = prevTail + accept;
            const int blocks = total / ratio;
            const int rem = total - blocks * ratio;
            cache.compressedBlocks = prevBlocks + blocks;
            for (Data *buffer : {&cache.compressedKV, &cache.indexK}) {
                if (buffer->dims.size() != 3 || buffer->dims[1] <= cache.compressedBlocks) {
                    continue;
                }
                const std::vector<int> dims = {buffer->dims[0], cache.compressedBlocks, buffer->dims[2]};
                buffer->Resize(dims);
                // CatDirect appends to each rank's local length on the next forward.
                // Roll back those lengths along with the root metadata.
                if (buffer->multiDeviceData) {
                    for (auto &replica : buffer->multiDeviceDatas) {
                        replica.second->Resize(dims);
                    }
                }
            }
            cache.rawTail = rem;
#ifdef USE_CUDA
            if (rem > 0 && scratch.commitKV) {
                // Full graph reserve already owns enough storage; the commit
                // kernel writes the prefix without changing its allocation.
                for (Data *tail : {&cache.rawTailKV, &cache.rawTailScore}) {
                    auto dims = tail->dims;
                    dims[1] = rem;
                    if (tail->dims != dims) tail->Resize(dims);
                    tail->tpGlobalDims = dims;
                    for (auto &local : tail->multiDeviceDatas)
                        if (local.second->dims != dims) local.second->Resize(dims);
                }
            }
#endif
            if (rem > 0 && !batchedTails) {
                AssertInFastLLM(scratch.rawKV[layer].dims.size() == 3 &&
                                scratch.rawKV[layer].dims[1] >= total,
                                "DeepSeekV41 DSpark: the compressor rollback buffer is too short.");
                auto copyTail = [&](Data &source, Data &tail) {
#ifdef USE_CUDA
                    // Keep rank-local cache storage: ordinary Split collapses
                    // replicas to root and the next reserve broadcasts them back.
                    if (MultiCudaCopyReplicatedRows(tail, source, total - rem, rem,
                                                     v41DsparkTpDevices)) return;
#endif
                    Split(source, 1, total - rem, total, tail);
                };
                copyTail(scratch.rawKV[layer], cache.rawTailKV);
                if (ratio > 1) copyTail(scratch.rawScore[layer], cache.rawTailScore);
            }
            AssertInFastLLM(cache.compressedBlocks == newLen / ratio,
                            "DeepSeekV41 DSpark: compressed cache rollback is inconsistent at layer " +
                            std::to_string(layer));
        }
#ifdef USE_CUDA
        // The successful batch only uses dedicated worker streams. Join their
        // completion events once; all cache writes still finish before return.
        // Fallback operators retain the existing full-device lifetime barrier.
        if (batchedWindows && batchedTails && MultiCudaSynchronizeDeviceCallbacks(v41DsparkTpDevices))
            commitDispatch.MarkDeviceWorkComplete();
#endif
        state.totalLen = newLen;
        if (!engram_layer_ids.empty() && (int)state.engramHistory.size() > newLen) {
            state.engramHistory.resize(newLen);
        }
    }

    // ==================== 草稿前向 ====================

    std::shared_ptr<DeepSeekV41DsparkState> DeepSeekV41Model::GetDsparkBatch(int batch, int device) {
        if (!v41DsparkBatches.count(batch)) {
            for (auto it = v41DsparkBatches.begin(); it != v41DsparkBatches.end();) {
                if (RetainDsparkBatch(it->first)) ++it;
                else it = v41DsparkBatches.erase(it);
            }
        }
        auto &slot = v41DsparkBatches[batch];
        if (!slot) slot = std::make_shared<DeepSeekV41DsparkState>();
        slot->layers.resize(v41DsparkLayers);
        for (auto &layer : slot->layers) {
            auto &window = layer.windowKV;
            window.dataType = DataType::BFLOAT16; window.UpdateUnitSize();
            window.Resize({batch, window_size, head_dim_full});
            window.ToDevice(DataDevice::CUDA, {device}, false); window.Allocate();
        }
        return slot;
    }

#ifdef USE_CUDA
    bool DeepSeekV41Model::DsparkProjectContext(Data &main, DeepSeekV41DsparkState &state, int batch,
                                                int rows, const int *positions) {
        if (!positions || state.layers.size() != 3 || head_dim_full != 512 || main.dataType != BFLOAT16 ||
            main.multiDeviceData || main.dataDevice != CUDA || !main.cudaData ||
            DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") || DsEnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") ||
            DsEnvFlag("FASTLLM_DSV41_LEGACY_ROTARY")) return false;
        const int device = FastllmCudaGetDevice(), tokens = batch * rows;
        Data *weights[3], *rings[3];
        const Data *norms[3], *projected[3];
        Data separate[3], joined;
        bool grouped = tokens >= 2 && tokens <= 8;
        for (int l = 0; l < 3; ++l) {
            const auto pre = "mtp." + std::to_string(l) + ".attn.";
            Data &w = weight[pre + "wkv.weight"], &n = weight[pre + "kv_norm.weight"];
            if (n.multiDeviceData || n.dataType != FLOAT32 || n.Count(0) != 512 ||
                w.multiDeviceData || w.dims != std::vector<int>{512, embed_dim}) return false;
            n.ToDevice(CUDA, {device}, true);
            rings[l] = &state.layers[l].windowKV;
            norms[l] = &n;
            weights[l] = &w;
            grouped = grouped && quantizedLinearNames.count(pre + "wkv.weight") &&
                FastllmCudaCanUseMarlinBFloat16FP8(w, tokens, embed_dim, 512);
        }
        if (grouped && embed_dim == 5120 && getCudaInfos()->cudaArch == 1030) {
            auto &descriptor = v41DsparkContextWeights[device];
            if (!descriptor.cudaData && !FastllmCudaGraphIsCapturingFast()) {
                const void *pointers[6];
                for (int i = 0; i < 3 && grouped; ++i) {
                    weights[i]->ToDevice(CUDA, {device}, true);
                    grouped = FastllmCudaPrepareMarlinBFloat16FP8(*weights[i], tokens, embed_dim, 512);
                    if (grouped) {
                        pointers[2 * i] = weights[i]->extraCudaData[4];
                        pointers[2 * i + 1] = weights[i]->extraCudaData[5];
                    }
                }
                if (grouped) {
                    descriptor.dataType = INT8; descriptor.UpdateUnitSize(); descriptor.Resize({int(sizeof(pointers))});
                    descriptor.ToDevice(CUDA, {device}, false); descriptor.Allocate(false);
                    FastllmCudaCopyFromHostToDevice(descriptor.cudaData, pointers, sizeof(pointers));
                }
            }
            grouped = descriptor.cudaData != nullptr;
            if (grouped) {
                joined.dataType = BFLOAT16; joined.UpdateUnitSize(); joined.Resize({3, tokens, 512});
                joined.ToDevice(CUDA, {device}, false); joined.Allocate(false);
                grouped = FastllmCudaDsparkContextProjection(main.cudaData, descriptor.cudaData, joined.cudaData, tokens);
                if (grouped) for (int i = 0; i < 3; ++i) {
                    separate[i].FakeFrom(joined, size_t(i) * tokens * 512 * 2);
                    separate[i].Resize({tokens, 512});
                    projected[i] = &separate[i];
                }
            }
        } else grouped = false;
        if (!grouped) for (int l = 0; l < 3; ++l) {
            LinearWithActivationQuant(main, "mtp." + std::to_string(l) + ".attn.wkv.weight", separate[l]);
            projected[l] = &separate[l];
        }
        return FastllmCudaDsparkContextStore(projected, norms, rings, false, batch, rows, positions,
            rms_norm_eps, qk_rope_head_dim, rope_base, rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow);
    }
#endif

    bool DeepSeekV41Model::DsparkAdvanceBatch(const std::vector<DeepSeekV41Segment> &segments) {
        v41DsparkContextReady.reset();
#ifdef USE_CUDA
        if (segments.empty() || !v41DsparkTpDevices.empty() || !DsDraftGraphEnabled()) return false;
        const int batch = segments.size(), rows = segments.front().seqlen;
        if (rows <= 0 || uint64_t(batch) * rows > INT_MAX) return false;
        const auto *first = segments.front().spec;
        if (!first || !first->commitGroup || first->mainHidden.size() != v41DsparkTargetLayerIds.size() ||
            first->mainHidden.empty()) return false;
        const auto devices = first->mainHidden.front().dataDeviceIds;
        if (devices.size() != 1) return false;
        for (int b = 0; b < batch; ++b) {
            const auto &s = segments[b];
            const auto *x = s.spec;
            if (!x || x->commitGroup != first->commitGroup || x->mainHiddenStartPos >= 0 ||
                s.seqlen != rows || s.offset != b * rows || x->acceptedDrafts < 0 || x->acceptedDrafts >= rows ||
                !s.state || !s.state->dspark || s.state->dspark->disabled ||
                s.state->dspark->layers.size() != size_t(v41DsparkLayers) ||
                x->mainHidden.size() != first->mainHidden.size()) return false;
            for (size_t l = 0; l < x->mainHidden.size(); ++l) {
                const auto &h = x->mainHidden[l];
                const auto &base = first->mainHidden[l];
                if (h.dataDevice != CUDA || h.dataType != BFLOAT16 || !h.cudaData || h.multiDeviceData ||
                    h.dataDeviceIds != devices || h.dims.size() != 3 ||
                    h.dims[0] != 1 || h.dims[1] != rows || h.dims[2] != embed_dim ||
                    h.strides.size() != 3 || h.strides[1] != uint64_t(embed_dim) || h.strides[2] != 1 ||
                    (const char *)h.cudaData != (const char *)base.cudaData + size_t(b) * rows * embed_dim * 2)
                    return false;
            }
            for (const auto &layer : s.state->dspark->layers) {
                const auto &window = layer.windowKV;
                if (window.dataDevice != CUDA || window.dataType != BFLOAT16 || !window.cudaData ||
                    window.multiDeviceData || window.dataDeviceIds != devices || window.dims.size() != 3 ||
                    window.dims[0] != 1 || window.dims[1] != window_size || window.dims[2] != head_dim_full)
                    return false;
            }
        }
        std::unique_lock<std::mutex> lock(v41CudaGraphMutex, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        ApplyDsparkDevice(true);
        auto storage = GetDsparkBatch(batch, devices.front());
        // Only the regular verifier capacity and one transient context shape
        // are retained. Accepted lengths are metadata, not graph cache keys.
        if (!storage->contextWorkspaces.count(rows)) {
            for (auto it = storage->contextWorkspaces.begin(); it != storage->contextWorkspaces.end();) {
                if (it->first == v41DsparkTokens + 1) ++it;
                else it = storage->contextWorkspaces.erase(it);
            }
        }
        auto &owned = storage->contextWorkspaces[rows];
        if (!owned) { owned = std::make_shared<DeepSeekV41DsparkWorkspace>(); owned->devices = devices; }
        auto &context = *owned;
        std::vector<Data> views(first->mainHidden.size());
        std::vector<const Data *> inputs;
        inputs.reserve(views.size());
        for (size_t l = 0; l < views.size(); ++l) {
            views[l].FakeFrom(first->mainHidden[l], 0);
            views[l].Resize({1, batch * rows, embed_dim});
            inputs.push_back(&views[l]);
        }
        if (!FastllmCudaNaiveDraftConcat(inputs, batch * rows, context.input)) return false;
        std::vector<int> metadata(2 * batch);
        storage->contextRequests.clear();
        storage->contextRequests.reserve(batch);
        for (int b = 0; b < batch; ++b) {
            metadata[b] = segments[b].startPos;
            metadata[batch + b] = segments[b].spec->acceptedDrafts + 1;
            storage->contextRequests.push_back(segments[b].state->dspark.get());
        }
        auto &position = context.position;
        position.dataType = INT32; position.UpdateUnitSize(); position.Resize({2 * batch});
        position.ToDevice(CUDA, devices, false); position.Allocate();
        FastllmCudaCopyFromHostToDevice(position.cudaData, metadata.data(), metadata.size() * sizeof(int));
        const size_t bytes = size_t(window_size) * head_dim_full * 2;
        for (int l = 0; l < v41DsparkLayers; ++l) for (int b = 0; b < batch; ++b)
            FastllmCudaCopyFromDeviceToDevice((char *)storage->layers[l].windowKV.cudaData + b * bytes,
                storage->contextRequests[b]->layers[l].windowKV.cudaData, bytes);
        auto body = [&] {
            Data projected, main;
            LinearWithActivationQuant(context.input, "mtp.0.main_proj.weight", projected);
            DsRMSNormBF16(projected, weight["mtp.0.main_norm.weight"], rms_norm_eps, main);
            main.Reshape({1, batch * rows, embed_dim});
            DsRopeParams rope = {qk_rope_head_dim, rope_base, 0, rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow};
            const auto *positions = (const int *)position.cudaData;
            if (DsparkProjectContext(main, *storage, batch, rows, positions)) return;
            for (int l = 0; l < v41DsparkLayers; ++l) {
                const std::string prefix = "mtp." + std::to_string(l);
                Data kv;
                LinearWithActivationQuant(main, prefix + ".attn.wkv.weight", kv);
                DsRMSNormBF16(kv, weight[prefix + ".attn.kv_norm.weight"], rms_norm_eps, kv);
                kv.Reshape({batch, rows, head_dim_full});
                for (int b = 0; b < batch; ++b) {
                    Data view; view.FakeFrom(kv, size_t(b) * rows * head_dim_full * 2);
                    view.Resize({1, rows, head_dim_full});
                    DsRotaryQuant(view, rope, 0, 1, false, 1, 32, -1, positions + b);
                }
                AssertInFastLLM(FastllmCudaDeepSeekV41WindowStoreBatchPrefix(kv,
                    storage->layers[l].windowKV, window_size, positions, positions + batch),
                    "DeepSeekV41 DSpark batched context store failed.");
            }
        };
        if (!DsRunDraftGraph(context, body)) body();
        for (int b = 0; b < batch; ++b) {
            auto &state = *storage->contextRequests[b];
            if (state.committed != metadata[b]) state.filled = 0;
            state.ClearProposal();
            state.committed = metadata[b] + metadata[batch + b];
            state.filled = std::min(window_size, state.filled + metadata[batch + b]);
            for (int l = 0; l < v41DsparkLayers; ++l)
                FastllmCudaCopyFromDeviceToDevice(state.layers[l].windowKV.cudaData,
                    (char *)storage->layers[l].windowKV.cudaData + b * bytes, bytes);
        }
        // The immediately following draft may consume this same slot without
        // copying all request windows back in. Private copies preserve fallback
        // and regrouping correctness; q remains request-owned until verified.
        v41DsparkContextReady = storage;
        return true;
#else
        return false;
#endif
    }

    void DeepSeekV41Model::DsparkAdvance(DeepSeekV41RequestState &state, DeepSeekV41SpecScratch &scratch,
                                         int startPos, int committed) {
        if (scratch.mainHiddenStartPos >= 0) {
            startPos = scratch.mainHiddenStartPos;
        }
        auto dsparkPtr = GetOrCreateDsparkState(state, startPos);
        DeepSeekV41DsparkState &dspark = *dsparkPtr;
        dspark.ClearProposal();
        if (dspark.disabled) {
            return;
        }
        const int rows = committed - startPos;
#ifdef USE_CUDA
        if (scratch.deviceDecision && dspark.workspace && scratch.acceptedDrafts >= 0) {
            auto &w = *dspark.workspace;
            const bool full = rows == int(scratch.draftTokens.size()) + 1;
            w.fullStreak = full ? std::min(2,w.fullStreak + 1) : 0;
            w.reusableDraftAt = scratch.earlyDraft && full ? committed : -1;
            w.reusableDraftAnchor = scratch.tokens[scratch.acceptedDrafts];
            dspark.workspace->nextMetadataAt = committed;
            dspark.workspace->nextMetadataAnchor = scratch.tokens[scratch.acceptedDrafts];
        }
        if (scratch.earlyContext && rows == int(scratch.draftTokens.size()) + 1) {
            dspark.committed = committed;
            dspark.filled = std::min(dspark.filled + rows, window_size);
            return;
        }
#endif
        const bool packed = scratch.packedMainHidden.dims.size() == 3;
        if (rows <= 0 || (packed ? rows > scratch.packedMainHidden.dims[1] :
                scratch.mainHidden.size() != v41DsparkTargetLayerIds.size())) {
            dspark.disabled = true;
            return;
        }
        if (dspark.committed != startPos) {
            // 例如批量前向的某些轮没有采集 main hidden：滑窗与目标缓存已经脱节，
            // 从当前位置重新开始累积（filled 归零，只影响接受率）。
            dspark.committed = startPos;
            dspark.filled = 0;
        }

        const double advanceStart = DsTick();
        // Full verification joins every rank before sampling, and prefix
        // commit joins its worker streams before returning. The target's
        // hidden states are ready; do not also wait for cache admissions.
        ApplyDsparkDevice(scratch.targetRowsReady || bool(scratch.commitKV));

        // 1) main_x = main_norm(main_proj(cat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39))))
        Data combined, tmp;
        std::vector<const Data *> inputs;
        inputs.reserve(scratch.mainHidden.size());
        for (size_t k = 0; k < scratch.mainHidden.size(); k++) {
            const Data *hidden = &scratch.mainHidden[k];
#ifdef USE_CUDA
            if (hidden->multiDeviceData) {
                // Replicated tensors only carry shape metadata at the root.
                // The draft runs on the first rank, so consume that replica
                // directly instead of staging it through host memory.
                auto it = hidden->multiDeviceDatas.find(v41DsparkTpDevices.front());
                AssertInFastLLM(it != hidden->multiDeviceDatas.end() && it->second != nullptr &&
                                it->second->cudaData != nullptr,
                                "DeepSeekV41 DSpark: cannot read the target TP hidden state.");
                hidden = it->second;
            }
#endif
            inputs.push_back(hidden);
        }
        DeepSeekV41DsparkWorkspace *contextState = nullptr;
#ifdef USE_CUDA
        std::unique_lock<std::mutex> contextLock(v41CudaGraphMutex, std::defer_lock);
        std::vector<void*> caches;
        bool cacheReady = true;
        for (auto &layer : dspark.layers) {
            caches.push_back(layer.windowKV.cudaData);
            cacheReady &= layer.windowKV.dataDevice == DataDevice::CUDA && layer.windowKV.cudaData;
        }
        if (DsDraftGraphEnabled() && rows <= v41DsparkBlockSize + 1 && cacheReady &&
            (packed || (!inputs.empty() && inputs.front()->dataDevice == DataDevice::CUDA)) && contextLock.try_lock()) {
            auto &slot = dspark.contextWorkspaces[rows];
            const std::vector<int> devices{v41DsparkTpDevices.empty() ? FastllmCudaGetDevice() : v41DsparkTpDevices.front()};
            if (slot && (slot->devices != devices || slot->cachePointers != caches)) slot.reset();
            if (!slot) {
                slot = std::make_shared<DeepSeekV41DsparkWorkspace>();
                slot->devices = devices; slot->cachePointers = caches;
            }
            if (!slot->disabled) {
                contextState = slot.get();
                Data &position = contextState->position;
                if (position.dims.empty()) {
                    position.dataType = DataType::INT32; position.UpdateUnitSize(); position.Resize({2});
                    position.ToDevice(DataDevice::CUDA, devices, false); position.Allocate();
                }
                int metadata[] = {startPos, rows};
                FastllmCudaCopyFromHostToDevice(position.cudaData, metadata, sizeof(metadata));
            }
        }
#endif
#ifdef USE_CUDA
        // Keep the input address captured by the context graph stable across requests.
        // Gathering directly also avoids a second device copy of all accepted rows.
        bool joinedOnDevice = packed;
        if (packed) {
            Data &source = scratch.packedMainHidden;
            const int width = source.dims.back();
            if (contextState) {
                Data &input = contextState->input;
                input.dataType = BFLOAT16; input.UpdateUnitSize(); input.Resize({1, rows, width});
                input.ToDevice(CUDA, source.dataDeviceIds, false); input.Allocate(false);
                const size_t bytes = size_t(rows) * width * sizeof(uint16_t);
                AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(
                    input.cudaData, bytes, source.cudaData, bytes, bytes, 1), "draft graph input copy failed");
            } else {
                combined.FakeFrom(source, 0); combined.Resize({1, rows, width});
            }
        } else {
            joinedOnDevice = FastllmCudaNaiveDraftConcat(inputs, rows, contextState ? contextState->input : combined);
        }
#else
        const bool joinedOnDevice = false;
#endif
        if (!joinedOnDevice) for (size_t k = 0; k < inputs.size(); ++k) {
            const Data *hidden = inputs[k];
            Data part;
            if (rows == hidden->dims[1]) {
                part.CopyFrom(*hidden);
            } else {
                Split(*hidden, 1, 0, rows, part);
            }
            if (k == 0) {
                combined.CopyFrom(part);
            } else {
                Cat(combined, part, -1, tmp);
                combined.CopyFrom(tmp);
            }
        }
#ifdef USE_CUDA
        if (contextState && !joinedOnDevice) contextState->input.CopyFrom(combined);
#endif
        auto contextBody = [&]() {
            Data &contextInput = contextState ? contextState->input : combined;
            const int *devicePosition = contextState ? (const int*)contextState->position.cudaData : nullptr;
            const std::string stage0 = "mtp.0";
            Data mainProj, mainX;
            // Decoding FP8 weights to the startup dtype does not remove the
            // checkpoint's block-32 activation quantization requirement.
            LinearWithActivationQuant(contextInput, stage0 + ".main_proj.weight", mainProj);
            DsRMSNormBF16(mainProj, weight[stage0 + ".main_norm.weight"], rms_norm_eps, mainX);
            mainX.Reshape({1, rows, embed_dim});

#ifdef USE_CUDA
            if (DsparkProjectContext(mainX, dspark, 1, rows, devicePosition)) return;
#endif
            // 2) 每个草稿层用自己的 wkv / kv_norm 把 main_x 写进滑窗环形缓冲
            DsRopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                       rope_scaling_beta_fast, rope_scaling_beta_slow};
            for (int stage = 0; stage < v41DsparkLayers; stage++) {
                const std::string pre = "mtp." + std::to_string(stage);
                Data kv;
                LinearWithActivationQuant(mainX, pre + ".attn.wkv.weight", kv);
                DsRMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
                kv.Reshape({1, rows, head_dim_full});
                DsRotaryQuant(kv, windowRope, startPos, 1, false, 1, 32, -1, devicePosition);
#ifdef USE_CUDA
                if (devicePosition) {
                    AssertInFastLLM(FastllmCudaDeepSeekV41WindowStoreBatchPrefix(kv, dspark.layers[stage].windowKV,
                        window_size, devicePosition, devicePosition + 1), "DeepSeekV41 DSpark dynamic window store failed.");
                } else
#endif
                DsWindowStore(kv, dspark.layers[stage].windowKV, startPos, window_size);
            }
        };
#ifdef USE_CUDA
        if (!contextState || !DsRunDraftGraph(*contextState, contextBody, scratch.targetRowsReady || bool(scratch.commitKV))) contextBody();
        if (contextLock.owns_lock()) contextLock.unlock();
#else
        contextBody();
#endif
        dspark.committed = committed;
        dspark.filled = std::min(dspark.filled + rows, window_size);
        const double mainMs = DsElapsed(advanceStart);

        if (DsProfiler().On()) DsProfiler().AddContext(mainMs);
    }

    void DeepSeekV41Model::DsparkGenerateDrafts(const std::vector<DeepSeekV41DsparkState*> &states,
            const std::vector<int> &anchors, const std::vector<GenerationConfig> &configs) {
        auto contextReady = std::move(v41DsparkContextReady);
        AssertInFastLLM(states.size() == anchors.size() && states.size() == configs.size(),
                       "DeepSeekV41 DSpark: mismatched draft requests.");
        // Constraints maintain per-request text state; the existing sampler handles those.
        auto compatible = [&](int a, int b) {
            const auto &x = configs[a]; const auto &y = configs[b];
            return !x.tool_call_name_constraint_enabled && !x.tool_call_parameter_name_constraint_enabled &&
                !y.tool_call_name_constraint_enabled && !y.tool_call_parameter_name_constraint_enabled &&
                x.top_k == y.top_k && x.top_p == y.top_p && x.temperature == y.temperature;
        };
        std::vector<std::vector<int>> groups;
        for (int i = 0; i < int(states.size()); ++i) {
            if (!states[i] || states[i]->disabled || anchors[i] < 0 ||
                std::min({states[i]->filled, window_size, states[i]->committed}) <= 0) continue;
            auto group = std::find_if(groups.begin(), groups.end(), [&](const auto &g) { return compatible(g.front(), i); });
            if (group == groups.end()) groups.push_back({i}); else group->push_back(i);
        }
        auto publish = [&](int request, const std::vector<int> &tokens, const std::vector<float> &confidence) {
            if (tokens.empty()) return;
            auto &state = *states[request];
            state.drafts = state.proposalTokens = tokens;
            state.confidence = confidence;
            state.anchor = anchors[request]; state.anchorPos = state.committed;
        };
        for (const auto &group : groups) {
            DsDraftTimingSlot().Reset();
            const double begin = DsTick();
            bool completed = false;
#ifdef USE_CUDA
            if (v41DsparkTpDevices.empty() && DsDraftGraphEnabled() && compatible(group.front(), group.front())) {
                ApplyDsparkDevice();
                const int batch = int(group.size()), device = FastllmCudaGetDevice();
                const std::vector<int> devices{device};
                bool local = true;
                for (int request : group) {
                    local &= int(states[request]->layers.size()) == v41DsparkLayers;
                    for (const auto &layer : states[request]->layers)
                        local &= layer.windowKV.dataType == DataType::BFLOAT16 &&
                            layer.windowKV.dataDevice == DataDevice::CUDA && layer.windowKV.cudaData &&
                            layer.windowKV.dataDeviceIds == devices && !layer.windowKV.multiDeviceData;
                }
                if (local) {
                    auto slot = GetDsparkBatch(batch, device);
                    for (int request : group) states[request]->workspace.reset();
                    auto &storage = *slot;
                    bool reuseContext = contextReady == slot && storage.contextRequests.size() == group.size();
                    for (size_t b = 0; reuseContext && b < group.size(); ++b)
                        reuseContext = storage.contextRequests[b] == states[group[b]];
                    if (!reuseContext) for (int layer = 0; layer < v41DsparkLayers; ++layer) {
                        auto &window = storage.layers[layer].windowKV;
                        const size_t bytes = size_t(window_size) * head_dim_full * window.unitSize;
                        for (int b = 0; b < batch; ++b)
                            FastllmCudaCopyFromDeviceToDevice((char*)window.cudaData + b * bytes,
                                states[group[b]]->layers[layer].windowKV.cudaData, bytes);
                    }
                    std::vector<int> positions, batchAnchors, valid;
                    for (int request : group) {
                        positions.push_back(states[request]->committed);
                        batchAnchors.push_back(anchors[request]);
                        valid.push_back(std::min({states[request]->filled, window_size, states[request]->committed}));
                    }
                    std::vector<int> tokens;
                    std::vector<float> confidence;
                    completed = DsparkRunDraft(storage, positions, batchAnchors, valid,
                                               tokens, confidence, configs[group.front()]);
                    if (completed) {
                        const int block = v41DsparkBlockSize;
                        AssertInFastLLM(tokens.size() == size_t(batch) * block && confidence.size() == tokens.size(),
                                       "DeepSeekV41 DSpark: invalid batched proposal shape.");
                        for (int b = 0; b < batch; ++b) {
                            auto &state = *states[group[b]];
                            state.sampledProposal = storage.sampledProposal;
                            if (state.sampledProposal) {
                                // Each request owns q until its next verification, even if a
                                // different batch reuses this graph or the request changes slots.
                                auto &prob = state.proposalProbs;
                                prob.dataType = DataType::FLOAT32; prob.UpdateUnitSize();
                                prob.Resize({block, storage.proposalProbs.dims.back()});
                                prob.ToDevice(DataDevice::CUDA, devices, false); prob.Allocate();
                                FastllmCudaCopyFromDeviceToDevice(prob.cudaData,
                                    (char*)storage.proposalProbs.cudaData + b * prob.GetBytes(), prob.GetBytes());
                            }
                            publish(group[b], std::vector<int>(tokens.begin() + b * block, tokens.begin() + (b + 1) * block),
                                std::vector<float>(confidence.begin() + b * block, confidence.begin() + (b + 1) * block));
                        }
                    }
                }
            }
#endif
            if (completed) {
                if (DsProfiler().On()) DsProfiler().AddDraft(DsElapsed(begin), DsDraftTimingSlot());
                continue;
            }
            for (int request : group) {
                DsDraftTimingSlot().Reset();
                const double requestBegin = DsTick();
                auto &state = *states[request];
                const int valid = std::min({state.filled, window_size, state.committed});
                std::vector<int> tokens;
                std::vector<float> confidence;
                DsparkRunDraft(state, {state.committed}, {anchors[request]}, {valid},
                               tokens, confidence, configs[request], states.size() == 1);
                publish(request, tokens, confidence);
                if (DsProfiler().On()) DsProfiler().AddDraft(DsElapsed(requestBegin), DsDraftTimingSlot());
            }
        }
    }

    bool DeepSeekV41Model::DsparkRunDraft(DeepSeekV41DsparkState &dspark,
            const std::vector<int> &positions, const std::vector<int> &anchors,
            const std::vector<int> &validRows, std::vector<int> &tokens,
            std::vector<float> &confidence, const GenerationConfig &config, bool deferGreedy) {
        tokens.clear();
        confidence.clear();
        const int block = v41DsparkBlockSize;
        // Kernels use signed row/token indices. This is an indexing-capacity
        // guard, independent of concurrency, draft length and physical tiles.
        if (block <= 0 || block > INT_MAX - window_size || positions.empty() || anchors.size() != positions.size() ||
            validRows.size() != positions.size() ||
            positions.size() > size_t(INT_MAX) / (uint64_t(block) + 1) ||
            std::any_of(validRows.begin(), validRows.end(), [](int n) { return n <= 0; })) return false;
        const int batch = int(positions.size()), rows = batch * block;
        int anchorToken = anchors.front();
        const int startPos = positions.front();    // 草稿的 RoPE 位置从 committed 开始
        const int valid = validRows.front();
        DsparkBuildMoeWeights();
#ifdef USE_CUDA
        auto cached = dspark.workspace;
        if (deferGreedy && batch == 1 && validRows.front() == window_size && cached &&
            cached->reusableDraftAt == startPos && cached->reusableDraftAnchor == anchorToken) {
            // Only this invocation can consume the preceding verified proposal.
            // Its root stream already joins the context and all draft producers.
            ApplyDsparkDevice(true);
            cached->reusableDraftAt = cached->nextMetadataAt = -1;
            if (cached->QueueProposal(block)) {
                dspark.anchor = anchorToken; dspark.anchorPos = startPos;
                return true;
            }
        }
#endif
#ifdef USE_CUDA
        if (dspark.workspace) {
            auto &w = *dspark.workspace;
            if (w.pendingCount && w.mirrorStream) {
                FastllmCudaSetDevice(w.devices.front());
                FastllmCudaCurrentThreadStreamWaitEvent(w.proposalReady);
            }
            w.rootWinnerRecorded = false;
        }
#endif
        ApplyDsparkDevice();
        const int dim = embed_dim;
        const int headDim = head_dim_full;
        const float softmaxScale = 1.0f / std::sqrt((float)headDim);
        DsRopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                   rope_scaling_beta_fast, rope_scaling_beta_slow};

        const int tableRows = window_size + block;
        const int compactWidth = (batch == 1 ? valid : window_size) + block - 1;
        int width = compactWidth;
        Data embedded, preMix, cmpIdxHost, cmpIdx;
        bool inputsPrepared = false;
        auto prepareInputs = [&]() {
            if (inputsPrepared) return;
            inputsPrepared = true;
            // ---- 输入：[anchor, noise, noise, ...] ----
            std::vector<float> idValues((uint64_t)rows, (float)v41DsparkNoiseTokenId);
            for (int b = 0; b < batch; ++b) idValues[b * block] = float(anchors[b]);
            Data draftIds(DataType::FLOAT32, {batch, block}, idValues);
            DsparkEmbedInput(draftIds, embedded);
            embedded.Reshape({batch, block, hc_mult, dim});

            {
                std::vector<float> values((uint64_t)rows * hc_mult, 0.0f);
                for (int i = 0; i < rows; i++) {
                    values[(uint64_t)i * hc_mult] = 1.0f;
                }
                preMix.CopyFrom(Data(DataType::FLOAT32, {batch, block, hc_mult}, values));
            }

            // ---- 稀疏注意力的 gather 表下标 ----
            // 现成的 DeepSeekV41SparseAttention 在片段内是因果的，而 DSpark 的草稿块内部
            // 是全连接（官方 topk_idxs 对每个 query 给出同一张表）。这里把 windowSize 设为 1、
            // startPos 设为 0，让"滑窗"部分只贡献 query 自己那一行，其余位置（环形缓冲里
            // 有效的行 + 块内其它草稿行）通过 compressedKV / cmpIdx 的任意 gather 提供。
            std::vector<int32_t> idxValues((uint64_t)rows * width, -1);
            for (int b = 0; b < batch; ++b) for (int i = 0; i < block; i++) {
                int32_t *row = idxValues.data() + (uint64_t)(b * block + i) * width;
                int cursor = 0;
                for (int p = positions[b] - validRows[b]; p < positions[b]; p++) {
                    row[cursor++] = p % window_size;
                }
                for (int j = 0; j < block; j++) {
                    if (j != i) {
                        row[cursor++] = window_size + j;
                    }
                }
            }
            cmpIdxHost.dataType = DataType::INT32;
            cmpIdxHost.Resize({batch, block, width});
            cmpIdxHost.Allocate();
            memcpy(cmpIdxHost.cpuData, idxValues.data(), idxValues.size() * sizeof(int32_t));
            cmpIdx.CopyFrom(cmpIdxHost);
        };

        const std::string last = "mtp." + std::to_string(v41DsparkLayers - 1);
        const auto confidenceIt = weight.weight.find(last + ".confidence_head.proj.weight");
        Data *confidenceWeight = v41DsparkConfidenceThreshold > 0.0f && confidenceIt != weight.weight.end()
            ? &confidenceIt->second : nullptr;
        const bool randomDraft = config.top_k > 1;
        Data &markovEmbedWeight = weight[last + ".markov_head.embed.weight"];
        Data &markovHeadWeight = weight[last + ".markov_head.head.weight"];
        Data eagerHeadHidden, eagerNormed, eagerMarkov;
        DeepSeekV41DsparkWorkspace *graphState = nullptr;
        bool residentDraft = false, hybridDraft = false, diskDraft = false;
#ifdef USE_CUDA
        // Device-resident experts can capture on one GPU as well as TP.
        // Partial windows use uploaded indices keyed by their gather width.
        std::unique_lock<std::mutex> graphLock(v41CudaGraphMutex, std::defer_lock);
        const std::vector<int> graphDevices = v41DsparkTpDevices.empty()
            ? std::vector<int>{FastllmCudaGetDevice()} : v41DsparkTpDevices;
        // A forced resident TP draft keeps every routed expert on its root.
        // Capture those local kernels with the existing TP attention/head;
        // they need neither CPU mailboxes nor the packet-output restriction.
        residentDraft = v41DsparkTpDevices.empty() || DsEnvFlag("FASTLLM_DSPARK_GPU_RESIDENT");
        if (residentDraft) for (auto &table : v41DsparkMoeWeights)
            residentDraft &= FastllmCudaMoeCacheFullyResident(table.data(), int(table.size()));
        // The draft backbone keeps its host experts on the root even with TP
        // attention/head. EP keeps its persistent packet path; native children
        // must all belong to that same root cache owner.
        if (!residentDraft && DsDraftGraphEnabled() && GetCudaSharedExpert()) {
            void *owner = nullptr;
            hybridDraft = !v41DsparkMoeWeights.empty();
            for (auto &table : v41DsparkMoeWeights) {
                void *next = FastllmCudaNativeMoeGraphOwner(table.data(), int(table.size()), rows);
                hybridDraft &= next && (!owner || owner == next);
                owner = next;
            }
        }
        // Native disk drafts own a serial mailbox worker independently of
        // target verification. Add every child during eager warmup, then start
        // that worker only when the complete draft graph is replayed.
        if (!residentDraft && !hybridDraft &&
                rows <= 32 && DsDraftGraphEnabled() && GetCudaSharedExpert()) {
            diskDraft = !v41DsparkMoeWeights.empty();
            for (const auto &table : v41DsparkMoeWeights)
                diskDraft &= DiskMoeV41Graph::Supports(table);
        }
        const int vocab = weight["head.weight"].dims[0];
        const bool fusedMarkovAllowed = !config.tool_call_name_constraint_enabled &&
            !config.tool_call_parameter_name_constraint_enabled && !DsEnvFlag("FASTLLM_DSPARK_DISABLE_FUSED_MARKOV");
        auto denseMarkovWeight = [](const Data &w) {
            return !w.multiDeviceData && w.dims.size() == 2 && w.strides.size() == 2 &&
                w.strides[0] == w.dims[1] && w.strides[1] == 1;
        };
        // Device sampling retains the filtered conditional q for rejection.
        const bool sampledMarkov = randomDraft && std::isfinite(config.top_p) &&
            config.top_p > 0 && config.top_p <= 1 &&
            std::isfinite(config.temperature) && config.temperature > 0.0f &&
            denseMarkovWeight(markovEmbedWeight) && denseMarkovWeight(markovHeadWeight);
        auto prepareProposalProbs = [&](int device, int vocabulary) {
            FastllmCudaSetDevice(device);
            dspark.proposalProbs.ToDevice(DataDevice::CUDA, {device}, false);
            dspark.proposalProbs.Resize({rows, vocabulary});
            dspark.proposalProbs.Allocate();
        };
        const bool graphMarkov = markovEmbedWeight.dims.size() == 2 && markovHeadWeight.dims.size() == 2 &&
            markovEmbedWeight.dims[1] > 0 && markovEmbedWeight.dims[1] <= 512 &&
            markovHeadWeight.dims[1] == markovEmbedWeight.dims[1] &&
            markovEmbedWeight.dims[0] >= vocab && markovHeadWeight.dims[0] >= vocab &&
            anchorToken >= 0 && anchorToken < markovEmbedWeight.dims[0] &&
            (markovEmbedWeight.dataType == DataType::BFLOAT16 ||
             markovEmbedWeight.dataType == DataType::FLOAT16 || markovEmbedWeight.dataType == DataType::FLOAT32) &&
            (markovHeadWeight.dataType == DataType::BFLOAT16 ||
             markovHeadWeight.dataType == DataType::FLOAT16 || markovHeadWeight.dataType == DataType::FLOAT32);
        const bool graphEligible = graphMarkov && DsDraftGraphEnabled() &&
            fusedMarkovAllowed && (!randomDraft || sampledMarkov) &&
            (residentDraft || hybridDraft || diskDraft || v41DsparkTpDevices.size() > 1) &&
            !DsEnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE") && !DsEnvFlag("FASTLLM_DISABLE_NCCL");
        // On a short prompt, the growing history previously invalidated the
        // graph at each acceptance. Keep indices in compact order and mask the
        // unused suffix with -1 as in the regular multi-request gather.
        const bool fixedPartialHistory = graphEligible && batch == 1 && valid < window_size &&
            !randomDraft && v41DsparkTpDevices.size() > 1 && window_size == 128 && block == 5 &&
            FastllmCudaRuntimeArch() == 103 && !DsEnvFlag("FASTLLM_DSV41_LEGACY_ATTN") &&
            !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH");
        if (fixedPartialHistory) {
            // The self row occupies one lane in the 64-candidate MMA tile.
            // Pad only that tile, preserving its partial sums and tile count.
            const int tiledWidth = ((compactWidth + 1 + 63) / 64) * 64 - 1;
            width = std::min(tiledWidth, window_size + block - 1);
        }
        if (graphEligible && randomDraft) prepareProposalProbs(FastllmCudaGetDevice(), vocab);
        const Data &embedding = weight["embed.weight"];
        const bool deviceInputs = batch == 1 && graphEligible && valid == window_size && embedding.dataType == DataType::BFLOAT16 &&
            embedding.dataDevice == DataDevice::CUDA && embedding.cudaData && !embedding.multiDeviceData &&
            !embedding.dataDeviceIds.empty() && embedding.dataDeviceIds.front() == graphDevices.front() &&
            embedding.dims.size() == 2 && embedding.dims[1] == dim &&
            embedding.strides.size() == 2 && embedding.strides[0] == dim && embedding.strides[1] == 1 &&
            anchorToken >= 0 && anchorToken < embedding.dims[0] &&
            v41DsparkNoiseTokenId >= 0 && v41DsparkNoiseTokenId < embedding.dims[0];
        if (!deviceInputs) {
            prepareInputs();
            // Low-memory placement keeps embedding weights on the CPU. Only
            // upload this small draft activation before capture/replay; the
            // backbone and Markov head can still use the same complete graph.
            if (graphEligible && embedded.dataDevice == DataDevice::CPU)
                embedded.ToDevice(DataDevice::CUDA, std::vector<int>{graphDevices.front()});
        }
        std::vector<void*> cachePointers;
        for (auto &layer : dspark.layers) cachePointers.push_back(layer.windowKV.cudaData);
        if (deviceInputs) cachePointers.push_back(embedding.cudaData);
        if (sampledMarkov) cachePointers.push_back(dspark.proposalProbs.cudaData);
        if (graphEligible && (deviceInputs || embedded.dataDevice == DataDevice::CUDA) && graphLock.try_lock()) {
            if (dspark.workspace && (dspark.workspace->devices != graphDevices ||
                dspark.workspace->block != block || dspark.workspace->width != width ||
                dspark.workspace->deviceInputs != deviceInputs ||
                dspark.workspace->sampledMarkov != sampledMarkov ||
                (sampledMarkov && (dspark.workspace->temperature != config.temperature ||
                    dspark.workspace->topK != config.top_k || dspark.workspace->topP != config.top_p)) ||
                dspark.workspace->cachePointers != cachePointers)) dspark.workspace.reset();
            if (!dspark.workspace) {
                dspark.workspace = std::make_shared<DeepSeekV41DsparkWorkspace>();
                auto &w = *dspark.workspace;
                w.devices = graphDevices; w.block = block; w.width = width;
                w.hybridExperts = hybridDraft;
                if (diskDraft) w.diskExperts = std::make_unique<DiskMoeV41Graph>();
                if (hybridDraft || diskDraft) w.parallelExperts = false;
                w.deviceInputs = deviceInputs;
                w.sampledMarkov = sampledMarkov; w.temperature = config.temperature;
                w.topK = config.top_k; w.topP = config.top_p;
                w.cachePointers = cachePointers;
            }
            if (!dspark.workspace->disabled) {
                graphState = dspark.workspace.get();
                graphState->sharedExperts.resize(v41DsparkLayers);
                if (deviceInputs && dim == 5120 && rows == 5 && getCudaInfos()->cudaArch == 1030) {
                    auto &branch = graphState->hcBranch;
                    if (!branch.stream) branch.stream = FastllmCudaStreamCreate();
                    if (!branch.begin) branch.begin = FastllmCudaEventCreate();
                    if (!branch.done) branch.done = FastllmCudaEventCreate();
                }
                auto upload = [&](Data &dst, DataType type, const std::vector<int> &dims, const void *data) {
                    if (dst.dims.empty()) {
                        dst.dataType = type; dst.UpdateUnitSize(); dst.Resize(dims);
                        dst.ToDevice(DataDevice::CUDA, {graphDevices.front()}, false); dst.Allocate();
                    }
                    FastllmCudaCopyFromHostToDevice(dst.cudaData, const_cast<void*>(data), dst.GetBytes());
                };
                const int metadata[2] = {startPos, anchorToken};
                graphState->reusableDraftAt = -1;
                if (deviceInputs && graphState->position.cudaData && graphState->nextMetadataAt == startPos &&
                    graphState->nextMetadataAnchor == anchorToken && graphState->verifyDecision.cudaData) {
                    AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(graphState->position.cudaData,
                        static_cast<const int *>(graphState->verifyDecision.cudaData) + 4, sizeof(metadata)),
                        "DSpark device anchor handoff failed");
                } else {
                    upload(graphState->position, DataType::INT32, {deviceInputs ? 2 : batch},
                           deviceInputs ? metadata : positions.data());
                }
                graphState->nextMetadataAt = -1;
                if (!deviceInputs) {
                    graphState->input.CopyFrom(embedded);
                    upload(graphState->preMix, DataType::FLOAT32, preMix.dims, preMix.cpuData);
                    upload(graphState->indices, DataType::INT32, cmpIdxHost.dims, cmpIdxHost.cpuData);
                    if (graphState->markovTokens.dims.empty()) {
                        Data &d = graphState->markovTokens;
                        d.dataType = DataType::INT32; d.UpdateUnitSize(); d.Resize({batch * (block + 1)});
                        d.ToDevice(DataDevice::CUDA, {graphDevices.front()}, false); d.Allocate();
                    }
                    std::vector<int> initialTokens(batch * (block + 1), 0);
                    for (int b = 0; b < batch; ++b) initialTokens[b * (block + 1)] = anchors[b];
                    FastllmCudaCopyFromHostToDevice(graphState->markovTokens.cudaData, initialTokens.data(),
                                                   initialTokens.size() * sizeof(int));
                }
                markovEmbedWeight.ToDevice(DataDevice::CUDA);
                markovHeadWeight.ToDevice(DataDevice::CUDA);
            }
        }
#endif
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
        FastllmCudaMoeRouteScope draftRouteScope(FastllmMoeRoutePhase::Draft);
        HybridMoeDecodeStep draftStep;
        // Eager and captured drafts use the same frequency/staging policy.
        // Keep the scope through all draft layers and close it after sampling.
        if (!v41DsparkMoeWeights.empty() &&
                (v41DsparkTpDevices.empty() || (graphState && graphState->hybridExperts))) {
            auto &table = v41DsparkMoeWeights.front();
            draftStep.Begin(table.data(), int(table.size()), v41DsparkTopk);
        }
        if (graphState && graphState->hybridExperts && !draftStep) {
            graphState->disabled = true; graphState = nullptr;
        }
#endif
#ifdef USE_CUDA
        if (!graphState && fixedPartialHistory) {
            width = compactWidth;
            inputsPrepared = false;
        }
#endif
        if (batch > 1 && !graphState) return false;
        if (!graphState) prepareInputs();
        Data eagerLogits;
        Data &logits = graphState ? graphState->logits : eagerLogits;
        Data &headHidden = graphState ? graphState->headHidden : eagerHeadHidden;
        Data &normed = graphState ? graphState->normalized : eagerNormed;
        Data &markovAll = graphState ? graphState->markovEmbeds : eagerMarkov;
        auto projectConfidence = [&](Data &output) {
            Data hiddenFloat, markovFloat, features;
            ToDataType(headHidden, hiddenFloat, DataType::FLOAT32);
            ToDataType(markovAll, markovFloat, DataType::FLOAT32);
            Cat(hiddenFloat, markovFloat, -1, features);
            Linear(features, *confidenceWeight, Data(), output);
            ToDataType(output, DataType::FLOAT32);
        };
        auto backbone = [&]() {
#ifdef USE_CUDA
            if (graphState && graphState->deviceInputs) {
                AssertInFastLLM(FastllmCudaDeepSeekV41DraftInput(weight["embed.weight"], graphState->position,
                    v41DsparkNoiseTokenId, block, hc_mult, window_size, graphState->input,
                    graphState->preMix, graphState->indices, graphState->markovTokens),
                    "DeepSeekV41 DSpark device input preparation rejected input.");
            }
#endif
            Data hiddenStates, hiddenTemp, layerPreMix;
            hiddenStates.CopyFrom(graphState ? graphState->input : embedded);
            layerPreMix.CopyFrom(graphState ? graphState->preMix : preMix);
            Data *curHidden = &hiddenStates;
            Data *nextHidden = &hiddenTemp;
            const int *devicePosition = graphState ? (const int*)graphState->position.cudaData : nullptr;
            Data &attentionIndices = graphState ? graphState->indices : cmpIdx;
            Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
            Data x, attnInput, attnQuant, qr, qNorm, qQuant, q, kv, table, attnOut, woAOut, attnProj;
            Data ffnInput, ffnOut, expertIndex, expertScore;
            Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;

            const double layersStart = DsTick();
            for (int stage = 0; stage < v41DsparkLayers; stage++) {
                const std::string pre = "mtp." + std::to_string(stage);
                const int layerId = std::max(0, block_cnt - v41DsparkLayers + stage);
                ApplyDsparkDevice();

                // Keep the original BF16 normalization and block-32 FP8 rounding,
                // but produce QKV's quantized input in the normalization kernel.
                const bool quantizeInput = !DsEnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT") &&
                    quantizedLinearNames.count(pre + ".attn.wq_a.weight") &&
                    quantizedLinearNames.count(pre + ".attn.wkv.weight") &&
                    quantizedLinearNames.count(pre + ".attn.wq_b.weight");
                bool forked = false;
#ifdef USE_CUDA
                if (graphState && graphState->hcBranch.stream && graphState->hcBranch.begin &&
                    graphState->hcBranch.done && quantizeInput && DsFusedHcEnabled()) {
                    forked = FastllmCudaDsparkHcFork(*curHidden, layerPreMix, weight[pre + ".hc_attn_fn"],
                        weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                        weight[pre + ".attn_norm.weight"], hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                        attnPre, attnPost, attnComb, attnInput, &attnQuant, graphState->hcBranch);
                }
#endif
                const bool inputQuantized = forked || DsHcMixPreNorm(*curHidden, layerPreMix, weight[pre + ".hc_attn_fn"],
                    weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                    weight[pre + ".attn_norm.weight"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                    attnPre, attnPost, attnComb, x, attnInput, quantizeInput ? &attnQuant : nullptr);

                const bool qkvPrepared = DsparkPrepareQkv(stage, attnInput, attnQuant, qr, qNorm, qQuant, kv,
                    inputQuantized, graphState ? &graphState->packedQkvWeights[stage] : nullptr);
                if (!qkvPrepared) {
                    LinearWithActivationQuant(attnInput, pre + ".attn.wq_a.weight", qr);
                    DsRMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
                }
                bool parallelAttention = false;
#ifdef USE_CUDA
                const bool ownerAttention = qkvPrepared && block == 5 && width == 132 &&
                    num_attention_heads == 64 && FastllmCudaRuntimeArch() == 103 &&
                    !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && !DsEnvFlag("FASTLLM_DSV41_LEGACY_ATTN");
                if (!ownerAttention && graphState && graphState->devices.size() > 1 && qkvPrepared && graphState->deviceInputs) {
                    parallelAttention = DsParallelAttention(
                        graphState->devices, qQuant, kv, dspark.layers[stage].windowKV, attentionIndices,
                        graphState->position, weight[pre + ".attn.wq_b.weight"],
                        weight[pre + ".attn.wo_a.weight"], weight[pre + ".attn.wo_b.weight"],
                        weight[pre + ".attn.attn_sink"], num_attention_heads, o_groups, o_lora_rank,
                        windowRope, startPos, softmaxScale, v41DsparkAttentionWeights[stage],
                        graphState->attentionTp[stage], attnProj);
                    graphState->parallelAttention |= parallelAttention;
                }
#endif
                if (!parallelAttention) {
                    LinearWithActivationQuant(qkvPrepared ? qQuant : qNorm, pre + ".attn.wq_b.weight", q,
                                              false, nullptr, qkvPrepared);
                    q.Reshape({batch, block, num_attention_heads, headDim});
                    if (!qkvPrepared) {
                        LinearWithActivationQuant(attnInput, pre + ".attn.wkv.weight", kv);
                        DsRMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
                    }
                    kv.Reshape({batch, block, headDim});
                    bool rotaryPrepared = false;
#ifdef USE_CUDA
                    rotaryPrepared = FastllmCudaDeepSeekV41DraftRotaryTable(
                        q, kv, dspark.layers[stage].windowKV, windowRope.ropeDim, windowRope.base, startPos,
                        windowRope.originalSeqLen, windowRope.factor, windowRope.betaFast,
                        windowRope.betaSlow, devicePosition, table);
#endif
                    auto rotate = [&](Data &data, bool inverse, int quantMode) {
                        if (batch == 1) {
                            DsRotaryQuant(data, windowRope, startPos, 1, inverse, quantMode, 32, -1, devicePosition);
                        } else for (int b = 0; b < batch; ++b) {
                            Data view; view.FakeFrom(data, size_t(b) * data.Count(1) * data.unitSize);
                            auto dims = data.dims; dims[0] = 1; view.Resize(dims);
                            DsRotaryQuant(view, windowRope, positions[b], 1, inverse, quantMode, 32, -1,
                                          devicePosition ? devicePosition + b : nullptr);
                        }
                    };
                    if (!rotaryPrepared) {
                        rotate(q, false, 0);
                        rotate(kv, false, 1);
                        Cat(dspark.layers[stage].windowKV, kv, 1, table);
                    }
                    AssertInFastLLM(table.dims.size() == 3 && table.dims[1] == tableRows,
                                    "DeepSeekV41 DSpark: draft attention gather table has a wrong size.");
#ifdef USE_CUDA
                    Data scores;
                    if (!ownerAttention || !FastllmCudaDsparkDraftSparseAttention(q, kv, table, attentionIndices,
                            weight[pre + ".attn.attn_sink"], softmaxScale, scores, attnOut))
#endif
                    DsSparseAttention(q, kv, nullptr, &table, &attentionIndices,
                                      weight[pre + ".attn.attn_sink"], 1, 0, softmaxScale, attnOut);
                    rotate(attnOut, true, 0);

                    DeepSeekV4WoA(attnOut, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
                    LinearWithActivationQuant(woAOut, pre + ".attn.wo_b.weight", attnProj);
                }
                bool fusedPost = false;
#ifdef USE_CUDA
                if (forked) FastllmCudaCurrentThreadStreamWaitEvent(graphState->hcBranch.done);
                fusedPost = DsHcPostMixPreNorm(attnProj, *curHidden, attnPre, attnPost, attnComb,
                    weight[pre + ".hc_ffn_fn"], weight[pre + ".hc_ffn_scale"],
                    weight[pre + ".hc_ffn_base"], weight[pre + ".ffn_norm.weight"],
                    hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                    *nextHidden, ffnPre, ffnPost, ffnComb, ffnInput);
#endif
                if (!fusedPost) {
                    DsHcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
                    DsHcMixPreNorm(*nextHidden, attnPre, weight[pre + ".hc_ffn_fn"],
                        weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                        weight[pre + ".ffn_norm.weight"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                        ffnPre, ffnPost, ffnComb, x, ffnInput);
                }
                std::swap(curHidden, nextHidden);
                std::vector<int> ffnDims = ffnInput.dims;
                ffnInput.Reshape({rows, dim});

                // 路由：sqrt(softplus(logits))，bias 只参与选择（草稿层 128 专家 top-3）
                {
                    const std::string gpre = pre + ".ffn.gate";
                    Data xFloat, logits;
                    Data &gateBias = weight[gpre + ".bias"];
                    bool routed = false;
#ifdef USE_CUDA
                    if (norm_topk_prob && !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                        !DsEnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE")) {
                        routed = FastllmCudaV41RouterSelectTopK(ffnInput, weight[gpre + ".weight"],
                            gateBias, gate_temp, routed_scaling_factor, logits,
                            expertIndex, expertScore, v41DsparkTopk);
                    }
#endif
                    if (!routed) {
                        ToDataType(ffnInput, xFloat, DataType::FLOAT32);
                        Linear(xFloat, weight[gpre + ".weight"], Data(), logits);
                        ToDataType(logits, DataType::FLOAT32);
                        if (std::fabs(gate_temp - 1.0f) > 1e-6f) {
                            Mul(logits, 1.0f / gate_temp, logits);
                        }
#ifdef USE_CUDA
                        if (logits.dataDevice == DataDevice::CUDA &&
                            !DsEnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE") &&
                            FastllmCudaDeepSeekV4RouteScoreTransform(logits, 2)) {
                            gateBias.ToDevice(DataDevice::CUDA);
                            SelectExpert(logits, expertIndex, expertScore, v41DsparkTopk, true,
                                         routed_scaling_factor, &gateBias);
                            routed = true;
                        }
#endif
                    }
                    if (!routed) {
                        logits.ToDevice(DataDevice::CPU);
                        gateBias.ToDevice(DataDevice::CPU);
                        const float *raw = (const float*)logits.cpuData;
                        const float *bias = (const float*)gateBias.cpuData;
                        std::vector<int> indices((uint64_t)rows * v41DsparkTopk);
                        std::vector<float> scores((uint64_t)rows * v41DsparkTopk);
                        std::vector<float> original(v41DsparkExperts), select(v41DsparkExperts);
                        for (int t = 0; t < rows; t++) {
                            for (int e = 0; e < v41DsparkExperts; e++) {
                                original[e] = std::sqrt(DsSoftplus(raw[(uint64_t)t * v41DsparkExperts + e]));
                                select[e] = original[e] + bias[e];
                            }
                            float sum = 0.0f;
                            for (int k = 0; k < v41DsparkTopk; k++) {
                                int best = 0;
                                for (int e = 1; e < v41DsparkExperts; e++) {
                                    if (select[e] > select[best]) {
                                        best = e;
                                    }
                                }
                                indices[(uint64_t)t * v41DsparkTopk + k] = best;
                                scores[(uint64_t)t * v41DsparkTopk + k] = original[best];
                                sum += original[best];
                                select[best] = -std::numeric_limits<float>::infinity();
                            }
                            for (int k = 0; k < v41DsparkTopk; k++) {
                                float &v = scores[(uint64_t)t * v41DsparkTopk + k];
                                if (norm_topk_prob && v41DsparkTopk > 1) {
                                    v /= (sum + 1e-20f);
                                }
                                v *= routed_scaling_factor;
                            }
                        }
                        Data idxData(DataType::INT32, {rows, v41DsparkTopk});
                        idxData.Allocate();
                        memcpy(idxData.cpuData, indices.data(), indices.size() * sizeof(int));
                        expertIndex.CopyFrom(idxData);
                        expertScore.CopyFrom(Data(DataType::FLOAT32, {rows, v41DsparkTopk}, scores));
                    }
                }

                bool expertPostReady = false;
#ifdef FASTLLM_ENABLE_DSV41_SM100_MEGA
                if (GetCudaSharedExpert() && swiglu_limit == 10.f && v41DsparkTopk == 3 &&
                    !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH"))
                    expertPostReady = MultiCudaDsv41DraftExperts(ffnInput, expertIndex, expertScore,
                        *curHidden, ffnPost, ffnComb, *nextHidden, v41DsparkMoeWeights[stage], v41DsparkTpDevices);
#endif
                if (!expertPostReady) {
                    std::vector<Data*> moeWeights = v41DsparkMoeWeights[stage];
#ifdef USE_NUMAS
                    HostMoePlacementScope placement(moeWeights.size() > 2 ? moeWeights[2] : nullptr);
#endif
                    DeepSeekV41DsparkWorkspace::SharedExpertWorkspace localShared;
                    auto &shared = graphState ? graphState->sharedExperts[stage] : localShared;
                    bool hasSharedExpertOut = false;
                    auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
                    auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
                    if (GetCudaSharedExpert() && !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                        sharedGateupIt != weight.weight.end() &&
                        sharedDownIt != weight.weight.end() && !sharedGateupIt->second.isDiskWeight &&
                        !sharedDownIt->second.isDiskWeight) {
#ifdef USE_CUDA
                        if (dsparkCapturing && graphState->parallelExperts) FastllmCudaGraphMarkParallelFork(stage);
#endif
                        LinearWithActivationQuant(ffnInput, pre + ".ffn.shared_experts.gateup.weight", shared.gateup);
#ifdef USE_CUDA
                        AssertInFastLLM(FastllmCudaDeepSeekV41SharedSwiglu(shared.gateup, swiglu_limit, shared.activated),
                                        "DeepSeekV41 DSpark: CUDA shared expert activation rejected input.");
#else
                        Swiglu(shared.gateup, shared.activated);
#endif
                        LinearWithActivationQuant(shared.activated, pre + ".ffn.shared_experts.w2.weight", shared.output);
                        ToDataType(shared.output, DataType::BFLOAT16);
                        moeWeights[0] = moeWeights[1] = nullptr;
                        hasSharedExpertOut = true;
#ifdef USE_CUDA
                        if (dsparkCapturing && graphState->parallelExperts) {
                            FastllmCudaGraphMarkParallelFirstDone(stage);
                            FastllmCudaGraphMarkParallelSecondBegin(stage);
                        }
#endif
                    }
                    ((Executor*)GetExecutor())->SetFirstDevice(
                        SelectSpecialWeightDevice(pre + ".ffn.experts.0.gateup.weight", layerId));
                    bool hybrid = false;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
                    if (graphState && (graphState->hybridExperts || graphState->diskExperts) && !hasSharedExpertOut)
                        graphState->disabled = true;
                    if (graphState && graphState->diskExperts && hasSharedExpertOut) {
                        auto &state = *graphState;
                        state.routedExperts.resize(v41DsparkLayers);
                        auto &routed = state.routedExperts[stage];
                        routed.input.CopyFrom(ffnInput);
                        routed.index.CopyFrom(expertIndex);
                        routed.score.CopyFrom(expertScore);
                        if (!routed.graph && !dsparkCapturing) {
                            routed.graph = state.diskExperts->Add(routed.input, routed.index, routed.score,
                                routed.output, moeWeights, swiglu_limit);
                            if (!routed.graph) state.disabled = true;
                        }
                        if (dsparkCapturing && !state.disabled) {
                            AssertInFastLLM(FastllmCudaGraphAppendChild(routed.graph),
                                           "DeepSeekV41 DSpark disk MoE graph failed.");
                            ffnOut.CopyFrom(routed.output);
                            hybrid = true;
                        }
                        // Eager warmups use MergeMOE below. A partial draft
                        // must not launch a worker that expects later layers.
                    }
                    if (graphState && graphState->hybridExperts && hasSharedExpertOut) {
                        auto &state = *graphState;
                        state.routedExperts.resize(v41DsparkLayers);
                        state.moeHandles.resize(v41DsparkLayers, nullptr);
                        auto &routed = state.routedExperts[stage];
                        routed.input.CopyFrom(ffnInput);
                        routed.index.CopyFrom(expertIndex);
                        routed.score.CopyFrom(expertScore);
                        if (!state.moeHandles[stage] && !dsparkCapturing) {
                            state.moeHandles[stage] = FastllmCudaCreateNativeMoeFullGraph(
                                routed.input, routed.index, routed.score, routed.output,
                                moeWeights.data(), int(moeWeights.size()), layerId, nullptr, &routed.graph);
                            if (state.moeHandles[stage] &&
                                !FastllmCudaGraphInstantiate(routed.graph, &routed.exec)) state.disabled = true;
                            if (!state.moeHandles[stage]) state.disabled = true;
                        }
                        if (!state.disabled) {
                            AssertInFastLLM(dsparkCapturing ? FastllmCudaGraphAppendChild(routed.graph) :
                                FastllmCudaLaunchNativeMoeFullGraph(routed.exec, &state.moeHandles[stage], 1),
                                "DeepSeekV41 DSpark native MoE graph failed.");
                            ffnOut.CopyFrom(routed.output);
                            hybrid = true;
                        }
                    }
                    if (!hybrid && hasSharedExpertOut &&
                        ffnInput.dataDevice == DataDevice::CUDA && ffnInput.dims.size() == 2 &&
                        FastllmCudaCanRunMoeHybrid(moeWeights.data(), (int)moeWeights.size())) {
                        expertIndex.ToDevice(DataDevice::CUDA, ffnInput.dataDeviceIds, true);
                        expertScore.ToDevice(DataDevice::CUDA, ffnInput.dataDeviceIds, true);
                        hybrid = fastllm::RunHybridMoe(ffnInput, expertIndex, expertScore, ffnOut,
                            moeWeights.data(), (int)moeWeights.size(), layerId);
                    }
#endif
                    if (!hybrid) MergeMOEBlock(&ffnInput, &expertIndex, &expertScore, &moeWeights, &v41DsparkMoeBiass[stage],
                                  &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layerId,
                                  ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                                  MoeGateSwiglu, true, swiglu_limit, true, nullptr,
                                  quantizedLinearNames.count(pre + ".ffn.experts.0.w1.weight") ? 32 : 128,
                                  quantizedLinearNames.count(pre + ".ffn.shared_experts.w1.weight") != 0);
                    // Only the persistent packet path supplies graph-safe EP buffers.
                    if (graphState && !residentDraft && !graphState->hybridExperts && !graphState->diskExperts && !ffnOut.cudaDataBorrowed) graphState->disabled = true;
                    ApplyDsparkDevice();
                    if (hasSharedExpertOut) {
#ifdef USE_CUDA
                        if (dsparkCapturing && graphState->parallelExperts) FastllmCudaGraphMarkParallelJoin(stage);
#endif
                        ffnOut.ToDevice(shared.output.dataDevice);
#ifdef USE_CUDA
                        const auto &owner = curHidden->dataDeviceIds;
                        if (!ffnOut.multiDeviceData && !shared.output.multiDeviceData &&
                            owner.size() == 1 && owner.front() == FastllmCudaGetDevice() &&
                            ffnOut.dataDeviceIds == owner && shared.output.dataDeviceIds == owner &&
                            ffnPost.dataDeviceIds == owner && ffnComb.dataDeviceIds == owner) {
                            expertPostReady = FastllmCudaDeepSeekV41MoeHcPost(ffnOut, shared.output,
                                *curHidden, ffnPost, ffnComb, *nextHidden);
                        }
#endif
                        if (!expertPostReady) {
                            Data *sharedOutput = &shared.output;
                            if (shared.output.dataType != ffnOut.dataType) {
                                ToDataType(shared.output, shared.floatOutput, ffnOut.dataType);
                                sharedOutput = &shared.floatOutput;
                            }
                            AddTo(ffnOut, *sharedOutput);
                        }
                    }
                }
                if (!expertPostReady) {
                    if (ffnOut.dataType != DataType::BFLOAT16) ToDataType(ffnOut, DataType::BFLOAT16);
                    ffnOut.Reshape(ffnDims);
                    DsHcPost(ffnOut, *curHidden, ffnPost, ffnComb, *nextHidden);
                }
                std::swap(curHidden, nextHidden);
                layerPreMix.CopyFrom(ffnPre);
            }

            DsDraftTimingSlot().layers = DsElapsed(layersStart);

            DsHcApplyPreNorm(*curHidden, layerPreMix, weight[last + ".norm.weight"],
                             rms_norm_eps, headHidden, normed, confidenceWeight != nullptr);
            // ---- head：mtp.<last>.norm + 共享 lm_head ----
            const double headStart = DsTick();
            bool headProjected = false, shardedMarkov = false;
#ifdef USE_CUDA
            // Sharded sampling retains an unfiltered proposal distribution.
            // Top-k/top-p requests use the filtered device sampler below.
            if (graphState && !v41DsparkTpDevices.empty() &&
                (!randomDraft || (config.top_k >= vocab && config.top_p == 1.f)) &&
                normed.dataType == DataType::BFLOAT16 &&
                !quantizedLinearNames.count("head.weight")) {
                ApplyDsparkDevice(); FastllmMultiCudaSetDevice(v41DsparkTpDevices);
                MultiCudaDsparkMarkovRequest request{markovEmbedWeight, markovHeadWeight,
                    graphState->markovTokens, markovAll, graphState->markovWorkspace};
                if (randomDraft) {
                    request.proposalProbs = &dspark.proposalProbs;
                    request.temperature = config.temperature;
                }
                headProjected = MultiCudaLinearOutputGatherPersistent(normed, weight["head.weight"], logits,
                    v41DsparkHeadWorkspace, &request, true);
                shardedMarkov = request.completed;
            }
#endif
#if defined(USE_CUDA) && defined(FASTLLM_ENABLE_CUTLASS_MXFP8_SM120)
            // This entry is draft-only. DsparkProjectHead also projects target
            // verification logits, which must retain the original BF16 head.
            if (!headProjected && v41DsparkTpDevices.empty() &&
                !DsEnvFlag("FASTLLM_DSV41_REFERENCE_MATH")) {
                headProjected = FastllmCudaTryFp8DraftHead(normed, weight["head.weight"], logits);
            }
#endif
            if (!headProjected) {
                if (v41DsparkTpDevices.empty()) {
                    if (!TryBFloat16HeadLinear(normed, logits))
                        Linear(normed, weight["head.weight"], *GetEmptyData(), logits);
                } else {
                    DsparkProjectHead(normed, logits, true);
                }
            }
            if (!shardedMarkov) ToDataType(logits, DataType::FLOAT32);

            DsDraftTimingSlot().head = DsElapsed(headStart);
            if (graphState && !v41DsparkTpDevices.empty() && !v41DsparkHeadWorkspace) graphState->disabled = true;
#ifdef USE_CUDA
            if (graphState && !shardedMarkov) {
                if (batch == 1) {
                    const bool ok = randomDraft ? FastllmCudaDeepSeekV41MarkovFilteredSampleDevice(logits,
                        markovEmbedWeight, markovHeadWeight, block, config.temperature, config.top_k, config.top_p,
                        graphState->markovTokens, dspark.proposalProbs, markovAll, graphState->samplingWorkspace) :
                        FastllmCudaDeepSeekV41MarkovChainDevice(logits, markovEmbedWeight, markovHeadWeight,
                        block, graphState->markovTokens, graphState->markovValues, graphState->markovIndices, markovAll);
                    AssertInFastLLM(ok, "DeepSeekV41 DSpark device Markov chain rejected input.");
                } else if (randomDraft) {
                    AssertInFastLLM(FastllmCudaDeepSeekV41MarkovFilteredSampleBatchDevice(logits,
                        markovEmbedWeight, markovHeadWeight, block, config.temperature, config.top_k, config.top_p,
                        graphState->markovTokens, dspark.proposalProbs, markovAll, graphState->markovValues,
                        graphState->samplingWorkspace), "DeepSeekV41 DSpark batched sampling rejected input.");
                } else {
                    graphState->greedyWorkspaces.resize(batch);
                    markovAll.dataType = DataType::FLOAT32; markovAll.UpdateUnitSize();
                    markovAll.Resize({batch, block, markovEmbedWeight.dims[1]});
                    markovAll.ToDevice(DataDevice::CUDA, graphDevices, false); markovAll.Allocate();
                    for (int b = 0; b < batch; ++b) {
                        Data localLogits, localTokens, localEmbeds;
                        localLogits.FakeFrom(logits, size_t(b) * block * vocab * sizeof(float));
                        localLogits.Resize({1, block, vocab});
                        localTokens.FakeFrom(graphState->markovTokens, size_t(b) * (block + 1) * sizeof(int));
                        localTokens.Resize({block + 1});
                        localEmbeds.FakeFrom(markovAll, size_t(b) * block * markovEmbedWeight.dims[1] * sizeof(float));
                        localEmbeds.Resize({1, block, markovEmbedWeight.dims[1]});
                        auto &sample = graphState->greedyWorkspaces[b];
                        const bool ok = FastllmCudaDeepSeekV41MarkovChainDevice(localLogits, markovEmbedWeight, markovHeadWeight,
                            block, localTokens, sample.values, sample.indices, localEmbeds);
                        AssertInFastLLM(ok, "DeepSeekV41 DSpark batched Markov chain rejected input.");
                    }
                }
            }
            if (graphState && confidenceWeight && markovAll.dims.size() == 3)
                projectConfidence(graphState->confidenceLogits);
#endif
        };
#ifdef USE_CUDA
        // A growing single-request history does not reuse its exact-width graph.
        // Run the same prepared kernels once, preserving the compact attention layout.
        if (graphState && batch == 1 && valid < window_size && !fixedPartialHistory && !randomDraft &&
            !hybridDraft && !diskDraft && !DsEnvFlag("FASTLLM_DSPARK_COMPACT_WINDOW")) {
            FastllmCudaGraphPreparation preparation;
            backbone();
        } else if (!graphState || !DsRunDraftGraph(*graphState, backbone,
            deferGreedy && !randomDraft && !confidenceWeight && !hybridDraft && !diskDraft)) backbone();
#else
        backbone();
#endif

        // ---- markov head：逐位置加 bigram 偏置后生成候选 ----
        //
        // CUDA 融合路径把串行 token 依赖留在设备上，每轮只回读一次。
        // 截断采样、工具约束及不支持的 dtype / 布局继续使用通用实现。
        const double markovStart = DsTick();
        dspark.sampledProposal = randomDraft;
        const int markovRank = markovEmbedWeight.dims.size() == 2 ? markovEmbedWeight.dims[1]
                                                                  : v41DsparkMarkovRank;
        bool fusedMarkov = false;
#ifdef USE_CUDA
        if (randomDraft && !graphState) {
            AssertInFastLLM(logits.dataDevice == DataDevice::CUDA && logits.cudaData,
                            "DeepSeekV41 DSpark: random draft logits must be on CUDA.");
            const int device = logits.dataDeviceIds.empty() ? FastllmCudaGetDevice() : logits.dataDeviceIds[0];
            prepareProposalProbs(device, logits.dims.back());
        }
        // Only a request-owned greedy graph may keep its readback in flight across
        // the scheduler call. Acceptance, force hooks and context bounds still run
        // when the same request consumes the proposal. Reset/discard drains its slot.
        if (deferGreedy && batch == 1 && !randomDraft && !confidenceWeight && !hybridDraft && !diskDraft &&
            graphState && !graphState->execs.empty() && dspark.workspace.get() == graphState &&
            graphState->QueueProposal(block)) {
            dspark.anchor = anchorToken;
            dspark.anchorPos = dspark.committed;
            return true;
        }
        if (graphState) {
            std::vector<int> allTokens(batch * (block + 1));
            FastllmCudaCopyFromDeviceToHost(allTokens.data(), graphState->markovTokens.cudaData,
                                            allTokens.size() * sizeof(int));
            tokens.resize(rows);
            for (int b = 0; b < batch; ++b)
                std::copy_n(allTokens.begin() + b * (block + 1) + 1, block, tokens.begin() + b * block);
            fusedMarkov = true;
            DsDraftTimingSlot().markovFused = true;
        }
        if (!fusedMarkov && fusedMarkovAllowed && logits.dataDevice == DataDevice::CUDA &&
            (!randomDraft || (sampledMarkov && graphMarkov))) {
            if (!logits.dataDeviceIds.empty()) FastllmCudaSetDevice(logits.dataDeviceIds[0]);
            markovEmbedWeight.ToDevice(DataDevice::CUDA);
            markovHeadWeight.ToDevice(DataDevice::CUDA);
            if (randomDraft) {
                Data tokenBuffer(DataType::INT32, {block + 1}), samplingWorkspace;
                tokenBuffer.ToDevice(DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
                tokenBuffer.Allocate();
                FastllmCudaCopyFromHostToDevice(tokenBuffer.cudaData, &anchorToken, sizeof(int));
                fusedMarkov = FastllmCudaDeepSeekV41MarkovFilteredSampleDevice(logits, markovEmbedWeight,
                    markovHeadWeight, block, config.temperature, config.top_k, config.top_p,
                    tokenBuffer, dspark.proposalProbs, markovAll, samplingWorkspace);
                if (fusedMarkov) {
                    tokens.resize(block);
                    FastllmCudaCopyFromDeviceToHost(tokens.data(), (int*)tokenBuffer.cudaData + 1, block * sizeof(int));
                }
            } else {
                fusedMarkov = FastllmCudaDeepSeekV41MarkovChain(logits, markovEmbedWeight, markovHeadWeight,
                    anchorToken, block, tokens, markovAll);
            }
            DsDraftTimingSlot().markovFused = fusedMarkov;
        }
#endif
        if (!fusedMarkov) {
            std::vector<Data> markovEmbeds(block);
            int previous = anchorToken;
            GenerationConfig stepConfig = config;
            tokens.clear();
            tokens.reserve(block);
            for (int i = 0; i < block; i++) {
                const double biasStart = DsTick();
                Data ids(DataType::FLOAT32, {1, 1}, {(float)previous});
                Data embedOut, bias, stepLogits;
                EmbeddingDirect(ids, markovEmbedWeight, embedOut);
                markovEmbeds[i].CopyFrom(embedOut);
                Linear(embedOut, markovHeadWeight, Data(), bias);
                ToDataType(bias, DataType::FLOAT32);
                Split(logits, 1, i, i + 1, stepLogits);
                AddTo(stepLogits, bias);
                DsparkMaskToolLogits(stepLogits, stepConfig, {});
                DsDraftTimingSlot().markovBias += DsElapsed(biasStart);
                const double argmaxStart = DsTick();
                std::vector<int> best;
#ifdef USE_CUDA
                if (randomDraft) {
                    AssertInFastLLM(stepLogits.dataDevice == DataDevice::CUDA && stepLogits.cudaData,
                                    "DeepSeekV41 DSpark: random draft logits must be on CUDA.");
                    int token = -1;
                    if (!stepLogits.dataDeviceIds.empty()) FastllmCudaSetDevice(stepLogits.dataDeviceIds[0]);
                    const int vocab = stepLogits.dims.back();
                    AssertInFastLLM(FastllmCudaMtpSampleDraft((float*)stepLogits.cudaData,
                        (float*)dspark.proposalProbs.cudaData + (size_t)i * vocab,
                        &config.temperature, &config.top_k, &config.top_p, &token, 1, vocab),
                        "DeepSeekV41 DSpark: random draft sampling failed.");
                    best.push_back(token);
                } else
#endif
                {
                    best = DsArgmaxRows(stepLogits);
                }
                DsDraftTimingSlot().markovArgmax += DsElapsed(argmaxStart);
                AssertInFastLLM(best.size() == 1, "DeepSeekV41 DSpark: draft sampling failed.");
                previous = best[0];
                tokens.push_back(previous);
                if (config.tool_call_name_constraint_enabled || config.tool_call_parameter_name_constraint_enabled)
                    AdvanceToolCallConstraintText(stepConfig.tool_call_generated_text, previous);
            }
            Data tmp;
            for (int i = 0; i < block; i++) {
                if (i == 0) {
                    markovAll.CopyFrom(markovEmbeds[0]);
                } else {
                    Cat(markovAll, markovEmbeds[i], 1, tmp);
                    markovAll.CopyFrom(tmp);
                }
            }
            markovAll.Reshape({1, block, markovRank});
        }

        DsDraftTimingSlot().markov = DsElapsed(markovStart);

        // ---- confidence head：sigmoid(proj(cat(hidden, markov_embed))) ----
        const double confStart = DsTick();
        confidence.assign(rows, 1.0f);
        if (confidenceWeight && markovAll.dims.size() == 3) {
            Data confLogits;
            std::vector<float> capturedValues;
            const float *values = nullptr;
#ifdef USE_CUDA
            if (graphState) {
                capturedValues.resize(rows);
                FastllmCudaCopyFromDeviceToHost(capturedValues.data(), graphState->confidenceLogits.cudaData,
                    rows * sizeof(float));
                values = capturedValues.data();
            } else
#endif
            {
                projectConfidence(confLogits);
                confLogits.ToDevice(DataDevice::CPU);
                values = (const float*)confLogits.cpuData;
            }
            for (int i = 0; i < rows; i++) {
                float p = DsSigmoid(values[i]);
                confidence[i] = std::isfinite(p) ? p : 1.0f;
            }
        }
        DsDraftTimingSlot().conf = DsElapsed(confStart);
        return true;
    }

    void DeepSeekV41Model::DsparkReportStats() {
        const long long rounds = v41DsparkVerifyRounds.load();
        if (rounds <= 0) {
            return;
        }
        const long long proposed = v41DsparkProposedTokens.load();
        const long long accepted = v41DsparkAcceptedTokens.load();
        printf("[Fastllm] DeepSeek-V4.1 DSpark: %lld verify rounds, %lld / %lld drafts accepted "
               "(%.1f%%), %.2f tokens per target forward.\n",
               rounds, accepted, proposed,
               proposed > 0 ? 100.0 * (double)accepted / (double)proposed : 0.0,
               1.0 + (double)accepted / (double)rounds);
        fflush(stdout);
    }
}
