//
// Created by huangyuyang on 2/19/26.
//

#include "utils.h"

#include "qwen3_5.h"
#include "blocks/baseblock.h"
#include "executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "json11.hpp"

#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

#ifdef USE_TFACC
#include "fastllm-tfacc.h"
#endif

namespace fastllm {
#ifdef USE_NUMAS
    extern void RegisterNumas(fastllm::Data *data, std::string weightType);
#endif

    static std::string Qwen35MoeExpertPrefix(int layer, int expert) {
        return Qwen3_5Model::language_prefix + "layers." + std::to_string(layer) +
               ".mlp.experts." + std::to_string(expert) + ".";
    }

    static std::string Qwen35MoeFusedWeightName(int layer, const std::string &kind) {
        return Qwen3_5Model::language_prefix + "layers." + std::to_string(layer) +
               ".mlp.fused_experts." + kind + "_proj.weight";
    }

    static std::string Qwen35MoeExpertWeightName(int layer, int expert, const std::string &kind) {
        return Qwen35MoeExpertPrefix(layer, expert) + kind + "_proj.weight";
    }

    static bool Qwen35MoeParseExpertWeightName(const std::string &name,
                                               int &layer, int &expert, std::string &kind) {
        const std::string prefix = Qwen3_5Model::language_prefix + "layers.";
        if (!StartWith(name, prefix)) {
            return false;
        }
        size_t pos = prefix.size();
        if (pos >= name.size() || !std::isdigit((unsigned char)name[pos])) {
            return false;
        }
        layer = 0;
        while (pos < name.size() && std::isdigit((unsigned char)name[pos])) {
            layer = layer * 10 + (name[pos] - '0');
            pos++;
        }
        const std::string mid = ".mlp.experts.";
        if (name.compare(pos, mid.size(), mid) != 0) {
            return false;
        }
        pos += mid.size();
        if (pos >= name.size() || !std::isdigit((unsigned char)name[pos])) {
            return false;
        }
        expert = 0;
        while (pos < name.size() && std::isdigit((unsigned char)name[pos])) {
            expert = expert * 10 + (name[pos] - '0');
            pos++;
        }
        const std::string suffix = "_proj.weight";
        if (name.compare(pos, 1, ".") != 0 || name.size() <= pos + 1 + suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return false;
        }
        kind = name.substr(pos + 1, name.size() - pos - 1 - suffix.size());
        return kind == "gate" || kind == "up" || kind == "gateup" || kind == "down";
    }

    static int Qwen35MoeSourceLoadPriority(const std::string &name, int numExperts) {
        int layer = -1, expert = -1;
        std::string kind;
        if (!Qwen35MoeParseExpertWeightName(name, layer, expert, kind)) {
            return 0;
        }
        (void)expert;
        (void)numExperts;
        int layerStride = 3;
        int order = (kind == "down") ? 2 : (kind == "up" ? 1 : 0);
        return -100000000 + layer * layerStride + order;
    }

    static bool Qwen35MoeIsFusedFp8Type(DataType dataType) {
        return dataType == DataType::FP8_E4M3 ||
               dataType == DataType::FP8_E4M3_BLOCK_128;
    }

    static bool Qwen35MoeIsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return lowered == "true" || lowered == "1" || lowered == "yes" || lowered == "on";
    }

    static bool Qwen35MoeDisableFusedMoe() {
        const char *env = std::getenv("FASTLLM_QWEN35_MOE_DISABLE_FUSED_MOE");
        if (env == nullptr) {
            env = std::getenv("FASTLLM_QWEN3_MOE_DISABLE_FUSED_MOE");
        }
        return env != nullptr && Qwen35MoeIsTrueString(env);
    }

    static constexpr int QWEN35_MTP_LOG_INTERVAL = 64;
    static constexpr int QWEN35_MTP_MAX_DRAFTS = 8;
    static constexpr int QWEN35_MTP_FAST_SEQ_MAX = 6;
    static constexpr int QWEN35_MTP_PREFIX_SNAPSHOT_MAX =
        QWEN35_MTP_FAST_SEQ_MAX - 1;

    enum Qwen35MtpProfilePath {
        QWEN35_MTP_PROFILE_SEED = 0,
        QWEN35_MTP_PROFILE_TP_INPLACE = 1,
        QWEN35_MTP_PROFILE_TP_COPY = 2,
        QWEN35_MTP_PROFILE_SINGLE = 3
    };

    struct Qwen35MtpProfileAggregate {
        std::atomic<long long> samples{0};
        std::atomic<long long> seedPath{0};
        std::atomic<long long> tpInplacePath{0};
        std::atomic<long long> tpCopyPath{0};
        std::atomic<long long> singlePath{0};
        std::atomic<long long> speculative{0};
        std::atomic<long long> fullAccept{0};
        std::atomic<long long> partialAccept{0};
        std::atomic<long long> rejectFirst{0};
        std::atomic<long long> draftSlots{0};
        std::atomic<long long> matchedDrafts{0};
        std::atomic<long long> committedTokens{0};
        std::atomic<long long> setupUs{0};
        std::atomic<long long> cachePrepUs{0};
        std::atomic<long long> targetUs{0};
        std::atomic<long long> matchUs{0};
        std::atomic<long long> commitUs{0};
        std::atomic<long long> rollbackUs{0};
        std::atomic<long long> retryUs{0};
        std::atomic<long long> draftUs{0};
        std::atomic<long long> draftFirstUs{0};
        std::atomic<long long> draftExtraUs{0};
        std::atomic<long long> totalUs{0};
    };

    struct Qwen35MtpTargetProfileAggregate {
        std::atomic<long long> samples{0};
        std::atomic<long long> tensorParallelCalls{0};
        std::atomic<long long> seqTokens{0};
        std::atomic<long long> logitRows{0};
        std::atomic<long long> setupUs{0};
        std::atomic<long long> weightPrepUs{0};
        std::atomic<long long> inputPrepUs{0};
        std::atomic<long long> embeddingUs{0};
        std::atomic<long long> cacheLocalUs{0};
        std::atomic<long long> workerUs{0};
        std::atomic<long long> metaSyncUs{0};
        std::atomic<long long> samplingUs{0};
        std::atomic<long long> totalUs{0};
    };

    struct Qwen35MtpWorkerProfileAggregate {
        std::atomic<long long> samples{0};
        std::atomic<long long> firstRankCalls{0};
        std::atomic<long long> seqTokens{0};
        std::atomic<long long> setupUs{0};
        std::atomic<long long> layersUs{0};
        std::atomic<long long> headUs{0};
        std::atomic<long long> totalUs{0};
    };

    static Qwen35MtpProfileAggregate &Qwen35MtpProfileStats() {
        static Qwen35MtpProfileAggregate stats;
        return stats;
    }

    static Qwen35MtpTargetProfileAggregate &Qwen35MtpTargetProfileStats() {
        static Qwen35MtpTargetProfileAggregate stats;
        return stats;
    }

    static Qwen35MtpWorkerProfileAggregate &Qwen35MtpWorkerProfileStats() {
        static Qwen35MtpWorkerProfileAggregate stats;
        return stats;
    }

    static int Qwen35MtpDraftsPerStep() {
        const char *env = std::getenv("FASTLLM_QWEN35_ENABLE_MTP");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        if (Qwen35MoeIsTrueString(env)) {
            return 1;
        }
        int value = atoi(env);
        return std::max(0, std::min(value, QWEN35_MTP_MAX_DRAFTS));
    }

    static bool Qwen35MtpDisabledByEnv() {
        return Qwen35MtpDraftsPerStep() <= 0;
    }

    static bool Qwen35MtpWarmupEnabled() {
        const char *env = std::getenv("FASTLLM_QWEN35_MTP_WARMUP");
        return env == nullptr || env[0] == '\0' || Qwen35MoeIsTrueString(env);
    }

    static bool Qwen35MtpFusedLinearSeqEnabled() {
        static bool enabled = []() {
            const char *env = std::getenv("FASTLLM_QWEN35_MTP_FUSED_LINEAR_SEQ");
            return env == nullptr || env[0] == '\0' || Qwen35MoeIsTrueString(env);
        }();
        return enabled;
    }

    static int Qwen35MtpProfileInterval() {
        static int interval = []() {
            const char *env = std::getenv("FASTLLM_QWEN35_MTP_PROFILE");
            if (env == nullptr || env[0] == '\0') {
                return 0;
            }
            if (Qwen35MoeIsTrueString(env)) {
                return QWEN35_MTP_LOG_INTERVAL;
            }
            int value = atoi(env);
            return value > 0 ? value : 0;
        }();
        return interval;
    }

    static int Qwen35MtpWorkerProfileInterval() {
        static int interval = []() {
            const char *env = std::getenv("FASTLLM_QWEN35_MTP_WORKER_PROFILE");
            if (env == nullptr || env[0] == '\0') {
                return 0;
            }
            if (Qwen35MoeIsTrueString(env)) {
                return QWEN35_MTP_LOG_INTERVAL;
            }
            int value = atoi(env);
            return value > 0 ? value : 0;
        }();
        return interval;
    }

    static long long Qwen35MtpProfileElapsedUs(
            std::chrono::steady_clock::time_point begin,
            std::chrono::steady_clock::time_point end) {
        return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    static void Qwen35MtpProfilePrintIfNeeded(int interval,
                                              Qwen35MtpProfileAggregate &stats,
                                              long long samples) {
        if (interval <= 0 || samples <= 0 || samples % interval != 0) {
            return;
        }
        long long setupUs = stats.setupUs.load(std::memory_order_relaxed);
        long long cachePrepUs = stats.cachePrepUs.load(std::memory_order_relaxed);
        long long targetUs = stats.targetUs.load(std::memory_order_relaxed);
        long long matchUs = stats.matchUs.load(std::memory_order_relaxed);
        long long commitUs = stats.commitUs.load(std::memory_order_relaxed);
        long long rollbackUs = stats.rollbackUs.load(std::memory_order_relaxed);
        long long retryUs = stats.retryUs.load(std::memory_order_relaxed);
        long long draftUs = stats.draftUs.load(std::memory_order_relaxed);
        long long totalUs = stats.totalUs.load(std::memory_order_relaxed);
        long long knownUs = setupUs + cachePrepUs + targetUs + matchUs +
                            commitUs + rollbackUs + retryUs + draftUs;
        long long otherUs = std::max(0LL, totalUs - knownUs);
        auto avgMs = [&](long long us) {
            return samples > 0 ? (double)us / 1000.0 / (double)samples : 0.0;
        };
        auto avgCount = [&](long long count) {
            return samples > 0 ? (double)count / (double)samples : 0.0;
        };
        printf("[Qwen3.5 MTP profile] samples=%lld paths={seed=%lld,tp_inplace=%lld,tp_copy=%lld,single=%lld} "
               "accept={spec=%lld,full=%lld,partial=%lld,reject0=%lld} "
               "avg_tokens={commit=%.2f,matched_draft=%.2f,draft_slots=%.2f} "
               "avg_ms={total=%.3f,setup=%.3f,cache_prep=%.3f,target=%.3f,match=%.3f,"
               "commit=%.3f,rollback=%.3f,retry=%.3f,draft=%.3f,draft_first=%.3f,"
               "draft_extra=%.3f,other=%.3f}.\n",
               samples,
               stats.seedPath.load(std::memory_order_relaxed),
               stats.tpInplacePath.load(std::memory_order_relaxed),
               stats.tpCopyPath.load(std::memory_order_relaxed),
               stats.singlePath.load(std::memory_order_relaxed),
               stats.speculative.load(std::memory_order_relaxed),
               stats.fullAccept.load(std::memory_order_relaxed),
               stats.partialAccept.load(std::memory_order_relaxed),
               stats.rejectFirst.load(std::memory_order_relaxed),
               avgCount(stats.committedTokens.load(std::memory_order_relaxed)),
               avgCount(stats.matchedDrafts.load(std::memory_order_relaxed)),
               avgCount(stats.draftSlots.load(std::memory_order_relaxed)),
               avgMs(totalUs), avgMs(setupUs), avgMs(cachePrepUs), avgMs(targetUs),
               avgMs(matchUs), avgMs(commitUs), avgMs(rollbackUs), avgMs(retryUs),
               avgMs(draftUs),
               avgMs(stats.draftFirstUs.load(std::memory_order_relaxed)),
               avgMs(stats.draftExtraUs.load(std::memory_order_relaxed)),
               avgMs(otherUs));
        fflush(stdout);
    }

    static void Qwen35MtpProfileRecord(
            int interval, Qwen35MtpProfilePath path, bool speculative,
            int draftSlots, int matchedDrafts, int committedTokens,
            long long setupUs, long long cachePrepUs, long long targetUs,
            long long matchUs, long long commitUs, long long rollbackUs,
            long long retryUs, long long draftUs, long long draftFirstUs,
            long long draftExtraUs, long long totalUs) {
        if (interval <= 0) {
            return;
        }
        Qwen35MtpProfileAggregate &stats = Qwen35MtpProfileStats();
        long long samples = stats.samples.fetch_add(1, std::memory_order_relaxed) + 1;
        switch (path) {
        case QWEN35_MTP_PROFILE_SEED:
            stats.seedPath.fetch_add(1, std::memory_order_relaxed);
            break;
        case QWEN35_MTP_PROFILE_TP_INPLACE:
            stats.tpInplacePath.fetch_add(1, std::memory_order_relaxed);
            break;
        case QWEN35_MTP_PROFILE_TP_COPY:
            stats.tpCopyPath.fetch_add(1, std::memory_order_relaxed);
            break;
        case QWEN35_MTP_PROFILE_SINGLE:
            stats.singlePath.fetch_add(1, std::memory_order_relaxed);
            break;
        }
        if (speculative) {
            stats.speculative.fetch_add(1, std::memory_order_relaxed);
            if (matchedDrafts >= draftSlots) {
                stats.fullAccept.fetch_add(1, std::memory_order_relaxed);
            } else if (matchedDrafts > 0) {
                stats.partialAccept.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.rejectFirst.fetch_add(1, std::memory_order_relaxed);
            }
        }
        stats.draftSlots.fetch_add(std::max(0, draftSlots), std::memory_order_relaxed);
        stats.matchedDrafts.fetch_add(std::max(0, matchedDrafts), std::memory_order_relaxed);
        stats.committedTokens.fetch_add(std::max(0, committedTokens), std::memory_order_relaxed);
        stats.setupUs.fetch_add(setupUs, std::memory_order_relaxed);
        stats.cachePrepUs.fetch_add(cachePrepUs, std::memory_order_relaxed);
        stats.targetUs.fetch_add(targetUs, std::memory_order_relaxed);
        stats.matchUs.fetch_add(matchUs, std::memory_order_relaxed);
        stats.commitUs.fetch_add(commitUs, std::memory_order_relaxed);
        stats.rollbackUs.fetch_add(rollbackUs, std::memory_order_relaxed);
        stats.retryUs.fetch_add(retryUs, std::memory_order_relaxed);
        stats.draftUs.fetch_add(draftUs, std::memory_order_relaxed);
        stats.draftFirstUs.fetch_add(draftFirstUs, std::memory_order_relaxed);
        stats.draftExtraUs.fetch_add(draftExtraUs, std::memory_order_relaxed);
        stats.totalUs.fetch_add(totalUs, std::memory_order_relaxed);
        Qwen35MtpProfilePrintIfNeeded(interval, stats, samples);
    }

    static void Qwen35MtpTargetProfilePrintIfNeeded(
            int interval, Qwen35MtpTargetProfileAggregate &stats,
            long long samples) {
        if (interval <= 0 || samples <= 0 || samples % interval != 0) {
            return;
        }
        long long setupUs = stats.setupUs.load(std::memory_order_relaxed);
        long long weightPrepUs = stats.weightPrepUs.load(std::memory_order_relaxed);
        long long inputPrepUs = stats.inputPrepUs.load(std::memory_order_relaxed);
        long long embeddingUs = stats.embeddingUs.load(std::memory_order_relaxed);
        long long cacheLocalUs = stats.cacheLocalUs.load(std::memory_order_relaxed);
        long long workerUs = stats.workerUs.load(std::memory_order_relaxed);
        long long metaSyncUs = stats.metaSyncUs.load(std::memory_order_relaxed);
        long long samplingUs = stats.samplingUs.load(std::memory_order_relaxed);
        long long totalUs = stats.totalUs.load(std::memory_order_relaxed);
        long long knownUs = setupUs + weightPrepUs + inputPrepUs + embeddingUs +
                            cacheLocalUs + workerUs + metaSyncUs + samplingUs;
        long long otherUs = std::max(0LL, totalUs - knownUs);
        auto avgMs = [&](long long us) {
            return samples > 0 ? (double)us / 1000.0 / (double)samples : 0.0;
        };
        auto avgCount = [&](long long count) {
            return samples > 0 ? (double)count / (double)samples : 0.0;
        };
        printf("[Qwen3.5 MTP target profile] samples=%lld tp_calls=%lld "
               "avg={seq_tokens=%.2f,logit_rows=%.2f} "
               "avg_ms={total=%.3f,setup=%.3f,weight_prep=%.3f,input_prep=%.3f,"
               "embedding=%.3f,cache_local=%.3f,worker=%.3f,meta_sync=%.3f,"
               "sampling=%.3f,other=%.3f}.\n",
               samples,
               stats.tensorParallelCalls.load(std::memory_order_relaxed),
               avgCount(stats.seqTokens.load(std::memory_order_relaxed)),
               avgCount(stats.logitRows.load(std::memory_order_relaxed)),
               avgMs(totalUs), avgMs(setupUs), avgMs(weightPrepUs),
               avgMs(inputPrepUs), avgMs(embeddingUs), avgMs(cacheLocalUs),
               avgMs(workerUs), avgMs(metaSyncUs), avgMs(samplingUs),
               avgMs(otherUs));
        fflush(stdout);
    }

    static void Qwen35MtpTargetProfileRecord(
            int interval, bool tensorParallel, int seqTokens, int logitRows,
            long long setupUs, long long weightPrepUs, long long inputPrepUs,
            long long embeddingUs, long long cacheLocalUs, long long workerUs,
            long long metaSyncUs, long long samplingUs, long long totalUs) {
        if (interval <= 0) {
            return;
        }
        Qwen35MtpTargetProfileAggregate &stats = Qwen35MtpTargetProfileStats();
        long long samples = stats.samples.fetch_add(1, std::memory_order_relaxed) + 1;
        if (tensorParallel) {
            stats.tensorParallelCalls.fetch_add(1, std::memory_order_relaxed);
        }
        stats.seqTokens.fetch_add(std::max(0, seqTokens), std::memory_order_relaxed);
        stats.logitRows.fetch_add(std::max(0, logitRows), std::memory_order_relaxed);
        stats.setupUs.fetch_add(setupUs, std::memory_order_relaxed);
        stats.weightPrepUs.fetch_add(weightPrepUs, std::memory_order_relaxed);
        stats.inputPrepUs.fetch_add(inputPrepUs, std::memory_order_relaxed);
        stats.embeddingUs.fetch_add(embeddingUs, std::memory_order_relaxed);
        stats.cacheLocalUs.fetch_add(cacheLocalUs, std::memory_order_relaxed);
        stats.workerUs.fetch_add(workerUs, std::memory_order_relaxed);
        stats.metaSyncUs.fetch_add(metaSyncUs, std::memory_order_relaxed);
        stats.samplingUs.fetch_add(samplingUs, std::memory_order_relaxed);
        stats.totalUs.fetch_add(totalUs, std::memory_order_relaxed);
        Qwen35MtpTargetProfilePrintIfNeeded(interval, stats, samples);
    }

    static void Qwen35MtpWorkerProfilePrintIfNeeded(
            int interval, Qwen35MtpWorkerProfileAggregate &stats,
            long long samples) {
        if (interval <= 0 || samples <= 0 || samples % interval != 0) {
            return;
        }
        long long setupUs = stats.setupUs.load(std::memory_order_relaxed);
        long long layersUs = stats.layersUs.load(std::memory_order_relaxed);
        long long headUs = stats.headUs.load(std::memory_order_relaxed);
        long long totalUs = stats.totalUs.load(std::memory_order_relaxed);
        long long otherUs = std::max(0LL, totalUs - setupUs - layersUs - headUs);
        auto avgMs = [&](long long us) {
            return samples > 0 ? (double)us / 1000.0 / (double)samples : 0.0;
        };
        auto avgCount = [&](long long count) {
            return samples > 0 ? (double)count / (double)samples : 0.0;
        };
        printf("[Qwen3.5 MTP worker profile] samples=%lld first_rank=%lld "
               "avg={seq_tokens=%.2f} "
               "avg_ms={total=%.3f,setup=%.3f,layers=%.3f,head=%.3f,other=%.3f}.\n",
               samples,
               stats.firstRankCalls.load(std::memory_order_relaxed),
               avgCount(stats.seqTokens.load(std::memory_order_relaxed)),
               avgMs(totalUs), avgMs(setupUs), avgMs(layersUs), avgMs(headUs),
               avgMs(otherUs));
        fflush(stdout);
    }

    static void Qwen35MtpWorkerProfileRecord(
            int interval, bool firstRank, int seqTokens,
            long long setupUs, long long layersUs, long long headUs,
            long long totalUs) {
        if (interval <= 0) {
            return;
        }
        Qwen35MtpWorkerProfileAggregate &stats = Qwen35MtpWorkerProfileStats();
        long long samples = stats.samples.fetch_add(1, std::memory_order_relaxed) + 1;
        if (firstRank) {
            stats.firstRankCalls.fetch_add(1, std::memory_order_relaxed);
        }
        stats.seqTokens.fetch_add(std::max(0, seqTokens), std::memory_order_relaxed);
        stats.setupUs.fetch_add(setupUs, std::memory_order_relaxed);
        stats.layersUs.fetch_add(layersUs, std::memory_order_relaxed);
        stats.headUs.fetch_add(headUs, std::memory_order_relaxed);
        stats.totalUs.fetch_add(totalUs, std::memory_order_relaxed);
        Qwen35MtpWorkerProfilePrintIfNeeded(interval, stats, samples);
    }

    static void Qwen35MoeCopyLinearWeightMeta(Data &dst, const Data &src, const std::string &name) {
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.tpLinearType = src.tpLinearType;
        dst.tpPackType = src.tpPackType;
    }

    static size_t Qwen35MoeBytesPerRow(const Data &weight, int columns) {
        return GetDataBytes(weight.dataType, 1, columns);
    }

    static bool Qwen35MoeCheckFp8ScaleRows(const Data &weight, int rowStart, int rows) {
        if (weight.dataType != DataType::FP8_E4M3) {
            return true;
        }
        if (weight.blockK <= 0 || weight.blockM <= 0 || weight.scales.empty() ||
            weight.dims.size() != 2) {
            return false;
        }
        int cols = weight.dims[1];
        int totalRows = weight.dims[0];
        int ms = (cols - 1) / weight.blockM + 1;
        int scaleRows = (totalRows - 1) / weight.blockK + 1;
        int scaleOffset = (rowStart / weight.blockK) * ms;
        int scaleCount = ((rows - 1) / weight.blockK + 1) * ms;
        return rowStart >= 0 && rows > 0 && rowStart + rows <= totalRows &&
               rowStart % weight.blockK == 0 &&
               scaleOffset + scaleCount <= (int)weight.scales.size() &&
               scaleRows * ms <= (int)weight.scales.size();
    }

    static void Qwen35MoeAppendFp8ScaleRows(Data &dst, const Data &src, int rowStart, int rows) {
        if (src.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(Qwen35MoeCheckFp8ScaleRows(src, rowStart, rows),
                        "Qwen3.5 MoE fused FP8 scale slice is out of bounds.");
        int cols = src.dims[1];
        int ms = (cols - 1) / src.blockM + 1;
        int scaleOffset = (rowStart / src.blockK) * ms;
        int scaleCount = ((rows - 1) / src.blockK + 1) * ms;
        dst.scales.insert(dst.scales.end(),
                          src.scales.begin() + scaleOffset,
                          src.scales.begin() + scaleOffset + scaleCount);
    }

    static void Qwen35MoeCopyRows(Data &dst, int dstRowStart,
                                  Data &src, int srcRowStart, int rows) {
        AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 2,
                        "Qwen3.5 MoE fused row copy expects 3D destination and 2D source.");
        int cols = src.dims[1];
        AssertInFastLLM(dst.dims[2] == cols &&
                        srcRowStart >= 0 && rows > 0 && srcRowStart + rows <= src.dims[0],
                        "Qwen3.5 MoE fused row copy shape mismatch.");
        int dstRows = dst.dims[0] * dst.dims[1];
        AssertInFastLLM(dstRowStart >= 0 && dstRowStart + rows <= dstRows,
                        "Qwen3.5 MoE fused destination row range is out of bounds.");
        src.ToDevice(DataDevice::CPU);
        AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                        "Qwen3.5 MoE fused row copy requires CPU buffers.");
        size_t bytesPerRow = Qwen35MoeBytesPerRow(src, cols);
        memcpy(dst.cpuData + (size_t)dstRowStart * bytesPerRow,
               src.cpuData + (size_t)srcRowStart * bytesPerRow,
               (size_t)rows * bytesPerRow);
    }

    static bool Qwen35MoeCanBuildFusedLayer(const std::unordered_map<std::string, Data> &allWeights,
                                            int layer, int numExperts) {
        if (numExperts <= 0) {
            return false;
        }
        const Data *gateup0 = nullptr;
        const Data *down0 = nullptr;
        int inter = 0, hidden = 0;
        for (int expert = 0; expert < numExperts; expert++) {
            std::string prefix = Qwen35MoeExpertPrefix(layer, expert);
            auto gateupIt = allWeights.find(prefix + "gateup_proj.weight");
            auto downIt = allWeights.find(prefix + "down_proj.weight");
            if (gateupIt == allWeights.end() || downIt == allWeights.end()) {
                return false;
            }
            const Data &gateup = gateupIt->second;
            const Data &down = downIt->second;
            if (gateup.isDiskWeight || down.isDiskWeight ||
                gateup.cpuData == nullptr || down.cpuData == nullptr ||
                gateup.dims.size() != 2 || down.dims.size() != 2 ||
                gateup.dims[0] <= 0 || gateup.dims[1] <= 0 ||
                (gateup.dims[0] & 1) != 0 ||
                !Qwen35MoeIsFusedFp8Type(gateup.dataType) ||
                gateup.dataType != down.dataType) {
                return false;
            }
            int curInter = gateup.dims[0] / 2;
            int curHidden = gateup.dims[1];
            if (down.dims[0] != curHidden || down.dims[1] != curInter) {
                return false;
            }
            if (gateup.dataType == DataType::FP8_E4M3 &&
                (!Qwen35MoeCheckFp8ScaleRows(gateup, 0, curInter) ||
                 !Qwen35MoeCheckFp8ScaleRows(gateup, curInter, curInter) ||
                 !Qwen35MoeCheckFp8ScaleRows(down, 0, curHidden))) {
                return false;
            }
            if (expert == 0) {
                gateup0 = &gateup;
                down0 = &down;
                inter = curInter;
                hidden = curHidden;
            } else if (gateup.dataType != gateup0->dataType ||
                       gateup.dims != gateup0->dims ||
                       gateup.blockK != gateup0->blockK ||
                       gateup.blockM != gateup0->blockM ||
                       down.dataType != down0->dataType ||
                       down.dims != down0->dims ||
                       down.blockK != down0->blockK ||
                       down.blockM != down0->blockM ||
                       curInter != inter || curHidden != hidden) {
                return false;
            }
        }
        return true;
    }

    static void Qwen35MoeBuildFusedLayerWeight(std::unordered_map<std::string, Data> &allWeights,
                                               int layer, int numExperts, const std::string &kind,
                                               Data *&weightPtr) {
        std::string prefix0 = Qwen35MoeExpertPrefix(layer, 0);
        Data &gateup0 = allWeights[prefix0 + "gateup_proj.weight"];
        Data &down0 = allWeights[prefix0 + "down_proj.weight"];
        int inter = gateup0.dims[0] / 2;
        int hidden = gateup0.dims[1];
        bool isDown = kind == "down";
        AssertInFastLLM(kind == "gate" || kind == "up" || kind == "down",
                        "Qwen3.5 MoE fused layer weight kind is invalid.\n");

        std::string fusedName = Qwen35MoeFusedWeightName(layer, kind);
        if (isDown) {
            allWeights[fusedName] = Data(down0.dataType, {numExperts, hidden, inter});
        } else {
            allWeights[fusedName] = Data(gateup0.dataType, {numExperts, inter, hidden});
        }

        Data &fused = allWeights[fusedName];
        Qwen35MoeCopyLinearWeightMeta(fused, isDown ? down0 : gateup0, fusedName);
        fused.Allocate(false);
        fused.scales.clear();

        for (int expert = 0; expert < numExperts; expert++) {
            std::string expertPrefix = Qwen35MoeExpertPrefix(layer, expert);
            Data &gateup = allWeights[expertPrefix + "gateup_proj.weight"];
            Data &down = allWeights[expertPrefix + "down_proj.weight"];
            if (kind == "gate") {
                Qwen35MoeCopyRows(fused, expert * inter, gateup, 0, inter);
                Qwen35MoeAppendFp8ScaleRows(fused, gateup, 0, inter);
            } else if (kind == "up") {
                Qwen35MoeCopyRows(fused, expert * inter, gateup, inter, inter);
                Qwen35MoeAppendFp8ScaleRows(fused, gateup, inter, inter);
            } else {
                Qwen35MoeCopyRows(fused, expert * hidden, down, 0, hidden);
                Qwen35MoeAppendFp8ScaleRows(fused, down, 0, hidden);
            }
        }
        weightPtr = &fused;
    }

    static void Qwen35MoeResizeFusedFp8Scales(Data &weight) {
        if (weight.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(weight.dims.size() == 3 && weight.blockK > 0 && weight.blockM > 0,
                        "Qwen3.5 MoE fused FP8 scale allocation got invalid metadata.\n");
        int experts = weight.dims[0];
        int rowsPerExpert = weight.dims[1];
        int cols = weight.dims[2];
        int scaleRowsPerExpert = (rowsPerExpert - 1) / weight.blockK + 1;
        int scaleCols = (cols - 1) / weight.blockM + 1;
        size_t scaleCount = (size_t)experts * scaleRowsPerExpert * scaleCols;
        if (weight.scales.size() != scaleCount) {
            weight.scales.assign(scaleCount, 0.0f);
        }
    }

    static void Qwen35MoeCopyFp8ScaleRowsToExpert(Data &dst, const Data &src,
                                                  int expert, int srcRowStart, int rows) {
        if (src.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(dst.dataType == DataType::FP8_E4M3 &&
                        dst.dims.size() == 3 && src.dims.size() == 2 &&
                        dst.blockK == src.blockK && dst.blockM == src.blockM &&
                        expert >= 0 && expert < dst.dims[0],
                        "Qwen3.5 MoE fused FP8 scale copy got incompatible metadata.\n");
        AssertInFastLLM(Qwen35MoeCheckFp8ScaleRows(src, srcRowStart, rows),
                        "Qwen3.5 MoE fused FP8 scale source is not ready.\n");
        int cols = src.dims[1];
        int dstRowsPerExpert = dst.dims[1];
        int scaleCols = (cols - 1) / src.blockM + 1;
        int srcScaleOffset = (srcRowStart / src.blockK) * scaleCols;
        int scaleRowCount = (rows - 1) / src.blockK + 1;
        int dstScaleRowsPerExpert = (dstRowsPerExpert - 1) / dst.blockK + 1;
        size_t dstOffset = ((size_t)expert * dstScaleRowsPerExpert) * scaleCols;
        size_t count = (size_t)scaleRowCount * scaleCols;
        AssertInFastLLM(dstOffset + count <= dst.scales.size() &&
                        srcScaleOffset + count <= src.scales.size(),
                        "Qwen3.5 MoE fused FP8 scale copy is out of bounds.\n");
        memcpy(dst.scales.data() + dstOffset,
               src.scales.data() + srcScaleOffset,
               count * sizeof(float));
    }

    static void Qwen35MoeInitFusedLayerWeightMeta(std::unordered_map<std::string, Data> &allWeights,
                                                  int layer, int numExperts, const std::string &kind,
                                                  const Data &source, int rowsPerExpert, int columns,
                                                  Data *&weightPtr) {
        std::string fusedName = Qwen35MoeFusedWeightName(layer, kind);
        allWeights[fusedName] = Data(source.dataType, {numExperts, rowsPerExpert, columns});
        Data &fused = allWeights[fusedName];
        Qwen35MoeCopyLinearWeightMeta(fused, source, fusedName);
        weightPtr = &fused;
    }

    static void Qwen35MoeAllocateFusedWeightForLoad(Data *weight) {
        if (weight == nullptr || weight->cpuData != nullptr ||
            weight->multiDeviceData || weight->dataDevice != DataDevice::CPU) {
            return;
        }
        weight->Allocate(false);
    }

    static void Qwen35MoeEnsureFusedLayerWeight(std::unordered_map<std::string, Data> &allWeights,
                                                int layer, int numExperts, const std::string &kind,
                                                const Data &source, int rowsPerExpert, int columns,
                                                Data *&weightPtr) {
        if (weightPtr == nullptr) {
            Qwen35MoeInitFusedLayerWeightMeta(allWeights, layer, numExperts, kind,
                                              source, rowsPerExpert, columns, weightPtr);
        } else {
            Qwen35MoeCopyLinearWeightMeta(*weightPtr, source, weightPtr->name);
            AssertInFastLLM(weightPtr->dims.size() == 3 &&
                            weightPtr->dims[0] == numExperts &&
                            weightPtr->dims[1] == rowsPerExpert &&
                            weightPtr->dims[2] == columns &&
                            weightPtr->dataType == source.dataType,
                            "Qwen3.5 MoE fused weight metadata does not match source weight.\n");
        }
        Qwen35MoeAllocateFusedWeightForLoad(weightPtr);
        Qwen35MoeResizeFusedFp8Scales(*weightPtr);
    }

    static void Qwen35MoeReleaseConsumedSourceWeight(Data &weight) {
        weight.FreeSpace();
        weight.scales.clear();
        weight.scales.shrink_to_fit();
        weight.mins.clear();
        weight.mins.shrink_to_fit();
        weight.zeros.clear();
        weight.zeros.shrink_to_fit();
        weight.halfScales.clear();
        weight.halfScales.shrink_to_fit();
        weight.perChannelsConfigs.clear();
        weight.perChannelsConfigs.shrink_to_fit();
        weight.weightSum.clear();
        weight.weightSum.shrink_to_fit();
    }

#ifdef USE_CUDA
    namespace {
        static std::atomic<int> qwen35ThreadTpNextPagedCacheBase(3000000);

        static std::string Qwen35TrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool Qwen35IsDisabledTpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool Qwen35NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static bool Qwen35IsLogitsEnvEnabled(const char *value) {
            return value != nullptr && value[0] != '\0' &&
                   !(value[0] == '0' && value[1] == '\0');
        }

        static bool Qwen35ShouldPrintLogits() {
            return GetFastllmEnv().printLogits ||
                   Qwen35IsLogitsEnvEnabled(std::getenv("FASTLLM_PRINT_LOGITS"));
        }

        static void Qwen35PrintTopKRows(const char *tag, const float *topkData, int batch, int topK) {
            printf("%s top%d logits:\n", tag, topK);
            for (int b = 0; b < batch; b++) {
                printf("  batch %d:", b);
                const float *row = topkData + (size_t)b * topK * 2;
                for (int k = 0; k < topK; k++) {
                    printf(" %d:%g", (int)(row[k * 2] + 1e-3f), row[k * 2 + 1]);
                }
                printf("\n");
            }
            fflush(stdout);
        }

        static void Qwen35PrintCudaLogitsTopK(Data &logits, int batch, const char *tag) {
            if (!Qwen35ShouldPrintLogits() || logits.dims.empty()) {
                return;
            }
            int vocabSize = logits.dims.back();
            int topK = std::min(10, vocabSize);
            Data topk;
            TopK(logits, topk, topK);
            topk.ToDevice(DataDevice::CPU);
            Qwen35PrintTopKRows(tag, (const float*)topk.cpuData, batch, topK);
        }

        static void Qwen35PrintCpuLogitsTopK(Data &logits, int batch, const char *tag) {
            if (!Qwen35ShouldPrintLogits() || logits.dims.empty() || logits.cpuData == nullptr) {
                return;
            }
            int vocabSize = logits.dims.back();
            int topK = std::min(10, vocabSize);
            std::vector<float> topkData((size_t)batch * topK * 2);
            const float *logitsData = (const float*)logits.cpuData;
            for (int b = 0; b < batch; b++) {
                std::vector<std::pair<float, int>> best;
                best.reserve(topK);
                const float *row = logitsData + (size_t)b * vocabSize;
                for (int i = 0; i < vocabSize; i++) {
                    float value = row[i];
                    int insertPos = 0;
                    while (insertPos < (int)best.size() && best[insertPos].first >= value) {
                        insertPos++;
                    }
                    if (insertPos < topK) {
                        best.insert(best.begin() + insertPos, {value, i});
                        if ((int)best.size() > topK) {
                            best.pop_back();
                        }
                    }
                }
                float *out = topkData.data() + (size_t)b * topK * 2;
                for (int k = 0; k < topK; k++) {
                    out[k * 2] = best[k].second;
                    out[k * 2 + 1] = best[k].first;
                }
            }
            Qwen35PrintTopKRows(tag, topkData.data(), batch, topK);
        }

        static bool AppendQwen35CudaDevicesFromSpec(const std::string &spec,
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

        static bool ParseQwen35GPUForwardSpec(const std::string &rawSpec,
                                              std::vector<int> &devices,
                                              std::map<int, int> &ratios) {
            std::string spec = Qwen35TrimString(rawSpec);
            if (Qwen35IsDisabledTpSpec(spec)) {
                return false;
            }

            std::string lower = spec;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            if (lower == "1" || lower == "true" || lower == "on" || lower == "auto") {
                int count = FastllmCudaGetDeviceCount();
                for (int i = 0; i < count; i++) {
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
            return AppendQwen35CudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetQwen35GPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                               std::vector<int> &devices,
                                               std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env == nullptr || Qwen35IsDisabledTpSpec(Qwen35TrimString(env))) {
                env = std::getenv("FASTLLM_QWEN35_THREAD_TP");
            }
            if (env != nullptr) {
                ParseQwen35GPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        AppendQwen35CudaDevicesFromSpec(it.first, "multicuda", it.second, devices, ratios);
                    }
                }
            }
            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "cuda")) {
                        AppendQwen35CudaDevicesFromSpec(it.first, "cuda", it.second, devices, ratios);
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

        static bool Qwen35DeviceSpecStartsWith(const std::string &device, const std::string &prefix) {
            std::string lower = device;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return lower == prefix || lower.rfind(prefix + ":", 0) == 0;
        }

        static bool Qwen35DeviceSpecIsCuda(const std::string &device) {
            return Qwen35DeviceSpecStartsWith(device, "cuda") ||
                   Qwen35DeviceSpecStartsWith(device, "multicuda");
        }

        static bool Qwen35DeviceMapAllCuda(const std::map<std::string, int> &deviceMap) {
            bool hasCuda = false;
            for (auto &it : deviceMap) {
                if (it.second <= 0) {
                    continue;
                }
                if (!Qwen35DeviceSpecIsCuda(it.first)) {
                    return false;
                }
                hasCuda = true;
            }
            return hasCuda;
        }

        static bool Qwen35MoeDeviceMapAllowsCudaOnly(const std::map<std::string, int> &moeDeviceMap) {
            return moeDeviceMap.empty() || Qwen35DeviceMapAllCuda(moeDeviceMap);
        }

        static bool Qwen35SelectedDeviceIsCudaOrEmpty(const std::string &device) {
            return device.empty() || Qwen35DeviceSpecIsCuda(device);
        }

        static bool Qwen35LayerUsesMappedNonCudaMoe(const Qwen3_5Model *model, int layer) {
            return model != nullptr &&
                   !Qwen35SelectedDeviceIsCudaOrEmpty(model->SelectMoeDeviceForLayer(layer));
        }

        static bool Qwen35ModelMoeLayersAllowCudaOnly(const Qwen3_5Model *model) {
            if (model == nullptr) {
                return true;
            }
            for (int i = 0; i < model->block_cnt; i++) {
                if (Qwen35LayerUsesMappedNonCudaMoe(model, i)) {
                    return false;
                }
            }
            return true;
        }

        static bool Qwen35CanUseGPUForward(const std::map<std::string, int> &deviceMap,
                                           const std::map<std::string, int> &moeDeviceMap) {
            (void)moeDeviceMap;
            std::vector<int> devices;
            std::map<int, int> ratios;
            return GetQwen35GPUForwardDevices(deviceMap, devices, ratios);
        }

        static bool Qwen35CanPlanFusedMoe(const std::map<std::string, int> &deviceMap,
                                          const std::map<std::string, int> &moeDeviceMap) {
            return Qwen35CanUseGPUForward(deviceMap, moeDeviceMap) &&
                   Qwen35MoeDeviceMapAllowsCudaOnly(moeDeviceMap);
        }

        static bool GetQwen35ThreadTpDevices(const std::map<std::string, int> &deviceMap,
                                             std::vector<int> &devices,
                                             std::map<int, int> &ratios) {
            if (!GetQwen35GPUForwardDevices(deviceMap, devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static DataType ResolveQwen35ThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT16 || modelType == DataType::BFLOAT16) {
                return modelType;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveQwen35ThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FLOAT16 ||
                cacheType == DataType::BFLOAT16 ||
                cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static void PrepareQwen35EmbeddingWeightType(Data &embedWeight,
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

        static void Qwen35CpuEmbeddingDirect(Data &inputIds, Data &embedWeight,
                                             Data &hiddenStates, DataType outputType) {
            PrepareQwen35EmbeddingWeightType(embedWeight, outputType, true);
            inputIds.ToDevice(DataDevice::CPU);
            Executor *executor = (Executor*)GetExecutor();
            executor->RunOnDevice("cpu", "EmbeddingDirect",
                                  DataDict{{"input", &inputIds},
                                           {"weight", &embedWeight},
                                           {"output", &hiddenStates}},
                                  FloatDict(), IntDict());
        }

        static void PrepareQwen35CudaEmbeddingWeightType(Data &embedWeight,
                                                         DataType outputType) {
            if (embedWeight.dataType != outputType) {
                embedWeight.ResetMultiDeviceState();
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static Data *CreateQwen35CudaReplicaLike(const Data &source, int device) {
            Data *local = new Data(source.dataType);
            local->Resize(source.dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            local->name = source.name;
            local->isModelWeight = source.isModelWeight;
            if (local->Count(0) > 0) {
                FastllmCudaSetDevice(device);
                local->Allocate();
            }
            return local;
        }

        static void PrepareQwen35CpuEmbeddingHiddenStates(
                Data &hiddenStates,
                const std::vector<int> &devices,
                PersistentWorkerGroup &workerGroup) {
            if (devices.empty()) {
                return;
            }

            uint64_t count = hiddenStates.Count(0);
            AssertInFastLLM(count <= (uint64_t)INT_MAX,
                            "Qwen3.5 ForwardGPU CPU embedding result is too large for NCCL broadcast.\n");
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
                    CreateQwen35CudaReplicaLike(hiddenStates, device);
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
                                "Qwen3.5 ForwardGPU CPU embedding missing local CUDA replica.\n");
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

        static Data *EnsureQwen35ThreadTpLocalCache(Data &root, int device, DataType localDataType) {
            root.multiDeviceData = true;
            root.dataDevice = DataDevice::CUDA;
            auto &local = root.multiDeviceDatas[device];
            if (local == nullptr) {
                local = new Data(localDataType);
                local->SetKVCache();
                local->cacheUid = root.cacheUid;
                local->isLinearAttention = root.isLinearAttention;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
            } else if (local->dataType != localDataType && local->dims.empty()) {
                local->dataType = localDataType;
                local->UpdateUnitSize();
            }
            return local;
        }

        static void PrepareQwen35SingleCudaCache(Data &cache, int device, DataType localDataType) {
            cache.isKVCache = true;
            cache.lockInCPU = false;
            if (cache.dataType != localDataType && cache.dims.empty()) {
                cache.dataType = localDataType;
                cache.UpdateUnitSize();
            }
            cache.ToDevice(DataDevice::CUDA, {device}, false);
        }

        static Executor &Qwen35ThreadLocalGenericExecutor() {
            static thread_local std::unique_ptr<Executor> executor;
            if (executor == nullptr) {
                executor.reset(new Executor());
            }
            return *executor;
        }

        class Qwen35ScopedGenericExecutor {
        public:
            explicit Qwen35ScopedGenericExecutor(const std::string &firstDevice)
                    : oldExecutor(GetExecutor()) {
                Executor &executor = Qwen35ThreadLocalGenericExecutor();
                if (!firstDevice.empty()) {
                    executor.SetFirstDevice(firstDevice);
                }
                SetCurrentThreadExecutor(&executor);
            }

            ~Qwen35ScopedGenericExecutor() {
                SetCurrentThreadExecutor(oldExecutor);
            }

            Qwen35ScopedGenericExecutor(const Qwen35ScopedGenericExecutor&) = delete;
            Qwen35ScopedGenericExecutor &operator=(const Qwen35ScopedGenericExecutor&) = delete;

        private:
            void *oldExecutor;
        };

        static void Qwen35ResetCpuScratch(Data &data) {
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
            Qwen3CudaClearMultiDeviceState(data);
            data.dataDevice = DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.lockInCPU = false;
            data.expansionDims.clear();
        }

        static void Qwen35ZeroCudaLike(Data &dst, const Data &like, int device) {
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
                Qwen3CudaClearMultiDeviceState(dst);
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

        static void Qwen35PrepareFusedMoeWeightForCuda(Data &weight, int device) {
            FastllmCudaSetDevice(device);
            weight.ToDevice(DataDevice::CUDA, {device}, true);
            if (weight.dataType == DataType::FP8_E4M3 && weight.extraCudaData.empty()) {
                AssertInFastLLM(!weight.scales.empty(),
                                "Qwen3.5 MoE FusedMOE FP8 weight has no scales.\n");
                float *cudaScales = (float*)FastllmCudaMalloc(weight.scales.size() * sizeof(float));
                FastllmCudaCopyFromHostToDevice(cudaScales, (void*)weight.scales.data(),
                                                weight.scales.size() * sizeof(float));
                weight.extraCudaData.push_back((void*)cudaScales);
                weight.scales.clear();
                weight.scales.shrink_to_fit();
            }
        }

        static int Qwen35GcdInt(int a, int b) {
            a = a < 0 ? -a : a;
            b = b < 0 ? -b : b;
            while (b != 0) {
                int t = a % b;
                a = b;
                b = t;
            }
            return a == 0 ? 1 : a;
        }

        static int Qwen35LcmInt(int a, int b) {
            a = std::max(1, a);
            b = std::max(1, b);
            return a / Qwen35GcdInt(a, b) * b;
        }

        static int Qwen35FusedInterSplitUnit(const Data &weight) {
            int unit = weight.groupCnt <= 0 ? 128 : weight.groupCnt;
            if (weight.dataType == DataType::FP8_E4M3) {
                if (weight.blockK > 0) {
                    unit = Qwen35LcmInt(unit, weight.blockK);
                }
                if (weight.blockM > 0) {
                    unit = Qwen35LcmInt(unit, weight.blockM);
                }
            } else if (weight.dataType == DataType::FP8_E4M3_BLOCK_128) {
                unit = 128;
            }
            return std::max(1, unit);
        }

        static DivisionScheme Qwen35BuildFusedInterScheme(const Data &gate,
                                                          const std::vector<int> &devices,
                                                          std::map<int, int> ratios) {
            AssertInFastLLM(gate.dims.size() == 3 && gate.dims[1] > 0,
                            "Qwen3.5 MoE fused TP split requires 3D gate weight.\n");
            std::vector<int> devCopy = devices;
            std::vector<int> points = FastllmMultiCudaGetSplitPoints(
                devCopy, ratios, gate.dims[1], Qwen35FusedInterSplitUnit(gate));
            AssertInFastLLM((int)points.size() == (int)devices.size() + 1,
                            "Qwen3.5 MoE fused TP split got invalid split points.\n");
            DivisionScheme scheme;
            for (int i = 0; i < (int)devices.size(); i++) {
                scheme[devices[i]];
                if (points[i] < points[i + 1]) {
                    scheme[devices[i]].push_back({points[i], points[i + 1]});
                }
            }
            return scheme;
        }

        static int Qwen35Fp8ScaleCols(int cols, int blockM) {
            return (cols - 1) / blockM + 1;
        }

        static void Qwen35AppendFp8ExpertRowScales(Data &dst, const Data &src,
                                                   int expert, int expertRows, int cols,
                                                   int rowStart, int rows) {
            if (src.dataType != DataType::FP8_E4M3 || rows <= 0) {
                return;
            }
            AssertInFastLLM(src.blockK > 0 && src.blockM > 0 && !src.scales.empty(),
                            "Qwen3.5 MoE fused TP FP8 weight has invalid scale metadata.\n");
            AssertInFastLLM(expert >= 0 && expert < src.dims[0] &&
                            rowStart >= 0 && rowStart + rows <= expertRows &&
                            rowStart % src.blockK == 0,
                            "Qwen3.5 MoE fused TP FP8 row scale slice is unaligned.\n");
            int scaleCols = Qwen35Fp8ScaleCols(cols, src.blockM);
            int scaleRowsPerExpert = (expertRows - 1) / src.blockK + 1;
            int scaleRowStart = expert * scaleRowsPerExpert + rowStart / src.blockK;
            int scaleRowCount = (rows - 1) / src.blockK + 1;
            size_t offset = (size_t)scaleRowStart * scaleCols;
            size_t count = (size_t)scaleRowCount * scaleCols;
            AssertInFastLLM(offset + count <= src.scales.size(),
                            "Qwen3.5 MoE fused TP FP8 row scale slice is out of bounds.\n");
            dst.scales.insert(dst.scales.end(),
                              src.scales.begin() + offset,
                              src.scales.begin() + offset + count);
        }

        static void Qwen35CopyFusedInterRows(Data &dst, Data &src,
                                             int interStart, int localInter) {
            AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 3,
                            "Qwen3.5 MoE fused TP row shard expects 3D weights.\n");
            int experts = src.dims[0], inter = src.dims[1], hidden = src.dims[2];
            AssertInFastLLM(dst.dims[0] == experts && dst.dims[1] == localInter &&
                            dst.dims[2] == hidden && interStart >= 0 &&
                            localInter >= 0 && interStart + localInter <= inter,
                            "Qwen3.5 MoE fused TP row shard shape mismatch.\n");
            if (localInter == 0) {
                return;
            }
            src.ToDevice(DataDevice::CPU);
            AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                            "Qwen3.5 MoE fused TP row shard requires CPU buffers.\n");
            size_t rowBytes = Qwen35MoeBytesPerRow(src, hidden);
            for (int expert = 0; expert < experts; expert++) {
                memcpy(dst.cpuData + (size_t)expert * localInter * rowBytes,
                       src.cpuData + ((size_t)expert * inter + interStart) * rowBytes,
                       (size_t)localInter * rowBytes);
                Qwen35AppendFp8ExpertRowScales(dst, src, expert, inter, hidden,
                                               interStart, localInter);
            }
        }

        static void Qwen35AppendFp8DownColumnScales(Data &dst, const Data &src,
                                                    int interStart, int localInter) {
            if (src.dataType != DataType::FP8_E4M3 || localInter <= 0) {
                return;
            }
            int experts = src.dims[0], hidden = src.dims[1], inter = src.dims[2];
            AssertInFastLLM(src.blockK > 0 && src.blockM > 0 && !src.scales.empty() &&
                            interStart >= 0 && interStart + localInter <= inter &&
                            interStart % src.blockM == 0,
                            "Qwen3.5 MoE fused TP FP8 column scale slice is unaligned.\n");
            int srcScaleCols = Qwen35Fp8ScaleCols(inter, src.blockM);
            int dstScaleCols = Qwen35Fp8ScaleCols(localInter, src.blockM);
            int scaleColStart = interStart / src.blockM;
            int scaleRowsPerExpert = (hidden - 1) / src.blockK + 1;
            for (int expert = 0; expert < experts; expert++) {
                for (int scaleRow = 0; scaleRow < scaleRowsPerExpert; scaleRow++) {
                    size_t offset = ((size_t)expert * scaleRowsPerExpert + scaleRow) *
                                    srcScaleCols + scaleColStart;
                    AssertInFastLLM(offset + dstScaleCols <= src.scales.size(),
                                    "Qwen3.5 MoE fused TP FP8 column scale slice is out of bounds.\n");
                    dst.scales.insert(dst.scales.end(),
                                      src.scales.begin() + offset,
                                      src.scales.begin() + offset + dstScaleCols);
                }
            }
        }

        static void Qwen35CopyFusedDownInterColumns(Data &dst, Data &src,
                                                    int interStart, int localInter) {
            AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 3,
                            "Qwen3.5 MoE fused TP down shard expects 3D weights.\n");
            int experts = src.dims[0], hidden = src.dims[1], inter = src.dims[2];
            AssertInFastLLM(dst.dims[0] == experts && dst.dims[1] == hidden &&
                            dst.dims[2] == localInter && interStart >= 0 &&
                            localInter >= 0 && interStart + localInter <= inter,
                            "Qwen3.5 MoE fused TP down shard shape mismatch.\n");
            if (localInter == 0) {
                return;
            }
            src.ToDevice(DataDevice::CPU);
            AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                            "Qwen3.5 MoE fused TP down shard requires CPU buffers.\n");
            int rows = experts * hidden;
            size_t srcRowBytes = Qwen35MoeBytesPerRow(src, inter);
            size_t dstRowBytes = Qwen35MoeBytesPerRow(dst, localInter);
            if (src.dataType == DataType::FP8_E4M3_BLOCK_128) {
                const int block = 128;
                const int blockBytes = block + (int)sizeof(float);
                AssertInFastLLM(interStart % block == 0,
                                "Qwen3.5 MoE fused TP FP8 block shard is unaligned.\n");
                int blockStart = interStart / block;
                int blockCount = (localInter + block - 1) / block;
                for (int row = 0; row < rows; row++) {
                    memcpy(dst.cpuData + (size_t)row * dstRowBytes,
                           src.cpuData + (size_t)row * srcRowBytes + (size_t)blockStart * blockBytes,
                           (size_t)blockCount * blockBytes);
                }
            } else {
                AssertInFastLLM(src.dataType == DataType::FP8_E4M3,
                                "Qwen3.5 MoE fused TP only supports FP8 fused weights.\n");
                for (int row = 0; row < rows; row++) {
                    memcpy(dst.cpuData + (size_t)row * dstRowBytes,
                           src.cpuData + (size_t)row * srcRowBytes + interStart,
                           (size_t)localInter);
                }
                Qwen35AppendFp8DownColumnScales(dst, src, interStart, localInter);
            }
        }

        static Data *Qwen35CreateFusedInterShard(Data &src, int axis,
                                                 int device, std::pair<int, int> range) {
            AssertInFastLLM(axis == 1 || axis == 2,
                            "Qwen3.5 MoE fused TP only splits inter dimension.\n");
            int localInter = range.second - range.first;
            AssertInFastLLM(localInter >= 0,
                            "Qwen3.5 MoE fused TP got invalid shard range.\n");
            std::vector<int> localDims = src.dims;
            localDims[axis] = localInter;
            Data *local = new Data(src.dataType, localDims);
            Qwen35MoeCopyLinearWeightMeta(*local, src,
                                          src.name + ".tp" + std::to_string(device));
            local->scales.clear();
            local->dataDeviceIds = {device};
            if (local->Count(0) > 0) {
                local->Allocate(false);
                if (axis == 1) {
                    Qwen35CopyFusedInterRows(*local, src, range.first, localInter);
                } else {
                    Qwen35CopyFusedDownInterColumns(*local, src, range.first, localInter);
                }
                Qwen35PrepareFusedMoeWeightForCuda(*local, device);
            } else {
                local->dataDevice = DataDevice::CUDA;
            }
            return local;
        }

        static bool Qwen35FusedShardLayoutReady(const Data &weight,
                                                const std::vector<int> &devices,
                                                const DivisionScheme &scheme,
                                                int axis) {
            if (!weight.multiDeviceData || weight.tpLayout != TP_LAYOUT_SHARDED ||
                weight.tpAxis != axis || weight.tpRanges != scheme) {
                return false;
            }
            for (int device : devices) {
                auto localIt = weight.multiDeviceDatas.find(device);
                auto rangeIt = scheme.find(device);
                if (localIt == weight.multiDeviceDatas.end() || localIt->second == nullptr ||
                    rangeIt == scheme.end()) {
                    return false;
                }
                int localInter = 0;
                for (auto &range : rangeIt->second) {
                    localInter += range.second - range.first;
                }
                Data *local = localIt->second;
                if (local->dims.size() != weight.dims.size() ||
                    local->dims[axis] != localInter ||
                    local->dataDevice != DataDevice::CUDA ||
                    local->dataDeviceIds.empty() || local->dataDeviceIds[0] != device) {
                    return false;
                }
                if (local->Count(0) > 0 &&
                    (local->cudaData == nullptr ||
                     (local->dataType == DataType::FP8_E4M3 && local->extraCudaData.empty()))) {
                    return false;
                }
            }
            return true;
        }

        static void Qwen35PrepareFusedShardedWeight(Data &weight,
                                                    const std::vector<int> &devices,
                                                    const DivisionScheme &scheme,
                                                    int axis) {
            if (Qwen35FusedShardLayoutReady(weight, devices, scheme, axis)) {
                return;
            }
            weight.ToDevice(DataDevice::CPU);
            Qwen3CudaClearMultiDeviceState(weight);
            std::map<int, Data*> localDatas;
            for (int device : devices) {
                auto rangeIt = scheme.find(device);
                AssertInFastLLM(rangeIt != scheme.end(),
                                "Qwen3.5 MoE fused TP missing device range.\n");
                std::pair<int, int> range = {0, 0};
                if (!rangeIt->second.empty()) {
                    AssertInFastLLM(rangeIt->second.size() == 1,
                                    "Qwen3.5 MoE fused TP expects contiguous per-device shards.\n");
                    range = rangeIt->second[0];
                }
                localDatas[device] = Qwen35CreateFusedInterShard(weight, axis, device, range);
            }
            weight.multiDeviceDatas.swap(localDatas);
            weight.multiDeviceData = true;
            weight.dataDevice = DataDevice::CUDA;
            weight.dataDeviceIds = devices;
            weight.tpLayout = TP_LAYOUT_SHARDED;
            weight.tpAxis = axis;
            weight.tpGlobalDims = weight.dims;
            weight.tpRanges = scheme;
            weight.cudaData = nullptr;
            weight.deviceData = nullptr;
            if (weight.cpuData != nullptr) {
                delete[] weight.cpuData;
                weight.cpuData = nullptr;
            }
            weight.scales.clear();
            weight.scales.shrink_to_fit();
        }

        static bool Qwen35HasLocalFusedMoeShard(Data *gate, Data *up, Data *down) {
            return gate != nullptr && up != nullptr && down != nullptr &&
                   gate->dims.size() == 3 && up->dims.size() == 3 && down->dims.size() == 3 &&
                   gate->dims[1] > 0 && up->dims[1] > 0 && down->dims[2] > 0 &&
                   gate->cudaData != nullptr && up->cudaData != nullptr && down->cudaData != nullptr;
        }

        static void Qwen35CudaFusedMOE(Qwen3CudaDirectRunner &runner,
                                       Data &input, Data &expertIndex, Data &expertScore,
                                       Data &gate, Data &up, Data &down, Data &w1,
                                       Data &output, int layer) {
            runner.Run("FusedMOE",
                       DataDict{{"input", &input}, {"index", &expertIndex}, {"score", &expertScore},
                                {"gate", &gate}, {"up", &up}, {"down", &down},
                                {"w1", &w1}, {"output", &output}},
                       FloatDict{{"swigluLimit", 0.0f}},
                       IntDict{{"layer", layer}, {"gateType", (int)MoeGateSwiglu}},
                       {"w1", "output"});
        }

        static bool Qwen35HasLocalMoeShard(const std::vector<Data*> &localWeights) {
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

        static void SyncQwen35ThreadTpRootCacheMetaFromLocal(
                Data &root,
                Data *firstLocal,
                const std::vector<int> &devices,
                const DivisionScheme &scheme,
                const std::vector<int> &globalDims,
                int axis,
                bool linearAttention) {
            if (firstLocal == nullptr || firstLocal->dims.empty()) {
                return;
            }
            if (!linearAttention && firstLocal->dims.size() < 3) {
                return;
            }

            bool samePageIndex = root.pageIndex.size() == firstLocal->pageIndex.size() &&
                (root.pageIndex.empty() || root.pageIndex.back() == firstLocal->pageIndex.back());
            bool fastMetaUpdate =
                root.multiDeviceData &&
                root.dataDevice == DataDevice::CUDA &&
                root.tpLayout == TP_LAYOUT_SHARDED &&
                root.tpAxis == axis &&
                root.isKVCache &&
                root.isLinearAttention == linearAttention &&
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

            root.multiDeviceData = true;
            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = axis;
            root.tpRanges = scheme;
            root.tpGlobalDims = globalDims;
            if (root.dims != globalDims) {
                root.Resize(globalDims);
            }
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isLinearAttention = linearAttention;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
        }

        static DivisionScheme ExtractQwen35FirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                if (!it.second.empty()) {
                    ret[it.first].push_back(it.second[0]);
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35GatedAttentionQkvScheme(
                std::vector<int> devices,
                std::map<int, int> ratios,
                int qHeads,
                int kvHeads,
                int headDim) {
            AssertInFastLLM(kvHeads > 0 && qHeads > 0 && headDim > 0 && qHeads % kvHeads == 0,
                            "Qwen3.5 gated attention TP requires valid head metadata.\n");
            DivisionScheme scheme;
            int group = qHeads / kvHeads;
            int qGateWidth = qHeads * headDim * 2;
            int kvWidth = kvHeads * headDim;
            std::vector<int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvHeads, 1);
            for (int i = 0; i < (int)devices.size(); i++) {
                int st = points[i], end = points[i + 1];
                scheme[devices[i]];
                if (st >= end) {
                    continue;
                }
                scheme[devices[i]].push_back({st * group * headDim * 2, end * group * headDim * 2});
                scheme[devices[i]].push_back({qGateWidth + st * headDim, qGateWidth + end * headDim});
                scheme[devices[i]].push_back({qGateWidth + kvWidth + st * headDim,
                                               qGateWidth + kvWidth + end * headDim});
            }
            return scheme;
        }

        static DivisionScheme ExtractQwen35AttentionKVHeadScheme(
                const DivisionScheme &qkvScheme,
                int qGateWidth,
                int headDim) {
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                if (it.second.size() < 2) {
                    continue;
                }
                int st = (it.second[1].first - qGateWidth) / headDim;
                int end = (it.second[1].second - qGateWidth) / headDim;
                ret[it.first].push_back({st, end});
            }
            return ret;
        }

        static DivisionScheme ExtractQwen35AttentionOutputScheme(
                const DivisionScheme &qkvScheme) {
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                if (it.second.empty()) {
                    continue;
                }
                ret[it.first].push_back({it.second[0].first / 2, it.second[0].second / 2});
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearKeyHeadScheme(
                std::vector<int> devices,
                std::map<int, int> ratios,
                int keyHeads) {
            AssertInFastLLM(keyHeads > 0, "Qwen3.5 linear attention TP requires key heads.\n");
            AssertInFastLLM((int)devices.size() <= keyHeads,
                            "Qwen3.5 linear attention TP currently requires devices <= linear key heads.\n");
            DivisionScheme scheme;
            std::vector<int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, keyHeads, 1);
            for (int i = 0; i < (int)devices.size(); i++) {
                scheme[devices[i]];
                if (points[i] < points[i + 1]) {
                    scheme[devices[i]].push_back({points[i], points[i + 1]});
                }
            }
            return scheme;
        }

        static DivisionScheme BuildQwen35LinearValueHeadScheme(
                const DivisionScheme &keyHeadScheme,
                int valueHeadsPerKey) {
            DivisionScheme ret;
            for (auto &it : keyHeadScheme) {
                ret[it.first];
                for (auto &range : it.second) {
                    ret[it.first].push_back({range.first * valueHeadsPerKey,
                                             range.second * valueHeadsPerKey});
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearQkvzbaScheme(
                const DivisionScheme &keyHeadScheme,
                int numKHeads,
                int numVHeads,
                int headKDim,
                int headVDim) {
            DivisionScheme ret;
            int valueHeadsPerKey = numVHeads / numKHeads;
            int kd = numKHeads * headKDim;
            int vd = numVHeads * headVDim;
            int qBase = 0;
            int kBase = kd;
            int vBase = kd * 2;
            int zBase = kd * 2 + vd;
            int bBase = kd * 2 + vd * 2;
            int aBase = bBase + numVHeads;
            for (auto &it : keyHeadScheme) {
                ret[it.first];
                for (auto &kr : it.second) {
                    int vs = kr.first * valueHeadsPerKey;
                    int ve = kr.second * valueHeadsPerKey;
                    ret[it.first].push_back({qBase + kr.first * headKDim, qBase + kr.second * headKDim});
                    ret[it.first].push_back({kBase + kr.first * headKDim, kBase + kr.second * headKDim});
                    ret[it.first].push_back({vBase + vs * headVDim, vBase + ve * headVDim});
                    ret[it.first].push_back({zBase + vs * headVDim, zBase + ve * headVDim});
                    ret[it.first].push_back({bBase + vs, bBase + ve});
                    ret[it.first].push_back({aBase + vs, aBase + ve});
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearQkvzScheme(
                const DivisionScheme &keyHeadScheme,
                int numKHeads,
                int numVHeads,
                int headKDim,
                int headVDim) {
            DivisionScheme ret;
            int valueHeadsPerKey = numVHeads / numKHeads;
            int kd = numKHeads * headKDim;
            int vd = numVHeads * headVDim;
            int qBase = 0;
            int kBase = kd;
            int vBase = kd * 2;
            int zBase = kd * 2 + vd;
            for (auto &it : keyHeadScheme) {
                ret[it.first];
                for (auto &kr : it.second) {
                    int vs = kr.first * valueHeadsPerKey;
                    int ve = kr.second * valueHeadsPerKey;
                    ret[it.first].push_back({qBase + kr.first * headKDim, qBase + kr.second * headKDim});
                    ret[it.first].push_back({kBase + kr.first * headKDim, kBase + kr.second * headKDim});
                    ret[it.first].push_back({vBase + vs * headVDim, vBase + ve * headVDim});
                    ret[it.first].push_back({zBase + vs * headVDim, zBase + ve * headVDim});
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearBaScheme(
                const DivisionScheme &valueHeadScheme,
                int numVHeads) {
            DivisionScheme ret;
            int bBase = 0;
            int aBase = numVHeads;
            for (auto &it : valueHeadScheme) {
                ret[it.first];
                for (auto &range : it.second) {
                    ret[it.first].push_back({bBase + range.first, bBase + range.second});
                    ret[it.first].push_back({aBase + range.first, aBase + range.second});
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearConvScheme(
                const DivisionScheme &keyHeadScheme,
                int numKHeads,
                int numVHeads,
                int headKDim,
                int headVDim) {
            DivisionScheme ret;
            int valueHeadsPerKey = numVHeads / numKHeads;
            int kd = numKHeads * headKDim;
            for (auto &it : keyHeadScheme) {
                ret[it.first];
                for (auto &kr : it.second) {
                    int vs = kr.first * valueHeadsPerKey;
                    int ve = kr.second * valueHeadsPerKey;
                    ret[it.first].push_back({kr.first * headKDim, kr.second * headKDim});
                    ret[it.first].push_back({kd + kr.first * headKDim, kd + kr.second * headKDim});
                    ret[it.first].push_back({kd * 2 + vs * headVDim, kd * 2 + ve * headVDim});
                }
            }
            return ret;
        }

        static DivisionScheme BuildQwen35LinearOutProjScheme(
                const DivisionScheme &valueHeadScheme,
                int headVDim) {
            DivisionScheme ret;
            for (auto &it : valueHeadScheme) {
                ret[it.first];
                for (auto &range : it.second) {
                    ret[it.first].push_back({range.first * headVDim, range.second * headVDim});
                }
            }
            return ret;
        }

        static int Qwen35RangeTotal(const DivisionScheme &scheme, int device) {
            int total = 0;
            auto it = scheme.find(device);
            if (it == scheme.end()) {
                return 0;
            }
            for (auto &range : it->second) {
                total += range.second - range.first;
            }
            return total;
        }

        static int Qwen35LocalHeads(const DivisionScheme &scheme, int device) {
            int heads = 0;
            auto it = scheme.find(device);
            if (it == scheme.end()) {
                return 0;
            }
            for (auto &range : it->second) {
                heads += range.second - range.first;
            }
            return heads;
        }

        static bool SplitQwen35VectorWeight(
                Data &data,
                std::vector<int> devices,
                const DivisionScheme &scheme) {
            if (data.dims.empty()) {
                return true;
            }
            if (data.multiDeviceData) {
                bool ready = true;
                for (int device : devices) {
                    ready &= data.multiDeviceDatas.find(device) != data.multiDeviceDatas.end() &&
                             data.multiDeviceDatas[device] != nullptr;
                }
                if (ready) {
                    return true;
                }
            }
            AssertInFastLLM(data.dims.size() == 1 && data.unitSizeDiv == 1,
                            "Qwen3.5 vector TP expects a dense 1D tensor.\n");
            int rootDevice = devices.empty() ? 0 : devices[0];
            if (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                data.dataDeviceIds.empty() || data.dataDeviceIds[0] != rootDevice) {
                data.ToDevice(DataDevice::CUDA, {rootDevice}, true);
            }
            Qwen3CudaClearMultiDeviceState(data);
            data.multiDeviceData = true;
            data.tpLayout = TP_LAYOUT_SHARDED;
            data.tpAxis = 0;
            data.tpGlobalDims = data.dims;
            data.tpRanges = scheme;
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = devices;

            size_t elemBytes = (size_t)data.unitSize / data.unitSizeDiv;
            for (int device : devices) {
                int len = Qwen35RangeTotal(scheme, device);
                Data *local = new Data(data.dataType, {len});
                local->name = data.name;
                local->isModelWeight = data.isModelWeight;
                local->weightType = data.weightType;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
                if (local->Count(0) > 0) {
                    FastllmCudaSetDevice(device);
                    local->Allocate();
                }
                data.multiDeviceDatas[device] = local;

                size_t dstOffset = 0;
                auto rangesIt = scheme.find(device);
                if (rangesIt == scheme.end()) {
                    continue;
                }
                for (auto &range : rangesIt->second) {
                    int len = range.second - range.first;
                    if (len <= 0) {
                        continue;
                    }
                    size_t bytes = (size_t)len * elemBytes;
                    FastllmCudaMemcpy2DDeviceToDeviceAuto(
                        (uint8_t*)local->cudaData + dstOffset,
                        bytes,
                        (uint8_t*)data.cudaData + (size_t)range.first * elemBytes,
                        bytes,
                        bytes,
                        1,
                        device,
                        rootDevice);
                    dstOffset += bytes;
                }
            }
            return true;
        }

        static bool SplitQwen35Conv1DWeight(
                Data &weight,
                Data &bias,
                std::vector<int> devices,
                const DivisionScheme &scheme) {
            if (weight.multiDeviceData) {
                bool ready = true;
                for (int device : devices) {
                    ready &= weight.multiDeviceDatas.find(device) != weight.multiDeviceDatas.end() &&
                             weight.multiDeviceDatas[device] != nullptr;
                }
                if (ready) {
                    return true;
                }
            }
            AssertInFastLLM(weight.dims.size() == 3 && weight.unitSizeDiv == 1,
                            "Qwen3.5 linear attention conv TP expects a dense 3D weight.\n");
            int rootDevice = devices.empty() ? 0 : devices[0];
            if (weight.dataDevice != DataDevice::CUDA || weight.cudaData == nullptr ||
                weight.dataDeviceIds.empty() || weight.dataDeviceIds[0] != rootDevice) {
                weight.ToDevice(DataDevice::CUDA, {rootDevice}, true);
            }
            Qwen3CudaClearMultiDeviceState(weight);
            weight.multiDeviceData = true;
            weight.tpLayout = TP_LAYOUT_SHARDED;
            weight.tpAxis = 0;
            weight.tpGlobalDims = weight.dims;
            weight.tpRanges = scheme;
            weight.dataDevice = DataDevice::CUDA;
            weight.dataDeviceIds = devices;

            size_t rowBytes = (size_t)weight.Count(1) * weight.unitSize;
            for (int device : devices) {
                int len = Qwen35RangeTotal(scheme, device);
                std::vector<int> localDims = weight.dims;
                localDims[0] = len;
                Data *local = new Data(weight.dataType, localDims);
                local->name = weight.name;
                local->isModelWeight = weight.isModelWeight;
                local->weightType = weight.weightType;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
                if (local->Count(0) > 0) {
                    FastllmCudaSetDevice(device);
                    local->Allocate();
                }
                weight.multiDeviceDatas[device] = local;

                size_t dstOffset = 0;
                auto rangesIt = scheme.find(device);
                if (rangesIt == scheme.end()) {
                    continue;
                }
                for (auto &range : rangesIt->second) {
                    int rows = range.second - range.first;
                    if (rows <= 0) {
                        continue;
                    }
                    FastllmCudaMemcpy2DDeviceToDeviceAuto(
                        (uint8_t*)local->cudaData + dstOffset,
                        (size_t)rows * rowBytes,
                        (uint8_t*)weight.cudaData + (size_t)range.first * rowBytes,
                        (size_t)rows * rowBytes,
                        (size_t)rows * rowBytes,
                        1,
                        device,
                        rootDevice);
                    dstOffset += (size_t)rows * rowBytes;
                }
            }
            weight.cudaData = nullptr;
            if (bias.dims.size() > 0) {
                SplitQwen35VectorWeight(bias, devices, scheme);
            }
            return true;
        }

        struct Qwen35ForwardSingleBuffers {
            Data embedOutput, hiddenStates, attenInput;
            Data merged, qgate, gate, q, k, v, attenOutput, attenLastOutput, qForAttentionHolder;
            Data kAppend, vAppend;
            Data gateupResult, swigluResult, mlpPart;
            Data routerLogits, routerLogitsTemp, expertIndex, expertScore;
            Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
            Data moeFinal, sharedGate, sharedOutput;
            Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
            Data gdnMerged, qkvConvInput, z, ba, b, a, g, convOutput, coreAttnOut;
            Data linearQ, linearK, linearV;
            Data qRepeat, kRepeat, qq, kkPad, vvPad, bbPad, ggPad, decayMask;
            Data kBeta, vBeta, attn, at, kCumdecay, gExp, coreTemp;
            Data logitsHalf;
            Data batchConvCache, batchRecurrentState;
            std::vector<Data*> batchPastKeys;
            std::vector<Data*> batchPastValues;
            std::vector<Data*> linearConvCaches;
            std::vector<Data*> recurrentStates;

            Qwen35ForwardSingleBuffers() :
                batchPastKeys(1), batchPastValues(1),
                linearConvCaches(1), recurrentStates(1) {}
        };

        struct Qwen35CudaGraphDecodeState {
            std::mutex mutex;
            std::string signature;
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            void *graph = nullptr;
            void *exec = nullptr;
            Data inputIds;
            Data positionIds;
            Qwen35ForwardSingleBuffers buffers;
            Data logits;
            Data linearSlotIds;
            std::vector<int> lastInsertIndexHost;
            std::vector<int> lastPageSizesHost;
            std::vector<int> lastPageIndexHost;
            std::vector<int> lastDecodePageLensHost;
            std::vector<int> lastLinearSlotIdsHost;
            std::vector<const Data*> lastPastKeyHosts;

            ~Qwen35CudaGraphDecodeState() {
                if (exec != nullptr) {
                    FastllmCudaGraphExecDestroy(exec);
                    exec = nullptr;
                }
                if (graph != nullptr) {
                    FastllmCudaGraphDestroy(graph);
                    graph = nullptr;
                }
            }
        };

        static void Qwen35DestroyCudaGraph(Qwen35CudaGraphDecodeState &state) {
            if (state.exec != nullptr) {
                FastllmCudaGraphExecDestroy(state.exec);
                state.exec = nullptr;
            }
            if (state.graph != nullptr) {
                FastllmCudaGraphDestroy(state.graph);
                state.graph = nullptr;
            }
            state.captured = false;
            state.warmed = false;
            state.lastInsertIndexHost.clear();
            state.lastPageSizesHost.clear();
            state.lastPageIndexHost.clear();
            state.lastDecodePageLensHost.clear();
            state.lastLinearSlotIdsHost.clear();
            state.lastPastKeyHosts.clear();
        }

        static void Qwen35AbortCudaGraphCapture() {
            void *capturedGraph = nullptr;
            if (FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
        }

        static void Qwen35DetachGraphDataStorage(Data &data) {
            data.pageIndex.clear();
            data.pagedKVCacheData = nullptr;
            data.isPagedKVCache = false;
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
            data.cpuData = nullptr;
            data.cudaData = nullptr;
            data.cudaDataBorrowed = false;
            data.deviceData = nullptr;
            data.extraCudaData.clear();
            data.extraCudaHalfData.clear();
            data.extraDeviceData.clear();
            data.expansionSize = 0;
            data.expansionBytes = 0;
            data.isFake = true;
        }

        static void Qwen35DetachGraphBuffers(Qwen35ForwardSingleBuffers &buf) {
            Qwen35DetachGraphDataStorage(buf.embedOutput);
            Qwen35DetachGraphDataStorage(buf.hiddenStates);
            Qwen35DetachGraphDataStorage(buf.attenInput);
            Qwen35DetachGraphDataStorage(buf.merged);
            Qwen35DetachGraphDataStorage(buf.qgate);
            Qwen35DetachGraphDataStorage(buf.gate);
            Qwen35DetachGraphDataStorage(buf.q);
            Qwen35DetachGraphDataStorage(buf.k);
            Qwen35DetachGraphDataStorage(buf.v);
            Qwen35DetachGraphDataStorage(buf.attenOutput);
            Qwen35DetachGraphDataStorage(buf.attenLastOutput);
            Qwen35DetachGraphDataStorage(buf.qForAttentionHolder);
            Qwen35DetachGraphDataStorage(buf.kAppend);
            Qwen35DetachGraphDataStorage(buf.vAppend);
            Qwen35DetachGraphDataStorage(buf.gateupResult);
            Qwen35DetachGraphDataStorage(buf.swigluResult);
            Qwen35DetachGraphDataStorage(buf.mlpPart);
            Qwen35DetachGraphDataStorage(buf.routerLogits);
            Qwen35DetachGraphDataStorage(buf.routerLogitsTemp);
            Qwen35DetachGraphDataStorage(buf.expertIndex);
            Qwen35DetachGraphDataStorage(buf.expertScore);
            Qwen35DetachGraphDataStorage(buf.w1);
            Qwen35DetachGraphDataStorage(buf.w2);
            Qwen35DetachGraphDataStorage(buf.w3);
            Qwen35DetachGraphDataStorage(buf.tempInput);
            Qwen35DetachGraphDataStorage(buf.tempOutput);
            Qwen35DetachGraphDataStorage(buf.moeInputTemp);
            Qwen35DetachGraphDataStorage(buf.moeOutputTemp);
            Qwen35DetachGraphDataStorage(buf.moeFinal);
            Qwen35DetachGraphDataStorage(buf.sharedGate);
            Qwen35DetachGraphDataStorage(buf.sharedOutput);
            Qwen35DetachGraphDataStorage(buf.qSizes);
            Qwen35DetachGraphDataStorage(buf.pageSizes);
            Qwen35DetachGraphDataStorage(buf.pageIndexs);
            Qwen35DetachGraphDataStorage(buf.lastPageLens);
            Qwen35DetachGraphDataStorage(buf.insertIndexs);
            Qwen35DetachGraphDataStorage(buf.insertPositions);
            Qwen35DetachGraphDataStorage(buf.gdnMerged);
            Qwen35DetachGraphDataStorage(buf.qkvConvInput);
            Qwen35DetachGraphDataStorage(buf.z);
            Qwen35DetachGraphDataStorage(buf.ba);
            Qwen35DetachGraphDataStorage(buf.b);
            Qwen35DetachGraphDataStorage(buf.a);
            Qwen35DetachGraphDataStorage(buf.g);
            Qwen35DetachGraphDataStorage(buf.convOutput);
            Qwen35DetachGraphDataStorage(buf.coreAttnOut);
            Qwen35DetachGraphDataStorage(buf.linearQ);
            Qwen35DetachGraphDataStorage(buf.linearK);
            Qwen35DetachGraphDataStorage(buf.linearV);
            Qwen35DetachGraphDataStorage(buf.qRepeat);
            Qwen35DetachGraphDataStorage(buf.kRepeat);
            Qwen35DetachGraphDataStorage(buf.qq);
            Qwen35DetachGraphDataStorage(buf.kkPad);
            Qwen35DetachGraphDataStorage(buf.vvPad);
            Qwen35DetachGraphDataStorage(buf.bbPad);
            Qwen35DetachGraphDataStorage(buf.ggPad);
            Qwen35DetachGraphDataStorage(buf.decayMask);
            Qwen35DetachGraphDataStorage(buf.kBeta);
            Qwen35DetachGraphDataStorage(buf.vBeta);
            Qwen35DetachGraphDataStorage(buf.attn);
            Qwen35DetachGraphDataStorage(buf.at);
            Qwen35DetachGraphDataStorage(buf.kCumdecay);
            Qwen35DetachGraphDataStorage(buf.gExp);
            Qwen35DetachGraphDataStorage(buf.coreTemp);
            Qwen35DetachGraphDataStorage(buf.logitsHalf);
        }

        static void Qwen35PrepareGraphStateForErase(Qwen35CudaGraphDecodeState &state) {
            Qwen35DestroyCudaGraph(state);
            Qwen35DetachGraphDataStorage(state.inputIds);
            Qwen35DetachGraphDataStorage(state.positionIds);
            Qwen35DetachGraphDataStorage(state.logits);
            Qwen35DetachGraphDataStorage(state.linearSlotIds);
            Qwen35DetachGraphBuffers(state.buffers);
        }

        using Qwen35CudaGraphStateKey = std::tuple<const Qwen3_5Model*, int, int>;

        static std::mutex &Qwen35CudaGraphStatesMutex() {
            static std::mutex *statesMutex = new std::mutex();
            return *statesMutex;
        }

        static std::map<Qwen35CudaGraphStateKey, std::unique_ptr<Qwen35CudaGraphDecodeState> > &Qwen35CudaGraphStates() {
            static auto *states = new std::map<Qwen35CudaGraphStateKey, std::unique_ptr<Qwen35CudaGraphDecodeState> >();
            return *states;
        }

        static Qwen35CudaGraphDecodeState &GetQwen35CudaGraphDecodeState(const Qwen3_5Model *model, int gpuId, int batch) {
            std::lock_guard<std::mutex> guard(Qwen35CudaGraphStatesMutex());
            auto &states = Qwen35CudaGraphStates();
            auto key = std::make_tuple(model, gpuId, batch);
            auto &state = states[key];
            if (state == nullptr) {
                state.reset(new Qwen35CudaGraphDecodeState());
            }
            return *state;
        }

        static void Qwen35EraseCudaGraphDecodeStates(const Qwen3_5Model *model) {
            std::lock_guard<std::mutex> guard(Qwen35CudaGraphStatesMutex());
            auto &states = Qwen35CudaGraphStates();
            for (auto it = states.begin(); it != states.end();) {
                if (std::get<0>(it->first) == model) {
                    if (it->second != nullptr) {
                        std::lock_guard<std::mutex> stateGuard(it->second->mutex);
                        Qwen35PrepareGraphStateForErase(*it->second);
                    }
                    it = states.erase(it);
                } else {
                    ++it;
                }
            }
        }

        static bool Qwen35CudaGraphEnabled() {
            return GetFastllmEnv().cudaGraph;
        }

        static void Qwen35PrepareGraphCudaTensor(Data &dst, const Data &src, int device) {
            AssertInFastLLM(src.dataDevice == DataDevice::CUDA && src.cudaData != nullptr,
                            "Qwen3.5 CUDA graph requires CUDA source tensor.\n");
            FastllmCudaSetDevice(device);

            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != src.dataType || dst.dims != src.dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(src.dims);
            }
            dst.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
        }

        static void Qwen35PrepareGraphIntTensor(Data &dst, int device, const std::vector<int> &host) {
            AssertInFastLLM(!host.empty(), "Qwen3.5 CUDA graph got empty int metadata.\n");
            FastllmCudaSetDevice(device);
            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != DataType::INT32 ||
                             dst.dims != std::vector<int>{(int)host.size()} ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = DataType::INT32;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize({(int)host.size()});
            }
            dst.Allocate(false);
            FastllmCudaCopyFromHostToDevice(dst.cudaData, (void*)host.data(), host.size() * sizeof(int32_t));
            dst.cpuIntDatas = host;
        }

        enum Qwen35LinearSlotPoolKind {
            QWEN35_LINEAR_SLOT_CONV = 0,
            QWEN35_LINEAR_SLOT_RECURRENT = 1
        };

        using Qwen35LinearSlotPoolKey = std::tuple<const Qwen3_5Model*, int, int, int, int>;

        static std::mutex &Qwen35LinearSlotPoolsMutex() {
            static std::mutex *mutex = new std::mutex();
            return *mutex;
        }

        static std::map<Qwen35LinearSlotPoolKey, std::unique_ptr<PagedCacheManager> > &Qwen35LinearSlotPools() {
            static auto *pools = new std::map<Qwen35LinearSlotPoolKey, std::unique_ptr<PagedCacheManager> >();
            return *pools;
        }

        static int Qwen35MaxCudaGraphDecodeBatch() {
            return 32;
        }

        static int Qwen35PreCaptureMaxBatch(const Qwen3_5Model *model) {
            int maxGraphBatch = Qwen35MaxCudaGraphDecodeBatch();
            (void)model;
            return std::max(1, maxGraphBatch);
        }

        static int Qwen35LinearSlotCapacity(const Qwen3_5Model *model, int batch) {
            int capacity = std::max(1, batch);
            capacity = std::max(capacity, Qwen35PreCaptureMaxBatch(model));
            return capacity;
        }

        static uint64_t Qwen35Float16BytesForDims(const std::vector<int> &dims) {
            uint64_t count = 1;
            for (int dim : dims) {
                AssertInFastLLM(dim > 0, "Qwen3.5 linear slot cache got invalid dims.\n");
                count *= (uint64_t)dim;
            }
            return count * sizeof(uint16_t);
        }

        static PagedCacheManager *Qwen35GetLinearSlotPoolLocked(
                const Qwen3_5Model *model,
                int gpuId,
                int layer,
                Qwen35LinearSlotPoolKind kind,
                const std::vector<int> &managerDims) {
            AssertInFastLLM(!managerDims.empty() && managerDims[0] > 0,
                            "Qwen3.5 linear slot pool got invalid manager dims.\n");
            auto key = std::make_tuple(model, gpuId, layer, (int)kind, managerDims[0]);
            auto &pools = Qwen35LinearSlotPools();
            auto it = pools.find(key);
            if (it != pools.end()) {
                PagedCacheManager *manager = it->second.get();
                if (manager == nullptr || manager->dims != managerDims ||
                    manager->dataDevice != DataDevice::CUDA ||
                    manager->cudaData == nullptr) {
                    return nullptr;
                }
                return manager;
            }

            FastllmCudaSetDevice(gpuId);
            std::unique_ptr<PagedCacheManager> manager(new PagedCacheManager());
            manager->type = PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_MLP_CACHE;
            manager->pageLen = 1;
            ((Data*)manager.get())->dataType = DataType::FLOAT16;
            ((Data*)manager.get())->UpdateUnitSize();
            ((Data*)manager.get())->directMemory = true;
            ((Data*)manager.get())->dataDevice = DataDevice::CUDA;
            ((Data*)manager.get())->dataDeviceIds = {gpuId};
            ((Data*)manager.get())->Resize(managerDims);
            ((Data*)manager.get())->Allocate();
            FastllmCudaMemset0(manager->cudaData, manager->GetBytes());
            manager->SetMaxPages(managerDims[0]);

            PagedCacheManager *ret = manager.get();
            pools[key] = std::move(manager);
            return ret;
        }

        static PagedCacheManager *Qwen35FindLinearSlotPoolLocked(
                const Qwen3_5Model *model,
                int gpuId,
                int layer,
                Qwen35LinearSlotPoolKind kind,
                const std::vector<int> &managerDims) {
            auto &pools = Qwen35LinearSlotPools();
            auto it = pools.find(std::make_tuple(model, gpuId, layer, (int)kind, managerDims[0]));
            if (it == pools.end() || it->second == nullptr) {
                return nullptr;
            }
            PagedCacheManager *manager = it->second.get();
            if (manager->dims != managerDims ||
                manager->dataDevice != DataDevice::CUDA ||
                manager->cudaData == nullptr) {
                return nullptr;
            }
            return manager;
        }

        static bool Qwen35LinearSlotIsFree(PagedCacheManager *manager, int slot) {
            if (manager == nullptr || slot < 0 || slot >= manager->maxPages) {
                return false;
            }
            std::lock_guard<std::mutex> guard(manager->pageIndexLocker);
            return slot < (int)manager->pageRefCount.size() &&
                   manager->pageRefCount[slot] == 0;
        }

        static bool Qwen35DataUsesLinearSlot(const Data &data, PagedCacheManager *manager, int &slot) {
            if (data.cudaDataBorrowed && data.isPagedKVCache &&
                data.pagedKVCacheData == manager && data.pageIndex.size() == 1) {
                slot = data.pageIndex[0];
                return slot >= 0 && manager != nullptr && slot < manager->maxPages;
            }
            return false;
        }

        static void Qwen35ReleasePagedReference(Data &data) {
            if (data.isPagedKVCache && data.pagedKVCacheData != nullptr && !data.pageIndex.empty()) {
                data.pagedKVCacheData->ReleasePageIndices(data.pageIndex);
            }
            data.pageIndex.clear();
            data.pagedKVCacheData = nullptr;
            data.isPagedKVCache = false;
        }

        static void Qwen35FreeOwnedCudaPointer(void *ptr, bool borrowed, bool directMemory) {
            if (ptr == nullptr || borrowed) {
                return;
            }
            if (directMemory) {
                FastllmCudaDirectFree(ptr);
            } else {
                FastllmCudaFree(ptr);
            }
        }

        static bool Qwen35AttachLinearSlot(
                Data &cache,
                PagedCacheManager *manager,
                int slot,
                const std::vector<int> &viewDims,
                bool transposed,
                bool alreadyPicked) {
            if (manager == nullptr || manager->cudaData == nullptr ||
                slot < 0 || slot >= manager->maxPages) {
                return false;
            }
            uint64_t slotBytes = Qwen35Float16BytesForDims(viewDims);
            void *slotPtr = (void*)((uint8_t*)manager->cudaData + (uint64_t)slot * slotBytes);
            int existingSlot = -1;
            if (Qwen35DataUsesLinearSlot(cache, manager, existingSlot) &&
                existingSlot == slot && cache.cudaData == slotPtr) {
                cache.isKVCache = true;
                cache.isLinearAttention = true;
                cache.isLinearAttentionTransposed = transposed;
                return true;
            }

            void *oldPtr = cache.cudaData;
            bool oldBorrowed = cache.cudaDataBorrowed;
            bool oldDirectMemory = cache.directMemory;
            uint64_t oldBytes = (!cache.dims.empty() && oldPtr != nullptr) ? cache.GetBytes() : 0;

            if (!alreadyPicked) {
                if (!Qwen35LinearSlotIsFree(manager, slot)) {
                    return false;
                }
                std::vector<int> pickedPages = {slot};
                manager->Pick(pickedPages);
            }

            if (oldPtr != nullptr && oldPtr != slotPtr && oldBytes == slotBytes &&
                cache.dataDevice == DataDevice::CUDA) {
                FastllmCudaCopyFromDeviceToDevice(slotPtr, oldPtr, slotBytes);
            } else {
                FastllmCudaMemset0(slotPtr, slotBytes);
            }

            if (!(cache.isPagedKVCache && cache.pagedKVCacheData == manager &&
                  cache.pageIndex.size() == 1 && cache.pageIndex[0] == slot)) {
                Qwen35ReleasePagedReference(cache);
            }
            if (oldPtr != slotPtr) {
                Qwen35FreeOwnedCudaPointer(oldPtr, oldBorrowed, oldDirectMemory);
            }

            Qwen3CudaClearMultiDeviceState(cache);
            cache.isFake = false;
            cache.isKVCache = true;
            cache.isLinearAttention = true;
            cache.isLinearAttentionTransposed = transposed;
            cache.lockInCPU = false;
            cache.dataType = DataType::FLOAT16;
            cache.UpdateUnitSize();
            cache.dataDevice = DataDevice::CUDA;
            cache.dataDeviceIds = manager->dataDeviceIds;
            cache.directMemory = false;
            cache.Resize(viewDims);
            cache.expansionDims.clear();
            cache.expansionSize = cache.Count(0);
            cache.expansionBytes = slotBytes;
            cache.cpuData = nullptr;
            cache.cudaData = slotPtr;
            cache.cudaDataBorrowed = true;
            cache.isPagedKVCache = true;
            cache.pagedKVCacheData = manager;
            cache.pageIndex = {slot};
            cache.pageLen = 1;
            cache.lastPageLen = 1;
            return true;
        }

        static PagedCacheManager *Qwen35FindLinearSlotPool(
                const Qwen3_5Model *model,
                int gpuId,
                int layer,
                Qwen35LinearSlotPoolKind kind,
                int slotCapacity) {
            std::lock_guard<std::mutex> guard(Qwen35LinearSlotPoolsMutex());
            auto &pools = Qwen35LinearSlotPools();
            auto it = pools.find(std::make_tuple(model, gpuId, layer, (int)kind, slotCapacity));
            return it == pools.end() || it->second == nullptr ? nullptr : it->second.get();
        }

        struct Qwen35LinearPrefixSnapshotCache {
            bool valid = false;
            bool multiDevice = false;
            Data single;
            std::map<int, Data> locals;
            std::vector<int> dataDeviceIds;
            TensorParallelLayoutType tpLayout = TP_LAYOUT_NONE;
            int tpAxis = -1;
            DivisionScheme tpRanges;
            std::vector<int> tpGlobalDims;
            std::vector<int> dims;
            DataType dataType = DataType::FLOAT16;
            bool transposed = false;
        };

        struct Qwen35LinearPrefixSnapshotLayer {
            bool linear = false;
            Qwen35LinearPrefixSnapshotCache first;
            Qwen35LinearPrefixSnapshotCache second;
        };

        struct Qwen35LinearPrefixSnapshot {
            int cachedLen = 0;
            int requestId = 0;
            long long timestamp = 0;
            std::vector<int> tokens;
            std::vector<Qwen35LinearPrefixSnapshotLayer> layers;
            bool mtpValid = false;
            int mtpTokens = 0;
            Data mtpKey;
            Data mtpValue;
        };

        static std::mutex &Qwen35LinearPrefixSnapshotsMutex() {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        static std::map<const Qwen3_5Model*, std::vector<std::unique_ptr<Qwen35LinearPrefixSnapshot> > >
                &Qwen35LinearPrefixSnapshots() {
            static auto *snapshots =
                new std::map<const Qwen3_5Model*, std::vector<std::unique_ptr<Qwen35LinearPrefixSnapshot> > >();
            return *snapshots;
        }

        static long long &Qwen35LinearPrefixSnapshotTimestamp() {
            static auto *timestamp = new long long(0);
            return *timestamp;
        }

        static std::atomic<int> &Qwen35LinearPrefixSnapshotRequestCounter() {
            static auto *counter = new std::atomic<int>(0);
            return *counter;
        }

        static int Qwen35EnvInt(const char *name, int fallback) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == 0) {
                return fallback;
            }
            char *end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            if (end == value) {
                return fallback;
            }
            return (int)parsed;
        }

        static bool Qwen35LinearPrefixCacheEnabled() {
            const char *enabled = std::getenv("FASTLLM_PREFIX_CACHE");
            return enabled == nullptr || enabled[0] == 0 || Qwen35MoeIsTrueString(enabled);
        }

        static int Qwen35LinearPrefixSnapshotIntervalTokens() {
            int pages = Qwen35EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", 16);
            pages = std::max(1, pages);
            return pages * fastllm::GetPageLen();
        }

        static int Qwen35LinearPrefixSnapshotMaxPerRequest() {
            return std::max(1, Qwen35EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST", 4));
        }

        static int Qwen35LinearPrefixSnapshotMaxRecords() {
            return std::max(1, Qwen35EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", 8));
        }

        static bool Qwen35LayerIsLinearAttention(const Qwen3_5Model *model, int layer) {
            std::string prefix = Qwen3_5Model::language_prefix + "layers." + std::to_string(layer) + ".";
            return model->weight.weight.find(prefix + "self_attn.o_proj.weight") == model->weight.weight.end();
        }

        static bool Qwen35HasLinearAttentionLayers(const Qwen3_5Model *model, int blockCnt) {
            for (int i = 0; i < blockCnt; i++) {
                if (Qwen35LayerIsLinearAttention(model, i)) {
                    return true;
                }
            }
            return false;
        }

        static int Qwen35CacheTokenLen(const Data &cache) {
            if (cache.multiDeviceData && !cache.multiDeviceDatas.empty()) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        int len = Qwen35CacheTokenLen(*it.second);
                        if (len > 0) {
                            return len;
                        }
                    }
                }
            }
            if (cache.isPagedKVCache) {
                if (cache.pageIndex.empty()) {
                    return 0;
                }
                return ((int)cache.pageIndex.size() - 1) * cache.pageLen + cache.lastPageLen;
            }
            if (cache.dims.size() > 1) {
                return cache.dims[1];
            }
            return 0;
        }

        static int Qwen35CurrentTokenGrowingCacheLen(
                const Qwen3_5Model *model,
                int blockCnt,
                const std::vector<std::pair<Data, Data> > &pastKeyValues) {
            for (int i = 0; i < blockCnt && i < (int)pastKeyValues.size(); i++) {
                if (Qwen35LayerIsLinearAttention(model, i)) {
                    continue;
                }
                int len = Qwen35CacheTokenLen(pastKeyValues[i].first);
                if (len > 0) {
                    return len;
                }
                len = Qwen35CacheTokenLen(pastKeyValues[i].second);
                if (len > 0) {
                    return len;
                }
            }
            return 0;
        }

        static bool Qwen35SnapshotCopyTensor(const Data &src, Data &dst) {
            if (src.dims.empty()) {
                return false;
            }
            if (src.dataDevice == DataDevice::CUDA && src.cudaData == nullptr) {
                return false;
            }
            if (src.dataDevice == DataDevice::CPU && src.cpuData == nullptr) {
                return false;
            }
            dst.CopyFrom(src);
            dst.isKVCache = true;
            dst.isLinearAttention = src.isLinearAttention;
            dst.isLinearAttentionTransposed = src.isLinearAttentionTransposed;
            dst.isPagedKVCache = false;
            dst.pagedKVCacheData = nullptr;
            dst.pageIndex.clear();
            dst.lastPageLen = 0;
            dst.multiDeviceData = false;
            dst.multiDeviceDatas.clear();
            dst.ClearTensorParallelLayout();
            dst.ToDevice(DataDevice::CPU, true);
            return dst.cpuData != nullptr;
        }

        static bool Qwen35SnapshotCopyCache(
                const Data &src,
                Qwen35LinearPrefixSnapshotCache &dst) {
            dst.valid = false;
            dst.multiDevice = src.multiDeviceData && !src.multiDeviceDatas.empty();
            dst.dataDeviceIds = src.dataDeviceIds;
            dst.tpLayout = src.tpLayout;
            dst.tpAxis = src.tpAxis;
            dst.tpRanges = src.tpRanges;
            dst.tpGlobalDims = src.tpGlobalDims;
            dst.dims = src.dims;
            dst.dataType = src.dataType;
            dst.transposed = src.isLinearAttentionTransposed;
            if (dst.multiDevice) {
                for (auto &it : src.multiDeviceDatas) {
                    if (it.second == nullptr) {
                        continue;
                    }
                    Data &copied = dst.locals[it.first];
                    if (!Qwen35SnapshotCopyTensor(*it.second, copied)) {
                        return false;
                    }
                }
                dst.valid = !dst.locals.empty();
                return dst.valid;
            }
            dst.valid = Qwen35SnapshotCopyTensor(src, dst.single);
            return dst.valid;
        }

        static void Qwen35ReleasePagedReferencesNoDuplicate(
                Data &cache,
                std::set<std::pair<PagedCacheManager*, int> > &released) {
            if (cache.isPagedKVCache && cache.pagedKVCacheData != nullptr) {
                std::vector<int> pages;
                for (int page : cache.pageIndex) {
                    std::pair<PagedCacheManager*, int> key = {cache.pagedKVCacheData, page};
                    if (released.insert(key).second) {
                        pages.push_back(page);
                    }
                }
                if (!pages.empty()) {
                    cache.pagedKVCacheData->ReleasePageIndices(pages);
                }
            }
            cache.isPagedKVCache = false;
            cache.pagedKVCacheData = nullptr;
            cache.pageIndex.clear();
            cache.lastPageLen = 0;
        }

        static void Qwen35ReleaseAllPagedReferences(Data &cache) {
            std::set<std::pair<PagedCacheManager*, int> > released;
            Qwen35ReleasePagedReferencesNoDuplicate(cache, released);
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        Qwen35ReleasePagedReferencesNoDuplicate(*it.second, released);
                    }
                }
            }
        }

        static void Qwen35ClearCacheForSnapshotRestore(Data &cache) {
            Qwen35ReleaseAllPagedReferences(cache);
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    delete it.second;
                }
                cache.multiDeviceDatas.clear();
                cache.multiDeviceData = false;
            }
            cache.ClearTensorParallelLayout();
            cache.FreeSpace();
            cache.isFake = false;
            cache.isKVCache = true;
            cache.isLinearAttention = true;
            cache.isPagedKVCache = false;
            cache.pagedKVCacheData = nullptr;
            cache.pageIndex.clear();
            cache.lastPageLen = 0;
        }

        static bool Qwen35RestoreSnapshotTensor(const Data &snapshot, Data &dst, int device) {
            Qwen35ClearCacheForSnapshotRestore(dst);
            dst.CopyFrom(snapshot);
            dst.isKVCache = true;
            dst.isLinearAttention = true;
            dst.isLinearAttentionTransposed = snapshot.isLinearAttentionTransposed;
            dst.isPagedKVCache = false;
            dst.pagedKVCacheData = nullptr;
            dst.pageIndex.clear();
            dst.lastPageLen = 0;
            dst.ToDevice(DataDevice::CUDA, {device}, true);
            return !dst.dims.empty();
        }

        static bool Qwen35RestoreSnapshotCache(
                const Qwen35LinearPrefixSnapshotCache &snapshot,
                Data &dst) {
            if (!snapshot.valid) {
                return false;
            }
            if (!snapshot.multiDevice) {
                int device = snapshot.dataDeviceIds.empty() ? 0 : snapshot.dataDeviceIds[0];
                return Qwen35RestoreSnapshotTensor(snapshot.single, dst, device);
            }

            Qwen35ClearCacheForSnapshotRestore(dst);
            dst.multiDeviceData = true;
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = snapshot.dataDeviceIds;
            dst.tpLayout = snapshot.tpLayout;
            dst.tpAxis = snapshot.tpAxis;
            dst.tpRanges = snapshot.tpRanges;
            dst.tpGlobalDims = snapshot.tpGlobalDims;
            dst.dims = snapshot.dims.empty() ? snapshot.tpGlobalDims : snapshot.dims;
            dst.dataType = snapshot.dataType;
            dst.UpdateUnitSize();
            dst.isKVCache = true;
            dst.isLinearAttention = true;
            dst.isLinearAttentionTransposed = snapshot.transposed;
            dst.isPagedKVCache = false;
            dst.pagedKVCacheData = nullptr;
            dst.pageIndex.clear();
            dst.lastPageLen = 0;
            dst.cudaData = nullptr;
            for (auto &it : snapshot.locals) {
                int device = it.first;
                Data *local = new Data(it.second.dataType);
                if (!Qwen35RestoreSnapshotTensor(it.second, *local, device)) {
                    delete local;
                    return false;
                }
                local->cacheUid = dst.cacheUid;
                dst.multiDeviceDatas[device] = local;
            }
            return !dst.multiDeviceDatas.empty();
        }

        static bool Qwen35RestoreMtpSnapshotTensor(
                const Data &snapshot,
                Data &dst,
                int device) {
            dst.FreeSpace();
            dst.CopyFrom(snapshot);
            dst.isPagedKVCache = false;
            dst.pagedKVCacheData = nullptr;
            dst.pageIndex.clear();
            dst.lastPageLen = 0;
            dst.multiDeviceData = false;
            dst.multiDeviceDatas.clear();
            dst.ClearTensorParallelLayout();
            if (dst.expansionDims.size() != dst.dims.size()) {
                dst.expansionDims = dst.dims;
            }
            dst.ToDevice(DataDevice::CUDA, {device}, true);
            return dst.dataDevice == DataDevice::CUDA &&
                   dst.cudaData != nullptr && dst.dims.size() >= 2;
        }

        static const Qwen35LinearPrefixSnapshot *Qwen35FindLinearPrefixSnapshotLocked(
                const Qwen3_5Model *model,
                const std::vector<int> &tokens,
                int maxCachedLen,
                int exactLen = -1,
                bool requireMtp = false) {
            auto &all = Qwen35LinearPrefixSnapshots();
            auto it = all.find(model);
            if (it == all.end()) {
                return nullptr;
            }
            const Qwen35LinearPrefixSnapshot *best = nullptr;
            for (auto &snapshotPtr : it->second) {
                Qwen35LinearPrefixSnapshot *snapshot = snapshotPtr.get();
                if (snapshot == nullptr || snapshot->cachedLen <= 0 ||
                    snapshot->cachedLen > maxCachedLen ||
                    snapshot->cachedLen > (int)tokens.size()) {
                    continue;
                }
                if (exactLen >= 0 && snapshot->cachedLen != exactLen) {
                    continue;
                }
                if ((int)snapshot->tokens.size() != snapshot->cachedLen ||
                    !std::equal(snapshot->tokens.begin(), snapshot->tokens.end(), tokens.begin())) {
                    continue;
                }
                if (requireMtp &&
                    (!snapshot->mtpValid || snapshot->mtpTokens != snapshot->cachedLen ||
                     snapshot->mtpKey.dims.size() < 2 ||
                     snapshot->mtpValue.dims.size() < 2 ||
                     snapshot->mtpKey.dims[1] != snapshot->cachedLen ||
                     snapshot->mtpValue.dims[1] != snapshot->cachedLen)) {
                    continue;
                }
                if (best == nullptr ||
                    snapshot->cachedLen > best->cachedLen ||
                    (snapshot->cachedLen == best->cachedLen &&
                     snapshot->timestamp > best->timestamp)) {
                    best = snapshot;
                }
            }
            return best;
        }

        static void Qwen35EraseLinearPrefixSnapshots(const Qwen3_5Model *model) {
            std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
            Qwen35LinearPrefixSnapshots().erase(model);
        }

        static bool Qwen35PrepareLinearSlotCaches(
                const Qwen3_5Model *model,
                int gpuId,
                int batch,
                int blockCnt,
                const std::vector<int> &linearLayers,
                std::vector<std::pair<Data*, Data*> > &pastKeyValues,
                std::vector<int> &slotIdsHost) {
            if (linearLayers.empty()) {
                slotIdsHost.clear();
                return true;
            }
            int firstLinearLayer = linearLayers.front();
            int slotCapacity = Qwen35LinearSlotCapacity(model, batch);
            slotIdsHost.assign(batch, 0);
            std::lock_guard<std::mutex> guard(Qwen35LinearSlotPoolsMutex());

            for (int b = 0; b < batch; b++) {
                Data *ownerCache = pastKeyValues[b * blockCnt + firstLinearLayer].first;
                if (ownerCache == nullptr || ownerCache->dims.size() != 3 ||
                    ownerCache->dims[0] != 1 || ownerCache->dims[2] != 4 ||
                    ownerCache->dataDevice != DataDevice::CUDA ||
                    ownerCache->dataType != DataType::FLOAT16 ||
                    ownerCache->cudaData == nullptr) {
                    return false;
                }
                PagedCacheManager *ownerPool = Qwen35FindLinearSlotPoolLocked(
                    model, gpuId, firstLinearLayer, QWEN35_LINEAR_SLOT_CONV,
                    {slotCapacity, 1, ownerCache->dims[1], 4});
                if (ownerPool == nullptr) {
                    return false;
                }

                int slot = -1;
                bool ownerPicked = false;
                if (!Qwen35DataUsesLinearSlot(*ownerCache, ownerPool, slot)) {
                    if (ownerPool->FreePageCount() <= 0) {
                        return false;
                    }
                    slot = ownerPool->GetUnusedPageIndex(true);
                    ownerPicked = true;
                }
                slotIdsHost[b] = slot;

                for (int layer : linearLayers) {
                    Data *convCache = pastKeyValues[b * blockCnt + layer].first;
                    Data *recurrentState = pastKeyValues[b * blockCnt + layer].second;
                    if (convCache == nullptr || recurrentState == nullptr ||
                        convCache->dims.size() != 3 || convCache->dims[0] != 1 ||
                        convCache->dims[2] != 4 ||
                        recurrentState->dims.size() != 4 || recurrentState->dims[0] != 1 ||
                        convCache->dataDevice != DataDevice::CUDA ||
                        recurrentState->dataDevice != DataDevice::CUDA ||
                        convCache->dataType != DataType::FLOAT16 ||
                        recurrentState->dataType != DataType::FLOAT16 ||
                        convCache->cudaData == nullptr ||
                        recurrentState->cudaData == nullptr ||
                        !recurrentState->isLinearAttentionTransposed) {
                        return false;
                    }

                    PagedCacheManager *convPool = Qwen35FindLinearSlotPoolLocked(
                        model, gpuId, layer, QWEN35_LINEAR_SLOT_CONV,
                        {slotCapacity, 1, convCache->dims[1], 4});
                    PagedCacheManager *statePool = Qwen35FindLinearSlotPoolLocked(
                        model, gpuId, layer, QWEN35_LINEAR_SLOT_RECURRENT,
                        {slotCapacity, recurrentState->dims[1], recurrentState->dims[3],
                         recurrentState->dims[2]});
                    if (convPool == nullptr || statePool == nullptr) {
                        return false;
                    }

                    bool convAlreadyPicked = layer == firstLinearLayer && ownerPicked;
                    if (!Qwen35AttachLinearSlot(*convCache, convPool, slot,
                                                {1, convCache->dims[1], 4},
                                                false, convAlreadyPicked)) {
                        return false;
                    }
                    if (!Qwen35AttachLinearSlot(*recurrentState, statePool, slot,
                                                {1, recurrentState->dims[1],
                                                 recurrentState->dims[2],
                                                 recurrentState->dims[3]},
                                                true, false)) {
                        return false;
                    }
                }
            }
            return true;
        }

        static void Qwen35CudaPagedCacheCopyBatch(
                PagedCacheManager &manager,
                const Data &input,
                const Data &insertIndexs,
                const Data &insertPositions) {
            AssertInFastLLM(input.dataDevice == DataDevice::CUDA &&
                            insertIndexs.dataDevice == DataDevice::CUDA &&
                            insertPositions.dataDevice == DataDevice::CUDA,
                            "Qwen3.5 CUDA graph paged cache copy expects CUDA tensors.\n");
            AssertInFastLLM(input.dims.size() == 3 && insertIndexs.dims.size() == 1 &&
                            insertPositions.dims.size() == 1 &&
                            insertIndexs.dims[0] == input.dims[0] &&
                            insertPositions.dims[0] == input.dims[0],
                            "Qwen3.5 CUDA graph paged cache copy got invalid metadata shape.\n");
            Data &storage = *((Data*)&manager);
            AssertInFastLLM(storage.cudaData != nullptr && storage.dims.size() == 4,
                            "Qwen3.5 CUDA graph paged cache manager is not allocated.\n");
            FastllmCudaPagedCacheCopyBatch(
                (uint8_t*)storage.cudaData,
                (int32_t*)insertIndexs.cudaData,
                (int32_t*)insertPositions.cudaData,
                manager.pageLen,
                input.dims[0],
                input.dims[1],
                input.dims[2],
                storage.dataType,
                (uint8_t*)input.cudaData,
                input.dataType);
        }

        static void Qwen35CudaCopyTensor(Qwen3CudaDirectRunner &runner,
                                         const Data &input,
                                         Data &output) {
            runner.Run("Copy",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaSigmoid(Qwen3CudaDirectRunner &runner,
                                      Data &input, Data &output) {
            runner.Run("Sigmoid",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaSilu(Qwen3CudaDirectRunner &runner,
                                   Data &input, Data &output) {
            runner.Run("Silu",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaExp(Qwen3CudaDirectRunner &runner,
                                  Data &input, Data &output) {
            runner.Run("Exp",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaMul(Qwen3CudaDirectRunner &runner,
                                  const Data &input, float value, Data &output) {
            runner.Run("Mul",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict{{"v", value}}, IntDict(), {"output"});
        }

        static void Qwen35CudaMulTo(Qwen3CudaDirectRunner &runner,
                                    Data &input0, const Data &input1,
                                    float alpha = 1.0f) {
            runner.Run("MulTo",
                       DataDict{{"input0", &input0}, {"input1", (Data*)&input1}},
                       FloatDict{{"alpha", alpha}}, IntDict());
        }

        static void Qwen35CudaRepeat(Qwen3CudaDirectRunner &runner,
                                     const Data &input, int axis, int repeatTimes,
                                     Data &output) {
            runner.Run("Repeat",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}, {"repeatTimes", repeatTimes}},
                       {"output"});
        }

        static void Qwen35CudaPad(Qwen3CudaDirectRunner &runner,
                                  const Data &input, int axis, int padSize,
                                  Data &output) {
            runner.Run("Pad",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}, {"padSize", padSize}},
                       {"output"});
        }

        static void Qwen35CudaMatMul(Qwen3CudaDirectRunner &runner,
                                     const Data &input0, const Data &input1,
                                     Data &output, float alpha = 1.0f, int group = 1) {
            runner.Run("MatMul",
                       DataDict{{"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}},
                       FloatDict{{"alpha", alpha}}, IntDict{{"group", group}},
                       {"output"});
        }

        static void Qwen35CudaMatMulTransB(Qwen3CudaDirectRunner &runner,
                                           const Data &input0, const Data &input1,
                                           Data &output, float alpha = 1.0f, int group = 1) {
            runner.Run("MatMulTransB",
                       DataDict{{"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}},
                       FloatDict{{"alpha", alpha}}, IntDict{{"group", group}},
                       {"output"});
        }

        static void Qwen35CudaConv1DPerChannel(Qwen3CudaDirectRunner &runner,
                                               const Data &input, Data &weight, Data &bias,
                                               int inputChannels, int outputChannels,
                                               int kernel, int stride, int pad,
                                               Data &output) {
            runner.Run("Conv1DPerChannel",
                       DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"bias", &bias}, {"output", &output}},
                       FloatDict(),
                       IntDict{{"inputChannels", inputChannels}, {"outputChannels", outputChannels},
                               {"kernel", kernel}, {"stride", stride}, {"pad", pad}},
                       {"output"});
        }

        static void Qwen35CudaMambaSoftplus(Qwen3CudaDirectRunner &runner,
                                            Data &input, Data &aLog, Data &dtBias,
                                            Data &output) {
            runner.Run("MambaSoftplus",
                       DataDict{{"input", &input}, {"aLog", &aLog}, {"dtBias", &dtBias}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaSigmoidMambaSoftplus(Qwen3CudaDirectRunner &runner,
                                                   Data &sigmoidInputOutput,
                                                   const Data &softplusInput,
                                                   Data &aLog,
                                                   Data &dtBias,
                                                   Data &softplusOutput) {
            runner.Run("SigmoidMambaSoftplus",
                       DataDict{{"sigmoidInputOutput", &sigmoidInputOutput},
                                {"softplusInput", (Data*)&softplusInput},
                                {"aLog", &aLog}, {"dtBias", &dtBias},
                                {"softplusOutput", &softplusOutput}},
                       FloatDict(), IntDict(), {"softplusOutput"});
        }

        static bool Qwen35TryCudaRMSNormSiluMul(
                Data &input, Data &weight, Data &gateInput, Data &output, float eps) {
            if (input.dataDevice != DataDevice::CUDA ||
                gateInput.dataDevice != DataDevice::CUDA ||
                output.dataDevice != DataDevice::CUDA ||
                input.dataType != DataType::FLOAT16 ||
                gateInput.dataType != DataType::FLOAT16 ||
                output.dataType != DataType::FLOAT16) {
                return false;
            }
            if (weight.dataDevice != DataDevice::CUDA) {
                if (!input.dataDeviceIds.empty()) {
                    weight.ToDevice(DataDevice::CUDA, input.dataDeviceIds);
                } else {
                    weight.ToDevice(DataDevice::CUDA);
                }
            }
            if (weight.dataDevice != DataDevice::CUDA ||
                weight.dataType != DataType::FLOAT32 ||
                input.dims != gateInput.dims ||
                input.dims != output.dims) {
                return false;
            }
            return FastllmCudaRMSNormSiluMulFloat16(input, weight, gateInput, output, eps);
        }

        static void Qwen35CudaCausalMask(Qwen3CudaDirectRunner &runner,
                                         Data &input, int base, float maskValue) {
            runner.Run("CausalMask",
                       DataDict{{"input", &input}},
                       FloatDict{{"maskValue", maskValue}},
                       IntDict{{"base", base}});
        }

        static void Qwen35CudaTransferAttn(Qwen3CudaDirectRunner &runner,
                                           Data &input) {
            runner.Run("TransferAttn", DataDict{{"input", &input}}, FloatDict(), IntDict());
        }

        static void Qwen35CudaCumSumLastDim(Qwen3CudaDirectRunner &runner,
                                            Data &input) {
            runner.Run("CumSumLastDim", DataDict{{"input", &input}}, FloatDict(), IntDict());
        }

        static void Qwen35CudaMakeDecayMask(Qwen3CudaDirectRunner &runner,
                                            Data &input, Data &output) {
            runner.Run("MakeDecayMask",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen35CudaRecurrentGatedDeltaRule(
                Qwen3CudaDirectRunner &runner,
                Data &q, Data &k, Data &v, Data &g, Data &b,
                Data &lastRecurrentState, Data &coreAttnOut, float qScale) {
            runner.Run("RecurrentGatedDeltaRule",
                       DataDict{{"q", &q}, {"k", &k}, {"v", &v}, {"g", &g}, {"b", &b},
                                {"last_recurrent_state", &lastRecurrentState},
                                {"core_attn_out", &coreAttnOut}},
                       FloatDict{{"qScale", qScale}}, IntDict(), {"core_attn_out"});
        }

        static void Qwen35CudaChunkGatedDeltaRulePrefill(
                Qwen3CudaDirectRunner &runner,
                Data &q, Data &k, Data &v, Data &g,
                Data &attn, Data &kCumdecay,
                Data &lastRecurrentState, Data &coreAttnOut) {
            runner.Run("ChunkGatedDeltaRulePrefill",
                       DataDict{{"q", &q}, {"k", &k}, {"v", &v}, {"g", &g},
                                {"attn", &attn}, {"k_cumdecay", &kCumdecay},
                                {"last_recurrent_state", &lastRecurrentState},
                                {"core_attn_out", &coreAttnOut}},
                       FloatDict(), IntDict(), {"core_attn_out"});
        }

        static void Qwen35CudaInterleavedRope(
                Qwen3CudaDirectRunner &runner,
                Data &input,
                const Data &positionIds,
                int rotaryDim,
                const std::vector<int> &sections,
                float ropeTheta,
                float ropeScale) {
            runner.Run("Qwen35InterleavedRope",
                       DataDict{{"input", &input}, {"positionIds", (Data*)&positionIds}},
                       FloatDict{{"ropeTheta", ropeTheta}, {"ropeScale", ropeScale}},
                       IntDict{{"rotaryDim", rotaryDim},
                               {"sectionT", sections.size() > 0 ? sections[0] : 0},
                               {"sectionH", sections.size() > 1 ? sections[1] : 0},
                               {"sectionW", sections.size() > 2 ? sections[2] : 0}});
        }

        static void Qwen35CudaApplyRotary(
                Qwen3CudaDirectRunner &runner,
                Data &input,
                const Data &positionIds,
                int rotaryDim,
                const std::vector<int> &sections,
                float ropeTheta,
                float ropeScale) {
            if (positionIds.dims.size() == 2 && positionIds.dims[0] == 3) {
                Qwen35CudaInterleavedRope(runner, input, positionIds, rotaryDim,
                                          sections, ropeTheta, ropeScale);
            } else {
                qwen3cuda::Qwen3CudaRopeEncoding(runner, input, positionIds,
                                                 rotaryDim, ropeTheta, ropeScale);
            }
        }

        static void Qwen35CudaQGateKVRMSNormRopeSplitAppendPagedCache(
                Qwen3CudaDirectRunner &runner,
                Data &qgatekv,
                Data &qNormWeight,
                Data &kNormWeight,
                const Data &positionIds,
                Data &qOutput,
                Data &gateOutput,
                Data &pagedKCacheData,
                Data &pagedVCacheData,
                Data &insertIndexs,
                Data &insertPositions,
                int qHeads,
                int kHeads,
                int headDim,
                int rotaryDim,
                const std::vector<int> &sections,
                float eps,
                float ropeTheta,
                float ropeScale,
                int pageLen,
                int batch,
                bool doQKNorm,
                Data *lastPageLens) {
            AssertInFastLLM(qgatekv.dims.size() == 3,
                            "Qwen3.5 fused gated attention decode expects [bs, seq, dim].\n");
            int bsz = qgatekv.dims[0];
            int seqlen = qgatekv.dims[1];
            qOutput.dataType = qgatekv.dataType;
            qOutput.UpdateUnitSize();
            qOutput.Resize({bsz * qHeads, seqlen, headDim});
            gateOutput.dataType = qgatekv.dataType;
            gateOutput.UpdateUnitSize();
            gateOutput.Resize({bsz, seqlen, qHeads * headDim});

            DataDict datas = {
                    {"qgatekv", &qgatekv},
                    {"qNormWeight", &qNormWeight},
                    {"kNormWeight", &kNormWeight},
                    {"positionIds", (Data*)&positionIds},
                    {"qOutput", &qOutput},
                    {"gateOutput", &gateOutput},
                    {"pagedKCacheData", &pagedKCacheData},
                    {"pagedVCacheData", &pagedVCacheData},
                    {"insertIndexs", &insertIndexs},
                    {"insertPositions", &insertPositions}
            };
            std::vector<std::string> outputs = {"qOutput", "gateOutput"};
            if (lastPageLens != nullptr) {
                datas["lastPageLens"] = lastPageLens;
                outputs.push_back("lastPageLens");
            }
            runner.Run("Qwen35QGateKVRMSNormRopeSplitAppendPagedCache",
                       datas,
                       FloatDict{{"eps", eps}, {"ropeTheta", ropeTheta}, {"ropeScale", ropeScale}},
                       IntDict{{"q_heads", qHeads}, {"k_heads", kHeads}, {"head_dim", headDim},
                               {"rotaryDim", rotaryDim},
                               {"sectionT", sections.size() > 0 ? sections[0] : 0},
                               {"sectionH", sections.size() > 1 ? sections[1] : 0},
                               {"sectionW", sections.size() > 2 ? sections[2] : 0},
                               {"pageLen", pageLen}, {"batch", batch},
                               {"doQKNorm", (int)doQKNorm}},
                       outputs);
        }

        static void Qwen35AdvancePagedCacheLogicalTokens(Data &cache, int tokens) {
            if (tokens <= 0) {
                return;
            }
            if (cache.dims.size() >= 2) {
                cache.dims[1] += tokens;
                if (cache.expansionDims.empty()) {
                    cache.strides.assign(cache.dims.size(), 1);
                    for (int i = (int)cache.dims.size() - 2; i >= 0; i--) {
                        cache.strides[i] = (uint64_t)cache.dims[i + 1] *
                                           cache.strides[i + 1];
                    }
                }
            }
        }

        static void Qwen35CudaAttentionPagedBlock(
                Qwen3CudaDirectRunner &runner,
                Data *attenInput,
                Data *mergeQkvWeight, Data *mergeQkvBias,
                Data *qNormWeight, Data *kNormWeight,
                Data *oWeight, Data *oBias,
                Data *allPositionIds,
                std::vector<std::pair<Data*, Data*> > *pastKeyValues,
                std::vector<Data*> *batchPastKeys,
                std::vector<Data*> *batchPastValues,
                Data *merged, Data *qgate, Data *gate,
                Data *q, Data *k, Data *v,
                Data *attenOutput, Data *attenLastOutput,
                Data *qForAttentionHolder,
                Data *insertIndexs, Data *insertPositions,
                Data *qSizes, Data *pageSizes,
                Data *pageIndexs, Data *lastPageLens,
                bool *generatedAppendParams,
                bool *generatedDecodeParams,
                int batch, int blockCnt, int layerIdx,
                const std::vector<int> &seqLens,
                int numAttentionHeads, int numKeyValueHeads, int headDim,
                int rotaryDim, const std::vector<int> &mropeSections,
                float rmsNormEps, float ropeBase, float ropeFactor,
                int ropeType,
                bool isPrefill,
                Data *hiddenStates,
                int pagedCacheLayerOffset,
                bool skipOutputProjection,
                bool externalDecodeMeta,
                bool enableFlashInferCudaGraph = false,
                int flashInferCudaGraph = -1) {
            using namespace qwen3cuda;
            AssertInFastLLM(attenInput != nullptr && mergeQkvWeight != nullptr &&
                            mergeQkvBias != nullptr && qNormWeight != nullptr &&
                            kNormWeight != nullptr && allPositionIds != nullptr &&
                            pastKeyValues != nullptr && batchPastKeys != nullptr &&
                            batchPastValues != nullptr && merged != nullptr &&
                            qgate != nullptr && gate != nullptr && q != nullptr &&
                            k != nullptr && v != nullptr && attenOutput != nullptr &&
                            qForAttentionHolder != nullptr && insertIndexs != nullptr &&
                            insertPositions != nullptr && qSizes != nullptr &&
                            pageSizes != nullptr && pageIndexs != nullptr &&
                            lastPageLens != nullptr && generatedAppendParams != nullptr &&
                            generatedDecodeParams != nullptr,
                            "Qwen3.5 gated paged attention block got null input.\n");
            AssertInFastLLM(numAttentionHeads > 0 && numKeyValueHeads > 0 &&
                            headDim > 0 && numAttentionHeads % numKeyValueHeads == 0,
                            "Qwen3.5 gated paged attention block got invalid head metadata.\n");

            Qwen3CudaLinear(runner, *attenInput, *mergeQkvWeight, *mergeQkvBias, *merged);

            int bsz = attenInput->dims[0];
            int seqlen = attenInput->dims[1];
            int group = numAttentionHeads / numKeyValueHeads;
            float ropeScale = (ropeType == RoPEType::LINEAR_SCALE) ? ropeFactor : 1.0f;

            for (int b = 0; b < batch; b++) {
                (*batchPastKeys)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].first;
                (*batchPastValues)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].second;
            }

            auto makeCacheDesc = [](const Data &src, DataType targetType) {
                Data desc(targetType);
                desc.dims = src.dims;
                desc.strides = src.strides;
                desc.dataDevice = src.dataDevice;
                desc.dataDeviceIds = src.dataDeviceIds;
                desc.multiDeviceData = src.multiDeviceData;
                desc.tpLayout = src.tpLayout;
                desc.tpAxis = src.tpAxis;
                desc.tpGlobalDims = src.tpGlobalDims;
                desc.tpRanges = src.tpRanges;
                desc.UpdateUnitSize();
                return desc;
            };
            auto preparePagedAttentionQ = [&](Data &src, DataType cacheType) -> Data& {
                DataType targetType;
                if (cacheType == DataType::FLOAT16 || cacheType == DataType::BFLOAT16) {
                    targetType = cacheType;
                } else if (src.dataType == DataType::FLOAT16 || src.dataType == DataType::BFLOAT16) {
                    targetType = src.dataType;
                } else {
                    targetType = attenInput->dataType == DataType::BFLOAT16 ?
                        DataType::BFLOAT16 : DataType::FLOAT16;
                }
                if (src.dataType == targetType) {
                    return src;
                }
                Qwen3CudaConvertToDataType(runner, src, *qForAttentionHolder, targetType);
                return *qForAttentionHolder;
            };

            if (!isPrefill && (*batchPastKeys)[0]->pagedKVCacheData == nullptr) {
                isPrefill = true;
            }

            if (isPrefill) {
                int qgateDim = numAttentionHeads * headDim * 2;
                int kvDim = numKeyValueHeads * headDim;
                Qwen3CudaSplit(runner, *merged, -1, 0, qgateDim, *qgate);
                Qwen3CudaSplit(runner, *merged, -1, qgateDim, qgateDim + kvDim, *k);
                Qwen3CudaSplit(runner, *merged, -1, qgateDim + kvDim, qgateDim + kvDim * 2, *v);

                qgate->Reshape({bsz, seqlen, numAttentionHeads, headDim * 2});
                Qwen3CudaSplit(runner, *qgate, -1, 0, headDim, *q);
                Qwen3CudaSplit(runner, *qgate, -1, headDim, headDim * 2, *gate);
                gate->Reshape({bsz, seqlen, numAttentionHeads * headDim});
                k->Reshape({bsz, seqlen, numKeyValueHeads, headDim});
                v->Reshape({bsz, seqlen, numKeyValueHeads, headDim});

                Qwen3CudaRMSNorm(runner, *q, *qNormWeight, rmsNormEps, *q);
                Qwen3CudaRMSNorm(runner, *k, *kNormWeight, rmsNormEps, *k);
                Qwen35CudaApplyRotary(runner, *q, *allPositionIds,
                                      rotaryDim, mropeSections, ropeBase, ropeScale);
                Qwen35CudaApplyRotary(runner, *k, *allPositionIds,
                                      rotaryDim, mropeSections, ropeBase, ropeScale);

                Qwen3CudaPermuteSelf(runner, *q, {0, 2, 1, 3});
                Qwen3CudaPermuteSelf(runner, *k, {0, 2, 1, 3});
                Qwen3CudaPermuteSelf(runner, *v, {0, 2, 1, 3});
                q->Reshape({-1, seqlen, headDim});
                k->Reshape({-1, seqlen, headDim});
                v->Reshape({-1, seqlen, headDim});

                if (batch == 1) {
                    Data &pastKey = *(*batchPastKeys)[0];
                    Data &pastValue = *(*batchPastValues)[0];
                    Data kCacheDesc = makeCacheDesc(*k, pastKey.dataType);
                    Data vCacheDesc = makeCacheDesc(*v, pastValue.dataType);
                    int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, pastKey, *k);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, pastValue, *v);
                } else {
                    int total = 0;
                    Data curK, curV;
                    for (int b = 0; b < batch; b++) {
                        Data &pastKey = *(*batchPastKeys)[b];
                        Data &pastValue = *(*batchPastValues)[b];
                        Qwen3CudaSplit(runner, *k, 1, total, total + seqLens[b], curK);
                        Qwen3CudaSplit(runner, *v, 1, total, total + seqLens[b], curV);
                        Data kCacheDesc = makeCacheDesc(curK, pastKey.dataType);
                        Data vCacheDesc = makeCacheDesc(curV, pastValue.dataType);
                        int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                        PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                            cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                        PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                            cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                        Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, pastKey, curK);
                        Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, pastValue, curV);
                        total += seqLens[b];
                    }
                }

                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];
                Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType);
                Qwen3CudaGeneratePagedBatchParams(runner, qForAttention, *batchPastKeys, batch,
                                                  *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                                  seqLens);
                Qwen3CudaAttentionPagedBatch(runner, qForAttention,
                                             kCaches, vCaches,
                                             *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                             *attenOutput, group,
                                             1.0f / std::sqrt((float)headDim),
                                             1, layerIdx > 0,
                                             enableFlashInferCudaGraph,
                                             flashInferCudaGraph);
                attenOutput->Reshape({1, seqlen, numAttentionHeads * headDim});
            } else {
                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];
                PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                "Qwen3.5 gated paged attention decode requires paged KV cache.\n");

                if (!externalDecodeMeta && !(*generatedAppendParams)) {
                    Qwen3CudaGenerateAppendPagedCacheBatchParams(runner, *pagedCacheKManager,
                                                                 *batchPastKeys, batch,
                                                                 *insertIndexs, *insertPositions);
                    *generatedAppendParams = true;
                }

                bool fillLastPageLensOnDevice = merged->dataDevice == DataDevice::CUDA &&
                                                 !merged->multiDeviceData &&
                                                 !externalDecodeMeta &&
                                                 !(*generatedDecodeParams);
                Qwen35CudaQGateKVRMSNormRopeSplitAppendPagedCache(
                    runner, *merged, *qNormWeight, *kNormWeight, *allPositionIds,
                    *q, *gate,
                    *(Data*)pagedCacheKManager, *(Data*)pagedCacheVManager,
                    *insertIndexs, *insertPositions,
                    numAttentionHeads, numKeyValueHeads, headDim,
                    rotaryDim, mropeSections, rmsNormEps, ropeBase, ropeScale,
                    kCaches.pageLen, batch, true,
                    fillLastPageLensOnDevice ? lastPageLens : nullptr);

                if (!externalDecodeMeta) {
                    for (int b = 0; b < batch; b++) {
                        auto updatePageMeta = [](Data *cache, PagedCacheManager *mgr) {
                            if (cache->pageIndex.empty() || cache->lastPageLen >= cache->pageLen) {
                                cache->pageIndex.push_back(mgr->GetUnusedPageIndex(true));
                                cache->lastPageLen = 1;
                            } else {
                                cache->lastPageLen++;
                            }
                            Qwen35AdvancePagedCacheLogicalTokens(*cache, 1);
                        };
                        updatePageMeta((*batchPastKeys)[b], pagedCacheKManager);
                        updatePageMeta((*batchPastValues)[b], pagedCacheVManager);
                    }
                }
                if (!externalDecodeMeta && !(*generatedDecodeParams)) {
                    Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType);
                    Qwen3CudaGeneratePagedBatchParams(runner, qForAttention, *batchPastKeys, batch,
                                                      *qSizes, *pageSizes, *pageIndexs,
                                                      *lastPageLens, std::vector<int>(),
                                                      fillLastPageLensOnDevice);
                    *generatedDecodeParams = true;
                }
                Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType);
                Qwen3CudaAttentionPagedBatch(runner, qForAttention,
                                             kCaches, vCaches,
                                             *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                             *attenOutput, group,
                                             1.0f / std::sqrt((float)headDim),
                                             1, layerIdx > 0,
                                             enableFlashInferCudaGraph,
                                             flashInferCudaGraph);
                if (bsz == 1 && seqlen == 1) {
                    attenOutput->Reshape({bsz, seqlen, -1});
                } else {
                    attenOutput->Reshape({seqlen, bsz, -1});
                    Qwen3CudaPermuteSelf(runner, *attenOutput, {1, 0, 2});
                }
            }

            Qwen35CudaSigmoid(runner, *gate, *gate);
            if (gate->dataType != attenOutput->dataType) {
                Qwen3CudaToDataType(runner, *gate, attenOutput->dataType);
            }
            Qwen35CudaMulTo(runner, *attenOutput, *gate);

            if (!skipOutputProjection) {
                Qwen3CudaLinearAddBlock(runner, attenOutput, oWeight, oBias,
                                        attenLastOutput, hiddenStates);
            }
        }

        static bool Qwen35CanUseCudaFullLogitsSampling(
                const std::vector<GenerationConfig> &generationConfigs,
                std::vector<std::vector<float>*> *retLogits,
                int batch,
                bool &allSimple,
                int &maxTopK) {
            allSimple = true;
            maxTopK = 1;
            for (int b = 0; b < batch; b++) {
                const GenerationConfig &config = generationConfigs[b];
                allSimple &= config.IsSimpleGreedy();
                if (config.output_logits && retLogits != nullptr &&
                    b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                    return false;
                }
                if (Qwen35NeedRepeatPenalty(config)) {
                    return false;
                }
                int curTopK = config.IsSimpleGreedy() ? 1 : config.top_k;
                if (curTopK <= 0 || curTopK > 50) {
                    return false;
                }
                maxTopK = std::max(maxTopK, curTopK);
            }
            return true;
        }

        static Data &Qwen35ThreadLocalCudaSamplingFullLogits() {
            static thread_local Data fullLogits(DataType::FLOAT32);
            return fullLogits;
        }

        static Data &Qwen35ThreadLocalCudaSamplingOutput() {
            static thread_local Data data(DataType::INT32);
            return data;
        }

        static void Qwen35ReleaseThreadLocalCudaSamplingBuffers() {
            Qwen35ThreadLocalCudaSamplingFullLogits().FreeSpace();
            Qwen35ThreadLocalCudaSamplingOutput().FreeSpace();
        }

        static long long Qwen35CudaRuntimeScratchReserveBytes() {
            return (32LL + 16LL + 8LL + 8LL) * 1024LL * 1024LL;
        }

        static void Qwen35WarmupCudaRuntimeScratchBuffers() {
            for (size_t bytes : {
                    32ULL * 1024ULL * 1024ULL,
                    16ULL * 1024ULL * 1024ULL,
                    8ULL * 1024ULL * 1024ULL,
                    8ULL * 1024ULL * 1024ULL}) {
                FastllmCudaMallocBigBuffer(bytes);
            }
        }

        static void Qwen35GatherShardLogitsToRootCuda(
                int rootDevice,
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                Data &fullLogits) {
            FastllmCudaSetDevice(rootDevice);
            Qwen3CudaPrepareLocalOutput(fullLogits, rootDevice);
            fullLogits.dataType = DataType::FLOAT32;
            fullLogits.UpdateUnitSize();
            fullLogits.Resize({batch, vocabSize});
            fullLogits.Allocate();

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Qwen3.5 CUDA sampling: missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "Qwen3.5 CUDA sampling: local logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localLogits[r].Count(0) / localVocab;
                AssertInFastLLM(rows == batch,
                                "Qwen3.5 CUDA sampling: local logits batch mismatch.\n");

                uint8_t *dstBase = (uint8_t*)fullLogits.cudaData;
                uint8_t *srcBase = (uint8_t*)localLogits[r].cudaData;
                int localOffset = 0;
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                    localOffset + len <= localVocab,
                                    "Qwen3.5 CUDA sampling: invalid lm_head split range.\n");
                    FastllmCudaMemcpy2DDeviceToDeviceAuto(
                        dstBase + (size_t)range.first * sizeof(float),
                        (size_t)vocabSize * sizeof(float),
                        srcBase + (size_t)localOffset * sizeof(float),
                        (size_t)localVocab * sizeof(float),
                        (size_t)len * sizeof(float),
                        (size_t)batch,
                        rootDevice,
                        device);
                    localOffset += len;
                }
            }
        }

        static std::vector<int> Qwen35SampleGreedyFromShardedCudaLogits(
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                bool resetEos,
                const std::vector<int> &eosIds) {
            AssertInFastLLM(batch > 0 && !devices.empty(),
                            "Qwen3.5 CUDA greedy shard sampling got empty input.\n");
            std::vector<std::vector<int> > localBestIds(devices.size());
            std::vector<std::vector<float> > localBestScores(devices.size());
            std::vector<Data> cudaBestIds(devices.size());
            std::vector<Data> cudaBestScores(devices.size());

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Qwen3.5 CUDA greedy shard sampling missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "Qwen3.5 CUDA greedy shard sampling logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localVocab > 0 ? (int)(localLogits[r].Count(0) / localVocab) : 0;
                AssertInFastLLM(rows == batch,
                                "Qwen3.5 CUDA greedy shard sampling batch mismatch.\n");

                FastllmCudaSetDevice(device);
                if (resetEos && !eosIds.empty()) {
                    std::vector<int> localEosIds;
                    int localOffset = 0;
                    for (auto &range : schemeIt->second) {
                        int len = range.second - range.first;
                        for (int id : eosIds) {
                            if (id >= range.first && id < range.second) {
                                localEosIds.push_back(localOffset + id - range.first);
                            }
                        }
                        localOffset += len;
                    }
                    if (!localEosIds.empty()) {
                        FastllmResetLogitsOfEOSAll(batch, &localLogits[r], localEosIds);
                    }
                }

                Qwen3CudaPrepareLocalOutput(cudaBestIds[r], device);
                Qwen3CudaPrepareLocalOutput(cudaBestScores[r], device);
                cudaBestIds[r].dataType = DataType::INT32;
                cudaBestScores[r].dataType = DataType::FLOAT32;
                cudaBestIds[r].UpdateUnitSize();
                cudaBestScores[r].UpdateUnitSize();
                cudaBestIds[r].Resize({batch});
                cudaBestScores[r].Resize({batch});
                cudaBestIds[r].Allocate();
                cudaBestScores[r].Allocate();
                bool sampled = FastllmCudaGreedySamplingWithScores(
                    (float*)localLogits[r].cudaData,
                    (int*)cudaBestIds[r].cudaData,
                    (float*)cudaBestScores[r].cudaData,
                    batch, localVocab);
                AssertInFastLLM(sampled,
                                "Qwen3.5 CUDA greedy shard sampling failed.\n");

                localBestIds[r].resize(batch);
                localBestScores[r].resize(batch);
                FastllmCudaCopyFromDeviceToHost(localBestIds[r].data(),
                                                cudaBestIds[r].cudaData,
                                                (size_t)batch * sizeof(int));
                FastllmCudaCopyFromDeviceToHost(localBestScores[r].data(),
                                                cudaBestScores[r].cudaData,
                                                (size_t)batch * sizeof(float));
            }

            std::vector<int> ret(batch, 0);
            std::vector<float> bestScores(batch, -1.0e30f);
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Qwen3.5 CUDA greedy shard sampling missing lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    int localId = localBestIds[r][b];
                    int globalId = -1;
                    int localOffset = 0;
                    for (auto &range : schemeIt->second) {
                        int len = range.second - range.first;
                        if (localId >= localOffset && localId < localOffset + len) {
                            globalId = range.first + localId - localOffset;
                            break;
                        }
                        localOffset += len;
                    }
                    AssertInFastLLM(globalId >= 0 && globalId < vocabSize,
                                    "Qwen3.5 CUDA greedy shard sampling invalid local argmax.\n");
                    float score = localBestScores[r][b];
                    if (score > bestScores[b] ||
                        (score == bestScores[b] && globalId < ret[b])) {
                        bestScores[b] = score;
                        ret[b] = globalId;
                    }
                }
            }
            return ret;
        }

        static std::vector<int> Qwen35SampleFromRootCudaLogits(
                int rootDevice,
                Data &fullLogits,
                int batch,
                int maxTopK,
                bool allSimple,
                const std::vector<GenerationConfig> &generationConfigs) {
            FastllmCudaSetDevice(rootDevice);
            std::vector<int> lastRet;
            lastRet.reserve(batch);
            if (!allSimple) {
                static thread_local std::vector<float> temperatures;
                static thread_local std::vector<int> topKs;
                static thread_local std::vector<float> topPs;
                temperatures.resize(batch);
                topKs.resize(batch);
                topPs.resize(batch);
                for (int b = 0; b < batch; b++) {
                    temperatures[b] = std::max(generationConfigs[b].temperature, 1.0e-6f);
                    topKs[b] = generationConfigs[b].top_k;
                    topPs[b] = generationConfigs[b].top_p;
                }
                lastRet.resize(batch);
                int vocabSize = fullLogits.dims.back();
                FastllmCudaTopKTopPSampling((float*)fullLogits.cudaData,
                                            temperatures.data(), topKs.data(), topPs.data(),
                                            lastRet.data(), batch, vocabSize);
                return lastRet;
            }

            Data &cudaOutput = Qwen35ThreadLocalCudaSamplingOutput();
            Qwen3CudaPrepareLocalOutput(cudaOutput, rootDevice);
            cudaOutput.dataType = DataType::INT32;
            cudaOutput.UpdateUnitSize();
            cudaOutput.Resize({batch});
            cudaOutput.Allocate();

            int vocabSize = fullLogits.dims.back();
            FastllmCudaGreedySampling((float*)fullLogits.cudaData,
                                      (int*)cudaOutput.cudaData,
                                      batch, vocabSize);
            lastRet.resize(batch);
            FastllmCudaCopyFromDeviceToHost(lastRet.data(), cudaOutput.cudaData,
                                            (size_t)batch * sizeof(int));
            return lastRet;
        }
    }
#else
    namespace {
        struct Qwen35LinearPrefixSnapshotCache {
            bool valid = false;
        };

        struct Qwen35LinearPrefixSnapshotLayer {
            bool linear = false;
            Qwen35LinearPrefixSnapshotCache first;
            Qwen35LinearPrefixSnapshotCache second;
        };

        struct Qwen35LinearPrefixSnapshot {
            int cachedLen = 0;
            int requestId = 0;
            long long timestamp = 0;
            std::vector<int> tokens;
            std::vector<Qwen35LinearPrefixSnapshotLayer> layers;
            bool mtpValid = false;
            int mtpTokens = 0;
            Data mtpKey;
            Data mtpValue;
        };

        static std::mutex &Qwen35LinearPrefixSnapshotsMutex() {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        static std::map<const Qwen3_5Model*, std::vector<std::unique_ptr<Qwen35LinearPrefixSnapshot> > >
                &Qwen35LinearPrefixSnapshots() {
            static auto *snapshots =
                new std::map<const Qwen3_5Model*, std::vector<std::unique_ptr<Qwen35LinearPrefixSnapshot> > >();
            return *snapshots;
        }

        static long long &Qwen35LinearPrefixSnapshotTimestamp() {
            static auto *timestamp = new long long(0);
            return *timestamp;
        }

        static std::atomic<int> &Qwen35LinearPrefixSnapshotRequestCounter() {
            static auto *counter = new std::atomic<int>(0);
            return *counter;
        }

        static bool Qwen35LinearPrefixCacheEnabled() {
            return false;
        }

        static int Qwen35LinearPrefixSnapshotIntervalTokens() {
            return fastllm::GetPageLen();
        }

        static int Qwen35LinearPrefixSnapshotMaxPerRequest() {
            return 1;
        }

        static int Qwen35LinearPrefixSnapshotMaxRecords() {
            return 1;
        }

        static bool Qwen35LayerIsLinearAttention(const Qwen3_5Model *model, int layer) {
            std::string prefix = Qwen3_5Model::language_prefix + "layers." + std::to_string(layer) + ".";
            return model->weight.weight.find(prefix + "self_attn.o_proj.weight") == model->weight.weight.end();
        }

        static bool Qwen35HasLinearAttentionLayers(const Qwen3_5Model *model, int blockCnt) {
            for (int i = 0; i < blockCnt; i++) {
                if (Qwen35LayerIsLinearAttention(model, i)) {
                    return true;
                }
            }
            return false;
        }

        static int Qwen35CurrentTokenGrowingCacheLen(
                const Qwen3_5Model *model,
                int blockCnt,
                const std::vector<std::pair<Data, Data> > &pastKeyValues) {
            (void)model;
            (void)blockCnt;
            (void)pastKeyValues;
            return 0;
        }

        static bool Qwen35SnapshotCopyCache(
                const Data &src,
                Qwen35LinearPrefixSnapshotCache &dst) {
            (void)src;
            (void)dst;
            return false;
        }

        static bool Qwen35RestoreSnapshotCache(
                const Qwen35LinearPrefixSnapshotCache &snapshot,
                Data &dst) {
            (void)snapshot;
            (void)dst;
            return false;
        }

        static const Qwen35LinearPrefixSnapshot *Qwen35FindLinearPrefixSnapshotLocked(
                const Qwen3_5Model *model,
                const std::vector<int> &tokens,
                int maxCachedLen,
                int exactLen = -1,
                bool requireMtp = false) {
            (void)model;
            (void)tokens;
            (void)maxCachedLen;
            (void)exactLen;
            (void)requireMtp;
            return nullptr;
        }

        static void Qwen35EraseLinearPrefixSnapshots(const Qwen3_5Model *model) {
            (void)model;
        }
    }
#endif

    static DataType Qwen35LinearAttentionCacheDataType(DataType modelType) {
        if (modelType == DataType::FLOAT32 ||
            modelType == DataType::FLOAT16 ||
            modelType == DataType::BFLOAT16) {
            return modelType;
        }
        return DataType::FLOAT16;
    }

    static void Qwen35PrepareLinearAttentionCache(Data &cache, DataType cacheType) {
        cache.isKVCache = true;
        cache.isLinearAttention = true;
        if (cache.dims.empty() && cache.dataType != cacheType) {
            cache.dataType = cacheType;
            cache.UpdateUnitSize();
        }
    }

    static void Add1(Data &input) {
        if (input.dims.size() == 0) {
            return;
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    static bool IsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return lowered == "1" || lowered == "true" || lowered == "on";
    }

    static bool CanUseSingleRowLastDimView(const Data &input) {
        if (input.dims.empty() || input.unitSizeDiv != 1 || input.strides.empty() || input.strides.back() != 1) {
            return false;
        }
        int outer = 1;
        for (int i = 0; i + 1 < (int) input.dims.size(); i++) {
            outer *= input.dims[i];
        }
        return outer == 1;
    }

    static void MakeSingleRowLastDimView(const Data &input, int start, int end, Data &output) {
        AssertInFastLLM(CanUseSingleRowLastDimView(input),
                        "Single-row last-dim view requires a contiguous one-row tensor.");
        AssertInFastLLM(start >= 0 && end >= start && end <= input.dims.back(),
                        "Single-row last-dim view range is out of bounds.");

        std::vector<int> dims = input.dims;
        dims.back() = end - start;
        if (input.dataDevice == DataDevice::CPU) {
            output = Data(input.dataType, dims, input.dataDevice, input.cpuData + (size_t) start * input.unitSize);
        } else if (input.dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            output = Data(input.dataType, dims, input.dataDevice, (uint8_t*) input.cudaData + (size_t) start * input.unitSize);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        } else {
            ErrorInFastLLM("Single-row last-dim view only supports CPU/CUDA tensors.");
        }
        output.dataDeviceIds = input.dataDeviceIds;
    }

    static void ShiftAppendSingleTokenLinearAttentionCache(Data &cache, const Data &newToken) {
        AssertInFastLLM(cache.dataType == newToken.dataType && cache.unitSizeDiv == 1 && newToken.unitSizeDiv == 1,
                        "Linear attention decode cache update only supports float-like tensors.");
        AssertInFastLLM(cache.dataDevice == newToken.dataDevice,
                        "Linear attention decode cache update expects the same device.");
        AssertInFastLLM(cache.dims.size() == 3 && newToken.dims.size() == 3 &&
                        cache.dims[0] == newToken.dims[0] &&
                        cache.dims[1] == newToken.dims[1] && newToken.dims[2] == 1 &&
                        cache.strides.back() == 1 && newToken.strides.back() == 1,
                        "Linear attention decode cache update expects [batch, channels, window] and [batch, channels, 1].");

        int rows = cache.dims[0] * cache.dims[1];
        int window = cache.dims[2];
        int unitSize = cache.unitSize;
        if (cache.dataDevice == DataDevice::CPU) {
            for (int c = 0; c < rows; c++) {
                uint8_t *cacheRow = cache.cpuData + (size_t) c * window * unitSize;
                const uint8_t *newTokenRow = newToken.cpuData + (size_t) c * unitSize;
                memmove(cacheRow, cacheRow + unitSize, (size_t) (window - 1) * unitSize);
                memcpy(cacheRow + (size_t) (window - 1) * unitSize, newTokenRow, unitSize);
            }
        } else if (cache.dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            FastllmCudaShiftAppendWindow((uint8_t*) cache.cudaData, (const uint8_t*) newToken.cudaData,
                                         rows, window, unitSize);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        } else {
            ErrorInFastLLM("Linear attention decode cache update only supports CPU/CUDA tensors.");
        }
    }

    static void CatBatchFirstDim(const std::vector<Data*> &inputs, Data &output) {
        AssertInFastLLM(!inputs.empty(), "CatBatchFirstDim expects non-empty inputs.");
        Data *first = inputs[0];
        AssertInFastLLM(first != nullptr && !first->dims.empty() && first->dims[0] == 1,
                        "CatBatchFirstDim expects inputs with first dimension 1.");

        std::vector<int> dims = first->dims;
        dims[0] = (int) inputs.size();
        output.dataType = first->dataType;
        output.dataDevice = first->dataDevice;
        output.dataDeviceIds = first->dataDeviceIds;
        output.Resize(dims);
        output.Allocate(false);

        size_t inputBytes = first->GetBytes();
        for (int i = 0; i < (int) inputs.size(); i++) {
            Data *cur = inputs[i];
            AssertInFastLLM(cur != nullptr && cur->dataType == first->dataType &&
                            cur->dataDevice == first->dataDevice && cur->dims == first->dims,
                            "CatBatchFirstDim expects matching input tensors.");
            if (output.dataDevice == DataDevice::CPU) {
                memcpy(output.cpuData + (size_t) i * inputBytes, cur->cpuData, inputBytes);
            } else if (output.dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                FastllmCudaCopyFromDeviceToDevice((uint8_t*) output.cudaData + (size_t) i * inputBytes,
                                                  cur->cudaData, inputBytes);
#else
                ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
            } else {
                ErrorInFastLLM("CatBatchFirstDim only supports CPU and CUDA tensors.");
            }
        }
    }

    static void SplitBatchFirstDim(const Data &input, const std::vector<Data*> &outputs) {
        AssertInFastLLM(!input.dims.empty() && input.dims[0] == (int) outputs.size(),
                        "SplitBatchFirstDim expects input first dimension to match outputs.");
        std::vector<int> dims = input.dims;
        dims[0] = 1;
        size_t outputBytes = input.GetBytes() / input.dims[0];
        for (int i = 0; i < (int) outputs.size(); i++) {
            Data *cur = outputs[i];
            AssertInFastLLM(cur != nullptr, "SplitBatchFirstDim expects non-null outputs.");
            cur->dataType = input.dataType;
            cur->dataDevice = input.dataDevice;
            cur->dataDeviceIds = input.dataDeviceIds;
            cur->Resize(dims);
            cur->Allocate(false);
            if (input.dataDevice == DataDevice::CPU) {
                memcpy(cur->cpuData, input.cpuData + (size_t) i * outputBytes, outputBytes);
            } else if (input.dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                FastllmCudaCopyFromDeviceToDevice(cur->cudaData,
                                                  (uint8_t*) input.cudaData + (size_t) i * outputBytes,
                                                  outputBytes);
#else
                ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
            } else {
                ErrorInFastLLM("SplitBatchFirstDim only supports CPU and CUDA tensors.");
            }
        }
    }

    static DataType GetPagedAttentionDataType(const Data &q, DataType preferredType) {
        if (q.dataType == DataType::FLOAT16 || q.dataType == DataType::BFLOAT16) {
            return q.dataType;
        }
        return preferredType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
    }

    static void PreparePagedAttentionInputs(Data &q, Data &k, Data &v, DataType preferredType) {
#ifdef USE_CUDA
        if (q.dataDevice != DataDevice::CUDA) {
            return;
        }
        DataType targetType = GetPagedAttentionDataType(q, preferredType);
        if (q.dataType != targetType) {
            ToDataType(q, targetType);
        }
        if (k.dataType != targetType) {
            ToDataType(k, targetType);
        }
        if (v.dataType != targetType) {
            ToDataType(v, targetType);
        }
#else
        (void) q;
        (void) k;
        (void) v;
        (void) preferredType;
#endif
    }

    static void SwapSingleTokenSeqHeadByReshape(Data &input) {
        AssertInFastLLM(input.dims.size() == 3 || input.dims.size() == 4,
                        "Single-token seq/head reshape only supports 3D or 4D tensors.");
        AssertInFastLLM(input.dims.size() >= 3 && input.dims[0] == 1 && (input.dims[1] == 1 || input.dims[2] == 1),
                        "Single-token seq/head reshape expects batch=1 and one singleton seq/head axis.");

        std::vector<int> dims = input.dims;
        std::swap(dims[1], dims[2]);
        input.Reshape(dims);
    }

#ifdef USE_CUDA
    static bool Qwen35EnsureCudaLinearAttnStateTransposed(Data &state) {
        if (state.isLinearAttentionTransposed) {
            return true;
        }
        if (FastllmLinearAttentionStateTransposeKVToVKFloat16(state)) {
            state.isLinearAttentionTransposed = true;
            return true;
        }
        return false;
    }

    static bool Qwen35EnsureCudaLinearAttnStateKVLayout(Data &state) {
        if (!state.isLinearAttentionTransposed) {
            return true;
        }
        if (FastllmLinearAttentionStateTransposeVKToKVFloat16(state)) {
            state.isLinearAttentionTransposed = false;
            return true;
        }
        return false;
    }

    static bool Qwen35TryCudaLinearAttnSingleDecodeNormRecurrent(
            Data &q, Data &k, Data &v, Data &g, Data &b,
            Data &normWeight, float eps, Data &lastRecurrentState, Data &coreAttnOut) {
        if (q.dataDevice != DataDevice::CUDA ||
            k.dataDevice != DataDevice::CUDA ||
            v.dataDevice != DataDevice::CUDA ||
            g.dataDevice != DataDevice::CUDA ||
            b.dataDevice != DataDevice::CUDA ||
            q.dataType != DataType::FLOAT16 ||
            k.dataType != DataType::FLOAT16 ||
            v.dataType != DataType::FLOAT16 ||
            g.dataType != DataType::FLOAT16 ||
            b.dataType != DataType::FLOAT16 ||
            q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
            b.dims.size() != 3 || g.dims.size() != 3 ||
            q.dims[0] != 1 || k.dims[0] != 1 || v.dims[0] != 1 ||
            q.dims[1] != 1 || k.dims[1] != 1 || v.dims[1] != 1 ||
            b.dims[0] != 1 || g.dims[0] != 1 ||
            b.dims[1] != 1 || g.dims[1] != 1 ||
            q.dims[3] != 128 || k.dims[3] != 128 ||
            q.dims[2] <= 0 || q.dims[2] != k.dims[2] ||
            v.dims[2] <= 0 || b.dims[2] != v.dims[2] || g.dims[2] != v.dims[2] ||
            lastRecurrentState.dataDevice != DataDevice::CUDA ||
            lastRecurrentState.dataType != DataType::FLOAT16 ||
            lastRecurrentState.dims.size() != 4 ||
            lastRecurrentState.dims[0] != 1 ||
            lastRecurrentState.dims[1] != v.dims[2] ||
            lastRecurrentState.dims[2] != q.dims[3] ||
            lastRecurrentState.dims[3] != v.dims[3] ||
            lastRecurrentState.dims[1] % q.dims[2] != 0) {
            return false;
        }

        if (normWeight.dataDevice != DataDevice::CUDA) {
            if (!q.dataDeviceIds.empty()) {
                normWeight.ToDevice(DataDevice::CUDA, q.dataDeviceIds);
            } else {
                normWeight.ToDevice(DataDevice::CUDA);
            }
        }
        if (normWeight.dataDevice != DataDevice::CUDA ||
            normWeight.dataType != DataType::FLOAT32 ||
            normWeight.dims.size() != 1 ||
            normWeight.dims[0] != q.dims[3]) {
            return false;
        }

        SwapSingleTokenSeqHeadByReshape(q);
        SwapSingleTokenSeqHeadByReshape(k);
        SwapSingleTokenSeqHeadByReshape(v);
        SwapSingleTokenSeqHeadByReshape(b);
        SwapSingleTokenSeqHeadByReshape(g);

        float recurrentQScale = 1.0f / std::sqrt((float)q.dims.back());
        bool stateLayoutReady = Qwen35EnsureCudaLinearAttnStateTransposed(lastRecurrentState);
        bool fused = stateLayoutReady &&
            FastllmRecurrentGatedDeltaRuleNormTransposedFloat16(
                q, k, v, g, b, normWeight, lastRecurrentState, coreAttnOut, eps, recurrentQScale
            );
        if (fused) {
            SwapSingleTokenSeqHeadByReshape(coreAttnOut);
            return true;
        }
        if (stateLayoutReady) {
            Qwen35EnsureCudaLinearAttnStateKVLayout(lastRecurrentState);
        }

        SwapSingleTokenSeqHeadByReshape(q);
        SwapSingleTokenSeqHeadByReshape(k);
        SwapSingleTokenSeqHeadByReshape(v);
        SwapSingleTokenSeqHeadByReshape(b);
        SwapSingleTokenSeqHeadByReshape(g);
        return false;
    }

    static bool Qwen35TryCudaLinearAttnSingleDecodeNormBaRecurrent(
            Data &q, Data &k, Data &v, Data &a, Data &b,
            Data &normWeight, Data &aLog, Data &dtBias,
            float eps, Data &lastRecurrentState, Data &coreAttnOut) {
        if (q.dataDevice != DataDevice::CUDA ||
            k.dataDevice != DataDevice::CUDA ||
            v.dataDevice != DataDevice::CUDA ||
            a.dataDevice != DataDevice::CUDA ||
            b.dataDevice != DataDevice::CUDA ||
            q.dataType != DataType::FLOAT16 ||
            k.dataType != DataType::FLOAT16 ||
            v.dataType != DataType::FLOAT16 ||
            a.dataType != DataType::FLOAT16 ||
            b.dataType != DataType::FLOAT16 ||
            q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
            a.dims.size() != 3 || b.dims.size() != 3 ||
            q.dims[0] != 1 || k.dims[0] != 1 || v.dims[0] != 1 ||
            q.dims[1] != 1 || k.dims[1] != 1 || v.dims[1] != 1 ||
            a.dims[0] != 1 || b.dims[0] != 1 ||
            a.dims[1] != 1 || b.dims[1] != 1 ||
            q.dims[3] != 128 || k.dims[3] != 128 ||
            q.dims[2] <= 0 || q.dims[2] != k.dims[2] ||
            v.dims[2] <= 0 || a.dims[2] != v.dims[2] || b.dims[2] != v.dims[2] ||
            lastRecurrentState.dataDevice != DataDevice::CUDA ||
            lastRecurrentState.dataType != DataType::FLOAT16 ||
            lastRecurrentState.dims.size() != 4 ||
            lastRecurrentState.dims[0] != 1 ||
            lastRecurrentState.dims[1] != v.dims[2] ||
            lastRecurrentState.dims[2] != q.dims[3] ||
            lastRecurrentState.dims[3] != v.dims[3] ||
            lastRecurrentState.dims[1] % q.dims[2] != 0) {
            return false;
        }

        auto moveToQDevice = [&](Data &data) {
            if (data.dataDevice != DataDevice::CUDA) {
                if (!q.dataDeviceIds.empty()) {
                    data.ToDevice(DataDevice::CUDA, q.dataDeviceIds);
                } else {
                    data.ToDevice(DataDevice::CUDA);
                }
            }
        };
        moveToQDevice(normWeight);
        moveToQDevice(aLog);
        moveToQDevice(dtBias);
        if (normWeight.dataDevice != DataDevice::CUDA ||
            aLog.dataDevice != DataDevice::CUDA ||
            dtBias.dataDevice != DataDevice::CUDA ||
            normWeight.dataType != DataType::FLOAT32 ||
            aLog.dataType != DataType::FLOAT32 ||
            dtBias.dataType != DataType::FLOAT32 ||
            normWeight.dims.size() != 1 ||
            aLog.dims.size() != 1 ||
            dtBias.dims.size() != 1 ||
            normWeight.dims[0] != q.dims[3] ||
            aLog.dims[0] != v.dims[2] ||
            dtBias.dims[0] != v.dims[2]) {
            return false;
        }

        SwapSingleTokenSeqHeadByReshape(q);
        SwapSingleTokenSeqHeadByReshape(k);
        SwapSingleTokenSeqHeadByReshape(v);
        SwapSingleTokenSeqHeadByReshape(a);
        SwapSingleTokenSeqHeadByReshape(b);

        float recurrentQScale = 1.0f / std::sqrt((float)q.dims.back());
        bool stateLayoutReady = Qwen35EnsureCudaLinearAttnStateTransposed(lastRecurrentState);
        bool fused = stateLayoutReady &&
            FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16(
                q, k, v, a, b, normWeight, aLog, dtBias, lastRecurrentState,
                coreAttnOut, eps, recurrentQScale
            );
        if (fused) {
            SwapSingleTokenSeqHeadByReshape(coreAttnOut);
            return true;
        }
        if (stateLayoutReady) {
            Qwen35EnsureCudaLinearAttnStateKVLayout(lastRecurrentState);
        }

        SwapSingleTokenSeqHeadByReshape(q);
        SwapSingleTokenSeqHeadByReshape(k);
        SwapSingleTokenSeqHeadByReshape(v);
        SwapSingleTokenSeqHeadByReshape(a);
        SwapSingleTokenSeqHeadByReshape(b);
        return false;
    }
#endif

    static bool TryGetFusedMoeLayerPrefix(const std::string &weightName, std::string &layerPrefix) {
        static const std::string gateupSuffix = "experts.gate_up_proj";
        static const std::string downSuffix = "experts.down_proj";
        if (StringEndWith(weightName, gateupSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - gateupSuffix.size());
            return true;
        }
        if (StringEndWith(weightName, downSuffix)) {
            layerPrefix = weightName.substr(0, weightName.size() - downSuffix.size());
            return true;
        }
        return false;
    }

    static bool CanMergeLinearWeights(const Data &input0, const Data &input1) {
        return input0.dims.size() == 2 &&
               input1.dims.size() == 2 &&
               input0.dims[1] == input1.dims[1] &&
               input0.dataType == input1.dataType &&
               input0.ggmlType == input1.ggmlType &&
               input0.group == input1.group &&
               input0.groupCnt == input1.groupCnt &&
               input0.blockK == input1.blockK &&
               input0.blockM == input1.blockM &&
               input0.perChannelAxis == input1.perChannelAxis &&
               input0.cpuData != nullptr &&
               input1.cpuData != nullptr;
    }

    static bool CreateMergedLinearWeight(const Data &input0, const Data &input1,
                                         const std::string &mergeName, Data &mergeData) {
        if (!CanMergeLinearWeights(input0, input1)) {
            return false;
        }

        int dim0Len = input0.dims[0] + input1.dims[0];
        mergeData = Data(input0.dataType, {dim0Len, input0.dims[1]});
        mergeData.name = mergeName;
        mergeData.isModelWeight = true;
        mergeData.perChannelAxis = input0.perChannelAxis;
        mergeData.group = input0.group;
        mergeData.groupCnt = input0.groupCnt;
        mergeData.blockK = input0.blockK;
        mergeData.blockM = input0.blockM;
        mergeData.Allocate();

        uint64_t offset = 0;
        for (const Data *input : {&input0, &input1}) {
            mergeData.perChannelsConfigs = AppendVector(mergeData.perChannelsConfigs, input->perChannelsConfigs);
            mergeData.zeros = AppendVector(mergeData.zeros, input->zeros);
            mergeData.scales = AppendVector(mergeData.scales, input->scales);
            mergeData.mins = AppendVector(mergeData.mins, input->mins);
            mergeData.halfScales = AppendVector(mergeData.halfScales, input->halfScales);
            memcpy(mergeData.cpuData + offset, input->cpuData, input->GetBytes());
            offset += input->GetBytes();
        }
        mergeData.CalcWeightSum();
        return true;
    }

    static void SplitExpertLinearWeight(Data &dst, const Data &src, const std::string &name, int expertIndex) {
        AssertInFastLLM(src.dims.size() == 3, "Qwen3.5 MoE fused expert weight should be 3D.");
        AssertInFastLLM(expertIndex >= 0 && expertIndex < src.dims[0], "Qwen3.5 MoE expert index out of range.");
        AssertInFastLLM(src.dataType == DataType::FLOAT16 || src.dataType == DataType::BFLOAT16 ||
                        src.dataType == DataType::FLOAT32,
                        "Qwen3.5 MoE fused expert slicing currently supports float16/bfloat16/float32 weights only.");
        AssertInFastLLM(src.dataDevice == DataDevice::CPU && src.cpuData != nullptr,
                        "Qwen3.5 MoE fused expert slicing expects CPU weight data during load.");

        dst = Data(src.dataType, {src.dims[1], src.dims[2]});
        dst.Allocate();
        const uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expertIndex, bytesPerExpert);
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
    }

    static void RegisterExpertLinearWeight(Data &data, const std::string &weightType, bool registerFastllmData, bool registerNumas) {
        if (!registerFastllmData && !registerNumas) {
            return;
        }

        data.CalcWeightSum();

#ifdef USE_TFACC
        if (registerFastllmData) {
            data.weightSum.resize(1);
            RegisterFastllmData(&data, weightType);
        }
#endif

#if defined(USE_NUMAS)
        if (registerNumas) {
            RegisterNumas(&data, weightType);
        }
#endif
    }

    const std::string Qwen3_5Model::language_prefix = "model.language_model.";
    const std::string Qwen3_5Model::visual_prefix = "model.visual.";

    static inline int ClampInt(int value, int low, int high) {
        return std::max(low, std::min(value, high));
    }

    static float Qwen35BicubicWeight(float x) {
        const float a = -0.75f;
        x = fabsf(x);
        if (x <= 1.0f) {
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
        }
        return 0.0f;
    }

    static void ResizeRgbFrameBicubic(const float *src, int srcH, int srcW, int dstH, int dstW,
                                      std::vector<float> &dst) {
        dst.resize((size_t) dstH * dstW * 3);
        if (srcH == dstH && srcW == dstW) {
            memcpy(dst.data(), src, (size_t) srcH * srcW * 3 * sizeof(float));
            return;
        }
        const float scaleY = (float) srcH / (float) dstH;
        const float scaleX = (float) srcW / (float) dstW;
        for (int dy = 0; dy < dstH; dy++) {
            float srcY = ((float) dy + 0.5f) * scaleY - 0.5f;
            int baseY = (int) floorf(srcY);
            float wy[4];
            int iy[4];
            for (int ky = 0; ky < 4; ky++) {
                iy[ky] = ClampInt(baseY + ky - 1, 0, srcH - 1);
                wy[ky] = Qwen35BicubicWeight(srcY - (float) (baseY + ky - 1));
            }
            for (int dx = 0; dx < dstW; dx++) {
                float srcX = ((float) dx + 0.5f) * scaleX - 0.5f;
                int baseX = (int) floorf(srcX);
                float wx[4];
                int ix[4];
                for (int kx = 0; kx < 4; kx++) {
                    ix[kx] = ClampInt(baseX + kx - 1, 0, srcW - 1);
                    wx[kx] = Qwen35BicubicWeight(srcX - (float) (baseX + kx - 1));
                }
                float pixel[3] = {0.0f, 0.0f, 0.0f};
                for (int ky = 0; ky < 4; ky++) {
                    for (int kx = 0; kx < 4; kx++) {
                        float w = wy[ky] * wx[kx];
                        const float *srcPixel = src + ((size_t) iy[ky] * srcW + ix[kx]) * 3;
                        pixel[0] += srcPixel[0] * w;
                        pixel[1] += srcPixel[1] * w;
                        pixel[2] += srcPixel[2] * w;
                    }
                }
                float *dstPixel = dst.data() + ((size_t) dy * dstW + dx) * 3;
                dstPixel[0] = std::max(0.0f, std::min(255.0f, pixel[0]));
                dstPixel[1] = std::max(0.0f, std::min(255.0f, pixel[1]));
                dstPixel[2] = std::max(0.0f, std::min(255.0f, pixel[2]));
            }
        }
    }

    static void BuildQwen35VisionPatches(const float *rawData,
                                         int srcFrames,
                                         int srcH,
                                         int srcW,
                                         int gridT,
                                         int gridH,
                                         int gridW,
                                         int patchSize,
                                         int temporalPatchSize,
                                         int mergeSize,
                                         const std::vector<float> &imageMean,
                                         const std::vector<float> &imageStd,
                                         std::vector<float> &patches,
                                         std::vector<float> &posH,
                                         std::vector<float> &posW) {
        AssertInFastLLM(srcFrames > 0 && srcH > 0 && srcW > 0, "Qwen3.5 raw media shape is invalid.");
        AssertInFastLLM(gridH > 0 && gridW > 0 && gridT > 0, "Qwen3.5 grid_thw must be positive.");
        AssertInFastLLM(gridH % mergeSize == 0 && gridW % mergeSize == 0,
                        "Qwen3.5 vision grid must be divisible by spatial_merge_size.");
        AssertInFastLLM(gridT == (srcFrames + temporalPatchSize - 1) / temporalPatchSize,
                        "Qwen3.5 grid_thw temporal dimension does not match sampled frame count.");

        const int dstH = gridH * patchSize;
        const int dstW = gridW * patchSize;
        std::vector<float> resizedFrames((size_t) srcFrames * dstH * dstW * 3);
        for (int frame = 0; frame < srcFrames; frame++) {
            std::vector<float> resized;
            ResizeRgbFrameBicubic(rawData + (size_t) frame * srcH * srcW * 3, srcH, srcW, dstH, dstW, resized);
            memcpy(resizedFrames.data() + (size_t) frame * dstH * dstW * 3,
                   resized.data(),
                   (size_t) dstH * dstW * 3 * sizeof(float));
        }

        const int patchDim = 3 * temporalPatchSize * patchSize * patchSize;
        const int patchCount = gridT * gridH * gridW;
        patches.clear();
        posH.clear();
        posW.clear();
        patches.reserve((size_t) patchCount * patchDim);
        posH.reserve((size_t) patchCount);
        posW.reserve((size_t) patchCount);

        const int blockH = gridH / mergeSize;
        const int blockW = gridW / mergeSize;
        for (int t = 0; t < gridT; t++) {
            for (int bh = 0; bh < blockH; bh++) {
                for (int bw = 0; bw < blockW; bw++) {
                    for (int mh = 0; mh < mergeSize; mh++) {
                        for (int mw = 0; mw < mergeSize; mw++) {
                            const int patchHIndex = bh * mergeSize + mh;
                            const int patchWIndex = bw * mergeSize + mw;
                            posH.push_back((float) patchHIndex);
                            posW.push_back((float) patchWIndex);
                            for (int c = 0; c < 3; c++) {
                                for (int dt = 0; dt < temporalPatchSize; dt++) {
                                    const int srcFrame = std::min(t * temporalPatchSize + dt, srcFrames - 1);
                                    const float *framePtr = resizedFrames.data() + (size_t) srcFrame * dstH * dstW * 3;
                                    for (int ph = 0; ph < patchSize; ph++) {
                                        const int y = patchHIndex * patchSize + ph;
                                        for (int pw = 0; pw < patchSize; pw++) {
                                            const int x = patchWIndex * patchSize + pw;
                                            float pixel = framePtr[((size_t) y * dstW + x) * 3 + c];
                                            patches.push_back((pixel / 255.0f - imageMean[c]) / imageStd[c]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void BuildQwen35InterpolatedPosEmb(const float *embedWeight,
                                              int hiddenSize,
                                              int numGridPerSide,
                                              int gridT,
                                              int gridH,
                                              int gridW,
                                              int mergeSize,
                                              std::vector<float> &output) {
        AssertInFastLLM(gridH % mergeSize == 0 && gridW % mergeSize == 0,
                        "Qwen3.5 position interpolation requires merged grid alignment.");
        std::vector<float> hIdxs(gridH, 0.0f), wIdxs(gridW, 0.0f);
        for (int i = 0; i < gridH; i++) {
            hIdxs[i] = (gridH > 1) ? (float) i * (float) (numGridPerSide - 1) / (float) (gridH - 1) : 0.0f;
        }
        for (int i = 0; i < gridW; i++) {
            wIdxs[i] = (gridW > 1) ? (float) i * (float) (numGridPerSide - 1) / (float) (gridW - 1) : 0.0f;
        }

        std::vector<float> spatialEmbeds((size_t) gridH * gridW * hiddenSize, 0.0f);
        for (int h = 0; h < gridH; h++) {
            int hf = (int) floorf(hIdxs[h]);
            int hc = std::min(hf + 1, numGridPerSide - 1);
            float dh = hIdxs[h] - hf;
            for (int w = 0; w < gridW; w++) {
                int wf = (int) floorf(wIdxs[w]);
                int wc = std::min(wf + 1, numGridPerSide - 1);
                float dw = wIdxs[w] - wf;
                float w11 = dh * dw;
                float w10 = dh - w11;
                float w01 = dw - w11;
                float w00 = 1.0f - dh - w01;
                const float *e00 = embedWeight + ((size_t) hf * numGridPerSide + wf) * hiddenSize;
                const float *e01 = embedWeight + ((size_t) hf * numGridPerSide + wc) * hiddenSize;
                const float *e10 = embedWeight + ((size_t) hc * numGridPerSide + wf) * hiddenSize;
                const float *e11 = embedWeight + ((size_t) hc * numGridPerSide + wc) * hiddenSize;
                float *dst = spatialEmbeds.data() + ((size_t) h * gridW + w) * hiddenSize;
                for (int d = 0; d < hiddenSize; d++) {
                    dst[d] = e00[d] * w00 + e01[d] * w01 + e10[d] * w10 + e11[d] * w11;
                }
            }
        }

        output.clear();
        output.reserve((size_t) gridT * gridH * gridW * hiddenSize);
        const int blockH = gridH / mergeSize;
        const int blockW = gridW / mergeSize;
        for (int t = 0; t < gridT; t++) {
            for (int bh = 0; bh < blockH; bh++) {
                for (int bw = 0; bw < blockW; bw++) {
                    for (int mh = 0; mh < mergeSize; mh++) {
                        for (int mw = 0; mw < mergeSize; mw++) {
                            const int row = bh * mergeSize + mh;
                            const int col = bw * mergeSize + mw;
                            const float *src = spatialEmbeds.data() + ((size_t) row * gridW + col) * hiddenSize;
                            output.insert(output.end(), src, src + hiddenSize);
                        }
                    }
                }
            }
        }
    }

    int Qwen3_5Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
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
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    Qwen3_5Model::Qwen3_5Model() {
        this->model_struct = "qwen3_5";
        this->model_type = "qwen3_5";
        this->use_new_engine = true;
        this->num_experts = 0;
        this->num_experts_per_tok = 0;
        this->norm_topk_prob = true;

        weight.embeddingNames.insert(language_prefix + "embed_tokens.weight");
        weight.embeddingNames.insert(visual_prefix + "pos_embed.weight");
        weight.linearNames = {
            "lm_head.weight",
            language_prefix + "layers.*.mlp.down_proj.weight", language_prefix + "layers.*.mlp.up_proj.weight",
            language_prefix + "layers.*.mlp.gate_proj.weight", language_prefix + "layers.*.mlp.gate_proj.weight",
            language_prefix + "layers.*.mlp.gateup_proj.weight",
            language_prefix + "layers.*.mlp.gate.weight",
            language_prefix + "layers.*.mlp.shared_expert_gate.weight",
            language_prefix + "layers.*.mlp.shared_expert.gate_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.up_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.down_proj.weight",
            language_prefix + "layers.*.mlp.shared_expert.gateup_proj.weight",
            language_prefix + "layers.*.mlp.experts.gate_up_proj",
            language_prefix + "layers.*.mlp.experts.down_proj",
            language_prefix + "layers.*.mlp.experts.*.gate_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.up_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.down_proj.weight",
            language_prefix + "layers.*.mlp.experts.*.gateup_proj.weight",
            language_prefix + "layers.*.self_attn.o_proj.weight", language_prefix + "layers.*.self_attn.q_proj.weight",
            language_prefix + "layers.*.self_attn.k_proj.weight",
            language_prefix + "layers.*.self_attn.v_proj.weight", language_prefix + "layers.*.self_attn.mergeqkv.weight",
            language_prefix + "layers.*.self_attn.W_pack.weight",
            language_prefix + "layers.*.linear_attn.in_proj_qkv.weight",
            language_prefix + "layers.*.linear_attn.in_proj_z.weight",
            language_prefix + "layers.*.linear_attn.in_proj_b.weight",
            language_prefix + "layers.*.linear_attn.in_proj_a.weight",
            language_prefix + "layers.*.linear_attn.in_proj_qkvz.weight",
            language_prefix + "layers.*.linear_attn.in_proj_ba.weight",
            language_prefix + "layers.*.linear_attn.out_proj.weight",
            "mtp.fc.weight",
            "mtp.layers.*.mlp.down_proj.weight",
            "mtp.layers.*.mlp.up_proj.weight",
            "mtp.layers.*.mlp.gate_proj.weight",
            "mtp.layers.*.mlp.gateup_proj.weight",
            "mtp.layers.*.self_attn.o_proj.weight",
            "mtp.layers.*.self_attn.q_proj.weight",
            "mtp.layers.*.self_attn.k_proj.weight",
            "mtp.layers.*.self_attn.v_proj.weight",
            "mtp.layers.*.self_attn.mergeqkv.weight",
            visual_prefix + "patch_embed.proj.weight",
            visual_prefix + "blocks.*.attn.qkv.weight",
            visual_prefix + "blocks.*.attn.proj.weight",
            visual_prefix + "blocks.*.mlp.linear_fc1.weight",
            visual_prefix + "blocks.*.mlp.linear_fc2.weight",
            visual_prefix + "merger.linear_fc1.weight",
            visual_prefix + "merger.linear_fc2.weight",
            visual_prefix + "deepstack_merger_list.*.linear_fc1.weight",
            visual_prefix + "deepstack_merger_list.*.linear_fc2.weight"
        };
    }

    Qwen3_5Model::~Qwen3_5Model() {
        Qwen35EraseLinearPrefixSnapshots(this);
#ifdef USE_CUDA
        Qwen35EraseCudaGraphDecodeStates(this);
#endif
    }

    bool Qwen3_5Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetQwen35ThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    long long Qwen3_5Model::GetAutoWarmupCudaRuntimeReserveBytes(int deviceId, int batch) const {
#ifdef USE_CUDA
        if (batch <= 0) {
            return 0;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) ||
            devices.empty() || devices[0] != deviceId) {
            return 0;
        }

        auto it = this->weight.weight.find("lm_head.weight");
        if (it == this->weight.weight.end()) {
            it = this->weight.weight.find(language_prefix + "embed_tokens.weight");
        }
        if (it == this->weight.weight.end() || it->second.dims.empty() || it->second.dims[0] <= 0) {
            return 0;
        }

        long long vocabSize = it->second.dims[0];
        DataType computeType = ResolveQwen35ThreadTpComputeType(this->dataType);
        long long localLogitsBytes = computeType == DataType::FLOAT32 ? 0 :
            (long long)GetDataBytes(computeType, (size_t)batch, (size_t)vocabSize);
        long long reserveBytes =
            (long long)batch * vocabSize * (long long)sizeof(float) +
            localLogitsBytes +
            (long long)batch * (long long)sizeof(int) +
            Qwen35CudaRuntimeScratchReserveBytes();

        int mtpDrafts = Qwen35MtpDraftsPerStep();
        if (!Qwen35MtpDisabledByEnv() && mtpDrafts > 0 && HasMtpWeights() &&
            devices.size() == 1) {
            long long linearStateBytes = 0;
            for (int layer = 0; layer < block_cnt; layer++) {
                std::string prefix = language_prefix + "layers." +
                                     std::to_string(layer) + ".";
                bool isLinearAttentionLayer =
                    this->weight.weight.find(prefix + "linear_attn.out_proj.weight") !=
                        this->weight.weight.end() ||
                    this->weight.weight.find(prefix + "linear_attn.in_proj_qkvz.weight") !=
                        this->weight.weight.end() ||
                    this->weight.weight.find(prefix + "linear_attn.in_proj_qkvzba.weight") !=
                        this->weight.weight.end();
                if (!isLinearAttentionLayer) {
                    continue;
                }
                long long localKd = (long long)num_k_heads * head_k_dim;
                long long localVd = (long long)num_v_heads * head_v_dim;
                long long keyElements = (localKd * 2 + localVd) * 4;
                long long valueElements =
                    (long long)num_v_heads * head_k_dim * head_v_dim;
                linearStateBytes += (long long)GetDataBytes(
                    computeType, 1, (size_t)(keyElements + valueElements));
            }
            // MTP validation keeps one temporary linear cache plus one
            // snapshot for each partial draft position.
            reserveBytes += (long long)batch * (mtpDrafts + 1) * linearStateBytes;
        }
        return reserveBytes;
#else
        return 0;
#endif
    }

    void Qwen3_5Model::WarmupCudaRuntimeBuffers(int batch) {
#ifdef USE_CUDA
        if (batch <= 0) {
            return;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) || devices.empty()) {
            return;
        }

        auto it = this->weight.weight.find("lm_head.weight");
        if (it == this->weight.weight.end()) {
            it = this->weight.weight.find(language_prefix + "embed_tokens.weight");
        }
        if (it == this->weight.weight.end() || it->second.dims.empty() || it->second.dims[0] <= 0) {
            return;
        }

        int rootDevice = devices[0];
        int vocabSize = it->second.dims[0];
        FastllmCudaSetDevice(rootDevice);

        size_t logitsBytes = (size_t)batch * (size_t)vocabSize * sizeof(float);
        FastllmCudaMallocBigBuffer(logitsBytes);
        DataType computeType = ResolveQwen35ThreadTpComputeType(this->dataType);
        if (computeType != DataType::FLOAT32) {
            size_t localLogitsBytes = GetDataBytes(computeType, (size_t)batch, (size_t)vocabSize);
            FastllmCudaMallocBigBuffer(localLogitsBytes);
        }
        Qwen35WarmupCudaRuntimeScratchBuffers();

        Data cudaOutput(DataType::INT32);
        Qwen3CudaPrepareLocalOutput(cudaOutput, rootDevice);
        cudaOutput.Resize({batch});
        cudaOutput.Allocate();
        cudaOutput.FreeSpace();

        long long reserveBytes = GetAutoWarmupCudaRuntimeReserveBytes(rootDevice, batch);
        printf("[Fastllm] AutoWarmup Qwen3.5 CUDA sampling buffers: batch %d, vocab %d, reserve %.2f MB on GPU %d.\n",
               batch, vocabSize, reserveBytes / 1e6, rootDevice);
#endif
    }

    void Qwen3_5Model::OnAutoWarmupFinished() {
#ifdef USE_CUDA
        if (!Qwen35MtpDisabledByEnv() && Qwen35MtpWarmupEnabled() && HasMtpWeights()) {
            std::vector<int> devices;
            std::map<int, int> ratios;
            if (GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) &&
                !devices.empty()) {
                int device = devices[0];
                PrepareMtpWeightsForDevice(device, false);
                printf("[Qwen3.5 MTP] warmup: device=cuda:%d.\n", device);
                fflush(stdout);
            }
        }
        if (GetFastllmEnv().cudaGraph) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            ClearAllPagedCacheManagers();
            FastllmCudaClearBigBuffer();
            PreCaptureCudaGraphAfterWarmup();
        }
        Qwen35ReleaseThreadLocalCudaSamplingBuffers();
#endif
    }

    PagedCacheManager* Qwen3_5Model::GetPagedKVCacheManager(int layerIndex, bool isKey) const {
        if (layerIndex >= 0 && Qwen35LayerIsLinearAttention(this, layerIndex)) {
            return nullptr;
        }
        if (layerIndex >= 0 && this->threadTpPagedCacheBase >= 0) {
            PagedCacheManager *manager = GetPagedCacheManager(
                (this->threadTpPagedCacheBase + layerIndex) * 2 + (isKey ? 0 : 1));
            if (manager != nullptr) {
                return manager;
            }
        }
        return basellm::GetPagedKVCacheManager(layerIndex, isKey);
    }

    std::vector<std::pair<int, PagedCacheManager*> > Qwen3_5Model::GetPagedKVCacheManagers(int layerIndex, bool isKey) const {
        if (layerIndex >= 0 && Qwen35LayerIsLinearAttention(this, layerIndex)) {
            return {};
        }
        if (layerIndex >= 0 && this->threadTpPagedCacheBase >= 0) {
            std::vector<std::pair<int, PagedCacheManager*> > ret;
            int ranks = this->threadTpPreparedDevices.empty() ? 1 : (int)this->threadTpPreparedDevices.size();
            for (int r = 0; r < ranks; r++) {
                PagedCacheManager *manager = GetPagedCacheManager(
                    (this->threadTpPagedCacheBase + r * this->block_cnt + layerIndex) * 2 + (isKey ? 0 : 1));
                if (manager == nullptr) {
                    ret.clear();
                    break;
                }
                int device = r < (int)this->threadTpPreparedDevices.size() ? this->threadTpPreparedDevices[r] : -1;
                if (device < 0) {
                    Data *managerData = (Data*)manager;
                    if (!managerData->dataDeviceIds.empty()) {
                        device = managerData->dataDeviceIds[0];
                    }
                }
                ret.push_back(std::make_pair(device, manager));
            }
            if (!ret.empty()) {
                return ret;
            }
            if (!this->threadTpPreparedDevices.empty()) {
                return ret;
            }
        }
        return basellm::GetPagedKVCacheManagers(layerIndex, isKey);
    }

    bool Qwen3_5Model::RequiresMtpPrefixSnapshot(const ResponseContext *context) const {
        if (context == nullptr) {
            return false;
        }
        return !Qwen35MtpDisabledByEnv() &&
               HasMtpWeights() &&
               context->generationConfig.IsSimpleGreedy() &&
               !context->generationConfig.output_logits;
    }

    bool Qwen3_5Model::TryRecordPagedPrefixCacheExtra(ResponseContext *context) {
        if (context == nullptr || !Qwen35LinearPrefixCacheEnabled() ||
            !Qwen35HasLinearAttentionLayers(this, this->block_cnt)) {
            return false;
        }
        int pageLen = fastllm::GetPageLen();
        int currentLen = Qwen35CurrentTokenGrowingCacheLen(this, this->block_cnt, context->pastKeyValues);
        if (currentLen <= 0 || currentLen > (int)context->allTokens.size() ||
            currentLen % pageLen != 0) {
            return false;
        }
        int lastSnapshotLen = context->intParams["qwen35_linear_prefix_last_len"];
        int snapshotCount = context->intParams["qwen35_linear_prefix_count"];
        if (currentLen <= lastSnapshotLen) {
            return false;
        }
        int interval = Qwen35LinearPrefixSnapshotIntervalTokens();
        if (snapshotCount > 0 && currentLen % interval != 0) {
            return false;
        }
        int requestId = context->intParams["qwen35_linear_prefix_request_id"];
        if (requestId <= 0) {
            requestId = ++Qwen35LinearPrefixSnapshotRequestCounter();
            if (requestId <= 0) {
                requestId = 1;
                Qwen35LinearPrefixSnapshotRequestCounter().store(1);
            }
            context->intParams["qwen35_linear_prefix_request_id"] = requestId;
        }

        std::unique_ptr<Qwen35LinearPrefixSnapshot> snapshot(new Qwen35LinearPrefixSnapshot());
        snapshot->cachedLen = currentLen;
        snapshot->requestId = requestId;
        snapshot->tokens.assign(context->allTokens.begin(), context->allTokens.begin() + currentLen);
        snapshot->layers.resize(this->block_cnt);
        for (int i = 0; i < this->block_cnt; i++) {
            if (!Qwen35LayerIsLinearAttention(this, i)) {
                continue;
            }
            snapshot->layers[i].linear = true;
            if (i >= (int)context->pastKeyValues.size() ||
                !Qwen35SnapshotCopyCache(context->pastKeyValues[i].first, snapshot->layers[i].first) ||
                !Qwen35SnapshotCopyCache(context->pastKeyValues[i].second, snapshot->layers[i].second)) {
                return false;
            }
        }

        bool requireMtp = RequiresMtpPrefixSnapshot(context);
        if (requireMtp) {
#ifdef USE_CUDA
            std::lock_guard<std::mutex> guard(mtpCacheMutex);
            auto mtpIt = mtpCaches.find(context);
            if (mtpIt == mtpCaches.end() ||
                mtpIt->second.tokens != currentLen ||
                mtpIt->second.key.dims.size() < 2 ||
                mtpIt->second.value.dims.size() < 2 ||
                mtpIt->second.key.dims[1] != currentLen ||
                mtpIt->second.value.dims[1] != currentLen ||
                !Qwen35SnapshotCopyTensor(mtpIt->second.key, snapshot->mtpKey) ||
                !Qwen35SnapshotCopyTensor(mtpIt->second.value, snapshot->mtpValue)) {
                return false;
            }
            snapshot->mtpValid = true;
            snapshot->mtpTokens = currentLen;
#else
            return false;
#endif
        }

        {
            std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
            auto &items = Qwen35LinearPrefixSnapshots()[this];
            for (auto it = items.begin(); it != items.end(); ) {
                Qwen35LinearPrefixSnapshot *old = it->get();
                if (old != nullptr && old->cachedLen == snapshot->cachedLen &&
                    old->tokens == snapshot->tokens) {
                    // MTP snapshots also contain everything needed by non-MTP requests.
                    if (old->mtpValid && !snapshot->mtpValid) {
                        snapshot = std::move(*it);
                    }
                    it = items.erase(it);
                } else {
                    ++it;
                }
            }
            snapshot->requestId = requestId;
            snapshot->timestamp = ++Qwen35LinearPrefixSnapshotTimestamp();
            items.push_back(std::move(snapshot));
            int maxPerRequest = Qwen35LinearPrefixSnapshotMaxPerRequest();
            int requestRecords = 0;
            for (auto &item : items) {
                if (item != nullptr && item->requestId == requestId) {
                    requestRecords++;
                }
            }
            while (requestRecords > maxPerRequest) {
                auto oldest = items.end();
                for (auto it = items.begin(); it != items.end(); ++it) {
                    Qwen35LinearPrefixSnapshot *old = it->get();
                    if (old == nullptr || old->requestId != requestId) {
                        continue;
                    }
                    if (oldest == items.end() ||
                        old->timestamp < (*oldest)->timestamp) {
                        oldest = it;
                    }
                }
                if (oldest == items.end()) {
                    break;
                }
                items.erase(oldest);
                requestRecords--;
            }
            int maxRecords = Qwen35LinearPrefixSnapshotMaxRecords();
            while ((int)items.size() > maxRecords) {
                items.erase(items.begin());
            }
        }
        context->intParams["qwen35_linear_prefix_last_len"] = currentLen;
        context->intParams["qwen35_linear_prefix_count"] = snapshotCount + 1;
        return true;
    }

    int Qwen3_5Model::QueryPagedPrefixCacheExtra(ResponseContext *context, int maxCachedLen) const {
        if (context == nullptr || maxCachedLen <= 0 ||
            !Qwen35HasLinearAttentionLayers(this, this->block_cnt)) {
            return maxCachedLen;
        }
        if (!Qwen35LinearPrefixCacheEnabled()) {
            return 0;
        }
        bool requireMtp = RequiresMtpPrefixSnapshot(context);
        std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
        const Qwen35LinearPrefixSnapshot *snapshot =
            Qwen35FindLinearPrefixSnapshotLocked(
                this, context->currentTokens, maxCachedLen, -1, requireMtp);
        return snapshot == nullptr ? 0 : snapshot->cachedLen;
    }

    bool Qwen3_5Model::RestorePagedPrefixCacheExtra(ResponseContext *context, int cachedLen) const {
        if (context == nullptr || cachedLen <= 0 ||
            !Qwen35HasLinearAttentionLayers(this, this->block_cnt)) {
            return true;
        }
        bool requireMtp = RequiresMtpPrefixSnapshot(context);
        const Qwen35LinearPrefixSnapshot *snapshot = nullptr;
        {
            std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
            snapshot = Qwen35FindLinearPrefixSnapshotLocked(
                this, context->currentTokens, cachedLen, cachedLen, requireMtp);
            if (snapshot == nullptr || (int)snapshot->layers.size() < this->block_cnt) {
                return false;
            }
            for (int i = 0; i < this->block_cnt; i++) {
                if (!Qwen35LayerIsLinearAttention(this, i)) {
                    continue;
                }
                if (!snapshot->layers[i].linear ||
                    !snapshot->layers[i].first.valid ||
                    !snapshot->layers[i].second.valid) {
                    return false;
                }
            }
            for (int i = 0; i < this->block_cnt; i++) {
                if (!Qwen35LayerIsLinearAttention(this, i)) {
                    continue;
                }
                if (i >= (int)context->pastKeyValues.size() ||
                    !Qwen35RestoreSnapshotCache(snapshot->layers[i].first,
                                                context->pastKeyValues[i].first) ||
                    !Qwen35RestoreSnapshotCache(snapshot->layers[i].second,
                                                context->pastKeyValues[i].second)) {
                    return false;
                }
            }
            if (requireMtp) {
#ifdef USE_CUDA
                std::vector<int> devices;
                std::map<int, int> ratios;
                if (!snapshot->mtpValid || snapshot->mtpTokens != cachedLen ||
                    !GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) ||
                    devices.empty()) {
                    return false;
                }
                std::lock_guard<std::mutex> mtpGuard(mtpCacheMutex);
                mtpCaches.erase(context);
                MtpKvCache &mtpCache = mtpCaches[context];
                if (!Qwen35RestoreMtpSnapshotTensor(
                        snapshot->mtpKey, mtpCache.key, devices[0]) ||
                    !Qwen35RestoreMtpSnapshotTensor(
                        snapshot->mtpValue, mtpCache.value, devices[0])) {
                    mtpCaches.erase(context);
                    return false;
                }
                mtpCache.tokens = cachedLen;
#else
                return false;
#endif
            }
        }
        return true;
    }

    int Qwen3_5Model::GetChunkedPrefillSize() {
        int base = basellm::GetChunkedPrefillSize();
        if (!Qwen35LinearPrefixCacheEnabled() ||
            !Qwen35HasLinearAttentionLayers(this, this->block_cnt)) {
            return base;
        }
        int interval = Qwen35LinearPrefixSnapshotIntervalTokens();
        int pageLen = fastllm::GetPageLen();
        interval = std::max(pageLen, interval / pageLen * pageLen);
        return std::max(pageLen, std::min(base, interval));
    }

    void Qwen3_5Model::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(mtpCacheMutex);
        mtpCaches.erase(context);
    }

    void Qwen3_5Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(mtpCacheMutex);
        mtpCaches.erase(context);
    }

    bool Qwen3_5Model::UseModelSpecificScheduler() const {
#ifndef USE_CUDA
        return false;
#else
        std::vector<int> devices;
        std::map<int, int> ratios;
        return !Qwen35MtpDisabledByEnv() &&
               HasMtpWeights() &&
               GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) &&
               !devices.empty();
#endif
    }

    void Qwen3_5Model::RunModelSpecificScheduler() {
        if (UseModelSpecificScheduler()) {
            Qwen35MTPLoop();
        } else {
            NewMainLoop();
        }
    }

    void Qwen3_5Model::PreAllocateLinearSlotPoolsForCudaGraph(
            const std::vector<int> &devices,
            const std::map<int, int> &ratios,
            int slotCapacity) {
#ifdef USE_CUDA
        if (devices.empty() || slotCapacity <= 0 ||
            num_k_heads <= 0 || num_v_heads <= 0 ||
            head_k_dim <= 0 || head_v_dim <= 0 ||
            num_v_heads % num_k_heads != 0) {
            return;
        }

        bool tensorParallel = devices.size() > 1;

        std::lock_guard<std::mutex> guard(Qwen35LinearSlotPoolsMutex());
        for (int gpuId : devices) {
            for (int layer = 0; layer < block_cnt; layer++) {
                std::string prefix = language_prefix + "layers." + std::to_string(layer) + ".";
                bool isAttentionLayer =
                    weight.weight.find(prefix + "self_attn.o_proj.weight") != weight.weight.end();
                if (isAttentionLayer) {
                    continue;
                }
                int localKeyHeads = num_k_heads;
                int localValueHeads = num_v_heads;
                if (tensorParallel) {
                    DivisionScheme keyScheme = BuildQwen35LinearKeyHeadScheme(
                        devices, ratios, num_k_heads);
                    BalanceMultiCudaDivisionSchemeByLayer(
                        prefix + "linear_attn.in_proj_qkvz.weight", devices, keyScheme);
                    DivisionScheme valueScheme = BuildQwen35LinearValueHeadScheme(
                        keyScheme, num_v_heads / num_k_heads);
                    localKeyHeads = Qwen35LocalHeads(keyScheme, gpuId);
                    localValueHeads = Qwen35LocalHeads(valueScheme, gpuId);
                }
                if (localKeyHeads <= 0 || localValueHeads <= 0) {
                    continue;
                }

                int convDim = localKeyHeads * head_k_dim * 2 + localValueHeads * head_v_dim;
                Qwen35GetLinearSlotPoolLocked(
                    this, gpuId, layer, QWEN35_LINEAR_SLOT_CONV,
                    {slotCapacity, 1, convDim, 4});
                Qwen35GetLinearSlotPoolLocked(
                    this, gpuId, layer, QWEN35_LINEAR_SLOT_RECURRENT,
                    {slotCapacity, localValueHeads, head_v_dim, head_k_dim});
            }
        }
#else
        (void)devices;
        (void)ratios;
        (void)slotCapacity;
#endif
    }

    void Qwen3_5Model::PreCaptureCudaGraphAfterWarmup() {
#ifdef USE_CUDA
        if (!GetFastllmEnv().cudaGraph || autoWarmupRunning.load() ||
            GetKVCacheInCPU()) {
            return;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        if ((num_experts > 0 && !Qwen35ModelMoeLayersAllowCudaOnly(this)) ||
            !GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) || devices.empty()) {
            return;
        }

        int maxWarmupBatch = Qwen35PreCaptureMaxBatch(this);
        int linearSlotCapacity = Qwen35LinearSlotCapacity(this, maxWarmupBatch);
        PreAllocateLinearSlotPoolsForCudaGraph(devices, ratios, linearSlotCapacity);

        struct CudaGraphPreCaptureScope {
            std::atomic<bool> &flag;
            explicit CudaGraphPreCaptureScope(std::atomic<bool> &flag) : flag(flag) {
                flag.store(true, std::memory_order_release);
            }
            ~CudaGraphPreCaptureScope() {
                flag.store(false, std::memory_order_release);
            }
        } preCaptureScope(cudaGraphPreCaptureRunning);

        auto printProgress = [](int done, int total, int batch) {
            const int barWidth = 32;
            int filled = total > 0 ? done * barWidth / total : barWidth;
            printf("\r[Fastllm] Qwen3.5 CUDA graph warmup capture [");
            for (int i = 0; i < barWidth; i++) {
                putchar(i < filled ? '#' : '-');
            }
            printf("] %d/%d batch=%d%s", done, total, batch,
                   done >= total ? " done" : "     ");
            if (done >= total) {
                printf("\n");
            }
            fflush(stdout);
        };
        printProgress(0, maxWarmupBatch, 0);

        for (int batch = 1; batch <= maxWarmupBatch; batch++) {
            std::vector<Data*> attentionMasks(batch, nullptr);
            std::vector<GenerationConfig> generationConfigs(batch);
            LastTokensManager lastTokens;

            std::vector<std::pair<Data, Data> > pastKeyValuesStorage;
            std::vector<std::pair<Data*, Data*> > pastKeyValues;
            pastKeyValuesStorage.reserve(batch * block_cnt);
            pastKeyValues.reserve(batch * block_cnt);
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    bool isLinearLayer = Qwen35LayerIsLinearAttention(this, i);
                    DataType cacheType = isLinearLayer ?
                        Qwen35LinearAttentionCacheDataType(this->dataType) : this->kvCacheDataType;
                    pastKeyValuesStorage.push_back(std::make_pair(Data(cacheType),
                                                                  Data(cacheType)));
                    pastKeyValuesStorage.back().first.SetKVCache();
                    pastKeyValuesStorage.back().second.SetKVCache();
                    if (isLinearLayer) {
                        Qwen35PrepareLinearAttentionCache(pastKeyValuesStorage.back().first, cacheType);
                        Qwen35PrepareLinearAttentionCache(pastKeyValuesStorage.back().second, cacheType);
                    }
                    pastKeyValues.push_back(std::make_pair(&pastKeyValuesStorage.back().first,
                                                           &pastKeyValuesStorage.back().second));
                }
            }

            const int cudaGraphPreCapturePrefillTokens = 8;
            std::vector<float> prefillInputIdsHost(batch * cudaGraphPreCapturePrefillTokens, 1.0f);
            Data prefillInputIds(DataType::FLOAT32,
                                 {1, batch * cudaGraphPreCapturePrefillTokens},
                                 prefillInputIdsHost);
            std::vector<int> prefillSeqLens(batch, cudaGraphPreCapturePrefillTokens);
            std::vector<Data> prefillPositionIdsStorage;
            std::vector<Data*> prefillPositionIds;
            prefillPositionIdsStorage.reserve(batch);
            prefillPositionIds.reserve(batch);
            std::vector<float> prefillPositions(cudaGraphPreCapturePrefillTokens);
            for (int i = 0; i < cudaGraphPreCapturePrefillTokens; i++) {
                prefillPositions[i] = (float)i;
            }
            for (int b = 0; b < batch; b++) {
                prefillPositionIdsStorage.push_back(
                    Data(DataType::FLOAT32, {1, cudaGraphPreCapturePrefillTokens}, prefillPositions));
                prefillPositionIds.push_back(&prefillPositionIdsStorage.back());
            }
            ForwardGPU(batch, prefillInputIds, attentionMasks, prefillPositionIds,
                       prefillSeqLens, pastKeyValues, generationConfigs, lastTokens, nullptr);

            std::vector<float> inputIdsHost(batch, 1.0f);
            Data inputIds(DataType::FLOAT32, {1, batch}, inputIdsHost);
            std::vector<int> seqLens(batch, 1);
            const int cudaGraphPreCaptureDecodeSteps = 2;
            for (int step = 0; step < cudaGraphPreCaptureDecodeSteps; step++) {
                std::vector<Data> positionIdsStorage;
                std::vector<Data*> positionIds;
                positionIdsStorage.reserve(batch);
                positionIds.reserve(batch);
                for (int b = 0; b < batch; b++) {
                    positionIdsStorage.push_back(
                        Data(DataType::FLOAT32, {1, 1},
                             {(float)(cudaGraphPreCapturePrefillTokens + step)}));
                    positionIds.push_back(&positionIdsStorage.back());
                }
                ForwardGPU(batch, inputIds, attentionMasks, positionIds, seqLens,
                           pastKeyValues, generationConfigs, lastTokens, nullptr);
            }
            printProgress(batch, maxWarmupBatch, batch);
        }
#endif
    }

    Data &Qwen3_5Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end() && !it->second.dims.empty()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    bool Qwen3_5Model::ForwardSingleGPUDecodeGraph(
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
        return false;
#else
        (void)ratios;
        const int maxCudaGraphDecodeBatch = Qwen35MaxCudaGraphDecodeBatch();
        if (!Qwen35CudaGraphEnabled() || batch <= 0 || batch > maxCudaGraphDecodeBatch ||
            !all1 || isPrefill || (int)seqLens.size() < batch ||
            (int)pastKeyValues.size() < batch * block_cnt || seqLens[0] != 1 ||
            positionIds.Count(0) != (uint64_t)batch) {
            return false;
        }
        for (int b = 0; b < batch; b++) {
            if (seqLens[b] != 1) {
                return false;
            }
        }
        if (precomputedHiddenStates == nullptr && inputIds.Count(0) != (uint64_t)batch) {
            return false;
        }
        if (num_experts > 0 && !Qwen35ModelMoeLayersAllowCudaOnly(this)) {
            return false;
        }

        std::vector<int> attentionLayers;
        std::vector<int> linearLayers;
        attentionLayers.reserve(block_cnt);
        linearLayers.reserve(block_cnt);
        for (int i = 0; i < block_cnt; i++) {
            std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
            bool isAttentionLayer =
                weight.weight.find(prefix + "self_attn.o_proj.weight") != weight.weight.end();
            if (isAttentionLayer) {
                attentionLayers.push_back(i);
            } else {
                linearLayers.push_back(i);
            }
        }
        if (attentionLayers.empty()) {
            return false;
        }

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (data.dims.empty()) {
                return &data;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Qwen3.5 ForwardSingleGPU graph missing local tensor: " + name + ".\n");
            return nullptr;
        };

        Data *localPositionIds = requireLocal((Data&)positionIds, "positionIds");
        Data *localInputIds = nullptr;
        if (precomputedHiddenStates == nullptr) {
            localInputIds = requireLocal((Data&)inputIds, "inputIds");
            if (localInputIds->dims.size() != 2 || localInputIds->Count(0) != (uint64_t)batch) {
                return false;
            }
        }
        if (localPositionIds->dims.empty() || localPositionIds->Count(0) != (uint64_t)batch) {
            return false;
        }

        Data *localPrecomputedHiddenStates = nullptr;
        if (precomputedHiddenStates != nullptr) {
            localPrecomputedHiddenStates = requireLocal(*precomputedHiddenStates, "precomputedHiddenStates");
            if (localPrecomputedHiddenStates->dataDevice != DataDevice::CUDA ||
                localPrecomputedHiddenStates->cudaData == nullptr ||
                localPrecomputedHiddenStates->dims.size() != 3 ||
                localPrecomputedHiddenStates->dims[0] != 1 ||
                localPrecomputedHiddenStates->dims[1] != batch) {
                return false;
            }
        }

        int firstAttentionLayer = attentionLayers[0];
        int currentTokens = 0;
        for (int layer : attentionLayers) {
            for (int b = 0; b < batch; b++) {
                Data *pastKey = pastKeyValues[b * block_cnt + layer].first;
                Data *pastValue = pastKeyValues[b * block_cnt + layer].second;
                if (pastKey == nullptr || pastValue == nullptr ||
                    pastKey->pagedKVCacheData == nullptr ||
                    pastValue->pagedKVCacheData == nullptr ||
                    pastKey->pageIndex.empty() || pastValue->pageIndex.empty() ||
                    pastKey->dataDevice != DataDevice::CUDA ||
                    pastValue->dataDevice != DataDevice::CUDA ||
                    pastKey->dataType == DataType::FP8_E4M3 ||
                    pastValue->dataType == DataType::FP8_E4M3 ||
                    pastKey->pageLen <= 0 || pastKey->pageLen != pastValue->pageLen ||
                    pastKey->pageIndex.size() != pastValue->pageIndex.size() ||
                    pastKey->lastPageLen != pastValue->lastPageLen) {
                    return false;
                }
                int layerTokens = ((int)pastKey->pageIndex.size() - 1) * pastKey->pageLen + pastKey->lastPageLen;
                currentTokens = std::max(currentTokens, layerTokens);
            }
        }
        if (rope_type == RoPEType::DYMAMIC_NTK && currentTokens + 1 >= max_positions) {
            return false;
        }
        if (currentTokens < 8) {
            return false;
        }

        std::vector<int> insertIndexHost(batch, -1);
        std::vector<int> insertPositionHost(batch, 0);
        std::vector<int> lastPageLensHost(batch, 0);
        std::vector<int> qSizesHost(batch + 1, 0);
        std::vector<int> pageSizesHost(batch + 1, 0);
        std::vector<int> graphPlanPageSizesHost(batch + 1, 0);
        std::vector<int> pageIndexHost;
        std::vector<const Data*> currentPastKeyHosts(batch, nullptr);
        std::vector<bool> needNewPageHost(batch, false);
        bool anyNeedNewPage = false;
        for (int b = 0; b < batch; b++) {
            Data *firstKeyBeforeUpdate = pastKeyValues[b * block_cnt + firstAttentionLayer].first;
            currentPastKeyHosts[b] = firstKeyBeforeUpdate;
            needNewPageHost[b] = firstKeyBeforeUpdate->pageIndex.empty() ||
                                 firstKeyBeforeUpdate->lastPageLen >= firstKeyBeforeUpdate->pageLen;
            anyNeedNewPage = anyNeedNewPage || needNewPageHost[b];
            if (!needNewPageHost[b]) {
                insertIndexHost[b] = firstKeyBeforeUpdate->pageIndex.back();
                insertPositionHost[b] = firstKeyBeforeUpdate->lastPageLen;
            }
        }

        PagedCacheManager *graphPagedManager =
            pastKeyValues[firstAttentionLayer].first->pagedKVCacheData;
        int graphMaxPagesPerRequest = graphPagedManager != nullptr ? graphPagedManager->maxPages : 0;
        if (graphMaxPagesPerRequest <= 0 && graphPagedManager != nullptr &&
            !graphPagedManager->dims.empty()) {
            graphMaxPagesPerRequest = graphPagedManager->dims[0];
        }
        if (graphMaxPagesPerRequest <= 0) {
            return false;
        }
        for (int b = 0; b < batch; b++) {
            Data *firstKey = pastKeyValues[b * block_cnt + firstAttentionLayer].first;
            int prospectivePages = (int)firstKey->pageIndex.size() +
                                   (needNewPageHost[b] ? 1 : 0);
            if (prospectivePages > graphMaxPagesPerRequest) {
                return false;
            }
        }
        int graphPlanPagesPerRequest = graphMaxPagesPerRequest;
        for (int b = 0; b < batch; b++) {
            graphPlanPageSizesHost[b + 1] =
                graphPlanPageSizesHost[b] + graphPlanPagesPerRequest;
        }
        int pageIndexCapacity = batch * graphPlanPagesPerRequest;

        for (int layer : linearLayers) {
            for (int b = 0; b < batch; b++) {
                Data *convCache = pastKeyValues[b * block_cnt + layer].first;
                Data *recurrentState = pastKeyValues[b * block_cnt + layer].second;
                if (convCache == nullptr || recurrentState == nullptr ||
                    convCache->dims.empty() || recurrentState->dims.empty() ||
                    convCache->dataDevice != DataDevice::CUDA ||
                    recurrentState->dataDevice != DataDevice::CUDA ||
                    convCache->cudaData == nullptr || recurrentState->cudaData == nullptr) {
                    return false;
                }
            }
        }

        Qwen35CudaGraphDecodeState &state = GetQwen35CudaGraphDecodeState(this, gpuId, batch);
        std::unique_lock<std::mutex> graphLock(state.mutex);
        if (state.disabled) {
            return false;
        }
        bool allowCapture = autoWarmupRunning.load(std::memory_order_acquire) ||
                            cudaGraphPreCaptureRunning.load(std::memory_order_acquire);
        if (!state.captured && !allowCapture) {
            return false;
        }

        FastllmCudaSetDevice(gpuId);
        state.buffers.batchPastKeys.resize(batch);
        state.buffers.batchPastValues.resize(batch);
        state.buffers.linearConvCaches.resize(batch);
        state.buffers.recurrentStates.resize(batch);
        bool useTransposedLinearState = true;
        if (useTransposedLinearState) {
            for (int layer : linearLayers) {
                for (int b = 0; b < batch; b++) {
                    if (!Qwen35EnsureCudaLinearAttnStateTransposed(
                            *pastKeyValues[b * block_cnt + layer].second)) {
                        return false;
                    }
                }
            }
        } else {
            for (int layer : linearLayers) {
                for (int b = 0; b < batch; b++) {
                    if (!Qwen35EnsureCudaLinearAttnStateKVLayout(
                            *pastKeyValues[b * block_cnt + layer].second)) {
                        return false;
                    }
                }
            }
        }
        std::vector<int> linearSlotIdsHost;
        int linearSlotCapacity = Qwen35LinearSlotCapacity(this, batch);
        if (!Qwen35PrepareLinearSlotCaches(this, gpuId, batch, block_cnt,
                                           linearLayers, pastKeyValues,
                                           linearSlotIdsHost)) {
            return false;
        }

        struct GraphPagedCacheMetaSnapshot {
            Data *cache = nullptr;
            size_t pageIndexSize = 0;
            int lastPageLen = 0;
            bool hasLogicalTokens = false;
            int logicalTokens = 0;
            bool hasExpansionTokens = false;
            int expansionTokens = 0;
            size_t stridesOffset = 0;
            size_t stridesSize = 0;
        };
        std::vector<GraphPagedCacheMetaSnapshot> graphPagedMetaSnapshots;
        std::vector<uint64_t> graphPagedOldStrides;
        const size_t graphPagedCacheCount = attentionLayers.size() * (size_t)batch * 2;
        graphPagedMetaSnapshots.reserve(graphPagedCacheCount);
        size_t graphPagedStrideCount = 0;
        for (int layer : attentionLayers) {
            for (int b = 0; b < batch; b++) {
                graphPagedStrideCount += pastKeyValues[b * block_cnt + layer].first->strides.size();
                graphPagedStrideCount += pastKeyValues[b * block_cnt + layer].second->strides.size();
            }
        }
        graphPagedOldStrides.reserve(graphPagedStrideCount);
        auto snapshotGraphPagedMeta = [&](Data *cache) {
            GraphPagedCacheMetaSnapshot snapshot;
            snapshot.cache = cache;
            snapshot.pageIndexSize = cache->pageIndex.size();
            snapshot.lastPageLen = cache->lastPageLen;
            snapshot.hasLogicalTokens = cache->dims.size() >= 2;
            if (snapshot.hasLogicalTokens) {
                snapshot.logicalTokens = cache->dims[1];
            }
            snapshot.hasExpansionTokens = cache->expansionDims.size() >= 2;
            if (snapshot.hasExpansionTokens) {
                snapshot.expansionTokens = cache->expansionDims[1];
            }
            snapshot.stridesOffset = graphPagedOldStrides.size();
            snapshot.stridesSize = cache->strides.size();
            graphPagedOldStrides.insert(graphPagedOldStrides.end(),
                                        cache->strides.begin(), cache->strides.end());
            graphPagedMetaSnapshots.push_back(snapshot);
        };
        for (int layer : attentionLayers) {
            for (int b = 0; b < batch; b++) {
                snapshotGraphPagedMeta(pastKeyValues[b * block_cnt + layer].first);
                snapshotGraphPagedMeta(pastKeyValues[b * block_cnt + layer].second);
            }
        }

        std::vector<std::pair<PagedCacheManager*, int> > graphAllocatedPages;
        graphAllocatedPages.reserve(graphPagedCacheCount);
        bool graphPagedRollbackArmed = true;
        auto rollbackGraphPagedMeta = [&]() noexcept {
            if (!graphPagedRollbackArmed) {
                return;
            }
            graphPagedRollbackArmed = false;
            for (const auto &snapshot : graphPagedMetaSnapshots) {
                Data *cache = snapshot.cache;
                cache->pageIndex.resize(snapshot.pageIndexSize);
                cache->lastPageLen = snapshot.lastPageLen;
                if (snapshot.hasLogicalTokens && cache->dims.size() >= 2) {
                    cache->dims[1] = snapshot.logicalTokens;
                }
                if (snapshot.hasExpansionTokens && cache->expansionDims.size() >= 2) {
                    cache->expansionDims[1] = snapshot.expansionTokens;
                }
                cache->strides.resize(snapshot.stridesSize);
                std::copy(graphPagedOldStrides.begin() + snapshot.stridesOffset,
                          graphPagedOldStrides.begin() + snapshot.stridesOffset + snapshot.stridesSize,
                          cache->strides.begin());
            }
            for (auto it = graphAllocatedPages.rbegin(); it != graphAllocatedPages.rend(); ++it) {
                try {
                    it->first->ReleasePageIndex(it->second);
                } catch (...) {
                    // Metadata is already restored. Conservatively lose a free
                    // page rather than terminate from a noexcept rollback path.
                }
            }
        };
        int graphPagedRollbackToken = 0;
        auto graphPagedRollbackDeleter = [&](int*) noexcept {
            rollbackGraphPagedMeta();
        };
        std::unique_ptr<int, decltype(graphPagedRollbackDeleter)> graphPagedRollbackGuard(
            &graphPagedRollbackToken, graphPagedRollbackDeleter);

        for (int layer : attentionLayers) {
            for (int b = 0; b < batch; b++) {
                Data *pastKey = pastKeyValues[b * block_cnt + layer].first;
                Data *pastValue = pastKeyValues[b * block_cnt + layer].second;
                bool layerNeedNewPage = pastKey->pageIndex.empty() || pastKey->lastPageLen >= pastKey->pageLen;
                AssertInFastLLM(layerNeedNewPage == needNewPageHost[b],
                                "Qwen3.5 CUDA graph requires aligned paged cache layout across attention layers.\n");
                if (needNewPageHost[b]) {
                    int keyPage = pastKey->pagedKVCacheData->GetUnusedPageIndex(true);
                    graphAllocatedPages.push_back({pastKey->pagedKVCacheData, keyPage});
                    int valuePage = pastValue->pagedKVCacheData->GetUnusedPageIndex(true);
                    graphAllocatedPages.push_back({pastValue->pagedKVCacheData, valuePage});
                    if (insertIndexHost[b] < 0) {
                        insertIndexHost[b] = keyPage;
                    }
                    AssertInFastLLM(keyPage == insertIndexHost[b] && valuePage == insertIndexHost[b],
                                    "Qwen3.5 CUDA graph requires aligned K/V page indices across attention layers.\n");
                    pastKey->pageIndex.push_back(keyPage);
                    pastValue->pageIndex.push_back(valuePage);
                    pastKey->lastPageLen = 1;
                    pastValue->lastPageLen = 1;
                } else {
                    AssertInFastLLM(pastKey->pageIndex.back() == insertIndexHost[b] &&
                                    pastValue->pageIndex.back() == insertIndexHost[b] &&
                                    pastKey->lastPageLen == insertPositionHost[b] &&
                                    pastValue->lastPageLen == insertPositionHost[b],
                                    "Qwen3.5 CUDA graph requires aligned paged cache positions across attention layers.\n");
                    pastKey->lastPageLen++;
                    pastValue->lastPageLen++;
                }
                Qwen35AdvancePagedCacheLogicalTokens(*pastKey, 1);
                Qwen35AdvancePagedCacheLogicalTokens(*pastValue, 1);
            }
        }

        for (int b = 0; b < batch; b++) {
            Data *firstKey = pastKeyValues[b * block_cnt + firstAttentionLayer].first;
            Data *firstValue = pastKeyValues[b * block_cnt + firstAttentionLayer].second;
            qSizesHost[b + 1] = qSizesHost[b] + 1;
            lastPageLensHost[b] = firstKey->lastPageLen;
            int requestPages = (int)firstKey->pageIndex.size();
            if (requestPages > graphMaxPagesPerRequest) {
                return false;
            }
            pageSizesHost[b + 1] = pageSizesHost[b] + requestPages;
            pageIndexHost.insert(pageIndexHost.end(), firstKey->pageIndex.begin(), firstKey->pageIndex.end());
            AssertInFastLLM(firstKey->pageIndex.size() == firstValue->pageIndex.size() &&
                            firstKey->lastPageLen == firstValue->lastPageLen,
                            "Qwen3.5 CUDA graph requires aligned K/V page metadata.\n");
            for (int layer : attentionLayers) {
                Data *pastKey = pastKeyValues[b * block_cnt + layer].first;
                Data *pastValue = pastKeyValues[b * block_cnt + layer].second;
                AssertInFastLLM(pastKey->pageIndex == firstKey->pageIndex &&
                                pastValue->pageIndex == firstKey->pageIndex &&
                                pastKey->lastPageLen == firstKey->lastPageLen &&
                                pastValue->lastPageLen == firstKey->lastPageLen,
                                "Qwen3.5 CUDA graph requires identical page metadata across attention layers.\n");
            }
        }
        Qwen35PrepareGraphCudaTensor(state.positionIds, *localPositionIds, gpuId);
        if (precomputedHiddenStates == nullptr) {
            Qwen35PrepareGraphCudaTensor(state.inputIds, *localInputIds, gpuId);
        } else {
            Qwen35PrepareGraphCudaTensor(state.buffers.hiddenStates,
                                         *localPrecomputedHiddenStates, gpuId);
        }

        std::ostringstream signature;
        signature << "gpu=" << gpuId
                  << ";tp=" << (tensorParallel ? 1 : 0)
                  << ";tpRank0=" << (firstTensorParallelRank ? 1 : 0)
                  << ";batch=" << batch
                  << ";preHidden=" << (precomputedHiddenStates != nullptr ? 1 : 0)
                  << ";posDims=";
        for (int dim : state.positionIds.dims) {
            signature << dim << ",";
        }
        if (precomputedHiddenStates == nullptr) {
            signature << ";inputDims=";
            for (int dim : state.inputIds.dims) {
                signature << dim << ",";
            }
            signature << ";inputType=" << (int)state.inputIds.dataType;
        } else {
            signature << ";hiddenDims=";
            for (int dim : state.buffers.hiddenStates.dims) {
                signature << dim << ",";
            }
            signature << ";hiddenType=" << (int)state.buffers.hiddenStates.dataType;
        }
        signature << ";posType=" << (int)state.positionIds.dataType
                  << ";pageSizes=";
        for (int pageSize : graphPlanPageSizesHost) {
            signature << pageSize << ",";
        }
        signature << ";pages=" << pageIndexCapacity
                  << ";lmLocal=" << requireLocal(weight["lm_head.weight"], "lm_head.weight")->dims[0];
        for (int layer : attentionLayers) {
            signature << ";kCache" << layer << "=" << pastKeyValues[layer].first->pagedKVCacheData->cudaData
                      << ";vCache" << layer << "=" << pastKeyValues[layer].second->pagedKVCacheData->cudaData;
        }
        for (int layer : linearLayers) {
            PagedCacheManager *convPool = Qwen35FindLinearSlotPool(
                this, gpuId, layer, QWEN35_LINEAR_SLOT_CONV,
                linearSlotCapacity);
            PagedCacheManager *statePool = Qwen35FindLinearSlotPool(
                this, gpuId, layer, QWEN35_LINEAR_SLOT_RECURRENT,
                linearSlotCapacity);
            signature << ";linConvPool" << layer << "=" << (convPool == nullptr ? nullptr : convPool->cudaData)
                      << ";linStatePool" << layer << "=" << (statePool == nullptr ? nullptr : statePool->cudaData);
        }
        std::string newSignature = signature.str();
        bool signatureChanged = state.signature != newSignature;
        if (signatureChanged) {
            Qwen35DestroyCudaGraph(state);
            state.signature = newSignature;
        }

        if (!linearLayers.empty() &&
            (state.linearSlotIds.cudaData == nullptr ||
             state.lastLinearSlotIdsHost != linearSlotIdsHost ||
             signatureChanged)) {
            Qwen35PrepareGraphIntTensor(state.linearSlotIds, gpuId, linearSlotIdsHost);
            state.lastLinearSlotIdsHost = linearSlotIdsHost;
        }

        bool graphMetaMissing =
            state.buffers.insertIndexs.cudaData == nullptr ||
            state.buffers.insertPositions.cudaData == nullptr ||
            state.buffers.qSizes.cudaData == nullptr ||
            state.buffers.pageSizes.cudaData == nullptr ||
            state.buffers.pageIndexs.cudaData == nullptr ||
            state.buffers.lastPageLens.cudaData == nullptr;
        bool metadataChanged =
            state.lastInsertIndexHost != insertIndexHost ||
            state.lastPageSizesHost != pageSizesHost ||
            state.lastPageIndexHost != pageIndexHost ||
            state.lastDecodePageLensHost != insertPositionHost ||
            state.lastPastKeyHosts != currentPastKeyHosts;
        bool needFullMetaCopy = graphMetaMissing || signatureChanged || anyNeedNewPage || metadataChanged;
        if (needFullMetaCopy) {
            AssertInFastLLM((int)pageIndexHost.size() <= pageIndexCapacity,
                            "Qwen3.5 CUDA graph page metadata exceeds fixed graph capacity.\n");
            std::vector<int> paddedPageIndexHost = pageIndexHost;
            paddedPageIndexHost.resize(pageIndexCapacity,
                                       paddedPageIndexHost.empty() ? 0 : paddedPageIndexHost.back());

            Qwen35PrepareGraphIntTensor(state.buffers.insertIndexs, gpuId, insertIndexHost);
            Qwen35PrepareGraphIntTensor(state.buffers.insertPositions, gpuId, insertPositionHost);
            Qwen35PrepareGraphIntTensor(state.buffers.qSizes, gpuId, qSizesHost);
            Qwen35PrepareGraphIntTensor(state.buffers.pageSizes, gpuId, pageSizesHost);
            state.buffers.pageSizes.cpuIntDatas = graphPlanPageSizesHost;
            Qwen35PrepareGraphIntTensor(state.buffers.pageIndexs, gpuId, paddedPageIndexHost);
            Qwen35PrepareGraphIntTensor(state.buffers.lastPageLens, gpuId, lastPageLensHost);
            state.lastInsertIndexHost = insertIndexHost;
            state.lastPageSizesHost = pageSizesHost;
            state.lastPageIndexHost = pageIndexHost;
            state.lastDecodePageLensHost = lastPageLensHost;
            state.lastPastKeyHosts = currentPastKeyHosts;
        } else {
            FastllmCudaSetDevice(gpuId);
            if (!FastllmCudaAdvanceDecodeMeta(
                    (int32_t*)state.buffers.insertPositions.cudaData,
                    (int32_t*)state.buffers.lastPageLens.cudaData,
                    batch)) {
                Qwen35DestroyCudaGraph(state);
                state.disabled = true;
                return false;
            }
            state.buffers.insertPositions.cpuIntDatas = insertPositionHost;
            state.buffers.lastPageLens.cpuIntDatas = lastPageLensHost;
            state.lastDecodePageLensHost = lastPageLensHost;
            state.lastPastKeyHosts = currentPastKeyHosts;
        }

        const DataType computeType = ResolveQwen35ThreadTpComputeType(this->dataType);
        const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;
        auto localHeadsFromScheme = [&](const DivisionScheme &scheme, int defaultHeads) {
            if (!tensorParallel) {
                return defaultHeads;
            }
            int heads = Qwen35LocalHeads(scheme, gpuId);
            return heads;
        };

        auto runGraphBody = [&]() {
            using namespace qwen3cuda;
            Qwen3CudaDirectRunner cudaRunner(gpuId);
            Qwen35ForwardSingleBuffers &buf = state.buffers;
            if (precomputedHiddenStates == nullptr) {
                Qwen3CudaEmbeddingDirect(cudaRunner, state.inputIds,
                                         *requireLocal(weight[language_prefix + "embed_tokens.weight"],
                                                       language_prefix + "embed_tokens.weight"),
                                         buf.hiddenStates);
            }
            if (buf.hiddenStates.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, buf.hiddenStates, computeType);
            }

            auto addPartialToResidualReduce = [&](Data &partial) {
                if (partial.dataType != buf.hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, partial, buf.hiddenStates.dataType);
                }
                if (tensorParallel) {
                    if (firstTensorParallelRank) {
                        Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, partial);
                    } else {
                        Qwen35CudaCopyTensor(cudaRunner, partial, buf.hiddenStates);
                    }
                    FastllmNcclAllReduce(buf.hiddenStates.cudaData, buf.hiddenStates.cudaData,
                                         buf.hiddenStates.Count(0), buf.hiddenStates.dataType, gpuId);
                } else {
                    Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, partial);
                }
            };

            bool generatedAppendParams = false;
            bool generatedDecodeParams = false;
            for (int i = 0; i < block_cnt; i++) {
                std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
                std::string inputRmsName = prefix + "input_layernorm.weight";
                std::string postRmsName = prefix + "post_attention_layernorm.weight";
                std::string swigluWeightName = prefix + "mlp.gateup_proj.weight";
                std::string downWeightName = prefix + "mlp.down_proj.weight";
                std::string downBiasName = prefix + "mlp.down_proj.bias";
                bool isAttentionLayer =
                    weight.weight.find(prefix + "self_attn.o_proj.weight") != weight.weight.end();

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[inputRmsName], inputRmsName),
                                 rms_norm_eps, buf.attenInput);
                int bsz = buf.attenInput.dims[0];
                int seqlen = buf.attenInput.dims[1];

                if (isAttentionLayer) {
                    std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                    std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
                    std::string qNormName = prefix + "self_attn.q_norm.weight";
                    std::string kNormName = prefix + "self_attn.k_norm.weight";
                    std::string oWeightName = prefix + "self_attn.o_proj.weight";
                    std::string oBiasName = prefix + "self_attn.o_proj.bias";

                    int localKVHeads = num_key_value_heads;
                    if (tensorParallel) {
                        AssertInFastLLM(i < (int)threadTpAttentionKVHeadSchemes.size(),
                                        "Qwen3.5 ForwardSingleGPU graph missing attention KV scheme.\n");
                        localKVHeads = localHeadsFromScheme(threadTpAttentionKVHeadSchemes[i],
                                                            num_key_value_heads);
                    }
                    if (localKVHeads > 0) {
                        int localQHeads = localKVHeads * (num_attention_heads / num_key_value_heads);
                        Qwen35CudaAttentionPagedBlock(
                            cudaRunner,
                            &buf.attenInput,
                            requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName),
                            requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                            requireLocal(weight[qNormName], qNormName),
                            requireLocal(weight[kNormName], kNormName),
                            requireLocal(weight[oWeightName], oWeightName),
                            requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                            &state.positionIds,
                            &pastKeyValues,
                            &buf.batchPastKeys, &buf.batchPastValues,
                            &buf.merged, &buf.qgate, &buf.gate,
                            &buf.q, &buf.k, &buf.v,
                            &buf.attenOutput, &buf.attenLastOutput,
                            &buf.qForAttentionHolder,
                            &buf.insertIndexs, &buf.insertPositions,
                            &buf.qSizes, &buf.pageSizes,
                            &buf.pageIndexs, &buf.lastPageLens,
                            &generatedAppendParams, &generatedDecodeParams,
                            batch, block_cnt, i, seqLens,
                            localQHeads, localKVHeads, head_dim,
                            rotary_dim, mrope_sections,
                            rms_norm_eps, rope_base, rope_factor,
                            rope_type, isPrefill,
                            &buf.hiddenStates,
                            pagedCacheLayerOffset,
                            true, true, true, 1);
                        Qwen3CudaLinearResidualReduce(
                            cudaRunner, buf.attenOutput,
                            *requireLocal(weight[oWeightName], oWeightName),
                            *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                            buf.attenLastOutput, buf.hiddenStates,
                            tensorParallel, firstTensorParallelRank, gpuId);
                    } else {
                        Qwen35ZeroCudaLike(buf.attenLastOutput, buf.hiddenStates, gpuId);
                        addPartialToResidualReduce(buf.attenLastOutput);
                    }
                } else {
                    std::string qkvzWeightName = prefix + "linear_attn.in_proj_qkvz.weight";
                    std::string baWeightName = prefix + "linear_attn.in_proj_ba.weight";
                    std::string qkvzbaWeightName = prefix + "linear_attn.in_proj_qkvzba.weight";
                    std::string conv1dWeightName = prefix + "linear_attn.conv1d.weight";
                    std::string conv1dBiasName = prefix + "linear_attn.conv1d.bias";
                    std::string aLogName = prefix + "linear_attn.A_log";
                    std::string dtBiasName = prefix + "linear_attn.dt_bias";
                    std::string outNormWeightName = prefix + "linear_attn.norm.weight";
                    std::string outProjWeightName = prefix + "linear_attn.out_proj.weight";

                    int localKeyHeads = num_k_heads;
                    int localValueHeads = num_v_heads;
                    if (tensorParallel) {
                        AssertInFastLLM(i < (int)threadTpLinearKeyHeadSchemes.size(),
                                        "Qwen3.5 ForwardSingleGPU graph missing linear key scheme.\n");
                        localKeyHeads = localHeadsFromScheme(threadTpLinearKeyHeadSchemes[i],
                                                             num_k_heads);
                        AssertInFastLLM(i < (int)threadTpLinearValueHeadSchemes.size(),
                                        "Qwen3.5 ForwardSingleGPU graph missing linear value scheme.\n");
                        localValueHeads = localHeadsFromScheme(threadTpLinearValueHeadSchemes[i],
                                                               num_v_heads);
                    }
                    int localKd = localKeyHeads * head_k_dim;
                    int localVd = localValueHeads * head_v_dim;
                    int localQkvDim = localKd * 2 + localVd;

                    bool hasMergedGdnInLinear =
                        weight.weight.find(qkvzbaWeightName) != weight.weight.end();
                    if (hasMergedGdnInLinear) {
                        Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                        *requireLocal(weight[qkvzbaWeightName], qkvzbaWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(qkvzbaWeightName + ".tp_bias"),
                                                      qkvzbaWeightName + ".tp_bias"),
                                        buf.gdnMerged);
                    } else {
                        AssertInFastLLM(weight.weight.find(qkvzWeightName) != weight.weight.end() &&
                                        weight.weight.find(baWeightName) != weight.weight.end(),
                                        "Qwen3.5 CUDA graph requires linear attention qkvz/ba weights.\n");
                        Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                        *requireLocal(weight[qkvzWeightName], qkvzWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(qkvzWeightName + ".tp_bias"),
                                                      qkvzWeightName + ".tp_bias"),
                                        buf.gdnMerged);
                    }
                    Qwen3CudaSplit(cudaRunner, buf.gdnMerged, -1, 0, localQkvDim, buf.qkvConvInput);
                    Qwen3CudaSplit(cudaRunner, buf.gdnMerged, -1, localQkvDim,
                                   localQkvDim + localVd, buf.z);
                    if (hasMergedGdnInLinear) {
                        Qwen3CudaSplit(cudaRunner, buf.gdnMerged, -1, localQkvDim + localVd,
                                       localQkvDim + localVd + localValueHeads * 2, buf.ba);
                    } else {
                        Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                        *requireLocal(weight[baWeightName], baWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(baWeightName + ".tp_bias"),
                                                      baWeightName + ".tp_bias"),
                                        buf.ba);
                    }

                    Data &pastKey = *pastKeyValues[i].first;
                    if (batch == 1) {
                        SwapSingleTokenSeqHeadByReshape(buf.qkvConvInput);
                    } else {
                        buf.qkvConvInput.Reshape({batch, buf.qkvConvInput.dims.back(), 1});
                    }
                    buf.z.Reshape({bsz, seqlen, localValueHeads, head_v_dim});

                    for (int bidx = 0; bidx < batch; bidx++) {
                        buf.linearConvCaches[bidx] = pastKeyValues[bidx * block_cnt + i].first;
                    }
                    bool directBatchDecodeConvSilu = false;
                    PagedCacheManager *linearConvPool = Qwen35FindLinearSlotPool(
                        this, gpuId, i, QWEN35_LINEAR_SLOT_CONV,
                        linearSlotCapacity);
                    if (linearConvPool != nullptr && state.linearSlotIds.cudaData != nullptr) {
                        directBatchDecodeConvSilu =
                            FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots(
                                linearConvPool->cudaData, state.linearSlotIds.cudaData,
                                batch, *buf.linearConvCaches[0], buf.qkvConvInput,
                                *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                buf.convOutput);
                    }
                    if (directBatchDecodeConvSilu) {
                        buf.convOutput.Reshape({1, batch, buf.convOutput.dims[1]});
                    } else if (batch == 1) {
                        bool fusedDecodeConvSilu = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                            pastKey, buf.qkvConvInput,
                            *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                            *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                            buf.convOutput);
                        if (!fusedDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(pastKey, buf.qkvConvInput);
                            Qwen35CudaConv1DPerChannel(cudaRunner, pastKey,
                                                       *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                                       *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                                       pastKey.dims[1],
                                                       requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                                       4, 1, 0, buf.convOutput);
                            Qwen35CudaSilu(cudaRunner, buf.convOutput, buf.convOutput);
                        }
                        SwapSingleTokenSeqHeadByReshape(buf.convOutput);
                    } else {
                        CatBatchFirstDim(buf.linearConvCaches, buf.batchConvCache);
                        bool fusedBatchDecodeConvSilu = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                            buf.batchConvCache, buf.qkvConvInput,
                            *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                            *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                            buf.convOutput);
                        if (!fusedBatchDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(buf.batchConvCache, buf.qkvConvInput);
                            Qwen35CudaConv1DPerChannel(cudaRunner, buf.batchConvCache,
                                                       *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                                       *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                                       buf.batchConvCache.dims[1],
                                                       requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                                       4, 1, 0, buf.convOutput);
                            Qwen35CudaSilu(cudaRunner, buf.convOutput, buf.convOutput);
                        }
                        SplitBatchFirstDim(buf.batchConvCache, buf.linearConvCaches);
                        buf.convOutput.Reshape({1, batch, buf.convOutput.dims[1]});
                    }
                    for (int bidx = 0; bidx < batch; bidx++) {
                        Qwen35PrepareLinearAttentionCache(*buf.linearConvCaches[bidx], computeType);
                    }

                    for (int rb = 0; rb < batch; rb++) {
                        buf.recurrentStates[rb] = pastKeyValues[rb * block_cnt + i].second;
                    }
                    PagedCacheManager *linearStatePool = Qwen35FindLinearSlotPool(
                        this, gpuId, i, QWEN35_LINEAR_SLOT_RECURRENT,
                        linearSlotCapacity);
                    bool fusedBatchRecurrentFromConvBa = false;
                    if (linearStatePool != nullptr && state.linearSlotIds.cudaData != nullptr) {
                        float recurrentQScale = 1.0f / std::sqrt((float)head_k_dim);
                        fusedBatchRecurrentFromConvBa =
                            FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots(
                                buf.convOutput, buf.ba,
                                *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                *requireLocal(weight[aLogName], aLogName),
                                *requireLocal(weight[dtBiasName], dtBiasName),
                                linearStatePool->cudaData, state.linearSlotIds.cudaData, batch,
                                buf.coreAttnOut,
                                localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                rms_norm_eps, recurrentQScale);
                    }
                    if (fusedBatchRecurrentFromConvBa) {
                        for (int rb = 0; rb < batch; rb++) {
                            Qwen35PrepareLinearAttentionCache(*buf.recurrentStates[rb], computeType);
                        }
                        buf.coreAttnOut.Reshape({1, batch, buf.coreAttnOut.dims[1],
                                                 buf.coreAttnOut.dims[3]});
                    }

                    if (!fusedBatchRecurrentFromConvBa) {
                        Qwen3CudaSplit(cudaRunner, buf.convOutput, -1, 0, localKd, buf.linearQ);
                        Qwen3CudaSplit(cudaRunner, buf.convOutput, -1, localKd, localKd * 2, buf.linearK);
                        Qwen3CudaSplit(cudaRunner, buf.convOutput, -1, localKd * 2,
                                       localKd * 2 + localVd, buf.linearV);
                        buf.linearQ.Reshape({buf.linearQ.dims[0], buf.linearQ.dims[1],
                                             localKeyHeads, head_k_dim});
                        buf.linearK.Reshape({buf.linearK.dims[0], buf.linearK.dims[1],
                                             localKeyHeads, head_k_dim});
                        buf.linearV.Reshape({buf.linearV.dims[0], buf.linearV.dims[1],
                                             localValueHeads, head_v_dim});
                        Qwen3CudaSplit(cudaRunner, buf.ba, -1, 0, localValueHeads, buf.b);
                        Qwen3CudaSplit(cudaRunner, buf.ba, -1,
                                       localValueHeads, localValueHeads * 2, buf.a);
                        Qwen35CudaSigmoidMambaSoftplus(cudaRunner, buf.b, buf.a,
                                                       *requireLocal(weight[aLogName], aLogName),
                                                       *requireLocal(weight[dtBiasName], dtBiasName),
                                                       buf.g);
                        Qwen3CudaRMSNorm(cudaRunner, buf.linearQ,
                                         *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                         rms_norm_eps, buf.linearQ);
                        Qwen3CudaRMSNorm(cudaRunner, buf.linearK,
                                         *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                         rms_norm_eps, buf.linearK);
                        buf.linearQ.Reshape({batch, localKeyHeads, 1, head_k_dim});
                        buf.linearK.Reshape({batch, localKeyHeads, 1, head_k_dim});
                        buf.linearV.Reshape({batch, localValueHeads, 1, head_v_dim});
                        buf.b.Reshape({batch, localValueHeads, 1});
                        buf.g.Reshape({batch, localValueHeads, 1});
                        CatBatchFirstDim(buf.recurrentStates, buf.batchRecurrentState);
                        float recurrentQScale = 1.0f / std::sqrt((float)head_k_dim);
                        Qwen35CudaRecurrentGatedDeltaRule(cudaRunner, buf.linearQ, buf.linearK,
                                                          buf.linearV, buf.g, buf.b,
                                                          buf.batchRecurrentState, buf.coreAttnOut,
                                                          recurrentQScale);
                        SplitBatchFirstDim(buf.batchRecurrentState, buf.recurrentStates);
                        for (int rb = 0; rb < batch; rb++) {
                            Qwen35PrepareLinearAttentionCache(*buf.recurrentStates[rb], computeType);
                        }
                        buf.coreAttnOut.Reshape({1, batch, buf.coreAttnOut.dims[1],
                                                 buf.coreAttnOut.dims[3]});
                    }

                    std::vector<int> zShape = buf.z.dims;
                    buf.coreAttnOut.Reshape({-1, buf.coreAttnOut.dims.back()});
                    buf.z.Reshape({-1, buf.z.dims.back()});
                    bool fusedPostLinearAttn =
                        Qwen35TryCudaRMSNormSiluMul(
                            buf.coreAttnOut,
                            *requireLocal(weight[outNormWeightName], outNormWeightName),
                            buf.z, buf.coreAttnOut, rms_norm_eps);
                    if (!fusedPostLinearAttn) {
                        Qwen3CudaRMSNorm(cudaRunner, buf.coreAttnOut,
                                         *requireLocal(weight[outNormWeightName], outNormWeightName),
                                         rms_norm_eps, buf.coreAttnOut);
                        Qwen35CudaSilu(cudaRunner, buf.z, buf.z);
                        if (buf.z.dataType != buf.coreAttnOut.dataType) {
                            Qwen3CudaToDataType(cudaRunner, buf.z, buf.coreAttnOut.dataType);
                        }
                        Qwen35CudaMulTo(cudaRunner, buf.coreAttnOut, buf.z);
                    }
                    buf.coreAttnOut.Reshape({zShape[0], zShape[1], localVd});
                    Qwen3CudaLinearResidualReduce(
                        cudaRunner, buf.coreAttnOut,
                        *requireLocal(weight[outProjWeightName], outProjWeightName),
                        *requireLocal(GetThreadTensorParallelBias(outProjWeightName + ".tp_bias"),
                                      outProjWeightName + ".tp_bias"),
                        buf.attenLastOutput, buf.hiddenStates,
                        tensorParallel, firstTensorParallelRank, gpuId);
                }

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[postRmsName], postRmsName),
                                 rms_norm_eps, buf.attenInput);
                bool hasDenseMlp = weight.weight.find(swigluWeightName) != weight.weight.end() &&
                                   weight.weight.find(downWeightName) != weight.weight.end();
                if (hasDenseMlp) {
                    Data &gateUpWeight = *requireLocal(weight[swigluWeightName], swigluWeightName);
                    Data &gateUpBias = *requireLocal(GetThreadTensorParallelBias(swigluWeightName + ".tp_bias"),
                                                     swigluWeightName + ".tp_bias");
                    Data &downWeight = *requireLocal(weight[downWeightName], downWeightName);
                    Data &downBias = *requireLocal(GetThreadTensorParallelBias(downBiasName), downBiasName);
                    if (!Qwen3CudaTrySwigluLinearResidualReduce(
                            cudaRunner, buf.attenInput, gateUpWeight, gateUpBias,
                            downWeight, downBias, buf.gateupResult, buf.swigluResult, buf.mlpPart,
                            buf.hiddenStates, tensorParallel)) {
                        Qwen3CudaLinearSwiglu(cudaRunner, buf.attenInput,
                                              gateUpWeight, gateUpBias,
                                              buf.gateupResult, buf.swigluResult);
                        Qwen3CudaLinearResidualReduce(
                            cudaRunner, buf.swigluResult,
                            downWeight, downBias,
                            buf.mlpPart, buf.hiddenStates,
                            tensorParallel, firstTensorParallelRank, gpuId);
                    }
                    continue;
                }

                std::string gateWeightName = prefix + "mlp.gate.weight";
                std::string gateBiasName = prefix + "mlp.gate.e_score_correction_bias";
                std::string sharedGateupWeightName = prefix + "mlp.shared_expert.gateup_proj.weight";
                std::string sharedDownWeightName = prefix + "mlp.shared_expert.down_proj.weight";
                std::string sharedExpertGateWeightName = prefix + "mlp.shared_expert_gate.weight";
                AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                "Qwen3.5 CUDA graph layer has neither dense MLP nor router gate weight.\n");
                AssertInFastLLM(!Qwen35LayerUsesMappedNonCudaMoe(this, i),
                                "Qwen3.5 CUDA graph doesn't support non-CUDA moe_device layers.\n");

                bool sharedExpertPending = false;
                if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                    AssertInFastLLM(weight.weight.find(sharedGateupWeightName) != weight.weight.end(),
                                    "Qwen3.5 CUDA graph requires merged shared expert gateup weight.\n");
                    Qwen3CudaLinearSwiglu(cudaRunner, buf.attenInput,
                                          *requireLocal(weight[sharedGateupWeightName], sharedGateupWeightName),
                                          *requireLocal(GetThreadTensorParallelBias(sharedGateupWeightName + ".tp_bias"),
                                                        sharedGateupWeightName + ".tp_bias"),
                                          buf.gateupResult, buf.swigluResult);
                    Qwen3CudaLinear(cudaRunner, buf.swigluResult,
                                    *requireLocal(weight[sharedDownWeightName], sharedDownWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(sharedDownWeightName + ".tp_bias"),
                                                  sharedDownWeightName + ".tp_bias"),
                                    buf.sharedOutput);
                    if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                        Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                        *requireLocal(weight[sharedExpertGateWeightName], sharedExpertGateWeightName),
                                        *GetEmptyData(), buf.sharedGate);
                        Qwen35CudaSigmoid(cudaRunner, buf.sharedGate, buf.sharedGate);
                        if (buf.sharedGate.dataType != buf.sharedOutput.dataType) {
                            Qwen3CudaToDataType(cudaRunner, buf.sharedGate, buf.sharedOutput.dataType);
                        }
                        Qwen35CudaMulTo(cudaRunner, buf.sharedOutput, buf.sharedGate);
                    }
                    // 共享专家输出不单独归约：延后与路由专家输出本地相加后一次 allReduce，
                    // 每个 MoE 层省一次集合通信（小 hidden 下 allReduce 为纯延迟型，TP 解码收益明显）。
                    sharedExpertPending = true;
                }

                int localBatch = buf.attenInput.dims[0];
                int localLen = buf.attenInput.dims[1];
                buf.attenInput.Reshape({localBatch * localLen, buf.attenInput.dims[2]});
                Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                *requireLocal(weight[gateWeightName], gateWeightName),
                                *GetEmptyData(), buf.routerLogits, true);
                Qwen3CudaConvertToDataType(cudaRunner, buf.routerLogits,
                                           buf.routerLogitsTemp, DataType::FLOAT32);
                Qwen3CudaSoftmax(cudaRunner, buf.routerLogitsTemp, buf.routerLogitsTemp, -1);
                Data *localGateBias = nullptr;
                if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                    localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
                }
                Qwen3CudaSelectExpert(cudaRunner, buf.routerLogitsTemp,
                                      buf.expertIndex, buf.expertScore,
                                      this->num_experts_per_tok, this->norm_topk_prob,
                                      this->routed_scaling_factor, localGateBias);

                if (HasFusedMoeWeights(i)) {
                    Data *localGate = GetFusedMoeWeightForDevice(moeGate3DWeights[i], gpuId);
                    Data *localUp = GetFusedMoeWeightForDevice(moeUp3DWeights[i], gpuId);
                    Data *localDown = GetFusedMoeWeightForDevice(moeDown3DWeights[i], gpuId);
                    if (Qwen35HasLocalFusedMoeShard(localGate, localUp, localDown)) {
                        Qwen35CudaFusedMOE(cudaRunner, buf.attenInput,
                                           buf.expertIndex, buf.expertScore,
                                           *localGate, *localUp, *localDown,
                                           buf.w1, buf.moeFinal, i);
                    } else {
                        Qwen35ZeroCudaLike(buf.moeFinal, buf.hiddenStates, gpuId);
                    }
                } else {
                    auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                    auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                    if (Qwen35HasLocalMoeShard(localWeights)) {
                        Qwen3CudaMergeMOEBlock(cudaRunner, &buf.attenInput,
                            &buf.expertIndex, &buf.expertScore,
                            &localWeights, &localBiass,
                            &buf.w1, &buf.w2, &buf.w3,
                            &buf.tempInput, &buf.tempOutput,
                            1.0f, &buf.moeFinal, i,
                            computeType, threadTpMoeAtype,
                            &buf.moeInputTemp, &buf.moeOutputTemp);
                    } else {
                        Qwen35ZeroCudaLike(buf.moeFinal, buf.hiddenStates, gpuId);
                    }
                }
                buf.moeFinal.Reshape(buf.hiddenStates.dims);
                if (sharedExpertPending) {
                    if (buf.sharedOutput.dataType != buf.moeFinal.dataType) {
                        Qwen3CudaToDataType(cudaRunner, buf.sharedOutput, buf.moeFinal.dataType);
                    }
                    buf.sharedOutput.Reshape(buf.moeFinal.dims);
                    Qwen3CudaAddTo(cudaRunner, buf.moeFinal, buf.sharedOutput);
                }
                addPartialToResidualReduce(buf.moeFinal);
            }

            Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                             *requireLocal(weight[language_prefix + "norm.weight"],
                                           language_prefix + "norm.weight"),
                             rms_norm_eps, buf.hiddenStates);
            Qwen3CudaLinear(cudaRunner, buf.hiddenStates,
                            *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                            *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                          "lm_head.weight.tp_bias"),
                            buf.logitsHalf);
            Qwen3CudaConvertToDataType(cudaRunner, buf.logitsHalf, state.logits, DataType::FLOAT32);
        };

        auto finishWithLogits = [&]() {
            Qwen35PrepareGraphCudaTensor(logits, state.logits, gpuId);
        };

        auto runWithoutGraph = [&]() -> bool {
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            runGraphBody();
            bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
            finishWithLogits();
            return usedUnsafeMoeFallback;
        };

        // From here on every graph failure is handled by executing the same
        // prepared decode eagerly and returning true. Keep the appended cache
        // metadata; the guard above only rolls it back on a false/exceptional
        // pre-execution exit so the outer eager fallback cannot append twice.
        graphPagedRollbackArmed = false;
        if (state.captured) {
            bool replayOk = FastllmCudaGraphLaunch(state.exec);
            if (replayOk) {
                finishWithLogits();
                return true;
            }
            printf("Warning: Qwen3.5 CUDA graph replay failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        if (!allowCapture) {
            runWithoutGraph();
            return true;
        }

        if (!state.warmed) {
            FastllmCudaClearThreadError();
            bool usedUnsafeMoeFallback = runWithoutGraph();
            if (usedUnsafeMoeFallback) {
                printf("Warning: Qwen3.5 CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during warmup.\n",
                       gpuId);
                fflush(stdout);
                Qwen35DestroyCudaGraph(state);
                state.disabled = true;
                return true;
            }
            if (FastllmCudaGetThreadError()) {
                printf("Warning: Qwen3.5 CUDA graph disabled on gpu %d because CUDA errors occurred during warmup run.\n",
                       gpuId);
                fflush(stdout);
                Qwen35DestroyCudaGraph(state);
                state.disabled = true;
                return true;
            }
            state.warmed = true;
            return true;
        }

        void *capturedGraph = nullptr;
        FastllmCudaClearThreadError();
        if (!FastllmCudaGraphBeginCapture()) {
            printf("Warning: Qwen3.5 CUDA graph begin capture failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
        runGraphBody();
        bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
        if (usedUnsafeMoeFallback) {
            printf("Warning: Qwen3.5 CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during capture.\n",
                   gpuId);
            fflush(stdout);
            Qwen35AbortCudaGraphCapture();
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        if (FastllmCudaGetThreadError() || FastllmCudaGraphCaptureInvalidated()) {
            printf("Warning: Qwen3.5 CUDA graph disabled on gpu %d because errors occurred inside capture body. Fallback to eager mode.\n",
                   gpuId);
            fflush(stdout);
            Qwen35AbortCudaGraphCapture();
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        if (!FastllmCudaGraphEndCapture(&capturedGraph) || capturedGraph == nullptr) {
            printf("Warning: Qwen3.5 CUDA graph end capture failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            if (capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        void *capturedExec = nullptr;
        if (!FastllmCudaGraphInstantiate(capturedGraph, &capturedExec) || capturedExec == nullptr) {
            printf("Warning: Qwen3.5 CUDA graph instantiate failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            FastllmCudaGraphDestroy(capturedGraph);
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        state.graph = capturedGraph;
        state.exec = capturedExec;
        state.captured = true;
        bool firstLaunchOk = FastllmCudaGraphLaunch(state.exec);
        if (!firstLaunchOk) {
            printf("Warning: Qwen3.5 CUDA graph first launch failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            Qwen35DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        finishWithLogits();
        return true;
#endif
    }

    void Qwen3_5Model::ForwardSingleGPU(
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
        ErrorInFastLLM("Qwen3.5 ForwardSingleGPU requires CUDA.\n");
#else
        using namespace qwen3cuda;
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "Qwen3.5 ForwardSingleGPU got invalid GPU ratio.\n");
        FastllmCudaSetDevice(gpuId);
        Qwen3CudaDirectRunner cudaRunner(gpuId);
        int mtpWorkerProfileInterval = speculativeCollectAllLogits ?
            Qwen35MtpWorkerProfileInterval() : 0;
        bool mtpWorkerProfileEnabled = mtpWorkerProfileInterval > 0;
        int mtpWorkerProfileSeqTokens = 0;
        for (int len : seqLens) {
            mtpWorkerProfileSeqTokens += len;
        }
        auto mtpWorkerProfileStart = mtpWorkerProfileEnabled ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();
        auto mtpWorkerProfileLast = mtpWorkerProfileStart;
        long long mtpWorkerProfileSetupUs = 0;
        long long mtpWorkerProfileLayersUs = 0;
        long long mtpWorkerProfileHeadUs = 0;
        auto mtpWorkerProfileSyncMark = [&](long long &slot) {
            if (!mtpWorkerProfileEnabled) {
                return;
            }
            ForceDeviceSync();
            auto now = std::chrono::steady_clock::now();
            slot += Qwen35MtpProfileElapsedUs(mtpWorkerProfileLast, now);
            mtpWorkerProfileLast = now;
        };
        auto mtpWorkerProfileRecord = [&]() {
            if (!mtpWorkerProfileEnabled) {
                return;
            }
            long long totalUs = Qwen35MtpProfileElapsedUs(
                mtpWorkerProfileStart, std::chrono::steady_clock::now());
            Qwen35MtpWorkerProfileRecord(
                mtpWorkerProfileInterval, firstTensorParallelRank,
                mtpWorkerProfileSeqTokens, mtpWorkerProfileSetupUs,
                mtpWorkerProfileLayersUs, mtpWorkerProfileHeadUs, totalUs);
        };

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (data.dims.empty()) {
                return &data;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Qwen3.5 ForwardSingleGPU missing local tensor: " + name + ".\n");
            return nullptr;
        };

        auto localHeadsFromScheme = [&](const DivisionScheme &scheme, int defaultHeads) {
            if (!tensorParallel) {
                return defaultHeads;
            }
            int heads = Qwen35LocalHeads(scheme, gpuId);
            return heads;
        };

        const DataType computeType = ResolveQwen35ThreadTpComputeType(this->dataType);
        const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
        Data localHiddenStates;
        Data *hiddenStatesPtr = nullptr;
        if (!speculativeCollectAllLogits &&
            ForwardSingleGPUDecodeGraph(gpuId, ratios, batch, inputIds, positionIds,
                                        seqLens, pastKeyValues, all1, isPrefill,
                                        tensorParallel, firstTensorParallelRank,
                                        pagedCacheLayerOffset, logits,
                                        precomputedHiddenStates)) {
            return;
        }
        if (precomputedHiddenStates != nullptr) {
            hiddenStatesPtr = requireLocal(*precomputedHiddenStates, "precomputedHiddenStates");
        } else {
            Qwen3CudaEmbeddingDirect(cudaRunner,
                                     *requireLocal((Data&)inputIds, "inputIds"),
                                     *requireLocal(weight[language_prefix + "embed_tokens.weight"],
                                                   language_prefix + "embed_tokens.weight"),
                                     localHiddenStates);
            hiddenStatesPtr = &localHiddenStates;
        }
        Data &hiddenStates = *hiddenStatesPtr;
        if (hiddenStates.dataType != computeType) {
            Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
        }
        mtpWorkerProfileSyncMark(mtpWorkerProfileSetupUs);

        Data attenInput, merged, qgate, gate, q, k, v, attenOutput, attenLastOutput;
        Data qForAttentionHolder;
        Data gateupResult, swigluResult, mlpPart;
        Data routerLogits, routerLogitsTemp, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        Data moeFinal, sharedGate, sharedOutput;
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        Data gdnMerged, baMerged, qkvConvInput, z, b, a, g, conv, convOutput, coreAttnOut, coreTemp;
        Data convInputWithCache;
        Data convToken0, convToken1, convOutput0, convOutput1;
        Data convRow0, convRow1, baRow0, baRow1;
        Data q0, k0, v0, a0, b0, coreAttnOut0;
        Data q1, k1, v1, a1, b1, coreAttnOut1;
        Data qRepeat, kRepeat, qq, qTemp, kkPad, vvPad, bbPad, ggPad, decayMask;
        Data kBeta, vBeta, attn, at, kCumdecay, gExp;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        bool generatedAppendParams = false;
        bool generatedDecodeParams = false;
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;

        auto addPartialToResidualReduce = [&](Data &partial) {
            if (partial.dataType != hiddenStates.dataType) {
                Qwen3CudaToDataType(cudaRunner, partial, hiddenStates.dataType);
            }
            if (tensorParallel) {
                if (firstTensorParallelRank) {
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, partial);
                } else {
                    Qwen35CudaCopyTensor(cudaRunner, partial, hiddenStates);
                }
                FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                     hiddenStates.Count(0), hiddenStates.dataType, gpuId);
            } else {
                Qwen3CudaAddTo(cudaRunner, hiddenStates, partial);
            }
        };

        for (int i = 0; i < block_cnt; i++) {
            std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
            std::string inputRmsName = prefix + "input_layernorm.weight";
            std::string postRmsName = prefix + "post_attention_layernorm.weight";
            std::string swigluWeightName = prefix + "mlp.gateup_proj.weight";
            std::string downWeightName = prefix + "mlp.down_proj.weight";
            std::string downBiasName = prefix + "mlp.down_proj.bias";
            bool isAttentionLayer =
                weight.weight.find(prefix + "self_attn.o_proj.weight") != weight.weight.end();
            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[inputRmsName], inputRmsName),
                             rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0];
            int seqlen = attenInput.dims[1];

            if (isAttentionLayer) {
                std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
                std::string qNormName = prefix + "self_attn.q_norm.weight";
                std::string kNormName = prefix + "self_attn.k_norm.weight";
                std::string oWeightName = prefix + "self_attn.o_proj.weight";
                std::string oBiasName = prefix + "self_attn.o_proj.bias";

                int localKVHeads = num_key_value_heads;
                if (tensorParallel) {
                    AssertInFastLLM(i < (int)threadTpAttentionKVHeadSchemes.size(),
                                    "Qwen3.5 ForwardSingleGPU missing attention KV scheme.\n");
                    localKVHeads = localHeadsFromScheme(threadTpAttentionKVHeadSchemes[i],
                                                        num_key_value_heads);
                }
                if (localKVHeads > 0) {
                    int localQHeads = localKVHeads * (num_attention_heads / num_key_value_heads);
                    Qwen35CudaAttentionPagedBlock(
                        cudaRunner,
                        &attenInput,
                        requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName),
                        requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                        requireLocal(weight[qNormName], qNormName),
                        requireLocal(weight[kNormName], kNormName),
                        requireLocal(weight[oWeightName], oWeightName),
                        requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                        requireLocal((Data&)positionIds, "positionIds"),
                        &pastKeyValues,
                        &batchPastKeys, &batchPastValues,
                        &merged, &qgate, &gate,
                        &q, &k, &v,
                        &attenOutput, &attenLastOutput,
                        &qForAttentionHolder,
                        &insertIndexs, &insertPositions,
                        &qSizes, &pageSizes,
                        &pageIndexs, &lastPageLens,
                        &generatedAppendParams, &generatedDecodeParams,
                        batch, block_cnt, i, seqLens,
                        localQHeads, localKVHeads, head_dim,
                        rotary_dim, mrope_sections,
                        rms_norm_eps, rope_base, rope_factor,
                        rope_type, isPrefill,
                        &hiddenStates,
                        pagedCacheLayerOffset,
                        true, false);
                    Qwen3CudaLinearResidualReduce(
                        cudaRunner, attenOutput,
                        *requireLocal(weight[oWeightName], oWeightName),
                        *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                        attenLastOutput, hiddenStates,
                        tensorParallel, firstTensorParallelRank, gpuId);
                } else {
                    Qwen35ZeroCudaLike(attenLastOutput, hiddenStates, gpuId);
                    addPartialToResidualReduce(attenLastOutput);
                }
            } else {
                std::string qkvzWeightName = prefix + "linear_attn.in_proj_qkvz.weight";
                std::string baWeightName = prefix + "linear_attn.in_proj_ba.weight";
                std::string qkvzbaWeightName = prefix + "linear_attn.in_proj_qkvzba.weight";
                std::string conv1dWeightName = prefix + "linear_attn.conv1d.weight";
                std::string conv1dBiasName = prefix + "linear_attn.conv1d.bias";
                std::string aLogName = prefix + "linear_attn.A_log";
                std::string dtBiasName = prefix + "linear_attn.dt_bias";
                std::string outNormWeightName = prefix + "linear_attn.norm.weight";
                std::string outProjWeightName = prefix + "linear_attn.out_proj.weight";

                int localKeyHeads = num_k_heads;
                int localValueHeads = num_v_heads;
                if (tensorParallel) {
                    AssertInFastLLM(i < (int)threadTpLinearKeyHeadSchemes.size(),
                                    "Qwen3.5 ForwardSingleGPU missing linear key scheme.\n");
                    localKeyHeads = localHeadsFromScheme(threadTpLinearKeyHeadSchemes[i],
                                                         num_k_heads);
                    AssertInFastLLM(i < (int)threadTpLinearValueHeadSchemes.size(),
                                    "Qwen3.5 ForwardSingleGPU missing linear value scheme.\n");
                    localValueHeads = localHeadsFromScheme(threadTpLinearValueHeadSchemes[i],
                                                           num_v_heads);
                }
                int localKd = localKeyHeads * head_k_dim;
                int localVd = localValueHeads * head_v_dim;
                int localQkvDim = localKd * 2 + localVd;
                bool hasMergedGdnInLinear =
                    weight.weight.find(qkvzbaWeightName) != weight.weight.end();
                if (hasMergedGdnInLinear) {
                    Qwen3CudaLinear(cudaRunner, attenInput,
                                    *requireLocal(weight[qkvzbaWeightName], qkvzbaWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(qkvzbaWeightName + ".tp_bias"),
                                                  qkvzbaWeightName + ".tp_bias"),
                                    gdnMerged);
                } else {
                    AssertInFastLLM(weight.weight.find(qkvzWeightName) != weight.weight.end() &&
                                    weight.weight.find(baWeightName) != weight.weight.end(),
                                    "Qwen3.5 ForwardSingleGPU requires linear attention qkvz/ba weights.\n");
                    Qwen3CudaLinear(cudaRunner, attenInput,
                                    *requireLocal(weight[qkvzWeightName], qkvzWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(qkvzWeightName + ".tp_bias"),
                                                  qkvzWeightName + ".tp_bias"),
                                    gdnMerged);
                }
                Qwen3CudaSplit(cudaRunner, gdnMerged, -1, 0, localQkvDim, qkvConvInput);
                Qwen3CudaSplit(cudaRunner, gdnMerged, -1, localQkvDim, localQkvDim + localVd, z);
                bool captureLinearState =
                    speculativeCaptureFirstTokenLinearState;
                int linearStateCaptureSlots =
                    speculativeLinearStateCaptureSlots > 0 ?
                    speculativeLinearStateCaptureSlots : 1;
                auto getLinearCaptureSlot = [&](int tokenIndex, bool keyState) -> Data* {
                    if (!captureLinearState || tokenIndex < 0) {
                        return nullptr;
                    }
                    if (speculativeLinearStateCaptureSlots > 0 &&
                        i < (int)speculativeLinearStates.size() &&
                        tokenIndex < (int)speculativeLinearStates[i].size()) {
                        Data &slotRoot = keyState ?
                            speculativeLinearStates[i][tokenIndex].first :
                            speculativeLinearStates[i][tokenIndex].second;
                        if (!tensorParallel) {
                            return &slotRoot;
                        }
                        auto slotIt = slotRoot.multiDeviceDatas.find(gpuId);
                        return slotIt == slotRoot.multiDeviceDatas.end() ?
                            nullptr : slotIt->second;
                    }
                    if (tokenIndex != 0 ||
                        i >= (int)speculativeFirstTokenLinearStates.size()) {
                        return nullptr;
                    }
                    Data &slotRoot = keyState ?
                        speculativeFirstTokenLinearStates[i].first :
                        speculativeFirstTokenLinearStates[i].second;
                    if (!tensorParallel) {
                        return &slotRoot;
                    }
                    auto slotIt = slotRoot.multiDeviceDatas.find(gpuId);
                    return slotIt == slotRoot.multiDeviceDatas.end() ?
                        nullptr : slotIt->second;
                };
                auto markLinearCaptured = [&](int tokenIndex, int mask) {
                    if (!tensorParallel && speculativeLinearStateCaptureSlots > 0 &&
                        i < (int)speculativeLinearCaptureMask.size() &&
                        tokenIndex >= 0 &&
                        tokenIndex < (int)speculativeLinearCaptureMask[i].size()) {
                        speculativeLinearCaptureMask[i][tokenIndex] |= mask;
                        return;
                    }
                    if (tokenIndex != 0) {
                        return;
                    }
                    if (!tensorParallel &&
                        i < (int)speculativeFirstTokenLinearCaptureMask.size()) {
                        speculativeFirstTokenLinearCaptureMask[i] |= mask;
                    }
                };
                auto getFirstTokenLinearCaptureSlot = [&](bool keyState) -> Data* {
                    return getLinearCaptureSlot(0, keyState);
                };
                auto markFirstTokenLinearCaptured = [&](int mask) {
                    markLinearCaptured(0, mask);
                };
                bool keepCombinedBaForTwoToken =
                    speculativeCaptureFirstTokenLinearState && batch == 1 && bsz == 1 && seqlen == 2;
                bool keepCombinedBaForSmallDecode =
                    speculativeCaptureFirstTokenLinearState && batch == 1 && bsz == 1 &&
                    seqlen > 1 && seqlen <= QWEN35_MTP_FAST_SEQ_MAX;
                bool keepCombinedBaForBatchRecurrent = batch > 1 && all1;
                bool keepCombinedBa =
                    keepCombinedBaForSmallDecode || keepCombinedBaForBatchRecurrent;
                bool baSplitReady = false;
                if (hasMergedGdnInLinear) {
                    if (keepCombinedBa) {
                        Qwen3CudaSplit(cudaRunner, gdnMerged, -1,
                                       localQkvDim + localVd,
                                       localQkvDim + localVd + localValueHeads * 2,
                                       baMerged);
                    } else {
                        Qwen3CudaSplit(cudaRunner, gdnMerged, -1, localQkvDim + localVd,
                                       localQkvDim + localVd + localValueHeads, b);
                        Qwen3CudaSplit(cudaRunner, gdnMerged, -1,
                                       localQkvDim + localVd + localValueHeads,
                                       localQkvDim + localVd + localValueHeads * 2, a);
                        baSplitReady = true;
                    }
                } else {
                    Qwen3CudaLinear(cudaRunner, attenInput,
                                    *requireLocal(weight[baWeightName], baWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(baWeightName + ".tp_bias"),
                                                  baWeightName + ".tp_bias"),
                                    baMerged);
                    if (!keepCombinedBa) {
                        Qwen3CudaSplit(cudaRunner, baMerged, -1, 0, localValueHeads, b);
                        Qwen3CudaSplit(cudaRunner, baMerged, -1, localValueHeads,
                                       localValueHeads * 2, a);
                        baSplitReady = true;
                    }
                }
                auto ensureBaSplit = [&]() {
                    if (baSplitReady) {
                        return;
                    }
                    AssertInFastLLM(!baMerged.dims.empty(),
                                    "Qwen3.5 linear attention missing combined ba tensor.\n");
                    Qwen3CudaSplit(cudaRunner, baMerged, -1, 0, localValueHeads, b);
                    Qwen3CudaSplit(cudaRunner, baMerged, -1, localValueHeads,
                                   localValueHeads * 2, a);
                    baSplitReady = true;
                };

                Data &pastKey = *pastKeyValues[i].first;
                Data &pastValue = *pastKeyValues[i].second;
                Qwen35PrepareLinearAttentionCache(pastKey, computeType);
                Qwen35PrepareLinearAttentionCache(pastValue, computeType);
                if (tensorParallel) {
                    pastKey.dataDeviceIds = {gpuId};
                    pastValue.dataDeviceIds = {gpuId};
                }
                if (batch == 1 && all1 && pastKey.dims.size() > 0) {
                    SwapSingleTokenSeqHeadByReshape(qkvConvInput);
                } else if (batch > 1 && all1) {
                    qkvConvInput.Reshape({batch, qkvConvInput.dims.back(), 1});
                } else {
                    Qwen3CudaPermuteSelf(cudaRunner, qkvConvInput, {0, 2, 1});
                }
                z.Reshape({bsz, seqlen, localValueHeads, head_v_dim});

                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    bool fusedDecodeConvSilu = false;
                    bool canTryFusedDecodeConvSilu =
                        pastKey.dataDevice == DataDevice::CUDA &&
                        pastKey.dataType == DataType::FLOAT16 &&
                        qkvConvInput.dataDevice == DataDevice::CUDA &&
                        qkvConvInput.dataType == DataType::FLOAT16 &&
                        requireLocal(weight[conv1dWeightName], conv1dWeightName)->dataType == DataType::FLOAT32;
                    if (!canTryFusedDecodeConvSilu) {
                        ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                    }
                    if (canTryFusedDecodeConvSilu) {
                        fusedDecodeConvSilu = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                            pastKey, qkvConvInput,
                            *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                            *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                            convOutput);
                    }
                    if (!fusedDecodeConvSilu) {
                        if (canTryFusedDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                        }
                        Qwen35CudaConv1DPerChannel(cudaRunner, pastKey,
                                                   *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                                   *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                                   pastKey.dims[1],
                                                   requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                                   4, 1, 0, convOutput);
                        Qwen35CudaSilu(cudaRunner, convOutput, convOutput);
                    }
                } else if (batch > 1 && all1) {
                    std::vector<Data*> linearConvCaches(batch);
                    for (int bidx = 0; bidx < batch; bidx++) {
                        linearConvCaches[bidx] = pastKeyValues[bidx * block_cnt + i].first;
                    }
                    bool directBatchDecodeConvSilu =
                        FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(
                            linearConvCaches, qkvConvInput,
                            *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                            *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                            convOutput);
                    if (!directBatchDecodeConvSilu) {
                        Data batchConvCache;
                        CatBatchFirstDim(linearConvCaches, batchConvCache);
                        bool fusedBatchDecodeConvSilu =
                            FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                                batchConvCache, qkvConvInput,
                                *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                convOutput);
                        if (!fusedBatchDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(batchConvCache, qkvConvInput);
                            Qwen35CudaConv1DPerChannel(cudaRunner, batchConvCache,
                                                       *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                                       *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                                       batchConvCache.dims[1],
                                                       requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                                       4, 1, 0, convOutput);
                            Qwen35CudaSilu(cudaRunner, convOutput, convOutput);
                        }
                        SplitBatchFirstDim(batchConvCache, linearConvCaches);
                    }
                    for (int bidx = 0; bidx < batch; bidx++) {
                        Qwen35PrepareLinearAttentionCache(*linearConvCaches[bidx], computeType);
                    }
                } else if (speculativeCaptureFirstTokenLinearState &&
                           batch == 1 && pastKey.dims.size() == 3 &&
                           pastKey.dims[0] == 1 && pastKey.dims[1] == qkvConvInput.dims[1] &&
                           pastKey.dims[2] == 4 && qkvConvInput.dims[2] > 0) {
                    bool fusedMultiTokenConvSilu = false;
                    bool canTryFusedMultiTokenConvSilu =
                        qkvConvInput.dims[2] >= 1 &&
                        qkvConvInput.dims[2] <= QWEN35_MTP_FAST_SEQ_MAX &&
                        pastKey.dataDevice == DataDevice::CUDA &&
                        pastKey.dataType == DataType::FLOAT16 &&
                        qkvConvInput.dataDevice == DataDevice::CUDA &&
                        qkvConvInput.dataType == DataType::FLOAT16 &&
                        requireLocal(weight[conv1dWeightName], conv1dWeightName)->dataType == DataType::FLOAT32;
                    if (canTryFusedMultiTokenConvSilu) {
                        int convCaptureTokens = std::min(
                            std::min(seqlen, linearStateCaptureSlots),
                            QWEN35_MTP_PREFIX_SNAPSHOT_MAX);
                        Data *tokenConvCaches[QWEN35_MTP_PREFIX_SNAPSHOT_MAX] = {};
                        for (int tokenIndex = 0; tokenIndex < convCaptureTokens; tokenIndex++) {
                            tokenConvCaches[tokenIndex] = getLinearCaptureSlot(tokenIndex, true);
                        }
                        fusedMultiTokenConvSilu =
                            FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                                pastKey, qkvConvInput,
                                *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                convOutput, tokenConvCaches, convCaptureTokens);
                        if (fusedMultiTokenConvSilu) {
                            for (int tokenIndex = 0; tokenIndex < convCaptureTokens; tokenIndex++) {
                                if (tokenConvCaches[tokenIndex] != nullptr) {
                                    tokenConvCaches[tokenIndex]->cacheUid = pastKey.cacheUid;
                                    Qwen35PrepareLinearAttentionCache(*tokenConvCaches[tokenIndex], computeType);
                                    markLinearCaptured(tokenIndex, 1);
                                }
                            }
                        }
                    }
                    if (!fusedMultiTokenConvSilu) {
                        Qwen3CudaCat(cudaRunner, pastKey, qkvConvInput, -1, convInputWithCache);
                        Qwen35CudaConv1DPerChannel(cudaRunner, convInputWithCache,
                                                   *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                                   *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                                   convInputWithCache.dims[1],
                                                   requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                                   4, 1, 0, conv);
                        Qwen3CudaSplit(cudaRunner, conv, -1, 1, 1 + seqlen, convOutput);
                        Qwen3CudaSplit(cudaRunner, convInputWithCache, -1,
                                       convInputWithCache.dims.back() - 4,
                                       convInputWithCache.dims.back(), pastKey);
                        int captureTokens = std::min(seqlen, linearStateCaptureSlots);
                        for (int tokenIndex = 0; tokenIndex < captureTokens; tokenIndex++) {
                            Data *tokenConvCache = getLinearCaptureSlot(tokenIndex, true);
                            int cacheBegin = tokenIndex + 1;
                            int cacheEnd = cacheBegin + 4;
                            if (tokenConvCache != nullptr &&
                                cacheEnd <= convInputWithCache.dims.back()) {
                                Qwen3CudaSplit(cudaRunner, convInputWithCache, -1,
                                               cacheBegin, cacheEnd,
                                               *tokenConvCache);
                                tokenConvCache->cacheUid = pastKey.cacheUid;
                                Qwen35PrepareLinearAttentionCache(*tokenConvCache, computeType);
                                markLinearCaptured(tokenIndex, 1);
                            }
                        }
                        pastKey.expansionDims = pastKey.dims;
                        Qwen35CudaSilu(cudaRunner, convOutput, convOutput);
                    }
                } else {
                    if (qkvConvInput.dims.back() >= 4) {
                        Qwen3CudaSplit(cudaRunner, qkvConvInput, -1,
                                       qkvConvInput.dims.back() - 4,
                                       qkvConvInput.dims.back(), pastKey);
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        Data temp;
                        Qwen35CudaMul(cudaRunner, qkvConvInput, 1.0f, temp);
                        Qwen35CudaRepeat(cudaRunner, temp, -1, 4, qkvConvInput);
                    }
                    Qwen35CudaConv1DPerChannel(cudaRunner, qkvConvInput,
                                               *requireLocal(weight[conv1dWeightName], conv1dWeightName),
                                               *requireLocal(GetThreadTensorParallelBias(conv1dBiasName), conv1dBiasName),
                                               qkvConvInput.dims[1],
                                               requireLocal(weight[conv1dWeightName], conv1dWeightName)->dims[0],
                                               4, 1, 3, conv);
                    Qwen3CudaSplit(cudaRunner, conv, -1, 0, seqlen, convOutput);
                    Qwen35CudaSilu(cudaRunner, convOutput, convOutput);
                }

                if (batch == 1 && all1 && pastKey.dims.size() > 0) {
                    SwapSingleTokenSeqHeadByReshape(convOutput);
                } else if (batch > 1 && all1) {
                    convOutput.Reshape({1, batch, convOutput.dims[1]});
                } else {
                    Qwen3CudaPermuteSelf(cudaRunner, convOutput, {0, 2, 1});
                }
                bool convQkvSplitReady = false;
                auto ensureConvQkvSplit = [&]() {
                    if (convQkvSplitReady) {
                        return;
                    }
                    Qwen3CudaSplit(cudaRunner, convOutput, -1, 0, localKd, q);
                    Qwen3CudaSplit(cudaRunner, convOutput, -1, localKd, localKd * 2, k);
                    Qwen3CudaSplit(cudaRunner, convOutput, -1, localKd * 2, localKd * 2 + localVd, v);
                    q.Reshape({q.dims[0], q.dims[1], localKeyHeads, head_k_dim});
                    k.Reshape({k.dims[0], k.dims[1], localKeyHeads, head_k_dim});
                    v.Reshape({v.dims[0], v.dims[1], localValueHeads, head_v_dim});
                    convQkvSplitReady = true;
                };

                bool fusedBatchRecurrentFromConvBa = false;
#ifdef USE_CUDA
                std::vector<Data*> fusedRecurrentStates;
                if (batch > 1 && all1) {
                    Data *linearNormWeight = requireLocal(inv_scale_data, "linear_attn.inv_scale");
                    Data *aLog = requireLocal(weight[aLogName], aLogName);
                    Data *dtBias = requireLocal(weight[dtBiasName], dtBiasName);
                    fusedRecurrentStates.resize(batch);
                    bool canUseFusedBatchRecurrent =
                        convOutput.dataDevice == DataDevice::CUDA &&
                        convOutput.dataType == DataType::FLOAT16 &&
                        baMerged.dataDevice == DataDevice::CUDA &&
                        baMerged.dataType == DataType::FLOAT16 &&
                        baMerged.dims.size() > 0 &&
                        baMerged.dims.back() == localValueHeads * 2 &&
                        linearNormWeight->dataDevice == DataDevice::CUDA &&
                        linearNormWeight->dataType == DataType::FLOAT32 &&
                        linearNormWeight->dims.size() == 1 &&
                        linearNormWeight->dims[0] == head_k_dim &&
                        aLog->dataDevice == DataDevice::CUDA &&
                        aLog->dataType == DataType::FLOAT32 &&
                        aLog->dims.size() == 1 &&
                        aLog->dims[0] == localValueHeads &&
                        dtBias->dataDevice == DataDevice::CUDA &&
                        dtBias->dataType == DataType::FLOAT32 &&
                        dtBias->dims.size() == 1 &&
                        dtBias->dims[0] == localValueHeads &&
                        localKeyHeads > 0 &&
                        localValueHeads % localKeyHeads == 0;
                    bool recurrentStatesTransposed = false;
                    for (int rb = 0; rb < batch; rb++) {
                        fusedRecurrentStates[rb] = pastKeyValues[rb * block_cnt + i].second;
                        Data *state = fusedRecurrentStates[rb];
                        if (rb == 0 && state != nullptr) {
                            recurrentStatesTransposed = state->isLinearAttentionTransposed;
                        }
                        canUseFusedBatchRecurrent &= state != nullptr &&
                                                     state->dataDevice == DataDevice::CUDA &&
                                                     state->dataType == DataType::FLOAT16 &&
                                                     state->cudaData != nullptr &&
                                                     state->dims.size() == 4 &&
                                                     state->dims[0] == 1 &&
                                                     state->dims[1] == localValueHeads &&
                                                     state->dims[2] == head_k_dim &&
                                                     state->dims[3] == head_v_dim &&
                                                     state->isLinearAttentionTransposed == recurrentStatesTransposed;
                    }
                    canUseFusedBatchRecurrent &= !recurrentStatesTransposed || head_k_dim == 128;
                    if (canUseFusedBatchRecurrent) {
                        float recurrentQScale = 1.0f / std::sqrt((float)head_k_dim);
                        if (recurrentStatesTransposed) {
                            FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposed(
                                convOutput, baMerged, *linearNormWeight, *aLog, *dtBias,
                                fusedRecurrentStates, coreAttnOut,
                                localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                rms_norm_eps, recurrentQScale);
                        } else {
                            FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
                                convOutput, baMerged, *linearNormWeight, *aLog, *dtBias,
                                fusedRecurrentStates, coreAttnOut,
                                localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                rms_norm_eps, recurrentQScale);
                        }
                        fusedBatchRecurrentFromConvBa = true;
                        for (int rb = 0; rb < batch; rb++) {
                            Qwen35PrepareLinearAttentionCache(*fusedRecurrentStates[rb], computeType);
                        }
                        coreAttnOut.Reshape({1, batch, coreAttnOut.dims[1], coreAttnOut.dims[3]});
                    }
                }
#endif

                bool isSingleTokenLinearDecode = bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0;
                bool canTryTwoTokenLinearDecode =
                    speculativeCaptureFirstTokenLinearState &&
                    bsz == 1 && seqlen == 2 && batch == 1 &&
                    pastKey.dims.size() == 3 && pastKey.dims[0] == 1 &&
                    pastKey.dims[2] == 4 && pastValue.dims.size() > 0;
                bool canTrySmallSpeculativeLinearDecode =
                    speculativeCaptureFirstTokenLinearState &&
                    bsz == 1 && seqlen > 2 &&
                    seqlen <= QWEN35_MTP_FAST_SEQ_MAX && batch == 1 &&
                    pastKey.dims.size() == 3 && pastKey.dims[0] == 1 &&
                    pastKey.dims[2] == 4 && pastValue.dims.size() > 0;
                auto runChunkLinearAttention = [&]() {
                    ensureConvQkvSplit();
                    ensureBaSplit();
                    if (batch == 1 && pastValue.dims.size() > 0) {
                        Qwen35EnsureCudaLinearAttnStateKVLayout(pastValue);
                    }
                    Qwen35CudaSigmoidMambaSoftplus(cudaRunner, b, a,
                                                   *requireLocal(weight[aLogName], aLogName),
                                                   *requireLocal(weight[dtBiasName], dtBiasName),
                                                   g);
                    if (num_v_heads / num_k_heads > 1) {
                        Qwen35CudaMul(cudaRunner, q, 1.0f, qRepeat);
                        Qwen35CudaMul(cudaRunner, k, 1.0f, kRepeat);
                        qRepeat.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                        kRepeat.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});
                        Qwen35CudaRepeat(cudaRunner, qRepeat, 3, num_v_heads / num_k_heads, q);
                        Qwen35CudaRepeat(cudaRunner, kRepeat, 3, num_v_heads / num_k_heads, k);
                        q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                        k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                    }

                    Qwen3CudaRMSNorm(cudaRunner, q, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                     rms_norm_eps, q);
                    Qwen3CudaRMSNorm(cudaRunner, k, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                     rms_norm_eps, k);
                    Qwen3CudaPermuteSelf(cudaRunner, q, {0, 2, 1, 3});
                    Qwen3CudaPermuteSelf(cudaRunner, k, {0, 2, 1, 3});
                    Qwen3CudaPermuteSelf(cudaRunner, v, {0, 2, 1, 3});
                    Qwen3CudaPermuteSelf(cudaRunner, b, {0, 2, 1});
                    Qwen3CudaPermuteSelf(cudaRunner, g, {0, 2, 1});

                    int keyBatchSize = k.dims[0];
                    int keySequenceLength = k.dims[1];
                    int keyKHeadDim = k.dims[3];
                    int chunkSize = 64;
                    int vHeadDimLocal = v.dims.back();
                    int seq = k.dims[2];
                    int padSize = (chunkSize - seq % chunkSize) % chunkSize;

                    Data *pkk = &k, *pvv = &v, *pbb = &b, *pgg = &g;
                    if (padSize > 0) {
                        Qwen35CudaPad(cudaRunner, q, 2, padSize, qTemp);
                        Qwen35CudaPad(cudaRunner, k, 2, padSize, kkPad);
                        Qwen35CudaPad(cudaRunner, v, 2, padSize, vvPad);
                        Qwen35CudaPad(cudaRunner, b, 2, padSize, bbPad);
                        Qwen35CudaPad(cudaRunner, g, 2, padSize, ggPad);
                        Qwen35CudaMul(cudaRunner, qTemp, 1.0f / std::sqrt((float)head_k_dim), qq);
                        pkk = &kkPad;
                        pvv = &vvPad;
                        pbb = &bbPad;
                        pgg = &ggPad;
                    } else {
                        Qwen35CudaMul(cudaRunner, q, 1.0f / std::sqrt((float)head_k_dim), qq);
                    }

                    pbb->Resize({pbb->dims[0], pbb->dims[1], pbb->dims[2], 1});
                    Qwen35CudaMul(cudaRunner, *pkk, 1.0f, kBeta);
                    Qwen35CudaMul(cudaRunner, *pvv, 1.0f, vBeta);
                    Qwen35CudaMulTo(cudaRunner, kBeta, *pbb);
                    Qwen35CudaMulTo(cudaRunner, vBeta, *pbb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunkSize, qq.dims.back()});
                    pkk->Reshape({pkk->dims[0], pkk->dims[1], -1, chunkSize, pkk->dims.back()});
                    kBeta.Reshape({kBeta.dims[0], kBeta.dims[1], -1, chunkSize, kBeta.dims.back()});
                    vBeta.Reshape({vBeta.dims[0], vBeta.dims[1], -1, chunkSize, vBeta.dims.back()});
                    pgg->Reshape({pgg->dims[0], pgg->dims[1], -1, chunkSize});

                    Qwen35CudaCumSumLastDim(cudaRunner, *pgg);
                    Qwen35CudaMakeDecayMask(cudaRunner, *pgg, decayMask);
                    Qwen35CudaMatMulTransB(cudaRunner, kBeta, *pkk, at);
                    Qwen35CudaMul(cudaRunner, at, -1.0f, attn);
                    Qwen35CudaMulTo(cudaRunner, attn, decayMask);
                    Qwen35CudaCausalMask(cudaRunner, attn, 0, 0.0f);
                    Qwen35CudaTransferAttn(cudaRunner, attn);
                    Qwen35CudaMatMul(cudaRunner, attn, vBeta, vvPad);
                    Qwen35CudaExp(cudaRunner, *pgg, gExp);
                    Qwen35CudaMulTo(cudaRunner, kBeta, gExp);
                    Qwen35CudaMatMul(cudaRunner, attn, kBeta, kCumdecay);
                    Qwen35CudaMatMulTransB(cudaRunner, qq, *pkk, attn);
                    Qwen35CudaMulTo(cudaRunner, attn, decayMask);
                    Qwen35CudaCausalMask(cudaRunner, attn, 1, 0.0f);

                    if (pastValue.dims.size() == 0) {
                        pastValue.dataDevice = DataDevice::CUDA;
                        pastValue.dataDeviceIds = {gpuId};
                        pastValue.Resize({keyBatchSize, keySequenceLength, keyKHeadDim, vHeadDimLocal});
                        pastValue.Allocate(0.0f);
                    }
                    Qwen35CudaChunkGatedDeltaRulePrefill(cudaRunner, qq, *pkk, vvPad, *pgg,
                                                         attn, kCumdecay,
                                                         pastValue, coreAttnOut);
                    coreAttnOut.Reshape({coreAttnOut.dims[0], coreAttnOut.dims[1],
                                         -1, coreAttnOut.dims.back()});
                    if (padSize > 0) {
                        Qwen3CudaSplit(cudaRunner, coreAttnOut, 2, 0, seq, coreTemp);
                        Qwen3CudaPermuteSelf(cudaRunner, coreTemp, {0, 2, 1, 3});
                        Qwen35CudaMul(cudaRunner, coreTemp, 1.0f, coreAttnOut);
                    } else {
                        Qwen3CudaPermuteSelf(cudaRunner, coreAttnOut, {0, 2, 1, 3});
                    }
                };
                if (fusedBatchRecurrentFromConvBa) {
                    // coreAttnOut is already produced by the fused batch recurrent path.
                } else if (isSingleTokenLinearDecode) {
                    ensureConvQkvSplit();
                    ensureBaSplit();
                    bool fusedSingleDecode = Qwen35TryCudaLinearAttnSingleDecodeNormBaRecurrent(
                        q, k, v, a, b,
                        *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                        *requireLocal(weight[aLogName], aLogName),
                        *requireLocal(weight[dtBiasName], dtBiasName),
                        rms_norm_eps, pastValue, coreAttnOut
                    );
                    if (!fusedSingleDecode) {
                        Qwen35CudaSigmoidMambaSoftplus(cudaRunner, b, a,
                                                       *requireLocal(weight[aLogName], aLogName),
                                                       *requireLocal(weight[dtBiasName], dtBiasName),
                                                       g);
                        fusedSingleDecode = Qwen35TryCudaLinearAttnSingleDecodeNormRecurrent(
                            q, k, v, g, b,
                            *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                            rms_norm_eps, pastValue, coreAttnOut
                        );
                    }
                    if (!fusedSingleDecode) {
                        Qwen3CudaRMSNorm(cudaRunner, q, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                         rms_norm_eps, q);
                        Qwen3CudaRMSNorm(cudaRunner, k, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                         rms_norm_eps, k);
                        SwapSingleTokenSeqHeadByReshape(q);
                        SwapSingleTokenSeqHeadByReshape(k);
                        SwapSingleTokenSeqHeadByReshape(v);
                        SwapSingleTokenSeqHeadByReshape(b);
                        SwapSingleTokenSeqHeadByReshape(g);
                        float recurrentQScale = 1.0f / std::sqrt((float)head_k_dim);
                        Qwen35CudaRecurrentGatedDeltaRule(cudaRunner, q, k, v, g, b,
                                                          pastValue, coreAttnOut, recurrentQScale);
                        SwapSingleTokenSeqHeadByReshape(coreAttnOut);
                    }
                } else if (canTrySmallSpeculativeLinearDecode) {
                    bool fusedSmallDecode = false;
                    bool needLinearStateSnapshots =
                        speculativeLinearStateCaptureSlots > 0;
                    if (keepCombinedBaForSmallDecode && !baMerged.dims.empty() &&
                        convOutput.dims.size() >= 3 && convOutput.dims[0] == 1 &&
                        convOutput.dims[1] == seqlen && convOutput.dims.back() == localQkvDim &&
                        baMerged.dims.size() >= 3 && baMerged.dims[0] == 1 &&
                        baMerged.dims[1] == seqlen &&
                        baMerged.dims.back() == localValueHeads * 2 &&
                        Qwen35EnsureCudaLinearAttnStateTransposed(pastValue)) {
                        if (Qwen35MtpFusedLinearSeqEnabled()) {
                            if (!needLinearStateSnapshots) {
                                fusedSmallDecode =
                                    FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
                                        convOutput, baMerged,
                                        *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                        *requireLocal(weight[aLogName], aLogName),
                                        *requireLocal(weight[dtBiasName], dtBiasName),
                                        pastValue, coreAttnOut,
                                        localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                        rms_norm_eps, 1.0f / std::sqrt((float)head_k_dim));
                            } else {
                                // 融合序列 kernel 内部在处理完第 t 个 token 后直接把递归状态
                                // 写入快照 slot, 替代逐行前向 + 全量 CopyFrom 的慢速路径。
                                int valueCaptureTokens = std::min(
                                    std::min(seqlen, linearStateCaptureSlots),
                                    QWEN35_MTP_PREFIX_SNAPSHOT_MAX);
                                Data *tokenValueStates[QWEN35_MTP_PREFIX_SNAPSHOT_MAX] = {};
                                for (int tokenIndex = 0; tokenIndex < valueCaptureTokens; tokenIndex++) {
                                    tokenValueStates[tokenIndex] = getLinearCaptureSlot(tokenIndex, false);
                                }
                                fusedSmallDecode =
                                    FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                                        convOutput, baMerged,
                                        *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                        *requireLocal(weight[aLogName], aLogName),
                                        *requireLocal(weight[dtBiasName], dtBiasName),
                                        pastValue, coreAttnOut,
                                        tokenValueStates, valueCaptureTokens,
                                        localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                        rms_norm_eps, 1.0f / std::sqrt((float)head_k_dim));
                                if (fusedSmallDecode) {
                                    for (int tokenIndex = 0; tokenIndex < valueCaptureTokens; tokenIndex++) {
                                        if (tokenValueStates[tokenIndex] != nullptr) {
                                            tokenValueStates[tokenIndex]->cacheUid = pastValue.cacheUid;
                                            Qwen35PrepareLinearAttentionCache(*tokenValueStates[tokenIndex], computeType);
                                            markLinearCaptured(tokenIndex, 2);
                                        }
                                    }
                                }
                            }
                        }
                        if (!fusedSmallDecode) {
                            Data fusedRows;
                            fusedSmallDecode = true;
                            for (int row = 0; row < seqlen; row++) {
                                Data convRow, baRow, rowOut;
                                Qwen3CudaSplit(cudaRunner, convOutput, 1, row, row + 1, convRow);
                                Qwen3CudaSplit(cudaRunner, baMerged, 1, row, row + 1, baRow);
                                bool rowOk =
                                    FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                                        convRow, baRow,
                                        *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                        *requireLocal(weight[aLogName], aLogName),
                                        *requireLocal(weight[dtBiasName], dtBiasName),
                                        pastValue, rowOut,
                                        localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                        rms_norm_eps, 1.0f / std::sqrt((float)head_k_dim));
                                if (!rowOk) {
                                    if (row == 0) {
                                        fusedSmallDecode = false;
                                        Qwen35EnsureCudaLinearAttnStateKVLayout(pastValue);
                                        break;
                                    }
                                    AssertInFastLLM(false,
                                                    "Qwen3.5 small speculative linear decode failed after updating state.\n");
                                }
                                Data *tokenValueState =
                                    getLinearCaptureSlot(row, false);
                                if (tokenValueState != nullptr) {
                                    tokenValueState->CopyFrom(pastValue);
                                    markLinearCaptured(row, 2);
                                }
                                SwapSingleTokenSeqHeadByReshape(rowOut);
                                if (row == 0) {
                                    fusedRows.CopyFrom(rowOut);
                                } else {
                                    Data catRows;
                                    Qwen3CudaCat(cudaRunner, fusedRows, rowOut, 1, catRows);
                                    fusedRows.CopyFrom(catRows);
                                }
                            }
                            if (fusedSmallDecode) {
                                coreAttnOut.CopyFrom(fusedRows);
                            }
                        }
                    }
                    if (!fusedSmallDecode) {
                        runChunkLinearAttention();
                    }
                } else if (canTryTwoTokenLinearDecode) {
                    bool fusedTwoTokenDecode = false;
                    bool fusedConvBaTwoTokenDecode = false;
                    if (keepCombinedBaForTwoToken && !baMerged.dims.empty() &&
                        convOutput.dims.size() >= 3 && convOutput.dims[0] == 1 &&
                        convOutput.dims[1] == 2 && convOutput.dims.back() == localQkvDim &&
                        baMerged.dims.size() >= 3 && baMerged.dims[0] == 1 &&
                        baMerged.dims[1] == 2 && baMerged.dims.back() == localValueHeads * 2 &&
                        Qwen35EnsureCudaLinearAttnStateTransposed(pastValue)) {
                        Qwen3CudaSplit(cudaRunner, convOutput, 1, 0, 1, convRow0);
                        Qwen3CudaSplit(cudaRunner, baMerged, 1, 0, 1, baRow0);
                        bool fusedFirstToken =
                            FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                                convRow0, baRow0,
                                *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                *requireLocal(weight[aLogName], aLogName),
                                *requireLocal(weight[dtBiasName], dtBiasName),
                                pastValue, coreAttnOut0,
                                localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                rms_norm_eps, 1.0f / std::sqrt((float)head_k_dim));
                        if (fusedFirstToken) {
                            Data *firstTokenValueState =
                                getFirstTokenLinearCaptureSlot(false);
                            if (firstTokenValueState != nullptr) {
                                firstTokenValueState->CopyFrom(pastValue);
                                markFirstTokenLinearCaptured(2);
                            }
                            SwapSingleTokenSeqHeadByReshape(coreAttnOut0);
                            Qwen3CudaSplit(cudaRunner, convOutput, 1, 1, 2, convRow1);
                            Qwen3CudaSplit(cudaRunner, baMerged, 1, 1, 2, baRow1);
                            bool fusedSecondToken =
                                FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                                    convRow1, baRow1,
                                    *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                    *requireLocal(weight[aLogName], aLogName),
                                    *requireLocal(weight[dtBiasName], dtBiasName),
                                    pastValue, coreAttnOut1,
                                    localKeyHeads, localValueHeads, head_k_dim, head_v_dim,
                                    rms_norm_eps, 1.0f / std::sqrt((float)head_k_dim));
                            AssertInFastLLM(fusedSecondToken,
                                            "Qwen3.5 two-token conv+ba linear decode fused first token but failed second token.\n");
                            SwapSingleTokenSeqHeadByReshape(coreAttnOut1);
                            Qwen3CudaCat(cudaRunner, coreAttnOut0, coreAttnOut1, 1, coreAttnOut);
                            fusedTwoTokenDecode = true;
                            fusedConvBaTwoTokenDecode = true;
                        }
                    }
                    if (!fusedTwoTokenDecode) {
                        ensureConvQkvSplit();
                        ensureBaSplit();
                        Qwen3CudaSplit(cudaRunner, q, 1, 0, 1, q0);
                        Qwen3CudaSplit(cudaRunner, k, 1, 0, 1, k0);
                        Qwen3CudaSplit(cudaRunner, v, 1, 0, 1, v0);
                        Qwen3CudaSplit(cudaRunner, a, 1, 0, 1, a0);
                        Qwen3CudaSplit(cudaRunner, b, 1, 0, 1, b0);
                        fusedTwoTokenDecode = Qwen35TryCudaLinearAttnSingleDecodeNormBaRecurrent(
                            q0, k0, v0, a0, b0,
                            *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                            *requireLocal(weight[aLogName], aLogName),
                            *requireLocal(weight[dtBiasName], dtBiasName),
                            rms_norm_eps, pastValue, coreAttnOut0
                        );
                    }
                    if (fusedTwoTokenDecode && !fusedConvBaTwoTokenDecode) {
                        Data *firstTokenValueState =
                            getFirstTokenLinearCaptureSlot(false);
                        if (firstTokenValueState != nullptr) {
                            firstTokenValueState->CopyFrom(pastValue);
                            markFirstTokenLinearCaptured(2);
                        }
                        Qwen3CudaSplit(cudaRunner, q, 1, 1, 2, q1);
                        Qwen3CudaSplit(cudaRunner, k, 1, 1, 2, k1);
                        Qwen3CudaSplit(cudaRunner, v, 1, 1, 2, v1);
                        Qwen3CudaSplit(cudaRunner, a, 1, 1, 2, a1);
                        Qwen3CudaSplit(cudaRunner, b, 1, 1, 2, b1);
                        bool fusedSecondToken = Qwen35TryCudaLinearAttnSingleDecodeNormBaRecurrent(
                            q1, k1, v1, a1, b1,
                            *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                            *requireLocal(weight[aLogName], aLogName),
                            *requireLocal(weight[dtBiasName], dtBiasName),
                            rms_norm_eps, pastValue, coreAttnOut1
                        );
                        AssertInFastLLM(fusedSecondToken,
                                        "Qwen3.5 two-token linear decode fused first token but failed second token.\n");
                        Qwen3CudaCat(cudaRunner, coreAttnOut0, coreAttnOut1, 1, coreAttnOut);
                    } else if (!fusedTwoTokenDecode) {
                        runChunkLinearAttention();
                    }
                } else if (batch > 1 && all1) {
                    ensureConvQkvSplit();
                    ensureBaSplit();
                    Qwen35CudaSigmoidMambaSoftplus(cudaRunner, b, a,
                                                   *requireLocal(weight[aLogName], aLogName),
                                                   *requireLocal(weight[dtBiasName], dtBiasName),
                                                   g);
                    Qwen3CudaRMSNorm(cudaRunner, q, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                     rms_norm_eps, q);
                    Qwen3CudaRMSNorm(cudaRunner, k, *requireLocal(inv_scale_data, "linear_attn.inv_scale"),
                                     rms_norm_eps, k);
                    q.Reshape({batch, localKeyHeads, 1, head_k_dim});
                    k.Reshape({batch, localKeyHeads, 1, head_k_dim});
                    v.Reshape({batch, localValueHeads, 1, head_v_dim});
                    b.Reshape({batch, localValueHeads, 1});
                    g.Reshape({batch, localValueHeads, 1});

                    std::vector<Data*> recurrentStates(batch);
                    for (int rb = 0; rb < batch; rb++) {
                        recurrentStates[rb] = pastKeyValues[rb * block_cnt + i].second;
                    }
                    Data batchRecurrentState;
                    CatBatchFirstDim(recurrentStates, batchRecurrentState);
                    float recurrentQScale = 1.0f / std::sqrt((float)head_k_dim);
                    Qwen35CudaRecurrentGatedDeltaRule(cudaRunner, q, k, v, g, b,
                                                      batchRecurrentState, coreAttnOut,
                                                      recurrentQScale);
                    SplitBatchFirstDim(batchRecurrentState, recurrentStates);
                    for (int rb = 0; rb < batch; rb++) {
                        Qwen35PrepareLinearAttentionCache(*recurrentStates[rb], computeType);
                    }
                    coreAttnOut.Reshape({1, batch, coreAttnOut.dims[1], coreAttnOut.dims[3]});
                } else {
                    runChunkLinearAttention();
                }
                std::vector<int> zShape = z.dims;
                coreAttnOut.Reshape({-1, coreAttnOut.dims.back()});
                z.Reshape({-1, z.dims.back()});
                bool fusedPostLinearAttn = Qwen35TryCudaRMSNormSiluMul(
                    coreAttnOut,
                    *requireLocal(weight[outNormWeightName], outNormWeightName),
                    z, coreAttnOut, rms_norm_eps);
                if (!fusedPostLinearAttn) {
                    Qwen3CudaRMSNorm(cudaRunner, coreAttnOut,
                                     *requireLocal(weight[outNormWeightName], outNormWeightName),
                                     rms_norm_eps, coreAttnOut);
                    Qwen35CudaSilu(cudaRunner, z, z);
                    if (z.dataType != coreAttnOut.dataType) {
                        Qwen3CudaToDataType(cudaRunner, z, coreAttnOut.dataType);
                    }
                    Qwen35CudaMulTo(cudaRunner, coreAttnOut, z);
                }
                coreAttnOut.Reshape({zShape[0], zShape[1], localVd});
                Qwen3CudaLinearResidualReduce(
                    cudaRunner, coreAttnOut,
                    *requireLocal(weight[outProjWeightName], outProjWeightName),
                    *requireLocal(GetThreadTensorParallelBias(outProjWeightName + ".tp_bias"),
                                  outProjWeightName + ".tp_bias"),
                    attenLastOutput, hiddenStates,
                    tensorParallel, firstTensorParallelRank, gpuId);
            }
            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[postRmsName], postRmsName),
                             rms_norm_eps, attenInput);
            bool hasDenseMlp = weight.weight.find(swigluWeightName) != weight.weight.end() &&
                               weight.weight.find(downWeightName) != weight.weight.end();
            if (hasDenseMlp) {
                Data &gateUpWeight = *requireLocal(weight[swigluWeightName], swigluWeightName);
                Data &gateUpBias = *requireLocal(GetThreadTensorParallelBias(swigluWeightName + ".tp_bias"),
                                                 swigluWeightName + ".tp_bias");
                Data &downWeight = *requireLocal(weight[downWeightName], downWeightName);
                Data &downBias = *requireLocal(GetThreadTensorParallelBias(downBiasName), downBiasName);
                if (!Qwen3CudaTrySwigluLinearResidualReduce(
                        cudaRunner, attenInput, gateUpWeight, gateUpBias,
                        downWeight, downBias, gateupResult, swigluResult, mlpPart,
                        hiddenStates, tensorParallel)) {
                    Qwen3CudaLinearSwiglu(cudaRunner, attenInput,
                                          gateUpWeight, gateUpBias,
                                          gateupResult, swigluResult);
                    Qwen3CudaLinearResidualReduce(
                        cudaRunner, swigluResult,
                        downWeight, downBias,
                        mlpPart, hiddenStates,
                        tensorParallel, firstTensorParallelRank, gpuId);
                }
                continue;
            }

            std::string gateWeightName = prefix + "mlp.gate.weight";
            std::string gateBiasName = prefix + "mlp.gate.e_score_correction_bias";
            std::string sharedGateupWeightName = prefix + "mlp.shared_expert.gateup_proj.weight";
            std::string sharedDownWeightName = prefix + "mlp.shared_expert.down_proj.weight";
            std::string sharedExpertGateWeightName = prefix + "mlp.shared_expert_gate.weight";
            AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                            "Qwen3.5 ForwardGPU layer has neither dense MLP nor router gate weight.\n");

            bool sharedExpertPending = false;
            if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                AssertInFastLLM(weight.weight.find(sharedGateupWeightName) != weight.weight.end(),
                                "Qwen3.5 ForwardGPU requires merged shared expert gateup weight.\n");
                Qwen3CudaLinearSwiglu(cudaRunner, attenInput,
                                      *requireLocal(weight[sharedGateupWeightName], sharedGateupWeightName),
                                      *requireLocal(GetThreadTensorParallelBias(sharedGateupWeightName + ".tp_bias"),
                                                    sharedGateupWeightName + ".tp_bias"),
                                      gateupResult, swigluResult);
                Qwen3CudaLinear(cudaRunner, swigluResult,
                                *requireLocal(weight[sharedDownWeightName], sharedDownWeightName),
                                *requireLocal(GetThreadTensorParallelBias(sharedDownWeightName + ".tp_bias"),
                                              sharedDownWeightName + ".tp_bias"),
                                sharedOutput);
                if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                    Qwen3CudaLinear(cudaRunner, attenInput,
                                    *requireLocal(weight[sharedExpertGateWeightName], sharedExpertGateWeightName),
                                    *GetEmptyData(), sharedGate);
                    Qwen35CudaSigmoid(cudaRunner, sharedGate, sharedGate);
                    if (sharedGate.dataType != sharedOutput.dataType) {
                        Qwen3CudaToDataType(cudaRunner, sharedGate, sharedOutput.dataType);
                    }
                    Qwen35CudaMulTo(cudaRunner, sharedOutput, sharedGate);
                }
                // 共享专家输出延后与路由专家输出本地相加后一次 allReduce，每层省一次集合通信。
                sharedExpertPending = true;
            }

            int localBatch = attenInput.dims[0];
            int localLen = attenInput.dims[1];
            attenInput.Reshape({localBatch * localLen, attenInput.dims[2]});
            Qwen3CudaLinear(cudaRunner, attenInput,
                            *requireLocal(weight[gateWeightName], gateWeightName),
                            *GetEmptyData(), routerLogits, true);
            Qwen3CudaConvertToDataType(cudaRunner, routerLogits, routerLogitsTemp, DataType::FLOAT32);
            Qwen3CudaSoftmax(cudaRunner, routerLogitsTemp, routerLogitsTemp, -1);
            Data *localGateBias = nullptr;
            if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
            }
            Qwen3CudaSelectExpert(cudaRunner, routerLogitsTemp, expertIndex, expertScore,
                                  this->num_experts_per_tok, this->norm_topk_prob,
                                  this->routed_scaling_factor, localGateBias);

            bool layerMappedNonCudaMoe = Qwen35LayerUsesMappedNonCudaMoe(this, i);
            if (layerMappedNonCudaMoe) {
                if (!tensorParallel || firstTensorParallelRank) {
                    std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(i);
                    Qwen35ResetCpuScratch(moeFinal);
                    FastllmCudaSetDevice(gpuId);
                    Qwen35ScopedGenericExecutor scopedExecutor(selectedMoeDevice);
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
                    Qwen35ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
            } else if (HasFusedMoeWeights(i)) {
                Data *localGate = GetFusedMoeWeightForDevice(moeGate3DWeights[i], gpuId);
                Data *localUp = GetFusedMoeWeightForDevice(moeUp3DWeights[i], gpuId);
                Data *localDown = GetFusedMoeWeightForDevice(moeDown3DWeights[i], gpuId);
                if (Qwen35HasLocalFusedMoeShard(localGate, localUp, localDown)) {
                    Qwen35CudaFusedMOE(cudaRunner, attenInput, expertIndex, expertScore,
                                       *localGate, *localUp, *localDown,
                                       w1, moeFinal, i);
                } else {
                    Qwen35ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
            } else {
                auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                if (Qwen35HasLocalMoeShard(localWeights)) {
                    Qwen3CudaMergeMOEBlock(cudaRunner, &attenInput, &expertIndex, &expertScore,
                        &localWeights, &localBiass,
                        &w1, &w2, &w3,
                        &tempInput, &tempOutput,
                        1.0f, &moeFinal, i,
                        computeType, threadTpMoeAtype,
                        &moeInputTemp, &moeOutputTemp);
                } else {
                    Qwen35ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
            }
            moeFinal.Reshape(hiddenStates.dims);
            if (sharedExpertPending) {
                if (sharedOutput.dataType != moeFinal.dataType) {
                    Qwen3CudaToDataType(cudaRunner, sharedOutput, moeFinal.dataType);
                }
                sharedOutput.Reshape(moeFinal.dims);
                Qwen3CudaAddTo(cudaRunner, moeFinal, sharedOutput);
            }
            addPartialToResidualReduce(moeFinal);
        }
        mtpWorkerProfileSyncMark(mtpWorkerProfileLayersUs);
        if (speculativeCacheOnlyForward) {
            logits.FreeSpace();
            logits.dims.clear();
            logits.strides.clear();
            logits.expansionDims.clear();
            return;
        }
        Data lastHiddenStates;
        Data *headInput = &hiddenStates;
        bool keepAllRowsForSpeculative =
            (speculativeCollectAllLogits || speculativeCaptureAllHiddenStates) && batch == 1;
        if (!all1 && !keepAllRowsForSpeculative) {
            int total = 0;
            std::vector<Data> lastToks(seqLens.size());
            std::vector<Data*> lastTokPointers;
            lastTokPointers.reserve(seqLens.size());
            for (int bidx = 0; bidx < (int)seqLens.size(); bidx++) {
                Qwen3CudaSplit(cudaRunner, hiddenStates, 1,
                               total + seqLens[bidx] - 1, total + seqLens[bidx],
                               lastToks[bidx]);
                total += seqLens[bidx];
                lastTokPointers.push_back(&lastToks[bidx]);
            }
            Qwen3CudaCatBatch(cudaRunner, lastTokPointers, 1, lastHiddenStates);
            headInput = &lastHiddenStates;
        }
        Qwen3CudaRMSNorm(cudaRunner, *headInput,
                         *requireLocal(weight[language_prefix + "norm.weight"],
                                       language_prefix + "norm.weight"),
                         rms_norm_eps, *headInput);
        if (keepAllRowsForSpeculative && (!tensorParallel || firstTensorParallelRank)) {
            speculativeHiddenStates.CopyFrom(*headInput);
        }
        if (!all1 && speculativeCaptureAllHiddenStates &&
            !speculativeCollectAllLogits) {
            Qwen3CudaSplit(cudaRunner, *headInput, 1,
                           headInput->dims[1] - 1, headInput->dims[1],
                           lastHiddenStates);
            headInput = &lastHiddenStates;
        }
        Qwen3CudaLinear(cudaRunner, *headInput,
                        *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                        *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                      "lm_head.weight.tp_bias"),
                        logits);
        Qwen3CudaToDataType(cudaRunner, logits, DataType::FLOAT32);
        mtpWorkerProfileSyncMark(mtpWorkerProfileHeadUs);
        mtpWorkerProfileRecord();
#endif
    }

    std::vector <int> Qwen3_5Model::ForwardGPU(
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
        (void)attentionMask;
        int mtpTargetProfileInterval = speculativeCollectAllLogits ?
            Qwen35MtpProfileInterval() : 0;
        bool mtpTargetProfileEnabled = mtpTargetProfileInterval > 0;
        int mtpTargetProfileSeqTokens = 0;
        for (int len : seqLens) {
            mtpTargetProfileSeqTokens += len;
        }
        auto mtpTargetProfileStart = mtpTargetProfileEnabled ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();
        auto mtpTargetProfileLast = mtpTargetProfileStart;
        long long mtpTargetProfileSetupUs = 0;
        long long mtpTargetProfileWeightPrepUs = 0;
        long long mtpTargetProfileInputPrepUs = 0;
        long long mtpTargetProfileEmbeddingUs = 0;
        long long mtpTargetProfileCacheLocalUs = 0;
        long long mtpTargetProfileWorkerUs = 0;
        long long mtpTargetProfileMetaSyncUs = 0;
        long long mtpTargetProfileSamplingUs = 0;
        auto mtpTargetProfileMark = [&](long long &slot) {
            if (!mtpTargetProfileEnabled) {
                return;
            }
            auto now = std::chrono::steady_clock::now();
            slot += Qwen35MtpProfileElapsedUs(mtpTargetProfileLast, now);
            mtpTargetProfileLast = now;
        };
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios)) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;
        auto mtpTargetProfileRecord = [&](int logitRows) {
            if (!mtpTargetProfileEnabled) {
                return;
            }
            long long totalUs = Qwen35MtpProfileElapsedUs(
                mtpTargetProfileStart, std::chrono::steady_clock::now());
            Qwen35MtpTargetProfileRecord(
                mtpTargetProfileInterval, tensorParallel,
                mtpTargetProfileSeqTokens, logitRows,
                mtpTargetProfileSetupUs, mtpTargetProfileWeightPrepUs,
                mtpTargetProfileInputPrepUs, mtpTargetProfileEmbeddingUs,
                mtpTargetProfileCacheLocalUs, mtpTargetProfileWorkerUs,
                mtpTargetProfileMetaSyncUs, mtpTargetProfileSamplingUs,
                totalUs);
        };
        bool useCpuEmbedding = !GetCudaEmbedding() || GetLowMemMode();
        const DataType computeType = ResolveQwen35ThreadTpComputeType(this->dataType);

        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "Qwen3.5 ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "Qwen3.5 ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "Qwen3.5 ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "Qwen3.5 ForwardGPU doesn't support CPU KV cache.\n");
        AssertInFastLLM(num_k_heads > 0 && num_v_heads > 0 &&
                        head_k_dim > 0 && head_v_dim > 0 &&
                        num_v_heads % num_k_heads == 0,
                        "Qwen3.5 ForwardGPU requires valid linear attention head metadata.\n");
        if (tensorParallel) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "Qwen3.5 ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = qwen35ThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;

        auto runSplitBatchForward = [&]() -> std::vector<int> {
            std::vector<int> ret;
            ret.reserve(batch);
            int inputOffset = 0;
            for (int b = 0; b < batch; b++) {
                Data curInputIds;
                Split(inputIds, 1, inputOffset, inputOffset + seqLens[b], curInputIds);
                inputOffset += seqLens[b];
                std::vector<Data*> curAttentionMask = {
                    b < (int)attentionMask.size() ? attentionMask[b] : nullptr
                };
                std::vector<Data*> curPositionIds = {
                    b < (int)positionIds.size() ? positionIds[b] : nullptr
                };
                std::vector<int> curSeqLens = {seqLens[b]};
                std::vector<GenerationConfig> curGenerationConfigs = {generationConfigs[b]};
                LastTokensManager curLastTokens;
                if (b < (int)lastTokens.units.size()) {
                    curLastTokens.units.push_back(lastTokens.units[b]);
                }
                std::vector<std::pair<Data*, Data*> > curPastKeyValues;
                curPastKeyValues.reserve(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    curPastKeyValues.push_back(pastKeyValues[b * block_cnt + i]);
                }
                std::vector<std::vector<float>*> curLogits;
                std::vector<std::vector<float>*> *curLogitsPtr = nullptr;
                if (retLogits != nullptr) {
                    curLogits.push_back(b < (int)retLogits->size() ? (*retLogits)[b] : nullptr);
                    curLogitsPtr = &curLogits;
                }
                std::vector<int> curRet = ForwardGPU(1, curInputIds, curAttentionMask,
                                                     curPositionIds, curSeqLens,
                                                     curPastKeyValues, curGenerationConfigs,
                                                     curLastTokens, curLogitsPtr);
                ret.push_back(curRet[0]);
            }
            return ret;
        };

        auto canRunFusedBatchDecode = [&]() -> bool {
            if (batch <= 1 || !all1 || (int)pastKeyValues.size() < batch * block_cnt) {
                return false;
            }
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                    Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                    if (pastKey == nullptr || pastValue == nullptr) {
                        return false;
                    }
                    bool isAttentionLayer =
                        weight.weight.find(language_prefix + "layers." + std::to_string(i) +
                                           ".self_attn.o_proj.weight") != weight.weight.end();
                    if (isAttentionLayer) {
                        if (!pastKey->isPagedKVCache || !pastValue->isPagedKVCache ||
                            pastKey->pagedKVCacheData == nullptr ||
                            pastValue->pagedKVCacheData == nullptr ||
                            pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                            return false;
                        }
                    } else if (pastKey->dims.empty() || pastValue->dims.empty() ||
                               pastKey->dims[0] != 1 || pastValue->dims[0] != 1) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (batch > 1 && (!all1 || !canRunFusedBatchDecode())) {
            return runSplitBatchForward();
        }

        mtpTargetProfileMark(mtpTargetProfileSetupUs);
        if (num_experts > 0) {
            if (!Qwen35MoeDisableFusedMoe() &&
                Qwen35CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap)) {
                TryBuildFusedMoeWeightsFromLoaded();
            }
            PrepareMoeWeights();
        }
        PrepareGdnWeights();
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight[language_prefix + "embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
        if (!useCpuEmbedding) {
            PrepareQwen35CudaEmbeddingWeightType(weight[language_prefix + "embed_tokens.weight"], computeType);
        }
        mtpTargetProfileMark(mtpTargetProfileWeightPrepUs);

        Data allPositionIds = BuildFlattenedPositionIds(positionIds, seqLens, all1);
        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        if (tensorParallel) {
            PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
            PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
        }
        mtpTargetProfileMark(mtpTargetProfileInputPrepUs);

        std::vector<DivisionScheme> localAttentionKvSchemes(block_cnt);
        std::vector<DivisionScheme> localLinearValueSchemes(block_cnt);
        DivisionScheme localLmHeadScheme;
        const std::vector<DivisionScheme> *attentionKvSchemes = &localAttentionKvSchemes;
        const std::vector<DivisionScheme> *linearValueSchemes = &localLinearValueSchemes;
        const DivisionScheme *lmHeadScheme = &localLmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        auto layerHasMoe = [&](int layer) {
            return num_experts > 0 &&
                   weight.weight.find(language_prefix + "layers." + std::to_string(layer) +
                                      ".mlp.gate.weight") != weight.weight.end();
        };
        auto layerNeedsCudaMoeCache = [&](int layer) {
            return layerHasMoe(layer) &&
                   !Qwen35LayerUsesMappedNonCudaMoe(this, layer) &&
                   !HasFusedMoeWeights(layer);
        };
        auto anyLayerNeedsCudaMoeCache = [&]() {
            for (int i = 0; i < block_cnt; i++) {
                if (layerNeedsCudaMoeCache(i)) {
                    return true;
                }
            }
            return false;
        };
        auto hasMoeCache = [&](const std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                               const std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache) {
            if (!anyLayerNeedsCudaMoeCache()) {
                return true;
            }
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
                    if (!layerNeedsCudaMoeCache(i)) {
                        continue;
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
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto &deviceWeights = weightCache[device];
                auto &deviceBiass = biasCache[device];
                deviceWeights.resize(block_cnt);
                deviceBiass.resize(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    auto &layerWeights = deviceWeights[i];
                    auto &layerBiass = deviceBiass[i];
                    layerWeights.reserve(expectedSize);
                    layerBiass.reserve(expectedSize);
                    layerWeights.push_back(nullptr);
                    layerWeights.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    if (!layerNeedsCudaMoeCache(i)) {
                        layerWeights.resize(expectedSize, nullptr);
                        layerBiass.resize(expectedSize, nullptr);
                        continue;
                    }
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = language_prefix + "layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = language_prefix + "layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
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

        auto ensureInitializedAdd1 = [&]() {
            if (initialized_add1) {
                return;
            }
            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(this->weight[language_prefix + "norm.weight"]);
            AddMtpRmsNormOffset();
            initialized_add1 = true;
        };

        if (tensorParallel) {
            auto usePreparedThreadTpSchemes = [&]() {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "Qwen3.5 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpAttentionKVHeadSchemes.size() == block_cnt &&
                                (int)threadTpLinearKeyHeadSchemes.size() == block_cnt &&
                                (int)threadTpLinearValueHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                hasMoeCache(threadTpMoeWeights, threadTpMoeBiass),
                                "Qwen3.5 ForwardGPU thread TP cached weight schemes are incomplete.\n");
                attentionKvSchemes = &threadTpAttentionKVHeadSchemes;
                linearValueSchemes = &threadTpLinearValueHeadSchemes;
                lmHeadScheme = &threadTpLmHeadScheme;
            };

            if (threadTpWeightsPrepared.load(std::memory_order_acquire)) {
                usePreparedThreadTpSchemes();
            } else {
                std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
                ensureInitializedAdd1();
                if (!threadTpWeightsPrepared.load(std::memory_order_relaxed)) {
                    auto prepareReplicated = [&](const std::string &name) {
                        if (weight.weight.find(name) != weight.weight.end()) {
                            PrepareMultiCudaReplicatedData(this->weight[name], devices, true);
                        }
                    };
                    if (!useCpuEmbedding) {
                        prepareReplicated(language_prefix + "embed_tokens.weight");
                    }
                    prepareReplicated(language_prefix + "norm.weight");
                    PrepareMultiCudaReplicatedData(inv_scale_data, devices, true);

                    threadTpAttentionKVHeadSchemes.assign(block_cnt, DivisionScheme());
                    threadTpLinearKeyHeadSchemes.assign(block_cnt, DivisionScheme());
                    threadTpLinearValueHeadSchemes.assign(block_cnt, DivisionScheme());

                    for (int i = 0; i < block_cnt; i++) {
                        std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
                        std::string inputRmsName = prefix + "input_layernorm.weight";
                        std::string postRmsName = prefix + "post_attention_layernorm.weight";
                        std::string swigluWeightName = prefix + "mlp.gateup_proj.weight";
                        std::string downWeightName = prefix + "mlp.down_proj.weight";
                        std::string downBiasName = prefix + "mlp.down_proj.bias";

                        prepareReplicated(inputRmsName);
                        prepareReplicated(postRmsName);

                        bool isAttentionLayer =
                            weight.weight.find(prefix + "self_attn.o_proj.weight") != weight.weight.end();
                        if (isAttentionLayer) {
                            std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                            std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
                            std::string qNormName = prefix + "self_attn.q_norm.weight";
                            std::string kNormName = prefix + "self_attn.k_norm.weight";
                            std::string oWeightName = prefix + "self_attn.o_proj.weight";
                            std::string oBiasName = prefix + "self_attn.o_proj.bias";
                            AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                            "Qwen3.5 ForwardGPU requires merged qgate/k/v weight.\n");
                            prepareReplicated(qNormName);
                            prepareReplicated(kNormName);

                            Data &mergeW = weight[mergeQkvWeightName];
                            Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                            mergeW.tpPackType = TP_PACK_NONE;
                            DivisionScheme qkvScheme = BuildQwen35GatedAttentionQkvScheme(
                                devices, ratios, num_attention_heads, num_key_value_heads, head_dim);
                            std::vector<int> devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                            "Qwen3.5 ForwardGPU failed to split " + mergeQkvWeightName + ".\n");
                            int qGateWidth = num_attention_heads * head_dim * 2;
                            threadTpAttentionKVHeadSchemes[i] =
                                ExtractQwen35AttentionKVHeadScheme(qkvScheme, qGateWidth, head_dim);
                            DivisionScheme oScheme = ExtractQwen35AttentionOutputScheme(qkvScheme);
                            Data &oB = GetThreadTensorParallelBias(oBiasName);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, oScheme, 1, true),
                                            "Qwen3.5 ForwardGPU failed to split " + oWeightName + ".\n");
                        } else {
                            std::string qkvzWeightName = prefix + "linear_attn.in_proj_qkvz.weight";
                            std::string baWeightName = prefix + "linear_attn.in_proj_ba.weight";
                            std::string qkvzbaWeightName = prefix + "linear_attn.in_proj_qkvzba.weight";
                            std::string conv1dWeightName = prefix + "linear_attn.conv1d.weight";
                            std::string conv1dBiasName = prefix + "linear_attn.conv1d.bias";
                            std::string aLogName = prefix + "linear_attn.A_log";
                            std::string dtBiasName = prefix + "linear_attn.dt_bias";
                            std::string outNormWeightName = prefix + "linear_attn.norm.weight";
                            std::string outProjWeightName = prefix + "linear_attn.out_proj.weight";
                            bool hasMergedGdnInLinear =
                                weight.weight.find(qkvzbaWeightName) != weight.weight.end();
                            AssertInFastLLM(hasMergedGdnInLinear ||
                                            (weight.weight.find(qkvzWeightName) != weight.weight.end() &&
                                             weight.weight.find(baWeightName) != weight.weight.end()),
                                            "Qwen3.5 ForwardGPU requires linear attention qkvzba or qkvz/ba weights.\n");
                            prepareReplicated(outNormWeightName);

                            DivisionScheme keyScheme = BuildQwen35LinearKeyHeadScheme(
                                devices, ratios, num_k_heads);
                            BalanceMultiCudaDivisionSchemeByLayer(
                                hasMergedGdnInLinear ? qkvzbaWeightName : qkvzWeightName,
                                devices, keyScheme);
                            DivisionScheme valueScheme = BuildQwen35LinearValueHeadScheme(
                                keyScheme, num_v_heads / num_k_heads);
                            threadTpLinearKeyHeadSchemes[i] = keyScheme;
                            threadTpLinearValueHeadSchemes[i] = valueScheme;

                            std::vector<int> devCopy = devices;
                            if (hasMergedGdnInLinear) {
                                DivisionScheme qkvzbaScheme = BuildQwen35LinearQkvzbaScheme(
                                    keyScheme, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
                                Data &qkvzbaBias = GetThreadTensorParallelBias(qkvzbaWeightName + ".tp_bias");
                                AssertInFastLLM(SplitMultiCudaWeight(weight[qkvzbaWeightName], qkvzbaBias,
                                                                     devCopy, qkvzbaScheme, 0, true),
                                                "Qwen3.5 ForwardGPU failed to split " + qkvzbaWeightName + ".\n");
                            } else {
                                DivisionScheme qkvzScheme = BuildQwen35LinearQkvzScheme(
                                    keyScheme, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
                                Data &qkvzBias = GetThreadTensorParallelBias(qkvzWeightName + ".tp_bias");
                                AssertInFastLLM(SplitMultiCudaWeight(weight[qkvzWeightName], qkvzBias,
                                                                     devCopy, qkvzScheme, 0, true),
                                                "Qwen3.5 ForwardGPU failed to split " + qkvzWeightName + ".\n");
                                DivisionScheme baScheme = BuildQwen35LinearBaScheme(
                                    valueScheme, num_v_heads);
                                Data &baBias = GetThreadTensorParallelBias(baWeightName + ".tp_bias");
                                devCopy = devices;
                                AssertInFastLLM(SplitMultiCudaWeight(weight[baWeightName], baBias,
                                                                     devCopy, baScheme, 0, true),
                                                "Qwen3.5 ForwardGPU failed to split " + baWeightName + ".\n");
                            }

                            DivisionScheme convScheme = BuildQwen35LinearConvScheme(
                                keyScheme, num_k_heads, num_v_heads, head_k_dim, head_v_dim);
                            AssertInFastLLM(SplitQwen35Conv1DWeight(weight[conv1dWeightName],
                                                                    weight[conv1dBiasName],
                                                                    devices, convScheme),
                                            "Qwen3.5 ForwardGPU failed to split " + conv1dWeightName + ".\n");

                            AssertInFastLLM(SplitQwen35VectorWeight(weight[aLogName], devices, valueScheme),
                                            "Qwen3.5 ForwardGPU failed to split " + aLogName + ".\n");
                            AssertInFastLLM(SplitQwen35VectorWeight(weight[dtBiasName], devices, valueScheme),
                                            "Qwen3.5 ForwardGPU failed to split " + dtBiasName + ".\n");

                            DivisionScheme outScheme = BuildQwen35LinearOutProjScheme(valueScheme, head_v_dim);
                            Data &outBias = GetThreadTensorParallelBias(outProjWeightName + ".tp_bias");
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[outProjWeightName], outBias,
                                                                 devCopy, outScheme, 1, true),
                                            "Qwen3.5 ForwardGPU failed to split " + outProjWeightName + ".\n");
                        }

                        bool hasDenseMlp = weight.weight.find(swigluWeightName) != weight.weight.end() &&
                                           weight.weight.find(downWeightName) != weight.weight.end();
                        if (hasDenseMlp) {
                            Data &gateup = weight[swigluWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(swigluWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            std::vector<int> devCopy = devices;
                            DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            BalanceMultiCudaPairedHalfDivisionSchemeSizesByLayer(
                                swigluWeightName, devices, gateScheme, gateup.dims[0] / 2);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0, true),
                                            "Qwen3.5 ForwardGPU failed to split " + swigluWeightName + ".\n");

                            Data &downBias = GetThreadTensorParallelBias(downBiasName);
                            weight[downWeightName].tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractQwen35FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[downWeightName], downBias,
                                                                 devCopy, downScheme, 1, true),
                                            "Qwen3.5 ForwardGPU failed to split " + downWeightName + ".\n");
                            continue;
                        }

                        std::string gateWeightName = prefix + "mlp.gate.weight";
                        std::string gateBiasName = prefix + "mlp.gate.e_score_correction_bias";
                        AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                        "Qwen3.5 ForwardGPU layer has neither dense MLP nor router gate weight.\n");
                        prepareReplicated(gateWeightName);
                        if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                            prepareReplicated(gateBiasName);
                        }

                        std::string sharedGateupWeightName = prefix + "mlp.shared_expert.gateup_proj.weight";
                        std::string sharedDownWeightName = prefix + "mlp.shared_expert.down_proj.weight";
                        std::string sharedExpertGateWeightName = prefix + "mlp.shared_expert_gate.weight";
                        if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                            AssertInFastLLM(weight.weight.find(sharedGateupWeightName) != weight.weight.end(),
                                            "Qwen3.5 ForwardGPU requires merged shared expert gateup weight.\n");
                            Data &sharedGateup = weight[sharedGateupWeightName];
                            Data &sharedGateupBias = GetThreadTensorParallelBias(sharedGateupWeightName + ".tp_bias");
                            sharedGateup.tpLinearType = TP_LINEAR_ROW;
                            sharedGateup.tpPackType = TP_PACK_GATEUP;
                            std::vector<int> devCopy = devices;
                            DivisionScheme sharedGateScheme = BuildMultiCudaRowSplitScheme(sharedGateup, devCopy, ratios);
                            BalanceMultiCudaPairedHalfDivisionSchemeSizesByLayer(
                                sharedGateupWeightName, devices, sharedGateScheme, sharedGateup.dims[0] / 2);
                            AssertInFastLLM(SplitMultiCudaWeight(sharedGateup, sharedGateupBias,
                                                                 devCopy, sharedGateScheme, 0, true),
                                            "Qwen3.5 ForwardGPU failed to split " + sharedGateupWeightName + ".\n");

                            Data &sharedDownBias = GetThreadTensorParallelBias(sharedDownWeightName + ".tp_bias");
                            weight[sharedDownWeightName].tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme sharedDownScheme = ExtractQwen35FirstRangeScheme(sharedGateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[sharedDownWeightName], sharedDownBias,
                                                                 devCopy, sharedDownScheme, 1, true),
                                            "Qwen3.5 ForwardGPU failed to split " + sharedDownWeightName + ".\n");
                            if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                                prepareReplicated(sharedExpertGateWeightName);
                            }
                        }

                        if (Qwen35LayerUsesMappedNonCudaMoe(this, i) || HasFusedMoeWeights(i)) {
                            continue;
                        }

                        DivisionScheme gateScheme;
                        for (int j = 0; j < this->num_experts; j++) {
                            std::string gateupWeightName = prefix + "mlp.experts." +
                                                           std::to_string(j) + ".gateup_proj.weight";
                            std::string expertDownWeightName = prefix + "mlp.experts." +
                                                               std::to_string(j) + ".down_proj.weight";
                            AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                            "Qwen3.5 ForwardGPU requires merged expert gateup weight.\n");
                            AssertInFastLLM(weight.weight.find(expertDownWeightName) != weight.weight.end(),
                                            "Qwen3.5 ForwardGPU requires expert down weight.\n");

                            Data &gateup = weight[gateupWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            std::vector<int> devCopy = devices;
                            gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            BalanceMultiCudaPairedHalfDivisionSchemeSizesByLayer(
                                gateupWeightName, devices, gateScheme, gateup.dims[0] / 2);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0, true),
                                            "Qwen3.5 ForwardGPU failed to split " + gateupWeightName + ".\n");

                            Data &down = weight[expertDownWeightName];
                            Data &downBias = GetThreadTensorParallelBias(expertDownWeightName + ".tp_bias");
                            down.tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractQwen35FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1, true),
                                            "Qwen3.5 ForwardGPU failed to split " + expertDownWeightName + ".\n");
                        }
                    }

                    if (HasPlannedFusedMoeLayers()) {
                        PrepareFusedMoeWeightsForDevices(devices, ratios);
                    }
                    if (anyLayerNeedsCudaMoeCache()) {
                        fillMoeCache(threadTpMoeWeights, threadTpMoeBiass, true);
                    } else {
                        threadTpMoeWeights.clear();
                        threadTpMoeBiass.clear();
                    }

                    Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                    std::vector<int> devCopy = devices;
                    threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0, true),
                                    "Qwen3.5 ForwardGPU failed to split lm_head.weight.\n");

                    threadTpPreparedDevices = devices;
                    threadTpPreparedRatios = ratios;
                    threadTpWeightsPrepared.store(true, std::memory_order_release);
                }
                usePreparedThreadTpSchemes();
            }
        } else {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            ensureInitializedAdd1();
            if (!singleGpuWeightsPrepared.load(std::memory_order_relaxed) ||
                !hasMoeCache(singleGpuMoeWeights, singleGpuMoeBiass)) {
                int device = devices[0];
                for (int i = 0; i < block_cnt; i++) {
                    std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
                    std::string swigluWeightName = prefix + "mlp.gateup_proj.weight";
                    std::string downWeightName = prefix + "mlp.down_proj.weight";
                    bool hasDenseMlp = weight.weight.find(swigluWeightName) != weight.weight.end() &&
                                       weight.weight.find(downWeightName) != weight.weight.end();
                    if (hasDenseMlp) {
                        weight[swigluWeightName].tpPackType = TP_PACK_GATEUP;
                        continue;
                    }
                    if (!layerHasMoe(i)) {
                        continue;
                    }

                    std::string gateWeightName = prefix + "mlp.gate.weight";
                    std::string gateBiasName = prefix + "mlp.gate.e_score_correction_bias";
                    AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                    "Qwen3.5 ForwardGPU requires router gate weight.\n");
                    weight[gateWeightName].ToDevice(DataDevice::CUDA, {device}, true);
                    if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                        weight[gateBiasName].ToDevice(DataDevice::CUDA, {device}, true);
                    }

                    std::string sharedGateupWeightName = prefix + "mlp.shared_expert.gateup_proj.weight";
                    std::string sharedDownWeightName = prefix + "mlp.shared_expert.down_proj.weight";
                    std::string sharedExpertGateWeightName = prefix + "mlp.shared_expert_gate.weight";
                    if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                        AssertInFastLLM(weight.weight.find(sharedGateupWeightName) != weight.weight.end(),
                                        "Qwen3.5 ForwardGPU requires merged shared expert gateup weight.\n");
                        weight[sharedGateupWeightName].tpPackType = TP_PACK_GATEUP;
                        weight[sharedGateupWeightName].ToDevice(DataDevice::CUDA, {device}, true);
                        weight[sharedDownWeightName].ToDevice(DataDevice::CUDA, {device}, true);
                        if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                            weight[sharedExpertGateWeightName].ToDevice(DataDevice::CUDA, {device}, true);
                        }
                    }

                    if (Qwen35LayerUsesMappedNonCudaMoe(this, i)) {
                        continue;
                    }
                    if (HasFusedMoeWeights(i)) {
                        Qwen35PrepareFusedMoeWeightForCuda(*moeGate3DWeights[i], device);
                        Qwen35PrepareFusedMoeWeightForCuda(*moeUp3DWeights[i], device);
                        Qwen35PrepareFusedMoeWeightForCuda(*moeDown3DWeights[i], device);
                        continue;
                    }
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = prefix + "mlp.experts." +
                                                       std::to_string(j) + ".gateup_proj.weight";
                        std::string expertDownWeightName = prefix + "mlp.experts." +
                                                           std::to_string(j) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                        "Qwen3.5 ForwardGPU requires merged expert gateup weight.\n");
                        AssertInFastLLM(weight.weight.find(expertDownWeightName) != weight.weight.end(),
                                        "Qwen3.5 ForwardGPU requires expert down weight.\n");
                        Data &gateup = weight[gateupWeightName];
                        Data &down = weight[expertDownWeightName];
                        gateup.tpLinearType = TP_LINEAR_ROW;
                        gateup.tpPackType = TP_PACK_GATEUP;
                        down.tpLinearType = TP_LINEAR_COLUMN;
                        gateup.ToDevice(DataDevice::CUDA, {device}, true);
                        down.ToDevice(DataDevice::CUDA, {device}, true);
                    }
                }
                if (anyLayerNeedsCudaMoeCache()) {
                    fillMoeCache(singleGpuMoeWeights, singleGpuMoeBiass, false);
                } else {
                    singleGpuMoeWeights.clear();
                    singleGpuMoeBiass.clear();
                }
                singleGpuWeightsPrepared.store(true, std::memory_order_release);
            }
            for (int i = 0; i < block_cnt; i++) {
                if (weight.weight.find(language_prefix + "layers." + std::to_string(i) +
                                       ".self_attn.o_proj.weight") != weight.weight.end()) {
                    localAttentionKvSchemes[i][devices[0]].push_back({0, num_key_value_heads});
                } else {
                    localLinearValueSchemes[i][devices[0]].push_back({0, num_v_heads});
                }
            }
            localLmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        if (tensorParallel && !useCpuEmbedding) {
            PrepareMultiCudaReplicatedData(weight[language_prefix + "embed_tokens.weight"], devices, true);
        }
        mtpTargetProfileMark(mtpTargetProfileWeightPrepUs);

        Data cpuEmbeddingHiddenStates;
        Data *precomputedHiddenStates = nullptr;
        if (useCpuEmbedding) {
            Data cpuInputIds;
            cpuInputIds.CopyFrom(inputIds);
            Qwen35CpuEmbeddingDirect(cpuInputIds, weight[language_prefix + "embed_tokens.weight"],
                                     cpuEmbeddingHiddenStates, computeType);
            PrepareQwen35CpuEmbeddingHiddenStates(cpuEmbeddingHiddenStates, devices, threadTpWorkerGroup);
            precomputedHiddenStates = &cpuEmbeddingHiddenStates;
        }
        mtpTargetProfileMark(mtpTargetProfileEmbeddingUs);

        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues;
        if (tensorParallel) {
            localPastKeyValues.resize(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                localPastKeyValues[r].resize(pastKeyValues.size());
                for (int idx = 0; idx < (int)pastKeyValues.size(); idx++) {
                    int layer = idx % block_cnt;
                    bool isLinearLayer =
                        weight.weight.find(language_prefix + "layers." + std::to_string(layer) +
                                           ".self_attn.o_proj.weight") == weight.weight.end();
                    DataType keyCacheType;
                    DataType valueCacheType;
                    if (isLinearLayer) {
                        keyCacheType = computeType;
                        valueCacheType = computeType;
                        Qwen35PrepareLinearAttentionCache(*pastKeyValues[idx].first, keyCacheType);
                        Qwen35PrepareLinearAttentionCache(*pastKeyValues[idx].second, valueCacheType);
                    } else {
                        pastKeyValues[idx].first->isLinearAttention = false;
                        pastKeyValues[idx].second->isLinearAttention = false;
                        keyCacheType = ResolveQwen35ThreadTpCacheType(
                            pastKeyValues[idx].first->dataType, computeType);
                        valueCacheType = ResolveQwen35ThreadTpCacheType(
                            pastKeyValues[idx].second->dataType, computeType);
                    }
                    localPastKeyValues[r][idx].first = EnsureQwen35ThreadTpLocalCache(
                        *pastKeyValues[idx].first, device, keyCacheType);
                    localPastKeyValues[r][idx].second = EnsureQwen35ThreadTpLocalCache(
                        *pastKeyValues[idx].second, device, valueCacheType);
                    if (isLinearLayer) {
                        Qwen35PrepareLinearAttentionCache(*localPastKeyValues[r][idx].first, keyCacheType);
                        Qwen35PrepareLinearAttentionCache(*localPastKeyValues[r][idx].second, valueCacheType);
                    } else {
                        localPastKeyValues[r][idx].first->isLinearAttention = false;
                        localPastKeyValues[r][idx].second->isLinearAttention = false;
                    }
                }
            }
        } else {
            int device = devices[0];
            for (int idx = 0; idx < (int)pastKeyValues.size(); idx++) {
                int layer = idx % block_cnt;
                bool isLinearLayer =
                    weight.weight.find(language_prefix + "layers." + std::to_string(layer) +
                                       ".self_attn.o_proj.weight") == weight.weight.end();
                DataType keyCacheType;
                DataType valueCacheType;
                if (isLinearLayer) {
                    keyCacheType = computeType;
                    valueCacheType = computeType;
                    Qwen35PrepareLinearAttentionCache(*pastKeyValues[idx].first, keyCacheType);
                    Qwen35PrepareLinearAttentionCache(*pastKeyValues[idx].second, valueCacheType);
                } else {
                    pastKeyValues[idx].first->isLinearAttention = false;
                    pastKeyValues[idx].second->isLinearAttention = false;
                    keyCacheType = ResolveQwen35ThreadTpCacheType(
                        pastKeyValues[idx].first->dataType, computeType);
                    valueCacheType = ResolveQwen35ThreadTpCacheType(
                        pastKeyValues[idx].second->dataType, computeType);
                }
                PrepareQwen35SingleCudaCache(*pastKeyValues[idx].first, device, keyCacheType);
                PrepareQwen35SingleCudaCache(*pastKeyValues[idx].second, device, valueCacheType);
            }
        }
        mtpTargetProfileMark(mtpTargetProfileCacheLocalUs);

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
        mtpTargetProfileMark(mtpTargetProfileWorkerUs);

        if (tensorParallel) {
            auto validLocalMeta = [](Data *data) {
                return data != nullptr && data->dims.size() >= 3;
            };
            int globalLinearConvDim = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim;
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
                                if (!validLocalMeta(localKeyMeta) && validLocalMeta(rankPastKeyValues[idx].first)) {
                                    localKeyMeta = rankPastKeyValues[idx].first;
                                }
                                if (!validLocalMeta(localValueMeta) && validLocalMeta(rankPastKeyValues[idx].second)) {
                                    localValueMeta = rankPastKeyValues[idx].second;
                                }
                                if (validLocalMeta(localKeyMeta) && validLocalMeta(localValueMeta)) {
                                    break;
                                }
                            }
                        }
                    }
                    bool isLinearLayer =
                        weight.weight.find(language_prefix + "layers." + std::to_string(i) +
                                           ".self_attn.o_proj.weight") == weight.weight.end();
                    if (isLinearLayer) {
                        std::string prefix = language_prefix + "layers." + std::to_string(i) + ".";
                        DivisionScheme linearKeyScheme = BuildQwen35LinearKeyHeadScheme(
                            devices, ratios, num_k_heads);
                        BalanceMultiCudaDivisionSchemeByLayer(
                            prefix + "linear_attn.in_proj_qkvz.weight", devices, linearKeyScheme);
                        DivisionScheme convScheme = BuildQwen35LinearConvScheme(
                            linearKeyScheme,
                            num_k_heads, num_v_heads, head_k_dim, head_v_dim);
                        SyncQwen35ThreadTpRootCacheMetaFromLocal(
                            *pastKeyValues[idx].first, localKeyMeta, devices, convScheme,
                            {1, globalLinearConvDim, 4}, 1, true);
                        SyncQwen35ThreadTpRootCacheMetaFromLocal(
                            *pastKeyValues[idx].second, localValueMeta, devices,
                            (*linearValueSchemes)[i],
                            {1, num_v_heads, head_k_dim, head_v_dim}, 1, true);
                    } else {
                        std::vector<int> globalDims = {num_key_value_heads,
                                                       localKeyMeta != nullptr && localKeyMeta->dims.size() > 1 ?
                                                           localKeyMeta->dims[1] : 0,
                                                       head_dim};
                        SyncQwen35ThreadTpRootCacheMetaFromLocal(
                            *pastKeyValues[idx].first, localKeyMeta, devices,
                            (*attentionKvSchemes)[i], globalDims, 0, false);
                        SyncQwen35ThreadTpRootCacheMetaFromLocal(
                            *pastKeyValues[idx].second, localValueMeta, devices,
                            (*attentionKvSchemes)[i], globalDims, 0, false);
                    }
                }
            }
        }
        mtpTargetProfileMark(mtpTargetProfileMetaSyncUs);

        if (speculativeCacheOnlyForward) {
            return {};
        }

        int vocabSize = lmHead.dims[0];
        int logitRows = batch;
        auto singleDeviceHasFullVocabLogits = [&]() {
            if (devices.size() != 1) {
                return false;
            }
            auto schemeIt = lmHeadScheme->find(devices[0]);
            return schemeIt != lmHeadScheme->end() &&
                   schemeIt->second.size() == 1 &&
                   schemeIt->second[0].first == 0 &&
                   schemeIt->second[0].second == vocabSize &&
                   !localLogits.empty() &&
                   localLogits[0].dataDevice == DataDevice::CUDA &&
                   localLogits[0].cudaData != nullptr &&
                   localLogits[0].dims.size() >= 2 &&
                   localLogits[0].dims.back() == vocabSize;
        };
        if (speculativeCollectAllLogits) {
            AssertInFastLLM(!localLogits.empty() && !localLogits[0].dims.empty(),
                            "Qwen3.5 speculative forward did not produce logits.\n");
            int localVocab = localLogits[0].dims.back();
            logitRows = localVocab > 0 ? (int)(localLogits[0].Count(0) / localVocab) : 0;
            AssertInFastLLM(logitRows > 0,
                            "Qwen3.5 speculative forward produced empty logits.\n");
            std::vector<GenerationConfig> rowConfigs(logitRows, generationConfigs[0]);
            auto getCacheLenForMtpReset = [](const Data *cache) {
                if (cache == nullptr) {
                    return 0;
                }
                if (cache->isPagedKVCache) {
                    if (cache->pageIndex.empty()) {
                        return 0;
                    }
                    return ((int)cache->pageIndex.size() - 1) * cache->pageLen + cache->lastPageLen;
                }
                if (cache->dims.size() > 1) {
                    return cache->dims[1];
                }
                if (cache->expansionDims.size() > 1) {
                    return cache->expansionDims[1];
                }
                return 0;
            };
            auto getTokenGrowingCacheLenForMtpReset = [&]() {
                auto tryGetLen = [&](int idx) {
                    if (idx < 0 || idx >= (int)pastKeyValues.size()) {
                        return 0;
                    }
                    Data *pastKey = pastKeyValues[idx].first;
                    Data *pastValue = pastKeyValues[idx].second;
                    if (pastKey == nullptr || pastValue == nullptr ||
                        pastKey->isLinearAttention || pastValue->isLinearAttention) {
                        return 0;
                    }
                    int len = getCacheLenForMtpReset(pastKey);
                    return len > 0 ? len : getCacheLenForMtpReset(pastValue);
                };
                int len = tryGetLen(kvCacheId);
                if (len > 0) {
                    return len;
                }
                for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                    len = tryGetLen(i);
                    if (len > 0) {
                        return len;
                    }
                }
                return 0;
            };
            bool resetEosForMtp = false;
            std::vector<int> eosIdsForMtp;
            if (generationConfigs[0].output_token_least > 0) {
                int cacheLen = getTokenGrowingCacheLenForMtpReset();
                resetEosForMtp =
                    generationConfigs[0].output_token_least - cacheLen +
                    generationConfigs[0].input_token_length > 0;
                if (resetEosForMtp) {
                    eosIdsForMtp.push_back(this->eos_token_id);
                    eosIdsForMtp.insert(eosIdsForMtp.end(),
                                        this->eos_token_ids.begin(),
                                        this->eos_token_ids.end());
                    eosIdsForMtp.insert(eosIdsForMtp.end(),
                                        generationConfigs[0].stop_token_ids.begin(),
                                        generationConfigs[0].stop_token_ids.end());
                }
            }
            if (devices.size() > 1 && generationConfigs[0].IsSimpleGreedy()) {
                std::vector<int> sampled = Qwen35SampleGreedyFromShardedCudaLogits(
                    devices, *lmHeadScheme, localLogits, logitRows, vocabSize,
                    resetEosForMtp, eosIdsForMtp);
                mtpTargetProfileMark(mtpTargetProfileSamplingUs);
                mtpTargetProfileRecord(logitRows);
                return sampled;
            }
            void *oldExecutor = GetExecutor();
            Executor samplingExecutor;
            samplingExecutor.SetFirstDevice("cuda:" + std::to_string(devices[0]));
            SetCurrentThreadExecutor(&samplingExecutor);
            Data *sampleLogits = nullptr;
            Data &fullCudaLogits = Qwen35ThreadLocalCudaSamplingFullLogits();
            if (singleDeviceHasFullVocabLogits()) {
                sampleLogits = &localLogits[0];
            } else {
                Qwen35GatherShardLogitsToRootCuda(devices[0], devices, *lmHeadScheme,
                                                  localLogits, logitRows, vocabSize,
                                                  fullCudaLogits);
                sampleLogits = &fullCudaLogits;
            }
            ResetLogitsOfEOS(logitRows, sampleLogits, pastKeyValues, rowConfigs);
            SetCurrentThreadExecutor(oldExecutor);
            std::vector<int> sampled = Qwen35SampleFromRootCudaLogits(devices[0], *sampleLogits, logitRows,
                                                                       1, true, rowConfigs);
            mtpTargetProfileMark(mtpTargetProfileSamplingUs);
            mtpTargetProfileRecord(logitRows);
            return sampled;
        }
        bool allSimpleCudaSampling = true;
        int cudaSamplingTopK = 1;
        if (Qwen35CanUseCudaFullLogitsSampling(generationConfigs, retLogits, batch,
                                               allSimpleCudaSampling, cudaSamplingTopK)) {
            Data *rootCudaLogits = nullptr;
            if (devices.size() == 1) {
                rootCudaLogits = &localLogits[0];
                AssertInFastLLM(rootCudaLogits->dataDevice == DataDevice::CUDA &&
                                rootCudaLogits->cudaData != nullptr,
                                "Qwen3.5 CUDA sampling: single GPU logits must stay on CUDA.\n");
                AssertInFastLLM(rootCudaLogits->dims.size() > 0 &&
                                rootCudaLogits->dims.back() == vocabSize &&
                                rootCudaLogits->Count(0) / vocabSize == batch,
                                "Qwen3.5 CUDA sampling: single GPU logits shape mismatch.\n");
            } else {
                Data &fullCudaLogits = Qwen35ThreadLocalCudaSamplingFullLogits();
                Qwen35GatherShardLogitsToRootCuda(devices[0], devices, *lmHeadScheme,
                                                  localLogits, batch, vocabSize,
                                                  fullCudaLogits);
                rootCudaLogits = &fullCudaLogits;
            }
            void *oldExecutor = GetExecutor();
            Executor samplingExecutor;
            samplingExecutor.SetFirstDevice("cuda:" + std::to_string(devices[0]));
            SetCurrentThreadExecutor(&samplingExecutor);
            ResetLogitsOfEOS(batch, rootCudaLogits, pastKeyValues, generationConfigs);
            Qwen35PrintCudaLogitsTopK(*rootCudaLogits, batch, "Qwen3.5 CUDA");
            SetCurrentThreadExecutor(oldExecutor);
            return Qwen35SampleFromRootCudaLogits(devices[0], *rootCudaLogits, batch,
                                                  cudaSamplingTopK, allSimpleCudaSampling,
                                                  generationConfigs);
        }

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
                            "Qwen3.5 ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            auto schemeIt = lmHeadScheme->find(device);
            AssertInFastLLM(schemeIt != lmHeadScheme->end(),
                            "Qwen3.5 ForwardGPU: missing lm_head split scheme.\n");
            for (auto &range : schemeIt->second) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "Qwen3.5 ForwardGPU: invalid lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    memcpy(dst + (long long)b * vocabSize + range.first,
                           src + (long long)b * localVocab + localOffset,
                           (size_t)len * sizeof(float));
                }
                localOffset += len;
            }
        }

        ResetLogitsOfEOS(batch, &fullLogits, pastKeyValues, generationConfigs);
        Qwen35PrintCpuLogitsTopK(fullLogits, batch, "Qwen3.5 CPU");
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

    bool Qwen3_5Model::Qwen35MTPForward(
        bool useGPUForward,
        ResponseContext *context,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        std::vector <std::vector <int> > &acceptedTokens,
        std::vector <std::vector <int> > &nextInputTokens,
        std::vector <int> &keptInputLens) {
#ifndef USE_CUDA
        (void)useGPUForward;
        (void)context;
        (void)inputIds;
        (void)attentionMask;
        (void)positionIds;
        (void)seqLens;
        (void)pastKeyValues;
        (void)generationConfigs;
        (void)acceptedTokens;
        (void)nextInputTokens;
        (void)keptInputLens;
        return false;
#else
        if (Qwen35MtpDisabledByEnv()) {
            return false;
        }

        auto logMtpSkip = [&](const std::string &reason) {
            if (mtpSkipLogPrinted.exchange(true)) {
                return;
            }
            bool simpleGreedy = !generationConfigs.empty() && generationConfigs[0].IsSimpleGreedy();
            bool outputLogits = !generationConfigs.empty() && generationConfigs[0].output_logits;
            int outputLeast = generationConfigs.empty() ? -1 : generationConfigs[0].output_token_least;
            int topK = generationConfigs.empty() ? -1 : generationConfigs[0].top_k;
            float repeatPenalty = generationConfigs.empty() ? -1.0f : generationConfigs[0].repeat_penalty;
            bool hasPositionIds = !positionIds.empty() && positionIds[0] != nullptr;
            printf("[Qwen3.5 MTP] not enabled: %s (use_gpu=%d seq_lens=%zu mtp_layers=%d has_mtp_weights=%d simple_greedy=%d top_k=%d repeat_penalty=%.4f output_logits=%d output_least=%d has_position_ids=%d past_kv=%zu block_cnt=%d).\n",
                   reason.c_str(), useGPUForward ? 1 : 0, seqLens.size(),
                   mtp_num_hidden_layers, HasMtpWeights() ? 1 : 0,
                   simpleGreedy ? 1 : 0, topK, repeatPenalty,
                   outputLogits ? 1 : 0, outputLeast, hasPositionIds ? 1 : 0,
                   pastKeyValues.size(), block_cnt);
            fflush(stdout);
        };

        if (!useGPUForward) {
            logMtpSkip("GPU forward is disabled");
            return false;
        }
        if (seqLens.size() != 1 || context == nullptr) {
            return false;
        }
        if (generationConfigs.empty() || !generationConfigs[0].IsSimpleGreedy()) {
            logMtpSkip("generation config is not simple greedy");
            return false;
        }
        if (generationConfigs[0].output_logits) {
            logMtpSkip("generation config needs logits");
            return false;
        }
        if (!HasMtpWeights()) {
            logMtpSkip("MTP weights were not found");
            return false;
        }
        if (positionIds.empty() || positionIds[0] == nullptr ||
            (int)pastKeyValues.size() < block_cnt) {
            logMtpSkip("position ids or KV cache are incomplete");
            return false;
        }
        if (seqLens[0] <= 0) {
            return false;
        }

        std::lock_guard<std::mutex> mtpCacheGuard(mtpCacheMutex);
        MtpKvCache &mtpCache = mtpCaches[context];
        if (context->cacheLen > 0 && mtpCache.tokens == 0) {
            logMtpSkip("prefix cache hit without MTP cache");
            return false;
        }
        if (mtpCache.tokens == 0 && context->preTokens > seqLens[0]) {
            logMtpSkip("MTP cache was not seeded during prefill");
            return false;
        }

        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) ||
            devices.empty()) {
            logMtpSkip("CUDA forward device is unavailable");
            return false;
        }
        int device = devices[0];
        bool tensorParallel = devices.size() > 1;
        const int logInterval = QWEN35_MTP_LOG_INTERVAL;
        int mtpDraftsPerStep = Qwen35MtpDraftsPerStep();
        if (mtpDraftsPerStep <= 0) {
            return false;
        }
        if (!mtpLogPrinted.exchange(true)) {
            printf("[Qwen3.5 MTP] enabled: layers=%d, drafts_per_step=%d, root_device=cuda:%d, tp_devices=%zu, log_interval=%d validations.\n",
                   mtp_num_hidden_layers, mtpDraftsPerStep, device,
                   devices.size(), logInterval);
            fflush(stdout);
        }

        auto logMtpStats = [&]() {
            long long validations = mtpValidationCount.load(std::memory_order_relaxed);
            if (validations <= 0) {
                return;
            }
            if (validations % logInterval != 0) {
                return;
            }
            printf("[Qwen3.5 MTP] pos_accept_rate=[");
            for (int i = 0; i < mtpDraftsPerStep; i++) {
                long long attempts =
                    mtpDraftPositionAttempts[i].load(std::memory_order_relaxed);
                long long accepts =
                    mtpDraftPositionAccepts[i].load(std::memory_order_relaxed);
                double rate = attempts > 0 ?
                    (double)accepts * 100.0 / (double)attempts : 0.0;
                printf("%s%.2f%%", i == 0 ? "" : ", ", rate);
            }
            printf("].\n");
            fflush(stdout);
        };

        int mtpProfileInterval = Qwen35MtpProfileInterval();
        bool mtpProfileEnabled = mtpProfileInterval > 0;
        auto mtpProfileStart = mtpProfileEnabled ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point();
        auto mtpProfileLast = mtpProfileStart;
        long long mtpProfileSetupUs = 0;
        long long mtpProfileCachePrepUs = 0;
        long long mtpProfileTargetUs = 0;
        long long mtpProfileMatchUs = 0;
        long long mtpProfileCommitUs = 0;
        long long mtpProfileRollbackUs = 0;
        long long mtpProfileRetryUs = 0;
        long long mtpProfileDraftUs = 0;
        long long mtpProfileDraftFirstUs = 0;
        long long mtpProfileDraftExtraUs = 0;
        auto mtpProfileMark = [&](long long &slot) {
            if (!mtpProfileEnabled) {
                return;
            }
            auto now = std::chrono::steady_clock::now();
            slot += Qwen35MtpProfileElapsedUs(mtpProfileLast, now);
            mtpProfileLast = now;
        };
        auto mtpProfileAddSpan = [&](long long &slot,
                                     std::chrono::steady_clock::time_point begin) {
            if (!mtpProfileEnabled) {
                return;
            }
            slot += Qwen35MtpProfileElapsedUs(begin, std::chrono::steady_clock::now());
        };
        auto mtpProfileRecord = [&](Qwen35MtpProfilePath path, bool speculative,
                                    int draftSlots, int matchedDrafts,
                                    int committedTokens) {
            if (!mtpProfileEnabled) {
                return;
            }
            long long totalUs = Qwen35MtpProfileElapsedUs(
                mtpProfileStart, std::chrono::steady_clock::now());
            Qwen35MtpProfileRecord(
                mtpProfileInterval, path, speculative, draftSlots,
                matchedDrafts, committedTokens,
                mtpProfileSetupUs, mtpProfileCachePrepUs,
                mtpProfileTargetUs, mtpProfileMatchUs,
                mtpProfileCommitUs, mtpProfileRollbackUs,
                mtpProfileRetryUs, mtpProfileDraftUs,
                mtpProfileDraftFirstUs, mtpProfileDraftExtraUs,
                totalUs);
        };

        struct CacheMeta {
            bool isPagedKVCache = false;
            int pageLen = 128;
            PagedCacheManager *pagedKVCacheData = nullptr;
            std::vector<int> pageIndex;
            int lastPageLen = 0;
            std::vector<int> dims;
            std::vector<uint64_t> strides;
            std::vector<int> expansionDims;
            uint64_t expansionSize = 0;
            uint64_t expansionBytes = 0;
        };

        auto makeMeta = [](Data &cache) {
            CacheMeta meta;
            meta.isPagedKVCache = cache.isPagedKVCache;
            meta.pageLen = cache.pageLen;
            meta.pagedKVCacheData = cache.pagedKVCacheData;
            meta.pageIndex = cache.pageIndex;
            meta.lastPageLen = cache.lastPageLen;
            meta.dims = cache.dims;
            meta.strides = cache.strides;
            meta.expansionDims = cache.expansionDims;
            meta.expansionSize = cache.expansionSize;
            meta.expansionBytes = cache.expansionBytes;
            return meta;
        };

        auto restoreMeta = [](Data &cache, const CacheMeta &meta) {
            std::vector<int> newPageIndex = meta.pageIndex;
            std::vector<int> newDims = meta.dims;
            std::vector<uint64_t> newStrides = meta.strides;
            std::vector<int> newExpansionDims = meta.expansionDims;
            std::vector<int> releasePages;
            PagedCacheManager *oldManager = cache.pagedKVCacheData;
            if (cache.isPagedKVCache && cache.pagedKVCacheData != nullptr) {
                releasePages.reserve(cache.pageIndex.size());
                for (int page : cache.pageIndex) {
                    if (std::find(meta.pageIndex.begin(), meta.pageIndex.end(), page) ==
                        meta.pageIndex.end()) {
                        releasePages.push_back(page);
                    }
                }
            }
            cache.isPagedKVCache = meta.isPagedKVCache;
            cache.pageLen = meta.pageLen;
            cache.pagedKVCacheData = meta.pagedKVCacheData;
            cache.pageIndex.swap(newPageIndex);
            cache.lastPageLen = meta.lastPageLen;
            cache.dims.swap(newDims);
            cache.strides.swap(newStrides);
            cache.expansionDims.swap(newExpansionDims);
            cache.expansionSize = meta.expansionSize;
            cache.expansionBytes = meta.expansionBytes;
            if (!releasePages.empty() && oldManager != nullptr) {
                try {
                    oldManager->ReleasePageIndices(releasePages);
                } catch (...) {
                    // The new metadata is already installed. If the manager
                    // cannot enqueue an old page under OOM, keep the live
                    // cache valid and conservatively lose that free page.
                }
            }
        };

        auto assignMetaNoRelease = [](Data &cache, const CacheMeta &meta) {
            std::vector<int> newPageIndex = meta.pageIndex;
            std::vector<int> newDims = meta.dims;
            std::vector<uint64_t> newStrides = meta.strides;
            std::vector<int> newExpansionDims = meta.expansionDims;
            cache.isPagedKVCache = meta.isPagedKVCache;
            cache.pageLen = meta.pageLen;
            cache.pagedKVCacheData = meta.pagedKVCacheData;
            cache.pageIndex.swap(newPageIndex);
            cache.lastPageLen = meta.lastPageLen;
            cache.dims.swap(newDims);
            cache.strides.swap(newStrides);
            cache.expansionDims.swap(newExpansionDims);
            cache.expansionSize = meta.expansionSize;
            cache.expansionBytes = meta.expansionBytes;
        };

        auto installPreparedMetaNoRelease = [](Data &cache, CacheMeta &meta) noexcept {
            cache.isPagedKVCache = meta.isPagedKVCache;
            cache.pageLen = meta.pageLen;
            cache.pagedKVCacheData = meta.pagedKVCacheData;
            cache.pageIndex.swap(meta.pageIndex);
            cache.lastPageLen = meta.lastPageLen;
            cache.dims.swap(meta.dims);
            cache.strides.swap(meta.strides);
            cache.expansionDims.swap(meta.expansionDims);
            cache.expansionSize = meta.expansionSize;
            cache.expansionBytes = meta.expansionBytes;
        };

        std::vector<uint8_t> attentionLayerMask(block_cnt, 0);
        for (int layer = 0; layer < block_cnt; layer++) {
            std::string outputWeightName = language_prefix + "layers." +
                std::to_string(layer) + ".self_attn.o_proj.weight";
            attentionLayerMask[layer] =
                weight.weight.find(outputWeightName) != weight.weight.end();
        }
        auto isAttentionLayerAt = [&](int layer) noexcept {
            return layer >= 0 && layer < (int)attentionLayerMask.size() &&
                   attentionLayerMask[layer] != 0;
        };

        auto copyPagedCacheView = [&](Data &dst, const Data &src) {
            dst.isFake = false;
            dst.cacheUid = src.cacheUid;
            dst.isKVCache = src.isKVCache;
            dst.isLinearAttention = src.isLinearAttention;
            dst.isLinearAttentionTransposed = src.isLinearAttentionTransposed;
            dst.isPagedKVCache = src.isPagedKVCache;
            dst.pageLen = src.pageLen;
            dst.pagedKVCacheData = src.pagedKVCacheData;
            dst.pageIndex = src.pageIndex;
            dst.lastPageLen = src.lastPageLen;
            dst.dataType = src.dataType;
            dst.unitSize = src.unitSize;
            dst.unitSizeDiv = src.unitSizeDiv;
            dst.dims = src.dims;
            dst.strides = src.strides;
            dst.expansionSize = src.expansionSize;
            dst.expansionBytes = src.expansionBytes;
            dst.expansionDims = src.expansionDims;
            dst.cudaData = src.cudaData;
            dst.cudaDataBorrowed = true;
            dst.dataDevice = src.dataDevice;
            dst.dataDeviceIds = src.dataDeviceIds;
            dst.tpLayout = src.tpLayout;
            dst.tpAxis = src.tpAxis;
            dst.tpGlobalDims = src.tpGlobalDims;
            dst.tpRanges = src.tpRanges;
            dst.tpLinearType = src.tpLinearType;
            dst.tpPackType = src.tpPackType;
            dst.tpQHeads = src.tpQHeads;
            dst.tpKVHeads = src.tpKVHeads;
            dst.tpHeadDim = src.tpHeadDim;
        };

        auto recomputeContiguousStrides = [](std::vector<int> &dims,
                                             std::vector<uint64_t> &strides) {
            strides.assign(dims.size(), 1);
            if (dims.empty()) {
                return;
            }
            for (int i = (int)dims.size() - 2; i >= 0; i--) {
                strides[i] = (uint64_t)dims[i + 1] * strides[i + 1];
            }
        };

        auto advancePagedMetaTokens = [&](CacheMeta &meta, int tokens) {
            if (tokens <= 0) {
                return;
            }
            if (meta.dims.size() >= 2) {
                meta.dims[1] += tokens;
                if (meta.expansionDims.empty()) {
                    recomputeContiguousStrides(meta.dims, meta.strides);
                }
            }
        };

        auto copyPagedCachePage = [&](Data &cache, int srcPage, int dstPage) {
            PagedCacheManager *manager = cache.pagedKVCacheData;
            AssertInFastLLM(manager != nullptr && manager->dims.size() == 4,
                            "Qwen3.5 MTP validation paged cache manager is invalid.\n");
            size_t pageBytes = (size_t)manager->pageLen * manager->dims[2] *
                               manager->dims[3] * manager->unitSize / manager->unitSizeDiv;
            size_t srcOffset = (size_t)srcPage * pageBytes;
            size_t dstOffset = (size_t)dstPage * pageBytes;
            if (manager->dataDevice == DataDevice::CUDA) {
                int oldDevice = FastllmCudaGetDevice();
                if (!manager->dataDeviceIds.empty()) {
                    FastllmCudaSetDevice(manager->dataDeviceIds[0]);
                }
                try {
                    FastllmCudaCopyFromDeviceToDevice((uint8_t*)manager->cudaData + dstOffset,
                                                      (uint8_t*)manager->cudaData + srcOffset,
                                                      pageBytes);
                } catch (...) {
                    FastllmCudaSetDevice(oldDevice);
                    throw;
                }
                FastllmCudaSetDevice(oldDevice);
            } else {
                memcpy(manager->cpuData + dstOffset, manager->cpuData + srcOffset, pageBytes);
            }
        };

        auto detachPagedCacheView = [&](Data &cache) {
            cache.pageIndex.clear();
            cache.pagedKVCacheData = nullptr;
            cache.isPagedKVCache = false;
            cache.cudaData = nullptr;
            cache.cudaDataBorrowed = true;
        };

        auto copyPagedCacheForValidation = [&](Data &dst, const Data &src) {
            int newPage = -1;
            PagedCacheManager *manager = nullptr;
            try {
                if (!src.isPagedKVCache || src.pagedKVCacheData == nullptr ||
                    src.pageLen <= 0 ||
                    (!src.pageIndex.empty() &&
                     (src.lastPageLen <= 0 || src.lastPageLen > src.pageLen))) {
                    throw std::runtime_error(
                        "Qwen3.5 MTP validation got invalid paged-cache metadata.");
                }
                // Copying the borrowed view itself can allocate (pageIndex,
                // dims, TP metadata). Keep that inside the same ownership
                // guard as the copy-on-write page allocation.
                copyPagedCacheView(dst, src);
                if (dst.pageIndex.empty() || dst.lastPageLen == dst.pageLen) {
                    return;
                }
                int oldPage = dst.pageIndex.back();
                manager = dst.pagedKVCacheData;
                newPage = manager->GetUnusedPageIndex(true);
                copyPagedCachePage(dst, oldPage, newPage);
                dst.pageIndex.back() = newPage;
            } catch (...) {
                detachPagedCacheView(dst);
                if (newPage >= 0 && manager != nullptr) {
                    try {
                        manager->ReleasePageIndex(newPage);
                    } catch (...) {
                        // The view is already detached, so allocator failure
                        // in the free-page bookkeeping cannot double-release.
                    }
                }
                throw;
            }
        };

        auto copyTensorParallelRootMeta = [&](Data &dst, const Data &src) {
            std::map<int, Data*> keepLocals = dst.multiDeviceDatas;
            bool keepMultiDeviceData = dst.multiDeviceData;
            dst.isFake = false;
            dst.cacheUid = src.cacheUid;
            dst.isKVCache = src.isKVCache;
            dst.isLinearAttention = src.isLinearAttention;
            dst.isLinearAttentionTransposed = src.isLinearAttentionTransposed;
            dst.dataType = src.dataType;
            dst.UpdateUnitSize();
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = devices;
            dst.dims = src.dims;
            dst.strides = src.strides;
            dst.expansionDims = src.expansionDims;
            dst.expansionSize = src.expansionSize;
            dst.expansionBytes = src.expansionBytes;
            dst.tpLayout = src.tpLayout;
            dst.tpAxis = src.tpAxis;
            dst.tpGlobalDims = src.tpGlobalDims;
            dst.tpRanges = src.tpRanges;
            dst.tpLinearType = src.tpLinearType;
            dst.tpPackType = src.tpPackType;
            dst.tpQHeads = src.tpQHeads;
            dst.tpKVHeads = src.tpKVHeads;
            dst.tpHeadDim = src.tpHeadDim;
            dst.isPagedKVCache = src.isPagedKVCache;
            dst.pageLen = src.pageLen;
            dst.pagedKVCacheData = src.pagedKVCacheData;
            dst.pageIndex = src.pageIndex;
            dst.lastPageLen = src.lastPageLen;
            dst.cudaData = nullptr;
            dst.cudaDataBorrowed = true;
            dst.multiDeviceData = keepMultiDeviceData || !keepLocals.empty();
            dst.multiDeviceDatas = keepLocals;
        };

        auto getThreadTpLocalCache = [](Data &root, int localDevice) -> Data* {
            auto it = root.multiDeviceDatas.find(localDevice);
            return it == root.multiDeviceDatas.end() ? nullptr : it->second;
        };

        // Gated attention is sharded by KV head.  When TP is wider than the
        // number of KV heads, the prepared split scheme intentionally assigns
        // no heads to some ranks.  ForwardSingleGPU leaves those rank-local
        // cache objects pristine, so they must not be treated as broken paged
        // caches by speculative validation.
        auto tpAttentionLocalHeads = [&](int layer, int localDevice) {
            if (!tensorParallel || !isAttentionLayerAt(layer) ||
                layer < 0 || layer >= (int)threadTpAttentionKVHeadSchemes.size()) {
                return -1;
            }
            const DivisionScheme &scheme = threadTpAttentionKVHeadSchemes[layer];
            auto schemeIt = scheme.find(localDevice);
            if (schemeIt == scheme.end()) {
                return -1;
            }
            int heads = 0;
            for (const auto &range : schemeIt->second) {
                if (range.first < 0 || range.second < range.first ||
                    range.second > num_key_value_heads) {
                    return -1;
                }
                heads += range.second - range.first;
            }
            return heads;
        };

        auto isValidEmptyTpAttentionLocal = [&](const Data *cache,
                                                 int localDevice) {
            return cache != nullptr && !cache->isFake && cache->isKVCache &&
                   !cache->isLinearAttention && !cache->isPagedKVCache &&
                   cache->pagedKVCacheData == nullptr && cache->pageIndex.empty() &&
                   cache->lastPageLen == 0 && cache->dims.empty() &&
                   cache->strides.empty() && cache->expansionDims.empty() &&
                   cache->expansionSize == 0 && cache->expansionBytes == 0 &&
                   cache->cpuData == nullptr && cache->cudaData == nullptr &&
                   cache->deviceData == nullptr && !cache->multiDeviceData &&
                   cache->dataDevice == DataDevice::CUDA &&
                   cache->dataDeviceIds.size() == 1 &&
                   cache->dataDeviceIds[0] == localDevice;
        };

        auto isValidPagedTpAttentionLocal = [&](const Data *cache,
                                                 int localDevice,
                                                 int expectedHeads) {
            if (cache == nullptr || expectedHeads <= 0 || cache->isFake ||
                !cache->isKVCache || cache->isLinearAttention ||
                !cache->isPagedKVCache || cache->pagedKVCacheData == nullptr ||
                cache->pageLen <= 0 || cache->dataDevice != DataDevice::CUDA ||
                cache->dataDeviceIds.size() != 1 ||
                cache->dataDeviceIds[0] != localDevice ||
                cache->dims.size() != 3 || cache->dims[0] != expectedHeads ||
                cache->dims[2] != head_dim) {
                return false;
            }
            PagedCacheManager *manager = cache->pagedKVCacheData;
            if (manager->type != PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE ||
                manager->pageLen != cache->pageLen || manager->maxPages <= 0 ||
                manager->dims.size() != 4 || manager->dims[0] != manager->maxPages ||
                manager->dims[1] != cache->pageLen ||
                manager->dims[2] != expectedHeads || manager->dims[3] != head_dim) {
                return false;
            }
            if (cache->pageIndex.empty()) {
                return cache->lastPageLen == 0 && cache->dims[1] == 0;
            }
            if (cache->lastPageLen <= 0 || cache->lastPageLen > cache->pageLen) {
                return false;
            }
            for (int page : cache->pageIndex) {
                if (page < 0 || page >= manager->maxPages) {
                    return false;
                }
            }
            int cachedTokens = ((int)cache->pageIndex.size() - 1) * cache->pageLen +
                               cache->lastPageLen;
            return cache->dims[1] == cachedTokens;
        };

        auto describeTpAttentionLocal = [](const Data *cache) {
            if (cache == nullptr) {
                return std::string("null");
            }
            std::ostringstream ss;
            ss << "isFake=" << cache->isFake
               << ",isKV=" << cache->isKVCache
               << ",isLinear=" << cache->isLinearAttention
               << ",isPaged=" << cache->isPagedKVCache
               << ",manager=" << (cache->pagedKVCacheData != nullptr)
               << ",pageLen=" << cache->pageLen
               << ",pages=" << cache->pageIndex.size()
               << ",lastPageLen=" << cache->lastPageLen
               << ",device=" << (int)cache->dataDevice
               << ",deviceIds=[";
            for (size_t i = 0; i < cache->dataDeviceIds.size(); i++) {
                if (i > 0) {
                    ss << ',';
                }
                ss << cache->dataDeviceIds[i];
            }
            ss << "],dims=[";
            for (size_t i = 0; i < cache->dims.size(); i++) {
                if (i > 0) {
                    ss << ',';
                }
                ss << cache->dims[i];
            }
            ss << ']';
            if (cache->pagedKVCacheData != nullptr) {
                const PagedCacheManager *manager = cache->pagedKVCacheData;
                ss << ",managerType=" << (int)manager->type
                   << ",managerPageLen=" << manager->pageLen
                   << ",managerMaxPages=" << manager->maxPages
                   << ",managerDims=[";
                for (size_t i = 0; i < manager->dims.size(); i++) {
                    if (i > 0) {
                        ss << ',';
                    }
                    ss << manager->dims[i];
                }
                ss << ']';
            }
            return ss.str();
        };

        auto initializeEmptyTpAttentionLocal = [&](Data &dst, const Data &src,
                                                    int localDevice) {
            dst.isFake = false;
            dst.cacheUid = src.cacheUid;
            dst.isKVCache = true;
            dst.isLinearAttention = false;
            dst.isLinearAttentionTransposed = false;
            dst.isPagedKVCache = false;
            dst.pageLen = src.pageLen;
            dst.pagedKVCacheData = nullptr;
            dst.pageIndex.clear();
            dst.lastPageLen = 0;
            dst.dataType = src.dataType;
            dst.UpdateUnitSize();
            dst.dims.clear();
            dst.strides.clear();
            dst.expansionDims.clear();
            dst.expansionSize = 0;
            dst.expansionBytes = 0;
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = {localDevice};
            dst.cudaData = nullptr;
            dst.cudaDataBorrowed = false;
            dst.multiDeviceData = false;
            dst.multiDeviceDatas.clear();
            dst.tpLayout = src.tpLayout;
            dst.tpAxis = src.tpAxis;
            dst.tpGlobalDims = src.tpGlobalDims;
            dst.tpRanges = src.tpRanges;
            dst.tpLinearType = src.tpLinearType;
            dst.tpPackType = src.tpPackType;
            dst.tpQHeads = src.tpQHeads;
            dst.tpKVHeads = src.tpKVHeads;
            dst.tpHeadDim = src.tpHeadDim;
        };

        auto copyTensorForValidationLocal = [&](Data &dst, const Data &src, int localDevice) {
            if (src.dataDevice != DataDevice::CUDA || src.cudaData == nullptr) {
                dst.CopyFrom(src);
                return;
            }
            dst.isFake = false;
            dst.isKVCache = src.isKVCache;
            dst.isLinearAttention = src.isLinearAttention;
            dst.isLinearAttentionTransposed = src.isLinearAttentionTransposed;
            dst.cacheUid = src.cacheUid;
            dst.dataType = src.dataType;
            dst.UpdateUnitSize();
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = {localDevice};
            dst.tpLayout = src.tpLayout;
            dst.tpAxis = src.tpAxis;
            dst.tpGlobalDims = src.tpGlobalDims;
            dst.tpRanges = src.tpRanges;
            dst.tpLinearType = src.tpLinearType;
            dst.tpPackType = src.tpPackType;
            dst.tpQHeads = src.tpQHeads;
            dst.tpKVHeads = src.tpKVHeads;
            dst.tpHeadDim = src.tpHeadDim;
            int oldDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(localDevice);
            try {
                if (!src.expansionDims.empty() && src.expansionDims != src.dims) {
                    dst.Expansion(src.expansionDims);
                    dst.Resize(src.dims);
                    dst.Allocate();
                } else {
                    dst.Resize(src.dims);
                    dst.Allocate();
                }
                if (src.GetBytes() > 0) {
                    FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
                }
                dst.strides = src.strides;
                dst.expansionDims = src.expansionDims;
                dst.expansionSize = src.expansionSize;
                dst.expansionBytes = src.expansionBytes;
            } catch (...) {
                FastllmCudaSetDevice(oldDevice);
                throw;
            }
            FastllmCudaSetDevice(oldDevice);
        };

        auto copyTensorIntoExistingStorage = [&](Data &dst, const Data &src) {
            if (dst.dataDevice == DataDevice::CUDA && src.dataDevice == DataDevice::CUDA &&
                dst.cudaData != nullptr && src.cudaData != nullptr &&
                dst.dataType == src.dataType && dst.dims == src.dims &&
                dst.GetBytes() == src.GetBytes()) {
                int oldDevice = FastllmCudaGetDevice();
                if (!dst.dataDeviceIds.empty()) {
                    FastllmCudaSetDevice(dst.dataDeviceIds[0]);
                }
                try {
                    FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
                } catch (...) {
                    FastllmCudaSetDevice(oldDevice);
                    throw;
                }
                FastllmCudaSetDevice(oldDevice);
                dst.isKVCache = src.isKVCache;
                dst.isLinearAttention = src.isLinearAttention;
                dst.isLinearAttentionTransposed = src.isLinearAttentionTransposed;
                dst.cacheUid = src.cacheUid;
                dst.strides = src.strides;
                dst.expansionDims = src.expansionDims;
                dst.expansionSize = src.expansionSize;
                dst.expansionBytes = src.expansionBytes;
            } else {
                dst.CopyFrom(src);
            }
        };

        auto adoptTensorIntoExistingStorage = [&](Data &dst, Data &src) {
            if (dst.isFake || src.isFake ||
                dst.multiDeviceData || src.multiDeviceData ||
                dst.isPagedKVCache || src.isPagedKVCache) {
                copyTensorIntoExistingStorage(dst, src);
                return;
            }

            // Transfer the validation tensor into the real cache. The temp
            // tensor then owns the old real storage and releases it normally.
            using std::swap;
            swap(dst.isFake, src.isFake);
            swap(dst.cacheUid, src.cacheUid);
            swap(dst.isKVCache, src.isKVCache);
            swap(dst.isLinearAttention, src.isLinearAttention);
            swap(dst.isLinearAttentionTransposed, src.isLinearAttentionTransposed);
            swap(dst.lockInCPU, src.lockInCPU);
            swap(dst.weightType, src.weightType);
            swap(dst.dataType, src.dataType);
            swap(dst.unitSize, src.unitSize);
            swap(dst.unitSizeDiv, src.unitSizeDiv);
            swap(dst.dims, src.dims);
            swap(dst.strides, src.strides);
            swap(dst.expansionSize, src.expansionSize);
            swap(dst.expansionBytes, src.expansionBytes);
            swap(dst.expansionDims, src.expansionDims);
            swap(dst.cpuData, src.cpuData);
            swap(dst.cudaData, src.cudaData);
            swap(dst.cudaDataBorrowed, src.cudaDataBorrowed);
            swap(dst.extraCudaData, src.extraCudaData);
            swap(dst.extraCudaHalfData, src.extraCudaHalfData);
            swap(dst.deviceData, src.deviceData);
            swap(dst.extraDeviceData, src.extraDeviceData);
            swap(dst.dataDevice, src.dataDevice);
            swap(dst.dataDeviceIds, src.dataDeviceIds);
            swap(dst.perChannelAxis, src.perChannelAxis);
            swap(dst.group, src.group);
            swap(dst.groupCnt, src.groupCnt);
            swap(dst.blockK, src.blockK);
            swap(dst.blockM, src.blockM);
            swap(dst.perChannelsConfigs, src.perChannelsConfigs);
            swap(dst.scales, src.scales);
            swap(dst.mins, src.mins);
            swap(dst.zeros, src.zeros);
            swap(dst.weightSum, src.weightSum);
            swap(dst.halfScales, src.halfScales);
            swap(dst.directMemory, src.directMemory);
            swap(dst.tpLayout, src.tpLayout);
            swap(dst.tpAxis, src.tpAxis);
            swap(dst.tpGlobalDims, src.tpGlobalDims);
            swap(dst.tpRanges, src.tpRanges);
            swap(dst.tpLinearType, src.tpLinearType);
            swap(dst.tpPackType, src.tpPackType);
            swap(dst.tpQHeads, src.tpQHeads);
            swap(dst.tpKVHeads, src.tpKVHeads);
            swap(dst.tpHeadDim, src.tpHeadDim);
            swap(dst.isGGUFData, src.isGGUFData);
            swap(dst.ggmlTensor, src.ggmlTensor);
            swap(dst.ggmlType, src.ggmlType);
            swap(dst.IsRepacked, src.IsRepacked);
            swap(dst.isPinned, src.isPinned);
            swap(dst.cpuIntDatas, src.cpuIntDatas);
        };

        auto cacheTokenCountFromMeta = [](const CacheMeta &meta) {
            int pageLen = std::max(1, meta.pageLen);
            if (meta.isPagedKVCache) {
                if (meta.pageIndex.empty()) {
                    return 0;
                }
                int lastLen = meta.lastPageLen;
                if (lastLen <= 0 || lastLen > pageLen) {
                    return -1;
                }
                return ((int)meta.pageIndex.size() - 1) * pageLen + lastLen;
            }
            if (meta.dims.size() >= 2) {
                return meta.dims[1];
            }
            if (meta.expansionDims.size() >= 2) {
                return meta.expansionDims[1];
            }
            return 0;
        };

        auto tryPrefixTokenMetaFromTemp = [&](const CacheMeta &base,
                                              const Data &temp,
                                              int tokens,
                                              CacheMeta &prefix,
                                              std::string *error) {
            prefix = base;
            prefix.isPagedKVCache = temp.isPagedKVCache;
            prefix.pageLen = temp.pageLen;
            prefix.pagedKVCacheData = temp.pagedKVCacheData;
            if (tokens < 0) {
                if (error != nullptr) {
                    *error = "negative prefix tokens";
                }
                return false;
            }
            int pageLen = std::max(1, prefix.pageLen);
            int baseTokens = cacheTokenCountFromMeta(base);
            if (baseTokens < 0) {
                if (error != nullptr) {
                    *error = "invalid base paged-cache lastPageLen";
                }
                return false;
            }
            int newTokens = baseTokens + tokens;
            int newPageCount = newTokens <= 0 ? 0 :
                (newTokens + pageLen - 1) / pageLen;
            if (newPageCount > (int)temp.pageIndex.size()) {
                if (error != nullptr) {
                    *error = "temp cache missing prefix pages: baseTokens=" +
                             std::to_string(baseTokens) +
                             ", addTokens=" + std::to_string(tokens) +
                             ", newTokens=" + std::to_string(newTokens) +
                             ", pageLen=" + std::to_string(pageLen) +
                             ", basePages=" + std::to_string((int)base.pageIndex.size()) +
                             ", tempPages=" + std::to_string((int)temp.pageIndex.size());
                }
                return false;
            }
            prefix.pageIndex.assign(temp.pageIndex.begin(),
                                    temp.pageIndex.begin() + newPageCount);
            prefix.lastPageLen = newTokens <= 0 ? 0 : newTokens % pageLen;
            if (newTokens > 0 && prefix.lastPageLen == 0) {
                prefix.lastPageLen = pageLen;
            }
            advancePagedMetaTokens(prefix, tokens);
            return true;
        };

        auto prefixTokenMetaFromTemp = [&](const CacheMeta &base,
                                           const Data &temp,
                                           int tokens) {
            CacheMeta prefix;
            std::string error;
            AssertInFastLLM(tryPrefixTokenMetaFromTemp(base, temp, tokens,
                                                       prefix, &error),
                            "Qwen3.5 MTP validation " + error + ".\n");
            return prefix;
        };

        auto firstTokenMetaFromTemp = [&](const CacheMeta &base, const Data &temp) {
            return prefixTokenMetaFromTemp(base, temp, 1);
        };

        auto releaseUncommittedPagedPages = [&](Data &temp, const CacheMeta &committed) {
            if (!temp.isPagedKVCache || temp.pagedKVCacheData == nullptr) {
                detachPagedCacheView(temp);
                return;
            }
            if (temp.pageIndex == committed.pageIndex) {
                detachPagedCacheView(temp);
                return;
            }
            PagedCacheManager *manager = temp.pagedKVCacheData;
            std::vector<int> releasePages;
            releasePages.swap(temp.pageIndex);
            detachPagedCacheView(temp);
            size_t write = 0;
            for (int page : releasePages) {
                bool keep = std::find(committed.pageIndex.begin(),
                                      committed.pageIndex.end(), page) !=
                            committed.pageIndex.end();
                if (!keep) {
                    releasePages[write++] = page;
                }
            }
            releasePages.resize(write);
            if (releasePages.empty()) {
                return;
            }
            try {
                manager->ReleasePageIndices(releasePages);
            } catch (...) {
                // The borrowed view is already detached. Under allocator
                // failure, conservatively leave pages unavailable instead of
                // letting a noexcept scope-guard destructor double-release
                // them or terminate the process.
            }
        };

        auto runTargetWithPast = [&](const Data &curInputIds,
                                     const std::vector<Data*> &curAttentionMask,
                                     const std::vector<Data*> &curPositionIds,
                                     const std::vector<int> &curSeqLens,
                                     std::vector<std::pair<Data*, Data*> > &curPastKeyValues) {
            bool oldSpeculativeCollectAllLogits = speculativeCollectAllLogits;
            speculativeCollectAllLogits = true;
            speculativeHiddenStates.FreeSpace();
            speculativeHiddenStates.dims.clear();
            speculativeHiddenStates.strides.clear();
            speculativeHiddenStates.expansionDims.clear();
            std::vector<int> ret;
            try {
                ret = ForwardGPU(1, curInputIds, curAttentionMask,
                                 curPositionIds, curSeqLens,
                                 curPastKeyValues, generationConfigs,
                                 LastTokensManager(), nullptr);
            } catch (...) {
                speculativeCollectAllLogits = oldSpeculativeCollectAllLogits;
                throw;
            }
            speculativeCollectAllLogits = oldSpeculativeCollectAllLogits;
            return ret;
        };

        auto runTarget = [&](const Data &curInputIds,
                             const std::vector<Data*> &curAttentionMask,
                             const std::vector<Data*> &curPositionIds,
                             const std::vector<int> &curSeqLens) {
            return runTargetWithPast(curInputIds, curAttentionMask, curPositionIds,
                                     curSeqLens, pastKeyValues);
        };
        auto runTargetCacheOnly = [&](const Data &curInputIds,
                                      const std::vector<Data*> &curAttentionMask,
                                      const std::vector<Data*> &curPositionIds,
                                      const std::vector<int> &curSeqLens) {
            bool oldSpeculativeCollectAllLogits = speculativeCollectAllLogits;
            bool oldSpeculativeCacheOnlyForward = speculativeCacheOnlyForward;
            speculativeCollectAllLogits = false;
            speculativeCacheOnlyForward = true;
            try {
                ForwardGPU(1, curInputIds, curAttentionMask, curPositionIds, curSeqLens,
                           pastKeyValues, generationConfigs, LastTokensManager(), nullptr);
            } catch (...) {
                speculativeCacheOnlyForward = oldSpeculativeCacheOnlyForward;
                speculativeCollectAllLogits = oldSpeculativeCollectAllLogits;
                throw;
            }
            speculativeCacheOnlyForward = oldSpeculativeCacheOnlyForward;
            speculativeCollectAllLogits = oldSpeculativeCollectAllLogits;
        };

        Data inputCpu;
        inputCpu.CopyFrom(inputIds);
        inputCpu.ToDevice(DataDevice::CPU);
        if (inputCpu.dataType != DataType::FLOAT32) {
            ToDataType(inputCpu, DataType::FLOAT32);
            inputCpu.ToDevice(DataDevice::CPU);
        }
        float *inputPtr = (float*)inputCpu.cpuData;
        Data allPositionIds = BuildFlattenedPositionIds(positionIds, seqLens, seqLens[0] == 1);

        acceptedTokens.assign(1, std::vector<int>());
        nextInputTokens.assign(1, std::vector<int>());
        keptInputLens.assign(1, seqLens[0]);

        int seqLen = seqLens[0];
        auto tokenAt = [&](int index) {
            return (int)(inputPtr[index] + 1e-3f);
        };
        auto buildInputIdsSlice = [&](int begin, int end) {
            std::vector<float> values;
            values.reserve(std::max(0, end - begin));
            for (int i = begin; i < end; i++) {
                values.push_back((float)tokenAt(i));
            }
            return Data(DataType::FLOAT32, {1, end - begin}, values);
        };
        struct MtpRuntimeDataMeta {
            std::vector<int> dims;
            std::vector<uint64_t> strides;
            std::vector<int> expansionDims;
            uint64_t expansionSize = 0;
            uint64_t expansionBytes = 0;
        };
        struct MtpRuntimeCacheMeta {
            MtpRuntimeDataMeta key;
            MtpRuntimeDataMeta value;
            int tokens = 0;
        };
        auto makeMtpRuntimeDataMeta = [](const Data &data) {
            MtpRuntimeDataMeta meta;
            meta.dims = data.dims;
            meta.strides = data.strides;
            meta.expansionDims = data.expansionDims;
            meta.expansionSize = data.expansionSize;
            meta.expansionBytes = data.expansionBytes;
            return meta;
        };
        auto restoreMtpRuntimeDataMeta = [](Data &data, const MtpRuntimeDataMeta &meta) {
            data.dims = meta.dims;
            data.strides = meta.strides;
            data.expansionDims = meta.expansionDims;
            data.expansionSize = meta.expansionSize;
            data.expansionBytes = meta.expansionBytes;
        };
        auto makeMtpRuntimeCacheMeta = [&]() {
            MtpRuntimeCacheMeta meta;
            meta.key = makeMtpRuntimeDataMeta(mtpCache.key);
            meta.value = makeMtpRuntimeDataMeta(mtpCache.value);
            meta.tokens = mtpCache.tokens;
            return meta;
        };
        auto restoreMtpRuntimeCacheMeta = [&](const MtpRuntimeCacheMeta &meta) {
            restoreMtpRuntimeDataMeta(mtpCache.key, meta.key);
            restoreMtpRuntimeDataMeta(mtpCache.value, meta.value);
            mtpCache.tokens = meta.tokens;
        };
        auto setNextInputWithDrafts = [&](int firstToken, const std::vector<int> &drafts) {
            nextInputTokens[0].clear();
            nextInputTokens[0].reserve(1 + drafts.size());
            nextInputTokens[0].push_back(firstToken);
            nextInputTokens[0].insert(nextInputTokens[0].end(), drafts.begin(), drafts.end());
        };
        auto runMtpDraftChain = [&](const Data &targetHiddenStates,
                                    const std::vector<int> &mtpInputTokens,
                                    const Data &mtpPositionIds,
                                    int sampleRow,
                                    int lastPositionRow) {
            std::vector<int> drafts;
            drafts.reserve(mtpDraftsPerStep);
            Data draftHidden;
            auto firstDraftStart = mtpProfileEnabled ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point();
            int draft = RunMtpGreedyDraft(device, devices, mtpCache, targetHiddenStates,
                                          mtpInputTokens, mtpPositionIds,
                                          sampleRow,
                                          mtpDraftsPerStep > 1 ? &draftHidden : nullptr);
            mtpProfileAddSpan(mtpProfileDraftFirstUs, firstDraftStart);
            drafts.push_back(draft);
            if (mtpDraftsPerStep > 1) {
                MtpRuntimeCacheMeta runtimeMeta = makeMtpRuntimeCacheMeta();
                // 双缓冲保存上一轮 draft 的 hidden state, 避免每轮多一次 CopyFrom
                Data extraHiddenBuffers[2];
                Data *prevHidden = &draftHidden;
                int prevDraft = draft;
                try {
                    for (int extra = 1; extra < mtpDraftsPerStep; extra++) {
                        Data &extraHidden = extraHiddenBuffers[extra & 1];
                        Data extraPositionIds = BuildMtpPositionIdsSlice(
                            allPositionIds, lastPositionRow, lastPositionRow + 1, extra);
                        std::vector<int> extraInputTokens(1, prevDraft);
                        bool needNextHidden = extra + 1 < mtpDraftsPerStep;
                        auto extraDraftStart = mtpProfileEnabled ? std::chrono::steady_clock::now()
                                                                 : std::chrono::steady_clock::time_point();
                        int nextDraft = RunMtpGreedyDraft(device, devices, mtpCache, *prevHidden,
                                                          extraInputTokens, extraPositionIds,
                                                          0, needNextHidden ? &extraHidden : nullptr);
                        mtpProfileAddSpan(mtpProfileDraftExtraUs, extraDraftStart);
                        drafts.push_back(nextDraft);
                        if (needNextHidden) {
                            prevHidden = &extraHidden;
                        }
                        prevDraft = nextDraft;
                    }
                } catch (...) {
                    restoreMtpRuntimeCacheMeta(runtimeMeta);
                    throw;
                }
                restoreMtpRuntimeCacheMeta(runtimeMeta);
            }
            return drafts;
        };

        mtpProfileMark(mtpProfileSetupUs);
        std::vector<int> targetRet;
        bool isSpeculativeValidation =
            seqLen >= 2 && seqLen <= mtpDraftsPerStep + 1 &&
            mtpCache.tokens > 0 && context->preTokens > seqLen;
        if (!isSpeculativeValidation) {
            targetRet = runTarget(inputIds, attentionMask, positionIds, seqLens);
            mtpProfileMark(mtpProfileTargetUs);
            AssertInFastLLM((int)targetRet.size() >= seqLen,
                            "Qwen3.5 MTP target forward returned no token.\n");
            int nextToken = targetRet[seqLen - 1];
            std::vector<int> mtpInputTokens;
            mtpInputTokens.reserve(seqLen);
            for (int i = 1; i < seqLen; i++) {
                mtpInputTokens.push_back(tokenAt(i));
            }
            mtpInputTokens.push_back(nextToken);
            Data mtpPositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, seqLen, 0);
            std::vector<int> drafts = runMtpDraftChain(
                speculativeHiddenStates, mtpInputTokens, mtpPositionIds,
                seqLen - 1, seqLen - 1);
            mtpProfileMark(mtpProfileDraftUs);
            acceptedTokens[0].push_back(nextToken);
            setNextInputWithDrafts(nextToken, drafts);
            mtpProfileRecord(QWEN35_MTP_PROFILE_SEED, false, 0, 0, 1);
            return true;
        }

        auto canUseTpInplaceValidation = [&]() {
            if (!tensorParallel || seqLen < 2 ||
                seqLen > QWEN35_MTP_FAST_SEQ_MAX) {
                return false;
            }
            for (int i = 0; i < block_cnt; i++) {
                if (isAttentionLayerAt(i)) {
                    for (int localDevice : devices) {
                        int localHeads = tpAttentionLocalHeads(i, localDevice);
                        Data *localKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                               localDevice);
                        Data *localValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                 localDevice);
                        if (localHeads < 0) {
                            return false;
                        }
                        if (localHeads == 0) {
                            if (!isValidEmptyTpAttentionLocal(localKey, localDevice) ||
                                !isValidEmptyTpAttentionLocal(localValue, localDevice)) {
                                return false;
                            }
                            continue;
                        }
                        if (!isValidPagedTpAttentionLocal(localKey, localDevice, localHeads) ||
                            !isValidPagedTpAttentionLocal(localValue, localDevice, localHeads)) {
                            return false;
                        }
                    }
                    continue;
                }
                for (int localDevice : devices) {
                    Data *localKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                           localDevice);
                    Data *localValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                             localDevice);
                    if (localKey == nullptr || localValue == nullptr ||
                        localKey->dataDevice != DataDevice::CUDA ||
                        localValue->dataDevice != DataDevice::CUDA ||
                        localKey->dataType != DataType::FLOAT16 ||
                        localValue->dataType != DataType::FLOAT16 ||
                        localKey->cudaData == nullptr ||
                        localValue->cudaData == nullptr ||
                        localKey->dims.size() != 3 ||
                        localKey->dims[0] != 1 ||
                        localKey->dims[2] != 4 ||
                        localValue->dims.size() != 4 ||
                        localValue->dims[0] != 1 ||
                        localValue->dims[2] != head_k_dim) {
                        return false;
                    }
                    localKey->dataDeviceIds = {localDevice};
                    localValue->dataDeviceIds = {localDevice};
                    int oldDevice = FastllmCudaGetDevice();
                    FastllmCudaSetDevice(localDevice);
                    if (!Qwen35EnsureCudaLinearAttnStateTransposed(*localValue)) {
                        FastllmCudaSetDevice(oldDevice);
                        return false;
                    }
                    FastllmCudaSetDevice(oldDevice);
                }
            }
            return true;
        };

        if (canUseTpInplaceValidation()) {
            std::vector<CacheMeta> baseRootKeyMetas(block_cnt), baseRootValueMetas(block_cnt);
            std::vector<std::map<int, CacheMeta> > baseLocalKeyMetas(block_cnt), baseLocalValueMetas(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                Data *realKey = pastKeyValues[i].first;
                Data *realValue = pastKeyValues[i].second;
                AssertInFastLLM(realKey != nullptr && realValue != nullptr,
                                "Qwen3.5 MTP TP inplace validation got null cache.\n");
                if (!isAttentionLayerAt(i)) {
                    continue;
                }
                baseRootKeyMetas[i] = makeMeta(*realKey);
                baseRootValueMetas[i] = makeMeta(*realValue);
                for (int localDevice : devices) {
                    Data *localKey = getThreadTpLocalCache(*realKey, localDevice);
                    Data *localValue = getThreadTpLocalCache(*realValue, localDevice);
                    int localHeads = tpAttentionLocalHeads(i, localDevice);
                    AssertInFastLLM(localHeads >= 0,
                                    "Qwen3.5 MTP TP inplace validation has an invalid attention head scheme.\n");
                    if (localHeads == 0) {
                        AssertInFastLLM(
                            isValidEmptyTpAttentionLocal(localKey, localDevice) &&
                            isValidEmptyTpAttentionLocal(localValue, localDevice),
                            "Qwen3.5 MTP TP inplace validation got a malformed empty attention cache.\n");
                        continue;
                    }
                    AssertInFastLLM(
                        isValidPagedTpAttentionLocal(localKey, localDevice, localHeads) &&
                        isValidPagedTpAttentionLocal(localValue, localDevice, localHeads),
                        "Qwen3.5 MTP TP inplace validation got invalid local attention cache metadata.\n");
                    baseLocalKeyMetas[i][localDevice] = makeMeta(*localKey);
                    baseLocalValueMetas[i][localDevice] = makeMeta(*localValue);
                }
            }

            int oldLinearStateCaptureSlots = speculativeLinearStateCaptureSlots;
            bool oldCaptureFirstTokenLinearState = speculativeCaptureFirstTokenLinearState;
            int tpCaptureRestoreToken = 0;
            auto tpCaptureRestoreDeleter = [&](int*) {
                speculativeLinearStateCaptureSlots = oldLinearStateCaptureSlots;
                speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
            };
            std::unique_ptr<int, decltype(tpCaptureRestoreDeleter)> tpCaptureRestoreGuard(
                &tpCaptureRestoreToken, tpCaptureRestoreDeleter);

            int linearCaptureSlots = std::max(1, seqLen - 1);
            // The final scratch slot is a persistent base-state backup used
            // only by the exception/validation rollback guard.  The capture
            // kernels still see linearCaptureSlots, so their hot-path layout
            // and per-token snapshot count stay unchanged.
            int linearRollbackSlot = linearCaptureSlots;
            int linearScratchSlots = linearCaptureSlots + 1;
            auto tpCaptureRootReady = [&](const Data &root) {
                if (!root.multiDeviceData || root.dataDevice != DataDevice::CUDA ||
                    root.dataDeviceIds != devices ||
                    root.multiDeviceDatas.size() != devices.size()) {
                    return false;
                }
                for (int localDevice : devices) {
                    auto it = root.multiDeviceDatas.find(localDevice);
                    if (it == root.multiDeviceDatas.end() || it->second == nullptr ||
                        it->second->multiDeviceData) {
                        return false;
                    }
                }
                return true;
            };
            bool rebuildLinearCaptureScratch =
                speculativeLinearStates.size() != (size_t)block_cnt;
            if (!rebuildLinearCaptureScratch) {
                for (int i = 0; i < block_cnt && !rebuildLinearCaptureScratch; i++) {
                    if (isAttentionLayerAt(i)) {
                        continue;
                    }
                    if ((int)speculativeLinearStates[i].size() < linearScratchSlots) {
                        rebuildLinearCaptureScratch = true;
                        break;
                    }
                    for (int slot = 0; slot < linearScratchSlots; slot++) {
                        if (!tpCaptureRootReady(speculativeLinearStates[i][slot].first) ||
                            !tpCaptureRootReady(speculativeLinearStates[i][slot].second)) {
                            rebuildLinearCaptureScratch = true;
                            break;
                        }
                    }
                }
            }
            if (rebuildLinearCaptureScratch) {
                speculativeLinearStates.clear();
                speculativeLinearStates.resize(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    if (isAttentionLayerAt(i)) {
                        continue;
                    }
                    speculativeLinearStates[i].resize(linearScratchSlots);
                    for (int slot = 0; slot < linearScratchSlots; slot++) {
                        Data *roots[2] = {
                            &speculativeLinearStates[i][slot].first,
                            &speculativeLinearStates[i][slot].second
                        };
                        for (Data *root : roots) {
                            root->multiDeviceData = true;
                            root->dataDevice = DataDevice::CUDA;
                            root->dataDeviceIds = devices;
                            for (int localDevice : devices) {
                                root->multiDeviceDatas[localDevice] = new Data();
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < block_cnt; i++) {
                if (isAttentionLayerAt(i)) {
                    continue;
                }
                for (int slot = 0; slot < linearCaptureSlots; slot++) {
                    Data *roots[2] = {
                        &speculativeLinearStates[i][slot].first,
                        &speculativeLinearStates[i][slot].second
                    };
                    for (Data *root : roots) {
                        for (int localDevice : devices) {
                            Data *local = root->multiDeviceDatas.at(localDevice);
                            // Keep the shape and allocated storage for reuse.
                            // A per-rank sentinel avoids the data race that a
                            // shared capture bitmask would introduce under TP.
                            local->cacheUid = LLONG_MIN;
                        }
                    }
                }
            }
            speculativeLinearStateCaptureSlots = linearCaptureSlots;
            speculativeLinearCaptureMask.clear();

            // Snapshot every rank-local linear state before the in-place
            // target forward.  These tensors live in speculativeLinearStates
            // and are reused across decode steps; after the first validation
            // this is allocation-free and costs one D2D copy per state.  Run
            // ranks concurrently so TP does not serialize those copies across
            // devices.
            std::vector<std::exception_ptr> linearBackupErrors(devices.size());
            threadTpWorkerGroup.Run(devices, [&](int deviceIndex) {
                int localDevice = devices[deviceIndex];
                for (int i = 0; i < block_cnt; i++) {
                    if (isAttentionLayerAt(i)) {
                        continue;
                    }
                    Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                           localDevice);
                    Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                             localDevice);
                    Data *backupKey = getThreadTpLocalCache(
                        speculativeLinearStates[i][linearRollbackSlot].first,
                        localDevice);
                    Data *backupValue = getThreadTpLocalCache(
                        speculativeLinearStates[i][linearRollbackSlot].second,
                        localDevice);
                    if (realKey == nullptr || realValue == nullptr ||
                        backupKey == nullptr || backupValue == nullptr) {
                        throw std::runtime_error(
                            "Qwen3.5 MTP TP inplace validation cannot prepare linear rollback state.");
                    }
                    auto backupLinearState = [&](Data &backup, const Data &real) {
                        bool reusable = backup.dataDevice == DataDevice::CUDA &&
                            backup.cudaData != nullptr &&
                            backup.dataDeviceIds.size() == 1 &&
                            backup.dataDeviceIds[0] == localDevice &&
                            backup.dataType == real.dataType &&
                            backup.dims == real.dims &&
                            backup.GetBytes() == real.GetBytes();
                        if (reusable) {
                            copyTensorIntoExistingStorage(backup, real);
                        } else {
                            copyTensorForValidationLocal(backup, real, localDevice);
                        }
                    };
                    backupLinearState(*backupKey, *realKey);
                    backupLinearState(*backupValue, *realValue);
                }
                FastllmCudaSetDevice(localDevice);
                ForceDeviceSync();
            }, linearBackupErrors);
            for (auto &error : linearBackupErrors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            mtpProfileMark(mtpProfileCachePrepUs);

            bool tpInplaceRollbackArmed = true;
            auto restoreTpInplaceBase = [&]() {
                // A worker may have thrown before its normal ForceDeviceSync.
                // Drain every rank first so no late kernel can overwrite the
                // restored recurrent state after this function returns.
                int oldDevice = FastllmCudaGetDevice();
                try {
                    for (int localDevice : devices) {
                        FastllmCudaSetDevice(localDevice);
                        ForceDeviceSync();
                    }
                } catch (...) {
                    FastllmCudaSetDevice(oldDevice);
                    throw;
                }
                FastllmCudaSetDevice(oldDevice);
                for (int i = 0; i < block_cnt; i++) {
                    if (isAttentionLayerAt(i)) {
                        for (int localDevice : devices) {
                            int localHeads = tpAttentionLocalHeads(i, localDevice);
                            if (localHeads < 0) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace rollback lost the attention head scheme.");
                            }
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            if (localHeads == 0) {
                                if (!isValidEmptyTpAttentionLocal(realKey, localDevice) ||
                                    !isValidEmptyTpAttentionLocal(realValue, localDevice)) {
                                    throw std::runtime_error(
                                        "Qwen3.5 MTP TP inplace rollback found a mutated empty attention cache.");
                                }
                                continue;
                            }
                            auto keyMeta = baseLocalKeyMetas[i].find(localDevice);
                            auto valueMeta = baseLocalValueMetas[i].find(localDevice);
                            if (realKey == nullptr || realValue == nullptr ||
                                keyMeta == baseLocalKeyMetas[i].end() ||
                                valueMeta == baseLocalValueMetas[i].end()) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace rollback lost local attention metadata.");
                            }
                            restoreMeta(*realKey, keyMeta->second);
                            restoreMeta(*realValue, valueMeta->second);
                        }
                        assignMetaNoRelease(*pastKeyValues[i].first, baseRootKeyMetas[i]);
                        assignMetaNoRelease(*pastKeyValues[i].second, baseRootValueMetas[i]);
                    } else {
                        for (int localDevice : devices) {
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            Data *backupKey = getThreadTpLocalCache(
                                speculativeLinearStates[i][linearRollbackSlot].first,
                                localDevice);
                            Data *backupValue = getThreadTpLocalCache(
                                speculativeLinearStates[i][linearRollbackSlot].second,
                                localDevice);
                            if (realKey == nullptr || realValue == nullptr ||
                                backupKey == nullptr || backupValue == nullptr) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace rollback lost a linear base snapshot.");
                            }
                            copyTensorIntoExistingStorage(*realKey, *backupKey);
                            copyTensorIntoExistingStorage(*realValue, *backupValue);
                        }
                    }
                }
            };
            auto rollbackTpInplace = [&]() {
                if (!tpInplaceRollbackArmed) {
                    return;
                }
                try {
                    restoreTpInplaceBase();
                } catch (...) {
                    mtpCaches.erase(context);
                    throw;
                }
                tpInplaceRollbackArmed = false;
                mtpCaches.erase(context);
            };
            int tpInplaceRollbackToken = 0;
            auto tpInplaceRollbackDeleter = [&](int*) noexcept {
                if (!tpInplaceRollbackArmed) {
                    return;
                }
                try {
                    rollbackTpInplace();
                } catch (...) {
                    // Continuing with a partially rolled-back recurrent state
                    // would silently corrupt every following token.  The
                    // rollback uses preallocated same-shape CUDA buffers, so a
                    // failure here indicates an unrecoverable CUDA/runtime
                    // error and must be fail-stop.
                    std::terminate();
                }
            };
            std::unique_ptr<int, decltype(tpInplaceRollbackDeleter)> tpInplaceRollbackGuard(
                &tpInplaceRollbackToken, tpInplaceRollbackDeleter);

            speculativeCaptureFirstTokenLinearState = true;
            try {
                targetRet = runTargetWithPast(inputIds, attentionMask, positionIds, seqLens,
                                              pastKeyValues);
            } catch (...) {
                speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
                throw;
            }
            speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
            mtpProfileMark(mtpProfileTargetUs);
            if ((int)targetRet.size() < seqLen) {
                rollbackTpInplace();
                mtpProfileMark(mtpProfileRollbackUs);
                return false;
            }

            int draftTokenCount = seqLen - 1;
            int matchedDrafts = 0;
            while (matchedDrafts < draftTokenCount &&
                   targetRet[matchedDrafts] == tokenAt(matchedDrafts + 1)) {
                matchedDrafts++;
            }
            for (int i = 0; i < draftTokenCount && i < QWEN35_MTP_MAX_DRAFTS; i++) {
                mtpDraftPositionAttempts[i].fetch_add(1, std::memory_order_relaxed);
                if (matchedDrafts > i) {
                    mtpDraftPositionAccepts[i].fetch_add(1, std::memory_order_relaxed);
                }
            }
            mtpValidationCount.fetch_add(1, std::memory_order_relaxed);
            mtpProfileMark(mtpProfileMatchUs);

            int commitLen = matchedDrafts == draftTokenCount ? seqLen : matchedDrafts + 1;
            std::vector<int> committedRet(targetRet.begin(), targetRet.begin() + commitLen);
            if (matchedDrafts != draftTokenCount) {
                int captureSlot = commitLen - 1;
                std::vector<CacheMeta> prefixRootKeyMetas(block_cnt), prefixRootValueMetas(block_cnt);
                std::vector<std::map<int, CacheMeta> > prefixLocalKeyMetas(block_cnt), prefixLocalValueMetas(block_cnt);

                // Prepare and validate every prefix before mutating cache
                // ownership. This keeps a missing snapshot from producing a
                // partially committed TP state.
                bool prefixStateReady = true;
                std::string prefixStateError;
                for (int i = 0; i < block_cnt && prefixStateReady; i++) {
                    if (isAttentionLayerAt(i)) {
                        if (!tryPrefixTokenMetaFromTemp(
                                baseRootKeyMetas[i], *pastKeyValues[i].first,
                                commitLen, prefixRootKeyMetas[i], &prefixStateError) ||
                            !tryPrefixTokenMetaFromTemp(
                                baseRootValueMetas[i], *pastKeyValues[i].second,
                                commitLen, prefixRootValueMetas[i], &prefixStateError)) {
                            prefixStateReady = false;
                            break;
                        }
                        for (int localDevice : devices) {
                            int localHeads = tpAttentionLocalHeads(i, localDevice);
                            Data *localKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *localValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            if (localHeads < 0) {
                                prefixStateReady = false;
                                prefixStateError = "lost the attention head scheme";
                                break;
                            }
                            if (localHeads == 0) {
                                if (!isValidEmptyTpAttentionLocal(localKey, localDevice) ||
                                    !isValidEmptyTpAttentionLocal(localValue, localDevice)) {
                                    prefixStateReady = false;
                                    prefixStateError = "mutated an empty attention cache";
                                    break;
                                }
                                continue;
                            }
                            auto baseKeyMeta = baseLocalKeyMetas[i].find(localDevice);
                            auto baseValueMeta = baseLocalValueMetas[i].find(localDevice);
                            if (localKey == nullptr || localValue == nullptr ||
                                baseKeyMeta == baseLocalKeyMetas[i].end() ||
                                baseValueMeta == baseLocalValueMetas[i].end()) {
                                prefixStateReady = false;
                                prefixStateError = "lost local attention cache";
                                break;
                            }
                            CacheMeta prefixKeyMeta, prefixValueMeta;
                            if (!tryPrefixTokenMetaFromTemp(
                                    baseKeyMeta->second, *localKey, commitLen,
                                    prefixKeyMeta, &prefixStateError) ||
                                !tryPrefixTokenMetaFromTemp(
                                    baseValueMeta->second, *localValue, commitLen,
                                    prefixValueMeta, &prefixStateError)) {
                                prefixStateReady = false;
                                break;
                            }
                            prefixLocalKeyMetas[i].emplace(localDevice,
                                                           std::move(prefixKeyMeta));
                            prefixLocalValueMetas[i].emplace(localDevice,
                                                             std::move(prefixValueMeta));
                        }
                        continue;
                    }
                    if (i >= (int)speculativeLinearStates.size() ||
                        captureSlot < 0 ||
                        captureSlot >= (int)speculativeLinearStates[i].size()) {
                        prefixStateReady = false;
                        prefixStateError = "missing linear snapshot slot";
                        break;
                    }
                    for (int localDevice : devices) {
                        Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                               localDevice);
                        Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                 localDevice);
                        Data *snapshotKey = getThreadTpLocalCache(
                            speculativeLinearStates[i][captureSlot].first, localDevice);
                        Data *snapshotValue = getThreadTpLocalCache(
                            speculativeLinearStates[i][captureSlot].second, localDevice);
                        if (realKey == nullptr || realValue == nullptr ||
                            snapshotKey == nullptr || snapshotValue == nullptr ||
                            snapshotKey->dataDevice != DataDevice::CUDA ||
                            snapshotValue->dataDevice != DataDevice::CUDA ||
                            snapshotKey->cudaData == nullptr || snapshotValue->cudaData == nullptr ||
                            snapshotKey->cacheUid == LLONG_MIN ||
                            snapshotValue->cacheUid == LLONG_MIN ||
                            snapshotKey->dataType != realKey->dataType ||
                            snapshotValue->dataType != realValue->dataType ||
                            snapshotKey->dims != realKey->dims ||
                            snapshotValue->dims != realValue->dims) {
                            prefixStateReady = false;
                            prefixStateError = "incomplete rank-local linear snapshot";
                            break;
                        }
                    }
                }
                if (!prefixStateReady) {
                    rollbackTpInplace();
                    mtpProfileMark(mtpProfileRollbackUs);
                    return false;
                }

                for (int i = 0; i < block_cnt; i++) {
                    if (isAttentionLayerAt(i)) {
                        assignMetaNoRelease(*pastKeyValues[i].first, prefixRootKeyMetas[i]);
                        assignMetaNoRelease(*pastKeyValues[i].second, prefixRootValueMetas[i]);
                        for (int localDevice : devices) {
                            int localHeads = tpAttentionLocalHeads(i, localDevice);
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            if (localHeads < 0) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace validation restore lost the attention head scheme.");
                            }
                            if (localHeads == 0) {
                                if (!isValidEmptyTpAttentionLocal(realKey, localDevice) ||
                                    !isValidEmptyTpAttentionLocal(realValue, localDevice)) {
                                    throw std::runtime_error(
                                        "Qwen3.5 MTP TP inplace validation restore found a malformed empty attention cache.");
                                }
                                continue;
                            }
                            auto prefixKeyMeta = prefixLocalKeyMetas[i].find(localDevice);
                            auto prefixValueMeta = prefixLocalValueMetas[i].find(localDevice);
                            if (realKey == nullptr || realValue == nullptr ||
                                prefixKeyMeta == prefixLocalKeyMetas[i].end() ||
                                prefixValueMeta == prefixLocalValueMetas[i].end()) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace validation restore missing local attention cache.");
                            }
                            restoreMeta(*realKey, prefixKeyMeta->second);
                            restoreMeta(*realValue, prefixValueMeta->second);
                        }
                    } else {
                        for (int localDevice : devices) {
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            Data *snapshotKey = getThreadTpLocalCache(
                                speculativeLinearStates[i][captureSlot].first,
                                localDevice);
                            Data *snapshotValue = getThreadTpLocalCache(
                                speculativeLinearStates[i][captureSlot].second,
                                localDevice);
                            if (realKey == nullptr || realValue == nullptr ||
                                snapshotKey == nullptr || snapshotValue == nullptr) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP inplace validation lost a linear snapshot.");
                            }
                            adoptTensorIntoExistingStorage(*realKey, *snapshotKey);
                            adoptTensorIntoExistingStorage(*realValue, *snapshotValue);
                        }
                    }
                }
                mtpProfileMark(mtpProfileRollbackUs);
            } else {
                mtpProfileMark(mtpProfileCommitUs);
            }

            Data hiddenForMtp;
            Split(speculativeHiddenStates, 1, 0, commitLen, hiddenForMtp);
            Data mtpPositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, commitLen, 0);
            std::vector<int> mtpInputTokens;
            mtpInputTokens.reserve(commitLen);
            for (int i = 1; i < commitLen; i++) {
                mtpInputTokens.push_back(tokenAt(i));
            }
            mtpInputTokens.push_back(committedRet[commitLen - 1]);
            std::vector<int> drafts = runMtpDraftChain(
                hiddenForMtp, mtpInputTokens, mtpPositionIds,
                commitLen - 1, commitLen - 1);
            mtpProfileMark(mtpProfileDraftUs);
            acceptedTokens[0].assign(committedRet.begin(),
                                     committedRet.begin() + commitLen);
            setNextInputWithDrafts(committedRet[commitLen - 1], drafts);
            keptInputLens[0] = commitLen;
            logMtpStats();
            mtpProfileRecord(QWEN35_MTP_PROFILE_TP_INPLACE, true,
                             draftTokenCount, matchedDrafts, commitLen);
            tpInplaceRollbackArmed = false;
            return true;
        }

        if (tensorParallel) {
            std::vector<std::pair<Data, Data> > validationPastStorage(block_cnt);
            std::vector<std::pair<Data*, Data*> > validationPastKeyValues(block_cnt);
            std::vector<std::map<int, CacheMeta> > baseLocalKeyMetas(block_cnt), baseLocalValueMetas(block_cnt);
            auto cleanupTpValidationPaged = [&](const std::vector<std::map<int, CacheMeta> > &keepKeyMetas,
                                                const std::vector<std::map<int, CacheMeta> > &keepValueMetas) {
                for (int i = 0; i < block_cnt; i++) {
                    if (!isAttentionLayerAt(i)) {
                        continue;
                    }
                    detachPagedCacheView(validationPastStorage[i].first);
                    detachPagedCacheView(validationPastStorage[i].second);
                    for (int localDevice : devices) {
                        int localHeads = tpAttentionLocalHeads(i, localDevice);
                        if (localHeads == 0) {
                            continue;
                        }
                        Data *localKey = getThreadTpLocalCache(validationPastStorage[i].first,
                                                               localDevice);
                        Data *localValue = getThreadTpLocalCache(validationPastStorage[i].second,
                                                                 localDevice);
                        auto keyIt = keepKeyMetas[i].find(localDevice);
                        auto valueIt = keepValueMetas[i].find(localDevice);
                        if (localKey != nullptr && keyIt != keepKeyMetas[i].end()) {
                            releaseUncommittedPagedPages(*localKey, keyIt->second);
                        } else if (localKey != nullptr) {
                            detachPagedCacheView(*localKey);
                        }
                        if (localValue != nullptr && valueIt != keepValueMetas[i].end()) {
                            releaseUncommittedPagedPages(*localValue, valueIt->second);
                        } else if (localValue != nullptr) {
                            detachPagedCacheView(*localValue);
                        }
                    }
                }
            };
            bool tpBaseCleanupArmed = false;
            int tpBaseCleanupToken = 0;
            auto tpBaseCleanupDeleter = [&](int*) {
                if (tpBaseCleanupArmed) {
                    cleanupTpValidationPaged(baseLocalKeyMetas, baseLocalValueMetas);
                    mtpCaches.erase(context);
                }
            };
            std::unique_ptr<int, decltype(tpBaseCleanupDeleter)> tpBaseCleanupGuard(
                &tpBaseCleanupToken, tpBaseCleanupDeleter);
            tpBaseCleanupArmed = true;
            auto prepareTpValidationCache = [&](int layer, Data &dstRoot, Data &srcRoot,
                                                bool paged,
                                                std::map<int, CacheMeta> &baseLocalMetas,
                                                const char *cacheKind) {
                copyTensorParallelRootMeta(dstRoot, srcRoot);
                dstRoot.multiDeviceData = true;
                dstRoot.dataDevice = DataDevice::CUDA;
                dstRoot.dataDeviceIds = devices;
                for (int localDevice : devices) {
                    Data *srcLocal = getThreadTpLocalCache(srcRoot, localDevice);
                    AssertInFastLLM(srcLocal != nullptr,
                                    "Qwen3.5 MTP TP validation missing local cache.\n");
                    auto slot = dstRoot.multiDeviceDatas.emplace(localDevice, nullptr);
                    if (!slot.second) {
                        throw std::runtime_error(
                            "Qwen3.5 MTP TP validation got duplicate CUDA device.");
                    }
                    std::unique_ptr<Data> dstLocal(new Data());
                    if (paged) {
                        int localHeads = tpAttentionLocalHeads(layer, localDevice);
                        AssertInFastLLM(localHeads >= 0,
                                        "Qwen3.5 MTP TP validation has an invalid attention head scheme.\n");
                        if (localHeads == 0) {
                            AssertInFastLLM(
                                isValidEmptyTpAttentionLocal(srcLocal, localDevice),
                                "Qwen3.5 MTP TP validation got a malformed empty attention cache.\n");
                            initializeEmptyTpAttentionLocal(*dstLocal, *srcLocal,
                                                            localDevice);
                        } else {
                            if (!isValidPagedTpAttentionLocal(srcLocal, localDevice, localHeads)) {
                                throw std::runtime_error(
                                    "Qwen3.5 MTP TP validation got invalid local attention " +
                                    std::string(cacheKind) + " cache metadata at layer " +
                                    std::to_string(layer) + ", device " +
                                    std::to_string(localDevice) + ", expectedHeads=" +
                                    std::to_string(localHeads) + ": " +
                                    describeTpAttentionLocal(srcLocal));
                            }
                            baseLocalMetas[localDevice] = makeMeta(*srcLocal);
                            copyPagedCacheForValidation(*dstLocal, *srcLocal);
                        }
                    } else {
                        baseLocalMetas[localDevice] = makeMeta(*srcLocal);
                        copyTensorForValidationLocal(*dstLocal, *srcLocal, localDevice);
                    }
                    slot.first->second = dstLocal.release();
                }
            };
            bool oldCaptureFirstTokenLinearState = speculativeCaptureFirstTokenLinearState;
            try {
                for (int i = 0; i < block_cnt; i++) {
                    Data *realKey = pastKeyValues[i].first;
                    Data *realValue = pastKeyValues[i].second;
                    AssertInFastLLM(realKey != nullptr && realValue != nullptr,
                                    "Qwen3.5 MTP TP validation got null cache.\n");
                    bool paged = isAttentionLayerAt(i);
                    prepareTpValidationCache(i, validationPastStorage[i].first, *realKey,
                                             paged, baseLocalKeyMetas[i], "key");
                    prepareTpValidationCache(i, validationPastStorage[i].second, *realValue,
                                             paged, baseLocalValueMetas[i], "value");
                    validationPastKeyValues[i] = {
                        &validationPastStorage[i].first,
                        &validationPastStorage[i].second
                    };
                }
                mtpProfileMark(mtpProfileCachePrepUs);
                // This flag also selects the cached multi-token linear-attention
                // decode path. No capture slots are allocated here because TP
                // rollback uses the temporary validation cache instead.
                speculativeCaptureFirstTokenLinearState = true;
                targetRet = runTargetWithPast(inputIds, attentionMask, positionIds, seqLens,
                                              validationPastKeyValues);
            } catch (...) {
                speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
                cleanupTpValidationPaged(baseLocalKeyMetas, baseLocalValueMetas);
                tpBaseCleanupArmed = false;
                mtpCaches.erase(context);
                throw;
            }
            speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
            mtpProfileMark(mtpProfileTargetUs);
            if ((int)targetRet.size() < seqLen) {
                cleanupTpValidationPaged(baseLocalKeyMetas, baseLocalValueMetas);
                tpBaseCleanupArmed = false;
                mtpCaches.erase(context);
                return false;
            }

            int draftTokenCount = seqLen - 1;
            int matchedDrafts = 0;
            while (matchedDrafts < draftTokenCount &&
                   targetRet[matchedDrafts] == tokenAt(matchedDrafts + 1)) {
                matchedDrafts++;
            }
            for (int i = 0; i < draftTokenCount && i < QWEN35_MTP_MAX_DRAFTS; i++) {
                mtpDraftPositionAttempts[i].fetch_add(1, std::memory_order_relaxed);
                if (matchedDrafts > i) {
                    mtpDraftPositionAccepts[i].fetch_add(1, std::memory_order_relaxed);
                }
            }
            mtpProfileMark(mtpProfileMatchUs);

            mtpValidationCount.fetch_add(1, std::memory_order_relaxed);
            int commitLen = matchedDrafts == draftTokenCount ? seqLen : matchedDrafts + 1;
            std::vector<int> committedRet;
            if (matchedDrafts == draftTokenCount) {
                std::vector<CacheMeta> finalRootKeyMetas(block_cnt), finalRootValueMetas(block_cnt);
                std::vector<std::map<int, CacheMeta> > finalLocalKeyMetas(block_cnt), finalLocalValueMetas(block_cnt);
                try {
                    for (int i = 0; i < block_cnt; i++) {
                        finalRootKeyMetas[i] = makeMeta(validationPastStorage[i].first);
                        finalRootValueMetas[i] = makeMeta(validationPastStorage[i].second);
                        for (int localDevice : devices) {
                            Data *localKey = getThreadTpLocalCache(validationPastStorage[i].first,
                                                                   localDevice);
                            Data *localValue = getThreadTpLocalCache(validationPastStorage[i].second,
                                                                     localDevice);
                            if (isAttentionLayerAt(i)) {
                                int localHeads = tpAttentionLocalHeads(i, localDevice);
                                AssertInFastLLM(localHeads >= 0,
                                                "Qwen3.5 MTP TP validation lost the attention head scheme.\n");
                                if (localHeads == 0) {
                                    AssertInFastLLM(
                                        isValidEmptyTpAttentionLocal(localKey, localDevice) &&
                                        isValidEmptyTpAttentionLocal(localValue, localDevice),
                                        "Qwen3.5 MTP TP validation mutated an empty attention cache.\n");
                                    continue;
                                }
                                AssertInFastLLM(
                                    isValidPagedTpAttentionLocal(localKey, localDevice, localHeads) &&
                                    isValidPagedTpAttentionLocal(localValue, localDevice, localHeads),
                                    "Qwen3.5 MTP TP validation produced invalid local attention metadata.\n");
                            }
                            AssertInFastLLM(localKey != nullptr && localValue != nullptr,
                                            "Qwen3.5 MTP TP validation lost local cache.\n");
                            finalLocalKeyMetas[i][localDevice] = makeMeta(*localKey);
                            finalLocalValueMetas[i][localDevice] = makeMeta(*localValue);
                        }
                    }
                } catch (...) {
                    cleanupTpValidationPaged(baseLocalKeyMetas, baseLocalValueMetas);
                    tpBaseCleanupArmed = false;
                    mtpCaches.erase(context);
                    throw;
                }
                std::vector<std::vector<uint8_t> > tpFinalKeyTransferred(
                    block_cnt, std::vector<uint8_t>(devices.size(), 0));
                std::vector<std::vector<uint8_t> > tpFinalValueTransferred(
                    block_cnt, std::vector<uint8_t>(devices.size(), 0));
                auto cleanupTpCommittedValidationPaged = [&]() {
                    for (int i = 0; i < block_cnt; i++) {
                        if (!isAttentionLayerAt(i)) {
                            continue;
                        }
                        detachPagedCacheView(validationPastStorage[i].first);
                        detachPagedCacheView(validationPastStorage[i].second);
                        for (int deviceIndex = 0;
                             deviceIndex < (int)devices.size(); deviceIndex++) {
                            int localDevice = devices[deviceIndex];
                            if (tpAttentionLocalHeads(i, localDevice) == 0) {
                                continue;
                            }
                            Data *localKey = getThreadTpLocalCache(
                                validationPastStorage[i].first, localDevice);
                            Data *localValue = getThreadTpLocalCache(
                                validationPastStorage[i].second, localDevice);
                            const auto &keyMetas = tpFinalKeyTransferred[i][deviceIndex] ?
                                finalLocalKeyMetas[i] : baseLocalKeyMetas[i];
                            const auto &valueMetas = tpFinalValueTransferred[i][deviceIndex] ?
                                finalLocalValueMetas[i] : baseLocalValueMetas[i];
                            auto keyIt = keyMetas.find(localDevice);
                            auto valueIt = valueMetas.find(localDevice);
                            if (localKey != nullptr && keyIt != keyMetas.end()) {
                                releaseUncommittedPagedPages(*localKey, keyIt->second);
                            } else if (localKey != nullptr) {
                                detachPagedCacheView(*localKey);
                            }
                            if (localValue != nullptr && valueIt != valueMetas.end()) {
                                releaseUncommittedPagedPages(*localValue, valueIt->second);
                            } else if (localValue != nullptr) {
                                detachPagedCacheView(*localValue);
                            }
                        }
                    }
                };
                bool tpFinalCleanupArmed = true;
                int tpFinalCleanupToken = 0;
                auto tpFinalCleanupDeleter = [&](int*) {
                    if (tpFinalCleanupArmed) {
                        cleanupTpCommittedValidationPaged();
                    }
                };
                std::unique_ptr<int, decltype(tpFinalCleanupDeleter)> tpFinalCleanupGuard(
                    &tpFinalCleanupToken, tpFinalCleanupDeleter);
                for (int i = 0; i < block_cnt; i++) {
                    if (isAttentionLayerAt(i)) {
                        for (int deviceIndex = 0;
                             deviceIndex < (int)devices.size(); deviceIndex++) {
                            int localDevice = devices[deviceIndex];
                            int localHeads = tpAttentionLocalHeads(i, localDevice);
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            AssertInFastLLM(localHeads >= 0,
                                            "Qwen3.5 MTP TP validation commit lost the attention head scheme.\n");
                            if (localHeads == 0) {
                                AssertInFastLLM(
                                    isValidEmptyTpAttentionLocal(realKey, localDevice) &&
                                    isValidEmptyTpAttentionLocal(realValue, localDevice),
                                    "Qwen3.5 MTP TP validation commit found a malformed empty attention cache.\n");
                                continue;
                            }
                            auto finalKeyMeta = finalLocalKeyMetas[i].find(localDevice);
                            auto finalValueMeta = finalLocalValueMetas[i].find(localDevice);
                            AssertInFastLLM(realKey != nullptr && realValue != nullptr &&
                                            finalKeyMeta != finalLocalKeyMetas[i].end() &&
                                            finalValueMeta != finalLocalValueMetas[i].end(),
                                            "Qwen3.5 MTP TP validation commit missing local attention cache.\n");
                            restoreMeta(*realKey, finalKeyMeta->second);
                            tpFinalKeyTransferred[i][deviceIndex] = 1;
                            restoreMeta(*realValue, finalValueMeta->second);
                            tpFinalValueTransferred[i][deviceIndex] = 1;
                        }
                        installPreparedMetaNoRelease(
                            *pastKeyValues[i].first, finalRootKeyMetas[i]);
                        installPreparedMetaNoRelease(
                            *pastKeyValues[i].second, finalRootValueMetas[i]);
                    } else {
                        copyTensorParallelRootMeta(*pastKeyValues[i].first,
                                                   validationPastStorage[i].first);
                        copyTensorParallelRootMeta(*pastKeyValues[i].second,
                                                   validationPastStorage[i].second);
                        for (int localDevice : devices) {
                            Data *realKey = getThreadTpLocalCache(*pastKeyValues[i].first,
                                                                   localDevice);
                            Data *realValue = getThreadTpLocalCache(*pastKeyValues[i].second,
                                                                     localDevice);
                            Data *tempKey = getThreadTpLocalCache(validationPastStorage[i].first,
                                                                  localDevice);
                            Data *tempValue = getThreadTpLocalCache(validationPastStorage[i].second,
                                                                    localDevice);
                            AssertInFastLLM(realKey != nullptr && realValue != nullptr &&
                                            tempKey != nullptr && tempValue != nullptr,
                                            "Qwen3.5 MTP TP validation commit missing local linear cache.\n");
                            adoptTensorIntoExistingStorage(*realKey, *tempKey);
                            adoptTensorIntoExistingStorage(*realValue, *tempValue);
                        }
                    }
                }
                cleanupTpCommittedValidationPaged();
                tpFinalCleanupArmed = false;
                committedRet.assign(targetRet.begin(), targetRet.begin() + commitLen);
                mtpProfileMark(mtpProfileCommitUs);
            } else {
                cleanupTpValidationPaged(baseLocalKeyMetas, baseLocalValueMetas);
                mtpProfileMark(mtpProfileRollbackUs);
                committedRet.assign(targetRet.begin(), targetRet.begin() + commitLen);
                Data singleInputIds = buildInputIdsSlice(0, commitLen);
                Data singlePositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, commitLen, 0);
                std::vector<Data*> singleAttentionMask = {nullptr};
                std::vector<Data*> singlePositionIdVec = {&singlePositionIds};
                std::vector<int> singleSeqLens = {commitLen};
                bool oldCaptureFirstTokenLinearState = speculativeCaptureFirstTokenLinearState;
                speculativeCaptureFirstTokenLinearState = true;
                try {
                    runTargetCacheOnly(singleInputIds, singleAttentionMask,
                                       singlePositionIdVec, singleSeqLens);
                } catch (...) {
                    speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
                    mtpCaches.erase(context);
                    throw;
                }
                speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
                mtpProfileMark(mtpProfileRetryUs);
            }
            Data hiddenForMtp;
            Split(speculativeHiddenStates, 1, 0, commitLen, hiddenForMtp);
            Data mtpPositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, commitLen, 0);
            std::vector<int> mtpInputTokens;
            mtpInputTokens.reserve(commitLen);
            for (int i = 1; i < commitLen; i++) {
                mtpInputTokens.push_back(tokenAt(i));
            }
            mtpInputTokens.push_back(committedRet[commitLen - 1]);
            std::vector<int> drafts = runMtpDraftChain(
                hiddenForMtp, mtpInputTokens, mtpPositionIds,
                commitLen - 1, commitLen - 1);
            mtpProfileMark(mtpProfileDraftUs);
            acceptedTokens[0].assign(committedRet.begin(),
                                     committedRet.begin() + commitLen);
            setNextInputWithDrafts(committedRet[commitLen - 1], drafts);
            keptInputLens[0] = commitLen;
            logMtpStats();
            mtpProfileRecord(QWEN35_MTP_PROFILE_TP_COPY, true,
                             draftTokenCount, matchedDrafts, commitLen);
            tpBaseCleanupArmed = false;
            return true;
        }

        std::vector<std::pair<Data, Data> > validationPastStorage(block_cnt);
        std::vector<std::pair<Data*, Data*> > validationPastKeyValues(block_cnt);
        std::vector<CacheMeta> baseKeyMetas(block_cnt), baseValueMetas(block_cnt);
        std::vector<std::vector<CacheMeta> > prefixKeyMetas(
            seqLen + 1, std::vector<CacheMeta>(block_cnt));
        std::vector<std::vector<CacheMeta> > prefixValueMetas(
            seqLen + 1, std::vector<CacheMeta>(block_cnt));
        auto releaseValidationPagedViews = [&](const std::vector<CacheMeta> &keepKeyMetas,
                                               const std::vector<CacheMeta> &keepValueMetas) {
            for (int i = 0; i < block_cnt; i++) {
                if (!isAttentionLayerAt(i)) {
                    continue;
                }
                releaseUncommittedPagedPages(validationPastStorage[i].first, keepKeyMetas[i]);
                releaseUncommittedPagedPages(validationPastStorage[i].second, keepValueMetas[i]);
            }
        };
        bool oldCaptureFirstTokenLinearState = speculativeCaptureFirstTokenLinearState;
        int oldLinearStateCaptureSlots = speculativeLinearStateCaptureSlots;
        auto clearSpeculativeLinearCapture = [&]() {
            speculativeLinearStateCaptureSlots = oldLinearStateCaptureSlots;
            speculativeLinearStates.clear();
            speculativeLinearCaptureMask.clear();
        };
        std::vector<uint8_t> singleKeyUsesPrefix(block_cnt, 0);
        std::vector<uint8_t> singleValueUsesPrefix(block_cnt, 0);
        int singleCleanupPrefixTokens = 0;
        auto cleanupSingleValidationPaged = [&]() {
            for (int i = 0; i < block_cnt; i++) {
                if (!isAttentionLayerAt(i)) {
                    continue;
                }
                const CacheMeta &keepKey = singleKeyUsesPrefix[i] ?
                    prefixKeyMetas[singleCleanupPrefixTokens][i] : baseKeyMetas[i];
                const CacheMeta &keepValue = singleValueUsesPrefix[i] ?
                    prefixValueMetas[singleCleanupPrefixTokens][i] : baseValueMetas[i];
                releaseUncommittedPagedPages(validationPastStorage[i].first, keepKey);
                releaseUncommittedPagedPages(validationPastStorage[i].second, keepValue);
            }
        };
        bool singleCleanupArmed = true;
        int singleCleanupToken = 0;
        auto singleCleanupDeleter = [&](int*) {
            if (singleCleanupArmed) {
                speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
                cleanupSingleValidationPaged();
                clearSpeculativeLinearCapture();
                mtpCaches.erase(context);
            }
        };
        std::unique_ptr<int, decltype(singleCleanupDeleter)> singleCleanupGuard(
            &singleCleanupToken, singleCleanupDeleter);
        try {
            for (int i = 0; i < block_cnt; i++) {
                Data *realKey = pastKeyValues[i].first;
                Data *realValue = pastKeyValues[i].second;
                AssertInFastLLM(realKey != nullptr && realValue != nullptr,
                                "Qwen3.5 MTP validation got null cache.\n");
                if (isAttentionLayerAt(i)) {
                    baseKeyMetas[i] = makeMeta(*realKey);
                    baseValueMetas[i] = makeMeta(*realValue);
                    copyPagedCacheForValidation(validationPastStorage[i].first, *realKey);
                    copyPagedCacheForValidation(validationPastStorage[i].second, *realValue);
                } else {
                    validationPastStorage[i].first.CopyFrom(*realKey);
                    validationPastStorage[i].second.CopyFrom(*realValue);
                }
                validationPastKeyValues[i] = {
                    &validationPastStorage[i].first,
                    &validationPastStorage[i].second
                };
            }
        } catch (...) {
            speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            mtpCaches.erase(context);
            throw;
        }
        prefixKeyMetas[0] = baseKeyMetas;
        prefixValueMetas[0] = baseValueMetas;
        mtpProfileMark(mtpProfileCachePrepUs);
        try {
            speculativeCaptureFirstTokenLinearState = true;
            int linearCaptureSlots = std::max(1, seqLen - 1);
            speculativeLinearStateCaptureSlots = linearCaptureSlots;
            speculativeLinearStates.clear();
            speculativeLinearStates.resize(block_cnt);
            speculativeLinearCaptureMask.clear();
            speculativeLinearCaptureMask.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                if (isAttentionLayerAt(i)) {
                    continue;
                }
                speculativeLinearStates[i].resize(linearCaptureSlots);
                speculativeLinearCaptureMask[i].assign(linearCaptureSlots, 0);
            }
            speculativeFirstTokenLinearStates.clear();
            speculativeFirstTokenLinearStates.resize(block_cnt);
            speculativeFirstTokenLinearCaptureMask.assign(block_cnt, 0);
            targetRet = runTargetWithPast(inputIds, attentionMask, positionIds, seqLens,
                                          validationPastKeyValues);
        } catch (...) {
            speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            mtpCaches.erase(context);
            throw;
        }
        speculativeCaptureFirstTokenLinearState = oldCaptureFirstTokenLinearState;
        mtpProfileMark(mtpProfileTargetUs);
        if ((int)targetRet.size() < seqLen) {
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            mtpCaches.erase(context);
            return false;
        }
        bool prefixMetaOk = true;
        std::string prefixMetaError;
        try {
            for (int i = 0; i < block_cnt; i++) {
                if (!isAttentionLayerAt(i)) {
                    continue;
                }
                for (int tokens = 1; tokens <= seqLen; tokens++) {
                    if (!tryPrefixTokenMetaFromTemp(baseKeyMetas[i],
                                                    validationPastStorage[i].first,
                                                    tokens,
                                                    prefixKeyMetas[tokens][i],
                                                    &prefixMetaError) ||
                        !tryPrefixTokenMetaFromTemp(baseValueMetas[i],
                                                    validationPastStorage[i].second,
                                                    tokens,
                                                    prefixValueMetas[tokens][i],
                                                    &prefixMetaError)) {
                        prefixMetaOk = false;
                        break;
                    }
                }
                if (!prefixMetaOk) {
                    break;
                }
            }
        } catch (...) {
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            mtpCaches.erase(context);
            throw;
        }
        if (!prefixMetaOk) {
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            // The caller will advance the target cache through the non-MTP
            // fallback. Drop the now-stale draft cache so it cannot be reused
            // with a one-token offset on the next decode step.
            mtpCaches.erase(context);
            return false;
        }
        auto canCommitValidationCachePrefix = [&](int tokens) {
            if (tokens == seqLen) {
                return true;
            }
            if (tokens <= 0 || tokens > seqLen) {
                return false;
            }
            int slot = tokens - 1;
            for (int i = 0; i < block_cnt; i++) {
                if (isAttentionLayerAt(i)) {
                    continue;
                }
                if (i >= (int)speculativeLinearCaptureMask.size() ||
                    slot >= (int)speculativeLinearCaptureMask[i].size() ||
                    (speculativeLinearCaptureMask[i][slot] & 3) != 3) {
                    return false;
                }
            }
            return true;
        };
        auto commitValidationCachePrefix = [&](int tokens) {
            AssertInFastLLM(tokens > 0 && tokens <= seqLen,
                            "Qwen3.5 MTP validation got invalid commit prefix.\n");
            bool useValidationLinearFinal = tokens == seqLen;
            int slot = tokens - 1;
            for (int i = 0; i < block_cnt; i++) {
                if (isAttentionLayerAt(i)) {
                    restoreMeta(*pastKeyValues[i].first, prefixKeyMetas[tokens][i]);
                    singleKeyUsesPrefix[i] = 1;
                    restoreMeta(*pastKeyValues[i].second, prefixValueMetas[tokens][i]);
                    singleValueUsesPrefix[i] = 1;
                } else if (useValidationLinearFinal) {
                    copyTensorIntoExistingStorage(*pastKeyValues[i].first,
                                                  validationPastStorage[i].first);
                    copyTensorIntoExistingStorage(*pastKeyValues[i].second,
                                                  validationPastStorage[i].second);
                } else {
                    AssertInFastLLM(i < (int)speculativeLinearStates.size() &&
                                    slot < (int)speculativeLinearStates[i].size(),
                                    "Qwen3.5 MTP validation missing linear state slot.\n");
                    copyTensorIntoExistingStorage(*pastKeyValues[i].first,
                                                  speculativeLinearStates[i][slot].first);
                    copyTensorIntoExistingStorage(*pastKeyValues[i].second,
                                                  speculativeLinearStates[i][slot].second);
                }
            }
            releaseValidationPagedViews(prefixKeyMetas[tokens], prefixValueMetas[tokens]);
        };
        int draftTokenCount = seqLen - 1;
        int matchedDrafts = 0;
        while (matchedDrafts < draftTokenCount &&
               targetRet[matchedDrafts] == tokenAt(matchedDrafts + 1)) {
            matchedDrafts++;
        }
        for (int i = 0; i < draftTokenCount && i < QWEN35_MTP_MAX_DRAFTS; i++) {
            mtpDraftPositionAttempts[i].fetch_add(1, std::memory_order_relaxed);
            if (matchedDrafts > i) {
                mtpDraftPositionAccepts[i].fetch_add(1, std::memory_order_relaxed);
            }
        }
        mtpProfileMark(mtpProfileMatchUs);
        if (matchedDrafts == draftTokenCount) {
            singleCleanupPrefixTokens = seqLen;
            commitValidationCachePrefix(seqLen);
            mtpProfileMark(mtpProfileCommitUs);
            std::vector<int> mtpInputTokens;
            mtpInputTokens.reserve(seqLen);
            for (int i = 1; i < seqLen; i++) {
                mtpInputTokens.push_back(tokenAt(i));
            }
            mtpInputTokens.push_back(targetRet[seqLen - 1]);
            Data mtpPositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, seqLen, 0);
            std::vector<int> drafts = runMtpDraftChain(
                speculativeHiddenStates, mtpInputTokens, mtpPositionIds,
                seqLen - 1, seqLen - 1);
            mtpProfileMark(mtpProfileDraftUs);
            mtpValidationCount.fetch_add(1, std::memory_order_relaxed);
            acceptedTokens[0].assign(targetRet.begin(), targetRet.begin() + seqLen);
            setNextInputWithDrafts(targetRet[seqLen - 1], drafts);
            keptInputLens[0] = seqLen;
            logMtpStats();
            mtpProfileRecord(QWEN35_MTP_PROFILE_SINGLE, true,
                             draftTokenCount, matchedDrafts, seqLen);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            return true;
        }

        mtpValidationCount.fetch_add(1, std::memory_order_relaxed);
        int commitLen = matchedDrafts + 1;
        std::vector<int> committedRet(targetRet.begin(), targetRet.begin() + commitLen);
        if (!canCommitValidationCachePrefix(commitLen)) {
            releaseValidationPagedViews(baseKeyMetas, baseValueMetas);
            clearSpeculativeLinearCapture();
            singleCleanupArmed = false;
            // The validation cache is still isolated, so fall back before
            // touching the real target cache. Its MTP draft cache must be
            // discarded because the caller will advance only the target model.
            mtpCaches.erase(context);
            return false;
        }
        singleCleanupPrefixTokens = commitLen;
        commitValidationCachePrefix(commitLen);
        mtpProfileMark(mtpProfileRollbackUs);
        Data hiddenForMtp;
        Split(speculativeHiddenStates, 1, 0, commitLen, hiddenForMtp);
        Data mtpPositionIds = BuildMtpPositionIdsSlice(allPositionIds, 0, commitLen, 0);
        std::vector<int> mtpInputTokens;
        mtpInputTokens.reserve(commitLen);
        for (int i = 1; i < commitLen; i++) {
            mtpInputTokens.push_back(tokenAt(i));
        }
        mtpInputTokens.push_back(committedRet[commitLen - 1]);
        std::vector<int> drafts = runMtpDraftChain(
            hiddenForMtp, mtpInputTokens, mtpPositionIds,
            commitLen - 1, commitLen - 1);
        mtpProfileMark(mtpProfileDraftUs);
        acceptedTokens[0].assign(committedRet.begin(), committedRet.begin() + commitLen);
        setNextInputWithDrafts(committedRet[commitLen - 1], drafts);
        keptInputLens[0] = commitLen;
        logMtpStats();
        mtpProfileRecord(QWEN35_MTP_PROFILE_SINGLE, true,
                         draftTokenCount, matchedDrafts, commitLen);
        clearSpeculativeLinearCapture();
        singleCleanupArmed = false;
        return true;
#endif
    }

    void Qwen3_5Model::Qwen35MTPLoop() {
#ifndef USE_CUDA
        NewMainLoop();
#else
        Qwen3_5Model *model = this;
        int maxTotalLens = 0;
        int totalPages = 0;
        int pagesLimit = 0;
        int pageLen = fastllm::GetPageLen();

        const int mtpSchedulerLanes = 1;
        int prefillChunkSize = model->GetChunkedPrefillSize();

        auto releasePagedCachePages = [](Data &cache, bool clearDims = false) {
            std::set<std::pair<PagedCacheManager*, int> > releasedPages;
            auto releaseUnique = [&](Data &pagedCache) {
                if (!pagedCache.isPagedKVCache || pagedCache.pagedKVCacheData == nullptr ||
                    pagedCache.pageIndex.empty()) {
                    return;
                }
                std::vector<int> uniquePages;
                for (int page : pagedCache.pageIndex) {
                    std::pair<PagedCacheManager*, int> key = {pagedCache.pagedKVCacheData, page};
                    if (releasedPages.insert(key).second) {
                        uniquePages.push_back(page);
                    }
                }
                if (!uniquePages.empty()) {
                    pagedCache.pagedKVCacheData->ReleasePageIndices(uniquePages);
                }
                pagedCache.pageIndex.clear();
                pagedCache.lastPageLen = 0;
            };
            releaseUnique(cache);
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        releaseUnique(*it.second);
                        if (clearDims) {
                            it.second->dims.clear();
                        }
                    }
                }
            }
            if (clearDims) {
                cache.dims.clear();
            }
        };

        auto eraseMtpCache = [&](ResponseContext *ctx) {
            if (ctx == nullptr) {
                return;
            }
            std::lock_guard<std::mutex> guard(model->mtpCacheMutex);
            model->mtpCaches.erase(ctx);
        };

        auto releaseAndReinitRequest = [&](ResponseContext *ctx) {
            if (ctx == nullptr) {
                return;
            }
            for (int i = 0; i < model->block_cnt && i < (int)ctx->pastKeyValues.size(); i++) {
                releasePagedCachePages(ctx->pastKeyValues[i].first, true);
                releasePagedCachePages(ctx->pastKeyValues[i].second, true);
            }
            eraseMtpCache(ctx);
            ctx->currentTokens = ctx->allTokens;
            ctx->preTokens = 0;
            ctx->cacheLen = 0;
            ctx->intParams.clear();
        };

        auto getPagedManagerFromCache = [](Data &cache) -> PagedCacheManager* {
            if (cache.multiDeviceData) {
                for (auto &it : cache.multiDeviceDatas) {
                    if (it.second != nullptr && it.second->pagedKVCacheData != nullptr) {
                        return it.second->pagedKVCacheData;
                    }
                }
            }
            return cache.pagedKVCacheData;
        };

        auto findRuntimePagedManager = [&]() -> PagedCacheManager* {
            if (model->kvCacheId >= 0) {
                for (auto &it : model->responseContextDict.dicts) {
                    ResponseContext *ctx = it.second;
                    if (ctx == nullptr || model->kvCacheId >= (int)ctx->pastKeyValues.size()) {
                        continue;
                    }
                    PagedCacheManager *manager =
                        getPagedManagerFromCache(ctx->pastKeyValues[model->kvCacheId].first);
                    if (manager != nullptr) {
                        return manager;
                    }
                }
            }
            return GetPagedCacheManager(model->kvCacheId * 2);
        };

        auto canBudgetForTpInplace = [&](ResponseContext *ctx, int decodeTokens) {
            if (ctx == nullptr || decodeTokens < 2 ||
                decodeTokens > QWEN35_MTP_FAST_SEQ_MAX ||
                (int)ctx->pastKeyValues.size() < model->block_cnt) {
                return false;
            }
            std::vector<int> devices;
            std::map<int, int> ratios;
            if (!GetQwen35GPUForwardDevices(model->deviceMap, devices, ratios) ||
                devices.size() <= 1) {
                return false;
            }
            auto getLocal = [](Data &root, int device) -> Data* {
                auto it = root.multiDeviceDatas.find(device);
                return it == root.multiDeviceDatas.end() ? nullptr : it->second;
            };
            auto hasDenseStrides = [](const Data &data) {
                if (data.dims.empty() || data.strides.size() != data.dims.size()) {
                    return false;
                }
                uint64_t expected = 1;
                for (int i = (int)data.dims.size() - 1; i >= 0; i--) {
                    if (data.dims[i] < 0 || data.strides[i] != expected) {
                        return false;
                    }
                    expected *= (uint64_t)data.dims[i];
                }
                return true;
            };
            auto localAttentionHeads = [&](int layer, int device) {
                if (layer < 0 ||
                    layer >= (int)model->threadTpAttentionKVHeadSchemes.size()) {
                    return -1;
                }
                const DivisionScheme &scheme =
                    model->threadTpAttentionKVHeadSchemes[layer];
                auto schemeIt = scheme.find(device);
                if (schemeIt == scheme.end()) {
                    return -1;
                }
                int heads = 0;
                for (const auto &range : schemeIt->second) {
                    if (range.first < 0 || range.second < range.first ||
                        range.second > model->num_key_value_heads) {
                        return -1;
                    }
                    heads += range.second - range.first;
                }
                return heads;
            };
            auto validEmptyAttention = [](const Data *cache, int device) {
                return cache != nullptr && !cache->isFake && cache->isKVCache &&
                       !cache->isLinearAttention && !cache->isPagedKVCache &&
                       cache->pagedKVCacheData == nullptr && cache->pageIndex.empty() &&
                       cache->lastPageLen == 0 && cache->dims.empty() &&
                       cache->strides.empty() && cache->expansionDims.empty() &&
                       cache->expansionSize == 0 && cache->expansionBytes == 0 &&
                       cache->cpuData == nullptr && cache->cudaData == nullptr &&
                       cache->deviceData == nullptr && !cache->multiDeviceData &&
                       cache->dataDevice == DataDevice::CUDA &&
                       cache->dataDeviceIds == std::vector<int>({device});
            };
            auto validPagedAttention = [&](const Data *cache, int device,
                                           int expectedHeads) {
                if (cache == nullptr || expectedHeads <= 0 || cache->isFake ||
                    !cache->isKVCache || cache->isLinearAttention ||
                    !cache->isPagedKVCache || cache->pagedKVCacheData == nullptr ||
                    cache->pageLen <= 0 || cache->dataDevice != DataDevice::CUDA ||
                    cache->dataDeviceIds != std::vector<int>({device}) ||
                    cache->dims.size() != 3 || cache->dims[0] != expectedHeads ||
                    cache->dims[2] != model->head_dim) {
                    return false;
                }
                PagedCacheManager *manager = cache->pagedKVCacheData;
                if (manager->type != PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE ||
                    manager->pageLen != cache->pageLen || manager->maxPages <= 0 ||
                    manager->dims.size() != 4 || manager->dims[0] != manager->maxPages ||
                    manager->dims[1] != cache->pageLen ||
                    manager->dims[2] != expectedHeads ||
                    manager->dims[3] != model->head_dim) {
                    return false;
                }
                if (cache->pageIndex.empty()) {
                    return cache->lastPageLen == 0 && cache->dims[1] == 0;
                }
                if (cache->lastPageLen <= 0 || cache->lastPageLen > cache->pageLen) {
                    return false;
                }
                for (int page : cache->pageIndex) {
                    if (page < 0 || page >= manager->maxPages) {
                        return false;
                    }
                }
                int cachedTokens = ((int)cache->pageIndex.size() - 1) *
                                   cache->pageLen + cache->lastPageLen;
                return cache->dims[1] == cachedTokens;
            };

            for (int layer = 0; layer < model->block_cnt; layer++) {
                Data &rootKey = ctx->pastKeyValues[layer].first;
                Data &rootValue = ctx->pastKeyValues[layer].second;
                bool isLinearLayer = Qwen35LayerIsLinearAttention(model, layer);
                for (int device : devices) {
                    Data *localKey = getLocal(rootKey, device);
                    Data *localValue = getLocal(rootValue, device);
                    if (!isLinearLayer) {
                        int localHeads = localAttentionHeads(layer, device);
                        if (localHeads < 0 ||
                            (localHeads == 0 &&
                             (!validEmptyAttention(localKey, device) ||
                              !validEmptyAttention(localValue, device))) ||
                            (localHeads > 0 &&
                             (!validPagedAttention(localKey, device, localHeads) ||
                              !validPagedAttention(localValue, device, localHeads)))) {
                            return false;
                        }
                        continue;
                    }
                    if (localKey == nullptr || localValue == nullptr ||
                        localKey->dataDevice != DataDevice::CUDA ||
                        localValue->dataDevice != DataDevice::CUDA ||
                        localKey->dataDeviceIds != std::vector<int>({device}) ||
                        localValue->dataDeviceIds != std::vector<int>({device}) ||
                        localKey->dataType != DataType::FLOAT16 ||
                        localValue->dataType != DataType::FLOAT16 ||
                        localKey->cudaData == nullptr || localValue->cudaData == nullptr ||
                        localKey->dims.size() != 3 || localKey->dims[0] != 1 ||
                        localKey->dims[1] <= 0 || localKey->dims[2] != 4 ||
                        localValue->dims.size() != 4 || localValue->dims[0] != 1 ||
                        localValue->dims[1] <= 0 ||
                        localValue->dims[2] != model->head_k_dim ||
                        localValue->dims[3] != model->head_v_dim ||
                        !localValue->isLinearAttentionTransposed ||
                        !hasDenseStrides(*localKey) || !hasDenseStrides(*localValue)) {
                        return false;
                    }
                }
            }
            return true;
        };

        auto collectDecodePageNeeds = [&](ResponseContext *ctx) -> std::map<PagedCacheManager*, int> {
            std::map<PagedCacheManager*, int> needs;
            int decodeTokens = ctx == nullptr ? 1 :
                std::max(1, (int)ctx->currentTokens.size());
            // Multi-token MTP validation runs against a paged-cache view. A
            // partial last page is cloned before appending so rejected draft
            // tokens cannot overwrite the real cache.
            bool needsValidationCopy =
                decodeTokens > 1 && !canBudgetForTpInplace(ctx, decodeTokens);
            std::function<void(Data&)> addCacheNeed = [&](Data &cache) {
                if (cache.multiDeviceData && !cache.multiDeviceDatas.empty()) {
                    bool usedLocal = false;
                    for (auto &it : cache.multiDeviceDatas) {
                        if (it.second != nullptr) {
                            addCacheNeed(*it.second);
                            usedLocal = true;
                        }
                    }
                    if (usedLocal) {
                        return;
                    }
                }
                if (!cache.isPagedKVCache || cache.pagedKVCacheData == nullptr) {
                    return;
                }
                if (cache.pagedKVCacheData->type != PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE) {
                    return;
                }
                int cachePageLen = std::max(1, cache.pageLen);
                if (!cache.pageIndex.empty() &&
                    (cache.pageLen <= 0 || cache.lastPageLen <= 0 ||
                     cache.lastPageLen > cachePageLen)) {
                    // Force this request through releaseAndReinitRequest. An
                    // invalid tail length cannot safely participate in COW:
                    // treating zero as either empty or full would produce a
                    // different page budget and may overwrite a shared page.
                    needs[cache.pagedKVCacheData] = INT_MAX;
                    return;
                }
                int lastLen = cache.pageIndex.empty() ? 0 : cache.lastPageLen;
                bool partialLastPage = !cache.pageIndex.empty() &&
                    lastLen > 0 && lastLen < cachePageLen;
                int freeSlots = partialLastPage ? cachePageLen - lastLen : 0;
                int tokensNeedingPages = std::max(0, decodeTokens - freeSlots);
                int appendPages = (tokensNeedingPages + cachePageLen - 1) / cachePageLen;
                int clonePages = needsValidationCopy && partialLastPage ? 1 : 0;
                int requiredPages = clonePages + appendPages;
                if (requiredPages > 0) {
                    int &managerNeed = needs[cache.pagedKVCacheData];
                    managerNeed = managerNeed > INT_MAX - requiredPages ?
                        INT_MAX : managerNeed + requiredPages;
                }
            };
            if (ctx != nullptr) {
                for (int i = 0; i < model->block_cnt && i < (int)ctx->pastKeyValues.size(); i++) {
                    addCacheNeed(ctx->pastKeyValues[i].first);
                    addCacheNeed(ctx->pastKeyValues[i].second);
                }
            }
            return needs;
        };

        auto hasPagedManagerShortage = [](const std::map<PagedCacheManager*, int> &needs) -> bool {
            for (auto &it : needs) {
                PagedCacheManager *manager = it.first;
                if (manager == nullptr || it.second <= 0) {
                    continue;
                }
                int freePages = 0;
                {
                    std::lock_guard<std::mutex> guard(manager->pageIndexLocker);
                    freePages = manager->FreePageCount();
                }
                if (freePages < it.second) {
                    return true;
                }
            }
            return false;
        };

        auto tryRestorePrefixCache = [&](ResponseContext *ctx) -> int {
            if (ctx == nullptr || ctx->cacheLen != 0 || ctx->currentTokens.empty()) {
                return 0;
            }
            auto probeRefs = model->GetPagedKVCacheManagers(model->kvCacheId, true);
            PagedCacheManager *probeManager = nullptr;
            for (auto &ref : probeRefs) {
                if (ref.second != nullptr) {
                    probeManager = ref.second;
                    break;
                }
            }
            if (probeManager == nullptr) {
                return 0;
            }

            std::map<PagedCacheManager*, std::vector<int> > queriedPages;
            auto queryManager = [&](PagedCacheManager *manager) -> std::vector<int>& {
                auto it = queriedPages.find(manager);
                if (it == queriedPages.end()) {
                    std::vector<int> pages;
                    manager->Query(ctx->currentTokens, pages);
                    it = queriedPages.emplace(manager, std::move(pages)).first;
                }
                return it->second;
            };

            int minCachedPages = (int)queryManager(probeManager).size();
            if (minCachedPages <= 0) {
                return 0;
            }
            for (int layer = 0; layer < model->block_cnt; layer++) {
                for (int keyFlag = 0; keyFlag < 2; keyFlag++) {
                    auto refs = model->GetPagedKVCacheManagers(layer, keyFlag == 0);
                    for (auto &ref : refs) {
                        PagedCacheManager *manager = ref.second;
                        if (manager == nullptr ||
                            manager->pageLen != probeManager->pageLen) {
                            continue;
                        }
                        minCachedPages = std::min(
                            minCachedPages, (int)queryManager(manager).size());
                    }
                }
            }
            if (minCachedPages <= 0) {
                return 0;
            }

            int cachedLen = minCachedPages * probeManager->pageLen;
            if (cachedLen >= (int)ctx->currentTokens.size()) {
                minCachedPages--;
                cachedLen = minCachedPages * probeManager->pageLen;
            }
            if (minCachedPages <= 0) {
                return 0;
            }

            int extraCachedLen = model->QueryPagedPrefixCacheExtra(ctx, cachedLen);
            extraCachedLen = std::max(0, std::min(extraCachedLen, cachedLen));
            minCachedPages = extraCachedLen / probeManager->pageLen;
            if (minCachedPages <= 0) {
                return 0;
            }
            cachedLen = minCachedPages * probeManager->pageLen;
            if (!model->RestorePagedPrefixCacheExtra(ctx, cachedLen)) {
                return -1;
            }

            auto managerDevice = [](PagedCacheManager *manager) {
                if (manager == nullptr) {
                    return -1;
                }
                Data *managerData = (Data*)manager;
                return managerData->dataDeviceIds.empty() ?
                    -1 : managerData->dataDeviceIds[0];
            };
            auto restoreOne = [&](Data &cache,
                                  PagedCacheManager *manager,
                                  const std::vector<int> &pages) {
                if (manager == nullptr ||
                    (int)pages.size() < minCachedPages) {
                    return false;
                }
                Data *managerData = (Data*)manager;
                if (managerData->dims.size() < 4) {
                    return false;
                }
                cache.isKVCache = true;
                cache.isLinearAttention = false;
                cache.isPagedKVCache = true;
                cache.pagedKVCacheData = manager;
                cache.pageLen = manager->pageLen;
                cache.pageIndex.assign(
                    pages.begin(), pages.begin() + minCachedPages);
                manager->Pick(cache.pageIndex);
                cache.lastPageLen = manager->pageLen;
                cache.dataType = managerData->dataType;
                cache.UpdateUnitSize();
                cache.dataDevice = managerData->dataDevice;
                cache.dataDeviceIds = managerData->dataDeviceIds;
                cache.Resize({managerData->dims[2], cachedLen,
                              managerData->dims[3]});
                return true;
            };
            auto restorePagedCache = [&]
                    (Data &cache,
                     const std::vector<std::pair<int, PagedCacheManager*> > &refs) {
                std::vector<std::pair<int, PagedCacheManager*> > validRefs;
                for (auto ref : refs) {
                    if (ref.second == nullptr ||
                        ref.second->pageLen != probeManager->pageLen ||
                        (int)queryManager(ref.second).size() < minCachedPages) {
                        continue;
                    }
                    validRefs.push_back(ref);
                }
                if (validRefs.empty()) {
                    return refs.empty();
                }
                if (validRefs.size() == 1) {
                    return restoreOne(cache, validRefs[0].second,
                                      queryManager(validRefs[0].second));
                }

                cache.multiDeviceData = true;
                cache.dataDevice = DataDevice::CUDA;
                cache.dataDeviceIds.clear();
                cache.isKVCache = true;
                cache.isLinearAttention = false;
                cache.isPagedKVCache = true;
                Data *firstLocal = nullptr;
                for (auto &ref : validRefs) {
                    int device = ref.first >= 0 ?
                        ref.first : managerDevice(ref.second);
                    if (device < 0) {
                        continue;
                    }
                    cache.dataDeviceIds.push_back(device);
                    Data *managerData = (Data*)ref.second;
                    Data *&local = cache.multiDeviceDatas[device];
                    if (local == nullptr) {
                        local = new Data(managerData->dataType);
                        local->SetKVCache();
                        local->cacheUid = cache.cacheUid;
                    }
                    if (!restoreOne(*local, ref.second,
                                    queryManager(ref.second))) {
                        return false;
                    }
                    if (firstLocal == nullptr) {
                        firstLocal = local;
                    }
                }
                if (firstLocal == nullptr) {
                    cache.multiDeviceData = false;
                    cache.isPagedKVCache = false;
                    cache.pagedKVCacheData = nullptr;
                    cache.pageIndex.clear();
                    return false;
                }
                cache.dataType = firstLocal->dataType;
                cache.UpdateUnitSize();
                cache.cudaData = nullptr;
                cache.pageLen = firstLocal->pageLen;
                cache.pageIndex = firstLocal->pageIndex;
                cache.lastPageLen = firstLocal->lastPageLen;
                cache.pagedKVCacheData = firstLocal->pagedKVCacheData;
                cache.dims = firstLocal->dims;
                return true;
            };

            for (int layer = 0; layer < model->block_cnt; layer++) {
                auto keyRefs = model->GetPagedKVCacheManagers(layer, true);
                auto valueRefs = model->GetPagedKVCacheManagers(layer, false);
                if (!restorePagedCache(ctx->pastKeyValues[layer].first, keyRefs) ||
                    !restorePagedCache(ctx->pastKeyValues[layer].second, valueRefs)) {
                    return -1;
                }
            }
            ctx->currentTokens.erase(
                ctx->currentTokens.begin(),
                ctx->currentTokens.begin() + cachedLen);
            ctx->cacheLen = cachedLen;
            if (model->verbose) {
                printf("[Qwen3.5 MTP] prefix cache hit: tokens=%d.\n", cachedLen);
                fflush(stdout);
            }
            return cachedLen;
        };

        auto createPendingResultLogits = [](const GenerationConfig &config) {
            return config.output_logits ? new std::vector<float>() : nullptr;
        };

        auto queueGeneratedResultLogits = [](ResponseContext *ctx,
                                             std::vector<std::vector<float>*> &logits,
                                             int index) {
            if (ctx == nullptr || index < 0 || index >= (int)logits.size() || logits[index] == nullptr) {
                return;
            }
            ctx->resultLogits.push(logits[index]);
            logits[index] = nullptr;
        };

        auto releasePendingResultLogits = [](std::vector<std::vector<float>*> &logits) {
            for (auto *&item : logits) {
                delete item;
                item = nullptr;
            }
        };

        auto *pcm = GetPagedCacheManager(model->kvCacheId * 2);
        if (pcm != nullptr) {
            totalPages = pcm->maxPages;
            pageLen = pcm->pageLen;
            maxTotalLens = totalPages * pageLen;
            pagesLimit = totalPages * 4 / 5;
            model->tokensLimit = maxTotalLens;
            model->promptLimit = pagesLimit * pageLen;
            if (model->verbose) {
                printf("Fastllm KV Cache Token limit: %d tokens (totalPages=%d, pageLen=%d).\n",
                       maxTotalLens, totalPages, pageLen);
                printf("Fastllm AddPrefill Pages limit: %d pages (80%% of %d).\n",
                       pagesLimit, totalPages);
                printf("Fastllm Scheduler: Qwen3.5 MTP (lane=%d).\n", mtpSchedulerLanes);
            }
        } else if (model->tokensLimit > 0) {
            maxTotalLens = model->tokensLimit;
            totalPages = std::max(1, maxTotalLens / pageLen);
            pagesLimit = model->promptLimit > 0 ?
                std::max(1, (model->promptLimit + pageLen - 1) / pageLen) :
                totalPages * 4 / 5;
            if (model->promptLimit <= 0) {
                model->promptLimit = pagesLimit * pageLen;
            }
            if (model->verbose) {
                printf("Fastllm KV Cache Token limit: %d tokens (pageLen=%d).\n",
                       maxTotalLens, pageLen);
                printf("Fastllm AddPrefill Pages limit: %d pages.\n", pagesLimit);
                printf("Fastllm Scheduler: Qwen3.5 MTP (lane=%d).\n", mtpSchedulerLanes);
            }
        } else if (model->verbose) {
            printf("Fastllm Scheduler: Qwen3.5 MTP (lane=%d).\n", mtpSchedulerLanes);
        }

        auto lastRecordTime = std::chrono::system_clock::now();
        long long genTokens = 0;
        while (true) {
            if (model->isFree) {
                break;
            }

            std::vector<Data*> attentionMasks;
            std::vector<Data*> positionIds;
            std::vector<Data*> ownedAttentionMasks;
            std::vector<Data*> ownedPositionIds;
            std::vector<std::pair<Data*, Data*> > pastKeyValues;
            std::vector<float> ids;
            std::vector<int> seqLens;
            std::vector<int> handles;
            std::vector<GenerationConfig> generationConfigs;
            LastTokensManager tokensManager;
            std::vector<ResponseContext*> tokenContexts;
            std::vector<std::vector<float>*> logits;
            std::vector<float> decodePositionValues;
            std::vector<Data> decodePositionIds;
            static const std::vector<int> decodeScalarDims = {1, 1};
            bool selectedNeedLastTokens = false;
            bool selectedIsPrompt = false;
            bool selectedMultimodal = false;

            attentionMasks.reserve(1);
            positionIds.reserve(1);
            ownedAttentionMasks.reserve(1);
            ownedPositionIds.reserve(1);
            pastKeyValues.reserve(model->block_cnt);
            ids.reserve(4);
            seqLens.reserve(1);
            handles.reserve(1);
            generationConfigs.reserve(1);
            tokenContexts.reserve(1);
            logits.reserve(1);
            decodePositionValues.reserve(1);
            decodePositionIds.reserve(1);

            std::unique_lock<std::mutex> dictLocker(model->dictLocker);
            auto &forwardLocker = model->forwardLocker;

            std::vector<int> abortHandles;
            int busyPages = 0;
            int currentActivate = 0;
            bool hasPrefill = false;
            struct DecodeOrder {
                int sortKey;
                int handle;
                ResponseContext *context;
            };
            std::vector<DecodeOrder> orders;
            orders.reserve(model->responseContextDict.dicts.size());

            for (auto &it : model->responseContextDict.dicts) {
                ResponseContext *ctx = it.second;
                if (ctx == nullptr) {
                    continue;
                }
                if (ctx->isAbort) {
                    ctx->TryRecordPagedCache(model);
                    abortHandles.push_back(it.first);
                    continue;
                }
                if (ctx->isEnding) {
                    for (int i = 0; i < model->block_cnt && i < (int)ctx->pastKeyValues.size(); i++) {
                        releasePagedCachePages(ctx->pastKeyValues[i].first);
                        releasePagedCachePages(ctx->pastKeyValues[i].second);
                    }
                    eraseMtpCache(ctx);
                    continue;
                }
                if (ctx->preTokens > 0) {
                    currentActivate++;
                }
                if (ctx->preTokens == 0) {
                    hasPrefill = true;
                }
                orders.push_back({-(int)ctx->currentTokens.size(), it.first, ctx});
            }
            for (int handle : abortHandles) {
                model->RemoveResponseContext(handle);
            }
            sort(orders.begin(), orders.end(), [](const DecodeOrder &a, const DecodeOrder &b) {
                if (a.sortKey != b.sortKey) {
                    return a.sortKey < b.sortKey;
                }
                return a.handle < b.handle;
            });

            if (totalPages > 0) {
                PagedCacheManager *probeManager = findRuntimePagedManager();
                if (probeManager != nullptr) {
                    std::lock_guard<std::mutex> guard(probeManager->pageIndexLocker);
                    busyPages = probeManager->maxPages - probeManager->FreePageCount();
                    totalPages = probeManager->maxPages;
                    pageLen = probeManager->pageLen;
                    maxTotalLens = totalPages * pageLen;
                    pagesLimit = totalPages * 4 / 5;
                }
            }
            bool canAddPrefill = (pagesLimit > 0) ? (busyPages < pagesLimit) : true;

            for (int isPrompt = 1; isPrompt >= 0 && seqLens.empty(); isPrompt--) {
                if (isPrompt == 1 && !canAddPrefill) {
                    continue;
                }
                if (isPrompt == 0 && hasPrefill && canAddPrefill) {
                    continue;
                }

                for (auto &ii : orders) {
                    ResponseContext *ctx = ii.context;
                    if (ctx == nullptr || ctx->isEnding) {
                        continue;
                    }
                    if (isPrompt && ctx->preTokens != 0) {
                        continue;
                    }
                    if (!isPrompt && ctx->preTokens == 0) {
                        continue;
                    }
                    if (isPrompt && ctx->cacheLen == 0 &&
                        tryRestorePrefixCache(ctx) < 0) {
                        releaseAndReinitRequest(ctx);
                    }

                    if ((maxTotalLens > 0 &&
                         ctx->cacheLen + (int)ctx->currentTokens.size() > maxTotalLens) ||
                        ctx->cacheLen + (int)ctx->currentTokens.size() > model->max_positions) {
                        ctx->isEnding = true;
                        ctx->error = ResponseContextErrorPromptTooLong;
                        continue;
                    }

                    if (!isPrompt) {
                        auto pageNeeds = collectDecodePageNeeds(ctx);
                        if (!pageNeeds.empty() && hasPagedManagerShortage(pageNeeds)) {
                            releaseAndReinitRequest(ctx);
                            continue;
                        }
                    }

                    tokenContexts.push_back(ctx);
                    handles.push_back(ii.handle);
                    generationConfigs.push_back(ctx->generationConfig);
                    logits.push_back(createPendingResultLogits(ctx->generationConfig));
                    selectedNeedLastTokens |= Qwen35NeedRepeatPenalty(ctx->generationConfig) ||
                                              ctx->generationConfig.output_logits;
                    selectedIsPrompt = isPrompt != 0;
                    selectedMultimodal = !ctx->multimodalInput.empty();

                    if (!isPrompt && !selectedMultimodal && !ctx->currentTokens.empty()) {
                        int seqLen = (int)ctx->currentTokens.size();
                        if (seqLen == 1) {
                            ids.push_back((float)ctx->currentTokens[0]);
                            seqLens.push_back(1);
                            attentionMasks.push_back(nullptr);
                            float position = ctx->allTokens.empty() ?
                                0.0f : (float)((int)ctx->allTokens.size() - 1);
                            decodePositionValues.push_back(position);
                            decodePositionIds.emplace_back(DataType::FLOAT32, decodeScalarDims,
                                                           DataDevice::CPU,
                                                           (void*)&decodePositionValues.back());
                            positionIds.push_back(&decodePositionIds.back());
                            ctx->preTokens += 1;
                        } else {
                            int keyLen = (int)ctx->allTokens.size() + seqLen - 1;
                            int startPosition = keyLen - seqLen;
                            for (int token : ctx->currentTokens) {
                                ids.push_back((float)token);
                            }
                            seqLens.push_back(seqLen);

                            std::vector<float> vPositionIds(seqLen, 0.0f);
                            for (int i = 0; i < seqLen; i++) {
                                vPositionIds[i] = (float)(startPosition + i);
                            }
                            positionIds.push_back(new Data());
                            ownedPositionIds.push_back(positionIds.back());
                            positionIds.back()->CopyFrom(
                                Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));

                            if (model->NeedAttentionMask(seqLen, keyLen)) {
                                std::vector<float> vmask(seqLen * keyLen, 0.0f);
                                for (int i = 0; i < seqLen; i++) {
                                    for (int j = i + 1; j < seqLen; j++) {
                                        vmask[i * keyLen + startPosition + j] = 1.0f;
                                    }
                                }
                                attentionMasks.push_back(new Data());
                                ownedAttentionMasks.push_back(attentionMasks.back());
                                attentionMasks.back()->CopyFrom(
                                    Data(DataType::FLOAT32, {seqLen, keyLen}, vmask));
                                ToDataType(*attentionMasks.back(), model->dataType);
                            } else {
                                attentionMasks.push_back(nullptr);
                            }
                            ctx->preTokens += seqLen;
                        }
                    } else {
                        if (ctx->preTokens == 0) {
                            ctx->intParams["add_special_tokens"] =
                                ctx->cacheLen > 0 ? false : ctx->generationConfig.add_special_tokens;
                            ctx->intParams["promptLen"] =
                                ctx->cacheLen + (int)ctx->currentTokens.size();
                            ctx->intParams["index"] = 0;
                        } else {
                            ctx->intParams["index"]++;
                        }
                        Data inputIds, attentionMask, curPositionIds;
                        std::vector<std::vector<float> > tokens(1);
                        tokens[0].reserve(ctx->currentTokens.size());
                        for (int token : ctx->currentTokens) {
                            tokens[0].push_back((float)token);
                        }
                        model->FillLLMInputs(tokens, ctx->intParams,
                                             inputIds, attentionMask, curPositionIds);
                        ToDataType(attentionMask, model->dataType);

                        seqLens.push_back(inputIds.Count(0));
                        for (int i = 0; i < inputIds.Count(0); i++) {
                            ids.push_back(((float*)inputIds.cpuData)[i]);
                        }
                        if (attentionMask.dims.empty()) {
                            attentionMasks.push_back(nullptr);
                        } else {
                            attentionMasks.push_back(new Data());
                            ownedAttentionMasks.push_back(attentionMasks.back());
                            attentionMask.ToDevice(DataDevice::CPU);
                            attentionMasks.back()->CopyFrom(attentionMask);
                        }
                        if (curPositionIds.dims.empty()) {
                            positionIds.push_back(nullptr);
                        } else {
                            positionIds.push_back(new Data());
                            ownedPositionIds.push_back(positionIds.back());
                            positionIds.back()->CopyFrom(curPositionIds);
                        }
                        ctx->preTokens += seqLens.back();
                    }

                    for (int i = 0; i < model->block_cnt; i++) {
                        pastKeyValues.push_back(std::make_pair(&ctx->pastKeyValues[i].first,
                                                               &ctx->pastKeyValues[i].second));
                    }
                    break;
                }
            }

            if (selectedNeedLastTokens) {
                tokensManager.units.reserve(tokenContexts.size());
                for (auto *ctx : tokenContexts) {
                    tokensManager.units.push_back(ctx->tokens);
                }
            }

            if (!seqLens.empty()) {
                ResponseContext *singleContext = tokenContexts.empty() ? nullptr : tokenContexts[0];
                dictLocker.unlock();
                forwardLocker.lock();

                Data inputIds(DataType::FLOAT32, {1, (int)ids.size()}, ids);
                std::vector<int> ret;
                std::vector<std::vector<int> > acceptedTokenLists;
                std::vector<std::vector<int> > nextInputTokenLists;
                std::vector<int> keptInputLens;
                bool usedMtpForward = false;

                if (selectedMultimodal && singleContext != nullptr) {
                    ret = model->ForwardMultimodal(
                        inputIds,
                        attentionMasks[0] == nullptr ? Data() : *attentionMasks[0],
                        positionIds[0] == nullptr ? Data() : *positionIds[0],
                        singleContext->pastKeyValues,
                        singleContext->multimodalInput,
                        singleContext->generationConfig,
                        tokensManager,
                        &logits);
                } else if (seqLens.size() == 1 && selectedIsPrompt &&
                           seqLens[0] > prefillChunkSize && singleContext != nullptr) {
                    int len = seqLens[0];
                    std::vector<std::pair<Data, Data> > *pastKeyValue1 = nullptr;
                    dictLocker.lock();
                    auto contextIt = model->responseContextDict.dicts.find(handles[0]);
                    if (contextIt != model->responseContextDict.dicts.end()) {
                        pastKeyValue1 = &contextIt->second->pastKeyValues;
                    }
                    dictLocker.unlock();
                    if (pastKeyValue1 == nullptr) {
                        ret.push_back(model->eos_token_id);
                    } else {
                        std::vector<int> longPrefillMtpDevices;
                        std::map<int, int> longPrefillMtpRatios;
                        int longPrefillMtpDrafts = Qwen35MtpDraftsPerStep();
                        bool seedLongPrefillMtp =
                            !Qwen35MtpDisabledByEnv() &&
                            longPrefillMtpDrafts > 0 &&
                            model->HasMtpWeights() &&
                            generationConfigs.size() == 1 &&
                            generationConfigs[0].IsSimpleGreedy() &&
                            !generationConfigs[0].output_logits &&
                            positionIds[0] != nullptr &&
                            GetQwen35GPUForwardDevices(
                                model->deviceMap, longPrefillMtpDevices,
                                longPrefillMtpRatios) &&
                            !longPrefillMtpDevices.empty();
                        int longPrefillMtpBaseTokens = singleContext->cacheLen;
                        if (seedLongPrefillMtp) {
                            std::lock_guard<std::mutex> guard(model->mtpCacheMutex);
                            if (longPrefillMtpBaseTokens == 0) {
                                model->mtpCaches.erase(singleContext);
                            } else {
                                auto mtpIt = model->mtpCaches.find(singleContext);
                                seedLongPrefillMtp =
                                    mtpIt != model->mtpCaches.end() &&
                                    mtpIt->second.tokens == longPrefillMtpBaseTokens &&
                                    mtpIt->second.key.dims.size() >= 2 &&
                                    mtpIt->second.value.dims.size() >= 2 &&
                                    mtpIt->second.key.dims[1] == longPrefillMtpBaseTokens &&
                                    mtpIt->second.value.dims[1] == longPrefillMtpBaseTokens;
                            }
                        }
                        auto eraseLongPrefillMtpCache = [&]() {
                            std::lock_guard<std::mutex> guard(model->mtpCacheMutex);
                            model->mtpCaches.erase(singleContext);
                        };
                        auto appendLongPrefillMtpCache =
                            [&](const Data &targetHiddenStates,
                                const std::vector<int> &mtpInputTokens,
                                const Data &mtpPositionIds,
                                int expectedTokens, bool cacheOnly,
                                int &draftToken) {
                                std::lock_guard<std::mutex> guard(model->mtpCacheMutex);
                                auto cacheIt = model->mtpCaches.find(singleContext);
                                if (cacheIt == model->mtpCaches.end()) {
                                    if (expectedTokens != 0) {
                                        return false;
                                    }
                                    cacheIt = model->mtpCaches.emplace(
                                        singleContext, MtpKvCache()).first;
                                }
                                MtpKvCache &cache = cacheIt->second;
                                if (cache.tokens != expectedTokens) {
                                    model->mtpCaches.erase(cacheIt);
                                    return false;
                                }
                                try {
                                    draftToken = model->RunMtpGreedyDraft(
                                        longPrefillMtpDevices[0],
                                        longPrefillMtpDevices, cache,
                                        targetHiddenStates, mtpInputTokens,
                                        mtpPositionIds,
                                        (int)mtpInputTokens.size() - 1,
                                        nullptr, cacheOnly);
                                } catch (...) {
                                    model->mtpCaches.erase(singleContext);
                                    throw;
                                }
                                int newTokens = expectedTokens +
                                    (int)mtpInputTokens.size();
                                bool valid = cache.tokens == newTokens &&
                                    cache.key.dims.size() >= 2 &&
                                    cache.value.dims.size() >= 2 &&
                                    cache.key.dims[1] == newTokens &&
                                    cache.value.dims[1] == newTokens;
                                if (!valid) {
                                    model->mtpCaches.erase(singleContext);
                                }
                                return valid;
                            };

                        int longPrefillFirstDraft = -1;
                        bool longPrefillMtpSeeded = false;
                        auto prefillStartTime = std::chrono::system_clock::now();
                        for (int st = 0; st < len; ) {
                            int curLen = std::min(prefillChunkSize, len - st);
                            auto chunkStartTime = std::chrono::system_clock::now();
                            Data curInput, curPositionIds;
                            Split(inputIds, 1, st, st + curLen, curInput);
                            if (positionIds[0] != nullptr) {
                                curPositionIds.dataType = positionIds[0]->dataType;
                                curPositionIds.Resize({1, curLen});
                                curPositionIds.Allocate();
                                int unitSize = curPositionIds.unitSize;
                                memcpy(curPositionIds.cpuData,
                                       positionIds[0]->cpuData + st * unitSize,
                                       curLen * unitSize);
                            }

                            std::vector<int> curSeqLens = {curLen};
                            std::vector<Data*> curAttentionMasks = {nullptr};
                            std::vector<Data*> curPositionIdsVec =
                                {positionIds[0] == nullptr ? nullptr : &curPositionIds};
                            std::vector<std::pair<Data*, Data*> > curPastKeyValues;
                            curPastKeyValues.reserve(model->block_cnt);
                            for (int i = 0; i < model->block_cnt; i++) {
                                curPastKeyValues.push_back(
                                    std::make_pair(&(*pastKeyValue1)[i].first,
                                                   &(*pastKeyValue1)[i].second));
                            }
                            bool oldCaptureAllHiddenStates =
                                model->speculativeCaptureAllHiddenStates;
                            model->speculativeCaptureAllHiddenStates = seedLongPrefillMtp;
                            if (seedLongPrefillMtp) {
                                model->speculativeHiddenStates.FreeSpace();
                                model->speculativeHiddenStates.dims.clear();
                                model->speculativeHiddenStates.strides.clear();
                                model->speculativeHiddenStates.expansionDims.clear();
                            }
                            try {
                                ret = model->ForwardGPU(1, curInput, curAttentionMasks,
                                                        curPositionIdsVec, curSeqLens,
                                                        curPastKeyValues, generationConfigs,
                                                        tokensManager, &logits);
                            } catch (...) {
                                model->speculativeCaptureAllHiddenStates =
                                    oldCaptureAllHiddenStates;
                                if (seedLongPrefillMtp) {
                                    eraseLongPrefillMtpCache();
                                }
                                throw;
                            }
                            model->speculativeCaptureAllHiddenStates =
                                oldCaptureAllHiddenStates;

                            bool isLastChunk = st + curLen == len;
                            if (seedLongPrefillMtp) {
                                AssertInFastLLM(!ret.empty(),
                                                "Qwen3.5 long prefill returned no token.\n");
                                std::vector<int> mtpInputTokens(curLen);
                                for (int i = 0; i < curLen; i++) {
                                    int nextIndex = st + i + 1;
                                    mtpInputTokens[i] = nextIndex < len ?
                                        (int)(ids[nextIndex] + 1e-3f) : ret.back();
                                }
                                int draftToken = -1;
                                bool appended = appendLongPrefillMtpCache(
                                    model->speculativeHiddenStates,
                                    mtpInputTokens, curPositionIds,
                                    longPrefillMtpBaseTokens + st,
                                    !isLastChunk, draftToken);
                                if (!appended) {
                                    seedLongPrefillMtp = false;
                                } else if (isLastChunk) {
                                    longPrefillFirstDraft = draftToken;
                                    longPrefillMtpSeeded = draftToken >= 0;
                                }
                            }
                            st += curLen;
                            int cachedTokens = longPrefillMtpBaseTokens + st;
                            if (cachedTokens % pageLen == 0) {
                                singleContext->TryRecordPagedCache(model);
                            }
                            if (model->verbose) {
                                auto chunkEndTime = std::chrono::system_clock::now();
                                float chunkSpend = GetSpan(chunkStartTime, chunkEndTime);
                                float totalSpend = GetSpan(prefillStartTime, chunkEndTime);
                                float chunkSpeed = chunkSpend > 0 ? curLen / chunkSpend : 0;
                                (void)totalSpend;
                                printf("[Prompt] Long Prefill ... (%d/%d, %d%%). Speed: %f tokens / s.\n",
                                       st, len, st * 100 / len, chunkSpeed);
                            }
                        }
                        if (longPrefillMtpSeeded) {
                            int nextToken = ret.back();
                            usedMtpForward = true;
                            acceptedTokenLists = {{nextToken}};
                            nextInputTokenLists = {
                                {nextToken, longPrefillFirstDraft}
                            };
                            keptInputLens = {len};
                            if (!model->mtpLogPrinted.exchange(true)) {
                                printf("[Qwen3.5 MTP] enabled: layers=%d, drafts_per_step=%d, root_device=cuda:%d, tp_devices=%zu, log_interval=%d validations.\n",
                                       model->mtp_num_hidden_layers,
                                       longPrefillMtpDrafts,
                                       longPrefillMtpDevices[0],
                                       longPrefillMtpDevices.size(),
                                       QWEN35_MTP_LOG_INTERVAL);
                            }
                            printf("[Qwen3.5 MTP] long prefill cache seeded: tokens=%d.\n",
                                   longPrefillMtpBaseTokens + len);
                            fflush(stdout);
                        }
                    }
                } else {
                    auto batchStartTime = std::chrono::system_clock::now();
                    usedMtpForward = model->Qwen35MTPForward(
                        true, singleContext, inputIds, attentionMasks, positionIds,
                        seqLens, pastKeyValues, generationConfigs,
                        acceptedTokenLists, nextInputTokenLists, keptInputLens);

                    if (!usedMtpForward && !selectedIsPrompt && seqLens[0] > 1 &&
                        singleContext != nullptr && !singleContext->currentTokens.empty()) {
                        singleContext->preTokens -= seqLens[0];
                        std::vector<float> fallbackIds = {
                            (float)singleContext->currentTokens[0]
                        };
                        Data fallbackInputIds(DataType::FLOAT32, {1, 1}, fallbackIds);
                        float position = singleContext->allTokens.empty() ?
                            0.0f : (float)((int)singleContext->allTokens.size() - 1);
                        Data fallbackPositionIds(DataType::FLOAT32, {1, 1},
                                                 DataDevice::CPU, (void*)&position);
                        std::vector<Data*> fallbackAttentionMasks = {nullptr};
                        std::vector<Data*> fallbackPositionIdVec = {&fallbackPositionIds};
                        std::vector<int> fallbackSeqLens = {1};
                        singleContext->preTokens += 1;
                        ret = model->ForwardGPU(1, fallbackInputIds,
                                                fallbackAttentionMasks,
                                                fallbackPositionIdVec,
                                                fallbackSeqLens, pastKeyValues,
                                                generationConfigs, tokensManager, &logits);
                        seqLens = fallbackSeqLens;
                    } else if (!usedMtpForward) {
                        ret = model->ForwardGPU((int)seqLens.size(), inputIds,
                                                attentionMasks, positionIds, seqLens,
                                                pastKeyValues, generationConfigs,
                                                tokensManager, &logits);
                    }

                    if (model->verbose && selectedIsPrompt) {
                        int prefillTokens = 0;
                        for (int len : seqLens) {
                            if (len > 1) {
                                prefillTokens += len;
                            }
                        }
                        if (prefillTokens > 0) {
                            auto batchEndTime = std::chrono::system_clock::now();
                            float batchSpend = GetSpan(batchStartTime, batchEndTime);
                            float prefillSpeed = batchSpend > 0 ? prefillTokens / batchSpend : 0;
                            printf("[Prompt] %d Tokens. Speed: %f tokens / s.\n",
                                   prefillTokens, prefillSpeed);
                        }
                    }
                }

                if (!usedMtpForward) {
                    acceptedTokenLists.clear();
                    nextInputTokenLists.clear();
                    keptInputLens.assign(seqLens.begin(), seqLens.end());
                    acceptedTokenLists.reserve(ret.size());
                    nextInputTokenLists.reserve(ret.size());
                    for (int token : ret) {
                        acceptedTokenLists.push_back(std::vector<int>{token});
                        nextInputTokenLists.push_back(std::vector<int>{token});
                    }
                }

                forwardLocker.unlock();
                dictLocker.lock();

                if (selectedIsPrompt) {
                    for (int i = 0; i < (int)handles.size(); i++) {
                        auto contextIt = model->responseContextDict.dicts.find(handles[i]);
                        if (contextIt == model->responseContextDict.dicts.end()) {
                            continue;
                        }
                        if (i < (int)seqLens.size() && seqLens[i] > 1 &&
                            (int)contextIt->second->allTokens.size() >= pageLen) {
                            contextIt->second->TryRecordPagedCache(model);
                        }
                    }
                }

                if (model->verbose) {
                    for (auto &tokens : acceptedTokenLists) {
                        genTokens += (int)tokens.size();
                    }
                    auto nowTime = std::chrono::system_clock::now();
                    float spend = GetSpan(lastRecordTime, nowTime);
                    if (spend > 1) {
                        int logPending = (int)orders.size() - currentActivate;
                        float kvUsage = totalPages > 0 ? busyPages * 100.0f / totalPages : 0;
                        printf("[Decode] alive = %d, pending = %d, context usages: %.1f%%, Speed: %f tokens / s.\n",
                               currentActivate, logPending, kvUsage,
                               spend > 0 ? (float)genTokens / spend : 0.0f);
                        lastRecordTime = nowTime;
                        genTokens = 0;
                    }
                }

                for (int i = 0; i < (int)handles.size(); i++) {
                    auto contextIt = model->responseContextDict.dicts.find(handles[i]);
                    if (contextIt == model->responseContextDict.dicts.end()) {
                        continue;
                    }
                    ResponseContext *ctx = contextIt->second;
                    if (i < (int)keptInputLens.size() && i < (int)seqLens.size() &&
                        keptInputLens[i] >= 0 && keptInputLens[i] < seqLens[i]) {
                        ctx->preTokens -= seqLens[i] - keptInputLens[i];
                    }

                    static const std::vector<int> emptyAcceptedTokens;
                    const std::vector<int> &curAcceptedTokens =
                        i < (int)acceptedTokenLists.size() ?
                            acceptedTokenLists[i] : emptyAcceptedTokens;
                    for (int tokenIndex = 0; tokenIndex < (int)curAcceptedTokens.size(); tokenIndex++) {
                        int curRet = curAcceptedTokens[tokenIndex];
                        if (curRet == model->eos_token_id ||
                            model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            eraseMtpCache(ctx);
                            break;
                        }
                        auto itStopTk = ctx->generationConfig.stop_token_ids.find(curRet);
                        if (itStopTk != ctx->generationConfig.stop_token_ids.end()) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            eraseMtpCache(ctx);
                            break;
                        }

                        ctx->resultTokenQueue.push(curRet);
                        if (tokenIndex == 0) {
                            queueGeneratedResultLogits(ctx, logits, i);
                        }
                        ctx->allTokens.push_back(curRet);
                        if (Qwen35NeedRepeatPenalty(ctx->generationConfig)) {
                            ctx->tokens.Push(curRet);
                        }
                        ctx->curTokens++;
                        if (ctx->curTokens == ctx->generationConfig.output_token_limit) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            eraseMtpCache(ctx);
                        } else if (ctx->allTokens.size() >= model->max_positions) {
                            ctx->isEnding = true;
                            ctx->TryRecordPagedCache(model);
                            eraseMtpCache(ctx);
                        }
                        if (ctx->isEnding) {
                            break;
                        }
                    }
                    if (!ctx->isEnding && !curAcceptedTokens.empty()) {
                        if (i < (int)nextInputTokenLists.size() &&
                            !nextInputTokenLists[i].empty()) {
                            ctx->currentTokens = nextInputTokenLists[i];
                        } else {
                            ctx->currentTokens.assign(1, curAcceptedTokens.back());
                        }
                    }
                }
                releasePendingResultLogits(logits);
            }

            for (auto *ptr : ownedAttentionMasks) {
                delete ptr;
            }
            for (auto *ptr : ownedPositionIds) {
                delete ptr;
            }

            if (seqLens.empty()) {
                if (!orders.empty()) {
                    model->dictCV.wait_for(dictLocker, std::chrono::milliseconds(10));
                } else {
                    model->dictCV.wait(dictLocker);
                }
            }
        }
#endif
    }

    std::vector <int> Qwen3_5Model::ForwardV2ThreadTensorParallel(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                          pastKeyValues, generationConfigs, lastTokens, retLogits);
    }

    void Qwen3_5Model::InitParams() {
        auto getDictValue = [&](const std::string &key, const std::string &defaultValue) {
            auto it = this->weight.dicts.find(key);
            if (it != this->weight.dicts.end()) {
                return it->second;
            }
            return defaultValue;
        };
        std::map<std::string, std::string> extra;
        for (auto &it : this->weight.dicts) {
            std::string key = it.first;
            if (key.substr(0, 12) == "text_config.") {
                std::string stripped = key.substr(12);
                if (stripped.substr(0, 16) == "rope_parameters.") {
                    stripped = stripped.substr(16);
                }
                extra[stripped] = it.second;
            }
        }
        for (auto &it : extra) {
            if (this->weight.dicts.find(it.first) == this->weight.dicts.end()) {
                this->weight.dicts[it.first] = it.second;
            }
        }
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_key_heads") != this->weight.dicts.end()) {
            num_k_heads = atoi(this->weight.dicts["linear_num_key_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_value_heads") != this->weight.dicts.end()) {
            num_v_heads = atoi(this->weight.dicts["linear_num_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_key_head_dim") != this->weight.dicts.end()) {
            head_k_dim = atoi(this->weight.dicts["linear_key_head_dim"].c_str());
        }
        if (this->weight.dicts.find("linear_value_head_dim") != this->weight.dicts.end()) {
            head_v_dim = atoi(this->weight.dicts["linear_value_head_dim"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        std::string partialRotaryFactor = getDictValue(
            "partial_rotary_factor",
            getDictValue("rope_parameters.partial_rotary_factor", "")
        );
        if (!partialRotaryFactor.empty()) {
            rotary_dim = (int)(head_dim * atof(partialRotaryFactor.c_str()) + 1e-5);
        } else {
            rotary_dim = (int)(head_dim * 0.25 + 1e-5); // qwen3.5的默认值
        }
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        std::string ropeType = getDictValue(
            "rope_scaling.type",
            getDictValue("rope_parameters.rope_type", "")
        );
        if (!ropeType.empty()) {
            std::string type = ropeType;
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        std::string ropeTheta = getDictValue(
            "rope_theta",
            getDictValue("rope_parameters.rope_theta", "")
        );
        if (!ropeTheta.empty()) {
            rope_base = atof(ropeTheta.c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        mrope_sections = {11, 11, 10};
        std::string mropeSection = getDictValue(
            "mrope_section",
            getDictValue("rope_parameters.mrope_section", "")
        );
        if (!mropeSection.empty() && mropeSection[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(mropeSection, error);
            if (error.empty() && arr.is_array()) {
                std::vector <int> parsed;
                for (auto &item : arr.array_items()) {
                    parsed.push_back(item.int_value());
                }
                if (!parsed.empty()) {
                    mrope_sections = parsed;
                }
            }
        }
        if ((int) mrope_sections.size() != 3 || mrope_sections[0] + mrope_sections[1] + mrope_sections[2] != rotary_dim / 2) {
            int half = rotary_dim / 2;
            if (half == 32) {
                mrope_sections = {11, 11, 10};
            } else {
                int base = half / 3;
                mrope_sections = {base, base, half - base * 2};
            }
        }
        vision_depth = atoi(getDictValue("vision_config.depth", "0").c_str());
        vision_hidden_size = atoi(getDictValue("vision_config.hidden_size", "0").c_str());
        vision_num_heads = atoi(getDictValue("vision_config.num_heads", "0").c_str());
        vision_intermediate_size = atoi(getDictValue("vision_config.intermediate_size", "0").c_str());
        vision_patch_size = atoi(getDictValue("vision_config.patch_size", "16").c_str());
        vision_temporal_patch_size = atoi(getDictValue("vision_config.temporal_patch_size", "2").c_str());
        vision_spatial_merge_size = atoi(getDictValue("vision_config.spatial_merge_size", "2").c_str());
        vision_out_hidden_size = atoi(getDictValue("vision_config.out_hidden_size", std::to_string(embed_dim)).c_str());
        vision_num_position_embeddings = atoi(getDictValue("vision_config.num_position_embeddings", "0").c_str());
        vision_num_grid_per_side = (vision_num_position_embeddings > 0) ? (int) (sqrt((double) vision_num_position_embeddings) + 0.5) : 0;
        vision_head_dim = (vision_num_heads > 0) ? (vision_hidden_size / vision_num_heads) : 0;
        image_token_id = atoi(getDictValue("image_token_id", "-1").c_str());
        video_token_id = atoi(getDictValue("video_token_id", "-1").c_str());
        vision_start_token_id = atoi(getDictValue("vision_start_token_id", "-1").c_str());
        vision_end_token_id = atoi(getDictValue("vision_end_token_id", "-1").c_str());
        vision_deepstack_visual_indexes.clear();
        auto deepstackIt = this->weight.dicts.find("vision_config.deepstack_visual_indexes");
        if (deepstackIt != this->weight.dicts.end() && !deepstackIt->second.empty() && deepstackIt->second[0] == '[') {
            std::string deepstackError;
            auto deepstackJson = json11::Json::parse(deepstackIt->second, deepstackError);
            if (deepstackError.empty() && deepstackJson.is_array()) {
                for (auto &item : deepstackJson.array_items()) {
                    vision_deepstack_visual_indexes.push_back(item.int_value());
                }
            }
        }
        num_experts = 0;
        if (this->weight.dicts.find("num_experts") != this->weight.dicts.end()) {
            num_experts = atoi(this->weight.dicts["num_experts"].c_str());
        }
        num_experts_per_tok = 0;
        if (this->weight.dicts.find("num_experts_per_tok") != this->weight.dicts.end()) {
            num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        }
        if (this->weight.dicts.find("norm_topk_prob") != this->weight.dicts.end()) {
            norm_topk_prob = IsTrueString(this->weight.dicts["norm_topk_prob"]);
        } else {
            norm_topk_prob = true;
        }
        n_shared_experts = 0;
        if (this->weight.dicts.find("shared_expert_intermediate_size") != this->weight.dicts.end() &&
            atoi(this->weight.dicts["shared_expert_intermediate_size"].c_str()) > 0) {
            n_shared_experts = 1;
        }
        mtp_num_hidden_layers = atoi(getDictValue("mtp_num_hidden_layers", "0").c_str());
        weights.clear();
        biass.clear();
        moeWeightsPrepared = false;

        for (int i = 0; i < mtp_num_hidden_layers; i++) {
            std::string mtpGateWeightName = "mtp.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string mtpUpWeightName = "mtp.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string mtpGateupWeightName = "mtp.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({mtpGateWeightName, mtpUpWeightName},
                                                        mtpGateupWeightName, std::string("linearSwiglu"))})
            );

            std::string mtpQWeightName = "mtp.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string mtpKWeightName = "mtp.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string mtpVWeightName = "mtp.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string mtpMergeQkvWeightName = "mtp.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({mtpQWeightName, mtpKWeightName, mtpVWeightName},
                                                        mtpMergeQkvWeightName, std::string("linear"))})
            );
        }

        for (int i = 0; i < block_cnt; i++) {
            std::string w1WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );

            if (num_experts > 0) {
                std::string sharedGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
                std::string sharedUpWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
                std::string sharedGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({sharedGateWeightName, sharedUpWeightName}, sharedGateupWeightName, std::string("linearSwiglu"))})
                );

                for (int j = 0; j < num_experts; j++) {
                    std::string expertGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                    std::string expertUpWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                    std::string expertGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                    std::string expertDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                    this->weightMergeRules.push_back(
                        WeightMergeRule({WeightMergeRuleSingle({expertGateWeightName, expertUpWeightName}, expertGateupWeightName, std::string("linearSwiglu"))})
                    );
                    this->AddSpecialWeight(expertGateupWeightName, "linearSwiglu", i);
                    this->AddSpecialWeight(expertDownWeightName, "linearColumn", i);
                    this->moeLinears.insert(expertGateWeightName);
                    this->moeLinears.insert(expertUpWeightName);
                    this->moeLinears.insert(expertDownWeightName);
                }
            }

            std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );

            // Merge GDN linear projections: qkv + z -> qkvz, b + a -> ba
            std::string qkvWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkv.weight";
            std::string zWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_z.weight";
            std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qkvWeightName, zWeightName}, qkvzWeightName, std::string("linear"))})
            );

            std::string bWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_b.weight";
            std::string aWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_a.weight";
            std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({bWeightName, aWeightName}, baWeightName, std::string("linear"))})
            );
        }

        float inv_scale = pow((float)head_k_dim, -0.5);
        std::vector <float> v_inv_scale(head_k_dim, inv_scale);
        Data temp(DataType::FLOAT32, std::vector<int>{head_k_dim}, v_inv_scale);
        inv_scale_data.CopyFrom(temp);
    }

    bool Qwen3_5Model::IsFusedMoeLayerPlanned(int layer) const {
        return layer >= 0 &&
               layer < (int)this->moeFusedLayerPlanned.size() &&
               this->moeFusedLayerPlanned[layer];
    }

    bool Qwen3_5Model::HasPlannedFusedMoeLayers() const {
        for (char planned : this->moeFusedLayerPlanned) {
            if (planned) {
                return true;
            }
        }
        return false;
    }

    bool Qwen3_5Model::HasFusedMoeWeights(int layer) const {
        return layer >= 0 &&
               layer < (int)moeGate3DWeights.size() &&
               layer < (int)moeUp3DWeights.size() &&
               layer < (int)moeDown3DWeights.size() &&
               moeGate3DWeights[layer] != nullptr &&
               moeUp3DWeights[layer] != nullptr &&
               moeDown3DWeights[layer] != nullptr;
    }

    static bool Qwen35MoeAllExpertsReady(const std::vector<std::vector<char>> &ready,
                                         int layer, int numExperts) {
        if (layer < 0 || layer >= (int)ready.size() ||
            (int)ready[layer].size() != numExperts) {
            return false;
        }
        for (int expert = 0; expert < numExperts; expert++) {
            if (!ready[layer][expert]) {
                return false;
            }
        }
        return true;
    }

    static bool Qwen35MoeLayerStreamReady(const std::vector<Data*> &gateWeights,
                                          const std::vector<Data*> &upWeights,
                                          const std::vector<Data*> &downWeights,
                                          int layer,
                                          const std::vector<std::vector<char>> &gateReady,
                                          const std::vector<std::vector<char>> &upReady,
                                          const std::vector<std::vector<char>> &downReady,
                                          int numExperts) {
        return layer >= 0 &&
               layer < (int)gateWeights.size() &&
               layer < (int)upWeights.size() &&
               layer < (int)downWeights.size() &&
               gateWeights[layer] != nullptr &&
               upWeights[layer] != nullptr &&
               downWeights[layer] != nullptr &&
               Qwen35MoeAllExpertsReady(gateReady, layer, numExperts) &&
               Qwen35MoeAllExpertsReady(upReady, layer, numExperts) &&
               Qwen35MoeAllExpertsReady(downReady, layer, numExperts);
    }

    bool Qwen3_5Model::ArePlannedFusedMoeLayersReady() const {
        if (!HasPlannedFusedMoeLayers()) {
            return false;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            if (!HasFusedMoeWeights(i) ||
                i >= (int)moeGate3DExpertReady.size() ||
                i >= (int)moeUp3DExpertReady.size() ||
                i >= (int)moeDown3DExpertReady.size() ||
                (int)moeGate3DExpertReady[i].size() != this->num_experts ||
                (int)moeUp3DExpertReady[i].size() != this->num_experts ||
                (int)moeDown3DExpertReady[i].size() != this->num_experts) {
                return false;
            }
            for (int expert = 0; expert < this->num_experts; expert++) {
                if (!moeGate3DExpertReady[i][expert] ||
                    !moeUp3DExpertReady[i][expert] ||
                    !moeDown3DExpertReady[i][expert]) {
                    return false;
                }
            }
        }
        return true;
    }

    Data *Qwen3_5Model::GetFusedMoeWeightForDevice(Data *weight, int device) const {
        AssertInFastLLM(weight != nullptr,
                        "Qwen3.5 MoE fused weight is missing.\n");
        if (!weight->multiDeviceData) {
            return weight;
        }
        auto it = weight->multiDeviceDatas.find(device);
        AssertInFastLLM(it != weight->multiDeviceDatas.end() && it->second != nullptr,
                        "Qwen3.5 MoE fused local shard is missing.\n");
        return it->second;
    }

    void Qwen3_5Model::PrepareFusedMoeLayerForDevices(int layer,
                                                      const std::vector<int> &devices,
                                                      std::map<int, int> ratios) {
#ifdef USE_CUDA
        if (!HasFusedMoeWeights(layer) || devices.empty()) {
            return;
        }
        Data &gate = *moeGate3DWeights[layer];
        Data &up = *moeUp3DWeights[layer];
        Data &down = *moeDown3DWeights[layer];
        AssertInFastLLM(gate.dims.size() == 3 && up.dims.size() == 3 &&
                        down.dims.size() == 3 &&
                        gate.dims[1] == up.dims[1] &&
                        gate.dims[1] == down.dims[2],
                        "Qwen3.5 MoE fused TP weights have incompatible shapes.\n");
        if (devices.size() == 1) {
            int device = devices[0];
            Qwen35PrepareFusedMoeWeightForCuda(gate, device);
            Qwen35PrepareFusedMoeWeightForCuda(up, device);
            Qwen35PrepareFusedMoeWeightForCuda(down, device);
        } else {
            DivisionScheme interScheme = Qwen35BuildFusedInterScheme(gate, devices, ratios);
            Qwen35PrepareFusedShardedWeight(gate, devices, interScheme, 1);
            Qwen35PrepareFusedShardedWeight(up, devices, interScheme, 1);
            Qwen35PrepareFusedShardedWeight(down, devices, interScheme, 2);
        }
#else
        (void)layer;
        (void)devices;
        (void)ratios;
#endif
    }

    void Qwen3_5Model::PrepareFusedMoeWeightsForDevices(const std::vector<int> &devices,
                                                        std::map<int, int> ratios) {
        if (!moeFusedWeightsPrepared || !HasPlannedFusedMoeLayers() || devices.empty()) {
            return;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            AssertInFastLLM(HasFusedMoeWeights(i),
                            "Qwen3.5 MoE fused weights are incomplete.\n");
            PrepareFusedMoeLayerForDevices(i, devices, ratios);
        }
    }

    void Qwen3_5Model::TryFinalizeFusedMoeLayerParts(int layer) {
        if (layer < 0 || layer >= block_cnt || !IsFusedMoeLayerPlanned(layer)) {
            return;
        }
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        bool prepareCuda = !Qwen35MoeDisableFusedMoe() &&
            !Qwen35LayerUsesMappedNonCudaMoe(this, layer) &&
            Qwen35CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) &&
            GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) &&
            !devices.empty();
        if (prepareCuda && Qwen35MoeAllExpertsReady(moeGate3DExpertReady, layer, this->num_experts) &&
            moeGate3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeGate3DWeights[layer], devices[0]);
            } else {
                DivisionScheme interScheme = Qwen35BuildFusedInterScheme(*moeGate3DWeights[layer], devices, ratios);
                Qwen35PrepareFusedShardedWeight(*moeGate3DWeights[layer], devices, interScheme, 1);
            }
        }
        if (prepareCuda && Qwen35MoeAllExpertsReady(moeUp3DExpertReady, layer, this->num_experts) &&
            moeUp3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeUp3DWeights[layer], devices[0]);
            } else if (moeGate3DWeights[layer] != nullptr &&
                       moeGate3DWeights[layer]->multiDeviceData &&
                       !moeGate3DWeights[layer]->tpRanges.empty()) {
                Qwen35PrepareFusedShardedWeight(*moeUp3DWeights[layer], devices,
                                                moeGate3DWeights[layer]->tpRanges, 1);
            }
        }
        if (prepareCuda && Qwen35MoeAllExpertsReady(moeDown3DExpertReady, layer, this->num_experts) &&
            moeDown3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeDown3DWeights[layer], devices[0]);
            } else if (moeGate3DWeights[layer] != nullptr &&
                       moeGate3DWeights[layer]->multiDeviceData &&
                       !moeGate3DWeights[layer]->tpRanges.empty()) {
                Qwen35PrepareFusedShardedWeight(*moeDown3DWeights[layer], devices,
                                                moeGate3DWeights[layer]->tpRanges, 2);
            }
        }
#endif
        bool layerReady = Qwen35MoeLayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                                    layer, moeGate3DExpertReady,
                                                    moeUp3DExpertReady,
                                                    moeDown3DExpertReady, this->num_experts);
        if (!layerReady) {
            return;
        }
        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
    }

    bool Qwen3_5Model::TryConsumeFusedMoeSourceWeight(const std::string &weightName) {
        if (!loadFusedMoePlanned ||
            loadFusedMoeSourceWeights.find(weightName) == loadFusedMoeSourceWeights.end()) {
            return false;
        }
        int layer = -1, expert = -1;
        std::string kind;
        if (!Qwen35MoeParseExpertWeightName(weightName, layer, expert, kind) ||
            layer < 0 || layer >= block_cnt || expert < 0 || expert >= this->num_experts) {
            return false;
        }
        if (!IsFusedMoeLayerPlanned(layer)) {
            return false;
        }
        auto weightIt = this->weight.weight.find(weightName);
        if (weightIt == this->weight.weight.end() || weightIt->second.cpuData == nullptr) {
            return false;
        }
        Data &src = weightIt->second;
        if ((int)moeGate3DWeights.size() != block_cnt) {
            moeGate3DWeights.assign(block_cnt, nullptr);
            moeUp3DWeights.assign(block_cnt, nullptr);
            moeDown3DWeights.assign(block_cnt, nullptr);
        }
        if ((int)moeGate3DExpertReady.size() != block_cnt) {
            moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        }

        if (kind == "gate" || kind == "up") {
            if (src.dims.size() != 2 || !Qwen35MoeIsFusedFp8Type(src.dataType)) {
                return false;
            }
            int inter = src.dims[0];
            int hidden = src.dims[1];
            Data *&target = kind == "gate" ? moeGate3DWeights[layer] : moeUp3DWeights[layer];
            Qwen35MoeEnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                            kind, src, inter, hidden, target);
            Qwen35MoeCopyRows(*target, expert * inter, src, 0, inter);
            Qwen35MoeCopyFp8ScaleRowsToExpert(*target, src, expert, 0, inter);
            if (kind == "gate") {
                moeGate3DExpertReady[layer][expert] = 1;
            } else {
                moeUp3DExpertReady[layer][expert] = 1;
            }
            Qwen35MoeReleaseConsumedSourceWeight(src);
            consumedFusedMoeSourceWeights.insert(weightName);
            TryFinalizeFusedMoeLayerParts(layer);
            return true;
        }

        if (kind == "gateup") {
            if (src.dims.size() != 2 || (src.dims[0] & 1) != 0 ||
                !Qwen35MoeIsFusedFp8Type(src.dataType)) {
                return false;
            }
            int inter = src.dims[0] / 2;
            int hidden = src.dims[1];
            Qwen35MoeEnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                            "gate", src, inter, hidden, moeGate3DWeights[layer]);
            Qwen35MoeEnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                            "up", src, inter, hidden, moeUp3DWeights[layer]);
            Qwen35MoeCopyRows(*moeGate3DWeights[layer], expert * inter, src, 0, inter);
            Qwen35MoeCopyRows(*moeUp3DWeights[layer], expert * inter, src, inter, inter);
            Qwen35MoeCopyFp8ScaleRowsToExpert(*moeGate3DWeights[layer], src, expert, 0, inter);
            Qwen35MoeCopyFp8ScaleRowsToExpert(*moeUp3DWeights[layer], src, expert, inter, inter);
            moeGate3DExpertReady[layer][expert] = 1;
            moeUp3DExpertReady[layer][expert] = 1;
            Qwen35MoeReleaseConsumedSourceWeight(src);
            consumedFusedMoeSourceWeights.insert(weightName);
            TryFinalizeFusedMoeLayerParts(layer);
            return true;
        }

        if (src.dims.size() != 2 || !Qwen35MoeIsFusedFp8Type(src.dataType)) {
            return false;
        }
        int hidden = src.dims[0];
        int inter = src.dims[1];
        Qwen35MoeEnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                        "down", src, hidden, inter, moeDown3DWeights[layer]);
        Qwen35MoeCopyRows(*moeDown3DWeights[layer], expert * hidden, src, 0, hidden);
        Qwen35MoeCopyFp8ScaleRowsToExpert(*moeDown3DWeights[layer], src, expert, 0, hidden);
        moeDown3DExpertReady[layer][expert] = 1;
        Qwen35MoeReleaseConsumedSourceWeight(src);
        consumedFusedMoeSourceWeights.insert(weightName);
        TryFinalizeFusedMoeLayerParts(layer);
        return true;
    }

    bool Qwen3_5Model::TryBuildFusedMoeLayerFromLoaded(int layer) {
        if (layer < 0 || layer >= block_cnt) {
            return false;
        }
        if (!IsFusedMoeLayerPlanned(layer)) {
            return true;
        }
        if (HasFusedMoeWeights(layer)) {
            return Qwen35MoeLayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                             layer, moeGate3DExpertReady,
                                             moeUp3DExpertReady,
                                             moeDown3DExpertReady, this->num_experts);
        }
        if ((int)moeGate3DWeights.size() != block_cnt) {
            moeGate3DWeights.assign(block_cnt, nullptr);
            moeUp3DWeights.assign(block_cnt, nullptr);
            moeDown3DWeights.assign(block_cnt, nullptr);
        }
        if ((int)moeGate3DExpertReady.size() != block_cnt) {
            moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        }
        if (!Qwen35MoeCanBuildFusedLayer(this->weight.weight, layer, this->num_experts)) {
            return false;
        }

#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        bool prepareCuda = !Qwen35MoeDisableFusedMoe() &&
            !Qwen35LayerUsesMappedNonCudaMoe(this, layer) &&
            Qwen35CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) &&
            GetQwen35GPUForwardDevices(this->deviceMap, devices, ratios) &&
            !devices.empty();
        DivisionScheme interScheme;
#endif

        Qwen35MoeBuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                       "gate", moeGate3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeGate3DWeights[layer], devices[0]);
            } else {
                interScheme = Qwen35BuildFusedInterScheme(*moeGate3DWeights[layer], devices, ratios);
                Qwen35PrepareFusedShardedWeight(*moeGate3DWeights[layer], devices, interScheme, 1);
            }
        }
#endif

        Qwen35MoeBuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                       "up", moeUp3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeUp3DWeights[layer], devices[0]);
            } else {
                Qwen35PrepareFusedShardedWeight(*moeUp3DWeights[layer], devices, interScheme, 1);
            }
        }
#endif

        Qwen35MoeBuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                       "down", moeDown3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                Qwen35PrepareFusedMoeWeightForCuda(*moeDown3DWeights[layer], devices[0]);
            } else {
                Qwen35PrepareFusedShardedWeight(*moeDown3DWeights[layer], devices, interScheme, 2);
            }
        }
#endif

        for (int expert = 0; expert < this->num_experts; expert++) {
            std::string expertPrefix = Qwen35MoeExpertPrefix(layer, expert);
            this->weight.weight.erase(expertPrefix + "gate_proj.weight");
            this->weight.weight.erase(expertPrefix + "up_proj.weight");
            this->weight.weight.erase(expertPrefix + "gateup_proj.weight");
            this->weight.weight.erase(expertPrefix + "down_proj.weight");
        }
        if ((int)moeGate3DExpertReady.size() == block_cnt &&
            (int)moeUp3DExpertReady.size() == block_cnt &&
            (int)moeDown3DExpertReady.size() == block_cnt) {
            std::fill(moeGate3DExpertReady[layer].begin(), moeGate3DExpertReady[layer].end(), 1);
            std::fill(moeUp3DExpertReady[layer].begin(), moeUp3DExpertReady[layer].end(), 1);
            std::fill(moeDown3DExpertReady[layer].begin(), moeDown3DExpertReady[layer].end(), 1);
        }
        weights.clear();
        biass.clear();
        threadTpMoeWeights.clear();
        threadTpMoeBiass.clear();
        singleGpuMoeWeights.clear();
        singleGpuMoeBiass.clear();
        threadTpWeightsPrepared.store(false, std::memory_order_release);
        singleGpuWeightsPrepared.store(false, std::memory_order_release);
        moeWeightsPrepared = false;

        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
        return true;
    }

    bool Qwen3_5Model::TryBuildFusedMoeWeightsFromLoaded() {
        if (moeFusedWeightsPrepared && HasPlannedFusedMoeLayers()) {
            return true;
        }
        if (!HasPlannedFusedMoeLayers()) {
            return false;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            if (Qwen35MoeLayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                          i, moeGate3DExpertReady,
                                          moeUp3DExpertReady,
                                          moeDown3DExpertReady, this->num_experts)) {
                continue;
            }
            if (!TryBuildFusedMoeLayerFromLoaded(i)) {
                return false;
            }
        }
        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
        return moeFusedWeightsPrepared;
    }

    void Qwen3_5Model::OnWeightsCreated(const std::set<std::string> &allWeightNames) {
        loadFusedMoePlanned = false;
        loadFusedMoeSourceWeights.clear();
        consumedFusedMoeSourceWeights.clear();
        moeFusedLayerPlanned.clear();
        moeFusedWeightsPrepared = false;
#ifdef USE_CUDA
        if (Qwen35MoeDisableFusedMoe() ||
            !Qwen35CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) ||
            block_cnt <= 0 || this->num_experts <= 0) {
            return;
        }

        std::vector<bool> layerUsesGateup(block_cnt, false);
        std::vector<char> plannedLayers(block_cnt, 0);
        std::set<std::string> plannedSourceWeights;
        for (int i = 0; i < block_cnt; i++) {
            if (Qwen35LayerUsesMappedNonCudaMoe(this, i)) {
                continue;
            }
            bool hasCompleteLayer = true;
            bool layerHasAnyExpert = false;
            int layerInter = -1, layerHidden = -1;
            DataType layerType = DataType::FLOAT32;
            for (int j = 0; j < this->num_experts; j++) {
                std::string gateName = Qwen35MoeExpertWeightName(i, j, "gate");
                std::string upName = Qwen35MoeExpertWeightName(i, j, "up");
                std::string gateupName = Qwen35MoeExpertWeightName(i, j, "gateup");
                std::string downName = Qwen35MoeExpertWeightName(i, j, "down");
                bool hasMergedGateup = allWeightNames.find(gateupName) != allWeightNames.end();
                bool hasGateAndUp = allWeightNames.find(gateName) != allWeightNames.end() &&
                                    allWeightNames.find(upName) != allWeightNames.end();
                bool hasDown = allWeightNames.find(downName) != allWeightNames.end();
                if (hasMergedGateup || hasGateAndUp || hasDown) {
                    layerHasAnyExpert = true;
                }
                if ((!hasMergedGateup && !hasGateAndUp) || !hasDown) {
                    hasCompleteLayer = false;
                    break;
                }

                auto downIt = this->weight.weight.find(downName);
                auto gateupIt = this->weight.weight.find(gateupName);
                auto gateIt = this->weight.weight.find(gateName);
                auto upIt = this->weight.weight.find(upName);
                if (downIt == this->weight.weight.end() ||
                    (hasMergedGateup && gateupIt == this->weight.weight.end()) ||
                    (!hasMergedGateup && (gateIt == this->weight.weight.end() ||
                                          upIt == this->weight.weight.end()))) {
                    hasCompleteLayer = false;
                    break;
                }

                const Data &gateSource = hasMergedGateup ? gateupIt->second : gateIt->second;
                const Data &upSource = hasMergedGateup ? gateupIt->second : upIt->second;
                const Data &downSource = downIt->second;
                if (gateSource.dims.size() != 2 || upSource.dims.size() != 2 ||
                    downSource.dims.size() != 2 ||
                    !Qwen35MoeIsFusedFp8Type(gateSource.dataType) ||
                    gateSource.dataType != upSource.dataType ||
                    gateSource.dataType != downSource.dataType) {
                    hasCompleteLayer = false;
                    break;
                }
                int inter = hasMergedGateup ? gateSource.dims[0] / 2 : gateSource.dims[0];
                int hidden = gateSource.dims[1];
                if (inter <= 0 || hidden <= 0 ||
                    (hasMergedGateup && ((gateSource.dims[0] & 1) != 0 ||
                                         upSource.dims[0] != gateSource.dims[0])) ||
                    (!hasMergedGateup && (upSource.dims[0] != inter ||
                                          upSource.dims[1] != hidden)) ||
                    downSource.dims[0] != hidden || downSource.dims[1] != inter) {
                    hasCompleteLayer = false;
                    break;
                }
                if (j == 0) {
                    layerInter = inter;
                    layerHidden = hidden;
                    layerType = gateSource.dataType;
                    layerUsesGateup[i] = hasMergedGateup;
                } else if (inter != layerInter || hidden != layerHidden ||
                           gateSource.dataType != layerType ||
                           hasMergedGateup != layerUsesGateup[i]) {
                    hasCompleteLayer = false;
                    break;
                }
            }
            if (!hasCompleteLayer) {
                if (layerHasAnyExpert) {
                    plannedSourceWeights.clear();
                    return;
                }
                continue;
            }
            plannedLayers[i] = 1;
            for (int j = 0; j < this->num_experts; j++) {
                if (layerUsesGateup[i]) {
                    plannedSourceWeights.insert(Qwen35MoeExpertWeightName(i, j, "gateup"));
                } else {
                    plannedSourceWeights.insert(Qwen35MoeExpertWeightName(i, j, "gate"));
                    plannedSourceWeights.insert(Qwen35MoeExpertWeightName(i, j, "up"));
                }
                plannedSourceWeights.insert(Qwen35MoeExpertWeightName(i, j, "down"));
            }
        }
        bool hasPlannedLayer = false;
        for (char planned : plannedLayers) {
            if (planned) {
                hasPlannedLayer = true;
                break;
            }
        }
        if (!hasPlannedLayer) {
            return;
        }
        loadFusedMoeSourceWeights.swap(plannedSourceWeights);
        moeFusedLayerPlanned = plannedLayers;
        moeGate3DWeights.assign(block_cnt, nullptr);
        moeUp3DWeights.assign(block_cnt, nullptr);
        moeDown3DWeights.assign(block_cnt, nullptr);
        moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));

        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            std::string gateSourceName = layerUsesGateup[i] ?
                Qwen35MoeExpertWeightName(i, 0, "gateup") :
                Qwen35MoeExpertWeightName(i, 0, "gate");
            std::string upSourceName = layerUsesGateup[i] ?
                Qwen35MoeExpertWeightName(i, 0, "gateup") :
                Qwen35MoeExpertWeightName(i, 0, "up");
            std::string downSourceName = Qwen35MoeExpertWeightName(i, 0, "down");
            Data &gateSource = this->weight.weight[gateSourceName];
            Data &upSource = this->weight.weight[upSourceName];
            Data &downSource = this->weight.weight[downSourceName];
            int inter = layerUsesGateup[i] ? gateSource.dims[0] / 2 : gateSource.dims[0];
            int hidden = gateSource.dims[1];
            Qwen35MoeInitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                              "gate", gateSource, inter, hidden,
                                              moeGate3DWeights[i]);
            Qwen35MoeInitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                              "up", upSource, inter, hidden,
                                              moeUp3DWeights[i]);
            Qwen35MoeInitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                              "down", downSource, hidden, inter,
                                              moeDown3DWeights[i]);
        }

        moeFusedWeightsPrepared = false;
        loadFusedMoePlanned = true;
#endif
    }

    int Qwen3_5Model::GetWeightLoadPriority(
            const std::string &tensorName,
            const std::vector <std::pair <std::string, DataType> > &mappedWeights) const {
        if (!loadFusedMoePlanned) {
            return 0;
        }
        if (loadFusedMoeSourceWeights.find(tensorName) != loadFusedMoeSourceWeights.end()) {
            return Qwen35MoeSourceLoadPriority(tensorName, this->num_experts);
        }
        int priority = 0;
        for (auto &mapped : mappedWeights) {
            if (loadFusedMoeSourceWeights.find(mapped.first) != loadFusedMoeSourceWeights.end()) {
                int mappedPriority = Qwen35MoeSourceLoadPriority(mapped.first, this->num_experts);
                priority = priority == 0 ? mappedPriority : std::min(priority, mappedPriority);
            }
        }
        return priority;
    }

    bool Qwen3_5Model::ShouldLoadWeightSeriallyBeforeOthers(
            const std::string &tensorName,
            const std::vector <std::pair <std::string, DataType> > &mappedWeights) const {
        if (!loadFusedMoePlanned) {
            return false;
        }
        if (loadFusedMoeSourceWeights.find(tensorName) != loadFusedMoeSourceWeights.end()) {
            return true;
        }
        for (auto &mapped : mappedWeights) {
            if (loadFusedMoeSourceWeights.find(mapped.first) != loadFusedMoeSourceWeights.end()) {
                return true;
            }
        }
        return false;
    }

    void Qwen3_5Model::OnWeightLoadGroupStarted(const std::set<std::string> &weightNames) {
        if (!loadFusedMoePlanned || moeFusedWeightsPrepared) {
            return;
        }
        for (auto &weightName : weightNames) {
            if (loadFusedMoeSourceWeights.find(weightName) == loadFusedMoeSourceWeights.end()) {
                continue;
            }
            int layer = -1, expert = -1;
            std::string kind;
            if (!Qwen35MoeParseExpertWeightName(weightName, layer, expert, kind) ||
                layer < 0 || layer >= block_cnt) {
                continue;
            }
            if (kind == "gate" || kind == "gateup") {
                Qwen35MoeAllocateFusedWeightForLoad(moeGate3DWeights[layer]);
            }
            if (kind == "up" || kind == "gateup") {
                Qwen35MoeAllocateFusedWeightForLoad(moeUp3DWeights[layer]);
            }
            if (kind == "down") {
                Qwen35MoeAllocateFusedWeightForLoad(moeDown3DWeights[layer]);
            }
        }
    }

    void Qwen3_5Model::SplitFusedMoeWeightsIfNeeded(const std::string &layerPrefix) {
        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";
        if (this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end()) {
            return;
        }

        auto fusedGateupIt = this->weight.weight.find(fusedGateupName);
        if (fusedGateupIt == this->weight.weight.end()) {
            return;
        }

        auto fusedDownIt = this->weight.weight.find(fusedDownName);
        AssertInFastLLM(fusedDownIt != this->weight.weight.end(), "Qwen3.5 fused MoE weights are incomplete.");
        Data &fusedGateup = fusedGateupIt->second;
        Data &fusedDown = fusedDownIt->second;
        AssertInFastLLM(fusedGateup.dims.size() == 3 && fusedDown.dims.size() == 3,
                        "Qwen3.5 MoE fused expert weights should be 3D.");
        AssertInFastLLM(fusedGateup.dims[0] == num_experts && fusedDown.dims[0] == num_experts,
                        "Qwen3.5 MoE fused expert count mismatch.");

        for (int j = 0; j < num_experts; j++) {
            const std::string expertGateupName = layerPrefix + "experts." + std::to_string(j) + ".gateup_proj.weight";
            const std::string expertDownName = layerPrefix + "experts." + std::to_string(j) + ".down_proj.weight";
            SplitExpertLinearWeight(this->weight.weight[expertGateupName], fusedGateup, expertGateupName, j);
            SplitExpertLinearWeight(this->weight.weight[expertDownName], fusedDown, expertDownName, j);
            RegisterExpertLinearWeight(this->weight.weight[expertGateupName], "linearSwiglu",
                                       this->ShouldRegisterSpecialWeightForDeviceType(expertGateupName, "tfacc"),
                                       this->ShouldRegisterSpecialWeightForDeviceType(expertGateupName, "numa"));
            this->MoveSpecialWeightToCudaIfNeeded(expertGateupName, this->weight.weight[expertGateupName]);
            RegisterExpertLinearWeight(this->weight.weight[expertDownName], "linearColumn",
                                       this->ShouldRegisterSpecialWeightForDeviceType(expertDownName, "tfacc"),
                                       this->ShouldRegisterSpecialWeightForDeviceType(expertDownName, "numa"));
            this->MoveSpecialWeightToCudaIfNeeded(expertDownName, this->weight.weight[expertDownName]);
        }

        this->weight.weight.erase(fusedGateupName);
        this->weight.weight.erase(fusedDownName);
    }

    void Qwen3_5Model::OnWeightLoaded(const std::string &weightName, const std::set<std::string> &finishedWeightNames) {
        if (num_experts <= 0) {
            return;
        }

        if (loadFusedMoePlanned && !moeFusedWeightsPrepared &&
            loadFusedMoeSourceWeights.find(weightName) != loadFusedMoeSourceWeights.end()) {
            int layer = -1, expert = -1;
            std::string kind;
            if (Qwen35MoeParseExpertWeightName(weightName, layer, expert, kind)) {
                if (TryConsumeFusedMoeSourceWeight(weightName)) {
                    return;
                }
                TryBuildFusedMoeLayerFromLoaded(layer);
            }
        }

        std::string layerPrefix;
        if (!TryGetFusedMoeLayerPrefix(weightName, layerPrefix) ||
            !StartWith(layerPrefix, language_prefix + "layers.")) {
            return;
        }

        const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
        const std::string fusedDownName = layerPrefix + "experts.down_proj";
        if (finishedWeightNames.find(fusedGateupName) == finishedWeightNames.end() ||
            finishedWeightNames.find(fusedDownName) == finishedWeightNames.end()) {
            return;
        }

        SplitFusedMoeWeightsIfNeeded(layerPrefix);
    }

    bool Qwen3_5Model::IsWeightConsumedAfterLoad(const std::string &weightName) const {
        return consumedFusedMoeSourceWeights.find(weightName) != consumedFusedMoeSourceWeights.end();
    }

    void Qwen3_5Model::OnWeightLoadGroupFinished() {
        if (consumedFusedMoeSourceWeights.empty()) {
            return;
        }
        for (auto &weightName : consumedFusedMoeSourceWeights) {
            this->weight.weight.erase(weightName);
        }
        consumedFusedMoeSourceWeights.clear();
    }

    void Qwen3_5Model::OnModelWeightsLoaded() {
        if (!loadFusedMoePlanned || moeFusedWeightsPrepared) {
            return;
        }
        std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
        if (TryBuildFusedMoeWeightsFromLoaded()) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
            return;
        }
        loadFusedMoePlanned = false;
        loadFusedMoeSourceWeights.clear();
    }

    bool Qwen3_5Model::ShouldDelaySpecialWeightCudaMove(const std::string &weightName) const {
        return loadFusedMoePlanned &&
               loadFusedMoeSourceWeights.find(weightName) != loadFusedMoeSourceWeights.end();
    }

    void Qwen3_5Model::PrepareMoeWeights() {
        if (moeWeightsPrepared || num_experts <= 0) {
            moeWeightsPrepared = true;
            return;
        }

        weights.clear();
        biass.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);

        for (int i = 0; i < block_cnt; i++) {
            const std::string layerPrefix = language_prefix + "layers." + std::to_string(i) + ".mlp.";
            const std::string routerWeightName = layerPrefix + "gate.weight";
            const std::string fusedGateupName = layerPrefix + "experts.gate_up_proj";
            const std::string fusedDownName = layerPrefix + "experts.down_proj";
            const std::string firstExpertGateupName = layerPrefix + "experts.0.gateup_proj.weight";

            const bool hasMoeLayer =
                this->weight.weight.find(routerWeightName) != this->weight.weight.end() ||
                this->weight.weight.find(fusedGateupName) != this->weight.weight.end() ||
                this->weight.weight.find(firstExpertGateupName) != this->weight.weight.end();
            if (!hasMoeLayer) {
                continue;
            }

            if (HasFusedMoeWeights(i) &&
                this->weight.weight.find(firstExpertGateupName) == this->weight.weight.end()) {
                continue;
            }

            SplitFusedMoeWeightsIfNeeded(layerPrefix);

            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            for (int j = 0; j < num_experts; j++) {
                const std::string expertGateupName = layerPrefix + "experts." + std::to_string(j) + ".gateup_proj.weight";
                const std::string expertDownName = layerPrefix + "experts." + std::to_string(j) + ".down_proj.weight";
                AssertInFastLLM(this->weight.weight.find(expertGateupName) != this->weight.weight.end() &&
                                this->weight.weight.find(expertDownName) != this->weight.weight.end(),
                                "Qwen3.5 MoE expert weights are incomplete.");
                weights[i].push_back(&this->weight[expertGateupName]);
                weights[i].push_back(&this->weight[expertDownName]);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
            }
        }

        moeWeightsPrepared = true;
    }

    void Qwen3_5Model::PrepareGdnWeights() {
        if (gdnMergedWeightsPrepared) {
            return;
        }

        static std::mutex prepareMutex;
        std::lock_guard<std::mutex> guard(prepareMutex);
        if (gdnMergedWeightsPrepared) {
            return;
        }

        for (int i = 0; i < block_cnt; i++) {
            std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
            std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
            std::string mergedWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvzba.weight";
            if (this->weight.weight.find(mergedWeightName) != this->weight.weight.end()) {
                continue;
            }

            auto qkvzIt = this->weight.weight.find(qkvzWeightName);
            auto baIt = this->weight.weight.find(baWeightName);
            if (qkvzIt == this->weight.weight.end() || baIt == this->weight.weight.end()) {
                continue;
            }

            Data &mergedWeight = this->weight.weight[mergedWeightName];
            if (!CreateMergedLinearWeight(qkvzIt->second, baIt->second, mergedWeightName, mergedWeight)) {
                this->weight.weight.erase(mergedWeightName);
            }
        }

        gdnMergedWeightsPrepared = true;
    }

    void Qwen3_5Model::PrepareVision() {
        if (visionPrepared) {
            return;
        }
        AssertInFastLLM(vision_depth > 0 && vision_hidden_size > 0 && vision_num_heads > 0 &&
                        vision_intermediate_size > 0 && vision_out_hidden_size > 0 &&
                        vision_num_position_embeddings > 0 && vision_num_grid_per_side > 0,
                        "Qwen3.5 vision_config is incomplete.");
        AssertInFastLLM(vision_head_dim > 0 && vision_head_dim % 4 == 0,
                        "Qwen3.5 vision head dim must be divisible by 4.");
        AssertInFastLLM(vision_deepstack_visual_indexes.empty(),
                        "Qwen3.5 deepstack vision is not supported yet.");
        AssertInFastLLM(this->weight.weight.find(visual_prefix + "patch_embed.proj.weight") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "patch_embed.proj.bias") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "pos_embed.weight") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.norm.weight") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.norm.bias") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.linear_fc1.weight") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.linear_fc1.bias") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.linear_fc2.weight") != this->weight.weight.end() &&
                        this->weight.weight.find(visual_prefix + "merger.linear_fc2.bias") != this->weight.weight.end(),
                        "Qwen3.5 multimodal needs model.visual.* vision weights.");

        const int maxVisionPos = 8192;
        const int rotaryQuarter = vision_head_dim / 4;
        std::vector<float> invFreq;
        invFreq.reserve(rotaryQuarter);
        for (int i = 0; i < vision_head_dim / 2; i += 2) {
            invFreq.push_back(1.0f / powf(10000.0f, (float) i / (float) (vision_head_dim / 2)));
        }
        std::vector<float> visionSin;
        std::vector<float> visionCos;
        visionSin.reserve((size_t) maxVisionPos * rotaryQuarter);
        visionCos.reserve((size_t) maxVisionPos * rotaryQuarter);
        for (int pos = 0; pos < maxVisionPos; pos++) {
            for (int i = 0; i < rotaryQuarter; i++) {
                float angle = (float) pos * invFreq[i];
                visionSin.push_back(sinf(angle));
                visionCos.push_back(cosf(angle));
            }
        }
        visionSinData.CopyFrom(Data(DataType::FLOAT32, {maxVisionPos, rotaryQuarter}, visionSin));
        visionCosData.CopyFrom(Data(DataType::FLOAT32, {maxVisionPos, rotaryQuarter}, visionCos));
        visionPrepared = true;
    }

    void Qwen3_5Model::ApplyVisionRotary(Data &input, const Data &posX, const Data &posY) {
        AssertInFastLLM(input.dims.size() == 4 && input.dims.back() % 4 == 0,
                        "Qwen3.5 vision rotary expects [batch, seq, heads, dim] with dim divisible by 4.");
        int axis = (int) input.dims.size() - 1;
        int half = input.dims.back() / 2;
        int rotaryHalf = input.dims.back() / 4;
        Data xPart, yPart, rotated;
        Split(input, axis, 0, half, xPart);
        Split(input, axis, half, input.dims.back(), yPart);
        LlamaRotatePosition2DPart(xPart, posX, visionSinData, visionCosData, rotaryHalf, half);
        LlamaRotatePosition2DPart(yPart, posY, visionSinData, visionCosData, rotaryHalf, half);
        Cat(xPart, yPart, axis, rotated);
        input.CopyFrom(rotated);
    }

    void Qwen3_5Model::EncodeVisualItems(const std::vector <Data*> &rawInputs,
                                         const Data *gridThwData,
                                         bool isVideo,
                                         Data &features,
                                         std::vector<std::vector<int>> &gridThwList) {
        gridThwList.clear();
        features = Data();
        if (rawInputs.empty()) {
            return;
        }
        PrepareVision();
        AssertInFastLLM(gridThwData != nullptr, "Qwen3.5 multimodal raw media requires grid_thw metadata.");

        Data gridCpu(*gridThwData);
        gridCpu.ToDevice(DataDevice::CPU);
        if (gridCpu.dataType != DataType::FLOAT32) {
            // ToDataType 通过 Executor 调度时可能优先在 CUDA 上完成转换并释放 cpuData,
            // 后续会直接读取 cpuData, 必须再次 ToDevice(CPU).
            ToDataType(gridCpu, DataType::FLOAT32);
            gridCpu.ToDevice(DataDevice::CPU);
        }
        AssertInFastLLM(gridCpu.dims.size() == 2 && gridCpu.dims[0] == (int) rawInputs.size() && gridCpu.dims[1] == 3,
                        "Qwen3.5 grid_thw should have shape [count, 3].");

        const std::string patchWeightName = visual_prefix + "patch_embed.proj.weight";
        const std::string patchBiasName = visual_prefix + "patch_embed.proj.bias";
        const float attnScale = powf((float) vision_head_dim, -0.5f);
        std::vector<float> mergedFeatures;
        int totalFeatureCount = 0;
        float *gridPtr = (float*) gridCpu.cpuData;

        for (int mediaIndex = 0; mediaIndex < (int) rawInputs.size(); mediaIndex++) {
            std::vector<int> grid = {
                (int) gridPtr[mediaIndex * 3 + 0],
                (int) gridPtr[mediaIndex * 3 + 1],
                (int) gridPtr[mediaIndex * 3 + 2],
            };
            gridThwList.push_back(grid);

            Data rawCpu(*rawInputs[mediaIndex]);
            rawCpu.ToDevice(DataDevice::CPU);
            if (rawCpu.dataType != DataType::FLOAT32) {
                // ToDataType 可能在 CUDA 上完成转换并清空 cpuData,
                // BuildQwen35VisionPatches 直接读取 CPU 指针, 这里需要再次 ToDevice(CPU).
                ToDataType(rawCpu, DataType::FLOAT32);
                rawCpu.ToDevice(DataDevice::CPU);
            }

            int srcFrames = 1, srcH = 0, srcW = 0;
            if (isVideo) {
                AssertInFastLLM(rawCpu.dims.size() == 4 && rawCpu.dims[3] == 3,
                                "Qwen3.5 video raw tensor should have shape [T, H, W, 3].");
                srcFrames = rawCpu.dims[0];
                srcH = rawCpu.dims[1];
                srcW = rawCpu.dims[2];
            } else {
                AssertInFastLLM(rawCpu.dims.size() == 3 && rawCpu.dims[2] == 3,
                                "Qwen3.5 image raw tensor should have shape [H, W, 3].");
                srcFrames = 1;
                srcH = rawCpu.dims[0];
                srcW = rawCpu.dims[1];
            }

            std::vector<float> patchTokens, posH, posW;
            BuildQwen35VisionPatches(
                (float*) rawCpu.cpuData,
                srcFrames,
                srcH,
                srcW,
                grid[0],
                grid[1],
                grid[2],
                vision_patch_size,
                vision_temporal_patch_size,
                vision_spatial_merge_size,
                vision_image_mean,
                vision_image_std,
                patchTokens,
                posH,
                posW
            );

            const int patchDim = 3 * vision_temporal_patch_size * vision_patch_size * vision_patch_size;
            const int patchCount = grid[0] * grid[1] * grid[2];
            AssertInFastLLM((int) posH.size() == patchCount && (int) posW.size() == patchCount,
                            "Qwen3.5 vision patch positions count mismatch.");
            AssertInFastLLM((int) patchTokens.size() == patchCount * patchDim,
                            "Qwen3.5 vision patch packing size mismatch.");

            Data pixelInput(DataType::FLOAT32, {patchCount, patchDim}, patchTokens);
            Data pixelOnDevice(pixelInput);
            Data &patchWeight = this->weight[patchWeightName];
            if (patchWeight.dataDevice != DataDevice::CPU) {
                if (!patchWeight.dataDeviceIds.empty()) {
                    pixelOnDevice.ToDevice(patchWeight.dataDevice, patchWeight.dataDeviceIds);
                } else {
                    pixelOnDevice.ToDevice(patchWeight.dataDevice);
                }
            }

            Data hiddenStates;
            Linear(pixelOnDevice, patchWeight, this->weight[patchBiasName], hiddenStates);
            if (hiddenStates.dataType != this->dataType) {
                ToDataType(hiddenStates, this->dataType);
            }
            hiddenStates.Reshape({1, patchCount, vision_hidden_size});

            Data posWeightCpu(this->weight[visual_prefix + "pos_embed.weight"]);
            posWeightCpu.ToDevice(DataDevice::CPU);
            if (posWeightCpu.dataType != DataType::FLOAT32) {
                // ToDataType 经 Executor 调度可能在 CUDA 上完成并释放 CPU 镜像,
                // 后续会以 CPU 指针读取, 必须再次 ToDevice(CPU).
                ToDataType(posWeightCpu, DataType::FLOAT32);
                posWeightCpu.ToDevice(DataDevice::CPU);
            }
            AssertInFastLLM(posWeightCpu.dims.size() == 2 && posWeightCpu.dims[1] == vision_hidden_size,
                            "Qwen3.5 vision pos_embed weight shape is invalid.");
            std::vector<float> posEmbVec;
            BuildQwen35InterpolatedPosEmb(
                (float*) posWeightCpu.cpuData,
                vision_hidden_size,
                vision_num_grid_per_side,
                grid[0],
                grid[1],
                grid[2],
                vision_spatial_merge_size,
                posEmbVec
            );
            Data posEmb(DataType::FLOAT32, {1, patchCount, vision_hidden_size}, posEmbVec);
            if (hiddenStates.dataDevice != DataDevice::CPU) {
                if (!hiddenStates.dataDeviceIds.empty()) {
                    posEmb.ToDevice(hiddenStates.dataDevice, hiddenStates.dataDeviceIds);
                } else {
                    posEmb.ToDevice(hiddenStates.dataDevice);
                }
            }
            if (posEmb.dataType != hiddenStates.dataType) {
                ToDataType(posEmb, hiddenStates.dataType);
            }
            AddTo(hiddenStates, posEmb);

            Data posHData(DataType::FLOAT32, {1, patchCount}, posH);
            Data posWData(DataType::FLOAT32, {1, patchCount}, posW);

            Data blockInput, qkv, q, k, v, attnOutput, residual, mlpHidden, mlpOutput;
            for (int layer = 0; layer < vision_depth; layer++) {
                const std::string pre = visual_prefix + "blocks." + std::to_string(layer);
                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(hiddenStates,
                          this->weight[pre + ".norm1.weight"],
                          this->weight[pre + ".norm1.bias"],
                          -1,
                          blockInput);
                Linear(blockInput, this->weight[pre + ".attn.qkv.weight"], this->weight[pre + ".attn.qkv.bias"], qkv);
                Split(qkv, -1, 0, vision_hidden_size, q);
                Split(qkv, -1, vision_hidden_size, vision_hidden_size * 2, k);
                Split(qkv, -1, vision_hidden_size * 2, vision_hidden_size * 3, v);
                q.Reshape({1, patchCount, vision_num_heads, vision_head_dim});
                k.Reshape({1, patchCount, vision_num_heads, vision_head_dim});
                v.Reshape({1, patchCount, vision_num_heads, vision_head_dim});
                ApplyVisionRotary(q, posHData, posWData);
                ApplyVisionRotary(k, posHData, posWData);
                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                q.Reshape({vision_num_heads, patchCount, vision_head_dim});
                k.Reshape({vision_num_heads, patchCount, vision_head_dim});
                v.Reshape({vision_num_heads, patchCount, vision_head_dim});
                Attention(q, k, v, *GetEmptyData(), attnOutput, 1, attnScale, 2);
                PermuteSelf(attnOutput, {1, 0, 2});
                attnOutput.Reshape({patchCount, 1, vision_hidden_size});
                PermuteSelf(attnOutput, {1, 0, 2});
                Linear(attnOutput,
                       this->weight[pre + ".attn.proj.weight"],
                       this->weight[pre + ".attn.proj.bias"],
                       attnOutput);
                if (attnOutput.dataType != residual.dataType) {
                    ToDataType(attnOutput, residual.dataType);
                }
                AddTo(residual, attnOutput);
                hiddenStates.CopyFrom(residual);

                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(hiddenStates,
                          this->weight[pre + ".norm2.weight"],
                          this->weight[pre + ".norm2.bias"],
                          -1,
                          blockInput);
                Linear(blockInput,
                       this->weight[pre + ".mlp.linear_fc1.weight"],
                       this->weight[pre + ".mlp.linear_fc1.bias"],
                       mlpHidden);
                if (mlpHidden.dataType != DataType::FLOAT32) {
                    ToDataType(mlpHidden, DataType::FLOAT32);
                }
                GeluNew(mlpHidden, mlpHidden);
                Linear(mlpHidden,
                       this->weight[pre + ".mlp.linear_fc2.weight"],
                       this->weight[pre + ".mlp.linear_fc2.bias"],
                       mlpOutput);
                if (mlpOutput.dataType != residual.dataType) {
                    ToDataType(mlpOutput, residual.dataType);
                }
                AddTo(residual, mlpOutput);
                hiddenStates.CopyFrom(residual);
            }

            Data mergerInput;
            LayerNorm(hiddenStates,
                      this->weight[visual_prefix + "merger.norm.weight"],
                      this->weight[visual_prefix + "merger.norm.bias"],
                      -1,
                      mergerInput);
            const int mergeUnit = vision_spatial_merge_size * vision_spatial_merge_size;
            AssertInFastLLM(patchCount % mergeUnit == 0, "Qwen3.5 merger input length must be divisible by merge unit.");
            mergerInput.Reshape({patchCount / mergeUnit, vision_hidden_size * mergeUnit});
            Linear(mergerInput,
                   this->weight[visual_prefix + "merger.linear_fc1.weight"],
                   this->weight[visual_prefix + "merger.linear_fc1.bias"],
                   mlpHidden);
            if (mlpHidden.dataType != DataType::FLOAT32) {
                ToDataType(mlpHidden, DataType::FLOAT32);
            }
            GeluNew(mlpHidden, mlpHidden);
            Linear(mlpHidden,
                   this->weight[visual_prefix + "merger.linear_fc2.weight"],
                   this->weight[visual_prefix + "merger.linear_fc2.bias"],
                   mlpOutput);
            mlpOutput.ToDevice(DataDevice::CPU);
            if (mlpOutput.dataType != DataType::FLOAT32) {
                // ToDataType 经 Executor 调度可能在 CUDA 上完成并释放 CPU 镜像,
                // mergedFeatures 直接读取 cpuData, 必须再次 ToDevice(CPU).
                ToDataType(mlpOutput, DataType::FLOAT32);
                mlpOutput.ToDevice(DataDevice::CPU);
            }
            AssertInFastLLM(mlpOutput.dims.size() == 2 && mlpOutput.dims[1] == vision_out_hidden_size,
                            "Qwen3.5 merger output shape is invalid.");
            mergedFeatures.insert(
                mergedFeatures.end(),
                (float*) mlpOutput.cpuData,
                (float*) mlpOutput.cpuData + mlpOutput.Count(0)
            );
            totalFeatureCount += mlpOutput.dims[0];
        }

        if (totalFeatureCount > 0) {
            features.CopyFrom(Data(DataType::FLOAT32, {1, totalFeatureCount, vision_out_hidden_size}, mergedFeatures));
        }
    }

    void Qwen3_5Model::BuildMultimodalPositionData(const Data &inputIds,
                                                   const std::vector<std::vector<int>> &imageGridThwList,
                                                   const std::vector<std::vector<int>> &videoGridThwList,
                                                   Data &mmTokenTypeIds,
                                                   Data &mropePositionIds,
                                                   Data &mropePositionDelta) {
        Data idsCpu(inputIds);
        idsCpu.ToDevice(DataDevice::CPU);
        if (idsCpu.dataType != DataType::FLOAT32) {
            // ToDataType 经 Executor 调度可能在 CUDA 上完成并释放 CPU 镜像,
            // 后续以 CPU 指针读取 token id, 这里需要再次 ToDevice(CPU).
            ToDataType(idsCpu, DataType::FLOAT32);
            idsCpu.ToDevice(DataDevice::CPU);
        }
        AssertInFastLLM(idsCpu.dims.size() == 2 && idsCpu.dims[0] == 1,
                        "Qwen3.5 multimodal position generation expects [1, seq].");

        const int seqLen = idsCpu.dims[1];
        float *idPtr = (float*) idsCpu.cpuData;
        std::vector<int> mmTypes(seqLen, 0);
        for (int i = 0; i < seqLen; i++) {
            int tokenId = (int) idPtr[i];
            if (tokenId == image_token_id) {
                mmTypes[i] = 1;
            } else if (tokenId == video_token_id) {
                mmTypes[i] = 2;
            }
        }

        std::vector<std::vector<int>> repeatedVideoGridThwList;
        for (auto &grid : videoGridThwList) {
            for (int t = 0; t < grid[0]; t++) {
                repeatedVideoGridThwList.push_back({1, grid[1], grid[2]});
            }
        }

        std::vector<float> mmTypesFloat(seqLen, 0.0f);
        std::vector<float> positions0, positions1, positions2;
        positions0.reserve(seqLen);
        positions1.reserve(seqLen);
        positions2.reserve(seqLen);
        int currentPos = 0;
        int imageIndex = 0, videoIndex = 0;
        for (int i = 0; i < seqLen;) {
            int tokenType = mmTypes[i];
            int j = i;
            while (j < seqLen && mmTypes[j] == tokenType) {
                j++;
            }
            int groupLen = j - i;
            for (int k = i; k < j; k++) {
                mmTypesFloat[k] = (float) tokenType;
            }
            if (tokenType == 0) {
                for (int k = 0; k < groupLen; k++) {
                    positions0.push_back((float) (currentPos + k));
                    positions1.push_back((float) (currentPos + k));
                    positions2.push_back((float) (currentPos + k));
                }
                currentPos += groupLen;
            } else {
                const std::vector<int> *gridPtr = nullptr;
                if (tokenType == 1) {
                    AssertInFastLLM(imageIndex < (int) imageGridThwList.size(),
                                    "Qwen3.5 image grid_thw count does not match image placeholders.");
                    gridPtr = &imageGridThwList[imageIndex++];
                } else {
                    AssertInFastLLM(videoIndex < (int) repeatedVideoGridThwList.size(),
                                    "Qwen3.5 video grid_thw count does not match video placeholders.");
                    gridPtr = &repeatedVideoGridThwList[videoIndex++];
                }
                const std::vector<int> &grid = *gridPtr;
                const int llmGridT = grid[0];
                const int llmGridH = grid[1] / vision_spatial_merge_size;
                const int llmGridW = grid[2] / vision_spatial_merge_size;
                AssertInFastLLM(llmGridT > 0 && llmGridH > 0 && llmGridW > 0,
                                "Qwen3.5 grid_thw is invalid for multimodal positions.");
                AssertInFastLLM(groupLen == llmGridT * llmGridH * llmGridW,
                                "Qwen3.5 placeholder count does not match computed multimodal tokens.");
                for (int h = 0; h < llmGridH; h++) {
                    for (int t = 0; t < llmGridT; t++) {
                        (void) t;
                        for (int w = 0; w < llmGridW; w++) {
                            positions0.push_back((float) currentPos);
                            positions1.push_back((float) (currentPos + h));
                            positions2.push_back((float) (currentPos + w));
                        }
                    }
                }
                currentPos += std::max(grid[1], grid[2]) / vision_spatial_merge_size;
            }
            i = j;
        }

        AssertInFastLLM(imageIndex == (int) imageGridThwList.size(),
                        "Qwen3.5 image grid_thw metadata was not fully consumed.");
        AssertInFastLLM(videoIndex == (int) repeatedVideoGridThwList.size(),
                        "Qwen3.5 video grid_thw metadata was not fully consumed.");

        mmTokenTypeIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, mmTypesFloat));
        std::vector<float> flatPositions;
        flatPositions.reserve((size_t) seqLen * 3);
        flatPositions.insert(flatPositions.end(), positions0.begin(), positions0.end());
        flatPositions.insert(flatPositions.end(), positions1.begin(), positions1.end());
        flatPositions.insert(flatPositions.end(), positions2.begin(), positions2.end());
        mropePositionIds.CopyFrom(Data(DataType::FLOAT32, {3, seqLen}, flatPositions));
        float maxPos = 0.0f;
        if (!positions0.empty()) {
            maxPos = std::max(
                *std::max_element(positions0.begin(), positions0.end()),
                std::max(
                    *std::max_element(positions1.begin(), positions1.end()),
                    *std::max_element(positions2.begin(), positions2.end())
                )
            );
        }
        float delta = positions0.empty() ? 0.0f : (maxPos + 1.0f - (float) seqLen);
        mropePositionDelta.CopyFrom(Data(DataType::FLOAT32, {1, 1}, std::vector<float>{delta}));
    }

    Data Qwen3_5Model::BuildFlattenedPositionIds(const std::vector <Data*> &positionIds,
                                                const std::vector <int> &seqLens,
                                                bool all1) {
        Data allPositionIds;
        if (positionIds.empty() || positionIds[0] == nullptr) {
            return allPositionIds;
        }
        if (positionIds.size() == 1 && positionIds[0]->dims.size() == 2 && positionIds[0]->dims[0] == 3) {
            allPositionIds.CopyFrom(*positionIds[0]);
            allPositionIds.ToDevice(DataDevice::CPU);
            if (allPositionIds.dataType != DataType::FLOAT32) {
                // ToDataType 经 Executor 调度可能优先在 CUDA 上完成转换并释放 cpuData,
                // 调用方默认期望返回值在 CPU 上, 这里需要再次 ToDevice(CPU).
                ToDataType(allPositionIds, DataType::FLOAT32);
                allPositionIds.ToDevice(DataDevice::CPU);
            }
            return allPositionIds;
        }

        int totalLen = 0;
        for (int len : seqLens) {
            totalLen += len;
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;
            for (int b = 0; b < (int) positionIds.size(); b++) {
                vPositionIds.push_back(((float*) positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, totalLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;
            for (int b = 0; b < (int) positionIds.size(); b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*) positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, totalLen}, vPositionIds));
        }
        return allPositionIds;
    }

    void Qwen3_5Model::MergeMultimodalFeaturesIntoText(const Data &mmTokenTypeIds,
                                                       const Data *imageEmbeds,
                                                       const Data *videoEmbeds,
                                                       Data &hiddenStates) {
        Data mmCpu(mmTokenTypeIds);
        mmCpu.ToDevice(DataDevice::CPU);
        if (mmCpu.dataType != DataType::FLOAT32) {
            // 注意: ToDataType 经过 Executor 调度可能优先在 CUDA 上完成,
            // 转换后会释放 CPU 镜像, 因此后续访问 cpuData 前要再次 ToDevice(CPU).
            ToDataType(mmCpu, DataType::FLOAT32);
            mmCpu.ToDevice(DataDevice::CPU);
        }
        DataType hiddenType = hiddenStates.dataType;
        hiddenStates.ToDevice(DataDevice::CPU);
        AssertInFastLLM(hiddenType == DataType::FLOAT32 ||
                        hiddenType == DataType::FLOAT16 ||
                        hiddenType == DataType::BFLOAT16,
                        "Qwen3.5 multimodal hidden states must be float32/float16/bfloat16.");
        AssertInFastLLM(hiddenStates.dims.size() == 3 && hiddenStates.dims[0] == 1,
                        "Qwen3.5 multimodal currently supports a single text batch only.");
        AssertInFastLLM(mmCpu.Count(0) == hiddenStates.dims[1],
                        "Qwen3.5 mm_token_type_ids length must match input sequence length.");

        Data imageCpu, videoCpu;
        int imageCount = 0, videoCount = 0;
        int hiddenSize = hiddenStates.dims[2];
        size_t rowBytes = (size_t) hiddenSize * hiddenStates.unitSize / hiddenStates.unitSizeDiv;
        uint8_t *imagePtr = nullptr, *videoPtr = nullptr;

        auto prepareFeatures = [&](const Data *src, Data &dst, int &count, uint8_t* &ptr) {
            count = 0;
            ptr = nullptr;
            if (src == nullptr || src->dims.empty()) {
                return;
            }
            dst.CopyFrom(*src);
            dst.ToDevice(DataDevice::CPU);
            if (dst.dataType != hiddenType) {
                // 注意: ToDataType 可能会通过 Executor 自动把数据搬到 CUDA 上完成转换,
                // 之后 cpuData 会被释放. 因此完成转换后要再次显式 ToDevice(CPU).
                ToDataType(dst, hiddenType);
                dst.ToDevice(DataDevice::CPU);
            }
            if (dst.dims.size() == 3 && dst.dims[0] == 1) {
                dst.Reshape({dst.dims[1], dst.dims[2]});
            }
            AssertInFastLLM(dst.dims.size() == 2, "Qwen3.5 multimodal features should have shape [tokens, hidden].");
            AssertInFastLLM(dst.dims[1] == hiddenSize, "Qwen3.5 multimodal hidden size mismatch.");
            count = dst.dims[0];
            ptr = (uint8_t*) dst.cpuData;
        };

        prepareFeatures(imageEmbeds, imageCpu, imageCount, imagePtr);
        prepareFeatures(videoEmbeds, videoCpu, videoCount, videoPtr);

        uint8_t *hiddenPtr = (uint8_t*) hiddenStates.cpuData;
        float *mmPtr = (float*) mmCpu.cpuData;
        int nextImage = 0, nextVideo = 0;
        for (int i = 0; i < hiddenStates.dims[1]; i++) {
            int tokenType = (int) mmPtr[i];
            if (tokenType == 1) {
                AssertInFastLLM(nextImage < imageCount, "Qwen3.5 image embeds count does not match image placeholders.");
                memcpy(hiddenPtr + (size_t) i * rowBytes,
                       imagePtr + (size_t) nextImage * rowBytes,
                       rowBytes);
                nextImage++;
            } else if (tokenType == 2) {
                AssertInFastLLM(nextVideo < videoCount, "Qwen3.5 video embeds count does not match video placeholders.");
                memcpy(hiddenPtr + (size_t) i * rowBytes,
                       videoPtr + (size_t) nextVideo * rowBytes,
                       rowBytes);
                nextVideo++;
            }
        }
        AssertInFastLLM(nextImage == imageCount, "Qwen3.5 image embeds were not fully consumed.");
        AssertInFastLLM(nextVideo == videoCount, "Qwen3.5 video embeds were not fully consumed.");
    }

    void Qwen3_5Model::ApplyMultimodalRotary(Data &input, const Data &positionIds, float ropeScale) {
        if (positionIds.dims.size() == 2 && positionIds.dims[0] == 3) {
            fastllm::Qwen35InterleavedRope(
                input, positionIds, rotary_dim,
                mrope_sections[0], mrope_sections[1], mrope_sections[2],
                rope_base, ropeScale);
        } else {
            fastllm::RopeEncoding(input, positionIds, rotary_dim, rope_base, ropeScale);
        }
    }

    bool Qwen3_5Model::HasMtpWeights() const {
        if (mtp_num_hidden_layers <= 0) {
            return false;
        }
        bool hasMtpQkv =
            weight.weight.find("mtp.layers.0.self_attn.mergeqkv.weight") != weight.weight.end() ||
            (weight.weight.find("mtp.layers.0.self_attn.q_proj.weight") != weight.weight.end() &&
             weight.weight.find("mtp.layers.0.self_attn.k_proj.weight") != weight.weight.end() &&
             weight.weight.find("mtp.layers.0.self_attn.v_proj.weight") != weight.weight.end());
        return weight.weight.find("mtp.fc.weight") != weight.weight.end() &&
               weight.weight.find("mtp.norm.weight") != weight.weight.end() &&
               weight.weight.find("mtp.pre_fc_norm_embedding.weight") != weight.weight.end() &&
               weight.weight.find("mtp.pre_fc_norm_hidden.weight") != weight.weight.end() &&
               hasMtpQkv &&
               weight.weight.find("mtp.layers.0.self_attn.o_proj.weight") != weight.weight.end() &&
               weight.weight.find("mtp.layers.0.mlp.gateup_proj.weight") != weight.weight.end() &&
               weight.weight.find("mtp.layers.0.mlp.down_proj.weight") != weight.weight.end();
    }

    void Qwen3_5Model::AddMtpRmsNormOffset() {
        if (mtp_num_hidden_layers <= 0) {
            return;
        }
        Add1(this->weight["mtp.pre_fc_norm_embedding.weight"]);
        Add1(this->weight["mtp.pre_fc_norm_hidden.weight"]);
        Add1(this->weight["mtp.norm.weight"]);
        for (int i = 0; i < mtp_num_hidden_layers; i++) {
            std::string prefix = "mtp.layers." + std::to_string(i) + ".";
            Add1(this->weight[prefix + "input_layernorm.weight"]);
            Add1(this->weight[prefix + "self_attn.q_norm.weight"]);
            Add1(this->weight[prefix + "self_attn.k_norm.weight"]);
            Add1(this->weight[prefix + "post_attention_layernorm.weight"]);
        }
    }

    void Qwen3_5Model::PrepareMtpWeightsForDevice(int device, bool includeSharedWeights) {
#ifdef USE_CUDA
        if (!HasMtpWeights()) {
            return;
        }
        if (mtpWeightsPrepared && mtpWeightsPreparedDevice == device) {
            return;
        }
        auto moveWeight = [&](const std::string &name) {
            auto it = weight.weight.find(name);
            if (it != weight.weight.end() && !it->second.dims.empty()) {
                it->second.ToDevice(DataDevice::CUDA, {device}, true);
            }
        };
        if (includeSharedWeights && GetCudaEmbedding() && !GetLowMemMode()) {
            moveWeight(language_prefix + "embed_tokens.weight");
        }
        if (includeSharedWeights) {
            moveWeight("lm_head.weight");
        }
        moveWeight("mtp.fc.weight");
        moveWeight("mtp.pre_fc_norm_embedding.weight");
        moveWeight("mtp.pre_fc_norm_hidden.weight");
        moveWeight("mtp.norm.weight");
        for (int i = 0; i < mtp_num_hidden_layers; i++) {
            std::string prefix = "mtp.layers." + std::to_string(i) + ".";
            bool hasMergedQkv = weight.weight.find(prefix + "self_attn.mergeqkv.weight") != weight.weight.end();
            moveWeight(prefix + "input_layernorm.weight");
            if (hasMergedQkv) {
                moveWeight(prefix + "self_attn.mergeqkv.weight");
            } else {
                moveWeight(prefix + "self_attn.q_proj.weight");
                moveWeight(prefix + "self_attn.k_proj.weight");
                moveWeight(prefix + "self_attn.v_proj.weight");
            }
            moveWeight(prefix + "self_attn.o_proj.weight");
            moveWeight(prefix + "self_attn.q_norm.weight");
            moveWeight(prefix + "self_attn.k_norm.weight");
            moveWeight(prefix + "post_attention_layernorm.weight");
            weight[prefix + "mlp.gateup_proj.weight"].tpPackType = TP_PACK_GATEUP;
            moveWeight(prefix + "mlp.gateup_proj.weight");
            moveWeight(prefix + "mlp.down_proj.weight");
        }
        if (includeSharedWeights) {
            mtpWeightsPrepared = true;
            mtpWeightsPreparedDevice = device;
        }
#else
        (void)device;
        (void)includeSharedWeights;
#endif
    }

    Data Qwen3_5Model::BuildMtpPositionIds(const Data &positionIds, int row, int delta) {
        return BuildMtpPositionIdsSlice(positionIds, row, row + 1, delta);
    }

    Data Qwen3_5Model::BuildMtpPositionIdsSlice(const Data &positionIds, int begin, int end, int delta) {
        Data cpuPositionIds;
        cpuPositionIds.CopyFrom(positionIds);
        cpuPositionIds.ToDevice(DataDevice::CPU);
        if (cpuPositionIds.dataType != DataType::FLOAT32) {
            ToDataType(cpuPositionIds, DataType::FLOAT32);
            cpuPositionIds.ToDevice(DataDevice::CPU);
        }
        if (cpuPositionIds.dims.size() != 2 || cpuPositionIds.dims[1] <= 0) {
            int len = std::max(1, end - begin);
            std::vector<float> values(len);
            for (int i = 0; i < len; i++) {
                values[i] = (float)(begin + i + delta);
            }
            return Data(DataType::FLOAT32, {1, len}, values);
        }
        int rows = cpuPositionIds.dims[0];
        int cols = cpuPositionIds.dims[1];
        begin = std::max(0, std::min(begin, cols - 1));
        end = std::max(begin + 1, std::min(end, cols));
        int len = end - begin;
        float *ptr = (float*)cpuPositionIds.cpuData;
        std::vector<float> values(rows * len);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < len; c++) {
                values[r * len + c] = ptr[r * cols + begin + c] + (float)delta;
            }
        }
        return Data(DataType::FLOAT32, {rows, len}, values);
    }

    int Qwen3_5Model::RunMtpGreedyDraft(int device, const std::vector<int> &devices,
                                        MtpKvCache &cache,
                                        const Data &targetHiddenStates,
                                        const std::vector<int> &inputTokens,
                                        const Data &positionIds, int sampleRow,
                                        Data *sampledHiddenStates,
                                        bool cacheOnly) {
#ifndef USE_CUDA
        (void)device;
        (void)devices;
        (void)cache;
        (void)targetHiddenStates;
        (void)inputTokens;
        (void)positionIds;
        (void)sampleRow;
        (void)sampledHiddenStates;
        (void)cacheOnly;
        return -1;
#else
        std::vector<int> draftDevices = devices.empty() ? std::vector<int>{device} : devices;
        bool tensorParallelDraft = draftDevices.size() > 1;
        PrepareMtpWeightsForDevice(device, !tensorParallelDraft);
        AssertInFastLLM(HasMtpWeights(), "Qwen3.5 MTP weights are missing.\n");
        int seqLen = (int)inputTokens.size();
        AssertInFastLLM(seqLen > 0, "Qwen3.5 MTP needs at least one input token.\n");

        Qwen35ScopedGenericExecutor mtpExecutor("cuda:" + std::to_string(device));
        FastllmCudaSetDevice(device);

        std::vector<float> tokenValues(seqLen);
        for (int i = 0; i < seqLen; i++) {
            tokenValues[i] = (float)inputTokens[i];
        }
        Data tokenIds(DataType::FLOAT32, {1, seqLen}, tokenValues);
        Data hiddenStates;
        const Data *hiddenInput = &targetHiddenStates;
        if (targetHiddenStates.dataType != this->dataType) {
            hiddenStates.CopyFrom(targetHiddenStates);
            ToDataType(hiddenStates, this->dataType);
            hiddenInput = &hiddenStates;
        }
        AssertInFastLLM(hiddenInput->dims.size() == 3 && hiddenInput->dims[1] == seqLen,
                        "Qwen3.5 MTP hidden states length mismatch.\n");
        Data mtpPositionIds;
        mtpPositionIds.CopyFrom(positionIds);
        mtpPositionIds.ToDevice(DataDevice::CUDA, {device}, true);

        Data inputEmbeds, normEmbeds, normHidden, fusedInput;
        Data &embedWeight = weight[language_prefix + "embed_tokens.weight"];
        Data *embedWeightForMtp = &embedWeight;
        if (embedWeight.multiDeviceData) {
            auto embedIt = embedWeight.multiDeviceDatas.find(device);
            if (embedIt != embedWeight.multiDeviceDatas.end() && embedIt->second != nullptr) {
                embedWeightForMtp = embedIt->second;
            }
        }
        bool useCudaEmbeddingForMtp =
            GetCudaEmbedding() && !GetLowMemMode() &&
            embedWeightForMtp->dataDevice == DataDevice::CUDA &&
            embedWeightForMtp->cudaData != nullptr &&
            !embedWeightForMtp->dataDeviceIds.empty() &&
            embedWeightForMtp->dataDeviceIds[0] == device;
        if (!useCudaEmbeddingForMtp) {
            Qwen35CpuEmbeddingDirect(tokenIds, embedWeight, inputEmbeds, hiddenInput->dataType);
            inputEmbeds.ToDevice(DataDevice::CUDA, {device}, true);
        } else {
            tokenIds.ToDevice(DataDevice::CUDA, {device}, true);
            Embedding(tokenIds, *embedWeightForMtp, inputEmbeds);
        }
        if (inputEmbeds.dataType != hiddenInput->dataType) {
            ToDataType(inputEmbeds, hiddenInput->dataType);
        }
        RMSNorm(inputEmbeds, weight["mtp.pre_fc_norm_embedding.weight"], rms_norm_eps, normEmbeds);
        RMSNorm(*hiddenInput, weight["mtp.pre_fc_norm_hidden.weight"], rms_norm_eps, normHidden);
        Cat(normEmbeds, normHidden, -1, fusedInput);
        Linear(fusedInput, weight["mtp.fc.weight"], *GetEmptyData(), hiddenStates);

        std::string prefix = "mtp.layers.0.";
        Data attenInput, qgate, q, gate, k, v, attenOutput, projected, mergedQkv;
        RMSNorm(hiddenStates, weight[prefix + "input_layernorm.weight"], rms_norm_eps, attenInput);
        std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
        if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
            Linear(attenInput, weight[mergeQkvWeightName], *GetEmptyData(), mergedQkv);
            int qgateDim = num_attention_heads * this->head_dim * 2;
            int kvDim = num_key_value_heads * this->head_dim;
            Split(mergedQkv, -1, 0, qgateDim, qgate);
            Split(mergedQkv, -1, qgateDim, qgateDim + kvDim, k);
            Split(mergedQkv, -1, qgateDim + kvDim, qgateDim + kvDim * 2, v);
        } else {
            Linear(attenInput, weight[prefix + "self_attn.q_proj.weight"], *GetEmptyData(), qgate);
            Linear(attenInput, weight[prefix + "self_attn.k_proj.weight"], *GetEmptyData(), k);
            Linear(attenInput, weight[prefix + "self_attn.v_proj.weight"], *GetEmptyData(), v);
        }

        qgate.Reshape({1, seqLen, -1, this->head_dim * 2});
        Split(qgate, -1, 0, this->head_dim, q);
        Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
        gate.Reshape({1, seqLen, -1});
        k.Reshape({1, seqLen, -1, this->head_dim});
        v.Reshape({1, seqLen, -1, this->head_dim});
        RMSNorm(q, weight[prefix + "self_attn.q_norm.weight"], rms_norm_eps, q);
        RMSNorm(k, weight[prefix + "self_attn.k_norm.weight"], rms_norm_eps, k);
        float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
        ApplyMultimodalRotary(q, mtpPositionIds, ropeScale);
        ApplyMultimodalRotary(k, mtpPositionIds, ropeScale);
        PermuteSelf(q, {0, 2, 1, 3});
        PermuteSelf(k, {0, 2, 1, 3});
        PermuteSelf(v, {0, 2, 1, 3});
        q.Reshape({-1, seqLen, this->head_dim});
        k.Reshape({-1, seqLen, this->head_dim});
        v.Reshape({-1, seqLen, this->head_dim});

        auto appendMtpCache = [&](Data &past, Data &cur) {
            if (past.dims.empty() && past.expansionDims.empty()) {
                Data typed(cur.dataType);
                past.CopyFrom(typed);
                past.dataDevice = DataDevice::CUDA;
                past.dataDeviceIds = {device};
            }
            int unitLen = 128;
            auto needsExpansion = [&]() {
                if (past.dims.empty()) {
                    return past.expansionDims.size() != cur.dims.size() ||
                           cur.dims[1] > past.expansionDims[1];
                }
                return past.expansionDims.size() != past.dims.size() ||
                       past.dims[1] + cur.dims[1] > past.expansionDims[1];
            };
            while (needsExpansion()) {
                std::vector<int> newDims;
                if (past.Count(0) == 0 || past.dims.size() == 0) {
                    newDims = {cur.dims[0], ((cur.dims[1] - 1) / unitLen + 1) * unitLen, cur.dims[2]};
                } else {
                    newDims = past.dims;
                    newDims[1] += ((cur.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                past.Expansion(newDims);
            }
            CatDirect(past, cur, 1);
        };
        appendMtpCache(cache.key, k);
        appendMtpCache(cache.value, v);
        cache.tokens += seqLen;
        if (cacheOnly) {
            return -1;
        }

        Attention(q, cache.key, cache.value, *GetEmptyData(), attenOutput,
                  q.dims[0] / cache.key.dims[0], 1.0f / std::sqrt((float)this->head_dim), 1);
        PermuteSelf(attenOutput, {1, 0, 2});
        attenOutput.Reshape({1, seqLen, -1});
        Sigmoid(gate, gate);
        if (gate.dataType != attenOutput.dataType) {
            ToDataType(gate, attenOutput.dataType);
        }
        MulTo(attenOutput, gate);
        Linear(attenOutput, weight[prefix + "self_attn.o_proj.weight"], *GetEmptyData(), projected);
        AddTo(hiddenStates, projected);

        Data mlpInput, gateupResult, swigluResult;
        RMSNorm(hiddenStates, weight[prefix + "post_attention_layernorm.weight"], rms_norm_eps, mlpInput);
        MLPBlock(&mlpInput, &weight[prefix + "mlp.gateup_proj.weight"],
                 &weight[prefix + "mlp.down_proj.weight"],
                 &gateupResult, &swigluResult, &hiddenStates);

        sampleRow = std::max(0, std::min(sampleRow, seqLen - 1));
        Data logits, sampleHidden;
        Data *sampleHiddenPtr = &hiddenStates;
        if (seqLen > 1) {
            Split(hiddenStates, 1, sampleRow, sampleRow + 1, sampleHidden);
            sampleHiddenPtr = &sampleHidden;
        }
        RMSNorm(*sampleHiddenPtr, weight["mtp.norm.weight"], rms_norm_eps, *sampleHiddenPtr);
        if (sampledHiddenStates != nullptr) {
            sampledHiddenStates->CopyFrom(*sampleHiddenPtr);
        }

        auto sampleGreedyFromCudaLogits = [&](Data &cudaLogits) {
            ToDataType(cudaLogits, DataType::FLOAT32);
            AssertInFastLLM(cudaLogits.dataDevice == DataDevice::CUDA &&
                            cudaLogits.cudaData != nullptr,
                            "Qwen3.5 MTP greedy draft logits must stay on CUDA.\n");
            Data &cudaOutput = Qwen35ThreadLocalCudaSamplingOutput();
            Qwen3CudaPrepareLocalOutput(cudaOutput, device);
            cudaOutput.dataType = DataType::INT32;
            cudaOutput.UpdateUnitSize();
            cudaOutput.Resize({1});
            cudaOutput.Allocate();
            FastllmCudaGreedySampling((float*)cudaLogits.cudaData,
                                      (int*)cudaOutput.cudaData,
                                      1, cudaLogits.dims.back());
            int draft = 0;
            FastllmCudaCopyFromDeviceToHost(&draft, cudaOutput.cudaData, sizeof(int));
            return draft;
        };

        Data &lmHead = weight["lm_head.weight"];
        if (!tensorParallelDraft) {
            Linear(*sampleHiddenPtr, lmHead, *GetEmptyData(), logits);
            return sampleGreedyFromCudaLogits(logits);
        }

        AssertInFastLLM(threadTpWeightsPrepared.load(std::memory_order_acquire) &&
                        threadTpPreparedDevices == draftDevices &&
                        lmHead.multiDeviceData &&
                        !threadTpLmHeadScheme.empty(),
                        "Qwen3.5 MTP TP draft requires prepared lm_head shards.\n");
        using namespace qwen3cuda;
        Data replicatedHidden;
        replicatedHidden.CopyFrom(*sampleHiddenPtr);
        PrepareMultiCudaReplicatedData(replicatedHidden, draftDevices, true);

        Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
        std::vector<Data> localLogits(draftDevices.size());
        std::vector<int> localBestIds(draftDevices.size(), -1);
        std::vector<float> localBestScores(draftDevices.size(), -1.0e30f);
        std::vector<int> localBestReady(draftDevices.size(), 0);
        std::vector<std::exception_ptr> errors(draftDevices.size());
        threadTpWorkerGroup.Run(draftDevices, [&](int r) {
            int localDevice = draftDevices[r];
            auto hiddenIt = replicatedHidden.multiDeviceDatas.find(localDevice);
            auto weightIt = lmHead.multiDeviceDatas.find(localDevice);
            auto biasIt = lmHeadBias.multiDeviceDatas.find(localDevice);
            AssertInFastLLM(hiddenIt != replicatedHidden.multiDeviceDatas.end() &&
                            hiddenIt->second != nullptr &&
                            weightIt != lmHead.multiDeviceDatas.end() &&
                            weightIt->second != nullptr &&
                            biasIt != lmHeadBias.multiDeviceDatas.end() &&
                            biasIt->second != nullptr,
                            "Qwen3.5 MTP TP draft missing local lm_head data.\n");
            FastllmCudaSetDevice(localDevice);
            Qwen3CudaDirectRunner cudaRunner(localDevice);
            Qwen3CudaLinear(cudaRunner, *hiddenIt->second, *weightIt->second,
                            *biasIt->second, localLogits[r]);
            Qwen3CudaToDataType(cudaRunner, localLogits[r], DataType::FLOAT32);
            Data localBestId(DataType::INT32), localBestScore(DataType::FLOAT32);
            Qwen3CudaPrepareLocalOutput(localBestId, localDevice);
            Qwen3CudaPrepareLocalOutput(localBestScore, localDevice);
            localBestId.Resize({1});
            localBestScore.Resize({1});
            localBestId.Allocate();
            localBestScore.Allocate();
            bool sampled = FastllmCudaGreedySamplingWithScores(
                (float*)localLogits[r].cudaData,
                (int*)localBestId.cudaData,
                (float*)localBestScore.cudaData,
                1, localLogits[r].dims.back());
            if (sampled) {
                FastllmCudaCopyFromDeviceToHost(&localBestIds[r],
                                                localBestId.cudaData,
                                                sizeof(int));
                FastllmCudaCopyFromDeviceToHost(&localBestScores[r],
                                                localBestScore.cudaData,
                                                sizeof(float));
                localBestReady[r] = 1;
            } else {
                ForceDeviceSync();
            }
        }, errors);
        for (auto &error : errors) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
        bool allBestReady = true;
        for (int ready : localBestReady) {
            allBestReady = allBestReady && (ready != 0);
        }
        if (allBestReady) {
            int bestToken = 0;
            float bestScore = -1.0e30f;
            for (int r = 0; r < (int)draftDevices.size(); r++) {
                int localDevice = draftDevices[r];
                int localId = localBestIds[r];
                int globalId = -1;
                int localOffset = 0;
                auto schemeIt = threadTpLmHeadScheme.find(localDevice);
                AssertInFastLLM(schemeIt != threadTpLmHeadScheme.end(),
                                "Qwen3.5 MTP TP draft missing lm_head split scheme.\n");
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    if (localId >= localOffset && localId < localOffset + len) {
                        globalId = range.first + (localId - localOffset);
                        break;
                    }
                    localOffset += len;
                }
                AssertInFastLLM(globalId >= 0,
                                "Qwen3.5 MTP TP draft local greedy id is out of range.\n");
                if (localBestScores[r] > bestScore ||
                    (localBestScores[r] == bestScore && globalId < bestToken)) {
                    bestScore = localBestScores[r];
                    bestToken = globalId;
                }
            }
            return bestToken;
        }
        Data &fullCudaLogits = Qwen35ThreadLocalCudaSamplingFullLogits();
        Qwen35GatherShardLogitsToRootCuda(device, draftDevices, threadTpLmHeadScheme,
                                          localLogits, 1, lmHead.dims[0],
                                          fullCudaLogits);
        return sampleGreedyFromCudaLogits(fullCudaLogits);
#endif
    }

    void Qwen3_5Model::AdjustPositionIdsWithDelta(const Data &positionIds,
                                                  const Data &mropePositionDelta,
                                                  Data &adjustedPositionIds) {
        adjustedPositionIds.CopyFrom(positionIds);
        adjustedPositionIds.ToDevice(DataDevice::CPU);
        if (adjustedPositionIds.dataType != DataType::FLOAT32) {
            // 注意: ToDataType 经 Executor 调度可能在 CUDA 上完成转换并释放 cpuData,
            // 这里的后续逻辑需要在 CPU 上读写, 必须再次 ToDevice(CPU).
            ToDataType(adjustedPositionIds, DataType::FLOAT32);
            adjustedPositionIds.ToDevice(DataDevice::CPU);
        }
        Data deltaCpu(mropePositionDelta);
        deltaCpu.ToDevice(DataDevice::CPU);
        if (deltaCpu.dataType != DataType::FLOAT32) {
            ToDataType(deltaCpu, DataType::FLOAT32);
            deltaCpu.ToDevice(DataDevice::CPU);
        }
        float delta = ((float*) deltaCpu.cpuData)[0];
        if (adjustedPositionIds.dims.size() == 2 && adjustedPositionIds.dims[0] == 3) {
            float *positionPtr = (float*) adjustedPositionIds.cpuData;
            for (int i = 0; i < adjustedPositionIds.Count(0); i++) {
                positionPtr[i] += delta;
            }
            return;
        }

        int seqLen = adjustedPositionIds.Count(0);
        float *positionPtr = (float*) adjustedPositionIds.cpuData;
        std::vector<float> expandedPositions(3 * seqLen);
        for (int row = 0; row < 3; row++) {
            for (int i = 0; i < seqLen; i++) {
                expandedPositions[row * seqLen + i] = positionPtr[i] + delta;
            }
        }
        adjustedPositionIds.CopyFrom(Data(DataType::FLOAT32, {3, seqLen}, expandedPositions));
    }

    std::vector <int> Qwen3_5Model::ForwardFromHiddenStates(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const Data &allPositionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits,
        Data &hiddenStates,
        bool all1) {
        Data qkv;
        Data q;
        Data k;
        Data v;
        Data attenInput;
        Data attenLastOutput;
        Data attenOutput;

        bool isSingleTokenDecode = batch == 1 && all1 &&
                                   !pastKeyValues.empty() &&
                                   pastKeyValues[0].first->dims.size() > 0;
        bool isFusedBatchDecode = batch > 1 && all1;
        PrepareMoeWeights();
        PrepareGdnWeights();

        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(this->weight[language_prefix + "norm.weight"]);
            AddMtpRmsNormOffset();
            initialized_add1 = true;
        }

        int seqlen = hiddenStates.dims[1];
        bool pagedAttentionInited = false;
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string postRmsName = language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.down_proj.weight";

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];
            Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
            bool residualAddedInBranch = false;

            if (weight.weight.find(language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
                std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

                Data qgate, gate, mergedQkv;
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], mergedQkv);
                    int qgateDim = num_attention_heads * this->head_dim * 2;
                    int kvDim = num_key_value_heads * this->head_dim;
                    Split(mergedQkv, -1, 0, qgateDim, qgate);
                    Split(mergedQkv, -1, qgateDim, qgateDim + kvDim, k);
                    Split(mergedQkv, -1, qgateDim + kvDim, qgateDim + kvDim * 2, v);
                } else {
                    Linear(attenInput, weight[qWeightName], weight[qBiasName], qgate);
                    Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                    Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
                }

                qgate.Reshape({bsz, seqlen, -1, this->head_dim * 2});
                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({bsz, seqlen, -1});

                k.Reshape({bsz, seqlen, -1, this->head_dim});
                v.Reshape({bsz, seqlen, -1, this->head_dim});

                RMSNorm(q, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);
                float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
                ApplyMultimodalRotary(q, allPositionIds, ropeScale);
                ApplyMultimodalRotary(k, allPositionIds, ropeScale);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                std::vector <int> qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);
                PreparePagedAttentionInputs(q, k, v, this->dataType);

                PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                    i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                    i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                AppendPagedCache(*pagedCacheKManager, pastKey, k);
                AppendPagedCache(*pagedCacheVManager, pastValue, v);
                AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(head_dim), 1, pagedAttentionInited);
                pagedAttentionInited = true;

                if (batch > 1 && all1) {
                    qkv.Reshape({seqlen, bsz, -1});
                    PermuteSelf(qkv, {1, 0, 2});
                } else {
                    PermuteSelf(qkv, {1, 0, 2});
                    qkv.Reshape({seqlen, bsz, -1});
                    PermuteSelf(qkv, {1, 0, 2});
                }

                Sigmoid(gate, gate);
                if (gate.dataType != qkv.dataType) {
                    ToDataType(gate, qkv.dataType);
                }
                MulTo(qkv, gate);

                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
            } else {
                std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
                std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
                std::string qkvzbaWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvzba.weight";
                std::string conv1dWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.weight";
                std::string conv1dBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.bias";
                std::string aLogName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.A_log";
                std::string dtBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.dt_bias";
                DataType linearCacheType = Qwen35LinearAttentionCacheDataType(this->dataType);
                Qwen35PrepareLinearAttentionCache(pastKey, linearCacheType);
                Qwen35PrepareLinearAttentionCache(pastValue, linearCacheType);

                int kd = num_k_heads * head_k_dim, vd = num_v_heads * head_v_dim;
                int mixedQkvzDim = this->weight[qkvzWeightName].dims[0];
                int baMergedDim = this->weight[baWeightName].dims[0];
                bool hasMergedGdnInLinear = this->weight.weight.find(qkvzbaWeightName) != this->weight.weight.end();
                if (hasMergedGdnInLinear && !isSingleTokenDecode &&
                    attenInput.dataDevice == DataDevice::CUDA &&
                    this->weight[qkvzbaWeightName].dataDevice != DataDevice::CUDA) {
                    if (!attenInput.dataDeviceIds.empty()) {
                        this->weight[qkvzbaWeightName].ToDevice(DataDevice::CUDA, attenInput.dataDeviceIds);
                    } else {
                        this->weight[qkvzbaWeightName].ToDevice(DataDevice::CUDA);
                    }
                }
                bool useMergedGdnInLinear = (isSingleTokenDecode || isFusedBatchDecode) && hasMergedGdnInLinear;

                Data gdn_in_merged, mixed_qkvz, ba_merged, qkvConvInput, z, b, a, g;
                if (useMergedGdnInLinear) {
                    Linear(attenInput, weight[qkvzbaWeightName], Data(), gdn_in_merged);
                    if (CanUseSingleRowLastDimView(gdn_in_merged)) {
                        MakeSingleRowLastDimView(gdn_in_merged, 0, mixedQkvzDim, mixed_qkvz);
                        MakeSingleRowLastDimView(gdn_in_merged, mixedQkvzDim, mixedQkvzDim + baMergedDim, ba_merged);
                    } else {
                        Split(gdn_in_merged, -1, 0, mixedQkvzDim, mixed_qkvz);
                        Split(gdn_in_merged, -1, mixedQkvzDim, mixedQkvzDim + baMergedDim, ba_merged);
                    }
                } else {
                    Linear(attenInput, weight[qkvzWeightName], Data(), mixed_qkvz);
                    Linear(attenInput, weight[baWeightName], Data(), ba_merged);
                }

                int qkvzDim = kd * 2 + vd;
                if (isSingleTokenDecode && CanUseSingleRowLastDimView(mixed_qkvz)) {
                    MakeSingleRowLastDimView(mixed_qkvz, 0, qkvzDim, qkvConvInput);
                    MakeSingleRowLastDimView(mixed_qkvz, qkvzDim, qkvzDim + vd, z);
                } else {
                    Split(mixed_qkvz, -1, 0, qkvzDim, qkvConvInput);
                    Split(mixed_qkvz, -1, qkvzDim, qkvzDim + vd, z);
                }

                if (isSingleTokenDecode && CanUseSingleRowLastDimView(ba_merged)) {
                    MakeSingleRowLastDimView(ba_merged, 0, num_v_heads, b);
                    MakeSingleRowLastDimView(ba_merged, num_v_heads, num_v_heads * 2, a);
                } else {
                    Split(ba_merged, -1, 0, num_v_heads, b);
                    Split(ba_merged, -1, num_v_heads, num_v_heads * 2, a);
                }

                if (isSingleTokenDecode) {
                    SwapSingleTokenSeqHeadByReshape(qkvConvInput);
                } else if (isFusedBatchDecode) {
                    qkvConvInput.Reshape({batch, qkvConvInput.dims.back(), 1});
                } else {
                    PermuteSelf(qkvConvInput, {0, 2, 1});
                }
                z.Reshape({bsz, seqlen, -1, head_v_dim});
                Data conv, convOutput;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    bool fusedDecodeConvSilu = false;
                    bool canTryFusedDecodeConvSilu = false;
                    #ifdef USE_CUDA
                    canTryFusedDecodeConvSilu =
                        pastKey.dataDevice == DataDevice::CUDA &&
                        pastKey.dataType == DataType::FLOAT16 &&
                        qkvConvInput.dataDevice == DataDevice::CUDA &&
                        qkvConvInput.dataType == DataType::FLOAT16 &&
                        weight[conv1dWeightName].dataDevice == DataDevice::CUDA &&
                        weight[conv1dWeightName].dataType == DataType::FLOAT32 &&
                        weight[conv1dBiasName].dataDevice == DataDevice::CUDA;
                    #endif
                    if (!canTryFusedDecodeConvSilu) {
                        ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                    }

                    #ifdef USE_CUDA
                    if (canTryFusedDecodeConvSilu) {
                        fusedDecodeConvSilu = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                            pastKey, qkvConvInput, weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
                    }
                    if (!fusedDecodeConvSilu)
                    #endif
                    {
                        if (canTryFusedDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                        }
                        Conv1DPerChannel(
                            pastKey, weight[conv1dWeightName], weight[conv1dBiasName],
                            pastKey.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0,
                            convOutput
                        );
                    }
                    if (!fusedDecodeConvSilu) {
                        Silu(convOutput, convOutput);
                    }
                } else if (isFusedBatchDecode) {
                    std::vector<Data*> linearConvCaches(batch);
                    for (int b = 0; b < batch; b++) {
                        linearConvCaches[b] = pastKeyValues[b * block_cnt + i].first;
                    }
                    bool directBatchDecodeConvSilu = false;
#ifdef USE_CUDA
                    directBatchDecodeConvSilu =
                        FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(
                            linearConvCaches, qkvConvInput,
                            weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
#endif
                    if (!directBatchDecodeConvSilu) {
                        Data batchConvCache;
                        CatBatchFirstDim(linearConvCaches, batchConvCache);
                        bool fusedBatchDecodeConvSilu = false;
#ifdef USE_CUDA
                        fusedBatchDecodeConvSilu =
                            FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                                batchConvCache, qkvConvInput,
                                weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
#endif
                        if (!fusedBatchDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(batchConvCache, qkvConvInput);
                            Conv1DPerChannel(
                                batchConvCache, weight[conv1dWeightName], weight[conv1dBiasName],
                                batchConvCache.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0,
                                convOutput
                            );
                            Silu(convOutput, convOutput);
                        }
                        SplitBatchFirstDim(batchConvCache, linearConvCaches);
                    }
                    for (int b = 0; b < batch; b++) {
                        Qwen35PrepareLinearAttentionCache(*linearConvCaches[b], linearCacheType);
                    }
                } else {
                    if (qkvConvInput.dims.back() >= 4) {
                        Split(qkvConvInput, -1, qkvConvInput.dims.back() - 4, qkvConvInput.dims.back(), pastKey);
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        Data temp;
                        Mul(qkvConvInput, 1.0f, temp);
                        Repeat(temp, -1, 4, qkvConvInput);
                    }

                    Conv1DPerChannel(
                        qkvConvInput, weight[conv1dWeightName], weight[conv1dBiasName],
                        qkvConvInput.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 3,
                        conv
                    );
                    Split(conv, -1, 0, seqlen, convOutput);
                    Silu(convOutput, convOutput);
                }

                if (isSingleTokenDecode) {
                    SwapSingleTokenSeqHeadByReshape(convOutput);
                } else {
                    PermuteSelf(convOutput, {0, 2, 1});
                }

                // q / k / v are reused later as MLP scratch buffers, so they must not alias
                // convOutput's lifetime-limited storage in single-token decode.
                Split(convOutput, -1, 0, kd, q);
                Split(convOutput, -1, kd, kd + kd, k);
                Split(convOutput, -1, kd + kd, kd + kd + vd, v);

                q.Reshape({q.dims[0], q.dims[1], -1, head_k_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_k_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_v_dim});

                #ifdef USE_CUDA
                if (b.dataDevice == DataDevice::CUDA &&
                    a.dataDevice == DataDevice::CUDA &&
                    (b.dataType == DataType::FLOAT32 || b.dataType == DataType::FLOAT16) &&
                    a.dataType == b.dataType &&
                    weight[aLogName].dataDevice == DataDevice::CUDA &&
                    weight[dtBiasName].dataDevice == DataDevice::CUDA) {
                    SigmoidMambaSoftplus(b, a, weight[aLogName], weight[dtBiasName], g);
                } else
                #endif
                {
                    Sigmoid(b, b);
                    MambaSoftplus(a, weight[aLogName], weight[dtBiasName], g);
                }

                Data &last_recurrent_state = pastValue;
                Data core_attn_out, core_attn_out_temp;
                #ifdef USE_CUDA
                if (!isSingleTokenDecode) {
                    Qwen35EnsureCudaLinearAttnStateKVLayout(last_recurrent_state);
                }
                #endif
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    bool fusedSingleDecode = false;
#ifdef USE_CUDA
                    fusedSingleDecode = Qwen35TryCudaLinearAttnSingleDecodeNormRecurrent(
                        q, k, v, g, b, inv_scale_data, rms_norm_eps, last_recurrent_state, core_attn_out
                    );
#endif
                    if (!fusedSingleDecode) {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);

                        SwapSingleTokenSeqHeadByReshape(q);
                        SwapSingleTokenSeqHeadByReshape(k);
                        SwapSingleTokenSeqHeadByReshape(v);
                        SwapSingleTokenSeqHeadByReshape(b);
                        SwapSingleTokenSeqHeadByReshape(g);

                        float scale = 1.0f / pow(q.dims.back(), 0.5);
                        float recurrentQScale = 1.0f;
                        if (q.dataDevice == DataDevice::CUDA) {
                            recurrentQScale = scale;
                        } else {
                            Mul(q, scale, q);
                        }

                        RecurrentGatedDeltaRule(q, k, v, g, b, last_recurrent_state, core_attn_out, recurrentQScale);
                        SwapSingleTokenSeqHeadByReshape(core_attn_out);
                    }
                } else {
                    if (num_v_heads / num_k_heads > 1) {
                        Data qrepeat, krepeat;
                        Mul(q, 1.0f, qrepeat);
                        Mul(k, 1.0f, krepeat);

                        qrepeat.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                        krepeat.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});

                        Repeat(qrepeat, 3, num_v_heads / num_k_heads, q);
                        Repeat(krepeat, 3, num_v_heads / num_k_heads, k);

                        q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                        k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                    }

                    RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                    RMSNorm(k, inv_scale_data, rms_norm_eps, k);

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});

                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_k_head_dim = k.dims[3];
                    int chunk_size = 64;
                    int v_head_dim_local = v.dims.back();
                    int seq = k.dims[2];
                    int pad_size = (chunk_size - seq % chunk_size) % chunk_size;

                    Data qq, kk_pad, vv_pad, bb_pad, gg_pad, decayMask;
                    Data *pkk, *pvv, *pbb, *pgg;
                    if (pad_size > 0) {
                        Data qtemp;
                        Pad(q, 2, pad_size, qtemp);
                        Pad(k, 2, pad_size, kk_pad);
                        Pad(v, 2, pad_size, vv_pad);
                        Pad(b, 2, pad_size, bb_pad);
                        Pad(g, 2, pad_size, gg_pad);
                        float scale = 1.0f / pow(qtemp.dims.back(), 0.5);
                        Mul(qtemp, scale, qq);
                        pkk = &kk_pad;
                        pvv = &vv_pad;
                        pbb = &bb_pad;
                        pgg = &gg_pad;
                    } else {
                        float scale = 1.0f / pow(q.dims.back(), 0.5);
                        Mul(q, scale, qq);
                        pkk = &k;
                        pvv = &v;
                        pbb = &b;
                        pgg = &g;
                    }

                    int tot_heads = seq + pad_size;

                    pbb->Resize({(*pbb).dims[0], (*pbb).dims[1], (*pbb).dims[2], 1});
                    Data k_beta, v_beta;
                    Mul(*pkk, 1.0f, k_beta);
                    Mul(*pvv, 1.0f, v_beta);
                    MulTo(k_beta, *pbb);
                    MulTo(v_beta, *pbb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunk_size, qq.dims.back()});
                    pkk->Reshape({(*pkk).dims[0], (*pkk).dims[1], -1, chunk_size, (*pkk).dims.back()});
                    k_beta.Reshape({k_beta.dims[0], k_beta.dims[1], -1, chunk_size, k_beta.dims.back()});
                    v_beta.Reshape({v_beta.dims[0], v_beta.dims[1], -1, chunk_size, v_beta.dims.back()});
                    pgg->Reshape({(*pgg).dims[0], (*pgg).dims[1], -1, chunk_size});

                    CumSumLastDim(*pgg);
                    MakeDecayMask(*pgg, decayMask);

                    Data attn, at;
                    MatMulTransB(k_beta, *pkk, at);
                    Mul(at, -1.0f, attn);
                    MulTo(attn, decayMask);
                    CausalMask(attn, 0, 0.0f);
                    TransferAttn(attn);
                    MatMul(attn, v_beta, vv_pad);
                    Data k_cumdecay, g_exp;
                    Exp(*pgg, g_exp);

                    MulTo(k_beta, g_exp);
                    MatMul(attn, k_beta, k_cumdecay);

                    MatMulTransB(qq, *pkk, attn);
                    MulTo(attn, decayMask);
                    CausalMask(attn, 1, 0.0f);

                    if (last_recurrent_state.dims.size() == 0) {
                        #ifdef USE_CUDA
                        if (qq.dataDevice == DataDevice::CUDA) {
                            last_recurrent_state.dataDevice = qq.dataDevice;
                            last_recurrent_state.dataDeviceIds = qq.dataDeviceIds;
                        }
                        #endif
                        last_recurrent_state.Resize({key_batch_size, key_sequence_length, key_k_head_dim, v_head_dim_local});
                        last_recurrent_state.isLinearAttentionTransposed = false;
                        last_recurrent_state.Allocate(0.0f);
                    }

                    auto runChunkPrefillReference = [&](Data &state, Data &out) {
                        auto makeChunk4D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3], src.dims[4]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3], src.strides[4]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };
                        auto makeChunk3D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };

                        for (int ci = 0; ci < tot_heads / chunk_size; ci++) {
                            Data q_i, k_i, v_i, attn_i, k_cumdecay_i;
                            makeChunk4D(qq, ci, q_i);
                            makeChunk4D(*pkk, ci, k_i);
                            makeChunk4D(vv_pad, ci, v_i);
                            makeChunk4D(attn, ci, attn_i);
                            makeChunk4D(k_cumdecay, ci, k_cumdecay_i);

                            Data v_prime, v_new;
                            MatMul(k_cumdecay_i, state, v_prime);
                            Mul(v_prime, -1.0f, v_new);
                            AddTo(v_new, v_i);

                            Data attn_inter, g_i, g_i_exp;
                            Split(*pgg, 0, ci, ci + 1, g_i);
                            g_i.Resize({g_i.dims[1], g_i.dims[2], g_i.dims[3]});
                            Split(g_exp, 0, ci, ci + 1, g_i_exp);
                            g_i_exp.Resize({g_i_exp.dims[1], g_i_exp.dims[2], g_i_exp.dims[3], 1});
                            MulTo(q_i, g_i_exp);

                            MatMul(q_i, state, attn_inter);
                            Data atv;
                            MatMul(attn_i, v_new, atv);
                            AddTo(atv, attn_inter);
                            atv.Resize({atv.dims[0], atv.dims[1], 1, atv.dims[2], atv.dims[3]});
                            if (ci == 0) {
                                Mul(atv, 1.0f, out);
                            } else {
                                Mul(out, 1.0f, core_attn_out_temp);
                                Cat(core_attn_out_temp, atv, 3, out);
                            }

                            Data g_i_last, g_i_last_repeat, g_i_delta, g_i_scale;
                            Split(g_i, 2, g_i.dims[2] - 1, g_i.dims[2], g_i_last);
                            Repeat(g_i_last, 2, g_i.dims[2], g_i_last_repeat);
                            Mul(g_i, -1.0f, g_i_delta);
                            AddTo(g_i_last_repeat, g_i_delta);
                            Exp(g_i_last_repeat, g_i_scale);
                            g_i_scale.Resize({g_i_scale.dims[0], g_i_scale.dims[1], g_i_scale.dims[2], 1});
                            MulTo(k_i, g_i_scale);

                            Data k_i_v_new;
                            PermuteSelf(k_i, {0, 1, 3, 2});
                            MatMul(k_i, v_new, k_i_v_new);

                            Data g_i_exp_last;
                            Split(g_i_exp, 2, g_i_exp.dims[2] - 1, g_i_exp.dims[2], g_i_exp_last);
                            MulTo(state, g_i_exp_last);
                            AddTo(state, k_i_v_new);
                        }
                    };

                    bool useFusedChunkPrefill = false;
                    #ifdef USE_CUDA
                    if (qq.dataDevice == DataDevice::CUDA &&
                        pkk->dataDevice == DataDevice::CUDA &&
                        vv_pad.dataDevice == DataDevice::CUDA &&
                        pgg->dataDevice == DataDevice::CUDA &&
                        attn.dataDevice == DataDevice::CUDA &&
                        k_cumdecay.dataDevice == DataDevice::CUDA &&
                        last_recurrent_state.dataDevice == DataDevice::CUDA) {
                        useFusedChunkPrefill = GetFastllmEnv().useFusedGdnPrefill;
                    }
                    #endif

                    if (useFusedChunkPrefill) {
                        #ifdef USE_CUDA
                        ChunkGatedDeltaRulePrefill(
                            qq, *pkk, vv_pad, *pgg, attn, k_cumdecay,
                            last_recurrent_state, core_attn_out
                        );
                        #endif
                    } else {
                        PermuteSelf(qq, {2, 0, 1, 3, 4});
                        PermuteSelf(*pkk, {2, 0, 1, 3, 4});
                        PermuteSelf(vv_pad, {2, 0, 1, 3, 4});
                        PermuteSelf(attn, {2, 0, 1, 3, 4});
                        PermuteSelf(k_cumdecay, {2, 0, 1, 3, 4});
                        PermuteSelf(g_exp, {2, 0, 1, 3});
                        PermuteSelf(*pgg, {2, 0, 1, 3});
                        runChunkPrefillReference(last_recurrent_state, core_attn_out);
                    }

                    core_attn_out.Reshape({core_attn_out.dims[0], core_attn_out.dims[1], -1, core_attn_out.dims.back()});
                    if (pad_size > 0) {
                        Split(core_attn_out, 2, 0, seq, core_attn_out_temp);
                        PermuteSelf(core_attn_out_temp, {0, 2, 1, 3});
                        Mul(core_attn_out_temp, 1.0f, core_attn_out);
                    } else {
                        PermuteSelf(core_attn_out, {0, 2, 1, 3});
                    }
                }

                std::vector <int> zShape = z.dims;
                std::string outNormWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.norm.weight";
                core_attn_out.Reshape({-1, core_attn_out.dims.back()});
                z.Reshape({-1, z.dims.back()});

                #ifdef USE_CUDA
                if (isSingleTokenDecode &&
                    core_attn_out.dataDevice == DataDevice::CUDA &&
                    core_attn_out.dataType == DataType::FLOAT16 &&
                    z.dataDevice == DataDevice::CUDA &&
                    z.dataType == DataType::FLOAT16 &&
                    this->weight[outNormWeightName].dataDevice == DataDevice::CUDA &&
                    this->weight[outNormWeightName].dataType == DataType::FLOAT32 &&
                    core_attn_out.dims == z.dims &&
                    FastllmCudaRMSNormSiluMulFloat16(
                        core_attn_out, this->weight[outNormWeightName], z, core_attn_out, rms_norm_eps)) {
                } else
                #endif
                {
                    RMSNorm(core_attn_out, this->weight[outNormWeightName], rms_norm_eps, core_attn_out);
                    Silu(z, z);
                    MulTo(core_attn_out, z);
                }

                core_attn_out.Reshape({zShape[0], zShape[1], -1});
                if (isSingleTokenDecode &&
                    core_attn_out.dataDevice == DataDevice::CUDA &&
                    core_attn_out.dataType == DataType::FLOAT16 &&
                    hiddenStates.dataDevice == DataDevice::CUDA &&
                    hiddenStates.dataType == DataType::FLOAT16) {
                    int n = core_attn_out.Count(0) / core_attn_out.dims.back();
                    int m = core_attn_out.dims.back();
                    int kdim = hiddenStates.dims.back();
                    #ifdef USE_CUDA
                    if (FastllmCudaHalfMatMulFloat16AddToNoBias(
                            core_attn_out,
                            this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                            hiddenStates, n, m, kdim)) {
                        residualAddedInBranch = true;
                    } else
                    #endif
                    {
                        Linear(core_attn_out,
                               this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                               Data(), attenInput);
                    }
                } else {
                    Linear(core_attn_out,
                           this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                           Data(), attenInput);
                }
            }

            if (!residualAddedInBranch) {
                AddTo(hiddenStates, attenInput);
            }
            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            if (weight.weight.find(swigluWeightName) != weight.weight.end() &&
                weight.weight.find(downWeightName) != weight.weight.end()) {
                MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
                continue;
            }

            std::string gateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.weight";
            if (weight.weight.find(gateWeightName) == weight.weight.end()) {
                ErrorInFastLLM("Qwen3.5 layer " + std::to_string(i) + " has neither dense MLP nor MoE weights.");
            }

            Data w1, w2, w3;
            Data routerLogits;
            Data sharedGate;
            Data moeFinal, moeFinal2;
            Data tempInput, tempOutput;
            Data attenPart, moePart;

            std::string gateBiasName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";
            std::string firstExpertGateupName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight";
            std::string sharedGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
            std::string sharedGateProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
            std::string sharedUpProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
            std::string sharedDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.down_proj.weight";
            std::string sharedExpertGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert_gate.weight";
            Data *gateBiasData = weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr;

            int flatBatch = attenInput.dims[0];
            int flatLen = attenInput.dims[1];
            attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});

            Linear(attenInput, weight[gateWeightName], Data(), routerLogits);
            ToDataType(routerLogits, DataType::FLOAT32);
            if (gateBiasData != nullptr) {
                ToDataType(*gateBiasData, DataType::FLOAT32);
            }
            Softmax(routerLogits, routerLogits, -1);

            if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                if (weight.weight.find(sharedGateupWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateupWeightName], Data(), w3);
                    Swiglu(w3, w1);
                } else if (weight.weight.find(sharedGateProjWeightName) != weight.weight.end() &&
                           weight.weight.find(sharedUpProjWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateProjWeightName], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[sharedUpProjWeightName], Data(), w3);
                    MulTo(w1, w3);
                }
                if (w1.dims.size() != 0) {
                    Linear(w1, weight[sharedDownWeightName], Data(), moeFinal2);
                    if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[sharedExpertGateWeightName], Data(), sharedGate);
                        Sigmoid(sharedGate, sharedGate);
                        MulTo(moeFinal2, sharedGate);
                    }
                }
            }

            bool useMergeMoe = weight.weight.find(firstExpertGateupName) != weight.weight.end() &&
                               !weights[i].empty() && CanRunMergeMOE(attenInput, biass[i]);
            if (useMergeMoe) {
                Data expertIndex, expertScore;
                SelectExpert(routerLogits, expertIndex, expertScore, this->num_experts_per_tok, this->norm_topk_prob,
                             this->routed_scaling_factor, gateBiasData);
                this->ApplyMoeDeviceMapForLayer(i);
                MergeMOE(
                    attenInput, expertIndex, expertScore,
                    weights[i], biass[i],
                    w1, w2, w3, tempInput, tempOutput,
                    1.0f,
                    moeFinal, i
                );
            } else {
                routerLogits.ToDevice(DataDevice::CPU);
                float *cpuRouterLogits = (float*) routerLogits.cpuData;
                float *cpuBias = nullptr;
                if (gateBiasData != nullptr) {
                    gateBiasData->ToDevice(DataDevice::CPU);
                    cpuBias = (float*) gateBiasData->cpuData;
                }
                int expertCount = routerLogits.dims.back();

                moeFinal.dataType = hiddenStates.dataType;
                moeFinal.dataDevice = attenInput.dataDevice;
                moeFinal.dataDeviceIds = attenInput.dataDeviceIds;
                moeFinal.UpdateUnitSize();
                moeFinal.Resize({0, attenInput.dims[1]});
                moeFinal.Expansion(attenInput.dims);
                for (int bidx = 0; bidx < flatBatch * flatLen; bidx++) {
                    float *cur = cpuRouterLogits + bidx * expertCount;
                    std::vector <std::pair <float, int> > candidates;
                    candidates.reserve(expertCount);
                    for (int j = 0; j < expertCount; j++) {
                        float score = cur[j];
                        if (cpuBias != nullptr) {
                            score += cpuBias[j];
                        }
                        candidates.push_back(std::make_pair(-score, j));
                    }
                    std::sort(candidates.begin(), candidates.end());

                    Data *currentData = &attenInput;
                    if (flatBatch * flatLen != 1) {
                        Split(attenInput, 0, bidx, bidx + 1, attenPart);
                        currentData = &attenPart;
                    }
                    moePart.dataType = hiddenStates.dataType;
                    moePart.dataDevice = currentData->dataDevice;
                    moePart.dataDeviceIds = currentData->dataDeviceIds;
                    moePart.UpdateUnitSize();
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);

                    float sum = 0.0f;
                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        sum += cur[candidates[j].second];
                    }
                    if (!this->norm_topk_prob) {
                        sum = 1.0f;
                    }

                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        int idx = candidates[j].second;
                        float value = cur[idx];
                        if (sum != 0.0f) {
                            value /= sum;
                        }
                        value *= this->routed_scaling_factor;

                        std::string expertGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight";
                        std::string expertDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(expertGateupWeightName) != weight.weight.end() &&
                                        weight.weight.find(expertDownWeightName) != weight.weight.end(),
                                        "Qwen3.5 MoE expert weights are incomplete.");
                        Linear(*currentData, weight[expertGateupWeightName], Data(), w3);
                        Swiglu(w3, w1);
                        Linear(w1, weight[expertDownWeightName], Data(), w2);
                        if (w2.dataType != moePart.dataType) {
                            ToDataType(w2, moePart.dataType);
                        }
                        AddTo(moePart, w2, value);
                    }
                    if (moePart.dataType != moeFinal.dataType) {
                        ToDataType(moePart, moeFinal.dataType);
                    }
                    CatDirect(moeFinal, moePart, 0);
                }
                moeFinal.expansionDims.clear();
            }

            moeFinal.Reshape(hiddenStates.dims);
            Data tempMoeFinal;
            tempMoeFinal.CopyFrom(moeFinal);
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            if (tempMoeFinal.dataType != hiddenStates.dataType) {
                ToDataType(tempMoeFinal, hiddenStates.dataType);
            }
            AddTo(hiddenStates, tempMoeFinal);
            if (moeFinal2.dims.size() != 0) {
                moeFinal2.Reshape(hiddenStates.dims);
                if (moeFinal2.dataType != hiddenStates.dataType) {
                    ToDataType(moeFinal2, hiddenStates.dataType);
                }
                AddTo(hiddenStates, moeFinal2);
            }
        }

        std::string lmHeadWeightName = "lm_head.weight";
        if (this->weight.weight.find(lmHeadWeightName) == this->weight.weight.end()) {
            lmHeadWeightName = language_prefix + "embed_tokens.weight";
        }
        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight[language_prefix + "norm.weight"], &weight[lmHeadWeightName],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    std::vector <int> Qwen3_5Model::ForwardMultimodal(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                                      const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                                                      const std::map <std::string, std::vector <Data*> > &multimodalInput,
                                                      const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                                                      std::vector <std::vector <float>*> *retLogits) {
        std::vector <int> ret;
        std::vector <float> *logits = nullptr;
        if (retLogits != nullptr && !retLogits->empty()) {
            logits = (*retLogits)[0];
        }

        if (pastKeyValues.size() > 0 && pastKeyValues[0].second.dims.size() > 0) {
            Data adjustedPositionIds;
            auto deltaIt = multimodalInput.find("mrope_position_delta");
            if (deltaIt != multimodalInput.end() && !deltaIt->second.empty()) {
                AdjustPositionIdsWithDelta(positionIds, *deltaIt->second[0], adjustedPositionIds);
            } else {
                adjustedPositionIds.CopyFrom(positionIds);
            }
            ret.push_back(Forward(inputIds, attentionMask, adjustedPositionIds, pastKeyValues, generationConfig, lastTokens, logits));
            return ret;
        }

        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "Qwen3.5 multimodal currently supports a single prompt batch only.");

        auto &mutableMultimodalInput =
            const_cast<std::map <std::string, std::vector <Data*> >&>(multimodalInput);
        auto mmTypeIt = multimodalInput.find("mm_token_type_ids");
        auto mropeIt = multimodalInput.find("mrope_position_ids");
        const Data *imageEmbeds = nullptr;
        const Data *videoEmbeds = nullptr;

        auto rawImageIt = multimodalInput.find("image_frames");
        auto rawVideoIt = multimodalInput.find("video_frames");
        auto imageGridIt = multimodalInput.find("image_grid_thw");
        auto videoGridIt = multimodalInput.find("video_grid_thw");
        bool hasRawMedia =
            (rawImageIt != multimodalInput.end() && !rawImageIt->second.empty()) ||
            (rawVideoIt != multimodalInput.end() && !rawVideoIt->second.empty());

        Data imageFeatures, videoFeatures;
        std::vector<std::vector<int>> imageGridThwList, videoGridThwList;
        if (hasRawMedia) {
            EncodeVisualItems(
                rawImageIt != multimodalInput.end() ? rawImageIt->second : std::vector<Data*>(),
                (imageGridIt != multimodalInput.end() && !imageGridIt->second.empty()) ? imageGridIt->second[0] : nullptr,
                false,
                imageFeatures,
                imageGridThwList
            );
            EncodeVisualItems(
                rawVideoIt != multimodalInput.end() ? rawVideoIt->second : std::vector<Data*>(),
                (videoGridIt != multimodalInput.end() && !videoGridIt->second.empty()) ? videoGridIt->second[0] : nullptr,
                true,
                videoFeatures,
                videoGridThwList
            );

            Data computedMmTokenTypeIds, computedMropePositionIds, computedMropePositionDelta;
            BuildMultimodalPositionData(
                inputIds,
                imageGridThwList,
                videoGridThwList,
                computedMmTokenTypeIds,
                computedMropePositionIds,
                computedMropePositionDelta
            );

            mutableMultimodalInput["mm_token_type_ids"].clear();
            mutableMultimodalInput["mrope_position_ids"].clear();
            mutableMultimodalInput["mrope_position_delta"].clear();
            mutableMultimodalInput["mm_token_type_ids"].push_back(new Data(computedMmTokenTypeIds));
            mutableMultimodalInput["mrope_position_ids"].push_back(new Data(computedMropePositionIds));
            mutableMultimodalInput["mrope_position_delta"].push_back(new Data(computedMropePositionDelta));

            mmTypeIt = mutableMultimodalInput.find("mm_token_type_ids");
            mropeIt = mutableMultimodalInput.find("mrope_position_ids");
            imageEmbeds = imageFeatures.dims.empty() ? nullptr : &imageFeatures;
            videoEmbeds = videoFeatures.dims.empty() ? nullptr : &videoFeatures;
        } else {
            AssertInFastLLM(mmTypeIt != multimodalInput.end() && !mmTypeIt->second.empty(),
                            "Qwen3.5 multimodal requires mm_token_type_ids.");
            AssertInFastLLM(mropeIt != multimodalInput.end() && !mropeIt->second.empty(),
                            "Qwen3.5 multimodal requires mrope_position_ids.");

            auto imageIt = multimodalInput.find("image_embeds");
            if (imageIt != multimodalInput.end() && !imageIt->second.empty()) {
                imageEmbeds = imageIt->second[0];
            }
            auto videoIt = multimodalInput.find("video_embeds");
            if (videoIt != multimodalInput.end() && !videoIt->second.empty()) {
                videoEmbeds = videoIt->second[0];
            }
        }

        AssertInFastLLM(mmTypeIt != multimodalInput.end() && !mmTypeIt->second.empty(),
                        "Qwen3.5 multimodal requires mm_token_type_ids.");
        AssertInFastLLM(mropeIt != multimodalInput.end() && !mropeIt->second.empty(),
                        "Qwen3.5 multimodal requires mrope_position_ids.");

        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            SetCudaEmbedding(true);
        }

        Data hiddenStates;
        Data embeddingResult;
        Embedding(inputIds, this->weight[language_prefix + "embed_tokens.weight"], embeddingResult);
        ToDataType(embeddingResult, hiddenStates, this->dataType);
        MergeMultimodalFeaturesIntoText(*mmTypeIt->second[0], imageEmbeds, videoEmbeds, hiddenStates);

        Data &embedWeight = this->weight[language_prefix + "embed_tokens.weight"];
        if (embedWeight.dataDevice != DataDevice::CPU) {
            if (!embedWeight.dataDeviceIds.empty()) {
                hiddenStates.ToDevice(embedWeight.dataDevice, embedWeight.dataDeviceIds);
            } else {
                hiddenStates.ToDevice(embedWeight.dataDevice);
            }
        }
        if (hiddenStates.dataType != this->dataType) {
            ToDataType(hiddenStates, this->dataType);
        }

        Data mropePositionIds;
        mropePositionIds.CopyFrom(*mropeIt->second[0]);
        mropePositionIds.ToDevice(DataDevice::CPU);
        if (mropePositionIds.dataType != DataType::FLOAT32) {
            // ToDataType 经 Executor 调度可能优先在 CUDA 上完成转换并释放 cpuData,
            // ForwardFromHiddenStates 期望 allPositionIds 在 CPU 上, 这里需要再次 ToDevice(CPU).
            ToDataType(mropePositionIds, DataType::FLOAT32);
            mropePositionIds.ToDevice(DataDevice::CPU);
        }

        Data attentionMaskCopy(attentionMask);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <int> seqLens = {inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        return ForwardFromHiddenStates(1, inputIds, attentionMasks, mropePositionIds, seqLens,
                                       pagedPastKeyValues, generationConfigs, lastTokens,
                                       retLogits, hiddenStates, inputIds.dims[1] == 1);
    }

    std::vector <int> Qwen3_5Model::ForwardV2(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
#ifdef USE_CUDA
        if (IsThreadTensorParallelEnabled()) {
            return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
#endif
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }

        auto runSplitBatchForward = [&]() -> std::vector<int> {
            std::vector<int> ret;
            ret.reserve(batch);

            int inputOffset = 0;
            for (int b = 0; b < batch; b++) {
                Data curInputIds;
                Split(inputIds, 1, inputOffset, inputOffset + seqLens[b], curInputIds);
                inputOffset += seqLens[b];

                std::vector<Data*> curAttentionMask = {
                    b < (int) attentionMask.size() ? attentionMask[b] : nullptr
                };
                std::vector<Data*> curPositionIds = {
                    b < (int) positionIds.size() ? positionIds[b] : nullptr
                };
                std::vector<int> curSeqLens = {seqLens[b]};
                std::vector<GenerationConfig> curGenerationConfigs = {generationConfigs[b]};

                LastTokensManager curLastTokens;
                if (b < (int) lastTokens.units.size()) {
                    curLastTokens.units.push_back(lastTokens.units[b]);
                } else {
                    int lastN = generationConfigs[b].last_n <= 0 ? max_positions : generationConfigs[b].last_n;
                    curLastTokens = LastTokensManager(1, lastN);
                }

                std::vector<std::pair<Data*, Data*> > curPastKeyValues;
                curPastKeyValues.reserve(block_cnt);
                int pastOffset = b * block_cnt;
                for (int i = 0; i < block_cnt; i++) {
                    curPastKeyValues.push_back(pastKeyValues[pastOffset + i]);
                }

                std::vector<std::vector<float>*> curLogits;
                std::vector<std::vector<float>*> *curLogitsPtr = nullptr;
                if (retLogits != nullptr) {
                    curLogits.push_back(b < (int) retLogits->size() ? (*retLogits)[b] : nullptr);
                    curLogitsPtr = &curLogits;
                }

                std::vector<int> curRet = ForwardV2(
                    1, curInputIds, curAttentionMask, curPositionIds, curSeqLens,
                    curPastKeyValues, curGenerationConfigs, curLastTokens, curLogitsPtr
                );
                ret.push_back(curRet[0]);
            }
            return ret;
        };

        auto canRunFusedBatchDecode = [&]() -> bool {
            if (batch <= 1 || !all1 || (int) pastKeyValues.size() < batch * block_cnt) {
                return false;
            }
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                    Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                    if (pastKey == nullptr || pastValue == nullptr) {
                        return false;
                    }
                    bool isGatedAttentionLayer =
                        weight.weight.find(language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end();
                    if (isGatedAttentionLayer) {
                        if (!pastKey->isPagedKVCache || !pastValue->isPagedKVCache ||
                            pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                            pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                            return false;
                        }
                    } else if (pastKey->dims.empty() || pastValue->dims.empty()) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (batch > 1 && !canRunFusedBatchDecode()) {
            return runSplitBatchForward();
        }

        int seqLen = inputIds.dims[1];

        Data qkv;
        // Data &qkv = this->forwardDataManager.GetData("qkv");
        Data q;
        // Data &q = this->forwardDataManager.GetData("q");
        Data k;
        // Data &k = this->forwardDataManager.GetData("k");
        Data v;
        // Data &v = this->forwardDataManager.GetData("v");
        Data embeddingResult;
        // Data &embeddingResult = this->forwardDataManager.GetData("embeddingResult");
        Data hiddenStates;
        // Data &hiddenStates = this->forwardDataManager.GetData("hiddenStates");
        Data attenInput;
        // Data &attenInput = this->forwardDataManager.GetData("attenInput");
        Data attenLastOutput;
        // Data &attenLastOutput = this->forwardDataManager.GetData("attenLastOutput");
        std::vector <Data*> pointersK;
        pointersK.resize(batch);


        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        // Data &allPositionIds = this->forwardDataManager.GetData("allPositionIds");
        Data qSizes;
        // Data &qSizes = this->forwardDataManager.GetData("qSizes");
        Data pageSizes;
        // Data &pageSizes = this->forwardDataManager.GetData("pageSizes");
        Data pageIndexs;
        // Data &pageIndexs = this->forwardDataManager.GetData("pageIndexs");
        Data lastPageLens;
        // Data &lastPageLens = this->forwardDataManager.GetData("lastPageLens");
        Data insertIndexs;
        // Data &insertIndexs = this->forwardDataManager.GetData("insertIndexs");
        Data insertPositions;
        // Data &insertPositions = this->forwardDataManager.GetData("insertPositions");
        Data attenOutput;
        // Data &attenOutput = this->forwardDataManager.GetData("attenOutput");
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;
        auto makeCacheDesc = [](const Data &src, DataType targetType) {
            Data desc(targetType);
            desc.dims = src.dims;
            desc.strides = src.strides;
            desc.dataDevice = src.dataDevice;
            desc.dataDeviceIds = src.dataDeviceIds;
            desc.UpdateUnitSize();
            return desc;
        };
        auto resolvePagedAttentionQType = [&](DataType cacheType, DataType queryType) -> DataType {
            if (cacheType == DataType::FLOAT16 || cacheType == DataType::BFLOAT16) {
                return cacheType;
            }
            if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
                return queryType;
            }
            if (this->dataType == DataType::BFLOAT16) {
                return DataType::BFLOAT16;
            }
            return DataType::FLOAT16;
        };
        auto preparePagedAttentionQ = [&](Data &src, DataType cacheType, Data &casted) -> Data& {
            DataType targetType = resolvePagedAttentionQType(cacheType, src.dataType);
            if (src.dataType == targetType) {
                return src;
            }
            ToDataType(src, casted, targetType);
            return casted;
        };

        allPositionIds.CopyFrom(BuildFlattenedPositionIds(positionIds, seqLens, all1));

        bool isSingleTokenDecode = batch == 1 && all1 &&
                                   !pastKeyValues.empty() &&
                                   pastKeyValues[0].first->dims.size() > 0;
        bool isFusedBatchDecode = batch > 1 && all1;

        PrepareMoeWeights();
        PrepareGdnWeights();

        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(this->weight[language_prefix + "norm.weight"]);
            AddMtpRmsNormOffset();
            initialized_add1 = true;
        }

        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            // 如果tie weight，那么embedding在cuda上处理
            SetCudaEmbedding(true);
        }
        Embedding(inputIds, this->weight[language_prefix + "embed_tokens.weight"], embeddingResult);

        ToDataType(embeddingResult, hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        bool pagedAttentionInited = false;
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.down_proj.weight";

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
            bool residualAddedInBranch = false;

            if (weight.weight.find(language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
                // Gate Attention Block
                std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

                Data qgate, q, gate, k, v, mergedQkv;
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], mergedQkv);

                    int qgateDim = num_attention_heads * this->head_dim * 2;
                    int kvDim = num_key_value_heads * this->head_dim;
                    Split(mergedQkv, -1, 0, qgateDim, qgate);
                    Split(mergedQkv, -1, qgateDim, qgateDim + kvDim, k);
                    Split(mergedQkv, -1, qgateDim + kvDim, qgateDim + kvDim * 2, v);
                } else {
                    Linear(attenInput, weight[qWeightName], weight[qBiasName], qgate);
                    Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                    Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
                }

                qgate.Reshape({bsz, seqlen, -1, this->head_dim * 2});
                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({bsz, seqlen, -1});

                k.Reshape({bsz, seqlen, -1, this->head_dim});
                v.Reshape({bsz, seqlen, -1, this->head_dim});

                RMSNorm(q, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);
                float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
                ApplyMultimodalRotary(q, allPositionIds, ropeScale);
                ApplyMultimodalRotary(k, allPositionIds, ropeScale);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                std::vector <int> qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);
                PreparePagedAttentionInputs(q, k, v, this->dataType);

                if (batch > 1 && all1) {
                    for (int b = 0; b < batch; b++) {
                        batchPastKeys[b] = pastKeyValues[b * block_cnt + i].first;
                        batchPastValues[b] = pastKeyValues[b * block_cnt + i].second;
                    }

                    Data &kCaches = *batchPastKeys[0];
                    Data &vCaches = *batchPastValues[0];
                    PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                    PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                    AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                    "Qwen3.5 fused batch decode requires paged KV cache.");

                    if (!generatedAppendPagedCacheBatchParams) {
                        GenerateAppendPagedCacheBatchParams(*pagedCacheKManager, batchPastKeys, batch,
                                                            insertIndexs, insertPositions);
                        generatedAppendPagedCacheBatchParams = true;
                    }

                    Data kAppend, vAppend;
                    Permute(k, {1, 0, 2}, kAppend);
                    Permute(v, {1, 0, 2}, vAppend);
                    AppendPagedCacheBatch(*pagedCacheKManager, batchPastKeys, kAppend, insertIndexs, insertPositions);
                    AppendPagedCacheBatch(*pagedCacheVManager, batchPastValues, vAppend, insertIndexs, insertPositions);

                    if (!generatedBatchDecodeParams) {
                        Data qForAttentionHolder;
                        Data &qForAttention = preparePagedAttentionQ(q, kCaches.dataType, qForAttentionHolder);
                        GeneratePagedBatchParams(qForAttention, batchPastKeys, batch,
                                                 qSizes, pageSizes, pageIndexs, lastPageLens);
                        generatedBatchDecodeParams = true;
                    }

                    Data qForAttentionHolder;
                    Data &qForAttention = preparePagedAttentionQ(q, kCaches.dataType, qForAttentionHolder);
                    AttentionPagedBatch(qForAttention, kCaches, vCaches,
                                        qSizes, pageSizes, pageIndexs, lastPageLens,
                                        qkv, qForAttention.dims[0] / kCaches.dims[0],
                                        1.0 / sqrt(head_dim), 1, pagedAttentionInited);
                    pagedAttentionInited = true;
                } else {
                    // Paged Attention
                    Data kCacheDesc = makeCacheDesc(k, pastKey.dataType);
                    Data vCacheDesc = makeCacheDesc(v, pastValue.dataType);
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                    AppendPagedCache(*pagedCacheKManager, pastKey, k);
                    AppendPagedCache(*pagedCacheVManager, pastValue, v);
                    AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(head_dim), 1, pagedAttentionInited);
                    pagedAttentionInited = true;
                }

                if (batch > 1 && all1) {
                    qkv.Reshape({seqlen, bsz, -1});
                    PermuteSelf(qkv, {1, 0, 2});
                } else {
                    PermuteSelf(qkv, {1, 0, 2});
                    qkv.Reshape({seqlen, bsz, -1});
                    PermuteSelf(qkv, {1, 0, 2});
                }

                Sigmoid(gate, gate);
                if (gate.dataType != qkv.dataType) {
                    ToDataType(gate, qkv.dataType);
                }
                MulTo(qkv, gate);

                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
            } else {
                // Gated Delta Net Block
                Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
                DataType linearCacheType = Qwen35LinearAttentionCacheDataType(this->dataType);
                Qwen35PrepareLinearAttentionCache(pastKey, linearCacheType);
                Qwen35PrepareLinearAttentionCache(pastValue, linearCacheType);
                std::string qkvzWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvz.weight";
                std::string baWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_ba.weight";
                std::string qkvzbaWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkvzba.weight";
                std::string conv1dWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.weight";
                std::string conv1dBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.bias";
                std::string aLogName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.A_log";
                std::string dtBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.dt_bias";

                int kd = num_k_heads * head_k_dim, vd = num_v_heads * head_v_dim;
                int mixedQkvzDim = this->weight[qkvzWeightName].dims[0];
                int baMergedDim = this->weight[baWeightName].dims[0];
                bool hasMergedGdnInLinear = this->weight.weight.find(qkvzbaWeightName) != this->weight.weight.end();
                if (hasMergedGdnInLinear && !isSingleTokenDecode &&
                    attenInput.dataDevice == DataDevice::CUDA &&
                    this->weight[qkvzbaWeightName].dataDevice != DataDevice::CUDA) {
                    if (!attenInput.dataDeviceIds.empty()) {
                        this->weight[qkvzbaWeightName].ToDevice(DataDevice::CUDA, attenInput.dataDeviceIds);
                    } else {
                        this->weight[qkvzbaWeightName].ToDevice(DataDevice::CUDA);
                    }
                }
                bool useMergedGdnInLinear = (isSingleTokenDecode || isFusedBatchDecode) && hasMergedGdnInLinear;

                Data gdn_in_merged, mixed_qkvz, ba_merged, qkvConvInput, z, b, a, g;
                if (useMergedGdnInLinear) {
                    Linear(attenInput, weight[qkvzbaWeightName], Data(), gdn_in_merged);
                    if (CanUseSingleRowLastDimView(gdn_in_merged)) {
                        MakeSingleRowLastDimView(gdn_in_merged, 0, mixedQkvzDim, mixed_qkvz);
                        MakeSingleRowLastDimView(gdn_in_merged, mixedQkvzDim, mixedQkvzDim + baMergedDim, ba_merged);
                    } else {
                        Split(gdn_in_merged, -1, 0, mixedQkvzDim, mixed_qkvz);
                        Split(gdn_in_merged, -1, mixedQkvzDim, mixedQkvzDim + baMergedDim, ba_merged);
                    }
                } else {
                    Linear(attenInput, weight[qkvzWeightName], Data(), mixed_qkvz);
                    Linear(attenInput, weight[baWeightName], Data(), ba_merged);
                }

                // Split qkvz -> mixed_qkv + z
                int qkvzDim = kd * 2 + vd;
                if (isSingleTokenDecode && CanUseSingleRowLastDimView(mixed_qkvz)) {
                    MakeSingleRowLastDimView(mixed_qkvz, 0, qkvzDim, qkvConvInput);
                    MakeSingleRowLastDimView(mixed_qkvz, qkvzDim, qkvzDim + vd, z);
                } else {
                    Split(mixed_qkvz, -1, 0, qkvzDim, qkvConvInput);
                    Split(mixed_qkvz, -1, qkvzDim, qkvzDim + vd, z);
                }

                // Split ba -> b + a (note: b and a have dim num_v_heads, not vd).
                // Fused batch decode can consume ba_merged directly below, so defer this split
                // unless the fused recurrent path is unavailable.
                bool baSplitReady = false;
                if (isSingleTokenDecode && CanUseSingleRowLastDimView(ba_merged)) {
                    MakeSingleRowLastDimView(ba_merged, 0, num_v_heads, b);
                    MakeSingleRowLastDimView(ba_merged, num_v_heads, num_v_heads * 2, a);
                    baSplitReady = true;
                } else if (!isFusedBatchDecode) {
                    Split(ba_merged, -1, 0, num_v_heads, b);
                    Split(ba_merged, -1, num_v_heads, num_v_heads * 2, a);
                    baSplitReady = true;
                }

                // mixed_qkv: (bsz, seqlen, key_dim*2+value_dim) -> transpose to (bsz, key_dim*2+value_dim, seqlen)
                if (isSingleTokenDecode) {
                    SwapSingleTokenSeqHeadByReshape(qkvConvInput);
                } else if (isFusedBatchDecode) {
                    qkvConvInput.Reshape({batch, qkvConvInput.dims.back(), 1});
                } else {
                    PermuteSelf(qkvConvInput, {0, 2, 1});
                }
                z.Reshape({bsz, seqlen, -1, head_v_dim});
                Data conv, convOutput;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    bool fusedDecodeConvSilu = false;
                    bool canTryFusedDecodeConvSilu = false;
                    #ifdef USE_CUDA
                    canTryFusedDecodeConvSilu =
                        pastKey.dataDevice == DataDevice::CUDA &&
                        pastKey.dataType == DataType::FLOAT16 &&
                        qkvConvInput.dataDevice == DataDevice::CUDA &&
                        qkvConvInput.dataType == DataType::FLOAT16 &&
                        weight[conv1dWeightName].dataDevice == DataDevice::CUDA &&
                        weight[conv1dWeightName].dataType == DataType::FLOAT32 &&
                        weight[conv1dBiasName].dataDevice == DataDevice::CUDA;
                    #endif
                    if (!canTryFusedDecodeConvSilu) {
                        ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                    }

                    #ifdef USE_CUDA
                    if (canTryFusedDecodeConvSilu) {
                        fusedDecodeConvSilu = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                            pastKey, qkvConvInput, weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
                    }
                    if (!fusedDecodeConvSilu)
                    #endif
                    {
                        if (canTryFusedDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(pastKey, qkvConvInput);
                        }
                        Conv1DPerChannel(
                            pastKey, weight[conv1dWeightName], weight[conv1dBiasName],
                            pastKey.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0,
                            convOutput
                        );
                    }
                    if (!fusedDecodeConvSilu) {
                        Silu(convOutput, convOutput);
                    }
                } else if (isFusedBatchDecode) {
                    std::vector<Data*> linearConvCaches(batch);
                    for (int b = 0; b < batch; b++) {
                        linearConvCaches[b] = pastKeyValues[b * block_cnt + i].first;
                    }
                    bool directBatchDecodeConvSilu = false;
#ifdef USE_CUDA
                    directBatchDecodeConvSilu =
                        FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(
                            linearConvCaches, qkvConvInput,
                            weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
#endif
                    if (!directBatchDecodeConvSilu) {
                        Data batchConvCache;
                        CatBatchFirstDim(linearConvCaches, batchConvCache);
                        bool fusedBatchDecodeConvSilu = false;
#ifdef USE_CUDA
                        fusedBatchDecodeConvSilu =
                            FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                                batchConvCache, qkvConvInput,
                                weight[conv1dWeightName], weight[conv1dBiasName], convOutput);
#endif
                        if (!fusedBatchDecodeConvSilu) {
                            ShiftAppendSingleTokenLinearAttentionCache(batchConvCache, qkvConvInput);
                            Conv1DPerChannel(
                                batchConvCache, weight[conv1dWeightName], weight[conv1dBiasName],
                                batchConvCache.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0,
                                convOutput
                            );
                            Silu(convOutput, convOutput);
                        }
                        SplitBatchFirstDim(batchConvCache, linearConvCaches);
                    }
                    for (int b = 0; b < batch; b++) {
                        Qwen35PrepareLinearAttentionCache(*linearConvCaches[b], linearCacheType);
                    }
                } else {
                    if (qkvConvInput.dims.back() >= 4) {
                        Split(qkvConvInput, -1, qkvConvInput.dims.back() - 4, qkvConvInput.dims.back(), pastKey);
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        Data temp;
                        Mul(qkvConvInput, 1.0f, temp);
                        Repeat(temp, -1, 4, qkvConvInput);
                    }

                    Conv1DPerChannel(
                        qkvConvInput, weight[conv1dWeightName], weight[conv1dBiasName],
                        qkvConvInput.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 3,
                        conv
                    );
                    Split(conv, -1, 0, seqlen, convOutput);
                    Silu(convOutput, convOutput);
                }

                // mixed_qkv: (bsz, conv_dim, seqlen) -> transpose back to (bsz, seqlen, conv_dim)
                if (isSingleTokenDecode) {
                    SwapSingleTokenSeqHeadByReshape(convOutput);
                } else if (isFusedBatchDecode) {
                    convOutput.Reshape({1, batch, convOutput.dims[1]});
                } else {
                    PermuteSelf(convOutput, {0, 2, 1});
                }

                Data core_attn_out, core_attn_out_temp;
                bool fusedBatchRecurrentFromConvBa = false;
#ifdef USE_CUDA
                if (isFusedBatchDecode &&
                    convOutput.dataDevice == DataDevice::CUDA &&
                    convOutput.dataType == DataType::FLOAT16 &&
                    ba_merged.dataDevice == DataDevice::CUDA &&
                    ba_merged.dataType == DataType::FLOAT16 &&
                    ba_merged.dims.size() > 0 &&
                    ba_merged.dims.back() == num_v_heads * 2 &&
                    convOutput.dims.size() > 0 &&
                    convOutput.dims.back() == kd + kd + vd &&
                    weight[aLogName].dataDevice == DataDevice::CUDA &&
                    weight[aLogName].dataType == DataType::FLOAT32 &&
                    weight[dtBiasName].dataDevice == DataDevice::CUDA &&
                    weight[dtBiasName].dataType == DataType::FLOAT32 &&
                    num_k_heads > 0 &&
                    num_v_heads % num_k_heads == 0) {
                    if (inv_scale_data.dataDevice != DataDevice::CUDA) {
                        if (!convOutput.dataDeviceIds.empty()) {
                            inv_scale_data.ToDevice(DataDevice::CUDA, convOutput.dataDeviceIds);
                        } else {
                            inv_scale_data.ToDevice(DataDevice::CUDA);
                        }
                    }

                    std::vector<Data*> recurrentStates(batch);
                    bool canUseFusedRecurrentFromConvBa =
                        inv_scale_data.dataDevice == DataDevice::CUDA &&
                        inv_scale_data.dataType == DataType::FLOAT32 &&
                        inv_scale_data.dims.size() == 1 &&
                        inv_scale_data.dims[0] == head_k_dim;
                    for (int rb = 0; rb < batch; rb++) {
                        recurrentStates[rb] = pastKeyValues[rb * block_cnt + i].second;
                        canUseFusedRecurrentFromConvBa &= recurrentStates[rb] != nullptr &&
                                                          recurrentStates[rb]->dataDevice == DataDevice::CUDA &&
                                                          recurrentStates[rb]->dataType == DataType::FLOAT16 &&
                                                          recurrentStates[rb]->dims.size() == 4 &&
                                                          recurrentStates[rb]->dims[0] == 1 &&
                                                          recurrentStates[rb]->dims[1] == num_v_heads &&
                                                          recurrentStates[rb]->dims[2] == head_k_dim &&
                                                          recurrentStates[rb]->dims[3] == head_v_dim;
                    }

                    if (canUseFusedRecurrentFromConvBa) {
                        float recurrentQScale = 1.0f / pow((float)head_k_dim, 0.5f);
                        FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
                            convOutput, ba_merged, inv_scale_data, weight[aLogName], weight[dtBiasName],
                            recurrentStates, core_attn_out,
                            num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                            rms_norm_eps, recurrentQScale
                        );
                        for (int rb = 0; rb < batch; rb++) {
                            Qwen35PrepareLinearAttentionCache(*recurrentStates[rb], linearCacheType);
                        }
                        core_attn_out.Reshape({1, batch, core_attn_out.dims[1], core_attn_out.dims[3]});
                        fusedBatchRecurrentFromConvBa = true;
                    }
                }
#endif

                if (!fusedBatchRecurrentFromConvBa) {
                    if (!baSplitReady) {
                        Split(ba_merged, -1, 0, num_v_heads, b);
                        Split(ba_merged, -1, num_v_heads, num_v_heads * 2, a);
                        baSplitReady = true;
                    }

                // q / k / v are reused later as MLP scratch buffers, so they must not alias
                // convOutput's lifetime-limited storage in single-token decode.
                Split(convOutput, -1, 0, kd, q);
                Split(convOutput, -1, kd, kd + kd, k);
                Split(convOutput, -1, kd + kd, kd + kd + vd, v);

                q.Reshape({q.dims[0], q.dims[1], -1, head_k_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_k_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_v_dim});

                #ifdef USE_CUDA
                if (b.dataDevice == DataDevice::CUDA &&
                    a.dataDevice == DataDevice::CUDA &&
                    (b.dataType == DataType::FLOAT32 || b.dataType == DataType::FLOAT16) &&
                    a.dataType == b.dataType &&
                    weight[aLogName].dataDevice == DataDevice::CUDA &&
                    weight[dtBiasName].dataDevice == DataDevice::CUDA) {
                    SigmoidMambaSoftplus(b, a, weight[aLogName], weight[dtBiasName], g);
                } else
                #endif
                {
                    Sigmoid(b, b);
                    MambaSoftplus(a, weight[aLogName], weight[dtBiasName], g);
                }


                Data &last_recurrent_state = pastValue;
                #ifdef USE_CUDA
                if (!isSingleTokenDecode) {
                    Qwen35EnsureCudaLinearAttnStateKVLayout(last_recurrent_state);
                }
                #endif

                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    bool fusedSingleDecode = false;
#ifdef USE_CUDA
                    fusedSingleDecode = Qwen35TryCudaLinearAttnSingleDecodeNormRecurrent(
                        q, k, v, g, b, inv_scale_data, rms_norm_eps, last_recurrent_state, core_attn_out
                    );
#endif
                    if (!fusedSingleDecode) {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);

                        SwapSingleTokenSeqHeadByReshape(q);
                        SwapSingleTokenSeqHeadByReshape(k);
                        SwapSingleTokenSeqHeadByReshape(v);
                        SwapSingleTokenSeqHeadByReshape(b);
                        SwapSingleTokenSeqHeadByReshape(g);

                        float scale = 1.0f / pow(q.dims.back(), 0.5);
                        float recurrentQScale = 1.0f;
                        if (q.dataDevice == DataDevice::CUDA) {
                            recurrentQScale = scale;
                        } else {
                            Mul(q, scale, q);
                        }

                        RecurrentGatedDeltaRule (
                            q, k, v, g, b,
                            last_recurrent_state,
                            core_attn_out, recurrentQScale
                        );
                        SwapSingleTokenSeqHeadByReshape(core_attn_out);
                    }
                } else if (isFusedBatchDecode) {
                    RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                    RMSNorm(k, inv_scale_data, rms_norm_eps, k);

                    q.Reshape({batch, q.dims[2], 1, q.dims[3]});
                    k.Reshape({batch, k.dims[2], 1, k.dims[3]});
                    v.Reshape({batch, v.dims[2], 1, v.dims[3]});
                    b.Reshape({batch, b.dims[2], 1});
                    g.Reshape({batch, g.dims[2], 1});

                    std::vector<Data*> recurrentStates(batch);
                    bool canUseCudaBatchRecurrent = q.dataDevice == DataDevice::CUDA &&
                                                    q.dataType == DataType::FLOAT16;
                    for (int rb = 0; rb < batch; rb++) {
                        recurrentStates[rb] = pastKeyValues[rb * block_cnt + i].second;
                        canUseCudaBatchRecurrent &= recurrentStates[rb] != nullptr &&
                                                    recurrentStates[rb]->dataDevice == DataDevice::CUDA &&
                                                    recurrentStates[rb]->dataType == q.dataType &&
                                                    recurrentStates[rb]->dims.size() == 4 &&
                                                    recurrentStates[rb]->dims[0] == 1;
                    }

                    float scale = 1.0f / pow(q.dims.back(), 0.5);
                    float recurrentQScale = 1.0f;
                    if (q.dataDevice == DataDevice::CUDA) {
                        recurrentQScale = scale;
                    } else {
                        Mul(q, scale, q);
                    }

                    if (canUseCudaBatchRecurrent) {
#ifdef USE_CUDA
                        FastllmRecurrentGatedDeltaRuleBatch(q, k, v, g, b, recurrentStates, core_attn_out, recurrentQScale);
#else
                        ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
                    } else {
                        Data batchRecurrentState;
                        CatBatchFirstDim(recurrentStates, batchRecurrentState);

                        RecurrentGatedDeltaRule(
                            q, k, v, g, b,
                            batchRecurrentState,
                            core_attn_out, recurrentQScale
                        );

                        SplitBatchFirstDim(batchRecurrentState, recurrentStates);
                    }
                    for (int rb = 0; rb < batch; rb++) {
                        Qwen35PrepareLinearAttentionCache(*recurrentStates[rb], linearCacheType);
                    }
                    core_attn_out.Reshape({1, batch, core_attn_out.dims[1], core_attn_out.dims[3]});
                } else {
                    if (num_v_heads / num_k_heads > 1) {
                        Data qrepeat, krepeat;
                        Mul(q, 1.0f, qrepeat);
                        Mul(k, 1.0f, krepeat);

                        qrepeat.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                        krepeat.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});

                        Repeat(qrepeat, 3, num_v_heads / num_k_heads, q);
                        Repeat(krepeat, 3, num_v_heads / num_k_heads, k);

                        q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                        k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                    }

                    {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});

                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];

                    int chunk_size = 64;
                    int v_head_dim_local = v.dims.back();
                    int seq = k.dims[2];
                    int pad_size = (chunk_size - seq % chunk_size) % chunk_size;

                    Data qq, kk, vv, bb, gg, decayMask;
                    Data kk_pad, vv_pad, bb_pad, gg_pad;
                    Data *pkk, *pvv, *pbb, *pgg;
                    if (pad_size > 0) {
                        Data qtemp;
                        Pad(q, 2, pad_size, qtemp);
                        Pad(k, 2, pad_size, kk_pad);
                        Pad(v, 2, pad_size, vv_pad);
                        Pad(b, 2, pad_size, bb_pad);
                        Pad(g, 2, pad_size, gg_pad);
                        float scale = 1.0f / pow(qtemp.dims.back(), 0.5);
                        Mul(qtemp, scale, qq);
                        pkk = &kk_pad; pvv = &vv_pad; pbb = &bb_pad; pgg = &gg_pad;
                    } else {
                        // Avoid 5 Mul(x, 1.0f, y) copies when no padding needed
                        float scale = 1.0f / pow(q.dims.back(), 0.5);
                        Mul(q, scale, qq);
                        pkk = &k; pvv = &v; pbb = &b; pgg = &g;
                    }

                    int tot_heads = seq + pad_size;

                    pbb->Resize({(*pbb).dims[0], (*pbb).dims[1], (*pbb).dims[2], 1});
                    Data k_beta, v_beta;
                    Mul(*pkk, 1.0f, k_beta);
                    Mul(*pvv, 1.0f, v_beta);
                    MulTo(k_beta, *pbb);
                    MulTo(v_beta, *pbb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunk_size, qq.dims.back()});
                    pkk->Reshape({(*pkk).dims[0], (*pkk).dims[1], -1, chunk_size, (*pkk).dims.back()});
                    k_beta.Reshape({k_beta.dims[0], k_beta.dims[1], -1, chunk_size, k_beta.dims.back()});
                    v_beta.Reshape({v_beta.dims[0], v_beta.dims[1], -1, chunk_size, v_beta.dims.back()});
                    pgg->Reshape({(*pgg).dims[0], (*pgg).dims[1], -1, chunk_size});

                    CumSumLastDim(*pgg);
                    MakeDecayMask(*pgg, decayMask);

                    Data at, attn;
                    MatMulTransB(k_beta, *pkk, at);
                    Mul(at, -1.0f, attn);
                    MulTo(attn, decayMask);

                    CausalMask(attn, 0, 0.0f);
                    TransferAttn(attn);
                    MatMul(attn, v_beta, vv);
                    Data k_cumdecay, g_exp;
                    Exp(*pgg, g_exp);

                    MulTo(k_beta, g_exp);
                    MatMul(attn, k_beta, k_cumdecay);

                    MatMulTransB(qq, *pkk, attn);
                    MulTo(attn, decayMask);
                    CausalMask(attn, 1, 0.0f);

                    if (last_recurrent_state.dims.size() == 0) {
#ifdef USE_CUDA
                        if (qq.dataDevice == DataDevice::CUDA) {
                            last_recurrent_state.dataDevice = qq.dataDevice;
                            last_recurrent_state.dataDeviceIds = qq.dataDeviceIds;
                        }
#endif
                        last_recurrent_state.Resize({key_batch_size, key_sequence_length, key_k_head_dim, v_head_dim_local});
                        last_recurrent_state.isLinearAttentionTransposed = false;
                        last_recurrent_state.Allocate(0.0f);
                    }

                    auto runChunkPrefillReference = [&](Data &state, Data &out) {
                        auto makeChunk4D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3], src.dims[4]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3], src.strides[4]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };
                        auto makeChunk3D = [](Data &src, int idx, Data &dst) {
                            dst.dims = {src.dims[1], src.dims[2], src.dims[3]};
                            dst.strides = {src.strides[1], src.strides[2], src.strides[3]};
                            dst.FakeFrom(src, (size_t) idx * src.strides[0] * src.unitSize);
                        };

                        for (int ci = 0; ci < tot_heads / chunk_size; ci++) {
                            Data q_i, k_i, v_i, attn_i, k_cumdecay_i;
                            makeChunk4D(qq, ci, q_i);
                            makeChunk4D(*pkk, ci, k_i);
                            makeChunk4D(vv, ci, v_i);
                            makeChunk4D(attn, ci, attn_i);
                            makeChunk4D(k_cumdecay, ci, k_cumdecay_i);

                            Data v_prime, v_new;
                            MatMul(k_cumdecay_i, state, v_prime);
                            Mul(v_prime, -1.0f, v_new);
                            AddTo(v_new, v_i);

                            Data attn_inter, g_i, g_i_exp;
                            Split(*pgg, 0, ci, ci + 1, g_i);
                            g_i.Resize({g_i.dims[1], g_i.dims[2], g_i.dims[3]});
                            Split(g_exp, 0, ci, ci + 1, g_i_exp);
                            g_i_exp.Resize({g_i_exp.dims[1], g_i_exp.dims[2], g_i_exp.dims[3], 1});
                            MulTo(q_i, g_i_exp);

                            MatMul(q_i, state, attn_inter);
                            Data atv;
                            MatMul(attn_i, v_new, atv);
                            AddTo(atv, attn_inter);
                            atv.Resize({atv.dims[0], atv.dims[1], 1, atv.dims[2], atv.dims[3]});
                            if (ci == 0) {
                                Mul(atv, 1.0f, out);
                            } else {
                                Mul(out, 1.0f, core_attn_out_temp);
                                Cat(core_attn_out_temp, atv, 3, out);
                            }

                            Data g_i_last, g_i_last_repeat, g_i_delta, g_i_scale;
                            Split(g_i, 2, g_i.dims[2] - 1, g_i.dims[2], g_i_last);
                            Repeat(g_i_last, 2, g_i.dims[2], g_i_last_repeat);
                            Mul(g_i, -1.0f, g_i_delta);
                            AddTo(g_i_last_repeat, g_i_delta);
                            Exp(g_i_last_repeat, g_i_scale);
                            g_i_scale.Resize({g_i_scale.dims[0], g_i_scale.dims[1], g_i_scale.dims[2], 1});
                            MulTo(k_i, g_i_scale);

                            Data k_i_v_new;
                            PermuteSelf(k_i, {0, 1, 3, 2});
                            MatMul(k_i, v_new, k_i_v_new);

                            Data g_i_exp_last;
                            Split(g_i_exp, 2, g_i_exp.dims[2] - 1, g_i_exp.dims[2], g_i_exp_last);
                            MulTo(state, g_i_exp_last);
                            AddTo(state, k_i_v_new);
                        }
                    };

                    bool useFusedChunkPrefill = false;
#ifdef USE_CUDA
                    if (qq.dataDevice == DataDevice::CUDA &&
                        pkk->dataDevice == DataDevice::CUDA &&
                        vv.dataDevice == DataDevice::CUDA &&
                        pgg->dataDevice == DataDevice::CUDA &&
                        attn.dataDevice == DataDevice::CUDA &&
                        k_cumdecay.dataDevice == DataDevice::CUDA &&
                        last_recurrent_state.dataDevice == DataDevice::CUDA) {
                        useFusedChunkPrefill = GetFastllmEnv().useFusedGdnPrefill;
                    }
#endif

                    if (useFusedChunkPrefill) {
#ifdef USE_CUDA
                        ChunkGatedDeltaRulePrefill(
                            qq, *pkk, vv, *pgg, attn, k_cumdecay,
                            last_recurrent_state, core_attn_out
                        );
#endif
                    } else {
                        PermuteSelf(qq, {2, 0, 1, 3, 4});
                        PermuteSelf(*pkk, {2, 0, 1, 3, 4});
                        PermuteSelf(vv, {2, 0, 1, 3, 4});
                        PermuteSelf(attn, {2, 0, 1, 3, 4});
                        PermuteSelf(k_cumdecay, {2, 0, 1, 3, 4});
                        PermuteSelf(g_exp, {2, 0, 1, 3});
                        PermuteSelf(*pgg, {2, 0, 1, 3});
                        runChunkPrefillReference(last_recurrent_state, core_attn_out);
                    }

                    core_attn_out.Reshape({core_attn_out.dims[0], core_attn_out.dims[1], -1, core_attn_out.dims.back()});
                    if (pad_size > 0) {
                        Split(core_attn_out, 2, 0, seq, core_attn_out_temp);
                        PermuteSelf(core_attn_out_temp, {0, 2, 1, 3});
                        Mul(core_attn_out_temp, 1.0f, core_attn_out);
                    } else {
                        PermuteSelf(core_attn_out, {0, 2, 1, 3});
                    }
                }
                }

                {
                    std::vector <int> zShape = z.dims;
                    std::string outNormWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.norm.weight";
                    core_attn_out.Reshape({-1, core_attn_out.dims.back()});
                    z.Reshape({-1, z.dims.back()});

#ifdef USE_CUDA
                    if (isSingleTokenDecode &&
                        core_attn_out.dataDevice == DataDevice::CUDA &&
                        core_attn_out.dataType == DataType::FLOAT16 &&
                        z.dataDevice == DataDevice::CUDA &&
                        z.dataType == DataType::FLOAT16 &&
                        this->weight[outNormWeightName].dataDevice == DataDevice::CUDA &&
                        this->weight[outNormWeightName].dataType == DataType::FLOAT32 &&
                        core_attn_out.dims == z.dims &&
                        FastllmCudaRMSNormSiluMulFloat16(
                            core_attn_out, this->weight[outNormWeightName], z, core_attn_out, rms_norm_eps)) {
                    } else
#endif
                    {
                        RMSNorm(core_attn_out, this->weight[outNormWeightName], rms_norm_eps, core_attn_out);
                        Silu(z, z);
                        MulTo(core_attn_out, z);
                    }

                    core_attn_out.Reshape({zShape[0], zShape[1], -1});
                    if (isSingleTokenDecode &&
                        core_attn_out.dataDevice == DataDevice::CUDA &&
                        core_attn_out.dataType == DataType::FLOAT16 &&
                        hiddenStates.dataDevice == DataDevice::CUDA &&
                        hiddenStates.dataType == DataType::FLOAT16) {
                        int n = core_attn_out.Count(0) / core_attn_out.dims.back();
                        int m = core_attn_out.dims.back();
                        int k = hiddenStates.dims.back();
#ifdef USE_CUDA
                        if (FastllmCudaHalfMatMulFloat16AddToNoBias(
                                core_attn_out,
                                this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                                hiddenStates, n, m, k)) {
                            residualAddedInBranch = true;
                        } else
#endif
                        {
                            Linear(core_attn_out,
                                   this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                                   Data(), attenInput);
                        }
                    } else {
                        Linear(core_attn_out,
                            this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"],
                            Data(), attenInput);
                    }
                }
            }

            if (!residualAddedInBranch) {
                AddTo(hiddenStates, attenInput);
            }
            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            if (weight.weight.find(swigluWeightName) != weight.weight.end() &&
                weight.weight.find(downWeightName) != weight.weight.end()) {
                MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
                continue;
            }

            std::string gateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.weight";
            if (weight.weight.find(gateWeightName) == weight.weight.end()) {
                ErrorInFastLLM("Qwen3.5 layer " + std::to_string(i) + " has neither dense MLP nor MoE weights.");
            }

            Data w1, w2, w3;
            Data routerLogits;
            Data sharedGate;
            Data moeFinal, moeFinal2;
            Data tempInput, tempOutput;
            Data attenPart, moePart;

            std::string gateBiasName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";
            std::string firstExpertGateupName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight";
            std::string sharedGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gateup_proj.weight";
            std::string sharedGateProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.gate_proj.weight";
            std::string sharedUpProjWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.up_proj.weight";
            std::string sharedDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert.down_proj.weight";
            std::string sharedExpertGateWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.shared_expert_gate.weight";
            Data *gateBiasData = weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr;

            int flatBatch = attenInput.dims[0];
            int flatLen = attenInput.dims[1];
            attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});

            Linear(attenInput, weight[gateWeightName], Data(), routerLogits);
            ToDataType(routerLogits, DataType::FLOAT32);
            if (gateBiasData != nullptr) {
                ToDataType(*gateBiasData, DataType::FLOAT32);
            }
            Softmax(routerLogits, routerLogits, -1);

            if (weight.weight.find(sharedDownWeightName) != weight.weight.end()) {
                if (weight.weight.find(sharedGateupWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateupWeightName], Data(), w3);
                    Swiglu(w3, w1);
                } else if (weight.weight.find(sharedGateProjWeightName) != weight.weight.end() &&
                           weight.weight.find(sharedUpProjWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateProjWeightName], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[sharedUpProjWeightName], Data(), w3);
                    MulTo(w1, w3);
                }
                if (w1.dims.size() != 0) {
                    Linear(w1, weight[sharedDownWeightName], Data(), moeFinal2);
                    if (weight.weight.find(sharedExpertGateWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[sharedExpertGateWeightName], Data(), sharedGate);
                        Sigmoid(sharedGate, sharedGate);
                        MulTo(moeFinal2, sharedGate);
                    }
                }
            }

            bool useMergeMoe = weight.weight.find(firstExpertGateupName) != weight.weight.end() &&
                               !weights[i].empty() && CanRunMergeMOE(attenInput, biass[i]);
            if (useMergeMoe) {
                Data expertIndex, expertScore;
                SelectExpert(routerLogits, expertIndex, expertScore, this->num_experts_per_tok, this->norm_topk_prob,
                             this->routed_scaling_factor, gateBiasData);
                this->ApplyMoeDeviceMapForLayer(i);
                MergeMOE(
                    attenInput, expertIndex, expertScore,
                    weights[i], biass[i],
                    w1, w2, w3, tempInput, tempOutput,
                    1.0f,
                    moeFinal, i
                );
            } else {
                routerLogits.ToDevice(DataDevice::CPU);
                float *cpuRouterLogits = (float*)routerLogits.cpuData;
                float *cpuBias = nullptr;
                if (gateBiasData != nullptr) {
                    gateBiasData->ToDevice(DataDevice::CPU);
                    cpuBias = (float*)gateBiasData->cpuData;
                }
                int expertCount = routerLogits.dims.back();

                moeFinal.dataType = hiddenStates.dataType;
                moeFinal.dataDevice = attenInput.dataDevice;
                moeFinal.dataDeviceIds = attenInput.dataDeviceIds;
                moeFinal.UpdateUnitSize();
                moeFinal.Resize({0, attenInput.dims[1]});
                moeFinal.Expansion(attenInput.dims);
                for (int b = 0; b < flatBatch * flatLen; b++) {
                    float *cur = cpuRouterLogits + b * expertCount;
                    std::vector <std::pair <float, int> > candidates;
                    candidates.reserve(expertCount);
                    for (int j = 0; j < expertCount; j++) {
                        float score = cur[j];
                        if (cpuBias != nullptr) {
                            score += cpuBias[j];
                        }
                        candidates.push_back(std::make_pair(-score, j));
                    }
                    std::sort(candidates.begin(), candidates.end());

                    Data *currentData = &attenInput;
                    if (flatBatch * flatLen != 1) {
                        Split(attenInput, 0, b, b + 1, attenPart);
                        currentData = &attenPart;
                    }
                    moePart.dataType = hiddenStates.dataType;
                    moePart.dataDevice = currentData->dataDevice;
                    moePart.dataDeviceIds = currentData->dataDeviceIds;
                    moePart.UpdateUnitSize();
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);

                    float sum = 0.0f;
                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        sum += cur[candidates[j].second];
                    }
                    if (!this->norm_topk_prob) {
                        sum = 1.0f;
                    }

                    for (int j = 0; j < this->num_experts_per_tok; j++) {
                        int idx = candidates[j].second;
                        float value = cur[idx];
                        if (sum != 0.0f) {
                            value /= sum;
                        }
                        value *= this->routed_scaling_factor;

                        std::string expertGateupWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".gateup_proj.weight";
                        std::string expertDownWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.experts." + std::to_string(idx) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(expertGateupWeightName) != weight.weight.end() &&
                                        weight.weight.find(expertDownWeightName) != weight.weight.end(),
                                        "Qwen3.5 MoE expert weights are incomplete.");
                        Linear(*currentData, weight[expertGateupWeightName], Data(), w3);
                        Swiglu(w3, w1);
                        Linear(w1, weight[expertDownWeightName], Data(), w2);
                        if (w2.dataType != moePart.dataType) {
                            ToDataType(w2, moePart.dataType);
                        }
                        AddTo(moePart, w2, value);
                    }
                    if (moePart.dataType != moeFinal.dataType) {
                        ToDataType(moePart, moeFinal.dataType);
                    }
                    CatDirect(moeFinal, moePart, 0);
                }
                moeFinal.expansionDims.clear();
            }

            moeFinal.Reshape(hiddenStates.dims);
            Data tempMoeFinal;
            tempMoeFinal.CopyFrom(moeFinal);
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            if (tempMoeFinal.dataType != hiddenStates.dataType) {
                ToDataType(tempMoeFinal, hiddenStates.dataType);
            }
            AddTo(hiddenStates, tempMoeFinal);
            if (moeFinal2.dims.size() != 0) {
                moeFinal2.Reshape(hiddenStates.dims);
                if (moeFinal2.dataType != hiddenStates.dataType) {
                    ToDataType(moeFinal2, hiddenStates.dataType);
                }
                AddTo(hiddenStates, moeFinal2);
            }
        }

        std::string lmHeadWeightName = "lm_head.weight";
        if (this->weight.weight.find(lmHeadWeightName) == this->weight.weight.end()) {
            lmHeadWeightName = language_prefix + "embed_tokens.weight";
        }
        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight[language_prefix + "norm.weight"], &weight[lmHeadWeightName],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool Qwen3_5Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    std::string Qwen3_5Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3_5Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3_5Model::WarmUp() {
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight[language_prefix + "embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->kvCacheId = 0;
        elementsInKVCachePerToken = 0;
        bool foundTokenGrowingCache = false;
        for (int i = 0; i < block_cnt; i++) {
            if (pastKeyValues[i].first.isLinearAttention || pastKeyValues[i].second.isLinearAttention) {
                continue;
            }
            if (pastKeyValues[i].first.dims.size() < 3 || pastKeyValues[i].second.dims.size() < 3) {
                continue;
            }
            if (!foundTokenGrowingCache) {
                this->kvCacheId = i;
                foundTokenGrowingCache = true;
            }
            elementsInKVCachePerToken +=
                (long long)pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                (long long)pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
    }
}
