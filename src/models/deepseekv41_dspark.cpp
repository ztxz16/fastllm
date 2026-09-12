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
//     bigram 修正后逐位置贪心采样，得到 block_size 个候选 token 和它们的置信度。
//
// 校验：候选 token 与锚点拼成一个 block_size + 1 长度的片段喂给目标模型
// （ForwardSegments 天然支持一次多 token），逐位置贪心比对，第一个不匹配处截断。
// 因此开启 DSpark 与关闭时的贪心输出完全一致；草稿模型只影响接受率。
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
#include "utils.h"
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

namespace fastllm {
    namespace {
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

        struct DsRopeParams {
            int ropeDim;
            float base;
            int originalSeqLen;
            float factor;
            int betaFast;
            int betaSlow;
        };

        void DsRotaryQuant(Data &x, const DsRopeParams &rope, int startPos, int posStep,
                           bool inverse, int quantMode, int quantBlock, int quantDim = -1) {
            static const bool disableFakeQuant = DsEnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            if (disableFakeQuant) {
                quantMode = 0;
            }
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

        void DsWindowStore(const Data &chunkKV, Data &ring, int startPos, int windowSize) {
            DsExecutor().Run("DeepSeekV41WindowStore", {
                {"chunk", (Data*)&chunkKV}, {"ring", &ring}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        float DsSoftplus(float x) {
            return x > 20.0f ? x : std::log1p(std::exp(x));
        }

        float DsSigmoid(float x) {
            return x >= 0.0f ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
        }

        // ---------------- 接受率与分段计时 ----------------
        // FASTLLM_DSPARK_STATS=1 累计统计（退出时与每 N 轮打印一次），=2 额外逐轮打印一行。
        // FASTLLM_DSPARK_STATS_EVERY=N 控制中途汇总的频率（默认 64，0 表示只在退出时打印）。
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
            std::vector<uint64_t> acceptHist;   // 接受长度 0..blockSize
            std::vector<uint64_t> offerHist;    // 置信度截断后送去校验的候选数 0..blockSize
            uint64_t noProposal = 0;        // 该轮没有可用候选（还没生成 / 锚点不匹配）
            double verifyForward = 0.0, plainForward = 0.0, commit = 0.0;
            double draftTotal = 0.0, draftMain = 0.0, draftLayers = 0.0;
            double draftHead = 0.0, draftMarkov = 0.0, draftConf = 0.0;
            double draftMarkovBias = 0.0, draftMarkovArgmax = 0.0;
            uint64_t draftMarkovFused = 0;
            uint64_t draftCalls = 0;        // 真正生成了候选的次数
            double mainOnly = 0.0;          // 只更新草稿滑窗、不生成候选（prefill 分块）
            uint64_t mainOnlyCalls = 0;
        };

        // DsparkRunDraft 把自己的分段耗时放这里，由 DsparkAdvance 取走（避免改函数签名）
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
            int level = 0;
            uint64_t reportEvery = 64;
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
                if (v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0) {
                    level = atoi(v);
                    if (level <= 0) {
                        level = 1;
                    }
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

            void AddDraft(double total, double main, const DsDraftTiming &parts) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.draftCalls++;
                stat.draftTotal += total;
                stat.draftMain += main;
                stat.draftLayers += parts.layers;
                stat.draftHead += parts.head;
                stat.draftMarkov += parts.markov;
                stat.draftMarkovBias += parts.markovBias;
                stat.draftMarkovArgmax += parts.markovArgmax;
                stat.draftMarkovFused += parts.markovFused ? 1 : 0;
                stat.draftConf += parts.conf;
            }

            void AddMainOnly(double ms) {
                if (level == 0) {
                    return;
                }
                std::lock_guard<std::mutex> guard(mutex);
                if (!WarmedLocked()) {
                    return;
                }
                stat.mainOnlyCalls++;
                stat.mainOnly += ms;
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
                printf("[DSpark %s] 每次草稿 %.3f ms = main %.3f + 三层 %.3f + head %.3f + markov %.3f"
                       " + confidence %.3f（%llu 次）\n",
                       tag, s.draftTotal / draft, s.draftMain / draft, s.draftLayers / draft,
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
                if (s.mainOnlyCalls > 0) {
                    printf("[DSpark %s] 只更新草稿滑窗（prefill 分块等）%llu 次，每次 %.3f ms\n",
                           tag, (unsigned long long)s.mainOnlyCalls,
                           s.mainOnly / (double)s.mainOnlyCalls);
                }
                // 每产出一个 token 的实际开销：目标前向 + 草稿 + 回滚
                const double totalMs = s.verifyForward + s.plainForward + s.draftTotal + s.mainOnly + s.commit;
                if (outputs > 0) {
                    printf("[DSpark %s] 每个输出 token 合计 %.3f ms（目标前向 %.3f + 草稿 %.3f + 回滚 %.3f）\n",
                           tag, totalMs / (double)outputs,
                           (s.verifyForward + s.plainForward) / (double)outputs,
                           (s.draftTotal + s.mainOnly) / (double)outputs,
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

        // FASTLLM_DSV41_DEBUG_STATE：每次前向后把各层缓存的长度写到文件，
        // 用来对比"开 DSpark（含回滚）"与"不开 DSpark"两条路径的状态是否一致
        const char *DsDebugStatePath() {
            static const char *path = std::getenv("FASTLLM_DSV41_DEBUG_STATE");
            return path != nullptr && path[0] != '\0' ? path : nullptr;
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
        AssertInFastLLM(v41DsparkTokens <= v41DsparkBlockSize,
                        "DeepSeekV41 DSpark: --dspark N must not exceed dspark_block_size (" +
                        std::to_string(v41DsparkBlockSize) + ").");
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
        printf("[Fastllm] DeepSeek-V4.1 DSpark: %d draft layers, block size %d (verifying %d), "
               "%d experts (top-%d), target layers = [", v41DsparkLayers, v41DsparkBlockSize,
               v41DsparkTokens, v41DsparkExperts, v41DsparkTopk);
        for (size_t i = 0; i < v41DsparkTargetLayerIds.size(); i++) {
            printf("%s%d", i == 0 ? "" : ", ", v41DsparkTargetLayerIds[i]);
        }
        printf("], confidence threshold %.3f\n", v41DsparkConfidenceThreshold);
        fflush(stdout);
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

    bool DeepSeekV41Model::DsparkSupportsRequest(const GenerationConfig &config,
                                                 const DeepSeekV41RequestState &state) const {
        if (!v41DsparkEnabled) {
            return false;
        }
        // 校验一次要比较 block 个位置的贪心 token，采样 / 重复惩罚 / 工具约束 / logits
        // 输出都会改变 token 的产生方式，这些请求退回普通解码。
        if (!config.IsSimpleGreedy() || config.output_logits || config.output_token_least > 0) {
            return false;
        }
        if (config.do_sample && config.temperature > 1e-6f) {
            return false;
        }
        // 图文请求的 span 只在 prefill 出现，decode 与纯文本相同，但 gate.bias_vl
        // 与 Engram 掩码的处理只覆盖 prefill，这里保守地关闭投机。
        if (state.pendingMultimodal != nullptr || !state.imageSpans.empty()) {
            return false;
        }
        return true;
    }

    std::shared_ptr<DeepSeekV41DsparkState> DeepSeekV41Model::GetOrCreateDsparkState(
            DeepSeekV41RequestState &state, int startPos) {
        if (!state.dspark) {
            state.dspark = std::make_shared<DeepSeekV41DsparkState>();
            state.dspark->layers.resize(v41DsparkLayers);
            // 前缀缓存恢复出来的前缀没有草稿侧的滑窗（main hidden 无法从目标缓存反推），
            // committed 从恢复长度开始，filled 从 0 重新累积。
            state.dspark->committed = startPos;
            state.dspark->filled = 0;
        }
        return state.dspark;
    }

    int DeepSeekV41Model::DsparkTakePending(DeepSeekV41RequestState &state, const Data &inputIds,
                                            int seqlen) {
        if (!state.dspark || state.dspark->pending.empty()) {
            return -1;
        }
        auto &pending = state.dspark->pending;
        if (seqlen == 1) {
            std::vector<int> ids = DsReadIds(inputIds);
            // 调度器可能改写返回的 token（工具约束、停止词等），只在完全一致时出队
            if (ids.size() == 1 && ids[0] == pending.front().first) {
                const int ret = pending.front().second;
                pending.pop_front();
                DsProfiler().AddPendingHit();
                return ret;
            }
        }
        // 不匹配：丢弃剩余的已校验 token，退回普通解码。缓存里多出来的 token
        // 位置在 state.totalLen 里，下一次前向会因为位置不一致而报错，因此这里
        // 也把草稿状态标为失效，由调用方走完整前向。
        pending.clear();
        state.dspark->disabled = true;
        return -1;
    }

    int DeepSeekV41Model::DsparkBuildVerifyInput(DeepSeekV41RequestState &state, const Data &inputIds,
                                                 int startPos, Data &verifyIds, std::vector<int> &drafts) {
        drafts.clear();
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
        std::vector<int> ids = DsReadIds(inputIds);
        if (ids.size() != 1 || ids[0] != dspark.anchor) {
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
            }
        }
        std::vector<float> values;
        values.reserve(count + 1);
        values.push_back((float)dspark.anchor);
        for (int t : drafts) {
            values.push_back((float)t);
        }
        // Data 没有深拷贝赋值，必须用 CopyFrom，否则临时对象析构后缓冲区悬空
        verifyIds.CopyFrom(Data(DataType::FLOAT32, {1, count + 1}, values));
        dspark.drafts.clear();
        return count;
    }

    void DeepSeekV41Model::DsparkDebugDumpState(const DeepSeekV41RequestState &state, const char *tag) {
        const char *path = DsDebugStatePath();
        if (path == nullptr) {
            return;
        }
        FILE *file = fopen(path, "a");
        if (file == nullptr) {
            return;
        }
        fprintf(file, "%s total=%d engram=%d", tag, state.totalLen, (int)state.engramHistory.size());
        for (int layer = 0; layer < (int)state.layers.size(); layer++) {
            const DeepSeekV41LayerCache &cache = state.layers[layer];
            fprintf(file, " | L%d len=%d blocks=%d tail=%d ring=%d idx=%d ckv=%d", layer, cache.totalLen,
                    cache.compressedBlocks, cache.rawTail,
                    cache.windowKV.dims.size() == 3 ? cache.windowKV.dims[1] : -1,
                    cache.indexK.dims.size() == 3 ? cache.indexK.dims[1] : -1,
                    cache.compressedKV.dims.size() == 3 ? cache.compressedKV.dims[1] : -1);
        }
        fprintf(file, "\n");
        fclose(file);
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
        if (!v41DsparkEnabled || inputEmbeds != nullptr || imageMask != nullptr) {
            std::vector<int> plain = ForwardSegments(segments, inputIds, inputEmbeds, imageMask,
                                                     generationConfigs, lastTokens, retLogits,
                                                     samplingPastKeyValues);
            for (auto &seg : segments) {
                DsparkDebugDumpState(*seg.state, "plain");
            }
            return plain;
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

        // 单片段时把候选拼进输入，一次前向校验 block 个位置
        std::vector<int> draftTokens;
        Data verifyIds;
        const Data *idsPtr = &inputIds;
        int drafts = 0;
        const int startPos = numSegments == 1 ? segments[0].startPos : 0;
        const int seqlen = numSegments == 1 ? segments[0].seqlen : 0;
        if (allowSpeculate && numSegments == 1 && active[0] && seqlen == 1 && startPos > 0) {
            drafts = DsparkBuildVerifyInput(*segments[0].state, inputIds, startPos, verifyIds, draftTokens);
            if (drafts > 0) {
                idsPtr = &verifyIds;
                segments[0].seqlen = 1 + drafts;
                scratches[0].deferWindow = true;
                scratches[0].wantAllGreedy = true;
                scratches[0].windowKV.resize(block_cnt);
                scratches[0].rawKV.resize(block_cnt);
                scratches[0].rawScore.resize(block_cnt);
                scratches[0].prevRawTail.assign(block_cnt, 0);
                scratches[0].prevBlocks.assign(block_cnt, 0);
            }
        }

        const double forwardStart = DsTick();
        std::vector<int> ret = ForwardSegments(segments, *idsPtr, nullptr, nullptr, generationConfigs,
                                               lastTokens, retLogits, samplingPastKeyValues);
        const double forwardMs = DsElapsed(forwardStart);
        // 单请求单 token 的普通前向：与校验前向对照，差值就是多校验 N 个候选的边际代价
        if (drafts == 0 && numSegments == 1 && seqlen == 1 && active[0]) {
            DsProfiler().AddPlainForward(forwardMs);
        }

        if (drafts > 0) {
            DeepSeekV41RequestState &state = *segments[0].state;
            DeepSeekV41SpecScratch &scratch = scratches[0];
            AssertInFastLLM((int)scratch.greedy.size() == drafts + 1,
                            "DeepSeekV41 DSpark: the verifier did not return per-position tokens.");
            int accepted = 0;
            while (accepted < drafts && scratch.greedy[accepted] == draftTokens[accepted]) {
                accepted++;
            }
            const int commitCount = accepted + 1;
            const double commitStart = DsTick();
            DsparkCommitPrefix(state, scratch, startPos, commitCount, drafts + 1);
            const double commitMs = DsElapsed(commitStart);
            DsProfiler().AddVerify(drafts, accepted, forwardMs, commitMs);
            ret.assign(1, scratch.greedy[0]);
            for (int j = 1; j <= accepted; j++) {
                state.dspark->pending.push_back(std::make_pair(scratch.greedy[j - 1], scratch.greedy[j]));
            }
            const long long round = v41DsparkVerifyRounds.fetch_add(1) + 1;
            v41DsparkProposedTokens.fetch_add(drafts);
            v41DsparkAcceptedTokens.fetch_add(accepted);
            DsRecordStats(round, drafts, accepted);
            state.dspark->rounds++;
            state.dspark->proposed += drafts;
            state.dspark->accepted += accepted;
            DsparkDebugDumpState(state, "verify");
            DsparkAdvance(state, scratch, startPos, startPos + commitCount, scratch.greedy[accepted]);
            segments[0].seqlen = 1;
            return ret;
        }

        for (int i = 0; i < numSegments; i++) {
            if (!active[i]) {
                continue;
            }
            // prefill 分块的中间结果不是真正的下一个 token，这时只更新滑窗、不生成候选
            const int anchor = segments[i].seqlen == 1 && (int)ret.size() > i ? ret[i] : -1;
            DsparkDebugDumpState(*segments[i].state, "plain");
            DsparkAdvance(*segments[i].state, scratches[i], segments[i].startPos,
                          segments[i].startPos + segments[i].seqlen, anchor);
        }
        return ret;
    }

    // ==================== 回滚 / 提交 ====================

    void DeepSeekV41Model::DsparkCommitPrefix(DeepSeekV41RequestState &state,
                                              DeepSeekV41SpecScratch &scratch,
                                              int startPos, int accept, int forwarded) {
        AssertInFastLLM(accept >= 1 && accept <= forwarded,
                        "DeepSeekV41 DSpark: invalid accepted prefix length.");
        const int newLen = startPos + accept;
        for (int layer = 0; layer < block_cnt; layer++) {
            DeepSeekV41LayerCache &cache = state.layers[layer];
            // 1) 滑窗环形缓冲：只写入被接受的行
            if (layer < (int)scratch.windowKV.size() && scratch.windowKV[layer].dims.size() == 3) {
                Data rows;
                if (accept == forwarded) {
                    DsWindowStore(scratch.windowKV[layer], cache.windowKV, startPos, window_size);
                } else {
                    Split(scratch.windowKV[layer], 1, 0, accept, rows);
                    DsWindowStore(rows, cache.windowKV, startPos, window_size);
                }
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
            if (cache.compressedKV.dims.size() == 3 && cache.compressedKV.dims[1] > cache.compressedBlocks) {
                cache.compressedKV.Resize({cache.compressedKV.dims[0], cache.compressedBlocks,
                                           cache.compressedKV.dims[2]});
            }
            if (cache.indexK.dims.size() == 3 && cache.indexK.dims[1] > cache.compressedBlocks) {
                cache.indexK.Resize({cache.indexK.dims[0], cache.compressedBlocks, cache.indexK.dims[2]});
            }
            cache.rawTail = rem;
            if (rem > 0) {
                AssertInFastLLM(scratch.rawKV[layer].dims.size() == 3 &&
                                scratch.rawKV[layer].dims[1] >= total,
                                "DeepSeekV41 DSpark: the compressor rollback buffer is too short.");
                Data tail;
                Split(scratch.rawKV[layer], 1, total - rem, total, tail);
                cache.rawTailKV.CopyFrom(tail);
                if (ratio > 1) {
                    Data tailScore;
                    Split(scratch.rawScore[layer], 1, total - rem, total, tailScore);
                    cache.rawTailScore.CopyFrom(tailScore);
                }
            }
            AssertInFastLLM(cache.compressedBlocks == newLen / ratio,
                            "DeepSeekV41 DSpark: compressed cache rollback is inconsistent at layer " +
                            std::to_string(layer));
        }
        state.totalLen = newLen;
        if (!engram_layer_ids.empty() && (int)state.engramHistory.size() > newLen) {
            state.engramHistory.resize(newLen);
        }
    }

    // ==================== 草稿前向 ====================

    void DeepSeekV41Model::DsparkAdvance(DeepSeekV41RequestState &state, DeepSeekV41SpecScratch &scratch,
                                         int startPos, int committed, int anchorToken) {
        auto dsparkPtr = GetOrCreateDsparkState(state, startPos);
        DeepSeekV41DsparkState &dspark = *dsparkPtr;
        dspark.drafts.clear();
        dspark.confidence.clear();
        dspark.anchor = -1;
        dspark.anchorPos = -1;
        if (dspark.disabled) {
            return;
        }
        const int rows = committed - startPos;
        if (rows <= 0 || (int)scratch.mainHidden.size() != (int)v41DsparkTargetLayerIds.size()) {
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

        // 1) main_x = main_norm(main_proj(cat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39))))
        Data combined, tmp;
        for (size_t k = 0; k < scratch.mainHidden.size(); k++) {
            Data part;
            if (rows == scratch.mainHidden[k].dims[1]) {
                part.CopyFrom(scratch.mainHidden[k]);
            } else {
                Split(scratch.mainHidden[k], 1, 0, rows, part);
            }
            if (k == 0) {
                combined.CopyFrom(part);
            } else {
                Cat(combined, part, -1, tmp);
                combined.CopyFrom(tmp);
            }
        }
        const std::string stage0 = "mtp.0";
        Data mainProj, mainX;
        Linear(combined, weight[stage0 + ".main_proj.weight"], Data(), mainProj);
        DsRMSNormBF16(mainProj, weight[stage0 + ".main_norm.weight"], rms_norm_eps, mainX);
        mainX.Reshape({1, rows, embed_dim});

        // 2) 每个草稿层用自己的 wkv / kv_norm 把 main_x 写进滑窗环形缓冲
        DsRopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                   rope_scaling_beta_fast, rope_scaling_beta_slow};
        for (int stage = 0; stage < v41DsparkLayers; stage++) {
            const std::string pre = "mtp." + std::to_string(stage);
            Data kv;
            Linear(mainX, weight[pre + ".attn.wkv.weight"], Data(), kv);
            DsRMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            kv.Reshape({1, rows, head_dim_full});
            DsRotaryQuant(kv, windowRope, startPos, 1, false, 1, 32);
            DsWindowStore(kv, dspark.layers[stage].windowKV, startPos, window_size);
        }
        dspark.committed = committed;
        dspark.filled = std::min(dspark.filled + rows, window_size);
        const double mainMs = DsElapsed(advanceStart);

        // 3) 生成下一轮的候选
        if (anchorToken < 0 || dspark.committed <= 0) {
            // prefill 分块的中间结果不是真正的下一个 token，这一轮只更新滑窗
            if (DsProfiler().On()) {
                DsProfiler().AddMainOnly(mainMs);
            }
            return;
        }
        std::vector<int> tokens;
        std::vector<float> confidence;
        DsDraftTimingSlot().Reset();
        const double draftStart = DsTick();
        DsparkRunDraft(dspark, anchorToken, tokens, confidence);
        const double draftMs = DsElapsed(draftStart);
        if (DsProfiler().On()) {
            DsProfiler().AddDraft(mainMs + draftMs, mainMs, DsDraftTimingSlot());
        }
        if (tokens.empty()) {
            return;
        }
        dspark.drafts = tokens;
        dspark.confidence = confidence;
        dspark.anchor = anchorToken;
        dspark.anchorPos = dspark.committed;
    }

    void DeepSeekV41Model::DsparkRunDraft(DeepSeekV41DsparkState &dspark, int anchorToken,
                                          std::vector<int> &tokens, std::vector<float> &confidence) {
        tokens.clear();
        confidence.clear();
        const int block = v41DsparkBlockSize;
        const int startPos = dspark.committed;    // 草稿的 RoPE 位置从 committed 开始
        const int valid = std::min(std::min(dspark.filled, window_size), startPos);
        if (block <= 0 || valid <= 0) {
            return;
        }
        DsparkBuildMoeWeights();
        const int dim = embed_dim;
        const int headDim = head_dim_full;
        const float softmaxScale = 1.0f / std::sqrt((float)headDim);
        DsRopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                   rope_scaling_beta_fast, rope_scaling_beta_slow};

        // ---- 输入：[anchor, noise, noise, ...] ----
        std::vector<float> idValues((uint64_t)block, (float)v41DsparkNoiseTokenId);
        idValues[0] = (float)anchorToken;
        Data draftIds(DataType::FLOAT32, {1, block}, idValues);
        Data hiddenStates, hiddenTemp;
        {
            Data embedOut;
            Embedding(draftIds, weight["embed.weight"], embedOut);
            ToDataType(embedOut, DataType::BFLOAT16);
            embedOut.Reshape({1, block, 1, dim});
            Repeat(embedOut, 2, hc_mult, hiddenStates);
        }
        Data *curHidden = &hiddenStates;
        Data *nextHidden = &hiddenTemp;

        Data preMix;
        {
            std::vector<float> values((uint64_t)block * hc_mult, 0.0f);
            for (int i = 0; i < block; i++) {
                values[(uint64_t)i * hc_mult] = 1.0f;
            }
            preMix.CopyFrom(Data(DataType::FLOAT32, {1, block, hc_mult}, values));
        }

        // ---- 稀疏注意力的 gather 表下标 ----
        // 现成的 DeepSeekV41SparseAttention 在片段内是因果的，而 DSpark 的草稿块内部
        // 是全连接（官方 topk_idxs 对每个 query 给出同一张表）。这里把 windowSize 设为 1、
        // startPos 设为 0，让"滑窗"部分只贡献 query 自己那一行，其余位置（环形缓冲里
        // 有效的行 + 块内其它草稿行）通过 compressedKV / cmpIdx 的任意 gather 提供。
        const int tableRows = window_size + block;
        const int width = valid + block - 1;
        std::vector<int32_t> idxValues((uint64_t)block * width, -1);
        for (int i = 0; i < block; i++) {
            int32_t *row = idxValues.data() + (uint64_t)i * width;
            int cursor = 0;
            for (int p = startPos - valid; p < startPos; p++) {
                row[cursor++] = p % window_size;
            }
            for (int j = 0; j < block; j++) {
                if (j != i) {
                    row[cursor++] = window_size + j;
                }
            }
        }
        Data cmpIdxHost(DataType::INT32, {1, block, width});
        cmpIdxHost.Allocate();
        memcpy(cmpIdxHost.cpuData, idxValues.data(), idxValues.size() * sizeof(int32_t));
        Data cmpIdx;
        cmpIdx.CopyFrom(cmpIdxHost);

        Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
        Data x, attnInput, qr, qNorm, q, kv, table, attnOut, woAOut, attnProj;
        Data ffnInput, ffnOut, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        Data confidenceInput;

        const double layersStart = DsTick();
        for (int stage = 0; stage < v41DsparkLayers; stage++) {
            const std::string pre = "mtp." + std::to_string(stage);
            const int layerId = std::max(0, block_cnt - v41DsparkLayers + stage);
            ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);

            DsHcMix(*curHidden, weight[pre + ".hc_attn_fn"], weight[pre + ".hc_attn_scale"],
                    weight[pre + ".hc_attn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                    attnPre, attnPost, attnComb);
            DsHcApplyPre(*curHidden, preMix, x);
            DsRMSNormBF16(x, weight[pre + ".attn_norm.weight"], rms_norm_eps, attnInput);

            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr);
            DsRMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({1, block, num_attention_heads, headDim});
            DsRotaryQuant(q, windowRope, startPos, 1, false, 0, 32);

            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv);
            DsRMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            kv.Reshape({1, block, headDim});
            DsRotaryQuant(kv, windowRope, startPos, 1, false, 1, 32);

            Cat(dspark.layers[stage].windowKV, kv, 1, table);
            AssertInFastLLM(table.dims.size() == 3 && table.dims[1] == tableRows,
                            "DeepSeekV41 DSpark: draft attention gather table has a wrong size.");
            DsSparseAttention(q, kv, nullptr, &table, &cmpIdx, weight[pre + ".attn.attn_sink"],
                              1, 0, softmaxScale, attnOut);
            DsRotaryQuant(attnOut, windowRope, startPos, 1, true, 0, 32);

            DeepSeekV4WoA(attnOut, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnProj);
            DeepSeekV4HcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
            std::swap(curHidden, nextHidden);

            DsHcMix(*curHidden, weight[pre + ".hc_ffn_fn"], weight[pre + ".hc_ffn_scale"],
                    weight[pre + ".hc_ffn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                    ffnPre, ffnPost, ffnComb);
            DsHcApplyPre(*curHidden, attnPre, x);
            DsRMSNormBF16(x, weight[pre + ".ffn_norm.weight"], rms_norm_eps, ffnInput);
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({block, dim});

            // 路由：sqrt(softplus(logits))，bias 只参与选择（草稿层 128 专家 top-3）
            {
                const std::string gpre = pre + ".ffn.gate";
                Data xFloat, logits;
                ToDataType(ffnInput, xFloat, DataType::FLOAT32);
                Linear(xFloat, weight[gpre + ".weight"], Data(), logits);
                ToDataType(logits, DataType::FLOAT32);
                if (std::fabs(gate_temp - 1.0f) > 1e-6f) {
                    Mul(logits, 1.0f / gate_temp, logits);
                }
                Data &gateBias = weight[gpre + ".bias"];
                bool routed = false;
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
                if (!routed) {
                    logits.ToDevice(DataDevice::CPU);
                    gateBias.ToDevice(DataDevice::CPU);
                    const float *raw = (const float*)logits.cpuData;
                    const float *bias = (const float*)gateBias.cpuData;
                    std::vector<int> indices((uint64_t)block * v41DsparkTopk);
                    std::vector<float> scores((uint64_t)block * v41DsparkTopk);
                    std::vector<float> original(v41DsparkExperts), select(v41DsparkExperts);
                    for (int t = 0; t < block; t++) {
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
                    Data idxData(DataType::INT32, {block, v41DsparkTopk});
                    idxData.Allocate();
                    memcpy(idxData.cpuData, indices.data(), indices.size() * sizeof(int));
                    expertIndex.CopyFrom(idxData);
                    expertScore.CopyFrom(Data(DataType::FLOAT32, {block, v41DsparkTopk}, scores));
                }
            }

            {
                std::vector<Data*> moeWeights = v41DsparkMoeWeights[stage];
                Data sharedExpertOut;
                bool hasSharedExpertOut = false;
                auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
                auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
                if (GetCudaSharedExpert() && sharedGateupIt != weight.weight.end() &&
                    sharedDownIt != weight.weight.end() && !sharedGateupIt->second.isDiskWeight &&
                    !sharedDownIt->second.isDiskWeight) {
                    Data ww1, ww3;
                    LinearSwigluBlock(&ffnInput, &sharedGateupIt->second, GetEmptyData(), &ww3, &ww1);
                    Linear(ww1, sharedDownIt->second, *GetEmptyData(), sharedExpertOut);
                    moeWeights[0] = moeWeights[1] = nullptr;
                    hasSharedExpertOut = true;
                }
                this->ApplyMoeDeviceMapForLayer(layerId);
                MergeMOEBlock(&ffnInput, &expertIndex, &expertScore, &moeWeights, &v41DsparkMoeBiass[stage],
                              &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layerId,
                              ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                              MoeGateSwiglu, false, swiglu_limit, true);
                ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
                if (hasSharedExpertOut) {
                    ffnOut.ToDevice(sharedExpertOut.dataDevice);
                    AddTo(ffnOut, sharedExpertOut);
                }
            }
            ffnOut.Reshape(ffnDims);
            DeepSeekV4HcPost(ffnOut, *curHidden, ffnPost, ffnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            preMix.CopyFrom(ffnPre);
        }

        DsDraftTimingSlot().layers = DsElapsed(layersStart);

        // ---- head：mtp.<last>.norm + 共享 lm_head ----
        const double headStart = DsTick();
        const std::string last = "mtp." + std::to_string(v41DsparkLayers - 1);
        Data headHidden;
        DsHcApplyPre(*curHidden, preMix, headHidden);       // confidence head 的输入（未归一化）
        Data normed, logits;
        RMSNorm(headHidden, weight[last + ".norm.weight"], rms_norm_eps, normed);
        Linear(normed, weight["head.weight"], *GetEmptyData(), logits);
        ToDataType(logits, DataType::FLOAT32);

        DsDraftTimingSlot().head = DsElapsed(headStart);

        // ---- markov head：逐位置加 bigram 偏置后贪心采样 ----
        //
        // 这条链天然串行（每一步要用上一步的 token 去查嵌入），用通用算子拼出来的话
        // 每步都得把 token 取回主机，于是每步一次完整的设备同步。实测迷你模型上 5 步
        // 就要 2.6 ms，比整个目标模型的一次前向还贵，而 DSpark 在真实模型上的总收益
        // 也才几毫秒。所以 CUDA 上走一个把整条链留在设备上的融合 kernel，
        // 主机只在最后取回一次；其它设备（或 dtype / 布局不满足前提时）退回下面的通用实现。
        const double markovStart = DsTick();
        Data &markovEmbedWeight = weight[last + ".markov_head.embed.weight"];
        Data &markovHeadWeight = weight[last + ".markov_head.head.weight"];
        const int markovRank = markovEmbedWeight.dims.size() == 2 ? markovEmbedWeight.dims[1]
                                                                  : v41DsparkMarkovRank;
        Data markovAll;                 // [1, block, rank]，confidence head 的输入
        bool fusedMarkov = false;
#ifdef USE_CUDA
        if (!DsEnvFlag("FASTLLM_DSPARK_DISABLE_FUSED_MARKOV") && logits.dataDevice == DataDevice::CUDA) {
            markovEmbedWeight.ToDevice(DataDevice::CUDA);
            markovHeadWeight.ToDevice(DataDevice::CUDA);
            fusedMarkov = FastllmCudaDeepSeekV41MarkovChain(logits, markovEmbedWeight, markovHeadWeight,
                                                            anchorToken, block, tokens, markovAll);
            DsDraftTimingSlot().markovFused = fusedMarkov;
        }
#endif
        if (!fusedMarkov) {
            std::vector<Data> markovEmbeds(block);
            int previous = anchorToken;
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
                DsDraftTimingSlot().markovBias += DsElapsed(biasStart);
                const double argmaxStart = DsTick();
                std::vector<int> best = DsArgmaxRows(stepLogits);
                DsDraftTimingSlot().markovArgmax += DsElapsed(argmaxStart);
                AssertInFastLLM(best.size() == 1, "DeepSeekV41 DSpark: draft sampling failed.");
                previous = best[0];
                tokens.push_back(previous);
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
        confidence.assign(block, 1.0f);
        auto confIt = weight.weight.find(last + ".confidence_head.proj.weight");
        if (confIt != weight.weight.end() && v41DsparkConfidenceThreshold > 0.0f &&
            markovAll.dims.size() == 3) {
            Data hiddenFloat, markovFloat, features, confLogits;
            ToDataType(headHidden, hiddenFloat, DataType::FLOAT32);
            ToDataType(markovAll, markovFloat, DataType::FLOAT32);
            Cat(hiddenFloat, markovFloat, -1, features);
            Linear(features, confIt->second, Data(), confLogits);
            ToDataType(confLogits, DataType::FLOAT32);
            confLogits.ToDevice(DataDevice::CPU);
            const float *values = (const float*)confLogits.cpuData;
            for (int i = 0; i < block; i++) {
                float p = DsSigmoid(values[i]);
                confidence[i] = std::isfinite(p) ? p : 1.0f;
            }
        }
        DsDraftTimingSlot().conf = DsElapsed(confStart);
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
