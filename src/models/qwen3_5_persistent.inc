// Human-readable output remains plain text.  The terminal colorizer consumes
// structured [fastllm-event] lines on a separate channel, so tee/server.log
// never receives ANSI escape sequences.

// Included inside namespace fastllm after the native snapshot helpers. This
// adapter deliberately shares those helpers and the existing paged KV writers.
static int Qwen35ReadyPagedPrefix(Qwen3_5Model *model, ResponseContext *ctx) {
    if (!ctx || ctx->currentTokens.empty() || !ctx->CanUsePagedPrefixCache()) return 0;
    int pages = INT_MAX;
    for (int layer = 0; layer < model->block_cnt; ++layer)
        for (bool key : {true,false}) for (auto ref : model->GetPagedKVCacheManagers(layer,key)) {
            if (!ref.second) return 0;
            std::vector<int> ids;
            ref.second->Query(ctx->currentTokens,ids,ctx->PrefixCachePageKeys());
            pages = std::min(pages,(int)ids.size());
        }
    const int maximum = ((int)ctx->currentTokens.size()-1)/fastllm::GetPageLen()*fastllm::GetPageLen();
    return pages == INT_MAX || maximum <= 0 ? 0 :
        model->QueryPagedPrefixCacheExtra(ctx,std::min(maximum,pages*fastllm::GetPageLen()));
}
#if defined(FASTLLM_DISK_PREFIX_CACHE) && defined(USE_CUDA)
// 0925 适配：本 .inc 用到 POSIX struct stat / stat()，而 qwen3_5.cpp 侧只包含
// disk_prefix_cache.h（该头不含 <sys/stat.h>），故在此显式引入。
#include <sys/stat.h>
class Qwen35PersistentCache {
    using Json = json11::Json;
    using Clock = std::chrono::steady_clock;
    // 可用主机内存（/proc/meminfo 的 MemAvailable）；读不到返回 0（调用方必须保守处理）。
    // 用 C stdio 而不是 <fstream>：本文件在**类体内部**被包含，不能在这里引入新的标准头。
    static size_t HostAvailableBytes() {
        FILE *file = std::fopen("/proc/meminfo", "r");
        if (file == nullptr) return 0;
        char line[256];
        uint64_t availableKb = 0;
        while (std::fgets(line, sizeof(line), file) != nullptr) {
            if (std::strncmp(line, "MemAvailable:", 13) == 0) {
                availableKb = std::strtoull(line + 13, nullptr, 10);
                break;
            }
        }
        std::fclose(file);
        return size_t(availableKb) << 10;   // kB -> B
    }
    // 主机侧内存预算。
    //   显式设置 FASTLLM_PREFIX_CACHE_HOST_LIMIT_MB ⇒ **精确使用该值、不扩张**（这样才能故意压小额度过载测试）。
    //   未设置 ⇒ 启动时按"可用内存 − 安全余量"取值：下界 2048 MiB；
    //            `FASTLLM_PREFIX_CACHE_HOST_LIMIT_MAX_MB` 默认 8192 MiB，但它**只夹住 base**（`.inc:76`），
    //            **没有夹住 usableCeiling**。
    //   原文写"夹在 [2048 MiB, 上限] 之间；上限默认 8192 MiB"，与实现不符：
    //     当 GPU 侧 KV 池 > base 时走 `.inc:86-88` 的 gpuFloor 分支，取 min(gpuFloor, usableCeiling)，
    //     其中 usableCeiling = max(MemAvailable − 4096 MiB, 2048 MiB)（`.inc:70-74`，**未被 8192 夹过**）
    //     ⇒ 实测 `prefix_cache_host_limit_mb=-1`（自动模式）时 watermark 打印 host_limit = 26,998,734,848 B
    //     = **25,748 MiB（25.14 GiB）**（出处 `build-rewrite/ssd-throughput-v4/user-test-256k-a1/server.log`），
    //     另一份 manual16k 为 25,516 MiB。**显式设 HOST_LIMIT_MB 才是精确值**；自动模式下 8192 不是硬上限。
    // 为什么在**启动时**取而不是每次准入动态改：协调器的额度是创建时一次性建立的，运行中改它要动
    // 并发关键路径（票在飞），风险高；启动时取值即可拿到"内存够就多给"的全部收益。
    // 为什么必须有硬上限与安全余量：**真正的硬上限是 usableCeiling**（MemAvailable − 4096 MiB 的安全余量，
    //   `.inc:70-74`）；没有它会让整个服务被系统 OOM 杀掉——那比丢一段缓存严重得多。
    //   （注意：`HOST_LIMIT_MAX_MB` 只夹 base、**不夹 ceiling**，见上方 2026-10-05 R11 更正。）
    // A-2（2026-10-04，本次改动；业界对齐）：内存中转区（host tier）**不得小于 GPU 侧 KV 池**。
    // 业界一手依据（本次实抓正文）：NVIDIA Dynamo KVBM 配置参考原文警告——
    //   "Set the CPU cache larger than the engine's GPU KV cache — a good rule of thumb is
    //    DYN_KVBM_CPU_CACHE_GB >= 100 and DYN_KVBM_DISK_CACHE_GB >= DYN_KVBM_CPU_CACHE_GB.
    //    **If the CPU cache is smaller than the device cache, KVBM churns (offloading blocks after
    //    every forward pass) and you may see performance degradation.**"
    // vLLM 同义（docs/features/kv_offloading_usage.md）："a smaller CPU tier just **mirrors what the
    // GPU already holds and adds no hit rate**"。
    // 我方实测佐证：T2 观测 host_limit=2 GiB，而 pin 预算默认 = 池/10（GpuPinLimitBytes）⇒ GPU 池
    // 约 2–2.5 GiB ⇒ **host < 池** ⇒ 结构性 churn，并伴随 slot_waits/acquires=94.8%、
    // credit_peak=1,456,226,304 逼近上限。
    // 实现要点：GPU 池总量只作为 host 限额的**下限**，且**绝不越过安全包络**（可用内存 − 安全余量），
    // 因为 OOM 掉整个服务远比少一段缓存严重；显式设置 FASTLLM_PREFIX_CACHE_HOST_LIMIT_MB 时
    // **仍精确使用、不加下限**（T1 压缩短跑依赖"故意压小额度"，语义不能变）。
    static std::atomic<size_t> &GpuPoolFloorRef() {
        static std::atomic<size_t> value{0};
        return value;
    }
    struct HostLimitBase { size_t base; size_t usableCeiling; };
    static const HostLimitBase &HostLimitBaseRef() {
        static const HostLimitBase value = [] {
            const size_t floorBytes = size_t(2048) << 20;
            const char *maxText = std::getenv("FASTLLM_PREFIX_CACHE_HOST_LIMIT_MAX_MB");
            const int maxMb = maxText != nullptr ? std::atoi(maxText) : 0;
            const size_t maxBytes = size_t(maxMb > 0 ? maxMb : 8192) << 20;
            const size_t reserveBytes = size_t(4096) << 20;           // 留 4 GiB 给模型/系统/其他人
            const size_t available = HostAvailableBytes();
            if (available == 0) return HostLimitBase{floorBytes, floorBytes};   // 读不到 ⇒ 保守用下限
            const size_t usable = available > reserveBytes ? available - reserveBytes : 0;
            const size_t ceiling = usable < floorBytes ? floorBytes : usable;
            size_t base = usable < floorBytes ? floorBytes : usable;
            if (base > maxBytes) base = maxBytes;
            return HostLimitBase{base, ceiling};
        }();
        return value;
    }
    static size_t HostLimit() {
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_HOST_LIMIT_MB");
        const int configuredMb = text != nullptr ? std::atoi(text) : 0;
        if (configuredMb > 0) return size_t(configuredMb) << 20;   // 显式指定：精确使用（不加 GPU 池下限）
        const HostLimitBase &b = HostLimitBaseRef();
        const size_t gpuFloor = GpuPoolFloorRef().load(std::memory_order_relaxed);
        if (gpuFloor <= b.base) return b.base;
        return gpuFloor < b.usableCeiling ? gpuFloor : b.usableCeiling;
    }
    // Default source profile: checkpoint every 256 tokens; explicit positive
    // FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL overrides it at startup.
    static int CheckpointInterval() {
        static const int value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_CHECKPOINT_INTERVAL");
            const int parsed = text != nullptr ? std::atoi(text) : 0;
            return parsed > 0 ? parsed : 256;
        }();
        return value;
    }
    // Full-snapshot budget headroom: 8; this does not enforce a queue/thread limit.
    static int CheckpointJobsLimit() {
        static const int value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_CHECKPOINT_JOBS");
            const int parsed = text != nullptr ? std::atoi(text) : 0;
            return parsed > 0 ? parsed : 8;
        }();
        return value;
    }
    // A-2 finite resource contract. These are contract values, not tunable
    // defaults: queue-full and retention failures must be explicit terminals.
    static constexpr size_t kMaxInflightJobs = 16;
    static constexpr size_t kMaxSnapshotQueue = 16;
    static constexpr size_t kMaxFailedRetained = 16;
    static constexpr size_t kMaxFailedBytes = size_t(256) << 20;
    static constexpr int kDrainTimeoutMs = 30000;
    static size_t MaxPendingCopy(int workers) {
        return size_t(4) * size_t(std::max(1, std::min(16, workers)));
    }
    // W1（2026-10-31）：发布端写入节点的 pageHash 是否等于"内容链哈希"。
    // 仅当 onRecordPage/onRecyclePage 注册时才等于（fastllm.cpp:6727-6733），而这两个回调
    // 由 Qwen35L2HostKVEnabled() 决定（qwen3_5_spill.inc:894-895，环境变量
    // FASTLLM_PREFIX_CACHE_L2_HOST_KV）。未启用时节点里是另一个 h，不能拿来判"过期"。
    static bool PublisherPageHashIsChain() {
        static const bool value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_L2_HOST_KV");
            return text != nullptr && text[0] != '0';
        }();
        return value;
    }
    static std::string PageIdentityHex(uint64_t value) {
        char text[17];
        std::snprintf(text, sizeof(text), "%016llx", (unsigned long long)value);
        return std::string(text);
    }
    static std::vector<std::string> ComputePageIdentities(const std::vector<int> &tokens, int length, int pageLen,
                                                           const std::vector<std::string> *pageKeys = nullptr) {
        Check(length >= 0 && pageLen > 0 && length <= (int)tokens.size() && length % pageLen == 0,
              "ssd_page_identity_shape");
        const size_t pages = size_t(length / pageLen);
        Check(pageKeys == nullptr || pageKeys->size() >= pages, "ssd_page_identity_keys");
        std::vector<std::string> identities;
        identities.reserve(pages);
        uint64_t chain = 0;
        for (size_t p = 0; p < pages; ++p) {
            const std::string extra = pageKeys != nullptr ? (*pageKeys)[p] : std::string();
            chain = fastllm::PagedCacheManager::HashTokenPrefixPage(
                chain, tokens.data() + p * size_t(pageLen), pageLen, extra);
            identities.push_back(PageIdentityHex(chain));
        }
        return identities;
    }
    // A-6 fix E：FASTLLM_TEST_IMPORT_IDENTITY_MISMATCH_AT=N 让第 N 次 SSD Import
    // 在内存里看到被改过一位的 page_identity[0]，从而走真实的 import-identity-mismatch 分支（G8 反例）。
    // 不设变量时 target=0：不计数、不复制、不打印；任何情况下都不写盘。
    static uint64_t TestImportIdentityMismatchOrdinal() {
        static const uint64_t target = [] {
            const char *text = std::getenv("FASTLLM_TEST_IMPORT_IDENTITY_MISMATCH_AT");
            if (text == nullptr || *text == '\0') return uint64_t(0);
            char *end = nullptr;
            const unsigned long long parsed = std::strtoull(text, &end, 10);
            return end != text && *end == '\0' && parsed > 0 ? uint64_t(parsed) : uint64_t(0);
        }();
        static std::atomic<uint64_t> imports{0};
        if (target == 0) return 0;
        const uint64_t ordinal = imports.fetch_add(1, std::memory_order_acq_rel) + 1;
        return ordinal == target ? ordinal : 0;
    }
    // A-6 fix F part 2: the last Import failure text on this thread (the restore worker clears it
    // before each Import and reads it right after).
    static std::string &LastImportFailureReason() {
        thread_local std::string reason;
        return reason;
    }
    static constexpr size_t BufferBytes = size_t(8) << 20;
    // S1-5 修复（2026-10-02）：D2 流暂存预算（字节）。原为硬编码 64 MiB
    // （kSlotBytes×kPinnedSlots×2）；长上下文单个检查点的分块字节远超该窗口，而提交阶段不
    // PumpCopy ⇒ 预算一满就被拒（真机 r1–r7 每轮 2–8 次 reason=ssd_d2_submit）。
    // 默认值：256 MiB → **1024 MiB**；可用
    // FASTLLM_PREFIX_CACHE_D2_STAGING_MB 覆盖；上限仍受内存约束。
    // R1 注意：本预算是**每个写 lane 各一份**（每个 D2Stream 一个），W 个 lane 的账面上限
    // 是 W×该值；它是"拒绝而不是排队"的窗口，不是预分配内存，实际占用 ≈ 在途分块字节。
    static size_t D2StagingBudgetBytes() {
        static const size_t value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_D2_STAGING_MB");
            const long long parsed = text != nullptr ? std::atoll(text) : 0;
            return parsed > 0 ? (size_t)parsed << 20 : (size_t(1024) << 20);
        }();
        return value;
    }
    // Request paths never wait for writer/pin budget. The existing worker
    // reserves bounded copy windows and resumes when another window completes.
    // 【最快可用·核心】读/写共用的「上下文长度门槛」。
    // 依据（本轮实测，两种口径交叉验证）：命中量 < ~1.1–1.6 万 token 时，省下的重算时间 **小于**
    // 恢复的固定开销 ⇒ 短/中文本开缓存反而更慢（实测 H−M：4K **+0.68~1.12s**、6K **+0.42~0.74s**、
    // 14K +0.24~−0.08s；只有 60K **−8.12~−8.57s**）。最小二乘平衡点 11,256/10,818 tok，
    // 相邻插值 15,629/12,995 tok ⇒ 取上界 **16384** 作保守阈值（**仅在显式设置该 env 时生效**，见下方"默认关闭"）。
    // 做法（读侧，Prepare）：长度不够 ⇒ **根本不查、不恢复** ⇒ 该请求与"不开缓存"完全同路径
    // ⇒「开了不比不开慢」**由构造保证成立**，不依赖任何调试或根因定位。
    // 业界同构：LMCache `min_retrieve_tokens` 原文——"如果需要加载的 token 数低于此值，LMCache 将
    // 跳过检索，并让推理引擎重新计算它们"（https://docs.lmcache.ai/zh_CN/production/performance_tuning.html ）。
    // 做法（写侧，同源）：短请求既不用缓存也不写缓存（写了也永远不可能被**有利地**读回，纯增写盘量）。
    // **默认值 = 0 = 关闭**（实现见 `.inc:187` `if (text == nullptr) return 0;`）：
    //   不设该 env、或设成 0／负数 ⇒ **不存在门槛**，读写都照常；上文"长度不够 ⇒ 根本不查、不恢复"与
    //   "短请求既不用缓存也不写缓存"**必须显式设 >0 才成立**。
    // 原文写"一键回退：`FASTLLM_PREFIX_CACHE_MIN_CONTEXT_TOKENS=0` ⇒ 回到旧行为"，
    //   措辞**暗示默认是开启的**，与实现（默认 0＝关）相反。
    static int MinCacheContextTokens() {
        static const int value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_MIN_CONTEXT_TOKENS");
            if (text == nullptr) return 0;              // 默认关闭门槛：短请求照常读写缓存
            const int parsed = std::atoi(text);
            return parsed > 0 ? parsed : 0;             // 显式 >0 才开门槛；0 = 关
        }();
        return value;
    }
    // A-1（2026-10-04，本次改动；业界对齐的"只记值得记的"＝写盘门槛）。
    // 业界一手依据（本次实抓正文）：
    //   · vLLM `store_threshold` = "**Min lookups before a block is offloaded**"（默认 0；>0 即门槛），
    //     以及每请求 `max_offload_tokens`（"only the first N tokens of the request are offloaded …
    //     useful when a known prefix is worth caching but later request-specific tokens are not"）。
    //   · SGLang `--hicache-write-policy write_through_selective` = "uses **hit-count tracking** to back
    //     up only frequently accessed data"。
    //   · Dynamo KVBM `offload.g2_to_g3` 策略 `pass_all` / `presence` / `presence_lfu`
    //     （`presence_lfu.min_lfu_count` 默认 **8**）。
    // 我方额外依据（实测）：读盘 ≪ 重算**只在长上下文成立**——AB 四点拟合盈亏平衡 ≈10,800~15,600
    // token，短请求写盘纯亏（这正是 A2 短请求负收益的来源之一）。
    // **与读侧同源**（同一个 `MinCacheContextTokens()`）：**当前默认 0 ＝门槛关闭**（2026-10-04
    // 历史设计阈值取 16384，见上文），短请求照常读写缓存；显式设 >0 才开门槛。
    // 纪律：**刻意不写**不等于丢点 ⇒ 打印 `write_gated`（**不是** `write_skipped`）并单独计数，
    //       以免污染判据①（zero write_skipped）与合同"degraded=0"。
    static int WriteMinContextTokens() { return MinCacheContextTokens(); }
    static int WriteMinReuse() {
        static const int value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_WRITE_MIN_REUSE");
            const int parsed = text != nullptr ? std::atoi(text) : 0;
            return parsed > 1 ? parsed : 0;   // 0/1 = 关闭（每次出现都写）
        }();
        return value;
    }
    // 定位用阶段打卡：默认关闭；FASTLLM_PREFIX_CACHE_CAPTURE_TRACE=1 打开。
    // 纯观测：只在 stderr 新增"[PrefixCache][trace] stage=… t_ms=…"行并强制刷新，不改任何逻辑。
    // 用途：真机重启重放后 Capture 内出现约 195 s 无日志悬停（服务进程 394% CPU、GPU 空闲），
    // 悬停时**最后一条 stage 行**即出问题的工段（无需专业 profiler，本机无 root 无法用 perf）。
    static bool CaptureTraceEnabled() {
        if (!PrefixCacheDebugEnabled()) return false;
        static const bool value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_CAPTURE_TRACE");
            return text != nullptr && text[0] != '\0' && text[0] != '0';
        }();
        return value;
    }
    // 定位用：Ensure() 各步时间戳。默认关闭；
    // FASTLLM_PREFIX_CACHE_SELFCHECK_TRACE=1 打开。纯观测：只新增 stderr 行并 flush。
    static bool SelfCheckTraceEnabled() {
        if (!PrefixCacheDebugEnabled()) return false;
        static const bool value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_SELFCHECK_TRACE");
            return text != nullptr && text[0] != '\0' && text[0] != '0';
        }();
        return value;
    }
    struct ExportState {
        std::unique_ptr<DiskPrefixCache::WriteSession> writer;
        std::unique_ptr<DiskPrefixCache::ReadSession> inherited;
        Json::array kv;
        int issued = 0;
        // Incremental Capture carries the rolling parent for the next new page.
        uint64_t issuedChainHash = 0;
        size_t issuedChainPages = 0;
        bool issuedChainValid = false;
        // R4 彻底版（2026-10-02）：把"状态属于哪一次请求"显式记下来。
        // 上一次修复用的是启发式（issued > length 才重建），残余局限是"新请求首个检查点边界
        // 大于旧 issued 时仍会误用旧状态、把开头几个检查点跳过"。现在用请求的 prompt 长度
        // （ctx->inputTokens）作为请求身份：不同请求 ⇒ 长度不同 ⇒ 必重建；同一请求重复出现
        // （长度相同）则继续沿用，符合预期。
        int requestTokens = -1;
        uint64_t request_id = 0, request_generation = 0;
        uint64_t reservation = 0;
        std::atomic<bool> failed{false};
        std::string identity, family, provenance;
        Json source;
    };
    struct Pages {
        PagedCacheManager *pool = nullptr;
        int layer = 0, side = 0, rank = 0, begin = 0;
        size_t pageBytes = 0;
        // Own complete pages until D2H finishes, independently of request
        // lifetime. Capture is also called by Restored without a following Wait.
        std::vector<int> ids;
        std::vector<uint64_t> generations;
        // A-3/S7：内容身份（前缀哈希链）。0 表示该页当时没有 trie 节点，回落到时间戳校验。
        std::vector<uint64_t> hashes;
        std::vector<std::pair<int,int>> ranges;
        void *ready = nullptr;
        // S4-C（2026-10-04）：本 lease 中**已经被 Pick 保护过**的前缀页数（keepalive 引用）。
        // 方案 C：Capture 对整段导出范围 Pick 一次（自己持一份引用，补上"请求提前结束、
        // 页被复用"的窗口——与 vLLM 引用计数语义一致）；因此 pickedPages = 整段页数。
        // 析构/异常路径只归还 [0,pickedPages) 里仍未拷贝（id>=0）的页：未 Pick 的页从未
        // 加过引用，归还会造成引用计数下溢。
        int pickedPages = 0;
        std::vector<uint8_t> reused;
        // S4-C：已计入 pinnedGpuBytes、尚未被 D2CopyIntoSlot 扣回的页数。
        // 保护是整段的，但**预算记账只按"当前窗口"**（拷贝并发/预算是窗口粒度），
        // 因此这里与 pickedPages 分离，Work() 结束只能按 accountedPages 扣账。
        int accountedPages = 0;
        ~Pages() {
            if (ready) {
                const int previous = FastllmCudaGetDevice();
                FastllmCudaSetDevice(rank); FastllmCudaEventDestroy(ready);
                FastllmCudaSetDevice(previous);
            }
            if (pool) {
                std::vector<int> held;
                const int limit = std::min<int>(pickedPages,(int)ids.size());
                for (int i = 0; i < limit; ++i) if (ids[i] >= 0) held.push_back(ids[i]);
                if (!held.empty()) pool->ReleasePageIndices(held);
            }
        }
    };
    struct RequestLeaseState {
        uint64_t request_id = 0;
        std::atomic<uint64_t> generation{0};
        std::atomic<bool> cancelled{false};
    };
    struct Job {
        RequestDiagnosticToken diagnosticToken;
        // Complete immutable request/cache identity carried by every async Job.
        uint64_t request_id = 0, request_generation = 0;
        uint64_t page_generation = 0, copy_generation = 0;
        std::string cache_identity;
        std::shared_ptr<RequestLeaseState> request_state;
        std::shared_ptr<const void> owner_lease;
        bool identity_terminal = false;
        // P3 模型 adapter：该检查点在协调器里的票据（准入成功才有意义）。
        fastllm::prefix_write::TicketId ticket;
        bool admitted = false;
        std::vector<std::shared_ptr<Qwen35PrefixWriteStatus>> writeStatuses;
        std::shared_ptr<ExportState> state;
        std::vector<std::shared_ptr<Pages>> pages;
        std::shared_ptr<Qwen35LinearPrefixSnapshot> snapshot;
        std::shared_ptr<fastllm::qwen35_state_stage::Ticket> stateStage;
        std::map<int, size_t> stateStageBytes;
        uint64_t stateStageD2DEnqueueNs = 0;
        uint64_t stateStageReadyWaitNs = 0;
        uint64_t stateStageMaterializeNs = 0;
        uint64_t stateStageFallbackSyncCount = 0;
        std::string stateStageFallbackReason;
        bool runtimeUsesPageKeys = false;
        std::string prefix, key;
        // Runtime page-ledger checkpoint ownership for the v4 durable root.
        // The store callback completes this root only after the v4 commit
        // reaches its durable terminal state.
        std::string pagedCheckpointIdentity;
        std::vector<PrefixCachePageIdentity> pagedCheckpointPages;
        uint64_t pagedCheckpointGeneration = 0;
        Json prefetchContext;
        Json adoptContext;
        int adoptLength = 0;
        int length = 0;
        // V5 (FASTLLM_PREFIX_V5=1): last checkpoint of this prompt; its state goes to a ping-pong slot.
        bool promptEnd = false;
        std::string hintWallNs;
        bool hintOnly = false;
        // S4：本笔导出的窗口页数（一次 Pick 的页数上界）。0 = 未窗口化（整段一次）。
        int windowPages = 0;
        // Capture retains sources; all window budget is reserved in Work.
        bool deferredPin = false;
        size_t stateBytes = 0;
        std::shared_ptr<std::vector<float>> image;
        // R1/A2：入队时刻。end_to_end_ms = 作业完成墙钟 − 本时刻（含排队 + 服务）。
        Clock::time_point enqueuedAt;
    };
    Qwen3_5Model *model;
    std::string directory, baseFamily, encoderIdentity, configuredIdentity, draftIdentity, policy, startupManifest;
    int restoreTimeoutMs = 100000;
    uint64_t diskLimit = uint64_t(256) << 30;
    std::shared_ptr<DiskPrefixCache> store;
    std::shared_ptr<DiskPrefixCache> imageStore;
    std::string identity, family;
    std::map<ResponseContext*, std::shared_ptr<ExportState>> exports;
    std::map<ResponseContext*, std::shared_ptr<RequestLeaseState>> requestStates;
    // Request hold (default on with V5; FASTLLM_PREFIX_V5_REQUEST_HOLD=0 disables): a request keeps the V5 pin of the checkpoint it restored
    // until the request ends, so eviction/D8 cannot cut the chain its saves continue (SGLang
    // 8920463c90 radix_cache.py:587-614 inc_lock_ref/dec_lock_ref; stab3 s2: 15 v5_parent_missing).
    std::map<ResponseContext*, std::shared_ptr<DiskPrefixCache::ReadSession>> requestHolds;
    static bool RequestHoldEnabled() {
        static const bool value = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_V5_REQUEST_HOLD");
            return fastllm::DiskPrefixV5Enabled() && (text == nullptr || std::string(text) != "0");
        }();
        return value;
    }
    std::map<std::string,Json> provenanceSources;
    mutable std::mutex mutex;
    std::condition_variable changed;
    // B4-correct：pin 等待诊断计数（只追加计数，不改既有结构/接口）
    std::atomic<uint64_t> pinDeferWaits{0}, pinDeferTimeouts{0}, pinDeferMsTotal{0};
    // A-0（2026-10-04）：因"请求线程不等待"而改为交给写 worker 有界等待的检查点数（本笔照常入队、未在请求线程丢弃）。
    std::atomic<uint64_t> pinDeferredJobs{0};
    std::atomic<uint64_t> writeFailures{0};
    std::atomic<uint64_t> coalescedCheckpoints{0};
    // A-1（2026-10-04）：写盘门槛（"只记值得记的"）的状态。
    //   writeReuseSeen：key -> 本进程内"被尝试写入"的次数（mutex 保护；上限 4096 条后整体清零，
    //                   避免长跑无限增长；清零点只影响门槛的保守性，不影响正确性）。
    //   writeGated   ：因**刻意不写**而跳过的检查点数（**不计入丢点**，另打 write_gated 行）。
    std::map<std::string,int> writeReuseSeen;
    std::atomic<uint64_t> writeGated{0};
    // 【最快可用】读侧被长度门槛拦下（不查不恢复）的请求数。刻意不计入任何失败计数。
    std::atomic<uint64_t> retrieveGated{0};
    // R1（2026-10-02）写路径并发化：**每个 worker 一条 FIFO**。同一 ExportState（同一请求）
    // 的检查点按地址亲和固定到同一条 lane ⇒ 同请求串行（writer/kv 累积与提交顺序不变），
    // 不同请求分布到不同 lane 并行；copy windows retain the existing byte ledger.
    std::vector<std::deque<Job>> laneQueues;
    std::set<std::string> pendingCheckpointKeys;
    // Failed real I/O remains visible and keeps its snapshot/page/writer owners.
    std::vector<Job> failedJobs;
    std::vector<std::thread> workers;
    int writeWorkerCount = 1;
    size_t queuedJobs = 0;          // 原 queue.size()：全局排队作业数（mutex 保护）
    int activeWorkers = 0;          // 原 working：正在 Export 的 worker 数（mutex 保护）
    // SGLang cache_controller.py:1260-1274 / prefetch queues: reads have
    // their own worker; the request path never calls the storage backend.
    std::mutex prefetchMutex;
    std::mutex prefetchHintMutex;
    std::condition_variable prefetchChanged;
    std::thread prefetchWorker;
    // Reuse the existing read worker. Tickets never contain a live context
    // pointer, so abort/timeout cannot race background payload reads.
    struct ReadTicket {
        RequestDiagnosticToken diagnosticToken;
        uint64_t request_id = 0, request_generation = 0;
        uint64_t page_generation = 0, copy_generation = 0;
        std::string cache_identity;
        std::shared_ptr<RequestLeaseState> request_state;
        std::shared_ptr<const void> owner_lease;
        DiskPrefixCache::Checkpoint candidate;
        Clock::time_point started = Clock::now(), lastProgress = started;
        Clock::time_point deadline;
        std::unique_ptr<DiskPrefixCache::ReadSession> reader;
        Json state;
        std::string error;
        uint64_t id = 0, loadedBytes = 0;
        std::atomic<uint64_t> verifiedBytes{0};
        std::atomic<uint64_t> discoveredBytes{0};
        double readMs = 0;
        bool ok = false;
        std::atomic<bool> done{false}, cancelled{false}, consumed{false};
    };
    std::deque<std::shared_ptr<ReadTicket>> requestReads;
    std::map<ResponseContext *,std::shared_ptr<ReadTicket>> restoreTickets;
    // Shared by requests, background imports and the existing durable callback.
    // This stores metadata only. A tier hit still needs a fresh physical probe;
    // bookkeeping must never keep a GPU page or an SSD root alive by accident.
    struct PrefixLedger {
        std::mutex mutex;
        std::map<std::string, PrefixCacheRecord> records;
        PrefixCacheRecord Observe(const PrefixCacheRecord &record) {
            if (record.key.empty()) return record;
            std::lock_guard<std::mutex> lock(mutex);
            auto found = records.find(record.identity+":"+record.key);
            if (found == records.end()) {
                records.emplace(record.identity+":"+record.key,record);
            } else {
                PrefixCacheRecord &current = found->second;
                current.length = record.length;
                current.provenance = record.provenance;
                current.requiredComponents = record.requiredComponents;
                // Residency is a fresh observation, not an OR of every tier
                // ever seen. Evicted pages must not remain labelled ready.
                current.gpuComponents = record.gpuComponents;
                current.hostComponents = record.hostComponents;
                // The SSD catalog is queried on demand. An earlier COMMIT is
                // not evidence that a root survived a later quota eviction.
                current.ssdComponents = record.ssdComponents;
                current.runtimeTier = record.runtimeTier;
                current.runtimeState = record.runtimeState;
                current.durableState = record.durableState;
                current.runtimeLease = record.runtimeLease;
                current.loadingState = record.loadingState;
            }
            const PrefixCacheRecord resolved = records.at(record.identity+":"+record.key);
            if (records.size() > 4096) records.erase(records.begin());
            return resolved;
        }
        void Durable(const PrefixCacheRecord &record, bool durable) {
            std::lock_guard<std::mutex> lock(mutex);
            const std::string key = record.identity+":"+record.key;
            auto found = records.find(key);
            if (found == records.end()) found = records.emplace(key,record).first;
            if (durable) {
                found->second.ssdComponents = record.requiredComponents;
                found->second.durableState = PrefixCacheRecord::DurableReady;
            } else {
                found->second.durableState = PrefixCacheRecord::DurableFailed;
            }
            if (found->second.loadingState == PrefixCacheRecord::Unknown)
                found->second.loadingState = durable ? PrefixCacheRecord::Ready : PrefixCacheRecord::Failed;
            if (records.size() > 4096) records.erase(records.begin());
        }
    };
    std::shared_ptr<PrefixLedger> prefixLedger = std::make_shared<PrefixLedger>();
    // Startup/background hints remain a separate queue.  The same worker
    // services both queues so the existing startup warmup path is preserved
    // without adding another SSD reader or writer.
    std::deque<Json> prefetchQueue;
    std::set<std::string> prefetchPending;
    uint64_t nextReadId = 0;
    bool prefetchStopping = false, prefetchReady = false;
    std::atomic<uint64_t> prefetchQueued{0}, prefetchCompleted{0}, prefetchMisses{0};
    size_t nextLane = 0;            // 无 state 的作业（图片）用的轮转 lane（mutex 保护）
    bool stopping = false;
    size_t queuedStateBytes = 0;
    int checkpointJobs = 0;
    uint64_t captureTicketId = 0;
    // §13 第 2 条：关闭准入后不再接收新检查点，但在途作业照常排空。
    std::atomic<bool> admissionClosed{false};
    // P3 模型 adapter：请求线程只做"身份描述 + 去重 + 预算预约"，不等待 I/O。
    // 目前先建立实例；把准入判断切到 TryAdmit 的接线按计划下一步进行。
    std::unique_ptr<fastllm::prefix_write::Coordinator> coordinator;
    // §④ 每个待拷贝分块的上下文（按 ticket.sequence 索引，保证交付顺序与提交顺序一致）。
    struct D2Chunk {
        uint64_t copy_generation = 0;
        bool pending_copy_counted = false, copy_terminal = false;
        int rank = 0;
        int token = 0;
        int count = 0;
        size_t bytes = 0;
        Json segments;
        std::vector<std::shared_ptr<Pages>> leases;
    };
    // R1（2026-10-02）写路径并发化：**每个写 worker 独占一套** D2/哈希流水线与统计。
    //
    // 为什么必须"每 lane 一套"，而不是 N 个 worker 共用一个 D2Stream / HashPackPool：
    //   · HashPackPool::TakeOrderedChunk/Batch 与 D2Stream::TakeOrderedChunk 都是
    //     **单消费者 + 全局提交序**语义——取回的是"全局提交序的队首"，不校验票据归属。
    //     两个 worker 共用一个池时，A 会取到 B 提交的分块，Export 里的
    //     `ssd_d2_order`（ticket.requestId/sequence 必须与本 lane 一致）立刻失败。
    //   · PinnedSlotPool 的槽号只是内存里的下标；两个 D2Stream 共用同一块 pinned 内存
    //     （d2SlotMemory）会让两个并发拷贝写进同一块缓冲。
    //   因此并发单元 = lane，每个 lane 拥有：独立哈希池（pooled 模式）、独立 D2Stream
    //   （因而独立 pinned 槽池；inline 模式下由 D2Stream 自带池）、独立 pinned 内存、
    //   独立 D2/哈希 scratch 与计时。
    // **不变量**：store 的写者仍然只有"唯一备菜线程 + 唯一 actor"（disk_prefix_cache.cpp），
    // 本改动只并行 D2H 拷贝 / 哈希 / 编码 / 组批，绝不增加 store writer。
    // 下列字段只被本 lane 的 worker 线程读写，故无需加锁。
    struct WriteLane {
        std::unique_ptr<fastllm::prefix_write::HashPackPool> hashPool;   // 仅 pooled 模式非空
        std::unique_ptr<fastllm::prefix_write::D2Stream> d2Stream;
        void *d2SlotMemory = nullptr;      // 本 lane 的 kPinnedSlots×BufferBytes pinned 内存
        size_t d2SlotBytes = 0;
        // §④ 诊断（累计，按检查点做差）：提交/泵/取回耗时与取回次数。
        double d2SubmitMs = 0, d2PumpMs = 0, d2TakeMs = 0;
        double lastD2TakeMs = 0;           // A2：D2 取回增量的基线
        uint64_t d2TakeCount = 0;
        uint64_t d2UnhashedPages = 0;   // A-5：本笔作业里"没有内容指纹、跳过时间戳比对"的页数
        // A-6 fix D：主干 1 时间线。steady_clock 纳秒，与
        // FastllmObservationTimestampNs()/prefill_chunk_profile 同钟。每笔作业在 Work() 里清零。
        uint64_t saveStartNs = 0;       // 本笔作业第一次 D2H 入队被接受的时刻
        uint64_t saveTerminalNs = 0;    // commit-submitted（交给备菜线程，不是落盘）或失败的时刻
        // A-6 fix F (2026-11-01, user-approved, measurement only, FASTLLM_GPU_TIMELINE=1): device ->
        // reusable timing events on the copy stream (before the job's first D2H / after each chunk's
        // last D2H) and their ms offsets from the device anchor, resolved after the existing
        // copy-stream synchronize. Cleared per job in Work(); events live as long as the cache.
        std::map<int, std::pair<void *, void *>> timelineEvents;
        std::set<int> timelineBeginRecorded;
        std::map<int, double> timelineBeginMs, timelineEndMs;
        uint64_t d2Requests = 0;           // D2-stream 用的请求号（本 lane 独占）
        // S4：流水线**内部**峰值（末尾采样只会看到 0，必须随流水推进取 max）
        uint64_t d2PendingMax = 0, d2SlotsMax = 0, poolInflightMax = 0;
        std::map<int, std::vector<void *>> d2RankEvents;   // rank -> 生产者事件（就绪门）
        std::vector<D2Chunk> d2Chunks;                     // ticket.sequence -> 分块上下文
        std::map<int, void *> d2CopyStreams;               // rank -> 该 rank 的拷贝 stream
        // §17.3：0=KV(store) 1=linear 2=spec，把"拷进池/提交"与"等池结果"分开计时。
        double storeSubmitMs[3] = {0,0,0}, storeTakeMs[3] = {0,0,0};
        double lastStoreSubmitMs[3] = {0,0,0}, lastStoreTakeMs[3] = {0,0,0};
        // A2（2026-10-02）：本 lane 的检查点统计（各字段定义见 export-profile 打印处注释）。
        //   hashCpuSumMs           花在哈希流水线上的墙钟总和（D2 泵 + linear/spec 提交与取回），
        //                          并行时会大于墙钟，用于判断"哈希是不是瓶颈"。
        //   hashCriticalWallMs     其中真正让 worker 阻塞等结果的墙钟（TakeOrdered*），关键路径。
        //   captureBackgroundWaitMs worker 等"后台流水线"（D2H/哈希）产出结果的墙钟
        //                          （D2 取回未命中的等待 + linear/spec 取回等待）。
        //   serviceMs / endToEndMs 单检查点服务墙钟 / 入队→完成端到端墙钟。
        double hashCpuSumMs = 0, hashCriticalWallMs = 0;
        double captureBackgroundWaitMs = 0;
        double serviceMs = 0, endToEndMs = 0;
        uint64_t jobsDone = 0;
        // §3/§14（R1 起 per-lane）：profile 差分基线。原本是类级成员（单 worker 独占），
        // 多 lane 并发读写同一基线会产生数据竞争与错误增量，故移到 lane 内独占。
        DiskPrefixCache::Json lastProfileSnapshot;
        fastllm::prefix_v4::JournalStats lastJournalStats;
    };
    std::vector<std::unique_ptr<WriteLane>> writeLanes;
    // A-6 fix E rev2：v4 回收谓词跑在存储 actor 上，契约禁止 I/O（disk_prefix_cache.h:182-192）。
    // 否决只在内存里计数并记下最后一个 key；由导出 lane 线程在作业结束后打印（G7 运行时证据）。
    struct ReclaimVetoLog {
        std::atomic<uint64_t> count{0};
        std::mutex mutex;
        std::string lastKey, lastIdentity;
    };
    std::shared_ptr<ReclaimVetoLog> reclaimVetoLog = std::make_shared<ReclaimVetoLog>();
    std::atomic<uint64_t> reclaimVetoPrinted{0};
    // §④ 真机 CopyIntoSlot：把该 lane 的 ticket D2H 拷进 slot、归还页、返回待哈希字节。
    std::string D2CopyIntoSlot(size_t lane, int slot,
                               const fastllm::prefix_write::TicketId &ticket, size_t bytes);
    std::map<int, size_t> pinnedGpuBytes;
    std::map<int, size_t> gpuBudget;
    // S4-C（2026-10-04）方案 C（请求生命期 keepalive）观测量：
    //   exportProtectedBytes = **当前**为导出而额外持有的页字节（整段 Pick 的引用，
    //                          D2H 完成即逐块释放）；
    //   exportProtectedPeak  = 其历史峰值。真机用它观察"整段保护"带来的容量压力。
    // 两者都只在 mutex 下读写（Capture/Export/D2CopyIntoSlot/Work），不新增接口。
    uint64_t exportProtectedBytes = 0;
    uint64_t exportProtectedPeak = 0;
    double prefillMsPerToken = 0, nativeReadMsPerToken = 0, convertedReadMsPerToken = 0;
    bool nativeCalibrationAttempted = false, convertedCalibrationAttempted = false;
    uint64_t storageDevice = 0;
    std::atomic<uint64_t> writes{0}, reads{0}, committed{0}, restored{0}, skipped{0};
    // A2（2026-10-02）写路径计数（只加观测量，不改任何行为）：
    //   expectedCheckpoints = "期望的检查点数"：**由 native capture 事件推导** —— 模型决定要写、
    //       且成功入队（job.snapshot != nullptr）的检查点数。不含被准入/预算/队列拒掉的（那些进 degraded）。
    //   admittedCheckpoints = 成功拿到协调器票（TryAdmit Accepted）的检查点数。
    //   terminalCheckpoints = 调用过 coordinator->Complete 的检查点数（成功/失败/取消都算），
    //       关停时 admitted − terminal 就是"仍未归还的票"，应与 credit_balance 同向为 0。
    //   degraded            = 准入失败/预算不足/队列满/gpu_pin 拒绝/Export 异常而被丢弃的检查点数。
    // captureCpuNs = 请求线程在 Capture() 里同步花的 CPU 墙钟（身份+准入+选页+记录事件+入队）；
    //                捕获后的 D2H/哈希在后台 worker 上，不计入。
    std::atomic<uint64_t> expectedCheckpoints{0}, admittedCheckpoints{0}, terminalCheckpoints{0},
        degraded{0}, captureCpuNs{0}, pageLeaseReleases{0};
    size_t inflightJobs = 0, pendingCopy = 0, failedRetainedBytes = 0;
    uint64_t failedRetentionRejects = 0;
    std::atomic<uint64_t> copyGenerationCounter{0}, requestGenerationCounter{0};
    std::atomic<bool> shutdownStarted{false}, shutdownClean{false}, shutdownFailed{false};
    std::atomic<bool> selfCheckRunning{false};
    uint64_t detachedTaskCount = 0;
    std::thread selfCheckWorker;

    static void Check(bool ok, const char *message) { DiskPrefixCache::Require(ok, message); }
    bool ValidateJobIdentityLocked(const Job &job, const char *stage, std::string &error,
                                   bool requireActiveRequest = true) const {
        if (job.hintOnly || job.image) return true;
        if (!job.request_id || !job.request_generation || !job.page_generation ||
            !job.copy_generation || job.cache_identity.empty() || !job.owner_lease ||
            !job.request_state) {
            error = "identity_missing";
        } else if (job.request_state->request_id != job.request_id ||
                   (job.state && (job.state->request_id != job.request_id ||
                                  job.state->request_generation != job.request_generation))) {
            error = "request_identity_mismatch";
        } else if (requireActiveRequest &&
                   (job.request_state->cancelled.load(std::memory_order_acquire) ||
                    job.request_state->generation.load(std::memory_order_acquire) !=
                        job.request_generation)) {
            error = "stale_request_generation";
        }
        if (!error.empty()) {
            std::fprintf(stderr, "[Prefix SSD] identity-rejected: stage=%s request_id=%llu "
                         "request_generation=%llu page_generation=%llu copy_generation=%llu "
                         "cache_identity=%s reason=%s.\n",
                         stage ? stage : "unknown", (unsigned long long)job.request_id,
                         (unsigned long long)job.request_generation,
                         (unsigned long long)job.page_generation,
                         (unsigned long long)job.copy_generation,
                         job.cache_identity.empty() ? "none" : job.cache_identity.c_str(),
                         error.c_str());
            return false;
        }
        return true;
    }
    void RetainFailedJobLocked(Job &&job, const char *reason) {
        const size_t bytes = job.stateBytes;
        if (failedJobs.size() >= kMaxFailedRetained ||
            bytes > kMaxFailedBytes || failedRetainedBytes > kMaxFailedBytes - bytes) {
            ++failedRetentionRejects;
            std::fprintf(stderr, "[Prefix SSD] failed-retention-rejected: status=FAILED_RETENTION_FULL "
                         "request_id=%llu generation=%llu cache_identity=%s bytes=%zu reason=%s.\n",
                         (unsigned long long)job.request_id,
                         (unsigned long long)job.request_generation,
                         job.cache_identity.empty() ? "none" : job.cache_identity.c_str(),
                         bytes, reason ? reason : "failed");
            return;
        }
        failedRetainedBytes += bytes;
        failedJobs.push_back(std::move(job));
    }
    // §A/B 对照开关（运行时可切，同一二进制）：
    //   inline（**默认**，2026-09-29 A/B 实测最快）-> 不用池，内联算摘要
    //   pool1 / pool4                              -> 用固定池，worker 数 1 或 4（保留做实验）
    // A/B 初步（同库、交错 3 轮、42 样本/组）：inline 中位 1643 ms、超线 14/42；
    // pool1 1755/25、pool4 1713/19 → 池在这台机器上是负收益（线程交接/调度开销大于并行收益）。
    static int RequestedHashWorkers() {
        const char *mode = std::getenv("FASTLLM_PREFIX_CACHE_HASH_MODE");
        if (mode == nullptr || mode[0] == 0) return 0;
        if (std::strcmp(mode, "inline") == 0) return 0;
        if (std::strcmp(mode, "pool1") == 0) return 1;
        if (std::strcmp(mode, "pool4") == 0) return 4;
        return 4;
    }
    static bool PooledHashEnabled() { return RequestedHashWorkers() > 0; }
    // R1（2026-10-02）写路径并发化：模型侧写 worker 数。业界依据（只读调研）：
    // vLLM `DualQueueThreadPool(n_write_threads=16)`、LMCache `_DEFAULT_THREAD_COUNT=4`、
    // vLLM simple_kv_offload 的独立 store/load 线程。默认 4（兼顾吞吐与内存/线程开销），
    // 上限 16；`FASTLLM_PREFIX_CACHE_WRITE_WORKERS` 覆盖；`FASTLLM_PREFIX_CACHE_WRITE_MODE=serial`
    // 回退到旧的单线程行为（A/B 对照）。
    // **不变量**：store 的写者仍只有"唯一备菜线程 + 唯一 actor"，本开关只并行模型侧的
    // D2H/哈希/打包/组批，绝不增加 store writer。
    static int RequestedWriteWorkers() {
        const char *mode = std::getenv("FASTLLM_PREFIX_CACHE_WRITE_MODE");
        if (mode != nullptr && std::strcmp(mode, "serial") == 0) return 1;
        const char *text = std::getenv("FASTLLM_PREFIX_CACHE_WRITE_WORKERS");
        const int parsed = text != nullptr ? std::atoi(text) : 0;
        if (parsed <= 0) return 4;
        return parsed > 16 ? 16 : parsed;
    }
    // R1：作业 → lane 的确定性亲和。同一 ExportState 地址永远映射到同一 lane ⇒ **同请求串行**
    // （writer/v4Chunks/state.kv 的累积与提交顺序不变）；不同地址经 64 位混合后分布到各 lane
    // ⇒ **不同请求并行**。地址在对象释放后被复用也只是落到同一条 lane，仍然安全。
    // 不按 ResponseContext 亲和：R4 会在同一请求内重建 ExportState（新 writer/new kv），
    // 两个 ExportState 各自独立，分布到不同 lane 也不会互相破坏。
    static size_t LaneForState(const ExportState *state, int lanes) {
        if (lanes <= 1 || state == nullptr) return 0;
        uint64_t x = uint64_t(reinterpret_cast<uintptr_t>(state));
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return size_t(x % uint64_t(lanes));
    }
    // ④：批量哈希的池预算（可用环境变量收紧/放宽）。
    // 2026-09-30 实验改动：默认 192 → 512 MiB —— KV(D2-stream) 与 linear/spec 现在**共用**
    // 同一个池，三者最坏同时在途约 64 + 121 + 121 MiB，192 MiB 会被拒收（SubmitBatch 返回 false）。
    // R1：池改为**每 lane 一份**（见 WriteLane），因此该预算是"每个 lane"的：单 lane 内 D2 与
    // linear/spec 是先后使用（D2 取回完才做 linear/spec），最坏同时只需容纳其中一段；默认值保持
    // 512 MiB 不变（父代理要求其余默认参数一律不动），但 W 个 lane 的账面上限是 W×512 MiB。
    static size_t HashPoolBytes() {
        const char *value = std::getenv("FASTLLM_PREFIX_CACHE_HASH_POOL_MB");
        const long mb = value ? std::atol(value) : 512;
        return size_t(mb <= 0 ? 512 : mb) << 20;
    }
    // §22.4.4 甲：把一段字节交给固定哈希池并行算摘要，再按提交顺序用
    // StoreVerified 提交（池保证"摘要与字节一起移动"，调用方无法伪造 hash）。
    // 同时最多保持 kStoreInflight 个分块在池里，内存有界。
    DiskPrefixCache::Json::array StoreHashed(DiskPrefixCache::WriteSession &writer, size_t bytes,
                                             const std::function<void(size_t,char*,size_t)> &read,
                                             int site, fastllm::prefix_write::HashPackPool *pool,
                                             double (&storeSubmitMs)[3], double (&storeTakeMs)[3]) {
        // §A/B：inline 模式完全走原路径（不经过线程池），用于对照。
        // R1：池改为**按 lane 传入**（每个写 worker 一套独立池），否则两个 worker 会在同一
        // 个"单消费者定序交付"的池上互相取走对方的分块（TakeOrdered* 取的是全局队首）。
        if (pool == nullptr || !PooledHashEnabled()) return writer.StoreBytes(bytes, read);
        const int index = (site >= 0 && site < 3) ? site : 0;
        const size_t chunk = DiskPrefixCache::ChunkBytes;
        const size_t total = (bytes + chunk - 1) / chunk;
        DiskPrefixCache::Json::array chunks;
        if (total == 0) return chunks;
        // §23（④）：整段**一次批量提交**——线程只交接一次，而不是每 8MiB 一次。
        if (bytes <= HashPoolBytes()) {
            // §23 修正：**切成 W 份**交给 W 个 worker 真并行（上一版把整段放进一个任务，
            // 只由一个 worker 串行算完，实测比内联慢 350ms——已由 A/B 证伪）。
            // 交付仍按提交顺序，合并后与串行分块顺序完全一致。
            const size_t workers = size_t(std::max(1, RequestedHashWorkers()));
            const size_t slices = std::min(workers, total);
            const auto submitStart = std::chrono::steady_clock::now();
            size_t first = 0;
            for (size_t slice = 0; slice < slices; ++slice) {
                const size_t count = (total - first) / (slices - slice);
                std::vector<std::string> payloads;
                payloads.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    const size_t position = first + i;
                    std::string payload(std::min(chunk, bytes - position * chunk), '\0');
                    read(position * chunk, payload.data(), payload.size());
                    payloads.push_back(std::move(payload));
                }
                first += count;
                Check(pool->SubmitBatch({uint64_t(slice + 1), uint64_t(total)},
                                            std::move(payloads)), "ssd_hash_batch_submit");
            }
            const auto takeStart = std::chrono::steady_clock::now();
            for (size_t slice = 0; slice < slices; ++slice) {
                fastllm::prefix_write::TicketId batchTicket;
                std::vector<fastllm::prefix_write::HashedChunk> batch;
                Check(pool->TakeOrderedBatch(batchTicket, batch, 600000), "ssd_hash_batch_take");
                for (auto &item : batch) {
                    Check(item.Valid(), "ssd_hash_batch_invalid");
                    const auto entry = writer.StoreVerified(std::move(item));
                    for (const auto &one : entry) chunks.push_back(one);
                }
            }
            const auto takeEnd = std::chrono::steady_clock::now();
            storeSubmitMs[index] += std::chrono::duration<double,std::milli>(takeStart-submitStart).count();
            storeTakeMs[index] += std::chrono::duration<double,std::milli>(takeEnd-takeStart).count();
            return chunks;
        }
        const size_t inflight = 4;
        size_t submitted = 0, taken = 0;
        while (taken < total) {
            const auto submitStart = std::chrono::steady_clock::now();
            while (submitted < total && submitted - taken < inflight) {
                std::string payload(std::min(chunk, bytes - submitted * chunk), '\0');
                read(submitted * chunk, payload.data(), payload.size());
                Check(pool->Submit({uint64_t(submitted + 1), uint64_t(total)}, std::move(payload)),
                      "ssd_hash_pool_submit");
                ++submitted;
            }
            const auto takeStart = std::chrono::steady_clock::now();
            fastllm::prefix_write::TicketId ticket;
            fastllm::prefix_write::HashedChunk hashed;
            Check(pool->TakeOrderedChunk(ticket, hashed, 600000), "ssd_hash_pool_take");
            const auto takeEnd = std::chrono::steady_clock::now();
            Check(hashed.Valid(), "ssd_hash_pool_invalid");
            const auto entry = writer.StoreVerified(std::move(hashed));
            storeSubmitMs[index] += std::chrono::duration<double,std::milli>(takeStart-submitStart).count();
            storeTakeMs[index] += std::chrono::duration<double,std::milli>(takeEnd-takeStart).count();
            for (const auto &item : entry) chunks.push_back(item);
            ++taken;
        }
        return chunks;
    }
    static std::string Env(const char *name) {
        const char *v = std::getenv(name); return v ? v : "";
    }
    static std::string Dtype(DataType type) {
        return type == DataType::FP8_E4M3 ? "fp8_e4m3" :
               type == DataType::FLOAT16 ? "float16" : "unsupported";
    }
    static std::vector<int> Ints(const Json &value) {
        Check(value.is_array() && value.array_items().size() <= 262144, "ssd_dimensions");
        std::vector<int> out;
        for (const auto &x : value.array_items()) {
            Check(x.is_number() && x.number_value() == x.int_value(), "ssd_integer");
            out.push_back(x.int_value());
        }
        return out;
    }
    static std::vector<float> Floats(const Json &value) {
        Check(value.is_array() && value.array_items().size() <= 262144, "ssd_speculative_values");
        std::vector<float> out;
        out.reserve(value.array_items().size());
        for (const auto &x : value.array_items()) {
            Check(x.is_number() && std::isfinite(x.number_value()), "ssd_speculative_value");
            out.push_back((float)x.number_value());
        }
        return out;
    }
    static Json IntArray(const std::vector<int> &values) {
        Check(values.size() <= 262144, "ssd_speculative_values");
        Json::array out;
        for (int value : values) out.emplace_back(value);
        return out;
    }
    static Json FloatArray(const std::vector<float> &values) {
        Check(values.size() <= 262144, "ssd_speculative_values");
        Json::array out;
        for (float value : values) {
            Check(std::isfinite(value), "ssd_speculative_value");
            out.emplace_back((double)value);
        }
        return out;
    }
    static size_t TensorBytes(const Data &data) { return data.GetBytes(); }
    static size_t SnapshotBytes(const Qwen35LinearPrefixSnapshot &snapshot) {
        size_t size = 0;
        for (auto &layer : snapshot.layers) if (layer.linear) {
            for (auto *cache : {&layer.first, &layer.second}) {
                if (cache->multiDevice) for (auto &local : cache->locals) size += TensorBytes(local.second);
                else size += TensorBytes(cache->single);
            }
        }
        if (snapshot.mtpValid) {
            size += TensorBytes(snapshot.mtpKey) + TensorBytes(snapshot.mtpValue);
        }
        if (snapshot.dflashValid) {
            for (const auto &pair : snapshot.dflashKeyValues)
                size += TensorBytes(pair.first) + TensorBytes(pair.second);
        }
        return size;
    }
    Json SaveSpeculativeTensor(std::vector<char> &packed, const Data &data) {
        Check(data.dataDevice == DataDevice::CPU && data.cpuData != nullptr &&
                  data.dims.size() == 3,
              "ssd_speculative_tensor");
        Check(data.dataType == DataType::FLOAT16 || data.dataType == DataType::BFLOAT16 ||
                  data.dataType == DataType::FLOAT32,
              "ssd_speculative_dtype");
        Check(data.strides.size() == 3 && data.dims[0] > 0 && data.dims[1] > 0 &&
                  data.dims[2] > 0 && data.strides[2] == 1 &&
                  data.strides[1] == (uint64_t)data.dims[2] &&
                  data.strides[0] == (uint64_t)data.dims[1] * data.dims[2],
              "ssd_speculative_layout");
        const size_t bytes = data.GetBytes();
        Check(bytes > 0 && bytes <= HostLimit() - packed.size(), "ssd_speculative_pack_budget");
        const size_t offset = packed.size();
        packed.resize(offset + bytes);
        memcpy(packed.data() + offset, data.cpuData, bytes);
        return Json::object{{"dims", data.dims}, {"dtype", (int)data.dataType},
                            {"offset", (double)offset}, {"bytes", (double)bytes}};
    }
    void LoadSpeculativeTensor(const std::vector<char> &packed, const Json &value, Data &data) {
        auto dims = Ints(value["dims"]);
        Check(dims.size() == 3 && dims[0] > 0 && dims[1] > 0 && dims[2] > 0,
              "ssd_speculative_shape");
        const int type = value["dtype"].int_value();
        Check(type == (int)DataType::FLOAT16 || type == (int)DataType::BFLOAT16 ||
                  type == (int)DataType::FLOAT32,
              "ssd_speculative_dtype");
        DataType dataType = (DataType)type;
        Data sample(dataType, dims);
        const size_t bytes = sample.GetBytes();
        Check(value["offset"].is_number() && value["offset"].number_value() >= 0 &&
                  value["bytes"].is_number() && value["bytes"].number_value() == (double)bytes,
              "ssd_speculative_bytes");
        const size_t offset = (size_t)value["offset"].number_value();
        Check((double)offset == value["offset"].number_value() &&
                  offset <= packed.size() && bytes <= packed.size() - offset,
              "ssd_speculative_bounds");
        data = Data(dataType, dims);
        data.Allocate();
        memcpy(data.cpuData, packed.data() + offset, bytes);
        data.isKVCache = true;
        data.isPagedKVCache = false;
        data.pagedKVCacheData = nullptr;
        data.pageIndex.clear();
        data.lastPageLen = 0;
        data.multiDeviceData = false;
        data.multiDeviceDatas.clear();
        data.ClearTensorParallelLayout();
    }
    Json SaveSpeculative(std::vector<char> &packed,
                         const Qwen35LinearPrefixSnapshot &snapshot) {
        Check(!(snapshot.mtpValid && snapshot.dflashValid), "ssd_speculative_kinds");
        if (snapshot.dflashValid) {
            Check(snapshot.dflashTokens > 0 &&
                      (int)snapshot.dflashKeyValues.size() == model->dflashLayers,
                  "ssd_dflash_snapshot_missing");
            Json::array layers;
            int windowTokens = -1;
            for (const auto &pair : snapshot.dflashKeyValues) {   // 形状检查保持串行、先做完
                Check(pair.first.dims == pair.second.dims && pair.first.dims.size() == 3 &&
                          pair.first.dataType == DataType::FLOAT16 &&
                          pair.first.dims[0] == model->dflashKvHeads &&
                          pair.first.dims[1] > 0 && pair.first.dims[1] <= snapshot.dflashTokens &&
                          pair.first.dims[2] == model->dflashHeadDim,
                      "ssd_dflash_snapshot_shape");
                if (windowTokens < 0) windowTokens = pair.first.dims[1];
                Check(windowTokens == pair.first.dims[1], "ssd_dflash_snapshot_window");
            }
            std::vector<PackPiece> pieces(snapshot.dflashKeyValues.size());
            ParallelPieces(pieces, [&](size_t index, PackPiece &piece) {
                const auto &pair = snapshot.dflashKeyValues[index];
                Json key = SaveSpeculativeTensor(piece.buffer, pair.first);
                Json value = SaveSpeculativeTensor(piece.buffer, pair.second);
                piece.descriptor = Json::object{{"key", key}, {"value", value}};
            }, PackWorkers());
            ReservePackMerge(packed, pieces, "ssd_speculative_pack_budget");
            for (const auto &piece : pieces) {
                const size_t base = packed.size();
                packed.insert(packed.end(), piece.buffer.begin(), piece.buffer.end());
                layers.emplace_back(Json::object{{"key", ShiftedOffset(piece.descriptor["key"], base)},
                                                 {"value", ShiftedOffset(piece.descriptor["value"], base)}});
            }
            Check(packed.size() <= HostLimit(), "ssd_speculative_pack_budget");
            return Json::object{{"version", 1}, {"kind", "dflash2"},
                {"tokens", snapshot.dflashTokens}, {"layers", layers},
                {"kv_heads", model->dflashKvHeads}, {"head_dim", model->dflashHeadDim},
                {"window_tokens", windowTokens}, {"proposal_tokens", IntArray(snapshot.dflashProposalTokens)},
                {"proposal_candidate_ids", IntArray(snapshot.dflashProposalCandidateIds)},
                {"proposal_candidate_probs", FloatArray(snapshot.dflashProposalCandidateProbs)}};
        }
        if (snapshot.mtpValid) {
            Check(snapshot.mtpTokens > 0 && snapshot.mtpKey.dims == snapshot.mtpValue.dims &&
                      snapshot.mtpKey.dataType == snapshot.mtpValue.dataType,
                  "ssd_mtp_snapshot_missing");
            return Json::object{{"version", 1}, {"kind", "mtp"},
                {"tokens", snapshot.mtpTokens},
                {"key", SaveSpeculativeTensor(packed, snapshot.mtpKey)},
                {"value", SaveSpeculativeTensor(packed, snapshot.mtpValue)},
                {"proposal_tokens", IntArray(snapshot.mtpProposalTokens)}};
        }
        return Json::object{{"version", 1}, {"kind", "none"}};
    }
    void LoadSpeculative(const Json &value, const std::vector<char> &packed,
                         ResponseContext *ctx, int length,
                         Qwen35LinearPrefixSnapshot &snapshot) {
        Check(value.is_object() && value["version"].int_value() == 1,
              "ssd_speculative_version");
        const std::string kind = value["kind"].string_value();
        const bool requireMtp = model->RequiresMtpPrefixSnapshot(ctx);
        const bool requireDFlash = model->RequiresDFlashPrefixSnapshot(ctx);
        if (kind == "none") {
            Check(!requireMtp && !requireDFlash, "ssd_missing_speculative_state");
            return;
        }
        if (kind == "dflash2") {
            Check(!requireMtp && value["tokens"].int_value() == length &&
                      value["kv_heads"].int_value() == model->dflashKvHeads &&
                      value["head_dim"].int_value() == model->dflashHeadDim &&
                      value["window_tokens"].int_value() > 0 &&
                      value["window_tokens"].int_value() <= length,
                  "ssd_dflash_state_identity");
            const auto &layerValues = value["layers"].array_items();
            Check((int)layerValues.size() == model->dflashLayers,
                  "ssd_dflash_state_layers");
            snapshot.dflashKeyValues.resize(model->dflashLayers);
            for (int layer = 0; layer < model->dflashLayers; ++layer) {
                LoadSpeculativeTensor(packed, layerValues[layer]["key"],
                                      snapshot.dflashKeyValues[layer].first);
                LoadSpeculativeTensor(packed, layerValues[layer]["value"],
                                      snapshot.dflashKeyValues[layer].second);
                Check(snapshot.dflashKeyValues[layer].first.dataType == DataType::FLOAT16 &&
                          snapshot.dflashKeyValues[layer].second.dataType == DataType::FLOAT16 &&
                          snapshot.dflashKeyValues[layer].first.dims[1] ==
                              value["window_tokens"].int_value(),
                      "ssd_dflash_state_window");
            }
            snapshot.dflashValid = true;
            snapshot.dflashTokens = length;
            snapshot.dflashProposalTokens = Ints(value["proposal_tokens"]);
            snapshot.dflashProposalCandidateIds = Ints(value["proposal_candidate_ids"]);
            snapshot.dflashProposalCandidateProbs = Floats(value["proposal_candidate_probs"]);
            Check(Qwen35DFlashSnapshotMatches(snapshot, length, model->dflashLayers,
                                               model->dflashKvHeads, model->dflashHeadDim),
                  "ssd_dflash_state_invalid");
            return;
        }
        if (kind == "mtp") {
            Check(!requireDFlash && value["tokens"].int_value() == length,
                  "ssd_mtp_state_identity");
            LoadSpeculativeTensor(packed, value["key"], snapshot.mtpKey);
            LoadSpeculativeTensor(packed, value["value"], snapshot.mtpValue);
            Check(snapshot.mtpKey.dims == snapshot.mtpValue.dims &&
                      snapshot.mtpKey.dataType == snapshot.mtpValue.dataType &&
                      snapshot.mtpKey.dims == std::vector<int>({model->num_key_value_heads,
                          length, model->head_dim}),
                  "ssd_mtp_state_shape");
            snapshot.mtpValid = true;
            snapshot.mtpTokens = length;
            snapshot.mtpProposalTokens = Ints(value["proposal_tokens"]);
            return;
        }
        Check(false, "ssd_unknown_speculative_state");
    }
    Json RuntimeLayout() const {
        Json::array linear;
        for (int layer = 0; layer < model->block_cnt; ++layer)
            linear.emplace_back(Qwen35LayerIsLinearAttention(model, layer));
        const bool dflash = model->HasDFlashWeights();
        const bool mtp = !dflash && model->HasMtpWeights();
        return Json::object{{"abi",4},{"layers",model->block_cnt},
            {"linear",linear},{"atype",(int)model->dataType},
            {"state", "ordinary_kv+linear_attention+speculative"},
            {"linear_layout","global_channels_kv_v1"},
            {"speculative", dflash ? "dflash2" : mtp ? "mtp" : "none"},
            {"draft_identity", draftIdentity},
            {"speculative_layers", dflash ? model->dflashLayers : mtp ? model->mtp_num_hidden_layers : 0},
            {"speculative_kv_heads", dflash ? model->dflashKvHeads : mtp ? model->num_key_value_heads : 0},
            {"speculative_head_dim", dflash ? model->dflashHeadDim : mtp ? model->head_dim : 0},
            {"dflash_window", dflash ? model->dflashSlidingWindow : 0}};
    }
    // R5 修复（2026-10-02）：单设备"钉住预算"此前硬编码 min(256 MiB, 池/10)。
    // KV 状态约 3.3–3.6 KB/token ⇒ 256 MiB 只能装约 73728 token，长上下文的检查点必然被拒
    // （真机实测在 77824/79872/81920/83968 四点超限）。改为可用环境变量配置：
    //   FASTLLM_PREFIX_CACHE_GPU_PIN_LIMIT_MB = 0 或不设 ⇒ 旧行为（min(256MiB, 池/10)）；
    //   > 0 ⇒ 直接使用该值（MiB），仍不超过池总字节，避免把整池钉死。
    static size_t GpuPinLimitBytes(size_t poolBytes) {
        static const size_t configured = [] {
            const char *text = std::getenv("FASTLLM_PREFIX_CACHE_GPU_PIN_LIMIT_MB");
            const long long parsed = text != nullptr ? std::atoll(text) : 0;
            return parsed > 0 ? (size_t)parsed << 20 : (size_t)0;
        }();
        if (configured > 0) return std::min(configured, poolBytes);
        return std::min(size_t(256) << 20, poolBytes / 10);
    }
    void RefreshGpuBudget() {
        gpuBudget.clear();
        for (int layer = 0; layer < model->block_cnt; ++layer)
            for (int side = 0; side < 2; ++side)
                for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0)) {
                    auto pool = ref.second;
                    gpuBudget[ref.first] += (size_t)pool->maxPages*pool->pageLen*pool->dims[2]*pool->dims[3]*pool->unitSize;
                }
        // A-2（2026-10-04）：必须在下一行把 gpuBudget 改写成"pin 预算（池/10）"**之前**，
        // 先记住 GPU 侧 KV 池的**总字节**——它是 host tier 的下限依据（Dynamo：CPU cache 必须大于
        // device cache，否则每个前向都往外倒 = churn）。
        {
            size_t poolTotal = 0;
            for (auto &entry : gpuBudget) poolTotal += entry.second;
            GpuPoolFloorRef().store(poolTotal, std::memory_order_relaxed);
        }
        for (auto &budget : gpuBudget) budget.second = GpuPinLimitBytes(budget.second);
    }
    bool Ensure() {
        if (!workers.empty()) return true;
        // 定位用：重启后首次使用前缀缓存时会出现约 190 s 的日志零增长，
        // 其位置被字节级证据钉在"[Prefix SSD] io threads = 8"之后、"SELFCHECK:"之前。
        // 这里给该窗口内每一步打时间戳，默认关闭，FASTLLM_PREFIX_CACHE_SELFCHECK_TRACE=1 打开。
        const bool ensureTrace = SelfCheckTraceEnabled();
        const auto ensureStart = std::chrono::steady_clock::now();
        auto markEnsure = [&](const char *stage) {
            if (!ensureTrace) return;
            std::fprintf(stderr,"[Prefix SSD][mark] ensure_%s t_ms=%.0f\n",stage,
                         std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - ensureStart).count() / 1000.0);
            std::fflush(stderr);
        };
        markEnsure("enter");
        try {
            Check(Dtype(model->kvCacheDataType) != "unsupported","ssd_unsupported_kv_encoding");
            family = baseFamily;
            identity = DiskPrefixCache::Digest(baseFamily + ":target-state-v4:" + Dtype(model->kvCacheDataType));
            model->pagedCacheModelIdentity = identity;
            model->pagedCacheFamilyIdentity = family;
            model->pagedCacheLayout = RuntimeLayout().dump();
            if (!configuredIdentity.empty()) {
                Check(DiskPrefixCache::IsDigest(configuredIdentity) &&
                          configuredIdentity == identity,
                      "ssd_identity_mismatch");
            }
            Check(DiskPrefixCache::IsDigest(encoderIdentity), "ssd_invalid_encoder");
            // 读并行度：原写死 3，现默认 8，可用 FASTLLM_PREFIX_CACHE_IO_THREADS 覆盖（1..64）。
            // 依据（实测）：读盘 1.05 GiB 用 16.65 s = 67.7 MB/s，而每批并行度 = ioThreads、
            // 每批总量 ≤ BatchBytes(32 MiB)、批间串行 → 约 33 批，与实测时间吻合，并行度是主瓶颈之一。
            int ioThreads = 8;
            { const std::string text = Env("FASTLLM_PREFIX_CACHE_IO_THREADS");
              if (!text.empty()) { const int parsed = std::atoi(text.c_str()); if (parsed > 0) ioThreads = parsed; } }
            ioThreads = std::max(1, std::min(64, ioThreads));
            PrefixCacheDebug("[Prefix SSD] io threads = %d (read parallelism; env FASTLLM_PREFIX_CACHE_IO_THREADS).\n",
                             ioThreads);
            // HOSTTRIM: return arena memory freed by SSD saves to the OS (FASTLLM_HOST_MALLOC_TRIM_SECONDS, default off).
            fastllm::HostMallocTrimStartOnce();
            // §13.1/§10.1.1：先做 storage 规划，再校验启动 manifest，最后才构造
            // writer。任何一步失败都必须在打开目录之前发生。
            markEnsure("plan_begin");
            PersistentPrefixStoragePlan storagePlan;
            {
                std::string error;
                const bool planned = PlanPersistentPrefixStorage(directory, storagePlan, error);
                const std::string detail = "ssd_storage_plan:" + error;
                Check(planned, detail.c_str());
            }
            markEnsure("plan_done");
            const bool useV4 = true;
            const int expectedFormat = 4;
            const int expectedManifestSchema = 2;
            Json startup;
            bool hasStartup = false;
            if (!startupManifest.empty()) {
                std::string error;
                startup = Json::parse(startupManifest,error);
                Check(error.empty() && startup.is_object(),"ssd_startup_manifest");
                Check(startup["format_version"].int_value() == expectedFormat &&
                          startup["state_abi"]["version"].int_value() == 4 &&
                          startup["state_abi"]["target"].string_value() ==
                              "ordinary_kv+linear_state+speculative",
                      "ssd_startup_abi");
                Check(startup["storage_format_version"].int_value() == 4 &&
                          startup["storage_manifest_schema_version"].int_value() ==
                              expectedManifestSchema,
                      "ssd_startup_storage_manifest");
                Check(startup["draft_identity"].string_value() == draftIdentity,
                      "ssd_startup_draft_identity");
                const auto &encoding = startup["encoding"];
                Check(encoding["container_version"].int_value() == expectedFormat &&
                          encoding["kv_dtype"].string_value() ==
                              Dtype(model->kvCacheDataType) &&
                          encoding["conversion_version"].string_value() ==
                              "fp16-to-fp8-v1",
                      "ssd_startup_encoding");
                hasStartup = true;
            }
            // v4 由同一个 DiskPrefixCache 公开接口承载（内部换成 v4 后端），
            // 因此捕获/恢复路径无需分叉。
            markEnsure("store_target_begin");
            auto targetStore = std::make_shared<DiskPrefixCache>(directory, identity, diskLimit, family,
                                                                  ioThreads);
            markEnsure("store_target_done");
            auto featuresStore = std::make_shared<DiskPrefixCache>(directory,encoderIdentity,diskLimit,
                                                                     encoderIdentity,1);
            markEnsure("store_features_done");
            store = std::move(targetStore); imageStore = std::move(featuresStore);
            // v4 FIFO retirement must consult the same runtime page ledger as
            // L1/L2 migration.  The predicate is metadata-only and therefore
            // safe on the store actor thread; the erase callback removes the
            // durable root only after the store has durably appended ERASE.
            if (store && model->pagedPageLedger) {
                std::weak_ptr<PrefixCachePageLedger> weakLedger = model->pagedPageLedger;
                auto vetoLog = reclaimVetoLog;
                store->SetCheckpointReclaimPredicate(
                    [weakLedger, vetoLog](const std::string &key, const std::string &checkpointIdentity) {
                        auto ledger = weakLedger.lock();
                        const bool allowed = ledger && ledger->CanEraseCheckpoint(key, checkpointIdentity);
                        // A-6 fix E rev2：只在内存里记一笔否决（无 I/O），打印见 Work()。
                        if (!allowed) {
                            std::lock_guard<std::mutex> guard(vetoLog->mutex);
                            vetoLog->lastKey = key;
                            vetoLog->lastIdentity = checkpointIdentity;
                            vetoLog->count.fetch_add(1, std::memory_order_acq_rel);
                        }
                        return allowed;
                    });
                store->SetCheckpointCorruptRetirePredicate(
                    [weakLedger](const std::string &key, const std::string &checkpointIdentity) {
                        auto ledger = weakLedger.lock();
                        if (!ledger) return false;
                        PrefixCachePageLedger::CheckpointSnapshot tracked;
                        return !ledger->Checkpoint(key, tracked) ||
                               ledger->CanRetireCorruptCheckpoint(key, checkpointIdentity);
                    });
                store->SetCheckpointEraseCallback(
                    [weakLedger](const std::string &key, const std::string &checkpointIdentity) {
                        auto ledger = weakLedger.lock();
                        if (ledger) ledger->EraseCheckpoint(key, checkpointIdentity);
                    });
            }
            // 可选启动自检——把"写了但恢复不了"的检查点直接打出来。
            if (store && std::getenv("FASTLLM_PREFIX_CACHE_SELFCHECK") != nullptr) {
                markEnsure("selfcheck_begin");
                // 修复：自检移到后台线程，不再占用请求路径。
                // 实测依据（打点定位，提交 7af1812b，58 检查点/11 GB 库）：
                //   SelfCheck 的 Load 循环 206.1 s（≈3.55 s/检查点），manifest 仅 1 ms、
                //   functional 仅 76 ms；冷启动 0 个检查点则 28 ms 走完。
                // 同步执行会把"重启后的首个请求"堵死并触发看门狗 180 s 零进展停机
                // （verify-fix-a1 / trace-stall-a2 / fix-cache-a1 三轮实测）。
                // 自检本身是只读的、不改任何状态；放后台后：请求路径立即继续，
                // SELFCHECK 四行照旧打印（门⑦仍可读），只是时间点后移。
                // Self-check is tracked and joined by Stop. No detached task may
                // survive shutdown or be mistaken for a completed close.
                std::shared_ptr<DiskPrefixCache> selfCheckStore = store;
                if (selfCheckWorker.joinable()) selfCheckWorker.join();
                selfCheckRunning.store(true, std::memory_order_release);
                selfCheckWorker = std::thread([this, selfCheckStore]() {
                    try {
                        selfCheckStore->SelfCheckStorage();
                    } catch (const std::exception &e) {
                        shutdownFailed.store(true, std::memory_order_release);
                        std::fprintf(stderr, "[Prefix SSD] SELFCHECK exception: %s.\n", e.what());
                    } catch (...) {
                        shutdownFailed.store(true, std::memory_order_release);
                        std::fprintf(stderr, "[Prefix SSD] SELFCHECK unknown exception.\n");
                    }
                    selfCheckRunning.store(false, std::memory_order_release);
                });
                markEnsure("selfcheck_tracked");
            }
            if (hasStartup) store->PutMetadata("identity:"+baseFamily,startup);
            RefreshGpuBudget();
            struct stat filesystem{};
            if (stat(directory.c_str(),&filesystem) == 0) storageDevice = filesystem.st_dev;
            auto cost = store->GetMetadata("cost:"+identity);
            if (cost["device"].string_value() == std::to_string(storageDevice)) {
                prefillMsPerToken = cost["prefill"].number_value();
                nativeReadMsPerToken = cost["native_read"].number_value();
                convertedReadMsPerToken = cost["converted_read"].number_value();
            }
            // host 预算是持久化总保留额度，与队列计数同源（方案 §7.3）。
            coordinator = fastllm::prefix_write::CreateCoordinator(HostLimit());
            // R1（2026-10-02）写路径并发化：N 个写 worker，每个 lane 一套 D2/哈希流水线。
            // 启动顺序：先把 N 条 lane 的池/流全部建好、再起线程 —— 这样 Wait()/Stop() 在
            // 任何时刻都不会看到"半个 lane"（stream 空指针或线程空转）。
            writeWorkerCount = RequestedWriteWorkers();
            laneQueues.assign(size_t(writeWorkerCount), std::deque<Job>());
            writeLanes.clear();
            writeLanes.resize(size_t(writeWorkerCount));
            for (int laneIndex = 0; laneIndex < writeWorkerCount; ++laneIndex) {
                auto lane = std::make_unique<WriteLane>();
                if (useV4) {
                    // §6.1/§22.4.4 甲：固定哈希池。R1 起**按 lane**独立建池（默认 inline 模式
                    // 下 RequestedHashWorkers()==0 ⇒ 不建池，StoreHashed 走内联；D2Stream 自带池）。
                    // 独立建池是必须的：池的 TakeOrdered* 是单消费者全局序，共用会串块。
                    const int workers = RequestedHashWorkers();
                    if (workers > 0) {
                        lane->hashPool = fastllm::prefix_write::CreateHashPackPool(HashPoolBytes(), workers);
                        Check(lane->hashPool != nullptr, "ssd_hash_pool");
                    }
                    // §④ 宽版：D2-stream。校验函数按"生产者事件是否完成"做非阻塞门；
                    // 槽位数与槽大小由协调器常量决定（kPinnedSlots × kSlotBytes = 4 × 8MiB，不改）。
                    // lane 的 pinned 内存在该 worker 入口分配（见 Work(lane)），此处只建流。
                    // 传入本 lane 自己的池作为 sharedPool（为空时流自带一个池）。
                    lane->d2Stream = fastllm::prefix_write::CreateD2Stream(
                        [this, laneIndex](uint64_t eventId) {
                            // eventId 用 rank 编号：该 rank 的生产者事件全部完成才允许拷贝。
                            auto &events = writeLanes[size_t(laneIndex)]->d2RankEvents;
                            auto found = events.find(int(eventId));
                            if (found == events.end()) return false;
                            for (void *event : found->second) {
                                if (FastllmCudaEventQuery(event) != 1) return false;
                            }
                            return true;
                        },
                        [this, laneIndex](int slot, const fastllm::prefix_write::TicketId &ticket,
                                          size_t bytes) {
                            return D2CopyIntoSlot(size_t(laneIndex), slot, ticket, bytes);
                        },
                        D2StagingBudgetBytes(),   // 默认 1024 MiB（原 256），可 env 覆盖
                        lane->hashPool.get());
                    Check(lane->d2Stream != nullptr, "ssd_d2_stream");
                }
                writeLanes[size_t(laneIndex)] = std::move(lane);
            }
            for (int laneIndex = 0; laneIndex < writeWorkerCount; ++laneIndex)
                workers.emplace_back([this, laneIndex] { Work(size_t(laneIndex)); });
            PrefixCacheDebug("[Prefix SSD] write workers = %d (env FASTLLM_PREFIX_CACHE_WRITE_WORKERS, "
                   "serial=%s).\n", writeWorkerCount,
                   writeWorkerCount == 1 ? "yes" : "no");
            // A2（2026-10-04）：把 D2 pinned 槽位数的**生效值**打进日志，供跑后核对——
            // 原硬编码 4 时 t2-256k-a1 实测 slot_waits/slot_acquires = 11585/12224（94.8% 获取失败）。
            PrefixCacheDebug("[Prefix SSD] pinned slots = %d (env FASTLLM_PREFIX_CACHE_PINNED_SLOTS, default %d "
                   "=kPinnedSlots frozen contract; per-lane %d x %.0f MiB).\n",
                   fastllm::prefix_write::PinnedSlots(), fastllm::prefix_write::kPinnedSlots,
                   fastllm::prefix_write::PinnedSlots(),
                   double(BufferBytes) / (1024.0 * 1024.0));
            return true;
        } catch (const std::exception &e) {
            std::fprintf(stderr,"[Prefix SSD] unavailable: %s.\n", e.what());
            return false;
        }
    }
    std::string Prefix(const ResponseContext *ctx, int length) const {
        std::string extra("fastllm-prefix-media-v4");
        const auto &positions = ctx->prefixCachePositionIds;
        if (!positions.empty()) {
            Check(positions.size() % 3 == 0, "ssd_position_identity");
            const size_t positionLength = positions.size() / 3;
            Check(positionLength <= ctx->allTokens.size(), "ssd_position_identity");
            const int available = std::min<int>(length, positionLength);
            bool nonDefault = false;
            for (int axis = 0; axis < 3 && !nonDefault; ++axis) {
                for (int i = 0; i < available; ++i) {
                    if (positions[(size_t)axis * positionLength + i] != i) {
                        nonDefault = true;
                        break;
                    }
                }
            }
            // Keep the historical pure-text key byte-for-byte compatible when
            // positions are the implicit [i, i, i] sequence.  Once a real
            // MRoPE coordinate differs, include the complete canonical prefix;
            // generated tokens beyond the prepared prompt use implicit i.
            if (nonDefault) {
                extra.append("positions-v1");
                for (int axis = 0; axis < 3; ++axis) {
                    for (int i = 0; i < length; ++i) {
                        const int value = i < (int)positionLength
                            ? positions[(size_t)axis * positionLength + i] : i;
                        AppendPrefixKeyInt(extra, static_cast<uint32_t>(value));
                    }
                }
            }
        }
        for (const auto &span : ctx->multimodalPrefixSpans) {
            if (span.begin >= length) continue;
            AppendPrefixKeyInt(extra, static_cast<uint32_t>(span.begin));
            AppendPrefixKeyInt(extra, static_cast<uint32_t>(std::min(span.end, length)));
            extra.append("encoder-v1");
            extra.append(encoderIdentity);
            extra.append(reinterpret_cast<const char *>(span.digest.data()), span.digest.size());
        }
        return DiskPrefixCache::PrefixKey(ctx->allTokens, length, extra);
    }
    static void SetProvenance(ResponseContext *ctx, const std::string &provenance) {
        ctx->prefixCacheProvenance = provenance;
        ctx->prefixCachePageKeys = ctx->prefixCacheBasePageKeys;
        if (provenance.empty()) return;
        int pages = (int)ctx->allTokens.size() / fastllm::GetPageLen();
        if (ctx->prefixCachePageKeys.empty()) {
            ctx->prefixCachePageKeys.resize(pages);
            for (int p = 0; p < pages; ++p)
                for (int t = 0; t < fastllm::GetPageLen(); ++t)
                    AppendPrefixKeyInt(ctx->prefixCachePageKeys[p], ctx->allTokens[p * fastllm::GetPageLen() + t]);
        }
        for (auto &key : ctx->prefixCachePageKeys) {
            AppendPrefixKeyInt(key, (uint32_t)provenance.size()); key += provenance;
        }
    }
    // Reuse MultiCUDA's global-range mapping, but copy CPU snapshots rather
    // than device allocations. The durable form never contains rank ordinals,
    // expansion capacity, or a device-dependent transposed recurrent matrix.
    // §打包并行：每个 piece 往**自己的** buffer 里写并返回描述符（offset 相对自身），
    // 然后按原顺序拼接、把 offset 平移到最终 buffer —— 字节布局与串行完全一致。
    struct PackPiece { std::vector<char> buffer; Json descriptor; };
    // Opt-in: size the final concatenation once, without changing serializers,
    // piece ordering, descriptors, worker count, hash submission or disk bytes.
    static bool PackFastEnabled() {
        static const bool enabled = [] {
            const char *value = std::getenv("FASTLLM_QWEN35_PACK_FAST");
            return value != nullptr && value[0] == '1' && value[1] == 0;
        }();
        return enabled;
    }
    static void ReservePackMerge(std::vector<char> &packed,
                                 const std::vector<PackPiece> &pieces,
                                 const char *budgetError) {
        if (!PackFastEnabled()) return;
        size_t total = packed.size();
        const size_t limit = HostLimit();
        Check(total <= limit, budgetError);
        for (const auto &piece : pieces) {
            // Existing final budget, checked before summing to avoid overflow.
            Check(piece.buffer.size() <= limit - total, budgetError);
            total += piece.buffer.size();
        }
        packed.reserve(total);
    }
    static int PackWorkers() {
        const char *value = std::getenv("FASTLLM_PREFIX_CACHE_PACK_WORKERS");
        if (value == nullptr || value[0] == 0) return 4;
        const int parsed = std::atoi(value);
        return parsed <= 0 ? 1 : (parsed > 16 ? 16 : parsed);
    }
    template <typename Fn>
    static void ParallelPieces(std::vector<PackPiece> &pieces, Fn &&fn, int workers) {
        const size_t total = pieces.size();
        if (workers <= 1 || total < 2) {
            for (size_t i = 0; i < total; ++i) fn(i, pieces[i]);
            return;
        }
        const size_t count = std::min<size_t>(size_t(workers), total);
        std::vector<std::thread> pool;
        std::vector<std::exception_ptr> failures(count);
        for (size_t worker = 0; worker < count; ++worker) {
            pool.emplace_back([&, worker] {
                const size_t begin = total * worker / count, end = total * (worker + 1) / count;
                try { for (size_t i = begin; i < end; ++i) fn(i, pieces[i]); }
                catch (...) { failures[worker] = std::current_exception(); }
            });
        }
        for (auto &thread : pool) if (thread.joinable()) thread.join();
        for (auto &failure : failures) if (failure) std::rethrow_exception(failure);
    }
    // 描述符是扁平的 {dims,dtype,offset,bytes}：只平移 offset，其他字段原样保留。
    static Json ShiftedOffset(const Json &descriptor, size_t base) {
        std::map<std::string, Json> fields;
        for (const auto &entry : descriptor.object_items()) fields[entry.first] = entry.second;
        fields["offset"] = Json(double(base) + descriptor["offset"].number_value());
        return Json(fields);
    }

    // Byte-equivalent canonical [K,V] packing from a transposed [V,K] snapshot.
    // Compile-time widths remove millions of runtime-size memcpy calls. Tiling
    // bounds both source and destination working sets; no alignment/aliasing
    // assumptions are made about snapshot storage.
    template<size_t Unit>
    static void ChunkTailPackTranspose(char *dst, const char *src, int keys, int values) {
        constexpr int tile = 16;
        for (int kb = 0; kb < keys; kb += tile)
            for (int vb = 0; vb < values; vb += tile)
                for (int k = kb; k < std::min(kb + tile, keys); ++k)
                    for (int v = vb; v < std::min(vb + tile, values); ++v)
                        std::memcpy(dst + (size_t(k) * values + v) * Unit,
                                    src + (size_t(v) * keys + k) * Unit, Unit);
    }
    Json SaveLinear(std::vector<char> &packed, const Qwen35LinearPrefixSnapshotCache &cache) {
        Check(cache.valid && (!cache.multiDevice || !cache.locals.empty()),"ssd_linear_snapshot_missing");
        const auto dims = cache.multiDevice ? cache.tpGlobalDims : cache.single.dims;
        Check((dims.size() == 3 || dims.size() == 4) && dims[0] == 1 && dims[1] > 0,
              "ssd_linear_global_shape");
        const Data &sample = cache.multiDevice ? cache.locals.begin()->second : cache.single;
        const size_t unit = sample.unitSize;
        Check(unit == 2 || unit == 4,"ssd_linear_unit");
        size_t channelBytes = unit;
        for (size_t d = 2; d < dims.size(); ++d) {
            Check(dims[d] > 0 && (size_t)dims[d] <= HostLimit()/channelBytes,"ssd_linear_shape");
            channelBytes *= dims[d];
        }
        Check((size_t)dims[1] <= (HostLimit()-packed.size())/channelBytes,"ssd_linear_pack_budget");
        const size_t offset = packed.size(), bytes = dims[1]*channelBytes;
        packed.resize(offset+bytes);
        std::vector<uint8_t> covered(dims[1],0);
        auto copy = [&](const Data &source, const std::vector<std::pair<int,int>> &ranges) {
            Check(source.cpuData && source.dataDevice == DataDevice::CPU && source.dataType == sample.dataType &&
                  source.dims.size() == dims.size() && source.dims[0] == 1 && source.strides.size() == dims.size() &&
                  source.strides[1]*unit == channelBytes,"ssd_linear_source_layout");
            for (size_t d = 2; d < dims.size(); ++d) Check(source.dims[d] == dims[d],"ssd_linear_source_shape");
            int local = 0;
            for (auto range : ranges) {
                Check(range.first >= 0 && range.second > range.first && range.second <= dims[1],"ssd_linear_range");
                for (int head = range.first; head < range.second; ++head, ++local) {
                    Check(local < source.dims[1] && !covered[head],"ssd_linear_duplicate_range");
                    char *dst = packed.data()+offset+head*channelBytes;
                    const char *src = (const char*)source.cpuData+local*channelBytes;
                    if (source.isLinearAttentionTransposed) {
                        Check(dims.size() == 4,"ssd_linear_transpose_shape");
                        if (Qwen35ChunkSavePackEnabled()) {
                            if (unit == 2) ChunkTailPackTranspose<2>(dst, src, dims[2], dims[3]);
                            else ChunkTailPackTranspose<4>(dst, src, dims[2], dims[3]);
                        } else {
                            for (int k = 0; k < dims[2]; ++k) for (int v = 0; v < dims[3]; ++v)
                                memcpy(dst+((size_t)k*dims[3]+v)*unit,src+((size_t)v*dims[2]+k)*unit,unit);
                        }
                    } else memcpy(dst,src,channelBytes);
                    covered[head] = 1;
                }
            }
            Check(local == source.dims[1],"ssd_linear_local_coverage");
        };
        if (cache.multiDevice) {
            Check(cache.tpAxis == 1 && cache.tpLayout == TP_LAYOUT_SHARDED,"ssd_linear_shard_axis");
            for (auto &item : cache.locals) {
                auto ranges = cache.tpRanges.find(item.first);
                Check(ranges != cache.tpRanges.end(),"ssd_linear_missing_ranges");
                copy(item.second,ranges->second);
            }
        } else copy(cache.single,{{0,dims[1]}});
        Check(std::all_of(covered.begin(),covered.end(),[](uint8_t v){return v != 0;}),"ssd_linear_incomplete");
        return Json::object{{"dims",dims},{"dtype",(int)sample.dataType},
                            {"offset",(double)offset},{"bytes",(double)bytes}};
    }
    void LoadLinear(const std::vector<char> &packed, const Json &value, int layer, int side,
                    Qwen35LinearPrefixSnapshotCache &cache) {
        auto dims = Ints(value["dims"]);
        const std::vector<int> expected = side == 0 ?
            std::vector<int>{1,model->num_k_heads*model->head_k_dim*2+model->num_v_heads*model->head_v_dim,4} :
            std::vector<int>{1,model->num_v_heads,model->head_k_dim,model->head_v_dim};
        Check(dims == expected,"ssd_linear_target_shape");
        const int type = value["dtype"].int_value();
        Check(type == (int)DataType::FLOAT32 || type == (int)DataType::FLOAT16 || type == (int)DataType::BFLOAT16,
              "ssd_linear_dtype");
        cache.dataType = (DataType)type;
        const size_t unit = type == (int)DataType::FLOAT32 ? 4 : 2;
        size_t channelBytes = unit;
        for (size_t d = 2; d < dims.size(); ++d) {
            Check(dims[d] > 0 && (size_t)dims[d] <= HostLimit()/channelBytes,"ssd_linear_shape");
            channelBytes *= dims[d];
        }
        Check(dims[1] > 0 && (size_t)dims[1] <= HostLimit()/channelBytes,"ssd_linear_shape");
        const size_t bytes = dims[1]*channelBytes;
        Check(value["offset"].is_number() && value["offset"].number_value() >= 0 &&
              value["offset"].number_value() <= (double)packed.size(),"ssd_linear_offset");
        const size_t offset = (size_t)value["offset"].number_value();
        Check((double)offset == value["offset"].number_value() && bytes == value["bytes"].number_value() &&
              bytes <= packed.size()-offset,"ssd_linear_bounds");
        cache.dataDeviceIds = model->threadTpPreparedDevices;
        if (cache.dataDeviceIds.empty()) {
            std::map<int,int> ratios;
            Check(GetQwen35GPUForwardDevices(model->deviceMap,cache.dataDeviceIds,ratios) &&
                  !cache.dataDeviceIds.empty(),"ssd_linear_devices");
        }
        cache.multiDevice = cache.dataDeviceIds.size() > 1;
        cache.dims = dims; cache.tpGlobalDims = dims; cache.transposed = false;
        cache.tpAxis = cache.multiDevice ? 1 : -1;
        cache.tpLayout = cache.multiDevice ? TP_LAYOUT_SHARDED : TP_LAYOUT_NONE;
        if (cache.multiDevice) {
            const auto &schemes = side == 0 ? model->threadTpLinearConvSchemes : model->threadTpLinearValueHeadSchemes;
            Check(layer < (int)schemes.size(),"ssd_linear_target_scheme");
            cache.tpRanges = schemes[layer];
        }
        for (int device : cache.dataDeviceIds) {
            auto ranges = cache.multiDevice ? cache.tpRanges[device] : std::vector<std::pair<int,int>>{{0,dims[1]}};
            int channels = 0;
            for (auto range : ranges) {
                Check(range.first >= 0 && range.second > range.first && range.second <= dims[1],"ssd_linear_target_range");
                channels += range.second-range.first;
            }
            Check(channels > 0 && channels <= dims[1],"ssd_linear_target_channels");
            Data &target = cache.multiDevice ? cache.locals[device] : cache.single;
            auto localDims = dims; localDims[1] = channels;
            target.dataType = cache.dataType; target.UpdateUnitSize(); target.Resize(localDims); target.Allocate();
            target.isKVCache = true; target.isLinearAttention = true; target.isLinearAttentionTransposed = false;
            size_t local = 0;
            for (auto range : ranges) {
                const size_t count = (range.second-range.first)*channelBytes;
                memcpy(target.cpuData+local,packed.data()+offset+range.first*channelBytes,count); local += count;
            }
        }
        cache.valid = true;
    }

    void Work(size_t lane);
    void Export(Job &job, void *buffer, size_t lane);
    void UpdatePrefetchHint(const Job &job) {
        if (!job.prefetchContext.is_object()) return;
        std::lock_guard<std::mutex> guard(prefetchHintMutex);
        const std::string metadataKey = "restart-prefix-v1:" + identity;
        auto hints = store->GetMetadata(metadataKey)["hints"].array_items();
        const auto &newTokens = job.prefetchContext["tokens"].array_items();
        const auto &newKeys = job.prefetchContext["keys"].array_items();
        hints.erase(std::remove_if(hints.begin(), hints.end(), [&](const Json &old) {
            const auto &oldTokens = old["context"]["tokens"].array_items();
            const auto &oldKeys = old["context"]["keys"].array_items();
            return oldTokens.size() <= newTokens.size() && oldKeys.size() <= newKeys.size() &&
                std::equal(oldTokens.begin(),oldTokens.end(),newTokens.begin()) &&
                std::equal(oldKeys.begin(),oldKeys.end(),newKeys.begin()) &&
                old["wall_ns"].string_value() <= job.hintWallNs;
        }), hints.end());
        hints.emplace_back(Json::object{{"key",job.key},{"length",job.length},{"identity",identity},
            {"context",job.prefetchContext},{"wall_ns",job.hintWallNs}});
        std::stable_sort(hints.begin(), hints.end(), [](const Json &a, const Json &b) {
            return a["wall_ns"].string_value() > b["wall_ns"].string_value();
        });
        if (hints.size() > 8) hints.resize(8);
        while (hints.size() > 1 && Json(hints).dump().size() > (8 << 20)) hints.pop_back();
        store->PutMetadata(metadataKey, Json::object{{"hints",hints}});
    }
    bool Import(ResponseContext *ctx, const DiskPrefixCache::Checkpoint &candidate,
                DiskPrefixCache::ReadSession *preloaded = nullptr,
                const Json *loadedState = nullptr,
                const std::function<bool()> &keepGoing = {});
    void PrefetchWork();
    static std::string EncodeKey(const std::string &key) {
        static const char *hex = "0123456789abcdef";
        std::string out; out.reserve(key.size()*2);
        for (unsigned char c : key) { out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]); }
        return out;
    }
    static std::string DecodeKey(const std::string &key) {
        Check(key.size()%2 == 0,"ssd_prefetch_key_encoding");
        auto nibble = [](char c) { return c >= '0' && c <= '9' ? c-'0' :
            c >= 'a' && c <= 'f' ? c-'a'+10 : -1; };
        std::string out; out.reserve(key.size()/2);
        for (size_t i = 0; i < key.size(); i += 2) {
            const int a = nibble(key[i]), b = nibble(key[i+1]);
            Check(a >= 0 && b >= 0,"ssd_prefetch_key_encoding");
            out.push_back(char(a*16+b));
        }
        return out;
    }
    static Json SavePrefetchContext(const ResponseContext *ctx, int length = -1) {
        if (length < 0) length = ctx->allTokens.size();
        Json::array keys, baseKeys, spans;
        for (int p = 0; p < length / fastllm::GetPageLen() && p < (int)ctx->prefixCachePageKeys.size(); ++p)
            keys.emplace_back(EncodeKey(ctx->prefixCachePageKeys[p]));
        for (int p = 0; p < length / fastllm::GetPageLen() && p < (int)ctx->prefixCacheBasePageKeys.size(); ++p)
            baseKeys.emplace_back(EncodeKey(ctx->prefixCacheBasePageKeys[p]));
        for (auto &span : ctx->multimodalPrefixSpans) {
            if (span.begin >= length) continue;
            Json::array digest;
            for (auto byte : span.digest) digest.emplace_back((int)byte);
            spans.emplace_back(Json::object{{"begin",span.begin},{"end",std::min(span.end,length)},{"digest",digest}});
        }
        std::vector<int> positions;
        const int positionLength = ctx->prefixCachePositionIds.size() / 3;
        if (positionLength > 0) for (int axis = 0; axis < 3; ++axis)
            positions.insert(positions.end(), ctx->prefixCachePositionIds.begin() + axis * positionLength,
                ctx->prefixCachePositionIds.begin() + axis * positionLength + std::min(length,positionLength));
        std::vector<int> tokens(ctx->allTokens.begin(),ctx->allTokens.begin()+length);
        return Json::object{{"tokens",IntArray(tokens)},
            {"keys",keys},{"base_keys",baseKeys},{"positions",IntArray(positions)},
            {"spans",spans},{"media",ctx->HasMultimodalPrefixIdentity()},
            {"provenance",ctx->prefixCacheProvenance},{"output_logits",ctx->generationConfig.output_logits}};
    }
    static void LoadPrefetchContext(const Json &value, ResponseContext &ctx) {
        ctx.allTokens = Ints(value["tokens"]);
        ctx.currentTokens = ctx.allTokens;
        ctx.inputTokens = ctx.allTokens.size();
        for (auto &key : value["keys"].array_items()) ctx.prefixCachePageKeys.push_back(DecodeKey(key.string_value()));
        for (auto &key : value["base_keys"].array_items()) ctx.prefixCacheBasePageKeys.push_back(DecodeKey(key.string_value()));
        ctx.prefixCachePositionIds = Ints(value["positions"]);
        ctx.prefixCacheProvenance = value["provenance"].string_value();
        ctx.generationConfig.output_logits = value["output_logits"].bool_value();
        for (auto &row : value["spans"].array_items()) {
            MultimodalPrefixSpan span;
            span.begin = row["begin"].int_value(); span.end = row["end"].int_value();
            auto digest = Ints(row["digest"]);
            Check(digest.size() == span.digest.size(),"ssd_prefetch_media_identity");
            for (size_t i = 0; i < digest.size(); ++i) {
                Check(digest[i] >= 0 && digest[i] <= 255,"ssd_prefetch_media_identity");
                span.digest[i] = digest[i];
            }
            ctx.multimodalPrefixSpans.push_back(span);
        }
        if (value["media"].bool_value()) ctx.multimodalInput["cache_identity"] = {};
    }
public:
    const std::string &EncoderIdentity() const { return encoderIdentity; }
    void Initialize() {
        Check(Ensure(),"ssd_startup_unavailable");
        // 【第 2 步-2：挂搬运钩子】
        // 页池不认识内存层，模型层把自己的搬运函数注册进去（照 vLLM OffloadingManager 摆位）。
        // 开关关 ⇒ 不注册 ⇒ 零行为变更。
        model->InstallPagedSpillHooks();    }
    void StartPrefetch() {
        if (policy == "never" || !Qwen35LinearPrefixCacheEnabled() ||
            !Ensure() || prefetchWorker.joinable()) return;
        prefetchWorker = std::thread(&Qwen35PersistentCache::PrefetchWork, this);
    }
    void PrefetchReady() {
        { std::lock_guard<std::mutex> lock(prefetchMutex); prefetchReady = true; }
        prefetchChanged.notify_all();
    }
    void ObservePrefill(int tokens, double ms) {
        if (tokens <= 0 || ms <= 0 || !std::isfinite(ms)) return;
        std::lock_guard<std::mutex> lock(mutex);
        const double sample = ms/tokens;
        prefillMsPerToken = prefillMsPerToken > 0 ? 0.8*prefillMsPerToken+0.2*sample : sample;
    }
    void SaveCost() {
        if (!store) return;
        Json cost;
        { std::lock_guard<std::mutex> lock(mutex);
          cost = Json::object{{"device",std::to_string(storageDevice)},
              {"prefill",prefillMsPerToken},{"native_read",nativeReadMsPerToken},
              {"converted_read",convertedReadMsPerToken}}; }
        // Calibration loss is a safe auto miss, never a correctness failure.
        try { store->PutMetadata("cost:"+identity,cost); }
        catch (const std::exception &e) { std::fprintf(stderr,"[Prefix SSD] cost profile skipped: %s.\n",e.what()); }
    }
    static std::shared_ptr<Qwen35PersistentCache> Create(Qwen3_5Model *model) {
        if (Env("FASTLLM_PREFIX_CACHE_DIR").empty()) return {};
        Check(PersistentPrefixStorageConfigSet() &&
                  GetPersistentPrefixStorageConfig().writerFormat == kStorageFormatV4,
              "ssd_v4_config_missing");
        return std::shared_ptr<Qwen35PersistentCache>(new Qwen35PersistentCache(model));
    }
    explicit Qwen35PersistentCache(Qwen3_5Model *model) : model(model),
        directory(Env("FASTLLM_PREFIX_CACHE_DIR")),baseFamily(Env("FASTLLM_PREFIX_CACHE_FAMILY_IDENTITY")),
        encoderIdentity(Env("FASTLLM_PREFIX_CACHE_ENCODER_IDENTITY")),
        configuredIdentity(Env("FASTLLM_PREFIX_CACHE_IDENTITY")),
        draftIdentity(Env("FASTLLM_PREFIX_CACHE_DRAFT_IDENTITY")),
        policy(Env("FASTLLM_PREFIX_CACHE_RESTORE_POLICY")),
        startupManifest(DiskPrefixCache::StartupManifest()) {
        Check(DiskPrefixCache::IsDigest(baseFamily), "ssd_invalid_family");
        auto quota = Env("FASTLLM_PREFIX_CACHE_DISK_BYTES");
        if (!quota.empty()) diskLimit = std::stoull(quota);
        // auto/always use ordered local-first, request-scoped SSD restoration.
        // never disables SSD retrieval; it does not disable background writes.
        if (policy.empty()) policy = "auto";
        Check(policy == "auto" || policy == "always" || policy == "never",
              "ssd_invalid_restore_policy");
        const std::string timeout = Env("FASTLLM_PREFIX_CACHE_RESTORE_TIMEOUT_MS");
        if (!timeout.empty()) {
            try {
                size_t parsedChars = 0;
                const long long parsed = std::stoll(timeout,&parsedChars);
                Check(parsedChars == timeout.size() && parsed > 0 && parsed <= INT_MAX,
                      "ssd_invalid_restore_timeout");
                restoreTimeoutMs = (int)parsed;
            } catch (const std::exception &) {
                Check(false, "ssd_invalid_restore_timeout");
            }
        }
    }
    ~Qwen35PersistentCache() { Stop(); }
    void Created(ResponseContext *ctx) {
        if (!ctx) return;
        std::shared_ptr<DiskPrefixCache::ReadSession> released;   // unpinned after the lock
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto &lifecycle = requestStates[ctx];
            if (lifecycle) lifecycle->cancelled.store(true, std::memory_order_release);
            lifecycle = std::make_shared<RequestLeaseState>();
            lifecycle->request_id = ctx->requestId;
            lifecycle->generation.store(
                requestGenerationCounter.fetch_add(1, std::memory_order_acq_rel) + 1,
                std::memory_order_release);
            exports.erase(ctx);
            auto hold = requestHolds.find(ctx);
            if (hold != requestHolds.end()) { released = std::move(hold->second); requestHolds.erase(hold); }
        }
        // PrepareMultimodalPrefixCache (or the common text planner) owns the
        // canonical positions and page keys.  Do not erase them here: doing so
        // used to drop MRoPE coordinates immediately after they were built and
        // made the SSD key disagree with the in-memory key.
        if (!ctx->HasMultimodalPrefixIdentity() && ctx->prefixCachePageKeys.empty()) {
            // Keep the historical text-only RAM trie key as a compatibility
            // fallback for callers that do not run the common planner.
            std::string parent = "qwen35-text-prefix-v2";
            const int pageLen = fastllm::GetPageLen();
            for (size_t start = 0; start + pageLen <= ctx->allTokens.size(); start += pageLen) {
                std::string content = parent;
                for (int offset = 0; offset < pageLen; ++offset)
                    AppendPrefixKeyInt(content,ctx->allTokens[start+offset]);
                parent = DiskPrefixCache::Digest(content);
                ctx->prefixCachePageKeys.push_back(parent);
            }
        }
        ctx->prefixCacheBasePageKeys = ctx->prefixCachePageKeys;
        ctx->prefixCacheProvenance.clear();
    }
    void Replanned(ResponseContext *ctx) {
        ctx->prefixCacheBasePageKeys = ctx->prefixCachePageKeys;
        SetProvenance(ctx,ctx->prefixCacheProvenance);
    }
    void Removed(ResponseContext *ctx) {
        if (!ctx) return;
        std::shared_ptr<DiskPrefixCache::ReadSession> released;
        if (ctx->intParams["ssd_loaded_len"] > 0) {
            std::fprintf(stderr,"[Prefix SSD] restore not adopted: loaded_tokens=%d cached_tokens=%d reason=scheduler_did_not_confirm.\n",
                         ctx->intParams["ssd_loaded_len"],ctx->cacheLen);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto lifecycle = requestStates.find(ctx);
            if (lifecycle != requestStates.end() && lifecycle->second) {
                lifecycle->second->cancelled.store(true, std::memory_order_release);
                lifecycle->second->generation.fetch_add(1, std::memory_order_acq_rel);
            }
            // Accepted tasks retain their own lifecycle shared_ptr; completed
            // contexts must not accumulate in the admission registry.
            requestStates.erase(ctx);
            exports.erase(ctx);
            auto hold = requestHolds.find(ctx);
            if (hold != requestHolds.end()) { released = std::move(hold->second); requestHolds.erase(hold); }
        }
        released.reset();   // V5 pin deleter takes the store lock: never under `mutex`
        CancelRead(ctx);
    }
    void CancelRead(ResponseContext *ctx) {
        {
            std::lock_guard<std::mutex> lock(prefetchMutex);
            auto found = restoreTickets.find(ctx);
            if (found != restoreTickets.end()) {
                found->second->cancelled.store(true, std::memory_order_release);
                found->second->consumed.store(true, std::memory_order_release);
                // Cancelling a restore ticket does not cancel an independent
                // save job belonging to the same request lifecycle.
                restoreTickets.erase(found);
            }
        }
        prefetchChanged.notify_all();
    }
    void Stop() {
        bool expected = false;
        if (!shutdownStarted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return;
        // Shutdown is a state machine: close admission before cancelling producers.
        CloseAdmission();
        Qwen35StateStageClose(model);
        std::map<ResponseContext*, std::shared_ptr<DiskPrefixCache::ReadSession>> holds;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            holds.swap(requestHolds);
        }
        holds.clear();
        Qwen35StopSpillPrefetch(model->pagedSpillState);
        {
            std::lock_guard<std::mutex> lock(prefetchMutex);
            prefetchStopping = true;
            for (auto &ticket : restoreTickets) {
                ticket.second->cancelled.store(true, std::memory_order_release);
                ticket.second->consumed.store(true, std::memory_order_release);
                if (ticket.second->request_state)
                    ticket.second->request_state->cancelled.store(true, std::memory_order_release);
            }
        }
        prefetchChanged.notify_all();
        changed.notify_all();
        const bool modelDrain = Wait(kDrainTimeoutMs);
        if (!modelDrain) shutdownFailed.store(true, std::memory_order_release);
        if (prefetchWorker.joinable()) prefetchWorker.join();
        for (auto &thread : workers) if (thread.joinable()) thread.join();
        workers.clear();
        {
            std::lock_guard<std::mutex> lock(prefetchMutex);
            requestReads.clear();
            prefetchQueue.clear();
            restoreTickets.clear();
        }
        for (auto &lane : writeLanes) {
            if (lane && lane->d2Stream) lane->d2Stream->Stop();
            if (lane && lane->hashPool) lane->hashPool->Stop();
        }
        std::string storeError;
        if (store && !store->DrainPending(storeError)) {
            shutdownFailed.store(true, std::memory_order_release);
            std::fprintf(stderr, "[Prefix SSD] shutdown-store-drain-failed: reason=%s.\n",
                         storeError.empty() ? "unknown" : storeError.c_str());
        }
        if (selfCheckWorker.joinable()) selfCheckWorker.join();
        try {
            if (store) store->SelfCheckStorage();
        } catch (const std::exception &error) {
            shutdownFailed.store(true, std::memory_order_release);
            std::fprintf(stderr, "[Prefix SSD] shutdown-self-check-failed: reason=%s.\n", error.what());
        } catch (...) {
            shutdownFailed.store(true, std::memory_order_release);
            std::fprintf(stderr, "[Prefix SSD] shutdown-self-check-failed: reason=unknown.\n");
        }
        bool queuesEmpty = true;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto &queue : laneQueues) if (!queue.empty()) queuesEmpty = false;
        }
        const uint64_t remainingCredits = coordinator ? coordinator->HostCreditsInUse() : 0;
        const uint64_t storeInflight = store ? store->InFlightBytes() : 0;
        const bool clean = modelDrain && queuesEmpty && inflightJobs == 0 &&
            pendingCopy == 0 && storeInflight == 0 && remainingCredits == 0 &&
            admittedCheckpoints.load() == terminalCheckpoints.load() &&
            failedRetainedBytes <= kMaxFailedBytes && failedJobs.size() <= kMaxFailedRetained &&
            detachedTaskCount == 0 && !selfCheckRunning.load(std::memory_order_acquire) &&
            degraded.load() == 0 && !shutdownFailed.load(std::memory_order_acquire);
        if (clean) shutdownClean.store(true, std::memory_order_release);
        else shutdownFailed.store(true, std::memory_order_release);
        std::fprintf(stderr, "[Prefix SSD] write-shutdown: state=%s admitted=%llu terminal=%llu "
                     "credit_balance=%llu store_inflight=%llu inflight_jobs=%zu pending_copy=%zu "
                     "failed_retained=%zu failed_bytes=%zu detached_tasks=%llu degraded=%llu.\n",
                     clean ? "SHUTDOWN_CLEAN" : (modelDrain ? "SHUTDOWN_FAILED" : "SHUTDOWN_TIMEOUT"),
                     (unsigned long long)admittedCheckpoints.load(),
                     (unsigned long long)terminalCheckpoints.load(),
                     (unsigned long long)remainingCredits, (unsigned long long)storeInflight,
                     inflightJobs, pendingCopy, failedJobs.size(), failedRetainedBytes,
                     (unsigned long long)detachedTaskCount,
                     (unsigned long long)degraded.load());
        fflush(stderr);
        SaveCost();
    }
    void CloseAdmission() { admissionClosed = true; }
    // Called only with mutex held and debug logging enabled. Unlike the
    // export-profile sample (taken before the current job releases its credit),
    // this idle sample reports current ownership after worker completion.
    void PrintIdleWatermarkLocked() const {
        uint64_t pinned = 0, poolInflight = 0, d2Pending = 0, d2Slots = 0;
        for (const auto &entry : pinnedGpuBytes) pinned += entry.second;
        for (const auto &lane : writeLanes) {
            if (!lane) continue;
            if (lane->hashPool) poolInflight += lane->hashPool->InFlightBytes();
            if (lane->d2Stream) {
                d2Pending += lane->d2Stream->Pending();
                d2Slots += lane->d2Stream->SlotsInUse();
            }
        }
        const uint64_t credits = coordinator ? coordinator->HostCreditsInUse() : 0;
        const uint64_t storeInflight = store ? store->InFlightBytes() : 0;
        // An all-zero sample carries no information: during model load and idle serving it
        // only floods the debug log (it used to print every ~500ms forever). Keep the sample
        // whenever anything is actually in flight or a checkpoint is still outstanding.
        if (queuedJobs == 0 && activeWorkers == 0 && credits == 0 && pinned == 0 &&
            exportProtectedBytes == 0 && queuedStateBytes == 0 && poolInflight == 0 &&
            d2Pending == 0 && d2Slots == 0 && storeInflight == 0 &&
            admittedCheckpoints.load() == terminalCheckpoints.load() &&
            expectedCheckpoints.load() == terminalCheckpoints.load() &&
            degraded.load() == 0) {
            return;
        }
        PrefixCacheDebug("[Prefix SSD] ssd-watermark: event=idle_sample queued=%zu "
                         "working=%d host_credits=%llu pinned_gpu_bytes=%llu "
                         "export_protected_current_bytes=%llu queued_state_bytes=%zu "
                         "pool_inflight=%llu d2_pending=%llu d2_slots=%llu "
                         "store_inflight=%llu admitted=%llu terminal=%llu "
                         "credit_balance=%llu expected_checkpoints=%llu degraded=%llu.\n",
                         queuedJobs, activeWorkers,
                         (unsigned long long)credits, (unsigned long long)pinned,
                         (unsigned long long)exportProtectedBytes, queuedStateBytes,
                         (unsigned long long)poolInflight,
                         (unsigned long long)d2Pending, (unsigned long long)d2Slots,
                         (unsigned long long)storeInflight,
                         (unsigned long long)admittedCheckpoints.load(),
                         (unsigned long long)terminalCheckpoints.load(),
                         (unsigned long long)credits,
                         (unsigned long long)expectedCheckpoints.load(),
                         (unsigned long long)degraded.load());
        fflush(stdout);
    }
    bool Wait(int timeoutMs) {
        // §13 第 4 条：drain 必须等到队列、GPU lease、host 信用全部归零；
        // 第 6 条：超时显式失败并保留证据，不无限等、不打印成功。
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(std::max(0,timeoutMs));
        std::unique_lock<std::mutex> lock(mutex);
        // RCA 1（2026-10-03，真机 merge-verify-a1 实证）：旧实现**只算一次条件**就 wait_until 到
        // deadline，醒来**不复查**直接判失败；而 `InFlightBytes()` 内部只做**一次**
        // `PruneFinishedHead()`（旧版每次只弹 1 个已完成队首）⇒ FIFO 深度为 2 时
        // "结束时残留 2 笔已完成作业"这一常见形态**必然误报超时**
        // （实测 455579196 − 228850590 = 226728606 恰为最后两笔作业之和）。
        // 现在改为**有界短轮询**：每轮重新计算条件（`InFlightBytes()` 会继续弹掉已完成队首），
        // 并在**宣布失败之前再复查一遍**，杜绝陈旧读数导致的假超时。
        // 注意：`storeInflight == 0` 这道门**不放宽**（否则会静默丢未落盘数据）。
        struct DrainDiag { uint64_t pinned = 0, credits = 0, storeInflight = 0; bool d2Idle = true; };
        DrainDiag diag;
        auto drainIdleNow = [&]() -> bool {
            diag = DrainDiag{};
            for (const auto &entry : pinnedGpuBytes) if (entry.second > 0) diag.pinned += (uint64_t)entry.second;
            diag.credits = coordinator ? coordinator->HostCreditsInUse() : 0;
            // §④ 宽版：**每个 lane** 的 D2-stream 待拷贝/在途槽位都必须归零，否则退出时可能丢在途拷贝。
            for (auto &lane : writeLanes) {
                if (lane && lane->d2Stream &&
                    (lane->d2Stream->Pending() != 0 || lane->d2Stream->SlotsInUse() != 0)) {
                    diag.d2Idle = false;
                    break;
                }
            }
            // S1-3 修复（2026-10-02，对标审计 P0-B）：drain 必须覆盖 store 的"已受理未落盘"。
            // 旧条件只看模型层队列/信用/钉住/D2，而 worker 把活交给 store 之后就把自己在途计数
            // 清了，于是 Wait 可能在 store 备菜 FIFO 仍有未 durable 检查点时报告"排空成功" ⇒ 关停
            // 静默丢数据。注意 InFlightBytes() 内部会做一次非阻塞收尾（PruneFinishedHead），
            // 不会把已完成作业算在途。
            diag.storeInflight = store ? store->InFlightBytes() : 0;
            return queuedJobs == 0 && activeWorkers == 0 && diag.credits == 0 && diag.pinned == 0 &&
                   diag.d2Idle && diag.storeInflight == 0;
        };
        auto finishDrain = [&]() {
            lock.unlock();
            const Json profile = store ? store->StorageProfile(false) : Json(Json::object{});
            const uint64_t failed = (uint64_t)profile["failed_checkpoints_cum"].number_value();
            const bool ok = writeFailures.load() == 0 && degraded.load() == 0 && failed == 0;
            if (ok) PrefixCacheInfo("[Prefix SSD] pending writes saved.\n");
            if (!ok || PrefixCacheDebugEnabled())
            std::fprintf(stderr,"[Prefix SSD] drain %s: durable=%.0f reused=%.0f failed=%llu "
                        "write_failures=%llu degraded=%llu.\n",ok ? "complete" : "failed",
                        profile["durable_checkpoints_cum"].number_value(),
                        profile["reused_checkpoints_cum"].number_value(),
                        (unsigned long long)failed,(unsigned long long)writeFailures.load(),
                        (unsigned long long)degraded.load());
            SaveCost();
            return ok;
        };
        constexpr int kDrainPollMs = 200;
        for (;;) {
            if (drainIdleNow()) return finishDrain();
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                // 宣布失败前**复查一遍**（收尾可能刚好完成，含 InFlightBytes 内部再弹一次队首）。
                if (drainIdleNow()) return finishDrain();
                std::fprintf(stderr,"[Prefix SSD] drain timeout: queued=%zu working=%d host_credits=%llu pinned_gpu_bytes=%llu store_inflight_bytes=%llu lanes=%d admitted=%llu terminal=%llu degraded=%llu.\n",
                             queuedJobs,(int)activeWorkers,(unsigned long long)diag.credits,
                             (unsigned long long)diag.pinned,
                             (unsigned long long)diag.storeInflight, writeWorkerCount,
                             (unsigned long long)admittedCheckpoints.load(),
                             (unsigned long long)terminalCheckpoints.load(),
                             (unsigned long long)degraded.load());
                lock.unlock(); SaveCost(); return false;
            }
            // 有界短睡：醒来不判失败，回到循环顶部重新算条件。
            changed.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(kDrainPollMs)));
        }
    }
    Json CapacityObservation() const {
        return store ? store->CapacityObservation() : Json::object{{"state", "disabled"}};
    }
    RequestStatusFields MemoryObservation() {
        RequestStatusFields fields;
        std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
        if (!lock.owns_lock()) return {{"ssd_writer_memory_state", "busy"}};
        fields["ssd_writer_memory_state"] = "observed";
        fields["ssd_queued_jobs"] = std::to_string(queuedJobs);
        fields["ssd_working_jobs"] = std::to_string(activeWorkers);
        fields["ssd_queued_state_bytes"] = std::to_string(queuedStateBytes);
        fields["ssd_export_protected_bytes"] = std::to_string(exportProtectedBytes);
        fields["ssd_export_contexts"] = std::to_string(exports.size());
        fields["ssd_host_credit_bytes"] = std::to_string(coordinator ? coordinator->HostCreditsInUse() : 0);
        fields["ssd_admitted_checkpoints"] = std::to_string(admittedCheckpoints.load());
        fields["ssd_terminal_checkpoints"] = std::to_string(terminalCheckpoints.load());
        fields["ssd_writer_sample_ns"] = std::to_string(RequestStatusNowNs());
        return fields;
    }
    std::string Stats() const {
        Json capacity;
        if (store) {
            try { capacity = store->CapacityStatus(); }
            catch (const std::exception &e) { capacity = Json::object{{"unavailable",e.what()}}; }
        }
        return Json(Json::object{{"written_bytes",(double)writes.load()},{"read_bytes",(double)reads.load()},
            {"committed",(double)committed.load()},{"restored_tokens",(double)restored.load()},
            {"skipped",(double)skipped.load()},{"storage",capacity},
            {"prefetch_queued",(double)prefetchQueued.load()},
            {"prefetch_completed",(double)prefetchCompleted.load()},
            {"prefetch_misses",(double)prefetchMisses.load()}}).dump();
    }
    void Capture(ResponseContext *ctx, const std::shared_ptr<Qwen35LinearPrefixSnapshot> &snapshot);
    int AdoptParent(Job &job);
    void Prepare(ResponseContext *ctx);
    void DescribePrefix(ResponseContext *ctx, int length) {
        auto &record = ctx->prefixRestore;
        record.length = length;
        record.identity = identity.empty() ? configuredIdentity : identity;
        record.provenance = ctx->prefixCacheProvenance;
        record.key = DiskPrefixCache::Digest(Prefix(ctx,length)+record.provenance);
        record.requiredComponents = PrefixCacheRecord::FullAttentionKV;
        if (Qwen35HasLinearAttentionLayers(model,model->block_cnt))
            record.requiredComponents |= PrefixCacheRecord::LinearState;
        if (model->RequiresMtpPrefixSnapshot(ctx) || model->RequiresDFlashPrefixSnapshot(ctx))
            record.requiredComponents |= PrefixCacheRecord::SpeculativeState;
    }
    void PublishPrefixRecord(ResponseContext *ctx) {
        if (ctx == nullptr || ctx->prefixRestore.key.empty()) return;
        try {
            static_cast<PrefixCacheRecord&>(ctx->prefixRestore) =
                prefixLedger->Observe(static_cast<const PrefixCacheRecord&>(ctx->prefixRestore));
        } catch (const std::exception &error) {
            // A metadata bookkeeping failure must not damage already verified
            // model state or abort an otherwise usable inference request.
            std::fprintf(stderr,"[PrefixCache] ledger update failed: reason=%s.\n",error.what());
        }
    }
    size_t LoadImage(const std::string &key, size_t expected, std::vector<float> &output) {
        if (policy == "never" || !Ensure()) return 0;
        try {
            const auto readBegin = Clock::now();
            auto reader = imageStore->BeginRead();
            auto state = reader.Load(DiskPrefixCache::Digest(encoderIdentity+key));
            Check(state["count"].number_value() == (double)expected && expected <= HostLimit()/sizeof(float),"ssd_image_shape");
            std::vector<float> data(expected);
            reader.ReadBytes(state["chunks"],expected*sizeof(float),[&](size_t offset,const char *p,size_t n) {
                memcpy((char*)data.data()+offset,p,n);
            });
            output.insert(output.end(),data.begin(),data.end()); reads += reader.ReadBytesCount();
            const double readMs = std::chrono::duration<double,std::milli>(
                Clock::now() - readBegin).count();
            const double speed = readMs > 0 ? expected * sizeof(float) /
                1048576.0 * 1000.0 / readMs : 0.0;
            PrefixCacheInfo("[Vision] Image SSD cache hit (%zu bytes).\n", expected*sizeof(float));
            FastllmEvent("vision", 0, "vision_cache_restore",
                         "tier=L3(SSD) bytes=%zu vision_ssd_load_mib_s=%.2f",
                         expected * sizeof(float), speed);
            return expected;
        } catch (const std::exception &) { return 0; }
    }
    void SaveImage(const std::string &key,const float *data,size_t count) {
        if (!Ensure()) return;
        if (count > HostLimit()/sizeof(float)) {
            ++skipped;
            std::fprintf(stderr,"[Prefix SSD] write_skipped: kind=encoder reason=encoder_too_large state_bytes=%zu host_limit=%zu.\n",
                         count*sizeof(float),HostLimit());
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);
        // P3 修复 B：把 store 在途字节也计入主机预算（只记账，不改其它行为）。
        const uint64_t storeInflight = store ? store->InFlightBytes() : 0;
        const bool overBudget = (uint64_t)(count*sizeof(float)) + (uint64_t)queuedStateBytes + storeInflight > (uint64_t)HostLimit();
        if (stopping || inflightJobs >= kMaxInflightJobs || queuedJobs >= kMaxSnapshotQueue || overBudget) {
            ++skipped;
            const char *reason = stopping ? "stopping" :
                inflightJobs >= kMaxInflightJobs ? "max_inflight_jobs" :
                queuedJobs >= kMaxSnapshotQueue ? "max_snapshot_queue" : "host_budget";
            std::fprintf(stderr,"[Prefix SSD] write_skipped: kind=encoder reason=%s queued_jobs=%zu queue_limit=16 state_bytes=%zu queued_state_bytes=%zu store_inflight_bytes=%llu host_limit=%zu.\n",
                         reason,queuedJobs,count*sizeof(float),queuedStateBytes,(unsigned long long)storeInflight,HostLimit());
            return;
        }
        Job job; job.key = DiskPrefixCache::Digest(encoderIdentity+key);
        job.image = std::make_shared<std::vector<float>>(data,data+count); job.stateBytes = count*sizeof(float);
        job.enqueuedAt = Clock::now();
        // 图片作业没有 ExportState（不走请求亲和），按轮转分发到各 lane。
        const size_t lane = writeWorkerCount > 0 ? (nextLane++ % size_t(writeWorkerCount)) : 0;
        queuedStateBytes += job.stateBytes; ++queuedJobs; ++inflightJobs;
        laneQueues[lane].push_back(std::move(job)); changed.notify_all();
    }
    void Restored(ResponseContext *ctx) {
        if (ctx->cacheLen <= 0) return;
        // A hot request updates existing restart metadata through the same
        // background lane, without reading/writing SSD on the request.
        Job hint;
        hint.hintOnly = true; hint.length = ctx->cacheLen;
        hint.prefetchContext = SavePrefetchContext(ctx, ctx->cacheLen);
        hint.key = DiskPrefixCache::Digest(Prefix(ctx,ctx->cacheLen) + ctx->prefixCacheProvenance);
        hint.hintWallNs = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!stopping && !admissionClosed.load() &&
                inflightJobs < kMaxInflightJobs && queuedJobs < kMaxSnapshotQueue) {
                laneQueues[nextLane++ % size_t(writeWorkerCount)].push_back(std::move(hint));
                ++queuedJobs; ++inflightJobs; changed.notify_all();
            } else {
                std::fprintf(stderr, "[Prefix SSD] hint-rejected: status=REJECTED_BEFORE_ADMISSION error=snapshot_queue_full.\n");
            }
        }
        const bool converted = ctx->intParams["ssd_loaded_len"] > 0 && ctx->intParams["ssd_source_fp16"] &&
            model->kvCacheDataType == DataType::FP8_E4M3;
        if (ctx->intParams["ssd_loaded_len"] <= 0 || converted) {
            std::shared_ptr<Qwen35LinearPrefixSnapshot> snapshot;
            { std::lock_guard<std::mutex> guard(Qwen35LinearPrefixSnapshotsMutex());
              auto &items = Qwen35LinearPrefixSnapshots()[model];
              for (auto &item : items) {
                  if (item->cachedLen != ctx->cacheLen || item->tokens.size() > ctx->allTokens.size() ||
                      !std::equal(item->tokens.begin(),item->tokens.end(),ctx->allTokens.begin())) continue;
                  if (!Qwen35SnapshotMatchesContext(*item, ctx)) continue;
                  snapshot = item; break;
              }
            }
            if (converted) ctx->intParams["ssd_force_checkpoint"] = ctx->cacheLen;
            if (snapshot && (converted || ctx->intParams["ssd_branch_boundary"] == ctx->cacheLen)) Capture(ctx,snapshot);
        }
        auto &status = ctx->prefixRestore;
        const int loaded = ctx->intParams["ssd_loaded_len"];
        const auto tier = loaded > 0 ? PrefixCacheRestoreStatus::SSD :
            status.tier == PrefixCacheRestoreStatus::Host ? PrefixCacheRestoreStatus::Host :
            PrefixCacheRestoreStatus::GPU;
        const char *name = tier == PrefixCacheRestoreStatus::SSD ? "L3(SSD)" :
            tier == PrefixCacheRestoreStatus::Host ? "L2(HOST)" : "L1(GPU)";
        status.tier = tier; status.phase = PrefixCacheRestoreStatus::Adopted;
        DescribePrefix(ctx,ctx->cacheLen);
        status.gpuComponents = status.requiredComponents;
        status.runtimeTier = PrefixCacheRecord::TierL1;
        status.runtimeState = PrefixCacheRecord::RuntimeResident;
        status.runtimeLease = 0;
        status.durableState = loaded > 0 ? PrefixCacheRecord::DurableReady : status.durableState;
        status.loadingState = PrefixCacheRecord::Resident;
        PublishPrefixRecord(ctx);
        Check(status.gpuReady(),"ssd_adopt_incomplete_components");
        status.cachedTokens = ctx->cacheLen;
        ctx->progressPrefixLookupMs = std::max(ctx->progressPrefixLookupMs,
                                               status.lookupMs > 0 ? status.lookupMs : status.totalMs);
        ctx->progressPrefixRestoreMs = std::max(ctx->progressPrefixRestoreMs,
                                                status.restoreMs);
        const double ssdReadMiBS = status.readMs > 0 ?
            status.loadedBytes / 1048576.0 * 1000.0 / status.readMs : 0.0;
        const double ssdRestoreTokensS = status.totalMs > 0 ?
            ctx->cacheLen * 1000.0 / status.totalMs : 0.0;
        ctx->progressSsdReadMiBS = std::max(ctx->progressSsdReadMiBS, ssdReadMiBS);
        ctx->progressSsdRestoreTokensS = std::max(
            ctx->progressSsdRestoreTokensS, ssdRestoreTokensS);
        if (tier == PrefixCacheRestoreStatus::SSD) {
            ctx->progressL3Tokens = std::max(ctx->progressL3Tokens, ctx->cacheLen);
        } else if (tier == PrefixCacheRestoreStatus::Host) {
            ctx->progressL2Tokens = std::max(ctx->progressL2Tokens, ctx->cacheLen);
        } else {
            ctx->progressL1Tokens = std::max(ctx->progressL1Tokens, ctx->cacheLen);
        }
        const int used = loaded > 0 ? std::min(ctx->cacheLen,loaded) : 0;
        if (used > 0) restored += used;
        const int copies = loaded > 0 ?
            int(int64_t(ctx->intParams["ssd_restore_copies"])*used/loaded) : 0;
        const char *hitLabel = tier == PrefixCacheRestoreStatus::SSD ?
            "[Prefix SSD] CACHE-HIT" : "[PrefixCache] CACHE-HIT";
        if (!PrefixCacheDebugEnabled()) {
            PrefixCacheInfo("[PrefixCache] hit: tier=%s reused_tokens=%d.\n",
                            name,ctx->cacheLen);
        }
        PrefixCacheDebug("%s: request=%llu tier=%s cached_tokens=%d candidate_tokens=%d requested_tokens=%d "
                    "input_tokens=%d need_recompute=%zu gpu_adopted=1 ssd_copies=%d "
                    "verified_payload_bytes=%llu read_ms=%.1f speed_mib_s=%.1f "
                    "gpu_restore_ms=%.1f total_ms=%.1f required_components=%u gpu_components=%u "
                    "host_components=%u ssd_components=%u.\n",
                    hitLabel,(unsigned long long)status.requestId,name,ctx->cacheLen,
                    status.candidateTokens,ctx->inputTokens,ctx->inputTokens,ctx->currentTokens.size(),copies,
                    (unsigned long long)status.loadedBytes,
                    status.readMs,status.readMs > 0 ? status.loadedBytes/1048576.0*1000/status.readMs : 0,
                    status.restoreMs,status.totalMs,status.requiredComponents,status.gpuComponents,
                    status.hostComponents,status.ssdComponents);
        if (!ctx->multimodalInput.empty() && used > 0)
            PrefixCacheDebug("[Prefix SSD] multimodal restore: spans=%zu restored_tokens=%d base_page_keys=%zu.\n",
                        ctx->multimodalPrefixSpans.size(),used,ctx->prefixCacheBasePageKeys.size());
        PrefixCacheDebug("[PrefixCache] restore: cached=%d need_recompute=%zu all=%zu pos_ids=%zu.\n",
                    ctx->cacheLen,ctx->currentTokens.size(),ctx->allTokens.size(),ctx->prefixCachePositionIds.size());
        ctx->cacheSegments.clear();
        PrefixCacheObservationSegment segment;
        segment.sourceTier = tier == PrefixCacheRestoreStatus::SSD ? "L3" :
            tier == PrefixCacheRestoreStatus::Host ? "L2" : "L1";
        segment.promotedTo = segment.sourceTier == "L1" ? "NONE" : "L1";
        segment.sourceTokens = used > 0 ? used : ctx->cacheLen;
        segment.effectiveReusedTokens = segment.sourceTokens;
        segment.lookupMs = status.lookupMs > 0 ? status.lookupMs : status.totalMs;
        segment.restoreMs = status.restoreMs > 0 ? status.restoreMs : -1.0;
        segment.readMiBS = ssdReadMiBS > 0 ? ssdReadMiBS : -1.0;
        segment.restoreTokensS = ssdRestoreTokensS > 0 ? ssdRestoreTokensS : -1.0;
        ctx->cacheSegments.push_back(segment);
        if (tier == PrefixCacheRestoreStatus::SSD) {
            FastllmEvent("cache", ctx->requestId, "cache_result",
                         "source_tier=L3 promoted_to=L1 source_tokens=%d effective_reused_tokens=%d "
                         "lookup_ms=%.3f restore_ms=%.3f read_mib_s=%.3f restore_tokens_s=%.3f "
                         "read_ms=%.3f import_ms=%.3f restore_total_ms=%.3f "
                         "verified_payload_bytes=%llu restore_ms_scope=import_l1_publish "
                         "restore_timing_scope=ticket_start_through_import physical_io_state=not_measured",
                         segment.sourceTokens, segment.effectiveReusedTokens,
                         segment.lookupMs, segment.restoreMs, segment.readMiBS,
                         segment.restoreTokensS, status.readMs, status.restoreMs, status.totalMs,
                         (unsigned long long)status.loadedBytes);
        } else if (segment.restoreMs > 0) {
            FastllmEvent("cache", ctx->requestId, "cache_result",
                         "source_tier=%s promoted_to=%s source_tokens=%d effective_reused_tokens=%d "
                         "lookup_ms=%.3f restore_ms=%.3f",
                         segment.sourceTier.c_str(), segment.promotedTo.c_str(),
                         segment.sourceTokens, segment.effectiveReusedTokens,
                         segment.lookupMs, segment.restoreMs);
        } else {
            FastllmEvent("cache", ctx->requestId, "cache_result",
                         "source_tier=%s promoted_to=%s source_tokens=%d effective_reused_tokens=%d "
                         "lookup_ms=%.3f",
                         segment.sourceTier.c_str(), segment.promotedTo.c_str(),
                         segment.sourceTokens, segment.effectiveReusedTokens,
                         segment.lookupMs);
        }
        ctx->intParams["ssd_loaded_len"] = 0;
    }
};

void Qwen35PersistentCache::Capture(ResponseContext *ctx,
        const std::shared_ptr<Qwen35LinearPrefixSnapshot> &snapshot) {
    // A2：capture_cpu_ms = 请求线程在 Capture() 里同步花掉的墙钟（身份+准入+选页+记录事件+
    // 入队）；RAII 保证所有 return 分支都被计入。捕获之后的 D2H/哈希在后台 worker 上，不计入。
    const auto captureStart = Clock::now();
    struct CaptureCpuTimer {
        std::atomic<uint64_t> *target;
        Clock::time_point start;
        ~CaptureCpuTimer() {
            target->fetch_add(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start).count()), std::memory_order_relaxed);
        }
    } captureTimer{&captureCpuNs, captureStart};
    const bool traceCapture = CaptureTraceEnabled();
    auto traceStage = [&](const char *stage) {
        if (!traceCapture) return;
        std::fprintf(stderr,"[PrefixCache][trace] stage=%s t_ms=%.1f\n",stage,
                     std::chrono::duration_cast<std::chrono::microseconds>(
                         Clock::now() - captureStart).count() / 1000.0);
        std::fflush(stderr);
    };
    if (!ctx || !ctx->CanUsePagedPrefixCache() || !Qwen35LinearPrefixCacheEnabled()) return;
    RequestDiagnosticOperation diagnosticCapture(ctx->requestId, "ssd_checkpoint_capture",
        "Qwen35PersistentCache.Capture");
    const int length = snapshot->cachedLen, chunk = model->GetChunkedPrefillSize();
    if (length > ctx->inputTokens || chunk <= 0 || length % fastllm::GetPageLen()) return;
    if (!Ensure()) return;
    traceStage("enter");
    std::shared_ptr<ExportState> state;
    std::shared_ptr<RequestLeaseState> requestState;
    { RequestDiagnosticOperation diagnosticSlotLock(ctx->requestId, "ssd_capture_mutex_wait",
          RequestDiagnosticResource("persistent.mutex", &mutex).c_str());
      std::lock_guard<std::mutex> lock(mutex);
      diagnosticSlotLock.Finish();
      if (gpuBudget.empty()) RefreshGpuBudget(); // Pools may appear after skip-warmup initialization.
      auto lifecycle = requestStates.find(ctx);
      if (lifecycle == requestStates.end() || !lifecycle->second ||
          lifecycle->second->request_id != ctx->requestId ||
          lifecycle->second->cancelled.load(std::memory_order_acquire)) {
          std::fprintf(stderr, "[Prefix SSD] capture-rejected: status=REJECTED_BEFORE_ADMISSION reason=request_identity_missing_or_cancelled.\n");
          return;
      }
      requestState = lifecycle->second;
      auto &slot = exports[ctx];
      auto makeSlot = [&]() {
          auto created = std::make_shared<ExportState>();
          created->identity = identity; created->family = family;
          created->provenance = ctx->prefixCacheProvenance;
          created->requestTokens = ctx->inputTokens;
          created->request_id = ctx->requestId;
          created->request_generation = requestState->generation.load(std::memory_order_acquire);
          auto source = provenanceSources.find(created->provenance);
          if (source != provenanceSources.end()) created->source = source->second;
          uint64_t kvBytes = 0;
          for (int layer = 0; layer < model->block_cnt; ++layer)
              for (int side = 0; side < 2; ++side)
                  for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0))
                      kvBytes += (uint64_t)ctx->inputTokens*ref.second->dims[2]*ref.second->dims[3]*ref.second->unitSize;
          created->reservation = kvBytes + (ctx->inputTokens/CheckpointInterval() + ctx->multimodalPrefixSpans.size()+CheckpointJobsLimit())*SnapshotBytes(*snapshot) + (16<<20);
          return created;
      };
      // R4 修复（2026-10-02）：ResponseContext 被复用时（上一条请求结束后 Removed 未及时清），
      // 旧的 ExportState 会带上一条请求的 issued（真机实测 issued=8192）⇒ 本条请求开头几个小检查点
      // 被 "capture skip: length=2048 issued=8192" 直接丢掉，永远进不了盘。兜底规则：issued 大于
      // 本次要导出的长度，说明状态属于更早的请求，重建。
      // **残余局限（如实）**：若新请求的首个检查点边界大于旧 issued，旧状态仍会被沿用、前段检查点
      // 仍可能被跳过；彻底修法需要给请求绑定显式身份，另行安排。
      // R4 彻底版（2026-10-02）：请求身份变了（prompt 长度不同）就重建状态；
      // 保留 issued > length 的启发式作为兜底（同长度但内容不同的极端情形）。
      if (!slot || slot->request_id != ctx->requestId ||
          slot->request_generation != requestState->generation.load(std::memory_order_acquire) ||
          slot->requestTokens != ctx->inputTokens || slot->issued > length || slot->failed)
          slot = makeSlot();
      state = slot;
    }
    if (state->failed) { traceStage("failed_state"); return; }
    traceStage("slot_ready");
    if (length <= state->issued) {
        if (length < state->issued)
            PrefixCacheDebug("[PrefixCache] capture skip: length=%d issued=%d.\n",length,state->issued);
        traceStage("skip_range");
        return;
    }
    traceStage("range_ready");
    const int previousIssued = state->issued;
    const uint64_t previousChainHash = state->issuedChainValid ? state->issuedChainHash : 0;
    // Every valid prefill snapshot is a recoverable checkpoint. Keep the
    // existing page/chunk geometry; never discard a completed prefill state.
    const bool checkpoint = true;
    Job job; job.state = state; job.length = length;
    job.promptEnd = length + fastllm::GetPageLen() > ctx->inputTokens;
    job.request_id = ctx->requestId;
    job.request_state = requestState;
    job.request_generation = requestState ? requestState->generation.load(std::memory_order_acquire) : 0;
    job.owner_lease = requestState ? std::shared_ptr<const void>(requestState, requestState.get())
                                   : std::shared_ptr<const void>();
    job.copy_generation = copyGenerationCounter.fetch_add(1, std::memory_order_acq_rel) + 1;
    job.diagnosticToken = CaptureRequestDiagnosticToken(ctx->requestId);
    job.runtimeUsesPageKeys = ctx->PrefixCachePageKeys() != nullptr;
    job.hintWallNs = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    if (state->issued == 0 && ctx->cacheLen > 0) {
        job.adoptLength = std::min(ctx->cacheLen,length);
        job.adoptContext = SavePrefetchContext(ctx);
    }
    job.prefix = Prefix(ctx,length);
    job.key = DiskPrefixCache::Digest(job.prefix + state->provenance);
    job.cache_identity = DiskPrefixCache::Digest(identity + "|" + family + "|" +
        Dtype(model->kvCacheDataType) + "|" + state->provenance + "|" + job.prefix + "|" +
        RuntimeLayout().dump() + "|" + std::to_string(length));
    if (checkpoint) {
        job.snapshot = snapshot; job.stateBytes = SnapshotBytes(*snapshot);
        job.stateStage = snapshot->stateStage;
        job.stateStageBytes = snapshot->stateStageBytes;
        job.stateStageD2DEnqueueNs = snapshot->stateStageD2DEnqueueNs;
        job.stateStageFallbackSyncCount = snapshot->stateStageFallbackSyncCount;
        job.stateStageFallbackReason = snapshot->stateStageFallbackReason;
        job.prefetchContext = SavePrefetchContext(ctx,length);
    }
    if (job.stateStage) Qwen35StateStageBindIdentity(job.stateStage, job.request_id,
        job.request_generation, job.copy_generation, job.cache_identity);
    std::map<int,size_t> need;
    const int previousDevice = FastllmCudaGetDevice();
    struct RestoreDevice { int previous; ~RestoreDevice() { FastllmCudaSetDevice(previous); } } restore{previousDevice};
    try {
        traceStage("lock_wait");
        RequestDiagnosticOperation diagnosticLock(ctx->requestId, "ssd_capture_mutex_wait",
            RequestDiagnosticResource("persistent.mutex", &mutex).c_str());
        std::unique_lock<std::mutex> lock(mutex);   // B4-correct：等待 pin 时必须能 unlock（原 lock_guard）
        diagnosticLock.Finish();
        traceStage("lock_held");
        std::string identityError;
        if (inflightJobs >= kMaxInflightJobs) {
            ++degraded;
            std::fprintf(stderr, "[Prefix SSD] cache-job-rejected: status=REJECTED_BEFORE_ADMISSION error=queue_limit reason=max_inflight_jobs.\n");
            return;
        }
        if (queuedJobs >= kMaxSnapshotQueue) {
            ++degraded;
            std::fprintf(stderr, "[Prefix SSD] cache-job-rejected: status=REJECTED_BEFORE_ADMISSION error=snapshot_queue_full reason=max_snapshot_queue.\n");
            return;
        }
        // A-1（2026-10-04，本次改动）：**写盘门槛**——"只记值得记的"。
        // 位置：在拿到锁之后、TryAdmit 之前（此处尚未领票 ⇒ 直接 return 不会泄漏票据）。
        // 判据两条（都可配，默认 0=关闭以保持既有行为、便于 A/B 与一键回退）：
        //   ① 请求总上下文太短 ⇒ 不写：FASTLLM_PREFIX_CACHE_MIN_CONTEXT_TOKENS（**读写同源**）。
        //      原文写的是 …_WRITE_MIN_CONTEXT_TOKENS —— 该 env **全 src 只存在于
        //      这一条注释里、程序从不读取**（`grep -rn` 命中仅此一处），**照它设置完全无效**；真正读取它的是
        //      `MinCacheContextTokens()`（`.inc:186`）。且该门槛**默认 0 = 关闭**，不是默认开启。
        //      依据（我方实测）：读盘 ≪ 重算只在长上下文成立，AB 四点拟合盈亏平衡 ≈10,500 命中
        //      token；短请求写盘纯亏，正是 A2 短请求负收益的来源之一。**建议值取保守上界 16384（与读侧同源）**。
        //      原文写"建议值 10900"，与 `.inc:177`（同一旋钮、读侧取上界 16384）**冲突**；
        //      现统一为 **16384**（更保守 ⇒ 更贴合"开了不比不开慢"的构造保证），另一路估算（AB 四点拟合
        //      ≈10,500）保留备查。**这两处必须同步改，禁止只改一处。**
        //   ② 该 key 尚未被重复见到 ⇒ 不写：FASTLLM_PREFIX_CACHE_WRITE_MIN_REUSE。
        //      依据（业界）：vLLM store_threshold="Min lookups before a block is offloaded"；
        //      SGLang write_through_selective 按命中计数只备份热数据；Dynamo presence_lfu
        //      (min_lfu_count 默认 8)。注意：单遍负载下该门槛会**完全不写**，故默认关闭。
        // 纪律：刻意不写**不是丢点** ⇒ 打 write_gated（不是 write_skipped）并单独计数，
        //       以免污染判据①（zero write_skipped）与合同 degraded=0。
        if (checkpoint) {
            bool gated = false;
            const char *gatedReason = nullptr;
            if (WriteMinContextTokens() > 0 && ctx->inputTokens < WriteMinContextTokens()) {
                gated = true; gatedReason = "context_below_write_threshold";
            }
            if (!gated && WriteMinReuse() > 1) {
                if (writeReuseSeen.size() > 4096) writeReuseSeen.clear();
                const int seen = ++writeReuseSeen[job.key];
                if (seen < WriteMinReuse()) { gated = true; gatedReason = "reuse_below_write_threshold"; }
            }
            if (gated) {
                ++writeGated;
                PrefixCacheDebug("[Prefix SSD] write_gated: tokens=%d reason=%s context_tokens=%d min_context=%d min_reuse=%d gated=%llu (intentionally not written; not counted as a lost point).\n",
                             length,gatedReason,ctx->inputTokens,WriteMinContextTokens(),WriteMinReuse(),
                             (unsigned long long)writeGated.load(std::memory_order_relaxed));
                return;
            }
        }
        if (stopping || admissionClosed.load()) return; // Explicit service teardown.
        if (pendingCheckpointKeys.count(job.key)) {
            ++coalescedCheckpoints;
            return; // The exact prefix already has an immutable background owner.
        }
        job.ticket = {++captureTicketId, (uint64_t)length};
        job.deferredPin = true;
        // Coordinator byte admission belongs to Work. GPU source references
        // and the shared immutable snapshot survive until that admission succeeds.
        traceStage("admission_deferred");
        // S4 方案 C（有界/窗口化导出 + 请求生命期 keepalive，2026-10-04）：先收集全部
        // (layer,side,rank) 的完整页区间，再按"每窗口 ≤ min(pin 预算/2, BufferBytes,
        // 256 token 整数倍)"切窗口。Capture 对**整段** Pick（keepalive，见下方），
        // Export 逐窗 D2H 并在每个分块拷贝完成后立即放钉 ⇒ 峰值"拷贝并发"与上下文长度无关。
        // 整段保护的真实容量占用由 exportProtectedPeak（export_protected_bytes）单独观测。
        // 窗口拼接后仍是原来的完整 bundle 序列（顺序、begin/tokens 网格都不变）。
        std::map<int,size_t> rankBytesPerPage;   // rank -> 该 rank 每"页号"的字节（其全部 lease 之和）
        traceStage("plan_begin");
        const int chainPageLen = fastllm::GetPageLen();
        Check(chainPageLen > 0, "ssd_capture_chain_state");
        const size_t firstNewPage = size_t(std::max(previousIssued, 0)) / size_t(chainPageLen);
        const bool runtimeUsesPageKeys = !ctx->multimodalInput.empty();
        const size_t tokenPages = size_t(length) / size_t(chainPageLen);
        const size_t numPages = runtimeUsesPageKeys
            ? std::min(tokenPages, ctx->prefixCachePageKeys.size()) : tokenPages;
        // The saved rolling hash is reused only when it covers exactly the pages
        // before firstNewPage; otherwise rebuild the prefix (the old full walk).
        const bool chainReusable = firstNewPage > 0 && state->issuedChainValid &&
            state->issuedChainPages == firstNewPage && firstNewPage <= numPages;
        uint64_t chain = chainReusable ? previousChainHash : 0;
        const size_t chainStart = chainReusable ? firstNewPage : 0;
        std::vector<uint64_t> newChainHashes;
        if (numPages > firstNewPage) newChainHashes.reserve(numPages - firstNewPage);
        for (size_t p = chainStart; p < numPages; ++p) {
            const std::string extra = runtimeUsesPageKeys ? ctx->prefixCachePageKeys[p] : std::string();
            chain = fastllm::PagedCacheManager::HashTokenPrefixPage(
                chain, ctx->allTokens.data() + p * size_t(chainPageLen), chainPageLen, extra);
            if (p >= firstNewPage) newChainHashes.push_back(chain);
        }
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model,layer)) continue;
            for (int side = 0; side < 2; ++side) {
                auto &global = side ? ctx->pastKeyValues[layer].second : ctx->pastKeyValues[layer].first;
                for (auto ref : model->GetPagedKVCacheManagers(layer,side == 0)) {
                    Data *local = &global;
                    if (global.multiDeviceData) {
                        auto it = global.multiDeviceDatas.find(ref.first);
                        Check(it != global.multiDeviceDatas.end() && it->second,"ssd_missing_export_rank"); local = it->second;
                    }
                    Check(local->isPagedKVCache && local->pagedKVCacheData == ref.second &&
                        length % ref.second->pageLen == 0 && (int)local->pageIndex.size() >= length / ref.second->pageLen,
                        "ssd_export_pages");
                    auto lease = std::make_shared<Pages>(); lease->pool = ref.second;
                    lease->layer = layer; lease->side = side; lease->rank = ref.first; lease->begin = state->issued;
                    lease->pageBytes = (size_t)ref.second->pageLen * ref.second->dims[2] * ref.second->dims[3] * ref.second->unitSize;
                    if (layer >= 0 && layer < (int)model->threadTpAttentionKVHeadSchemes.size()) {
                        auto scheme = model->threadTpAttentionKVHeadSchemes[layer].find(ref.first);
                        if (scheme != model->threadTpAttentionKVHeadSchemes[layer].end())
                            lease->ranges = scheme->second;
                    }
                    if (lease->ranges.empty()) lease->ranges.push_back({0, ref.second->dims[2]});
                    lease->ids.assign(local->pageIndex.begin() + state->issued / ref.second->pageLen,
                                      local->pageIndex.begin() + length / ref.second->pageLen);
                    lease->generations.reserve(lease->ids.size());
                    lease->hashes.reserve(lease->ids.size());
                    for (int page : lease->ids) {
                        Check(page >= 0 && page < (int)ref.second->pageTimestamp.size(),
                              "ssd_page_generation_missing");
                        const uint64_t generation = uint64_t(ref.second->pageTimestamp[page]);
                        Check(generation != 0, "ssd_page_generation_missing");
                        lease->generations.push_back(generation);
                        job.page_generation = std::max(job.page_generation, generation);
                    }
                    // W1：发布端的 pageHash 只有回调注册时才是链哈希；部署未启用
                    // FASTLLM_PREFIX_CACHE_L2_HOST_KV 时 pageHash 可能是另一个 h。因此 Capture
                    // 始终保存本请求自算的链身份，已发布节点仅用于诊断，不覆盖保存值。
                    // numPages 与 RecordRange 同步取 token/page、物理页数组、extra 数组三者最小；
                    // 纯文本发布传 nullptr，故 extraKeys 为空时按空串处理。
                    {
                        const int pageLen = ref.second->pageLen;
                        std::lock_guard<std::mutex> guard(ref.second->pageIndexLocker);
                        size_t mismatches = 0;
                        // lease->ids[k] is a physical page id; its logical chain position is previousIssued/pageLen+k.
                        const size_t firstPosition = pageLen > 0 ? size_t(previousIssued) / size_t(pageLen) : 0;
                        for (size_t k = 0; k < lease->ids.size(); ++k) {
                            const int page = lease->ids[k];
                            const size_t position = firstPosition + k;
                            const uint64_t computed =
                                position >= firstNewPage && position - firstNewPage < newChainHashes.size()
                                    ? newChainHashes[position - firstNewPage] : 0;
                            lease->hashes.push_back(computed);
                            auto node = ref.second->pageToTrieNode.find(page);
                            if (node != ref.second->pageToTrieNode.end() && node->second != nullptr && computed != 0 &&
                                node->second->pageHash != computed) {
                                ++mismatches;
                            }
                        }
                        if (mismatches > 0) {
                            std::fprintf(stderr,
                                         "[Prefix SSD] page-hash-mismatch: pages=%llu of %llu "
                                         "(computed vs published).\n",
                                         (unsigned long long)mismatches,
                                         (unsigned long long)lease->ids.size());
                            std::fflush(stderr);
                        }
                    }
                    // Record on the actual persistent TP producers, not an unrelated
                    // scheduler-thread stream on the same device.
                    FastllmCudaSetDevice(ref.first);
                    lease->ready = FastllmCudaEventCreate();
                    rankBytesPerPage[ref.first] += lease->pageBytes;
                    job.pages.push_back(std::move(lease));
                }
            }
        }
        traceStage("plan_done");
        // Page generations are available only after the complete source-page
        // map is captured. Reject before Pick, events or queue ownership.
        if (!ValidateJobIdentityLocked(job, "enqueue", identityError)) {
            ++degraded;
            std::fprintf(stderr, "[Prefix SSD] cache-job-rejected: status=REJECTED_BEFORE_ADMISSION reason=%s.\n",
                         identityError.c_str());
            return;
        }
        // 窗口页数：以"允许窗口页数最小"的 rank 作绑定约束，用纯函数 SplitExportWindows 切分
        // （该函数可无 GPU 单测）。窗口字节上界 = min(pin 预算/2, BufferBytes)；再收窄到
        // 256 token 的整数倍（仅当窗口本身 > 256 token；预算极小/不足一页时保留 1 页以保证进展）。
        int windowPages = 0;
        if (!job.pages.empty()) {
            const int pageLen = job.pages.front()->pool->pageLen;
            const int totalPages = (int)job.pages.front()->ids.size();
            int bindingRank = -1;
            uint64_t bindingBytesPerPage = 0, bindingPages = 0;
            for (auto &entry : rankBytesPerPage) {
                const size_t rankBudget = gpuBudget.count(entry.first) ? gpuBudget[entry.first] : 0;
                const uint64_t cap = rankBudget > 0 ? std::min<uint64_t>(rankBudget / 2, BufferBytes)
                                                    : (uint64_t)BufferBytes;
                uint64_t pages = entry.second > 0 ? cap / entry.second : (uint64_t)totalPages;
                if (pages < 1) pages = 1;   // 预算不足一页也至少 1 页（否则无法推进）
                if (bindingRank < 0 || pages < bindingPages) {
                    bindingRank = entry.first; bindingBytesPerPage = entry.second; bindingPages = pages;
                }
            }
            const auto windows = fastllm::prefix_v4::SplitExportWindows(
                (uint64_t)totalPages, bindingBytesPerPage,
                bindingPages * bindingBytesPerPage, BufferBytes);
            windowPages = windows.empty() ? totalPages : (int)windows.front().endPage;
            if (windowPages < 1) windowPages = 1;
            if (windowPages > totalPages) windowPages = totalPages;
            const int windowTokens = windowPages * pageLen;
            if (pageLen > 0 && windowTokens > 256) {
                const int alignedPages = (windowTokens / 256) * 256 / pageLen;
                if (alignedPages >= 1) windowPages = alignedPages;
            }
            job.windowPages = windowPages;
        }
        traceStage("split_done");
        for (auto &entry : rankBytesPerPage)
            need[entry.first] = (size_t)windowPages * entry.second;   // 单窗口字节（每 rank）
        // No budget wait or rejection on the request. Work reserves the same
        // window ledger atomically and owns the source while capacity is busy.
        traceStage("budget_ok");
        // 方案 C（2026-10-04，请求生命期 keepalive）：对**整段** [issued,length) 的页 Pick
        // 一次——我们自己持一份引用，等于把"请求提前结束、页被复用"的窗口补齐；这正是
        // vLLM 引用计数式前缀缓存的语义（拷贝完成前页不可复用）。代价：请求已结束时，
        // 这批页在本笔导出完成前仍不可被复用/淘汰。
        // 注意：**保护是整段的，预算记账只按当前窗口**——pinnedGpuBytes 只加第 0 窗
        // （其余窗口在 Export 里逐窗加账），因此该预算仍代表"拷贝并发/瞬时压力"，
        // 与上下文长度无关；整段保护的真实占用由 export_protected_bytes 单独观测。
        traceStage("keepalive_begin");
        for (auto &lease : job.pages) {
            const int count = (int)lease->ids.size();
            if (count <= 0) continue;
            lease->pool->Pick(lease->ids);          // keepalive：整段加引用
            lease->pickedPages = count;
            lease->accountedPages = job.deferredPin ? 0 : std::min<int>(windowPages,count);   // 第 0 窗已记账
            const uint64_t protectedBytes = (uint64_t)count * lease->pageBytes;
            exportProtectedBytes += protectedBytes;
            if (exportProtectedBytes > exportProtectedPeak) exportProtectedPeak = exportProtectedBytes;
        }
        traceStage("keepalive_done");
        // Record on the actual persistent TP producers, not an unrelated
        // scheduler-thread stream on the same device.
        auto recordRank = [&](int device) {
            FastllmCudaSetDevice(device);
            for (auto &lease : job.pages)
                if (lease->rank == device) FastllmCudaEventRecordCurrentThread(lease->ready);
        };
        if (model->threadTpPreparedDevices.size() > 1 && model->threadTpWorkerGroup.HasWorkers()) {
            std::vector<std::exception_ptr> errors(model->threadTpPreparedDevices.size());
            model->threadTpWorkerGroup.RunWithCaller(model->threadTpPreparedDevices,[&](int rank) {
                recordRank(model->threadTpPreparedDevices[rank]);
            },errors);
            for (auto &error : errors) if (error) std::rethrow_exception(error);
        } else for (auto n : need) recordRank(n.first);
        traceStage("record_done");
        if (!job.deferredPin) for (auto n : need) pinnedGpuBytes[n.first] += n.second;
        state->issued = length;
        state->issuedChainHash = chain;
        state->issuedChainPages = numPages;
        state->issuedChainValid = true;
        queuedStateBytes += job.stateBytes; checkpointJobs += checkpoint;
        if (checkpoint) ++expectedCheckpoints;   // A2：native capture 事件（决定要写且成功入队）
        traceStage("enqueue_begin");
        const size_t laneIndex = LaneForState(job.state.get(), writeWorkerCount);
        job.enqueuedAt = Clock::now();
        PrefixCacheDebug("[PrefixCache] enqueue: tokens=%d checkpoint=%s issued=%d queued=%zu lane=%zu snap_mb=%.2f.\n",
               length, checkpoint ? "yes" : "no", state->issued, queuedJobs + 1, laneIndex,
               job.stateBytes / 1048576.0);
        // Match the actual paged-trie identity: ordinary text uses no extra
        // keys even though persistent identity retains canonical text keys.
        job.writeStatuses = Qwen35ProtectSpillPrefix(model, snapshot->tokens,
            ctx->multimodalInput.empty() ? std::vector<std::string>() : snapshot->mediaPageKeys,
            0, length, 0);
        pendingCheckpointKeys.insert(job.key);
        ++queuedJobs;
        ++inflightJobs;
        laneQueues[laneIndex].push_back(std::move(job));
        changed.notify_all();
    } catch (const std::exception &e) {
        traceStage("exception");
        ++writeFailures;
        ++skipped;
        if (job.snapshot) ++degraded;   // A2：Capture 同步段异常而丢弃的检查点
        std::fprintf(stderr,"[Prefix SSD] capture skipped: %s.\n",e.what());
    }
    FastllmCudaSetDevice(previousDevice);
}

// Reuse the existing verified parent-manifest adoption in the write worker.
// Capture only owns page leases and a value copy of the canonical context;
// the request never reads storage or waits for a previous durable commit.
int Qwen35PersistentCache::AdoptParent(Job &job) {
    if (job.adoptLength <= 0) return 0;
    ResponseContext context;
    LoadPrefetchContext(job.adoptContext,context);
    context.cacheLen = job.adoptLength;
    auto *ctx = &context;
    auto state = job.state;
    int adoptedLength = 0;
    // A branch may start from an already durable hot prefix. Adopt its
    // immutable references rather than re-exporting it in a different
    // grouping/order and trying to republish the same checkpoint key.
    //
    // 1A：采用失败此前被静默吞掉 ⇒ issued 停在 0
    // ⇒ 该检查点按 [0,length) **整段上下文**导出；pin 需求 = length × 每 token 页字节
    // （实测 2048 B/卡）⇒ 长度 ≥ 预算/2048 B 时必然超限丢点（r13：tokens=147456 需
    // 288 MiB > 预算 256 MiB；而该长度恰好 == cacheLen，本该"零字节跳过"）。
    // 故：①失败必须留证据；②按键 Load 失败时按"家族+长度+前缀"回退查真键（与
    // Lookup 路径 :2161-2185 同款查询、同款键自校验）；③给"刚落盘/投影未可见"一个
    // 很小的有界重试（本段在锁外，不存在 r12 那种持锁自死锁风险）。
    const auto parentPrefix = Prefix(ctx,ctx->cacheLen);
    const auto parentKey = DiskPrefixCache::Digest(parentPrefix+state->provenance);
    // S1：把校验长度/前缀参数化，以便**回退到更旧的可用父记录**。
    auto adoptFrom = [&](const std::string &key, int expectedLength, const std::string &expectedPrefix,
                         bool requireProvenance, std::string &error) -> bool {
        try {
            // A-3 P1/T2：父继承必须与恢复候选同口径过账本检查。
            // 此前本路径只校验 SSD live 与 manifest 身份，从不查账本，因而可能继承
            // 已 FAILED/quarantined 的父记录（独立交叉核验旁路 B1）。
            if (model->pagedPageLedger) {
                PrefixCachePageLedger::CheckpointSnapshot parentLedger;
                if (model->pagedPageLedger->Checkpoint(key, parentLedger) &&
                    !model->pagedPageLedger->CanRestoreCheckpoint(key, parentLedger.identity)) {
                    error = "parent_failed_or_quarantined";
                    return false;
                }
            }
            auto inherited = std::make_unique<DiskPrefixCache::ReadSession>(store->BeginRead());
            // B1：**清单-only 加载**。
            // 本分支只读 parent 的元数据（length/prefix/kv_dtype/layout/provenance/kv），
            // 从不取载荷字节 —— 证据：本函数把 `inherited` 存进 `state->inherited` 之后，
            // 该成员在**全仓**只有两处出现，都是 `reset()`（:2037 成功路径、:2561 失败路径），
            // **没有任何解引用** ⇒ `Load` 取回的 566.2 MiB 载荷（463/520 个对象）读了就扔。
            // 实测代价：`adopt_begin→adopt_done` = **2974.9 ms**（T1 真机 2026-10-05），
            // 且它阻塞在请求路径上 ⇒ GPU 空转。
            // 安全性：被跳过的载荷，存在性由提交侧「只引用条目」的批量校验把关
            // （disk_prefix_cache.cpp:4317-4336，缺一个即 v4_object_missing，fail-closed）。
            const auto parent = inherited->LoadManifest(key);
            Check(parent["length"].int_value() == expectedLength &&
                parent["prefix"].string_value() == expectedPrefix &&
                parent["kv_dtype"].string_value() == Dtype(model->kvCacheDataType) &&
                parent["layout"].dump() == RuntimeLayout().dump(),"ssd_parent_identity");
            // 直接命中时仍要求 provenance 一致（保持原语义）；回退查询命中的记录，
            // 其 provenance 就是该记录自己的（与 Lookup 的 Import 口径一致），不再强求。
            if (requireProvenance)
                Check(parent["provenance"].string_value() == state->provenance,"ssd_parent_identity");
            state->kv = parent["kv"].array_items();
            uint64_t inheritedBytes = 0;
            for (const auto &bundle : state->kv) inheritedBytes += (uint64_t)bundle["bytes"].number_value();
            state->reservation -= std::min(state->reservation,inheritedBytes);
            state->inherited = std::move(inherited);
            adoptedLength = expectedLength;
            return true;
        } catch (const std::exception &e) { error = e.what(); return false; }
    };
    std::string adoptError = "not_attempted";
    bool adopted = false;
    for (int attempt = 0; attempt < 3 && !adopted; ++attempt) {
        if (attempt > 0) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (adoptFrom(parentKey,ctx->cacheLen,parentPrefix,true,adoptError)) { adopted = true; break; }
        try {
            auto matching = store->ListCheckpoints("",family,ctx->cacheLen,"prefix",parentPrefix);
            for (auto &entry : matching) {
                if (entry.length != ctx->cacheLen) continue;
                if (entry.kvDtype != Dtype(model->kvCacheDataType)) continue;
                if (entry.key != DiskPrefixCache::Digest(parentPrefix+entry.provenance)) continue;
                if (adoptFrom(entry.key,ctx->cacheLen,parentPrefix,false,adoptError)) { adopted = true; break; }
            }
        } catch (const std::exception &e) { adoptError = e.what(); }
    }
    // S1：**回退到更旧的可用父记录**（此前不存在，只有"同一 cacheLen"重试）。
    // 复核证据：只重试同一 cacheLen+prefix，失败后 issued 停在 0 ⇒ 该检查点按 [0,length) 整段导出
    // ⇒ pin 需求 = length × 每 token 页字节 ⇒ 超预算 ⇒ write_skipped（t1b: v4_root_missing；
    // t1c: v4_block_parent_missing），且 issued 永不前移 ⇒ **后续每个检查点都要整段重导**（丢点自激）。
    // 现在按"长度降序"找一个能用的更旧父记录：采用它的 kv、把 issued 推到该长度，本次只导出
    // [该长度, length)。ListCheckpoints 的 maximum 参数语义是"length ≤ maximum"，正是所需。
    if (!adopted) {
        try {
            auto candidates = store->ListCheckpoints("",family,ctx->cacheLen,"prefix","");
            for (auto &entry : candidates) {
                if (entry.length >= ctx->cacheLen || entry.length <= 0) continue;
                if (entry.kvDtype != Dtype(model->kvCacheDataType)) continue;
                const auto entryPrefix = Prefix(ctx,entry.length);
                if (entry.key != DiskPrefixCache::Digest(entryPrefix+entry.provenance)) continue;
                std::string fallbackError;
                if (adoptFrom(entry.key,entry.length,entryPrefix,false,fallbackError)) {
                    adopted = true;
                    std::fprintf(stderr,"[Prefix SSD] parent-adopt fallback: cacheLen=%d adopted_older=%d key=%s.\n",
                                 ctx->cacheLen,entry.length,entry.key.c_str());
                    break;
                }
                adoptError = fallbackError;
            }
        } catch (const std::exception &e) { adoptError = e.what(); }
    }
    if (!adopted)
        std::fprintf(stderr,"[Prefix SSD] parent-adopt failed: cacheLen=%d prefix=%s reason=%s.\n",
                     ctx->cacheLen,parentPrefix.c_str(),adoptError.c_str());
    return adoptedLength;
}

void Qwen35PersistentCache::Export(Job &job, void *buffer, size_t laneIndex) {
    // R1：本函数只被本 lane 的 worker 线程调用；WriteLane 的所有字段因此无需加锁。
    WriteLane &lane = *writeLanes[laneIndex];
    (void)buffer;   // 旧签名遗留的 pinned 缓冲仅在 Work() 做非空校验，D2 走 lane.d2SlotMemory
    if (job.hintOnly) { UpdatePrefetchHint(job); return; }
    if (job.image) {
        auto writer = imageStore->BeginWrite(job.stateBytes + (1<<20));
        auto chunks = writer.StoreBytes(job.stateBytes,[&](size_t offset,char *out,size_t count) {
            memcpy(out,(char*)job.image->data()+offset,count);
        });
        writer.Commit(job.key,Json::object{{"kind","encoder"},{"length",0},
            {"count",(double)job.image->size()},{"chunks",chunks}});
        writes += writer.WrittenBytes(); return;
    }
    auto &state = *job.state;
    Check(!state.failed, "ssd_previous_checkpoint_failed_retained");
    // [export-profile] 写盘 5 段计时：定位单线程 worker 的 CPU 热点
    double exD2h=0, exStore=0, exLinear=0, exSpec=0, exCommit=0;
    // A2：本检查点在 D2 取回上"没取到、要再等一轮"的墙钟（capture_background_wait_ms 的 D2 部分）。
    double exD2Wait=0;
    // §④ 前置定位：把 linear/spec 各自拆成"打包"与"算摘要+暂存"
    double exLinearPack=0, exLinearStore=0, exSpecPack=0, exSpecStore=0;
    // §407ms（2026-10-01）：spec 打包与 linear 段并行后，被"藏起来"的那部分墙钟时间。
    // 打印 total_ms 是各段之和，并行后它会大于真实墙钟；用本值可还原真实墙钟。
    double exSpecHidden=0;
    // 未计入五段的部分（实测约 357ms）：Export 内除五段之外的开销 + 段间隙
    double exSetup=0, exGap=0;
    const auto exEntry = std::chrono::steady_clock::now();
    auto exNow = [] { return std::chrono::steady_clock::now(); };
    auto exMs = [](const std::chrono::steady_clock::time_point &a,
                   const std::chrono::steady_clock::time_point &b) {
        return std::chrono::duration<double,std::milli>(b-a).count();
    };
    const int adoptedLength = AdoptParent(job);
    if (adoptedLength > 0) {
        // Release the inherited prefix immediately. Only the uncopied suffix
        // remains protected; use the same lease/budget accounting as D2H.
        for (auto &lease : job.pages) {
            const int count = std::min<int>(lease->ids.size(),
                std::max(0,adoptedLength-lease->begin)/lease->pool->pageLen);
            lease->pool->ReleasePageIndices(std::vector<int>(lease->ids.begin(),lease->ids.begin()+count));
            lease->ids.erase(lease->ids.begin(),lease->ids.begin()+count);
            lease->generations.erase(lease->generations.begin(),
                                     lease->generations.begin() + std::min<int>(count, lease->generations.size()));
            lease->pickedPages -= count;
            lease->begin += count*lease->pool->pageLen;
            std::lock_guard<std::mutex> lock(mutex);
            pinnedGpuBytes[lease->rank] -= (size_t)lease->accountedPages*lease->pageBytes;
            lease->accountedPages = 0;
            const uint64_t released = (uint64_t)count*lease->pageBytes;
            exportProtectedBytes = exportProtectedBytes > released ? exportProtectedBytes-released : 0;
            pageLeaseReleases.fetch_add(count,std::memory_order_relaxed);
        }
        job.deferredPin = true; // Account the suffix's first window in Export.
        PrefixCacheDebug("[Prefix SSD] parent-adopt: tokens=%d background=1.\n",adoptedLength);
    }
    if (!state.writer) state.writer = std::make_unique<DiskPrefixCache::WriteSession>(store->BeginWrite(state.reservation));
    // V4: the writer's shared lease now protects all refs. V5 with REQUEST_HOLD: the parent pin
    // stays until the request's ExportState goes (a V5 lease covers only files this writer wrote).
    if (!RequestHoldEnabled()) state.inherited.reset();
    auto &writer = *state.writer;
    const auto before = writer.WrittenBytes();
    exSetup += exMs(exEntry, exNow());
    std::map<int,std::vector<std::shared_ptr<Pages>>> ranks;
    for (auto &lease : job.pages) ranks[lease->rank].push_back(lease);
    // S4 方案 C（有界/窗口化导出，2026-10-04）：把本笔的 KV 分块导出按"窗口"推进——
    // 每个窗口依次"等 pin 预算（记账）→ 提交分块 → 取回（D2H 完成即在 D2CopyIntoSlot 里
    // 逐块放钉）"。整段页的 keepalive 引用已在 Capture Pick；这里只做实时的窗口级预算记账，
    // 因此"拷贝并发"峰值 ≤ 单窗口字节，与上下文长度无关。所有窗口的 bundle 仍**顺序追加**
    // 到同一 state.kv，Commit 只发生一次；128-token 分块网格与 bundle 形状不变。
    // A-5：开写前清掉上一笔可能残留的条目（正常应恒为 0；非 0 说明上一笔异常没收干净）。
    if (lane.d2Stream && lane.d2Stream->Pending() != 0) {
        std::fprintf(stderr, "[Prefix SSD] d2-stream-residue: pending=%llu before job (reset).\n",
                     (unsigned long long)lane.d2Stream->Pending());
        lane.d2Stream->Reset();
    }
    lane.d2Chunks.clear();
    lane.d2RankEvents.clear();
    lane.d2CopyStreams.clear();
    lane.d2UnhashedPages = 0;
    const uint64_t d2RequestId = ++lane.d2Requests;
    uint64_t d2Submitted = 0;
    uint64_t d2Taken = 0;
    // A-5：提交与取回**交错**推进。
    // 原实现先在一个窗口内把 ceil(tokens/128) 块全部提交、再去取回，而每块都先在 :2730 占一个
    // pendingCopy 预约；窗口块数一旦超过上限（真机 tokens=4096 ⇒ 32 块 > cap 16）第 17 块必抛
    // copy_limit，整笔检查点被丢弃（logs/user-deploy-s7-20261031T0826Z/server.log:339）。
    // 现在每个 lane 的在途块数限制在 MaxPendingCopy/W（W=4 ⇒ 4，与全局 pinned 槽位 4 同量级），
    // 超限就先取回队首再继续提交：队列满只会让后台写变慢，不会毁掉这一笔（S4 同源）。
    const size_t inFlightLimit = std::max<size_t>(
        size_t(1), MaxPendingCopy(writeWorkerCount) / size_t(std::max(1, writeWorkerCount)));
    // 取回必须有总时限：队首若是上一笔残留、其生产者事件永远不就绪，coordinator 会一直返回
    // "事件未完成"，没有时限就会把后台写线程永久卡住（复核子代理 A-5-REVIEW 第 2 条）。
    const auto takeDeadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(kDrainTimeoutMs);
    size_t tokensTaken = 0;   // 已落盘 token 数，用于作业尾的完整性校验。
    auto retireNext = [&]() {
        for (;;) {
            Check(std::chrono::steady_clock::now() < takeDeadline, "ssd_d2_take_timeout");
            const auto pumpStart = exNow();
            lane.d2Stream->PumpCopy();
            exD2h += exMs(pumpStart, exNow());
            lane.d2PumpMs += exMs(pumpStart, exNow());
            // S4：随流水推进取峰值（末尾采样只会看到 0）
            {
                const uint64_t pending = lane.d2Stream->Pending();
                const uint64_t slots = uint64_t(lane.d2Stream->SlotsInUse());
                const uint64_t inflight = lane.hashPool ? lane.hashPool->InFlightBytes() : 0;
                if (pending > lane.d2PendingMax) lane.d2PendingMax = pending;
                if (slots > lane.d2SlotsMax) lane.d2SlotsMax = slots;
                if (inflight > lane.poolInflightMax) lane.poolInflightMax = inflight;
            }
            prefix_write::TicketId ticket;
            prefix_write::HashedChunk hashed;
            const auto takeStart = exNow();
            const bool taken = lane.d2Stream->TakeOrderedChunk(ticket, hashed, 5);
            const double takeMs = exMs(takeStart, exNow());
            lane.d2TakeMs += takeMs;
            if (taken) {
                Check(ticket.requestId == d2RequestId && ticket.sequence == d2Taken, "ssd_d2_order");
                const auto storeStart = exNow();
                auto chunks = writer.StoreVerified(std::move(hashed));
                exStore += exMs(storeStart, exNow()) + takeMs;
                ++lane.d2TakeCount;
                const D2Chunk &chunk = lane.d2Chunks[size_t(d2Taken)];
                Check(chunk.copy_generation != 0, "copy_generation_missing");
                Check(chunk.copy_terminal && !chunk.pending_copy_counted,
                      "copy_terminal_missing");
                state.kv.emplace_back(Json::object{{"bytes",(double)chunk.bytes},
                                                  {"segments",chunk.segments},{"chunks",chunks}});
                tokensTaken += size_t(chunk.count);
                ++d2Taken;
                return;
            }
            // A2：本轮没取到（后台 D2H/哈希还没产出）⇒ 计入"后台等待"；再看失败出口。
            exD2Wait += takeMs;
            Check(!lane.d2Stream->Failed({d2RequestId, d2Taken}), "ssd_d2_copy_failed");
        }
    };
    auto reserveWindowPinBudget = [&](const std::map<int,size_t> &need) {
        RequestDiagnosticOperation diagnosticBudget(job.diagnosticToken,
            "ssd_export_pin_budget_wait", RequestDiagnosticResource("persistent.changed", &changed).c_str());
        std::unique_lock<std::mutex> lock(mutex);
        for (const auto &entry : need) {
            const size_t budget = gpuBudget.count(entry.first) ? gpuBudget.at(entry.first) : 0;
            Check(budget == 0 || entry.second <= budget, "ssd_pin_window_larger_than_budget");
        }
        changed.wait(lock, [&] {
            for (const auto &entry : need) {
                const size_t budget = gpuBudget.count(entry.first) ? gpuBudget.at(entry.first) : 0;
                if (budget && pinnedGpuBytes[entry.first] + entry.second > budget) return false;
            }
            return true;
        });
        for (const auto &entry : need) pinnedGpuBytes[entry.first] += entry.second;
    };
    const int windowPages = job.windowPages > 0 ? job.windowPages : 0;
    const int totalPages = ranks.empty() ? 0 : (int)ranks.begin()->second.front()->ids.size();
    try {
        for (auto &rank : ranks) {
            FastllmCudaSetDevice(rank.first);
            void *copyStream = FastllmCudaStreamCreateLowestPriority();
            lane.d2CopyStreams[rank.first] = copyStream;
            std::vector<void *> events;
            for (auto &lease : rank.second) events.push_back(lease->ready);
            lane.d2RankEvents[rank.first] = std::move(events);
        }
        const int step = windowPages > 0 ? windowPages : totalPages;
        for (int wBegin = 0; wBegin < totalPages; wBegin += step) {
            const int wEnd = std::min(wBegin + step, totalPages);
            // 1) 方案 C：整段保护已在 Capture 完成（Pick 一次）；这里只为"拷贝并发/预算"
            //    按窗口加账（第 0 窗已在 Capture 记账，故从第 1 窗起）。有界等待不持 mutex，
            //    超时即抛错 ⇒ 整笔作废，已 Pick 的页由 ~Pages/Work 统一归还（不泄漏引用）。
            // A-0（2026-10-04）：若本笔在 Capture 被标记 deferredPin（请求线程没等预算），
            //    则**第 0 窗也必须在这里等**——否则等于把"没等到"直接变成"写坏/写丢"。
            if (wBegin > 0 || job.deferredPin) {
                std::map<int,size_t> windowNeed;
                for (auto &lease : job.pages)
                    windowNeed[lease->rank] += (size_t)(wEnd - wBegin) * lease->pageBytes;
                reserveWindowPinBudget(windowNeed);
                for (auto &lease : job.pages) lease->accountedPages += (wEnd - wBegin);
            }
            // 2) 提交本窗口的分块（128 token 网格；稳定 token 区间不依赖生产侧 page 大小/TP 宽度）。
            for (auto &rank : ranks) {
                FastllmCudaSetDevice(rank.first);
                auto &first = *rank.second.front();
                const int beginToken = wBegin * first.pool->pageLen;
                const int endToken = wEnd * first.pool->pageLen;
                for (int token = beginToken; token < endToken; token += 128) {
                    const int count = std::min(128,endToken-token);
                    D2Chunk chunk;
                    chunk.copy_generation = job.copy_generation;
                    chunk.rank = rank.first;
                    chunk.token = token;
                    chunk.count = count;
                    chunk.leases = rank.second;
                    Json::array segments;
                    size_t bytes = 0;
                    for (auto &lease : rank.second) {
                        Check((int)lease->ids.size() == totalPages &&
                              lease->begin == first.begin,"ssd_bundle_ranges");
                        const size_t rowBytes = lease->pageBytes/lease->pool->pageLen;
                        size_t segmentBytes = count*rowBytes;
                        Check(segmentBytes <= BufferBytes-bytes,"ssd_bundle_exceeds_buffer");
                        Json::array ranges;
                        for (auto range : lease->ranges) ranges.emplace_back(Json::array{range.first, range.second});
                        segments.emplace_back(Json::object{{"layer",lease->layer},{"side",lease->side},{"rank",lease->rank},
                            {"begin",lease->begin+token},{"tokens",count},
                            {"page",lease->pool->pageLen},{"heads",lease->pool->dims[2]},{"dim",lease->pool->dims[3]},
                            {"dtype",(int)lease->pool->dataType},{"ranges",ranges},
                            {"bytes",(double)segmentBytes},{"offset",(double)bytes}});
                        bytes += segmentBytes;
                    }
                    chunk.bytes = bytes;
                    chunk.segments = segments;
                    const uint64_t sequence = d2Submitted++;
                    lane.d2Chunks.push_back(std::move(chunk));
                    const auto submitStart = exNow();
                    // 修复 `ssd_d2_submit`：
                    // D2 暂存预算是"拒绝而不是排队"的有意契约（协调器侧有契约测试断言超预算必须返回 false），
                    // 但本层以前用 Check() 把一次拒绝当成致命错误 ⇒ 整个检查点被整笔丢弃（无重试、无降级）。
                    // 现在：被拒就先推进流水线（PumpCopy 完成 D2H+哈希并释放暂存）再重试本块；
                    // 只有在完全无法推进（事件不完成、Pending 归零）时才真失败。契约层语义不变。
                    {
                        std::lock_guard<std::mutex> copyLock(mutex);
                        Check(pendingCopy < MaxPendingCopy(writeWorkerCount), "copy_limit");
                        // Reserve atomically before asynchronous submission.
                        ++pendingCopy;
                        lane.d2Chunks.back().pending_copy_counted = true;
                    }
                    const bool d2SubmittedOk = lane.d2Stream->Submit(
                        {d2RequestId, sequence}, uint64_t(rank.first), bytes);
                    if (!d2SubmittedOk) {
                        // Submit(false) guarantees that this ticket was never
                        // accepted. Undo only this reservation, without retry.
                        std::lock_guard<std::mutex> copyLock(mutex);
                        lane.d2Chunks.back().pending_copy_counted = false;
                        Check(pendingCopy > 0, "copy_rejection_without_reservation");
                        --pendingCopy;
                    }
                    Check(d2SubmittedOk, "ssd_d2_submit");
                    // A-5：在途超限就先取回队首（释放 pinned 槽位与 pendingCopy 预约），再继续提交。
                    while (d2Submitted - d2Taken >= inFlightLimit) retireNext();
                    lane.d2SubmitMs += exMs(submitStart, exNow());
                }
            }
            // 3) 本窗口提交的分块必须全部取回并落盘（交错推进后剩余不足一批的部分在此收尾）。
            //    这一段在 A-5 第一版被误删，导致每笔作业最后 3 块永不取回（复核子代理 A-5-REVIEW 第 1 条）。
            while (d2Taken < d2Submitted) retireNext();
            Check(d2Taken == d2Submitted, "ssd_d2_incomplete_take");
        }
        // 4) 全文覆盖校验：落盘的 token 数必须与整笔检查点等长，否则写出去的货是残的
        //    （读侧 Import 末尾会用 coverage==length 判死，写侧这里先挡住）。
        if (!job.pages.empty() && !ranks.empty()) {
            // 分块网格是**按 rank** 各铺一遍的（每个 rank 都覆盖同一段 token），所以总数要乘 rank 数。
            // 第一版漏了这个乘数，把正常的三笔检查点全都误判成 ssd_kv_token_mismatch
            // （logs/user-deploy-a5c-20261031T1005Z/server.log:305/444/584）——错在断言，不在数据。
            Check(size_t(tokensTaken) == size_t(totalPages) *
                                             size_t(job.pages.front()->pool->pageLen) *
                                             size_t(ranks.size()),
                  "ssd_kv_token_mismatch");
        }
        if (lane.d2UnhashedPages > 0) {
            std::fprintf(stderr,
                         "[Prefix SSD] d2-unhashed-pages: pages=%llu (timestamp fallback skipped; "
                         "lease+producer-event ownership used).\n",
                         (unsigned long long)lane.d2UnhashedPages);
            std::fflush(stderr);
        }
    } catch (...) {
        // 失败路径：先尝试把在途拷贝排空（丢弃结果），再**确定性清理**本 lane 的残留，
        // 避免旧 ticket 留在队首把下一笔作业撞死（A5 真机 d2_pending=2 即此残留）。
        try {
            for (int guard = 0; guard < 4096; ++guard) {
                lane.d2Stream->PumpCopy();
                if (lane.d2Stream->Pending() == 0 && lane.d2Stream->SlotsInUse() == 0) break;
                prefix_write::TicketId ticket;
                prefix_write::HashedChunk hashed;
                if (!lane.d2Stream->TakeOrderedChunk(ticket, hashed, 5)) {
                    if (lane.d2Stream->Pending() == 0 && lane.d2Stream->SlotsInUse() == 0) break;
                }
            }
        } catch (...) {
            // 排空过程中拷贝再次抛错（例如内容校验失败）：交给下面的 Reset 终结。
        }
        if (lane.d2Stream) {
            {
                std::lock_guard<std::mutex> copyLock(mutex);
                size_t released = 0;
                for (auto &chunk : lane.d2Chunks) {
                    if (chunk.pending_copy_counted) {
                        chunk.pending_copy_counted = false;
                        chunk.copy_terminal = true;
                        ++released;
                    }
                }
                Check(pendingCopy >= released, "copy_reservation_underflow");
                pendingCopy -= released;
            }
            lane.d2Stream->Reset();
            changed.notify_all();
        }
        for (auto &entry : lane.d2CopyStreams) {
            FastllmCudaSetDevice(entry.first);
            FastllmCudaStreamDestroy(entry.second);
        }
        lane.d2CopyStreams.clear();
        lane.d2RankEvents.clear();
        throw;
    }
    for (auto &entry : lane.d2CopyStreams) {
        FastllmCudaSetDevice(entry.first);
        FastllmCudaStreamDestroy(entry.second);
    }
    lane.d2CopyStreams.clear();
    lane.d2RankEvents.clear();
    if (job.snapshot) {
        const auto linearStart = exNow();
        Json::array linear;
        std::vector<char> packed;
        {
            const size_t layerCount = job.snapshot->layers.size();
            std::vector<PackPiece> pieces(layerCount);
            ParallelPieces(pieces, [&](size_t index, PackPiece &piece) {
                auto &entry = job.snapshot->layers[index];
                if (!entry.linear) return;
                Json first = SaveLinear(piece.buffer, entry.first);
                Json second = SaveLinear(piece.buffer, entry.second);
                piece.descriptor = Json::object{{"layer", int(index)}, {"first", first}, {"second", second}};
            }, PackWorkers());
            ReservePackMerge(packed, pieces, "ssd_linear_pack_budget");
            for (const auto &piece : pieces) {
                if (piece.descriptor.is_null()) continue;
                const size_t base = packed.size();
                packed.insert(packed.end(), piece.buffer.begin(), piece.buffer.end());
                linear.emplace_back(Json::object{
                    {"layer", piece.descriptor["layer"]},
                    {"first", ShiftedOffset(piece.descriptor["first"], base)},
                    {"second", ShiftedOffset(piece.descriptor["second"], base)}});
            }
            Check(packed.size() <= HostLimit(), "ssd_linear_pack_budget");
        }
        exLinearPack += exMs(linearStart, exNow());
        const size_t linearBytes = packed.size();
        // §407ms 安全版（2026-10-01）：spec 打包是纯 CPU（只读不可变快照 + 写本地缓冲），
        // 与 linear 段（打包 + 等哈希池）没有任何依赖，此前却排在 linear 之后串行执行。
        // 这里把它放到独立线程，与 linear 段重叠；linear 段实测有 94.5ms 是请求线程纯等池，
        // 而 spec 打包只需 84.5ms CPU，正好塞得进去（101 个真机样本的中位）。
        // 语义不变：存储/池的提交与取回顺序完全没动，只是把一段独立的 CPU 工作提前。
        std::vector<char> speculativePacked;
        // 本 .inc 是**在类体内部**被包含的（qwen3_5.cpp:9089），因此不能在这里 #include 任何
        // 标准头（会把 std 嵌进类里直接编译失败——已实测踩中）。改用本文件已在使用的
        // std::thread + std::exception_ptr 做并行，异常在 join 之后原样重抛给请求线程。
        Json speculative;
        double specPackMs = 0;
        std::exception_ptr specFailure;
        std::thread specThread([&]() {
            try {
                const auto specPackStart = exNow();
                speculative = SaveSpeculative(speculativePacked, *job.snapshot);
                specPackMs = exMs(specPackStart, exNow());
            } catch (...) {
                specFailure = std::current_exception();
            }
        });
        const auto linearStoreStart = exNow();
        auto linearChunks = StoreHashed(writer, linearBytes,[&](size_t offset,char *out,size_t count) {
            memcpy(out,packed.data()+offset,count);
        }, 1, lane.hashPool.get(), lane.storeSubmitMs, lane.storeTakeMs);
        exLinearStore += exMs(linearStoreStart, exNow());
        std::vector<char>().swap(packed);
        const double linearSegmentMs = exMs(linearStart, exNow());
        exLinear += linearSegmentMs;
        specThread.join();
        if (specFailure) std::rethrow_exception(specFailure);
        const auto specStoreStart = exNow();
        exSpecPack += specPackMs;
        // 被 linear 段遮住的部分（不可能超过 linear 段本身的墙钟）。
        exSpecHidden += std::min(specPackMs, linearSegmentMs);
        const size_t speculativeBytes = speculativePacked.size();
        Json::array speculativeChunks;
        if (speculativeBytes > 0) {
            speculativeChunks = StoreHashed(writer, speculativeBytes,
                [&](size_t offset,char *out,size_t count) {
                    memcpy(out,speculativePacked.data()+offset,count);
                }, 2, lane.hashPool.get(), lane.storeSubmitMs, lane.storeTakeMs);
        }
        exSpecStore += exMs(specStoreStart, exNow());
        // exSpec 仍表示 spec 自己的工作量（打包 + 进池取回），不含被并行遮住的 linear 段墙钟。
        exSpec += specPackMs + exMs(specStoreStart, exNow());
        const auto commitStart = exNow();
        PrefixCacheRecord durableRecord;
        durableRecord.key = job.key; durableRecord.identity = identity;
        durableRecord.provenance = state.provenance; durableRecord.length = job.length;
        durableRecord.requiredComponents = PrefixCacheRecord::FullAttentionKV | PrefixCacheRecord::LinearState;
        if (job.snapshot->mtpValid || job.snapshot->dflashValid)
            durableRecord.requiredComponents |= PrefixCacheRecord::SpeculativeState;
        // Bind the durable v4 checkpoint to the same canonical page identities
        // used by L1/L2 migration.  Without this registration the runtime page
        // ledger can never prove that an L2 page has an SSD recovery root, so
        // safe L2 eviction is permanently disabled.
        if (model->pagedPageLedger && job.pagedCheckpointGeneration == 0) {
            job.pagedCheckpointIdentity = identity;
            job.pagedCheckpointPages = model->DescribePagedPrefixPages(
                job.snapshot->tokens,
                job.runtimeUsesPageKeys ? job.snapshot->mediaPageKeys :
                                          std::vector<std::string>(),
                job.length);
            if (!job.pagedCheckpointPages.empty()) {
                job.pagedCheckpointGeneration = model->pagedPageLedger->RegisterCheckpoint(
                    job.key, job.pagedCheckpointIdentity, job.pagedCheckpointPages);
                PrefixCacheDebug("[PrefixCache] paged-checkpoint-register: key=%s "
                                 "pages=%zu generation=%llu.\n",
                                 job.key.c_str(), job.pagedCheckpointPages.size(),
                                 (unsigned long long)job.pagedCheckpointGeneration);
            }
        }
        const auto pagedLedger = model->pagedPageLedger;
        const std::string pagedKey = job.key;
        const int pageLen = job.pages.front()->pool->pageLen;
        const std::vector<std::string> pageIdentity = ComputePageIdentities(
            job.snapshot->tokens, job.length, pageLen,
            job.runtimeUsesPageKeys ? &job.snapshot->mediaPageKeys : nullptr);
        const std::string pagedIdentity = job.pagedCheckpointIdentity;
        const uint64_t pagedGeneration = job.pagedCheckpointGeneration;
        const DiskPrefixCache::CommitIdentity commitIdentity{
            job.request_id, job.request_generation, job.page_generation, job.copy_generation,
            job.cache_identity, uint64_t(reinterpret_cast<uintptr_t>(job.owner_lease.get()))};
        Check(commitIdentity.Valid(), "identity_missing");
        Json::object commitState{{"kind","prefix"},{"length",job.length},
            {"family_identity",family},{"kv_dtype",Dtype(model->kvCacheDataType)},
            {"provenance",state.provenance},{"source",state.source},{"prefix",job.prefix},{"layout",RuntimeLayout()},
            {"kv",state.kv},{"linear",linear},{"linear_bytes",(double)linearBytes},
            {"linear_chunks",linearChunks},{"speculative",speculative},
            {"speculative_bytes",(double)speculativeBytes},
            {"speculative_chunks",speculativeChunks},{"page_identity",pageIdentity}};
        // V4 records stay byte-identical: the field exists only when V5 is on.
        if (fastllm::DiskPrefixV5Enabled()) commitState["v5_prompt_end"] = job.promptEnd;
        const bool published = writer.CommitTracked(job.key,Json(commitState),true, commitIdentity,
            [expectedIdentity = commitIdentity, spill = model->pagedSpillState,
             statuses = job.writeStatuses, ledger = prefixLedger, durableRecord, pagedLedger,
             pagedKey, pagedIdentity, pagedGeneration,
             exportState = std::weak_ptr<ExportState>(job.state)](bool durable,
                 const DiskPrefixCache::CommitIdentity &callbackIdentity) {
                if (!(callbackIdentity == expectedIdentity)) {
                    std::fprintf(stderr, "[Prefix SSD] publish-rejected: status=IDENTITY_REJECTED "
                                 "reason=commit_identity_mismatch request_id=%llu.\n",
                                 (unsigned long long)expectedIdentity.requestId);
                    return;
                }
                // V5 commits are asynchronous: a skipped/failed commit leaves later checkpoints of
                // this request pointing at bundles that were never stored (stab4 s2: 59
                // v5_parent_missing after 5 quota skips). Mark the export state failed so the next
                // Capture rebuilds it from a stored parent (existing slot->failed -> makeSlot path).
                if (!durable && RequestHoldEnabled())
                    if (auto st = exportState.lock()) st->failed = true;
                // The existing PrepJob already owns all packed state and KV
                // until durability; no model tensor lifetime extends into actor.
                Qwen35FinishSpillWrite(spill, statuses, durable);
                if (pagedLedger && pagedGeneration != 0) {
                    try {
                        const bool finished = pagedLedger->FinishCheckpoint(
                            pagedKey, pagedIdentity, pagedGeneration, durable,
                            durable ? std::string() : "v4 checkpoint commit failed");
                        PrefixCacheDebug("[PrefixCache] paged-checkpoint-finish: key=%s "
                                         "generation=%llu durable=%d result=%d.\n",
                                         pagedKey.c_str(),
                                         (unsigned long long)pagedGeneration,
                                         durable ? 1 : 0, finished ? 1 : 0);
                    } catch (const std::exception &error) {
                        std::fprintf(stderr,
                                     "[PrefixCache] paged durable ledger update failed: reason=%s.\n",
                                     error.what());
                    }
                }
                // Already admitted writes keep the same actors, bytes and
                // acknowledgement point. Only their logical SSD state changes.
                try { ledger->Durable(durableRecord,durable); }
                catch (const std::exception &error) {
                    std::fprintf(stderr,"[PrefixCache] durable ledger update failed: reason=%s.\n",error.what());
                }
            });
        exCommit += exMs(commitStart, exNow());
        if (published) ++committed;
        if (published) {
            try { UpdatePrefetchHint(job); }
            catch (const std::exception &error) {
                fprintf(stderr,"[Prefix SSD] prefetch hint skipped: %s.\n",error.what());
            }
        }
        // §⑥ 备菜流水线口径：这里的 true 只代表"已受理"（状态 Json + 暂存分块已移交
        // 后台唯一备菜线程），落盘结果在下一次 reap / 读侧收尾 / 关闭时上报。
        // 因此 published 现在恒为 true；"内容已在库里"的去重命中改由备菜线程识别为
        // no-op（结果仍回报成功），日志不再出现 already_present。文案必须如实反映"已受理"。
        lane.saveTerminalNs = FastllmObservationTimestampNs();
        PrefixCacheDebug("[Prefix SSD] %s: tokens=%d key=%s.\n",
               published ? "commit-submitted" : "already_present",job.length,job.key.c_str());
        double hashSubmitMs = 0, hashWaitMs = 0, hashWaitLinear = 0, hashWaitSpec = 0;
        for (int i = 0; i < 3; ++i) {
            const double submit = lane.storeSubmitMs[i] - lane.lastStoreSubmitMs[i];
            const double wait = lane.storeTakeMs[i] - lane.lastStoreTakeMs[i];
            lane.lastStoreSubmitMs[i] = lane.storeSubmitMs[i];
            lane.lastStoreTakeMs[i] = lane.storeTakeMs[i];
            hashSubmitMs += submit; hashWaitMs += wait;
            if (i == 1) hashWaitLinear = wait;
            if (i == 2) hashWaitSpec = wait;
        }
        const double hashWaitStore = hashWaitMs - hashWaitLinear - hashWaitSpec;
        // A2（2026-10-02）本检查点的哈希/服务口径（命名与既有 hash_* 一致，均为**单检查点**增量）：
        //   hash_cpu_sum_ms      = 花在哈希流水线上的墙钟总和 = D2 泵(exD2h) + linear/spec 提交与取回
        //                          + D2 取回(d2TakeDelta)。并行时会大于墙钟，用于判断哈希是否瓶颈。
        //   hash_critical_wall_ms= 其中真正阻塞等结果的墙钟 = linear/spec 取回 + D2 取回，即关键路径。
        //   capture_background_wait_ms = 等"后台流水线"（D2H/哈希）产出结果的墙钟
        //                          = D2 取回未命中的等待(exD2Wait) + linear/spec 取回等待(hashWaitMs)。
        //   service_ms           = 本检查点的服务墙钟（Export 入口→本行）；end_to_end_ms = 入队→本行。
        //   capture_cpu_ms       = **会话累计**：请求线程在 Capture() 里同步花掉的 CPU 墙钟。
        const double d2TakeDelta = lane.d2TakeMs - lane.lastD2TakeMs;
        lane.lastD2TakeMs = lane.d2TakeMs;
        const double hashCpuSumMs = hashSubmitMs + hashWaitMs + exD2h + d2TakeDelta;
        const double hashCriticalWallMs = hashWaitMs + d2TakeDelta;
        const double captureBackgroundWaitMs = exD2Wait + hashWaitMs;
        const double serviceMs = exMs(exEntry, exNow());
        const double endToEndMs = exMs(job.enqueuedAt, exNow());
        lane.hashCpuSumMs += hashCpuSumMs;
        lane.hashCriticalWallMs += hashCriticalWallMs;
        lane.captureBackgroundWaitMs += captureBackgroundWaitMs;
        lane.serviceMs += serviceMs;
        lane.endToEndMs += endToEndMs;
        ++lane.jobsDone;
        // S4-C：整段 keepalive（方案 C）的**额外持有页字节峰值**——观察"保护整段"带来的
        // 容量压力（与 pinnedGpuBytes 的窗口记账是两回事）。全局计数，读时持 mutex。
        uint64_t exportProtectedPeakBytes = 0;
        { std::lock_guard<std::mutex> lock(mutex); exportProtectedPeakBytes = exportProtectedPeak; }
        PrefixCacheDebug("[Prefix SSD] export-profile: tokens=%d d2h_ms=%.1f store_ms=%.1f linear_ms=%.1f spec_ms=%.1f commit_ms=%.1f total_ms=%.1f setup_ms=%.1f linear_pack_ms=%.1f linear_store_ms=%.1f spec_pack_ms=%.1f spec_store_ms=%.1f hash_submit_ms=%.1f hash_wait_ms=%.1f hash_wait_store_ms=%.1f hash_wait_linear_ms=%.1f hash_wait_spec_ms=%.1f spec_hidden_ms=%.1f hash_mode=%s service_ms=%.1f end_to_end_ms=%.1f capture_cpu_ms=%.1f capture_background_wait_ms=%.1f hash_cpu_sum_ms=%.1f hash_critical_wall_ms=%.1f export_protected_bytes=%llu expected_checkpoints=%llu degraded=%llu enqueue_ns=%llu save_start_ns=%llu save_terminal_ns=%llu terminal_kind=commit_submitted lane=%zu.\n",
               job.length,exD2h,exStore,exLinear,exSpec,exCommit,
               exD2h+exStore+exLinear+exSpec+exCommit,
               exSetup,
               exLinearPack,exLinearStore,exSpecPack,exSpecStore,
               hashSubmitMs,hashWaitMs,
               hashWaitStore,hashWaitLinear,hashWaitSpec,exSpecHidden,
               PooledHashEnabled() ? "pool" : "inline",
               serviceMs,endToEndMs,
               double(captureCpuNs.load())/1e6,captureBackgroundWaitMs,
               hashCpuSumMs,hashCriticalWallMs,
               (unsigned long long)exportProtectedPeakBytes,
               (unsigned long long)expectedCheckpoints.load(),
               (unsigned long long)degraded.load(),
               (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                   job.enqueuedAt.time_since_epoch()).count(),
               (unsigned long long)lane.saveStartNs,(unsigned long long)lane.saveTerminalNs,
               laneIndex);
        std::string stateStageBytesText = "none";
        if (!job.stateStageBytes.empty()) {
            stateStageBytesText.clear();
            for (const auto &entry : job.stateStageBytes) {
                if (!stateStageBytesText.empty()) stateStageBytesText += ",";
                stateStageBytesText += std::to_string(entry.first) + ":" + std::to_string(entry.second);
            }
        }
        PrefixCacheDebug("[Prefix SSD] state-stage-profile: stage_bytes=%s stage_fallback_sync=%llu "
                         "stage_fallback_reason=%s d2d_enqueue_ns=%llu ready_wait_ns=%llu staged_materialize_ns=%llu.\n",
                         stateStageBytesText.c_str(),
                         (unsigned long long)job.stateStageFallbackSyncCount,
                         job.stateStageFallbackReason.empty() ? "none" : job.stateStageFallbackReason.c_str(),
                         (unsigned long long)job.stateStageD2DEnqueueNs,
                         (unsigned long long)job.stateStageReadyWaitNs,
                         (unsigned long long)job.stateStageMaterializeNs);
        // A-6 fix F: one line per device, on this export worker thread (off the compute path).
        if (Qwen35GpuTimelineEnabled()) {
            for (const auto &begin : lane.timelineBeginMs) {
                const auto end = lane.timelineEndMs.find(begin.first);
                std::printf("[GpuTimeline] export-save: lane=%zu device=%d tokens=%d save_begin_dev_ms=%.3f "
                            "save_end_dev_ms=%.3f save_start_ns=%llu save_terminal_ns=%llu request=%llu "
                            "generation=%llu.\n",
                            laneIndex, begin.first, job.length, begin.second,
                            end != lane.timelineEndMs.end() ? end->second : -1.0,
                            (unsigned long long)lane.saveStartNs, (unsigned long long)lane.saveTerminalNs,
                            (unsigned long long)job.request_id, (unsigned long long)job.request_generation);
            }
            std::fflush(stdout);
        }
        // S4（计划 §17.3 要求）：每检查点采一次高水位相关量——队列、主机信用、池在途字节、
        // D2-stream 待拷贝/占槽。峰值由离线分析取 max，避免"只有均值、看不到尖峰"。
        // R1：当前快照量对全部 lane 求和（各 stream/pool 的查询都自带锁，跨 lane 读安全）；
        // **峰值只报本 lane**（d2PendingMax/d2SlotsMax 是本 lane 独占的非原子字段，跨 lane 读会
        // 形成数据竞争，故不做跨 lane 聚合；全局 pinned 槽峰值由协调器的 slot_peak 提供）。
        if (PrefixCacheDebugEnabled()) {
            std::lock_guard<std::mutex> lock(mutex);
            const uint64_t credits = coordinator ? coordinator->HostCreditsInUse() : 0;
            const uint64_t storeInflight = store ? store->InFlightBytes() : 0;
            uint64_t poolInflight = 0, poolPeakTotal = 0, d2Pending = 0, d2Slots = 0;
            for (auto &one : writeLanes) {
                if (!one) continue;
                if (one->hashPool) {
                    poolInflight += one->hashPool->InFlightBytes();
                    poolPeakTotal += one->hashPool->InFlightPeak();   // 池内部有锁，跨 lane 读安全
                }
                if (one->d2Stream) {
                    d2Pending += one->d2Stream->Pending();
                    d2Slots += uint64_t(one->d2Stream->SlotsInUse());
                }
            }
            const uint64_t d2PendingPeak = lane.d2PendingMax;   // 本 lane 的峰值（见上方注释）
            const uint64_t d2SlotsPeak = lane.d2SlotsMax;
            // A2 诊断快照（协调器侧全局计数 + 模型侧会话计数）。
            const auto pinnedStats = fastllm::prefix_write::PrefixWritePinnedSlotStatsSnapshot();
            const auto creditStats = fastllm::prefix_write::PrefixWriteHostCreditStatsSnapshot();
            const auto ownerStats = fastllm::prefix_write::PrefixWriteOwnerWaiterStatsSnapshot();
            printf("[Prefix SSD] ssd-watermark: queued=%zu working=%d host_credits=%llu pool_inflight=%llu d2_pending=%llu d2_slots=%d host_limit=%llu store_inflight=%llu pool_peak=%llu d2_pending_peak=%llu d2_slots_peak=%llu pool_peak_total=%llu "
                   "slot_acquires=%llu slot_waits=%llu slot_peak=%llu slot_reuses=%llu slot_releases=%llu "
                   "credit_acquires=%llu credit_releases=%llu credit_denied=%llu credit_peak_bytes=%llu "
                   "page_lease_releases=%llu owners=%llu joined_waiters=%llu conflicts=%llu durable_reuse=%llu lease_expired=%llu "
                   "admitted=%llu terminal=%llu credit_balance=%llu expected_checkpoints=%llu degraded=%llu lanes=%d.\n",
                   queuedJobs, (int)activeWorkers, (unsigned long long)credits,
                   (unsigned long long)poolInflight,
                   (unsigned long long)d2Pending,
                   int(d2Slots),
                   (unsigned long long)HostLimit(),
                   (unsigned long long)storeInflight,
                   (unsigned long long)poolPeakTotal,
                   (unsigned long long)d2PendingPeak,
                   (unsigned long long)d2SlotsPeak,
                   (unsigned long long)poolPeakTotal,
                   (unsigned long long)pinnedStats.acquires, (unsigned long long)pinnedStats.waits,
                   (unsigned long long)pinnedStats.peak, (unsigned long long)pinnedStats.reuses,
                   (unsigned long long)pinnedStats.releases,
                   (unsigned long long)creditStats.acquires, (unsigned long long)creditStats.releases,
                   (unsigned long long)creditStats.denied, (unsigned long long)creditStats.peakBytes,
                   (unsigned long long)pageLeaseReleases.load(),
                   (unsigned long long)ownerStats.owners, (unsigned long long)ownerStats.joinedWaiters,
                   (unsigned long long)ownerStats.conflicts, (unsigned long long)ownerStats.durableReuse,
                   (unsigned long long)fastllm::prefix_write::PrefixWriteLeaseExpiredTotal(),
                   (unsigned long long)admittedCheckpoints.load(),
                   (unsigned long long)terminalCheckpoints.load(),
                   (unsigned long long)credits,
                   (unsigned long long)expectedCheckpoints.load(),
                   (unsigned long long)degraded.load(), writeWorkerCount);
            // LEDGERBOUND: page-ledger size (O(1) counters), at most every 5 s, only when
            // FASTLLM_PREFIX_LEDGER_BOUND=1 or FASTLLM_PREFIX_LEDGER_STATS=1.
            if (model->pagedPageLedger && PrefixCachePageLedger::LedgerStatsEnabled()) {
                static std::atomic<int64_t> lastLedgerPrintMs{0};
                const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                int64_t last = lastLedgerPrintMs.load(std::memory_order_relaxed);
                if (nowMs - last >= 5000 &&
                    lastLedgerPrintMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed)) {
                    const auto ls = model->pagedPageLedger->Stats();
                    printf("[PrefixCache] ledger-stats: bound=%d checkpoints=%zu pages=%zu root_refs=%zu "
                           "durable_page_refs=%zu budget_page_refs=%zu forgotten=%zu.\n",
                           PrefixCachePageLedger::LedgerBoundEnabled() ? 1 : 0, ls.checkpoints, ls.pages,
                           ls.rootRefs, ls.durablePageRefs, PrefixCachePageLedger::LedgerMaxDurablePageRefs(),
                           ls.forgotten);
                }
            }
            // KVPAIR: opt-in K/V page-manager pair check (FASTLLM_PAGED_KV_PAIR_CHECK=1), at most every 5 s.
            // Prints each layer/rank whose K and V managers differ, with the first difference.
            static const bool pairCheck = [] { const char *v = std::getenv("FASTLLM_PAGED_KV_PAIR_CHECK"); return v && std::string(v) == "1"; }();
            if (pairCheck) {
                static std::atomic<int64_t> lastPairCheckMs{0};
                const int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                int64_t last = lastPairCheckMs.load(std::memory_order_relaxed);
                if (nowMs - last >= 5000 &&
                    lastPairCheckMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed)) {
                    int checked = 0, diverged = 0, linked = 0;
                    unsigned long long pairAllocs = 0, pairConsumed = 0, pairCancelled = 0, pairFallbacks = 0, reservedNow = 0;
                    for (int layer = 0; layer < model->block_cnt; ++layer) {
                        const auto keys = model->GetPagedKVCacheManagers(layer, true);
                        const auto values = model->GetPagedKVCacheManagers(layer, false);
                        for (size_t r = 0; r < keys.size() && r < values.size(); ++r) {
                            if (!keys[r].second || !values[r].second || keys[r].second == values[r].second) continue;
                            ++checked;
                            for (PagedCacheManager *m : {keys[r].second, values[r].second}) {
                                std::lock_guard<std::mutex> guard(m->pageIndexLocker);
                                pairAllocs += m->pairAllocs; pairConsumed += m->pairConsumed;
                                pairCancelled += m->pairCancelled; pairFallbacks += m->pairFallbacks;
                                reservedNow += m->pairReserved.size();
                            }
                            if (keys[r].second->pairPartner == values[r].second) ++linked;
                            const std::string diff = keys[r].second->PairDiff(*values[r].second);
                            if (!diff.empty()) {
                                ++diverged;
                                printf("[KVPAIR] diverged layer=%d rank=%zu device=%d diff=%s.\n", layer, r, keys[r].first, diff.c_str());
                            }
                        }
                    }
                    printf("[KVPAIR] pair-check: pairs=%d diverged=%d linked=%d pair_allocs=%llu consumed=%llu cancelled=%llu fallbacks=%llu reserved_now=%llu.\n",
                           checked, diverged, linked, pairAllocs, pairConsumed, pairCancelled, pairFallbacks, reservedNow);
                }
            }
            fflush(stdout);
            // 峰值按检查点重置：只重置**本 lane** 的峰值，本 lane 下一次打印的仍是"下一张检查点"
            // 的峰值；全局 pinned 槽峰值看 slot_peak。
        }
        lane.poolInflightMax = lane.d2PendingMax = lane.d2SlotsMax = 0;
    }
    // [store-profile] 写盘路径子项计数：验证"同一批字节被 SHA256 算了 3 遍"
    if (PrefixCacheDebugEnabled()) {
        {
            const auto journal = fastllm::prefix_v4::JournalStatsSnapshot();
            printf("[Prefix SSD] v4-journal: calls=%llu bytes=%.2fMiB write_ms=%.1f sync_ms=%.1f.\n",
                   (unsigned long long)(journal.calls - lane.lastJournalStats.calls),
                   (journal.bytes - lane.lastJournalStats.bytes) / 1048576.0,
                   (journal.writeNs - lane.lastJournalStats.writeNs) / 1e6,
                   (journal.syncNs - lane.lastJournalStats.syncNs) / 1e6);
            lane.lastJournalStats = journal;
        }
        // §14：v4 存储层计数（累计值，不是增量）。
        {
            // §⑥ 热路径：drain=false —— 本行每检查点都打，若在此排空备菜流水线，就会把
            // "请求线程只交出状态就返回、后台备菜与写盘重叠"的收益全部抵消。
            // 代价：本行 live_* 与分段计数可能滞后一笔在途提交（离线差分口径不变）。
            const DiskPrefixCache::Json v4 = store ? store->StorageProfile(false) : DiskPrefixCache::Json::object{};
            if (!v4.object_items().empty()) {
                printf("[Prefix SSD] v4-store-cum: pack_ms=%.1f catalog_ms=%.1f journal_write_ms=%.1f "
                       "journal_sync_ms=%.1f dir_syncs=%.0f dir_sync_ms=%.1f "
                       "rebuild_valid=%.0f rebuild_tail=%.0f rebuild_corrupt=%.0f rebuild_excluded=%.0f "
                       "sqlite_sync=%s data_sync_ms=%.1f/%.0f control_sync_ms=%.1f/%.0f "
                       "record_hash=%.1f/%.1fMiB metadata_encode_ms=%.1f ref_walk_ms=%.1f "
                       "assembly[quota=%.1f list=%.1f flush=%.1f prepare=%.1f commit=%.1f rootTx=%.1f evict=%.1f "
                       "loop[lookup=%.1f pscan=%.1f stage=%.1f assemble=%.1f objs=%.0f psteps=%.0f] "
                       "precheck_scan_ms=%.1f actor_wait_ms=%.1f actor_queue_ms=%.1f actor_exec_ms=%.1f "
                       "metadata_hash_ms=%.1f object_verify_ms=%.1f/%.0fMiB object_write_ms=%.1f "
                       "commit_lock_ms=%.1f actor_cpu_ms=%.1f live_checkpoints=%.0f live_bytes=%.0f "
                       "prep_exec_ms=%.1f prep_queue_ms=%.1f prep_jobs=%.0f prep_max_depth=%.0f "
                       "durable_checkpoints=%.0f expected_checkpoints=%llu degraded=%llu.\n",
                       v4["object_pack_ms"].number_value(), v4["catalog_apply_ms"].number_value(),
                       v4["journal_write_ms"].number_value(), v4["journal_sync_ms"].number_value(),
                       v4["dir_syncs"].number_value(), v4["dir_sync_ms"].number_value(),
                       v4["rebuild_valid"].number_value(), v4["rebuild_uncommitted_tail"].number_value(),
                       v4["rebuild_corrupt_control"].number_value(), v4["rebuild_excluded_roots"].number_value(),
                       v4["sqlite_sync"].string_value().c_str(),
                       v4["data_sync_ms"].number_value(), v4["data_sync_count"].number_value(),
                       v4["control_sync_ms"].number_value(), v4["control_sync_count"].number_value(),
                       v4["record_hash_ms"].number_value(), v4["record_hash_bytes"].number_value()/1048576.0,
                       v4["metadata_encode_ms"].number_value(), v4["ref_walk_ms"].number_value(),
                       v4["quota_check_ms"].number_value(), v4["root_list_ms"].number_value(),
                       v4["flush_objects_ms"].number_value(), v4["prepare_ms"].number_value(),
                       v4["commit_record_ms"].number_value(), v4["root_tx_ms"].number_value(),
                       v4["evict_ms"].number_value(),
                       v4["lookup_ms"].number_value(), v4["pending_scan_ms"].number_value(),
                       v4["stage_ms"].number_value(), v4["assemble_ms"].number_value(),
                       v4["object_count"].number_value(), v4["pending_scan_steps"].number_value(),
                       v4["precheck_scan_ms"].number_value(), v4["actor_wait_ms"].number_value(),
                       v4["actor_queue_ms"].number_value(), v4["actor_exec_ms"].number_value(),
                       v4["metadata_hash_ms"].number_value(),
                       v4["object_verify_ms"].number_value(),
                       v4["object_verify_bytes"].number_value()/1048576.0,
                       v4["object_write_ms"].number_value(), v4["commit_lock_ms"].number_value(),
                       v4["actor_cpu_ms"].number_value(),
                       v4["live_checkpoints"].number_value(), v4["live_bytes"].number_value(),
                       // §⑥ 备菜流水线：新阶段计数（备菜执行时间 / 队列等待 / 作业数 / 峰值深度）
                       v4["prep_exec_ms"].number_value(), v4["prep_queue_ms"].number_value(),
                       v4["prep_jobs"].number_value(), v4["prep_max_depth"].number_value(),
                       // A2：durable_checkpoints = 本进程 store 侧成功 COMMIT 的检查点累计；
                       // expected = 模型侧 native capture 事件数；degraded = 被丢弃的检查点数。
                       // 三者应满足 durable + 在途/被丢弃 = expected（在途可由 store_inflight 观察）。
                       v4["durable_checkpoints_cum"].number_value(),
                       (unsigned long long)expectedCheckpoints.load(),
                       (unsigned long long)degraded.load());
            }
        }
        DiskPrefixCache::Json p;
        {
            const DiskPrefixCache::Json current = DiskPrefixCache::StoreProfileSnapshot();
            if (lane.lastProfileSnapshot.is_object()) {
                std::map<std::string, DiskPrefixCache::Json> delta;
                for (const auto &entry : current.object_items())
                    delta[entry.first] = DiskPrefixCache::Json(
                        entry.second.number_value() - lane.lastProfileSnapshot[entry.first].number_value());
                p = DiskPrefixCache::Json(delta);
            } else {
                p = current;   // 首次调用没有基线，直接报累计值
            }
            lane.lastProfileSnapshot = current;
        }
        printf("[Prefix SSD] store-profile: tokens=%d chunks=%.0f flushes=%.0f "
               "digest_store=%.0f/%.2fMiB digest_record=%.0f/%.2fMiB digest_index=%.0f/%.2fMiB "
               "index_inserts=%.0f db_opens=%.0f flush_db_opens=%.0f find=%.0f hits=%.0f readbacks=%.0f/%.2fMiB "
               "alloc_ms=%.1f copy_ms=%.1f hash_ms=%.1f db_ms=%.1f json_ms=%.1f flush_ms=%.1f.\n",
               job.length,
               p["chunks"].number_value(), p["flushes"].number_value(),
               p["store_digest_calls"].number_value(), p["store_digest_bytes"].number_value()/1048576.0,
               p["record_digest_calls"].number_value(), p["record_digest_bytes"].number_value()/1048576.0,
               p["index_digest_calls"].number_value(), p["index_digest_bytes"].number_value()/1048576.0,
               p["index_inserts"].number_value(),
               p["db_opens"].number_value(), p["flush_db_opens"].number_value(),
               p["find_calls"].number_value(), p["find_hits"].number_value(),
               p["readbacks"].number_value(), p["readback_bytes"].number_value()/1048576.0,
               p["alloc_ms"].number_value(), p["copy_ms"].number_value(), p["hash_ms"].number_value(),
               p["db_ms"].number_value(), p["json_ms"].number_value(), p["flush_ms"].number_value());
    }
    writes += writer.WrittenBytes()-before;
}

void Qwen35PersistentCache::Work(size_t laneIndex) {
    // R1：本线程独占 writeLanes[laneIndex]；只通过 mutex 访问队列/账本等共享状态。
    WriteLane &lane = *writeLanes[laneIndex];
    void *buffer = FastllmCudaHostMalloc(BufferBytes);
    // §④ 宽版：K 个 pinned 槽位（一块连续分配、按 BufferBytes 切分；D2-stream 只给槽号）。
    // R1：**每个 lane 独立分配自己的 pinned 内存与槽池**，绝不与别的 lane 共享（共享会让两个
    // 并发 D2H 写进同一块缓冲，也会让槽号互相串用）。
    if (!lane.d2SlotMemory) {
        // A2/W4（2026-10-04）：槽位数与协调器池容量**必须同源** ⇒ 统一取 PinnedSlots()
        // （env `FASTLLM_PREFIX_CACHE_PINNED_SLOTS`，**默认 = kPinnedSlots = 4**，见
        // include/utils/disk_prefix_write_coordinator.h:119 与 PinnedSlots()；此前注释误写"默认 8"，
        // 已更正——默认 8 会直接违反契约测试 SlotCapacity()==kPinnedSlots 的 3 处断言）。
        const size_t slots = size_t(fastllm::prefix_write::PinnedSlots());
        lane.d2SlotMemory = FastllmCudaHostMalloc(slots * BufferBytes);
        lane.d2SlotBytes = lane.d2SlotMemory ? slots * BufferBytes : 0;
        Check(lane.d2SlotMemory != nullptr, "ssd_d2_slots_alloc");
    }
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex);
            auto ready = [&] { return stopping || !laneQueues[laneIndex].empty(); };
            if (laneIndex == 0 && PrefixCacheDebugEnabled()) {
                while (!ready()) {
                    if (!changed.wait_for(lock, std::chrono::milliseconds(500), ready))
                        PrintIdleWatermarkLocked();
                }
            } else {
                changed.wait(lock, ready);
            }
            if (laneQueues[laneIndex].empty() && stopping) break;
            job = std::move(laneQueues[laneIndex].front()); laneQueues[laneIndex].pop_front();
            --queuedJobs; ++activeWorkers;
        }
        RequestDiagnosticOperation diagnosticJob(job.diagnosticToken, "ssd_export_worker",
            "Qwen35PersistentCache.Export");
        std::string identityError;
        {
            std::lock_guard<std::mutex> identityLock(mutex);
            // Accepted writes own their snapshot/page leases, not the live request.
            // Keep immutable identity checks here; D2 still validates page content
            // and waits for the producer before copying any leased page.
            ValidateJobIdentityLocked(job, "worker_start", identityError, false);
        }
        bool failed = false;
        lane.saveStartNs = 0;
        lane.saveTerminalNs = 0;
        lane.timelineBeginRecorded.clear();
        lane.timelineBeginMs.clear();
        lane.timelineEndMs.clear();
        try {
            Check(identityError.empty(), identityError.empty() ? "identity_ok" : identityError.c_str());
            Check(buffer != nullptr,"ssd_pinned_allocation");
            Check(job.stateBytes <= HostLimit(), "ssd_snapshot_larger_than_host_budget");
            if (job.snapshot && coordinator) {
                RequestDiagnosticOperation diagnosticAdmission(job.diagnosticToken,
                    "ssd_coordinator_admission_wait", "prefix_write.Coordinator.TryAdmit");
                for (;;) {
                    prefix_write::TicketId admitted;
                    const auto result = coordinator->TryAdmit(job.ticket,
                        job.key + ":" + std::to_string(job.length),
                        std::max<size_t>(job.stateBytes, 1), admitted);
                    if (result == prefix_write::Admission::Accepted) {
                        job.ticket = admitted; job.admitted = true; ++admittedCheckpoints;
                        break;
                    }
                    Check(result == prefix_write::Admission::CapacityUnavailable,
                          "ssd_background_admission_failed");
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait_for(lock, std::chrono::milliseconds(10));
                }
            }
            if (job.stateStage) {
                Check(Qwen35StateStageIdentityMatches(job.stateStage, job.request_id,
                    job.request_generation, job.copy_generation, job.cache_identity),
                    "state_stage_identity");
                Check(Qwen35StateStageMaterialize(job.stateStage, lane.d2SlotMemory,
                    lane.d2SlotBytes, lane.d2CopyStreams), "state_stage_materialize_failed");
                job.stateStageReadyWaitNs = job.stateStage->readyWaitNs;
                job.stateStageMaterializeNs = job.stateStage->materializeNs;
                Qwen35PublishMaterializedSnapshot(model, job.snapshot);
                if (job.snapshot) job.snapshot->stateStage.reset();
                job.stateStage.reset();
            }
            Export(job,buffer,laneIndex);
        }
        catch (const std::exception &e) {
            failed = true;
            if (job.stateStage) fastllm::qwen35_state_stage::Fail(job.stateStage);
            ++writeFailures;
            if (job.state) {
                job.state->failed = true;
                // Keep owned staged bytes and uncopied pages on real I/O failure.
            }
            ++skipped;
            if (job.snapshot) ++degraded; // Failed I/O is retained and reported.
            if (job.pagedCheckpointGeneration != 0 && model->pagedPageLedger) {
                try {
                    model->pagedPageLedger->FinishCheckpoint(
                        job.key, job.pagedCheckpointIdentity,
                        job.pagedCheckpointGeneration, false,
                        "v4 export failed before durable completion");
                } catch (...) {
                    // Preserve the original export failure; the ledger remains
                    // fail-closed if its terminal update itself fails.
                }
            }
            Qwen35FinishSpillWrite(model->pagedSpillState, job.writeStatuses, false);
            lane.saveTerminalNs = FastllmObservationTimestampNs();
            std::fprintf(stderr,"[Prefix SSD] write_failed_retained: tokens=%d reason=%s "
                         "save_start_ns=%llu save_terminal_ns=%llu terminal_kind=write_failed_retained lane=%zu.\n",
                         job.length,e.what(),(unsigned long long)lane.saveStartNs,
                         (unsigned long long)lane.saveTerminalNs,laneIndex);
        }
        // A-6 fix E rev2：在导出线程（不是存储 actor）上打印新增的回收否决；只有一个 lane 抢到打印权。
        {
            const uint64_t vetoes = reclaimVetoLog->count.load(std::memory_order_acquire);
            uint64_t printed = reclaimVetoPrinted.load(std::memory_order_acquire);
            if (vetoes != printed &&
                reclaimVetoPrinted.compare_exchange_strong(printed, vetoes, std::memory_order_acq_rel)) {
                std::string lastKey, lastIdentity;
                {
                    std::lock_guard<std::mutex> guard(reclaimVetoLog->mutex);
                    lastKey = reclaimVetoLog->lastKey;
                    lastIdentity = reclaimVetoLog->lastIdentity;
                }
                const std::string vetoDiag = model->pagedPageLedger
                    ? model->pagedPageLedger->VetoDiagnostics() : std::string("no_ledger");
                PrefixCacheDebug("[PrefixCache] checkpoint-reclaim-veto: total=%llu new=%llu last_key=%s "
                                 "last_identity=%s lane=%zu %s.\n",
                                 (unsigned long long)vetoes, (unsigned long long)(vetoes - printed),
                                 lastKey.c_str(), lastIdentity.c_str(), laneIndex, vetoDiag.c_str());
            }
        }
        // 作业结束必须归还预算与槽位（成功/失败都一样，§7.2）。
        if (job.admitted && coordinator) {
            RequestDiagnosticOperation diagnosticTerminal(job.diagnosticToken,
                "ssd_coordinator_complete", "prefix_write.Coordinator.Complete");
            coordinator->Complete(job.ticket, failed ? fastllm::prefix_write::Terminal::Failed
                                                     : fastllm::prefix_write::Terminal::Committed);
            ++terminalCheckpoints;   // A2：本票已归还（成功/失败统一走 Terminal::*）
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            // S4-C：方案 C 下 pinnedGpuBytes 只按"窗口记账"，因此这里只能扣回
            // accountedPages（未扣完的窗口账）；整段 keepalive 的剩余引用由 ~Pages 释放，
            // exportProtectedBytes 按同样的"仍未释放（id>=0）"口径扣回。
            for (auto &lease : job.pages) {
                const int limit = std::min<int>(lease->pickedPages,(int)lease->ids.size());
                size_t held = 0;
                for (int i = 0; i < limit; ++i) if (lease->ids[i] >= 0) ++held;
                const size_t accounted = lease->accountedPages > 0 ? (size_t)lease->accountedPages : 0;
                pinnedGpuBytes[lease->rank] -= accounted * lease->pageBytes;
                const uint64_t heldBytes = (uint64_t)held * lease->pageBytes;
                exportProtectedBytes = exportProtectedBytes > heldBytes
                                           ? exportProtectedBytes - heldBytes : 0;
            }
            queuedStateBytes -= job.stateBytes; checkpointJobs -= job.snapshot != nullptr;
            if (!job.hintOnly) pendingCheckpointKeys.erase(job.key);
            if (failed) RetainFailedJobLocked(std::move(job), "worker_terminal");
            // A successful export transferred all bytes to the existing store.
            if (inflightJobs > 0) --inflightJobs;
            job = Job(); --activeWorkers;
        }
        changed.notify_all();
    }
    if (lane.d2SlotMemory) { FastllmCudaHostFree(lane.d2SlotMemory); lane.d2SlotMemory = nullptr; lane.d2SlotBytes = 0; }
    if (buffer) FastllmCudaHostFree(buffer);
}

// §④ 宽版真机 CopyIntoSlot：D2-stream 在"事件就绪且有空槽"时调用它。
// 契约（见 disk_prefix_write_coordinator.h）：执行 D2H 并返回待哈希的 payload；
// slot 只是编号，真正的宿主缓冲由本类持有（lane.d2SlotMemory 按 BufferBytes 切分）。
// R1：lane 由构造该流的 worker 通过捕获传入（每个 lane 一套 d2Chunks/d2CopyStreams/pinned 内存）。
std::string Qwen35PersistentCache::D2CopyIntoSlot(
        size_t laneIndex, int slot, const fastllm::prefix_write::TicketId &ticket, size_t bytes) {
    WriteLane &lane = *writeLanes[laneIndex];
    Check(lane.d2SlotMemory != nullptr, "ssd_d2_slot_unallocated");
    // A-5：单子上必须写清"哪一车"。
    // 旧实现只按 ticket.sequence 取块（:3328-3329），而上一笔失败作业的残留 ticket 可能仍停在
    // 本 lane 的 d2Stream 队首；新作业的 sequence 又从 0 起 ⇒ 旧单会顶到新块的头上，把新块的
    // 预约提前结算掉（真机 tokens=6144 报 duplicate_or_unadmitted_copy，server.log:470），
    // 且两笔分块字节数相同就发现不了（写错数据）。requestId 对不上一律拒绝。
    Check(ticket.requestId == lane.d2Requests, "ssd_d2_stale_ticket");
    const size_t index = size_t(ticket.sequence);
    Check(index < lane.d2Chunks.size(), "ssd_d2_ticket_unknown");
    D2Chunk &chunk = lane.d2Chunks[index];
    Check(bytes == chunk.bytes, "ssd_d2_bytes_mismatch");
    Check(slot >= 0 && size_t(slot) * BufferBytes + bytes <= lane.d2SlotBytes, "ssd_d2_slot_overflow");
    char *dst = (char *)lane.d2SlotMemory + size_t(slot) * BufferBytes;
    FastllmCudaSetDevice(chunk.rank);
    auto streamFound = lane.d2CopyStreams.find(chunk.rank);
    Check(streamFound != lane.d2CopyStreams.end(), "ssd_d2_copy_stream_missing");
    void *copyStream = streamFound->second;
    Check(chunk.pending_copy_counted && !chunk.copy_terminal, "duplicate_or_unadmitted_copy");
    // Every accepted D2H keeps its reservation and source owners through real
    // stream completion, including exception paths before hash handoff.
    struct CopyTerminalGuard {
        Qwen35PersistentCache *cache; D2Chunk *chunk; void *stream;
        bool synchronized = false;
        ~CopyTerminalGuard() {
            if (!synchronized) FastllmCudaStreamSynchronize(stream);
            std::lock_guard<std::mutex> lock(cache->mutex);
            if (chunk->pending_copy_counted && !chunk->copy_terminal && cache->pendingCopy > 0) {
                --cache->pendingCopy;
                chunk->pending_copy_counted = false;
                chunk->copy_terminal = true;
            } else {
                cache->shutdownFailed.store(true, std::memory_order_release);
                std::fprintf(stderr, "[Prefix SSD] copy-terminal-failed: reason=reservation_mismatch.\n");
            }
            cache->changed.notify_all();
        }
    } terminalGuard{this, &chunk, copyStream};
    // A-6 fix F: device timeline (measurement only). The begin event goes on the copy stream before
    // this job's first D2H on this device; the end event after this chunk's last D2H. Both are read
    // only after the copy-stream synchronize below, which already exists.
    std::pair<void *, void *> *timeline = nullptr;
    if (Qwen35GpuTimelineEnabled()) {
        auto &events = lane.timelineEvents[chunk.rank];
        if (events.first == nullptr) {
            Qwen35GpuTimelineAnchor(chunk.rank);
            events.first = FastllmCudaEventCreateTiming();
            events.second = FastllmCudaEventCreateTiming();
        }
        timeline = &events;
        if (lane.timelineBeginRecorded.insert(chunk.rank).second)
            FastllmCudaEventRecord(events.first, copyStream);
    }
    size_t offset = 0;
    Check(chunk.copy_generation != 0, "copy_generation_missing");
    for (auto &lease : chunk.leases) {
        const int firstPage = chunk.token / lease->pool->pageLen;
        const int completedPages = (chunk.token + chunk.count) / lease->pool->pageLen;
        Check(firstPage >= 0 && completedPages <= (int)lease->ids.size() &&
              completedPages <= (int)lease->generations.size(), "page_generation_missing");
        for (int page = firstPage; page < completedPages; ++page) {
            Check(lease->ids[page] >= 0, "page_generation_missing");
            Check(page < (int)lease->hashes.size(), "page_generation_missing");
            const uint64_t expectedHash = lease->hashes[page];
            // 发布端的 pageHash 只有在 onRecordPage/onRecyclePage 注册时才等于链哈希，而这两个回调
            // 由 FASTLLM_PREFIX_CACHE_L2_HOST_KV 决定（qwen3_5_spill.inc:894-895）。未启用时 node 里
            // 存的是另一个 h，拿它和链哈希比必然不等——不能据此判"过期"。未启用时只做计数（不可比），
            // 内容身份改由恢复端"按 token 重算并比对"来保证。
            if (expectedHash != 0 && PublisherPageHashIsChain()) {
                // S7：按内容身份判定。分块 prefill 会重新盖 pageTimestamp，但内容没变时
                // pageHash 不变，因此不再被误判为 stale。
                uint64_t actualHash = 0;
                bool published = false;
                {
                    std::lock_guard<std::mutex> guard(lease->pool->pageIndexLocker);
                    auto node = lease->pool->pageToTrieNode.find(lease->ids[page]);
                    if (node != lease->pool->pageToTrieNode.end() && node->second != nullptr) {
                        actualHash = node->second->pageHash;
                        published = true;
                    }
                }
                // A-6 (2026-10-31, user-approved option A): D2 waits only on lease->ready from
                // Capture, not on Record; an unpublished page has no node yet. Count it, do not
                // treat it as stale. A published node with a different hash is still stale.
                if (published) Check(actualHash == expectedHash, "stale_page_generation");
                else ++lane.d2UnhashedPages;
            } else {
                // A-5：Capture 早于 Record 发布（qwen3_5.cpp:11026-11028 经
                // basellm.cpp:867-872 调用 Capture，basellm.cpp:947/952 才 Record）⇒ 新页在捕获时
                // 没有 trie 节点，hashes=0；而 Record 发布时用新 ts 重盖 pageTimestamp
                // （fastllm.cpp:6734）⇒ 与捕获时的 generations 必然不等 ⇒ 旧判据把"自己的发布"
                // 误判成"过期"，真机三笔检查点因此全灭（logs/user-deploy-a5-20261031T0930Z，
                // 根因见 logs/p0-s04-a5-stale-audit-20261031T013057Z/STALE-ROOTCAUSE.md）。
                // 现改为：没有内容指纹的页不再比对会被重盖的时间戳；归属由"租约仍钉住该页
                // （lease 引用 + Pick）"与"生产者事件已完成（lease->ready 门）"共同保证。
                // 跳过量入账（lane.d2UnhashedPages），作业尾打印，供审计核对。
                // 未启用回调时（PublisherPageHashIsChain()==false）也走这里：号不可比，只计数。
                ++lane.d2UnhashedPages;
            }
        }
        const size_t rowBytes = lease->pageBytes / lease->pool->pageLen;
        for (int copied = 0; copied < chunk.count;) {
            const int page = (chunk.token + copied) / lease->pool->pageLen;
            const int within = (chunk.token + copied) % lease->pool->pageLen;
            const int part = std::min(chunk.count - copied, lease->pool->pageLen - within);
            Check(FastllmCudaCopyFromDeviceToPinnedHostAsync(
                      dst + offset + size_t(copied) * rowBytes,
                      (char *)lease->pool->cudaData +
                          (size_t)lease->ids[page] * lease->pageBytes + size_t(within) * rowBytes,
                      size_t(part) * rowBytes, copyStream),
                  "ssd_d2h");
            if (lane.saveStartNs == 0) lane.saveStartNs = FastllmObservationTimestampNs();
            copied += part;
        }
        offset += size_t(chunk.count) * rowBytes;
    }
    if (timeline != nullptr) FastllmCudaEventRecord(timeline->second, copyStream);
    // 拷贝全部完成后才能归还 GPU 页（与旧同步路径同一语义）。
    FastllmCudaStreamSynchronize(copyStream);
    terminalGuard.synchronized = true;
    if (timeline != nullptr) {
        if (lane.timelineBeginMs.count(chunk.rank) == 0)
            lane.timelineBeginMs[chunk.rank] = Qwen35GpuTimelineMs(chunk.rank, timeline->first);
        lane.timelineEndMs[chunk.rank] = Qwen35GpuTimelineMs(chunk.rank, timeline->second);
    }
    size_t released = 0;
    for (auto &lease : chunk.leases) {
        const int firstPage = chunk.token / lease->pool->pageLen;
        const int completedPages = (chunk.token + chunk.count) / lease->pool->pageLen;
        std::vector<int> copiedIds(lease->ids.begin() + firstPage,
                                   lease->ids.begin() + completedPages);
        // 方案 C：每个 D2H 分块完成即**立即**释放该分块已完成的页（保护时长 ≈ 导出时长），
        // 不等整个 job 结束。
        lease->pool->ReleasePageIndices(copiedIds);
        std::fill(lease->ids.begin() + firstPage, lease->ids.begin() + completedPages, -1);
        {
            // R1：pinnedGpuBytes 现在是多 lane 共享账本，所有读写都在 mutex 下（这一段很短）。
            std::lock_guard<std::mutex> lock(mutex);
            pinnedGpuBytes[lease->rank] -= copiedIds.size() * lease->pageBytes;
            const uint64_t releasedBytes = (uint64_t)copiedIds.size() * lease->pageBytes;
            exportProtectedBytes = exportProtectedBytes > releasedBytes
                                       ? exportProtectedBytes - releasedBytes : 0;
        }
        // 预算记账按窗口：扣回本分块已完成页数（accountedPages 只在 worker 线程读写）。
        lease->accountedPages = std::max(0,lease->accountedPages - (int)copiedIds.size());
        released += copiedIds.size();
    }
    // A2：page lease release 次数（D2H 完成后归还的 GPU 页数）。原子累加，多 lane 安全。
    pageLeaseReleases.fetch_add(released, std::memory_order_relaxed);
    changed.notify_all();
    (void)released;
    return std::string(dst, bytes);
}

void Qwen35PersistentCache::Prepare(ResponseContext *ctx) {
    using Status = PrefixCacheRestoreStatus;
    if (!ctx || ctx->cacheLen || !ctx->CanUsePagedPrefixCache() ||
        !Qwen35LinearPrefixCacheEnabled()) return;
    RequestDiagnosticOperation diagnosticPrepare(ctx->requestId, "prefix_cache_prepare",
        "Qwen35PersistentCache.Prepare");
    auto &status = ctx->prefixRestore;
    if (status.phase == Status::Miss || status.phase == Status::Failed ||
        status.phase == Status::TimedOut) return;
    // Captured at model construction: default 100 seconds, or the explicit
    // deployment value supplied through FASTLLM_PREFIX_CACHE_RESTORE_TIMEOUT_MS.
    const int timeoutMs = restoreTimeoutMs;
    if (status.phase == Status::Reading) {
        std::shared_ptr<ReadTicket> ticket;
        {
            std::lock_guard<std::mutex> lock(prefetchMutex);
            const auto found = restoreTickets.find(ctx);
            if (found != restoreTickets.end()) ticket = found->second;
        }
        if (!ticket) { status.phase = Status::Failed; status.reason = "read_ticket_missing"; return; }
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double,std::milli>(now-ticket->started).count();
        auto consume = [&]() {
            ticket->consumed.store(true, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lock(prefetchMutex);
                const auto found = restoreTickets.find(ctx);
                if (found != restoreTickets.end() && found->second == ticket)
                    restoreTickets.erase(found);
            }
            prefetchChanged.notify_all();
        };
        if (!ticket->request_state || ticket->request_id != ctx->requestId ||
            ticket->request_state->cancelled.load(std::memory_order_acquire) ||
            ticket->request_state->generation.load(std::memory_order_acquire) !=
                ticket->request_generation) {
            status.phase = Status::Failed;
            status.reason = "stale_request_generation";
            status.loadingState = PrefixCacheRecord::Failed;
            PublishPrefixRecord(ctx);
            consume();
            return;
        }
        if (elapsed >= timeoutMs) {
            ticket->cancelled.store(true);
            status.phase = Status::TimedOut; status.reason = "ssd_restore_timeout";
            status.runtimeTier = PrefixCacheRecord::TierNone;
            status.runtimeState = PrefixCacheRecord::RuntimeUnknown;
            status.durableState = PrefixCacheRecord::DurableReady;
            status.loadingState = PrefixCacheRecord::Failed;
            PublishPrefixRecord(ctx);
            std::fprintf(stderr,"[PrefixCache] restore timed out: request=%llu tier=L3(SSD) reason=%s wait_ms=%.1f fallback=compute.\n",
                        (unsigned long long)ticket->id,status.reason.c_str(),elapsed);
            std::fflush(stderr); consume(); return;
        }
        if (!ticket->done.load(std::memory_order_acquire)) {
            if (std::chrono::duration<double,std::milli>(now-ticket->lastProgress).count() >= 1000) {
                ticket->lastProgress = now;
                const uint64_t verified = ticket->verifiedBytes.load(std::memory_order_relaxed);
                const uint64_t discovered = ticket->discoveredBytes.load(std::memory_order_relaxed);
                PrefixCacheDebug("[Prefix SSD] request-restore-progress: request=%llu tier=L3(SSD) phase=read "
                            "bytes=%llu/%llu total_state=manifest_in_progress elapsed_ms=%.1f speed_mib_s=%.1f.\n",
                            (unsigned long long)ticket->id,(unsigned long long)verified,
                            (unsigned long long)discovered,elapsed,
                            elapsed > 0 ? verified/1048576.0*1000/elapsed : 0);
            }
            return;
        }
        status.readMs = ticket->readMs; status.loadedBytes = ticket->loadedBytes;
        // A-6 fix E：恢复失败（不含超时/取消）后，本进程内不再选这份检查点；
        // 只改内存账本，SSD 字节保留（logs/a6-g8fix-design-20261031T230156/G8FIX-DESIGN.md §4.1）。
        // 物理损坏的精确退役不在本改动内（G8RETIRE）。
        const auto blockFailedCandidate = [&](const std::string &why) {
            if (!model->pagedPageLedger || ticket->cancelled.load(std::memory_order_acquire) ||
                Clock::now() >= ticket->deadline) return;   // 取消/超时不是坏档证据：保留，不屏蔽
            const auto &key = ticket->candidate.key;
            const auto &checkpointIdentity = ticket->candidate.identity;
            PrefixCachePageLedger::CheckpointSnapshot tracked;
            const bool wasTracked = model->pagedPageLedger->Checkpoint(key, tracked);
            const bool blocked = wasTracked ?
                model->pagedPageLedger->FailRestoreCheckpoint(key, checkpointIdentity, why) :
                model->pagedPageLedger->RegisterFailedRestoreCheckpoint(key, checkpointIdentity, why);
            FastllmEvent("cache", ctx->requestId, "cache_restore_candidate_blocked",
                         "checkpoint=%s tokens=%d tracked=%d blocked=%d reason=%s",
                         key.c_str(), ticket->candidate.length, wasTracked ? 1 : 0, blocked ? 1 : 0,
                         why.c_str());
        };
        if (!ticket->ok) {
            status.phase = Status::Failed; status.reason = ticket->error;
            status.runtimeTier = PrefixCacheRecord::TierNone;
            status.runtimeState = PrefixCacheRecord::RuntimeUnknown;
            status.durableState = PrefixCacheRecord::DurableReady;
            status.loadingState = PrefixCacheRecord::Failed;
            PublishPrefixRecord(ctx);
            std::fprintf(stderr,"[PrefixCache] CACHE-RESTORE-FAILED: request=%llu tier=L3(SSD) reason=%s fallback=compute.\n",
                         (unsigned long long)ticket->id,status.reason.c_str());
            ticket->reader.reset();
            bool retired = false;
            if (store && DiskPrefixCache::ClassifyRestoreFailure(status.reason) ==
                    DiskPrefixCache::RestoreFailureClass::STORED_REPRESENTATION) {
                std::string retireError;
                retired = store->RetireCorruptCheckpoint(ticket->candidate.key,
                    ticket->candidate.identity, status.reason, retireError);
                FastllmEvent("cache", ctx->requestId, "cache_corrupt_checkpoint_retire",
                    "checkpoint=%s retired=%d error=%s", ticket->candidate.key.c_str(),
                    retired ? 1 : 0, retireError.c_str());
            }
            if (!retired) blockFailedCandidate(status.reason);
            consume(); return;
        }
        // Disk I/O is already finished. GPU publication uses the established
        // Import parser and the existing scheduler/forward lock order.
        RequestDiagnosticOperation diagnosticForwardTry(ctx->requestId, "ssd_restore_forward_lock_try",
            RequestDiagnosticResource("model.forwardLocker", &model->forwardLocker).c_str());
        std::unique_lock<std::mutex> forward(model->forwardLocker,std::try_to_lock);
        diagnosticForwardTry.Finish();
        if (!forward.owns_lock()) return;
        RequestDiagnosticOperation diagnosticRestore(ctx->requestId, "ssd_restore_import",
            RequestDiagnosticResource("model.forwardLocker", &model->forwardLocker).c_str());
        const auto gpuBegin = Clock::now();
        auto lastGpuProgress = gpuBegin;
        const auto keepGoing = [ticket]() {
            return !ticket->cancelled.load(std::memory_order_acquire) && Clock::now() < ticket->deadline;
        };
        const auto restoreProgress = [&]() {
            const auto now = Clock::now();
            if (std::chrono::duration<double,std::milli>(now-lastGpuProgress).count() >= 1000) {
                lastGpuProgress = now;
                PrefixCacheDebug("[Prefix SSD] request-restore-progress: request=%llu phase=gpu_restore "
                            "read_bytes=%llu total_read_bytes=%llu read_complete=1 "
                            "gpu_restore_elapsed_ms=%.1f adopted=0.\n",
                            (unsigned long long)ticket->id,(unsigned long long)ticket->loadedBytes,
                            (unsigned long long)ticket->loadedBytes,
                            std::chrono::duration<double,std::milli>(now-gpuBegin).count());
            }
            return keepGoing();
        };
        // A-3 P1/T3：候选筛选到真正导入之间可能相隔数百毫秒到数秒，
        // 期间账本状态可能被其他路径改写（独立交叉核验旁路 B2：TOCTOU）。导入前复查一次。
        bool ledgerStillRestorable = true;
        if (model->pagedPageLedger) {
            PrefixCachePageLedger::CheckpointSnapshot importLedger;
            ledgerStillRestorable =
                !model->pagedPageLedger->Checkpoint(ticket->candidate.key, importLedger) ||
                model->pagedPageLedger->CanRestoreCheckpoint(ticket->candidate.key,
                                                              ticket->candidate.identity);
        }
        if (!ledgerStillRestorable)
            FastllmEvent("cache", ctx->requestId, "cache_restore_recheck_failed",
                         "checkpoint=%s tokens=%d reason=checkpoint_invalidated_before_import",
                         ticket->candidate.key.c_str(), ticket->candidate.length);
        LastImportFailureReason().clear();
        const bool imported = ledgerStillRestorable &&
            Import(ctx,ticket->candidate,ticket->reader.get(),&ticket->state,restoreProgress);
        status.restoreMs = std::chrono::duration<double,std::milli>(Clock::now()-gpuBegin).count();
        status.totalMs = std::chrono::duration<double,std::milli>(Clock::now()-ticket->started).count();
        FastllmEvent("cache", ctx->requestId, "cache_restore_profile",
                     "measurement=native_ssd_restore tokens=%d "
                     "read_ms=%.3f import_ms=%.3f total_ms=%.3f "
                     "verified_payload_bytes=%llu imported=%d",
                     ticket->candidate.length, status.readMs, status.restoreMs, status.totalMs,
                     (unsigned long long)status.loadedBytes, imported ? 1 : 0);
        status.phase = imported ? Status::Ready :
            Clock::now() >= ticket->deadline ? Status::TimedOut : Status::Failed;
        // A-6 fix F part 2: keep the real Import failure (e.g. ssd_import_identity_mismatch) instead
        // of the generic label, so the block event says why (VERDICT-G7G8.md).
        if (!imported) status.reason = status.phase == Status::TimedOut ? std::string("ssd_restore_timeout") :
            !LastImportFailureReason().empty() ? LastImportFailureReason() : std::string("gpu_restore_failed");
        if (imported) {
            if (RequestHoldEnabled() && ticket->reader) {
                std::shared_ptr<DiskPrefixCache::ReadSession> hold(std::move(ticket->reader));
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    requestHolds[ctx].swap(hold);
                }
                hold.reset();   // a previous hold of this ctx, released outside `mutex`
            }
            DescribePrefix(ctx,ticket->candidate.length);
            status.ssdComponents = status.requiredComponents;
            status.gpuComponents = status.requiredComponents;
            status.runtimeTier = PrefixCacheRecord::TierL1;
            status.runtimeState = PrefixCacheRecord::RuntimeResident;
            status.runtimeLease = 0;
            status.durableState = PrefixCacheRecord::DurableReady;
            status.loadingState = PrefixCacheRecord::Ready;
            PublishPrefixRecord(ctx);
        } else {
            status.runtimeTier = PrefixCacheRecord::TierNone;
            status.runtimeState = PrefixCacheRecord::RuntimeUnknown;
            status.durableState = PrefixCacheRecord::DurableReady;
            status.loadingState = PrefixCacheRecord::Failed;
            PublishPrefixRecord(ctx);
            if (status.phase == Status::Failed && ledgerStillRestorable) {
                ticket->reader.reset();
                bool retired = false;
                if (store && DiskPrefixCache::ClassifyRestoreFailure(status.reason) ==
                        DiskPrefixCache::RestoreFailureClass::STORED_REPRESENTATION) {
                    std::string retireError;
                    retired = store->RetireCorruptCheckpoint(ticket->candidate.key,
                        ticket->candidate.identity, status.reason, retireError);
                    FastllmEvent("cache", ctx->requestId, "cache_corrupt_checkpoint_retire",
                        "checkpoint=%s retired=%d error=%s", ticket->candidate.key.c_str(),
                        retired ? 1 : 0, retireError.c_str());
                }
                if (!retired) blockFailedCandidate(status.reason);
            }
        }
        PrefixCacheDebug("[PrefixCache] restore-ready: request=%llu tier=L3(SSD) candidate_tokens=%d "
                    "verified_payload_bytes=%llu read_ms=%.1f payload_mib_s=%.1f restore_ms=%.1f "
                    "total_ms=%.1f adopted=0 result=%s.\n",
                    (unsigned long long)ticket->id,status.candidateTokens,
                    (unsigned long long)status.loadedBytes,status.readMs,
                    status.readMs > 0 ? status.loadedBytes/1048576.0*1000/status.readMs : 0,
                    status.restoreMs,status.totalMs, imported ? "ready" : "failed");
        consume(); return;
    }
    // Foreground L2 promotion may have dispatched deferred GPU victims and
    // retained host-owner holds. Retry that exact prefix before the lookup
    // gate; re-querying local tiers would treat the held host/GPU transaction
    // as a duplicate and lose the pending candidate.
    const int promotionState = ctx->intParams["qwen35_l2_promotion_pending"];
    if (promotionState == 2) {
        // No safe GPU capacity became available within the bounded
        // preflight. Clear the retry gate and let ordinary prefill continue.
        ctx->intParams["qwen35_l2_promotion_pending"] = 0;
        ctx->intParams["qwen35_l2_promotion_tokens"] = 0;
        model->ReleasePagedSpillRetryHolds(ctx);
        status.runtimeTier = PrefixCacheRecord::TierNone;
        status.runtimeState = PrefixCacheRecord::RuntimeUnknown;
        status.hostComponents = 0;
        status.phase = Status::Failed;
        status.reason = "capacity_unavailable";
        status.loadingState = PrefixCacheRecord::Failed;
        PublishPrefixRecord(ctx);
        return;
    }
    if (promotionState == 1) {
        const int pendingTokens = ctx->intParams["qwen35_l2_promotion_tokens"];
        if (pendingTokens > 0 && model->PromotePagedSpillPrefix(ctx, pendingTokens)) {
            ctx->intParams["qwen35_l2_promotion_pending"] = 0;
            ctx->intParams["qwen35_l2_promotion_tokens"] = 0;
            status.runtimeTier = PrefixCacheRecord::TierL1;
            status.runtimeState = PrefixCacheRecord::RuntimeResident;
            status.runtimeLease = 0;
            status.gpuComponents = status.requiredComponents;
            status.hostComponents = 0;
            status.phase = Status::Ready;
            status.loadingState = PrefixCacheRecord::Ready;
            PublishPrefixRecord(ctx);
        } else if (ctx->intParams["qwen35_l2_promotion_pending"] != 1) {
            // No ticket was dispatched and no progress is possible. A retry
            // with pending tickets remains state=1 and is revisited on the
            // next scheduler iteration without blocking this one.
            ctx->intParams["qwen35_l2_promotion_pending"] = 2;
            ctx->intParams["qwen35_l2_promotion_reason"] = 1;
        }
        return;
    }
    if (ctx->intParams["ssd_lookup_done"]) return;
    ctx->intParams["ssd_lookup_done"] = 1;
    RequestDiagnosticOperation diagnosticLookup(ctx->requestId, "prefix_cache_initial_lookup",
        "Qwen35PersistentCache.Prepare.initial_lookup");
    const auto lookupBegin = Clock::now();
    FastllmEvent("cache", ctx->requestId, "cache_lookup",
                 "phase=begin current_tokens=%zu", ctx->currentTokens.size());
    const int pageLen = fastllm::GetPageLen();
    const int maxCandidate = ((int)ctx->currentTokens.size() - 1) / pageLen * pageLen;
    const int l1Length = Qwen35ReadyPagedPrefix(model,ctx);
    const int l2Length = Qwen35L2HostKVEnabled() ?
        model->QueryPagedSpillPrefix(ctx,maxCandidate) : 0;
    const int localLength = std::max(l1Length,l2Length);
    int l1ResidentTokens = ctx->intParams["qwen35_l1_resident_tokens"];
    int l2HostTokens = ctx->intParams["qwen35_l2_host_tokens"];
    int combinedLocalTokens = ctx->intParams["qwen35_combined_local_tokens"];
    const int l2QueryStatus = ctx->intParams["qwen35_l2_query_status"];
    if (l1ResidentTokens <= 0 && l2HostTokens <= 0 && l1Length > 0)
        l1ResidentTokens = l1Length;
    if (combinedLocalTokens <= 0) combinedLocalTokens = localLength;
    const char *l2QueryStatusText =
        l2QueryStatus == 1 ? "ok" : l2QueryStatus == 2 ? "busy" :
        l2QueryStatus == 3 ? "disabled" : l2QueryStatus == 4 ? "duplicate" :
        l2QueryStatus == 5 ? "incomplete" : "unknown";
    FastllmEvent("cache", ctx->requestId, "cache_candidates",
                 "l1_resident_tokens=%d l2_host_tokens=%d combined_local_tokens=%d "
                 "l2_query_status=%s duplicate_pages=%d first_gap_page=%d "
                 "max_candidate_tokens=%d",
                 l1ResidentTokens, l2HostTokens, combinedLocalTokens,
                 l2QueryStatusText, ctx->intParams["qwen35_l2_duplicate_pages"],
                 ctx->intParams["qwen35_l2_first_gap_page"], maxCandidate);
    int l2SegmentIndex = -1;
    auto recordLocalSegments = [&]() {
        ctx->progressL1Tokens = std::max(ctx->progressL1Tokens, l1ResidentTokens);
        ctx->progressL2Tokens = std::max(ctx->progressL2Tokens, l2HostTokens);
        ctx->cacheSegments.clear();
        if (l1ResidentTokens > 0) {
            PrefixCacheObservationSegment segment;
            segment.sourceTier = "L1";
            segment.promotedTo = "NONE";
            segment.sourceTokens = l1ResidentTokens;
            segment.effectiveReusedTokens = l1ResidentTokens;
            segment.lookupMs = status.lookupMs;
            ctx->cacheSegments.push_back(segment);
        }
        if (l2HostTokens > 0) {
            PrefixCacheObservationSegment segment;
            segment.sourceTier = "L2";
            segment.promotedTo = "L1";
            segment.sourceTokens = l2HostTokens;
            segment.effectiveReusedTokens = l2HostTokens;
            segment.lookupMs = status.lookupMs;
            ctx->cacheSegments.push_back(segment);
            l2SegmentIndex = (int)ctx->cacheSegments.size() - 1;
        }
    };
    auto emitLocalCacheEvents = [&]() {
        if (l1ResidentTokens > 0) {
            FastllmEvent("cache", ctx->requestId, "cache_result",
                         "source_tier=L1 promoted_to=NONE source_tokens=%d "
                         "effective_reused_tokens=%d lookup_ms=%.3f",
                         l1ResidentTokens, l1ResidentTokens, status.lookupMs);
        }
        if (l2HostTokens > 0 && l2SegmentIndex >= 0 &&
            ctx->cacheSegments[l2SegmentIndex].restoreMs > 0) {
            FastllmEvent("cache", ctx->requestId, "cache_result",
                         "source_tier=L2 promoted_to=L1 source_tokens=%d "
                         "effective_reused_tokens=%d lookup_ms=%.3f restore_ms=%.3f",
                         l2HostTokens, l2HostTokens, status.lookupMs,
                         ctx->cacheSegments[l2SegmentIndex].restoreMs);
        }
    };
    PrefixCacheDebug("[PrefixCache] candidates: l1=%d l2=%d max=%d.\n",
                     l1Length,l2Length,maxCandidate);
    status.lookupMs = std::chrono::duration<double,std::milli>(Clock::now()-lookupBegin).count();
    if (policy == "never" || !Ensure()) {
        if (localLength > 0) {
            const bool useL2 = l2HostTokens > 0;
            status.tier = useL2 ? Status::Host : Status::GPU;
            status.phase = Status::Ready;
            status.candidateTokens = localLength;
            DescribePrefix(ctx,localLength);
            if (useL2) {
                status.hostComponents = status.requiredComponents;
                status.runtimeTier = PrefixCacheRecord::TierL2;
                status.runtimeState = PrefixCacheRecord::RuntimePromotePending;
            } else {
                status.gpuComponents = status.requiredComponents;
                status.runtimeTier = PrefixCacheRecord::TierL1;
                status.runtimeState = PrefixCacheRecord::RuntimeResident;
            }
            status.loadingState = PrefixCacheRecord::Ready;
            PublishPrefixRecord(ctx);
            recordLocalSegments();
            if (useL2) {
                const auto promoteBegin = Clock::now();
                if (!model->PromotePagedSpillPrefix(ctx,localLength)) {
                    if (ctx->intParams["qwen35_l2_promotion_pending"]) {
                        status.phase = Status::Ready;
                        status.reason = "l2_promotion_pending";
                        status.loadingState = PrefixCacheRecord::Ready;
                        PublishPrefixRecord(ctx);
                        return;
                    }
                    status.phase = Status::Failed;
                    status.reason = "l2_promotion_failed";
                    status.loadingState = PrefixCacheRecord::Failed;
                    PublishPrefixRecord(ctx);
                } else {
                    const double restoreMs = std::chrono::duration<double,std::milli>(
                        Clock::now()-promoteBegin).count();
                    ctx->cacheSegments[l2SegmentIndex].restoreMs = restoreMs;
                    status.runtimeTier = PrefixCacheRecord::TierL1;
                    status.runtimeState = PrefixCacheRecord::RuntimeResident;
                    status.runtimeLease = 0;
                    status.gpuComponents = status.requiredComponents;
                    status.hostComponents = 0;
                    PublishPrefixRecord(ctx);
                }
            }
            emitLocalCacheEvents();
            return;
        }
        status.phase = Status::Miss; status.reason = "ssd_retrieval_disabled";
        status.loadingState = PrefixCacheRecord::Failed;
        PublishPrefixRecord(ctx); return;
    }
    StartPrefetch();
    try {
        if (GetFastllmObservabilityMode() != FastllmObservabilityMode::Off) {
            FastllmEvent("cache", ctx->requestId, "cache_lookup_identity",
                         "family=%s identity=%s encoder_identity=%s input_tokens=%d media_spans=%zu "
                         "page_len=%d provenance=%s",
                         family.c_str(), identity.c_str(), encoderIdentity.c_str(), ctx->inputTokens,
                         ctx->multimodalPrefixSpans.size(), fastllm::GetPageLen(),
                         ctx->prefixCacheProvenance.empty() ? "none" : ctx->prefixCacheProvenance.c_str());
            for (int length : {128, 512, 640, 768, 1024, 2048}) {
                if (length > (int)ctx->allTokens.size()) continue;
                const auto digest = DiskPrefixCache::PrefixKey(ctx->allTokens, length, "");
                FastllmEvent("cache", ctx->requestId, "cache_token_fingerprint",
                             "tokens=%d token_sha256=%s", length, digest.c_str());
            }
            for (size_t i = 0; i < ctx->multimodalPrefixSpans.size() && i < 8; ++i) {
                const auto &span = ctx->multimodalPrefixSpans[i];
                std::string raw(reinterpret_cast<const char *>(span.digest.data()), span.digest.size());
                const auto digest = EncodeKey(raw);
                FastllmEvent("cache", ctx->requestId, "cache_media_identity",
                             "media_index=%zu begin_token=%d end_token=%d preprocessed_image_sha256=%s",
                             i, span.begin, span.end, digest.c_str());
            }
        }
        // Only committed WAL rows are queried: a miss never waits for an
        // unrelated background write or reads its payload.
        RequestDiagnosticOperation diagnosticCatalog(ctx->requestId, "ssd_catalog_lookup",
            "DiskPrefixCache.CheckpointLengths_and_MatchPrefixes");
        const auto lengths = store->CheckpointLengths(family,(int)ctx->allTokens.size()-1,"prefix",true);
        std::vector<std::pair<int,std::string>> wanted;
        for (int candidateLength : lengths)
            if (candidateLength > 0) wanted.push_back({candidateLength,Prefix(ctx,candidateLength)});
        auto candidates = store->MatchPrefixes(family,wanted,"prefix",true);
        // V5 junction (draft3 sec. 4 step 4, H24/H25): the deepest page shared with ANY
        // stored conversation. Prefill ends a chunk there so a state is saved at the
        // branch point (text: Qwen35JunctionClamp; images: the multimodal chunk loop).
        // Page identities use the export's key decision (PrefixCachePageKeys(): media keys for
        // image prompts, none for text). The old condition prefixCachePageKeys.empty() was never
        // true (Created() fills the text keys), so the junction never fired (0 events in v5e/stab6).
        if (fastllm::DiskPrefixV5Enabled()) {
            const int pageLen = fastllm::GetPageLen();
            const std::vector<std::string> *junctionKeys = ctx->PrefixCachePageKeys();
            int limit = ((int)ctx->allTokens.size()-1)/pageLen*pageLen;
            if (junctionKeys) limit = std::min(limit, (int)junctionKeys->size()*pageLen);
            int junction = limit > 0 ?
                store->JunctionLength(ComputePageIdentities(ctx->allTokens,limit,pageLen,junctionKeys),pageLen) : 0;
            // Never cut inside an image: roll the junction back to the page boundary before the
            // media item (vLLM 83c078d955 v1/core/sched/scheduler.py:2016-2034, disable_chunked_mm_input
            // "roll back to before the mm item"). Applied to the junction cut only; the regular chunk
            // grid is unchanged. fin1/fin3: mimg junction 1408 inside media 0 (4..1534) -> restore != cold;
            // mi1-v5 (no junction) -> restore == cold.
            for (bool moved = true; moved && junction > 0;) {
                moved = false;
                for (const auto &span : ctx->multimodalPrefixSpans)
                    if (span.begin < junction && junction < span.end) {
                        junction = span.begin / pageLen * pageLen; moved = true;
                    }
            }
            ctx->intParams["qwen35_v5_junction"] = junction;
            if (junction > 0)
                FastllmEvent("cache", ctx->requestId, "cache_v5_junction", "junction_tokens=%d", junction);
        }
        diagnosticCatalog.Finish();
        status.lookupMs = std::chrono::duration<double,std::milli>(Clock::now()-lookupBegin).count();
        FastllmEvent("cache", ctx->requestId, "cache_lookup_catalog",
                     "published_length_count=%zu matching_candidate_count=%zu "
                     "lookup_ms=%.3f no_candidate_reason=%s",
                     lengths.size(), candidates.size(), status.lookupMs,
                     !lengths.empty() ? "no_matching_prefix" : "no_live_checkpoint_for_family");
        for (size_t i = 0; i < wanted.size() && i < 32; ++i)
            FastllmEvent("cache", ctx->requestId, "cache_checkpoint_query",
                         "tokens=%d prefix_sha256=%s", wanted[i].first, wanted[i].second.c_str());
        std::stable_sort(candidates.begin(),candidates.end(),
                         [](const auto &a,const auto &b) { return a.length > b.length; });
        size_t rejectedCandidates = 0;
        for (const auto &candidate : candidates) {
            const char *rejection = nullptr;
            if (candidate.identity != identity) rejection = "model_identity";
            else if (candidate.kvDtype != Dtype(model->kvCacheDataType)) rejection = "kv_dtype";
            else if (candidate.length <= 0 || candidate.length % fastllm::GetPageLen()) rejection = "page_alignment";
            else if (!DiskPrefixCache::IsDigest(candidate.metadata["state"].string_value())) rejection = "state_descriptor";
            else if (!candidate.metadata["layout"].is_null() &&
                     candidate.metadata["layout"].dump() != RuntimeLayout().dump()) rejection = "runtime_layout";
            else if (candidate.length <= localLength) rejection = "not_longer_than_local";
            else if (candidate.key != DiskPrefixCache::Digest(Prefix(ctx,candidate.length)+candidate.provenance))
                rejection = "prefix_provenance_key";
            if (rejection) {
                if (rejectedCandidates++ < 32)
                    FastllmEvent("cache", ctx->requestId, "cache_candidate_rejected",
                                 "checkpoint=%s tokens=%d reason=%s", candidate.key.c_str(),
                                 candidate.length, rejection);
                continue;
            }
            if (model->pagedPageLedger) {
                // A-3 P1/T1 ：账本只在同一进程内有效（无持久化），
                // 重启后为空是预期状态；v4 catalog 只返回 live 根，完整性由存储层保证。
                // 因此这里不因"查不到"而拒绝，但必须把这条旁路显式记录、可计数，
                // 而不是像此前那样静默 && 短路（CROSSCHECK.md 旁路 1：冷启动短路）。
                PrefixCachePageLedger::CheckpointSnapshot checkpoint;
                const bool ledgerTracked =
                    model->pagedPageLedger->Checkpoint(candidate.key, checkpoint);
                if (ledgerTracked &&
                    !model->pagedPageLedger->CanRestoreCheckpoint(candidate.key, candidate.identity)) {
                    if (rejectedCandidates++ < 32)
                        FastllmEvent("cache", ctx->requestId, "cache_candidate_rejected",
                                     "checkpoint=%s tokens=%d reason=failed_or_quarantined",
                                     candidate.key.c_str(), candidate.length);
                    continue;
                }
                if (!ledgerTracked)
                    FastllmEvent("cache", ctx->requestId, "cache_restore_ledger_untracked",
                                 "checkpoint=%s tokens=%d reason=cross_process_ledger_empty",
                                 candidate.key.c_str(), candidate.length);
            }
            auto ticket = std::make_shared<ReadTicket>();
            ticket->diagnosticToken = CaptureRequestDiagnosticToken(ctx->requestId);
            {
                std::lock_guard<std::mutex> identityLock(mutex);
                auto lifecycle = requestStates.find(ctx);
                if (lifecycle == requestStates.end() || !lifecycle->second ||
                    lifecycle->second->request_id != ctx->requestId ||
                    lifecycle->second->cancelled.load(std::memory_order_acquire)) {
                    status.phase = Status::Failed;
                    status.reason = "request_identity_missing_or_cancelled";
                    status.loadingState = PrefixCacheRecord::Failed;
                    return;
                }
                ticket->request_state = lifecycle->second;
            }
            ticket->request_id = ctx->requestId;
            ticket->request_generation = ticket->request_state->generation.load(std::memory_order_acquire);
            ticket->page_generation = std::max<uint64_t>(1, candidate.createdNs);
            ticket->copy_generation = copyGenerationCounter.fetch_add(1, std::memory_order_acq_rel) + 1;
            ticket->cache_identity = DiskPrefixCache::Digest(candidate.identity + "|" + candidate.key);
            ticket->owner_lease = std::shared_ptr<const void>(ticket->request_state, ticket->request_state.get());
            ticket->candidate = candidate;
            ticket->deadline = ticket->started + std::chrono::milliseconds(timeoutMs);
            {
                std::lock_guard<std::mutex> lock(prefetchMutex);
                if (prefetchStopping) break;
                if (requestReads.size() >= kMaxInflightJobs) {
                    std::fprintf(stderr, "[Prefix SSD] restore-rejected: status=REJECTED_BEFORE_ADMISSION "
                                 "error=prefetch_queue_full max_prefetch_inflight=%zu.\n",
                                 kMaxInflightJobs);
                    break;
                }
                ticket->id = ++nextReadId;
                restoreTickets[ctx] = ticket;
                requestReads.push_back(ticket);
                ++prefetchQueued;
            }
            status.tier = Status::SSD; status.phase = Status::Reading;
            status.requestId = ticket->id; status.key = candidate.key;
            status.candidateTokens = candidate.length;
            DescribePrefix(ctx,candidate.length);
            status.key = candidate.key; status.identity = candidate.identity;
            status.provenance = candidate.provenance;
            status.runtimeTier = PrefixCacheRecord::TierMigrating;
            status.runtimeState = PrefixCacheRecord::RuntimePromotePending;
            status.durableState = PrefixCacheRecord::DurableReady;
            status.loadingState = PrefixCacheRecord::Loading;
            PublishPrefixRecord(ctx);
            PrefixCacheDebug("[PrefixCache] lookup: request=%llu tier=L3(SSD) state=HIT candidate_tokens=%d components=%u.\n",
                        (unsigned long long)ticket->id,candidate.length,status.requiredComponents);
            PrefixCacheDebug("[Prefix SSD] request-restore-begin: request=%llu key=%s tier=L3(SSD) candidate_tokens=%d "
                        "timeout_ms=%d route=SSD_to_GPU.\n",
                        (unsigned long long)ticket->id,candidate.key.c_str(),candidate.length,timeoutMs);
            FastllmEvent("cache", ctx->requestId, "cache_restore_begin",
                         "tier=L3(SSD) phase=read candidate_tokens=%d ticket=%llu",
                         candidate.length, (unsigned long long)ticket->id);
            prefetchChanged.notify_one(); return;
        }
        if (localLength > 0) {
            const bool useL2 = l2HostTokens > 0;
            status.tier = useL2 ? Status::Host : Status::GPU;
            status.phase = Status::Ready;
            status.candidateTokens = localLength;
            DescribePrefix(ctx,localLength);
            if (useL2) {
                status.hostComponents = status.requiredComponents;
                status.runtimeTier = PrefixCacheRecord::TierL2;
                status.runtimeState = PrefixCacheRecord::RuntimePromotePending;
            } else {
                status.gpuComponents = status.requiredComponents;
                status.runtimeTier = PrefixCacheRecord::TierL1;
                status.runtimeState = PrefixCacheRecord::RuntimeResident;
            }
            status.loadingState = PrefixCacheRecord::Ready;
            PublishPrefixRecord(ctx);
            recordLocalSegments();
            if (useL2) {
                const auto promoteBegin = Clock::now();
                if (!model->PromotePagedSpillPrefix(ctx,localLength)) {
                    if (ctx->intParams["qwen35_l2_promotion_pending"]) {
                        status.phase = Status::Ready;
                        status.reason = "l2_promotion_pending";
                        status.loadingState = PrefixCacheRecord::Ready;
                        PublishPrefixRecord(ctx);
                        return;
                    }
                    status.phase = Status::Failed;
                    status.reason = "l2_promotion_failed";
                    status.loadingState = PrefixCacheRecord::Failed;
                    PublishPrefixRecord(ctx);
                } else {
                    const double restoreMs = std::chrono::duration<double,std::milli>(
                        Clock::now()-promoteBegin).count();
                    ctx->cacheSegments[l2SegmentIndex].restoreMs = restoreMs;
                    status.runtimeTier = PrefixCacheRecord::TierL1;
                    status.runtimeState = PrefixCacheRecord::RuntimeResident;
                    status.runtimeLease = 0;
                    status.gpuComponents = status.requiredComponents;
                    status.hostComponents = 0;
                    PublishPrefixRecord(ctx);
                }
            }
            emitLocalCacheEvents();
            return;
        }
        status.phase = Status::Miss; status.reason = "no_published_checkpoint";
        status.loadingState = PrefixCacheRecord::Failed;
        PublishPrefixRecord(ctx);
    } catch (const std::exception &error) {
        status.phase = Status::Miss; status.reason = error.what();
        status.loadingState = PrefixCacheRecord::Failed;
        PublishPrefixRecord(ctx);
    }
    PrefixCacheInfo("[PrefixCache] miss: tier=L3(SSD) reason=%s.\n",
                status.reason.c_str());
    FastllmEvent("cache", ctx->requestId, "cache_result",
                 "source_tier=MISS promoted_to=NONE source_tokens=0 effective_reused_tokens=0 "
                 "lookup_ms=%.3f reason=%s", status.lookupMs, status.reason.c_str());
}

bool Qwen35PersistentCache::Import(ResponseContext *ctx, const DiskPrefixCache::Checkpoint &candidate,
                                  DiskPrefixCache::ReadSession *preloaded,
                                  const Json *loadedState,const std::function<bool()> &keepGoing) {
    RequestDiagnosticOperation diagnosticImport(ctx->requestId, "ssd_import",
        "Qwen35PersistentCache.Import");
    const int oldDevice = FastllmCudaGetDevice();
    const int length = candidate.length;
    const bool convert = candidate.kvDtype == "float16" && model->kvCacheDataType == DataType::FP8_E4M3;
    std::map<std::tuple<int,int,int>,std::shared_ptr<Pages>> pools;
    const auto started = Clock::now();
    const std::string previousProvenance = ctx->prefixCacheProvenance;
    // [import-profile] 分段计时：定位 ReadMany 之外约 1.6s 的耗时
    double tLoad=0, tState=0, tKv=0, tCopy=0;
    auto importMark = [&]() {
        return std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    };
    try {
        Check(!keepGoing || keepGoing(),"ssd_restore_cancelled_or_timeout");
        std::optional<DiskPrefixCache::ReadSession> owned;
        if (!preloaded) owned.emplace(store->BeginRead());
        auto &reader = preloaded ? *preloaded : *owned;
        auto state = loadedState ? *loadedState : reader.Load(candidate.key,candidate.identity);
        if (const uint64_t mismatchImport = TestImportIdentityMismatchOrdinal()) {
            Json::object tamperedState = state.object_items();
            Json::array tamperedPageIdentity = tamperedState["page_identity"].array_items();
            Check(!tamperedPageIdentity.empty() && tamperedPageIdentity[0].is_string() &&
                  !tamperedPageIdentity[0].string_value().empty(), "ssd_test_import_identity_hook_shape");
            std::string first = tamperedPageIdentity[0].string_value();
            first[0] = first[0] == '0' ? '1' : '0';
            tamperedPageIdentity[0] = Json(first);
            tamperedState["page_identity"] = Json(std::move(tamperedPageIdentity));
            state = Json(std::move(tamperedState));
            std::fprintf(stderr, "[Prefix SSD] test-import-identity-mismatch-hook: fired=1 import=%llu request=%llu page=0.\n",
                         (unsigned long long)mismatchImport, (unsigned long long)ctx->requestId);
        }
        tLoad = importMark();
        Check(state["prefix"].string_value() == Prefix(ctx,length) &&
            state["layout"].dump() == RuntimeLayout().dump(),"ssd_checkpoint_identity");
        auto snapshot = std::make_shared<Qwen35LinearPrefixSnapshot>();
        snapshot->cachedLen = length; snapshot->tokens.assign(ctx->allTokens.begin(),ctx->allTokens.begin()+length);
        snapshot->layers.resize(model->block_cnt);
        Check(state["linear_bytes"].number_value() > 0 && state["linear_bytes"].number_value() <= HostLimit(),
              "ssd_linear_pack_size");
        std::vector<char> packed((size_t)state["linear_bytes"].number_value());
        Check((double)packed.size() == state["linear_bytes"].number_value(),"ssd_linear_pack_size");
        reader.ReadBytes(state["linear_chunks"],packed.size(),[&](size_t offset,const char *p,size_t n) {
            memcpy(packed.data()+offset,p,n);
        });
        size_t stateBytes = 0;
        for (auto &entry : state["linear"].array_items()) {
            Check(!keepGoing || keepGoing(),"ssd_restore_cancelled_or_timeout");
            int layer = entry["layer"].int_value();
            Check(layer >= 0 && layer < model->block_cnt && Qwen35LayerIsLinearAttention(model,layer) &&
                !snapshot->layers[layer].linear,"ssd_linear_layer");
            auto &out = snapshot->layers[layer]; out.linear = true;
            LoadLinear(packed,entry["first"],layer,0,out.first); LoadLinear(packed,entry["second"],layer,1,out.second);
            for (auto *v : {&out.first,&out.second}) {
                if (v->multiDevice) for (auto &item : v->locals) stateBytes += item.second.GetBytes();
                else stateBytes += v->single.GetBytes();
            }
            Check(stateBytes <= HostLimit(),"ssd_state_budget");
        }
        for (int layer = 0; layer < model->block_cnt; ++layer)
            Check(snapshot->layers[layer].linear == Qwen35LayerIsLinearAttention(model,layer),"ssd_incomplete_linear");
        std::vector<char>().swap(packed);
        tState = importMark();
        std::vector<char> speculativePacked;
        const auto &speculative = state["speculative"];
        Check(speculative.is_object() && state["speculative_bytes"].is_number() &&
                  state["speculative_bytes"].number_value() >= 0 &&
                  state["speculative_bytes"].number_value() <= HostLimit(),
              "ssd_speculative_pack_size");
        const size_t speculativeBytes = (size_t)state["speculative_bytes"].number_value();
        Check((double)speculativeBytes == state["speculative_bytes"].number_value(),
              "ssd_speculative_pack_size");
        speculativePacked.resize(speculativeBytes);
        if (speculativeBytes > 0) {
            reader.ReadBytes(state["speculative_chunks"], speculativeBytes,
                [&](size_t offset,const char *p,size_t n) {
                    memcpy(speculativePacked.data()+offset,p,n);
                });
        }
        LoadSpeculative(speculative, speculativePacked, ctx, length, *snapshot);
        std::vector<char>().swap(speculativePacked);
        const Json &storedPageIdentity = state["page_identity"];
        if (!storedPageIdentity.is_array() || storedPageIdentity.array_items().empty()) {
            std::fprintf(stderr, "[Prefix SSD] import-identity-legacy: page_identity missing or empty; skip page verification.\n");
        } else {
            if (ctx->allTokens.size() < size_t(length)) {
                std::fprintf(stderr, "[Prefix SSD] import-identity-short-context: tokens=%zu length=%d;\n",
                             ctx->allTokens.size(), length);
                throw std::runtime_error("ssd_import_identity_short_context");
            }
            const int pageLen = fastllm::GetPageLen();
            // Same key decision as export (job.runtimeUsesPageKeys = PrefixCachePageKeys() != nullptr).
            const std::vector<std::string> *importPageKeys = ctx->PrefixCachePageKeys();
            const bool usePageKeys = importPageKeys != nullptr;
            const std::vector<std::string> actualPageIdentity = ComputePageIdentities(
                ctx->allTokens, length, pageLen, importPageKeys);
            const auto &expectedPageIdentity = storedPageIdentity.array_items();
            size_t mismatches = 0;
            const size_t pages = std::min(actualPageIdentity.size(), expectedPageIdentity.size());
            for (size_t p = 0; p < pages; ++p)
                if (expectedPageIdentity[p].string_value() != actualPageIdentity[p]) ++mismatches;
            mismatches += actualPageIdentity.size() > pages ? actualPageIdentity.size()-pages : 0;
            mismatches += expectedPageIdentity.size() > pages ? expectedPageIdentity.size()-pages : 0;
            if (mismatches != 0) {
                std::fprintf(stderr, "[Prefix SSD] import-identity-mismatch: pages=%zu of %zu;\n",
                             mismatches, std::max(actualPageIdentity.size(), expectedPageIdentity.size()));
                throw std::runtime_error("ssd_import_identity_mismatch");
            }
            std::fprintf(stderr, "[Prefix SSD] import-identity-ok: pages=%zu length=%d page_keys=%d.\n",
                         actualPageIdentity.size(), length, usePageKeys ? 1 : 0);
        }
        std::vector<std::vector<std::pair<int,PagedCacheManager*>>> attentionRefs(
            (size_t)model->block_cnt * 2);
        struct ImportPageRefsGuard {
            PagedCacheManager *pool = nullptr;
            std::shared_ptr<Pages> lease;
            std::vector<int> held;
            bool committed = false;

            ~ImportPageRefsGuard() {
                if (committed) return;
                if (pool != nullptr && !held.empty()) pool->ReleasePageIndices(held);
                if (lease) {
                    std::fill(lease->ids.begin(), lease->ids.end(), -1);
                    lease->pickedPages = 0;
                }
            }
        };
        std::vector<std::unique_ptr<ImportPageRefsGuard>> importPageGuards;
        // A direct SSD restore may need to replace many L1 pages while the
        // requested prefix is imported. Deferred recycle exposes one reserve
        // page at a time, but PrepareFreePages can dispatch a whole batch.
        // Prepare once per pool and wait at the transaction level; the old
        // per-page GetUnusedPageIndex/10ms loop made a 2048-token restore
        // spend about 20 seconds sleeping in the scheduler thread.
        auto prepareImportCapacity = [&](PagedCacheManager *pool, int missing) {
            if (pool == nullptr || missing <= 0) return;
            RequestDiagnosticOperation diagnosticCapacity(ctx->requestId, "ssd_import_page_capacity_wait",
                RequestDiagnosticResource("PagedCacheManager", pool).c_str());
            const auto deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(30);
            for (;;) {
                {
                    std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
                    if (pool->FreePageCount() >= missing) return;
                    if (!keepGoing || !keepGoing())
                        throw std::runtime_error("ssd_restore_cancelled_or_timeout");
                    if (pool->pendingRecycleTickets.empty()) {
                        // No worker ticket can make progress. One bounded
                        // dispatch attempt below may still find a victim.
                    }
                }
                try { pool->PrepareFreePages(missing); } catch (...) {}
                {
                    std::lock_guard<std::mutex> lock(pool->pageIndexLocker);
                    if (pool->FreePageCount() >= missing) return;
                    if (pool->pendingRecycleTickets.empty())
                        throw std::runtime_error("ssd_gpu_capacity");
                }
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("ssd_gpu_capacity");
                if (!keepGoing || !keepGoing())
                    throw std::runtime_error("ssd_restore_cancelled_or_timeout");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        for (int layer = 0; layer < model->block_cnt; ++layer) {
            if (Qwen35LayerIsLinearAttention(model,layer)) continue;
            Check(!keepGoing || keepGoing(),"ssd_restore_cancelled_or_timeout");
            auto keyRefs = model->GetPagedKVCacheManagers(layer,true);
            auto valueRefs = model->GetPagedKVCacheManagers(layer,false);
            Check(!keyRefs.empty() && keyRefs.size() == valueRefs.size(),
                  "ssd_missing_pool_pair");
            for (size_t index = 0; index < keyRefs.size(); ++index)
                Check(keyRefs[index].first == valueRefs[index].first &&
                          keyRefs[index].second != nullptr &&
                          valueRefs[index].second != nullptr,
                      "ssd_missing_pool_rank");
            attentionRefs[(size_t)layer * 2] = std::move(keyRefs);
            attentionRefs[(size_t)layer * 2 + 1] = std::move(valueRefs);
            for (int side = 0; side < 2; ++side) {
                for (auto ref : attentionRefs[(size_t)layer * 2 + side]) {
                    auto pool = ref.second;
                    Check(length % pool->pageLen == 0,"ssd_checkpoint_page_boundary");
                    int pages = length / pool->pageLen;
                    // Atomically protect any already resident prefix pages
                    // while the SSD payload is assembled. Missing positions
                    // remain -1 and receive fresh GPU pages below.
                    auto lease = std::make_shared<Pages>();
                    lease->pool = pool; lease->rank = ref.first;
                    lease->layer = layer; lease->side = side;
                    lease->pageBytes = (size_t)pool->pageLen*pool->dims[2]*pool->dims[3]*pool->unitSize;
                    if (layer >= 0 && layer < (int)model->threadTpAttentionKVHeadSchemes.size()) {
                        auto scheme = model->threadTpAttentionKVHeadSchemes[layer].find(ref.first);
                        if (scheme != model->threadTpAttentionKVHeadSchemes[layer].end())
                            lease->ranges = scheme->second;
                    }
                    if (lease->ranges.empty()) lease->ranges.push_back({0, pool->dims[2]});
                    lease->ids.reserve(pages);
                    lease->ids.resize(pages, -1);
                    lease->reused.assign(pages, 0);
                    lease->pickedPages = 0;
                    const auto poolKey = std::make_tuple(layer, side, ref.first);
                    Check(pools.emplace(poolKey, lease).second, "ssd_duplicate_pool");
                    auto refsGuard = std::make_unique<ImportPageRefsGuard>();
                    refsGuard->pool = pool;
                    refsGuard->lease = lease;
                    refsGuard->held.reserve((size_t)pages);
                    importPageGuards.push_back(std::move(refsGuard));
                    ImportPageRefsGuard &guard = *importPageGuards.back();

                    std::vector<int> existing;
                    // QueryPagePositions pins resident pages before returning;
                    // reserve its result storage first so a later resize
                    // cannot throw with untracked references already held.
                    existing.reserve((size_t)pages);
                    // Conversion changes the payload dtype, not the logical
                    // page identity. Reuse and pin overlapping L1 pages for
                    // both native and FP16->FP8 restores; allocating a second
                    // page for convert=true would make Record replace the old
                    // trie owner and leave the ledger pointing at a stale GPU
                    // page if later adoption failed.
                    // Diagnosis (default off): FASTLLM_PREFIX_SSD_NO_L1_REUSE=1 restores
                    // every page from this SSD checkpoint instead of reusing resident
                    // L1 pages of the same tokens, so KV and linear state come from
                    // one checkpoint.
                    static const bool noL1Reuse = [] {
                        const char *env = std::getenv("FASTLLM_PREFIX_SSD_NO_L1_REUSE");
                        const bool on = env != nullptr && env[0] == '1' && env[1] == '\0';
                        if (on) std::fprintf(stderr, "[Prefix SSD] no-l1-reuse enabled\n");
                        return on;
                    }();
                    if (!noL1Reuse)
                        pool->QueryPagePositions(ctx->currentTokens, existing,
                                                 ctx->PrefixCachePageKeys(), pages, true);
                    existing.resize(pages, -1);
                    const int missingPages = (int)std::count(existing.begin(), existing.end(), -1);
                    prepareImportCapacity(pool, missingPages);
                    // Register every reference acquired by the atomic query
                    // before allocating any hole. If a later fresh-page
                    // allocation fails, the guard must already own all
                    // resident pins returned by QueryPagePositions.
                    for (int p = 0; p < pages; ++p) {
                        if (existing[p] >= 0) {
                            lease->ids[p] = existing[p];
                            lease->reused[p] = 1;
                            guard.held.push_back(existing[p]);
                            ++lease->pickedPages;
                        }
                    }
                    for (int p = 0; p < pages; ++p) {
                        if (existing[p] >= 0) continue;
                        int page = -1;
                        try { page = pool->GetUnusedPageIndex(true); }
                        catch (...) { throw std::runtime_error("ssd_gpu_capacity"); }
                        Check(page >= 0,"ssd_gpu_capacity");
                        lease->ids[p] = page;
                        guard.held.push_back(page);
                        ++lease->pickedPages;
                    }
                    guard.committed = true;
                }
            }
        }
        std::map<int, Json::array> bundlesByBegin;
        for (const auto &bundle : state["kv"].array_items()) {
            const auto &segments = bundle["segments"].array_items();
            Check(!segments.empty(), "ssd_empty_bundle");
            const int begin = segments.front()["begin"].int_value();
            for (const auto &entry : segments)
                Check(entry["begin"].int_value() == begin, "ssd_mixed_bundle_range");
            bundlesByBegin[begin].push_back(bundle);
        }
        // 2026-10-18（B21）：bundle 缓冲改成不零初始化。std::vector<char>::resize
        // 会先把 3.3 GB 置零再被 ReadMany 覆盖，实测 kv_ms 2.7 s 里有相当一部分是
        // 这次多余的写（fresh page fault + 置零）。unique_ptr<char[]> 不触碰内容。
        struct LoadedBundle { std::unique_ptr<char[]> data; };
        struct ImportSourceSegment {
            const char *source = nullptr;
            int heads = 0;
            int rangeHeads = 0;
            std::vector<std::pair<int,int>> ranges;
        };
        std::map<std::tuple<int,int,int>,int> coverage;
        // H2D 组装路径的实测计数（2026-10-18，B21）：证明整段 span 拷贝是否真的命中，
        // 以及源/目标 head 布局差异；只计数，不改变装配结果。
        uint64_t h2dSpanCopies = 0, h2dHeadCopies = 0;
        uint64_t h2dAssembleBytes = 0, h2dCompareBytes = 0, h2dSegmentVisits = 0;
        int h2dEntryHeads = -1, h2dTargetHeads = -1;
        // 装配缓冲复用（2026-10-18，B21）：原来每个 target pool 都 new 一块
        // tokens*heads*rowBytes（25k 次、每次 128 KB），首次写入的缺页占满 assemble。
        // 改成整次 Import 复用一块只增不减的 scratch。
        std::unique_ptr<char[]> assembleScratch;
        size_t assembleScratchBytes = 0;
        // 按 group 累加的三段计时（2026-10-18，B21）：原来的 tKv 在 group 循环里被覆盖，
        // 导致 kv_ms 把 chunk 拷贝、拼装和 H2D 混在一起。这里分别累计。
        double h2dReadManyMs = 0, h2dAssembleMs = 0, h2dCopyMs = 0;
        // span 拷贝的单次字节直方图（2026-10-18，B21）：64 B vs 256 B 是本次变慢的直接
        // 可疑点，先统计每次 memcpy 的字节数分布，再决定合并策略。
        std::map<size_t, uint64_t> h2dSpanBytes;
        for (const auto &group : bundlesByBegin) {
            Check(!keepGoing || keepGoing(),"ssd_restore_cancelled_or_timeout");
            const int begin = group.first;
            int tokens = -1, dim = -1;
            DataType sourceType = DataType::FLOAT16;
            size_t sourceUnit = 0, loadedBytes = 0;
            std::vector<LoadedBundle> loaded;
            // Preserve segment order within each layer/side. Source storage
            // stays owned by loaded through all target copies and H2D calls.
            std::map<std::pair<int,int>, std::vector<ImportSourceSegment>> sourceSegments;
            std::vector<DiskPrefixCache::ReadSession::Request> requests;
            for (const auto &bundle : group.second) {
                const size_t bundleBytes = (size_t)bundle["bytes"].number_value();
                Check(bundleBytes > 0 && bundleBytes <= BufferBytes &&
                      loadedBytes <= HostLimit() - bundleBytes, "ssd_bundle_bytes");
                LoadedBundle item;
                const auto &segments = bundle["segments"].array_items();
                item.data.reset(new char[bundleBytes]);
                size_t nextOffset = 0;
                for (const auto &entry : segments) {
                    const int entryTokens = entry["tokens"].int_value();
                    const int entryHeads = entry["heads"].int_value();
                    const int entryDim = entry["dim"].int_value();
                    const DataType entryType = (DataType)entry["dtype"].int_value();
                    const size_t entryUnit = entryType == DataType::FP8_E4M3 ? 1 : 2;
                    const size_t entryBytes = (size_t)entryTokens * entryHeads * entryDim * entryUnit;
                    const size_t entryOffset = (size_t)entry["offset"].number_value();
                    Check(entry["begin"].int_value() == begin && entryTokens > 0 &&
                          entryBytes == (size_t)entry["bytes"].number_value() &&
                          entryOffset == nextOffset && entryBytes <= bundleBytes-entryOffset,
                          "ssd_bundle_segment");
                    if (tokens < 0) {
                        tokens = entryTokens; dim = entryDim;
                        sourceType = entryType; sourceUnit = entryUnit;
                    } else {
                        Check(tokens == entryTokens && dim == entryDim &&
                              sourceType == entryType && sourceUnit == entryUnit,
                              "ssd_kv_source_layout");
                    }
                    ImportSourceSegment segment;
                    segment.source = item.data.get() + entryOffset;
                    segment.heads = entryHeads;
                    for (const auto &range : entry["ranges"].array_items()) {
                        Check(range.is_array() && range.array_items().size() == 2, "ssd_kv_ranges");
                        segment.ranges.push_back({range[0].int_value(),range[1].int_value()});
                    }
                    if (segment.ranges.empty()) segment.ranges.push_back({0,entryHeads});
                    for (auto range : segment.ranges)
                        segment.rangeHeads += range.second-range.first;
                    sourceSegments[{entry["layer"].int_value(),entry["side"].int_value()}]
                        .push_back(std::move(segment));
                    nextOffset += entryBytes;
                }
                Check(nextOffset == bundleBytes, "ssd_bundle_coverage");
                loadedBytes += bundleBytes;
                const size_t index = loaded.size();
                loaded.push_back(std::move(item));
                requests.push_back({bundle["chunks"],bundleBytes,
                    [&loaded,index](size_t offset,const char *p,size_t n) {
                        memcpy(loaded[index].data.get()+offset,p,n);
                    }});
            }
            {
                const auto readManyStart = std::chrono::steady_clock::now();
                reader.ReadMany(requests);
                h2dReadManyMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - readManyStart).count();
            }
            tKv = importMark();
            Check(tokens > 0 && begin >= 0 && begin + tokens <= length, "ssd_kv_token_range");
            for (auto &target : pools) {
                const auto targetKey = target.first;
                auto &lease = *target.second;
                const int layer = std::get<0>(targetKey), side = std::get<1>(targetKey);
                Check(layer >= 0 && layer < model->block_cnt && side >= 0 && side <= 1 &&
                      coverage[targetKey] == begin, "ssd_kv_layout");
                const int targetHeads = lease.pool->dims[2];
                const int targetDim = lease.pool->dims[3];
                const size_t targetUnit = lease.pool->unitSize;
                Check(targetDim == dim &&
                      (targetUnit == sourceUnit || (convert && sourceType == DataType::FLOAT16)),
                      "ssd_kv_dtype");
                int targetRangeHeads = 0;
                for (auto range : lease.ranges) {
                    Check(range.first >= 0 && range.second > range.first &&
                              range.second <= model->num_key_value_heads,
                          "ssd_target_range_bounds");
                    targetRangeHeads += range.second-range.first;
                }
                Check(targetRangeHeads == targetHeads, "ssd_target_range_heads");
                const size_t rowBytes = (size_t)targetDim * sourceUnit;
                // 不零初始化：下面按 head/token 完整覆盖，并由 all_of(filled) 兜底；
                // 置零 3.3 GB 是纯浪费（2026-10-18，B21）。
                const auto assembleStart = std::chrono::steady_clock::now();
                const size_t outputBytes = (size_t)tokens * targetHeads * rowBytes;
                if (outputBytes > assembleScratchBytes) {
                    assembleScratch.reset(new char[outputBytes]);
                    assembleScratchBytes = outputBytes;
                }
                char *outputPtr = assembleScratch.get();
                std::vector<uint8_t> filled(targetHeads, 0);
                const auto matching = sourceSegments.find({layer,side});
                if (matching != sourceSegments.end()) for (const auto &entry : matching->second) {
                    ++h2dSegmentVisits;
                    const int entryHeads = entry.heads;
                    const auto &sourceRanges = entry.ranges;
                    Check(entry.rangeHeads == entryHeads, "ssd_kv_range_heads");
                    const char *source = entry.source;
                    int sourceHeadBase = 0;
                    // 同构快速路径（2026-10-18，B21）：当本段覆盖全部 head、源/目标 head
                    // 布局一致（单 range 0..targetHeads）时，整块 tokens*heads*rowBytes 一次
                    // memcpy 完成，替代逐 head/逐 token 的小拷贝。真实请求里 KV 组装
                    // 2.58 s 只跑到约 1.25 GB/s，瓶颈就是这种小拷贝。非 identity 布局仍走
                    // 原逐 head 路径（含 replicated head 的 memcmp 校验）。
                    // The global head range must match: equally sized TP shards
                    // may hold different heads. An already filled replica must
                    // take the checked comparison path below, never overwrite it.
                    const bool identityLayout =
                        entryHeads == targetHeads && sourceRanges.size() == 1 &&
                        sourceRanges[0].second - sourceRanges[0].first == entryHeads &&
                        lease.ranges.size() == 1 &&
                        lease.ranges[0].second - lease.ranges[0].first == targetHeads &&
                        sourceRanges[0] == lease.ranges[0] &&
                        std::none_of(filled.begin(), filled.end(), [](uint8_t v){return v != 0;});
                    if (identityLayout) {
                        memcpy(outputPtr, source, (size_t)tokens * entryHeads * rowBytes);
                        h2dAssembleBytes += (size_t)tokens * entryHeads * rowBytes;
                        std::fill(filled.begin(), filled.end(), uint8_t(1));
                        sourceHeadBase += targetHeads;
                        h2dSpanCopies += (uint64_t)tokens;
                        h2dSpanBytes[(size_t)tokens * entryHeads * rowBytes] += 1;
                        h2dEntryHeads = entryHeads;
                        h2dTargetHeads = targetHeads;
                    } else
                    for (auto sourceRange : sourceRanges) {
                        Check(sourceRange.first >= 0 && sourceRange.second > sourceRange.first &&
                                  sourceRange.second <= model->num_key_value_heads,
                              "ssd_kv_range_size");
                        int targetHeadBase = 0;
                        for (auto targetRange : lease.ranges) {
                            const int lo = std::max(sourceRange.first,targetRange.first);
                            const int hi = std::min(sourceRange.second,targetRange.second);
                            const int spanHeads = hi - lo;
                            if (spanHeads > 0) {
                                const int sourceHeadStart = sourceHeadBase+lo-sourceRange.first;
                                const int targetHeadStart = targetHeadBase+lo-targetRange.first;
                                Check(targetHeadStart >= 0 && targetHeadStart+spanHeads <= targetHeads &&
                                          sourceHeadStart >= 0 && sourceHeadStart+spanHeads <= entryHeads,
                                      "ssd_kv_range_bounds");
                                bool anyFilled = false;
                                for (int head = 0; head < spanHeads; ++head)
                                    if (filled[targetHeadStart+head]) { anyFilled = true; break; }
                                if (!anyFilled) {
                                    // 连续 span：每个 token 一次 memcpy 覆盖整段 head，
                                    // 替代原来每 head 一次的小拷贝（2026-10-18，B21）。
                                    const size_t spanBytes = (size_t)spanHeads*rowBytes;
                                    for (int token = 0; token < tokens; ++token) {
                                        memcpy(outputPtr+
                                                   ((size_t)token*targetHeads+targetHeadStart)*rowBytes,
                                               source+
                                                   ((size_t)token*entryHeads+sourceHeadStart)*rowBytes,
                                               spanBytes);
                                    }
                                    h2dAssembleBytes += (uint64_t)tokens * spanBytes;
                                    for (int head = 0; head < spanHeads; ++head)
                                        filled[targetHeadStart+head] = 1;
                                    h2dSpanCopies += (uint64_t)tokens;
                                    h2dSpanBytes[spanBytes] += (uint64_t)tokens;
                                    h2dEntryHeads = entryHeads;
                                    h2dTargetHeads = targetHeads;
                                } else {
                                    for (int globalHead = lo; globalHead < hi; ++globalHead) {
                                        const int sourceHead = sourceHeadBase+globalHead-sourceRange.first;
                                        const int targetHead = targetHeadBase+globalHead-targetRange.first;
                                        for (int token = 0; token < tokens; ++token) {
                                            char *destination = outputPtr+
                                                ((size_t)token*targetHeads+targetHead)*rowBytes;
                                            const char *sourceRow = source+
                                                ((size_t)token*entryHeads+sourceHead)*rowBytes;
                                            // TP schemes may intentionally replicate a
                                            // KV head; redundant copies are safe only when
                                            // their bytes agree.
                                            h2dCompareBytes += rowBytes;
                                            Check(memcmp(destination,sourceRow,rowBytes) == 0,
                                                  "ssd_kv_replicated_conflict");
                                        }
                                    }
                                    for (int head = 0; head < spanHeads; ++head)
                                        filled[targetHeadStart+head] = 1;
                                    h2dHeadCopies += (uint64_t)spanHeads * tokens;
                                    h2dSpanBytes[rowBytes] += (uint64_t)spanHeads * tokens;
                                }
                            }
                            targetHeadBase += targetRange.second-targetRange.first;
                        }
                        sourceHeadBase += sourceRange.second-sourceRange.first;
                    }
                    (void)sourceHeadBase;
                }
                Check(std::all_of(filled.begin(),filled.end(),[](uint8_t v){return v != 0;}),
                      "ssd_kv_missing_range");
                h2dAssembleMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - assembleStart).count();
                const auto copyStart = std::chrono::steady_clock::now();
                if (convert) {
                    FastllmCudaSetDevice(lease.rank);
                    Data source(DataType::FLOAT16,std::vector<int>{targetHeads,tokens,targetDim});
                    source.Allocate();
                    for (int token = 0; token < tokens; ++token) for (int head = 0; head < targetHeads; ++head)
                        memcpy(source.cpuData+((size_t)head*tokens+token)*rowBytes,
                               outputPtr+((size_t)token*targetHeads+head)*rowBytes,rowBytes);
                    source.ToDevice(DataDevice::CUDA,{lease.rank},true);
                    bool hasReused = false;
                    const int firstPage = begin / lease.pool->pageLen;
                    const int lastPage = (begin + tokens - 1) / lease.pool->pageLen;
                    for (int page = firstPage; page <= lastPage; ++page)
                        if (page >= 0 && page < (int)lease.reused.size() && lease.reused[page]) {
                            hasReused = true;
                            break;
                        }
                    if (!hasReused) {
                        Check(FastllmCudaPagedCacheCopyMultiPage((uint8_t*)lease.pool->cudaData,
                            lease.ids.data()+begin/lease.pool->pageLen,
                            (begin%lease.pool->pageLen+tokens+lease.pool->pageLen-1)/lease.pool->pageLen,
                            begin%lease.pool->pageLen,
                            lease.pool->pageLen,targetHeads,targetDim,lease.pool->dataType,
                            (uint8_t*)source.cudaData,source.dataType,tokens),"ssd_kv_convert");
                    } else {
                        // A reused L1 page is already visible to other readers.
                        // Convert only fresh pages; never overwrite a pinned page
                        // while assembling a failing SSD import.
                        for (int copied = 0; copied < tokens; ) {
                            const int page = (begin+copied) / lease.pool->pageLen;
                            const int within = (begin+copied) % lease.pool->pageLen;
                            const int count = std::min(tokens-copied,lease.pool->pageLen-within);
                            if (page >= 0 && page < (int)lease.reused.size() && lease.reused[page]) {
                                copied += count;
                                continue;
                            }
                            Data pageSource(DataType::FLOAT16,
                                std::vector<int>{targetHeads,count,targetDim});
                            pageSource.Allocate();
                            for (int token = 0; token < count; ++token) {
                                for (int head = 0; head < targetHeads; ++head) {
                                    memcpy(pageSource.cpuData+
                                               ((size_t)head*count+token)*rowBytes,
                                           outputPtr+
                                               ((size_t)(copied+token)*targetHeads+head)*rowBytes,
                                           rowBytes);
                                }
                            }
                            pageSource.ToDevice(DataDevice::CUDA,{lease.rank},true);
                            Check(FastllmCudaPagedCacheCopyMultiPage(
                                (uint8_t*)lease.pool->cudaData,
                                lease.ids.data()+page, 1, within,
                                lease.pool->pageLen,targetHeads,targetDim,
                                lease.pool->dataType,(uint8_t*)pageSource.cudaData,
                                pageSource.dataType,count),"ssd_kv_convert");
                            FastllmCudaSyncCurrentThreadStream();
                            copied += count;
                        }
                    }
                    FastllmCudaSyncCurrentThreadStream();
                } else {
                    FastllmCudaSetDevice(lease.rank);
                    const size_t tokenBytes = targetHeads*targetDim*targetUnit;
                    for (int copied = 0; copied < tokens;) {
                        const int page = (begin+copied)/lease.pool->pageLen;
                        const int within = (begin+copied)%lease.pool->pageLen;
                        const int count = std::min(tokens-copied,lease.pool->pageLen-within);
                        if (page >= 0 && page < (int)lease.reused.size() &&
                            lease.reused[page]) {
                            copied += count;
                            continue;
                        }
                        FastllmCudaCopyFromHostToDevice(
                            (char*)lease.pool->cudaData+(size_t)lease.ids[page]*lease.pageBytes+within*tokenBytes,
                            outputPtr+(size_t)copied*tokenBytes,count*tokenBytes);
                        copied += count;
                    }
                }
                h2dCopyMs += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - copyStart).count();
                coverage[targetKey] += tokens;
            }
        }
        for (auto &entry : pools) Check(coverage[entry.first] == length,"ssd_incomplete_kv");
        Check(!keepGoing || keepGoing(),"ssd_restore_cancelled_or_timeout");
        // Make every native/conversion H2D write complete before recording the
        // producer events consumed by the direct L1 publication transaction.
        const auto h2dSyncStart = std::chrono::steady_clock::now();
        RequestDiagnosticOperation diagnosticSync(ctx->requestId, "ssd_h2d_completion_wait",
            "CUDA.current_thread_stream");
        FastllmCudaSyncCurrentThreadStream();
        diagnosticSync.Finish();
        const double h2dSyncMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - h2dSyncStart).count();
        tCopy = importMark();
        char spanHistText[256]; spanHistText[0] = 0;
        {
            size_t used = 0; int shown = 0;
            for (const auto &entry : h2dSpanBytes) {
                if (shown++ >= 6) break;
                used += (size_t)std::snprintf(spanHistText+used, sizeof(spanHistText)-used,
                                              "%s%zu:%llu", used ? "," : "", entry.first,
                                              (unsigned long long)entry.second);
                if (used + 32 >= sizeof(spanHistText)) break;
            }
        }
        // 2026-10-18（B21）：H2D 窗口的原生分段。copy_ms 是 tKv→sync 完成的整段
        // （含拷贝调用与最后一次全流同步），sync_ms 单独量最后一个同步的等待；
        // 两者都是在既有同步点上计时，不新增任何同步或拷贝。
        FastllmEvent("cache", ctx->requestId, "cache_h2d_profile",
                     "tokens=%d load_ms=%.3f state_ms=%.3f kv_ms=%.3f copy_ms=%.3f "
                     "sync_ms=%.3f span_copies=%llu head_copies=%llu entry_heads=%d "
                     "target_heads=%d readmany_ms=%.3f assemble_ms=%.3f h2d_ms=%.3f "
                     "span_bytes=%s assemble_bytes=%llu compare_bytes=%llu "
                     "assemble_segment_visits=%llu total_ms=%.3f",
                     length, tLoad, tState-tLoad, tKv-tState, tCopy-tKv, h2dSyncMs,
                     (unsigned long long)h2dSpanCopies, (unsigned long long)h2dHeadCopies,
                     h2dEntryHeads, h2dTargetHeads, h2dReadManyMs, h2dAssembleMs, h2dCopyMs,
                     spanHistText, (unsigned long long)h2dAssembleBytes,
                     (unsigned long long)h2dCompareBytes,
                     (unsigned long long)h2dSegmentVisits, importMark());
        PrefixCacheDebug("[Prefix SSD] import-profile: load_ms=%.1f state_ms=%.1f kv_ms=%.1f copy_ms=%.1f total_ms=%.1f.\n",
                         tLoad, tState-tLoad, tKv-tState, tCopy-tKv, importMark());
        std::string provenance = candidate.provenance;
        if (convert) provenance = DiskPrefixCache::Digest("fp16-to-fp8-v1:"+candidate.identity+candidate.key+provenance);
        if (!provenance.empty()) {
            std::lock_guard<std::mutex> lock(mutex);
            provenanceSources[provenance] = convert ? Json(Json::object{{"checkpoint",candidate.key},
                {"identity",candidate.identity},{"source_dtype",candidate.kvDtype},
                {"target_dtype",Dtype(model->kvCacheDataType)},{"conversion_version",1},
                {"parent_provenance",candidate.provenance}}) : state["source"];
        }
        SetProvenance(ctx,provenance);
        if (!ctx->prefixCachePageKeys.empty()) snapshot->mediaPageKeys.assign(ctx->prefixCachePageKeys.begin(),
            ctx->prefixCachePageKeys.begin()+length/fastllm::GetPageLen());
        Check(Qwen35SetSnapshotMatchIdentity(ctx, *snapshot), "ssd_snapshot_prefix_identity");
        // The H2D work above is the producer for the newly adopted L1 pages.
        // Record its real stream events before the atomic ledger publication.
        model->RecordPagedPrefixProducerEventsOnCurrentThread();
        bool directPagesRolledBack = false;
        auto rollbackDirectPages = [&]() {
            if (directPagesRolledBack) return;
            directPagesRolledBack = true;
            // PublishDirectPagedPrefix rolls back trie/ledger state, while the
            // Import-owned Pages objects own the allocator references. Release
            // every picked or newly allocated page exactly once and mark it so
            // ~Pages cannot decrement the same reference again.
            for (auto &entry : pools) {
                auto &lease = *entry.second;
                if (!lease.ids.empty()) lease.pool->ReleasePageIndices(lease.ids);
                std::fill(lease.ids.begin(), lease.ids.end(), -1);
                lease.pickedPages = 0;
            }
        };
        // Reserve the post-adoption snapshot slot before touching GPU/L1.
        // Once the ledger commits, vector growth must not turn a successful
        // adoption into an exception that the outer Import catch reports as a
        // failed restore.
        {
            std::lock_guard<std::mutex> lock(Qwen35LinearPrefixSnapshotsMutex());
            auto &items = Qwen35LinearPrefixSnapshots()[model];
            items.reserve(Qwen35LinearPrefixSnapshotMaxRecords() + 1);
        }
        const double publishBeginMs = importMark();
        RequestDiagnosticOperation diagnosticPublish(ctx->requestId, "ssd_l1_publication",
            "Qwen3_5Model.PublishDirectPagedPrefix");
        Check(model->PublishDirectPagedPrefix(ctx, length, [&]() {
            for (auto &entry : pools)
                entry.second->pool->Record(snapshot->tokens, entry.second->ids,
                                           ctx->PrefixCachePageKeys());
            return true;
        }, rollbackDirectPages), "ssd_direct_l1_adoption_failed");
        diagnosticPublish.Finish();
        const double publishEndMs = importMark();
        FastllmEvent("cache", ctx->requestId, "cache_import_profile",
                     "measurement=native_ssd_import timing_scope=through_l1_publication tokens=%d "
                     "pre_publication_ms=%.3f l1_publication_ms=%.3f "
                     "elapsed_ms=%.3f",
                     length, publishBeginMs, publishEndMs-publishBeginMs, publishEndMs);
        // Keep the imported GPU references alive until the scheduler installs
        // the pages into ctx->pastKeyValues and Pick() takes its normal owner.
        if (model->pagedSpillState) {
            std::vector<std::pair<PagedCacheManager *, std::vector<int>>> ownerPages;
            for (auto &entry : pools) {
                auto &lease = *entry.second;
                if (!lease.ids.empty()) {
                    ownerPages.emplace_back(lease.pool, lease.ids);
                    std::fill(lease.ids.begin(), lease.ids.end(), -1);
                    lease.pickedPages = 0;
                }
            }
            model->AdoptPagedPrefixRestoreOwners(ctx, std::move(ownerPages));
        }
        // Publish only after every page/layer/rank has passed the unified
        // ledger adoption.  A failed import therefore cannot leave a GPU/L2
        // pair visible to a later request.
        {
            std::lock_guard<std::mutex> lock(Qwen35LinearPrefixSnapshotsMutex());
            snapshot->timestamp = ++Qwen35LinearPrefixSnapshotTimestamp();
            auto &items = Qwen35LinearPrefixSnapshots()[model];
            items.push_back(snapshot);
            while ((int)items.size() > Qwen35LinearPrefixSnapshotMaxRecords())
                items.erase(items.begin());
        }
        try {
            ctx->intParams["ssd_loaded_len"] = length;
            int copies = 0;
            for (const auto &entry : pools) copies += (int)entry.second->ids.size();
            ctx->intParams["ssd_restore_copies"] = copies;
            ctx->intParams["ssd_source_fp16"] = candidate.kvDtype == "float16";
        } catch (const std::exception &postCommitError) {
            // The GPU/L1 adoption is already durable in the runtime ledger;
            // metadata bookkeeping must not roll it back into a false miss.
            std::fprintf(stderr, "[Prefix SSD] direct restore metadata warning: %s.\n",
                         postCommitError.what());
        }
        // 0925 适配撤回（2026-09-26）：直接接管 ctx->cacheLen 会让回调/打印正确，但绕过模型侧
        // 采纳流程（currentTokens 裁剪、分页缓存接管）导致 Aborted。恢复长度必须由模型侧
        // tryRestorePrefixCache 正常认下来，故此处不再接管。
        reads += reader.ReadBytesCount();
        const double elapsedMs = std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        { std::lock_guard<std::mutex> lock(mutex);
          double &rate = convert ? convertedReadMsPerToken : nativeReadMsPerToken;
          rate = rate > 0 ? 0.8*rate + 0.2*elapsedMs/length : elapsedMs/length; }
        const double loadedMs = std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        PrefixCacheDebug("[Prefix SSD] loaded: tokens=%d source_dtype=%s target_dtype=%s ms=%.3f "
               "read_bytes=%llu tok_s=%.0f phase=%s adopted=0.\n",length,
            candidate.kvDtype.c_str(),Dtype(model->kvCacheDataType).c_str(),loadedMs,
            (unsigned long long)reader.ReadBytesCount(),loadedMs > 0 ? length*1000.0/loadedMs : 0.0,
               "request_gpu_restore");
        FastllmCudaSetDevice(oldDevice);
        return true;
    } catch (const std::exception &e) {
        SetProvenance(ctx,previousProvenance);
        ctx->intParams["ssd_loaded_len"] = 0;
        ctx->intParams["ssd_restore_copies"] = 0;
        ++skipped;
        LastImportFailureReason() = e.what();
        std::fprintf(stderr,"[Prefix SSD] restore miss: tokens=%d reason=%s.\n",length,e.what());
        FastllmCudaSetDevice(oldDevice);
        return false;
    }
}

    #include "qwen3_5_prefetch.inc"

#else
class Qwen35PersistentCache {
public:
    const std::string &EncoderIdentity() const { static const std::string empty; return empty; }
    void Initialize() {}
    void StartPrefetch() {}
    void PrefetchReady() {}
    static std::shared_ptr<Qwen35PersistentCache> Create(Qwen3_5Model*) { return {}; }
    void Created(ResponseContext*) {}
    void Replanned(ResponseContext*) {}
    void Removed(ResponseContext*) {}
    void CancelRead(ResponseContext*) {}
    void Stop() {}
    void CloseAdmission() {}
    void ObservePrefill(int,double) {}
    bool Wait(int) { return true; }
    std::string Stats() const { return "{}"; }
    void Prepare(ResponseContext*) {}
    size_t LoadImage(const std::string&,size_t,std::vector<float>&) { return 0; }
    void SaveImage(const std::string&,const float*,size_t) {}
    void Restored(ResponseContext*) {}
    void Capture(ResponseContext*,const std::shared_ptr<Qwen35LinearPrefixSnapshot>&) {}
};
#endif

void Qwen3_5Model::InitializePersistentPrefixCache() {
    if (persistentPrefixCache) persistentPrefixCache->Initialize();
}
void Qwen3_5Model::PreparePersistentPrefixCache(ResponseContext *context) {
    if (persistentPrefixCache) persistentPrefixCache->Prepare(context);
    else if (context && context->cacheLen == 0 && Qwen35ReadyPagedPrefix(this,context) == 0)
        PreparePagedSpillPrefix(context);
}
void Qwen3_5Model::ResetPersistentPrefixRestore(ResponseContext *context) {
    if (!context) return;
    ReleasePagedPrefixRestoreOwners(context);
    ReleasePagedSpillRetryHolds(context);
    if (persistentPrefixCache) persistentPrefixCache->CancelRead(context);
    context->prefixRestore = PrefixCacheRestoreStatus{};
}
void Qwen3_5Model::OnPersistentPrefixRestored(ResponseContext *context) {
    if (persistentPrefixCache) persistentPrefixCache->Restored(context);
}
void Qwen3_5Model::ClosePersistentPrefixAdmission() {
    if (persistentPrefixCache) persistentPrefixCache->CloseAdmission();
}
bool Qwen3_5Model::WaitPersistentPrefixCache(int timeoutMs) {
    return !persistentPrefixCache || persistentPrefixCache->Wait(timeoutMs);
}
std::string Qwen3_5Model::PersistentPrefixCacheStatistics() const {
    return persistentPrefixCache ? persistentPrefixCache->Stats() : "{}";
}
void Qwen3_5Model::ObservePersistentPrefill(int tokens, double milliseconds) {
    if (persistentPrefixCache) persistentPrefixCache->ObservePrefill(tokens,milliseconds);
}
