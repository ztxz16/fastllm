#ifndef FASTLLM_DISK_PREFIX_CACHE_H
#define FASTLLM_DISK_PREFIX_CACHE_H

#ifdef FASTLLM_DISK_PREFIX_CACHE
#include "json11.hpp"
#include "utils/disk_prefix_write_coordinator.h"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {
// Immutable checksummed prefix records and complete atomic checkpoints backed by
// the v4 manifest-tree store.
class DiskPrefixCache {
    struct Impl;
public:
    using Json = json11::Json;
    enum class RestoreFailureClass { UNKNOWN, LOGICAL, STORED_REPRESENTATION, OPERATIONAL };
    // (verified-byte delta, discovered-byte delta). Total becomes exact when
    // the complete manifest has been visited. Returning false cancels at an
    // object boundary; an absent observer leaves writer/adopt reads unchanged.
    using ReadProgress = std::function<bool(uint64_t, uint64_t)>;
    using CheckpointReclaimPredicate =
        std::function<bool(const std::string &key, const std::string &identity)>;
    using CheckpointCorruptRetirePredicate = CheckpointReclaimPredicate;
    static constexpr int Version = 4;
    static constexpr size_t ChunkBytes = 8 << 20;
    static constexpr size_t MaxMetadataBytes = 64 << 20;
    // All packing/read workers share this bound rather than allocating a whole
    // checkpoint payload at once.
    static constexpr size_t BatchBytes = 32 << 20;
    struct Checkpoint {
        std::string key, identity, familyIdentity, kind, kvDtype, provenance;
        int length = 0;
        uint64_t createdNs = 0;
        Json metadata;
    };
    // Immutable identity carried from request admission through the durable
    // commit callback. An empty value is retained only for legacy non-A paths;
    // once any field is supplied the complete tuple is mandatory.
    struct CommitIdentity {
        uint64_t requestId = 0, requestGeneration = 0;
        uint64_t pageGeneration = 0, copyGeneration = 0;
        std::string cacheIdentity;
        uint64_t ownerLease = 0;
        bool HasAny() const noexcept {
            return requestId || requestGeneration || pageGeneration || copyGeneration ||
                   !cacheIdentity.empty() || ownerLease;
        }
        bool Valid() const noexcept {
            return requestId && requestGeneration && pageGeneration && copyGeneration &&
                   !cacheIdentity.empty() && ownerLease;
        }
        bool operator==(const CommitIdentity &other) const noexcept {
            return requestId == other.requestId && requestGeneration == other.requestGeneration &&
                   pageGeneration == other.pageGeneration && copyGeneration == other.copyGeneration &&
                   cacheIdentity == other.cacheIdentity && ownerLease == other.ownerLease;
        }
    };
    class WriteSession {
        struct State;
        std::unique_ptr<State> state;
        explicit WriteSession(std::unique_ptr<State> state);
        friend class DiskPrefixCache;
    public:
        WriteSession(WriteSession &&) noexcept;
        WriteSession &operator=(WriteSession &&) noexcept;
        ~WriteSession();
        WriteSession(const WriteSession &) = delete;
        WriteSession &operator=(const WriteSession &) = delete;
        // At most ChunkBytes per callback. A previous descriptor is only a
        // hint: its position never proves unchanged contents.
        Json::array StoreBytes(size_t bytes,
            const std::function<void(size_t, char *, size_t)> &read,
            const Json &previous = Json());
        // v4 only: accepts an owned result made by the fixed hash pool.
        Json::array StoreVerified(prefix_write::HashedChunk &&chunk);
        // State supplies kind, length, family_identity, kv_dtype, provenance.
        // All nested {sha256,bytes} references join the complete commit,
        // including shared KV objects stored by earlier sessions.
        // keepEquivalent retains an existing encoding of the same fully
        // classified target-prefix semantics; arbitrary conflicts still fail.
        bool Commit(const std::string &key, const Json &state, bool keepEquivalent = false);
        // Completion is delivered by the existing commit pipeline after the
        // COMMIT is durable (or fails), never at asynchronous admission.
        bool CommitTracked(const std::string &key, const Json &state,
                           bool keepEquivalent, std::function<void(bool)> completion);
        bool CommitTracked(const std::string &key, const Json &state,
                           bool keepEquivalent, const CommitIdentity &identity,
                           std::function<void(bool, const CommitIdentity &)> completion);
        uint64_t WrittenBytes() const;
    };
    class ReadSession {
        struct State;
        std::unique_ptr<State> state;
        explicit ReadSession(std::unique_ptr<State> state);
        friend class DiskPrefixCache;
        // B1：Load / LoadManifest 的公共实现体。
        // manifestOnly=true ⇒ 走清单-only 加载（见 LoadManifest 的说明）。
        Json LoadInternal(const std::string &key, const std::string &sourceIdentity, bool manifestOnly,
                          bool publishedOnly = false, const ReadProgress &progress = {});
    public:
        struct Request {
            Json chunks;
            size_t expected = 0;
            std::function<void(size_t, const char *, size_t)> write;
        };
        ReadSession(ReadSession &&) noexcept;
        ReadSession &operator=(ReadSession &&) noexcept;
        ~ReadSession();
        ReadSession(const ReadSession &) = delete;
        ReadSession &operator=(const ReadSession &) = delete;
        Json Load(const std::string &key, const std::string &sourceIdentity = "",
                  bool publishedOnly = false, const ReadProgress &progress = {});
        // B1：**清单-only 加载**。只取「状态对象 + 块对象」——
        // 也就是合成 state["kv"] 所需的全部 JSON 元数据 —— 而**跳过块所引用的 KV 分片载荷**
        // （T0 实测：一次 Load 共 520 个对象，其中 463 个是载荷、共 566.2 MiB）。
        //
        // 只有「父检查点继承」（adopt）能用它：那条路径只读 parent["kv"] 的**元数据**，
        // 从不调用 ReadMany 去取载荷（证据：成员 `state->inherited` —— 它才是载荷的唯一持有者
        // —— 从被赋值到被 reset 之间**没有任何解引用**，见 qwen3_5_persistent.inc:1598/2037/2561）。
        // **读回路径（Import）绝不能用**：它要的是真字节。
        //
        // 安全性：被跳过的载荷，其存在性由**下游提交**的「只引用条目」批量校验把关
        // （disk_prefix_cache.cpp:4317-4336，缺一个即 fail-closed 报 v4_object_missing），
        // 该校验本来就存在、与本次改动无关；因此不会发布"引用不存在对象"的坏检查点。
        Json LoadManifest(const std::string &key, const std::string &sourceIdentity = "");
        // Each chunk is verified before callback. The caller imports into
        // unpublished state and rolls back if a subsequent chunk fails.
        void ReadBytes(const Json &chunks, size_t expected,
            const std::function<void(size_t, const char *, size_t)> &write);
        // Verifies bounded object reads on I/O workers. Callbacks run on the
        // calling thread.
        void ReadMany(const std::vector<Request> &requests);
        uint64_t ReadBytesCount() const;
        uint64_t LoadedBytesCount() const;
    };
    DiskPrefixCache(const std::string &directory, const std::string &identity,
                    uint64_t limitBytes = uint64_t(256) << 30,
                    const std::string &familyIdentity = "",
                    int ioThreads = 4);
    ~DiskPrefixCache();
    DiskPrefixCache(const DiskPrefixCache &) = delete;
    DiskPrefixCache &operator=(const DiskPrefixCache &) = delete;
    WriteSession BeginWrite(uint64_t reservationBytes = 0);
    ReadSession BeginRead() const;
    // Exact lookup: exactIdentity. Family lookup: empty exactIdentity + family.
    std::vector<Checkpoint> ListCheckpoints(const std::string &exactIdentity = "",
        const std::string &familyIdentity = "", int maxLength = INT_MAX,
        const std::string &kind = "prefix", const std::string &prefix = "") const;
    std::vector<int> CheckpointLengths(const std::string &familyIdentity, int maxLength,
        const std::string &kind = "prefix", bool publishedOnly = false) const;
    // 【业界式「L3 实时问一次」】对标 SGLang HiCache 官方设计原文：
    //   "HiRadixTree does **not** store or continuously synchronize metadata for L3 KV cache.
    //    Instead, when accessing L3 data, it **queries the backend in real time**."
    // 本函数就是那"一次查询"：给定候选 (长度, 该长度的前缀摘要) 列表，**一条 SQL** 回答
    // "这些前缀里哪些在库里、各自对应哪个 identity/key/长度"。
    //
    // 取代模型侧现在的 "1 次 CheckpointLengths + N 次 ListCheckpoints"（N≈23 ⇒ 约 24 条 SQL/请求）。
    // **语义与逐长度 ListCheckpoints 逐条等价**（同 kind/family/live 过滤；同 length 降序 + created 升序），
    // 由 T0 探针**逐条对拍**验证（用旧接口的结果当基准，喂给新接口，要求输出完全相同）。
    //
    // 纯新增：不改盘上格式、不加列/表（走已有的 `checkpoints_prefix` 索引）。
    // 空 `prefixes` ⇒ 返回空。base/derived 与 `Checkpoint` 同构，便于调用方零改动接排序逻辑。
    std::vector<Checkpoint> MatchPrefixes(const std::string &familyIdentity,
        const std::vector<std::pair<int, std::string>> &prefixes,
        const std::string &kind = "prefix", bool publishedOnly = false) const;
    // Calling thread must not hold sessions. Reconstructs references from
    // commits and reclaims interrupted writes and unreachable objects.
    void RebuildIndex();
    // V5 junction (FASTLLM_PREFIX_V5=1): tokens shared with any stored checkpoint,
    // from per-page chain identities. Always 0 for V4.
    int JunctionLength(const std::vector<std::string> &pageIdentities, int pageLen) const;
    // 启动自检转发（v4 模式生效）：报告"写了但恢复不了"的检查点。
    void SelfCheckStorage();
    std::filesystem::path Directory() const;
    void PutMetadata(const std::string &key, const Json &value);
    Json GetMetadata(const std::string &key) const;
    Json CapacityStatus() const;
    Json CapacityObservation() const; // No actor drain, write or index rebuild.
    // Metadata invalidation after a durable ERASE and catalog retirement. Each
    // cache instance subscribes independently to its shared canonical store.
    // Called on the CommitActor without an actor-queue mutex or SQL transaction;
    // callbacks must not perform I/O or reenter the store. An empty callback
    // unregisters this instance. Capture external owners weakly.
    void SetCheckpointEraseCallback(
        std::function<void(const std::string &key, const std::string &identity)> callback);
    // Metadata-only safety gate used by the unified page ledger. Returning
    // false keeps a live checkpoint out of FIFO retirement; callbacks must not
    // perform I/O or reenter this store. Empty unregisters this cache instance.
    void SetCheckpointReclaimPredicate(CheckpointReclaimPredicate predicate);
    void SetCheckpointCorruptRetirePredicate(CheckpointCorruptRetirePredicate predicate);
    bool RetireCorruptCheckpoint(const std::string &key, const std::string &identity,
                                 const std::string &confirmedReason, std::string &error);
    // The Python/C API passes the startup identity through thread-local memory,
    // not environment variables (large manifests would exceed execve limits).
    static void SetStartupManifest(const std::string &manifest);
    static std::string StartupManifest();
    static void Require(bool condition, const char *message);
    static std::string Digest(const void *data, size_t size);
    static std::string Digest(const std::string &value);
    static bool IsDigest(const std::string &value);
    static RestoreFailureClass ClassifyRestoreFailure(const std::string &error);
    // [store-profile] 写盘路径诊断：返回自上次 reset 以来的哈希/DB 调用次数、
    // 累计字节数与分段耗时。仅用于诊断，不改变任何写入字节。
    static Json StoreProfile(bool reset);
    // §3 表格要求"ticket 独立计数、最终聚合"：本接口只返回**单调累计值**，
    // 从不清零全局计数；调用方自行与上一次快照做差，避免并行任务互相清零。
    static Json StoreProfileSnapshot();
    // §14：v4 存储层结构化诊断（对象打包/索引事务/目录同步耗时、重建分类计数、
    // 运行时 sqlite 同步档位）。纯只读诊断，不影响写入。
    // drain=true（默认）先做读侧收尾，保证计数与 live_* 覆盖已受理的提交；
    // 热路径（模型每检查点的 export-profile）传 false，避免把刚建立的后台重叠抵消。
    Json StorageProfile(bool drain = true) const;
    // v4 存储“已受理未持久化”的在途字节。
    uint64_t InFlightBytes() const;
    // Wait for every accepted v4 commit to reach a terminal result. The caller
    // supplies the bounded model-side drain deadline; this method never drops a
    // pending record.
    bool DrainPending(std::string &error);
    // §保最新（2026-10-01，2026-10-05 修正语义）：**只归还主机额度（会计），不丢弃作业**。
    // 选中最老的一个"已受理未持久化、且额度尚未归还"的作业，归还其 heldBytes 让最新检查点
    // 能被受理；该作业仍照常备菜、照常由 actor 发布（绝不删除已持久化数据）。v4 专属；
    // 成功回填 freedBytes（= 本次归还的字节数）并返回 true；无牺牲者时 error 为
    // v4_no_cancellable_pending。注意：归还额度后作业仍占内存，故 InFlightBytes() 会低于
    // 真实占用；差额由 Profile() 的 resident_quota_released_bytes 单独观测（不改进准入口径）。
    bool CancelOldestPending(uint64_t *freedBytes, std::string &error);
    static std::string PrefixKey(const std::vector<int> &tokens, int length,
                                 const std::string &media = "");
private:
    std::shared_ptr<Impl> impl;
};

// v4 manifest-tree store (SSD_CACHE_REWRITE_PLAN.md §10/§11).
//
// Objects and control records are appended to one v4 control log and the SQLite catalog is only a
// projection that can be rebuilt from that log. Nothing becomes visible until
// the COMMIT record is durable, and an unfinished tail is never published.
class DiskPrefixV4Store {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit DiskPrefixV4Store(std::unique_ptr<Impl> value);
public:
    struct Object {
        std::string hash;
        uint32_t kind = 1;   // prefix_v4::StorageObjectKind
        std::string bytes;
        // 只引用（reference-only）条目：对象已在库里（被更早的检查点写过），本次不重写字节，
        // 但必须把 (kind, bytesSize, hash) 登记进本检查点的 root 清单——否则清单不完整。
        uint64_t bytesSize = 0;
    };
    // Opens (creating if needed) <prefixCacheDir>/v4. The FORMAT file must be
    // the v4 text; a missing catalog is rebuilt from the control log, a
    // damaged one is reported rather than trusted.
    static std::unique_ptr<DiskPrefixV4Store> Open(const std::string &prefixCacheDir,
                                                   uint64_t quotaBytes, std::string &error);
    // Same as Open, but reuses the store this process already has open for the
    // same canonical <prefixCacheDir>/v4 root. Plan §15 requires exactly one
    // CommitActor per canonical root per process, so two namespaces (target KV
    // state and encoder/image features) must share one actor, one SQLite
    // connection and one quota instead of racing on the same control log
    // (§22.4.3). A different quota fails closed with `v4_shared_quota_mismatch`.
    static std::shared_ptr<DiskPrefixV4Store> OpenShared(const std::string &prefixCacheDir,
                                                         uint64_t quotaBytes, std::string &error);
    ~DiskPrefixV4Store();
    DiskPrefixV4Store(const DiskPrefixV4Store &) = delete;
    DiskPrefixV4Store &operator=(const DiskPrefixV4Store &) = delete;

    // Returns by value: a reference to a temporary path string would dangle.
    std::string Root() const;
    // Writes the listed objects, their manifest list object, PREPARE and COMMIT
    // in that order, syncing each. Returns the publication sequence. Objects
    // that already exist are not rewritten.
    bool Commit(const std::string &key, const std::string &identity,
                const std::vector<Object> &objects, uint64_t *outSequence, std::string &error);
    // Resolves the live root of `key` and returns the exact object bytes.
    // 旧签名（outIdentity 出参）行为保持不变，仅供既有调用方/测试使用；内部委托为
    // “不过滤 identity”，等价于历史行为。
    bool Load(const std::string &key, std::string &identity,
              std::vector<std::string> &objectBytes, std::string &error);
    // S2-4：v4 查询走 SQL 过滤，不退化成“全表遍历 + 逐条
    // Json::parse”）：带期望 identity 的加载重载。旧签名只按 key 取“最新 sequence”的
    // root，同一 key 属于不同 identity 时会解析到别人的 root；新代码必须把 identity
    // 传下来（ReadSession::Load 的 id）。
    bool Load(const std::string &key, const std::string &expectedIdentity,
              std::vector<std::string> &objectBytes, std::string &error);
    // A2：在带 identity 的重载上多回传**每个对象的摘要**
    // （与 objectBytes 一一对应、同序）。调用方（ReadSession::Load）本来就按摘要建 map，
    // 这样就不必对同一份字节再算一遍 SHA-256（T0 实测省 1525 ms）。
    // 摘要是「已验证结论」的搬运：ObjectBytes 出参前已校验 Digest(body)==hash，
    // 本重载只把同一个 hash 原样带上，不引入新的信任假设；一一对应由同序 push 保证，
    // 并有 size 相等的兜底（不符即 fail-closed，报 v4_object_hash_count_mismatch）。
    bool Load(const std::string &key, const std::string &expectedIdentity,
              std::vector<std::string> &objectBytes, std::vector<std::string> &objectHashes,
              std::string &error, bool manifestOnly = false, bool publishedOnly = false,
               const DiskPrefixCache::ReadProgress &progress = {},
               DiskPrefixCache::Json *assembledState = nullptr);
    // S2-4：按 (identity,key) 点查该 live root 的描述符 JSON 原文，替代
    // ReadSession::Load 里的“遍历全部 live 检查点逐条 JSON parse”；缺行返回 false
    // （error=v4_root_missing）。
    bool Descriptor(const std::string &identity, const std::string &key,
                    std::string &metadata, std::string &error, bool publishedOnly = false) const;
    // Deletes the projection and rebuilds it from the control log, publishing
    // only complete PREPARE/COMMIT pairs (plan §10.4).
    bool RebuildIndex(std::string &error);
    // Reclaims space by rewriting a fresh log containing only reachable objects
    // and live roots, then atomically replacing the old one.
    bool Compact(std::string &error);
    // Opaque read lease: protects this checkpoint from FIFO retirement until
    // released. The token holds per-root and store lifecycle shared leases.
    std::shared_ptr<void> PinCheckpoint(const std::string &key,
                                      const std::string &identity, std::string &error,
                                      bool drain = true) const;
    // The subscription expires with the returned token or store lifetime.
    // Same callback restrictions as DiskPrefixCache::SetCheckpointEraseCallback.
    std::shared_ptr<void> ObserveCheckpointErases(
        std::function<void(const std::string &, const std::string &)> callback);
    std::shared_ptr<void> ObserveCheckpointReclaims(
        DiskPrefixCache::CheckpointReclaimPredicate predicate);
    std::shared_ptr<void> ObserveCheckpointCorruptRetires(
        DiskPrefixCache::CheckpointCorruptRetirePredicate predicate);
    bool RetireCorruptCheckpoint(const std::string &key, const std::string &identity,
                                 const std::string &confirmedReason, std::string &error);
    // Compact copies these exact existing objects even if no live root owns
    // them yet. The lease is cross-process and does not pin the entire store.
    class ObjectLease;
    std::shared_ptr<ObjectLease> BeginObjectLease(std::string &error) const;
    bool ExtendObjectLease(const std::shared_ptr<ObjectLease> &lease,
                           const std::vector<std::string> &hashes, std::string &error) const;
    uint64_t LiveCheckpoints() const;
    uint64_t LiveBytes() const;
    // S2-5a：control.log 的**真实物理占用字节数**（此前 CapacityStatus 拿 live 冒充
    // physical）。这是给 S2-5b 分段落盘预留的接口，本期只做单文件；只读、不做读侧收尾。
    uint64_t PhysicalBytes() const;
    // Checked read-only diagnostic path: stat failure must never become zero use.
    bool ObservePhysicalBytes(uint64_t &bytes, std::string &error) const;
    // P3 修复 B（2026-10-01）：已受理但未持久化的在途字节（保守上界；只读诊断/预算用）。
    uint64_t InFlightBytes() const;
    // §保最新（2026-10-01，2026-10-05 修正语义）：**只归还主机额度（会计），不丢弃作业**。
    // 1d（2026-10-03）之后不再有"取消=丢弃检查点"：选中的作业继续走完 PrepLoop → 唯一 actor
    // → 发布，块链因此保持连续（旧语义"取消尚未开工的队首作业"会把检查点挖掉，已在 T1c 实证
    // 为 parent-adopt 失败的根因）。本接口绝不删除已持久化数据、也绝不丢检查点。
    // 成功回填 freedBytes（= 归还的字节数）并返回 true；无牺牲者时 error 为
    // v4_no_cancellable_pending（调用方须保持 fail-closed）。
    // 观测（2026-10-05 复核 R6）：归还额度后作业仍占内存 ⇒ InFlightBytes() 低于真实占用，
    // 差额见 Profile() 的 resident_quota_released_bytes；准入语义不变（取消本就是为了让最新
    // 检查点能进，不因此收紧）。
    bool CancelOldestPending(uint64_t *freedBytes, std::string &error);
    // §14：提交段分项计时入口。
    void AddMetadataTiming(uint64_t refWalkNs, uint64_t encodeNs, uint64_t hashNs, size_t bytes);
    void AddPrecheckTiming(uint64_t scanNs);
    void AddActorWaitTiming(uint64_t waitNs);
    // §14：结构化诊断（JSON 文本）——控制日志写入/同步、目录同步、打包与索引事务
    // 耗时，以及重建分类计数。只读计数，不影响任何写入路径。
    // drain=true（默认）先做读侧收尾，保证计数与 live_* 覆盖已受理的提交；热路径传 false。
    std::string Profile(bool drain = true) const;

    // Checkpoint descriptor projection. `metadata` is the canonical JSON the
    // caller supplied (kind/length/family/dtype/provenance/prefix/created/state);
    // it is also appended as a METADATA record so RebuildIndex can restore it.
    struct CheckpointRow {
        std::string key, identity, metadata;
        uint64_t sequence = 0;
        // S2-4：checkpoints 表新增的**投影索引列**。权威源仍是 control.log 的
        // METADATA 记录；这些列只是让查询走 SQL/索引，不再全表解析 metadata JSON。
        // 旧 catalog 升级（catalog_schema 闸门）期间可能尚未回填，但不影响点查。
        std::string familyIdentity, kind, kvDtype, provenance, prefix;
        int length = 0;
        uint64_t createdNs = 0;
    };
    // S2-4：v4 查询改走 SQL（JOIN roots 限定 live）。
    std::vector<int> CheckpointLengths(const std::string &familyIdentity, int maxLength,
        const std::string &kind, bool publishedOnly = false) const;
    std::vector<CheckpointRow> ListCheckpoints(const std::string &exactIdentity,
        const std::string &familyIdentity, int maxLength, const std::string &kind,
        const std::string &prefix) const;
    // 见 DiskPrefixCache::MatchPrefixes 的说明。走 `checkpoints_prefix(kind,family,prefix,length)` 索引，
    // 一条 SQL 用行值 IN（`(prefix,length) IN (VALUES (?,?),…)`）问完全部候选。
    // **要求 SQLite ≥ 3.15**（行值支持）；本项目 CMake 用系统 sqlite（实测 CLI 3.53.2）。
    std::vector<CheckpointRow> MatchPrefixes(const std::string &familyIdentity,
        const std::vector<std::pair<int, std::string>> &prefixes,
        const std::string &kind, bool publishedOnly = false) const;
    // -----------------------------------------------------------------------
    // Route B（2026-10-02）：按块索引/恢复。
    // 每个 256-token 块一条 `blocks` 行（catalog 投影，重建时从控制日志里的块对象复原）：
    //   blocks(chain_key PK, identity, key, block_index, token_start, token_end,
    //          parent, object_hash, bytes)
    // chain_key = H(parent, 块内 token 数, 块内容摘要)，内容寻址且与前缀一一对应。
    struct BlockRow {
        std::string chainKey, parent, objectHash, identity, key;
        uint64_t index = 0, tokenStart = 0, tokenEnd = 0, bytes = 0;
    };
    // 从链尾按 parent 反向走 count 步，返回**升序**（第 0 块在前）的块行。
    // fail-closed：链尾块缺失返回 v4_block_missing；需要父块而父块缺失返回
    // v4_block_parent_missing（绝不静默拼出一条错误链）。
    bool ChainBlocks(const std::string &chainTip, uint64_t count,
                     std::vector<BlockRow> &out, std::string &error, bool publishedOnly = false) const;
    bool CommitCheckpoint(const std::string &key, const std::string &identity,
                          const std::vector<Object> &objects, const std::string &metadata,
                          uint64_t *outSequence, std::string &error);
    // §③：rvalue 版本——对象字节直接**移交**给待写队列，不再拷一份（模型写路径用）。
    bool CommitCheckpoint(const std::string &key, const std::string &identity,
                          std::vector<Object> &&objects, const std::string &metadata,
                          uint64_t *outSequence, std::string &error);
    // §④ 两阶段提交：① 先把对象+控制记录落盘（不依赖描述符）；② 再补描述符与索引投影。
    // 调用方可以在 ① 与 ② 之间做纯 CPU 工作（例如状态文字序列化），从而与写盘重叠。
    bool CommitObjects(const std::string &key, const std::string &identity,
                       std::vector<Object> &&objects, uint64_t *outSequence, std::string &error);
    bool CommitDescriptor(const std::string &key, const std::string &identity,
                          const std::string &descriptor, std::string &error);
    // §④/§⑥ 异步提交：Submit 只按受理顺序入 FIFO（不等待、不直接投盘）；Reap 等队首并
    // **原样上报**错误。语义不变：Submit 返回 true 只代表"已受理"，不代表"已持久化"；
    // 关闭/退出前必须 Reap（析构会自动收尾）。
    bool SubmitCheckpoint(const std::string &key, const std::string &identity,
                          std::vector<Object> &&objects, const std::string &metadata,
                          std::string &error);
    bool ReapCheckpoint(uint64_t *outSequence, std::string &error);
    // §⑥ 后台备菜流水线（唯一备菜线程 + FIFO，深度上限 2）：请求线程只移交
    // "状态 Json + 本次暂存分块"就返回"已受理"；状态正文序列化(MetadataText)、
    // 引用遍历(CollectRefs)、摘要(Digest)、描述符构造、预检扫描、对象装配与
    // root 清单登记（含只引用对象）全部在后台备菜线程上按受理顺序完成，再投递给
    // 唯一的写盘 actor。磁盘格式、记录布局、清单语义一律不变。
    // defaultFamily：状态 Json 缺 family_identity 时使用的家族名。
    // 语义：true 只代表"已受理"，持久化结果在 ReapCheckpoint / 读侧收尾 / 关服收尾上报。
    bool SubmitStaged(const std::string &key, const std::string &identity,
                      const std::string &defaultFamily, DiskPrefixCache::Json state,
                      bool keepEquivalent,
                      std::vector<std::pair<std::string, std::string>> &&staged,
                      std::string &error);
    bool SubmitStaged(const std::string &key, const std::string &identity,
                      const std::string &defaultFamily, DiskPrefixCache::Json state,
                      bool keepEquivalent,
                      std::vector<std::pair<std::string, std::string>> &&staged,
                      std::string &error, std::function<void(bool)> completion);
    bool SubmitStaged(const std::string &key, const std::string &identity,
                      const std::string &defaultFamily, DiskPrefixCache::Json state,
                      bool keepEquivalent,
                      std::vector<std::pair<std::string, std::string>> &&staged,
                      std::string &error, std::function<void(bool)> completion,
                      std::shared_ptr<ObjectLease> objectLease,
                      const DiskPrefixCache::CommitIdentity &commitIdentity = {});
    // 诊断用（只读）：某个对象摘要是否已在对象表里。用于区分
    // "对象从未落盘" 与 "对象在库里、只是本次 Load 的清单没带它"。
    bool HasObject(const std::string &hash) const;
    // 启动自检（env FASTLLM_PREFIX_CACHE_SELFCHECK 门控）：逐个 live checkpoint 试读对象，
    // 打印"可恢复/不可恢复"统计与失败原因，让"写了但恢复不了"的检查点直接可见。
    void SelfCheck();
    // 自检的**清单完整性**部分（P2 收口，2026-09-30）：逐个 live root 校验"状态对象本体 +
    // 它按写方同一口径引用的每个对象"都能在该 root 的清单里找到，返回**缺失条目总数**
    // （0 = 通过）；details 里给出前若干条 root 的缺失说明。抽成独立接口是为了能写反例断言
    // ——历史上"清单漏登记被去重对象"正是只靠 SelfCheck 的"能不能 Load"布尔值溜过去的。
    size_t SelfCheckManifest(std::string &details);
    // S6（2026-10-05 复核）：功能性可恢复校验——与 SelfCheckManifest（"对象在不在"）互补，
    // 本函数看"就算对象都在，读回来能不能真的恢复"。判据三条（都取自写方落进状态对象的字段，
    // store 级即可判定）：
    //   ① KV 覆盖连续：Route B 块链必须从 token 0 起首尾相接，链尾 token_end == 检查点 length；
    //   ② linear 层齐全：layout.linear 标 true 的每一层在状态里恰好有一条对应条目（不多不少）；
    //   ③ spec tokens == length：若检查点确实记录了 spec 状态（speculative.kind != none），
    //      其 kind 必须与 layout.speculative 一致、tokens 必须 == length。dflash2 与 mtp
    //      **同等对待**（两者 spec 状态都随 length 增长，不能只查 dflash2）。kind==none 不判红
    //      （那是"本次 generation config 不需要 spec"时写下的合法检查点，由 Import 按请求再校验）。
    // 只对"状态对象带 layout 字段"的检查点判定；老式/合成状态对象（无 layout）跳过、不误报。
    // 返回不合格检查点数（0 = 通过）；details 给出前若干条判据失败说明。只读。
    size_t SelfCheckFunctional(std::string &details);
    // Live checkpoints only: a root retired by an ERASE record is excluded.
    std::vector<CheckpointRow> Checkpoints() const;
private:
    // 内部实现：只允许在 catalog actor 线程上执行（见 §11.1 单写者）。
    bool CommitObjectsLocked(const std::string &key, const std::string &identity,
                             std::vector<Object> &objects, uint64_t *outSequence,
                             std::string &error);
    bool CommitDescriptorLocked(const std::string &key, const std::string &identity,
                                const std::string &descriptor, std::string &error);
    bool CommitLocked(const std::string &key, const std::string &identity,
                      std::vector<Object> &objects, uint64_t *outSequence,
                      std::string &error);
    bool CommitCheckpointLocked(const std::string &key, const std::string &identity,
                                std::vector<Object> &objects, const std::string &metadata,
                                uint64_t *outSequence, std::string &error);
    // §④ 提交流水线内部实现（调用方必须已持有 Impl::commitPipeline）。
    bool SubmitPipeline(const std::string &key, const std::string &identity,
                        std::vector<Object> &&objects, const std::string &metadata,
                        std::string &error);
    bool ReapPipeline(uint64_t *outSequence, std::string &error);
    bool PutMetadataLocked(const std::string &key, const std::string &value, std::string &error);
    bool RebuildIndexLocked(std::string &error);
    // S2-4：带身份过滤的加载实现体；两个 public Load 重载都委托到它。
    bool LoadWithIdentity(const std::string &key, const std::string &expectedIdentity,
                          std::string &outIdentity, std::vector<std::string> &objectBytes,
                          std::vector<std::string> *outHashes, std::string &error,
                          bool manifestOnly = false, bool publishedOnly = false,
                           const DiskPrefixCache::ReadProgress &progress = {},
                           DiskPrefixCache::Json *assembledState = nullptr);
    // 持有 root 租约后的实现体（§11.2 锁序：先 lifecycle，再 append）。
    bool CommitBareLocked(const std::string &key, const std::string &identity,
                          std::vector<Object> &objects, uint64_t *outSequence,
                          std::string &error,
                          const std::string *checkpointMetadata = nullptr);
    bool RebuildIndexBareLocked(std::string &error);
    bool CompactLocked(std::string &error);
    bool PreparePhysicalCapacityLocked(const std::string &key, const std::string &identity,
        const std::vector<Object> &objects, const std::string *metadata,
        std::string &error);
public:
    // Durable key/value metadata (startup identity, cost profile, ...).
    bool PutMetadata(const std::string &key, const std::string &value, std::string &error);
    bool GetMetadata(const std::string &key, std::string &value, std::string &error);
};
} // namespace fastllm
#endif
#endif
