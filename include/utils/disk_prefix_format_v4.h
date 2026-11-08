#ifndef FASTLLM_DISK_PREFIX_FORMAT_V4_H
#define FASTLLM_DISK_PREFIX_FORMAT_V4_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// v4 record/container encoding (SSD_CACHE_REWRITE_PLAN.md §10.1/§10.2).
//
// This module only encodes and decodes bytes: it never opens a directory, never
// allocates from an untrusted length and never touches SQLite. Every field is
// fixed-width little-endian and every length is validated before any allocation,
// so a truncated or hostile record is rejected instead of trusted.
namespace fastllm {
namespace prefix_v4 {

// Distinct from the v3 magics ("FLKV3P01" / "FLKV3L01"): a v3 reader must not
// mistake a v4 container for its own, even beside it.
extern const char kContainerMagic[8];
extern const char kRecordMagic[8];
constexpr const char *kFormatText = "fastllm-prefix-v4 manifest-tree-1\n";

constexpr uint64_t kAlignment = 4096;
constexpr uint64_t kContainerHeaderBytes = kAlignment;
constexpr uint64_t kRecordHeaderBytes = 128;
// Initial bounds from the plan. A smaller file limit must shrink the batch
// further instead of exceeding these and fixing up afterwards.
constexpr uint64_t kMaxPackedPayloadBytes = 16ull << 20;
constexpr uint64_t kMaxObjectBytes = 8ull << 20;
constexpr uint64_t kMaxNodeBytes = 1ull << 20;
constexpr uint32_t kMaxManifestFanout = 64;
constexpr uint32_t kMaxManifestDepth = 8;

enum StorageObjectKind : uint32_t {
    kObjectData = 1,
    kObjectLayout = 2,
    kObjectKvNode = 3,
    kObjectStateNode = 4,
    // Route B（2026-10-02）：一个 256 token 块一条独立对象（内容寻址），状态对象只保存
    // 链尾键 + 块数，不再保存"从开头到本点"的全部 KV 分块描述。
    kObjectBlock = 5,
};

// ---------------------------------------------------------------------------
// Route B: block-chained KV layout (对标 vLLM 16-token 链式哈希 / SGLang page 哈希 /
// LMCache 256-token 一 chunk 一文件)。
//
// 每个块一条独立对象，块键链式：BlockKey_i = H(BlockKey_{i-1}, tokens_i, extra_i)。
// 原始 token ID **绝不落盘**（§17.1），因此这里的 "tokens_i" 用该块 KV 分块描述的内容
// 摘要（它是这些 token 的 KV 的确定性函数）+ 块内 token 数共同承诺；首块父键用固定种子。
constexpr uint32_t kDefaultBlockTokens = 256;
constexpr uint64_t kMaxBlockCount = 1ull << 22;   // 4M 块上限：防越界/防坏链无限循环
// 块键的新域串；与 manifest-tree 的摘要域完全分开，互不可重解释。
constexpr const char *kBlockChainDomain = "fastllm-block-chain-v1";
// env FASTLLM_PREFIX_CACHE_BLOCK_TOKENS 可覆盖；默认 256（= 默认检查点间隔）。
uint32_t BlockTokens();
// 首块父键（固定种子），使第一块的键也是"前缀决定"的。
std::string BlockGenesisKey();
// H(parentKey, 块内 token 数, 块内容摘要)——preimage 为
// "<parentKey>|<tokens>|<contentDigest>"，域为 kBlockChainDomain。
std::string BlockChainKey(const std::string &parentKey, uint64_t tokens,
                          const std::string &contentDigest);

enum RecordType : uint32_t {
    kRecordObjectBatch = 1,
    kRecordManifestPrepare = 2,
    kRecordManifestCommit = 3,
    kRecordErase = 4,
    kRecordMetadata = 5,
};

// The object table alone decides an OBJECT_BATCH digest: the digest binds what
// objects should exist, their kind/length/offset and their content hashes; it
// deliberately does not re-scan every large payload.
enum CoverageMode : uint32_t {
    kCoverageObjectTableOnly = 1,
    kCoverageFullRecord = 2,
};

// No critical flag is defined yet, so every non-zero bit is unknown and rejected
// rather than ignored.
constexpr uint32_t kKnownRecordFlags = 0;

uint32_t Crc32(const void *data, size_t size);
// Domain-separated digest so a LAYOUT node hash can never be reinterpreted as a
// DATA hash or vice versa.
std::string DomainDigest(const char *domain, const void *data, size_t size);
std::string DomainDigest(const char *domain, const std::string &value);
bool IsDigest(const std::string &value);
uint64_t AlignUp(uint64_t value, std::string &error);

// ---------------------------------------------------------------------------
// S4（2026-10-04）：有界（窗口化）导出的窗口切分。
//
// 背景：导出曾经一次性 Pick 整段 [issued, length) 的页并把整段字节与 GPU pin 预算比较；
// 区间随上下文增长 ⇒ 必然超预算 ⇒ write_skipped，且 issued 不前移 ⇒ 下一次区间更大 ⇒ 连丢。
// 这里把"一段连续页区间"切成若干窗口，使**每个窗口的页字节 ≤ 预算**，导出按窗口
// pick→拷贝→放钉，峰值 pin 与上下文长度无关。
//
// 纯函数：只做整数除法与边界计算，不碰 CUDA/文件/全局状态，可无 GPU 单测。
//   pageCount     : 待切分页数，区间为 [0, pageCount)
//   bytesPerPage  : 每页字节数
//   budgetBytes   : 每窗口字节上界（调用方通常传 pin 预算的一半）；0 表示该约束无效
//   maxWindowBytes: 每窗口字节硬上界（如 BufferBytes=8 MiB）；0 表示该约束无效
// 返回窗口 beginPage/endPage（半开区间 [begin,end)，页为单位），满足：
//   · 升序、严格连续、无洞无重叠，恰好覆盖 [0, pageCount)；
//   · 每个窗口至少 1 页（预算被压到不足一页时退化为逐页，而不是死循环）；
//   · 在 bytesPerPage>0 且预算有意义时，每个窗口字节 ≤ min(budgetBytes, maxWindowBytes)。
// 0 输入（pageCount==0）返回空。
struct ExportWindow {
    uint64_t beginPage = 0;
    uint64_t endPage = 0;   // exclusive
};
std::vector<ExportWindow> SplitExportWindows(uint64_t pageCount, uint64_t bytesPerPage,
                                             uint64_t budgetBytes, uint64_t maxWindowBytes);

struct RecordHeader {
    uint32_t type = 0;
    uint32_t flags = 0;
    uint32_t headerBytes = uint32_t(kRecordHeaderBytes);
    uint32_t digestAlgo = 1;                  // 1 = SHA-256
    uint32_t coverageMode = 0;
    uint64_t payloadBytes = 0;
    uint64_t physicalBytes = 0;
    uint64_t sequence = 0;
    std::string digest;                       // 64 lowercase hex chars
};

std::vector<char> EncodeRecordHeader(const RecordHeader &header);
bool DecodeRecordHeader(const char *data, size_t size, RecordHeader &out, std::string &error);

struct ObjectEntry {
    uint32_t kind = 0;
    uint64_t bytes = 0;
    uint64_t offset = 0;   // offset inside this record's payload region
    std::string hash;
};

// Table layout: u32 count, u32 reserved, u64 payloadRegionBytes, then entries
// (u32 kind, u32 flags, u64 bytes, u64 offset, 64-byte hex hash).
std::vector<char> EncodeObjectTable(const std::vector<ObjectEntry> &entries,
                                    uint64_t payloadRegionBytes);
bool DecodeObjectTable(const char *data, size_t size, std::vector<ObjectEntry> &out,
                       uint64_t &payloadRegionBytes, std::string &error);

struct BundleRef {
    uint32_t rank = 0;
    uint32_t layoutId = 0;
    uint64_t tokenStart = 0;
    uint64_t tokenEnd = 0;   // exclusive
    uint64_t bytes = 0;
    std::string hash;
};

struct ChildRef {
    uint64_t tokenStart = 0;
    uint64_t tokenEnd = 0;   // exclusive
    uint64_t nodeBytes = 0;
    std::string hash;
};

struct ManifestNode {
    bool leaf = true;
    uint64_t tokenStart = 0;
    uint64_t tokenEnd = 0;   // exclusive
    std::vector<BundleRef> bundles;
    std::vector<ChildRef> children;
};

// Leaf layout: u8 type, u8 reserved[3], u32 count, u64 tokenStart, u64 tokenEnd,
//              then count * (u32 rank, u32 layoutId, u64 tokenStart, u64 tokenEnd,
//              u64 bytes, 64-byte hash).
// Inner layout: same prefix, then count * (u64 tokenStart, u64 tokenEnd,
//              u64 nodeBytes, 64-byte hash).
std::vector<char> EncodeManifestNode(const ManifestNode &node);
// `parentRange` (when hasRange) and `depth` are supplied by the caller walking
// the tree, so a node cannot claim a token range its parent never covered.
bool DecodeManifestNode(const char *data, size_t size, uint64_t depth,
                        bool hasParentRange, uint64_t parentStart, uint64_t parentEnd,
                        ManifestNode &out, std::string &error);

// ---------------------------------------------------------------------------
// OBJECT_BATCH payload: object table followed by the payload region it
// describes. `payload` is the concatenation of the objects in offset order.
std::vector<char> EncodeObjectBatchPayload(const std::vector<ObjectEntry> &entries,
                                           const std::string &payload);
bool DecodeObjectBatchPayload(const char *data, size_t size,
                              std::vector<ObjectEntry> &entries, std::string &payload,
                              std::string &error);
// A1b：与 DecodeObjectBatchPayload **校验与错误串完全一致**，但
// **不拷贝** payload 区域 —— 只回填 entries 与区域在 data 中的偏移
// [regionStart, regionStart+regionBytes)。供读路径"按偏移直接取目标对象"使用，避免
// 每取一个对象就把整条记录 region 拷一遍（T0 实测：520 个对象 ⇒ 520 次 region 拷贝
// ≈ 3.9 GB memcpy + 约 98 万次缺页）。
bool DecodeObjectBatchPayloadView(const char *data, size_t size,
                                  std::vector<ObjectEntry> &entries,
                                  uint64_t &regionStart, uint64_t &regionBytes,
                                  std::string &error);

// ---------------------------------------------------------------------------
// Control-log records. Each has a fixed canonical layout; every length is
// validated before allocation, and the record digest covers the full payload
// with a type-specific domain.
constexpr uint32_t kMaxKeyBytes = 4096;
// A metadata record's payload must fit kMaxPackedPayloadBytes, so the value
// bound is the record bound minus the fixed prefix and the key cap. The writer
// and the reader share this constant: previously the writer accepted values it
// could not decode back (`v4_record_length`), which blocked every full rebuild.
constexpr uint32_t kMaxMetadataValueBytes = (16u << 20) - 8192u;

struct PrepareRecord {
    uint64_t sequence = 0;
    std::string key, identity, rootHash;
    uint64_t firstBatch = 0, lastBatch = 0;   // dependency batch sequence range
};
struct CommitRecord {
    uint64_t sequence = 0;
    uint64_t prepareOffset = 0, prepareBytes = 0;
    std::string prepareDigest, rootHash;
};
struct EraseRecord {
    uint64_t sequence = 0;
    std::string key, identity;
};
struct MetadataRecord {
    uint64_t sequence = 0;
    std::string key, value;
};

std::vector<char> EncodePrepare(const PrepareRecord &record);
bool DecodePrepare(const char *data, size_t size, PrepareRecord &out, std::string &error);
std::vector<char> EncodeCommit(const CommitRecord &record);
bool DecodeCommit(const char *data, size_t size, CommitRecord &out, std::string &error);
std::vector<char> EncodeErase(const EraseRecord &record);
bool DecodeErase(const char *data, size_t size, EraseRecord &out, std::string &error);
std::vector<char> EncodeMetadata(const MetadataRecord &record);
bool DecodeMetadata(const char *data, size_t size, MetadataRecord &out, std::string &error);

// Digest domain used for the record digest of each type.
const char *RecordDomain(uint32_t type);

// Public wrapper so the store can reproduce exactly the digest that
// AppendRecordToFile wrote (OBJECT_BATCH binds only the object table).
std::string RecordDigest(uint32_t type, const std::string &payload);

// ---------------------------------------------------------------------------
// Manifest list object: the canonical list of objects that make up one
// checkpoint. It is stored as an ordinary object, so its own hash is the
// checkpoint root hash and can be referenced by PREPARE/COMMIT.
struct ObjectListEntry {
    uint32_t kind = 0;
    uint64_t bytes = 0;
    std::string hash;
};
// Layout: u32 count, u32 reserved, then count * (u32 kind, u32 reserved,
// u64 bytes, 64-byte hex hash).
std::vector<char> EncodeObjectList(const std::vector<ObjectListEntry> &entries);
bool DecodeObjectList(const char *data, size_t size, std::vector<ObjectListEntry> &out,
                      std::string &error);
constexpr uint32_t kMaxObjectListEntries = 4096;

// ---------------------------------------------------------------------------
// Incremental manifest tree (plan §9): fixed fan-out 64, immutable nodes, and
// append-only path copying. Appending a delta rewrites only the rightmost leaf
// and its ancestors; every untouched subtree keeps its old hash and stays
// shared. Nodes are content-addressed, so the root hash names the whole tree.
struct TreeNode {
    std::string hash;
    std::string bytes;
    uint64_t tokenStart = 0;
    uint64_t tokenEnd = 0;   // exclusive
    bool leaf = false;
};
// Fetches a previously stored node by hash. Returning false means "not
// available", which is an error rather than "empty node".
using NodeResolver = std::function<bool(const std::string &hash, std::string &bytes)>;

// `existingRoot` empty means a brand-new tree (existingTokens must then be 0).
// `delta` must be contiguous and must start exactly at existingTokens, otherwise
// a gap or an overlap would silently change the meaning of a token position.
// Returns only the nodes that must be written; unchanged nodes are not repeated.
bool AppendManifestDelta(const std::string &existingRoot, uint64_t existingTokens,
                         const std::vector<BundleRef> &delta, const NodeResolver &resolver,
                         std::vector<TreeNode> &outNodes, std::string &outRootHash,
                         uint64_t &outTokens, std::string &error);

// Walks the tree and returns every bundle in token order. Rejects a cycle, a
// repeated node, a depth overflow and any range that escapes its parent.
bool FlattenManifest(const std::string &root, const NodeResolver &resolver,
                     std::vector<BundleRef> &out, std::string &error);

// ---------------------------------------------------------------------------
// Crash classification for a control log (plan §10.4). A record is only
// published when its bytes are complete, its digest matches and its sequence
// increases; anything else is either an unfinished tail or a corrupt log.
enum class LogClassification {
    Complete,          // every byte accounted for by published records
    UncommittedTail,   // short/unfinished trailing record: publish nothing from it
    CorruptControl,    // looks complete but is inconsistent: fail closed
};

struct ScannedRecord {
    RecordHeader header;
    uint64_t offset = 0;   // byte offset of this record in the log
};

// `out` receives only records that are individually valid and sequence
// monotonic. The return value is the classification of the whole buffer.
LogClassification ScanControlLog(const char *data, size_t size,
                                 std::vector<ScannedRecord> &out, std::string &error);

// Streaming control-log reader used by the v4 index rebuild. Memory stays
// bounded by one record (at most kMaxPackedPayloadBytes) plus the per-record
// metadata, so recovery does not scale with the control-log size.
class ControlLogReader {
public:
    ControlLogReader() = default;
    ~ControlLogReader();
    ControlLogReader(const ControlLogReader &) = delete;
    ControlLogReader &operator=(const ControlLogReader &) = delete;

    // A missing control log is an empty log, not an error.
    bool Open(const std::string &path, std::string &error);
    uint64_t Size() const { return size_; }
    bool ReadAt(uint64_t offset, void *out, size_t bytes, std::string &error) const;
    bool ReadPayload(const ScannedRecord &record, std::string &out, std::string &error) const;
    // Same classification semantics as ScanControlLog, read record by record.
    LogClassification Scan(std::vector<ScannedRecord> &out, std::string &error);

private:
    int fd_ = -1;
    uint64_t size_ = 0;
};

// Appends one fully formed record (header + payload, physical length aligned)
// and fdatasyncs before returning. The caller must only publish after this
// returns true.
bool AppendRecordToFile(const std::string &path, uint32_t type, uint64_t sequence,
                        const std::string &payload, uint64_t *outOffset, std::string &error);

// Reads a whole file. Returns false when it cannot be read; an empty file yields
// an empty buffer and true.
bool ReadWholeFile(const std::string &path, std::string &out, std::string &error);

// §14 计时：控制日志追加路径的写入与同步分开计量（单调累计，供诊断差分）。
struct JournalStats {
    uint64_t calls = 0, bytes = 0, writeNs = 0, syncNs = 0;
    uint64_t dirSyncs = 0, dirSyncNs = 0;   // §14：目录同步单独计量
    // §14 要求 data_sync 与 control_sync 分开：对象批（DATA）落盘同步属前者，
    // PREPARE/COMMIT/METADATA 等控制记录属后者。
    uint64_t dataSyncs = 0, dataSyncNs = 0, controlSyncs = 0, controlSyncNs = 0;
    // §14：record_hash_bytes/ms —— 记录级摘要的字节数与耗时。
    uint64_t recordHashBytes = 0, recordHashNs = 0;
};
JournalStats JournalStatsSnapshot();

} // namespace prefix_v4
} // namespace fastllm
#endif
