// DeepSeek-V4.1 量化 KV 缓存（--kv_cache_dtype fp8_e4m3 / fp4_e2m1）的算子级回归。
//
// 覆盖三件事：
//   1. 行布局：三种缓存行的字节宽度，以及按真实 config 几何折算出的每 token 字节数
//      （BF16 3200 / FP8 1650 / FP4 890）；
//   2. 无损性：压缩 KV 与 indexer key 在写进 cache 前已由 DeepSeekV41RotaryQuant 伪量化
//      到 FP4 网格上（压缩 KV 是 E2M1 + 每 16 个一组的 E4M3 scale，indexer key 是 E2M1 +
//      每 32 个一组的 UE8M0 scale）。DeepSeekV41QuantizeKV 用同一份 scale 推导，因此
//      「伪量化 -> 存储 -> 解码」必须逐 bit（含 0 的符号）回到伪量化后的值。这里的解码器
//      是按文档独立写的，不复用实现里的代码；
//   3. 读路径：SparseAttention 与 IndexerScore 读量化缓存行的结果，与读同样数值的 BF16
//      行逐 bit 相同（CPU 参考实现与 CUDA kernel 各测一遍，CUDA 还要测 legacy 回退开关）。

#include "fastllm.h"
#include "executor.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

#ifdef USE_CUDA
// 融合 kernel 直接调用（模型里也是这样用的：拒绝时退回 HcApplyPre + RMSNorm 两步）
extern "C" bool FastllmCudaDeepSeekV41HcPreNorm(const fastllm::Data &x, const fastllm::Data &pre,
                                     fastllm::Data &normWeight, float eps, fastllm::Data &output);
#endif

static Executor &Exec() {
    return *((Executor*)GetExecutor());
}

static bool gHasCuda = false;
static int gChecks = 0;

static void Require(bool ok, const std::string &msg) {
    if (!ok) {
        throw std::runtime_error(msg);
    }
    gChecks++;
}

// ---------------- 小工具 ----------------

static Data MakeBf16(const std::vector<int> &dims, const std::vector<float> &values) {
    Data d(DataType::BFLOAT16, dims);
    d.Allocate();
    uint16_t *p = (uint16_t*)d.cpuData;
    for (size_t i = 0; i < values.size(); i++) {
        p[i] = Float32ToBFloat16RNEBits(values[i]);
    }
    return d;
}

static std::vector<float> ReadFloats(const Data &data) {
    Data cpu;
    cpu.CopyFrom(data);
    cpu.ToDevice(DataDevice::CPU);
    std::vector<float> out(cpu.Count(0));
    for (size_t i = 0; i < out.size(); i++) {
        if (cpu.dataType == DataType::FLOAT32) {
            out[i] = ((const float*)cpu.cpuData)[i];
        } else if (cpu.dataType == DataType::BFLOAT16) {
            out[i] = BFloat16BitsToFloat32(((const uint16_t*)cpu.cpuData)[i]);
        } else {
            throw std::runtime_error("ReadFloats: unexpected dtype");
        }
    }
    return out;
}

static std::vector<uint8_t> ReadBytes(const Data &data) {
    Data cpu;
    cpu.CopyFrom(data);
    cpu.ToDevice(DataDevice::CPU);
    std::vector<uint8_t> out(cpu.Count(0));
    memcpy(out.data(), cpu.cpuData, out.size());
    return out;
}

static std::vector<int32_t> ReadInts(const Data &data) {
    Data cpu;
    cpu.CopyFrom(data);
    cpu.ToDevice(DataDevice::CPU);
    std::vector<int32_t> out(cpu.Count(0));
    memcpy(out.data(), cpu.cpuData, out.size() * sizeof(int32_t));
    return out;
}

static void ToDev(Data &d, bool cuda) {
    d.ToDevice(cuda ? DataDevice::CUDA : DataDevice::CPU);
    d.Allocate(false);
}

static bool BitEqual(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

// ---------------- 1. 行布局与每 token 字节数 ----------------

static long long RowBytes(int dim, int quantMode, int blockSize) {
    if (quantMode <= 0) {
        return (long long)dim * 2;
    }
    return (quantMode == 1 ? dim : dim / 2) + dim / blockSize;
}

static void CheckLayout() {
    Require(RowBytes(512, 1, 32) == 528, "FP8 压缩/滑窗行应为 528 字节");
    Require(RowBytes(512, 3, 16) == 288, "FP4 压缩 KV 行应为 288 字节");
    Require(RowBytes(128, 2, 32) == 68, "FP4 indexer key 行应为 68 字节");
    Require(RowBytes(128, 1, 32) == 132, "FP8 indexer key 行应为 132 字节");
    // 三种行宽两两不同，读路径才能靠行宽自动识别布局
    Require(RowBytes(512, 1, 32) != RowBytes(512, 3, 16) && RowBytes(512, 3, 16) != RowBytes(512, 2, 32),
            "dim=512 的三种行宽应互不相同");
    Require(RowBytes(128, 1, 32) != RowBytes(128, 3, 16) && RowBytes(128, 3, 16) != RowBytes(128, 2, 32),
            "dim=128 的三种行宽应互不相同");

    // 真实 config：kv_source_layer_ids = [2, 8, 14, 20]，其中 2/8/14 的 compress_ratio 是 2，
    // 20 是 1；四层都是 index source。head_dim = 512，index_head_dim = 128。
    const int ratios[4] = {2, 2, 2, 1};
    struct Tier { const char *name; int cmpMode, cmpBlock, idxMode, idxBlock; long long expect; };
    const Tier tiers[3] = {
        {"bf16",     0, 32, 0, 32, 3200},
        {"fp8_e4m3", 1, 32, 1, 32, 1650},
        {"fp4_e2m1", 3, 16, 2, 32,  890},
    };
    for (const Tier &t : tiers) {
        long long total = 0;
        for (int i = 0; i < 4; i++) {
            total += RowBytes(512, t.cmpMode, t.cmpBlock) / ratios[i];
            total += RowBytes(128, t.idxMode, t.idxBlock) / ratios[i];
        }
        if (total != t.expect) {
            throw std::runtime_error(std::string("每 token KV 字节数不符：") + t.name + " = " +
                                     std::to_string(total) + "，期望 " + std::to_string(t.expect));
        }
        gChecks++;
        std::cout << "  " << t.name << ": " << total << " B/token\n";
    }
}

// ---------------- 2. QuantizeKV 的无损性 ----------------

// 按 docs/deepseek_v41.md 描述的行布局独立解码一个元素（不复用实现里的代码）
static float DecodeE4M3(uint8_t c) {
    int e = (c >> 3) & 0xF, m = c & 7;
    float v = (e == 0) ? std::ldexp((float)m, -9) : std::ldexp(1.0f + (float)m / 8.0f, e - 7);
    return (c & 0x80) ? -v : v;
}

static float DecodeRowElem(const uint8_t *row, int dim, int quantMode, int blockSize, int d) {
    const uint8_t *scales = row + (quantMode == 1 ? dim : dim / 2);
    const uint8_t sb = scales[d / blockSize];
    const float scale = (quantMode == 3) ? DecodeE4M3(sb) : std::ldexp(1.0f, (int)sb - 127);
    if (quantMode == 1) {
        return DecodeE4M3(row[d]) * scale;
    }
    static const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint8_t packed = row[d / 2];
    const int code = (d & 1) ? (packed >> 4) : (packed & 0xF);
    const float v = (code & 8) ? -grid[code & 7] : grid[code & 7];
    return v * scale;
}

// 先跑 RotaryQuant（inverse=false, 只做伪量化：ropeDim 之外的部分不受 RoPE 影响，
// 这里用 startPos=0 让 RoPE 退化为恒等），再 QuantizeKV，最后用 SparseAttention
// 的读路径把值拿回来比较。为了直接比较值，这里用一个只有一行候选的注意力做“解码器”。
struct QuantCase {
    const char *name;
    int dim;
    int quantMode;
    int blockSize;
};

static Data FakeQuantRows(const std::vector<float> &values, int rows, int dim,
                          int quantMode, int blockSize, bool cuda) {
    Data x = MakeBf16({1, rows, dim}, values);
    ToDev(x, cuda);
    Exec().Run("DeepSeekV41RotaryQuant", {{"input", &x}}, {{"ropeBase", 10000.0f}, {"ropeFactor", 1.0f}},
               {{"ropeDim", 64}, {"startPos", 0}, {"posStep", 0}, {"inverse", 0},
                {"originalSeqLen", 0}, {"betaFast", 32}, {"betaSlow", 1},
                {"quantMode", quantMode}, {"quantBlock", blockSize}});
    return x;
}

static Data QuantizeKV(const Data &input, int quantMode, int blockSize, bool cuda) {
    Data out;
    Exec().RunOnDevice(cuda ? "cuda" : "cpu", "DeepSeekV41QuantizeKV",
                       {{"input", (Data*)&input}, {"output", &out}}, {},
                       {{"quantMode", quantMode}, {"quantBlock", blockSize}});
    return out;
}

static void CheckQuantizeLossless(bool cuda) {
    const char *tag = cuda ? "cuda" : "cpu";
    std::mt19937 rng(20260911);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const QuantCase cases[3] = {
        {"压缩 KV  FP4 E2M1 + E4M3/16", 512, 3, 16},
        {"indexer FP4 E2M1 + UE8M0/32", 128, 2, 32},
        {"滑窗 KV  FP8 E4M3 + UE8M0/32", 512, 1, 32},
    };
    const int rows = 37;
    for (const QuantCase &c : cases) {
        std::vector<float> values(rows * c.dim);
        for (size_t i = 0; i < values.size(); i++) {
            // 混入不同量级，覆盖块内 amax 差异很大的情况
            values[i] = dist(rng) * std::ldexp(1.0f, (int)(i % 11) - 5);
        }
        // 伪量化一次（cache 里存的就是这个）
        Data q1 = FakeQuantRows(values, rows, c.dim, c.quantMode, c.blockSize, cuda);
        std::vector<float> after = ReadFloats(q1);

        // 存储
        Data packed = QuantizeKV(q1, c.quantMode, c.blockSize, cuda);
        const int rowBytes = (int)RowBytes(c.dim, c.quantMode, c.blockSize);
        Require(packed.dims.size() == 3 && packed.dataType == DataType::INT8 && packed.dims[2] == rowBytes,
                std::string(tag) + " " + c.name + "：缓存行宽不对");

        // 按文档描述的布局在测试里独立解码，要求逐 bit（含 0 的符号）回到伪量化后的值
        std::vector<uint8_t> bytes = ReadBytes(packed);
        for (int r = 0; r < rows; r++) {
            const uint8_t *row = bytes.data() + (size_t)r * rowBytes;
            for (int d = 0; d < c.dim; d++) {
                float got = DecodeRowElem(row, c.dim, c.quantMode, c.blockSize, d);
                float want = after[(size_t)r * c.dim + d];
                if (memcmp(&got, &want, sizeof(float)) != 0) {
                    throw std::runtime_error(std::string(tag) + " " + c.name + "：存储有损，row=" +
                                             std::to_string(r) + " d=" + std::to_string(d) + " 存 " +
                                             std::to_string(got) + " 期望 " + std::to_string(want));
                }
            }
        }
        gChecks++;

        // CPU / CUDA 的打包结果必须逐字节相同
        if (cuda) {
            Data q1cpu;
            q1cpu.CopyFrom(q1);
            q1cpu.ToDevice(DataDevice::CPU);
            Data packedCpu = QuantizeKV(q1cpu, c.quantMode, c.blockSize, false);
            Require(ReadBytes(packed) == ReadBytes(packedCpu),
                    std::string(c.name) + "：CUDA 与 CPU 的打包字节不一致");
        }
        std::cout << "  [" << tag << "] " << c.name << " -> " << rowBytes << " B/row, 逐 bit 无损\n";
    }
}

// ---------------- 3. SparseAttention 的量化读路径 ----------------

static Data RunAttention(const Data &q, const Data &chunkKV, Data *ring, Data *cmp, Data *idx,
                         Data &sink, int windowSize, int startPos, float scale, bool cuda) {
    Data out;
    DataDict datas = {{"q", (Data*)&q}, {"chunkKV", (Data*)&chunkKV}, {"attnSink", &sink}, {"output", &out}};
    if (ring != nullptr) {
        datas["ringKV"] = ring;
    }
    if (cmp != nullptr) {
        datas["compressedKV"] = cmp;
        datas["cmpIdx"] = idx;
    }
    Exec().RunOnDevice(cuda ? "cuda" : "cpu", "DeepSeekV41SparseAttention", datas,
                       {{"softmaxScale", scale}},
                       {{"windowSize", windowSize}, {"startPos", startPos}});
    return out;
}

static void CheckAttentionRead(bool cuda) {
    const char *tag = cuda ? "cuda" : "cpu";
    const int dim = 512, heads = 32, windowSize = 16, cap = 24, topWidth = 6;
    const int seqlen = 3, startPos = 40;
    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 0.6f);

    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = dist(rng);
        }
        return v;
    };

    Data qData = MakeBf16({1, seqlen, heads, dim}, rnd((size_t)seqlen * heads * dim));
    Data chunk = MakeBf16({1, seqlen, dim}, rnd((size_t)seqlen * dim));
    Data sink(DataType::FLOAT32, {heads});
    sink.Allocate();
    for (int h = 0; h < heads; h++) {
        ((float*)sink.cpuData)[h] = 0.1f * h - 1.0f;
    }
    Data idx(DataType::INT32, {1, seqlen, topWidth});
    idx.Allocate();
    for (int i = 0; i < seqlen * topWidth; i++) {
        ((int32_t*)idx.cpuData)[i] = (i * 5 + 1) % cap;
    }

    // 滑窗 KV 走 FP8 网格，压缩 KV 走 FP4 网格
    Data ringQ = FakeQuantRows(rnd((size_t)windowSize * dim), windowSize, dim, 1, 32, cuda);
    ringQ.Resize({1, windowSize, dim});
    Data cmpQ = FakeQuantRows(rnd((size_t)cap * dim), cap, dim, 3, 16, cuda);
    cmpQ.Resize({1, cap, dim});

    Data qDev, chunkDev, sinkDev, idxDev;
    qDev.CopyFrom(qData); chunkDev.CopyFrom(chunk); sinkDev.CopyFrom(sink); idxDev.CopyFrom(idx);
    ToDev(qDev, cuda); ToDev(chunkDev, cuda); ToDev(sinkDev, cuda); ToDev(idxDev, cuda);

    Data ringBf, cmpBf;
    ringBf.CopyFrom(ringQ); cmpBf.CopyFrom(cmpQ);
    Data ref = RunAttention(qDev, chunkDev, &ringBf, &cmpBf, &idxDev, sinkDev,
                            windowSize, startPos, 0.044f, cuda);

    Data ringPacked = QuantizeKV(ringQ, 1, 32, cuda);
    Data cmpPacked = QuantizeKV(cmpQ, 3, 16, cuda);
    Data got = RunAttention(qDev, chunkDev, &ringPacked, &cmpPacked, &idxDev, sinkDev,
                            windowSize, startPos, 0.044f, cuda);
    Require(BitEqual(ReadFloats(ref), ReadFloats(got)),
            std::string(tag) + " SparseAttention：读量化缓存行与读 BF16 行的结果不一致");
    std::cout << "  [" << tag << "] SparseAttention：FP8 滑窗 + FP4 压缩 KV 与 BF16 逐 bit 相同\n";

    // 只量化压缩 KV（滑窗仍是 BF16）也要一致，验证两路可以混用
    Data ringBf2;
    ringBf2.CopyFrom(ringQ);
    Data mixed = RunAttention(qDev, chunkDev, &ringBf2, &cmpPacked, &idxDev, sinkDev,
                              windowSize, startPos, 0.044f, cuda);
    Require(BitEqual(ReadFloats(ref), ReadFloats(mixed)),
            std::string(tag) + " SparseAttention：BF16 滑窗 + FP4 压缩 KV 的结果不一致");
}

static void CheckIndexerRead(bool cuda) {
    const char *tag = cuda ? "cuda" : "cpu";
    const int dim = 128, heads = 4, m = 70, seqlen = 5;
    std::mt19937 rng(4242);
    std::normal_distribution<float> dist(0.0f, 0.7f);
    auto rnd = [&](size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++) {
            v[i] = dist(rng);
        }
        return v;
    };

    Data q = MakeBf16({1, seqlen, heads, dim}, rnd((size_t)seqlen * heads * dim));
    Data w(DataType::FLOAT32, {1, seqlen, heads});
    w.Allocate();
    for (int i = 0; i < seqlen * heads; i++) {
        ((float*)w.cpuData)[i] = 0.3f + 0.05f * i;
    }
    Data kQ = FakeQuantRows(rnd((size_t)m * dim), m, dim, 2, 32, cuda);
    kQ.Resize({1, m, dim});

    Data qDev, wDev;
    qDev.CopyFrom(q); wDev.CopyFrom(w);
    ToDev(qDev, cuda); ToDev(wDev, cuda);

    Data kBf, refScore;
    kBf.CopyFrom(kQ);
    Exec().RunOnDevice(cuda ? "cuda" : "cpu", "DeepSeekV41IndexerScore",
                       {{"q", &qDev}, {"weights", &wDev}, {"k", &kBf}, {"output", &refScore}}, {},
                       {{"compressRatio", 1}, {"startPos", 1000}});

    Data kPacked = QuantizeKV(kQ, 2, 32, cuda), gotScore;
    Exec().RunOnDevice(cuda ? "cuda" : "cpu", "DeepSeekV41IndexerScore",
                       {{"q", &qDev}, {"weights", &wDev}, {"k", &kPacked}, {"output", &gotScore}}, {},
                       {{"compressRatio", 1}, {"startPos", 1000}});
    Require(BitEqual(ReadFloats(refScore), ReadFloats(gotScore)),
            std::string(tag) + " IndexerScore：读 FP4 key 与读 BF16 key 的分数不一致");
    std::cout << "  [" << tag << "] IndexerScore：FP4 indexer key 与 BF16 逐 bit 相同\n";
}

// ---------------- 4. IndexerTopK 的 CPU / CUDA 逐字节一致 ----------------
//
// CPU 参考实现是「按 (分数降序, 下标升序) 稳定排序取前 keep 个，再按下标升序输出，
// 不足 width 补 -1」。CUDA kernel 用 radix select + 阈值 + 并列预算复现这条规则，
// 必须逐字节一致，否则候选集变化会直接改掉注意力结果。
//
// 这里刻意把分数量化到很少的几档，制造大量并列（并列处理是最容易写错的地方），
// 并覆盖：有 / 无候选掩码、blockSize 是 / 不是 2 的幂（后者不走候选块压缩）、
// visible 小于 / 大于 width、topK 大于可用候选数。
static void CheckIndexerTopK() {
    struct Case {
        int seqlen, m, ratio, startPos, topK, blockSize, topkBlocks;
        bool useCand;
        const char *name;
    };
    const std::vector<Case> cases = {
        {6, 200, 1, 0,   32, 8,  6,  true,  "候选掩码 + blockSize 8"},
        {6, 200, 1, 0,   32, 0,  0,  false, "无候选掩码"},
        {5, 130, 2, 500, 24, 16, 3,  true,  "ratio 2 + startPos 500"},
        {4, 100, 1, 0,   16, 5,  4,  true,  "blockSize 5（非 2 的幂，不压缩候选块）"},
        {4, 100, 1, 0,  200, 8,  6,  true,  "topK 大于可用候选数"},
        {3, 64,  1, 0,    8, 8,  8,  true,  "候选块全选"},
        {3, 40,  1, 0,   16, 8,  1,  true,  "只留一个候选块"},
    };
    std::mt19937 rng(20260911);
    for (const Case &c : cases) {
        // 分数只取 6 档，制造大量并列
        Data score(DataType::FLOAT32, {1, c.seqlen, c.m});
        score.Allocate();
        for (int i = 0; i < c.seqlen * c.m; i++) {
            ((float*)score.cpuData)[i] = 0.25f * (float)(rng() % 6);
        }
        // 候选掩码由 CPU 的 CandidateBlocks 生成，两边共用同一份
        Data candCpu;
        bool hasCand = c.useCand && c.blockSize > 0;
        if (hasCand) {
            Data scoreCopy;
            scoreCopy.CopyFrom(score);
            Exec().RunOnDevice("cpu", "DeepSeekV41CandidateBlocks",
                               {{"score", &scoreCopy}, {"output", &candCpu}}, {},
                               {{"blockSize", c.blockSize}, {"topkBlocks", c.topkBlocks},
                                {"compressRatio", c.ratio}, {"startPos", c.startPos}});
        }
        IntDict ints = {{"topK", c.topK}, {"compressRatio", c.ratio}, {"startPos", c.startPos},
                        {"blockSize", std::max(1, c.blockSize)}};

        Data scoreCpu, candRefCpu, refOut;
        scoreCpu.CopyFrom(score);
        DataDict cpuDatas = {{"score", &scoreCpu}, {"output", &refOut}};
        if (hasCand) {
            candRefCpu.CopyFrom(candCpu);
            cpuDatas["candidates"] = &candRefCpu;
        }
        Exec().RunOnDevice("cpu", "DeepSeekV41IndexerTopK", cpuDatas, {}, ints);
        const std::vector<int32_t> ref = ReadInts(refOut);

        auto runCuda = [&](const char *legacy) {
            if (legacy != nullptr) {
                setenv(legacy, "1", 1);
            }
            Data scoreDev, candDev, gotOut;
            scoreDev.CopyFrom(score);
            ToDev(scoreDev, true);
            DataDict datas = {{"score", &scoreDev}, {"output", &gotOut}};
            if (hasCand) {
                candDev.CopyFrom(candCpu);
                ToDev(candDev, true);
                datas["candidates"] = &candDev;
            }
            Exec().RunOnDevice("cuda", "DeepSeekV41IndexerTopK", datas, {}, ints);
            std::vector<int32_t> got = ReadInts(gotOut);
            if (legacy != nullptr) {
                unsetenv(legacy);
            }
            return got;
        };

        Require(runCuda(nullptr) == ref,
                std::string("IndexerTopK[") + c.name + "]：CUDA 新 kernel 与 CPU 参考不一致");
        Require(runCuda("FASTLLM_DSV41_LEGACY_TOPK") == ref,
                std::string("IndexerTopK[") + c.name + "]：CUDA legacy kernel 与 CPU 参考不一致");
        std::cout << "  [cuda] IndexerTopK " << c.name << "：与 CPU 参考逐字节相同（新 / legacy）\n";
    }
}

// ---------------- 5. RotaryQuant 新旧 kernel 逐 bit 一致 ----------------
//
// 新实现把 quantMode == 0 拆成「只旋转」的 kernel，quantMode > 0 改成 shuffle 换对 +
// 一个 block 多行；两者都必须与旧的「一行一个 block + 共享内存」实现逐 bit 相同。
static void CheckRotaryQuantEquiv() {
    struct Case { int rows, rowsPerToken, dim, ropeDim, quantMode, quantBlock, startPos, posStep, inverse; };
    const std::vector<Case> cases = {
        {37,  1, 512, 128, 0, 32, 0,    1, 0},
        {36,  4, 512, 128, 0, 32, 1000, 1, 1},
        {33,  1, 512, 128, 1, 32, 7,    1, 0},
        {33,  1, 512, 128, 3, 16, 7,    2, 0},
        {130, 1, 128, 128, 2, 32, 5,    1, 0},
        {128, 8, 128, 64,  2, 32, 5,    1, 0},
    };
    std::mt19937 rng(777);
    std::normal_distribution<float> dist(0.0f, 1.3f);
    for (const Case &c : cases) {
        std::vector<int> dims = c.rowsPerToken > 1
            ? std::vector<int>{1, c.rows / c.rowsPerToken, c.rowsPerToken, c.dim}
            : std::vector<int>{1, c.rows, c.dim};
        size_t count = 1;
        for (int d : dims) {
            count *= (size_t)d;
        }
        std::vector<float> values(count);
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = dist(rng);
        }
        FloatDict floats = {{"ropeBase", 10000.0f}, {"ropeFactor", 1.0f}};
        IntDict ints = {{"ropeDim", c.ropeDim}, {"startPos", c.startPos}, {"posStep", c.posStep},
                        {"inverse", c.inverse}, {"originalSeqLen", 0}, {"betaFast", 32}, {"betaSlow", 1},
                        {"quantMode", c.quantMode}, {"quantBlock", c.quantBlock}};
        auto run = [&](bool legacy) {
            Data x = MakeBf16(dims, values);
            ToDev(x, true);
            if (legacy) {
                setenv("FASTLLM_DSV41_LEGACY_ROTARY", "1", 1);
            }
            Exec().RunOnDevice("cuda", "DeepSeekV41RotaryQuant", {{"input", &x}}, floats, ints);
            if (legacy) {
                unsetenv("FASTLLM_DSV41_LEGACY_ROTARY");
            }
            return ReadFloats(x);
        };
        Require(BitEqual(run(false), run(true)),
                "RotaryQuant：新 kernel 与 legacy kernel 不一致（quantMode " +
                std::to_string(c.quantMode) + ", dim " + std::to_string(c.dim) + ")");
    }
    std::cout << "  [cuda] RotaryQuant：新 kernel 与 legacy kernel 逐 bit 相同（6 组配置）\n";
}

// ---------------- 6. HcApplyPre + RMSNorm 融合与两步分开做逐 bit 一致 ----------------
//
// 融合 kernel 必须复现两件事：HcApplyPre 折叠后写 BF16 的舍入，以及 RMSNorm 的归约树
// （平方和的累加顺序、warp shuffle-down、rsqrtf(val / channels + eps)、写出的 lo * s * w）。
// 归约树与 THREAD_PER_BLOCK 绑定，所以这里覆盖 RMSNorm 会选到的三档线程数
// （channels < 512 -> 64、< 4096 -> 512、否则 1024），以及真实模型的 5120。
#ifdef USE_CUDA
static void CheckHcPreNormFused() {
    struct Case { int seqlen, hcMult, channels; };
    const std::vector<Case> cases = {
        {3, 4, 5120},   // 真实模型：hidden_size 5120、hc_mult 4
        {2, 4, 4096},
        {5, 4, 2048},
        {4, 2, 256},
        {2, 4, 1024},
    };
    std::mt19937 rng(31337);
    std::normal_distribution<float> dist(0.0f, 0.8f);
    for (const Case &c : cases) {
        const size_t n = (size_t)c.seqlen * c.hcMult * c.channels;
        std::vector<float> xv(n);
        for (size_t i = 0; i < n; i++) {
            xv[i] = dist(rng);
        }
        Data pre(DataType::FLOAT32, {1, c.seqlen, c.hcMult});
        pre.Allocate();
        for (int i = 0; i < c.seqlen * c.hcMult; i++) {
            ((float*)pre.cpuData)[i] = dist(rng);
        }
        Data w(DataType::FLOAT32, {c.channels});
        w.Allocate();
        for (int i = 0; i < c.channels; i++) {
            ((float*)w.cpuData)[i] = 0.5f + 0.001f * (i % 97);
        }
        const float eps = 1e-6f;

        // 两步：HcApplyPre + RMSNorm
        Data xA = MakeBf16({1, c.seqlen, c.hcMult, c.channels}, xv), preA, wA, mid, refOut;
        preA.CopyFrom(pre); wA.CopyFrom(w);
        ToDev(xA, true); ToDev(preA, true); ToDev(wA, true);
        Exec().RunOnDevice("cuda", "DeepSeekV41HcApplyPre",
                           {{"input", &xA}, {"pre", &preA}, {"output", &mid}}, {}, {});
        Exec().RunOnDevice("cuda", "RMSNorm", {{"input", &mid}, {"weight", &wA}, {"output", &refOut}},
                           {{"eps", eps}}, {});

        // 融合
        Data xB = MakeBf16({1, c.seqlen, c.hcMult, c.channels}, xv), preB, wB, gotOut;
        preB.CopyFrom(pre); wB.CopyFrom(w);
        ToDev(xB, true); ToDev(preB, true); ToDev(wB, true);
        Require(FastllmCudaDeepSeekV41HcPreNorm(xB, preB, wB, eps, gotOut),
                "HcPreNorm：融合 kernel 拒绝了 channels " + std::to_string(c.channels));
        Require(BitEqual(ReadFloats(refOut), ReadFloats(gotOut)),
                "HcPreNorm：融合与两步分开做不一致（channels " + std::to_string(c.channels) + ")");
    }
    std::cout << "  [cuda] HcApplyPre + RMSNorm：融合与两步分开做逐 bit 相同（5 组配置）\n";
}
#endif

int main() {
    try {
        gHasCuda = Exec().HasDevice("cuda");
        std::cout << "== 行布局与每 token 字节数 ==\n";
        CheckLayout();

        std::cout << "== QuantizeKV 无损性（CPU）==\n";
        CheckQuantizeLossless(false);
        std::cout << "== 读路径（CPU 参考实现）==\n";
        CheckAttentionRead(false);
        CheckIndexerRead(false);

        if (gHasCuda) {
            std::cout << "== QuantizeKV 无损性（CUDA）==\n";
            CheckQuantizeLossless(true);
            std::cout << "== 读路径（CUDA kernel）==\n";
            CheckAttentionRead(true);
            CheckIndexerRead(true);
            std::cout << "== 读路径（CUDA legacy 回退）==\n";
            setenv("FASTLLM_DSV41_LEGACY_ATTN", "1", 1);
            setenv("FASTLLM_DSV41_LEGACY_INDEXER", "1", 1);
            // 这两个开关在 kernel 里是每次调用都读的 getenv，进程内改动即时生效
            CheckAttentionRead(true);
            CheckIndexerRead(true);
            unsetenv("FASTLLM_DSV41_LEGACY_ATTN");
            unsetenv("FASTLLM_DSV41_LEGACY_INDEXER");
            std::cout << "== 读路径（CUDA 逐元素 FP4 解包回退）==\n";
            setenv("FASTLLM_DSV41_LEGACY_FP4_UNPACK", "1", 1);
            CheckAttentionRead(true);
            CheckIndexerRead(true);
            unsetenv("FASTLLM_DSV41_LEGACY_FP4_UNPACK");
            std::cout << "== IndexerTopK（CUDA vs CPU 逐字节）==\n";
            CheckIndexerTopK();
            std::cout << "== RotaryQuant（新 kernel vs legacy）==\n";
            CheckRotaryQuantEquiv();
#ifdef USE_CUDA
            std::cout << "== HcApplyPre + RMSNorm 融合 ==\n";
            CheckHcPreNormFused();
#endif
        } else {
            std::cout << "(未编译 / 未检测到 CUDA，跳过 GPU 部分)\n";
        }
        std::cout << "PASS: " << gChecks << " checks\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
