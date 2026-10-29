#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
static void Require(bool ok, const char *s) {
    if (!ok)
        throw std::runtime_error(s);
}
static void Init(Data &d, std::vector<int> shape) {
    d.Resize(shape);
    d.ToDevice(DataDevice::CUDA, {0}, false);
    d.Allocate(false);
}
static std::vector<unsigned char> Read(const Data &d, size_t bytes = 0) {
    std::vector<unsigned char> v(bytes ? bytes : d.GetBytes());
    Check(cudaMemcpy(v.data(), d.cudaData, v.size(), cudaMemcpyDeviceToHost));
    return v;
}
static void Write(Data &d, const std::vector<unsigned char> &v) {
    Check(cudaMemcpy(d.cudaData, v.data(), v.size(), cudaMemcpyHostToDevice));
}
static void Fill(Data &d, unsigned seed) {
    std::vector<__nv_bfloat16> v(d.Count(0));
    for (auto &x : v) {
        seed = seed * 1664525 + 1013904223;
        x = __float2bfloat16_rn((int(seed >> 24) - 128) / 128.f);
    }
    Check(cudaMemcpy(d.cudaData, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
}
static void BatchCommit(int requests, DataType type) {
    std::vector<Data> keys(requests), values(requests), rk(requests), rv(requests);
    std::vector<FastllmNaiveCacheCommit> updates;
    for (int i = 0; i < requests; ++i) {
        const int kc = 128 + i % 3 * 64, vc = 64 + i % 4 * 32;
        for (Data *d : {&keys[i], &values[i], &rk[i], &rv[i]}) {
            d->dataType = type; d->UpdateUnitSize();
            int width = d == &keys[i] || d == &rk[i] ? kc : vc;
            Init(*d, {1, 144, width}); d->expansionDims = d->dims;
            std::vector<unsigned char> bytes(d->GetBytes());
            for (size_t j = 0; j < bytes.size(); ++j) bytes[j] = (j * 17 + i * 37) % 256;
            Write(*d, bytes); d->Resize({1, 127, width});
        }
        int past = i % 3 == 0 ? 3 : 123 + i * 1023, count = i % 9;
        updates.push_back({&keys[i], &values[i], past, count});
        FastllmCudaNaiveCommitSwaVerify(rk[i], rv[i], past, count);
    }
    FastllmCudaNaiveBatchCommitSwaVerify(updates);
    for (int i = 0; i < requests; ++i) {
        Require(keys[i].dims == rk[i].dims && values[i].dims == rv[i].dims, "batch SWA commit metadata differs");
        Require(Read(keys[i], (size_t)144 * keys[i].dims[2] * keys[i].unitSize) == Read(rk[i], (size_t)144 * rk[i].dims[2] * rk[i].unitSize), "batch SWA key commit differs");
        Require(Read(values[i], (size_t)144 * values[i].dims[2] * values[i].unitSize) == Read(rv[i], (size_t)144 * rv[i].dims[2] * rv[i].unitSize), "batch SWA value commit differs");
    }
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        Check(cudaSetDevice(0));
        for (int n : {2, 8, 80}) for (DataType type : {BFLOAT16, FP8_E4M3}) BatchCommit(n, type);
        int checks = 0;
        for (DataType type : {BFLOAT16, FP8_E4M3})
            for (int heads : {4, 8, 16, 32})
                for (int rows : {1, 2, 3, 4, 5, 6, 7, 8})
                    for (int past : {0, 1, 120, 127, 128, 129, 254, 4095}) {
                        Data q(BFLOAT16), qr(BFLOAT16), k(BFLOAT16), v(BFLOAT16), packed(BFLOAT16),
                            pos(FLOAT32), live(INT32);
                        Data key(type), value(type), kr(type), vr(type), sink(FLOAT32), out, ref, empty;
                        FastllmNaiveDecodeScratch scratch, refScratch;
                        Init(q, {1, rows, heads * 192});
                        Init(qr, q.dims);
                        Init(k, {1, rows, 192});
                        Init(v, {1, rows, 128});
                        Init(packed, {1, rows, (heads + 1) * 192 + 128});
                        Fill(packed, past + rows + 11);
                        Init(pos, {1, rows});
                        Init(live, {rows});
                        Init(sink, {heads});
                        std::vector<float> positions(rows), bias(heads, .1f);
                        std::vector<int> lengths(rows);
                        for (int i = 0; i < rows; ++i) {
                            positions[i] = past + i;
                            lengths[i] = past + i + 1;
                        }
                        Check(cudaMemcpy(pos.cudaData, positions.data(), rows * 4, cudaMemcpyHostToDevice));
                        Check(cudaMemcpy(live.cudaData, lengths.data(), rows * 4, cudaMemcpyHostToDevice));
                        Check(cudaMemcpy(sink.cudaData, bias.data(), heads * 4, cudaMemcpyHostToDevice));
                        for (auto *d : {&key, &kr, &value, &vr}) {
                            int width = (d == &key || d == &kr) ? 192 : 128;
                            Init(*d, {1, 256, width});
                            d->expansionDims = d->dims;
                            d->Resize({1, std::min(past, 127), width});
                        }
                        const int unit = key.unitSize, count = std::min(past, 127);
                        auto prefix = [&](Data &linear, Data &ring, int width) {
                            std::vector<unsigned char> a(256 * width * unit, 0x20), b = a;
                            for (int row = 0; row < count; ++row)
                                for (int col = 0; col < width; ++col) {
                                    unsigned short bf = __bfloat16_as_ushort(
                                        __float2bfloat16_rn(((row * 31 + col * 7) % 63 - 31) / 32.f));
                                    unsigned char fp8 = 0x20 + (row * 13 + col) % 40;
                                    int target = (past - count + row) & 127;
                                    if (unit == 2) {
                                        memcpy(a.data() + (row * width + col) * 2, &bf, 2);
                                        memcpy(b.data() + (target * width + col) * 2, &bf, 2);
                                    } else {
                                        a[row * width + col] = fp8;
                                        b[target * width + col] = fp8;
                                    }
                                }
                            Write(linear, a);
                            Write(ring, b);
                            return b;
                        };
                        auto oldKey = prefix(kr, key, 192), oldValue = prefix(vr, value, 128);
                        auto fused = [&](bool ring) {
                            Require(FastllmCudaNaiveRopeAppendCache(ring ? q : qr, k, v, empty, pos,
                                                                    ring ? key : kr, ring ? value : vr, live,
                                                                    heads, 1, 192, 128, 64, 10000.f, .707f,
                                                                    128, &packed, false, ring),
                                    "staged RoPE rejected");
                            FastllmCudaNaiveGraphVerifyAttention(
                                ring ? q : qr, ring ? key : kr, ring ? value : vr, empty, sink, live, 256,
                                heads, 1, 192, 128, 128, ring ? scratch : refScratch, ring ? out : ref, ring);
                        };
                        fused(false);
                        fused(true);
                        for (bool ring : {false, true}) {
                            std::vector<FastllmNaiveSWARow> descriptors(rows);
                            for (int row = 0; row < rows; ++row) {
                                auto &d = descriptors[row];
                                d.key = ring ? key.cudaData : kr.cudaData;
                                d.value = ring ? value.cudaData : vr.cudaData;
                                d.fp8 = type == FP8_E4M3;
                                d.ring = ring;
                                d.length = (ring ? past : std::min(past, 127)) + row + 1;
                                d.verifyOffset = ring ? row : -1;
                            }
                            Data batched;
                            FastllmCudaNaiveBatchSWADecode(ring ? q : qr, sink, descriptors,
                                empty, heads, 1, batched);
                            Require(Read(batched) == Read(ref), "batched causal SWA differs bitwise");
                            ++checks;
                        }
                        // Cross the descriptor chunk boundary with independent
                        // query payloads. Reference each request separately.
                        if (past == 4095 && rows % 2 == 0 && (heads == 8 || heads == 16)) {
                            for (int requests : {2, 3, 4, 8, 16, 33, 40}) {
                                Data queries(BFLOAT16), batched;
                                Init(queries, {1, rows * requests, heads * 192});
                                Fill(queries, 918 + requests);
                                std::vector<FastllmNaiveSWARow> descriptors(rows * requests);
                                for (int i = 0; i < rows * requests; ++i) {
                                    auto &d = descriptors[i]; d.key = key.cudaData; d.value = value.cudaData;
                                    d.fp8 = type == FP8_E4M3; d.length = past + i % rows + 1;
                                    d.verifyOffset = i % rows;
                                }
                                FastllmCudaNaiveBatchSWADecode(queries, sink, descriptors, empty,
                                    heads, 1, batched);
                                auto got = Read(batched);
                                for (int request = 0; request < requests; ++request) {
                                    Data query(BFLOAT16, {1, rows, heads * 192}), expected;
                                    query.FakeFrom(queries, (size_t)request * rows * heads * 192 * 2);
                                    FastllmCudaNaiveGraphVerifyAttention(query, key, value, empty, sink, live,
                                        256, heads, 1, 192, 128, 128, refScratch, expected, true);
                                    auto values = Read(expected);
                                    Require(std::equal(values.begin(), values.end(),
                                        got.begin() + (size_t)request * rows * heads * 128 * 2),
                                        "large batch staged SWA differs from independent request");
                                    ++checks;
                                }
                            }
                        }
                        Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                        fused(true);
                        cudaGraph_t graph;
                        cudaGraphExec_t exec;
                        Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                        Check(cudaGraphInstantiate(&exec, graph, 0));
                        Check(cudaGraphDestroy(graph));
                        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
                        Require(Read(out) == Read(ref), "staged causal attention differs");
                        auto stagedKey = Read(key, oldKey.size()), stagedValue = Read(value, oldValue.size());
                        Require(
                            std::equal(oldKey.begin(), oldKey.begin() + 128 * 192 * unit, stagedKey.begin()),
                            "verification changed key prefix");
                        Require(std::equal(oldValue.begin(), oldValue.begin() + 128 * 128 * unit,
                                           stagedValue.begin()),
                                "verification changed value prefix");
                        for (int keep = 0; keep <= rows; ++keep) {
                            Write(key, stagedKey);
                            Write(value, stagedValue);
                            auto expectedKey = stagedKey, expectedValue = stagedValue;
                            for (int i = 0; i < keep; ++i) {
                                memcpy(expectedKey.data() + ((past + i) & 127) * 192 * unit,
                                       stagedKey.data() + (128 + i) * 192 * unit, 192 * unit);
                                memcpy(expectedValue.data() + ((past + i) & 127) * 128 * unit,
                                       stagedValue.data() + (128 + i) * 128 * unit, 128 * unit);
                            }
                            FastllmCudaNaiveCommitSwaVerify(key, value, past, keep);
                            Require(Read(key, oldKey.size()) == expectedKey &&
                                        Read(value, oldValue.size()) == expectedValue,
                                    "commit/rejection changed wrong rows");
                            Require(key.dims[1] == std::min(past + keep, 127), "commit length differs");
                            ++checks;
                        }
                        Check(cudaGraphExecDestroy(exec));
                    }
        printf("SWA VERIFY PASS checks=%d; bitwise causal attention, prefix "
               "preservation, keep=0..rows\n",
               checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
