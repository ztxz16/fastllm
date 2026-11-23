#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <functional>
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
static double HostBF(const unsigned char *data, size_t i) {
    unsigned short bits; memcpy(&bits, data + i * 2, 2);
    return __bfloat162float(__ushort_as_bfloat16(bits));
}
static double HostKV(const unsigned char *data, size_t i, bool fp8) {
    if (!fp8) return HostBF(data, i);
    __nv_fp8_e4m3 value; value.__x = data[i]; return float(value);
}
static double Rounded(double x) { return __bfloat162float(__float2bfloat16_rn(float(x))); }
static bool TensorPVEnabled() {
    const char *setting = std::getenv("FASTLLM_NAIVE_SWA_CORE");
    return FastllmCudaRuntimeArch() == 103 && (!setting || std::strcmp(setting, "1") == 0);
}
// Only Tensor Core PV changes accumulation order. Cache contents, graph replay,
// and scalar/fallback attention retain the bitwise checks below.
static void CheckTensorPV(const std::vector<unsigned char> &expected,
        const unsigned char *actual) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < expected.size() / 2; ++i) {
        const double a = HostBF(expected.data(), i), b = HostBF(actual, i), delta = b - a;
        Require(std::isfinite(b) && std::abs(delta) <= .0005 + .008 * std::abs(a),
            "Tensor Core PV element error");
        error += delta * delta; norm += a * a;
    }
    Require(std::sqrt(error / std::max(norm, 1e-20)) < .001, "Tensor Core PV relative error");
}
static void CheckSwaOracle(const Data &q, const Data &key, const Data &value, const Data &sink,
        const std::vector<unsigned char> &output, int requests, int rows, int heads, int past, int kvHeads = 1, bool verify = true) {
    const auto query = Read(q), keys = Read(key, (size_t)256 * kvHeads * 192 * key.unitSize),
        values = Read(value, (size_t)256 * kvHeads * 128 * value.unitSize), bias = Read(sink);
    const bool fp8 = key.dataType == FP8_E4M3;
    double error = 0, norm = 0;
    for (int request : {0, requests - 1}) for (int row : {0, rows - 1}) for (int head : {0, heads / 2, heads - 1}) {
        const int length = past + row + 1, start = std::max(0, length - 128);
        std::vector<double> scores(length - start), probabilities(scores.size());
        float sinkValue; memcpy(&sinkValue, bias.data() + head * sizeof(float), sizeof(float));
        double maximum = sinkValue;
        auto physical = [&](int logical) {
            const int token = verify && logical >= past ? 128 + logical - past : logical & 127;
            return token * kvHeads + head / (heads / kvHeads);
        };
        for (int logical = start; logical < length; ++logical) {
            double dot = 0;
            for (int d = 0; d < 192; ++d)
                dot += HostBF(query.data(), ((size_t)(request * rows + row) * heads + head) * 192 + d) *
                    HostKV(keys.data(), (size_t)physical(logical) * 192 + d, fp8);
            scores[logical - start] = Rounded(Rounded(dot) / std::sqrt(192.0));
            maximum = std::max(maximum, scores[logical - start]);
        }
        double denominator = std::exp(double(sinkValue) - maximum);
        for (double score : scores) denominator += std::exp(score - maximum);
        for (size_t i = 0; i < scores.size(); ++i) probabilities[i] = Rounded(std::exp(scores[i] - maximum) / denominator);
        for (int d = 0; d < 128; ++d) {
            double expected = 0;
            for (int logical = start; logical < length; ++logical)
                expected += probabilities[logical - start] * HostKV(values.data(), (size_t)physical(logical) * 128 + d, fp8);
            expected = Rounded(expected);
            const double got = HostBF(output.data(), ((size_t)(request * rows + row) * heads + head) * 128 + d);
            const double delta = got - expected;
            Require(std::isfinite(got) && std::abs(delta) <= .004 + .01 * std::abs(expected), "SWA double oracle element error");
            error += delta * delta; norm += expected * expected;
        }
    }
    Require(std::sqrt(error / std::max(norm, 1e-20)) < .004, "SWA double oracle relative error");
}
static float Time(const std::function<void()> &run) {
    run(); Check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph; cudaGraphExec_t exec; cudaEvent_t a,b;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    for (int i=0; i<100; ++i) run();
    Check(cudaStreamEndCapture(cudaStreamPerThread,&graph)); Check(cudaGraphInstantiate(&exec,graph,0));
    Check(cudaEventCreate(&a)); Check(cudaEventCreate(&b)); std::vector<float> samples;
    for (int i=0;i<8;++i) {
        Check(cudaEventRecord(a,cudaStreamPerThread)); Check(cudaGraphLaunch(exec,cudaStreamPerThread));
        Check(cudaEventRecord(b,cudaStreamPerThread)); Check(cudaEventSynchronize(b));
        float ms; Check(cudaEventElapsedTime(&ms,a,b)); if(i>=2)samples.push_back(ms*10);
    }
    std::sort(samples.begin(),samples.end()); Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
    Check(cudaEventDestroy(a)); Check(cudaEventDestroy(b)); return samples[samples.size()/2];
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
static void GroupedRingTiles() {
    // Exercise every supported KV head layout and the 32-row dispatch boundary with
    // independent requests. The original verification kernel is the reference.
    for (int heads : {8, 16, 32}) for (int rows : {1, 2, 4, 5, 7, 8})
        for (int requests : {1, 2, 3, 4, 5, 8, 9}) for (int past : {0, 127, 4095}) {
            const int kvHeads = heads / 8, total = rows * requests;
            Data query(BFLOAT16), live(INT32), sink(FLOAT32), output, empty;
            Init(query, {1, total, heads * 192}); Fill(query, 71 + total + heads);
            Init(live, {total}); Init(sink, {heads});
            std::vector<float> bias(heads, .1f);
            Check(cudaMemcpy(sink.cudaData, bias.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
            std::vector<int> lengths(total);
            std::vector<Data> keys(requests), values(requests);
            std::vector<FastllmNaiveSWARow> descriptors(total);
            for (int r = 0; r < requests; ++r) {
                keys[r].dataType = values[r].dataType = BFLOAT16;
                keys[r].UpdateUnitSize(); values[r].UpdateUnitSize();
                Init(keys[r], {1, 256, kvHeads * 192}); Fill(keys[r], 113 + r);
                Init(values[r], {1, 256, kvHeads * 128}); Fill(values[r], 217 + r);
                for (int row = 0; row < rows; ++row) {
                    const int index = r * rows + row;
                    lengths[index] = past + row + 1;
                    auto &d = descriptors[index];
                    d.key = keys[r].cudaData; d.value = values[r].cudaData;
                    d.length = lengths[index]; d.verifyOffset = row;
                }
            }
            Check(cudaMemcpy(live.cudaData, lengths.data(), total * sizeof(int), cudaMemcpyHostToDevice));
            // Include the no-sink path without changing the per-request reference.
            const Data &biasData = past == 0 ? empty : sink;
            FastllmCudaNaiveBatchSWADecode(query, biasData, descriptors, live, heads, kvHeads, output);
            auto got = Read(output);
            for (int r = 0; r < requests; ++r) {
                Data q(BFLOAT16, {1, rows, heads * 192}), n(INT32, {rows}), reference;
                q.FakeFrom(query, (size_t)r * rows * heads * 192 * 2);
                n.FakeFrom(live, r * rows * sizeof(int));
                FastllmNaiveDecodeScratch scratch;
                FastllmCudaNaiveGraphVerifyAttention(q, keys[r], values[r], empty, biasData,
                    n, 256, heads, kvHeads, 192, 128, 128, scratch, reference, true);
                auto expected = Read(reference);
                const auto *actual = got.data() + (size_t)r * rows * heads * 128 * 2;
                if (TensorPVEnabled()) CheckTensorPV(expected, actual);
                else Require(std::equal(expected.begin(), expected.end(), actual), "grouped SWA fallback differs");
            }
        }
}
static void RaggedRingReplay() {
    for (int heads : {8, 16, 32})
        for (const std::vector<int> sizes : {std::vector<int>{3,8,7,8,6},
                std::vector<int>{7,8,8,8,8}, std::vector<int>{1,8,8,8,8,8}}) {
            int total = 0; for (int n : sizes) total += n;
            const int requests = sizes.size(), kvHeads = heads / 8;
            Data q(BFLOAT16), live(INT32), table(INT32), sink(FLOAT32), output, empty;
            Init(q, {1,total,heads*192}); Fill(q, 193+total);
            Init(live, {total}); Init(table, {requests*4}); Init(sink, {heads});
            std::vector<float> bias(heads,.1f);
            Check(cudaMemcpy(sink.cudaData,bias.data(),heads*4,cudaMemcpyHostToDevice));
            std::vector<Data> keys(requests*2), values(requests*2);
            for (int r=0;r<requests*2;++r) {
                keys[r].dataType=values[r].dataType=BFLOAT16;
                keys[r].UpdateUnitSize(); values[r].UpdateUnitSize();
                Init(keys[r],{1,256,kvHeads*192}); Fill(keys[r],71+r*17);
                Init(values[r],{1,256,kvHeads*128}); Fill(values[r],91+r*31);
            }
            std::vector<FastllmNaiveSWARow> rows(total);
            std::vector<int> lengths(total);
            for(int r=0,offset=0;r<requests;++r) {
                for(int i=0;i<sizes[r];++i) {
                    auto &d=rows[offset+i]; d.key=keys[r].cudaData; d.value=values[r].cudaData;
                    d.cache=(const void *const *)table.cudaData+r*2; d.verifyOffset=i;
                    lengths[offset+i]=d.length=(r%2?8192:127)+i+1;
                }
                offset+=sizes[r];
            }
            auto bind=[&](int replay) {
                std::vector<const void *> pointers;
                for(int r=0;r<requests;++r) {
                    const int index=r+(replay%2)*requests;
                    pointers.push_back(keys[index].cudaData); pointers.push_back(values[index].cudaData);
                }
                auto current=lengths; for(auto &n:current)n+=replay*257;
                Check(cudaMemcpy(table.cudaData,pointers.data(),pointers.size()*sizeof(void*),cudaMemcpyHostToDevice));
                Check(cudaMemcpy(live.cudaData,current.data(),total*4,cudaMemcpyHostToDevice));
            };
            bind(0);
            auto run=[&]{FastllmCudaNaiveBatchSWADecode(q,sink,rows,live,heads,kvHeads,output);};
            run(); Check(cudaStreamSynchronize(cudaStreamPerThread));
            cudaGraph_t graph; cudaGraphExec_t exec;
            Check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)); run();
            Check(cudaStreamEndCapture(cudaStreamPerThread,&graph)); Check(cudaGraphInstantiate(&exec,graph,0));
            for(int replay=0;replay<3;++replay) {
                bind(replay); Check(cudaGraphLaunch(exec,cudaStreamPerThread));
                Check(cudaStreamSynchronize(cudaStreamPerThread)); auto got=Read(output);
                for(int r=0,offset=0;r<requests;++r) {
                    Data query(BFLOAT16,{1,sizes[r],heads*192}), counts(INT32,{sizes[r]}), reference;
                    query.FakeFrom(q,size_t(offset)*heads*192*2); counts.FakeFrom(live,offset*4);
                    const int index=r+(replay%2)*requests; FastllmNaiveDecodeScratch scratch;
                    FastllmCudaNaiveGraphVerifyAttention(query,keys[index],values[index],empty,sink,
                        counts,256,heads,kvHeads,192,128,128,scratch,reference,true);
                    CheckTensorPV(Read(reference),got.data()+size_t(offset)*heads*128*2);
                    offset+=sizes[r];
                }
            }
            Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
        }
}


// Attention DP can retain all 64 query / 8 KV heads on one GPU. Cover the
// bulk threshold, descriptor chunks and both decode/verification layouts.
static void DPHeads(int requests, int rows, int heads, DataType type, int past) {
    const int kvHeads = heads / 8, count = requests * rows;
    Data q(BFLOAT16), key(type), value(type), sink(FLOAT32), output, empty;
    Init(q, {1, count, heads * 192}); Fill(q, 793 + count);
    Init(key, {1, 256, kvHeads * 192}); Init(value, {1, 256, kvHeads * 128});
    for (Data *d : {&key, &value}) {
        if (type == BFLOAT16) Fill(*d, 117 + d->dims[2]);
        else {
            std::vector<unsigned char> bytes(d->GetBytes());
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = 0x20 + (i * 13 + i / 127) % 40;
            Write(*d, bytes);
        }
    }
    Init(sink, {heads}); std::vector<float> bias(heads);
    for (int h = 0; h < heads; ++h) bias[h] = h == 0 ? .1f : h == heads - 1 ? -.2f : h % 3 == 0 ? 80.f : h % 3 == 1 ? -80.f : .1f;
    Check(cudaMemcpy(sink.cudaData, bias.data(), heads * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<FastllmNaiveSWARow> descriptors(count);
    for (int i = 0; i < count; ++i) {
        auto &d = descriptors[i]; d.key = key.cudaData; d.value = value.cudaData;
        d.fp8 = type == FP8_E4M3; d.ring = true;
        d.length = past + i % rows + 1; d.verifyOffset = rows > 1 ? i % rows : -1;
    }
    auto run = [&] { FastllmCudaNaiveBatchSWADecode(q, sink, descriptors, empty, heads, kvHeads, output); };
    run(); auto expected = Read(output);
    if (count < 32) {
        // One-row calls retain the previous scalar path. Check every output
        // bit when sharing KV across heads, including all verification rows.
        for (int i = 0; i < count; ++i) {
            Data query(BFLOAT16, {1, 1, heads * 192}), reference;
            query.FakeFrom(q, (size_t)i * heads * 192 * sizeof(uint16_t));
            FastllmCudaNaiveBatchSWADecode(query, sink, {descriptors[i]}, empty, heads, kvHeads, reference);
            const auto bytes = Read(reference);
            Require(std::equal(bytes.begin(), bytes.end(), expected.begin() + (size_t)i * heads * 128 * sizeof(uint16_t)),
                "shared-head SWA changed scalar output bits");
        }
    }
    CheckSwaOracle(q, key, value, sink, expected, requests, rows, heads, past, kvHeads, rows > 1);
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    run(); Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, 0)); Check(cudaGraphLaunch(exec, cudaStreamPerThread));
    Require(expected == Read(output), "DP-head SWA Graph replay differs");
    Check(cudaGraphExecDestroy(exec)); Check(cudaGraphDestroy(graph));
}

int main(int argc, char **argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        Check(cudaSetDevice(0));
        if (argc > 1) {
            int checks = 0;
            for (DataType type : {BFLOAT16, FP8_E4M3})
                for (int heads : {16, 32, 64})
                    for (int rows : {1, 3, 5, 6, 8})
                        for (int requests : {1, 3, 4, 7, 8, 16, 31, 32, 33}) {
                            DPHeads(requests, rows, heads, type, requests < 8 ? 127 : 200003);
                            ++checks;
                        }
            printf("DP HEADS SWA PASS checks=%d; double oracle and exact Graph replay\n", checks);
            return 0;
        }

        GroupedRingTiles();
        RaggedRingReplay();
        for (int n : {2, 8, 80, 112, 113, 380, 1025})
            for (DataType type : {BFLOAT16, FP8_E4M3}) BatchCommit(n, type);
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
                            auto expected = Read(ref), actual = Read(batched);
                            if (ring && type == BFLOAT16 && heads == 8 && TensorPVEnabled())
                                CheckTensorPV(expected, actual.data());
                            else Require(actual == expected, "batched causal SWA fallback differs bitwise");
                            ++checks;
                            if (ring && type == BFLOAT16 && heads == 8 && past == 127) {
                                Data table(INT32), alternateKey(BFLOAT16), alternateValue(BFLOAT16), captured;
                                Init(table, {4}); Init(alternateKey, {1,256,192}); Init(alternateValue, {1,256,128});
                                Fill(alternateKey, 71 + rows); Fill(alternateValue, 83 + rows);
                                for (int i = 0; i < rows; ++i) {
                                    descriptors[i].cache = (const void *const *)table.cudaData;
                                }
                                auto bind = [&](bool other) {
                                    const void *p[] = {other ? alternateKey.cudaData : key.cudaData,
                                                      other ? alternateValue.cudaData : value.cudaData};
                                    Check(cudaMemcpy(table.cudaData, p, sizeof(p), cudaMemcpyHostToDevice));
                                };
                                bind(false);
                                FastllmCudaNaiveBatchSWADecode(q, sink, descriptors, live, heads, 1, captured);
                                cudaGraph_t g; cudaGraphExec_t e;
                                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                                FastllmCudaNaiveBatchSWADecode(q, sink, descriptors, live, heads, 1, captured);
                                Check(cudaStreamEndCapture(cudaStreamPerThread, &g));
                                Check(cudaGraphInstantiate(&e, g, 0)); Check(cudaGraphDestroy(g));
                                for (int pass = 0; pass < 4; ++pass) {
                                    bind(pass % 2);
                                    auto active = lengths;
                                    if (pass >= 2) for (int i = (rows+1)/2; i < rows; ++i) active[i] = 0;
                                    Check(cudaMemcpy(live.cudaData, active.data(), rows*4, cudaMemcpyHostToDevice));
                                    auto direct = descriptors;
                                    for (auto &d : direct) {
                                        d.cache = nullptr;
                                        d.key = pass % 2 ? alternateKey.cudaData : key.cudaData;
                                        d.value = pass % 2 ? alternateValue.cudaData : value.cudaData;
                                    }
                                    FastllmCudaNaiveBatchSWADecode(q, sink, direct, live, heads, 1, batched);
                                    Check(cudaGraphLaunch(e, cudaStreamPerThread));
                                    Require(Read(captured) == Read(batched), "SWA verify graph cache rebinding/inactive row differs");
                                    ++checks;
                                }
                                Check(cudaMemcpy(live.cudaData, lengths.data(), rows*4, cudaMemcpyHostToDevice));
                                bind(false);
                                float oldTime=Time([&]{FastllmCudaNaiveGraphVerifyAttention(q,key,value,empty,sink,live,
                                    256,heads,1,192,128,128,refScratch,ref,true);});
                                float newTime=Time([&]{FastllmCudaNaiveBatchSWADecode(q,sink,descriptors,live,heads,1,captured);});
                                printf("VERIFY SWA CORE rows=%d old_us=%.3f fused_us=%.3f\n",rows,oldTime,newTime);
                                Check(cudaGraphExecDestroy(e));
                            }
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
                                CheckSwaOracle(queries, key, value, sink, got, requests, rows, heads, past);
                                for (int request = 0; request < requests; ++request) {
                                    Data query(BFLOAT16, {1, rows, heads * 192}), expected;
                                    query.FakeFrom(queries, (size_t)request * rows * heads * 192 * 2);
                                    FastllmCudaNaiveGraphVerifyAttention(query, key, value, empty, sink, live,
                                        256, heads, 1, 192, 128, 128, refScratch, expected, true);
                                    auto values = Read(expected);
                                    if (rows * requests >= 32) {
                                        // The bulk path uses Tensor Core accumulation;
                                        // use the existing BF16 bound plus the independent
                                        // double oracle, not scalar-kernel bit identity.
                                        double err = 0, norm = 0;
                                        for (size_t j = 0; j < values.size() / 2; ++j) {
                                            double ref = HostBF(values.data(), j);
                                            double gotValue = HostBF(got.data(), (size_t)request * rows * heads * 128 + j);
                                            double delta = gotValue - ref;
                                            Require(std::isfinite(gotValue) && std::abs(delta) <= .004 + .01 * std::abs(ref), "bulk SWA element error");
                                            err += delta * delta; norm += ref * ref;
                                        }
                                        Require(std::sqrt(err / std::max(norm, 1e-20)) < .004, "bulk SWA relative error");
                                    } else if (type == BFLOAT16 && heads == 8 && TensorPVEnabled()) {
                                        CheckTensorPV(values, got.data() + (size_t)request * rows * heads * 128 * 2);
                                    } else Require(std::equal(values.begin(), values.end(),
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
        printf("SWA VERIFY PASS checks=%d; bounded Tensor Core PV, bitwise scalar "
               "attention/cache/replay, grouped KV heads, keep=0..rows\n",
               checks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
