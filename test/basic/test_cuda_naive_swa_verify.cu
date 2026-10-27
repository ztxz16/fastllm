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
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            return 77;
        Check(cudaSetDevice(0));
        int checks = 0;
        for (DataType type : {BFLOAT16, FP8_E4M3})
            for (int heads : {8, 16, 32})
                for (int rows : {1, 2, 6, 8})
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
