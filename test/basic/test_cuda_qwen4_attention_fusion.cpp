#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Init(Data &data, const std::vector<int> &shape, int seed) {
    data.Resize(shape);
    data.Allocate();
    for (uint64_t i = 0; i < data.Count(0); ++i) {
        float value = std::sin((i + seed) * 1.137f) * 2.3f;
        if (data.dataType == FLOAT32) ((float*)data.cpuData)[i] = value;
        else if (data.dataType == FLOAT16) ((uint16_t*)data.cpuData)[i] = float_to_half(value);
        else {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            ((uint16_t*)data.cpuData)[i] = bits >> 16;
        }
    }
    data.ToDevice(DataDevice::CUDA, {0}, true);
}

static void Cache(Data &data, int heads, int capacity, int length, int dim) {
    data.Expansion({heads, capacity, dim});
    data.Resize({heads, length, dim});
    std::memset(data.cpuData, 0x35, data.expansionBytes);
    data.ToDevice(DataDevice::CUDA, {0}, true);
}

static std::vector<unsigned char> Read(const Data &data, bool capacity = false) {
    std::vector<unsigned char> bytes(capacity ? data.expansionBytes : data.GetBytes());
    Require(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
            "CUDA copy failed");
    return bytes;
}

static void Equal(const Data &a, const Data &b, const char *name, bool capacity = false) {
    const auto av = Read(a, capacity), bv = Read(b, capacity);
    if (av == bv) return;
    size_t mismatches = 0;
    float maxError = 0;
    bool close = a.dataType == FLOAT32;
    const int unit = a.dataType == FLOAT32 ? 4 : 2;
    for (size_t i = 0; i < av.size(); i += unit) {
        if (std::memcmp(av.data() + i, bv.data() + i, unit) == 0) continue;
        ++mismatches;
        auto number = [&](const unsigned char *p) {
            float x;
            if (unit == 4) std::memcpy(&x, p, 4);
            else {
                uint16_t h; std::memcpy(&h, p, 2);
                if (a.dataType == FLOAT16) x = half_to_float(h);
                else { uint32_t u = (uint32_t)h << 16; std::memcpy(&x, &u, 4); }
            }
            return x;
        };
        const float actual = number(av.data() + i), expected = number(bv.data() + i);
        const float error = std::abs(actual - expected);
        maxError = std::max(maxError, error);
        close = close && std::isfinite(actual) && std::isfinite(expected) &&
                error <= 2e-6f + 2e-6f * std::abs(expected);
    }
    if (close) return;
    std::fprintf(stderr, "%s: %zu differing elements, max error %.9g\n", name, mismatches, maxError);
    throw std::runtime_error("fused attention differs from separate operators");
}

static void Run(DataType type, int batch, int sequence, int dim, bool mrope) {
    const int qHeads = 6, kvHeads = 2, previous = 11;
    const int rotary = dim / 2, sectionH = rotary / 8, sectionW = rotary / 8;
    Data qgate(type), key(type), value(type), qnorm(FLOAT32), knorm(FLOAT32), positions(FLOAT32);
    Init(qgate, {batch, sequence, qHeads * dim * 2}, 1);
    Init(key, {batch, sequence, kvHeads * dim}, 7);
    Init(value, key.dims, 13);
    Init(qnorm, {dim}, 17); Init(knorm, {dim}, 23);
    std::vector<float> pos((mrope ? 3 : batch) * sequence);
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = 100003 + i * 7;
    positions.CopyFrom(Data(FLOAT32, {mrope ? 3 : batch, sequence}, pos));
    positions.ToDevice(DataDevice::CUDA, {0}, true);
    Data kc(type), vc(type), rk(type), rv(type), q(type), gate(type), rq(type), rg(type);
    Cache(kc, batch * kvHeads, previous + sequence + 17, previous, dim);
    Cache(vc, batch * kvHeads, previous + sequence + 31, previous, dim);
    Cache(rk, batch * kvHeads, previous + sequence + 17, previous, dim);
    Cache(rv, batch * kvHeads, previous + sequence + 31, previous, dim);
    auto fused = [&](int offset) {
        Require(FastllmCudaQwen4AttentionPrepare(qgate, key, value, qnorm, knorm,
            positions, q, gate, kc, vc, dim, rotary, sectionH, sectionW,
            1e-6f, 1000000.f, offset), "fused preparation rejected");
    };
    auto reference = [&](int offset) {
        Data split(type), nk(type), nv(type);
        // Only metadata changes; restore it before the fused call.
        qgate.Reshape({batch, sequence, qHeads, 2 * dim});
        Split(qgate, -1, 0, dim, split);
        Split(qgate, -1, dim, 2 * dim, rg);
        qgate.Reshape({batch, sequence, qHeads * dim * 2});
        rg.Reshape({batch, sequence, qHeads * dim});
        RMSNorm(split, qnorm, 1e-6f, rq);
        key.Reshape({batch, sequence, kvHeads, dim});
        RMSNorm(key, knorm, 1e-6f, nk);
        key.Reshape({batch, sequence, kvHeads * dim});
        nv.CopyFrom(value); nv.Reshape({batch, sequence, kvHeads, dim});
        if (mrope) {
            Qwen35InterleavedRope(rq, positions, rotary,
                rotary / 2 - sectionH - sectionW, sectionH, sectionW, 1000000.f, 1.f);
            Qwen35InterleavedRope(nk, positions, rotary,
                rotary / 2 - sectionH - sectionW, sectionH, sectionW, 1000000.f, 1.f);
        } else {
            RopeEncoding(rq, positions, rotary, 1000000.f, 1.f, true);
            RopeEncoding(nk, positions, rotary, 1000000.f, 1.f, true);
        }
        PermuteSelf(rq, {0, 2, 1, 3}); rq.Reshape({batch * qHeads, sequence, dim});
        PermuteSelf(nk, {0, 2, 1, 3}); nk.Reshape({batch * kvHeads, sequence, dim});
        PermuteSelf(nv, {0, 2, 1, 3}); nv.Reshape({batch * kvHeads, sequence, dim});
        Require(FastllmCudaQwen4KVAppend(nk, nv, offset, rk, rv), "reference append rejected");
    };
    for (int offset : {previous, previous - 3}) {
        reference(offset);
        const auto oldK = Read(kc, true), oldV = Read(vc, true);
        fused(offset);
        Equal(q, rq, "Q"); Equal(gate, rg, "gate");
        // Compare the entire allocation to also check untouched prefix/tail.
        Equal(kc, rk, "K cache", true); Equal(vc, rv, "V cache", true);
        auto untouched = [&](const Data &cache, const std::vector<unsigned char> &old) {
            const auto now = Read(cache, true);
            const int unit = type == FLOAT32 ? 4 : 2;
            for (int h = 0; h < batch * kvHeads; ++h) {
                for (int token = 0; token < cache.expansionDims[1]; ++token) {
                    if (token >= offset && token < offset + sequence) continue;
                    const auto index = (h * cache.strides[0] + token * dim) * unit;
                    Require(std::memcmp(now.data() + index, old.data() + index, dim * unit) == 0,
                            "fused preparation changed cache outside append interval");
                }
            }
        };
        untouched(kc, oldK); untouched(vc, oldV);
    }
    Data context(type), fusedOut(type), referenceOut(type);
    Init(context, {batch * qHeads, sequence, dim}, 31);
    Require(FastllmCudaQwen4AttentionOutput(context, gate, fusedOut), "fused output rejected");
    referenceOut.CopyFrom(context);
    PermuteSelf(referenceOut, {1, 0, 2});
    referenceOut.Reshape({sequence, batch, -1});
    PermuteSelf(referenceOut, {1, 0, 2});
    SigmoidMulTo(referenceOut, gate);
    Equal(fusedOut, referenceOut, "output gate");

    const auto before = Read(kc, true);
    Require(!FastllmCudaQwen4AttentionPrepare(qgate, key, value, qnorm, knorm,
        positions, q, gate, kc, vc, dim, rotary, sectionH, sectionW,
        1e-6f, 1000000.f, previous + sequence + 17), "invalid capacity accepted");
    Require(before == Read(kc, true), "fallback modified cache");

    std::printf("dtype=%d batch=%d seq=%d dim=%d mrope=%d %s PASS\n",
                (int)type, batch, sequence, dim, mrope,
                type == FLOAT32 ? "FP32_TOLERANCE" : "EXACT");
}

int main() {
    try {
        if (FastllmCudaGetDeviceCount() < 1) return 77;
        FastllmCudaSetDevice(0); SetThreads(2);
        for (auto type : {FLOAT16, BFLOAT16, FLOAT32}) {
            for (int dim : {32, 64, 96, 128, 192, 256, 384, 508}) {
                for (int seq : {1, 2, 5, 9, 17})
                    Run(type, 1, seq, dim, false);
            }
            Run(type, 2, 5, 256, false);
            Run(type, 1, 5, 256, true);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
    return 0;
}
