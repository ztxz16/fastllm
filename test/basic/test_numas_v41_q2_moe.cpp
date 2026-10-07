#include "fastllm.h"
#include "executor.h"
#include "devices/numas/numasdevice.h"
#include "utils.h"
#include "gguf.h"
#include "devices/disk/diskdevice.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unistd.h>

using namespace fastllm;
namespace fastllm {
    void RegisterNumas(Data *data, std::string weightType);
}

static float Bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Scalar E4M3 oracle, independent of FastLLM's activation quantizer.
static void Fp8(std::vector<float> &values) {
    for (size_t start = 0; start < values.size(); start += 32) {
        float maximum = 1e-4f;
        for (size_t i = start; i < start + 32; ++i) maximum = std::max(maximum, std::abs(values[i]));
        float scale = std::exp2(std::ceil(std::log2(maximum / 448.f)));
        for (size_t i = start; i < start + 32; ++i) {
            float x = std::abs(values[i]) / scale, best = 0, distance = x;
            int previous = 0;
            for (int code = 1; code <= 126; ++code) {
                int exponent = code >> 3, mantissa = code & 7;
                float candidate = exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7) : mantissa / 512.f;
                float delta = std::abs(candidate - x);
                if (delta < distance || (delta == distance && !(code & 1) && (previous & 1))) {
                    best = candidate; distance = delta; previous = code;
                }
            }
            values[i] = std::copysign(best * scale, values[i]);
        }
    }
}

static void GgmlActivation(std::vector<float> &values, const Data &weight) {
    // Decode canonical Q8 blocks for a scalar matmul oracle. K32 changes
    // partial-sum metadata, while its scale and quant bytes keep this layout.
    std::vector<block_q8_K> packed(values.size() / QK_K);
    auto type = (ggml_type)weight.ggmlType;
    iqk_quantize_row_q8_K(values.data(), packed.data(), values.size(), ggml_type_vec_dot_type(type), type);
    for (size_t i = 0; i < values.size(); ++i) values[i] = packed[i / QK_K].d * packed[i / QK_K].qs[i % QK_K];
}

struct DiskFixtureFile {
    std::string path;
    DiskFixtureFile() {
        char name[] = "/tmp/fastllm-v41-q2-XXXXXX";
        int fd = mkstemp(name);
        if (fd < 0) throw std::runtime_error("disk fixture creation failed");
        close(fd); path = name;
    }
    ~DiskFixtureFile() { unlink(path.c_str()); }
};

struct RestoreEnv {
    const char *key;
    bool present;
    std::string value;
    explicit RestoreEnv(const char *key) : key(key) {
        const char *saved = std::getenv(key);
        present = saved != nullptr;
        if (saved) value = saved;
    }
    ~RestoreEnv() {
        if (present) setenv(key, value.c_str(), 1);
        else unsetenv(key);
    }
};

int main(int argc, char **argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        const bool disk = mode == "disk", cache = mode == "cache", host = mode == "host";
        const bool decode = mode == "decode", verify = mode == "verify";
        const bool decodeOrVerify = decode || verify;
        // Exercise the row-wise decode path even for adversarial multi-row
        // fixtures; otherwise 2..8 rows take a separate grouped implementation.
        if (decode) setenv("FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE", "1", 1);
        const bool cuda = argc > 3 && std::string(argv[3]) == "cuda";
        DiskFixtureFile fixture;
        if (disk) {
            FILE *file = fopen(fixture.path.c_str(), "ab");
            if (!file) throw std::runtime_error("disk fixture prefix open failed");
            const uint8_t prefix[13] = {};
            fwrite(prefix, 1, sizeof(prefix), file);
            fclose(file);
        }
        EnableAMX(false);
        SetThreads(4);
        ((Executor*)GetExecutor())->SetFirstDevice(disk ? "disk" : "numa");
        if (disk) { SetMoeCpuCacheBytes(4 * 1024 * 1024); SetMoeCudaCacheBytes(0); }
        const int hidden = argc > 2 ? std::stoi(argv[2]) : 256;
        const int inter = (host || decodeOrVerify) && argc > 3 ? std::stoi(argv[3]) : 256;
        const int experts = cache ? 24 : decodeOrVerify ? 6 : 3, topk = (cache || decodeOrVerify) ? 6 : 2;
        // Q4_K_R4 consumes Q8_K32, while Q2_K_R4 consumes Q8_K.
        const bool q2Down = decodeOrVerify && argc > 4 && std::string(argv[4]) == "q2";
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<Data*> weights(2 * (experts + 1), nullptr), biases(weights);
        std::vector<std::vector<float>> decoded(weights.size());
        std::vector<std::vector<uint8_t>> canonical(weights.size());
        for (int e = disk ? 0 : 1; e <= experts; ++e) for (int part = 0; part < 2; ++part) {
            int rows = part ? hidden : 2 * inter, cols = part ? inter : hidden;
            auto type = part && !q2Down ? GGML_TYPE_Q4_K : GGML_TYPE_Q2_K;
            auto weight = e == 0 ? std::make_unique<Data>(FLOAT16, std::vector<int>{rows, cols}) :
                std::make_unique<Data>(DATA_GGUF_FORMAT, type, std::vector<int>{rows, cols});
            std::vector<float> original(rows * cols);
            for (int i = 0; i < rows * cols; ++i) original[i] = .03f * std::sin(i * .719f + e * .43f + part);
            weight->CreateFromOriData(WeightType::LINEAR, FLOAT32, (uint8_t*)original.data(), nullptr, nullptr);
            if (host) canonical[e*2+part].assign(weight->cpuData, weight->cpuData+weight->GetBytes());
            auto &plain = decoded[e * 2 + part];
            plain.resize(original.size());
            if (e == 0) for (size_t i = 0; i < plain.size(); ++i)
                plain[i] = half_to_float(((uint16_t*)weight->cpuData)[i]);
            else ggml_type_to_float(type)(weight->cpuData, plain.data(), plain.size());
            if (disk) {
                FILE *file = fopen(fixture.path.c_str(), "ab");
                if (!file) throw std::runtime_error("disk fixture open failed");
                DiskWeightPart part;
                part.fileName = fixture.path; part.fileOffset = ftell(file);
                part.bytes = weight->GetBytes(); part.sourceDataType = weight->dataType;
                part.dims = weight->dims;
                size_t written;
                if (e > 0 && type == GGML_TYPE_Q2_K) {
                    // An odd file prefix plus different inter-projection
                    // gaps exercises both in-place shifts and preservation of
                    // the preceding projection's tail. The final weight also
                    // ends in a partial O_DIRECT sector.
                    part.bytes /= 2;
                    part.dims[0] /= 2;
                    weight->diskWeightParts.push_back(part);
                    written = fwrite(weight->cpuData, 1, part.bytes, file);
                    const std::vector<uint8_t> gap(4096 + (e == 2 ? 24 : e == 3 ? 1 : 0));
                    if (fwrite(gap.data(), 1, gap.size(), file) != gap.size()) {
                        fclose(file);
                        throw std::runtime_error("disk fixture gap write failed");
                    }
                    part.fileOffset = ftell(file);
                    written += fwrite(weight->cpuData + part.bytes, 1, part.bytes, file);
                } else written = fwrite(weight->cpuData, 1, part.bytes, file);
                fclose(file);
                if (written != weight->GetBytes()) throw std::runtime_error("disk fixture write failed");
                delete[] weight->cpuData; weight->cpuData = nullptr;
                weight->isDiskWeight = true;
                weight->diskWeightParts.push_back(part);
            }
            weights[e * 2 + part] = weight.get();
            owned.push_back(std::move(weight));
        }
        Data output(BFLOAT16), w1, w2, w3, currentInput, currentOutput;
#ifdef USE_CUDA
        if (cache) {
            FastllmCudaSetDevice(0);
            const size_t gateBytes = weights[2]->GetBytes();
            const size_t downOffset = (gateBytes + 15) & ~size_t(15);
            const size_t stride = (downOffset + weights[3]->GetBytes() + 127) & ~size_t(127);
            SetMoeCudaCacheBytes(16 * stride);
            FastllmCudaMoeCacheLayer layer{weights.data(), (int)weights.size(), true, 10.f};
            if (!FastllmCudaPrepareMoeCache(&layer, 1, [&] {
                for (int e = 1; e <= experts; ++e) {
                    RegisterNumas(weights[e * 2], "linearSwiglu");
                    RegisterNumas(weights[e * 2 + 1], "linear");
                }
            })) throw std::runtime_error("Q2 CUDA cache preparation failed");
        }
#else
        if (cache) throw std::runtime_error("Q2 CUDA cache test requires CUDA");
#endif
        const std::vector<int> batches = host && argc > 4 ? std::vector<int>{std::stoi(argv[4])} :
            // Straddle wide-MMQ admission and exercise expert tails that are
            // neither 16- nor 64-row aligned.
            host ? std::vector<int>{33, 65, 128, 408, 1023, 1024, 1041, 4096} :
            decode ? std::vector<int>{1, 3} :
            verify ? std::vector<int>{1, 2, 3, 6, 8} :
            std::vector<int>{1, 7, 32, 64, 260, 2, 3, 6, 8};
        for (bool quantizeShared : {false, true}) for (int rows : batches) {
            if ((cache || host || decodeOrVerify) && quantizeShared) continue;
            std::vector<float> source(rows * hidden);
            std::vector<uint16_t> inputBits(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                source[i] = Bf16(std::sin(i * .31f + rows) * std::ldexp(1.f, int(i / 32 % 4)));
                if (cache && rows == 2 && i >= size_t(hidden)) source[i] = 0;
                if (cache && rows == 6) source[i] = Bf16(source[i] * 32.f);
                if (host && i / hidden == 1) source[i] = 0;
                if (host && rows == 128) source[i] = Bf16(source[i] * 32.f);
                if (decodeOrVerify && i / hidden == 1) source[i] = 0;
                if (decodeOrVerify && i / hidden == 2) source[i] = Bf16(source[i] * 32.f);
                if (host && rows == 65 && i/hidden%7 == 3 && i%256 < 2)
                    source[i] = i%256 == 0 ? 32.f : -32.f; // signed-maximum tie
                uint32_t bits; memcpy(&bits, &source[i], sizeof(bits)); inputBits[i] = bits >> 16;
            }
            std::vector<int32_t> ids(rows * topk);
            std::vector<float> scores(ids.size());
            for (int r = 0; r < rows; ++r) for (int k = 0; k < topk; ++k) {
                ids[r * topk + k] = (r + k + (cache ? rows * 3 : 0)) % experts;
                scores[r * topk + k] = k ? .375f : .625f;
                if (cache && rows == 3 && r == 1) scores[r * topk + k] = 0;
                if (host && r%7 == 0) scores[r * topk + k] = 0;
                if (host && r%7 == 2) scores[r * topk + k] *= -1;
                if (decodeOrVerify && r == 2 && k % 2) scores[r * topk + k] *= -1;
            }
            Data input(BFLOAT16, {rows, hidden}, DataDevice::CPU, inputBits.data());
#ifdef USE_CUDA
            if (cuda) {
                FastllmCudaSetDevice(0);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                input.ToDevice(DataDevice::CPU);
            }
#endif
            if (disk) { SetMoeCpuCacheBytes(1); SetMoeCpuCacheBytes(4 * 1024 * 1024); }
            Data index(INT32, {rows, topk}, DataDevice::CPU, ids.data());
            Data score(FLOAT32, {rows, topk}, DataDevice::CPU, scores.data());
            MergeMOE(input, index, score, weights, biases, w1, w2, w3, currentInput, currentOutput,
                     0.7f, output, 0, MoeGateSwiglu, false, 10.f, true, nullptr, 32, quantizeShared);
            // GPU-assisted prefill may return a CUDA tensor. The numerical
            // references below always read the host representation.
            output.ToDevice(DataDevice::CPU);
            if (verify) {
                // The cache verifier uses the grouped path even for one row.
                // Compare its scalar fallback exactly, including untouched GPU
                // route slots, signed routing and non-BF16 clamp thresholds.
                RestoreEnv fast("FASTLLM_DSV4_DISABLE_NUMAS_MOE_LARGE_FAST");
                for (bool perRoute : {false, true}) for (bool subset : {false, true}) {
                    if (subset && !perRoute) continue;
                    std::vector<int32_t> gpuIds(ids.size(), -1);
                    if (subset) for (size_t i = 0; i < ids.size(); ++i)
                        if (ids[i] % 2) gpuIds[i] = ids[i];
                    const size_t bytes = size_t(rows) * hidden *
                        (perRoute ? topk * sizeof(float) : sizeof(uint16_t));
                    std::vector<uint8_t> reference(bytes), actual(bytes);
                    for (float limit : {0.f, 1.234567f, 10.f}) {
                        auto run = [&](std::vector<uint8_t> &buffer) {
                            std::fill(buffer.begin(), buffer.end(), 0x5a);
                            NumasMoeVerifyExperts(inputBits.data(), buffer.data(), rows,
                                weights.data(), weights.size(), ids.data(), gpuIds.data(),
                                scores.data(), topk, 0, limit, perRoute);
                        };
                        setenv(fast.key, "1", 1); run(reference);
                        unsetenv(fast.key); run(actual);
                        if (reference != actual)
                            throw std::runtime_error("Q2 cache verifier differs from scalar fallback");
                    }
                }
                // The same preparation capability also admits V4 block-128
                // activations. Exercise that public MergeMOE path separately
                // from the V4.1 cache verifier's fixed block-32 boundary.
                for (DataType type : {BFLOAT16, FLOAT32}) {
                    Data check(type);
                    auto merge = [&] {
                        MergeMOE(input, index, score, weights, biases, w1, w2, w3,
                                 currentInput, currentOutput, .7f, check, 0, MoeGateSwiglu,
                                 false, 1.234567f, true, nullptr, 128, false);
                    };
                    setenv(fast.key, "1", 1); merge();
                    std::vector<uint8_t> reference(check.cpuData, check.cpuData + check.GetBytes());
                    unsetenv(fast.key); merge();
                    if (memcmp(reference.data(), check.cpuData, reference.size()))
                        throw std::runtime_error("Q2 grouped block-128 differs from scalar fallback");
                }
                printf("Q2 verifier exact: rows=%d hidden=%d inter=%d down=%s\n",
                       rows, hidden, inter, q2Down ? "Q2" : "Q4");
                continue;
            }
            if (decode) {
                // Compare the complete optimized operator byte-for-byte with
                // its scalar fallback, including non-BF16 clamp thresholds,
                // zero inputs, signed routes and both FP8 block sizes.
                RestoreEnv fast("FASTLLM_DSV4_DISABLE_NUMAS_MOE_FAST"),
                           taskCache("FASTLLM_DSV4_DISABLE_NUMAS_MOE_TASK_CACHE");
                for (int block : {32, 128}) for (float limit : {0.f, 1.234567f, 10.f}) {
                    for (DataType outputType : {BFLOAT16, FLOAT32}) {
                        Data check(outputType);
                        auto merge = [&] {
                            MergeMOE(input, index, score, weights, biases, w1, w2, w3,
                                     currentInput, currentOutput, .7f, check, 0, MoeGateSwiglu,
                                     false, limit, true, nullptr, block, quantizeShared);
                        };
                        setenv(fast.key, "1", 1);
                        merge();
                        const std::vector<uint8_t> reference(check.cpuData, check.cpuData + check.GetBytes());
                        unsetenv(fast.key);
                        for (bool cached : {true, false}) {
                            if (cached) unsetenv(taskCache.key);
                            else setenv(taskCache.key, "1", 1);
                            merge();
                            if (memcmp(reference.data(), check.cpuData, reference.size()))
                                throw std::runtime_error("Q2 decode fast/scalar outputs differ: block=" +
                                    std::to_string(block) + " limit=" + std::to_string(limit) +
                                    " output=" + GetDataTypeName(outputType) +
                                    " task_cache=" + std::to_string(cached));
                        }
                    }
                }
                printf("Q2 decode fast/scalar bitwise PASS rows=%d inter=%d\n", rows, inter);
            }
#ifdef USE_CUDA
            if (host) {
                const std::vector<uint8_t> reference(output.cpuData, output.cpuData + output.GetBytes());
                std::vector<std::vector<uint8_t>> packed(weights.size());
                for (size_t i = 2; i < weights.size(); ++i) {
                    auto *w = weights[i];
                    const auto *p = w->cpuData ? w->cpuData : w->numasData.at(0);
                    packed[i].assign(p, p+w->GetBytes());
                }
                for (int device = 0; device < FastllmCudaGetDeviceCount(); ++device) {
                    FastllmCudaSetDevice(device);
                    Data gpuInput(input), gate, workspace, gpuOutput;
                    gpuInput.ToDevice(DataDevice::CUDA, std::vector<int>{device}, true);
                    std::unordered_set<int> selected{1,2,3};
                    auto launch = [&](bool v4 = true, int block = 32) {
                        return FastllmCudaMergeMOEGGUFHost(gpuInput, gate, workspace, gpuOutput,
                            weights.data(), experts, ids.data(), scores.data(), topk, selected, true, v4, 10.f, block);
                    };
                    // Block 128 is now supported by the separate GLM path.
                    // Keep testing an unsupported activation mode here.
                    if (launch(false) || launch(true, 64) || gpuOutput.cudaData)
                        throw std::runtime_error("unsupported grouped math changed output");
                    for (int rejected : {32, 4097}) {
                        gpuInput.Resize({rejected, hidden});
                        if (launch() || gpuOutput.cudaData) throw std::runtime_error("grouped row capability check failed");
                    }
                    gpuInput.Resize({rows, hidden});
                    gpuInput.dataType = FLOAT16;
                    if (launch() || gpuOutput.cudaData) throw std::runtime_error("V4.1 grouped admitted FP16 input");
                    gpuInput.dataType = BFLOAT16;
                    selected.insert(0);
                    if (launch()) throw std::runtime_error("V4.1 grouped admitted shared expert");
                    selected.erase(0);
                    if (!launch()) throw std::runtime_error("V4.1 grouped rejected Q2_K/Q4_K prefill");
                    gpuOutput.ToDevice(DataDevice::CPU);
                    const std::vector<uint8_t> grouped(gpuOutput.cpuData, gpuOutput.cpuData+gpuOutput.GetBytes());
                    double error2 = 0, reference2 = 0;
                    for (size_t i = 0; i < reference.size()/2; ++i) {
                        float a = BFloat16BitsToFloat32(((uint16_t*)gpuOutput.cpuData)[i]);
                        float b = BFloat16BitsToFloat32(((const uint16_t*)reference.data())[i]);
                        if (!std::isfinite(a)) throw std::runtime_error("nonfinite grouped output");
                        error2 += (a-b)*(a-b); reference2 += b*b;
                    }
                    const double relative = std::sqrt(error2/std::max(reference2, 1e-20));
                    printf("Q2/Q4 V4.1 grouped device=%d rows=%d CPU_relative_rms=%.6f\n", device, rows, relative);
                    if (relative > .005) throw std::runtime_error("V4.1 grouped differs from NUMA reference");
                    if (rows == 65) {
                        for (bool ordinary : {false, true}) {
                            std::vector<std::unique_ptr<Data>> ownedHost;
                            std::vector<Data*> hostWeights(weights.size(), nullptr);
                            std::vector<uint8_t*> pointers;
                            for (size_t i = 2; i < weights.size(); ++i) {
                                auto *w = weights[i];
                                auto clone = std::make_unique<Data>(DATA_GGUF_FORMAT,
                                    ordinary ? (i%2 ? GGML_TYPE_Q4_K : GGML_TYPE_Q2_K) : w->ggmlType, w->dims);
                                clone->Allocate(false);
                                const auto &bytes = ordinary ? canonical[i] : packed[i];
                                memcpy(clone->cpuData, bytes.data(), bytes.size());
                                pointers.push_back(clone->cpuData);
                                clone->numasData = {clone->cpuData, clone->cpuData+bytes.size()/2};
                                clone->cpuData = nullptr;
                                hostWeights[i] = clone.get(); ownedHost.push_back(std::move(clone));
                            }
                            const bool ok = FastllmCudaMergeMOEGGUFHost(gpuInput, gate, workspace, gpuOutput,
                                hostWeights.data(), experts, ids.data(), scores.data(), topk, selected, !ordinary, true, 10.f, 32);
                            for (size_t i = 0; i < ownedHost.size(); ++i) {
                                ownedHost[i]->cpuData = pointers[i]; ownedHost[i]->numasData.clear();
                            }
                            if (!ok) throw std::runtime_error("V4.1 grouped rejected canonical/R4 NUMA shards");
                            gpuOutput.ToDevice(DataDevice::CPU);
                            if (memcmp(gpuOutput.cpuData, grouped.data(), grouped.size()))
                                throw std::runtime_error("V4.1 grouped canonical/R4 shards disagree");
                        }
                        std::vector<float> partial(rows*hidden, 0);
                        for (const auto &subset : std::vector<std::unordered_set<int>>{{1,3}, {2}}) {
                            selected = subset;
                            if (!launch()) throw std::runtime_error("V4.1 grouped subset rejected");
                            gpuOutput.ToDevice(DataDevice::CPU);
                            for (size_t i = 0; i < partial.size(); ++i)
                                partial[i] += BFloat16BitsToFloat32(((uint16_t*)gpuOutput.cpuData)[i]);
                        }
                        double diff = 0, norm = 0;
                        for (size_t i = 0; i < partial.size(); ++i) {
                            const float ref = BFloat16BitsToFloat32(((const uint16_t*)grouped.data())[i]);
                            diff += (partial[i]-ref)*(partial[i]-ref); norm += ref*ref;
                        }
                        if (std::sqrt(diff/std::max(norm, 1e-20)) > .005)
                            throw std::runtime_error("V4.1 grouped expert subsets disagree");
                    }
                    // Keep the last GPU result for the independent scalar oracle below.
                    memcpy(output.cpuData, grouped.data(), reference.size());
                    gpuInput.ToDevice(DataDevice::CPU);
                    if (memcmp(gpuInput.cpuData, inputBits.data(), inputBits.size()*2))
                        throw std::runtime_error("V4.1 grouped changed input");
                }
                for (size_t i = 2; i < weights.size(); ++i) {
                    auto *w = weights[i];
                    const auto *p = w->cpuData ? w->cpuData : w->numasData.at(0);
                    if (memcmp(p, packed[i].data(), packed[i].size()))
                        throw std::runtime_error("V4.1 grouped changed packed host weights");
                }
            }
            if (cache && rows <= 8) {
                const std::vector<uint8_t> cpuReference(output.cpuData, output.cpuData + output.GetBytes());
                Data gpuInput(input), gpuIndex(index), gpuScore(score), gpuOutput(BFLOAT16);
                gpuInput.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                gpuIndex.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                gpuScore.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", "6", 1);
                // Prime every route used by this verifier through cold decode
                // refills; the sixteen-slot budget forces eviction across cases.
                for (int r = 0; r < rows; ++r) {
                    Data x(BFLOAT16, {1, hidden}, DataDevice::CPU, inputBits.data() + r * hidden);
                    Data idsOne(INT32, {1, topk}, DataDevice::CPU, ids.data() + r * topk);
                    Data scoresOne(FLOAT32, {1, topk}, DataDevice::CPU, scores.data() + r * topk);
                    x.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                    idsOne.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                    scoresOne.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
                    if (!FastllmCudaMergeMOEHybrid(x, idsOne, scoresOne, gpuOutput,
                            weights.data(), (int)weights.size(), 0))
                        throw std::runtime_error("Q2 cold CUDA decode rejected");
                }
                for (const char *split : {"0", "1", "6"}) {
                    setenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", split, 1);
                    if (!FastllmCudaMergeMOEHybrid(gpuInput, gpuIndex, gpuScore, gpuOutput,
                            weights.data(), (int)weights.size(), 0)) {
                        if (rows == 1 || std::string(split) != "0")
                            throw std::runtime_error("Q2 CUDA hybrid request rejected");
                        continue; // CPU-only verifier intentionally uses its fallback.
                    }
                    gpuOutput.ToDevice(DataDevice::CPU);
                    if (std::string(split) == "0" && memcmp(gpuOutput.cpuData, cpuReference.data(), cpuReference.size()))
                        throw std::runtime_error("Q2 cache CPU subset differs from NUMA");
                    double error = 0, norm = 0;
                    for (int i = 0; i < rows * hidden; ++i) {
                        const float actual = BFloat16BitsToFloat32(((uint16_t*)gpuOutput.cpuData)[i]);
                        const float expected = BFloat16BitsToFloat32(((const uint16_t*)cpuReference.data())[i]);
                        if (!std::isfinite(actual)) throw std::runtime_error("non-finite Q2 cached output");
                        error += (actual - expected) * (actual - expected); norm += expected * expected;
                    }
                    printf("Q2 CUDA cache rows=%d split=%s relative_rms=%.6f\n", rows, split, std::sqrt(error / std::max(norm, 1e-20)));
                    if (error > .000025 * std::max(norm, 1e-20))
                        throw std::runtime_error("Q2 CUDA hybrid differs from NUMA oracle");
                }
                // The existing independent scalar oracle below checks cached
                // output as well as the direct NUMA path.
                output.CopyFrom(gpuOutput);
            }
#endif
            if (disk) {
                const std::vector<uint8_t> directOutput(output.cpuData, output.cpuData + output.GetBytes());
                auto before = GetDiskMoeCacheStats();
                MergeMOE(input, index, score, weights, biases, w1, w2, w3, currentInput, currentOutput,
                         0.7f, output, 0, MoeGateSwiglu, false, 10.f, true, nullptr, 32, quantizeShared);
                auto after = GetDiskMoeCacheStats();
                if (after.diskBytes != before.diskBytes || after.cpuHits <= before.cpuHits ||
                    after.cpuBytes > GetMoeCpuCacheBytes())
                    throw std::runtime_error("Q2 disk expert cache hit/budget failed");
                if (memcmp(directOutput.data(), output.cpuData, directOutput.size()) != 0)
                    throw std::runtime_error("Q2 disk cold/warm outputs differ");
                SetMoeCpuCacheBytes(1); SetMoeCpuCacheBytes(4 * 1024 * 1024);
                const char *oldOverride = std::getenv("FASTLLM_DISK_NO_CACHE");
                const bool hadOverride = oldOverride != nullptr;
                const std::string savedOverride = hadOverride ? oldOverride : "";
                setenv("FASTLLM_DISK_NO_CACHE", "0", 1);
                MergeMOE(input, index, score, weights, biases, w1, w2, w3, currentInput, currentOutput,
                         0.7f, output, 0, MoeGateSwiglu, false, 10.f, true, nullptr, 32, quantizeShared);
                if (hadOverride) setenv("FASTLLM_DISK_NO_CACHE", savedOverride.c_str(), 1);
                else unsetenv("FASTLLM_DISK_NO_CACHE");
                if (memcmp(directOutput.data(), output.cpuData, directOutput.size()) != 0)
                    throw std::runtime_error("Q2 disk aligned/buffered read outputs differ");
            }
            double error2 = 0, reference2 = 0;
            for (int r = 0; r < rows; ++r) {
                std::vector<float> total(hidden, 0);
                for (int k = 0; k < topk + (disk ? 1 : 0); ++k) {
                    int e = k == topk ? 0 : ids[r * topk + k] + 1;
                    std::vector<float> x(source.begin() + r * hidden, source.begin() + (r + 1) * hidden);
                    if (e || quantizeShared) Fp8(x);
                    if (e) GgmlActivation(x, *weights[e * 2]);
                    std::vector<float> middle(inter);
                    for (int j = 0; j < inter; ++j) {
                        float gate = 0, up = 0;
                        for (int c = 0; c < hidden; ++c) {
                            gate += decoded[e * 2][j * hidden + c] * x[c];
                            up += decoded[e * 2][(j + inter) * hidden + c] * x[c];
                        }
                        gate = Bf16(gate); up = Bf16(up);
                        if (e) { gate = std::min(gate, 10.f); up = std::max(-10.f, std::min(up, 10.f)); }
                        const float h = (gate / (1.f + std::exp(-gate))) * up;
                        middle[j] = Bf16((e ? scores[r * topk + k] : 1.f) * h);
                    }
                    if (e || quantizeShared) Fp8(middle);
                    if (e) GgmlActivation(middle, *weights[e * 2 + 1]);
                    for (int c = 0; c < hidden; ++c) {
                        float value = 0;
                        for (int j = 0; j < inter; ++j) value += decoded[e * 2 + 1][c * inter + j] * middle[j];
                        total[c] += Bf16(value) * (e ? 1.f : .7f);
                    }
                }
                for (int c = 0; c < hidden; ++c) {
                    uint32_t bits = uint32_t(((uint16_t*)output.cpuData)[r * hidden + c]) << 16;
                    float actual; memcpy(&actual, &bits, sizeof(actual));
                    float expected = Bf16(total[c]);
                    if (!std::isfinite(actual)) throw std::runtime_error("non-finite Q2/Q4 MoE output");
                    error2 += (actual - expected) * (actual - expected); reference2 += expected * expected;
                }
            }
            double relative = std::sqrt(error2 / std::max(reference2, 1e-20));
            printf("Q2/Q4 V4.1 %s rows=%d quant_shared=%d relative_rms=%.6f\n", disk ? "disk" : host ? "CUDA grouped" : "NUMA", rows, quantizeShared, relative);
            if (relative > (host ? .005 : .01)) throw std::runtime_error("Q2/Q4 MoE differs from scalar activation/GEMM oracle");
        }
#ifdef USE_CUDA
        if (cache) {
            uint64_t stats[5]{};
            if (!fastllm_moe_cuda_cache_stats(0, stats, false) || !stats[0] || !stats[1] ||
                !stats[2] || stats[2] > GetMoeCudaCacheBytes() || stats[3] != 16)
                throw std::runtime_error("Q2 cache residency/refill/budget failed");
            FastllmCudaReleaseMoeCache(weights.data(), (int)weights.size());
            if (!fastllm_moe_cuda_cache_stats(0, stats, false) || stats[2])
                throw std::runtime_error("Q2 cache release failed");
            unsetenv("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS");
        }
#endif
        ClearNumasMoeRuntimeCache();
        puts("ALL_PASS");
        return 0;
    } catch (const std::exception &error) {
        ClearNumasMoeRuntimeCache();
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
