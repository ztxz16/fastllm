#pragma once

#include "devices/cpu/computeutils.h"
#include "gguf.h"
#include "exl3.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

// Negative linear calibration kinds encode EXL3's twice-bits value. Keep
// its actual decoder and worker pool, rather than timing an unrelated GGUF.
static std::string BenchmarkCpuExl3Threads(int hidden, int intermediate, int bits,
        const int *counts, int count, int budgetMs) {
    using namespace fastllm;
    using Clock=std::chrono::steady_clock;
    if(!Exl3BitRateSupported(bits) || hidden<128 || hidden>8192 || intermediate<128 ||
       intermediate>8192 || hidden%128 || intermediate%128) throw std::runtime_error("invalid EXL3 calibration shape");
    const auto start=Clock::now();
    auto elapsed=[&]{return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    const size_t pairBytes=2*Exl3PartBytes(intermediate,hidden,bits)+Exl3PartBytes(hidden,intermediate,bits);
    const int banks=std::clamp(int(((32ull<<20)+pairBytes-1)/pairBytes),2,8);
    std::vector<std::unique_ptr<Data>> weights;
    uint32_t random=9127;
    for(int bank=0;bank<banks;++bank) for(int part=0;part<2;++part) {
        auto w=std::make_unique<Data>(EXL3);
        w->blockK=bits; w->blockM=part?1:2;
        const int k=part?intermediate:hidden,n=part?hidden:intermediate;
        w->Resize({w->blockM*n,k}); w->Allocate(false);
        const size_t trellis=size_t(k)*n*bits/16, stride=Exl3PartBytes(n,k,bits);
        for(int projection=0;projection<w->blockM;++projection) {
            auto *data=w->cpuData+projection*stride;
            for(size_t at=0;at<trellis;at+=4) {
                random=random*1664525u+1013904223u; std::memcpy(data+at,&random,4);
            }
            auto *scales=reinterpret_cast<uint16_t *>(data+trellis);
            std::fill_n(scales,k+n,uint16_t(0x2800));
        }
        weights.push_back(std::move(w));
    }
    Data x(FLOAT32,{1,hidden}),downX(FLOAT32,{1,intermediate}),
        gateY(FLOAT32,{1,2*intermediate}),downY(FLOAT32,{1,hidden}),empty;
    for(auto *v:{&x,&downX}) {
        v->Allocate(false);
        for(int i=0;i<v->Count(0);++i) reinterpret_cast<float *>(v->cpuData)[i]=float(i%31-15)/32;
    }
    int next=0;
    auto pass=[&] {
        const int bank=next++%banks;
        Exl3LinearCPU(x,*weights[2*bank],empty,gateY);
        Exl3LinearCPU(downX,*weights[2*bank+1],empty,downY);
    };
    SetThreads(1);pass();
    const double setup=elapsed();
    std::ostringstream samples; int previous=0;
    for(int i=0;i<count;++i) {
        const int threads=counts[i];
        if(threads<=previous || threads>128 || elapsed()>=budgetMs) break;
        previous=threads; SetThreads(threads);pass();
        std::vector<double> times;
        for(int repeat=0;repeat<3;++repeat) {
            const auto begin=Clock::now();pass();
            times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
        }
        if(!std::isfinite(reinterpret_cast<float *>(downY.cpuData)[0])) throw std::runtime_error("nonfinite EXL3 calibration output");
        std::sort(times.begin(),times.end());
        if(i) samples<<',';
        samples<<'"'<<threads<<"\":"<<times[1];
    }
    std::ostringstream out;
    out<<"{\"samples_ms\":{"<<samples.str()<<"},\"setup_ms\":"<<setup<<",\"native_elapsed_ms\":"<<elapsed()
       <<",\"hidden\":"<<hidden<<",\"intermediate\":"<<intermediate<<",\"kernel_type\":\"exl3\",\"twice_bits\":"<<bits
       <<",\"weight_bytes\":"<<pairBytes*banks<<'}';
    return out.str();
}

// Only invoked by cpu_tuning.py's short-lived subprocess, so calibration's
// synthetic weights and cache state do not affect the inference process.
static std::string BenchmarkCpuLinearThreads(int hidden, int intermediate, int kind,
                                            const int *counts, int count, int budgetMs) {
    if (kind < 0) return BenchmarkCpuExl3Threads(hidden, intermediate, -kind, counts, count, budgetMs);
    using namespace fastllm;
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - started).count(); };
    hidden = (std::clamp(hidden, 256, 8192) + 255) / 256 * 256;
    intermediate = (std::clamp(intermediate, 256, 4096) + 255) / 256 * 256;
    const auto type = kind == GGML_TYPE_IQ2_XXS ? GGML_TYPE_IQ2_XXS : GGML_TYPE_Q4_K;
    const size_t pairBytes = ggml_row_size(type, hidden) * (2 * intermediate) +
                             ggml_row_size(type, intermediate) * hidden;
    const int banks = std::clamp(int(((32ull << 20) + pairBytes - 1) / pairBytes), 2, 8);
    std::vector<std::unique_ptr<Data>> weights;
    for (int bank = 0; bank < banks; ++bank) {
        for (int part = 0; part < 2; ++part) {
            auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, type,
                std::vector<int>{part ? hidden : 2 * intermediate, part ? intermediate : hidden});
            w->disableGGUFRepack = true; // canonical records, as in the disk expert cache
            w->Allocate();
            const size_t bytes = w->GetBytes(), blockBytes = ggml_type_size(type);
            std::memset(w->cpuData, 0x11 + bank, bytes);
            const uint16_t scale = 0x3c00; // finite, nonzero FP16 scale in each canonical block
            for (size_t off = 0; off < bytes; off += blockBytes)
                std::memcpy(w->cpuData + off, &scale, sizeof(scale));
            weights.push_back(std::move(w));
        }
    }
    std::vector<float> input(hidden), downInput(intermediate), output(std::max(hidden, 2 * intermediate));
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) / 32;
    for (size_t i = 0; i < downInput.size(); ++i) downInput[i] = float(int(i % 23) - 11) / 32;
    auto pool = std::make_unique<AliveThreadPool>(1);
    const double setupMs = elapsed();
    int nextBank = 0;
    auto pass = [&](int threads) {
        const int bank = nextBank++ % banks;
        RunLinearFloat32GGUF(input.data(), weights[bank * 2]->cpuData, output.data(), nullptr,
            weights[bank * 2].get(), 1, hidden, 2 * intermediate, pool.get(), 0, threads);
        RunLinearFloat32GGUF(downInput.data(), weights[bank * 2 + 1]->cpuData, output.data(), nullptr,
            weights[bank * 2 + 1].get(), 1, intermediate, hidden, pool.get(), 0, threads);
    };
    pass(1); // warm one expert's full shapes; do not time first-use dispatch
    std::ostringstream samples;
    bool first = true;
    int previous = 0;
    for (int i = 0; i < count; ++i) {
        const int threads = counts[i];
        if (threads <= previous || threads > 128 || elapsed() >= budgetMs) break;
        previous = threads;
        pool->ResizeThreads(threads);
        std::vector<double> times;
        for (int repeat = 0; repeat < (threads == 1 ? 1 : 3); ++repeat) {
            const auto begin = Clock::now();
            pass(threads);
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
            if (elapsed() >= budgetMs) break;
        }
        if (!std::isfinite(output[0])) throw std::runtime_error("nonfinite linear calibration output");
        std::sort(times.begin(), times.end());
        if (!first) samples << ',';
        first = false;
        samples << '"' << threads << "\":" << times[times.size() / 2];
    }
    std::ostringstream out;
    out << "{\"samples_ms\":{" << samples.str() << "},\"setup_ms\":" << setupMs
        << ",\"native_elapsed_ms\":" << elapsed()
        << ",\"hidden\":" << hidden << ",\"intermediate\":" << intermediate
        << ",\"gguf_type\":\"" << ggml_type_name(type)
        << "\",\"kernel_type\":\"" << ggml_type_name((ggml_type)weights[0]->ggmlType)
        << "\",\"weight_bytes\":" << pairBytes * banks << '}';
    return out.str();
}

#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"

// Match the selected host arithmetic profile, including activation conversion,
// Gate/Up, SwiGLU and Down. This runs only in the isolated calibration worker.
static std::string BenchmarkCpuMoeThreads(int hidden, int intermediate, int kind,
        int rows, int topk, const int *counts, int count, int budgetMs,
        bool standard = false, int downKind = -2) {
    using namespace fastllm;
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - started).count(); };
    const bool nvfp4 = standard && kind == -1 && downKind == -1;
    const auto type = static_cast<ggml_type>(kind);
    const auto downType = static_cast<ggml_type>(downKind == -2 ? kind : downKind);
    auto supported = [&](ggml_type t) {
        if (!standard) return t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS ||
            t == GGML_TYPE_Q4_K || t == GGML_TYPE_IQ1_M;
        switch (t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_0: case GGML_TYPE_Q5_0: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
            return true;
        default: return false;
        }
    };
    if (!nvfp4 && (!supported(type) || !supported(downType)))
        throw std::runtime_error("unsupported MoE calibration format");
    SetThreads(counts[count - 1]);
    const int nodes = GetNumasMoeShardCount();
    if (nodes < 1 || hidden % (nvfp4 ? 16 : ggml_blck_size(type)) ||
        intermediate % (nvfp4 ? 16 : ggml_blck_size(downType)) ||
        hidden % (4 * nodes) || intermediate % (4 * nodes))
        throw std::runtime_error("unsupported MoE calibration shard shape");
    const size_t pairBytes = nvfp4 ? size_t(3) * hidden * intermediate * 9 / 16 :
        ggml_row_size(type, hidden) * (2 * intermediate) + ggml_row_size(downType, intermediate) * hidden;
    // Rotate a substantial expert set instead of repeatedly timing one hot GEMV.
    const int experts = std::max(topk, std::min(standard ? 256 : 64, int(((512ull << 20) + pairBytes - 1) / pairBytes)));
    struct Weights {
        std::vector<std::unique_ptr<Data>> data;
        ~Weights() {
            ClearNumasMoeRuntimeCache();
            for (const auto &weight : data) ReleaseNumasMoeWeightArena(weight.get());
        }
    } owned;
    std::vector<Data *> weights((experts + 1) * 2, nullptr);
    std::vector<std::pair<Data *, std::string>> registration;
    for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
        const int columns = part ? intermediate : hidden;
        const int weightRows = part ? hidden : 2 * intermediate;
        const auto weightType = part ? downType : type;
        std::unique_ptr<Data> w;
        if (nvfp4) {
            w = std::make_unique<Data>(NVFP4_BLOCK_16_E4M3, std::vector<int>{weightRows, columns});
            w->blockK = 1; w->blockM = 16;
            w->Allocate();
            const size_t packed = GetNVFP4WeightBytes(weightRows, columns);
            for (size_t byte = 0; byte < packed; ++byte)
                w->cpuData[byte] = uint8_t(0x21 + (byte + expert) % 6);
            std::memset(w->cpuData + packed, 0x30, size_t(weightRows) * columns / 16);
            w->scales = part ? std::vector<float>{.03125f} : std::vector<float>{.03125f, .0625f};
        } else {
            w = std::make_unique<Data>(DATA_GGUF_FORMAT, weightType, std::vector<int>{weightRows, columns});
            // Standard host inference uses the NUMA backend's normal packing.
            w->disableGGUFRepack = !standard;
            w->Allocate();
            std::memset(w->cpuData, 0x11 + expert % 16, w->GetBytes());
            const uint16_t scale = 0x2400;
            for (size_t offset = 0; offset < w->GetBytes(); offset += ggml_type_size(weightType)) {
                if (weightType == GGML_TYPE_IQ1_M) {
                    auto *block = reinterpret_cast<block_iq1_m *>(w->cpuData + offset);
                    for (int j = 0; j < 4; ++j) {
                        const uint16_t packed = 0x111 | (((scale >> (4 * j)) & 15) << 12);
                        std::memcpy(block->scales + 2 * j, &packed, sizeof(packed));
                    }
                } else {
                    std::memcpy(w->cpuData + offset, &scale, sizeof(scale));
                }
            }
        }
        weights[(expert + 1) * 2 + part] = w.get();
        registration.emplace_back(w.get(), part ? "linear" : "linearSwiglu");
        owned.data.push_back(std::move(w));
    }
    RegisterNumasMoeWeightBatch(registration);
    if (!CanRunNumasMoeDecodeExperts(weights.data(), weights.size()))
        throw std::runtime_error("NUMA MoE calibration path unavailable");
    std::vector<uint16_t> input(size_t(rows) * hidden);
    std::vector<float> standardInput(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float value = float(int(i % 31) - 15) / 32;
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
        input[i] = uint16_t(bits >> 16); // exactly representable BF16 and FP16 values
        standardInput[i] = value;
    }
    std::vector<int32_t> indices(rows * topk), gpuIndices(rows * topk, -1);
    std::vector<float> scores(rows * topk, 1.f / topk);
    std::vector<float> output(size_t(rows) * topk * hidden), reference;
    auto pass = [&](int step) {
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k)
            indices[row * topk + k] = (step * topk + row + k) % experts;
        if (standard) NumasMoeDecodeExpertsBatch(standardInput.data(), output.data(), rows,
            weights.data(), weights.size(), indices.data(), gpuIndices.data(), scores.data(), topk, 0);
        else NumasMoeVerifyExperts(input.data(), output.data(), rows, weights.data(), weights.size(),
            indices.data(), gpuIndices.data(), scores.data(), topk, 0, 7.f, true, 32);
    };
    const double setupMs = elapsed();
    std::ostringstream samples;
    bool first = true;
    double maxL2 = 0;
    int nextStep = 1;
    for (int i = 0; i < count; ++i) {
        const int threads = counts[i];
        if (threads < nodes || threads % nodes || (i && threads <= counts[i - 1]))
            throw std::runtime_error("invalid NUMA thread candidate");
        SetThreads(threads);
        pass(0); // bind resized workers and warm dispatch outside the timer
        std::vector<double> times;
        for (int repeat = 0; repeat < 5; ++repeat) {
            if (elapsed() - setupMs >= budgetMs)
                throw std::runtime_error("incomplete NUMA calibration samples");
            const auto begin = Clock::now();
            pass(standard ? nextStep++ : repeat + 1);
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
        }
        if (standard) pass(0); // same untimed correctness probe at every thread count
        double error = 0, norm = 0, outputNorm = 0;
        for (size_t j = 0; j < output.size(); ++j) {
            if (!std::isfinite(output[j])) throw std::runtime_error("nonfinite NUMA calibration output");
            outputNorm += double(output[j]) * output[j];
            if (!reference.empty()) {
                const double delta = double(output[j]) - reference[j];
                error += delta * delta; norm += double(reference[j]) * reference[j];
            }
        }
        if (!(outputNorm > 0)) throw std::runtime_error("empty NUMA calibration output");
        if (reference.empty()) reference = output;
        const double l2 = std::sqrt(error / std::max(norm, 1e-30));
        if (l2 > 1e-6) throw std::runtime_error("NUMA thread count changed calibration output");
        maxL2 = std::max(maxL2, l2);
        std::sort(times.begin(), times.end());
        if (!first) samples << ',';
        first = false;
        samples << '"' << threads << "\":" << times[times.size() / 2];
    }
    std::ostringstream out;
    out << "{\"samples_ms\":{" << samples.str() << "},\"setup_ms\":" << setupMs
        << ",\"native_elapsed_ms\":" << elapsed() << ",\"hidden\":" << hidden
        << ",\"intermediate\":" << intermediate << ",\"numa_nodes\":" << nodes
        << ",\"rows\":" << rows << ",\"topk\":" << topk << ",\"experts\":" << experts
        << ",\"math_profile\":\"" << (standard ? "standard" : "deepseek_v41")
        << "\",\"gguf_type\":\"" << (nvfp4 ? "nvfp4_e4m3" : ggml_type_name(type))
        << "\",\"down_type\":\"" << (nvfp4 ? "nvfp4_e4m3" : ggml_type_name(downType))
        << "\",\"weight_bytes\":" << pairBytes * experts
        << ",\"thread_output_l2\":" << maxL2 << '}';
    return out.str();
}
#endif
