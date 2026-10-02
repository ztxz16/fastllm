#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "gguf.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>
#include <thread>
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#include "devices/cpu/computeutils.h"
namespace fastllm { void RegisterNumas(Data *, std::string); }
#endif

static void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cuda(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> static T Cast(float f) { return T(f); }
static void Gpu(fastllm::Data &d) { d.dataDevice = fastllm::CUDA; d.dataDeviceIds = {0}; d.Allocate(false); }

static std::unique_ptr<fastllm::Data> Weight(ggml_type type, int rows, int cols, int seed, bool ordinaryGGUF = false) {
    const auto dtype = type == GGML_TYPE_F32 ? fastllm::FLOAT32 :
        type == GGML_TYPE_F16 ? fastllm::FLOAT16 : type == GGML_TYPE_BF16 ? fastllm::BFLOAT16 : fastllm::DATA_GGUF_FORMAT;
    auto d = std::make_unique<fastllm::Data>(ordinaryGGUF ? fastllm::DATA_GGUF_FORMAT : dtype);
    d->isGGUFData = true;
    d->ggmlType = type; d->Resize({rows, cols}); d->Allocate(false);
    unsigned state = 937 + seed;
    for (size_t i = 0; i < d->GetBytes(); ++i) {
        state = state * 1664525U + 1013904223U;
        d->cpuData[i] = state >> 24;
    }
    const size_t size = ggml_type_size(type);
    for (size_t b = 0; b < d->GetBytes() / size; ++b) {
        uint8_t *p = d->cpuData + b * size;
        uint16_t scale = __half_as_ushort(__float2half_rn(0.00390625f * (1 + seed % 3)));
        // IQ4_XS adds a signed 6-bit subscale. Keep random gate/down fixtures
        // within FP16 range even when both stages use its extreme codes.
        if (type == GGML_TYPE_IQ4_XS)
            scale = __half_as_ushort(__float2half_rn(0.00390625f * (1 + seed % 3) / 32));
        if (type == GGML_TYPE_F32) { float f = float(int(state % 17) - 8) / 128; std::memcpy(p, &f, 4); }
        else if (type == GGML_TYPE_BF16) { uint16_t bf = 0x3b80; std::memcpy(p, &bf, 2); }
        else if (type == GGML_TYPE_IQ1_M) {
            // IQ1_M distributes its FP16 scale among the top nibbles of four words.
            auto *q = reinterpret_cast<block_iq1_m *>(p);
            uint16_t scales[4]; std::memcpy(scales, q->scales, sizeof(scales));
            for (int i = 0; i < 4; ++i) scales[i] = (scales[i] & 0xfff) | ((scale >> (4 * i)) & 15) << 12;
            std::memcpy(q->scales, scales, sizeof(scales));
        } else {
            std::memcpy(p, &scale, 2);
            if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q2_K ||
                type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) std::memcpy(p + 2, &scale, 2);
        }
        if (type == GGML_TYPE_Q2_K) {
            auto *q = reinterpret_cast<block_q2_K *>(p); q->d = scale; q->dmin = scale;
        } else if (type == GGML_TYPE_Q3_K) reinterpret_cast<block_q3_K *>(p)->d = scale;
        else if (type == GGML_TYPE_Q6_K) reinterpret_cast<block_q6_K *>(p)->d = scale;
    }
    return d;
}

static std::vector<float> Decode(const fastllm::Data &w) {
    std::vector<float> result(w.dims[0] * w.dims[1]);
    auto type = static_cast<ggml_type>(w.ggmlType);
    if (type == GGML_TYPE_F32) std::memcpy(result.data(), w.cpuData, result.size() * 4);
    else if (type == GGML_TYPE_Q8_1) {
        auto *q = reinterpret_cast<const block_q8_1 *>(w.cpuData);
        for (size_t c = 0; c < result.size(); ++c) result[c] = __half2float(__ushort_as_half(q[c / 32].d)) * q[c / 32].qs[c % 32];
    } else {
        auto convert = ggml_type_to_float(type);
        Require(convert != nullptr, "CPU fixture decoder missing");
        const size_t stride = ggml_row_size(type, w.dims[1]);
        for (int row = 0; row < w.dims[0]; ++row)
            convert(w.cpuData + row * stride, result.data() + row * w.dims[1], w.dims[1]);
    }
    for (float f : result) Require(std::isfinite(f), "nonfinite fixture");
    return result;
}

// Independent CPU activation quantizer: serial max and round-away-from-zero,
// with an FP16 block scale. Compare GPU dots to decoded GGUF, not GPU unpackers.
template<class T> static std::vector<float> Q8Reference(const T *values, int count) {
    std::vector<float> result(count);
    for (int b = 0; b < count; b += 32) {
        float maximum = 0;
        for (int c = 0; c < 32; ++c) maximum = std::max(maximum, std::fabs(float(values[b+c])));
        const float scale = maximum/127.0f;
        const float stored = __half2float(__float2half_rn(scale));
        for (int c = 0; c < 32; ++c)
            result[b+c] = maximum == 0 ? 0 : std::round(float(values[b+c])/scale)*stored;
    }
    return result;
}

template<class T> static void CheckReference(ggml_type type, fastllm::DataType dtype,
        int pass, int batch, int hidden, int inter, int topk,
        const std::vector<std::vector<float>> &decoded,
        const std::vector<T> &x, const std::vector<float> &score,
        const std::vector<int32_t> &routes, const std::vector<T> &actual, bool useQ8,
        const std::vector<T> &actualGate = {}) {
        // MMQ changes the FP32 reduction order. A tiny activation difference
        // can cross a Q8 rounding boundary, so validate both stages separately:
        // CPU gate/up + SwiGLU, then CPU down using the validated GPU activation.
        Require(actualGate.empty() || actualGate.size() == size_t(batch*topk*inter), "gate size mismatch");
        std::vector<float> expected(batch * hidden), magnitude(batch * hidden);
        for (int r = 0; r < batch; ++r) for (int k = 0; k < topk; ++k) {
            const int e = routes[r * topk + k];
            if (e < 0 || e >= int(decoded.size()/2)) {
                if (!actualGate.empty()) for (int i = 0; i < inter; ++i)
                    Require(float(actualGate[(r*topk+k)*inter+i]) == 0, "invalid route retained an activation");
                continue;
            }
            const auto &gu = decoded[2 * e], &down = decoded[2 * e + 1];
            std::vector<float> inputQ8;
            if (useQ8) inputQ8 = Q8Reference(x.data() + r*hidden, hidden);
            std::vector<T> activated(inter);
            for (int i = 0; i < inter; ++i) {
                double g = 0, u = 0, gMagnitude = 0, uMagnitude = 0;
                for (int c = 0; c < hidden; ++c) {
                    const double value = useQ8 ? inputQ8[c] : float(x[r*hidden+c]);
                    const double gt = (useQ8 ? gu[i*hidden+c] : float(Cast<T>(gu[i*hidden+c]))) * value;
                    const double ut = (useQ8 ? gu[(i+inter)*hidden+c] : float(Cast<T>(gu[(i+inter)*hidden+c]))) * value;
                    g += gt; u += ut;
                    gMagnitude += std::fabs(gt); uMagnitude += std::fabs(ut);
                }
                const float gf = float(Cast<T>(float(g))), uf = float(Cast<T>(float(u)));
                activated[i] = Cast<T>(gf / (1 + std::exp(-gf)) * uf);
                if (!actualGate.empty()) {
                    const T observed = actualGate[(r*topk+k)*inter+i];
                    const float tolerance = dtype == fastllm::FLOAT32 ? 0.0001f :
                        dtype == fastllm::FLOAT16 ? 0.002f : 0.02f;
                    // A nearly cancelled dot can be amplified by the other
                    // SwiGLU factor. Bound FP32 reduction error in the two
                    // dots before comparing their nonlinear product.
                    const double allowed = tolerance*std::max(1.0f, std::fabs(float(activated[i]))) +
                        8*std::numeric_limits<float>::epsilon() *
                        (std::fabs(uf)*gMagnitude + std::fabs(gf)*uMagnitude);
                    if (!std::isfinite(float(observed)) ||
                        std::fabs(float(observed)-float(activated[i])) > allowed)
                        std::fprintf(stderr, "gate type=%d dtype=%d pass=%d row=%d route=%d col=%d g=%g u=%g expected=%g actual=%g\n",
                            type, dtype, pass, r, k, i, gf, uf, float(activated[i]), float(observed));
                    Require(std::isfinite(float(observed)) &&
                        std::fabs(float(observed)-float(activated[i])) <= allowed,
                        "GGUF grouped gate/up disagrees with CPU decoded reference");
                    activated[i] = observed;
                }
            }
            std::vector<float> midQ8;
            if (useQ8) midQ8 = Q8Reference(activated.data(), inter);
            for (int h = 0; h < hidden; ++h) {
                double sum = 0;
                for (int c = 0; c < inter; ++c)
                    sum += double(useQ8 ? down[h*inter+c] : float(Cast<T>(down[h*inter+c]))) *
                        (useQ8 ? midQ8[c] : float(activated[c]));
                volatile float weighted = float(Cast<T>(float(sum))) * score[r * topk + k];
                expected[r * hidden + h] += weighted;
                magnitude[r * hidden + h] += std::fabs(weighted);
            }
        }
        const float tolerance = dtype == fastllm::FLOAT32 ? 0.0001f : dtype == fastllm::FLOAT16 ? 0.003f : 0.025f;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float reference = float(Cast<T>(expected[i])), observed = float(actual[i]);
            if (!std::isfinite(observed) || std::fabs(reference - observed) > tolerance * std::max(1.0f, magnitude[i])) {
                std::fprintf(stderr, "type=%d dtype=%d pass=%d i=%zu expected=%g actual=%g\n", type, dtype, pass, i, reference, observed);
                throw std::runtime_error("GGUF cache disagrees with CPU decoded reference");
            }
        }
}

template<class T> static void Run(ggml_type type, fastllm::DataType dtype, int batch = 1, bool compact = false) {
    const int hidden = 256, inter = 256, experts = 24, topk = 10;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> tables[2];
    std::vector<std::vector<float>> decoded[2];
    FastllmCudaMoeCacheLayer layers[2];
    size_t stride = 0;
    for (int layer = 0; layer < 2; ++layer) {
        const int intermediate = layer == 0 ? inter : inter * 2;
        tables[layer].resize(2 * (experts + 1), nullptr);
        for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
            ggml_type format = layer == 0 ? (part == 0 ? type : GGML_TYPE_Q2_0) : (part == 0 ? GGML_TYPE_IQ2_XS : type);
            auto weight = Weight(format, part == 0 ? intermediate * 2 : hidden, part == 0 ? hidden : intermediate, e * 11 + part + layer * 5);
            decoded[layer].push_back(Decode(*weight));
            tables[layer][2 * (e + 1) + part] = weight.get(); owned.push_back(std::move(weight));
        }
        const size_t gate = tables[layer][2]->GetBytes(), down = tables[layer][3]->GetBytes();
        stride = std::max(stride, ((gate + 15) / 16 * 16 + down + 127) / 128 * 128);
        layers[layer] = {tables[layer].data(), int(tables[layer].size())};
    }
    fastllm::SetMoeCudaCacheBytes((compact ? 32 : 16) * stride);
    bool called = false;
    if (type == GGML_TYPE_F32) {
        tables[0][2]->isGGUFData = false;
        Require(!FastllmCudaPrepareMoeCache(layers, 2), "unmarked native floating expert admitted as GGUF");
        tables[0][2]->isGGUFData = true;
    }
    Require(FastllmCudaPrepareMoeCache(layers, 2, [&]{ called = true; }), "GGUF prepare failed");
#ifdef USE_NUMAS
    Require(called, "GGUF snapshot did not register the CPU weights");
#else
    Require(!called, "NUMA registration called in a CUDA-only build");
#endif
    // Original host tensors can be repacked after preparation without changing the snapshot.
    for (auto &w : owned) std::memset(w->cpuData, 0, w->GetBytes());
    fastllm::Data input(dtype, {batch, hidden}), ids(fastllm::INT32, {batch, topk});
    fastllm::Data scores(fastllm::FLOAT32, {batch, topk}), gate, output;
    Gpu(input); Gpu(ids); Gpu(scores);
    std::vector<T> x(batch * hidden);
    std::vector<float> score(batch * topk);
    // Include non-grid values, negative extrema and a whole zero Q8 block.
    for (int i = 0; i < batch * hidden; ++i)
        x[i] = Cast<T>(i%hidden >= 32 && i%hidden < 64 ? 0 :
            0.47f * std::sin(float(i)*0.713f) + 0.031f * std::cos(float(i)*1.37f));
    for (int i = 0; i < batch * topk; ++i) score[i] = (i % 3 == 0 ? -1 : 1) * float(i % 7 + 1) / 32;
    Cuda(cudaMemcpy(input.cudaData, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(scores.cudaData, score.data(), score.size() * 4, cudaMemcpyHostToDevice));
    uint64_t stats[5];
    cudaGraph_t graph[2]{}; cudaGraphExec_t exec[2]{};
    auto launch = [&](int layer) {
        Require(FastllmCudaMergeMOECache(input, gate, output, tables[layer].data(), tables[layer].size(),
            static_cast<int32_t *>(ids.cudaData), static_cast<float *>(scores.cudaData), topk), "GGUF compute failed");
    };
    for (int pass = 0; pass < 9; ++pass) {
        const int layer = pass < 2 ? 0 : pass % 2;
        std::vector<int32_t> routes(batch * topk);
        for (int r = 0; r < batch; ++r) for (int k = 0; k < topk; ++k)
            routes[r * topk + k] = pass < 2 ? k : (pass * 7 + k + r * 3) % experts;
        if (pass == 8) { routes[0] = -1; routes[1] = experts; routes[3] = routes[2]; }
        Cuda(cudaMemcpy(ids.cudaData, routes.data(), routes.size() * 4, cudaMemcpyHostToDevice));
        if (pass == 0) Require(fastllm_moe_cuda_cache_stats(0, stats, true), "stats reset failed");
        if (pass < 4) launch(layer);
        else {
            if (!exec[layer]) {
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                launch(layer); Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph[layer]));
                Cuda(cudaGraphInstantiate(&exec[layer], graph[layer], nullptr, nullptr, 0));
            }
            Cuda(cudaGraphLaunch(exec[layer], cudaStreamPerThread));
        }
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<T> actual(batch * hidden);
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        if (pass < 2 && batch == 1) {
            Require(fastllm_moe_cuda_cache_stats(0, stats, false), "stats snapshot failed");
            Require(stats[0] == uint64_t(pass * topk) && stats[1] == topk, "cold/hot counter mismatch");
            if (compact) Require(stats[2] <= stride * 32 && stats[3] > 32 && stats[4] == 48,
                "compact cache failed to hold more experts in the budget");
            else Require(stats[2] == stride * 16 && stats[3] == 16 && stats[4] == 48, "cache allocation counters mismatch");
        }
        const int intermediate = layer == 0 ? inter : inter * 2;
        const bool useQ8 = FastllmCudaMoeGGUFCacheQ8Supported(tables[layer][2]->ggmlType,
                tables[layer][3]->ggmlType, hidden, intermediate);
        CheckReference(type, dtype, pass, batch, hidden, intermediate, topk,
                       decoded[layer], x, score, routes, actual, useQ8);
    }
    for (int i = 0; i < 2; ++i) { Cuda(cudaGraphExecDestroy(exec[i])); Cuda(cudaGraphDestroy(graph[i])); }
    FastllmCudaReleaseMoeCache(tables[1].data(), tables[1].size());
    Require(!FastllmCudaCanRunMoeCache(tables[0].data(), tables[0].size()), "cache release leaked table");
    Require(fastllm_moe_cuda_cache_stats(0, stats, false) && stats[2] == 0, "released GPU allocation counted");
    std::printf("PASS GGUF cache type=%d dtype=%d batch=%d: mixed layers, snapshot, cold/hot, eviction, duplicate/invalid routes, graph, release\n", type, dtype, batch);
}
// Validate the FP32 prefill entry against CPU-decoded weights and an
// independent Q8 activation oracle. These formats all use MMQ's D4 layout.
static void RunFloatMmq(ggml_type type, int batch, int columns, int width) {
    auto weight = Weight(type, width, columns, 5, true);
    const auto decoded = Decode(*weight);
    weight->ToDevice(fastllm::CUDA, {0}, true);
    fastllm::Data input(fastllm::FLOAT32, {batch, columns});
    fastllm::Data output(fastllm::FLOAT32, {batch, width});
    Gpu(input); Gpu(output);
    std::vector<float> values(batch*columns), quantized(batch*columns);
    for (int i = 0; i < batch*columns; ++i)
        values[i] = i%columns < 32 ? 0 : .37f*std::sin(i*.173f)+.013f*std::cos(i*.71f);
    for (int i = 0; i < batch*columns; i += 32) {
        float maximum = 0;
        for (int c = 0; c < 32; ++c) maximum = std::max(maximum, std::fabs(values[i+c]));
        const float scale = maximum/127;
        for (int c = 0; c < 32; ++c)
            quantized[i+c] = maximum == 0 ? 0 : std::round(values[i+c]/scale)*scale;
    }
    Cuda(cudaMemcpy(input.cudaData, values.data(), values.size()*sizeof(float), cudaMemcpyHostToDevice));
    Require(!FastllmCudaFloatMatMulGGUFMMQ(input.cudaData, weight->cudaData, output.cudaData,
        type, 1, columns, width, cudaStreamPerThread), "MMQ stole single-token decode");
    Require(!FastllmCudaFloatMatMulGGUFMMQ(input.cudaData, weight->cudaData, output.cudaData,
        type, batch, columns-1, width, cudaStreamPerThread), "MMQ accepted a partial weight block");
    Require(FastllmCudaFloatMatMulGGUFMMQ(input.cudaData, weight->cudaData, output.cudaData,
        type, batch, columns, width, cudaStreamPerThread), "FP32 MMQ rejected valid prefill");
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> actual(batch*width);
    Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (int row = 0; row < batch; ++row) for (int col = 0; col < width; ++col) {
        double reference = 0, magnitude = 0;
        for (int c = 0; c < columns; ++c) {
            const double term = double(decoded[col*columns+c])*quantized[row*columns+c];
            reference += term; magnitude += std::fabs(term);
        }
        Require(std::isfinite(actual[row*width+col]) &&
            std::fabs(actual[row*width+col]-reference) <= 2e-5*std::max(1.0, magnitude),
            "FP32 MMQ disagrees with CPU decoded Q8 dot");
    }
    Require(FastllmCudaMatMulFloatGGUF(input, *weight, fastllm::Data(), output,
        batch, columns, width), "FP32 GGUF Linear rejected prefill");
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> dispatched(actual.size());
    Cuda(cudaMemcpy(dispatched.data(), output.cudaData, dispatched.size()*sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i)
        Require(std::fabs(actual[i]-dispatched[i]) <= 1e-6f*std::max(1.0f,std::fabs(actual[i])),
                "FP32 GGUF Linear bypassed MMQ");
}

template<class T> static void RunResident(ggml_type type, fastllm::DataType dtype,
                                        int device, int hidden = 256, int inter = 256, int batch = 1,
                                        ggml_type downType = GGML_TYPE_Q2_0) {
    Cuda(cudaSetDevice(device));
    const int experts = 16, topk = 10;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> table(2*(experts+1), nullptr);
    std::vector<std::vector<float>> decoded;
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto w = Weight(part ? downType : type,
            part ? hidden : 2*inter, part ? inter : hidden, 11*e+part, true);
        decoded.push_back(Decode(*w));
        w->isModelWeight = true;
        w->ToDevice(fastllm::CUDA, {device}, true);
        table[2*(e+1)+part] = w.get();
        owned.push_back(std::move(w));
    }
    fastllm::Data input(dtype, {batch, hidden}), ids(fastllm::INT32, {batch, topk});
    fastllm::Data scores(fastllm::FLOAT32, {batch, topk}), gate, workspace, output;
    for (auto *d : {&input, &ids, &scores}) {
        d->dataDevice = fastllm::CUDA; d->dataDeviceIds = {device}; d->Allocate(false);
    }
    std::vector<T> x(batch*hidden);
    for (int i = 0; i < batch*hidden; ++i) x[i] = Cast<T>(i >= 32 && i < 64 ? 0 :
        .47f*std::sin(float(i)*.713f)+.031f*std::cos(float(i)*1.37f));
    std::vector<float> score(batch*topk);
    for (int k = 0; k < batch*topk; ++k) score[k] = (k%3 ? 1 : -1)*float(k%7+1)/32;
    Cuda(cudaMemcpy(input.cudaData, x.data(), x.size()*sizeof(T), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(scores.cudaData, score.data(), score.size()*4, cudaMemcpyHostToDevice));
    auto launch = [&] {
        return FastllmCudaMergeMOEGGUFResidentIndexed(input, gate, workspace, output,
            table.data(), table.size(), static_cast<int32_t *>(ids.cudaData),
            static_cast<float *>(scores.cudaData), topk);
    };
    input.dataType = fastllm::INT32;
    Require(!launch() && !output.cudaData, "unsupported resident activation modified output");
    input.dataType = dtype;
    input.Resize({4097, hidden});
    Require(!launch(), "oversized batch admitted to resident kernel");
    input.Resize({batch, hidden});
    // Cold graph capture must reject before allocating/uploading metadata.
    cudaGraph_t cold{};
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(!launch(), "resident metadata initialized during graph capture");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &cold));
    Cuda(cudaGraphDestroy(cold));
    const bool useQ8 = batch > 32 ||
        FastllmCudaMoeGGUFCacheQ8Supported(type, downType, hidden, inter);
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    std::vector<int32_t> routes(batch*topk);
    for (int pass = 0; pass < 5; ++pass) {
        for (int k = 0; k < batch*topk; ++k) routes[k] = (7*pass+k)%experts;
        if (batch > 32 && pass == 1) std::fill(routes.begin(), routes.end(), 0);
        if (batch > 32 && pass == 3) std::fill(routes.begin(), routes.end(), -1);
        if (pass == 2 || pass == 4) for (int r = 0; r < batch; ++r) {
            routes[r*topk] = -1; routes[r*topk+1] = experts; routes[r*topk+3] = routes[r*topk+2];
        }
        Cuda(cudaMemcpy(ids.cudaData, routes.data(), routes.size()*4, cudaMemcpyHostToDevice));
        if (pass < 3) Require(launch(), "resident eager kernel rejected valid weights");
        else {
            if (!exec) {
                Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                Require(launch(), "resident graph capture failed");
                Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            }
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        }
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<T> actual(batch*hidden);
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
        std::vector<T> actualGate(batch > 32 ? batch*topk*inter : 0);
        if (!actualGate.empty()) Cuda(cudaMemcpy(actualGate.data(), gate.cudaData,
            actualGate.size()*sizeof(T), cudaMemcpyDeviceToHost));
        CheckReference(type, dtype, pass, batch, hidden, inter, topk,
                       decoded, x, score, routes, actual, useQ8, actualGate);
        // A verifier row must compute exactly the same result as decode. This
        // catches row-dependent quantization/rounding that changes acceptance.
        if (pass == 2 && batch > 1 && batch <= 32) {
            fastllm::Data rowGate, rowWorkspace, rowOutput;
            for (int r = 0; r < batch; ++r) {
                fastllm::Data rowInput(dtype, {1, hidden});
                rowInput.FakeFrom(input, size_t(r)*hidden*sizeof(T));
                Require(FastllmCudaMergeMOEGGUFResidentIndexed(rowInput, rowGate, rowWorkspace,
                    rowOutput, table.data(), table.size(),
                    static_cast<int32_t *>(ids.cudaData)+r*topk,
                    static_cast<float *>(scores.cudaData)+r*topk, topk), "single row rejected");
                std::vector<T> serial(hidden);
                Cuda(cudaMemcpy(serial.data(), rowOutput.cudaData, hidden*sizeof(T), cudaMemcpyDeviceToHost));
                Require(std::memcmp(serial.data(), actual.data()+r*hidden, hidden*sizeof(T)) == 0,
                        "batched resident result differs from single row");
            }
        }

    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    // Moving any expert, including one that is not the table key, invalidates
    // the uploaded pointer table. Restoring it must build fresh GPU metadata.
    owned[6]->ToDevice(fastllm::CPU);
    Require(!launch(), "host expert admitted to resident GPU path");
    owned[6]->ToDevice(fastllm::CUDA, {device}, true);
    Require(launch(), "resident table failed to rebuild after weight migration");
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<T> actual(batch*hidden);
    Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<T> actualGate(batch > 32 ? batch*topk*inter : 0);
    if (!actualGate.empty()) Cuda(cudaMemcpy(actualGate.data(), gate.cudaData,
        actualGate.size()*sizeof(T), cudaMemcpyDeviceToHost));
    CheckReference(type, dtype, 5, batch, hidden, inter, topk, decoded, x, score, routes, actual, useQ8, actualGate);
    int devices = 0; Cuda(cudaGetDeviceCount(&devices));
    if (devices > 1) {
        const int foreign = (device+1)%devices;
        Cuda(cudaSetDevice(foreign));
        FastllmCudaReleaseMoeGGUFResident(owned[7].get());
        int current = -1; Cuda(cudaGetDevice(&current));
        Require(current == foreign, "metadata release changed caller's CUDA device");
        Cuda(cudaSetDevice(device));
        Require(launch(), "resident table failed to rebuild after release on another GPU");
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    }
    std::printf("PASS GGUF resident type=%d dtype=%d device=%d H=%d I=%d batch=%d: CPU oracle, GPU routes, duplicate/invalid routes, graph, migration, release\n",
        type, dtype, device, hidden, inter, batch);
}

// Exercise the real packed GGUF splitter, not independently manufactured
// 320-wide weights. The oracle checks both byte-preserving partitions and
// the fused projection arithmetic on each rank.
template<class T> static void RunTPShards(ggml_type type, fastllm::DataType dtype, int batch) {
    const int hidden = 2560, inter = 640, localInter = 320, experts = 4, topk = 10;
    std::vector<int> devices{0, 1};
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> tables[2];
    std::vector<std::vector<float>> decoded[2];
    for (auto &table : tables) table.resize(2*(experts+1), nullptr);
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto weight = Weight(part ? GGML_TYPE_Q2_0 : type,
            part ? hidden : 2*inter, part ? inter : hidden, 7*e+part, true);
        weight->isModelWeight = true;
        const size_t bytes = weight->GetBytes();
        std::vector<uint8_t> packed(weight->cpuData, weight->cpuData+bytes);
        DivisionScheme scheme;
        for (int rank = 0; rank < 2; ++rank) {
            scheme[rank] = {{rank*localInter, (rank+1)*localInter}};
            if (!part) scheme[rank].push_back({inter+rank*localInter, inter+(rank+1)*localInter});
        }
        fastllm::Data bias;
        Require(SplitMultiCudaWeight(*weight, bias, devices, scheme, part, true), "GGUF TP split failed");
        size_t shardBytes = 0;
        for (int rank = 0; rank < 2; ++rank) {
            auto *shard = weight->multiDeviceDatas.at(rank);
            Require(shard->dataType == fastllm::DATA_GGUF_FORMAT && shard->ggmlType == weight->ggmlType,
                    "GGUF TP changed the packed type");
            Require(shard->dims == (part ? std::vector<int>{hidden,localInter} :
                std::vector<int>{2*localInter,hidden}), "GGUF TP shard shape");
            shardBytes += shard->GetBytes();
            shard->ToDevice(fastllm::CPU);
            const size_t rowBytes = ggml_row_size(static_cast<ggml_type>(shard->ggmlType), shard->dims[1]);
            const size_t originalRowBytes = ggml_row_size(static_cast<ggml_type>(weight->ggmlType), weight->dims[1]);
            for (int row = 0; row < shard->dims[0]; ++row) {
                const size_t offset = part ? size_t(row)*originalRowBytes+rank*rowBytes :
                    size_t((row/localInter)*inter+rank*localInter+row%localInter)*originalRowBytes;
                Require(std::memcmp(shard->cpuData+size_t(row)*rowBytes, packed.data()+offset, rowBytes)==0,
                        "GGUF TP changed/reordered quantization blocks");
            }
            decoded[rank].push_back(Decode(*shard));
            shard->ToDevice(fastllm::CUDA, {rank}, true);
            tables[rank][2*(e+1)+part] = shard;
        }
        Require(shardBytes == bytes && !weight->cpuData && !weight->cudaData,
                "GGUF TP retained or expanded the source payload");
        owned.push_back(std::move(weight));
    }
    std::vector<T> x(batch*hidden);
    for (size_t i = 0; i < x.size(); ++i) x[i] = Cast<T>(.27f*std::sin(i*.713f)+.03f*std::cos(i*1.37f));
    std::vector<int32_t> routes(batch*topk);
    std::vector<float> score(batch*topk);
    for (int k = 0; k < batch*topk; ++k) { routes[k] = k%experts; score[k] = (k%3 ? 1 : -1)*float(k%7+1)/32; }
    for (int r = 0; r < batch; ++r) { routes[r*topk] = -1; routes[r*topk+1] = experts; }
    const bool q8 = FastllmCudaMoeGGUFCacheQ8Supported(type, GGML_TYPE_Q2_0, hidden, localInter);
    for (int rank = 0; rank < 2; ++rank) {
        Cuda(cudaSetDevice(rank));
        fastllm::Data input(dtype, {batch,hidden}), ids(fastllm::INT32,{batch,topk});
        fastllm::Data scores(fastllm::FLOAT32,{batch,topk}), gate, workspace, output;
        for (auto *d : {&input,&ids,&scores}) { d->dataDevice=fastllm::CUDA; d->dataDeviceIds={rank}; d->Allocate(false); }
        Cuda(cudaMemcpy(input.cudaData,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(ids.cudaData,routes.data(),routes.size()*4,cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(scores.cudaData,score.data(),score.size()*4,cudaMemcpyHostToDevice));
        Require(FastllmCudaMergeMOEGGUFResidentIndexed(input,gate,workspace,output,tables[rank].data(),
            tables[rank].size(),static_cast<int32_t *>(ids.cudaData),static_cast<float *>(scores.cudaData),topk),
            "GGUF TP resident kernel rejected 320-column shard");
        std::vector<T> actual(batch*hidden);
        Cuda(cudaMemcpy(actual.data(),output.cudaData,actual.size()*sizeof(T),cudaMemcpyDeviceToHost));
        std::vector<T> actualGate(batch > 32 ? batch*topk*localInter : 0);
        if (!actualGate.empty()) Cuda(cudaMemcpy(actualGate.data(), gate.cudaData,
            actualGate.size()*sizeof(T), cudaMemcpyDeviceToHost));
        CheckReference(type,dtype,rank,batch,hidden,localInter,topk,decoded[rank],x,score,routes,actual,q8,actualGate);
    }
    std::printf("PASS GGUF TP shards type=%d dtype=%d batch=%d: packed bytes, ownership, 320/320, two-rank CPU oracle\n",type,dtype,batch);
}

// Exercise the CPU repacker, cross-SwiGLU layout, selected expert subsets and
// NUMA row shards against independently decoded ordinary GGUF weights.
template<class T> static void RunHost(ggml_type type, fastllm::DataType dtype,
        int device, int batch = 65, int inter = 256, bool cross = true,
        ggml_type downType = GGML_TYPE_Q2_0) {
    Cuda(cudaSetDevice(device));
    const int hidden = 256, experts = 4, topk = 3;
    std::vector<std::unique_ptr<fastllm::Data>> owned;
    std::vector<fastllm::Data *> table(2*(experts+1), nullptr);
    std::vector<std::vector<float>> decoded;
    std::vector<std::vector<uint8_t>> packed;
    for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
        auto w = Weight(part ? downType : type, part ? hidden : 2*inter,
            part ? inter : hidden, 11*e+part, true);
        decoded.push_back(Decode(*w));
        const int width = w->dims[1], height = w->dims[0];
        const size_t stride = ggml_row_size(static_cast<ggml_type>(w->ggmlType), width);
        std::vector<uint8_t> plain(w->cpuData,w->cpuData+w->GetBytes());
        if (!part && cross) for (int r = 0; r < height; ++r)
            std::memcpy(w->cpuData+r*stride, plain.data()+(r/2+(r%2)*inter)*stride, stride);
        auto *repack = get_repack_info(static_cast<ggml_type>(w->ggmlType));
        if (repack) {
            plain.assign(w->cpuData,w->cpuData+w->GetBytes());
            repack->repack(height,width,reinterpret_cast<const char *>(plain.data()),
                reinterpret_cast<char *>(w->cpuData),false);
            w->ggmlType = repack->new_type;
        }
        packed.emplace_back(w->cpuData,w->cpuData+w->GetBytes());
        table[2*(e+1)+part] = w.get(); owned.push_back(std::move(w));
    }
    fastllm::Data input(dtype,{batch,hidden}),gate,workspace,output;
    input.dataDevice = fastllm::CUDA; input.dataDeviceIds = {device}; input.Allocate(false);
    std::vector<T> x(batch*hidden);
    for (int i = 0; i < batch*hidden; ++i) x[i] = Cast<T>(.3f*std::sin(i*.713f));
    Cuda(cudaMemcpy(input.cudaData,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::vector<int32_t> routes(batch*topk),masked;
    std::vector<float> scores(batch*topk);
    for (int i = 0; i < batch*topk; ++i) {
        routes[i] = (i%13 == 0) ? -1 : i%experts;
        scores[i] = (i%2 ? 1 : -1)*float(i%7+1)/32;
    }
    std::unordered_set<int> selected{1,3};
    auto launch = [&] { return FastllmCudaMergeMOEGGUFHost(input,gate,workspace,output,
        table.data(),experts,routes.data(),scores.data(),topk,selected,cross); };
    if (type == GGML_TYPE_IQ1_M) {
        Require(!launch() && !output.cudaData,"IQ1_M host GEMM fallback not preserved");
        std::puts("PASS GGUF host IQ1_M retains per-expert GEMM");
        return;
    }
    input.dataType = fastllm::INT32;
    Require(!launch() && !output.cudaData,"unsupported host activation modified output");
    input.dataType = dtype;
    const int originalType = table[2]->ggmlType;
    table[2]->ggmlType = GGML_TYPE_Q4_0;
    const bool unsupported = launch(); table[2]->ggmlType = originalType;
    Require(!unsupported && !output.cudaData,"unsupported host format modified output");
    selected.insert(0); Require(!launch(),"shared expert admitted"); selected.erase(0);
    input.Resize({32,hidden}); Require(!launch(),"decode entered host prefill"); input.Resize({batch,hidden});
    cudaGraph_t graph{};
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    Require(!launch(),"host uploads admitted during graph capture");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph)); Cuda(cudaGraphDestroy(graph));
    for (int pass = 0; pass < 3; ++pass) {
        selected = pass == 0 ? std::unordered_set<int>{1,3} :
            pass == 1 ? std::unordered_set<int>{2,4} : std::unordered_set<int>{1,2,3,4};
        // Alias two immutable NUMA shards; restore ownership before assertions.
        std::vector<uint8_t *> originals;
        for (auto &w : owned) {
            originals.push_back(w->cpuData);
            if (pass) { w->numasData = {w->cpuData,w->cpuData+w->GetBytes()/2}; w->cpuData = nullptr; }
        }
        const bool ok = launch();
        for (size_t i = 0; i < owned.size(); ++i) {
            owned[i]->cpuData = originals[i]; owned[i]->numasData.clear();
        }
        Require(ok,"host grouped prefill rejected fixture");
        std::vector<T> actual(batch*hidden),actualGate(batch*topk*inter);
        Cuda(cudaMemcpy(actual.data(),output.cudaData,actual.size()*sizeof(T),cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(actualGate.data(),gate.cudaData,actualGate.size()*sizeof(T),cudaMemcpyDeviceToHost));
        masked = routes;
        for (auto &e : masked) if (!selected.count(e+1)) e = -1;
        CheckReference(type,dtype,pass,batch,hidden,inter,topk,decoded,x,scores,masked,actual,true,actualGate);
        for (size_t i = 0; i < owned.size(); ++i) {
            Require(!owned[i]->cudaData,"host weight gained persistent CUDA storage");
            Require(std::memcmp(owned[i]->cpuData,packed[i].data(),packed[i].size()) == 0,
                "host packed weights were modified");
        }
    }
    std::printf("PASS GGUF host type=%d dtype=%d rows=%d inter=%d cross=%d device=%d\n",type,dtype,batch,inter,cross,device);
}

#ifdef USE_NUMAS
// A serial, whole-shard reference exercises the ordinary GEMM/quantize path,
// independently of the decoder's expert/worker splitting and fused Q8 writer.
static void CpuExpertReference(const float *input, float *output,
                              fastllm::Data &gate, fastllm::Data &down) {
    using namespace fastllm;
    const int hidden = gate.dims[1], inter = down.dims[1];
    const auto gateAct = gate.GetLinearActDataType(1), downAct = down.GetLinearActDataType(1);
    std::vector<uint8_t> x(GetDataBytes(gateAct, 1, hidden)), y(GetDataBytes(downAct, 1, inter));
    std::vector<float> gu(2 * inter), swiglu(inter);
    ConvertFromFloat32(x.data(), gateAct, input, 1, hidden);
    const int nodes = gate.numasData.size();
    for (int node = 0; node < nodes; ++node)
        MultiThreadGemmOp(x.data(), gateAct, gate.numasData[node], gate.GetDataType(),
            reinterpret_cast<uint8_t *>(gu.data() + node * 2 * inter / nodes), FLOAT32,
            1, hidden, 2 * inter, 0, 2 * inter / nodes).Run();
    MultiThreadCrossSwigluOp(gu.data(), inter, inter, swiglu.data(), 1, inter * 2, inter).Run();
    ConvertFromFloat32(y.data(), downAct, swiglu.data(), 1, inter);
    for (int node = 0; node < nodes; ++node)
        MultiThreadGemmOp(y.data(), downAct, down.numasData[node], down.GetDataType(),
            reinterpret_cast<uint8_t *>(output + node * hidden / nodes), FLOAT32,
            1, inter, hidden, 0, hidden / nodes).Run();
}

static void RunHybrid(ggml_type format, int rows, bool single = false) {
    using namespace fastllm;
    constexpr int experts = 24, topk = 7;
    int hidden = 256;
    const int ranks = single ? 1 : 2;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> tables[2];
    FastllmCudaMoeCacheLayer layers[2];
    size_t stride = 0;
    for (int layer = 0; layer < 2; ++layer) {
        const int inter = (layer + 1) * 256, width = (layer + 1) * 256;
        tables[layer].resize(2 * (experts + 1));
        for (int e = 0; e < experts; ++e) for (int part = 0; part < 2; ++part) {
            const auto type = layer == 0 ? (part == 0 ? format : GGML_TYPE_Q2_0) :
                (part == 0 ? GGML_TYPE_IQ2_XS : format);
            auto w = Weight(type, part == 0 ? inter * 2 : width,
                            part == 0 ? width : inter, 11 * e + part);
            tables[layer][2 * (e + 1) + part] = w.get(); owned.push_back(std::move(w));
        }
        stride = std::max(stride, (tables[layer][2]->GetBytes() + 15) / 16 * 16 + tables[layer][3]->GetBytes());
        layers[layer] = {tables[layer].data(), int(tables[layer].size())};
    }
    SetMoeCudaCacheBytes(16 * ((stride + 127) / 128 * 128));
    Require(FastllmCudaPrepareMoeCache(layers, 2, [&] {
        for (auto &table : tables) for (size_t i = 2; i < table.size(); ++i)
            RegisterNumas(table[i], i % 2 == 0 ? "linearSwiglu" : "linearColumn");
    }), "GGUF hybrid preparation failed");
    for (auto &table : tables)
        Require(CanRunNumasMoeDecodeExperts(table.data(), table.size()), "GGUF CPU subset unavailable");
    auto context = FastllmCudaCreateMoeExpertParallel(2);
    Data input[2]{{FLOAT32, {rows, hidden}}, {FLOAT32, {rows, hidden}}}, output[2];
    for (int rank = 0; rank < ranks; ++rank) {
        Cuda(cudaSetDevice(rank)); input[rank].ToDevice(CUDA, std::vector<int>{rank}); input[rank].Allocate(false);
    }
    Cuda(cudaSetDevice(0));
    Data ids(INT32, {rows, topk}), scores(FLOAT32, {rows, topk}), gpuGate, gpuOutput;
    Gpu(ids); Gpu(scores);
    std::vector<float> x(rows * hidden), score(rows * topk), cpu(rows * topk * hidden),
        serial(cpu.size()), gpu(rows * hidden), actual[2]{std::vector<float>(rows * hidden), std::vector<float>(rows * hidden)};
    std::vector<int32_t> route(rows * topk), mask(rows * topk, -1);
    for (int step = 0; step < 18; ++step) {
        const int layer = step % 2;
        auto &table = tables[layer];
        hidden = table[2]->dims[1];
        x.resize(rows * hidden); cpu.resize(rows * topk * hidden); serial.resize(cpu.size());
        gpu.resize(x.size()); actual[0].resize(x.size()); actual[1].resize(x.size());
        for (int i = 0; i < int(x.size()); ++i) x[i] = float((i * 13) % 31 - 15) / 64;
        for (int rank = 0; rank < ranks; ++rank) {
            Cuda(cudaSetDevice(rank)); input[rank].Resize({rows, hidden}); input[rank].Allocate(false);
            Cuda(cudaMemcpy(input[rank].cudaData, x.data(), x.size()*4, cudaMemcpyHostToDevice));
        }
        for (int r = 0; r < rows * topk; ++r) {
            const int k = r % topk;
            route[r] = (k == 6 ? 1 : k) + (step >= 10 ? 12 : 0);
            score[r] = k == 3 ? 0 : k == 5 ? -.125f : float(k + 1) / 32;
        }
        NumasMoeDecodeExpertsBatch(x.data(), cpu.data(), rows, table.data(), table.size(),
            route.data(), mask.data(), score.data(), topk, layer);
        if (rows == 1) {
            std::vector<float> overlapped(cpu.size(), 0);
            int submitted = 0;
            NumasMoeDecodeExpertsWithOverlap(x.data(), overlapped.data(), table.data(),
                route.data(), mask.data(), topk, layer, [&] { ++submitted; });
            Require(submitted == 1 && overlapped == cpu,
                    "overlap changed CPU expert arithmetic or callback count");
            if (step == 0) {
                bool caught = false;
                try {
                    NumasMoeDecodeExpertsWithOverlap(x.data(), overlapped.data(), table.data(),
                        route.data(), mask.data(), topk, layer,
                        [] { throw std::runtime_error("expected submit failure"); });
                } catch (const std::runtime_error &) { caught = true; }
                Require(caught, "overlap callback failure was swallowed");
                // Reuse immediately: failure must drain the borrowed jobs.
                NumasMoeDecodeExpertsWithOverlap(x.data(), overlapped.data(), table.data(),
                    route.data(), mask.data(), topk, layer, [] {});
                Require(overlapped == cpu, "overlap failure left expert workers active");
            }
        }
        for (int r = 0; r < rows * topk; ++r)
            CpuExpertReference(x.data() + (r / topk) * hidden, serial.data() + r * hidden,
                *table[2 * (route[r] + 1)], *table[2 * (route[r] + 1) + 1]);
        for (size_t i = 0; i < cpu.size(); ++i)
            Require(std::isfinite(cpu[i]) && std::abs(cpu[i] - serial[i]) < 3e-5f * (1 + std::abs(serial[i])),
                    "GGUF CPU decoder differs from serial GEMM/quantize reference");
        Cuda(cudaSetDevice(0));
        Cuda(cudaMemcpy(ids.cudaData, route.data(), route.size()*4, cudaMemcpyHostToDevice));
        std::vector<float> lower(rows * hidden, 0), upper(lower.size(), 0);
        // Cache GPU and CPU arithmetic may round differently: independently
        // bound each weighted route, then verify neither duplication nor loss.
        for (int k = 0; k < topk; ++k) {
            std::vector<float> one(rows * topk, 0);
            for (int row = 0; row < rows; ++row) one[row * topk + k] = 1;
            Cuda(cudaMemcpy(scores.cudaData, one.data(), one.size()*4, cudaMemcpyHostToDevice));
            Require(FastllmCudaMergeMOECache(input[0], gpuGate, gpuOutput, table.data(), table.size(),
                static_cast<int32_t *>(ids.cudaData), static_cast<float *>(scores.cudaData), topk), "GPU oracle rejected");
            Cuda(cudaMemcpy(gpu.data(), gpuOutput.cudaData, gpu.size()*4, cudaMemcpyDeviceToHost));
            for (int c = 0; c < rows * hidden; ++c) {
                const int r = (c / hidden) * topk + k;
                const float a = serial[r * hidden + c % hidden] * score[r], b = gpu[c] * score[r];
                lower[c] += std::min(a,b); upper[c] += std::max(a,b);
            }
        }
        Cuda(cudaMemcpy(scores.cudaData, score.data(), score.size()*4, cudaMemcpyHostToDevice));
        std::exception_ptr errors[2]; bool accepted[2]{};
        auto run = [&](int rank) {
            try {
                Cuda(cudaSetDevice(rank)); Data empty;
                if (single) accepted[rank] = FastllmCudaMergeMOEHybrid(input[rank], ids, scores, output[rank],
                    table.data(), table.size(), layer, [] {});
                else accepted[rank] = FastllmCudaMergeMOEExpertParallel(*context, rank, input[rank],
                    rank == 0 ? ids : empty, rank == 0 ? scores : empty, output[rank],
                    table.data(), table.size(), layer, [] {});
                if (accepted[rank]) Cuda(cudaMemcpy(actual[rank].data(), output[rank].cudaData,
                    actual[rank].size()*4, cudaMemcpyDeviceToHost));
            } catch (...) { errors[rank] = std::current_exception(); }
        };
        if (single) run(0);
        else { std::thread a(run,0), b(run,1); a.join(); b.join(); }
        for (int rank = 0; rank < ranks; ++rank) {
            if (errors[rank]) std::rethrow_exception(errors[rank]);
            Require(accepted[rank], "GGUF EP hybrid path rejected");
        }
        for (size_t c = 0; c < lower.size(); ++c) {
            const float v = actual[0][c] + (single ? 0 : actual[1][c]);
            const float tol = 3e-5f * (1 + std::max(std::abs(lower[c]), std::abs(upper[c])));
            Require(std::isfinite(v) && v >= lower[c]-tol && v <= upper[c]+tol, "GGUF EP lost or duplicated a route");
        }
    }
    const auto stats = FastllmCudaGetMoeExpertParallelStats(*context);
    Require(single || (stats.cpuRoutes && stats.gpuRoutes[0] && stats.gpuRoutes[1] && stats.multiGpuSteps),
            "GGUF EP did not exercise CPU and both GPUs");
    context.reset();
    FastllmCudaReleaseMoeCache(tables[0].data(), tables[0].size()); SetMoeCudaCacheBytes(0);
    ClearNumasMoeRuntimeCache(); Cuda(cudaSetDevice(0));
    std::printf("PASS GGUF hybrid format=%d rows=%d ranks=%d: CPU serial reference, mixed layers, both GPUs, duplicate/zero/negative routes\n", format, rows, ranks);
}
#endif

int main(int argc, char **argv) {
    try {
        int count = 0; Cuda(cudaGetDeviceCount(&count)); if (!count) { std::puts("SKIP: no CUDA device"); return 0; }
        Cuda(cudaSetDevice(0));
#ifdef USE_NUMAS
        if (argc > 1 && std::strcmp(argv[1], "--hybrid") == 0) {
            if (count < 2) { std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: requires two GPUs"); return 0; }
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS,
                              GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL}) {
                RunHybrid(type, 1); RunHybrid(type, 4);
            }
            RunHybrid(GGML_TYPE_IQ2_S, 1, true);
            RunHybrid(GGML_TYPE_Q4_K, 4, true);
            std::puts("PASS: GGUF CPU/GPU hybrid expert decode"); return 0;
        }
#endif
        if (argc > 1 && std::strcmp(argv[1], "--host") == 0) {
            cudaDeviceProp properties{}; Cuda(cudaGetDeviceProperties(&properties, 0));
            if (properties.major*10+properties.minor < 75) {
                std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: host MMQ requires SM75+"); return 0;
            }
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS,
                              GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S}) {
                RunHost<float>(type, fastllm::FLOAT32, 0);
                RunHost<half>(type, fastllm::FLOAT16, 0, 33, 320);
            }
            RunHost<__nv_bfloat16>(GGML_TYPE_IQ2_S, fastllm::BFLOAT16, 0);
            RunHost<float>(GGML_TYPE_IQ2_XS, fastllm::FLOAT32, 0, 33, 256, false);
            RunHost<float>(GGML_TYPE_IQ2_S, fastllm::FLOAT32, 0, 33, 256, true, GGML_TYPE_IQ2_XS);
            if (count >= 2) RunHost<float>(GGML_TYPE_IQ2_XXS, fastllm::FLOAT32, 1);
            std::puts("PASS: streamed GGUF prefill, NUMA shards, selected subsets, immutable weights");
            return 0;
        }
        if (argc > 1 && std::strcmp(argv[1], "--compact") == 0) {
            Run<float>(GGML_TYPE_IQ3_S, fastllm::FLOAT32, 1, true);
            Run<half>(GGML_TYPE_IQ4_XS, fastllm::FLOAT16, 4, true);
            std::puts("PASS: compact heterogeneous GGUF cache and graph replay"); return 0;
        }
        bool tpOnly = argc > 1 && std::strcmp(argv[1], "--tp-shards") == 0;
        if (argc > 1 && std::strcmp(argv[1], "--mmq") == 0) {
            cudaDeviceProp properties{}; Cuda(cudaGetDeviceProperties(&properties, 0));
            if (properties.major*10+properties.minor < 75) {
                std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: MMQ requires SM75+"); return 0;
            }
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                              GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS}) {
                RunFloatMmq(type, 9, 256, 47);
                RunFloatMmq(type, 33, 256, 129);
            }
            RunFloatMmq(GGML_TYPE_IQ3_XXS, 65, 2560, 65);
            RunFloatMmq(GGML_TYPE_IQ4_NL, 129, 320, 47);
            RunFloatMmq(GGML_TYPE_Q2_0, 129, 320, 47);
            std::puts("PASS: FP32 GGUF MMQ and Linear dispatch"); return 0;
        }
        if (argc > 1 && std::strcmp(argv[1], "--grouped") == 0) {
            cudaDeviceProp properties{}; Cuda(cudaGetDeviceProperties(&properties, 0));
            if (properties.major*10+properties.minor < 75) {
                std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: grouped MMQ requires SM75+"); return 0;
            }
            fastllm::SetMoeCudaCacheBytes(0);
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS,
                              GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S}) {
                RunResident<float>(type, fastllm::FLOAT32, 0, 256, 256, 33);
                RunResident<half>(type, fastllm::FLOAT16, 0, 256, 256, 65);
            }
            RunResident<__nv_bfloat16>(GGML_TYPE_IQ2_S, fastllm::BFLOAT16, 0, 256, 320, 129);
            // Q2 supports non-256-aligned gate input as well as down input.
            // Exercise both tails in the quantize-once/gather path.
            RunResident<__nv_bfloat16>(GGML_TYPE_Q2_0, fastllm::BFLOAT16, 0, 320, 192, 65);
            RunResident<float>(GGML_TYPE_Q2_0, fastllm::FLOAT32, 0, 64, 64, 33);
            for (auto type : {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS}) {
                RunResident<float>(type, fastllm::FLOAT32, 0, 256, 320, 33, GGML_TYPE_IQ4_NL);
                RunResident<half>(type, fastllm::FLOAT16, 0, 256, 256, 65, type);
            }
            RunResident<__nv_bfloat16>(GGML_TYPE_IQ3_S, fastllm::BFLOAT16, 0, 256, 320, 33, GGML_TYPE_IQ4_NL);
            RunResident<float>(GGML_TYPE_IQ4_NL, fastllm::FLOAT32, 0, 96, 32, 33, GGML_TYPE_IQ4_NL);
            for (auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S})
                RunResident<float>(GGML_TYPE_Q2_0, fastllm::FLOAT32, 0, 256, 256, 33, type);
            if (count >= 2) {
                RunTPShards<float>(GGML_TYPE_IQ2_S, fastllm::FLOAT32, 33);
                RunTPShards<half>(GGML_TYPE_IQ2_XXS, fastllm::FLOAT16, 33);
            }
            std::puts("PASS: grouped GGUF expert prefill, Q2 K-tail and TP shards");
            return 0;
        }
#ifdef FASTLLM_GGUF_TP_SHARD_TEST
        tpOnly = true;
#endif
        if (tpOnly) {
            if (count < 2) { std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: requires two GPUs"); return 0; }
            fastllm::SetMoeCudaCacheBytes(0);
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S}) {
                RunTPShards<half>(type, fastllm::FLOAT16, 1);
                RunTPShards<half>(type, fastllm::FLOAT16, 4);
            }
            RunTPShards<float>(GGML_TYPE_IQ2_S, fastllm::FLOAT32, 4);
            RunTPShards<__nv_bfloat16>(GGML_TYPE_IQ2_S, fastllm::BFLOAT16, 4);
            std::puts("PASS: packed GGUF TP shards and resident kernels"); return 0;
        }
        const ggml_type formats[] = {GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16};
        Require(!FastllmCudaMoeGGUFCacheSupported(GGML_TYPE_IQ2_S_R4, 256), "CPU repacked layout admitted");
        Require(!FastllmCudaMoeGGUFCacheSupported(GGML_TYPE_IQ2_S, 640), "partial IQ block admitted");
        Require(!FastllmCudaMoeGGUFCacheQ8Supported(GGML_TYPE_Q4_1, GGML_TYPE_Q2_0, 256, 256), "unsupported Q8 encoding admitted");
        Require(!FastllmCudaMoeGGUFCacheQ8Supported(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, 32768, 256), "Q8 shared memory limit not enforced");
        for (auto type : formats) { Run<float>(type, fastllm::FLOAT32); Run<half>(type, fastllm::FLOAT16); Run<__nv_bfloat16>(type, fastllm::BFLOAT16); }
        Run<half>(GGML_TYPE_IQ2_S, fastllm::FLOAT16, 3);
        Run<half>(GGML_TYPE_IQ2_XXS, fastllm::FLOAT16, 9);
        for (int batch : {8, 9}) Run<half>(GGML_TYPE_Q2_0, fastllm::FLOAT16, batch);
        fastllm::SetMoeCudaCacheBytes(0);
        for (auto type : formats) {
            RunResident<float>(type, fastllm::FLOAT32, 0);
            RunResident<half>(type, fastllm::FLOAT16, 0);
            RunResident<__nv_bfloat16>(type, fastllm::BFLOAT16, 0);
            RunResident<float>(type, fastllm::FLOAT32, 0, 256, 256, 4);
            RunResident<half>(type, fastllm::FLOAT16, 0, 256, 256, 4);
            RunResident<__nv_bfloat16>(type, fastllm::BFLOAT16, 0, 256, 256, 4);
        }
        // Exercise the actual Qwen shape, mixed gate/down formats and both GPUs.
        for (int device = 0; device < std::min(2, count); ++device) {
            for (auto type : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M})
                RunResident<float>(type, fastllm::FLOAT32, device, 2560, 640);
            for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M}) {
                RunResident<float>(type, fastllm::FLOAT32, device, 2560, 640, 4);
                RunResident<half>(type, fastllm::FLOAT16, device, 2560, 640, 4);
            }
        }
        RunResident<half>(GGML_TYPE_IQ2_S, fastllm::FLOAT16, 0, 256, 256, 3);
        RunResident<half>(GGML_TYPE_IQ2_XXS, fastllm::FLOAT16, 0, 256, 256, 9);
        RunResident<half>(GGML_TYPE_Q2_0, fastllm::FLOAT16, 0, 256, 256, 32);
        for (int batch : {8, 9}) {
            RunResident<float>(GGML_TYPE_Q2_0, fastllm::FLOAT32, 0, 256, 256, batch);
            RunResident<half>(GGML_TYPE_Q2_0, fastllm::FLOAT16, 0, 256, 256, batch);
            RunResident<__nv_bfloat16>(GGML_TYPE_Q2_0, fastllm::BFLOAT16, 0, 256, 256, batch);
        }
        std::puts("PASS: generic GGUF CUDA expert cache and resident fused kernels");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
