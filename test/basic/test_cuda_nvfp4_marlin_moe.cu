// Standalone regression for BF16 row-packed NVFP4 grouped Marlin.
// Exercises independent gate/up globals, sparse routing, all three tile sizes,
// CUDA Graph replay, rejected layouts and cache retirement/address reuse.
#include "fastllm.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include <algorithm>
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
static float Round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float Code(int c) { static const float v[] = {0,.5,1,1.5,2,3,4,6}; return v[c & 7] * (c & 8 ? -1 : 1); }
static unsigned Mix(unsigned x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
static void Move(Data &d) { d.ToDevice(DataDevice::CUDA, std::vector<int>{0}); }
struct Fixture {
    int hidden, intermediate, experts;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights;
    std::vector<std::vector<float>> decoded;
    Fixture(int seed, bool invalid = false, bool planar = false,
            int hidden = 256, int intermediate = 128, int experts = 16,
            bool directMemory = true, bool variedScales = false, bool slab = false)
        : hidden(hidden), intermediate(intermediate), experts(experts), weights(2 + experts * 2, nullptr) {
        const int H=hidden,I=intermediate,E=experts;
        for (int e = 0; e < E; ++e) for (int matrix = 0; matrix < 2; ++matrix) {
            int n = matrix ? H : 2 * I, k = matrix ? I : H, stride = 4 + k / 16 * 9;
            auto d = std::make_unique<Data>(planar ? DataType::NVFP4_BLOCK_16_E4M3 : DataType::NVFP4_BLOCK_16_E4M3_PACKED);
            d->blockK = 1; d->blockM = 16; d->directMemory = directMemory;
            if (slab) {
                d->isModelWeight = true;
                d->tpLinearType = matrix ? TP_LINEAR_COLUMN : TP_LINEAR_ROW;
                d->name = "test.mlp.experts." + std::to_string(e);
            }
            d->Resize({n, k}); d->Allocate(false);
            auto *bytes = reinterpret_cast<unsigned char *>(d->cpuData);
            std::vector<float> full(n * k);
            for (int r = 0; r < n; ++r) {
                float global = matrix ? .015625f : (r < I ? .03125f : .046875f);
                global *= 1.f + float(e % 3) * .25f;
                if (planar) {
                    if (r == 0 || (!matrix && r == I)) d->scales.push_back(global);
                } else std::memcpy(bytes + r * stride, &global, 4);
                for (int g = 0; g < k / 16; ++g) {
                    // Exercise unequal group scales as well as the uniform-scale fixtures.
                    int exponent = variedScales
                        ? int(Mix(r * (k / 16) + g + e * 719) % 7) - 3 : 0;
                    float groupScale = std::ldexp(1.0f, exponent);
                    // E4M3: biased exponent 7 and zero mantissa encode 1.0.
                    bytes[planar ? n * k / 2 + r * (k / 16) + g : r * stride + 12 + g * 9] =
                        (invalid && r == 0 && g == 0) ? 1 : 56 + exponent * 8;
                    for (int j = 0; j < 8; ++j) {
                        int c0 = Mix(r * k + g * 16 + j * 2 + e * n * k + seed * 719) % 16;
                        int c1 = Mix(r * k + g * 16 + j * 2 + 1 + e * n * k + seed * 719) % 16;
                        bytes[planar ? r * (k / 2) + g * 8 + j : r * stride + 4 + g * 9 + j] = c0 | (c1 << 4);
                        full[r * k + g * 16 + j * 2] = global * groupScale * Code(c0);
                        full[r * k + g * 16 + j * 2 + 1] = global * groupScale * Code(c1);
                    }
                }
            }
            Move(*d); weights[2 + e * 2 + matrix] = d.get(); owned.push_back(std::move(d)); decoded.push_back(std::move(full));
        }
    }
};
static void Run(Fixture &f, int m, int topk, float limit = 0.0f,
                float amplitude = .15f, bool expectMarlin = true, bool spreadRoutes = false) {
    const int H=f.hidden,I=f.intermediate,E=f.experts;
    bool bf16 = f.weights[2]->dataType == DataType::NVFP4_BLOCK_16_E4M3_PACKED;
    DataType dtype = bf16 ? DataType::BFLOAT16 : DataType::FLOAT16;
    Data x(dtype, {m,H}), y(dtype, {m,H}), a(dtype), b(dtype), c(dtype);
    Data ids(DataType::INT32, {m,topk}), scores(DataType::FLOAT32, {m,topk});
    x.Allocate(false); ids.Allocate(false); scores.Allocate(false);
    std::vector<float> xf(m * H), sf(m * topk); std::vector<int> ix(m * topk);
    for (int i = 0; i < m * H; ++i) {
        float value = amplitude * std::sin(i * .137f);
        xf[i] = bf16 ? Round(value) : __half2float(__float2half_rn(value));
        if (bf16) ((__nv_bfloat16 *)x.cpuData)[i] = __float2bfloat16_rn(xf[i]);
        else ((half *)x.cpuData)[i] = __float2half_rn(xf[i]);
    }
    for (int i = 0; i < m * topk; ++i) {
        ix[i] = spreadRoutes
            ? (Mix(i / topk + 319) % E + (i % topk) * 31) % E
            : (i / topk % 5 + (i % topk) * 3) % E; // include unused experts
        sf[i] = 1.f / topk + (i % topk) * .013f;
        ((int *)ids.cpuData)[i] = ix[i]; ((float *)scores.cpuData)[i] = sf[i];
    }
    Move(x); Move(ids); Move(scores);
    for (Data *d : {&y,&a,&b,&c}) {
        d->dataDevice = DataDevice::CUDA; d->dataDeviceIds = {0};
    }
    auto call = [&]() {
        if (limit > 0.0f) {
            // Exercise the model-facing operator, including limit forwarding
            // and bypass of fast paths that implement only ordinary SwiGLU.
            Move(ids); Move(scores);
            CudaMergeMOE op;
            ((BaseOperator*)&op)->Run("MergeMOE", {{"input",&x},{"output",&y},{"index",&ids},{"score",&scores},
                {"w1",&a},{"w2",&b},{"w3",&c},{"weights",(Data*)f.weights.data()},{"biass",nullptr}},
                {{"sharedScale",0.0f},{"swigluLimit",limit}},
                {{"weights___batch",(int)f.weights.size()},{"gateType",(int)MoeGateSwiglu}});
        } else {
            Check(FastllmCudaMergeMOENVFP4E4M3MarlinIndexed(x,a,b,y,f.weights.data(),f.weights.size(),
                (int32_t *)ids.cudaData,(float *)scores.cudaData,m,topk), "Marlin rejected supported input");
        }
    };
    call(); Cuda(cudaDeviceSynchronize());
    for (auto &w : f.owned) Check(expectMarlin ? w->cudaData == nullptr : w->cudaData != nullptr, "unexpected source weight residency");
    std::vector<uint16_t> actual(m * H), replay(m * H);
    auto asFloat = [bf16](uint16_t bits) {
        if (bf16) { __nv_bfloat16 v; std::memcpy(&v,&bits,2); return __bfloat162float(v); }
        half v; std::memcpy(&v,&bits,2); return __half2float(v);
    };
    Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size()*2,cudaMemcpyDeviceToHost));
    double err = 0, norm = 0;
    size_t gateClipped=0, upHighClipped=0, upLowClipped=0;
    for (int r = 0; r < m; ++r) {
        std::vector<double> ref(H,0);
        for (int t = 0; t < topk; ++t) {
            int e = ix[r * topk + t]; auto &g = f.decoded[e * 2]; auto &d = f.decoded[e * 2 + 1];
            std::vector<double> act(I);
            for (int n = 0; n < I; ++n) {
                double gate = 0, up = 0;
                for (int k = 0; k < H; ++k) { gate += double(xf[r * H + k]) * g[n * H + k]; up += double(xf[r * H + k]) * g[(I + n) * H + k]; }
                if (limit > 0.0f) {
                    gateClipped += gate > limit; upHighClipped += up > limit; upLowClipped += up < -limit;
                    gate = std::min(gate, double(limit));
                    up = std::max(-double(limit), std::min(up, double(limit)));
                }
                act[n] = gate / (1 + std::exp(-gate)) * up;
            }
            for (int n = 0; n < H; ++n) { double sum = 0; for (int k = 0; k < I; ++k) sum += act[k] * d[n * I + k]; ref[n] += sum * sf[r * topk + t]; }
        }
        for (int n = 0; n < H; ++n) { double v = asFloat(actual[r * H + n]); Check(std::isfinite(v), "nonfinite output"); err += (v-ref[n])*(v-ref[n]); norm += ref[n]*ref[n]; }
    }
    double nrmse = std::sqrt(err/norm);
    Check(nrmse < .01, "FP64 reference NRMSE exceeds 1 percent");
    if (limit > 0.0f) Check(gateClipped && upHighClipped && upLowClipped, "fixture did not exercise all clamp branches");
    std::printf("mode=%s dtype=%s m=%d topk=%d limit=%g FP64_nrmse=%.8g clipped=%zu/%zu/%zu\n",
        expectMarlin ? "marlin" : "native",bf16 ? "bf16" : "fp16",m,topk,limit,nrmse,gateClipped,upHighClipped,upLowClipped);
    std::fflush(stdout);
    if (!expectMarlin) return; // Native fallback synchronizes routing on CPU.
    cudaGraph_t graph; cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)); call();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph)); Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::vector<__nv_bfloat16> changedInput(spreadRoutes ? m * H : 0);
    for (int i = 0; i < 3; ++i) {
        if (spreadRoutes) {
            // Replay captured metadata/GEMMs with new routes, scores and input.
            for (auto &expert : ix) expert = (expert + 3) % E;
            for (auto &score : sf) score *= .875f;
            for (int j = 0; j < m * H; ++j) {
                xf[j] = Round(-xf[j] * .9375f);
                changedInput[j] = __float2bfloat16_rn(xf[j]);
            }
            Cuda(cudaMemcpyAsync(x.cudaData,changedInput.data(),m*H*2,cudaMemcpyHostToDevice,cudaStreamPerThread));
            Cuda(cudaMemcpyAsync(ids.cudaData,ix.data(),ix.size()*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
            Cuda(cudaMemcpyAsync(scores.cudaData,sf.data(),sf.size()*4,cudaMemcpyHostToDevice,cudaStreamPerThread));
            call(); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size()*2,cudaMemcpyDeviceToHost));
        }
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(replay.data(),y.cudaData,replay.size()*2,cudaMemcpyDeviceToHost));
        Check(std::memcmp(actual.data(),replay.data(),actual.size()*2)==0,"graph result changed");
    }
    Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    std::printf("dtype=%s m=%d topk=%d FP64_nrmse=%.8g graph=bitwise_equal\n",bf16 ? "bf16" : "fp16",m,topk,nrmse);
}

static void CheckActivation() {
    const float gates[] = {20,10,9,-20,-10,0,1,1};
    const float ups[] = {20,-20,9,20,-20,10,10,-10};
    for (auto dtype : {DataType::BFLOAT16,DataType::FLOAT16,DataType::FLOAT32}) {
        Data x(dtype,{1,16}), y;
        x.Allocate(false);
        for (int i=0;i<16;++i) {
            float value = i<8 ? gates[i] : ups[i-8];
            if (dtype==DataType::BFLOAT16) ((__nv_bfloat16*)x.cpuData)[i]=__float2bfloat16_rn(value);
            else if(dtype==DataType::FLOAT16) ((half*)x.cpuData)[i]=__float2half_rn(value);
            else ((float*)x.cpuData)[i]=value;
        }
        Move(x);
        for (float limit : {0.0f,10.0f}) {
            Check(FastllmCudaSwigluClamped(x,limit,y),"clamped activation rejected");
            Cuda(cudaDeviceSynchronize());
            Check(y.dataType==dtype,"activation changed dtype");
            std::vector<unsigned char> raw(y.GetBytes());
            Cuda(cudaMemcpy(raw.data(),y.cudaData,raw.size(),cudaMemcpyDeviceToHost));
            for (int i=0;i<8;++i) {
                double g=gates[i],u=ups[i];
                if(limit>0) {g=std::min(g,double(limit));u=std::max(-double(limit),std::min(u,double(limit)));}
                double ref=(g/(1+std::exp(-g)))*u;
                if(dtype==DataType::BFLOAT16) ref=Round(ref);
                else if(dtype==DataType::FLOAT16) ref=__half2float(__float2half_rn(ref));
                float actual=dtype==DataType::FLOAT32 ? ((float*)raw.data())[i] :
                    dtype==DataType::BFLOAT16 ? __bfloat162float(((__nv_bfloat16*)raw.data())[i]) :
                    __half2float(((half*)raw.data())[i]);
                Check(std::abs(actual-ref)<=std::max(1e-8,std::abs(ref)*2e-6),"activation clamp boundary mismatch");
            }
        }
    }
    std::puts("Activation boundary/dtype PASS");
}

static void CheckGroupedRows(Fixture &f, int rows, int topk, bool foreignScratch = false, bool duplicateRoutes = false) {
    const int H = f.hidden;
    Data x(BFLOAT16, {rows, H}), ids(INT32, {rows, topk}), scores(FLOAT32, {rows, topk});
    x.Allocate();
    ids.Allocate();
    scores.Allocate();
    for (int i = 0; i < rows * H; ++i)
        ((__nv_bfloat16 *)x.cpuData)[i] = __float2bfloat16_rn(.3f * std::sin(i * .031f));
    for (int i = 0; i < rows * topk; ++i) {
        ((int *)ids.cpuData)[i] = duplicateRoutes ? 0 : (i / topk * 3 + i % topk * 7) % f.experts;
        ((float *)scores.cpuData)[i] = (1.f + i % topk * .07f) / topk;
    }
    Move(x);
    Move(ids);
    Move(scores);
    Data gate(BFLOAT16), act(BFLOAT16), actual(BFLOAT16), reference(BFLOAT16, {rows, H});
    reference.dataDevice = DataDevice::CUDA;
    reference.dataDeviceIds = {0};
    reference.Allocate(false);
    auto single = [&]() {
        for (int r = 0; r < rows; ++r) {
            Data in(BFLOAT16, {1, H}), out(BFLOAT16, {1, H});
            in.FakeFrom(x, (size_t)r * H * 2);
            out.FakeFrom(reference, (size_t)r * H * 2);
            Check(FastllmCudaMergeMOENVFP4E4M3MarlinIndexed(
                      in, gate, act, out, f.weights.data(), f.weights.size(),
                      (int32_t *)ids.cudaData + r * topk, (float *)scores.cudaData + r * topk, 1,
                      topk),
                  "single-row path rejected");
        }
    };
    auto multi = [&]() {
        Check(FastllmCudaMergeMOENVFP4E4M3MarlinRows(x, gate, act, actual, f.weights.data(),
                                                     f.weights.size(), (int32_t *)ids.cudaData,
                                                     (float *)scores.cudaData, rows, topk),
              "grouped rows rejected");
    };
    single();
    Cuda(cudaDeviceSynchronize());
    if (foreignScratch) {
        // Reproduce serial layer placement: workspace storage belongs to the
        // previous GPU, while the next invocation's input and weights are on 0.
        actual.Resize({rows, H});
        for (Data *d : {&gate, &act, &actual}) {
            d->ToDevice(DataDevice::CUDA, std::vector<int>{1}, false);
            d->Allocate(false);
        }
        FastllmCudaSetDevice(0);
    }
    multi();
    Cuda(cudaDeviceSynchronize());
    for (Data *d : {&gate, &act, &actual}) {
        cudaPointerAttributes attributes{};
        Cuda(cudaPointerGetAttributes(&attributes, d->cudaData));
        Check(attributes.device == 0, "grouped-row workspace stayed on another GPU");
    }
    std::vector<uint16_t> a(rows * H), b(rows * H);
    double largestRelative = 0;
    auto compare = [&]() {
        Cuda(cudaMemcpy(a.data(), reference.cudaData, a.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(b.data(), actual.cudaData, b.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
        // Grouping changes the reduction layout versus single-token GEMM.
        // Keep the existing MoE FP64 reference's 1% relative error bound and
        // additionally reject nonfinite values and large isolated errors.
        double error = 0, norm = 0, maximum = 0, scale = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            double expected = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16 *>(&a[i]));
            double value = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16 *>(&b[i]));
            Check(std::isfinite(expected) && std::isfinite(value), "nonfinite grouped output");
            error += (expected - value) * (expected - value);
            norm += expected * expected;
            maximum = std::max(maximum, std::abs(expected - value));
            scale = std::max(scale, std::abs(expected));
        }
        double relative = std::sqrt(error / std::max(norm, 1e-30));
        largestRelative = std::max(largestRelative, relative);
        Check(relative <= .01 && maximum <= 1e-6 + .02 * scale,
              "grouped rows numerical error");
    };
    compare();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    multi();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<int> changedIds(rows * topk);
    std::vector<__nv_bfloat16> changedInput(rows * H);
    for (int iteration = 0; iteration < 4; ++iteration) {
        // New routing and activations must be consumed by the same graph.
        for (int i = 0; i < rows * topk; ++i)
            changedIds[i] = duplicateRoutes ? iteration : (iteration * 5 + i / topk * 3 + i % topk * 7) % f.experts;
        for (int i = 0; i < rows * H; ++i)
            changedInput[i] = __float2bfloat16_rn(.3f * std::sin(i * .031f + iteration));
        Cuda(cudaMemcpyAsync(ids.cudaData, changedIds.data(), ids.GetBytes(),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
        Cuda(cudaMemcpyAsync(x.cudaData, changedInput.data(), x.GetBytes(), cudaMemcpyHostToDevice,
                             cudaStreamPerThread));
        single();
        multi();
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        std::vector<uint16_t> eager(rows * H);
        Cuda(cudaMemcpy(eager.data(), actual.cudaData, eager.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        compare();
        Check(eager == b, "grouped eager and graph differ");
    }
    // Later routes cannot affect any retained prefix. Check all prefix
    // lengths, including routes that merge with earlier experts' tiles.
    const auto prior = b;
    const auto originalIds = changedIds;
    for (int keep = 1; keep < rows; ++keep) {
        changedIds = originalIds;
        for (int i = keep * topk; i < rows * topk; ++i)
            changedIds[i] = (changedIds[i] + 3) % f.experts;
        Cuda(cudaMemcpyAsync(ids.cudaData, changedIds.data(), ids.GetBytes(),
                             cudaMemcpyHostToDevice, cudaStreamPerThread));
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(b.data(), actual.cudaData, b.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
        Check(std::equal(prior.begin(), prior.begin() + keep * H, b.begin()),
              "suffix routes changed prefix");
        auto replay = b;
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(b.data(), actual.cudaData, b.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
        Check(replay == b, "grouped graph replay differs");
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
    std::printf("grouped_rows H=%d I=%d rows=%d topk=%d relative_max=%g eager/graph/prefix/replay=bitwise_equal\n", H,
                f.intermediate, rows, topk, largestRelative);
}

int main(int argc, char **argv) { try {
    int devices = 0; cudaDeviceProp prop{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 ||
        cudaGetDeviceProperties(&prop,0) != cudaSuccess || prop.major < 8 ||
        prop.sharedMemPerBlockOptin < 71680) return 77;
    FastllmCudaSetDevice(0); SetThreads(4);
    CheckActivation();
    if (argc == 2 && std::strcmp(argv[1], "--grouped-decode") == 0) {
        Fixture f(17);
        // CPU FP64 reference, clipping and graph replay with changing routes
        // cover grouped batches, shrinking batches and exact verification.
        for (int threshold : {0, 10}) {
            FastllmCudaSetLinearExactBatchThreshold(threshold);
            for (int m : {8, 7, 6, 5, 4, 3, 2, 1, 9, 8}) {
                Run(f, m, 8, 10.0f, 15.0f, true, true);
            }
        }
        FastllmCudaSetLinearExactBatchThreshold(0);
        std::puts("Grouped decode PASS"); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--cross-device-rows") == 0) {
            if (devices < 2)
                return 77;
            Fixture f(59, false, false, 4096, 2048, 16, true, true);
            for (int rows : {2, 8})
                for (int topk : {1, 8})
                    CheckGroupedRows(f, rows, topk, true);
            std::puts("Cross-device grouped rows PASS");
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--grouped-rows") == 0) {
            for (auto shape :
                 std::vector<std::pair<int, int>>{{256, 128}, {4096, 256}, {4096, 2048}}) {
                Fixture f(53, false, false, shape.first, shape.second, 16, true, true);
                for (int rows : {2, 3, 4, 5, 6, 7, 8})
                    for (int topk : {1, 8, 16})
                        CheckGroupedRows(f, rows, topk);
            }
            {
                Fixture f(61, false, false, 4096, 256, 16, true, true);
                CheckGroupedRows(f, 8, 16, false, true);
            }
            std::puts("Grouped rows PASS");
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--tp-slab") == 0) {
        FastllmCudaSetWeightSlabBytes(16ULL << 20);
        Fixture f(47, false, false, 4096, 256, 8, false, true, true);
        for (auto &w : f.owned) Check(FastllmCudaIsWeightSlabPointer(w->cudaData), "expected slab source");
        for (int m : {1, 9, 512}) Run(f, m, 8, 0.0f, .15f, true, true);
        for (auto &w : f.owned) Check(w->cudaData == nullptr, "slab source was not released");
        std::puts("TP slab source PASS"); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--decode-tiles") == 0) {
        // Keep the single-token reduction checks small enough for Sanitizer.
        Fixture f(43, false, false, 1536, 768, 16, true, true);
        Run(f, 1, 1, 0.0f, .15f, true, true);
        Run(f, 1, 16, 0.0f, .15f, true, true);
        Run(f, 1, 16, 10.0f, 15.0f, true, true);
        std::puts("Decode tile scheduling PASS");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--narrow-prefill") == 0) {
        Fixture narrow(19,false,false,256,256,16,true);
        for (int m : {9,32,33,32}) Run(narrow,m,8,0.0f,.15f,true,true);
        std::puts("Narrow prefill boundary PASS"); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--clamped-small") == 0) {
        Fixture f(17); Run(f,33,8,10.0f,15.0f);
        std::puts("Clamped small PASS"); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--clamped-native") == 0) {
        // Pool-backed weights cannot be retired by Marlin and must use the
        // native fallback, even when their matrix dimensions are supported.
        Fixture f(17,false,false,256,128,16,false);
        for (int m : {1,8,33,129}) Run(f,m,8,10.0f,15.0f,false);
        std::puts("Clamped native PASS"); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--route-capacity") == 0) {
        Fixture legacy(9,false,true); Run(legacy,9,16);
        std::puts("Route capacity boundary PASS"); return 0;
    }
    { Fixture invalid(0,true); Check(!FastllmCudaPrepareNVFP4E4M3Moe(invalid.weights.data(),invalid.weights.size()),"unrepresentable scales accepted");
      for(auto &w:invalid.owned) Check(w->cudaData!=nullptr,"failed preparation retired source"); }
    for(int pass=0;pass<2;++pass) { Fixture f(3+pass); for(int m:{1,8,9,10,32,33,129}) Run(f,m,2); Run(f,1,8); }
    { Fixture legacy(9,false,true); Run(legacy,1,2); Run(legacy,129,2); Run(legacy,9,16); }
    { Fixture alternate(11,false,false,512,256,7); Run(alternate,1,1); Run(alternate,9,7); Run(alternate,65,3); }
    { Fixture clamped(17); for (int m : {1,8,9,10,32,33,129,1024}) Run(clamped,m,8,10.0f,15.0f); }
    { Fixture clampedHalf(17,false,true); Run(clampedHalf,33,8,10.0f,15.0f); }
    { Fixture glm(19,false,false,4096,2048,8); Run(glm,1,8,10.0f,15.0f); }
    // Single-token tile scheduling: both projections, asymmetric widths,
    // routing changes under Graph replay, and ordinary/clamped SwiGLU.
    {
        Fixture decode(37, false, false, 4096, 2048, 8, true, true);
        for (int topk : {1, 5, 6, 8})
            Run(decode, 1, topk, 0.0f, .15f, true, true);
        Run(decode, 1, 8, 10.0f, 15.0f, true, true);
    }
    {
        Fixture asymmetric(41, false, false, 4096, 1024, 8, true, true);
        Run(asymmetric, 1, 8, 0.0f, .15f, true, true);
    }

    { Fixture fallback(23,false,false,128,64,8); Run(fallback,33,8,10.0f,30.0f,false); }
    { Fixture narrow(19,false,false,256,256,256,true);
      for (int m : {128,129,255,256,511,512,513,1024}) Run(narrow,m,8,0.0f,.15f,true,true); }
    // Gate/up halves of 128 columns must retain the old 128-column tile.
    { Fixture unaligned(29,false,false,256,128,16,true); Run(unaligned,32,8,0.0f,.15f,true,true); }
    { Fixture uneven(23,false,false,512,256,7,true);
      for (int m : {9,10,37,38,10}) Run(uneven,m,3,0.0f,.15f,true,true); }
    // Cover the merge boundary: narrow-prefill routing with upstream clipping.
    { Fixture narrowClamped(31,false,false,256,256,16,true);
      Run(narrowClamped,32,8,10.0f,15.0f,true,true); }
    Cuda(cudaDeviceSynchronize()); std::puts("PASS"); return 0;
} catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; } }
