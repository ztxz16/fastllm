// Standalone regression for BF16 row-packed NVFP4 grouped Marlin.
// Exercises independent gate/up globals, sparse routing, both tile sizes,
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
            bool directMemory = true)
        : hidden(hidden), intermediate(intermediate), experts(experts), weights(2 + experts * 2, nullptr) {
        const int H=hidden,I=intermediate,E=experts;
        for (int e = 0; e < E; ++e) for (int matrix = 0; matrix < 2; ++matrix) {
            int n = matrix ? H : 2 * I, k = matrix ? I : H, stride = 4 + k / 16 * 9;
            auto d = std::make_unique<Data>(planar ? DataType::NVFP4_BLOCK_16_E4M3 : DataType::NVFP4_BLOCK_16_E4M3_PACKED);
            d->blockK = 1; d->blockM = 16; d->directMemory = directMemory;
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
                    bytes[planar ? n * k / 2 + r * (k / 16) + g : r * stride + 12 + g * 9] = (invalid && r == 0 && g == 0) ? 1 : 56; // E4M3 1
                    for (int j = 0; j < 8; ++j) {
                        int c0 = Mix(r * k + g * 16 + j * 2 + e * n * k + seed * 719) % 16;
                        int c1 = Mix(r * k + g * 16 + j * 2 + 1 + e * n * k + seed * 719) % 16;
                        bytes[planar ? r * (k / 2) + g * 8 + j : r * stride + 4 + g * 9 + j] = c0 | (c1 << 4);
                        full[r * k + g * 16 + j * 2] = global * Code(c0);
                        full[r * k + g * 16 + j * 2 + 1] = global * Code(c1);
                    }
                }
            }
            Move(*d); weights[2 + e * 2 + matrix] = d.get(); owned.push_back(std::move(d)); decoded.push_back(std::move(full));
        }
    }
};
static void Run(Fixture &f, int m, int topk, float limit = 0.0f,
                float amplitude = .15f, bool expectMarlin = true) {
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
        ix[i] = (i / topk % 5 + (i % topk) * 3) % E; // include unused experts
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
    for (int i=0;i<3;++i) { Cuda(cudaGraphLaunch(exec,cudaStreamPerThread)); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(replay.data(),y.cudaData,replay.size()*2,cudaMemcpyDeviceToHost)); Check(std::memcmp(actual.data(),replay.data(),actual.size()*2)==0,"graph result changed"); }
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

int main(int argc, char **argv) { try {
    int devices = 0; cudaDeviceProp prop{};
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0 ||
        cudaGetDeviceProperties(&prop,0) != cudaSuccess || prop.major < 8 ||
        prop.sharedMemPerBlockOptin < 71680) return 77;
    FastllmCudaSetDevice(0); SetThreads(4);
    CheckActivation();
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
    { Fixture fallback(23,false,false,128,64,8); Run(fallback,33,8,10.0f,30.0f,false); }
    Cuda(cudaDeviceSynchronize()); std::puts("PASS"); return 0;
} catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; } }
