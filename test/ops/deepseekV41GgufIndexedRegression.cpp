// Compare fused GPU routing to the existing per-expert V4.1 CUDA reference.
#include "fastllm.h"
#include "utils.h"
#include "gguf.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
static std::vector<float> Read(const Data &data) {
    Check(data.dataType == FLOAT32, "expert sum must stay FP32");
    std::vector<float> out(data.Count(0));
    Cuda(cudaMemcpy(out.data(), data.cudaData, out.size()*sizeof(float), cudaMemcpyDeviceToHost));
    return out;
}

static bool groupedTest = false;
static bool sparseTest = false;
static int Case(int gt, int dt, int hidden, int inter, int device, bool missing) {
    FastllmCudaSetDevice(device);
    const int experts = sparseTest ? 384 : 8, topk = 6;
    std::mt19937 rng(317 + gt*17+dt);
    std::normal_distribution<float> normal(0, .04f);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights(2*(experts+1), nullptr);
    for (int expert = 0; expert < experts; ++expert) for (int part = 0; part < 2; ++part) {
        if (missing && expert == 3) continue;
        const int rows = part ? hidden : 2*inter, cols = part ? inter : hidden;
        auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, part ? dt : gt, std::vector<int>{rows, cols});
        w->disableGGUFRepack = true;
        std::vector<float> values(size_t(rows)*cols);
        for (float &x : values) x = normal(rng);
        const int type = part ? dt : gt;
        if (type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
            // These CUDA-only formats do not all have a CPU encoder. Construct valid
            // packed fixtures directly; the reference consumes the same bytes.
            w->Allocate(false);
            for (size_t i=0; i<w->GetBytes(); ++i) w->cpuData[i] = rng()>>24;
            for (size_t i=0; i<w->GetBytes(); i+=ggml_type_size(ggml_type(type))) {
                const uint16_t scale = float_to_half(.02f + .002f*(rng()%8));
                if (type == GGML_TYPE_IQ1_M) {
                    // Four high nibbles encode one shared FP16 scale. The
                    // lower twelve bits are valid subblock scales.
                    auto *sc = reinterpret_cast<uint16_t *>(w->cpuData+i+48);
                    for (int j=0; j<4; ++j) sc[j] = (sc[j]&0xfff) | ((scale>>(4*j)&15)<<12);
                } else std::memcpy(w->cpuData+i, &scale, sizeof(scale));
            }
        } else w->CreateFromOriData(WeightType::LINEAR, FLOAT32, (uint8_t *)values.data(), nullptr, nullptr);
        w->ToDevice(CUDA, {device}, true);
        weights[2*(expert+1)+part] = w.get();
        owned.push_back(std::move(w));
    }
    Data gate, activation, workspace, output, ref;
    int checks = 0;
    for (int rows : sparseTest ? std::vector<int>{512,1024} : groupedTest ? std::vector<int>{33,64,129} : std::vector<int>{1,2,8,32}) for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<float> x(size_t(rows)*hidden), scales(rows*topk);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = pattern == 2 ? 0.f : std::ldexp(normal(rng)*30.f, int(i/32%5)-2);
        Data input(FLOAT32, {rows, hidden}, x);
        ToDataType(input, BFLOAT16);
        Data indices(INT32, {rows, topk}); indices.Allocate();
        for (int row = 0; row < rows; ++row) for (int k = 0; k < topk; ++k) {
            ((int32_t *)indices.cpuData)[row*topk+k] = (7-k+row)%experts;
            scales[row*topk+k] = k == 0 ? 0.f : k == 1 ? -.17f : .25f + .13f*k;
        }
        Data scores(FLOAT32, {rows, topk}, scales);
        for (auto *v : {&input, &indices, &scores}) v->ToDevice(CUDA, {device}, true);
        const float limit = pattern == 0 ? 0.f : .8f;
        const bool accepted = FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
            weights.data(), weights.size(), indices, scores, 0.f, limit);
        if (groupedTest && (gt == GGML_TYPE_Q2_K || dt == GGML_TYPE_Q2_K)) {
            Check(!accepted, "unsupported grouped type did not preserve fallback");
            checks++;
            continue;
        }
        Check(accepted, "resident dispatch rejected fixture");
        const auto actual = Read(output);
        if (groupedTest && gt != GGML_TYPE_Q2_K && dt != GGML_TYPE_Q2_K) {
            const size_t ordinary = (size_t(rows)*(hidden/32) + size_t(rows*topk)*(inter/32))*36 +
                size_t(rows*topk)*hidden*2;
            Check(workspace.GetBytes() > ((ordinary+255)&~size_t(255)), "grouped test did not select MMQ");

        }
        // Per-row reference retains the arithmetic of decode independent of
        // fallback's large-batch GEMM selection and expert occupancy.
        std::vector<float> expected;
        for (int row = 0; row < rows; ++row) {
            Data xi, ii, ss;
            xi.FakeFrom(input, size_t(row)*hidden*2); xi.Resize({1, hidden});
            ii.FakeFrom(indices, size_t(row)*topk*4); ii.Resize({1, topk});
            ss.FakeFrom(scores, size_t(row)*topk*4); ss.Resize({1, topk});
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            DoCudaMergeMOEV41Fallback(xi, ref, ii, ss, weights.data(), weights.size(), 0.f, limit);
            Check(FastllmCudaMergeMOEUsedGraphUnsafeFallback(), "reference did not use per-expert fallback");
            auto values = Read(ref); expected.insert(expected.end(), values.begin(), values.end());
        }
        double err2=0, norm2=0, maxabs=0; int different=0;
        for (size_t i=0;i<actual.size();++i) {
            Check(std::isfinite(actual[i]), "non-finite result");
            const double diff=double(actual[i])-expected[i];
            different += actual[i]!=expected[i]; err2+=diff*diff;
            norm2+=double(expected[i])*expected[i]; maxabs=std::max(maxabs,std::abs(diff));
        }
        printf("gt=%d dt=%d h=%d i=%d device=%d missing=%d rows=%d pattern=%d different=%d maxabs=%.9g rel=%.9g\n",
            gt,dt,hidden,inter,device,missing,rows,pattern,different,maxabs,std::sqrt(err2/std::max(norm2,1e-30)));
        fflush(stdout);
        if (!groupedTest) Check(different == 0, "indexed differs from per-token CUDA reference");
        else Check(std::sqrt(err2/std::max(norm2,1e-30)) < .002, "grouped differs materially from per-token CUDA reference");
        if ((rows == 2 || rows == 64) && pattern == 1) {
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            Check(FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
                weights.data(), weights.size(), indices, scores, 0.f, limit), "graph dispatch");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            Check(Read(output) == actual, "graph replay changed output");
            // Replay must read changing routes, not baked-in host routing.
            std::vector<int32_t> changed(rows*topk);
            Cuda(cudaMemcpy(changed.data(), indices.cudaData, changed.size()*4, cudaMemcpyDeviceToHost));
            for (auto &e : changed) e = (e+3)%experts;
            Cuda(cudaMemcpy(indices.cudaData, changed.data(), changed.size()*4, cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            const auto replayChanged = Read(output);
            Check(FastllmCudaMergeMOEV41GGUFResident(input, gate, activation, workspace, output,
                weights.data(), weights.size(), indices, scores, 0.f, limit), "changed-route dispatch");
            Check(Read(output) == replayChanged, "graph replay did not consume changed GPU routes");
            Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        }
        checks++;
    }
    return checks;
}

int main(int argc, char **argv) {
    try {
        int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        SetThreads(4);
        const std::string mode = argc>1 ? argv[1] : "";
        groupedTest = mode.find("--grouped") == 0 || mode == "--sparse";
        sparseTest = mode == "--sparse";
        if (sparseTest) {
            int n = 0;
            for (auto pair : std::vector<std::pair<int,int>>{{GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_IQ1_S,GGML_TYPE_Q2_0}, {GGML_TYPE_Q2_0,GGML_TYPE_Q4_K}})
                n += Case(pair.first,pair.second,512,256,0,false);
            printf("PASS: V4.1 sparse routing grouped MMQ (%d checks)\n", n);
            return 0;
        }
        int checks=0;
        for(int gt : {GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M})
        for(int dt : {GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M})
            checks+=Case(gt,dt,512,256,0,false);
        checks+=Case(GGML_TYPE_Q3_K,GGML_TYPE_Q2_0,512,256,devices>1?1:0,true);
        if (mode == "--full" || mode == "--grouped-full") {
            for (auto pair : std::vector<std::pair<int,int>>{
                    {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_S}, {GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_IQ1_S,GGML_TYPE_Q2_0}, {GGML_TYPE_Q2_0,GGML_TYPE_IQ1_M},
                    {GGML_TYPE_Q2_0,GGML_TYPE_Q3_K}, {GGML_TYPE_Q2_0,GGML_TYPE_Q4_K}})
                checks+=Case(pair.first,pair.second,5120,2304,0,false);
        }
        printf("PASS: V4.1 resident GGUF %s (%d checks)\n", groupedTest ? "grouped prefill" : "indexed decode", checks);
        fflush(stdout);
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
