#include "utils/qwen38_hybrid.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "gguf.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <exception>
#include <vector>

static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cuda(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
static void Gpu(fastllm::Data &d, bool copy) {
    d.ToDevice(fastllm::CUDA,std::vector<int>{0},copy); d.Allocate(false);
}

static void Run(ggml_type type, int rows, int columns, int outputs, bool biasEnabled,
                bool concurrent = false) {
    using namespace fastllm;
    Data x(FLOAT16,{rows,columns}), w(DATA_GGUF_FORMAT,type,{outputs,columns});
    Data y(FLOAT16,{rows+1,outputs}), bias;
    w.isGGUFData = true; w.Allocate(); x.Allocate();
    unsigned random = 1729;
    for (size_t i = 0; i < w.GetBytes(); ++i) {
        random = random*1664525u+1013904223u; w.cpuData[i] = random>>24;
    }
    for (size_t offset = 0; offset < w.GetBytes(); offset += ggml_type_size(type)) {
        auto *p = w.cpuData+offset;
        const uint16_t scale = __half_as_ushort(__float2half_rn(
            float(1+(offset/ggml_type_size(type))%4)/16384));
        if (type == GGML_TYPE_Q3_K) reinterpret_cast<block_q3_K *>(p)->d = scale;
        else if (type == GGML_TYPE_Q6_K) reinterpret_cast<block_q6_K *>(p)->d = scale;
        else {
            std::memcpy(p,&scale,2);
            if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) std::memcpy(p+2,&scale,2);
        }
    }
    std::vector<float> decoded(size_t(outputs)*columns);
    const auto decode = ggml_type_to_float(type); Check(decode,"CPU dequantizer unavailable");
    const size_t rowBytes = ggml_row_size(type,columns);
    for (int row = 0; row < outputs; ++row) {
        decode(w.cpuData+size_t(row)*rowBytes,decoded.data()+size_t(row)*columns,columns);
        // The independent oracle includes the new path's FP16 weight rounding.
        for (int col = 0; col < columns; ++col) {
            auto &v = decoded[size_t(row)*columns+col];
            Check(std::isfinite(v),"invalid packed fixture"); v = float(__float2half_rn(v));
        }
    }
    std::vector<half> inputs(size_t(rows)*columns);
    for (size_t i = 0; i < inputs.size(); ++i)
        inputs[i] = __float2half_rn(std::sin(i*.137f)*std::ldexp(.25f,int(i%7)-3));
    std::memcpy(x.cpuData,inputs.data(),inputs.size()*sizeof(half));
    std::vector<float> biasValues(outputs);
    if (biasEnabled) {
        bias.dataType = FLOAT32; bias.Resize({outputs}); bias.Allocate();
        for (int i = 0; i < outputs; ++i) biasValues[i] = (i%7-3)/128.f;
        std::memcpy(bias.cpuData,biasValues.data(),outputs*sizeof(float)); Gpu(bias,true);
    }
    Gpu(x,true); Gpu(w,true); Gpu(y,false);
    std::vector<half> actual(size_t(rows+1)*outputs), legacy(actual.size());
    Check(!qwen38ForwardActive, "Qwen dispatch context leaked between calls");
    if (rows >= 128 && !concurrent) {
        setenv("FASTLLM_QWEN38_PREFILL_DEQUANT","0",1);
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs), "generic projection baseline");
        Cuda(cudaMemcpy(legacy.data(),y.cudaData,legacy.size()*sizeof(half),cudaMemcpyDeviceToHost));
        setenv("FASTLLM_QWEN38_PREFILL_DEQUANT","1",1);
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs), "generic projection with Qwen environment");
        Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
        Check(!std::memcmp(legacy.data(),actual.data(),size_t(rows)*outputs*sizeof(half)),
              "Qwen environment changed another caller's projection");
    }
    Qwen38ForwardScope qwenDispatch;
    if (rows < 128) {
        setenv("FASTLLM_QWEN38_PREFILL_DEQUANT","0",1);
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs),"legacy small projection");
        Cuda(cudaMemcpy(legacy.data(),y.cudaData,legacy.size()*sizeof(half),cudaMemcpyDeviceToHost));
    }
    if (!concurrent) setenv("FASTLLM_QWEN38_PREFILL_DEQUANT","1",1);
    const bool arenaTest = std::getenv("FASTLLM_QWEN38_PREFILL_ARENA") &&
        std::strcmp(std::getenv("FASTLLM_QWEN38_PREFILL_ARENA"), "1") == 0;
    if (arenaTest && rows >= 128 && !concurrent) {
        setenv("FASTLLM_QWEN38_PREFILL_ARENA","0",1);
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs),"pooled prefill reference");
        Cuda(cudaMemcpy(legacy.data(),y.cudaData,legacy.size()*sizeof(half),cudaMemcpyDeviceToHost));
        setenv("FASTLLM_QWEN38_PREFILL_ARENA","1",1);
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
        Cuda(cudaMemset(y.cudaData,0x5a,y.GetBytes()));
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs),"prefill dequant projection");
        Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows*outputs; ++i) Check(std::isfinite(float(actual[i])),"nonfinite projection");
        const auto *tail = reinterpret_cast<const unsigned char *>(actual.data()+size_t(rows)*outputs);
        for (size_t i = 0; i < outputs*sizeof(half); ++i) Check(tail[i]==0x5a,"output tail overwritten");
        if (rows < 128) {
            Check(!std::memcmp(legacy.data(),actual.data(),size_t(rows)*outputs*sizeof(half)),
                "prefill option changed decode/verifier arithmetic");
            continue;
        }
        if (arenaTest && !concurrent) Check(!std::memcmp(legacy.data(),actual.data(),size_t(rows)*outputs*sizeof(half)),
            "shared prefill arena changed projection arithmetic");
        double squaredError = 0, squaredNorm = 0;
        // Cover first/last rows and output boundaries, including the 64 MiB
        // dequant-chunk boundary in the deliberately wide projection.
        for (int sample = 0; sample < 19; ++sample) {
            const int r = sample == 18 ? rows-1 : sample*997%rows;
            for (int colSample = 0; colSample < 131; ++colSample) {
                const int c = colSample == 130 ? outputs-1 : colSample*269%outputs;
                double sum = 0;
                for (int k = 0; k < columns; ++k)
                    sum += double(float(inputs[size_t(r)*columns+k]))*decoded[size_t(c)*columns+k];
                const float projected = float(__float2half_rn(float(sum)));
                const float expected = biasEnabled ? float(__float2half_rn(projected+float(__float2half_rn(biasValues[c])))) : projected;
                const float observed = float(actual[size_t(r)*outputs+c]);
                const double error = double(observed)-expected;
                Check(std::abs(error) <= .002*(1+std::abs(expected)),"dequant GEMM differs from FP64 dot oracle");
                squaredError += error*error; squaredNorm += double(expected)*expected;
            }
        }
        const double relative = std::sqrt(squaredError/std::max(squaredNorm,1e-30));
        Check(relative < .001,"dequant GEMM relative error exceeded FP16 tolerance");
        std::printf("PASS dequant type=%d rows=%d K=%d N=%d bias=%d relativeL2=%.8g\n",
            int(type),rows,columns,outputs,int(biasEnabled),relative);
    }
    if (arenaTest && !concurrent && type == GGML_TYPE_Q8_0 && rows == 128) {
        // Match production graph-pool registration and its pre-capture sync.
        // Warm the private fallback while eager calls use the separate arena.
        setenv("FASTLLM_QWEN38_PREFILL_ARENA","0",1);
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs),"capture private workspace warmup");
        Cuda(cudaDeviceSynchronize());
        setenv("FASTLLM_QWEN38_PREFILL_ARENA","1",1);
        void *graph = nullptr, *executable = nullptr;
        std::vector<void *> reserved;
        Check(FastllmCudaGraphMemoryPoolBegin(),"graph pool begin");
        Check(FastllmCudaGraphBeginCapture(),"graph capture begin");
        Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,rows,columns,outputs),"captured prefill fallback");
        Check(FastllmCudaGraphEndCapture(&graph),"graph capture end");
        Check(FastllmCudaGraphMemoryPoolEnd(reserved),"graph pool end");
        Check(FastllmCudaGraphInstantiate(graph, &executable),"graph instantiate");
        for (int repeat = 0; repeat < 2; ++repeat) {
            Cuda(cudaMemset(y.cudaData,0x5a,y.GetBytes()));
            Check(FastllmCudaGraphLaunch(executable),"graph launch");
            Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size()*sizeof(half),cudaMemcpyDeviceToHost));
            Check(!std::memcmp(legacy.data(),actual.data(),size_t(rows)*outputs*sizeof(half)),
                "captured prefill fallback changed projection arithmetic");
        }
        FastllmCudaGraphExecDestroy(executable);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
        std::puts("PASS captured prefill bypasses shared eager arena");
    }
}

int main() {
    try {
        if (!FastllmCudaGetDeviceCount()) return 77;
        FastllmCudaSetDevice(0);
        for (auto type : {GGML_TYPE_Q3_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,
                          GGML_TYPE_IQ4_XS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_Q8_0}) {
            Run(type,128,2560,641,false); Run(type,257,6144,129,true);
            Run(type,1,2560,129,true); Run(type,4,2560,129,false);
        }
        Run(GGML_TYPE_Q5_K,129,2560,16385,true);
        Run(GGML_TYPE_Q8_0,129,2560,16385,true);
        if (std::getenv("FASTLLM_QWEN38_PREFILL_ARENA") &&
                std::strcmp(std::getenv("FASTLLM_QWEN38_PREFILL_ARENA"), "1") == 0) {
            std::exception_ptr errors[2];
            std::thread workers[2];
            for (int i = 0; i < 2; ++i) workers[i] = std::thread([&,i] {
                try {
                    Cuda(cudaSetDevice(0));
                    for (int repeat = 0; repeat < 4; ++repeat)
                        Run(i ? GGML_TYPE_Q5_K : GGML_TYPE_Q8_0, 128 + i * 129,
                            i ? 6144 : 2560, 641 + i * 128, i, true);
                } catch (...) { errors[i] = std::current_exception(); }
            });
            for (auto &worker : workers) worker.join();
            for (auto &error : errors) if (error) std::rethrow_exception(error);
            std::puts("PASS prefill arena handoff between concurrent host streams");
        }
        std::puts("PASS: bounded GGUF prefill dequant, independent FP64 oracle and unchanged decode");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
