#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace fastllm;
static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Gpu(Data &data) {
    data.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}

static void Run(int batch, int hidden, int inter, int experts, int topk) {
    std::mt19937 random(123 + batch + hidden + inter);
    std::uniform_real_distribution<float> value(-0.0625f, 0.0625f);
    auto tensor = [&](DataType dtype, int rows, int cols) {
        std::vector<float> values((size_t)rows * cols);
        for (float &v : values) v = value(random);
        return std::make_unique<Data>(dtype, std::vector<int>{rows,cols}, values);
    };
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data*> weights((experts+1)*2, nullptr), biases(weights.size(), nullptr);
    for (int e=0;e<experts;++e) for (int part=0;part<2;++part) {
        auto weight=tensor(BFLOAT16,part ? hidden : 2*inter,part ? inter : hidden);
        Gpu(*weight);weights[2*(e+1)+part]=weight.get();owned.push_back(std::move(weight));
    }
    auto input=tensor(FLOAT32,batch,hidden);Gpu(*input);
    Data index(fastllm::INT32,{batch,topk}), score(FLOAT32,{batch,topk});
    index.Allocate(false);score.Allocate(false);Gpu(index);Gpu(score);
    Data w1,w2,w3,actual,reference,r1,r2,r3;
    actual.Resize({batch,hidden});actual.ToDevice(DataDevice::CUDA,{0},false);
    reference.Resize({batch,hidden});reference.ToDevice(DataDevice::CUDA,{0},false);
    std::vector<int32_t> ids(batch*topk);
    std::vector<float> scales(batch*topk), expected(batch*hidden), result(expected.size());
    auto route=[&](int turn) {
        std::vector<int> pool(experts);std::iota(pool.begin(),pool.end(),0);
        for (int row=0;row<batch;++row) {
            std::shuffle(pool.begin(),pool.end(),random);
            for (int j=0;j<topk;++j) {
                // Mix shared routes and unrelated rows, including the pointer chunk boundary.
                ids[row*topk+j]=turn%2 ? pool[j] : ((experts-1-j*7)%experts+experts)%experts;
                scales[row*topk+j]=value(random)*8;
            }
        }
        Check(cudaMemcpy(index.cudaData,ids.data(),ids.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
        Check(cudaMemcpy(score.cudaData,scales.data(),scales.size()*sizeof(float),cudaMemcpyHostToDevice));
    };
    auto fallback=[&]() {
        Data hostIndex(fastllm::INT32,{batch,topk}), hostScore(FLOAT32,{batch,topk});
        hostIndex.Allocate(false);hostScore.Allocate(false);
        std::memcpy(hostIndex.cpuData,ids.data(),ids.size()*sizeof(int32_t));
        std::memcpy(hostScore.cpuData,scales.data(),scales.size()*sizeof(float));
        DoCudaMergeMOE(*input,reference,hostIndex,hostScore,r1,r2,r3,
            weights.data(),biases.data(),1.0f,MoeGateSwiglu,weights.size());
        Check(cudaMemcpy(expected.data(),reference.cudaData,expected.size()*sizeof(float),cudaMemcpyDeviceToHost));
    };
    auto verify=[&]() {
        Check(cudaMemcpy(result.data(),actual.cudaData,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for (size_t i=0;i<result.size();++i) {
            if (!std::isfinite(result[i]) || std::memcmp(&result[i],&expected[i],sizeof(float))) {
                std::cerr<<"mismatch batch="<<batch<<" hidden="<<hidden<<" inter="<<inter
                    <<" at "<<i<<" expected="<<expected[i]<<" actual="<<result[i]<<std::endl;
                throw std::runtime_error("grouped BF16 output differs from generic MoE");
            }
        }
    };
    auto grouped=[&]() {
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
        DoCudaMergeMOE(*input,actual,index,score,w1,w2,w3,
            weights.data(),biases.data(),1.0f,MoeGateSwiglu,weights.size());
        Require(!FastllmCudaMergeMOEUsedGraphUnsafeFallback(),"grouped path fell back to CPU");
    };
    for (int turn=0;turn<3;++turn) {route(turn);fallback();grouped();verify();}
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    grouped();
    Check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    Check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for (int turn=3;turn<6;++turn) {
        route(turn);fallback();
        Check(cudaGraphLaunch(exec,cudaStreamPerThread));verify();
    }
    Check(cudaGraphExecDestroy(exec));Check(cudaGraphDestroy(graph));
    auto supported=[&]() {return FastllmCudaFloat32MergeMOEBFloat16Indexed(
        *input,index,score,w3,w1,w2,actual,weights.data(),weights.size());};
    weights[2]->lockInCPU=true;Require(!supported(),"CPU-locked expert accepted");weights[2]->lockInCPU=false;
    weights[2]->isDiskWeight=true;Require(!supported(),"disk expert accepted");weights[2]->isDiskWeight=false;
    weights[2]->dataDeviceIds={1};Require(!supported(),"other-device expert accepted");weights[2]->dataDeviceIds={0};
    Data *saved=weights[3];weights[3]=nullptr;Require(!supported(),"missing expert accepted");weights[3]=saved;
    weights[0]=weights[2];Require(!supported(),"shared expert silently omitted");weights[0]=nullptr;
    auto type=weights[2]->dataType;weights[2]->dataType=FLOAT16;Require(!supported(),"non-BF16 expert accepted");weights[2]->dataType=type;
    index.ToDevice(DataDevice::CPU);Require(!supported(),"host routing accepted by grouped path");
    std::cout<<"PASS batch="<<batch<<" hidden="<<hidden<<" inter="<<inter<<" experts="<<experts<<std::endl;
}

int main() {
    try {
        int devices=0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        Check(cudaSetDevice(0));
        for (int batch : {1,2,3,4}) {
            Run(batch,128,31,17,5);
            Run(batch,64,16,513,10);
            Run(batch,2560,batch%2 ? 128 : 256,16,10);
        }
        Run(4,129,128,257,16);
        Run(4,129,256,17,16);
        std::cout<<"BF16 grouped MoE: exact generic reference, changing GPU routes, graph replay and fallback guards PASS"<<std::endl;
        return 0;
    } catch(const std::exception &error) {std::cerr<<error.what()<<std::endl;return 1;}
}
