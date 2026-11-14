#include "fastllm.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
extern "C" bool FastllmCudaMatMulFloat32(const Data &,Data &,const Data &,Data &,int,int,int);
static void Require(bool ok){if(!ok)throw std::runtime_error("FP32 auxiliary bias cache regression");}
int main(){
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    for(int rows:{1,5,17})for(bool withBias:{false,true}){
        const int cols=128,outputs=128;
        Data input(FLOAT32,{rows,cols},std::vector<float>(rows*cols,.5f));
        Data weight(FLOAT32,{outputs,cols},std::vector<float>(outputs*cols,.25f));
        Data bias(FLOAT32),result(FLOAT32,{rows,outputs});
        if(withBias){bias.Resize({outputs});bias.Allocate(2.0f);}
        for(Data *d:{&input,&weight,&bias,&result})d->ToDevice(DataDevice::CUDA,std::vector<int>{0});
        result.Allocate(false);
        // Compact router preparation may reserve these slots, then decline
        // the fast path. The ordinary Linear must initialize slot zero.
        weight.extraCudaData.resize(7,nullptr);
        Require(FastllmCudaMatMulFloat32(input,weight,bias,result,rows,cols,outputs));
        Require(cudaDeviceSynchronize()==cudaSuccess);
        Require(weight.extraCudaData.size()==7&&weight.extraCudaData[0]);
        result.ToDevice(DataDevice::CPU);
        for(int i=0;i<rows*outputs;++i)Require(std::fabs(reinterpret_cast<float*>(result.cpuData)[i]-(16.f+(withBias?2.f:0.f)))<1e-5f);
    }
    std::cout<<"FP32 BIAS CACHE PASS: reserved auxiliary slots, zero and explicit bias"<<std::endl;
}
