#include "fastllm.h"
#include "devices/cuda/fastllm-cuda-native-prefill.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <cstdio>
using namespace fastllm;
void require(bool v,const char*s){if(!v)throw std::runtime_error(s);}
int main(){
 FastllmCudaSetDevice(0);SetDeviceMap({{"cuda",1}});
 int N=256,K=256,M=3;size_t bytes=size_t(N)*K*12/16,codes=size_t(N)*K/2,scales=size_t(N)*K/16;
 Data w(DataType::NVFP4_BLOCK_16,{N,K});w.blockM=16;w.blockK=1;w.scales={1.f};w.isModelWeight=true;w.Allocate(false);
 memset(w.cpuData,0,bytes);memset(w.cpuData,0x21,codes);memset(w.cpuData+codes,0x38,scales);float global=128;memcpy(w.cpuData+codes+scales,&global,4);
 w.ToDevice(DataDevice::CUDA,{0},true);w.cudaNativeNvfp4Layout=true;w.IsRepacked=true;
 Data copy(w);require(copy.cudaNativeNvfp4Layout && copy.IsRepacked,"copy marker");require(copy.blockM==16 && copy.blockK==1,"copy metadata");
 Data input(DataType::FLOAT16,{M,K},std::vector<float>(M*K,.125f));input.ToDevice(DataDevice::CUDA,{0},true);
 Data output(DataType::FLOAT16,{M,N});output.dataDevice=DataDevice::CUDA;output.dataDeviceIds={0};output.Allocate(false);Data bias;
 require(FastllmCudaTryNativeNvfp4Linear(input,copy,bias,output,M,K,N),"native copy linear");
 std::vector<half>got(M*N);FastllmCudaCopyFromDeviceToHost(got.data(),output.cudaData,got.size()*2);for(half x:got)require(__half2float(x)==24.f,"native copy value");
 require(!FastllmCudaNativeNvfp4LayoutFusedCanRun(input,copy,bias,input,false),"alias CanRun");
 copy.ToDevice(DataDevice::CPU,{0},true);require(!copy.cudaNativeNvfp4Layout && !copy.IsRepacked,"CPU restore marker");
 std::vector<uint8_t>raw(bytes);for(size_t i=0;i<size_t(N)*K/16;i++){memset(raw.data()+i*12,0x21,8);float one=1;memcpy(raw.data()+i*12+8,&one,4);}
 require(!memcmp(copy.cpuData,raw.data(),bytes),"CPU restore bytes");require(w.cudaNativeNvfp4Layout,"copy changed source");
 w.Reshape({128,512});require(!w.cudaNativeNvfp4Layout && !w.IsRepacked,"reshape marker");w.ToDevice(DataDevice::CPU,{0},true);require(!memcmp(w.cpuData,raw.data(),bytes),"reshape raw bytes");
 copy.FreeSpace();require(!copy.cudaNativeNvfp4Layout,"free marker");puts("copy, native dispatch, alias rejection, CPU restore, reshape and free passed");
}
