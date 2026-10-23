#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static std::vector<char> Read(Data &data) {
    std::vector<char> bytes(data.GetBytes());
    Check(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "read failed");
    return bytes;
}
int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    try {
        SetDeviceMap({{"cuda:0",1}}); FastllmCudaSetDevice(0);
        int cases=0;
        for(auto sourceType:{FLOAT32,FLOAT16,BFLOAT16}) for(auto targetType:{FLOAT32,FLOAT16,BFLOAT16}) {
            if(sourceType==targetType) continue;
            for(int rows:{1,5,8}) {
                std::vector<float> values(rows*512);
                for(size_t i=0;i<values.size();++i) values[i]=(int(i%127)-63)*.03125f;
                Data owner(sourceType,{rows,512},values), expected(targetType,{rows,512},values);
                owner.ToDevice(DataDevice::CUDA,std::vector<int>{0});
                expected.ToDevice(DataDevice::CUDA,std::vector<int>{0});
                auto original=Read(owner), oracle=Read(expected);
                for(int iteration=0;iteration<8;++iteration) {
                    Data view(sourceType); view.Resize(owner.dims);
                    view.dataDevice=DataDevice::CUDA;view.dataDeviceIds={0};
                    view.cudaData=owner.cudaData;view.cudaDataBorrowed=true;
                    view.expansionSize=owner.expansionSize;view.expansionBytes=owner.expansionBytes;
                    ToDataType(view,targetType);
                    Check(!view.cudaDataBorrowed && view.cudaData!=owner.cudaData,"converted view must own separate storage");
                    Check(Read(view)==oracle,"conversion differs from exact dyadic oracle");
                    Check(Read(owner)==original,"conversion modified its owner");
                    // Allocate matching blocks after the conversion. The owner
                    // must remain busy in the pool throughout view teardown.
                    std::vector<std::unique_ptr<Data>> churn;
                    for(int j=0;j<32;++j) {
                        auto d=std::make_unique<Data>(sourceType);
                        d->Resize(owner.dims);d->dataDevice=DataDevice::CUDA;d->dataDeviceIds={0};d->Allocate();
                        Check(d->cudaData!=owner.cudaData,"conversion released borrowed storage to pool");
                        churn.push_back(std::move(d));
                    }
                    ++cases;
                }
                Check(Read(owner)==original,"owner changed after borrowed-view destruction");
            }
        }
        std::cout<<"PASS "<<cases<<" borrowed CUDA conversions: exact values, owner preserved, owned result and pool lifetime\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
