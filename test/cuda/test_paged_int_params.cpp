#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(cudaError_t state) {
    if (state != cudaSuccess) throw std::runtime_error(cudaGetErrorString(state));
}
static void require(bool passed, const char *message) {
    if (!passed) throw std::runtime_error(message);
}
int main() {
    check(cudaSetDevice(0));
    int cases = 0;
    for (int pages : {0,1,255,256,257,511,512,513,1024,1025,2048,2049,4096,4097}) {
        for (int metadata : {2,64}) {
            for (bool capture : {false,true}) {
                const int count = metadata * 3 + pages;
                std::vector<int> q(metadata), p(metadata), last(metadata), ids(pages);
                for (int i=0;i<metadata;++i) {q[i]=i*7+1;p[i]=i*11+2;last[i]=i*13+3;}
                for (int i=0;i<pages;++i) ids[i]=i*3+1;
                std::vector<int> expected=q;
                expected.insert(expected.end(),p.begin(),p.end());
                expected.insert(expected.end(),ids.begin(),ids.end());
                expected.insert(expected.end(),last.begin(),last.end());
                int32_t *device;
                check(cudaMalloc(&device,(count+2)*sizeof(int32_t)));
                check(cudaMemset(device,0xa5,(count+2)*sizeof(int32_t)));
                cudaGraph_t graph=nullptr;
                cudaGraphExec_t instance=nullptr;
                if(capture) check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
                bool ok=FastllmCudaUploadPagedIntParams(device+1,metadata,device+1+metadata,metadata,
                    device+1+metadata*2,pages,device+1+metadata*2+pages,metadata,
                    q.data(),p.data(),ids.data(),last.data());
                if(capture) {
                    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));
                    check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
                }
                require(ok==(pages<=4096),"incorrect upload/fallback selection");
                // Replaying after host arrays change must retain captured values.
                for(int replay=0;replay<(capture?2:1);++replay) {
                    if(capture) {
                        for(int &x:ids)x=-1;
                        check(cudaMemsetAsync(device,0xa5,(count+2)*sizeof(int32_t),cudaStreamPerThread));
                        check(cudaGraphLaunch(instance,cudaStreamPerThread));
                    }
                    check(cudaStreamSynchronize(cudaStreamPerThread));
                    std::vector<int> actual(count+2);
                    check(cudaMemcpy(actual.data(),device,actual.size()*sizeof(int),cudaMemcpyDeviceToHost));
                    require(actual.front()==int(0xa5a5a5a5u)&&actual.back()==int(0xa5a5a5a5u),"guard overwritten");
                    if(ok) require(std::vector<int>(actual.begin()+1,actual.end()-1)==expected,"uploaded values differ");
                    else for(int x:actual)require(x==int(0xa5a5a5a5u),"fallback wrote partial data");
                }
                if(instance)check(cudaGraphExecDestroy(instance));
                if(graph)check(cudaGraphDestroy(graph));
                check(cudaFree(device));
                ++cases;
            }
        }
    }
    std::cout<<"{\"cases\":"<<cases<<",\"pass\":true}"<<std::endl;
}
