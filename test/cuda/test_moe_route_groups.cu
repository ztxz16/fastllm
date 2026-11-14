#include "devices/cuda/moe/fastllm-moe-route-groups.cuh"
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using fastllm::moe_routes::Plan;
static void Check(bool value) { if (!value) throw std::runtime_error("route grouping differs from stable CPU order"); }
static void Cuda(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
extern "C" int main() {
    try {
        std::mt19937_64 random(91287);
        int checks=0;
        for (int count : {1,2,7,31,32,33,40,80,127,128,129,255,256,257,511,512,1025}) {
            const size_t temporary=Plan::TemporaryBytes(count),bytes=Plan(nullptr,count,temporary).bytes;
            void *workspace; Cuda(cudaMalloc(&workspace,bytes));
            Plan plan(workspace,count,temporary);
            cudaGraph_t graph; cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            Check(plan.Build(cudaStreamPerThread));
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            for (int pattern=0;pattern<12;++pattern) {
                std::vector<uint64_t> keys(count),sorted(count),records(count);
                std::vector<int> values(count),order(count),entries(count),starts(count+1);
                for (int i=0;i<count;++i) {
                    keys[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?17:
                        pattern==3?uint64_t(count-i):pattern==4?uint64_t(i):random();
                    if(pattern>=5 && pattern<=9)keys[i]=keys[i]%23;
                    if(pattern>=10)keys[i]=keys[i]%4?0:0x7f0000000000ull+(keys[i]%127)*16384;
                    values[i]=int(random());
                }
                std::iota(order.begin(),order.end(),0);
                std::stable_sort(order.begin(),order.end(),[&](int a,int b){return keys[a]<keys[b];});
                Cuda(cudaMemcpy(plan.keys,keys.data(),count*sizeof(uint64_t),cudaMemcpyHostToDevice));
                Cuda(cudaMemcpy(plan.routes,values.data(),count*sizeof(int),cudaMemcpyHostToDevice));
                Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                int groups=0;
                Cuda(cudaMemcpy(&groups,plan.groups,sizeof(int),cudaMemcpyDeviceToHost));
                Cuda(cudaMemcpy(sorted.data(),plan.sortedKeys,count*sizeof(uint64_t),cudaMemcpyDeviceToHost));
                Cuda(cudaMemcpy(entries.data(),plan.entries,count*sizeof(int),cudaMemcpyDeviceToHost));
                Cuda(cudaMemcpy(records.data(),plan.records,groups*sizeof(uint64_t),cudaMemcpyDeviceToHost));
                Cuda(cudaMemcpy(starts.data(),plan.starts,(groups+1)*sizeof(int),cudaMemcpyDeviceToHost));
                int group=0;
                for(int i=0;i<count;++i) {
                    Check(sorted[i]==keys[order[i]] && entries[i]==values[order[i]]);
                    if(sorted[i] && (!i || sorted[i]!=sorted[i-1])) {
                        Check(group<groups && records[group]==sorted[i] && starts[group]==i);++group;
                    }
                }
                Check(group==groups && starts[groups]==count);++checks;
            }
            Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));Cuda(cudaFree(workspace));
        }
        std::printf("ROUTE_GROUPS_PASS checks=%d\n",checks);return 0;
    } catch(const std::exception &error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}
