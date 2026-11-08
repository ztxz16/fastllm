#include "models/qwen4_exp.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Commit() {
        SetDeviceMap({{"cpu",1}}); SetThreads(1);
        Qwen4ExpModel model;
        model.stageId = 1;
        model.stageBegin = 24;
        model.stageEnd = 28;
        model.threadTpRank = 0;
        struct Restore { Qwen4ExpModel &m; ~Restore(){m.threadTpRank=-1;} } restore{model};
        model.deviceMap = {{"cpu",1}};
        model.indexerHeadDim = 2;
        model.linearLayers.assign(48,true);
        for(int i=24;i<28;++i) model.linearLayers[i]=false;
        std::vector<std::pair<Data,Data>> cache;
        for(int i=0;i<48;++i) cache.emplace_back(
            Data(FLOAT32,{1,3,2},std::vector<float>(6,float(i))),
            Data(FLOAT32,{1,3,2},std::vector<float>(6,float(i+100))));
        Qwen4ExpModel::RequestState state;
        for(int i=24;i<28;++i) {
            state.indexerTailKeyTensors[i] = std::make_shared<Data>(
                FLOAT32,std::vector<int>{3,2},std::vector<float>(6,0.25f));
            state.indexerTailPositionTensors[i] = std::make_shared<Data>(
                FLOAT32,std::vector<int>{3},std::vector<float>{0,1,2});
        }
        Qwen4ExpModel::TargetRuntimeCheckpoint checkpoint;
        checkpoint.keyLengths.assign(48,0);
        checkpoint.valueLengths.assign(48,0);
        Qwen4ExpModel::TargetVerificationCapture capture; // Stage 1 never owns PLE input.
        Data ids(FLOAT32,{1,3},{1,2,3}),positions(FLOAT32,{1,3},{0,1,2});
        model.CommitTargetVerificationPrefix(ids,positions,{1,2,3},1,
                                             cache,state,checkpoint,capture);
        for(int i=0;i<48;++i) {
            const int expected=i>=24&&i<28?1:3;
            if(cache[i].first.dims[1]!=expected || cache[i].second.dims[1]!=expected)
                throw std::runtime_error("stage 1 rejection commit crossed the layer boundary");
        }
        for(int i=24;i<28;++i) {
            if(state.indexerTailKeyTensors.at(i)->dims!=std::vector<int>({1,2}))
                throw std::runtime_error("stage 1 QSA tail was not truncated");
        }
    }
};
}
int main() {
    unsetenv("FASTLLM_TP");
    pid_t child=fork(); if(child<0)return 1;
    if(child==0) {
        try{fastllm::Qwen4PrefixCacheTestAccess::Commit();_exit(42);}
        catch(...){_exit(1);}
    }
    int status=0;
    if(waitpid(child,&status,0)!=child || !WIFEXITED(status) || WEXITSTATUS(status)!=42) {
        std::cerr<<"FAIL stage 1 rejection commit, child status="<<status<<'\n';return 1;
    }
    std::cout<<"PASS stage 1 rejected suffix: QSA truncated, other stage untouched, no PLE access\n";
}
