#include "models/qwen4_exp.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Boundary(int rank) {
        SetDeviceMap({{"cpu",1}});
        SetThreads(1);
        Qwen4ExpModel model;
        model.deviceMap = {{"cpu",1}};
        model.preparedWeights = true;
        model.block_cnt = 48;
        model.embed_dim = 8;
        model.hcCount = 4;
        model.stageBegin = model.stageEnd = 0; // Skip layers; exercise real boundary code.
        model.threadTpRank = rank;
        model.stageId = 0;
        model.pipelineStageActive = true;
        Data output;
        model.pipelineOutputHidden = rank == 0 ? &output : nullptr;
        Data ids(FLOAT32,{1,1},{1.0f}), positions(FLOAT32,{1,1},{0.0f});
        Data embedding(FLOAT32,{1,1,8},std::vector<float>(8,0.25f));
        std::vector<std::pair<Data,Data>> cache(48);
        model.ForwardTarget(1,ids,Data(),positions,cache,
            GenerationConfig(),LastTokensManager(),nullptr,nullptr,nullptr,
            nullptr,nullptr,false,false,false,nullptr,false,&embedding);
        model.threadTpRank = -1; // Do not fabricate a CUDA owner for the CPU fixture.
        if (rank == 0 && (output.dims != std::vector<int>({1,1,32}) ||
                         output.cpuData == nullptr)) {
            throw std::runtime_error("stage leader lost its boundary hidden");
        }
    }
};
}

int main() {
    unsetenv("FASTLLM_TP");
    for (int rank = 0; rank < 5; ++rank) {
        const pid_t child = fork();
        if (child < 0) return 1;
        if (child == 0) {
            try {
                fastllm::Qwen4PrefixCacheTestAccess::Boundary(rank);
                // Distinguish returning normally from FastLLM's exit(0) error path.
                _exit(42);
            } catch (const std::exception &error) {
                std::fprintf(stderr,"rank=%d %s\n",rank,error.what());
                _exit(1);
            }
        }
        int status=0;
        if (waitpid(child,&status,0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 42) {
            std::fprintf(stderr,"FAIL real ForwardTarget boundary rank=%d status=%d\n",rank,status);
            return 1;
        }
    }
    std::cout << "PASS real ForwardTarget boundary: leader publishes; four followers exit without a destination\n";
    return 0;
}
