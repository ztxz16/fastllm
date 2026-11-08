#include "models/qwen4_exp.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace fastllm {
struct Qwen4PrefixCacheTestAccess {
    static void Inputs(int drafts) {
        const std::vector<int> tokens(drafts+1,31415);
        std::vector<int> positions(drafts+1);
        for (int i=0; i<=drafts; ++i) positions[i]=8190+i;
        Data ids,positionIds;
        Qwen4ExpModel::BuildPipelineVerificationInputs(tokens,positions,ids,positionIds);
        if (ids.dims!=std::vector<int>({1,drafts+1}) ||
            positionIds.dims!=ids.dims || ids.cpuData==positionIds.cpuData) {
            throw std::runtime_error("candidate row shape/storage is wrong");
        }
        for (int i=0; i<=drafts; ++i) {
            if (reinterpret_cast<float *>(ids.cpuData)[i]!=float(tokens[i]) ||
                reinterpret_cast<float *>(positionIds.cpuData)[i]!=float(positions[i])) {
                throw std::runtime_error("candidate buffers did not survive temporary destruction");
            }
        }
    }
};
}
int main() {
    unsetenv("FASTLLM_TP");
    try {
        for (int drafts:{1,2,4,8})
            fastllm::Qwen4PrefixCacheTestAccess::Inputs(drafts);
        std::cout << "PASS owned MTP candidate inputs for k=1/2/4/8\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
