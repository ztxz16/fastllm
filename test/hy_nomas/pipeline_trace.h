#pragma once
#include "models/hy_nomas/runtime.h"
#include <cstdlib>
#include <fstream>
#include <stdexcept>

// Private to UNIT_TEST builds. Normal libraries have no environment lookup,
// tensor downloads, file output, or trace-name construction on this path.
namespace fastllm { namespace nomas { namespace test {
class PipelineTrace {
    Runtime &rt_;
    std::string directory_,prefix_;
public:
    explicit PipelineTrace(Runtime &rt,const std::string &prefix={}):rt_(rt),prefix_(prefix) {
        if(const char *directory=std::getenv("FASTLLM_NOMAS_TRACE")) directory_=directory;
    }
    void operator()(const std::string &name,const std::vector<float> &values) const {
        if(directory_.empty()) return;
        std::ofstream file(directory_+"/"+prefix_+name+".f32",std::ios::binary);
        file.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
        if(!file) throw std::runtime_error("cannot write HY-Nomas pipeline trace");
    }
    void operator()(const std::string &name,const Tensor &values) const {
        if(!directory_.empty()) (*this)(name,rt_.download(values));
    }
};
}}}
#define NOMAS_TRACE_SCOPE(...) fastllm::nomas::test::PipelineTrace trace(__VA_ARGS__)
#define NOMAS_TRACE(...) trace(__VA_ARGS__)
