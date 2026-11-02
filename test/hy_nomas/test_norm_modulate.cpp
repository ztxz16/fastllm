#include "models/hy_nomas/runtime.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace fastllm::nomas;
int main() {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    try {
        int cases=0;
        for(auto dtype:{DType::BF16,DType::FP16,DType::FP32}) {
            Runtime rt(0,dtype);
            for(int rows:{1,11,105,257}) for(int cols:{67,128,2048}) for(int pattern=0;pattern<3;++pattern) {
                std::vector<float> x(size_t(rows)*cols),m(6*cols);
                for(size_t i=0;i<x.size();++i)
                    x[i]=pattern==0 ? float(std::sin(i*.017)*2+std::cos(i*.137)*.7)
                         : pattern==1 ? .125f : (int(i%7)-3)*.0001f;
                for(size_t i=0;i<m.size();++i) m[i]=std::sin(i*.043)*1.2;
                // Include scale=-1, rounding ties, and near-zero variance.
                for(int c=0;c<cols;c+=7) { m[cols+c]=-1.f; m[4*cols+c]=.00390625f; }
                auto input=rt.upload(x.data(),rows,cols),mod=rt.upload(m.data(),1,6*cols);
                for(int shift:{0,3}) for(float eps:{1e-6f,1e-5f}) {
                    auto expected=rt.download(rt.modulate(rt.norm(input,false,nullptr,nullptr,eps),mod,shift,shift+1));
                    auto actual=rt.download(rt.normModulate(input,mod,shift,shift+1,eps));
                    if(expected.size()!=actual.size() || std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(float)))
                        throw std::runtime_error("fused norm modulation changed output: dtype="+std::to_string(int(dtype))+" rows="+std::to_string(rows)+" cols="+std::to_string(cols));
                    ++cases;
                }
            }
            std::vector<float> zeros(768,0.f);
            auto input=rt.upload(zeros.data(),1,128),mod=rt.upload(zeros.data(),1,768);
            for(int shift:{-1,6}) {
                bool rejected=false;
                try { rt.normModulate(input,mod,shift,1); } catch(const std::exception &) { rejected=true; }
                if(!rejected) throw std::runtime_error("invalid modulation index accepted");
            }
        }
        std::cout<<"PASS norm modulation: "<<cases<<" cases preserve every output bit\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
