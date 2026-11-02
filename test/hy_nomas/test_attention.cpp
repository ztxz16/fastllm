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
        Runtime rt(0,DType::BF16);
        // Unmasked queries are independent. Padding the query sequence forces
        // the original large-query kernel without changing these output rows.
        for(int rows:{65,88,116,221,256}) for(int keys:{511,512,3017}) {
            constexpr int cols=2048,paddedRows=257;
            std::vector<float> q(paddedRows*cols),k(keys*cols),v(k.size());
            for(size_t j=0;j<q.size();++j) q[j]=std::sin(j*.017)*.8+std::cos(j*.113)*.3;
            for(size_t j=0;j<k.size();++j) { k[j]=std::sin(j*.007)*.7; v[j]=std::cos(j*.031)*.9; }
            auto qt=rt.upload(q.data(),paddedRows,cols),kt=rt.upload(k.data(),keys,cols),vt=rt.upload(v.data(),keys,cols);
            auto actual=rt.download(rt.attention(qt.rowSlice(0,rows),kt,vt,16));
            auto expected=rt.download(rt.attention(qt,kt,vt,16).rowSlice(0,rows));
            if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)))
                throw std::runtime_error("short-query attention changed output after query padding");
        }
        std::cout<<"PASS short-query attention: query padding preserves every output bit\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
