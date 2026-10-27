#include "fastllm-gguf-dequant.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

// The fixture weights are decoded by an independent llama.cpp CPU library.
// Reconstruct Q8_1 activations on the CPU to distinguish layout errors from
// the intentional activation quantization introduced by the fast path.
static float RoundInput(float x, fastllm::DataType dtype) {
    if (dtype == fastllm::DataType::FLOAT16) return __half2float(__float2half_rn(x));
    if (dtype == fastllm::DataType::BFLOAT16) return __bfloat162float(__float2bfloat16_rn(x));
    return x;
}
static float HalfBits(const char *p) {
    half h; std::memcpy(&h,p,sizeof(h)); return __half2float(h);
}
static double AffineSumCorrection(int type, const char *row, int col, double sumDelta) {
    // Q8_1 stores both a rounded scale and the rounded sum of the original
    // input block. These three affine formats use that sum for their offset,
    // rather than the sum reconstructed from int8 values and the scale.
    if(type == GGML_TYPE_Q4_0) return -8.0 * HalfBits(row+(col/32)*18) * sumDelta;
    if(type == GGML_TYPE_Q4_1) return HalfBits(row+(col/32)*20+2) * sumDelta;
    if(type == GGML_TYPE_IQ1_S) {
        const char *block=row+(col/256)*50;
        uint16_t qh; std::memcpy(&qh,block+34+2*((col%256)/32),2);
        double scale=HalfBits(block)*(2*((qh>>12)&7)+1);
        double delta=(qh&0x8000) ? -.125 : .125;
        return scale*(-1.0+delta)*sumDelta;
    }
    return 0;
}
static std::vector<float> ReadOutput(fastllm::Data data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
    const float *p = reinterpret_cast<const float *>(data.cpuData);
    return {p, p + data.Count(0)};
}
static void Run(const fastllm::Data &x, fastllm::Data &w, fastllm::Data &y,
                fastllm::DataType dtype, int cols, int rows, int tokens) {
    fastllm::Data bias;
    bool ok = dtype == fastllm::DataType::FLOAT16 ? FastllmCudaHalfMatMulGGUF(x,w,bias,y,tokens,cols,rows) :
              dtype == fastllm::DataType::BFLOAT16 ? FastllmCudaBFloat16MatMulGGUF(x,w,bias,y,tokens,cols,rows) :
              FastllmCudaMatMulFloatGGUF(x,w,bias,y,tokens,cols,rows);
    if (!ok) throw std::runtime_error("GGUF entry returned false");
    FastllmCudaSyncCurrentThreadStream();
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA failure");
}
int main(int argc, char **argv) try {
    if (argc < 2 || argc > 3) throw std::runtime_error("usage: test-fastpath fixtures.bin [tokens:1..9]");
    const int tokens = argc == 3 ? std::stoi(argv[2]) : 1;
    if (tokens < 1 || tokens > 9) throw std::runtime_error("tokens must be in [1,9]");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // On Blackwell, unflagged batch 8 can choose MMQ while flagged batch 8
    // explicitly chooses MMVQ. Compare both to the CPU reference there.
    const bool compareFlagBits = tokens < 8 || (tokens == 8 && prop.major < 12);
    FastllmCudaSetDevice(0);
    std::ifstream f(argv[1], std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t cases; f.read(reinterpret_cast<char *>(&cases), 4);
    int checks = 0, failures = 0;
    std::map<int,double> worst;
    for (uint32_t ci = 0; ci < cases; ++ci) {
        uint32_t h[4]; f.read(reinterpret_cast<char *>(h), sizeof(h));
        const int type=h[0], rows=h[1], cols=h[2];
        std::vector<char> packed(h[3]);
        std::vector<float> weights(size_t(rows)*cols);
        f.read(packed.data(), packed.size());
        f.read(reinterpret_cast<char *>(weights.data()), weights.size()*sizeof(float));
        fastllm::Data w(fastllm::DataType::DATA_GGUF_FORMAT, type, {rows,cols});
        w.disableGGUFRepack = true;
        w.name = "regression.fastpath." + std::to_string(type);
        w.Allocate(); std::memcpy(w.cpuData,packed.data(),packed.size());
        w.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        for (auto dtype : {fastllm::DataType::FLOAT16, fastllm::DataType::FLOAT32, fastllm::DataType::BFLOAT16}) {
            std::mt19937 rng(1921+cols);
            std::normal_distribution<float> normal(0,.2f);
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<float> input(tokens*cols), quantized(tokens*cols);
                std::vector<double> sumDelta(tokens*cols/32);
                for (int j=0;j<tokens*cols;++j) input[j]=RoundInput(pattern==0 ? normal(rng) : pattern==1 ? ((j%3)-1)*.125f : 0.f,dtype);
                for (int j=0;j<tokens*cols;j+=32) {
                    float maxabs=0;
                    for(int k=0;k<32;++k) maxabs=std::max(maxabs,std::abs(input[j+k]));
                    float d=maxabs/127.f, stored=__half2float(__float2half_rn(d));
                    for(int k=0;k<32;++k) quantized[j+k]=maxabs==0 ? 0.f : std::round(input[j+k]/d)*stored;
                    float originalSum=0; double reconstructedSum=0;
                    for(int k=0;k<32;++k) { originalSum+=input[j+k]; reconstructedSum+=quantized[j+k]; }
                    sumDelta[j/32]=__half2float(__float2half_rn(originalSum))-reconstructedSum;
                }
                fastllm::Data x(dtype,{tokens,cols},input);
                x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                for (int count : {1,std::min(7,rows),rows}) {
                    std::vector<float> previous;
                    for (bool flag : {false,true}) {
                        // Nine rows must retain the flagged dequant path.
                        if (tokens == 9 && !flag) continue;
                        const int outputCount = tokens*count;
                        fastllm::Data y(dtype,{1,outputCount+2},std::vector<float>(outputCount+2,-17.f));
                        y.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                        w.forceGGUFFp32Dequant=flag;
                        Run(x,w,y,dtype,cols,count,tokens);
                        auto actual=ReadOutput(y);
                        if(actual[outputCount]!=-17.f || actual[outputCount+1]!=-17.f) throw std::runtime_error("output guard overwritten");
                        if(flag && compareFlagBits && actual!=previous) throw std::runtime_error("MMVQ result depends on prefill safety flag");
                        previous=actual;
                        double e2=0,r2=0,a2=0;
                        for(int i=0;i<outputCount;++i) {
                            const int row = i % count, token = i / count;
                            double expected=0,absSum=0;
                            for(int j=0;j<cols;++j) {
                                const float activation = tokens == 9 ? input[token*cols+j] : quantized[token*cols+j];
                                const float decodedWeight = weights[size_t(row)*cols+j];
                                const float referenceWeight = tokens == 9 ? RoundInput(decodedWeight, dtype) : decodedWeight;
                                double term=double(referenceWeight)*activation;
                                expected+=term; absSum+=std::abs(term);
                            }
                            if (tokens <= 8) for(int j=0;j<cols;j+=32) expected+=AffineSumCorrection(type,packed.data()+size_t(row)*(packed.size()/rows),j,sumDelta[(token*cols+j)/32]);
                            if(!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
                            e2+=std::pow(actual[i]-expected,2);r2+=expected*expected;a2+=absSum*absSum;
                        }
                        // Most types match the quantized reference closely.
                        // Affine formats include their Q8_1.ds.y offset above.
                        const double rel=std::sqrt(e2/std::max(r2,1e-30));
                        const double bound=(dtype==fastllm::DataType::BFLOAT16 ? .012 : .006)*std::sqrt(r2)+.00008*std::sqrt(a2)+1e-6;
                        if(count==rows && pattern==0) worst[type]=std::max(worst[type],rel);
                        if(std::sqrt(e2)>bound) {
                            if(failures<30) std::cerr<<"FAIL case="<<ci<<" type="<<type<<" dtype="<<int(dtype)<<" cols="<<cols<<" rows="<<count<<" pattern="<<pattern<<" flag="<<flag<<" rel="<<rel<<" error="<<std::sqrt(e2)<<" bound="<<bound<<"\n";
                            ++failures;
                        }
                        ++checks;
                    }
                }
            }
        }
        std::cout<<"CASE "<<ci<<" type="<<type<<" cols="<<cols<<" failures="<<failures<<"\n";
    }
    for(auto [type,rel]:worst) std::cout<<"TYPE_WORST type="<<type<<" random_full_rows_relative_L2="<<rel<<"\n";
    std::cout<<"RESULT tokens="<<tokens<<" cases="<<cases<<" checks="<<checks<<" failures="<<failures<<"\n";
    return failures ? 1 : 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
