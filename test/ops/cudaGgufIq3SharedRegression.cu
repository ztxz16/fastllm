#include "fastllm-gguf-dequant.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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
static void CheckCodebooks() {
    int checks=0;
    for(auto table : {std::make_pair(iq3s_grid,512),std::make_pair(iq3xxs_grid,256)}) {
        for(int i=0;i<table.second;++i)for(unsigned signs=0;signs<16;++signs) {
            uint32_t mask=0,expected=0;
            for(int j=0;j<4;++j) {
                int value=(table.first[i]>>(8*j))&255;
                if(value<=0 || value>127)throw std::runtime_error("IQ3 packed-negation precondition changed");
                if(signs&(1u<<j)){mask|=255u<<(8*j);value=-value;}
                expected|=uint32_t(uint8_t(value))<<(8*j);
            }
            uint32_t packed=(table.first[i]^mask)+(mask&0x01010101u);
            if(packed!=expected)throw std::runtime_error("signed IQ3 codebook mismatch");
            ++checks;
        }
    }
    for(unsigned code=0;code<128;++code) {
        unsigned signs=code|((__builtin_popcount(code)&1)<<7);
        uint64_t expected=0;
        for(int j=0;j<8;++j)if(signs&(1u<<j))expected|=uint64_t(255)<<(8*j);
        if(expected!=ksigns64[code])throw std::runtime_error("IQ3_XXS sign parity mismatch");
        ++checks;
    }
    std::cout<<"CODEBOOK_CHECKS "<<checks<<" passed\n";
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
    if (argc < 2 || argc > 3) throw std::runtime_error("usage: test-iq3-shared fixtures.bin [tokens:1..8]");
    const int tokens = argc == 3 ? std::stoi(argv[2]) : 1;
    if (tokens < 1 || tokens > 8) throw std::runtime_error("tokens must be in [1,8]");
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    const bool compareFlagBits = tokens < 8 || prop.major < 12;
    CheckCodebooks();
    FastllmCudaSetDevice(0);
    std::ifstream f(argv[1], std::ios::binary);
    f.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t cases; f.read(reinterpret_cast<char *>(&cases), 4);
    int checks = 0, failures = 0, tested = 0;
    std::map<int,double> worst;
    for (uint32_t ci = 0; ci < cases; ++ci) {
        uint32_t h[4]; f.read(reinterpret_cast<char *>(h), sizeof(h));
        const int type=h[0], sourceRows=h[1], cols=h[2];
        const int rows=tokens>1 && cols==256?4097:(tokens==1 && cols>=5120?1031:129);
        std::vector<char> packed(h[3]);
        std::vector<float> weights(size_t(sourceRows)*cols);
        f.read(packed.data(), packed.size());
        f.read(reinterpret_cast<char *>(weights.data()), weights.size()*sizeof(float));
        if(type!=GGML_TYPE_IQ3_S && type!=GGML_TYPE_IQ3_XXS && type!=GGML_TYPE_IQ4_XS &&
           type!=GGML_TYPE_IQ2_S && type!=GGML_TYPE_IQ2_XS && type!=GGML_TYPE_IQ2_XXS && type!=GGML_TYPE_Q4_K && type!=GGML_TYPE_Q2_K && type!=GGML_TYPE_IQ1_M)continue;
        ++tested;
        const auto originalPacked=packed;
        const auto originalWeights=weights;
        const size_t rowBytes=packed.size()/sourceRows;
        packed.resize(rowBytes*rows);weights.resize(size_t(rows)*cols);
        for(int i=0;i<rows;++i) {
            std::memcpy(packed.data()+i*rowBytes,originalPacked.data()+(i%sourceRows)*rowBytes,rowBytes);
            std::memcpy(weights.data()+size_t(i)*cols,originalWeights.data()+size_t(i%sourceRows)*cols,cols*sizeof(float));
        }
        fastllm::Data w(fastllm::DataType::DATA_GGUF_FORMAT, type, {rows,cols});
        w.disableGGUFRepack = true;
        w.name = "regression.fastpath." + std::to_string(type);
        w.Allocate(); std::memcpy(w.cpuData,packed.data(),packed.size());
        w.ToDevice(fastllm::DataDevice::CUDA, std::vector<int>{0}, true);
        for (auto dtype : {fastllm::DataType::FLOAT16, fastllm::DataType::FLOAT32, fastllm::DataType::BFLOAT16}) {
            std::mt19937 rng(1921+cols);
            std::normal_distribution<float> normal(0,.2f);
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<float> input(tokens*cols), quantized(tokens*cols), inputScales(tokens*cols/32);
                for (int j=0;j<tokens*cols;++j) input[j]=RoundInput(pattern==0 ? normal(rng) : pattern==1 ? ((j%3)-1)*.125f : 0.f,dtype);
                for (int j=0;j<tokens*cols;j+=32) {
                    float maxabs=0;
                    for(int k=0;k<32;++k) maxabs=std::max(maxabs,std::abs(input[j+k]));
                    float d=maxabs/127.f, stored=__half2float(__float2half_rn(d));
                    inputScales[j/32]=stored;
                    for(int k=0;k<32;++k) quantized[j+k]=maxabs==0 ? 0.f : std::round(input[j+k]/d)*stored;
                }
                fastllm::Data x(dtype,{tokens,cols},input);
                x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                for (int count : {1,std::min(7,rows),rows}) {
                    const int outputCount = tokens*count;
                    std::vector<float> previous;
                    for (bool flag : {false,true}) {
                        fastllm::Data y(dtype,{1,outputCount+2},std::vector<float>(outputCount+2,-17.f));
                        y.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                        w.forceGGUFFp32Dequant=flag;
                        Run(x,w,y,dtype,cols,count,tokens);
                        auto actual=ReadOutput(y);
                        if(actual[outputCount]!=-17.f || actual[outputCount+1]!=-17.f) throw std::runtime_error("output guard overwritten");
                        if(flag && compareFlagBits && actual!=previous) throw std::runtime_error("MMVQ result depends on prefill safety flag");
                        previous=actual;
                        double e2=0,r2=0,a2=0,trunc2=0;
                        for(int i=0;i<outputCount;++i) {
                            double expected=0,absSum=0;
                            for(int j=0;j<cols;++j) {
                                double term=double(weights[size_t(i%count)*cols+j])*quantized[(i/count)*cols+j];
                                expected+=term; absSum+=std::abs(term);
                            }
                            if(!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
                            // IQ2 dot primitives intentionally truncate integer divisions.
                            // Their deviation from ideal CPU-dequantized arithmetic is
                            // strictly below one d_weight*d_input per 32-value group.
                            if(type==GGML_TYPE_IQ2_S || type==GGML_TYPE_IQ2_XS || type==GGML_TYPE_IQ2_XXS) {
                                double trunc=0;
                                const size_t blockBytes=rowBytes/(cols/256);
                                for(int k=0;k<cols;k+=32) {
                                    half wd;std::memcpy(&wd,packed.data()+(i%count)*rowBytes+(k/256)*blockBytes,2);
                                    trunc+=std::abs(double(__half2float(wd))*inputScales[(i/count)*cols/32+k/32]);
                                }
                                trunc2+=trunc*trunc;
                            }
                            e2+=std::pow(actual[i]-expected,2);r2+=expected*expected;a2+=absSum*absSum;
                        }
                        // Independent CPU-decoded weights and Q8_1 inputs;
                        // FP32 uses a tighter bound than rounded FP16/BF16.
                        const double rel=std::sqrt(e2/std::max(r2,1e-30));
                        // A lane accumulates one 32-value group per 1024 input
                        // values, followed by a warp reduction and scale products.
                        // gamma_n accounts for FP32 rounding of long, cancelling
                        // sums; a fixed epsilon*sum(abs(terms)) is not sufficient.
                        const double unitRoundoff=std::numeric_limits<float>::epsilon()/2.0;
                        const double steps=(cols+1023)/1024+8;
                        const double gamma=steps*unitRoundoff/(1.0-steps*unitRoundoff);
                        const double bound=(dtype==fastllm::DataType::BFLOAT16 ? .012 : dtype==fastllm::DataType::FLOAT16 ? .002 : .00002)*std::sqrt(r2)+gamma*std::sqrt(a2)+std::sqrt(trunc2)+1e-6;
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
        if (tokens == 1 && type != GGML_TYPE_IQ4_XS) {
            // Different gate/up rows catch merged-weight offsets and tail handling.
            fastllm::Data up(fastllm::DataType::DATA_GGUF_FORMAT,type,{rows,cols});
            fastllm::Data merged(fastllm::DataType::DATA_GGUF_FORMAT,type,{2*rows,cols});
            up.disableGGUFRepack=merged.disableGGUFRepack=true;
            up.Allocate();merged.Allocate();
            std::memcpy(merged.cpuData,packed.data(),packed.size());
            for(int i=0;i<rows;++i) {
                const void *src=originalPacked.data()+((i+3)%sourceRows)*rowBytes;
                std::memcpy(up.cpuData+i*rowBytes,src,rowBytes);
                std::memcpy(merged.cpuData+(rows+i)*rowBytes,src,rowBytes);
            }
            up.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
            merged.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
            for(int pattern=0;pattern<3;++pattern) {
                std::mt19937 rng(777+cols);std::normal_distribution<float>normal(0,.2f);
                std::vector<float>input(cols),quantized(cols);
                for(int j=0;j<cols;++j)input[j]=RoundInput(pattern==0?normal(rng):pattern==1?((j%3)-1)*.125f:0.f,fastllm::DataType::FLOAT16);
                for(int j=0;j<cols;j+=32) {
                    float maxabs=0;for(int k=0;k<32;++k)maxabs=std::max(maxabs,std::abs(input[j+k]));
                    float d=maxabs/127.f,stored=__half2float(__float2half_rn(d));
                    for(int k=0;k<32;++k)quantized[j+k]=maxabs==0?0.f:std::round(input[j+k]/d)*stored;
                }
                fastllm::Data x(fastllm::DataType::FLOAT16,{1,cols},input);
                x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                std::vector<float>expected(rows),previous;
                for(int i=0;i<rows;++i) {
                    double a=0,b=0;
                    for(int j=0;j<cols;++j) {
                        a+=double(originalWeights[size_t(i%sourceRows)*cols+j])*quantized[j];
                        b+=double(originalWeights[size_t((i+3)%sourceRows)*cols+j])*quantized[j];
                    }
                    const half h=__float2half_rn(a),u=__float2half_rn(b);
                    const half ex=__float2half_rn(std::exp(-__half2float(h)));
                    expected[i]=__half2float(__hmul(__hdiv(h,__hadd(__float2half(1.f),ex)),u));
                }
                for(int mode=0;mode<2;++mode) {
                    fastllm::Data y(fastllm::DataType::FLOAT16,{1,rows+2},std::vector<float>(rows+2,-17.f));
                    y.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                    bool ok=mode==0?FastllmCudaHalfGgufGateUpSiluMul(x,w,up,y,1,cols,rows):FastllmCudaHalfGgufMergedGateUpSiluMul(x,merged,y,1,cols,rows);
                    if(!ok)throw std::runtime_error("fused entry returned false");
                    FastllmCudaSyncCurrentThreadStream();
                    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("fused CUDA failure");
                    auto out=ReadOutput(y);
                    if(out[rows]!=-17.f || out[rows+1]!=-17.f)throw std::runtime_error("fused guard overwritten");
                    if(mode==1 && out!=previous)throw std::runtime_error("merged and separate gate/up differ");
                    previous=out;
                    double e2=0,r2=0;
                    for(int i=0;i<rows;++i) {
                        if(!std::isfinite(out[i]))throw std::runtime_error("fused nonfinite");
                        e2+=std::pow(out[i]-expected[i],2);r2+=double(expected[i])*expected[i];
                    }
                    double rel=std::sqrt(e2/std::max(r2,1e-30));
                    if(std::sqrt(e2)>.004*std::sqrt(r2)+1e-6) {
                        std::cerr<<"FUSED FAIL type="<<type<<" cols="<<cols<<" mode="<<mode<<" pattern="<<pattern<<" rel="<<rel<<"\n";++failures;
                    }
                    ++checks;
                }
            }
        }
        std::cout<<"CASE "<<ci<<" type="<<type<<" cols="<<cols<<" failures="<<failures<<"\n";
    }
    for(auto [type,rel]:worst) std::cout<<"TYPE_WORST type="<<type<<" random_full_rows_relative_L2="<<rel<<"\n";
    std::cout<<"RESULT tokens="<<tokens<<" fixtures="<<cases<<" cases="<<tested<<" checks="<<checks<<" failures="<<failures<<"\n";
    return failures ? 1 : 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
