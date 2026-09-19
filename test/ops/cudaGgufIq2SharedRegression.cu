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
static void CheckCodebooks() {
    int checks=0;
    for(auto table : {std::make_pair(iq2s_grid,1024),std::make_pair(iq2xs_grid,512),std::make_pair(iq2xxs_grid,256)}) {
        for(int i=0;i<table.second;++i)for(int part=0;part<2;++part)for(unsigned signs=0;signs<16;++signs) {
            const uint32_t word=uint32_t(table.first[i]>>(32*part));
            uint32_t mask=0,expected=0;
            for(int j=0;j<4;++j) {
                int value=(word>>(8*j))&255;
                if(value<=0 || value>127)throw std::runtime_error("IQ2 packed-negation precondition changed");
                if(signs&(1u<<j)){mask|=255u<<(8*j);value=-value;}
                expected|=uint32_t(uint8_t(value))<<(8*j);
            }
            uint32_t packed=(word^mask)+(mask&0x01010101u);
            if(packed!=expected)throw std::runtime_error("signed IQ2 codebook mismatch");
            ++checks;
        }
    }
    for(unsigned code=0;code<128;++code) {
        unsigned signs=code|((__builtin_popcount(code)&1)<<7);
        uint64_t expected=0;
        for(int j=0;j<8;++j)if(signs&(1u<<j))expected|=uint64_t(255)<<(8*j);
        if(expected!=ksigns64[code] || signs!=ksigns_iq2xs[code])throw std::runtime_error("IQ2_XXS sign parity mismatch");
        ++checks;
    }
    std::cout<<"CODEBOOK_CHECKS "<<checks<<" passed\n";
}
// Decode with independent llama.cpp fixture weights, then reconstruct their
// signed integer codebook entries. IQ2 MMVQ truncates two integer divisions
// per 32 inputs; a plain float dot of decoded weights omits that rounding.
// This reference uses scalar CPU arithmetic, without the CUDA lookup helpers.
static double IQ2Reference(int type, const char *packedRow, const float *decoded,
                          const std::vector<float> &input, int cols) {
    const size_t blockBytes = type == GGML_TYPE_IQ2_XS ? sizeof(block_iq2_xs) :
                              type == GGML_TYPE_IQ2_S ? sizeof(block_iq2_s) : sizeof(block_iq2_xxs);
    double result = 0;
    for (int j=0;j<cols;j+=32) {
        const char *block = packedRow + (j/256)*blockBytes;
        half storedWeight; std::memcpy(&storedWeight,block,2);
        const float dw = __half2float(storedWeight);
        const int sub = (j%256)/32;
        int ls0,ls1;
        if(type==GGML_TYPE_IQ2_XXS) {
            const auto *w = reinterpret_cast<const block_iq2_xxs *>(block);
            uint32_t aux;std::memcpy(&aux,w->qs+4*sub+2,4);
            ls0=ls1=aux>>28;
        } else {
            const int scales = type==GGML_TYPE_IQ2_XS ?
                reinterpret_cast<const block_iq2_xs *>(block)->scales[sub] :
                reinterpret_cast<const block_iq2_s *>(block)->scales[sub];
            ls0=scales&15;ls1=scales>>4;
        }
        float amax=0;
        for(int k=0;k<32;++k)amax=std::max(amax,std::abs(input[j+k]));
        if(amax==0 || dw==0)continue;
        const float dx=amax/127.f,storedDx=__half2float(__float2half_rn(dx));
        int sums[2]={0,0};
        for(int k=0;k<32;++k) {
            const double scale=double(dw)*((k<16?ls0:ls1)+.5)*.25;
            const int code=std::lround(decoded[j+k]/scale);
            if(std::abs(code)!=8 && std::abs(code)!=25 && std::abs(code)!=43)
                throw std::runtime_error("independent IQ2 decoder/codebook disagreement");
            sums[k/16]+=code*int(std::round(input[j+k]/dx));
        }
        const int integerDot=(sums[0]*ls0+sums[1]*ls1+(sums[0]+sums[1])/2)/4;
        result+=double(dw)*storedDx*integerDot;
    }
    return result;
}
static std::vector<float> ReadOutput(fastllm::Data data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
    const float *p = reinterpret_cast<const float *>(data.cpuData);
    return {p, p + data.Count(0)};
}
static void Run(const fastllm::Data &x, fastllm::Data &w, fastllm::Data &y,
                fastllm::DataType dtype, int cols, int rows) {
    fastllm::Data bias;
    bool ok = dtype == fastllm::DataType::FLOAT16 ? FastllmCudaHalfMatMulGGUF(x,w,bias,y,1,cols,rows) :
              dtype == fastllm::DataType::BFLOAT16 ? FastllmCudaBFloat16MatMulGGUF(x,w,bias,y,1,cols,rows) :
              FastllmCudaMatMulFloatGGUF(x,w,bias,y,1,cols,rows);
    if (!ok) throw std::runtime_error("GGUF entry returned false");
    FastllmCudaSyncCurrentThreadStream();
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA failure");
}
// Batch two retains the established four-warp kernel. Duplicate the input
// to compare the new batch-one output with that path using identical weights.
static std::vector<float> RunLegacy(const std::vector<float> &input,
        fastllm::Data &w, fastllm::Data *up, fastllm::DataType dtype,
        int type, int cols, int rows) {
    std::vector<float> twice=input;twice.insert(twice.end(),input.begin(),input.end());
    fastllm::Data x(dtype,{2,cols},twice),y(dtype,{2,rows},std::vector<float>(2*rows));
    x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
    y.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
    void *stream=reinterpret_cast<void *>(cudaStreamPerThread);
    bool ok;
    if(up) {
        ok=FastllmCudaHalfGgufGateUpSiluMulMMVQ(x.cudaData,w.cudaData,up->cudaData,
                                             y.cudaData,type,2,cols,rows,stream);
    } else if(dtype==fastllm::DataType::FLOAT16) {
        ok=FastllmCudaHalfMatMulGGUFMMVQ(x.cudaData,w.cudaData,y.cudaData,type,2,cols,rows,stream);
    } else if(dtype==fastllm::DataType::BFLOAT16) {
        ok=FastllmCudaBFloat16MatMulGGUFMMVQ(x.cudaData,w.cudaData,y.cudaData,type,2,cols,rows,stream);
    } else {
        ok=FastllmCudaFloatMatMulGGUFMMVQ(x.cudaData,w.cudaData,y.cudaData,type,2,cols,rows,stream);
    }
    if(!ok)throw std::runtime_error("legacy batch-two MMVQ rejected input");
    FastllmCudaSyncCurrentThreadStream();
    auto result=ReadOutput(y);
    if(std::memcmp(result.data(),result.data()+rows,rows*sizeof(float))!=0)
        throw std::runtime_error("duplicate legacy inputs disagree");
    result.resize(rows);return result;
}
int main(int argc, char **argv) try {
    if (argc != 2) throw std::runtime_error("usage: test-iq2-shared fixtures.bin");
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
        const int rows=cols>=5120?1031:129;
        std::vector<char> packed(h[3]);
        std::vector<float> weights(size_t(sourceRows)*cols);
        f.read(packed.data(), packed.size());
        f.read(reinterpret_cast<char *>(weights.data()), weights.size()*sizeof(float));
        if(type!=GGML_TYPE_IQ2_S && type!=GGML_TYPE_IQ2_XXS && type!=GGML_TYPE_IQ2_XS)continue;
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
                std::vector<float> input(cols), quantized(cols);
                for (int j=0;j<cols;++j) input[j]=RoundInput(pattern==0 ? normal(rng) : pattern==1 ? ((j%3)-1)*.125f : 0.f,dtype);
                for (int j=0;j<cols;j+=32) {
                    float maxabs=0;
                    for(int k=0;k<32;++k) maxabs=std::max(maxabs,std::abs(input[j+k]));
                    float d=maxabs/127.f, stored=__half2float(__float2half_rn(d));
                    for(int k=0;k<32;++k) quantized[j+k]=maxabs==0 ? 0.f : std::round(input[j+k]/d)*stored;
                }
                fastllm::Data x(dtype,{1,cols},input);
                x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                const auto legacy=RunLegacy(input,w,nullptr,dtype,type,cols,rows);
                for (int count : {1,7,127,128,rows}) {
                    std::vector<float> previous;
                    for (bool flag : {false,true}) {
                        fastllm::Data y(dtype,{1,count+2},std::vector<float>(count+2,-17.f));
                        y.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
                        w.forceGGUFFp32Dequant=flag;
                        Run(x,w,y,dtype,cols,count);
                        auto actual=ReadOutput(y);
                        if(actual[count]!=-17.f || actual[count+1]!=-17.f) throw std::runtime_error("output guard overwritten");
                        if(flag && actual!=previous) throw std::runtime_error("single-token result depends on prefill safety flag");
                        previous=actual;
                        if(count==rows && !flag) {
                            if(std::memcmp(actual.data(),legacy.data(),rows*sizeof(float))!=0)
                                throw std::runtime_error("single-token output changed from four-warp MMVQ");
                            ++checks;
                        }
                        double e2=0,r2=0,a2=0;
                        for(int i=0;i<count;++i) {
                            double expected=0,absSum=0;
                            for(int j=0;j<cols;++j) {
                                double term=double(weights[size_t(i)*cols+j])*quantized[j];
                                absSum+=std::abs(term);
                            }
                            expected=IQ2Reference(type,packed.data()+size_t(i)*rowBytes,
                                                   weights.data()+size_t(i)*cols,input,cols);
                            if(!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
                            e2+=std::pow(actual[i]-expected,2);r2+=expected*expected;a2+=absSum*absSum;
                        }
                        // Independent CPU-decoded weights, Q8_1 inputs, and IQ2 truncation;
                        // FP32 uses a tighter bound than rounded FP16/BF16.
                        const double rel=std::sqrt(e2/std::max(r2,1e-30));
                        const double bound=(dtype==fastllm::DataType::BFLOAT16 ? .012 : dtype==fastllm::DataType::FLOAT16 ? .002 : .00002)*std::sqrt(r2)+.0000001*std::sqrt(a2)+1e-6;
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
            std::vector<float>input(cols);
            for(int j=0;j<cols;++j)input[j]=RoundInput(pattern==0?normal(rng):pattern==1?((j%3)-1)*.125f:0.f,fastllm::DataType::FLOAT16);
            fastllm::Data x(fastllm::DataType::FLOAT16,{1,cols},input);
            x.ToDevice(fastllm::DataDevice::CUDA,std::vector<int>{0},true);
            std::vector<float>expected(rows),previous;
            for(int i=0;i<rows;++i) {
                const double a=IQ2Reference(type,originalPacked.data()+size_t(i%sourceRows)*rowBytes,
                               originalWeights.data()+size_t(i%sourceRows)*cols,input,cols);
                const double b=IQ2Reference(type,originalPacked.data()+size_t((i+3)%sourceRows)*rowBytes,
                               originalWeights.data()+size_t((i+3)%sourceRows)*cols,input,cols);
                const half h=__float2half_rn(a),u=__float2half_rn(b);
                const half ex=__float2half_rn(std::exp(-__half2float(h)));
                expected[i]=__half2float(__hmul(__hdiv(h,__hadd(__float2half(1.f),ex)),u));
            }
            const auto legacy=RunLegacy(input,w,&up,fastllm::DataType::FLOAT16,type,cols,rows);
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
                if(mode==0) {
                    if(std::memcmp(out.data(),legacy.data(),rows*sizeof(float))!=0)
                        throw std::runtime_error("fused output changed from four-warp MMVQ");
                    ++checks;
                }
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
        std::cout<<"CASE "<<ci<<" type="<<type<<" cols="<<cols<<" failures="<<failures<<"\n";
    }
    for(auto [type,rel]:worst) std::cout<<"TYPE_WORST type="<<type<<" random_full_rows_relative_L2="<<rel<<"\n";
    std::cout<<"RESULT fixtures="<<cases<<" cases="<<tested<<" checks="<<checks<<" failures="<<failures<<"\n";
    return failures ? 1 : 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
