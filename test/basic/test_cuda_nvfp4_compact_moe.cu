#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static float Fp4(int x) {
    static const float values[] = {0,.5,1,1.5,2,3,4,6};
    return (x & 8 ? -1 : 1) * values[x & 7];
}
static float Fp8(unsigned x) {
    int exponent = (x >> 3) & 15, mantissa = x & 7;
    return exponent ? std::ldexp(1.f + mantissa / 8.f, exponent - 7)
                    : std::ldexp(float(mantissa), -9);
}
static float Round(float x, fastllm::DataType type) {
    return type == fastllm::DataType::BFLOAT16 ? __bfloat162float(__float2bfloat16_rn(x))
                                             : __half2float(__float2half_rn(x));
}
struct OracleWeight {
    int rows, cols;
    std::vector<unsigned char> packed, scales;
    std::vector<float> globals;
    double Get(int row, int column) const {
        unsigned byte = packed[size_t(row) * cols / 2 + column / 2];
        return double(Fp4((byte >> (4 * (column % 2))) & 15)) *
            Fp8(scales[size_t(row) * (cols / 16) + column / 16]) *
            globals[globals.size() == 2 && row >= rows / 2 ? 1 : 0];
    }
};
// Keep matrices alive across shapes: the existing native pointer-table cache
// is keyed by the weight object address.
static std::vector<std::unique_ptr<fastllm::Data>> retainedWeights;
static bool RunNative(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &part, fastllm::Data &output, std::vector<fastllm::Data*>& weights,
        const std::vector<int32_t>& indices, const std::vector<float>& scores,
        int32_t *deviceIndices, float *deviceScores, int batch, int topk, int hidden, int inter) {
    const bool bf16=input.dataType==fastllm::DataType::BFLOAT16;
    if(batch==1) return (bf16 ? FastllmCudaBFloat16MergeMOENVFP4Batch1Indexed :
        FastllmCudaHalfMergeMOENVFP4Batch1Indexed)(input,gate,output,weights.data(),weights.size(),
        deviceIndices,deviceScores,topk,hidden,inter);
    if(batch<=64) return (bf16 ? FastllmCudaBFloat16MergeMOENVFP4SmallBatchIndexed :
        FastllmCudaHalfMergeMOENVFP4SmallBatchIndexed)(input,gate,output,weights.data(),weights.size(),
        deviceIndices,deviceScores,batch,topk,hidden,inter);
    int experts=weights.size()/2-1, total=batch*topk, maximum=0;
    std::vector<int> counts(experts),starts(experts),cursor(experts),rows(total),positions(total);
    std::vector<float> routeScales(total);
    for(int e:indices)++counts[e];
    for(int e=0;e<experts;++e) {
        starts[e]=e ? starts[e-1]+counts[e-1] : 0;
        cursor[e]=starts[e];maximum=std::max(maximum,counts[e]);
    }
    for(int i=0;i<total;++i) {
        int pos=cursor[indices[i]]++;rows[pos]=i/topk;routeScales[pos]=scores[i];positions[i]=pos;
    }
    return (bf16 ? FastllmCudaBFloat16MergeMOENVFP4GroupedIndexed :
        FastllmCudaHalfMergeMOENVFP4GroupedIndexed)(input,gate,part,output,weights.data(),weights.size(),
        rows.data(),routeScales.data(),positions.data(),starts.data(),counts.data(),
        batch,topk,total,maximum,hidden,inter);
}
static void RunShape(int hidden, int inter, const std::vector<int>& batches, int topk = 3,
                     bool checkDirect = true) {
    constexpr int experts = 8;
    using fastllm::Data; using fastllm::DataType; using fastllm::DataDevice;
    std::mt19937 rng(941);
    auto& owned=retainedWeights;
    std::vector<Data*> weights(2 * (experts + 1), nullptr), referenceWeights(weights.size(),nullptr);
    std::vector<OracleWeight> original;
    for (int expert = 0; expert < experts; ++expert) {
        for (int part = 0; part < 2; ++part) {
            OracleWeight w{part == 0 ? inter * 2 : hidden, part == 0 ? hidden : inter};
            w.packed.resize(size_t(w.rows) * w.cols / 2);
            w.scales.resize(size_t(w.rows) * w.cols / 16);
            for (auto& v : w.packed) v = rng() % 256;
            static const unsigned char scaleValues[] = {0,1,8,15,0x38,0x48,0x62,0x7e};
            for (auto& v : w.scales) v = expert == 0 ? 0 : scaleValues[rng() % 8];
            // Real Naive layer-1 globals: unequal gate/up and maximum E4M3.
            // Also exercise ratios substantially beyond the old FP16 limit.
            w.globals = part == 0 ? std::vector<float>{9.264264e-5f * (expert + 1), 4.196167e-5f}
                                 : std::vector<float>{1.1e-4f / (expert + 1)};
            for(bool compact:{true,false}) {
                auto data=std::make_unique<Data>(compact ? DataType::NVFP4_BLOCK_16_E4M3_PACKED : DataType::NVFP4_BLOCK_16);
                data->blockK=1;data->blockM=16;data->Resize({w.rows,w.cols});data->Allocate(false);
                const size_t perRow=data->GetBytes()/w.rows;
                Require(!compact || perRow==((4+size_t(w.cols/16)*9+3)&~size_t(3)),"compact row storage expanded");
                for(int row=0;row<w.rows;++row) {
                    auto* dst=data->cpuData+size_t(row)*perRow;
                    float global=w.globals[w.globals.size()==2 && row>=w.rows/2 ? 1 : 0];
                    if(compact) {memcpy(dst,&global,4);dst+=4;}
                    for(int block=0;block<w.cols/16;++block) {
                        memcpy(dst,w.packed.data()+size_t(row)*w.cols/2+block*8,8);
                        unsigned char scale=w.scales[size_t(row)*(w.cols/16)+block];
                        if(compact) dst[8]=scale;
                        else {float scaleFloat=Fp8(scale)*global;memcpy(dst+8,&scaleFloat,4);}
                        dst+=compact ? 9 : 12;
                    }
                }
                data->directMemory=true;data->ToDevice(DataDevice::CUDA,std::vector<int>{0});
                (compact ? weights : referenceWeights)[2+expert*2+part]=data.get();owned.push_back(std::move(data));
            }
            original.push_back(std::move(w));
        }
    }
    for (auto type : {DataType::BFLOAT16,DataType::FLOAT16}) {
        for (int batch : batches) {
            std::vector<float> inputValues(size_t(batch)*hidden), routeScores(size_t(batch)*topk);
            std::vector<int32_t> indices(size_t(batch)*topk);
            for (auto& v : inputValues) v = Round((int(rng()%1001)-500)/1000.f,type);
            for (int row=0;row<batch;++row) for (int slot=0;slot<topk;++slot) {
                indices[row*topk+slot]=(row+slot*3)%experts;
                routeScores[row*topk+slot]=(slot+1)/6.f;
            }
            Data input(type,{batch,hidden});input.Allocate(false);
            for (size_t i=0;i<inputValues.size();++i) {
                if (type==DataType::BFLOAT16) reinterpret_cast<__nv_bfloat16*>(input.cpuData)[i]=__float2bfloat16_rn(inputValues[i]);
                else reinterpret_cast<half*>(input.cpuData)[i]=__float2half_rn(inputValues[i]);
            }
            input.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            // The native fallback uses ordinary expert linears, so
            // validate GEMV and dequantized GEMM independently of fused MoE.
            auto linear = type == DataType::BFLOAT16 ?
                FastllmCudaBFloat16MatMulNVFP4Block16 : FastllmCudaHalfMatMulFloatNVFP4Block16;
            Data linearCompact(type,{batch,inter*2}), linearExpanded(type,{batch,inter*2});
            linearCompact.Allocate(false);linearExpanded.Allocate(false);
            linearCompact.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            linearExpanded.ToDevice(DataDevice::CUDA,std::vector<int>{0});
            Require(linear(input,*weights[4],*fastllm::GetEmptyData(),linearCompact,batch,hidden,inter*2),
                    "compact expert linear failed");
            Require(linear(input,*referenceWeights[4],*fastllm::GetEmptyData(),linearExpanded,batch,hidden,inter*2),
                    "expanded expert linear failed");
            Check(cudaDeviceSynchronize());
            linearCompact.ToDevice(DataDevice::CPU);linearExpanded.ToDevice(DataDevice::CPU);
            size_t mismatches=0;
            for(size_t i=0;i<linearCompact.GetBytes()/2;++i)
                mismatches += reinterpret_cast<uint16_t*>(linearCompact.cpuData)[i] !=
                              reinterpret_cast<uint16_t*>(linearExpanded.cpuData)[i];
            printf("linear dtype=%d hidden=%d inter=%d batch=%d mismatches=%zu\n",
                   int(type),hidden,inter,batch,mismatches);fflush(stdout);
            Require(mismatches==0,"compact expert linear differs bitwise from FP32-scale layout");
            int32_t* deviceIndices=nullptr;float* deviceScores=nullptr;
            Check(cudaMalloc(&deviceIndices,indices.size()*sizeof(int32_t)));
            Check(cudaMalloc(&deviceScores,routeScores.size()*sizeof(float)));
            Check(cudaMemcpy(deviceIndices,indices.data(),indices.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
            Check(cudaMemcpy(deviceScores,routeScores.data(),routeScores.size()*sizeof(float),cudaMemcpyHostToDevice));
            Data gate,activation,output,referenceGate,referencePart,referenceOutput;
            bool ok=RunNative(input,gate,activation,output,weights,indices,routeScores,
                deviceIndices,deviceScores,batch,topk,hidden,inter);
            Require(ok,"compact native CUDA dispatch failed");Check(cudaDeviceSynchronize());
            Require(output.dataType==type,"output dtype changed");
            Require(RunNative(input,referenceGate,referencePart,referenceOutput,referenceWeights,
                indices,routeScores,deviceIndices,deviceScores,batch,topk,hidden,inter),"FP32-scale reference dispatch failed");
            Check(cudaDeviceSynchronize());output.ToDevice(DataDevice::CPU);referenceOutput.ToDevice(DataDevice::CPU);
            Require(output.GetBytes()==referenceOutput.GetBytes() &&
                memcmp(output.cpuData,referenceOutput.cpuData,output.GetBytes())==0,
                "compact output differs bitwise from native FP32-scale layout");
            if (batch == 1 && checkDirect) {
                auto runDirect = type == DataType::BFLOAT16 ? FastllmCudaBFloat16MergeMOENVFP4Batch1
                                                           : FastllmCudaHalfMergeMOENVFP4Batch1;
                for (auto *table : {&weights, &referenceWeights}) {
                    std::vector<Data *> gates(topk), downs(topk);
                    for (int slot = 0; slot < topk; ++slot) {
                        gates[slot] = (*table)[2 + indices[slot] * 2];
                        downs[slot] = (*table)[3 + indices[slot] * 2];
                    }
                    Data directGate, directOutput;
                    Require(runDirect(input, directGate, directOutput, gates.data(), downs.data(),
                                      routeScores.data(), false, topk, hidden, inter),
                            "direct native CUDA dispatch failed");
                    directOutput.ToDevice(DataDevice::CPU);
                    Require(directOutput.GetBytes() == output.GetBytes() &&
                            std::memcmp(directOutput.cpuData, output.cpuData, output.GetBytes()) == 0,
                            "direct and indexed NVFP4 dispatch differ");
                }
            }
            std::vector<double> expected(size_t(batch)*hidden), acts(inter);
            for (int row=0;row<batch;++row) for(int slot=0;slot<topk;++slot) {
                int expert=indices[row*topk+slot];const auto& g=original[expert*2];const auto& d=original[expert*2+1];
                for(int col=0;col<inter;++col) {
                    double gv=0,uv=0;
                    for(int k=0;k<hidden;++k) {
                        double x=Round(inputValues[size_t(row)*hidden+k],type);
                        gv+=x*g.Get(col,k);uv+=x*g.Get(col+inter,k);
                    }
                    float gr=Round(float(gv),type),ur=Round(float(uv),type);
                    acts[col]=Round(gr/(1.f+std::exp(-gr))*ur,type);
                }
                for(int col=0;col<hidden;++col) {
                    double sum=0;
                    for(int k=0;k<inter;++k)sum+=acts[k]*d.Get(col,k);
                    float value=Round(float(sum),type)*routeScores[row*topk+slot];
                    if(batch>64)value=Round(value,type);
                    auto& total=expected[size_t(row)*hidden+col];
                    total=float(total)+value;
                }
            }
            double error=0,energy=0,maxError=0,peak=0;
            for(size_t i=0;i<expected.size();++i) {
                double actual=type==DataType::BFLOAT16 ? __bfloat162float(reinterpret_cast<__nv_bfloat16*>(output.cpuData)[i])
                    : __half2float(reinterpret_cast<half*>(output.cpuData)[i]);
                Require(std::isfinite(actual),"nonfinite compact output");
                expected[i]=Round(float(expected[i]),type);
                double diff=actual-expected[i];error+=diff*diff;energy+=expected[i]*expected[i];
                maxError=std::max(maxError,std::abs(diff));peak=std::max(peak,std::abs(expected[i]));
            }
            double relative=std::sqrt(error/std::max(energy,1e-30));
            printf("{\"dtype\":%d,\"hidden\":%d,\"inter\":%d,\"batch\":%d,\"relative_rmse\":%.9g,\"max_abs\":%.9g,\"reference_peak\":%.9g,\"native_bitwise_equal\":true}\n",int(type),hidden,inter,batch,relative,maxError,peak);fflush(stdout);
            Require(relative<(type==DataType::BFLOAT16 ? .02 : .004),"compact result differs from independent FP64 oracle");
            Require(maxError<(type==DataType::BFLOAT16 ? .03 : .006)*peak+1e-6,"compact maximum error exceeds dtype tolerance");
            Check(cudaFree(deviceIndices));Check(cudaFree(deviceScores));
        }
    }
}
int main(int argc,char**argv) {
    try {
        Require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--quick") == 0),
                "usage: cuda_nvfp4_compact_moe_test [--quick]");
        int devices=0;auto status=cudaGetDeviceCount(&devices);
        if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || !devices)return 77;
        Check(status);cudaDeviceProp properties;Check(cudaGetDeviceProperties(&properties,0));
        if(properties.major<8)return 77; // BF16 support is required.
        FastllmCudaSetDevice(0);
        RunShape(128,128,argc>1 ? std::vector<int>{1,65} : std::vector<int>{1,9,32,65,200});
        if(argc==1) {
            RunShape(256,256,{1,17,96});
            RunShape(4096,2048,{1});
            // The direct kernel can fuse score multiplication with accumulation;
            // indexed kernels round the product first. Keep the existing direct
            // checks and verify top-8 against indexed FP32 scales and FP64 instead.
            RunShape(4096,2048,{1},8,false);
        }
        puts("compact NVFP4 native bitwise and FP64-oracle regression passed");return 0;
    } catch(const std::exception& e) {fprintf(stderr,"FAILED: %s\n",e.what());return 1;}
}
