#include "fastllm.h"
#include "executor.h"
#include "gguf.h"
#include "utils.h"
#include "devices/numas/numasdevice.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        SetThreads(4);
        auto *executor=static_cast<Executor*>(GetExecutor());
        executor->SetFirstDevice("numa");
        constexpr int hidden=256, inter=256;
        Data gate(FLOAT32,{2*inter,hidden}), down(DATA_GGUF_FORMAT,GGML_TYPE_IQ4_XS,{hidden,inter});
        gate.Allocate();down.Allocate();
        std::memset(gate.cpuData,0,gate.GetBytes());
        for(int row=0;row<2*inter;++row) reinterpret_cast<float*>(gate.cpuData)[row*hidden+row%hidden]=row<inter ? .5f : 1.f;
        auto *blocks=reinterpret_cast<block_iq4_xs*>(down.cpuData);
        for(int row=0;row<hidden;++row) {
            blocks[row].d=float_to_half(std::ldexp(1.f,-12+row%3));
            blocks[row].scales_h=0xaaaa;
            std::memset(blocks[row].scales_l,0x88,sizeof(blocks[row].scales_l));
            std::memset(blocks[row].qs,0x99,sizeof(blocks[row].qs));
        }
        std::vector<float> decoded(hidden*inter);
        dequantize_row_iq4_xs(blocks,decoded.data(),decoded.size());
        std::vector<Data*> weights{nullptr,nullptr,&gate,&down},biases(4,nullptr);
        for(bool clamped : {false,true}) for(int rows : {1,8,33,65}) {
            Data input(BFLOAT16,{rows,hidden}),ids(INT32,{rows,1}),scores(FLOAT32,{rows,1});
            input.Allocate();ids.Allocate();scores.Allocate();
            for(int r=0;r<rows;++r) {
                const float x=std::ldexp(127.f/128,r%3-1);
                for(int c=0;c<hidden;++c) reinterpret_cast<uint16_t*>(input.cpuData)[r*hidden+c]=Float32ToBFloat16RNEBits(x);
                reinterpret_cast<int*>(ids.cpuData)[r]=0;reinterpret_cast<float*>(scores.cpuData)[r]=1.f;
            }
            Data output(BFLOAT16),w1,w2,w3,ci,co;
            for(int repeat=0;repeat<2;++repeat) {
                MergeMOE(input,ids,scores,weights,biases,w1,w2,w3,ci,co,0.f,output,0,MoeGateSwiglu,false,clamped?10.f:0.f,clamped,nullptr,128);
                for(int r=0;r<rows;++r) {
                    const float x=std::ldexp(127.f/128,r%3-1);
                    const float activation=(.5f*x/(1+std::exp(-.5f*x)))*x;
                    for(int c=0;c<hidden;++c) {
                        double expected=0;
                        for(int j=0;j<inter;++j) expected+=decoded[c*inter+j]*activation;
                        const float actual=BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(output.cpuData)[r*hidden+c]);
                        if(!std::isfinite(actual) || std::fabs(actual-expected)>1e-4+.025*std::fabs(expected)) {
                            std::fprintf(stderr,"rows=%d clamp=%d row=%d col=%d got=%g expected=%g\n",rows,clamped,r,c,actual,expected);
                            throw std::runtime_error("NUMA IQ4_XS fallback differs from scalar reference");
                        }
                    }
                }
            }
        }
        ClearNumasMoeRuntimeCache();
        std::puts("PASS: NUMA IQ4_XS quantized-weight fallback, decode/prefill, repeated calls and clamp");return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
