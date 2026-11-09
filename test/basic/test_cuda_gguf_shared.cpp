// Same-input GGUF projection batches: ordinary Linear is the bitwise oracle.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define Graph ExistingGraph
#include "cuda_gguf_fusion_test.cuh"
#undef Graph
static std::map<void *,std::vector<void *>> reservations;
static bool Begin(){Cuda(cudaDeviceSynchronize());return FastllmCudaGraphMemoryPoolBegin() && FastllmCudaGraphBeginCapture();}
static bool End(void **g){bool ok=FastllmCudaGraphEndCapture(g);std::vector<void *> p;ok=FastllmCudaGraphMemoryPoolEnd(p)&&ok;if(ok)reservations[*g]=std::move(p);else FastllmCudaGraphMemoryPoolAbort();return ok;}
static void Destroy(void *g){FastllmCudaGraphDestroy(g);FastllmCudaGraphMemoryPoolRelease(reservations[g]);reservations.erase(g);}
static void Graph(const std::function<void()> &run,const std::function<void()> &compare,const std::function<void(int)> &prepare){
 prepare(0);run();Cuda(cudaDeviceSynchronize());compare();
 void *g=nullptr,*exec=nullptr;Check(Begin(),"begin");run();
 Check(End(&g),"end");Check(FastllmCudaGraphInstantiate(g,&exec),"instantiate");
 for(int i=1;i<=2;++i){prepare(i);Check(FastllmCudaGraphLaunch(exec),"launch");Cuda(cudaDeviceSynchronize());compare();}
 FastllmCudaGraphExecDestroy(exec);Destroy(g);++graphs;++cases;
}
#include "devices/cuda/fastllm-cuda-linear-shared.h"
#include <memory>
#include <array>

static void Group(ggml_type type, DataType dtype, int tokens, int columns,
                  const std::vector<int> &widths, bool mixed = false) {
    context = "shared type=" + std::to_string(type) + " dtype=" + std::to_string(dtype) +
        " tokens=" + std::to_string(tokens) + " jobs=" + std::to_string(widths.size());
    Data input(dtype,{1,tokens,columns}), bias;
    Allocate(input);
    std::vector<std::unique_ptr<Data>> weights, outputs, refs;
    std::vector<Data *> w,y;
    for(int n:widths) {
        const ggml_type format = mixed ? std::array<ggml_type,3>{GGML_TYPE_Q8_0,GGML_TYPE_Q5_K,GGML_TYPE_Q5_1}[weights.size()%3] : type;
        weights.emplace_back(new Data(DATA_GGUF_FORMAT,int(format),{n,columns}));
        Allocate(*weights.back()); weights.back()->strides={1};
        weights.back()->forceGGUFFp32Dequant=tokens<=8;
        const size_t block=ggml_type_size(format);
        std::vector<uint8_t> bytes(size_t(n)*ggml_row_size(format,columns));
        uint32_t rng=713+int(format)+n;
        for(auto &v:bytes){rng=rng*1664525U+1013904223U;v=rng>>24;}
        for(size_t j=0;j<bytes.size();j+=block) {
            void *p=bytes.data()+j;half scale=__float2half_rn(float(1+j/block%4)/16384);
            switch(format) {
                case GGML_TYPE_Q8_0:static_cast<block_q8_0 *>(p)->d=scale;break;
                case GGML_TYPE_Q5_0:static_cast<block_q5_0 *>(p)->d=scale;break;
                case GGML_TYPE_Q5_1:static_cast<block_q5_1 *>(p)->dm=__halves2half2(scale,scale);break;
                case GGML_TYPE_Q5_K:static_cast<block_q5_K *>(p)->dm=__halves2half2(scale,scale);break;
                case GGML_TYPE_Q6_K:static_cast<block_q6_K *>(p)->d=scale;break;
                case GGML_TYPE_IQ4_NL:static_cast<block_iq4_nl *>(p)->d=scale;break;
                default:throw std::runtime_error("unsupported fixture format");
            }
        }
        Upload(*weights.back(),bytes);
        outputs.emplace_back(new Data(dtype,{1,tokens,n}));refs.emplace_back(new Data(dtype,{1,tokens,n}));
        Allocate(*outputs.back());Allocate(*refs.back());w.push_back(weights.back().get());y.push_back(outputs.back().get());
    }
    auto reference=[&] {
        for(size_t i=0;i<w.size();++i) {
            bool ok=dtype==BFLOAT16 ? FastllmCudaBFloat16MatMulGGUF(input,*w[i],bias,*refs[i],tokens,columns,widths[i])
                : FastllmCudaHalfMatMulGGUF(input,*w[i],bias,*refs[i],tokens,columns,widths[i]);
            Check(ok,"ordinary projection rejected");
        }
    };
    auto run=[&] {Check(FastllmCudaLinearShared(input,w.data(),y.data(),w.size()),"shared rejected");};
    auto prepare=[&](int seed) {
        std::vector<uint16_t> values(size_t(tokens)*columns);
        for(size_t j=0;j<values.size();++j) {
            float x=j%columns<32?0:.8f*std::sin(float(j+seed*73)*.113f);
            if(dtype==BFLOAT16) {auto v=__float2bfloat16_rn(x);std::memcpy(&values[j],&v,2);}
            else {auto v=__float2half_rn(x);std::memcpy(&values[j],&v,2);}
        }
        Upload(input,values);reference();
    };
    Graph(run,[&]{for(size_t i=0;i<w.size();++i) {
        Same(*y[i],Download(*refs[i],refs[i]->Count(0)),"shared output");
    }},prepare);
    // One quantization and ceil(jobs/8) projection nodes; no per-job copies.
    run(); Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph;cudaGraphExec_t exec;
    Check(Begin(),"managed begin");run();
    Check(End(reinterpret_cast<void **>(&graph)),"managed end");
    size_t size=0;Cuda(cudaGraphGetNodes(graph,nullptr,&size));std::vector<cudaGraphNode_t> nodes(size);
    Cuda(cudaGraphGetNodes(graph,nodes.data(),&size));int kernels=0;
    for(auto n:nodes){cudaGraphNodeType t;Cuda(cudaGraphNodeGetType(n,&t));kernels+=t==cudaGraphNodeTypeKernel;}
    const int expected = (tokens<=8 ? 1+((int(w.size())+7)/8) : (mixed?2:1)+2*int(w.size()));
    if(kernels!=expected)fprintf(stderr,"launches rows=%d actual=%d expected=%d\n",tokens,kernels,expected);
    Check(kernels==expected,"unexpected shared launch count");
    Check(FastllmCudaGraphInstantiate(graph,reinterpret_cast<void **>(&exec)),"managed instantiate");
    FastllmCudaGraphExecDestroy(exec);Destroy(graph);
    auto before=Download(*y[0],y[0]->Count(0));auto *last=y.back();y.back()=y[0];
    Check(!FastllmCudaLinearShared(input,w.data(),y.data(),w.size()),"overlap accepted");y.back()=last;
    auto *lastWeight=w.back();w.back()=&input;Check(!FastllmCudaLinearShared(input,w.data(),y.data(),w.size()),"invalid late job accepted");w.back()=lastWeight;
    Same(*y[0],before,"rejected batch wrote outputs");

}
int main() {
    try {
        Cuda(cudaSetDevice(0));
        for(int threshold:{1,9}) {
        fastllm::FastllmCudaSetLinearExactBatchThreshold(threshold);
        for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_NL})
            for(auto dtype:{FLOAT16,BFLOAT16})for(int rows : {1,2,3,4,5,6,7,8,9,12,15,16,17,24,31,32})
                Group(type,dtype,rows,256,{1,13,65});
        }
        for(auto dtype:{FLOAT16,BFLOAT16}) {
            Group(GGML_TYPE_Q8_0,dtype,6,768,{1,2,7,16,17,32,33,65,129,257});
            for(int rows:{1,5,6,9,12,17,24,32})Group(GGML_TYPE_Q8_0,dtype,rows,5120,{1280,512});
        }
        for(int rows:{9,17,33,129})for(auto dtype:{FLOAT16,BFLOAT16})
            Group(GGML_TYPE_Q8_0,dtype,rows,512,{1280,512,256,1280,512,256},true);
        printf("PASS GGUF shared MMVQ/MMQ cases=%d graph_cases=%d, bitwise ordinary Linear, launch counts, fallback\n",cases,graphs);
        return 0;
    }catch(const std::exception&e){std::cerr<<context<<": "<<e.what()<<std::endl;return 1;}
}
