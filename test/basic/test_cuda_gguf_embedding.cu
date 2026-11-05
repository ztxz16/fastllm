#include "cuda_gguf_data_test.cuh"
#include <limits>

static int cases = 0, checks = 0;
static std::vector<uint8_t> Packed(ggml_type type, int vocab, int width) {
    if (type != GGML_TYPE_IQ4_NL && type != GGML_TYPE_Q3_K) return Weights(type,vocab,width);
    std::vector<uint8_t> bytes(size_t(vocab)*ggml_row_size(type,width));
    uint32_t random=7821;
    for(auto &v:bytes){random=random*1664525U+1013904223U;v=random>>24;}
    for(size_t i=0;i<bytes.size();i+=ggml_type_size(type)) {
        const half scale=__float2half_rn(.003f);
        if(type==GGML_TYPE_IQ4_NL) reinterpret_cast<block_iq4_nl*>(bytes.data()+i)->d=scale;
        else reinterpret_cast<block_q3_K*>(bytes.data()+i)->d=scale;
    }
    return bytes;
}
static void Test(ggml_type type,int width,int rows,int repeats) {
    constexpr int vocab=17;
    Data weight(DATA_GGUF_FORMAT,int(type),{vocab,width}); Allocate(weight);
    Data ids(FLOAT32,{1,rows}); Allocate(ids);
    Data plain(FLOAT32,{rows,width}); Allocate(plain);
    const size_t count=size_t(rows)*repeats*width;
    Data storage(FLOAT32,{int(count)+16}),result; Allocate(storage);
    result.FakeFrom(storage,0);result.Resize({rows,repeats*width});
    Upload(weight,Packed(type,vocab,width));
    std::vector<float> input(rows,1),sentinel(count+16,42.0f);
    Upload(ids,input);Upload(storage,sentinel);
    Check(FastllmCudaGGUFEmbedding(ids,weight,plain),"embedding warmup failed");
    Check(FastllmCudaGGUFEmbeddingRepeat(ids,weight,result,repeats),"repeat warmup failed");
    Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph;cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    Check(FastllmCudaGGUFEmbeddingRepeat(ids,weight,result,repeats),"capture failed");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
    Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int pass=0;pass<6;++pass) {
        for(int row=0;row<rows;++row) {
            input[row]=float((row*7+pass)%vocab)+.25f;
            if(pass==2)input[row]=-1;
            if(pass==3)input[row]=float(vocab);
            if(pass==4)input[row]=std::numeric_limits<float>::quiet_NaN();
            if(pass==5)input[row]=std::numeric_limits<float>::infinity();
        }
        Upload(ids,input);Upload(storage,sentinel);
        Check(FastllmCudaGGUFEmbedding(ids,weight,plain),"reference embedding failed");
        if(pass%2) Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
        else Check(FastllmCudaGGUFEmbeddingRepeat(ids,weight,result,repeats),"eager repeat failed");
        std::vector<float> reference(size_t(rows)*width),actual(count+16);
        Cuda(cudaMemcpy(reference.data(),plain.cudaData,reference.size()*4,cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(actual.data(),storage.cudaData,actual.size()*4,cudaMemcpyDeviceToHost));
        for(int row=0;row<rows;++row)for(int r=0;r<repeats;++r) {
            Check(!std::memcmp(reference.data()+size_t(row)*width,
                  actual.data()+(size_t(row)*repeats+r)*width,width*4),"embedding repeat differs bitwise");
            ++checks;
        }
        Check(!std::memcmp(actual.data()+count,sentinel.data()+count,16*4),"embedding repeat overwrote guard");
    }
    Check(!FastllmCudaGGUFEmbeddingRepeat(ids,weight,result,0),"zero repeats accepted");
    Data invalid;
    invalid.FakeFrom(result,0); invalid.Resize({1});
    Check(!FastllmCudaGGUFEmbeddingRepeat(ids,weight,invalid,repeats),"short output accepted");
    invalid.FakeFrom(ids,0); invalid.Resize({1,0});
    Check(!FastllmCudaGGUFEmbeddingRepeat(invalid,weight,result,repeats),"empty input accepted");
    weight.dataDeviceIds={1};
    const bool rejectsForeignDevice = !FastllmCudaGGUFEmbeddingRepeat(ids,weight,result,repeats);
    weight.dataDeviceIds={0};
    Check(rejectsForeignDevice,"foreign weight device accepted");
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));++cases;
}
int main() {
    try {
        if (!FastllmCudaGetDeviceCount()) return 77;
        Cuda(cudaSetDevice(0));
        for(auto type:{GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_Q3_K})
            for(int width:{256,512,2560,4096})for(int rows:{1,3,8})for(int repeats:{1,4})
                Test(type,width,rows,repeats);
        std::cout<<"PASS GGUF embedding repeat cases="<<cases<<" bitwise_rows="<<checks<<"\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<"\n"; return 1; }
}
