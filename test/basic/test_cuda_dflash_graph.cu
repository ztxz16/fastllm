#include <cuda_runtime.h>
#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
using namespace fastllm;

// Linked with --wrap against fastllm's objects (see CMakeLists.txt). Observe
// actual successful runtime calls, including destruction during serving reset.
// No counters or hooks are needed in the inference library.
struct GraphTrace {
    size_t instantiated = 0, launched = 0, destroyed = 0;
    bool invalid = false;
    std::map<cudaGraphExec_t, size_t> live;
} graphTrace;

extern "C" cudaError_t CUDARTAPI __real_cudaGraphInstantiate(
    cudaGraphExec_t *, cudaGraph_t, cudaGraphNode_t *, char *, size_t);
extern "C" cudaError_t CUDARTAPI __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
extern "C" cudaError_t CUDARTAPI __real_cudaGraphExecDestroy(cudaGraphExec_t);

extern "C" cudaError_t CUDARTAPI __wrap_cudaGraphInstantiate(
        cudaGraphExec_t *exec, cudaGraph_t graph, cudaGraphNode_t *error,
        char *log, size_t size) {
    const auto status = __real_cudaGraphInstantiate(exec, graph, error, log, size);
    if (status == cudaSuccess) {
        size_t nodes = 0;
        if (cudaGraphGetNodes(graph, nullptr, &nodes) != cudaSuccess || nodes == 0 ||
            !exec || !*exec || !graphTrace.live.emplace(*exec, 0).second)
            graphTrace.invalid = true;
        ++graphTrace.instantiated;
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    const auto status = __real_cudaGraphLaunch(exec, stream);
    if (status == cudaSuccess) {
        const auto found = graphTrace.live.find(exec);
        if (found == graphTrace.live.end()) graphTrace.invalid = true;
        else ++found->second;
        ++graphTrace.launched;
    }
    return status;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGraphExecDestroy(cudaGraphExec_t exec) {
    const auto status = __real_cudaGraphExecDestroy(exec);
    if (status == cudaSuccess) {
        const auto found = graphTrace.live.find(exec);
        if (found == graphTrace.live.end() || found->second < 2) graphTrace.invalid = true;
        else graphTrace.live.erase(found);
        ++graphTrace.destroyed;
    }
    return status;
}

void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Sync() { Check(cudaDeviceSynchronize() == cudaSuccess, "CUDA sync"); }
std::vector<float> Values(size_t count, float scale, int seed) {
    std::vector<float> v(count);
    for (size_t i=0;i<count;++i) v[i]=scale*std::sin(float(i%1009+seed)*.13f);
    return v;
}
void Fill(Data &x, DataType type, const std::vector<int> &shape, int device, int seed) {
    size_t n=1; for (int d:shape) n*=d;
    x.CopyFrom(Data(type,shape,Values(n,.2f,seed)));
    x.ToDevice(DataDevice::CUDA,{device},true);
}
std::vector<unsigned char> Download(const Data &x) {
    Sync(); std::vector<unsigned char> v(x.GetBytes());
    const auto kind=x.dataDevice==DataDevice::CUDA?cudaMemcpyDeviceToHost:cudaMemcpyHostToHost;
    Check(cudaMemcpy(v.data(),x.dataDevice==DataDevice::CUDA?x.cudaData:x.cpuData,v.size(),kind)==cudaSuccess,"download");
    return v;
}
std::vector<unsigned char> DownloadCache(const Data &x) {
    Sync();const size_t row=x.dims[1]*x.dims[2]*2;
    std::vector<unsigned char> v(x.dims[0]*row);
    Check(cudaMemcpy2D(v.data(),row,x.cudaData,x.strides[0]*2,row,x.dims[0],cudaMemcpyDeviceToHost)==cudaSuccess,"cache download");
    return v;
}
void FillCache(Data &x,int device,int seed) {
    if (x.dims.empty() || x.dataDeviceIds!=std::vector<int>{device}) {
        x.FreeSpace();x.dims.clear();x.strides.clear();x.expansionDims.clear();
        x.dataType=DataType::FLOAT16;x.UpdateUnitSize();
        x.dataDevice=DataDevice::CUDA;x.dataDeviceIds={device};x.Resize({2,16,128});x.Allocate(false);
    }
    Data host(DataType::FLOAT16,{2,16,128},Values(2*16*128,.2f,seed));
    Check(cudaMemcpy2D(x.cudaData,x.strides[0]*2,host.cpuData,16*128*2,16*128*2,2,cudaMemcpyHostToDevice)==cudaSuccess,"cache fill");
}
struct Fixture : Qwen3_5Model {
    Fixture() {
        const int vocab=getenv("TEST_VOCAB")?atoi(getenv("TEST_VOCAB")):512;
        embed_dim=512; dflashEnabled=true; dflashLayers=1;
        dflashMaskTokenId=127; dflashSlidingWindow=256; max_positions=262144;
        dflashCheckpointBlockSize=dflashRuntimeBlockSize=8;
        dflashHeads=4; dflashKvHeads=2; dflashHeadDim=128;
        dflashIntermediateSize=512; dflashConvGroupSize=64; dflashConvKernelSize=4;
        dflashSelectorRank=64; dflashSelectorTopK=16; dflashTargetLayerIds={0,1};
        auto add=[&](const std::string &name,std::vector<int> shape,bool norm=false) {
            size_t n=1;for(int d:shape)n*=d;
            Data &w=weight[name];
            w.CopyFrom(Data(norm?DataType::FLOAT32:DataType::BFLOAT16,shape,
                           norm?std::vector<float>(n,1.0f):Values(n,.025f,int(name.size()))));
            w.name=name;w.isModelWeight=true;
        };
        add("dflash.fc.weight",{512,1024});
        add("dflash.hidden_norm.weight",{512},true); add("dflash.norm.weight",{512},true);
        add("dflash.candidate_selector.hidden_projection.weight",{64,512});
        add("dflash.candidate_selector.predecessor_codebook",{vocab,64});
        add("dflash.candidate_selector.successor_codebook",{vocab,64});
        add("dflash.fused_kv_qkv.weight",{1536,512});
        const std::string p="dflash.layers.0.";
        add(p+"input_layernorm.weight",{512},true);add(p+"post_attention_layernorm.weight",{512},true);
        add(p+"self_attn.q_norm.weight",{128},true);add(p+"self_attn.k_norm.weight",{128},true);
        add(p+"self_attn.o_proj.weight",{512,512});
        add(p+"mlp.gateup_proj.weight",{1024,512});add(p+"mlp.down_proj.weight",{512,512});
        for(const std::string side:{"attention_conv.","mlp_conv."}) {
            add(p+side+"kernel_projection.weight",{64,512});add(p+side+"base_kernel",{2,4,512});
        }
        add(language_prefix+"embed_tokens.weight",{vocab,512});add("lm_head.weight",{vocab,512});
    }
    void PrepareForDevice(int device) {
        FastllmCudaSetDevice(device);
        deviceMap={{"cuda:"+std::to_string(device),1}};
        weight["lm_head.weight"].ToDevice(DataDevice::CUDA,{device},true);
        weight[language_prefix+"embed_tokens.weight"].ToDevice(DataDevice::CUDA,{device},true);
        PrepareDFlashWeightsForDevice(device);PrepareDFlashRotary(device);
        Check(dflashNvfp4ViewWeights.size()==2,"private view count");
        Check(!dflashNvfp4DraftLmHead.dims.empty(),"private head");
        Check(weight["lm_head.weight"].dataType==DataType::BFLOAT16,"target head changed");
        Check(weight["dflash.all_kv.weight"].isFake,"original view ownership changed");
    }
    void SetShape(int checkpoint, int runtime) { dflashCheckpointBlockSize=checkpoint; dflashRuntimeBlockSize=runtime; }
    void CheckRotaryStable(int device) {
        void *pointer=dflashRopeInvFreq.cudaData;
        PrepareDFlashRotary(device);
        Check(dflashRopeInvFreq.cudaData==pointer,"RoPE frequencies reallocated");
        Check(dflashRopeInvFreq.GetBytes()==64*4,"RoPE frequencies size");
    }
    size_t graphCalls = 0;
    void ResetServing() {ResetCudaServingForKvCacheResize();graphCalls=0;}
    DFlashContext context;
    std::vector<int> Run(int device,int seed,bool graph) {
        SetCudaGraph(graph);
        const int positions[]={32,65528,65535,131065,204792,262128};
        context.committedTokens=positions[seed%6];
        if (context.draftKeyValues.empty()) context.draftKeyValues.resize(1);
        FillCache(context.draftKeyValues[0].first,device,seed);
        FillCache(context.draftKeyValues[0].second,device,seed+9);
        const auto beforeKey=DownloadCache(context.draftKeyValues[0].first);
        const auto beforeValue=DownloadCache(context.draftKeyValues[0].second);
        GenerationConfig config;config.temperature=0;config.top_k=1;config.top_p=1;
        const size_t instantiated = graphTrace.instantiated, launched = graphTrace.launched;
        auto tokens=RunDFlashDraft(device,{device},seed*7%100,config,context);
        Sync();
        const size_t expectedCaptures = graph && graphCalls == 1 ? 2 * dflashLayers : 0;
        const size_t expectedLaunches = graph && graphCalls >= 1 ? 2 * dflashLayers : 0;
        Check(!graphTrace.invalid,"invalid CUDA graph lifecycle");
        Check(graphTrace.instantiated-instantiated==expectedCaptures,
              "draft graph capture count mismatch (unexpected eager fallback or recapture)");
        Check(graphTrace.launched-launched==expectedLaunches,
              "draft graph replay count mismatch (unexpected eager fallback)");
        if (graph) ++graphCalls;
        Check(DownloadCache(context.draftKeyValues[0].first)==beforeKey,"key cache changed");
        Check(DownloadCache(context.draftKeyValues[0].second)==beforeValue,"value cache changed");
        Check(tokens.size()==size_t(dflashRuntimeBlockSize-1),"wrong proposal size");
        tokens.insert(tokens.end(),context.proposalCandidateIds.begin(),context.proposalCandidateIds.end());
        for (float value:context.proposalCandidateProbs) {int bits;memcpy(&bits,&value,4);tokens.push_back(bits);}
        SetCudaGraph(true);
        return tokens;
    }
};
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            puts("SKIP: no CUDA device");return 77;
        }
        FastllmCudaSetDevice(0);
        if (!FastllmCudaMarlinNVFP4Supported(64,512) || !FastllmCudaMarlinNVFP4Supported(512,512)) {
            puts("SKIP: draft graph fixture requires NVFP4 Marlin kernels");return 77;
        }
        setenv("FASTLLM_DRAFT_QUANT","nvfp4",1);
        setenv("FASTLLM_CUDA_NVFP4_NATIVE_LAYOUT","0",1);
        SetCudaGraph(true);SetCudaEmbedding(true);
        Executor executor; SetCurrentThreadExecutor(&executor);
        Fixture graphed,eager;
        for(int device:{0}) {
            executor.SetFirstDevice("cuda:"+std::to_string(device));
            FastllmCudaSetNcclForceSync(true);
            graphed.PrepareForDevice(device);eager.PrepareForDevice(device);
            FastllmCudaSetNcclForceSync(false);
            int cases=0;
            // Repeating the final shape verifies reset forces fresh capture
            // even when checkpoint/runtime sizes have not changed.
            for(int checkpoint:{2,4,8,16,8,8}) {
                for(int runtime=2;runtime<=std::min(checkpoint,8);++runtime) {
                    graphed.SetShape(checkpoint,runtime);eager.SetShape(checkpoint,runtime);
                    for(int i=0;i<4;++i) {
                        if(i==2 && runtime==2) {graphed.CheckRotaryStable(device);eager.CheckRotaryStable(device);}
                        auto actual=graphed.Run(device,++cases,true);
                        auto expected=eager.Run(device,cases,false);
                        Check(actual==expected,"graph proposal/candidate/probability mismatch");
                    }
                    printf("PASS checkpoint=%d runtime=%d cases=%d\n",checkpoint,runtime,cases);fflush(stdout);
                }
                const size_t destroyed=graphTrace.destroyed;
                graphed.ResetServing();eager.ResetServing();
                Check(!graphTrace.invalid && graphTrace.live.empty(),"reset retained CUDA graph executables");
                Check(graphTrace.destroyed-destroyed==2,"reset did not destroy prefix and tail graphs");
            }
            graphed.ResetServing();eager.ResetServing();
        }
        Check(!graphTrace.invalid && graphTrace.live.empty(),"invalid final CUDA graph lifecycle");
        printf("ALL_PASS full draft graph lifecycle captures=%zu launches=%zu destroys=%zu\n",
               graphTrace.instantiated,graphTrace.launched,graphTrace.destroyed);return 0;
    } catch(const std::exception &e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
      catch(...) {fprintf(stderr,"FAIL unknown exception\n");return 2;}
}
