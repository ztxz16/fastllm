#include "models/qwen4_exp.h"
#include "../../src/models/qwen4_exp_execution.h"
#include "utils.h"
#include "executor.h"
#include "gguf.h"
#include "utils/kernel_registry.h"
#include <cstring>
#include "fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
namespace {
void Check(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
void Equal(const Data &a, const Data &b, const char *name, double tolerance = 0.003) {
    Check(a.dims == b.dims, name);
    if (a.dims.empty()) return;
    Data x, y; ToDataType(a, x, FLOAT32); ToDataType(b, y, FLOAT32);
    x.ToDevice(CPU); y.ToDevice(CPU);
    double error = 0, energy = 0, maximum = 0;
    uint64_t count=1; for(int dim:x.dims)count*=dim;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t remainder=i, xi=0, yi=0;
        for(int d=int(x.dims.size())-1;d>=0;--d) {
            const uint64_t c=remainder%x.dims[d];remainder/=x.dims[d];
            xi+=c*x.strides[d];yi+=c*y.strides[d];
        }
        double v = ((float*)x.cpuData)[xi], ref = ((float*)y.cpuData)[yi];
        Check(std::isfinite(v) && std::isfinite(ref), "nonfinite tensor");
        error += (v-ref)*(v-ref); energy += ref*ref; maximum = std::max(maximum, std::abs(v-ref));
    }
    double l2 = std::sqrt(error / std::max(energy, 1e-16));
    if (l2 > tolerance && maximum > 1e-5) {
        std::cerr << name << " relative_l2=" << l2 << " max=" << maximum << '\n';
        Check(false, name);
    }
}
}
namespace fastllm {
struct Qwen4GGUFTestAccess {
    static std::unique_ptr<Qwen4ExpModel> Model(bool uniformHead=false, bool quantizedExperts=false, int sparseBudget=8) {
        auto p = std::make_unique<Qwen4ExpModel>(); auto &m = *p;
        m.block_cnt=2; m.embed_dim=256; m.head_dim=128; m.rotary_dim=64;
        m.num_attention_heads=2; m.num_key_value_heads=1;
        m.num_k_heads=1; m.num_v_heads=2; m.head_k_dim=128; m.head_v_dim=128;
        m.linearConvKernel=4; m.hcCount=2; m.linearLayers={true,false}; m.kvCacheId=1;
        m.num_experts=4; m.num_experts_per_tok=2; m.norm_topk_prob=true; m.routed_scaling_factor=1;
        m.indexerHeads=2; m.indexerKvHeads=1; m.indexerHeadDim=128;
        m.indexerBudget=sparseBudget; m.indexerCompressRatio=4; m.mropeSections={11,11,10};
        m.pleLayer=1; m.pleEmbedDim=32; m.pleConvKernel=4; m.ngramSize=3;
        m.ngramHeads=2; m.headsPerNgram=1; m.ngramHeadDim=16; m.ngramShardCount=1;
        m.pleMultipliers={23703573157769ULL,20109073645365ULL,8052911324071ULL};
        m.pleHeadOffsets={0,17}; m.pleHeadVocabSizes={17,19}; m.eosToken=1;
        m.eos_token_id=1; m.dataType=FLOAT16; m.kvCacheDataType=FLOAT16;
        m.executionMode=ExecutionMode::Gpu; m.deviceMap={{"cuda:0",1}}; m.moeDeviceMap={{"cuda:0",1}};
        m.maxBatch=7; m.saveHistoryChat=false;
        unsigned seed=97;
        auto put=[&](const std::string &name, std::vector<int> dims, DataType type=FLOAT16, float constant=999.f) {
            int n=1; for(int d:dims)n*=d; std::vector<float> v(n);
            for(int i=0;i<n;++i) { seed=1664525u*seed+1013904223u; v[i]=constant==999.f ? (float(int(seed>>16)-32768)/32768)*.025f : constant; }
            Data value(FLOAT32,dims,v); ToDataType(value,type); m.weight[name].CopyFrom(value);
            if ((uniformHead || quantizedExperts) && name.find(".experts.") != std::string::npos) {
                Data packed(DATA_GGUF_FORMAT, GGML_TYPE_Q8_0, dims); packed.Allocate(false);
                for(size_t off=0;off<packed.GetBytes();off+=ggml_type_size(GGML_TYPE_Q8_0)) {
                    const uint16_t scale=0x1400; std::memcpy(packed.cpuData+off,&scale,2);
                    for(int j=0;j<32;++j){ seed=1664525u*seed+1013904223u; packed.cpuData[off+2+j]=uint8_t(int(seed>>27)-16); }
                }
                m.weight[name].CopyFrom(packed);
            }
            m.weight[name].name=name; m.weight[name].isModelWeight=true;
        };
        auto hyper=[&](const std::string &prefix, bool inject) {
            put(prefix+"hc_norm.weight",{512},FLOAT32,0);
            put(prefix+"input_mix_weight_down.weight",{8,512});
            put(prefix+"input_mix_weight_up.weight",{512,8});
            if(inject)put(prefix+"block_inject_weight.weight",{2,512});
        };
        auto full=[&](const std::string &a) {
            put(a+"q_proj.weight",{512,256}); put(a+"k_proj.weight",{128,256});
            put(a+"v_proj.weight",{128,256}); put(a+"o_proj.weight",{256,256});
            for(const auto &n:{"q_norm.weight","k_norm.weight","indexer.q_layernorm.weight","indexer.k_layernorm.weight"})put(a+n,{128},FLOAT32,0);
            put(a+"indexer.index_qk_proj.weight",{384,256});
        };
        auto moe=[&](const std::string &a) {
            put(a+"gate.weight",{4,256}); put(a+"shared_expert_gate.weight",{1,256});
            put(a+"shared_expert.gateup_proj.weight",{128,256}); put(a+"shared_expert.down_proj.weight",{256,64});
            for(int e=0;e<4;++e) {
                put(a+"experts."+std::to_string(e)+".gateup_proj.weight",{128,256});
                put(a+"experts."+std::to_string(e)+".down_proj.weight",{256,64});
            }
        };
        put(m.languagePrefix+"embed_tokens.weight",{128,256}); put("lm_head.weight",{128,256},FLOAT16,uniformHead ? 0.f : 999.f);
        for(int l=0;l<2;++l) {
            std::string a=m.languagePrefix+"layers."+std::to_string(l)+".";
            hyper(a+"attn_hyper_connection.",true); hyper(a+"mlp_hyper_connection.",true); moe(a+"mlp.");
            if(l==1)full(a+"self_attn.");
        }
        std::string g=m.languagePrefix+"layers.0.linear_attn.";
        put(g+"in_proj_qkv.weight",{512,256}); put(g+"in_proj_z.weight",{256,256});
        put(g+"in_proj_a.weight",{2,256}); put(g+"in_proj_b.weight",{2,256});
        put(g+"conv1d.weight",{512,4},FLOAT32,.2f); put(g+"A_log",{2},FLOAT32,0); put(g+"dt_bias",{2},FLOAT32,0);
        put(g+"norm.weight",{128},FLOAT32,1); put(g+"out_proj.weight",{256,256});
        m.linearInvScaleData.CopyFrom(Data(FLOAT32,{128},std::vector<float>(128,1/std::sqrt(128.f))));
        std::string ple=m.languagePrefix+"layers.1.ple.";
        put(ple+"ple_embedding.ngram_embedding.shard_0.weight",{36,16},BFLOAT16);
        put(ple+"key_proj.weight",{512,32}); put(ple+"value_proj.weight",{256,32});
        for(const auto &n:{"norm_key.weight","norm_query.weight","norm_conv.weight"})put(ple+n,{512},FLOAT32,0);
        put(ple+"conv1d.weight",{512,1,4},FLOAT32,.2f);
        hyper(m.languagePrefix+"hyper_connection_mixer.",false);
        put("mtp.pre_fc_norm_embedding.weight",{256},FLOAT32,0); put("mtp.pre_fc_norm_hidden.weight",{512},FLOAT32,0);
        put("mtp.fc_embedding.weight",{256,256}); put("mtp.fc_hidden.weight",{256,256});
        hyper("mtp.layers.0.attn_hyper_connection.",true);hyper("mtp.layers.0.mlp_hyper_connection.",true);
        hyper("mtp.hyper_connection_mixer.",false); full("mtp.layers.0.self_attn."); moe("mtp.layers.0.mlp.");
        // The oracle compares the same floating head; automatic NVFP4 head
        // quantization can flip near-tied greedy proposals after tiny rounding
        // differences. Quantized operator accuracy is tested independently.
        m.mtpDraftLmHeadAttempted=true;
        m.PrepareWeights(); return p;
    }
    static void Run(bool tuned = false) {
        Check(fastllm_kernel_configure(tuned ? "{\"mode\":\"quick\",\"budget_ms\":30000}" : "{\"mode\":\"off\"}") == 1, "kernel policy");
        SetDeviceMap({{"cuda:0",1}});SetMoeDeviceMap({{"cuda:0",1}});SetCudaGraph(false);
        SetCudaEmbedding(true);SetMoeCudaCacheBytes(0);setenv("FASTLLM_QWEN4_ENABLE_MTP","0",1);
        // Quantized resident experts support graph composition; retain the
        // nonconstant head so the independent logit oracle remains meaningful.
        auto model=Model(false,true,tuned ? 128 : 8); auto &m=*model;
        using Cache=std::vector<std::pair<Data,Data>>;
        const int count=5;
        std::vector<Cache> serial(count),batch(count);
        std::vector<int> positions(count),anchors(count);
        GenerationConfig cfg; cfg.top_k=1;cfg.output_logits=true;cfg.repeat_penalty=1;cfg.output_token_limit=64;
        for(int i=0;i<count;++i) {
            serial[i].resize(2);batch[i].resize(2);
            for(auto *c:{&serial[i],&batch[i]})for(auto &kv:*c){kv.first.isKVCache=true;kv.second.isKVCache=true;}
            positions[i]=(tuned ? 129 : 9)+i*3;anchors[i]=27+i;
            // Single-row priming exercises decode recurrence without a separate
            // numerical prefill algorithm. Histories deliberately differ.
            for(int t=0;t<positions[i];++t) {
                Data ids(FLOAT32,{1,1},{float((3+i*7+t)%120+2)}),pos(FLOAT32,{1,1},{float(t)});
                m.Forward(ids,Data(),pos,serial[i],cfg);m.Forward(ids,Data(),pos,batch[i],cfg);
            }
        }
        // Stable stretches must replay, while request departures/reordering
        // must invalidate the composition without losing independent state.
        for(int round=0;round<48;++round) {
            SetCudaGraph(false);
            std::vector<int> order={4,1,3,0,2}; if((round/8)%2)std::reverse(order.begin(),order.end());
            const int sizes[]={5,3,2,4,1,5};
            int live=sizes[round/8]; order.resize(live);
            std::vector<float> ids;std::vector<Data> pos(live);
            std::vector<Data*> masks(live,nullptr),ptrs;std::vector<Cache*> caches;
            std::vector<GenerationConfig> configs(live,cfg);std::vector<int> lens(live,1);
            std::vector<std::vector<float>> expected(live),actual(live);std::vector<std::vector<float>*> out;
            for(int j=0;j<live;++j) {
                int i=order[j];ids.push_back(anchors[i]);pos[j].CopyFrom(Data(FLOAT32,{1,1},{float(positions[i])}));
                Data input(FLOAT32,{1,1},{float(anchors[i])});m.Forward(input,Data(),pos[j],serial[i],cfg,LastTokensManager(),&expected[j]);
                ptrs.push_back(&pos[j]);caches.push_back(&batch[i]);out.push_back(&actual[j]);
            }
            SetCudaGraph(true);
            if(round%2) {
                std::vector<std::pair<Data*,Data*>> pairs;
                for(auto *cache:caches)for(auto &kv:*cache)pairs.emplace_back(&kv.first,&kv.second);
                m.ForwardBatch(live,Data(FLOAT32,{1,live},ids),masks,ptrs,lens,pairs,configs,LastTokensManager(),&out);
            } else m.ForwardRequests(Data(FLOAT32,{1,live},ids),masks,ptrs,lens,caches,configs,LastTokensManager(),&out);
            for(int j=0;j<live;++j) {
                int i=order[j];Equal(Data(FLOAT32,{int(actual[j].size())},actual[j]),Data(FLOAT32,{int(expected[j].size())},expected[j]),"decode logits");
                for(int l=0;l<2;++l) {Equal(batch[i][l].first,serial[i][l].first,"key/conv");Equal(batch[i][l].second,serial[i][l].second,"value/recurrent");}
                auto &a=m.requestStates[&batch[i][0].first],&b=m.requestStates[&serial[i][0].first];
                Check(a.processedTokens==b.processedTokens,"PLE token history");
                Check(a.previousToken1==b.previousToken1 && a.previousToken2==b.previousToken2,"PLE previous tokens");
                Equal(*a.pleConvHistoryTensors[a.pleConvHistoryIndex],*b.pleConvHistoryTensors[b.pleConvHistoryIndex],"PLE convolution history");
                Equal(*a.indexerTailKeyTensors[1],*b.indexerTailKeyTensors[1],"QSA tail keys");
                Equal(*a.indexerTailPositionTensors[1],*b.indexerTailPositionTensors[1],"QSA tail positions");
                Equal(*a.indexerBlockKeyTensors[1],*b.indexerBlockKeyTensors[1],"QSA compressed keys");
                ++positions[i];++anchors[i];
            }
        }
        std::cout<<"PASS no-spec graph decode: five unrelated histories, replay, reorder, partial/single batch, logits and recurrent/KV/QSA/PLE state\n";
        for(auto &c:serial)m.ClearWarmupCache(c);for(auto &c:batch)m.ClearWarmupCache(c);
        for(bool uniformHead : {false,true}) {
        SetCudaGraph(false);
        setenv("FASTLLM_QWEN4_ENABLE_MTP","3",1); setenv("FASTLLM_QWEN4_MTP_MIN_P","0",1);
        auto speculative = Model(uniformHead,false,tuned ? 128 : 8); auto &m = *speculative;
        cfg.output_logits=false;
        std::vector<GenerationConfig> mtpConfigs(count,cfg);
        std::vector<int> emitted(count);
        for(int i=0;i<count;++i) {
            serial[i].resize(2);batch[i].resize(2);
            for(auto *c:{&serial[i],&batch[i]})for(auto &kv:*c){kv.first.isKVCache=true;kv.second.isKVCache=true;}
            positions[i]=tuned ? 129 : 9;
            mtpConfigs[i].input_token_length=positions[i];
            std::vector<float> ids(positions[i]),pos(positions[i]);
            for(int t=0;t<positions[i];++t){ids[t]=(3+i*7+t)%120+2;pos[t]=t;}
            Data input(FLOAT32,{1,positions[i]},ids),position(FLOAT32,{1,positions[i]},pos);
            anchors[i]=m.Forward(input,Data(),position,serial[i],mtpConfigs[i]);
            Data before; before.CopyFrom(serial[i][1].first); ForceDeviceSync();
            Data input2(FLOAT32,{1,positions[i]},ids),position2(FLOAT32,{1,positions[i]},pos);
            Equal(input,input2,"MTP mutated input IDs"); Equal(position,position2,"MTP mutated positions");
            emitted[i]=m.Forward(input2,Data(),position2,batch[i],mtpConfigs[i]);
            Equal(serial[i][1].first,before,"scalar target cache changed after another request");
            Check(anchors[i]==emitted[i],"MTP prefill token mismatch");
            auto &ps=m.requestStates[&serial[i][0].first], &pb=m.requestStates[&batch[i][0].first];
            Equal(*ps.pleConvHistoryTensors[ps.pleConvHistoryIndex],*pb.pleConvHistoryTensors[pb.pleConvHistoryIndex],"prefill PLE history");
            Check(ps.previousToken1==pb.previousToken1 && ps.previousToken2==pb.previousToken2,"prefill PLE tokens");

            for(int l=0;l<2;++l){Equal(batch[i][l].first,serial[i][l].first,"prefill key/conv");Equal(batch[i][l].second,serial[i][l].second,"prefill value/recurrent");}
            for(int t=0;t<i*3;++t) {
                Data pos(FLOAT32,{1,1},{float(positions[i])}),a(FLOAT32,{1,1},{float(anchors[i])}),b(FLOAT32,{1,1},{float(emitted[i])});
                anchors[i]=m.Forward(a,Data(),pos,serial[i],mtpConfigs[i]);
                emitted[i]=m.Forward(b,Data(),pos,batch[i],mtpConfigs[i]);++positions[i];
                Check(anchors[i]==emitted[i],"MTP priming token mismatch");
            }
        }
        bool queued=false;
        for(int round=0;round<(uniformHead ? 72 : 16);++round) {
            std::vector<int> order={3,0,4,2,1};if((round/24)%2)std::reverse(order.begin(),order.end());
            SetCudaGraph(false);
            if(round==7)order.resize(3);
            const int live=order.size();
            std::vector<float> ids;std::vector<Data> pos(live);std::vector<Data*> masks(live,nullptr),ptrs;
            std::vector<Cache*> caches;std::vector<GenerationConfig> configs;std::vector<int> lens(live,1),expected;
            for(int j=0;j<live;++j) {
                int i=order[j]; ids.push_back(emitted[i]);pos[j].CopyFrom(Data(FLOAT32,{1,1},{float(positions[i])}));
                Data input(FLOAT32,{1,1},{float(anchors[i])});
                expected.push_back(m.Forward(input,Data(),pos[j],serial[i],mtpConfigs[i]));
                ptrs.push_back(&pos[j]);caches.push_back(&batch[i]);configs.push_back(mtpConfigs[i]);
            }
            SetCudaGraph(true);
            auto actual=m.ForwardRequests(Data(FLOAT32,{1,live},ids),masks,ptrs,lens,caches,configs,LastTokensManager(),nullptr);
            Check(actual==expected,"MTP emitted token mismatch");
            for(int j=0;j<live;++j) {
                const int i=order[j];anchors[i]=expected[j];emitted[i]=actual[j];++positions[i];
                auto &a=m.requestStates[&batch[i][0].first],&b=m.requestStates[&serial[i][0].first];
                if(!a.mtpState->targetCapture.linearConvInputs.empty() && !b.mtpState->targetCapture.linearConvInputs.empty())
                    Equal(a.mtpState->targetCapture.linearConvInputs[0],b.mtpState->targetCapture.linearConvInputs[0],"captured conv input");
                for(int l=0;l<2;++l){Equal(batch[i][l].first,serial[i][l].first,"MTP committed key/conv");Equal(batch[i][l].second,serial[i][l].second,"MTP committed value/recurrent");}
                Check(a.processedTokens==b.processedTokens,"MTP processed token history");
                if (uniformHead) Check(a.mtpState->proposals==b.mtpState->proposals,"MTP proposal mismatch");
                else {
                    // A tiny random head has near ties: graph and eager QSA
                    // rounding may pick different valid drafts. Feed the same
                    // candidates to both verifiers so the numeric/state oracle
                    // continues to compare identical inputs after that point.
                    for(int token:a.mtpState->proposals) Check(token>=0 && token<128,"invalid draft token");
                    a.mtpState->proposals=b.mtpState->proposals;
                }
                Check(a.mtpState->pendingOutputTokens==b.mtpState->pendingOutputTokens,"MTP output queue mismatch");
                queued |= !a.mtpState->pendingOutputTokens.empty();
                Equal(a.mtpState->key,b.mtpState->key,"MTP committed draft key");
                Equal(a.mtpState->value,b.mtpState->value,"MTP committed draft value");
                Equal(*a.pleConvHistoryTensors[a.pleConvHistoryIndex],*b.pleConvHistoryTensors[b.pleConvHistoryIndex],"MTP PLE rollback");
            }
        }
        Check(!uniformHead || queued,"high-acceptance fixture never queued verified tokens");
        for(auto &c:serial)m.ClearWarmupCache(c);for(auto &c:batch)m.ClearWarmupCache(c);
        std::cout<<"PASS cross-request MTP: verifier/draft batching, partial acceptance, queues and independent rollback\n";
        }
        {
            SetCudaGraph(false);
            auto sampledModel=Model(true);auto &m=*sampledModel;
            std::vector<Cache> caches(count);std::vector<Cache*> owners;std::vector<int> next(count),positions(count,1),lens(count,1);
            std::vector<GenerationConfig> configs(count,cfg);
            for(int i=0;i<count;++i) {
                configs[i].top_k=128;configs[i].top_p=i<3 ? .01f+.2f*i : 1;
                configs[i].temperature=.5f+.2f*i;
                // Different target filters cause unequal accepted prefixes;
                // draft replay must update only each request's valid rows.
                configs[i].input_token_length=1;configs[i].output_logits=false;
                caches[i].resize(2);for(auto &kv:caches[i]){kv.first.isKVCache=true;kv.second.isKVCache=true;}
                owners.push_back(&caches[i]);
                next[i]=m.Forward(Data(FLOAT32,{1,1},{float(4+i)}),Data(),Data(FLOAT32,{1,1},{0}),caches[i],configs[i]);
            }
            bool accepted=false; SetCudaGraph(true);
            for(int step=0;step<48;++step) {
                std::vector<Data> pos(count);std::vector<Data*> pointers,masks(count,nullptr);
                for(int i=0;i<count;++i){pos[i].CopyFrom(Data(FLOAT32,{1,1},{float(positions[i]++)}));pointers.push_back(&pos[i]);}
                next=m.ForwardRequests(Data(FLOAT32,{1,count},std::vector<float>(next.begin(),next.end())),masks,pointers,lens,owners,configs,LastTokensManager(),nullptr);
                for(int i=0;i<count;++i) {
                    Check(next[i]>=0 && next[i]<128,"sampled token out of vocabulary");
                    auto &mtp=*m.requestStates[&caches[i][0].first].mtpState;
                    Check(mtp.sampleProposal && mtp.proposalCount==3 && mtp.proposals.size()==3,"sampled draft distributions missing");
                    Check(mtp.proposalTemperature==configs[i].temperature,"cross-request temperature contamination");
                    Data saved; saved.CopyFrom(mtp.proposalLogits); saved.ToDevice(CPU);
                    for(int j=0;j<3*128;++j)Check(std::isfinite(((float*)saved.cpuData)[j]) && ((float*)saved.cpuData)[j]==0,"uniform proposal logits corrupted");
                    accepted |= !mtp.pendingOutputTokens.empty();
                }
            }
            Check(accepted,"uniform sampled fixture never accepted a draft");
            for(auto &c:caches)m.ClearWarmupCache(c);
            std::cout<<"PASS sampled MTP batching: independent temperatures, full proposal distributions and accepted queues\n";
        }
        {
            auto model=Model(true); auto &m=*model;
            ResponseContext context;context.pastKeyValues.resize(2);
            for(auto &kv:context.pastKeyValues){kv.first.isKVCache=true;kv.second.isKVCache=true;}
            cfg.input_token_length=1;cfg.output_logits=false;cfg.top_k=1;
            int token=3,position=0,published=0;
            for(int cycle=0;cycle<4;++cycle) {
                token=m.Forward(Data(FLOAT32,{1,1},{float(token)}),Data(),Data(FLOAT32,{1,1},{float(position++)}),context.pastKeyValues,cfg);
                const int length=context.pastKeyValues[1].first.dims[1];
                while(m.PopPendingResponseToken(&context,token,token)) {++position;++published;}
                Check(context.pastKeyValues[1].first.dims[1]==length,"publishing pending tokens executed attention");
            }
            Check(published>=6,"scheduler did not drain accepted blocks");
            m.ClearWarmupCache(context.pastKeyValues);
            std::cout<<"PASS scheduler publishes accepted blocks without additional forwards\n";
        }
    }
};
}
int main(int argc,char **argv) { try {SetThreads(2);FastllmCudaSetDevice(0);Qwen4GGUFTestAccess::Run(argc>1 && std::string(argv[1])=="--tuned-sparse");return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
