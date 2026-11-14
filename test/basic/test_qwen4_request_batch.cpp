#include "models/qwen4_exp.h"
#include "models/qwen4_paged_cache.h"
#include "../../src/models/qwen4_exp_execution.h"
#include "../../src/models/qwen4_exp_graph.h"
#include "../../src/models/qwen4_exp_qsa.h"
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
#include <thread>
#include <exception>
#include <future>
#include <cuda_runtime_api.h>
using namespace fastllm;
namespace {
bool quantizedAttention = false;
void Check(bool ok, const char *what) { if (!ok) throw std::runtime_error(what); }
void Equal(const Data &a, const Data &b, const char *name, double tolerance = 0.003) {
    Check(a.dims == b.dims, name);
    if (a.dims.empty()) return;
    auto materialize=[](const Data &source, Data &result) {
        if (!Qwen4IsPagedKV(source)) {
            // The assertion must not migrate the live request's tensor to the
            // executor's current device when inspecting a different PP stage.
            Data copy; copy.CopyFrom(source); copy.ToDevice(CPU);
            ToDataType(copy,result,FLOAT32); result.ToDevice(CPU); return;
        }
        // A paged cache's cudaData is its page table, not its KV payload.
        // Gather logical rows independently from the pool's physical pages.
        const auto &pool=*source.pagedKVCacheData;
        Check(pool.dims.size()==4 && source.dims.size()==3,"paged oracle layout");
        std::vector<uint8_t> pages(pool.GetBytes());
        { Qwen4CudaDeviceGuard guard(pool.dataDeviceIds);
          FastllmCudaCopyFromDeviceToHost(pages.data(),pool.cudaData,pages.size()); }
        Data contiguous(source.dataType,source.dims);contiguous.Allocate(false);
        const int heads=source.dims[0],length=source.dims[1],width=source.dims[2];
        const size_t rowBytes=size_t(width)*source.unitSize;
        for(int h=0;h<heads;++h)for(int t=0;t<length;++t) {
            const size_t from=(size_t(source.pageIndex[t/source.pageLen])*source.pageLen+t%source.pageLen)*heads+h;
            std::memcpy(contiguous.cpuData+(size_t(h)*length+t)*rowBytes,pages.data()+from*rowBytes,rowBytes);
        }
        ToDataType(contiguous,result,FLOAT32);result.ToDevice(CPU);
    };
    Data x,y;materialize(a,x);materialize(b,y);
    double error = 0, energy = 0, maximum = 0;
    uint64_t count=1; for(int dim:x.dims)count*=dim;
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t remainder=i, xi=0, yi=0;
        for(int d=int(x.dims.size())-1;d>=0;--d) {
            const uint64_t c=remainder%x.dims[d];remainder/=x.dims[d];
            xi+=c*x.strides[d];yi+=c*y.strides[d];
        }
        double v = ((float*)x.cpuData)[xi], ref = ((float*)y.cpuData)[yi];
        if (!std::isfinite(v) || !std::isfinite(ref)) { std::cerr<<name<<" index="<<i<<" actual="<<v<<" expected="<<ref<<"\n"; Check(false,"nonfinite tensor"); }
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
    static bool pipelineTests = false;
struct Qwen4GGUFTestAccess {
    static void LinearUpdatePrefixes() {
        SetCudaGraph(false); FastllmCudaSetDevice(0);
        unsigned seed = 673;
        auto make = [&](DataType type, const std::vector<int> &dims, float scale) {
            size_t count = 1; for (int n : dims) count *= n;
            std::vector<float> values(count);
            for (auto &v : values) { seed = 1664525u * seed + 1013904223u; v = (int(seed >> 16) - 32768) * (scale / 32768); }
            auto result = std::make_unique<Data>(FLOAT32, dims, values);
            ToDataType(*result, type); result->ToDevice(CUDA, std::vector<int>{0}); return result;
        };
        auto prefix = [&](const Data &source, int rows) {
            Data host; ToDataType(source, host, FLOAT32); host.ToDevice(CPU);
            const int batch = source.dims[0], columns = source.dims[2];
            std::vector<float> values(size_t(batch) * rows * columns);
            for (int b = 0; b < batch; ++b)
                std::copy_n((float *)host.cpuData + size_t(b) * source.dims[1] * columns,
                    rows * columns, values.data() + size_t(b) * rows * columns);
            auto result = std::make_unique<Data>(FLOAT32, std::vector<int>{batch, rows, columns}, values);
            ToDataType(*result, source.dataType); result->ToDevice(CUDA, std::vector<int>{0}); return result;
        };
        for (auto type : {FLOAT32, FLOAT16, BFLOAT16}) for (int rows : {2, 6, 17}) {
            const int batch = rows == 6 ? 2 : 1, heads = 2, channels = 512, kernel = 4;
            auto qkv = make(FLOAT32, {batch, rows, channels}, .1f);
            auto alpha = make(type, {batch, rows, heads}, .3f), beta = make(type, {batch, rows, heads}, .3f);
            auto aLog = make(FLOAT32, {heads}, .5f), bias = make(FLOAT32, {heads}, .2f);
            auto initial = make(FLOAT32, {batch, heads, 128, 128}, .01f);
            auto next = make(FLOAT32, initial->dims, 0);
            auto core = make(FLOAT32, {batch, rows, heads * 128}, 0);
            auto updates = make(FLOAT32, {batch, rows, heads, 257}, 0);
            auto raw = make(type, {batch, rows, channels}, .4f);
            auto savedConv = make(FLOAT32, {batch, channels, kernel}, .2f);
            Data rawHost, convHost; ToDataType(*raw, rawHost, FLOAT32);rawHost.ToDevice(CPU);
            convHost.CopyFrom(*savedConv);convHost.ToDevice(CPU);
            Check(FastllmCudaQwen4GatedDeltaRuleCaptureUpdates(*qkv, *alpha, *beta, *aLog, *bias,
                *initial, *next, *core, *updates, 1, heads, 1e-6f), "update capture refused supported input");
            const int itemCount = rows == 17 ? 65 : rows == 6 ? 33 : 1;
            std::vector<Data> states(itemCount), convs(itemCount);
            std::vector<FastllmCudaQwen4LinearCommitItem> items;
            for (int i = 0; i < itemCount; ++i) {
                states[i].CopyFrom(*initial); convs[i].CopyFrom(*savedConv);
                items.push_back({raw.get(), savedConv.get(), initial.get(), updates.get(), &convs[i], &states[i]});
            }
            for (int committed = 1; committed <= rows; ++committed) {
                auto q = prefix(*qkv, committed), a = prefix(*alpha, committed), b = prefix(*beta, committed);
                Data expected;expected.CopyFrom(*initial);
                auto output = make(FLOAT32, {batch, committed, heads * 128}, 0);
                Check(FastllmCudaQwen4GatedDeltaRuleDecode(*q, *a, *b, *aLog, *bias,
                    expected, *output, 1, heads, 128, 128, 1e-6f), "independent prefix oracle failed");
                std::vector<float> expectedConv(size_t(batch) * channels * kernel);
                for (int request = 0; request < batch; ++request) for (int c = 0; c < channels; ++c)
                    for (int tap = 0; tap < kernel; ++tap) {
                        int token = committed + tap - kernel;
                        size_t index = (size_t(request) * channels + c) * kernel + tap;
                        expectedConv[index] = token < 0 ? ((float *)convHost.cpuData)[index + committed]
                            : ((float *)rawHost.cpuData)[(size_t(request) * rows + token) * channels + c];
                    }
                Check(FastllmCudaQwen4CommitLinearUpdates(items, committed), "prefix commit refused input");
                Data expectedC(FLOAT32, savedConv->dims, expectedConv);
                for (int i = 0; i < itemCount; ++i) {
                    Equal(states[i], expected, "rank-one prefix state", 1e-5);
                    Equal(convs[i], expectedC, "rank-one prefix convolution", 0);
                }
                if (committed == rows) Equal(*next, expected, "captured full state", 1e-5);
            }
            Data unchanged; unchanged.CopyFrom(states.back());
            auto invalid = items; invalid.back().updates = nullptr;
            Check(!FastllmCudaQwen4CommitLinearUpdates(invalid, 1), "invalid batch partially accepted");
            Equal(states.back(), unchanged, "invalid batch modified state", 0);
            // Capture and replay both operations without any host upload or
            // scratch pointer change. Compare against the established kernel.
            void *graph = nullptr, *exec = nullptr;
            Check(FastllmCudaGraphBeginCapture(), "update graph begin");
            Check(FastllmCudaQwen4GatedDeltaRuleCaptureUpdates(*qkv, *alpha, *beta, *aLog, *bias,
                *initial, *next, *core, *updates, 1, heads, 1e-6f), "captured update kernel");
            Check(FastllmCudaQwen4CommitLinearUpdates(items, rows), "captured commit kernel");
            Check(FastllmCudaGraphEndCapture(&graph) && FastllmCudaGraphInstantiate(graph, &exec), "update graph instantiate");
            for (int replay = 0; replay < 2; ++replay) {
                Check(FastllmCudaGraphLaunch(exec), "update graph launch");
                Equal(states.back(), *next, "captured prefix replay", 1e-5);
            }
            FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
        }
        std::cout << "PASS compact recurrent updates: FP32/FP16/BF16, every prefix of 2/6/17 rows, 1/33/65 states, batch 1/2, CUDA graph, independent original recurrence oracle\n";
    }
    static void MirrorHandoff() {
        SetCudaGraph(false);
        FastllmCudaSetDevice(0);
        std::vector<float> values(32), positions{0,1,2,3};
        std::iota(values.begin(), values.end(), 0.25f);
        Data keys(FLOAT32,{4,8},values), pos(FLOAT32,{4},positions);
        keys.ToDevice(CUDA, std::vector<int>{0}); pos.ToDevice(CUDA, std::vector<int>{0});
        Qwen4ExpModel::QsaHostMirrorTransfer mirror;
        // Allocate storage and the event before starting an unrelated stream.
        mirror.Queue(keys,pos,0,4,8);
        std::vector<float> actual, actualPositions;
        mirror.Materialize(actual,actualPositions); mirror.Rollback(0);
        struct Gate { std::atomic<bool> entered{false}, release{false}; } gate;
        void *side=FastllmCudaStreamCreate();
        Check(cudaLaunchHostFunc(static_cast<cudaStream_t>(side), [](void *opaque) {
            auto &g=*static_cast<Gate*>(opaque);g.entered.store(true);
            while(!g.release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, &gate)==cudaSuccess,"mirror independent stream callback");
        while(!gate.entered.load()) std::this_thread::yield();
        std::promise<void> completion;auto ready=completion.get_future();
        std::thread consumer([&] {
            try {
                // Three different producer/consumer workers must follow the
                // mirror's event, not the caller's per-thread stream alone.
                std::thread first([&] { FastllmCudaSetDevice(0);mirror.Queue(keys,pos,0,2,8); });first.join();
                std::thread second([&] {
                    FastllmCudaSetDevice(0);
                    Data tailKeys,tailPos;tailKeys.FakeFrom(keys,16*sizeof(float));tailKeys.Resize({2,8});
                    tailPos.FakeFrom(pos,2*sizeof(float));tailPos.Resize({2});
                    mirror.Queue(tailKeys,tailPos,2,2,8);
                });second.join();
                FastllmCudaSetDevice(0);mirror.Rollback(2);mirror.Materialize(actual,actualPositions);
                completion.set_value();
            } catch (...) { completion.set_exception(std::current_exception()); }
        });
        const bool independent=ready.wait_for(std::chrono::seconds(2))==std::future_status::ready;
        // Always release the callback, including when the old device-wide
        // synchronization would block, so a regression fails without hanging.
        gate.release.store(true);consumer.join();
        FastllmCudaStreamSynchronize(side);FastllmCudaStreamDestroy(side);ready.get();
        Check(independent,"QSA mirror rollback drained an unrelated CUDA stream");
        Check(actual==std::vector<float>(values.begin(),values.begin()+16) &&
              actualPositions==std::vector<float>({0,1}),"QSA mirror producer handoff or rollback data");
        std::cout<<"PASS QSA mirror: cross-worker append/rollback, exact pinned data, unrelated stream remains independent\n";
        for (int count : {1, 17, 65}) {
            std::vector<std::unique_ptr<Qwen4ExpModel::QsaHostMirrorTransfer>> mirrors;
            for (int i=0;i<count;++i) mirrors.emplace_back(new Qwen4ExpModel::QsaHostMirrorTransfer);
            for (int step=0;step<4;++step) {
                std::thread producer([&] {
                    FastllmCudaSetDevice(0);
                    Qwen4ExpModel::QsaHostMirrorTransfer::Batch batch;
                    for (auto &m : mirrors) {
                        m->Queue(keys,pos,step*4,2,8,&batch);
                        Data tailKeys,tailPos;
                        tailKeys.FakeFrom(keys,16*sizeof(float));tailKeys.Resize({2,8});
                        tailPos.FakeFrom(pos,2*sizeof(float));tailPos.Resize({2});
                        m->Queue(tailKeys,tailPos,step*4+2,2,8,&batch);
                    }
                }); producer.join();
                for (auto &m : mirrors) {
                    m->Materialize(actual,actualPositions);
                    for (int i=0;i<(step+1)*32;++i) Check(actual[i]==values[i%32],"batched mirror raw bytes");
                    for (int i=0;i<(step+1)*4;++i) Check(actualPositions[i]==positions[i%4],"batched mirror positions");
                }
            }
        }
        std::cout<<"PASS batched QSA mirrors: 1/17/65 histories, split appends, shared completion, fresh workers\n";

    }
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
        m.executionMode=ExecutionMode::Gpu; m.deviceMap={{"cuda:0",1}};
        if (pipelineTests) m.deviceMap["cuda:1"] = 1;
        m.moeDeviceMap=m.deviceMap;
        m.maxBatch=7; m.saveHistoryChat=false;
        unsigned seed=97;
        auto put=[&](const std::string &name, std::vector<int> dims, DataType type=FLOAT16, float constant=999.f) {
            int n=1; for(int d:dims)n*=d; std::vector<float> v(n);
            for(int i=0;i<n;++i) { seed=1664525u*seed+1013904223u; v[i]=constant==999.f ? (float(int(seed>>16)-32768)/32768)*.025f : constant; }
            Data value(FLOAT32,dims,v); ToDataType(value,type);
            // Match model loading: weights start on CPU, then each PP stage
            // places its own experts. Do not inherit the previous test's GPU.
            value.ToDevice(CPU); m.weight[name].CopyFrom(value);
            const bool attentionProjection = quantizedAttention &&
                (name.find("in_proj_qkv.") != std::string::npos || name.find("in_proj_z.") != std::string::npos ||
                 name.find(".self_attn.") != std::string::npos &&
                 name.find("proj.weight") != std::string::npos);
            if (attentionProjection || ((uniformHead || quantizedExperts) && name.find(".experts.") != std::string::npos)) {
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
    static void MtpEmbeddingPlacement() {
        SetCudaGraph(false); SetCudaEmbedding(true);
        auto model=Model(true);auto &m=*model;
        auto &weight=m.weight[m.languagePrefix+"embed_tokens.weight"];
        for (int rows : {1,3,17}) {
            std::vector<float> tokens(rows);std::iota(tokens.begin(),tokens.end(),2.f);
            Data ids(FLOAT32,{1,rows},tokens),expected,actual;
            ApplyDeviceMap({{"cuda:0",1}},1,1);FastllmCudaSetDevice(0);
            Embedding(ids,weight,expected);
            const void *storage=weight.cudaData;
            ApplyDeviceMap({{"cuda:1",1}},1,1);FastllmCudaSetDevice(1);
            ids.ToDevice(CUDA,std::vector<int>{1});
            Data borrowed;borrowed.FakeFrom(ids,0);borrowed.Resize(ids.dims);
            m.RunMtpEmbedding(borrowed,actual);
            Check(weight.cudaData==storage && weight.dataDeviceIds==std::vector<int>{0},"MTP migrated shared embedding weights");
            Check(actual.dataDeviceIds==std::vector<int>{1},"MTP embedding output on wrong device");
            Equal(actual,expected,"cross-device embedding rows",0);
        }
        std::cout<<"PASS MTP embedding ownership: 1/3/17 borrowed device tokens, exact rows, unchanged weight allocation\n";
    }
    static void ServingReuse() {
        SetCudaGraph(false); SetDeviceMap({{"cuda:0",1}}); SetMoeDeviceMap({{"cuda:0",1}});
        SetCudaEmbedding(true); SetMoeCudaCacheBytes(0); SetMaxTokens(4096);
        setenv("FASTLLM_QWEN4_ENABLE_MTP","0",1);
        auto model=Model(false,true,128), oracle=Model(false,true,128); auto &m=*model;
        m.maxBatch=4; m.max_positions=512; m.tokensLimit=4096;
        using Cache=std::vector<std::pair<Data,Data>>;
        GenerationConfig cfg; cfg.top_k=17; cfg.top_p=.95f; cfg.temperature=.8f;
        cfg.repeat_penalty=1; cfg.output_logits=true; cfg.output_token_limit=64;
        auto init=[](Cache &c) { c.resize(2); for(auto &kv:c){kv.first.isKVCache=true;kv.second.isKVCache=true;} };
        Cache warm; init(warm);
        for(int t=0;t<132;++t) m.Forward(Data(FLOAT32,{1,1},{float(t%120+2)}),Data(),Data(FLOAT32,{1,1},{float(t)}),warm,cfg);
        m.ReserveServingCache(warm); Check(bool(m.servingCache),"serving pool reservation");
        // Repeated clients, varying batch sizes and arrival/removal order must
        // retain bounded slots and isolate every history from the prior wave.
        const int counts[]={1,1,2,2,4,4,2,1};
        std::vector<const Data*> firstSlots;
        for(int wave=0;wave<8;++wave) {
            const int n=counts[wave];
            std::vector<std::unique_ptr<ResponseContext>> clients;
            std::vector<Cache> serial(n);
            std::vector<int> positions(n,132);
            for(int j=0;j<n;++j) {
                clients.emplace_back(new ResponseContext); auto &c=*clients.back();
                init(c.pastKeyValues);init(serial[j]); c.generationConfig=cfg;
                for(int t=0;t<132;++t)c.allTokens.push_back((wave*13+j*7+t)%120+2);
                SetCudaGraph(true);m.OnResponseContextCreated(&c);
                Check(m.requestStates[&c.pastKeyValues.front().first].reusableGraphRequest,"non-greedy max_batch=4 slot disabled");
                for(int t=0;t<132;++t) {
                    Data ids(FLOAT32,{1,1},{float(c.allTokens[t])}),pos(FLOAT32,{1,1},{float(t)});
                    SetCudaGraph(false);oracle->Forward(ids,Data(),pos,serial[j],cfg);
                    // Priming without graphs retains the independently evaluated
                    // recurrence; graph use begins at the same sparse context.
                    m.Forward(ids,Data(),pos,c.pastKeyValues,cfg);
                }
            }
            for(int step=0;step<20;++step) {
                if (wave==5 && step==10) {
                    // Replace one client while three unrelated histories stay
                    // live. The new client must reuse the released slot, with
                    // clean PLE/QSA/recurrent state and unchanged batch binding.
                    const auto *oldSlot=m.decodeServingSlots.at(&clients[3]->pastKeyValues.front().first).get();
                    m.OnResponseContextRemoved(clients[3].get()); oracle->ClearWarmupCache(serial[3]);
                    clients[3].reset(new ResponseContext); auto &c=*clients[3];
                    init(c.pastKeyValues);init(serial[3]);c.generationConfig=cfg;
                    for(int t=0;t<132;++t)c.allTokens.push_back((73+t*3)%120+2);
                    SetCudaGraph(true);m.OnResponseContextCreated(&c);
                    for(int t=0;t<132;++t) {
                        Data input(FLOAT32,{1,1},{float(c.allTokens[t])}),position(FLOAT32,{1,1},{float(t)});
                        SetCudaGraph(false);oracle->Forward(input,Data(),position,serial[3],cfg);
                        m.Forward(input,Data(),position,c.pastKeyValues,cfg);
                    }
                    Check(m.decodeServingSlots.at(&c.pastKeyValues.front().first).get()==oldSlot,"replacement client did not reuse its idle slot");
                    positions[3]=132;
                }
                std::vector<float> ids; std::vector<Data> pos(n); std::vector<Data*> ptrs,masks(n,nullptr);
                std::vector<Cache*> caches; std::vector<GenerationConfig> configs(n,cfg);
                std::vector<std::vector<float>> expected(n),actual(n);std::vector<std::vector<float>*> out;
                for(int j=0;j<n;++j) {
                    int slot=(step%2)?n-1-j:j; int token=(wave*9+slot*11+step)%120+2;
                    ids.push_back(token);pos[j].CopyFrom(Data(FLOAT32,{1,1},{float(positions[slot])}));
                    SetCudaGraph(false);oracle->Forward(Data(FLOAT32,{1,1},{float(token)}),Data(),pos[j],serial[slot],cfg,LastTokensManager(),&expected[j]);
                    ptrs.push_back(&pos[j]); caches.push_back(&clients[slot]->pastKeyValues);out.push_back(&actual[j]);
                }
                SetCudaGraph(true);
                m.ForwardRequests(Data(FLOAT32,{1,n},ids),masks,ptrs,std::vector<int>(n,1),caches,configs,LastTokensManager(),&out);
                for(int j=0;j<n;++j) {
                    int slot=(step%2)?n-1-j:j;
                    Equal(Data(FLOAT32,{int(actual[j].size())},actual[j]),Data(FLOAT32,{int(expected[j].size())},expected[j]),"reused slot logits");
                    // Scheduler tensors are borrowed metadata; recurrent content
                    // is still owned and reset by the persistent slot.
                    for(int l=0;l<1;++l) {Equal((*caches[j])[l].first,serial[slot][l].first,"reused conv");Equal((*caches[j])[l].second,serial[slot][l].second,"reused recurrent");}
                    ++positions[slot];
                }
            }
            for(int j=n-1;j>=0;--j) {
                if(wave==5 && j==3) clients[j]->isAbort=true;
                m.OnResponseContextRemoved(clients[j].get());oracle->ClearWarmupCache(serial[j]);
            }
            Check(m.decodeServingSlots.empty(),"live serving slot leaked");
            Check(m.idleDecodeServingSlots.size()<=4,"idle serving pool unbounded");
        }
        std::cout<<"PASS serving graph reuse: non-greedy max_batch=4, 1/2/4 clients, eight waves, live client replacement, abort cleanup, reversed row/removal order, independent logit/recurrent oracle\n";
    }
    static void ServingMtpHandoff() {
        SetDeviceMap({{"cuda:0",1}}); SetMoeDeviceMap({{"cuda:0",1}});
        SetCudaEmbedding(true); SetMoeCudaCacheBytes(0); SetMaxTokens(4096);
        using Cache=std::vector<std::pair<Data,Data>>;
        auto init=[](Cache &c) { c.resize(2); for(auto &kv:c){kv.first.isKVCache=true;kv.second.isKVCache=true;} };
        for(int drafts : {1,3,5}) {
            setenv("FASTLLM_QWEN4_ENABLE_MTP",std::to_string(drafts).c_str(),1);
            setenv("FASTLLM_QWEN4_MTP_MIN_P","0",1); SetCudaGraph(false);
            auto model=Model(true,true,128), oracle=Model(true,true,128); auto &m=*model;
            m.maxBatch=4; m.max_positions=512; m.tokensLimit=4096;
            GenerationConfig cfg; cfg.top_k=1; cfg.repeat_penalty=1;
            cfg.input_token_length=132; cfg.output_token_limit=256;
            std::vector<float> ids(132), positions(132);
            for(int i=0;i<132;++i){ids[i]=i%120+2;positions[i]=i;}
            Cache warm;init(warm);
            m.Forward(Data(FLOAT32,{1,132},ids),Data(),Data(FLOAT32,{1,132},positions),warm,cfg);
            m.ReserveServingCache(warm); Check(bool(m.servingCache),"MTP serving reservation");
            for(int wave=0;wave<2;++wave) {
                ResponseContext client;init(client.pastKeyValues);client.generationConfig=cfg;
                client.allTokens.assign(ids.begin(),ids.end());
                SetCudaGraph(true);m.OnResponseContextCreated(&client);
                Cache reference;init(reference);
                auto run=[&](const Data &input,const Data &position) {
                    int result=-1;std::exception_ptr error;
                    // A different host worker owns the next per-thread CUDA
                    // stream; state must be published before Forward returns.
                    std::thread worker([&] {try {
                        FastllmCudaSetDevice(0);
                        result=m.Forward(input,Data(),position,client.pastKeyValues,cfg);
                    }catch(...){error=std::current_exception();}});
                    worker.join();if(error)std::rethrow_exception(error);return result;
                };
                SetCudaGraph(false);
                int expected=oracle->Forward(Data(FLOAT32,{1,132},ids),Data(),Data(FLOAT32,{1,132},positions),reference,cfg);
                int actual=run(Data(FLOAT32,{1,132},ids),Data(FLOAT32,{1,132},positions));
                Check(actual==expected,"MTP serving prefill token");
                auto &bound=m.BindDecodeServingCache(client.pastKeyValues);
                auto &state=m.requestStates.at(&bound.front().first);
                Check(state.servingSlot==m.decodeServingSlots.at(&client.pastKeyValues.front().first).get(),"MTP serving handoff path unused");
                bool accepted=false,rejected=false, captured=false;int verifications=0;
                std::set<int> rejectedIndices;
                for(int step=0;step<std::max(40,drafts*24);++step) {
                    auto &ref=oracle->requestStates.at(&reference.front().first);
                    if(state.mtpState->pendingOutputTokens.empty() && ++verifications%3==0) {
                        // Uniform heads deterministically pick zero. Force an
                        // incorrect proposal to exercise recurrent/QSA rollback.
                        const int rejectedIndex=(verifications/3-1)%drafts;
                        rejectedIndices.insert(rejectedIndex);
                        state.mtpState->proposals[rejectedIndex]=47;
                        ref.mtpState->proposals[rejectedIndex]=47;rejected=true;
                    }
                    Data position(FLOAT32,{1,1},{float(132+step)});
                    SetCudaGraph(false);
                    expected=oracle->Forward(Data(FLOAT32,{1,1},{float(expected)}),Data(),position,reference,cfg);
                    SetCudaGraph(true);
                    actual=run(Data(FLOAT32,{1,1},{float(actual)}),position);
                    Check(actual==expected,"MTP handoff emitted token");
                    captured |= !state.mtpState->targetCapture.linearUpdates.empty() &&
                        state.mtpState->targetCapture.linearUpdates[0].dims.size() == 4;
                    for(int l=0;l<2;++l) {
                        Equal(bound[l].first,reference[l].first,"MTP handoff key/conv");
                        Equal(bound[l].second,reference[l].second,"MTP handoff value/recurrent");
                    }
                    Equal(*state.indexerTailKeyTensors.at(1), *ref.indexerTailKeyTensors.at(1), "MTP handoff QSA tail keys");
                    Equal(*state.indexerTailPositionTensors.at(1), *ref.indexerTailPositionTensors.at(1), "MTP handoff QSA tail positions");
                    Equal(*state.indexerBlockKeyTensors.at(1), *ref.indexerBlockKeyTensors.at(1), "MTP handoff QSA blocks");
                    Equal(state.mtpState->key,ref.mtpState->key,"MTP handoff draft key");
                    Equal(state.mtpState->value,ref.mtpState->value,"MTP handoff draft value");
                    Check(state.mtpState->pendingOutputTokens==ref.mtpState->pendingOutputTokens,"MTP handoff accepted queue");
                    accepted|=!state.mtpState->pendingOutputTokens.empty();
                }
                Check(accepted&&rejected,"MTP handoff did not cover both acceptance outcomes");
                Check(captured, "MTP handoff did not capture compact recurrent updates");
                Check(rejectedIndices.size()==size_t(drafts),"MTP handoff did not cover every rejected prefix");
                m.OnResponseContextRemoved(&client);oracle->ClearWarmupCache(reference);
            }
        }
        std::cout<<"PASS MTP serving handoff: 1/3/5 drafts, changing host workers, varying rejected prefixes, slot reuse, independent state oracle\n";
    }
    static void ServingMtpBatchHandoff() {
        SetDeviceMap({{"cuda:0",1}}); SetMoeDeviceMap({{"cuda:0",1}});
        SetCudaEmbedding(true); SetMoeCudaCacheBytes(0); SetMaxTokens(4096);
        setenv("FASTLLM_QWEN4_ENABLE_MTP","3",1); setenv("FASTLLM_QWEN4_MTP_MIN_P","0",1);
        using Cache=std::vector<std::pair<Data,Data>>;
        auto init=[](Cache &c) {c.resize(2);for(auto &kv:c){kv.first.isKVCache=true;kv.second.isKVCache=true;}};
        for(int count : {2,4}) {
            SetCudaGraph(false);
            auto model=Model(true,true,128),oracle=Model(true,true,128);auto &m=*model;
            m.maxBatch=4;m.max_positions=512;m.tokensLimit=4096;
            GenerationConfig cfg;cfg.top_k=1;cfg.repeat_penalty=1;cfg.input_token_length=132;cfg.output_token_limit=256;
            std::vector<float> ids(132),positions(132);for(int i=0;i<132;++i){ids[i]=i%120+2;positions[i]=i;}
            Cache warm;init(warm);
            m.Forward(Data(FLOAT32,{1,132},ids),Data(),Data(FLOAT32,{1,132},positions),warm,cfg);
            m.ReserveServingCache(warm);
            std::vector<std::unique_ptr<ResponseContext>> clients(count);
            std::vector<Cache> references(count);std::vector<Cache*> bound(count),caches(count);
            std::vector<int> actual(count),expected(count),verifications(count),lens(count,1);
            std::vector<GenerationConfig> configs(count,cfg);
            for(int i=0;i<count;++i) {
                clients[i].reset(new ResponseContext());auto &c=*clients[i];init(c.pastKeyValues);init(references[i]);
                c.generationConfig=cfg;c.allTokens.assign(ids.begin(),ids.end());
                SetCudaGraph(true);m.OnResponseContextCreated(&c);caches[i]=&c.pastKeyValues;
                SetCudaGraph(false);
                actual[i]=m.Forward(Data(FLOAT32,{1,132},ids),Data(),Data(FLOAT32,{1,132},positions),c.pastKeyValues,cfg);
                expected[i]=oracle->Forward(Data(FLOAT32,{1,132},ids),Data(),Data(FLOAT32,{1,132},positions),references[i],cfg);
                bound[i]=&m.BindDecodeServingCache(c.pastKeyValues);
                std::rotate(ids.begin(),ids.begin()+1,ids.end());
            }
            for(int step=0;step<96;++step) {
                std::vector<Data> pos(count);std::vector<Data*> pointers(count),masks(count,nullptr);
                std::vector<float> input(actual.begin(),actual.end());
                SetCudaGraph(false);
                for(int i=0;i<count;++i) {
                    auto &a=*m.requestStates.at(&bound[i]->front().first).mtpState;
                    auto &b=*oracle->requestStates.at(&references[i].front().first).mtpState;
                    if(a.pendingOutputTokens.empty() && ++verifications[i]%3==0) {
                        int reject=(verifications[i]/3+i)%3;a.proposals[reject]=47;b.proposals[reject]=47;
                    }
                    pos[i].CopyFrom(Data(FLOAT32,{1,1},{float(132+step)}));pointers[i]=&pos[i];
                    expected[i]=oracle->Forward(Data(FLOAT32,{1,1},{float(expected[i])}),Data(),pos[i],references[i],cfg);
                }
                SetCudaGraph(true);std::exception_ptr error;
                std::thread worker([&] {try {
                    FastllmCudaSetDevice(0);
                    actual=m.ForwardRequests(Data(FLOAT32,{1,count},input),masks,pointers,lens,caches,configs,LastTokensManager(),nullptr);
                }catch(...){error=std::current_exception();}});worker.join();if(error)std::rethrow_exception(error);
                Check(actual==expected,"serving MTP batch emitted tokens");
                for(int i=0;i<count;++i) {
                    auto &a=m.requestStates.at(&bound[i]->front().first),&b=oracle->requestStates.at(&references[i].front().first);
                    Check(bool(a.mtpState->targetCheckpoint.transfer),"serving MTP batch did not use independent state transfer");
                    for(int l=0;l<2;++l){Equal((*bound[i])[l].first,references[i][l].first,"serving batch key/conv");Equal((*bound[i])[l].second,references[i][l].second,"serving batch value/recurrent");}
                    // QSA projections use different batched/serial FP16 reductions.
                    // Keep a separate L2 bound for these small raw keys.
                    Equal(*a.indexerTailKeyTensors.at(1), *b.indexerTailKeyTensors.at(1), "serving batch QSA tail keys", 0.005);
                    Equal(*a.indexerTailPositionTensors.at(1), *b.indexerTailPositionTensors.at(1), "serving batch QSA tail positions", 0);
                    Equal(*a.indexerBlockKeyTensors.at(1), *b.indexerBlockKeyTensors.at(1), "serving batch QSA blocks", 0.005);
                    Equal(a.mtpState->key,b.mtpState->key,"serving batch draft key");Equal(a.mtpState->value,b.mtpState->value,"serving batch draft value");
                    Equal(*a.pleConvHistoryTensors[a.pleConvHistoryIndex],*b.pleConvHistoryTensors[b.pleConvHistoryIndex],"serving batch PLE history");
                    Check(a.mtpState->pendingOutputTokens==b.mtpState->pendingOutputTokens,"serving batch accepted queues");
                }
            }
            for(int i=0;i<count;++i){m.OnResponseContextRemoved(clients[i].get());oracle->ClearWarmupCache(references[i]);}
        }
        std::cout<<"PASS MTP serving batch handoff: 2/4 clients, mixed rejected prefixes, fresh host workers, independent state transfer and oracle\n";
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
int main(int argc,char **argv) { try {SetThreads(2);FastllmCudaSetDevice(0);const bool tuned=argc>1 && std::string(argv[1])=="--tuned-sparse";
if (argc>1 && std::string(argv[1])=="--embedding-only") {
    int devices=0;Check(cudaGetDeviceCount(&devices)==cudaSuccess && devices>=2,"pipeline regression requires two GPUs");
    pipelineTests=true;
    Qwen4GGUFTestAccess::MtpEmbeddingPlacement();
    return 0;
}
Qwen4GGUFTestAccess::MirrorHandoff();
if (argc>1 && std::string(argv[1])=="--mirror-only") return 0;
Qwen4GGUFTestAccess::LinearUpdatePrefixes();
if (argc>1 && std::string(argv[1])=="--mtp-serving-only") { Qwen4GGUFTestAccess::ServingMtpHandoff(); Qwen4GGUFTestAccess::ServingMtpBatchHandoff(); return 0; }
if (!(argc>1 && std::string(argv[1])=="--serving-only")) Qwen4GGUFTestAccess::Run(tuned);
Qwen4GGUFTestAccess::ServingReuse();
Qwen4GGUFTestAccess::ServingMtpHandoff();
Qwen4GGUFTestAccess::ServingMtpBatchHandoff();
quantizedAttention=true;
if (!(argc>1 && std::string(argv[1])=="--serving-only")) Qwen4GGUFTestAccess::Run(tuned);
Qwen4GGUFTestAccess::ServingReuse();
Qwen4GGUFTestAccess::ServingMtpHandoff();
Qwen4GGUFTestAccess::ServingMtpBatchHandoff();
std::cout<<"PASS GGUF attention projections with independent graph branches\n";return 0;}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;} }
