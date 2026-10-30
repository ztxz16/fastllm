// Included after the disk cache helpers. Only the ordinary GGUF disk adapter
// uses this coordinator; CPU compute still uses the configured worker pool.
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda-disk-moe-async.h"
namespace fastllm {
struct DiskMoeAsyncRuntime {
    struct Job { int layer, topk; unsigned sequence; };
    struct CpuBatch : CpuMergeMOE { using CpuMergeMOE::Run; };
    struct CpuGateTask : MultiThreadBaseOp {
        struct Part { uint8_t *input,*weight,*quantized;int hidden,inter,begin,count,type; };
        std::array<Part,16> parts{};int count=0;
        std::vector<float> scratch;
        void Run() override {
            for(int i=0;i<count;++i){
                const auto &p=parts[i];
                if(scratch.size()<size_t(2*p.inter))scratch.resize(2*p.inter);
                const auto type=static_cast<DataType>(int(DATA_GGUF_FORMAT)+p.type);
                LinearQ8K_GGUF_Kernel(p.input,p.weight,nullptr,scratch.data(),1,p.hidden,2*p.inter,
                                     p.begin,p.begin+p.count,type,type);
                LinearQ8K_GGUF_Kernel(p.input,p.weight,nullptr,scratch.data(),1,p.hidden,2*p.inter,
                                     p.inter+p.begin,p.inter+p.begin+p.count,type,type);
                MultiThreadSwigluOp activation(scratch.data()+p.begin,p.inter,p.count,
                                               scratch.data()+p.begin,1,2*p.inter,2*p.inter);
                activation.Run();
                MultiThreadFloat32ToQ8KOp quantize(scratch.data()+p.begin,p.quantized,
                                                  p.count,GGML_TYPE_Q2_0);
                quantize.Run();
            }
        }
    };
    struct CpuDownTask : MultiThreadBaseOp {
        struct Part { uint8_t *input,*weight;float *output;float score;int inter,hidden,begin,end; };
        std::array<Part,16> parts{};int count=0;
        void Run() override {
            const auto type=static_cast<DataType>(int(DATA_GGUF_FORMAT)+int(GGML_TYPE_Q2_0));
            for(int i=0;i<count;++i){
                const auto &p=parts[i];
                LinearQ8K_GGUF_Kernel(p.input,p.weight,nullptr,p.output,1,p.inter,p.hidden,p.begin,p.end,type,type);
                for(int c=p.begin;c<p.end;++c)p.output[c]*=p.score;
            }
        }
    };
    std::vector<CpuGateTask> cpuGateTasks;
    std::vector<CpuDownTask> cpuDownTasks;
    std::vector<uint8_t> cpuGateQ8,cpuDownQ8;

    DiskExpertCache &cache;
    int device,hidden;
    std::vector<std::vector<Data *>> layers;
    std::unordered_set<const Data *> keys;
    std::unordered_multimap<DiskExpertEntry *, const uint8_t **> recordSlots;
    std::unique_ptr<FastllmCudaDiskMoeAsync> mailbox;
    std::thread coordinator;
    std::mutex mutex;
    std::condition_variable wake,changed;
    std::deque<Job> jobs;
    struct Candidate {
        const Data *key = nullptr;
        std::unique_ptr<DiskCachedExpert> staged;
    };
    std::vector<Candidate> candidates;
    uint64_t stagedBytes = 0;
    bool stopping=false,start=false,finishing=false,ready=false,running=false;
    bool retainPrefillWorkspace=false;
    std::atomic<bool> cancelled{false};
    std::exception_ptr error;
    unsigned sequence=0;
    uint64_t snapshotGeneration=std::numeric_limits<uint64_t>::max();
    explicit DiskMoeAsyncRuntime(DiskExpertCache &cache,int device,
            const std::vector<std::vector<Data *>> &layers)
        :cache(cache),device(device),hidden(layers[0][2]->dims[1]),layers(layers),candidates(layers.size()) {
        std::vector<FastllmCudaDiskMoeAsyncLayout> layout;
        for(const auto &table:layers) {
            const Data &g=*table[2],&d=*table[3];
            layout.push_back({g.ggmlType,d.ggmlType,hidden,d.dims[1],int(table.size()/2-1),g.GetBytes(),g.GetBytes()+d.GetBytes()});
            for(size_t i=2;i<table.size();++i)keys.insert(table[i]);
        }
        mailbox=FastllmCudaDiskMoeAsync::Create(layout,device);
        if(mailbox){
            recordSlots.reserve(keys.size()/2);
            for(size_t l=0;l<layers.size();++l){
                auto **records=mailbox->Records(l);
                for(size_t i=2;i<layers[l].size();i+=2)
                    recordSlots.emplace(&cache.entries[layers[l][i]],records+i/2-1);
            }
            coordinator=std::thread([this]{Loop();});
        }
    }
    ~DiskMoeAsyncRuntime(){
        {std::lock_guard<std::mutex> guard(mutex);stopping=true;wake.notify_one();}
        if(coordinator.joinable())coordinator.join();
    }
    void Begin(bool retainWorkspace){
        std::unique_lock<std::mutex> lock(mutex);
        AssertInFastLLM(!running,"Disk async decode request is already active.\n");
        error=nullptr;cancelled=false;finishing=false;ready=false;running=true;start=true;
        retainPrefillWorkspace=retainWorkspace;
        mailbox->ResetFailure();
        wake.notify_one();changed.wait(lock,[&]{return ready;});
        if(error)std::rethrow_exception(error);
    }
    unsigned Queue(int layer,int topk){
        std::lock_guard<std::mutex> guard(mutex);
        const unsigned current=++sequence;
        jobs.push_back({layer,topk,current});wake.notify_one();return current;
    }
    void End(bool cancel){
        std::unique_lock<std::mutex> lock(mutex);
        cancelled=cancel;finishing=true;wake.notify_one();
        changed.wait(lock,[&]{return !running;});
        if(error)std::rethrow_exception(error);
    }
    template<class Group>
    bool RunFusedCpu(const std::vector<Group *> &selected,const Job &job,
                     const FastllmCudaDiskMoeAsync::HostRequest &request){
        if(selected.empty())return true;
        const Data &gate=*layers[job.layer][2],&down=*layers[job.layer][3];
        const int inter=down.dims[1];
        // Q2_0 consumes independent Q8_0 blocks of 32. This lets each worker
        // finish its gate/up pairs, activation and quantization locally.
        if(down.ggmlType!=GGML_TYPE_Q2_0 || inter%32 || hidden%8)return false;
        std::array<Data *,16> gates{},downs{};
        std::array<int,16> destination{};
        std::array<float,16> scores{};
        int routes=0;uint64_t bytes=0;
        for(auto *g:selected){
            auto *host=g->loaded?g->loaded.get():g->entry->cpu.get();
            if(!host || host->gate->dataType!=DATA_GGUF_FORMAT || host->down->dataType!=DATA_GGUF_FORMAT ||
               host->gate->IsRepacked || host->down->IsRepacked ||
               host->gate->ggmlType!=gate.ggmlType || host->down->ggmlType!=down.ggmlType ||
               host->gate->dims!=gate.dims || host->down->dims!=down.dims)return false;
            for(int r:g->routes){
                if(routes==16)return false;
                gates[routes]=host->gate.get();downs[routes]=host->down.get();
                destination[routes]=r;scores[routes]=request.scores[r];++routes;
            }
            bytes+=g->routes.size()*(host->gate->GetBytes()+host->down->GetBytes());
        }
        auto *pool=GetAlivePool();const int threads=pool->threads.size();
        if(!threads)return false;
        const auto gateType=static_cast<ggml_type>(gate.ggmlType);
        const size_t gateBytes=ggml_row_size(ggml_type_vec_dot_type(gateType),hidden);
        const size_t downBytes=ggml_row_size(GGML_TYPE_Q8_0,inter),blockBytes=ggml_type_size(GGML_TYPE_Q8_0);
        if(cpuGateTasks.size()<size_t(threads))cpuGateTasks.resize(threads);
        if(cpuDownTasks.size()<size_t(threads))cpuDownTasks.resize(threads);
        if(cpuGateQ8.size()<gateBytes)cpuGateQ8.resize(gateBytes);
        if(cpuDownQ8.size()<routes*downBytes)cpuDownQ8.resize(routes*downBytes);
        for(int r=0;r<routes;++r)gates[r]->disableGGUFRepack=downs[r]->disableGGUFRepack=true;
        const auto started=std::chrono::steady_clock::now();
        iqk_quantize_row_q8_K(request.input,cpuGateQ8.data(),hidden,ggml_type_vec_dot_type(gateType),gateType);
        const int gateBlocksPerRoute=inter/32,gateBlocks=routes*gateBlocksPerRoute;
        for(int tid=0;tid<threads;++tid){
            auto &task=cpuGateTasks[tid];task.count=0;
            const int limit=int(int64_t(tid+1)*gateBlocks/threads);
            for(int at=int(int64_t(tid)*gateBlocks/threads);at<limit;){
                const int r=at/gateBlocksPerRoute,begin=at%gateBlocksPerRoute;
                const int count=std::min(limit-at,gateBlocksPerRoute-begin);
                task.parts[task.count++]={cpuGateQ8.data(),gates[r]->cpuData,
                    cpuDownQ8.data()+r*downBytes+begin*blockBytes,hidden,inter,begin*32,count*32,gate.ggmlType};
                at+=count;
            }
        }
        for(int tid=0;tid<threads;++tid)pool->PushOp(tid,&cpuGateTasks[tid]);
        for(int tid=0;tid<threads;++tid)pool->Wait(tid);
        const int downBlocksPerRoute=hidden/8,downBlocks=routes*downBlocksPerRoute;
        for(int tid=0;tid<threads;++tid){
            auto &task=cpuDownTasks[tid];task.count=0;
            const int limit=int(int64_t(tid+1)*downBlocks/threads);
            for(int at=int(int64_t(tid)*downBlocks/threads);at<limit;){
                const int r=at/downBlocksPerRoute,begin=at%downBlocksPerRoute;
                const int count=std::min(limit-at,downBlocksPerRoute-begin);
                task.parts[task.count++]={cpuDownQ8.data()+r*downBytes,downs[r]->cpuData,
                    request.output+size_t(destination[r])*hidden,scores[r],inter,hidden,begin*8,(begin+count)*8};
                at+=count;
            }
        }
        for(int tid=0;tid<threads;++tid)pool->PushOp(tid,&cpuDownTasks[tid]);
        for(int tid=0;tid<threads;++tid)pool->Wait(tid);
        const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        if(bytes)cache.canonicalCpuUsPerByte.Observe(us/bytes);
        return true;
    }
    void Compute(const Job &job){
        while(mailbox->Published(job.layer)!=job.sequence && !cancelled.load(std::memory_order_relaxed))
            std::this_thread::yield();
        auto request=mailbox->Request(job.layer);
        if(cancelled){std::fill_n(request.output,size_t(job.topk)*hidden,0.f);return;}
        struct Group {
            int id;DiskExpertEntry *entry;
            std::vector<int> routes;
            std::unique_ptr<DiskCachedExpert> loaded;
            std::future<std::unique_ptr<DiskCachedExpert>> read;
        };
        std::vector<Group> groups;
        std::vector<int> order(job.topk);
        for(int r=0;r<job.topk;++r)order[r]=r;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return request.ids[a]<request.ids[b];});
        const auto &table=layers[job.layer];
        for(int r:order){
            const int id=request.ids[r];
            if(groups.empty() || groups.back().id!=id)groups.push_back({id,&cache.entries[table[2+2*id]],{},nullptr,{}});
            groups.back().routes.push_back(r);
        }
        struct WaitReads {
            std::vector<Group> &groups;
            ~WaitReads(){for(auto &g:groups)if(g.read.valid())g.read.wait();}
        }wait{groups};
        std::vector<Group *> hits,misses;
        for(auto &g:groups){
            cache.Touch(*g.entry,g.routes.size(),1);
            if(request.gpu[g.routes[0]]){cache.stats.cudaHits+=g.routes.size();continue;}
            if(g.entry->cpu){cache.stats.cpuHits+=g.routes.size();hits.push_back(&g);}
            else{
                cache.stats.misses+=g.routes.size();
                for(const auto *w:{table[2+2*g.id],table[3+2*g.id]})
                    for(const auto &part:w->diskWeightParts)cache.stats.diskBytes+=part.bytes;
                g.read=cache.prefetch.Read(table[2+2*g.id],table[3+2*g.id],false,2);misses.push_back(&g);
            }
        }
        auto batch=[&](const std::vector<Group *> &selected){
            if(selected.empty())return;
            if(RunFusedCpu(selected,job,request))return;
            std::vector<Data *> weights{nullptr,nullptr};
            std::vector<int32_t> ids;
            std::vector<float> scores;
            std::vector<int> destinations;
            uint64_t bytes=0;
            for(auto *g:selected){
                auto *host=g->loaded?g->loaded.get():g->entry->cpu.get();
                host->gate->disableGGUFRepack=host->down->disableGGUFRepack=true;
                const int local=weights.size()/2-1;
                weights.push_back(host->gate.get());weights.push_back(host->down.get());
                for(int r:g->routes){ids.push_back(local);scores.push_back(request.scores[r]);destinations.push_back(r);}
                bytes+=g->routes.size()*(host->gate->GetBytes()+host->down->GetBytes());
            }
            Data x(FLOAT32,{1,hidden},CPU,const_cast<float *>(request.input)),index(INT32PARAM,{1,int(ids.size())},CPU,ids.data());
            Data score(FLOAT32,{1,int(ids.size())},CPU,scores.data()),output(FLOAT32,{int(ids.size()),hidden}),a,b,c;
            std::vector<Data *> biases(weights.size(),nullptr);
            const auto started=std::chrono::steady_clock::now();
            CpuBatch cpu;
            cpu.Run("MergeMOE",{{"input",&x},{"output",&output},{"index",&index},{"score",&score},
                {"weights",reinterpret_cast<Data *>(weights.data())},{"biass",reinterpret_cast<Data *>(biases.data())},
                {"w1",&a},{"w2",&b},{"w3",&c}}, {},
                {{"weights___batch",int(weights.size())},{"biass___batch",int(weights.size())},
                 {"diskMoeDecode",1},{"diskMoePerRoute",1},{"gateType",int(MoeGateSwiglu)}});
            const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
            if(bytes)cache.canonicalCpuUsPerByte.Observe(us/bytes);
            for(size_t j=0;j<destinations.size();++j)
                memcpy(request.output+size_t(destinations[j])*hidden,output.cpuData+j*hidden*4,hidden*4);
        };
        batch(hits);
        if(!misses.empty()){
            for(auto *g:misses)g->loaded=g->read.get();
            batch(misses);
            for(auto *g:misses)if(cache.MakeRoom(g->entry,-1,g->loaded->Bytes(),GetMoeCpuCacheBytes()))
                cache.AdmitCpu(*g->entry,std::move(g->loaded));
        }
        auto &candidate=candidates[job.layer];
        for (auto &g : groups) {
            if (request.gpu[g.routes[0]] || g.entry->visits < 2 ||
                (!g.entry->cpu && !g.loaded) ||
                (candidate.key && g.entry->Heat(cache.Epoch()) <=
                    cache.entries.at(candidate.key).Heat(cache.Epoch()))) continue;
            const uint64_t previous = candidate.staged ? candidate.staged->Bytes() : 0;
            const uint64_t bytes = g.entry->cpu ? 0 : g.loaded->Bytes();
            if (bytes) {
                const uint64_t payload = g.loaded->gate->GetBytes() + g.loaded->down->GetBytes();
                if (payload > GetMoeCudaCacheBytes()) continue;
                // Sources survive only until this step's GPU readers finish.
                // Keep at most one per layer and 128 MiB across the step, or
                // a single larger expert. This is not a persistent DDR tier.
                const uint64_t other = stagedBytes - previous;
                constexpr uint64_t limit = 128ULL << 20;
                if (other && (bytes > limit || other > limit - bytes)) continue;
            }
            stagedBytes = stagedBytes - previous + bytes;
            candidate.staged = g.entry->cpu ? nullptr : std::move(g.loaded);
            candidate.key = table[2+2*g.id];
        }
    }
    void AdmitCandidates(){
        for(auto &candidate:candidates)if(candidate.key){
            auto &entry=cache.entries.at(candidate.key);
            auto *source=entry.cpu?entry.cpu.get():candidate.staged.get();
            if(!source || entry.Cuda(device) || source->gate->IsRepacked || source->down->IsRepacked)continue;
            const uint64_t bytes=source->gate->GetBytes()+source->down->GetBytes();
            uint64_t budget=GetMoeCudaCacheBytes();
            if(!DiskCudaRecordAvailable(bytes,device,cache.clock))budget=std::min(budget,cache.cudaBytes[device]+bytes-1);
            if(!cache.CanonicalReplacementPays(entry,device,bytes,budget) ||
               !cache.MakeRoom(&entry,device,bytes,budget,nullptr,false))continue;
            const bool fromRam=bool(entry.cpu);
            auto promotion=fromRam?cache.TakeCpu(entry):std::move(candidate.staged);
            if(!fromRam)stagedBytes-=promotion->Bytes();
            promotion->packedCuda=true;
            std::unique_ptr<DiskCachedExpert> recycled;
            bool uploaded=false;
            if(cache.MakeRoom(&entry,device,bytes,budget,&recycled)){
                cache.retiredCpuBytes+=UploadDiskExpert(*promotion,device,uploaded,std::move(recycled),&entry);
            }
            cache.FinishPromotion(entry,promotion,device,uploaded);
        }
        FastllmCudaSyncCurrentThreadStream();
        AssertInFastLLM(!FastllmCudaGetThreadError(),"Disk asynchronous cache maintenance failed.\n");
    }
    void ClearCandidates() {
        for (auto &candidate : candidates) {
            if (candidate.staged) {
                cache.retiredCpuBytes += candidate.staged->Bytes() - candidate.staged->RecycleCpuStorage();
                candidate.staged.reset();
            }
            candidate.key = nullptr;
        }
        stagedBytes = 0;
        cache.ReclaimCpu();
    }
    void Loop(){
        FastllmCudaSetDevice(device);
        std::unique_lock<std::mutex> lock(mutex);
        while(true){
            wake.wait(lock,[&]{return stopping||start;});
            if(stopping)return;
            start=false;lock.unlock();
            {
                auto cacheGuard=cache.Lock();
                // Streamed prefill has synchronized its consumers before
                // returning. Decode owns separate scratch; retaining this
                // O(prompt * topk * hidden) allocation can starve dense graphs.
                // Warmup keeps it so KV planning still measures the complete
                // prefill working set and does not hide idle pooled memory.
                if (!retainPrefillWorkspace) cache.ggufWorkspace.erase(device);
                // Residency is leased until GPU completion. Apply only the
                // recorded changes between steps; after a bounded log wraps,
                // rebuild from the still-owned entries. Foreign/stale pointers
                // are compared as keys and never dereferenced.
                const auto refresh=[&](DiskExpertEntry *entry,const uint8_t **slot){
                    auto *gpu=entry->Cuda(device);
                    *slot=gpu?static_cast<const uint8_t *>(gpu->cudaRecord):nullptr;
                };
                if(snapshotGeneration!=cache.cudaGeneration){
                    if(snapshotGeneration==std::numeric_limits<uint64_t>::max() ||
                       cache.cudaGeneration-snapshotGeneration>cache.cudaChanges.size()){
                        for(const auto &item:recordSlots)refresh(item.first,item.second);
                    }else{
                        for(auto generation=snapshotGeneration+1;generation<=cache.cudaGeneration;++generation){
                            auto *entry=cache.cudaChanges[generation%cache.cudaChanges.size()];
                            const auto slots=recordSlots.equal_range(entry);
                            for(auto it=slots.first;it!=slots.second;++it)refresh(it->first,it->second);
                        }
                    }
                    snapshotGeneration=cache.cudaGeneration;
                }
                lock.lock();ready=true;changed.notify_all();
                while(true){
                    wake.wait(lock,[&]{return finishing||!jobs.empty();});
                    if(jobs.empty()&&finishing)break;
                    const auto job=jobs.front();jobs.pop_front();lock.unlock();
                    try{Compute(job);}catch(...){
                        if(!error)error=std::current_exception();
                        auto request=mailbox->Request(job.layer);
                        std::fill_n(request.output,size_t(job.topk)*hidden,0.f);
                    }
                    mailbox->Complete(job.layer,job.sequence);
                    lock.lock();
                }
                lock.unlock();
                if(!cancelled && !error)try{AdmitCandidates();}catch(...){error=std::current_exception();}
                ClearCandidates();
            }
            lock.lock();running=false;changed.notify_all();
        }
    }
};
static thread_local DiskMoeAsyncScope *activeDiskMoeAsync=nullptr;
struct DiskMoeAsyncScope::Impl {
    std::shared_ptr<DiskMoeAsyncRuntime> runtime;
    // Mailbox slots contain only disk layers; model layer IDs may have gaps.
    std::vector<int> layerMap;
    bool posted=false;
    bool active=false;
};
static void ReleaseDiskMoeAsync(const Data *weight){
    auto &runtimes=GetDiskExpertCache().asyncRuntimes;
    for(auto it=runtimes.begin();it!=runtimes.end();)
        if(!it->second || it->second->keys.count(weight))it=runtimes.erase(it);else ++it;
}
DiskMoeAsyncScope::DiskMoeAsyncScope(const std::vector<std::vector<Data *>> &layers,int device,
                                   bool enabled,bool retainPrefillWorkspace){
    if(!enabled || activeDiskMoeAsync || !GetMoeCudaCacheBytes())return;
    const auto indices=DiskMoeLayerIndices(layers,device);
    if(indices.empty())return;
    auto state=std::make_unique<Impl>();
    state->layerMap.assign(layers.size(),-1);
    std::vector<std::vector<Data *>> diskLayers;
    diskLayers.reserve(indices.size());
    for(int layer:indices){
        state->layerMap[layer]=int(diskLayers.size());
        diskLayers.push_back(layers[layer]);
    }
    {
        auto &cache=GetDiskExpertCache();auto guard=cache.Lock();
        const auto found=cache.asyncRuntimes.find(diskLayers[0][2]);
        auto runtime=found==cache.asyncRuntimes.end()?std::shared_ptr<DiskMoeAsyncRuntime>():found->second;
        if(!runtime){
            for(const auto &table:diskLayers){
                if(table.size()<4 || table.size()%2 || table[0] || table[1])return;
                const Data *g=table[2],*d=table[3];
                if(!g || !d)return;
                for(size_t i=2;i<table.size();i+=2)
                    if(!table[i] || !table[i+1] || !table[i]->isDiskWeight || !table[i+1]->isDiskWeight ||
                       !DiskOrdinaryGGUFPair(*table[i],*table[i+1]) ||
                       table[i]->dims!=g->dims || table[i+1]->dims!=d->dims ||
                       table[i]->ggmlType!=g->ggmlType || table[i+1]->ggmlType!=d->ggmlType)return;
            }
            runtime=std::make_shared<DiskMoeAsyncRuntime>(cache,device,diskLayers);
            if(!runtime->mailbox)return;
            cache.asyncRuntimes.emplace(diskLayers[0][2],runtime);
        }
        if(runtime->device!=device || runtime->layers!=diskLayers)return;
        state->runtime=runtime;
    }
    state->runtime->Begin(retainPrefillWorkspace);state->active=true;
    impl=std::move(state);activeDiskMoeAsync=this;
}
DiskMoeAsyncScope::~DiskMoeAsyncScope(){try{Finish();}catch(...) {}}
void DiskMoeAsyncScope::Finish(){
    if(!impl || !impl->active)return;
    if(impl->posted)FastllmCudaSyncCurrentThreadStream();
    const bool gpuFailure=impl->runtime->mailbox->Failed();
    const bool failed=FastllmCudaGetThreadError() || gpuFailure || std::uncaught_exceptions()>0;
    activeDiskMoeAsync=nullptr;impl->active=false;
    impl->runtime->End(failed);
    AssertInFastLLM(!gpuFailure,"Disk asynchronous CPU completion timed out.\n");
}
bool DiskMoeAsyncScope::TryRun(int layer,const Data &input,const Data &index,const Data &score,Data &output,
                             const std::function<void()> &shared){
    auto *scope=activeDiskMoeAsync;
    if(!scope || !scope->impl || !scope->impl->active)return false;
    auto &r=*scope->impl->runtime;
    if(layer<0 || size_t(layer)>=scope->impl->layerMap.size()){
        scope->Finish();return false;
    }
    layer=scope->impl->layerMap[layer];
    // A resident layer is handled by its ordinary CUDA implementation. It
    // must not end the lease needed by disk layers later in the same step.
    if(layer<0)return false;
    if(FastllmCudaGraphIsCapturingFast() || !r.mailbox->CanRun(layer,input,index,score,output)){
        scope->Finish();return false;
    }
    const unsigned seq=r.Queue(layer,index.dims[1]);scope->impl->posted=true;
    return r.mailbox->Run(layer,input,index,score,output,seq,shared);
}
}
#else
namespace fastllm {
struct DiskMoeAsyncScope::Impl {};
static void ReleaseDiskMoeAsync(const Data *){}
DiskMoeAsyncScope::DiskMoeAsyncScope(const std::vector<std::vector<Data *>> &,int,bool,bool){}
DiskMoeAsyncScope::~DiskMoeAsyncScope()=default;
void DiskMoeAsyncScope::Finish(){}
bool DiskMoeAsyncScope::TryRun(int,const Data &,const Data &,const Data &,Data &,const std::function<void()> &){return false;}
}
#endif
