// nvcc -std=c++17 -Iinclude/devices/cuda test/basic/test_cuda_expert_cache.cu -o cache_test
#include "fastllm-cuda-expert-cache.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T *data = nullptr;
    size_t count;
    explicit Buffer(size_t n): count(n) { Check(cudaMalloc(&data, n * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    void clear(int value = 0) { Check(cudaMemset(data, value, count * sizeof(T))); }
    void put(const std::vector<T> &v) { Check(cudaMemcpy(data, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get() { std::vector<T> v(count); Check(cudaMemcpy(v.data(), data, count*sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};
struct Request { int base; std::vector<int> ids; };
struct Reference {
    std::vector<int> owner, mapping;
    std::vector<unsigned long long> ages;
    std::set<std::pair<unsigned long long,int>> order;
    unsigned long long tick, hits=0, misses=0;
    Reference(int slots, int records, unsigned long long start):
        owner(slots,-1), mapping(records,-1), ages(slots,0), tick(start) {
        for (int s=0;s<slots;++s) order.emplace(0,s);
    }
    void touch(int s) { order.erase({ages[s],s}); ages[s]=tick; order.emplace(tick,s); }
    void run(const Request &r, int experts, std::vector<int> &routes,
             std::vector<int> &missingIds, std::vector<int> &missingSlots) {
        ++tick;
        std::set<int> active;
        int valid=0;
        for (int id:r.ids) if(id>=0 && id<experts) { active.insert(r.base+id); ++valid; }
        std::vector<int> need;
        for (int key:active) {
            if (mapping[key]>=0) touch(mapping[key]);
            else need.push_back(key);
        }
        for (int key:need) {
            auto victim=*order.begin();
            if (victim.first==tick) throw std::runtime_error("no evictable slot");
            int s=victim.second;
            if(owner[s]>=0) mapping[owner[s]]=-1;
            owner[s]=key; mapping[key]=s; touch(s);
            missingIds.push_back(key-r.base); missingSlots.push_back(s);
        }
        hits+=valid-need.size(); misses+=need.size();
        for(int id:r.ids) routes.push_back(id>=0 && id<experts ? mapping[r.base+id] : -1);
    }
};
static void Run(const std::vector<Request> &requests, int experts, int records,
                int slots, int threads, bool graph, int passes=2) {
    int k=requests[0].ids.size(); size_t n=requests.size();
    Buffer<int> map(records), owner(slots), input(n*k), routes(n*k), ids(n*k), dest(n*k), missing(n);
    Buffer<unsigned long long> ages(slots), tick(1), hits(1), misses(1);
    map.clear(255); owner.clear(255); ages.clear(); hits.clear(); misses.clear();
    const unsigned long long initial=(1ULL<<32)-20;
    tick.put({initial});
    std::vector<int> flat; for(auto &r:requests) flat.insert(flat.end(),r.ids.begin(),r.ids.end()); input.put(flat);
    fastllm::cuda::ExpertCacheView view{map.data,owner.data,ages.data,tick.data,hits.data,misses.data,slots};
    cudaStream_t stream; Check(cudaStreamCreate(&stream));
    auto launch=[&] {
        for(size_t i=0;i<n;++i) {
            bool ok = k<=16 ? fastllm::cuda::EnsureExpertCache<16>(view,input.data+i*k,requests[i].base,experts,k,
                    routes.data+i*k,ids.data+i*k,dest.data+i*k,missing.data+i,threads,stream)
                : fastllm::cuda::EnsureExpertCache<128>(view,input.data+i*k,requests[i].base,experts,k,
                    routes.data+i*k,ids.data+i*k,dest.data+i*k,missing.data+i,threads,stream);
            if(!ok)
                throw std::runtime_error("dispatch rejected valid shape");
        }
    };
    cudaGraph_t g=nullptr; cudaGraphExec_t exec=nullptr;
    if(graph) { Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal)); launch();
        Check(cudaStreamEndCapture(stream,&g)); Check(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0)); }
    Reference ref(slots,records,initial);
    cudaEvent_t begin,end;Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
    for(int pass=0;pass<passes;++pass) {
        Check(cudaEventRecord(begin,stream));
        if(graph) Check(cudaGraphLaunch(exec,stream)); else launch();
        Check(cudaEventRecord(end,stream));
        Check(cudaStreamSynchronize(stream));
        float elapsed=0;Check(cudaEventElapsedTime(&elapsed,begin,end));
        std::printf("TIME slots=%d threads=%d graph=%d us/call=%.6f\n",slots,threads,graph,elapsed*1000/n);
        auto actualRoutes=routes.get(), actualIds=ids.get(), actualDest=dest.get(), actualMissing=missing.get();
        for(size_t i=0;i<n;++i) {
            std::vector<int> a,b,c; ref.run(requests[i],experts,a,b,c);
            if(actualMissing[i] != int(b.size()) ||
               !std::equal(a.begin(),a.end(),actualRoutes.begin()+i*k) ||
               !std::equal(b.begin(),b.end(),actualIds.begin()+i*k) ||
               !std::equal(c.begin(),c.end(),actualDest.begin()+i*k)) {
                std::fprintf(stderr,"mismatch slots=%d threads=%d pass=%d step=%zu missing=%d expected=%zu\n",
                             slots,threads,pass,i,actualMissing[i],b.size());
                throw std::runtime_error("LRU request mismatch");
            }
        }
        if(map.get()!=ref.mapping || owner.get()!=ref.owner || ages.get()!=ref.ages ||
           tick.get()[0]!=ref.tick || hits.get()[0]!=ref.hits || misses.get()[0]!=ref.misses)
            throw std::runtime_error("LRU state mismatch");
    }
    Check(cudaEventDestroy(begin));Check(cudaEventDestroy(end));
    if(exec) Check(cudaGraphExecDestroy(exec)); if(g) Check(cudaGraphDestroy(g)); Check(cudaStreamDestroy(stream));
    std::printf("PASS slots=%d experts=%d topk=%d threads=%d graph=%d steps=%zu passes=%d\n",slots,experts,k,threads,graph,n,passes);
}
int main(int argc,char **argv) {
    try {
        if(argc>=3) {
            std::ifstream file(argv[1],std::ios::binary); int row[16]; std::vector<Request> requests;
            while(file.read(reinterpret_cast<char *>(row),sizeof(row))) {
                if(row[1]!=10) throw std::runtime_error("unexpected trace topk");
                requests.push_back({row[0]*512,std::vector<int>(row+3,row+13)});
            }
            if(requests.empty()) throw std::runtime_error("empty trace");
            int slots=std::atoi(argv[2]);
            int threads=argc>=4 ? std::atoi(argv[3]) : fastllm::cuda::ExpertCacheThreads(slots,1024);
            Run(requests,512,48*512,slots,threads,true,1);
        } else {
            std::mt19937 rng(93751);
            struct Shape {int slots,experts,layers,k,steps;};
            for(auto s:std::vector<Shape>{{3,7,3,1,100},{17,37,5,8,200},{128,129,3,16,250},
                {1165,512,7,10,400},{1941,512,7,10,500},{3883,1025,7,32,500},
                {4097,1025,7,64,180},{16385,4099,5,64,300},{128,257,2,97,40}}) {
                if (argc == 2 && std::strcmp(argv[1], "--quick") == 0)
                    s.steps = std::min(s.steps, 24);
                std::vector<Request> requests;
                for(int i=0;i<s.steps;++i) {
                    Request r{int(rng()%s.layers)*s.experts,{}};
                    for(int q=0;q<s.k;++q) {
                        int id=rng()%s.experts;
                        if(i%7==0 && q>0) id=r.ids[0];
                        if(i%11==0) id=q%2 ? -1 : s.experts+3;
                        r.ids.push_back(id);
                    }
                    if(i%5==0 && !requests.empty()) r=requests.back();
                    requests.push_back(r);
                }
                int threads=fastllm::cuda::ExpertCacheThreads(s.slots,1024);
                Run(requests,s.experts,s.experts*s.layers,s.slots,threads,true);
            }
            std::vector<Request> small{{0,{0,1,1,-1}},{17,{2,3,4,5}},{0,{0,4,5,6}}};
            for(int threads:{32,64,128,256}) Run(small,17,34,17,threads,false);
            fastllm::cuda::ExpertCacheView empty{};
            if(fastllm::cuda::EnsureExpertCache(empty,nullptr,0,1,0,nullptr,nullptr,nullptr,nullptr,128,nullptr))
                throw std::runtime_error("invalid query count accepted");
        }
        Check(cudaDeviceSynchronize()); std::puts("ALL_PASS"); return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
