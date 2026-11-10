// Fastllm adapter for ExLlamaV3-derived EXL3 kernels, imported via Project Maya.
// Original format/tile decoding: https://github.com/turboderp-org/exllamav3 (MIT).
// Provenance, licenses and local upstream changes: third_party/maya_exl3/NOTICE.txt.
#include "exl3.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "strata/kernels/exl3.hpp"
#include "utils/kernel_registry.h"
#include "../moe/fastllm-moe-route-groups.cuh"
#include <climits>
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <set>
#include <thread>

// Selection is shared by ordinary, grouped and fused-input projections.
int FastllmCudaExl3SelectProjection(const fastllm::Data &, const fastllm::Data &,
                                  const fastllm::Data &, fastllm::Data &, bool, bool);
bool FastllmCudaExl3ProjectionFallback(const fastllm::Data &, const fastllm::Data &,
                                     const fastllm::Data &, fastllm::Data &, bool);
bool FastllmCudaExl3MayaLinear(const fastllm::Data &, fastllm::Data &,
                              const fastllm::Data &, fastllm::Data &);

namespace {
namespace maya = strata::kernels::exl3;
using namespace fastllm;
bool Enabled() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    return true;
#endif
}
struct Workspace {
    Data plan{INT8}, h{FLOAT32}, up{FLOAT32}, part{FLOAT32}, groups{INT8}, scores{FLOAT32}, counters{INT32};
    explicit Workspace(int dev) {
        for (Data *d : {&plan, &h, &up, &part, &groups, &scores, &counters}) { d->dataDevice = DataDevice::CUDA; d->dataDeviceIds = {dev}; }
    }
};
Workspace &Work(int k, int n, int rows, int topk = 0) {
    static thread_local std::map<std::tuple<int,int,int,int,int>, std::unique_ptr<Workspace>> map;
    const int dev = FastllmCudaGetDevice();
    auto &p = map[{dev,k,n,rows,topk}];
    if (!p) p.reset(new Workspace(dev));
    return *p;
}
maya::Mat Matrix(const Data &weight, int part) {
    maya::Mat m;
    m.k = weight.dims[1]; m.n = weight.dims[0] / weight.blockM; m.k2 = weight.blockK;
    const auto *base = static_cast<const uint8_t *>(weight.cudaData) + part * Exl3PartBytes(m.n,m.k,m.k2);
    m.trellis = reinterpret_cast<const uint32_t *>(base);
    m.suh = reinterpret_cast<const uint16_t *>(base + maya::trellis_bytes_k2(m.k,m.n,m.k2));
    m.svh = m.suh + m.k;
    return m;
}
struct Signature {
    const void *gate, *down;
    int kg, kd;
    bool operator==(const Signature &b) const { return gate==b.gate && down==b.down && kg==b.kg && kd==b.kd; }
};
struct Directory {
    std::vector<Signature> signature;
    std::vector<const Data *> sources;
    std::set<std::tuple<std::thread::id, int, int>> warmedShapes;
    Data table{INT8}; // [expert descriptors: 3 pointers each][expert-id -> descriptor pointer]
    maya::ExpertLayout lay;
    maya::MixedLayouts mix;
    const unsigned long long *lookup = nullptr;
    int k = 0, mid = 0;
};
using Key = std::pair<const Data *,int>;
std::mutex &DirectoryMutex() { static auto *p = new std::mutex; return *p; }
std::map<Key,std::shared_ptr<Directory>> &Directories() { static auto *p = new std::map<Key,std::shared_ptr<Directory>>; return *p; }
maya::ExpertLayout Layout(int k, int mid, int kg, int kd) {
    maya::ExpertLayout lay;
    for (int role = 0; role < 3; ++role) {
        int in = role==2 ? mid : k, out = role==2 ? k : mid;
        lay.k2[role] = role==2 ? kd : kg;
        lay.suh[role] = maya::trellis_bytes_k2(in,out,lay.k2[role]);
        lay.svh[role] = lay.suh[role] + 2ull * in;
    }
    return lay;
}
__global__ void Plan(const int *ids, const unsigned long long *lookup, unsigned long long *plan, int count, int experts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) plan[i] = ids[i] >= 0 && ids[i] < experts ? lookup[ids[i]] : 0ull;
}
__global__ void CachePlan(const uint8_t *records, const int *slots, const uint64_t *offsets,
                         size_t stride, unsigned long long *plan, int count) {
    const int i = threadIdx.x;
    if (i < count) {
        const int slot = slots[i];
        plan[i] = slot < 0 ? 0ull : reinterpret_cast<unsigned long long>(
            records + (offsets ? offsets[slot] : size_t(slot) * stride));
    }
}

// Bind the complete logical batch before grouping. Entries retain the
// arithmetic kernel's token * 8 + top-k encoding; 8 is the route stride,
// not the number of tokens admitted by the adapter.
__global__ void WindowRoutes(const int *selection, const float *scores, int rows, int topk,
        const unsigned long long *lookup, int experts, const uint8_t *records,
        const uint64_t *offsets, size_t stride, const int *routeMap, int count,
        unsigned long long *plan, float *weights, uint64_t *keys, int *entries) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int route = routeMap ? routeMap[i] : i;
    unsigned long long ptr = 0;
    const int encoded = route >= 0 && route < rows * topk ? route / topk * 8 + route % topk : 0;
    if (route >= 0 && route < rows * topk) {
        const int id = selection[i];
        if (id >= 0) {
            if (lookup) ptr = id < experts ? lookup[id] : 0;
            else ptr = reinterpret_cast<unsigned long long>(records + (offsets ? offsets[id] : size_t(id) * stride));
        }
        plan[encoded] = ptr;
        weights[encoded] = scores[route];
    }
    keys[i] = ptr;
    entries[i] = encoded;
}

bool Window(const Data &input, const int *selection, const float *scores, int topk,
        int mid, float limit, const maya::ExpertLayout &layout, const maya::MixedLayouts &mix,
        bool indirect, const unsigned long long *lookup, int experts, const uint8_t *records,
        const uint64_t *offsets, size_t stride, const int *routeMap, int routeCount,
        float *output, bool routeOutput, bool preserveMissing, int scratchLane) {
    using fastllm::moe_routes::Plan;
    const int rows = input.dims[0], k = input.dims[1];
    if (rows < 1 || rows > INT_MAX / 8 || topk < 1 || topk > 8) return false;
    const int count = routeMap ? routeCount : rows * topk, capacity = rows * 8;
    if (count < 1 || count > rows * topk) return false;
    // Graphs share capacity classes, not one persistent allocation per exact
    // row count. Allocate the whole class before capture so later shapes cannot
    // move addresses referenced by an earlier graph. Total retained storage
    // grows geometrically with the largest batch rather than quadratically.
    int workspaceRows = 1;
    while (workspaceRows < rows && workspaceRows <= INT_MAX/16) workspaceRows *= 2;
    workspaceRows = std::max(workspaceRows,rows);
    const int workspaceRoutes = workspaceRows * topk, workspaceEntries = workspaceRows * 8;
    const size_t scratch = Plan::TemporaryBytes(workspaceRoutes);
    const size_t bytes = scratch ? Plan(nullptr, workspaceRoutes, scratch).bytes : 0;
    if (!bytes || bytes > INT_MAX) return false;
    auto &work = Work(k, mid, workspaceRows, -128-16*scratchLane-topk);
    work.plan.Resize({workspaceEntries, int(sizeof(unsigned long long))});
    work.scores.Resize({workspaceEntries});
    work.h.Resize({workspaceEntries, mid}); work.up.Resize({workspaceEntries, mid});
    work.part.Resize({workspaceEntries, k}); work.groups.Resize({int(bytes)});
    work.counters.Resize({workspaceRoutes, mid / 128});
    for (auto *d : {&work.plan, &work.scores, &work.h, &work.up, &work.part, &work.groups, &work.counters}) d->Allocate(false);
    auto *plan = static_cast<unsigned long long *>(work.plan.cudaData);
    auto *weights = static_cast<float *>(work.scores.cudaData);
    Plan grouped(work.groups.cudaData, count, scratch);
    if (cudaMemsetAsync(plan, 0, size_t(capacity) * sizeof(*plan), cudaStreamPerThread) != cudaSuccess ||
        cudaMemsetAsync(weights, 0, size_t(capacity) * sizeof(*weights), cudaStreamPerThread) != cudaSuccess ||
        cudaMemsetAsync(work.counters.cudaData, 0, work.counters.GetBytes(), cudaStreamPerThread) != cudaSuccess) return false;
    WindowRoutes<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(selection,scores,rows,topk,lookup,experts,records,
        offsets,stride,routeMap,count,plan,weights,grouped.keys,grouped.routes);
    if (!grouped.Build(cudaStreamPerThread)) return false;
    maya::WinArgs a;
    a.T=rows; a.k=topk; a.n_embd=k; a.n_ff=mid; a.limit=limit;
    a.grp_ptr=reinterpret_cast<const unsigned long long *>(grouped.records);
    a.grp_start=grouped.starts; a.grp_ent=grouped.entries; a.n_groups=grouped.groups;
    a.plan_ptr=plan; a.plan_w=weights; a.lay=layout; a.mix=mix; a.indirect=indirect;
    a.x16=static_cast<const uint16_t *>(input.cudaData);
    a.x_bf16=true; a.x_stride=k;
    a.h=static_cast<float *>(work.h.cudaData); a.part=static_cast<float *>(work.part.cudaData);
    a.up=static_cast<float *>(work.up.cudaData);
    a.group_counters=static_cast<unsigned *>(work.counters.cudaData); a.group_capacity=count;
    a.out=output; a.out_stride=k; a.round_bf16=true; a.route_output=routeOutput;
    a.preserve_missing=preserveMissing; a.scratch_lane=scratchLane;
    return maya::win_experts(a,cudaStreamPerThread) && cudaGetLastError()==cudaSuccess;
}

}

bool FastllmCudaExl3MayaCache(const fastllm::Data &input, const uint8_t *records,
    const int32_t *slots, const uint64_t *offsets, size_t stride, size_t downOffset,
    int inter, int gateBits, int downBits, const float *scores, int topk,
    float limit, float *perExpert, int workspace, const int32_t *routeMap, int routeCount, bool preserveMissing) {
    using namespace fastllm;
    if (!Enabled() || input.dataType != BFLOAT16 || input.dims.size() != 2 ||
        input.dims[0] < 1 || topk < 1 || topk > 8 ||
        !maya::k2_supported(gateBits,2) || !maya::k2_supported(downBits,2)) return false;
    const int k = input.dims[1], rows = input.dims[0];
    if (k < 128 || inter < 128 || k > 8192 || inter > 8192 || k%128 || inter%128) return false;
    if (rows > 1 || routeMap || preserveMissing) {
        auto layout = Layout(k, inter, gateBits, downBits);
        const size_t offsetsByRole[3] = {0, Exl3PartBytes(inter,k,gateBits), downOffset};
        for (int role=0; role<3; ++role) {
            layout.trellis[role] += offsetsByRole[role];
            layout.suh[role] += offsetsByRole[role]; layout.svh[role] += offsetsByRole[role];
        }
        return Window(input, slots, scores, topk, inter, limit, layout, {}, false,
            nullptr, 0, records, offsets, stride, routeMap, routeCount,
            perExpert, true, preserveMissing, 1+workspace);
    }
    auto &work = Work(k, inter, rows, -topk-16*workspace);
    work.plan.Resize({rows*topk*int(sizeof(unsigned long long))}); work.plan.Allocate(false);
    work.h.Resize({rows*topk,inter}); work.h.Allocate(false);
    auto *plan = static_cast<unsigned long long *>(work.plan.cudaData);
    CachePlan<<<1,64,0,cudaStreamPerThread>>>(records,slots,offsets,stride,plan,rows*topk);
    maya::MoeArgs a; a.k=topk; a.n_embd=k; a.n_ff=inter; a.limit=limit; a.route_output=true; a.scratch_lane=1+workspace;
    a.lay=Layout(k,inter,gateBits,downBits); a.round_bf16=true;
    const size_t roleOffset[3]={0,Exl3PartBytes(inter,k,gateBits),downOffset};
    for (int role=0;role<3;++role) {
        a.lay.trellis[role]+=roleOffset[role];
        a.lay.suh[role]+=roleOffset[role]; a.lay.svh[role]+=roleOffset[role];
    }
    // Multi-row requests use Window above; keep the single-row specialization.
    a.plan_ptr=plan; a.plan_w=scores;
    a.x16=static_cast<const uint16_t *>(input.cudaData); a.x_bf16=true;
    auto *hidden=static_cast<float *>(work.h.cudaData);
    maya::moe_gate_up(a,nullptr,hidden,cudaStreamPerThread);
    maya::moe_down(a,hidden,nullptr,perExpert,cudaStreamPerThread);
    return cudaGetLastError()==cudaSuccess;
}

// Keep the backend register tile separate from the logical projection batch.
// The registry benchmarks the decode-plus-GEMM alternative during warmup;
// a register tile is never a logical batch admission limit.
// Untuned shapes use a weight-traffic estimate, not a logical row cap.
// Compressed tiles reread packed weights; expansion reads them once, then
// writes and reads an FP16 matrix. Warmup may override this estimate. The
// tile width comes from the actual modern/legacy kernel, not an SM table here.
static bool Exl3PreferExpansion(const maya::MvJob *jobs,int count,int rows) {
    double tiled=0,expanded=0;
    for(int i=0;i<count;++i) {
        const auto &m=*jobs[i].m;
        const int columns=std::max(1,maya::mv_cols(m));
        const double packed=double(maya::trellis_bytes_k2(m.k,m.n,m.k2))+2.0*(m.k+m.n);
        tiled+=((int64_t(rows)+columns-1)/columns)*packed;
        expanded+=packed+4.0*m.k*m.n;
    }
    return tiled>expanded;
}

static bool Exl3LinearTiles(const maya::MvJob *source, int count, int rows, cudaStream_t stream) {
    for (int first = 0; first < rows; first += maya::kMaxMvCols) {
        std::array<maya::MvJob, maya::kMaxMvJobs> jobs;
        for (int i = 0; i < count; ++i) {
            jobs[i] = source[i];
            auto &j = jobs[i];
            j.ncol = std::min(maya::kMaxMvCols, rows - first);
            if (j.xf) j.xf += size_t(first) * j.xf_cs;
            if (j.x16) j.x16 += size_t(first) * j.xf_cs;
            if (j.y) j.y += size_t(first) * j.y_cs;
            if (j.y16) j.y16 += size_t(first) * j.y_cs;
        }
        if (!maya::mv(jobs.data(), count, stream)) return false;
    }
    return true;
}

bool FastllmCudaExl3MayaLinear(const fastllm::Data &input, fastllm::Data &weight,
                              const fastllm::Data &bias, fastllm::Data &output) {
    using namespace fastllm;
    if (!Enabled() || weight.dataType != EXL3 || weight.dims.size()!=2 || weight.blockM<1 || weight.blockM>maya::kMaxMvJobs ||
        !input.cudaData || !weight.cudaData || !output.cudaData ||
        (input.dataType!=FLOAT32 && input.dataType!=FLOAT16 && input.dataType!=BFLOAT16) ||
        (output.dataType!=FLOAT32 && output.dataType!=FLOAT16 && output.dataType!=BFLOAT16)) return false;
    int k = weight.dims[1], total = weight.dims[0], rows = input.Count(0)/k;
    if (rows<1) return false;
    AssertInFastLLM(input.dims.back()==k && output.dims.back()==total,"EXL3 Maya: linear shape mismatch.");
    Exl3StorageBytes(weight);
    std::array<maya::Mat,maya::kMaxMvJobs> mats;
    for (int p=0;p<weight.blockM;++p) { mats[p]=Matrix(weight,p); if(!maya::supported(mats[p])) return false; }
    auto stream = cudaStreamPerThread;
    std::array<maya::MvJob,maya::kMaxMvJobs> jobs;
    for (int p=0;p<weight.blockM;++p) {
        auto &j=jobs[p]; j.m=&mats[p];
        if (input.dataType==FLOAT32) j.xf=static_cast<const float *>(input.cudaData);
        else { j.x16=static_cast<const uint16_t *>(input.cudaData); j.x_bf16=input.dataType==BFLOAT16; }
        if (output.dataType==FLOAT32) j.y=static_cast<float *>(output.cudaData)+p*mats[p].n;
        else { j.y16=static_cast<uint16_t *>(output.cudaData)+p*mats[p].n; j.y_bf16=output.dataType==BFLOAT16; }
        j.bias=bias.dims.empty()?nullptr:static_cast<const float *>(bias.cudaData)+p*mats[p].n;
        j.ncol=rows; j.xf_cs=k; j.y_cs=total;
    }
    if (FastllmCudaExl3SelectProjection(input,weight,bias,output,false,
            Exl3PreferExpansion(jobs.data(),weight.blockM,rows)) == 1)
        return FastllmCudaExl3ProjectionFallback(input,weight,bias,output,false);
    AssertInFastLLM(Exl3LinearTiles(jobs.data(),weight.blockM,rows,stream),"EXL3 Maya: matrix launch rejected.");
    AssertInFastLLM(cudaGetLastError()==cudaSuccess,"EXL3 Maya: linear CUDA failure.");
    return true;
}

bool FastllmCudaExl3MayaLinearShared(const fastllm::Data &input,
        fastllm::Data *const *weights, fastllm::Data *const *outputs, int count) {
    using namespace fastllm;
    if (!Enabled() || count < 2 || count > maya::kMaxMvJobs || !weights || !outputs ||
        input.dims.empty() || input.dims.back() <= 0 || !input.cudaData ||
        (input.dataType != FLOAT32 && input.dataType != FLOAT16 && input.dataType != BFLOAT16)) return false;
    const int k = input.dims.back(), rows = input.Count(0) / k;
    if (rows < 1) return false;
    std::array<maya::Mat, maya::kMaxMvJobs> mats;
    std::array<maya::MvJob, maya::kMaxMvJobs> jobs;
    int used = 0;
    // Build every descriptor before launching: a rejected group must leave all
    // outputs intact so the caller can safely use its ordinary linear path.
    for (int i = 0; i < count; ++i) {
        if (!weights[i] || !outputs[i]) return false;
        const Data &w = *weights[i]; Data &y = *outputs[i];
        if (w.dataType != EXL3 || w.dims.size() != 2 || w.dims[1] != k ||
            w.blockM < 1 || w.blockM > maya::kMaxMvJobs - used || w.dims[0] % w.blockM ||
            !w.cudaData || !y.cudaData || y.dims.empty() || y.dims.back() != w.dims[0] ||
            y.Count(0) != uint64_t(rows) * w.dims[0] ||
            (y.dataType != FLOAT32 && y.dataType != FLOAT16 && y.dataType != BFLOAT16)) return false;
        Exl3StorageBytes(w);
        for (int p = 0; p < w.blockM; ++p) {
            auto &m = mats[used]; m = Matrix(w, p);
            if (!maya::supported(m)) return false;
            auto &j = jobs[used++]; j.m = &m;
            if (input.dataType == FLOAT32) j.xf = static_cast<const float *>(input.cudaData);
            else { j.x16 = static_cast<const uint16_t *>(input.cudaData); j.x_bf16 = input.dataType == BFLOAT16; }
            if (y.dataType == FLOAT32) j.y = static_cast<float *>(y.cudaData) + p * m.n;
            else { j.y16 = static_cast<uint16_t *>(y.cudaData) + p * m.n; j.y_bf16 = y.dataType == BFLOAT16; }
            j.ncol = rows; j.xf_cs = k; j.y_cs = w.dims[0];
        }
    }
    if (!fastllm::kernels::bypassDepth) {
        Data empty;
        std::array<int,maya::kMaxMvJobs> selected{};
        bool separate = false;
        int first=0;
        for (int i=0;i<count;++i) {
            selected[i]=FastllmCudaExl3SelectProjection(input,*weights[i],empty,*outputs[i],false,
                Exl3PreferExpansion(jobs.data()+first,weights[i]->blockM,rows));
            first+=weights[i]->blockM;
            separate |= selected[i]==1;
        }
        if (separate) {
            fastllm::kernels::Bypass bypass;
            for (int i=0;i<count;++i) {
                const bool ok=selected[i]==1
                    ? FastllmCudaExl3ProjectionFallback(input,*weights[i],empty,*outputs[i],false)
                    : FastllmCudaExl3MayaLinear(input,*weights[i],empty,*outputs[i]);
                AssertInFastLLM(ok,"EXL3 shared projection selected launch rejected.");
            }
            return true;
        }
    }
    AssertInFastLLM(Exl3LinearTiles(jobs.data(), used, rows, cudaStreamPerThread), "EXL3 shared projection launch rejected.");
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "EXL3 shared projection CUDA failure.");
    return true;
}

void FastllmCudaReleaseExl3Maya(const fastllm::Data *owner) {
    std::lock_guard<std::mutex> lock(DirectoryMutex());
    auto &map=Directories();
    for (auto it=map.begin(); it!=map.end();) {
        const auto &sources = it->second->sources;
        if (std::find(sources.begin(), sources.end(), owner) != sources.end()) it=map.erase(it);
        else ++it;
    }
}

bool FastllmCudaExl3MayaMoeGraphReady(const fastllm::Data *owner, int rows, int topk) {
    if (!Enabled()) return false;
    std::lock_guard<std::mutex> lock(DirectoryMutex());
    auto it = Directories().find({owner, FastllmCudaGetDevice()});
    return it != Directories().end() &&
        it->second->warmedShapes.count({std::this_thread::get_id(), rows, topk}) != 0;
}

bool FastllmCudaExl3MayaMoe(const fastllm::Data &input, fastllm::Data &output,
                           const fastllm::Data &index, const fastllm::Data &score,
                           fastllm::Data **weights, int count, float limit) {
    using namespace fastllm;
    if (!Enabled() || input.dataType!=BFLOAT16 || input.dims.size()!=2 || input.dims[0]<1 || count<4 || count%2 ||
        weights[0] || !weights[2] || weights[2]->dataType!=EXL3 || weights[2]->dims.size()!=2 ||
        input.dataDevice!=DataDevice::CUDA || !input.cudaData ||
        index.dataDevice!=DataDevice::CUDA || score.dataDevice!=DataDevice::CUDA ||
        index.dims.size()!=2 || index.dims[0]!=input.dims[0] || score.dims!=index.dims ||
        !index.cudaData || !score.cudaData ||
        index.dataDeviceIds!=input.dataDeviceIds || score.dataDeviceIds!=input.dataDeviceIds ||
        index.dataType!=INT32 || score.dataType!=FLOAT32) return false;
    const int k=input.dims[1], mid=weights[2]->dims[0]/2, rows=input.dims[0], topk=index.dims.back();
    const int experts=count/2-1, device=FastllmCudaGetDevice();
    if (topk<1 || topk>8 || k<128 || k>8192 || mid<128 || mid>8192 || k%128 || mid%128) return false;
    std::vector<Signature> signature; signature.reserve(experts);
    std::map<std::pair<int,int>,std::vector<int>> classes;
    for(int e=0;e<experts;++e) {
        Data *g=weights[2+2*e], *d=weights[3+2*e];
        if(!g || !d || g->dataType!=EXL3 || d->dataType!=EXL3 || !g->cudaData || !d->cudaData ||
           g->dims!=std::vector<int>({2*mid,k}) || d->dims!=std::vector<int>({k,mid}) || g->blockM!=2 || d->blockM!=1 ||
           !maya::k2_supported(g->blockK,2) || !maya::k2_supported(d->blockK,2)) return false;
        signature.push_back({g->cudaData,d->cudaData,g->blockK,d->blockK});
        classes[{g->blockK,d->blockK}].push_back(e);
    }
    if(classes.size()>maya::kMaxClasses) return false;
    if(classes.size()>1) for(const auto &c:classes) if(!maya::mixed_supported(Layout(k,mid,c.first.first,c.first.second))) return false;
    std::shared_ptr<Directory> dir;
    {
        std::lock_guard<std::mutex> lock(DirectoryMutex());
        auto &entry=Directories()[{weights[2],device}];
        if(!entry || entry->k!=k || entry->mid!=mid || entry->signature!=signature) {
            if (FastllmCudaGraphIsCapturing()) return false;
            entry=std::make_shared<Directory>(); auto &d=*entry;
            d.signature=std::move(signature); d.k=k; d.mid=mid;
            d.sources.assign(weights+2, weights+count);
            d.table.dataDevice=DataDevice::CUDA; d.table.dataDeviceIds={device};
            d.table.Resize({experts*4*int(sizeof(unsigned long long))}); d.table.Allocate(false);
            std::vector<unsigned long long> data(experts*4);
            auto base=reinterpret_cast<unsigned long long>(d.table.cudaData);
            int slot=0, cls=0;
            for(const auto &c:classes) {
                auto lay=Layout(k,mid,c.first.first,c.first.second);
                d.mix.lo[cls]=base+slot*3*sizeof(unsigned long long);
                for(int e:c.second) {
                    const auto &sig=d.signature[e];
                    data[slot*3]=reinterpret_cast<unsigned long long>(sig.gate);
                    data[slot*3+1]=data[slot*3]+Exl3PartBytes(mid,k,sig.kg);
                    data[slot*3+2]=reinterpret_cast<unsigned long long>(sig.down);
                    data[experts*3+e]=base+slot*3*sizeof(unsigned long long); ++slot;
                }
                d.mix.hi[cls]=base+slot*3*sizeof(unsigned long long); d.mix.lay[cls]=lay; d.lay=lay; ++cls;
            }
            d.mix.n=classes.size()>1?cls:0;
            d.lookup=reinterpret_cast<const unsigned long long *>(base+experts*3*sizeof(unsigned long long));
            FastllmCudaCopyFromHostToDevice(d.table.cudaData,data.data(),d.table.GetBytes());
        }
        dir=entry;
        if (FastllmCudaGraphIsCapturing() &&
            !dir->warmedShapes.count({std::this_thread::get_id(), rows, topk})) return false;
    }
    auto markWarm = [&] {
        if (!FastllmCudaGraphIsCapturing()) {
            std::lock_guard<std::mutex> lock(DirectoryMutex());
            dir->warmedShapes.emplace(std::this_thread::get_id(), rows, topk);
        }
    };
    if (rows > 1) {
        if (output.dataType != FLOAT32) { output.FreeSpace(); output.dataType=FLOAT32; }
        output.Resize({rows,k}); output.ToDevice(CUDA,{device},false); output.Allocate(false);
        if (!Window(input, static_cast<const int *>(index.cudaData),
            static_cast<const float *>(score.cudaData), topk, mid, limit,
            dir->lay, dir->mix, true, dir->lookup, experts, nullptr, nullptr, 0,
            nullptr, 0, static_cast<float *>(output.cudaData), false, false, 0)) return false;
        markWarm();
        return true;
    }
    auto stream=cudaStreamPerThread;
    auto &work=Work(k,mid,rows,topk);
    work.plan.Resize({rows*topk*int(sizeof(unsigned long long))}); work.plan.Allocate(false);
    work.h.Resize({topk,mid}); work.h.Allocate(false);
    auto *plan=static_cast<unsigned long long *>(work.plan.cudaData);
    Plan<<<(rows*topk+127)/128,128,0,stream>>>(static_cast<const int *>(index.cudaData),dir->lookup,plan,rows*topk,experts);
    if(output.dataType!=FLOAT32) { output.FreeSpace(); output.dataType=FLOAT32; }
    output.Resize({rows,k}); output.ToDevice(DataDevice::CUDA,{device},false); output.Allocate(false);
    maya::MoeArgs a; a.k=topk; a.n_embd=k; a.n_ff=mid; a.limit=limit;
    a.lay=dir->lay; a.mix=dir->mix; a.indirect=true; a.round_bf16=true;
    for(int row=0;row<rows;++row) {
        a.plan_ptr=plan+row*topk; a.plan_w=static_cast<const float *>(score.cudaData)+row*topk;
        auto *h=static_cast<float *>(work.h.cudaData);
        a.x16=static_cast<const uint16_t *>(input.cudaData)+row*k; a.x_bf16=true;
        maya::moe_gate_up(a,nullptr,h,stream);
        maya::moe_down(a,h,nullptr,static_cast<float *>(output.cudaData)+row*k,stream);
    }
    AssertInFastLLM(cudaGetLastError()==cudaSuccess,"EXL3 Maya: MoE CUDA failure.");
    markWarm();
    return true;
}

bool FastllmCudaExl3MayaGrouped(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output) {
    using namespace fastllm;
    if (!Enabled() || weight.dims.size() != 3) return false;
    const int groups = weight.dims[0], n = weight.dims[1], k = weight.dims[2];
    const int rows = input.Count(0) / (groups * k);
    if (groups < 1 || groups > maya::kMaxMvJobs || rows < 1) return false;
    std::array<maya::Mat,maya::kMaxMvJobs> mats;
    std::array<maya::MvJob,maya::kMaxMvJobs> jobs;
    for (int g = 0; g < groups; ++g) {
        auto &m = mats[g]; m.k = k; m.n = n; m.k2 = weight.blockK;
        const auto *base = static_cast<const uint8_t *>(weight.cudaData) + size_t(g) * Exl3PartBytes(n,k,m.k2);
        m.trellis = reinterpret_cast<const uint32_t *>(base);
        m.suh = reinterpret_cast<const uint16_t *>(base + maya::trellis_bytes_k2(k,n,m.k2)); m.svh = m.suh + k;
        if (!maya::supported(m)) return false;
        auto &j = jobs[g]; j.m = &m;
        if (input.dataType == FLOAT32) j.xf = static_cast<const float *>(input.cudaData) + size_t(g) * k;
        else { j.x16 = static_cast<const uint16_t *>(input.cudaData) + size_t(g) * k; j.x_bf16 = input.dataType == BFLOAT16; }
        if (output.dataType == FLOAT32) j.y = static_cast<float *>(output.cudaData) + size_t(g) * n;
        else { j.y16 = static_cast<uint16_t *>(output.cudaData) + size_t(g) * n; j.y_bf16 = output.dataType == BFLOAT16; }
        j.ncol = rows; j.xf_cs = groups * k; j.y_cs = groups * n;
    }
    if (FastllmCudaExl3SelectProjection(input,weight,Data(),output,true,
            Exl3PreferExpansion(jobs.data(),groups,rows)) == 1)
        return FastllmCudaExl3ProjectionFallback(input,weight,Data(),output,true);
    AssertInFastLLM(Exl3LinearTiles(jobs.data(),groups,rows,cudaStreamPerThread),"EXL3 Maya: grouped launch rejected");
    return cudaGetLastError() == cudaSuccess;
}
