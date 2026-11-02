#include "models/hy_nomas/parallel.h"
#include "utils/persistent_worker_group.h"
#include "devices/multicuda/ncclsubmitrendezvous.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef FASTLLM_CONTINUOUS_NCCL
#include <nccl.h>
#endif
#include <algorithm>
#include <atomic>
#include <climits>
#include <mutex>
#include <stdexcept>

namespace fastllm { namespace nomas {
namespace {
void check(cudaError_t s) {
    if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
#ifdef FASTLLM_CONTINUOUS_NCCL
void check(ncclResult_t s) {
    if(s!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(s));
}
ncclDataType_t type(DType t) {
    return t==DType::FP32?ncclFloat32:t==DType::FP16?ncclFloat16:ncclBfloat16;
}
template<class T> __global__ void gatherLayout(const T *input,T *output,size_t n,int rows,int cols,int ranks,int groups) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x) {
        size_t row=i/(cols*ranks),col=i%(cols*ranks);
        int width=cols/groups;
        size_t group=col/(width*ranks),within=col%(width*ranks);
        output[i]=input[(within/width*rows+row)*cols+group*width+within%width];
    }
}
template<class T> __global__ void finishShard(const float *input, const float *bias, T *output,
                                             size_t n, size_t offset, int cols) {
    for(size_t j=size_t(blockIdx.x)*blockDim.x+threadIdx.x;j<n;j+=size_t(blockDim.x)*gridDim.x) {
        float x=input[j];
        if(bias) x=__fadd_rn(x,bias[(offset+j)%cols]);
        output[j]=T(x);
    }
}
void castShard(const Tensor &input,const Tensor *bias,Tensor &output,size_t offset,int cols,cudaStream_t stream) {
    size_t count=input.size();
    int blocks=std::min<size_t>((count+255)/256,65535);
    const auto *b=bias?reinterpret_cast<const float *>(bias->data):nullptr;
    if(output.dtype==DType::BF16)
        finishShard<<<blocks,256,0,stream>>>((const float *)input.data,b,
            (__nv_bfloat16 *)output.data,count,offset,cols);
    else
        finishShard<<<blocks,256,0,stream>>>((const float *)input.data,b,
            (__half *)output.data,count,offset,cols);
    check(cudaGetLastError());
}
#endif
}
#ifdef FASTLLM_CONTINUOUS_NCCL
struct ParallelGroup::Impl {
    std::vector<int> devices;
    std::vector<ncclComm_t> comms;
    struct Communication {
        cudaStream_t stream=nullptr;
        cudaEvent_t ready=nullptr,done=nullptr;
        std::vector<Tensor> retained;
        bool pending=false;
    };
    std::vector<Communication> communication;
    Communication &beginCommunication(int rank) {
        auto &c=communication[rank];
        if(!c.stream) {
            check(cudaStreamCreateWithFlags(&c.stream,cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&c.ready,cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&c.done,cudaEventDisableTiming));
        }
        c.pending=true;
        check(cudaEventRecord(c.ready,cudaStreamPerThread));
        check(cudaStreamWaitEvent(c.stream,c.ready,0));
        return c;
    }
    PersistentWorkerGroup workers;
    std::unique_ptr<NcclSubmitRendezvous> rendezvous;
    std::mutex abortMutex;
    std::atomic<bool> failed{false};
    explicit Impl(const std::vector<int> &d):devices(d),comms(d.size(),nullptr),communication(d.size()) {
        int count=0; check(cudaGetDeviceCount(&count));
        if(d.empty()) throw std::invalid_argument("empty tensor-parallel device list");
        auto sorted=d; std::sort(sorted.begin(),sorted.end());
        if(sorted.front()<0 || sorted.back()>=count || std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())
            throw std::invalid_argument("tensor-parallel devices must be distinct visible CUDA indices");
        initialize();
    }
    void initialize() {
        std::fill(comms.begin(),comms.end(),nullptr);
        rendezvous.reset(new NcclSubmitRendezvous(devices.size()));
        failed=false;
        if(devices.size()>1) {
            ncclResult_t status;
#if NCCL_VERSION_CODE >= 21700
            int minimumSMs=INT_MAX;
            for(int device:devices) {
                cudaDeviceProp properties{};
                check(cudaGetDeviceProperties(&properties,device));
                minimumSMs=std::min(minimumSMs,properties.multiProcessorCount);
            }
            if(minimumSMs>=64) {
                // Continuous tensors have large collective payloads. A four-CTA
                // PCIe default underfills these transfers; use at most a quarter
                // of the smallest GPU for the minimum, leaving NCCL room to grow.
                ncclConfig_t config=NCCL_CONFIG_INITIALIZER;
                config.minCTAs=16;
                config.maxCTAs=32;
                ncclUniqueId id;
                check(ncclGetUniqueId(&id));
                check(ncclGroupStart());
                status=ncclSuccess;
                for(size_t rank=0;rank<devices.size();++rank) {
                    auto deviceStatus=cudaSetDevice(devices[rank]);
                    if(deviceStatus!=cudaSuccess) { status=ncclUnhandledCudaError; break; }
                    auto initialized=ncclCommInitRankConfig(&comms[rank],devices.size(),id,rank,&config);
                    if(initialized!=ncclSuccess) { status=initialized; break; }
                }
                auto completed=ncclGroupEnd();
                if(status==ncclSuccess) status=completed;
            } else
#endif
                status=ncclCommInitAll(comms.data(),devices.size(),devices.data());
            if(status!=ncclSuccess) { abort(); check(status); }
        }
    }
    void abort() {
        std::lock_guard<std::mutex> lock(abortMutex);
        rendezvous->Abort("tensor-parallel rank failed");
        if(failed.exchange(true)) return;
        for(auto comm:comms) {
            if(comm) ncclCommAbort(comm);
        }
    }
    ~Impl() {
        workers.Stop();
        if(!failed) for(auto comm:comms) if(comm) ncclCommDestroy(comm);
        for(size_t rank=0;rank<devices.size();++rank) {
            cudaSetDevice(devices[rank]);
            auto &c=communication[rank];
            if(c.stream) cudaStreamDestroy(c.stream);
            if(c.ready) cudaEventDestroy(c.ready);
            if(c.done) cudaEventDestroy(c.done);
        }
    }
    void boundary(int rank,NcclSubmitRendezvous::Phase phase,const Tensor &x,int operation) {
        if(x.size()>INT_MAX) throw std::invalid_argument("tensor-parallel tensor too large");
        if(!rendezvous->Wait(rank,phase,int(x.size()),operation*16+int(type(x.dtype))))
            throw std::runtime_error(rendezvous->Error());
    }
};
ParallelGroup::ParallelGroup(const std::vector<int> &d):impl_(new Impl(d)) {}
ParallelGroup::~ParallelGroup()=default;
int ParallelGroup::size() const { return impl_->devices.size(); }
void ParallelGroup::run(const std::function<void(int)> &task) {
    auto &i=*impl_;
    if(i.failed) i.initialize();
    std::vector<std::exception_ptr> errors(size());
    std::exception_ptr originalError;
    std::mutex errorMutex;
    i.workers.RunWithCaller(i.devices,[&](int rank) {
        try {
            check(cudaSetDevice(i.devices[rank]));
            task(rank);
            wait(rank);
            check(cudaStreamSynchronize(cudaStreamPerThread));
        } catch(...) {
            auto error=std::current_exception();
            {
                std::lock_guard<std::mutex> lock(errorMutex);
                if(!originalError) originalError=error;
            }
            i.abort();
            auto &c=i.communication[rank];
            if(c.stream) cudaStreamSynchronize(c.stream);
            cudaStreamSynchronize(cudaStreamPerThread);
            c.retained.clear(); c.pending=false;
            std::rethrow_exception(error);
        }
    },errors);
    check(cudaSetDevice(i.devices[0]));
    if(originalError) std::rethrow_exception(originalError);
    for(auto error:errors) if(error) std::rethrow_exception(error);
}
Tensor ParallelGroup::broadcast(const Tensor &root,int rank) {
    if(size()==1) return root;
    Tensor out=rank==0?root:Tensor(root.rows,root.cols,root.dtype);
    auto &i=*impl_;
    i.boundary(rank,NcclSubmitRendezvous::Before,out,0);
    check(ncclBroadcast(out.data,out.data,out.size(),type(out.dtype),0,i.comms[rank],cudaStreamPerThread));
    i.boundary(rank,NcclSubmitRendezvous::After,out,0);
    return out;
}
void ParallelGroup::sum(Tensor &x,int rank) {
    if(size()==1) return;
    auto &i=*impl_;
    i.boundary(rank,NcclSubmitRendezvous::Before,x,1);
    check(ncclAllReduce(x.data,x.data,x.size(),type(x.dtype),ncclSum,i.comms[rank],cudaStreamPerThread));
    i.boundary(rank,NcclSubmitRendezvous::After,x,1);
}
void ParallelGroup::sumAsync(const Tensor &x,int rank) {
    if(size()==1) return;
    if(!x.data) throw std::invalid_argument("empty asynchronous reduction");
    auto &i=*impl_; auto &c=i.beginCommunication(rank);
    c.retained.push_back(x);
    i.boundary(rank,NcclSubmitRendezvous::Before,x,8);
    check(ncclAllReduce(x.data,x.data,x.size(),type(x.dtype),ncclSum,i.comms[rank],c.stream));
    i.boundary(rank,NcclSubmitRendezvous::After,x,8);
}
Tensor ParallelGroup::sumCast(Runtime &rt,Tensor partial,const Tensor *bias,DType output,int rank) {
    if(!partial.isFloat32() || (bias && (!bias->isFloat32() || bias->size()!=size_t(partial.cols))))
        throw std::invalid_argument("parallel sum/cast requires FP32 partials and bias");
    // Small messages favor one collective. Uneven element counts retain the
    // general all-reduce path; no padding or change to FP32 diagnostics.
    if(size()==1 || output==DType::FP32 || partial.bytes()<(1u<<20) || partial.size()%size()) {
        sum(partial,rank);
        if(bias) partial=rt.bias(partial,*bias);
        return rt.cast(partial,output);
    }
    const size_t count=partial.size()/size();
    if(count>INT_MAX) throw std::invalid_argument("parallel reduction shard too large");
    Tensor reduced(1,int(count),DType::FP32),rounded(1,int(count),output);
    Tensor result(partial.rows,partial.cols,output);
    auto &i=*impl_;
    i.boundary(rank,NcclSubmitRendezvous::Before,partial,3);
    check(ncclReduceScatter(partial.data,reduced.data,count,ncclFloat32,ncclSum,i.comms[rank],cudaStreamPerThread));
    i.boundary(rank,NcclSubmitRendezvous::After,partial,3);
    castShard(reduced,bias,rounded,size_t(rank)*count,partial.cols,cudaStreamPerThread);
    i.boundary(rank,NcclSubmitRendezvous::Before,rounded,4);
    check(ncclAllGather(rounded.data,result.data,count,type(output),i.comms[rank],cudaStreamPerThread));
    i.boundary(rank,NcclSubmitRendezvous::After,rounded,4);
    return result;
}
void ParallelGroup::sumCastAsync(const Tensor &partial,const Tensor *bias,Tensor &output,int rank) {
    if(!partial.data || !output.data || !partial.isFloat32() || output.isFloat32() || partial.rows!=output.rows || partial.cols!=output.cols ||
       partial.storage==output.storage || (bias && (!bias->isFloat32() || bias->size()!=size_t(partial.cols))))
        throw std::invalid_argument("invalid asynchronous row reduction");
    auto &i=*impl_;
    const bool scatter=size()>1 && partial.bytes()>=(1u<<20) && partial.size()%size()==0;
    const size_t count=scatter?partial.size()/size():partial.size();
    if(count>INT_MAX) throw std::invalid_argument("parallel reduction shard too large");
    Tensor reduced=scatter?Tensor(1,int(count),DType::FP32):partial;
    Tensor rounded=scatter?Tensor(1,int(count),output.dtype):output;
    auto &c=i.beginCommunication(rank);
    // All allocations and GEMMs are on the forward stream. Keep their storage
    // alive while NCCL consumes it on the communication stream.
    c.retained.insert(c.retained.end(),{partial,output,reduced,rounded});
    if(bias) c.retained.push_back(*bias);
    i.boundary(rank,NcclSubmitRendezvous::Before,partial,scatter?5:6);
    if(scatter)
        check(ncclReduceScatter(partial.data,reduced.data,count,ncclFloat32,ncclSum,i.comms[rank],c.stream));
    else if(size()>1)
        check(ncclAllReduce(partial.data,reduced.data,count,ncclFloat32,ncclSum,i.comms[rank],c.stream));
    i.boundary(rank,NcclSubmitRendezvous::After,partial,scatter?5:6);
    size_t offset=scatter?size_t(rank)*count:0;
    castShard(reduced,bias,rounded,offset,partial.cols,c.stream);
    if(scatter) {
        i.boundary(rank,NcclSubmitRendezvous::Before,rounded,7);
        check(ncclAllGather(rounded.data,output.data,count,type(output.dtype),i.comms[rank],c.stream));
        i.boundary(rank,NcclSubmitRendezvous::After,rounded,7);
    }
}
void ParallelGroup::wait(int rank) {
    auto &c=impl_->communication[rank];
    if(!c.pending) return;
    check(cudaEventRecord(c.done,c.stream));
    check(cudaStreamWaitEvent(cudaStreamPerThread,c.done,0));
    c.retained.clear(); c.pending=false;
}
Tensor ParallelGroup::gatherColumns(const Tensor &x,int rank,int groups,bool async) {
    if(size()==1) return x;
    if(x.cols>INT_MAX/size()) throw std::invalid_argument("gathered tensor too wide");
    if(groups<1 || x.cols%groups) throw std::invalid_argument("invalid gathered projection groups");
    Tensor packed(x.rows,x.cols*size(),x.dtype),out(x.rows,x.cols*size(),x.dtype);
    auto &i=*impl_;
    cudaStream_t stream=cudaStreamPerThread;
    if(async) {
        auto &c=i.beginCommunication(rank);
        c.retained.insert(c.retained.end(),{x,packed,out});
        stream=c.stream;
    }
    i.boundary(rank,NcclSubmitRendezvous::Before,x,2);
    check(ncclAllGather(x.data,packed.data,x.size(),type(x.dtype),i.comms[rank],stream));
    i.boundary(rank,NcclSubmitRendezvous::After,x,2);
    int blocks=std::min<size_t>((out.size()+255)/256,65535);
    if(x.isFloat32())
        gatherLayout<<<blocks,256,0,stream>>>((const uint32_t *)packed.data,(uint32_t *)out.data,out.size(),x.rows,x.cols,size(),groups);
    else
        gatherLayout<<<blocks,256,0,stream>>>(packed.data,out.data,out.size(),x.rows,x.cols,size(),groups);
    check(cudaGetLastError());
    return out;
}
std::vector<Tensor> ParallelGroup::mapToRoot(size_t count,const std::function<Tensor(int,size_t)> &task) {
    std::vector<Tensor> local(count),result(count);
    run([&](int rank) {
        for(size_t j=rank;j<count;j+=size()) local[j]=task(rank,j);
    });
    try {
        for(size_t j=0;j<count;++j) {
            int rank=j%size();
            if(rank==0) result[j]=local[j];
            else {
                result[j]=Tensor(local[j].rows,local[j].cols,local[j].dtype);
                check(cudaMemcpyPeerAsync(result[j].data,impl_->devices[0],local[j].data,
                                          impl_->devices[rank],local[j].bytes(),cudaStreamPerThread));
            }
        }
        check(cudaStreamSynchronize(cudaStreamPerThread));
    } catch(...) {
        cudaStreamSynchronize(cudaStreamPerThread);
        throw;
    }
    return result;
}
#else
struct ParallelGroup::Impl { int device; };
ParallelGroup::ParallelGroup(const std::vector<int> &devices):impl_(new Impl) {
    if(devices.size()!=1)
        throw std::runtime_error("tensor parallelism requires a build with NCCL headers and library");
    impl_->device=devices[0]; check(cudaSetDevice(impl_->device));
}
ParallelGroup::~ParallelGroup()=default;
int ParallelGroup::size() const { return 1; }
void ParallelGroup::run(const std::function<void(int)> &task) {
    check(cudaSetDevice(impl_->device)); task(0); check(cudaStreamSynchronize(cudaStreamPerThread));
}
Tensor ParallelGroup::broadcast(const Tensor &x,int) { return x; }
Tensor ParallelGroup::gatherColumns(const Tensor &x,int,int,bool) { return x; }
void ParallelGroup::sum(Tensor &,int) {}
void ParallelGroup::sumAsync(const Tensor &,int) {}
Tensor ParallelGroup::sumCast(Runtime &rt,Tensor partial,const Tensor *bias,DType output,int) {
    if(bias) partial=rt.bias(partial,*bias);
    return rt.cast(partial,output);
}
void ParallelGroup::sumCastAsync(const Tensor &,const Tensor *,Tensor &,int) {
    throw std::runtime_error("asynchronous row reduction requires NCCL");
}
void ParallelGroup::wait(int) {}
std::vector<Tensor> ParallelGroup::mapToRoot(size_t count,const std::function<Tensor(int,size_t)> &task) {
    std::vector<Tensor> result(count);
    run([&](int rank) { for(size_t j=0;j<count;++j) result[j]=task(rank,j); });
    return result;
}
#endif
void ParallelContext::columnWeights(Weights &w,const std::string &p,int groups) const {
    if(size()==1) return;
    w.shard(p+".weight",0,rank,size(),groups);
    w.shard(p+".bias",0,rank,size(),groups);
}
void ParallelContext::rowWeights(Weights &w,const std::string &p) const {
    if(size()>1) w.shard(p+".weight",1,rank,size());
}
Tensor ParallelContext::column(Weights &w,const Tensor &x,const std::string &p,bool gather) const {
    auto out=w.linear(x,p);
    return gather && group?group->gatherColumns(out,rank,w.shardGroups(p+".weight")):out;
}
Tensor ParallelContext::gatherRows(Runtime &rt,const Tensor &x,int rows,int groups,bool async) const {
    if(rows<1 || groups<1 || !x.data || x.cols<1 || x.rows%groups || rank<0 || rank>=size())
        throw std::invalid_argument("invalid row gather shape");
    int chunk=int((int64_t(rows)+size()-1)/size());
    if(int64_t(chunk)*size()*std::max(groups,x.cols)>INT_MAX)
        throw std::invalid_argument("row gather partition too large");
    int local=std::min(chunk,rows-rank*chunk);
    if(local<1 || x.rows/groups!=local)
        throw std::invalid_argument("row gather partition mismatch");
    if(size()==1) return x;
    Tensor packed=x;
    if(local<chunk) {
        std::vector<Tensor> pieces;
        for(int g=0;g<groups;++g) {
            auto part=x.rowSlice(g*local,local);
            pieces.push_back(part);
            pieces.push_back(rt.repeatRows(part.rowSlice(local-1,1),chunk-local));
        }
        packed=rt.concat(pieces);
    }
    packed.rows=groups;packed.cols=chunk*x.cols;
    auto full=group->gatherColumns(packed,rank,1,async);
    int padded=chunk*size();full.rows=groups*padded;full.cols=x.cols;
    if(padded==rows) return full;
    if(groups==1) return full.rowSlice(0,rows);
    if(async) group->wait(rank);
    std::vector<Tensor> pieces;
    for(int g=0;g<groups;++g) pieces.push_back(full.rowSlice(g*padded,rows));
    return rt.concat(pieces);
}
Tensor ParallelContext::row(Runtime &rt,Weights &w,const Tensor &x,const std::string &p) const {
    if(size()==1) return w.linear(x,p);
    const auto &weight=w.get(p+".weight");
    Tensor partial(x.rows,weight.rows,partialType(x.dtype));
    // Keep the same GEMM shapes when changing collective precision.
    if(!x.isFloat32() && x.rows>4096 && partial.size()*sizeof(float)>=(64u<<20)) {
        Tensor output=partial.dtype==x.dtype?partial:Tensor(x.rows,weight.rows,x.dtype);
        for(int first=0;first<x.rows;first+=4096) {
            int rows=std::min(4096,x.rows-first);
            auto target=partial.rowSlice(first,rows),result=output.rowSlice(first,rows);
            rowPartial(rt,w,x.rowSlice(first,rows),p,target);
            finishRowAsync(rt,w,target,p,result);
        }
        group->wait(rank);
        return output;
    }
    rowPartial(rt,w,x,p,partial);
    return finishRow(rt,w,std::move(partial),p,x.dtype);
}
void ParallelContext::rowPartial(Runtime &rt,Weights &w,const Tensor &x,const std::string &p,Tensor &out) const {
    if(out.dtype!=partialType(x.dtype)) throw std::invalid_argument("row-parallel partial precision mismatch");
    const auto *bias=!out.isFloat32() && rank==0?w.optional(p+".bias"):nullptr;
    rt.linearInto(x,w.get(p+".weight"),out,bias);
}
Tensor ParallelContext::finishRow(Runtime &rt,Weights &w,Tensor partial,const std::string &p,DType output) const {
    if(!partial.isFloat32()) {
        if(rowReduction!=RowReduction::Activation || partial.dtype!=output)
            throw std::invalid_argument("row-parallel partial precision mismatch");
        if(group) group->sum(partial,rank);
        return partial;
    }
    const auto *bias=w.optional(p+".bias");
    Tensor fullBias;
    if(bias) fullBias=rt.cast(*bias,DType::FP32);
    if(group) return group->sumCast(rt,std::move(partial),bias?&fullBias:nullptr,output,rank);
    if(bias) partial=rt.bias(partial,fullBias);
    return rt.cast(partial,output);
}
void ParallelContext::finishRowAsync(Runtime &rt,Weights &w,const Tensor &partial,const std::string &p,
                                    Tensor &output) const {
    if(!group) throw std::logic_error("asynchronous row reduction requires a parallel group");
    if(!partial.isFloat32()) {
        if(rowReduction!=RowReduction::Activation || partial.dtype!=output.dtype ||
           partial.data!=output.data || partial.rows!=output.rows || partial.cols!=output.cols)
            throw std::invalid_argument("activation reduction requires an in-place output");
        group->sumAsync(partial,rank);
        return;
    }
    const auto *bias=w.optional(p+".bias");
    Tensor fullBias;
    if(bias) fullBias=rt.cast(*bias,DType::FP32);
    group->sumCastAsync(partial,bias?&fullBias:nullptr,output,rank);
}
std::vector<int> parseParallelDevices(const std::string &value) {
    std::vector<int> out;
    size_t first=0;
    do {
        size_t last=value.find(',',first);
        auto item=value.substr(first,last-first);
        if(item.empty() || item.find_first_not_of("0123456789")!=std::string::npos)
            throw std::invalid_argument("devices must be comma-separated nonnegative CUDA indices");
        int device=std::stoi(item);
        if(std::find(out.begin(),out.end(),device)!=out.end())
            throw std::invalid_argument("duplicate CUDA device");
        out.push_back(device);
        if(last==std::string::npos) break;
        first=last+1;
    } while(true);
    return out;
}
}} // namespace fastllm::nomas
