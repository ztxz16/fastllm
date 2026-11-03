#include "parallel/cuda_runtime.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/multicuda/ncclsubmitrendezvous.h"
#include "executor.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>

namespace fastllm { namespace parallel {
namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void Check(ncclResult_t status) {
    if (status != ncclSuccess) throw std::runtime_error(ncclGetErrorString(status));
}
ncclDataType_t Type(ScalarType type) {
    switch (type) {
        case ScalarType::Byte: return ncclUint8;
        case ScalarType::Float16: return ncclHalf;
        case ScalarType::BFloat16: return ncclBfloat16;
        case ScalarType::Float32: return ncclFloat32;
        case ScalarType::Int32: return ncclInt32;
    }
    throw std::invalid_argument("unsupported collective dtype");
}
size_t Bytes(size_t count, size_t width) {
    if (width && count > std::numeric_limits<size_t>::max() / width)
        throw std::overflow_error("parallel buffer size overflow");
    return count * width;
}
std::atomic<uint64_t> nextGeneration{1};
std::mutex leaseMutex;
std::set<int> leasedDevices;
thread_local const void *activeRuntime = nullptr;
thread_local int activeRank = -1;

__global__ void CopyRowField(const unsigned char *source, unsigned char *dest,
        const int *indices, int rows, size_t width, size_t sourceStride, size_t destStride) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size_t(rows) * width; i += size_t(gridDim.x) * blockDim.x) {
        const size_t row = i / width, column = i % width;
        dest[row * destStride + column] = source[size_t(indices ? indices[row] : row) * sourceStride + column];
    }
}
void CopyField(const void *source, void *dest, const int *indices, int rows,
               size_t width, size_t sourceStride, size_t destStride) {
    const size_t bytes = Bytes(rows, width);
    if (!bytes) return;
    CopyRowField<<<std::min<size_t>((bytes + 255) / 256, 4096), 256, 0, cudaStreamPerThread>>>(
        static_cast<const unsigned char *>(source), static_cast<unsigned char *>(dest),
        indices, rows, width, sourceStride, destStride);
    Check(cudaGetLastError());
}

void Copy(const void *source, void *dest, const int *indices, int rows, size_t rowBytes) {
    const size_t bytes = Bytes(rows, rowBytes);
    if (!bytes) return;
    if (!source || !dest || !indices || source == dest)
        throw std::invalid_argument("row transfer requires distinct valid buffers");
    CopyField(source, dest, indices, rows, rowBytes, rowBytes, rowBytes);
}
}

void RestoreWeightOnHost(Data &weight) {
    if (!weight.multiDeviceData) return;
    const int axis = weight.tpAxis;
    if (weight.tpRanges.empty() || weight.dims.size() != 2 ||
        (axis != 0 && axis != 1) || weight.IsRepacked || weight.unitSizeDiv != 1 ||
        (weight.dataType != FLOAT32 && weight.dataType != FLOAT16 &&
         weight.dataType != BFLOAT16 && weight.dataType != FP8_E4M3))
        throw std::invalid_argument("unsupported load-time weight layout: " + weight.name);
    const int length = weight.dims[axis];
    std::vector<bool> covered(length, false);
    for (const auto &entry : weight.multiDeviceDatas) {
        const Data *local = entry.second;
        auto ranges = weight.tpRanges.find(entry.first);
        if (!local || local->IsRepacked || local->dataType != weight.dataType ||
            local->dims.size() != 2 || ranges == weight.tpRanges.end() ||
            local->dims[1-axis] != weight.dims[1-axis])
            throw std::invalid_argument("invalid load-time weight shard");
        int localRows = 0;
        for (auto range : ranges->second) {
            if (range.first < 0 || range.second <= range.first || range.second > length)
                throw std::invalid_argument("invalid load-time weight range");
            for (int i = range.first; i < range.second; ++i) {
                if (covered[i]) throw std::invalid_argument("overlapping load-time weight ranges");
                covered[i] = true;
            }
            localRows += range.second-range.first;
        }
        if (localRows != local->dims[axis]) throw std::invalid_argument("weight shard shape mismatch");
    }
    if (std::find(covered.begin(), covered.end(), false) != covered.end())
        throw std::invalid_argument("incomplete load-time weight ranges");
    if (weight.dataType == FP8_E4M3 && (weight.blockK <= 0 || weight.blockM <= 0 ||
        weight.scales.size() != size_t((weight.dims[0] + weight.blockK - 1) / weight.blockK) *
                                ((weight.dims[1] + weight.blockM - 1) / weight.blockM)))
        throw std::invalid_argument("missing original FP8 block scales");
    Data full(weight.dataType, weight.dims);
    full.name = weight.name;
    full.Allocate();
    const size_t unit = weight.unitSize;
    for (const auto &entry : weight.multiDeviceDatas) {
        Data &local = *entry.second;
        local.ToDevice(CPU);
        int offset = 0;
        for (auto range : weight.tpRanges.at(entry.first)) {
            const int width = range.second-range.first;
            if (axis == 0) std::memcpy(full.cpuData + size_t(range.first) * weight.dims[1] * unit,
                local.cpuData + size_t(offset) * local.dims[1] * unit, size_t(width) * weight.dims[1] * unit);
            else for (int row = 0; row < weight.dims[0]; ++row)
                std::memcpy(full.cpuData + (size_t(row) * weight.dims[1] + range.first) * unit,
                    local.cpuData + (size_t(row) * local.dims[1] + offset) * unit, width * unit);
            offset += width;
        }
    }
    for (auto &entry : weight.multiDeviceDatas) delete entry.second;
    weight.multiDeviceDatas.clear(); weight.multiDeviceData = false;
    weight.ClearTensorParallelLayout(); weight.FreeSpace();
    weight.CopyFrom(full);
}

struct CudaRuntime::Impl {
    struct State {
        std::vector<ncclComm_t> comms;
        std::unique_ptr<NcclSubmitRendezvous> rendezvous;
    };
    Topology topology;
    uint64_t generation = nextGeneration.fetch_add(1);
    std::vector<State> groups;
    std::vector<std::unique_ptr<Executor>> executors;
    std::vector<std::thread::id> owners;
    std::vector<cudaEvent_t> handoff, completion;
    std::mutex submitMutex, errorMutex;
    std::exception_ptr firstError;
    bool lease = false;

    explicit Impl(Topology topology) : topology(std::move(topology)) {
        std::lock_guard<std::mutex> lock(leaseMutex);
        for (int device : this->topology.devices)
            if (leasedDevices.count(device)) throw std::runtime_error("GPU already leased by a parallel runtime");
        for (int device : this->topology.devices) leasedDevices.insert(device);
        lease = true;
    }
    ~Impl() {
        Release([&](int rank, int) {
            for (size_t g = 0; g < groups.size(); ++g) {
                int local = topology.groups[g].Rank(rank);
                if (local >= 0 && local < int(groups[g].comms.size()) && groups[g].comms[local])
                    ncclCommDestroy(groups[g].comms[local]);
            }
            if (rank < int(handoff.size()) && handoff[rank]) cudaEventDestroy(handoff[rank]);
            if (rank < int(completion.size()) && completion[rank]) cudaEventDestroy(completion[rank]);
        });
        if (lease) {
            std::lock_guard<std::mutex> lock(leaseMutex);
            for (int device : topology.devices) leasedDevices.erase(device);
        }
    }
    void Release(const std::function<void(int,int)> &callback) noexcept {
        // Cleanup is still dispatched after a failed step. Collectives were
        // aborted before that error escaped; do not submit any new collective.
        try {
            if (MultiCudaRunDeviceCallbacksEnqueueOnly(topology.devices, callback))
                MultiCudaSynchronizeDeviceCallbacks(topology.devices);
        } catch (...) {}
    }
    void Fail(std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(errorMutex);
        if (firstError) return;
        firstError = error;
        for (auto &state : groups) if (state.rendezvous)
            state.rendezvous->Abort("parallel runtime participant failed");
        // Do not destroy handles here: other ranks may be unwinding a call.
    }
    void Run(const std::function<void(int,int)> &callback, bool synchronize = true) {
        if (activeRuntime) throw std::logic_error("nested parallel GPU dispatch");
        std::lock_guard<std::mutex> lock(submitMutex);
        if (firstError) std::rethrow_exception(firstError);
        int callerDevice = 0;
        Check(cudaGetDevice(&callerDevice));
        for (size_t r = 0; r < handoff.size(); ++r) if (handoff[r]) {
            Check(cudaSetDevice(topology.devices[r]));
            Check(cudaEventRecord(handoff[r], cudaStreamPerThread));
        }
        Check(cudaSetDevice(callerDevice));
        bool ok = MultiCudaRunDeviceCallbacksEnqueueOnly(topology.devices, [&](int rank, int device) {
            Executor *previous = (Executor*)GetExecutor();
            try {
                if (owners[rank] == std::thread::id{}) owners[rank] = std::this_thread::get_id();
                if (owners[rank] != std::this_thread::get_id()) throw std::logic_error("GPU worker changed");
                SetCurrentThreadExecutor(executors[rank].get());
                activeRuntime = this; activeRank = rank;
                if (handoff[rank]) Check(cudaStreamWaitEvent(cudaStreamPerThread, handoff[rank], 0));
                // Joining via an event also has to update the local operator
                // capture flag on the owning worker (it is thread-local).
                FastllmCudaGraphIsCapturing();
                callback(rank, device);
                Check(cudaGetLastError());
                if (FastllmCudaGetThreadError()) throw std::runtime_error("parallel local CUDA operator failed");
            } catch (...) { Fail(std::current_exception()); }
            // A local operator may temporarily select a tensor's device.
            // The worker registry records completion on this rank's event.
            try {
                Check(cudaSetDevice(device));
                if (completion[rank]) Check(cudaEventRecord(completion[rank], cudaStreamPerThread));
            } catch (...) { Fail(std::current_exception()); }
            activeRuntime = nullptr; activeRank = -1;
            SetCurrentThreadExecutor(previous);
        });
        if (!ok) throw std::runtime_error("persistent MultiCUDA workers unavailable");
        if (firstError) {
            for (auto &group : groups) for (auto &comm : group.comms) if (comm) {
                ncclCommAbort(comm); comm = nullptr;
            }
            std::rethrow_exception(firstError);
        }
        if (synchronize) {
            if (!MultiCudaSynchronizeDeviceCallbacks(topology.devices))
                throw std::runtime_error("parallel GPU completion failed");
        } else {
            for (size_t r = 0; r < completion.size(); ++r) {
                Check(cudaSetDevice(topology.devices[r]));
                Check(cudaStreamWaitEvent(cudaStreamPerThread, completion[r], 0));
            }
            Check(cudaSetDevice(callerDevice));
        }
    }
    template<class Function>
    void Collective(int id, int rank, size_t count, ScalarType type, uint64_t signature, Function fn) {
        if (activeRuntime != this || activeRank != rank) throw std::logic_error("collective outside owning worker");
        auto &description = topology.groups.at(id);
        int local = description.Rank(rank);
        if (local < 0) throw std::invalid_argument("rank is not in the collective group");
        auto &state = groups.at(id);
        if (description.ranks.size() == 1) { fn(nullptr); return; }
        // NCCL requires every member to agree on eager versus captured
        // submission. Include the actual worker stream state in the existing
        // rendezvous contract; replay then needs no host-side rendezvous.
        cudaStreamCaptureStatus capture;
        Check(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
        if (capture == cudaStreamCaptureStatusInvalidated)
            throw std::runtime_error("parallel collective capture invalidated");
        signature |= uint64_t(capture == cudaStreamCaptureStatusActive) << 63;
        if (!state.rendezvous->Wait(local, NcclSubmitRendezvous::Before, count, int(type), {}, signature))
            throw std::runtime_error(state.rendezvous->Error());
        fn(state.comms.at(local));
        if (!state.rendezvous->Wait(local, NcclSubmitRendezvous::After, count, int(type), {}, signature))
            throw std::runtime_error(state.rendezvous->Error());
    }
};

CudaRuntime::CudaRuntime(Topology topology) : impl(new Impl(std::move(topology))) {
    const int world = impl->topology.Size();
    impl->owners.resize(world);
    impl->handoff.resize(world, nullptr);
    impl->completion.resize(world, nullptr);
    for (int r = 0; r < world; ++r) {
        impl->executors.emplace_back(new Executor(true));
        impl->executors.back()->SetFirstDevice("cuda:" + std::to_string(impl->topology.devices[r]));
    }
    impl->Run([&](int rank, int) {
        Check(cudaEventCreateWithFlags(&impl->handoff[rank], cudaEventDisableTiming));
        Check(cudaEventCreateWithFlags(&impl->completion[rank], cudaEventDisableTiming));
    });
    impl->groups.resize(impl->topology.groups.size());
    for (size_t i = 0; i < impl->groups.size(); ++i) {
        const auto &group = impl->topology.groups[i];
        auto &state = impl->groups[i];
        state.comms.resize(group.ranks.size(), nullptr);
        state.rendezvous.reset(new NcclSubmitRendezvous(group.ranks.size()));
        if (group.ranks.size() == 1) continue;
        ncclUniqueId unique;
        Check(ncclGetUniqueId(&unique));
        impl->Run([&](int rank, int) {
            int local = group.Rank(rank);
            if (local >= 0) Check(ncclCommInitRank(&state.comms[local], group.ranks.size(), unique, local));
        });
    }
}
CudaRuntime::~CudaRuntime() = default;
const Topology& CudaRuntime::Layout() const { return impl->topology; }
uint64_t CudaRuntime::Generation() const { return impl->generation; }
void CudaRuntime::Run(const std::function<void(int,int)> &callback) { impl->Run(callback); }
void CudaRuntime::Enqueue(const std::function<void(int,int)> &callback) { impl->Run(callback, false); }
void CudaRuntime::AllReduce(int group, int rank, void *data, size_t count, ScalarType type) {
    if (count && !data) throw std::invalid_argument("null all-reduce buffer");
    impl->Collective(group, rank, count, type, 1, [&](ncclComm_t comm) {
        if (comm && count) Check(ncclAllReduce(data, data, count, Type(type), ncclSum, comm, cudaStreamPerThread));
    });
}
void CudaRuntime::AllGather(int group, int rank, const void *input, void *output, size_t count, ScalarType type) {
    if (count && (!input || !output)) throw std::invalid_argument("null all-gather buffer");
    impl->Collective(group, rank, count, type, 2, [&](ncclComm_t comm) {
        if (comm && count) Check(ncclAllGather(input, output, count, Type(type), comm, cudaStreamPerThread));
        else if (count && input != output) {
            const size_t unit = type == ScalarType::Byte ? 1 :
                (type == ScalarType::Float32 || type == ScalarType::Int32 ? 4 : 2);
            Check(cudaMemcpyAsync(output, input, Bytes(count, unit), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
    });
}
void CudaRuntime::Broadcast(int group, int rank, void *data, size_t count, ScalarType type, int root) {
    if (count && !data) throw std::invalid_argument("null broadcast buffer");
    if (root < 0 || root >= int(impl->topology.groups.at(group).ranks.size()))
        throw std::invalid_argument("invalid broadcast root");
    impl->Collective(group, rank, count, type, 3 + (uint64_t(root) << 32), [&](ncclComm_t comm) {
        if (comm && count) Check(ncclBroadcast(data, data, count, Type(type), root, comm, cudaStreamPerThread));
    });
}
struct CudaRowTransfer::Impl {
    struct Scratch {
        int *select = nullptr, *unpack = nullptr;
        void *send = nullptr, *gather = nullptr;
        size_t capacity = 0, selectCapacity = 0, unpackCapacity = 0;
    };
    CudaRuntime &runtime;
    RowPartition rows;
    std::vector<Scratch> scratch;
    size_t reserved = 0;
    Impl(CudaRuntime &runtime, RowPartition rows)
        : runtime(runtime), rows(std::move(rows)), scratch(runtime.Layout().Size()) {
        if (this->rows.localToGlobal.size() != size_t(runtime.Layout().attentionDP))
            throw std::invalid_argument("row partition does not match topology");
    }
    void Initialize() {
        runtime.Run([&](int rank, int) {
            auto &s = scratch[rank];
            const auto &indices = this->rows.localToGlobal[runtime.Layout().AttentionOwner(rank)];
            if (!indices.empty()) {
                if (s.selectCapacity < indices.size()) {
                    Check(cudaFree(s.select)); s.select = nullptr; s.selectCapacity = 0;
                    Check(cudaMalloc(&s.select, indices.size() * sizeof(int)));
                    s.selectCapacity = indices.size();
                }
                Check(cudaMemcpyAsync(s.select, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
            if (this->rows.Rows()) {
                if (s.unpackCapacity < size_t(this->rows.Rows())) {
                    Check(cudaFree(s.unpack)); s.unpack = nullptr; s.unpackCapacity = 0;
                    Check(cudaMalloc(&s.unpack, this->rows.Rows() * sizeof(int)));
                    s.unpackCapacity = this->rows.Rows();
                }
                Check(cudaMemcpyAsync(s.unpack, this->rows.globalToPadded.data(), this->rows.Rows() * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
        });
    }
    ~Impl() {
        runtime.impl->Release([&](int rank, int) {
            auto &s = scratch[rank];
            cudaFree(s.select); cudaFree(s.unpack); cudaFree(s.send); cudaFree(s.gather);
        });
    }
};
CudaRowTransfer::CudaRowTransfer(CudaRuntime &runtime, RowPartition rows) : impl(new Impl(runtime, std::move(rows))) {
    impl->Initialize();
}
CudaRowTransfer::~CudaRowTransfer() = default;
void CudaRowTransfer::Reset(RowPartition rows) {
    if (rows.localToGlobal.size() != size_t(impl->runtime.Layout().attentionDP))
        throw std::invalid_argument("row partition does not match topology");
    if (rows.localToGlobal == impl->rows.localToGlobal) return;
    impl->rows = std::move(rows);
    impl->Initialize();
}
void CudaRowTransfer::Reserve(size_t maxRowBytes) {
    const size_t capacity = Bytes(impl->rows.PaddedRows(), maxRowBytes);
    if (impl->reserved >= capacity) return;
    impl->runtime.Run([&](int rank, int) {
        auto &s = impl->scratch[rank];
        if (s.capacity >= capacity) return;
        Check(cudaFree(s.send)); Check(cudaFree(s.gather));
        s.send = s.gather = nullptr; s.capacity = 0;
        Check(cudaMalloc(&s.send, Bytes(impl->rows.bucket, maxRowBytes)));
        Check(cudaMalloc(&s.gather, capacity));
        s.capacity = capacity;
    });
    impl->reserved = capacity;
}
void CudaRowTransfer::Select(int rank, const void *global, void *local, size_t rowBytes) {
    if (activeRuntime != impl->runtime.impl.get() || activeRank != rank)
        throw std::logic_error("row select outside owning worker");
    auto &s = impl->scratch.at(rank);
    Copy(global, local, s.select, impl->rows.localToGlobal[impl->runtime.Layout().AttentionOwner(rank)].size(), rowBytes);
}
void CudaRowTransfer::Collect(int rank, const void *local, void *global, size_t rowBytes) {
    const Field field{local, global, rowBytes};
    Collect(rank, &field, 1);
}
void CudaRowTransfer::Collect(int rank, const Field *fields, size_t count) {
    if (activeRuntime != impl->runtime.impl.get() || activeRank != rank)
        throw std::logic_error("row collect outside owning worker");
    if (count && !fields) throw std::invalid_argument("missing row fields");
    const auto &topology = impl->runtime.Layout();
    auto &s = impl->scratch.at(rank);
    const auto &rows = impl->rows;
    const size_t localRows = rows.localToGlobal[topology.AttentionOwner(rank)].size();
    size_t rowBytes = 0;
    for (size_t i = 0; i < count; ++i) {
        if (fields[i].rowBytes > std::numeric_limits<size_t>::max() - rowBytes)
            throw std::overflow_error("row record size overflow");
        rowBytes += fields[i].rowBytes;
        if (fields[i].rowBytes && rows.Rows() &&
            (!fields[i].global || (localRows && !fields[i].local)))
            throw std::invalid_argument("missing row field buffer");
    }
    if (!rows.Rows() || !rowBytes) return;
    const size_t sendBytes = Bytes(rows.bucket, rowBytes), gatherBytes = Bytes(rows.PaddedRows(), rowBytes);
    if (s.capacity < gatherBytes) throw std::logic_error("row transfer scratch not reserved");
    if (topology.AttentionRank(rank) == 0) {
        if (localRows < size_t(rows.bucket))
            Check(cudaMemsetAsync(static_cast<unsigned char *>(s.send) + localRows * rowBytes, 0,
                (rows.bucket - localRows) * rowBytes, cudaStreamPerThread));
        size_t offset = 0;
        for (size_t i = 0; i < count; ++i) {
            CopyField(fields[i].local, static_cast<unsigned char *>(s.send) + offset, nullptr,
                localRows, fields[i].rowBytes, fields[i].rowBytes, rowBytes);
            offset += fields[i].rowBytes;
        }
        impl->runtime.AllGather(topology.leaders, rank, s.send, s.gather, sendBytes, ScalarType::Byte);
    }
    impl->runtime.Broadcast(topology.attention[topology.AttentionOwner(rank)], rank, s.gather, gatherBytes, ScalarType::Byte);
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) {
        CopyField(static_cast<unsigned char *>(s.gather) + offset, fields[i].global, s.unpack,
            rows.Rows(), fields[i].rowBytes, rowBytes, fields[i].rowBytes);
        offset += fields[i].rowBytes;
    }
}

}} // namespace fastllm::parallel
