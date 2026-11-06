#include "distributed/process_allreduce.h"
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace fastllm { namespace distributed {
namespace {
constexpr size_t kAllocation = 2 * 1024 * 1024; // IPC 独占 2 MiB，避免暴露相邻小分配。
constexpr size_t kDataOffset = 4096;
constexpr size_t kMaxElements = 65536;
constexpr int kBlocks = 16;

struct alignas(128) Signals {
    unsigned long long ready[kBlocks];
    unsigned long long done[kBlocks];
    unsigned long long ncclCalls;
};
static_assert(sizeof(Signals) <= kDataOffset, "signal/payload overlap");
static_assert(kDataOffset + kMaxElements * sizeof(float) <= kAllocation, "IPC payload overflow");

/** CUDA/NCCL 失败交由 worker 的退出路径回收；不继续使用半初始化通信器。 */
void Cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("process AllReduce CUDA: ") + cudaGetErrorString(status));
}
/** NCCL 错误必须终止整组，不能在单个 rank 上重试同一次集体操作。 */
void Nccl(ncclResult_t status) {
    if (status != ncclSuccess) throw std::runtime_error(std::string("process AllReduce NCCL: ") + ncclGetErrorString(status));
}

/** 系统作用域 acquire/release；仅在通过 SM70+ 与 P2P 能力检查后执行。 */
__device__ unsigned long long Acquire(const unsigned long long *address) {
    unsigned long long value = 0;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
#else
    asm volatile("trap;");
#endif
    return value;
}
/** 系统作用域 release 发布；与对端 acquire 配对，不用 volatile 代替内存序。 */
__device__ void Release(unsigned long long *address, unsigned long long value) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(address), "l"(value) : "memory");
#else
    asm volatile("trap;");
#endif
}

/** 对端 payload 绕过非一致的 L1；可见性由前置系统 acquire 建立。 */
__device__ float Read(const float *address) {
    float value;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(value) : "l"(address) : "memory");
    return value;
}

/**
 * 两卡一阶段规约；输入是已经按 stream 顺序复制好的固定 IPC FP32 缓冲。
 * 每个 block 独立握手并处理固定模 stride 的元素；所有调用固定 16 blocks，
 * 包括小消息的空 block。每个 block 等对端读完对应分片才结束，因此整个
 * kernel 完成后才能安全覆盖下一轮 payload。序号由 GPU 更新，Graph 重放
 * 不复用 host 常量；使用 >= 等待允许对端先进入下一轮，避免漏掉旧信号。
 * @param self 本 rank 的 GPU 信号；只由本 rank 写入，同 stream 串行复用。
 * @param peer 经 IPC 导入的对端信号；系统 acquire 等待，故障由父进程超时回收。
 * @param first rank 0 的连续 FP32[count] staging 数据，与 output 不重叠。
 * @param second rank 1 的连续 FP32[count] staging 数据，与 output 不重叠。
 * @param output 本 rank 的连续 FP32[count] 输出，每个元素只由一个线程写入。
 * @param count 1..65536；两个 rank 必须使用相同值、grid 和 block 配置。
 */
__global__ void Reduce(Signals *self, Signals *peer, const float *first,
                       const float *second, float *output, size_t count) {
    const int block = blockIdx.x;
    __shared__ unsigned long long epoch;
    if (threadIdx.x == 0) {
        epoch = Acquire(&self->ready[block]) + 1;
        Release(&self->ready[block], epoch);
        while (Acquire(&peer->ready[block]) < epoch) {}
    }
    __syncthreads();

    for (size_t i = block * blockDim.x + threadIdx.x; i < count; i += kBlocks * blockDim.x) {
        output[i] = Read(first + i) + Read(second + i);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        Release(&self->done[block], epoch);
        while (Acquire(&peer->done[block]) < epoch) {}
    }
}

/** GPU 计数节点随 Graph 重放执行，不能以 host 的捕获次数冒充实际规约次数。 */
__global__ void CountNccl(Signals *signals) {
    ++signals->ncclCalls;
}
} // namespace

struct ProcessAllReduce::Impl {
    cudaStream_t stream;
    ncclComm_t comm;
    bool custom;
    int rank = 0;
    void *local = nullptr, *peer = nullptr;
    float *barrier = nullptr;

    /** 所有 rank 同 stream 汇合；仅用于资源生命周期，不计入模型通信次数。 */
    void Barrier() {
        Cuda(cudaMemsetAsync(barrier, 0, sizeof(float), stream));
        Nccl(ncclAllReduce(barrier, barrier, 1, ncclFloat, ncclSum, comm, stream));
        Cuda(cudaStreamSynchronize(stream));
    }
};

ProcessAllReduce::ProcessAllReduce(cudaStream_t stream, ncclComm_t comm, bool custom)
    : impl(new Impl{stream, comm, custom}) {
    auto &r = *impl;
    int world, device;
    Nccl(ncclCommCount(comm, &world));
    Nccl(ncclCommUserRank(comm, &r.rank));
    Cuda(cudaGetDevice(&device));
    cudaDeviceProp properties;
    Cuda(cudaGetDeviceProperties(&properties, device));
    if (custom && (world != 2 || properties.major < 7 || !properties.unifiedAddressing ||
                   properties.multiProcessorCount < kBlocks)) {
        throw std::runtime_error("custom AllReduce requires TP=2, SM70+, CUDA UVA and at least 16 SMs");
    }
    if (custom) {
        cudaFuncAttributes attributes{};
        Cuda(cudaFuncGetAttributes(&attributes, Reduce));
        if (attributes.ptxVersion < 70) {
            throw std::runtime_error("custom AllReduce requires kernels compiled for compute_70 or newer");
        }
    }
    Cuda(cudaMalloc(&r.local, custom ? kAllocation : sizeof(Signals)));
    Cuda(cudaMemsetAsync(r.local, 0, custom ? kAllocation : sizeof(Signals), stream));
    Cuda(cudaMalloc(reinterpret_cast<void**>(&r.barrier), sizeof(float)));
    if (!custom) {
        Cuda(cudaStreamSynchronize(stream));
        return;
    }

    // NCCL 只负责引导交换；句柄按 GPU UUID 配对，不能传递进程私有的裸地址。
    struct Entry { cudaIpcMemHandle_t handle; cudaUUID_t uuid; } entry{}, entries[2]{};
    entry.uuid = properties.uuid;
    Cuda(cudaIpcGetMemHandle(&entry.handle, r.local));
    void *exchange = nullptr;
    Cuda(cudaMalloc(&exchange, 3 * sizeof(Entry)));
    auto receive = static_cast<char*>(exchange) + sizeof(Entry);
    Cuda(cudaMemcpyAsync(exchange, &entry, sizeof(entry), cudaMemcpyHostToDevice, stream));
    Nccl(ncclAllGather(exchange, receive, sizeof(Entry), ncclUint8, comm, stream));
    Cuda(cudaMemcpyAsync(entries, receive, sizeof(entries), cudaMemcpyDeviceToHost, stream));
    Cuda(cudaStreamSynchronize(stream));
    Cuda(cudaFree(exchange));

    int devices, peerDevice = -1;
    Cuda(cudaGetDeviceCount(&devices));
    for (int candidate = 0; candidate < devices; ++candidate) {
        cudaDeviceProp prop;
        Cuda(cudaGetDeviceProperties(&prop, candidate));
        if (!std::memcmp(prop.uuid.bytes, entries[1 - r.rank].uuid.bytes, sizeof(prop.uuid.bytes))) peerDevice = candidate;
    }
    int access = 0, reverse = 0, atomics = 0, reverseAtomics = 0;
    if (peerDevice >= 0 && peerDevice != device) {
        Cuda(cudaDeviceCanAccessPeer(&access, device, peerDevice));
        Cuda(cudaDeviceCanAccessPeer(&reverse, peerDevice, device));
        Cuda(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, device, peerDevice));
        Cuda(cudaDeviceGetP2PAttribute(&reverseAtomics, cudaDevP2PAttrNativeAtomicSupported, peerDevice, device));
    }
    if (!access || !reverse || !atomics || !reverseAtomics) {
        throw std::runtime_error("custom AllReduce requires visible peer GPUs with bidirectional P2P/native atomics; select NCCL instead");
    }
    Cuda(cudaIpcOpenMemHandle(&r.peer, entries[1 - r.rank].handle, cudaIpcMemLazyEnablePeerAccess));
    r.Barrier();
}

/** 异常路径禁止隐式集体析构；父进程终止全部 worker 并回收 CUDA 上下文。 */
ProcessAllReduce::~ProcessAllReduce() = default;

void ProcessAllReduce::Sum(float *data, size_t count) {
    auto &r = *impl;
    if (!r.local || !data || !count) throw std::invalid_argument("invalid process AllReduce input/state");
    if (r.custom && count <= kMaxElements) {
        auto localData = reinterpret_cast<float*>(static_cast<char*>(r.local) + kDataOffset);
        auto peerData = reinterpret_cast<float*>(static_cast<char*>(r.peer) + kDataOffset);
        Cuda(cudaMemcpyAsync(localData, data, count * sizeof(float), cudaMemcpyDeviceToDevice, r.stream));
        Reduce<<<kBlocks,256,0,r.stream>>>(static_cast<Signals*>(r.local), static_cast<Signals*>(r.peer),
            r.rank == 0 ? localData : peerData, r.rank == 0 ? peerData : localData, data, count);
    } else {
        CountNccl<<<1,1,0,r.stream>>>(static_cast<Signals*>(r.local));
        Nccl(ncclAllReduce(data, data, count, ncclFloat, ncclSum, r.comm, r.stream));
    }
    Cuda(cudaGetLastError());
}

ProcessAllReduceInfo ProcessAllReduce::Info() {
    auto &r = *impl;
    if (!r.local) throw std::runtime_error("process AllReduce is closed");
    Signals signals{};
    Cuda(cudaMemcpyAsync(&signals, r.local, sizeof(signals), cudaMemcpyDeviceToHost, r.stream));
    Cuda(cudaStreamSynchronize(r.stream));
    for (int block = 0; block < kBlocks; ++block) {
        if (signals.ready[block] != signals.ready[0] || signals.done[block] != signals.ready[0]) {
            throw std::runtime_error("incomplete custom AllReduce barrier");
        }
    }
    return {r.custom, r.peer != nullptr, signals.ready[0], signals.ncclCalls,
            (r.custom ? kAllocation : sizeof(Signals)) + sizeof(float)};
}

void ProcessAllReduce::Close() {
    auto &r = *impl;
    if (!r.local) return;
    r.Barrier(); // 全部 stream 完成，才能关闭本进程导入的对端映射。
    if (r.peer) {
        Cuda(cudaIpcCloseMemHandle(r.peer));
        r.peer = nullptr;
    }
    r.Barrier(); // 所有映射都已关闭，才能释放 owner 分配。
    Cuda(cudaFree(r.local));
    r.local = nullptr;
    Cuda(cudaFree(r.barrier));
    r.barrier = nullptr;
}

}} // namespace fastllm::distributed
