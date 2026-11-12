#pragma once
#include "fastllm-kernel-benchmark.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>

namespace fastllm_topk_logits {
constexpr int Threads = 256, Capacity = FastllmCudaLogitsTopKCapacity, RowTile = 32;
using Candidate = FastllmCudaTopKCandidate;
struct Rows { FastllmCudaTopKRow row[RowTile]; };
struct Workspace {
    Candidate *first, *second;
    int *invalid;
    size_t bytes;
    Workspace(void *base, int batch, int vocab) {
        const size_t blocks = (size_t(vocab) + 1023) / 1024;
        const size_t rows = std::min(batch, RowTile);
        const size_t pairs = rows * blocks * Capacity * sizeof(Candidate);
        bytes = 2 * pairs + ((rows * blocks * sizeof(int) + 255) / 256) * 256;
        first = static_cast<Candidate *>(base);
        second = base ? reinterpret_cast<Candidate *>(static_cast<char *>(base) + pairs) : nullptr;
        invalid = base ? reinterpret_cast<int *>(static_cast<char *>(base) + 2 * pairs) : nullptr;
    }
};

// CUB's stable block sort retains token order at score ties. Every merge
// consumes consecutive original token ranges, so this remains true at every
// level. No full-vocabulary sort or serial per-token heap is needed.
template<int Items>
__global__ void SelectInput(Rows rows, int vocab, int blocks, Candidate *output, int *invalid) {
    using Sort = cub::BlockRadixSort<float, Threads, Items, int>;
    using Reduce = cub::BlockReduce<int, Threads>;
    __shared__ union { typename Sort::TempStorage sort; typename Reduce::TempStorage reduce; } storage;
    float scores[Items]; int ids[Items], bad = 0;
    const auto request = rows.row[blockIdx.y];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int token = blockIdx.x * Threads * Items + threadIdx.x * Items + i;
        float value = token < vocab ? __fmul_rn(request.logits[token], request.inverseTemperature) : -INFINITY;
        bad |= isnan(value) || value == INFINITY;
        scores[i] = isnan(value) ? -INFINITY : value == 0.f ? 0.f : value;
        ids[i] = token < vocab ? token : INT_MAX;
    }
    bad = Reduce(storage.reduce).Sum(bad);
    if (!threadIdx.x) invalid[blockIdx.y * blocks + blockIdx.x] = bad;
    __syncthreads();
    Sort(storage.sort).SortDescending(scores, ids);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = threadIdx.x * Items + i;
        if (rank < Capacity) {
            int token = ids[i] == INT_MAX ? -1 : ids[i];
            if (blocks == 1 && rank == 0 && (bad || !isfinite(scores[i]))) token = -1;
            output[(size_t(blockIdx.y) * blocks + blockIdx.x) * Capacity + rank] = {scores[i], token};
        }
    }
}

template<int Items>
__global__ void MergeCandidates(const Candidate *input, int inputCount, int blocks,
        int vocabBlocks, const int *invalid, Candidate *output) {
    using Sort = cub::BlockRadixSort<float, Threads, Items, int>;
    __shared__ typename Sort::TempStorage storage;
    float scores[Items]; int ids[Items];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int offset = blockIdx.x * Threads * Items + threadIdx.x * Items + i;
        Candidate value = offset < inputCount ? input[size_t(blockIdx.y) * inputCount + offset]
            : Candidate{-INFINITY, -1};
        scores[i] = value.score; ids[i] = value.token;
    }
    Sort(storage).SortDescending(scores, ids);
    int bad = 0;
    if (blocks == 1 && !threadIdx.x)
        for (int i = 0; i < vocabBlocks; ++i) bad |= invalid[blockIdx.y * vocabBlocks + i];
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = threadIdx.x * Items + i;
        if (rank < Capacity) {
            int token = ids[i];
            if (blocks == 1 && rank == 0 && (bad || !isfinite(scores[i]))) token = -1;
            output[(size_t(blockIdx.y) * blocks + blockIdx.x) * Capacity + rank] = {scores[i], token};
        }
    }
}

struct Operation {
    const FastllmCudaTopKRow *rows;
    int batch, vocab;
    Candidate *output;
    Workspace workspace;
    cudaStream_t stream;
};
template<int Items>
bool Run(const Operation &c) {
    constexpr int Tile = Threads * Items;
    const int vocabBlocks = (int64_t(c.vocab) + Tile - 1) / Tile;
    for (int begin = 0; begin < c.batch;) {
        const int count = std::min(RowTile, c.batch - begin);
        Rows rows{};
        for (int i=0;i<count;++i) rows.row[i] = c.rows[begin+i];
        Candidate *a = c.workspace.first, *b = c.workspace.second;
        Candidate *out = c.output + size_t(begin) * Capacity;
        SelectInput<Items><<<dim3(vocabBlocks,count),Threads,0,c.stream>>>(
            rows, c.vocab, vocabBlocks, vocabBlocks == 1 ? out : a, c.workspace.invalid);
        int blocks = vocabBlocks;
        while (blocks > 1) {
            const int next = (int64_t(blocks) * Capacity + Tile - 1) / Tile;
            MergeCandidates<Items><<<dim3(next,count),Threads,0,c.stream>>>(
                a, blocks * Capacity, next, vocabBlocks, c.workspace.invalid, next == 1 ? out : b);
            blocks = next; std::swap(a,b);
        }
        begin += count;
    }
    return cudaGetLastError() == cudaSuccess;
}
bool Run(void *opaque, int implementation) {
    const auto &c = *static_cast<Operation *>(opaque);
    if (implementation == 1) return Run<8>(c);
    if (implementation == 2) return Run<16>(c);
    return Run<4>(c);
}

using Json = json11::Json;
const fastllm::kernels::Family family{9,"sampling.logits_topk",__DATE__ " " __TIME__,{
    {"cuda.sampling.topk.tile1024",Json::object{{"backend","cuda"},{"operation","stable partial selection, 1024 values per tile"},{"rows","arbitrary; bounded 32-row scratch tiles"},{"top_k","1..64"},{"architecture","portable CUDA block primitives"}}},
    {"cuda.sampling.topk.tile2048",Json::object{{"backend","cuda"},{"operation","stable partial selection, 2048 values per tile"},{"rows","arbitrary; bounded 32-row scratch tiles"},{"top_k","1..64"},{"architecture","portable CUDA block primitives"}}},
    {"cuda.sampling.topk.tile4096",Json::object{{"backend","cuda"},{"operation","stable partial selection, 4096 values per tile"},{"rows","arbitrary; bounded 32-row scratch tiles"},{"top_k","1..64"},{"architecture","portable CUDA block primitives"}}}
}};
const bool registered = [] { fastllm::kernels::Register(&family); return true; }();
std::string Unsupported(void *, int) { return ""; }
double Measure(void *opaque, int impl, int repeats) {
    auto &c = *static_cast<Operation *>(opaque);
    struct Resources {
        cudaStream_t stream;
        cudaEvent_t a=nullptr,b=nullptr;
        cudaGraph_t graph=nullptr;
        cudaGraphExec_t exec=nullptr;
        bool capturing=false;
        ~Resources() {
            if (capturing && cudaStreamEndCapture(stream,&graph)!=cudaSuccess) cudaGetLastError();
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            if (a) cudaEventDestroy(a);
            if (b) cudaEventDestroy(b);
        }
    } resources{c.stream};
    using fastllm::kernels::CudaCheck;
    CudaCheck(cudaEventCreate(&resources.a)); CudaCheck(cudaEventCreate(&resources.b));
    CudaCheck(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeThreadLocal));
    resources.capturing=true;
    const bool ok=Run(opaque,impl);
    const auto ended=cudaStreamEndCapture(c.stream,&resources.graph);resources.capturing=false;
    CudaCheck(ended);
    if (!ok) return -1.;
    CudaCheck(cudaGraphInstantiate(&resources.exec,resources.graph,nullptr,nullptr,0));
    CudaCheck(cudaGraphLaunch(resources.exec,c.stream));
    CudaCheck(cudaEventRecord(resources.a,c.stream));
    for (int i=0;i<repeats;++i) CudaCheck(cudaGraphLaunch(resources.exec,c.stream));
    CudaCheck(cudaEventRecord(resources.b,c.stream)); CudaCheck(cudaEventSynchronize(resources.b));
    float ms; CudaCheck(cudaEventElapsedTime(&ms,resources.a,resources.b));
    return ms/repeats;
}
bool Validate(void *opaque, int impl, double &l2, std::string &reason) {
    auto &c = *static_cast<Operation *>(opaque);
    const size_t bytes = size_t(c.batch) * Capacity * sizeof(Candidate);
    std::vector<unsigned char> original(bytes), reference(bytes), actual(bytes);
    using fastllm::kernels::CudaCheck;
    auto read = [&](auto &v) {
        CudaCheck(cudaMemcpyAsync(v.data(),c.output,bytes,cudaMemcpyDeviceToHost,c.stream));
        CudaCheck(cudaStreamSynchronize(c.stream));
    };
    read(original);
    bool ok=Run(opaque,0); read(reference);
    ok=Run(opaque,impl)&&ok; read(actual);
    CudaCheck(cudaMemcpyAsync(c.output,original.data(),bytes,cudaMemcpyHostToDevice,c.stream));
    CudaCheck(cudaStreamSynchronize(c.stream));
    l2=0.;
    if (!ok || reference != actual) { reason="candidate scores or token/tie order differ"; return false; }
    return true;
}
} // namespace fastllm_topk_logits

size_t FastllmCudaLogitsTopKWorkspaceBytes(int batch, int vocab) {
    if (batch <= 0 || vocab <= 0) return 0;
    return fastllm_topk_logits::Workspace(nullptr,batch,vocab).bytes;
}
bool FastllmCudaLogitsTopK(const FastllmCudaTopKRow *rows, int batch, int vocab,
        FastllmCudaTopKCandidate *output, void *workspace, size_t workspaceBytes, void *stream) {
    using namespace fastllm_topk_logits;
    const size_t bytes = FastllmCudaLogitsTopKWorkspaceBytes(batch,vocab);
    if (!rows || !output || !workspace || !bytes || workspaceBytes < bytes || uintptr_t(workspace)%256) return false;
    for (int i=0;i<batch;++i)
        if (!rows[i].logits || !std::isfinite(rows[i].inverseTemperature) || rows[i].inverseTemperature <= 0.f) return false;
    Operation op{rows,batch,vocab,output,Workspace(workspace,batch,vocab),
        stream ? static_cast<cudaStream_t>(stream) : cudaStreamPerThread};
    int selected = 0;
    if (fastllm::kernels::Enabled()) {
        fastllm::kernels::Signature sig;
        sig.fields={9,FastllmCudaRuntimeArch(),FastllmCudaGetDevice(),batch,vocab,1,Capacity};
        cudaStreamCaptureStatus status; fastllm::kernels::CudaCheck(cudaStreamIsCapturing(op.stream,&status));
        selected=fastllm::kernels::Select({&family,sig,&op,
            status != cudaStreamCaptureStatusNone || FastllmCudaGraphMemoryPoolActive(),
            Unsupported,Measure,Validate,fastllm::kernels::CudaHardware,0});
    }
    return Run(&op,std::max(0,selected));
}
