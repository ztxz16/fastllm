// Eight-row MXFP4 verification. All weights, tags and mailboxes are borrowed.
#include <cutlass/version.h>
#include "verify8/api.cuh"
#include "fastllm-cuda-launch.cuh"
#include "verify8/gate.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
namespace fastllm_verify8 {
namespace sel = moe_select_gate_up;

constexpr int kThreads = 384, kGrid = 296, kShared = 108 * 1024;
constexpr int kTmemSlotOffset = 106752, kTmemOwnerWarp = 10, kEarlyDownColumns = 256;
using Args = FastllmVerify8Args;
#include "verify8/down-layout.cuh"
#include "verify8/down.cuh"
// Interleave tokens across physical CTAs so each pair's two owner groups
// quantize activations that their partner groups reuse.
__device__ int token_interleave_tile(int base, int group) {
    const int job = base + group * kGrid;
    return job < 512 ? (job % 8) * 64 + job / 8 : 512;
}
template <bool Pages, bool Preselected>
__global__ __launch_bounds__(kThreads, 2) void moe_gate_up_down_rows8(const __grid_constant__ Args args) {
    extern __shared__ __align__(1024) unsigned char shared[];
    if (threadIdx.x >= sel::kRows8HelperBaseThread)
        cutlass::arch::warpgroup_reg_dealloc<32>();
    using Select = sel::SelectGateUpBody<Pages, Preselected>;
    unsigned generation_tag;
    Select::template run<kEarlyDownColumns, kTmemSlotOffset, kTmemOwnerWarp>(
        args, shared, args.xq, args.tagged_up, generation_tag, args.epoch, shared + moe_down::kRouteScratch);
    // Select ends with the full-CTA register handoff barrier.
    if (threadIdx.x < 2) {
        naivert::Barrier::invalidate(Select::barriers(shared) + threadIdx.x);
        naivert::Barrier::invalidate(Select::release_barriers(shared) + threadIdx.x);
    }
    if (threadIdx.x == 0) {
        naivert::Barrier::invalidate(Select::activation_barriers(shared));
        naivert::Barrier::invalidate(Select::weight_ids_ready(shared));
        naivert::Barrier::invalidate(Select::route_barrier(shared));
        naivert::Barrier::invalidate(Select::route_release_barrier(shared));
    }
    naivert::pdl::trigger_dependents<true>();
    const auto *routes = reinterpret_cast<const sel::PreparedRouteScratch *>(shared + moe_down::kRouteScratch);

    auto *common_ready = reinterpret_cast<std::uint64_t *>(shared + moe_down::kCommonReady);
    if (threadIdx.x < 8)
        naivert::Barrier::init(common_ready + threadIdx.x, 128);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    auto *tmem_slot = reinterpret_cast<unsigned *>(shared + kTmemSlotOffset);
    const bool active = threadIdx.x < 320;
    const int group = moe_down::group_id();
    auto *local_shared = shared + group * moe_down::kGroupStride;
    if (active)
        down_per_token::initialize(local_shared, moe_down::group_tid());
    if (threadIdx.x == 0) {
        const int gate_remainder = (routes->count * 16) % kGrid;
        tmem_slot[1] = unsigned((int(blockIdx.x) + kGrid - gate_remainder) % kGrid);
    }
    __syncthreads();
    const int down_base = int(tmem_slot[1]);
    if (active)
        down_per_token::run(args, local_shared, *tmem_slot + group * moe_down::kPerTokenTmemGroupColumns,
                            token_interleave_tile(down_base, group), routes->selected, generation_tag);
    __syncthreads();
    if (threadIdx.x < 8)
        naivert::Barrier::invalidate(common_ready + threadIdx.x);
    if (threadIdx.x >= kTmemOwnerWarp * 32 && threadIdx.x < (kTmemOwnerWarp + 1) * 32) {
        cute::TMEM::Allocator1Sm allocator;
        allocator.free(*tmem_slot, kEarlyDownColumns);
    }
}
__global__ __launch_bounds__(128, 16) void moe_tp_reduce_rows8(const __grid_constant__ Args args) {
    __shared__ unsigned received[8 * 32];
    const int tid = threadIdx.x, lane = tid & 31, peer_group = tid / 32, rank = args.rank;
    const int token = blockIdx.x / 64, tile = blockIdx.x % 64;
    naivert::pdl::trigger_dependents<true>();
    unsigned generation = moe_down::generation(args);
    const unsigned long long *source[moe_down::kPeersPerWarp];
    unsigned long long word[moe_down::kPeersPerWarp];
#pragma unroll
    for (int local = 0; local < moe_down::kPeersPerWarp; ++local) {
        const int peer = peer_group * moe_down::kPeersPerWarp + local;
        source[local] = moe_down::mailbox(args, generation, rank, peer, token, tile, lane);
        word[local] = naivert::load_tagged<false>(source[local]);
    }
    static_assert(moe_down::kPeersPerWarp == 2, "pending checks both words");
    bool pending;
    do {
#pragma unroll
        for (int local = 0; local < moe_down::kPeersPerWarp; ++local)
            if (naivert::tagged_tag(word[local]) != generation)
                word[local] = naivert::load_tagged<false>(source[local]);
        pending = (naivert::tagged_tag(word[0]) != generation) | (naivert::tagged_tag(word[1]) != generation);
    } while (pending);
#pragma unroll
    for (int local = 0; local < moe_down::kPeersPerWarp; ++local)
        received[(peer_group * moe_down::kPeersPerWarp + local) * 32 + lane] = naivert::tagged_payload(word[local]);
    __syncthreads();
    if (peer_group != 0 || token >= args.rows)
        return;
    float2 sum = make_float2(0.f, 0.f);
#pragma unroll
    for (int source = 0; source < 8; ++source) {
        const unsigned packed = received[source * 32 + lane];
        const float2 value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&packed));
        sum.x = __fadd_rn(sum.x, value.x);
        sum.y = __fadd_rn(sum.y, value.y);
    }
    reinterpret_cast<__nv_bfloat162 *>(args.out)[token * 2048 + tile * 32 + lane] = __floats2bfloat162_rn(sum.x, sum.y);
}

} // namespace fastllm_verify8
static bool supports(int device, const void *kernel) {
    using namespace fastllm_verify8;
    cudaDeviceProp prop{};
    cudaFuncAttributes attr{};
    return cudaGetDeviceProperties(&prop, device) == cudaSuccess && prop.major == 10 && prop.minor == 3 &&
           prop.multiProcessorCount == 148 &&
           cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) == cudaSuccess &&
           cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.binaryVersion == 103 && attr.numRegs <= 80 &&
           prop.regsPerMultiprocessor >= 2 * kThreads * 80 && prop.maxThreadsPerMultiProcessor >= 2 * kThreads &&
           prop.sharedMemPerMultiprocessor >= 2 * (kShared + attr.sharedSizeBytes + prop.reservedSharedMemPerBlock);
}
bool FastllmVerify8Supported(int device) {
    return supports(device, (const void *)fastllm_verify8::moe_gate_up_down_rows8<false, false>);
}
bool FastllmVerify8PagesSupported(int device) {
    using namespace fastllm_verify8;
    return supports(device, (const void *)moe_gate_up_down_rows8<true, false>) &&
           supports(device, (const void *)moe_gate_up_down_rows8<true, true>);
}
cudaError_t FastllmVerify8Launch(const FastllmVerify8Args &args, cudaStream_t stream) {
    using namespace fastllm_verify8;
    if (!args.gatePages)
        moe_gate_up_down_rows8<false, false><<<kGrid, kThreads, kShared, stream>>>(args);
    else if (args.router) {
        auto err = fastllm::cuda_launch::LaunchProgrammatic<moe_gate_up_down_rows8<true, false>>(kGrid, kThreads,
                                                                                                 kShared, stream, args);
        if (err != cudaSuccess)
            return err;
    } else
        moe_gate_up_down_rows8<true, true><<<kGrid, kThreads, kShared, stream>>>(args);
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        return error;
    return fastllm::cuda_launch::LaunchProgrammatic<moe_tp_reduce_rows8>(512, 128, 0, stream, args);
}
