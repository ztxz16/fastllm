#include <cutlass/version.h>
// Optional native SM103 Flex backend using the pinned NaiveRT arithmetic subset.
// FastLLM supplies the row scheduler and ownership adapter; provenance and
// deployment boundaries are documented in docs/naive-moe-backends.md.
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "naivert/body/expert_down_allreduce/expert_down_allreduce_body.cuh"
#include "naivert/body/expert_select_up_gate_silu_rows8_body.cuh"
#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_body.cuh"

namespace {
namespace down = naivert_replica_expert_down_allreduce;
namespace gate = naivert_replica_expert_select_up_gate_silu;
using Down = down::ExpertDownAllreduceArgs;
using Domain = down::ExpertDownRuntimeDomain;
using Gate = gate::ExpertSelectUpGateSiluArgs;

__device__ __forceinline__ Down row_args(Down a, int row) {
  a.up_gate += row * 8 * 256;
  a.sel_indices += row * 8;
  a.sel_probs += row * 8;
  a.residual += row * 4096;
  a.exp_out += row * 4096;
  a.tp_result += row * 4096;
  a.stable_domain += row;
  return a;
}

__global__ __launch_bounds__(288, 1)
void small_down1(const __grid_constant__ Down args) {
  extern __shared__ __align__(128) unsigned char sm[];
  down::ExpertDownAllreduceDecodeBody::run(args, sm);
}

__global__ __launch_bounds__(288, 4)
void small_down_rows(const __grid_constant__ Down args) {
  extern __shared__ __align__(128) unsigned char sm[];
  down::ExpertDownAllreduceBody::run(row_args(args, blockIdx.y), sm);
}

__global__ __launch_bounds__(384, 1)
void gate_row1(const __grid_constant__ Gate args) {
  extern __shared__ __align__(128) unsigned char sm[];
  gate::ExpertSelectUpGateSiluBody::run<false>(args,sm);
}

__global__ __launch_bounds__(384, 1)
void gate_rows8_tiled(const __grid_constant__ Gate base) {
  extern __shared__ __align__(128) unsigned char sm[];
  Gate a = base;
  const int row = blockIdx.y * 8;
  a.seq_len=min(8,base.seq_len-row);
  a.x += row * 4096;
  a.scores += row * 256;
  a.y += row * 8 * 256;
  a.expert_probs += row * 8;
  a.expert_indices += row * 8;
  a.total_tiles = gridDim.x;
  a.progress.total_tiles = gridDim.x;
  a.progress.active_ctas = gridDim.x;
  naivert_mtp_expert_select_rows8::ExpertSelectUpGateSiluRows8Body::run(a, sm);
}

static Down make_down(const uint64_t* p, int rank) {
  Down a{};
  a.up_gate = (const float*)p[0]; a.moe_w_down = (const __nv_fp8_e4m3*)p[1];
  a.moe_w_down_scale = (const float*)p[2]; a.sel_indices = (const int*)p[3];
  a.sel_probs = (const float*)p[4]; a.residual = (const __nv_bfloat16*)p[5];
  a.exp_out = (__nv_bfloat16*)p[6]; a.tp_result = a.exp_out;
  a.stable_domain = (const Domain*)p[7];
  a.dim=4096; a.moe_inter_dim=256; a.n_activated=8; a.rank=rank;
  return a;
}

static int LaunchDown(const uint64_t *p, int rows, int rank, cudaStream_t stream) {
  if (!p || rows < 1 || rows > 8 || rank < 0 || rank >= 8) return cudaErrorInvalidValue;
  Down a = make_down(p, rank);
  if (rows == 1) {
    auto error = cudaFuncSetAttribute(small_down1, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (error != cudaSuccess) return error;
    small_down1<<<148,288,49152,stream>>>(a);
  } else {
    auto error = cudaFuncSetAttribute(small_down_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, 49152);
    if (error != cudaSuccess) return error;
    small_down_rows<<<dim3(64,rows),288,49152,stream>>>(a);
  }
  return cudaGetLastError();
}

static int LaunchGate(const uint64_t* p, int rows, cudaStream_t stream) {
  if(!p || rows < 1 || rows > 8) return cudaErrorInvalidValue;
  const int groups=(rows+7)/8;
  Gate a{};
  a.x=(const __nv_bfloat16*)p[0]; a.scores=(const float*)p[1]; a.bias=(const float*)p[2];
  a.packed_weights=(const __nv_fp8_e4m3*)p[3]; a.y=(float*)p[4];
  a.expert_probs=(float*)p[5]; a.expert_indices=(int*)p[6];
  a.hidden=4096; a.routed_experts=256; a.total_experts=256; a.selected_experts=8;
  a.output_slots=8; a.output_groups=16; a.outputs_per_group=16;
  a.seq_len=rows; a.pages=4; a.page_k=1024;
  a.x_stride=4096; a.y_slot_stride=256; a.y_group_stride=16;
  a.packed_group_stride=16; a.packed_page_stride=gate::kPackedPageStride;
  a.packed_expert_stride=16*4*gate::kPackedPageStride; a.route_scale=1.0f;
  if(rows==1){
    a.total_tiles=128;a.progress.total_tiles=128;a.progress.active_ctas=128;
    cudaFuncSetAttribute(gate_row1,cudaFuncAttributeMaxDynamicSharedMemorySize,gate::kSharedBytes);
    cudaFuncSetAttribute(gate_row1,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    gate_row1<<<128,384,gate::kSharedBytes,stream>>>(a);
    return cudaGetLastError();
  }
  // Avoid launching just over one SM wave (e.g. 19*8 = 152 on 148 SMs),
  // which leaves a long tail of only four heavyweight CTAs.
  int ctas=std::max(8,148/groups);
  ctas=std::min(148,ctas);
  cudaFuncSetAttribute(gate_rows8_tiled,cudaFuncAttributeMaxDynamicSharedMemorySize,
                      naivert_mtp_expert_select_rows8::kRows8SharedBytes);
  cudaFuncSetAttribute(gate_rows8_tiled,cudaFuncAttributePreferredSharedMemoryCarveout,100);
  gate_rows8_tiled<<<dim3(ctas,groups),384,naivert_mtp_expert_select_rows8::kRows8SharedBytes,stream>>>(a);
  return cudaGetLastError();
}

} // namespace

#include "fastllm-naive-flexmoe-adapter.cuh"

extern "C" int fastflex_supported(const int*devices,int ranks) {
    if (!devices || ranks != 8) return 0;
    int previous=0; if(cudaGetDevice(&previous)!=cudaSuccess)return 0;
    bool ok=true;
    for(int r=0;r<ranks&&ok;++r) {
        cudaDeviceProp p{};
        ok=cudaGetDeviceProperties(&p,devices[r])==cudaSuccess && p.major==10 &&
            p.minor==3 && p.multiProcessorCount==148 && p.sharedMemPerBlockOptin>=232448;
        for(int q=0;q<ranks&&ok;++q)if(q!=r) {int access=0;
            ok=devices[q]!=devices[r]&&cudaDeviceCanAccessPeer(&access,devices[r],devices[q])==cudaSuccess&&access;}
        if(!ok||cudaSetDevice(devices[r])!=cudaSuccess){ok=false;break;}
        cudaFuncAttributes a{};
        ok=cudaFuncGetAttributes(&a,gate_row1)==cudaSuccess &&
           cudaFuncGetAttributes(&a,gate_rows8_tiled)==cudaSuccess &&
           cudaFuncGetAttributes(&a,small_down1)==cudaSuccess &&
           cudaFuncGetAttributes(&a,small_down_rows)==cudaSuccess;
    }
    cudaGetLastError();cudaSetDevice(previous);return ok?1:0;
}
