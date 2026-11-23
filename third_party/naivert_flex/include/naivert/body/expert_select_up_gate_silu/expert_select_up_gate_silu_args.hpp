#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "naivert/common/expert_activation.hpp"
#include "naivert/slot_ring/state.cuh"

namespace naivert_replica_expert_select_up_gate_silu {

struct ExpertSelectUpGateSiluArgs {
  const __nv_bfloat16* x;
  const float* scores;
  const float* bias;
  const __nv_fp8_e4m3* packed_weights;
  naivert::ExpertActivation* y;
  float* expert_probs;
  int* expert_indices;

  int hidden;
  int routed_experts;
  int total_experts;
  int selected_experts;
  int output_slots;
  int output_groups;
  int outputs_per_group;
  int seq_len;
  int pages;
  int page_k;
  int total_tiles;

  int x_stride;
  int y_slot_stride;
  int y_group_stride;
  int packed_group_stride;
  int packed_page_stride;
  int packed_expert_stride;
  float route_scale;

  naivert::SlotRingWorkArgs progress;
};

}  // namespace naivert_replica_expert_select_up_gate_silu
