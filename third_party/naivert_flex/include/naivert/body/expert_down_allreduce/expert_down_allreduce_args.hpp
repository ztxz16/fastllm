#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "naivert/common/expert_activation.hpp"

// Fixed expert-down descriptor ABI. The body owns CTA execution in run();
// this structure carries operands and runtime protocol state, not control-flow.
// CTA participation is derived from the compiled output-tile geometry.
// The stable domain supplies replay epochs and mailbox addressing; the consumer
// epilogue completes publication and reduction before returning.
namespace naivert_replica_expert_down_allreduce {

struct ExpertDownRuntimeDomain;

struct ExpertDownAllreduceArgs {

  // --- body payload: per-expert down proj + weighted combine + residual ---
  const naivert::ExpertActivation* up_gate; // FP32 for W8A8, BF16 otherwise
  const float* sel_probs;                // SEL_PROBS (top-k gate weights)
  const int* sel_indices;                // SEL_INDICES (top-k expert ids)
  const __nv_fp8_e4m3* moe_w_down;       const naivert::ExpertWeightScale* moe_w_down_scale;
  const __nv_bfloat16* residual;         // UNPROJ_O, residual add
  __nv_bfloat16* exp_out;                // output: weighted expert sum + residual -> next layer
  __nv_bfloat16* tp_result;              // all-reduced result write-back (may alias exp_out)

  // --- shape config ---
  int dim;                               // hidden (4096)
  int moe_inter_dim;                     // per-rank expert width (256)
  int n_activated;                       // top-k (8)

  // --- all-reduce rank ---
  int rank;                              // = TP rank; own-lane / peer indexing

  // Required replay domain: mailbox addresses, layer tag and optional epoch.
  const ExpertDownRuntimeDomain* stable_domain;
};

}  // namespace naivert_replica_expert_down_allreduce
