#pragma once

#include <cuda_bf16.h>

#include <cstdint>

// RMSNormHeadProj entry args: the common entry header + final RMSNorm
// and lm_head body payload.
// Interface reference:
//   dev_redesign_abi/new_abi/kernel_interface.py::rmsnorm_headproj(layer)
namespace naivert_replica_rmsnorm_head_proj {

struct RMSNormHeadProjArgs {
  // --- common entry header ---
  int total_tiles;
  int active_ctas;
  std::uint8_t* workspace_base;
  const std::uint8_t* step_block_base;

  // --- body payload: final RMSNorm + lm_head projection ---
  const __nv_bfloat16* exp_out;
  const float* final_norm;
  const __nv_bfloat16* lm_head;
  float* logits_out;
  __nv_bfloat16* last_hidden_states;

  // --- shape config ---
  int dim;
  int vocab;

  // --- optional TP8 sampler inbox publication ---
  // Every pointer names one destination rank's local sampler mailbox.  The
  // final HeadProj CTA of `tp_rank` writes this shard's packed top-1 into the
  // matching source lane of all eight destinations.  All-null preserves the
  // original HeadProj ABI and leaves the generic sampler rendezvous enabled.
  const std::uint8_t* sampler_inbox_peers[8];
  int tp_rank;
};

}  // namespace naivert_replica_rmsnorm_head_proj
