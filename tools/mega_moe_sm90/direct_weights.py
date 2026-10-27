"""Adapt SM90 MegaMoE TMA loads to FastLLM's existing per-expert weight layout.

Regenerate the vendored kernel header without an installed DeepGEMM package.
Exact replacement counts reject incompatible upstream changes.
"""


def adapt(source):
    def replace(old, new, count=1):
        nonlocal source
        actual = source.count(old)
        if actual != count:
            raise RuntimeError(f"Unsupported DeepGEMM header: expected {count} occurrences of {old!r}, got {actual}")
        source = source.replace(old, new)

    for layer in (1, 2):
        replace(f"const __grid_constant__ cute::TmaDescriptor tensor_map_l{layer}_weights,",
                f"const cute::TmaDescriptor* tensor_map_l{layer}_weights,")
        replace(f"const float* __restrict__ l{layer}_weights_sf",
                f"const float* const* __restrict__ l{layer}_weights_sf")
        replace(f"        cute::prefetch_tma_descriptor(&tensor_map_l{layer}_weights);", "")
        replace(f"l{layer}_weights_sf + local_expert_idx * kL{layer}SFPerExpert",
                f"l{layer}_weights_sf[local_expert_idx]", 3)
    replace("? &tensor_map_l2_weights : &tensor_map_l1_weights;",
            """? tensor_map_l2_weights + local_expert_idx : tensor_map_l1_weights + local_expert_idx;
            // Descriptors live in persistent device storage, populated before capture.
            if (cute::elect_one_sync()) {
                asm volatile ("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                              :: "l"(tensor_map_b_ptr) : "memory");
            }
            __syncwarp();""")
    replace("const uint32_t n_idx = local_expert_idx * shape_n + n_block_idx * BLOCK_N;",
            "const uint32_t n_idx = n_block_idx * BLOCK_N;")
    # Original gate/up halves are contiguous. Interleave 8-row tiles only in
    # shared memory, using TMA, instead of keeping a second copy in device RAM.
    replace("                    if constexpr (LOAD_BLOCK_N <= 256) {", """                    if (block_phase == sched::BlockPhase::Linear1) {
                        #pragma unroll
                        for (uint32_t slice = 0; slice < LOAD_BLOCK_N / 16; ++slice) {
                            const uint32_t gate_row = n_idx / 2 + slice * 8;
                            tma::copy<BLOCK_K, 8, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] + slice * 16 * BLOCK_K,
                                k_idx, gate_row, 1);
                            tma::copy<BLOCK_K, 8, kSwizzleBMode, b_dtype_t>(
                                tensor_map_b_ptr, full_barriers[stage_idx],
                                smem_b[stage_idx] + (slice * 16 + 8) * BLOCK_K,
                                k_idx, gate_row + kIntermediateHidden, 1);
                        }
                    } else if constexpr (LOAD_BLOCK_N <= 256) {""")
    return "// Generated FastLLM direct-weight TMA adaptation; based on installed DeepGEMM.\n" + source


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the committed header without rewriting it")
    args = parser.parse_args()
    headers = Path(__file__).resolve().parents[2] / "third_party/deep_gemm_ep8/include/deep_gemm/impls"
    generated = adapt((headers / "sm90_fp8_mega_moe.cuh").read_text())
    output = headers / "sm90_fp8_mega_moe_fastllm.cuh"
    if args.check:
        if output.read_text() != generated:
            raise SystemExit("Generated MegaMoE header differs; run direct_weights.py to regenerate it")
        print("PASS vendored MegaMoE header reproduction")
    else:
        output.write_text(generated)
