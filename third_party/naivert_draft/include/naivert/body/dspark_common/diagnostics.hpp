#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.

// Intermediate exports are correctness probes, never serving/benchmark work.
// Build the explicit diagnostics extension to inspect them.
#ifndef NAIVERT_DSPARK_DIAGNOSTICS
#define NAIVERT_DSPARK_DIAGNOSTICS 0
#endif
static_assert(NAIVERT_DSPARK_DIAGNOSTICS == 0 || NAIVERT_DSPARK_DIAGNOSTICS == 1);
namespace fastllm_draft_ops {
inline constexpr bool kExportDiagnostics = NAIVERT_DSPARK_DIAGNOSTICS != 0;
}
