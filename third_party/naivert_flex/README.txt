This directory contains only the headers listed in UPSTREAM.json.
The pinned upstream original and installed hashes are recorded separately.
PATCHES.diff is the complete local adaptation against that upstream commit.
LICENSE and upstream notices apply to the imported code.
No upstream Python/runtime package is needed to build or run these kernels.
Existing FastLLM CUDA and CUTLASS headers are used by the native CMake targets.

The eight-row MXFP4 TP verification sources under src/devices/cuda/moe/verify8
are derived adapters. Their source revision and hashes are recorded separately
in UPSTREAM.json. A TP8 MXFP4 model configured for seven draft tokens can pack
each owned expert directly into Gate/Up MMA pages and row-major Down scales.
Each page retains the checkpoint's K32 UE8M0 exponents. All row counts,
including prefill and partial tiles, read the same allocation. Other model
configurations retain the native packed layout. Both layouts use the
existing TP activation workspace and mailboxes.
