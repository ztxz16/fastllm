# FlexKV SSD data-plane extraction

Upstream: https://github.com/taco-project/FlexKV
Commit: `738ddc141a198b4e20de6c5d1f0128e387f7fdb2`
License: Apache-2.0; complete upstream LICENSE retained here.

`transfer.h` directly extracts `transfer_full` and `transfer_iov_full` from
`csrc/transfer_ssd.cpp`, and `do_transfer_packed_blocks`/`PackedSpan` from
`csrc/compression/common/packed_ssd.cpp` and `.h`.

Local changes: inline linkage, standalone POSIX/C++ includes, isolated namespace.
The retry, residual-iovec, gather/scatter and packed-transfer implementations are
retained. Torch/CUDA and nvcomp coordinate producers are deliberately not bundled:
FastLLM supplies its own canonical tensor coordinates and bounded staging spans.
Buffered POSIX is used; this extraction does not claim io_uring, O_DIRECT or GDS.

The append-only checkpoint publication log in FastLLM is model-specific catalog
recovery logic. It follows the durable-data-before-metadata/checkpoint approach
reviewed in LMCache raw-block but is not a copied LMCache allocator or I/O wrapper.
