# DeepGEMM headers used by FastLLM

This directory contains a curated header-only subset of
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).  The starting point was
the DeepGEMM tree bundled with the local vLLM 0.26.0 DeepSeek-V4 reference
environment.  FastLLM-specific SM120 BF16, MoE, and MQA integrations adapt
parts of that subset, so this directory is not a byte-for-byte upstream
snapshot.

Only the headers needed to compile FastLLM's optional SM120 kernels are kept.
They are excluded from lower-SM builds by the CMake architecture gate.

DeepGEMM is distributed under the MIT license; see `LICENSE`.
