Linux x86_64 wheel build
=======================

Use the Ubuntu 20.04 / GCC 11 / CUDA 12.9 environment in whl_docker/Dockerfile:

    bash make_whl.sh --build-dir /work/build-wheel --output-dir /work/dist

The default wheel recipe retains every selected native GPU target and omits
LLM PTX fallback to reduce package size. CPU and media backends remain included.
Media PTX is preserved: some media kernels need it for GPUs beyond their
single native target. Do not indiscriminately strip PTX from all libraries.

To retain LLM PTX as well:

    bash make_whl.sh --keep-ptx --build-dir /work/build-wheel --output-dir /work/dist

Ordinary CMake builds still retain LLM PTX by default. The corresponding CMake
option is FASTLLM_LLM_CUDA_PTX (ON by default, OFF in the wheel recipe).
The OFF setting requires explicit native architecture values; use --keep-ptx
with virtual-only or automatic architecture selections.

The native-only wheel cannot use CUDA_FORCE_PTX_JIT for its LLM library.
This is a packaging choice, not a change to kernel arithmetic or runtime dispatch.
When changing the architecture list, check native coverage with cuobjdump
--list-elf and test on the intended hardware. GLIBC auditing and CUDA runtime
dependencies remain handled by package_wheel.py and the normal release process.

The release wheel also builds and checks the Qwen-Image-2.1 native library,
server, generator and web/API assets. Run ftllm qwen-image server --help or
ftllm qwen-image generate --help after installation; no Torch/Diffusers runtime
is needed. Media targets retain their own SM80+ architecture/PTX policy.
