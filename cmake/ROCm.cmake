# Included only for USE_ROCM; NVIDIA targets keep their original setup.
if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
endif()
set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
list(INSERT CMAKE_PREFIX_PATH 0 "${ROCM_PATH}")
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
if(NOT CMAKE_HIP_COMPILER)
    foreach(compiler IN ITEMS "${ROCM_PATH}/lib/llvm/bin/clang++"
                              "${ROCM_PATH}/llvm/bin/clang++")
        if(EXISTS "${compiler}")
            set(CMAKE_HIP_COMPILER "${compiler}" CACHE FILEPATH "HIP compiler")
            break()
        endif()
    endforeach()
endif()
set(CMAKE_HIP_ARCHITECTURES "${ROCM_ARCH}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/third_party/hipify_torch/cmake")
include(Hipify)
enable_language(HIP)
# Match CUDA: implicit kernel launches must join the stream captured by
# FastllmCudaBeginCaptureCurrentThread, not the legacy default stream.
add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-fgpu-default-stream=per-thread>)

# Hipify private copies of the CUDA headers alongside the GPU sources.
# Host C++ sources continue to use the original public headers.
set(FASTLLM_HIP_HEADERS "${CMAKE_CURRENT_BINARY_DIR}/hip-headers")
file(REMOVE_RECURSE "${FASTLLM_HIP_HEADERS}")
file(COPY "${PROJECT_SOURCE_DIR}/include/devices/cuda"
          "${PROJECT_SOURCE_DIR}/include/devices/multicuda"
     DESTINATION "${FASTLLM_HIP_HEADERS}")
# fastllm-hip.h is already HIP code; translating it would rewrite the
# native shuffle calls inside our wrappers into recursive calls.
set(FASTLLM_HIPIFY_MAP "${PROJECT_SOURCE_DIR}/cmake/rocm_hipify_mappings.json")
include_directories(BEFORE "${FASTLLM_HIP_HEADERS}/hip"
                           "${FASTLLM_HIP_HEADERS}/cuda"
                           "${FASTLLM_HIP_HEADERS}/multicuda")

# Generated sources belong to this build, so CUDA builds and source packages
# never see HIP output and separate ROCm builds cannot overwrite each other.
set(FASTLLM_HIP_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/hip-src")
file(REMOVE_RECURSE "${FASTLLM_HIP_GENERATED_DIR}")
set(FASTLLM_HIPIFY_ARGS
    HEADER_INCLUDE_DIR "${FASTLLM_HIP_HEADERS}/cuda" "${FASTLLM_HIP_HEADERS}/multicuda"
    CUSTOM_MAP_FILE "${FASTLLM_HIPIFY_MAP}"
    IGNORES "${FASTLLM_HIP_HEADERS}/cuda/fastllm-hip.h")
hipify(CUDA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/cuda"
    HIP_SOURCE_DIR "${FASTLLM_HIP_GENERATED_DIR}/hip" ${FASTLLM_HIPIFY_ARGS})
hipify(CUDA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/multicuda"
    HIP_SOURCE_DIR "${FASTLLM_HIP_GENERATED_DIR}/multihip" ${FASTLLM_HIPIFY_ARGS})
include_directories(include/devices/cuda)
include_directories(include/devices/multicuda)
set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp
    ${FASTLLM_HIP_GENERATED_DIR}/hip/fastllm-hip.hip ${FASTLLM_HIP_GENERATED_DIR}/hip/fastllm-ggml-hip.hip
    src/devices/multicuda/multicudadevice.cpp ${FASTLLM_HIP_GENERATED_DIR}/multihip/fastllm-multihip.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/attention/fastllm-attention.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/attention/paged/fastllm-paged-attention-native.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-fp32.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-fp16.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-bf16.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-fp8.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-int8.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-int4.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-int4group.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/linear/fastllm-linear-int4nozero.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/moe/fastllm-moe-fp8.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/models/dots3-note-kernels.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/models/deepseekv4-kernels.hip
    ${FASTLLM_HIP_GENERATED_DIR}/hip/models/qwen4-kernels.hip
    src/devices/rocm/fastllm-rocm-fallbacks.hip
    src/devices/rocm/fastllm-rocm-sampling.hip)
add_compile_definitions(USE_ROCM)
add_compile_definitions(USE_CUDA)
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hipblaslt REQUIRED)
find_package(rocprim REQUIRED)
find_package(rccl REQUIRED)
# Native HIP compilation handles GPU sources; keep C++ sources as host code.
list(APPEND FASTLLM_LINKED_LIBS hip::host roc::hipblas roc::hipblaslt roc::rocprim roc::rccl)
add_compile_definitions(HIPBLAS_V2)

if(ROCM_HAS_MI50)
    add_compile_definitions(USE_MI50_WORKAROUND)
    add_compile_definitions(HIP_NO_TENSOR_CORE)
    message(STATUS "MI50 support enabled (USE_MI50_WORKAROUND defined)")
endif()
