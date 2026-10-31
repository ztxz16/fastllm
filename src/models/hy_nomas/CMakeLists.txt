# Keep the continuous-tensor backend separate from fastllm's LLM targets.
# In particular, do not inherit global CUDA fast-math/stream options: the
# reference-checked kernels use explicit streams and FP32 rounding boundaries.
set_directory_properties(PROPERTIES
    COMPILE_OPTIONS ""
    COMPILE_DEFINITIONS ""
    INCLUDE_DIRECTORIES "")

add_library(fastllm_continuous STATIC
    attention.cu runtime.cu parallel.cu tokenizer.cpp vision.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen_image/image.cpp
    ${PROJECT_SOURCE_DIR}/third_party/json11/json11.cpp)
target_include_directories(fastllm_continuous PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/json11
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer)
target_compile_options(fastllm_continuous PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_continuous PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(fastllm_continuous PUBLIC cublas cublasLt pthread)
# NCCL is already used by the LLM CUDA backend. Keep single-device continuous
# builds possible without it; only the opt-in multi-device path requires NCCL.
find_path(FASTLLM_NCCL_INCLUDE_DIR nccl.h)
find_library(FASTLLM_NCCL_LIBRARY NAMES nccl libnccl.so.2)
if(FASTLLM_NCCL_INCLUDE_DIR AND FASTLLM_NCCL_LIBRARY)
    target_compile_definitions(fastllm_continuous PRIVATE FASTLLM_CONTINUOUS_NCCL)
    target_include_directories(fastllm_continuous PRIVATE ${FASTLLM_NCCL_INCLUDE_DIR})
    target_link_libraries(fastllm_continuous PUBLIC ${FASTLLM_NCCL_LIBRARY})
endif()

if(UNIT_TEST)
    enable_testing()
    add_executable(continuous_parallel ${PROJECT_SOURCE_DIR}/test/hy_nomas/test_parallel.cpp)
    target_include_directories(continuous_parallel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(continuous_parallel PRIVATE fastllm_continuous cudart)
    set_target_properties(continuous_parallel PROPERTIES CXX_STANDARD 17)
    if(FASTLLM_NCCL_INCLUDE_DIR AND FASTLLM_NCCL_LIBRARY)
        target_compile_definitions(continuous_parallel PRIVATE FASTLLM_TEST_NCCL)
    endif()
    add_test(NAME continuous_parallel COMMAND continuous_parallel)
    set_tests_properties(continuous_parallel PROPERTIES LABELS cuda TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

if(BUILD_HY_NOMAS)
    add_library(fastllm_nomas SHARED bindings.cpp model.cpp pipeline.cpp noise.cu)
    target_link_libraries(fastllm_nomas PRIVATE fastllm_continuous)
    target_compile_options(fastllm_nomas PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
        $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
    # CUDA randn uses cuRAND's default fused arithmetic in its normal transform.
    # Keep this exception local; the model operators retain their rounding rules.
    set_source_files_properties(noise.cu PROPERTIES COMPILE_OPTIONS "--fmad=true")
    set_target_properties(fastllm_nomas PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
        CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
