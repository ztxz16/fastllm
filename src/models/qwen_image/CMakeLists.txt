set_directory_properties(PROPERTIES COMPILE_OPTIONS "" COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
add_library(fastllm_qwen_image SHARED ops.cu conv.cu model.cpp pipeline.cpp parallel.cpp quantized.cpp
    ../minimax_h3_image/fp8.cu ../minimax_h3_image/int8.cu
    ../minimax_h3_image/gguf.cu ../minimax_h3_image/gguf_checkpoint.cpp ../minimax_h3_image/source.cpp)
target_link_libraries(fastllm_qwen_image PUBLIC fastllm_continuous)
target_include_directories(fastllm_qwen_image PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cutlass/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(fastllm_qwen_image PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_qwen_image PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Reuse H3's native FP8 reader/GEMM, including its cuBLAS fallback on SM89+.
# Keep architecture-specific instantiations separate from portable kernels.
if(FASTLLM_CUTLASS_FP8_SM120_ENABLED AND NOT CUDA_NO_TENSOR_CORE AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8" AND
   EXISTS "${FASTLLM_CUTLASS_DIR}/include/cutlass/cutlass.h")
    add_library(fastllm_qwen_fp8_sm120 OBJECT ../minimax_h3_image/fp8_scaled.cu)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        set(qwen_fp8_arch "120f")
    else()
        set(qwen_fp8_arch "120a")
    endif()
    set_target_properties(fastllm_qwen_fp8_sm120 PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${qwen_fp8_arch}" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_qwen_fp8_sm120 PRIVATE
        "${FASTLLM_CUTLASS_DIR}/include" "${FASTLLM_CUTLASS_DIR}/tools/util/include")
    target_compile_options(fastllm_qwen_fp8_sm120 PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_qwen_image PRIVATE FASTLLM_IMAGE_FP8_SM120)
    target_sources(fastllm_qwen_image PRIVATE $<TARGET_OBJECTS:fastllm_qwen_fp8_sm120>)
endif()
# Reuse GGUF MMQ kernels without linking the separate LLM runtime/allocator.
if(NOT CUDA_NO_TENSOR_CORE)
    add_library(fastllm_qwen_gguf_mmq OBJECT ../minimax_h3_image/gguf_mmq.cu)
    set_target_properties(fastllm_qwen_gguf_mmq PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_qwen_gguf_mmq PRIVATE
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/devices/cuda
        ${PROJECT_SOURCE_DIR}/include/utils ${PROJECT_SOURCE_DIR}/include/models ${PROJECT_SOURCE_DIR}/include/blocks
        ${PROJECT_SOURCE_DIR}/third_party/json11 ${PROJECT_SOURCE_DIR}/third_party/gguf
        ${PROJECT_SOURCE_DIR}/src/devices/cuda/gguf_mmq)
    target_compile_options(fastllm_qwen_gguf_mmq PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_qwen_image PRIVATE FASTLLM_IMAGE_GGUF_MMQ)
    target_sources(fastllm_qwen_image PRIVATE $<TARGET_OBJECTS:fastllm_qwen_gguf_mmq>)
endif()
# The fused INT8 epilogue uses mma.sync and is available from Ampere onward.
if(NOT CUDA_NO_TENSOR_CORE AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8" AND
   EXISTS "${FASTLLM_CUTLASS_DIR}/include/cutlass/epilogue/threadblock/fusion/visitors.hpp")
    add_library(fastllm_qwen_int8_scaled OBJECT ../minimax_h3_image/int8_scaled.cu)
    set_target_properties(fastllm_qwen_int8_scaled PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "80" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_qwen_int8_scaled PRIVATE "${FASTLLM_CUTLASS_DIR}/include")
    target_compile_options(fastllm_qwen_int8_scaled PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_qwen_image PRIVATE FASTLLM_IMAGE_INT8_SCALED)
    target_sources(fastllm_qwen_image PRIVATE $<TARGET_OBJECTS:fastllm_qwen_int8_scaled>)
endif()
add_executable(qwen_image_server server.cpp)
target_include_directories(qwen_image_server PRIVATE ${PROJECT_SOURCE_DIR}/example/webui)
target_link_libraries(qwen_image_server PRIVATE fastllm_qwen_image pthread)
set_target_properties(qwen_image_server PROPERTIES CXX_STANDARD 17)
add_executable(qwen_image_generate ${PROJECT_SOURCE_DIR}/tools/qwen_image/generate.cpp)
target_link_libraries(qwen_image_generate PRIVATE fastllm_qwen_image)
set_target_properties(qwen_image_generate PROPERTIES CXX_STANDARD 17)

if(UNIT_TEST)
    # The root enables testing after adding model subdirectories.
    enable_testing()
    # Real model paths and two explicitly reserved GPUs are supplied by the caller.
    add_executable(qwen_image_pipeline_parallel_test EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/test/qwen_image/test_pipeline_parallel.cpp)
    target_include_directories(qwen_image_pipeline_parallel_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qwen_image_pipeline_parallel_test PRIVATE fastllm_qwen_image cudart)
    set_target_properties(qwen_image_pipeline_parallel_test PROPERTIES CXX_STANDARD 17)
    add_executable(qwen_image_parallel_test ${PROJECT_SOURCE_DIR}/test/qwen_image/test_parallel.cpp)
    target_include_directories(qwen_image_parallel_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../minimax_h3_image ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qwen_image_parallel_test PRIVATE fastllm_qwen_image cudart)
    set_target_properties(qwen_image_parallel_test PROPERTIES CXX_STANDARD 17)
    target_compile_options(qwen_image_parallel_test PRIVATE -ffp-contract=off)
    if(FASTLLM_NCCL_INCLUDE_DIR AND FASTLLM_NCCL_LIBRARY)
        target_compile_definitions(qwen_image_parallel_test PRIVATE FASTLLM_QWEN_TEST_NCCL)
    endif()
    add_test(NAME qwen_image_parallel COMMAND qwen_image_parallel_test)
    set_tests_properties(qwen_image_parallel PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    add_executable(qwen_image_quantized_test ${PROJECT_SOURCE_DIR}/test/qwen_image/test_quantized.cpp)
    target_include_directories(qwen_image_quantized_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qwen_image_quantized_test PRIVATE fastllm_qwen_image cudart)
    if(TARGET fastllm_qwen_gguf_mmq)
        target_compile_definitions(qwen_image_quantized_test PRIVATE FASTLLM_IMAGE_GGUF_MMQ)
    endif()
    set_target_properties(qwen_image_quantized_test PROPERTIES CXX_STANDARD 17)
    add_test(NAME qwen_image_quantized COMMAND qwen_image_quantized_test)
    set_tests_properties(qwen_image_quantized PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    add_executable(qwen_image_fp8_test ${PROJECT_SOURCE_DIR}/test/qwen_image/test_fp8.cpp)
    target_include_directories(qwen_image_fp8_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(qwen_image_fp8_test PRIVATE fastllm_qwen_image cudart)
    set_target_properties(qwen_image_fp8_test PROPERTIES CXX_STANDARD 17)
    add_test(NAME qwen_image_fp8 COMMAND qwen_image_fp8_test)
    set_tests_properties(qwen_image_fp8 PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    add_library(qwen_image_reference SHARED ${PROJECT_SOURCE_DIR}/test/qwen_image/reference_bridge.cpp)
    target_link_libraries(qwen_image_reference PRIVATE fastllm_qwen_image)
    set_target_properties(qwen_image_reference PROPERTIES CXX_STANDARD 17
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    add_executable(qwen_image_codec_test image.cpp ${PROJECT_SOURCE_DIR}/test/qwen_image/image_codec.cpp)
    target_include_directories(qwen_image_codec_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(qwen_image_codec_test PROPERTIES CXX_STANDARD 17)
endif()
