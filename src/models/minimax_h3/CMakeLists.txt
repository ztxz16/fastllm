set_directory_properties(PROPERTIES COMPILE_OPTIONS "" COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
# Exceptions cross the shared-library/executable boundary. All H3 targets must
# use the same unwinder even when the wheel statically links libstdc++.
string(REPLACE "-static-libgcc" "-shared-libgcc" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Export the continuous backend from this DSO instead of linking a second copy
# into each executable. Keep only the bundled C++ runtime symbols private.
string(REPLACE "--exclude-libs,ALL" "--exclude-libs,libstdc++.a:libgcc.a:libgcc_eh.a"
    CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# A shared library and its callers must also use the same CUDA runtime.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set_target_properties(fastllm_continuous PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
add_library(fastllm_minimax_h3 SHARED
    ops.cu encode.cu conditioning.cpp transformer.cpp audio.cpp video.cpp
    text.cpp parallel.cpp pipeline.cpp pika.cpp model_source.cpp quantized.cpp media.cpp media_input.cpp
    ../minimax_h3_image/model.cpp ../minimax_h3_image/ops.cu ../minimax_h3_image/parallel.cpp
    ../minimax_h3_image/input.cpp ../minimax_h3_image/memory.cpp ../minimax_h3_image/fp8.cu ../minimax_h3_image/source.cpp
    ../minimax_h3_image/gguf_checkpoint.cpp ../minimax_h3_image/gguf.cu
    ../minimax_h3_image/quantized.cpp ../minimax_h3_image/int8.cu ../minimax_h3_image/nvfp4.cu)
target_link_libraries(fastllm_minimax_h3 PUBLIC fastllm_continuous pthread)
target_include_directories(fastllm_minimax_h3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(fastllm_minimax_h3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_minimax_h3 PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Reuse GGUF MMQ kernels without linking the separate LLM runtime/allocator.
if(NOT CUDA_NO_TENSOR_CORE)
    add_library(fastllm_image_gguf_mmq OBJECT ../minimax_h3_image/gguf_mmq.cu)
    set_target_properties(fastllm_image_gguf_mmq PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_image_gguf_mmq PRIVATE
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/devices/cuda
        ${PROJECT_SOURCE_DIR}/include/utils ${PROJECT_SOURCE_DIR}/include/models ${PROJECT_SOURCE_DIR}/include/blocks
        ${PROJECT_SOURCE_DIR}/third_party/json11 ${PROJECT_SOURCE_DIR}/third_party/gguf
        ${PROJECT_SOURCE_DIR}/src/devices/cuda/gguf_mmq)
    target_compile_options(fastllm_image_gguf_mmq PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_minimax_h3 PRIVATE FASTLLM_IMAGE_GGUF_MMQ)
    target_sources(fastllm_minimax_h3 PRIVATE $<TARGET_OBJECTS:fastllm_image_gguf_mmq>)
endif()
# Keep SM120-only CUTLASS instantiations out of legacy architecture builds.
# The ordinary FP8 translation unit retains the cuBLAS fallback.
if(FASTLLM_CUTLASS_FP8_SM120_ENABLED AND NOT CUDA_NO_TENSOR_CORE AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8" AND
   EXISTS "${FASTLLM_CUTLASS_DIR}/include/cutlass/cutlass.h")
    add_library(fastllm_image_fp8_sm120 OBJECT ../minimax_h3_image/fp8_scaled.cu ../minimax_h3_image/nvfp4_scaled.cu
        ../minimax_h3_image/nvfp4_scaled_fp32.cu)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
        set(image_fp8_arch "120f")
    else()
        set(image_fp8_arch "120a")
    endif()
    set_target_properties(fastllm_image_fp8_sm120 PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "${image_fp8_arch}" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_image_fp8_sm120 PRIVATE
        "${FASTLLM_CUTLASS_DIR}/include" "${FASTLLM_CUTLASS_DIR}/tools/util/include")
    target_compile_options(fastllm_image_fp8_sm120 PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_minimax_h3 PRIVATE FASTLLM_IMAGE_FP8_SM120 FASTLLM_IMAGE_NVFP4_SM120)
    target_sources(fastllm_minimax_h3 PRIVATE $<TARGET_OBJECTS:fastllm_image_fp8_sm120>)
endif()
# The fused INT8 epilogue uses mma.sync and is available from Ampere onward.
if(NOT CUDA_NO_TENSOR_CORE AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.8" AND
   EXISTS "${FASTLLM_CUTLASS_DIR}/include/cutlass/epilogue/threadblock/fusion/visitors.hpp")
    add_library(fastllm_image_int8_scaled OBJECT ../minimax_h3_image/int8_scaled.cu)
    set_target_properties(fastllm_image_int8_scaled PROPERTIES
        CUDA_STANDARD 17 CUDA_ARCHITECTURES "80" POSITION_INDEPENDENT_CODE ON)
    target_include_directories(fastllm_image_int8_scaled PRIVATE "${FASTLLM_CUTLASS_DIR}/include")
    target_compile_options(fastllm_image_int8_scaled PRIVATE --expt-relaxed-constexpr --fmad=false)
    target_compile_definitions(fastllm_minimax_h3 PRIVATE FASTLLM_IMAGE_INT8_SCALED)
    target_sources(fastllm_minimax_h3 PRIVATE $<TARGET_OBJECTS:fastllm_image_int8_scaled>)
endif()
add_executable(minimax_h3_generate ${PROJECT_SOURCE_DIR}/tools/minimax_h3/generate.cpp)
target_link_libraries(minimax_h3_generate PRIVATE fastllm_minimax_h3)
set_target_properties(minimax_h3_generate PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
add_executable(minimax_h3_image_generate ${PROJECT_SOURCE_DIR}/tools/minimax_h3_image/generate.cpp)
target_link_libraries(minimax_h3_image_generate PRIVATE fastllm_minimax_h3)
set_target_properties(minimax_h3_image_generate PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
add_executable(minimax_h3_image_server ../minimax_h3_image/server.cpp)
target_include_directories(minimax_h3_image_server PRIVATE ${PROJECT_SOURCE_DIR}/example/webui)
target_link_libraries(minimax_h3_image_server PRIVATE fastllm_minimax_h3)
set_target_properties(minimax_h3_image_server PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
add_executable(minimax_h3_server server.cpp)
target_include_directories(minimax_h3_server PRIVATE ${PROJECT_SOURCE_DIR}/example/webui)
target_link_libraries(minimax_h3_server PRIVATE fastllm_minimax_h3)
set_target_properties(minimax_h3_server PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
if(MAKE_WHL_X86)
    # The BF16 video attention backend targets Ampere and newer GPUs. Keep
    # legacy LLM architectures in the same wheel without compiling FlashInfer
    # synchronization primitives for unsupported targets such as SM60.
    set(FASTLLM_VIDEO_CUDA_ARCHITECTURES "")
    foreach(arch IN LISTS FASTLLM_CUDA_ARCHITECTURES)
        if(arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
            continue()
        endif()
        list(APPEND FASTLLM_VIDEO_CUDA_ARCHITECTURES "${arch}")
    endforeach()
    if(NOT FASTLLM_VIDEO_CUDA_ARCHITECTURES)
        message(FATAL_ERROR "The release video backend requires at least one SM80+ CUDA target")
    endif()
    set_target_properties(fastllm_continuous fastllm_minimax_h3 PROPERTIES
        CUDA_ARCHITECTURES "${FASTLLM_VIDEO_CUDA_ARCHITECTURES}")
    if(TARGET fastllm_image_gguf_mmq)
        set_target_properties(fastllm_image_gguf_mmq PROPERTIES
            CUDA_ARCHITECTURES "${FASTLLM_VIDEO_CUDA_ARCHITECTURES}")
    endif()
    # Wheel executables and the shared library live together in ftllm/.
    # Resolve pip's CUDA dependencies without requiring LD_LIBRARY_PATH.
    set_target_properties(fastllm_minimax_h3 minimax_h3_server minimax_h3_generate minimax_h3_image_server minimax_h3_image_generate PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cublas/lib;$ORIGIN/../nvidia/nccl/lib")
endif()
if(UNIT_TEST)
    enable_testing()
    # Explicit targets: these checks require the full image checkpoint.
    foreach(check encode decode state parallel)
        add_executable(minimax_h3_image_${check} EXCLUDE_FROM_ALL
            ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/test_${check}.cpp)
        target_link_libraries(minimax_h3_image_${check} PRIVATE fastllm_minimax_h3)
        set_target_properties(minimax_h3_image_${check} PROPERTIES
            CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
    endforeach()
    target_include_directories(minimax_h3_image_parallel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_image_parallel PRIVATE cudart)
    foreach(check fp8 int8 nvfp4 gguf source)
        add_executable(minimax_h3_image_${check} ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/test_${check}.cpp)
        target_include_directories(minimax_h3_image_${check} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_link_libraries(minimax_h3_image_${check} PRIVATE fastllm_minimax_h3 cudart)
        set_target_properties(minimax_h3_image_${check} PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
        add_test(NAME minimax_h3_image_${check} COMMAND minimax_h3_image_${check})
        set_tests_properties(minimax_h3_image_${check} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
    add_executable(minimax_h3_image_ops ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/test_ops.cpp)
    target_include_directories(minimax_h3_image_ops PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_image_ops PRIVATE fastllm_minimax_h3 cudart)
    target_compile_options(minimax_h3_image_ops PRIVATE -ffp-contract=off)
    set_target_properties(minimax_h3_image_ops PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
    add_test(NAME minimax_h3_image_ops COMMAND minimax_h3_image_ops)
    set_tests_properties(minimax_h3_image_ops PROPERTIES LABELS cuda TIMEOUT 60 SKIP_RETURN_CODE 77)
    add_executable(minimax_h3_image_memory
        ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/test_memory.cpp ../minimax_h3_image/memory.cpp)
    set_target_properties(minimax_h3_image_memory PROPERTIES CXX_STANDARD 17)
    add_test(NAME minimax_h3_image_memory COMMAND minimax_h3_image_memory)
    add_executable(minimax_h3_image_condition EXCLUDE_FROM_ALL
        ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/condition_fixture.cpp)
    target_link_libraries(minimax_h3_image_condition PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_image_condition PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME minimax_h3_image_http
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/minimax_h3_image/test_server.py
                    --binary $<TARGET_FILE:minimax_h3_image_server>)
        set_tests_properties(minimax_h3_image_http PROPERTIES TIMEOUT 30)
    endif()
    add_executable(minimax_h3_quantized_source ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_quantized_source.cpp)
    target_link_libraries(minimax_h3_quantized_source PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_quantized_source PROPERTIES CXX_STANDARD 17)
    add_test(NAME minimax_h3_quantized_source COMMAND minimax_h3_quantized_source)
    add_executable(minimax_h3_model_source ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_model_source.cpp)
    target_link_libraries(minimax_h3_model_source PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_model_source PROPERTIES CXX_STANDARD 17)
    add_test(NAME minimax_h3_model_source COMMAND minimax_h3_model_source)
    add_executable(minimax_h3_media ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_media.cpp)
    target_link_libraries(minimax_h3_media PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_media PROPERTIES CXX_STANDARD 17)
    add_test(NAME minimax_h3_media COMMAND minimax_h3_media)
    set_tests_properties(minimax_h3_media PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)
    add_executable(minimax_h3_fused_ops ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_fused_ops.cpp)
    target_include_directories(minimax_h3_fused_ops PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_fused_ops PRIVATE fastllm_minimax_h3 cudart)
    target_compile_options(minimax_h3_fused_ops PRIVATE -ffp-contract=off)
    set_target_properties(minimax_h3_fused_ops PROPERTIES CXX_STANDARD 17)
    add_test(NAME minimax_h3_fused_ops COMMAND minimax_h3_fused_ops)
    set_tests_properties(minimax_h3_fused_ops PROPERTIES LABELS cuda TIMEOUT 180 SKIP_RETURN_CODE 77)
    add_executable(minimax_h3_quantized_parallel EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_quantized_parallel.cpp)
    target_include_directories(minimax_h3_quantized_parallel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_quantized_parallel PRIVATE fastllm_minimax_h3 cudart)
    set_target_properties(minimax_h3_quantized_parallel PROPERTIES CXX_STANDARD 17)
    add_executable(minimax_h3_pika_parallel EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_pika_parallel.cpp)
    target_include_directories(minimax_h3_pika_parallel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_pika_parallel PRIVATE fastllm_minimax_h3 cudart)
    set_target_properties(minimax_h3_pika_parallel PROPERTIES CXX_STANDARD 17)
    add_library(minimax_h3_reference SHARED ${PROJECT_SOURCE_DIR}/test/minimax_h3/reference_bridge.cpp)
    target_link_libraries(minimax_h3_reference PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_reference PROPERTIES CXX_STANDARD 17
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
