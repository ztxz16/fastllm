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
    text.cpp parallel.cpp pipeline.cpp pika.cpp model_source.cpp media.cpp media_input.cpp)
target_link_libraries(fastllm_minimax_h3 PUBLIC fastllm_continuous pthread)
target_include_directories(fastllm_minimax_h3 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(fastllm_minimax_h3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_minimax_h3 PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(minimax_h3_generate ${PROJECT_SOURCE_DIR}/tools/minimax_h3/generate.cpp)
target_link_libraries(minimax_h3_generate PRIVATE fastllm_minimax_h3)
set_target_properties(minimax_h3_generate PROPERTIES CXX_STANDARD 17 BUILD_RPATH_USE_ORIGIN TRUE)
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
    # Wheel executables and the shared library live together in ftllm/.
    # Resolve pip's CUDA dependencies without requiring LD_LIBRARY_PATH.
    set_target_properties(fastllm_minimax_h3 minimax_h3_server minimax_h3_generate PROPERTIES
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cublas/lib;$ORIGIN/../nvidia/nccl/lib")
endif()
if(UNIT_TEST)
    enable_testing()
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
    add_executable(minimax_h3_pika_parallel EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/test/minimax_h3/test_pika_parallel.cpp)
    target_include_directories(minimax_h3_pika_parallel PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(minimax_h3_pika_parallel PRIVATE fastllm_minimax_h3 cudart)
    set_target_properties(minimax_h3_pika_parallel PROPERTIES CXX_STANDARD 17)
    add_library(minimax_h3_reference SHARED ${PROJECT_SOURCE_DIR}/test/minimax_h3/reference_bridge.cpp)
    target_link_libraries(minimax_h3_reference PRIVATE fastllm_minimax_h3)
    set_target_properties(minimax_h3_reference PROPERTIES CXX_STANDARD 17
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
