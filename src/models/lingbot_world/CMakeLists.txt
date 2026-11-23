set_directory_properties(PROPERTIES COMPILE_OPTIONS "" COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
add_library(fastllm_lingbot ${FASTLLM_MODEL_LIBRARY_TYPE} ops.cu dit.cpp text.cpp tokenizer.cpp bindings.cpp
    vae.cpp vae_ops.cu vae_bf16.cu vae_attention.cu sampling.cu camera.cpp camera.cu pipeline.cpp image.cpp)
if(FASTLLM_MERGE_MEDIA)
    # These objects are supplied by Qwen Image and the continuous runtime.
    if(FASTLLM_CUDA_COMPRESSION_OPTION)
        target_compile_options(fastllm_lingbot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${FASTLLM_CUDA_COMPRESSION_OPTION}>)
    endif()
else()
    target_sources(fastllm_lingbot PRIVATE ../qwen_image/ops.cu ../qwen_image/image.cpp)
endif()
target_link_libraries(fastllm_lingbot PUBLIC fastllm_continuous)
target_include_directories(fastllm_lingbot PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(fastllm_lingbot PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_lingbot PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
if(FASTLLM_MERGE_MEDIA)
    # FlashInfer in ops.cu uses CUDA barriers available on SM70 and newer.
    set(lingbot_archs ${FASTLLM_CUDA_ARCHITECTURES})
    list(FILTER lingbot_archs EXCLUDE REGEX "^(5[0-9]|6[0-9])(-real|-virtual)?$")
    set_target_properties(fastllm_lingbot PROPERTIES CUDA_ARCHITECTURES "${lingbot_archs}")
endif()
add_executable(lingbot_world_generate ${PROJECT_SOURCE_DIR}/tools/lingbot_world/generate.cpp)
fastllm_link_model(lingbot_world_generate PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_generate PROPERTIES CXX_STANDARD 17)
add_library(lingbot_world_test_ops SHARED EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/lingbot_world/operator_bridge.cpp)
fastllm_link_model(lingbot_world_test_ops PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_test_ops PROPERTIES CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(lingbot_world_test_condition_cache EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/lingbot_world/test_condition_cache.cpp)
fastllm_link_model(lingbot_world_test_condition_cache PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_test_condition_cache PROPERTIES CXX_STANDARD 17)
add_executable(lingbot_world_test_pipeline_state EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/lingbot_world/test_pipeline_state.cpp)
fastllm_link_model(lingbot_world_test_pipeline_state PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_test_pipeline_state PROPERTIES CXX_STANDARD 17)
add_executable(lingbot_world_test_shared_tensor_ops EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/lingbot_world/test_shared_tensor_ops.cpp)
fastllm_link_model(lingbot_world_test_shared_tensor_ops PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_test_shared_tensor_ops PROPERTIES CXX_STANDARD 17)
