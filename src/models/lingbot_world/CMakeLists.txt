set_directory_properties(PROPERTIES COMPILE_OPTIONS "" COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
add_library(fastllm_lingbot SHARED ops.cu dit.cpp text.cpp tokenizer.cpp bindings.cpp
    vae.cpp vae_ops.cu vae_bf16.cu vae_attention.cu sampling.cu camera.cpp camera.cu pipeline.cpp image.cpp ../qwen_image/ops.cu ../qwen_image/image.cpp)
target_link_libraries(fastllm_lingbot PUBLIC fastllm_continuous)
target_include_directories(fastllm_lingbot PRIVATE ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(fastllm_lingbot PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_lingbot PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(lingbot_world_generate ${PROJECT_SOURCE_DIR}/tools/lingbot_world/generate.cpp)
target_link_libraries(lingbot_world_generate PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_generate PROPERTIES CXX_STANDARD 17)
add_library(lingbot_world_test_ops SHARED EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/test/lingbot_world/operator_bridge.cpp)
target_link_libraries(lingbot_world_test_ops PRIVATE fastllm_lingbot)
set_target_properties(lingbot_world_test_ops PROPERTIES CXX_STANDARD 17
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
