set_directory_properties(PROPERTIES COMPILE_OPTIONS "" COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
add_library(fastllm_qwen_image SHARED ops.cu model.cpp pipeline.cpp)
target_link_libraries(fastllm_qwen_image PUBLIC fastllm_continuous)
target_compile_options(fastllm_qwen_image PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_qwen_image PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(qwen_image_server server.cpp)
target_include_directories(qwen_image_server PRIVATE ${PROJECT_SOURCE_DIR}/example/webui)
target_link_libraries(qwen_image_server PRIVATE fastllm_qwen_image pthread)
set_target_properties(qwen_image_server PROPERTIES CXX_STANDARD 17)
add_executable(qwen_image_generate ${PROJECT_SOURCE_DIR}/tools/qwen_image/generate.cpp)
target_link_libraries(qwen_image_generate PRIVATE fastllm_qwen_image)
set_target_properties(qwen_image_generate PROPERTIES CXX_STANDARD 17)

if(UNIT_TEST)
    add_library(qwen_image_reference SHARED ${PROJECT_SOURCE_DIR}/test/qwen_image/reference_bridge.cpp)
    target_link_libraries(qwen_image_reference PRIVATE fastllm_qwen_image)
    set_target_properties(qwen_image_reference PROPERTIES CXX_STANDARD 17
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    add_executable(qwen_image_codec_test image.cpp ${PROJECT_SOURCE_DIR}/test/qwen_image/image_codec.cpp)
    target_include_directories(qwen_image_codec_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(qwen_image_codec_test PROPERTIES CXX_STANDARD 17)
endif()
