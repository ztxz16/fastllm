# Keep the continuous-tensor backend separate from fastllm's LLM targets.
# In particular, do not inherit global CUDA fast-math/stream options: the
# reference-checked kernels use explicit streams and FP32 rounding boundaries.
set_directory_properties(PROPERTIES
    COMPILE_OPTIONS ""
    COMPILE_DEFINITIONS ""
    INCLUDE_DIRECTORIES "")

add_library(fastllm_nomas SHARED
    attention.cu runtime.cu model.cpp tokenizer.cpp pipeline.cpp
    ${PROJECT_SOURCE_DIR}/third_party/json11/json11.cpp)
target_include_directories(fastllm_nomas PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/json11
    ${PROJECT_SOURCE_DIR}/third_party/flashinfer)
target_compile_options(fastllm_nomas PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
set_target_properties(fastllm_nomas PROPERTIES
    CUDA_STANDARD 17
    CXX_STANDARD 17
    CUDA_ARCHITECTURES "${FASTLLM_CUDA_ARCHITECTURES}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_link_libraries(fastllm_nomas PRIVATE cublas cublasLt)
