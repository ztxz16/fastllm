# Standalone CUDA kernel correctness tests.
#
# Keep these tests independent from full model loading. Each test should compare
# a small CPU reference against the CUDA kernel output and print numerical error
# statistics.

message(STATUS "CUDA kernel tests enabled")

add_library(cuda_test_utils STATIC
    test_utils.cpp
)

target_include_directories(cuda_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(test_awq_gemm_compare
    test_awq_gemm_compare.cu
)

target_link_libraries(test_awq_gemm_compare PRIVATE
    cuda_test_utils
    fastllm
)

target_include_directories(test_awq_gemm_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/devices/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/devices/cuda
)

if(FASTLLM_CUTLASS_W4A8_ENABLED)
    add_executable(test_w4a8_cutlass_linear
        test_w4a8_cutlass_linear.cu
    )

    target_link_libraries(test_w4a8_cutlass_linear PRIVATE
        cuda_test_utils
        fastllm
    )

    target_include_directories(test_w4a8_cutlass_linear PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/devices/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/devices/cuda
    )

    message(STATUS "CUDA W4A8 CUTLASS test: ON")
else()
    message(STATUS "CUDA W4A8 CUTLASS test: OFF")
endif()
