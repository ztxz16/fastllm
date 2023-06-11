cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

option(USE_CUDA "use cuda" OFF)

message(STATUS "USE_CUDA: ${USE_CUDA}")

set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -O2 /std:c++17 /arch:AVX /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()

include_directories(include)

message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/chatglm.cpp src/moss.cpp src/vicuna.cpp)

if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    set(FASTLLM_CUDA_SOURCES src/fastllm-cuda.cu)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
endif()

add_library(fastllm OBJECT
            ${FASTLLM_CXX_SOURCES}
            ${FASTLLM_CUDA_SOURCES}
            )
target_link_libraries(fastllm PUBLIC ${FASTLLM_LINKED_LIBS})

add_executable(main main.cpp)
target_link_libraries(main fastllm)

add_executable(quant tools/quant.cpp)
target_link_libraries(quant fastllm)

add_executable(webui example/webui/webui.cpp)
target_link_libraries(webui fastllm)

add_executable(benchmark example/benchmark/benchmark.cpp)
target_link_libraries(benchmark fastllm)