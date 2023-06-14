cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

option(USE_CUDA "use cuda" OFF)

option(PY_API "python api" OFF)

message(STATUS "USE_CUDA: ${USE_CUDA}")

message(STATUS "PYTHON_API: ${PY_API}")

set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX -O2 /std:c++17 /arch:AVX /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()

if (PY_API)
    # add_compile_definitions(PY_API)
    add_definitions(-DPY_API)
    set(PYBIND third_party/pybind11)
    add_subdirectory(${PYBIND})
    pybind11_add_module(pyfastllm src/pybinding.cpp src/chatglm.cpp src/moss.cpp src/vicuna.cpp src/fastllm.cpp src/fastllm.cpp include/basellm.h include/chatglm.h include/factoryllm.h include/fastllm.h include/utils.h) 
endif()

find_package(Python3 REQUIRED)
include_directories(include third_party/pybind11/include)

message(STATUS "CMAKE_CXX_FLAGS" ${CMAKE_CXX_FLAGS})
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/devices/cpu/cpudevice.cpp src/executor.cpp src/chatglm.cpp src/moss.cpp src/vicuna.cpp)

if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    set(FASTLLM_CUDA_SOURCES src/fastllm-cuda.cu src/devices/cuda/cudadevice.cpp)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
    #set(CMAKE_CUDA_ARCHITECTURES "70")
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