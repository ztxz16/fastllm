cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()

include_directories(include)

add_executable(main main.cpp src/fastllm.cpp src/chatglm.cpp src/moss.cpp)
add_executable(quant tools/quant.cpp src/fastllm.cpp src/chatglm.cpp src/moss.cpp)
