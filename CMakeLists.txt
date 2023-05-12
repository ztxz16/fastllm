cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")

include_directories(include)

add_executable(main main.cpp src/fastllm.cpp src/chatglm.cpp src/moss.cpp)
