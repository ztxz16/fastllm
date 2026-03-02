cmake_minimum_required(VERSION 3.18.1)
project("fastllm_jni")

set(FASTLLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++17 -O2 -march=armv8.2-a+fp16")

file(GLOB CPU_DEVICE_FILES "${FASTLLM_ROOT}/src/devices/cpu/*.cpp")
file(GLOB BLOCK_FILES "${FASTLLM_ROOT}/src/blocks/*.cpp")
file(GLOB GRAPH_MODEL_FILES "${FASTLLM_ROOT}/src/models/graph/*.cpp")

set(FASTLLM_SOURCES
    ${FASTLLM_ROOT}/src/fastllm.cpp
    ${FASTLLM_ROOT}/src/device.cpp
    ${FASTLLM_ROOT}/src/model.cpp
    ${FASTLLM_ROOT}/src/executor.cpp
    ${FASTLLM_ROOT}/src/template.cpp
    ${FASTLLM_ROOT}/src/graph.cpp
    ${FASTLLM_ROOT}/src/tokenizer.cpp
    ${FASTLLM_ROOT}/src/devices/cpu/cpudevice.cpp
    ${FASTLLM_ROOT}/src/devices/cpu/cpudevicebatch.cpp
    ${FASTLLM_ROOT}/src/models/graphllm.cpp
    ${FASTLLM_ROOT}/src/models/chatglm.cpp
    ${FASTLLM_ROOT}/src/models/moss.cpp
    ${FASTLLM_ROOT}/src/models/llama.cpp
    ${FASTLLM_ROOT}/src/models/qwen.cpp
    ${FASTLLM_ROOT}/src/models/basellm.cpp
    ${FASTLLM_ROOT}/src/models/glm.cpp
    ${FASTLLM_ROOT}/src/models/minicpm.cpp
    ${FASTLLM_ROOT}/src/models/minicpm3.cpp
    ${FASTLLM_ROOT}/src/models/internlm2.cpp
    ${FASTLLM_ROOT}/src/models/bert.cpp
    ${FASTLLM_ROOT}/src/models/moe.cpp
    ${FASTLLM_ROOT}/src/models/deepseekv2.cpp
    ${FASTLLM_ROOT}/src/models/phi3.cpp
    ${FASTLLM_ROOT}/src/models/xlmroberta.cpp
    ${FASTLLM_ROOT}/src/models/cogvlm.cpp
    ${FASTLLM_ROOT}/src/models/qwen3.cpp
    ${FASTLLM_ROOT}/src/models/qwen3_moe.cpp
    ${FASTLLM_ROOT}/src/models/qwen3_next.cpp
    ${FASTLLM_ROOT}/src/models/minimax.cpp
    ${FASTLLM_ROOT}/src/models/hunyuan.cpp
    ${FASTLLM_ROOT}/src/models/ernie4_5.cpp
    ${FASTLLM_ROOT}/src/models/pangu_moe.cpp
    ${FASTLLM_ROOT}/src/models/glm4_moe.cpp
    ${FASTLLM_ROOT}/src/models/gpt_oss.cpp
    ${FASTLLM_ROOT}/src/models/minimax_m2.cpp
    ${FASTLLM_ROOT}/src/models/qwen2.cpp
    ${FASTLLM_ROOT}/third_party/json11/json11.cpp
    ${FASTLLM_ROOT}/third_party/gguf/gguf.cpp
    ${FASTLLM_ROOT}/third_party/gguf/ggml-quant.cpp
    ${FASTLLM_ROOT}/third_party/gguf/ggml-iqk.cpp
    ${FASTLLM_ROOT}/third_party/gguf/ggml-dequantize.cpp
    ${FASTLLM_ROOT}/third_party/gguf/gguf-adapter.cpp
    ${CPU_DEVICE_FILES}
    ${BLOCK_FILES}
    ${GRAPH_MODEL_FILES})

include_directories(
    ${FASTLLM_ROOT}/include
    ${FASTLLM_ROOT}/include/utils
    ${FASTLLM_ROOT}/include/models
    ${FASTLLM_ROOT}/include/blocks
    ${FASTLLM_ROOT}/include/devices/cpu
    ${FASTLLM_ROOT}/third_party/json11
    ${FASTLLM_ROOT}/third_party/gguf
    ${FASTLLM_ROOT}/third_party/flashinfer)

add_library(fastllm SHARED ${FASTLLM_SOURCES} jni_bridge.cpp)
find_library(log-lib log)
target_link_libraries(fastllm ${log-lib})
