cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

option(USE_CUDA "use CUDA" OFF)
option(CUDA_NO_TENSOR_CORE "Optimize for legacy CUDA GPUs which has Tensor Core." OFF)

option(USE_TFACC "use tfacc" OFF)

option(PY_API "python api" OFF)

option(USE_MMAP "use mmap" OFF)

option(USE_SENTENCEPIECE "use sentencepiece" OFF)

option(USE_IVCOREX "use iluvatar corex gpu" OFF)

option(BUILD_CLI "build cli" OFF)

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "native")
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")

message(STATUS "CUDA_ARCH: ${CUDA_ARCH}")

message(STATUS "USE_TFACC: ${USE_TFACC}")

message(STATUS "For legacy CUDA GPUs: ${CUDA_NO_TENSOR_CORE}")

message(STATUS "PYTHON_API: ${PY_API}")

message(STATUS "BUILD_CLI: ${BUILD_CLI}")

message(STATUS "USE_SENTENCEPIECE: ${USE_SENTENCEPIECE}")

message(STATUS "USE_IVCOREX: ${USE_IVCOREX}")

set(CMAKE_BUILD_TYPE "Release")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /std:c++17 /arch:AVX2 /source-charset:utf-8")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
endif()


message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
file(GLOB GRAPH_MODEL_FILES "src/models/graph/*.cpp")
file(GLOB CPU_DEVICE_FILES "src/devices/cpu/*.cpp")
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/model.cpp src/executor.cpp src/template.cpp src/graph.cpp
        src/devices/cpu/cpudevice.cpp src/devices/cpu/cpudevicebatch.cpp
        src/models/graphllm.cpp src/models/chatglm.cpp src/models/moss.cpp src/models/llama.cpp src/models/qwen.cpp src/models/basellm.cpp
        src/models/glm.cpp src/models/minicpm.cpp src/models/minicpm3.cpp src/models/internlm2.cpp src/models/bert.cpp src/models/moe.cpp src/models/deepseekv2.cpp
        src/models/phi3.cpp src/models/xlmroberta.cpp src/models/cogvlm.cpp
        third_party/json11/json11.cpp
        ${CPU_DEVICE_FILES}
        ${GRAPH_MODEL_FILES})

include_directories(include)
include_directories(include/utils)
include_directories(include/models)
include_directories(include/devices/cpu)
include_directories(third_party/json11)

if (USE_MMAP)
    add_compile_definitions(USE_MMAP)
endif()

if (USE_SENTENCEPIECE)
    set(CMAKE_CXX_STANDARD 17)
    add_compile_definitions(USE_SENTENCEPIECE)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} sentencepiece)
endif()

if (USE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(USE_CUDA)
    if (CUDA_NO_TENSOR_CORE)
        add_compile_definitions(CUDA_NO_TENSOR_CORE)
    endif()
    include_directories(include/devices/cuda)
    #message(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp src/devices/cuda/fastllm-cuda.cu)

    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES ${FASTLLM_CUDA_SOURCES} src/devices/multicuda/multicudadevice.cpp src/devices/multicuda/fastllm-multicuda.cu)

    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if (USE_IVCOREX)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cudart)
    set(CMAKE_CUDA_ARCHITECTURES ${IVCOREX_ARCH})
endif()

if (USE_TFACC)
    #execute_process(
    #    COMMAND "./insmodTFDriver.sh"
    #    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thinkforce"
    #)

    add_compile_definitions(USE_TFACC)
    set(FASTLLM_TFACC_SOURCES src/devices/tfacc/tfaccdevice.cpp src/devices/tfacc/fastllm-tfacc.cpp)
    include_directories(include/devices/tfacc)
endif()

if (USE_NUMA)
    add_compile_definitions(USE_NUMA)
    set(FASTLLM_CXX_SOURCES ${FASTLLM_CXX_SOURCES} src/devices/numa/numadevice.cpp src/devices/numa/fastllm-numa.cpp src/devices/numa/computeserver.cpp src/devices/numa/kvcache.cpp)
    include_directories(include/devices/numa)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} numa)
endif()

if (USE_TOPS) # For suiyuan
    add_compile_definitions(USE_TOPS)
    set(FASTLLM_TOPS_SOURCES src/devices/tops/topsdevice.cpp)
    include_directories(include/devices/tops)

    include_directories(/usr/include/gcu/)
    include_directories(/opt/tops/include)
    include_directories(/usr/include/gcu/topsdnn)
    include_directories(/usr/src/topsaten_samples/common)

    link_directories(/usr/lib)
    link_directories(/opt/tops/lib)
    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} topsaten /opt/tops/lib/libtopsrt.so)
    #target_link_libraries(main rtcu)
endif()

if (PY_API)
    if(POLICY CMP0148)
        cmake_policy(SET CMP0148 NEW)
    endif()
    set(PYBIND third_party/pybind11)
    add_subdirectory(${PYBIND})
    add_compile_definitions(PY_API)
    
    set(Python3_ROOT_DIR "/usr/local/python3.10.6/bin/")
    find_package(Python3 REQUIRED)

    include_directories(third_party/pybind11/include)
    file(GLOB FASTLLM_CXX_HEADERS include/**/*.h)
    add_library(pyfastllm MODULE src/pybinding.cpp ${FASTLLM_CXX_SOURCES} ${FASTLLM_CXX_HEADERS} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES})
    target_link_libraries(pyfastllm PUBLIC pybind11::module ${FASTLLM_LINKED_LIBS})
    pybind11_extension(pyfastllm)
else()
add_library(fastllm OBJECT
            ${FASTLLM_CXX_SOURCES}
            ${FASTLLM_CUDA_SOURCES}
            ${FASTLLM_TFACC_SOURCES}
            ${FASTLLM_TOPS_SOURCES}
            )
target_link_libraries(fastllm PUBLIC ${FASTLLM_LINKED_LIBS})

add_executable(main main.cpp)
target_link_libraries(main fastllm)

add_executable(quant tools/src/quant.cpp)
target_link_libraries(quant fastllm)

add_executable(testOps test/ops/cppOps.cpp)
target_link_libraries(testOps fastllm)

add_executable(webui example/webui/webui.cpp)
target_link_libraries(webui fastllm)
add_custom_command(
        TARGET webui
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory web
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/example/webui/web ${CMAKE_BINARY_DIR}/web
)

add_executable(benchmark example/benchmark/benchmark.cpp)
target_link_libraries(benchmark fastllm)

add_executable(apiserver example/apiserver/apiserver.cpp)
target_link_libraries(apiserver fastllm)

if (BUILD_CLI)
    add_executable(FastllmStudio_cli example/FastllmStudio/cli/cli.cpp example/FastllmStudio/cli/ui.cpp)
    target_link_libraries(FastllmStudio_cli fastllm)
endif()

add_library(fastllm_tools SHARED ${FASTLLM_CXX_SOURCES} ${FASTLLM_CUDA_SOURCES} ${FASTLLM_TFACC_SOURCES} ${FASTLLM_TOPS_SOURCES} tools/src/pytools.cpp)
target_link_libraries(fastllm_tools PUBLIC ${FASTLLM_LINKED_LIBS})

if (${CMAKE_HOST_WIN32})
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/$(Configuration)/fastllm_tools.dll
    )
else()
    add_custom_command(
            TARGET fastllm_tools
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory tools
            COMMAND ${CMAKE_COMMAND} -E make_directory tools/ftllm
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/fastllm_pytools ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/tools/scripts ${CMAKE_BINARY_DIR}/tools/.
            COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libfastllm_tools.* ${CMAKE_BINARY_DIR}/tools/ftllm/.
            COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/libfastllm_tools.*
    )
endif()

endif()
