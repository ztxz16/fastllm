cmake_minimum_required(VERSION 3.5)

project(fastllm LANGUAGES CXX)

option(USE_CUDA "use CUDA" OFF)
option(CUDA_NO_TENSOR_CORE "Optimize for legacy CUDA GPUs which has Tensor Core." OFF)

option(USE_ROCM "use ROCm" OFF)

option(USE_TFACC "use tfacc" OFF)

option(PY_API "python api" OFF)

option(USE_MMAP "use mmap" OFF)

option(USE_SENTENCEPIECE "use sentencepiece" OFF)

option(USE_IVCOREX "use iluvatar corex gpu" OFF)

option(BUILD_CLI "build cli" OFF)

option(UNIT_TEST "build unit tests" OFF)

if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "native")
endif()

# ROCm device detection function
function(detect_rocm_devices)
    if(USE_ROCM)
        # Try to find rocminfo
        find_program(ROCMINFO rocminfo PATHS ${ROCM_PATH}/bin /opt/rocm/bin)
        
        if(ROCMINFO)
            # Execute rocminfo to get device info
            execute_process(
                COMMAND ${ROCMINFO}
                OUTPUT_VARIABLE ROCMINFO_OUTPUT
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            
            # Parse the output to find device architectures
            set(DETECTED_ARCHS "")
            set(HAS_MI50 FALSE)
            
            # Extract gfx architectures from rocminfo output
            string(REGEX MATCHALL "Name:[ \t]+gfx[0-9a-f]+" GFX_MATCHES "${ROCMINFO_OUTPUT}")
            foreach(MATCH ${GFX_MATCHES})
                string(REGEX REPLACE "Name:[ \t]+" "" ARCH ${MATCH})
                list(APPEND DETECTED_ARCHS ${ARCH})
                if(${ARCH} STREQUAL "gfx906")
                    set(HAS_MI50 TRUE)
                endif()
            endforeach()
            
            # Also check for device names containing MI50
            string(REGEX MATCH "Marketing Name:[ \t]+[^\n]*MI50" MI50_MATCH "${ROCMINFO_OUTPUT}")
            if(MI50_MATCH)
                set(HAS_MI50 TRUE)
            endif()
            
            # Remove duplicates
            if(DETECTED_ARCHS)
                list(REMOVE_DUPLICATES DETECTED_ARCHS)
            endif()
            
            # Set parent scope variables
            set(ROCM_DETECTED_ARCHS "${DETECTED_ARCHS}" PARENT_SCOPE)
            set(ROCM_HAS_MI50 ${HAS_MI50} PARENT_SCOPE)
        else()
            message(WARNING "rocminfo not found, cannot auto-detect ROCm devices")
            set(ROCM_DETECTED_ARCHS "" PARENT_SCOPE)
            set(ROCM_HAS_MI50 FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Call device detection
detect_rocm_devices()

# Set ROCM_ARCH based on detection or use default
if(USE_ROCM)
    if(NOT DEFINED ROCM_ARCH)
        if(ROCM_DETECTED_ARCHS)
            # Use detected architectures
            set(ROCM_ARCH "${ROCM_DETECTED_ARCHS}")
            message(STATUS "Auto-detected ROCm architectures: ${ROCM_ARCH}")
        else()
            # Use default if detection failed
            set(ROCM_ARCH "gfx908;gfx90a;gfx1100")
            message(STATUS "ROCM_ARCH not set and auto-detection failed, using default: ${ROCM_ARCH}")
        endif()
    else()
        message(STATUS "Using user-specified ROCM_ARCH: ${ROCM_ARCH}")
        # Check if user specified gfx906
        if("gfx906" IN_LIST ROCM_ARCH)
            set(ROCM_HAS_MI50 TRUE)
            message(STATUS "gfx906 (MI50) found in user-specified ROCM_ARCH")
        endif()
    endif()
    
    # Report MI50 detection status
    if(ROCM_HAS_MI50)
        message(STATUS "MI50 device detected (gfx906)")
    endif()
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")

message(STATUS "USE_ROCM: ${USE_ROCM}")

message(STATUS "CUDA_ARCH: ${CUDA_ARCH}")

message(STATUS "USE_TFACC: ${USE_TFACC}")

message(STATUS "For legacy CUDA GPUs: ${CUDA_NO_TENSOR_CORE}")

message(STATUS "PYTHON_API: ${PY_API}")

message(STATUS "BUILD_CLI: ${BUILD_CLI}")

message(STATUS "USE_SENTENCEPIECE: ${USE_SENTENCEPIECE}")

message(STATUS "USE_IVCOREX: ${USE_IVCOREX}")

message(STATUS "MAKE_WHL_X86: ${MAKE_WHL_X86}")

set(CMAKE_BUILD_TYPE "Release")

string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+$" "\\1" CXX_MAJOR_VERSION "${CMAKE_CXX_COMPILER_VERSION}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    string(REPLACE "/Ob2" "/Ob1 /Gy" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX /std:c++17 /arch:AVX2 /source-charset:utf-8")
    if (CXX_MAJOR_VERSION GREATER_EQUAL 19 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX10.1"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX10.1"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512vnni.cpp"
            PROPERTIES
            COMPILE_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX10.1"
        )
    endif()
else()
    if (MAKE_WHL_X86)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -mavx -mavx2 -mf16c -mfma -static-libstdc++ -static-libgcc")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --std=c++17 -O2 -march=native")
    endif()
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    if (CXX_MAJOR_VERSION GREATER_EQUAL 10 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512f"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512bw"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512f.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512vl"
        )
        
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512bf16"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512f"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512bw"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512bf16.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512vl"
        )

        set_source_files_properties(
            "src/devices/cpu/avx512vnni.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512vnni"
        )
        set_source_files_properties(
            "src/devices/cpu/avx512vnni.cpp"
            PROPERTIES
            COMPILE_OPTIONS "-mavx512f"
        )
    endif()
endif()


message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
file(GLOB GRAPH_MODEL_FILES "src/models/graph/*.cpp")
file(GLOB CPU_DEVICE_FILES "src/devices/cpu/*.cpp")
set(FASTLLM_CXX_SOURCES src/fastllm.cpp src/device.cpp src/model.cpp src/executor.cpp src/template.cpp src/graph.cpp
        src/devices/cpu/cpudevice.cpp src/devices/cpu/cpudevicebatch.cpp
        src/models/graphllm.cpp src/models/chatglm.cpp src/models/moss.cpp src/models/llama.cpp src/models/qwen.cpp src/models/basellm.cpp
        src/models/glm.cpp src/models/minicpm.cpp src/models/minicpm3.cpp src/models/internlm2.cpp src/models/bert.cpp src/models/moe.cpp src/models/deepseekv2.cpp
        src/models/phi3.cpp src/models/xlmroberta.cpp src/models/cogvlm.cpp src/models/qwen3.cpp src/models/qwen3_moe.cpp src/models/minimax.cpp src/models/hunyuan.cpp
        src/models/ernie4_5.cpp src/models/pangu_moe.cpp
        third_party/json11/json11.cpp
        third_party/gguf/gguf.cpp third_party/gguf/ggml-quant.cpp third_party/gguf/ggml-iqk.cpp third_party/gguf/ggml-dequantize.cpp
        third_party/gguf/gguf-adapter.cpp
        ${CPU_DEVICE_FILES}
        ${GRAPH_MODEL_FILES})

include_directories(include)
include_directories(include/utils)
include_directories(include/models)
include_directories(include/devices/cpu)
include_directories(third_party/json11)
include_directories(third_party/gguf)

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
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/cuda/fastllm-cuda.cu src/devices/cuda/fastllm-ggml-cuda.cu)

    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES ${FASTLLM_CUDA_SOURCES} src/devices/multicuda/multicudadevice.cpp src/devices/multicuda/fastllm-multicuda.cu)

    set(FASTLLM_LINKED_LIBS ${FASTLLM_LINKED_LIBS} cublas)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
endif()

if(USE_ROCM)
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/third_party/hipify_torch/cmake")
    include(Hipify)
    enable_language(HIP)

    # remove target hip dirs first
    EXECUTE_PROCESS(COMMAND rm -rf ${PROJECT_SOURCE_DIR}/src/devices/hip)
    EXECUTE_PROCESS(COMMAND rm -rf ${PROJECT_SOURCE_DIR}/src/devices/multihip)
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/cuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/hip")
    hipify(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src/devices/multicuda HIP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/devices/multihip")
    include_directories(include/devices/cuda)
    include_directories(include/devices/multicuda)
    set(FASTLLM_CUDA_SOURCES src/devices/cuda/cudadevice.cpp src/devices/cuda/cudadevicebatch.cpp 
        src/devices/hip/fastllm-hip.hip src/devices/hip/fastllm-ggml-hip.hip 
        src/devices/multicuda/multicudadevice.cpp src/devices/multihip/fastllm-multihip.hip)
    add_compile_definitions(USE_ROCM)
    add_compile_definitions(USE_CUDA)
    if (DEFINED $ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
    set(CMAKE_HIP_COMPILER_ROCM_ROOT ${ROCM_PATH})
    set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin/amdclang++")
    message(STATUS "CMAKE_HIP_COMPILER_ROCM_ROOT: ${CMAKE_HIP_COMPILER_ROCM_ROOT}")
    message(STATUS "CMAKE_HIP_COMPILER: ${CMAKE_HIP_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
    
    set(CMAKE_PREFIX_PATH ${ROCM_PATH})
    find_package(hip REQUIRED) 
    find_package(hipblas REQUIRED)
    find_package(rocprim REQUIRED)
    list(APPEND FASTLLM_LINKED_LIBS hip::device roc::hipblas roc::rocprim)
    set(CMAKE_HIP_ARCHITECTURES ${ROCM_ARCH})
    add_compile_definitions(HIPBLAS_V2)

    # Check if gfx906(MI50) is in the architecture list
    set(HAS_MI50_IN_TARGETS FALSE)
    foreach(ARCH IN LISTS CMAKE_HIP_ARCHITECTURES)
        if(${ARCH} STREQUAL "gfx906")
            set(HAS_MI50_IN_TARGETS TRUE)
            break()
        endif()
    endforeach()
    
    # Enable MI50 workaround if MI50 is detected or in target architectures
    if(ROCM_HAS_MI50 OR HAS_MI50_IN_TARGETS)
        add_compile_definitions(USE_MI50_WORKAROUND)
        add_compile_definitions(HIP_NO_TENSOR_CORE)
        message(STATUS "MI50 support enabled (USE_MI50_WORKAROUND defined)")
    endif()
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

if (UNIT_TEST)
    add_executable(testOps test/ops/cppOps.cpp)
    target_link_libraries(testOps fastllm)

    add_executable(testTokenizer test/ops/tokenizerTest.cpp)
    target_link_libraries(testTokenizer fastllm)
endif()

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