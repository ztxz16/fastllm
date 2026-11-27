# FlashInfer v0.6.16.post1 的 CCCL 子模块版本。整个仓库按提交获取，禁止摘取头文件。
set(FASTLLM_CCCL_VERSION "3.3.2")
set(FASTLLM_CCCL_REVISION "876867684f7fac130e0f5911236e0a92a970d4fd")
set(FASTLLM_CCCL_CUB_VERSION 300302)
set(FASTLLM_CCCL_ROOT "" CACHE PATH "Existing clean checkout of the pinned NVIDIA CCCL revision")
option(FASTLLM_FETCH_CCCL "Fetch pinned CCCL when CUDA Toolkit headers lack required APIs" ON)

# 从当前 NVCC 的 Toolkit 查找完整头文件入口，不以 GPU 架构推断库版本。
set(_fastllm_toolkit_cccl_version "")
set(_fastllm_toolkit_cccl_ready OFF)
foreach(_fastllm_cuda_include IN LISTS CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    foreach(_fastllm_cccl_include "${_fastllm_cuda_include}" "${_fastllm_cuda_include}/cccl")
        if(EXISTS "${_fastllm_cccl_include}/cub/version.cuh")
            file(STRINGS "${_fastllm_cccl_include}/cub/version.cuh" _fastllm_cub_version_line
                REGEX "^#define CUB_VERSION +[0-9]+")
            if(_fastllm_cub_version_line MATCHES "CUB_VERSION +([0-9]+)")
                set(_fastllm_toolkit_cccl_version "${CMAKE_MATCH_1}")
            endif()
            if(EXISTS "${_fastllm_cccl_include}/cuda/__functional/maximum.h" AND
               EXISTS "${_fastllm_cccl_include}/cuda/__functional/minimum.h" AND
               EXISTS "${_fastllm_cccl_include}/cuda/__cmath/fast_modulo_division.h" AND
               EXISTS "${_fastllm_cccl_include}/thrust/version.h")
                set(_fastllm_toolkit_cccl_ready ON)
            endif()
            break()
        endif()
    endforeach()
    if(NOT "${_fastllm_toolkit_cccl_version}" STREQUAL "")
        break()
    endif()
endforeach()

if(NOT FASTLLM_CCCL_ROOT AND _fastllm_toolkit_cccl_ready)
    message(STATUS "FastLLM CCCL: CUDA Toolkit headers (CUB ${_fastllm_toolkit_cccl_version})")
    return()
endif()

# 新版 CCCL 可以搭配受支持的旧 Toolkit；不允许用固定旧头文件覆盖更高版本。
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.0 OR
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
    message(FATAL_ERROR "Pinned CCCL ${FASTLLM_CCCL_VERSION} requires a CUDA 12.x or 13.x Toolkit")
endif()
if("${_fastllm_toolkit_cccl_version}" STREQUAL "")
    message(FATAL_ERROR "Cannot determine the CUDA Toolkit CUB version; check CUDA include paths")
endif()
if(_fastllm_toolkit_cccl_version GREATER FASTLLM_CCCL_CUB_VERSION)
    message(FATAL_ERROR "Pinned CCCL would downgrade the CUDA Toolkit headers; use its complete newer CCCL installation")
endif()

find_package(Git REQUIRED)
if(FASTLLM_CCCL_ROOT)
    get_filename_component(_fastllm_cccl_source "${FASTLLM_CCCL_ROOT}" REALPATH)
else()
    if(NOT FASTLLM_FETCH_CCCL)
        message(FATAL_ERROR "FlashInfer needs newer CCCL headers. Set FASTLLM_CCCL_ROOT to revision ${FASTLLM_CCCL_REVISION}, or enable FASTLLM_FETCH_CCCL")
    endif()
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR "Fetching header-only CCCL requires CMake 3.18 or newer, or set FASTLLM_CCCL_ROOT")
    endif()
    include(FetchContent)
    # 只获取完整源码，不进入 CCCL 的构建、安装或测试子项目。
    FetchContent_Declare(fastllm_cccl
        GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
        GIT_TAG ${FASTLLM_CCCL_REVISION}
        SOURCE_SUBDIR _fastllm_header_only)
    FetchContent_MakeAvailable(fastllm_cccl)
    set(_fastllm_cccl_source "${fastllm_cccl_SOURCE_DIR}")
endif()

# 离线目录与自动获取目录均校验固定版本及工作区完整性，拒绝混入本地补丁。
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_fastllm_cccl_source}" rev-parse HEAD
    RESULT_VARIABLE _fastllm_cccl_git_result OUTPUT_VARIABLE _fastllm_cccl_head
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT "${_fastllm_cccl_git_result}" STREQUAL "0" OR
   NOT "${_fastllm_cccl_head}" STREQUAL "${FASTLLM_CCCL_REVISION}")
    message(FATAL_ERROR "CCCL must be a complete Git checkout at ${FASTLLM_CCCL_REVISION}: ${_fastllm_cccl_source}")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${_fastllm_cccl_source}"
    status --porcelain --untracked-files=normal
    RESULT_VARIABLE _fastllm_cccl_status_result OUTPUT_VARIABLE _fastllm_cccl_status
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT "${_fastllm_cccl_status_result}" STREQUAL "0" OR NOT "${_fastllm_cccl_status}" STREQUAL "")
    message(FATAL_ERROR "CCCL checkout has local changes; restore the complete pinned version: ${_fastllm_cccl_source}")
endif()
foreach(_fastllm_cccl_header
    cub/cub/version.cuh thrust/thrust/version.h
    libcudacxx/include/cuda/functional libcudacxx/include/cuda/cmath
    libcudacxx/include/cuda/__functional/maximum.h
    libcudacxx/include/cuda/__functional/minimum.h
    libcudacxx/include/cuda/__cmath/fast_modulo_division.h)
    if(NOT EXISTS "${_fastllm_cccl_source}/${_fastllm_cccl_header}")
        message(FATAL_ERROR "Incomplete CCCL checkout: missing ${_fastllm_cccl_header}")
    endif()
endforeach()

# 三个组件必须来自同一版本。使用普通 -I 提前于 NVCC 隐式 Toolkit 头文件，不能使用 SYSTEM。
include_directories(BEFORE
    "${_fastllm_cccl_source}/cub"
    "${_fastllm_cccl_source}/thrust"
    "${_fastllm_cccl_source}/libcudacxx/include")
message(STATUS "FastLLM CCCL: ${FASTLLM_CCCL_VERSION} (${FASTLLM_CCCL_REVISION}) at ${_fastllm_cccl_source}")
