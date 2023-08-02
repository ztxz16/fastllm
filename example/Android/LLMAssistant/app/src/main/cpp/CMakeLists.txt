# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("assistant")
set(CMAKE_BUILD_TYPE "Release")

option(USE_CUDA "use cuda" OFF)

option(PY_API "python api" OFF)

#可以注释掉下面优化选项
#add_definitions(${CMAKE_CXX_FLAGS} "${CMAKE_CXX_FLAGS} -march=armv8.2a+dotprod")
#
#file(GLOB_RECURSE NANODET_SOURCE ../../../../../../../src/*.cpp
#                                 ../../../../../../../src/devices/cpu/*.cpp
#                                 ../../../../../../../src/models/*.cpp)
#
#set(PROJECT_SOURCE
#        ${NANODET_SOURCE}
#        )

set(PROJECT_SOURCE
        ../../../../../../../src/fastllm.cpp
        ../../../../../../../src/device.cpp
        ../../../../../../../src/model.cpp
        ../../../../../../../src/executor.cpp
        ../../../../../../../src/devices/cpu/cpudevice.cpp
        ../../../../../../../src/devices/cpu/cpudevicebatch.cpp
        ../../../../../../../src/models/chatglm.cpp
        ../../../../../../../src/models/moss.cpp
        ../../../../../../../src/models/llama.cpp
        ../../../../../../../src/models/basellm.cpp
        )

include_directories(
        ./
        ../../../../../../../include
        ../../../../../../../include/models
        ../../../../../../../include/utils
        ../../../../../../../include/devices/cpu)

add_library( # Sets the name of the library.
        assistant
        # Sets the library as a shared library.
        SHARED
        # Provides a relative path to your source file(s).
        ${PROJECT_SOURCE} LLMChat.cpp native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib
        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        assistant
        # Links the target library to the log library
        # included in the NDK.
        ${log-lib})

#add_executable(main main.cpp ../../../../../../../src/fastllm.cpp
#        ../../../../../../../src/chatglm.cpp
#        ../../../../../../../src/moss.cpp)