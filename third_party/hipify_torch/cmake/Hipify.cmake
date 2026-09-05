# cmake file to trigger hipify

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(HIPIFY_DICT_FILE ${CMAKE_BINARY_DIR}/hipify_output_dict_dump.txt)
set(_temp_file_cuda_to_hip_list "cuda_to_hip_list")

# Get the hipify_cli.py source directory from current directory by going to parent directory
get_filename_component(HIPIFY_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

include(CMakeParseArguments)

# This is an internal function, NOT an API
# TODO: Move this function into a separate file
function(write_file_list FILE_SUFFIX INPUT_LIST)
  message(STATUS "Writing ${FILE_SUFFIX} into file - file_${FILE_SUFFIX}.txt")
  set(_FULL_FILE_NAME "${CMAKE_BINARY_DIR}/${_temp_file_cuda_to_hip_list}_${FILE_SUFFIX}.txt")
  file(WRITE ${_FULL_FILE_NAME} "")
  foreach(_SOURCE_FILE ${INPUT_LIST})
    if(NOT IS_ABSOLUTE ${_SOURCE_FILE})
      file(APPEND ${_FULL_FILE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_FILE})
    else()
      file(APPEND ${_FULL_FILE_NAME} ${_SOURCE_FILE})
    endif()
    file(APPEND ${_FULL_FILE_NAME} "\n")
  endforeach()
endfunction()

# This is an internal function, NOT an API
# TODO: Move this function into a separate file
function(get_file_list FILE_SUFFIX OUTPUT_LIST)
  set(_FULL_FILE_NAME "${CMAKE_BINARY_DIR}/${_temp_file_cuda_to_hip_list}_${FILE_SUFFIX}.txt")
  file(STRINGS ${_FULL_FILE_NAME} _FILE_LIST)
  set(${OUTPUT_LIST}_HIP ${_FILE_LIST} PARENT_SCOPE)
endfunction()

# This is an internal function, NOT an API
# TODO: Move this function into a separate file
function(update_list_with_hip_files FILE_SUFFIX)
  set(_SCRIPTS_DIR ${HIPIFY_DIR}/tools)
  set(_FULL_FILE_NAME "${CMAKE_BINARY_DIR}/${_temp_file_cuda_to_hip_list}_${FILE_SUFFIX}.txt")
  set(_EXE_COMMAND
    ${_SCRIPTS_DIR}/replace_cuda_with_hip_files.py
    --io-file ${_FULL_FILE_NAME}
    --dump-dict-file ${HIPIFY_DICT_FILE})
  execute_process(
    COMMAND ${_EXE_COMMAND}
    RESULT_VARIABLE _return_value)
  if (NOT _return_value EQUAL 0)
    message(FATAL_ERROR "Failed to get the list of hipified files!")
  endif()
endfunction()

# Function -> get_hipified_list()
# - INPUT_LIST - Input list containing cuda files
# - OUTPUT_LIST - Returns an output list containing the HIP files
function(get_hipified_list INPUT_LIST OUTPUT_LIST)
  string(RANDOM LENGTH 16 RAND_STRING)
  set(TEMP_FILE_NAME "tmp_${RAND_STRING}")
  file(REMOVE ${TEMP_FILE_NAME})

  write_file_list("${TEMP_FILE_NAME}" "${INPUT_LIST}")
  update_list_with_hip_files("${TEMP_FILE_NAME}")
  get_file_list("${TEMP_FILE_NAME}" __temp_srcs)

  set(${OUTPUT_LIST} ${__temp_srcs_HIP} PARENT_SCOPE)
endfunction()

# Function -> hipify()
# - Takes 3 optional arguments, either CUDA_SOURCE_DIR or CONFIG_FILE argument is required
# - CUDA_SOURCE_DIR --> Full path of input cuda source directory which needs to be hipified.
# - HIP_SOURCE_DIR --> Full path of output directory where the hipified files will be placed.
#                      If not provided, it is set to CUDA_SOURCE_DIR.
# - CONFIG_FILE --> JSON format file, which provides additional arguments for hipify_cli.py file.
#                   When set, it is having higher precendence over CUDA_SOURCE_DIR/HIP_SOURCE_DIR.
function(hipify)
  set(flags)
  set(singleValueArgs CUDA_SOURCE_DIR HIP_SOURCE_DIR CONFIG_FILE CUSTOM_MAP_FILE)
  set(multiValueArgs HEADER_INCLUDE_DIR IGNORES)

  cmake_parse_arguments(HIPIFY "${flags}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  if(HIPIFY_CONFIG_FILE)
    set(HIPIFY_COMMAND
      ${HIPIFY_DIR}/hipify_cli.py
      --config-json ${HIPIFY_CONFIG_FILE}
      --dump-dict-file ${HIPIFY_DICT_FILE}
    )
  elseif(HIPIFY_CUDA_SOURCE_DIR)
    if(NOT HIPIFY_HIP_SOURCE_DIR)
      set(HIPIFY_HIP_SOURCE_DIR ${HIPIFY_CUDA_SOURCE_DIR})
    endif()
    set(HIPIFY_COMMAND
      ${HIPIFY_DIR}/hipify_cli.py
      --project-directory ${HIPIFY_CUDA_SOURCE_DIR}
      --output-directory ${HIPIFY_HIP_SOURCE_DIR}
      --header-include-dirs [${HIPIFY_HEADER_INCLUDE_DIR}]
      --ignores [${HIPIFY_IGNORES}]
      --dump-dict-file ${HIPIFY_DICT_FILE}
    )
    if (HIPIFY_CUSTOM_MAP_FILE)
      list(APPEND HIPIFY_COMMAND --custom-map-json ${HIPIFY_CUSTOM_MAP_FILE})
    endif()
  else()
    message(FATAL_ERROR "Wrong invocation, either CUDA_SOURCE_DIR or CONFIG_FILE input parameter is required")
  endif()

  execute_process(
   COMMAND ${HIPIFY_COMMAND}
   RESULT_VARIABLE hipify_return_value
  )
  if (NOT hipify_return_value EQUAL 0)
    message(FATAL_ERROR "Failed to hipify files!")
  endif()
endfunction()
