# Check both process status and the final success marker. CTest's
# PASS_REGULAR_EXPRESSION alone can override a nonzero process exit code.
execute_process(
    COMMAND "${TEST_EXECUTABLE}" ${TEST_ARGUMENTS}
    RESULT_VARIABLE test_status
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr
    TIMEOUT "${TEST_TIMEOUT}")
set(test_output "${test_stdout}${test_stderr}")
if (NOT "${test_output}" STREQUAL "")
    message("${test_output}")
endif()
if (test_output MATCHES "FastLLM Error:|FAIL")
    message(FATAL_ERROR "Test reported an error")
endif()
if ("${test_status}" STREQUAL "77")
    # CTest recognizes this marker via SKIP_REGULAR_EXPRESSION. On older
    # CTest versions without that property, fail instead of reporting PASS.
    message(FATAL_ERROR "FASTLLM_TEST_SKIP_NO_DEVICE")
endif()
if (NOT "${test_status}" STREQUAL "0")
    message(FATAL_ERROR "Test process failed: ${test_status}")
endif()
if (NOT test_output MATCHES "${TEST_PASS_REGEX}")
    message(FATAL_ERROR "Test did not print its completion marker")
endif()
