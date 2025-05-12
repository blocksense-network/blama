# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(blama
    LANGUAGES C CXX
    VERSION 0.0.1
)

# cpm
include(./get_cpm.cmake)

#######################################
# cmake lib
CPMAddPackage(gh:iboB/icm@1.5.5)

list(APPEND CMAKE_MODULE_PATH
    "${icm_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)

set(ICM_DEV_C_STANDARD 11)
set(ICM_DEV_CXX_STANDARD 20)

include(icm_dev_mode)
include(icm_add_lib)
include(icm_build_failure_testing)
include(bl_dep)
include(bl_lib_subdirs)

#######################################
# options

option(BUILD_SHARED_LIBS "${PROJECT_NAME}: build shared libraries" ON)
option(BLAMA_BUILD_TESTS "${PROJECT_NAME}: enable testing and build tests" ${ICM_DEV_MODE})
option(BLAMA_BUILD_EXAMPLES "${PROJECT_NAME}: build examples" ${ICM_DEV_MODE})
option(BLAMA_BUILD_TUTORIALS "${PROJECT_NAME}: build tutorials" ${ICM_DEV_MODE})
option(BLAMA_BUILD_BENCH "${PROJECT_NAME}: build benchmarks" ${ICM_DEV_MODE})
option(BLAMA_BUILD_SANDBOX "${PROJECT_NAME}: build sandbox project (dev experiments)" ${ICM_DEV_MODE})
mark_as_advanced(BLAMA_BUILD_SANDBOX)

#######################################
# global packages

if(BLAMA_BUILD_TESTS)
    if(NOT TARGET doctest::util)
        CPMAddPackage(gh:iboB/doctest-util@0.1.3)
        set_target_properties(doctest PROPERTIES FOLDER test)
        set_target_properties(doctest-main PROPERTIES FOLDER test)
    endif()
    enable_testing()
endif()

if(BLAMA_BUILD_TESTS OR BLAMA_BUILD_EXAMPLES)
    CPMAddPackage(
        NAME ac-test-data-llama
        VERSION 1.0.0
        GIT_REPOSITORY https://huggingface.co/alpaca-core/ac-test-data-llama
        GIT_TAG 164e484873785c804fe724f88bdb96088b573ebc
    )
endif()

#######################################
# code
add_subdirectory(common)

add_subdirectory(inference)

add_subdirectory(server)

if(BLAMA_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
