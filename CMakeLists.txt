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
include(bl_dep)

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
# code
#~ add_subdirectory(code)

#~ if(BLAMA_BUILD_TESTS)
#~     enable_testing()
#~     add_subdirectory(test)
#~ endif()

#~ if(BLAMA_BUILD_BENCH)
#~     add_subdirectory(bench)
#~ endif()

#~ if(BLAMA_BUILD_EXAMPLES)
#~     add_subdirectory(example)
#~ endif()

if(BLAMA_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()
