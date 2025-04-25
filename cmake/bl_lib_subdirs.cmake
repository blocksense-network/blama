# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#

# macros which optionally add subdirs according to the project options

macro(bl_add_test_subdir)
    if(BLAMA_BUILD_TESTS)
        add_subdirectory(test)
    endif()
endmacro()

macro(bl_add_example_subdir)
    if(BLAMA_BUILD_EXAMPLES)
        add_subdirectory(example)
    endif()
endmacro()
