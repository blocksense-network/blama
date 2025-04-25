# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
macro(bstl_test test)
    add_doctest_lib_test(${test} bstl t-${test}.cpp)
endmacro()

bstl_test(iile)
bstl_test(mem_ext)
bstl_test(move)
bstl_test(move_capture)
bstl_test(u8c)

icm_add_multiple_build_failure_tests(
    SOURCES
        t-move-bf.cpp
    LIBRARIES bstl
    PREFIX bstl
    FOLDER bf-test
)
