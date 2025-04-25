// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <bstl/u8c.h>
#include <doctest/doctest.h>
#include <string>

TEST_CASE("c_unique_ptr") {
    std::string s = U8C("Hello");
    s += ", World!";
    CHECK(s == "Hello, World!");

    s = U8C("ハロー");
    s += U8C("、ワールド！");
    CHECK(s == U8C("ハロー、ワールド！"));
}
