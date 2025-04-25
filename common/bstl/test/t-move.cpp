// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>

#include <bstl/move.hpp>

#include <string>

TEST_CASE("move") {
    auto i = bstl::move(7);
    CHECK(i == 7);
    const auto str = bstl::move(std::string("str"));
    CHECK(str == "str");
    // won't compile:
    // auto s2 = bstl::move(str);
}
