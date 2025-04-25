// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <bstl/iile.h>
#include <doctest/doctest.h>

TEST_CASE("iile") {
    CHECK(iile([] {return 5; }) == 5);
}