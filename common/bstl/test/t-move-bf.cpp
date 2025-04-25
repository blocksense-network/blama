// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <bstl/move.hpp>

#include <string>

int main() {
    const std::string cstr = "asdf";
    // build error: cannot move a const object
    auto fail = bstl::move(cstr);
}
