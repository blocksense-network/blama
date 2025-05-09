// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <cstdint>
#include <vector>

namespace bl::llama {
using Token = std::int32_t;
inline constexpr Token Token_Invalid = -1;

struct TokenData {
    Token token;
    float logit;
};

using TokenDataVector = std::vector<TokenData>;
} // namespace bl::llama
