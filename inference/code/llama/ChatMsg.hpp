// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <string>

namespace bl::llama {

struct ChatMsg {
    std::string role; // who sent the message
    std::string text; // the message's content
};

} // namespace bl::llama

