// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "IncrementalStringFinder.hpp"

#include <vector>
#include <queue>
#include <string>
#include <string_view>

namespace bl::llama {
class BL_LLAMA_API AntipromptManager {
public:
    AntipromptManager() = default;

    // adds new antiprompt to check
    void addAntiprompt(std::string_view antiprompt);

    // feed each antiprompt with the text
    std::string feedGeneratedText(std::string_view text);

    // reset the state of all antiprompts
    void reset();

    // clear all stored antiprompts
    void clear();

    // check if there are any antiprompts that are in intermidiate state
    bool hasRunningAntiprompts();
private:
    std::vector<IncrementalStringFinder> m_antiprompts;
};
} // namespace bl::llama
