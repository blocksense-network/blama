// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace bl::llama {
class BL_LLAMA_API IncrementalStringFinder {
public:
    IncrementalStringFinder(std::string searchStr);

    // incremental search for `m_str` in `text`
    // returns -1 if the search string was not found
    // returns >=0 the search string was found and the count of matched characters of last feed
    int feedText(std::string_view text);

    // reset the `currentPos`
    void reset();

    // return the string that was searched for
    const std::string& getString() const { return m_searchStr; }
    uint16_t getCurrentPos() const { return m_currentPos; }
private:
    std::string m_searchStr;
    uint16_t m_currentPos;
};
}
