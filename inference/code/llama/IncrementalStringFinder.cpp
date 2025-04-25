// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "IncrementalStringFinder.hpp"

namespace bl::llama {

IncrementalStringFinder::IncrementalStringFinder(std::string searchStr)
    : m_searchStr(std::move(searchStr))
    , m_currentPos(0)
{}

int IncrementalStringFinder::feedText(std::string_view text) {
    if (m_searchStr.length() == 0) {
        return -1;
    }

    uint32_t promptPos = 0;

    while(promptPos < text.length() && m_currentPos < m_searchStr.length()) {
        if (m_searchStr[m_currentPos] != text[promptPos]) {
            // different character was found
            // need to start from the beginning
            m_currentPos = 0;
        }

        if (m_searchStr[m_currentPos] == text[promptPos]) {
            m_currentPos++;
        }

        promptPos++;
    }

    if (m_currentPos == m_searchStr.length()) {
        m_currentPos = 0;
        return promptPos;
    }

    return -1;
}

void IncrementalStringFinder::reset() {
    m_currentPos = 0;
}
}
