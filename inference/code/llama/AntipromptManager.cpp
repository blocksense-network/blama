// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "AntipromptManager.hpp"

namespace bl::llama {

void AntipromptManager::addAntiprompt(std::string_view antiprompt) {
    m_antiprompts.push_back(std::string(antiprompt));
}

std::string AntipromptManager::feedGeneratedText(std::string_view text) {
    for (auto& ap : m_antiprompts) {
        int found = ap.feedText(text);
        if (found > 0) {
            reset();
            return found == 0 ?
                        ap.getString():
                        ap.getString() + std::string(text.substr(found, text.length()));
        }
    }

    return {};
}

void AntipromptManager::reset() {
    for (auto& ap : m_antiprompts) {
        ap.reset();
    }
}

void AntipromptManager::clear() {
    m_antiprompts.clear();
}

bool AntipromptManager::hasRunningAntiprompts() {
    for (auto& ap : m_antiprompts) {
        if (ap.getCurrentPos() > 0) {
            return true;
        }
    }

    return false;
}

} // namespace bl::llama
