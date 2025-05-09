// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Token.hpp"
#include <vector>
#include <string>
#include <string_view>

struct llama_vocab;
namespace bl::llama {

class Model;

class BL_LLAMA_API Vocab {
public:
    Vocab(const Model& model);
    ~Vocab();

    std::vector<Token> tokenize(std::string_view text, bool addSpecial, bool parseSpecial) const;

    Token decoderStartToken() const noexcept; // fallback to bos if not available

    bool isEog(Token token) const noexcept;
    int32_t nTokens() const noexcept;

    std::string tokenToString(Token token, bool special = true) const;

    const llama_vocab* lvocab() const { return m_lVocab; }
private:
    const Model& m_model;
    const llama_vocab* m_lVocab;
};

} // namespace bl::llama
