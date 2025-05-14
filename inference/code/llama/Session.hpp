// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Token.hpp"
#include <span>
#include <utility>
#include <exception>
#include <coroutine>
#include <vector>
#include <cassert>

struct llama_context;

namespace bl::llama {
class Instance;

struct TokenPrediction {
    Token token;
    TokenDataVector logits;
};

class BL_LLAMA_API Session {
public:
    struct InitParams {
        uint32_t gaFactor = 1; // group-attention factor
        uint32_t gaWidth = 512; // group-attention width

        // if true, the inference tries to extend the context by truncating previous tokens
        // only used if gaFactor == 1
        bool infiniteContext = true;
    };
    Session(Instance& instance, llama_context* ctx, InitParams params);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session();

    // initial functions to prepare the session
    void setInitialPrompt(std::span<const Token> prompt);
    bool setState(std::span<uint8_t> state);

    // main functions to interact with the model
    void pushPrompt(std::span<const Token> prompt, std::span<const Token> postfix = {});
    TokenPrediction getToken();
    struct CompleteParams{
        std::span<const Token> prompt;
        std::span<const Token> postfix;
        int32_t maxTokens = 0;
    };
    std::vector<TokenPrediction> complete(CompleteParams params);

    std::vector<TokenPrediction> fillCtx(std::span<TokenPrediction> tokens);
    std::vector<uint8_t> getState();
private:
    enum class Source {
        InitialPrompt,
        InteractivePrompt,
        Generated
    };

    void doDecode(std::span<const Token> tokens, Source src);
    void flushPendingState();
    TokenDataVector getLogitsFromCtx(int32_t topK);
    TokenDataVector getLogitsFromCtx(TokenDataVector tokens);

    struct State {
        enum class Phase {
            Initial,
            Generating
        };

        Phase m_phase = Phase::Initial;
        Token m_currToken = Token_Invalid;

        unsigned maxTokens = 0;
        unsigned numKeep = 0;
        uint32_t gaIndex = 0; // number of grouped KV tokens (only used if params.gaFactor > 1)
        uint32_t numPast = 0; // number of tokens in the context (that's prompts + generated)
    };

    Instance& m_instance;
    llama_context* m_ctx;
    InitParams m_params;
    State m_state;
};

} // namespace bl::llama
