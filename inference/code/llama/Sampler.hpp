// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Token.hpp"
#include <itlib/flat_map.hpp>
#include <bstl/mem_ext.hpp>
#include <vector>
#include <string>

struct llama_token_data;
struct llama_context;
struct llama_token_data_array;
struct llama_sampler;
struct llama_vocab;

namespace bl::llama {

class Model;

class BL_LLAMA_API Sampler {
public:
    enum class SamplingType {
        Top_K,
        Top_P,
        Min_P,
        Typical_P,
        Temperature,
        XTC,
        Infill,
    };

    struct Params {
        uint32_t rngSeed = 0; // seed for the random number generator

        int32_t minKeep = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

        int32_t topK = 40;       // <= 0 to use vocab size
        float topP = 0.95f;      // 1.0 = disabled
        float minP = 0.05f;      // 0.0 = disabled
        float tfsZ = 1.00f;      // 1.0 = disabled
        float typicalP = 1.00f;  // 1.0 = disabled
        float temp = 0.80f;      // <= 0.0 to sample greedily, 0.0 to not output probabilities
        float tempRange = 0.00f; // 0.0 = disabled
        float tempExp = 1.00f;   // controls how entropy maps to temperature in dynamic temperature sampler

        struct RepetitionPenalty {
            int32_t numTokens = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
            float   repeat    = 1.00f; // 1.0 = disabled
            float   freq      = 0.00f; // 0.0 = disabled
            float   present   = 0.00f; // 0.0 = disabled
        } repetitionPenalty;

        struct Mirostat {
            int32_t ver = 0; // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
            float tau = 5.00f; // target entropy
            float eta = 0.10f; // learning rate
        } mirostat;

        struct XTC {
            float probability = 0.00f; // 0.0 = disabled
            float threshold = 0.10f;   // > 0.5 disables XTC
        } xtc;

        std::vector<SamplingType> samplerSequence = {
            SamplingType::Top_K,
            SamplingType::Typical_P,
            SamplingType::Top_P,
            SamplingType::Min_P,
            SamplingType::Temperature
        };

        std::string grammar; // optional BNF-like grammar to constrain sampling

        itlib::flat_map<Token, float> logitBias; // bias for specific tokens
    };

    explicit Sampler(Model& model, const Params& params);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    // reset the sampler state
    void reset();

    // reset the performance counters
    void perfReset();

    // extended sampling implementation:
    //
    // - set logits
    // - apply the configured sampler chain
    // - check if the token fits the grammar (if any)
    // - if not: resample by first applying the grammar constraints and then sampling again (slower path)
    //
    // if grammarFirst is true, the grammar is applied before the samplers (slower)
    // useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
    //
    // idx is optional for sampling from the logits of the ith token
    Token sample(llama_context* lctx, int idx = -1, bool grammarFirst = false);

    TokenDataVector extractTokenData(llama_context* lctx);

    // accept token as sampled
    // if acceptGrammar is true, the token is accepted both by the sampling chain and the grammar
    void accept(Token id, bool acceptGrammar);

private:
    bstl::c_unique_ptr<llama_sampler> m_grammarSampler;
    bstl::c_unique_ptr<llama_sampler> m_samplerChain;

    // current tokens for sampling (one for each vocabulary entry)
    // kept as member so as to avoid reallocation on every sample call
    std::vector<llama_token_data> m_cur;
};

} // namespace bl::llama
