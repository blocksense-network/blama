// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "Sampler.hpp"
#include "Model.hpp"
#include <llama.h>
#include <bstl/move.hpp>
#include <bstl/iile.h>
#include <span>
#include <cmath>
#include <cstddef>

namespace bl::llama {

Sampler::Sampler(Model& model, const Params& params)
    : m_grammarSampler(llama_sampler_init_grammar(llama_model_get_vocab(model.lmodel()), params.grammar.c_str(), "root"), llama_sampler_free)
    , m_samplerChain(llama_sampler_chain_init({ .no_perf = false }), llama_sampler_free)
{
    auto lmodel = model.lmodel();
    auto chain = m_samplerChain.get();

    // static assertions to add logitBias
    auto& logitBiasBuf = params.logitBias.container();

    static_assert(sizeof(*logitBiasBuf.data()) == sizeof(llama_logit_bias));
    static_assert(sizeof(logitBiasBuf.data()->first) == sizeof(llama_token));
    static_assert(sizeof(logitBiasBuf.data()->second) == sizeof(float));
    static_assert(offsetof(llama_logit_bias, token) == 0);

    llama_sampler_chain_add(chain,
        llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(llama_model_get_vocab(lmodel)),
            int32_t(params.logitBias.size()),
            reinterpret_cast<const llama_logit_bias*>(logitBiasBuf.data())
        )
    );

    llama_sampler_chain_add(chain,
        llama_sampler_init_penalties(
            params.repetitionPenalty.numTokens,
            params.repetitionPenalty.repeat,
            params.repetitionPenalty.freq,
            params.repetitionPenalty.present
        )
    );

    const auto& miro = params.mirostat;

    if (miro.ver == 1) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(chain,
            llama_sampler_init_mirostat(
                llama_vocab_n_tokens(llama_model_get_vocab(lmodel)),
                params.rngSeed,
                miro.tau, miro.eta, 100
            )
        );
    }
    else if (miro.ver == 2) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(chain,
            llama_sampler_init_mirostat_v2(
                params.rngSeed,
                miro.tau, miro.eta
            )
        );
    }
    else if (miro.ver > 2) {
        throw std::runtime_error("Unsupported mirostat version");
    }
    else {
        // sequence sampling
        const size_t minKeep = params.minKeep;
        for (auto type : params.samplerSequence) {
            auto sampler = iile([&]() -> llama_sampler* {
                switch (type)
                {
                case SamplingType::Top_K: return llama_sampler_init_top_k(params.topK);
                case SamplingType::Typical_P: return llama_sampler_init_typical(params.typicalP, minKeep);
                case SamplingType::Top_P: return llama_sampler_init_top_p(params.topP, minKeep);
                case SamplingType::Min_P: return llama_sampler_init_min_p(params.minP, minKeep);
                case SamplingType::Temperature:
                    return llama_sampler_init_temp_ext(params.temp, params.tempRange, params.tempExp);
                case SamplingType::XTC:
                    return llama_sampler_init_xtc(params.xtc.probability, params.xtc.threshold, minKeep, params.rngSeed);
                case SamplingType::Infill:
                    return llama_sampler_init_infill(llama_model_get_vocab(lmodel));
                default:
                    throw std::runtime_error("Unsupported sampler type");
                }
            });
            llama_sampler_chain_add(chain, sampler);
        }

        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.rngSeed));
    }
}

Sampler::~Sampler() = default;

void Sampler::accept(Token id, bool acceptGrammar) {
    if (acceptGrammar) {
        llama_sampler_accept(m_grammarSampler.get(), id);
    }

    llama_sampler_accept(m_samplerChain.get(), id);
}

namespace {
llama_token_data_array fillLogits(std::vector<llama_token_data>& out, llama_context* lctx, int idx) {
    const auto* logits = llama_get_logits_ith(lctx, idx);

    const auto* lmodel = llama_get_model(lctx);
    const int vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(lmodel));

    out.resize(vocabSize);

    for (llama_token id = 0; id < vocabSize; id++) {
        out[id] = {id, logits[id], 0.0f};
    }

    return {out.data(), out.size(), -1, false};
}
}

Token Sampler::sample(llama_context* lctx, int idx, bool grammarFirst) {
    auto grammar = m_grammarSampler.get();
    auto chain = m_samplerChain.get();

    auto cur = fillLogits(m_cur, lctx, idx);

    if (grammarFirst) {
        llama_sampler_apply(grammar, &cur);
    }

    llama_sampler_apply(chain, &cur);

    if (cur.selected == -1) {
        throw std::runtime_error("no selected token during sampling - check your sampling configuration");
    }

    const llama_token id = cur.data[cur.selected].id;

    if (grammarFirst) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       singleTokenData = {id, 1.0f, 0.0f};
        llama_token_data_array singleTokenDataAr = {&singleTokenData, 1, -1, false};

        llama_sampler_apply(grammar, &singleTokenDataAr);

        const bool is_valid = singleTokenDataAr.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    cur = fillLogits(m_cur, lctx, idx);

    llama_sampler_apply(grammar, &cur);
    llama_sampler_apply(chain, &cur);

    if (cur.selected == -1) {
        throw std::runtime_error("no selected token during re-sampling - check your sampling configuration");
    }

    return cur.data[cur.selected].id;
}

TokenDataVector Sampler::extractTokenData(llama_context* lctx) {
   auto chain = m_samplerChain.get();

    auto cur = fillLogits(m_cur, lctx, -1);

    llama_sampler_apply(chain, &cur);

    TokenDataVector result(cur.size);

    for (size_t i = 0; i < cur.size; i++)
    {
        result[i] = {cur.data[i].id, cur.data[i].logit, cur.data[i].p};
    }

    return result;
}

void Sampler::reset() {
    llama_sampler_reset(m_grammarSampler.get());
    llama_sampler_reset(m_samplerChain.get());
}

void Sampler::perfReset() {
    // perf on grammar samplers is not supported upstream
    //llama_perf_sampler_reset(m_grammarSampler.get());
    llama_perf_sampler_reset(m_samplerChain.get());
}

} // namespace bl::llama
