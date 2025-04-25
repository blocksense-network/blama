// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "InstanceEmbedding.hpp"
#include "Model.hpp"
#include "LoraAdapter.hpp"
#include "Logging.hpp"
#include "Session.hpp"
#include "ControlVector.hpp"

#include <llama.h>

#include <bstl/throw_stdex.hpp>
#include <bstl/iile.h>
#include <bstl/move.hpp>

#include <cassert>
#include <span>
#include <fstream>

namespace bl::llama {

namespace {
llama_context_params llamaFromInstanceInitParams(const InstanceEmbedding::InitParams& params) {
    llama_context_params llamaParams = llama_context_default_params();
    llamaParams.n_ctx = params.ctxSize;
    llamaParams.n_batch = params.batchSize;
    llamaParams.n_ubatch = params.ubatchSize;
    llamaParams.flash_attn = params.flashAttn;
    llamaParams.embeddings = true;
    return llamaParams;
}
} // namespace

InstanceEmbedding::InstanceEmbedding(Model& model, InitParams params)
    : m_model(model)
    , m_sampler(model, {})
    , m_params(std::move(params))
    , m_lctx(llama_init_from_model(model.lmodel(), llamaFromInstanceInitParams(params)), llama_free)
{
    if (!m_lctx) {
        throw_ex{} << "Failed to create llama context";
    }
    assert(model.lmodel() == llama_get_model(m_lctx.get()));

    const auto ctxLen = llama_n_ctx(m_lctx.get());
    const auto ctxTrain = model.trainCtxLength();
    if (ctxLen > ctxTrain) {
        LLAMA_LOG(Warning, "Instance requested context length ", ctxLen, " is greater than the model's training context length ", ctxTrain);
    }

    if (llama_model_has_encoder(m_model.lmodel()) && llama_model_has_decoder(m_model.lmodel())) {
        LLAMA_LOG(Error, "Computing embeddings in encoder-decoder models is not supported");
    }
}

InstanceEmbedding::~InstanceEmbedding() = default;

namespace {
void normalizeEmbedding(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? float(1.0 / sum) : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

void batchAddSeq(llama_batch& batch, std::span<const Token> tokens, llama_seq_id seq_id) {
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token   [batch.n_tokens] = tokens[i];
        batch.pos     [batch.n_tokens] = llama_pos(i);
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = seq_id;
        batch.logits  [batch.n_tokens] = false;

        batch.n_tokens++;
    }

    // We want to extract the embeddings
    // for the last token in the sequence because
    // it will capture the all tokens in the sequence.
    batch.logits[batch.n_tokens - 1] = true;
}
}

std::vector<float> InstanceEmbedding::getEmbeddingVector(std::span<const Token> prompt, int32_t normalization) const {
    const enum llama_pooling_type pooling_type = llama_pooling_type(m_lctx.get());
    llama_context* ctx = m_lctx.get();
    llama_model* model = m_model.lmodel();
    int n_embd_count = 1; // TODO: support multiple prompts

        // allocate output
    const int n_embd = llama_model_n_embd(model);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float* embData = embeddings.data();

    llama_batch batch = llama_batch_init(m_params.batchSize, 0, 1);
    batchAddSeq(batch, prompt, 0);

    llama_kv_self_clear(ctx);

    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        if (llama_encode(ctx, batch) < 0) {
            LLAMA_LOG(Error, "Failed to encode!");
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        if (llama_decode(ctx, batch) < 0) {
            LLAMA_LOG(Error, "Failed to decode!");
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            assert(embd != NULL && "Failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            assert(embd != NULL && "Failed to get sequence embeddings");
        }

        float * outRes = embData + embd_pos * n_embd;
        normalizeEmbedding(embd, outRes, n_embd, normalization);
    }

    return embeddings;
}

uint32_t InstanceEmbedding::embeddingDim() const noexcept {
     return llama_model_n_embd(m_model.lmodel());
}

} // namespace bl::llama
