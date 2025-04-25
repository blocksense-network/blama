// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Sampler.hpp"
#include "Session.hpp"
#include <bstl/mem_ext.hpp>

#include <cmath>
#include <vector>
#include <optional>

struct llama_context;

namespace bl::llama {
class Model;
class Session;
class StringSession;
class ControlVector;

class BL_LLAMA_API InstanceEmbedding {
public:
    struct InitParams {
        uint32_t ctxSize = 0; // context size for the model (0 = maximum allowed by model)
        uint32_t batchSize = 2048; // logical batch size for prompt processing (may be silently truncated to ctxSize)
        uint32_t ubatchSize = 512; // physical batch size for prompt processing (0 = batchSize)
        bool flashAttn = false; // enable flash attention
    };

    explicit InstanceEmbedding(Model& model, InitParams params);
    ~InstanceEmbedding();

    // Get the embedding vector for the given prompt
    // the normalization parameter is used to normalize the embeddings
    // values are (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean[default], >2=p-norm)
    std::vector<float> getEmbeddingVector(std::span<const Token> prompt, int32_t normalization = 2) const;

    // Get the embedding dimension
    uint32_t embeddingDim() const noexcept;

    const Model& model() const noexcept { return m_model; }
    Sampler& sampler() noexcept { return m_sampler; }

private:
    Model& m_model;
    Sampler m_sampler;
    InitParams m_params;
    bstl::c_unique_ptr<llama_context> m_lctx;
    std::optional<Session> m_session;
};

} // namespace bl::llama
