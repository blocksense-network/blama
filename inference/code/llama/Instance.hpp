// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Sampler.hpp"
#include "Session.hpp"
#include <bstl/mem_ext.hpp>
#include <optional>

struct llama_context;

namespace bl::llama {
class Model;
class Session;
class StringSession;
class LoraAdapter;
class ControlVector;

class BL_LLAMA_API Instance {
public:
    struct InitParams {
        uint32_t ctxSize = 0; // context size for the model (0 = maximum allowed by model)
        uint32_t batchSize = 2048; // logical batch size for prompt processing (may be silently truncated to ctxSize)
        uint32_t ubatchSize = 512; // physical batch size for prompt processing (0 = batchSize)
        bool flashAttn = false; // enable flash attention
        std::string grammar; // BNF-styled grammar
    };

    explicit Instance(Model& model, InitParams params);
    ~Instance();

    void addLora(LoraAdapter& lora, float scale = 1.f);

    void clearLoraState();

    // add control to the context
    void addControlVector(const ControlVector& ctrlVector);

    // do an empty model run to load model data in cache
    void warmup();

    // only one session per instance can be active at a time
    Session& startSession(const Session::InitParams params);
    void stopSession() noexcept;

    const Model& model() const noexcept { return m_model; }

    Sampler& sampler() noexcept { return *m_sampler; }

    // Change sampler settings by resetting it
    // warning: this will clear any previous sampler state
    void resetSampler(const Sampler::Params& params) {
        m_sampler.reset(new Sampler(m_model, params));
    }

private:
    Model& m_model;
    std::unique_ptr<Sampler> m_sampler;
    bstl::c_unique_ptr<llama_context> m_lctx;
    std::optional<Session> m_session;
};

} // namespace bl::llama
