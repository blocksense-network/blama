// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include "Vocab.hpp"

#include <bstl/mem_ext.hpp>
#include <itlib/ufunction.hpp>

#include <algorithm>
#include <string>
#include <span>
#include <vector>

struct llama_model;

namespace bl::llama {
class Job;
class LoraAdapter;

using ModelLoadProgressCb = itlib::ufunction<void(float)>;

struct LlamaModelResource;

class BL_LLAMA_API Model {
public:
    struct Params {
        bool gpu = true; // try to load data on gpu
        bool vocabOnly = false; // do not load model, only vocab
        bool prefixInputsWithBos = false; // add bos token to interactive inputs (#13)

        bool operator==(const Params& other) const noexcept = default;
    };

    Model(const std::string& gguf, Params params, ModelLoadProgressCb pcb = {});
    ~Model();

    const Params& params() const noexcept { return m_params; }

    uint32_t trainCtxLength() const noexcept;
    bool shouldAddBosToken() const noexcept;
    bool hasEncoder() const noexcept;
    bool prefixInputsWithBos() const noexcept { return m_params.prefixInputsWithBos; }

    // fallback to "chatml" if the underlying model does not provide a chat template
    std::string getChatTemplateId() const;

    llama_model* lmodel() noexcept { return m_lmodel.get(); }
    const llama_model* lmodel() const noexcept { return m_lmodel.get(); }

    const Vocab& vocab() const noexcept { return m_vocab; }
private:
    const Params m_params;
    bstl::c_unique_ptr<llama_model> m_lmodel;

    Vocab m_vocab{*this};
};

} // namespace bl::llama
