// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"

#include <bstl/mem_ext.hpp>
#include <string>

struct llama_adapter_lora;

namespace bl::llama {
class Model;
struct LLamaLoraResource;

class BL_LLAMA_API LoraAdapter {
public:
    LoraAdapter(Model& model, std::string path);

    llama_adapter_lora* ladapter() const noexcept { return m_adapter.get(); }

    const Model& model() const noexcept { return m_model; }

private:
    Model& m_model;
    bstl::c_unique_ptr<llama_adapter_lora> m_adapter;
};

} // namespace bl::llama
