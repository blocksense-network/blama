// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "LoraAdapter.hpp"
#include "Model.hpp"

#include <llama.h>
#include <stdexcept>

namespace bl::llama {

LoraAdapter::LoraAdapter(Model& model, std::string path)
    : m_model(model)
    , m_adapter(llama_adapter_lora_init(model.lmodel(), path.c_str()), llama_adapter_lora_free)
{
    if (!m_adapter) {
        throw std::runtime_error("Failed to create lora adapter");
    }
}

} // namespace bl::llama
