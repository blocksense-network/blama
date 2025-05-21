// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include <memory>

namespace bl::llama {

class Model;

namespace server {

class BL_LLAMA_SERVER_API Server {
public:
    Server(std::shared_ptr<Model> model);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace server
} // namespace bl::llama
