// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "api.h"
#include <memory>

namespace bl::llama::server {

class BL_LLAMA_SERVER_API Server {
public:
    Server();
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace bl::llama::server