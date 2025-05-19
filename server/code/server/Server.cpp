// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "Server.hpp"

namespace bl::llama::server {

struct Server::Impl {
  Impl() = default;
  ~Impl() = default;
};

Server::Server()
    : m_impl(std::make_unique<Impl>())
{}

Server::~Server() = default;

} // namespace bl::llama::server