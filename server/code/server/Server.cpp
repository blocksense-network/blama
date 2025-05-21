// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "Server.hpp"

namespace bl::llama::server {

struct Server::Impl {
    std::shared_ptr<Model> m_model;
    Impl(std::shared_ptr<Model> model)
        : m_model(std::move(model))
    {}
};

Server::Server(std::shared_ptr<Model> model)
    : m_impl(std::make_unique<Impl>(std::move(model)))
{}

Server::~Server() = default;

} // namespace bl::llama::server