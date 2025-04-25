// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <jalog/Scope.hpp>
#include <jalog/Log.hpp>

namespace bl::llama::log {
extern jalog::Scope scope;
}

#define LLAMA_LOG(lvl, ...) JALOG_SCOPE(::bl::llama::log::scope, lvl, __VA_ARGS__)
