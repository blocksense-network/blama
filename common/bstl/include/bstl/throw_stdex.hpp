// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <itlib/throw_ex.hpp>
#include <stdexcept>

namespace bl {
using throw_ex = itlib::throw_ex<std::runtime_error>;
}
