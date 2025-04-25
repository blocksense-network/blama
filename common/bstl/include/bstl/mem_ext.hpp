// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <memory>

namespace bstl {

template <typename T>
using c_unique_ptr = std::unique_ptr<T, void(*)(T*)>;

} // namespace bstl
