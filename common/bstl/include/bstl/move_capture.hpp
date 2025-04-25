// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <splat/pp_select.h>
#include "move.hpp"

#define I_bstl_MOVE_CAPTURE_ONE(a, i) a = bstl::move(a)

#define movecap(...) SPLAT_ITERATE_WITH(I_bstl_MOVE_CAPTURE_ONE, ##__VA_ARGS__)
