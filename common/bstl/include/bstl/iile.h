// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once

// trivial macro to more visibly show that a lambda is immediately invoked
// use like
// auto x = iile([&] { return y; });
#define iile(f) f()
