// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <splat/symbol_export.h>

#if BL_LLAMA_SHARED
#   if BUILDING_BL_LLAMA
#       define BL_LLAMA_API SYMBOL_EXPORT
#   else
#       define BL_LLAMA_API SYMBOL_IMPORT
#   endif
#else
#   define BL_LLAMA_API
#endif
