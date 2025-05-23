# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
if(NOT TARGET nlohmann_json::nlohmann_json)
    CPMAddPackage(gh:nlohmann/json@3.12.0)
endif()
