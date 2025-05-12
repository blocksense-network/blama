# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
bl_dep(boost)

add_executable(blama-server
    ServerMain.cpp
)

target_link_libraries(blama-server PRIVATE
    bl::bstl
    bl::llama
    jalog::jalog
    jalog::sinklib
    Boost::beast
    ac-test-data::llama
)