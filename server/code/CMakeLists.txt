# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
bl_dep(boost)

######################

icm_add_lib(bl-llama-server BL_LLAMA_SERVER)
add_library(bl::llama-server ALIAS bl-llama-server)
target_sources(bl-llama-server
    INTERFACE FILE_SET HEADERS FILES
        server/api.h
        server/Server.hpp
    PRIVATE
        server/Server.cpp
)

target_link_libraries(bl-llama-server
    PUBLIC
        bl::bstl
    PRIVATE
        bl::llama
        jalog::jalog
)

######################

add_executable(blama-http-server
    http/HttpServerMain.cpp
)

target_link_libraries(blama-http-server PRIVATE
    bl::bstl
    bl::llama # temp
    bl::llama-server
    jalog::jalog
    jalog::sinklib
    Boost::beast
    ac-test-data::llama
)
