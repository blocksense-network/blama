# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
bl_dep(boost)
bl_dep(nl-json)

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
        bl::llama
    PRIVATE
        Boost::asio
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
    nlohmann_json::nlohmann_json
    ac-test-data::llama
)
