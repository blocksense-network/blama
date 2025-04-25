# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#
bl_dep(splat)

add_library(bstl INTERFACE)
add_library(bl::bstl ALIAS bstl)
set_target_properties(bstl PROPERTIES
    EXPORT_NAME bl::bstl
)

target_link_libraries(bstl INTERFACE
    splat::splat
)

target_sources(bstl INTERFACE FILE_SET HEADERS FILES
    bstl/iile.h
    bstl/mem_ext.hpp
    bstl/move.hpp
    bstl/move_capture.hpp
    bstl/throw_stdex.hpp
    bstl/u8c.h
)
