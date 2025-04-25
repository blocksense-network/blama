# SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
# SPDX-License-Identifier: MIT
#

#######################################
# setup deps

bl_dep(itlib)
bl_dep(splat)
bl_dep(jalog)
bl_dep(minja)
bl_dep(nl-json)

#######################################
# setup llama.cpp

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    set(haveCuda YES)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    message(STATUS "${CMAKE_PROJECT_NAME}: system CUDA found")
    # we should add -forward-unknown-to-host-compiler but it somehow works without it
    # could it be that it depends on the CMake version?
endif()

if(haveCuda)
    set(GGML_CUDA ON)
endif()

set(GGML_CCACHE OFF)

set(LLAMA_BUILD_COMMON OFF)

CPMAddPackage("gh:ggml-org/llama.cpp#b5187")

icm_add_lib(bl-llama BL_LLAMA)
add_library(bl::llama ALIAS bl-llama)

#######################################
# setup inference library

target_link_libraries(bl-llama
    PUBLIC
        bl::bstl
        splat::splat
        itlib::itlib
    PRIVATE
        llama # llama.cpp
        jalog::jalog
        nlohmann_json::nlohmann_json
        minja
)

target_sources(bl-llama
    INTERFACE FILE_SET HEADERS FILES
        llama/api.h
        llama/Init.hpp
        llama/Model.hpp
        llama/ChatFormat.hpp
        llama/Vocab.hpp
        llama/Sampler.hpp
        llama/Instance.hpp
        llama/InstanceEmbedding.hpp
        llama/Session.hpp
        llama/AntipromptManager.hpp
        llama/IncrementalStringFinder.hpp
        llama/ControlVector.hpp
        llama/LoraAdapter.hpp
        llama/LogitComparer.hpp
        llama/ResourceCache.hpp
    PRIVATE
        llama/Logging.hpp
        llama/Logging.cpp
        llama/Init.cpp
        llama/Model.cpp
        llama/ChatFormat.cpp
        llama/Vocab.cpp
        llama/Sampler.cpp
        llama/Instance.cpp
        llama/InstanceEmbedding.cpp
        llama/Session.cpp
        llama/AntipromptManager.cpp
        llama/IncrementalStringFinder.cpp
        llama/ControlVector.cpp
        llama/LoraAdapter.cpp
        llama/LogitComparer.cpp
)
