// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <server/Server.hpp>
#include <llama/Init.hpp>
#include <llama/Model.hpp>

// logging
#include <jalog/Instance.hpp>
#include <jalog/sinks/DefaultSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>

int main() {
    jalog::Instance jl;
    jl.setup().async().add<jalog::sinks::DefaultSink>();

    bl::llama::initLibrary();

    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";

    auto modelLoadProgressCallback = [](float progress) {
        const int barWidth = 50;
        static float currProgress = 0;
        auto delta = int(progress * barWidth) - int(currProgress * barWidth);
        for (int i = 0; i < delta; i++) {
            std::cout.put('=');
        }
        currProgress = progress;
        if (progress == 1.f) {
            std::cout << '\n';
        }
        return true;
    };

    auto model = std::make_shared<bl::llama::Model>(modelGguf, bl::llama::Model::Params{}, modelLoadProgressCallback);

    bl::llama::server::Server srv(model);

    bl::llama::server::Server::CompleteRequestParams req = {
        .prompt = "The first man to",
        .maxTokens = 10,
    };

    srv.completeText(req, [](std::vector<bl::llama::server::Server::TokenData> gen) {
        for (auto& g : gen) {
            std::cout << g.tokenStr;
        }
    });

    std::cout << "\n";

    return 0;
}