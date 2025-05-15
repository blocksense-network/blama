// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//

// trivial example of using llama inference

// llama
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/ControlVector.hpp>

// logging
#include <jalog/Instance.hpp>
#include <jalog/sinks/DefaultSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>

int main() try {
    jalog::Instance jl;
    jl.setup().add<jalog::sinks::DefaultSink>();

    // initialize the library
    bl::llama::initLibrary();

    // load model
    std::string modelGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    std::string ctrlVectorGguf = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6-control_vector.gguf";
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

    bl::llama::Model model(modelGguf, {}, modelLoadProgressCallback);

    // create inference instance
    bl::llama::Instance instance(model, {});

    // To add control vector uncomment the following lines
    // bl::llama::ControlVector ctrlVector(model, {{ctrlVectorGguf, 2.f}});
    // instance.addControlVector(ctrlVector);

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";

    // start session
    auto& session = instance.startSession({});
    session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));

    auto stream = session.completeStream({
        .maxTokens = 100
    });

    while (auto p = stream.complete()) {
        std::cout << model.vocab().tokenToString(p.token);
    }
    std::cout << '\n';

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
