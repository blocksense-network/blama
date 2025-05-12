// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//

// trivial example of using llama inference

// llama
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/LogitComparer.hpp>

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

    std::string prompt = "The first person to";
    std::cout << "Prompt: " << prompt << "\n";

    // start session
    auto& session = instance.startSession({});
    session.setInitialPrompt(model.vocab().tokenize(prompt, true, true));

    std::vector<bl::llama::TokenPrediction> iRes;

    constexpr int maxTokens = 20;
    for (int i = 0; i < maxTokens; ++i) {
        auto pred = session.getToken();
        if (pred.token == bl::llama::Token_Invalid) {
            // no more tokens
            break;
        }
        iRes.push_back(pred);
        std::cout << model.vocab().tokenToString(pred.token);
    }

    bl::llama::Model modelCpu(modelGguf, {
        .gpu = false
    }, modelLoadProgressCallback);

    bl::llama::Instance instanceCpu(modelCpu, {});
    auto& sessionCpu = instanceCpu.startSession({});
    sessionCpu.setInitialPrompt(modelCpu.vocab().tokenize(prompt, true, true));
    auto iRes2 = sessionCpu.fillCtx(iRes);

    std::vector<bl::llama::LogitComparer::ComparisonMetrics> metrics(iRes.size());
    float sumSim = 0;
    for (size_t i = 0; i < iRes.size(); i++) {
        float sim = bl::llama::LogitComparer::logitSimilarity(iRes[i].logits, iRes2[i].logits);
        metrics[i] = bl::llama::LogitComparer::compare(iRes[i].logits, iRes2[i].logits);
        std::cout   << "Token: '" << model.vocab().tokenToString(iRes[i].token) << "' - "
                    << " Logits: " << iRes[i].logits[0].logit << " Logits2: " << iRes2[i].logits[0].logit
                    << " Sim: " << sim
                    << "\n";
        sumSim += sim;
    }

    std::cout << "\n\nAverage similarity: " << sumSim / iRes.size() << "\n";
    std::cout << "Final metrics score: " << bl::llama::LogitComparer::comparisonFinalScore(metrics) << "\n";
    std::cout << '\n';

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
